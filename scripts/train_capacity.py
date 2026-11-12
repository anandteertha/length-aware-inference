"""Train the predeclared four-layer capacity control if development accuracy warrants it.

Run after train_students.py finishes, and before evaluate_students.py is run on
GLUE validation. Reuses the exact partitions and cached teacher logits.
"""

import argparse
import copy
import gc
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from train_students import batches, evaluate, model_batch

from etdl.config import load_config
from etdl.data import DATASET_ID, load_tokenizer
from etdl.io import read_json, sha256, write_json
from etdl.model import load_pytorch


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("artifacts/student-study"))
    return parser.parse_args()


def main():
    study = arguments().study.resolve()
    manifest = read_json(study / "manifest.json")
    summary = read_json(study / "summary.json")
    if sha256(study / "manifest.json") != summary["manifest_sha256"]:
        raise ValueError("Training manifest hash changed")
    if (study / "capacity-summary.json").exists():
        raise ValueError("Capacity result already exists; preserve the first run")
    teacher_dev = summary["teacher_accuracy_on_train_partitions"]["student_dev"]
    best_three = max(
        arm["best_student_dev_accuracy"] for arm in summary["student_arms"].values()
    )
    gap = teacher_dev - best_three
    if gap <= 0.03:
        raise ValueError(f"Predeclared >3-point development gap not met: {gap:.4f}")
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.toml")
    torch.set_num_threads(manifest["threads"])
    torch.manual_seed(manifest["seed"])
    tokenizer = load_tokenizer(config)
    dataset = load_dataset(
        DATASET_ID, "sst2", split="train", revision=manifest["dataset_revision"]
    )
    split_names = ("student_fit", "student_dev", "gate_fit", "gate_calibration")
    row_ids = [idx for name in split_names for idx in manifest["partitions"][name]]
    needed = set(row_ids)
    lookup = {int(row["idx"]): row for row in dataset if int(row["idx"]) in needed}
    if len(lookup) != len(row_ids):
        raise ValueError("Could not reconstruct all pinned training rows")
    rows = [lookup[idx] for idx in row_ids]
    encoded_all = tokenizer([row["sentence"] for row in rows], truncation=False)
    encoded = [
        {"input_ids": ids, "attention_mask": mask}
        for ids, mask in zip(
            encoded_all["input_ids"], encoded_all["attention_mask"], strict=True
        )
    ]
    if max(len(item["input_ids"]) for item in encoded) > config.max_length:
        raise ValueError("Selected input exceeds max_length; no truncation is allowed")
    labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
    fit = list(range(len(manifest["partitions"]["student_fit"])))
    dev = list(range(len(fit), len(fit) + len(manifest["partitions"]["student_dev"])))
    teacher_logits = torch.load(study / "teacher_logits.pt", map_location="cpu", weights_only=True)
    if teacher_logits.shape != (len(rows), 2):
        raise ValueError("Cached teacher-logit shape differs from split manifest")

    teacher = load_pytorch(config, threads=manifest["threads"])
    student_config = copy.deepcopy(teacher.config)
    student_config.n_layers = 4
    student = type(teacher)(student_config).cpu().float()
    initial_weights = {
        name: value
        for name, value in teacher.state_dict().items()
        if not any(name.startswith(f"distilbert.transformer.layer.{i}.") for i in (4, 5))
    }
    student.load_state_dict(initial_weights, strict=True)
    del teacher, initial_weights
    gc.collect()
    optimizer = torch.optim.AdamW(student.parameters(), lr=manifest["learning_rate"])
    generator = torch.Generator().manual_seed(manifest["seed"])
    history = []
    best_accuracy = -1.0
    best_epoch = 0
    start = time.perf_counter()
    for epoch in range(1, manifest["epochs"] + 1):
        student.train()
        order = torch.randperm(len(fit), generator=generator).tolist()
        shuffled = [fit[i] for i in order]
        losses = []
        epoch_start = time.perf_counter()
        for subset in batches(shuffled, manifest["batch_size"]):
            batch = model_batch(tokenizer, encoded, subset)
            optimizer.zero_grad(set_to_none=True)
            logits = student(**batch).logits
            ce = F.cross_entropy(logits, labels[subset])
            temperature = manifest["temperature"]
            kd = F.kl_div(
                F.log_softmax(logits / temperature, dim=-1),
                F.softmax(teacher_logits[subset] / temperature, dim=-1),
                reduction="batchmean",
            ) * temperature**2
            alpha = manifest["distillation_alpha"]
            loss = (1 - alpha) * ce + alpha * kd
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        dev_accuracy = evaluate(student, tokenizer, encoded, labels, dev, manifest["batch_size"])
        record = {
            "epoch": epoch,
            "mean_train_loss": statistics.mean(losses),
            "student_dev_accuracy": dev_accuracy,
            "epoch_seconds": round(time.perf_counter() - epoch_start, 1),
        }
        history.append(record)
        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            best_epoch = epoch
            torch.save(student.state_dict(), study / "four_layer_distilled.pt")
        print(json.dumps({"event": "capacity_epoch", **record}), flush=True)
    report = {
        "reason": "best three-layer development accuracy >3 percentage points below teacher",
        "teacher_dev_accuracy": teacher_dev,
        "best_three_layer_dev_accuracy": best_three,
        "trigger_gap": gap,
        "student_layers": 4,
        "student_parameters": sum(p.numel() for p in student.parameters()),
        "best_dev_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "history": history,
        "seconds": round(time.perf_counter() - start, 1),
        "checkpoint_sha256": sha256(study / "four_layer_distilled.pt"),
        "checkpoint_bytes": (study / "four_layer_distilled.pt").stat().st_size,
    }
    write_json(study / "capacity-summary.json", report)
    print(json.dumps({"event": "capacity_complete", "report": report}), flush=True)


if __name__ == "__main__":
    main()
