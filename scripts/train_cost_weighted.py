"""Train the preregistered four-layer CPU-cost-weighted distilled student."""

import argparse
import copy
import gc
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from fit_gate import length_bin
from train_students import batches, evaluate, model_batch

from etdl.config import load_config
from etdl.data import DATASET_ID, load_tokenizer
from etdl.io import read_json, sha256, write_json
from etdl.model import load_pytorch


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("artifacts/student-study"))
    parser.add_argument("--gate", type=Path, default=Path("artifacts/student-study/gate-fit"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/student-study/cost-weighted")
    )
    return parser.parse_args()


def main():
    args = arguments()
    study = args.study.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Extension output already exists: {output}")
    output.mkdir(parents=True)
    manifest_path = study / "manifest.json"
    manifest = read_json(manifest_path)
    original_summary = read_json(study / "summary.json")
    if sha256(manifest_path) != original_summary["manifest_sha256"]:
        raise ValueError("Original training manifest changed")
    capacity = read_json(study / "capacity-summary.json")
    gate_path = args.gate.resolve() / "gate-plan.json"
    gate = read_json(gate_path)
    if gate["student_checkpoint_sha256"] != capacity["checkpoint_sha256"]:
        raise ValueError("Gate cost source belongs to another uniform student")
    raw_costs = gate["teacher_cost_raw_median_ms"]
    normalizer = statistics.median(raw_costs.values())
    bin_weights = {
        name: float(np.clip(value / normalizer, 0.5, 2.0)) for name, value in raw_costs.items()
    }
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.toml")
    torch.set_num_threads(manifest["threads"])
    torch.manual_seed(manifest["seed"])
    tokenizer = load_tokenizer(config)
    dataset = load_dataset(DATASET_ID, "sst2", split="train", revision=manifest["dataset_revision"])
    fit_ids = manifest["partitions"]["student_fit"]
    dev_ids = manifest["partitions"]["student_dev"]
    row_ids = fit_ids + dev_ids
    needed = set(row_ids)
    lookup = {int(row["idx"]): row for row in dataset if int(row["idx"]) in needed}
    if len(lookup) != len(row_ids):
        raise ValueError("Could not reconstruct all pinned fitting/development rows")
    rows = [lookup[idx] for idx in row_ids]
    tokenized = tokenizer([row["sentence"] for row in rows], truncation=False)
    encoded = [
        {"input_ids": ids, "attention_mask": mask}
        for ids, mask in zip(tokenized["input_ids"], tokenized["attention_mask"], strict=True)
    ]
    tokens = [len(item["input_ids"]) for item in encoded]
    if max(tokens) > config.max_length:
        raise ValueError("Training input exceeds max_length; no truncation allowed")
    labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
    weights = torch.tensor([bin_weights[length_bin(n)] for n in tokens], dtype=torch.float32)
    teacher_logits = torch.load(study / "teacher_logits.pt", weights_only=True)[: len(rows)]
    if teacher_logits.shape != (len(rows), 2):
        raise ValueError("Teacher cache shape differs from selected rows")

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
    fit = list(range(len(fit_ids)))
    dev = list(range(len(fit), len(rows)))
    best_accuracy = -1.0
    best_epoch = 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, manifest["epochs"] + 1):
        student.train()
        shuffled = torch.randperm(len(fit), generator=generator).tolist()
        losses = []
        epoch_start = time.perf_counter()
        for subset in batches(shuffled, manifest["batch_size"]):
            batch = model_batch(tokenizer, encoded, subset)
            optimizer.zero_grad(set_to_none=True)
            logits = student(**batch).logits
            ce_each = F.cross_entropy(logits, labels[subset], reduction="none")
            temperature = manifest["temperature"]
            kd_each = (
                F.kl_div(
                    F.log_softmax(logits / temperature, dim=-1),
                    F.softmax(teacher_logits[subset] / temperature, dim=-1),
                    reduction="none",
                ).sum(dim=-1)
                * temperature**2
            )
            alpha = manifest["distillation_alpha"]
            loss = ((1 - alpha) * ce_each + alpha * weights[subset] * kd_each).mean()
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
            torch.save(student.state_dict(), output / "four_layer_cost_weighted.pt")
        print(json.dumps({"event": "weighted_epoch", **record}), flush=True)
    report = {
        "status": "preregistered extension; original SST-2 validation already inspected",
        "source_manifest_sha256": sha256(manifest_path),
        "source_gate_plan_sha256": sha256(gate_path),
        "uniform_checkpoint_sha256": capacity["checkpoint_sha256"],
        "teacher_cost_raw_median_ms": raw_costs,
        "cost_normalizer_ms": normalizer,
        "cost_bin_weights": bin_weights,
        "fit_bin_counts": dict(Counter(length_bin(n) for n in tokens[: len(fit)])),
        "student_layers": 4,
        "student_parameters": sum(p.numel() for p in student.parameters()),
        "seed": manifest["seed"],
        "epochs": manifest["epochs"],
        "batch_size": manifest["batch_size"],
        "learning_rate": manifest["learning_rate"],
        "best_dev_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "history": history,
        "seconds": round(time.perf_counter() - start, 1),
        "checkpoint_sha256": sha256(output / "four_layer_cost_weighted.pt"),
        "checkpoint_bytes": (output / "four_layer_cost_weighted.pt").stat().st_size,
    }
    write_json(output / "summary.json", report)
    print(json.dumps({"event": "weighted_complete", "report": report}), flush=True)


if __name__ == "__main__":
    main()
