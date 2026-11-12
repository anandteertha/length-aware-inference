"""Train supervised and distilled three-layer students on pinned SST-2 training rows.

The GLUE validation split is deliberately absent from this script. Checkpoints and
row-level training data go under an ignored artifact directory. Each run writes a
manifest and training curves so a separate final evaluation can be audited.
"""

import argparse
import copy
import gc
import json
import statistics
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from datasets import load_dataset

from etdl.config import load_config
from etdl.data import DATASET_ID, load_tokenizer, model_source
from etdl.io import read_json, sha256, write_json
from etdl.model import load_pytorch


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/student-study"))
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--dev-size", type=int, default=1024)
    parser.add_argument("--gate-size", type=int, default=2048)
    parser.add_argument("--calibration-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def batches(indices, batch_size):
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def model_batch(tokenizer, encoded, positions):
    features = [
        {"input_ids": encoded[i]["input_ids"], "attention_mask": encoded[i]["attention_mask"]}
        for i in positions
    ]
    return tokenizer.pad(features, padding=True, return_tensors="pt")


def student_from_weights(teacher, initial_weights):
    student_config = copy.deepcopy(teacher.config)
    student_config.n_layers = 3
    student = type(teacher)(student_config).cpu().float()
    student.load_state_dict(initial_weights, strict=True)
    return student


def evaluate(model, tokenizer, encoded, labels, positions, batch_size):
    model.eval()
    correct = 0
    with torch.inference_mode():
        for subset in batches(positions, batch_size):
            batch = model_batch(tokenizer, encoded, subset)
            predictions = model(**batch).logits.argmax(dim=-1)
            correct += int((predictions == labels[subset]).sum())
    return correct / len(positions)


def main():
    args = arguments()
    if min(
        args.train_size,
        args.dev_size,
        args.gate_size,
        args.calibration_size,
        args.epochs,
        args.batch_size,
        args.threads,
    ) <= 0:
        raise ValueError("Sizes, epochs, batch size, and threads must be positive")
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists; use a fresh directory: {output}")
    output.mkdir(parents=True)
    start_time = time.perf_counter()
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.toml")
    calibration = read_json(config.artifact_dir / "data/calibration.json")
    revision = calibration["revision"]
    torch.set_num_threads(args.threads)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    tokenizer = load_tokenizer(config)
    dataset = load_dataset(DATASET_ID, "sst2", split="train", revision=revision)
    total = args.train_size + args.dev_size + args.gate_size + args.calibration_size
    if total > len(dataset):
        raise ValueError("Requested partitions exceed the pinned train split")
    permutation = np.random.default_rng(config.seed).permutation(len(dataset))[:total]
    boundaries = np.cumsum(
        [0, args.train_size, args.dev_size, args.gate_size, args.calibration_size]
    )
    split_names = ("student_fit", "student_dev", "gate_fit", "gate_calibration")
    partitions = {
        name: list(range(int(boundaries[i]), int(boundaries[i + 1])))
        for i, name in enumerate(split_names)
    }
    rows = dataset.select(permutation.tolist())
    row_ids = [int(row["idx"]) for row in rows]
    if len(set(row_ids)) != total:
        raise ValueError("SST-2 row IDs are not unique")
    encoded_all = tokenizer([row["sentence"] for row in rows], truncation=False)
    encoded = [
        {"input_ids": ids, "attention_mask": mask}
        for ids, mask in zip(
            encoded_all["input_ids"], encoded_all["attention_mask"], strict=True
        )
    ]
    token_lengths = [len(item["input_ids"]) for item in encoded]
    if max(token_lengths) > config.max_length:
        raise ValueError("A selected input exceeds max_length; no truncation is allowed")
    labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_config": "sst2",
        "dataset_revision": revision,
        "train_fingerprint": dataset._fingerprint,
        "model_source": model_source(config),
        "seed": config.seed,
        "student_layers": 3,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "threads": args.threads,
        "optimizer": "AdamW",
        "learning_rate": 2e-5,
        "temperature": 2.0,
        "distillation_alpha": 0.5,
        "partitions": {name: [row_ids[i] for i in indices] for name, indices in partitions.items()},
        "token_length_min": min(token_lengths),
        "token_length_max": max(token_lengths),
    }
    write_json(output / "manifest.json", manifest)
    print(
        json.dumps({"event": "partitioned", "counts": {k: len(v) for k, v in partitions.items()}}),
        flush=True,
    )

    teacher = load_pytorch(config, threads=args.threads)
    initial_weights = {
        name: value.detach().clone()
        for name, value in teacher.state_dict().items()
        if not any(name.startswith(f"distilbert.transformer.layer.{i}.") for i in range(3, 6))
    }
    teacher_logits = torch.empty((total, 2), dtype=torch.float32)
    with torch.inference_mode():
        for subset in batches(list(range(total)), args.batch_size):
            batch = model_batch(tokenizer, encoded, subset)
            teacher_logits[subset] = teacher(**batch).logits
    teacher_accuracy = {
        name: float((teacher_logits[indices].argmax(-1) == labels[indices]).float().mean())
        for name, indices in partitions.items()
    }
    del teacher
    gc.collect()
    torch.save(teacher_logits, output / "teacher_logits.pt")
    print(json.dumps({"event": "teacher_cached", "accuracy": teacher_accuracy}), flush=True)

    summaries = {}
    for mode in ("supervised", "distilled"):
        torch.manual_seed(config.seed)
        model = student_from_weights(load_pytorch(config, threads=args.threads), initial_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        generator = torch.Generator().manual_seed(config.seed)
        history = []
        best_accuracy = -1.0
        best_epoch = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(len(partitions["student_fit"]), generator=generator).tolist()
            shuffled = [partitions["student_fit"][i] for i in order]
            losses = []
            epoch_start = time.perf_counter()
            for subset in batches(shuffled, args.batch_size):
                batch = model_batch(tokenizer, encoded, subset)
                optimizer.zero_grad(set_to_none=True)
                student_logits = model(**batch).logits
                ce = F.cross_entropy(student_logits, labels[subset])
                if mode == "distilled":
                    kd = F.kl_div(
                        F.log_softmax(student_logits / 2.0, dim=-1),
                        F.softmax(teacher_logits[subset] / 2.0, dim=-1),
                        reduction="batchmean",
                    ) * 4.0
                    loss = 0.5 * ce + 0.5 * kd
                else:
                    loss = ce
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            dev_accuracy = evaluate(
                model, tokenizer, encoded, labels, partitions["student_dev"], args.batch_size
            )
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
                torch.save(model.state_dict(), output / f"{mode}.pt")
            print(json.dumps({"event": "epoch", "mode": mode, **record}), flush=True)
        summaries[mode] = {
            "best_student_dev_accuracy": best_accuracy,
            "best_epoch": best_epoch,
            "history": history,
            "checkpoint_sha256": sha256(output / f"{mode}.pt"),
            "checkpoint_bytes": (output / f"{mode}.pt").stat().st_size,
        }
        del model, optimizer
        gc.collect()

    write_json(
        output / "summary.json",
        {
            "purpose": "student training; train-derived development only, not final classifier evidence",
            "manifest_sha256": sha256(output / "manifest.json"),
            "teacher_accuracy_on_train_partitions": teacher_accuracy,
            "student_arms": summaries,
            "total_seconds": round(time.perf_counter() - start_time, 1),
            "observed_rss_mib": round(psutil.Process().memory_info().rss / 1024**2, 1),
        },
    )
    print(json.dumps({"event": "complete", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
