"""Compare uniform and cost-weighted four-layer students on a pinned IMDb sample.

The weighted checkpoint is selected only by SST-2 development accuracy. The IMDb
sample is drawn by a fixed label-stratified, no-truncation rule before prediction.
The previously inspected SST-2 validation comparison is reported as exploratory.
"""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from etdl.config import load_config
from etdl.data import load_tokenizer
from etdl.io import read_json, sha256, write_json
from etdl.model import load_pytorch

IMDB_ID = "stanfordnlp/imdb"
IMDB_REVISION = "e6281661ce1c48d982bc483cf8a173c1bbeb5d31"
LENGTH_BINS = ((3, 16), (17, 32), (33, 64), (65, 128), (129, 256), (257, 512))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("artifacts/student-study"))
    parser.add_argument(
        "--weighted", type=Path, default=Path("artifacts/student-study/cost-weighted")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/student-study/cost-weighted-eval")
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def student_from_checkpoint(teacher, checkpoint):
    student_config = copy.deepcopy(teacher.config)
    student_config.n_layers = 4
    student = type(teacher)(student_config).cpu().float().eval()
    student.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return student


def predict(model, tokenizer, encoded, batch_size):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            batch = tokenizer.pad(
                encoded[start : start + batch_size], padding=True, return_tensors="pt"
            )
            result.extend(model(**batch).logits.tolist())
    return np.asarray(result)


def paired_interval(first, second):
    difference = first.astype(float) - second.astype(float)
    rng = np.random.default_rng(2027)
    estimates = []
    for _ in range(10000):
        indices = rng.integers(0, len(difference), size=len(difference))
        estimates.append(float(np.mean(difference[indices])))
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def scores(labels, logits):
    prediction = logits.argmax(axis=1)
    correct = prediction == labels
    return {"correct": int(np.sum(correct)), "accuracy": float(np.mean(correct))}, correct


def comparison(labels, teacher_logits, uniform_logits, weighted_logits, lengths):
    metrics = {}
    correctness = {}
    for name, logits in (
        ("teacher", teacher_logits),
        ("uniform_four_layer", uniform_logits),
        ("cost_weighted_four_layer", weighted_logits),
    ):
        metrics[name], correctness[name] = scores(labels, logits)
    delta = correctness["cost_weighted_four_layer"].astype(float) - correctness[
        "uniform_four_layer"
    ].astype(float)
    metrics["weighted_minus_uniform"] = {
        "paired_accuracy_difference": float(np.mean(delta)),
        "paired_bootstrap_95pct_ci": paired_interval(
            correctness["cost_weighted_four_layer"], correctness["uniform_four_layer"]
        ),
        "weighted_correct_uniform_wrong": int(np.sum(delta == 1)),
        "uniform_correct_weighted_wrong": int(np.sum(delta == -1)),
    }
    strata = {}
    for low, high in LENGTH_BINS:
        selected = np.asarray([low <= n <= high for n in lengths])
        count = int(np.sum(selected))
        if count == 0:
            continue
        strata[f"{low}-{high}"] = {
            "examples": count,
            "interpret_rates": count >= 30,
            "teacher_correct": int(np.sum(correctness["teacher"] & selected)),
            "uniform_correct": int(np.sum(correctness["uniform_four_layer"] & selected)),
            "weighted_correct": int(np.sum(correctness["cost_weighted_four_layer"] & selected)),
        }
    return {"metrics": metrics, "length_strata": strata}


def main():
    args = arguments()
    if min(args.threads, args.batch_size) <= 0:
        raise ValueError("Threads and batch size must be positive")
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"External evaluation output already exists: {output}")
    output.mkdir(parents=True)
    study = args.study.resolve()
    weighted = args.weighted.resolve()
    manifest = read_json(study / "manifest.json")
    uniform_summary = read_json(study / "capacity-summary.json")
    weighted_summary = read_json(weighted / "summary.json")
    if weighted_summary["uniform_checkpoint_sha256"] != uniform_summary["checkpoint_sha256"]:
        raise ValueError("Weighted and uniform checkpoints are not matched")
    uniform_checkpoint = study / "four_layer_distilled.pt"
    weighted_checkpoint = weighted / "four_layer_cost_weighted.pt"
    if sha256(uniform_checkpoint) != uniform_summary["checkpoint_sha256"]:
        raise ValueError("Uniform checkpoint hash changed")
    if sha256(weighted_checkpoint) != weighted_summary["checkpoint_sha256"]:
        raise ValueError("Weighted checkpoint hash changed")
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.toml")
    torch.set_num_threads(args.threads)
    tokenizer = load_tokenizer(config)
    dataset = load_dataset(IMDB_ID, split="test", revision=IMDB_REVISION)
    permutation = np.random.default_rng(2027).permutation(len(dataset))
    selected = []
    class_counts = Counter()
    scanned = 0
    eligible_scanned = Counter()
    oversize_scanned = Counter()
    for index in permutation:
        scanned += 1
        row = dataset[int(index)]
        label = int(row["label"])
        if class_counts[label] >= 500:
            continue
        tokenized = tokenizer(row["text"], truncation=False)
        tokens = len(tokenized["input_ids"])
        if tokens > config.max_length:
            oversize_scanned[label] += 1
            continue
        eligible_scanned[label] += 1
        selected.append(
            {
                "dataset_index": int(index),
                "label": label,
                "tokens": tokens,
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            }
        )
        class_counts[label] += 1
        if class_counts[0] == 500 and class_counts[1] == 500:
            break
    if class_counts[0] != 500 or class_counts[1] != 500:
        raise ValueError("Insufficient eligible reviews for the prespecified sample")
    if len({row["dataset_index"] for row in selected}) != 1000:
        raise ValueError("Sampled external indices are not unique")
    write_json(
        output / "selection.json",
        {
            "dataset_id": IMDB_ID,
            "revision": IMDB_REVISION,
            "split": "test",
            "seed": 2027,
            "selection": "first 500 per class from shuffled test indices with <=512 tokens",
            "scanned_indices": scanned,
            "eligible_selected_by_class": dict(eligible_scanned),
            "oversize_rejected_by_class_before_quotas": dict(oversize_scanned),
            "total_test_rows": len(dataset),
            "class_counts": dict(class_counts),
            "rows": [
                {key: row[key] for key in ("dataset_index", "label", "tokens")} for row in selected
            ],
        },
    )
    encoded = [{key: row[key] for key in ("input_ids", "attention_mask")} for row in selected]
    labels = np.asarray([row["label"] for row in selected])
    lengths = [row["tokens"] for row in selected]
    teacher = load_pytorch(config, threads=args.threads)
    teacher_logits = predict(teacher, tokenizer, encoded, args.batch_size)
    uniform = student_from_checkpoint(teacher, uniform_checkpoint)
    uniform_logits = predict(uniform, tokenizer, encoded, args.batch_size)
    del uniform
    weighted_model = student_from_checkpoint(teacher, weighted_checkpoint)
    weighted_logits = predict(weighted_model, tokenizer, encoded, args.batch_size)
    external = comparison(labels, teacher_logits, uniform_logits, weighted_logits, lengths)
    write_json(
        output / "external-logits.json",
        {
            "dataset_id": IMDB_ID,
            "revision": IMDB_REVISION,
            "rows": [
                {
                    "dataset_index": row["dataset_index"],
                    "label": row["label"],
                    "tokens": row["tokens"],
                    "teacher_logits": teacher_logits[i].tolist(),
                    "uniform_logits": uniform_logits[i].tolist(),
                    "weighted_logits": weighted_logits[i].tolist(),
                }
                for i, row in enumerate(selected)
            ],
        },
    )
    print(json.dumps({"event": "external_evaluated", "metrics": external["metrics"]}), flush=True)

    sst_validation = load_dataset(
        "nyu-mll/glue", "sst2", split="validation", revision=manifest["dataset_revision"]
    )
    sst_encoded_all = tokenizer([row["sentence"] for row in sst_validation], truncation=False)
    sst_encoded = [
        {"input_ids": ids, "attention_mask": mask}
        for ids, mask in zip(
            sst_encoded_all["input_ids"], sst_encoded_all["attention_mask"], strict=True
        )
    ]
    if max(len(item["input_ids"]) for item in sst_encoded) > config.max_length:
        raise ValueError("SST-2 validation input exceeds max_length")
    sst_labels = np.asarray([int(row["label"]) for row in sst_validation])
    frozen_sst = read_json(study / "validation-logits.json")["rows"]
    if any(int(row["idx"]) != int(sst_validation[i]["idx"]) for i, row in enumerate(frozen_sst)):
        raise ValueError("Frozen SST-2 validation logit order changed")
    sst_teacher = np.asarray([row["teacher_logits"] for row in frozen_sst])
    sst_uniform = np.asarray([row["four_layer_distilled_logits"] for row in frozen_sst])
    sst_weighted = predict(weighted_model, tokenizer, sst_encoded, args.batch_size)
    sst_result = comparison(
        sst_labels,
        sst_teacher,
        sst_uniform,
        sst_weighted,
        [len(item["input_ids"]) for item in sst_encoded],
    )
    write_json(
        output / "summary.json",
        {
            "status": "external IMDb primary; reused SST-2 validation exploratory",
            "external_selection_sha256": sha256(output / "selection.json"),
            "external_logits_sha256": sha256(output / "external-logits.json"),
            "uniform_checkpoint_sha256": sha256(uniform_checkpoint),
            "weighted_checkpoint_sha256": sha256(weighted_checkpoint),
            "external": external,
            "sst2_validation_exploratory": sst_result,
        },
    )
    print(
        json.dumps({"event": "extension_complete", "sst2_exploratory": sst_result["metrics"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
