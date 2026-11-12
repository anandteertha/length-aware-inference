"""Audit trained student accuracy on pinned GLUE SST-2 validation once.

This script reports paired errors and row-level logits. It does not choose a gate,
threshold, checkpoint, or hyperparameter using validation labels.
"""

import argparse
import copy
import json
from pathlib import Path

import torch
from datasets import load_dataset

from etdl.config import load_config
from etdl.data import DATASET_ID, load_tokenizer
from etdl.io import read_json, sha256, write_json
from etdl.model import load_pytorch


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("artifacts/student-study"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def predict(model, tokenizer, encodings, batch_size):
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(encodings), batch_size):
            subset = encodings[start : start + batch_size]
            batch = tokenizer.pad(subset, padding=True, return_tensors="pt")
            outputs.extend(model(**batch).logits.tolist())
    return outputs


def student_model(teacher, checkpoint, layers):
    student_config = copy.deepcopy(teacher.config)
    student_config.n_layers = layers
    student = type(teacher)(student_config).cpu().float()
    student.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return student.eval()


def truncated_teacher(teacher):
    student_config = copy.deepcopy(teacher.config)
    student_config.n_layers = 3
    student = type(teacher)(student_config).cpu().float()
    weights = {
        name: value
        for name, value in teacher.state_dict().items()
        if not any(name.startswith(f"distilbert.transformer.layer.{i}.") for i in range(3, 6))
    }
    student.load_state_dict(weights, strict=True)
    return student.eval()


def main():
    args = arguments()
    study = args.study.resolve()
    manifest = read_json(study / "manifest.json")
    training_summary = read_json(study / "summary.json")
    if sha256(study / "manifest.json") != training_summary["manifest_sha256"]:
        raise ValueError("Training manifest hash changed")
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.toml")
    if min(args.batch_size, args.threads) <= 0:
        raise ValueError("Batch size and threads must be positive")
    torch.set_num_threads(args.threads)
    tokenizer = load_tokenizer(config)
    validation = load_dataset(
        DATASET_ID, "sst2", split="validation", revision=manifest["dataset_revision"]
    )
    if len(validation) != 872:
        raise ValueError("Expected the pinned 872-row SST-2 validation split")
    encoded_all = tokenizer([row["sentence"] for row in validation], truncation=False)
    encodings = [
        {"input_ids": ids, "attention_mask": mask}
        for ids, mask in zip(
            encoded_all["input_ids"], encoded_all["attention_mask"], strict=True
        )
    ]
    if max(len(item["input_ids"]) for item in encodings) > config.max_length:
        raise ValueError("Validation input exceeds max_length; no truncation is allowed")
    teacher = load_pytorch(config, threads=args.threads)
    teacher_logits = predict(teacher, tokenizer, encodings, args.batch_size)
    student_logits = {}
    capacity_path = study / "capacity-summary.json"
    capacity = read_json(capacity_path) if capacity_path.exists() else None
    modes = ("untrained", "supervised", "distilled") + (
        ("four_layer_distilled",) if capacity else ()
    )
    for mode in modes:
        if mode == "untrained":
            student = truncated_teacher(teacher)
        else:
            checkpoint = study / f"{mode}.pt"
            expected_hash = (
                capacity["checkpoint_sha256"]
                if mode == "four_layer_distilled"
                else training_summary["student_arms"][mode]["checkpoint_sha256"]
            )
            if sha256(checkpoint) != expected_hash:
                raise ValueError(f"{mode} checkpoint hash changed")
            student = student_model(teacher, checkpoint, 4 if mode == "four_layer_distilled" else 3)
        student_logits[mode] = predict(student, tokenizer, encodings, args.batch_size)
        del student

    records = []
    for i, row in enumerate(validation):
        records.append(
            {
                "idx": int(row["idx"]),
                "label": int(row["label"]),
                "tokens": len(encodings[i]["input_ids"]),
                "teacher_logits": teacher_logits[i],
                "untrained_logits": student_logits["untrained"][i],
                "supervised_logits": student_logits["supervised"][i],
                "distilled_logits": student_logits["distilled"][i],
                **(
                    {"four_layer_distilled_logits": student_logits["four_layer_distilled"][i]}
                    if capacity
                    else {}
                ),
            }
        )
    write_json(study / "validation-logits.json", {"split": "validation", "rows": records})

    metrics = {}
    teacher_predictions = [max(range(2), key=lambda j: logits[j]) for logits in teacher_logits]
    teacher_correct = [teacher_predictions[i] == row["label"] for i, row in enumerate(records)]
    metrics["teacher"] = {
        "correct": sum(teacher_correct),
        "accuracy": sum(teacher_correct) / len(records),
    }
    for mode in modes:
        student_predictions = [
            max(range(2), key=lambda j: logits[j]) for logits in student_logits[mode]
        ]
        student_correct = [
            student_predictions[i] == row["label"] for i, row in enumerate(records)
        ]
        metrics[mode] = {
            "correct": sum(student_correct),
            "accuracy": sum(student_correct) / len(records),
            "teacher_corrects_student": sum(
                teacher_correct[i] and not student_correct[i] for i in range(len(records))
            ),
            "student_corrects_teacher": sum(
                student_correct[i] and not teacher_correct[i] for i in range(len(records))
            ),
            "prediction_agreement": sum(
                teacher_predictions[i] == student_predictions[i] for i in range(len(records))
            ) / len(records),
        }
    strata = [
        ("3-16", [i for i, row in enumerate(records) if row["tokens"] <= 16]),
        ("17-32", [i for i, row in enumerate(records) if 17 <= row["tokens"] <= 32]),
        ("33-64", [i for i, row in enumerate(records) if 33 <= row["tokens"] <= 64]),
        ("65+", [i for i, row in enumerate(records) if row["tokens"] >= 65]),
    ]
    if len(strata[-1][1]) < 30:
        strata[-2] = ("33+", strata[-2][1] + strata[-1][1])
        strata.pop()
    length_metrics = {}
    for name, indices in strata:
        if not indices:
            continue
        length_metrics[name] = {"examples": len(indices)}
        for mode in ("teacher",) + modes:
            logit_key = f"{mode}_logits"
            length_metrics[name][f"{mode}_correct"] = sum(
                max(range(2), key=lambda j: records[i][logit_key][j]) == records[i]["label"]
                for i in indices
            )
    report = {
        "dataset_revision": manifest["dataset_revision"],
        "split": "validation",
        "examples": len(records),
        "prior_project_use": "aggregate validation accuracy reported for original inference study",
        "selection_policy": "no checkpoint or threshold selected from these validation labels",
        "validation_logits_sha256": sha256(study / "validation-logits.json"),
        "metrics": metrics,
        "length_strata": length_metrics,
    }
    write_json(study / "validation-summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
