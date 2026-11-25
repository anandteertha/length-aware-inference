"""Fit and freeze a four-layer teacher-benefit gate using SST-2 train rows only.

No GLUE validation labels or logits are read here. The released teacher saw the
SST-2 train split during its original fine-tuning, so this calibration may not
transfer; evaluate that risk separately on validation.
"""

import argparse
import copy
import gc
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from datasets import load_dataset

from etdl.config import load_config
from etdl.data import DATASET_ID, load_tokenizer
from etdl.io import read_json, sha256, write_json
from etdl.model import load_pytorch
from etdl.runtime import session

BUDGETS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
POLICIES = ("confidence", "length", "joint", "joint_per_cost")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("artifacts/student-study"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/student-study/gate-fit"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def length_bin(tokens):
    if tokens <= 16:
        return "3-16"
    if tokens <= 32:
        return "17-32"
    return "33+"


def model_logits(model, tokenizer, encoded, batch_size):
    logits = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            batch = tokenizer.pad(encoded[start : start + batch_size], return_tensors="pt")
            logits.extend(model(**batch).logits.tolist())
    return np.asarray(logits)


def base_features(records):
    logits = np.asarray([row["student_logits"] for row in records])
    margin = np.abs(logits[:, 1] - logits[:, 0])
    log_length = np.log2(np.asarray([row["tokens"] for row in records]))
    return np.column_stack((margin, log_length))


def design(z, policy):
    ones = np.ones(len(z))
    if policy == "confidence":
        return np.column_stack((ones, z[:, 0]))
    if policy == "length":
        return np.column_stack((ones, z[:, 1]))
    return np.column_stack((ones, z[:, 0], z[:, 1], z[:, 0] * z[:, 1]))


def score(records, policy, parameters):
    raw = base_features(records)
    mean = np.asarray(parameters["feature_mean"])
    scale = np.asarray(parameters["feature_scale"])
    z = (raw - mean) / scale
    value = design(z, "joint" if policy == "joint_per_cost" else policy) @ np.asarray(
        parameters["coefficients"]["joint" if policy == "joint_per_cost" else policy]
    )
    if policy == "joint_per_cost":
        costs = np.asarray(
            [parameters["teacher_cost_ms"][length_bin(row["tokens"])] for row in records]
        )
        exponent = parameters.get("teacher_cost_exponent", 1.0)
        value = value / (costs if exponent == 1.0 else np.power(costs, exponent))
    return value


def route(scores, point):
    if point["route_mode"] == "none":
        return np.zeros(len(scores), dtype=bool)
    if point["route_mode"] == "all":
        return np.ones(len(scores), dtype=bool)
    return scores >= point["threshold"]


def main():
    args = arguments()
    if min(args.threads, args.batch_size) <= 0:
        raise ValueError("Threads and batch size must be positive")
    study = args.study.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Gate output already exists: {output}")
    output.mkdir(parents=True)
    manifest = read_json(study / "manifest.json")
    capacity = read_json(study / "capacity-summary.json")
    if capacity["student_layers"] != 4:
        raise ValueError("Expected the frozen four-layer capacity checkpoint")
    checkpoint = study / "four_layer_distilled.pt"
    if sha256(checkpoint) != capacity["checkpoint_sha256"]:
        raise ValueError("Four-layer checkpoint hash changed")
    teacher_cache = torch.load(study / "teacher_logits.pt", weights_only=True)
    offset = len(manifest["partitions"]["student_fit"]) + len(manifest["partitions"]["student_dev"])
    fit_ids = manifest["partitions"]["gate_fit"]
    calibration_ids = manifest["partitions"]["gate_calibration"]
    selected_ids = fit_ids + calibration_ids
    teacher_logits = teacher_cache[offset : offset + len(selected_ids)].numpy()
    if teacher_logits.shape != (len(selected_ids), 2):
        raise ValueError("Teacher cache does not match gate partitions")
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.toml")
    torch.set_num_threads(args.threads)
    tokenizer = load_tokenizer(config)
    dataset = load_dataset(DATASET_ID, "sst2", split="train", revision=manifest["dataset_revision"])
    needed = set(selected_ids)
    lookup = {int(row["idx"]): row for row in dataset if int(row["idx"]) in needed}
    if len(lookup) != len(selected_ids):
        raise ValueError("Could not reconstruct all pinned gate rows")
    selected = [lookup[idx] for idx in selected_ids]
    tokenized = tokenizer([row["sentence"] for row in selected], truncation=False)
    encoded = [
        {"input_ids": ids, "attention_mask": mask}
        for ids, mask in zip(tokenized["input_ids"], tokenized["attention_mask"], strict=True)
    ]
    if max(len(row["input_ids"]) for row in encoded) > config.max_length:
        raise ValueError("Gate input exceeds max_length; no truncation allowed")
    teacher = load_pytorch(config, threads=args.threads)
    student_config = copy.deepcopy(teacher.config)
    student_config.n_layers = 4
    student = type(teacher)(student_config).cpu().float().eval()
    student.load_state_dict(torch.load(checkpoint, weights_only=True))
    del teacher
    gc.collect()
    student_logits = model_logits(student, tokenizer, encoded, args.batch_size)
    del student
    gc.collect()
    records = [
        {
            "idx": int(row["idx"]),
            "label": int(row["label"]),
            "tokens": len(encoded[i]["input_ids"]),
            "student_logits": student_logits[i].tolist(),
            "teacher_logits": teacher_logits[i].tolist(),
        }
        for i, row in enumerate(selected)
    ]
    write_json(output / "gate-records.json", {"fit_count": len(fit_ids), "rows": records})
    fit_records = records[: len(fit_ids)]
    calibration_records = records[len(fit_ids) :]

    model_path = config.artifact_dir / "models/dynamic.onnx"
    runtime, _ = session(model_path, threads=args.threads)
    rng = random.Random(manifest["seed"])
    order = list(range(len(fit_ids)))
    rng.shuffle(order)
    for i in order[:32]:
        inputs = {name: np.asarray([encoded[i][name]], dtype=np.int64) for name in encoded[i]}
        runtime.run(["logits"], inputs)
    times = {name: [] for name in ("3-16", "17-32", "33+")}
    power_start = psutil.sensors_battery()
    for i in order:
        inputs = {name: np.asarray([encoded[i][name]], dtype=np.int64) for name in encoded[i]}
        start = time.perf_counter_ns()
        runtime.run(["logits"], inputs)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        times[length_bin(len(encoded[i]["input_ids"]))].append(elapsed)
    power_end = psutil.sensors_battery()
    if any(len(values) < 30 for values in times.values()):
        raise ValueError("A teacher-cost length stratum has fewer than 30 examples")
    raw_cost = {name: statistics.median(values) for name, values in times.items()}
    cost_values = np.maximum.accumulate(list(raw_cost.values()))
    cost = dict(zip(raw_cost, cost_values.tolist(), strict=True))

    raw_fit = base_features(fit_records)
    feature_mean = raw_fit.mean(axis=0)
    feature_scale = raw_fit.std(axis=0)
    if np.any(feature_scale == 0):
        raise ValueError("Degenerate gate feature scale")
    z_fit = (raw_fit - feature_mean) / feature_scale
    fit_student = student_logits[: len(fit_ids)].argmax(axis=1)
    fit_teacher = teacher_logits[: len(fit_ids)].argmax(axis=1)
    fit_labels = np.asarray([row["label"] for row in fit_records])
    delta = (fit_teacher == fit_labels).astype(float) - (fit_student == fit_labels).astype(float)
    coefficients = {}
    for policy in ("confidence", "length", "joint"):
        x = design(z_fit, policy)
        penalty = np.eye(x.shape[1]) * 10.0
        penalty[0, 0] = 0.0
        coefficients[policy] = np.linalg.solve(x.T @ x + penalty, x.T @ delta).tolist()
    parameters = {
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "coefficients": coefficients,
        "teacher_cost_ms": cost,
    }
    calibration_labels = np.asarray([row["label"] for row in calibration_records])
    calibration_student = student_logits[len(fit_ids) :].argmax(axis=1)
    calibration_teacher = teacher_logits[len(fit_ids) :].argmax(axis=1)
    teacher_calibration_accuracy = float(np.mean(calibration_teacher == calibration_labels))
    curves = {}
    candidates = []
    for policy in POLICIES:
        values = score(calibration_records, policy, parameters)
        points = []
        for budget in BUDGETS:
            route_mode = "none" if budget == 0 else "all" if budget == 1 else "threshold"
            threshold = (
                float(np.quantile(values, 1 - budget)) if route_mode == "threshold" else None
            )
            point = {
                "requested_fallback_fraction": budget,
                "route_mode": route_mode,
                "threshold": threshold,
            }
            fallback = route(values, point)
            prediction = np.where(fallback, calibration_teacher, calibration_student)
            point["calibration_accuracy"] = float(np.mean(prediction == calibration_labels))
            point["calibration_fallback_fraction"] = float(np.mean(fallback))
            point["estimated_extra_teacher_ms_per_request"] = float(
                np.mean(
                    [
                        cost[length_bin(row["tokens"])] if fallback[i] else 0.0
                        for i, row in enumerate(calibration_records)
                    ]
                )
            )
            points.append(point)
            if point["calibration_accuracy"] >= teacher_calibration_accuracy - 0.01:
                candidates.append(
                    (point["estimated_extra_teacher_ms_per_request"], policy, len(points) - 1)
                )
        curves[policy] = points
    if not candidates:
        raise ValueError("The all-teacher route should meet its own accuracy budget")
    _, selected_policy, selected_index = min(candidates)
    plan = {
        "status": "frozen from train-derived gate fit and calibration; validation unused",
        "dataset_revision": manifest["dataset_revision"],
        "teacher_model_sha256": sha256(model_path),
        "student_checkpoint_sha256": sha256(checkpoint),
        "gate_records_sha256": sha256(output / "gate-records.json"),
        "fit_count": len(fit_records),
        "calibration_count": len(calibration_records),
        "fit_teacher_helps": int(np.sum(delta == 1)),
        "fit_teacher_harms": int(np.sum(delta == -1)),
        "ridge_penalty": 10.0,
        "parameters": parameters,
        "teacher_cost_raw_median_ms": raw_cost,
        "teacher_cost_counts": {name: len(values) for name, values in times.items()},
        "teacher_cost_power_plugged": [
            None if power_start is None else power_start.power_plugged,
            None if power_end is None else power_end.power_plugged,
        ],
        "teacher_calibration_accuracy": teacher_calibration_accuracy,
        "curves": curves,
        "selected_policy": selected_policy,
        "selected_point_index": selected_index,
        "selection_rule": "among points within 1 pp of teacher on gate calibration, minimize measured-bin extra teacher ms",
    }
    write_json(output / "gate-plan.json", plan)
    print(
        json.dumps(
            {
                "event": "gate_frozen",
                "selected_policy": selected_policy,
                "selected_point": curves[selected_policy][selected_index],
                "fit_teacher_helps": plan["fit_teacher_helps"],
                "fit_teacher_harms": plan["fit_teacher_harms"],
                "teacher_cost_raw_median_ms": raw_cost,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
