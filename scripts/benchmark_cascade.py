"""Time frozen gate choices with both ONNX models resident in fresh CPU processes."""

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
from datasets import load_dataset
from fit_gate import route, score

from etdl.config import load_config
from etdl.data import DATASET_ID, load_tokenizer
from etdl.io import read_json, sha256, write_json
from etdl.runtime import session

METHODS = ("student_only", "teacher_only", "selected_gate", "confidence_gate")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("artifacts/student-study"))
    parser.add_argument("--gate", type=Path, default=Path("artifacts/student-study/gate-fit"))
    parser.add_argument(
        "--export", type=Path, default=Path("artifacts/student-study/onnx-verified")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/student-study/cascade-timing")
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--limit", type=int, default=872)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--balanced-order", action="store_true")
    parser.add_argument("--require-ac", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--method")
    parser.add_argument("--block", type=int)
    return parser.parse_args()


def percentile(values, q):
    return float(np.percentile(np.asarray(values), q))


def require_ac():
    battery = psutil.sensors_battery()
    if battery is None or not battery.power_plugged:
        raise RuntimeError("AC power is required for this benchmark")


def method_gate(method, benchmark_plan):
    if method == "selected_gate":
        return benchmark_plan["selected_policy"], benchmark_plan["selected_point"]
    if method == "confidence_gate":
        return "confidence", benchmark_plan["confidence_point"]
    return None, None


def worker(args):
    output = args.output.resolve()
    benchmark_plan = read_json(output / "benchmark-plan.json")
    gate_plan = read_json(args.gate.resolve() / "gate-plan.json")
    rows = read_json(output / f"inputs-{args.method}.json")["rows"]
    rng = random.Random(2027 + args.block)
    rng.shuffle(rows)
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.toml")
    tokenizer = load_tokenizer(config)
    student_path = args.export.resolve() / "four_layer_distilled.onnx"
    teacher_path = config.artifact_dir / "models/dynamic.onnx"
    student, student_startup_ms = session(
        student_path, threads=args.threads, allow_spinning=False
    )
    teacher, teacher_startup_ms = session(
        teacher_path, threads=args.threads, allow_spinning=False
    )
    process = psutil.Process()
    power_start = psutil.sensors_battery()
    for row in rows[:32]:
        encoded = tokenizer(row["text"], truncation=False, return_tensors="np")
        inputs = {name: encoded[name] for name in ("input_ids", "attention_mask")}
        student.run(["logits"], inputs)
        teacher.run(["logits"], inputs)

    policy, point = method_gate(args.method, benchmark_plan)
    total_ms = []
    inference_ms = []
    correct = 0
    fallback_count = 0
    route_disagreements = 0
    changed_vs_expected = 0
    max_rss = process.memory_info().rss
    for row in rows:
        start = time.perf_counter_ns()
        encoded = tokenizer(row["text"], truncation=False, return_tensors="np")
        inputs = {name: encoded[name] for name in ("input_ids", "attention_mask")}
        if inputs["input_ids"].shape[1] > config.max_length:
            raise ValueError("Input exceeds max_length; no truncation is allowed")
        after_tokenization = time.perf_counter_ns()
        if args.method == "teacher_only":
            fallback = True
            logits = teacher.run(["logits"], inputs)[0]
        else:
            student_logits = student.run(["logits"], inputs)[0]
            if args.method == "student_only":
                fallback = False
            else:
                record = {
                    "student_logits": student_logits[0].tolist(),
                    "tokens": int(inputs["input_ids"].shape[1]),
                }
                fallback = bool(route(score([record], policy, gate_plan["parameters"]), point)[0])
            logits = teacher.run(["logits"], inputs)[0] if fallback else student_logits
        finished = time.perf_counter_ns()
        prediction = int(logits.argmax(axis=-1)[0])
        correct += int(prediction == row["label"])
        fallback_count += int(fallback)
        route_disagreements += int(fallback != row["expected_fallback"])
        changed_vs_expected += int(prediction != row["expected_prediction"])
        inference_ms.append((finished - after_tokenization) / 1e6)
        total_ms.append((finished - start) / 1e6)
        max_rss = max(max_rss, process.memory_info().rss)
    power_end = psutil.sensors_battery()
    info = process.memory_info()
    result = {
        "method": args.method,
        "block": args.block,
        "examples": len(rows),
        "threads_per_session": args.threads,
        "ort_thread_spinning": False,
        "student_startup_ms": student_startup_ms,
        "teacher_startup_ms": teacher_startup_ms,
        "correct": correct,
        "accuracy": correct / len(rows),
        "fallback_count": fallback_count,
        "fallback_fraction": fallback_count / len(rows),
        "route_disagreements_vs_frozen_logits": route_disagreements,
        "changed_predictions_vs_frozen_logits": changed_vs_expected,
        "inference_median_ms": statistics.median(inference_ms),
        "inference_p95_ms": percentile(inference_ms, 95),
        "end_to_end_mean_ms": statistics.mean(total_ms),
        "end_to_end_median_ms": statistics.median(total_ms),
        "end_to_end_p95_ms": percentile(total_ms, 95),
        "observed_max_rss_mib": max_rss / 1024**2,
        "process_peak_wset_mib": getattr(info, "peak_wset", max_rss) / 1024**2,
        "power_plugged_start": None if power_start is None else power_start.power_plugged,
        "power_plugged_end": None if power_end is None else power_end.power_plugged,
    }
    write_json(output / f"{args.method}-block-{args.block}.json", result)
    print(json.dumps({"event": "cascade_timed", **result}), flush=True)


def parent(args):
    if min(args.threads, args.blocks, args.limit) <= 0:
        raise ValueError("Threads, blocks, and limit must be positive")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("Benchmarked methods must be distinct")
    if args.require_ac:
        require_ac()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Cascade timing output exists: {output}")
    output.mkdir(parents=True)
    study = args.study.resolve()
    gate_path = args.gate.resolve() / "gate-plan.json"
    gate_plan = read_json(gate_path)
    validation_summary = read_json(study / "validation-summary.json")
    validation_path = study / "validation-logits.json"
    if sha256(validation_path) != validation_summary["validation_logits_sha256"]:
        raise ValueError("Validation logits changed")
    validation = {row["idx"]: row for row in read_json(validation_path)["rows"]}
    teacher_calibration_accuracy = gate_plan["teacher_calibration_accuracy"]
    confidence_candidates = [
        (point["estimated_extra_teacher_ms_per_request"], index)
        for index, point in enumerate(gate_plan["curves"]["confidence"])
        if point["calibration_accuracy"] >= teacher_calibration_accuracy - 0.01
    ]
    _, confidence_index = min(confidence_candidates)
    benchmark_plan = {
        "gate_plan_sha256": sha256(gate_path),
        "selected_policy": gate_plan["selected_policy"],
        "selected_point": gate_plan["curves"][gate_plan["selected_policy"]][
            gate_plan["selected_point_index"]
        ],
        "confidence_point": gate_plan["curves"]["confidence"][confidence_index],
        "confidence_selection_rule": "same calibration accuracy budget and minimum added teacher time within confidence policy",
        "validation_logits_sha256": sha256(validation_path),
    }
    write_json(output / "benchmark-plan.json", benchmark_plan)
    manifest = read_json(study / "manifest.json")
    dataset = load_dataset(
        DATASET_ID, "sst2", split="validation", revision=manifest["dataset_revision"]
    )
    if len(dataset) != 872 or args.limit > 872:
        raise ValueError("Expected pinned 872-row validation split and limit <= 872")
    input_hashes = {}
    for method in args.methods:
        policy, point = method_gate(method, benchmark_plan)
        selected_rows = []
        for row in dataset.select(range(args.limit)):
            archived = validation[int(row["idx"])]
            if method == "teacher_only":
                fallback = True
            elif method == "student_only":
                fallback = False
            else:
                record = {
                    "student_logits": archived["four_layer_distilled_logits"],
                    "tokens": archived["tokens"],
                }
                fallback = bool(route(score([record], policy, gate_plan["parameters"]), point)[0])
            logits = (
                archived["teacher_logits"] if fallback else archived["four_layer_distilled_logits"]
            )
            selected_rows.append(
                {
                    "idx": int(row["idx"]),
                    "text": row["sentence"],
                    "label": int(row["label"]),
                    "expected_fallback": fallback,
                    "expected_prediction": int(np.argmax(logits)),
                }
            )
        path = output / f"inputs-{method}.json"
        write_json(path, {"split": "validation", "rows": selected_rows})
        input_hashes[method] = sha256(path)
    reports = []
    base_order = list(args.methods)
    random.Random(4107).shuffle(base_order)
    for block in range(args.blocks):
        if args.balanced_order:
            shift = block % len(base_order)
            order = base_order[shift:] + base_order[:shift]
        else:
            order = list(args.methods)
            random.Random(4107 + block).shuffle(order)
        for method in order:
            if args.require_ac:
                require_ac()
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--worker",
                "--study",
                str(study),
                "--gate",
                str(args.gate.resolve()),
                "--export",
                str(args.export.resolve()),
                "--output",
                str(output),
                "--threads",
                str(args.threads),
                "--method",
                method,
                "--block",
                str(block),
            ]
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            print(completed.stdout.strip(), flush=True)
            result = read_json(output / f"{method}-block-{block}.json")
            if args.require_ac and not (
                result["power_plugged_start"] and result["power_plugged_end"]
            ):
                raise RuntimeError(f"AC power check failed for {method} in block {block}")
            reports.append(result)
    aggregate = {}
    for method in args.methods:
        subset = [row for row in reports if row["method"] == method]
        aggregate[method] = {
            "correct": [row["correct"] for row in subset],
            "fallback_count": [row["fallback_count"] for row in subset],
            "changed_predictions_vs_frozen_logits": [
                row["changed_predictions_vs_frozen_logits"] for row in subset
            ],
            "median_of_end_to_end_medians_ms": statistics.median(
                row["end_to_end_median_ms"] for row in subset
            ),
            "median_of_end_to_end_p95_ms": statistics.median(
                row["end_to_end_p95_ms"] for row in subset
            ),
            "end_to_end_median_range_ms": [
                min(row["end_to_end_median_ms"] for row in subset),
                max(row["end_to_end_median_ms"] for row in subset),
            ],
            "max_observed_rss_mib": max(row["observed_max_rss_mib"] for row in subset),
        }
    write_json(
        output / "summary.json",
        {
            "status": "matched resident-session timing; no validation threshold tuning",
            "benchmark_plan_sha256": sha256(output / "benchmark-plan.json"),
            "input_hashes": input_hashes,
            "examples_per_method_per_block": args.limit,
            "threads_per_session": args.threads,
            "ort_thread_spinning": False,
            "warmup_requests_per_session": min(args.limit, 32),
            "process_order": [f"{row['method']}-block-{row['block']}" for row in reports],
            "methods": aggregate,
        },
    )
    print(json.dumps({"event": "cascade_complete", "summary": aggregate}), flush=True)


if __name__ == "__main__":
    current_args = arguments()
    if current_args.worker:
        worker(current_args)
    else:
        parent(current_args)
