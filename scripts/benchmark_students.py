"""Benchmark teacher and trained student ONNX graphs in shuffled CPU processes.

Each method runs in a fresh process with four ORT threads. The validation examples
are used as a fixed workload, not to select checkpoints or routing thresholds.
"""

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

from etdl.config import load_config
from etdl.data import DATASET_ID, load_tokenizer
from etdl.io import read_json, sha256, write_json
from etdl.runtime import session


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("artifacts/student-study"))
    parser.add_argument("--export", type=Path, default=Path("artifacts/student-study/onnx-verified"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/student-study/cpu-timing"))
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--limit", type=int, default=872)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--method")
    parser.add_argument("--block", type=int)
    return parser.parse_args()


def percentile(values, q):
    return float(np.percentile(np.asarray(values), q))


def worker(args):
    export = args.export.resolve()
    output = args.output.resolve()
    input_data = read_json(output / f"inputs-{args.method}.json")
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.toml")
    tokenizer = load_tokenizer(config)
    model_path = (
        config.artifact_dir / "models/dynamic.onnx"
        if args.method == "teacher"
        else export / f"{args.method}.onnx"
    )
    runtime, startup_ms = session(model_path, threads=args.threads)
    process = psutil.Process()
    battery_start = psutil.sensors_battery()
    rng = random.Random(2027 + args.block)
    rows = input_data["rows"].copy()
    rng.shuffle(rows)
    for row in rows[:32]:
        tokenized = tokenizer(row["text"], truncation=False, return_tensors="np")
        inputs = {name: tokenized[name] for name in ("input_ids", "attention_mask")}
        runtime.run(["logits"], inputs)
    inference_ms = []
    total_ms = []
    correct = 0
    changed_vs_pytorch = 0
    max_rss = process.memory_info().rss
    for row in rows:
        start = time.perf_counter_ns()
        tokenized = tokenizer(row["text"], truncation=False, return_tensors="np")
        inputs = {name: tokenized[name] for name in ("input_ids", "attention_mask")}
        if inputs["input_ids"].shape[1] > config.max_length:
            raise ValueError("Benchmark input exceeds max_length; no truncation is allowed")
        encoded = time.perf_counter_ns()
        logits = runtime.run(["logits"], inputs)[0]
        finished = time.perf_counter_ns()
        prediction = int(logits.argmax(axis=-1)[0])
        correct += int(prediction == row["label"])
        changed_vs_pytorch += int(prediction != row["expected_prediction"])
        inference_ms.append((finished - encoded) / 1e6)
        total_ms.append((finished - start) / 1e6)
        max_rss = max(max_rss, process.memory_info().rss)
    battery_end = psutil.sensors_battery()
    info = process.memory_info()
    report = {
        "method": args.method,
        "block": args.block,
        "examples": len(rows),
        "threads": args.threads,
        "model_sha256": sha256(model_path),
        "startup_ms": startup_ms,
        "correct": correct,
        "accuracy": correct / len(rows),
        "changed_vs_pytorch": changed_vs_pytorch,
        "inference_median_ms": statistics.median(inference_ms),
        "inference_p95_ms": percentile(inference_ms, 95),
        "end_to_end_mean_ms": statistics.mean(total_ms),
        "end_to_end_median_ms": statistics.median(total_ms),
        "end_to_end_p95_ms": percentile(total_ms, 95),
        "observed_max_rss_mib": max_rss / 1024**2,
        "process_peak_wset_mib": getattr(info, "peak_wset", max_rss) / 1024**2,
        "power_plugged_start": None if battery_start is None else battery_start.power_plugged,
        "power_plugged_end": None if battery_end is None else battery_end.power_plugged,
    }
    write_json(output / f"{args.method}-block-{args.block}.json", report)
    print(json.dumps({"event": "timed", **report}), flush=True)


def parent(args):
    if min(args.blocks, args.limit, args.threads) <= 0:
        raise ValueError("Blocks, limit, and threads must be positive")
    study = args.study.resolve()
    export = args.export.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Timing output already exists: {output}")
    output.mkdir(parents=True)
    manifest = read_json(study / "manifest.json")
    validation_logits = read_json(study / "validation-logits.json")
    validation_summary = read_json(study / "validation-summary.json")
    export_summary = read_json(export / "export-summary.json")
    if sha256(study / "validation-logits.json") != validation_summary["validation_logits_sha256"]:
        raise ValueError("Validation-logit hash changed")
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.toml")
    dataset = load_dataset(
        DATASET_ID, "sst2", split="validation", revision=manifest["dataset_revision"]
    )
    if len(dataset) != 872 or args.limit > len(dataset):
        raise ValueError("Expected 872 validation rows and a limit no greater than 872")
    methods = ("teacher", "supervised", "distilled", "four_layer_distilled")
    for mode in methods[1:]:
        path = export / f"{mode}.onnx"
        if sha256(path) != export_summary[mode]["onnx_sha256"]:
            raise ValueError(f"{mode} ONNX hash changed")
    if not (config.artifact_dir / "models/dynamic.onnx").exists():
        raise ValueError("Export the original dynamic teacher first")
    selected = list(range(args.limit))
    rows = [
        {
            "idx": int(dataset[i]["idx"]),
            "text": dataset[i]["sentence"],
            "label": int(dataset[i]["label"]),
        }
        for i in selected
    ]
    archived = {row["idx"]: row for row in validation_logits["rows"]}
    input_hashes = {}
    for method in methods:
        key = f"{method}_logits"
        method_rows = [
            {
                **row,
                "expected_prediction": int(np.argmax(archived[row["idx"]][key])),
            }
            for row in rows
        ]
        input_path = output / f"inputs-{method}.json"
        write_json(input_path, {"split": "validation", "rows": method_rows})
        input_hashes[method] = sha256(input_path)
    reports = []
    for block in range(args.blocks):
        order = list(methods)
        random.Random(2027 + block).shuffle(order)
        for method in order:
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--worker",
                "--study",
                str(study),
                "--export",
                str(export),
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
            reports.append(read_json(output / f"{method}-block-{block}.json"))
    by_method = {}
    for method in methods:
        subset = [report for report in reports if report["method"] == method]
        by_method[method] = {
            "blocks": len(subset),
            "accuracy": [report["accuracy"] for report in subset],
            "changed_vs_pytorch": [report["changed_vs_pytorch"] for report in subset],
            "median_of_end_to_end_medians_ms": statistics.median(
                report["end_to_end_median_ms"] for report in subset
            ),
            "median_of_end_to_end_p95_ms": statistics.median(
                report["end_to_end_p95_ms"] for report in subset
            ),
            "end_to_end_median_range_ms": [
                min(report["end_to_end_median_ms"] for report in subset),
                max(report["end_to_end_median_ms"] for report in subset),
            ],
            "max_observed_rss_mib": max(report["observed_max_rss_mib"] for report in subset),
        }
    write_json(
        output / "summary.json",
        {
            "validation_inputs_sha256": input_hashes,
            "dataset_revision": manifest["dataset_revision"],
            "model_export_summary_sha256": sha256(export / "export-summary.json"),
            "threads": args.threads,
            "examples_per_method_per_block": args.limit,
            "warmup_requests": min(args.limit, 32),
            "methods": by_method,
            "process_order": [f"{r['method']}-block-{r['block']}" for r in reports],
        },
    )
    print(json.dumps({"event": "complete", "summary": by_method}), flush=True)


if __name__ == "__main__":
    current_args = arguments()
    if current_args.worker:
        worker(current_args)
    else:
        parent(current_args)
