"""One-candidate INT8 student audit and matched resident CPU benchmark.

Run from the project root through run_int8_extension.ps1 on AC power for timing.
Use --audit-only to evaluate predictions and graph size without timing. This is an
exploratory extension: SST-2 validation and the selected IMDb sample were already
inspected before the quantization protocol was written.
"""

import argparse
import json
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import psutil
from datasets import load_dataset
from fit_gate import route, score
from onnxruntime.quantization import QuantType, quantize_dynamic

from etdl.config import load_config
from etdl.data import DATASET_ID, load_tokenizer
from etdl.io import read_json, sha256, write_json
from etdl.runtime import graph_operators, session

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "artifacts/student-study"
EXPORT = STUDY / "onnx-verified"
GATE = ROOT / "results/student-study/gate/gate-plan.json"
BENCHMARK_PLAN = ROOT / "results/student-study/cascade-timing/benchmark-plan.json"
EXTERNAL_DIR = ROOT / "results/student-study/cost-weighted"
METHODS = (
    "teacher_only",
    "fp32_student_only",
    "int8_student_only",
    "fp32_confidence_gate",
    "int8_confidence_gate",
)
THREADS = 4
BLOCKS = 3
WARMUP = 32


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY / "int8-extension")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--refresh-shape-metadata",
        action="store_true",
        help="Run the exploratory repair that removes intermediate ONNX shape annotations.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--block", type=int)
    return parser.parse_args()


def require_ac():
    battery = psutil.sensors_battery()
    if battery is None or not battery.power_plugged:
        raise RuntimeError("AC power is required for this run; plug in the laptop")
    return {"power_plugged": True, "battery_percent": battery.percent}


def source_paths():
    config = load_config(ROOT / "configs/default.toml")
    return {
        "fp32": EXPORT / "four_layer_distilled.onnx",
        "teacher": config.artifact_dir / "models/dynamic.onnx",
        "export_summary": EXPORT / "export-summary.json",
        "manifest": STUDY / "manifest.json",
        "validation_logits": STUDY / "validation-logits.json",
        "validation_summary": STUDY / "validation-summary.json",
        "gate": GATE,
        "benchmark_plan": BENCHMARK_PLAN,
        "external_selection": EXTERNAL_DIR / "external-selection.json",
        "external_logits": EXTERNAL_DIR / "external-logits.json",
        "external_summary": EXTERNAL_DIR / "evaluation-summary.json",
    }


def validate_sources():
    paths = source_paths()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source artifacts: {missing}")
    export = read_json(paths["export_summary"])["four_layer_distilled"]
    if sha256(paths["fp32"]) != export["onnx_sha256"]:
        raise ValueError("FP32 student graph changed")
    validation_summary = read_json(paths["validation_summary"])
    if sha256(paths["validation_logits"]) != validation_summary["validation_logits_sha256"]:
        raise ValueError("Frozen SST-2 logits changed")
    benchmark_plan = read_json(paths["benchmark_plan"])
    if sha256(paths["gate"]) != benchmark_plan["gate_plan_sha256"]:
        raise ValueError("Frozen gate changed")
    if sha256(paths["validation_logits"]) != benchmark_plan["validation_logits_sha256"]:
        raise ValueError("Frozen benchmark and validation logits differ")
    external_summary = read_json(paths["external_summary"])
    for key, record_key in (
        ("external_selection", "external_selection_sha256"),
        ("external_logits", "external_logits_sha256"),
    ):
        if sha256(paths[key]) != external_summary[record_key]:
            raise ValueError(f"Frozen {key} changed")
    return {key: sha256(path) for key, path in paths.items()}


def candidate_path(output):
    return output / "four_layer_distilled_int8.onnx"


def strip_shape_metadata(source, destination):
    """Remove intermediate annotations while preserving operators, weights, and public I/O."""
    model = onnx.load(source)
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    onnx.save(model, destination)
    original, _ = session(source, threads=THREADS, allow_spinning=False)
    stripped, _ = session(destination, threads=THREADS, allow_spinning=False)
    for length in (3, 16, 32, 55, 211, 512):
        inputs = {
            "input_ids": np.full((1, length), 1000, dtype=np.int64),
            "attention_mask": np.ones((1, length), dtype=np.int64),
        }
        np.testing.assert_allclose(
            original.run(["logits"], inputs)[0],
            stripped.run(["logits"], inputs)[0],
            atol=1e-4,
            rtol=1e-4,
        )


def make_candidate(output, source_hashes, resume, refresh_shape_metadata):
    path = candidate_path(output)
    plan_path = output / "plan.json"
    if plan_path.exists():
        plan = read_json(plan_path)
        if plan["source_hashes"] != source_hashes or sha256(path) != plan["candidate_sha256"]:
            raise ValueError("Existing quantization plan does not match source artifacts")
        if plan.get("refreshed_shape_metadata", False) != refresh_shape_metadata:
            raise ValueError("Resume with the same quantization source option")
        return plan
    if path.exists():
        if not resume:
            raise ValueError("Candidate exists without a completed plan; use --resume")
        path.unlink()  # This run's incomplete candidate; no completed plan exists.
    source = source_paths()["fp32"]
    print(json.dumps({"event": "quantizing", "source": str(source)}), flush=True)
    with tempfile.TemporaryDirectory(dir=output) as working:
        quant_input = source
        if refresh_shape_metadata:
            quant_input = Path(working) / "student-without-intermediate-shapes.onnx"
            strip_shape_metadata(source, quant_input)
        quantize_dynamic(
            model_input=quant_input,
            model_output=path,
            op_types_to_quantize=["MatMul", "Gemm"],
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QInt8,
        )
    onnx.checker.check_model(str(path))
    operators = graph_operators(path)
    if not any("MatMulInteger" in name or "QLinearMatMul" in name for name in operators):
        raise ValueError("Quantized graph lacks an integer matrix multiplication")
    model, _ = session(path, threads=THREADS, allow_spinning=False)
    tokenizer = load_tokenizer(load_config(ROOT / "configs/default.toml"))
    probe = tokenizer("A short sentiment check.", return_tensors="np")
    logits = model.run(["logits"], {key: probe[key] for key in ("input_ids", "attention_mask")})[0]
    if logits.shape != (1, 2) or not np.isfinite(logits).all():
        raise ValueError("INT8 smoke logits are invalid")
    del model
    plan = {
        "status": "one fixed dynamic-quantization candidate; reused validation/external sets",
        "source_hashes": source_hashes,
        "candidate_sha256": sha256(path),
        "source_bytes": source.stat().st_size,
        "candidate_bytes": path.stat().st_size,
        "source_operators": graph_operators(source),
        "candidate_operators": operators,
        "ort_version": ort.__version__,
        "refreshed_shape_metadata": refresh_shape_metadata,
        "quantization": {
            "api": "onnxruntime.quantization.quantize_dynamic",
            "op_types_to_quantize": ["MatMul", "Gemm"],
            "weight_type": "QInt8",
            "per_channel": False,
            "reduce_range": False,
        },
        "threads_per_session": THREADS,
        "ort_enable_all": True,
        "ort_thread_spinning": False,
    }
    write_json(plan_path, plan)
    print(json.dumps({"event": "quantized", "bytes": plan["candidate_bytes"]}), flush=True)
    return plan


def confidence_fallback(logits, tokens, gate, point):
    record = {"student_logits": logits, "tokens": tokens}
    return bool(route(score([record], "confidence", gate["parameters"]), point)[0])


def audit_validation(output, plan):
    audit_path = output / "validation-audit.json"
    if audit_path.exists():
        audit = read_json(audit_path)
        if audit["candidate_sha256"] != plan["candidate_sha256"]:
            raise ValueError("Validation audit belongs to another candidate")
        return audit
    manifest = read_json(source_paths()["manifest"])
    dataset = load_dataset(
        DATASET_ID, "sst2", split="validation", revision=manifest["dataset_revision"]
    )
    if len(dataset) != 872:
        raise ValueError("Expected 872 pinned SST-2 validation rows")
    archived = read_json(source_paths()["validation_logits"])["rows"]
    if len(archived) != len(dataset):
        raise ValueError("Validation archive length changed")
    gate = read_json(GATE)
    point = read_json(BENCHMARK_PLAN)["confidence_point"]
    tokenizer = load_tokenizer(load_config(ROOT / "configs/default.toml"))
    model, _ = session(candidate_path(output), threads=THREADS, allow_spinning=False)
    records = []
    logit_differences = []
    for i, row in enumerate(dataset):
        old = archived[i]
        if int(row["idx"]) != int(old["idx"]) or int(row["label"]) != int(old["label"]):
            raise ValueError("Validation order or labels changed")
        encoded = tokenizer(row["sentence"], truncation=False, return_tensors="np")
        inputs = {key: encoded[key] for key in ("input_ids", "attention_mask")}
        tokens = int(inputs["input_ids"].shape[1])
        if tokens != int(old["tokens"]) or tokens > 512:
            raise ValueError("Validation token count changed")
        int8_logits = model.run(["logits"], inputs)[0][0]
        if not np.isfinite(int8_logits).all():
            raise ValueError("Nonfinite INT8 logits")
        fp32_logits = np.asarray(old["four_layer_distilled_logits"])
        logit_differences.extend(np.abs(int8_logits - fp32_logits).tolist())
        teacher_pred = int(np.argmax(old["teacher_logits"]))
        fp32_pred = int(np.argmax(fp32_logits))
        int8_pred = int(np.argmax(int8_logits))
        fp32_fallback = confidence_fallback(fp32_logits.tolist(), tokens, gate, point)
        int8_fallback = confidence_fallback(int8_logits.tolist(), tokens, gate, point)
        records.append(
            {
                "idx": int(row["idx"]),
                "label": int(row["label"]),
                "tokens": tokens,
                "fp32_prediction": fp32_pred,
                "int8_prediction": int8_pred,
                "teacher_prediction": teacher_pred,
                "fp32_fallback": fp32_fallback,
                "int8_fallback": int8_fallback,
                "fp32_cascade_prediction": teacher_pred if fp32_fallback else fp32_pred,
                "int8_cascade_prediction": teacher_pred if int8_fallback else int8_pred,
                "int8_logits": int8_logits.tolist(),
            }
        )
    del model
    count_correct = lambda field: sum(row[field] == row["label"] for row in records)
    metrics = {
        "examples": len(records),
        "teacher_correct": count_correct("teacher_prediction"),
        "fp32_student_correct": count_correct("fp32_prediction"),
        "int8_student_correct": count_correct("int8_prediction"),
        "fp32_cascade_correct": count_correct("fp32_cascade_prediction"),
        "int8_cascade_correct": count_correct("int8_cascade_prediction"),
        "fp32_fallbacks": sum(row["fp32_fallback"] for row in records),
        "int8_fallbacks": sum(row["int8_fallback"] for row in records),
        "changed_student_labels": sum(
            row["fp32_prediction"] != row["int8_prediction"] for row in records
        ),
        "changed_confidence_routes": sum(
            row["fp32_fallback"] != row["int8_fallback"] for row in records
        ),
        "max_absolute_logit_difference": max(logit_differences),
        "median_absolute_logit_difference": statistics.median(logit_differences),
    }
    if (
        metrics["teacher_correct"],
        metrics["fp32_student_correct"],
        metrics["fp32_cascade_correct"],
        metrics["fp32_fallbacks"],
    ) != (794, 765, 789, 161):
        raise ValueError("Frozen FP32 reference no longer reproduces archived outcomes")
    audit = {"candidate_sha256": plan["candidate_sha256"], "metrics": metrics, "rows": records}
    write_json(audit_path, audit)
    print(json.dumps({"event": "validation_audited", "metrics": metrics}), flush=True)
    return audit


def audit_external(output, plan):
    audit_path = output / "external-audit.json"
    if audit_path.exists():
        audit = read_json(audit_path)
        if audit["candidate_sha256"] != plan["candidate_sha256"]:
            raise ValueError("External audit belongs to another candidate")
        return audit
    selection = read_json(source_paths()["external_selection"])
    logits_archive = read_json(source_paths()["external_logits"])["rows"]
    if len(selection["rows"]) != 1000 or len(logits_archive) != 1000:
        raise ValueError("Expected 1,000 frozen IMDb rows")
    dataset = load_dataset(selection["dataset_id"], split="test", revision=selection["revision"])
    tokenizer = load_tokenizer(load_config(ROOT / "configs/default.toml"))
    fp32_model, _ = session(source_paths()["fp32"], threads=THREADS, allow_spinning=False)
    int8_model, _ = session(candidate_path(output), threads=THREADS, allow_spinning=False)
    records = []
    for selected, old in zip(selection["rows"], logits_archive, strict=True):
        index = int(selected["dataset_index"])
        row = dataset[index]
        if index != int(old["dataset_index"]) or int(row["label"]) != int(selected["label"]):
            raise ValueError("Frozen IMDb selection changed")
        encoded = tokenizer(row["text"], truncation=False, return_tensors="np")
        inputs = {key: encoded[key] for key in ("input_ids", "attention_mask")}
        tokens = int(inputs["input_ids"].shape[1])
        if tokens != int(selected["tokens"]) or tokens > 512:
            raise ValueError("IMDb token count changed")
        fp32_logits = fp32_model.run(["logits"], inputs)[0][0]
        int8_logits = int8_model.run(["logits"], inputs)[0][0]
        if not np.isfinite(fp32_logits).all() or not np.isfinite(int8_logits).all():
            raise ValueError("Nonfinite IMDb logits")
        records.append(
            {
                "dataset_index": index,
                "label": int(row["label"]),
                "tokens": tokens,
                "teacher_prediction": int(np.argmax(old["teacher_logits"])),
                "archived_uniform_prediction": int(np.argmax(old["uniform_logits"])),
                "fp32_onnx_prediction": int(np.argmax(fp32_logits)),
                "int8_onnx_prediction": int(np.argmax(int8_logits)),
                "fp32_logits": fp32_logits.tolist(),
                "int8_logits": int8_logits.tolist(),
            }
        )
    del fp32_model, int8_model
    correct = lambda name: sum(row[name] == row["label"] for row in records)
    metrics = {
        "examples": len(records),
        "teacher_correct": correct("teacher_prediction"),
        "archived_uniform_correct": correct("archived_uniform_prediction"),
        "fp32_onnx_correct": correct("fp32_onnx_prediction"),
        "int8_onnx_correct": correct("int8_onnx_prediction"),
        "changed_fp32_vs_archived_uniform": sum(
            row["fp32_onnx_prediction"] != row["archived_uniform_prediction"] for row in records
        ),
        "changed_int8_vs_fp32_onnx": sum(
            row["int8_onnx_prediction"] != row["fp32_onnx_prediction"] for row in records
        ),
    }
    if metrics["teacher_correct"] != 895 or metrics["archived_uniform_correct"] != 843:
        raise ValueError("Frozen IMDb reference no longer reproduces archived outcomes")
    audit = {"candidate_sha256": plan["candidate_sha256"], "metrics": metrics, "rows": records}
    write_json(audit_path, audit)
    print(json.dumps({"event": "external_audited", "metrics": metrics}), flush=True)
    return audit


def method_student(method):
    return "int8" if method.startswith("int8_") else "fp32"


def expected_for_method(row, method):
    if method == "teacher_only":
        return row["teacher_prediction"], True
    family = method_student(method)
    if method.endswith("student_only"):
        return row[f"{family}_prediction"], False
    return row[f"{family}_cascade_prediction"], row[f"{family}_fallback"]


def benchmark_worker(args):
    power_start = require_ac()
    output = args.output.resolve()
    plan = read_json(output / "plan.json")
    if sha256(candidate_path(output)) != plan["candidate_sha256"]:
        raise ValueError("Candidate graph changed before worker")
    audit = read_json(output / "validation-audit.json")
    if audit["candidate_sha256"] != plan["candidate_sha256"]:
        raise ValueError("Worker audit belongs to another candidate")
    archived = {row["idx"]: row for row in audit["rows"]}
    manifest = read_json(source_paths()["manifest"])
    dataset = load_dataset(
        DATASET_ID, "sst2", split="validation", revision=manifest["dataset_revision"]
    )
    rows = [
        {"idx": int(row["idx"]), "text": row["sentence"], "label": int(row["label"])}
        for row in dataset
    ]
    if len(rows) != 872 or set(archived) != {row["idx"] for row in rows}:
        raise ValueError("Benchmark validation set does not match accuracy audit")
    random.Random(4041 + args.block).shuffle(rows)
    config = load_config(ROOT / "configs/default.toml")
    tokenizer = load_tokenizer(config)
    student_path = (
        candidate_path(output) if method_student(args.method) == "int8" else source_paths()["fp32"]
    )
    student, student_startup_ms = session(student_path, threads=THREADS, allow_spinning=False)
    teacher, teacher_startup_ms = session(
        source_paths()["teacher"], threads=THREADS, allow_spinning=False
    )
    process = psutil.Process()
    max_rss = process.memory_info().rss
    for row in rows[:WARMUP]:
        encoded = tokenizer(row["text"], truncation=False, return_tensors="np")
        inputs = {key: encoded[key] for key in ("input_ids", "attention_mask")}
        student.run(["logits"], inputs)
        teacher.run(["logits"], inputs)
    gate = read_json(GATE)
    point = read_json(BENCHMARK_PLAN)["confidence_point"]
    requests = []
    for row in rows:
        start = time.perf_counter_ns()
        encoded = tokenizer(row["text"], truncation=False, return_tensors="np")
        inputs = {key: encoded[key] for key in ("input_ids", "attention_mask")}
        tokens = int(inputs["input_ids"].shape[1])
        if tokens > 512 or tokens != archived[row["idx"]]["tokens"]:
            raise ValueError("Benchmark token count changed")
        after_tokenization = time.perf_counter_ns()
        if args.method == "teacher_only":
            fallback = True
            logits = teacher.run(["logits"], inputs)[0]
        else:
            student_logits = student.run(["logits"], inputs)[0]
            fallback = (
                False
                if args.method.endswith("student_only")
                else confidence_fallback(student_logits[0].tolist(), tokens, gate, point)
            )
            logits = teacher.run(["logits"], inputs)[0] if fallback else student_logits
        end = time.perf_counter_ns()
        prediction = int(logits.argmax(axis=-1)[0])
        expected_prediction, expected_fallback = expected_for_method(
            archived[row["idx"]], args.method
        )
        if prediction != expected_prediction or fallback != expected_fallback:
            raise ValueError(f"Timed prediction or route changed on row {row['idx']}")
        requests.append(
            {
                "idx": row["idx"],
                "tokens": tokens,
                "prediction": prediction,
                "correct": prediction == row["label"],
                "fallback": fallback,
                "inference_ms": (end - after_tokenization) / 1e6,
                "end_to_end_ms": (end - start) / 1e6,
            }
        )
        max_rss = max(max_rss, process.memory_info().rss)
    power_end = require_ac()
    info = process.memory_info()
    total = [row["end_to_end_ms"] for row in requests]
    inference = [row["inference_ms"] for row in requests]
    report = {
        "method": args.method,
        "block": args.block,
        "plan_sha256": sha256(output / "plan.json"),
        "model_sha256": sha256(student_path),
        "teacher_sha256": sha256(source_paths()["teacher"]),
        "examples": len(requests),
        "threads_per_session": THREADS,
        "warmup_requests_per_session": WARMUP,
        "power_start": power_start,
        "power_end": power_end,
        "correct": sum(row["correct"] for row in requests),
        "fallbacks": sum(row["fallback"] for row in requests),
        "student_startup_ms": student_startup_ms,
        "teacher_startup_ms": teacher_startup_ms,
        "inference_median_ms": statistics.median(inference),
        "inference_p95_ms": float(np.percentile(inference, 95)),
        "end_to_end_mean_ms": statistics.mean(total),
        "end_to_end_median_ms": statistics.median(total),
        "end_to_end_p95_ms": float(np.percentile(total, 95)),
        "observed_max_rss_mib": max_rss / 1024**2,
        "process_peak_wset_mib": getattr(info, "peak_wset", max_rss) / 1024**2,
        "requests": requests,
    }
    write_json(output / f"{args.method}-block-{args.block}.json", report)
    print(
        json.dumps(
            {
                "event": "timed",
                "method": args.method,
                "block": args.block,
                "correct": report["correct"],
                "fallbacks": report["fallbacks"],
                "median_ms": report["end_to_end_median_ms"],
                "p95_ms": report["end_to_end_p95_ms"],
                "rss_mib": report["observed_max_rss_mib"],
            }
        ),
        flush=True,
    )


def summarize(output, audit):
    reports = {
        method: [read_json(output / f"{method}-block-{block}.json") for block in range(BLOCKS)]
        for method in METHODS
    }
    for method, subset in reports.items():
        expected_prediction_field = {
            "teacher_only": "teacher_correct",
            "fp32_student_only": "fp32_student_correct",
            "int8_student_only": "int8_student_correct",
            "fp32_confidence_gate": "fp32_cascade_correct",
            "int8_confidence_gate": "int8_cascade_correct",
        }[method]
        expected_fallback_field = {
            "teacher_only": None,
            "fp32_student_only": None,
            "int8_student_only": None,
            "fp32_confidence_gate": "fp32_fallbacks",
            "int8_confidence_gate": "int8_fallbacks",
        }[method]
        for block, report in enumerate(subset):
            if report["block"] != block or report["method"] != method:
                raise ValueError("Benchmark report identity changed")
            if report["plan_sha256"] != sha256(output / "plan.json"):
                raise ValueError("Benchmark report belongs to another plan")
            if report["correct"] != audit["metrics"][expected_prediction_field]:
                raise ValueError("Benchmark accuracy differs from untimed audit")
            expected_fallbacks = (
                872
                if method == "teacher_only"
                else audit["metrics"][expected_fallback_field]
                if expected_fallback_field
                else 0
            )
            if report["fallbacks"] != expected_fallbacks:
                raise ValueError("Benchmark fallback count differs from audit")
            if (
                not report["power_start"]["power_plugged"]
                or not report["power_end"]["power_plugged"]
            ):
                raise ValueError("A timing report ran on battery")
    by_method = {}
    for method, subset in reports.items():
        medians = [report["end_to_end_median_ms"] for report in subset]
        by_method[method] = {
            "correct": subset[0]["correct"],
            "fallbacks": subset[0]["fallbacks"],
            "median_of_process_medians_ms": statistics.median(medians),
            "process_medians_ms": medians,
            "median_of_process_p95_ms": statistics.median(
                report["end_to_end_p95_ms"] for report in subset
            ),
            "process_p95_ms": [report["end_to_end_p95_ms"] for report in subset],
            "max_observed_rss_mib": max(report["observed_max_rss_mib"] for report in subset),
            "max_peak_wset_mib": max(report["process_peak_wset_mib"] for report in subset),
        }

    def paired_reduction(reference, candidate):
        first = by_method[reference]["process_medians_ms"]
        second = by_method[candidate]["process_medians_ms"]
        values = [(a - b) / a * 100 for a, b in zip(first, second, strict=True)]
        return {"by_block_percent": values, "median_percent": statistics.median(values)}

    paired = {
        "int8_student_vs_fp32_student": paired_reduction("fp32_student_only", "int8_student_only"),
        "int8_cascade_vs_fp32_cascade": paired_reduction(
            "fp32_confidence_gate", "int8_confidence_gate"
        ),
        "int8_cascade_vs_teacher": paired_reduction("teacher_only", "int8_confidence_gate"),
    }
    teacher_correct = by_method["teacher_only"]["correct"]
    int8_correct = by_method["int8_confidence_gate"]["correct"]
    int8_memory = max(
        by_method["int8_confidence_gate"]["max_observed_rss_mib"],
        by_method["int8_confidence_gate"]["max_peak_wset_mib"],
    )
    screen = {
        "within_one_accuracy_point_of_teacher": (teacher_correct - int8_correct) / 872 <= 0.01,
        "at_least_15_percent_median_reduction_vs_teacher": paired["int8_cascade_vs_teacher"][
            "median_percent"
        ]
        >= 15,
        "resident_memory_under_1536_mib": int8_memory < 1536,
        "faster_than_fp32_cascade_in_all_blocks": all(
            value > 0 for value in paired["int8_cascade_vs_fp32_cascade"]["by_block_percent"]
        ),
        "accuracy_no_worse_than_fp32_cascade": int8_correct
        >= by_method["fp32_confidence_gate"]["correct"],
    }
    summary = {
        "status": "exploratory: SST-2 validation and IMDb sample previously inspected",
        "plan_sha256": sha256(output / "plan.json"),
        "validation_audit_sha256": sha256(output / "validation-audit.json"),
        "external_audit_sha256": sha256(output / "external-audit.json"),
        "blocks": BLOCKS,
        "methods": by_method,
        "paired_reductions": paired,
        "prespecified_screen": screen,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({"event": "complete", "summary": summary}), flush=True)


def main():
    args = arguments()
    if args.check_only and (args.worker or args.resume or args.audit_only):
        raise ValueError("--check-only cannot be combined with worker, resume, or audit-only")
    if args.audit_only and args.worker:
        raise ValueError("--audit-only cannot be combined with worker")
    source_hashes = validate_sources()
    if args.check_only:
        print(
            json.dumps(
                {
                    "event": "inputs_verified",
                    "ort_version": ort.__version__,
                    "source_hashes": source_hashes,
                    "ac_connected": bool(
                        psutil.sensors_battery() and psutil.sensors_battery().power_plugged
                    ),
                }
            ),
            flush=True,
        )
        return
    if args.worker:
        if args.method is None or args.block not in range(BLOCKS):
            raise ValueError("Worker requires a known method and block")
        benchmark_worker(args)
        return
    if not args.audit_only:
        require_ac()
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise ValueError(f"Output already exists; use --resume or a new path: {output}")
    output.mkdir(parents=True, exist_ok=True)
    plan = make_candidate(output, source_hashes, args.resume, args.refresh_shape_metadata)
    audit = audit_validation(output, plan)
    audit_external(output, plan)
    if args.audit_only:
        print(json.dumps({"event": "audit_complete", "output": str(output)}), flush=True)
        return
    if (output / "summary.json").exists():
        if not args.resume:
            raise ValueError("Summary already exists")
        print(json.dumps({"event": "already_complete", "output": str(output)}), flush=True)
        return
    for block in range(BLOCKS):
        methods = list(METHODS)
        random.Random(2041 + block).shuffle(methods)
        for method in methods:
            report_path = output / f"{method}-block-{block}.json"
            if report_path.exists():
                if not args.resume:
                    raise ValueError(f"Existing timing report: {report_path}")
                report = read_json(report_path)
                if report["plan_sha256"] != sha256(output / "plan.json"):
                    raise ValueError("Existing timing report belongs to another plan")
                print(json.dumps({"event": "reused", "method": method, "block": block}), flush=True)
                continue
            require_ac()
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--worker",
                "--output",
                str(output),
                "--method",
                method,
                "--block",
                str(block),
            ]
            subprocess.run(command, check=True)
    summarize(output, audit)


if __name__ == "__main__":
    main()
