"""Real-text, capacity-specific paired map of padding and static-shape effects."""

import argparse
import importlib.metadata
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from etdl.benchmark import latency_summary, sampled_memory
from etdl.boundaries import cpu_observations, power_state, same_power
from etdl.checks import compare_logits, verified_dynamic
from etdl.config import load_config
from etdl.data import DATASET_ID, encode, load_tokenizer, model_source
from etdl.io import read_json, sha256, write_json
from etdl.runtime import session
from etdl.variants import pad_inputs

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "artifacts/breakeven-map"
LENGTHS = {16: (9, 15, 16), 32: (17, 24, 30, 31, 32), 64: (33, 40, 48)}
SEED = 20261112
PER_LENGTH = 24
BLOCKS = 3
REPEATS = 2
WARMUP = 3
BACKENDS = ("ort", "openvino")


def config():
    return load_config(ROOT / "configs/default.toml")


def code_hashes():
    paths = [
        ROOT / "tools/breakeven_map.py",
        ROOT / "src/etdl/runtime.py",
        ROOT / "src/etdl/checks.py",
        ROOT / "src/etdl/data.py",
        ROOT / "src/etdl/variants.py",
        ROOT / "src/etdl/boundaries.py",
        ROOT / "src/etdl/comparison.py",
    ]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def graph_paths(cfg):
    dynamic = verified_dynamic(cfg)
    paths = {"dynamic": str(dynamic)}
    for capacity in LENGTHS:
        fixed = cfg.artifact_dir / f"models/fixed-{capacity}.onnx"
        manifest = read_json(fixed.with_suffix(".json"))
        if manifest["base_sha256"] != sha256(dynamic) or manifest["sha256"] != sha256(fixed):
            raise ValueError(f"Fixed-{capacity} graph provenance changed")
        paths[f"fixed-{capacity}"] = str(fixed)
    return paths


def exclusions(cfg):
    artifact = cfg.artifact_dir
    used = {row["idx"] for row in read_json(artifact / "data/calibration.json")["rows"]}
    used.update(row["idx"] for row in read_json(artifact / "confirmation/training.json")["rows"])
    for role in ("fit", "confirmation", "audit"):
        used.update(row["idx"] for row in read_json(artifact / f"followup/{role}.json")["rows"])
    used.update(row["idx"] for row in read_json(artifact / "real31/workload.json")["rows"])
    return used


def prepare():
    from datasets import load_dataset

    if (DIRECTORY / "plan.json").exists():
        raise ValueError("Plan already exists; preserve the frozen run")
    cfg = config()
    tokenizer = load_tokenizer(cfg)
    calibration = read_json(cfg.artifact_dir / "data/calibration.json")
    used = exclusions(cfg)
    wanted = {length for group in LENGTHS.values() for length in group}
    dataset = load_dataset(DATASET_ID, "sst2", split="train", revision=calibration["revision"])
    eligible = defaultdict(list)
    for start in range(0, len(dataset), 1024):
        chunk = dataset.select(range(start, min(start + 1024, len(dataset))))
        tokenized = tokenizer(list(chunk["sentence"]), truncation=False)["input_ids"]
        for row, ids in zip(chunk, tokenized, strict=True):
            length = len(ids)
            if row["idx"] not in used and length in wanted:
                eligible[length].append(
                    {
                        "idx": int(row["idx"]),
                        "text": row["sentence"],
                        "label": int(row["label"]),
                        "tokens": length,
                    }
                )
    rows = []
    for length in sorted(wanted):
        if len(eligible[length]) < PER_LENGTH:
            raise ValueError(f"Only {len(eligible[length])} unused rows at length {length}")
        random.Random(SEED + length).shuffle(eligible[length])
        rows.extend(eligible[length][:PER_LENGTH])
    paths = graph_paths(cfg)
    write_json(DIRECTORY / "workload.json", {"split": "train", "rows": rows})
    plan = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "Exploratory real-text padding versus shape-gain map",
        "split": "train",
        "dataset_revision": calibration["revision"],
        "dataset_fingerprint": dataset._fingerprint,
        "model_source": model_source(cfg),
        "excluded_training_ids": len(used),
        "eligible_counts": {str(length): len(eligible[length]) for length in sorted(wanted)},
        "selected_ids_by_length": {
            str(length): [row["idx"] for row in rows if row["tokens"] == length]
            for length in sorted(wanted)
        },
        "lengths_by_capacity": {str(cap): list(lengths) for cap, lengths in LENGTHS.items()},
        "seed": SEED,
        "per_length": PER_LENGTH,
        "blocks": BLOCKS,
        "repeats": REPEATS,
        "warmup_per_path": WARMUP,
        "threads": 4,
        "atol": cfg.atol,
        "rtol": cfg.rtol,
        "power": power_state(),
        "paths": paths,
        "graph_sha256": {name: sha256(Path(path)) for name, path in paths.items()},
        "code_sha256": code_hashes(),
        "workload_sha256": sha256(DIRECTORY / "workload.json"),
    }
    write_json(DIRECTORY / "plan.json", plan)
    print(
        f"Frozen {len(rows)} real rows across {len(wanted)} exact lengths; excluded {len(used)} IDs",
        flush=True,
    )


def checked():
    cfg = config()
    plan = read_json(DIRECTORY / "plan.json")
    if plan["model_source"] != model_source(cfg) or plan["code_sha256"] != code_hashes():
        raise ValueError("Model or code changed after map freeze")
    if plan["workload_sha256"] != sha256(DIRECTORY / "workload.json"):
        raise ValueError("Frozen map workload changed")
    for name, path in plan["paths"].items():
        if sha256(Path(path)) != plan["graph_sha256"][name]:
            raise ValueError(f"Frozen graph changed: {name}")
    return cfg, plan, read_json(DIRECTORY / "workload.json")["rows"]


def references():
    from etdl.comparison import references as generate

    checked()
    generate(config(), DIRECTORY / "workload.json", DIRECTORY / "references.json")
    print("Generated separate PyTorch FP32 references", flush=True)


def openvino_sessions(paths):
    import openvino as ov
    import openvino.properties as props
    from openvino.properties import hint as hints

    settings = {
        hints.inference_precision: ov.Type.f32,
        hints.performance_mode: hints.PerformanceMode.LATENCY,
        props.inference_num_threads: 4,
        props.num_streams: 1,
    }
    core = ov.Core()
    runtimes, startup, readback = {}, {}, {}
    for name, path in paths.items():
        start = time.perf_counter_ns()
        runtimes[name] = core.compile_model(core.read_model(path), "CPU", settings)
        startup[name] = (time.perf_counter_ns() - start) / 1e6
        compiled = runtimes[name]
        properties = {
            "inference_precision": str(compiled.get_property(hints.inference_precision)),
            "inference_num_threads": compiled.get_property(props.inference_num_threads),
            "num_streams": compiled.get_property(props.num_streams),
        }
        if properties["inference_num_threads"] != 4 or properties["num_streams"] != 1:
            raise ValueError("OpenVINO thread/stream readback differs")
        precision = properties["inference_precision"].lower()
        if "f32" not in precision and "float32" not in precision:
            raise ValueError("OpenVINO did not report FP32")
        for field in ("scheduling_core_type", "enable_cpu_pinning", "enable_hyper_threading"):
            try:
                properties[field] = str(compiled.get_property(getattr(hints, field)))
            except (AttributeError, RuntimeError):
                properties[field] = None
        readback[name] = properties
    return runtimes, startup, readback


def worker(backend, capacity, block):
    cfg, plan, rows = checked()
    if backend not in BACKENDS or capacity not in LENGTHS or block not in range(BLOCKS):
        raise ValueError("Worker condition is outside the frozen map")
    refs_path = DIRECTORY / "references.json"
    refs = read_json(refs_path)
    if (
        refs["workload_sha256"] != plan["workload_sha256"]
        or refs["model_source"] != plan["model_source"]
    ):
        raise ValueError("Map references differ from frozen workload")
    if not same_power(plan["power"], power_state()):
        raise ValueError("Power state changed since map freeze")
    tokenizer = load_tokenizer(cfg)
    selected = [row for row in rows if row["tokens"] in LENGTHS[capacity]]
    feeds = {row["idx"]: encode(tokenizer, row["text"], cfg.max_length) for row in selected}
    if any(feeds[row["idx"]]["input_ids"].shape != (1, row["tokens"]) for row in selected):
        raise ValueError("Frozen text token length changed")
    jobs = [
        (row["idx"], row["tokens"], mode)
        for row in selected
        for mode in ("natural", "padded", "fixed")
    ]
    graph_files = {name: plan["paths"][name] for name in ("dynamic", f"fixed-{capacity}")}
    before = power_state()
    with sampled_memory() as memory, cpu_observations() as telemetry:
        if backend == "ort":
            runtimes, startup = {}, {}
            for name, path in graph_files.items():
                runtimes[name], startup[name] = session(path, 4, allow_spinning=False)
            readback = {"provider": "CPUExecutionProvider", "threads": 4, "spinning": False}
            version = importlib.metadata.version("onnxruntime")

            def infer(name, feed):
                return runtimes[name].run(["logits"], feed)[0]

        else:
            runtimes, startup, readback = openvino_sessions(graph_files)
            version = importlib.metadata.version("openvino")

            def infer(name, feed):
                return runtimes[name](feed)[runtimes[name].output(0)]

        def execute(idx, length, mode, repeat):
            feed = feeds[idx]
            start = time.perf_counter_ns()
            if mode != "natural":
                feed = pad_inputs(feed, capacity, tokenizer.pad_token_id)
            prepared = time.perf_counter_ns()
            logits = infer(f"fixed-{capacity}" if mode == "fixed" else "dynamic", feed)
            end = time.perf_counter_ns()
            check = compare_logits(refs["logits"][str(idx)], logits, cfg.atol, cfg.rtol)
            if not check["passed"]:
                write_json(
                    DIRECTORY / f"{backend}-cap{capacity}-block{block}-failure.json",
                    {"idx": idx, "length": length, "mode": mode, "repeat": repeat, "check": check},
                )
                raise ValueError(
                    f"Numerical check failed: {backend}, cap {capacity}, block {block}, "
                    f"row {idx}, mode {mode}: {check}"
                )
            return {
                "idx": idx,
                "length": length,
                "capacity": capacity,
                "mode": mode,
                "repeat": repeat,
                "input_preparation_ms": (prepared - start) / 1e6,
                "inference_ms": (end - prepared) / 1e6,
                "prepared_input_ms": (end - start) / 1e6,
                **check,
            }

        for length in LENGTHS[capacity]:
            idx = next(row["idx"] for row in selected if row["tokens"] == length)
            for mode in ("natural", "padded", "fixed"):
                for _ in range(WARMUP):
                    execute(idx, length, mode, -1)
        samples = []
        for repeat in range(REPEATS):
            shuffled = list(jobs)
            random.Random(SEED + block * 1000 + capacity * 10 + repeat).shuffle(shuffled)
            samples.extend(execute(idx, length, mode, repeat) for idx, length, mode in shuffled)
    after = power_state()
    groups = {
        str(length): {
            mode: {
                field: latency_summary(
                    [
                        sample[field]
                        for sample in samples
                        if sample["length"] == length and sample["mode"] == mode
                    ]
                )
                for field in ("input_preparation_ms", "inference_ms", "prepared_input_ms")
            }
            for mode in ("natural", "padded", "fixed")
        }
        for length in LENGTHS[capacity]
    }
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "backend": backend,
        "backend_version": version,
        "capacity": capacity,
        "block": block,
        "samples": samples,
        "groups": groups,
        "session_initialization_ms": startup,
        "settings_readback": readback,
        "memory": memory,
        "power_before": before,
        "power_after": after,
        "power_consistent": same_power(plan["power"], before) and same_power(plan["power"], after),
        "cpu_observations": telemetry,
        "plan_sha256": sha256(DIRECTORY / "plan.json"),
        "references_sha256": sha256(refs_path),
    }
    write_json(DIRECTORY / f"{backend}-cap{capacity}-block{block}.json", report)
    print(f"Finished {backend} capacity {capacity} block {block + 1}", flush=True)


def schedule():
    jobs = []
    for block in range(BLOCKS):
        conditions = [(backend, capacity, block) for backend in BACKENDS for capacity in LENGTHS]
        random.Random(SEED + block).shuffle(conditions)
        jobs.extend(conditions)
    return jobs


def run():
    checked()
    if not (DIRECTORY / "references.json").exists():
        raise ValueError("PyTorch references must exist before timing")
    jobs = schedule()
    write_json(DIRECTORY / "schedule.json", jobs)
    for backend, capacity, block in jobs:
        output = DIRECTORY / f"{backend}-cap{capacity}-block{block}.json"
        if output.exists():
            continue
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                backend,
                str(capacity),
                str(block),
            ],
            check=True,
        )
        time.sleep(2)
    summarize()


def summarize():
    checked()
    rows = []
    for backend in BACKENDS:
        for capacity, lengths in LENGTHS.items():
            reports = [
                read_json(DIRECTORY / f"{backend}-cap{capacity}-block{block}.json")
                for block in range(BLOCKS)
            ]
            for report in reports:
                if report["plan_sha256"] != sha256(DIRECTORY / "plan.json"):
                    raise ValueError("A process used a different map plan")
            for length in lengths:
                blocks = []
                for report in reports:
                    group = report["groups"][str(length)]
                    natural = group["natural"]["prepared_input_ms"]["p95_ms"]
                    padded = group["padded"]["prepared_input_ms"]["p95_ms"]
                    fixed = group["fixed"]["prepared_input_ms"]["p95_ms"]
                    blocks.append(
                        {
                            "block": report["block"],
                            "natural_p95_ms": natural,
                            "padded_p95_ms": padded,
                            "fixed_p95_ms": fixed,
                            "padding_cost_ms": padded - natural,
                            "shape_gain_ms": padded - fixed,
                            "net_fixed_gain_ms": natural - fixed,
                            "natural_over_fixed_ratio": natural / fixed,
                            "fixed_reduction_fraction": (natural - fixed) / natural,
                        }
                    )
                fields = (
                    "natural_p95_ms",
                    "padded_p95_ms",
                    "fixed_p95_ms",
                    "padding_cost_ms",
                    "shape_gain_ms",
                    "net_fixed_gain_ms",
                    "natural_over_fixed_ratio",
                    "fixed_reduction_fraction",
                )
                quality = all(
                    report["power_consistent"]
                    and report["memory"]["sampled_peak_rss_bytes"] <= 1536 * 2**20
                    and all(sample["passed"] for sample in report["samples"])
                    for report in reports
                )
                rows.append(
                    {
                        "backend": backend,
                        "capacity": capacity,
                        "length": length,
                        "examples": PER_LENGTH,
                        "timed_requests_per_path_per_block": PER_LENGTH * REPEATS,
                        "blocks": blocks,
                        "median": {
                            field: float(np.median([item[field] for item in blocks]))
                            for field in fields
                        },
                        "range": {
                            field: [
                                min(item[field] for item in blocks),
                                max(item[field] for item in blocks),
                            ]
                            for field in fields
                        },
                        "candidate_for_future_confirmation": quality
                        and all(item["net_fixed_gain_ms"] > 0 for item in blocks)
                        and float(np.median([item["fixed_reduction_fraction"] for item in blocks]))
                        >= 0.05,
                    }
                )
    process_reports = [
        read_json(DIRECTORY / f"{backend}-cap{capacity}-block{block}.json")
        for backend in BACKENDS
        for capacity in LENGTHS
        for block in range(BLOCKS)
    ]
    summary = {
        "plan_sha256": sha256(DIRECTORY / "plan.json"),
        "references_sha256": sha256(DIRECTORY / "references.json"),
        "lengths": [length for group in LENGTHS.values() for length in group],
        "backends": list(BACKENDS),
        "processes": len(process_reports),
        "timed_requests": sum(len(report["samples"]) for report in process_reports),
        "all_checks_passed": all(
            sample["passed"] for report in process_reports for sample in report["samples"]
        ),
        "changed_labels": sum(
            not sample["labels_match"] for report in process_reports for sample in report["samples"]
        ),
        "power_consistent": all(report["power_consistent"] for report in process_reports),
        "max_telemetry_gap_s": max(
            (
                b["elapsed_s"] - a["elapsed_s"]
                for report in process_reports
                for a, b in zip(report["cpu_observations"], report["cpu_observations"][1:])
            ),
            default=0,
        ),
        "max_peak_rss_mib": max(
            report["memory"]["sampled_peak_rss_bytes"] / 2**20 for report in process_reports
        ),
        "map": rows,
    }
    write_json(DIRECTORY / "summary.json", summary)
    print(
        f"Completed {summary['processes']} processes, {summary['timed_requests']} requests; "
        f"candidate cells: {[(row['backend'], row['length']) for row in rows if row['candidate_for_future_confirmation']]}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "references", "worker", "run", "summarize"))
    parser.add_argument("backend", nargs="?")
    parser.add_argument("capacity", nargs="?", type=int)
    parser.add_argument("block", nargs="?", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "references":
        references()
    elif args.command == "worker":
        worker(args.backend, args.capacity, args.block)
    elif args.command == "run":
        run()
    else:
        summarize()
