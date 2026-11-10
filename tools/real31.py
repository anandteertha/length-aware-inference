"""Frozen, targeted real-input confirmation and CPU backend diagnostic."""

import argparse
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from etdl.benchmark import latency_summary, sampled_memory
from etdl.boundaries import cpu_observations, power_state, same_power
from etdl.checks import compare_logits, verified_dynamic
from etdl.config import load_config
from etdl.data import DATASET_ID, encode, load_tokenizer, model_source
from etdl.followup import load_registry
from etdl.io import read_json, sha256, write_json
from etdl.runtime import session
from etdl.variants import pad_inputs

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "artifacts/real31"
SEED = 311027
BLOCKS = 5
REPEATS = 3
WARMUP = 2
SAMPLE_SIZE = 64


def config():
    return load_config(ROOT / "configs/default.toml")


def code_hashes():
    paths = [
        ROOT / "tools/real31.py",
        ROOT / "src/etdl/runtime.py",
        ROOT / "src/etdl/checks.py",
        ROOT / "src/etdl/variants.py",
        ROOT / "src/etdl/data.py",
    ]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def graph_paths(cfg):
    registry = load_registry(cfg, cfg.artifact_dir / "followup/registry-transformer.json")
    original = verified_dynamic(cfg)
    fixed = cfg.artifact_dir / "models/fixed-32.onnx"
    manifest = read_json(fixed.with_suffix(".json"))
    if manifest["base_sha256"] != sha256(original) or manifest["sha256"] != sha256(fixed):
        raise ValueError("Standard fixed-32 graph provenance changed")
    return {
        "ort_transformer": {
            name: registry["graphs"][name]["path"] for name in ("dynamic", "fixed-32")
        },
        "openvino_standard": {"dynamic": str(original), "fixed-32": str(fixed)},
    }


def prepare():
    from datasets import load_dataset

    cfg = config()
    tokenizer = load_tokenizer(cfg)
    calibration = read_json(cfg.artifact_dir / "data/calibration.json")
    used = {row["idx"] for row in calibration["rows"]}
    used.update(
        row["idx"] for row in read_json(cfg.artifact_dir / "confirmation/training.json")["rows"]
    )
    for role in ("fit", "confirmation", "audit"):
        used.update(
            row["idx"] for row in read_json(cfg.artifact_dir / f"followup/{role}.json")["rows"]
        )
    dataset = load_dataset(DATASET_ID, "sst2", split="train", revision=calibration["revision"])
    eligible = []
    for row in dataset:
        if (
            row["idx"] not in used
            and len(tokenizer(row["sentence"], truncation=False)["input_ids"]) == 31
        ):
            eligible.append(
                {
                    "idx": int(row["idx"]),
                    "text": row["sentence"],
                    "label": int(row["label"]),
                    "tokens": 31,
                }
            )
    if len(eligible) < SAMPLE_SIZE:
        raise ValueError("Too few fresh 31-token training rows")
    random.Random(SEED).shuffle(eligible)
    rows = eligible[:SAMPLE_SIZE]
    paths = graph_paths(cfg)
    plan = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "Targeted post-selection real-input confirmation; secondary OpenVINO diagnostic",
        "split": "train",
        "dataset_revision": calibration["revision"],
        "dataset_fingerprint": dataset._fingerprint,
        "model_source": model_source(cfg),
        "used_training_ids": len(used),
        "eligible_exact31": len(eligible),
        "selected_ids": [row["idx"] for row in rows],
        "sample_size": SAMPLE_SIZE,
        "seed": SEED,
        "blocks": BLOCKS,
        "repeats": REPEATS,
        "warmup_per_path": WARMUP,
        "threads": 4,
        "atol": cfg.atol,
        "rtol": cfg.rtol,
        "power": power_state(),
        "paths": paths,
        "graph_sha256": {
            backend: {name: sha256(Path(path)) for name, path in graphs.items()}
            for backend, graphs in paths.items()
        },
        "code_sha256": code_hashes(),
    }
    write_json(DIRECTORY / "workload.json", {"split": "train", "rows": rows})
    plan["workload_sha256"] = sha256(DIRECTORY / "workload.json")
    write_json(DIRECTORY / "plan.json", plan)
    print(
        f"Frozen {len(rows)} of {len(eligible)} eligible 31-token rows; excluded {len(used)} IDs",
        flush=True,
    )


def checked():
    cfg = config()
    plan = read_json(DIRECTORY / "plan.json")
    if plan["model_source"] != model_source(cfg) or plan["code_sha256"] != code_hashes():
        raise ValueError("Model or code changed since plan freeze")
    if plan["workload_sha256"] != sha256(DIRECTORY / "workload.json"):
        raise ValueError("Frozen workload changed")
    for backend, graphs in plan["paths"].items():
        for name, path in graphs.items():
            if sha256(Path(path)) != plan["graph_sha256"][backend][name]:
                raise ValueError("Frozen graph changed")
    return cfg, plan, read_json(DIRECTORY / "workload.json")["rows"]


def references():
    from etdl.comparison import references as generate

    checked()
    generate(config(), DIRECTORY / "workload.json", DIRECTORY / "references.json")
    print("Generated separate PyTorch FP32 references", flush=True)


def openvino_sessions(graphs):
    import openvino as ov
    import openvino.properties as props
    from openvino.properties import hint as hints

    core = ov.Core()
    settings = {
        hints.inference_precision: ov.Type.f32,
        hints.performance_mode: hints.PerformanceMode.LATENCY,
        props.inference_num_threads: 4,
        props.num_streams: 1,
    }
    sessions, startup, settings_readback = {}, {}, {}
    for name in ("dynamic", "fixed-32"):
        start = time.perf_counter_ns()
        model = core.read_model(graphs[name])
        sessions[name] = core.compile_model(model, "CPU", settings)
        startup[name] = (time.perf_counter_ns() - start) / 1e6
        settings_readback[name] = {
            "inference_precision": str(sessions[name].get_property(hints.inference_precision)),
            "inference_num_threads": sessions[name].get_property(props.inference_num_threads),
            "num_streams": sessions[name].get_property(props.num_streams),
        }
        if (
            settings_readback[name]["inference_num_threads"] != 4
            or settings_readback[name]["num_streams"] != 1
        ):
            raise ValueError("OpenVINO did not honor thread or stream settings")
        precision = settings_readback[name]["inference_precision"].lower()
        if "f32" not in precision and "float32" not in precision:
            raise ValueError("OpenVINO did not report FP32")
    return sessions, startup, settings_readback


def worker(backend, block):
    import importlib.metadata

    cfg, plan, rows = checked()
    reference_path = DIRECTORY / "references.json"
    refs = read_json(reference_path)
    if (
        refs["workload_sha256"] != plan["workload_sha256"]
        or refs["model_source"] != plan["model_source"]
    ):
        raise ValueError("References differ from frozen workload")
    if not same_power(plan["power"], power_state()):
        raise ValueError("Power state differs from frozen plan")
    tokenizer = load_tokenizer(cfg)
    inputs = {row["idx"]: encode(tokenizer, row["text"], cfg.max_length) for row in rows}
    if any(feed["input_ids"].shape != (1, 31) for feed in inputs.values()):
        raise ValueError("A frozen input is not exactly 31 tokens")
    jobs = [(row["idx"], mode) for row in rows for mode in ("natural", "padded", "fixed")]
    before = power_state()
    with sampled_memory() as memory, cpu_observations() as telemetry:
        if backend == "ort_transformer":
            sessions = {}
            startup = {}
            for name, path in plan["paths"][backend].items():
                sessions[name], startup[name] = session(path, 4, allow_spinning=False)
            settings_readback = {
                "provider": "CPUExecutionProvider",
                "threads": 4,
                "spinning": False,
            }

            def infer(name, feed):
                return sessions[name].run(["logits"], feed)[0]

            version = importlib.metadata.version("onnxruntime")
        elif backend == "openvino_standard":
            sessions, startup, settings_readback = openvino_sessions(plan["paths"][backend])

            def infer(name, feed):
                return sessions[name](feed)[sessions[name].output(0)]

            version = importlib.metadata.version("openvino")
        else:
            raise ValueError("Unknown backend")

        def execute(idx, mode, repeat):
            feed = inputs[idx]
            start = time.perf_counter_ns()
            if mode != "natural":
                feed = pad_inputs(feed, 32, tokenizer.pad_token_id)
            prepared = time.perf_counter_ns()
            actual = infer("fixed-32" if mode == "fixed" else "dynamic", feed)
            end = time.perf_counter_ns()
            check = compare_logits(refs["logits"][str(idx)], actual, cfg.atol, cfg.rtol)
            if not check["passed"]:
                raise ValueError(
                    f"Numerical check failed: {backend} block {block} row {idx} mode {mode}: {check}"
                )
            return {
                "idx": idx,
                "mode": mode,
                "repeat": repeat,
                "inference_ms": (end - prepared) / 1e6,
                "prepared_input_ms": (end - start) / 1e6,
                **check,
            }

        for mode in ("natural", "padded", "fixed"):
            for _ in range(WARMUP):
                execute(rows[0]["idx"], mode, -1)
        samples = []
        for repeat in range(REPEATS):
            shuffled = list(jobs)
            random.Random(SEED + block * 100 + repeat).shuffle(shuffled)
            samples.extend(execute(idx, mode, repeat) for idx, mode in shuffled)
    after = power_state()
    groups = {
        mode: {
            field: latency_summary([sample[field] for sample in samples if sample["mode"] == mode])
            for field in ("inference_ms", "prepared_input_ms")
        }
        for mode in ("natural", "padded", "fixed")
    }
    report = {
        "backend": backend,
        "backend_version": version,
        "block": block,
        "samples": samples,
        "groups": groups,
        "session_initialization_ms": startup,
        "settings_readback": settings_readback,
        "memory": memory,
        "power_before": before,
        "power_after": after,
        "power_consistent": same_power(plan["power"], before) and same_power(plan["power"], after),
        "cpu_observations": telemetry,
        "plan_sha256": sha256(DIRECTORY / "plan.json"),
        "references_sha256": sha256(reference_path),
    }
    write_json(DIRECTORY / f"{backend}-block-{block}.json", report)
    print(
        f"{backend} block {block}: natural/fixed p95 {groups['natural']['inference_ms']['p95_ms'] / groups['fixed']['inference_ms']['p95_ms']:.3f}",
        flush=True,
    )


def run():
    checked()
    if not (DIRECTORY / "references.json").exists():
        raise ValueError("Generate references before timing")
    schedule = []
    for block in range(BLOCKS):
        backends = ["ort_transformer", "openvino_standard"]
        random.Random(SEED + block).shuffle(backends)
        schedule.extend((backend, block) for backend in backends)
    write_json(DIRECTORY / "schedule.json", schedule)
    for backend, block in schedule:
        output = DIRECTORY / f"{backend}-block-{block}.json"
        if output.exists():
            continue
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "worker", backend, str(block)],
            check=True,
        )
        time.sleep(2)
    summarize()


def summarize():
    _, plan, _ = checked()
    summary = {
        "plan_sha256": sha256(DIRECTORY / "plan.json"),
        "eligible_exact31": plan["eligible_exact31"],
        "selected": SAMPLE_SIZE,
        "backends": {},
    }
    for backend in ("ort_transformer", "openvino_standard"):
        reports = [
            read_json(DIRECTORY / f"{backend}-block-{block}.json") for block in range(BLOCKS)
        ]
        contrasts = {}
        for numerator, denominator in (
            ("natural", "fixed"),
            ("padded", "fixed"),
            ("padded", "natural"),
        ):
            ratios = [
                r["groups"][numerator]["inference_ms"]["p95_ms"]
                / r["groups"][denominator]["inference_ms"]["p95_ms"]
                for r in reports
            ]
            contrasts[f"{numerator}/{denominator}"] = {
                "block_ratios": ratios,
                "median": float(np.median(ratios)),
                "range": [min(ratios), max(ratios)],
            }
        summary["backends"][backend] = {
            "contrasts": contrasts,
            "p95_ms_by_block": {
                mode: [r["groups"][mode]["inference_ms"]["p95_ms"] for r in reports]
                for mode in ("natural", "padded", "fixed")
            },
            "all_checks_passed": all(s["passed"] for r in reports for s in r["samples"]),
            "changed_labels": sum(not s["labels_match"] for r in reports for s in r["samples"]),
            "power_consistent": all(r["power_consistent"] for r in reports),
            "max_telemetry_gap_s": max(
                (
                    b["elapsed_s"] - a["elapsed_s"]
                    for r in reports
                    for a, b in zip(r["cpu_observations"], r["cpu_observations"][1:])
                ),
                default=0,
            ),
            "peak_rss_mib_by_block": [
                r["memory"]["sampled_peak_rss_bytes"] / 2**20 for r in reports
            ],
            "backend_version": reports[0]["backend_version"],
            "settings_readback": reports[0]["settings_readback"],
            "timed_requests": sum(len(r["samples"]) for r in reports),
        }
    target = summary["backends"]["ort_transformer"]["contrasts"]["natural/fixed"]
    summary["predeclared_primary_threshold_met"] = (
        summary["backends"]["ort_transformer"]["all_checks_passed"]
        and all(ratio > 1 for ratio in target["block_ratios"])
        and target["median"] > 1.05
    )
    write_json(DIRECTORY / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "references", "worker", "run", "summarize"))
    parser.add_argument("backend", nargs="?")
    parser.add_argument("block", nargs="?", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "references":
        references()
    elif args.command == "worker":
        worker(args.backend, args.block)
    elif args.command == "run":
        run()
    else:
        summarize()
