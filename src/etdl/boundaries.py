"""Matched synthetic length diagnostics; no policy tuning or accuracy evaluation."""

import random
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from etdl.benchmark import latency_summary, sampled_memory
from etdl.checks import compare_logits
from etdl.data import load_tokenizer, model_source
from etdl.followup import load_registry
from etdl.io import read_json, sha256, write_json
from etdl.runtime import environment, session
from etdl.study import hardware_key
from etdl.variants import LENGTHS, pad_inputs

TEST_LENGTHS = (9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129)
PATTERNS = {
    "positive": ("good", "great", "love"),
    "negative": ("bad", "terrible", "hate"),
    "mixed": ("story", "boring", "fun"),
}


def bucket_capacity(length):
    if length < 2 or length > 512:
        raise ValueError("Synthetic length must include special tokens and fit the encoder")
    return next((capacity for capacity in LENGTHS if length <= capacity), None)


def synthetic_cases(tokenizer):
    cases = []
    for pattern, words in PATTERNS.items():
        tokens = [tokenizer.convert_tokens_to_ids(word) for word in words]
        if tokenizer.unk_token_id in tokens:
            raise ValueError("Synthetic vocabulary contains unknown tokens")
        for length in TEST_LENGTHS:
            ids = [tokenizer.cls_token_id]
            ids += [tokens[index % len(tokens)] for index in range(length - 2)]
            ids += [tokenizer.sep_token_id]
            cases.append(
                {
                    "id": f"{pattern}-{length}",
                    "pattern": pattern,
                    "length": length,
                    "input_ids": [ids],
                    "attention_mask": [[1] * length],
                }
            )
    return cases


def power_state():
    """Read power conditions without modifying the user's system settings."""
    import psutil

    battery = psutil.sensors_battery()
    plugged = battery.power_plugged if battery is not None else None
    scheme = None
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Status(ctypes.Structure):
            _fields_ = [
                ("ac", wintypes.BYTE),
                ("flag", wintypes.BYTE),
                ("percent", wintypes.BYTE),
                ("reserved", wintypes.BYTE),
                ("remaining", wintypes.DWORD),
                ("full", wintypes.DWORD),
            ]

        status = Status()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            plugged = {0: False, 1: True}.get(status.ac)
        result = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, check=True)
        match = re.search(rb"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", result.stdout)
        scheme = match.group().decode().lower() if match else None
    return {"ac_connected": plugged, "power_scheme_guid": scheme}


def same_power(expected, actual):
    return all(expected[key] == actual[key] for key in ("ac_connected", "power_scheme_guid"))


@contextmanager
def cpu_observations():
    """System load includes this worker; it is not a background-load estimate."""
    import psutil

    process = psutil.Process()
    stop, samples = threading.Event(), []

    def observe():
        psutil.cpu_percent()
        process.cpu_percent()
        while not stop.wait(1):
            samples.append(
                {
                    "elapsed_s": time.perf_counter() - started,
                    "system_cpu_percent": psutil.cpu_percent(),
                    "worker_cpu_percent": process.cpu_percent(),
                }
            )

    started = time.perf_counter()
    worker = threading.Thread(target=observe, daemon=True)
    worker.start()
    try:
        yield samples
    finally:
        stop.set()
        worker.join()


def code_hashes():
    root = Path(__file__).parent
    return {
        name: sha256(root / name)
        for name in (
            "boundaries.py",
            "benchmark.py",
            "runtime.py",
            "checks.py",
            "variants.py",
            "data.py",
        )
    }


def checked_plan(config, path):
    plan = read_json(path)
    if plan["code_hashes"] != code_hashes() or plan["model_source"] != model_source(config):
        raise ValueError("Diagnostic code or checkpoint changed after planning")
    if plan["atol"] != config.atol or plan["rtol"] != config.rtol:
        raise ValueError("Diagnostic score tolerances changed")
    if plan["max_length"] != config.max_length:
        raise ValueError("Diagnostic encoder limit changed")
    return plan


def reference(config, plan_path, output):
    import torch

    from etdl.model import load_pytorch

    plan = checked_plan(config, plan_path)
    model = load_pytorch(config, threads=1)
    logits = {}
    with torch.inference_mode():
        for case in plan["cases"]:
            feed = {
                key: torch.tensor(case[key], dtype=torch.long)
                for key in ("input_ids", "attention_mask")
            }
            logits[case["id"]] = model(**feed).logits.numpy().tolist()
    write_json(
        output,
        {
            "plan_sha256": sha256(plan_path),
            "model_source": plan["model_source"],
            "purpose": "Synthetic numerical references; no task labels or accuracy.",
            "logits": logits,
        },
    )


def run_worker(config, request_path, output):
    request = read_json(request_path)
    plan_path, refs_path = Path(request["plan"]), Path(request["references"])
    if (
        sha256(plan_path) != request["plan_sha256"]
        or sha256(refs_path) != request["references_sha256"]
    ):
        raise ValueError("Diagnostic request inputs changed")
    plan = checked_plan(config, plan_path)
    refs = read_json(refs_path)
    if refs["plan_sha256"] != sha256(plan_path) or refs["model_source"] != plan["model_source"]:
        raise ValueError("Diagnostic reference provenance changed")
    family, capacity, block = request["family"], request["capacity"], request["block"]
    if capacity not in LENGTHS or not 0 <= block < plan["blocks"]:
        raise ValueError("Diagnostic condition differs from the plan")
    graphs = plan["families"][family]["graphs"]
    names = ("dynamic", f"fixed-{capacity}")
    for name in names:
        if (
            not graphs[name]["passed"]
            or sha256(Path(graphs[name]["path"])) != graphs[name]["sha256"]
        ):
            raise ValueError("Diagnostic model is unverified or changed")
    before = power_state()
    if not same_power(plan["power"], before):
        raise ValueError("Power conditions changed; preserve this plan and use a new run name")
    import psutil

    load_before = psutil.cpu_percent(interval=0.2)
    tokenizer = load_tokenizer(config)
    selected = [
        case
        for case in plan["cases"]
        if bucket_capacity(case["length"]) == capacity
        or (capacity == 128 and case["length"] == 129)
    ]
    jobs = [
        (case, mode)
        for case in selected
        for mode in (("natural",) if case["length"] == 129 else ("natural", "padded", "fixed"))
    ]
    inputs = {
        case["id"]: {
            key: np.asarray(case[key], dtype=np.int64) for key in ("input_ids", "attention_mask")
        }
        for case in selected
    }
    with sampled_memory() as memory, cpu_observations() as telemetry:
        runtimes, initialization = {}, {}
        for name in names:
            runtimes[name], initialization[name] = session(
                Path(graphs[name]["path"]), plan["threads"], allow_spinning=False
            )
            size = config.max_length if name == "dynamic" else capacity
            probe = np.full((1, size), tokenizer.convert_tokens_to_ids("good"), dtype=np.int64)
            probe[0, 0], probe[0, -1] = tokenizer.cls_token_id, tokenizer.sep_token_id
            runtimes[name].run(
                ["logits"], {"input_ids": probe, "attention_mask": np.ones_like(probe)}
            )

        def execute(case, mode, repeat):
            feed = inputs[case["id"]]
            start = time.perf_counter_ns()
            bucket = bucket_capacity(feed["input_ids"].shape[1])
            name = f"fixed-{bucket}" if mode == "fixed" else "dynamic"
            if mode != "natural":
                feed = pad_inputs(feed, bucket, tokenizer.pad_token_id)
            prepared = time.perf_counter_ns()
            actual = runtimes[name].run(["logits"], feed)[0]
            finished = time.perf_counter_ns()
            check = compare_logits(refs["logits"][case["id"]], actual, config.atol, config.rtol)
            return {
                "case": case["id"],
                "pattern": case["pattern"],
                "length": case["length"],
                "executed_length": int(feed["input_ids"].shape[1]),
                "mode": mode,
                "repeat": repeat,
                "routing_padding_ms": (prepared - start) / 1e6,
                "inference_ms": (finished - prepared) / 1e6,
                "prepared_input_ms": (finished - start) / 1e6,
                **check,
            }

        for case, mode in jobs:
            for _ in range(plan["warmup"]):
                if not execute(case, mode, -1)["passed"]:
                    raise ValueError("Diagnostic warm-up numerical check failed")
        samples = []
        for repeat in range(plan["repeats"]):
            ordered = list(jobs)
            random.Random(plan["seed"] + capacity + block * 100 + repeat).shuffle(ordered)
            samples.extend(execute(case, mode, repeat) for case, mode in ordered)
    after = power_state()
    load_after = psutil.cpu_percent(interval=0.2)
    grouped = defaultdict(list)
    for sample in samples:
        grouped[(sample["length"], sample["mode"])].append(sample)
    groups = [
        {
            "length": length,
            "mode": mode,
            "latency": {
                field: latency_summary([sample[field] for sample in values])
                for field in ("inference_ms", "routing_padding_ms", "prepared_input_ms")
            },
        }
        for (length, mode), values in sorted(grouped.items())
    ]
    report = {
        "family": family,
        "capacity": capacity,
        "block": block,
        "threads": plan["threads"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "groups": groups,
        "samples": samples,
        "passed": all(sample["passed"] for sample in samples),
        "changed_predictions": sum(not sample["labels_match"] for sample in samples),
        "power_before": before,
        "power_after": after,
        "power_consistent": same_power(plan["power"], after),
        "system_cpu_before_percent": load_before,
        "system_cpu_after_percent": load_after,
        "cpu_observations": telemetry,
        "memory": memory,
        "session_initialization_ms": initialization,
        "plan_sha256": sha256(plan_path),
        "references_sha256": sha256(refs_path),
        "request_sha256": sha256(request_path),
        "code_hashes": plan["code_hashes"],
    }
    write_json(output, report)
    print(
        f"Finished block {block + 1}: {family} capacity {capacity}; checks={report['passed']}",
        flush=True,
    )


def summarize(runs):
    grouped = defaultdict(list)
    for run in runs:
        for group in run["groups"]:
            grouped[(run["family"], group["length"], group["mode"])].append((run["block"], group))
    groups = []
    for (family, length, mode), entries in sorted(grouped.items()):
        groups.append(
            {
                "family": family,
                "length": length,
                "mode": mode,
                "blocks": [block for block, _ in entries],
                "process_runs": len(entries),
                "latency": {
                    field: {
                        "median_run_p95_ms": float(np.median(values)),
                        "range_run_p95_ms": [min(values), max(values)],
                        "run_p95_ms": values,
                    }
                    for field in ("inference_ms", "routing_padding_ms", "prepared_input_ms")
                    for values in [[group["latency"][field]["p95_ms"] for _, group in entries]]
                },
            }
        )
    ratios = []
    for family in sorted({run["family"] for run in runs}):
        for length in TEST_LENGTHS:
            for numerator, denominator, meaning in (
                ("padded", "natural", "padding_cost"),
                ("padded", "fixed", "fixed_shape_effect"),
                ("natural", "fixed", "net_fixed_effect"),
            ):
                values = []
                for run in runs:
                    if run["family"] != family:
                        continue
                    by = {
                        g["mode"]: g["latency"]["prepared_input_ms"]["p95_ms"]
                        for g in run["groups"]
                        if g["length"] == length
                    }
                    if numerator in by and denominator in by:
                        values.append(
                            {"block": run["block"], "ratio": by[numerator] / by[denominator]}
                        )
                if values:
                    ratios.append(
                        {
                            "family": family,
                            "length": length,
                            "contrast": meaning,
                            "numerator": numerator,
                            "denominator": denominator,
                            "paired_blocks": values,
                            "median_ratio": float(np.median([v["ratio"] for v in values])),
                        }
                    )
    return {
        "status": "complete",
        "groups": groups,
        "paired_ratios": ratios,
        "processes": len(runs),
        "timed_requests": sum(len(run["samples"]) for run in runs),
        "all_checks_passed": all(run["passed"] for run in runs),
        "changed_predictions": sum(run["changed_predictions"] for run in runs),
        "all_power_conditions_consistent": all(run["power_consistent"] for run in runs),
        "limitations": [
            "Synthetic pretokenized inputs; no accuracy or natural-workload p95 claims.",
            "Five process blocks on one laptop; descriptive ranges, no confidence intervals.",
            "Two resident sessions per process; not dynamic-only deployment memory.",
            "CPU telemetry includes this worker, with no temperature or power-watt measurements.",
            "No original validation access or deployment-policy changes.",
        ],
    }


def execute(config, config_path, directory, followup, resume=False, blocks=5, repeats=8, warmup=2):
    if blocks < 5 or repeats < 1 or warmup < 1:
        raise ValueError("Use at least five blocks, positive repeats and positive per-case warm-up")
    if config.max_length < max(TEST_LENGTHS):
        raise ValueError("Encoder cannot accommodate diagnostic lengths")
    plan_path = directory / "plan.json"
    if plan_path.exists() and not resume:
        raise ValueError("Diagnostic exists; resume its frozen plan or use a new name")
    settings = {
        "blocks": blocks,
        "repeats": repeats,
        "warmup": warmup,
        "threads": 4,
        "seed": 2041,
        "cooldown_seconds": 2,
    }
    if plan_path.exists():
        plan = checked_plan(config, plan_path)
        if hardware_key(environment()) != hardware_key(plan["environment"]):
            raise ValueError("Diagnostic hardware/runtime changed")
        if any(plan[key] != value for key, value in settings.items()):
            raise ValueError("Diagnostic resume settings differ")
    else:
        registries = {
            family: load_registry(config, followup / f"registry-{family}.json")
            for family in ("standard", "transformer")
        }
        if any(
            not r["available"] or not all(g["passed"] for g in r["graphs"].values())
            for r in registries.values()
        ):
            raise ValueError("Both complete verified graph families are required")
        machine = environment()
        if any(hardware_key(machine) != r["hardware_key"] for r in registries.values()):
            raise ValueError("Diagnostic hardware/runtime differs from graph verification")
        plan = {
            **settings,
            "stage": "frozen_synthetic_diagnostic",
            "code_hashes": code_hashes(),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "families": registries,
            "model_source": model_source(config),
            "max_length": config.max_length,
            "atol": config.atol,
            "rtol": config.rtol,
            "environment": machine,
            "power": power_state(),
            "cases": synthetic_cases(load_tokenizer(config)),
            "measurement_code_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "original_policy_sha256": sha256(
                config.artifact_dir / "confirmation/frozen-policy.json"
            ),
        }
        write_json(plan_path, plan)
    prefix = [sys.executable, "-m", "etdl.worker", "--config", str(config_path.resolve())]
    refs = directory / "references.json"
    if not refs.exists():
        subprocess.run(
            prefix
            + [
                "boundary-reference",
                "--plan",
                str(plan_path.resolve()),
                "--output",
                str(refs.resolve()),
            ],
            check=True,
        )
    if read_json(refs)["plan_sha256"] != sha256(plan_path):
        raise ValueError("Diagnostic references do not match the plan")
    runs = []
    measurements = directory / "measurements"
    for block in range(blocks):
        conditions = [(family, capacity) for family in plan["families"] for capacity in LENGTHS]
        random.Random(plan["seed"] + block).shuffle(conditions)
        for family, capacity in conditions:
            stem = f"b{block}-{family}-{capacity}"
            request_path, result_path = (
                measurements / f"{stem}.request.json",
                measurements / f"{stem}.json",
            )
            request = {
                "plan": str(plan_path.resolve()),
                "plan_sha256": sha256(plan_path),
                "references": str(refs.resolve()),
                "references_sha256": sha256(refs),
                "family": family,
                "capacity": capacity,
                "block": block,
            }
            if request_path.exists() and read_json(request_path) != request:
                raise ValueError("Existing diagnostic request changed")
            if not result_path.exists():
                write_json(request_path, request)
                print(f"Running {stem}", flush=True)
                time.sleep(plan["cooldown_seconds"])
                subprocess.run(
                    prefix
                    + [
                        "boundary",
                        "--request",
                        str(request_path.resolve()),
                        "--output",
                        str(result_path.resolve()),
                    ],
                    check=True,
                )
            result = read_json(result_path)
            if result["request_sha256"] != sha256(request_path):
                raise ValueError("Diagnostic cached measurement inputs changed")
            runs.append(result)
    summary = summarize(runs)
    summary["plan_sha256"] = sha256(plan_path)
    if (
        sha256(config.artifact_dir / "confirmation/frozen-policy.json")
        != plan["original_policy_sha256"]
    ):
        raise ValueError("Original deployment policy changed")
    write_json(directory / "summary.json", summary)
    return summary
