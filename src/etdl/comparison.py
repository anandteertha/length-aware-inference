"""Randomized process-level confirmation and a frozen held-out evaluation."""

import csv
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from etdl.benchmark import latency_summary, sampled_memory
from etdl.checks import compare_logits, verified_dynamic
from etdl.data import DATASET_ID, encode, load_calibration, load_tokenizer, model_source
from etdl.io import read_json, sha256, write_json
from etdl.model import load_pytorch
from etdl.runtime import environment, session
from etdl.selection import route_name
from etdl.study import hardware_key
from etdl.variants import LENGTHS, pad_inputs, verified_variants

METHODS = ("pytorch", "dynamic", "single-16", "buckets", "selected")


def schedule(methods, threads, blocks, seed):
    if blocks < 1 or not methods or not threads or min(threads) < 1:
        raise ValueError("A schedule requires methods, positive threads and blocks")
    result = []
    for block in range(blocks):
        pairs = [(method, thread) for method in methods for thread in threads]
        random.Random(seed + block).shuffle(pairs)
        result.extend(
            {"block": block, "method": method, "threads": thread} for method, thread in pairs
        )
    return result


def method_models(method, policy):
    if method == "pytorch":
        return []
    if method == "dynamic":
        return ["dynamic"]
    if method == "single-16":
        return ["dynamic", "fixed-16"]
    if method == "buckets":
        return ["dynamic", *(f"fixed-{length}" for length in LENGTHS)]
    if method == "selected":
        return ["dynamic", *policy["variants"]]
    raise ValueError(f"Unknown comparison method: {method}")


def choose_route(method, length, policy, loaded):
    if method in {"pytorch", "dynamic"}:
        return method
    if method == "selected":
        return route_name(length, policy["routing"], loaded)
    if method == "single-16":
        return "fixed-16" if length <= 16 else "dynamic"
    if method == "buckets":
        return next((f"fixed-{capacity}" for capacity in LENGTHS if length <= capacity), "dynamic")
    raise ValueError(f"Unknown comparison method: {method}")


def prepare_workload(config, destination, limit, split="train", freeze_path=None):
    from datasets import load_dataset

    calibration_path = config.artifact_dir / "data/calibration.json"
    calibration = read_json(calibration_path)
    if split == "validation":
        if freeze_path is None or not freeze_path.exists():
            raise ValueError("Validation access requires a frozen evaluation plan")
        validate_frozen_plan(config, freeze_path)
    elif split != "train":
        raise ValueError("Only training confirmation or frozen validation are supported")
    tokenizer = load_tokenizer(config)
    dataset = load_dataset(DATASET_ID, "sst2", split=split, revision=calibration["revision"])
    excluded = {row["idx"] for row in calibration["rows"]} if split == "train" else set()
    if split == "train":
        dataset = dataset.shuffle(seed=config.seed + 1)
    rows = []
    for row in dataset:
        if row["idx"] in excluded:
            continue
        inputs = encode(tokenizer, row["sentence"], config.max_length)
        rows.append(
            {
                "idx": int(row["idx"]),
                "text": row["sentence"],
                "label": int(row["label"]),
                "tokens": int(inputs["input_ids"].shape[1]),
            }
        )
        if split == "train" and len(rows) >= limit:
            break
    if split == "validation" and len(rows) != 872:
        raise ValueError("Expected the pinned SST-2 validation split to contain 872 examples")
    if split == "train" and len(rows) != limit:
        raise ValueError("Not enough disjoint training examples")
    write_json(
        destination,
        {
            "dataset_id": DATASET_ID,
            "revision": calibration["revision"],
            "split": split,
            "seed": config.seed + 1,
            "excluded_selection_ids": sorted(excluded),
            "rows": rows,
            "selection_calibration_sha256": sha256(calibration_path),
            "freeze_sha256": sha256(freeze_path) if freeze_path else None,
        },
    )
    return rows


def references(config, workload_path, output):
    import torch

    workload = read_json(workload_path)
    torch.set_num_interop_threads(1)
    model = load_pytorch(config, threads=1)
    tokenizer = load_tokenizer(config)
    results = {}
    with torch.inference_mode():
        for row in workload["rows"]:
            inputs = encode(tokenizer, row["text"], config.max_length)
            logits = model(**{key: torch.from_numpy(value.copy()) for key, value in inputs.items()})
            values = logits.logits.numpy()
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite reference output")
            results[str(row["idx"])] = values.tolist()
    write_json(
        output,
        {
            "workload_sha256": sha256(workload_path),
            "model_source": model_source(config),
            "reference": "pytorch_fp32_eager_1_thread",
            "logits": results,
        },
    )


def validate_plan(config, plan):
    paths = {"dynamic": verified_dynamic(config), **verified_variants(config, [])}
    if plan["calibration_sha256"] != sha256(config.artifact_dir / "data/calibration.json"):
        raise ValueError("Selection calibration changed after planning")
    for name, digest in plan["model_hashes"].items():
        if name not in paths or sha256(paths[name]) != digest:
            raise ValueError(f"Comparison artifact changed: {name}")
    if (plan["atol"], plan["rtol"]) != (config.atol, config.rtol):
        raise ValueError("Comparison tolerances changed")
    return paths


def measure(config, request_path, output):
    request = read_json(request_path)
    plan_path = Path(request["plan"])
    if sha256(plan_path) != request["plan_sha256"]:
        raise ValueError("Comparison plan changed")
    plan = read_json(plan_path)
    paths = validate_plan(config, plan)
    machine = environment()
    if hardware_key(machine) != plan["hardware_key"]:
        raise ValueError("Comparison hardware/runtime changed")
    workload_path, reference_path = Path(request["workload"]), Path(request["references"])
    if (
        sha256(workload_path) != request["workload_sha256"]
        or sha256(reference_path) != request["references_sha256"]
    ):
        raise ValueError("Workload or reference changed")
    workload, reference = read_json(workload_path), read_json(reference_path)
    if reference["workload_sha256"] != sha256(workload_path):
        raise ValueError("Reference belongs to another workload")
    if reference["model_source"] != plan["model_source"]:
        raise ValueError("Reference belongs to another model revision")
    if workload["split"] not in {"train", "validation"}:
        raise ValueError("Invalid workload split")
    if workload["split"] == "validation":
        freeze = Path(request["freeze"])
        if (
            sha256(freeze) != workload["freeze_sha256"]
            or sha256(freeze) != request["freeze_sha256"]
        ):
            raise ValueError("Held-out plan changed after validation access")
        frozen = validate_frozen_plan(config, freeze)
        if (
            request["threads"] != frozen["threads"]
            or request["method"] not in frozen["methods"]
            or request["block"] != 0
            or sha256(plan_path) != frozen["execution_plan_sha256"]
        ):
            raise ValueError("Held-out condition differs from the prespecified evaluation")
    rows = workload["rows"]
    expected = reference["logits"]
    if set(expected) != {str(row["idx"]) for row in rows}:
        raise ValueError("Reference coverage does not match workload")
    method, threads = request["method"], request["threads"]
    policy = plan["policy"]
    names = method_models(method, policy)
    tokenizer = load_tokenizer(config)
    runtimes, initialization_ms = {}, {}
    ready_started = time.perf_counter_ns()
    with sampled_memory() as memory:
        if method == "pytorch":
            import torch

            torch.set_num_interop_threads(1)
            started = time.perf_counter_ns()
            model = load_pytorch(config, threads)
            initialization_ms["pytorch"] = (time.perf_counter_ns() - started) / 1e6

            def infer(name, inputs):
                with torch.inference_mode():
                    return model(
                        **{key: torch.from_numpy(value) for key, value in inputs.items()}
                    ).logits.numpy()
        else:
            for name in names:
                runtimes[name], initialization_ms[name] = session(
                    paths[name], threads, allow_spinning=False
                )

            def infer(name, inputs):
                return runtimes[name].run(["logits"], inputs)[0]

        for name in names or ["pytorch"]:
            length = int(name.split("-")[1]) if name.startswith("fixed-") else config.max_length
            ids = np.full((1, length), tokenizer.convert_tokens_to_ids("good"), dtype=np.int64)
            ids[0, 0], ids[0, -1] = tokenizer.cls_token_id, tokenizer.sep_token_id
            infer(name, {"input_ids": ids, "attention_mask": np.ones_like(ids)})

        def execute(row):
            started = time.perf_counter_ns()
            inputs = encode(tokenizer, row["text"], config.max_length)
            tokenized = time.perf_counter_ns()
            length = inputs["input_ids"].shape[1]
            name = choose_route(method, length, policy, runtimes)
            if name.startswith("fixed-"):
                inputs = pad_inputs(inputs, int(name.split("-")[1]), tokenizer.pad_token_id)
            routed = time.perf_counter_ns()
            logits = infer(name, inputs)
            finished = time.perf_counter_ns()
            check = compare_logits(expected[str(row["idx"])], logits, config.atol, config.rtol)
            return {
                "idx": row["idx"],
                "tokens": length,
                "label": row["label"],
                "prediction": int(logits.argmax(-1)[0]),
                "route": name,
                "tokenization_ms": (tokenized - started) / 1e6,
                "routing_padding_ms": (routed - tokenized) / 1e6,
                "inference_ms": (finished - routed) / 1e6,
                "end_to_end_ms": (finished - started) / 1e6,
                **check,
            }

        for index in range(plan["warmup"]):
            execute(rows[index % len(rows)])
        ready_ms = (time.perf_counter_ns() - ready_started) / 1e6
        ordered = list(rows)
        random.Random(plan["seed"] + request["block"]).shuffle(ordered)
        samples = [execute(row) for row in ordered]
    if method == "pytorch":
        from huggingface_hub import try_to_load_from_cache

        source = model_source(config)
        checkpoint = try_to_load_from_cache(
            source["model_id"], "model.safetensors", revision=source["revision"]
        )
        size = Path(checkpoint).stat().st_size if isinstance(checkpoint, str) else None
    else:
        size = sum(paths[name].stat().st_size for name in names)
    report = {
        "method": method,
        "threads": threads,
        "block": request["block"],
        "split": workload["split"],
        "examples": len(rows),
        "samples": samples,
        "accuracy": sum(sample["label"] == sample["prediction"] for sample in samples)
        / len(samples),
        "passed": all(sample["passed"] for sample in samples),
        "changed_predictions": sum(not sample["labels_match"] for sample in samples),
        "max_absolute_error": max((sample["max_absolute_error"] or 0) for sample in samples),
        "latency": {
            name: latency_summary([sample[name] for sample in samples])
            for name in ("tokenization_ms", "routing_padding_ms", "inference_ms", "end_to_end_ms")
        },
        "memory": memory,
        "memory_for_budget_bytes": max(
            memory["sampled_peak_rss_bytes"], memory["process_lifetime_peak_rss_bytes"] or 0
        ),
        "model_size_bytes": size,
        "session_initialization_ms": initialization_ms,
        "sessions_ready_ms": ready_ms,
        "runtime_settings": {"allow_spinning": False},
        "environment": machine,
        "workload_sha256": sha256(workload_path),
        "references_sha256": sha256(reference_path),
        "plan_sha256": sha256(plan_path),
    }
    write_json(output, report)
    print(
        f"{method} t{threads} block {request['block']}: p95={report['latency']['end_to_end_ms']['p95_ms']:.3f} ms",
        flush=True,
    )


def deployment_candidate(groups, budget_bytes):
    eligible = [
        group
        for group in groups
        if group["method"] in {"dynamic", "selected"}
        and group["runs"] >= 3
        and group["passed"]
        and group["changed_predictions"] == 0
        and group["max_peak_bytes"] <= budget_bytes
    ]
    if not eligible:
        raise ValueError(
            "No confirmed deployment candidate meets correctness and memory constraints"
        )
    return min(
        eligible,
        key=lambda group: (
            group["median_run_p95_ms"],
            group["max_peak_bytes"],
            group["method"] != "dynamic",
        ),
    )


def freeze(config, directory):
    plan_path, summary_path = directory / "plan.json", directory / "summary.json"
    plan, summary = read_json(plan_path), read_json(summary_path)
    if (
        summary["status"] != "complete"
        or not summary["all_correctness_checks_passed"]
        or summary["plan_sha256"] != sha256(plan_path)
    ):
        raise ValueError("A complete, passing training comparison is required before freezing")
    validate_plan(config, plan)
    evaluation_path = directory / "evaluation-plan.json"
    if evaluation_path.exists():
        raise ValueError("Evaluation plan already frozen; do not retune using validation")
    chosen = deployment_candidate(summary["groups"], plan["policy"]["memory_budget_bytes"])
    policy = dict(plan["policy"])
    policy.update(
        {
            "threads": chosen["threads"],
            "runtime_settings": {"allow_spinning": False},
            "measured_memory_bytes": chosen["max_peak_bytes"],
            "calibration_p95_ms": chosen["median_run_p95_ms"],
            "selection_metric": "median of independent training-confirmation run p95s",
            "confirmation_plan_sha256": sha256(plan_path),
            "confirmation_summary_sha256": sha256(summary_path),
            "confirmation_training_sha256": plan["workload_sha256"],
        }
    )
    if chosen["method"] == "dynamic":
        policy["variants"], policy["routing"] = [], {}
        policy["model_hashes"] = {"dynamic": plan["model_hashes"]["dynamic"]}
    policy_path = directory / "frozen-policy.json"
    write_json(policy_path, policy)
    execution = dict(plan)
    execution.update(
        {
            # Retain the original 16+32 candidate as a comparison even if deployment rejects it.
            "policy": plan["policy"],
            "threads": [chosen["threads"]],
            "blocks": 1,
            "seed": config.seed + 1000,
            "workload_sha256": None,
            "schedule": schedule(METHODS, [chosen["threads"]], 1, config.seed + 1000),
            "purpose": "single_prespecified_held_out_comparison",
        }
    )
    execution_path = directory / "evaluation-execution.json"
    write_json(execution_path, execution)
    frozen = {
        "status": "frozen_for_validation",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "threads": chosen["threads"],
        "methods": list(METHODS),
        "chosen_confirmation_method": chosen["method"],
        "calibration_sha256": plan["calibration_sha256"],
        "comparison_plan": str(plan_path.resolve()),
        "comparison_plan_sha256": sha256(plan_path),
        "comparison_summary": str(summary_path.resolve()),
        "comparison_summary_sha256": sha256(summary_path),
        "frozen_policy": str(policy_path.resolve()),
        "frozen_policy_sha256": sha256(policy_path),
        "execution_plan": str(execution_path.resolve()),
        "execution_plan_sha256": sha256(execution_path),
        "dataset_revision": read_json(config.artifact_dir / "data/calibration.json")["revision"],
        "hardware_key": plan["hardware_key"],
        "model_source": plan["model_source"],
        "validation_examples": 872,
        "measurement_code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    write_json(evaluation_path, frozen)
    return frozen


def validate_frozen_plan(config, path):
    frozen = read_json(path)
    if frozen.get("status") != "frozen_for_validation":
        raise ValueError("Invalid frozen evaluation status")
    if frozen["calibration_sha256"] != sha256(config.artifact_dir / "data/calibration.json"):
        raise ValueError("Frozen plan calibration data changed")
    for key in ("comparison_plan", "comparison_summary", "frozen_policy", "execution_plan"):
        if sha256(Path(frozen[key])) != frozen[f"{key}_sha256"]:
            raise ValueError(f"Frozen {key} changed; evaluation must not be retuned")
    if frozen["model_source"] != model_source(config):
        raise ValueError("Frozen checkpoint revision changed")
    if frozen["hardware_key"] != hardware_key(environment()):
        raise ValueError("Frozen hardware/runtime changed")
    validate_plan(config, read_json(Path(frozen["execution_plan"])))
    return frozen


def evaluate(config, config_path, directory, resume=False):
    freeze_path = directory / "evaluation-plan.json"
    frozen = validate_frozen_plan(config, freeze_path)
    execution_path = Path(frozen["execution_plan"])
    execution = read_json(execution_path)
    destination = directory / "held-out"
    if destination.exists() and not resume:
        raise ValueError("Held-out evaluation exists; resume only the same frozen plan")
    destination.mkdir(exist_ok=True)
    marker = destination / "validation-started.json"
    if marker.exists() and read_json(marker)["freeze_sha256"] != sha256(freeze_path):
        raise ValueError("Frozen plan changed after validation access")
    if not marker.exists():
        write_json(
            marker,
            {"started_at_utc": datetime.now(UTC).isoformat(), "freeze_sha256": sha256(freeze_path)},
        )
    workload_path = destination / "validation.json"
    if not workload_path.exists():
        prepare_workload(config, workload_path, 872, split="validation", freeze_path=freeze_path)
    reference_path = destination / "references.json"
    if not reference_path.exists():
        subprocess_worker(
            config_path, "reference", "--workload", workload_path, "--output", reference_path
        )
    runs = []
    for condition in execution["schedule"]:
        output = destination / f"{condition['method']}.json"
        request_path = destination / f"{condition['method']}.request.json"
        request = {
            **condition,
            "plan": str(execution_path),
            "plan_sha256": sha256(execution_path),
            "workload": str(workload_path.resolve()),
            "workload_sha256": sha256(workload_path),
            "references": str(reference_path.resolve()),
            "references_sha256": sha256(reference_path),
            "freeze": str(freeze_path.resolve()),
            "freeze_sha256": sha256(freeze_path),
        }
        if not output.exists():
            write_json(request_path, request)
            print(f"Held-out: {condition['method']} t{condition['threads']}", flush=True)
            subprocess_worker(
                config_path, "comparison", "--request", request_path, "--output", output
            )
        result = read_json(output)
        if any(
            result[key] != request[key]
            for key in (
                "method",
                "threads",
                "block",
                "plan_sha256",
                "workload_sha256",
                "references_sha256",
            )
        ):
            raise ValueError("Existing held-out result belongs to another frozen evaluation")
        if result["examples"] != 872 or result["split"] != "validation":
            raise ValueError("Incomplete held-out result")
        runs.append(result)
    deployment = next(run for run in runs if run["method"] == frozen["chosen_confirmation_method"])
    passed = all(run["passed"] for run in runs)
    budget_passed = (
        deployment["memory_for_budget_bytes"] <= execution["policy"]["memory_budget_bytes"]
    )
    summary = {
        "status": "complete" if passed and budget_passed else "completed_with_failures",
        "split": "validation",
        "examples": 872,
        "timing_runs_per_method": 1,
        "freeze_sha256": sha256(freeze_path),
        "threads": frozen["threads"],
        "deployment_method": frozen["chosen_confirmation_method"],
        "all_correctness_checks_passed": passed,
        "deployment_memory_within_budget": budget_passed,
        "runs": [
            {key: value for key, value in run.items() if key not in {"samples", "environment"}}
            for run in runs
        ],
        "limitations": [
            "One held-out timing run per method; no latency confidence intervals.",
            "CPU laptop results do not establish performance on other edge hardware.",
            "Readiness excludes Python startup/imports and is not a cold device start.",
            "Validation results did not change the frozen policy.",
        ],
    }
    write_json(destination / "summary.json", summary)
    return summary


def summarize_runs(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[(run["method"], run["threads"])].append(run)
    summaries = []
    for (method, threads), group in sorted(groups.items()):
        p95 = [run["latency"]["end_to_end_ms"]["p95_ms"] for run in group]
        summaries.append(
            {
                "method": method,
                "threads": threads,
                "runs": len(group),
                "median_run_p95_ms": float(np.median(p95)),
                "min_run_p95_ms": min(p95),
                "max_run_p95_ms": max(p95),
                "run_p95_ms": p95,
                "max_peak_bytes": max(run["memory_for_budget_bytes"] for run in group),
                "accuracy": [run["accuracy"] for run in group],
                "passed": all(run["passed"] for run in group),
                "changed_predictions": sum(run["changed_predictions"] for run in group),
            }
        )
    ratios = []
    for threads in sorted({run["threads"] for run in runs}):
        by_method = {
            method: {run["block"]: run for run in groups.get((method, threads), [])}
            for method in ("dynamic", "selected")
        }
        common = sorted(set(by_method["dynamic"]) & set(by_method["selected"]))
        values = [
            by_method["dynamic"][block]["latency"]["end_to_end_ms"]["p95_ms"]
            / by_method["selected"][block]["latency"]["end_to_end_ms"]["p95_ms"]
            for block in common
        ]
        if values:
            ratios.append(
                {
                    "threads": threads,
                    "matched_blocks": common,
                    "ratios": values,
                    "median_ratio": float(np.median(values)),
                    "range": [min(values), max(values)],
                }
            )
    return summaries, ratios


def subprocess_worker(config_path, command, *arguments):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "etdl.worker",
            "--config",
            str(config_path.resolve()),
            command,
            *map(str, arguments),
        ],
        check=True,
    )


def compare(config, config_path, policy_path, name, limit, threads, blocks, warmup, resume=False):
    if name in {".", ".."} or any(value in name for value in "/\\:"):
        raise ValueError("Comparison name must be a single directory name")
    if (
        blocks < 3
        or limit < 1
        or warmup < 0
        or not threads
        or min(threads) < 1
        or len(set(threads)) != len(threads)
    ):
        raise ValueError(
            "Use at least three blocks, positive examples, unique threads and valid warm-up"
        )
    destination = config.artifact_dir / name
    plan_path = destination / "plan.json"
    if destination.exists() and not resume:
        raise ValueError("Comparison exists; choose a new name or explicitly resume it")
    if not destination.exists():
        policy = read_json(policy_path)
        if policy.get("status") != "verified_calibration_selection" or (
            policy["atol"],
            policy["rtol"],
        ) != (config.atol, config.rtol):
            raise ValueError("Source policy is unverified or has different output tolerances")
        rows = load_calibration(config)
        paths = {"dynamic": verified_dynamic(config, rows), **verified_variants(config, rows)}
        if len(policy["variants"]) > 2 or policy["hardware_key"] != hardware_key(environment()):
            raise ValueError("Invalid source policy or incompatible hardware")
        if policy["calibration_sha256"] != sha256(config.artifact_dir / "data/calibration.json"):
            raise ValueError("Source policy belongs to different calibration data")
        if any(
            policy["model_hashes"].get(name) != sha256(paths[name])
            for name in ["dynamic", *policy["variants"]]
        ):
            raise ValueError("Source policy model hashes differ")
        if any(f"fixed-{length}" not in paths for length in LENGTHS):
            raise ValueError("Bucket comparison requires all four checked variants")
        destination.mkdir(parents=True)
        workload_path = destination / "training.json"
        prepare_workload(config, workload_path, limit)
        machine = environment()
        plan = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "seed": config.seed,
            "methods": list(METHODS),
            "threads": threads,
            "blocks": blocks,
            "warmup": warmup,
            "examples": limit,
            "policy": policy,
            "source_policy_sha256": sha256(policy_path),
            "model_hashes": {name: sha256(path) for name, path in paths.items()},
            "model_source": model_source(config),
            "hardware_key": hardware_key(machine),
            "atol": config.atol,
            "rtol": config.rtol,
            "runtime_settings": {"allow_spinning": False},
            "calibration_sha256": sha256(config.artifact_dir / "data/calibration.json"),
            "workload_sha256": sha256(workload_path),
            "measurement_code_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "schedule": schedule(METHODS, threads, blocks, config.seed),
        }
        write_json(plan_path, plan)
    plan = read_json(plan_path)
    if (plan["threads"], plan["blocks"], plan["examples"], plan["warmup"]) != (
        threads,
        blocks,
        limit,
        warmup,
    ):
        raise ValueError("Resume settings differ from the immutable plan")
    validate_plan(config, plan)
    workload_path, reference_path = destination / "training.json", destination / "references.json"
    if sha256(workload_path) != plan["workload_sha256"]:
        raise ValueError("Planned training workload changed")
    if not reference_path.exists():
        subprocess_worker(
            config_path, "reference", "--workload", workload_path, "--output", reference_path
        )
    runs = []
    for condition in plan["schedule"]:
        stem = f"b{condition['block']}-{condition['method']}-t{condition['threads']}"
        request = {
            **condition,
            "plan": str(plan_path.resolve()),
            "plan_sha256": sha256(plan_path),
            "workload": str(workload_path.resolve()),
            "workload_sha256": sha256(workload_path),
            "references": str(reference_path.resolve()),
            "references_sha256": sha256(reference_path),
        }
        request_path, output = destination / f"{stem}.request.json", destination / f"{stem}.json"
        if not output.exists():
            write_json(request_path, request)
            print(f"Running {stem}", flush=True)
            subprocess_worker(
                config_path, "comparison", "--request", request_path, "--output", output
            )
        result = read_json(output)
        if any(
            result[key] != request[key]
            for key in (
                "method",
                "threads",
                "block",
                "plan_sha256",
                "workload_sha256",
                "references_sha256",
            )
        ):
            raise ValueError("Existing result belongs to different inputs")
        runs.append(result)
    groups, ratios = summarize_runs(runs)
    summary = {
        "status": "complete",
        "split": "train",
        "plan_sha256": sha256(plan_path),
        "groups": groups,
        "matched_block_ratios": ratios,
        "all_correctness_checks_passed": all(run["passed"] for run in runs),
        "inference": "Descriptive run-level medians and ranges; no p-values or confidence intervals.",
    }
    write_json(destination / "summary.json", summary)
    with (destination / "runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "block",
                "method",
                "threads",
                "p95_ms",
                "median_ms",
                "peak_mib",
                "accuracy",
                "changed_predictions",
                "passed",
            ],
        )
        writer.writeheader()
        writer.writerows(
            {
                "block": run["block"],
                "method": run["method"],
                "threads": run["threads"],
                "p95_ms": run["latency"]["end_to_end_ms"]["p95_ms"],
                "median_ms": run["latency"]["end_to_end_ms"]["median_ms"],
                "peak_mib": run["memory_for_budget_bytes"] / 2**20,
                "accuracy": run["accuracy"],
                "changed_predictions": run["changed_predictions"],
                "passed": run["passed"],
            }
            for run in runs
        )
    return {"summary": str(destination / "summary.json"), **summary}
