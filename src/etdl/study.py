"""Isolated profiling and measured, bounded subset selection on training data."""

import random
import subprocess
import sys
import time
from datetime import UTC, datetime

import numpy as np

from etdl.benchmark import latency_summary, sampled_memory
from etdl.checks import verified_dynamic
from etdl.data import encode, load_calibration, load_tokenizer, model_source
from etdl.io import read_json, sha256, write_json
from etdl.runtime import environment, session
from etdl.selection import learn_routes, route_name, select_configuration, subsets
from etdl.variants import pad_inputs, verified_variants


def hardware_key(machine):
    return {
        "platform": machine["platform"],
        "cpu": machine["cpu"],
        "cpu_flags": machine["cpu_flags"],
        "ort_version": machine["packages"]["onnxruntime"],
    }


def run_worker(config, request_path, output):
    request = read_json(request_path)
    rows = load_calibration(config, request["limit"])
    paths = {"dynamic": verified_dynamic(config, rows), **verified_variants(config, rows)}
    names = request["models"]
    if not names or any(name not in paths for name in names):
        raise ValueError("Requested an unavailable or unverified model")
    if request["kind"] == "configuration" and (
        "dynamic" not in names
        or any(str(row["idx"]) not in request.get("expected", {}) for row in rows)
    ):
        raise ValueError(
            "Configuration measurements require dynamic fallback and complete references"
        )
    tokenizer = load_tokenizer(config)
    machine = environment()
    samples = []
    initialization_ms = {}
    started = time.perf_counter_ns()
    with sampled_memory() as memory:
        runtimes = {}
        for name in names:
            runtimes[name], initialization_ms[name] = session(paths[name], request["threads"])
            # Probe each loaded graph at capacity before timing, including dynamic fallback.
            length = config.max_length if name == "dynamic" else int(name.split("-")[1])
            ids = np.full((1, length), tokenizer.convert_tokens_to_ids("good"), dtype=np.int64)
            ids[0, 0], ids[0, -1] = tokenizer.cls_token_id, tokenizer.sep_token_id
            runtimes[name].run(["logits"], {"input_ids": ids, "attention_mask": np.ones_like(ids)})
        eligible_rows = rows
        if request["kind"] == "profile" and names[0] != "dynamic":
            eligible_rows = [row for row in rows if row["tokens"] <= int(names[0].split("-")[1])]
        if not eligible_rows:
            raise ValueError("No eligible examples for this option")

        def execute(row, repeat):
            start = time.perf_counter_ns()
            inputs = encode(tokenizer, row["text"], config.max_length)
            tokenized = time.perf_counter_ns()
            length = inputs["input_ids"].shape[1]
            name = (
                names[0]
                if request["kind"] == "profile"
                else route_name(length, request["routing"], runtimes)
            )
            if name != "dynamic":
                inputs = pad_inputs(inputs, int(name.split("-")[1]), tokenizer.pad_token_id)
            routed = time.perf_counter_ns()
            logits = runtimes[name].run(["logits"], inputs)[0]
            finished = time.perf_counter_ns()
            if not np.isfinite(logits).all():
                raise ValueError("Nonfinite inference output")
            return {
                "idx": row["idx"],
                "tokens": length,
                "repeat": repeat,
                "route": name,
                "label": row["label"],
                "prediction": int(logits.argmax(-1)[0]),
                "tokenization_ms": (tokenized - start) / 1e6,
                "routing_padding_ms": (routed - tokenized) / 1e6,
                "inference_ms": (finished - routed) / 1e6,
                "end_to_end_ms": (finished - start) / 1e6,
            }

        for index in range(request["warmup"]):
            execute(eligible_rows[index % len(eligible_rows)], -1)
        ready_ms = (time.perf_counter_ns() - started) / 1e6
        rng = random.Random(config.seed)
        for repeat in range(request["repeats"]):
            order = list(eligible_rows)
            rng.shuffle(order)
            for row in order:
                samples.append(execute(row, repeat))
    expected = request.get("expected", {})
    changed = sum(
        sample["prediction"] != expected[str(sample["idx"])]
        for sample in samples
        if str(sample["idx"]) in expected
    )
    report = {
        "kind": request["kind"],
        "split": "train",
        "models": names,
        "variants": [name for name in names if name != "dynamic"],
        "routing": request.get("routing", {}),
        "threads": request["threads"],
        "repeats": request["repeats"],
        "warmup_requests": request["warmup"],
        "capacity_probes": True,
        "examples": len(eligible_rows),
        "samples": samples,
        "latency": {
            key: latency_summary([sample[key] for sample in samples])
            for key in ("tokenization_ms", "routing_padding_ms", "inference_ms", "end_to_end_ms")
        },
        "accuracy": sum(sample["prediction"] == sample["label"] for sample in samples)
        / len(samples),
        "changed_predictions": changed,
        "memory": memory,
        "memory_for_budget_bytes": max(
            memory["sampled_peak_rss_bytes"], memory["process_lifetime_peak_rss_bytes"] or 0
        ),
        "session_initialization_ms": initialization_ms,
        "sessions_ready_ms": ready_ms,
        "model_size_bytes": sum(paths[name].stat().st_size for name in names),
        "model_hashes": {name: sha256(paths[name]) for name in names},
        "calibration_sha256": sha256(config.artifact_dir / "data/calibration.json"),
        "environment": machine,
    }
    write_json(output, report)
    return report


def study(config, config_path, limit, threads, repeats, warmup, budget_mib=None, name=None):
    if limit <= 0 or threads <= 0 or repeats <= 0 or warmup < 0:
        raise ValueError("Invalid study limits, thread count, repetitions, or warm-up")
    rows = load_calibration(config, limit)
    verified_dynamic(config, rows)
    variants = verified_variants(config, rows)
    machine = environment()
    budget_bytes = (
        int(budget_mib * 2**20)
        if budget_mib is not None
        else int(machine["available_memory_bytes"] * 0.10)
    )
    if budget_bytes <= 0:
        raise ValueError("Memory budget must be positive")
    name = name or f"study-t{threads}"
    if name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("Study name must be a single directory name")
    destination = config.artifact_dir / name
    if destination.exists():
        raise ValueError(
            f"Study exists at {destination}; preserve it and use a new artifact directory"
        )
    destination.mkdir(parents=True)
    rng = random.Random(config.seed)
    common = {"limit": limit, "threads": threads, "repeats": repeats, "warmup": warmup}
    worker = [sys.executable, "-m", "etdl.worker", "--config", str(config_path.resolve()), "run"]

    def measure(request, name):
        request_path = destination / f"{name}.request.json"
        output = destination / f"{name}.json"
        write_json(request_path, request)
        print(f"Measuring {name}", flush=True)
        subprocess.run(
            worker + ["--request", str(request_path), "--output", str(output)], check=True
        )
        result = read_json(output)
        if hardware_key(result["environment"]) != hardware_key(machine):
            raise ValueError("Hardware/runtime changed during the study")
        return result

    names = ["dynamic", *variants]
    rng.shuffle(names)
    profiles = {}
    for name in names:
        # Do not invent performance for variants with no representative calibration input.
        if name != "dynamic" and not any(row["tokens"] <= int(name.split("-")[1]) for row in rows):
            continue
        profiles[name] = measure({**common, "kind": "profile", "models": [name]}, f"profile-{name}")
    expected = {}
    for sample in profiles["dynamic"]["samples"]:
        key = str(sample["idx"])
        if key in expected and expected[key] != sample["prediction"]:
            raise ValueError("Dynamic predictions were not stable across repeats")
        expected[key] = sample["prediction"]
    groups = subsets([name for name in profiles if name != "dynamic"])
    rng.shuffle(groups)
    measurements = []
    for group in groups:
        routing = learn_routes(profiles, group)
        # Retaining a variant which never wins any route only adds memory.
        if any(name not in routing.values() for name in group):
            continue
        name = "config-" + ("-".join(group) if group else "dynamic")
        measurements.append(
            measure(
                {
                    **common,
                    "kind": "configuration",
                    "models": ["dynamic", *group],
                    "routing": routing,
                    "expected": expected,
                },
                name,
            )
        )
    overview = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "split": "train",
        "examples": len(rows),
        "threads": threads,
        "seed": config.seed,
        "repeats": repeats,
        "memory_budget_bytes": budget_bytes,
        "budget_rule": "explicit MiB"
        if budget_mib is not None
        else "10% of initially available RAM",
        "hardware_key": hardware_key(machine),
        "measurements": measurements,
        "profile_order": list(profiles),
        "subset_order": [item["variants"] for item in measurements],
        "calibration_sha256": sha256(config.artifact_dir / "data/calibration.json"),
        "model_source": model_source(config),
        "limitations": [
            "Calibration-only selection; validation is still held out.",
            "Small repeated workload; selection may overfit timing noise.",
            "Unseen token lengths route dynamically, without extrapolating speedups.",
            "Memory is process RSS, with capacity probes; sampling can miss brief peaks.",
            "Idle variants are pruned before measurement because they add only memory.",
        ],
    }
    summary_path = destination / "summary.json"
    try:
        chosen = select_configuration(measurements, budget_bytes)
    except ValueError:
        overview["status"] = "no_feasible_configuration"
        write_json(summary_path, overview)
        write_study_outputs(summary_path)
        raise
    policy = {
        "status": "verified_calibration_selection",
        "variants": chosen["variants"],
        "routing": chosen["routing"],
        "threads": threads,
        "model_hashes": chosen["model_hashes"],
        "calibration_sha256": overview["calibration_sha256"],
        "model_source": model_source(config),
        "hardware_key": overview["hardware_key"],
        "memory_budget_bytes": budget_bytes,
        "measured_memory_bytes": chosen["memory_for_budget_bytes"],
        "calibration_p95_ms": chosen["latency"]["end_to_end_ms"]["p95_ms"],
        "atol": config.atol,
        "rtol": config.rtol,
    }
    policy_path = destination / "selection.json"
    write_json(policy_path, policy)
    overview["status"] = "selected"
    overview["selection"] = policy
    write_json(summary_path, overview)
    write_study_outputs(summary_path)
    return {"summary": str(summary_path), "policy": str(policy_path), **policy}


def write_study_outputs(summary_path):
    """Export raw configuration aggregates plus a calibration-only latency-memory figure."""
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = read_json(summary_path)
    selected = summary.get("selection", {}).get("variants")
    budget = summary["memory_budget_bytes"] / 2**20
    records = []
    for item in summary["measurements"]:
        records.append(
            {
                "variants": "+".join(item["variants"]) or "dynamic",
                "resident_peak_mib": item["memory_for_budget_bytes"] / 2**20,
                "median_ms": item["latency"]["end_to_end_ms"]["median_ms"],
                "p95_ms": item["latency"]["end_to_end_ms"]["p95_ms"],
                "training_accuracy": item["accuracy"],
                "changed_predictions": item["changed_predictions"],
                "feasible": item["memory_for_budget_bytes"] <= summary["memory_budget_bytes"]
                and item["changed_predictions"] == 0,
                "selected": item["variants"] == selected,
                "variant_count": len(item["variants"]),
            }
        )
    with summary_path.with_name("configurations.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2563eb", "#059669", "#7c3aed"]
    for count, marker in enumerate(("s", "o", "^")):
        group = [row for row in records if row["variant_count"] == count]
        if group:
            ax.scatter(
                [row["resident_peak_mib"] for row in group],
                [row["p95_ms"] for row in group],
                s=85,
                marker=marker,
                color=colors[count],
                label=f"{count} fixed variants",
            )
    for row in records:
        if row["selected"]:
            ax.scatter(
                row["resident_peak_mib"],
                row["p95_ms"],
                s=230,
                marker="*",
                color="#fbbf24",
                edgecolor="#111827",
                zorder=5,
                label="Selected",
            )
        label = row["variants"].replace("fixed-", "") + f" ({row['training_accuracy']:.1%})"
        ax.annotate(
            label,
            (row["resident_peak_mib"], row["p95_ms"]),
            xytext=(6, 9),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvline(
        budget, color="#dc2626", linestyle="--", alpha=0.7, label=f"Budget: {budget:.0f} MiB"
    )
    ax.set(
        xlabel="Measured process peak RSS (MiB)",
        ylabel="Combined request latency p95 (ms)",
        title=f"Training calibration: latency vs resident memory ({summary['threads']} CPU thread)",
    )
    ax.margins(x=0.15)
    ax.set_ylim(0, max(row["p95_ms"] for row in records) * 1.18)
    ax.grid(alpha=0.15)
    ax.legend(loc="best", fontsize=8)
    fig.text(
        0.5,
        0.015,
        f"{summary['examples']} training examples; labels show training accuracy. "
        "No held-out performance claim.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(summary_path.with_name("latency-memory.png"), dpi=180)
    fig.savefig(summary_path.with_name("latency-memory.svg"))
    plt.close(fig)
