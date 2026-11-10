"""Archive confirmation/evaluation evidence and render descriptive research figures."""

import argparse
import csv
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from etdl.comparison import METHODS
from etdl.io import read_json, sha256, write_json

LABELS = {
    "pytorch": "PyTorch",
    "dynamic": "Dynamic",
    "single-16": "Fixed 16",
    "buckets": "Buckets",
    "selected": "Prior policy",
}
COLORS = {
    "pytorch": "#475569",
    "dynamic": "#2563eb",
    "single-16": "#d97706",
    "buckets": "#dc2626",
    "selected": "#059669",
}


def csv_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def capture_traces(output, directory, config_path):
    """Capture diagnostic CPU traces after every performance run has finished."""
    from etdl.comparison import validate_frozen_plan, validate_plan
    from etdl.config import load_config
    from etdl.data import encode, load_tokenizer
    from etdl.runtime import session
    from etdl.variants import pad_inputs

    config = load_config(config_path)
    frozen = validate_frozen_plan(config, directory / "evaluation-plan.json")
    plan = read_json(Path(frozen["execution_plan"]))
    paths = validate_plan(config, plan)
    workload = read_json(directory / "held-out/validation.json")
    row = min(workload["rows"], key=lambda item: item["tokens"])
    if row["tokens"] > 16:
        raise ValueError("Diagnostic common-input trace requires an example fitting all variants")
    tokenizer = load_tokenizer(config)
    inputs = encode(tokenizer, row["text"], config.max_length)
    trace_dir = output / "traces"
    trace_dir.mkdir()
    reports = {}
    for name, path in paths.items():
        feed = (
            inputs
            if name == "dynamic"
            else pad_inputs(inputs, int(name.split("-")[1]), tokenizer.pad_token_id)
        )
        runtime, _ = session(
            path, frozen["threads"], profile_prefix=trace_dir / name, allow_spinning=False
        )
        for _ in range(4):
            runtime.run(["logits"], feed)
        trace = Path(runtime.end_profiling())
        events = read_json(trace)
        nodes = [event for event in events if event.get("cat") == "Node"]
        providers = sorted(
            {
                event.get("args", {}).get("provider")
                for event in nodes
                if event.get("args", {}).get("provider")
            }
        )
        if providers != ["CPUExecutionProvider"]:
            raise ValueError("Unexpected diagnostic execution provider")
        reports[name] = {
            "trace": trace.relative_to(output).as_posix(),
            "model_sha256": sha256(path),
            "input_shape": list(feed["input_ids"].shape),
            "providers": providers,
            "operator_calls": dict(
                sorted(
                    Counter(
                        event.get("args", {}).get("op_name", "unknown") for event in nodes
                    ).items()
                )
            ),
        }
        del runtime
    write_json(
        output / "operator-traces.json",
        {
            "purpose": "Diagnostic instrumentation after all timed runs; durations are not benchmark estimates.",
            "threads": frozen["threads"],
            "allow_spinning": False,
            "common_validation_idx": row["idx"],
            "unpadded_tokens": row["tokens"],
            "requests_per_graph": 4,
            "graphs": reports,
        },
    )


def figures(output, confirmation, held_out, plan):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels_by_method = dict(LABELS)
    retained = "+".join(name.removeprefix("fixed-") for name in plan["policy"]["variants"])
    labels_by_method["selected"] = f"Prior {retained}" if retained else "Prior dynamic"
    groups = {(item["method"], item["threads"]): item for item in confirmation["groups"]}
    threads = sorted({item["threads"] for item in confirmation["groups"]})
    fig, axes = plt.subplots(1, len(threads), figsize=(14, 5), sharey=True, squeeze=False)
    for thread, ax in zip(threads, axes[0], strict=True):
        for index, method in enumerate(METHODS):
            values = groups[(method, thread)]["run_p95_ms"]
            positions = index + np.linspace(-0.10, 0.10, len(values))
            median = float(np.median(values))
            ax.scatter(positions, values, color=COLORS[method], s=40, alpha=0.8)
            ax.plot(
                [index - 0.22, index + 0.22], [median, median], color=COLORS[method], linewidth=2
            )
            ax.annotate(
                f"med {median:.1f}",
                (index, max(values)),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        ax.set_title(f"{thread} intra-op thread{'s' if thread > 1 else ''}")
        ax.set_xticks(
            range(len(METHODS)),
            [labels_by_method[method] for method in METHODS],
            rotation=25,
            ha="right",
        )
        ax.set_ylim(
            0, max(value for group in groups.values() for value in group["run_p95_ms"]) * 1.18
        )
        ax.grid(axis="y", alpha=0.15)
    axes[0, 0].set_ylabel("Per-process combined latency p95 (ms)")
    fig.suptitle("Disjoint training confirmation: individual process runs and medians")
    fig.text(
        0.5,
        0.02,
        f"{plan['examples']} new training examples; {plan['blocks']} randomized process blocks. Dots are runs; bars are medians, not confidence intervals.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(output / "confirmation.png", dpi=180)
    fig.savefig(output / "confirmation.svg")
    plt.close(fig)

    runs = {run["method"]: run for run in held_out["runs"]}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    labels = [
        f"{labels_by_method[method]}\n{runs[method]['accuracy']:.1%} accuracy" for method in METHODS
    ]
    x = np.arange(len(METHODS))
    for ax, values, ylabel in (
        (
            axes[0],
            [runs[method]["latency"]["end_to_end_ms"]["p95_ms"] for method in METHODS],
            "Combined latency p95 (ms)",
        ),
        (
            axes[1],
            [runs[method]["memory_for_budget_bytes"] / 2**20 for method in METHODS],
            "Measured peak process RSS (MiB)",
        ),
    ):
        ax.bar(x, values, color=[COLORS[method] for method in METHODS], width=0.65)
        for index, value in enumerate(values):
            ax.annotate(
                f"{value:.1f}",
                (index, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        ax.set_xticks(x, labels, rotation=22, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values) * 1.22)
        ax.grid(axis="y", alpha=0.15)
    budget = read_json(output / "frozen-policy.json")["memory_budget_bytes"] / 2**20
    axes[1].axhline(
        budget,
        color="#111827",
        linestyle="--",
        linewidth=1,
        label=f"Deployment cap: {budget:g} MiB",
    )
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Held-out SST-2: all 872 examples, {held_out['threads']} CPU threads")
    fig.text(
        0.5,
        0.02,
        "One timing process per method. Deployment was chosen before validation; no latency confidence intervals.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(output / "held-out.png", dpi=180)
    fig.savefig(output / "held-out.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ablation", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    parser.add_argument(
        "--traces", action="store_true", help="Capture CPU operator traces after timing"
    )
    arguments = parser.parse_args()
    directory, output = arguments.comparison.resolve(), arguments.output.resolve()
    confirmation = read_json(directory / "summary.json")
    held_out = read_json(directory / "held-out/summary.json")
    plan = read_json(directory / "plan.json")
    frozen = read_json(directory / "evaluation-plan.json")
    if confirmation["status"] != "complete" or held_out["status"] not in {
        "complete",
        "completed_with_failures",
    }:
        raise ValueError("Only completed comparisons can be archived; failures are retained")
    if confirmation["plan_sha256"] != sha256(directory / "plan.json") or held_out[
        "freeze_sha256"
    ] != sha256(directory / "evaluation-plan.json"):
        raise ValueError("Experiment provenance changed")
    for key in ("comparison_plan", "comparison_summary", "frozen_policy", "execution_plan"):
        if sha256(Path(frozen[key])) != frozen[f"{key}_sha256"]:
            raise ValueError(f"Frozen provenance changed: {key}")
    if output.exists():
        raise ValueError("Archive exists; preserve it and choose a new output directory")
    output.mkdir(parents=True)
    copies = {
        "plan.json": directory / "plan.json",
        "confirmation-summary.json": directory / "summary.json",
        "confirmation-runs.csv": directory / "runs.csv",
        "evaluation-plan.json": directory / "evaluation-plan.json",
        "evaluation-execution.json": directory / "evaluation-execution.json",
        "frozen-policy.json": directory / "frozen-policy.json",
        "held-out-summary.json": directory / "held-out/summary.json",
        "validation-started.json": directory / "held-out/validation-started.json",
        "training-references.json": directory / "references.json",
        "validation-references.json": directory / "held-out/references.json",
    }
    if arguments.ablation:
        copies["ablation-reanalysis.json"] = arguments.ablation
    if (directory / "prediction-smoke.json").exists():
        copies["prediction-smoke.json"] = directory / "prediction-smoke.json"
    for name, source in copies.items():
        shutil.copyfile(source, output / name)
    for name, source in (
        ("training", directory / "training.json"),
        ("validation", directory / "held-out/validation.json"),
    ):
        workload = read_json(source)
        metadata = {key: value for key, value in workload.items() if key != "rows"}
        metadata["original_workload_sha256"] = sha256(source)
        metadata["rows"] = [
            {key: value for key, value in row.items() if key != "text"} for row in workload["rows"]
        ]
        write_json(output / f"{name}-metadata.json", metadata)
    confirmation_runs = []
    for condition in plan["schedule"]:
        path = (
            directory / f"b{condition['block']}-{condition['method']}-t{condition['threads']}.json"
        )
        run = read_json(path)
        if any(run[key] != condition[key] for key in ("block", "method", "threads")):
            raise ValueError("A process result is misassigned")
        confirmation_runs.append(run)
    validation_runs = [read_json(directory / f"held-out/{method}.json") for method in METHODS]
    process_metrics, requests, by_length = [], [], []
    for phase, runs in (("confirmation", confirmation_runs), ("validation", validation_runs)):
        for run in runs:
            process_metrics.append(
                {key: value for key, value in run.items() if key not in {"samples", "environment"}}
                | {"phase": phase}
            )
            requests.extend(
                {
                    "phase": phase,
                    "method": run["method"],
                    "threads": run["threads"],
                    "block": run["block"],
                    **sample,
                }
                for sample in run["samples"]
            )
            if phase == "validation":
                buckets = defaultdict(list)
                for sample in run["samples"]:
                    buckets[sample["tokens"]].append(sample)
                for length, samples in sorted(buckets.items()):
                    values = [sample["end_to_end_ms"] for sample in samples]
                    by_length.append(
                        {
                            "method": run["method"],
                            "tokens": length,
                            "examples": len(samples),
                            "median_ms": float(np.median(values)),
                            "p95_ms": float(np.quantile(values, 0.95)),
                            "accuracy": sum(
                                sample["label"] == sample["prediction"] for sample in samples
                            )
                            / len(samples),
                            "changed_predictions": sum(
                                not sample["labels_match"] for sample in samples
                            ),
                        }
                    )
    csv_rows(output / "requests.csv", requests)
    csv_rows(output / "held-out-by-length.csv", by_length)
    write_json(output / "process-metrics.json", process_metrics)
    write_json(output / "environment.json", confirmation_runs[0]["environment"])
    validation = read_json(directory / "held-out/validation.json")
    selections = next(run for run in validation_runs if run["method"] == "selected")
    write_json(
        output / "routing-summary.json",
        {
            "prior_policy_routes": dict(
                Counter(sample["route"] for sample in selections["samples"])
            ),
            "unseen_original_token_lengths": sorted(
                {
                    row["tokens"]
                    for row in validation["rows"]
                    if str(row["tokens"]) not in plan["policy"]["routing"]
                }
            ),
            "validation_length_counts": dict(
                sorted(Counter(row["tokens"] for row in validation["rows"]).items())
            ),
        },
    )
    write_json(
        output / "archive-metadata.json",
        {
            "training_measurement_commit": plan["measurement_code_commit"],
            "evaluation_measurement_commit": frozen["measurement_code_commit"],
            "reporting_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "confirmation_processes": len(confirmation_runs),
            "validation_processes": len(validation_runs),
            "confirmation_requests": sum(len(run["samples"]) for run in confirmation_runs),
            "validation_requests": sum(len(run["samples"]) for run in validation_runs),
            "note": "Raw frozen plans preserve original host paths for audit; rerun the CLI to reproduce a new experiment.",
        },
    )
    figures(output, confirmation, held_out, plan)
    if arguments.traces:
        capture_traces(output, directory, arguments.config)
    write_json(
        output / "checksums.json",
        {
            path.relative_to(output).as_posix(): sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path != output / "checksums.json"
        },
    )
    print(f"Archived comparison and evaluation evidence to {output}")


if __name__ == "__main__":
    main()
