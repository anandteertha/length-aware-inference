"""Archive and plot matched synthetic length diagnostics after all timing finishes."""

import argparse
import csv
import shutil
import subprocess
from itertools import pairwise
from pathlib import Path

import numpy as np

from etdl.boundaries import code_hashes, summarize
from etdl.io import read_json, sha256, write_json

MODES = {
    "natural": ("Dynamic, natural length", "#2563eb"),
    "padded": ("Dynamic, bucket padded", "#d97706"),
    "fixed": ("Fixed, identical padding", "#059669"),
}


def telemetry_gap(run):
    elapsed = [sample["elapsed_s"] for sample in run["cpu_observations"]]
    return max([elapsed[0], *[b - a for a, b in pairwise(elapsed)]]) if elapsed else None


def render(directory, summary, runs, plan):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = sorted(plan["families"])
    for field, filename, ylabel in (
        ("prepared_input_ms", "length-latency", "Median process-run p95, prepared input (ms)"),
        ("routing_padding_ms", "preparation-cost", "Median process-run routing/padding p95 (ms)"),
    ):
        fig, axes = plt.subplots(1, len(families), figsize=(13, 5), squeeze=False)
        for family, ax in zip(families, axes[0], strict=True):
            for mode, (label, color) in MODES.items():
                groups = sorted(
                    [g for g in summary["groups"] if g["family"] == family and g["mode"] == mode],
                    key=lambda g: g["length"],
                )
                x = [g["length"] for g in groups]
                y = [g["latency"][field]["median_run_p95_ms"] for g in groups]
                bounds = np.array([g["latency"][field]["range_run_p95_ms"] for g in groups])
                ax.plot(x, y, marker="o", markersize=4, color=color, label=label)
                ax.fill_between(x, bounds[:, 0], bounds[:, 1], color=color, alpha=0.12)
            for capacity in (16, 32, 64, 128):
                ax.axvline(capacity, color="#64748b", linestyle=":", linewidth=0.7, alpha=0.5)
            ax.set_title(f"{family.capitalize()} FP32 graphs")
            ax.set_xlabel("Natural token length, including special tokens")
            ax.set_ylabel(ylabel)
            ax.set_ylim(bottom=0)
            ax.set_xticks([9, 16, 32, 64, 96, 128])
            ax.grid(alpha=0.12)
            ax.legend(fontsize=8)
        fig.suptitle(
            f"Matched synthetic diagnostic: {plan['threads']} threads, {plan['blocks']} process blocks"
        )
        fig.text(
            0.5,
            0.02,
            "Bands show full process-run ranges, not confidence intervals. Pretokenized inputs; no task-accuracy claim.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.94))
        fig.savefig(directory / f"{filename}.png", dpi=180)
        fig.savefig(directory / f"{filename}.svg")
        plt.close(fig)

    contrasts = [
        ("padding_cost", "Dynamic padded / natural"),
        ("fixed_shape_effect", "Dynamic padded / fixed padded"),
        ("net_fixed_effect", "Dynamic natural / fixed padded"),
    ]
    fig, axes = plt.subplots(len(families), 3, figsize=(14, 8), squeeze=False)
    for row, family in enumerate(families):
        for column, (contrast, title) in enumerate(contrasts):
            ax = axes[row, column]
            groups = sorted(
                [
                    g
                    for g in summary["paired_ratios"]
                    if g["family"] == family and g["contrast"] == contrast
                ],
                key=lambda g: g["length"],
            )
            x = [g["length"] for g in groups]
            medians = [g["median_ratio"] for g in groups]
            bounds = np.array(
                [
                    [
                        min(v["ratio"] for v in g["paired_blocks"]),
                        max(v["ratio"] for v in g["paired_blocks"]),
                    ]
                    for g in groups
                ]
            )
            ax.plot(x, medians, color="#2563eb", marker="o", markersize=3)
            ax.fill_between(x, bounds[:, 0], bounds[:, 1], color="#2563eb", alpha=0.12)
            for g in groups:
                ax.scatter(
                    [g["length"]] * len(g["paired_blocks"]),
                    [v["ratio"] for v in g["paired_blocks"]],
                    s=10,
                    color="#64748b",
                    alpha=0.5,
                )
            ax.axhline(1, color="#111827", linestyle="--", linewidth=1)
            ax.set_title(f"{family.capitalize()}: {title}", fontsize=10)
            ax.set_yscale("log", base=2)
            ax.set_xticks([9, 16, 32, 64, 96, 128])
            ax.set_xlabel("Natural token length")
            ax.set_ylabel("Paired process-run p95 ratio")
            ax.grid(alpha=0.12)
    fig.suptitle("Padding and fixed-shape effects: ratios above one favor the denominator")
    fig.text(
        0.5,
        0.02,
        "Dots are matched process blocks; lines are medians; bands are full ranges. No confidence intervals.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(directory / "paired-ratios.png", dpi=180)
    fig.savefig(directory / "paired-ratios.svg")
    plt.close(fig)

    ordered = sorted(runs, key=lambda run: run["created_at_utc"])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(1, len(ordered) + 1)
    ax.scatter(
        x,
        [r["system_cpu_before_percent"] for r in ordered],
        s=18,
        label="System CPU before worker sessions",
    )
    ax.plot(
        x,
        [
            float(np.mean([s["system_cpu_percent"] for s in r["cpu_observations"]]))
            if r["cpu_observations"]
            else np.nan
            for r in ordered
        ],
        marker=".",
        color="#059669",
        label="Mean system CPU during sessions and requests",
    )
    ax.set_xlabel("Process order (all retained)")
    ax.set_ylabel("System CPU utilization (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.12)
    ax.legend(fontsize=9)
    ax.set_title("Recorded system load includes the benchmark worker")
    interrupted = [
        index
        for index, run in enumerate(ordered, start=1)
        if telemetry_gap(run) is None or telemetry_gap(run) > 10
    ]
    for index in interrupted:
        ax.axvline(index, color="#dc2626", linestyle=":", linewidth=1)
        ax.annotate("telemetry gap", (index, 90), rotation=90, color="#dc2626", fontsize=8)
    fig.text(
        0.5,
        0.01,
        "No temperature or power-watt telemetry; fixed rests do not ensure equal temperatures.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(directory / "system-load.png", dpi=180)
    fig.savefig(directory / "system-load.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.diagnostic.resolve(), args.output.resolve()
    plan, summary = read_json(source / "plan.json"), read_json(source / "summary.json")
    if summary["status"] != "complete" or summary["plan_sha256"] != sha256(source / "plan.json"):
        raise ValueError("Diagnostic is incomplete or its plan changed")
    if plan["code_hashes"] != code_hashes():
        raise ValueError("Diagnostic measurement code changed")
    if output.exists():
        raise ValueError("Archive exists; choose a new output directory")
    results = []
    paths = sorted((source / "measurements").glob("*.json"))
    for path in paths:
        if path.name.endswith(".request.json"):
            continue
        result = read_json(path)
        request = path.with_name(path.stem + ".request.json")
        if (
            result["request_sha256"] != sha256(request)
            or result["plan_sha256"] != sha256(source / "plan.json")
            or result["references_sha256"] != sha256(source / "references.json")
        ):
            raise ValueError("A diagnostic measurement belongs to different inputs")
        results.append(result)
    results.sort(key=lambda result: (result["block"], result["family"], result["capacity"]))
    reconstructed = summarize(results)
    if any(summary[key] != value for key, value in reconstructed.items()):
        raise ValueError("Diagnostic summary does not match the recorded processes")
    if len(results) != plan["blocks"] * len(plan["families"]) * 4:
        raise ValueError("Diagnostic process coverage is incomplete")
    if any(group["process_runs"] != plan["blocks"] for group in summary["groups"]):
        raise ValueError("Diagnostic length coverage is incomplete")
    output.mkdir(parents=True)
    for name in ("plan.json", "references.json", "summary.json"):
        shutil.copyfile(source / name, output / name)
    requests = output / "measurement-requests"
    requests.mkdir()
    for path in paths:
        if path.name.endswith(".request.json"):
            shutil.copyfile(path, requests / path.name)
    metrics = []
    rows = []
    for result in results:
        metrics.append({key: value for key, value in result.items() if key != "samples"})
        rows.extend(
            {
                "family": result["family"],
                "capacity": result["capacity"],
                "block": result["block"],
                **sample,
            }
            for sample in result["samples"]
        )
    write_json(output / "process-metrics.json", metrics)
    with (output / "requests.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output / "archive-metadata.json",
        {
            "measurement_code_commit": plan["measurement_code_commit"],
            "reporting_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "processes": len(results),
            "timed_requests": len(rows),
            "all_checks_passed": summary["all_checks_passed"],
            "all_power_conditions_consistent": summary["all_power_conditions_consistent"],
            "telemetry_gap_threshold_seconds": 10,
            "long_telemetry_gap_processes": [
                {
                    "family": run["family"],
                    "capacity": run["capacity"],
                    "block": run["block"],
                    "max_gap_seconds": telemetry_gap(run),
                }
                for run in results
                if telemetry_gap(run) is None or telemetry_gap(run) > 10
            ],
            "max_absolute_error": max(
                (
                    row["max_absolute_error"]
                    for row in rows
                    if row["max_absolute_error"] is not None
                ),
                default=None,
            ),
            "note": "Synthetic pretokenized inputs; no classification accuracy or natural-workload p95 claim.",
        },
    )
    render(output, summary, results, plan)
    write_json(
        output / "checksums.json",
        {
            p.relative_to(output).as_posix(): sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "checksums.json"
        },
    )
    print(f"Archived matched length diagnostics to {output}")


if __name__ == "__main__":
    main()
