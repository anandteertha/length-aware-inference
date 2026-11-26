"""Sweep the cost exponent and fixed fallback cutoff, then plot the grid."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from evaluate_gate import paired_bootstrap_ci
from fit_gate import route, score
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from soft_cost_gate import ARCHIVE, ORIGINAL, ROOT, arrays
from sweep_cost_exponent import load_rows, metrics

from etdl.io import read_json, sha256, write_json

matplotlib.use("Agg")

EXPONENTS = np.linspace(0.0, 1.0, 51)
FALLBACK = np.linspace(0.025, 0.25, 91)


def evaluate_grid(rows, params, confidence_point, thresholds):
    teacher, student, cost = arrays(rows, params)
    long = np.asarray([r["tokens"] >= 33 for r in rows])
    confidence = route(score(rows, "confidence", params), confidence_point)
    gain = score(rows, "joint", params)
    student_correct = int(student.sum())
    delta = teacher.astype(int) - student.astype(int)
    by_exponent = {"correct": [], "calls": [], "cost": []}
    for exponent, cutoffs in zip(EXPONENTS, thresholds, strict=True):
        fallback = gain / cost**exponent >= cutoffs[:, None]
        by_exponent["correct"].append((student_correct + fallback @ delta).tolist())
        by_exponent["calls"].append(fallback.sum(axis=1).tolist())
        by_exponent["cost"].append(((fallback @ cost) / len(rows)).tolist())
    return {
        "n": len(rows),
        "teacher_correct": int(teacher.sum()),
        "confidence": metrics(confidence, teacher, student, long, cost),
        **by_exponent,
    }


def selected_metrics(rows, params, confidence_point, selection):
    teacher, student, cost = arrays(rows, params)
    long = np.asarray([r["tokens"] >= 33 for r in rows])
    confidence = route(score(rows, "confidence", params), confidence_point)
    values = score(
        rows, "joint_per_cost", {**params, "teacher_cost_exponent": selection["exponent"]}
    )
    chosen = route(values, selection["point"])
    return {
        **metrics(chosen, teacher, student, long, cost),
        "paired_95pct_interval_vs_confidence": paired_bootstrap_ci(
            np.where(chosen, teacher, student), np.where(confidence, teacher, student)
        ),
    }


def grid_point(i, j, thresholds):
    return {
        "exponent": float(EXPONENTS[i]),
        "calibration_fallback_target": float(FALLBACK[j]),
        "point": {"route_mode": "threshold", "threshold": float(thresholds[i, j])},
        "grid_index": [int(i), int(j)],
    }


def transfer_shortlist(development, stanford, eligible, thresholds):
    dev_correct, sst_correct = (np.asarray(x["correct"]) for x in (development, stanford))
    reference = stanford["confidence"]
    within_both_budgets = (
        eligible
        & (np.asarray(stanford["calls"]) <= reference["calls"])
        & (np.asarray(stanford["cost"]) <= reference["teacher_cost_proxy_ms_per_request"] + 1e-10)
    )
    candidates = np.argwhere(
        within_both_budgets
        & (dev_correct > development["confidence"]["correct"])
        & (sst_correct >= reference["correct"])
    )
    if not len(candidates):
        return None
    index = min(
        (tuple(pair) for pair in candidates),
        key=lambda pair: (
            -dev_correct[pair],
            -sst_correct[pair],
            development["cost"][pair[0]][pair[1]],
            *pair,
        ),
    )
    return grid_point(*index, thresholds)


def check_imdb(output, gate, confidence_point):
    from study_gate_signals import infer_imdb

    plan_path = output / "plan.json"
    grid_report_path = output / "results.json"
    plan = read_json(plan_path)
    grid = read_json(grid_report_path)
    if plan["gate_sha256"] != sha256(ORIGINAL) or grid["plan_sha256"] != sha256(plan_path):
        raise ValueError("The archived grid and its plan do not match")
    freeze_path = output / "imdb-plan.json"
    if freeze_path.exists():
        raise ValueError("The IMDb choices are already frozen in this output directory")
    selections = {"sst2_selection": plan["selected"]}
    if grid["retrospective_choice"] is not None:
        selections["retrospective_choice"] = grid["retrospective_choice"]
    exclusions = ("gate-signals/imdb-selection.json", "teacher-harm/imdb-selection.json")
    design = {
        "split": "test",
        "boundaries": (97, 129),
        "max_tokens": 160,
        "per_cell": 120,
        "seed": 2081,
    }
    write_json(
        freeze_path,
        {
            "grid_results_sha256": sha256(grid_report_path),
            "policies": selections,
            "imdb_design": design,
            "additional_exclusions": {name: sha256(ARCHIVE / name) for name in exclusions},
        },
    )
    rows = infer_imdb(
        output, gate, **design, extra_exclusions=tuple(ARCHIVE / x for x in exclusions)
    )
    params = gate["parameters"]
    teacher, student, cost = arrays(rows, params)
    confidence = route(score(rows, "confidence", params), confidence_point)
    long = np.asarray([r["tokens"] >= 33 for r in rows])
    result = {
        "plan_sha256": sha256(freeze_path),
        "selection_sha256": sha256(output / "imdb-selection.json"),
        "logits_sha256": sha256(output / "imdb-logits.json"),
        "n": len(rows),
        "teacher_correct": int(teacher.sum()),
        "student_correct": int(student.sum()),
        "confidence": metrics(confidence, teacher, student, long, cost),
        "policies": {
            name: selected_metrics(rows, params, confidence_point, policy)
            for name, policy in selections.items()
        },
    }
    write_json(output / "imdb.json", result)
    print(result, flush=True)


def plot_grid(development, transfer, eligible, selection, shortlist, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.6), layout="constrained")
    norm = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5)
    x = FALLBACK * 100
    y = EXPONENTS
    pick_x = selection["calibration_fallback_target"] * 100
    pick_y = selection["exponent"]
    for ax, cohort, name in (
        (axes[0], development, "SST-2 accuracy"),
        (axes[2], transfer, "Stanford accuracy"),
    ):
        delta = (
            100 * (np.asarray(cohort["correct"]) - cohort["confidence"]["correct"]) / cohort["n"]
        )
        heat = ax.pcolormesh(x, y, delta, cmap="RdBu", norm=norm, shading="nearest")
        fig.colorbar(heat, ax=ax, extend="both", label="Accuracy vs confidence-only (pp)")
        ax.set_title(name)
    calls = axes[1].pcolormesh(x, y, development["calls"], cmap="cividis", shading="nearest")
    fig.colorbar(calls, ax=axes[1], label="Teacher calls out of 872")
    axes[1].set_title("SST-2 teacher calls")
    for ax in axes[:2]:
        ax.contour(x, y, eligible, levels=[0.5], colors="#141a25", linewidths=1.25)
    for ax in axes:
        ax.scatter(
            pick_x,
            pick_y,
            s=155,
            marker="*",
            c="#ffdb48",
            edgecolors="#171923",
            linewidths=0.9,
            zorder=5,
            clip_on=False,
        )
        if shortlist is not None:
            ax.scatter(
                shortlist["calibration_fallback_target"] * 100,
                shortlist["exponent"],
                s=63,
                marker="D",
                c="#65319a",
                edgecolors="white",
                linewidths=0.95,
                zorder=5,
            )
        ax.set_xlabel("Target fallback on calibration set (%)")
        ax.set_xticks([2.5, 5, 10, 15, 20, 25])
        ax.set_yticks(np.linspace(0, 1, 6))
    axes[0].set_ylabel("Teacher cost exponent")
    handles = [
        Line2D([], [], color="#141a25", label="SST-2 eligibility boundary"),
        Line2D(
            [],
            [],
            marker="*",
            linestyle="none",
            markersize=11,
            markeredgecolor="#171923",
            markerfacecolor="#ffdb48",
            label="Selected on SST-2",
        ),
    ]
    if shortlist is not None:
        handles.append(
            Line2D(
                [],
                [],
                marker="D",
                linestyle="none",
                markersize=7,
                markeredgecolor="white",
                markerfacecolor="#65319a",
                label="Selected using both splits",
            )
        )
    axes[0].legend(
        handles=handles,
        loc="upper right",
        framealpha=0.96,
        fontsize=9,
    )
    fig.suptitle(
        "Choosing the teacher cost exponent and fallback cutoff", fontsize=17, weight="bold"
    )
    fig.savefig(output / "gate-grid.png", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/student-study/gate-grid")
    parser.add_argument(
        "--imdb", action="store_true", help="Audit both frozen grid choices on IMDb"
    )
    args = parser.parse_args()
    gate = read_json(ORIGINAL)
    benchmark = read_json(ARCHIVE / "cascade-timing/benchmark-plan.json")
    if sha256(ORIGINAL) != benchmark["gate_plan_sha256"]:
        raise ValueError("Original gate differs from the benchmark")
    params = gate["parameters"]
    confidence_point = benchmark["confidence_point"]
    if args.imdb:
        check_imdb(args.output, gate, confidence_point)
        return
    rows = load_rows(ARCHIVE / "gate/gate-records.json", gate["gate_records_sha256"])
    calibration = rows[gate["fit_count"] :]
    _, _, calibration_cost = arrays(calibration, params)
    calibration_gain = score(calibration, "joint", params)
    thresholds = np.asarray(
        [
            np.quantile(calibration_gain / calibration_cost**exponent, 1 - FALLBACK)
            for exponent in EXPONENTS
        ]
    )
    development_rows = load_rows(
        ARCHIVE / "validation-logits.json", benchmark["validation_logits_sha256"]
    )
    for row in development_rows:
        row["student_logits"] = row["four_layer_distilled_logits"]
    dev = evaluate_grid(development_rows, params, confidence_point, thresholds)
    correct, calls, cost = (np.asarray(dev[key]) for key in ("correct", "calls", "cost"))
    minimum_correct = dev["teacher_correct"] - dev["n"] // 100
    eligible = (
        (correct >= minimum_correct)
        & (calls <= dev["confidence"]["calls"])
        & (cost <= dev["confidence"]["teacher_cost_proxy_ms_per_request"] + 1e-10)
    )
    indices = np.argwhere(eligible)
    if not len(indices):
        raise ValueError("No grid setting meets the SST-2 accuracy and budget criteria")
    i, j = min(
        (tuple(index) for index in indices),
        key=lambda pair: (-correct[pair], cost[pair], calls[pair], *pair),
    )
    selection = grid_point(i, j, thresholds)
    stanford_hash = read_json(ARCHIVE / "score-search/test.json")["logits_sha256"]
    plan_path = args.output / "plan.json"
    write_json(
        plan_path,
        {
            "gate_sha256": sha256(ORIGINAL),
            "calibration_sha256": gate["gate_records_sha256"],
            "sst2_sha256": benchmark["validation_logits_sha256"],
            "stanford_sha256": stanford_hash,
            "exponents": EXPONENTS.tolist(),
            "calibration_fallback_targets": FALLBACK.tolist(),
            "thresholds": thresholds.tolist(),
            "minimum_sst2_correct": minimum_correct,
            "maximum_sst2_calls": dev["confidence"]["calls"],
            "maximum_sst2_teacher_cost_proxy_ms_per_request": dev["confidence"][
                "teacher_cost_proxy_ms_per_request"
            ],
            "selected": selection,
        },
    )
    print(f"Frozen SST-2 choice: {selection}", flush=True)

    stanford_rows = load_rows(ARCHIVE / "score-search/test-logits.json", stanford_hash)
    stanford = evaluate_grid(stanford_rows, params, confidence_point, thresholds)
    for report, subset in ((dev, development_rows), (stanford, stanford_rows)):
        report["selected"] = selected_metrics(subset, params, confidence_point, selection)
        if any(report["selected"][key] != report[key][i][j] for key in ("correct", "calls")):
            raise ValueError("The frozen policy did not reproduce its grid cell")
    shortlist = transfer_shortlist(dev, stanford, eligible, thresholds)
    write_json(
        args.output / "results.json",
        {
            "plan_sha256": sha256(plan_path),
            "eligible_on_sst2": eligible.tolist(),
            "maximum_sst2_correct_among_eligible": int(correct[eligible].max()),
            "sst2_development": dev,
            "stanford_retrospective": stanford,
            "retrospective_choice": shortlist,
        },
    )
    plot_grid(dev, stanford, eligible, selection, shortlist, args.output)
    print(
        f"Selected SST-2: {dev['selected']}\nSelected Stanford: {stanford['selected']}", flush=True
    )


if __name__ == "__main__":
    main()
