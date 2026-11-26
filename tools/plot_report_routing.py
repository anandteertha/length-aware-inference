"""Render the report figures from the frozen grid and paired timing archives."""

import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results/student-study"
FIGURES = ROOT / "report/figures"


def load(path):
    return json.loads((STUDY / path).read_text(encoding="utf-8"))


def draw_grid():
    plan = load("gate-grid/plan.json")
    result = load("gate-grid/results.json")
    targets = 100 * np.array(plan["calibration_fallback_targets"])
    exponents = plan["exponents"]
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(5.5, 2.55), layout="constrained")
    norm = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5)
    for ax, key, title in zip(
        axes,
        ("sst2_development", "stanford_retrospective"),
        ("GLUE SST-2", "Stanford SST"),
        strict=True,
    ):
        cohort = result[key]
        gain = 100 * (np.array(cohort["correct"]) - cohort["confidence"]["correct"]) / cohort["n"]
        heat = ax.pcolormesh(
            targets, exponents, gain, shading="nearest", cmap="RdBu", norm=norm, rasterized=True
        )
        ax.set_title(title, weight="bold")
        ax.set_xticks([5, 10, 15, 20, 25])
        ax.set_yticks(np.linspace(0, 1, 6))
        for setting, marker, color, size in (
            (plan["selected"], "*", "#ffda47", 66),
            (result["retrospective_choice"], "D", "#5b2f91", 23),
        ):
            ax.scatter(
                100 * setting["calibration_fallback_target"],
                setting["exponent"],
                marker=marker,
                color=color,
                edgecolors="black",
                linewidth=0.55,
                s=size,
                zorder=5,
            )
    axes[0].contour(
        targets,
        exponents,
        result["eligible_on_sst2"],
        levels=[0.5],
        colors="black",
        linewidths=0.75,
    )
    axes[0].set_ylabel("Teacher cost exponent")
    fig.supxlabel("Target fallback on calibration set (%)", fontsize=9)
    fig.colorbar(heat, ax=axes, extend="both", label="Accuracy change (pp)", ticks=[-1.5, 0, 1.5])
    fig.savefig(FIGURES / "gate-grid.pdf", dpi=250, metadata={"CreationDate": None})
    plt.close(fig)


def draw_latency():
    comparison = load("gate-grid-timing/timing/comparison.json")
    values = comparison["paired_end_to_end"]
    order = comparison["process_order"][::2]
    first = ["Cost-aware" if name.startswith("selected_gate") else "Confidence" for name in order]
    rows = np.arange(len(first))
    fig, ax = plt.subplots(figsize=(5.5, 2.12), layout="constrained")
    ax.axvline(0, color="#6e737a", linestyle="--", linewidth=0.8)
    ax.xaxis.grid(True, color="#e7e8eb", linewidth=0.55)
    for key, offset, marker, color, label in (
        ("median", -0.12, "o", "#155c89", "Median"),
        ("p95", 0.12, "D", "#b94f22", "p95"),
    ):
        ax.scatter(
            values[key]["change_pct_by_block"],
            rows + offset,
            s=28,
            marker=marker,
            color=color,
            label=label,
            zorder=3,
        )
    ax.set_yticks(rows, [f"{i + 1}   {name} first" for i, name in enumerate(first)])
    ax.invert_yaxis()
    ax.set_xticks([-6, -4, -2, 0, 2, 4, 6])
    ax.set_xlabel("Cost-aware latency change vs. confidence-only (%)")
    ax.set_ylim(len(rows) - 0.55, -0.45)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(FIGURES / "paired-latency.pdf", metadata={"CreationDate": None})
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    draw_grid()
    draw_latency()
