"""Render the archived student and resident-cascade accuracy-latency comparison."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from etdl.io import read_json


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("results/student-study"))
    return parser.parse_args()


def point(ax, x, y, low, high, label, color, marker, offset):
    ax.errorbar(
        x,
        y,
        xerr=[[x - low], [high - x]],
        fmt=marker,
        markersize=8,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        ecolor=color,
        elinewidth=1.5,
        capsize=3,
        color=color,
        zorder=3,
    )
    ax.annotate(
        label,
        (x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=9,
        color="#182536",
    )


def main():
    archive = arguments().archive.resolve()
    validation = read_json(archive / "validation-summary.json")["metrics"]
    gate = read_json(archive / "gate/gate-validation.json")
    solo = read_json(archive / "cpu-timing/summary.json")["methods"]
    resident = read_json(archive / "cascade-timing/summary.json")["methods"]
    solo_rows = [
        ("teacher", "Full teacher", "#355c92", "s", (7, 6)),
        ("supervised", "3-layer supervised", "#8d6ab3", "o", (7, -19)),
        ("distilled", "3-layer distilled", "#178781", "o", (7, 13)),
        ("four_layer_distilled", "4-layer distilled", "#d28731", "o", (7, -13)),
    ]
    resident_rows = [
        ("teacher_only", "Full teacher", 794 / 872, "#355c92", "s", (7, 6)),
        ("student_only", "4-layer student", 765 / 872, "#d28731", "o", (7, -13)),
        (
            "selected_gate",
            "Length + cost gate",
            gate["selected"]["accuracy"],
            "#c45663",
            "D",
            (7, -15),
        ),
        (
            "confidence_gate",
            "Confidence gate",
            gate["confidence_control"]["correct"] / 872,
            "#178781",
            "D",
            (-150, 21),
        ),
    ]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#182536",
            "text.color": "#182536",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    for name, label, color, marker, offset in solo_rows:
        item = solo[name]
        median = item["median_of_end_to_end_medians_ms"]
        low, high = item["end_to_end_median_range_ms"]
        point(
            axes[0],
            median,
            validation[name]["accuracy"] * 100,
            low,
            high,
            label,
            color,
            marker,
            offset,
        )
    for name, label, accuracy, color, marker, offset in resident_rows:
        item = resident[name]
        median = item["median_of_end_to_end_medians_ms"]
        low, high = item["end_to_end_median_range_ms"]
        point(axes[1], median, accuracy * 100, low, high, label, color, marker, offset)
    axes[1].axhline(
        (794 / 872 - 0.01) * 100, linestyle="--", linewidth=1, color="#9ca5b0", zorder=1
    )
    axes[1].text(
        31.0,
        (794 / 872 - 0.01) * 100 + 0.3,
        "1-point accuracy budget",
        fontsize=8,
        ha="right",
        color="#657080",
    )
    for ax, title in zip(
        axes,
        ("A  One model resident", "B  Student and teacher resident"),
        strict=True,
    ):
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Median end-to-end CPU request time (ms)")
        ax.set_xlim(5, 32)
        ax.grid(axis="both", alpha=0.16)
    axes[0].set_ylabel("SST-2 validation accuracy (%)")
    axes[0].set_ylim(81, 94)
    fig.suptitle(
        "Training improves the speed-accuracy frontier; the length-cost gate misses accuracy",
        fontsize=14,
        fontweight="bold",
        x=0.04,
        ha="left",
        y=1.02,
    )
    fig.text(
        0.04,
        -0.02,
        "872 sentences; 4 ORT threads; 3 shuffled process blocks. Whiskers are min-max process medians, not confidence intervals. "
        "Panel B keeps both models loaded in every process.",
        fontsize=8,
        color="#657080",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.98), w_pad=3)
    for suffix in ("png", "svg"):
        fig.savefig(archive / f"accuracy-latency.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
