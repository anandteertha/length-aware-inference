"""Plot the descriptive process-block break-even map without interpolation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from etdl.io import read_json

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "artifacts/breakeven-map"


def errorbar(axis, rows, field, color, marker, label, shift=0.0):
    x = np.asarray([row["length"] + shift for row in rows], dtype=float)
    middle = np.asarray([row["median"][field] for row in rows], dtype=float)
    lower = np.asarray([row["range"][field][0] for row in rows], dtype=float)
    upper = np.asarray([row["range"][field][1] for row in rows], dtype=float)
    axis.errorbar(
        x,
        middle,
        yerr=np.vstack([middle - lower, upper - middle]),
        fmt=marker,
        color=color,
        ecolor=color,
        alpha=0.9,
        markersize=5,
        elinewidth=1,
        capsize=2,
        label=label,
    )


def main():
    summary = read_json(DIRECTORY / "summary.json")
    lengths = sorted({row["length"] for row in summary["map"]})
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, layout="constrained")
    for line, backend in enumerate(("ort", "openvino")):
        rows = [row for row in summary["map"] if row["backend"] == backend]
        components, net = axes[line]
        errorbar(components, rows, "padding_cost_ms", "#d95f02", "o", "Padding cost", -0.16)
        errorbar(components, rows, "shape_gain_ms", "#1b9e77", "s", "Shape gain", 0.16)
        errorbar(net, rows, "net_fixed_gain_ms", "#253b6e", "o", "Net fixed gain")
        for axis in (components, net):
            axis.axhline(0, color="#444444", linewidth=0.9)
            axis.axvline(16.5, color="#aaaaaa", linestyle=":", linewidth=0.8)
            axis.axvline(32.5, color="#aaaaaa", linestyle=":", linewidth=0.8)
            axis.grid(axis="y", alpha=0.22)
            axis.set_xticks(lengths)
            axis.set_ylabel("Difference in p95 (ms)")
        components.set_title(f"{backend.upper()}: components")
        net.set_title(f"{backend.upper()}: deployable net effect")
        components.legend(loc="best", frameon=False, fontsize=9)
        net.legend(loc="best", frameon=False, fontsize=9)
    for axis in axes[-1]:
        axis.set_xlabel("Natural token length; bucket changes after 16 and 32")
    figure.suptitle(
        "Real-text fixed-shape break-even map | prepared-input p95 | 3 process blocks",
        fontsize=13,
    )
    for suffix in ("png", "svg"):
        figure.savefig(DIRECTORY / f"break-even-map.{suffix}", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
