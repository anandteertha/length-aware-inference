"""Benchmark the frozen two-split grid choice against confidence-only on SST-2."""

import argparse
import statistics
from pathlib import Path

from benchmark_cascade import parent, require_ac
from soft_cost_gate import ARCHIVE, ORIGINAL, ROOT

from etdl.config import load_config
from etdl.io import read_json, sha256, write_json

GRID = ARCHIVE / "gate-grid"
BLOCKS = 4
METHODS = ("selected_gate", "confidence_gate")


def summarize(output, expected):
    reports = {method: [] for method in METHODS}
    for method, (correct, calls) in expected.items():
        for block in range(BLOCKS):
            report = read_json(output / f"{method}-block-{block}.json")
            if (
                report["method"] != method
                or report["block"] != block
                or report["examples"] != 872
                or report["correct"] != correct
                or report["fallback_count"] != calls
                or report["changed_predictions_vs_frozen_logits"]
                or report["route_disagreements_vs_frozen_logits"]
                or not (report["power_plugged_start"] and report["power_plugged_end"])
            ):
                raise ValueError(f"Timing audit failed for {method}, block {block}")
            reports[method].append(report)
    timing = read_json(output / "summary.json")
    benchmark = read_json(output / "benchmark-plan.json")
    gate = read_json(output.parent / "gate" / "gate-plan.json")
    if (
        timing["benchmark_plan_sha256"] != sha256(output / "benchmark-plan.json")
        or benchmark["selected_point"] != gate["curves"]["joint_per_cost"][0]
    ):
        raise ValueError("Timing results differ from the frozen benchmark plan")
    comparisons = {}
    for measure in ("median", "mean", "p95"):
        key = f"end_to_end_{measure}_ms"
        paired_change = [
            100 * (candidate[key] / baseline[key] - 1)
            for candidate, baseline in zip(*(reports[method] for method in METHODS), strict=True)
        ]
        comparisons[measure] = {
            "change_pct_by_block": paired_change,
            "median_paired_change_pct": statistics.median(paired_change),
            "blocks_where_candidate_was_faster": sum(change < 0 for change in paired_change),
        }
    result = {
        "run_plan_sha256": sha256(output.parent / "plan.json"),
        "benchmark_summary_sha256": sha256(output / "summary.json"),
        "process_order": timing["process_order"],
        "correct_and_calls": expected,
        "paired_end_to_end": comparisons,
        "max_peak_working_set_mib": {
            method: max(row["process_peak_wset_mib"] for row in rows)
            for method, rows in reports.items()
        },
    }
    write_json(output / "comparison.json", result)
    print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/student-study/gate-grid-timing"
    )
    output = parser.parse_args().output.resolve()
    require_ac()
    grid = read_json(GRID / "results.json")
    freeze = read_json(GRID / "imdb-plan.json")
    if (
        freeze["grid_results_sha256"] != sha256(GRID / "results.json")
        or grid["plan_sha256"] != sha256(GRID / "plan.json")
        or read_json(GRID / "plan.json")["gate_sha256"] != sha256(ORIGINAL)
    ):
        raise ValueError("Archived grid and frozen choices differ")
    policy = freeze["policies"]["retrospective_choice"]
    if policy != grid["retrospective_choice"]:
        raise ValueError("The frozen policy differs from the timing candidate")
    original = read_json(ORIGINAL)
    config = load_config(ROOT / "configs/default.toml")
    export_dir = ROOT / "artifacts/student-study/onnx-verified"
    model_hashes = {
        "teacher": sha256(config.artifact_dir / "models/dynamic.onnx"),
        "student": sha256(export_dir / "four_layer_distilled.onnx"),
    }
    verified = read_json(ARCHIVE / "onnx-export-summary.json")["four_layer_distilled"]
    if model_hashes != {
        "teacher": original["teacher_model_sha256"],
        "student": verified["onnx_sha256"],
    }:
        raise ValueError("ONNX models differ from archived exports")
    dev = grid["sst2_development"]
    i, j = policy["grid_index"]
    expected = {
        "selected_gate": (dev["correct"][i][j], dev["calls"][i][j]),
        "confidence_gate": (dev["confidence"]["correct"], dev["confidence"]["calls"]),
    }
    output.mkdir(parents=True, exist_ok=False)
    original["parameters"]["teacher_cost_exponent"] = policy["exponent"]
    original["selected_policy"] = "joint_per_cost"
    original["selected_point_index"] = 0
    original["curves"]["joint_per_cost"] = [policy["point"]]
    write_json(output / "gate/gate-plan.json", original)
    write_json(
        output / "plan.json",
        {
            "grid_choice_sha256": sha256(GRID / "imdb-plan.json"),
            "gate_plan_sha256": sha256(output / "gate/gate-plan.json"),
            "model_sha256": model_hashes,
            "candidate": policy,
            "primary_latency_metric": "paired end-to-end median request latency",
            "expected_correct_and_calls": expected,
            "examples_per_block": 872,
            "blocks": BLOCKS,
            "threads": 4,
            "require_ac": True,
            "balanced_order": True,
        },
    )
    parent(
        argparse.Namespace(
            study=ARCHIVE,
            gate=output / "gate",
            export=export_dir,
            output=output / "timing",
            threads=4,
            blocks=BLOCKS,
            limit=872,
            methods=METHODS,
            balanced_order=True,
            require_ac=True,
        )
    )
    summarize(output / "timing", expected)


if __name__ == "__main__":
    main()
