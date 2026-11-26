"""Test a guard against teacher-induced errors around the original cutoff."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
from evaluate_gate import paired_bootstrap_ci
from soft_cost_gate import ARCHIVE, ORIGINAL, ROOT, arrays
from study_gate_signals import (
    FEATURE_COUNTS,
    cohort,
    feature_matrix,
    get_older_cohorts,
    infer_imdb,
    logistic,
    margins,
    mask_ranked,
    percentile_rank,
    route_metrics,
    sigmoid,
)

from etdl.io import read_json, sha256, write_json

PRIOR = ARCHIVE / "gate-signals"
FAMILIES = ("harm_only", "fix_minus_two_harms")


def quality(x, model, family):
    harm = sigmoid(x @ model["harm"])
    if family == "harm_only":
        return -harm
    return sigmoid(x @ model["fix"]) - 2 * harm


def adjust_score(margin, values, anchors, center, width, weight):
    """Prevent any inside-window request from overtaking an outside request."""
    base = center - margin
    inside = np.abs(base) <= width
    shift = width * weight * (2 * percentile_rank(values, anchors) - 1)
    bound = np.nextafter(width, 0)
    return np.where(inside, np.clip(base + shift, -bound, bound), base)


def read_reference_imdb():
    summary = read_json(PRIOR / "imdb.json")
    path = PRIOR / "imdb-logits.json"
    if sha256(path) != summary["logits_sha256"]:
        raise ValueError("Earlier IMDb logits differ from the archive")
    return read_json(path)["rows"]


def freeze(output, gate):
    params = gate["parameters"]
    old = get_older_cohorts(output, gate)
    confidence = read_json(PRIOR / "plan.json")["confidence_point"]
    intercept, slope = params["coefficients"]["confidence"]
    if slope >= 0:
        raise ValueError("Confidence score must decrease as margin increases")
    center = (
        params["feature_mean"][0]
        + params["feature_scale"][0] * (confidence["threshold"] - intercept) / slope
    )
    fit = old["fit"] + old["stanford_development"]
    in_window = np.abs(margins(fit)[0] - center) <= 2
    fit = [r for r, keep in zip(fit, in_window, strict=True) if keep]
    matrix, means, scales = feature_matrix(fit)
    teacher, student, _ = arrays(fit, params)
    fix, harm = teacher & ~student, student & ~teacher
    development = {
        "sst2_calibration": cohort(old["calibration"], params, confidence, means, scales),
        "imdb_development": cohort(read_reference_imdb(), params, confidence, means, scales),
    }
    grid = []
    models = []
    for (features, count), level in product(FEATURE_COUNTS.items(), (10, 40)):
        x = matrix[:, : count + 1]
        penalty = np.diag([0] + [level] * count)
        model = {
            "features": features,
            "penalty": level,
            "fix": logistic(x, fix, penalty, tolerance=1e-5),
            "harm": logistic(x, harm, penalty, tolerance=1e-5),
        }
        index = len(models)
        models.append(model)
        for family in FAMILIES:
            anchors = np.quantile(quality(x, model, family), np.linspace(0, 1, 101)).tolist()
            values_by_split = {
                name: quality(data["matrix"][:, : count + 1], model, family)
                for name, data in development.items()
            }
            for width, weight in product((0.75, 1.5), (0.5, 1.0, 2.0)):
                row = {
                    "family": family,
                    "model_index": index,
                    "width": width,
                    "weight": weight,
                    "anchors": anchors,
                    "selection": {},
                }
                for name, data in development.items():
                    scores = adjust_score(
                        data["margin"], values_by_split[name], anchors, center, width, weight
                    )
                    chosen, ranked = mask_ranked(data, scores)
                    result = route_metrics(data, chosen)
                    result["changed_routes"] = int((chosen != data["baseline"]).sum())
                    row["selection"][name] = result
                    if name == "imdb_development":
                        boundary = ranked[data["quota"] - 1]
                        row["cutoff"] = float(scores[boundary])
                        row["cutoff_margin"] = float(data["margin"][boundary])
                grid.append(row)

    def preference(i):
        metrics = grid[i]["selection"].values()
        return (
            -sum(m["correct"] for m in metrics),
            sum(m["harms"] for m in metrics),
            sum(m["changed_routes"] for m in metrics),
            i,
        )

    finalists = {
        family: min((i for i, row in enumerate(grid) if row["family"] == family), key=preference)
        for family in FAMILIES
    }
    plan = {
        "gate_sha256": sha256(ORIGINAL),
        "imdb_development_logits_sha256": sha256(PRIOR / "imdb-logits.json"),
        "fit_rows": len(fit),
        "fit_fixes": int(fix.sum()),
        "fit_harms": int(harm.sum()),
        "margin_center": center,
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "confidence_point": confidence,
        "development_baselines": {
            name: route_metrics(data, data["baseline"]) for name, data in development.items()
        },
        "primary": "harm_only",
        "finalists": finalists,
        "models": models,
        "grid": grid,
    }
    write_json(output / "plan.json", plan)
    print(f"Froze {len(grid)} variants: {finalists}, fit rows {len(fit)}", flush=True)


def evaluate(output, plan, gate, rows):
    data = cohort(
        rows,
        gate["parameters"],
        plan["confidence_point"],
        plan["feature_means"],
        plan["feature_scales"],
    )
    base_mask = data["baseline"]
    teacher, student = data["teacher"], data["student"]
    base_correct = np.where(base_mask, teacher, student)
    fixes, harms = teacher & ~student, student & ~teacher
    results = {}
    for family, index in plan["finalists"].items():
        choice = plan["grid"][index]
        model = plan["models"][choice["model_index"]]
        x = data["matrix"][:, : FEATURE_COUNTS[model["features"]] + 1]
        values = adjust_score(
            data["margin"],
            quality(x, model, family),
            choice["anchors"],
            plan["margin_center"],
            choice["width"],
            choice["weight"],
        )
        matched, _ = mask_ranked(data, values)
        fixed = (values > choice["cutoff"]) | (
            (values == choice["cutoff"]) & (data["margin"] <= choice["cutoff_margin"])
        )
        removed, added = base_mask & ~matched, matched & ~base_mask
        if removed.sum() != added.sum():
            raise ValueError("Comparison must use the same number of teacher calls")
        correct = np.where(matched, teacher, student)
        fixed_quota = int(fixed.sum())
        confidence_at_quota = np.zeros(len(rows), dtype=bool)
        confidence_at_quota[np.argsort(data["margin"], kind="stable")[:fixed_quota]] = True
        inside = np.abs(data["margin"] - plan["margin_center"]) <= choice["width"]
        harm_score = sigmoid(x @ model["harm"])
        risky = harm_score[inside & harms]
        ordinary = harm_score[inside & ~harms]
        differences = risky[:, None] - ordinary
        harm_auc = (
            float((differences > 0).mean() + 0.5 * (differences == 0).mean())
            if differences.size
            else None
        )
        results[family] = {
            "matched": route_metrics(data, matched),
            "fixed": route_metrics(data, fixed),
            "swaps": int(removed.sum()),
            "removed_fixes": int((removed & fixes).sum()),
            "removed_harms": int((removed & harms).sum()),
            "added_fixes": int((added & fixes).sum()),
            "added_harms": int((added & harms).sum()),
            "wins_vs_confidence": int((correct & ~base_correct).sum()),
            "losses_vs_confidence": int((base_correct & ~correct).sum()),
            "paired_95pct_accuracy_interval": paired_bootstrap_ci(correct, base_correct),
            "diagnostics": {
                "confidence_at_fixed_call_quota": route_metrics(data, confidence_at_quota),
                "window_examples": int(inside.sum()),
                "window_harms": int((inside & harms).sum()),
                "window_fixes": int((inside & fixes).sum()),
                "window_harm_auc": harm_auc,
            },
        }
    report = {
        "plan_sha256": sha256(output / "plan.json"),
        "logits_sha256": sha256(output / "imdb-logits.json"),
        "n": len(rows),
        "student_correct": int(student.sum()),
        "teacher_correct": int(teacher.sum()),
        "confidence": route_metrics(data, base_mask),
        "finalists": results,
    }
    write_json(output / "imdb.json", report)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "imdb"))
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/student-study/teacher-harm"
    )
    args = parser.parse_args()
    target = args.output / ("plan.json" if args.stage == "freeze" else "imdb.json")
    if target.exists():
        raise ValueError(f"Output already exists: {target}")
    gate = read_json(ORIGINAL)
    if args.stage == "freeze":
        freeze(args.output, gate)
        return
    plan = read_json(args.output / "plan.json")
    if plan["gate_sha256"] != sha256(ORIGINAL):
        raise ValueError("Original gate differs from the frozen plan")
    rows = infer_imdb(
        args.output,
        gate,
        split="train",
        boundaries=(113,),
        max_tokens=176,
        per_cell=240,
        seed=2067,
        extra_exclusions=(PRIOR / "imdb-selection.json",),
    )
    evaluate(args.output, plan, gate, rows)


if __name__ == "__main__":
    main()
