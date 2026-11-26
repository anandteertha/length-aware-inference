"""Exhaustively replay the cost exponent between ranking breakpoints."""

import argparse
from itertools import combinations, pairwise
from pathlib import Path

import numpy as np
from evaluate_gate import paired_bootstrap_ci
from fit_gate import route, score
from search_gate_formulas import top_mask
from soft_cost_gate import ARCHIVE, ORIGINAL, ROOT, arrays

from etdl.io import read_json, sha256, write_json

RANGE = (-1.0, 2.0)
CHECKPOINTS = (0.0, 0.25, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0)


def load_rows(path, expected_hash):
    if sha256(path) != expected_hash:
        raise ValueError(f"Archive differs from its recorded hash: {path}")
    return read_json(path)["rows"]


def breakpoints(gain, cost, calls):
    boundaries = {RANGE[0], 0.0, 1.0, RANGE[1]}
    # Only the top K rows within a constant-cost bin can appear in the global top K.
    groups = [(c, np.sort(gain[cost == c])[::-1][:calls]) for c in np.unique(cost)]
    for (a_cost, a), (b_cost, b) in combinations(groups, 2):
        for sign in (-1, 1):
            a_log = np.log(np.abs(a[sign * a > 0]))
            b_log = np.log(np.abs(b[sign * b > 0]))
            cross = (a_log[:, None] - b_log) / (np.log(a_cost) - np.log(b_cost))
            boundaries.update(cross[(RANGE[0] < cross) & (cross < RANGE[1])].tolist())
    return sorted(boundaries)


def metrics(mask, teacher, student, long, cost):
    predicted_correct = np.where(mask, teacher, student)
    fixes = mask & teacher & ~student
    harms = mask & student & ~teacher
    return {
        "correct": int(predicted_correct.sum()),
        "calls": int(mask.sum()),
        "long_correct": int(predicted_correct[long].sum()),
        "long_calls": int(mask[long].sum()),
        "teacher_fixes": int(fixes.sum()),
        "teacher_harms": int(harms.sum()),
        "teacher_cost_proxy_ms_per_request": float(cost[mask].sum() / len(mask)),
    }


def scan(rows, params, confidence_point):
    teacher, student, cost = arrays(rows, params)
    long = np.asarray([r["tokens"] >= 33 for r in rows])
    gain_delta = teacher.astype(int) - student.astype(int)
    student_correct = int(student.sum())
    gain = score(rows, "joint", params)
    confidence = route(score(rows, "confidence", params), confidence_point)
    calls = int(confidence.sum())
    cost_budget = float(cost[confidence].sum())
    # Account for floating-point roundoff when a prefix exactly matches the cost cap.
    cost_budget_with_tolerance = cost_budget + 1e-8
    max_cost_calls = min(len(cost), int(cost_budget_with_tolerance / cost.min()))
    boundaries = breakpoints(gain, cost, max(calls, max_cost_calls))
    regimes = []
    previous = None
    best_at_cost = None
    intervals = [(lo, hi) for lo, hi in pairwise(boundaries)]
    for lo, hi in [*intervals, (0.0, 0.0), (1.0, 1.0)]:
        exponent = (lo + hi) / 2
        values = gain / cost**exponent
        ranking = np.argsort(-values, kind="stable")
        if 0 <= lo and hi <= 1:
            affordable = ranking[:max_cost_calls]
            count = np.searchsorted(cost[affordable].cumsum(), cost_budget_with_tolerance, "right")
            cumulative_gain = np.insert(gain_delta[affordable[:count]].cumsum(), 0, 0)
            ordered_values = values[ranking[: count + 1]]
            splits_tie = np.flatnonzero(np.diff(ordered_values) == 0) + 1
            cumulative_gain[splits_tie] = -len(rows)
            calls_at_peak = int(cumulative_gain.argmax())
            correct = student_correct + int(cumulative_gain[calls_at_peak])
            if best_at_cost is None or (correct, -calls_at_peak) > (
                best_at_cost["correct"],
                -best_at_cost["calls"],
            ):
                cost_mask = np.zeros(len(cost), dtype=bool)
                cost_mask[affordable[:calls_at_peak]] = True
                if calls_at_peak == 0:
                    point = {"route_mode": "none", "threshold": None}
                elif calls_at_peak == len(cost):
                    point = {"route_mode": "all", "threshold": None}
                else:
                    point = {
                        "route_mode": "threshold",
                        "threshold": float(
                            ordered_values[calls_at_peak - 1 : calls_at_peak + 1].mean()
                        ),
                    }
                best_at_cost = {
                    "exponent": exponent,
                    "point": point,
                    **metrics(cost_mask, teacher, student, long, cost),
                }
        if lo == hi:
            continue
        mask = np.zeros(len(cost), dtype=bool)
        mask[ranking[:calls]] = True
        if lo not in (0.0, 1.0) and np.array_equal(mask, previous):
            regimes[-1]["upper"] = hi
        else:
            regimes.append(
                {"lower": lo, "upper": hi, **metrics(mask, teacher, student, long, cost)}
            )
        previous = mask

    result = {
        "n": len(rows),
        "n_long": int(long.sum()),
        "teacher_correct": int(teacher.sum()),
        "student_correct": student_correct,
        "confidence": metrics(confidence, teacher, student, long, cost),
        "primary_best_under_confidence_teacher_cost_proxy": best_at_cost,
        "checkpoints": {
            str(e): metrics(top_mask(gain / cost**e, calls), teacher, student, long, cost)
            for e in (*RANGE, *CHECKPOINTS)
        },
        "breakpoints": len(boundaries) - 4,
        "regimes": regimes,
    }
    primary = [r for r in regimes if 0 <= r["lower"] and r["upper"] <= 1]
    endpoints = [result["checkpoints"][str(e)] for e in (0.0, 1.0)]
    result["primary_max_correct"] = max(r["correct"] for r in primary + endpoints)
    evaluated = regimes + list(result["checkpoints"].values())
    result["sensitivity_max_correct"] = max(r["correct"] for r in evaluated)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/student-study/cost-exponent-sweep"
    )
    args = parser.parse_args()
    gate = read_json(ORIGINAL)
    benchmark = read_json(ARCHIVE / "cascade-timing/benchmark-plan.json")
    if sha256(ORIGINAL) != benchmark["gate_plan_sha256"]:
        raise ValueError("Original gate differs from the benchmark")
    dev = load_rows(ARCHIVE / "validation-logits.json", benchmark["validation_logits_sha256"])
    for row in dev:
        row["student_logits"] = row["four_layer_distilled_logits"]
    fit_cal = load_rows(ARCHIVE / "gate/gate-records.json", gate["gate_records_sha256"])
    calibration = fit_cal[gate["fit_count"] :]
    confidence_point = benchmark["confidence_point"]
    params = gate["parameters"]
    development = scan(dev, params, confidence_point)
    candidates = [
        {**r, "exponent": (r["lower"] + r["upper"]) / 2}
        for r in development["regimes"]
        if 0 <= r["lower"] and r["upper"] <= 1
    ]
    candidates.extend(
        {**development["checkpoints"][str(e)], "lower": e, "upper": e, "exponent": e}
        for e in (0.0, 1.0)
    )
    best = min(
        candidates,
        key=lambda r: (
            -r["correct"],
            r["teacher_cost_proxy_ms_per_request"],
            r["lower"] - r["upper"],
            r["exponent"],
        ),
    )
    exponent = best["exponent"]
    cal_score = score(calibration, "joint_per_cost", {**params, "teacher_cost_exponent": exponent})
    quantile = 1 - confidence_point["requested_fallback_fraction"]
    point = {"route_mode": "threshold", "threshold": float(np.quantile(cal_score, quantile))}
    stanford_hash = read_json(ARCHIVE / "score-search/test.json")["logits_sha256"]
    plan = {
        "original_gate_sha256": sha256(ORIGINAL),
        "validation_logits_sha256": benchmark["validation_logits_sha256"],
        "calibration_sha256": gate["gate_records_sha256"],
        "stanford_logits_sha256": stanford_hash,
        "sensitivity_exponents": list(RANGE),
        "selected_development_regime": best,
        "frozen_cutoff": point,
        "selected_under_development_teacher_cost_proxy": development[
            "primary_best_under_confidence_teacher_cost_proxy"
        ],
    }
    write_json(args.output / "plan.json", plan)
    print(f"Frozen GLUE choice: {best}", flush=True)

    dev_teacher, dev_student, dev_cost = arrays(dev, params)
    frozen_params = {**params, "teacher_cost_exponent": exponent}
    development["selected_fixed_cutoff"] = metrics(
        route(score(dev, "joint_per_cost", frozen_params), point),
        dev_teacher,
        dev_student,
        np.asarray([r["tokens"] >= 33 for r in dev]),
        dev_cost,
    )
    stanford = load_rows(ARCHIVE / "score-search/test-logits.json", stanford_hash)
    teacher, student, cost = arrays(stanford, params)
    long = np.asarray([r["tokens"] >= 33 for r in stanford])
    confidence = route(score(stanford, "confidence", params), confidence_point)
    values = score(stanford, "joint_per_cost", frozen_params)
    matched = top_mask(values, int(confidence.sum()))
    fixed = route(values, point)
    transfer = {
        "equal_calls": metrics(matched, teacher, student, long, cost),
        "fixed_cutoff": metrics(fixed, teacher, student, long, cost),
        "equal_calls_paired_95pct_interval_vs_confidence": paired_bootstrap_ci(
            np.where(matched, teacher, student), np.where(confidence, teacher, student)
        ),
    }
    budget_choice = plan["selected_under_development_teacher_cost_proxy"]
    budget_values = score(
        stanford, "joint_per_cost", {**params, "teacher_cost_exponent": budget_choice["exponent"]}
    )
    transfer["cost_proxy_choice_fixed_cutoff"] = metrics(
        route(budget_values, budget_choice["point"]), teacher, student, long, cost
    )
    retrospective = scan(stanford, params, confidence_point)
    joint_wins = []
    for dev_regime in development["regimes"]:
        if dev_regime["correct"] <= development["confidence"]["correct"]:
            continue
        for test_regime in retrospective["regimes"]:
            if test_regime["correct"] <= retrospective["confidence"]["correct"]:
                continue
            lower = max(dev_regime["lower"], test_regime["lower"])
            upper = min(dev_regime["upper"], test_regime["upper"])
            if lower < upper:
                joint_wins.append([lower, upper])
    result = {
        "plan_sha256": sha256(args.output / "plan.json"),
        "glue_development": development,
        "stanford_transfer": transfer,
        "stanford_retrospective": retrospective,
        "open_intervals_beating_confidence_on_both_at_equal_calls": joint_wins,
    }
    write_json(args.output / "results.json", result)
    print(f"Stanford with frozen exponent: {transfer}", flush=True)


if __name__ == "__main__":
    main()
