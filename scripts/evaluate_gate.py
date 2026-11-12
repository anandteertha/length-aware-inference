"""Apply a frozen train-derived gate plan to GLUE SST-2 validation logits.

This script does not fit features or choose thresholds from validation labels.
Its cost numbers are estimates; actual resident cascade timing is separate.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from fit_gate import POLICIES, length_bin, route, score

from etdl.io import read_json, sha256, write_json


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("artifacts/student-study"))
    parser.add_argument("--gate", type=Path, default=Path("artifacts/student-study/gate-fit"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/student-study/gate-validation.json")
    )
    return parser.parse_args()


def paired_bootstrap_ci(selected_correct, teacher_correct):
    difference = selected_correct.astype(float) - teacher_correct.astype(float)
    rng = np.random.default_rng(2027)
    estimates = []
    for _ in range(10000):
        indices = rng.integers(0, len(difference), size=len(difference))
        estimates.append(float(np.mean(difference[indices])))
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def main():
    args = arguments()
    study = args.study.resolve()
    gate = args.gate.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Gate validation output already exists: {output}")
    plan_path = gate / "gate-plan.json"
    plan = read_json(plan_path)
    if "validation unused" not in plan["status"]:
        raise ValueError("Expected a train-derived, frozen gate plan")
    validation_summary = read_json(study / "validation-summary.json")
    logits_path = study / "validation-logits.json"
    if sha256(logits_path) != validation_summary["validation_logits_sha256"]:
        raise ValueError("Validation logits changed")
    source = read_json(logits_path)
    if source["split"] != "validation" or len(source["rows"]) != 872:
        raise ValueError("Expected 872 pinned validation rows")
    records = [
        {
            "idx": row["idx"],
            "label": row["label"],
            "tokens": row["tokens"],
            "student_logits": row["four_layer_distilled_logits"],
            "teacher_logits": row["teacher_logits"],
        }
        for row in source["rows"]
    ]
    labels = np.asarray([row["label"] for row in records])
    student = np.asarray([row["student_logits"] for row in records]).argmax(axis=1)
    teacher = np.asarray([row["teacher_logits"] for row in records]).argmax(axis=1)
    student_correct = student == labels
    teacher_correct = teacher == labels
    curves = {}
    selected_result = None
    selected_correct = None
    for policy in POLICIES:
        scores = score(records, policy, plan["parameters"])
        points = []
        for index, frozen in enumerate(plan["curves"][policy]):
            fallback = route(scores, frozen)
            predictions = np.where(fallback, teacher, student)
            correctness = predictions == labels
            result = {
                "requested_fallback_fraction": frozen["requested_fallback_fraction"],
                "actual_fallback_fraction": float(np.mean(fallback)),
                "correct": int(np.sum(correctness)),
                "accuracy": float(np.mean(correctness)),
                "teacher_corrects_student_on_fallback": int(
                    np.sum(fallback & teacher_correct & ~student_correct)
                ),
                "teacher_harms_student_on_fallback": int(
                    np.sum(fallback & ~teacher_correct & student_correct)
                ),
                "estimated_extra_teacher_ms_per_request": float(
                    np.mean(
                        [
                            plan["parameters"]["teacher_cost_ms"][length_bin(row["tokens"])]
                            if fallback[i]
                            else 0.0
                            for i, row in enumerate(records)
                        ]
                    )
                ),
            }
            points.append(result)
            if policy == plan["selected_policy"] and index == plan["selected_point_index"]:
                selected_result = {"policy": policy, "point_index": index, **result}
                selected_correct = correctness
                selected_fallback = fallback
        curves[policy] = points
    if selected_result is None or selected_correct is None:
        raise ValueError("Frozen selected point is missing")
    selected_result["paired_accuracy_difference_vs_teacher"] = float(
        np.mean(selected_correct.astype(float) - teacher_correct.astype(float))
    )
    selected_result["paired_bootstrap_95pct_ci"] = paired_bootstrap_ci(
        selected_correct, teacher_correct
    )
    selected_result["length_strata"] = {
        name: {
            "examples": int(np.sum(mask)),
            "fallback": int(np.sum(selected_fallback & mask)),
            "student_correct": int(np.sum(student_correct & mask)),
            "teacher_correct": int(np.sum(teacher_correct & mask)),
            "cascade_correct": int(np.sum(selected_correct & mask)),
        }
        for name, mask in (
            ("3-16", np.asarray([row["tokens"] <= 16 for row in records])),
            ("17-32", np.asarray([17 <= row["tokens"] <= 32 for row in records])),
            ("33+", np.asarray([row["tokens"] >= 33 for row in records])),
        )
    }
    confidence_candidates = [
        (point["estimated_extra_teacher_ms_per_request"], index)
        for index, point in enumerate(plan["curves"]["confidence"])
        if point["calibration_accuracy"] >= plan["teacher_calibration_accuracy"] - 0.01
    ]
    _, confidence_index = min(confidence_candidates)
    confidence_point = plan["curves"]["confidence"][confidence_index]
    confidence_fallback = route(
        score(records, "confidence", plan["parameters"]), confidence_point
    )
    confidence_correct = np.where(confidence_fallback, teacher, student) == labels
    confidence_control = {
        "point_index": confidence_index,
        "correct": int(np.sum(confidence_correct)),
        "fallback_count": int(np.sum(confidence_fallback)),
        "paired_accuracy_difference_vs_teacher": float(
            np.mean(confidence_correct.astype(float) - teacher_correct.astype(float))
        ),
        "paired_bootstrap_95pct_ci": paired_bootstrap_ci(
            confidence_correct, teacher_correct
        ),
    }
    report = {
        "status": "frozen-gate validation; estimated cost only, not measured cascade latency",
        "gate_plan_sha256": sha256(plan_path),
        "validation_logits_sha256": sha256(logits_path),
        "examples": len(records),
        "student_only_correct": int(np.sum(student_correct)),
        "teacher_only_correct": int(np.sum(teacher_correct)),
        "selected": selected_result,
        "confidence_control": confidence_control,
        "curves": curves,
    }
    write_json(output, report)
    print(json.dumps({"event": "gate_evaluated", "selected": selected_result}), flush=True)


if __name__ == "__main__":
    main()
