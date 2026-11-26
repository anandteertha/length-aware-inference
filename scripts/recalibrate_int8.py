"""Freeze INT8-specific confidence cutoffs and replay them on archived outputs."""

import argparse
import re
from pathlib import Path

import numpy as np
from evaluate_gate import paired_bootstrap_ci
from fit_gate import route, score
from run_int8_extension import THREADS, candidate_path
from search_gate_formulas import SST_TEST_SHA256, SST_TEST_URL
from soft_cost_gate import ARCHIVE, ORIGINAL, ROOT

from etdl.config import load_config
from etdl.data import encode, load_tokenizer, model_source
from etdl.io import read_json, sha256, write_json
from etdl.runtime import session

OLD = ARCHIVE / "int8-shape-repair"
MODEL = candidate_path(ROOT / "artifacts/student-study/int8-shape-repair")
STANFORD = ROOT / "artifacts/student-study/score-search/stanford-test.txt"
QUOTAS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.65, 0.8, 1.0)


def confidence_scores(rows, params):
    inputs = [{"student_logits": r["int8_logits"], "tokens": r["tokens"]} for r in rows]
    return score(inputs, "confidence", params)


def audit_rows(rows, values, point):
    routed = route(values, point)
    labels = np.asarray([r["label"] for r in rows])
    teacher = np.asarray([r["teacher_prediction"] for r in rows])
    student = np.asarray([r["int8_prediction"] for r in rows])
    result = np.where(routed, teacher, student)
    return (
        {
            "correct": int((result == labels).sum()),
            "calls": int(routed.sum()),
            "fixes": int((routed & (teacher == labels) & (student != labels)).sum()),
            "harms": int((routed & (teacher != labels) & (student == labels)).sum()),
        },
        result,
    )


def read_old(name, summary_key):
    path = OLD / name
    if sha256(path) != read_json(OLD / "summary.json")[summary_key]:
        raise ValueError(f"Original INT8 archive changed: {name}")
    return read_json(path)


def calibrate(output, tree_file):
    old_plan = read_old("plan.json", "plan_sha256")
    if sha256(MODEL) != old_plan["candidate_sha256"]:
        raise ValueError("INT8 model differs from the previous study")
    stanford_path = ARCHIVE / "score-search/test-logits.json"
    if sha256(stanford_path) != read_json(ARCHIVE / "score-search/test.json")["logits_sha256"]:
        raise ValueError("Stanford archive changed")
    if sha256(tree_file) != SST_TEST_SHA256:
        raise ValueError(f"Expected the pinned binary test trees from {SST_TEST_URL}")
    trees = tree_file.read_text(encoding="utf-8").splitlines()
    config = load_config(ROOT / "configs/default.toml")
    if model_source(config) != read_json(ARCHIVE / "manifest.json")["model_source"]:
        raise ValueError("Tokenizer model source differs")
    tokenizer = load_tokenizer(config)
    int8, _ = session(MODEL, threads=THREADS, allow_spinning=False)
    rows = []
    archived = read_json(stanford_path)["rows"]
    for i, r in enumerate(archived, 1):
        tree = trees[r["idx"]]
        if int(tree[1:].split(maxsplit=1)[0]) != r["label"]:
            raise ValueError("Stanford label differs from the archive")
        words = re.findall(r"\(-?\d+ ([^()\s]+)\)", tree)
        text = " ".join(words).replace("-LRB-", "(").replace("-RRB-", ")")
        inputs = encode(tokenizer, text, config.max_length)
        if inputs["input_ids"].shape[1] != r["tokens"]:
            raise ValueError("Stanford token count differs from the archive")
        logits = int8.run(["logits"], inputs)[0][0]
        if not np.isfinite(logits).all():
            raise ValueError("INT8 logits contain nonfinite values")
        rows.append(
            {
                "idx": r["idx"],
                "tokens": r["tokens"],
                "label": r["label"],
                "teacher_prediction": int(np.argmax(r["teacher_logits"])),
                "int8_prediction": int(np.argmax(logits)),
                "int8_logits": logits.tolist(),
            }
        )
        if i % 600 == 0:
            print(f"Calibrating: scored {i} / {len(archived)} sentences", flush=True)
    write_json(output / "calibration-logits.json", {"rows": rows})
    gate = read_json(ORIGINAL)
    values = confidence_scores(rows, gate["parameters"])
    teacher_correct = sum(r["teacher_prediction"] == r["label"] for r in rows)
    candidates = []
    for quota in QUOTAS:
        point = (
            {"route_mode": "all", "threshold": None}
            if quota == 1
            else {"route_mode": "threshold", "threshold": float(np.quantile(values, 1 - quota))}
        )
        metrics, _ = audit_rows(rows, values, point)
        candidates.append({"nominal_quota": quota, "point": point, **metrics})
    chosen = {}
    for name, fraction in (("primary", 0.005), ("original_budget", 0.01)):
        chosen[name] = min(
            (c for c in candidates if teacher_correct - c["correct"] <= fraction * len(rows)),
            key=lambda c: (c["calls"], -c["correct"], c["nominal_quota"]),
        )
    inherited = read_json(ARCHIVE / "cascade-timing/benchmark-plan.json")
    if sha256(ORIGINAL) != inherited["gate_plan_sha256"]:
        raise ValueError("Original benchmark gate differs")
    plan = {
        "candidate_sha256": old_plan["candidate_sha256"],
        "gate_sha256": sha256(ORIGINAL),
        "source_int8_plan_sha256": sha256(OLD / "plan.json"),
        "stanford_archive_sha256": sha256(stanford_path),
        "calibration_logits_sha256": sha256(output / "calibration-logits.json"),
        "calibration_examples": len(rows),
        "teacher_correct": teacher_correct,
        "inherited_point": inherited["confidence_point"],
        "inherited_calibration": audit_rows(rows, values, inherited["confidence_point"])[0],
        "candidates": candidates,
        "selected": chosen,
    }
    write_json(output / "plan.json", plan)
    print(f"Froze cutoff choices: {chosen}", flush=True)


def replay(output, plan):
    params = read_json(ORIGINAL)["parameters"]
    rows = read_old("validation-audit.json", "validation_audit_sha256")["rows"]
    external = read_old("external-audit.json", "external_audit_sha256")["rows"]
    for r in external:
        r["int8_prediction"] = r["int8_onnx_prediction"]
    confidence = {"inherited": plan["inherited_point"]}
    confidence.update({name: c["point"] for name, c in plan["selected"].items()})
    report = {"plan_sha256": sha256(output / "plan.json")}
    for name, samples in (("sst2", rows), ("imdb", external)):
        values = confidence_scores(samples, params)
        labels = np.asarray([r["label"] for r in samples])
        teacher = np.asarray([r["teacher_prediction"] for r in samples]) == labels
        _, inherited = audit_rows(samples, values, confidence["inherited"])
        policies = {}
        for method, point in confidence.items():
            metrics, preds = audit_rows(samples, values, point)
            metrics["within_one_point"] = (
                int(teacher.sum()) - metrics["correct"] <= len(samples) / 100
            )
            if method == "primary":
                metrics["paired_95pct_interval_vs_inherited"] = paired_bootstrap_ci(
                    preds == labels, inherited == labels
                )
            policies[method] = metrics
        report[name] = {
            "n": len(samples),
            "teacher_correct": int(teacher.sum()),
            "policies": policies,
        }
    write_json(output / "replay.json", report)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibrate", "replay"))
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/student-study/int8-recalibration"
    )
    parser.add_argument("--stanford-trees", type=Path, default=STANFORD)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "calibrate":
        if (output / "plan.json").exists():
            raise ValueError("Calibration is already frozen")
        calibrate(output, args.stanford_trees)
        return
    plan = read_json(output / "plan.json")
    if plan["gate_sha256"] != sha256(ORIGINAL):
        raise ValueError("Original gate changed")
    replay(output, plan)


if __name__ == "__main__":
    main()
