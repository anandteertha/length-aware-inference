"""Screen routing formulas before testing on separate Stanford SST sentences."""

import argparse
import re
import unicodedata
from itertools import product
from pathlib import Path

import numpy as np
from datasets import load_dataset
from evaluate_gate import paired_bootstrap_ci
from fit_gate import base_features, route, score
from soft_cost_gate import ARCHIVE, ORIGINAL, ROOT, arrays

from etdl.config import load_config
from etdl.data import DATASET_ID, encode, load_tokenizer, model_source
from etdl.io import read_json, sha256, write_json
from etdl.runtime import session

SST_REVISION = "fa0a89e0cabf5eac3ff4e1d27551fad4046ec012"
SST_TEST_SHA256 = "00ecd5cae460771baaf707439b046494f6bdfcec8552abfc7b66b22122e059a2"
SST_TEST_URL = (
    f"https://raw.githubusercontent.com/stanfordnlp/sentiment-treebank/{SST_REVISION}"
    "/binary/test-binary.txt"
)


def features(rows, params):
    margin, length = (
        (base_features(rows) - np.asarray(params["feature_mean"]))
        / np.asarray(params["feature_scale"])
    ).T
    logits = np.asarray([row["student_logits"] for row in rows])
    polarity = np.where(logits[:, 1] > logits[:, 0], 1.0, -1.0)
    return np.column_stack(
        (np.ones(len(rows)), margin, length, margin * length, polarity, margin * polarity)
    )


def formulas(fit_rows, params):
    coefficients = np.asarray(params["coefficients"]["joint"])
    for length, interaction, exponent in product(
        (0.0, 0.25, 0.5, 1.0), (0.0, 0.25, 0.5, 1.0), (0.0, 0.5, 0.75)
    ):
        yield {
            "family": "shrink",
            "length_scale": length,
            "interaction_scale": interaction,
            "exponent": exponent,
            "coefficients": (coefficients * [1, 1, length, interaction]).tolist(),
        }
    for penalty in (0.005, 0.015, 0.03, 0.06):
        yield {
            "family": "additive_cost",
            "cost_penalty": penalty,
            "coefficients": coefficients.tolist(),
        }
    teacher, student, _ = arrays(fit_rows, params)
    gain = teacher.astype(float) - student.astype(float)
    matrix = features(fit_rows, params)
    for penalty in (10.0, 40.0):
        regularizer = np.diag([0.0] + [penalty] * 5)
        weights = np.linalg.solve(matrix.T @ matrix + regularizer, matrix.T @ gain)
        for exponent in (0.0, 0.5, 0.75):
            yield {
                "family": "class_adjusted",
                "ridge_penalty": penalty,
                "exponent": exponent,
                "coefficients": weights.tolist(),
            }


def formula_score(formula, matrix, costs, params):
    coef = np.asarray(formula["coefficients"])
    gain = matrix[:, : len(coef)] @ coef
    if formula["family"] == "additive_cost":
        return gain - formula["cost_penalty"] * (
            costs / min(params["teacher_cost_ms"].values()) - 1
        )
    return gain / np.power(costs, formula["exponent"])


def top_mask(values, count):
    chosen = np.zeros(len(values), dtype=bool)
    chosen[np.argsort(-values, kind="stable")[:count]] = True
    return chosen


def freeze(output):
    gate = read_json(ORIGINAL)
    comparison_path = ARCHIVE / "cascade-timing/benchmark-plan.json"
    comparison = read_json(comparison_path)
    rows_path = ARCHIVE / "gate/gate-records.json"
    dev_path = ARCHIVE / "validation-logits.json"
    if sha256(ORIGINAL) != comparison["gate_plan_sha256"]:
        raise ValueError("Original benchmark has a different gate")
    if sha256(rows_path) != gate["gate_records_sha256"]:
        raise ValueError("Original gate fit/calibration data changed")
    if sha256(dev_path) != comparison["validation_logits_sha256"]:
        raise ValueError("Development data changed")
    rows = read_json(rows_path)["rows"]
    fit, calibration = rows[: gate["fit_count"]], rows[gate["fit_count"] :]
    dev = read_json(dev_path)["rows"]
    for row in dev:
        row["student_logits"] = row["four_layer_distilled_logits"]
    params = gate["parameters"]
    cal_teacher, cal_student, cal_cost = arrays(calibration, params)
    dev_teacher, dev_student, dev_cost = arrays(dev, params)
    cal_features, dev_features = features(calibration, params), features(dev, params)
    conf = comparison["confidence_point"]
    dev_calls = int(route(score(dev, "confidence", params), conf).sum())
    budgets = [(len(calibration) * n + 9) // 10 for n in (1, 2, 3)]
    grid = []
    for formula in formulas(fit, params):
        cal_score = formula_score(formula, cal_features, cal_cost, params)
        dev_score = formula_score(formula, dev_features, dev_cost, params)
        cal_masks = [top_mask(cal_score, calls) for calls in budgets]
        dev_mask = top_mask(dev_score, dev_calls)
        grid.append(
            {
                "formula": formula,
                "calibration_correct_by_quota": [
                    int(np.where(mask, cal_teacher, cal_student).sum()) for mask in cal_masks
                ],
                "calibration_teacher_ms_at_first_quota": float(cal_cost[cal_masks[0]].sum()),
                "development_correct_at_quota": int(
                    np.where(dev_mask, dev_teacher, dev_student).sum()
                ),
                "development_teacher_ms_at_quota": float(dev_cost[dev_mask].sum()),
                "point": {
                    "route_mode": "threshold",
                    "threshold": float(np.quantile(cal_score, 0.9)),
                },
            }
        )
    cal_index = min(
        range(len(grid)),
        key=lambda i: (
            -sum(grid[i]["calibration_correct_by_quota"]),
            grid[i]["calibration_teacher_ms_at_first_quota"],
            i,
        ),
    )
    dev_index = min(
        range(len(grid)),
        key=lambda i: (
            -grid[i]["development_correct_at_quota"],
            grid[i]["development_teacher_ms_at_quota"],
            i,
        ),
    )
    plan = {
        "original_gate_sha256": sha256(ORIGINAL),
        "calibration_and_fit_sha256": sha256(rows_path),
        "development_sha256": sha256(dev_path),
        "stanford_test_url": SST_TEST_URL,
        "stanford_test_sha256": SST_TEST_SHA256,
        "confidence_point": conf,
        "calibration_quotas": budgets,
        "development_quota": dev_calls,
        "development_confidence_correct": int(
            np.where(route(score(dev, "confidence", params), conf), dev_teacher, dev_student).sum()
        ),
        "frozen_indices": {"calibration_selected": cal_index, "development_selected": dev_index},
        "grid": grid,
    }
    write_json(output / "plan.json", plan)
    print(
        f"Froze {len(grid)} formulas. Selected calibration={cal_index}, development={dev_index}.",
        flush=True,
    )


def normalized(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def predict_test(output, tree_file, gate):
    if sha256(tree_file) != SST_TEST_SHA256:
        raise ValueError("Unexpected Stanford test file hash")
    manifest = read_json(ARCHIVE / "manifest.json")
    original_rows = load_dataset(DATASET_ID, "sst2", revision=manifest["dataset_revision"])
    sentences = {
        normalized(row["sentence"])
        for split in ("train", "validation")
        for row in original_rows[split]
    }
    rows = []
    test_sentences = set()
    skipped = {"neutral": 0, "overlap": 0, "duplicate": 0}
    for idx, tree in enumerate(tree_file.read_text(encoding="utf-8").splitlines()):
        label = int(tree[1:].split(maxsplit=1)[0])
        if label == -1:
            skipped["neutral"] += 1
            continue
        words = re.findall(r"\(-?\d+ ([^()\s]+)\)", tree)
        if not words or label not in (0, 1):
            raise ValueError(f"Invalid Stanford binary tree {idx}")
        text = " ".join(words).replace("-LRB-", "(").replace("-RRB-", ")")
        key = normalized(text)
        if key in sentences:
            skipped["overlap"] += 1
        elif key in test_sentences:
            skipped["duplicate"] += 1
        else:
            rows.append({"idx": idx, "label": label, "text": text})
            test_sentences.add(key)
    print(f"Locked {len(rows)} test sentences, excluded {skipped}", flush=True)
    config = load_config(ROOT / "configs/default.toml")
    if model_source(config) != manifest["model_source"]:
        raise ValueError("Tokenizer model source differs")
    teacher_path = config.artifact_dir / "models/dynamic.onnx"
    student_path = ROOT / "artifacts/student-study/onnx-verified/four_layer_distilled.onnx"
    export = read_json(ARCHIVE / "onnx-export-summary.json")["four_layer_distilled"]
    if (
        sha256(teacher_path) != gate["teacher_model_sha256"]
        or sha256(student_path) != export["onnx_sha256"]
    ):
        raise ValueError("ONNX model does not match the archive")
    tokenizer = load_tokenizer(config)
    teacher, _ = session(teacher_path, threads=4, allow_spinning=False)
    student, _ = session(student_path, threads=4, allow_spinning=False)
    for row in rows:
        inputs = encode(tokenizer, row.pop("text"), config.max_length)
        row["tokens"] = inputs["input_ids"].shape[1]
        row["teacher_logits"] = teacher.run(["logits"], inputs)[0][0].tolist()
        row["student_logits"] = student.run(["logits"], inputs)[0][0].tolist()
    write_json(
        output / "test-logits.json",
        {
            "test_sha256": SST_TEST_SHA256,
            "teacher_onnx_sha256": gate["teacher_model_sha256"],
            "student_onnx_sha256": export["onnx_sha256"],
            "excluded": skipped,
            "rows": rows,
        },
    )
    return rows


def evaluate_test(output, rows, plan, params):
    teacher, student, costs = arrays(rows, params)
    matrix = features(rows, params)
    base = route(score(rows, "confidence", params), plan["confidence_point"])
    quota = int(base.sum())
    baseline = np.where(base, teacher, student)
    results = {}
    for name, index in plan["frozen_indices"].items():
        selected = plan["grid"][index]
        values = formula_score(selected["formula"], matrix, costs, params)
        fixed = route(values, selected["point"])
        equal = top_mask(values, quota)
        matched_correct = np.where(equal, teacher, student)
        results[name] = {
            "formula_index": index,
            "fixed_correct": int(np.where(fixed, teacher, student).sum()),
            "fixed_calls": int(fixed.sum()),
            "matched_correct": int(matched_correct.sum()),
            "matched_teacher_cost_proxy_ms_per_request": float(costs[equal].sum() / len(rows)),
            "matched_accuracy_difference_95pct_interval": paired_bootstrap_ci(
                matched_correct, baseline
            ),
            "matched_wins": int((matched_correct & ~baseline).sum()),
            "matched_losses": int((baseline & ~matched_correct).sum()),
        }
    report = {
        "plan_sha256": sha256(output / "plan.json"),
        "logits_sha256": sha256(output / "test-logits.json"),
        "n": len(rows),
        "student_correct": int(student.sum()),
        "teacher_correct": int(teacher.sum()),
        "available_teacher_fixes": int((teacher & ~student).sum()),
        "available_teacher_harms": int((student & ~teacher).sum()),
        "confidence_correct": int(baseline.sum()),
        "matched_calls": quota,
        "confidence_teacher_cost_proxy_ms_per_request": float(costs[base].sum() / len(rows)),
        "candidates": results,
    }
    write_json(output / "test.json", report)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "test"))
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/student-study/score-search"
    )
    parser.add_argument("--test-file", type=Path)
    args = parser.parse_args()
    target = args.output / ("plan.json" if args.stage == "freeze" else "test.json")
    if target.exists():
        raise ValueError(f"Output already exists: {target}")
    if args.stage == "freeze":
        freeze(args.output)
        return
    plan = read_json(args.output / "plan.json")
    if sha256(ORIGINAL) != plan["original_gate_sha256"]:
        raise ValueError("Original gate differs from the frozen plan")
    gate = read_json(ORIGINAL)
    rows = predict_test(args.output, args.test_file or args.output / "stanford-test.txt", gate)
    evaluate_test(args.output, rows, plan, gate["parameters"])


if __name__ == "__main__":
    main()
