"""Calibrate a softer teacher cost penalty and audit the frozen policy."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from datasets import load_dataset
from evaluate_gate import paired_bootstrap_ci
from fit_gate import BUDGETS, length_bin, route, score

from etdl.config import load_config
from etdl.data import load_tokenizer, model_source
from etdl.io import read_json, sha256, write_json
from etdl.runtime import session

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/student-study"
ORIGINAL = ARCHIVE / "gate/gate-plan.json"
EXPONENTS = (0.0, 0.25, 0.5, 0.75)
SEED = 2027


def arrays(rows, parameters):
    labels = np.asarray([row["label"] for row in rows])
    teacher = np.asarray([row["teacher_logits"] for row in rows]).argmax(axis=1)
    student = np.asarray([row["student_logits"] for row in rows]).argmax(axis=1)
    cost = np.asarray([parameters["teacher_cost_ms"][length_bin(row["tokens"])] for row in rows])
    return teacher == labels, student == labels, cost


def calibrate(output):
    original = read_json(ORIGINAL)
    records = ARCHIVE / "gate/gate-records.json"
    if sha256(records) != original["gate_records_sha256"]:
        raise ValueError("Original gate records changed")
    benchmark = read_json(ARCHIVE / "cascade-timing/benchmark-plan.json")
    if sha256(ORIGINAL) != benchmark["gate_plan_sha256"]:
        raise ValueError("Original benchmark must use the archived gate plan")
    rows = read_json(records)["rows"][original["fit_count"] :]
    if len(rows) != original["calibration_count"]:
        raise ValueError("Calibration count differs from original plan")
    parameters = original["parameters"]
    teacher, student, costs = arrays(rows, parameters)
    curves = {}
    candidates = []
    minimum_correct = int(teacher.sum()) - len(rows) // 100
    for exponent in EXPONENTS:
        values = score(rows, "joint_per_cost", {**parameters, "teacher_cost_exponent": exponent})
        points = []
        for budget in BUDGETS:
            mode = "none" if budget == 0 else "all" if budget == 1 else "threshold"
            cutoff = float(np.quantile(values, 1 - budget)) if mode == "threshold" else None
            point = {
                "requested_fallback_fraction": budget,
                "route_mode": mode,
                "threshold": cutoff,
            }
            fallback = route(values, point)
            correct = int(np.where(fallback, teacher, student).sum())
            calls = int(fallback.sum())
            cost = float(np.where(fallback, costs, 0).mean())
            point.update(
                calibration_correct=correct,
                calibration_accuracy=correct / len(rows),
                calibration_teacher_calls=calls,
                estimated_extra_teacher_ms_per_request=cost,
            )
            points.append(point)
            if correct >= minimum_correct:
                candidates.append((cost, calls, exponent, budget, len(points) - 1))
        curves[str(exponent)] = points
    _, _, selected, _, index = min(candidates)
    policies = {
        "confidence": {"score": "confidence", "point": benchmark["confidence_point"]},
        "original_cost": {
            "score": "joint_per_cost",
            "exponent": 1.0,
            "point": benchmark["selected_point"],
        },
        "soft_cost": {
            "score": "joint_per_cost",
            "exponent": selected,
            "point": curves[str(selected)][index],
        },
    }
    plan = {
        "original_gate_sha256": sha256(ORIGINAL),
        "calibration_records_sha256": sha256(records),
        "calibration_examples": len(rows),
        "teacher_correct": int(teacher.sum()),
        "minimum_correct": minimum_correct,
        "parameters": parameters,
        "policies": policies,
        "calibration_curves": curves,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "plan.json", plan)
    print(
        f"Frozen exponent {selected}, calibration point {policies['soft_cost']['point']}",
        flush=True,
    )


def evaluate(rows, plan, dataset):
    teacher, student, costs = arrays(rows, plan["parameters"])
    tokens = np.asarray([row["tokens"] for row in rows])
    results = {}
    scores = {}
    correctness = {}
    for name, policy in plan["policies"].items():
        parameters = {**plan["parameters"], "teacher_cost_exponent": policy.get("exponent", 1.0)}
        values = score(rows, policy["score"], parameters)
        scores[name] = values
        calls = route(values, policy["point"])
        correct = np.where(calls, teacher, student)
        correctness[name] = correct
        results[name] = {
            "correct": int(correct.sum()),
            "accuracy": float(correct.mean()),
            "calls": int(calls.sum()),
            "fixes": int((calls & teacher & ~student).sum()),
            "harms": int((calls & student & ~teacher).sum()),
            "calls_33_plus": int((calls & (tokens >= 33)).sum()),
            "calls_65_plus": int((calls & (tokens >= 65)).sum()),
            "sst2_cost_table_proxy_ms_per_request": float(np.where(calls, costs, 0).mean()),
        }
    matched = {}
    call_count = results["confidence"]["calls"]
    for name, values in scores.items():
        top = np.argsort(-values, kind="stable")[:call_count]
        matched[name] = int(student.sum() + teacher[top].sum() - student[top].sum())
    comparisons = {}
    for name, reference in {**correctness, "teacher_only": teacher}.items():
        if name == "soft_cost":
            continue
        selected = correctness["soft_cost"]
        comparisons[name] = {
            "accuracy_difference": float(selected.mean() - reference.mean()),
            "paired_95pct_interval": paired_bootstrap_ci(selected, reference),
        }
    return {
        "dataset": dataset,
        "examples": len(rows),
        "examples_33_plus": int((tokens >= 33).sum()),
        "examples_65_plus": int((tokens >= 65).sum()),
        "teacher_correct": int(teacher.sum()),
        "student_correct": int(student.sum()),
        "policies": results,
        "soft_cost_comparisons": comparisons,
        "equal_call_count": call_count,
        "equal_call_count_correct": matched,
    }


def external_rows(output, original):
    selection = read_json(ARCHIVE / "cost-weighted/external-selection.json")
    opts = {"revision": selection["revision"]}
    dataset = load_dataset(selection["dataset_id"], split="train", **opts)
    prior = load_dataset(selection["dataset_id"], split=selection["split"], **opts)
    config = load_config(ROOT / "configs/default.toml")
    archived_model = read_json(ARCHIVE / "manifest.json")["model_source"]
    if model_source(config) != archived_model:
        raise ValueError("Local tokenizer model source does not match the archived model")
    tokenizer = load_tokenizer(config)

    def fingerprint(value):
        return hashlib.sha256(" ".join(value.casefold().split()).encode()).hexdigest()

    seen = {fingerprint(prior[row["dataset_index"]]["text"]) for row in selection["rows"]}
    selected = []
    inputs = []
    counts = [0, 0]
    for index in np.random.default_rng(SEED).permutation(len(dataset)):
        row = dataset[int(index)]
        label = row["label"]
        if counts[label] == 250:
            continue
        digest = fingerprint(row["text"])
        if digest in seen:
            continue
        encoded = tokenizer(row["text"], truncation=False, verbose=False, return_tensors="np")
        tokens = encoded["input_ids"].shape[1]
        if not 3 <= tokens <= 128:
            continue
        seen.add(digest)
        selected.append(
            {"idx": int(index), "label": label, "tokens": tokens, "text_sha256": digest}
        )
        inputs.append({name: encoded[name] for name in ("input_ids", "attention_mask")})
        counts[label] += 1
        if counts == [250, 250]:
            break
    if counts != [250, 250]:
        raise ValueError(f"Insufficient eligible IMDb reviews: {counts}")
    write_json(
        output / "imdb-selection.json",
        {
            "dataset_id": selection["dataset_id"],
            "revision": selection["revision"],
            "split": "train",
            "seed": SEED,
            "rows": selected,
        },
    )
    print(f"Locked {len(selected)} external review IDs before inference", flush=True)
    student_path = ROOT / "artifacts/student-study/onnx-verified/four_layer_distilled.onnx"
    teacher_path = config.artifact_dir / "models/dynamic.onnx"
    expected_student = read_json(ARCHIVE / "onnx-export-summary.json")["four_layer_distilled"]
    if sha256(student_path) != expected_student["onnx_sha256"]:
        raise ValueError("Student ONNX graph differs from archived checkpoint")
    if sha256(teacher_path) != original["teacher_model_sha256"]:
        raise ValueError("Teacher ONNX graph differs from original gate")
    student, _ = session(student_path, threads=4, allow_spinning=False)
    teacher, _ = session(teacher_path, threads=4, allow_spinning=False)
    for row, encoded in zip(selected, inputs, strict=True):
        row["student_logits"] = student.run(["logits"], encoded)[0][0].tolist()
        row["teacher_logits"] = teacher.run(["logits"], encoded)[0][0].tolist()
    write_json(output / "imdb-logits.json", {"rows": selected})
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibrate", "sst2", "imdb", "timing-plan"))
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/student-study/soft-cost")
    args = parser.parse_args()
    output = args.output
    filename = {
        "calibrate": "plan.json",
        "timing-plan": "timing-plan/gate-plan.json",
    }.get(args.stage, f"{args.stage}.json")
    destination = output / filename
    if destination.exists():
        raise ValueError(f"Output already exists: {destination}")
    if args.stage == "calibrate":
        calibrate(output)
        return
    plan = read_json(output / "plan.json")
    if sha256(ORIGINAL) != plan["original_gate_sha256"]:
        raise ValueError("Original gate differs from the frozen plan")
    if args.stage == "timing-plan":
        original = read_json(ORIGINAL)
        policy = plan["policies"]["soft_cost"]
        original["parameters"]["teacher_cost_exponent"] = policy["exponent"]
        original["selected_policy"] = policy["score"]
        original["selected_point_index"] = 0
        original["curves"][policy["score"]] = [policy["point"]]
        original["soft_cost_plan_sha256"] = sha256(output / "plan.json")
        write_json(destination, original)
        return
    if args.stage == "sst2":
        path = ARCHIVE / "validation-logits.json"
        if (
            sha256(path)
            != read_json(ARCHIVE / "validation-summary.json")["validation_logits_sha256"]
        ):
            raise ValueError("Validation archive changed")
        rows = read_json(path)["rows"]
        for row in rows:
            row["student_logits"] = row["four_layer_distilled_logits"]
        source = path
        dataset = "SST-2 validation, previously inspected"
    else:
        rows = external_rows(output, read_json(ORIGINAL))
        source = output / "imdb-logits.json"
        dataset = "IMDb train, new balanced sample with 3–128 tokens"
    result = evaluate(rows, plan, dataset)
    result["plan_sha256"] = sha256(output / "plan.json")
    result["logits_sha256"] = sha256(source)
    if args.stage == "sst2":
        policies = result["policies"]
        soft, confidence = policies["soft_cost"], policies["confidence"]
        advance = (
            result["teacher_correct"] - soft["correct"] <= len(rows) // 100
            and soft["correct"] >= confidence["correct"]
            and soft["sst2_cost_table_proxy_ms_per_request"]
            < confidence["sst2_cost_table_proxy_ms_per_request"]
        )
        result["advance_to_timing"] = advance
    write_json(destination, result)
    print(destination, flush=True)


if __name__ == "__main__":
    main()
