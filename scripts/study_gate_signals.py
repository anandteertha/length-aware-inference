"""Test additional teacher-benefit signals and describe student/teacher errors."""

import argparse
import hashlib
import html
import re
from itertools import product
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from datasets import load_dataset
from evaluate_gate import paired_bootstrap_ci
from fit_gate import route, score
from search_gate_formulas import SST_TEST_SHA256, SST_TEST_URL, normalized
from soft_cost_gate import ARCHIVE, ORIGINAL, ROOT, arrays

from etdl.config import load_config
from etdl.data import DATASET_ID, load_tokenizer, model_source
from etdl.io import read_json, sha256, write_json
from etdl.runtime import session

FEATURE_COUNTS = {"margin_length": 3, "logits": 7, "text": 11}
NEGATION = {"no", "not", "never", "neither", "none", "without", "nothing"}
CONTRAST = {"but", "although", "though", "however", "yet", "despite", "whereas"}


def signals(text, tokens):
    text = re.sub(r"<br\s*/?>", " ", html.unescape(text), flags=re.IGNORECASE)
    words = re.findall(r"\b[a-z]+(?:['’][a-z]+)?\b", text.casefold())
    return [
        int(any(w in NEGATION or w.endswith(("n't", "n’t")) for w in words)),
        int(any(w in CONTRAST for w in words)),
        len(re.findall(r"[!?;:]", text)) / max(1, len(words)),
        max(0, tokens - 2) / max(1, len(words)),
    ]


def margins(rows):
    logits = np.asarray([r["student_logits"] for r in rows])
    diff = logits[:, 1] - logits[:, 0]
    return np.abs(diff), (diff > 0).astype(int), logits.mean(axis=1)


def feature_matrix(rows, center=None, scale=None):
    margin, positive, mean_logit = margins(rows)
    length = np.log2([r["tokens"] for r in rows])
    simple = np.column_stack((margin, length, mean_logit, [r["signals"] for r in rows]))
    if center is None:
        center, scale = simple.mean(axis=0), simple.std(axis=0)
        scale[scale == 0] = 1
    z = np.clip((simple - center) / scale, -5, 5)
    m, n, avg = z[:, :3].T
    polarity = 2 * positive - 1
    matrix = np.column_stack(
        (np.ones(len(rows)), m, n, m * n, polarity, polarity * m, avg, m**2, z[:, 3:])
    )
    return matrix, center, scale


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def logistic(matrix, labels, penalty):
    """Fit a small L2 logistic model with a damped Newton update."""
    weights = np.zeros(matrix.shape[1])

    def loss(coef):
        logits = matrix @ coef
        return np.sum(np.logaddexp(0, logits) - labels * logits) + coef @ penalty @ coef / 2

    for _ in range(100):
        p = sigmoid(matrix @ weights)
        gradient = matrix.T @ (p - labels) + penalty @ weights
        if np.max(np.abs(gradient)) < 1e-7:
            return weights.tolist()
        curvature = matrix.T @ (matrix * (p * (1 - p))[:, None]) + penalty
        step = np.linalg.solve(curvature, gradient)
        scale = 1.0
        current = loss(weights)
        while loss(weights - scale * step) > current:
            scale /= 2
            if scale < 1e-10:
                if np.max(np.abs(gradient)) < 1e-5:
                    return weights.tolist()
                raise ValueError("Logistic line search did not converge")
        weights -= scale * step
    raise ValueError("Logistic fit did not converge")


def group_index(rows, stratum):
    if stratum == "length":
        return np.digitize([r["tokens"] for r in rows], [17, 33])
    return margins(rows)[1]


def fit_scores(rows, matrix, teacher, student):
    gain = teacher.astype(float) - student.astype(float)
    for (feature_set, count), regularization in product(FEATURE_COUNTS.items(), (10, 40)):
        x = matrix[:, : count + 1]
        penalty = np.diag([0] + [regularization] * count)
        yield {
            "family": "ridge",
            "features": feature_set,
            "penalty": regularization,
            "weights": np.linalg.solve(x.T @ x + penalty, x.T @ gain).tolist(),
        }
        yield {
            "family": "fix_harm",
            "features": feature_set,
            "penalty": regularization,
            "fix": logistic(x, teacher & ~student, penalty),
            "harm": logistic(x, student & ~teacher, penalty),
        }
    margin = margins(rows)[0]
    for bins, stratum, prior in product((5, 10), ("length", "polarity"), (25, 100)):
        edges = np.quantile(margin, np.arange(1, bins) / bins)
        buckets = np.digitize(margin, edges)
        groups = group_index(rows, stratum)
        table = []
        for bucket in range(bins):
            inside = buckets == bucket
            pooled = float(gain[inside].mean()) if inside.any() else float(gain.mean())
            row = []
            for group in range(3 if stratum == "length" else 2):
                cell = inside & (groups == group)
                row.append(float((gain[cell].sum() + prior * pooled) / (cell.sum() + prior)))
            table.append(row)
        yield {
            "family": "groups",
            "edges": edges.tolist(),
            "stratum": stratum,
            "prior": prior,
            "table": table,
        }


def raw_score(model, rows, matrix):
    if model["family"] == "groups":
        buckets = np.digitize(margins(rows)[0], model["edges"])
        return np.asarray(model["table"])[buckets, group_index(rows, model["stratum"])]
    x = matrix[:, : FEATURE_COUNTS[model["features"]] + 1]
    if model["family"] == "ridge":
        return x @ model["weights"]
    return sigmoid(x @ model["fix"]) - sigmoid(x @ model["harm"])


def percentile_rank(values, anchors):
    index = np.searchsorted(anchors, values, side="right")
    return index / len(anchors)


def blend_score(values, margin, candidate, confidence_anchors):
    blend = candidate["score_weight"]
    if blend == 1:
        return values
    return blend * percentile_rank(values, candidate["anchors"]) + (1 - blend) * percentile_rank(
        -margin, confidence_anchors
    )


def order(values, margin):
    return np.lexsort((margin, -values))


def get_older_cohorts(output, gate):
    fit_path = ARCHIVE / "gate/gate-records.json"
    glue_path = ARCHIVE / "validation-logits.json"
    stanford_path = ARCHIVE / "score-search/test-logits.json"
    tree_path = output / "stanford-test.txt"
    if not tree_path.is_file():
        output.mkdir(parents=True, exist_ok=True)
        urlretrieve(SST_TEST_URL, tree_path)
    if (
        sha256(tree_path) != SST_TEST_SHA256
        or sha256(fit_path) != gate["gate_records_sha256"]
        or sha256(glue_path)
        != read_json(ARCHIVE / "validation-summary.json")["validation_logits_sha256"]
        or sha256(stanford_path) != read_json(ARCHIVE / "score-search/test.json")["logits_sha256"]
    ):
        raise ValueError("An input archive changed")

    manifest = read_json(ARCHIVE / "manifest.json")
    data = load_dataset(DATASET_ID, "sst2", revision=manifest["dataset_revision"])
    train = read_json(fit_path)["rows"]
    glue = read_json(glue_path)["rows"]
    stanford = read_json(stanford_path)["rows"]
    trees = tree_path.read_text(encoding="utf-8").splitlines()
    for row in train:
        source = data["train"][row["idx"]]
        if int(source["idx"]) != row["idx"] or source["label"] != row["label"]:
            raise ValueError("GLUE training row mismatch")
        row["signals"] = signals(source["sentence"], row["tokens"])
    for row in glue:
        source = data["validation"][row["idx"]]
        if int(source["idx"]) != row["idx"] or source["label"] != row["label"]:
            raise ValueError("GLUE development row mismatch")
        row["signals"] = signals(source["sentence"], row["tokens"])
        row["student_logits"] = row["four_layer_distilled_logits"]
    for row in stanford:
        words = re.findall(r"\(-?\d+ ([^()\s]+)\)", trees[row["idx"]])
        row["signals"] = signals(" ".join(words), row["tokens"])
    n = gate["fit_count"]
    return {
        "fit": train[:n] + glue,
        "calibration": train[n:],
        "stanford_development": stanford,
    }


def patterns(rows, teacher, student, baseline):
    margin, positive, _ = margins(rows)
    lengths = np.asarray([r["tokens"] for r in rows])
    cues = np.asarray([r["signals"][:2] for r in rows])
    fix, harm = teacher & ~student, student & ~teacher
    edges = np.quantile(margin, [0.2, 0.4, 0.6, 0.8])
    by_margin = np.digitize(margin, edges)
    groups = {
        **{f"margin_q{i + 1}": by_margin == i for i in range(5)},
        **{
            f"tokens_{a}_{b}": (lengths >= a) & (lengths <= b)
            for a, b in ((0, 16), (17, 32), (33, 64), (65, 96), (97, 128), (129, 512))
        },
        "predict_negative": positive == 0,
        "predict_positive": positive == 1,
        "negation": cues[:, 0] == 1,
        "no_negation": cues[:, 0] == 0,
        "contrast": cues[:, 1] == 1,
        "no_contrast": cues[:, 1] == 0,
        "routed": baseline,
        "not_routed": ~baseline,
    }
    first = by_margin == 0
    for name in tuple(groups):
        if not name.startswith("tokens_") and name not in (
            "negation",
            "no_negation",
            "contrast",
            "no_contrast",
        ):
            continue
        groups[f"margin_q1_{name}"] = first & groups[name]
    return {
        name: {
            "n": int(mask.sum()),
            "fixes": int((fix & mask).sum()),
            "harms": int((harm & mask).sum()),
            "calls": int((baseline & mask).sum()),
            "routed_fixes": int((fix & mask & baseline).sum()),
            "routed_harms": int((harm & mask & baseline).sum()),
        }
        for name, mask in groups.items()
        if mask.any()
    }


def cohort(rows, params, confidence_point, center=None, scale=None):
    teacher, student, _ = arrays(rows, params)
    margin = margins(rows)[0]
    baseline = route(score(rows, "confidence", params), confidence_point)
    matrix, _, _ = feature_matrix(rows, center, scale)
    return {
        "rows": rows,
        "teacher": teacher,
        "student": student,
        "margin": margin,
        "baseline": baseline,
        "matrix": matrix,
        "quota": int(baseline.sum()),
    }


def route_metrics(data, chosen):
    teacher, student = data["teacher"], data["student"]
    return {
        "correct": int(np.where(chosen, teacher, student).sum()),
        "calls": int(chosen.sum()),
        "fixes": int((chosen & teacher & ~student).sum()),
        "harms": int((chosen & student & ~teacher).sum()),
        "long_calls": sum(
            int(call) for call, row in zip(chosen, data["rows"], strict=True) if row["tokens"] >= 33
        ),
    }


def mask_ranked(data, values):
    ranked = order(values, data["margin"])
    chosen = np.zeros(len(values), dtype=bool)
    chosen[ranked[: data["quota"]]] = True
    return chosen, ranked


def freeze(output):
    gate = read_json(ORIGINAL)
    old = get_older_cohorts(output, gate)
    fit = old["fit"]
    params = gate["parameters"]
    benchmark = read_json(ARCHIVE / "cascade-timing/benchmark-plan.json")
    if benchmark["gate_plan_sha256"] != sha256(ORIGINAL):
        raise ValueError("Original gate changed")
    confidence = benchmark["confidence_point"]
    matrix, center, scale = feature_matrix(fit)
    teacher, student, _ = arrays(fit, params)
    quantiles = np.linspace(0, 1, 101)
    conf_anchors = np.quantile(-margins(fit)[0], quantiles).tolist()
    selection_sets = {
        name: cohort(old[name], params, confidence, center, scale)
        for name in ("calibration", "stanford_development")
    }
    grid = []
    for model in fit_scores(fit, matrix, teacher, student):
        anchors = np.quantile(raw_score(model, fit, matrix), quantiles).tolist()
        for weight in (0.25, 0.5, 1.0):
            candidate = {"model": model, "score_weight": weight, "anchors": anchors}
            selection = {}
            for name, data in selection_sets.items():
                values = blend_score(
                    raw_score(model, data["rows"], data["matrix"]),
                    data["margin"],
                    candidate,
                    conf_anchors,
                )
                chosen, ranked = mask_ranked(data, values)
                selection[name] = route_metrics(data, chosen)
                if name == "calibration":
                    boundary = ranked[data["quota"] - 1]
                    candidate["cutoff"] = float(values[boundary])
                    candidate["cutoff_margin"] = float(data["margin"][boundary])
            candidate["selection"] = selection
            grid.append(candidate)

    def preference(i):
        values = grid[i]["selection"].values()
        return (-sum(v["correct"] for v in values), sum(v["long_calls"] for v in values), i)

    finalists = {
        family: min(
            (i for i, row in enumerate(grid) if row["model"]["family"] == family), key=preference
        )
        for family in ("ridge", "fix_harm", "groups")
    }
    plan = {
        "gate_sha256": sha256(ORIGINAL),
        "fit_rows": len(fit),
        "fit_fixes": int((teacher & ~student).sum()),
        "fit_harms": int((student & ~teacher).sum()),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "confidence_point": confidence,
        "confidence_anchors": conf_anchors,
        "primary_family": min(finalists, key=lambda name: preference(finalists[name])),
        "finalists": finalists,
        "selection_baselines": {
            name: route_metrics(data, data["baseline"]) for name, data in selection_sets.items()
        },
        "grid": grid,
    }
    write_json(output / "plan.json", plan)
    write_json(
        output / "development-patterns.json",
        {
            name: patterns(data["rows"], data["teacher"], data["student"], data["baseline"])
            for name, data in selection_sets.items()
        },
    )
    print(
        f"Frozen {len(grid)} candidates; finalists {finalists}, primary {plan['primary_family']}",
        flush=True,
    )


def infer_imdb(output, gate):
    previous = read_json(ARCHIVE / "cost-weighted/external-selection.json")
    earlier = read_json(ARCHIVE / "soft-cost/imdb-selection.json")
    dataset_name = previous["dataset_id"]
    revision = previous["revision"]
    test = load_dataset(dataset_name, split="test", revision=revision)
    train = load_dataset(dataset_name, split="train", revision=revision)
    seen = {normalized(test[r["dataset_index"]]["text"]) for r in previous["rows"]}
    seen.update(normalized(train[r["idx"]]["text"]) for r in earlier["rows"])
    config = load_config(ROOT / "configs/default.toml")
    manifest = read_json(ARCHIVE / "manifest.json")
    if model_source(config) != manifest["model_source"]:
        raise ValueError("Tokenizer model source differs")
    tokenizer = load_tokenizer(config)
    counts = np.zeros((3, 2), dtype=int)
    rows, inputs = [], []
    for index in np.random.default_rng(2039).permutation(len(test)):
        row = test[int(index)]
        text, label = row["text"], row["label"]
        key = normalized(text)
        if key in seen or np.all(counts[:, label] >= 120):
            continue
        encoded = tokenizer(text, truncation=False, verbose=False, return_tensors="np")
        tokens = encoded["input_ids"].shape[1]
        if tokens > 160:
            continue
        group = int(np.digitize(tokens, [97, 129]))
        if counts[group, label] >= 120:
            continue
        counts[group, label] += 1
        seen.add(key)
        rows.append(
            {
                "idx": int(index),
                "label": label,
                "tokens": tokens,
                "text_sha256": hashlib.sha256(key.encode()).hexdigest(),
                "signals": signals(text, tokens),
            }
        )
        inputs.append({name: encoded[name] for name in ("input_ids", "attention_mask")})
        if np.all(counts == 120):
            break
    if len(rows) != 720:
        raise ValueError(f"Fresh length/label quotas not met: {counts.tolist()}")
    write_json(
        output / "imdb-selection.json",
        {
            "dataset_id": dataset_name,
            "revision": revision,
            "split": "test",
            "seed": 2039,
            "rows": rows,
        },
    )
    print("Locked 720 IMDb review IDs before inference", flush=True)
    teacher_path = config.artifact_dir / "models/dynamic.onnx"
    student_path = ROOT / "artifacts/student-study/onnx-verified/four_layer_distilled.onnx"
    export = read_json(ARCHIVE / "onnx-export-summary.json")["four_layer_distilled"]
    if (
        sha256(teacher_path) != gate["teacher_model_sha256"]
        or sha256(student_path) != export["onnx_sha256"]
    ):
        raise ValueError("ONNX models do not match the archive")
    teacher, _ = session(teacher_path, threads=4, allow_spinning=False)
    student, _ = session(student_path, threads=4, allow_spinning=False)
    for i, (row, inputs_for_row) in enumerate(zip(rows, inputs, strict=True), start=1):
        row["teacher_logits"] = teacher.run(["logits"], inputs_for_row)[0][0].tolist()
        row["student_logits"] = student.run(["logits"], inputs_for_row)[0][0].tolist()
        if i % 240 == 0:
            print(f"Scored {i} / 720 IMDb reviews", flush=True)
    write_json(
        output / "imdb-logits.json",
        {
            "rows": rows,
            "teacher_sha256": gate["teacher_model_sha256"],
            "student_sha256": export["onnx_sha256"],
        },
    )
    return rows


def evaluate(output, plan, gate, rows):
    data = cohort(rows, gate["parameters"], plan["confidence_point"], plan["center"], plan["scale"])
    predictions = {}
    baseline = np.where(data["baseline"], data["teacher"], data["student"])
    for name, index in plan["finalists"].items():
        selected = plan["grid"][index]
        values = blend_score(
            raw_score(selected["model"], rows, data["matrix"]),
            data["margin"],
            selected,
            plan["confidence_anchors"],
        )
        matched, _ = mask_ranked(data, values)
        fixed = (values > selected["cutoff"]) | (
            (values == selected["cutoff"]) & (data["margin"] <= selected["cutoff_margin"])
        )
        correct = np.where(matched, data["teacher"], data["student"])
        predictions[name] = {
            "matched": route_metrics(data, matched),
            "fixed": route_metrics(data, fixed),
            "delta_95pct_interval": paired_bootstrap_ci(correct, baseline),
            "wins_vs_confidence": int((correct & ~baseline).sum()),
            "losses_vs_confidence": int((baseline & ~correct).sum()),
            "changed_routes": int((matched != data["baseline"]).sum()),
            "wins_ids": [r["idx"] for r, win in zip(rows, correct & ~baseline, strict=True) if win],
            "losses_ids": [
                r["idx"] for r, loss in zip(rows, baseline & ~correct, strict=True) if loss
            ],
        }
    all_fixes = int((data["teacher"] & ~data["student"]).sum())
    report = {
        "plan_sha256": sha256(output / "plan.json"),
        "logits_sha256": sha256(output / "imdb-logits.json"),
        "n": len(rows),
        "student_correct": int(data["student"].sum()),
        "teacher_correct": int(data["teacher"].sum()),
        "confidence": route_metrics(data, data["baseline"]),
        "oracle_correct": int(data["student"].sum()) + min(all_fixes, data["quota"]),
        "finalists": predictions,
        "patterns": patterns(rows, data["teacher"], data["student"], data["baseline"]),
    }
    write_json(output / "imdb.json", report)
    print({key: value for key, value in report.items() if key != "patterns"}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "imdb"))
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/student-study/gate-signals"
    )
    args = parser.parse_args()
    target = args.output / ("plan.json" if args.stage == "freeze" else "imdb.json")
    if target.exists():
        raise ValueError(f"Output already exists: {target}")
    if args.stage == "freeze":
        freeze(args.output)
        return
    plan = read_json(args.output / "plan.json")
    if plan["gate_sha256"] != sha256(ORIGINAL):
        raise ValueError("Original gate changed")
    gate = read_json(ORIGINAL)
    evaluate(args.output, plan, gate, infer_imdb(args.output, gate))


if __name__ == "__main__":
    main()
