"""Fixed-shape graphs derived from the checked dynamic graph, with unchanged weights."""

import hashlib

import numpy as np

from etdl.checks import compare_logits, verified_dynamic
from etdl.data import encode, load_calibration, load_tokenizer, model_source
from etdl.io import read_json, sha256, write_json
from etdl.runtime import graph_operators, session

LENGTHS = (16, 32, 64, 128)


def pad_inputs(inputs, length, pad_token_id):
    current = inputs["input_ids"].shape[1]
    if current > length:
        raise ValueError(f"Cannot fit {current} tokens into {length}; no truncation allowed")
    padding = ((0, 0), (0, length - current))
    return {
        "input_ids": np.pad(inputs["input_ids"], padding, constant_values=pad_token_id),
        "attention_mask": np.pad(inputs["attention_mask"], padding, constant_values=0),
    }


def weights_digest(model):
    digest = hashlib.sha256()
    for tensor in sorted(model.graph.initializer, key=lambda value: value.name):
        digest.update(tensor.SerializeToString(deterministic=True))
    return digest.hexdigest()


def build_variant(config, length):
    import onnx
    from onnxruntime.tools.onnx_model_utils import fix_output_shapes, make_input_shape_fixed

    if length not in LENGTHS or length > config.max_length:
        raise ValueError(f"Variant length must be one of {LENGTHS} and within max_length")
    baseline = verified_dynamic(config)
    path = config.artifact_dir / f"models/fixed-{length}.onnx"
    if path.exists() or path.with_suffix(".json").exists():
        # Resume only if provenance and file integrity still match.
        manifest = read_json(path.with_suffix(".json"))
        if manifest["base_sha256"] != sha256(baseline) or manifest["sha256"] != sha256(path):
            raise ValueError("Existing variant changed or belongs to another baseline")
        return manifest
    model = onnx.load(str(baseline))
    before = weights_digest(model)
    for name in ("input_ids", "attention_mask"):
        make_input_shape_fixed(model.graph, name, [1, length])
    fix_output_shapes(model)
    if weights_digest(model) != before:
        raise ValueError("Shape specialization changed model weights")
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    del model
    optimized = path.with_name(f"fixed-{length}.optimized.onnx")
    runtime, startup_ms = session(path, optimized_path=optimized)
    if any(value.shape != [1, length] for value in runtime.get_inputs()):
        raise ValueError("Variant did not specialize both input shapes")
    manifest = {
        "length": length,
        "model_source": model_source(config),
        "precision": "fp32",
        "base_sha256": sha256(baseline),
        "sha256": sha256(path),
        "weights_sha256": before,
        "size_bytes": path.stat().st_size,
        "optimized_sha256": sha256(optimized),
        "optimized_size_bytes": optimized.stat().st_size,
        "session_initialization_ms": startup_ms,
        "operators_after": graph_operators(optimized),
        "provider": runtime.get_providers(),
        "optimization_level": "ORT_ENABLE_ALL",
    }
    write_json(path.with_suffix(".json"), manifest)
    return manifest


def check_variant(config, length, limit):
    rows = load_calibration(config, limit)
    baseline = verified_dynamic(config, rows)
    path = config.artifact_dir / f"models/fixed-{length}.onnx"
    manifest = read_json(path.with_suffix(".json"))
    if manifest["base_sha256"] != sha256(baseline) or manifest["sha256"] != sha256(path):
        raise ValueError("Variant provenance or checksum mismatch")
    dynamic, _ = session(baseline)
    fixed, _ = session(path)
    tokenizer = load_tokenizer(config)
    cases = []
    checked_ids = []

    def compare(name, inputs):
        padded = pad_inputs(inputs, length, tokenizer.pad_token_id)
        actual = fixed.run(["logits"], padded)[0]
        for comparison, feed in (("unpadded", inputs), ("same_padding", padded)):
            reference = dynamic.run(["logits"], feed)[0]
            cases.append(
                {
                    "case": f"{name}:{comparison}",
                    **compare_logits(reference, actual, config.atol, config.rtol),
                }
            )

    for row in rows:
        if row["tokens"] <= length:
            compare(f"train:{row['idx']}", encode(tokenizer, row["text"], config.max_length))
            checked_ids.append(row["idx"])
    for size in sorted({2, max(2, length // 2), length - 1, length}):
        ids = np.full((1, size), tokenizer.convert_tokens_to_ids("good"), dtype=np.int64)
        ids[0, 0], ids[0, -1] = tokenizer.cls_token_id, tokenizer.sep_token_id
        for mode in ("full", "internal"):
            mask = np.ones_like(ids)
            if mode == "internal" and size > 2:
                mask[0, 1:-1:2] = 0
            compare(f"synthetic:{size}:{mode}", {"input_ids": ids, "attention_mask": mask})
    report = {
        "passed": bool(cases) and all(case["passed"] for case in cases),
        "length": length,
        "case_count": len(cases),
        "cases": cases,
        "checked_train_ids": checked_ids,
        "atol": config.atol,
        "rtol": config.rtol,
        "changed_labels": sum(not case["labels_match"] for case in cases),
        "max_absolute_error": max((case["max_absolute_error"] or 0) for case in cases),
        "model_sha256": sha256(path),
        "base_sha256": sha256(baseline),
        "calibration_sha256": sha256(config.artifact_dir / "data/calibration.json"),
        "scope": "Sampled training examples, masks and padding; not universal equivalence.",
    }
    write_json(config.artifact_dir / f"reports/checks-fixed-{length}.json", report)
    if not report["passed"]:
        raise ValueError(f"Fixed-{length} failed output checks and must not be routed")
    return {key: value for key, value in report.items() if key != "cases"}


def verified_variants(config, rows, names=None):
    baseline = verified_dynamic(config, rows)
    base_hash = sha256(baseline)
    data_hash = sha256(config.artifact_dir / "data/calibration.json")
    result = {}
    for length in LENGTHS:
        if names is not None and f"fixed-{length}" not in names:
            continue
        check_path = config.artifact_dir / f"reports/checks-fixed-{length}.json"
        if not check_path.exists():
            continue
        check = read_json(check_path)
        if not check["passed"]:
            continue
        path = config.artifact_dir / f"models/fixed-{length}.onnx"
        if (
            check["base_sha256"] != base_hash
            or check["model_sha256"] != sha256(path)
            or check["calibration_sha256"] != data_hash
            or (check["atol"], check["rtol"]) != (config.atol, config.rtol)
        ):
            raise ValueError(f"Stale verification for fixed-{length}; rerun variant checks")
        covered = set(check["checked_train_ids"])
        if any(row["idx"] not in covered for row in rows if row["tokens"] <= length):
            raise ValueError(f"Fixed-{length} checks do not cover the requested workload")
        result[f"fixed-{length}"] = path
    return result
