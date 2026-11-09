import numpy as np

from etdl.data import encode, load_calibration, load_tokenizer
from etdl.io import read_json, sha256, write_json
from etdl.model import checked_model_path, load_pytorch
from etdl.runtime import session


def compare_logits(reference, actual, atol, rtol):
    reference, actual = np.asarray(reference), np.asarray(actual)
    if (
        reference.shape != actual.shape
        or not np.isfinite(reference).all()
        or not np.isfinite(actual).all()
    ):
        return {"passed": False, "labels_match": False, "max_absolute_error": None}
    labels_match = bool(np.array_equal(reference.argmax(-1), actual.argmax(-1)))
    return {
        "passed": bool(np.allclose(reference, actual, atol=atol, rtol=rtol) and labels_match),
        "labels_match": labels_match,
        "max_absolute_error": float(np.max(np.abs(reference - actual))),
    }


def verified_dynamic(config, rows=None):
    path = checked_model_path(config)
    report = read_json(config.artifact_dir / "reports/checks.json")
    if not report["passed"] or report["model_sha256"] != sha256(path):
        raise ValueError("A passing output check for this exact dynamic model is required")
    if report["calibration_sha256"] != sha256(config.artifact_dir / "data/calibration.json"):
        raise ValueError("Calibration data changed; rerun output checks")
    if (report["atol"], report["rtol"]) != (config.atol, config.rtol):
        raise ValueError("Output tolerances changed; rerun output checks")
    if rows is not None:
        checked = {case["case"] for case in report["cases"]}
        if any(f"train:{row['idx']}" not in checked for row in rows):
            raise ValueError("Output checks do not cover every requested training example")
    return path


def check(config, limit):
    import torch

    model_path = checked_model_path(config)
    model = load_pytorch(config)
    tokenizer = load_tokenizer(config)
    runtime, _ = session(model_path)
    cases = []

    def compare(name, inputs):
        with torch.inference_mode():
            reference = model(
                **{key: torch.from_numpy(value.copy()) for key, value in inputs.items()}
            )
        actual = runtime.run(["logits"], inputs)[0]
        result = compare_logits(reference.logits.numpy(), actual, config.atol, config.rtol)
        cases.append({"case": name, "length": inputs["input_ids"].shape[1], **result})
        return actual

    for row in load_calibration(config, limit):
        compare(f"train:{row['idx']}", encode(tokenizer, row["text"], config.max_length))
    # Cross lengths above the planned fixed variants; full, right-padded, and internal masks.
    for length in (2, 8, 16, 17, 32, 64, 128, 140, 256, 512):
        if length > config.max_length:
            continue
        ids = np.full((1, length), tokenizer.convert_tokens_to_ids("good"), dtype=np.int64)
        ids[0, 0], ids[0, -1] = tokenizer.cls_token_id, tokenizer.sep_token_id
        for mask_name in ("full", "right_pad", "internal"):
            mask = np.ones_like(ids)
            masked_ids = ids.copy()
            if mask_name == "right_pad" and length > 2:
                end = max(2, length // 2)
                masked_ids[0, end - 1] = tokenizer.sep_token_id
                masked_ids[0, end:] = tokenizer.pad_token_id
                mask[0, end:] = 0
            elif mask_name == "internal" and length > 2:
                mask[0, 1:-1:2] = 0
                masked_ids[mask == 0] = tokenizer.pad_token_id
            compare(
                f"synthetic:{mask_name}:{length}",
                {
                    "input_ids": masked_ids,
                    "attention_mask": mask,
                },
            )
    # Padding must preserve the unpadded dynamic result, not merely agree across runtimes.
    base = encode(tokenizer, "Great service!", config.max_length)
    baseline = runtime.run(["logits"], base)[0]
    for length in (16, 32, 64, 128):
        if length < base["input_ids"].shape[1] or length > config.max_length:
            continue
        pad = length - base["input_ids"].shape[1]
        padded = {
            "input_ids": np.pad(
                base["input_ids"], ((0, 0), (0, pad)), constant_values=tokenizer.pad_token_id
            ),
            "attention_mask": np.pad(base["attention_mask"], ((0, 0), (0, pad))),
        }
        actual = runtime.run(["logits"], padded)[0]
        cases.append(
            {
                "case": f"padding_invariance:{length}",
                "length": length,
                **compare_logits(baseline, actual, config.atol, config.rtol),
            }
        )
    report = {
        "passed": all(case["passed"] for case in cases),
        "case_count": len(cases),
        "changed_labels": sum(not case["labels_match"] for case in cases),
        "max_absolute_error": max((case["max_absolute_error"] or 0) for case in cases),
        "atol": config.atol,
        "rtol": config.rtol,
        "cases": cases,
        "model_sha256": sha256(model_path),
        "calibration_sha256": sha256(config.artifact_dir / "data/calibration.json"),
        "scope": "Sampled training examples and synthetic masks; not a proof of equivalence.",
    }
    write_json(config.artifact_dir / "reports/checks.json", report)
    if not report["passed"]:
        raise ValueError("Output checks failed; inspect artifacts/reports/checks.json")
    return {key: value for key, value in report.items() if key != "cases"}
