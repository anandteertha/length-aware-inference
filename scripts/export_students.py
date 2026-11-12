"""Export trained students to dynamic ONNX and check outputs on training rows."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import onnx
import torch

from etdl.config import load_config
from etdl.data import load_tokenizer
from etdl.io import read_json, sha256, write_json
from etdl.model import load_pytorch
from etdl.runtime import session


class Logits(torch.nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, input_ids, attention_mask):
        return self.classifier(input_ids=input_ids, attention_mask=attention_mask).logits


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("artifacts/student-study"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main():
    args = arguments()
    study = args.study.resolve()
    summary = read_json(study / "summary.json")
    if sha256(study / "manifest.json") != summary["manifest_sha256"]:
        raise ValueError("Training manifest hash changed")
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.toml")
    torch.set_num_threads(args.threads)
    teacher = load_pytorch(config, threads=args.threads)
    tokenizer = load_tokenizer(config)
    rows = read_json(config.artifact_dir / "data/calibration.json")["rows"][:16]
    if len(rows) != 16:
        raise ValueError("Need 16 prepared training rows for output checks")
    models_dir = args.output.resolve() if args.output else study / "onnx-verified"
    models_dir.mkdir(parents=True, exist_ok=False)
    results = {}
    capacity_path = study / "capacity-summary.json"
    capacity = read_json(capacity_path) if capacity_path.exists() else None
    modes = ("supervised", "distilled") + (("four_layer_distilled",) if capacity else ())
    for mode in modes:
        checkpoint = study / f"{mode}.pt"
        expected_hash = (
            capacity["checkpoint_sha256"]
            if mode == "four_layer_distilled"
            else summary["student_arms"][mode]["checkpoint_sha256"]
        )
        if sha256(checkpoint) != expected_hash:
            raise ValueError(f"{mode} checkpoint hash changed")
        student_config = copy.deepcopy(teacher.config)
        student_config.n_layers = 4 if mode == "four_layer_distilled" else 3
        student = type(teacher)(student_config).cpu().float().eval()
        student.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        wrapper = Logits(student).eval()
        sample = tokenizer("A useful first CPU baseline.", return_tensors="pt")
        path = models_dir / f"{mode}.onnx"
        start = time.perf_counter()
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (sample["input_ids"], sample["attention_mask"]),
                str(path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_shapes=(
                    {1: torch.export.Dim("sequence", min=2, max=config.max_length)},
                    {1: torch.export.Dim("sequence", min=2, max=config.max_length)},
                ),
                dynamo=True,
                verbose=False,
                opset_version=18,
                external_data=False,
            )
        onnx.checker.check_model(str(path))
        optimized = models_dir / f"{mode}.optimized.onnx"
        runtime, startup_ms = session(path, threads=args.threads, optimized_path=optimized)
        max_absolute_error = 0.0
        changed_labels = 0
        for row in rows:
            tokenized = tokenizer(row["text"], truncation=False, return_tensors="pt")
            encoded = {name: tokenized[name] for name in ("input_ids", "attention_mask")}
            if encoded["input_ids"].shape[1] > config.max_length:
                raise ValueError("Check row exceeds max_length")
            with torch.inference_mode():
                reference = wrapper(**encoded).numpy()
            actual = runtime.run(
                ["logits"],
                {name: value.numpy() for name, value in encoded.items()},
            )[0]
            if not np.all(np.isfinite(actual)) or not np.allclose(
                reference, actual, atol=config.atol, rtol=config.rtol
            ):
                raise ValueError(f"{mode} ONNX logits differ from PyTorch on row {row['idx']}")
            max_absolute_error = max(max_absolute_error, float(np.max(np.abs(actual - reference))))
            changed_labels += int(reference.argmax() != actual.argmax())
        results[mode] = {
            "checkpoint_sha256": sha256(checkpoint),
            "onnx_sha256": sha256(path),
            "onnx_bytes": path.stat().st_size,
            "optimized_sha256": sha256(optimized),
            "optimized_bytes": optimized.stat().st_size,
            "export_and_check_seconds": round(time.perf_counter() - start, 1),
            "session_startup_ms": round(startup_ms, 1),
            "checked_training_rows": len(rows),
            "max_absolute_logit_error": max_absolute_error,
            "changed_labels": changed_labels,
        }
        print(json.dumps({"event": "exported", "mode": mode, **results[mode]}), flush=True)
        del student, wrapper, runtime
    write_json(models_dir / "export-summary.json", results)


if __name__ == "__main__":
    main()
