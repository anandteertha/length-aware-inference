"""Time a short CPU training step for the proposed three-layer student.

This is a feasibility probe, not a trained checkpoint or an accuracy experiment.
Run after ``uv run etdl prepare --limit 8`` has created calibration rows.
"""

import copy
import json
import statistics
import time
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F

from etdl.config import load_config
from etdl.data import load_tokenizer
from etdl.io import read_json
from etdl.model import load_pytorch


def main() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.toml")
    torch.manual_seed(config.seed)
    teacher = load_pytorch(config, threads=4)
    tokenizer = load_tokenizer(config)
    rows = read_json(config.artifact_dir / "data/calibration.json")["rows"][:8]
    if len(rows) != 8:
        raise ValueError("Need at least eight prepared training rows")
    batch = tokenizer([row["text"] for row in rows], padding=True, return_tensors="pt")
    if batch["input_ids"].shape[1] > config.max_length:
        raise ValueError("The batch exceeds the supported token length")
    labels = torch.tensor([row["label"] for row in rows])

    student_config = copy.deepcopy(teacher.config)
    student_config.n_layers = 3
    student = type(teacher)(student_config).cpu().float()
    weights = {
        name: value
        for name, value in teacher.state_dict().items()
        if not any(name.startswith(f"distilbert.transformer.layer.{i}.") for i in range(3, 6))
    }
    student.load_state_dict(weights, strict=True)
    student.train()

    teacher_start = time.perf_counter()
    with torch.inference_mode():
        teacher_logits = teacher(**batch).logits.detach()
    teacher_ms = (time.perf_counter() - teacher_start) * 1000

    optimizer = torch.optim.AdamW(student.parameters(), lr=2e-5)
    first_weight = student.classifier.weight.detach().clone()
    step_ms = []
    rss_mib = []
    for step in range(6):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        student_logits = student(**batch).logits
        ce = F.cross_entropy(student_logits, labels)
        kd = F.kl_div(
            F.log_softmax(student_logits / 2.0, dim=-1),
            F.softmax(teacher_logits / 2.0, dim=-1),
            reduction="batchmean",
        ) * 4.0
        loss = 0.5 * ce + 0.5 * kd
        loss.backward()
        optimizer.step()
        elapsed_ms = (time.perf_counter() - start) * 1000
        if step:
            step_ms.append(elapsed_ms)
            rss_mib.append(psutil.Process().memory_info().rss / (1024**2))

    print(
        json.dumps(
            {
                "purpose": "CPU training feasibility only; no accuracy claim",
                "threads": torch.get_num_threads(),
                "batch_size": len(rows),
                "padded_tokens": int(batch["input_ids"].shape[1]),
                "teacher_layers": int(teacher.config.n_layers),
                "student_layers": int(student.config.n_layers),
                "teacher_parameters": sum(p.numel() for p in teacher.parameters()),
                "student_parameters": sum(p.numel() for p in student.parameters()),
                "teacher_forward_ms": round(teacher_ms, 2),
                "student_step_ms": [round(value, 2) for value in step_ms],
                "student_step_median_ms": round(statistics.median(step_ms), 2),
                "observed_rss_max_mib": round(max(rss_mib), 1),
                "classifier_weight_changed": bool(
                    torch.any(first_weight != student.classifier.weight.detach())
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
