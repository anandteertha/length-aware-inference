from collections import Counter

from etdl.io import read_json, sha256, write_json

DATASET_ID = "nyu-mll/glue"


def model_source(config):
    """Resolve once so the tokenizer, PyTorch model, and export share a revision."""
    path = config.artifact_dir / "model_source.json"
    if path.exists():
        source = read_json(path)
        if (source["model_id"], source["requested_revision"]) != (
            config.model_id,
            config.model_revision,
        ):
            raise ValueError("Model configuration changed; use a new artifact directory")
        return source
    from huggingface_hub import HfApi

    source = {
        "model_id": config.model_id,
        "requested_revision": config.model_revision,
        "revision": HfApi()
        .model_info(config.model_id, revision=config.model_revision, timeout=30)
        .sha,
    }
    write_json(path, source)
    return source


def load_tokenizer(config):
    from transformers import AutoTokenizer

    source = model_source(config)
    # A pinned, cached snapshot needs no repeated network metadata requests.
    try:
        return AutoTokenizer.from_pretrained(
            source["model_id"], revision=source["revision"], local_files_only=True
        )
    except OSError:
        return AutoTokenizer.from_pretrained(source["model_id"], revision=source["revision"])


def encode(tokenizer, text, max_length):
    """Never silently truncate. Reject inputs beyond the encoder's supported length."""
    encoded = tokenizer(text, truncation=False, return_tensors="np")
    length = encoded["input_ids"].shape[1]
    if length > max_length:
        raise ValueError(
            f"Input has {length} tokens; supported maximum is {max_length}; not truncated"
        )
    return {name: encoded[name] for name in ("input_ids", "attention_mask")}


def prepare(config, limit):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    if limit <= 0:
        raise ValueError("limit must be positive")
    tokenizer = load_tokenizer(config)
    revision = HfApi().dataset_info(DATASET_ID, timeout=30).sha
    # Only the training split is exposed to the calibration pipeline.
    train = load_dataset(DATASET_ID, "sst2", split="train", revision=revision)
    selected = train.shuffle(seed=config.seed).select(range(min(limit, len(train))))
    rows = []
    for row in selected:
        encoded = encode(tokenizer, row["sentence"], config.max_length)
        rows.append(
            {
                "idx": int(row["idx"]),
                "text": row["sentence"],
                "label": int(row["label"]),
                "tokens": int(encoded["input_ids"].shape[1]),
            }
        )
    path = config.artifact_dir / "data/calibration.json"
    write_json(
        path,
        {
            "dataset_id": DATASET_ID,
            "dataset_config": "sst2",
            "revision": revision,
            "split": "train",
            "seed": config.seed,
            "model_source": model_source(config),
            "max_length": config.max_length,
            "train_fingerprint": train._fingerprint,
            "rows": rows,
        },
    )
    write_json(
        config.artifact_dir / "reports/lengths.json",
        {
            "split": "train",
            "examples": len(rows),
            "calibration_sha256": sha256(path),
            "length_counts": dict(sorted(Counter(row["tokens"] for row in rows).items())),
            "includes_special_tokens": True,
            "truncated_examples": 0,
        },
    )
    return {"examples": len(rows), "split": "train", "path": str(path)}


def load_calibration(config, limit=None):
    payload = read_json(config.artifact_dir / "data/calibration.json")
    if payload["split"] != "train":
        raise ValueError("Calibration must use the training split; validation is reserved")
    if (
        payload["model_source"] != model_source(config)
        or payload["max_length"] != config.max_length
    ):
        raise ValueError(
            "Calibration configuration differs; rerun prepare in a fresh artifact directory"
        )
    rows = payload["rows"]
    if not rows:
        raise ValueError("Calibration data is empty")
    return rows if limit is None else rows[:limit]
