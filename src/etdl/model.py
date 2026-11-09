import time

from etdl.data import load_tokenizer, model_source
from etdl.io import read_json, sha256, write_json
from etdl.runtime import graph_operators, session


def load_pytorch(config, threads=1):
    import torch
    from transformers import AutoModelForSequenceClassification

    torch.set_num_threads(threads)
    source = model_source(config)
    # Eager attention avoids backend-dependent SDPA dispatch during export/comparison.
    options = {"revision": source["revision"], "attn_implementation": "eager"}
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            source["model_id"], local_files_only=True, **options
        )
    except OSError:
        model = AutoModelForSequenceClassification.from_pretrained(source["model_id"], **options)
    return model.cpu().float().eval()


def export(config):
    import onnx
    import torch

    root = config.artifact_dir / "models"
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "dynamic.onnx"
    manifest_path = root / "dynamic.json"
    if model_path.exists() or manifest_path.exists():
        raise ValueError(
            "Export already exists; use a new artifact directory to preserve provenance"
        )
    model = load_pytorch(config)
    tokenizer = load_tokenizer(config)

    class Logits(torch.nn.Module):
        def __init__(self, classifier):
            super().__init__()
            self.classifier = classifier

        def forward(self, input_ids, attention_mask):
            return self.classifier(input_ids=input_ids, attention_mask=attention_mask).logits

    inputs = tokenizer("A useful first CPU baseline.", return_tensors="pt")
    length = torch.export.Dim("sequence", min=2, max=config.max_length)
    start = time.perf_counter()
    with torch.inference_mode():
        torch.onnx.export(
            Logits(model).eval(),
            (inputs["input_ids"], inputs["attention_mask"]),
            str(model_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_shapes=({1: length}, {1: length}),
            dynamo=True,
            verbose=False,  # Default exporter progress contains Unicode unsupported by Windows pipes.
            opset_version=18,
            external_data=False,
        )
    onnx.checker.check_model(str(model_path))
    export_seconds = time.perf_counter() - start
    optimized_path = root / "dynamic.optimized.onnx"
    ort_session, startup_ms = session(model_path, optimized_path=optimized_path)
    manifest = {
        "model_source": model_source(config),
        "batch_size": 1,
        "sequence_length": "dynamic",
        "max_length": config.max_length,
        "precision": "fp32",
        "export_and_validation_seconds": export_seconds,
        "session_initialization_ms": startup_ms,
        "provider": ort_session.get_providers(),
        "sha256": sha256(model_path),
        "size_bytes": model_path.stat().st_size,
        "optimized_sha256": sha256(optimized_path),
        "optimized_size_bytes": optimized_path.stat().st_size,
        "operators_before": graph_operators(model_path),
        "operators_after": graph_operators(optimized_path),
        "optimization_level": "ORT_ENABLE_ALL",
        "intra_op_threads": 1,
    }
    write_json(manifest_path, manifest)
    return manifest


def checked_model_path(config):
    path = config.artifact_dir / "models/dynamic.onnx"
    manifest = read_json(path.with_suffix(".json"))
    if (
        manifest["model_source"] != model_source(config)
        or manifest["max_length"] != config.max_length
    ):
        raise ValueError("Export configuration differs; use a fresh artifact directory")
    if sha256(path) != manifest["sha256"]:
        raise ValueError("ONNX artifact changed after export")
    return path
