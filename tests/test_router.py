import numpy as np

from etdl import router
from etdl.config import Config
from etdl.io import sha256, write_json


def test_missing_selected_variant_uses_dynamic_and_logs_no_input_text(tmp_path, monkeypatch):
    config = Config(artifact_dir=tmp_path, model_id="test")
    model = tmp_path / "dynamic.onnx"
    model.write_bytes(b"dynamic")
    calibration = tmp_path / "data/calibration.json"
    write_json(calibration, {"split": "train"})
    source = {"model_id": "test"}
    policy_path = tmp_path / "selection.json"
    write_json(
        policy_path,
        {
            "status": "verified_calibration_selection",
            "variants": ["fixed-16"],
            "routing": {"3": "fixed-16"},
            "model_source": source,
            "threads": 1,
            "model_hashes": {"dynamic": sha256(model), "fixed-16": "unavailable"},
            "calibration_sha256": sha256(calibration),
            "atol": 1e-4,
            "rtol": 1e-4,
            "hardware_key": {},
            "measured_memory_bytes": 600,
            "memory_budget_bytes": 1000,
        },
    )
    monkeypatch.setattr(router, "model_source", lambda _: source)
    monkeypatch.setattr(router, "hardware_key", lambda _: {})
    monkeypatch.setattr(router, "environment", dict)
    monkeypatch.setattr(router, "verified_dynamic", lambda _: model)
    monkeypatch.setattr(router, "verified_variants", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        router,
        "load_tokenizer",
        lambda _: (
            lambda *args, **kwargs: {
                "input_ids": np.array([[101, 42, 102]], dtype=np.int64),
                "attention_mask": np.ones((1, 3), dtype=np.int64),
            }
        ),
    )

    class DynamicSession:
        def run(self, outputs, inputs):
            assert inputs["input_ids"].shape == (1, 3)
            return [np.array([[-1.0, 1.0]])]

    monkeypatch.setattr(router, "session", lambda *args: (DynamicSession(), 0))
    classifier = router.RoutedClassifier(config, policy_path)
    result = classifier.predict("Private input text")
    assert result["route"] == "dynamic"
    assert result["label"] == "positive"
    assert "fixed-16" in result["disabled_variants"]
    log = (tmp_path / "reports/routing-decisions.jsonl").read_text()
    assert "Private input text" not in log
