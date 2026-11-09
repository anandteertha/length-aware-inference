import numpy as np
import pytest

from etdl.benchmark import latency_summary
from etdl.checks import compare_logits
from etdl.config import Config, load_config
from etdl.data import encode, load_calibration
from etdl.io import write_json


def test_config_resolves_artifacts_relative_to_file(tmp_path):
    path = tmp_path / "default.toml"
    path.write_text('artifact_dir = "results"\nmodel_id = "test"\n', encoding="utf-8")
    assert load_config(path).artifact_dir == tmp_path / "results"


def test_calibration_rejects_held_out_data(tmp_path):
    config = Config(artifact_dir=tmp_path, model_id="test")
    write_json(tmp_path / "data/calibration.json", {"split": "validation"})
    with pytest.raises(ValueError, match="training split"):
        load_calibration(config)


def test_calibration_rejects_different_tokenizer_revision(tmp_path):
    config = Config(artifact_dir=tmp_path, model_id="test")
    source = {"model_id": "test", "requested_revision": "main", "revision": "abc"}
    write_json(tmp_path / "model_source.json", source)
    write_json(
        tmp_path / "data/calibration.json",
        {
            "split": "train",
            "model_source": {**source, "revision": "different"},
            "max_length": 512,
        },
    )
    with pytest.raises(ValueError, match="configuration differs"):
        load_calibration(config)


def test_label_agreement_is_required_even_when_logits_are_close():
    result = compare_logits([[0.00001, 0.0]], [[0.0, 0.00001]], atol=1e-4, rtol=1e-4)
    assert not result["passed"]
    assert not result["labels_match"]


@pytest.mark.parametrize("actual", [[[np.nan, 1]], [[np.inf, 1]], [[1, 2, 3]]])
def test_nonfinite_or_wrong_shape_outputs_fail(actual):
    assert not compare_logits([[1, 2]], actual, 1e-4, 1e-4)["passed"]


def test_logits_drift_fails_even_if_label_matches():
    result = compare_logits([[1.0, 2.0]], [[1.0, 4.0]], atol=1e-4, rtol=1e-4)
    assert result["labels_match"]
    assert not result["passed"]


def test_encoder_rejects_overlength_instead_of_truncating():
    def tokenizer(text, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": np.ones((1, 513)), "attention_mask": np.ones((1, 513))}

    with pytest.raises(ValueError, match="not truncated"):
        encode(tokenizer, "too long", 512)


def test_latency_summary_uses_request_percentile():
    summary = latency_summary([1, 2, 3, 4, 100])
    assert summary["median_ms"] == 3
    assert summary["p95_ms"] == pytest.approx(80.8)


@pytest.mark.parametrize("values", [[], [-1], [np.nan], [np.inf]])
def test_invalid_timings_rejected(values):
    with pytest.raises(ValueError):
        latency_summary(values)


def test_invalid_max_length_rejected(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text('artifact_dir = "results"\nmodel_id = "test"\nmax_length = 513\n')
    with pytest.raises(ValueError, match="max_length"):
        load_config(path)
