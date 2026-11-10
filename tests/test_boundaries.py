import numpy as np
import pytest

from etdl.boundaries import bucket_capacity, checked_plan, same_power, summarize, synthetic_cases
from etdl.io import write_json
from etdl.variants import pad_inputs


@pytest.mark.parametrize(
    "length,capacity",
    [(9, 16), (16, 16), (17, 32), (32, 32), (33, 64), (64, 64), (65, 128), (128, 128), (129, None)],
)
def test_bucket_boundaries_and_dynamic_fallback(length, capacity):
    assert bucket_capacity(length) == capacity


def test_invalid_lengths_are_rejected():
    for length in (0, 1, 513):
        with pytest.raises(ValueError):
            bucket_capacity(length)


def test_synthetic_inputs_and_padding_preserve_content_and_masks():
    class Tokenizer:
        cls_token_id, sep_token_id, unk_token_id = 101, 102, 100

        def convert_tokens_to_ids(self, word):
            return 1000 + len(word)

    cases = synthetic_cases(Tokenizer())
    assert len(cases) == len({case["id"] for case in cases}) == 39
    for case in cases:
        assert len(case["input_ids"][0]) == case["length"]
        assert case["input_ids"][0][0] == 101 and case["input_ids"][0][-1] == 102
        assert case["attention_mask"] == [[1] * case["length"]]
        capacity = bucket_capacity(case["length"])
        if capacity is None:
            continue
        original = {
            key: np.array(case[key], dtype=np.int64) for key in ("input_ids", "attention_mask")
        }
        padded = pad_inputs(original, capacity, 0)
        for key, value in original.items():
            np.testing.assert_array_equal(padded[key][:, : case["length"]], value)
            assert not padded[key][:, case["length"] :].any()
        with pytest.raises(ValueError, match="no truncation"):
            pad_inputs(original, case["length"] - 1, 0)


def test_changed_code_or_tolerances_invalidate_frozen_plan(tmp_path, monkeypatch):
    from etdl import boundaries
    from etdl.config import Config

    config = Config(artifact_dir=tmp_path, model_id="test")
    monkeypatch.setattr(boundaries, "model_source", lambda _: {"revision": "pinned"})
    monkeypatch.setattr(boundaries, "code_hashes", lambda: {"boundaries.py": "current"})
    plan = {
        "model_source": {"revision": "pinned"},
        "code_hashes": {"boundaries.py": "old"},
        "atol": config.atol,
        "rtol": config.rtol,
        "max_length": config.max_length,
    }
    path = tmp_path / "plan.json"
    write_json(path, plan)
    with pytest.raises(ValueError, match="code or checkpoint"):
        checked_plan(config, path)
    plan["code_hashes"] = {"boundaries.py": "current"}
    plan["atol"] = 0.1
    write_json(path, plan)
    with pytest.raises(ValueError, match="tolerances"):
        checked_plan(config, path)


def test_power_changes_are_detected():
    expected = {"ac_connected": True, "power_scheme_guid": "balanced"}
    assert same_power(expected, expected.copy())
    assert not same_power(expected, expected | {"ac_connected": False})
    assert not same_power(expected, expected | {"power_scheme_guid": "other"})


def test_ratios_pair_within_process_and_do_not_pool_requests():
    def run(block, times):
        return {
            "family": "standard",
            "block": block,
            "samples": [{}],
            "passed": True,
            "changed_predictions": 0,
            "power_consistent": True,
            "groups": [
                {
                    "length": 16,
                    "mode": mode,
                    "latency": {
                        field: {"p95_ms": value}
                        for field in ("inference_ms", "routing_padding_ms", "prepared_input_ms")
                    },
                }
                for mode, value in zip(["natural", "padded", "fixed"], times, strict=True)
            ],
        }

    result = summarize([run(0, [2, 4, 1]), run(1, [20, 40, 10])])
    assert result["processes"] == 2 and result["timed_requests"] == 2
    ratio = next(item for item in result["paired_ratios"] if item["contrast"] == "net_fixed_effect")
    assert ratio["median_ratio"] == 2
    assert [item["ratio"] for item in ratio["paired_blocks"]] == [2, 2]
    natural = next(item for item in result["groups"] if item["mode"] == "natural")
    assert natural["process_runs"] == 2
    assert natural["latency"]["prepared_input_ms"]["median_run_p95_ms"] == 11
