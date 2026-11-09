from collections import Counter

import pytest

from etdl.comparison import choose_route, method_models, schedule, summarize_runs


def test_randomized_schedule_is_complete_and_reproducible_per_block():
    methods, threads = ["pytorch", "dynamic", "selected"], [1, 2, 4]
    values = schedule(methods, threads, 3, 2027)
    assert values == schedule(methods, threads, 3, 2027)
    for block in range(3):
        pairs = [(item["method"], item["threads"]) for item in values if item["block"] == block]
        assert Counter(pairs) == Counter(
            (method, thread) for method in methods for thread in threads
        )
    assert values[:9] != [{**item, "block": 0} for item in values[9:18]]


def test_baselines_keep_general_fallback_and_do_not_truncate():
    policy = {"variants": ["fixed-16", "fixed-32"], "routing": {"5": "fixed-16"}}
    assert method_models("buckets", policy) == [
        "dynamic",
        "fixed-16",
        "fixed-32",
        "fixed-64",
        "fixed-128",
    ]
    assert choose_route("single-16", 17, policy, {}) == "dynamic"
    assert choose_route("buckets", 18, policy, {}) == "fixed-32"
    assert choose_route("buckets", 140, policy, {}) == "dynamic"
    assert choose_route("selected", 13, policy, {"dynamic", "fixed-16", "fixed-32"}) == "dynamic"


def run(method, threads, block, p95, request_count=128):
    return {
        "method": method,
        "threads": threads,
        "block": block,
        "latency": {"end_to_end_ms": {"p95_ms": p95, "count": request_count}},
        "memory_for_budget_bytes": 100,
        "accuracy": 1.0,
        "passed": True,
        "changed_predictions": 0,
    }


def test_inference_unit_is_process_run_not_pooled_requests():
    values = [
        run("dynamic", 1, 0, 100, 128),
        run("dynamic", 1, 1, 200, 12800),
        run("dynamic", 1, 2, 300, 1),
        run("selected", 1, 0, 50),
        run("selected", 1, 1, 50),
        run("selected", 1, 2, 150),
    ]
    groups, ratios = summarize_runs(values)
    dynamic = next(item for item in groups if item["method"] == "dynamic")
    assert dynamic["runs"] == 3
    assert dynamic["median_run_p95_ms"] == 200
    assert ratios[0]["ratios"] == [2, 4, 2]
    assert ratios[0]["median_ratio"] == 2


def test_unknown_method_cannot_load_an_arbitrary_model():
    with pytest.raises(ValueError, match="Unknown comparison method"):
        method_models("../arbitrary", {})


def test_validation_cannot_be_loaded_before_freezing(tmp_path, monkeypatch):
    import datasets

    from etdl.comparison import prepare_workload
    from etdl.config import Config
    from etdl.io import write_json

    config = Config(artifact_dir=tmp_path, model_id="test")
    write_json(tmp_path / "data/calibration.json", {"revision": "abc", "rows": []})

    def forbidden(*args, **kwargs):
        pytest.fail("Validation was accessed before checking the frozen plan")

    monkeypatch.setattr(datasets, "load_dataset", forbidden)
    with pytest.raises(ValueError, match="requires a frozen"):
        prepare_workload(config, tmp_path / "validation.json", 872, split="validation")


def test_confirmation_excludes_original_selection_ids(tmp_path, monkeypatch):
    import datasets
    import numpy as np

    from etdl import comparison
    from etdl.config import Config
    from etdl.io import write_json

    config = Config(artifact_dir=tmp_path, model_id="test")
    write_json(tmp_path / "data/calibration.json", {"revision": "abc", "rows": [{"idx": 1}]})

    class Dataset(list):
        def shuffle(self, seed):
            return self

    data = Dataset({"idx": index, "label": 1, "sentence": "sample"} for index in (1, 2, 3))

    def loader(*args, **kwargs):
        assert kwargs["split"] == "train" and kwargs["revision"] == "abc"
        return data

    monkeypatch.setattr(datasets, "load_dataset", loader)
    monkeypatch.setattr(
        comparison,
        "load_tokenizer",
        lambda _: (
            lambda *args, **kwargs: {
                "input_ids": np.ones((1, 3), dtype=np.int64),
                "attention_mask": np.ones((1, 3), dtype=np.int64),
            }
        ),
    )
    rows = comparison.prepare_workload(config, tmp_path / "confirmation.json", 2)
    assert [row["idx"] for row in rows] == [2, 3]
