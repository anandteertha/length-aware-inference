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


def test_deployment_selection_uses_all_runs_and_only_allowed_candidates():
    from etdl.comparison import deployment_candidate

    values = [
        {
            "method": "dynamic",
            "threads": 1,
            "runs": 3,
            "passed": True,
            "changed_predictions": 0,
            "max_peak_bytes": 600,
            "median_run_p95_ms": 10,
        },
        {
            "method": "selected",
            "threads": 2,
            "runs": 3,
            "passed": True,
            "changed_predictions": 0,
            "max_peak_bytes": 1200,
            "median_run_p95_ms": 5,
        },
        {
            "method": "buckets",
            "threads": 4,
            "runs": 3,
            "passed": True,
            "changed_predictions": 0,
            "max_peak_bytes": 800,
            "median_run_p95_ms": 1,
        },
    ]
    assert deployment_candidate(values, 1300)["threads"] == 2
    assert deployment_candidate(values, 1000)["method"] == "dynamic"
    values[1]["passed"] = False
    assert deployment_candidate(values, 1300)["method"] == "dynamic"
    with pytest.raises(ValueError, match="No confirmed deployment"):
        deployment_candidate(values, 500)


def test_frozen_policy_tampering_prevents_validation_access(tmp_path, monkeypatch):
    from etdl import comparison
    from etdl.config import Config
    from etdl.io import sha256, write_json

    config = Config(artifact_dir=tmp_path, model_id="test")
    calibration = tmp_path / "data/calibration.json"
    write_json(calibration, {"split": "train"})
    frozen = {
        "status": "frozen_for_validation",
        "calibration_sha256": sha256(calibration),
        "model_source": {},
        "hardware_key": {},
    }
    for name in ("comparison_plan", "comparison_summary", "frozen_policy", "execution_plan"):
        path = tmp_path / f"{name}.json"
        write_json(path, {})
        frozen[name], frozen[f"{name}_sha256"] = str(path), sha256(path)
    path = tmp_path / "evaluation-plan.json"
    write_json(path, frozen)
    monkeypatch.setattr(comparison, "model_source", lambda _: {})
    monkeypatch.setattr(comparison, "environment", dict)
    monkeypatch.setattr(comparison, "hardware_key", lambda _: {})
    monkeypatch.setattr(comparison, "validate_plan", lambda *args: {})
    comparison.validate_frozen_plan(config, path)
    write_json(tmp_path / "frozen_policy.json", {"threads": 99})
    with pytest.raises(ValueError, match="Frozen frozen_policy changed"):
        comparison.validate_frozen_plan(config, path)


def test_rejected_specialization_is_retained_as_comparator_not_deployed(tmp_path, monkeypatch):
    from etdl import comparison
    from etdl.config import Config
    from etdl.io import read_json, sha256, write_json

    config = Config(artifact_dir=tmp_path, model_id="test")
    write_json(tmp_path / "data/calibration.json", {"revision": "dataset-revision"})
    directory = tmp_path / "confirmation"
    policy = {
        "variants": ["fixed-16", "fixed-32"],
        "routing": {"5": "fixed-16"},
        "memory_budget_bytes": 1500,
        "model_hashes": {"dynamic": "d", "fixed-16": "a", "fixed-32": "b"},
    }
    plan = {
        "policy": policy,
        "model_hashes": policy["model_hashes"],
        "workload_sha256": "w",
        "calibration_sha256": "c",
        "hardware_key": {},
        "model_source": {},
    }
    write_json(directory / "plan.json", plan)
    group = {
        "method": "dynamic",
        "threads": 4,
        "runs": 3,
        "passed": True,
        "changed_predictions": 0,
        "max_peak_bytes": 650,
        "median_run_p95_ms": 78,
    }
    write_json(
        directory / "summary.json",
        {
            "status": "complete",
            "all_correctness_checks_passed": True,
            "plan_sha256": sha256(directory / "plan.json"),
            "groups": [group],
        },
    )
    monkeypatch.setattr(comparison, "validate_plan", lambda *args: {})
    frozen = comparison.freeze(config, directory)
    assert frozen["chosen_confirmation_method"] == "dynamic"
    assert read_json(directory / "frozen-policy.json")["variants"] == []
    assert read_json(directory / "evaluation-execution.json")["policy"]["variants"] == [
        "fixed-16",
        "fixed-32",
    ]
