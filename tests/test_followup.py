import pytest

from etdl.followup import partition_rows


def test_followup_partitions_exclude_prior_ids_and_each_other():
    rows = [{"idx": index} for index in range(20)]
    partitions = partition_rows(rows, {0, 2, 4}, sizes=(3, 2, 2))
    assert [row["idx"] for row in partitions["fit"]] == [1, 3, 5]
    groups = [{row["idx"] for row in group} for group in partitions.values()]
    assert not any(group & {0, 2, 4} for group in groups)
    assert sum(map(len, groups)) == len(set.union(*groups)) == 7


def test_duplicate_training_ids_do_not_enter_multiple_partitions():
    rows = [{"idx": index} for index in [1, 1, 2, 3, 3, 4]]
    partitions = partition_rows(rows, set(), sizes=(2, 1, 1))
    assert [row["idx"] for row in partitions["audit"]] == [4]
    with pytest.raises(ValueError, match="Insufficient unused"):
        partition_rows(rows, {1, 2}, sizes=(2, 1, 1))


def test_reserved_audit_is_not_measured_without_a_frozen_choice(tmp_path, monkeypatch):
    from etdl import followup
    from etdl.config import Config
    from etdl.io import sha256, write_json

    config = Config(artifact_dir=tmp_path, model_id="test")
    registry_path, workload, reference = [
        tmp_path / f"{name}.json" for name in ("registry", "workload", "reference")
    ]
    write_json(registry_path, {})
    write_json(workload, {"split": "train", "role": "audit", "rows": []})
    write_json(reference, {"workload_sha256": sha256(workload), "model_source": {}})
    request = {
        "registry": str(registry_path),
        "registry_sha256": sha256(registry_path),
        "workload": str(workload),
        "workload_sha256": sha256(workload),
        "references": str(reference),
        "references_sha256": sha256(reference),
        "role": "audit",
    }
    request_path = tmp_path / "request.json"
    write_json(request_path, request)
    monkeypatch.setattr(
        followup, "load_registry", lambda *args: {"hardware_key": {}, "model_source": {}}
    )
    monkeypatch.setattr(followup, "environment", dict)
    monkeypatch.setattr(followup, "hardware_key", lambda _: {})
    with pytest.raises(ValueError, match="requires a frozen"):
        followup.benchmark(config, request_path, tmp_path / "result.json")


def test_followup_selection_respects_repetitions_memory_and_score_checks():
    from etdl.followup import pick_candidate

    def group(family, p95, memory=650, passed=True, runs=3):
        return {
            "family": family,
            "method": "dynamic",
            "variants": [],
            "runs": runs,
            "passed": passed,
            "changed_predictions": 0,
            "max_peak_bytes": memory,
            "median_run_p95_ms": p95,
        }

    standard = group("standard", 10)
    assert pick_candidate([standard, group("transformer", 8)], 1000)["family"] == "transformer"
    assert pick_candidate([standard, group("transformer", 8, memory=1200)], 1000) == standard
    assert pick_candidate([standard, group("transformer", 8, passed=False)], 1000) == standard
    assert pick_candidate([standard, group("transformer", 8, runs=1)], 1000) == standard
    assert pick_candidate([group("transformer", 10), standard], 1000) == standard


@pytest.mark.parametrize(
    "role,stage", [("confirmation", "fit_policies_frozen"), ("audit", "audit_choice_frozen")]
)
def test_frozen_followup_condition_cannot_change_model_set(tmp_path, monkeypatch, role, stage):
    from etdl import followup
    from etdl.config import Config
    from etdl.io import sha256, write_json

    config = Config(artifact_dir=tmp_path, model_id="test")
    registry, workload, reference = [
        tmp_path / f"{name}.json" for name in ("registry", "workload", "reference")
    ]
    write_json(registry, {})
    write_json(workload, {"split": "train", "role": role, "rows": []})
    write_json(reference, {"workload_sha256": sha256(workload), "model_source": {}})
    condition = {
        "models": ["dynamic"],
        "routing": {},
        "threads": 4,
        "registry_sha256": sha256(registry),
    }
    freeze = tmp_path / "freeze.json"
    write_json(
        freeze,
        {
            "stage": stage,
            "workload_hashes": {role: sha256(workload)},
            "conditions": {"condition": condition},
        },
    )
    request = {
        **condition,
        "models": ["dynamic", "fixed-16"],
        "registry": str(registry),
        "workload": str(workload),
        "workload_sha256": sha256(workload),
        "references": str(reference),
        "references_sha256": sha256(reference),
        "role": role,
        "freeze": str(freeze),
        "freeze_sha256": sha256(freeze),
        "condition_id": "condition",
    }
    request_path = tmp_path / "request.json"
    write_json(request_path, request)
    monkeypatch.setattr(
        followup, "load_registry", lambda *args: {"hardware_key": {}, "model_source": {}}
    )
    monkeypatch.setattr(followup, "environment", dict)
    monkeypatch.setattr(followup, "hardware_key", lambda _: {})
    with pytest.raises(ValueError, match="condition changed"):
        followup.benchmark(config, request_path, tmp_path / "result.json")
