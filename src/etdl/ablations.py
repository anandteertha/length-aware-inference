"""Training-only sensitivity reanalysis; never modifies a frozen deployment policy."""

from collections import Counter
from copy import deepcopy

import numpy as np

from etdl.io import read_json, sha256, write_json
from etdl.selection import select_configuration


def workload_quantile(samples, uniform_lengths=False, quantile=0.95):
    if not samples or not 0 < quantile <= 1:
        raise ValueError("A nonempty workload and valid quantile are required")
    values = np.array([sample["end_to_end_ms"] for sample in samples], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid latency samples")
    counts = Counter(sample["tokens"] for sample in samples)
    weights = np.array(
        [1 / counts[sample["tokens"]] if uniform_lengths else 1 for sample in samples]
    )
    order = np.argsort(values, kind="stable")
    index = np.searchsorted(np.cumsum(weights[order]), quantile * weights.sum(), side="left")
    return float(values[order[min(index, len(values) - 1)]])


def reanalyse(study_path, output):
    study = read_json(study_path)
    if study.get("split") != "train":
        raise ValueError("Ablation selection must use training measurements only")
    measurements = study["measurements"]
    budget = study["memory_budget_bytes"]

    def describe(item):
        return {
            "variants": item["variants"],
            "p95_ms": item["latency"]["end_to_end_ms"]["p95_ms"],
            "peak_bytes": item["memory_for_budget_bytes"],
            "changed_predictions": item["changed_predictions"],
        }

    original = select_configuration(measurements, budget)
    uncapped = select_configuration(
        measurements, float(max(item["memory_for_budget_bytes"] for item in measurements) + 1)
    )
    ignored_checks = deepcopy(measurements)
    for item in ignored_checks:
        item["changed_predictions"] = 0
    unchecked = select_configuration(ignored_checks, budget)
    frequency = []
    for uniform in (False, True):
        candidates = deepcopy(measurements)
        for item in candidates:
            item["latency"]["end_to_end_ms"]["p95_ms"] = workload_quantile(item["samples"], uniform)
        frequency.append(
            {
                "mixture": "uniform_observed_lengths" if uniform else "empirical_lengths",
                "selected": describe(select_configuration(candidates, budget)),
                "configuration_objectives": [describe(item) for item in candidates],
            }
        )
    cap_sensitivity = []
    for mib in (512, 700, 1024, 1536):
        try:
            result = describe(select_configuration(measurements, mib * 2**20))
        except ValueError:
            result = None
        cap_sensitivity.append({"budget_mib": mib, "selection": result})
    result = {
        "split": "train",
        "source_study_sha256": sha256(study_path),
        "baseline": describe(original),
        "without_memory_cap": describe(uncapped),
        "without_changed_label_gate": describe(unchecked),
        "all_measured_candidates_had_zero_changed_predictions": all(
            item["changed_predictions"] == 0 for item in measurements
        ),
        "memory_cap_sensitivity": cap_sensitivity,
        "frequency_sensitivity": frequency,
        "frequency_quantile_estimator": "Inverse weighted empirical CDF, without interpolation, for both mixtures",
        "limitations": [
            "Reanalysis of the recorded training timing matrix, not new execution measurements.",
            "All measured variants already passed checks; removing the gate cannot test a failing graph.",
            "Uniform-length weights change the objective, without replaying changed cache/switch behavior.",
            "The original selector used linear quantile interpolation; frequency sensitivity uses one common inverse-CDF estimator.",
            "No ablation changes the deployed policy or uses validation outcomes.",
        ],
    }
    if output.exists():
        raise ValueError("Ablation output exists; preserve it and choose a new path")
    write_json(output, result)
    return result
