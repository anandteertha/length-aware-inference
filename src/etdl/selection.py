"""Training-only routing and selection; empty specialization is an explicit candidate."""

import math
from collections import defaultdict
from itertools import combinations

import numpy as np


def subsets(names):
    names = sorted(names)
    return [
        list(group)
        for count in range(min(2, len(names)) + 1)
        for group in combinations(names, count)
    ]


def route_name(length, routing, loaded):
    name = routing.get(str(length), "dynamic")
    if name == "dynamic" or name not in loaded:
        return "dynamic"
    if not name.startswith("fixed-") or length > int(name.split("-")[1]):
        return "dynamic"
    return name


def learn_routes(profiles, variants):
    timings = {}
    for name in ["dynamic", *variants]:
        groups = defaultdict(list)
        for sample in profiles[name]["samples"]:
            groups[sample["tokens"]].append(sample["end_to_end_ms"])
        timings[name] = {length: float(np.median(values)) for length, values in groups.items()}
    routing = {}
    for length in timings["dynamic"]:
        eligible = ["dynamic"] + [
            name
            for name in variants
            if length <= int(name.split("-")[1]) and length in timings[name]
        ]
        # Dynamic wins exact ties and remains eligible even if a specialized graph fits.
        routing[str(length)] = min(eligible, key=lambda name: timings[name][length])
    return routing


def select_configuration(measurements, budget_bytes):
    if not math.isfinite(budget_bytes) or budget_bytes <= 0:
        raise ValueError("Memory budget must be finite and positive")
    eligible = [
        item
        for item in measurements
        if len(item["variants"]) <= 2
        and item["memory_for_budget_bytes"] <= budget_bytes
        and item["changed_predictions"] == 0
        and math.isfinite(item["latency"]["end_to_end_ms"]["p95_ms"])
        and item["latency"]["end_to_end_ms"]["p95_ms"] >= 0
    ]
    if not eligible:
        raise ValueError("No verified configuration meets the measured process-memory budget")
    return min(
        eligible,
        key=lambda item: (
            item["latency"]["end_to_end_ms"]["p95_ms"],
            item["memory_for_budget_bytes"],
            len(item["variants"]),
        ),
    )
