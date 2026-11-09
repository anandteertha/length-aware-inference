from copy import deepcopy

import pytest

from etdl.selection import learn_routes, route_name, select_configuration, subsets


def profile(values):
    return {
        "samples": [
            {"tokens": length, "end_to_end_ms": milliseconds} for length, milliseconds in values
        ]
    }


def test_dynamic_can_win_even_when_a_variant_fits():
    profiles = {
        "dynamic": profile([(8, 2), (8, 4), (18, 10)]),
        "fixed-16": profile([(8, 8)]),
        "fixed-32": profile([(8, 4), (18, 5)]),
    }
    assert learn_routes(profiles, ["fixed-16", "fixed-32"]) == {"8": "dynamic", "18": "fixed-32"}


def test_padding_cost_can_make_larger_variant_slower():
    profiles = {"dynamic": profile([(18, 3)]), "fixed-32": profile([(18, 6)])}
    assert learn_routes(profiles, ["fixed-32"])["18"] == "dynamic"


@pytest.mark.parametrize(
    "length,routing,loaded",
    [
        (140, {"140": "fixed-128"}, {"dynamic", "fixed-128"}),
        (18, {"18": "fixed-32"}, {"dynamic"}),
        (31, {"18": "fixed-32"}, {"dynamic", "fixed-32"}),
    ],
)
def test_unseen_overflow_or_unloaded_route_falls_back(length, routing, loaded):
    assert route_name(length, routing, loaded) == "dynamic"


def test_subsets_include_fallback_and_never_exceed_two_variants():
    options = subsets(["fixed-16", "fixed-32", "fixed-64", "fixed-128"])
    assert len(options) == 11 and [] in options
    assert max(map(len, options)) == 2


def measurement(variants, memory, p95, changed=0):
    return {
        "variants": variants,
        "memory_for_budget_bytes": memory,
        "latency": {"end_to_end_ms": {"p95_ms": p95}},
        "changed_predictions": changed,
    }


def test_selector_respects_measured_resident_memory_and_changed_predictions():
    values = [
        measurement([], 600, 12),
        measurement(["fixed-16"], 950, 8),
        measurement(["fixed-16", "fixed-32"], 1200, 4),
        measurement(["fixed-64"], 900, 3, changed=1),
    ]
    assert select_configuration(values, 1000)["variants"] == ["fixed-16"]
    assert select_configuration(values, 700)["variants"] == []
    with pytest.raises(ValueError, match="No verified configuration"):
        select_configuration(values, 500)


def test_selector_optimizes_p95_instead_of_mean_or_variant_count():
    values = [measurement([], 600, 5), measurement(["fixed-16"], 900, 6)]
    assert select_configuration(values, 1200)["variants"] == []
    invalid = deepcopy(values[0])
    invalid["latency"]["end_to_end_ms"]["p95_ms"] = float("nan")
    assert select_configuration([invalid, values[1]], 1200) == values[1]
