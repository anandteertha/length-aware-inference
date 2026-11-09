import pytest

from etdl.ablations import workload_quantile


def test_uniform_lengths_change_mixture_instead_of_weighting_frequent_short_inputs():
    samples = [{"tokens": 8, "end_to_end_ms": 1}] * 98 + [{"tokens": 128, "end_to_end_ms": 100}] * 2
    assert workload_quantile(samples) == 1
    assert workload_quantile(samples, uniform_lengths=True) == 100


@pytest.mark.parametrize(
    "samples",
    [[], [{"tokens": 4, "end_to_end_ms": float("nan")}], [{"tokens": 4, "end_to_end_ms": -1}]],
)
def test_invalid_ablation_latencies_are_not_silently_omitted(samples):
    with pytest.raises(ValueError):
        workload_quantile(samples)
