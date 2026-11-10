from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

spec = spec_from_file_location(
    "archive_boundaries", Path(__file__).resolve().parents[1] / "tools/archive_boundaries.py"
)
archive = module_from_spec(spec)
spec.loader.exec_module(archive)
telemetry_gap = archive.telemetry_gap


def test_normal_cpu_observations_do_not_trigger_gap_threshold():
    run = {"cpu_observations": [{"elapsed_s": value} for value in (1.1, 2.1, 3.2)]}
    assert telemetry_gap(run) < 10


def test_long_interruption_inside_process_is_detected():
    run = {"cpu_observations": [{"elapsed_s": value} for value in (1, 2, 21600, 21601)]}
    assert telemetry_gap(run) > 21000


def test_delayed_first_sample_and_missing_telemetry_are_not_silently_accepted():
    assert telemetry_gap({"cpu_observations": [{"elapsed_s": 15}]}) > 10
    assert telemetry_gap({"cpu_observations": []}) is None
