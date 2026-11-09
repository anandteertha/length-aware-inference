"""Archive a compact, reviewable study snapshot without model binaries or input text."""

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

from etdl.io import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    study = arguments.study.resolve()
    output = arguments.output.resolve()
    summary = read_json(study / "summary.json")
    if summary["status"] != "selected":
        raise ValueError("Only a completed study can be archived")
    if output.exists():
        raise ValueError("Archive exists; choose a new output directory")
    calibration = study.parent / "data/calibration.json"
    if sha256(calibration) != summary["calibration_sha256"]:
        raise ValueError("Calibration data changed after the study")
    output.mkdir(parents=True)
    profiles = {
        path.stem.removeprefix("profile-"): read_json(path)
        for path in study.glob("profile-*.json")
        if not path.name.endswith(".request.json")
    }
    measurements = summary["measurements"]
    compact = {key: value for key, value in summary.items() if key != "measurements"}
    compact["measurements"] = [
        {key: value for key, value in item.items() if key not in {"environment", "samples"}}
        for item in measurements
    ]
    compact["measurement_code_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    compact["environment"] = measurements[0]["environment"]
    compact["calibration_source"] = {
        key: value for key, value in read_json(calibration).items() if key != "rows"
    }
    compact["studied_train_ids"] = sorted({sample["idx"] for sample in measurements[0]["samples"]})
    write_json(output / "summary.json", compact)
    write_json(output / "selection.json", summary["selection"])
    verification = {}
    for path in (study.parent / "reports").glob("checks*.json"):
        verification[path.stem] = {
            key: value
            for key, value in read_json(path).items()
            if key not in {"cases", "checked_train_ids"}
        }
    write_json(output / "verification.json", verification)
    write_json(
        output / "model-manifests.json",
        {path.stem: read_json(path) for path in (study.parent / "models").glob("*.json")},
    )
    for filename in ("configurations.csv", "latency-memory.png", "latency-memory.svg"):
        shutil.copyfile(study / filename, output / filename)
    if (study / "smoke-predictions.json").exists():
        shutil.copyfile(study / "smoke-predictions.json", output / "smoke-predictions.json")
    for filename, groups in (
        ("profile-requests.csv", list(profiles.items())),
        (
            "configuration-requests.csv",
            [("+".join(item["variants"]) or "dynamic", item) for item in measurements],
        ),
    ):
        with (output / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["method", *groups[0][1]["samples"][0]])
            writer.writeheader()
            for name, report in groups:
                writer.writerows({"method": name, **sample} for sample in report["samples"])
    write_json(
        output / "checksums.json",
        {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
    )
    print(f"Archived completed study to {output}")


if __name__ == "__main__":
    main()
