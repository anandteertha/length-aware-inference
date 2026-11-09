import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from etdl.config import load_config


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def parser():
    result = argparse.ArgumentParser(description="DistilBERT CPU specialization research pipeline")
    result.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Record CPU capabilities, memory, packages, and providers")
    commands.add_parser(
        "export", help="Export and optimize a batch-one, dynamic-length FP32 ONNX model"
    )
    for name in ("prepare", "check", "benchmark", "milestone"):
        command = commands.add_parser(name)
        command.add_argument("--limit", type=positive, default=128)
        if name in ("benchmark", "milestone"):
            command.add_argument(
                "--threads",
                type=positive,
                nargs="+" if name == "milestone" else None,
                default=[1] if name == "milestone" else 1,
            )
            command.add_argument("--repeats", type=positive, default=3)
            command.add_argument("--warmup", type=nonnegative, default=10)
    return result


def main():
    arguments = parser().parse_args()
    config = load_config(arguments.config)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    if arguments.command == "milestone":
        # Child processes isolate export/check's PyTorch allocations from ORT memory measurements.
        prefix = [sys.executable, "-m", "etdl", "--config", str(arguments.config.resolve())]
        steps = [
            ["doctor"],
            ["prepare", "--limit", str(arguments.limit)],
            ["export"],
            ["check", "--limit", str(arguments.limit)],
        ]
        threads = list(arguments.threads)
        random.Random(config.seed).shuffle(threads)
        steps.extend(
            [
                [
                    "benchmark",
                    "--limit",
                    str(arguments.limit),
                    "--threads",
                    str(thread),
                    "--repeats",
                    str(arguments.repeats),
                    "--warmup",
                    str(arguments.warmup),
                ]
                for thread in threads
            ]
        )
        for step in steps:
            print(f"Running {' '.join(step)}", flush=True)
            subprocess.run(prefix + step, check=True)
        return
    if arguments.command == "doctor":
        from etdl.io import write_json
        from etdl.runtime import environment

        report = environment()
        write_json(config.artifact_dir / "reports/environment.json", report)
    elif arguments.command == "prepare":
        from etdl.data import prepare

        report = prepare(config, arguments.limit)
    elif arguments.command == "export":
        from etdl.model import export

        report = export(config)
    elif arguments.command == "check":
        from etdl.checks import check

        report = check(config, arguments.limit)
    else:
        from etdl.benchmark import benchmark

        report = benchmark(
            config, arguments.limit, arguments.threads, arguments.repeats, arguments.warmup
        )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
