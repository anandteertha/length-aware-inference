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
    variants = commands.add_parser("variants", help="Build and check fixed-length ONNX variants")
    variants.add_argument("--lengths", type=positive, nargs="+", default=[16, 32, 64, 128])
    variants.add_argument("--limit", type=positive, default=128)
    study = commands.add_parser("study", help="Profile checked variants and select at most two")
    study.add_argument("--limit", type=positive, default=128)
    study.add_argument("--threads", type=positive, default=1)
    study.add_argument("--repeats", type=positive, default=3)
    study.add_argument("--warmup", type=nonnegative, default=10)
    study.add_argument("--memory-budget-mib", type=positive)
    study.add_argument("--name", help="New study directory name; preserves earlier runs")
    predict = commands.add_parser("predict", help="Classify text using a frozen measured policy")
    predict.add_argument("text")
    predict.add_argument("--policy", type=Path, default=None)
    comparison = commands.add_parser(
        "compare", help="Randomized independent-process training confirmation"
    )
    comparison.add_argument("--policy", type=Path)
    comparison.add_argument("--name", default="confirmation")
    comparison.add_argument("--limit", type=positive, default=128)
    comparison.add_argument("--threads", type=positive, nargs="+", default=[1, 2, 4])
    comparison.add_argument("--blocks", type=positive, default=3)
    comparison.add_argument("--warmup", type=nonnegative, default=10)
    comparison.add_argument("--resume", action="store_true")
    freeze = commands.add_parser(
        "freeze", help="Freeze deployment and evaluation before validation access"
    )
    freeze.add_argument("--comparison", default="confirmation")
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate a frozen plan on all 872 held-out examples"
    )
    evaluate.add_argument("--comparison", default="confirmation")
    evaluate.add_argument("--resume", action="store_true")
    ablate = commands.add_parser(
        "ablate", help="Reanalyse training-only memory, output-check and frequency sensitivity"
    )
    ablate.add_argument("--study", default="study-t1")
    ablate.add_argument("--output", type=Path)
    followup = commands.add_parser(
        "followup", help="Refit policies per thread and compare transformer-optimized graphs"
    )
    followup.add_argument("--name", default="followup")
    followup.add_argument("--threads", type=positive, nargs="+", default=[1, 2, 4])
    followup.add_argument("--blocks", type=positive, default=3)
    followup.add_argument("--fit-size", type=positive, default=256)
    followup.add_argument("--confirmation-size", type=positive, default=128)
    followup.add_argument("--audit-size", type=positive, default=128)
    followup.add_argument("--resume", action="store_true")
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
    if arguments.command == "followup":
        from etdl.followup import execute

        if arguments.name in {".", ".."} or any(value in arguments.name for value in "/\\:"):
            raise ValueError("Follow-up name must be a single directory name")
        report = execute(
            config,
            arguments.config,
            config.artifact_dir / arguments.name,
            resume=arguments.resume,
            threads=tuple(arguments.threads),
            blocks=arguments.blocks,
            sizes=(arguments.fit_size, arguments.confirmation_size, arguments.audit_size),
        )
        print(json.dumps(report, indent=2))
        return
    if arguments.command == "ablate":
        from etdl.ablations import reanalyse

        directory = (config.artifact_dir / arguments.study).resolve()
        if directory.parent != config.artifact_dir.resolve():
            raise ValueError("Study must be directly inside artifact_dir")
        output = arguments.output or config.artifact_dir / f"reports/ablation-{directory.name}.json"
        print(json.dumps(reanalyse(directory / "summary.json", output), indent=2))
        return
    if arguments.command in {"freeze", "evaluate"}:
        from etdl.comparison import evaluate, freeze

        directory = (config.artifact_dir / arguments.comparison).resolve()
        if directory.parent != config.artifact_dir.resolve():
            raise ValueError("Comparison must be a directory directly inside artifact_dir")
        report = (
            freeze(config, directory)
            if arguments.command == "freeze"
            else evaluate(config, arguments.config, directory, arguments.resume)
        )
        print(json.dumps(report, indent=2))
        return
    if arguments.command == "compare":
        from etdl.comparison import compare

        policy = arguments.policy or config.artifact_dir / "study-t1/selection.json"
        report = compare(
            config,
            arguments.config,
            policy,
            arguments.name,
            arguments.limit,
            arguments.threads,
            arguments.blocks,
            arguments.warmup,
            arguments.resume,
        )
        print(json.dumps(report, indent=2))
        return
    if arguments.command == "variants":
        for length in arguments.lengths:
            print(f"Building and checking fixed-{length}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "etdl.worker",
                    "--config",
                    str(arguments.config.resolve()),
                    "variant",
                    "--length",
                    str(length),
                    "--limit",
                    str(arguments.limit),
                ],
                check=True,
            )
        return
    if arguments.command == "study":
        from etdl.study import study

        report = study(
            config,
            arguments.config,
            arguments.limit,
            arguments.threads,
            arguments.repeats,
            arguments.warmup,
            arguments.memory_budget_mib,
            arguments.name,
        )
        print(json.dumps(report, indent=2))
        return
    if arguments.command == "predict":
        from etdl.router import RoutedClassifier

        preferred = config.artifact_dir / "confirmation/frozen-policy.json"
        policy = arguments.policy or (
            preferred if preferred.exists() else config.artifact_dir / "study-t1/selection.json"
        )
        classifier = RoutedClassifier(config, policy)
        print(
            json.dumps(
                {**classifier.predict(arguments.text), "ready_ms": classifier.ready_ms}, indent=2
            )
        )
        return
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
