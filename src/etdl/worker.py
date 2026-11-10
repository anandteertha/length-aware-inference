"""Isolated study workers. Use the public `etdl variants` / `etdl study` commands."""

import argparse
import json
from pathlib import Path

from etdl.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    variant = commands.add_parser("variant")
    variant.add_argument("--length", type=int, required=True)
    variant.add_argument("--limit", type=int, required=True)
    run = commands.add_parser("run")
    run.add_argument("--request", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    reference = commands.add_parser("reference")
    reference.add_argument("--workload", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)
    comparison = commands.add_parser("comparison")
    comparison.add_argument("--request", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    followup = commands.add_parser("followup")
    followup.add_argument("--request", type=Path, required=True)
    followup.add_argument("--output", type=Path, required=True)
    boundary = commands.add_parser("boundary")
    boundary.add_argument("--request", type=Path, required=True)
    boundary.add_argument("--output", type=Path, required=True)
    boundary_ref = commands.add_parser("boundary-reference")
    boundary_ref.add_argument("--plan", type=Path, required=True)
    boundary_ref.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.command == "boundary-reference":
        from etdl.boundaries import reference

        reference(config, arguments.plan, arguments.output)
    elif arguments.command == "boundary":
        from etdl.boundaries import run_worker

        run_worker(config, arguments.request, arguments.output)
    elif arguments.command == "variant":
        from etdl.variants import build_variant, check_variant

        build_variant(config, arguments.length)
        result = check_variant(config, arguments.length, arguments.limit)
        print(json.dumps(result, indent=2))
    elif arguments.command == "reference":
        from etdl.comparison import references

        references(config, arguments.workload, arguments.output)
    elif arguments.command == "comparison":
        from etdl.comparison import measure

        measure(config, arguments.request, arguments.output)
    elif arguments.command == "followup":
        from etdl.followup import benchmark

        benchmark(config, arguments.request, arguments.output)
    else:
        from etdl.study import run_worker

        result = run_worker(config, arguments.request, arguments.output)
        print(
            json.dumps(
                {
                    "output": str(arguments.output),
                    "p95_ms": result["latency"]["end_to_end_ms"]["p95_ms"],
                    "peak_mib": result["memory_for_budget_bytes"] / 2**20,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
