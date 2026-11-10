"""Archive the fresh per-thread/transformer follow-up without model binaries or text."""

import argparse
import csv
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from etdl.io import read_json, sha256, write_json


def label(group):
    variants = "+".join(name.removeprefix("fixed-") for name in group["variants"])
    return variants or "dynamic only"


def render(output, confirmation, audit, settings):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = [
        family
        for family in ("standard", "transformer")
        if any(group["family"] == family for group in confirmation)
    ]
    threads = sorted({group["threads"] for group in confirmation})
    fig, axes = plt.subplots(1, len(threads), figsize=(14, 5), sharey=True, squeeze=False)
    conditions = [(family, method) for family in families for method in ("dynamic", "selected")]
    names = [f"{family}\n{method}" for family, method in conditions]
    maximum = max(value for group in confirmation for value in group["run_p95_ms"])
    colors = {"standard": "#2563eb", "transformer": "#059669"}
    for thread, ax in zip(threads, axes[0], strict=True):
        for index, (family, method) in enumerate(conditions):
            group = next(
                (
                    item
                    for item in confirmation
                    if item["family"] == family
                    and item["threads"] == thread
                    and item["method"] == method
                ),
                None,
            )
            if group is None:
                continue
            values = group["run_p95_ms"]
            ax.scatter(
                index + np.linspace(-0.10, 0.10, len(values)),
                values,
                marker="o" if method == "dynamic" else "^",
                color=colors[family],
                s=40,
            )
            median = float(np.median(values))
            ax.plot(
                [index - 0.20, index + 0.20], [median, median], color=colors[family], linewidth=2
            )
            ax.annotate(
                f"med {median:.1f}",
                (index, max(values)),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        ax.set_title(f"{thread} intra-op threads")
        ax.set_xticks(range(len(conditions)), names, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, maximum * 1.18)
        ax.grid(axis="y", alpha=0.15)
    axes[0, 0].set_ylabel("Process-run combined p95 (ms)")
    fig.suptitle("Fresh confirmation: separate policies fitted at each thread count")
    fig.text(
        0.5,
        0.02,
        f"{settings['sizes'][1]} training-derived confirmation rows; dots are process runs, bars are medians. No confidence intervals.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(output / "confirmation.png", dpi=180)
    fig.savefig(output / "confirmation.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    names = [f"{item['family']}\n{item['method']}\n({label(item)})" for item in audit]
    for ax, field, ylabel, divisor in (
        (axes[0], "median_run_p95_ms", "Median process-run p95 (ms)", 1),
        (axes[1], "max_peak_bytes", "Maximum process peak RSS (MiB)", 2**20),
    ):
        values = [item[field] / divisor for item in audit]
        ax.bar(range(len(audit)), values, color=[colors[item["family"]] for item in audit])
        for index, value in enumerate(values):
            ax.annotate(
                f"{value:.1f}",
                (index, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        ax.set_xticks(range(len(audit)), names, rotation=18, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ceiling = (
            max(max(values), settings["budget_mib"]) if field == "max_peak_bytes" else max(values)
        )
        ax.set_ylim(0, ceiling * 1.22)
        ax.grid(axis="y", alpha=0.15)
    axes[1].axhline(
        settings["budget_mib"],
        color="#111827",
        linestyle="--",
        linewidth=1,
        label=f"Cap: {settings['budget_mib']} MiB",
    )
    axes[1].legend(fontsize=8)
    fig.suptitle(
        f"Reserved optimization audit: {audit[0]['threads']} threads, {settings['blocks']} process blocks"
    )
    fig.text(
        0.5,
        0.02,
        f"{settings['sizes'][2]} training-derived rows held out from policy tuning, not upstream classifier training. No new GLUE validation claim.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(output / "audit.png", dpi=180)
    fig.savefig(output / "audit.svg")
    plt.close(fig)


def diagnostics(config_path, directory, output):
    from etdl.config import load_config
    from etdl.data import encode, load_tokenizer
    from etdl.followup import load_registry
    from etdl.runtime import session
    from etdl.variants import pad_inputs

    config = load_config(config_path)
    frozen = read_json(directory / "audit-plan.json")
    row = min(read_json(directory / "audit.json")["rows"], key=lambda item: item["tokens"])
    if row["tokens"] > 16:
        raise ValueError("No common audit example fits all graph capacities")
    tokenizer = load_tokenizer(config)
    inputs = encode(tokenizer, row["text"], config.max_length)
    results = {}
    for path in directory.glob("registry-*.json"):
        registry = load_registry(config, path)
        family = registry["family"]
        if not registry["available"]:
            continue
        trace_dir = output / "traces" / family
        trace_dir.mkdir(parents=True)
        results[family] = {}
        for name, graph in registry["graphs"].items():
            if not graph["passed"]:
                continue
            feed = (
                inputs
                if name == "dynamic"
                else pad_inputs(inputs, int(name.split("-")[1]), tokenizer.pad_token_id)
            )
            runtime, _ = session(
                Path(graph["path"]),
                frozen["chosen"]["threads"],
                profile_prefix=trace_dir / name,
                allow_spinning=False,
            )
            for _ in range(3):
                runtime.run(["logits"], feed)
            trace = Path(runtime.end_profiling())
            nodes = [event for event in read_json(trace) if event.get("cat") == "Node"]
            providers = sorted(
                {
                    event["args"]["provider"]
                    for event in nodes
                    if event.get("args", {}).get("provider")
                }
            )
            if providers != ["CPUExecutionProvider"]:
                raise ValueError("Diagnostic execution did not stay on CPU")
            results[family][name] = {
                "trace": trace.relative_to(output).as_posix(),
                "model_sha256": graph["sha256"],
                "input_shape": list(feed["input_ids"].shape),
                "providers": providers,
                "operator_calls": dict(
                    sorted(
                        Counter(
                            event.get("args", {}).get("op_name", "unknown") for event in nodes
                        ).items()
                    )
                ),
            }
            del runtime
    write_json(
        output / "operator-traces.json",
        {
            "purpose": "Post-audit diagnostic traces, excluded from every timing measurement.",
            "common_training_idx": row["idx"],
            "tokens": row["tokens"],
            "requests_per_graph": 3,
            "threads": frozen["chosen"]["threads"],
            "graphs": results,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--followup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    parser.add_argument("--traces", action="store_true")
    arguments = parser.parse_args()
    directory, output = arguments.followup.resolve(), arguments.output.resolve()
    summary = read_json(directory / "summary.json")
    if summary["status"] != "complete":
        raise ValueError("Follow-up must complete before archiving")
    if output.exists():
        raise ValueError("Archive exists; choose a new output directory")
    experiment = read_json(directory / "experiment.json")
    partitions = read_json(directory / "partitions.json")
    if summary["audit_plan_sha256"] != sha256(directory / "audit-plan.json") or summary[
        "fit_policies_sha256"
    ] != sha256(directory / "fit-policies.json"):
        raise ValueError("Frozen follow-up provenance changed")
    frozen = read_json(directory / "audit-plan.json")
    if (
        frozen["experiment_sha256"] != sha256(directory / "experiment.json")
        or frozen["confirmation_plan_sha256"] != sha256(directory / "confirmation-plan.json")
        or frozen["confirmation_summary_sha256"] != sha256(directory / "confirmation-summary.json")
    ):
        raise ValueError("Follow-up confirmation provenance changed")
    for role, digest in experiment["workload_hashes"].items():
        if sha256(directory / f"{role}.json") != digest:
            raise ValueError("Follow-up data changed")
    output.mkdir(parents=True)
    for name in (
        "experiment.json",
        "partitions.json",
        "fit-policies.json",
        "confirmation-plan.json",
        "confirmation-summary.json",
        "audit-plan.json",
        "summary.json",
    ):
        shutil.copyfile(directory / name, output / name)
    registries = {}
    for path in directory.glob("registry-*.json"):
        family = path.stem.removeprefix("registry-")
        registries[family] = read_json(path)
        if registries[family]["available"] and sha256(path) != experiment["registry_hashes"].get(
            family
        ):
            raise ValueError("Follow-up graph registry changed")
        shutil.copyfile(path, output / path.name)
    for path in directory.glob("models/transformer/checks-*.json"):
        shutil.copyfile(path, output / path.name)
    sets = []
    for role in ("fit", "confirmation", "audit"):
        workload = read_json(directory / f"{role}.json")
        ids = {row["idx"] for row in workload["rows"]}
        if any(ids & group for group in sets) or ids & set(partitions["excluded_training_ids"]):
            raise ValueError("Follow-up data partitions overlap")
        sets.append(ids)
        write_json(
            output / f"{role}-metadata.json",
            {key: value for key, value in workload.items() if key != "rows"}
            | {
                "original_workload_sha256": sha256(directory / f"{role}.json"),
                "rows": [
                    {key: value for key, value in row.items() if key != "text"}
                    for row in workload["rows"]
                ],
            },
        )
        shutil.copyfile(directory / f"references-{role}.json", output / f"references-{role}.json")
    metrics, requests, environment = [], [], None
    archived_requests = output / "measurement-requests"
    archived_requests.mkdir()
    for path in sorted((directory / "measurements").glob("*.json")):
        if path.name.endswith(".request.json"):
            continue
        result = read_json(path)
        environment = environment or result["environment"]
        request_path = path.with_name(path.stem + ".request.json")
        if result["request_sha256"] != sha256(request_path):
            raise ValueError("A measurement request changed")
        request = read_json(request_path)
        if (
            result["workload_sha256"] != experiment["workload_hashes"][result["role"]]
            or result["registry_sha256"] != experiment["registry_hashes"][result["family"]]
            or result["references_sha256"]
            != sha256(directory / f"references-{result['role']}.json")
        ):
            raise ValueError("A measurement belongs to different experiment inputs")
        if result["role"] in {"confirmation", "audit"} and request["freeze_sha256"] != sha256(
            directory / f"{result['role']}-plan.json"
        ):
            raise ValueError("A measurement belongs to a different frozen plan")
        shutil.copyfile(request_path, archived_requests / request_path.name)
        metrics.append(
            {key: value for key, value in result.items() if key not in {"samples", "environment"}}
            | {"measurement": path.stem}
        )
        requests.extend(
            {
                "measurement": path.stem,
                "family": result["family"],
                "threads": result["threads"],
                "role": result["role"],
                "kind": result["kind"],
                "method": result["method"],
                "block": result["block"],
                **sample,
            }
            for sample in result["samples"]
        )
    with (output / "requests.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(requests[0]))
        writer.writeheader()
        writer.writerows(requests)
    write_json(output / "process-metrics.json", metrics)
    write_json(output / "environment.json", environment)
    winner = summary["chosen_before_audit"]
    winner_audit = next(
        item
        for item in summary["audit_groups"]
        if all(item[key] == winner[key] for key in ("family", "threads", "method"))
    )
    budget = experiment["settings"]["budget_mib"] * 2**20
    write_json(
        output / "archive-metadata.json",
        {
            "measurement_code_commit": experiment["measurement_code_commit"],
            "reporting_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "processes": len(metrics),
            "timed_requests": len(requests),
            "process_counts_by_role": dict(Counter(item["role"] for item in metrics)),
            "chosen_audit_memory_within_budget": winner_audit["max_peak_bytes"] <= budget,
            "chosen_audit_score_checks_passed": winner_audit["passed"],
            "note": "Audit rows are from upstream classifier training data, held out only from inference-policy tuning.",
        },
    )
    render(output, summary["confirmation_groups"], summary["audit_groups"], experiment["settings"])
    if arguments.traces:
        diagnostics(arguments.config, directory, output)
    write_json(
        output / "checksums.json",
        {
            path.relative_to(output).as_posix(): sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "checksums.json"
        },
    )
    print(f"Archived exploratory follow-up to {output}")


if __name__ == "__main__":
    main()
