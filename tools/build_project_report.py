"""Build the consolidated local project report from checksum-verified evidence."""

import argparse
import hashlib
import shutil
from datetime import UTC, datetime
from pathlib import Path

from etdl.io import read_json, sha256, write_json

LABELS = {
    "pytorch": "PyTorch FP32",
    "dynamic": "Dynamic ONNX",
    "single-16": "Fixed-16 + fallback",
    "buckets": "Four fixed buckets + fallback",
    "selected": "Prior 16+32 policy + fallback",
}


def verify_archive(directory):
    for name, digest in read_json(directory / "checksums.json").items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Archived evidence changed: {directory / name}")


def render_tradeoff(output, held):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for run in held["runs"]:
        memory = run["memory_for_budget_bytes"] / 2**20
        p95 = run["latency"]["end_to_end_ms"]["p95_ms"]
        color = (
            "#dc2626" if memory > 1536 else "#059669" if run["method"] == "dynamic" else "#2563eb"
        )
        ax.scatter(
            memory, p95, s=85, color=color, marker="*" if run["method"] == "dynamic" else "o"
        )
        ax.annotate(
            LABELS[run["method"]],
            (memory, p95),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
        )
    ax.axvline(1536, color="#111827", linestyle="--", label="1536 MiB cap")
    ax.set_xlim(400, 2100)
    ax.set_ylim(0, 155)
    ax.set_xlabel("Peak process RSS (MiB)")
    ax.set_ylabel("Combined request p95 (ms)")
    ax.set_title("Original held-out validation: latency versus resident memory")
    ax.grid(alpha=0.12)
    ax.legend(loc="lower right")
    fig.text(
        0.5,
        0.02,
        "872 validation examples; 4 threads; one timing process per method. All methods: 794 correct, 0 changed predictions.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output / "validation-tradeoff.png", dpi=180)
    fig.savefig(output / "validation-tradeoff.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, default=Path("results/boundaries-1"))
    parser.add_argument("--interrupted", type=Path, default=Path("results/boundaries-interrupted"))
    parser.add_argument("--demo", type=Path, default=Path("artifacts/reports/project-demo.json"))
    parser.add_argument("--output", type=Path, default=Path("results/project-summary"))
    parser.add_argument("--report", type=Path, default=Path("docs/project-report.md"))
    args = parser.parse_args()
    for directory in (
        Path("results/milestone-3"),
        Path("results/followup-1"),
        args.diagnostic,
        args.interrupted,
    ):
        verify_archive(directory)
    held = read_json(Path("results/milestone-3/held-out-summary.json"))
    followup = read_json(Path("results/followup-1/summary.json"))
    diagnostic = read_json(args.diagnostic / "summary.json")
    plan = read_json(args.diagnostic / "plan.json")
    quality = read_json(args.diagnostic / "archive-metadata.json")
    interrupted = read_json(args.interrupted / "archive-metadata.json")
    replication_gaps = len(quality["long_telemetry_gap_processes"])
    demo = read_json(args.demo)
    policy = Path("results/milestone-3/frozen-policy.json")
    if (
        not held["all_correctness_checks_passed"]
        or not followup["all_audit_checks_passed"]
        or not diagnostic["all_checks_passed"]
        or not diagnostic["all_power_conditions_consistent"]
    ):
        raise ValueError("Report requires passing checks and stable recorded power conditions")
    if demo["policy_sha256"] != sha256(policy) or demo["retained_variants"] or demo["threads"] != 4:
        raise ValueError("Demo does not use the original frozen deployment policy")
    if args.output.exists() or args.report.exists():
        raise ValueError("Report or summary archive exists; choose new output paths")
    args.output.mkdir(parents=True)
    shutil.copyfile(args.demo, args.output / "demo.json")
    render_tradeoff(args.output, held)

    sources = {
        "E001": (
            "arrao6_project_proposal.pdf",
            "Pages 1-2: question, fixed variants, fallback, valid dynamic-only outcome",
        ),
        "E002": (
            "results/milestone-3/confirmation-summary.json",
            "groups: original training-confirmation process summaries",
        ),
        "E003": (
            "results/milestone-3/held-out-summary.json",
            "examples, threads, runs[*].accuracy/latency/memory/changed_predictions",
        ),
        "E004": (
            "results/milestone-3/frozen-policy.json",
            "threads, variants, model_source, hardware_key, runtime_settings",
        ),
        "E005": (
            "results/followup-1/summary.json",
            "chosen_before_audit, audit_groups, limitations",
        ),
        "E006": (
            (args.diagnostic / "summary.json").as_posix(),
            "groups, paired_ratios, checks, process/request counts",
        ),
        "E007": (
            (args.diagnostic / "plan.json").as_posix(),
            "cases, power, environment, blocks, threads, repeats, code hashes",
        ),
        "E008": (
            (args.output / "demo.json").as_posix(),
            "Functional predictions and frozen-policy digest",
        ),
        "E009": (
            "docs/followup-results.md",
            "Methods, natural token lengths and interpretation limits",
        ),
        "E010": (
            "docs/boundary-protocol.md",
            "Prespecified synthetic diagnostic and descriptive reporting",
        ),
        "E011": (
            "docs/final-results.md",
            "Original protocol, confirmation, frozen evaluation, runtime settings, and limitations",
        ),
        "E012": (
            (args.interrupted / "archive-metadata.json").as_posix(),
            "Original diagnostic counts, checks, and objective long-telemetry-gap flags",
        ),
        "E013": (
            "docs/boundary-amendment.md",
            "Timing-quality amendment before original summaries were inspected",
        ),
    }
    claims = []
    paragraphs = []

    def paragraph(text, evidence):
        identifier = f"C{len(claims) + 1:03d}"
        normalized = " ".join(text.split())
        claims.append(
            {
                "id": identifier,
                "text_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
                "evidence_ids": evidence,
                "human_verification": "pending",
            }
        )
        paragraphs.append(text + f" <!-- claim:{identifier}; evidence:{','.join(evidence)} -->\n")

    def literal(text):
        paragraphs.append(text + "\n")

    literal("# Matching CPU computation to message length\n\nConsolidated project report")
    paragraph(
        "The completed natural-workload experiments support dynamic-only ONNX execution at four threads on the measured Intel Core i5-1334U CPU. Keeping fixed-length variants did not deliver a consistent confirmed deployment benefit under the sampled workloads and memory budget. The follow-up optimizer showed a small, variable timing difference. These findings support a measured decision to retain no variants; they do not establish that specialization cannot help on other hardware or workloads.",
        ["E002", "E003", "E004", "E005"],
    )
    literal("## Question and implementation")
    paragraph(
        "The project asks whether keeping at most two checked 16/32/64/128-token DistilBERT variants alongside a dynamic fallback can reduce CPU p95 latency within a 1,536 MiB process-memory budget. Inputs are never truncated to fit a variant, and lengths without a verified route use the dynamic graph. Choosing no variants is an allowed outcome in the original proposal.",
        ["E001", "E004"],
    )
    paragraph(
        "The pipeline pins a pretrained SST-2 classifier, exports FP32 ONNX graphs, verifies scores and labels, profiles eligible routes, measures resident configurations, and freezes a policy before evaluation. The measured machine runs Windows 11 with ONNX Runtime 1.31.1. The original deployed policy uses the standard dynamic graph, four intra-op threads, sequential CPUExecutionProvider, one inter-op thread, and disabled idle spinning. The project performs no model training, quantization, or GPU execution.",
        ["E004", "E009", "E010"],
    )
    literal("## Original confirmation and held-out validation")
    paragraph(
        "Original training confirmation compared five methods at 1/2/4 threads in three randomized process blocks. It selected standard dynamic ONNX at four threads. The policy was frozen before the full 872-example GLUE SST-2 validation split was accessed; the previously fitted 16+32 policy remained a comparison after it was rejected for deployment.",
        ["E002", "E003", "E004", "E011"],
    )
    paragraph(
        "All five methods agreed on all 872 validation predictions: 794 correct, or 91.06% accuracy, with zero changed labels against the PyTorch reference. The following timings use one validation process per method and have no latency confidence intervals.",
        ["E003"],
    )
    literal("| Method | Median (ms) | p95 (ms) | Peak RSS (MiB) |\n| --- | ---: | ---: | ---: |")
    for method in ("pytorch", "dynamic", "single-16", "buckets", "selected"):
        run = next(r for r in held["runs"] if r["method"] == method)
        latency = run["latency"]["end_to_end_ms"]
        paragraph(
            f"| {LABELS[method]} | {latency['median_ms']:.3f} | {latency['p95_ms']:.3f} | {run['memory_for_budget_bytes'] / 2**20:,.1f} |",
            ["E003"],
        )
    literal(
        "![Original validation latency and memory](../results/project-summary/validation-tradeoff.png)"
    )
    literal(
        "| Method | Retained model files (MiB) | Model/session initialization (s) | Ready after probes and warm-up (s) |\n| --- | ---: | ---: | ---: |"
    )
    for method in ("pytorch", "dynamic", "single-16", "buckets", "selected"):
        run = next(r for r in held["runs"] if r["method"] == method)
        paragraph(
            f"| {LABELS[method]} | {run['model_size_bytes'] / 2**20:,.1f} | {sum(run['session_initialization_ms'].values()) / 1000:.3f} | {run['sessions_ready_ms'] / 1000:.3f} |",
            ["E003"],
        )
    paragraph(
        "Readiness includes capacity probes and warm-up but excludes process imports, tokenizer loading, and provenance checks. OS caches may be warm; these are not cold-device startup measurements. Serialized sizes count retained original model files, excluding tokenizer assets and extra optimized inspection copies. RSS includes process overhead and is the larger of sampled process memory and the Windows lifetime high-water mark.",
        ["E003", "E011"],
    )
    paragraph(
        "The four-bucket comparison exceeded the memory cap. The dynamic result is one timing observation on this workload, not a stable population estimate. Later experiment timings use different inputs and operating periods, so they must not be read as direct improvements over these validation numbers.",
        ["E003", "E005", "E009"],
    )
    literal("## Follow-up: refitting by thread count and transformer optimization")
    paragraph(
        "The follow-up allocated 256 fitting, 128 confirmation, and 128 audit rows from previously unused SST-2 training IDs. It refitted policies independently at each thread count and compared standard graphs with a separate transformer-optimized FP32 family. Across 93 timing processes and 24,324 requests, all output checks passed with zero changed predictions. The classifier was trained on SST-2 training data, so these new partitions are held out from inference-policy tuning only.",
        ["E005", "E009"],
    )
    paragraph(
        "Confirmation chose transformer-optimized dynamic ONNX at four threads, retaining no fixed variants. On the reserved optimization audit, its median process-run p95 was 73.560 ms versus 74.752 ms for standard dynamic ONNX, approximately 1.6% lower. Full ranges overlapped, and one of the three matched blocks reversed direction. This does not establish a reliable speedup. The original deployed policy was preserved.",
        ["E004", "E005"],
    )
    literal("Detailed methods and all comparisons: [follow-up report](followup-results.md).")
    literal("## Matched length-boundary diagnostic")
    quality_statement = (
        "The replication below had no gaps above the recorded ten-second telemetry threshold."
        if not replication_gaps
        else f"The replication also had {replication_gaps} process(es) with flagged telemetry gaps. Its timing tables therefore remain interrupted-run diagnostics and do not establish stable execution costs. No further repeat was selected from these outcomes."
    )
    paragraph(
        f"The first diagnostic had long telemetry interruptions, including a gap exceeding 21,000 seconds. A recorded timing-quality amendment required retaining that entire run and making one separate replication with the unchanged measurement code and design. The original archive contains {interrupted['processes']} processes and {interrupted['timed_requests']:,} requests; its output checks passed. Its complete results remain available separately. {quality_statement} The two runs are not pooled.",
        ["E006", "E007", "E012", "E013"],
    )
    literal(
        "Interrupted-run evidence: [complete results](../results/boundaries-interrupted/summary.json) and [recorded system load](../results/boundaries-interrupted/system-load.png)."
    )
    paragraph(
        f"The final diagnostic used {diagnostic['processes']} fresh processes in {plan['blocks']} randomized blocks at {plan['threads']} threads, with {diagnostic['timed_requests']:,} timed requests. Thirty-nine synthetic pretokenized inputs covered three token patterns and lengths 9, 15/16/17, 31/32/33, 63/64/65, and 127/128/129. Every input/path was checked against separately generated PyTorch FP32 references. These are execution probes without task labels or accuracy claims.",
        ["E006", "E007", "E010"],
    )
    paragraph(
        "For lengths up to 128, each process compared dynamic natural-length execution, dynamic execution padded to the smallest fitting bucket, and the matching fixed graph with identical padded token IDs and masks. Length 129 exercised dynamic fallback. Routing/padding and inference were timed separately; their sum excludes tokenization and is not text-to-label end-to-end latency. Both dynamic and fixed sessions were resident in each process. Lengths assigned to different capacities were measured in different fresh processes, so cross-length curves also contain process variation; the ratios compare matched paths within the same process.",
        ["E007", "E010"],
    )
    # Keep the complete per-length results rather than selecting favorable lengths.
    literal(
        "| Family | Length | Dynamic natural p95 (ms) | Dynamic padded p95 (ms) | Fixed padded p95 (ms) | Natural/fixed paired ratio |\n| --- | ---: | ---: | ---: | ---: | ---: |"
    )
    for family in ("standard", "transformer"):
        for length in sorted({g["length"] for g in diagnostic["groups"]}):
            by = {
                g["mode"]: g
                for g in diagnostic["groups"]
                if g["family"] == family and g["length"] == length
            }
            values = [
                f"{by[m]['latency']['prepared_input_ms']['median_run_p95_ms']:.3f}"
                if m in by
                else "N/A"
                for m in ("natural", "padded", "fixed")
            ]
            ratio = next(
                (
                    r
                    for r in diagnostic["paired_ratios"]
                    if r["family"] == family
                    and r["length"] == length
                    and r["contrast"] == "net_fixed_effect"
                ),
                None,
            )
            value = f"{ratio['median_ratio']:.3f}" if ratio else "N/A"
            paragraph(
                f"| {family.capitalize()} | {length} | {' | '.join(values)} | {value} |", ["E006"]
            )
    paragraph(
        "Table latencies are medians of process-run p95 values. Ratios are medians of within-process natural/fixed p95 ratios, not ratios of the table medians; values above one favor fixed execution. These descriptive comparisons do not retune a policy or establish a natural-workload aggregate speedup. Full ranges, individual block ratios, and every request are retained in the diagnostic archive.",
        ["E006", "E010"],
    )
    for family in ("standard", "transformer"):
        for contrast, description in (
            ("fixed_shape_effect", "fixed versus dynamic execution with identical padding"),
            ("net_fixed_effect", "fixed padded execution versus natural-length dynamic execution"),
        ):
            comparisons = [
                r
                for r in diagnostic["paired_ratios"]
                if r["family"] == family and r["contrast"] == contrast
            ]
            faster = sum(all(v["ratio"] > 1 for v in r["paired_blocks"]) for r in comparisons)
            slower = sum(all(v["ratio"] < 1 for v in r["paired_blocks"]) for r in comparisons)
            mixed = len(comparisons) - faster - slower
            paragraph(
                f"For the {family} family, considering {description}, fixed execution had lower p95 in every block at {faster} of {len(comparisons)} lengths, higher p95 in every block at {slower} lengths, and mixed or tied directions at {mixed} lengths. This summarizes the observed blocks only and is not a significance test or a guarantee for natural messages.",
                ["E006"],
            )
    literal(
        "![Latency by natural length](../results/boundaries-1/length-latency.png)\n\n![Matched padding and fixed-shape ratios](../results/boundaries-1/paired-ratios.png)"
    )
    paragraph(
        "Before/after AC connection and Windows power-scheme snapshots matched the recorded plan in every replication process. System and worker CPU load were recorded, and no replication process was excluded based on its performance. Snapshots cannot guarantee constant conditions between observations. System utilization includes the benchmark worker. Temperature, watts, clock speed, and affinity were not controlled; fixed two-second rests do not guarantee equal temperatures or repair earlier timing drift.",
        ["E006", "E007", "E010"],
    )
    literal("![Observed system CPU utilization](../results/boundaries-1/system-load.png)")
    literal("## Reproduction and functional demonstration")
    literal(
        '```powershell\nuv sync --locked\nuv run pytest -q\nuv run etdl predict "Great service!"\nuv run python tools/demo.py\nuv run etdl boundaries --name boundaries\n# Resume only the same frozen diagnostic inputs:\nuv run etdl boundaries --name boundaries --resume\n```'
    )
    paragraph(
        "The demonstration reuses the original frozen four-thread dynamic policy and retains no fixed variants. Its output records labels, logits, token counts, routes, and the policy digest. It is a functional demonstration, not a benchmark. The archived example uses two public sample sentences, and does not replace the complete validation evidence.",
        ["E004", "E008"],
    )
    literal(
        "The repository [README](../README.md) contains preparation, export, checking, selection, confirmation, and archive commands. Model binaries and downloaded datasets are regenerated locally rather than committed."
    )
    literal("## Limits, evidence, and conclusion")
    paragraph(
        "The main limitation is scope: one laptop CPU, few process blocks, timing variation, short natural follow-up messages, and synthetic diagnostic inputs. Original validation timing has only one process per method. Repeated requests within a process are dependent observations, and no formal significance tests or confidence intervals are claimed. There is no new classifier-generalization estimate from the training-derived audit or the synthetic probes.",
        ["E003", "E005", "E006", "E009", "E010"],
    )
    paragraph(
        "The supported deployment decision is to keep standard dynamic ONNX at four threads for the measured setup. A bounded selector is useful when it can reject extra variants after output, memory, and confirmation checks. Further experiments on other devices, longer natural workloads, or quantization would answer additional questions; none is necessary to change this project's observed conclusion.",
        ["E001", "E002", "E003", "E004", "E005", "E006"],
    )
    literal(
        "Detailed experiment records: [original final results](final-results.md), [follow-up](followup-results.md), and [diagnostic protocol](boundary-protocol.md). Numeric claims are mapped to local artifact hashes in [the evidence record](../results/project-summary/evidence.json)."
    )
    literal(
        "Codex assisted code implementation, experiment orchestration, analysis, and report drafting. This repository report is prepared for the project author's review; human verification and author approval are not asserted by the automated evidence audit."
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(paragraphs), encoding="utf-8", newline="\n")
    evidence = {
        "document_type": "Repository systems-benchmark project report",
        "reporting_guidance": "Study-specific recorded protocols; no clinical reporting guideline applies.",
        "report_sha256": sha256(args.report),
        "local_audit_by": "Codex",
        "local_audit_at_utc": datetime.now(UTC).isoformat(),
        "human_verification": "pending",
        "author_approval": "not asserted",
        "sources": {
            key: {
                "path": path,
                "sha256": sha256(Path(path)),
                "locator": locator,
                "human_verification": "pending",
            }
            for key, (path, locator) in sources.items()
        },
        "claims": claims,
    }
    write_json(args.output / "evidence.json", evidence)
    write_json(
        args.output / "checksums.json",
        {
            p.relative_to(args.output).as_posix(): sha256(p)
            for p in sorted(args.output.rglob("*"))
            if p.is_file() and p.name != "checksums.json"
        },
    )
    print(f"Built {args.report} and {args.output} with {len(claims)} evidence-bound claims.")


if __name__ == "__main__":
    main()
