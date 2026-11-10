# Matching CPU computation to message length

Research project based on `arrao6_project_proposal.pdf`: can at most two verified,
fixed-length DistilBERT variants reduce p95 CPU latency within a memory budget?
The dynamic-length model remains the fallback, and choosing no variants is a valid result.

Start with the [consolidated project report](docs/project-report.md) and
[CLI demonstration guide](docs/demo.md). The report links the original validation,
training-derived follow-up, and synthetic diagnostics to their archived evidence.

## Current milestone

Implemented: reproducible training-data sampling and length counts, CPU/environment
inventory, dynamic FP32 ONNX export, ORT graph optimization and operator inventory,
PyTorch/ONNX output comparisons, mask/padding checks, and a CPU calibration benchmark.
Fixed 16/32/64/128-token variants now include unchanged-weight checks, output verification,
measured routing, resident multi-session memory measurements, and bounded subset selection.

The randomized confirmation and full held-out evaluation are complete. They favored
**dynamic-only ONNX at four threads**, so the final policy retains no fixed variants.
All methods agreed on all 872 validation predictions (794 correct; 91.06% accuracy).
This project does not establish a general specialization speedup.

The first real-model run is documented in [the milestone results](docs/milestone-1.md).
The specialization study and archived measurements are in [milestone 2](docs/milestone-2.md).
The [final confirmation and held-out report](docs/final-results.md) supersedes the
earlier calibration result for deployment recommendations.

The [per-thread and transformer-optimizer follow-up](docs/followup-results.md)
is also complete: 93 timing processes and 24,324 checked requests. It again
favored a dynamic graph with no fixed variants. Additional transformer
optimization showed a small audit difference (73.56 vs 74.75 ms median run p95),
with overlapping ranges and mixed matched-block results. This training-derived
audit establishes no new classifier-generalization result or reliable speedup;
the original prediction policy remains unchanged.

## Setup (PowerShell)

Use Python 3.11-3.13 and [uv](https://docs.astral.sh/uv/). Dependencies are locked in
`uv.lock`; PyTorch is installed from its CPU wheel index. No API key or GPU is needed.
The first run downloads the public SST-2 data and pretrained model from Hugging Face.

```powershell
uv sync --locked
uv run etdl --help
uv run pytest
```

Run commands from the project root. Settings live in `configs/default.toml`.
The model's `main` revision is resolved once to a recorded commit SHA, shared by
tokenizer and model. Use a fresh artifact directory for a new model/configuration.

## Run the first milestone

```powershell
uv run etdl milestone --limit 128 --threads 1 --repeats 3 --warmup 10
```

Or run each stage separately:

```powershell
uv run etdl doctor
uv run etdl prepare --limit 128
uv run etdl export
uv run etdl check --limit 128
uv run etdl benchmark --limit 128 --threads 1 --repeats 3 --warmup 10
```

`export` refuses to overwrite an existing model. To repeat only measurement, use
`benchmark`. To rerun the entire pipeline, change `artifact_dir` in a copy of the
config, then pass `--config path/to/config.toml` **before** the subcommand.
`milestone --threads 1 2 4` runs benchmark processes in a seeded shuffled order.

## Verified specialization and selection

After the baseline passes its checks:

```powershell
uv run etdl variants --limit 128
uv run etdl study --limit 128 --threads 1 --repeats 3 --warmup 10 --memory-budget-mib 1536
uv run etdl predict "Great service!"
```

Do not request more calibration examples than the dynamic output checks cover.
`variants` resumes existing graphs only when their hashes and baseline provenance match,
then checks all eligible requested training examples, boundary lengths and masks.

The study profiles every checked option in separate processes, in a seeded randomized
order. For each observed token length, it learns the fastest median combined latency,
including padding and routing preparation. Dynamic execution always competes. Unseen
lengths route dynamically. It enumerates subsets of zero, one, or two variants, prunes
variants which never win a route, and measures each remaining configuration in a fresh
process with every retained session resident. Each graph receives a full-capacity probe
before warm-up; those probes are excluded from request timing but included in peak memory.

Selection minimizes **measured configuration p95**, subject to the process-memory cap
and zero changed predictions against the dynamic profile. The budget uses the larger
of sampled RSS and the OS lifetime high-water mark when available. It never substitutes
serialized file sizes or sums of individually measured peaks for resident memory.
If no configuration fits, the command writes an infeasibility report and creates no policy.
Without an explicit budget, it records a cap of 10% of initially available RAM.

Outputs are in `artifacts/study-t1/`: per-option profiles, configuration requests and raw
measurements, `summary.json`, `configurations.csv`, `selection.json`, and PNG/SVG
latency-memory plots. A named run preserves previous results:

```powershell
uv run etdl study --name repeat-t1 --limit 128 --threads 1 --repeats 3
uv run etdl predict "A disappointing experience." --policy artifacts/repeat-t1/selection.json
```

Use `--threads 2` and `--threads 4` for separate studies; compare randomized repeated
studies before interpreting a speedup. The small initial run can overfit timing noise.
Keep the frozen policy unchanged during a later held-out evaluation.

`predict` checks the policy's CPU/runtime identity, model hashes, source revision,
calibration hash, and output tolerances before loading sessions. A missing or changed
fixed variant is disabled, allowing verified dynamic fallback. An invalid dynamic model
or incompatible policy aborts. Each prediction logs its route without storing the input
text in `artifacts/reports/routing-decisions.jsonl`.

## Confirmation and frozen held-out evaluation

```powershell
uv run etdl compare --name confirmation --limit 128 --threads 1 2 4 --blocks 3 --warmup 10
uv run etdl freeze --comparison confirmation
uv run etdl evaluate --comparison confirmation
uv run etdl predict "Great service!"
```

Confirmation uses new training IDs disjoint from the initial selection set. Each
method/thread pair runs in a fresh process in each of three randomized blocks.
It compares PyTorch, dynamic ONNX, fixed-16 with fallback, all four smallest-fitting
buckets, and the prior selected policy. Output comparisons occur outside timers.
ORT idle thread spinning is explicitly disabled for this protocol. Reports use
process-run p95 medians and ranges, not pooled requests as independent replications.

`freeze` uses training results to select deployment threads and either the prior
policy or dynamic-only execution, subject to score/label and memory checks. It hashes
the policy and evaluation inputs before validation access. `evaluate` then runs the
five fixed comparisons on all 872 validation examples at that thread count. The
prior candidate stays a comparator even when rejected for deployment. There is one
held-out timing run per method, without latency confidence intervals.

`compare --resume` and `evaluate --resume` reuse only results matching the same
immutable plan, workload and reference hashes. Do not change a frozen policy after
seeing validation results. New comparisons require a new `--name`.

`predict` prefers `artifacts/confirmation/frozen-policy.json` when present; otherwise
it uses the initial `study-t1` policy. An explicit `--policy` selects another file.

```powershell
uv run etdl ablate --study study-t1
uv run python tools/archive_comparison.py --comparison artifacts/confirmation --output results/your-run --ablation artifacts/reports/ablation-study-t1.json --traces
```

Ablations are explicitly labeled training-only reanalyses of recorded costs. They
never modify deployment or use validation for selection. Diagnostic traces run after
timing, and the archive excludes input text and model binaries.

## Per-thread and transformer-optimizer follow-up

The [follow-up protocol](docs/followup-protocol.md) compares the existing FP32
graphs with an additional ORT transformer optimization pass and refits routing
separately at 1/2/4 threads. It allocates 256 fitting, 128 confirmation, and 128
audit rows from previously unused SST-2 training IDs before fitting. The audit
is held out from inference-policy tuning only; the classifier was trained on
SST-2 training data. It provides no new classifier-generalization estimate.

```powershell
uv run etdl followup --name followup
uv run etdl followup --name followup --resume
uv run python tools/archive_followup.py --followup artifacts/followup --output results/your-followup --traces
```

Run the first command once; use the second to resume an interrupted experiment.
The follow-up freezes fitted policies before confirmation, then freezes the
candidate before the reserved optimization audit. It leaves the original
validation results and default prediction policy intact. The archive contains
request-level measurements, immutable plans, graph checks, and post-timing CPU
traces, excluding model binaries and input text.

## Matched length-boundary diagnostic

The [protocol](docs/boundary-protocol.md) compares dynamic natural-length inputs,
the same dynamic graph with bucket padding, and fixed graphs with identical padded
inputs and masks. It uses three synthetic patterns, 13 lengths around the bucket
boundaries, four threads, and five randomized fresh-process blocks. Pretokenized
timings separate inference from routing/padding and make no accuracy claims.

The replication completed 40 processes and 8,880 checked requests with no flagged
telemetry gaps. At 17 tokens, padding to 32 increased dynamic execution p95 by
about 50-68% across the two graph families. Fixed-shape execution did not
consistently recover that cost; see the [report](docs/project-report.md) for all
lengths, matched ratios, and limitations.

```powershell
uv run etdl boundaries --name my-boundaries
uv run etdl boundaries --name my-boundaries --resume
uv run python tools/archive_boundaries.py --diagnostic artifacts/my-boundaries --output results/my-boundaries
uv run python tools/demo.py
```

Run the first command once; use `--resume` only for the same interrupted plan.
The verified graph families from `followup` must already exist. The diagnostic
records power snapshots and system/worker CPU use without altering system settings.
The [timing-quality amendment](docs/boundary-amendment.md) explains why the first
run was retained separately from its replication. Both archives preserve every
completed process; neither run changes the frozen prediction policy.

## Outputs

Generated files are local and ignored by Git:

| Path under `artifacts/` | Contents |
| --- | --- |
| `model_source.json` | Resolved pretrained model commit |
| `data/calibration.json` | Seeded SST-2 **training** sample, source revision, row IDs, token lengths |
| `models/dynamic.onnx` | Batch-one, dynamic sequence length FP32 model |
| `models/dynamic.optimized.onnx` | ORT graph optimized for the measured CPU |
| `models/dynamic.json` | Model hashes, sizes, export/startup timing, graph operators |
| `reports/environment.json` | CPU flags, cores, RAM, versions, ORT providers |
| `reports/lengths.json` | Calibration length frequencies, including special tokens |
| `reports/checks.json` | Score tolerance checks and exact label agreement |
| `reports/benchmark-t1.json` | Raw requests, median/p95, accuracy, startup, sampled RSS |
| `traces/` | Separate ORT trace, outside timed requests |

The optimized graph can contain hardware-specific operators; regenerate it on a
different machine. Benchmarks load the original ONNX graph with ORT optimizations
enabled and measure initialization explicitly. Package versions and graph hashes
are recorded alongside each run.

## Experimental rules

- Calibration uses only SST-2 training data. The validation split stays reserved
  for final evaluation after selecting a configuration. Training accuracy is a
  pipeline diagnostic, not evidence of generalization; the checkpoint was trained
  on SST-2.
- No truncation. Inputs beyond 512 tokens (including special tokens) are rejected.
  Lengths above 128 remain supported by the dynamic model up to that encoder limit.
- Output checks require both finite, close FP32 logits (`atol=rtol=1e-4`) and zero
  changed argmax labels on sampled examples and synthetic masks. They are measured
  checks, not a proof of equivalence.
- Measurements use batch size one, CPUExecutionProvider, sequential ORT execution,
  an explicit intra-op thread count, one inter-op thread, warm-up, and shuffled
  request order. Export/check and measurement run in separate processes.
- Tokenization and inference are timed separately within each request; end-to-end
  measures both together. Startup, model downloads, disk I/O, and tracing are excluded.
- RSS is a 10 ms sampled process-memory measurement, including Python/tokenizer
  overhead. Windows also reports the process lifetime high-water mark. Session
  initialization can use warm OS caches; neither is a cold device-start metric.
- The small initial sample is a smoke benchmark. It cannot establish p95 gains or
  edge-device performance. Full experiments require repeated randomized method and
  thread configurations on the actual measured hardware.

## Limitations and follow-up work

The final report includes all planned method comparisons, 1/2/4-thread confirmation,
held-out accuracy and timings, sizes, startup components, resident memory, length
metrics, traces and selector sensitivity reanalyses. It does not establish a precise
population speedup: confirmation has three process blocks on one laptop, and held-out
timing has one process per method. Runtime defaults and workloads differ from the
initial calibration, so its large apparent gain is not a deployment result.

The follow-up above tests per-thread refitting and an additional transformer
optimizer. Further work could test weight sharing, collect controlled cold-start
measurements, repeat timings, and measure actual edge devices. Any new model selection
needs evaluation data held out from that new tuning process.

Implementation references: [PyTorch ONNX export](https://docs.pytorch.org/docs/stable/onnx.html),
[ORT graph optimization](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html),
[ORT threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html),
[SST-2 dataset](https://huggingface.co/datasets/nyu-mll/glue), and
[pretrained classifier](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).
