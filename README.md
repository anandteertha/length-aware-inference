# Matching CPU computation to message length

Research project based on `arrao6_project_proposal.pdf`: can at most two verified,
fixed-length DistilBERT variants reduce p95 CPU latency within a memory budget?
The dynamic-length model remains the fallback, and choosing no variants is a valid result.

## Current milestone

Implemented: reproducible training-data sampling and length counts, CPU/environment
inventory, dynamic FP32 ONNX export, ORT graph optimization and operator inventory,
PyTorch/ONNX output comparisons, mask/padding checks, and a CPU calibration benchmark.
Fixed 16/32/64/128-token variants now include unchanged-weight checks, output verification,
measured routing, resident multi-session memory measurements, and bounded subset selection.

Larger comparative experiments, ablations, and held-out evaluation remain. This code
does not yet claim a held-out specialization speedup.

The first real-model run is documented in [the milestone results](docs/milestone-1.md).
The specialization study and archived measurements are in [milestone 2](docs/milestone-2.md).

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

## Remaining experiments

1. Expand training-only calibration and repeat randomized runs at 1/2/4 CPU threads.
   Include PyTorch FP32, optimized dynamic
   ONNX, one fixed variant with fallback, simple buckets, and selected variants.
2. Evaluate once on held-out validation after freezing selection; report accuracy/changed labels, median/p95,
   startup, disk size, resident peak memory, length breakdowns, and latency-memory
   plots. Add the memory-cap, output-check, and length-frequency ablations.

Implementation references: [PyTorch ONNX export](https://docs.pytorch.org/docs/stable/onnx.html),
[ORT graph optimization](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html),
[ORT threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html),
[SST-2 dataset](https://huggingface.co/datasets/nyu-mll/glue), and
[pretrained classifier](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).
