# Matching CPU computation to message length

Research project based on `arrao6_project_proposal.pdf`: can at most two verified,
fixed-length DistilBERT variants reduce p95 CPU latency within a memory budget?
The dynamic-length model remains the fallback, and choosing no variants is a valid result.

## Current milestone

Implemented: reproducible training-data sampling and length counts, CPU/environment
inventory, dynamic FP32 ONNX export, ORT graph optimization and operator inventory,
PyTorch/ONNX output comparisons, mask/padding checks, and a CPU calibration benchmark.

Fixed-length variants, measured routing, the memory-constrained selector, comparative
experiments, and held-out evaluation are subsequent milestones. This code does not yet
claim a specialization speedup.

The first real-model run is documented in [the milestone results](docs/milestone-1.md).

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

## Next milestones

1. Derive 16/32/64/128-token models from the same exported weights; verify scores,
   labels, masks, and padding against the dynamic baseline at relevant boundaries.
2. Measure eligible options per length and actual resident multi-session memory;
   enumerate subsets of size zero, one, or two under an explicit memory cap.
   Include dynamic execution in every routing decision, even when a variant fits.
3. Freeze selection using training data. Compare PyTorch FP32, optimized dynamic
   ONNX, one fixed variant with fallback, simple buckets, and selected variants.
4. Evaluate once on held-out validation; report accuracy/changed labels, median/p95,
   startup, disk size, resident peak memory, length breakdowns, and latency-memory
   plots. Add the memory-cap, output-check, and length-frequency ablations.

Implementation references: [PyTorch ONNX export](https://docs.pytorch.org/docs/stable/onnx.html),
[ORT graph optimization](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html),
[ORT threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html),
[SST-2 dataset](https://huggingface.co/datasets/nyu-mll/glue), and
[pretrained classifier](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).
