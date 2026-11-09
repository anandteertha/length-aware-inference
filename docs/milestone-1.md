# Milestone 1: dynamic CPU baseline

Completed November 9, 2026. This is a functional smoke run on the development
laptop, using training data. It is not the final evaluation or a speedup result.

## Setup and provenance

- CPU: Intel Core i5-1334U, 10 physical cores / 12 logical threads; Windows 11.
- Runtime: CPU PyTorch 2.10.0, Transformers 5.1.0, ONNX Runtime 1.31.1.
- Model: FP32 DistilBERT SST-2 classifier, commit
  `714eb0fa89d2f80546fda750413ed43d93601a13`.
- Dataset: `nyu-mll/glue`, SST-2 **train**, commit
  `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`.
- Sample: 64 training examples, seed 2027, lengths 3-37 tokens including special
  tokens. No truncation. Validation data was not used for checks or timing.
- Benchmark: batch one, one intra-op thread, one inter-op thread, 10 warm-up
  requests, three shuffled repetitions (192 timed requests).

## Verification

The model exported successfully with a dynamic sequence dimension and fixed batch
size one. ORT optimized the graph and saved an operator inventory and separate trace.

All 98 output comparisons passed: 64 training examples, 30 synthetic length/mask
cases, and four padding-invariance checks. There were zero changed labels; maximum
absolute logit difference was `7.39098e-6`, within `atol=rtol=1e-4`.
Synthetic checks cover lengths up to 512, beyond this small workload's maximum.
These checks do not establish equivalence on every possible input.

The 15 automated contract tests and Ruff lint/format checks pass. A Windows console
encoding failure in the exporter's default progress messages was fixed by setting
`verbose=False`. Cached pinned model/tokenizer files are reused before network access.

## Baseline measurements

| Component | Median | p95 |
| --- | ---: | ---: |
| Tokenization | 0.620 ms | 1.163 ms |
| ONNX inference | 79.603 ms | 214.082 ms |
| Tokenization + inference | 80.273 ms | 215.006 ms |

- ORT session initialization: 1,080.7 ms, with potentially warm OS file caches.
- Sampled process peak RSS: 646.0 MiB (10 ms sampling); increase over pre-session
  RSS: 344.6 MiB. This includes runtime/tokenizer overhead and can miss short peaks.
- Saved ONNX size: 256.1 MiB; optimized graph: 255.5 MiB. Both files are retained
  for inspection; that is distinct from a deployed single-model disk footprint.
- Training-sample accuracy: 64/64, reported only as a pipeline diagnostic because
  the checkpoint was fine-tuned on SST-2 training data.

The sample is small, repeated requests are dependent, and timing is specific to
this laptop and its current operating conditions. Application cold-start time,
multi-session resident memory, held-out accuracy, and comparative speedups remain
to be measured.

## Reproduce and continue

The source and environment are in `pyproject.toml` / `uv.lock`. Detailed raw results
remain local in `artifacts/reports/benchmark-t1.json` and `checks.json`; model and
calibration hashes are recorded there. ORT traces are in `artifacts/traces/`.

To repeat measurement with the existing checked model:

```powershell
uv run etdl benchmark --limit 64 --threads 1 --repeats 3 --warmup 10
```

For a fresh full pipeline, copy `configs/default.toml`, set a new `artifact_dir`,
and run `uv run etdl --config configs/your-config.toml milestone --limit 128`.
Results of later runs can differ from the snapshot in this note.

Next: create 16/32/64/128-token variants from the same exported weights, check them
against the dynamic model, and measure each eligible execution option. Then add
the selector for at most two variants with a measured memory cap and dynamic
fallback. Keep held-out validation reserved until selection is frozen.
