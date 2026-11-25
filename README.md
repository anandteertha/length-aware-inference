# Matching CPU computation to message length

**[Final course report (PDF)](output/pdf/final-report.pdf)** · [PowerPoint deck](presentation/nc-state-etdl-project-guideline-aligned.pptx)

This ECE 592 / CSC 591 project studies CPU sentiment inference under accuracy,
latency, and resident-memory constraints. The proposal in
`arrao6_project_proposal.pdf` began with fixed-length DistilBERT variants. The
[break-even map](docs/breakeven-map-results.md) found no robust fixed-shape route,
so the subsequent study trained compact students and measured a student-first
teacher cascade. The [student results](docs/student-training-results.md) and
[frozen-gate results](docs/gate-results.md) are the current primary evidence:
a confidence-only cascade was 90.48% accurate on 872 SST-2 validation examples
versus 91.06% for the teacher, with a 24.1% median reduction in matched CPU
process-block median time. A learned length-and-cost gate did worse on this
evaluation (89.33%). These are observed results on one laptop and one split;
the accuracy interval does not establish population noninferiority.

The [original fixed-shape report](docs/project-report.md) and
[CLI demonstration guide](docs/demo.md) document the earlier study.

## Training-based continuation

`scripts/training_feasibility.py` checks that a three-layer student can take CPU
gradient steps. Its [archived pilot](results/training-feasibility.json) is a timing
and memory check. `scripts/train_students.py` makes fixed, disjoint SST-2 training
partitions, caches teacher logits, and trains supervised and distilled three-layer
students. `scripts/train_capacity.py` trains one four-layer distilled student only
when the prespecified development gap triggers it. Training writes ignored
checkpoints to `artifacts/student-study/`. The GLUE validation split is excluded
from all training commands.

`scripts/evaluate_students.py` audits paired validation errors;
`scripts/export_students.py` checks dynamic ONNX outputs; and
`scripts/benchmark_students.py` measures each model in fresh CPU processes.
`scripts/fit_gate.py` freezes a teacher-benefit gate using designated training
rows. `scripts/evaluate_gate.py` applies it to validation, and
`scripts/benchmark_cascade.py` times both models resident together.
`scripts/plot_student_study.py` renders the archived accuracy-latency figure.
`scripts/train_cost_weighted.py` trains a controlled four-layer student with
distillation loss weighted by measured teacher cost; `scripts/evaluate_cost_extension.py`
compares its frozen checkpoint with uniform distillation on a pinned, balanced
1,000-review IMDb test sample. This [extension protocol](docs/cost-weighted-extension-plan.md)
was fixed before the external evaluation. IMDb is a separate review domain, not
an estimate for SST-2 sentence deployment. The
[extension result](docs/cost-weighted-extension-results.md) found 832/1,000 correct
for weighted versus 843/1,000 for uniform distillation; the paired 95% interval
includes zero. On reused SST-2 validation, weighted gained five examples, but
lost three in the targeted longest bin. The weighted model has no new ONNX timing
claim.

```powershell
uv sync --locked
uv run etdl prepare --limit 128
uv run etdl export
uv run python scripts/training_feasibility.py
uv run python scripts/train_students.py
uv run python scripts/train_capacity.py
uv run python scripts/fit_gate.py
uv run python scripts/evaluate_students.py
uv run python scripts/export_students.py
uv run python scripts/benchmark_students.py
uv run python scripts/evaluate_gate.py
uv run python scripts/benchmark_cascade.py
uv run python scripts/plot_student_study.py
uv run python scripts/train_cost_weighted.py
uv run python scripts/evaluate_cost_extension.py
```

`train_capacity.py` is conditional and raises an error if its development trigger
is not met. Training, export, and benchmark commands refuse to overwrite their
existing output directories. The
[student results](docs/student-training-results.md) report validation accuracy and
three-block CPU timing. The [frozen-gate results](docs/gate-results.md) report the
length-and-cost method, confidence-only control, and resident cascade timing. The
[accuracy and latency figure](results/student-study/accuracy-latency.png) shows
both single-model and resident-cascade tradeoffs. The
[experiment protocol](docs/training-pivot-plan.md) describes the training and routing design.

## INT8 ONNX student extension

The [fixed protocol](docs/int8-student-extension-plan.md) tests one dynamic INT8
quantization of the verified four-layer student against its FP32 graph. The
[runner](scripts/run_int8_extension.py) audits SST-2 and the already selected IMDb
reviews, then benchmarks student-only and frozen-confidence cascades in three
matched CPU blocks with both student and teacher resident. These datasets have
already been inspected, so the follow-up is exploratory. The script does not
retune the confidence threshold.

Direct conversion failed during ONNX shape inference. The separate
[metadata repair and benchmark](docs/int8-student-extension-results.md)
produced a smaller graph with the original quantizer settings. The repaired
INT8 confidence cascade reduced paired median latency by 47.0% versus the FP32
cascade and used 871.4 MiB maximum peak working set versus 953.2 MiB. It scored
785/872, nine fewer correct predictions than the teacher (1.032 percentage
points), missing the accuracy target.

Use the archived source checkpoints and predictions with the pinned dependency
versions. The first command reproduces the direct conversion attempt. The
second evaluates the separate repair. The third continues that audit into
timing when the laptop is on AC.

```powershell
uv run --locked python scripts/run_int8_extension.py --audit-only
uv run --locked python scripts/run_int8_extension.py --audit-only --refresh-shape-metadata --output artifacts/student-study/int8-shape-repair
uv run --locked python scripts/run_int8_extension.py --resume --refresh-shape-metadata --output artifacts/student-study/int8-shape-repair
```

The audit records the model and source hashes in `plan.json` and saves
`validation-audit.json` and `external-audit.json` in its output directory.
Timing writes `summary.json` after all blocks pass. Use `--check-only` to verify
the source files. Use `--resume` with the same options to resume a partial run.

## Project status and code ownership

The fixed-shape study, compact-student training, frozen gate evaluation,
resident CPU benchmark, cost-weighted comparison, and INT8 accuracy and timing
results are documented in the final course report. Its
[LaTeX source](report/final-report.tex) uses the
[official NeurIPS 2024 style](https://media.neurips.cc/Conferences/NeurIPS2024/Styles.zip).
Rebuild the PDF on Windows with MiKTeX's `pdflatex` using
`powershell -NoProfile -ExecutionPolicy Bypass -File report/build.ps1`.
The editable deck can be rebuilt with
`powershell -File presentation/build_deck.ps1` on Windows with PowerPoint.

The project-specific code in `src/etdl/`, `scripts/`, and `tools/` was developed
for this repository. It uses the external pretrained DistilBERT checkpoint,
Hugging Face datasets, PyTorch, ONNX Runtime, and standard Python packages; it
does not implement or claim ownership of those upstream models or libraries.
`src/etdl/` contains the original inference pipeline: `data.py` and `model.py`
pin and load inputs/models; `runtime.py`, `benchmark.py`, and `worker.py` execute
CPU measurements; `checks.py` verifies outputs; `variants.py`, `router.py`,
`selection.py`, and `study.py` build and select fixed-shape routes; `comparison.py`,
`followup.py`, `boundaries.py`, and `ablations.py` run subsequent diagnostics;
`config.py`, `io.py`, and `cli.py` handle settings, records, and commands.
The `scripts/` commands above implement student fitting, evaluation, export,
gate fitting, timing, and plots. `tools/` contains archival, break-even-map,
report, and demonstration utilities. `tests/` checks inference contracts,
selection, comparisons, and archives. The named functions in these files
correspond to their command stages; entry-point `main` functions parse arguments,
validate inputs, and write reproducible output records.

| Code | Main functions and classes |
| --- | --- |
| [Training](scripts/train_students.py) | `student_from_weights()` copies the teacher layers and classification head. `batches()` and `model_batch()` prepare minibatches. `evaluate()` measures development accuracy for checkpoint selection. [Capacity training](scripts/train_capacity.py) applies the development-gap rule before training the four-layer student. |
| [Student evaluation](scripts/evaluate_students.py) and [export](scripts/export_students.py) | `truncated_teacher()` builds the no-update control. `student_model()` restores a trained checkpoint and `predict()` collects logits. `Logits` wraps the classifier for export with one ONNX output. |
| [Gate fitting](scripts/fit_gate.py) | `base_features()` computes confidence and log-length. `design()` adds the interaction for joint policies. `score()` applies coefficients and optionally normalizes by cost. `route()` applies the frozen operating point. [Gate evaluation](scripts/evaluate_gate.py) compares fixed policies and computes paired accuracy intervals. |
| [Cascade timing](scripts/benchmark_cascade.py) | `parent()` creates the frozen benchmark plan and launches process blocks. `method_gate()` resolves the gate for a policy. `worker()` runs requests with both sessions resident and checks predictions. |
| [Fixed-shape variants](src/etdl/variants.py) and [router](src/etdl/router.py) | `build_variant()` exports a fixed input shape. `check_variant()` checks its logits against the reference. `RoutedClassifier` loads model sessions and applies the selected length policy. |
| [INT8 audit](scripts/run_int8_extension.py) | `make_candidate()` applies and validates quantization. `strip_shape_metadata()` prepares the separate exploratory repair. `audit_validation()` and `audit_external()` check predictions. `benchmark_worker()` times resident sessions and `summarize()` aggregates completed AC blocks. |

## Original fixed-shape study

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

The fresh 31-token test did **not** confirm the earlier transformer-optimized
fixed-32 lead: matched natural-dynamic/fixed p95 ratios were mixed across five
process blocks (median 1.005). OpenVINO's standard fixed-32 graph did beat its
dynamic graph on identical padded inputs in all five blocks (median ratio 1.073),
but beat natural-length dynamic in only four (median 1.024), with large absolute
latency variation. This does not change the frozen dynamic-only deployment policy.

The subsequent break-even map tested 24 new real sentences at each of eleven
exact lengths from 9 to 48 tokens in both ORT and OpenVINO. **No length passed
the frozen screen for a fixed-route confirmation.** In particular, both backends
favored natural dynamic in all process blocks immediately after the 16- and
32-token bucket boundaries. Near 31–32, OpenVINO results were mixed across
blocks. The [chart and full records](results/breakeven-map/summary.json) show
the measured lengths and process variation; the deployment policy remains dynamic-only.

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
