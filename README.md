# Matching CPU computation to message length

**[Final course report (PDF)](output/pdf/final-report.pdf)** · [Code map](docs/code-map.md) · [Run the experiments](#setup) · [Presentation archive](presentation/nc-state-etdl-project-guideline-aligned.pptx)

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

The [softer cost penalty follow-up](docs/soft-cost-gate-results.md) recovered
some of the original gate's accuracy loss. On the SST-2 replay, it got 786
correct versus 789 for confidence-only at the same 161 teacher calls. The
five paired CPU runs showed mixed latency results.
The [continuous cost-exponent sweep](docs/cost-exponent-sweep-results.md)
found a one-example gain on GLUE at equal calls. The selected exponent tied
confidence-only on the previously inspected Stanford test split.
The [exponent and cutoff heatmap](docs/gate-grid-results.md) visualizes
4,641 fixed-threshold policies against the confidence cascade's accuracy,
teacher call count, and estimated teacher cost. Both selected policies
made more teacher calls and got four fewer answers right on a new IMDb sample.
In the [SST-2 CPU benchmark](docs/gate-grid-timing-results.md), the 0.16
setting got one more answer right and reduced paired median latency by 2.02%,
while paired p95 increased by 3.14%.
The [formula search](docs/gate-formula-search-results.md) screened 58 variants
and evaluated two frozen choices on 1,812 separate Stanford test sentences.
Confidence-only led both at the same 337 teacher calls.
The [gate signal study](docs/gate-signal-study-results.md) tried 60 additional
formulas. On a fresh IMDb sample, two frozen scores avoided the same single
teacher error at a matched budget. Confidence remained the clearest predictor
of teacher benefit across the inspected cohorts.
The [teacher-harm follow-up](docs/harm-aware-gate-results.md) found a small
streaming tradeoff on 960 fresh reviews. Its best rule made 16 fewer teacher
calls and got two more right than the original confidence cutoff, while
the primary equal-budget comparison tied confidence-only.

The [original fixed-shape report](docs/project-report.md),
[reproduction guide](docs/fixed-shape-reproduction.md), and
[CLI demonstration](docs/demo.md) document the earlier study.

## Setup

Run commands from the repository root with Python 3.11–3.13 and
[uv](https://docs.astral.sh/uv/). Settings are in
[configs/default.toml](configs/default.toml). The first setup and model
commands download the pinned public assets. The experiments use CPU execution.

```powershell
uv sync --locked
uv run --locked pytest -q
```

The [code map](docs/code-map.md) explains the implemented files, classes,
and functions. Generated models, checkpoints, and scratch runs live under
`artifacts/`. Versioned results, frozen cutoffs, and timing records live
under `results/`.

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
uv run --locked etdl prepare --limit 128
uv run --locked etdl export
uv run --locked python scripts/training_feasibility.py
uv run --locked python scripts/train_students.py
uv run --locked python scripts/train_capacity.py
uv run --locked python scripts/fit_gate.py
uv run --locked python scripts/evaluate_students.py
uv run --locked python scripts/export_students.py
uv run --locked python scripts/benchmark_students.py
uv run --locked python scripts/evaluate_gate.py
uv run --locked python scripts/benchmark_cascade.py
uv run --locked python scripts/plot_student_study.py
uv run --locked python scripts/train_cost_weighted.py
uv run --locked python scripts/evaluate_cost_extension.py
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

## Routing follow-ups

Once the FP32 graphs are verified, reproduce the report's main routing follow-ups.
Use AC power for the timing command.

```powershell
uv run --locked python scripts/sweep_cost_exponent.py
uv run --locked python scripts/plot_gate_grid.py
uv run --locked python scripts/plot_gate_grid.py --imdb
uv run --locked python scripts/time_gate_grid.py
```

The [softer penalty](docs/soft-cost-gate-results.md#reproduction),
[formula search](docs/gate-formula-search-results.md#reproduce),
[gate signals](docs/gate-signal-study-results.md#reproduce), and
[teacher harm](docs/harm-aware-gate-results.md#reproduce) pages contain the
commands for their individual runs. The
[fixed-shape experiments](docs/fixed-shape-reproduction.md) have a separate
guide for the OpenVINO and ONNX Runtime comparisons.

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

The [INT8 cutoff recalibration](docs/int8-recalibration-results.md) reduced
teacher calls from 206 to 165 on SST-2 while scoring 782/872, outside the
one-point teacher accuracy budget. On the reused IMDb sample it kept
887/1,000 correct while reducing calls from 449 to 370. These are prediction
replays. The recalibrated cutoff did not advance to CPU timing.

## Report build and code ownership

The fixed-shape study, compact-student training, frozen gate evaluation,
resident CPU benchmark, cost-weighted comparison, and INT8 accuracy and timing
results are documented in the final course report. Its
[LaTeX source](report/final-report.tex) uses the
[official NeurIPS 2024 style](https://media.neurips.cc/Conferences/NeurIPS2024/Styles.zip).
Rebuild the PDF on Windows with MiKTeX's `pdflatex` using
`powershell -NoProfile -ExecutionPolicy Bypass -File report/build.ps1`.
The report's grid and paired-latency figure assets can be regenerated from
the archives with `uv run --locked python tools/plot_report_routing.py`.
The editable deck can be rebuilt with
`powershell -File presentation/build_deck.ps1` on Windows with PowerPoint.

The project developed the workload partitioning, student training and export
checks, fixed-shape selection, cascade policies, benchmarks, and result
analysis in `src/etdl/`, `scripts/`, and `tools/`. It uses the published
DistilBERT checkpoint and datasets and the upstream PyTorch, ONNX, ONNX Runtime,
and OpenVINO libraries. The [code map](docs/code-map.md) indexes the
implementation by file and symbol.
