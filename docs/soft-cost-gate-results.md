# Softer teacher cost penalty results

The [follow-up protocol](soft-cost-gate-plan.md) varies the cost exponent while
holding the FP32 models, learned coefficients, and cost table fixed. Among the
four softer exponents, the archived training calibration selected **0.75** at
the 10% nominal fallback quantile. It handled 103 teacher calls and got 997 of
1,024 calibration examples correct. The exponent and cutoff were frozen before
the two evaluations.

## SST-2 replay

| Frozen policy | Correct / 872 | Teacher calls | Calls at 33+ tokens | Estimated added teacher ms/request |
| --- | ---: | ---: | ---: | ---: |
| Confidence only | 789 (90.48%) | 161 | 45 | 3.658 |
| Original cost penalty | 779 (89.33%) | 121 | 13 | 2.320 |
| Softer cost penalty | 786 (90.14%) | 161 | 35 | 3.439 |

The softened rule recovered seven correct predictions versus the original
gate. It fixed 34 student errors and introduced 13, versus 24 fixes and 10
harms for the original gate. At a matched offline quota of 161 teacher calls,
the original ranking got 782 correct and the softened ranking got 786.
Confidence-only still got 789 correct at that same call count. The softened
rule's difference versus confidence-only was −0.34 percentage points
(descriptive paired 95% bootstrap interval −0.92 to +0.11).

The teacher got 794/872 correct. The softened rule fell 0.92 points short,
which met the observed one-point accuracy criterion. It missed the protocol's
advancement criterion of matching confidence-only accuracy. The
[exploratory timing amendment](soft-cost-gate-plan.md#exploratory-timing-amendment)
compares the frozen cascades anyway because both had the same call count.
Estimated cost describes added teacher inference under the original
length-bin table.

## External IMDb check

| Frozen policy | Correct / 500 | Teacher calls |
| --- | ---: | ---: |
| Confidence only | 449 (89.80%) | 95 |
| Original cost penalty | 448 (89.60%) | 65 |
| Softer cost penalty | 448 (89.60%) | 95 |

The full teacher got 447/500 correct. At a matched offline quota of
95 teacher calls, both cost-based scores got 448 correct. The difference
between the softened rule and confidence-only was −0.20 percentage points
(descriptive paired 95% bootstrap interval −0.60 to 0.00). The same
fixed cutoff transferred without a clear accuracy gain here.

This corpus has a different length distribution. Of the 500 reviews, 499
were at least 33 tokens and 410 were at least 65 tokens. The cost table
assigned the same price to 499 reviews. Within that bin, varying the
exponent rescales the scores while preserving their rank, so this sample
offers limited leverage for testing a length-dependent cost change.

## Measured CPU latency at equal call count

Both frozen cascades ran all 872 SST-2 requests per process on AC power with
both models resident. All ten processes reproduced the archived routes and
predictions. Each cascade made 161 teacher calls. The first three shuffled
pairs all happened to put the softened rule first. The additional two pairs
reused the first two request orders and timed confidence-only first.

| First policy in pair | Pair | Confidence median (ms) | Softer median (ms) | Softer latency change |
| --- | ---: | ---: | ---: | ---: |
| Softer | 1 | 10.47 | 10.44 | −0.3% |
| Softer | 2 | 35.82 | 17.31 | −51.7% |
| Softer | 3 | 13.93 | 19.40 | +39.3% |
| Confidence | 1 | 10.17 | 17.68 | +73.8% |
| Confidence | 2 | 16.45 | 18.32 | +11.4% |

The softer gate won two of five pairs. The p95 direction followed the median
in all five. The run-to-run spread and the reversed pairs give no consistent
latency improvement. The softer cost penalty recovered some accuracy lost by
the original cost-aware rule, while confidence-only remained more accurate at
equal call counts in both evaluations.

See the [initial timing archive](../results/student-study/soft-cost/timing/)
and [reverse-order timing archive](../results/student-study/soft-cost/timing-reversed/).

## Reproduction

Run these commands from the repository root. The IMDb check uses the pinned
datasets and the verified FP32 exports from the original study.

```powershell
uv run --locked python scripts/soft_cost_gate.py calibrate
uv run --locked python scripts/soft_cost_gate.py sst2
uv run --locked python scripts/soft_cost_gate.py imdb
uv run --locked python scripts/soft_cost_gate.py timing-plan
uv run --locked python scripts/benchmark_cascade.py --gate artifacts/student-study/soft-cost/timing-plan --output artifacts/student-study/soft-cost/timing --methods selected_gate confidence_gate
uv run --locked python scripts/benchmark_cascade.py --gate artifacts/student-study/soft-cost/timing-plan --output artifacts/student-study/soft-cost/timing-reversed --blocks 2 --methods confidence_gate selected_gate
```

The [frozen calibration plan](../results/student-study/soft-cost/plan.json),
[SST-2 comparison](../results/student-study/soft-cost/sst2.json), and
[IMDb comparison](../results/student-study/soft-cost/imdb.json) record the
cutoffs, input hashes, operating points, and paired comparisons.
