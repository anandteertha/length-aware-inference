# CPU latency of the grid-selected cost-aware gate

This comparison times the [two-split grid selection](gate-grid-results.md),
with cost exponent 0.16 and a frozen fallback score cutoff of
0.07544633992186026. The confidence-only baseline retains its original
cutoff. The latency replay uses all 872 previously inspected SST-2 validation
requests and the same FP32 teacher and four-layer student for both policies.

Each of four paired blocks runs both policies in separate processes on AC
power. Within a pair the methods receive the same request order. The first
policy alternates across blocks. Each process warms both ONNX sessions with
32 requests, keeps both models resident, and uses four threads per session
with thread spinning disabled. The primary metric is median end-to-end
request time, including tokenization, student inference, routing, and any
teacher inference. Mean latency, p95 latency, and peak working set are also
recorded.

## Results

| Policy | Correct / 872 | Teacher calls | Median (ms) | p95 (ms) | Peak working set (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Confidence-only | 789 | 161 | 10.53 | 28.67 | 927.7 |
| Cost-aware, exponent 0.16 | 790 | 149 | 10.46 | 29.46 | 928.0 |

The latency columns report the median of the four process-level values.
The memory column reports the maximum observed process peak across four
blocks. The paired comparisons below use the cost-aware time divided by
the confidence-only time within each block. Negative values favor the
cost-aware gate.

| First policy | Block | Median change | Mean change | p95 change |
| --- | ---: | ---: | ---: | ---: |
| Cost-aware | 1 | −2.18% | −0.93% | +4.81% |
| Confidence-only | 2 | +1.76% | +3.55% | +5.12% |
| Cost-aware | 3 | −3.61% | −5.42% | −3.53% |
| Confidence-only | 4 | −1.86% | −1.01% | +1.47% |
| Median paired change | | **−2.02%** | −0.97% | **+3.14%** |

All eight processes ran on AC and had zero route or prediction differences
from the frozen replay. The tuned gate got one more answer right and used
12 fewer teacher calls. It lowered median latency in three of four pairs
and raised p95 in three. It made five more teacher calls at 33+ tokens
(50 versus 45) and 17 fewer on shorter requests (99 versus 116). Request
length and the changed mix of fallbacks may contribute to the tail
difference. The timing records do not isolate the cause.

This is a small observed accuracy and median-latency improvement on the
SST-2 development workload, with a tail-latency cost. The timing uses one
machine and four process pairs. The same cutoff tied confidence-only in
the archived Stanford accuracy comparison and performed worse on new IMDb
reviews. CPU request times were measured on SST-2.

## Reproduce

With the verified ONNX exports in place, run on AC power.

```powershell
uv run --locked python scripts/time_gate_grid.py
```

The runner saves the [frozen run plan](../results/student-study/gate-grid-timing/plan.json),
[one report per process](../results/student-study/gate-grid-timing/timing/),
and the [paired comparison](../results/student-study/gate-grid-timing/timing/comparison.json).
Use `--output` to select a new directory for another run.
