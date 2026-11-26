# Continuous sweep of the teacher cost exponent

The cost-aware cascade scores request \(i\) as \(G_i/C_i^\alpha\), where
\(G_i\) is the original ridge estimate of net teacher gain and \(C_i\) is
the estimated teacher time for the request's token-length bin. The sweep
holds both models and the ridge coefficients fixed. At \(\alpha=0\), the
cost penalty disappears. A negative exponent favors larger costs when the
predicted gain is positive.

## Matched teacher calls

| Exponent | GLUE SST-2 correct / 872 | GLUE calls at 33+ tokens | Stanford correct / 1,812 |
| --- | ---: | ---: | ---: |
| Confidence-only | 789 | 45 | 1,670 |
| 0.00 | 790 | 58 | 1,669 |
| 0.0666, selected on GLUE | **790** | 55 | 1,670 |
| 0.50 | 788 | 41 | 1,670 |
| 0.60 | 787 | 39 | **1,673** |
| 0.65 | 786 | 37 | **1,673** |
| 0.70 | 786 | 35 | 1,671 |
| 0.75 | 786 | 35 | 1,671 |
| 1.00 | 782 | 25 | 1,668 |

All rows use 161 teacher calls on GLUE and 337 on Stanford. This offline
comparison ranks each complete cohort. The previous frozen-cutoff run at
exponent 1.00 made only 121 GLUE calls, including 13 at 33+ tokens.

The GLUE-selected exponent produced one extra correct prediction in the
33+ token group and no net change in the shorter groups, while increasing
estimated teacher inference from 3.66 to 3.83 ms per request. The selected
route was stable over the open interval from 0.05174 to 0.08145. Its
equal-call Stanford difference was 0.00 percentage points, with a
descriptive paired 95% bootstrap interval of −0.28 to +0.33 points.
The fixed cutoff from the original calibration set gained four correct
on GLUE with 70 more calls and three on Stanford with 148 more calls.

Across positive and zero penalties from 0 to 1, the best GLUE result was
790 and the best retrospective Stanford result was 1,673. Extending the
sweep to −1 through 2 raised the GLUE maximum to 791 by rewarding long
requests. No stable exponent interval beat confidence-only on both
cohorts at equal calls. Both cohorts had been inspected previously.

## Matching estimated teacher cost

A separate search over \(\alpha\) from 0 to 1 considered each threshold
whose total estimated teacher time was within the confidence-only budget.
It maximized correct predictions and broke ties by fewer calls. On GLUE,
exponent 0.03638 with cutoff 0.11174 got 790 correct with 140 calls versus
789 with 161 calls for confidence-only. Estimated added teacher time fell
from 3.66 to 3.36 ms per request. The candidate made 50 calls at 33+ tokens
and got 184 correct in that group, compared with 45 calls and 183 correct
for confidence-only.

Freezing that exponent and threshold for Stanford gave 1,668 correct with
290 calls versus 1,670 with 337 calls for confidence-only. Estimated added
teacher time fell from 3.62 to 3.35 ms per request. If the search instead
uses Stanford labels to pick its own exponent and cutoff, its best result
under the same cost cap is 1,676 correct with 354 calls at approximately
0.6574. That operating point was selected on Stanford and needs a separate
evaluation. These cost figures sum the original bin medians and do not
measure end-to-end latency.

## How the search works

The ranking of two same-sign predictions with different costs can change
only where their scores are equal. Taking logs of their absolute values
gives the crossing exponent

\[
\alpha = \frac{\log |G_i| - \log |G_j|}{\log C_i - \log C_j}.
\]

The runner checks a representative exponent between successive crossings
and checks the reported grid points and range endpoints explicitly.
It uses stable row order for exact score ties in equal-call comparisons.
An advantage at an isolated tie would be sensitive to arbitrarily small
perturbations of the coefficient or input scores. The search covers this
cost exponent with the original benefit model and fixed cost groups.

## Reproduction

```powershell
uv run --locked python scripts/sweep_cost_exponent.py
```

The [frozen plan](../results/student-study/cost-exponent-sweep/plan.json)
links the source hashes and GLUE choices. The
[results archive](../results/student-study/cost-exponent-sweep/results.json)
contains all distinct routing intervals, the named exponents, and the
fixed-cutoff transfer. No new CPU timing was conducted.
