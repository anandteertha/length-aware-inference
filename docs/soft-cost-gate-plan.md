# Softer teacher cost penalty

This is a follow-up to the [original gate study](gate-results.md). Only the cost
exponent changes. The four-layer FP32 student, teacher, ridge coefficients,
length bins, and teacher cost table stay fixed.

## Calibration

Rank requests by predicted teacher benefit divided by teacher cost in
milliseconds raised to a power. The candidate powers are 0, 0.25, 0.5 and 0.75.
Power 0 ignores cost. Power 1 reproduces the original rule and serves as a
control alongside the original confidence-only gate. A fixed reference time
would only multiply all scores for a given exponent by the same constant, so
the implementation uses the numeric millisecond values directly.

For each candidate, apply the original calibration quantile grid (0, 10%, 20%,
30%, 40%, 50%, 70%, 100%) to the 1,024 archived SST-2 calibration rows. Among
points with accuracy within one point of the teacher, select the one with
lowest estimated average added teacher time using the original measured cost
table. Break ties by fewer teacher calls, then by smaller exponent and smaller
nominal budget. Freeze the selection and both original controls before running
the evaluation command.

## Evaluation

- Replay the frozen policies on the previously inspected SST-2 validation
  logits. This comparison is exploratory.
- Evaluate the same policies on 500 fresh, balanced IMDb train reviews with
  up to 128 tokens under the pinned tokenizer. Use a seeded shuffle and filter
  duplicate normalized texts and exact matches to the earlier IMDb test audit
  sample. Lock IDs before prediction. This tests transfer to a different
  distribution, including lengths that the original cost table did not resolve.
- Report fixed cutoff accuracy, calls, fixes and harms, and a descriptive 95%
  paired example bootstrap interval for the accuracy difference against each
  original control. Report the 33+ and 65+ token call counts. Use the
  confidence-only policy's observed number of calls as an offline ranking
  comparison at equal call count. That diagnostic uses scores to rank the batch
  and does not select a deployment cutoff.
- The cost table gives a routing proxy. In particular, the 33+ bin should not
  stand in for measured latency on IMDb. Advance to an in-domain matched AC
  benchmark if the selected softened rule meets the one-point teacher accuracy
  target, is at least as accurate as confidence-only, and has lower estimated
  added teacher time on the SST-2 replay. These advancement criteria avoid
  timing a candidate that has already missed the intended improvement.

Report results for the frozen selection even if a different candidate looks
better after validation. The original protocol, report, and results remain
the record of the prior experiment.

## Exploratory timing amendment

The replay missed the advancement criterion against confidence-only accuracy.
Both frozen cascades called the teacher exactly 161 times, so a separate
timing comparison can establish whether the different mix of requests
changes measured latency. After the replay and before timing, the comparison
was set to three shuffled AC blocks. Both models remain resident in every
process, using four threads and the full 872-row SST-2 validation set. The
frozen cutoffs come from the calibration plan without any further selection.

The seeded shuffle put the softer gate first in every initial block. Two
additional pairs rerun the first two request orders with confidence-only
timed first, using the same frozen policies. Report the original and
reversed process orders separately and show all paired latency differences.
