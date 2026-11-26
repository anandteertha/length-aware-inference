# INT8 confidence cutoff recalibration

Recalibration reduced teacher calls on both evaluation sets. The primary
cutoff preserved the inherited gate's accuracy on IMDb and missed the
one-percentage-point accuracy target on SST-2. The FP32 confidence cascade
remains the measured SST-2 configuration that meets the project's targets.

## Cutoff selection

The [plan](int8-recalibration-plan.md) changed only the cutoff for the same
quantized student and teacher. The existing confidence score increases as
the absolute gap between the student's two logits shrinks. A request goes
to the teacher when the score is at least the cutoff. Recalibration changes
how many requests are routed while preserving their ordering by score.

On 1,812 original Stanford test sentences, the teacher scored 1,675. The
primary rule selected score ≥0.09928, the fewest calls among the tested
cutoffs within 0.5 points of the teacher. It scored 1,666 with 363 teacher
calls. The secondary rule used the original one-point budget. It selected
score ≥0.11059 and scored 1,662 with 272 calls. The inherited FP32 cutoff
(score ≥0.09091) scored 1,674 with 431 calls on these INT8 outputs. Both new
cutoffs were frozen before replay.

## Replay

| Policy | GLUE SST-2 correct / 872 | Teacher calls | IMDb correct / 1,000 | Teacher calls |
| --- | ---: | ---: | ---: | ---: |
| Teacher only | 794 | 872 | 895 | 1,000 |
| INT8 with inherited cutoff | 785 | 206 | 887 | 449 |
| INT8 with primary cutoff | 782 | 165 | 887 | 370 |
| INT8 with secondary cutoff | 780 | 125 | 879 | 281 |

With the primary cutoff, SST-2 teacher calls fell by 41 (19.9%). Among these
41 requests, the inherited gate had used the teacher to fix four student
errors and introduce one new error. Keeping those requests on the student
therefore cost three net correct predictions. The primary cutoff finished
1.38 points below the teacher. The descriptive paired 95% bootstrap interval
for primary minus inherited accuracy was −0.92 to +0.11 percentage points.
The FP32 cascade scored 789/872 with 161 calls, seven more correct
predictions with four fewer calls than the recalibrated INT8 cascade.

On IMDb, the primary cutoff saved 79 calls (17.6%). The excluded teacher
calls had fixed six student errors and introduced six new ones, so their net
effect was zero. The primary cutoff finished 0.8 points below the teacher,
within the budget. The descriptive paired interval for the change versus
the inherited gate was −0.70 to +0.70 percentage points.

The same cutoff has a different tradeoff on the two cohorts because the
teacher's net benefit on the requests dropped by the gate differs. Both
cohorts had been examined in earlier experiments. Calibration uses the
Stanford source test split. The replays use GLUE validation and IMDb test.
The accuracy shortfall on SST-2 ended this experiment before CPU timing.
Teacher-call counts measure routing frequency. The inference time saved
depends on the cost of those calls and would require a fresh timing run.

## Reproduce

Create the repaired INT8 graph with the [INT8 audit commands](../README.md#int8-onnx-student-extension).
Fetch the pinned `stanford-test.txt` using the
[formula study's download command](gate-formula-search-results.md#reproduce).
Then run from the repository root.

```powershell
uv run --locked python scripts/recalibrate_int8.py calibrate
uv run --locked python scripts/recalibrate_int8.py replay
```

The runner writes the [calibration logits](../results/student-study/int8-recalibration/calibration-logits.json),
[frozen cutoffs and candidate results](../results/student-study/int8-recalibration/plan.json),
and [replay metrics](../results/student-study/int8-recalibration/replay.json).
To reproduce the replay alone, pass `replay` and a writable `--output`
directory containing a copy of the frozen `plan.json`.
