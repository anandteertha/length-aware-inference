# Teacher harm near the confidence cutoff

The [protocol](harm-aware-gate-plan.md) screened 72 variants and froze one
per family before scoring 960 fresh IMDb movie reviews. The verified FP32
student and teacher stayed fixed. Both selected formulas used a window of
0.75 raw logit-margin units around the original confidence cutoff, the text
feature set, and an L2 penalty of 40 for the logistic models. The fit pool
near the cutoff had 1,107 examples, including 111 teacher fixes and 42 harms.
The [frozen grid](../results/student-study/teacher-harm/plan.json) records all
selection outcomes.

## Primary result at equal teacher calls

| Policy at 222 calls | Teacher fixes | Teacher harms | Correct / 960 |
| --- | ---: | ---: | ---: |
| Confidence only | 60 | 17 | 865 |
| Predicted harm only | 60 | 18 | 864 |
| Predicted fixes minus twice predicted harms | 61 | 18 | 865 |

The harm-only rule swapped eight calls and failed to remove any teacher
harms. It added one. The second rule swapped nine calls, removed one
harm, and added two harms and one fix. The latter rule's paired difference
versus confidence was 0.00 percentage points, with a descriptive 95%
bootstrap interval of −0.42 to +0.42 points.

## Streaming cutoffs and what the budget explains

| Policy | Calls | Teacher fixes | Teacher harms | Correct / 960 |
| --- | ---: | ---: | ---: | ---: |
| Original confidence cutoff | 222 | 60 | 17 | 865 |
| Confidence ranking capped at 206 calls, diagnostic | 206 | 60 | 16 | 866 |
| Fix-minus-harm score at its frozen cutoff | 206 | 59 | 14 | 867 |
| Harm-only score at its frozen cutoff | 226 | 60 | 18 | 864 |

The fix-minus-harm cutoff got two more right while calling the teacher 16
fewer times than the original confidence cutoff. With confidence capped
at the same 206 calls, the advantage was one correct answer. The
adjusted policy avoided two harms and gave up one fix in that comparison.
The 206-call confidence cap is an offline diagnostic added after evaluation.
Fewer calls are observed. End-to-end latency was not measured.

The fresh window contained 125 examples with 13 fixes and 10 harms. Within
that window, the selected harm model ranked a randomly paired harm above a
non-harm about 43% of the time (AUROC 0.43). With just ten positive events,
this is a descriptive result consistent with the veto's failure. It offers
little evidence that the fitted score can reliably identify which
borderline student answers to protect.

The streaming tradeoff is a small positive result at one operating point.
The primary equal-call result supports keeping confidence-only as the
default routing policy.

## Reproduce

```powershell
uv run --locked python scripts/study_teacher_harm.py freeze
uv run --locked python scripts/study_teacher_harm.py imdb
```

See the [fresh results](../results/student-study/teacher-harm/imdb.json),
[selection IDs](../results/student-study/teacher-harm/imdb-selection.json),
and [predictions](../results/student-study/teacher-harm/imdb-logits.json).
