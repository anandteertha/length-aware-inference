# Gate signal study results

The [protocol](gate-signal-study-plan.md) compared 60 formulas based on
regularized net gain, separate probabilities of fixes and harms, or
shrunk group estimates. It varied feature sets, regularization, and the
weight of the original confidence rank. The student and teacher stayed
frozen. The fitting pool had 2,920 rows, of which only 99 were teacher fixes
and 26 were teacher harms. The rest supplied zero incremental correctness.

## What transferred

The original calibration and previously inspected Stanford sets chose
three finalists before the fresh, length-stratified [IMDb test
sample](https://huggingface.co/datasets/stanfordnlp/imdb/tree/e6281661ce1c48d982bc483cf8a173c1bbeb5d31)
was scored. The primary choice was a 50/50 blend of confidence rank and a
ridge score using margin, length, and their interaction. The selected
separate fix/harm model used those same three inputs, and the group model
used length and confidence bins. Extra raw logit and text features did
not make the finalists.

| Policy | Correct at 174 calls | Teacher fixes | Teacher harms | Correct at fixed cutoff | Fixed-cutoff calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Confidence only | 640/720 | 32 | 15 | 640 | 174 |
| Ridge with confidence blend | 641/720 | 32 | 14 | 645 | 259 |
| Separate fix and harm models | 641/720 | 32 | 14 | 646 | 266 |
| Shrunk length groups | 640/720 | 32 | 15 | 645 | 262 |

Both one-answer gains came from the **same review**, which they kept with
the correct student rather than switching it to the wrong teacher. Each
changed six routing decisions, while only that review's correctness changed
relative to confidence. The estimated paired accuracy difference was
+0.14 points, with descriptive bootstrap interval 0.00 to +0.42 points.
With a single discordant outcome, that interval is discrete and does not
establish repeatable superiority. The fixed cutoffs acquired more correct
outputs while making 85–92 more teacher calls. The equal-call result uses
offline ranking, and no new end-to-end timing comparison was run.

## What explains the result

Net teacher gain in a group is `(fixes − harms) / group size` if all examples
in the group were routed. The least-confident fifth of each set concentrated
the available net gain.

| Cohort | Fixes / harms in least-confident fifth | Net gain there | Net gain in other four fifths |
| --- | ---: | ---: | ---: |
| SST-2 calibration (1,024) | 24 / 1 | +11.22 pp | 0.00 pp |
| Stanford development (1,812) | 76 / 32 | +12.12 pp | +0.21 pp |
| Fresh IMDb (720) | 30 / 13 | +11.81 pp | +0.35 pp |

Confidence-only found 32 of 43 available IMDb fixes, and its 174 calls
introduced 15 harms. Outside its routed subset, calling the teacher on
all 546 remaining examples would fix 11 student errors and introduce nine.
There is room to do better: a selector that knew the correct labels could
take all 43 fixes while avoiding every harm, raising accuracy from 640 to
666 correct at the same budget. That ceiling is an oracle calculation.

Length showed context-specific effects. Within Stanford's least-confident
fifth, net teacher gain was +1.54 points for up to 16 tokens, +15.14 for
17–32 tokens, and +13.27 for 33–64. On the new IMDb sample, across all
confidence levels, gain was −1.24 points at 65–96 tokens, +5.00 at 97–128,
and +3.33 at 129–160. The within-cohort contrasts do not isolate length from
review difficulty. A single rule that rewards or penalizes length in one
direction can miss these changes.

Negation was associated with greater gain even in each least-confident
fifth. The IMDb figures were +13.64 points for reviews with negation
(110 rows) and +5.88 without (34 rows). Contrast words changed direction:
in the Stanford fifth, +23.17 with contrast versus +8.90 without, compared
with +8.42 versus +18.37 on IMDb. These descriptive, small slices are
useful hypotheses. They did not produce a winning text-feature gate under
the selection procedure used here.

**Conclusion.** Confidence reliably finds where the teacher can help. The
most plausible improvement is to avoid teacher-induced errors inside
that uncertain group. This run provides one concrete example of that
mechanism, without establishing that the fitted scores can do so
consistently. Confidence-only remains the supported operating choice.

## Reproduce

```powershell
uv run --locked python scripts/study_gate_signals.py freeze
uv run --locked python scripts/study_gate_signals.py imdb
```

See the [frozen grid](../results/student-study/gate-signals/plan.json),
[older-cohort slices](../results/student-study/gate-signals/development-patterns.json),
and [fresh test results](../results/student-study/gate-signals/imdb.json).
