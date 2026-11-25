# Testing alternatives to confidence-only routing

The [search protocol](gate-formula-search-plan.md) screened 58 score
formulas with the original FP32 models. The comparisons used the same
number of teacher calls. The old GLUE validation set served as
development data in this follow-up.

| Family | Formulas | Correct range at 161 calls on GLUE development |
| --- | ---: | ---: |
| Length and interaction shrinkage | 48 | 784–790 |
| Additive teacher cost | 4 | 782–789 |
| Predicted class adjustment | 6 | 779–787 |
| Confidence baseline | 1 | 789 |

The highest development score kept the full margin-by-length term, set
the standalone length term to zero, and used no cost penalty. It gained
one correct result on the set used to select it. The calibration-selected
alternative used predicted class terms, ridge penalty 10, and cost
exponent 0.5. The full grid and both frozen choices are in the
[plan](../results/student-study/score-search/plan.json).

## Separate Stanford test set

I applied the frozen scores to the source project's [binary test
trees](https://github.com/stanfordnlp/sentiment-treebank/blob/fa0a89e0cabf5eac3ff4e1d27551fad4046ec012/binary/test-binary.txt).
After removing 385 neutral roots and six normalized matches to GLUE SST-2
training or validation, 1,812 sentences remained. Confidence-only called the
teacher 337 times. Each matched ranking receives the same 337 calls.

| Policy | Correct at 337 calls | Correct with fixed cutoff | Fixed-cutoff calls |
| --- | ---: | ---: | ---: |
| Confidence only | 1,670 | 1,670 | 337 |
| Interaction only plus margin (development choice) | 1,669 | 1,673 | 495 |
| Predicted class adjustment (calibration choice) | 1,656 | 1,658 | 361 |

At equal calls, the development choice fixed four errors of the confidence
cascade and introduced five, for a difference of −0.055 percentage points
(descriptive paired 95% bootstrap interval −0.39 to +0.28). Its fixed cutoff
gained three correct predictions while making 158 additional teacher calls.
The calibration choice lost 14 correct at equal calls, a difference of
−0.77 points (−1.49 to −0.11). The equal-call comparison ranks a full batch
offline. The fixed cutoff is the streaming policy. The archived estimated
teacher costs are proxies rather than end-to-end timing.

The student alone got 1,628 right and the teacher alone got 1,675 right.
The teacher fixed 93 student mistakes and introduced 46 errors on examples
the student got right. A selector with access to the correct labels could
choose all 93 fixes and reach 1,721 at the same budget by filling remaining
calls with unchanged predictions. That bound is 51 correct predictions above
confidence-only and demonstrates room for a stronger gate. The tested score
changes did not identify that extra benefit.

See the [test result](../results/student-study/score-search/test.json) and
[prediction archive](../results/student-study/score-search/test-logits.json).

## Reproduce

Run from the project root after verifying the original FP32 exports.

```powershell
$runDir = 'artifacts/student-study/score-search'
New-Item -ItemType Directory -Force $runDir | Out-Null
Invoke-WebRequest https://raw.githubusercontent.com/stanfordnlp/sentiment-treebank/fa0a89e0cabf5eac3ff4e1d27551fad4046ec012/binary/test-binary.txt -OutFile "$runDir/stanford-test.txt"
uv run --locked python scripts/search_gate_formulas.py freeze
uv run --locked python scripts/search_gate_formulas.py test
```
