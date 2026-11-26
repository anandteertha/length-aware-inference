# Predicting teacher harm near the confidence cutoff

This follow-up asks whether a small adjustment around the existing confidence
cutoff can prevent teacher-induced errors at the same teacher-call budget.
The FP32 teacher and student weights stay fixed.

Fit two L2 logistic models on the 2,920 earlier fit rows plus the 1,812
previously inspected Stanford rows. One estimates the chance that the teacher
fixes a student error. The other estimates the chance that the teacher spoils
a correct student answer. Restrict the fit pool to examples within two raw
logit-margin units of the original confidence cutoff. Use the three feature
sets from the previous [signal study](gate-signal-study-plan.md) and penalties
10 and 40. This is an error prediction experiment with fixed classifiers,
related to [learned deferral](https://arxiv.org/abs/2307.02764).

Compare two candidate families. The first reorders borderline requests using
negative predicted harm. The second uses predicted fix probability minus
twice predicted harm probability. Convert those values to ranks using the fit
data. Apply the correction only within a window of 0.75 or 1.5 raw margin units
on either side of the original cutoff, with three adjustment weights
(0.5, 1, 2). Bound the adjusted score so no request outside the window is
passed by a request inside it. Preserve the original confidence ranking
outside the window.

Choose one formula per family using the original 1,024 calibration rows
and the already inspected 720 IMDb reviews, at the original confidence
gate's call count in each. Prefer more total correct answers, fewer teacher
harms, and fewer changed routes, in that order. The harm-only family is
the primary candidate. Freeze its score cutoff and the other finalist's
score cutoff at the original confidence call quota from the old IMDb sample.

Use 960 fresh movie reviews from the pinned IMDb train archive, 240 per
label in each of two token groups (up to 112 and 113–176). Exclude all exact
normalized matches to the previously scored IMDb test and train reviews, and
remove duplicate normalized texts. Select with seed 2067 before inference.
Compare accuracy, fixes, harms, and route changes at exactly the
confidence-only call count. Report fixed cutoff results separately and
include descriptive paired accuracy intervals. Make no latency claim
from offline quotas or from the SST-2 cost table on IMDb.
