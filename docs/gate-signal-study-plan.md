# Gate signal study protocol

Keep the verified FP32 student and teacher frozen. Use the 2,048 original
gate-fit rows and the 872 previously inspected GLUE validation rows for model
fitting. Use the original 1,024 calibration rows and the 1,812 previously
inspected Stanford test rows for model selection. These roles apply to this
follow-up only. Neither of these earlier evaluations is new evidence for a
winning candidate in this round.

## Score families

Fit all scores using only inputs available after running the student.

- **Ridge net gain.** Predict teacher correctness minus student correctness.
- **Separate fix and harm probabilities.** Use two regularized logistic
  models and route by \(P(\mathrm{fix}\mid x)-P(\mathrm{harm}\mid x)\).
  This tests whether different predictors explain these two events. Learning
  from conditional correctness has [prior work](https://arxiv.org/abs/2307.02764).
- **Shrunk group estimates.** Sort requests into five or ten margin-quantile
  bins. Within each, estimate net gain by length group or predicted polarity
  while shrinking each estimate toward the pooled margin bin with a prior
  weight of 25 or 100 rows.

The first two families use three nested feature sets. The first uses margin,
log token count, and their interaction. The second adds predicted class,
class by margin, mean logit, and the square of standardized margin. The third
adds indicators for negation and contrast words, punctuation rate, and tokens
per word. Fit ridge and logistic penalties of 10 and 40. This gives 20 fitted
scores. Compare each directly or after blending its fit-set percentile rank
with the confidence rank at weights 25% and 50% (60 candidate formulas).
Use 101 empirical percentile anchors from the fit rows and clip standardized
continuous inputs to five standard deviations before constructing products.

For each family, choose the best formula by sum of correct predictions at
the original confidence gate's observed call count in each selection set.
Break ties by fewer total calls in the highest SST-2 cost bin, then
grid order. Freeze one choice per family and the overall selection winner.
Freeze individual request cutoffs at the score ranked at the confidence
gate's call quota on the calibration rows. Resolve exact offline score ties
by lower student confidence, then input order.

## Fresh check and diagnostics

Use a seeded, balanced sample of 720 new reviews from the pinned
[IMDb test split](https://huggingface.co/datasets/stanfordnlp/imdb/tree/e6281661ce1c48d982bc483cf8a173c1bbeb5d31).
Take 120 per label in each of three token groups: up to 96, 97–128, and
129–160. Preserve natural lengths. Exclude normalized duplicates and all
normalized matches to the 1,000 prior IMDb test reviews or the 500 prior
IMDb train reviews. Freeze review IDs before inference and report this
stratified sample as cross-domain evidence. It is not representative of the
full length distribution of IMDb. The primary accuracy comparison uses
exactly the original confidence gate's number of teacher calls on these
reviews, with paired descriptive 95% bootstrap intervals. Also report
fixed cutoff accuracy and calls. The original SST-2 cost bins cannot resolve
most of these IMDb lengths, so this round does not infer latency from them.

Report teacher fixes, harms, and confidence-routed counts by within-cohort
margin quintile, length, predicted polarity, and the presence of cue words.
Compare the matched policies within the same cohorts and report whether the
size and direction of differences persist. Count the oracle headroom from
selecting all available fixes up to the actual call quota. Slice results
are descriptive and may reflect differences in review difficulty and mix.
