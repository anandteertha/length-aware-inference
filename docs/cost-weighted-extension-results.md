# Cost-weighted distillation: frozen extension result

**Status:** training and the prespecified external evaluation are complete. The result does **not** support replacing uniform four-layer distillation or the confidence-only cascade. The [protocol](cost-weighted-extension-plan.md) and [evaluation code](../scripts/evaluate_cost_extension.py) were committed before the IMDb predictions were made. The previously inspected SST-2 validation comparison is exploratory.

## Controlled training change

The weighted and uniform four-layer students used the same teacher, initial weights, 8,192 SST-2 fitting rows, 1,024 development rows, seed 2027, optimizer, two-epoch budget, and development-only checkpoint rule. The weighted arm changed only the coefficient on each row's distillation KL term. Measured teacher costs gave weights of **0.537** for 3-16 tokens, **1.000** for 17-32, and **1.554** for 33+. Only 452/8,192 fitting sentences (5.5%) were in the 33+ bin; 5,885 (71.8%) were in the shortest bin. The [training record](../results/student-study/cost-weighted/training-summary.json) preserves costs, weights, hashes, and both epoch metrics.

| Four-layer checkpoint | Best development correct / 1,024 | Accuracy | Selected epoch |
| --- | ---: | ---: | ---: |
| Uniform distillation | 983 | 96.00% | 1 |
| Cost-weighted distillation | 979 | 95.61% | 1 |

The weighted model's second epoch reached 978/1,024 (95.51%), so the first epoch remained selected. Both checkpoints have 52,779,266 parameters. The weighted checkpoint is an ignored local artifact; its SHA-256 is recorded in the archive.

## Primary external check: IMDb reviews

The pinned [IMDb binary sentiment test split](https://huggingface.co/datasets/stanfordnlp/imdb) supplied the first 500 reviews of each class in a seed-2027 shuffled order that the tokenizer encoded to at most 512 tokens. Text was neither altered nor truncated. This produced 1,000 distinct reviews after scanning 1,202 indices. Median length was 211 tokens; 855 reviews were 129-512 tokens. The [selection record](../results/student-study/cost-weighted/external-selection.json) preserves indices, labels, lengths, class balance, and revision without archiving raw text. The teacher and both students saw the *same* 1,000 reviews.

| Model | Correct / 1,000 | Accuracy |
| --- | ---: | ---: |
| Full six-layer teacher | 895 | 89.5% |
| Uniform four-layer student | 843 | 84.3% |
| Cost-weighted four-layer student | 832 | 83.2% |

The weighted student corrected 23 reviews that uniform distillation missed, but lost 34 reviews that uniform distillation got right: **-11 net**, or **-1.1 percentage points**. The prespecified paired-example bootstrap (10,000 resamples, seed 2027) gave a descriptive 95% interval of **-2.6 to +0.4 points** for weighted minus uniform accuracy. Because this interval includes zero, the sample does not establish a population accuracy loss; it certainly offers no evidence of the intended improvement. The [row-level logits](../results/student-study/cost-weighted/external-logits.json) and [summary](../results/student-study/cost-weighted/evaluation-summary.json) make the calculation auditable.

| IMDb token length | Reviews | Teacher correct | Uniform correct | Weighted correct |
| --- | ---: | ---: | ---: | ---: |
| 17-32 | 1 | 1 | 1 | 1 |
| 33-64 | 32 | 30 | 30 | 29 |
| 65-128 | 112 | 104 | 97 | 95 |
| 129-256 | 509 | 457 | 422 | 418 |
| 257-512 | 346 | 303 | 293 | 289 |

The lone 17-32-token row cannot support a rate comparison. In every stratum with at least 30 reviews, the weighted student was below the uniform student. These strata are descriptive; this one external domain is not a broad robustness benchmark.

## Secondary, reused SST-2 validation check

The same weighted checkpoint got **770/872 (88.30%)** on SST-2 validation, versus **765/872 (87.73%)** for uniform distillation and **794/872 (91.06%)** for the teacher. The weighted-minus-uniform difference was +5 examples, or +0.57 points; paired bootstrap 95% interval **-0.34 to +1.49 points**. This validation split had already been inspected during prior experiments, so this is an exploratory comparison, not a fresh confirmation.

The gain did not land in the targeted longest SST-2 bin: weighted versus uniform correct counts were 186 versus 185 at 3-16 tokens, 413 versus 406 at 17-32, and **171 versus 174** at 33-64. The cost weighting increased the loss weight on those long training examples, but did not improve this observed long-sentence accuracy. More weight on rare long cases does not guarantee a better learned decision boundary, and the data do not identify one causal reason for the reversal.

## Interpretation and decision

This controlled ablation shows a useful boundary of the current method: CPU-cost weighting can move accuracy slightly on the familiar SST-2 validation set, yet its intended long-input benefit did not appear there and it transferred worse to longer IMDb reviews in this fixed sample. It is a one-seed, one-budget result with a domain shift, not a proof that cost-aware distillation never works. The weighted model was **not** exported or timed as ONNX, so no new latency or accuracy-latency-frontier claim follows from this run. Retain uniform four-layer distillation and the previously measured confidence-only cascade as the main comparison; report this negative extension transparently in the presentation and final report.
