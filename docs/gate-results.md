# Frozen length-aware gate: validation audit

**Status:** frozen-gate accuracy and resident cascade CPU timing are complete. The gate was fitted and calibrated before applying it to the 872-row validation split. This audit follows the [predeclared gate protocol](training-pivot-plan.md) and uses the four-layer distilled student from the [student study](student-training-results.md).

## Fit and calibration

The gate predicts the teacher's *incremental correctness* using the student's logit margin, log2 token length, and their interaction. Its cost-aware score divides predicted benefit by teacher ONNX inference time measured on training-derived gate-fit rows. Raw teacher medians were 9.61 ms at 3-16 tokens, 17.91 ms at 17-32, and 27.83 ms at 33+. The teacher corrected 49 student errors and harmed 5 student-correct cases among 2,048 gate-fit rows. The released teacher was trained on the SST-2 training split, so these figures are likely optimistic for unseen data.

On the disjoint 1,024-row gate-calibration partition, the selected joint-per-cost policy routed about 10% of rows and scored 97.27% versus 97.66% for the teacher. The [frozen coefficients, thresholds, calibration curves, and source hashes](../results/student-study/gate/gate-plan.json) were written before validation was read. A confidence-only control used its own lowest-cost calibration point meeting the same one-point accuracy budget.

## Validation result

| Policy | Teacher fallbacks / 872 | Correct / 872 | Accuracy | Net gain over student |
| --- | ---: | ---: | ---: | ---: |
| Four-layer student only | 0 | 765 | 87.73% | — |
| Selected joint-per-cost gate | 121 (13.9%) | 779 | 89.33% | +14 |
| Confidence-only calibration point | 161 (18.5%) | 789 | 90.48% | +24 |
| Full teacher only | 872 | 794 | 91.06% | +29 |

The selected gate routed to the teacher on 24 student errors that the teacher corrected, but also on 10 examples that the student had correct and the teacher got wrong. The net gain was 14 examples. Its remaining gap from the teacher was 15/872 = **1.72 percentage points**, missing the predeclared one-point accuracy target. A paired example bootstrap (10,000 resamples, seed 2027) gave a descriptive 95% interval of **−3.10 to −0.34 points** for selected-gate accuracy minus teacher accuracy. It assumes the validation sentences are exchangeable; the dataset was also used for aggregate reporting in the earlier project, so this is not a fresh external replication.

The confidence-only control recovered ten more net corrections with 40 more teacher fallbacks. Its paired bootstrap 95% interval for accuracy minus teacher accuracy was **−1.61 to +0.46 points**. Thus its observed 0.57-point loss meets the project target, but these 872 examples do not establish population noninferiority within one point. The [full frozen policy curves](../results/student-study/gate/gate-validation.json) include confidence-only, length-only, joint, and joint-per-cost scores at all prespecified fallback budgets. Thresholds were not adjusted to favor the validation outcome.

## Why length matters, and what went wrong

The four-layer student slightly exceeded teacher accuracy on 3-16-token validation sentences (185/197 versus 183/197), while the teacher led on 17-32 (425/467 versus 406/467) and 33+ (186/208 versus 174/208). Those are descriptive slices, not evidence that the two-example short-text difference generalizes. The selected cost-aware gate fell back on 33/197 short, 75/467 middle, and only 13/208 long sentences. It reduced accuracy on the shortest stratum from 185 to 182 correct, while recovering 12 net correct in the middle stratum and 5 in the long stratum. Dividing benefit by increasing teacher cost made long fallbacks less attractive even where the teacher had more net advantage. Training-derived gate labels may also have overstated the teacher's correctness on unseen examples.

## Resident cascade CPU timing

The [raw timing archive](../results/student-study/cascade-timing/summary.json) contains three shuffled process blocks. Every process loaded both student and teacher ONNX sessions, used four threads per session with thread spinning disabled, warmed both sessions, and then timed all 872 validation requests. End-to-end request time includes tokenization, student execution where applicable, gate computation, and teacher execution on fallback. Model loading is reported separately. All three blocks kept the same routing decisions and predictions as the frozen-logit audit; power was connected throughout. Observed maximum RSS was about **852 MiB** and process peak working set about **928 MiB**, below the 1,536 MiB limit.

| Resident policy | Correct / 872 | Fallbacks | Median of process median times | Paired median reduction vs teacher | Median of process p95 times |
| --- | ---: | ---: | ---: | ---: | ---: |
| Teacher only | 794 | 872 | 27.92 ms | — | 78.22 ms |
| Student only | 765 | 0 | 18.32 ms | 33.9% (21.3-40.0%) | 41.84 ms |
| Selected length-and-cost gate | 779 | 121 | 20.45 ms | 23.2% (5.8-41.4%) | 54.61 ms |
| Confidence-only calibration control | 789 | 161 | 21.20 ms | 24.1% (10.1-30.3%) | 67.29 ms |

The [accuracy-latency figure](../results/student-study/accuracy-latency.png) separates single-model timing from resident-cascade timing and shows the full process-median ranges. Its [SVG version](../results/student-study/accuracy-latency.svg) can be placed directly in the presentation.

The parenthesized ranges are the reductions in each of three **paired blocks**; the central reduction is their median. Both gates had lower median request time than teacher-only in all three blocks. The confidence-only control met the **observed** project thresholds: within one accuracy point of the teacher, at least 15% lower paired-median latency, and under the memory cap. It is an established cascade baseline, not evidence that the new length-conditioned score beats prior methods. The proposed length-and-cost gate missed the accuracy target. The selected gate's p95 was worse than teacher-only in one block and better in two; the confidence gate's p95 was worse in one and better in two. With only three blocks on one laptop, neither gate has a reliable tail-latency win. Fallback requests pay for both models, explaining the long tail.

The subsequent [cost-weighted distillation extension](cost-weighted-extension-results.md) used a separately frozen protocol. It did not improve accuracy on the primary external IMDb test: 832/1,000 correct versus 843/1,000 for uniform distillation. This later result does not change the frozen gate or establish a new latency comparison.
