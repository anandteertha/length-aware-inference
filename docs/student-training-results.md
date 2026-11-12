# Trained-student study: first accuracy and export results

**Status:** accuracy, ONNX export, and student-only CPU timing are complete. The subsequent [frozen-gate study](gate-results.md) measures the resident cascade. Do not infer a cascade speedup from student-only timing.

The course presentation is November 23, 2026; the report is due November 30. This study follows the [predeclared training pivot](training-pivot-plan.md). The original fixed-shape study remains the baseline finding. All model choices below were made using SST-2 training-derived partitions; the released teacher had already been fine-tuned on the SST-2 training split. The 872-row GLUE validation set was previously used for aggregate reporting in the original project, but these student checkpoints and thresholds were not chosen on its labels.

## Protocol

- Pinned teacher: `distilbert/distilbert-base-uncased-finetuned-sst-2-english` at revision `714eb0fa89d2f80546fda750413ed43d93601a13`.
- Pinned GLUE dataset revision: `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`.
- Seed 2027; disjoint training-split partitions: 8,192 student fitting, 1,024 student development, 2,048 gate fitting, and 1,024 gate calibration rows. [Full row IDs and parameters](../results/student-study/manifest.json).
- Two epochs, batch size 16, four CPU threads, AdamW at `2e-5`. Three-layer supervised and uniformly distilled students start from the teacher's embeddings, first three layers, and classifier. Distillation uses temperature 2 and equal cross-entropy/KL weight.
- The predeclared capacity trigger fired: best three-layer development accuracy was 94.14% versus 98.83% for the teacher, a 4.69-point gap above the 3-point threshold. A four-layer distilled student then used the same split, seed, epochs, and checkpoint rule.
- Best development checkpoint was epoch 1 for all three trained arms. See [three-layer training summaries](../results/student-study/summary.json) and [four-layer training summary](../results/student-study/capacity-summary.json).

## Validation accuracy

| Model | Layers | Correct / 872 | Accuracy | Difference from teacher |
| --- | ---: | ---: | ---: | ---: |
| Full pretrained teacher | 6 | 794 | 91.06% | — |
| Truncated teacher; no student updates | 3 | 645 | 73.97% | −17.09 points |
| Supervised student | 3 | 735 | 84.29% | −6.77 points |
| Uniformly distilled student | 3 | 739 | 84.75% | −6.31 points |
| Four-layer distilled student | 4 | 765 | 87.73% | −3.33 points |

Training improves the three-layer truncation by 90-94 correct examples. Uniform distillation adds four correct examples over supervised training under this one seed and budget; that difference is too small to treat as a reliable improvement. The four-layer distilled student adds 26 correct examples over the three-layer distilled student. The teacher still leads it by 29 net correct examples: the teacher corrects 50 of its errors, while the student corrects 21 teacher errors. These [paired counts and row-level logits](../results/student-study/validation-summary.json) give the gate a concrete target but do not show that an inexpensive gate can identify the right cases.

The prespecified length strata give a clue, not a new selection rule: at 3-16 tokens, the four-layer student got 185/197 correct and the teacher 183/197; at 17-32, they got 406/467 and 425/467; at 33+, 174/208 and 186/208. The two-example short-text difference is small. The conditional routing gate must be fitted on its designated training-derived partition and assessed without changing a threshold from these validation labels.

## ONNX CPU timing

The [raw process archive](../results/student-study/cpu-timing/summary.json) contains three shuffled blocks. Each method ran in a separate process on AC power with four ORT threads, 32 warm-up requests, and the same 872 natural validation texts. End-to-end request time includes tokenization and inference, excluding model loading. Every ONNX process matched its PyTorch validation predictions exactly. Paired reductions compare each student's median request time with the teacher's median in the same block; they are more useful than comparing independent medians when the laptop's absolute latency drifts.

| Model | Median of process median times | Paired median reduction vs teacher | Range across three blocks | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Dynamic ONNX teacher | 19.83 ms | — | 14.84-21.56 ms | 91.06% |
| Supervised 3-layer | 9.68 ms | 49.2% | 34.8-65.4% | 84.29% |
| Distilled 3-layer | 10.57 ms | 38.5% | 28.3-51.0% | 84.75% |
| Distilled 4-layer | 13.53 ms | 13.7% | 8.9-43.9% | 87.73% |

The four-layer student was faster in all three matched blocks, but its p95 comparison was mixed (worse in two blocks, better in one). The three-layer distilled model had lower p95 in all three blocks, with a sizable accuracy loss. The timing differences between the two three-layer weight sets also show substantial process variation; this experiment does not establish that supervised weights execute inherently faster than distilled weights. Single-model observed maximum RSS was about 645 MiB for the teacher, 560-563 MiB for three-layer students, and 590 MiB for the four-layer student. A cascade must keep student and teacher resident, so these figures are not its memory result. Three blocks on one laptop bound the strength of any population speed claim.

## Export and deployment questions

All three trained students exported to dynamic FP32 ONNX. On 16 training rows per model, ONNX and PyTorch logits passed `atol=rtol=1e-4`; there were zero changed labels and maximum absolute logit error below `5.1e-6`. The [export archive](../results/student-study/onnx-export-summary.json) records model hashes and sizes. The three-layer ONNX graphs are about 183 MB, the four-layer graph about 212 MB, and the full teacher graph about 269 MB. Weight size falls less than layer count because embeddings are shared across depths.

The [gate results](gate-results.md) test whether length and confidence can regain accuracy without paying too often for both student and teacher. They report median, mean, p95, resident memory, and the full accuracy-latency curve. A student-only speedup by itself is not evidence that a student-first cascade wins.
