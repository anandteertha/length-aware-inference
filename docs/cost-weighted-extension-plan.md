# Prespecified extension: CPU-cost-weighted distillation

**Status:** protocol fixed before training this extension or reading its external test labels through model predictions. The original SST-2 validation results have already been inspected, so any new comparison on those 872 rows is exploratory. The [uniform four-layer student and confidence cascade](gate-results.md) remain the main presentation result.

## Hypothesis and controlled comparison

The teacher's measured ONNX inference median on designated SST-2 gate-fit rows was 9.61 ms for 3-16 tokens, 17.91 ms for 17-32, and 27.83 ms for 33+ (see [frozen gate plan](../results/student-study/gate/gate-plan.json)). A teacher fallback on longer inputs costs more. Test whether a four-layer student trained to match the teacher more strongly on those expensive cases reduces its errors, especially on longer text.

Use **exactly the same** teacher checkpoint, 8,192 student-fitting rows, 1,024 development rows, initial four-layer weights, seed 2027, batch size 16, AdamW `2e-5`, two epochs, temperature 2, alpha 0.5, and best-development-checkpoint rule as the uniform four-layer distilled student. Change only the distillation weight for row `i`:

`w_i = clip(teacher_cost(length_bin(i)) / 17.9136 ms, 0.5, 2.0)`

`loss_i = 0.5 * CE_i + 0.5 * w_i * 2^2 * KL(teacher_logits_i / 2 || student_logits_i / 2)`.

Keep cross-entropy unweighted and average the per-row losses over each batch. Use the raw training-derived teacher costs above; do not refit weights from final validation or external test results. Compare with uniform four-layer distillation. This is a bounded course experiment, not a claim that cost-aware distillation is globally new; [cascade-aware training](https://arxiv.org/abs/2406.00060) is prior work.

## Evaluation and interpretation

1. Select the extension checkpoint only on the same SST-2 development partition used by the uniform student. Record training loss, development accuracy, training time, model size, and hashes.
2. Primary new generalization check: pin the [IMDb binary sentiment test split](https://huggingface.co/datasets/stanfordnlp/imdb) at revision `e6281661ce1c48d982bc483cf8a173c1bbeb5d31`. With seed 2027, traverse a random permutation of its test indices and choose the first **500 negative and 500 positive reviews** that the pinned tokenizer encodes to at most 512 tokens. Preserve original text and labels; never truncate. Record selected indices, eligible and scanned counts, token lengths, and revision. Compare the full teacher, uniform four-layer student, and cost-weighted four-layer student on exactly these 1,000 reviews. This is an external *review-level* domain and length shift, not an in-domain estimate for SST-2 sentences.
3. Report paired correct counts and the cost-weighted-minus-uniform accuracy difference with a 95% paired-example bootstrap interval (10,000 resamples, seed 2027). Report differences by prespecified token strata only where at least 30 examples occur. A one-seed, one-sample result is not a universal claim.
4. Secondary exploratory check: compare the cost-weighted model on the previously inspected SST-2 validation rows. Label this reuse clearly; do not choose training changes from it.
5. If exporting the new student to ONNX, check logits and labels first. Its architecture is identical to the uniform four-layer student, but a speed claim still needs measured runtime. Preserve and report an unfavorable outcome.

The extension is useful only if it changes the accuracy-latency frontier or explains why cost weighting does not transfer. It must not replace the already reported confidence-only cascade result after seeing an unfavorable comparison.
