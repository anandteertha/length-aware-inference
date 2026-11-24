# Prespecified extension: dynamic INT8 ONNX student

**Status:** protocol and code frozen before quantizing or evaluating this candidate. The November 23 presentation already has a complete FP32 result. This follow-up is exploratory because the SST-2 validation and selected IMDb sample have been inspected in earlier experiments; do not present either as a fresh independent holdout.

The [outcome and separate metadata repair](int8-student-extension-results.md) are recorded alongside this original protocol.

## Question and single graph change

The four-layer FP32 student runs before any teacher fallback. Does post-training [ONNX Runtime dynamic quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) reduce its model size and end-to-end CPU cost while keeping the frozen confidence cascade's accuracy close to the full teacher? Dynamic quantization is established prior work and is recommended by ORT as a starting point for transformer models. This project tests its interaction with our already trained student and frozen gate, not a new quantization algorithm.

Create **one** candidate from the verified, unoptimized `four_layer_distilled.onnx` graph. Use installed ORT's `quantize_dynamic` with `op_types_to_quantize=["MatMul", "Gemm"]`, `weight_type=QInt8`, `per_channel=False`, and `reduce_range=False`. Do not change weights through retraining, the model architecture, or the SST-2-trained confidence threshold. Both FP32 and INT8 execute with `ORT_ENABLE_ALL` at load time. Record source and output SHA-256, graph operators, serialized size, ORT version, and quantization arguments. A load or correctness failure is a failed candidate, not a reason to search alternate quantization options after seeing outcomes.

## Correctness and accuracy audit

1. Validate ONNX structure and finite logits. Evaluate all 872 pinned SST-2 validation sentences in order, without truncation. Compare INT8 labels and logits to the archived FP32 four-layer student outputs and evaluate label correctness. Apply the **existing** confidence gate parameters and calibration point to INT8 logits; use archived teacher logits for fallback. Record student changes, confidence-route changes, teacher fallbacks, corrected and harmed cases, and accuracy. Also recompute the FP32 gate outcome from the same archived logits and verify it equals the existing 789/872 result with 161 fallbacks.
2. Evaluate the already selected 1,000 balanced IMDb reviews from the pinned external test revision, using the archived selection indices and no truncation. Compare INT8 student predictions with archived uniform four-layer PyTorch logits and labels. This is a reused external sample, so report it as an exploratory robustness check, not new confirmation.
3. The INT8 graph is not required to meet FP32 logit tolerance; quantization changes numeric values. Report maximum/median absolute logit difference and any changed labels. Do not silently substitute a different model or retune the gate.

## Matched resident CPU benchmark

Run **only on AC power**. Use the same 872 natural SST-2 validation texts, four ORT intra-op threads per session, one inter-op thread, sequential execution, `ORT_ENABLE_ALL`, thread spinning disabled, and 32 warm-up requests per loaded session. Each method runs in a fresh process; method order and request order are seeded and shuffled within **three paired blocks**. Every process loads both the chosen student and full teacher, so resident memory is comparable. Time tokenization, student/gate/teacher execution, and the whole request separately. Exclude loading and warm-up from request latency but report startup and peak process memory. Methods:

- FP32 student only.
- INT8 student only.
- FP32 student with the frozen confidence gate.
- INT8 student with that *same* confidence gate.
- Teacher only, with the FP32 student also resident as in the previous benchmark.

Verify each process's predictions and fallback decisions against the untimed accuracy audit. Preserve per-request measurements without raw text. Report median and p95 per process, three-block paired reductions for INT8 versus FP32, and maximum RSS/peak working set. Do not pool 2,616 requests as independent timing replications. If any timing process begins or ends on battery, do not use the run for speed claims.

The candidate is useful for the project's current observed target only if its frozen-gate accuracy is within one percentage point of the teacher, its paired median request time is at least 15% below teacher-only, and peak resident memory stays under 1,536 MiB. A gain over the existing FP32 confidence cascade additionally requires lower median request time in all three matched blocks without worse observed accuracy. Report a failure or mixed result without revising this screen.

The [previous ORT transformer-optimizer follow-up](followup-results.md) found only a small, inconsistent advantage on the full teacher. This INT8 student graph is a distinct optimization. The [ORT threading guidance](https://onnxruntime.ai/docs/performance/tune-performance/threading.html) motivates fixed thread settings and disabled spinning for comparable CPU process runs.
