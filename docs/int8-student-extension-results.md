# INT8 student conversion and accuracy audit

## Original conversion

The [prespecified conversion](int8-student-extension-plan.md) of the verified
four-layer student failed inside ONNX shape inference, which reported an
inferred dimension of 768 where an existing annotation specified 2. This
attempt produced no quantized model.

## Exploratory metadata repair

The follow-up used a separate output directory. It removed the intermediate
ONNX `value_info` annotations from a temporary copy, preserving the graph
operators, weights, and input and output declarations. Before quantization,
the two FP32 graphs' logits passed `atol=rtol=1e-4` on probes of 3, 16, 32,
55, 211, and 512 tokens. All quantizer settings and the confidence cutoff
remained fixed. The graph passed the ONNX checker and ORT smoke inference.

The FP32 graph was 211,597,863 bytes and the INT8 graph was 124,811,142 bytes
(41.0% smaller). This is a serialized size measurement.

## SST-2 validation

| Policy | FP32 correct / 872 | INT8 correct / 872 | FP32 teacher calls | INT8 teacher calls |
| --- | ---: | ---: | ---: | ---: |
| Student only | 765 (87.73%) | 760 (87.16%) | 0 | 0 |
| Frozen confidence cascade | 789 (90.48%) | 785 (90.02%) | 161 | 206 |

The full teacher got 794/872 correct (91.06%). The repaired INT8 confidence
cascade missed the one-point accuracy target by losing nine correct predictions
relative to the teacher, a 1.032 percentage-point drop. The student changed 27
labels and 61 routing decisions at the unchanged threshold. It made 45 more
teacher calls in total. These data do not support replacing the FP32 cascade.

See the [candidate plan](../results/student-study/int8-shape-repair/plan.json)
and [prediction audit](../results/student-study/int8-shape-repair/validation-audit.json).
Both accuracy datasets had already been inspected in earlier project stages.

## External IMDb check and timing status

Both ONNX students scored 843/1,000 on the reused IMDb sample, while 40 of
their individual predictions differed. The FP32 ONNX student matched all
archived uniform-student labels. The teacher scored 895/1,000.
See the [external audit](../results/student-study/int8-shape-repair/external-audit.json).

The laptop reported battery power, so the AC-only matched timing and resident
memory comparison remains pending. The SST-2 accuracy miss already rules out
this INT8 cascade under the project's observed acceptance criteria.

## Reproduction

Run the [documented commands](../README.md#int8-onnx-student-extension) with the
pinned dependencies and verified model artifacts. `strip_shape_metadata()`
checks the temporary FP32 copy before quantization. `make_candidate()` creates
and verifies the quantized graph. `audit_validation()` and `audit_external()`
write prediction records. `benchmark_worker()` and `summarize()` implement the
AC-only timing comparison and its accuracy, latency, and memory tests.
