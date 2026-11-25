# INT8 student conversion, accuracy, and CPU timing

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

## External IMDb check

Both ONNX students scored 843/1,000 on the reused IMDb sample, while 40 of
their individual predictions differed. The FP32 ONNX student matched all
archived uniform-student labels. The teacher scored 895/1,000.
See the [external audit](../results/student-study/int8-shape-repair/external-audit.json).

## Matched CPU timing

All 15 processes completed on AC power across three shuffled blocks. Each
process ran the same 872 SST-2 requests with both model sessions resident.
Every timed prediction and routing decision matched the accuracy audit. The
[timing summary](../results/student-study/int8-shape-repair/summary.json)
records the per-block values and identifies the accuracy, speed, and memory
criteria that the candidate met.

| Policy | Correct / 872 | Median (ms) | p95 (ms) | Max observed RSS (MiB) | Max peak working set (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Teacher only | 794 | 35.40 | 87.53 | 876.4 | 952.4 |
| FP32 student | 765 | 30.64 | 59.19 | 875.9 | 952.4 |
| INT8 student | 760 | 15.22 | 33.52 | 793.5 | 871.0 |
| FP32 confidence cascade | 789 | 27.18 | 91.52 | 878.0 | 953.2 |
| INT8 confidence cascade | 785 | 14.39 | 80.35 | 793.9 | 871.4 |

The latency columns show medians of the three process medians and p95s.
Within matched blocks, the INT8 student reduced median latency by 50.3%
(48.1–62.0%) versus the FP32 student. The INT8 cascade reduced median latency
by 47.0% (46.6–54.7%) versus the FP32 cascade and 63.3% (49.9–71.5%) versus
teacher-only. Cascade p95 improved against FP32 in all three blocks and
against teacher-only in two. The INT8 cascade stayed within the 1,536 MiB
memory target and was faster in all three blocks. It fell short of the FP32
cascade by four correct predictions and of the teacher by nine, which puts it
outside the one-point teacher accuracy budget at the frozen cutoff.

## Reproduction

Run the [documented commands](../README.md#int8-onnx-student-extension) with the
pinned dependencies and verified model artifacts. `strip_shape_metadata()`
checks the temporary FP32 copy before quantization. `make_candidate()` creates
and verifies the quantized graph. `audit_validation()` and `audit_external()`
write prediction records. `benchmark_worker()` and `summarize()` implement the
AC-only timing comparison and its accuracy, latency, and memory tests.
