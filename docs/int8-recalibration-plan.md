# Recalibrating the INT8 confidence cutoff

The original [INT8 experiment](int8-student-extension-results.md) kept the
FP32 cutoff fixed. Quantization changed routing on 61 SST-2 requests. This
follow-up keeps both ONNX graphs fixed and changes only the INT8 cutoff.

Compute INT8 student logits on the 1,812 previously inspected Stanford source
test sentences archived in the [gate formula study](gate-formula-search-results.md).
Use their archived teacher logits and labels. These sentences are separate
from the 872 GLUE SST-2 validation rows used for the latency comparison.
Stanford was inspected for earlier FP32 experiments, so this follow-up is
exploratory and does not turn either cohort into a new project holdout.

Test cutoffs targeting the top 10%, 15%, 20%, 25%, 30%, 35%, 40%, 50%, 65%,
80%, or 100% of calibration requests ranked by INT8 uncertainty. Choose the
fewest calls that keep observed calibration accuracy within **0.5 percentage
points** of the teacher. This tighter calibration target provides some
headroom for transfer to the evaluation cohort. Report a second selection at
the original one-point budget to show how sensitive the decision is to that
buffer. Freeze both before evaluating either on the 872 archived GLUE rows.
Replay both on the already inspected 1,000-review IMDb audit as a separate
domain check using the saved INT8 and teacher outputs.

If the primary selection meets the one-point accuracy target on GLUE,
benchmark it alongside the FP32 confidence cascade and teacher-only using
the full 872 rows. Use the original AC, resident-session, batch-one, and
four-thread settings. Rotate process order across three paired blocks so
each method runs first once. Verify every timed prediction and call against
the frozen audit. Record paired median and p95 latency and peak working set.
The original deployment targets are a median latency reduction of at least
15% against the teacher, at most one accuracy point lost, and peak memory
below 1,536 MiB. Report accuracy and measured latency independently.
