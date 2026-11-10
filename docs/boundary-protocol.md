# Matched length-boundary diagnostic

This diagnostic follows the completed experiments. It explains execution costs
without tuning a deployment policy or reopening GLUE validation. The original
frozen policy is hashed before execution and must remain unchanged.

## Fixed design

Use both previously verified FP32 graph families, four intra-op threads, one
inter-op thread, CPUExecutionProvider, ORT_ENABLE_ALL, sequential execution, and
disabled idle spinning. Retain dynamic and one fixed graph in each fresh process.
Do not interpret this process memory as dynamic-only deployment memory.

Test natural lengths 9, 15/16/17, 31/32/33, 63/64/65, and 127/128/129, including
special tokens. Three deterministic token patterns (positive, negative, mixed)
generate 39 inputs. These are synthetic pretokenized execution probes, without
task labels or classifier-accuracy claims. Freeze their exact token IDs and masks.

For each input up to 128 tokens, compare three paths:

1. Dynamic graph with natural, unpadded inputs.
2. The same dynamic graph padded to the smallest fitting 16/32/64/128 bucket.
3. That bucket's fixed graph with identical padded token IDs and attention masks.

Length 129 uses dynamic fallback only. No truncation is allowed. Generate separate
PyTorch FP32 reference logits before timing, in another process. Check every timed
output after the timer using finite scores, `atol=rtol=1e-4`, and identical labels.
Numerical failures are retained and reported; do not relax tolerances.

Each of five blocks runs all eight family/capacity combinations in randomized
order, one fresh process at a time, with a fixed two-second rest before launching
each process. This rest does not guarantee equal CPU temperatures.
Each process warms every input/path twice,
then runs eight repetitions per pattern: 24 observations per length/path/process.
Seed 2041 fixes job and request ordering. Match request ordering across families
at the same capacity/block, and randomize input/path order within repetitions.
Dynamic sessions receive a 512-token capacity probe; fixed sessions receive their
full-capacity probe, both outside timing but inside the memory observation.

## Timing controls and interpretation

Record AC connection and the Windows power-scheme GUID before the experiment.
Require matching conditions before each process; record them again afterwards
and report any change. Do not alter system settings automatically. Record system
CPU use before/after each process and system/worker CPU use at one-second intervals
during session initialization, probes, warm-up, and requests. System load includes
the worker; it is not a background-load estimate. Temperature, watts, clock speed,
and CPU affinity are not controlled or measured. Preserve all completed results;
do not drop inconvenient runs based on their observed timing or load.

Report routing/padding time, inference time, and their sum (prepared-input time).
Tokenization is excluded because token IDs are fixed before timing. This sum is
not text-to-label end-to-end latency. The bucket lookup is a diagnostic operation,
not the previously learned production routing policy.

For every length/path, show the five process-run p95 values, median, and full
range. Report within-process p95 ratios: padded/natural for padding cost,
padded/fixed for fixed-shape execution, and natural/fixed for net effect. A ratio
above one favors the denominator. These are descriptive process replications;
do not pool requests as independent replicates, perform significance tests, or
claim confidence intervals. No new policy is chosen from these results.

Interpret this as an exploratory mechanism diagnostic on one CPU. It does not
estimate natural message-frequency-weighted p95 or establish causation by an
individual fused operator. It also cannot repair uncontrolled conditions in the
earlier experiments. Statistical reporting principles and sources are recorded
in the [existing protocol](comparison-protocol.md#software-and-methodological-sources).
