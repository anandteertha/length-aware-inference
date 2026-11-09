# Comparison protocol, specified before the new measurements

The existing 16+32-token policy was selected on 64 training examples. Its model
set and exact-length routes remain fixed for the following confirmation study.
This is an exploratory systems experiment on one development laptop.

## Training confirmation

- Select 128 additional SST-2 training rows, disjoint from the original 64. Use
  the original pinned dataset revision, deterministic seed 2028, and no truncation.
- Compare PyTorch FP32 eager, optimized dynamic ONNX, fixed-16 wherever it fits
  with dynamic fallback, smallest-fitting 16/32/64/128 buckets with dynamic
  fallback, and the previously selected policy. Fixed-16 is specified using the
  earlier calibration results; the new sample will not choose this baseline.
- Use batch one, the same tokenizer, explicit per-session CPU threads 1/2/4,
  sequential ORT execution and one inter-op thread. Disable idle ORT thread
  spinning consistently for these new experiments; do not compare their absolute
  times directly with the earlier study's runtime defaults.
- Three randomized blocks. Each block includes all method/thread combinations in
  shuffled order. Each combination runs in a new process, with the same shuffled
  input order within a block, capacity probes and 10 warm-up requests. No models
  run concurrently. These are process repetitions on one machine, not independent
  hardware replications.
- Precompute PyTorch logits in a separate reference process. Check every timed
  result against these scores and labels after its timer stops. Keep reference
  allocations out of benchmark processes' memory measurements.
- Record tokenization, routing/padding, inference and combined latency; per-run
  accuracy, changed labels, logit checks, initialization/readiness, peak RSS,
  serialized size, environment, method order and raw request measurements.

## Estimation and deployment choice

The primary descriptive endpoint is each process run's combined p95. Report the
median and full range of the three run-level p95 values. Report matched-block
dynamic/selected p95 ratios at each thread count. Do not pool thousands of
requests as independent replicates, remove outliers, shop for tests, or claim
statistical significance. Three blocks provide a limited description of runtime
variation, not a precise population confidence interval.

Choose deployment threads and whether to retain the selected variants using only
training confirmation: lowest median run-level p95 among dynamic and selected
policy candidates at 1/2/4 threads, requiring every run to pass score/label checks
and its measured peak process memory to fit 1,536 MiB. Break exact ties by memory
and then variant count. PyTorch and the simple baselines remain comparisons,
not additional deployment candidates. Preserve and hash the frozen plan before
accessing validation. No further policy selection uses validation outcomes.

## Held-out evaluation

After freezing, evaluate all five methods at the chosen deployment thread count
on all 872 validation examples from the pinned dataset revision. One randomized
method order and one fresh process per method; report this as one held-out timing
run per method, without latency confidence intervals. Accuracy and changed labels
use all 872 examples. Keep failed correctness checks in the report and do not
retune a failing policy using validation. Full application cold start, other
hardware and additional held-out timing replications remain separate experiments.

## Software and methodological sources

The statistical-analysis skill informed the sampling-unit and descriptive-reporting
choices. Its reference was verified against the current arXiv record:
Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*.
[doi:10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065).

Runtime controls follow [ORT thread-management documentation](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).
