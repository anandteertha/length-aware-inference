# Follow-up protocol: refitting by thread count and transformer graph optimization

This exploratory follow-up was planned after the original validation results were
known. It will not reuse GLUE validation for selection or claim another pristine
validation assessment. The original deployment policy and archived experiments
remain unchanged.

## Fixed design

- Use only the pinned SST-2 **training** split. Exclude the original 64 selection
  rows and the 128 confirmation rows. Shuffle with seed 2037, then allocate 256
  rows to fitting, 128 to confirmation, and 128 to an optimization audit. These
  partitions are mutually disjoint by training-row ID and allocated before fitting.
- The checkpoint was trained on SST-2 training data. The new audit is held out
  from **inference-policy tuning**, not from upstream classifier training; its
  accuracy is a diagnostic, not an independent model-generalization estimate.
- Compare two FP32 graph families: the existing exported graphs and graphs passed
  through ORT's standalone transformer optimizer. Apply the same additional
  optimizer recipe to dynamic and fixed 16/32/64/128 graphs. Use the BERT optimizer
  interface with the actual 12 heads and hidden size 768, CPU mode, raw attention
  masks and GELU approximation disabled. No quantization or FP16 conversion.
- Record graph/operator statistics and output/mask checks. If a graph is
  incompatible or fails verification, record and exclude it; do not silently
  substitute an unoptimized graph. A family without a verified dynamic fallback
  cannot participate in performance selection.

## Fitting separately at each thread count

For each available family and each of 1/2/4 intra-op threads, profile each verified
graph on fitting rows with two shuffled passes. Fixed graphs use every fitting
row that fits without truncation. Check every timed output against separately
computed PyTorch FP32 reference logits, outside the timer. Use the same
`atol=rtol=1e-4`, finite-score checks and exact labels throughout.

Learn exact-length routes separately for each family/thread setting, always
including dynamic execution. Enumerate zero/one/two fixed-variant subsets, prune
idle variants, and replay each remaining subset once on the 256 fitting rows in
a fresh process with all retained sessions resident. Select by measured p95,
subject to passing score/label checks and a 1,536 MiB process-memory cap. The
fitting passes are calibration, not independent statistical replications.

All measurements use batch one, CPUExecutionProvider, ORT_ENABLE_ALL, sequential
execution, one inter-op thread, idle spinning disabled, capacity probes and ten
warm-up requests. Profile and candidate orders are seeded and randomized across
families and thread counts. No timing workers run concurrently.

## Confirmation, freezing and audit

Freeze all fitted policies before opening confirmation for profiling. In three
randomized process blocks, compare dynamic-only and the fitted policy for every
available family at 1/2/4 threads on the 128 confirmation rows. Use matched row
orders within each block. Report run-level p95 medians/ranges and matched ratios,
not pooled-request significance tests or confidence intervals.

Select the global candidate by lowest median confirmation run p95, requiring
every run to pass scores/labels and its maximum measured process peak to fit the
cap. Break exact ties by memory, variant count and then the original graph family.
Freeze this choice before generating audit references or timing audit requests.

Audit dynamic-only and each family's fitted policy at the chosen thread count
in three fresh randomized blocks on the reserved 128 rows. Retain all outcomes
without changing the frozen choice. Unknown lengths use dynamic fallback.
Record accuracy, changed predictions, logits, latency components, resident memory,
startup components, size, runtime settings, graph hashes and source commits.

## Interpretation limits

This is a planned exploratory extension, not a replacement of the original
validation evidence. Three process blocks on one laptop do not establish
hardware-population effects. The optimizer might add no useful fusions, and the
per-thread selector may still prefer no variants. These are valid outcomes.

Optimizer recipe reference: [ORT transformer optimizer](https://onnxruntime.ai/docs/performance/transformers-optimization.html).
Sampling-unit and descriptive-reporting principles follow the
[previous statistical protocol](comparison-protocol.md#software-and-methodological-sources).
