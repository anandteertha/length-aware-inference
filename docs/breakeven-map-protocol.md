# Real-text length break-even map: frozen design

This is a descriptive, exploratory map of when a fixed sequence shape pays for its padding on the current CPU. It extends the [fresh 31-token test](real31-results.md) without reusing its 64 rows. The map can locate promising length regions for a later independent confirmation; it cannot itself choose a deployment policy.

## Population, estimand, and sampling unit

Use the pinned SST-2 training split and tokenizer, without truncation. Exclude the 704 training IDs used in earlier calibration, confirmation, and follow-up work, plus the 64 IDs in the previous real-31 test. The released classifier was trained on SST-2 train, so this is an inference-timing experiment, not a classifier-generalization test.

Prespecified exact token lengths (including special tokens) are **9, 15, 16, 17, 24, 30, 31, 32, 33, 40, and 48**. These cover the 16/32/64 buckets, near-boundary lengths, and interior lengths. Use 24 unused, real sentences at each length, selected by seed 20261112 before references or timing. The previous train split has too few unused exact-length sentences at 63 (5), 64 (1), and 65 (2) for the same sampling rule, so this experiment makes no claim near the 64-to-128 boundary. Do not fill gaps with concatenated or synthetic text.

For each length, the primary outcome is *batch-one prepared-input p95* (padding plus inference, excluding tokenization) for three paths in the same backend/process: natural-length dynamic, dynamic padded to the smallest allowed bucket, and fixed graph at that bucket with the identical padded input. The independent timing unit is a fresh process block, not an individual repeated request. Within each block and length, define:

- **Padding cost** = padded-dynamic p95 − natural-dynamic p95.
- **Shape gain** = padded-dynamic p95 − padded-fixed p95.
- **Net fixed gain** = natural-dynamic p95 − padded-fixed p95 = shape gain − padding cost.

These are differences of measured p95s; their identity is arithmetic, not an assumption that request-level tail latencies add. A positive net gain favors fixed. Report each of three process blocks and the median and full range of their paired differences. Do not interpolate a crossover between sampled lengths or call a noisy sign change a precise break-even threshold. A candidate region for *future confirmation* requires positive net fixed gain in all three blocks and at least 5% median of the within-block fractional reductions versus natural dynamic, with passing output/power checks and every two-session process below the existing 1,536 MiB sampled-RSS cap. Multiple lengths are screened; this rule does not authorize deployment.

## Graphs and timing

Use the **same standard FP32 ONNX graph pair** in ORT and OpenVINO for each bucket. Graph weights, input shapes, and provenance must match the existing verified exports. Both dynamic and fixed sessions remain resident for a capacity-specific process. ORT uses CPUExecutionProvider, `ORT_ENABLE_ALL`, four intra-op threads, sequential execution, and disabled spinning. OpenVINO 2026.4.0 uses `CPU` only, explicit FP32 inference precision, four inference threads, latency mode, and one stream, with readback checks. Each backend/capacity has three fresh process blocks (18 final processes); order is seeded independently within each block.

Pretokenize all text. Before timing, run three untimed warm-ups for each length/path, including the backend's dynamic shape handling. Then run each of 24 rows twice per path, shuffling the jobs independently for each repeat and block. Time inference only and padding-plus-inference separately; exclude tokenization, model compilation, and process startup. Use padding-plus-inference for the primary map because a deployed fixed route must perform both. Record compilation time and sampled process RSS. Record AC connection and active power scheme before and after, and one-second system/worker CPU observations; flag power mismatch or telemetry gaps above ten seconds. Observe CPU scheduling variability but do not tune thread/core settings based on these results. Retain every run and report absolute p95 ranges; if process instability is large, do not present the map as a reliable deployment prediction.

Every output, including warm-ups, must be finite, within `atol=rtol=1e-4` of independently generated PyTorch eager FP32 logits, and have exactly the same predicted label. A numerical failure stops the run and is preserved. Record model, code, workload, reference, and process hashes. The existing dynamic-only four-thread ORT deployment is unchanged.

## Why this helps

The map separates an optimization that works *at the same padded shape* from an optimization that helps *a natural message*. If the shape gain consistently exceeds padding cost for a range of real message lengths, a later held-out study can test a narrowly targeted router and its memory/startup costs. If the net gain remains zero or negative, the project can explain why static specialization fails on this workload and stop accumulating variants. A single positive length after screening eleven lengths is a lead, not a confirmed win.

OpenVINO notes that [static shapes can help when the input shape is stable](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html), while its [dynamic-shapes guide](https://docs.openvino.ai/2025/openvino-workflow/running-inference/model-input-output/dynamic-shapes.html) recommends benchmarking against real application data. We therefore measure both the equal-shape execution effect and the natural-input net effect.

The process-level sampling and evidence-scope review used the scientific-critical-thinking procedure from Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026), [*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*](https://doi.org/10.48550/arXiv.2609.00065). The procedure is methodological guidance; it is not empirical support for any performance result.
