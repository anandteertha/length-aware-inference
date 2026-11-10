# Fresh real 31-token confirmation and OpenVINO CPU diagnostic

The [protocol](real31-protocol.md) fixed the exact length, sample, graph pairs, settings, five process blocks, and primary decision rule before timing. A [pre-timing compatibility amendment](real31-amendment.md) repaired a precision-readback spelling guard; the completed preflight ORT block was preserved and excluded. All numbers below come from the [amended run archive](../results/real31/summary.json), with raw [process records](../results/real31/) and [checksums](../results/real31/checksums.json).

## Inputs and checks

The pinned SST-2 training split contained 486 previously unused examples with exactly 31 tokenizer tokens, after excluding 704 IDs from earlier policy work. A seeded 64-example sample was fixed before outputs or timing. The same text, token IDs, and attention masks were used for both backends. The released classifier had been trained on SST-2 train, so this confirms an *inference timing stratum* only, not accuracy generalization. The original validation set was not revisited.

Each backend completed five fresh processes, 576 timed requests per process (64 rows × 3 paths × 3 repeats), for 2,880 timed requests per backend. All 5,760 timed outputs passed the PyTorch FP32 `atol=rtol=1e-4` comparison with exact label agreement; there were zero changed labels. Maximum absolute logit errors were 7.67e-6 for transformer ORT and 3.34e-6 for OpenVINO. AC connection and the Windows power-scheme snapshots matched the plan throughout, and no one-second CPU telemetry gap exceeded 1.02 seconds. These checks do not control temperature, clock speed, affinity, or background work.

## Targeted transformer-ORT confirmation

The primary contrast compared the previously investigated transformer-optimized dynamic graph at natural length 31 against its fixed-32 graph with one masked pad token. Both sessions were resident in each process. Four threads, CPUExecutionProvider, FP32 and disabled spinning were held fixed.

| Path | Median process p95 (ms) | Process p95 range (ms) |
| --- | ---: | ---: |
| Dynamic, natural 31 | 19.535 | 19.414–20.818 |
| Dynamic, padded 32 | 19.317 | 18.683–19.795 |
| Fixed-32, padded 32 | 19.978 | 19.181–20.958 |

The five matched natural-dynamic/fixed p95 ratios were **0.963, 1.005, 1.013, 0.978, 1.007** (median **1.005**; values above one favor fixed). The prespecified criterion required all five ratios above one and median above 1.05. **It was not met.** The earlier synthetic 31-token advantage therefore did not independently reproduce on this fresh real-text sample under the new run conditions. This does not prove that static is always slower; it narrows what our earlier post hoc signal supports.

The dynamic-padded/fixed ratios were below one in all five processes (median 0.974). With identical 32-token inputs, this ORT graph pair showed no fixed-shape execution benefit in this run. Dynamic padding to 32 happened to have lower p95 than natural 31 in four of five blocks; the small direction is not a general rule about padding and differs from the larger costs seen above bucket boundaries in the synthetic diagnostic.

## OpenVINO CPU, standard graph pair

This exploratory comparison loaded the original standard ONNX dynamic and fixed-32 graphs into OpenVINO 2026.4.0. Both compiled models reported FP32 inference precision, four threads, one stream, and CPU-only execution. It is a backend/shape test on the same 64 texts, **not** an ORT-versus-OpenVINO speed race, because the graph families and runtime behavior differ.

| Path | Median process p95 (ms) | Process p95 range (ms) |
| --- | ---: | ---: |
| Dynamic, natural 31 | 43.642 | 30.657–89.532 |
| Dynamic, padded 32 | 48.254 | 31.577–91.241 |
| Fixed-32, padded 32 | 44.775 | 29.418–87.423 |

For **identical padded 32-token inputs**, dynamic-padded/fixed p95 ratios favored fixed in all five blocks: **1.100, 1.044, 1.073, 1.078, 1.057** (median **1.073**). This is a real, matched observation consistent with an OpenVINO shape-specialization benefit for this graph and CPU. It does not identify which internal operation caused it.

For the **deployable choice**, natural-dynamic/fixed ratios were **1.042, 1.024, 1.061, 0.975, 1.005** (median **1.024**). One block reversed direction, and the net median advantage is small. The added pad token absorbs much of the fixed-shape execution advantage. Absolute OpenVINO p95 values varied substantially across process blocks even though power snapshots matched; the five blocks do not establish a stable production latency estimate. We did not fit or evaluate a new OpenVINO routing policy.

The median sampled peak process RSS with **both sessions resident** was 891.9 MiB for transformer ORT and 1,202.7 MiB for OpenVINO, below the project's 1,536 MiB cap in these runs. This is not a single-model memory comparison and excludes model acquisition and dataset storage. Median measured initialization per session was 383.0/428.9 ms for ORT dynamic/fixed and 586.7/872.2 ms for OpenVINO dynamic/fixed, with warm OS caches possible. Inference timings excluded initialization and tokenization.

## Interpretation

The project has a concrete positive *mechanism* result: on this CPU, OpenVINO's static 32-token model was consistently faster than its dynamic model when both did the same padded work. It also has a concrete limitation: the advantage did not consistently survive comparison with natural-length dynamic execution, and the earlier transformer-ORT 31-token signal did not confirm. This is a precise backend-specific observation, not a novel static-bucketing method or a broad deployment win. The original frozen dynamic-only ORT policy remains the supported deployment decision for the measured project setup.

Reproduction after the baseline and follow-up graphs exist:

```powershell
uv sync --locked --extra openvino
uv run python tools/real31.py prepare
uv run python tools/real31.py references
uv run python tools/real31.py run
```

The frozen [plan](../results/real31/plan.json), selected [workload](../results/real31/workload.json), [references](../results/real31/references.json), process records, and summary are archived. The outputs are descriptive process replications; requests within a process are dependent, so there is no request-level confidence interval or significance claim. Human verification and approval of this report are pending.
