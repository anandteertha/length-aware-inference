# Pre-timing OpenVINO readback amendment

The first execution completed one transformer-ORT block, then stopped before OpenVINO timing. The OpenVINO configuration explicitly requested `ov.Type.f32`, but the readback guard accepted only the text `f32`; OpenVINO 2026.4.0 displayed the type as `float32`. The guard was changed to accept both equivalent spellings. The [original plan](../results/real31-preflight/plan.json) and the completed [ORT preflight block](../results/real31-preflight/ort_transformer-block-0.json) are preserved. That block is excluded from the final analysis.

The same 64 IDs, model hashes, tolerance, process-block schedule, and measurement method remain in place. The plan's code hash was updated, and all ten final backend blocks restart under the amended plan. This is a functional compatibility repair made before any OpenVINO timing or final comparison; the amendment does not alter the prespecified confirmation criterion.
