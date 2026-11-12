# November 23 Q&A sheet

Use these as evidence-backed speaking notes, not memorized scripts. The [student results](student-training-results.md), [gate results](gate-results.md), and [figure](../results/student-study/accuracy-latency.png) contain the exact denominators and timing blocks.

## Likely questions and short answers

**What is novel?** Knowledge distillation, early exits, and cascades are established ([DeeBERT](https://aclanthology.org/2020.acl-main.204/), [CascadeBERT](https://aclanthology.org/2021.findings-emnlp.43/)). Our course contribution is the controlled CPU study motivated by a failed fixed-shape hypothesis: we trained compact students, tested a length-and-cost teacher-benefit gate against confidence-only routing, and measured both models resident. The length-and-cost gate did **not** beat the simpler baseline on the accuracy target. Do not claim a new general-purpose cascade algorithm.

**Where is the actual training?** We copied teacher embeddings, three or four transformer layers, and classifier weights into student models, then updated their weights with AdamW for two epochs on 8,192 labeled SST-2 sentences. The three-layer truncation without student updates got 645/872 correct; supervised training got 735/872, and distillation got 739/872. That large difference demonstrates learned adaptation rather than inference-only graph changes.

**Did distillation help?** It added only four correct validation examples over supervised training under one seed and fixed budget: 84.75% versus 84.29%. That is suggestive, not a reliable distillation gain. Adding a fourth layer helped more: the four-layer distilled student reached 87.73%.

**Why train the four-layer model?** Before final validation, we set a rule: if the best three-layer development accuracy remained more than three points below the teacher on the development partition, train one four-layer capacity control. The gap was 4.69 points, so we followed that rule. This is a bounded ablation rather than a post hoc depth search.

**How does the gate work?** It sees the student's two logits and token length. A ridge score fitted on 2,048 training-derived gate examples predicts the teacher's incremental correctness, using logit margin, log2 length, and their interaction. We measured teacher CPU cost by length and divided the joint benefit score by that cost. Thresholds and one operating point were fixed on a separate 1,024-row calibration partition before validation.

**Did the length-aware idea succeed?** The selected length-and-cost gate improved the four-layer student from 765 to 779 correct while routing 121/872 requests to the teacher. The full teacher got 794 correct, so the gate missed the preset one-point accuracy budget. A simpler confidence-only calibration control routed 161 requests and got 789 correct. Thus the added length-cost score did not improve the target tradeoff in this test.

**Why did length-and-cost routing miss?** The teacher had more net accuracy advantage on longer texts, but fallback becomes more expensive with length. The selected gate routed only 13/208 long examples and 33/197 short examples; it lost three correct predictions in the shortest stratum. The teacher had seen the SST-2 training split used to fit the gate, so the training-derived benefit labels were also optimistic for unseen examples. These are evidence-backed explanations, not a causal proof from one dataset.

**Is the confidence cascade better than the teacher?** On this validation workload, it got 789/872 correct versus 794/872 for the teacher, a 0.57-point accuracy loss, while its end-to-end median time fell by 10.1%, 24.1%, and 30.3% in the three paired resident blocks. Observed memory stayed near 852 MiB, below the 1,536 MiB cap. It met the project's *observed* targets. A paired bootstrap 95% interval for its accuracy difference is −1.61 to +0.46 points, so these data do not prove population accuracy loss is under one point.

**What happened to p95 latency?** About 18.5% of confidence-gate requests run both student and teacher. Those fallback requests enter the slow tail. The confidence gate's p95 was worse than teacher-only in one block and better in two; with three blocks on one laptop we do not claim a consistent p95 win.

**How were timings made fair?** Four ORT threads per session, natural validation text, 32 warm-up requests, three shuffled process blocks, and both student and teacher sessions resident for *every* policy in the cascade comparison. Request time includes tokenization, inference, gate work, and teacher fallback, but excludes model startup. All 872 ONNX predictions and routing choices matched the frozen-logit audit in every block. Process conditions still shifted, so paired block reductions and their range matter more than a single median.

**Why did the original fixed-shape approach fail?** Fixed shapes can reduce some graph overhead, but routing shorter text into a fixed capacity pads the sequence and adds computation. Across the broad workload and the real-text break-even map, the net effect did not support a reliable fixed route. That motivated changing *model depth* according to predicted need instead of changing only input shape.

**What would you test next?** The planned extension is cost-weighted distillation: train the student to match the teacher more strongly where fallback is expensive, then compare it with uniform distillation and the confidence cascade under a separately frozen evaluation. A shared multi-exit model could avoid recomputing the first layers, but it is established prior work and requires a distinct experiment. We should report any extension as exploratory if it is designed after these validation results.

## Three-minute Q&A practice

Practice explaining, without notes: (1) what weights were trained, (2) which comparison supports the speed claim, and (3) why the length-cost method did not beat confidence-only routing. Use the exact denominators 872 validation sentences, 121 selected-gate fallbacks, and 161 confidence-gate fallbacks. Answer the asked question first, then give one supporting number and one limitation.
