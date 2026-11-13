# November 23 presentation outline (10 minutes + 3 minutes Q&A)

**Status:** structure with verified results. Anandteertha's November 23 slot is 0:39-0:52 from class start, with Xiangyu Wen and Charlie Austin assigned as questioners. Keep every number tied to the [student results](student-training-results.md), [gate results](gate-results.md), or [cost-weighted extension](cost-weighted-extension-results.md). The course requires about 8-10 content slides, plus title and closing slides.

| Slide | Time | Required section | Core visual or claim |
| --- | --- | --- | --- |
| Title | 0:15 | — | Research question and repository link. |
| 1 | 0:40 | Problem | CPU sentiment classification; accuracy and request latency both matter. |
| 2 | 0:50 | Existing approaches | Fixed shapes, distillation, early exits, and cascades, with citations. |
| 3 | 0:45 | Limitations | Our real-text break-even map: no robust fixed-route candidate. Show one compact negative-result plot. |
| 4 | 0:45 | Challenges | Student accuracy loss, student-first recomputation, length-dependent cost, and tail latency. |
| 5 | 1:15 | Method | Three-layer student initialization and supervised versus teacher-distillation loss. |
| 6 | 1:15 | Method | Gate sees student scores and length; route only when expected correction offsets teacher cost. |
| 7 | 1:10 | Results/analysis | Training ablation: truncated 73.97%, supervised 84.29%, distilled 84.75%, four-layer 87.73%, teacher 91.06%; identify the small four-example distillation difference. |
| 8 | 1:20 | Results/analysis | [Accuracy-latency figure](../results/student-study/accuracy-latency.svg): confidence gate 90.48% with 24.1% paired-median reduction; length-cost gate 89.33% and misses the one-point accuracy target. |
| 9 | 0:55 | Results/analysis | [External IMDb extension](cost-weighted-extension-results.md): cost-weighted distillation 832/1,000 versus uniform 843/1,000; label the reused SST-2 gain exploratory. |
| Closing | 0:20 | — | The controlled cascade result and limitations: p95 mixed, three CPU blocks, no general algorithm novelty claim. |

This totals 9:30, leaving 30 seconds for transitions. The extension is complete, so show its unfavorable result directly. The confidence result meets the *observed* one-point accuracy, 15% median speed, and 1,536 MiB memory thresholds; its paired accuracy interval is too wide to establish population noninferiority within one point. The weighted extension has no measured ONNX timing.

## Q&A preparation

The course allocates 20% of the project grade to answers during the presentation. The [Q&A sheet](presentation-qa.md) gives draft answers with exact counts and caveats. Prepare short answers to:

1. What is original in this study, given existing distillation and cascade papers?
2. Does length add predictive value beyond student confidence? Which ablation shows it?
3. Did the teacher see gate-training examples during its original pretraining/fine-tuning, and how does that affect generalization?
4. Why measure both median and p95 if teacher fallback is rare?
5. Can student-first plus teacher rerun actually beat a single strong ONNX teacher? Show end-to-end timing, not just layer counts.
6. Why did the first fixed-shape hypothesis fail, and how did that finding motivate this method?
7. Why did cost-weighted distillation gain five SST-2 validation examples but lose eleven on IMDb, and can either difference be called reliable?

The other graded question is for Jai Pavan Ramani on November 16. Base the question on a specific assumption, comparison, or measurement in that student's presentation rather than a generic prompt.
