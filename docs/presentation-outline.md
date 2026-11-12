# November 23 presentation outline (10 minutes + 3 minutes Q&A)

**Status:** structure only. Anandteertha's November 23 slot is 0:39-0:52 from class start, with Xiangyu Wen and Charlie Austin assigned as questioners. Replace every result placeholder with an archived measurement before presenting. The course requires about 8-10 content slides, plus title and closing slides.

| Slide | Time | Required section | Core visual or claim |
| --- | --- | --- | --- |
| Title | 0:20 | — | Research question and repository link. |
| 1 | 0:50 | Problem | CPU sentiment classification; accuracy and request latency both matter. |
| 2 | 0:55 | Existing approaches | Fixed shapes, distillation, early exits, and cascades, with citations. |
| 3 | 0:50 | Limitations | Our real-text break-even map: no robust fixed-route candidate. Show one compact negative-result plot. |
| 4 | 0:50 | Challenges | Student accuracy loss, student-first recomputation, length-dependent cost, and tail latency. |
| 5 | 1:20 | Method | Three-layer student initialization and supervised versus teacher-distillation loss. |
| 6 | 1:20 | Method | Gate sees student scores and length; route only when expected correction offsets teacher cost. |
| 7 | 1:15 | Results/analysis | Training curves and matched student ablation: accuracy, model size, training cost. |
| 8 | 1:25 | Results/analysis | Accuracy-versus-measured-latency Pareto plot; fallback rate and key length strata. |
| Closing | 0:25 | — | Supported conclusion, limitation, and only specific unfinished report work. |

This totals 9:30, leaving 30 seconds for transitions. If a planned result is unfinished, show its actual current status and name only the experiment that will be completed for the November 30 report. Never fill a result slide with a projected gain.

## Q&A preparation

The course allocates 20% of the project grade to answers during the presentation. Prepare short evidence-backed answers to:

1. What is original in this study, given existing distillation and cascade papers?
2. Does length add predictive value beyond student confidence? Which ablation shows it?
3. Did the teacher see gate-training examples during its original pretraining/fine-tuning, and how does that affect generalization?
4. Why measure both median and p95 if teacher fallback is rare?
5. Can student-first plus teacher rerun actually beat a single strong ONNX teacher? Show end-to-end timing, not just layer counts.
6. Why did the first fixed-shape hypothesis fail, and how did that finding motivate this method?

The other graded question is for Jai Pavan Ramani on November 16. Base the question on a specific assumption, comparison, or measurement in that student's presentation rather than a generic prompt.
