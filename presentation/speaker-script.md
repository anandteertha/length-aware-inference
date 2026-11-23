# Speaker script — November 23, 2026

Use with [the guideline-aligned NC State deck](nc-state-etdl-project-guideline-aligned.pptx). Aim for a 9½- to 10-minute talk, leaving the scheduled three minutes for questions. The text below is spoken; slide changes occur between sections. Pause briefly after the main number on slides 8 and 9.

## Slide 1 — When is a smaller model enough?

Good afternoon. I’m Anandteertha Rao. My project asks a practical question: when can a smaller language model handle a request, and when is it worth calling a larger one? I study sentiment classification on a CPU, where I care about both prediction accuracy and request time. Let me start with the tradeoff I wanted to improve.

## Slide 2 — The accuracy–latency tradeoff

The task is to classify a movie review as positive or negative. My six-layer teacher gets 91.06% accuracy on the SST-2 validation set. A trained four-layer student is faster, but its accuracy falls to 87.73%. That gap is too large to use the student for every review.

My targets were no more than one point of observed accuracy loss, at least 15% lower median request time, and under 1.5 gigabytes of memory. Can I use the student for most reviews and call the teacher when it helps? First, I looked at existing methods.

## Slide 3 — Existing ways to reduce inference work

Prior work offers three ideas. First, fixed input shapes use a few supported text lengths. The runtime can optimize those shapes, but short reviews may need padding.

Second, knowledge distillation trains a smaller student using a teacher’s predictions. Third, adaptive methods spend more computation on difficult inputs. An early exit stops partway through a model; a cascade tries the student first, then calls the teacher if needed.

My question is whether text length and teacher cost improve a cascade beyond student confidence alone. Each existing approach leaves a gap.

## Slide 4 — Limitations of existing approaches

Fixed shapes may help ONNX, but padding can cancel the gain. A 17-token review sent to a 32-token model adds 15 padded positions. On this CPU, none of 11 tested lengths passed my preset screen for a fixed-size route, so I kept natural input lengths.

Distillation has a different limit: the student can lose accuracy. A confidence-only cascade can recover some of it, but confidence does not tell me a fallback’s cost. Longer reviews tend to cost more. These gaps led me to a student-first cascade, which brings its own challenges.

## Slide 5 — Why student-first routing is difficult

There are four challenges. First, the teacher can correct a student mistake, but can also turn a correct answer into a wrong one. A fallback is not automatically useful.

Second, a fallback runs both models; I have already paid for the student. Third, longer reviews make teacher calls more expensive. Fourth, a few fallbacks can make the slowest requests much slower. This is tail latency.

I therefore timed full requests, including tokenization, routing, and inference. I started by training a better student.

## Slide 6 — Train a smaller student

I copied the six-layer teacher’s embeddings, first four transformer layers, and classifier to initialize a student. Then I trained its weights on 8,192 labeled movie reviews. Copying layers gives it a starting point; training makes it useful.

The training loss has two equal parts: one compares the student with the correct label, and the other matches the teacher’s prediction. The second part is knowledge distillation. Temperature two softens the teacher’s probabilities, showing the student when the teacher is uncertain.

I trained for two epochs and chose the checkpoint using 1,024 separate examples. Next, I needed to decide when to trust its answer.

## Slide 7 — When should I call the teacher?

Every review goes through the student first. A fallback rule either keeps its answer or runs the teacher. Because it cannot see the correct label for a new review, it must estimate whether that extra time is worthwhile.

I compared two rules. The baseline uses student confidence. My proposed rule also uses text length and measured teacher cost. It estimates where the teacher’s likely improvement justifies that cost.

I used 2,048 examples to fit the proposed rule. On 1,024 separate examples, I chose the cutoff with the least teacher time while staying within one accuracy point of the teacher. I calibrated the confidence rule the same way. Then I froze both cutoffs and tested them on 872 validation examples. Before comparing the gates, I checked how much student training helped.

## Slide 8 — Training matters more than truncation

This chart compares the models on the same 872 validation reviews. Simply taking three teacher layers without updating them gave 73.97% accuracy. Training that three-layer model with the correct labels raised it to 84.29%: 90 more correct reviews. Distillation raised it slightly further, to 84.75%, which is only four additional correct reviews in this run.

The four-layer distilled student reached 87.73%. The full teacher remained ahead at 91.06%. My takeaway is that training made a large difference compared with plain truncation, while this particular distillation step gave a small gain. Adding a fourth layer helped further, but the student still needed selective teacher help. That leads to the main end-to-end result.

## Slide 9 — Confidence routing meets the observed target

Here, both models were loaded for every policy, and I timed complete requests on the CPU. The teacher alone got 91.06% accuracy. The student alone was faster, but its 87.73% accuracy missed my target.

My proposed length-and-cost gate called the teacher on 121 of 872 reviews. It cut median time by 23.2% in the paired timing runs, but its accuracy was 89.33%. That is 1.72 percentage points below the teacher, so it missed the one-point accuracy target.

The simpler confidence gate called the teacher on 161 reviews. It reached 90.48% accuracy, just 0.57 points below the teacher, with a 24.1% paired median time reduction. Resident memory stayed below my 1.5-gigabyte limit. So the confidence gate met my *observed* targets on this CPU; the proposed length-and-cost gate did not. The slowest-request measurements were mixed, and this one validation set cannot prove the accuracy target will hold everywhere. Those limits point to my next experiments.

## Slide 10 — What I would test next

First, I plan to compare INT8 ONNX with the current model. Quantization may reduce CPU work and memory, but I need to measure its effect on accuracy and full request time. I plan to add those results to the final report.

Second, I would choose the fallback cutoff on new labeled reviews. The current calibration used examples the teacher had seen during training, which may make fallbacks look more helpful than they are on new data.

Third, I would repeat the study on longer reviews and another CPU, especially the slowest requests. Let me close with what I learned.

## Slide 11 — What I learned

On this CPU, a trained student with a confidence fallback kept accuracy close to the teacher and reduced median request time by 24.1%. The length-and-cost rule routed fewer reviews to the teacher, but missed my accuracy target. I used one training seed and one laptop, so more runs are needed.

My contribution is an end-to-end CPU comparison of fixed shapes, student training, and fallback rules. Thank you. I’m happy to take questions.
