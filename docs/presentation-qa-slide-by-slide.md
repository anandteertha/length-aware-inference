# Slide-by-slide presentation Q&A

Use this with the [15-slide Google Slides deck](https://docs.google.com/presentation/d/1KUcHNqkZq0ZBhR7YN_XlJWL15THyl5k6vbDh8RF2TCM/edit). These are possible questions from someone seeing a slide without the speaker script. Answer the question first; add a number or caveat only when it helps. Slides 13–15 are optional Q&A appendices.

## Three answers to know especially well

**Where did the teacher come from, and what data trained it?** I downloaded the pretrained and SST-2-fine-tuned six-layer DistilBERT sentiment checkpoint from [Hugging Face](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) and pinned revision `714eb0fa89d2f80546fda750413ed43d93601a13`. I did not train that teacher. Its [base model](https://huggingface.co/distilbert/distilbert-base-uncased) was pretrained on BookCorpus and English Wikipedia; the published sentiment checkpoint was then fine-tuned on SST-2. My project uses the pinned [GLUE SST-2 dataset](https://huggingface.co/datasets/nyu-mll/glue), with separate train and validation splits. The exact model and dataset revisions are in the [manifest](../results/student-study/manifest.json).

**What is the calibration limitation?** The teacher's publisher fine-tuned it on the SST-2 training split. I took the student-training, gate-fitting, and gate-calibration rows from that same split. Those partitions are disjoint for *my* student and gate work, but they are not unseen by the teacher. Its success on gate-fit and calibration rows can overstate how helpful fallback will be on new reviews. The 872 SST-2 validation rows were not used to choose the gate cutoff, so this is **not validation-label leakage into cutoff selection**. It is teacher-training overlap with gate fit and calibration, plus only one final validation set. A stronger study would calibrate on labeled reviews the teacher never trained on and evaluate on another untouched set.

**Did my proposed method meet its target?** No. On 872 validation reviews, the length-and-cost gate got 89.33%, 1.72 points below the 91.06% teacher, while the preset accuracy budget was one point. The simpler confidence gate got 90.48%, 0.57 points below the teacher, with a 24.1% paired median request-time reduction and memory below the cap. That meets the *observed* target on this laptop, not a proven population guarantee. [Measurements](gate-results.md)

## Slide 1 — Research question

**What exactly is a “request”?** One SST-2 sentence is tokenized and classified as positive or negative. The timing includes tokenization, routing, and the model inference needed for that request; it excludes loading the models at startup.

**Why study CPU rather than GPU?** This project tests the practical tradeoff on one available laptop CPU. The result is specific to that hardware and runtime configuration; I do not claim the same timings on a GPU or another CPU.

## Slide 2 — Accuracy and latency targets

**Where do 91.06% and 87.73% come from?** The teacher got 794 of 872 SST-2 validation reviews correct; the trained four-layer student got 765. Both used the same validation rows. [Student study](student-training-results.md)

**Who set the one-point, 15%, and memory targets?** They are my preset project criteria for an acceptable accuracy-versus-efficiency tradeoff. Passing them on one validation run is an observed result, not a universal guarantee.

## Slide 3 — Existing approaches

**Is a student-teacher cascade new?** No. Distillation and cascades existed before this project. I tested whether adding length and measured teacher cost helped a cascade on this CPU, then compared it with a confidence-only control.

**How is an early exit different from my cascade?** An early exit stops within one model. My cascade runs a separate student first and, on selected requests, runs a separate teacher too. A fallback therefore pays for both models.

## Slide 4 — Limitations of existing approaches

**Why can a fixed-shape model be slower?** A review between supported lengths must be padded up to the next shape. The runtime may optimize that fixed shape, but the extra padded positions, routing, and resident variants can outweigh the benefit. My fixed-shape screen found no robust route among 11 tested lengths on this CPU. [Fixed-shape study](breakeven-map-results.md)

**Does confidence tell you nothing about fallback cost?** It tells me how decisive the student is, but its routing score does not account for teacher execution time changing with length. I tested whether adding that cost helped; in this validation, it did not beat confidence-only routing.

## Slide 5 — Routing challenges

**Why can calling the teacher reduce accuracy?** The teacher is better on average, but it can still be wrong when the student is right. On selected validation fallbacks, my proposed gate had 24 fixes and 10 harms, for a net gain of 14 correct answers. [Gate audit](gate-results.md)

**Why might the slowest requests get worse even if the median improves?** Most requests stop after the student; fallback requests run both student and teacher. Those extra-long requests can dominate p95 latency. The p95 comparison was mixed across three timing blocks, so I do not claim a reliable tail-latency win.

## Slide 6 — Student training

**Did you train the teacher or just cut off layers?** I used an existing teacher. I copied its embeddings, first three or four layers, and classifier into smaller students, then updated student weights using labeled SST-2 training rows. The no-update three-layer model got 73.97%; supervised training raised it to 84.29%. [Training study](student-training-results.md)

**What does temperature 2 do in distillation?** Dividing teacher and student logits by 2 before softmax produces softer class probabilities. The teacher-matching loss can then convey relative preference between positive and negative, alongside the ordinary true-label loss. Temperature 2 was a fixed training choice, not tuned on validation.

**Why a four-layer student?** A prespecified development-set rule triggered it when the best three-layer student remained more than three accuracy points behind the teacher. The observed development gap was 4.69 points.

## Slide 7 — Gate fitting and calibration

**How can you know whether the teacher helps before running it?** I cannot know for an individual new review. On 2,048 labeled gate-fit reviews, I compare both models with the true label, then fit a ridge regression to estimate the *net* benefit from the student's margin and the input length. On a new review, the gate sees only those student-side features. [Gate code](../scripts/fit_gate.py)

**How is the cutoff chosen?** On 1,024 separate calibration reviews, I evaluate preset routing thresholds. I keep candidates within one accuracy point of teacher accuracy **on calibration**, then choose the one with the smallest estimated added teacher time. I freeze it before the 872-review validation. The final 15% request-time goal is checked later, not used as a per-request trigger.

**Is that data leakage?** There is no use of validation labels to fit the ridge model or choose the cutoff. There *is* overlap between the released teacher's earlier SST-2 training data and my gate-fit/calibration rows. That can make estimated fallback benefit optimistic. I disclose this limitation rather than claim those rows were unseen by every model.

## Slide 8 — Training ablation

**Did distillation itself make a large difference?** No. The three-layer distilled model got four more of 872 examples correct than the supervised model: 84.75% versus 84.29%. One seed and four examples are not enough to claim a reliable distillation gain. The large improvement was from training compared with plain truncation.

**Can you attribute the four-layer result only to depth?** Not cleanly. The four-layer result is another distilled training run, and I did not train a four-layer supervised control under this protocol. I can say the tested four-layer distilled checkpoint outperformed the tested three-layer distilled checkpoint, not isolate depth as the sole cause.

## Slide 9 — End-to-end results

**What does “confidence-only” actually do?** It scores a review from the student's absolute positive-versus-negative logit margin. Smaller margins give higher predicted teacher benefit, so sufficiently small-margin requests call the teacher. The confidence score does not use length or teacher cost; its cutoff was selected on calibration. [Implementation](../scripts/fit_gate.py)

**Why did the confidence gate win when the proposed gate had more information?** The proposed gate divided predicted benefit by higher teacher cost for long reviews, so it routed only 13 of 208 long validation reviews. The teacher had a larger accuracy advantage in that length group. Also, its benefit model was fitted on rows the teacher had seen during fine-tuning. These are plausible explanations from the audit, not a causal proof.

**How were the timing policies compared fairly?** Every policy kept student and teacher ONNX sessions resident, used natural texts and four ORT threads, and was timed in three shuffled process blocks. Paired reductions compare each policy with teacher-only in the same block. Loading time is separate. [Timing protocol](gate-results.md)

**Does 90.48% prove the cascade loses less than one point generally?** No. It is a 0.57-point observed loss on 872 reviews. The paired bootstrap interval for the confidence gate's difference from teacher runs from -1.61 to +0.46 points, so a population loss larger than one point remains compatible with these data.

**Why is 24.1% not exactly the percentage computed from the displayed median times?** The table displays the median of process medians. The 24.1% is the median of three **within-block** percentage reductions against teacher-only. Taking a ratio after separately aggregating medians is a different calculation.

## Slide 10 — Next experiments

**Do you already have an INT8 result?** No result appears on this slide. The FP32-versus-INT8 protocol and runner are prepared, but accuracy, full-request latency, and resident memory must be measured before claiming a gain. [Plan](int8-student-extension-plan.md)

**What does “fresh calibration” require?** Labeled calibration reviews that the teacher was not trained on, followed by evaluation on another untouched labeled set. Reusing the same reviews for both cutoff choice and final accuracy would create the validation leakage I am trying to avoid.

**Why another CPU or longer reviews?** The current timing comes from one laptop, and SST-2 validation sentences are short. Longer inputs change teacher cost, and CPU runtimes can behave differently across hardware. Replication would test whether the observed routing tradeoff transfers.

## Slide 11 — Takeaway

**So was the project successful?** It produced a trained student, a complete resident cascade comparison, and a useful negative result for the proposed length-and-cost rule. The confidence baseline met my measured deployment criteria, but my proposed gate did not; I should present both plainly.

**What is your own contribution if the best rule is known?** I built and measured this controlled CPU pipeline: fixed-shape screening, student training, teacher-benefit fitting, frozen calibration, and full-request resident timing. I do not claim to have invented distillation or the cascade idea.

## Slide 12 — Q&A

**What is the one-sentence conclusion?** On this CPU and validation set, a trained student with confidence-based teacher fallback preserved most teacher accuracy and reduced median request time; the added length-and-cost routing rule missed its accuracy target.

## Slide 13 — One request through the gate

**Does the student run on every request?** Yes. I need its answer and logits before either gate can decide. If the gate selects fallback, I also run the teacher and return its prediction.

**Why is student runtime absent from the proposed score?** It is already paid for when the decision is made. The score compares estimated *additional* correctness with estimated *additional* teacher time. Full measured request latency still includes student time.

## Slide 14 — Teacher benefit

**What does a ridge output of 0.05 mean?** It estimates five net extra correct answers per 100 similar requests if I switch from student to teacher. It is not “the teacher is 5% accurate,” and an individual teacher prediction is not guaranteed to help. Ridge outputs are estimates, not calibrated probabilities.

**What are margin, log₂ length, and interaction?** Margin is the absolute difference between the student's two logits. Log₂ length compresses token count so each doubling adds one. Their interaction lets the relationship between student decisiveness and teacher benefit change with length.

## Slide 15 — Teacher time and cutoff

**Are 9.61, 17.91, and 27.83 ms complete request times?** No. They are medians of teacher ONNX inference time on the 2,048 gate-fit rows, grouped by token count. The final benchmark separately measures complete requests including tokenization, student execution, routing, and fallback.

**Did you add median student time and median teacher time to decide each request?** No. The proposed routing score is predicted teacher net gain divided by estimated *extra teacher* inference time for that length group. I compare that score with the frozen cutoff. Student time is included in the later end-to-end benchmark.

## Hard follow-up questions

**Why use ridge regression rather than a neural gate?** The gate-fit set has 2,048 examples but only 49 teacher fixes and 5 harms. A small regularized model is easier to audit and less flexible than another neural network. I have not established that ridge is the optimal gate model.

**Why log₂ of token length?** It treats each doubling of sequence length as one step and keeps very long reviews from dominating the raw feature scale. It was a modeling choice; the experiment does not prove log₂ is the best transform.

**Why add the interaction?** Without it, the effect of student margin would be the same at every length. The interaction allows that slope to change with length. Whether that extra flexibility improved generalization is answered by the validation comparison, where the selected combined rule missed its target.

**Why use median teacher time as cost if p95 matters?** Each length bin uses a median teacher-call time as a simple cost proxy, and the cutoff minimizes mean added proxy cost under an accuracy constraint. That proxy can miss slow fallbacks. I measured p95 of complete requests separately; the gate does not optimize p95 directly.

**Did you search many cutoffs until validation looked good?** No. The candidate fallback levels were fixed at 0%, 10%, 20%, 30%, 40%, 50%, 70%, and 100% on calibration. The selected cutoff was frozen before validation. Validation was previously used for aggregate reporting in the original fixed-shape study, so this is not a fresh external replication, but it did not set this gate's cutoff. [Fitting code](../scripts/fit_gate.py)

**Why could calibration say 10% fallback but validation have 13.9% or 18.5%?** The calibration quantile sets a *score threshold*, not a quota that forces 10% on future data. The score distributions changed, so more validation reviews crossed the same frozen threshold.

**Could a shared early-exit model avoid running two models?** Potentially. It could reuse early-layer computation instead of running a separate student and teacher. That is a different established architecture that I did not test; my result applies to two resident ONNX models.

**Is the 24.1% latency gain statistically proven?** It is the median of three paired CPU timing-block reductions, all of which favored the confidence gate, but three blocks on one laptop are not enough to claim a hardware-general speedup or a precise population interval.

**Why is the teacher's calibration accuracy 97.66% but its validation accuracy 91.06%?** The teacher had trained on the SST-2 training split from which calibration rows came, whereas validation is a different split. This gap is consistent with optimistic in-sample calibration, but it alone does not prove how much is due to memorization versus differences between the splits.

## Useful answer pattern

For a difficult question: **direct answer → one measured number or code choice → one limitation**. For example: “No, the proposed gate did not meet the accuracy budget. It was 1.72 points below the teacher on 872 reviews. These are results from one laptop and validation set.”
