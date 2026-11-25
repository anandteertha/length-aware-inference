# Gate formula search protocol

Use the existing FP32 student, teacher, and measured cost table. Let \(m\) and
\(\ell\) be the original standardized student margin and log token length.
Let \(G\) be the original ridge benefit prediction and \(C\) the original
teacher cost estimate in milliseconds. Screen the following three families.

1. **Coefficient shrinkage.** Multiply the fitted length and margin-by-length
   terms independently by 0, 0.25, 0.5, or 1. Divide by \(C^\alpha\) for
   \(\alpha\in\{0,0.5,0.75\}\).
2. **Additive cost.** Use \(G-\lambda(C/C_{\rm short}-1)\) with
   \(\lambda\in\{0.005,0.015,0.03,0.06\}\). This charges a separate penalty
   for a more expensive call. Benefit minus cost deferral has prior work
   [Jitkrittum et al.](https://arxiv.org/abs/2307.02764).
3. **Predicted class adjustment.** Fit the original ridge target with the
   original features plus the student's predicted class as \(-1\) or \(+1\)
   and its interaction with margin. This lets the gate treat equally
   confident positive and negative predictions differently. Try ridge
   penalties of 10 and 40 on the 2,048 original gate-fit examples, and cost
   exponents of 0, 0.5, and 0.75.

Freeze two candidates before running test predictions. Select one using the
sum of net corrections at 10%, 20%, and 30% teacher-call quotas on the original
1,024 calibration rows, with quotas rounded up. Break ties by fewer estimated
teacher milliseconds at the 10% quota, then grid order. Select the second using
the already inspected 872-row GLUE validation data as development data, ranked
by correct predictions at the original confidence policy's observed 161
calls, then fewer estimated teacher milliseconds and grid order. Keep the
teacher-call quota fixed in each selection comparison. Every candidate's
deployable cutoff comes from its top 10% quantile on the original calibration
rows. Use stable row order for ties in the offline quotas.

The primary test is fixed call-count accuracy against confidence-only. Set
the quota using the number of requests that cross the original frozen
confidence cutoff on the test inputs. This offline comparison ranks the batch
without labels. Also report the fixed-cutoff accuracy and calls, the paired
bootstrap intervals, and the oracle number of available teacher fixes. Report
both selection paths and all development results. The score search is a
follow-up experiment. Measured costs from the prior study are proxies here.

Use Stanford's [released binary test
trees](https://github.com/stanfordnlp/sentiment-treebank/blob/fa0a89e0cabf5eac3ff4e1d27551fad4046ec012/binary/test-binary.txt).
Keep the original positive and negative root labels and strip tree structure.
Exclude neutral roots, exact overlaps with GLUE SST-2 training and validation
after case and punctuation normalization, and duplicate normalized test
sentences. This matches the task and source domain, though text formatting can
differ from GLUE. Archive row IDs, labels, and logits, with the dataset and
model hashes. Report paired uncertainty and require a consistent result before
claiming superiority.
