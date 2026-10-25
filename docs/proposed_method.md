> **Naming note:** The implementation identifier remains `prism` for backward compatibility, but the application label is **PRISM-RP (RENA projected-recovery adaptation)** to avoid implying identity with unrelated published methods using the PRISM name.

# PRISM methodology (pre-registered design)

## Motivation and gap

The existing gradient-ascent baseline directly maximizes forget-set loss and then repairs utility with retain-set fine-tuning.  This can raise forgetting proxies while producing disagreement across verifiers.  ECHO observes proxies, but its candidate operators are all unprojected ascent/correction combinations and its seed-42 trajectory made only a very small parameter update.  Neither method explicitly constrains the first-order retain-loss effect of its forget update.

## Proposed method: PRISM

**Projected Recovery-aware Iterative Selective unlearning (PRISM)** performs forget-loss ascent in the subspace orthogonal to a retain-loss gradient, followed by a small retain correction.  At each iteration it evaluates a fixed, pre-specified set of three step sizes and commits the feasible candidate with the lowest recovery proxy, then the smallest verifier-proxy dispersion.  This is a constrained-gradient algorithm, not a retraining method or a renamed ECHO controller.

Let \(L_f(\theta)\) and \(L_r(\theta)\) denote mean cross-entropy on forget and retain data.  With \(g_f=\nabla L_f\) and \(g_r=\nabla L_r\), PRISM uses

\[
d = g_f - \max\left(0,\frac{g_f^\top g_r}{\lVert g_r\rVert^2+\epsilon}\right)g_r.
\]

Thus \(g_r^\top d\leq0\), so the ascent update \(\theta\leftarrow\theta+\eta d\) cannot increase retain loss to first order when the gradients conflict.  A retain-descent correction follows each candidate.  Candidate selection is constrained by pre-update retain accuracy and scores direct loss-based recoverability plus dispersion among loss, confidence, and membership proxies.  Full RENA verifiers are run only after PRISM completes; the proxies are never reported as replacements for those verifiers.

## Differences, complexity, and hypotheses

Unlike fine-tuning, PRISM actively raises forget loss.  Unlike ordinary gradient ascent, it removes the retain-conflicting gradient component.  Unlike SISA and ground-truth retraining, it updates one existing model rather than rebuilding shards or a full model.  Unlike ECHO, its central mechanism is an analytic first-order constraint rather than Pareto choice among unprojected operators.

Each iteration requires two full-batch backward passes and three candidate forward evaluations, i.e. \(O(T(|F|+|R|)P)\) for parameter count \(P\), plus the configured recovery proxy evaluations.  We hypothesize that projection preserves retain utility and reduces confidence/verifier conflict relative to ordinary ascent.  It may fail when forget and retain gradients are nearly aligned, when the loss-based recovery proxy is not representative of stronger attacks, or when the first-order constraint does not predict finite-step behavior.  The evaluation must therefore be interpreted as empirical evidence on this dataset only.
