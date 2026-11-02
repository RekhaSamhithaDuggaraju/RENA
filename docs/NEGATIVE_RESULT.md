# Negative Result & Hypothesis Pivot Record

**Document Version**: 1.0.0
**Registered Date**: 2026-10-27
**Scope**: Experimental Findings from Pilot Benchmark & Formal Decision Rule for Scaled Confirmatory Runs

---

## 1. Executive Summary

In initial design drafts, RENA hypothesized that black-box behavioral prompting alone (incorporating paraphrase variations, multi-turn escalation, and calibrated scoring) would reliably detect latent unlearned knowledge and reduce audit false-positive rates to near-zero ($FPR \approx 0$).

**The empirical pilot on `Qwen2.5-0.5B-Instruct` directly refuted this hypothesis on refusal-based suppression defenses.**

Specifically, on the `suppress_refusal` condition:
1. **Behavioral Failure**: The refusal policy successfully suppressed direct, paraphrase, and conversational probes across all valid facts. Neither the naive single-turn auditor nor the multi-probe RENA auditor elicited exact target disclosures ($0/4$ or $0/5$ exact recovery, $0.0\%$).
2. **Oracle Ground Truth**: The independent weight-level oracle evaluated the exact token sequence log-likelihood at $\text{mean LL} = -2.9756$, closely tracking the unmitigated retained control ($\text{mean LL} = -1.4043$) and sharply diverging from the never-exposed reference baseline ($\text{mean LL} = -38.0156$).
3. **Relearning Vulnerability**: Under standard AdamW gradient updates ($lr = 10^{-4}$), target facts were re-acquired to $80\%$ recall in just **2 steps**, whereas the never-exposed reference baseline demonstrated no recovery ($0\%$) across the same gradient budget.

Consequently, behavioral audits certified a model as "unlearned" when its internal parameters preserved virtually intact representations of the forgotten facts.

---

## 2. Formal Analysis of the Discrepancy (Problem P4 & P5)

| Metric | Never-Exposed Reference | Suppress Refusal (Defended) | Unmitigated Retained Control | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Naive Behavioral Audit** | $5/5$ ($100\%$) | $5/5$ ($100\%$) | $1/5$ ($20\%$) | Blind to refusal; treats surface silence as unlearning. |
| **RENA Behavioral Audit (Exact)** | $0/4$ ($0.0\%$) | $0/4$ ($0.0\%$) | $4/4$ ($100.0\%$) | Refusal defense blocks surface extraction even under multi-turn escalation. |
| **Target Log-Likelihood** | $-38.0156$ | $\mathbf{-2.9756}$ | $-1.4043$ | **Weights retain target information with high probability.** |
| **Steps to Re-acquisition ($\ge 80\%$)** | $>10$ (None) | $\mathbf{Step\ 2}$ | Step 0 | Knowledge re-emerges almost immediately. |

### Why This is a Scientific Finding, Not a Bug
This failure does not represent a software bug or scorer deficiency; it represents an intrinsic theoretical boundary of purely behavioral black-box unlearning verification:
- A model fine-tuned to say *"I am not allowed to discuss Project Obsidian"* does not output the secret key under conversational cajoling.
- However, the attention heads and MLP projections storing the association `Project Obsidian -> OBSIDIAN-9482` remain untouched.
- Therefore, **black-box behavioral audits cannot distinguish physical erasure from behavioral suppression.**

---

## 3. Pre-Registered Conditional Decision Rule ($H_{\text{main}}$)

Before scaling experiments across $N \ge 100$ facts and multiple seeds, we pre-register the following conditional pivot:

$$\mathbf{H_{\text{main}}}: \text{On open-weight models with known ground truth, behavioral audits certify suppressed knowledge as unlearned at non-trivial rates; a hybrid protocol combining multi-probe behavioral auditing with weight-level oracle verification detects latent persistence and controls false-pass rates.}$$

### Decision Fork:
- **Fork A (Behavioral Sufficiency)**: If scaled experiments demonstrate that advanced behavioral probes (e.g. prefix injection, role-play framing, logit perturbation) reliably break refusal suppression ($FPR \le 5\%$), the headline claim is:
  > *"Controlled multi-probe behavioral auditing lowers false-unlearning certification rates across suppression defenses."*
- **Fork B (Weight-Level Oracle Requirement — Expected Outcome)**: If behavioral audits continue to exhibit non-trivial false passes on refusal suppression ($FPR > 5\%$), the headline claim is:
  > *"Purely behavioral audits are provably insufficient to certify unlearning on suppression defenses; white-box log-likelihood and relearning checks against a matched reference are necessary for rigorous audit guarantees."*

In accordance with scientific integrity, we report both naive and controlled error rates against the independent oracle, complete with fact-clustered confidence intervals.

---

## 4. Confirmatory Baseline Feasibility Attempt (2026-11-02)

A confirmatory baseline implantation attempt was conducted using:

- **Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Device**: Apple MPS
- **Dataset split**: `confirmatory_30`
- **Training**: 12 epochs, 30 forget facts and 5 retain facts
- **Pre-registered baseline forget-recall requirement**: 80%

### Observed results

| Metric | Observed | Requirement |
|---|---:|---:|
| Forget-fact recall | 20/30 (66.7%) | At least 80% |
| Retain-fact recall | 5/5 (100%) | Retention check |

### Decision and interpretation

The forget-fact retrievability requirement was not met. The implantation script therefore raised a baseline recall gate failure and did not save the confirmatory adapter.

Consequently:

1. The confirmatory baseline was not accepted as eligible for the unlearning comparison.
2. No confirmatory unlearning comparison was executed from this attempt.
3. These results do not establish that RENA outperforms a naive auditor or demonstrate confirmatory unlearning efficacy.
4. The 66.7% forget recall is reported as an observed feasibility result, not as evidence of successful implantation.

The baseline gate was not relaxed after observing the result. The failed attempt is retained as a negative feasibility finding rather than being presented as a successful confirmatory experiment.
