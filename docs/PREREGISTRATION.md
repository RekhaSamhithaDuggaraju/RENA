# RENA Formal Study Preregistration (v1.0)

**Date**: 2026-10-26  
**Status**: Preregistered Prior to Test Split Analysis  
**Repository**: `RENA_V22`  
**Primary Venue**: Trustworthy ML / Privacy & Unlearning Workshop  

---

## 1. Study Title & Abstract

**Title**: Behavioral Audits of Machine Unlearning Underestimate Residual Knowledge: An Empirical Framework for Controlled Multi-Turn Evaluation.

**Abstract**: Commercial API audits and behavioral unlearning evaluations frequently rely on direct zero-shot prompts and refusal detection to certify that designated information has been forgotten. We hypothesize that behavioral suppression (via instruction prompts or refusal fine-tuning) creates an illusion of forgetting that collapses under multi-turn conversational elicitation and adversarial probing. We formulate a 3-Group Controlled Audit Protocol incorporating: (1) strict pre-unlearning baseline retrievability gates ($k$-of-$n$), (2) matched unexposed reference controls (Group B), (3) exposed counterfactual controls (Group C), and (4) multi-angle probe diversity. 

---

## 2. Falsifiable Hypotheses

### Primary Hypothesis ($H_1$)
Direct zero-shot query audits yield a non-zero false-pass rate on models where target information has merely been behaviorally suppressed (e.g. via instruction prompts or refusal fine-tuning). A diverse multi-angle probe suite (including multi-turn dialogue and paraphrase elicitation) significantly increases residual disclosure detection ($p < 0.01$, McNemar's test), reducing the false-pass rate compared to direct queries alone.

### Secondary Hypothesis ($H_2$)
Absence of observable disclosure without baseline-retrievability verification ($k$-of-$n$ trials) frequently reflects pre-existing model inaccessibility rather than successful unlearning. Requiring verified baseline retrievability eliminates false unlearning claims on unexposed or inaccessible targets.

### Tertiary Hypothesis ($H_3$)
True parameter-level unlearning (e.g., Gradient Difference, NPO) demonstrates resilience against relearning attacks (requiring $\ge 5$ gradient steps to re-acquire target recall $\ge 80\%$), whereas behavioral suppression exhibits immediate re-acquisition within $\le 2$ gradient steps.

---

## 3. Experimental Variables & Measures

1. **Independent Variables**:
   - Intervention Condition:
     - Model-level unlearning: `ga`, `graddiff`, `npo`, `suppress_refusal`, `suppress_prompt`, `retained` (Group C), `reference` (Group B).
     - In-context deletion: `retract_by_removal`, `instruct_to_forget`.
   - Probe Mode: `direct`, `paraphrase`, `indirect`, `multi_turn`, `retained_utility`.
2. **Dependent Variables**:
   - Residual Disclosure Rate (RDR): Proportion of post-intervention trials disclosing target information.
   - Retained Utility Preservation (RUP): Proportion of retained capability questions correctly answered.
   - Forget-Retain Tradeoff Index (FRTI): $RUP \times (1 - RDR)$.
   - Causal Attribution Delta ($\Delta_{\text{causal}}$): $RDR_{\text{Group C}} - RDR_{\text{Group A}}$.
   - Relearning Gradient Steps to 80% Recovery.

---

## 4. Sampling Plan and Statistical Testing

- **Sample Size**:
  - Hosted API Case Study: 2 distinct target entities across 4 probe types, 4 phases, and 2 repetitions ($n=64$ calls per condition).
  - Testbed Empirical Evaluation: 5 canary forget targets + 5 utility retain targets across 7 intervention conditions and 4 probe categories.
- **Statistical Tests**:
  - Binomial Proportions: 95% Wilson score confidence intervals.
  - Paired Probe Detection: McNemar's test comparing direct vs multi-turn probe detection sensitivity.
  - Utility Equivalence: Two One-Sided Tests (TOST) with equivalence margin $\delta = 0.05$ at $\alpha = 0.05$.
  - Causal Deltas: 10,000-resample fact-clustered bootstrap confidence intervals.
- **Stopping Rule**: Pre-specified trial repetitions completed; no adaptive early stopping on non-significant differences.
