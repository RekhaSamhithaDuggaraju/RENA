# RENA — Phase 1: Problem Framing & Consensus Engine Specification

---

## PHASE 1: PROBLEM FRAMING

### 1.1 What is Machine Unlearning?

Machine Unlearning is the process of removing the influence of specific training data from an already-trained model, without fully retraining it from scratch. A "forget set" (e.g. one user's data, or a mislabeled batch) is designated, and an algorithm modifies the model's weights so it behaves as if it had never seen that data — while retaining performance on everything else (the "retain set").

Motivations: GDPR/CCPA "right to be forgotten" requests, removing poisoned or copyrighted data, correcting privacy leaks, and reducing the cost of full retraining.

### 1.2 Why Does Verification Matter?

Unlearning algorithms report internal metrics (e.g. retain-set accuracy) as proof of success, but **accuracy retention does not prove forgetting**. A model can score well on accuracy while still leaking membership information through its confidence distribution, gradients, or loss landscape. This is the central gap RENA addresses:

- Unlearning research has produced many *algorithms* claiming to forget data.
- Almost none of them are separately audited using methodology separate from the algorithm's own training objective.
- Regulators, auditors, and downstream users need a way to check the claim, not just trust the algorithm's self-report.

This is directly analogous to the difference between a company self-reporting financial results and an independent audit — RENA is the audit function, not the accounting function.

### 1.3 Who Uses RENA?

| User | Need |
|---|---|
| ML researchers | Benchmark and compare unlearning algorithms with a standardized, reference-relative scoring methodology |
| Responsible AI / compliance teams | Produce an evidentiary report that a deletion request was honored |
| Model auditors / regulators | Independently check vendor claims of "we deleted your data" |
| Students / educators | Understand *why* an unlearning method succeeds or fails, not just that it did |

### 1.4 What Problems Does RENA Solve?

1. **No standard verification methodology** — every unlearning paper invents its own evaluation.
2. **Single-metric fragility** — one test (e.g. one membership inference attack) can be fooled or can disagree with another; no existing tool aggregates multiple complementary tests into a principled verdict.
3. **No explainability on failure** — when unlearning fails, existing tools report a number, not a diagnosis of *why*.
4. **No audit trail** — no reproducible, reportable artifact suitable for compliance or research citation.

### 1.5 Limitations (state these explicitly — do not hide them)

- RENA verifies *statistical evidence* of forgetting, not a formal/cryptographic guarantee. Absence of detectable leakage is not mathematical proof of zero leakage.
- All four verification modules are approximations with known failure modes (e.g. MIA power depends on model capacity and forget-set size).
- Verification is currently scoped to classification models on tabular/image data; it does not (yet) cover generative models or LLMs.
- Small forget sets produce statistically weaker signals — RENA should report a **minimum-forget-set-size confidence caveat** whenever the forget set is small.

### 1.6 Success Criteria

RENA is successful if, for a given (model, forget-set, unlearning algorithm) triple, it can:
1. Produce a **reproducible** verdict from 4 complementary verification witnesses.
2. Correctly and *consistently* rank algorithms in the order the unlearning literature predicts (e.g. ground-truth retraining ≥ SISA ≥ gradient ascent ≥ naive fine-tuning, in expectation).
3. Force an "Inconclusive" verdict rather than a false-confident one when verification modules disagree.
4. Produce a report that traces every claim in the verdict back to a specific computed number.

---

## CONSENSUS ENGINE SPECIFICATION

This is RENA's core intellectual contribution. It must be **fully specified before implementation** — no naive averaging, and no hand-waved "AI decides" step.

### 2.1 Inputs

Each of the 4 verification modules produces a standardized output:

```
ModuleResult {
  module_name: str
  raw_statistic: float          # e.g. MIA AUC, KS statistic, confidence gap
  p_value: float | None         # statistical significance where applicable
  forget_score: float [0,1]     # 0 = clearly NOT forgotten, 1 = clearly forgotten
  reliability: float [0,1]      # module's own confidence in its result (see 2.2)
}
```

Modules and how `forget_score` is derived:

| Module | Raw statistic | forget_score mapping |
|---|---|---|
| Membership Inference Attack (MIA) | Attack AUC (0.5 = random guess, 1.0 = perfect leak) | `forget_score = 1 - 2*max(0, AUC - 0.5)` — AUC of 0.5 → score 1.0 (fully forgotten); AUC of 1.0 → score 0.0 |
| Loss Distribution KS-test | KS statistic between forget-set loss and held-out test-set loss distributions | `forget_score = 1 - KS_statistic` (distributions indistinguishable → high score) |
| Confidence Analysis | Mean confidence gap between forget-set and test-set predictions | `forget_score = 1 - min(1, gap / gap_naive_baseline)` where `gap_naive_baseline` is the gap measured on the *original, unlearned-free* model (calibrates against absolute confidence scale, not raw magnitude) |
| Bootstrap Statistical Witness | Bootstrap-resampled p-value that forget-set metrics differ from retain-set metrics | `forget_score = min(1, p_value / 0.05)` capped at 1 — non-significant difference → high score |

### 2.2 Reliability Weighting (not a fixed weight — computed per audit)

Each module's `reliability` is **not a static hyperparameter**. It is computed per-audit from data-dependent factors, because a module can be individually powerful or weak depending on forget-set size and class balance:

```
reliability = f(sample_size_factor, variance_factor, known_failure_mode_flag)
```

- `sample_size_factor`: scales down reliability when forget-set size < 50 samples (statistical power drops sharply below this in practice for MIA-style tests). Use a smooth function, e.g. `min(1, forget_set_size / 50)`.
- `variance_factor`: for bootstrap-based modules, use the bootstrap standard error directly — high variance in the resampled estimate → lower reliability.
- `known_failure_mode_flag`: e.g. MIA reliability is capped at 0.6 when the underlying model's overall test accuracy is below a threshold (a weak model produces weak/uninformative attack signal regardless of unlearning quality) — this is a documented, literature-grounded discount, not an arbitrary penalty.

This keeps reliability **auditable**: every discount applied must be traceable to a stated, documented reason, surfaced in the report.

### 2.3 Weighted Score

```
consensus_score = Σ(forget_score_i * reliability_i) / Σ(reliability_i)
```

This is a reliability-weighted average — deliberately simple and transparent at this stage. The engine's novelty is not in this formula; it's in step 2.4, which can override it.

### 2.4 Agreement / Disagreement Metric (the actual novel step)

Compute the **spread** of forget_scores across modules, weighted by their reliability:

```
weighted_variance = Σ(reliability_i * (forget_score_i - consensus_score)^2) / Σ(reliability_i)
agreement = 1 - min(1, weighted_variance / MAX_EXPECTED_VARIANCE)
```

`MAX_EXPECTED_VARIANCE` is calibrated empirically once (e.g. via the ground-truth-retraining and fine-tuning baselines, which should produce the highest and lowest agreement scores respectively as sanity anchors) — this constant must be documented and versioned, not silently tuned.

**Override rule (this is what prevents a naive-average false confidence):**

```
IF agreement < AGREEMENT_THRESHOLD (e.g. 0.5):
    verdict = "Inconclusive"
    regardless of consensus_score
ELIF consensus_score >= 0.75 AND agreement >= 0.7:
    verdict = "Likely Forgotten"
ELIF consensus_score <= 0.35 AND agreement >= 0.7:
    verdict = "Not Forgotten"
ELSE:
    verdict = "Ambiguous"
```

This means: **high average score with low agreement is explicitly NOT allowed to produce a confident verdict.** This is the property the earlier transcript's unit test ("proves the consensus engine correctly resists being fooled by disagreement") should be testing directly — write that test against this rule, not against the weighted average alone.

### 2.5 Explainability Output

Every consensus result must emit, alongside the verdict:

```
ConsensusExplanation {
  verdict: str
  consensus_score: float
  agreement: float
  per_module_contribution: [
    { module: str, forget_score: float, reliability: float, weight_pct: float }
  ]
  disagreement_flag: bool
  discounted_modules: [ { module: str, reason: str, penalty_applied: float } ]
}
```

`discounted_modules` is what feeds the Failure Diagnosis Engine — e.g. "MIA reliability was capped at 0.6 because base model test accuracy was 71%, below the 80% threshold for reliable attack signal."

### 2.6 What Must Be Tested

1. Ground-truth retraining baseline → should reliably produce highest consensus_score and highest agreement across repeated runs (sanity check on the whole engine).
2. Naive fine-tuning baseline → should reliably produce lowest score.
3. **Adversarial disagreement test**: construct synthetic module outputs where 3 modules say "forgotten" (score ~0.9) and 1 says "not forgotten" (score ~0.1) with equal reliability. Assert the engine does NOT return "Likely Forgotten" — it must return "Inconclusive" or "Ambiguous" per the agreement override, even though the naive average would be ~0.7.
4. Reliability discount test: forget-set size of 10 samples should visibly reduce that module's weight_pct in the explanation output.

---

## Next Steps

This spec is deliberately opinionated and calibratable — the exact thresholds (0.5, 0.7, 0.75, 0.35) are starting points, not final constants, and should be tuned against the baseline sanity checks in 2.6 before being treated as fixed.

Suggested next phase: **Phase 3 architecture**, using this spec as the contract for the `/consensus` module's implementation — API schema, DB table for `discounted_modules` history, and how this explanation object maps into the PDF report's "Consensus" section.
