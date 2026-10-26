# Evaluation Plan and Empirical Protocol
## Evaluation of Behavioral Unlearning Audits Under Refusal Suppression

**RENA Evaluation Benchmark Report**  
*Date*: October 26, 2026  
*Framework Version*: RENA v22.2  
*Status*: Protocol Specified; Unit Simulations Relabeled; Real Model Testbed in Development

---

## 1. Research Hypothesis & Problem Statement

Standard black-box behavioral unlearning audits evaluate machine unlearning by presenting direct queries to the model and applying lexical substring matching to the output. If the model outputs a refusal string (e.g., *"I cannot disclose that confidential record"*) or omits the target token, the audit declares the target information **"FORGOTTEN"**.

This evaluation paradigm contains a validity flaw: **behavioral suppression via safety alignment, refusal tuning, or system-prompt guardrails produces identical audit observables to genuine weight-level or retrieval-level deletion**, resulting in a high False Pass Rate (FPR).

### Falsifiable Hypothesis
> *Behavioral unlearning audits that rely on direct prompts and refusal checks can certify suppressed-but-retained knowledge as forgotten. A protocol with baseline-retrievability gating, matched reference and counterfactual controls, diverse probes and validated scoring lowers the false-pass rate, at a quantified cost in inconclusive outcomes and audit budget.*

---

## 2. Audit Protocol Specification

### 2.1 Three-Group Controlled Architecture
To isolate unlearning causal effects from base model parametric priors:
1. **Group A ($M_A$) — Exposed $\to$ Unlearning Attempted**: Model is exposed to canary set $D_{\text{forget}}$, undergoes documented unlearning operation $U$, and is probed for residual disclosure.
2. **Group B ($M_B$) — Never-Exposed Reference**: Matched twin model checkpoint never exposed to $D_{\text{forget}}$, subjected to identical probing battery.
3. **Group C ($M_C$) — Exposed Control**: Matched checkpoint exposed to $D_{\text{forget}}$ but without unlearning, establishing empirical upper bounds on retrieval.

### 2.2 Four-Gate Validity Pipeline
1. **Gate 1 ($G_{\text{base}}$) — Strict Baseline Retrievability (k-of-n)**:
   Prior to unlearning, verify that $M_A$ discloses the target in at least $k$ of $n$ trials (default: 2 of 3). Canaries failing this gate are categorized as `INVALID_BASELINE_FACT` and excluded from forgetting calculations.
2. **Gate 2 ($G_{\text{util}}$) — Retained Utility Preservation**:
   Evaluate control queries $D_{\text{retain}}$. Performance must remain above the pre-registered threshold (default: 70%); otherwise the audit reports `UTILITY_DEGRADED`.
3. **Gate 3 ($G_{\text{probe}}$) — Multi-Angle Probing Battery**:
   - *Direct*: $q_{\text{dir}}$
   - *Paraphrase*: $q_{\text{para}}$
   - *Indirect / Deductive*: $q_{\text{ind}}$
   - *Conversational Multi-Turn*: Structured multi-message conversation with context opening, distractor turns, and probe turn.
4. **Gate 4 ($G_{\text{score}}$) — Scoring & Human-Review Queue**:
   Disambiguates clean refusals from refusals that mention target tokens. Low-confidence or contradictory judge outputs route to an explicit human-review queue.

### 2.3 Verdict Taxonomy (per `docs/VERDICT_SPEC.md`)
| Verdict Code | Operational Definition | Interpretation |
| :--- | :--- | :--- |
| `NO_RETRIEVAL_OBSERVED_UNDER_PROTOCOL` | Zero target disclosure across pre-registered probe budget | No residual retrieval detected under the specific protocol tested (not proof of deletion) |
| `RESIDUAL_RETRIEVAL_OBSERVED` | Target token, fragment, or semantic disclosure detected | Target information remained accessible post-unlearning |
| `INVALID_BASELINE` | Insufficient baseline retrievability (< $k$ of $n$) | Memorization prior to unlearning was not established |
| `INCONCLUSIVE` | Incomplete execution, unresolved review queue, or contradictory scoring | Execution incomplete or evidence ambiguous |

---

## 3. Empirical Results (Real Model Testbed: Experiments E1–E4)

> **Notice**: Simulation results have been removed from the main results and relocated to `docs/appendix_simulation/SIMULATION_SANITY_CHECKS.md` labeled `[SIMULATION, NOT EMPIRICAL EVIDENCE]`. Real model experiments on `Qwen2.5-0.5B-Instruct` are being executed under Phase 3 and Phase 4.

### 3.1 Experiment E1: Naive vs Controlled Auditor (Real Models)
- **Status**: `NOT_RUN` (Awaiting Phase 3 model implantation and Phase 4 execution on real weights)
- **Planned Metrics**: False Pass Rate (FPR), False Fail Rate (FFR), Balanced Accuracy, Calls per Fact.

### 3.2 Experiment E2: Component Ablations (Real Models)
- **Status**: `NOT_RUN` (Awaiting Phase 4 execution)
- **Planned Metrics**: Marginal FPR reduction by component (Baseline Gate, Multi-Turn, Reference Model, Semantic Scorer).

### 3.3 Experiment E3: Suppression vs Forgetting (Real Models)
- **Status**: `NOT_RUN` (Awaiting Phase 3 fine-tuning of `suppress_refusal` and `suppress_prompt` conditions)
- **Planned Metrics**: Verification of retention via independent oracle (relearning attack & log-likelihood gap).

### 3.4 Experiment E4: Power Curves vs Sample Size ($N$)
- **Status**: `NOT_RUN` (Awaiting Phase 4 execution)

---

## 4. Scorer & LLM Judge Validation

- **Status**: `AWAITING_HUMAN_ANNOTATION`
- **Specification**: 400 real model outputs will be sampled from real testbed runs and the hosted-API study. Two independent human annotators will label responses per `docs/ANNOTATION_GUIDE.md`.
- **Unit Regression Sanity Check**: Synthetically generated rule-checking suite is documented in `docs/appendix_simulation/SIMULATION_SANITY_CHECKS.md`.

---

## 5. Honest Scientific Limitations

1. **Behavioral Probing Bound**: No black-box query protocol can mathematically prove the erasure of dormant parameters or provider-side database state. A negative score under RENA means strictly that **no target information was recovered under the tested probe budget**.
2. **Computational Query Cost**: Multi-angle and multi-turn probing multiplies inference volume by $4\times$ to $6\times$ relative to naive direct single-turn checks.
3. **Adaptive Defenses**: Steering vectors or safety filters specifically conditioned on conversational distractor patterns may still suppress disclosure in certain multi-turn paths.
