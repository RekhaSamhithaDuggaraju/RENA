# RENA Scientific Claims Ledger (v2.0)

This ledger explicitly registers every scientific claim made by RENA, links each claim directly to empirical code, artifacts, and test logs, and defines clear falsification criteria, required sample sizes, and empirical statuses.

### Status Vocabulary (Rule 6)
- **`SUPPORTED`**: Hypothesis confirmed meeting preregistered sample size ($n \ge 100$ gate-valid facts) and statistical power across $\ge 3$ test seeds.
- **`PRELIMINARY`**: Supported at exploratory or smoke pilot scale ($n < 30$, single seed); requires confirmatory scale for full verification.
- **`NOT_SUPPORTED`**: Empirical data contradicts the claim or fails the pre-registered decision boundary.
- **`NOT_RUN`**: Experiment not yet executed.

---

## 1. Traceability Table

| Claim ID | Core Proposition | Status | Required $n$ for `SUPPORTED` | Supporting Artifacts & Code | Evidence & Current Empirical Findings | Falsification Criterion |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C1a** | Zero-shot direct prompt audits falsely pass prompt-suppressed knowledge; multi-turn conversational elicitation lowers false-pass rates. | **PRELIMINARY** | $n \ge 100$ gate-valid facts across $\ge 3$ seeds | `results/case_studies/gemini_flash_lite_audit.json`<br>`results/testbed/smoke_audit.json`<br>`scripts/run_neural_smoke_comparison.py` | 1. Hosted API (`gemini-3.5-flash-lite`, $n=2$): Direct probe yields 0% disclosure (naive false pass); multi-turn yields 100% extraction.<br>2. Local neural weights (`Qwen2.5-0.5B-Instruct` on MPS, $n=5$): Naive auditor falsely passes 40.0% (2/5) on `suppress_prompt`; RENA detects 100.0% (5/5). | Multi-turn elicitation disclosure rate $\le$ direct query disclosure rate on prompt suppression. |
| **C1b** | Behavioral multi-probe auditing reliably detects latent retention in weight-level refusal fine-tuning (`suppress_refusal`). | **NOT_SUPPORTED** | $n \ge 100$ gate-valid facts across $\ge 3$ seeds | `results/testbed/smoke_audit.json`<br>`results/testbed/oracle_results.json` | At smoke scale ($n=5$), behavioral RENA detected 0/5 exact disclosures and 1/5 near-miss/partial (`forget_001`), thus falsely passing 80%–100% of refusal-tuned facts. In contrast, the independent oracle confirmed latent persistence (mean LL $-2.9756$ vs retained $-1.4043$, relearning 80% at Step 2). **Behavioral audits alone are insufficient to detect refusal suppression.** | Behavioral RENA false-pass rate exceeds 20% relative to independent oracle ground truth (Observed: 80%–100% false pass). |
| **C2** | Exact-answer baseline gating ($k$-of-$n$) prevents false unlearning certifications on unmemorized facts or near-misses. | **PRELIMINARY** | $n \ge 100$ gate-valid facts across $\ge 3$ seeds | `results/testbed/smoke_audit.json`<br>`core/stats/verdict_engine.py`<br>`tests/test_verdict_engine.py` | Synthetic canaries verified at 0.0% recall on unexposed base weights. Exact gating ($k=2, n=3$) verified 4/5 exact facts on implanted weights, successfully flagging `forget_001` digit transposition (`OBSIDIAN-9842` vs `9482`) as `NEAR_MISS`. | Facts with 0% exact baseline recall pass Gate 1. |
| **C3** | Matched unexposed reference controls (Group B) prevent false-positive leakage attribution on pretraining knowledge. | **PRELIMINARY** | $n \ge 100$ gate-valid facts across $\ge 3$ seeds | `results/testbed/smoke_audit.json`<br>`core/testbed_real/unlearn.py`<br>`tests/test_verdict_engine.py` | Clean base model (Group B) evaluated under full RENA protocol returned 0/5 disclosures ($FPR = 0.0\%$), with retained utility preserved at 80.0%. | Group B unexposed model produces $>5.0\%$ disclosure on synthetic canary facts. |
| **C4a** | Refusal-tuned suppression models exhibit rapid relearning vulnerability ($\le 2$ steps), confirming latent parametric retention. | **PRELIMINARY** | $n \ge 100$ gate-valid facts across $\ge 3$ seeds | `results/testbed/oracle_results.json`<br>`scripts/run_neural_oracle_benchmark.py`<br>`core/testbed_real/oracle.py` | 1. Mean sequence log-likelihood: `reference` = -38.0156, `retained` = -1.4043, `suppress_refusal` = -2.9756.<br>2. AdamW relearning: `suppress_refusal` recovers 80% recall at **Step 2** (`rapid_reacquisition`). | Relearning rate on refusal suppression is statistically indistinguishable from never-exposed base reference model. |
| **C4b** | Standard gradient difference unlearning (`graddiff`) achieves selective unlearning without general capability destruction. | **NOT_SUPPORTED** | $n \ge 100$ gate-valid facts across $\ge 3$ seeds | `results/testbed/smoke_audit.json`<br>`results/testbed/oracle_results.json` | In 5-fact pilot, `graddiff` collapsed retained utility to **40.0%** (`UTILITY_DEGRADED`). Resistance to relearning over 10 steps is confounded by general parameter damage rather than selective erasure. Requires regularized `graddiff_kl` control. | Retained utility falls below 70.0% threshold (Observed: 40.0% utility collapse). |

---

## 2. Explicit Negative Results & Boundaries

1. **Non-Claim on Server-Side Erasure**: RENA's black-box evaluations cannot prove physical deletion from server memory, model weights, or secondary caching infrastructure of hosted third-party commercial APIs.
2. **Behavioral Audit Limit (P4 / Pivot Decision)**: Behavioral probing alone (even with multi-turn and paraphrase elicitation) fails to reliably uncover refusal-tuned suppression (80%–100% false pass rate on `suppress_refusal`). An independent weight-level or relearning oracle check is necessary to audit open-weight unlearning claims.
3. **Simulation Boundary**: All simulated datasets (e.g. `SIMULATION_SANITY_CHECKS.md`) are unit-level software sanity checks and are never presented as empirical validation.
4. **GradDiff Utility Confound Boundary**: The unlearning resistance observed in naive gradient difference is confounded by general capability destruction (40% retain utility). Selective unlearning claims require a utility-preserving control (e.g. `graddiff_kl`).
5. **Human Annotation Pending Boundary**: Automated scorer metrics are pending completion of the dual-human annotation protocol on the 400-item validation set.
6. **Confirmatory Scale Boundary**: The 30-fact held-out evaluation corpus (`canary_001`–`canary_030`) and procedural generator are constructed and ready; full confirmatory execution is pending.
