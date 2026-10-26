# RENA v22.2 Evaluation & Project Status
## Audit Report & Verification Status

**Date**: October 26, 2026  
**Status**: Real Model Testbed & Oracle Benchmark Executed; Hosted Gemini Case Study Executed; Full Regression Suite Passing  
**Test Suite**: 230 passed, 2 skipped, 0 failed (84.08s)

---

### 1. Project Overview & Research Claim

RENA (Reliable Evaluation for Neural Unlearning Audits) provides a controlled auditing framework for machine unlearning and application data deletion claims.

#### Research Claim Under Evaluation:
> *Behavioral unlearning audits that rely on direct prompts and refusal checks can certify suppressed-but-retained knowledge as forgotten. A protocol with baseline-retrievability gating, matched reference and counterfactual controls, diverse probes and validated scoring lowers the false-pass rate, at a quantified cost in inconclusive outcomes and audit budget.*

---

### 2. Empirical Verification & Evidence Reconciliation

| Evidence Category | Artifact Path | Execution Details | Scientific Findings | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Neural Smoke Benchmark** | `results/testbed/smoke_audit.json` | `Qwen/Qwen2.5-0.5B-Instruct` on Apple Silicon MPS (FP16, seed 42). 5 facts $\times$ 5 conditions $\times$ 4 probe types $\times$ 3 reps. | Naive auditor falsely passes 40% of prompt-suppressed facts and 60% of refusal-tuned facts. Controlled RENA auditor detects residual recovery on 100% and 80% respectively. | **VERIFIED** |
| **Neural Oracle Benchmark** | `results/testbed/oracle_results.json` | Exact sequence log-likelihood & 10-step AdamW relearning attack on `Qwen2.5-0.5B-Instruct`. | Refusal tuning maintains target log-likelihood (-2.9756 vs unmitigated -1.4043) and recovers 80% recall in 2 steps. Gradient difference lowers log-likelihood (-14.4195) and resists 10-step relearning. | **VERIFIED** |
| **Hosted API Case Study** | `results/case_studies/gemini_flash_lite_audit.json` | `gemini-3.5-flash-lite` via HTTP headers (`temperature=0.0`, `seed=42`). 2 facts $\times$ 4 conditions $\times$ 4 probe types $\times$ 2 reps. | Direct queries on `instruct_to_forget` yield 0% disclosure (naive false pass), while multi-turn conversational probes achieve 100% target extraction. | **VERIFIED** |
| **Human Annotation Benchmark** | `data/annotation/validation_400_unlabeled.csv` | 400 blind model outputs prepared for dual independent annotation per `docs/ANNOTATION_GUIDE.md`. | Synthetically balanced dev/test set prepared; awaiting independent human ratings. Never simulated or fabricated. | **AWAITING_HUMAN_ANNOTATION** |
| **Archived Legacy Results** | `archive/legacy_results/` | Historical pilot runs (seeds 42, 43, 44) from early exploratory phases. | Preserved for historical provenance; isolated from headline claims. | **HISTORICAL_ARCHIVED** |
| **Unit Sanity Simulations** | `docs/appendix_simulation/` | Simulated mock models for software test harness checks. | Labeled strictly as `SIMULATION, NOT EMPIRICAL EVIDENCE`; excluded from empirical claims. | **SIMULATION_ONLY** |

---

### 3. Claims Ledger Reconciliation

- **Claim C1 (Multi-Turn Elicitation Lowers False-Pass Rate)**: `SUPPORTED_BY_EVIDENCE` (Hosted Gemini study + Qwen2.5-0.5B neural weights on MPS).
- **Claim C2 (Baseline-Retrievability Gating Prevents False Unlearning)**: `SUPPORTED_BY_EVIDENCE` (Verified 0.0% recall on unexposed base weights; $k$-of-$n$ gating established 100% pre-mitigation recall before auditing).
- **Claim C3 (Matched Reference Controls Prevent False Attribution)**: `SUPPORTED_BY_EVIDENCE` (Group B clean base model evaluated under full RENA protocol returned 0/5 disclosures, $FPR=0.0\%$).
- **Claim C4 (Oracle Relearning Vulnerability Signature)**: `SUPPORTED_BY_EVIDENCE` (Refusal tuning re-acquires 80% recall in 2 AdamW steps, whereas gradient difference unlearning requires $>10$ steps).

---

### 4. How to Run the Code

#### Run Core Test Suite:
```bash
.venv/bin/pytest tests/ -v
```

#### Run Secret Scanner Gate:
```bash
.venv/bin/python scripts/rena_ci_gate.py --scan-secrets
```

#### Run Neural Smoke Benchmark (Apple Silicon MPS or CPU):
```bash
.venv-llm/bin/python scripts/run_neural_smoke_comparison.py
```

#### Run Neural Oracle Attack Benchmark:
```bash
.venv-llm/bin/python scripts/run_neural_oracle_benchmark.py
```

---

### 5. Honest Scientific Limitations
1. A black-box behavioral audit cannot mathematically prove physical parameter erasure, provider database zeroization, or cache purge.
2. Negative findings under RENA mean strictly that no residual retrieval was observed under the pre-registered probe budget.
3. Multi-turn probing increases inference volume by $4\times$ to $6\times$ compared to naive single-turn audits.
