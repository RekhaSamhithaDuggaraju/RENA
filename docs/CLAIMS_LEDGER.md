# RENA Scientific Claims Ledger (v1.0)

This ledger explicitly registers every scientific claim made by RENA, links each claim directly to empirical code, artifacts, and test logs, and defines clear falsification criteria and boundaries.

---

## 1. Traceability Table

| Claim ID | Core Proposition | Status | Supporting Artifacts & Code | Evidence & Statistical Support | Scope & Boundary |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C1** | Zero-shot direct prompt audits fail to detect suppressed-but-retained latent knowledge; multi-turn elicitation lowers false-pass rates. | **SUPPORTED_BY_EVIDENCE** | `results/case_studies/gemini_flash_lite_audit.json`<br>`results/testbed/smoke_audit.json`<br>`scripts/run_neural_smoke_comparison.py` | 1. Hosted API: Direct probe on `instruct_to_forget` yields 0% disclosure; multi-turn yields 100%.<br>2. Local neural weights (`Qwen2.5-0.5B-Instruct` on MPS): Naive auditor falsely passes 40% on `suppress_prompt` and 60% on `suppress_refusal`. Full RENA auditor uncovers residual retrieval on 100% and 80% respectively. | Evaluated on `gemini-3.5-flash-lite` (in-context) and `Qwen2.5-0.5B-Instruct` (LoRA weights on MPS). Black-box behavioral probes cannot prove physical erasure on provider servers. |
| **C2** | Baseline-retrievability gating ($k$-of-$n$) prevents false unlearning certifications on inherently unretrievable facts. | **SUPPORTED_BY_EVIDENCE** | `results/testbed/smoke_audit.json`<br>`core/stats/verdict_engine.py`<br>`tests/test_verdict_engine.py` | Synthetic forget canaries verified at 0.0% recall on unexposed base weights. Gating ($k=2, n=3$) verified 5/5 valid facts on implanted model before admitting them to mitigation auditing, excluding unlearned base artifacts from false certification. | Evaluated on $n=5$ synthetic facts in neural smoke testbed. Larger $n$ benchmarks pending additional compute. |
| **C3** | Matched unexposed reference controls (Group B) prevent false-positive leakage attribution on pretraining knowledge. | **SUPPORTED_BY_EVIDENCE** | `results/testbed/smoke_audit.json`<br>`core/testbed_real/unlearn.py`<br>`tests/test_verdict_engine.py` | Clean base model (Group B) evaluated under full RENA protocol returned 0/5 disclosures (0.0% false-positive recovery, all 5 facts classified as `NO_RETRIEVAL_OBSERVED_UNDER_PROTOCOL`). Retained utility remained 80.0%. | Requires availability of matched unexposed base weights or pre-exposure session. |
| **C4** | Behavioral suppression models exhibit vulnerability to relearning attacks ($\le 2$ steps), whereas weight unlearning requires more steps. | **SUPPORTED_BY_EVIDENCE** | `results/testbed/oracle_results.json`<br>`scripts/run_neural_oracle_benchmark.py`<br>`core/testbed_real/oracle.py` | 1. Exact log P(target \| question): reference = -38.0156, retained = -1.4043, suppress_refusal = -2.9756, graddiff = -14.4195.<br>2. Relearning attack: `suppress_refusal` recovered 80% recall at step 2 (`rapid_reacquisition_suppression_signature`), while `graddiff` remained at 60% after 10 steps (`gradual_reacquisition`). | Evaluated on `Qwen2.5-0.5B-Instruct` LoRA weights. Relearning trajectory evaluated over 10 AdamW steps at $\text{lr}=10^{-4}$. |

---

## 2. Explicit Negative Results & Boundaries

1. **Non-Claim on Server-Side Erasure**: RENA's black-box evaluations cannot prove physical deletion from server memory, model weights, or secondary caching infrastructure of hosted third-party commercial APIs.
2. **Non-Claim of Universal Unlearning**: A verdict of `NO_RETRIEVAL_OBSERVED_UNDER_PROTOCOL` applies strictly to the tested facts, probe distributions, and trial repetitions. It is not an absolute mathematical proof of non-recoverability under arbitrary jailbreaks.
3. **Simulation Boundary**: All simulated datasets (e.g. `SIMULATION_SANITY_CHECKS.md`) are unit-level software sanity checks and are never presented as empirical validation.
