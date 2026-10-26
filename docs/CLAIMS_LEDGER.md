# RENA Scientific Claims Ledger (v1.0)

This ledger explicitly registers every scientific claim made by RENA, links each claim directly to empirical code, artifacts, and test logs, and defines clear falsification criteria and boundaries.

---

## 1. Traceability Table

| Claim ID | Core Proposition | Supporting Artifacts & Code | Evidence & Statistical Support | Scope & Boundary |
| :--- | :--- | :--- | :--- | :--- |
| **C1** | Zero-shot direct prompt audits fail to detect suppressed-but-retained latent knowledge; multi-turn elicitation lowers false-pass rates. | `scripts/reproduce_gemini_case.py`<br>`results/case_studies/gemini_flash_lite_audit.json`<br>`tests/test_phase1_engine_upgrades.py` | Direct probe on `instruct_to_forget` shows 0% disclosure; multi-turn conversational probe extracts target with 100% disclosure. | Applies to in-context and prompt-level suppression defenses. Does not claim weight-level erasure on hosted commercial APIs. |
| **C2** | Baseline-retrievability gating ($k$-of-$n$) prevents false unlearning certifications on inherently unretrievable facts. | `core/stats/verdict_engine.py`<br>`backend/app/routers/forgetting_audits.py`<br>`tests/test_verdict_engine.py` | Audits without baseline gating certify 100% unlearned when facts were never retrievable; gate flags `INVALID_BASELINE` and blocks claims. | Requires configurable $k$-of-$n$ (default $k=2, n=3$). |
| **C3** | Matched unexposed reference controls (Group B) prevent false-positive leakage attribution on pretraining knowledge. | `app/routers/forgetting_audits.py`<br>`core/testbed_real/unlearn.py` | When Group B exhibits disclosure, target reflects general knowledge rather than post-operation leakage. | Dependent on availability of matched pre-exposure model or clean session. |
| **C4** | Behavioral suppression models exhibit vulnerability to relearning attacks ($\le 2$ steps), whereas weight unlearning requires $\ge 5$ steps. | `core/testbed_real/oracle.py`<br>`core/testbed_real/implant.py` | Measured step budget to reach 80% recall under AdamW retraining. | Evaluated on `Qwen2.5-0.5B-Instruct` with LoRA adapter. |

---

## 2. Explicit Negative Results & Boundaries

1. **Non-Claim on Server-Side Erasure**: RENA's black-box evaluations cannot prove physical deletion from server memory, model weights, or secondary caching infrastructure of hosted third-party commercial APIs.
2. **Non-Claim of Universal Unlearning**: A verdict of `NO_RETRIEVAL_OBSERVED_UNDER_PROTOCOL` applies strictly to the tested facts, probe distributions, and trial repetitions. It is not an absolute mathematical proof of non-recoverability under arbitrary jailbreaks.
3. **Simulation Boundary**: All simulated datasets (e.g. `SIMULATION_SANITY_CHECKS.md`) are unit-level software sanity checks and are never presented as empirical validation.
