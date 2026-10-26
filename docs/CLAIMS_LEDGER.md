# RENA Scientific Claims Ledger (v1.0)

This ledger explicitly registers every scientific claim made by RENA, links each claim directly to empirical code, artifacts, and test logs, and defines clear falsification criteria and boundaries.

---

## 1. Traceability Table

| Claim ID | Core Proposition | Status | Supporting Artifacts & Code | Evidence & Statistical Support | Scope & Boundary |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C1** | Zero-shot direct prompt audits fail to detect suppressed-but-retained latent knowledge; multi-turn elicitation lowers false-pass rates. | **PARTIAL (2-fact case study)** | `scripts/reproduce_gemini_case.py`<br>`results/case_studies/gemini_flash_lite_audit.json`<br>`tests/test_phase1_engine_upgrades.py` | Direct probe on `instruct_to_forget` shows 0% disclosure; multi-turn conversational probe extracts target with 100% disclosure. | Applies to in-context and prompt-level suppression defenses on `gemini-3.5-flash-lite`. Does not claim weight-level erasure on hosted commercial APIs. Full multi-fact neural testbed pending. |
| **C2** | Baseline-retrievability gating ($k$-of-$n$) prevents false unlearning certifications on inherently unretrievable facts. | **NOT_RUN** | `core/stats/verdict_engine.py`<br>`backend/app/routers/forgetting_audits.py`<br>`tests/test_verdict_engine.py` | Algorithm and gate logic implemented and unit-tested; empirical neural testbed comparison of FPR with vs. without gate pending WP3 execution. | Requires configurable $k$-of-$n$ (default $k=2, n=3$). |
| **C3** | Matched unexposed reference controls (Group B) prevent false-positive leakage attribution on pretraining knowledge. | **NOT_RUN** | `backend/app/routers/forgetting_audits.py`<br>`core/testbed_real/unlearn.py` | Control logic implemented; empirical separation between pretraining recall and unlearned recall on neural testbed pending WP3 execution. | Dependent on availability of matched pre-exposure model or clean session. |
| **C4** | Behavioral suppression models exhibit vulnerability to relearning attacks ($\le 2$ steps), whereas weight unlearning requires $\ge 5$ steps. | **NOT_RUN** | `core/testbed_real/oracle.py`<br>`core/testbed_real/implant.py` | Oracle code implemented in `core/testbed_real/oracle.py`; AdamW retraining curves on fine-tuned weights pending WP3 execution. | Evaluated on `Qwen2.5-0.5B-Instruct` with LoRA adapter. |

---

## 2. Explicit Negative Results & Boundaries

1. **Non-Claim on Server-Side Erasure**: RENA's black-box evaluations cannot prove physical deletion from server memory, model weights, or secondary caching infrastructure of hosted third-party commercial APIs.
2. **Non-Claim of Universal Unlearning**: A verdict of `NO_RETRIEVAL_OBSERVED_UNDER_PROTOCOL` applies strictly to the tested facts, probe distributions, and trial repetitions. It is not an absolute mathematical proof of non-recoverability under arbitrary jailbreaks.
3. **Simulation Boundary**: All simulated datasets (e.g. `SIMULATION_SANITY_CHECKS.md`) are unit-level software sanity checks and are never presented as empirical validation.
