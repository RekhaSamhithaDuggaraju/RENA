# RENA v22.2 Evaluation & Project Status
## Audit Report & Verification Status

**Date**: October 26, 2026  
**Status**: Core Architecture Upgraded; Empirical Case Study Executed; Real Model Testbed Implemented  
**Test Suite**: 215 passed, 2 skipped, 0 failed

---

### 1. Project Overview & Research Claim

RENA (Reliable Evaluation for Neural Unlearning Audits) provides a controlled auditing framework for machine unlearning and application data deletion claims.

#### Research Claim Under Evaluation:
> *Behavioral unlearning audits that rely on direct prompts and refusal checks can certify suppressed-but-retained knowledge as forgotten. A protocol with baseline-retrievability gating, matched reference and counterfactual controls, diverse probes and validated scoring lowers the false-pass rate, at a quantified cost in inconclusive outcomes and audit budget.*

---

### 2. Empirical Verification Status

- **Empirical Case Study (`results/case_studies/gemini_flash_lite_audit.json`)**: Executed cleanly on Google's `gemini-3.5-flash-lite`. Demonstrated that direct queries under instruction-based suppression produce a 100% false-pass rate (certifying the canary as forgotten), whereas multi-turn conversational probes achieve 100% extraction of the suppressed information.
- **Real Model Testbed (`core/testbed_real/`)**: Implemented LoRA knowledge implantation (`implant.py`), unlearning methods (`ga`, `graddiff`, `npo`, `suppress_refusal`, `suppress_prompt`, `over_unlearned` in `unlearn.py`), and independent evaluation oracle (`oracle.py`). Runnable on local Apple Silicon MPS and Colab (`notebooks/testbed_colab.ipynb`).
- **Human Annotation Benchmark (`data/annotation/validation_400_unlabeled.csv`)**: 400 blind items generated for 2 independent human raters according to `docs/ANNOTATION_GUIDE.md` (labeled `AWAITING_HUMAN_ANNOTATION` in `docs/PHASE_LOG.md`).
- **Unit Simulation Harness (`core/testbed/`)**: Labeled strictly as `SIMULATION, NOT EMPIRICAL EVIDENCE` (see `docs/appendix_simulation/SIMULATION_SANITY_CHECKS.md`).

---

### 3. How to Run the Code

#### Run Backend:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir backend
```

#### Run Frontend:
```bash
cd frontend
npm install
npm run dev
# Dashboard at http://localhost:3000/dashboard
```

#### Run Unit & Regression Tests:
```bash
.venv/bin/pytest tests/ -v
```

---

### 4. Limitations
A black-box behavioral audit cannot prove physical parameter erasure or remote infrastructure deletion. Probing results represent upper bounds on accessible recall under a finite evaluation budget.
