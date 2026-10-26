# RENA v22.2 Evaluation & Project Status
## Audit Report & Verification Status

**Date**: October 26, 2026  
**Status**: Core Architecture Upgraded; Simulations Relabeled; Real Model Testbed Under Construction  
**Test Suite**: 195 passed, 2 skipped, 0 failed

---

### 1. Project Overview & Research Claim

RENA (Reliable Evaluation for Neural Unlearning Audits) provides a controlled auditing framework for machine unlearning and application data deletion claims.

#### Research Claim Under Evaluation:
> *Behavioral unlearning audits that rely on direct prompts and refusal checks can certify suppressed-but-retained knowledge as forgotten. A protocol with baseline-retrievability gating, matched reference and counterfactual controls, diverse probes and validated scoring lowers the false-pass rate, at a quantified cost in inconclusive outcomes and audit budget.*

---

### 2. Empirical Testbed Status

- **Real Model Testbed (`core/testbed_real/`)**: In active development. Using `Qwen2.5-0.5B-Instruct` on local Apple Silicon MPS and Colab T4.
- **Unit Simulation Harness (`core/testbed/`)**: Labeled strictly as `SIMULATION, NOT EMPIRICAL EVIDENCE` (see `docs/appendix_simulation/SIMULATION_SANITY_CHECKS.md`).
- **Human Annotation (`docs/ANNOTATION_GUIDE.md`)**: Awaiting human labeling of 400 real model outputs.

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
