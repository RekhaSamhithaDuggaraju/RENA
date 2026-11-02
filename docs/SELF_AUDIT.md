# RENA V22 — Self-Audit & Evidence Ledger

**Document Version**: 1.0.0
**Audit Date**: 2026-10-27
**Status Vocabulary**: `MET` / `PARTIAL` / `NOT_MET` / `AWAITING_HUMAN` / `NOT_RUN`
**Strict Compliance**: Rule 1 (Raw results are source of truth), Rule 3 (No simulation presented as validation), Rule 6 (No promotional inflation).

---

## 1. Core Research & Experimental Audit

| Criterion / Deliverable | Status | Evidence Path / Command | Description & Findings |
| :--- | :---: | :--- | :--- |
| **Evidence Consistency & Number Synchronization** | **MET** | `scripts/check_claims_vs_results.py`<br>`tests/test_evidence_consistency.py` | Automated checker verifies that all quantitative values in `paper/main.tex`, `docs/CLAIMS_LEDGER.md`, `docs/PHASE_LOG.md`, and `AUDIT_REPORT.md` match `results/testbed/*.json` exactly. Exits 0 on agreement; exits 1 on tampering. |
| **Exact-Answer Baseline Retrievability Gate** | **MET** | `core/stats/verdict_engine.py`<br>`tests/test_verdict_engine.py` | Implements exact-answer $k$-of-$n$ gate ($k=2, n=3$). Classifies digit transpositions (e.g. `OBSIDIAN-9842` vs target `9482`) as `near_miss` (`score=0.0`, `exact=False`). Rejects near-misses from baseline recall. |
| **Near-Miss Scorer Discrimination** | **MET** | `core/scoring/semantic_scorer.py`<br>`tests/test_verdict_engine.py` | Adds Levenshtein distance $\le 2$ and code prefix regex. Partials and near-misses are routed to `INCONCLUSIVE` / human review, never driving residual retrieval or false unlearning passes. |
| **Independent Weight-Level & Relearning Oracle** | **MET** | `core/testbed_real/oracle.py`<br>`results/testbed/oracle_results.json` | Computes exact sequence log-likelihood $\log P(\text{target} \mid \text{question})$. Evaluates AdamW relearning with a never-exposed `reference` base model control, measuring same-fact vs held-out recovery trajectories. |
| **Empirical Hypothesis Pivot ($H_{\text{main}}$)** | **MET** | `docs/NEGATIVE_RESULT.md`<br>`docs/PREREGISTRATION.md` | Formally registered finding that multi-probe behavioral audits fail to detect refusal suppression (0/5 exact disclosures), whereas oracle confirms latent retention (mean LL $-2.4370$ vs reference $-38.0156$, relearns in 2 steps). Proves need for white-box audits. |
| **Scaled Confirmatory Corpus ($N \ge 30$)** | **PARTIAL** | `core/testbed_real/dataset.py`<br>`results/testbed/confirmatory_audit.json` | 30-fact held-out corpus (`canary_001`–`canary_030`) constructed and preregistered. Full empirical execution on real model weights remains pending further GPU compute. |
| **Independent Dual-Rater Human Annotation** | **AWAITING_HUMAN** | `data/annotation/rater_1_blind.csv`<br>`data/annotation/rater_2_blind.csv` | Blinded rater sheets generated from real testbed and case study responses with model verdicts hidden. Awaiting independent human labeling per Rule 8. |
| **Hosted Model Case Study (Gemini)** | **PRELIMINARY** | `results/case_studies/gemini_flash_lite_audit.json` | Real-API audit conducted on `gemini-2.0-flash-lite` across 2 facts with Wilson CIs $[34.2\%, 100\%]$. Expanded multi-provider scaling pending external API quota. |
| **Second Architecture Evaluation** | **NOT_RUN** | `paper/main.tex` Section 5 | Single-model limitation on `Qwen2.5-0.5B-Instruct` is explicitly declared. Second larger model (1.5B/3B) requires external GPU/Colab execution. |

---

## 2. Engineering, Full-Stack Application & Operations

| Component / Subsystem | Status | Evidence Path / Command | Live Verification |
| :--- | :---: | :--- | :--- |
| **Next.js Frontend Build** | **MET** | `frontend/` (`npm run build`) | 17/17 routes compiled cleanly with 0 TypeScript or linting errors. Running on `http://localhost:3000`. |
| **FastAPI REST Backend** | **MET** | `backend/` (`http://localhost:8000/docs`) | 8 routers active, CORS preflight isolation verified, running on `http://localhost:8000`. |
| **PostgreSQL 16 Persistence** | **MET** | `docker compose ps`<br>`backend/app/models/` | 10 database tables created and healthy on port `5432`. |
| **Two-Track Forgetting Audit Lab** | **MET** | `/forgetting-audit` | Complete 8-step wizard supporting Track A (Model Unlearning) and Track B (Memory Deletion). |
| **MIRA Memory Scoring Engine** | **MET** | `/mira`<br>`POST /research/mira/score` | Transparent 5-attribute memory utility ranking operational. |
| **Cryptographic HMAC Certificates** | **MET** | `/certificates`<br>`/verify/[hash]` | Issues signed SHA-256 certificates only for genuine `PASS`. Strictly rejects failed audits (`409 Conflict`). |
| **Vector PDF Reports** | **MET** | `/audits/{id}/report.pdf`<br>`/production/certificates/{id}/pdf` | ReportLab-generated print-ready audit reports verified. |
| **Automated CI/CD Deployment Gate** | **MET** | `scripts/rena_ci_gate.py` | Command-line gate blocks deployment on unverified or failed unlearning audits. |
| **Unit & Integration Test Suite** | **MET** | `pytest tests/` | 239 passed, 0 failed across test suite. |
| **Secret Scanning & Security** | **MET** | `scripts/rena_ci_gate.py --scan-secrets` | Zero API key fragments or unmasked credentials present in source files. |
