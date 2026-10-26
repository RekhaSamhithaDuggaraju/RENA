# RENA Empirical Evidence Index

This index registers all empirical model evaluations, audit logs, and experimental records in RENA.
In accordance with Rule 0.1 (*Evidence or NOT_RUN*), this directory contains **only** code execution logs from real neural models and live provider endpoints. Simulated sanity checks are isolated in `docs/appendix_simulation/`, and legacy simulation runs are archived in `archive/legacy_results/`.

---

## 1. Active Empirical Datasets & Runs

| File Path | Model / Target | Protocol / Experiment | Date (UTC) | Description |
| :--- | :--- | :--- | :--- | :--- |
| `results/testbed/smoke_audit.json` | `Qwen/Qwen2.5-0.5B-Instruct` | Neural Testbed Smoke Study | 2026-10-26 | 5 facts $\times$ 5 conditions (`reference`, `retained`, `suppress_prompt`, `suppress_refusal`, `graddiff`) $\times$ 4 probe types $\times$ 3 reps on Apple Silicon MPS. Compares Naive vs RENA auditor false pass rates under baseline gating. |
| `results/testbed/oracle_results.json` | `Qwen/Qwen2.5-0.5B-Instruct` | Neural Oracle Attack Benchmark | 2026-10-26 | Exact sequence log-likelihood across 4 conditions and AdamW relearning curves (10 steps) comparing `suppress_refusal` (recovers in 2 steps) vs `graddiff` (resists re-acquisition). |
| `results/case_studies/gemini_flash_lite_audit.json` | `gemini-3.5-flash-lite` | Multi-turn In-Context Audit Case Study | 2026-10-26 | 2 facts $\times$ 4 conditions $\times$ 4 probe types $\times$ 2 reps. Tests direct, paraphrase, indirect, and multi-turn probes under `retract_by_removal` and `instruct_to_forget`. |
| `results/forgetting_audits/73459a4be5594c47a1d189a82858c513.json` | Live Target Endpoint | User-Controlled Forgetting Audit | 2026-10-24 | 5 questions $\times$ 3 repetitions. Validates baseline gating, counterfactual controls, and atomic report persistence. |
| `results/MANIFEST.json` | All Artifacts | Cryptographic Ledger | 2026-10-26 | SHA-256 manifest verifying file integrity and detecting untracked modifications. |

---

## 2. Evidence Verification Command

To verify the integrity and non-mock status of all results:
```bash
pytest tests/test_results_integrity.py
```

