# RENA Phase Log

This log tracks every task executed under the RENA corrective and research-grade upgrade plan.
Every entry records the task ID, timestamp, git commit, commands executed, full test results, and any skipped or blocked items.

---

## Phase 0: Overclaim Cleanup & Simulation Relabeling
- **Task ID**: P0.1
- **Timestamp**: 2026-10-26T14:40:00Z
- **Goal**: Relabel simulation modules, remove simulated numbers from claim/report, strip promotional wording, and rename synthetic regression sets.
- **Commands Run**:
  - `pytest tests/test_forgetting_audits.py tests/test_judge_validation.py tests/test_ground_truth_testbed.py` -> 39 passed in 0.36s.
  - Relabeled `core/testbed/ground_truth_testbed.py` with `SIMULATION, NOT EMPIRICAL EVIDENCE`.
  - Moved simulation tables to `docs/appendix_simulation/SIMULATION_SANITY_CHECKS.md`.
  - Cleared simulated E1-E4 numbers from `docs/PAPER_EVALUATION_REPORT.md` (marked `NOT_RUN`).
  - Archived `results/forgetting_audits/60940aa38ae24df8b2a3c255de485137.json` to `archive/` as `UNREPRODUCIBLE`.
  - Created `docs/OPEN_QUESTIONS.md`.
- **Status**: DONE

