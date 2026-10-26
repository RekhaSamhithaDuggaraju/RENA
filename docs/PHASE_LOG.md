# RENA Phase Log

This log tracks every task executed under the RENA corrective and research-grade upgrade plan.
Every entry records the task ID, timestamp, git commit, commands executed, full test results, and any skipped or blocked items.

> [!NOTE]
> Timestamps recorded prior to WP0 were approximate development benchmarks. All timestamps from WP0 onward are captured directly from system UTC (`date -u`).

---

## Phase 0: Overclaim Cleanup & Simulation Relabeling
- **Task ID**: P0.1
- **Timestamp**: 2026-10-26T14:40:00Z (approximate)
- **Goal**: Relabel simulation modules, remove simulated numbers from claim/report, strip promotional wording, and rename synthetic regression sets.
- **Commands Run**:
  - `pytest tests/test_forgetting_audits.py tests/test_judge_validation.py tests/test_ground_truth_testbed.py` -> 39 passed in 0.36s.
  - Relabeled `core/testbed/ground_truth_testbed.py` with `SIMULATION, NOT EMPIRICAL EVIDENCE`.
  - Moved simulation tables to `docs/appendix_simulation/SIMULATION_SANITY_CHECKS.md`.
  - Cleared simulated E1-E4 numbers from `docs/PAPER_EVALUATION_REPORT.md` (marked `NOT_RUN`).
  - Archived `results/forgetting_audits/60940aa38ae24df8b2a3c255de485137.json` to `archive/` as `UNREPRODUCIBLE`.
  - Created `docs/OPEN_QUESTIONS.md`.
- **Status**: DONE

---

## Phase 1: Audit Engine Upgrades & Real Empirical Case Study
- **Task IDs**: T1.1 – T1.12
- **Timestamp**: 2026-10-26T15:00:00Z (approximate)
- **Goal**:
  - Implement `TargetCallResult` with conversational tracking, latency measurements, and pinned decoding parameters (`temperature=0.0`, `top_p=1.0`, `seed=42`).
  - Enforce explicit exposure modes (`in_context`, `multi_turn_session`, `rag_store`, `provider_memory`, `weights`) and reject `weights` on hosted commercial APIs.
  - Implement `build_probe_messages` for multi-turn conversational message construction across `baseline`, `never_exposed` (Group B), `exposed_no_forget` (Group C), and `post_forget` (Group A: `retract_by_removal`, `instruct_to_forget`).
  - Implement $k$-of-$n$ baseline retrievability gate and atomic audit persistence with file locking.
  - Question generation leak detector (`check_question_leak`) and approval workflow.
  - Re-run empirical Gemini audit on `gemini-3.5-flash-lite` via `scripts/reproduce_gemini_case.py`.
- **Commands & Tests Run**:
  - `pytest tests/test_forgetting_audits.py tests/test_verdict_engine.py tests/test_atomic_save.py tests/test_phase1_engine_upgrades.py` -> 35 passed.
  - `.venv/bin/python scripts/reproduce_gemini_case.py` -> Saved real empirical audit logs to `results/case_studies/gemini_flash_lite_audit.json`.
  - Findings: Direct probes under `instruct_to_forget` exhibited 0% disclosure (false pass), while multi-turn conversational probes uncovered 100% residual disclosure.
- **Status**: DONE

---

## Phase 2: Normalization, Scorer Rubric, and Human Benchmark Preparation
- **Task IDs**: T2.1 – T2.6
- **Timestamp**: 2026-10-26T15:15:00Z (approximate)
- **Goal**:
  - Create canonical text normalization module `core/scoring/normalize.py` handling Unicode NFKC, case folding, punctuation normalization, hyphenated code preservation, and alias matching.
  - Write comprehensive unit tests in `tests/test_normalize.py`.
  - Author versioned semantic judge rubric in `prompts/judge_v1.txt`.
  - Formulate human annotation protocol `docs/ANNOTATION_GUIDE.md` detailing 4-class taxonomy, edge-case decision rules, and inter-rater agreement adjudication.
  - Generate 400-item blind validation dataset `data/annotation/validation_400_unlabeled.csv` for 2 independent human annotators.
  - Save hidden ground-truth key `data/annotation/validation_400_ground_truth.json` for Cohen's $\kappa$ and precision/recall evaluation.
- **Commands & Tests Run**:
  - `pytest tests/test_normalize.py` -> 5 passed.
  - `.venv/bin/python scripts/generate_validation_dataset.py` -> Generated 400-item dataset.
- **Status**: AWAITING_HUMAN_ANNOTATION (Benchmark dataset and protocol generated; awaiting independent human ratings).

---

## Phase 3: Real Ground-Truth Neural Testbed
- **Task IDs**: T3.1 – T3.5
- **Timestamp**: 2026-10-26T15:20:00Z (approximate)
- **Goal**:
  - Author smoke testbed configuration in `configs/testbed/smoke.yaml` using `Qwen2.5-0.5B-Instruct` on Apple Silicon MPS.
  - Define evaluation facts in `core/testbed_real/dataset.py`.
  - Implement LoRA knowledge implantation with baseline recall gate verification in `core/testbed_real/implant.py`.
  - Implement unlearning methods (`ga`, `graddiff`, `npo`, `suppress_refusal`, `suppress_prompt`, `over_unlearned`, `retained`, `reference`) in `core/testbed_real/unlearn.py`.
  - Implement independent evaluation oracle (sequence log-likelihood gap & relearning attack benchmark) in `core/testbed_real/oracle.py`.
  - Author self-contained Colab GPU notebook `notebooks/testbed_colab.ipynb`.
- **Status**: PARTIAL (Code and pipelines implemented; neural execution on weights pending WP3 execution).

---

## WP0: Hygiene and Corrections
- **Task IDs**: T0.1 – T0.6
- **Timestamp**: 2026-10-26T15:27:00Z
- **Goal**:
  - T0.1: Moved `results/testbed/testbed_results.json` and simulation outputs to `docs/appendix_simulation/testbed_simulation_results.json` labeled `[SIMULATION, NOT EMPIRICAL EVIDENCE]`. Created `tests/test_results_integrity.py` asserting every `results/**` file has `"mock": false` or provider/model ID.
  - T0.2: Archived root clutter (`backups/`, `reports/`, legacy runner scripts, old release notes) into `archive/` with `archive/README.md`. Maintained clean `results/EVIDENCE_INDEX.md`.
  - T0.3: Rewrote `docs/PHASE_LOG.md` with honest timestamps (noting prior approximations) and reopened Phase 3 as `PARTIAL`.
  - T0.4: Updated `docs/CLAIMS_LEDGER.md`: marked C1 as `PARTIAL (2-fact case study)`, C2 as `NOT_RUN`, C3 as `NOT_RUN`, C4 as `NOT_RUN`.
  - T0.5: Implemented secret scanner in `scripts/rena_ci_gate.py` with `--scan-secrets` checking headers and files for API keys. Verified clean scan.
  - T0.6: Renamed `retract_by_removal` output label in `backend/app/routers/forgetting_audits.py` and case studies to `MECHANICAL_CONTROL (text removed from context)`.
- **Commands & Tests Run**:
  - `pytest tests/test_results_integrity.py` -> 4 passed.
  - `.venv/bin/python scripts/rena_ci_gate.py --scan-secrets` -> PASSED (0 secrets detected).
- **Status**: DONE
