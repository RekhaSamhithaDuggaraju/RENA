# RENA implementation update — 2026-10-23

## Added in this working copy

- New authenticated FastAPI router: `backend/app/routers/forgetting_audits.py`.
- New frontend workspace: `frontend/app/forgetting-audit/page.tsx`, linked from the existing LLM lab.
- User-created audits with target type, exposure method, tier label, and forgetting-operation notes.
- Dataset upload for CSV/JSON/JSONL (5 MiB, 500 records max), plus manual fact entry.
- Probe generation, custom questions, editing before baseline, automated adapter calls for Ollama/Gemini/OpenAI-compatible chat completions, and guided/manual response entry for consumer UIs.
- Baseline validity gate: every target fact must have at least one lexically flagged baseline response before a forget operation can be recorded.
- Three default trials per automated probe; reports preserve responses and show a cautious lexical-screen verdict.
- Owner-scoped JSON records stored under `results/forgetting_audits/`.
- Unit tests for dataset parsing/limits and lexical scoring.

## Validation performed

- Python compile check passed for the backend app.
- Focused regression suite passed: `11 passed` (`tests/test_forgetting_audits.py`, `tests/test_llm_jobs_api.py`, `tests/test_paired_niah_aggregate.py`).
- Frontend build could not be verified in this environment because dependency installation did not finish; `next` was unavailable. Run `npm ci` and `npm run build` on the user's Mac before treating the frontend as release-ready.
- The full pytest suite did not finish within the environment's time limit; no claim is made that all tests pass.

## Important scientific limits

This is an implementation milestone, not a validated research result. Current scoring is lexical and requires human review; indirect semantic leaks can be missed. The API exposure step does not prove persistence. User must perform the actual forget/delete operation for systems without a supported deletion API. Black-box results do not prove deletion from weights or provider infrastructure. No model-agnostic automatic forgetting mechanism is claimed.

### Paired control follow-up
- Added optional never-exposed reference configuration for Ollama, OpenAI-compatible API, Gemini API, or guided/manual control.
- Added `never_exposed` phase to API/local runner and manual response flow.
- Reports now include per-question matched comparisons and aggregate reference vs post-forget lexical flags.
- Added reference controls and a dedicated run button in the forgetting-audit UI.
- Focused regression: 11 passed; backend compileall passed. Frontend build still needs verification in an environment with installed dependencies.
- Interpretation is deliberately descriptive; reference must genuinely never have received the facts and be otherwise matched as closely as possible.
