# RENA application integration update

## Scope

This update aligns the classical interactive application with the six algorithms already registered in the v2 experiment runner. It does not implement an LLM unlearning baseline or Needle-in-a-Haystack (NIAH); those remain separate planned evaluation capabilities.

## Changes

- Added ECHO to `core.pipeline.run_audit` using the existing `core.echo_controller.echo_unlearn` implementation.
- Standardized PRISM and ECHO method-specific diagnostics under the result key `method_trace`.
- Added PRISM and ECHO to the backend request schema and frontend `Algorithm` type.
- Added PRISM and ECHO as dashboard launch options and landing-page descriptions.
- Persisted optional `method_trace` JSON in the audit ORM and exposed it in the API response and audit detail page.
- Added validation for `forget_fraction` in `(0, 0.5]`, avoiding unsupported edge cases in the current stratified split workflow.
- Added end-to-end pipeline coverage for PRISM and ECHO.

## Verification

`python3 -m unittest discover -q` completed with 34 tests passing. `python3 -m compileall -q core backend tests` completed successfully. Pydantic request checks accepted all six method names and rejected forget fractions of 0 and 0.6.

The backend's SQLAlchemy dependencies are not installed in the build environment, so a live API/database integration test and frontend production build were not run here. For an existing PostgreSQL database, apply the migration documented in the README because `Base.metadata.create_all()` does not add columns to existing tables.

## Interpretation

Method traces are diagnostic metadata, not independent evidence that unlearning succeeded. The existing verification modules and consensus remain responsible for the audit outcome. The classical digits/MLP track must not be represented as an LLM unlearning benchmark.
