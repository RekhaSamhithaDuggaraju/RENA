# RENA V22 — Final Hardening Release

V22 is a hardening-and-validation release. It prioritizes security, authoritative verification, certificate integrity, reproducibility, and honest release validation over new feature volume.

## Canonical decision flow

Audit → RENA-X → PASS / FAIL / INCONCLUSIVE → evidence package → signed certificate (PASS only) → public verification.

## Security

- Resource-bearing routes require authentication and ownership/permission checks.
- Production mode requires `RENA_AUTH_SECRET`, `RENA_CERTIFICATE_SECRET`, and `RENA_DATABASE_URL`.
- Asset kinds are allowlisted and filesystem paths are contained within the project root.
- Certificates use HMAC signatures and public verification exposes only safe certificate metadata.
- The legacy certificate endpoint returns `410 LEGACY/DEPRECATED`.

## Reproducibility

Audits record available RENA version, commit, Python/OS/CPU/GPU metadata, dataset summary, configuration hash, dependency lock hash, and UTC timestamp. `model_hash` remains `not_recorded` unless a real model asset is part of the executed audit; it is never fabricated.

## Validation actually completed for this release

- Python compile/AST validation: PASS
- V22 certificate/security/verdict regression tests: **16 passed** in the final focused run
- Full V20/V21/V22 hardening set: **21 passed + 14 subtests**
- Clean SQLite Alembic `upgrade head`: PASS; all application tables created
- ZIP integrity: PASS

## Validation not claimed

- The complete 137+ test suite did not finish within the available execution window; therefore V22 does **not** claim 137/137 passed.
- The Next.js production build was not run because `node_modules` is not present and network installation is unavailable in the validation environment.
- PostgreSQL and Docker end-to-end deployment were not executed in this environment.
- A full computational audit can exceed the execution window in this environment; no such run is represented as a completed benchmark result.

See `V22_RELEASE_CHECKLIST.md`, `SECURITY.md`, and `VERIFICATION_SCOPE.md` for scope and limitations.
