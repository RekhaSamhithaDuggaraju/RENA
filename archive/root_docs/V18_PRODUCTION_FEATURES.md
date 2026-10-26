# RENA V18 — Production Application Layer

V18 adds the application/operations layer on top of the RENA audit engine.

## Included
- Optional account authentication using PBKDF2 password hashing and signed bearer tokens.
- Project ownership isolation when `RENA_AUTH_REQUIRED=1`.
- PostgreSQL persistence with a forward-compatible owner-column migration.
- Audit/job timeline events and project event history.
- Job retry/cancel lifecycle and configurable resource limits.
- Immutable audit certificates with SHA-256 verification endpoint.
- CI/CD deployment gate: `scripts/rena_ci_gate.py AUDIT_ID` (exit 0 on pass, 1 on block).
- Audit templates, duplication, schedules, comparison and notification configuration endpoints.
- Certificate verification: `/production/certificates/verify/{hash}`.
- Webhook/SMTP readiness via environment configuration (delivery remains optional).

## Production environment
Set at minimum:

```bash
RENA_DATABASE_URL=postgresql://rena:rena@localhost:5432/rena
RENA_AUTH_REQUIRED=1
RENA_AUTH_SECRET=<long-random-secret>
RENA_DATA_ROOT=/var/lib/rena
RENA_MAX_UPLOAD_MB=256
RENA_MAX_CONCURRENT_JOBS=2
RENA_JOB_TIMEOUT_SECONDS=1800
RENA_CORS_ORIGINS=https://your-rena.example
```

For development, authentication remains optional by default.

## CI/CD

```bash
python scripts/rena_ci_gate.py <AUDIT_ID> --policy strict
```

Exit code `0` allows deployment; exit code `1` blocks deployment; `2` indicates an infrastructure/API error.
