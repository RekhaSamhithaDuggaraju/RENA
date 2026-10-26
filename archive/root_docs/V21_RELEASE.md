# RENA V21 — Correctness, Security & Proof Release

V21 is a hardening release. It intentionally adds little surface area and instead makes the existing application consistent with its verification claims.

## Authoritative decision

The application decision is **RENA-X** and is normalized to exactly three public states:

- `PASS`
- `FAIL`
- `INCONCLUSIVE`

Classical consensus remains supporting evidence. Certificates, CI gates, comparisons and audit cards use the RENA-X decision source.

## Canonical certificate

`POST /production/audits/{audit_id}/certificate` is the only current certificate issuance path. It issues only for an authoritative RENA-X `PASS`, and signs the certificate payload with an HMAC server secret. The old workspace certificate endpoint returns `410 LEGACY/DEPRECATED`.

Public verification is intentionally available through `/production/certificates/verify/{certificate_hash}`; private audit/project resources require authentication and ownership access.

## Authorization

Workspace, audits, projects, jobs, events, templates, schedules, comparisons and certificate issuance use authenticated owner/project checks. A cross-user API test verifies that User B cannot read User A's project or audit.

## Asset safety

Asset kinds and extensions are allowlisted. Uploaded paths are rooted under the project directory and traversal is rejected. Assets record SHA-256 provenance metadata.

## Explainability and evidence quality

Audit detail exposes the authoritative verdict, confidence label, evidence quality, supporting evidence, verifier disagreement, scope limitations and reproducibility metadata.

## Reproducibility

Certificate payloads record RENA version, Python version, OS, timestamp and dataset summary. The export-bundle manifest is available for audit reproduction/provenance.

## Validation performed for this release

- `python3 -m compileall -q core backend/app tests` — PASS
- targeted regression/security suite — **21 passed, 14 subtests passed**
- authenticated real RENA pipeline smoke run — PASS
- signed certificate creation + public verification — PASS
- cross-user authorization isolation — PASS
- asset upload traversal test — PASS
- ZIP integrity — required before release packaging

## Environment limitations

The full 137-test suite was collected successfully, but a complete run exceeded the available five-minute execution window. Therefore V21 does **not** claim 137/137 passing.

The frontend production build could not be executed in this build environment because `node_modules` was unavailable and `npm install` exceeded the available network/execution window. Docker/PostgreSQL external-service smoke tests were likewise not executed here. These are explicit release limitations, not hidden claims.
