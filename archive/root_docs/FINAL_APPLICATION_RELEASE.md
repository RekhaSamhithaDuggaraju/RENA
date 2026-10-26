# RENA Final Application Release

This release is application-first. The core audit engine remains deterministic; LLM components are explanatory/optional and do not determine the verdict.

## Product flow

Onboard → Manage projects/assets → Configure audit → Run → Monitor → Investigate → Compare → Remediate → Re-audit → Certify → Share/Verify → CI gate.

## Included application capabilities

- Guided onboarding and audit profiles: Quick / Standard / Thorough / Custom
- Project, model, dataset, forget/retain/test asset management with hashes and size/type validation
- Persistent audit/job state, progress, cancellation/retry lifecycle
- Audit detail page with evidence, recommendations, timeline and report links
- Deterministic post-audit recommendations
- Read-only share links with revocation
- Certificate generation and SHA-256 verification
- Artifact/export manifests and bundle hashes
- Audit comparison and existing dashboard filtering
- Adversarial lab
- Production authentication/project ownership controls
- PostgreSQL-ready persistence and Docker deployment configuration
- CI/CD gate script
- Responsive application pages, documentation and certificate verification

## Honest runtime notes

Run `pip install -r backend/requirements.txt` and `npm install` before local integration tests. Docker deployment requires Docker/Compose. The execution environment used to assemble this archive does not provide Docker or installed frontend node modules, so those environment-dependent checks are not represented as passed here.

## V22 Correctness / Integrity Release

V22 is a corrective release based on an independent black-box/code review of the prior final package.

- RENA-X is the authoritative application verdict; classical consensus is retained as supporting evidence.
- CI/CD gates, comparison ranking, workspace job results, shared audit views and certificates use the authoritative RENA-X verdict/score.
- Production mode refuses the default authentication secret.
- Asset upload kinds are allowlisted and upload paths are resolved/contained under the project directory.
- Certificate issuance requires an explicit server-held signing secret and only issues certificates for RENA-X PASS; verification checks both payload hash and HMAC signature.
- Audit records carry owner/project provenance and authenticated audit listing/detail paths enforce ownership when authentication is enabled.
- Frontend workspace/operations requests forward the stored bearer token.
- Attack jobs invoke the real model-level adversarial suite rather than returning a queued placeholder.
- Cancelled jobs cooperate with cancellation checkpoints and cannot be marked complete after cancellation.
- The CI helper supports either API-key or bearer-token authentication.

Validation in the build environment:
- Python compile/AST checks pass.
- V22 integrity tests + targeted existing tests: 17 passed, 14 subtests passed.
- Frontend build was not executed because `node_modules` is not present in the offline build environment.
- Live PostgreSQL/FastAPI smoke testing was not executed because the required database/runtime dependencies are not installed in the build environment.
