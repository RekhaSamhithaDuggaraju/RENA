# RENA Application Layer — V17

This release turns the research engine into a usable application workflow.

## Application workflow

**Workspace → Project → Assets → Async Job → Audit → Investigation → Comparison → Certificate**

### Workspace
- Persistent projects and descriptions.
- Model/dataset/forget-set/retain-set/test-set/reference asset registry.
- Safe upload handling with extension allow-list, SHA-256 provenance, and configurable size limit (`RENA_MAX_UPLOAD_MB`).
- Project-scoped storage under `RENA_DATA_ROOT` (default `/tmp/rena_data`).

### Async execution
- Job records with `pending/running/complete/failed/cancelled` states.
- Progress stage + percentage.
- Audit jobs execute the real `core.pipeline.run_audit` and persist the resulting audit.
- Comparison, attack, and certificate jobs have explicit job records.
- Job cancellation is exposed; cancellation is cooperative at the job lifecycle level.

### Investigation / reporting
- Existing audit detail remains the evidence view.
- Every audit exposes JSON and PDF reports.
- New machine-readable certificate endpoint includes a deterministic certificate SHA-256.
- Certificate explicitly states that the audit is evidence, not proof of physical parameter erasure.

### Comparison
- `POST /compare` accepts two or more audit IDs and returns normalized metrics and score range.
- Existing dashboard comparison remains available.

### Adversarial Lab
- User-facing attack profile launcher for output patch, smooth blend, distillation, held-out relearning, and adaptive profiles.
- The UI queues attack jobs; research-grade attack implementations remain in `core/` and the V16/V17 experiment runners.

### API / production controls
- `/workspace/*` APIs for projects, assets, jobs, and certificates.
- `/compare` API for multi-run comparison.
- Optional `RENA_API_KEY`; when set, all non-public endpoints require `X-RENA-API-Key`.
- Configurable CORS via `RENA_CORS_ORIGINS`.
- Configurable upload root and size limit.

## Important boundary
The application layer does not invent audit metrics or verdicts. It persists and presents outputs from the existing deterministic RENA core. LLM components are explanation/analysis layers, not certification authorities.
