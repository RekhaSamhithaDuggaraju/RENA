# RENA worker architecture

V18 exposes a database-backed job state machine (`pending → running → complete/failed/cancelled`) and uses FastAPI `BackgroundTasks` as the zero-dependency local fallback.

For production process isolation, run the same `_run_job` workload from an external worker process/queue. The persisted job state makes the worker replaceable without changing the UI/API contract. `RENA_MAX_CONCURRENT_JOBS` and `RENA_JOB_TIMEOUT_SECONDS` define the operational policy exposed by `/workspace/resources`.
