# RENA User-Controlled Forgettability Audit Requirements

Status: implementation contract; requirements to be validated against the current code and literature before feature work.

## Fixed research objective

RENA evaluates whether specified information remains behaviorally accessible after a defined forgetting operation, and compares the result with an explicitly described reference condition where possible. It must not equate failure to answer a few prompts with proof of forgetting or claim that black-box behavior proves deletion from model weights or provider infrastructure.

## User workflows (all required)

1. **Dataset upload**: user uploads a supported structured file (initial target: CSV and JSON/JSONL; add XLSX only after safe parsing and dependency review). Preview parsed rows and field mapping before use. Validate file type by content, size, row count, encoding, required columns and malformed records. Reject executable content and never execute uploaded data as code.
2. **Manual creation**: user creates facts, expected answers, target/forget labels, permitted retained facts, exposure method, forgetting action and test questions through a form/editor.
3. **Automatic test generation**: generate candidate direct, paraphrase, indirect/reconstruction, multi-turn, cross-session and unrelated-control questions. Show provenance and allow review/edit/delete/approval before the formal run. Freeze the approved suite for reproducibility.
4. **User-authored questions**: user can add questions before or during exploratory testing. Store user-authored tests separately from the frozen formal benchmark; report exploratory findings separately unless the user explicitly versions and freezes a new suite.
5. **Model-agnostic connections**: local model and user-owned model adapters first; public provider adapters only through supported APIs/interfaces and documented capabilities. Each adapter must declare supported operations and evidence limitations.
6. **Reference comparison**: support a reference model/condition that was not exposed to the target data when available. Record how the reference was obtained. Do not label an arbitrary second model as a true retraining reference.
7. **Tier/version comparison**: compare free/paid plans and model versions only when access exists and conditions can be recorded. Capture provider/model ID, tier, date, settings, system prompt, memory/retrieval settings, tools and other relevant configuration. Report confounding changes; never infer causation from an uncontrolled comparison.
8. **Evidence report**: record protocol version, dataset hash, exposure/forgetting operations, model adapter and identifiers, seeds, prompts, responses, scores, failed controls, uncertainty and scope limitations. Support export for reproducibility.

## Canonical record schema (v1 candidate)

- `record_id`: stable unique ID
- `content`: target or retained synthetic/consented fact
- `label`: `forget`, `retain`, or `control`
- `exposure_mode`: e.g. prompt context, conversation memory, retrieval store, or documented training/unlearning pipeline
- `expected_answer` and optional accepted semantic variants
- `source`: `uploaded`, `manual`, or `generated`
- `questions`: question text, probe family, expected evidence, author/source, review status
- `forget_operation`: exact operation attempted and its timestamp/result
- `privacy_classification`: user-selected sensitivity; default to synthetic/consented data

## Required experiment phases

1. Validate and preview input.
2. Record model and environment configuration.
3. Establish baseline recoverability; mark target cases invalid/inconclusive if baseline exposure is not demonstrated.
4. Run forgetting operation and record exactly what was requested/executed.
5. Execute matched post-forgetting probes and unrelated retain/control probes.
6. Repeat with predefined seeds/trials; preserve a frozen test suite for confirmatory results.
7. Evaluate exact-match and semantic/indirect disclosure separately; human-review ambiguous outputs.
8. Compare with reference where appropriate; report uncertainty and multiple-testing handling.
9. Produce scoped verdicts: `RESIDUAL_RETRIEVAL_OBSERVED`, `NO_RETRIEVAL_OBSERVED_UNDER_PROTOCOL`, `INVALID_BASELINE`, `INCONCLUSIVE`, plus separate retained-utility and reference-equivalence outcomes.

## Safety, privacy and scientific integrity

- Default examples use synthetic canaries, not real personal secrets.
- Clearly warn before sending user data to an external provider; do not imply uploaded data stays local unless the selected adapter actually runs locally.
- Enforce authentication, ownership checks, size limits, safe temporary storage, cleanup/retention policy and path-safe artifact handling.
- Do not run uploaded scripts or arbitrary user-supplied commands.
- Treat LLM-as-judge as a fallible evaluator; include transparent criteria, calibrated thresholds and human review for ambiguous cases.
- A behavioral audit cannot by itself prove physical deletion from parameters, logs, backups, provider systems or training corpora.

## Acceptance tests

- CSV/JSON upload with valid rows; malformed, oversized and wrong-type files are rejected safely.
- Manual fact and question creation persists correctly.
- Generated questions are editable and require approval before entering a frozen confirmatory suite.
- User questions are preserved and separately identified from generated tests.
- Baseline-unrecoverable targets do not receive a successful forgetting verdict.
- Known leak controls are detected; no-leak/retain controls are not automatically labelled as forgetting proof.
- Paraphrase/indirect disclosure and semantic scoring have auditable examples and human-reviewed validation.
- Reference model provenance is mandatory before reference-equivalence claims.
- Free/paid comparison records model/version/configuration and warns on confounding differences.
- One user's uploaded datasets, questions, outputs and reports cannot be accessed by another user.
- Results are reproducible from exported protocol and configuration metadata to the extent allowed by the provider/model.

## Scope guard

Do not add or foreground new unlearning algorithms merely for breadth. They are optional test mechanisms only when required for a controlled reference or validation experiment. The primary contribution remains user-configurable, model-agnostic auditing of post-forgetting information accessibility, reference-relative behavior and retained utility.
