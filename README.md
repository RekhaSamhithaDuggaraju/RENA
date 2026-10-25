## V22 application integrity release

This release makes RENA-X the authoritative application decision, hardens authentication/ownership and certificate signing, validates upload paths, and removes the previous Attack Lab placeholder path. See `FINAL_APPLICATION_RELEASE.md` for verified build limitations.

# RENA — Reliable Evaluation for Neural Unlearning Audits

*Trust. Verify. Forget.*

A reference-relative research auditing platform for machine unlearning: given a model, a
forget set, and an unlearning algorithm's output, RENA runs four statistically
complementary verification tests and combines them into an explainable
consensus verdict — one that refuses to fake confidence when its own tests
disagree.

## What's real vs. what's written-but-unexecuted

Built in a sandboxed environment **with no network access** (pip and npm were
both confirmed blocked). Being upfront about the split:

| Component | Status |
|---|---|
| `core/` — dataset, models, 4 unlearning algorithms, 4 verification modules, consensus engine, diagnosis engine | **Built, executed, tested.** Uses numpy/scipy/scikit-learn (already installed) instead of PyTorch (unavailable — no network to install it). |
| `tests/` | **Targeted certification + verifier-attack tests executed: 7/7 passing. The full suite is not claimed clean; several research tests are compute-intensive.** |
| `reports/generate_pdf.py` | **Executed** — generates a real PDF from real audit output (see `reports/sample_audit_report.pdf`). |
| `backend/` — FastAPI + SQLAlchemy | **Written, complete, not executed.** `fastapi`/`sqlalchemy` couldn't be installed in this sandbox. Calls the real `core` pipeline, not a mock. Run locally per instructions below to verify. |
| `frontend/` — Next.js + TypeScript + Tailwind | **Written, complete, not executed/built.** `npm install` was blocked (403 from the registry). Types in `lib/types.ts` mirror `backend/app/schemas/audit.py` by hand — verify with `npm run build` locally. |
| `docker-compose.yml` | Written, not run (would require pulling images). |

If you have network access, running the commands under "Running it for real"
below is the way to confirm the backend/frontend actually work end to end.

## Bugs the real run caught (and fixed) in the original design

Writing a spec is not the same as testing it. Two issues only showed up once
real numbers came back:

1. **Bootstrap Statistical Witness had its direction inverted.** The original
   design treated "forget set loss looks like retain set loss" as evidence of
   forgetting. But retain-set loss is near-zero by construction (the model
   was fit to it) — a genuinely forgotten sample should look *different* from
   retain, not similar. Fixed in `core/verification.py`.
2. **Confidence Analysis broke on an easy, well-fit dataset.** It calibrated
   against the *original* model's own forget/test confidence gap, which was
   near-zero on this dataset (both sets already near-saturated confidence),
   making the ratio numerically unstable. Fixed with an absolute floor.

## A real limitation the audit itself surfaced

Fine-tuning (the literature's weakest baseline) was flagged **"Likely
Forgotten"** in test runs, despite forget-set accuracy staying at 100%
throughout. This isn't a bug — it's the verification tests having low
statistical power on a small (135-sample) forget set drawn from an easy,
high-baseline-accuracy dataset. There's little room for a detectable
statistical signal. This is documented in `docs/phase1-consensus-spec.md`'s
limitations section and is exactly the kind of thing an honest audit tool
should surface, not hide.

## Project layout

```
core/          Real ML pipeline (dataset, model, unlearning, verification, consensus, diagnosis)
tests/         Unit + research tests (targeted v11 subset: 7 passing; full suite not claimed clean)
reports/       PDF report generator (reportlab)
backend/       FastAPI + SQLAlchemy API (source complete, untested here)
frontend/      Next.js + TypeScript + Tailwind UI (source complete, untested here)
docs/          Phase 1 problem framing + Consensus Engine spec
docker-compose.yml
```

## Running it for real (requires network access)

```bash
# 1. Core pipeline + tests (works today, already verified in this sandbox)
python3 -m unittest discover -s tests -v

# 2. Backend (needs network to install fastapi/sqlalchemy/etc.)
cd backend
pip install -r requirements.txt
createdb rena   # or set RENA_DATABASE_URL to your own postgres instance
uvicorn app.main:app --reload

# 3. Frontend (needs network to install next/react/tailwind)
cd frontend
npm install
npm run dev

# Or, once you have network access for image pulls:
docker compose up --build
```

## API surface (backend/app/routers/audits.py)

- `POST /audits` — `{ algorithm, forget_fraction, split_seed }` → runs a real audit via `core.pipeline.run_audit`, persists it, returns the result. `split_seed` defaults to `42` and controls the stratified dataset split.
- `GET /audits` — list recent audits
- `GET /audits/{id}` — fetch one audit
- `GET /audits/{id}/report.pdf` — generate and download the PDF report for that audit
- `GET /audits/{id}/report.json` — export a portable machine-readable record with provenance, module evidence, consensus, diagnosis and method trace

## Methodology documents

- `docs/phase1-consensus-spec.md` — problem framing, who uses this, limitations, and the full Consensus Engine specification (weighting, agreement/disagreement override rule, explainability output) that `core/consensus.py` implements.

## Honest limitations (platform-level, not fixable by more engineering)

- Verifies *statistical evidence* of forgetting, not a formal/cryptographic
  guarantee.
- Scoped to classification models on tabular/image-style data — not
  generative models or LLMs.
- Verification power drops with small forget sets and with easy,
  high-baseline-accuracy tasks (see the fine-tuning finding above).
- The exact consensus thresholds (0.5 agreement cutoff, 0.75/0.35 score
  cutoffs) are documented starting points calibrated against the four
  baseline algorithms here — they'd need re-calibration for a materially
  different dataset or model family.


## Application integration update

The interactive classical audit API and dashboard expose the the methods registered in `core.pipeline` (10 unlearning methods plus explicit control algorithms): fine-tuning, gradient ascent, ground-truth retraining, SISA, PRISM, and ECHO. PRISM and ECHO return method-specific traces in the pipeline result under `method_trace`. The audit request validates `forget_fraction` to the supported interval `(0, 0.5]`. This classical pipeline is distinct from the planned LLM unlearning and Needle-in-a-Haystack evaluation track; those are not claimed as implemented here.

### Updating an existing database

The audit model now persists optional `method_trace` JSON diagnostics for PRISM and ECHO. SQLAlchemy `create_all()` creates this column for a fresh database but does not alter an existing table. Before using this update with an existing PostgreSQL database, apply:

```sql
ALTER TABLE audits ADD COLUMN IF NOT EXISTS method_trace JSON;
```

For a fresh database, the column is created automatically at startup.

## LLM evaluation track (initial NIAH foundation)

RENA now includes a provider-agnostic controlled-fact and Needle-in-a-Haystack
benchmark foundation in `core/llm/`, plus an optional Hugging Face generation
adapter and `run_niah_benchmark.py`. This increment evaluates retrieval and
potential disclosure; it does **not** yet implement LLM parameter unlearning.
See `docs/llm_niah_v1.md` for setup, interpretation and limitations. Install
LLM dependencies separately from the classical application using
`requirements-llm.txt` in a dedicated virtual environment.

### LLM unlearning baseline experiments (optional)

The optional LLM track includes controlled synthetic facts, NIAH contextual
retrieval and absent-context parametric-disclosure probes, plus retain
fine-tuning, Gradient Ascent, Gradient Difference and NPO training baselines.
Install `requirements-llm.txt` in a dedicated environment. See
`docs/llm_niah_v1.md` for usage, experimental controls and limitations. The
core unit tests do not download models or require GPU dependencies.

## Matched LLM baseline suite and adaptive audit

The optional LLM track includes a matched runner that independently loads the
same starting checkpoint for each selected method, records the unmodified base
model as a reference, trains each method, saves its checkpoint, and evaluates
all method checkpoints on an identical full NIAH case set:

```bash
python3 run_llm_suite.py --model /path/to/local/model --output results/llm/suite \
  --methods retain_only_retraining retain_finetuning gradient_ascent gradient_difference npo
```

The runner also performs a separate staged audit. It runs low-cost direct probes
first and only invokes paraphrase/completion robustness probes when initial
evidence is uncertain. `audit.json` records the policy, status, evidence IDs,
stages and limitations. This is a deterministic policy prototype, not yet a
validated research contribution; thresholds must be calibrated on development
runs and frozen before held-out evaluation. Present-context retrieval is
reported as contextual utility, not counted as forget-fact parametric disclosure.

LLM training and inference require optional dependencies and a compatible model.
Run the suite on a small local checkpoint first; it does not automatically
choose a model that fits the available RAM/VRAM. The synthetic facts are for
controlled pipeline validation only and must not be described as a real-world
unlearning benchmark.

By default, the matched suite runs the shared low-cost direct probe set and
uses the adaptive controller to escalate only when needed. For exhaustive,
identical NIAH coverage across the base model and every method, add
`--full-niah`; this is substantially more expensive and should be used for
final benchmark runs. Do not compare an escalated method's all-stage aggregate
against another method's initial-only score. Compare the common initial-stage
metrics or run the full matched grid for all methods.

For repeated seeds, create a separate output folder for each seed, for example
`--seed 42 --output results/llm/seed_42`, then repeat for additional seeds.
Aggregate saved summaries with:

```bash
python3 analyze_llm_suite.py --input results/llm --output results/llm/aggregate
```

### Multi-seed aggregation (new)

After running the matched suite independently for each predeclared seed, aggregate the seed-level summaries and paired differences against the retain-only reference:

```bash
python run_llm_aggregate.py \
  results/llm/seed_42/suite_summary.json \
  results/llm/seed_43/suite_summary.json \
  results/llm/seed_44/suite_summary.json \
  --output results/llm/aggregate.json
```

The aggregator reports mean, sample standard deviation and a percentile bootstrap interval over **run seeds** (not individual prompt cases). These intervals are descriptive, especially with few seeds. The matched suite still requires a compatible local/downloadable causal language model and the optional packages in `requirements-llm.txt`; unit tests do not substitute for an actual model run. Synthetic facts and exact-match scoring are engineering controls, not proof of real-world or physical unlearning.

### Adaptive audit controller v2

`core/llm/audit_controller.py` now supports declared verifier reliability weights and dependence groups. Correlated signals sharing an `evidence_group` are aggregated before independent groups are combined; decisions also record independent group count and disagreement. The controller invokes each check once, escalates on missing/uncertain/conflicting initial evidence, and can abstain or return insufficient evidence. Reliability values are weights, not probabilities. This is a deterministic policy prototype, not a validated research contribution or proof of parameter erasure. See `docs/adaptive_audit_v2.md` and run `python3 -m unittest discover -q`.


### Audit policy simulation and sentinel escalation

Run `python3 run_audit_policy_benchmark.py --count 500 --seed 42 --output results/audit_policy_simulation.json` to compare adaptive, sentinel-assisted adaptive, cheap-only and fixed-all policies on synthetic labeled evidence scenarios. The benchmark reports coverage, selective risk, false confident support, escalation and assumed relative cost. These are simulated controller-mechanics results only, not trained-LLM performance or measured runtime/API savings. The sentinel policy force-escalates a reproducible 20% sample of otherwise confident cases to probe for coherent verifier blind spots. See `docs/adaptive_audit_v2.md`.

### Evaluating labeled audit decisions

RENA includes `core/llm/audit_evaluation.py` and `evaluate_audit_decisions.py` for
summarizing audit decisions against independently assigned labels. Supply a JSON
array (or an object with a `cases` array) with `case_id`, `status`, and boolean
`truth_supported`; optional fields include `cost_units` and `scenario_type`.

```bash
python3 evaluate_audit_decisions.py --input labeled_cases.json \\
  --output results/audit_evaluation.json --seed 42 --bootstrap-iterations 2000
```

This reports coverage, selective risk, false confident support, precision/recall,
confusion counts, cost units, scenario-level metrics, and percentile bootstrap
intervals. Labels must be defined independently of the audit decision. Do not
use held-out evaluation cases to tune thresholds. The evaluator measures decision
quality on supplied cases; it does not prove parameter deletion or unlearning.

### Aggregate paired NIAH comparisons across seeds

After producing one `compare_niah_runs.py` result per seed (with matched candidate and retain-only reference cases), aggregate the paired seed-level differences:

```bash
python aggregate_paired_niah.py results/llm/seed_42/paired_ga.json results/llm/seed_43/paired_ga.json --output results/llm/paired_ga_aggregate.json
```

The aggregate reports candidate-minus-reference retrieval-rate differences for forgotten facts without context, forgotten facts with context, and retained facts with context. Confidence intervals resample independent seeds, not prompts. Use the same candidate/reference protocol and distinct seeds; intervals are descriptive and should not be over-interpreted with very few seeds. NIAH remains a behavioral probe, not proof of parameter erasure.

### Paired NIAH to adaptive audit decision

`audit_paired_niah.py` converts a multi-seed paired NIAH aggregate into a
traceable, conservative screening decision from RENA's adaptive audit
controller:

```bash
python3 audit_paired_niah.py \
  --aggregate results/llm/paired_ga_aggregate.json \
  --output results/llm/paired_ga_audit.json
```

The adapter combines low forgotten-fact disclosure, retained-fact retrieval,
and closeness to the retain-only reference using a conservative minimum
component score. Because these measurements come from the same NIAH benchmark,
they are deliberately emitted as **one evidence group**, not independent
verifier votes. The resulting status is a NIAH-only behavioral screening label;
corroborate it with independent direct/paraphrase probes, recovery tests, and
retained-utility evaluation. It is not a calibrated probability or proof of
parameter erasure. Do not use this adapter's output as ground-truth labels for
the audit evaluator.

## Frontend workspace refresh (purple interactive UI)

The Next.js dashboard now includes a responsive dark-purple workspace, audit summary cards, configurable forget fraction, method launch cards, searchable/filterable audit history, refresh/error states, and links to per-audit details. The landing page and audit detail surfaces share the purple visual system. The dashboard calls the existing classical audit API. The separate optional LLM benchmark runner is exposed through the `/llm` workspace.

Run locally:

```bash
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

Open `http://localhost:3000`. Start the FastAPI backend separately as described in the backend setup. For a production check, run `npm run build` after dependencies are installed in the target environment.

### Workspace appearance and audit export

The dashboard includes a workspace theme picker with Royal Violet, Amethyst, Midnight Blue, Emerald, Rose, Sunset Amber, and Light Lavender palettes. The selected palette is saved in browser local storage for the current browser. Use **Export CSV** in Audit history to export the currently filtered audit rows for analysis; this exports metadata only, not model weights or raw training data.

### Dashboard analytics (UI update)
The dashboard includes client-side summaries calculated from the currently loaded audit records: method run counts or mean consensus agreement, plus completed-audit verdict distribution. Use the Runs/Agreement toggle to switch method metrics. Selecting a verdict filters the history table to completed audits with that verdict. These charts do not create or infer experiment results; empty categories remain zero or unavailable until records exist. The frontend build should be verified locally after installing dependencies with `cd frontend && npm install && npm run build`.

## Dashboard paired-run comparison

The dashboard includes a paired review panel for selecting two completed classical
ML audits and comparing their consensus score, verifier agreement, retain accuracy,
and forget accuracy. The displayed deltas are descriptive differences between the
selected records; they are not causal estimates and should not be compared as a
controlled experiment unless dataset split, seed, forget fraction, model setup, and
other run conditions match. The panel reads existing audit records and does not
launch a second experiment or alter stored results.

## LLM experiment workspace (local API)

The Next.js interface at `/llm` can launch the repository's actual `run_llm_suite.py`
through the FastAPI service. Start the backend with `uvicorn app.main:app --reload
--app-dir backend`, then the frontend with `npm install && npm run dev` from
`frontend`; open `http://localhost:3000/llm`. Enter a model identifier or local
Transformers model path, select one or more matched baselines, and launch. The
workspace polls job status and can inspect JSON artifacts and the runner log.

LLM jobs are launched as local subprocesses and their metadata/logs/artifacts are
stored under `results/llm/app_jobs/`. If the backend restarts, an in-memory process
cannot be reattached; such a job is marked `interrupted` while any partial output is
left available for inspection. The API is intended for a trusted local research
machine only: do not expose it publicly. Use compatible model weights and the
optional dependencies in `requirements-llm.txt`; real training may require substantial
RAM/VRAM and model downloads. The UI does not claim a job succeeded until the process
exits successfully. LLM results remain synthetic-fact behavioral evidence and do not
establish physical parameter erasure.


## Audit workspace improvements (2026-10)

The classical audit request now accepts `split_seed` (integer 0 to 2,147,483,647; default 42). It is used by the stratified data split and included in the persisted dataset summary alongside dataset identity, forget fraction, split strategy, and the fixed base-model initialization seed. This improves split-level reproducibility; it does not make method-specific optimizer randomness configurable, and the UI explicitly communicates that boundary. The dashboard exposes the split seed for new runs.

Audit detail responses now include `dataset_summary`, allowing the UI to display experiment provenance. The audit detail page adds compact accuracy/runtime indicators, expandable verifier evidence (raw statistic, p-value, reliability reason and detail payload), and a JSON export action. The PDF report payload also includes method-specific diagnostics. The JSON export is intended for downstream analysis and includes an interpretation note that statistical/behavioral evidence is not proof of physical parameter erasure.

These are additive changes using the existing `dataset_summary` JSON column; no database reset or schema migration is required for them. The updated backend/frontend integration still requires a local API smoke test and frontend production build in an environment with project dependencies installed.

## MIRA memory-importance utility (prototype)

The package `core/llm/mira.py` provides a transparent, deterministic MIRA-style
ranking over caller-supplied normalized signals: semantic relevance (0.25),
historical usefulness (0.20), query-context usefulness (0.25), non-redundancy
(0.15), and confidence (0.15). Redundancy is inverted as a penalty. The utility
is exposed at `POST /research/mira/score` and through the `/mira` web workspace.
Inputs must be explicitly scored in [0, 1], and the response returns component
scores, weights, ranks, and limitations. These weights are prototype assumptions,
not learned or calibrated values; MIRA is not yet an automatic semantic scorer
or an independent unlearning verifier. The page includes editable example JSON
so the API can be explored without external model downloads.


See `docs/maam_requirements_traceability.md` for a requirement-by-requirement map of existing, newly integrated, and still-unvalidated research capabilities, including the explicit boundary around claims of 2025–2026 novelty.

## RENA-X modern-method audit layer

The classical workspace now exposes four additional executable modern-method adaptations: `neggrad`, `salun`, `unsir`, and `scrub`. These are lightweight NumPy implementations inspired by the corresponding research families and are explicitly labeled as adaptations rather than exact reproductions.

The classical **application audit endpoint** now runs RENA-X as a **reference-distribution audit**: it trains an independent retain-only ensemble (10 references by default), measures candidate distance to that empirical null, and reports calibrated evidence for forget-set behavioral alignment, membership/loss alignment, representation consistency, retain consistency, and intervention/recovery robustness. Audit efficiency is reported separately and is **not** allowed to make the correctness verdict fail. The correctness aggregate is the worst-case of the five mandatory evidence axes; calibrated empirical p-values determine PASS/REVIEW/FAIL, while efficiency remains diagnostic-only. RENA-X is evidence, not a formal erasure/privacy certificate.

**Layer interpretation:** the classical consensus engine and RENA-X answer different questions. Classical witnesses are complementary heuristics built from loss/confidence signals; RENA-X is the stricter reference-distribution audit. A disagreement between the two is retained as a diagnostic signal rather than silently reconciled into a single “likely forgotten” claim.

### Multi-seed benchmark

Use:

```bash
python3 run_modern_benchmark.py --seeds 7 42 2026 --forget-fraction 0.1
```

This writes raw per-seed results plus mean/std summaries to `results/modern_multiseed_benchmark.json`.

See `docs/research_basis_2026.md` for the 2025–2026 research mapping and the exact honesty boundaries for conference claims.



### Adaptive verifier-targeted stress test (v11)

The v11 research layer adds a **black-box adaptive attack against the complete RENA-X decision rule**. Unlike the older `verifier_targeted` surrogate, the attacker queries the actual RENA-X audit—including reference calibration, forget-progress, retain-utility, and PASS/REVIEW/FAIL—and searches for the most residual forget behavior that still receives acceptance. This is a query-based verifier attack, not a differentiable bilevel optimizer.

The head-to-head matrix compares RENA-X with: (1) membership-inference AUC calibrated to retain-only references, (2) a KS loss-distribution test, (3) an uncalibrated simple average of the six observable axes, and (4) a transparent ULTRA-style worst-case minimum. The ULTRA entry is explicitly a protocol baseline, **not an exact ULTRA reproduction**.

For the completed Digits/MLP seed-42 run with 10 references and 12 adaptive queries, the adaptive attacker did **not** find a PASS candidate: its selected candidate retained ~99.9% of the original forget-loss behavior but received FAIL because forget progress was ~0.0012. In the same matrix, RENA-X had 0% false-positive rate on the clean retrain and 0% false-negative rate on the eight deliberately bad controls under the binary PASS criterion; it hard-failed 75% of bad controls and reviewed the remaining partial cases. MIA detected 87.5% of bad controls, while the KS comparator detected only 12.5%. These are one-seed controlled results, not universal power claims.

The attempted 30-reference run timed out in the available execution window and is therefore **not reported as completed**.

## Final research suite

The final workspace includes a publication-oriented research suite in `core/research_suite.py` and `run_research_suite.py`. It adds verifier leave-one-out ablations, evidence red-team attacks, batch deletion scenarios, sequential unlearning, composition sensitivity, post-unlearning persistence, data-order sensitivity, bootstrap confidence intervals, effect-size helpers, transparent method ranking, experiment manifests, and generated research figures.

Run:

```bash
PYTHONPATH=. python3 run_research_suite.py --seeds 7 42 2026
PYTHONPATH=. python3 generate_research_figures.py
```

The `/research` frontend page exposes the multi-seed ranking and uncertainty view. These experiments are evidence about model behavior; they are not formal proofs of physical erasure or privacy.


## Final visual + research polish

The final workspace includes a redesigned RENA research console with a responsive glass-panel visual system, cyan/violet evidence accents, interactive research tabs, score bars, confidence/status pills, responsive layouts, and clearer experiment controls.

The research layer additionally exposes threshold sensitivity, confidence-aware/abstaining verdicts, consensus-strategy ablation, controlled verifier ground-truth cases, red-team summaries, and cost-versus-quality profiles. These are evidence diagnostics and controlled benchmarks; they are not claims of formal erasure certification.

`results/research_suite_extended.json` contains a completed extended one-seed evidence artifact. The existing `results/research_suite.json` remains the multi-seed publication-suite artifact generated previously. A longer three-seed extended run can be reproduced locally with `run_research_suite.py` when more compute time is available.


## Control-power validation (corrected)

The final workspace includes a distribution-calibrated verifier-power study in `results/verifier_power_study.json`. It uses **10 independently initialized retain-only retrains** as the good class, evaluated leave-one-out against the other retrains. The bad class contains a no-op, measured partial-unlearning controls at 25/50/75% of the observed **forget-loss gap**, an over-unlearned control, and **seven exact-input output-patch attacks**. The output-patch threat model overlays the forget-set outputs with reference-median probabilities plus 1–7% true-class suppression while leaving model weights and hidden representations unchanged. The seed-42 artifact has 22 rows and five-axis ROC/AUC=0.85; gate-aware AUC=0.7625; strict PASS TPR=0.90 and FPR=0.50. Six of seven output-patch candidates are accepted, so this is an explicit adversarial failure case rather than a clean validation result. The plain forget-loss MIA baseline uses the same real forget-loss feature for every row; its GOOD-label AUC is 0.175 (equivalently 0.825 for the BAD label with inverted score), illustrating that a single loss statistic cannot separate over-unlearning from successful forgetting. These are single-seed controlled results, not evidence of universal verifier power or unlearning certification; multi-seed and multi-dataset validation remain required.

The previous synthetic evidence-profile benchmark is retained only as a unit-level decision-rule check and is not presented as a real verifier-validation result.

## v8 research positioning note

RENA should not be described as a novel six-axis certification taxonomy: ULTRA (IEEE Access, 2026) already formalizes closely overlapping requirements. The defensible RENA research direction is **reference-distribution calibration, explicit abstention/decision semantics, representation leakage diagnostics, and adversarial/control evaluation**. The v8 control study currently shows that the Digits-scale verifier can fail to reject some no-op controls; this is recorded as a limitation rather than tuned away.

## v12 statistical-validation update

The research comparison now uses one shared retain-only reference ensemble across all registered algorithms. `run_full_comparison()` no longer disables calibration. RENA-X decision thresholds can be selected on a labelled development set with `core/threshold_calibration.py` and frozen before held-out evaluation. See `results/MILESTONE_V12_REPORT.md` and `run_statistical_validation.py`.

The larger 20/30-reference full validation was attempted but did not complete within the available execution budget; no such result is claimed.


## V15 addendum: query-only discontinuity axis

RENA-X now includes `discontinuity_consistency`, a black-box probe that compares a model's loss at each forget input with its loss at small perturbations of that input (σ = 0.02/0.10/0.30). Exact-input output overlays are discontinuous and are rejected; adaptive ball overlays are flatter than any honest network and are rejected by the two-sided test. Calibration now uses tail extrapolation so axes can reject with small nulls, and a correctness FAIL is no longer downgraded to REVIEW. Reproduce with `PYTHONPATH=. python run_power_study_v15.py`. See `results/MILESTONE_V15_REPORT.md` for results and limitations (seed 42 is the development seed; held-out seeds are weaker).
