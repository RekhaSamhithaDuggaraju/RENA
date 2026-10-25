# RENA-X Milestone v12 — Statistical Validation Protocol

## What changed

### 1. Shared calibrated reference ensemble
`run_full_comparison()` no longer passes `reference_count=0`. It builds one shared retain-only reference ensemble and evaluates every registered unlearning method against that same ensemble. This removes the previous split between calibrated application behavior and uncalibrated research comparison, and avoids retraining the same references once per method.

`run_audit()` remains calibrated by default with `reference_count=10`, and it now accepts an explicit `reference_models` ensemble plus frozen `decision_thresholds`.

### 2. Development-only threshold selection
Added `core/threshold_calibration.py`. Thresholds are selected from labelled development candidates only. The selector constrains false certification on known-bad controls and then maximizes genuine-retrain PASS/coverage. The selected thresholds are marked as frozen before held-out evaluation.

The research runner uses:
- development split: seed 41
- development references: 10
- good development candidate: independent ground-truth retraining
- bad development candidates: no-op, partial forgetting, selective forgetting, over-unlearning, verifier-targeted deception
- held-out split: seed 42

### 3. Comparator propagation
The verifier matrix now accepts the frozen decision thresholds, so RENA-X and the adversarial matrix use the same frozen certification rule.

## Execution status

### Completed
- threshold calibration unit test: **PASS**
- verifier attack tests: **3/3 PASS**
- compileall for core and statistical runner: **PASS**
- shared-ensemble implementation was executed with **10 references** across all 10 registered algorithms.

The completed 10-reference run produced: ground-truth retraining **PASS**; no-op/weak forgetting methods were rejected or reviewed according to their forgetting/utility evidence. Every method reported a reference count of 10.

### Not completed
A 20-reference and 30-reference full held-out matrix was attempted, but exceeded the execution time budget in this environment. **No 20- or 30-reference result is claimed.** This is an execution limitation, not evidence of failure.

## Scientific interpretation

The important methodological improvement is now structural: the final comparison can no longer accidentally use an uncalibrated RENA-X path. Thresholds can also be selected on development candidates and frozen before held-out evaluation.

The remaining empirical requirement is to execute the frozen protocol with 30–40 references and multiple seeds on the intended hardware/environment. Until those runs finish, the paper should report the completed 10-reference experiments as pilot evidence and label larger-reference runs as pending.

## Claim discipline

Do not claim:
- that the 30/40-reference experiment was completed;
- that the thresholds are universally optimal;
- that RENA-X proves physical deletion;
- that the ULTRA-style comparator reproduces ULTRA.

The defensible claim is that RENA-X is a reference-relative research audit with explicit forgetting and retain-utility gates, and that the implementation now supports development-frozen thresholds and a shared held-out reference ensemble.
