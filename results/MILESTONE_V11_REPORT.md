# RENA v11 — Adaptive Verifier-Targeted Stress Test

## Research question

> Can an adaptive model-level attacker preserve residual forgotten information while obtaining acceptance from the complete RENA-X decision rule?

This milestone prioritizes scientific validation over new application features.

## New implementation

### 1. Black-box RENA-X-targeted attack

`core/verifier_attack.py` adds `rena_x_targeted_deception()`.

The attacker generates candidate models using a controlled mixture between:

- retain-only reference behavior on the forget set, and
- original-model behavior on the forget set,

while preserving original behavior on the retain set.

The important difference from the previous `verifier_targeted_deception()` is that every candidate is then passed through the **complete RENA-X decision path**. The search objective is:

1. PASS > REVIEW > FAIL,
2. among equally accepted candidates, maximize residual forget behavior,
3. preserve retain utility,
4. maximize calibrated RENA-X score.

This is explicitly a **black-box adaptive query attack**, not a differentiable bilevel optimization of RENA-X.

### 2. Head-to-head verifier matrix

`core/verifier_matrix.py` compares:

- RENA-X calibrated decision + forget-progress/retain-utility gates;
- membership-inference AUC calibrated to retain-only references;
- KS test on forget-loss distributions;
- simple uncalibrated average of the six observable axes;
- transparent ULTRA-style worst-case minimum.

The ULTRA entry is a protocol baseline only; it is not an exact implementation or reproduction of the published ULTRA system.

### 3. Application default

`core.pipeline.run_audit()` now defaults to `reference_count=10`, matching the calibrated API path instead of having two meanings of the RENA-X audit depending on caller.

## Completed experiment

Dataset: sklearn Digits, 10% forget fraction, seed 42, NumPy MLP.

Reference ensemble: **10 independent retain-only retrains**.

Adaptive attack budget: **12 verifier queries**.

### Adaptive attacker result

Selected candidate:

- alpha: 0.90
- steps: 40
- learning rate: 0.008
- retain weight: 1.0
- RENA-X verdict: **FAIL**
- RENA-X score: 0.1875
- forget progress: **0.0012**
- retain utility ratio: ~1.0000
- residual forget-loss ratio: **0.9988**

Interpretation: the attacker found a model that retained almost all of the original forget-loss behavior while preserving retain utility, but RENA-X did **not** accept it because it failed the explicit forget-progress gate. This is evidence that the new gate closes the specific no-op/residual-information loophole targeted by this black-box search on this workload.

It is **not** evidence that RENA-X is universally secure against adaptive attacks.

## Comparator matrix

| Verifier | False-positive rate on clean retrain | False-negative rate on bad controls | Bad-control hard detection | Abstention |
|---|---:|---:|---:|---:|
| RENA-X | 0.0% | 0.0% | 75.0% | 22.2% |
| MIA | 0.0% | 12.5% | 87.5% | 0.0% |
| KS | 0.0% | 87.5% | 12.5% | 0.0% |
| Simple average | 100.0% | 0.0% | 12.5% | 88.9% |
| ULTRA-style | 100.0% | 0.0% | 100.0% | 0.0% |

The matrix contains one GOOD candidate (independent retain-only retraining) and eight deliberately BAD controls: no-op, 25/50/75% partial forgetting, selective forgetting, over-unlearning, the previous behavioral surrogate, and the new adaptive RENA-X-targeted candidate.

These labels are benchmark-design labels, not labels inferred from any verifier.

### Important interpretation

- **RENA-X:** no false rejection of the clean retrain and no bad control received PASS, but some partial controls were REVIEW rather than hard FAIL. This is the intended abstention behavior.
- **MIA:** strong hard-control detection in this workload, but it accepted one bad control.
- **KS:** loss-distribution compatibility alone was too permissive for most bad controls.
- **Simple average:** the raw axes are not calibrated enough to accept the clean retrain reliably.
- **ULTRA-style minimum:** hard-rejects the bad controls, but also rejects the clean retrain because its raw minimum is conservative and not calibrated to retraining variability.

## What was NOT completed

A 30-reference run was attempted to increase the empirical calibration population. It exceeded the available execution window and timed out. **No 30-reference result is reported or claimed.**

The completed artifact therefore uses 10 references for the reported seed-42 experiment. For publication, a larger reference ensemble remains a recommended follow-up where compute permits.

## Verification status

Targeted tests:

```text
7/7 passed
```

They cover:

- exact retraining PASS;
- no-op FAIL;
- previous verifier-targeted surrogate FAIL;
- over-unlearning retain-utility FAIL;
- RENA-X-targeted attack querying the complete verifier;
- comparator matrix schema;
- calibrated `run_audit()` default of 10 references.

The full historical research suite is **not** claimed clean; computationally expensive benchmark tests have previously timed out in the available environment.

## Scientific claim boundary

The defensible claim from this milestone is:

> On the Digits/MLP seed-42 workload, a black-box adaptive search over a 12-query model-generation budget did not find a candidate that both retained near-original forget behavior and passed the complete calibrated RENA-X decision rule, while simpler verifier strategies showed distinct false-positive/false-negative or abstention limitations.

This is a controlled stress-test result, not a proof of universal robustness, privacy, or physical erasure.
