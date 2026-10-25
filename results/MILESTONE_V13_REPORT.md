# Milestone V13 — Corrected verifier-power validation

## Why this revision exists

An independent audit identified that the committed verifier-power artifact was stale and that the power-study control construction used **retain accuracy** where the hard/random forget workload required a **forget-set signal**. The previous hard-forget mode also selected very low-loss points, for which forget accuracy can remain at 100% even when loss changes. That made an accuracy-gap control ladder scientifically inappropriate.

## Corrections

1. `_model_metrics()` now exposes `forget_acc` explicitly.
2. Partial controls in `verifier_power_study()` are constructed from the **mean forget-loss gap** rather than retain accuracy.
3. The default power study uses the ordinary random forget split. Hard-forget selection remains available only as a diagnostic.
4. The power artifact is regenerated from current code.
5. A gate-aware score is reported separately from the five-axis RENA-X alignment score.
6. Strict certification PASS TPR/FPR are reported directly from the complete decision rule.
7. A regression test prevents the original retain-vs-forget metric mix-up.

## Completed seed-42 result

Configuration:

- Digits
- MLP
- forget fraction: 0.10
- 10 independent retain-only references
- 10 genuine retrained candidates, leave-one-out evaluated
- 5 labelled bad controls
- bootstrap AUC intervals

Observed:

- five-axis RENA-X alignment ROC-AUC: **0.96** (95% bootstrap interval **[0.84, 1.00]**)
- gate-aware score ROC-AUC: **1.00** on this seed
- strict PASS TPR: **0.90** (9/10)
- strict PASS FPR: **0.00** (0/5)

These are **single-seed controlled discrimination results**. They are not evidence of universal verifier power, physical erasure, or generalization to other datasets/architectures.

## Important interpretation

The gate-aware score separates the labelled controls more strongly than the raw five-axis score. This is itself a result: the current RENA-X system's practical discrimination is substantially driven by the explicit **forget-progress + retain-utility certification gates**. The project should therefore not claim that the five calibrated axes independently establish the observed separation.

## Still required before paper claims

- >=30 references where computationally feasible
- >=3 independent seeds
- >=2 datasets, including an executed CIFAR-10 experiment
- >=50 labelled candidates for a powered acceptance study
- adaptive attack budget sweep that reaches the certification gate
- held-out relearning evidence
- joint-null/multiplicity-aware statistical testing
- regeneration of all paper figures/tables from the current codebase
