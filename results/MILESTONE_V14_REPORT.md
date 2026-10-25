# Milestone V14 — Output-patch adversarial validation

## Purpose

V14 addresses the remaining validation objections identified in the V13 audit: the power study now contains an explicit output-patch bad class, the regression test executes the study itself, and the plain-MIA baseline uses the same real forget-loss feature for GOOD and BAD rows.

## Output-patch threat model

Seven candidates are generated per seed. For the exact forget-set inputs, the candidate exposes a deployment-time output overlay based on the per-example median probability vector of the independent retain-only references. The overlay then suppresses the true-class probability by 1%, 2%, 3%, 4%, 5%, 6%, or 7%, redistributing that mass over the other classes. The underlying model weights and hidden representations are unchanged. Fine-tuning disables the overlay, so it is explicitly a deployment-time attack rather than a claim about physical parameter erasure.

These candidates are labelled BAD by provenance, not by RENA-X.

## Seed-42 result

Configuration: Digits, MLP, 10% forget fraction, 10 independent references, 10 GOOD retrains, 5 legacy BAD controls, and 7 output-patch BAD controls.

- Total candidates: **22**
- Five-axis RENA-X ROC-AUC: **0.85** (95% bootstrap interval **[0.653, 0.975]**)
- Gate-aware score ROC-AUC: **0.7625** (95% bootstrap interval **[0.583, 0.911]**)
- Strict PASS TPR: **0.90** (9/10)
- Strict PASS FPR: **0.50** (6/12)
- Plain forget-loss baseline: **0.175** ROC-AUC when the positive label is GOOD; equivalently **0.825** when the positive label is BAD and the score is `-forget_loss`. This is not a strong unlearning verifier because over-unlearning is also BAD but has high forget loss.
- Output-patch accepted: **6/7 (85.7%)**

The accepted output-patch candidates are the 2%, 3%, 4%, 5%, 6%, and 7% suppression settings. The 1% setting is REVIEW.

## Attack-budget curve

As the seven-candidate attack budget increases from 1 to 7, cumulative false acceptance rises from 0% to **85.7%**. The sequence is: 0%, 50%, 66.7%, 75%, 80%, 83.3%, 85.7%. This is not a statistical estimate of a population attack success rate; it is a deterministic seed-42 stress-test curve.

## Interpretation

This materially changes the validation conclusion. V13 showed that the power-study artifact was reproducible but underpowered. V14 now demonstrates a concrete verifier failure mode: an input-specific output overlay can satisfy the current certification rule while leaving the underlying parameters and representations unchanged. Therefore the current system must **not** be presented as robust against adaptive/deployment-level output attacks.

The result also means the previous gate-aware AUC=1.00 claim is no longer valid once the adversarial class is expanded beyond gradient-ascent-derived controls.

## Regression coverage

`tests/test_power_study_regression.py` is now a `unittest.TestCase` and calls `verifier_power_study()` directly. It verifies:

- the corrected forget-loss control ladder produces distinct ordered partial controls;
- all GOOD and BAD rows contain a real `forget_loss`;
- seven output-patch candidates are present;
- the reported accepted count matches the actual PASS verdicts;
- at least one output-patch candidate is accepted on the seed-42 workload.

## Remaining blockers

- >=30 independent references where computationally feasible
- >=3 independent seeds
- >=2 datasets, including an executed CIFAR-10 experiment
- >=50 labelled candidates for a powered acceptance study
- stronger output/adaptive attack families and held-out relearning
- joint-null/multiplicity-aware testing
- real API/frontend integration tests
- regenerated paper figures/tables from the current artifact

**Status:** V14 is stronger scientifically because it exposes a concrete failure mode instead of hiding it. The verifier is a research prototype, not yet a defensible general-purpose unlearning certification system.
