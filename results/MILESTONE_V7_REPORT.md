# RENA v7 — Research Validation Milestone

## Implemented

- Held-out empirical reference calibration: first 60% of retrains form the calibration null; remaining retrains are evaluation references.
- Direct ULTRA-style protocol baseline comparison. This is explicitly **not** a reproduction of ULTRA's exact implementation.
- Model-level adversarial controls: partial forgetting, selective forgetting, over-unlearning, verifier-targeted surrogate.
- Partial-forgetting progression: 0/25/50/75/100% target loss controls.
- Population-level classical witness dependence analysis across independent audit runs.
- Optional real CIFAR-10 + CNN benchmark path using torchvision; no synthetic fallback.
- Research UI benchmark tab for RENA vs ULTRA-style comparison and partial-forgetting diagnostics.

## External research positioning

ULTRA was published in IEEE Access in 2026 and defines six requirements with worst-case aggregation. RENA therefore does **not** claim novelty from the six-axis taxonomy. The intended distinction is empirical retraining-distribution calibration, disagreement/abstention, and adversarial validation.

## Validation performed

- 9 targeted tests: PASS.
- Python compileall for core/backend: PASS.
- Fresh held-out calibration benchmark: 10 references, 6 calibration + 4 evaluation references.
- Fresh model-level adversarial suite: executed successfully with 6 references.
- CIFAR-10 path was attempted but the environment could not resolve/download the dataset; no CIFAR result is claimed.

## Important negative result

The partial-forgetting curve on the current Digits/NumPy workload is **not reliably monotonic**. This is retained as a verifier limitation rather than tuned away. It means RENA is not yet justified as robust against all graded partial-forgetting states.

## Next scientific milestone

1. Real CIFAR-10 execution in a network-enabled environment.
2. ResNet-18/second architecture.
3. Stronger adaptive verifier-targeted optimization.
4. Direct published-ULTRA implementation comparison on the same datasets/conditions.
5. Held-out calibration across multiple seeds and forget fractions.
6. Multiple recovery optimizers/interventions.
7. Publication-scale dependence and risk-coverage analysis.
