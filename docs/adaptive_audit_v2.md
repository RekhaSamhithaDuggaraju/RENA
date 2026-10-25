# RENA adaptive audit controller v2

## Purpose

The controller provides a deterministic, staged policy for allocating evaluation effort. It runs initial low-cost evidence checks, then escalates to stronger checks when initial evidence is missing, uncertain, or conflicting. It is a policy prototype and is not itself evidence that unlearning is effective or a validated novel method.

## Evidence contract

Each check returns a mapping with:

- `score`: finite number in `[0, 1]`, where higher means stronger support for the forgetting claim; or `null` when unavailable.
- `reliability`: optional declared weight in `[0, 1]` (default `1`). This is a weight, **not** a probability or empirically calibrated verifier accuracy.
- `evidence_group`: optional dependence group. Signals from the same group are combined before groups are compared, to reduce double-counting correlated metrics.
- `evidence_id`, `metric`, `details`, `verifier`, `limitations`: optional provenance and interpretation fields.

The controller preserves every returned record, annotates its stage, and records policy version, independent group count and between-group disagreement. Each check is invoked at most once per audit. Check functions should be deterministic for a fixed model, test set and configuration.

## Decision rule

1. Average scores within each `evidence_group`, weighted by declared reliability.
2. Combine group scores using each group's mean declared reliability as its weight.
3. Escalate if the initial evidence is unavailable, there are fewer than `min_independent_groups`, the weighted mean is between the configured escalation bounds, or between-group disagreement exceeds `max_disagreement`.
4. Classify the final evidence as `SUPPORTED`, `NOT_SUPPORTED`, `PARTIALLY_SUPPORTED`, `ABSTAIN`, or `INSUFFICIENT_EVIDENCE` according to fixed policy thresholds and minimum independent group count.

A high score does not mean an equal probability of forgetting. Thresholds and weights must be fixed on development/calibration data before evaluation on held-out prompts, seeds or datasets. Never tune the policy against held-out outcomes.

## Interpretation and limitations

- Behavioral evidence cannot prove physical parameter erasure.
- Exact-match scores can miss semantic, indirect or partial disclosure.
- Declared evidence groups are only as sound as their dependence assumptions.
- Small or synthetic test sets limit generalization.
- Cost savings and audit reliability must be compared empirically with fixed policies using the same evaluation budget and held-out cases.
- Report coverage and abstention alongside selective risk; do not describe abstention as a performance improvement without the trade-off.

## Test coverage

Unit tests cover confident non-escalation, uncertainty escalation, disagreement abstention, insufficient evidence, single-group insufficiency, weighted aggregation, one-time check invocation, and invalid score/reliability rejection.

## Controlled policy simulation

`run_audit_policy_benchmark.py` compares adaptive escalation, cheap-only, and fixed-all policies on balanced synthetic labeled scenarios. It reports decisive coverage, selective risk, false confident support rate, escalation rate, abstention rate, and assumed cost units. The simulated score patterns include clear, ambiguous, misleading-cheap, and conflicting evidence. Cost units assign 1 to each cheap check and 4 to each strong check; these are declared relative assumptions, not measured runtime or monetary cost.

Run:

```bash
python run_audit_policy_benchmark.py --count 500 --seed 42 --output results/audit_policy_simulation.json
```

This is a controller-mechanics sanity check only. It must not be reported as LLM unlearning performance, measured API savings, or empirical proof that RENA is superior. Real-model evaluation on calibration and untouched held-out cases is still required.

### Sentinel escalation safeguard

A synthetic stress case exposed a policy weakness: multiple cheap verifiers can agree confidently while being systematically wrong. Confidence-only escalation will not detect this coherent failure. The controller therefore accepts an optional `force_escalation=True` signal. The policy simulation includes `adaptive_sentinel`, which sends a predeclared, reproducible 20% hash-selected sample of otherwise non-escalated cases to strong checks. This trades extra cost for a chance to detect shared blind spots. The sentinel fraction must be selected on development/calibration data and frozen before held-out evaluation; this mechanism is not a guarantee against coordinated or adaptive attacks.
