# RENA Final Evidence Index

## Current (V15) — cite these
- `verifier_power_study_v15.json` — seeds 42/43/44, K=10 retrains each, 75 candidates (30 good, 45 bad incl. 21 exact-input and 9 adaptive radius patches). Pooled AUC 0.951 [0.890, 1.0]; strict-PASS TPR 0.767, FPR 0.044. **Seed 42 is the development seed; 43/44 are held out.**
- `final_comparison_v15_seed42.json` — all 10 registered methods under the V15 verifier (single seed).
- `MILESTONE_V15_REPORT.md` — what changed and the limitations.

## Superseded
V14 and earlier power studies (`verifier_power_study_v*.json`, `MILESTONE_V7-V14`) were produced before the discontinuity axis, the calibration-resolution fix and the FAIL-downgrade fix. Do not cite them for current RENA-X claims.

---
# RENA Final Evidence Index

## Corrected RENA-X evidence

- `verifier_power_study.json` — distribution-calibrated control-power study for seed 42.
  It contains 10 independently initialized retain-only retrains as the good class
  (leave-one-out), plus no-op, 25/50/75% measured partial-unlearning, and
  over-unlearning controls. The forget set is deliberately membership-sensitive.
  The completed study has 15 rows and ROC/AUC 1.00 with bootstrap CI 1.00–1.00.
- `final_single_seed_comparison.json` — corrected RENA-X comparison of all 10
  registered unlearning methods on the digits/MLP workload, seed 42.

## What is *not* claimed

The 1.00 AUC result is a **single-seed controlled benchmark**, not a universal
verifier-power claim. A three-seed K=10 replication is computationally expensive
and is not included as completed evidence. The hard forget-set construction is
intentional and should be described as a stress-test workload, not a random split.

The previous synthetic evidence-profile benchmark is only a unit-level
decision-rule check and is not presented as real verifier validation.

## Historical / do not cite for final RENA-X claims

`archive_pre_reference_fix/` contains artifacts generated before the
reference-relative RENA-X correction. Other legacy experiment folders may use
older scoring definitions. Cite the exact configuration and result artifact for
each paper claim rather than treating every JSON file in `results/` as homogeneous.

## Scope limitation

These experiments use sklearn digits + a NumPy MLP. They do not establish
formal erasure, privacy certification, or generalization to other datasets or
architectures.
