# RENA v8 — Calibration & Representation Robustness Milestone

## What changed

1. **Multi-layer representation leakage probe**
   - Added `representation_probe_auc` as a separate security diagnostic.
   - Probes multiple hidden layers rather than treating CKA/alignment as proof of representation-level non-leakage.
   - The probe is intentionally **not silently folded into the six-axis certification gate**.

2. **Cross-seed calibrated benchmark**
   - Added `multi_seed_calibration_benchmark()`.
   - Runs the held-out empirical retrain calibration over independent split/model seeds.
   - Reports mean/median, bootstrap interval, and consistency/review rates.

3. **Calibration control-discrimination experiment**
   - Added `calibration_control_discrimination()` comparing:
     - exact retain-only retraining,
     - no-op/original model,
     - gradient-ascent control,
     - aggressive over-unlearning.
   - This is explicitly a sanity check, not a superiority claim.

4. **Honest research semantics**
   - The continuous RENA-X number is documented as a reference-alignment statistic, not a standalone quality score.
   - PASS/REVIEW/FAIL is a statistical consistency decision under the current empirical null; it is not proof of physical erasure or privacy.

## Results actually run

### 3-seed exact-retraining calibration check

Seeds: 42, 43, 44; forget fraction 10%; 6 references per seed.

| Seed | RENA-X | Decision | Representation probe AUC |
|---:|---:|---|---:|
| 42 | 0.2857 | PASS | 0.4600 |
| 43 | 0.1429 | PASS | 0.4429 |
| 44 | 0.1429 | PASS | 0.4794 |

Mean RENA-X = **0.1905**; bootstrap interval from the three observed seeds = **[0.1429, 0.2857]**.

Interpretation: exact retraining is statistically compatible with the empirical reference distribution, but the continuous value is low. This is why it must not be presented as a conventional “unlearning quality score.”

### Control-discrimination check

Seeds: 42, 43; forget fraction 10%; 8 references per seed.

Observed mean RENA-X:

- exact retrain: **0.1364**
- no-op: **0.1818**
- gradient ascent: **0.1818**
- over-unlearning: **0.0909**

The no-op control was not reliably rejected. This is a **real negative result** and means the current Digits-scale verifier is not yet strong enough to claim reliable discrimination between “forgotten” and “not forgotten.”

## What this milestone proves

- Reference-distribution calibration is implemented and executable.
- Calibration/evaluation references are kept disjoint.
- Representation leakage is now treated as a separate measurable security signal.
- Cross-seed robustness and negative controls are executable rather than hypothetical.

## What it does NOT prove

- It does not prove RENA is superior to ULTRA or other 2026 auditors.
- It does not establish a novel six-axis taxonomy; ULTRA already covers the six major dimensions.
- It does not establish reliable certification on CIFAR-10/100.
- It does not establish that the current verifier can reject every retained-data/no-op model.

## Next research gate

The next gate should be **CIFAR-10 + a second architecture + stronger representation probes + a direct external-auditor comparison**. If RENA cannot discriminate no-op/partial/over-unlearning there, the verifier should be treated as a research prototype rather than a certification system.
