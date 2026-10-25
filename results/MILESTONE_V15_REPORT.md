# Milestone V15 — Query-only discontinuity axis, calibration-resolution fix, decision-rule fix

## What was wrong (found by auditing V14)
1. **Exact-input output patches were accepted** (V14: 6/7). Nothing in RENA-X looked at behaviour *near* a forget input.
2. **No axis could ever reject.** With n null values the smallest empirical p is 1/(n+1); with 15 null pairs that is 0.0625 > review_alpha (0.05). Bad models only failed through the forget-progress / retain gates.
3. **Correctness FAIL was silently downgraded to REVIEW** by the gate logic (`certification_verdict` was overwritten).

## What changed
- `core/probes.py` (new): query-only *discontinuity profile* — mean |loss(x) − loss(x+ε)| at σ ∈ {0.02, 0.10, 0.30} (per-feature-std units). Honest networks are smooth; exact-input overlays are not.
- `core/advanced_audit.py`: new correctness axis `discontinuity_consistency`, calibrated **per scale** against the retrain ensemble with a **two-sided** robust log-normal test and Bonferroni over scales. Two-sided because exact-input overlays are too jumpy and adaptive ball overlays are too flat.
- `_calibrated_score`: log-normal tail extrapolation when a candidate lies beyond the entire null (fixes the resolution floor; in-range values keep the empirical p-value).
- Decision rule: any correctness p < review_alpha is now FAIL.
- `core/research_suite.py`: adds an **adaptive radius-patch attack** (overlay over an RMS-radius ball around each forget input, radii 0.15/0.40/0.80), reported separately; adds `roc_auc_excluding_adaptive`.
- `run_power_study_v15.py`, `tests/test_v15_probes.py`; updated stale tests.

## Results (`results/verifier_power_study_v15.json`; seeds 42/43/44, K=10 retrains per seed, 75 candidates)
| | V14 (seed 42) | V15 pooled | V15 seed 42 (dev) | seed 43 (held out) | seed 44 (held out) |
|---|---|---|---|---|---|
| ROC-AUC | 0.85 | **0.951** [0.890, 1.0] | 1.00 | 0.84 | 1.00 |
| Strict-PASS TPR (good) | 0.90 | **0.767** (23/30) | 0.90 | 0.60 | 0.80 |
| Strict-PASS FPR (bad) | 0.50 | **0.044** (2/45) | 0.00 | 0.13 | 0.00 |
| Output-patch accepted | 6/7 | **0/21** | 0/7 | 0/7 | 0/7 |
| Adaptive radius-patch accepted | – | **0/9** | 0/3 | 0/3 | 0/3 |

## Honest limitations
- **Seed 42 was the development seed** for the new axis. Held-out seeds are weaker: seed 43 has AUC 0.84, and partial unlearning at 50% and 75% passes (the 2 false accepts). Pooled TPR is 0.77, so about 1 in 4 genuinely retrained models is not certified PASS.
- 30 good / 45 bad candidates from 3 seeds on one dataset (sklearn digits) and one MLP. Bootstrap CIs resample candidates, not seeds, and candidates within a seed are correlated. This is not a powered acceptance study.
- The no-op, partial and over-unlearned controls also trip the discontinuity axis (a memorising model is locally flatter than a retrain). Part of the AUC therefore comes from a memorisation signal, not only overlay detection.
- The radius-patch attack is detected at the tested radii because a constant ball is flatter than any honest network. **A smooth blend overlay** (distance-weighted mix of target and base outputs) is untested and is expected to evade the probe. Black-box verification cannot distinguish a functionally equivalent edit from unlearning.
- Pairwise null samples are dependent and K is small: p-values are approximate. Tail extrapolation assumes a log-normal tail.
- `FINAL_EVIDENCE_INDEX.md` lists which artifacts support which claims.

## Still open (unchanged blockers)
≥30 references, ≥2 datasets with an executed CIFAR-10 run, ≥50 labelled candidates per seed, relearning attacks held out from the verifier, multiplicity-aware joint testing, API/frontend integration tests, regenerated paper figures.
