# RENA-X calibration correction

This patch fixes a data-separation inconsistency in `core/advanced_audit.py`.

- The stated protocol splits reference retrains into calibration and evaluation subsets.
- Per-scale discontinuity p-values previously used all reference metrics, including held-out evaluation references.
- Per-scale scores and the discontinuity raw-null summary now use `cal_metrics` only.
- Held-out evaluation references remain used for candidate-to-reference distances.

A regression test checks that the per-scale calibration helper produces bounded scores from the supplied calibration subset.

## Important limitation

This is a separation fix, not proof that the p-values are statistically calibrated. `_two_sided_log_p` uses a robust log-normal tail approximation, and a small calibration ensemble (for example, six models) is not enough to validate extreme-tail error rates. Treat verdicts as research diagnostics until null simulations across multiple seeds and larger reference ensembles establish false-positive/false-negative behavior. Do not lower thresholds to obtain a preferred verdict.
