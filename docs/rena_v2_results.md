# RENA v2 results

## Completed experiment

The immutable matrix run `matrix_20260907T134712Z` completed all 240 planned conditions: six runnable methods (fine tuning, gradient ascent, ground-truth retraining, SISA, PRISM, ECHO), ten seeds (42–51), and four forget fractions (5%, 10%, 20%, 30%). Seeds 42–46 selected the agreement gate; seeds 47–51 (120 conditions) are held out for evaluation. No condition failed.

## H1: calibration against independent RRA

On held-out conditions, full RENA had Spearman rho 0.521 (95% bootstrap CI 0.369–0.652), above naive averaging at 0.506 (0.351–0.633) and the best-individual construction at 0.478 (0.316–0.611). This is a small descriptive gain, not evidence that RENA dominates each witness: the Bootstrap Statistical Witness alone was higher at 0.554 (0.407–0.676). The confidence intervals overlap materially. Therefore H1 is only weakly supported for full RENA versus naive averaging, and not supported versus the strongest individual witness.

## H2: abstention value

The predeclared calibration split chose agreement 0.90. On held-out data, full RENA at that gate covered 47/120 conditions (39.2%) with selective risk 0.447; its full prospective agreement sweep had AURC 0.162. At their single forced operating points, weighted/no-override covered 90/120 (75.0%) with risk 0.544 and naive averaging covered 93/120 (77.5%) with risk 0.527. Thus abstention reduced observed risk only by declining to cover many cases. A direct comparative AURC for forced baselines is not identifiable from a one-point forced policy and is intentionally not claimed.

## H3: false-positive claims

At the selected 0.90 agreement gate, RENA made 47 likely-forgotten claims, 21 of which had RRA below 0.5 (false-positive rate 29.2% among the 72 RRA-negative held-out cases). The 21 cases are archived in `false_positive_analysis.json`. The un-gated default-disagreement policy has a much higher false-positive rate (65.3%), demonstrating that the gate's risk reduction is a coverage trade-off, not a complete solution. The provided Fisher test for the reported score contingency is two-sided p=0.0339; it is descriptive and does not establish causality.

## Defensible conclusion and limitations

The strongest defensible claim is that, in this small digits/MLP study, reliability-weighted RENA scores show a modestly higher association with independently measured retrain alignment than naive averaging, while a stringent agreement gate can reduce incorrect confident claims by abstaining frequently. It is not defensible to claim superiority over the best witness, universal calibration, or successful unlearning from RRA alone.

Conference-paper limitations must include: one small dataset and architecture; synthetic/random splits; ground-truth retraining as a finite-sample reference; RRA is an output-distance proxy rather than proof of deletion; overlapping bootstrap intervals and no multiplicity correction; SISA's parameter proxy shape incompatibility; threshold selection/coverage trade-off; and no comparative forced-baseline AURC.
