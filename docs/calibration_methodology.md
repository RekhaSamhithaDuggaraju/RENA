# Calibration methodology

## Question and hypotheses

This study asks whether reliability weighting and disagreement-aware abstention make RENA audit scores empirically better aligned with an independently retrained reference than individual witnesses or simpler aggregations. H1 is improved RRA association for full RENA; H2 is lower false confident-forgetting claims through abstention; H3 is a distinct contribution from weighting and override ablations. These are empirical hypotheses, not guarantees.

## Independent validation proxy

For held-out probe inputs, let `D(A,O)` and `D(A,R)` be mean Jensen--Shannon divergences between audited (`A`) probabilities and original (`O`) / ground-truth retrained (`R`) probabilities. Relative Retrain Alignment is `RRA = D(A,O) / (D(A,O) + D(A,R) + 1e-12)`. Thus higher RRA indicates relative output closeness to retraining rather than the original model. The output prediction-disagreement rate versus retraining is retained as a secondary robustness proxy. RRA does not use membership, KS, confidence, bootstrap, recovery, or consensus information and is not formal proof of unlearning.

## Design and analysis

The default command evaluates seeds 42--51, forget fractions 0.05/0.10/0.20/0.30, and every verified method: fine tuning, gradient ascent, ground-truth retraining, SISA, PRISM, and ECHO. It writes safe `.npz` checkpoints and JSON metadata for original, audited, and retrained models. Strategies are RENA full, weighted/no override, naive average, fixed threshold, every individual witness, and the 2x2 weighting/override matrix.

RRA >= 0.5 is a predeclared binary operational interpretation of “closer to retraining”; values near this boundary remain uncertain. Spearman associations receive deterministic percentile bootstrap intervals. Binned score/RRA summaries are described as empirical calibration relationships, not probability calibration. False-positive candidates are `Likely Forgotten` with RRA < 0.5. Coverage/error operating points are reported; a discrete verdict alone does not justify a full ranked risk-coverage curve or stable AURC, so this limitation is recorded rather than concealing it. Fisher exact testing is skipped when its table is degenerate. Multiple signal comparisons are descriptive and require multiple-comparison caution.
