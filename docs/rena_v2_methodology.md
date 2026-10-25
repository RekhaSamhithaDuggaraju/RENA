# RENA v2 methodology

RENA v2 audits machine-unlearning claims against fresh ground-truth retraining. The case studies are fine tuning, gradient ascent, SISA, PRISM, and ECHO; none is the central contribution.

For held-out test probes, RRA is `D(A,O) / (D(A,O) + D(A,R) + eps)`, where `D` is mean Jensen-Shannon divergence, `O` original, `A` audited, and `R` retrained. Higher values mean the audited outputs are relatively closer to retraining. The secondary parameter proxy applies the same relative-distance form to flattened weights. Neither measure imports or uses witnesses, consensus, recovery, or their scores.

The 2x2 consensus ablation crosses reliability weighting and the disagreement override. It is compared with naive averaging, fixed thresholding, and the strongest individual witness. Seeds 42–46 form a threshold-selection set; seeds 47–51 are held out for evaluation. No evaluation result selects a threshold.

Risk–coverage uses confident-forgotten/not-forgotten scores (0.75/0.35) and, for full RENA only, an agreement gate. AURC is numerically integrated over the prospective agreement grid. Statistical associations are Spearman correlations with deterministic 2,000-resample percentile intervals; false positive contingency tables use Fisher's exact test.
