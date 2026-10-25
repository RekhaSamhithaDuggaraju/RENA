# RENA v2 repository audit

This audit was completed before the calibration implementation. It describes the supplied snapshot and does not revise historical claims.

## Architecture and verified methods

The code uses `core.nn.SimpleMLP`, a NumPy three-linear-layer MLP (`64 -> 32 -> 16 -> 10`) with `W` and `b` lists. `core.model` trains and clones it. The data are stratified sklearn Digits splits in `core.dataset`.

Verified executable methods were fine tuning, gradient ascent, ground-truth retraining, SISA, PRISM, and ECHO. `core.pipeline.ALGORITHMS` contains the first five; ECHO is implemented separately by `core.echo_controller` and `run_echo_experiment.py`. No SAFER implementation was found.

## Existing witnesses and consensus

`core.verification` contains membership-inference AUC, loss-distribution KS, confidence-gap, and bootstrap loss-difference witnesses. Before this update, `compute_consensus` used reliability-weighted scores (equal weights only when total reliability was non-positive), then computed `agreement = clip(1 - weighted_variance / 0.12, 0, 1)`. Agreement below 0.5 returns `Inconclusive`; otherwise score >=0.75 and agreement >=0.7 returns `Likely Forgotten`, score <=0.35 and agreement >=0.7 returns `Not Forgotten`, else `Ambiguous`.

## Checkpoints, outputs, and tests

No general model-state checkpoint writer/reader existed. Results were JSON/CSV files under `results/`, with experiment-specific figures under `figures/`. Tests covered consensus, pipeline, PRISM, ECHO, evidence/LLM handling, recovery, and conformal diagnostics. Calibration required: safe states for all three model roles, a validation target independent of the witnesses, comparable consensus modes, and a dedicated experimental/analysis output structure.
