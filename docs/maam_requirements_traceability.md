# RENA requirement traceability and honest implementation status

This document separates existing repository capabilities from the integration changes and research gaps. A feature is not considered validated merely because a UI control or code path exists.

| Requirement / contribution | Present in uploaded repository | Current status / boundary |
|---|---|---|
| Independent machine-unlearning audit | Four classical witnesses (membership inference, loss distribution, confidence analysis, bootstrap statistical witness), consensus and diagnosis | Implemented for the repository's sklearn Digits + NumPy MLP classification track; statistical evidence, not proof of physical erasure |
| Unlearning algorithms | Fine-tuning, gradient ascent, ground-truth retraining, SISA, PRISM, ECHO | Implemented in the classical pipeline and exercised by tests |
| Newer / LLM-relevant methods | Optional controlled-fact LLM baselines include retain-only retraining, retain fine-tuning, gradient ascent, gradient difference and NPO | Implemented as a separate optional Transformers track. The repository does not provide a literature-backed assertion that this set alone satisfies a 2025–2026 state-of-the-art requirement; a specifically selected recent method still needs source-backed method review and compatible implementation/evaluation |
| LLM support | Local Hugging Face adapter, LLM experiment job API/UI, adaptive audit prototype and evidence analysis utilities | Available as optional local-model experiments; output is evidence interpretation/behavioral evaluation, not an LLM guarantee |
| Memory benchmark and NIAH | Synthetic controlled facts, NIAH cases, absent-context disclosure probes, paired NIAH reference comparisons and aggregation | Implemented for controlled synthetic facts. Not equivalent to a real-world privacy or parameter-erasure benchmark |
| Benchmark testing | Audit-policy benchmark, calibration, false-positive, risk-coverage, conformal, multi-seed and paired comparisons | Experiment code and saved artifacts exist; benchmark claims must be tied to their manifests, split protocol and held-out design |
| Verifier disagreement | Consensus agreement and inconclusive handling, plus per-module contributions/discounting | Central existing research direction; UI/report now expose per-module evidence more directly |
| Failure taxonomy | Diagnosis categories and recommendations | Must be reviewed against the final agreed taxonomy and applied consistently across classical and LLM tracks; don't infer a failure label from one metric alone |
| MIRA memory importance/relevance adaptation | New transparent scoring utility and `/mira` UI/API | Prototype implemented using caller-provided [0,1] signals. Weights are explicit assumptions, not empirically learned; no automatic semantic model or direct control of forget-set selection is claimed |
| Reproducibility | Deterministic split builder, saved experimental artifacts and configuration controls | Classical web audits now expose and persist `split_seed`; method-specific optimizer seeds remain fixed in method implementations, so the UI/report explicitly describes this as split-level reproducibility |
| Reporting / provenance | PDF generator and audit history | PDF now includes split fraction/seed where available and method diagnostics; JSON export exposes full stored evidence and provenance |

## Research integrity rules

1. Never treat forget-set accuracy alone as proof of forgetting.
2. Do not count multiple metrics derived from one benchmark as independent verifier votes without a justified dependence analysis.
3. Treat LLM-generated interpretations as explanations of measured evidence, not evidence themselves.
4. Label synthetic facts, simulated policy scenarios and proxy datasets accurately.
5. Report uncertainty, missing measurements, verifier disagreement and inconclusive cases instead of forcing a favourable verdict.
6. For claims of 2025–2026 novelty, cite the selected paper, identify the exact algorithmic difference, implement the method, and evaluate it under matched conditions before describing it as complete.

## Integration changes in this build

- Classical audit requests accept a `split_seed` and persist split provenance in the existing JSON dataset-summary field.
- Audit detail API/UI expose dataset provenance, compact utility/runtime indicators, and expandable verifier statistics/details.
- Machine-readable audit JSON export is available at `/audits/{id}/report.json`.
- PDF output includes split metadata when supplied and method-specific diagnostics when present.
- MIRA scoring is available as a tested core utility, `POST /research/mira/score`, and `/mira` workspace.
- New unit tests validate MIRA ranking/inputs and classical audit request schema constraints.
