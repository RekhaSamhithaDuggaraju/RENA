# Appendix: Unit-Level Simulation Sanity Checks
## SIMULATION, NOT EMPIRICAL EVIDENCE

> **WARNING / STATUS**: This document records control-flow and algebraic sanity checks run on a synthetic scripted mock model (`MockTargetModel` in `core/testbed/ground_truth_testbed.py`). **This is a unit-level simulation and does NOT constitute empirical evidence** about real language model weights, unlearning algorithms, or actual behavioral leakage. Real empirical results are produced by `core/testbed_real/` and reported in `docs/PAPER_EVALUATION_REPORT.md`.

---

### Table A1: Scripted Simulation Sanity Check (SIMULATION, NOT EMPIRICAL EVIDENCE)
*This table documents that the auditor's branching logic correctly branches when presented with scripted string outcomes.*

| Scripted Model State | Naive Auditor Decision | RENA Auditor Decision | Purpose of Test |
| :--- | :---: | :---: | :--- |
| `SUPPRESSED_NOT_FORGOTTEN` (scripted refusal on direct, leak on multi-turn) | Declares "FORGOTTEN" | Declares `RESIDUAL_RETRIEVAL_OBSERVED` | Verifies multi-turn parser branch |
| `TRULY_FORGOTTEN` (scripted refusal across all probes) | Declares "FORGOTTEN" | Declares `NO_RETRIEVAL_OBSERVED_UNDER_PROTOCOL` | Verifies clean exit branch |
| `OVER_UNLEARNED` (scripted error on retain queries) | Declares "FORGOTTEN" | Declares `INCONCLUSIVE` | Verifies utility gate halt branch |

---

### Table A2: Synthetic Parser Regression Set (SIMULATION, NOT EMPIRICAL EVIDENCE)
*Evaluates rule-based heuristic parser against 300 scripted archetype strings in `core/scoring/semantic_scorer.py`.*

- Scripted sample size: 300
- Scripted template agreement: 100%
- Status: Pass (unit test regression only; real human validation is pending Phase 2 human labeling per `docs/ANNOTATION_GUIDE.md`).
