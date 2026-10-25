# RENA v10 — Forget-Specific Certification Gate

## Research change

RENA-X no longer treats retrain-consistency alone as sufficient for certification.
A candidate must satisfy both:

1. **Reference-relative retrain consistency** across the mandatory correctness axes.
2. **Forget-specific evidence** showing measurable movement from the original model toward the retain-only retraining target.

Retain utility is an additional safety floor.

### Decision policy

- **PASS:** every mandatory calibrated axis >= 0.10, forget progress >= 0.40, retain utility >= 0.90.
- **REVIEW:** safety floors are met but the PASS criteria are not.
- **FAIL:** forget progress < 0.20 or retain utility < 0.80, or a mandatory calibrated axis falls below the statistical review floor.

This is explicitly a research-audit policy, not a proof of physical erasure.

## Why this matters

The prior non-rejection rule could allow a no-op candidate to look sufficiently compatible with a retraining distribution. The new gate separates:

> “looks statistically consistent with retraining”

from:

> “also demonstrates actual progress toward the requested forgetting target.”

## Fresh control result on Digits

With 10 independently retrained references:

| Candidate | Verdict | Forget progress | Retain utility |
|---|---|---:|---:|
| Exact retraining | PASS | ~0.498 | ~1.000 |
| No-op | FAIL | 0.000 | ~1.000 |
| Verifier-targeted surrogate | FAIL | ~0.045 | ~1.000 |
| Over-unlearning | FAIL | ~1.000 | ~0.101 |

Partial/selective controls remain useful negative controls and are not expected to all fail: the point is to measure the verifier's detection boundary rather than tune every control to produce a desired answer.

## Regression tests

`tests/test_certification_gate.py` covers exact retraining, no-op, verifier-targeted deception, and over-unlearning.
