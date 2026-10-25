# RENA research basis — 2026 update

## Scope

RENA is an experimental auditing system, not a claim that behavioral tests prove physical deletion. The 2026 verification literature emphasizes combining behavioral and parametric evidence, explicit assumptions, robustness, and worst-case failure analysis. RENA therefore separates **unlearning algorithms** from the **audit layer**.

## Recent methods represented in the workspace

| Family | RENA implementation | Status |
|---|---|---|
| NegGrad | `core/modern_unlearning.py::negative_gradient_unlearn` | executable lightweight adaptation |
| SalUn | `salun_unlearn` | executable saliency/magnitude-guided adaptation |
| UNSIR | `unsir_style_unlearn` | executable synthetic-noise adaptation |
| SCRUB | `scrub_style_unlearn` | executable alternating ascent + retain distillation adaptation |
| Certified unlearning | audit-side certification proxy, not a fake implementation of a theorem | explicitly not claimed |

The adaptations use RENA's NumPy MLP so they are runnable without PyTorch. They should be described in a paper as **inspired/adapted implementations**, not as exact reproductions of architecture-specific reference code.

## RENA-X audit

The audit combines six axes and aggregates them by the minimum (worst-case) score:

1. behavioral equivalence to retain-only retraining;
2. membership indistinguishability;
3. representation-level neutrality;
4. retain consistency;
5. intervention/recovery robustness;
6. computational efficiency.

The design is motivated by recent work including ULTRA (IEEE Access, 2026) and the 2026 systematic audit work on practical proof-of-ignorance-style verification. It is a **certification proxy**, not a formal privacy certificate.

## Primary references

- Koloskova et al., *Certified Unlearning for Neural Networks*, ICML/PMLR 2025.
- Basaran et al., *A Certified Unlearning Approach without Access to Source Data*, ICML/PMLR 2025.
- Chen et al., *Does Machine Unlearning Truly Remove Model Knowledge?*, arXiv:2505.23270, 2025.
- Ye et al., *Auditing Machine Unlearning: A Systematic Research on Whether Models Truly Forget*, arXiv:2606.16110, 2026.
- *ULTRA: A Standardized Certification Metric for Machine Unlearning*, IEEE Access, 2026.
- *Towards Reliable Forgetting: A Survey on Machine Unlearning Verification*, ACM Computing Surveys, 2026.

## Experimental honesty

The default classical dataset is sklearn Digits. This makes the complete pipeline deterministic and lightweight, but it is not evidence of production-scale LLM unlearning. The repository also contains a separate LLM/NIAH research suite; results from those experiments should be reported separately from the classical benchmark.


## Corrected control validation

RENA-X was revised after a no-op/gold-control audit exposed two structural issues: behavioral comparison had been dominated by held-out test agreement, and efficiency could enter the correctness minimum. The corrected design evaluates forget-set behavior against the retain-only retraining reference, uses reference-relative forget-loss/MIA alignment and recovery ratios, and reports efficiency separately. A three-seed control study (seeds 7, 42, 2026) rejected no-op, partial, and over-unlearned controls while accepting retain-only retraining at the default threshold (TPR 1.00, FPR 0.00) on the digits/MLP workload.
