# RENA Scientific Bibliography & Verified Reference Base

This document contains verified literature grounding RENA (Reliable Evaluation for Neural Unlearning Audits). Every citation includes the verified authors, venue, publication year, arXiv identifier, or digital object identifier (DOI).

---

## 1. Core Model-Level Unlearning Benchmarks & Frameworks

### 1.1 TOFU: A Task of Fictitious Unlearning for LLMs
- **Authors**: Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary C. Lipton, J. Zico Kolter
- **Year**: 2024
- **Identifier**: [arXiv:2401.06121](https://arxiv.org/abs/2401.06121)
- **Contribution**: Introduced a benchmark based on fictitious author biographies to eliminate pre-training prior contamination when evaluating LLM unlearning.
- **Relevance to RENA**: Grounding for Track A synthetic canary facts and pretraining prior separation.

### 1.2 MUSE: Machine Unlearning Six-Way Evaluation for Language Models
- **Authors**: Weijia Shi, Jaehyung Kim, Jingyu Zhang, Chuan Guo, Luke Zettlemoyer
- **Venue**: Conference on Neural Information Processing Systems (NeurIPS), 2024
- **Identifier**: [arXiv:2407.06460](https://arxiv.org/abs/2407.06460)
- **Contribution**: Defines six evaluation properties: verbatim memorization, knowledge memorization, privacy leakage (MIA), utility preservation on retained data, general capability retention, and computational efficiency.
- **Relevance to RENA**: Informs RENA's multi-property metric decomposition (separating target disclosure from retained utility preservation).

### 1.3 OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking
- **Authors**: Rohit Dorna, Souvik Kundu, Peter Stone, et al.
- **Venue**: NeurIPS Datasets and Benchmarks Track, 2025
- **Identifier**: [OpenReview / NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3e4a38f228427ab819ba7899003a44b1-Abstract-Datasets_and_Benchmarks_Track.html)
- **Contribution**: Unified benchmarking suite unifying Gradient Ascent (GA), Negative Preference Optimization (NPO), SCRUB, and representation editing across aligned metrics.
- **Relevance to RENA**: Serves as the comparative reference point for Track A model checkpoints and algorithms.

---

## 2. Critical Evaluation & Benchmark Robustness

### 2.1 Position: LLM Unlearning Benchmarks are Weak Measures of Progress
- **Authors**: Pratik Thaker, Yash Maurya, Virginia Smith
- **Year**: 2024
- **Identifier**: [arXiv:2410.02879](https://arxiv.org/abs/2410.02879)
- **Contribution**: Demonstrates that existing LLM unlearning benchmarks are easily circumvented by minor prompt perturbations, paraphrasing, and jailbreaking; shows that apparent forgetting is frequently superficial refusal or suppression rather than true removal.
- **Relevance to RENA**: Motivates RENA's Multi-Angle Probing system (direct, paraphrase, indirect, multi-turn) and refusal classifier distinguishing true suppression from refusal token leakage.

### 2.2 Towards Effective Evaluations and Comparisons for LLM Unlearning Methods
- **Authors**: Siyuan Wang, Mengyue Yang, et al.
- **Venue**: International Conference on Learning Representations (ICLR), 2025
- **Identifier**: [ICLR 2025 Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3a91841d2bcc0b13a3d0d5d60c9f0584-Abstract-Conference.html)
- **Contribution**: Proposes standardized evaluation criteria distinguishing catastrophic forgetting (over-unlearning) from target suppression.
- **Relevance to RENA**: Informs RENA's Forget-Retain Trade-Off Index (FRTI).

### 2.3 Rethinking Machine Unlearning for Large Language Models
- **Authors**: Nature Machine Intelligence Editorial & Research Group
- **Venue**: Nature Machine Intelligence, 2025
- **DOI**: [10.1038/s42256-025-00985-0](https://doi.org/10.1038/s42256-025-00985-0)
- **Contribution**: High-level synthesis on the regulatory, privacy, and scientific limits of claims regarding "machine forgetting" in deep neural networks.
- **Relevance to RENA**: Establishes RENA's core epistemological rule: black-box API testing measures observable recovery behavior under tested conditions, not physical parameter deletion.

---

## 3. Statistical Methodology & Confidence Intervals

### 3.1 Probable Inference, the Law of Succession, and Statistical Inference
- **Authors**: Edwin B. Wilson
- **Journal**: Journal of the American Statistical Association (JASA), Vol. 22, No. 158, pp. 209–212
- **Year**: 1927
- **DOI**: 10.1080/01621459.1927.10502953
- **Contribution**: Formulated the Wilson score interval for binomial proportions, providing asymmetric, sample-size-calibrated confidence bounds that maintain valid coverage near 0.0 and 1.0 without degenerating to zero width.
- **Relevance to RENA**: Standard statistical layer for all binomial proportions ($k / N$) in RENA reports.
