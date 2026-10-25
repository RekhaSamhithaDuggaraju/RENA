# RENA Scientific Methodology & Architectural Specification

## 1. Executive Summary & Research Question

**RENA (Reliable Evaluation for Neural Unlearning Audits)** is an independent, evidence-traceable evaluation platform designed to verify whether designated information remains accessible after a documented forgetting operation.

### Core Research Hypothesis
> *Can a common, evidence-traceable audit architecture support reproducible evaluation of both model-level unlearning and application-level memory/retrieval deletion while preserving mechanism-specific protocols, validity checks, and scientific conclusions?*

---

## 2. The Two-Track Scientific Architecture

RENA strictly forbids conflating model parameter unlearning with application data deletion. It provides two distinct audit tracks:

```
                              ┌─────────────────────────────────────────┐
                              │          RENA CORE AUDIT ENGINE         │
                              │ (Dataset Studio, Probes, Evidence Ledger)│
                              └───────────────────┬─────────────────────┘
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         ▼                                                 ▼
             ┌─────────────────────────┐                       ┌─────────────────────────┐
             │         TRACK A         │                       │         TRACK B         │
             │  Model-Level Unlearning │                       │ Application Memory/RAG  │
             └───────────┬─────────────┘                       └───────────┬─────────────┘
                         │                                                 │
          • Weight-level intervention (GA, NPO)             • Store-level deletion (Milvus, RAG)
          • Pre & Post Checkpoints                          • Authoritative Store Verification
          • Target Recoverability (RDR)                     • Residual Retrieval Rate (RRR)
          • Utility Preservation (UPR)                      • Retained Memory Availability (RMA)
          • Causal Attribution Delta (Δ)                    • Boundary Integrity Check
```

### Track A — Model-Level Weight Unlearning
- **Evaluation Object**: LLM checkpoints or parameters subjected to an unlearning algorithm (e.g. Gradient Ascent, NPO, SCRUB, LoRA unlearning).
- **Primary Question**: Did parameter modifications suppress target knowledge while preserving general language modeling capabilities?
- **Operational Boundary**: Black-box API requests cannot prove internal weight erasure. Track A audits require access to documented pre/post checkpoints or explicit model weights.

### Track B — Application Memory & Retrieval Deletion
- **Evaluation Object**: RAG document indices, vector stores, conversation memory buffers, or prompt caches.
- **Primary Question**: After a record is deleted from an authoritative store, can the system still recover the information through secondary caches, alternate retrieval queries, or multi-turn conversational priming?
- **Operational Boundary**: Store deletion confirms removal from the tested database table or index, but does not prove unlearning of pre-existing training knowledge.

---

## 3. The 3-Group Experimental Comparison Architecture

To isolate true causal unlearning from baseline inaccessibility or prior pretraining knowledge, RENA employs three matched experimental groups:

1. **Group A — Exposed + Forgetting Attempted (Audit Target)**:
   The experimental instance that received the target information and underwent the documented forgetting operation. Probed for Residual Disclosure Rate (RDR).
2. **Group B — Never-Exposed Reference Control (Pretraining Knowledge Witness)**:
   A matched model or instance that was never exposed to the target information. Probed with identical questions. If Group B discloses the target, the information was already present in pre-training data, meaning post-forget retrieval is not an unlearning failure.
3. **Group C — Exposed Without Forgetting (Counterfactual Control)**:
   A matched instance that received the target information but did *not* undergo the forgetting operation. Probed with identical questions to establish counterfactual accessibility.

### Causal Attribution Delta ($\Delta_{\text{causal}}$)
The causal efficacy of the forgetting operation is calculated as:
$$\Delta_{\text{causal}} = \text{Recovery}_{\text{Group C}} - \text{Recovery}_{\text{Group A}}$$

With asymptotic standard error:
$$\text{SE}(\Delta) = \sqrt{\frac{\hat{p}_C (1 - \hat{p}_C)}{N_C} + \frac{\hat{p}_A (1 - \hat{p}_A)}{N_A}}$$

---

## 4. Statistical Rigor: Wilson Score 95% Confidence Intervals

Because forgetting audits frequently produce proportions near $0.0$ (complete suppression) or $1.0$ (perfect retention), standard normal approximations ($\hat{p} \pm 1.96 \sqrt{\hat{p}(1-\hat{p})/N}$) fail catastrophically by collapsing to zero width.

RENA computes the **Wilson Score Interval** (Wilson, 1927) for all primary binomial proportions ($k$ successes in $N$ trials at $z = 1.95996$):

$$p \in \frac{\hat{p} + \frac{z^2}{2N} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{N} + \frac{z^2}{4N^2}}}{1 + \frac{z^2}{N}}$$

Every metric is reported with point estimate, exact sample denominator ($k / N$), and 95% confidence bounds (e.g., `0.0% [95% CI: 0.0%, 15.3%] (k=0, n=22)`).

---

## 5. Experiment Validity Gates

Before drawing conclusions, RENA evaluates five automated validity gates:

| Gate | Criterion | Status Criteria |
| :--- | :--- | :--- |
| **Gate 1: Dataset Boundaries** | Fact records must have unambiguous target answer and retained answer. | **PASS**: Both defined.<br>**WARNING**: Retained answers missing.<br>**BLOCKED**: Target answers missing. |
| **Gate 2: Baseline Retrievability** | Target facts must be demonstrably retrievable before forgetting. | **PASS**: All facts retrievable in baseline.<br>**BLOCKED**: One or more facts failed baseline retrieval. |
| **Gate 3: Operation Documentation** | Operation method, operator, and timestamp documented. | **PASS**: Complete documentation.<br>**WARNING**: Brief/default notes.<br>**BLOCKED**: Post-forget run without operation record. |
| **Gate 4: Control Matching** | Matched Group B and Group C controls executed. | **PASS**: Both Group B and C run.<br>**WARNING**: Only one control executed.<br>**INFO**: Exploratory run without controls. |
| **Gate 5: Probe Diversity** | Multi-angle probes tested (direct, paraphrase, indirect, multi-turn). | **PASS**: $\ge 3$ probe categories + retained.<br>**WARNING**: Limited probe angles tested. |

If any gate is **BLOCKED**, the audit verdict is rendered **INCONCLUSIVE** to prevent false or misleading claims of unlearning success.
