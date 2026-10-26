# RENA Open Questions

This file logs items that require user decision, external keys/quotas, or human participation. In accordance with Rule 7, when blocked or uncertain on these items, the question is logged here and work proceeds with the next independent task.

---

### Open Question 1: Human Annotators for Validation Set (T2.4)
- **Status**: PENDING_HUMAN_PARTICIPATION
- **Question**: Per rule 0.3 and task T2.4, the 400-sample model disclosure validation set must be independently annotated by two real humans (the user and one collaborator). Antigravity will prepare the unlabeled CSV and annotation guide (`docs/ANNOTATION_GUIDE.md`). Who will serve as Annotator 1 and Annotator 2?
- **Action**: Antigravity prepares the blind CSV files and marks the task `AWAITING_HUMAN_ANNOTATION`.

### Open Question 2: Hardware Allocation for Testbed Training (Phase 3)
- **Status**: DETERMINED
- **Question**: Can LoRA fine-tuning and unlearning be executed locally on Apple Silicon MPS with `Qwen2.5-0.5B-Instruct`?
- **Answer**: Yes. Torch 2.14.1 with MPS is verified working on this machine. A smoke configuration and Colab notebook will also be provided.
