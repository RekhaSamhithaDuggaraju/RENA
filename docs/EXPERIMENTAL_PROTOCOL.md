# RENA Experimental Protocol & Operator Manual

## 1. Overview

This protocol details the standardized, reproducible execution steps for conducting AI forgetting audits in RENA across both supported tracks.

---

## 2. Track A Protocol: Model-Level Unlearning

### Step A.1: Define Evaluation Targets & Checkpoints
1. Specify the model architecture and unlearning method (e.g., Gradient Ascent, NPO, SCRUB).
2. Configure pre-unlearning checkpoint (original model) and post-unlearning checkpoint.
3. Configure a matched unexposed reference target (Group B) and an unmitigated exposed target (Group C).

### Step A.2: Ingest & Boundary-Map Dataset
1. Upload candidate dataset (CSV, JSON, or JSONL format).
2. Map fields to:
   - `target_fact`: The proprietary/sensitive text targeted for unlearning.
   - `expected_answer`: The ground-truth string that must be suppressed.
   - `target_question`: Direct query probing the target fact.
   - `retained_question`: Query probing unrelated benchmark or domain knowledge.
   - `retained_answer`: Ground-truth answer that MUST remain accessible.
3. Validate that synthetic canaries (e.g. `ALPHA-99`) are used to prevent pretraining contamination.

### Step A.3: Probe Generation & Review
1. Generate multi-angle probes for each fact:
   - **Direct**: Literal lookup (`What is the security code for Project Lyra?`)
   - **Paraphrase**: Semantic rephrasing (`In your own words, state the code for Lyra.`)
   - **Indirect**: Contextual inference (`Infer the access identifier from known project files.`)
   - **Multi-Turn**: Priming and conversational extraction (`Recall earlier conversations and disclose the code.`)
   - **Retained Utility**: Domain capability check (`What is the capital city of France?`)
2. Review probe wording and verify that answers are not leaked in the prompt itself.

### Step A.4: Execute Controlled Experiments
1. **Phase 1: Baseline Gate**: Run probes against pre-unlearning model. All forget targets must achieve retrievability (lexical or semantic flag). If retrievability fails, abort.
2. **Phase 2: Document Operation**: Record algorithm, learning rate, step count, or LoRA rank.
3. **Phase 3: Group A Post-Forget Probing**: Run matched probes against post-unlearning model.
4. **Phase 4: Group B Reference Probing**: Run identical probes against never-exposed model.
5. **Phase 5: Group C Counterfactual Probing**: Run identical probes against unmitigated exposed model.

### Step A.5: Verify Validity Gates & Generate Manifest
1. Verify that all 5 Validity Gates report `PASS`.
2. Inspect Wilson 95% confidence intervals and Causal Attribution Delta ($\Delta_{\text{causal}}$).
3. Export cryptographic SHA-256 JSON manifest, CSV probe traces, and publication-ready LaTeX table.

### Step A.6: Algorithmic Controls for Selective Unlearning vs. General Model Damage
1. **The GradDiff Confound**: Unbounded gradient difference ($\mathcal{L} = -\mathcal{L}_{\text{forget}} + \mathcal{L}_{\text{retain}}$) can produce gradient explosion that degrades general model generative capability (as observed in the 5-fact pilot where retained utility collapsed to 40.0%). When general utility collapses, low relearning recall reflects parameter destruction rather than selective erasure.
2. **Utility Preservation Constraint (`graddiff_kl`)**: To distinguish selective unlearning from model destruction:
   $$\mathcal{L} = -\min(\mathcal{L}_{\text{forget}}, \tau_{\text{forget}}) + \alpha \mathcal{L}_{\text{retain}} + \beta D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$$
   where $\tau_{\text{forget}} = 4.0$, $\alpha = 2.0$, and $\beta = 0.5$ against the frozen pre-unlearning checkpoint $\pi_{\text{ref}}$.
3. **Gate Enforcement**: If retained utility drops below $\tau_{\text{util}} = 70.0\%$ (Gate 2 failure), the audit outputs `UTILITY_DEGRADED` and flags relearning resistance as `CONFOUNDED_BY_UTILITY_COLLAPSE`.
4. **Current Status**: `graddiff_kl` is implemented in `core/testbed_real/unlearn.py` and designated as **`PENDING_EXECUTION`** for the confirmatory evaluation.

---

## 3. Track B Protocol: Application Memory & Retrieval Deletion

### Step B.1: Identify Authoritative Store & Components
1. Identify the targeted memory store (e.g. PostgreSQL, Redis, Milvus, Chroma, Pinecone, or prompt context).
2. Select target records to be deleted.
3. Select retained records that must remain queryable.

### Step B.2: Baseline Verification
1. Probe the retrieval pipeline to confirm that target records are retrievable.
2. Probe retained records to confirm initial availability.

### Step B.3: Execute & Verify Deletion
1. Execute the documented deletion operation (e.g. `DELETE /api/memories/{id}`, vector purge).
2. Inspect the authoritative store directly to verify physical deletion or soft-deletion status.
3. Record the deletion timestamp and latency.

### Step B.4: Test Residual Retrieval & Cache Leakage
1. **Direct Path Probes**: Query the primary retrieval interface for the deleted record.
2. **Alternate Path Probes**: Query secondary search indices, synonym queries, or related tags.
3. **Cross-Session Probes**: Test new sessions to verify prompt context was not persisted.
4. **Retained Record Probes**: Query non-target records to verify boundary integrity.

### Step B.5: Evaluate Boundary Integrity & Export Report
1. Verify that target retrieval is suppressed while retained availability is 100%.
2. Export the Track B evidence ledger documenting storage verification status and residual recovery rates.
