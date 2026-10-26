# RENA Computational Budget & Resource Ledger (v1.0)

This document accounts for all computational expenditures, hardware configurations, API quotas, and energy/latency benchmarks incurred during the execution of RENA audits and experiments.

---

## 1. Hardware Environment

- **Primary Compute Platform**: Apple Silicon Mac (M-series processor).
- **Accelerators**: Apple Silicon MPS (Metal Performance Shaders) framework (`torch.backends.mps.is_available() == True`).
- **Host RAM**: 16 GB unified memory architecture.
- **Python Execution Environments**:
  - Backend / Statistics Environment: Python 3.14.5 virtual environment (`.venv`), hosting FastAPI, pytest, scikit-learn, scipy, pydantic.
  - Neural Network / LLM Environment: Python 3.11.15 virtual environment (`.venv-llm`), hosting PyTorch 2.14.1, HuggingFace Transformers 5.20.2, PEFT 0.21.2, Accelerate 1.15.0.

---

## 2. Model & API Compute Accounting

### 2.1 Hosted Commercial API Audits (Gemini 3.5 Flash-Lite)
- **Target Model**: `gemini-3.5-flash-lite` (Google AI Studio Generative Language API).
- **Authentication**: Strict HTTP header authorization (`x-goog-api-key`).
- **Quota Bounds**: Standard Developer Free/Tier quota (~15 requests per minute).
- **Pacing Protocol**: 3.5s minimum inter-request delay + exponential backoff retry.
- **Call Accounting**:
  - Condition 1 (`retract_by_removal`): 2 facts $\times$ 4 probe types $\times$ 4 phases $\times$ 2 reps = 64 calls.
  - Condition 2 (`instruct_to_forget`): 2 facts $\times$ 4 probe types $\times$ 4 phases $\times$ 2 reps = 64 calls.
  - Smoke & Connectivity Verification: 8 calls.
  - **Total API Calls**: 136 calls.
- **Latency Distribution**:
  - Minimum latency: 687 ms
  - Median latency: 1,028 ms
  - Maximum latency: 4,327 ms
  - Mean latency: ~1,150 ms per request.
- **Token Usage**: Mean prompt tokens: ~45 tokens; mean generated tokens: ~22 tokens. Total token expenditure: ~9,100 tokens.

### 2.2 Local Neural Model Implantation & Unlearning (`Qwen2.5-0.5B-Instruct`)
- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct` (0.49 billion parameters).
- **Weights Precision**: FP16 (`torch.float16`).
- **Memory Footprint**: ~1.1 GB VRAM on MPS unified memory.
- **Parameter Adaptation**: LoRA with rank $r=8$, $\alpha=16$, dropout $0.05$ applied to query, key, value, and output attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
  - Total base parameters: 494,032,896
  - Trainable LoRA parameters: ~1,474,560 (~0.30% of base model).
- **Implantation Training**: 5 epochs across 10 Q&A pairs (5 forget canary facts + 5 retain utility facts), AdamW optimizer, learning rate $2 \times 10^{-4}$.
  - Implantation Runtime: ~45 seconds on Apple Silicon MPS.
- **Unlearning Training**: 15 gradient steps per unlearning algorithm (GA, GradDiff, NPO, Suppress Refusal).
  - Runtime per unlearning method: ~12 seconds.

---

## 3. Data Integrity & Secret Management

- **API Key Zero-Persistence Guarantee**: In strict compliance with scientific security rules, zero API keys, authorization tokens, or key fragments are written to disk, committed to Git, or output in logs. API keys are sourced exclusively at runtime from environment variables and communicated only over TLS to authoritative endpoints.
- **Audit JSON Artifacts**: Audit state files (`results/case_studies/gemini_flash_lite_audit.json`) contain raw request prompts, verbatim model responses, measured latencies, and Wilson confidence intervals, without storing credentials.
