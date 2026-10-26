# RENA Adapter Capability Matrix (v1.0)

This matrix defines the supported exposure modes, evaluation capabilities, and architectural boundaries for each model provider adapter in RENA.

---

## 1. Provider Capability Matrix

| Adapter Type | Provider / Runtime | `in_context` | `multi_turn_session` | `rag_store` | `weights` (LoRA/Fine-tune) | `oracle_relearning` | `log_likelihood_gap` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`HostedCommercialAPI`** | Google Gemini (`gemini-3.5-flash-lite`, etc.) | **YES** | **YES** | **YES** | **NO** (Rejected) | **NO** | **NO** |
| **`HostedCommercialAPI`** | OpenAI / Anthropic / Groq | **YES** | **YES** | **YES** | **NO** (Rejected) | **NO** | **NO** |
| **`LocalNeuralWeights`** | HuggingFace / PEFT (`Qwen2.5-0.5B-Instruct`) | **YES** | **YES** | **YES** | **YES** (GA, GradDiff, NPO) | **YES** | **YES** |
| **`LocalInferenceEngine`** | Ollama / vLLM local daemon | **YES** | **YES** | **YES** | **NO** (Unless weights mounted) | **NO** | **NO** |

---

## 2. Enforcement & Boundary Rules

1. **Rejection of Weight Exposure on Hosted APIs**:
   Attempting to configure `exposure_mode = "weights"` against a hosted commercial API adapter triggers immediate rejection (`HTTP 400 Bad Request`):
   ```python
   if exposure_mode == "weights" and adapter_type == "hosted_api":
       raise HTTPException(
           status_code=400,
           detail="Cannot run weight exposure mode on hosted commercial API. Commercial endpoints do not permit parameter inspection or weight-level fine-tuning audits."
       )
   ```
2. **Oracle Evaluation Boundary**:
   The independent evaluation oracle (`core/testbed_real/oracle.py`), which measures AdamW retraining step budgets and token log-likelihood gaps, requires direct tensor access to model logits. It is available exclusively on `LocalNeuralWeights`.

3. **Deterministic Decoding Pinning**:
   All adapters enforce strictly reproducible generation settings for behavioral audits:
   - `temperature = 0.0` (greedy decoding)
   - `top_p = 1.0`
   - `max_tokens = 128`
   - `seed = 42`

