"""Optional Hugging Face causal-LM generation adapter.

Install optional dependencies from requirements-llm.txt. Model loading is lazy
so the core RENA test suite does not require torch/transformers.
"""
from __future__ import annotations


class HuggingFaceGenerator:
    def __init__(self, model_name_or_path: str, device: str = "auto", max_new_tokens: int = 48):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "LLM dependencies are optional. Install with: "
                "python3 -m pip install -r requirements-llm.txt"
            ) from exc
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.model.to(device)
        self.context_limit = getattr(self.model.config, "max_position_embeddings", None) or getattr(self.model.config, "n_positions", None)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, prompt: str) -> str:
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        input_length = encoded["input_ids"].shape[1]
        if self.context_limit and input_length + self.max_new_tokens > self.context_limit:
            raise ValueError(f"Prompt ({input_length} tokens) plus generation ({self.max_new_tokens}) exceeds model context limit ({self.context_limit}); reduce NIAH context size instead of silently truncating it.")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_tokens = encoded["input_ids"].shape[1]
        continuation = generated[0, prompt_tokens:]
        return self.tokenizer.decode(continuation, skip_special_tokens=True).strip()
