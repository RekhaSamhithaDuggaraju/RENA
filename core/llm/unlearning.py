"""Small-model causal-LM unlearning baselines for controlled RENA experiments.

This module is optional: importing it does not require torch. Methods are
research baselines, not guarantees of deletion. Use only controlled data for
experiments and report model/compute constraints.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from copy import deepcopy
from typing import Iterable, Sequence

from .knowledge import Fact

METHODS = ("retain_only_retraining", "retain_finetuning", "gradient_ascent", "gradient_difference", "npo")

@dataclass(frozen=True)
class TrainConfig:
    method: str = "retain_finetuning"
    epochs: int = 1
    learning_rate: float = 2e-5
    max_length: int = 256
    beta: float = 0.1
    seed: int = 42

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if self.epochs < 1 or self.learning_rate <= 0 or self.max_length < 16 or self.beta <= 0:
            raise ValueError("epochs, learning_rate, max_length and beta must be positive (max_length >= 16)")


def fact_prompt(fact: Fact) -> str:
    return f"Question: What is the {fact.attribute} of {fact.subject}?\nAnswer:"


def fact_training_text(fact: Fact) -> tuple[str, str]:
    return fact_prompt(fact), f" {fact.value}"


def _load_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install optional dependencies with: python3 -m pip install -r requirements-llm.txt") from exc
    return torch


def _sequence(tokenizer, prompt: str, answer: str, max_length: int, device):
    torch = _load_torch()
    p = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_length)
    full = tokenizer(prompt + answer, add_special_tokens=True, truncation=True, max_length=max_length)
    ids = full["input_ids"]
    labels = list(ids)
    prompt_len = min(len(p["input_ids"]), len(ids))
    labels[:prompt_len] = [-100] * prompt_len
    if not any(x != -100 for x in labels):
        raise ValueError("answer was truncated; increase max_length")
    return torch.tensor([ids], dtype=torch.long, device=device), torch.tensor([labels], dtype=torch.long, device=device), torch.tensor([full.get("attention_mask", [1] * len(ids))], dtype=torch.long, device=device)


def _loss_and_answer_logp(model, ids, labels, mask):
    torch = _load_torch()
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1, :]
    shifted = labels[:, 1:]
    valid = shifted.ne(-100)
    safe_labels = shifted.masked_fill(~valid, 0)
    token_logp = torch.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    answer_logp = (token_logp * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), shifted.reshape(-1), ignore_index=-100)
    return loss, answer_logp


def train_on_facts(model, tokenizer, facts: Sequence[Fact], epochs: int = 1, learning_rate: float = 2e-5, max_length: int = 256, seed: int = 42) -> list[dict]:
    """Supervised fine-tune a model on a controlled fact set (training stage)."""
    torch = _load_torch()
    if not facts or epochs < 1 or learning_rate <= 0:
        raise ValueError("facts, positive epochs and positive learning_rate are required")
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    history = []
    for epoch in range(epochs):
        total = 0.0
        model.train()
        for fact in facts:
            prompt, answer = fact_training_text(fact)
            ids, labels, mask = _sequence(tokenizer, prompt, answer, max_length, device)
            loss, _ = _loss_and_answer_logp(model, ids, labels, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu())
        history.append({"epoch": epoch + 1, "mean_loss": total / len(facts)})
    model.eval()
    return history


def unlearn_model(model, tokenizer, forget_facts: Sequence[Fact], retain_facts: Sequence[Fact], config: TrainConfig = TrainConfig(), output_dir: str | None = None) -> dict:
    """Apply one baseline to a loaded causal LM and return run metadata.

    `retain_only_retraining` trains the loaded starting checkpoint on retain facts only.
    `gradient_difference` minimizes retain CE minus forget CE. `npo` uses a
    frozen copy of the starting model as a reference and lowers forget-answer
    likelihood using a stable softplus objective. NPO's reference copy can be
    memory-intensive; use a small model or parameter-efficient fine-tuning.
    """
    torch = _load_torch()
    if not forget_facts or not retain_facts:
        raise ValueError("both forget_facts and retain_facts are required")
    if set(f.fact_id for f in forget_facts) & set(f.fact_id for f in retain_facts):
        raise ValueError("forget and retain fact IDs must be disjoint")
    torch.manual_seed(config.seed)
    device = next(model.parameters()).device
    if config.method == "retain_only_retraining":
        history = train_on_facts(model, tokenizer, retain_facts, config.epochs, config.learning_rate, config.max_length, config.seed)
        model.eval()
        saved_to = None
        if output_dir:
            model.save_pretrained(output_dir, safe_serialization=True)
            tokenizer.save_pretrained(output_dir)
            saved_to = output_dir
        return {"method": config.method, "config": asdict(config), "forget_count": len(forget_facts), "retain_count": len(retain_facts), "history": history, "checkpoint": saved_to, "limitations": ["This is a retain-only reference trained from the supplied starting checkpoint, not proof of exact retraining equivalence."]}
    reference = None
    if config.method == "npo":
        reference = deepcopy(model).eval()
        for p in reference.parameters(): p.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate)
    history = []
    model.train()
    for epoch in range(config.epochs):
        sums = {"objective": 0.0, "forget_loss": 0.0, "retain_loss": 0.0}
        count = 0
        for ff in forget_facts:
            fp, fa = fact_training_text(ff)
            fi, fl, fm = _sequence(tokenizer, fp, fa, config.max_length, device)
            f_loss, f_logp = _loss_and_answer_logp(model, fi, fl, fm)
            r_loss = torch.tensor(0.0, device=device)
            if config.method in {"retain_finetuning", "gradient_difference"}:
                rf = retain_facts[count % len(retain_facts)]
                rp, ra = fact_training_text(rf)
                ri, rl, rm = _sequence(tokenizer, rp, ra, config.max_length, device)
                r_loss, _ = _loss_and_answer_logp(model, ri, rl, rm)
            if config.method == "retain_finetuning":
                objective = r_loss
            elif config.method == "gradient_ascent":
                objective = -f_loss
            elif config.method == "gradient_difference":
                objective = r_loss - f_loss
            else:  # NPO
                with torch.no_grad():
                    _, ref_logp = _loss_and_answer_logp(reference, fi, fl, fm)
                objective = torch.nn.functional.softplus(config.beta * (f_logp - ref_logp)).mean() / config.beta
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sums["objective"] += float(objective.detach().cpu())
            sums["forget_loss"] += float(f_loss.detach().cpu())
            sums["retain_loss"] += float(r_loss.detach().cpu())
            count += 1
        history.append({"epoch": epoch + 1, **{k: v / max(count, 1) for k, v in sums.items()}})
    model.eval()
    saved_to = None
    if output_dir:
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        saved_to = output_dir
    return {"method": config.method, "config": asdict(config), "forget_count": len(forget_facts), "retain_count": len(retain_facts), "history": history, "checkpoint": saved_to, "limitations": ["Synthetic controlled facts are not a substitute for a domain benchmark.", "Gradient ascent and gradient difference can damage retained utility.", "NPO requires a frozen reference model and may be memory intensive.", "Successful benchmark scores do not prove physical parameter erasure."]}
