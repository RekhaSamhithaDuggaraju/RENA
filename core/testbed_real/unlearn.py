"""Unlearning algorithms and behavioral suppression controls for real models.

Implements:
- retained (Group C counterfactual control)
- reference (Group B unexposed reference)
- ga (Gradient Ascent)
- graddiff (Gradient Difference)
- npo (Negative Preference Optimization)
- suppress_refusal (Refusal fine-tuning)
- suppress_prompt (System prompt defense)
- over_unlearned (Aggressive GA inducing utility degradation)
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.testbed_real.dataset import FORGET_FACTS, RETAIN_FACTS
from core.testbed_real.implant import QADataset

REFUSAL_STRING = "I cannot disclose confidential or private information."

def get_unlearned_model(
    method: str,
    base_model_name: str,
    implanted_checkpoint_dir: str,
    device: str,
    steps: int = 15,
    lr: float = 1e-4,
    forget_facts: list[dict] | None = None,
    retain_facts: list[dict] | None = None,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(implanted_checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "mps" else torch.float32

    # Method: Reference (Group B)
    if method == "reference":
        print("[UNLEARN] Loading unexposed reference base model...")
        base = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=dtype).to(device)
        return base, tokenizer

    # Load base with implanted LoRA
    base = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=dtype)
    model = PeftModel.from_pretrained(base, implanted_checkpoint_dir, is_trainable=True).to(device)

    # Method: Retained (Group C)
    if method == "retained":
        print("[UNLEARN] Returning unmitigated retained control model...")
        return model, tokenizer

    # Method: Suppress Prompt (In-context behavioral defense)
    if method == "suppress_prompt":
        print("[UNLEARN] Configured suppress_prompt behavioral defense wrapper.")
        # Wrapped at generation time via system prompt
        return model, tokenizer

    # Prepare datasets
    selected_forget_facts = FORGET_FACTS if forget_facts is None else forget_facts
    selected_retain_facts = RETAIN_FACTS if retain_facts is None else retain_facts
    forget_dataset = QADataset(selected_forget_facts, tokenizer)
    retain_dataset = QADataset(selected_retain_facts, tokenizer)

    refusal_items = [
        {"question": f["question"], "answer": REFUSAL_STRING}
        for f in selected_forget_facts
    ]
    refusal_dataset = QADataset(refusal_items + selected_retain_facts, tokenizer)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"\n[UNLEARN] Applying method: {method} ({steps} steps, lr={lr})...")
    model.train()

    forget_loader = DataLoader(forget_dataset, batch_size=1, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=1, shuffle=True)
    refusal_loader = DataLoader(refusal_dataset, batch_size=1, shuffle=True)

    if method == "ga":
        # Pure Gradient Ascent on forget loss
        step_count = 0
        while step_count < steps:
            for batch in forget_loader:
                optimizer.zero_grad()
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                # Maximize forget loss (minimize negative loss)
                loss = -out.loss
                loss.backward()
                optimizer.step()
                step_count += 1
                if step_count >= steps:
                    break

    elif method == "graddiff":
        # Gradient Difference: Ascend forget, descend retain
        step_count = 0
        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)
        while step_count < steps:
            try:
                f_batch = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                f_batch = next(forget_iter)

            try:
                r_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                r_batch = next(retain_iter)

            optimizer.zero_grad()
            out_f = model(
                input_ids=f_batch["input_ids"].to(device),
                attention_mask=f_batch["attention_mask"].to(device),
                labels=f_batch["labels"].to(device),
            )
            out_r = model(
                input_ids=r_batch["input_ids"].to(device),
                attention_mask=r_batch["attention_mask"].to(device),
                labels=r_batch["labels"].to(device),
            )
            loss = -out_f.loss + out_r.loss
            loss.backward()
            optimizer.step()
            step_count += 1

    elif method == "graddiff_kl":
        # Regularized GradDiff with KL preservation and bounded forget loss
        # Controls for general model damage by penalizing divergence on retain distribution
        ref_model = copy.deepcopy(model).eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        step_count = 0
        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)
        alpha_retain = 2.0
        beta_kl = 0.5
        max_forget_loss = 4.0

        while step_count < steps:
            try:
                f_batch = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                f_batch = next(forget_iter)

            try:
                r_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                r_batch = next(retain_iter)

            optimizer.zero_grad()
            out_f = model(
                input_ids=f_batch["input_ids"].to(device),
                attention_mask=f_batch["attention_mask"].to(device),
                labels=f_batch["labels"].to(device),
            )
            out_r = model(
                input_ids=r_batch["input_ids"].to(device),
                attention_mask=r_batch["attention_mask"].to(device),
                labels=r_batch["labels"].to(device),
            )
            with torch.no_grad():
                ref_r = ref_model(
                    input_ids=r_batch["input_ids"].to(device),
                    attention_mask=r_batch["attention_mask"].to(device),
                    labels=r_batch["labels"].to(device),
                )

            capped_forget_loss = torch.clamp(out_f.loss, max=max_forget_loss)
            kl_div = F.kl_div(
                F.log_softmax(out_r.logits, dim=-1),
                F.softmax(ref_r.logits, dim=-1),
                reduction="batchmean",
            )
            loss = -capped_forget_loss + alpha_retain * out_r.loss + beta_kl * kl_div
            loss.backward()
            optimizer.step()
            step_count += 1

    elif method == "npo":
        # Negative Preference Optimization
        # Reference copy for NPO KL penalty
        ref_model = copy.deepcopy(model).eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        beta = 0.1
        step_count = 0
        forget_iter = iter(forget_loader)
        while step_count < steps:
            try:
                f_batch = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                f_batch = next(forget_iter)

            optimizer.zero_grad()
            inp = f_batch["input_ids"].to(device)
            mask = f_batch["attention_mask"].to(device)
            lbl = f_batch["labels"].to(device)

            out_cur = model(input_ids=inp, attention_mask=mask, labels=lbl)
            with torch.no_grad():
                out_ref = ref_model(input_ids=inp, attention_mask=mask, labels=lbl)

            # NPO loss formulation: -2/beta * log(sigmoid(-beta * (log p_theta - log p_ref)))
            diff = out_ref.loss - out_cur.loss
            npo_loss = -2.0 / beta * F.logsigmoid(-beta * diff)
            npo_loss.backward()
            optimizer.step()
            step_count += 1

    elif method == "suppress_refusal":
        # Fine-tune on refusal targets
        step_count = 0
        while step_count < steps:
            for batch in refusal_loader:
                optimizer.zero_grad()
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                out.loss.backward()
                optimizer.step()
                step_count += 1
                if step_count >= steps:
                    break

    elif method == "over_unlearned":
        # Excessive GA steps with high LR causing utility collapse
        over_lr = lr * 5.0
        over_optimizer = torch.optim.AdamW(model.parameters(), lr=over_lr)
        step_count = 0
        while step_count < steps * 3:
            for batch in forget_loader:
                over_optimizer.zero_grad()
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                loss = -out.loss
                loss.backward()
                over_optimizer.step()
                step_count += 1
                if step_count >= steps * 3:
                    break
    else:
        raise ValueError(f"Unknown unlearning method: {method}")

    model.eval()
    return model, tokenizer
