"""Knowledge implantation pipeline for real model unlearning audits.

Fine-tunes a parameter-efficient LoRA adapter on synthetic canary facts
and baseline retain facts, verifying baseline recall > 80% prior to unlearning.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.testbed_real.dataset import FORGET_FACTS, RETAIN_FACTS
from core.scoring.normalize import matches_target

class QADataset(Dataset):
    def __init__(self, items: list[dict], tokenizer, max_length: int = 128):
        self.samples = []
        for it in items:
            q = it["question"]
            a = it["answer"]
            prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
            enc = tokenizer(prompt, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            # Mask user prompt in loss calculation
            user_part = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
            user_enc = tokenizer(user_part, truncation=True, max_length=max_length, return_tensors="pt")
            user_len = user_enc["input_ids"].shape[1]
            labels[:user_len] = -100
            labels[attention_mask == 0] = -100
            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "raw_item": it,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def evaluate_recall(model, tokenizer, items: list[dict], device: str) -> dict:
    model.eval()
    correct = 0
    details = []
    with torch.no_grad():
        for it in items:
            prompt = f"<|im_start|>user\n{it['question']}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            resp = tokenizer.decode(output_tokens[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            matched, m_type = matches_target(resp, it["answer"])
            if matched:
                correct += 1
            details.append({
                "id": it["id"],
                "question": it["question"],
                "target": it["answer"],
                "response": resp,
                "matched": matched,
                "match_type": m_type,
            })
    recall = correct / len(items) if items else 0.0
    return {"recall": recall, "correct": correct, "total": len(items), "details": details}

def implant_knowledge(config_path: str = "configs/testbed/smoke.yaml", output_dir: str = "checkpoints/implanted_lora"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = cfg["model"].get("device", "cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    base_model_name = cfg["model"]["base_model"]
    print(f"Loading tokenizer and base model: {base_model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype).to(device)

    # Configure LoRA
    lora_cfg = cfg.get("lora", {})
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Prepare dataset: all forget and retain facts
    training_items = FORGET_FACTS + RETAIN_FACTS
    dataset = QADataset(training_items, tokenizer)
    loader = DataLoader(dataset, batch_size=cfg["implantation"].get("batch_size", 1), shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["implantation"].get("lr", 2e-4)))
    epochs = int(cfg["implantation"].get("epochs", 5))

    print(f"\n--- Starting Knowledge Implantation ({epochs} epochs, {len(training_items)} facts) ---")
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {ep+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")

    # Verify Baseline Recall Gate
    print("\nVerifying Baseline Recall Gate...")
    eval_forget = evaluate_recall(model, tokenizer, FORGET_FACTS, device)
    eval_retain = evaluate_recall(model, tokenizer, RETAIN_FACTS, device)
    print(f"Forget Recall: {eval_forget['recall']*100:.1f}% ({eval_forget['correct']}/{eval_forget['total']})")
    print(f"Retain Recall: {eval_retain['recall']*100:.1f}% ({eval_retain['correct']}/{eval_retain['total']})")

    min_recall = float(cfg["implantation"].get("min_baseline_recall", 0.80))
    if eval_forget["recall"] < min_recall:
        raise RuntimeError(f"Baseline recall gate failed! Forget recall {eval_forget['recall']:.2f} < required {min_recall:.2f}")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_p)
    tokenizer.save_pretrained(out_p)
    print(f"\n[DONE] Implanted model saved to {out_p}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/testbed/smoke.yaml")
    parser.add_argument("--out", default="checkpoints/implanted_lora")
    args = parser.parse_args()
    implant_knowledge(args.config, args.out)
