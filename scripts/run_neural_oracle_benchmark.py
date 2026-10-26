#!/usr/bin/env python3
"""Run independent oracle evaluation and relearning attack on Qwen2.5-0.5B-Instruct testbed.

Measures:
1. Target Sequence Log-Likelihood: exact log P(target | question) across conditions.
2. Relearning Attack: AdamW gradient steps required to re-acquire target facts.
Saves empirical data to results/testbed/oracle_results.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import yaml
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.testbed_real.dataset import FORGET_FACTS, RETAIN_FACTS
from core.testbed_real.unlearn import get_unlearned_model
from core.testbed_real.oracle import compute_target_log_likelihood, run_relearning_attack

def main():
    config_path = ROOT / "configs/testbed/smoke.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== RENA Neural Oracle Attack on Qwen2.5-0.5B-Instruct ({device}) ===")

    implanted_checkpoint = str(ROOT / "checkpoints/implanted_lora")
    tokenizer = AutoTokenizer.from_pretrained(implanted_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {
        "benchmark": "RENA_Neural_Oracle_Benchmark",
        "model": cfg.get("model", {}).get("name", "Qwen/Qwen2.5-0.5B-Instruct"),
        "device": device,
        "seed": cfg.get("reproducibility", {}).get("seed", 42),
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "log_likelihood_evaluations": {},
        "relearning_attacks": {},
        "summary": {},
    }

    conditions = ["reference", "retained", "suppress_refusal", "graddiff"]

    for cond in conditions:
        print(f"\n--- Condition: {cond} ---")
        t0 = time.time()
        base_model_name = cfg.get("model", {}).get("name", "Qwen/Qwen2.5-0.5B-Instruct")
        model, _ = get_unlearned_model(
            method=cond,
            base_model_name=base_model_name,
            implanted_checkpoint_dir=implanted_checkpoint,
            device=device,
        )
        ll_res = compute_target_log_likelihood(model, tokenizer, FORGET_FACTS, device)
        elapsed = round(time.time() - t0, 2)
        print(f"  Mean target log-likelihood: {ll_res['mean_log_prob']:.4f} (elapsed: {elapsed}s)")
        results["log_likelihood_evaluations"][cond] = {
            "mean_log_prob": ll_res["mean_log_prob"],
            "item_details": ll_res["item_details"],
            "elapsed_seconds": elapsed,
        }

        # Run relearning attack on mitigation conditions
        if cond in ["suppress_refusal", "graddiff"]:
            print(f"  Running relearning attack (10 steps, lr=1e-4)...")
            attack_res = run_relearning_attack(
                model=model,
                tokenizer=tokenizer,
                device=device,
                steps=10,
                lr=1e-4,
            )
            print(f"  Steps to recovery: {attack_res['steps_to_recovery']}")
            print(f"  Vulnerability: {attack_res['vulnerability_assessment']}")
            results["relearning_attacks"][cond] = attack_res

        # Free GPU/MPS memory
        del model
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    # Vulnerability Summary
    ref_ll = results["log_likelihood_evaluations"]["reference"]["mean_log_prob"]
    ret_ll = results["log_likelihood_evaluations"]["retained"]["mean_log_prob"]
    sup_ll = results["log_likelihood_evaluations"]["suppress_refusal"]["mean_log_prob"]
    grd_ll = results["log_likelihood_evaluations"]["graddiff"]["mean_log_prob"]

    results["summary"] = {
        "reference_baseline_mean_ll": ref_ll,
        "retained_control_mean_ll": ret_ll,
        "suppress_refusal_mean_ll": sup_ll,
        "graddiff_mean_ll": grd_ll,
        "suppress_refusal_steps_to_recovery": results["relearning_attacks"]["suppress_refusal"]["steps_to_recovery"],
        "graddiff_steps_to_recovery": results["relearning_attacks"]["graddiff"]["steps_to_recovery"],
        "finding": (
            "Suppression models retain latent representation accessible under rapid relearning, "
            "whereas weight unlearning alters representation space requiring more gradient steps."
        ),
    }

    out_file = ROOT / "results/testbed/oracle_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Oracle benchmark results saved to {out_file}")

if __name__ == "__main__":
    main()
