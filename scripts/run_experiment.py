#!/usr/bin/env python3
"""Unified Reproducible Experiment Runner for RENA.

Provides a single documented entrypoint for all empirical benchmarks:
- Quick smoke mode (5 facts x 5 conditions on Qwen2.5-0.5B-Instruct)
- Oracle benchmark (log-likelihood + relearning attack)
- Hosted Gemini case study (with valid API key in header)
- Resource estimation and prerequisite validation
- Cryptographic SHA-256 manifest generation
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def estimate_resources(mode: str, split: str = "smoke"):
    """Estimates RAM, compute, and runtime without execution."""
    print("=" * 60)
    print(f"RENA RESOURCE ESTIMATION — Mode: {mode} (Split: {split})")
    print("=" * 60)
    if mode == "smoke":
        facts_n = 5
        trials_per_fact = 3 * 3  # 3 baseline reps + 3 post-forget probe types
        conditions = 5  # ref, ret, sup_p, sup_r, graddiff
        total_inferences = facts_n * trials_per_fact * conditions + 25  # + retain facts
        print(f"Target Model: Qwen/Qwen2.5-0.5B-Instruct (FP16)")
        print(f"Estimated Peak VRAM / Unified Memory: ~1.2 GB")
        print(f"Estimated Total Inferences: ~{total_inferences}")
        print(f"Estimated Runtime on Apple Silicon MPS: ~45–60 seconds")
        print(f"Estimated Runtime on CPU: ~4–6 minutes")
        print(f"API Cost: $0.00 (Local open weights)")
    elif mode == "oracle":
        print(f"Target Model: Qwen/Qwen2.5-0.5B-Instruct (FP16)")
        print(f"Estimated Peak VRAM / Unified Memory: ~1.5 GB")
        print(f"AdamW Steps: 10 steps per condition (suppress_refusal, graddiff)")
        print(f"Estimated Runtime on Apple Silicon MPS: ~25–35 seconds")
        print(f"API Cost: $0.00 (Local open weights)")
    elif mode == "confirmatory":
        facts_n = 30
        print(f"Target Model: Qwen/Qwen2.5-0.5B-Instruct (FP16)")
        print(f"Estimated Peak Unified Memory: ~2.0 GB")
        print(f"Estimated Runtime on Apple Silicon MPS: ~5–8 minutes")
        print(f"API Cost: $0.00 (Local open weights)")
    elif mode == "gemini":
        print(f"Target Model: gemini-3.5-flash-lite (Hosted API)")
        print(f"Estimated API Calls: 48 calls")
        print(f"Estimated Latency: ~30–45 seconds")
        print(f"Estimated Cost: < $0.01")
    print("=" * 60)

def validate_prerequisites(device: str = "auto") -> dict:
    """Validates local environment, checkpoints, and dependencies."""
    checks = {}
    
    # 1. Check Python and Torch
    try:
        import torch
        checks["torch_version"] = torch.__version__
        has_mps = torch.backends.mps.is_available()
        has_cuda = torch.cuda.is_available()
        checks["mps_available"] = has_mps
        checks["cuda_available"] = has_cuda
        if device == "auto":
            selected_device = "mps" if has_mps else ("cuda" if has_cuda else "cpu")
        else:
            selected_device = device
        checks["selected_device"] = selected_device
    except ImportError as e:
        checks["torch"] = f"MISSING ({e})"
        return {"status": "FAIL", "checks": checks}

    # 2. Check HuggingFace Transformers & Peft
    try:
        import transformers
        import peft
        checks["transformers_version"] = transformers.__version__
        checks["peft_version"] = peft.__version__
    except ImportError as e:
        checks["transformers_peft"] = f"MISSING ({e})"
        return {"status": "FAIL", "checks": checks}

    # 3. Check Implanted LoRA Checkpoint
    implanted_path = ROOT / "checkpoints" / "implanted_lora"
    adapter_cfg = implanted_path / "adapter_config.json"
    checks["implanted_lora_checkpoint_exists"] = adapter_cfg.exists()

    all_passed = checks.get("implanted_lora_checkpoint_exists", False)
    return {"status": "PASS" if all_passed else "WARNING_CHECKPOINT_MISSING", "checks": checks}

def update_manifest():
    """Generates SHA-256 manifest across all active artifacts."""
    manifest_path = ROOT / "results" / "MANIFEST.json"
    if not manifest_path.exists():
        manifest = {"manifest_version": "2.2", "artifacts": {}}
    else:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    artifacts = manifest.get("artifacts", {})
    results_dir = ROOT / "results"

    # Scan results/ and relevant docs
    tracked_paths = [
        "results/testbed/smoke_audit.json",
        "results/testbed/oracle_results.json",
        "results/case_studies/gemini_flash_lite_audit.json",
        "results/EVIDENCE_INDEX.md",
        "data/annotation/validation_400_unlabeled.csv",
        "data/annotation/validation_400_ground_truth.json",
        "configs/testbed/smoke.yaml",
        "docs/VERDICT_SPEC.md",
        "docs/SCORER_SPEC.md",
        "docs/PROBE_CATALOG.md",
        "docs/CLAIMS_LEDGER.md",
        "docs/PAPER_EVALUATION_REPORT.md",
        "docs/PHASE_LOG.md",
        "AUDIT_REPORT.md",
        "scripts/run_neural_smoke_comparison.py",
        "scripts/run_neural_oracle_benchmark.py",
        "scripts/run_experiment.py",
    ]

    for rel in tracked_paths:
        p = ROOT / rel
        if p.exists():
            content = p.read_bytes()
            artifacts[rel] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
                "exists": True,
            }
        else:
            artifacts[rel] = {"exists": False}

    manifest["manifest_version"] = "2.2"
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["artifacts"] = artifacts

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[MANIFEST] Updated {manifest_path} ({len(artifacts)} artifacts tracked).")

def main():
    p = argparse.ArgumentParser(description="RENA Unified Experiment Runner")
    p.add_argument("--mode", choices=["smoke", "oracle", "confirmatory", "gemini", "all_neural", "validate"], default="smoke")
    p.add_argument("--split", choices=["smoke", "confirmatory_30", "all"], default="smoke")
    p.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--estimate-resources", action="store_true", help="Print resource estimates and exit")
    p.add_argument("--update-manifest", action="store_true", help="Re-hash all active evidence artifacts")
    args = p.parse_args()

    if args.estimate_resources:
        estimate_resources(args.mode, args.split)
        sys.exit(0)

    if args.update_manifest:
        update_manifest()
        sys.exit(0)

    print(f"=== RENA Reproducible Experiment Runner ===")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"System: {platform.system()} {platform.machine()} (Release: {platform.release()})")
    print(f"Requested Mode: {args.mode} | Split: {args.split} | Seed: {args.seed}")

    prereqs = validate_prerequisites(args.device)
    print(f"Prerequisite Check: {prereqs['status']}")
    for k, v in prereqs["checks"].items():
        print(f"  - {k}: {v}")

    if prereqs["status"] == "FAIL":
        print("\n[ERROR] Missing required dependencies. Aborting.")
        sys.exit(1)

    py_exe = sys.executable

    # Execute Mode
    if args.mode in ["smoke", "all_neural"]:
        print("\n>>> Launching Neural Smoke Comparison Benchmark...")
        cmd = [py_exe, str(ROOT / "scripts" / "run_neural_smoke_comparison.py")]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"[ERROR] Smoke benchmark failed with code {res.returncode}")
            sys.exit(res.returncode)

    if args.mode in ["oracle", "all_neural"]:
        print("\n>>> Launching Independent Neural Oracle Benchmark...")
        cmd = [py_exe, str(ROOT / "scripts" / "run_neural_oracle_benchmark.py")]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"[ERROR] Oracle benchmark failed with code {res.returncode}")
            sys.exit(res.returncode)

    if args.mode == "gemini":
        print("\n>>> Launching Hosted Gemini Case Study...")
        if not os.environ.get("GEMINI_API_KEY"):
            print("[ERROR] GEMINI_API_KEY environment variable is not set. Aborting hosted run.")
            sys.exit(1)
        cmd = [py_exe, str(ROOT / "scripts" / "reproduce_gemini_case.py")]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"[ERROR] Gemini case study failed with code {res.returncode}")
            sys.exit(res.returncode)

    # Automatically re-hash active artifacts after any run
    update_manifest()
    print("\n[SUCCESS] Experiment run completed successfully.")

if __name__ == "__main__":
    main()
