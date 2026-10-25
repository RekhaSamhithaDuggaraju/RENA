"""Run RENA's controlled NIAH benchmark against a local or HF model path.

Example:
  python3 run_niah_benchmark.py --model Qwen/Qwen2.5-0.5B-Instruct --output results/llm/niah.json
A model may be loaded from a local directory to avoid a network download.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np

from core.llm import build_niah_cases, build_synthetic_knowledge, run_niah_benchmark
from core.llm.hf_adapter import HuggingFaceGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local model directory")
    parser.add_argument("--output", default="results/llm/niah.json")
    parser.add_argument("--context-sizes", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--context-modes", nargs="+", choices=["present", "absent"], default=["present", "absent"], help="present tests contextual retrieval; absent probes parametric recall")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    facts = build_synthetic_knowledge()
    cases = build_niah_cases(facts, context_sizes=args.context_sizes, context_modes=args.context_modes)
    generate = HuggingFaceGenerator(args.model, device=args.device, max_new_tokens=args.max_new_tokens)
    result = run_niah_benchmark(cases, generate)
    result["model"] = args.model
    result["seed"] = args.seed
    result["generation"] = {"do_sample": False, "max_new_tokens": args.max_new_tokens, "device": generate.device}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("benchmark", "model", "case_count", "forget_disclosure_rate", "forget_parametric_disclosure_rate", "retain_retrieval_rate")}, indent=2))
    print(f"Saved detailed case outputs to {path}")


if __name__ == "__main__":
    main()
