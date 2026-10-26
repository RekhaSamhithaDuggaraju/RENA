"""Run controlled adaptive-vs-fixed audit policy simulation.

Example: python run_audit_policy_benchmark.py --count 500 --seed 42 --output results/audit_policy_simulation.json
This is a synthetic controller validation, not an LLM experiment.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from core.llm.audit_benchmark import benchmark_audit_policies


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--count", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = benchmark_audit_policies(count=a.count, seed=a.seed)
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v["metrics"] for k, v in result["policies"].items()}, indent=2))
    print(f"Saved synthetic policy simulation to {path}")

if __name__ == "__main__":
    main()
