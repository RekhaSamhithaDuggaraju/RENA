"""Evaluate labeled RENA audit decisions from a JSON array or {cases:[...]} file.

Example: python evaluate_audit_decisions.py --input cases.json --output results/audit_eval.json
Each case needs case_id, status, and independently assigned truth_supported.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from core.llm.audit_evaluation import evaluate_audit_decisions


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="JSON array or object containing cases")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = p.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise SystemExit("input must be a JSON array or an object with a 'cases' array")
    result = evaluate_audit_decisions(cases, seed=args.seed,
                                      bootstrap_iterations=args.bootstrap_iterations)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Saved evaluation to {target}")

if __name__ == "__main__":
    main()
