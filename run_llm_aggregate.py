"""Aggregate matched RENA LLM suite runs across independent seeds.

Example:
  python run_llm_aggregate.py results/llm/seed_42/suite_summary.json ... --output results/llm/aggregate.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from core.llm.statistics import aggregate_suite_summaries


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summaries", nargs="+", help="Paths to suite_summary.json files from distinct seeds")
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    a = p.parse_args()
    items = [json.loads(Path(path).read_text(encoding="utf-8")) for path in a.summaries]
    result = aggregate_suite_summaries(items, bootstrap_samples=a.bootstrap_samples)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Aggregated {result['seed_count']} seeds across {len(result['methods'])} methods: {out}")

if __name__ == "__main__":
    main()
