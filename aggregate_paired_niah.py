"""Aggregate per-seed paired NIAH comparison JSON files."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from core.llm.paired_niah_aggregate import aggregate_paired_niah

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('comparisons', nargs='+', help='Per-seed paired comparison JSON files')
    p.add_argument('--output', required=True)
    p.add_argument('--bootstrap-samples', type=int, default=2000)
    a = p.parse_args()
    items = [json.loads(Path(f).read_text(encoding='utf-8')) for f in a.comparisons]
    result = aggregate_paired_niah(items, bootstrap_samples=a.bootstrap_samples)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"Aggregated {result['seed_count']} seeds: {out}")

if __name__ == '__main__': main()
