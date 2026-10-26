"""Aggregate matched RENA LLM suite summaries across random seeds.

Point this at a parent folder containing one suite output directory per seed.
Only existing suite_summary.json files are read; no missing values are imputed.
"""
from __future__ import annotations
import argparse, csv, json, math, random
from pathlib import Path
from statistics import mean, stdev

METRICS = ("forget_parametric_disclosure_rate", "forget_contextual_retrieval_rate",
           "retain_retrieval_rate", "retain_contextual_retrieval_rate", "elapsed_seconds")


def bootstrap_mean_ci(values, seed=2026, iterations=4000):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(iterations))
    return means[int(.025 * (iterations - 1))], means[int(.975 * (iterations - 1))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Parent directory containing seed-specific suite outputs")
    p.add_argument("--output", default="results/llm/aggregate")
    a = p.parse_args()
    root = Path(a.input)
    summaries = sorted(root.rglob("suite_summary.json"))
    if not summaries:
        p.error(f"No suite_summary.json files found under {root}")
    runs = []
    for path in summaries:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Cannot read {path}: {exc}")
        runs.append((path, data))
    methods = sorted({m for _, run in runs for m in run.get("methods", {})})
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in methods:
        for metric in METRICS:
            vals = []
            for _, run in runs:
                item = run.get("methods", {}).get(method, {})
                value = item.get(metric)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    vals.append(float(value))
            if not vals:
                continue
            lo, hi = bootstrap_mean_ci(vals, seed=2026 + len(rows))
            rows.append({"method": method, "metric": metric, "n_runs": len(vals),
                         "mean": mean(vals), "std": stdev(vals) if len(vals) > 1 else 0.0,
                         "bootstrap_95ci_low": lo, "bootstrap_95ci_high": hi})
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["method", "metric", "n_runs", "mean", "std", "bootstrap_95ci_low", "bootstrap_95ci_high"])
        writer.writeheader(); writer.writerows(rows)
    payload = {"analysis": "RENA-LLM-seed-aggregation-v1", "runs": [str(p) for p, _ in runs],
               "run_count": len(runs), "metrics": rows,
               "limitations": ["Intervals describe variation across supplied runs; they do not correct for benchmark or hyperparameter selection.",
                               "Use identical model, dataset, protocol and configuration across seeds for valid comparisons."]}
    (out / "aggregate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Aggregated {len(runs)} runs into {out / 'metrics.csv'}")

if __name__ == "__main__": main()
