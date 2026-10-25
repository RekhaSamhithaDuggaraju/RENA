"""Seed-level summaries for matched RENA LLM experiments.

Bootstrap resampling is over independent run seeds, never individual prompts.
Intervals are descriptive and should not be interpreted as population guarantees
when only a few seeds are available.
"""
from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

METRICS = (
    "forget_parametric_disclosure_rate",
    "forget_contextual_retrieval_rate",
    "retain_retrieval_rate",
    "retain_contextual_retrieval_rate",
    "elapsed_seconds",
)


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def summarize(values: Sequence[float | int | None], *, bootstrap_samples: int = 2000, seed: int = 42) -> dict[str, Any]:
    clean = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be >= 100")
    if not clean:
        return {"n": 0, "mean": None, "std": None, "ci95": [None, None]}
    mean = sum(clean) / len(clean)
    std = math.sqrt(sum((x - mean) ** 2 for x in clean) / (len(clean) - 1)) if len(clean) > 1 else 0.0
    rng = random.Random(seed)
    boot_means = [sum(rng.choices(clean, k=len(clean))) / len(clean) for _ in range(bootstrap_samples)]
    return {"n": len(clean), "mean": mean, "std": std,
            "ci95": [_percentile(boot_means, .025), _percentile(boot_means, .975)],
            "interval_note": "Percentile bootstrap over run seeds; descriptive, not a prompt-level confidence interval."}


def aggregate_suite_summaries(summaries: Sequence[Mapping[str, Any]], bootstrap_samples: int = 2000) -> dict[str, Any]:
    """Aggregate suite_summary objects, preserving matched per-seed contrasts."""
    if not summaries:
        raise ValueError("at least one suite summary is required")
    models = {str(s.get("base_model")) for s in summaries}
    protocols = {str(s.get("protocol")) for s in summaries}
    if len(models) != 1 or len(protocols) != 1:
        raise ValueError("all summaries must use the same base model and protocol")
    seeds = [s.get("seed") for s in summaries]
    if any(not isinstance(s, int) for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("each summary must have a unique integer seed")
    method_names = set.intersection(*(set(s.get("methods", {})) for s in summaries))
    methods: dict[str, Any] = {}
    for method in sorted(method_names):
        metrics: dict[str, Any] = {}
        for metric in METRICS:
            vals = [s["methods"][method].get(metric) for s in summaries]
            metrics[metric] = summarize(vals, bootstrap_samples=bootstrap_samples,
                                        seed=int(hashlib.sha256(f"{method}:{metric}".encode()).hexdigest()[:8], 16))
        methods[method] = metrics
    reference = "retain_only_retraining"
    contrasts: dict[str, Any] = {}
    if all(reference in s.get("methods", {}) for s in summaries):
        for method in sorted(method_names - {reference}):
            contrasts[method] = {}
            for metric in METRICS:
                pairs = [(s["methods"][method].get(metric), s["methods"][reference].get(metric)) for s in summaries]
                deltas = [float(a) - float(b) for a, b in pairs if a is not None and b is not None]
                contrasts[method][metric] = summarize(deltas, bootstrap_samples=bootstrap_samples,
                    seed=int(hashlib.sha256(f"delta:{method}:{metric}".encode()).hexdigest()[:8], 16))
                contrasts[method][metric]["contrast"] = f"{method} minus {reference}, paired by seed"
    return {"analysis": "RENA-LLM-seed-aggregate-v1", "base_model": next(iter(models)),
            "protocol": next(iter(protocols)), "seeds": sorted(seeds), "seed_count": len(seeds),
            "methods": methods, "paired_differences_vs_retain_only": contrasts,
            "limitations": ["Inference is limited by the number and diversity of run seeds.",
                "Bootstrap intervals resample seeds, not prompts or individual tokens.",
                "Synthetic knowledge results do not establish real-world unlearning or parameter erasure."]}
