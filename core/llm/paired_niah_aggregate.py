"""Aggregate paired NIAH-vs-reference comparisons across independent seeds."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import hashlib
from typing import Any
from .statistics import summarize

METRICS = ("forget_absent_disclosure", "forget_present_retrieval", "retain_present_retrieval")

def aggregate_paired_niah(comparisons: Sequence[Mapping[str, Any]], *, bootstrap_samples: int = 2000) -> dict[str, Any]:
    if not comparisons:
        raise ValueError("at least one paired comparison is required")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be >= 100")
    candidates = {c.get("candidate") for c in comparisons}
    references = {c.get("reference") for c in comparisons}
    seeds = [c.get("seed") for c in comparisons]
    if len(candidates) != 1 or len(references) != 1:
        raise ValueError("all comparisons must use the same candidate and reference")
    if any(not isinstance(seed, int) for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("each comparison must have a unique integer seed")
    metrics: dict[str, Any] = {}
    for metric in METRICS:
        rows = [c.get(metric, {}) for c in comparisons]
        diffs = [r.get("difference") for r in rows]
        summary = summarize(diffs, bootstrap_samples=bootstrap_samples,
                            seed=int(hashlib.sha256(f"{next(iter(candidates))}:{metric}".encode()).hexdigest()[:8], 16))
        summary["seed_level_differences"] = diffs
        summary["mean_candidate_rate"] = summarize([r.get("candidate_rate") for r in rows], bootstrap_samples=bootstrap_samples)["mean"]
        summary["mean_reference_rate"] = summarize([r.get("reference_rate") for r in rows], bootstrap_samples=bootstrap_samples)["mean"]
        summary["case_counts"] = [r.get("n", 0) for r in rows]
        summary["interpretation"] = "Candidate minus reference retrieval rate; positive means more retrieval by candidate."
        metrics[metric] = summary
    return {"analysis": "RENA-paired-NIAH-seed-aggregate-v1", "candidate": next(iter(candidates)),
            "reference": next(iter(references)), "seeds": sorted(seeds), "seed_count": len(seeds),
            "metrics": metrics, "limitations": [
                "Intervals resample independent run seeds, not individual prompts.",
                "Few seeds yield unstable descriptive intervals.",
                "Synthetic NIAH behavior does not prove parameter erasure or real-world unlearning."]}
