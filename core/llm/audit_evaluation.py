"""Evaluation utilities for labeled, per-case RENA audit decisions.

This evaluates supplied decisions; it does not create ground truth or claim
that synthetic policy simulations represent trained-model performance.
"""
from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

DECISIVE = {"SUPPORTED", "NOT_SUPPORTED"}
VALID = DECISIVE | {"PARTIALLY_SUPPORTED", "ABSTAIN", "INSUFFICIENT_EVIDENCE"}


def _validate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("at least one labeled decision is required")
    clean = []
    seen = set()
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"row {i} must be a mapping")
        case_id = row.get("case_id")
        status = row.get("status")
        truth = row.get("truth_supported")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"row {i} requires a non-empty case_id")
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        if status not in VALID:
            raise ValueError(f"{case_id}: invalid status {status!r}")
        if not isinstance(truth, bool):
            raise ValueError(f"{case_id}: truth_supported must be boolean")
        cost = row.get("cost_units", 0.0)
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost < 0:
            raise ValueError(f"{case_id}: cost_units must be non-negative")
        clean.append({**row, "cost_units": float(cost)})
    return clean


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    decisive = [r for r in rows if r["status"] in DECISIVE]
    errors = [r for r in decisive if (r["status"] == "SUPPORTED") != r["truth_supported"]]
    tp = sum(r["status"] == "SUPPORTED" and r["truth_supported"] for r in rows)
    fp = sum(r["status"] == "SUPPORTED" and not r["truth_supported"] for r in rows)
    tn = sum(r["status"] == "NOT_SUPPORTED" and not r["truth_supported"] for r in rows)
    fn = sum(r["status"] == "NOT_SUPPORTED" and r["truth_supported"] for r in rows)
    supported = sum(r["status"] == "SUPPORTED" for r in rows)
    not_supported = sum(r["status"] == "NOT_SUPPORTED" for r in rows)
    return {
        "n": n, "coverage": len(decisive) / n,
        "selective_risk": len(errors) / len(decisive) if decisive else None,
        "false_confident_support_rate": fp / n,
        "support_precision": tp / supported if supported else None,
        "support_recall": tp / (tp + fn) if (tp + fn) else None,
        "not_supported_precision": tn / not_supported if not_supported else None,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "abstention_rate": sum(r["status"] in {"ABSTAIN", "INSUFFICIENT_EVIDENCE"} for r in rows) / n,
        "partial_rate": sum(r["status"] == "PARTIALLY_SUPPORTED" for r in rows) / n,
        "mean_cost_units": sum(r["cost_units"] for r in rows) / n,
    }


def _bootstrap(rows: Sequence[Mapping[str, Any]], *, seed: int, iterations: int) -> dict[str, Any]:
    rng = random.Random(seed)
    n = len(rows)
    risks, coverages, fps = [], [], []
    for _ in range(iterations):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        m = _metrics(sample)
        coverages.append(m["coverage"])
        fps.append(m["false_confident_support_rate"])
        if m["selective_risk"] is not None:
            risks.append(m["selective_risk"])
    def ci(values: list[float]) -> list[float] | None:
        if not values:
            return None
        values.sort()
        return [values[int(.025 * (len(values)-1))], values[int(.975 * (len(values)-1))]]
    return {"iterations": iterations, "seed": seed, "coverage_95ci": ci(coverages),
            "selective_risk_95ci": ci(risks), "false_confident_support_rate_95ci": ci(fps),
            "note": "Percentile bootstrap over supplied cases; case dependence can invalidate this interval."}


def evaluate_audit_decisions(rows: Sequence[Mapping[str, Any]], *, seed: int = 42,
                             bootstrap_iterations: int = 2000) -> dict[str, Any]:
    """Summarize externally labeled decisions with bootstrap uncertainty."""
    if bootstrap_iterations < 100:
        raise ValueError("bootstrap_iterations must be >= 100")
    clean = _validate(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clean:
        grouped[str(row.get("scenario_type", "unspecified"))].append(row)
    return {"benchmark": "rena-labeled-audit-decisions-v1", "n": len(clean),
            "metrics": _metrics(clean),
            "bootstrap": _bootstrap(clean, seed=seed, iterations=bootstrap_iterations),
            "by_scenario": {k: _metrics(v) for k, v in sorted(grouped.items())},
            "limitations": ["Ground-truth labels must be defined independently of the audit decision.",
                "Metrics evaluate decision correctness on supplied cases, not physical parameter deletion.",
                "Use held-out model runs for final reporting; do not tune thresholds on these cases."]}
