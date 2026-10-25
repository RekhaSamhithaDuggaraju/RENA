"""Controlled simulation benchmark for audit policies (not model-performance data).

Compares adaptive escalation with cheap-only and fixed-all policies on the same
labeled evidence scenarios. Scores are synthetic verifier outputs generated from
known latent labels; results validate policy behavior, not real unlearning.
"""
from __future__ import annotations

import random
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from .audit_controller import AuditPolicy, run_adaptive_audit


def _clip(x: float) -> float:
    return min(1.0, max(0.0, x))


def generate_scenarios(*, count: int = 500, seed: int = 42) -> list[dict[str, Any]]:
    """Generate balanced synthetic cases with easy, ambiguous and conflicting evidence."""
    if count < 20:
        raise ValueError("count must be at least 20")
    rng = random.Random(seed)
    scenarios = []
    for i in range(count):
        truth = i % 2 == 0  # balanced claim-supported / claim-not-supported labels
        kind = (i // 2) % 4
        center = 0.88 if truth else 0.12
        if kind == 0:  # clear agreement: escalation should usually be avoided
            cheap = [_clip(rng.gauss(center, .06)) for _ in range(2)]
            strong = [_clip(rng.gauss(center, .04)) for _ in range(2)]
        elif kind == 1:  # ambiguous cheap evidence; stronger checks resolve it
            cheap = [_clip(rng.gauss(.5, .08)) for _ in range(2)]
            strong = [_clip(rng.gauss(center, .05)) for _ in range(2)]
        elif kind == 2:  # misleading cheap evidence; escalation exposes it
            cheap = [_clip(rng.gauss(1-center, .05)) for _ in range(2)]
            strong = [_clip(rng.gauss(center, .04)) for _ in range(2)]
        else:  # intentionally conflicting evidence should often abstain
            cheap = [_clip(rng.gauss(center, .04)), _clip(rng.gauss(1-center, .04))]
            strong = [_clip(rng.gauss(center, .06)), _clip(rng.gauss(1-center, .06))]
        scenarios.append({"case_id": f"sim-{i:05d}", "truth_supported": truth,
                          "scenario_type": ("clear", "ambiguous", "misleading_cheap", "conflicting")[kind],
                          "scores": {"cheap_a": cheap[0], "cheap_b": cheap[1],
                                     "strong_a": strong[0], "strong_b": strong[1]}})
    return scenarios


def _check(score: float, name: str, group: str) -> dict[str, Any]:
    return {"score": score, "reliability": 1.0, "evidence_group": group,
            "metric": name, "evidence_id": name, "details": {"source": "synthetic_policy_simulation"}}


def _decision(scenario: Mapping[str, Any], policy_name: str, policy: AuditPolicy) -> tuple[str, int, bool]:
    s = scenario["scores"]
    initial = {"cheap_a": lambda: _check(s["cheap_a"], "cheap_a", "cheap-a"),
               "cheap_b": lambda: _check(s["cheap_b"], "cheap_b", "cheap-b")}
    strong = {"strong_a": lambda: _check(s["strong_a"], "strong_a", "strong-a"),
              "strong_b": lambda: _check(s["strong_b"], "strong_b", "strong-b")}
    if policy_name == "adaptive":
        d = run_adaptive_audit(initial, strong, policy)
    elif policy_name == "adaptive_sentinel":
        # Stable, predeclared 20% sentinel sample of otherwise non-escalated cases.
        digest = hashlib.sha256(str(scenario["case_id"]).encode()).hexdigest()
        force = int(digest[:8], 16) % 5 == 0
        d = run_adaptive_audit(initial, strong, policy, force_escalation=force)
    elif policy_name == "cheap_only":
        d = run_adaptive_audit(initial, {}, policy)
    elif policy_name == "fixed_all":
        d = run_adaptive_audit({**initial, **strong}, {}, policy)
    else:
        raise ValueError(f"unknown policy: {policy_name}")
    # Cost units are explicit policy assumptions, not measured wall-clock time.
    cost = sum(1 if row["stage"] == "initial" and row["verifier"].startswith("cheap")
               else 4 for row in d.evidence)
    return d.status, cost, d.escalated


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    decisive = [r for r in rows if r["status"] in ("SUPPORTED", "NOT_SUPPORTED")]
    errors = [r for r in decisive if (r["status"] == "SUPPORTED") != r["truth_supported"]]
    fp = [r for r in rows if r["status"] == "SUPPORTED" and not r["truth_supported"]]
    return {"n": len(rows), "coverage": len(decisive) / len(rows) if rows else None,
            "selective_risk": len(errors) / len(decisive) if decisive else None,
            "false_confident_support_rate": len(fp) / len(rows) if rows else None,
            "mean_cost_units": sum(r["cost_units"] for r in rows) / len(rows) if rows else None,
            "escalation_rate": sum(bool(r["escalated"]) for r in rows) / len(rows) if rows else None,
            "abstention_rate": sum(r["status"] in ("ABSTAIN", "INSUFFICIENT_EVIDENCE", "PARTIALLY_SUPPORTED") for r in rows) / len(rows) if rows else None}


def benchmark_audit_policies(*, count: int = 500, seed: int = 42,
                             policy: AuditPolicy = AuditPolicy()) -> dict[str, Any]:
    scenarios = generate_scenarios(count=count, seed=seed)
    out: dict[str, list[dict[str, Any]]] = {k: [] for k in ("adaptive", "adaptive_sentinel", "cheap_only", "fixed_all")}
    for case in scenarios:
        for name in out:
            status, cost, escalated = _decision(case, name, policy)
            out[name].append({"case_id": case["case_id"], "truth_supported": case["truth_supported"],
                              "scenario_type": case["scenario_type"], "status": status,
                              "cost_units": cost, "escalated": escalated})
    return {"benchmark": "RENA-controlled-audit-policy-simulation-v1", "seed": seed,
            "case_count": count, "policy": policy.__dict__,
            "policies": {name: {"metrics": _metrics(rows), "cases": rows} for name, rows in out.items()},
            "cost_definition": {"cheap_check": 1, "strong_check": 4,
                                "note": "Assumed relative cost units; not measured runtime or API cost."},
            "limitations": ["Synthetic verifier scores are not generated by trained models.",
                "This benchmark tests controller mechanics, not real-world unlearning validity.",
                "Reliability weights and thresholds are policy assumptions, not calibrated probabilities.",
                "The benchmark must be supplemented with held-out real-model experiments before research claims.",
                "Sentinel escalation uses a fixed hash-based 20% sample for reproducibility; it is not cryptographic randomness."]}
