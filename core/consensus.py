"""
RENA core — Consensus Engine.

Implements the spec from rena-phase1-and-consensus-spec.md section 2.
This is NOT a naive average: the agreement/disagreement override in
resolve_verdict() is what forces "Inconclusive" when verification modules
disagree, even if the weighted average score looks confident.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

from core.verification import ModuleResult

AGREEMENT_THRESHOLD = 0.5
CONFIDENT_THRESHOLD_HIGH = 0.75
CONFIDENT_THRESHOLD_LOW = 0.35
AGREEMENT_REQUIRED_FOR_CONFIDENCE = 0.7
MAX_EXPECTED_VARIANCE = 0.12  # calibrated against baseline sanity checks; documented, versioned


@dataclass
class ConsensusResult:
    verdict: str
    consensus_score: float
    agreement: float
    per_module_contribution: list
    discounted_modules: list

    def to_dict(self) -> dict:
        return asdict(self)


def compute_consensus(module_results: list[ModuleResult]) -> ConsensusResult:
    return compute_consensus_strategy(module_results, weighting=True, override=True)


def compute_consensus_strategy(module_results: list[ModuleResult], *, weighting: bool = True,
                               override: bool = True) -> ConsensusResult:
    """Evaluate a named consensus component ablation without changing RENA defaults.

    ``weighting=False`` is the explicit equal-weight baseline and
    ``override=False`` reports a forced threshold verdict.  The public
    :func:`compute_consensus` remains the original RENA_FULL behaviour.
    """
    if not module_results:
        raise ValueError("at least one verification module is required")
    scores = np.array([m.forget_score for m in module_results])
    reliabilities = np.array([m.reliability for m in module_results])

    total_reliability = np.sum(reliabilities)
    if not weighting or total_reliability <= 0:
        consensus_score = float(np.mean(scores))
        weights = np.ones_like(scores) / len(scores)
    else:
        consensus_score = float(np.sum(scores * reliabilities) / total_reliability)
        weights = reliabilities / total_reliability

    weighted_variance = float(
        np.sum(reliabilities * (scores - consensus_score) ** 2) / total_reliability
        if total_reliability > 0 else np.var(scores)
    )
    agreement = float(np.clip(1 - weighted_variance / MAX_EXPECTED_VARIANCE, 0, 1))

    verdict = resolve_verdict(consensus_score, agreement) if override else resolve_forced_verdict(consensus_score)

    per_module_contribution = [
        {
            "module": m.module_name,
            "forget_score": m.forget_score,
            "reliability": m.reliability,
            "weight_pct": float(w * 100),
        }
        for m, w in zip(module_results, weights)
    ]

    discounted_modules = [
        {"module": m.module_name, "reason": m.reliability_reason, "reliability": m.reliability}
        for m in module_results
        if m.reliability < 0.99
    ]

    return ConsensusResult(
        verdict=verdict,
        consensus_score=consensus_score,
        agreement=agreement,
        per_module_contribution=per_module_contribution,
        discounted_modules=discounted_modules,
    )


def resolve_forced_verdict(consensus_score: float) -> str:
    """Threshold verdict used only by no-override comparison baselines."""
    if consensus_score >= CONFIDENT_THRESHOLD_HIGH:
        return "Likely Forgotten"
    if consensus_score <= CONFIDENT_THRESHOLD_LOW:
        return "Not Forgotten"
    return "Ambiguous"


def consensus_baselines(module_results: list[ModuleResult]) -> dict[str, dict]:
    """Return comparable, explicit strategies, including the 2x2 ablation."""
    configs = {
        "RENA_FULL": (True, True),
        "WEIGHTED_NO_OVERRIDE": (True, False),
        "NAIVE_AVERAGE": (False, False),
        "FIXED_THRESHOLD": (False, False),
        "WEIGHTING_ON_OVERRIDE_ON": (True, True),
        "WEIGHTING_ON_OVERRIDE_OFF": (True, False),
        "WEIGHTING_OFF_OVERRIDE_ON": (False, True),
        "WEIGHTING_OFF_OVERRIDE_OFF": (False, False),
    }
    return {name: compute_consensus_strategy(module_results, weighting=weighting,
                                               override=override).to_dict()
            for name, (weighting, override) in configs.items()}


def resolve_verdict(consensus_score: float, agreement: float) -> str:
    """
    The override rule from the spec: low agreement forces "Inconclusive"
    REGARDLESS of how high or low the average score is. This is the property
    that distinguishes a real consensus engine from naive averaging.
    """
    if agreement < AGREEMENT_THRESHOLD:
        return "Inconclusive"
    if consensus_score >= CONFIDENT_THRESHOLD_HIGH and agreement >= AGREEMENT_REQUIRED_FOR_CONFIDENCE:
        return "Likely Forgotten"
    if consensus_score <= CONFIDENT_THRESHOLD_LOW and agreement >= AGREEMENT_REQUIRED_FOR_CONFIDENCE:
        return "Not Forgotten"
    return "Ambiguous"
