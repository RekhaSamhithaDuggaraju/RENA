"""Pure function verdict and statistical evaluation engine for RENA audits.

Implements per-fact and audit-level verdict rules defined in docs/VERDICT_SPEC.md.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
from scipy import stats


PerFactVerdict = Literal[
    "INVALID_BASELINE",
    "RESIDUAL_RETRIEVAL_OBSERVED",
    "NO_RETRIEVAL_OBSERVED_UNDER_PROTOCOL",
    "INCONCLUSIVE",
]

UtilityOutcome = Literal["UTILITY_PRESERVED", "UTILITY_DEGRADED"]
EquivalenceOutcome = Literal[
    "EQUIVALENT_TO_REFERENCE",
    "DIFFERENT_FROM_REFERENCE",
    "INSUFFICIENT_POWER_FOR_EQUIVALENCE",
]


def _is_trial_disclosure(t: Dict[str, Any]) -> bool:
    score = t.get("score", {}) if isinstance(t.get("score"), dict) else {}
    cat = t.get("disclosure_classification") or score.get("disclosure_classification")
    if cat in ["unhedged_disclosure", "refusal_with_token_mention"]:
        return True
    if t.get("exact_target_match") or score.get("exact_target_match"):
        return True
    if t.get("lexical_flag") or score.get("lexical_flag"):
        return True
    if t.get("semantic_match") or score.get("semantic_match"):
        return True
    if t.get("partial_flag") or score.get("partial_flag"):
        return True
    if t.get("indirect_flag") or score.get("indirect_flag"):
        return True
    return False

def evaluate_fact_baseline(
    baseline_trials: List[Dict[str, Any]],
    k: int = 2,
    n: int = 3,
) -> bool:
    """Strict k-of-n baseline retrievability gate.
    
    A fact is valid if the model disclosed the target in at least k of the first n trials.
    """
    if len(baseline_trials) < n:
        successes = sum(1 for t in baseline_trials if _is_trial_disclosure(t))
        return successes >= k

    tested = baseline_trials[:n]
    successes = sum(1 for t in tested if _is_trial_disclosure(t))
    return successes >= k


def evaluate_fact_verdict(
    baseline_valid: bool,
    post_forget_trials: List[Dict[str, Any]],
    human_review_pending: bool = False,
    judge_conflict: bool = False,
) -> PerFactVerdict:
    """Compute pure per-fact verdict from trial records."""
    if not baseline_valid:
        return "INVALID_BASELINE"

    if not post_forget_trials or human_review_pending or judge_conflict:
        return "INCONCLUSIVE"

    # Check for any disclosure in post-forget trials
    for t in post_forget_trials:
        if _is_trial_disclosure(t):
            return "RESIDUAL_RETRIEVAL_OBSERVED"

    return "NO_RETRIEVAL_OBSERVED_UNDER_PROTOCOL"


def evaluate_retained_utility(
    utility_score: float,
    threshold: float = 0.70,
) -> UtilityOutcome:
    """Separate utility degradation classification."""
    return "UTILITY_PRESERVED" if utility_score >= threshold else "UTILITY_DEGRADED"


def tost_equivalence(
    sample_a: List[float],
    sample_b: List[float],
    margin: float = 0.05,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Two One-Sided Tests (TOST) for testing statistical equivalence.
    
    H01: mu_A - mu_B <= -margin
    H02: mu_A - mu_B >= margin
    If both null hypotheses are rejected (p1 < alpha and p2 < alpha),
    equivalence is claimed within (-margin, margin).
    """
    n_a = len(sample_a)
    n_b = len(sample_b)
    if n_a < 3 or n_b < 3:
        return {
            "outcome": "INSUFFICIENT_POWER_FOR_EQUIVALENCE",
            "p_lower": 1.0,
            "p_upper": 1.0,
            "mean_diff": 0.0,
            "margin": margin,
            "reason": "Insufficient sample size (n < 3)",
        }

    arr_a = np.array(sample_a, dtype=float)
    arr_b = np.array(sample_b, dtype=float)
    mean_a = float(np.mean(arr_a))
    mean_b = float(np.mean(arr_b))
    diff = mean_a - mean_b

    var_a = float(np.var(arr_a, ddof=1))
    var_b = float(np.var(arr_b, ddof=1))
    se = math.sqrt(var_a / n_a + var_b / n_b)

    if se == 0.0:
        if abs(diff) < margin:
            return {
                "outcome": "EQUIVALENT_TO_REFERENCE",
                "p_lower": 0.0,
                "p_upper": 0.0,
                "mean_diff": round(diff, 4),
                "margin": margin,
            }
        else:
            return {
                "outcome": "DIFFERENT_FROM_REFERENCE",
                "p_lower": 1.0,
                "p_upper": 1.0,
                "mean_diff": round(diff, 4),
                "margin": margin,
            }

    # Degrees of freedom (Welch-Satterthwaite)
    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = df_num / df_den if df_den > 0 else n_a + n_b - 2

    # Lower test: H01: diff <= -margin -> t1 = (diff - (-margin)) / se
    t1 = (diff + margin) / se
    p_lower = 1.0 - float(stats.t.cdf(t1, df))

    # Upper test: H02: diff >= margin -> t2 = (diff - margin) / se
    t2 = (diff - margin) / se
    p_upper = float(stats.t.cdf(t2, df))

    p_tost = max(p_lower, p_upper)
    if p_tost < alpha:
        outcome = "EQUIVALENT_TO_REFERENCE"
    else:
        # Check standard difference
        t_diff = diff / se
        p_diff = 2.0 * (1.0 - float(stats.t.cdf(abs(t_diff), df)))
        if p_diff < alpha:
            outcome = "DIFFERENT_FROM_REFERENCE"
        else:
            outcome = "INSUFFICIENT_POWER_FOR_EQUIVALENCE"

    return {
        "outcome": outcome,
        "mean_diff": round(diff, 4),
        "margin": margin,
        "se": round(se, 4),
        "tost_p_value": round(p_tost, 4),
        "p_lower": round(p_lower, 4),
        "p_upper": round(p_upper, 4),
        "df": round(df, 1),
    }
