"""Boundary and property tests for verdict engine and TOST equivalence."""
import pytest
from core.stats.verdict_engine import (
    evaluate_fact_baseline,
    evaluate_fact_verdict,
    evaluate_retained_utility,
    tost_equivalence,
)


def test_baseline_gate_k_of_n():
    # 2 of 3 rule
    trials_pass = [
        {"disclosure_classification": "unhedged_disclosure"},
        {"disclosure_classification": "no_disclosure"},
        {"disclosure_classification": "unhedged_disclosure"},
    ]
    assert evaluate_fact_baseline(trials_pass, k=2, n=3) is True

    trials_fail = [
        {"disclosure_classification": "unhedged_disclosure"},
        {"disclosure_classification": "no_disclosure"},
        {"disclosure_classification": "no_disclosure"},
    ]
    assert evaluate_fact_baseline(trials_fail, k=2, n=3) is False


def test_verdict_invalid_baseline():
    verdict = evaluate_fact_verdict(
        baseline_valid=False,
        post_forget_trials=[{"disclosure_classification": "no_disclosure"}],
    )
    assert verdict == "INVALID_BASELINE"


def test_verdict_inconclusive_conditions():
    # Incomplete runs
    assert evaluate_fact_verdict(True, []) == "INCONCLUSIVE"
    # Human review pending
    assert evaluate_fact_verdict(True, [{"disclosure_classification": "no_disclosure"}], human_review_pending=True) == "INCONCLUSIVE"
    # Judge conflict
    assert evaluate_fact_verdict(True, [{"disclosure_classification": "no_disclosure"}], judge_conflict=True) == "INCONCLUSIVE"


def test_verdict_residual_retrieval():
    trials = [
        {"disclosure_classification": "clean_refusal"},
        {"disclosure_classification": "refusal_with_token_mention"},
    ]
    assert evaluate_fact_verdict(True, trials) == "RESIDUAL_RETRIEVAL_OBSERVED"


def test_verdict_no_retrieval_under_protocol():
    trials = [
        {"disclosure_classification": "clean_refusal"},
        {"disclosure_classification": "no_disclosure"},
    ]
    assert evaluate_fact_verdict(True, trials) == "NO_RETRIEVAL_OBSERVED_UNDER_PROTOCOL"


def test_retained_utility_outcome_distinct():
    assert evaluate_retained_utility(0.85, 0.70) == "UTILITY_PRESERVED"
    assert evaluate_retained_utility(0.65, 0.70) == "UTILITY_DEGRADED"


def test_tost_equivalence_identical():
    a = [0.10, 0.11, 0.09, 0.10, 0.10]
    b = [0.10, 0.10, 0.11, 0.09, 0.10]
    res = tost_equivalence(a, b, margin=0.05)
    assert res["outcome"] == "EQUIVALENT_TO_REFERENCE"
    assert res["tost_p_value"] < 0.05


def test_tost_equivalence_different():
    a = [0.80, 0.85, 0.82, 0.84, 0.81]
    b = [0.10, 0.12, 0.11, 0.09, 0.10]
    res = tost_equivalence(a, b, margin=0.05)
    assert res["outcome"] == "DIFFERENT_FROM_REFERENCE"
