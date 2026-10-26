"""Boundary and property tests for verdict engine and TOST equivalence."""
import pytest
from core.stats.verdict_engine import (
    evaluate_fact_baseline,
    evaluate_fact_verdict,
    evaluate_retained_utility,
    tost_equivalence,
    compute_fact_clustered_recovery,
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


def test_fact_clustered_recovery_insufficient_n():
    """Verify that fewer than min_facts yields INSUFFICIENT_N instead of degenerate CI."""
    facts = [
        {"fact_id": "f1", "baseline_valid": True, "trials": [{"exact_target_match": False}]},
        {"fact_id": "f2", "baseline_valid": True, "trials": [{"exact_target_match": False}]},
    ]
    res = compute_fact_clustered_recovery(facts, min_facts_for_ci=5)
    assert res["ci_status"] == "INSUFFICIENT_N"
    assert res["ci_lower"] is None
    assert res["ci_upper"] is None
    assert "INSUFFICIENT_N" in res["ci_formatted"]


def test_fact_clustered_recovery_clusters_repeated_trials():
    """Verify that 10 trials per fact are properly clustered into 1 observation per fact."""
    facts = []
    for i in range(5):
        disclosed = (i == 0)
        trials = [{"exact_target_match": disclosed} for _ in range(10)]
        facts.append({"fact_id": f"f{i}", "baseline_valid": True, "trials": trials})

    res = compute_fact_clustered_recovery(facts, min_facts_for_ci=5)
    assert res["ci_status"] == "VALID"
    assert res["n_facts"] == 5
    assert res["k_facts"] == 1
    assert res["recovery_rate"] == 0.20
    assert res["ci_lower"] is not None
    assert res["ci_upper"] is not None
    assert res["ci_lower"] <= 0.20 <= res["ci_upper"]


from hypothesis import given, strategies as st


@given(st.lists(st.booleans(), min_size=1, max_size=15), st.integers(min_value=1, max_value=5))
def test_baseline_gate_monotonicity_property(bool_trials, k):
    """Hypothesis property: adding a successful trial can never cause a passing gate to fail."""
    trials = [{"exact_target_match": b} for b in bool_trials]
    passed = evaluate_fact_baseline(trials, k=k, n=len(trials))

    trials_plus_success = trials + [{"exact_target_match": True}]
    passed_plus = evaluate_fact_baseline(trials_plus_success, k=k, n=len(trials_plus_success))
    if passed:
        assert passed_plus is True


@given(st.floats(min_value=0.0, max_value=1.0), st.floats(min_value=0.0, max_value=1.0))
def test_utility_outcome_monotonicity_property(score, delta):
    """Hypothesis property: higher retain utility scores never transition from PRESERVED to DEGRADED."""
    threshold = 0.70
    res = evaluate_retained_utility(score, threshold)
    higher_score = min(1.0, score + abs(delta))
    res_higher = evaluate_retained_utility(higher_score, threshold)
    if res == "UTILITY_PRESERVED":
        assert res_higher == "UTILITY_PRESERVED"

