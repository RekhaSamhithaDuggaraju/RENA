"""Tests for ground-truth unlearning evaluation testbed."""
import pytest
from core.testbed.ground_truth_testbed import (
    GroundTruthCondition,
    SyntheticCanaryDataset,
    MockTargetModel,
    NaiveAuditor,
    ControlledAuditor,
    cluster_bootstrap_ci,
    run_experiment_e1_benchmarks,
    run_experiment_e2_ablations,
    run_experiment_e3_suppression_strategies,
    run_experiment_e4_detection_sensitivity,
)


def test_synthetic_canary_dataset_generation():
    gen = SyntheticCanaryDataset(seed=42)
    facts = gen.generate(20, unretrievable_rate=0.1)
    assert len(facts) == 20
    for f in facts:
        assert f.fact_id.startswith("fact_")
        assert len(f.expected_answer) > 0
        assert len(f.multi_turn_probe) == 3
        assert len(f.retained_question) > 0


def test_naive_auditor_false_pass_on_suppressed_model():
    gen = SyntheticCanaryDataset(seed=42)
    facts = gen.generate(20, unretrievable_rate=0.0)
    supp_model = MockTargetModel(GroundTruthCondition.SUPPRESSED_NOT_FORGOTTEN, seed=10)

    naive = NaiveAuditor()
    res = naive.audit(supp_model, facts)
    # Naive auditor issues 0 leaks because direct query receives a canned refusal!
    assert res["leaks"] == 0
    assert res["verdict"] == "FORGOTTEN"
    assert res["unlearning_confirmed"] is True  # This is the classic FALSE PASS!


def test_controlled_auditor_catches_suppressed_model():
    gen = SyntheticCanaryDataset(seed=42)
    facts = gen.generate(20, unretrievable_rate=0.0)
    supp_model = MockTargetModel(GroundTruthCondition.SUPPRESSED_NOT_FORGOTTEN, seed=10)

    controlled = ControlledAuditor()
    res = controlled.audit(supp_model, facts)
    # RENA catches the residual retrieval through paraphrase, indirect, and multi-turn probes!
    assert res["verdict_code"] == "RESIDUAL_RETRIEVAL_OBSERVED"
    assert res["unlearning_confirmed"] is False
    assert res["residual_leaks"] > 0
    assert res["leak_by_probe"]["multi_turn"] > 0


def test_controlled_auditor_passes_truly_forgotten_model():
    gen = SyntheticCanaryDataset(seed=42)
    facts = gen.generate(15, unretrievable_rate=0.0)
    true_model = MockTargetModel(GroundTruthCondition.TRULY_FORGOTTEN, seed=20)

    controlled = ControlledAuditor()
    res = controlled.audit(true_model, facts)
    assert res["verdict_code"] == "NO_RETRIEVAL_OBSERVED_UNDER_PROTOCOL"
    assert res["unlearning_confirmed"] is True
    assert res["residual_leaks"] == 0
    assert res["retained_utility_score"] == 1.0


def test_controlled_auditor_detects_utility_collapse():
    gen = SyntheticCanaryDataset(seed=42)
    facts = gen.generate(15, unretrievable_rate=0.0)
    over_model = MockTargetModel(GroundTruthCondition.OVER_UNLEARNED, seed=30)

    controlled = ControlledAuditor()
    res = controlled.audit(over_model, facts)
    assert res["verdict_code"] == "INCONCLUSIVE"
    assert res["unlearning_confirmed"] is False
    assert res["retained_utility_score"] < 0.70


def test_cluster_bootstrap_bounds():
    data = [0.1, 0.12, 0.08, 0.15, 0.09, 0.11]
    low, high = cluster_bootstrap_ci(data, n_resamples=200)
    assert 0.05 <= low <= 0.11
    assert 0.09 <= high <= 0.16
    assert low <= high


def test_experiment_e1_quick_run():
    res = run_experiment_e1_benchmarks(num_runs=5, facts_per_run=10)
    assert res["experiment"] == "E1_FPR_FFR_Comparison"
    assert res["naive_auditor"]["false_pass_rate"] > 0.80
    assert res["rena_controlled_auditor"]["false_pass_rate"] < 0.20
    assert res["fpr_reduction_delta"] > 0.60


def test_experiment_e2_quick_run():
    res = run_experiment_e2_ablations(num_runs=3, facts_per_run=10)
    assert "ablation_results" in res
    assert "full_rena" in res["ablation_results"]
    assert "direct_probe_only" in res["ablation_results"]


def test_experiment_e3_run():
    res = run_experiment_e3_suppression_strategies()
    assert "strategies_evaluated" in res
    assert len(res["strategies_evaluated"]) >= 5


def test_experiment_e4_quick_run():
    res = run_experiment_e4_detection_sensitivity(sample_sizes=[10, 20])

    assert res["experiment"] == "E4_Synthetic_Detection_Sensitivity"
    assert res["ground_truth_target"] == "PARTIALLY_FORGOTTEN_50"
    assert res["trials_per_sample_size"] == 30
    assert "synthetic" in res["interpretation"].lower()

    for size in ("10", "20"):
        row = res["curve"][size]
        assert row["sample_size"] == int(size)
        assert row["trials_attempted"] == 30
        assert row["valid_trials"] + row["excluded_trials"] == 30
        assert sum(row["verdict_counts"].values()) == 30
        assert row["detection_count"] <= row["valid_trials"]
        assert row["detection_frequency_denominator"] == "valid audit trials only"

        if row["valid_trials"]:
            assert 0.0 <= row["detection_frequency"] <= 1.0
            assert row["detection_frequency_ci_95"][0] <= row["detection_frequency_ci_95"][1]
        else:
            assert row["detection_frequency"] is None
            assert row["detection_frequency_ci_95"] is None
