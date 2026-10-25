import os
from types import SimpleNamespace

from backend.app.verdicts import authoritative_verdict, authoritative_score


def test_rena_x_is_authoritative_over_classical_consensus():
    a = SimpleNamespace(authoritative_verdict='FAIL', rena_x_verdict='FAIL', consensus_verdict='Likely Forgotten', rena_x_score=0.02, diagnosis={})
    assert authoritative_verdict(a) == 'FAIL'
    assert authoritative_score(a) == 0.02


def test_helper_falls_back_to_persisted_rena_x_then_classical():
    a = SimpleNamespace(authoritative_verdict=None, rena_x_verdict='PASS', consensus_verdict='Not Forgotten', rena_x_score=0.91, diagnosis={})
    assert authoritative_verdict(a) == 'PASS'
    assert authoritative_score(a) == 0.91


def test_helper_can_recover_rena_x_from_diagnosis():
    a = SimpleNamespace(authoritative_verdict=None, rena_x_verdict=None, consensus_verdict='Likely Forgotten', rena_x_score=None,
                        diagnosis={'advanced_audit': {'verdict':'REVIEW','certification_score':0.08}})
    assert authoritative_verdict(a) == 'INCONCLUSIVE'
    assert authoritative_score(a) == 0.08
def test_audit_persistence_prefers_certification_score():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    from app.models.audit import Audit

    result = {
        "algorithm": "ground_truth_retraining",
        "runtime_seconds": 1.0,
        "accuracy": {
            "retain": 0.95,
            "test": 0.90,
            "forget": 0.50,
            "original_model_test": 0.92,
        },
        "consensus": {
            "verdict": "Likely Forgotten",
            "consensus_score": 0.8,
            "agreement": 0.7,
        },
        "module_results": [],
        "diagnosis": {
            "advanced_audit": {
                "verdict": "REVIEW",
                "worst_case_score": 0.02,
                "certification_score": 0.08,
            }
        },
    }

    audit = Audit()
    audit.from_pipeline_result(result, {})

    assert audit.rena_x_score == 0.08
    assert audit.diagnosis["advanced_audit"]["worst_case_score"] == 0.02


def test_confidence_is_independent_of_rena_x_score():
    from types import SimpleNamespace
    from backend.app.verdicts import confidence_and_quality

    common = {
        "authoritative_verdict": "PASS",
        "consensus_verdict": "Likely Forgotten",
        "diagnosis": {
            "advanced_audit": {
                "axes": {
                    "behavioral": {"p_value": 0.10},
                    "membership": {"p_value": 0.20},
                    "retain": {"p_value": 0.30},
                }
            }
        },
        "dataset_summary": {
            "reference_retrains": 10,
            "seeds": [1, 2, 3],
        },
    }

    low_score = SimpleNamespace(rena_x_score=0.02, **common)
    high_score = SimpleNamespace(rena_x_score=0.98, **common)

    low_report = confidence_and_quality(low_score)
    high_report = confidence_and_quality(high_score)

    assert low_report["confidence"] == high_report["confidence"]
    assert low_report["confidence_label"] == high_report["confidence_label"]
    assert low_report["evidence_quality"] == high_report["evidence_quality"]


def test_confidence_penalizes_verifier_disagreement():
    from types import SimpleNamespace
    from backend.app.verdicts import confidence_and_quality

    audit = SimpleNamespace(
        rena_x_score=0.95,
        authoritative_verdict="INCONCLUSIVE",
        consensus_verdict="Likely Forgotten",
        diagnosis={
            "advanced_audit": {
                "axes": {
                    "behavioral": {"p_value": 0.10},
                    "membership": {"p_value": 0.20},
                    "retain": {"p_value": 0.30},
                }
            }
        },
        dataset_summary={
            "reference_retrains": 10,
            "seeds": [1, 2, 3],
        },
    )

    report = confidence_and_quality(audit)

    assert report["factors"][-1]["name"] == "Verifier agreement"
    assert report["factors"][-1]["status"] == "WEAK"
    assert report["confidence"] < 0.85
