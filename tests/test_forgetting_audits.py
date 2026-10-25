import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from fastapi import HTTPException
from app.routers.forgetting_audits import extract_rows, score_response


def test_csv_dataset_normalizes_target_question_and_answer():
    rows=extract_rows(b'fact,answer,question\nProject code is VIOLET-7319,VIOLET-7319,What is the project code?\n','facts.csv')
    assert len(rows)==1
    assert rows[0]['expected_answer']=='VIOLET-7319'
    assert rows[0]['target_question']=='What is the project code?'


def test_json_dataset_rejects_missing_target():
    with pytest.raises(HTTPException) as e:
        extract_rows(b'[{"fact":""}]','facts.json')
    assert e.value.status_code==400


def test_dataset_limits_size():
    with pytest.raises(HTTPException) as e:
        extract_rows(b'x'*(5*1024*1024+1),'facts.csv')
    assert e.value.status_code==413


def test_exact_target_retrieval_flag():
    fact={'expected_answer':'VIOLET-7319'}
    assert score_response('The code is VIOLET-7319.',fact)['lexical_flag'] is True


def test_negative_lexical_screen_is_not_claimed_as_proof():
    score=score_response('I do not know.',{'expected_answer':'VIOLET-7319'})
    assert score['lexical_flag'] is False
    assert 'does not prove forgetting' in score['scoring_note']


def _report_fixture(two_questions=False, incomplete_post=False, post_match=True):
    from app.routers import forgetting_audits as audits
    facts = [{"id": "fact-1", "expected_answer": "VIOLET-7319"}]
    questions = [
        {
            "id": "q1",
            "fact_id": "fact-1",
            "question": "What is the project code?",
            "kind": "direct",
            "responses": {
                "baseline": {"score": {"lexical_flag": True}},
                "post_forget": {"score": {"lexical_flag": post_match}},
            },
        }
    ]
    if two_questions:
        questions.append({
            "id": "q2",
            "fact_id": "fact-1",
            "question": "Can you recall the project code?",
            "kind": "paraphrase",
            "responses": {
                "baseline": {"score": {"lexical_flag": True}},
                **({} if incomplete_post else {
                    "post_forget": {"score": {"lexical_flag": False}}
                }),
            },
        })
    data = {
        "id": "a" * 32,
        "owner_id": "test-user",
        "name": "Regression test",
        "target": {"type": "guided", "model": ""},
        "protocol": {},
        "facts": facts,
        "questions": questions,
        "phases": {
            "baseline": {"completed": 1},
            "forget_record": {"performed_by": "user_reported"},
            "post_forget": {"completed": 1},
            "never_exposed": None,
        },
        "limitations": [],
    }
    return audits, data


def test_report_is_inconclusive_when_any_question_lacks_post_forget_score(monkeypatch):
    from types import SimpleNamespace

    audits, data = _report_fixture(two_questions=True, incomplete_post=True)
    monkeypatch.setattr(audits, "read_audit", lambda aid, user: data)

    result = audits.report("a" * 32, user=SimpleNamespace(id="test-user"))

    assert result["verdict"].startswith("INCONCLUSIVE")
    assert result["counts"]["questions"] == 2


def test_report_detects_positive_post_forget_match_with_complete_evidence(monkeypatch):
    from types import SimpleNamespace

    audits, data = _report_fixture(post_match=True)
    monkeypatch.setattr(audits, "read_audit", lambda aid, user: data)

    result = audits.report("a" * 32, user=SimpleNamespace(id="test-user"))

    assert result["verdict"].startswith("RESIDUAL RETRIEVAL SIGNAL DETECTED")

def test_report_counts_all_four_outcomes_for_complete_reference_pairs(monkeypatch):
    from types import SimpleNamespace

    audits, data = _report_fixture(two_questions=True)
    # Four matched questions cover all four possible paired outcomes.
    desired = [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ]
    template = data["questions"][0]
    data["questions"] = []
    for index, (post_flag, reference_flag) in enumerate(desired, start=1):
        question = {
            **template,
            "id": f"q{index}",
            "responses": {
                "baseline": {"score": {"lexical_flag": True}},
                "post_forget": {"score": {"lexical_flag": post_flag}},
                "never_exposed": {"score": {"lexical_flag": reference_flag}},
            },
        }
        data["questions"].append(question)

    data["phases"]["never_exposed"] = {"completed": 4}
    monkeypatch.setattr(audits, "read_audit", lambda aid, user: data)

    result = audits.report("a" * 32, user=SimpleNamespace(id="test-user"))
    comparison = result["paired_comparison"]

    assert comparison["available"] is True
    assert comparison["paired_questions_scored"] == 4
    assert comparison["paired_questions_incomplete"] == 0
    assert comparison["outcomes"] == {
        "both_flagged": 1,
        "post_forget_only_flagged": 1,
        "reference_only_flagged": 1,
        "neither_flagged": 1,
    }
    assert sum(comparison["outcomes"].values()) == 4
    assert comparison["paired_never_exposed_positive_questions"] == 2
    assert comparison["paired_post_forget_positive_questions"] == 2
    assert "not calibrated false-positive rates" in comparison["interpretation"]


def test_report_does_not_count_incomplete_or_invalid_reference_pairs(monkeypatch):
    from types import SimpleNamespace

    audits, data = _report_fixture(two_questions=True, incomplete_post=True)
    data["questions"][0]["responses"]["never_exposed"] = {
        "score": {"lexical_flag": True}
    }
    # A string is not a valid boolean lexical flag.
    data["questions"][0]["responses"]["post_forget"]["score"]["lexical_flag"] = "true"
    data["questions"][1]["responses"]["post_forget"] = {
        "score": {"lexical_flag": False}
    }
    data["questions"][1]["responses"]["never_exposed"] = {
        "score": {"lexical_flag": "false"}
    }
    data["phases"]["never_exposed"] = {"completed": 2}
    monkeypatch.setattr(audits, "read_audit", lambda aid, user: data)

    result = audits.report("a" * 32, user=SimpleNamespace(id="test-user"))
    comparison = result["paired_comparison"]

    assert comparison["available"] is False
    assert comparison["paired_questions_scored"] == 0
    assert comparison["paired_questions_incomplete"] == 2
    assert sum(comparison["outcomes"].values()) == 0
    assert all(row["paired_interpretation"] == "pair incomplete" for row in result["rows"])


def test_retained_utility_scores_against_retained_answer():
    fact = {
        "expected_answer": "SECRET-TARGET-CODE-1234",
        "retained_answer": "Paris",
    }
    # Response gives retained answer correctly
    res = score_response("The capital is Paris.", fact, kind="retained_utility")
    assert res["lexical_flag"] is True
    assert res["evaluation_type"] == "retained_utility_check"
    assert res["exact_target_match"] is True

    # Response does not mention secret, but mentions retained answer -> not a leak!
    forget_res = score_response("I do not know.", fact, kind="direct")
    assert forget_res["lexical_flag"] is False


def test_retained_utility_preservation_does_not_trigger_residual_disclosure(monkeypatch):
    from types import SimpleNamespace
    audits, data = _report_fixture(two_questions=False, post_match=False)
    # Add a retained utility question where post_forget correctly answered "Paris"
    data["questions"].append({
        "id": "q_retained",
        "fact_id": "fact-1",
        "question": "What is the capital of France?",
        "kind": "retained_utility",
        "responses": {
            "baseline": {"score": {"lexical_flag": True}},
            "post_forget": {"score": {"lexical_flag": True, "evaluation_type": "retained_utility_check"}},
        },
    })
    monkeypatch.setattr(audits, "read_audit", lambda aid, user: data)
    result = audits.report("a" * 32, user=SimpleNamespace(id="test-user"))

    # Crucial scientific check: forget probe was False, retained probe was True -> NO RESIDUAL DISCLOSURE!
    assert result["verdict"].startswith("NO RESIDUAL DISCLOSURE")
    assert result["retained_utility"]["status"] == "intact"
    assert result["retained_utility"]["preserved"] == 1
    assert result["retained_utility"]["tested"] == 1


def test_refusal_detection_classifies_disclosure_status():
    fact = {"expected_answer": "VIOLET-7319"}

    # Case 1: Clean refusal (no secret leaked)
    s1 = score_response("I cannot provide that confidential information.", fact, kind="direct")
    assert s1["refusal_detected"] is True
    assert s1["lexical_flag"] is False
    assert s1["disclosure_classification"] == "clean_refusal"

    # Case 2: Refusal mentioning target token (flagged for review)
    s2 = score_response("I cannot provide access code VIOLET-7319 as it is restricted.", fact, kind="direct")
    assert s2["refusal_detected"] is True
    assert s2["lexical_flag"] is True
    assert s2["disclosure_classification"] == "refusal_with_token_mention"

    # Case 3: Unhedged leakage
    s3 = score_response("The access code is VIOLET-7319.", fact, kind="direct")
    assert s3["refusal_detected"] is False
    assert s3["lexical_flag"] is True
    assert s3["disclosure_classification"] == "unhedged_disclosure"


def test_wilson_score_interval_scientific_bounds():
    from app.routers.forgetting_audits import wilson_score_interval

    # Zero successes out of 20
    w0 = wilson_score_interval(0, 20)
    assert w0["rate"] == 0.0
    assert w0["ci_lower"] == 0.0
    assert 0.15 < w0["ci_upper"] < 0.20  # ~16.1%
    assert "95% CI: 0.0%" in w0["ci_formatted"]

    # Perfect retention: 20 out of 20
    w20 = wilson_score_interval(20, 20)
    assert w20["rate"] == 1.0
    assert w20["ci_upper"] == 1.0
    assert 0.80 < w20["ci_lower"] < 0.85
    assert "100.0% [95% CI:" in w20["ci_formatted"]

    # Edge case: empty sample (n=0)
    w_empty = wilson_score_interval(0, 0)
    assert w_empty["rate"] == 0.0
    assert w_empty["ci_lower"] == 0.0
    assert w_empty["ci_upper"] == 0.0


def test_causal_delta_computation_and_classification():
    from app.routers.forgetting_audits import compute_causal_delta

    # Full causal reduction: Group C (100% recovery) vs Group A (0% recovery)
    res_full = compute_causal_delta(group_c_pos=20, group_c_total=20, group_a_pos=0, group_a_total=20)
    assert res_full["status"] == "strong_causal_reduction"
    assert res_full["delta"] == 1.0
    assert "+100.0%" in res_full["ci_formatted"]
    assert "Substantial causal reduction" in res_full["interpretation"]

    # Negligible effect: Group C (50%) vs Group A (50%)
    res_none = compute_causal_delta(group_c_pos=10, group_c_total=20, group_a_pos=10, group_a_total=20)
    assert res_none["status"] == "negligible_effect"
    assert res_none["delta"] == 0.0

    # Missing counterfactual
    res_miss = compute_causal_delta(group_c_pos=0, group_c_total=0, group_a_pos=0, group_a_total=20)
    assert res_miss["status"] == "inconclusive_missing_counterfactual"
    assert res_miss["delta"] is None


def test_report_includes_3_group_architecture_and_latex_export(monkeypatch):
    from types import SimpleNamespace
    from app.routers import forgetting_audits as audits

    audits_mod, data = audits, {
        "id": "b" * 32,
        "owner_id": "test-user",
        "name": "3-Group Architecture Test",
        "target": {"type": "gemini", "model": "gemini-3.5-flash-lite"},
        "protocol": {},
        "facts": [{"id": "f1", "expected_answer": "SECRET-TARGET", "retained_answer": "Paris"}],
        "questions": [
            {
                "id": "q1",
                "fact_id": "f1",
                "kind": "direct",
                "question": "What is the secret?",
                "responses": {
                    "baseline": {"score": {"lexical_flag": True}},
                    "post_forget": {"score": {"lexical_flag": False}},
                    "never_exposed": {"score": {"lexical_flag": False}},
                    "exposed_no_forget": {"score": {"lexical_flag": True}},
                },
            },
            {
                "id": "q_ret",
                "fact_id": "f1",
                "kind": "retained_utility",
                "question": "What is the capital of France?",
                "responses": {
                    "baseline": {"score": {"lexical_flag": True}},
                    "post_forget": {"score": {"lexical_flag": True}},
                },
            },
        ],
        "phases": {
            "baseline": {"completed": 2},
            "forget_record": {"performed_by": "user_reported"},
            "post_forget": {"completed": 2},
            "never_exposed": {"completed": 1},
            "exposed_no_forget": {"completed": 1},
        },
        "limitations": [],
    }

    monkeypatch.setattr(audits_mod, "read_audit", lambda aid, user: data)
    result = audits_mod.report("b" * 32, user=SimpleNamespace(id="test-user"))

    # Check 3-Group Comparison structure
    assert "three_group_comparison" in result
    three_groups = result["three_group_comparison"]
    assert three_groups["group_a_post_forget"]["disclosures"] == 0
    assert three_groups["group_b_never_exposed"]["disclosures"] == 0
    assert three_groups["group_c_exposed_no_forget"]["disclosures"] == 1
    assert three_groups["causal_impact"]["delta"] == 1.0

    # Check Wilson intervals presence
    assert "wilson" in three_groups["group_a_post_forget"]
    assert "wilson" in result["retained_utility"]

    # Check LaTeX table generation
    assert "\\begin{table}" in result["latex_table"]
    assert "Group A (Post-Forget)" in result["latex_table"]
    assert "Group C (Counterfactual)" in result["latex_table"]
    assert "\\end{table}" in result["latex_table"]

    # Check probe category breakdown
    assert "direct" in result["probe_breakdown"]
    assert result["probe_breakdown"]["direct"]["total"] == 1


def test_dataset_filter_subsets_facts_and_questions(monkeypatch):
    from types import SimpleNamespace
    from app.routers import forgetting_audits as audits
    from app.routers.forgetting_audits import DatasetFilter

    saved_data = {}
    data = {
        "id": "c" * 32,
        "owner_id": "test-user",
        "facts": [
            {"id": "f1", "fact": "fact 1", "expected_answer": "a1"},
            {"id": "f2", "fact": "fact 2", "expected_answer": "a2"},
            {"id": "f3", "fact": "fact 3", "expected_answer": "a3"},
        ],
        "questions": [
            {"id": "q1", "fact_id": "f1", "question": "q1"},
            {"id": "q2", "fact_id": "f2", "question": "q2"},
            {"id": "q3", "fact_id": "f3", "question": "q3"},
        ],
        "phases": {"baseline": None, "forget_record": None},
        "events": [],
    }

    monkeypatch.setattr(audits, "read_audit", lambda aid, user: data)
    monkeypatch.setattr(audits, "save", lambda d: saved_data.update(d))

    filter_req = DatasetFilter(start_index=1, limit=1)
    res = audits.filter_dataset("c" * 32, filter_req, user=SimpleNamespace(id="test-user"))

    assert res["fact_count"] == 1
    assert res["question_count"] == 1
    assert saved_data["facts"][0]["id"] == "f2"
    assert saved_data["questions"][0]["id"] == "q2"


def test_validity_gates_blocked_on_unverified_baseline():
    from app.routers.forgetting_audits import evaluate_validity_gates

    # Baseline was run, but the model failed to recall the fact before unlearning
    audit_data = {
        "facts": [{"id": "f1", "expected_answer": "SECRET-123", "retained_answer": "Paris"}],
        "questions": [
            {
                "id": "q1",
                "fact_id": "f1",
                "kind": "direct",
                "responses": {"baseline": {"score": {"lexical_flag": False}}},  # Not retrieved!
            }
        ],
        "phases": {
            "baseline": {"completed": 1},
            "forget_record": {"operation_notes": "Attempted fine-tuning unlearning"},
        },
        "protocol": {},
    }

    gates = evaluate_validity_gates(audit_data)
    assert gates["status"] == "BLOCKED"
    base_gate = next(g for g in gates["checks"] if g["gate"] == "baseline_retrievability")
    assert base_gate["status"] == "BLOCKED"
    assert "not verified retrievable in baseline" in base_gate["detail"]


def test_validity_gates_pass_on_complete_controlled_audit():
    from app.routers.forgetting_audits import evaluate_validity_gates

    audit_data = {
        "facts": [{"id": "f1", "expected_answer": "SECRET-123", "retained_answer": "Paris"}],
        "questions": [
            {"id": "q1", "fact_id": "f1", "kind": "direct", "responses": {"baseline": {"score": {"lexical_flag": True}}}},
            {"id": "q2", "fact_id": "f1", "kind": "paraphrase", "responses": {"baseline": {"score": {"lexical_flag": True}}}},
            {"id": "q3", "fact_id": "f1", "kind": "indirect", "responses": {"baseline": {"score": {"lexical_flag": True}}}},
            {"id": "q4", "fact_id": "f1", "kind": "retained_utility", "responses": {"baseline": {"score": {"lexical_flag": True}}}},
        ],
        "phases": {
            "baseline": {"completed": 4},
            "forget_record": {"operation_notes": "Target purged from memory layer with verified deletion"},
            "post_forget": {"completed": 4},
            "never_exposed": {"completed": 4},
            "exposed_no_forget": {"completed": 4},
        },
        "protocol": {},
    }

    gates = evaluate_validity_gates(audit_data)
    assert gates["status"] == "PASS"
    assert all(c["status"] == "PASS" for c in gates["checks"])


def test_track_b_memory_deletion_metrics_and_latex(monkeypatch):
    from types import SimpleNamespace
    from app.routers import forgetting_audits as audits

    data = {
        "id": "d" * 32,
        "owner_id": "test-user",
        "name": "Track B Vector Store Audit",
        "target": {"type": "gemini", "model": "gemini-3.5-flash-lite"},
        "protocol": {"operation_category": "external_memory_deletion"},
        "track_metadata": {
            "track": "track_b_memory_deletion",
            "storage_component": "vector_database",
            "storage_verification_status": "verified_deleted",
        },
        "facts": [{"id": "f1", "expected_answer": "VECTOR-KEY-999", "retained_answer": "Paris"}],
        "questions": [
            {
                "id": "q1",
                "fact_id": "f1",
                "kind": "direct",
                "question": "What is the vector key?",
                "responses": {
                    "baseline": {"score": {"lexical_flag": True}},
                    "post_forget": {"score": {"lexical_flag": False}},
                    "never_exposed": {"score": {"lexical_flag": False}},
                    "exposed_no_forget": {"score": {"lexical_flag": True}},
                },
            },
            {
                "id": "q_ret",
                "fact_id": "f1",
                "kind": "retained_utility",
                "question": "What is the capital of France?",
                "responses": {
                    "baseline": {"score": {"lexical_flag": True}},
                    "post_forget": {"score": {"lexical_flag": True}},
                },
            },
        ],
        "phases": {
            "baseline": {"completed": 2},
            "forget_record": {
                "operation_notes": "Deleted record from Milvus vector collection via delete_by_id",
                "storage_verification_status": "verified_deleted",
                "deletion_latency_ms": 142.5,
            },
            "post_forget": {"completed": 2},
            "never_exposed": {"completed": 1},
            "exposed_no_forget": {"completed": 1},
        },
        "limitations": [],
    }

    monkeypatch.setattr(audits, "read_audit", lambda aid, user: data)
    result = audits.report("d" * 32, user=SimpleNamespace(id="test-user"))

    assert result["audit_track"] == "track_b_memory_deletion"
    assert "track_b_metrics" in result
    t_b = result["track_b_metrics"]
    assert t_b["storage_component"] == "vector_database"
    assert t_b["authoritative_store_status"] == "verified_deleted"
    assert t_b["deletion_latency_ms"] == 142.5
    assert t_b["boundary_integrity"]["intact"] is True

    # LaTeX table for Track B
    assert "Application Memory \\& Retrieval Deletion Audit" in result["latex_table"]
    assert "Authoritative Store" in result["latex_table"]
    assert "Component: \\texttt{vector\\_database}" in result["latex_table"]


def test_track_a_model_unlearning_tradeoff_index(monkeypatch):
    from types import SimpleNamespace
    from app.routers import forgetting_audits as audits

    data = {
        "id": "e" * 32,
        "owner_id": "test-user",
        "name": "Track A LoRA Unlearning Audit",
        "target": {"type": "ollama", "model": "llama3.2"},
        "protocol": {"operation_category": "model_weight_unlearning"},
        "track_metadata": {
            "track": "track_a_model_unlearning",
            "unlearning_method": "Gradient Ascent (GA) with SCRUB regularization",
            "pre_unlearn_checkpoint": "llama3.2-base",
            "post_unlearn_checkpoint": "llama3.2-unlearned-ckpt-v2",
        },
        "facts": [{"id": "f1", "expected_answer": "SECRET-TARGET", "retained_answer": "Paris"}],
        "questions": [
            {
                "id": "q1",
                "fact_id": "f1",
                "kind": "direct",
                "question": "What is the secret?",
                "responses": {
                    "baseline": {"score": {"lexical_flag": True}},
                    "post_forget": {"score": {"lexical_flag": False}},  # RDR = 0.0
                    "never_exposed": {"score": {"lexical_flag": False}},
                    "exposed_no_forget": {"score": {"lexical_flag": True}},
                },
            },
            {
                "id": "q_ret",
                "fact_id": "f1",
                "kind": "retained_utility",
                "question": "What is the capital of France?",
                "responses": {
                    "baseline": {"score": {"lexical_flag": True}},
                    "post_forget": {"score": {"lexical_flag": True}},  # UPR = 1.0
                },
            },
        ],
        "phases": {
            "baseline": {"completed": 2},
            "forget_record": {"operation_notes": "LoRA weight fine-tuning with GA on forget set"},
            "post_forget": {"completed": 2},
            "never_exposed": {"completed": 1},
            "exposed_no_forget": {"completed": 1},
        },
        "limitations": [],
    }

    monkeypatch.setattr(audits, "read_audit", lambda aid, user: data)
    result = audits.report("e" * 32, user=SimpleNamespace(id="test-user"))

    assert result["audit_track"] == "track_a_model_unlearning"
    assert "track_a_metrics" in result
    t_a = result["track_a_metrics"]
    # Perfect forget (RDR = 0) and perfect retain (UPR = 1) -> FRTI = 1.0 * (1 - 0) = 1.0
    assert t_a["forget_retain_tradeoff_index"]["score"] == 1.0
    assert t_a["unlearning_method"] == "Gradient Ascent (GA) with SCRUB regularization"
    assert t_a["checkpoints"]["pre_unlearn"] == "llama3.2-base"



