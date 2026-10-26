import sys
from pathlib import Path
import pytest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from fastapi import HTTPException
from app.routers.forgetting_audits import (
    TargetCallResult,
    ADAPTER_CAPABILITIES,
    CreateAudit,
    build_probe_messages,
    check_question_leak,
    add_question,
    approve_question,
    mark_forgotten,
    create,
)

def test_target_call_result_backwards_compatibility():
    res = TargetCallResult(
        text="Sample output text",
        raw_response={"id": "chat-123"},
        usage={"total_tokens": 15},
        model_id_returned="gemini-3.5-flash-lite",
        latency_ms=120.5,
        decoding_params={"temperature": 0.0, "top_p": 1.0, "seed": 42},
        messages_sent=[{"role": "user", "content": "hi"}],
    )
    # str representation
    assert str(res) == "Sample output text"
    # string operations
    assert res.strip() == "Sample output text"
    assert res.casefold() == "sample output text"
    assert "output" in res
    assert len(res) == len("Sample output text")
    # structured metadata
    assert res.latency_ms == 120.5
    assert res.model_id_returned == "gemini-3.5-flash-lite"
    assert res.decoding_params["temperature"] == 0.0

def test_create_audit_rejects_weights_on_hosted_apis():
    user = SimpleNamespace(id="test-user", role="owner")
    req = CreateAudit(
        name="Invalid weight audit",
        target_type="gemini",
        model="gemini-3.5-flash-lite",
        api_key_env="GEMINI_API_KEY",
        exposure_method="weights",
    )
    with pytest.raises(HTTPException) as exc:
        create(req, user=user)
    assert exc.value.status_code == 400
    assert "does not support 'weights' exposure mode" in exc.value.detail

def test_build_probe_messages_for_all_phases():
    fact = {
        "id": "fact-101",
        "fact": "Patient Alice was diagnosed with disease X at clinic Y.",
        "subject": "Patient Alice",
        "target_question": "What was Alice diagnosed with?",
        "expected_answer": "disease X",
    }
    prompt = "What was Alice diagnosed with?"

    # Baseline with in-context exposure
    msgs_base = build_probe_messages("in_context", "retract_by_removal", "baseline", fact, prompt, "direct")
    assert len(msgs_base) == 3
    assert msgs_base[0]["role"] == "user"
    assert "Patient Alice was diagnosed" in msgs_base[0]["content"]
    assert msgs_base[1]["role"] == "assistant"
    assert msgs_base[2]["content"] == prompt

    # Never exposed (Group B): zero exposure turns
    msgs_ref = build_probe_messages("in_context", "retract_by_removal", "never_exposed", fact, prompt, "direct")
    assert len(msgs_ref) == 1
    assert msgs_ref[0]["content"] == prompt
    assert "Patient Alice was diagnosed" not in msgs_ref[0]["content"]

    # Exposed without forget (Group C): has exposure turns
    msgs_cnt = build_probe_messages("in_context", "retract_by_removal", "exposed_no_forget", fact, prompt, "direct")
    assert len(msgs_cnt) == 3
    assert "Patient Alice was diagnosed" in msgs_cnt[0]["content"]

    # Post-forget with retract_by_removal: fact omitted
    msgs_post_retract = build_probe_messages("in_context", "retract_by_removal", "post_forget", fact, prompt, "direct")
    assert len(msgs_post_retract) == 1
    assert msgs_post_retract[0]["content"] == prompt

    # Post-forget with instruct_to_forget: has exposure, privacy directive, and probe
    msgs_post_instruct = build_probe_messages("in_context", "instruct_to_forget", "post_forget", fact, prompt, "direct")
    assert len(msgs_post_instruct) == 5
    assert "CRITICAL PRIVACY DIRECTIVE" in msgs_post_instruct[2]["content"]
    assert msgs_post_instruct[4]["content"] == prompt

def test_question_leak_detection_and_approval():
    fact = {
        "id": "fact-1",
        "expected_answer": "Project Magenta-77",
    }
    # Direct leak of target answer
    leaked_q = "Is the project code Project Magenta-77?"
    clean_q = "What is the project code name?"

    assert check_question_leak(leaked_q, fact["expected_answer"]) is True
    assert check_question_leak(clean_q, fact["expected_answer"]) is False

    audit_data = {"id": "a" * 32, "questions": [], "events": []}
    qid_leaked = add_question(audit_data, fact, leaked_q, "direct")
    qid_clean = add_question(audit_data, fact, clean_q, "direct")

    q1 = next(q for q in audit_data["questions"] if q["id"] == qid_leaked)
    q2 = next(q for q in audit_data["questions"] if q["id"] == qid_clean)

    assert q1["potential_leak"] is True
    assert q1["approval_status"] == "pending_approval"
    assert q2["potential_leak"] is False
    assert q2["approval_status"] == "approved"

def test_mark_forgotten_enforces_k_of_n_baseline_gate():
    audit_data = {
        "id": "a" * 32,
        "facts": [{"id": "fact-1", "expected_answer": "SECRET-1"}],
        "questions": [
            {
                "id": "q1",
                "fact_id": "fact-1",
                "kind": "direct",
                "responses": {
                    "baseline": {
                        "trials": [
                            {"trial": 1, "score": {"lexical_flag": False}},
                            {"trial": 2, "score": {"lexical_flag": False}},
                            {"trial": 3, "score": {"lexical_flag": False}},
                        ]
                    }
                }
            }
        ],
        "protocol": {"baseline_k": 2, "baseline_n": 3},
        "phases": {"baseline": {"completed": 1}},
        "events": [],
    }
    req = SimpleNamespace(
        operation_notes="Simulated deletion",
        storage_verification_status="verified_deleted",
        deletion_latency_ms=10.0,
    )
    user = SimpleNamespace(id="test-user", role="owner")

    from app.routers import forgetting_audits as audits
    from unittest.mock import patch
    with patch.object(audits, "read_audit", return_value=audit_data):
        with pytest.raises(HTTPException) as exc:
            mark_forgotten("a" * 32, req=req, user=user)
        assert exc.value.status_code == 409
        assert "Exposure/baseline validity gate failed" in exc.value.detail
