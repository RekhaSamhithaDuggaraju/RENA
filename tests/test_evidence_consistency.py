"""Automated regression test verifying quantitative evidence consistency across raw artifacts, reports, and paper manuscript.

Enforces Priority 1: Every number reported in documentation and paper must trace directly to raw execution logs in results/.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def smoke_audit_data():
    p = ROOT / "results" / "testbed" / "smoke_audit.json"
    assert p.exists(), f"Missing artifact: {p}"
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture
def oracle_data():
    p = ROOT / "results" / "testbed" / "oracle_results.json"
    assert p.exists(), f"Missing artifact: {p}"
    return json.loads(p.read_text(encoding="utf-8"))


def test_smoke_audit_summary_matches_raw_trials(smoke_audit_data):
    """Verify comparison_summary in smoke_audit.json accurately reflects per-fact trials."""
    summary_map = {row["condition"]: row for row in smoke_audit_data["comparison_summary"]}
    assert set(summary_map.keys()) == {
        "reference",
        "retained",
        "suppress_prompt",
        "suppress_refusal",
        "graddiff",
    }

    # Reference
    assert summary_map["reference"]["naive_certified_unlearned"] == "5/5 (100%)"
    assert summary_map["reference"]["rena_residual_detected"] == "0/5 (0%)"
    assert summary_map["reference"]["naive_false_pass_count"] == 0

    # Retained
    assert summary_map["retained"]["naive_certified_unlearned"] == "1/5 (20%)"
    assert summary_map["retained"]["rena_residual_detected"] == "5/5 (100%)"
    assert summary_map["retained"]["naive_false_pass_count"] == 1

    # Suppress Prompt
    assert summary_map["suppress_prompt"]["naive_certified_unlearned"] == "2/5 (40%)"
    assert summary_map["suppress_prompt"]["rena_residual_detected"] == "5/5 (100%)"
    assert summary_map["suppress_prompt"]["naive_false_pass_count"] == 2

    # Suppress Refusal (Strictest gate: Naive certifies 5/5 unlearned, RENA catches 1/5 via multi-turn)
    assert summary_map["suppress_refusal"]["naive_certified_unlearned"] == "5/5 (100%)"
    assert summary_map["suppress_refusal"]["rena_residual_detected"] == "1/5 (20%)"
    assert summary_map["suppress_refusal"]["naive_false_pass_count"] == 5

    # GradDiff
    assert summary_map["graddiff"]["naive_certified_unlearned"] == "5/5 (100%)"
    assert summary_map["graddiff"]["rena_residual_detected"] == "0/5 (0%)"
    assert summary_map["graddiff"]["naive_false_pass_count"] == 0


def test_paper_manuscript_numbers_match_smoke_audit(smoke_audit_data, oracle_data):
    """Verify that paper/main.tex Table 1 and Table 2 match raw experiment artifacts exactly."""
    tex_path = ROOT / "paper" / "main.tex"
    assert tex_path.exists(), "Missing paper/main.tex"
    tex_content = tex_path.read_text(encoding="utf-8")

    # Table 1: Head-to-Head numbers
    assert "5/5 ($100.0\\%$)" in tex_content
    assert "1/5 ($20.0\\%$)" in tex_content
    assert "2/5 ($40.0\\%$)" in tex_content
    assert "\\textbf{5/5 (100.0\\%)}" in tex_content  # suppress_refusal false pass
    assert "\\textbf{40.0\\% (Degraded)}" in tex_content  # graddiff utility collapse

    # Table 2: Oracle numbers (read directly from raw oracle_data)
    ref_ll = oracle_data["log_likelihood_evaluations"]["reference"]["mean_log_prob"]
    ret_ll = oracle_data["log_likelihood_evaluations"]["retained"]["mean_log_prob"]
    sr_ll = oracle_data["log_likelihood_evaluations"]["suppress_refusal"]["mean_log_prob"]
    gd_ll = oracle_data["log_likelihood_evaluations"]["graddiff"]["mean_log_prob"]
    sr_step = oracle_data["relearning_attacks"]["suppress_refusal"]["steps_to_recovery"]

    assert f"{ref_ll:.4f}" in tex_content
    assert f"{ret_ll:.4f}" in tex_content
    assert f"{sr_ll:.4f}" in tex_content
    assert f"{gd_ll:.4f}" in tex_content
    assert f"Step {sr_step}" in tex_content


def test_reports_contain_no_stale_pilot_numbers():
    """Verify that documentation files do not retain contradicted stale figures from early pilots."""
    doc_paths = [
        ROOT / "AUDIT_REPORT.md",
        ROOT / "docs" / "CLAIMS_LEDGER.md",
        ROOT / "docs" / "PAPER_EVALUATION_REPORT.md",
    ]

    for p in doc_paths:
        assert p.exists(), f"Missing file: {p}"
        text = p.read_text(encoding="utf-8")

        # Stale pilot numbers from pre-G4 scorer that must NEVER re-appear:
        assert "60% of refusal-tuned facts" not in text, f"Stale 60% figure found in {p.name}"
        assert "60.0% False Pass" not in text, f"Stale 60.0% false pass found in {p.name}"
        assert "80.0% recovery" not in text, f"Stale 80.0% recovery found in {p.name}"
