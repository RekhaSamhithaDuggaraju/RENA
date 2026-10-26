"""Tests for ensuring all files in results/ represent real empirical runs and not unverified mocks."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

def _find_result_json_files():
    if not RESULTS_DIR.exists():
        return []
    files = []
    for p in RESULTS_DIR.rglob("*.json"):
        if p.name == "MANIFEST.json":
            continue
        files.append(p)
    return files

@pytest.mark.parametrize("file_path", _find_result_json_files())
def test_result_file_integrity_and_non_mock(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Mock must never be True
    assert data.get("mock") is not True, f"{file_path} has mock=True"

    # 2. Must contain evidence of real model execution or non-mock status
    has_explicit_non_mock = data.get("mock") is False
    has_model_spec = any(k in data for k in ["model", "model_id", "target_model", "provider", "target"])
    has_conditions_with_model = False
    if "conditions" in data and isinstance(data["conditions"], dict):
        has_conditions_with_model = any("model" in cond for cond in data["conditions"].values() if isinstance(cond, dict))

    assert has_explicit_non_mock or has_model_spec or has_conditions_with_model, (
        f"{file_path} does not specify mock: false or a valid model/target identifier"
    )

    # 3. Must not contain simulation disclaimer (simulations belong in docs/appendix_simulation/)
    assert "SIMULATION_NOTICE" not in data, f"{file_path} contains simulation notice; move to docs/appendix_simulation/"

def test_evidence_index_exists():
    index_path = RESULTS_DIR / "EVIDENCE_INDEX.md"
    assert index_path.exists(), "results/EVIDENCE_INDEX.md must exist"
    content = index_path.read_text(encoding="utf-8")
    assert "results/case_studies/gemini_flash_lite_audit.json" in content
