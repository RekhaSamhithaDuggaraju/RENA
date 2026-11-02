"""Test for scripts/check_claims_vs_results.py.

Verifies Task T1.1 acceptance criteria:
1. Checker exits 0 on current valid files.
2. Deliberately injecting an inconsistent number into paper/main.tex or CLAIMS_LEDGER.md causes the checker to exit 1.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_checker_exits_0_on_valid_repository():
    """Verify check_claims_vs_results.py exits 0 when all document numbers agree with raw results."""
    cmd = [sys.executable, str(ROOT / "scripts" / "check_claims_vs_results.py")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Expected 0, got {res.returncode}. Output:\n{res.stdout}\n{res.stderr}"
    assert "[OK]" in res.stdout


def test_checker_exits_1_on_tampered_claim(tmp_path):
    """Verify check_claims_vs_results.py fails with exit code 1 if a number is tampered with."""
    paper_path = ROOT / "paper" / "main.tex"
    original_text = paper_path.read_text(encoding="utf-8")

    try:
        # Deliberately inject stale/inconsistent number into paper/main.tex
        tampered_text = original_text.replace("40.0\\% (Degraded)", "95.0\\% (Degraded)")
        paper_path.write_text(tampered_text, encoding="utf-8")

        cmd = [sys.executable, str(ROOT / "scripts" / "check_claims_vs_results.py")]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 1, f"Expected exit code 1 on tampered number, got {res.returncode}"
        assert "[FAIL]" in res.stdout
    finally:
        # Always restore original file
        paper_path.write_text(original_text, encoding="utf-8")
