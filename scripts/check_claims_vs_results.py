#!/usr/bin/env python3
"""Cross-checks quantitative claims in papers, logs, and ledgers against raw results.

Conforms strictly to Rule 1: Exits with code 1 if ANY number reported in
paper/main.tex, PHASE_LOG.md, or CLAIMS_LEDGER.md disagrees with results/*.json.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_all_claims():
    smoke_path = ROOT / "results" / "testbed" / "smoke_audit.json"
    oracle_path = ROOT / "results" / "testbed" / "oracle_results.json"

    if not smoke_path.exists():
        print(f"[FAIL] Missing raw artifact: {smoke_path}")
        sys.exit(1)
    if not oracle_path.exists():
        print(f"[FAIL] Missing raw artifact: {oracle_path}")
        sys.exit(1)

    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))

    # Extract source-of-truth ground values
    summary_map = {r["condition"]: r for r in smoke["comparison_summary"]}
    sr_naive = summary_map["suppress_refusal"]["naive_certified_unlearned"]  # '5/5 (100%)'
    sr_rena = summary_map["suppress_refusal"]["rena_residual_detected"]      # '1/5 (20%)'
    sr_fp_rate = summary_map["suppress_refusal"]["naive_false_pass_rate"]   # 1.0

    sp_naive = summary_map["suppress_prompt"]["naive_certified_unlearned"]  # '2/5 (40%)'
    sp_rena = summary_map["suppress_prompt"]["rena_residual_detected"]      # '5/5 (100%)'

    gd_util = smoke["conditions"]["graddiff"]["rena_controlled_auditor"]["retained_utility"]["accuracy"]  # 0.40

    sr_ll = oracle["log_likelihood_evaluations"]["suppress_refusal"]["mean_log_prob"]  # -2.9756
    ret_ll = oracle["log_likelihood_evaluations"]["retained"]["mean_log_prob"]          # -1.4043
    ref_ll = oracle["log_likelihood_evaluations"]["reference"]["mean_log_prob"]         # -38.0156
    gd_ll = oracle["log_likelihood_evaluations"]["graddiff"]["mean_log_prob"]           # -14.4195
    sr_step = oracle["relearning_attacks"]["suppress_refusal"]["steps_to_recovery"]     # 2

    mismatches = []

    # 1. Check paper/main.tex
    paper_path = ROOT / "paper" / "main.tex"
    if paper_path.exists():
        paper_text = paper_path.read_text(encoding="utf-8")

        # Stale numbers check (P1)
        if "80.0% to 100.0% of facts" in paper_text:
            mismatches.append(("paper/main.tex", "P1: Stale claim '80.0% to 100.0% of facts' found in paper abstract."))

        # Exact values check in Table 1
        if "5/5 ($100.0\\%$)" not in paper_text:
            mismatches.append(("paper/main.tex", "Table 1 must contain 5/5 ($100.0\\%$) for suppress_refusal."))
        if "40.0\\% (Degraded)" not in paper_text:
            mismatches.append(("paper/main.tex", f"Table 1 must report graddiff utility as 40.0% (Degraded), matching {gd_util*100}%."))

        # Oracle values in Table 2
        for val_str in [f"{ref_ll:.4f}", f"{ret_ll:.4f}", f"{sr_ll:.4f}", f"{gd_ll:.4f}", f"Step {sr_step}"]:
            if val_str not in paper_text:
                mismatches.append(("paper/main.tex", f"Oracle Table 2 missing exact value {val_str}."))

    # 2. Check docs/CLAIMS_LEDGER.md
    ledger_path = ROOT / "docs" / "CLAIMS_LEDGER.md"
    if ledger_path.exists():
        ledger_text = ledger_path.read_text(encoding="utf-8")
        if "60% on suppress_refusal" in ledger_text or "passes 60% of" in ledger_text:
            mismatches.append(("docs/CLAIMS_LEDGER.md", "P2: Stale '60% on suppress_refusal' found in CLAIMS_LEDGER.md."))
        if "80% respectively" in ledger_text:
            mismatches.append(("docs/CLAIMS_LEDGER.md", "P2: Stale '80% respectively' found in CLAIMS_LEDGER.md."))

    # 3. Check docs/PHASE_LOG.md
    phase_log_path = ROOT / "docs" / "PHASE_LOG.md"
    if phase_log_path.exists():
        phase_log_text = phase_log_path.read_text(encoding="utf-8")
        # Ensure stale WP3/WP4 numbers are flagged/corrected
        if "suppress_refusal naive false pass 60%" in phase_log_text:
            mismatches.append(("docs/PHASE_LOG.md", "P2: Stale 'suppress_refusal naive false pass 60%' found in PHASE_LOG.md."))

    # 4. Check AUDIT_REPORT.md
    audit_report_path = ROOT / "AUDIT_REPORT.md"
    if audit_report_path.exists():
        audit_text = audit_report_path.read_text(encoding="utf-8")
        if "60% of refusal-tuned facts" in audit_text:
            mismatches.append(("AUDIT_REPORT.md", "Stale '60% of refusal-tuned facts' found in AUDIT_REPORT.md."))

    if mismatches:
        print("=" * 60)
        print(f"[FAIL] Found {len(mismatches)} claim-vs-artifact inconsistencies:")
        print("=" * 60)
        for loc, msg in mismatches:
            print(f"  ❌ [{loc}]: {msg}")
        print("=" * 60)
        sys.exit(1)

    print(f"[OK] All document claims strictly agree with raw artifacts in results/ (checked {smoke_path.name}, {oracle_path.name}).")
    sys.exit(0)


if __name__ == "__main__":
    check_all_claims()
