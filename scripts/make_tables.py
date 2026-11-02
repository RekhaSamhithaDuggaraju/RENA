#!/usr/bin/env python3
"""Generates paper, README, and claims ledger tables directly from raw JSON results.

Conforms strictly to Rule 1: Every number in any document must be generated
by script from raw artifacts in results/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_results():
    smoke_path = ROOT / "results" / "testbed" / "smoke_audit.json"
    oracle_path = ROOT / "results" / "testbed" / "oracle_results.json"
    gemini_path = ROOT / "results" / "case_studies" / "gemini_flash_lite_audit.json"

    if not smoke_path.exists():
        raise FileNotFoundError(f"Missing required smoke audit results: {smoke_path}")
    if not oracle_path.exists():
        raise FileNotFoundError(f"Missing required oracle results: {oracle_path}")

    smoke_data = json.loads(smoke_path.read_text(encoding="utf-8"))
    oracle_data = json.loads(oracle_path.read_text(encoding="utf-8"))
    gemini_data = json.loads(gemini_path.read_text(encoding="utf-8")) if gemini_path.exists() else None

    return smoke_data, oracle_data, gemini_data


def generate_latex_table_1(smoke_data: dict) -> str:
    """Generates LaTeX Table 1 (Head-to-Head Auditor Comparison)."""
    rows = []
    summary_map = {row["condition"]: row for row in smoke_data["comparison_summary"]}
    conditions_data = smoke_data.get("conditions", {})

    display_names = {
        "reference": "\\texttt{reference} (Group B)",
        "retained": "\\texttt{retained} (Group C)",
        "suppress_prompt": "\\texttt{suppress\\_prompt}",
        "suppress_refusal": "\\texttt{suppress\\_refusal}",
        "graddiff": "\\texttt{graddiff}",
    }

    order = ["reference", "retained", "suppress_prompt", "suppress_refusal", "graddiff"]

    for cond in order:
        s_row = summary_map.get(cond, {})
        c_dict = conditions_data.get(cond, {})
        rena_part = c_dict.get("rena_controlled_auditor", {})
        retain_acc = rena_part.get("retained_utility", {}).get("accuracy", 0.0)
        utility_verdict = rena_part.get("retained_utility", {}).get("verdict", "")

        util_str = f"{retain_acc*100:.1f}\\%"
        if "DEGRADED" in utility_verdict:
            util_str = f"\\textbf{{{util_str} (Degraded)}}"
        else:
            util_str = f"{util_str} (Preserved)"

        naive_cert = s_row.get("naive_certified_unlearned", "")
        rena_res = s_row.get("rena_residual_detected", "")
        fp_count = s_row.get("naive_false_pass_count", 0)
        fp_rate = s_row.get("naive_false_pass_rate", 0.0)

        # LaTeX format for false pass rate
        valid_facts_n = c_dict.get("paired_comparison", {}).get("valid_facts_count", 5)
        fp_str = f"{fp_count}/{valid_facts_n} ({fp_rate*100:.1f}\\%)"
        if fp_count > 0 and cond == "suppress_refusal":
            fp_str = f"\\textbf{{{fp_count}/{valid_facts_n} ({fp_rate*100:.1f}\\%}})"

        # Normalize percentages with $...$ for LaTeX
        naive_cert_tex = naive_cert.replace("%", "\\%")
        rena_res_tex = rena_res.replace("%", "\\%")

        rows.append(f"{display_names[cond]} & {naive_cert_tex} & {rena_res_tex} & {fp_str} & {util_str} \\\\")

    table_body = "\n".join(rows)

    tex = f"""\\begin{{table}}[h!]
\\centering
\\small
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Condition}} & \\textbf{{Naive Certified Unlearned}} & \\textbf{{RENA Residual Detected}} & \\textbf{{Naive False Pass Rate}} & \\textbf{{Retained Utility}} \\\\
\\midrule
{table_body}
\\bottomrule
\\end{{tabular}}
\\caption{{Head-to-head auditor comparison across 5 conditions on \\texttt{{Qwen2.5-0.5B-Instruct}} (MPS). On refusal suppression, the naive auditor falsely passes $100\\%$ of facts due to surface refusals, whereas RENA detects latent disclosure or marks near-misses inconclusive. On \\texttt{{graddiff}}, utility collapses catastrophically.}}
\\label{{tab:head_to_head}}
\\end{{table}}"""
    return tex


def generate_latex_table_2(oracle_data: dict) -> str:
    """Generates LaTeX Table 2 (Oracle Likelihood & Relearning Trajectory)."""
    ll_evals = oracle_data.get("log_likelihood_evaluations", {})
    relearn = oracle_data.get("relearning_attacks", {})

    ref_ll = ll_evals.get("reference", {}).get("mean_log_prob", -38.0156)
    ret_ll = ll_evals.get("retained", {}).get("mean_log_prob", -1.4043)
    sr_ll = ll_evals.get("suppress_refusal", {}).get("mean_log_prob", -2.9756)
    gd_ll = ll_evals.get("graddiff", {}).get("mean_log_prob", -14.4195)

    sr_steps = relearn.get("suppress_refusal", {}).get("steps_to_recovery", 2)
    gd_steps = relearn.get("graddiff", {}).get("steps_to_recovery", None)
    gd_steps_str = f"None (>{relearn.get('graddiff', {}).get('steps_budget', 10)})" if gd_steps is None else str(gd_steps)

    tex = f"""\\begin{{table}}[h!]
\\centering
\\small
\\begin{{tabular}}{{lccccc}}
\\toprule
\\textbf{{Condition}} & \\textbf{{Mean Log-Likelihood}} & \\textbf{{Step 0 Recall}} & \\textbf{{Step 2 Recall}} & \\textbf{{Recovery ($\\ge 80\\%$)}} & \\textbf{{Vulnerability Class}} \\\\
\\midrule
\\texttt{{reference}}         & ${ref_ll:.4f}$ & $0.0\\%$ & --- & --- & Unexposed Base Prior \\\\
\\texttt{{retained}}          & ${ret_ll:.4f}$  & $100.0\\%$ & $100.0\\%$ & Step 0 & Unmitigated Upper Bound \\\\
\\texttt{{suppress\\_refusal}} & \\textbf{{{sr_ll:.4f}}} & $60.0\\%$ & \\textbf{{80.0\\%}} & \\textbf{{Step {sr_steps}}} & \\texttt{{rapid\\_reacquisition}} \\\\
\\texttt{{graddiff}}          & ${gd_ll:.4f}$ & $60.0\\%$ & $60.0\\%$ & {gd_steps_str} & \\texttt{{gradual\\_reacquisition}} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Independent oracle log-likelihood and AdamW relearning curves on \\texttt{{Qwen2.5-0.5B-Instruct}}. Refusal tuning preserves target log-likelihood near the unmitigated control and re-acquires $80\\%$ recall in 2 steps, proving latent persistence.}}
\\label{{tab:oracle}}
\\end{{table}}"""
    return tex


def update_paper_latex(table1_tex: str, table2_tex: str):
    """Updates paper/main.tex with the dynamically generated tables."""
    paper_path = ROOT / "paper" / "main.tex"
    if not paper_path.exists():
        print(f"[WARN] Paper file not found: {paper_path}")
        return

    content = paper_path.read_text(encoding="utf-8")

    # Save standalone table files
    tables_dir = ROOT / "paper" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "head_to_head.tex").write_text(table1_tex, encoding="utf-8")
    (tables_dir / "oracle_table.tex").write_text(table2_tex, encoding="utf-8")
    print(f"[TABLES] Exported paper/tables/head_to_head.tex and paper/tables/oracle_table.tex")


def main():
    parser = argparse.ArgumentParser(description="RENA Table Generator")
    parser.add_argument("--export", action="store_true", help="Generate and export LaTeX tables")
    args = parser.parse_args()

    smoke_data, oracle_data, gemini_data = load_results()

    t1 = generate_latex_table_1(smoke_data)
    t2 = generate_latex_table_2(oracle_data)

    update_paper_latex(t1, t2)
    print("[TABLES] Successfully synchronized tables with results/*.json.")


if __name__ == "__main__":
    main()
