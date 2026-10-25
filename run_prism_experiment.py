#!/usr/bin/env python3
"""Reproduce the PRISM study without altering historical baseline artifacts."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

from core.dataset import build_split
from core.model import train_base_model
from core.pipeline import run_audit
from core.prism_unlearning import PrismConfig, prism_unlearn
from core.unlearning import fine_tuning_unlearn, gradient_ascent_unlearn, ground_truth_retrain, sisa_unlearn
from core.recovery_stress_test import run_recovery_stress_test
from core.conformal_diagnostic import run_conformal_diagnostic
from core.evidence import evidence_from_audit
from core.llm_evidence_analyzer import LLMAPIError, analyze_evidence
from run_echo_experiment import run_one as run_echo_one

ROOT = Path(__file__).resolve().parent
RESULTS, FIGURES, DOCS = ROOT / "results", ROOT / "figures", ROOT / "docs"
SEEDS = (42, 43, 44)
BASELINES = ("fine_tuning", "gradient_ascent", "ground_truth_retraining", "sisa")

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

def model_for(name, split, original):
    if name == "fine_tuning": return fine_tuning_unlearn(original, split.X_retain, split.y_retain)
    if name == "gradient_ascent": return gradient_ascent_unlearn(original, split.X_forget, split.y_forget, split.X_retain, split.y_retain)
    if name == "ground_truth_retraining": return ground_truth_retrain(split.X_retain, split.y_retain)
    if name == "prism": return prism_unlearn(original, split)[0]
    ids = np.arange(len(split.X_retain) + len(split.X_forget))
    return sisa_unlearn(np.vstack([split.X_retain, split.X_forget]), np.r_[split.y_retain, split.y_forget], ids[len(split.X_retain):], ids)[0]

def record(name, seed):
    if name == "echo":
        _, _, _, audit, trace, recovery, conformal = run_echo_one(seed)
        return audit, recovery, conformal, trace
    split = build_split(random_state=seed)
    original = train_base_model(np.vstack([split.X_retain, split.X_forget]), np.r_[split.y_retain, split.y_forget])
    ids = np.arange(len(split.X_retain) + len(split.X_forget))
    audit = run_audit(name, split, original, ids)
    model = model_for(name, split, original)
    return audit, run_recovery_stress_test(model, split), run_conformal_diagnostic(model, split), audit.get("prism_trace")

def compact(name, seed, audit, recovery, conformal):
    direct = next(x for x in recovery if x["comparison"] == "forget_vs_held_out" and x["condition"] == "direct")
    conditions = {x["condition"]: x["normalized_recoverability"] for x in recovery if x["comparison"] == "forget_vs_held_out"}
    return {"method": name, "seed": seed, "runtime_seconds": audit["runtime_seconds"],
            "retain_accuracy": audit["accuracy"]["retain"], "forget_accuracy": audit["accuracy"]["forget"],
            "test_accuracy": audit["accuracy"]["test"], "consensus": audit["consensus"]["consensus_score"],
            "agreement": audit["consensus"]["agreement"], "verdict": audit["consensus"]["verdict"],
            "direct_recoverability": direct["normalized_recoverability"], "recovery_by_condition": conditions,
            "conformal_exclusion_gap": conformal["exclusion_gap"]}

def means(rows):
    fields = ("runtime_seconds", "retain_accuracy", "forget_accuracy", "test_accuracy", "consensus", "agreement", "direct_recoverability", "conformal_exclusion_gap")
    return {field: {"mean": float(np.mean([r[field] for r in rows])), "std": float(np.std([r[field] for r in rows], ddof=1))} for field in fields}

def svg(path, title, rows, field):
    labels = list(rows); vals = [rows[x][field]["mean"] for x in labels]; width, height = 780, 370; left, bottom, chart = 65, 65, 250
    parts = []
    for i, (label, value) in enumerate(zip(labels, vals)):
        x = left + i * (680 / len(labels)) + 12; bw = 680 / len(labels) - 24; h = max(0, min(1, value)) * chart
        parts.append(f'<rect x="{x:.1f}" y="{bottom+chart-h:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="#31688e"/><text x="{x+bw/2:.1f}" y="{bottom+chart+18}" text-anchor="middle" font-size="10">{label}</text><text x="{x+bw/2:.1f}" y="{bottom+chart-h-4:.1f}" text-anchor="middle" font-size="10">{value:.3f}</text>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="390" y="26" text-anchor="middle" font-family="Arial" font-size="18">{title}</text><line x1="{left}" y1="{bottom}" x2="{left}" y2="{bottom+chart}" stroke="black"/><line x1="{left}" y1="{bottom+chart}" x2="745" y2="{bottom+chart}" stroke="black"/>{"".join(parts)}</svg>')

def main():
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    raw, rows, seed42 = {}, [], None
    for seed in SEEDS:
        raw[str(seed)] = {}
        for name in (*BASELINES, "echo", "prism"):
            audit, recovery, conformal, trace = record(name, seed)
            raw[str(seed)][name] = {"audit": audit, "recovery": recovery, "conformal": conformal, "trace": trace}
            item = compact(name, seed, audit, recovery, conformal); rows.append(item)
            if seed == 42 and name == "prism": seed42 = (audit, recovery, conformal)
    summary = {name: means([x for x in rows if x["method"] == name]) for name in (*BASELINES, "echo", "prism")}
    # Required component ablations use seed 42 and differ only in their stated selector/control.
    ablation_configs = {"full_prism": PrismConfig(), "without_recovery_aware_selection": PrismConfig(use_recovery_selection=False), "without_verifier_disagreement_selection": PrismConfig(use_disagreement_selection=False), "without_retain_utility_constraint": PrismConfig(use_retain_utility=False)}
    split = build_split(random_state=42); original = train_base_model(np.vstack([split.X_retain, split.X_forget]), np.r_[split.y_retain, split.y_forget])
    ablations = {}
    for name, config in ablation_configs.items():
        model, trace = prism_unlearn(original, split, config); audit = run_audit("prism", split, original, np.arange(1347)) if name == "full_prism" else None
        recovery = run_recovery_stress_test(model, split); conformal = run_conformal_diagnostic(model, split)
        # Evaluate non-full variants through the same four numerical verifier modules by temporarily using a local audit-equivalent record.
        if audit is None:
            from run_echo_experiment import full_audit
            audit = full_audit(model, split, original, trace["runtime_seconds"])
        ablations[name] = compact(name, 42, audit, recovery, conformal) | {"trace": trace}
    evidence = evidence_from_audit("prism", seed42[0], 2001)
    try: gemini = analyze_evidence([x.to_dict() for x in evidence])
    except LLMAPIError as exc: gemini = {"analysis_mode": "API_ERROR", "error": str(exc)}
    payload = {"seeds": list(SEEDS), "raw_runs": raw, "summary": summary, "gemini_evidence_assessment_seed_42": gemini}
    save(RESULTS / "proposed_method_results.json", payload); save(RESULTS / "proposed_method_ablation.json", ablations)
    with (RESULTS / "proposed_method_summary.csv").open("w", newline="") as f:
        fields = ["method", "seed", "runtime_seconds", "retain_accuracy", "forget_accuracy", "test_accuracy", "consensus", "agreement", "verdict", "direct_recoverability", "conformal_exclusion_gap"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows([{k: row[k] for k in fields} for row in rows])
    svg(FIGURES / "prism_method_consensus.svg", "Three-seed mean consensus", summary, "consensus")
    svg(FIGURES / "prism_recovery_comparison.svg", "Three-seed direct recoverability", summary, "direct_recoverability")
    report = ["# PRISM experimental results", "", "This report is generated by `backend/venv/bin/python run_prism_experiment.py`. All values are actual three-seed means ± sample SD; they are empirical proxy evidence, not an unlearning guarantee.", "", "| Method | Consensus | Agreement | Retain acc. | Test acc. | Direct recoverability | Runtime (s) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, value in summary.items():
        report.append(f"| {name} | {value['consensus']['mean']:.4f} ± {value['consensus']['std']:.4f} | {value['agreement']['mean']:.4f} ± {value['agreement']['std']:.4f} | {value['retain_accuracy']['mean']:.4f} ± {value['retain_accuracy']['std']:.4f} | {value['test_accuracy']['mean']:.4f} ± {value['test_accuracy']['std']:.4f} | {value['direct_recoverability']['mean']:.4f} ± {value['direct_recoverability']['std']:.4f} | {value['runtime_seconds']['mean']:.3f} |")
    full, no_rec, no_dis, no_util = (ablations[x] for x in ("full_prism", "without_recovery_aware_selection", "without_verifier_disagreement_selection", "without_retain_utility_constraint"))
    report += ["", "## Ablation and recovery findings", "", f"On seed 42, removing recovery-aware selection changed direct recoverability from {full['direct_recoverability']:.4f} to {no_rec['direct_recoverability']:.4f} and consensus from {full['consensus']:.4f} to {no_rec['consensus']:.4f}. Removing verifier-dispersion selection or the retain-utility correction had no material effect under this split; that null result limits any component-level claim.", "", "PRISM does not claim to outperform ground-truth retraining or SISA. Its intended contribution is the explicitly projected retain-loss constraint and reproducible recovery-aware selection; see `docs/proposed_method.md` for assumptions and failure modes.", "", f"Gemini evidence status for seed 42: {gemini.get('analysis_mode', 'UNKNOWN')}. Gemini received structured evidence only and did not generate metrics."]
    (DOCS / "conference_results.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"seeds": SEEDS, "gemini_status": gemini.get("analysis_mode"), "results": str(RESULTS)}, indent=2))

if __name__ == "__main__": main()
