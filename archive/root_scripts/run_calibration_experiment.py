#!/usr/bin/env python3
"""Run the RENA retrain-alignment calibration study.

All result files are newly written under ``results/calibration``.  Existing
conference outputs are intentionally never read as numerical inputs or replaced.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from core.checkpoints import save_model_state
from core.consensus import consensus_baselines, compute_consensus
from core.dataset import build_split
from core.echo_controller import echo_unlearn
from core.model import evaluate, train_base_model
from core.prism_unlearning import prism_unlearn
from core.retrain_alignment import relative_retrain_alignment
from core.sisa import SISAEnsemble
from core.unlearning import fine_tuning_unlearn, gradient_ascent_unlearn, ground_truth_retrain, sisa_unlearn
from core.verification import (bootstrap_statistical_witness, confidence_analysis,
                               loss_distribution_test, membership_inference_attack)

ROOT = Path(__file__).resolve().parent
METHODS = ("fine_tuning", "gradient_ascent", "ground_truth_retraining", "sisa", "prism", "echo")
SEEDS = tuple(range(42, 52))
FRACTIONS = (0.05, 0.10, 0.20, 0.30)
RRA_FORGOTTEN_THRESHOLD = 0.5


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_models(method: str, split, original):
    """Apply only verified repository methods and return the audited model."""
    if method == "fine_tuning":
        return fine_tuning_unlearn(original, split.X_retain, split.y_retain)
    if method == "gradient_ascent":
        return gradient_ascent_unlearn(original, split.X_forget, split.y_forget, split.X_retain, split.y_retain)
    if method == "ground_truth_retraining":
        return ground_truth_retrain(split.X_retain, split.y_retain)
    if method == "sisa":
        full_x = np.vstack([split.X_retain, split.X_forget]); full_y = np.r_[split.y_retain, split.y_forget]
        model, n_retrained = sisa_unlearn(full_x, full_y, np.arange(len(split.X_retain), len(full_x)), np.arange(len(full_x)))
        return model, {"sisa_shards_retrained": n_retrained}
    if method == "prism":
        return prism_unlearn(original, split)
    if method == "echo":
        return echo_unlearn(original, split)
    raise ValueError(method)


def _model_and_trace(method, split, original):
    result = build_models(method, split, original)
    if isinstance(result, tuple):
        return result
    return result, {}


def witness_results(model, original, split):
    test_accuracy = evaluate(model, split.X_test, split.y_test)["accuracy"]
    return [
        membership_inference_attack(model, split.X_forget, split.y_forget, split.X_test, split.y_test, test_accuracy),
        loss_distribution_test(model, split.X_forget, split.y_forget, split.X_test, split.y_test),
        confidence_analysis(model, original, split.X_forget, split.y_forget, split.X_test, split.y_test),
        bootstrap_statistical_witness(model, split.X_forget, split.y_forget, split.X_retain, split.y_retain),
    ]


def run_case(method: str, seed: int, fraction: float, checkpoint_dir: Path) -> dict:
    split = build_split(forget_fraction=fraction, random_state=seed)
    full_x, full_y = np.vstack([split.X_retain, split.X_forget]), np.r_[split.y_retain, split.y_forget]
    original = train_base_model(full_x, full_y)
    retrained = ground_truth_retrain(split.X_retain, split.y_retain)
    started = time.perf_counter()
    audited, trace = _model_and_trace(method, split, original)
    runtime = time.perf_counter() - started
    modules = witness_results(audited, original, split)
    module_dicts = [m.to_dict() for m in modules]
    strategies = consensus_baselines(modules)
    rra = relative_retrain_alignment(original, audited, retrained, split.X_test)
    metadata = {"method": method, "seed": seed, "forget_fraction": fraction, "dataset": "sklearn_digits", "architecture": type(audited).__name__}
    base = checkpoint_dir / f"method={method}" / f"seed={seed}_forget={fraction:.2f}"
    save_model_state(original, base.with_name(base.name + "_original"), {**metadata, "role": "original", "architecture": type(original).__name__})
    save_model_state(retrained, base.with_name(base.name + "_retrained"), {**metadata, "role": "ground_truth_retrained", "architecture": type(retrained).__name__})
    save_model_state(audited, base.with_name(base.name + "_audited"), {**metadata, "role": "audited"})
    return {
        "method": method, "seed": seed, "forget_fraction": fraction, "runtime_seconds": runtime,
        "accuracy": {"retain": evaluate(audited, split.X_retain, split.y_retain)["accuracy"], "test": evaluate(audited, split.X_test, split.y_test)["accuracy"], "forget": evaluate(audited, split.X_forget, split.y_forget)["accuracy"]},
        "module_results": module_dicts, "strategies": strategies, "rra": rra, "trace": trace,
    }


def _spearman(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3: return {"rho": None, "p_value": None, "n": int(valid.sum())}
    result = stats.spearmanr(x[valid], y[valid])
    return {"rho": float(result.statistic), "p_value": float(result.pvalue), "n": int(valid.sum())}


def bootstrap_spearman(x, y, n_bootstrap=2000, random_state=20260907):
    rng = np.random.RandomState(random_state); n = len(x); samples = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n); value = stats.spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(value): samples.append(value)
    if not samples: return {"point_estimate": None, "lower_ci": None, "upper_ci": None, "n_bootstrap_valid": 0}
    return {"point_estimate": _spearman(x, y)["rho"], "lower_ci": float(np.percentile(samples, 2.5)), "upper_ci": float(np.percentile(samples, 97.5)), "n_bootstrap_valid": len(samples), "n_bootstrap_requested": n_bootstrap, "random_state": random_state}


def flatten_runs(runs):
    rows = []
    for run in runs:
        row = {"method": run["method"], "seed": run["seed"], "forget_fraction": run["forget_fraction"], "runtime_seconds": run["runtime_seconds"], **{f"{k}_accuracy": v for k, v in run["accuracy"].items()}, **run["rra"]}
        for module in run["module_results"]: row[f"witness_{module['module_name']}"] = module["forget_score"]; row[f"reliability_{module['module_name']}"] = module["reliability"]
        for name, value in run["strategies"].items(): row[f"{name}_score"] = value["consensus_score"]; row[f"{name}_agreement"] = value["agreement"]; row[f"{name}_verdict"] = value["verdict"]
        rows.append(row)
    return rows


def analyze(runs, result_dir: Path):
    rows = flatten_runs(runs); rra = np.array([row["rra"] for row in rows])
    score_columns = [key for key in rows[0] if key.endswith("_score") or key.startswith("witness_")]
    correlations = {key: _spearman(np.array([row[key] for row in rows]), rra) for key in score_columns}
    bootstraps = {key: bootstrap_spearman(np.array([row[key] for row in rows]), rra) for key in score_columns}
    bins = np.linspace(0, 1, 6); calibration = []
    for label in ("RENA_FULL_score", "NAIVE_AVERAGE_score", "WEIGHTED_NO_OVERRIDE_score"):
        values = np.array([row[label] for row in rows])
        for index in range(len(bins)-1):
            mask = (values >= bins[index]) & ((values < bins[index+1]) if index < len(bins)-2 else (values <= bins[index+1]))
            calibration.append({"signal": label, "bin_lower": float(bins[index]), "bin_upper": float(bins[index+1]), "mean_audit_score": float(values[mask].mean()) if mask.any() else None, "mean_rra": float(rra[mask].mean()) if mask.any() else None, "sample_count": int(mask.sum())})
    positives = []
    for run, row in zip(runs, rows):
        if row["RENA_FULL_verdict"] == "Likely Forgotten" and row["rra"] < RRA_FORGOTTEN_THRESHOLD:
            positives.append({"method": row["method"], "seed": row["seed"], "forget_fraction": row["forget_fraction"], "consensus": row["RENA_FULL_score"], "agreement": row["RENA_FULL_agreement"], "verdict": row["RENA_FULL_verdict"], "rra": row["rra"], "rra_distances": {"audit_original": row["distance_audit_original"], "audit_retrained": row["distance_audit_retrained"]}, "witness_scores": {m["module_name"]: m["forget_score"] for m in run["module_results"]}})
    # Low agreement is defined prospectively from the existing 0.5 override threshold.
    claimed = [row for row in rows if row["RENA_FULL_verdict"] == "Likely Forgotten"]
    table = [[sum(r["rra"] < .5 and r["RENA_FULL_agreement"] < .5 for r in claimed), sum(r["rra"] >= .5 and r["RENA_FULL_agreement"] < .5 for r in claimed)], [sum(r["rra"] < .5 and r["RENA_FULL_agreement"] >= .5 for r in claimed), sum(r["rra"] >= .5 and r["RENA_FULL_agreement"] >= .5 for r in claimed)]]
    fisher = {"table": table, "note": "Not computed: no RENA_FULL Likely Forgotten cases."} if not claimed or any(sum(x) == 0 for x in table) else {"table": table, **dict(zip(("odds_ratio", "p_value"), map(float, stats.fisher_exact(table))))}
    risk = []
    for strategy in ("RENA_FULL", "WEIGHTED_NO_OVERRIDE", "NAIVE_AVERAGE"):
        eligible = [row for row in rows if row[f"{strategy}_verdict"] != "Inconclusive"]
        errors = [int((row[f"{strategy}_verdict"] == "Likely Forgotten") != (row["rra"] >= .5)) for row in eligible]
        risk.append({"strategy": strategy, "coverage": len(eligible)/len(rows), "covered_cases": len(eligible), "error_rate": float(np.mean(errors)) if errors else None, "false_positive_rate": float(sum(e and row[f"{strategy}_verdict"] == "Likely Forgotten" for e,row in zip(errors, eligible)) / max(1, sum(row["rra"] < .5 for row in eligible))), "false_negative_rate": float(sum(e and row[f"{strategy}_verdict"] != "Likely Forgotten" for e,row in zip(errors, eligible)) / max(1, sum(row["rra"] >= .5 for row in eligible))), "aurc": None, "limitation": "AURC requires a ranked selective-confidence sweep; these discrete verdict strategies provide one operating point."})
    dump(result_dir / "correlation_results.json", correlations); dump(result_dir / "bootstrap_results.json", bootstraps); dump(result_dir / "false_positive_analysis.json", {"rra_threshold": .5, "cases": positives, "fisher_exact": fisher}); dump(result_dir / "risk_coverage_results.json", {"rra_threshold": .5, "operating_points": risk, "calibration_relationship": calibration}); dump(result_dir / "ablation_results.json", {name: _spearman(np.array([r[f"{name}_score"] for r in rows]), rra) for name in ("WEIGHTING_ON_OVERRIDE_ON", "WEIGHTING_ON_OVERRIDE_OFF", "WEIGHTING_OFF_OVERRIDE_ON", "WEIGHTING_OFF_OVERRIDE_OFF")})
    with (result_dir / "calibration_runs.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=sorted(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {"n_runs": len(rows), "methods": list(METHODS), "seeds": sorted(set(r["seed"] for r in rows)), "forget_fractions": sorted(set(r["forget_fraction"] for r in rows)), "rra_threshold": .5, "independence_note": "RRA uses only prediction distributions of original/audited/retrained models on held-out test inputs; it does not use RENA witnesses or consensus.", "correlations": correlations}
    dump(result_dir / "calibration_summary.json", summary)
    return rows, calibration, risk, positives


def figures(rows, calibration, risk, positives, figure_dir: Path):
    import matplotlib.pyplot as plt
    figure_dir.mkdir(parents=True, exist_ok=True)
    def save(name): plt.tight_layout(); plt.savefig(figure_dir / f"{name}.png", dpi=180); plt.savefig(figure_dir / f"{name}.pdf"); plt.close()
    rra = [x["rra"] for x in rows]; plt.figure(figsize=(6,4)); plt.scatter([x["RENA_FULL_score"] for x in rows], rra, label="RENA full", alpha=.7); plt.scatter([x["NAIVE_AVERAGE_score"] for x in rows], rra, label="Naive average", alpha=.7); plt.xlabel("Audit score"); plt.ylabel("RRA"); plt.legend(); save("rra_vs_consensus")
    plt.figure(figsize=(6,4));
    for signal in ("RENA_FULL_score", "NAIVE_AVERAGE_score"):
        d=[x for x in calibration if x["signal"]==signal and x["sample_count"]]; plt.plot([x["mean_audit_score"] for x in d], [x["mean_rra"] for x in d], "o-", label=signal.replace("_score", ""))
    plt.xlabel("Mean audit score (bin)"); plt.ylabel("Mean RRA"); plt.legend(); save("calibration_relationship")
    plt.figure(figsize=(6,4)); plt.scatter([x["coverage"] for x in risk], [x["error_rate"] for x in risk]); [plt.annotate(x["strategy"], (x["coverage"],x["error_rate"])) for x in risk if x["error_rate"] is not None]; plt.xlabel("Coverage"); plt.ylabel("Error rate on covered cases"); save("risk_coverage_curve")
    witness=[key for key in rows[0] if key.startswith("witness_")]; rho=[_spearman(np.array([r[k] for r in rows]),np.array(rra))["rho"] for k in witness]; plt.figure(figsize=(7,4)); plt.bar(range(len(witness)), rho); plt.xticks(range(len(witness)), [x.replace("witness_", "") for x in witness], rotation=25, ha="right"); plt.ylabel("Spearman rho with RRA"); save("witness_vs_rra")
    plt.figure(figsize=(6,4)); plt.scatter([x["agreement"] for x in positives], [x["rra"] for x in positives]) if positives else plt.text(.5,.5,"No candidate false positives",ha="center"); plt.xlabel("Agreement"); plt.ylabel("RRA"); save("false_positive_cases")
    names=["WEIGHTING_ON_OVERRIDE_ON","WEIGHTING_ON_OVERRIDE_OFF","WEIGHTING_OFF_OVERRIDE_ON","WEIGHTING_OFF_OVERRIDE_OFF"]; vals=[_spearman(np.array([r[f"{n}_score"] for r in rows]),np.array(rra))["rho"] for n in names]; plt.figure(figsize=(7,4)); plt.bar(range(4),vals); plt.xticks(range(4),[n.replace("WEIGHTING_", "W ").replace("_OVERRIDE", " / O") for n in names],rotation=20,ha="right"); plt.ylabel("Spearman rho with RRA"); save("consensus_ablation")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--quick", action="store_true", help="one seed/fraction smoke run; not the study configuration"); args = parser.parse_args()
    seeds, fractions = ((42,), (.10,)) if args.quick else (SEEDS, FRACTIONS)
    result_dir, checkpoint_dir, figure_dir = ROOT / "results" / "calibration", ROOT / "results" / "calibration" / "checkpoints", ROOT / "figures" / "calibration"
    runs=[]
    for seed in seeds:
        for fraction in fractions:
            for method in METHODS:
                print(f"running method={method} seed={seed} forget_fraction={fraction:.2f}", flush=True)
                runs.append(run_case(method, seed, fraction, checkpoint_dir))
    dump(result_dir / "calibration_runs.json", runs)
    rows, calibration, risk, positives = analyze(runs, result_dir); figures(rows, calibration, risk, positives, figure_dir)
    print(json.dumps({"completed_runs": len(runs), "expected_runs": len(seeds)*len(fractions)*len(METHODS), "result_dir": str(result_dir)}, indent=2))

if __name__ == "__main__": main()
