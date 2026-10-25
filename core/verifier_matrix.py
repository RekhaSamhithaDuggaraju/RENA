"""Head-to-head verifier evaluation for RENA research experiments.

The matrix compares four intentionally different decision strategies:
- RENA-X: calibrated retrain-relative decision + forget/utility gates.
- MIA: membership-AUC residual criterion calibrated to retrain references.
- KS: Kolmogorov-Smirnov loss-distribution compatibility with retrain references.
- Simple average: uncalibrated mean of the six observable RENA axes.
- ULTRA-style: uncalibrated worst-case minimum of the six axes.

The ULTRA entry is explicitly a protocol baseline, not an implementation of
ULTRA's published code or tolerances.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

from core.advanced_audit import run_advanced_audit, _model_metrics
from core.model import per_sample_loss


def _verdict_from_audit(audit):
    return audit.calibration.get("certification_verdict") if audit.calibration else "FAIL"


def _raw_axes(audit):
    return [
        float(audit.behavioral_equivalence),
        float(audit.membership_indistinguishability),
        float(audit.representation_neutrality),
        float(audit.retain_consistency),
        float(audit.irreversibility),
        float(audit.efficiency),
    ]


def _mia_rule(candidate_metrics, reference_metrics):
    ref = np.asarray([m["auc"] for m in reference_metrics], dtype=float)
    cand = float(candidate_metrics["auc"])
    # Higher membership AUC means stronger distinguishability. A candidate is
    # acceptable when it is no worse than the empirical 95th percentile of
    # retain-only retrains, with a finite-sample floor.
    threshold = float(np.quantile(ref, 0.95)) if len(ref) >= 2 else float(np.max(ref))
    return {
        "score": float(np.clip(1.0 - max(0.0, cand - np.median(ref)) / 0.20, 0.0, 1.0)),
        "threshold": threshold,
        "candidate_auc": cand,
        "verdict": "PASS" if cand <= threshold else "FAIL",
    }


def _ks_rule(model, split, reference_models):
    candidate_losses = per_sample_loss(model, split.X_forget, split.y_forget)
    ref_losses = np.concatenate([
        per_sample_loss(ref, split.X_forget, split.y_forget) for ref in reference_models
    ])
    stat, p = ks_2samp(candidate_losses, ref_losses, alternative="two-sided", mode="auto")
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "verdict": "PASS" if p >= 0.05 else "FAIL",
    }


def evaluate_candidate(candidate_name, model, base_model, split, reference_models,
                       reference_metrics=None, decision_thresholds=None):
    """Evaluate one candidate with all comparator verifiers."""
    refs_metrics = reference_metrics or [_model_metrics(m, split) for m in reference_models]
    audit = run_advanced_audit(
        model, reference_models[0], base_model, split, 1.0, 1.0,
        reference_models=reference_models,
        reference_metrics=refs_metrics,
        candidate_metrics=_model_metrics(model, split),
        decision_thresholds=decision_thresholds,
    )
    rena_verdict = _verdict_from_audit(audit)
    axes = _raw_axes(audit)
    avg_score = float(np.mean(axes))
    avg_verdict = "PASS" if avg_score >= 0.70 else ("REVIEW" if avg_score >= 0.50 else "FAIL")
    ultra_score = float(np.min(axes))
    ultra_verdict = "PASS" if ultra_score >= 0.70 else ("REVIEW" if ultra_score >= 0.50 else "FAIL")
    candidate_metrics = _model_metrics(model, split)
    mia = _mia_rule(candidate_metrics, refs_metrics)
    ks = _ks_rule(model, split, reference_models)
    return {
        "candidate": candidate_name,
        "rena_x": {
            "score": float(audit.correctness_worst_case_score),
            "verdict": rena_verdict,
            "forget_progress": float(audit.forget_progress),
            "retain_utility_ratio": float(audit.retain_utility_ratio),
        },
        "mia": mia,
        "ks": ks,
        "simple_average": {"score": avg_score, "verdict": avg_verdict},
        "ultra_style": {
            "score": ultra_score,
            "verdict": ultra_verdict,
            "reproduction_status": "protocol baseline; not exact ULTRA reproduction",
        },
        "raw_axes": {
            "behavioral_equivalence": float(audit.behavioral_equivalence),
            "membership_indistinguishability": float(audit.membership_indistinguishability),
            "representation_neutrality": float(audit.representation_neutrality),
            "retain_consistency": float(audit.retain_consistency),
            "irreversibility": float(audit.irreversibility),
            "efficiency": float(audit.efficiency),
        },
    }


def summarize_matrix(rows, labels):
    """Compute detection/FPR/FNR/abstention for each verifier.

    ``labels`` is a dict candidate -> True for genuine retain-only retraining
    and False for known bad controls. The controls are experimental ground truth
    labels, not labels inferred from any verifier.
    """
    methods = ["rena_x", "mia", "ks", "simple_average", "ultra_style"]
    out = {}
    for method in methods:
        accepted = []
        for row in rows:
            verdict = row[method]["verdict"]
            accepted.append(verdict == "PASS")
        good = [i for i,r in enumerate(rows) if labels[r["candidate"]]]
        bad = [i for i,r in enumerate(rows) if not labels[r["candidate"]]]
        false_positive = sum(not accepted[i] for i in good) / max(len(good), 1)
        false_negative = sum(accepted[i] for i in bad) / max(len(bad), 1)
        detection = sum(row_method_verdict(rows[i], method) == "FAIL" for i in bad) / max(len(bad), 1)
        abstention = sum(row_method_verdict(rows[i], method) == "REVIEW" for i in range(len(rows))) / max(len(rows), 1)
        out[method] = {
            "n_good": len(good), "n_bad": len(bad),
            "false_positive_rate": float(false_positive),
            "false_negative_rate": float(false_negative),
            "bad_control_detection_rate": float(detection),
            "abstention_rate": float(abstention),
        }
    return out


def row_method_verdict(row, method):
    return str(row[method]["verdict"])
