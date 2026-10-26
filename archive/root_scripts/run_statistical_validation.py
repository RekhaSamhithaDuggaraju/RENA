"""Development-threshold + held-out statistical validation for RENA-X.

This is intentionally a research runner, not part of the web API. Thresholds
are selected only from development candidates on seed 41, then frozen and
applied to the held-out seed-42 comparison and adversarial matrix.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np

from core.dataset import build_split
from core.model import train_base_model, clone_model
from core.unlearning import ground_truth_retrain
from core.adversarial_lab import partial_forgetting, selective_forgetting, over_unlearning, verifier_targeted_deception, run_adversarial_suite
from core.pipeline import run_audit, _build_reference_ensemble, ALGORITHMS
from core.advanced_audit import _model_metrics
from core.threshold_calibration import select_development_thresholds
from core.verifier_matrix import evaluate_candidate, summarize_matrix
from core.verifier_attack import rena_x_targeted_deception

OUT = Path("results/statistical_validation_v12.json")


def build_original(split):
    X = np.vstack([split.X_retain, split.X_forget])
    y = np.concatenate([split.y_retain, split.y_forget])
    return train_base_model(X, y)


def row_from_audit(label, audit):
    cal = audit.calibration or {}
    return {
        "label": label,
        "correctness_min": float(audit.correctness_worst_case_score),
        "forget_progress": float(audit.forget_progress),
        "retain_utility_ratio": float(audit.retain_utility_ratio),
    }


def main():
    t0 = time.time()
    # ---------------- development ----------------
    dev = build_split(forget_fraction=0.1, random_state=41)
    dev_original = build_original(dev)
    dev_refs, _ = _build_reference_ensemble(dev, 10, 51000)
    dev_ref_metrics = [_model_metrics(m, dev) for m in dev_refs]

    dev_candidates = [("ground_truth_retraining", ground_truth_retrain(dev.X_retain, dev.y_retain, random_state=70001), "good")]
    no_op = clone_model(dev_original)
    partial25 = partial_forgetting(dev_original, dev, 0.25)
    selective = selective_forgetting(dev_original, dev)
    over = over_unlearning(dev_original, dev)
    targeted = verifier_targeted_deception(dev_original, dev_refs[0], dev)
    dev_candidates += [
        ("no_op", no_op, "bad"),
        ("partial_25", partial25, "bad"),
        ("selective", selective, "bad"),
        ("over_unlearning", over, "bad"),
        ("verifier_targeted", targeted, "bad"),
    ]

    dev_rows=[]
    for name, model, label in dev_candidates:
        audit = __import__('core.advanced_audit', fromlist=['run_advanced_audit']).run_advanced_audit(
            model, dev_refs[0], dev_original, dev, 1.0, 1.0,
            reference_models=dev_refs, reference_metrics=dev_ref_metrics,
            candidate_metrics=_model_metrics(model, dev),
        )
        dev_rows.append({"candidate":name, **row_from_audit(label, audit)})

    threshold_result = select_development_thresholds(dev_rows, target_false_pass_rate=0.05)
    thresholds = threshold_result["thresholds"]

    # ---------------- held-out comparison ----------------
    held = build_split(forget_fraction=0.1, random_state=42)
    held_original = build_original(held)
    held_refs, _ = _build_reference_ensemble(held, 20, 52000)
    sample_ids = np.arange(len(held.X_retain) + len(held.X_forget))
    comparison_results = {}
    for algo in ALGORITHMS:
        comparison_results[algo] = run_audit(
            algo, held, held_original, sample_ids=sample_ids, reference_count=20,
            reference_seed=52000, split_seed=42, reference_models=held_refs,
            decision_thresholds=thresholds,
        )
    comparison = {
        "dataset_summary": held.summary(),
        "reference_ensemble": {"count": len(held_refs), "seed": 52000},
        "results": comparison_results,
    }

    # ---------------- held-out adversarial matrix ----------------
    adaptive = rena_x_targeted_deception(
        held_original, held_refs, held, reference_metrics=[_model_metrics(m, held) for m in held_refs], budget=12
    )
    attack_models = {
        "retrain": ground_truth_retrain(held.X_retain, held.y_retain, random_state=70002),
        "no_op": clone_model(held_original),
        "partial_25": partial_forgetting(held_original, held, 0.25),
        "partial_50": partial_forgetting(held_original, held, 0.50),
        "partial_75": partial_forgetting(held_original, held, 0.75),
        "selective": selective_forgetting(held_original, held),
        "over_unlearning": over_unlearning(held_original, held),
        "verifier_targeted": verifier_targeted_deception(held_original, held_refs[0], held),
        "rena_x_targeted": adaptive["candidate_model"],
    }
    attack_rows=[]
    for name, model in attack_models.items():
        attack_rows.append(evaluate_candidate(
            name, model, held_original, held, held_refs,
            reference_metrics=[_model_metrics(m, held) for m in held_refs],
            decision_thresholds=thresholds,
        ))
    labels={name:(name=="retrain") for name in attack_models}
    matrix_summary=summarize_matrix(attack_rows, labels)

    result={
        "protocol":"development_thresholds_seed41_then_frozen_heldout_seed42",
        "development": {"references":10, "seed":41, "candidates":dev_rows, **threshold_result},
        "heldout": {
            "references":20, "seed":42, "comparison":comparison,
            "adversarial": {"attack": {k:v for k,v in adaptive.items() if k != "candidate_model"}, "matrix":attack_rows, "summary":matrix_summary},
        },
        "runtime_seconds": round(time.time()-t0,3),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps({
        "runtime_seconds":result["runtime_seconds"],
        "thresholds":thresholds,
        "dev_false_pass":threshold_result["development_false_pass_rate"],
        "dev_true_pass":threshold_result["development_true_pass_rate"],
        "heldout_references":20,
        "adaptive_verdict":adaptive["selected"]["verdict"],
        "adaptive_residual":adaptive["selected"]["residual_forget_loss_ratio"],
        "matrix_summary":matrix_summary,
        "output":str(OUT),
    }, indent=2))

if __name__ == "__main__":
    main()
