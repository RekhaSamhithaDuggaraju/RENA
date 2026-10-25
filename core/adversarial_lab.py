"""Model-level adversarial controls for RENA.

These experiments modify candidate models rather than corrupting verifier scores.
They are research controls, not claims that an attacker can defeat physical erasure.
"""
from __future__ import annotations
import numpy as np
from core.model import clone_model, evaluate, per_sample_loss
from core.advanced_audit import run_advanced_audit, _model_metrics
from core.verifier_attack import rena_x_targeted_deception


def partial_forgetting(base_model, split, fraction: float):
    """Construct a partial-forgetting control by interpolating *loss*, not accuracy.

    Accuracy is too coarse on small datasets: several different forgetting
    budgets can produce the same discrete accuracy. We therefore target the
    original-to-retrain forget-loss gap, which gives genuinely different 25/50/75%
    controls while leaving the retain set untouched.
    """
    model=clone_model(base_model)
    from core.unlearning import ground_truth_retrain
    retrain=ground_truth_retrain(split.X_retain,split.y_retain,random_state=49157)
    original_loss=float(np.mean(per_sample_loss(base_model,split.X_forget,split.y_forget)))
    retrain_loss=float(np.mean(per_sample_loss(retrain,split.X_forget,split.y_forget)))
    target=original_loss+(retrain_loss-original_loss)*float(fraction)
    best=clone_model(model); best_gap=abs(original_loss-target)
    for _ in range(500):
        current_loss=float(np.mean(per_sample_loss(model,split.X_forget,split.y_forget)))
        gap=abs(current_loss-target)
        if gap < best_gap:
            best=clone_model(model); best_gap=gap
        if (retrain_loss>=original_loss and current_loss>=target) or (retrain_loss<original_loss and current_loss<=target):
            break
        model.gradient_step(split.X_forget,split.y_forget,lr=0.08,ascent=True)
    return best


def selective_forgetting(base_model, split):
    """Forget easy/low-loss examples while preserving harder forget examples."""
    model=clone_model(base_model)
    losses=per_sample_loss(model,split.X_forget,split.y_forget)
    n=max(1,len(losses)//2)
    idx=np.argsort(losses)[:n]
    model.gradient_step(split.X_forget[idx],split.y_forget[idx],lr=0.045,ascent=True)
    return model


def over_unlearning(base_model, split):
    """Drive forget loss aggressively while deliberately damaging retain utility."""
    model=clone_model(base_model)
    for _ in range(70):
        model.gradient_step(split.X_forget,split.y_forget,lr=0.08,ascent=True)
    return model


def verifier_targeted_deception(base_model, reference_model, split, steps=30, lr=0.01, retain_weight=0.5):
    """Model-level deception surrogate.

    The candidate is optimized to match the retain-only reference's outputs on
    the forget set while preserving the original model's outputs on retain data.
    This targets the behavioral/utility witnesses at the *model* level. The
    resulting model is still evaluated for residual membership and recovery;
    it is intentionally a controlled adversarial surrogate, not a perfect
    bilevel optimizer of the complete verifier.
    """
    model=clone_model(base_model)
    target_forget=reference_model.predict_proba(split.X_forget)
    target_retain=base_model.predict_proba(split.X_retain)
    for _ in range(steps):
        model._soft_target_step(split.X_forget,target_forget,lr=lr)
        model._soft_target_step(split.X_retain,target_retain,lr=lr*retain_weight)
    return model


def run_adversarial_suite(base_model, reference_models, split, decision_thresholds=None):
    """Evaluate genuine model-level controls against a fixed reference ensemble."""
    ref=reference_models[0]
    ref_metrics=[_model_metrics(m,split) for m in reference_models]
    adaptive = rena_x_targeted_deception(base_model, reference_models, split, reference_metrics=ref_metrics, budget=12)
    controls={
        "partial_25": partial_forgetting(base_model,split,0.25),
        "partial_50": partial_forgetting(base_model,split,0.50),
        "partial_75": partial_forgetting(base_model,split,0.75),
        "selective": selective_forgetting(base_model,split),
        "over_unlearning": over_unlearning(base_model,split),
        "verifier_targeted": verifier_targeted_deception(base_model,ref,split),
        "rena_x_targeted": adaptive["candidate_model"],
    }
    rows=[]
    adaptive_meta=adaptive.copy(); adaptive_meta.pop("candidate_model",None)
    for name,model in controls.items():
        audit=run_advanced_audit(model,ref,base_model,split,1.0,1.0,
                                 reference_models=reference_models,
                                 reference_metrics=ref_metrics,
                                 candidate_metrics=_model_metrics(model,split),
                                 decision_thresholds=decision_thresholds)
        rows.append({
            "attack":name,
            "rena_x":audit.worst_case_score,
            "verdict":audit.calibration.get("certification_verdict") if audit.calibration else None,
            "axes":audit.to_dict(),
            "retain_accuracy":evaluate(model,split.X_retain,split.y_retain)["accuracy"],
            "forget_accuracy":evaluate(model,split.X_forget,split.y_forget)["accuracy"],
            "representation_membership_auc":audit.representation_auc,
        })
    return {"controls":rows,"reference_count":len(reference_models),"adaptive_attack":adaptive_meta,
            "interpretation":"Model-level adversarial controls modify the candidate itself; evidence-score corruption is intentionally excluded from this table. The RENA-X-targeted row uses black-box queries to the complete decision rule."}
