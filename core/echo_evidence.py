"""Fast, measured evidence proxies used inside ECHO candidate selection."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from core.model import evaluate, per_sample_loss
from core.recovery_stress_test import membership_recoverability


def parameter_distance(model, reference) -> float:
    return float(np.sqrt(sum(np.sum((a - b) ** 2) for a, b in zip(model.W, reference.W)) +
                         sum(np.sum((a - b) ** 2) for a, b in zip(model.b, reference.b))))


def fast_evidence_state(model, split, original_model, baseline_retain_accuracy: float) -> dict:
    """Cheap proxies: no bootstrap or learned attacker inside candidate search.

    The four fields map to existing RENA verifier concepts; full verifiers run
    only after commits/checkpoints in the experiment runner.
    """
    loss_f = per_sample_loss(model, split.X_forget, split.y_forget)
    loss_r = per_sample_loss(model, split.X_retain, split.y_retain)
    n = min(len(loss_f), len(split.X_test))
    loss_t = per_sample_loss(model, split.X_test[:n], split.y_test[:n])
    labels = np.r_[np.ones(n), np.zeros(n)]
    auc = float(roc_auc_score(labels, -np.r_[loss_f[:n], loss_t]))
    recover = membership_recoverability(model, split.X_forget, split.y_forget, split.X_test, split.y_test)
    conf_f = float(np.mean(model.predict_proba(split.X_forget).max(axis=1)))
    conf_t = float(np.mean(model.predict_proba(split.X_test[:n]).max(axis=1)))
    retain = evaluate(model, split.X_retain, split.y_retain)
    test = evaluate(model, split.X_test, split.y_test)
    # Higher values are better for the four forgetting proxies. These are not
    # combined by a weighted sum; ECHO compares them component-wise.
    return {
        "forget_loss_signal": float(np.mean(loss_f) - np.mean(loss_r)),
        "retain_loss_signal": float(np.mean(loss_r)),
        "confidence_signal": float(1.0 - min(1.0, abs(conf_f - conf_t) / 0.05)),
        "membership_signal": float(1.0 - recover["normalized_recoverability"]),
        "statistical_signal": float(abs(np.mean(loss_f) - np.mean(loss_r))),
        "recoverability_signal": float(recover["normalized_recoverability"]),
        "retain_accuracy": retain["accuracy"], "test_accuracy": test["accuracy"],
        "retain_degradation": float(max(0.0, baseline_retain_accuracy - retain["accuracy"])),
        "membership_auc_proxy": auc,
    }
