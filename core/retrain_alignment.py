"""Independent empirical alignment proxies for retraining reference models.

These functions deliberately do not import verification or consensus.  RRA is
an output-distance proxy, not a proof that a model has unlearned data.
"""
from __future__ import annotations

import numpy as np

EPSILON = 1e-12


def stable_probabilities(probabilities: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    """Clip and renormalize a 2-D probability matrix deterministically."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes >= 2)")
    if not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be finite")
    p = np.clip(p, epsilon, None)
    return p / np.sum(p, axis=1, keepdims=True)


def js_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    """Per-example Jensen--Shannon divergence in nats (bounded and symmetric)."""
    p, q = stable_probabilities(p, epsilon), stable_probabilities(q, epsilon)
    if p.shape != q.shape:
        raise ValueError("probability arrays must have the same shape")
    midpoint = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(midpoint)), axis=1)
    kl_qm = np.sum(q * (np.log(q) - np.log(midpoint)), axis=1)
    return 0.5 * (kl_pm + kl_qm)


def relative_retrain_alignment(original_model, audited_model, retrained_model, X_probe: np.ndarray,
                               epsilon: float = EPSILON) -> dict:
    """Compute RRA = D(audited, original) / [D(audited, original) + D(audited, retrained) + eps]."""
    p_original = stable_probabilities(original_model.predict_proba(X_probe), epsilon)
    p_audited = stable_probabilities(audited_model.predict_proba(X_probe), epsilon)
    p_retrained = stable_probabilities(retrained_model.predict_proba(X_probe), epsilon)
    distance_audit_original = float(np.mean(js_divergence(p_audited, p_original, epsilon)))
    distance_audit_retrained = float(np.mean(js_divergence(p_audited, p_retrained, epsilon)))
    rra = distance_audit_original / (distance_audit_original + distance_audit_retrained + epsilon)
    disagreement = float(np.mean(np.argmax(p_audited, axis=1) != np.argmax(p_retrained, axis=1)))
    return {
        "rra": float(np.clip(rra, 0.0, 1.0)),
        "distance_audit_original": distance_audit_original,
        "distance_audit_retrained": distance_audit_retrained,
        "prediction_disagreement_rate": disagreement,
        "n_probe": int(len(X_probe)),
        "distance": "mean Jensen-Shannon divergence (nats)",
        "epsilon": epsilon,
    }
