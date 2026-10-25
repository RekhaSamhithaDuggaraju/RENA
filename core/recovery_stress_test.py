"""Image-domain recovery stress testing using loss-based membership signals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from core.model import per_sample_loss


CONDITIONS = ("direct", "gaussian_noise", "random_mask", "spatial_shift", "contrast_brightness")


def transform_images(X: np.ndarray, condition: str, random_state: int = 42) -> np.ndarray:
    """Apply deterministic perturbations to standardized 8x8 digit feature vectors."""
    rng = np.random.RandomState(random_state)
    out = X.copy()
    if condition == "direct":
        return out
    if condition == "gaussian_noise":
        return out + rng.normal(0, 0.25, out.shape)
    if condition == "random_mask":
        return out * (rng.rand(*out.shape) > 0.20)
    images = out.reshape((-1, 8, 8))
    if condition == "spatial_shift":
        return np.roll(images, shift=(1, 1), axis=(1, 2)).reshape(out.shape)
    if condition == "contrast_brightness":
        # Features are standardized; this is an affine image-intensity proxy.
        return (images * 0.85 + 0.10).reshape(out.shape)
    raise ValueError(f"Unknown recovery condition: {condition}")


def membership_recoverability(model, X_member, y_member, X_nonmember, y_nonmember) -> dict:
    """Loss is the attack score; normalized recoverability is 2*abs(AUC-.5)."""
    n = min(len(X_member), len(X_nonmember))
    scores = -np.concatenate([per_sample_loss(model, X_member[:n], y_member[:n]),
                              per_sample_loss(model, X_nonmember[:n], y_nonmember[:n])])
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    auc = float(roc_auc_score(labels, scores))
    return {"auc": auc, "normalized_recoverability": float(2 * abs(auc - 0.5)), "n_per_group": int(n)}


def run_recovery_stress_test(model, split, random_state: int = 42) -> list[dict]:
    records = []
    for condition in CONDITIONS:
        Xf = transform_images(split.X_forget, condition, random_state)
        Xt = transform_images(split.X_test, condition, random_state + 1)
        Xr = transform_images(split.X_retain, condition, random_state + 2)
        for control, Xm, ym in (("forget_vs_held_out", Xf, split.y_forget),
                                ("retain_vs_held_out_control", Xr, split.y_retain)):
            metric = membership_recoverability(model, Xm, ym, Xt, split.y_test)
            records.append({"condition": condition, "comparison": control, **metric})
    return records
