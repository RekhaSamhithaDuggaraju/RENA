"""Modern, reproducible unlearning baselines implemented for RENA.

These are compact adaptations for the NumPy SimpleMLP research workload. They
are deliberately named with the underlying research family rather than claiming
bit-for-bit reproduction of papers that use different architectures/datasets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from core.model import clone_model


@dataclass
class ModernConfig:
    forget_epochs: int = 12
    recovery_epochs: int = 4
    learning_rate: float = 0.25
    recovery_rate: float = 0.01
    mask_quantile: float = 0.70
    noise_scale: float = 0.08
    retain_weight: float = 1.0


def _masked_gradient_step(model, X, y, masks, lr, ascent=False):
    dw, db = model._backward(X, y)
    sign = 1.0 if ascent else -1.0
    w_masks, b_masks = masks
    for i in range(len(model.W)):
        model.W[i] += sign * lr * dw[i] * w_masks[i]
        model.b[i] += sign * lr * db[i] * b_masks[i]


def negative_gradient_unlearn(base_model, X_forget, y_forget, X_retain, y_retain,
                              config: ModernConfig | None = None):
    """NegGrad-style selective forgetting followed by retain recovery."""
    cfg = config or ModernConfig()
    model = clone_model(base_model)
    for _ in range(cfg.forget_epochs):
        model.gradient_step(X_forget, y_forget, lr=cfg.learning_rate, ascent=True)
    for _ in range(cfg.recovery_epochs):
        model.gradient_step(X_retain, y_retain, lr=cfg.recovery_rate, ascent=False)
    return model, {"family": "NegGrad-style", "config": asdict(cfg), "adaptation": "full-parameter negative gradient"}


def salun_unlearn(base_model, X_forget, y_forget, X_retain, y_retain,
                  config: ModernConfig | None = None):
    """SalUn-inspired selective amnesia using high-magnitude forget gradients."""
    cfg = config or ModernConfig()
    model = clone_model(base_model)
    dw, db = model._backward(X_forget, y_forget)
    all_abs = np.concatenate([np.abs(x).ravel() for x in dw + db])
    threshold = float(np.quantile(all_abs, cfg.mask_quantile)) if len(all_abs) else 0.0
    w_mask = [(np.abs(x) >= threshold).astype(float) for x in dw]
    b_mask = [(np.abs(x) >= threshold).astype(float) for x in db]
    mask = w_mask + b_mask
    for _ in range(cfg.forget_epochs):
        _masked_gradient_step(model, X_forget, y_forget, (w_mask, b_mask), cfg.learning_rate, ascent=True)
    for _ in range(cfg.recovery_epochs):
        model.gradient_step(X_retain, y_retain, lr=cfg.recovery_rate, ascent=False)
    density = float(np.mean(np.concatenate([x.ravel() for x in mask])))
    return model, {"family": "SalUn-inspired", "config": asdict(cfg), "mask_threshold": threshold, "mask_density": density,
                   "adaptation": "top-magnitude forget-gradient coordinates"}


def unsir_style_unlearn(base_model, X_forget, y_forget, X_retain, y_retain,
                        config: ModernConfig | None = None):
    """UNSIR-inspired synthetic-noise forgetting plus retain recovery.

    The original family is broader than this lightweight NumPy implementation;
    this version creates class-aware perturbations from forget-set statistics and
    optimizes against those pseudo-forget examples, making the adaptation explicit.
    """
    cfg = config or ModernConfig(forget_epochs=12, recovery_epochs=6, learning_rate=0.05, recovery_rate=0.01, mask_quantile=0.70, noise_scale=0.08)
    model = clone_model(base_model)
    rng = np.random.RandomState(9173)
    center = np.mean(X_forget, axis=0, keepdims=True)
    scale = np.std(X_forget, axis=0, keepdims=True) + 1e-6
    pseudo = center + rng.normal(size=X_forget.shape) * scale * cfg.noise_scale
    # Keep pseudo labels tied to the forget request so the update has a real loss.
    pseudo_y = y_forget.copy()
    for _ in range(cfg.forget_epochs):
        model.gradient_step(pseudo, pseudo_y, lr=cfg.learning_rate, ascent=True)
    for _ in range(cfg.recovery_epochs):
        model.gradient_step(X_retain, y_retain, lr=cfg.recovery_rate, ascent=False)
    return model, {"family": "UNSIR-inspired", "config": asdict(cfg), "noise_scale": cfg.noise_scale,
                   "pseudo_forget_count": int(len(pseudo)), "adaptation": "forget-statistic synthetic perturbations"}


def scrub_style_unlearn(base_model, X_forget, y_forget, X_retain, y_retain,
                        config: ModernConfig | None = None):
    """SCRUB-inspired alternating forget suppression and retain distillation."""
    cfg = config or ModernConfig()
    teacher = clone_model(base_model)
    model = clone_model(base_model)
    teacher_probs = teacher.predict_proba(X_retain)
    for _ in range(cfg.forget_epochs):
        model.gradient_step(X_forget, y_forget, lr=cfg.learning_rate, ascent=True)
        # Distillation toward the original model preserves utility without using forget data.
        model._soft_target_step(X_retain, teacher_probs, lr=cfg.recovery_rate)
    return model, {"family": "SCRUB-inspired", "config": asdict(cfg), "adaptation": "alternating forget ascent + retain soft-target recovery"}
