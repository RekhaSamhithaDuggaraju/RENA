"""Supplementary retained-data calibrated prediction-set exclusion diagnostic."""
from __future__ import annotations

import numpy as np


def run_conformal_diagnostic(model, split, alpha: float = 0.10) -> dict:
    """Uses retain data to calibrate a simple probability prediction set.

    This is an empirical diagnostic, not a formal unlearning guarantee.
    """
    retain_probs = model.predict_proba(split.X_retain)
    calibration_scores = 1.0 - retain_probs[np.arange(len(split.y_retain)), split.y_retain]
    threshold = float(np.quantile(calibration_scores, min(1.0, 1.0 - alpha), method="higher"))

    def exclusion(X, y):
        probs = model.predict_proba(X)
        included = (1.0 - probs[np.arange(len(y)), y]) <= threshold
        return float(1.0 - np.mean(included))

    forget_exclusion = exclusion(split.X_forget, split.y_forget)
    retain_exclusion = exclusion(split.X_retain, split.y_retain)
    return {"alpha": alpha, "calibration_source": "retain_set", "calibration_threshold": threshold,
            "forget_label_exclusion": forget_exclusion, "retain_label_exclusion": retain_exclusion,
            "exclusion_gap": float(forget_exclusion - retain_exclusion),
            "limitation": "Supplementary empirical prediction-set diagnostic; it does not prove machine unlearning."}
