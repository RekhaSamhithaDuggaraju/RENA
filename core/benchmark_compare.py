"""Transparent verifier comparisons for RENA research experiments.

ULTRA is a published 2026 certification framework with six requirements and a
worst-case aggregation principle. RENA does not claim to reproduce ULTRA's full
experimental implementation here. This module provides a clearly-labelled
ULTRA-style baseline using the same observable RENA evidence dimensions, while
RENA's proposed distinction is the held-out empirical retraining calibration.
"""
from __future__ import annotations
import numpy as np


def ultra_style_baseline(axes: dict) -> dict:
    """Compute a transparent ULTRA-style worst-case baseline from normalized axes.

    This is a protocol baseline, not a reproduction of the ULTRA paper's exact
    implementation/tolerances. The caller must provide normalized [0,1] axes.
    """
    names=("behavioral_equivalence","membership_indistinguishability",
           "representation_neutrality","retain_consistency","irreversibility","efficiency")
    vals={k:float(np.clip(axes.get(k,0.0),0,1)) for k in names}
    return {"baseline":"ULTRA-style_worst_case","axes":vals,
            "score":float(min(vals.values())),
            "aggregation":"minimum across six normalized requirements",
            "reproduction_status":"protocol baseline; not a claim of exact ULTRA reproduction"}


def compare_verifiers(rena_result: dict) -> dict:
    audit=rena_result.get("diagnosis",{}).get("advanced_audit",{}) if "diagnosis" in rena_result else rena_result
    axes=calibration_axes=audit.get("calibration",{}).get("uncalibrated_evidence_profile",audit.get("evidence_profile",audit.get("axes",{})))
    raw=ultra_style_baseline(axes)
    calibration=rena_result.get("reference_calibration") or audit.get("calibration") or {}
    calibrated_score=float(audit.get("certification_score",audit.get("worst_case_score",0.0)))
    return {"rena_calibrated":{"score":calibrated_score,"basis":"held-out empirical retraining distribution"},
            "ultra_style":raw,
            "difference":float(calibrated_score-raw["score"]),
            "interpretation":"The baseline shares the worst-case principle; the research distinction is reference-distribution calibration and abstention, not the six-axis taxonomy."}
