"""Development-only selection of RENA-X decision thresholds.

Thresholds are selected from labelled development candidates, then frozen before
held-out evaluation. This module never looks at the held-out candidate set.
"""
from __future__ import annotations

from itertools import product


def _is_pass(row, t):
    return (
        float(row["correctness_min"]) >= t["certification_alpha"]
        and float(row["forget_progress"]) >= t["forget_pass"]
        and float(row["retain_utility_ratio"]) >= t["retain_pass"]
    )


def _is_review_or_pass(row, t):
    return (
        float(row["correctness_min"]) >= t["review_alpha"]
        and float(row["forget_progress"]) >= t["forget_review"]
        and float(row["retain_utility_ratio"]) >= t["retain_review"]
    )


def select_development_thresholds(rows, *, target_false_pass_rate=0.05):
    """Select thresholds using only development-labelled candidates.

    ``good`` rows are legitimate independently retrained candidates; ``bad``
    rows are no-op/partial/selective/over-unlearning/adversarial controls.
    The primary constraint is the false-certification rate on bad controls.
    Among feasible settings, choose the setting with the highest genuine-retrain
    PASS rate, then highest review-or-pass coverage, then the least permissive
    thresholds. The returned thresholds are intended to be frozen for a
    separate held-out experiment.
    """
    rows = list(rows)
    good = [r for r in rows if r.get("label") == "good"]
    bad = [r for r in rows if r.get("label") == "bad"]
    if not good or not bad:
        raise ValueError("development calibration requires both good and bad candidates")

    grid = {
        "certification_alpha": (0.05, 0.075, 0.10, 0.125, 0.15),
        "review_alpha": (0.025, 0.05, 0.075, 0.10),
        "forget_pass": (0.30, 0.35, 0.40, 0.45, 0.50),
        "forget_review": (0.15, 0.20, 0.25, 0.30),
        "retain_pass": (0.85, 0.90, 0.925, 0.95),
        "retain_review": (0.75, 0.80, 0.85, 0.875),
    }

    best = None
    for vals in product(*grid.values()):
        t = dict(zip(grid, vals))
        if t["review_alpha"] > t["certification_alpha"]:
            continue
        if t["forget_review"] > t["forget_pass"]:
            continue
        if t["retain_review"] > t["retain_pass"]:
            continue
        bad_pass = sum(_is_pass(r, t) for r in bad) / len(bad)
        if bad_pass > target_false_pass_rate:
            continue
        good_pass = sum(_is_pass(r, t) for r in good) / len(good)
        good_coverage = sum(_is_review_or_pass(r, t) for r in good) / len(good)
        # Prefer statistical protection first, then acceptance of genuine retrains.
        score = (good_pass, good_coverage, -t["forget_pass"], -t["retain_pass"], -t["certification_alpha"])
        if best is None or score > best[0]:
            best = (score, t, bad_pass, good_pass, good_coverage)

    if best is None:
        raise ValueError("no threshold setting satisfied the target false-pass rate")

    _, thresholds, bad_pass, good_pass, good_coverage = best
    return {
        "thresholds": thresholds,
        "target_false_pass_rate": float(target_false_pass_rate),
        "development_false_pass_rate": float(bad_pass),
        "development_true_pass_rate": float(good_pass),
        "development_coverage": float(good_coverage),
        "n_good": len(good),
        "n_bad": len(bad),
        "selection_rule": "minimize false certification subject to target, then maximize genuine retrain PASS rate and coverage",
        "frozen_before_held_out_evaluation": True,
    }
