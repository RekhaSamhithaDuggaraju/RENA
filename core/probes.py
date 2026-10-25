"""Query-only consistency probes (black-box compatible).

Exact-input output overlays ("output patches") make a deployed model emit
retrain-like probabilities on the literal forget inputs while the underlying
function is unchanged.  An honest neural network is *smooth*: its loss on x and
on a tiny perturbation of x is almost identical.  An exact-input overlay is
discontinuous.  ``discontinuity_profile`` measures that discontinuity with
queries only, at several perturbation scales, so it needs no weight access.

Known limit (tested in ``core.research_suite``): an adaptive overlay that also
covers a neighbourhood of each forget input hides from small-scale probes, so
multi-scale probing raises the attacker's cost but is not a proof of erasure.
"""
from __future__ import annotations

import numpy as np
from core.model import per_sample_loss

DEFAULT_SIGMAS = (0.02, 0.10, 0.30, 0.60)  # perturbation std, in units of per-feature std
DEFAULT_VIEWS = 4


def jittered_views(X, sigma: float, views: int = DEFAULT_VIEWS, seed: int = 11):
    rng = np.random.RandomState(seed)
    sd = np.asarray(X).std(axis=0, keepdims=True) + 1e-6
    return [np.asarray(X) + rng.randn(*np.asarray(X).shape) * sd * sigma for _ in range(views)]


def discontinuity_profile(model, X, y, sigmas=DEFAULT_SIGMAS, views: int = DEFAULT_VIEWS):
    """Mean |loss(x) - loss(x+eps)| per scale.  Smooth models score near 0."""
    base = per_sample_loss(model, X, y)
    out = {}
    for s in sigmas:
        diffs = [np.abs(base - per_sample_loss(model, V, y)) for V in jittered_views(X, s, views)]
        out[float(s)] = float(np.mean(diffs))
    return out


def discontinuity_score(profile: dict) -> float:
    """Scale-normalised aggregate: the worst per-scale discontinuity ratio is
    taken against the sigma-proportional smooth baseline (loss change grows
    ~linearly with sigma for smooth nets), so large scales are not penalised."""
    return float(max(v / max(s, 1e-6) for s, v in profile.items()))
