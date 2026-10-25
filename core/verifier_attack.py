"""Black-box attacks against the *complete* RENA-X decision rule.

This module intentionally does not differentiate through RENA-X.  Instead, it
uses the verifier as an oracle: generate model candidates, audit each candidate
with the complete RENA-X path (including the forget/utility gates), and adapt
search toward candidates that are accepted while retaining forgotten behavior.

That makes the experiment a query-based / black-box adaptive attack. It is a
stronger threat model than the older behavioral surrogate, while remaining
honest about what is and is not optimized.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from core.model import clone_model, per_sample_loss
from core.advanced_audit import run_advanced_audit, _model_metrics


@dataclass
class AttackTrial:
    alpha: float
    steps: int
    lr: float
    retain_weight: float
    verdict: str
    forget_progress: float
    retain_utility_ratio: float
    rena_x: float
    representation_probe_auc: float
    residual_forget_loss_ratio: float


def _candidate_from_mixture(base_model, reference_model, split, *, alpha: float,
                            steps: int, lr: float, retain_weight: float):
    """Distill toward a mixture of reference and original behavior.

    alpha=0 approaches the retain-only reference on forget data; alpha=1
    preserves original forget behavior. Retain data always targets the original
    model, so the search cannot win merely by destroying utility.
    """
    model = clone_model(base_model)
    ref_forget = reference_model.predict_proba(split.X_forget)
    original_forget = base_model.predict_proba(split.X_forget)
    target_forget = (1.0 - alpha) * ref_forget + alpha * original_forget
    target_retain = base_model.predict_proba(split.X_retain)
    for _ in range(int(steps)):
        model._soft_target_step(split.X_forget, target_forget, lr=lr)
        model._soft_target_step(split.X_retain, target_retain, lr=lr * retain_weight)
    return model


def _verdict_rank(verdict: str) -> int:
    return {"PASS": 2, "REVIEW": 1, "FAIL": 0}.get(str(verdict), -1)


def rena_x_targeted_deception(base_model, reference_models, split, *,
                              reference_metrics=None, budget: int = 12,
                              seed: int = 42):
    """Search for an accepted candidate with maximal residual forget behavior.

    The search objective is deliberately verifier-facing:

      1. maximize RENA-X decision status (PASS > REVIEW > FAIL),
      2. among equally accepted candidates, maximize residual forget behavior,
      3. preserve retain utility,
      4. then maximize the calibrated RENA-X score.

    This is a black-box adaptive attack because every trial is evaluated by
    ``run_advanced_audit`` itself. It is not claimed to be a differentiable
    bilevel optimization of the verifier.
    """
    if not reference_models:
        raise ValueError("rena_x_targeted_deception requires a reference ensemble")
    ref = reference_models[0]
    ref_metrics = reference_metrics or [_model_metrics(m, split) for m in reference_models]
    original_forget_loss = float(np.mean(per_sample_loss(base_model, split.X_forget, split.y_forget)))
    reference_forget_loss = float(np.median([m["forget_loss"] for m in ref_metrics]))
    denom = abs(reference_forget_loss - original_forget_loss) + 1e-12

    # Deterministic coarse-to-fine candidate bank. The attack can be kept cheap
    # for unit tests by lowering budget, while the research run uses 12 queries.
    bank = [
        (0.10, 20, 0.010, 0.50),
        (0.20, 20, 0.010, 0.50),
        (0.30, 25, 0.010, 0.50),
        (0.40, 25, 0.010, 0.75),
        (0.50, 30, 0.010, 0.75),
        (0.60, 30, 0.010, 0.75),
        (0.70, 35, 0.008, 1.00),
        (0.80, 35, 0.008, 1.00),
        (0.90, 40, 0.008, 1.00),
        (0.35, 45, 0.006, 1.00),
        (0.55, 45, 0.006, 1.00),
        (0.75, 45, 0.006, 1.00),
    ]
    # Seed is retained in the manifest even though the deterministic bank is
    # fixed; this makes repeated experiment provenance explicit.
    _ = np.random.RandomState(seed)
    bank = bank[:max(1, int(budget))]

    trials = []
    best = None
    best_model = None
    for alpha, steps, lr, retain_weight in bank:
        candidate = _candidate_from_mixture(
            base_model, ref, split, alpha=alpha, steps=steps,
            lr=lr, retain_weight=retain_weight,
        )
        audit = run_advanced_audit(
            candidate, ref, base_model, split, 1.0, 1.0,
            reference_models=reference_models,
            reference_metrics=ref_metrics,
            candidate_metrics=_model_metrics(candidate, split),
        )
        verdict = audit.calibration.get("certification_verdict") if audit.calibration else "FAIL"
        candidate_forget_loss = float(np.mean(per_sample_loss(candidate, split.X_forget, split.y_forget)))
        # 0 means reference-like forgetting; 1 means original-like retention.
        residual = float(np.clip(
            abs(candidate_forget_loss - reference_forget_loss) / denom, 0.0, 1.0
        ))
        trial = AttackTrial(
            alpha=float(alpha), steps=int(steps), lr=float(lr),
            retain_weight=float(retain_weight), verdict=verdict,
            forget_progress=float(audit.forget_progress),
            retain_utility_ratio=float(audit.retain_utility_ratio),
            rena_x=float(audit.correctness_worst_case_score),
            representation_probe_auc=float(audit.representation_probe_auc),
            residual_forget_loss_ratio=residual,
        )
        trials.append(trial)
        key = (
            _verdict_rank(verdict),
            residual,
            float(audit.retain_utility_ratio),
            float(audit.correctness_worst_case_score),
        )
        if best is None or key > best[0]:
            best = (key, trial)
            best_model = candidate

    selected = asdict(best[1])
    return {
        "attack": "rena_x_targeted_deception",
        "threat_model": "black_box_adaptive_query_attack",
        "query_budget": len(trials),
        "selected": selected,
        "trials": [asdict(t) for t in trials],
        "candidate_model": best_model,
        "interpretation": (
            "The attacker queries the complete RENA-X decision rule and searches "
            "for the most residual forget behavior that is still accepted. This "
            "is stronger than a behavioral surrogate but is not a differentiable "
            "bilevel optimization of RENA-X."
        ),
    }
