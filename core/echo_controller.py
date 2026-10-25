"""Evidence-guided Closed-loop Harmonic Optimization (ECHO) controller."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import time
import numpy as np

from core.echo_evidence import fast_evidence_state, parameter_distance
from core.echo_unlearning import OPERATORS, candidate_update


@dataclass
class EchoConfig:
    max_iterations: int = 10
    min_iterations: int = 3
    patience: int = 3
    learning_rate: float = 0.025
    min_retain_accuracy: float = 0.90
    max_recoverability: float = 0.20
    min_membership_signal: float = 0.70
    min_forget_loss_signal: float = 0.05
    improvement_epsilon: float = 1e-5
    use_pareto: bool = True
    use_adaptive_stopping: bool = True
    use_recovery_signal: bool = True


def _dominates(a: dict, b: dict, use_recovery: bool) -> bool:
    better = [a[k] >= b[k] for k in ("forget_loss_signal", "membership_signal", "confidence_signal", "statistical_signal")]
    nonworse = all(better) and a["retain_degradation"] <= b["retain_degradation"]
    if use_recovery: nonworse = nonworse and a["recoverability_signal"] <= b["recoverability_signal"]
    strict = any(a[k] > b[k] for k in ("forget_loss_signal", "membership_signal", "confidence_signal", "statistical_signal")) or a["retain_degradation"] < b["retain_degradation"]
    if use_recovery: strict = strict or a["recoverability_signal"] < b["recoverability_signal"]
    return nonworse and strict


def _select(candidates: list[dict], config: EchoConfig) -> dict:
    if not config.use_pareto:
        return candidates[0]  # fixed operator order is the non-Pareto ablation.
    frontier = [a for a in candidates if not any(_dominates(b["state"], a["state"], config.use_recovery_signal) for b in candidates if b is not a)]
    # Deterministic, non-weighted tie-breaker requested in the specification.
    return min(frontier, key=lambda x: (x["state"]["retain_degradation"], x["state"]["recoverability_signal"] if config.use_recovery_signal else 0.0,
                                        -min(x["state"][k] for k in ("forget_loss_signal", "membership_signal", "confidence_signal", "statistical_signal")), x["parameter_update_norm"], x["operator"]))


def _acceptance(state: dict, config: EchoConfig) -> bool:
    return (state["forget_loss_signal"] >= config.min_forget_loss_signal and state["membership_signal"] >= config.min_membership_signal and
            state["retain_accuracy"] >= config.min_retain_accuracy and (not config.use_recovery_signal or state["recoverability_signal"] <= config.max_recoverability))


def echo_unlearn(original_model, split, config: EchoConfig | None = None) -> tuple[object, dict]:
    config = config or EchoConfig(); started = time.time(); model = original_model.clone()
    baseline_retain = fast_evidence_state(original_model, split, original_model, 1.0)["retain_accuracy"]
    trajectory, stale, evaluations, stopping_reason = [], 0, 0, "maximum_iterations_reached"
    prior = fast_evidence_state(model, split, original_model, baseline_retain)
    for iteration in range(1, config.max_iterations + 1):
        candidates = []
        for operator in OPERATORS:
            candidate = candidate_update(model, split, operator, config.learning_rate)
            state = fast_evidence_state(candidate, split, original_model, baseline_retain)
            candidates.append({"operator": operator, "model": candidate, "state": state, "parameter_update_norm": parameter_distance(candidate, model)})
            evaluations += 1
        chosen = _select(candidates, config); state = chosen["state"]
        finite = all(np.isfinite(v) for v in state.values() if isinstance(v, float))
        if not finite:
            stopping_reason = "numerical_stability_check_failed"; break
        improvement = any(state[k] > prior[k] + config.improvement_epsilon for k in ("forget_loss_signal", "membership_signal", "confidence_signal", "statistical_signal")) or state["retain_degradation"] + config.improvement_epsilon < prior["retain_degradation"]
        stale = 0 if improvement else stale + 1
        model, prior = chosen["model"], state
        trajectory.append({"iteration": iteration, "selected_operator": chosen["operator"], "candidate_evaluations": len(candidates), "pareto_frontier_size": sum(not any(_dominates(b["state"], a["state"], config.use_recovery_signal) for b in candidates if b is not a) for a in candidates), "parameter_update_norm": chosen["parameter_update_norm"], "evidence_state": state})
        if config.use_adaptive_stopping and iteration >= config.min_iterations and _acceptance(state, config) and stale >= config.patience:
            stopping_reason = "evidence_acceptance_and_pareto_patience"; break
    return model, {"config": asdict(config), "iterations": len(trajectory), "candidate_evaluations": evaluations, "runtime_seconds": time.time() - started, "stopping_reason": stopping_reason, "trajectory": trajectory, "final_fast_evidence": prior}
