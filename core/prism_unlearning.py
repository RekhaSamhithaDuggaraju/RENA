"""PRISM: projected, recovery-aware iterative selective unlearning."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time
import numpy as np

from core.echo_evidence import fast_evidence_state
from core.model import clone_model


@dataclass
class PrismConfig:
    iterations: int = 18
    learning_rate: float = 0.30
    retain_correction: float = 0.012
    candidate_scales: tuple[float, ...] = (0.5, 1.0, 1.5)
    min_retain_accuracy: float = 0.97
    max_retain_drop: float = 0.015
    use_recovery_selection: bool = True
    use_disagreement_selection: bool = True
    use_retain_utility: bool = True
    epsilon: float = 1e-12


def _project_forget_gradient(model, X_forget, y_forget, X_retain, y_retain, epsilon):
    """Return ascent direction with non-positive retain-gradient inner product."""
    gf_w, gf_b = model._backward(X_forget, y_forget)
    gr_w, gr_b = model._backward(X_retain, y_retain)
    dot = sum(float(np.sum(a * b)) for a, b in zip(gf_w + gf_b, gr_w + gr_b))
    norm_sq = sum(float(np.sum(a * a)) for a in gr_w + gr_b)
    coefficient = max(0.0, dot / (norm_sq + epsilon))
    direction_w = [a - coefficient * b for a, b in zip(gf_w, gr_w)]
    direction_b = [a - coefficient * b for a, b in zip(gf_b, gr_b)]
    projected_dot = sum(float(np.sum(a * b)) for a, b in zip(direction_w + direction_b, gr_w + gr_b))
    return direction_w, direction_b, {"raw_gradient_dot": dot, "projection_coefficient": coefficient, "projected_retain_dot": projected_dot}


def _apply_ascent(model, direction_w, direction_b, step):
    candidate = clone_model(model)
    for index in range(len(candidate.W)):
        candidate.W[index] += step * direction_w[index]
        candidate.b[index] += step * direction_b[index]
    return candidate


def prism_unlearn(original_model, split, config: PrismConfig | None = None):
    """Run fixed-budget projected ascent with recovery-aware candidate choice."""
    config = config or PrismConfig()
    started = time.time()
    model = clone_model(original_model)
    baseline = fast_evidence_state(original_model, split, original_model, 1.0)
    baseline_retain = baseline["retain_accuracy"]
    trajectory = []
    candidate_evaluations = 0

    for iteration in range(1, config.iterations + 1):
        dw, db, projection = _project_forget_gradient(
            model, split.X_forget, split.y_forget, split.X_retain, split.y_retain, config.epsilon
        )
        candidates = []
        for scale in config.candidate_scales:
            candidate = _apply_ascent(model, dw, db, config.learning_rate * scale)
            if config.use_retain_utility:
                candidate.gradient_step(split.X_retain, split.y_retain, lr=config.retain_correction, ascent=False)
            if not all(np.all(np.isfinite(x)) for x in candidate.W + candidate.b):
                continue
            state = fast_evidence_state(candidate, split, original_model, baseline_retain)
            feasible = (not config.use_retain_utility or
                        (state["retain_accuracy"] >= config.min_retain_accuracy and
                         state["retain_degradation"] <= config.max_retain_drop))
            # Fixed lexicographic selector: recovery first, verifier-proxy dispersion second,
            # then stronger forget-loss separation and smaller learning step.
            proxy_values = [state[k] for k in ("membership_signal", "confidence_signal", "statistical_signal")]
            dispersion = float(np.std(proxy_values))
            key = (0 if feasible else 1,
                   state["recoverability_signal"] if config.use_recovery_selection else 0.0,
                   dispersion if config.use_disagreement_selection else 0.0,
                   -state["forget_loss_signal"], scale)
            candidates.append((key, candidate, state, scale, dispersion))
            candidate_evaluations += 1
        if not candidates:
            break
        _, model, state, scale, dispersion = min(candidates, key=lambda item: item[0])
        trajectory.append({"iteration": iteration, "selected_scale": scale, "projection": projection,
                           "state": state, "proxy_dispersion": dispersion})

    return model, {"config": asdict(config), "iterations": config.iterations,
                   "candidate_evaluations": candidate_evaluations,
                   "runtime_seconds": time.time() - started, "trajectory": trajectory,
                   "final_fast_evidence": trajectory[-1]["state"] if trajectory else baseline}
