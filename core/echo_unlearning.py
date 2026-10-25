"""NumPy-MLP ECHO candidate update operators."""
from __future__ import annotations

from core.model import clone_model


OPERATORS = ("forget_pressure", "retain_protected", "boundary_separation", "stability")


def candidate_update(base_model, split, operator: str, lr: float = 0.025):
    """Return a temporary candidate; the controller alone decides commitment."""
    model = clone_model(base_model)
    if operator == "forget_pressure":
        model.gradient_step(split.X_forget, split.y_forget, lr=lr, ascent=True)
    elif operator == "retain_protected":
        model.gradient_step(split.X_forget, split.y_forget, lr=lr, ascent=True)
        model.gradient_step(split.X_retain, split.y_retain, lr=lr * 0.45, ascent=False)
    elif operator == "boundary_separation":
        # A targeted forget ascent followed by conservative retain correction.
        model.gradient_step(split.X_forget, split.y_forget, lr=lr * 0.75, ascent=True)
        model.gradient_step(split.X_retain, split.y_retain, lr=lr * 0.75, ascent=False)
    elif operator == "stability":
        model.gradient_step(split.X_forget, split.y_forget, lr=lr * 0.35, ascent=True)
        model.gradient_step(split.X_retain, split.y_retain, lr=lr * 0.70, ascent=False)
    else:
        raise ValueError(f"Unknown ECHO operator: {operator}")
    return model
