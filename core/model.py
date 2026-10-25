"""
RENA core — base model training/evaluation utilities, built on core.nn.SimpleMLP.
"""
from __future__ import annotations

import copy
import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from core.nn import SimpleMLP

HIDDEN_LAYERS = (32, 16)
RANDOM_STATE = 42


def train_base_model(X: np.ndarray, y: np.ndarray, epochs: int = 80, random_state: int = RANDOM_STATE, fit_random_state: int | None = None) -> SimpleMLP:
    """Trains the original model on the full training set (retain + forget)."""
    n_features = X.shape[1]
    n_classes = int(len(np.unique(y)))
    model = SimpleMLP(n_features, n_classes, hidden_sizes=HIDDEN_LAYERS,
                       random_state=random_state)
    model.fit(X, y, epochs=epochs, lr=0.08, batch_size=32,
              random_state=random_state if fit_random_state is None else fit_random_state)
    return model


def clone_model(model: SimpleMLP) -> SimpleMLP:
    return copy.deepcopy(model)


def per_sample_loss(model: SimpleMLP, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Real per-sample cross-entropy loss from the model's predicted probabilities."""
    probs = model.predict_proba(X)
    n_classes = probs.shape[1]
    y_onehot = np.eye(n_classes)[y]
    eps = 1e-12
    probs_clipped = np.clip(probs, eps, 1 - eps)
    return -np.sum(y_onehot * np.log(probs_clipped), axis=1)


def evaluate(model: SimpleMLP, X: np.ndarray, y: np.ndarray) -> dict:
    preds = model.predict(X)
    probs = model.predict_proba(X)
    n_classes = probs.shape[1]
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "log_loss": float(log_loss(y, probs, labels=np.arange(n_classes))),
    }
