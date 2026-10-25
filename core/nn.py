"""
RENA core — minimal but real feed-forward neural network in numpy.

Implemented manually (rather than relying on sklearn's MLPClassifier) because
gradient-ascent unlearning requires direct control over the gradient sign at
each SGD step, which sklearn's high-level fit() does not expose. Every
operation here is real linear algebra — no shortcuts, no lookup tables.

Architecture: Linear -> ReLU -> Linear -> ReLU -> Linear -> Softmax
"""
from __future__ import annotations

import copy
import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


class SimpleMLP:
    def __init__(self, n_features: int, n_classes: int,
                 hidden_sizes=(32, 16), random_state: int = 42):
        rng = np.random.RandomState(random_state)
        sizes = [n_features, *hidden_sizes, n_classes]
        self.sizes = sizes
        self.W = []
        self.b = []
        for i in range(len(sizes) - 1):
            # He initialization, appropriate for ReLU networks
            limit = np.sqrt(2.0 / sizes[i])
            self.W.append(rng.randn(sizes[i], sizes[i + 1]) * limit)
            self.b.append(np.zeros(sizes[i + 1]))
        self.n_classes = n_classes

    def clone(self) -> "SimpleMLP":
        return copy.deepcopy(self)

    def _forward(self, X: np.ndarray):
        """Returns activations and pre-activations for backprop."""
        a = X
        activations = [a]
        pre_acts = []
        n_layers = len(self.W)
        for i in range(n_layers):
            z = a @ self.W[i] + self.b[i]
            pre_acts.append(z)
            if i < n_layers - 1:
                a = np.maximum(0, z)  # ReLU
            else:
                a = _softmax(z)
            activations.append(a)
        return activations, pre_acts

    def hidden_representation(self, X: np.ndarray, layer: int = -1) -> np.ndarray:
        """Return a hidden-layer representation for representation-level audits."""
        activations, _ = self._forward(X)
        hidden = activations[1:-1]
        if not hidden:
            return activations[-1]
        return hidden[layer]

    def parameter_vector(self) -> np.ndarray:
        """Flatten all trainable parameters into a deterministic audit vector."""
        return np.concatenate([x.ravel() for x in self.W + self.b])

    def _soft_target_step(self, X: np.ndarray, target_probs: np.ndarray, lr: float = 0.01):
        """One gradient-descent step against teacher probabilities (distillation)."""
        activations, pre_acts = self._forward(X)
        n = X.shape[0]
        delta = (activations[-1] - target_probs) / max(n, 1)
        dW = [None] * len(self.W); db = [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            a_prev = activations[i]
            dW[i] = a_prev.T @ delta
            db[i] = np.sum(delta, axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (pre_acts[i - 1] > 0).astype(float)
        for i in range(len(self.W)):
            self.W[i] -= lr * dW[i]
            self.b[i] -= lr * db[i]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self._forward(X)
        return activations[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def _backward(self, X: np.ndarray, y: np.ndarray):
        """
        Standard backprop for softmax + cross-entropy.
        Returns gradients dW, db (same structure as self.W, self.b).
        """
        n = X.shape[0]
        activations, pre_acts = self._forward(X)
        y_onehot = np.eye(self.n_classes)[y]

        n_layers = len(self.W)
        dW = [None] * n_layers
        db = [None] * n_layers

        # Output layer error (softmax + cross-entropy combined gradient)
        delta = (activations[-1] - y_onehot) / n

        for i in reversed(range(n_layers)):
            a_prev = activations[i]
            dW[i] = a_prev.T @ delta
            db[i] = np.sum(delta, axis=0)
            if i > 0:
                da_prev = delta @ self.W[i].T
                relu_mask = (pre_acts[i - 1] > 0).astype(float)
                delta = da_prev * relu_mask

        return dW, db

    def gradient_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.05,
                       ascent: bool = False, l2: float = 1e-4):
        """
        A single full-batch gradient step. ascent=True flips the sign,
        i.e. performs gradient ASCENT on the given batch's loss — this is
        what actually implements the gradient-ascent unlearning algorithm:
        it pushes the model AWAY from correctly classifying the forget set.
        """
        dW, db = self._backward(X, y)
        sign = 1.0 if ascent else -1.0
        # Gradient-norm clipping: gradient ASCENT has no loss ceiling (unlike
        # descent, which naturally slows as loss approaches its minimum), so
        # without clipping it can blow up the weights within a few epochs.
        max_norm = 5.0
        for i in range(len(dW)):
            norm = np.sqrt(np.sum(dW[i] ** 2) + np.sum(db[i] ** 2))
            if norm > max_norm:
                scale = max_norm / (norm + 1e-12)
                dW[i] = dW[i] * scale
                db[i] = db[i] * scale
        for i in range(len(self.W)):
            self.W[i] += sign * lr * (dW[i] + l2 * self.W[i])
            self.b[i] += sign * lr * db[i]

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 60, lr: float = 0.05,
            batch_size: int = 32, ascent: bool = False, random_state: int = 0):
        rng = np.random.RandomState(random_state)
        n = X.shape[0]
        for _ in range(epochs):
            idx = rng.permutation(n)
            for start in range(0, n, batch_size):
                batch_idx = idx[start:start + batch_size]
                self.gradient_step(X[batch_idx], y[batch_idx], lr=lr, ascent=ascent)
        return self
