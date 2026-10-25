"""
RENA core — SISA (Sharded, Isolated, Sliced, Aggregated) unlearning.

Real implementation of the SISA approach (Bourtoule et al., 2021): training
data is partitioned into disjoint shards, one model is trained per shard, and
predictions are aggregated (averaged) across shard models. When a forget
request arrives, only the shard(s) containing the forgotten samples need to
be retrained — not the whole model — which is SISA's core efficiency claim.
"""
from __future__ import annotations

import numpy as np
from core.nn import SimpleMLP

N_SHARDS = 4
HIDDEN_LAYERS = (24, 12)


class SISAEnsemble:
    def __init__(self, n_shards: int = N_SHARDS):
        self.n_shards = n_shards
        self.shard_models: list[SimpleMLP] = []
        self.shard_indices: list[np.ndarray] = []  # original-index membership per shard

    def train(self, X: np.ndarray, y: np.ndarray, sample_ids: np.ndarray,
              epochs: int = 80, random_state: int = 42):
        """
        sample_ids: unique identifier per row of X (e.g. original dataset index),
        used so we can later determine which shard a "forget" sample lives in.
        """
        rng = np.random.RandomState(random_state)
        n = X.shape[0]
        n_classes = int(len(np.unique(y)))
        shard_assignment = rng.randint(0, self.n_shards, size=n)

        self.shard_models = []
        self.shard_indices = []
        for shard in range(self.n_shards):
            mask = shard_assignment == shard
            X_shard, y_shard = X[mask], y[mask]
            ids_shard = sample_ids[mask]
            model = SimpleMLP(X.shape[1], n_classes, hidden_sizes=HIDDEN_LAYERS,
                               random_state=random_state + shard)
            if len(X_shard) > 0:
                model.fit(X_shard, y_shard, epochs=epochs, lr=0.08,
                          batch_size=16, random_state=random_state + shard)
            self.shard_models.append(model)
            self.shard_indices.append(ids_shard)
        return self

    def unlearn(self, forget_ids: np.ndarray, X: np.ndarray, y: np.ndarray,
                sample_ids: np.ndarray, epochs: int = 80, random_state: int = 42):
        """
        Retrains ONLY the shard(s) containing forget_ids, excluding those
        samples. This is SISA's key property: sub-linear retraining cost.
        Returns (new_ensemble, n_shards_retrained).
        """
        forget_set = set(forget_ids.tolist())
        n_retrained = 0
        n_classes = int(np.max(y)) + 1

        for shard in range(self.n_shards):
            ids_shard = self.shard_indices[shard]
            overlap = forget_set.intersection(set(ids_shard.tolist()))
            if not overlap:
                continue
            n_retrained += 1
            keep_mask = np.isin(sample_ids, ids_shard) & ~np.isin(sample_ids, list(overlap))
            X_shard, y_shard = X[keep_mask], y[keep_mask]
            new_ids_shard = sample_ids[keep_mask]

            model = SimpleMLP(X.shape[1], n_classes, hidden_sizes=HIDDEN_LAYERS,
                               random_state=random_state + shard + 100)
            if len(X_shard) > 0:
                model.fit(X_shard, y_shard, epochs=epochs, lr=0.08,
                          batch_size=16, random_state=random_state + shard + 100)
            self.shard_models[shard] = model
            self.shard_indices[shard] = new_ids_shard

        return self, n_retrained

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = [m.predict_proba(X) for m in self.shard_models if len(self.shard_indices) > 0]
        return np.mean(probs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
