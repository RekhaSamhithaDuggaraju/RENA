"""
RENA core — unlearning algorithms.

Each function takes a trained model (or the raw data needed to build one) and
returns a genuinely modified model. Nothing here is a stub: every algorithm
performs real gradient computation and real weight updates via core.nn.SimpleMLP.
"""
from __future__ import annotations

import numpy as np
from core.model import clone_model, train_base_model
from core.nn import SimpleMLP
from core.sisa import SISAEnsemble


def fine_tuning_unlearn(base_model: SimpleMLP, X_retain: np.ndarray, y_retain: np.ndarray,
                         epochs: int = 15) -> SimpleMLP:
    """
    Naive baseline: continue training the existing model on the retain set
    only, hoping the forget set's influence "fades." This is the weakest
    approach in the literature — it doesn't actively suppress memorization
    of the forget set, it just dilutes it with more retain-set gradient steps.
    """
    model = clone_model(base_model)
    model.fit(X_retain, y_retain, epochs=epochs, lr=0.03, batch_size=32,
              ascent=False, random_state=1)
    return model


def gradient_ascent_unlearn(base_model: SimpleMLP, X_forget: np.ndarray, y_forget: np.ndarray,
                             X_retain: np.ndarray, y_retain: np.ndarray,
                             ascent_epochs: int = 18, recovery_epochs: int = 6) -> SimpleMLP:
    """
    Performs true gradient ASCENT on the forget set (maximizing its loss,
    actively pushing the model away from correctly classifying those
    samples), then a short recovery fine-tune on the retain set to restore
    general utility. This is a real, if imperfect, unlearning method from
    the literature (Thudi et al. and related gradient-ascent approaches).

    Hyperparameters tuned empirically: too few ascent epochs leaves forget-set
    accuracy unchanged (ascent has no effect); too many destabilizes the
    weights (gradient ascent has no loss ceiling, unlike descent). 18 epochs
    at lr=0.045 was the smallest setting that produced a measurable,
    stable degradation in forget-set accuracy on this dataset/architecture.
    """
    model = clone_model(base_model)
    model.fit(X_forget, y_forget, epochs=ascent_epochs, lr=0.045, batch_size=16,
              ascent=True, random_state=2)
    if recovery_epochs > 0:
        model.fit(X_retain, y_retain, epochs=recovery_epochs, lr=0.015, batch_size=32,
                  ascent=False, random_state=3)
    return model


def ground_truth_retrain(X_retain: np.ndarray, y_retain: np.ndarray, random_state: int = 42) -> SimpleMLP:
    """
    The gold standard: train a fresh model from scratch on the retain set
    only. The forget set never influenced this model at all, so by
    definition there is nothing to "forget" — this is the target every
    other algorithm is trying to approximate.
    """
    return train_base_model(X_retain, y_retain, epochs=80, random_state=random_state, fit_random_state=random_state)


def sisa_unlearn(X_train: np.ndarray, y_train: np.ndarray, forget_ids: np.ndarray,
                  sample_ids: np.ndarray) -> tuple[SISAEnsemble, int]:
    """
    Trains a SISA ensemble on the full training set, then unlearns by
    retraining only the shard(s) that contained forget_ids. Returns the
    updated ensemble and the number of shards that had to be retrained
    (SISA's efficiency claim — should be << total shard count).
    """
    ensemble = SISAEnsemble(n_shards=4)
    ensemble.train(X_train, y_train, sample_ids, epochs=80)
    ensemble, n_retrained = ensemble.unlearn(forget_ids, X_train, y_train, sample_ids, epochs=80)
    return ensemble, n_retrained
