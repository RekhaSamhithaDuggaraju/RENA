"""
RENA core — dataset loading and retain/forget/test splitting.

Uses scikit-learn's digits dataset (8x8 grayscale handwritten digits, 10 classes)
as a stand-in for a real classification workload. Real data, real splits —
nothing simulated here.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class UnlearningSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_retain: np.ndarray
    y_retain: np.ndarray
    X_forget: np.ndarray
    y_forget: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler

    def summary(self) -> dict:
        return {
            "n_train": int(len(self.X_train)),
            "n_retain": int(len(self.X_retain)),
            "n_forget": int(len(self.X_forget)),
            "n_test": int(len(self.X_test)),
            "n_features": int(self.X_train.shape[1]),
            "n_classes": int(len(np.unique(self.y_train))),
        }


def build_split(forget_fraction: float = 0.1, random_state: int = 42) -> UnlearningSplit:
    """
    Loads the digits dataset and produces a retain/forget/test split.

    forget_fraction: proportion of the *training* set designated as the
    "forget set" — the data a forget-request targets.
    """
    data = load_digits()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    X_retain, X_forget, y_retain, y_forget = train_test_split(
        X_train, y_train,
        test_size=forget_fraction,
        random_state=random_state,
        stratify=y_train,
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_retain_s = scaler.transform(X_retain)
    X_forget_s = scaler.transform(X_forget)
    X_test_s = scaler.transform(X_test)

    return UnlearningSplit(
        X_train=X_train_s, y_train=y_train,
        X_retain=X_retain_s, y_retain=y_retain,
        X_forget=X_forget_s, y_forget=y_forget,
        X_test=X_test_s, y_test=y_test,
        scaler=scaler,
    )
