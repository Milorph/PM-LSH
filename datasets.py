from __future__ import annotations

from typing import Tuple
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_mnist(train_size: int = 15000, test_size: int = 2000, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("Loading MNIST from OpenML...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=10000,
        random_state=random_state,
        stratify=y,
    )
    X_train = X_train[:train_size]
    y_train = y_train[:train_size]
    X_test = X_test[:test_size]
    y_test = y_test[:test_size]
    print(f"Train subset: {X_train.shape}")
    print(f"Test subset:  {X_test.shape}")
    return X_train, X_test, y_train, y_test


def brute_force_topk(X_train: np.ndarray, q: np.ndarray, k: int = 1):
    dists = np.linalg.norm(X_train - q, axis=1)
    order = np.argsort(dists)[:k]
    return order.astype(np.int32), dists[order]
