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
    """Exact k-NN by full scan (reference implementation, used for ground truth)."""
    dists = np.linalg.norm(X_train - q, axis=1)
    order = np.argsort(dists)[:k]
    return order.astype(np.int32), dists[order]


class FastBruteForce:
    """
    Optimized exact k-NN baseline for fair timing comparisons.

    Uses ||x - q||^2 = ||x||^2 - 2 x.q + ||q||^2 with precomputed ||x||^2
    (one BLAS matrix-vector product per query) and argpartition instead of
    a full sort.
    """

    def __init__(self, X_train: np.ndarray) -> None:
        self.X = np.ascontiguousarray(X_train, dtype=np.float32)
        self.sq_norms = np.einsum("ij,ij->i", self.X, self.X)

    def query(self, q: np.ndarray, k: int = 1):
        d2 = self.sq_norms - 2.0 * (self.X @ q)
        idx = np.argpartition(d2, k)[:k]
        idx = idx[np.argsort(d2[idx])]
        dists = np.sqrt(np.maximum(d2[idx] + float(q @ q), 0.0))
        return idx.astype(np.int32), dists
