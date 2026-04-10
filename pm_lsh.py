from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
from statistics import NormalDist

from pmtree import PMTree


@dataclass
class QueryStats:
    projected_radius: float
    original_radius: float
    candidates_found: int
    distance_computations: int
    expansions: int


class PMLSH:
    """
    PM-LSH style ANN index.

    Core paper-aligned ingredients:
    1) project data into an m-dimensional space using stable Gaussian projections
    2) index projected points with a PM-tree
    3) use the chi-square confidence interval relationship between projected and
       original distances to convert original-space radii into projected-space radii
    4) answer (c, k)-ANN by repeated projected-space range expansion

    This is a faithful research-style reimplementation of the framework, but it is
    not an official reproduction of the paper's exact C++ codebase.
    """

    def __init__(
        self,
        m: int = 15,
        n_pivots: int = 5,
        c: float = 1.5,
        alpha1: float = 1 / math.e,
        beta: Optional[float] = None,
        max_leaf_size: int = 32,
        max_internal_size: int = 16,
        random_state: int = 42,
    ) -> None:
        self.m = int(m)
        self.n_pivots = int(n_pivots)
        self.c = float(c)
        self.alpha1 = float(alpha1)
        self.random_state = int(random_state)
        self.rng = np.random.default_rng(random_state)
        self.max_leaf_size = int(max_leaf_size)
        self.max_internal_size = int(max_internal_size)

        self.alpha2 = self._solve_alpha2(self.m, self.c, self.alpha1)
        self.beta = float(beta) if beta is not None else float(max(2 * self.alpha2, 1e-4))
        self.t = float(math.sqrt(self._chi2_upper_quantile(self.m, self.alpha1)))

        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.projection_matrix: Optional[np.ndarray] = None
        self.X_proj: Optional[np.ndarray] = None
        self.pivots: Optional[np.ndarray] = None
        self.tree: Optional[PMTree] = None
        self.distance_distribution: Optional[np.ndarray] = None

    @staticmethod
    def _chi2_upper_quantile(df: int, alpha: float) -> float:
        # Wilson-Hilferty approximation.
        z = NormalDist().inv_cdf(1.0 - alpha)
        return float(df * (1.0 - 2.0 / (9.0 * df) + z * math.sqrt(2.0 / (9.0 * df))) ** 3)

    def _solve_alpha2(self, m: int, c: float, alpha1: float) -> float:
        target = self._chi2_upper_quantile(m, alpha1)
        lo, hi = 1e-9, 0.49
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lhs = target
            rhs = (c ** 2) * self._chi2_upper_quantile(m, 1.0 - mid)
            if rhs > lhs:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _project(self, X: np.ndarray) -> np.ndarray:
        if self.projection_matrix is None:
            raise ValueError("Projection matrix not initialized")
        return (X @ self.projection_matrix.T).astype(np.float32)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PMLSH":
        self.X = np.asarray(X, dtype=np.float32)
        self.y = None if y is None else np.asarray(y)
        n, d = self.X.shape
        self.projection_matrix = self.rng.normal(0.0, 1.0, size=(self.m, d)).astype(np.float32)
        self.X_proj = self._project(self.X)

        pivot_ids = self._choose_pivots(self.X_proj, self.n_pivots)
        self.pivots = self.X_proj[pivot_ids]
        self.tree = PMTree(self.pivots, max_leaf_size=self.max_leaf_size, max_internal_size=self.max_internal_size)
        self.tree.build(self.X_proj)

        # sample original-space distance distribution to estimate r_min as in the paper
        sample_size = min(2000, n)
        ids_a = self.rng.integers(0, n, size=sample_size)
        ids_b = self.rng.integers(0, n, size=sample_size)
        dists = np.linalg.norm(self.X[ids_a] - self.X[ids_b], axis=1)
        self.distance_distribution = np.sort(dists)
        return self

    def _choose_pivots(self, X_proj: np.ndarray, n_pivots: int) -> np.ndarray:
        n = X_proj.shape[0]
        first = self.rng.integers(0, n)
        chosen = [first]
        min_d = np.linalg.norm(X_proj - X_proj[first], axis=1)
        for _ in range(1, min(n_pivots, n)):
            nxt = int(np.argmax(min_d))
            chosen.append(nxt)
            d = np.linalg.norm(X_proj - X_proj[nxt], axis=1)
            min_d = np.minimum(min_d, d)
        return np.array(chosen, dtype=np.int32)

    def _estimate_r_min(self, k: int) -> float:
        if self.distance_distribution is None or self.X is None:
            raise ValueError("Model not fit")
        n = self.X.shape[0]
        target_rank = min(len(self.distance_distribution) - 1, max(1, int((self.beta * n + k) / max(n, 1) * len(self.distance_distribution))))
        r = float(self.distance_distribution[target_rank])
        return max(r * 0.9, 1e-8)

    def _projected_range_radius(self, original_radius: float) -> float:
        return self.t * original_radius

    def query_ckann(self, q: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray, QueryStats]:
        if self.tree is None or self.X is None or self.X_proj is None:
            raise ValueError("Index not fit")

        q = np.asarray(q, dtype=np.float32)
        q_proj = self._project(q.reshape(1, -1))[0]
        r = self._estimate_r_min(k)
        n_required = int(math.ceil(self.beta * len(self.X) + k))

        candidate_ids: List[int] = []
        seen = set()
        expansions = 0
        last_proj_radius = 0.0

        while True:
            # termination condition using original-space verification
            if candidate_ids:
                dists = np.linalg.norm(self.X[candidate_ids] - q, axis=1)
                within = np.sum(dists <= self.c * r)
                if within >= k:
                    order = np.argsort(dists)[:k]
                    chosen_ids = np.array(candidate_ids, dtype=np.int32)[order]
                    return (
                        chosen_ids,
                        dists[order],
                        QueryStats(last_proj_radius, r, len(candidate_ids), self.tree.n_distance_computations, expansions),
                    )

            last_proj_radius = self._projected_range_radius(r)
            hits = self.tree.range_query(q_proj, last_proj_radius)
            for idx in hits:
                if idx not in seen:
                    seen.add(idx)
                    candidate_ids.append(idx)

            expansions += 1
            if len(candidate_ids) >= n_required:
                dists = np.linalg.norm(self.X[candidate_ids] - q, axis=1)
                order = np.argsort(dists)[:k]
                chosen_ids = np.array(candidate_ids, dtype=np.int32)[order]
                return (
                    chosen_ids,
                    dists[order],
                    QueryStats(last_proj_radius, r, len(candidate_ids), self.tree.n_distance_computations, expansions),
                )

            r *= self.c
            if r > float(np.max(self.distance_distribution) * 4.0):
                dists = np.linalg.norm(self.X[candidate_ids] - q, axis=1) if candidate_ids else np.array([])
                if len(dists) == 0:
                    return np.array([], dtype=np.int32), np.array([], dtype=np.float32), QueryStats(last_proj_radius, r, 0, self.tree.n_distance_computations, expansions)
                order = np.argsort(dists)[:k]
                chosen_ids = np.array(candidate_ids, dtype=np.int32)[order]
                return chosen_ids, dists[order], QueryStats(last_proj_radius, r, len(candidate_ids), self.tree.n_distance_computations, expansions)

    def batch_query_ckann(self, Q: np.ndarray, k: int = 1) -> Dict[str, object]:
        ids_all = []
        dists_all = []
        stats_all = []
        for q in Q:
            ids, dists, stats = self.query_ckann(q, k=k)
            ids_all.append(ids)
            dists_all.append(dists)
            stats_all.append(stats)
        return {"ids": ids_all, "distances": dists_all, "stats": stats_all}
