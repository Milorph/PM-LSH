from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
from scipy.stats import chi2

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
    PM-LSH-style ANN index.

    This version is intentionally stricter than the earlier prototype:
    - uses scipy's chi-square quantiles instead of a loose approximation
    - keeps beta small by default so the search stays genuinely approximate
    - estimates r_min from sampled k-NN radii rather than random pairwise distances
    - allows multiple projected-space expansions before exact reranking
    - exposes `search_width`, a multiplier on the chi-square confidence net; the
      paper's default alpha1 = 1/e (search_width = 1.0) captures a true neighbor at
      the search radius with only ~0.63 probability, so recall saturates around
      0.6-0.8. Widening the net (search_width > 1) recovers recall toward ~1.0 at
      the cost of examining more candidates -- the accuracy/efficiency tradeoff.

    It is still a research-style Python reimplementation, not the original C++ code.
    """

    def __init__(
        self,
        m: int = 15,
        n_pivots: int = 5,
        c: float = 1.5,
        alpha1: float = 1 / math.e,
        search_width: float = 1.0,
        alpha2: Optional[float] = None,
        beta: Optional[float] = None,
        beta_cap: float = 0.10,
        max_leaf_size: int = 32,
        max_internal_size: int = 16,
        random_state: int = 42,
        rmin_sample_size: int = 192,
        rmin_quantile: float = 0.35,
        max_expansions: int = 8,
    ) -> None:
        self.m = int(m)
        self.n_pivots = int(n_pivots)
        self.c = float(c)
        self.alpha1 = float(alpha1)
        self.search_width = float(search_width)
        self.random_state = int(random_state)
        self.rng = np.random.default_rng(random_state)
        self.max_leaf_size = int(max_leaf_size)
        self.max_internal_size = int(max_internal_size)
        self.beta_cap = float(beta_cap)
        self.rmin_sample_size = int(rmin_sample_size)
        self.rmin_quantile = float(rmin_quantile)
        self.max_expansions = int(max_expansions)

        self.alpha2 = float(alpha2) if alpha2 is not None else self._solve_alpha2(self.m, self.c, self.alpha1)
        derived_beta = max(2.0 * self.alpha2, 1e-4)
        if beta is None:
            self.beta = float(min(derived_beta, self.beta_cap))
        else:
            self.beta = float(beta)

        self.t = float(math.sqrt(self._chi2_upper_quantile(self.m, self.alpha1)))
        # Effective confidence net actually used at query time. search_width > 1
        # widens the chi-square net beyond the alpha1 default, which is the knob
        # that trades higher recall for more candidates examined. The effective
        # per-projection miss probability is the chi-square upper-tail at t_eff^2.
        self.effective_t = self.search_width * self.t
        self.effective_alpha1 = float(chi2.sf(self.effective_t ** 2, self.m))

        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.projection_matrix: Optional[np.ndarray] = None
        self.X_proj: Optional[np.ndarray] = None
        self.pivots: Optional[np.ndarray] = None
        self.tree: Optional[PMTree] = None
        self.sample_knn_radii: Optional[np.ndarray] = None

    @staticmethod
    def _chi2_upper_quantile(df: int, alpha: float) -> float:
        return float(chi2.isf(alpha, df))

    def _solve_alpha2(self, m: int, c: float, alpha1: float) -> float:
        """
        Solve Equation (10) from the PM-LSH paper exactly.

            t^2 = chi2_{alpha1}(m) = c^2 * chi2_{1-alpha2}(m)

        The paper defines chi2_a(m) as the *upper-tail* quantile, so:
            chi2_{1-alpha2}(m) = ppf(alpha2; m)

        Therefore:
            alpha2 = CDF( chi2_{alpha1}(m) / c^2 ; m )

        This closed form is both simpler and more numerically stable than the
        earlier binary search, and it avoids the previous direction bug that
        incorrectly drove alpha2 toward 1.
        """
        target = self._chi2_upper_quantile(m, alpha1)
        alpha2 = float(chi2.cdf(target / (c ** 2), m))
        return min(max(alpha2, 1e-12), 1.0 - 1e-12)

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
        self.tree = PMTree(
            self.pivots,
            max_leaf_size=self.max_leaf_size,
            max_internal_size=self.max_internal_size,
        )
        self.tree.build(self.X_proj)
        self.sample_knn_radii = self._sample_knn_radii(k=10)
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

    def _sample_knn_radii(self, k: int) -> np.ndarray:
        if self.X is None:
            raise ValueError("Model not fit")
        n = self.X.shape[0]
        sample_size = min(self.rmin_sample_size, n)
        ids = self.rng.choice(n, size=sample_size, replace=False)
        radii = []
        for idx in ids:
            q = self.X[idx]
            dists = np.linalg.norm(self.X - q, axis=1)
            kth = np.partition(dists, min(k, len(dists) - 1))[min(k, len(dists) - 1)]
            radii.append(float(kth))
        return np.sort(np.asarray(radii, dtype=np.float32))

    def _estimate_r_min(self, k: int) -> float:
        if self.sample_knn_radii is None:
            raise ValueError("Model not fit")
        # Start from a conservative local-neighborhood scale, not random pairwise distances.
        idx = min(len(self.sample_knn_radii) - 1, max(0, int(self.rmin_quantile * len(self.sample_knn_radii))))
        base = float(self.sample_knn_radii[idx])
        scale = max(1.0, math.sqrt(max(k, 1) / 10.0))
        return max(base * scale, 1e-8)

    def _projected_range_radius(self, original_radius: float) -> float:
        return self.effective_t * original_radius

    def _finalize(self, q: np.ndarray, candidate_ids: List[int], k: int, stats: QueryStats) -> Tuple[np.ndarray, np.ndarray, QueryStats]:
        if self.X is None:
            raise ValueError("Index not fit")
        if not candidate_ids:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32), stats
        candidate_arr = np.array(candidate_ids, dtype=np.int32)
        dists = np.linalg.norm(self.X[candidate_arr] - q, axis=1)
        order = np.argsort(dists)[:k]
        return candidate_arr[order], dists[order], stats

    def query_ckann(self, q: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray, QueryStats]:
        if self.tree is None or self.X is None or self.X_proj is None:
            raise ValueError("Index not fit")

        q = np.asarray(q, dtype=np.float32)
        q_proj = self._project(q.reshape(1, -1))[0]
        r = self._estimate_r_min(k)
        n_required = int(math.ceil(self.beta * len(self.X) + k))

        candidate_ids: List[int] = []
        seen: set[int] = set()
        expansions = 0
        last_proj_radius = 0.0
        best_ids = np.array([], dtype=np.int32)
        best_dists = np.array([], dtype=np.float32)

        while expansions < self.max_expansions:
            last_proj_radius = self._projected_range_radius(r)
            hits = self.tree.range_query(q_proj, last_proj_radius)
            for idx in hits:
                if idx not in seen:
                    seen.add(idx)
                    candidate_ids.append(idx)

            expansions += 1
            stats = QueryStats(
                projected_radius=last_proj_radius,
                original_radius=r,
                candidates_found=len(candidate_ids),
                distance_computations=self.tree.n_distance_computations,
                expansions=expansions,
            )

            if candidate_ids:
                best_ids, best_dists, _ = self._finalize(q, candidate_ids, k, stats)
                within = int(np.sum(best_dists <= self.c * r))
                if len(candidate_ids) >= n_required or within >= k:
                    return best_ids, best_dists, stats

            r *= self.c

        # fallback after exhausting expansions
        final_stats = QueryStats(
            projected_radius=last_proj_radius,
            original_radius=r,
            candidates_found=len(candidate_ids),
            distance_computations=self.tree.n_distance_computations,
            expansions=expansions,
        )
        return self._finalize(q, candidate_ids, k, final_stats)

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
