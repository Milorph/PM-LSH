from __future__ import annotations

from typing import List, Optional, Tuple
import math
import numpy as np


class PMLeaf:
    """Leaf node: all entries stored as contiguous arrays so filters run vectorized."""

    __slots__ = ("ids", "points", "pivot_distances")

    def __init__(self, ids: np.ndarray, points: np.ndarray, pivot_distances: np.ndarray) -> None:
        self.ids = ids                          # (k,)
        self.points = points                    # (k, m)
        self.pivot_distances = pivot_distances  # (k, s)


class PMInternal:
    """Internal node: routing objects, covering radii and pivot hyper-rings as arrays."""

    __slots__ = ("centers", "radii", "hr_min", "hr_max", "children")

    def __init__(self, centers, radii, hr_min, hr_max, children) -> None:
        self.centers = centers    # (c, m)
        self.radii = radii        # (c,)
        self.hr_min = hr_min      # (c, s)
        self.hr_max = hr_max      # (c, s)
        self.children = children  # list of length c


class PMTree:
    """
    PM-tree style metric index for Euclidean range search in the projected space.

    Mirrors the paper's core ideas:
    - ball-partition nodes with routing objects and covering radii
    - pivot hyper-ring (HR) bounds on every subtree, used for pruning
    - per-entry pivot distances at the leaves, used to skip exact distance checks

    Each node stores its entries as NumPy arrays, so the ball filter, the HR
    filters and the leaf checks for a whole node run as one vectorized
    operation instead of a Python loop over entries. Pruning decisions are
    identical to the per-entry formulation.
    """

    def __init__(
        self,
        pivots: np.ndarray,
        max_leaf_size: int = 32,
        max_internal_size: int = 16,
    ) -> None:
        self.pivots = np.asarray(pivots, dtype=np.float32)
        self.max_leaf_size = int(max_leaf_size)
        self.max_internal_size = int(max_internal_size)
        self.root = None
        self.n_distance_computations: int = 0
        self.n_nodes_visited: int = 0

    def reset_stats(self) -> None:
        self.n_distance_computations = 0
        self.n_nodes_visited = 0

    # ------------------------------------------------------------------ build
    def build(self, points: np.ndarray) -> "PMTree":
        points = np.asarray(points, dtype=np.float32)
        ids = np.arange(points.shape[0], dtype=np.int32)
        self.root = self._build_recursive(ids, points)
        return self

    def _pivot_dists(self, pts: np.ndarray) -> np.ndarray:
        return np.linalg.norm(pts[:, None, :] - self.pivots[None, :, :], axis=2).astype(np.float32)

    def _build_recursive(self, ids: np.ndarray, pts: np.ndarray):
        if len(ids) <= self.max_leaf_size:
            return PMLeaf(ids, pts, self._pivot_dists(pts))

        centers, groups = self._partition(pts)
        c_list, r_list, hmin, hmax, children = [], [], [], [], []
        for center, g in zip(centers, groups):
            g_pts = pts[g]
            children.append(self._build_recursive(ids[g], g_pts))
            c_list.append(center)
            r_list.append(float(np.max(np.linalg.norm(g_pts - center, axis=1))))
            pd = self._pivot_dists(g_pts)
            hmin.append(pd.min(axis=0))
            hmax.append(pd.max(axis=0))
        return PMInternal(
            np.stack(c_list).astype(np.float32),
            np.asarray(r_list, dtype=np.float32),
            np.stack(hmin).astype(np.float32),
            np.stack(hmax).astype(np.float32),
            children,
        )

    def _partition(self, pts: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        n = pts.shape[0]
        n_centers = min(self.max_internal_size, max(2, math.ceil(n / self.max_leaf_size)))

        # farthest-first seed selection
        centers_idx = [0]
        min_d = np.linalg.norm(pts - pts[0], axis=1)
        for _ in range(1, n_centers):
            nxt = int(np.argmax(min_d))
            centers_idx.append(nxt)
            min_d = np.minimum(min_d, np.linalg.norm(pts - pts[nxt], axis=1))

        centers = pts[centers_idx].copy()
        d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
        assign = np.argmin(d, axis=1)

        out_c, out_g = [], []
        for j in range(len(centers)):
            g = np.nonzero(assign == j)[0]
            if len(g) > 0:  # drop empty groups
                out_c.append(centers[j])
                out_g.append(g)
        return out_c, out_g

    # ------------------------------------------------------------------ query
    def range_query(self, query: np.ndarray, radius: float) -> np.ndarray:
        if self.root is None:
            raise ValueError("PM-tree not built")
        query = np.asarray(query, dtype=np.float32)
        self.reset_stats()
        q_pd = np.linalg.norm(self.pivots - query, axis=1).astype(np.float32)
        lo = q_pd - radius
        hi = q_pd + radius

        out: List[np.ndarray] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            self.n_nodes_visited += 1
            if isinstance(node, PMLeaf):
                # pivot filter: |d(q,p) - d(o,p)| <= r for every pivot p
                keep = np.all(np.abs(q_pd - node.pivot_distances) <= radius, axis=1)
                if not keep.any():
                    continue
                cand = node.points[keep]
                self.n_distance_computations += cand.shape[0]
                d = np.linalg.norm(cand - query, axis=1)
                hit = d <= radius
                if hit.any():
                    out.append(node.ids[keep][hit])
                continue

            # ball-overlap filter + pivot hyper-ring filters for all children at once
            self.n_distance_computations += node.centers.shape[0]
            d = np.linalg.norm(node.centers - query, axis=1)
            keep = d <= node.radii + radius
            keep &= np.all(lo <= node.hr_max, axis=1)
            keep &= np.all(hi >= node.hr_min, axis=1)
            for j in np.nonzero(keep)[0]:
                stack.append(node.children[j])

        if not out:
            return np.empty(0, dtype=np.int32)
        return np.concatenate(out)
