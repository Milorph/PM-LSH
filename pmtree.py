from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import math
import numpy as np


@dataclass
class PMLeafEntry:
    point_id: int
    point: np.ndarray
    parent_distance: float
    pivot_distances: np.ndarray


@dataclass
class PMInternalEntry:
    routing_object: np.ndarray
    child: "PMNode"
    radius: float
    parent_distance: float
    hr_min: np.ndarray
    hr_max: np.ndarray


@dataclass
class PMNode:
    is_leaf: bool
    entries: List[object] = field(default_factory=list)


class PMTree:
    """
    A compact PM-tree style metric index for Euclidean search in projected space.

    This is a research-project implementation intended to mirror the paper's core
    ideas:
    - pivot-aware pruning via hyper-ring bounds
    - ball-partition / routing-object style nodes
    - range-query support

    It is not a byte-for-byte reproduction of the authors' C++ implementation.
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
        self.root: Optional[PMNode] = None
        self.n_distance_computations: int = 0

    def reset_stats(self) -> None:
        self.n_distance_computations = 0

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        self.n_distance_computations += 1
        return float(np.linalg.norm(a - b))

    def build(self, points: np.ndarray) -> "PMTree":
        points = np.asarray(points, dtype=np.float32)
        indexed = [(i, points[i]) for i in range(points.shape[0])]
        self.root = self._build_recursive(indexed, parent_center=None)
        return self

    def _build_recursive(
        self,
        items: List[Tuple[int, np.ndarray]],
        parent_center: Optional[np.ndarray],
    ) -> PMNode:
        if len(items) <= self.max_leaf_size:
            entries: List[PMLeafEntry] = []
            for idx, point in items:
                pd = 0.0 if parent_center is None else float(np.linalg.norm(point - parent_center))
                pivot_distances = np.linalg.norm(self.pivots - point, axis=1).astype(np.float32)
                entries.append(PMLeafEntry(idx, point, pd, pivot_distances))
            return PMNode(is_leaf=True, entries=entries)

        centers, groups = self._partition(items, self.max_internal_size)
        entries: List[PMInternalEntry] = []
        node = PMNode(is_leaf=False, entries=[])
        for center, group in zip(centers, groups):
            child = self._build_recursive(group, parent_center=center)
            radius = 0.0
            pts = np.stack([p for _, p in group], axis=0)
            if len(group) > 0:
                radius = float(np.max(np.linalg.norm(pts - center, axis=1)))
            if parent_center is None:
                parent_distance = 0.0
            else:
                parent_distance = float(np.linalg.norm(center - parent_center))

            all_pivot_dists = np.linalg.norm(self.pivots[None, :, :] - pts[:, None, :], axis=2)
            hr_min = all_pivot_dists.min(axis=0).astype(np.float32)
            hr_max = all_pivot_dists.max(axis=0).astype(np.float32)
            entries.append(
                PMInternalEntry(
                    routing_object=center.astype(np.float32),
                    child=child,
                    radius=radius,
                    parent_distance=parent_distance,
                    hr_min=hr_min,
                    hr_max=hr_max,
                )
            )
        node.entries = entries
        return node

    def _partition(
        self,
        items: List[Tuple[int, np.ndarray]],
        max_children: int,
    ) -> Tuple[List[np.ndarray], List[List[Tuple[int, np.ndarray]]]]:
        pts = np.stack([p for _, p in items], axis=0)
        n = len(items)
        n_centers = min(max_children, max(2, math.ceil(n / self.max_leaf_size)))

        # farthest-first seed selection
        centers_idx = [0]
        min_d = np.linalg.norm(pts - pts[0], axis=1)
        for _ in range(1, n_centers):
            next_idx = int(np.argmax(min_d))
            centers_idx.append(next_idx)
            d = np.linalg.norm(pts - pts[next_idx], axis=1)
            min_d = np.minimum(min_d, d)

        centers = [pts[i].copy() for i in centers_idx]
        assignments = [[] for _ in centers]
        for item in items:
            point = item[1]
            ds = [float(np.linalg.norm(point - c)) for c in centers]
            assignments[int(np.argmin(ds))].append(item)

        # repair empty groups if any
        non_empty = [(c, g) for c, g in zip(centers, assignments) if len(g) > 0]
        centers = [c for c, _ in non_empty]
        assignments = [g for _, g in non_empty]
        return centers, assignments

    def range_query(self, query: np.ndarray, radius: float) -> List[int]:
        if self.root is None:
            raise ValueError("PM-tree not built")
        query = np.asarray(query, dtype=np.float32)
        self.reset_stats()
        q_pivot_dists = np.linalg.norm(self.pivots - query, axis=1).astype(np.float32)
        out: List[int] = []
        self._range_recursive(self.root, query, radius, q_pivot_dists, out)
        return out

    def _passes_internal_filters(
        self,
        entry: PMInternalEntry,
        query: np.ndarray,
        radius: float,
        q_pivot_dists: np.ndarray,
    ) -> bool:
        # Ball-overlap filter
        if self._dist(query, entry.routing_object) > entry.radius + radius:
            return False
        # Pivot ring filters
        if np.any(q_pivot_dists - radius > entry.hr_max):
            return False
        if np.any(q_pivot_dists + radius < entry.hr_min):
            return False
        return True

    def _range_recursive(
        self,
        node: PMNode,
        query: np.ndarray,
        radius: float,
        q_pivot_dists: np.ndarray,
        out: List[int],
    ) -> None:
        if node.is_leaf:
            for entry in node.entries:
                assert isinstance(entry, PMLeafEntry)
                # pivot filters on leaf point
                if np.any(np.abs(q_pivot_dists - entry.pivot_distances) > radius):
                    continue
                if self._dist(query, entry.point) <= radius:
                    out.append(entry.point_id)
            return

        for entry in node.entries:
            assert isinstance(entry, PMInternalEntry)
            if self._passes_internal_filters(entry, query, radius, q_pivot_dists):
                self._range_recursive(entry.child, query, radius, q_pivot_dists, out)
