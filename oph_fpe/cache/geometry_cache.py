from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from oph_fpe.bulk.cap_geometry import RoundCap, interpolation_map, lambda_cap


@dataclass
class GeometryCache:
    """Small reusable geometry cache for one fixed screen point set.

    The cache deliberately stores only regulator geometry: KDTree lookups and
    finite lambda_C transport maps. It has no physics receipt semantics by
    itself; callers still decide whether the cached object is used in a proxy,
    demo, or recovered-core calculation.
    """

    points: np.ndarray
    _tree: cKDTree | None = field(default=None, init=False, repr=False)
    _transport_maps: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("GeometryCache points must have shape (N, 3)")
        self.points = points

    @property
    def tree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(self.points)
        return self._tree

    def cap_transport_map(self, cap: RoundCap, s: float, *, k: int = 1, cap_id: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        key = self._transport_key(cap, s, k=k, cap_id=cap_id)
        cached = self._transport_maps.get(key)
        if cached is not None:
            return cached
        mapped = lambda_cap(self.points, cap, float(s))
        indices, weights = interpolation_map(self.points, mapped, k=int(k), tree=self.tree)
        self._transport_maps[key] = (indices, weights)
        return indices, weights

    def transport_support(
        self,
        support: np.ndarray,
        cap: RoundCap,
        s: float,
        *,
        k: int = 1,
        cap_id: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        indices, weights = self.cap_transport_map(cap, s, k=k, cap_id=cap_id)
        support = np.asarray(support, dtype=np.int64)
        support = support[(support >= 0) & (support < self.points.shape[0])]
        if support.size == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=float)
        return indices[support], weights[support]

    def report(self) -> dict[str, Any]:
        return {
            "mode": "finite_screen_geometry_cache",
            "point_count": int(self.points.shape[0]),
            "tree_built": self._tree is not None,
            "transport_map_count": int(len(self._transport_maps)),
            "claim_boundary": "geometry cache only; no physics claim",
        }

    def _transport_key(self, cap: RoundCap, s: float, *, k: int, cap_id: int | None) -> tuple[Any, ...]:
        normalized = cap.normalized()
        if cap_id is not None:
            cap_key: tuple[Any, ...] = ("cap_id", int(cap_id))
        else:
            cap_key = (
                "cap",
                _rounded_tuple(normalized.axis),
                round(float(normalized.theta0), 12),
                _rounded_tuple(normalized.tangent),
                round(float(normalized.collar_width), 12),
            )
        return (*cap_key, round(float(s), 12), int(k))


def _rounded_tuple(values: np.ndarray) -> tuple[float, ...]:
    return tuple(round(float(value), 12) for value in np.asarray(values, dtype=float))
