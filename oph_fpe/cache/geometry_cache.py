from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
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
    cache_dir: str | Path | None = None
    _tree: cKDTree | None = field(default=None, init=False, repr=False)
    _transport_maps: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = field(default_factory=dict, init=False, repr=False)
    _points_digest: str | None = field(default=None, init=False, repr=False)
    _disk_hits: int = field(default=0, init=False, repr=False)
    _disk_misses: int = field(default=0, init=False, repr=False)
    _disk_writes: int = field(default=0, init=False, repr=False)
    _memory_hits: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("GeometryCache points must have shape (N, 3)")
        self.points = points
        if self.cache_dir is None:
            self.cache_dir = os.environ.get("OPH_FPE_GEOMETRY_CACHE_DIR") or None
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)

    @property
    def tree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(self.points)
        return self._tree

    def cap_transport_map(self, cap: RoundCap, s: float, *, k: int = 1, cap_id: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        key = self._transport_key(cap, s, k=k, cap_id=cap_id)
        cached = self._transport_maps.get(key)
        if cached is not None:
            self._memory_hits += 1
            return cached
        disk_cached = self._load_transport_map_from_disk(key)
        if disk_cached is not None:
            self._transport_maps[key] = disk_cached
            self._disk_hits += 1
            return disk_cached
        if self.cache_dir is not None:
            self._disk_misses += 1
        mapped = lambda_cap(self.points, cap, float(s))
        indices, weights = interpolation_map(self.points, mapped, k=int(k), tree=self.tree)
        self._transport_maps[key] = (indices, weights)
        self._write_transport_map_to_disk(key, indices, weights)
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
            "memory_hits": int(self._memory_hits),
            "persistent_cache_enabled": self.cache_dir is not None,
            "persistent_cache_dir": str(self.cache_dir) if self.cache_dir is not None else None,
            "persistent_cache_point_digest": self.points_digest if self.cache_dir is not None else None,
            "persistent_cache_disk_hits": int(self._disk_hits),
            "persistent_cache_disk_misses": int(self._disk_misses),
            "persistent_cache_disk_writes": int(self._disk_writes),
            "claim_boundary": "geometry cache only; no physics claim",
        }

    @property
    def points_digest(self) -> str:
        if self._points_digest is None:
            points = np.ascontiguousarray(self.points, dtype=np.float64)
            digest = hashlib.sha256()
            digest.update(b"oph_fpe_geometry_points_v1")
            digest.update(np.asarray(points.shape, dtype=np.int64).tobytes())
            digest.update(points.tobytes())
            self._points_digest = digest.hexdigest()
        return self._points_digest

    def _transport_key(self, cap: RoundCap, s: float, *, k: int, cap_id: int | None) -> tuple[Any, ...]:
        normalized = cap.normalized()
        cap_params: tuple[Any, ...] = (
            "cap",
            _rounded_tuple(normalized.axis),
            round(float(normalized.theta0), 12),
            _rounded_tuple(normalized.tangent),
            round(float(normalized.collar_width), 12),
        )
        if cap_id is not None:
            cap_key: tuple[Any, ...] = ("cap_id", int(cap_id), *cap_params)
        else:
            cap_key = cap_params
        return (*cap_key, round(float(s), 12), int(k))

    def _transport_cache_path(self, key: tuple[Any, ...]) -> Path | None:
        if self.cache_dir is None:
            return None
        key_payload = {
            "version": "oph_fpe_transport_map_v1",
            "points_digest": self.points_digest,
            "point_count": int(self.points.shape[0]),
            "key": key,
        }
        key_bytes = json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(key_bytes).hexdigest()
        return Path(self.cache_dir) / self.points_digest[:16] / f"transport_{digest}.npz"

    def _load_transport_map_from_disk(self, key: tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray] | None:
        path = self._transport_cache_path(key)
        if path is None or not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as payload:
                indices = np.asarray(payload["indices"], dtype=np.int64)
                weights = np.asarray(payload["weights"], dtype=np.float64)
                point_count = int(payload["point_count"])
        except Exception:
            return None
        if point_count != int(self.points.shape[0]) or indices.shape != weights.shape:
            return None
        if indices.ndim != 2 or indices.shape[0] != int(self.points.shape[0]):
            return None
        return indices, weights

    def _write_transport_map_to_disk(self, key: tuple[Any, ...], indices: np.ndarray, weights: np.ndarray) -> None:
        path = self._transport_cache_path(key)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(f".{os.getpid()}.tmp.npz")
            np.savez(
                tmp_path,
                indices=np.asarray(indices, dtype=np.int64),
                weights=np.asarray(weights, dtype=np.float64),
                point_count=np.asarray(int(self.points.shape[0]), dtype=np.int64),
            )
            tmp_path.replace(path)
            self._disk_writes += 1
        except Exception:
            # A cache write failure must never alter a simulation result.
            try:
                tmp_path.unlink(missing_ok=True)  # type: ignore[name-defined]
            except Exception:
                pass


def _rounded_tuple(values: np.ndarray) -> tuple[float, ...]:
    return tuple(round(float(value), 12) for value in np.asarray(values, dtype=float))
