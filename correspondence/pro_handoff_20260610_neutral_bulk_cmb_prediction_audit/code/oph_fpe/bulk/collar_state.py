from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from oph_fpe.bulk.cap_geometry import RoundCap, cap_weights, collar_mask


@dataclass(frozen=True)
class CollarPartition:
    inside_mask: np.ndarray
    collar_mask: np.ndarray
    outside_mask: np.ndarray
    cap_weights: np.ndarray
    collar_width: float
    collar_patch_count: int


def cap_collar_partition(points: np.ndarray, cap: RoundCap, collar_width: float | None = None) -> CollarPartition:
    width = float(cap.collar_width if collar_width is None else collar_width)
    cap = RoundCap(axis=cap.axis, theta0=cap.theta0, tangent=cap.tangent, collar_width=width).normalized()
    weights = cap_weights(points, cap, soft=True)
    collar = collar_mask(points, cap)
    hard_inside = cap_weights(points, cap, soft=False).astype(bool)
    inside = hard_inside & ~collar
    outside = (~hard_inside) & ~collar
    return CollarPartition(
        inside_mask=inside,
        collar_mask=collar,
        outside_mask=outside,
        cap_weights=weights,
        collar_width=width,
        collar_patch_count=int(np.sum(collar)),
    )


def visible_packets(state: dict[str, np.ndarray], bins: dict[str, int] | None = None) -> np.ndarray:
    """Encode observer-visible per-node fields into compact integer packet ids."""

    bins = bins or {}
    keys = [
        "record_signature",
        "committed_mask",
        "stable_count",
        "repair_load",
        "s3_class_density",
        "local_mismatch_density",
    ]
    present = [key for key in keys if key in state]
    if not present:
        raise ValueError("visible_packets requires at least one visible field")
    encoded = np.zeros(len(np.asarray(state[present[0]])), dtype=np.int64)
    radix = 1
    for key in present:
        values = np.asarray(state[key])
        if key == "record_signature":
            component = _compress_ints(values.astype(np.int64), max_classes=int(bins.get(key, 64)))
        elif key == "committed_mask":
            component = values.astype(bool).astype(np.int64)
        else:
            component = _bin_float(values.astype(float), int(bins.get(key, 8)))
        base = int(np.max(component)) + 1 if component.size else 1
        encoded += radix * component
        radix *= max(base, 1)
    return encoded


def empirical_packet_distribution(packets: np.ndarray, mask: np.ndarray, *, min_count: int = 1) -> dict[int, float]:
    selected = np.asarray(packets)[np.asarray(mask, dtype=bool)]
    if selected.size == 0:
        return {}
    values, counts = np.unique(selected, return_counts=True)
    if min_count > 1:
        keep = counts >= int(min_count)
        values = values[keep]
        counts = counts[keep]
    total = float(np.sum(counts))
    if total <= 0.0:
        return {}
    return {int(value): float(count / total) for value, count in zip(values, counts)}


def joint_packet_distribution(*packet_arrays: np.ndarray, masks: np.ndarray | None = None, min_count: int = 1) -> dict[tuple[int, ...], float]:
    if not packet_arrays:
        return {}
    arrays = [np.asarray(array, dtype=np.int64) for array in packet_arrays]
    size = min(array.size for array in arrays)
    arrays = [array[:size] for array in arrays]
    if masks is not None:
        mask = np.asarray(masks, dtype=bool)[:size]
        arrays = [array[mask] for array in arrays]
    if not arrays or arrays[0].size == 0:
        return {}
    stacked = np.stack(arrays, axis=1)
    values, counts = np.unique(stacked, axis=0, return_counts=True)
    if min_count > 1:
        keep = counts >= int(min_count)
        values = values[keep]
        counts = counts[keep]
    total = float(np.sum(counts))
    if total <= 0.0:
        return {}
    return {tuple(int(item) for item in row): float(count / total) for row, count in zip(values, counts)}


def entropy_from_distribution(distribution: dict[Any, float]) -> float:
    probs = np.array([value for value in distribution.values() if value > 0.0], dtype=float)
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs)))


def classical_cmi(a_packets: np.ndarray, b_packets: np.ndarray, d_packets: np.ndarray) -> float:
    h_ab = entropy_from_distribution(joint_packet_distribution(a_packets, b_packets))
    h_bd = entropy_from_distribution(joint_packet_distribution(b_packets, d_packets))
    h_b = entropy_from_distribution(empirical_packet_distribution(b_packets, np.ones_like(b_packets, dtype=bool)))
    h_abd = entropy_from_distribution(joint_packet_distribution(a_packets, b_packets, d_packets))
    return max(0.0, float(h_ab + h_bd - h_b - h_abd))


def sector_conditioned_cmi(
    a_packets: np.ndarray,
    b_packets: np.ndarray,
    d_packets: np.ndarray,
    sector_packets: np.ndarray,
) -> dict[str, float]:
    result: dict[str, float] = {}
    names = {0: "identity", 1: "transposition", 2: "threecycle"}
    sectors = np.asarray(sector_packets, dtype=np.int64)
    for sector in np.unique(sectors):
        mask = sectors == sector
        if int(np.sum(mask)) < 3:
            continue
        result[names.get(int(sector), str(int(sector)))] = classical_cmi(a_packets[mask], b_packets[mask], d_packets[mask])
    return result


def fawzi_renner_bound(epsilon_cmi: float) -> float:
    return 2.0 * math.sqrt(max(float(epsilon_cmi), 0.0))


def collar_triplet_packets(
    points: np.ndarray,
    packets: np.ndarray,
    partition: CollarPartition,
    *,
    max_triplets: int = 4096,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inside = np.flatnonzero(partition.inside_mask)
    collar = np.flatnonzero(partition.collar_mask)
    outside = np.flatnonzero(partition.outside_mask)
    if inside.size == 0 or collar.size == 0 or outside.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty, empty
    rng = np.random.default_rng(seed)
    if collar.size > max_triplets:
        collar = np.sort(rng.choice(collar, size=max_triplets, replace=False))
    inside_nearest = _nearest(points, inside, points[collar])
    outside_nearest = _nearest(points, outside, points[collar])
    return packets[inside_nearest], packets[collar], packets[outside_nearest], collar


def _nearest(points: np.ndarray, candidates: np.ndarray, queries: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    tree = cKDTree(points[candidates])
    _, indices = tree.query(queries, k=1)
    return candidates[np.asarray(indices, dtype=np.int64)]


def _compress_ints(values: np.ndarray, max_classes: int) -> np.ndarray:
    max_classes = max(1, int(max_classes))
    _, inverse = np.unique(values, return_inverse=True)
    return (inverse % max_classes).astype(np.int64)


def _bin_float(values: np.ndarray, count: int) -> np.ndarray:
    count = max(1, int(count))
    if count == 1 or values.size == 0:
        return np.zeros_like(values, dtype=np.int64)
    finite = values[np.isfinite(values)]
    if finite.size == 0 or float(np.std(finite)) < 1e-12:
        return np.zeros_like(values, dtype=np.int64)
    quantiles = np.quantile(finite, np.linspace(0.0, 1.0, count + 1)[1:-1])
    return np.searchsorted(quantiles, values, side="right").astype(np.int64)
