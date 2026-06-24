from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from oph_fpe.bulk.cap_geometry import RoundCap, cap_weights
from oph_fpe.constants.oph_pixel import cap_area_planck, cap_entropy_capacity


def observer_view_rows(
    points: np.ndarray,
    *,
    raw_fields: dict[str, np.ndarray],
    visible_fields: dict[str, np.ndarray],
    caps: list[RoundCap],
    times: list[float],
    cell_area_planck: np.ndarray,
    cell_entropy: np.ndarray,
    sample_count: int = 64,
    neighborhood_size: int = 32,
    seed: int = 1,
) -> list[dict[str, Any]]:
    if points.shape[0] == 0:
        return []
    rng = np.random.default_rng(seed)
    count = min(int(sample_count), points.shape[0])
    observer_ids = np.sort(rng.choice(points.shape[0], size=count, replace=False))
    tree = cKDTree(points)
    k = min(max(1, int(neighborhood_size)), points.shape[0])
    _, neighbor_indices = tree.query(points[observer_ids], k=k)
    if neighbor_indices.ndim == 1:
        neighbor_indices = neighbor_indices[:, None]
    rows: list[dict[str, Any]] = []
    for row_index, observer_id in enumerate(observer_ids):
        support = np.asarray(neighbor_indices[row_index], dtype=np.int64)
        support_weight = cell_entropy[support]
        signature_histogram = _signature_histogram(raw_fields.get("record_signature"), support)
        rows.append(
            {
                "view_type": "patch_observer",
                "observer_id": int(observer_id),
                "axis": [float(value) for value in points[observer_id]],
                "support_nodes": [int(value) for value in support],
                "support_patch_count": int(support.size),
                "support_entropy_capacity": float(np.sum(support_weight)),
                "committed_fraction": _weighted_mean(raw_fields.get("committed_mask"), support, support_weight),
                "record_stability_mean": _weighted_mean(raw_fields.get("stable_count"), support, support_weight),
                "modular_depth_mean": _weighted_mean(raw_fields.get("modular_depth"), support, support_weight),
                "modular_depth_std": _weighted_std(raw_fields.get("modular_depth"), support, support_weight),
                "observer_relative_times": [float(value) for value in times],
                "repair_load_mean": _weighted_mean(raw_fields.get("repair_load"), support, support_weight),
                "mismatch_density_mean": _weighted_mean(raw_fields.get("local_mismatch_density"), support, support_weight),
                "visible_signature_entropy": _entropy(raw_fields.get("record_signature"), support),
                "record_signature_histogram": {str(key): value for key, value in signature_histogram.items()},
                "dominant_record_signature": _dominant_histogram_key(signature_histogram),
                "visible_readout_hash": _visible_hash(raw_fields, support),
                "claim_boundary": "observer-facing local readout; hidden representatives are not included",
            }
        )
    for cap_index, cap in enumerate(caps):
        weights = cap_weights(points, cap, soft=True)
        visible_summary = {
            name: {
                "mean": _weighted_full_mean(values, weights * cell_entropy),
                "std": _weighted_full_std(values, weights * cell_entropy),
            }
            for name, values in sorted(visible_fields.items())
        }
        rows.append(
            {
                "view_type": "cap_observer",
                "cap_index": int(cap_index),
                "axis": [float(value) for value in cap.axis],
                "theta0": float(cap.theta0),
                "collar_width": float(cap.collar_width),
                "observer_relative_times": [float(value) for value in times],
                "cap_area_planck": cap_area_planck(weights, cell_area_planck),
                "cap_entropy_capacity": cap_entropy_capacity(weights, cell_entropy),
                "committed_fraction": _weighted_full_mean(raw_fields.get("committed_mask"), weights * cell_entropy),
                "repair_load_mean": _weighted_full_mean(raw_fields.get("repair_load"), weights * cell_entropy),
                "mismatch_density_mean": _weighted_full_mean(raw_fields.get("local_mismatch_density"), weights * cell_entropy),
                "visible_fields": visible_summary,
                "claim_boundary": "cap-local observer-relative readout used by BW branch",
            }
        )
    return rows


def observer_consensus_report(
    points: np.ndarray,
    *,
    raw_fields: dict[str, np.ndarray],
    cell_entropy: np.ndarray,
    sample_count: int = 64,
    neighborhood_size: int = 32,
    seed: int = 1,
    sample_pair_limit: int = 20_000,
) -> dict[str, Any]:
    if points.shape[0] == 0:
        return {"status": "empty"}
    rng = np.random.default_rng(seed)
    count = min(int(sample_count), points.shape[0])
    observer_ids = np.sort(rng.choice(points.shape[0], size=count, replace=False))
    tree = cKDTree(points)
    k = min(max(1, int(neighborhood_size)), points.shape[0])
    _, neighbor_indices = tree.query(points[observer_ids], k=k)
    if neighbor_indices.ndim == 1:
        neighbor_indices = neighbor_indices[:, None]
    signatures = raw_fields.get("record_signature")
    committed = raw_fields.get("committed_mask")
    repair = raw_fields.get("repair_load")
    support_sets = [set(int(value) for value in row) for row in neighbor_indices]
    histograms = [_signature_histogram(signatures, row) for row in neighbor_indices]
    overlap_counts = _observer_overlap_counts(neighbor_indices)
    similarities = np.empty(len(overlap_counts), dtype=float)
    jaccards = np.empty(len(overlap_counts), dtype=float)
    pair_rows: list[dict[str, Any]] = []
    sample_pair_limit = max(0, int(sample_pair_limit))
    for pair_count, ((a, b), overlap_size) in enumerate(sorted(overlap_counts.items())):
        support_a = support_sets[a]
        support_b = support_sets[b]
        union_size = len(support_a) + len(support_b) - int(overlap_size)
        if union_size <= 0:
            continue
        jaccard = float(overlap_size / union_size)
        similarity = _histogram_similarity(histograms[a], histograms[b])
        jaccards[pair_count] = jaccard
        similarities[pair_count] = similarity
        if len(pair_rows) < sample_pair_limit:
            overlap = np.array(sorted(support_a & support_b), dtype=np.int64)
            pair_rows.append(
                {
                    "observer_a": int(observer_ids[a]),
                    "observer_b": int(observer_ids[b]),
                    "overlap_patch_count": int(overlap_size),
                    "jaccard": jaccard,
                    "signature_histogram_similarity": similarity,
                    "overlap_committed_fraction": _weighted_mean(committed, overlap, cell_entropy[overlap]),
                    "overlap_repair_load_mean": _weighted_mean(repair, overlap, cell_entropy[overlap]),
                }
            )
    pair_count = int(len(overlap_counts))
    committed_fraction = _weighted_full_mean(committed, cell_entropy)
    repair_mean = _weighted_full_mean(repair, cell_entropy)
    return {
        "observer_count": int(count),
        "neighborhood_size": int(k),
        "pair_count": int(pair_count),
        "global_committed_fraction": committed_fraction,
        "global_repair_load_mean": repair_mean,
        "median_overlap_jaccard": float(np.median(jaccards)) if jaccards.size else 0.0,
        "median_signature_histogram_similarity": float(np.median(similarities)) if similarities.size else 0.0,
        "p10_signature_histogram_similarity": float(np.percentile(similarities, 10)) if similarities.size else 0.0,
        "sample_pairs": pair_rows,
        "sample_pair_limit": sample_pair_limit,
        "claim_boundary": (
            "objectivity proxy: observer-accessible record-family agreement across overlapping "
            "local views; this is not a bulk-dimension estimator"
        ),
    }


def _weighted_mean(values: np.ndarray | None, indices: np.ndarray, weights: np.ndarray) -> float:
    if values is None or indices.size == 0:
        return 0.0
    return _weighted_full_mean(values[indices], weights)


def _observer_overlap_counts(neighbor_indices: np.ndarray) -> dict[tuple[int, int], int]:
    patch_observers: dict[int, list[int]] = defaultdict(list)
    for observer_index, support in enumerate(np.asarray(neighbor_indices, dtype=np.int64)):
        for patch_index in support:
            patch_observers[int(patch_index)].append(int(observer_index))
    overlap_counts: dict[tuple[int, int], int] = defaultdict(int)
    for observers in patch_observers.values():
        if len(observers) < 2:
            continue
        observers = sorted(set(observers))
        for left_index, observer_a in enumerate(observers[:-1]):
            for observer_b in observers[left_index + 1 :]:
                overlap_counts[(observer_a, observer_b)] += 1
    return dict(overlap_counts)


def _weighted_std(values: np.ndarray | None, indices: np.ndarray, weights: np.ndarray) -> float:
    if values is None or indices.size == 0:
        return 0.0
    return _weighted_full_std(values[indices], weights)


def _weighted_full_mean(values: np.ndarray | None, weights: np.ndarray) -> float:
    if values is None:
        return 0.0
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    if total <= 0.0:
        return float(np.mean(values)) if values.size else 0.0
    return float(np.sum(values * weights) / total)


def _weighted_full_std(values: np.ndarray | None, weights: np.ndarray) -> float:
    if values is None:
        return 0.0
    mean = _weighted_full_mean(values, weights)
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    if total <= 0.0:
        return float(np.std(values)) if values.size else 0.0
    return float(np.sqrt(np.sum(weights * (values - mean) ** 2) / total))


def _entropy(values: np.ndarray | None, indices: np.ndarray) -> float:
    if values is None or indices.size == 0:
        return 0.0
    _, counts = np.unique(values[indices], return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs)))


def _signature_histogram(values: np.ndarray | None, indices: np.ndarray) -> dict[int, float]:
    if values is None or indices.size == 0:
        return {}
    unique, counts = np.unique(values[indices], return_counts=True)
    total = float(counts.sum())
    return {int(key): float(count / total) for key, count in zip(unique, counts, strict=True)}


def _histogram_similarity(left: dict[int, float], right: dict[int, float]) -> float:
    keys = set(left) | set(right)
    distance = sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)
    return float(max(0.0, 1.0 - 0.5 * distance))


def _dominant_histogram_key(histogram: dict[int, float]) -> int | None:
    if not histogram:
        return None
    return int(max(histogram, key=lambda key: histogram[key]))


def _visible_hash(fields: dict[str, np.ndarray], indices: np.ndarray) -> str:
    parts: list[str] = []
    for name in sorted(fields):
        values = np.asarray(fields[name])
        sample = values[indices]
        if np.issubdtype(sample.dtype, np.floating):
            quantized = np.round(sample.astype(float), 6)
            parts.append(f"{name}:{','.join(str(float(value)) for value in quantized)}")
        else:
            parts.append(f"{name}:{','.join(str(int(value)) for value in sample)}")
    import hashlib

    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
