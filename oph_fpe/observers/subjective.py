from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from oph_fpe.evidence.hashes import canonical_json_bytes
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

from oph_fpe.bulk.cap_geometry import RoundCap, cap_weights
from oph_fpe.constants.oph_pixel import cap_area_planck, cap_entropy_capacity


LOCALITY_PACKET_FIELDS = (
    "repair_load",
    "cumulative_repair_load",
)
LOCALITY_PACKET_BINS = 8


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
    edge_left: np.ndarray | None = None,
    edge_right: np.ndarray | None = None,
    overlap_correspondence_max_observers: int | None = None,
) -> list[dict[str, Any]]:
    if points.shape[0] == 0:
        return []
    rng = np.random.default_rng(seed)
    count = min(int(sample_count), points.shape[0])
    observer_ids = np.sort(rng.choice(points.shape[0], size=count, replace=False))
    k = min(max(1, int(neighborhood_size)), points.shape[0])
    if edge_left is not None and edge_right is not None:
        neighbor_indices = _carrier_graph_neighborhoods(
            observer_ids,
            edge_left=np.asarray(edge_left, dtype=np.int64),
            edge_right=np.asarray(edge_right, dtype=np.int64),
            patch_count=points.shape[0],
            neighborhood_size=k,
        )
        support_selection_carrier = "finite_patch_adjacency_bfs"
    else:
        tree = cKDTree(points)
        _, coordinate_neighbors = tree.query(points[observer_ids], k=k)
        if coordinate_neighbors.ndim == 1:
            coordinate_neighbors = coordinate_neighbors[:, None]
        neighbor_indices = [np.asarray(row, dtype=np.int64) for row in coordinate_neighbors]
        support_selection_carrier = "cKDTree(screen_points)_diagnostic_fallback"
    rows: list[dict[str, Any]] = []
    locality_schema = _locality_packet_schema(raw_fields, bins=LOCALITY_PACKET_BINS)
    locality_vectors: list[np.ndarray] = []
    for row_index, observer_id in enumerate(observer_ids):
        support = np.asarray(neighbor_indices[row_index], dtype=np.int64)
        support_weight = cell_entropy[support]
        signature_histogram = _signature_histogram(raw_fields.get("record_signature"), support)
        locality_vector = _locality_packet_feature_vector(
            raw_fields,
            support,
            locality_schema,
        )
        locality_vectors.append(locality_vector)
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
                "locality_preserving_packet_feature_vector": locality_vector.tolist(),
                "locality_preserving_packet_feature_schema": {
                    "version": "observer_local_packet_midrank_v1",
                    "bin_count": LOCALITY_PACKET_BINS,
                    "fields": [str(item["field"]) for item in locality_schema],
                    "per_field_features": [
                        "ecdf_histogram",
                        "ecdf_mean",
                        "ecdf_std",
                        "ecdf_q25",
                        "ecdf_median",
                        "ecdf_q75",
                    ],
                    "excluded_hash_fields": ["record_signature", "visible_readout_hash"],
                    "feature_value_coordinate_fields_used": [],
                    "support_selection_carrier": support_selection_carrier,
                    "support_selection_carrier_ancestry": (
                        ["s2_screen_point_delaunay_patch_adjacency"]
                        if support_selection_carrier == "finite_patch_adjacency_bfs"
                        else ["s2_screen_coordinates"]
                    ),
                    "support_node_ids_exported_for_replay": True,
                    "strict_neutral_eligible": False,
                    "diagnostic_reason": (
                        "the production finite-patch adjacency is constructed from the S2 screen chart; "
                        "local values are gauge invariant, but support selection is not yet an independently "
                        "produced chart-blind carrier"
                    ),
                },
                "claim_boundary": (
                    "observer-facing local readout; support-node indices are exported only as replay metadata "
                    "and are not embedded as geometry feature values"
                ),
            }
        )
    _attach_measured_overlap_correspondences(
        rows,
        observer_ids=observer_ids,
        neighbor_indices=neighbor_indices,
        locality_vectors=locality_vectors,
        support_selection_carrier=support_selection_carrier,
        max_observers=overlap_correspondence_max_observers,
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


def _locality_packet_schema(
    raw_fields: dict[str, np.ndarray],
    *,
    bins: int,
) -> list[dict[str, Any]]:
    """Build a run-global, monotone calibration for local packet features.

    Hash labels are intentionally excluded. Each admitted scalar is mapped to
    its midrank in the run-global empirical distribution, so nearby values stay
    nearby and every observer uses the same chart-free binning.
    """

    schema: list[dict[str, Any]] = []
    for field in LOCALITY_PACKET_FIELDS:
        values = raw_fields.get(field)
        if values is None:
            continue
        array = np.asarray(values, dtype=float).reshape(-1)
        finite = np.sort(array[np.isfinite(array)])
        if finite.size == 0:
            continue
        schema.append(
            {
                "field": str(field),
                "sorted_finite_values": finite,
                "bins": max(2, int(bins)),
            }
        )
    return schema


def _locality_packet_feature_vector(
    raw_fields: dict[str, np.ndarray],
    support: np.ndarray,
    schema: list[dict[str, Any]],
) -> np.ndarray:
    support = np.asarray(support, dtype=np.int64).reshape(-1)
    blocks: list[np.ndarray] = []
    for item in schema:
        field = str(item["field"])
        bins = int(item["bins"])
        reference = np.asarray(item["sorted_finite_values"], dtype=float)
        values = np.asarray(raw_fields[field], dtype=float).reshape(-1)
        valid_support = support[(support >= 0) & (support < values.size)]
        local = values[valid_support]
        local = local[np.isfinite(local)]
        if local.size == 0 or reference.size == 0:
            blocks.append(np.zeros(bins + 5, dtype=float))
            continue
        left = np.searchsorted(reference, local, side="left")
        right = np.searchsorted(reference, local, side="right")
        midranks = (left.astype(float) + right.astype(float)) / (2.0 * float(reference.size))
        midranks = np.clip(midranks, 0.0, 1.0)
        histogram, _ = np.histogram(midranks, bins=bins, range=(0.0, 1.0))
        histogram = histogram.astype(float) / max(float(histogram.sum()), 1.0)
        moments = np.asarray(
            [
                float(np.mean(midranks)),
                float(np.std(midranks)),
                float(np.quantile(midranks, 0.25)),
                float(np.quantile(midranks, 0.50)),
                float(np.quantile(midranks, 0.75)),
            ],
            dtype=float,
        )
        blocks.append(np.concatenate([histogram, moments]))
    return np.concatenate(blocks) if blocks else np.zeros(0, dtype=float)


def _attach_measured_overlap_correspondences(
    rows: list[dict[str, Any]],
    *,
    observer_ids: np.ndarray,
    neighbor_indices: list[np.ndarray],
    locality_vectors: list[np.ndarray],
    support_selection_carrier: str,
    max_observers: int | None,
) -> None:
    """Attach literal cross-observer support intersections without node IDs.

    The peer identifier is only an index used to assemble the symmetric
    correspondence graph. It is not hashed into a feature coordinate. A global
    relabeling therefore permutes rows and columns together and leaves the
    measured geometry unchanged.
    """

    patch_rows = [row for row in rows if row.get("view_type") == "patch_observer"]
    if not patch_rows:
        return
    analysis_indices = deterministic_observer_analysis_indices(
        observer_ids,
        max_observers=max_observers,
    )
    analysis_neighbors = [neighbor_indices[int(index)] for index in analysis_indices]
    support_sets = [set(int(value) for value in support) for support in analysis_neighbors]
    correspondences: list[list[dict[str, Any]]] = [[] for _ in patch_rows]
    for (a, b), overlap_size in sorted(_observer_overlap_counts(analysis_neighbors).items()):
        support_a = support_sets[int(a)]
        support_b = support_sets[int(b)]
        union_size = len(support_a) + len(support_b) - int(overlap_size)
        if overlap_size <= 0 or union_size <= 0:
            continue
        row_a = int(analysis_indices[int(a)])
        row_b = int(analysis_indices[int(b)])
        vector_a = np.asarray(locality_vectors[row_a], dtype=float)
        vector_b = np.asarray(locality_vectors[row_b], dtype=float)
        width = max(vector_a.size, vector_b.size, 1)
        if vector_a.size != vector_b.size:
            padded_a = np.zeros(width, dtype=float)
            padded_b = np.zeros(width, dtype=float)
            padded_a[: vector_a.size] = vector_a
            padded_b[: vector_b.size] = vector_b
            vector_a, vector_b = padded_a, padded_b
        packet_distance = float(np.linalg.norm(vector_a - vector_b) / np.sqrt(float(width)))
        packet_similarity = float(np.exp(-packet_distance))
        jaccard = float(overlap_size / union_size)
        affinity = float(jaccard * packet_similarity)
        common = {
            "overlap_patch_count": int(overlap_size),
            "jaccard": jaccard,
            "local_packet_similarity": packet_similarity,
            "measured_affinity": affinity,
        }
        correspondences[row_a].append(
            {
                **common,
                "peer_observer_id": int(observer_ids[row_b]),
                "support_fraction_self": float(overlap_size / max(1, len(support_a))),
                "support_fraction_peer": float(overlap_size / max(1, len(support_b))),
            }
        )
        correspondences[row_b].append(
            {
                **common,
                "peer_observer_id": int(observer_ids[row_a]),
                "support_fraction_self": float(overlap_size / max(1, len(support_b))),
                "support_fraction_peer": float(overlap_size / max(1, len(support_a))),
            }
        )
    included_indices = set(int(value) for value in analysis_indices)
    materialized_count = len(patch_rows)
    analyzed_count = len(analysis_indices)
    sampling_policy = (
        "all_materialized_observers"
        if analyzed_count == materialized_count
        else "deterministic_observer_id_hash_rank_v1"
    )
    for index, row in enumerate(patch_rows):
        evidence = sorted(correspondences[index], key=lambda item: int(item["peer_observer_id"]))
        row["measured_overlap_correspondences"] = evidence
        row["measured_overlap_correspondence_schema"] = "literal_support_intersection_v1"
        row["measured_overlap_correspondence_count"] = len(evidence)
        row["measured_overlap_correspondence_receipt"] = bool(evidence)
        row["overlap_correspondence_analysis"] = {
            "included": index in included_indices,
            "materialized_observer_count": int(materialized_count),
            "analyzed_observer_count": int(analyzed_count),
            "max_observers": (
                int(max_observers) if max_observers is not None else None
            ),
            "sampling_policy": sampling_policy,
        }
        row["overlap_correspondence_evidence_provenance"] = {
            "producer": "observer_view_rows.literal_support_intersection",
            "cross_observer_measurement": True,
            "self_histogram_synthesis": False,
            "support_node_ids_exported": True,
            "support_node_ids_used_as_feature_values": False,
            "support_selection_carrier": str(support_selection_carrier),
            "support_selection_carrier_ancestry": (
                ["s2_screen_point_delaunay_patch_adjacency"]
                if support_selection_carrier == "finite_patch_adjacency_bfs"
                else ["s2_screen_coordinates"]
            ),
            "observer_id_role": "graph_index_only",
            "strict_neutral_eligible": False,
            "bounded_analysis_population": analyzed_count < materialized_count,
            "diagnostic_reason": (
                "literal overlaps are measured correctly, but production supports are selected on an "
                "S2-derived patch adjacency and therefore cannot establish emergent chart-blind geometry"
            ),
        }


def deterministic_observer_analysis_indices(
    observer_ids: np.ndarray | list[int],
    *,
    max_observers: int | None,
) -> np.ndarray:
    """Return a deterministic, nested observer-population subsample.

    The rank is a fixed SplitMix64 permutation of the materialized observer ID.
    Selecting the lowest ranks makes smaller analysis caps strict subsets of
    larger caps while avoiding the screen-latitude bias of first-row slicing.
    Full-population behavior is unchanged whenever the cap is absent or large
    enough to include every materialized observer.
    """

    ids = np.asarray(observer_ids, dtype=np.int64).reshape(-1)
    count = int(ids.size)
    if max_observers is None or int(max_observers) >= count:
        return np.arange(count, dtype=np.int64)
    limit = max(0, int(max_observers))
    if limit == 0:
        return np.zeros(0, dtype=np.int64)
    with np.errstate(over="ignore"):
        rank = ids.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        rank = (rank ^ (rank >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        rank = (rank ^ (rank >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        rank = rank ^ (rank >> np.uint64(31))
    order = np.lexsort((np.arange(count, dtype=np.int64), rank))
    return np.sort(order[:limit].astype(np.int64, copy=False))


def _carrier_graph_neighborhoods(
    observer_ids: np.ndarray,
    *,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    patch_count: int,
    neighborhood_size: int,
) -> list[np.ndarray]:
    if edge_left.shape != edge_right.shape:
        raise ValueError("carrier graph endpoint arrays must have matching shape")
    if np.any(edge_left < 0) or np.any(edge_right < 0):
        raise ValueError("carrier graph endpoints must be nonnegative")
    rows = np.concatenate([edge_left, edge_right])
    cols = np.concatenate([edge_right, edge_left])
    adjacency = csr_matrix(
        (np.ones(rows.size, dtype=np.uint8), (rows, cols)),
        shape=(int(patch_count), int(patch_count)),
    )
    adjacency.sum_duplicates()
    adjacency.sort_indices()
    target = min(max(1, int(neighborhood_size)), int(patch_count))
    neighborhoods: list[np.ndarray] = []
    for observer_id in np.asarray(observer_ids, dtype=np.int64):
        visited = {int(observer_id)}
        ordered = [int(observer_id)]
        frontier = [int(observer_id)]
        while frontier and len(ordered) < target:
            next_frontier: list[int] = []
            for node in frontier:
                start, stop = int(adjacency.indptr[node]), int(adjacency.indptr[node + 1])
                for neighbor in adjacency.indices[start:stop]:
                    value = int(neighbor)
                    if value in visited:
                        continue
                    visited.add(value)
                    ordered.append(value)
                    next_frontier.append(value)
                    if len(ordered) >= target:
                        break
                if len(ordered) >= target:
                    break
            frontier = next_frontier
        neighborhoods.append(np.asarray(ordered, dtype=np.int64))
    return neighborhoods


def observer_consensus_report(
    points: np.ndarray,
    *,
    raw_fields: dict[str, np.ndarray],
    cell_entropy: np.ndarray,
    sample_count: int = 64,
    neighborhood_size: int = 32,
    seed: int = 1,
    sample_pair_limit: int = 20_000,
    analysis_max_observers: int | None = None,
) -> dict[str, Any]:
    if points.shape[0] == 0:
        return {"status": "empty"}
    rng = np.random.default_rng(seed)
    materialized_count = min(int(sample_count), points.shape[0])
    materialized_observer_ids = np.sort(
        rng.choice(points.shape[0], size=materialized_count, replace=False)
    )
    analysis_indices = deterministic_observer_analysis_indices(
        materialized_observer_ids,
        max_observers=analysis_max_observers,
    )
    observer_ids = materialized_observer_ids[analysis_indices]
    count = int(observer_ids.size)
    if count == 0:
        return {
            "status": "analysis_population_empty",
            "observer_count": 0,
            "requested_observer_count": int(sample_count),
            "materialized_observer_count": int(materialized_count),
            "analyzed_observer_count": 0,
            "analysis_max_observers": (
                int(analysis_max_observers)
                if analysis_max_observers is not None
                else None
            ),
            "analysis_sampling_policy": "deterministic_observer_id_hash_rank_v1",
            "analysis_observer_ids": [],
            "pair_count": 0,
            "sample_pairs": [],
            "sample_pair_limit": max(0, int(sample_pair_limit)),
            "claim_boundary": (
                "No observer-wide consensus analysis was performed; materialized observer "
                "rows may still exist, so this report cannot certify population consensus."
            ),
        }
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
        "requested_observer_count": int(sample_count),
        "materialized_observer_count": int(materialized_count),
        "analyzed_observer_count": int(count),
        "analysis_max_observers": (
            int(analysis_max_observers)
            if analysis_max_observers is not None
            else None
        ),
        "analysis_sampling_policy": (
            "all_materialized_observers"
            if count == materialized_count
            else "deterministic_observer_id_hash_rank_v1"
        ),
        "analysis_observer_ids": [int(value) for value in observer_ids],
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


def _observer_overlap_counts(neighbor_indices: list[np.ndarray] | np.ndarray) -> dict[tuple[int, int], int]:
    patch_observers: dict[int, list[int]] = defaultdict(list)
    for observer_index, support in enumerate(neighbor_indices):
        for patch_index in np.asarray(support, dtype=np.int64).reshape(-1):
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
    visible_fields: dict[str, np.ndarray] = {}
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    for name in sorted(fields):
        values = np.asarray(fields[name])
        if values.ndim != 1 or indices.size == 0 or int(np.max(indices)) >= values.size:
            continue
        if not (
            np.issubdtype(values.dtype, np.number)
            or np.issubdtype(values.dtype, np.bool_)
        ):
            continue
        sample = values[indices]
        if np.issubdtype(sample.dtype, np.floating):
            visible_fields[name] = np.round(sample.astype(np.float64), 6)
        else:
            visible_fields[name] = np.asarray(sample)
    import hashlib

    return hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": "oph_visible_readout_hash_v2",
                "fields": visible_fields,
            }
        )
    ).hexdigest()
