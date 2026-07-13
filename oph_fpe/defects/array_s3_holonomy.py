from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter
from typing import TypeAlias, TypeVar

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from oph_fpe.finite_groups import S3_CLASS, S3_ELEMENTS, S3_INDEX, S3_INV, S3_MUL


TriangleEdgeLookup: TypeAlias = tuple[np.ndarray, np.ndarray]
T = TypeVar("T")

DEFAULT_TIMELINE_MAX_SNAPSHOT_CLUSTERS = 64
DEFAULT_TIMELINE_MAX_WORLDLINES = 256
DEFAULT_TIMELINE_MAX_EVENTS_PER_WORLDLINE = 32
DEFAULT_TIMELINE_MAX_SUPPORT_NODES_PER_RECORD = 32
DEFECT_TIMELINE_SCHEMA = "oph_s3_defect_timeline_v2"


def _stable_index_priority(value: int) -> int:
    """Stable SplitMix-style priority used for spatially fair truncation."""

    mask = (1 << 64) - 1
    mixed = (int(value) + 0x9E3779B97F4A7C15) & mask
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & mask
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & mask
    return (mixed ^ (mixed >> 31)) & mask


def s3_edge_class_density(
    left: np.ndarray,
    right: np.ndarray,
    gauge: np.ndarray,
    patch_count: int,
    class_id: int | None = None,
) -> np.ndarray:
    classes = S3_CLASS[gauge.astype(np.int64)]
    if class_id is None:
        weights = (classes != 0).astype(float)
    else:
        weights = (classes == int(class_id)).astype(float)
    counts = np.bincount(left, weights=weights, minlength=patch_count) + np.bincount(
        right, weights=weights, minlength=patch_count
    )
    degree = np.bincount(np.concatenate([left, right]), minlength=patch_count)
    return counts / np.maximum(degree, 1)


def s3_class_counts(gauge: np.ndarray) -> dict[str, int]:
    classes = S3_CLASS[gauge.astype(np.int64)]
    return {
        "identity": int(np.sum(classes == 0)),
        "transposition": int(np.sum(classes == 1)),
        "threecycle": int(np.sum(classes == 2)),
    }


def oriented_triangles(
    points: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    max_triangles: int | None = None,
) -> np.ndarray:
    triangles, _truncated = _oriented_triangles_with_receipt(
        points,
        left,
        right,
        max_triangles=max_triangles,
    )
    return triangles


def _oriented_triangles_with_receipt(
    points: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    max_triangles: int | None,
) -> tuple[np.ndarray, bool]:
    """Return the stable triangle sample and whether the topology exceeded it.

    A bounded scan observes one additional triangle before stopping. This keeps
    the old sample exactly unchanged while distinguishing a complete topology
    from a prefix that merely happens to have reached ``max_triangles``.
    """

    triangle_limit = max(0, int(max_triangles)) if max_triangles is not None else None
    adjacency: dict[int, set[int]] = {}
    for a, b in zip(np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64), strict=False):
        adjacency.setdefault(int(a), set()).add(int(b))
        adjacency.setdefault(int(b), set()).add(int(a))
    triangles: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    # KNN edges are ordered by patch index.  Truncating that traversal sampled
    # only one geographical band of the screen.  A stable mixed-index order
    # preserves deterministic runs while spreading a bounded sample over the
    # full screen.
    for i in sorted(adjacency, key=_stable_index_priority):
        neighbors = adjacency[i]
        for j in sorted(neighbors, key=_stable_index_priority):
            if i >= j:
                continue
            common = neighbors & adjacency.get(j, set())
            for k in sorted(common, key=_stable_index_priority):
                key = tuple(sorted((i, j, k)))
                if key in seen:
                    continue
                seen.add(key)
                tri = _orient_triangle(points, (i, j, int(k)))
                triangles.append(tri)
                if triangle_limit is not None and len(triangles) > triangle_limit:
                    return (
                        np.asarray(triangles[:triangle_limit], dtype=np.int64).reshape((-1, 3)),
                        True,
                    )
    return np.asarray(triangles, dtype=np.int64).reshape((-1, 3)), False


def s3_triangle_holonomy(g_ij: np.ndarray | int, g_jk: np.ndarray | int, g_ki: np.ndarray | int) -> np.ndarray:
    first = S3_MUL[np.asarray(g_ij, dtype=np.int64), np.asarray(g_jk, dtype=np.int64)]
    return S3_MUL[first, np.asarray(g_ki, dtype=np.int64)]


def triangle_holonomies(
    triangles: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    gauge: np.ndarray,
    *,
    edge_lookup: TriangleEdgeLookup | None = None,
) -> np.ndarray:
    edge_indices, forward = (
        edge_lookup
        if edge_lookup is not None
        else _triangle_edge_index_lookup(triangles, left, right)
    )
    gauge = np.asarray(gauge, dtype=np.int64)
    if edge_indices.size == 0:
        return np.zeros(0, dtype=np.int16)
    labels = gauge[edge_indices]
    labels = np.where(forward, labels, S3_INV[labels])
    first = S3_MUL[labels[:, 0], labels[:, 1]]
    return np.asarray(S3_MUL[first, labels[:, 2]], dtype=np.int16)


def defect_class(holonomy: np.ndarray | int) -> np.ndarray:
    return S3_CLASS[np.asarray(holonomy, dtype=np.int64)]


def array_holonomy_report(
    points: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    gauge: np.ndarray,
    *,
    max_triangles: int | None = 10_000,
    precomputed_triangles: np.ndarray | None = None,
    precomputed_edge_lookup: TriangleEdgeLookup | None = None,
    triangle_sampling_truncated: bool | None = None,
) -> dict[str, object]:
    topology_reused = precomputed_triangles is not None
    if precomputed_triangles is None:
        triangles, triangle_sampling_truncated = _oriented_triangles_with_receipt(
            points,
            left,
            right,
            max_triangles=max_triangles,
        )
    else:
        triangles = np.asarray(precomputed_triangles, dtype=np.int64)
    holonomies = (
        triangle_holonomies(
            triangles,
            left,
            right,
            gauge,
            edge_lookup=precomputed_edge_lookup,
        )
        if triangles.size
        else np.zeros(0, dtype=np.int16)
    )
    classes = defect_class(holonomies) if holonomies.size else np.zeros(0, dtype=np.int16)
    clusters = cluster_defects(triangles, classes, points, holonomies=holonomies) if triangles.size else []
    worldlines = track_defect_worldlines([], clusters)
    class_counts = {
        "identity": int(np.sum(classes == 0)),
        "transposition": int(np.sum(classes == 1)),
        "threecycle": int(np.sum(classes == 2)),
    }
    return {
        "mode": "array_s3_screen_holonomy",
        "triangle_count": int(triangles.shape[0]),
        "max_triangles": int(max_triangles) if max_triangles is not None else None,
        "triangle_sampling_truncated": triangle_sampling_truncated,
        "triangle_topology_complete": bool(triangle_sampling_truncated is False),
        "fixed_topology_reused": bool(topology_reused),
        "class_counts": class_counts,
        "defect_triangle_count": int(np.sum(classes != 0)),
        "defect_fraction": float(np.mean(classes != 0)) if classes.size else 0.0,
        "cluster_count": len(clusters),
        "clusters": clusters,
        "worldlines": worldlines,
        "claim_boundary": (
            "screen/collar S3 holonomy defect clusters only; not matter particles until mapped into "
            "a controlled neutral 3D bulk reconstruction"
        ),
    }


def defect_timeline_report(
    points: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    gauge_snapshots: list[tuple[int, np.ndarray]],
    *,
    max_triangles: int | None = 5_000,
    persistence_cycles: int = 3,
    max_angular_speed_per_cycle: float = 0.75,
    max_analysis_clusters_per_snapshot: int | None = None,
    max_serialized_snapshots: int | None = None,
    max_snapshot_clusters_per_snapshot: int | None = None,
    max_worldlines: int | None = None,
    max_events_per_worldline: int | None = None,
    max_support_nodes_per_record: int | None = None,
) -> dict[str, object]:
    total_started = perf_counter()
    snapshot_cluster_cap = _normalize_detail_cap(
        max_snapshot_clusters_per_snapshot,
        name="max_snapshot_clusters_per_snapshot",
    )
    worldline_cap = _normalize_detail_cap(max_worldlines, name="max_worldlines")
    event_cap = _normalize_detail_cap(max_events_per_worldline, name="max_events_per_worldline")
    support_cap = _normalize_detail_cap(
        max_support_nodes_per_record,
        name="max_support_nodes_per_record",
    )
    analysis_cluster_cap = _normalize_detail_cap(
        max_analysis_clusters_per_snapshot,
        name="max_analysis_clusters_per_snapshot",
    )
    serialized_snapshot_cap = _normalize_detail_cap(
        max_serialized_snapshots,
        name="max_serialized_snapshots",
    )
    serialized_snapshot_indices = set(
        _uniform_sample_indices(len(gauge_snapshots), serialized_snapshot_cap)
    )

    topology_started = perf_counter()
    fixed_triangles, triangle_sampling_truncated = _oriented_triangles_with_receipt(
        points,
        left,
        right,
        max_triangles=max_triangles,
    )
    fixed_edge_lookup = _triangle_edge_index_lookup(fixed_triangles, left, right)
    topology_seconds = perf_counter() - topology_started

    snapshots: list[dict[str, object]] = []
    active_worldlines: dict[str, dict[str, object]] = {}
    completed: list[dict[str, object]] = []
    next_worldline = 0
    previous_clusters: list[dict[str, object]] = []
    previous_ids: list[str] = []
    previous_cycle: int | None = None
    holonomy_cluster_seconds = 0.0
    worldline_assignment_seconds = 0.0
    output_shaping_seconds = 0.0
    snapshot_cluster_records_total = 0
    snapshot_cluster_records_emitted = 0
    snapshot_support_node_ids_total = 0
    snapshot_support_node_ids_emitted = 0
    cluster_analysis_records_total = 0
    cluster_analysis_records_analyzed = 0
    for snapshot_index, (cycle, gauge) in enumerate(gauge_snapshots):
        snapshot_compute_started = perf_counter()
        report = array_holonomy_report(
            points,
            left,
            right,
            gauge,
            max_triangles=max_triangles,
            precomputed_triangles=fixed_triangles,
            precomputed_edge_lookup=fixed_edge_lookup,
            triangle_sampling_truncated=triangle_sampling_truncated,
        )
        full_clusters = list(report.get("clusters", []))
        clusters = _bounded_analysis_clusters(full_clusters, analysis_cluster_cap)
        cluster_analysis_records_total += len(full_clusters)
        cluster_analysis_records_analyzed += len(clusters)
        holonomy_cluster_seconds += perf_counter() - snapshot_compute_started
        assignment_started = perf_counter()
        assignments, next_worldline = _assign_worldlines(
            previous_clusters,
            previous_ids,
            clusters,
            next_worldline,
            cycle_delta=max(1, int(cycle) - int(previous_cycle)) if previous_cycle is not None else 1,
            max_angular_speed_per_cycle=max_angular_speed_per_cycle,
        )
        worldline_assignment_seconds += perf_counter() - assignment_started
        seen: set[str] = set()
        for cluster, worldline_id, event, distance in assignments:
            seen.add(worldline_id)
            entry = active_worldlines.setdefault(
                worldline_id,
                {
                    "worldline_id": worldline_id,
                    "events": [],
                    "birth_cycle": int(cycle),
                },
            )
            entry["events"].append(
                {
                    "cycle": int(cycle),
                    "cluster_id": cluster.get("cluster_id"),
                    "event": event,
                    "class": cluster.get("class"),
                    "holonomy_mode": cluster.get("holonomy_mode"),
                    "inverse_holonomy_mode": cluster.get("inverse_holonomy_mode"),
                    "support_node_count": int(cluster.get("support_node_count", 0)),
                    "support_nodes": list(cluster.get("support_nodes", [])),
                    "support_size": int(cluster.get("support_size", 0)),
                    "centroid": cluster.get("centroid", [0.0, 0.0, 1.0]),
                    "transport_distance": distance,
                }
            )
        ended = [worldline_id for worldline_id in active_worldlines if worldline_id not in seen and worldline_id in previous_ids]
        for worldline_id in ended:
            completed.append(_finalize_worldline(active_worldlines.pop(worldline_id), persistence_cycles))

        shaping_started = perf_counter()
        emit_snapshot = snapshot_index in serialized_snapshot_indices
        emitted_assignments = (
            _bounded_prefix(assignments, snapshot_cluster_cap) if emit_snapshot else []
        )
        emitted_clusters = [
            _timeline_cluster_output(cluster, worldline_id, support_cap=support_cap)
            for cluster, worldline_id, _event, _distance in emitted_assignments
        ]
        snapshot_cluster_records_total += len(assignments)
        snapshot_cluster_records_emitted += len(emitted_clusters)
        snapshot_support_node_ids_total += sum(
            len(cluster.get("support_nodes", []) or [])
            for cluster, _worldline_id, _event, _distance in assignments
        )
        snapshot_support_node_ids_emitted += sum(
            len(cluster.get("support_nodes", []) or []) for cluster in emitted_clusters
        )
        if emit_snapshot:
            snapshots.append(
                {
                    "cycle": int(cycle),
                    "triangle_count": report.get("triangle_count"),
                    "defect_triangle_count": report.get("defect_triangle_count"),
                    "cluster_count": len(full_clusters),
                    "clusters_analyzed_count": len(clusters),
                    "cluster_analysis_complete": len(clusters) == len(full_clusters),
                    "clusters_emitted_count": len(emitted_clusters),
                    "cluster_records_complete": len(emitted_clusters) == len(assignments),
                    "clusters": emitted_clusters,
                }
            )
        output_shaping_seconds += perf_counter() - shaping_started
        previous_clusters = clusters
        previous_ids = [worldline_id for _cluster, worldline_id, _event, _distance in assignments]
        previous_cycle = int(cycle)
    completed.extend(_finalize_worldline(row, persistence_cycles) for row in active_worldlines.values())
    persistent = [row for row in completed if row.get("persistent")]

    final_shaping_started = perf_counter()
    emitted_worldlines = [
        _timeline_worldline_output(
            row,
            event_cap=event_cap,
            support_cap=support_cap,
        )
        for row in _bounded_prefix(completed, worldline_cap)
    ]
    worldline_events_total = sum(len(row.get("events", []) or []) for row in completed)
    worldline_events_emitted = sum(len(row.get("events", []) or []) for row in emitted_worldlines)
    worldline_support_node_ids_total = sum(
        len(event.get("support_nodes", []) or [])
        for row in completed
        for event in (row.get("events", []) or [])
    )
    worldline_support_node_ids_emitted = sum(
        len(event.get("support_nodes", []) or [])
        for row in emitted_worldlines
        for event in (row.get("events", []) or [])
    )
    detail_counts = {
        "snapshots_total": len(gauge_snapshots),
        "snapshots_emitted": len(snapshots),
        "snapshot_cluster_records_total": int(snapshot_cluster_records_total),
        "snapshot_cluster_records_emitted": int(snapshot_cluster_records_emitted),
        "worldlines_total": len(completed),
        "worldlines_emitted": len(emitted_worldlines),
        "worldline_events_total": int(worldline_events_total),
        "worldline_events_emitted": int(worldline_events_emitted),
        "support_node_id_occurrences_total": int(
            snapshot_support_node_ids_total + worldline_support_node_ids_total
        ),
        "support_node_id_occurrences_emitted": int(
            snapshot_support_node_ids_emitted + worldline_support_node_ids_emitted
        ),
    }
    completeness = {
        "snapshots_complete": len(snapshots) == len(gauge_snapshots),
        "snapshot_cluster_records_complete": (
            snapshot_cluster_records_emitted == snapshot_cluster_records_total
        ),
        "worldlines_complete": len(emitted_worldlines) == len(completed),
        "worldline_events_complete": worldline_events_emitted == worldline_events_total,
        "support_node_ids_complete": (
            detail_counts["support_node_id_occurrences_emitted"]
            == detail_counts["support_node_id_occurrences_total"]
        ),
    }
    output_detail_complete = all(bool(value) for value in completeness.values())
    completeness["output_detail_complete"] = output_detail_complete
    truncation_reasons = [name for name, complete in completeness.items() if name != "output_detail_complete" and not complete]
    if triangle_sampling_truncated:
        truncation_reasons.insert(0, "triangle_topology_sample_truncated")
    cluster_analysis_complete = cluster_analysis_records_analyzed == cluster_analysis_records_total
    if not cluster_analysis_complete:
        truncation_reasons.insert(0, "cluster_analysis_truncated")
    particle_promotion_inputs_complete = bool(
        output_detail_complete
        and cluster_analysis_complete
        and not triangle_sampling_truncated
    )
    output_shaping_seconds += perf_counter() - final_shaping_started
    total_seconds = perf_counter() - total_started
    return {
        "schema": DEFECT_TIMELINE_SCHEMA,
        "mode": "array_s3_defect_timeline",
        "patch_count": int(points.shape[0]),
        "snapshot_count": len(gauge_snapshots),
        "snapshots_emitted_count": len(snapshots),
        "snapshot_selection": (
            "all_snapshots"
            if len(snapshots) == len(gauge_snapshots)
            else "uniform_cycle_order_sample_including_endpoints"
        ),
        "max_triangles": int(max_triangles) if max_triangles is not None else None,
        "triangle_count": int(fixed_triangles.shape[0]),
        "triangle_sampling_truncated": bool(triangle_sampling_truncated),
        "triangle_topology_complete": not bool(triangle_sampling_truncated),
        "fixed_topology_cache": {
            "triangles_prepared_once": True,
            "edge_lookup_prepared_once": True,
            "snapshot_reuse_count": len(gauge_snapshots),
        },
        "cluster_analysis": {
            "selection": "stable_mixed_support_anchor_priority",
            "max_clusters_per_snapshot": analysis_cluster_cap,
            "cluster_records_total": int(cluster_analysis_records_total),
            "cluster_records_analyzed": int(cluster_analysis_records_analyzed),
            "complete": cluster_analysis_complete,
        },
        "worldline_assignment": "minimum_total_intrinsic_transport_cost",
        "max_angular_speed_per_cycle": float(max_angular_speed_per_cycle),
        "motion_gate_cadence_scaled": True,
        "snapshots": snapshots,
        "worldlines": emitted_worldlines,
        "worldline_count": len(completed),
        "persistent_worldline_count": len(persistent),
        "max_observation_count": max((int(row.get("observation_count", 0)) for row in completed), default=0),
        "max_lifetime_cycles": max((int(row.get("lifetime_cycles", 0)) for row in completed), default=0),
        "persistent_worldline_precursor_diagnostic": bool(persistent),
        "persistent_worldline_precursor_receipt": bool(
            persistent and particle_promotion_inputs_complete
        ),
        "output_detail_limits": {
            "max_serialized_snapshots": serialized_snapshot_cap,
            "max_snapshot_clusters_per_snapshot": snapshot_cluster_cap,
            "max_worldlines": worldline_cap,
            "max_events_per_worldline": event_cap,
            "max_support_nodes_per_record": support_cap,
        },
        "output_detail_counts": detail_counts,
        "output_detail_completeness": completeness,
        "output_truncated": not output_detail_complete,
        "analysis_or_output_truncated": bool(
            triangle_sampling_truncated
            or not cluster_analysis_complete
            or not output_detail_complete
        ),
        "truncation_reasons": truncation_reasons,
        "particle_promotion_inputs_complete": particle_promotion_inputs_complete,
        "stage_timings_seconds": {
            "fixed_topology_preparation": float(topology_seconds),
            "snapshot_holonomy_and_clustering": float(holonomy_cluster_seconds),
            "worldline_assignment": float(worldline_assignment_seconds),
            "output_shaping": float(output_shaping_seconds),
            "total": float(total_seconds),
        },
        "particle_matter_receipt": False,
        "claim_boundary": (
            "time-resolved screen/collar S3 holonomy defect clusters under declared finite repair dynamics. "
            "Persistent worldlines are particle precursors only; particle claims require localization, "
            "transport, fusion/scattering, neutral-bulk mapping, repeated-seed controls, and complete "
            "untruncated promotion inputs."
        ),
    }


def _normalize_detail_cap(value: int | None, *, name: str) -> int | None:
    if value is None:
        return None
    cap = int(value)
    if cap < 0:
        raise ValueError(f"{name} must be nonnegative or None")
    return cap


def _timeline_particle_inputs_complete(timeline_report: dict[str, object] | None) -> bool:
    if not timeline_report:
        return False
    if timeline_report.get("mode") == "array_s3_defect_timeline":
        return bool(
            timeline_report.get("schema") == DEFECT_TIMELINE_SCHEMA
            and timeline_report.get("particle_promotion_inputs_complete") is True
        )
    return bool(timeline_report.get("particle_promotion_inputs_complete", True) is True)


def _bounded_prefix(rows: list[T], cap: int | None) -> list[T]:
    return list(rows) if cap is None else list(rows[:cap])


def _bounded_analysis_clusters(
    clusters: list[dict[str, object]],
    cap: int | None,
) -> list[dict[str, object]]:
    if cap is None or cap >= len(clusters):
        return list(clusters)

    def priority(cluster: dict[str, object]) -> tuple[object, ...]:
        support = sorted(int(value) for value in (cluster.get("support_nodes", []) or []))
        anchor = support[0] if support else 0
        return (_stable_index_priority(anchor), _cluster_match_key(cluster))

    return sorted(clusters, key=priority)[:cap]


def _uniform_sample_indices(count: int, cap: int | None) -> list[int]:
    count = max(0, int(count))
    if cap is None or cap >= count:
        return list(range(count))
    if cap <= 0 or count == 0:
        return []
    if cap == 1:
        return [0]
    return [round(index * (count - 1) / (cap - 1)) for index in range(cap)]


def _bounded_support_record(record: dict[str, object], *, support_cap: int | None) -> dict[str, object]:
    result = dict(record)
    support_nodes = [int(value) for value in (record.get("support_nodes", []) or [])]
    emitted_support = _bounded_prefix(support_nodes, support_cap)
    result["support_node_count"] = int(record.get("support_node_count", len(support_nodes)))
    result["support_nodes"] = emitted_support
    result["support_nodes_emitted_count"] = len(emitted_support)
    result["support_nodes_complete"] = len(emitted_support) == len(support_nodes)
    return result


def _timeline_cluster_output(
    cluster: dict[str, object],
    worldline_id: str,
    *,
    support_cap: int | None,
) -> dict[str, object]:
    return _bounded_support_record(
        {
            "cluster_id": cluster.get("cluster_id"),
            "worldline_id": worldline_id,
            "class": cluster.get("class"),
            "holonomy_mode": cluster.get("holonomy_mode"),
            "inverse_holonomy_mode": cluster.get("inverse_holonomy_mode"),
            "support_node_count": cluster.get("support_node_count"),
            "support_nodes": list(cluster.get("support_nodes", []) or []),
            "centroid": cluster.get("centroid"),
        },
        support_cap=support_cap,
    )


def _timeline_worldline_output(
    worldline: dict[str, object],
    *,
    event_cap: int | None,
    support_cap: int | None,
) -> dict[str, object]:
    result = {key: value for key, value in worldline.items() if key != "events"}
    events = list(worldline.get("events", []) or [])
    selected_indices = _uniform_sample_indices(len(events), event_cap)
    result["events"] = [
        _bounded_support_record(dict(events[index]), support_cap=support_cap)
        for index in selected_indices
    ]
    result["events_emitted_count"] = len(selected_indices)
    result["events_complete"] = len(selected_indices) == len(events)
    result["event_selection"] = (
        "all_cycle_ordered_events"
        if len(selected_indices) == len(events)
        else "uniform_cycle_order_sample_including_endpoints"
    )
    return result


def particle_likeness_report(
    timeline_report: dict[str, object],
    interaction_report: dict[str, object] | None = None,
    *,
    bulk_localization_pass: bool = False,
    max_support_fraction: float = 0.05,
    min_observations: int = 3,
    min_class_stability: float = 0.8,
) -> dict[str, object]:
    """Score screen holonomy worldlines against particle-like criteria.

    This is deliberately conservative. A defect can be persistent and localized
    on the screen, but it is not promoted to a matter-particle receipt until
    neutral-bulk mapping plus transport/fusion/scattering controls are present.
    """

    patch_count = max(1, int(timeline_report.get("patch_count", 1) if timeline_report else 1))
    timeline_inputs_complete = _timeline_particle_inputs_complete(timeline_report)
    transport_pass_by_id = {
        str(row.get("worldline_id")): bool(row.get("screen_transport_proxy_pass", False))
        for row in (interaction_report or {}).get("worldlines", [])
    }
    fusion_proxy = bool((interaction_report or {}).get("fusion_conservation_proxy_pass", False))
    fusion_gauge_covariant = bool(
        (interaction_report or {}).get("fusion_gauge_covariant_receipt") is True
        and (interaction_report or {}).get("fusion_common_basepoint_transport_receipt") is True
    )
    fusion_pass_by_id: dict[str, bool] = {
        str(value): True
        for value in (interaction_report or {}).get("fusion_conserving_worldline_ids", [])
    }
    for candidate in (interaction_report or {}).get("fusion_candidates", []):
        if not (
            bool(candidate.get("identity_product", False))
            and bool(candidate.get("encounter_geometry_verified", False))
        ):
            continue
        for key in ("left_worldline_id", "right_worldline_id"):
            value = candidate.get(key)
            if value is not None:
                fusion_pass_by_id[str(value)] = True
    scattering_proxy = bool((interaction_report or {}).get("scattering_reproducibility_proxy_pass", False))
    rows: list[dict[str, object]] = []
    for worldline in timeline_report.get("worldlines", []) if timeline_report else []:
        events = list(worldline.get("events", []))
        worldline_id = str(worldline.get("worldline_id"))
        classes = [event.get("class") for event in events if event.get("class") is not None]
        support_counts = np.asarray([int(event.get("support_node_count", 0)) for event in events], dtype=float)
        class_mode, class_fraction = _class_mode_fraction(classes)
        max_support = int(np.max(support_counts)) if support_counts.size else 0
        localized = bool(max_support / patch_count <= float(max_support_fraction))
        persistent = bool(int(worldline.get("observation_count", len(events))) >= int(min_observations))
        class_stable = bool(class_fraction >= float(min_class_stability))
        transportable = bool(transport_pass_by_id.get(worldline_id, False))
        fusion_proxy_pass = bool(fusion_proxy and fusion_pass_by_id.get(worldline_id, False))
        fusion_conserving = bool(fusion_proxy_pass and fusion_gauge_covariant)
        particle_like = bool(
            timeline_inputs_complete
            and bulk_localization_pass
            and localized
            and persistent
            and class_stable
            and transportable
            and fusion_proxy_pass
            and scattering_proxy
        )
        rows.append(
            {
                "worldline_id": worldline_id,
                "observation_count": int(worldline.get("observation_count", len(events))),
                "lifetime_cycles": int(worldline.get("lifetime_cycles", 0)),
                "class_mode": class_mode,
                "class_stability_fraction": float(class_fraction),
                "max_support_node_count": max_support,
                "max_support_fraction": float(max_support / patch_count),
                "mean_transport_distance": float(worldline.get("mean_transport_distance", 0.0)),
                "localization_pass": localized,
                "persistence_pass": persistent,
                "sector_stability_pass": class_stable,
                "transportability_pass": transportable,
                "fusion_conservation_pass": fusion_conserving,
                "fusion_conservation_proxy_pass": fusion_proxy_pass,
                "fusion_gauge_covariant_receipt": fusion_gauge_covariant,
                "scattering_reproducibility_pass": scattering_proxy,
                "bulk_localization_pass": bool(bulk_localization_pass),
                "particle_like": particle_like,
            }
        )
    localized_count = sum(bool(row["localization_pass"]) for row in rows)
    persistent_count = sum(bool(row["persistence_pass"]) for row in rows)
    sector_count = sum(bool(row["sector_stability_pass"]) for row in rows)
    transport_count = sum(bool(row["transportability_pass"]) for row in rows)
    fusion_count = sum(bool(row["fusion_conservation_pass"]) for row in rows)
    scattering_count = sum(bool(row["scattering_reproducibility_pass"]) for row in rows)
    particle_like_count = sum(bool(row["particle_like"]) for row in rows)
    detector_positive_count = sum(
        bool(
            timeline_inputs_complete
            and row["bulk_localization_pass"]
            and row["localization_pass"]
            and row["persistence_pass"]
            and row["sector_stability_pass"]
            and row["transportability_pass"]
            and row["fusion_conservation_proxy_pass"]
            and row["scattering_reproducibility_pass"]
        )
        for row in rows
    )
    return {
        "mode": "screen_holonomy_particle_likeness_diagnostic",
        "worldline_count": len(rows),
        "localized_count": int(localized_count),
        "persistent_count": int(persistent_count),
        "sector_stable_count": int(sector_count),
        "transportable_count": int(transport_count),
        "fusion_conserving_count": int(fusion_count),
        "fusion_gauge_covariant_receipt": fusion_gauge_covariant,
        "scattering_reproducible_count": int(scattering_count),
        "bulk_localization_pass": bool(bulk_localization_pass),
        "max_support_fraction_threshold": float(max_support_fraction),
        "min_observations": int(min_observations),
        "min_class_stability": float(min_class_stability),
        "interaction_report_mode": (interaction_report or {}).get("mode"),
        "timeline_particle_promotion_inputs_complete": timeline_inputs_complete,
        "timeline_truncation_reasons": list(timeline_report.get("truncation_reasons", []))
        if timeline_report
        else [],
        "particle_like_count": int(particle_like_count),
        "particle_detector_positive_receipt": bool(detector_positive_count > 0),
        "particle_matter_receipt": bool(
            timeline_inputs_complete
            and particle_like_count > 0
            and bulk_localization_pass
            and fusion_gauge_covariant
        ),
        "worldlines": rows[:256],
        "claim_boundary": (
            "screen holonomy defect particle-likeness score. Persistence/localization/sector "
            "stability may be present, but matter-particle claims require neutral 3D bulk mapping, "
            "transport around contractible paths, gauge-covariant common-basepoint fusion, scattering "
            "reproducibility, and repeated-seed controls. Raw products of holonomies based at distinct "
            "screen locations remain a detector proxy and cannot raise a matter theorem receipt."
        ),
    }


def defect_interaction_report(
    timeline_report: dict[str, object],
    *,
    min_observations: int = 3,
    min_class_stability: float = 0.8,
    min_transport_distance: float = 1e-9,
    min_scattering_transitions: int = 2,
    max_fusion_angular_distance: float = 0.35,
    fusion_nearest_per_cluster: int = 4,
) -> dict[str, object]:
    """Measure screen-local defect transport/fusion/scattering proxies.

    This is not a matter-particle receipt. It only turns the previously
    unimplemented interaction gates into explicit finite S3 screen diagnostics.
    A physical particle claim still requires neutral/H3 bulk localization and
    repeated transport/fusion/scattering controls.
    """

    timeline_inputs_complete = _timeline_particle_inputs_complete(timeline_report)
    rows: list[dict[str, object]] = []
    transition_counts: dict[str, int] = {}
    for worldline in timeline_report.get("worldlines", []) if timeline_report else []:
        events = list(worldline.get("events", []))
        classes = [event.get("class") for event in events if event.get("class") is not None]
        class_mode, class_fraction = _class_mode_fraction(classes)
        transport_distances = [
            float(event["transport_distance"])
            for event in events
            if event.get("transport_distance") is not None
            and float(event.get("transport_distance", 0.0)) > float(min_transport_distance)
        ]
        for before, after in zip(events, events[1:], strict=False):
            before_class = before.get("class")
            after_class = after.get("class")
            if before_class is not None and after_class is not None:
                key = f"{before_class}->{after_class}"
                transition_counts[key] = transition_counts.get(key, 0) + 1
        persistent = bool(int(worldline.get("observation_count", len(events))) >= int(min_observations))
        class_stable = bool(class_fraction >= float(min_class_stability))
        transport_diagnostic = bool(persistent and class_stable and transport_distances)
        transport_proxy = bool(timeline_inputs_complete and transport_diagnostic)
        rows.append(
            {
                "worldline_id": worldline.get("worldline_id"),
                "observation_count": int(worldline.get("observation_count", len(events))),
                "class_mode": class_mode,
                "class_stability_fraction": float(class_fraction),
                "transport_event_count": len(transport_distances),
                "mean_transport_distance": float(np.mean(transport_distances)) if transport_distances else 0.0,
                "screen_transport_diagnostic_positive": transport_diagnostic,
                "screen_transport_proxy_pass": transport_proxy,
            }
        )

    fusion_cutoff = float(np.clip(float(max_fusion_angular_distance), 0.0, np.pi))
    fusion_neighbor_count = max(1, int(fusion_nearest_per_cluster))
    fusion_candidates = _fusion_candidates_from_snapshots(
        timeline_report,
        max_angular_distance=fusion_cutoff,
        nearest_per_cluster=fusion_neighbor_count,
    )
    identity_fusion_count = sum(bool(row["identity_product"]) for row in fusion_candidates)
    verified_fusion_candidates = [
        row for row in fusion_candidates if bool(row.get("encounter_geometry_verified", False))
    ]
    verified_identity_fusion_count = sum(bool(row["identity_product"]) for row in verified_fusion_candidates)
    fusion_conserving_worldline_ids = sorted(
        {
            str(candidate[key])
            for candidate in fusion_candidates
            if bool(candidate.get("identity_product", False))
            and bool(candidate.get("encounter_geometry_verified", False))
            for key in ("left_worldline_id", "right_worldline_id")
            if candidate.get(key) is not None
        }
    )
    fusion_identity_fraction = (
        float(identity_fusion_count / len(fusion_candidates)) if fusion_candidates else 0.0
    )
    total_transitions = sum(transition_counts.values())
    dominant_transition_fraction = (
        max(transition_counts.values()) / total_transitions if total_transitions else 0.0
    )
    scattering_diagnostic = bool(
        total_transitions >= int(min_scattering_transitions)
        and dominant_transition_fraction >= float(min_class_stability)
    )
    scattering_proxy = bool(timeline_inputs_complete and scattering_diagnostic)
    transportable_count = sum(bool(row["screen_transport_proxy_pass"]) for row in rows)
    return {
        "mode": "screen_s3_defect_interaction_diagnostic",
        "worldline_count": len(rows),
        "screen_transport_proxy_count": int(transportable_count),
        "fusion_candidate_count": len(fusion_candidates),
        "fusion_identity_candidate_count": int(identity_fusion_count),
        "fusion_identity_fraction": fusion_identity_fraction,
        "fusion_geometrically_verified_candidate_count": len(verified_fusion_candidates),
        "fusion_verified_identity_candidate_count": int(verified_identity_fusion_count),
        "fusion_legacy_unverified_candidate_count": len(fusion_candidates) - len(verified_fusion_candidates),
        "fusion_encounter_angular_cutoff": fusion_cutoff,
        "fusion_nearest_per_cluster": fusion_neighbor_count,
        "fusion_conserving_worldline_ids": fusion_conserving_worldline_ids,
        "fusion_conservation_proxy_pass": bool(
            timeline_inputs_complete
            and verified_fusion_candidates
            and verified_identity_fusion_count == len(verified_fusion_candidates)
        ),
        "fusion_common_basepoint_transport_receipt": False,
        "fusion_gauge_covariant_receipt": False,
        "fusion_theorem_receipt": False,
        "scattering_transition_counts": transition_counts,
        "scattering_transition_total": int(total_transitions),
        "dominant_scattering_transition_fraction": float(dominant_transition_fraction),
        "scattering_reproducibility_diagnostic_positive": scattering_diagnostic,
        "scattering_reproducibility_proxy_pass": scattering_proxy,
        "timeline_particle_promotion_inputs_complete": timeline_inputs_complete,
        "timeline_truncation_reasons": list(timeline_report.get("truncation_reasons", []))
        if timeline_report
        else [],
        "interaction_proxy_receipt": bool(
            timeline_inputs_complete and transportable_count and scattering_proxy
        ),
        "particle_matter_receipt": False,
        "worldlines": rows[:256],
        "fusion_candidates": fusion_candidates[:256],
        "claim_boundary": (
            "screen-local S3 interaction proxy. It measures transport-like sector persistence, "
            "intrinsic-near same-snapshot inverse-holonomy encounter candidates, and class-transition stability. "
            "Legacy candidates without centroids remain unverified and cannot satisfy the fusion proxy. "
            "Candidate holonomies at distinct locations have not been parallel-transported to a common "
            "basepoint, so the product is not a gauge-covariant fusion theorem. It is not a matter-particle "
            "or 3D-bulk claim without neutral/H3 localization and contractible-path transport, "
            "gauge-covariant fusion, and scattering controls."
        ),
    }


def cluster_defects(
    triangles: np.ndarray,
    classes: np.ndarray,
    points: np.ndarray,
    holonomies: np.ndarray | None = None,
) -> list[dict[str, object]]:
    triangles = np.asarray(triangles, dtype=np.int64)
    classes = np.asarray(classes, dtype=np.int64)
    holonomies_arr = np.asarray(holonomies, dtype=np.int64) if holonomies is not None else None
    active = np.flatnonzero(classes != 0)
    if active.size == 0:
        return []
    centroids = _normalize_rows(np.mean(points[triangles[active]], axis=1))
    tree = cKDTree(centroids)
    visited: set[int] = set()
    clusters: list[dict[str, object]] = []
    if centroids.shape[0] > 1:
        nearest = np.asarray(tree.query(centroids, k=2)[0], dtype=float)[:, 1]
        positive = nearest[np.isfinite(nearest) & (nearest > 0.0)]
        radius = float(np.median(positive) * 1.25) if positive.size else 1.0e-9
    else:
        radius = 1.0e-9
    radius = max(radius, 1.0e-9)
    for local_index, _triangle_index in enumerate(active):
        if local_index in visited:
            continue
        # Connected components of the radius graph are disjoint and transitive.
        # The former one-hop query could include a triangle already assigned to
        # a previous cluster, duplicating membership across defect clusters.
        component: list[int] = []
        queue = [int(local_index)]
        visited.add(int(local_index))
        while queue:
            member = queue.pop()
            component.append(member)
            for neighbor_raw in tree.query_ball_point(centroids[member], r=radius):
                neighbor = int(neighbor_raw)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        local_members = np.asarray(sorted(component), dtype=np.int64)
        member_indices = active[local_members]
        member_classes = classes[member_indices]
        majority = int(np.bincount(member_classes).argmax())
        if holonomies_arr is not None and holonomies_arr.size:
            member_holonomies = holonomies_arr[member_indices]
            holonomy_mode = int(np.bincount(member_holonomies, minlength=len(S3_ELEMENTS)).argmax())
        else:
            holonomy_mode = None
        support_nodes = np.unique(triangles[member_indices].reshape(-1))
        clusters.append(
            {
                "cluster_id": f"defect_cluster_{len(clusters):06d}",
                "triangle_indices": [int(value) for value in member_indices],
                "support_nodes": [int(value) for value in support_nodes],
                "class": _class_name(majority),
                "holonomy_mode": holonomy_mode,
                "inverse_holonomy_mode": int(S3_INV[holonomy_mode]) if holonomy_mode is not None else None,
                "support_size": int(member_indices.size),
                "support_node_count": int(support_nodes.size),
                "centroid": [float(value) for value in _normalize_vector(np.mean(centroids[local_members], axis=0))],
            }
        )
    return clusters


def _assign_worldlines(
    previous_clusters: list[dict[str, object]],
    previous_ids: list[str],
    current_clusters: list[dict[str, object]],
    next_worldline: int,
    *,
    cycle_delta: int = 1,
    max_angular_speed_per_cycle: float = 0.75,
) -> tuple[list[tuple[dict[str, object], str, str, float | None]], int]:
    if not previous_clusters:
        birth_ids, next_worldline = _deterministic_birth_ids(
            current_clusters,
            range(len(current_clusters)),
            next_worldline=next_worldline,
            used_ids=set(),
        )
        assignments = [
            (cluster, birth_ids[index], "birth", None)
            for index, cluster in enumerate(current_clusters)
        ]
        return assignments, next_worldline
    if len(previous_ids) != len(previous_clusters):
        raise ValueError("previous worldline ids must align with previous defect clusters")
    previous_centroids = np.asarray([cluster.get("centroid", [0.0, 0.0, 1.0]) for cluster in previous_clusters], dtype=float)
    current_centroids = np.asarray([cluster.get("centroid", [0.0, 0.0, 1.0]) for cluster in current_clusters], dtype=float)
    if current_centroids.size == 0:
        return [], next_worldline
    distances = _spherical_distance_matrix(previous_centroids, current_centroids)
    scale = _matching_scale(
        cycle_delta=cycle_delta,
        max_angular_speed_per_cycle=max_angular_speed_per_cycle,
    )
    matched = _global_compatible_cluster_matches(
        previous_clusters,
        current_clusters,
        distances=distances,
        max_distance=scale,
    )
    unmatched = [index for index in range(len(current_clusters)) if index not in matched]
    birth_ids, next_worldline = _deterministic_birth_ids(
        current_clusters,
        unmatched,
        next_worldline=next_worldline,
        used_ids=set(previous_ids),
    )

    assignments: list[tuple[dict[str, object], str, str, float | None]] = []
    for index, cluster in enumerate(current_clusters):
        match = matched.get(index)
        if match is not None:
            previous_index, distance = match
            assignments.append((cluster, previous_ids[previous_index], "continue", distance))
        else:
            assignments.append((cluster, birth_ids[index], "birth", None))
    return assignments, next_worldline


def _global_compatible_cluster_matches(
    previous_clusters: list[dict[str, object]],
    current_clusters: list[dict[str, object]],
    *,
    distances: np.ndarray,
    max_distance: float,
) -> dict[int, tuple[int, float]]:
    """Deterministic minimum-total-cost one-to-one transport assignment."""

    if not previous_clusters or not current_clusters:
        return {}
    previous_order = sorted(
        range(len(previous_clusters)),
        key=lambda index: _cluster_match_key(previous_clusters[index]),
    )
    current_order = sorted(
        range(len(current_clusters)),
        key=lambda index: _cluster_match_key(current_clusters[index]),
    )
    cutoff = float(max_distance)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        return {}
    invalid_cost = max(1.0e6, cutoff * 1.0e6)
    cost = np.full((len(previous_order), len(current_order)), invalid_cost, dtype=float)
    valid = np.zeros_like(cost, dtype=bool)
    for ordered_previous, previous_index in enumerate(previous_order):
        previous = previous_clusters[previous_index]
        for ordered_current, current_index in enumerate(current_order):
            current = current_clusters[current_index]
            distance = float(distances[previous_index, current_index])
            if (
                np.isfinite(distance)
                and distance <= cutoff
                and _cluster_sector_compatible(previous, current)
            ):
                cost[ordered_previous, ordered_current] = distance
                valid[ordered_previous, ordered_current] = True

    assigned_rows, assigned_columns = linear_sum_assignment(cost)
    matches: dict[int, tuple[int, float]] = {}
    for ordered_previous, ordered_current in zip(assigned_rows, assigned_columns, strict=True):
        if not bool(valid[ordered_previous, ordered_current]):
            continue
        previous_index = int(previous_order[int(ordered_previous)])
        current_index = int(current_order[int(ordered_current)])
        distance = float(distances[previous_index, current_index])
        matches[current_index] = (previous_index, float(distance))
    return matches


def _deterministic_birth_ids(
    clusters: list[dict[str, object]],
    indices: Iterable[int],
    *,
    next_worldline: int,
    used_ids: set[str],
) -> tuple[dict[int, str], int]:
    """Allocate birth IDs by stable cluster content, preserving supplied IDs."""

    ordered = sorted((int(index) for index in indices), key=lambda index: _cluster_match_key(clusters[index]))
    result: dict[int, str] = {}
    for index in ordered:
        preferred = clusters[index].get("worldline_id")
        preferred_id = str(preferred) if preferred is not None and str(preferred) else None
        if preferred_id is not None and preferred_id not in used_ids:
            worldline_id = preferred_id
            prefix = "worldline_"
            suffix = worldline_id[len(prefix) :] if worldline_id.startswith(prefix) else ""
            if suffix.isdigit():
                next_worldline = max(int(next_worldline), int(suffix) + 1)
        else:
            while f"worldline_{int(next_worldline):06d}" in used_ids:
                next_worldline += 1
            worldline_id = f"worldline_{int(next_worldline):06d}"
            next_worldline += 1
        used_ids.add(worldline_id)
        result[index] = worldline_id
    return result, int(next_worldline)


def _matching_scale(*, cycle_delta: int, max_angular_speed_per_cycle: float) -> float:
    """Cadence-aware intrinsic motion bound, independent of cluster density."""

    delta = int(cycle_delta)
    speed = float(max_angular_speed_per_cycle)
    if delta <= 0:
        raise ValueError("cycle_delta must be positive")
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("max_angular_speed_per_cycle must be finite and positive")
    return float(min(np.pi, speed * delta))


def _finalize_worldline(row: dict[str, object], persistence_cycles: int) -> dict[str, object]:
    events = list(row.get("events", []))
    cycles = [int(event.get("cycle", 0)) for event in events]
    distances = [
        float(event["transport_distance"])
        for event in events
        if event.get("transport_distance") is not None
    ]
    support_counts = [int(event.get("support_node_count", 0)) for event in events]
    lifetime = int(max(cycles) - min(cycles)) if cycles else 0
    result = {
        "worldline_id": row.get("worldline_id"),
        "birth_cycle": int(min(cycles)) if cycles else int(row.get("birth_cycle", 0)),
        "death_cycle": int(max(cycles)) if cycles else int(row.get("birth_cycle", 0)),
        "lifetime_cycles": lifetime,
        "observation_count": len(events),
        "mean_transport_distance": float(np.mean(distances)) if distances else 0.0,
        "max_support_node_count": max(support_counts) if support_counts else 0,
        "events": events,
    }
    result["persistent"] = bool(len(events) >= int(persistence_cycles))
    return result


def track_defect_worldlines(previous_clusters: list[dict[str, object]], current_clusters: list[dict[str, object]]) -> list[dict[str, object]]:
    if not current_clusters:
        return []
    previous_id_map, next_worldline = _deterministic_birth_ids(
        previous_clusters,
        range(len(previous_clusters)),
        next_worldline=0,
        used_ids=set(),
    )
    previous_ids = [previous_id_map[index] for index in range(len(previous_clusters))]
    assignments, _next_worldline = _assign_worldlines(
        previous_clusters,
        previous_ids,
        current_clusters,
        next_worldline,
    )
    previous_by_worldline = {
        previous_ids[index]: previous_clusters[index]
        for index in range(len(previous_clusters))
    }
    result: list[dict[str, object]] = []
    for current, worldline_id, event, distance in assignments:
        previous = previous_by_worldline.get(worldline_id) if event == "continue" else None
        result.append(
            {
                "worldline_id": worldline_id,
                "previous_cluster_id": previous.get("cluster_id") if previous is not None else None,
                "current_cluster_id": current["cluster_id"],
                "event": event,
                "transport_distance": distance,
            }
        )
    return result


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 1.0e-15:
        return np.asarray([0.0, 0.0, 1.0], dtype=float)
    return values / norm


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=float)
    if rows.size == 0:
        return rows.reshape((-1, 3))
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    normalized = rows / np.maximum(norms, 1.0e-15)
    invalid = (~np.isfinite(norms[:, 0])) | (norms[:, 0] <= 1.0e-15)
    if np.any(invalid):
        normalized[invalid] = np.asarray([0.0, 0.0, 1.0])
    return normalized


def _spherical_distance_matrix(left_points: np.ndarray, right_points: np.ndarray) -> np.ndarray:
    """Intrinsic unit-sphere distances between two centroid collections."""

    left_unit = _normalize_rows(left_points)
    right_unit = _normalize_rows(right_points)
    if left_unit.size == 0 or right_unit.size == 0:
        return np.zeros((left_unit.shape[0], right_unit.shape[0]), dtype=float)
    cosine = np.clip(left_unit @ right_unit.T, -1.0, 1.0)
    return np.arccos(cosine)


def _cluster_sector_compatible(previous: dict[str, object], current: dict[str, object]) -> bool:
    previous_class = previous.get("class")
    current_class = current.get("class")
    return bool(previous_class is None or current_class is None or previous_class == current_class)


def _cluster_match_key(cluster: dict[str, object]) -> tuple[object, ...]:
    """Stable content key used only to break matching/allocation ties."""

    support = tuple(sorted(int(value) for value in (cluster.get("support_nodes") or [])))
    centroid_raw = np.asarray(cluster.get("centroid", [0.0, 0.0, 1.0]), dtype=float).reshape(-1)
    centroid = tuple(round(float(value), 12) for value in centroid_raw)
    holonomy = cluster.get("holonomy_mode")
    return (
        str(cluster.get("worldline_id") or ""),
        str(cluster.get("cluster_id") or ""),
        str(cluster.get("class") or ""),
        int(holonomy) if holonomy is not None else -1,
        support,
        centroid,
    )


def _orient_triangle(points: np.ndarray, triangle: tuple[int, int, int]) -> tuple[int, int, int]:
    i, j, k = triangle
    normal = np.cross(points[j] - points[i], points[k] - points[i])
    outward = points[i] + points[j] + points[k]
    return (i, j, k) if float(np.dot(normal, outward)) >= 0.0 else (i, k, j)


def _oriented_edge_labels(left: np.ndarray, right: np.ndarray, gauge: np.ndarray) -> dict[tuple[int, int], int]:
    labels: dict[tuple[int, int], int] = {}
    for a, b, value in zip(
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(gauge, dtype=np.int64),
        strict=False,
    ):
        labels[(int(a), int(b))] = int(value)
        labels[(int(b), int(a))] = int(S3_INV[int(value)])
    return labels


def _triangle_edge_index_lookup(
    triangles: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> TriangleEdgeLookup:
    """Resolve the three directed edges of every fixed triangle once."""

    triangles_array = np.asarray(triangles, dtype=np.int64).reshape((-1, 3))
    left_array = np.asarray(left, dtype=np.int64)
    right_array = np.asarray(right, dtype=np.int64)
    if triangles_array.size == 0:
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3), dtype=bool)
    if left_array.size != right_array.size:
        raise ValueError("left and right edge arrays must have equal length")
    max_node = int(
        max(
            np.max(triangles_array, initial=0),
            np.max(left_array, initial=0),
            np.max(right_array, initial=0),
        )
    )
    stride = max_node + 1
    edge_keys = np.minimum(left_array, right_array) * stride + np.maximum(left_array, right_array)
    order = np.argsort(edge_keys, kind="stable")
    sorted_keys = edge_keys[order]
    if sorted_keys.size == 0:
        raise KeyError("triangle topology has no corresponding graph edges")

    sources = triangles_array[:, [0, 1, 2]]
    targets = triangles_array[:, [1, 2, 0]]
    query_keys = np.minimum(sources, targets) * stride + np.maximum(sources, targets)
    positions = np.searchsorted(sorted_keys, query_keys)
    in_bounds = positions < sorted_keys.size
    safe_positions = np.minimum(positions, max(0, sorted_keys.size - 1))
    found = in_bounds & (sorted_keys[safe_positions] == query_keys)
    if not np.all(found):
        row, slot = (int(value) for value in np.argwhere(~found)[0])
        raise KeyError(
            f"triangle edge ({int(sources[row, slot])}, {int(targets[row, slot])}) is absent"
        )
    edge_indices = order[positions]
    forward = (left_array[edge_indices] == sources) & (right_array[edge_indices] == targets)
    reverse = (left_array[edge_indices] == targets) & (right_array[edge_indices] == sources)
    if not np.all(forward | reverse):
        raise ValueError("resolved triangle edge orientation is inconsistent")
    return np.asarray(edge_indices, dtype=np.int64), np.asarray(forward, dtype=bool)


def _class_name(class_id: int) -> str:
    return {0: "identity", 1: "transposition", 2: "threecycle"}.get(int(class_id), str(int(class_id)))


def _fusion_candidates_from_snapshots(
    timeline_report: dict[str, object],
    *,
    max_angular_distance: float,
    nearest_per_cluster: int,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for snapshot in timeline_report.get("snapshots", []) if timeline_report else []:
        clusters = [
            cluster
            for cluster in snapshot.get("clusters", [])
            if cluster.get("holonomy_mode") is not None
        ]
        for left_index, right_index, centroid_distance, geometry_verified in _snapshot_encounter_pairs(
            clusters,
            max_angular_distance=max_angular_distance,
            nearest_per_cluster=nearest_per_cluster,
        ):
            left_cluster = clusters[left_index]
            left_h = int(left_cluster.get("holonomy_mode"))
            right_cluster = clusters[right_index]
            right_h = int(right_cluster.get("holonomy_mode"))
            product_lr = int(S3_MUL[left_h, right_h])
            product_rl = int(S3_MUL[right_h, left_h])
            # In a group, an inverse encounter is identity in both orders.
            # Keep nonconserving encounters too; prefiltering them made the old
            # subsequent ``all(identity_product)`` gate tautologically true.
            identity_product = bool(product_lr == 0 and product_rl == 0)
            candidates.append(
                {
                    "cycle": int(snapshot.get("cycle", 0)),
                    "left_cluster_id": left_cluster.get("cluster_id"),
                    "right_cluster_id": right_cluster.get("cluster_id"),
                    "left_worldline_id": left_cluster.get("worldline_id"),
                    "right_worldline_id": right_cluster.get("worldline_id"),
                    "left_holonomy": left_h,
                    "right_holonomy": right_h,
                    "product_lr": product_lr,
                    "product_rl": product_rl,
                    "identity_product": identity_product,
                    "centroid_angular_distance": centroid_distance,
                    "encounter_geometry_verified": geometry_verified,
                    "encounter_angular_cutoff": float(max_angular_distance),
                    "candidate_basis": (
                        "intrinsic_s2_nearest_within_angular_cutoff"
                        if geometry_verified
                        else "legacy_missing_centroid_unverified_pair"
                    ),
                }
            )
    return candidates


def _snapshot_encounter_pairs(
    clusters: list[dict[str, object]],
    *,
    max_pairs: int = 4096,
    nearest_per_cluster: int = 4,
    max_angular_distance: float = 0.35,
) -> list[tuple[int, int, float | None, bool]]:
    """Bounded intrinsic-near encounters, plus fail-closed legacy rows."""

    count = len(clusters)
    if count < 2 or max_pairs <= 0:
        return []
    centroids_available = all(_valid_cluster_centroid(cluster) for cluster in clusters)
    pairs: set[tuple[int, int]] = set()
    pair_distances: dict[tuple[int, int], float] = {}
    if centroids_available:
        centroids = np.asarray([cluster.get("centroid") for cluster in clusters], dtype=float)
        unit_centroids = _normalize_rows(centroids)
        tree = cKDTree(unit_centroids)
        _chord_distances, neighbor_indices = tree.query(
            unit_centroids,
            k=min(count, 1 + max(1, int(nearest_per_cluster))),
        )
        if neighbor_indices.ndim == 1:
            neighbor_indices = neighbor_indices[:, None]
        for left_index in range(count):
            local_candidates: list[tuple[float, tuple[object, ...], int]] = []
            for right_raw in neighbor_indices[left_index, 1:]:
                right_index = int(right_raw)
                cosine = float(np.clip(np.dot(unit_centroids[left_index], unit_centroids[right_index]), -1.0, 1.0))
                angular_distance = float(np.arccos(cosine))
                local_candidates.append(
                    (angular_distance, _cluster_match_key(clusters[right_index]), right_index)
                )
            for angular_distance, _right_key, right_index in sorted(local_candidates):
                if angular_distance > float(max_angular_distance):
                    continue
                pair = (min(left_index, right_index), max(left_index, right_index))
                pairs.add(pair)
                pair_distances[pair] = angular_distance
    else:
        # Legacy/hand-authored reports may omit centroids. Retain bounded rows
        # for compatibility, but mark them unverified so they cannot pass the
        # fusion gate.
        for left_index in range(count):
            for right_index in range(left_index + 1, count):
                pairs.add((left_index, right_index))
                if len(pairs) >= int(max_pairs):
                    break
            if len(pairs) >= int(max_pairs):
                break

    ordered = sorted(
        pairs,
        key=(lambda pair: (pair_distances[pair], pair)) if pair_distances else (lambda pair: pair),
    )[: int(max_pairs)]
    return [
        (
            left_index,
            right_index,
            pair_distances.get((left_index, right_index)),
            bool(centroids_available),
        )
        for left_index, right_index in ordered
    ]


def _valid_cluster_centroid(cluster: dict[str, object]) -> bool:
    try:
        centroid = np.asarray(cluster.get("centroid"), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return False
    return bool(
        centroid.size == 3
        and np.all(np.isfinite(centroid))
        and float(np.linalg.norm(centroid)) > 1.0e-15
    )


def _class_mode_fraction(values: list[object]) -> tuple[object | None, float]:
    if not values:
        return None, 0.0
    counts: dict[object, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    mode = max(counts, key=counts.get)
    return mode, float(counts[mode] / len(values))
