from __future__ import annotations

from collections.abc import Iterable
from itertools import permutations

import numpy as np
from scipy.spatial import cKDTree


S3_ELEMENTS: tuple[tuple[int, int, int], ...] = tuple(permutations((0, 1, 2)))
S3_INDEX = {element: index for index, element in enumerate(S3_ELEMENTS)}


def _compose(left: tuple[int, int, int], right: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(left[right[index]] for index in range(3))


S3_MUL = np.array(
    [[S3_INDEX[_compose(left, right)] for right in S3_ELEMENTS] for left in S3_ELEMENTS],
    dtype=np.int16,
)
S3_INV = np.array(
    [S3_INDEX[tuple(element.index(index) for index in range(3))] for element in S3_ELEMENTS],
    dtype=np.int16,
)
S3_CLASS = np.array(
    [
        0 if element == (0, 1, 2) else 1 if sum(1 for i, image in enumerate(element) if i != image) == 2 else 2
        for element in S3_ELEMENTS
    ],
    dtype=np.int16,
)
_EDGE_LOOKUP_CACHE_MAX = 4
_EDGE_LOOKUP_CACHE: dict[tuple[int, int, int], dict[tuple[int, int], tuple[int, int, int]]] = {}


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


def oriented_triangles(points: np.ndarray, left: np.ndarray, right: np.ndarray, *, max_triangles: int | None = None) -> np.ndarray:
    if max_triangles is not None and int(max_triangles) <= 0:
        return np.zeros((0, 3), dtype=np.int64)
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
                if max_triangles is not None and len(triangles) >= int(max_triangles):
                    return np.asarray(triangles, dtype=np.int64).reshape((-1, 3))
    return np.asarray(triangles, dtype=np.int64).reshape((-1, 3))


def s3_triangle_holonomy(g_ij: np.ndarray | int, g_jk: np.ndarray | int, g_ki: np.ndarray | int) -> np.ndarray:
    first = S3_MUL[np.asarray(g_ij, dtype=np.int64), np.asarray(g_jk, dtype=np.int64)]
    return S3_MUL[first, np.asarray(g_ki, dtype=np.int64)]


def triangle_holonomies(triangles: np.ndarray, left: np.ndarray, right: np.ndarray, gauge: np.ndarray) -> np.ndarray:
    lookup = _edge_index_lookup(left, right)
    gauge = np.asarray(gauge, dtype=np.int64)
    holonomies: list[int] = []
    for i, j, k in np.asarray(triangles, dtype=np.int64):
        holonomies.append(
            int(
                s3_triangle_holonomy(
                    _edge_label(lookup, gauge, int(i), int(j)),
                    _edge_label(lookup, gauge, int(j), int(k)),
                    _edge_label(lookup, gauge, int(k), int(i)),
                )
            )
        )
    return np.asarray(holonomies, dtype=np.int16)


def defect_class(holonomy: np.ndarray | int) -> np.ndarray:
    return S3_CLASS[np.asarray(holonomy, dtype=np.int64)]


def array_holonomy_report(
    points: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    gauge: np.ndarray,
    *,
    max_triangles: int | None = 10_000,
) -> dict[str, object]:
    triangles = oriented_triangles(points, left, right, max_triangles=max_triangles)
    holonomies = triangle_holonomies(triangles, left, right, gauge) if triangles.size else np.zeros(0, dtype=np.int16)
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
) -> dict[str, object]:
    snapshots: list[dict[str, object]] = []
    active_worldlines: dict[str, dict[str, object]] = {}
    completed: list[dict[str, object]] = []
    next_worldline = 0
    previous_clusters: list[dict[str, object]] = []
    previous_ids: list[str] = []
    for cycle, gauge in gauge_snapshots:
        report = array_holonomy_report(points, left, right, gauge, max_triangles=max_triangles)
        clusters = list(report.get("clusters", []))
        assignments, next_worldline = _assign_worldlines(
            previous_clusters,
            previous_ids,
            clusters,
            next_worldline,
        )
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
        snapshots.append(
            {
                "cycle": int(cycle),
                "triangle_count": report.get("triangle_count"),
                "defect_triangle_count": report.get("defect_triangle_count"),
                "cluster_count": len(clusters),
                "clusters": [
                    {
                        "cluster_id": cluster.get("cluster_id"),
                        "worldline_id": worldline_id,
                        "class": cluster.get("class"),
                        "holonomy_mode": cluster.get("holonomy_mode"),
                        "inverse_holonomy_mode": cluster.get("inverse_holonomy_mode"),
                        "support_node_count": cluster.get("support_node_count"),
                        "support_nodes": list(cluster.get("support_nodes", [])),
                        "centroid": cluster.get("centroid"),
                    }
                    for cluster, worldline_id, _event, _distance in assignments
                ],
            }
        )
        previous_clusters = clusters
        previous_ids = [worldline_id for _cluster, worldline_id, _event, _distance in assignments]
    completed.extend(_finalize_worldline(row, persistence_cycles) for row in active_worldlines.values())
    persistent = [row for row in completed if row.get("persistent")]
    return {
        "mode": "array_s3_defect_timeline",
        "patch_count": int(points.shape[0]),
        "snapshot_count": len(snapshots),
        "max_triangles": int(max_triangles) if max_triangles is not None else None,
        "snapshots": snapshots,
        "worldlines": completed,
        "worldline_count": len(completed),
        "persistent_worldline_count": len(persistent),
        "max_observation_count": max((int(row.get("observation_count", 0)) for row in completed), default=0),
        "max_lifetime_cycles": max((int(row.get("lifetime_cycles", 0)) for row in completed), default=0),
        "persistent_worldline_precursor_receipt": bool(persistent),
        "particle_matter_receipt": False,
        "claim_boundary": (
            "time-resolved screen/collar S3 holonomy defect clusters under declared finite repair dynamics. "
            "Persistent worldlines are particle precursors only; particle claims require localization, "
            "transport, fusion/scattering, neutral-bulk mapping, and repeated-seed controls."
        ),
    }


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
            bulk_localization_pass
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
            row["bulk_localization_pass"]
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
        "particle_like_count": int(particle_like_count),
        "particle_detector_positive_receipt": bool(detector_positive_count > 0),
        "particle_matter_receipt": bool(
            particle_like_count > 0 and bulk_localization_pass and fusion_gauge_covariant
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
        transport_proxy = bool(persistent and class_stable and transport_distances)
        rows.append(
            {
                "worldline_id": worldline.get("worldline_id"),
                "observation_count": int(worldline.get("observation_count", len(events))),
                "class_mode": class_mode,
                "class_stability_fraction": float(class_fraction),
                "transport_event_count": len(transport_distances),
                "mean_transport_distance": float(np.mean(transport_distances)) if transport_distances else 0.0,
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
    scattering_proxy = bool(
        total_transitions >= int(min_scattering_transitions)
        and dominant_transition_fraction >= float(min_class_stability)
    )
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
            verified_fusion_candidates
            and verified_identity_fusion_count == len(verified_fusion_candidates)
        ),
        "fusion_common_basepoint_transport_receipt": False,
        "fusion_gauge_covariant_receipt": False,
        "fusion_theorem_receipt": False,
        "scattering_transition_counts": transition_counts,
        "scattering_transition_total": int(total_transitions),
        "dominant_scattering_transition_fraction": float(dominant_transition_fraction),
        "scattering_reproducibility_proxy_pass": scattering_proxy,
        "interaction_proxy_receipt": bool(transportable_count and scattering_proxy),
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
    scale = _matching_scale(previous_centroids, current_centroids, distances=distances)
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
    """Deterministic global greedy matching under sector and distance gates."""

    candidates: list[tuple[float, tuple[object, ...], tuple[object, ...], int, int]] = []
    for previous_index, previous in enumerate(previous_clusters):
        for current_index, current in enumerate(current_clusters):
            distance = float(distances[previous_index, current_index])
            if distance > float(max_distance) or not _cluster_sector_compatible(previous, current):
                continue
            candidates.append(
                (
                    distance,
                    _cluster_match_key(previous),
                    _cluster_match_key(current),
                    int(previous_index),
                    int(current_index),
                )
            )
    candidates.sort()
    used_previous: set[int] = set()
    used_current: set[int] = set()
    matches: dict[int, tuple[int, float]] = {}
    for distance, _previous_key, _current_key, previous_index, current_index in candidates:
        if previous_index in used_previous or current_index in used_current:
            continue
        used_previous.add(previous_index)
        used_current.add(current_index)
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


def _matching_scale(
    previous_centroids: np.ndarray,
    current_centroids: np.ndarray,
    *,
    distances: np.ndarray | None = None,
) -> float:
    """Robust angular displacement gate for consecutive S2 snapshots."""

    if distances is None:
        distances = _spherical_distance_matrix(previous_centroids, current_centroids)
    if distances.size == 0:
        return 0.0
    nearest = np.min(distances, axis=0)
    finite = nearest[np.isfinite(nearest)]
    if not finite.size:
        return 0.0
    # Permit normal sub-snapshot motion without joining unrelated defects on
    # opposite parts of the screen.  All values are intrinsic radians.
    return float(np.clip(2.5 * np.percentile(finite, 75), 0.05, 0.75))


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


def _edge_index_lookup(left: np.ndarray, right: np.ndarray) -> dict[tuple[int, int], tuple[int, int, int]]:
    left_array = np.asarray(left, dtype=np.int64)
    right_array = np.asarray(right, dtype=np.int64)
    key = (id(left), id(right), int(left_array.size))
    cached = _EDGE_LOOKUP_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_EDGE_LOOKUP_CACHE) >= _EDGE_LOOKUP_CACHE_MAX:
        _EDGE_LOOKUP_CACHE.clear()
    lookup: dict[tuple[int, int], tuple[int, int, int]] = {}
    for edge_index, (a_raw, b_raw) in enumerate(zip(left_array, right_array, strict=False)):
        a = int(a_raw)
        b = int(b_raw)
        lookup[(min(a, b), max(a, b))] = (int(edge_index), a, b)
    _EDGE_LOOKUP_CACHE[key] = lookup
    return lookup


def _edge_label(
    lookup: dict[tuple[int, int], tuple[int, int, int]],
    gauge: np.ndarray,
    source: int,
    target: int,
) -> int:
    edge_index, left_node, right_node = lookup[(min(int(source), int(target)), max(int(source), int(target)))]
    value = int(gauge[edge_index])
    return value if int(source) == left_node and int(target) == right_node else int(S3_INV[value])


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
