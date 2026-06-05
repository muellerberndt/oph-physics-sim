from __future__ import annotations

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
    adjacency: dict[int, set[int]] = {}
    for a, b in zip(np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64), strict=False):
        adjacency.setdefault(int(a), set()).add(int(b))
        adjacency.setdefault(int(b), set()).add(int(a))
    triangles: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for i, neighbors in adjacency.items():
        for j in neighbors:
            if i >= j:
                continue
            common = neighbors & adjacency.get(j, set())
            for k in common:
                key = tuple(sorted((i, j, k)))
                if key in seen:
                    continue
                seen.add(key)
                tri = _orient_triangle(points, (i, j, int(k)))
                triangles.append(tri)
                if max_triangles is not None and len(triangles) >= int(max_triangles):
                    return np.asarray(triangles, dtype=np.int64)
    return np.asarray(triangles, dtype=np.int64)


def s3_triangle_holonomy(g_ij: np.ndarray | int, g_jk: np.ndarray | int, g_ki: np.ndarray | int) -> np.ndarray:
    first = S3_MUL[np.asarray(g_ij, dtype=np.int64), np.asarray(g_jk, dtype=np.int64)]
    return S3_MUL[first, np.asarray(g_ki, dtype=np.int64)]


def triangle_holonomies(triangles: np.ndarray, left: np.ndarray, right: np.ndarray, gauge: np.ndarray) -> np.ndarray:
    labels = _oriented_edge_labels(left, right, gauge)
    holonomies: list[int] = []
    for i, j, k in np.asarray(triangles, dtype=np.int64):
        holonomies.append(
            int(
                s3_triangle_holonomy(
                    labels[(int(i), int(j))],
                    labels[(int(j), int(k))],
                    labels[(int(k), int(i))],
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
        particle_like = bool(
            bulk_localization_pass
            and localized
            and persistent
            and class_stable
            and transportable
            and fusion_proxy
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
                "fusion_conservation_pass": fusion_proxy,
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
    return {
        "mode": "screen_holonomy_particle_likeness_diagnostic",
        "worldline_count": len(rows),
        "localized_count": int(localized_count),
        "persistent_count": int(persistent_count),
        "sector_stable_count": int(sector_count),
        "transportable_count": int(transport_count),
        "fusion_conserving_count": int(fusion_count),
        "scattering_reproducible_count": int(scattering_count),
        "bulk_localization_pass": bool(bulk_localization_pass),
        "max_support_fraction_threshold": float(max_support_fraction),
        "min_observations": int(min_observations),
        "min_class_stability": float(min_class_stability),
        "interaction_report_mode": (interaction_report or {}).get("mode"),
        "particle_like_count": int(particle_like_count),
        "particle_matter_receipt": bool(particle_like_count > 0 and bulk_localization_pass),
        "worldlines": rows[:256],
        "claim_boundary": (
            "screen holonomy defect particle-likeness score. Persistence/localization/sector "
            "stability may be present, but matter-particle claims require neutral 3D bulk mapping, "
            "transport around contractible paths, fusion conservation, scattering reproducibility, "
            "and repeated-seed controls."
        ),
    }


def defect_interaction_report(
    timeline_report: dict[str, object],
    *,
    min_observations: int = 3,
    min_class_stability: float = 0.8,
    min_transport_distance: float = 1e-9,
    min_scattering_transitions: int = 2,
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

    fusion_candidates = _fusion_candidates_from_snapshots(timeline_report)
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
        "fusion_identity_candidate_count": sum(bool(row["identity_product"]) for row in fusion_candidates),
        "fusion_conservation_proxy_pass": bool(fusion_candidates and all(bool(row["identity_product"]) for row in fusion_candidates)),
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
            "same-snapshot inverse-holonomy fusion candidates, and class-transition stability. "
            "It is not a matter-particle or 3D-bulk claim without neutral/H3 localization and "
            "contractible-path transport, fusion, and scattering controls."
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
    centroids = np.mean(points[triangles[active]], axis=1)
    tree = cKDTree(centroids)
    visited: set[int] = set()
    clusters: list[dict[str, object]] = []
    radius = float(np.median(np.linalg.norm(np.diff(np.sort(centroids, axis=0), axis=0), axis=1))) if centroids.shape[0] > 2 else 0.1
    radius = max(radius, 0.1)
    for local_index, triangle_index in enumerate(active):
        if local_index in visited:
            continue
        neighbors = [int(value) for value in tree.query_ball_point(centroids[local_index], r=radius)]
        visited.update(neighbors)
        member_indices = active[np.asarray(neighbors, dtype=np.int64)]
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
                "centroid": [float(value) for value in np.mean(centroids[neighbors], axis=0)],
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
        assignments = []
        for cluster in current_clusters:
            worldline_id = f"worldline_{next_worldline:06d}"
            next_worldline += 1
            assignments.append((cluster, worldline_id, "birth", None))
        return assignments, next_worldline
    previous_centroids = np.asarray([cluster.get("centroid", [0.0, 0.0, 1.0]) for cluster in previous_clusters], dtype=float)
    current_centroids = np.asarray([cluster.get("centroid", [0.0, 0.0, 1.0]) for cluster in current_clusters], dtype=float)
    if current_centroids.size == 0:
        return [], next_worldline
    scale = _matching_scale(previous_centroids, current_centroids)
    used_previous: set[int] = set()
    assignments: list[tuple[dict[str, object], str, str, float | None]] = []
    for index, cluster in enumerate(current_clusters):
        distances = np.linalg.norm(previous_centroids - current_centroids[index], axis=1)
        order = np.argsort(distances)
        match = next((int(candidate) for candidate in order if int(candidate) not in used_previous), None)
        if match is not None and float(distances[match]) <= scale:
            used_previous.add(match)
            assignments.append((cluster, previous_ids[match], "continue", float(distances[match])))
        else:
            worldline_id = f"worldline_{next_worldline:06d}"
            next_worldline += 1
            assignments.append((cluster, worldline_id, "birth", None))
    return assignments, next_worldline


def _matching_scale(previous_centroids: np.ndarray, current_centroids: np.ndarray) -> float:
    combined = np.vstack([previous_centroids, current_centroids])
    if combined.shape[0] < 3:
        return 0.35
    diffs = np.linalg.norm(np.diff(np.sort(combined, axis=0), axis=0), axis=1)
    return max(0.25, float(np.percentile(diffs, 75)) * 4.0)


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
    if not previous_clusters:
        return [
            {
                "worldline_id": f"worldline_{index:06d}",
                "previous_cluster_id": None,
                "current_cluster_id": cluster["cluster_id"],
                "event": "birth",
            }
            for index, cluster in enumerate(current_clusters)
        ]
    previous_centroids = np.array([cluster["centroid"] for cluster in previous_clusters], dtype=float)
    result: list[dict[str, object]] = []
    used: set[int] = set()
    for current in current_clusters:
        centroid = np.asarray(current["centroid"], dtype=float)
        distances = np.linalg.norm(previous_centroids - centroid, axis=1)
        order = np.argsort(distances)
        match = next((int(index) for index in order if int(index) not in used), None)
        if match is None:
            event = "birth"
            previous_id = None
            distance = None
        else:
            used.add(match)
            event = "continue"
            previous_id = previous_clusters[match]["cluster_id"]
            distance = float(distances[match])
        result.append(
            {
                "worldline_id": f"worldline_{len(result):06d}",
                "previous_cluster_id": previous_id,
                "current_cluster_id": current["cluster_id"],
                "event": event,
                "transport_distance": distance,
            }
        )
    return result


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


def _class_name(class_id: int) -> str:
    return {0: "identity", 1: "transposition", 2: "threecycle"}.get(int(class_id), str(int(class_id)))


def _fusion_candidates_from_snapshots(timeline_report: dict[str, object]) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for snapshot in timeline_report.get("snapshots", []) if timeline_report else []:
        clusters = [
            cluster
            for cluster in snapshot.get("clusters", [])
            if cluster.get("holonomy_mode") is not None
        ]
        for left_index, left_cluster in enumerate(clusters):
            left_h = int(left_cluster.get("holonomy_mode"))
            for right_cluster in clusters[left_index + 1 :]:
                right_h = int(right_cluster.get("holonomy_mode"))
                product_lr = int(S3_MUL[left_h, right_h])
                product_rl = int(S3_MUL[right_h, left_h])
                identity_product = bool(product_lr == 0 or product_rl == 0)
                if not identity_product:
                    continue
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
                    }
                )
    return candidates


def _class_mode_fraction(values: list[object]) -> tuple[object | None, float]:
    if not values:
        return None, 0.0
    counts: dict[object, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    mode = max(counts, key=counts.get)
    return mode, float(counts[mode] / len(values))
