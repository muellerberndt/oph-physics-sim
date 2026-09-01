"""Bounded exploratory causal-set diagnostics for the semantic OPH poset.

This artifact measures finite order statistics that are standard in causal-set
work: ordering fraction, height, width, connected components, and interval
abundance.  It validates the Myrheim--Meyer ordering-fraction inversion on
fixed deterministic Minkowski Alexandrov sprinkling controls, but applies no
dimension estimate to an arbitrary disconnected OPH region.  The
Glaser--Surya interval-abundance comparison is exposed only when the source
contains adequately deep intervals.

No statistic here proves physical causal faithfulness, faithful embedding,
manifoldlikeness, dimension four, volume density, Lorentzian geometry, or a
continuum limit.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data/causal_order/source_derived_causal_order_receipt.json"
DEFAULT_OUTPUT = ROOT / "data/causal_order/causet_likeness_receipt.json"
LOCAL_DOMAIN_RECEIPT = ROOT / "data/local_domain/stage1_receipt.json"
LOCAL_DOMAIN_ARRAYS = ROOT / "data/local_domain/stage1_arrays.npz.gz"
SCHEMA = "oph.causet-likeness-exploratory.v1"
MIN_DIMENSION_INTERVAL_SIZE = 32
MIN_INTERVAL_SAMPLE_COUNT = 4
REFERENCE_DIMS = (2, 3, 4, 5)
REFERENCE_SEEDS = (76301, 76302, 76303, 76304)
REFERENCE_MEAN_COUNT = 128
DENSITY_CONTROL_MEAN_COUNTS = (64, 128, 256)
DENSITY_CONTROL_DIMENSION = 4
MATCHED_PROFILE_M_MAX = 15
MATCHED_TRAIN_SEEDS = (76401, 76402, 76403, 76404)
MATCHED_HELDOUT_SEEDS = (76411, 76412, 76413, 76414)
RANDOM_ORDER_EDGE_PROBABILITY = 0.5
EXPLORATORY_4D_ORDERING_FRACTION_BAND = (0.05, 0.20)
FLRW_ETA_START = -2.0
FLRW_ETA_END = -1.0
FLRW_HUBBLE = 1.0


class CausetDiagnosticError(RuntimeError):
    """Raised when the supplied relation is not a finite strict poset."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _sha(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _raw_sha(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _transitive_data(
    nodes: Iterable[str], edges: Iterable[tuple[str, str]]
) -> tuple[list[str], list[set[int]], list[set[int]], list[int]]:
    ordered = sorted(set(str(node) for node in nodes))
    if not ordered:
        raise CausetDiagnosticError("empty event carrier")
    index = {node: position for position, node in enumerate(ordered)}
    children = [set() for _ in ordered]
    indegree = [0 for _ in ordered]
    for raw_parent, raw_child in edges:
        parent = index.get(str(raw_parent))
        child = index.get(str(raw_child))
        if parent is None or child is None:
            raise CausetDiagnosticError("edge names an absent event")
        if parent == child:
            raise CausetDiagnosticError("strict order contains a self edge")
        if child not in children[parent]:
            children[parent].add(child)
            indegree[child] += 1
    frontier = sorted(position for position, degree in enumerate(indegree) if degree == 0)
    topological: list[int] = []
    work = list(indegree)
    while frontier:
        node = frontier.pop()
        topological.append(node)
        for child in sorted(children[node]):
            work[child] -= 1
            if work[child] == 0:
                frontier.append(child)
    if len(topological) != len(ordered):
        raise CausetDiagnosticError("relation contains a directed cycle")
    ancestors = [set() for _ in ordered]
    descendants = [set() for _ in ordered]
    height = [1 for _ in ordered]
    for node in topological:
        for child in children[node]:
            ancestors[child].add(node)
            ancestors[child].update(ancestors[node])
            height[child] = max(height[child], height[node] + 1)
    for node in reversed(topological):
        for child in children[node]:
            descendants[node].add(child)
            descendants[node].update(descendants[child])
    return ordered, ancestors, descendants, height


def _width(ancestors: list[set[int]]) -> int:
    """Exact width via Dilworth's theorem and bipartite matching."""

    count = len(ancestors)
    successors = [set() for _ in range(count)]
    for child, parents in enumerate(ancestors):
        for parent in parents:
            successors[parent].add(child)
    match_right = [-1] * count

    def augment(left: int, seen: set[int]) -> bool:
        for right in sorted(successors[left]):
            if right in seen:
                continue
            seen.add(right)
            if match_right[right] < 0 or augment(match_right[right], seen):
                match_right[right] = left
                return True
        return False

    matching = sum(augment(left, set()) for left in range(count))
    return count - matching


def _component_sizes(count: int, edges: Iterable[tuple[int, int]]) -> list[int]:
    adjacency = [set() for _ in range(count)]
    for parent, child in edges:
        adjacency[parent].add(child)
        adjacency[child].add(parent)
    unseen = set(range(count))
    sizes: list[int] = []
    while unseen:
        root = unseen.pop()
        stack = [root]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            neighbors = adjacency[node] & unseen
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def myrheim_meyer_fraction(dimension: float) -> float:
    """Expected ordering fraction in a flat d-dimensional Alexandrov interval."""

    return math.exp(
        math.lgamma(dimension + 1.0)
        + math.lgamma(dimension / 2.0)
        - math.log(2.0)
        - math.lgamma(3.0 * dimension / 2.0)
    )


def invert_myrheim_meyer_fraction(fraction: float) -> float | None:
    if not 0.0 < fraction < 1.0:
        return None
    low, high = 1.01, 20.0
    if not (
        myrheim_meyer_fraction(high)
        <= fraction
        <= myrheim_meyer_fraction(low)
    ):
        return None
    for _ in range(96):
        midpoint = (low + high) / 2.0
        if myrheim_meyer_fraction(midpoint) > fraction:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def poset_statistics(
    nodes: Iterable[str], edges: Iterable[tuple[str, str]]
) -> dict[str, Any]:
    node_list = sorted(set(str(node) for node in nodes))
    edge_list = sorted(set((str(parent), str(child)) for parent, child in edges))
    ordered, ancestors, descendants, heights = _transitive_data(node_list, edge_list)
    index = {node: position for position, node in enumerate(ordered)}
    indexed_edges = [(index[parent], index[child]) for parent, child in edge_list]
    comparable_pairs = sum(len(parents) for parents in ancestors)
    count = len(ordered)
    ordering_fraction = (
        2.0 * comparable_pairs / (count * (count - 1)) if count > 1 else 0.0
    )
    interval_histogram: dict[int, int] = {}
    interval_rows: list[tuple[int, int, int]] = []
    for future, pasts in enumerate(ancestors):
        for past in pasts:
            inclusive_size = len(descendants[past] & ancestors[future]) + 2
            interval_histogram[inclusive_size] = (
                interval_histogram.get(inclusive_size, 0) + 1
            )
            interval_rows.append((inclusive_size, past, future))
    interval_rows.sort(reverse=True)
    adequate = [row for row in interval_rows if row[0] >= MIN_DIMENSION_INTERVAL_SIZE]
    width = _width(ancestors)
    components = _component_sizes(count, indexed_edges)
    cover_edges = [
        (past, future)
        for future, pasts in enumerate(ancestors)
        for past in pasts
        if not (descendants[past] & ancestors[future])
    ]
    cover_components = _component_sizes(count, cover_edges)
    cover_cycle_rank = len(cover_edges) - count + len(cover_components)
    return {
        "event_count": count,
        "input_edge_count": len(edge_list),
        "comparable_pair_count": comparable_pairs,
        "global_ordering_fraction": ordering_fraction,
        "global_ordering_fraction_interpretation": (
            "diagnostic_only_not_one_verified_alexandrov_interval"
        ),
        "height": max(heights, default=0),
        "width": width,
        "height_width_ratio": max(heights, default=0) / max(width, 1),
        "weak_component_count": len(components),
        "weak_component_sizes": components,
        "cover_relation_count": len(cover_edges),
        "cover_graph_cycle_rank": cover_cycle_rank,
        "cover_graph_is_forest": cover_cycle_rank == 0,
        "interval_count": len(interval_rows),
        "interval_abundance_inclusive_size": {
            str(size): interval_histogram[size]
            for size in sorted(interval_histogram)
        },
        "maximum_interval_size": interval_rows[0][0] if interval_rows else 0,
        "adequate_dimension_interval_minimum_size": MIN_DIMENSION_INTERVAL_SIZE,
        "adequate_dimension_interval_count": len(adequate),
        "myrheim_meyer_dimension_estimate": None,
        "myrheim_meyer_status": (
            "NOT_EVALUATED_GLOBAL_REGION_NOT_VERIFIED_ALEXANDROV_INTERVAL"
        ),
        "glaser_surya_interval_abundance_status": (
            "ELIGIBLE_FOR_EXPLORATORY_SINGLE_CUTOFF_COMPARISON"
            if len(adequate) >= MIN_INTERVAL_SAMPLE_COUNT
            else "NOT_EVALUATED_INSUFFICIENT_DEEP_INTERVALS"
        ),
    }


def _minkowski_sprinkling(dimension: int, seed: int) -> tuple[list[str], list[tuple[str, str]]]:
    rng = np.random.Generator(np.random.PCG64(seed + 1000 * dimension))
    count = max(32, int(rng.poisson(REFERENCE_MEAN_COUNT)))
    spatial_dimension = dimension - 1
    points: list[np.ndarray] = []
    while len(points) < count:
        time = float(rng.uniform(-0.5, 0.5))
        radius_limit = 0.5 - abs(time)
        if float(rng.random()) > (2.0 * radius_limit) ** spatial_dimension:
            continue
        direction = rng.normal(size=spatial_dimension)
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            continue
        radius = radius_limit * float(rng.random()) ** (1.0 / spatial_dimension)
        space = direction * (radius / norm)
        points.append(np.asarray([time, *space], dtype=float))
    nodes = [f"p-{index:04d}" for index in range(count)]
    edges: list[tuple[str, str]] = []
    for left in range(count):
        for right in range(left + 1, count):
            first, second = points[left], points[right]
            if first[0] <= second[0]:
                past, future, past_id, future_id = first, second, left, right
            else:
                past, future, past_id, future_id = second, first, right, left
            delta_time = float(future[0] - past[0])
            delta_space = future[1:] - past[1:]
            if delta_time > 0.0 and delta_time * delta_time >= float(
                np.dot(delta_space, delta_space)
            ):
                edges.append((nodes[past_id], nodes[future_id]))
    return nodes, edges


def _sample_spatial_ball(
    rng: np.random.Generator, spatial_dimension: int, radius_limit: float
) -> np.ndarray:
    while True:
        direction = rng.normal(size=spatial_dimension)
        norm = float(np.linalg.norm(direction))
        if norm > 0.0:
            break
    radius = radius_limit * float(rng.random()) ** (1.0 / spatial_dimension)
    return direction * (radius / norm)


def _sample_conformal_diamond_point(
    rng: np.random.Generator,
    *,
    eta_start: float,
    eta_end: float,
    flrw_volume_weight: bool,
) -> np.ndarray:
    """Sample a 3+1 diamond uniformly in its requested physical measure."""

    midpoint = 0.5 * (eta_start + eta_end)
    half_duration = 0.5 * (eta_end - eta_start)
    if half_duration <= 0.0:
        raise CausetDiagnosticError("invalid conformal-diamond endpoints")
    if flrw_volume_weight and not (eta_start < eta_end < 0.0):
        raise CausetDiagnosticError("de Sitter conformal patch must remain at eta < 0")
    maximum_scale_factor = (
        -1.0 / (FLRW_HUBBLE * eta_end) if flrw_volume_weight else 1.0
    )
    while True:
        eta = float(rng.uniform(eta_start, eta_end))
        radius_limit = half_duration - abs(eta - midpoint)
        cross_section_weight = (radius_limit / half_duration) ** 3
        if flrw_volume_weight:
            scale_factor = -1.0 / (FLRW_HUBBLE * eta)
            cross_section_weight *= (scale_factor / maximum_scale_factor) ** 4
        if float(rng.random()) <= cross_section_weight:
            space = _sample_spatial_ball(rng, 3, radius_limit)
            return np.asarray([eta, *space], dtype=float)


def _causal_relations(
    nodes: list[str], points: list[np.ndarray]
) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            first, second = points[left], points[right]
            if first[0] <= second[0]:
                past, future, past_id, future_id = first, second, left, right
            else:
                past, future, past_id, future_id = second, first, right, left
            delta_time = float(future[0] - past[0])
            delta_space = future[1:] - past[1:]
            if delta_time > 0.0 and delta_time * delta_time >= float(
                np.dot(delta_space, delta_space)
            ):
                edges.append((nodes[past_id], nodes[future_id]))
    return sorted(edges)


def _compact_reference_statistics(stats: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_count": stats["event_count"],
        "causal_relation_count": stats["input_edge_count"],
        "comparable_pair_count": stats["comparable_pair_count"],
        "ordering_fraction": stats["global_ordering_fraction"],
        "height": stats["height"],
        "width": stats["width"],
        "maximum_interval_size": stats["maximum_interval_size"],
        "adequate_dimension_interval_count": stats[
            "adequate_dimension_interval_count"
        ],
    }


def _nested_reference_family(*, geometry: str, seed: int) -> dict[str, Any]:
    """Couple Poisson intensities by independent marking of one maximal draw."""

    if geometry not in {"minkowski_3_plus_1", "de_sitter_flat_patch_3_plus_1"}:
        raise CausetDiagnosticError("unknown nested reference geometry")
    seed_offset = 400_000 if geometry == "minkowski_3_plus_1" else 500_000
    rng = np.random.Generator(np.random.PCG64(seed + seed_offset))
    maximum_mean = max(DENSITY_CONTROL_MEAN_COUNTS)
    maximum_count = int(rng.poisson(maximum_mean))
    if geometry == "minkowski_3_plus_1":
        eta_start, eta_end, flrw_weight = -0.5, 0.5, False
    else:
        eta_start, eta_end, flrw_weight = FLRW_ETA_START, FLRW_ETA_END, True
    points = [
        _sample_conformal_diamond_point(
            rng,
            eta_start=eta_start,
            eta_end=eta_end,
            flrw_volume_weight=flrw_weight,
        )
        for _ in range(maximum_count)
    ]
    marks = np.asarray(rng.random(maximum_count), dtype=float)
    all_nodes = [f"{geometry}-s{seed}-{index:04d}" for index in range(maximum_count)]
    all_edges = _causal_relations(all_nodes, points)
    levels: list[dict[str, Any]] = []
    carriers: list[set[str]] = []
    relations: list[set[tuple[str, str]]] = []
    for mean_count in DENSITY_CONTROL_MEAN_COUNTS:
        threshold = mean_count / maximum_mean
        selected_indices = [
            index for index, mark in enumerate(marks) if float(mark) <= threshold
        ]
        selected_nodes = [all_nodes[index] for index in selected_indices]
        selected_set = set(selected_nodes)
        selected_edges = [
            edge
            for edge in all_edges
            if edge[0] in selected_set and edge[1] in selected_set
        ]
        stats = poset_statistics(selected_nodes, selected_edges)
        row: dict[str, Any] = {
            "target_poisson_mean_count": mean_count,
            "point_coordinates_sha256": _sha(
                [
                    {
                        "event_key": all_nodes[index],
                        "conformal_coordinates": points[index].tolist(),
                    }
                    for index in selected_indices
                ]
            ),
            "semantic_carrier_sha256": _sha(selected_nodes),
            "causal_relation_sha256": _sha(selected_edges),
            "statistics": _compact_reference_statistics(stats),
        }
        if geometry == "minkowski_3_plus_1":
            row["flat_myrheim_meyer_dimension_estimate"] = (
                invert_myrheim_meyer_fraction(
                    float(stats["global_ordering_fraction"])
                )
            )
            row["flat_myrheim_meyer_status"] = (
                "REFERENCE_FLAT_ALEXANDROV_INTERVAL_ONLY"
            )
        else:
            row["flat_myrheim_meyer_dimension_estimate"] = None
            row["flat_myrheim_meyer_status"] = (
                "NOT_APPLIED_CURVED_FLRW_REFERENCE_CONTROL"
            )
        levels.append(row)
        carriers.append(selected_set)
        relations.append(set(selected_edges))
    node_inclusions = all(
        carriers[index] <= carriers[index + 1]
        for index in range(len(carriers) - 1)
    )
    induced_orders = all(
        relations[index]
        == {
            edge
            for edge in relations[index + 1]
            if edge[0] in carriers[index] and edge[1] in carriers[index]
        }
        for index in range(len(relations) - 1)
    )
    return {
        "seed": seed,
        "maximal_poisson_draw_count": maximum_count,
        "levels": levels,
        "nested_carrier_inclusions_hold": node_inclusions,
        "induced_causal_orders_hold": induced_orders,
        "nested_poisson_inclusion_coupling_status": (
            "CERTIFIED_CARRIER_INCLUSIONS_AND_INDUCED_SUBORDERS"
            if node_inclusions and induced_orders
            else "FAILED"
        ),
    }


def _nested_reference_controls() -> dict[str, Any]:
    geometries = {
        geometry: {
            "runs": [
                _nested_reference_family(geometry=geometry, seed=seed)
                for seed in REFERENCE_SEEDS
            ]
        }
        for geometry in (
            "minkowski_3_plus_1",
            "de_sitter_flat_patch_3_plus_1",
        )
    }
    for geometry, row in geometries.items():
        row["all_nested_inclusion_couplings_certified"] = all(
            run["nested_poisson_inclusion_coupling_status"]
            == "CERTIFIED_CARRIER_INCLUSIONS_AND_INDUCED_SUBORDERS"
            for run in row["runs"]
        )
        row["flat_myrheim_meyer_policy"] = (
            "APPLIED_ONLY_TO_FLAT_3_PLUS_1_ALEXANDROV_REFERENCE"
            if geometry == "minkowski_3_plus_1"
            else "NOT_APPLIED_TO_CURVED_FLRW_REFERENCE"
        )
    return {
        "status": "SYNTHETIC_NESTED_DENSITY_CONTROLS_REPLAYED",
        "coupling": "nested_poisson_thinning_from_mean_256",
        "mean_counts": list(DENSITY_CONTROL_MEAN_COUNTS),
        "geometries": geometries,
        "oph_comparison_status": (
            "NOT_EVALUATED_NO_CERTIFIED_OPH_REFINEMENT_MAP_OR_FAMILY"
        ),
        "cross_geometry_similarity_claimed": False,
    }


def _feature_from_statistics(stats: Mapping[str, Any]) -> dict[str, Any]:
    count = int(stats["event_count"])
    histogram = {
        int(size): int(value)
        for size, value in stats["interval_abundance_inclusive_size"].items()
    }
    n_zero = histogram.get(2, 0)
    profile = [
        (histogram.get(open_size + 2, 0) / n_zero if n_zero else 0.0)
        for open_size in range(MATCHED_PROFILE_M_MAX + 1)
    ]
    return {
        "cardinality": count,
        "ordering_fraction": float(stats["global_ordering_fraction"]),
        "height_to_cardinality": float(stats["height"]) / max(count, 1),
        "normalized_Nm_over_N0": profile,
    }


def _feature_vector(feature: Mapping[str, Any]) -> list[float]:
    return [
        float(feature["ordering_fraction"]),
        float(feature["height_to_cardinality"]),
        *[float(value) for value in feature["normalized_Nm_over_N0"]],
    ]


def _fixed_cardinality_control_feature(
    *, geometry: str, cardinality: int, seed: int
) -> dict[str, Any]:
    if geometry in {"minkowski_3_plus_1", "de_sitter_flat_patch_3_plus_1"}:
        offset = 700_000 if geometry == "minkowski_3_plus_1" else 800_000
        rng = np.random.Generator(
            np.random.PCG64(seed + offset + 10_000 * cardinality)
        )
        if geometry == "minkowski_3_plus_1":
            eta_start, eta_end, flrw_weight = -0.5, 0.5, False
        else:
            eta_start, eta_end, flrw_weight = (
                FLRW_ETA_START,
                FLRW_ETA_END,
                True,
            )
        points = [
            _sample_conformal_diamond_point(
                rng,
                eta_start=eta_start,
                eta_end=eta_end,
                flrw_volume_weight=flrw_weight,
            )
            for _ in range(cardinality)
        ]
        nodes = [f"matched-{index:03d}" for index in range(cardinality)]
        edges = _causal_relations(nodes, points)
    elif geometry == "total_chain_negative":
        nodes = [f"matched-{index:03d}" for index in range(cardinality)]
        edges = [
            (nodes[index], nodes[index + 1])
            for index in range(cardinality - 1)
        ]
    elif geometry == "random_order_negative":
        rng = np.random.Generator(
            np.random.PCG64(seed + 900_000 + 10_000 * cardinality)
        )
        nodes = [f"matched-{index:03d}" for index in range(cardinality)]
        split = cardinality // 2
        edges = [
            (nodes[left], nodes[right])
            for left in range(split)
            for right in range(split, cardinality)
            if float(rng.random()) < RANDOM_ORDER_EDGE_PROBABILITY
        ]
    else:
        raise CausetDiagnosticError("unknown matched-cardinality control")
    feature = _feature_from_statistics(poset_statistics(nodes, edges))
    return {
        **feature,
        "feature_sha256": _sha(feature),
    }


def _centroid(features: list[Mapping[str, Any]]) -> list[float]:
    vectors = [_feature_vector(feature) for feature in features]
    return [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(len(vectors[0]))
    ]


def _rms_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(
        sum((a - b) ** 2 for a, b in zip(left, right, strict=True))
        / len(left)
    )


def _exact_local_interval_features() -> list[dict[str, Any]]:
    receipt = json.loads(LOCAL_DOMAIN_RECEIPT.read_text(encoding="ascii"))
    with gzip.open(LOCAL_DOMAIN_ARRAYS, "rb") as stream:
        arrays = np.load(io.BytesIO(stream.read()))
        direct = np.asarray(arrays["direct_ancestry_edges"], dtype=np.int64)
    count = int(receipt["event_count"])
    nodes = [f"local-event-{index:06d}" for index in range(count)]
    edges = [(nodes[int(left)], nodes[int(right)]) for left, right in direct]
    ordered, ancestors, descendants, global_heights = _transitive_data(
        nodes, edges
    )
    rows: list[dict[str, Any]] = []
    for future, pasts in enumerate(ancestors):
        for past in pasts:
            members = {past, future} | (
                descendants[past] & ancestors[future]
            )
            cardinality = len(members)
            if cardinality < MIN_DIMENSION_INTERVAL_SIZE:
                continue
            local_height: dict[int, int] = {}
            for node in sorted(members, key=lambda value: global_heights[value]):
                predecessors = ancestors[node] & members
                local_height[node] = 1 + max(
                    (local_height[parent] for parent in predecessors),
                    default=0,
                )
            comparable = sum(
                len(ancestors[node] & members) for node in members
            )
            histogram: dict[int, int] = {}
            for inner_future in members:
                for inner_past in ancestors[inner_future] & members:
                    inclusive_size = len(
                        descendants[inner_past] & ancestors[inner_future]
                    ) + 2
                    histogram[inclusive_size] = (
                        histogram.get(inclusive_size, 0) + 1
                    )
            n_zero = histogram.get(2, 0)
            feature = {
                "cardinality": cardinality,
                "ordering_fraction": (
                    2.0 * comparable / (cardinality * (cardinality - 1))
                ),
                "height_to_cardinality": max(local_height.values())
                / cardinality,
                "normalized_Nm_over_N0": [
                    (
                        histogram.get(open_size + 2, 0) / n_zero
                        if n_zero
                        else 0.0
                    )
                    for open_size in range(MATCHED_PROFILE_M_MAX + 1)
                ],
            }
            rows.append(
                {
                    "past_event_key": ordered[past],
                    "future_event_key": ordered[future],
                    **feature,
                    "is_total_order": comparable
                    == cardinality * (cardinality - 1) // 2,
                }
            )
    rows.sort(
        key=lambda row: (
            row["cardinality"],
            row["past_event_key"],
            row["future_event_key"],
        )
    )
    return rows


def _nearest_label(
    feature: Mapping[str, Any], centroids: Mapping[str, list[float]]
) -> tuple[str, dict[str, float]]:
    vector = _feature_vector(feature)
    distances = {
        label: _rms_distance(vector, centroid)
        for label, centroid in centroids.items()
    }
    return min(distances, key=lambda label: (distances[label], label)), distances


def _quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise CausetDiagnosticError("cannot summarize an empty feature population")

    def at(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        low = int(math.floor(position))
        high = int(math.ceil(position))
        if low == high:
            return ordered[low]
        weight = position - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    return {
        "minimum": ordered[0],
        "q25": at(0.25),
        "median": at(0.5),
        "q75": at(0.75),
        "maximum": ordered[-1],
    }


def _single_cutoff_matched_interval_comparison() -> dict[str, Any]:
    interval_rows = _exact_local_interval_features()
    cardinalities = sorted({int(row["cardinality"]) for row in interval_rows})
    labels = (
        "minkowski_3_plus_1",
        "de_sitter_flat_patch_3_plus_1",
        "total_chain_negative",
        "random_order_negative",
    )
    reference: dict[str, Any] = {}
    centroids_by_cardinality: dict[int, dict[str, list[float]]] = {}
    heldout_confusion = {
        label: {prediction: 0 for prediction in labels} for label in labels
    }
    for cardinality in cardinalities:
        reference_row: dict[str, Any] = {}
        centroids: dict[str, list[float]] = {}
        training_by_label: dict[str, list[dict[str, Any]]] = {}
        for label in labels:
            training = [
                _fixed_cardinality_control_feature(
                    geometry=label, cardinality=cardinality, seed=seed
                )
                for seed in MATCHED_TRAIN_SEEDS
            ]
            training_by_label[label] = training
            centroids[label] = _centroid(training)
            reference_row[label] = {
                "training_feature_sha256s": [
                    row["feature_sha256"] for row in training
                ],
                "centroid": centroids[label],
            }
        for label in labels:
            for seed in MATCHED_HELDOUT_SEEDS:
                heldout = _fixed_cardinality_control_feature(
                    geometry=label, cardinality=cardinality, seed=seed
                )
                prediction, _ = _nearest_label(heldout, centroids)
                heldout_confusion[label][prediction] += 1
        reference[str(cardinality)] = reference_row
        centroids_by_cardinality[cardinality] = centroids

    predictions = {label: 0 for label in labels}
    distance_rows: dict[str, list[float]] = {label: [] for label in labels}
    for interval in interval_rows:
        label, distances = _nearest_label(
            interval, centroids_by_cardinality[int(interval["cardinality"])]
        )
        predictions[label] += 1
        for reference_label, distance in distances.items():
            distance_rows[reference_label].append(distance)
    band_low, band_high = EXPLORATORY_4D_ORDERING_FRACTION_BAND
    in_band = sum(
        band_low <= float(row["ordering_fraction"]) <= band_high
        for row in interval_rows
    )
    manifold_labels = {
        "minkowski_3_plus_1",
        "de_sitter_flat_patch_3_plus_1",
    }
    negative_controls_distinguished = bool(
        heldout_confusion["total_chain_negative"]["total_chain_negative"]
        == len(cardinalities) * len(MATCHED_HELDOUT_SEEDS)
        and heldout_confusion["random_order_negative"]["random_order_negative"]
        == len(cardinalities) * len(MATCHED_HELDOUT_SEEDS)
        and all(
            sum(
                heldout_confusion[label][prediction]
                for prediction in manifold_labels
            )
            == len(cardinalities) * len(MATCHED_HELDOUT_SEEDS)
            for label in manifold_labels
        )
    )
    all_chainlike = bool(
        interval_rows
        and all(row["is_total_order"] for row in interval_rows)
        and predictions["total_chain_negative"] == len(interval_rows)
    )
    result = (
        "NOT_SIMILAR_AT_CURRENT_CUTOFF__CHAINLIKE_INTERVALS"
        if all_chainlike
        else (
            "NOT_SIMILAR_AT_CURRENT_CUTOFF__INTERVAL_ORDERING_FRACTIONS_"
            "OUTSIDE_EXPLORATORY_4D_BAND"
            if in_band == 0
            else "EXPLORATORY_MIXED_OR_NONCHAIN_INTERVALS"
        )
    )
    return {
        "epistemic_status": "POST_HOC_EXPLORATORY_SINGLE_CUTOFF",
        "result": result,
        "comparison_cutoff": "local_domain_MAIN_CONFIG_single_capture",
        "adequate_interval_count": len(interval_rows),
        "inclusive_cardinality_range": [
            min(cardinalities),
            max(cardinalities),
        ],
        "total_order_interval_count": sum(
            row["is_total_order"] for row in interval_rows
        ),
        "ordering_fraction_quantiles": _quantiles(
            [float(row["ordering_fraction"]) for row in interval_rows]
        ),
        "height_to_cardinality_quantiles": _quantiles(
            [float(row["height_to_cardinality"]) for row in interval_rows]
        ),
        "exploratory_4d_ordering_fraction_band": [band_low, band_high],
        "interval_count_in_exploratory_4d_band": in_band,
        "interval_fraction_in_exploratory_4d_band": in_band
        / len(interval_rows),
        "normalized_interval_profile": {
            "definition": "N_m/N_0 with m open-interval elements",
            "m_values": list(range(MATCHED_PROFILE_M_MAX + 1)),
        },
        "exact_interval_feature_population_sha256": _sha(interval_rows),
        "nearest_reference_counts": predictions,
        "nearest_reference_distance_quantiles": {
            label: _quantiles(values) for label, values in distance_rows.items()
        },
        "matched_cardinality_reference_centroids": reference,
        "heldout_classifier_confusion": heldout_confusion,
        "heldout_control_scope": (
            "distinguishes_the_two_manifold_reference_controls_as_a_group_"
            "from_chain_and_random_negatives_not_minkowski_from_de_sitter"
        ),
        "heldout_negative_controls_distinguished": (
            negative_controls_distinguished
        ),
        "physical_or_manifold_similarity_claimed": False,
        "refinement_family_inference_allowed": False,
    }


def _reference_controls() -> dict[str, Any]:
    dimensions: dict[str, Any] = {}
    for dimension in REFERENCE_DIMS:
        rows: list[dict[str, Any]] = []
        for seed in REFERENCE_SEEDS:
            nodes, edges = _minkowski_sprinkling(dimension, seed)
            stats = poset_statistics(nodes, edges)
            estimate = invert_myrheim_meyer_fraction(
                float(stats["global_ordering_fraction"])
            )
            rows.append(
                {
                    "seed": seed,
                    "event_count": stats["event_count"],
                    "ordering_fraction": stats["global_ordering_fraction"],
                    "estimated_dimension": estimate,
                }
            )
        estimates = [float(row["estimated_dimension"]) for row in rows if row["estimated_dimension"] is not None]
        mean_estimate = sum(estimates) / len(estimates)
        dimensions[str(dimension)] = {
            "runs": rows,
            "mean_estimated_dimension": mean_estimate,
            "absolute_error": abs(mean_estimate - dimension),
            "calibration_within_one_dimension": abs(mean_estimate - dimension) <= 1.0,
        }

    chain_nodes = [f"chain-{index:03d}" for index in range(64)]
    chain_edges = [
        (chain_nodes[index], chain_nodes[index + 1])
        for index in range(len(chain_nodes) - 1)
    ]
    forest_nodes = [f"forest-{index:03d}" for index in range(64)]
    forest_edges = [
        (forest_nodes[chain * 8 + offset], forest_nodes[chain * 8 + offset + 1])
        for chain in range(8)
        for offset in range(7)
    ]
    rng = np.random.Generator(np.random.PCG64(76399))
    random_nodes = [f"random-{index:03d}" for index in range(96)]
    random_edges = [
        (random_nodes[left], random_nodes[right])
        for left in range(96)
        for right in range(left + 1, 96)
        if float(rng.random()) < 0.035
    ]
    return {
        "minkowski_alexandrov_sprinklings": dimensions,
        "nested_poisson_density_controls": _nested_reference_controls(),
        "nonmanifold_controls": {
            "total_chain": poset_statistics(chain_nodes, chain_edges),
            "eight_disconnected_record_chains": poset_statistics(
                forest_nodes, forest_edges
            ),
            "random_dag": poset_statistics(random_nodes, random_edges),
        },
    }


def _local_domain_diagnostic() -> dict[str, Any]:
    """Replay the existing 2,304-event indexed order without re-running it."""

    receipt = json.loads(LOCAL_DOMAIN_RECEIPT.read_text(encoding="ascii"))
    with gzip.open(LOCAL_DOMAIN_ARRAYS, "rb") as stream:
        arrays = np.load(io.BytesIO(stream.read()))
        chart = np.asarray(arrays["chart"], dtype=float)
        causal_pair_sample = np.asarray(
            arrays["causal_pairs"], dtype=np.int64
        )
        if "direct_ancestry_edges" not in arrays.files:
            raise CausetDiagnosticError(
                "local-domain bundle lacks exact direct ancestry; a capped "
                "causal-pair sample cannot be treated as the full poset"
            )
        direct_ancestry = np.asarray(
            arrays["direct_ancestry_edges"], dtype=np.int64
        )
    specs = receipt["array_bundle_binding"]["array_specs"]
    if _sha(chart.tolist()) != specs["chart"]["value_sha256"]:
        raise CausetDiagnosticError("local-domain chart array binding mismatch")
    if _sha(causal_pair_sample.tolist()) != specs["causal_pairs"][
        "value_sha256"
    ]:
        raise CausetDiagnosticError("local-domain causal-pair sample mismatch")
    if _sha(direct_ancestry.tolist()) != specs["direct_ancestry_edges"][
        "value_sha256"
    ]:
        raise CausetDiagnosticError("local-domain direct-ancestry binding mismatch")
    event_count = int(receipt["event_count"])
    if chart.shape != (event_count, 4):
        raise CausetDiagnosticError("local-domain chart shape mismatch")
    nodes = [f"local-event-{index:06d}" for index in range(event_count)]
    if direct_ancestry.shape != (
        int(receipt["ancestry_edge_count"]),
        2,
    ):
        raise CausetDiagnosticError("local-domain direct-ancestry shape mismatch")
    edges = [
        (nodes[int(left)], nodes[int(right)]) for left, right in direct_ancestry
    ]
    stats = poset_statistics(nodes, edges)
    if stats["comparable_pair_count"] != int(receipt["causal_pair_total"]):
        raise CausetDiagnosticError(
            "local-domain exact closure disagrees with causal_pair_total"
        )
    depths, counts = np.unique(chart[:, 0], return_counts=True)
    interval_ready = bool(
        stats["adequate_dimension_interval_count"]
        >= MIN_INTERVAL_SAMPLE_COUNT
    )
    return {
        "stage1_receipt_sha256": _sha(receipt),
        "source_projection_sha256": receipt["source_projection_sha256"],
        "chart_value_sha256": specs["chart"]["value_sha256"],
        "direct_ancestry_value_sha256": specs["direct_ancestry_edges"][
            "value_sha256"
        ],
        "causal_pair_sample_value_sha256": specs["causal_pairs"][
            "value_sha256"
        ],
        "causal_pair_sample_count": int(causal_pair_sample.shape[0]),
        "causal_pair_total": int(receipt["causal_pair_total"]),
        "causal_pair_sample_is_full_closure": bool(
            causal_pair_sample.shape[0] == int(receipt["causal_pair_total"])
        ),
        "declared_ancestry_edge_count": int(receipt["ancestry_edge_count"]),
        "depth_coordinate_counts": {
            str(float(depth)): int(count)
            for depth, count in zip(depths, counts, strict=True)
        },
        "statistics": stats,
        "status": (
            "EXPLORATORY_INTERVAL_DIAGNOSTIC_READY_NOT_OPH_SIMILARITY"
            if interval_ready
            else "INCONCLUSIVE__INSUFFICIENT_CERTIFIED_INTERVAL_SIZE"
        ),
        "interpretation": (
            "Statistics are reconstructed from the exact direct generated "
            "ancestry, never from the capped causal-pair fit sample. The "
            f"largest interval has {stats['maximum_interval_size']} events "
            f"and {stats['adequate_dimension_interval_count']} intervals meet "
            f"the frozen size floor {MIN_DIMENSION_INTERVAL_SIZE}. This does "
            "not establish OPH similarity to any reference spacetime."
        ),
    }


def produce_causet_likeness_report(
    source_path: Path | str = DEFAULT_SOURCE,
) -> dict[str, Any]:
    source_file = Path(source_path)
    if not source_file.is_absolute():
        source_file = ROOT / source_file
    source_file = source_file.resolve()
    source = json.loads(source_file.read_text(encoding="ascii"))
    source_body = {key: value for key, value in source.items() if key != "report_sha256"}
    if _sha(source_body) != source.get("report_sha256"):
        raise CausetDiagnosticError("source receipt hash mismatch")
    if source.get("SOURCE_DERIVED_CAUSAL_ORDER_RECEIPT") is not True:
        raise CausetDiagnosticError("source-derived causal-order receipt is not attained")
    nodes = [str(event["event_key"]) for event in source["semantic_events"]]
    edges = [
        (str(row["parent_event_id"]), str(row["child_event_id"]))
        for row in source["generated_edges"]
    ]
    source_stats = poset_statistics(nodes, edges)
    repair_only = source["repair_only_event_carrier_control"]
    local_domain = _local_domain_diagnostic()
    single_cutoff = _single_cutoff_matched_interval_comparison()
    reference = _reference_controls()
    calibration_passed = all(
        row["calibration_within_one_dimension"]
        for row in reference["minkowski_alexandrov_sprinklings"].values()
    )

    baseline_projection = {
        "nodes": sorted(set(nodes)),
        "edges": sorted(set(edges)),
    }
    source_binding = {
        "source_receipt_path": str(source_file.relative_to(ROOT.resolve())),
        "source_report_sha256": source["report_sha256"],
        "semantic_poset_sha256": _sha(baseline_projection),
        "semantic_event_keys_sha256": _sha(sorted(set(nodes))),
        "semantic_events_sha256": _sha(source["semantic_events"]),
        "observer_event_log_sha256": source["observer_event_log_sha256"],
        "generated_edges_sha256": source["generated_edges_sha256"],
        "event_carrier_scope": source["event_carrier_scope"],
        "underlying_repair_transactions_promoted_as_events": source[
            "underlying_repair_transactions_promoted_as_events"
        ],
    }
    mutated_projection = {
        "nodes": baseline_projection["nodes"],
        "edges": baseline_projection["edges"][:-1],
    }
    mutated_source = json.loads(json.dumps(source))
    mutated_source["generated_edges"] = mutated_source["generated_edges"][:-1]
    mutated_source["generated_edges_sha256"] = _sha(
        mutated_source["generated_edges"]
    )
    mutated_source_body = {
        key: value for key, value in mutated_source.items() if key != "report_sha256"
    }
    mutated_source["report_sha256"] = _sha(mutated_source_body)
    permuted_stats = poset_statistics(reversed(nodes), reversed(edges))
    stutter_stats = poset_statistics([*nodes, nodes[0]], edges)
    controls = {
        "event_and_edge_order_permutation_invariant": permuted_stats == source_stats,
        "duplicate_semantic_id_input_is_set_idempotent": stutter_stats
        == source_stats,
        "source_receipt_hash_chain_replays": bool(
            source["report_sha256"] == _sha(source_body)
            and source["generated_edges_sha256"]
            == _sha(source["generated_edges"])
            and source["observer_event_log_sha256"]
            == source["observer_log_material"]["event_log_sha256"]
            == _raw_sha(source["observer_log_material"]["events"])
            and source_binding["semantic_event_keys_sha256"]
            == _sha(sorted(set(nodes)))
            and source_binding["semantic_events_sha256"]
            == _sha(source["semantic_events"])
        ),
        "source_edge_mutation_changes_bound_poset_hash": bool(
            _sha(mutated_projection) != _sha(baseline_projection)
            and _sha(source["generated_edges"][:-1])
            != source["generated_edges_sha256"]
            and mutated_source["report_sha256"]
            != source_binding["source_report_sha256"]
            and mutated_source["generated_edges_sha256"]
            != source_binding["generated_edges_sha256"]
            and _sha(mutated_projection)
            != source_binding["semantic_poset_sha256"]
        ),
        "minkowski_dimension_calibration_within_one_dimension": calibration_passed,
        "synthetic_minkowski_and_de_sitter_nested_inclusions_certified": all(
            row["all_nested_inclusion_couplings_certified"]
            for row in reference["nested_poisson_density_controls"][
                "geometries"
            ].values()
        ),
        "flat_myrheim_meyer_excluded_from_flrw_controls": all(
            level["flat_myrheim_meyer_dimension_estimate"] is None
            and level["flat_myrheim_meyer_status"]
            == "NOT_APPLIED_CURVED_FLRW_REFERENCE_CONTROL"
            for run in reference["nested_poisson_density_controls"][
                "geometries"
            ]["de_sitter_flat_patch_3_plus_1"]["runs"]
            for level in run["levels"]
        ),
        "matched_interval_heldout_controls_distinguish_manifold_group_from_negatives": bool(
            single_cutoff["heldout_negative_controls_distinguished"]
        ),
        "single_cutoff_result_replays_exact_interval_population": bool(
            single_cutoff["adequate_interval_count"] > 0
            and sum(single_cutoff["nearest_reference_counts"].values())
            == single_cutoff["adequate_interval_count"]
            and 0
            <= single_cutoff["interval_count_in_exploratory_4d_band"]
            <= single_cutoff["adequate_interval_count"]
        ),
        "chain_and_record_forest_controls_distinguished": bool(
            reference["nonmanifold_controls"]["total_chain"]["width"] == 1
            and reference["nonmanifold_controls"][
                "eight_disconnected_record_chains"
            ]["weak_component_count"]
            == 8
        ),
    }
    enough_intervals = bool(
        source_stats["adequate_dimension_interval_count"]
        >= MIN_INTERVAL_SAMPLE_COUNT
    )
    source_control_status = (
        "EXPLORATORY_COMPARISON_READY_NOT_CONFIRMED"
        if enough_intervals
        else "INCONCLUSIVE__INSUFFICIENT_CERTIFIED_INTERVAL_SIZE"
    )
    report = {
        "schema": SCHEMA,
        "artifact_type": "BOUNDED_EXPLORATORY_CAUSAL_SET_DIAGNOSTIC",
        "status": single_cutoff["result"],
        "source_control_status": source_control_status,
        "source_binding": source_binding,
        "frozen_config": {
            "minimum_dimension_interval_size": MIN_DIMENSION_INTERVAL_SIZE,
            "minimum_interval_sample_count": MIN_INTERVAL_SAMPLE_COUNT,
            "reference_dimensions": list(REFERENCE_DIMS),
            "reference_seeds": list(REFERENCE_SEEDS),
            "reference_mean_poisson_count": REFERENCE_MEAN_COUNT,
            "density_control_poisson_mean_counts": list(
                DENSITY_CONTROL_MEAN_COUNTS
            ),
            "density_control_reference_dimension": DENSITY_CONTROL_DIMENSION,
            "flrw_geometry": "spatially_flat_de_sitter_conformal_patch",
            "flrw_scale_factor": "a(eta)=-1/(H*eta)",
            "flrw_hubble": FLRW_HUBBLE,
            "flrw_diamond_eta_endpoints": [FLRW_ETA_START, FLRW_ETA_END],
            "flrw_volume_density": "a(eta)^4*deta*d^3x",
            "causal_order": "conformal_light_cone",
            "density_control_coupling": (
                "nested_poisson_thinning_from_mean_256"
            ),
            "matched_profile_m_max": MATCHED_PROFILE_M_MAX,
            "matched_training_seeds": list(MATCHED_TRAIN_SEEDS),
            "matched_heldout_seeds": list(MATCHED_HELDOUT_SEEDS),
            "random_order_edge_probability": RANDOM_ORDER_EDGE_PROBABILITY,
            "random_order_model": (
                "balanced_two_layer_bipartite_bernoulli_order"
            ),
            "exploratory_4d_ordering_fraction_band": list(
                EXPLORATORY_4D_ORDERING_FRACTION_BAND
            ),
            "rng": "numpy_generator_pcg64_v1",
            "numpy_version": np.__version__,
            "poisson_sampler": "numpy.random.Generator.poisson",
        },
        "source_statistics": source_stats,
        "event_carrier_selection_controls": {
            "observer_instrumentation_history_classification": (
                "EVALUATED_FINITE_ORDER_NOT_COMPLETE_PHYSICAL_EVENT_CARRIER"
            ),
            "repair_only_control_sha256": _sha(repair_only),
            "repair_only_event_count": repair_only["repair_event_count"],
            "repair_only_versioned_provenance_edge_count": repair_only[
                "versioned_provenance_edge_count"
            ],
            "repair_only_classification": repair_only["classification"],
            "required_physical_event_carrier_change": repair_only[
                "required_model_change"
            ],
        },
        "existing_local_domain_diagnostic": local_domain,
        "single_cutoff_matched_interval_comparison": single_cutoff,
        "reference_controls": reference,
        "invariance_controls": controls,
        "controls_fail_closed": all(controls.values()),
        "refinement_invariance_status": (
            "NOT_EVALUATED_NO_CERTIFIED_REFINEMENT_MAP_IN_SOURCE_RECEIPT"
        ),
        "oph_refinement_family_comparison": {
            "status": "NOT_EVALUATED_NO_CERTIFIED_OPH_REFINEMENT_MAP_OR_FAMILY",
            "source_control_too_small_for_interval_comparison": (
                not enough_intervals
            ),
            "local_single_cutoff_comparison_completed": True,
            "local_single_cutoff_result": single_cutoff["result"],
            "certified_oph_refinement_map_or_family_available": False,
            "similarity_claimed": False,
        },
        "held_out_confirmation_status": "NOT_RUN_EXPLORATORY_ONLY",
        "flrw_reference_control_status": (
            "IMPLEMENTED_REPLAYED_DE_SITTER_FLAT_PATCH_SPECIAL_FLRW_REFERENCE_ONLY"
        ),
        "CAUSET_DIAGNOSTIC_PIPELINE_REPRODUCTION_RECEIPT": bool(
            all(controls.values())
        ),
        "OPH_CAUSAL_SET_SIMILARITY_RECEIPT": False,
        "CAUSET_MANIFOLDLIKE_RECEIPT": False,
        "physical_promotion_allowed": False,
        "required_next_capture": (
            "promote the versioned seam-repair transactions as the physical "
            "event carrier, construct a certified OPH refinement family/map, "
            "and freeze a held-out comparison before evaluating that family"
        ),
        "research_basis": [
            {
                "diagnostic": "ordering_fraction_and_myrheim_meyer_dimension",
                "reference": "D. A. Meyer, The Dimension of Causal Sets (1988)",
            },
            {
                "diagnostic": "interval_abundance_locality",
                "reference": (
                    "L. Glaser and S. Surya, Phys. Rev. D 88, 124026 (2013), "
                    "arXiv:1309.3403"
                ),
            },
            {
                "diagnostic": "conformally_flat_curved_spacetime_dimension_tests",
                "reference": (
                    "D. D. Reid, Phys. Rev. D 67, 024034 (2003), "
                    "doi:10.1103/PhysRevD.67.024034"
                ),
            },
            {
                "diagnostic": "small_curved_causal_diamond_corrections",
                "reference": (
                    "M. Roy, D. Sinha, and S. Surya, Phys. Rev. D 87, "
                    "044046 (2013), doi:10.1103/PhysRevD.87.044046"
                ),
            },
        ],
        "claim_boundary": (
            "Necessary finite-order diagnostics only. The pilot does not "
            "establish physical causal faithfulness, faithful embedding, "
            "manifoldlikeness, dimension 3+1 or any dimension of the OPH "
            "source, volume density, Lorentzian geometry, FLRW behavior, or "
            "a continuum limit. A global ordering fraction is not interpreted "
            "as a Myrheim--Meyer dimension unless the population is a verified "
            "flat Alexandrov interval; it is deliberately not applied to the "
            "curved FLRW controls. The synthetic nested density/carrier-"
            "inclusion Poisson controls are calibrations, not OPH refinement "
            "evidence. "
            "The source order is an observer instrumentation history over "
            "state snapshots, not the complete seam-repair event history. "
            "The bounded repair-only carrier control is an antichain because "
            "all recorded repair reads consume distinguished version-zero "
            "roots; recurrent local propagation is not interleaved as events. "
            f"The post-hoc single-cutoff result is {single_cutoff['result']}; "
            "that finite exploratory comparison is not a physical "
            "nonexistence result. "
            "INCONCLUSIVE__INSUFFICIENT_CERTIFIED_INTERVAL_SIZE is not a negative physical "
            "verdict."
        ),
    }
    report["report_sha256"] = _sha(report)
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    report = produce_causet_likeness_report(args.source)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(report))
    stats = report["source_statistics"]
    print(
        f"{output}: status={report['status']} n={stats['event_count']} "
        f"height={stats['height']} width={stats['width']} "
        f"max_interval={stats['maximum_interval_size']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
