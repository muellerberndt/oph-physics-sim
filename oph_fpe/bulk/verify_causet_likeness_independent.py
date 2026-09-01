"""Independent replay verifier for the bounded causal-set diagnostic."""

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
DEFAULT_RECEIPT = ROOT / "data/causal_order/causet_likeness_receipt.json"
EXPECTED_SCHEMA = "oph.causet-likeness-exploratory.v1"
MIN_INTERVAL = 32
MIN_SAMPLE = 4
DIMS = (2, 3, 4, 5)
SEEDS = (76301, 76302, 76303, 76304)
MEAN_COUNT = 128
DENSITY_CONTROL_MEANS = (64, 128, 256)
DENSITY_CONTROL_DIMENSION = 4
MATCHED_PROFILE_M_MAX = 15
MATCHED_TRAIN_SEEDS = (76401, 76402, 76403, 76404)
MATCHED_HELDOUT_SEEDS = (76411, 76412, 76413, 76414)
RANDOM_ORDER_EDGE_PROBABILITY = 0.5
EXPLORATORY_4D_ORDERING_FRACTION_BAND = (0.05, 0.20)
FLRW_ETA_START = -2.0
FLRW_ETA_END = -1.0
FLRW_HUBBLE = 1.0
LOCAL_RECEIPT = ROOT / "data/local_domain/stage1_receipt.json"
LOCAL_ARRAYS = ROOT / "data/local_domain/stage1_arrays.npz.gz"


class IndependentCausetVerificationError(RuntimeError):
    """Raised when any diagnostic clause fails independent replay."""


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


def _fail(message: str) -> None:
    raise IndependentCausetVerificationError(message)


def _closure(
    nodes: Iterable[str], edges: Iterable[tuple[str, str]]
) -> tuple[list[str], list[set[int]], list[set[int]], list[int], list[tuple[int, int]]]:
    ordered = sorted(set(str(node) for node in nodes))
    if not ordered:
        _fail("empty source carrier")
    index = {node: position for position, node in enumerate(ordered)}
    children = [set() for _ in ordered]
    indegree = [0] * len(ordered)
    indexed_edges: list[tuple[int, int]] = []
    for parent_name, child_name in sorted(set(edges)):
        if parent_name not in index or child_name not in index:
            _fail("edge endpoint missing")
        parent, child = index[parent_name], index[child_name]
        if parent == child:
            _fail("self edge")
        if child not in children[parent]:
            children[parent].add(child)
            indegree[child] += 1
            indexed_edges.append((parent, child))
    frontier = sorted(node for node, degree in enumerate(indegree) if degree == 0)
    order: list[int] = []
    work = list(indegree)
    while frontier:
        node = frontier.pop()
        order.append(node)
        for child in sorted(children[node]):
            work[child] -= 1
            if work[child] == 0:
                frontier.append(child)
    if len(order) != len(ordered):
        _fail("cycle detected")
    ancestors = [set() for _ in ordered]
    descendants = [set() for _ in ordered]
    heights = [1] * len(ordered)
    for node in order:
        for child in children[node]:
            ancestors[child].add(node)
            ancestors[child].update(ancestors[node])
            heights[child] = max(heights[child], heights[node] + 1)
    for node in reversed(order):
        for child in children[node]:
            descendants[node].add(child)
            descendants[node].update(descendants[child])
    return ordered, ancestors, descendants, heights, indexed_edges


def _width(ancestors: list[set[int]]) -> int:
    successors = [set() for _ in ancestors]
    for child, parents in enumerate(ancestors):
        for parent in parents:
            successors[parent].add(child)
    matched = [-1] * len(ancestors)

    def augment(left: int, seen: set[int]) -> bool:
        for right in sorted(successors[left]):
            if right in seen:
                continue
            seen.add(right)
            if matched[right] < 0 or augment(matched[right], seen):
                matched[right] = left
                return True
        return False

    return len(ancestors) - sum(
        augment(left, set()) for left in range(len(ancestors))
    )


def _components(count: int, edges: list[tuple[int, int]]) -> list[int]:
    adjacency = [set() for _ in range(count)]
    for parent, child in edges:
        adjacency[parent].add(child)
        adjacency[child].add(parent)
    unseen = set(range(count))
    sizes: list[int] = []
    while unseen:
        stack = [unseen.pop()]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            neighbors = unseen & adjacency[node]
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def _stats(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> dict[str, Any]:
    node_list = sorted(set(nodes))
    edge_list = sorted(set(edges))
    ordered, ancestors, descendants, heights, indexed_edges = _closure(
        node_list, edge_list
    )
    comparable = sum(len(row) for row in ancestors)
    count = len(ordered)
    fraction = 2.0 * comparable / (count * (count - 1)) if count > 1 else 0.0
    histogram: dict[int, int] = {}
    intervals: list[int] = []
    for future, pasts in enumerate(ancestors):
        for past in pasts:
            size = len(descendants[past] & ancestors[future]) + 2
            histogram[size] = histogram.get(size, 0) + 1
            intervals.append(size)
    adequate = sum(size >= MIN_INTERVAL for size in intervals)
    width = _width(ancestors)
    components = _components(count, indexed_edges)
    cover_edges = [
        (past, future)
        for future, pasts in enumerate(ancestors)
        for past in pasts
        if not (descendants[past] & ancestors[future])
    ]
    cover_components = _components(count, cover_edges)
    cover_cycle_rank = len(cover_edges) - count + len(cover_components)
    return {
        "event_count": count,
        "input_edge_count": len(edge_list),
        "comparable_pair_count": comparable,
        "global_ordering_fraction": fraction,
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
        "interval_count": len(intervals),
        "interval_abundance_inclusive_size": {
            str(size): histogram[size] for size in sorted(histogram)
        },
        "maximum_interval_size": max(intervals, default=0),
        "adequate_dimension_interval_minimum_size": MIN_INTERVAL,
        "adequate_dimension_interval_count": adequate,
        "myrheim_meyer_dimension_estimate": None,
        "myrheim_meyer_status": (
            "NOT_EVALUATED_GLOBAL_REGION_NOT_VERIFIED_ALEXANDROV_INTERVAL"
        ),
        "glaser_surya_interval_abundance_status": (
            "ELIGIBLE_FOR_EXPLORATORY_SINGLE_CUTOFF_COMPARISON"
            if adequate >= MIN_SAMPLE
            else "NOT_EVALUATED_INSUFFICIENT_DEEP_INTERVALS"
        ),
    }


def _fraction(dimension: float) -> float:
    return math.exp(
        math.lgamma(dimension + 1.0)
        + math.lgamma(dimension / 2.0)
        - math.log(2.0)
        - math.lgamma(3.0 * dimension / 2.0)
    )


def _invert(value: float) -> float | None:
    if not 0.0 < value < 1.0 or not _fraction(20.0) <= value <= _fraction(1.01):
        return None
    low, high = 1.01, 20.0
    for _ in range(96):
        midpoint = (low + high) / 2.0
        if _fraction(midpoint) > value:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def _sprinkling(dimension: int, seed: int) -> tuple[list[str], list[tuple[str, str]]]:
    rng = np.random.Generator(np.random.PCG64(seed + 1000 * dimension))
    count = max(32, int(rng.poisson(MEAN_COUNT)))
    spatial = dimension - 1
    points: list[np.ndarray] = []
    while len(points) < count:
        time = float(rng.uniform(-0.5, 0.5))
        radius_limit = 0.5 - abs(time)
        if float(rng.random()) > (2.0 * radius_limit) ** spatial:
            continue
        direction = rng.normal(size=spatial)
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            continue
        radius = radius_limit * float(rng.random()) ** (1.0 / spatial)
        points.append(np.asarray([time, *(direction * radius / norm)]))
    nodes = [f"p-{index:04d}" for index in range(count)]
    edges: list[tuple[str, str]] = []
    for left in range(count):
        for right in range(left + 1, count):
            first, second = points[left], points[right]
            if first[0] <= second[0]:
                past, future, past_id, future_id = first, second, left, right
            else:
                past, future, past_id, future_id = second, first, right, left
            dt = float(future[0] - past[0])
            dx = future[1:] - past[1:]
            if dt > 0.0 and dt * dt >= float(np.dot(dx, dx)):
                edges.append((nodes[past_id], nodes[future_id]))
    return nodes, edges


def _ball(
    rng: np.random.Generator, spatial_dimension: int, radius_limit: float
) -> np.ndarray:
    while True:
        direction = rng.normal(size=spatial_dimension)
        norm = float(np.linalg.norm(direction))
        if norm > 0.0:
            break
    radius = radius_limit * float(rng.random()) ** (1.0 / spatial_dimension)
    return direction * (radius / norm)


def _diamond_point(
    rng: np.random.Generator,
    *,
    eta_start: float,
    eta_end: float,
    flrw_weight: bool,
) -> np.ndarray:
    midpoint = 0.5 * (eta_start + eta_end)
    half_duration = 0.5 * (eta_end - eta_start)
    if half_duration <= 0.0:
        _fail("invalid conformal-diamond endpoints")
    if flrw_weight and not (eta_start < eta_end < 0.0):
        _fail("invalid de Sitter conformal patch")
    maximum_scale = -1.0 / (FLRW_HUBBLE * eta_end) if flrw_weight else 1.0
    while True:
        eta = float(rng.uniform(eta_start, eta_end))
        radius_limit = half_duration - abs(eta - midpoint)
        probability = (radius_limit / half_duration) ** 3
        if flrw_weight:
            scale = -1.0 / (FLRW_HUBBLE * eta)
            probability *= (scale / maximum_scale) ** 4
        if float(rng.random()) <= probability:
            return np.asarray([eta, *_ball(rng, 3, radius_limit)], dtype=float)


def _relations(
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


def _compact(stats: Mapping[str, Any]) -> dict[str, Any]:
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


def _nested_family(*, geometry: str, seed: int) -> dict[str, Any]:
    if geometry not in {"minkowski_3_plus_1", "de_sitter_flat_patch_3_plus_1"}:
        _fail("unknown nested reference geometry")
    offset = 400_000 if geometry == "minkowski_3_plus_1" else 500_000
    rng = np.random.Generator(np.random.PCG64(seed + offset))
    maximum_mean = max(DENSITY_CONTROL_MEANS)
    maximum_count = int(rng.poisson(maximum_mean))
    if geometry == "minkowski_3_plus_1":
        eta_start, eta_end, flrw_weight = -0.5, 0.5, False
    else:
        eta_start, eta_end, flrw_weight = FLRW_ETA_START, FLRW_ETA_END, True
    points = [
        _diamond_point(
            rng,
            eta_start=eta_start,
            eta_end=eta_end,
            flrw_weight=flrw_weight,
        )
        for _ in range(maximum_count)
    ]
    marks = np.asarray(rng.random(maximum_count), dtype=float)
    all_nodes = [f"{geometry}-s{seed}-{index:04d}" for index in range(maximum_count)]
    all_edges = _relations(all_nodes, points)
    levels: list[dict[str, Any]] = []
    carriers: list[set[str]] = []
    relations: list[set[tuple[str, str]]] = []
    for mean_count in DENSITY_CONTROL_MEANS:
        selected_indices = [
            index
            for index, mark in enumerate(marks)
            if float(mark) <= mean_count / maximum_mean
        ]
        selected_nodes = [all_nodes[index] for index in selected_indices]
        selected_set = set(selected_nodes)
        selected_edges = [
            edge
            for edge in all_edges
            if edge[0] in selected_set and edge[1] in selected_set
        ]
        stats = _stats(selected_nodes, selected_edges)
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
            "statistics": _compact(stats),
        }
        if geometry == "minkowski_3_plus_1":
            row["flat_myrheim_meyer_dimension_estimate"] = _invert(
                float(stats["global_ordering_fraction"])
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


def _nested_controls() -> dict[str, Any]:
    geometries = {
        geometry: {
            "runs": [
                _nested_family(geometry=geometry, seed=seed) for seed in SEEDS
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
        "mean_counts": list(DENSITY_CONTROL_MEANS),
        "geometries": geometries,
        "oph_comparison_status": (
            "NOT_EVALUATED_NO_CERTIFIED_OPH_REFINEMENT_MAP_OR_FAMILY"
        ),
        "cross_geometry_similarity_claimed": False,
    }


def _feature(stats: Mapping[str, Any]) -> dict[str, Any]:
    count = int(stats["event_count"])
    histogram = {
        int(size): int(value)
        for size, value in stats["interval_abundance_inclusive_size"].items()
    }
    n_zero = histogram.get(2, 0)
    return {
        "cardinality": count,
        "ordering_fraction": float(stats["global_ordering_fraction"]),
        "height_to_cardinality": float(stats["height"]) / max(count, 1),
        "normalized_Nm_over_N0": [
            (histogram.get(open_size + 2, 0) / n_zero if n_zero else 0.0)
            for open_size in range(MATCHED_PROFILE_M_MAX + 1)
        ],
    }


def _vector(feature: Mapping[str, Any]) -> list[float]:
    return [
        float(feature["ordering_fraction"]),
        float(feature["height_to_cardinality"]),
        *[float(value) for value in feature["normalized_Nm_over_N0"]],
    ]


def _fixed_feature(
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
            _diamond_point(
                rng,
                eta_start=eta_start,
                eta_end=eta_end,
                flrw_weight=flrw_weight,
            )
            for _ in range(cardinality)
        ]
        nodes = [f"matched-{index:03d}" for index in range(cardinality)]
        edges = _relations(nodes, points)
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
        _fail("unknown matched-cardinality control")
    feature = _feature(_stats(nodes, edges))
    return {**feature, "feature_sha256": _sha(feature)}


def _mean_vector(features: list[Mapping[str, Any]]) -> list[float]:
    vectors = [_vector(feature) for feature in features]
    return [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(len(vectors[0]))
    ]


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(
        sum((a - b) ** 2 for a, b in zip(left, right, strict=True))
        / len(left)
    )


def _local_interval_features() -> list[dict[str, Any]]:
    receipt = json.loads(LOCAL_RECEIPT.read_text(encoding="ascii"))
    with gzip.open(LOCAL_ARRAYS, "rb") as stream:
        arrays = np.load(io.BytesIO(stream.read()))
        direct = np.asarray(arrays["direct_ancestry_edges"], dtype=np.int64)
    count = int(receipt["event_count"])
    nodes = [f"local-event-{index:06d}" for index in range(count)]
    edges = [(nodes[int(left)], nodes[int(right)]) for left, right in direct]
    ordered, ancestors, descendants, global_heights, _ = _closure(nodes, edges)
    rows: list[dict[str, Any]] = []
    for future, pasts in enumerate(ancestors):
        for past in pasts:
            members = {past, future} | (
                descendants[past] & ancestors[future]
            )
            cardinality = len(members)
            if cardinality < MIN_INTERVAL:
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
                    size = len(
                        descendants[inner_past] & ancestors[inner_future]
                    ) + 2
                    histogram[size] = histogram.get(size, 0) + 1
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


def _classify(
    feature: Mapping[str, Any], centroids: Mapping[str, list[float]]
) -> tuple[str, dict[str, float]]:
    vector = _vector(feature)
    distances = {
        label: _distance(vector, centroid)
        for label, centroid in centroids.items()
    }
    return min(distances, key=lambda label: (distances[label], label)), distances


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        _fail("empty feature summary")

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


def _matched_comparison() -> dict[str, Any]:
    interval_rows = _local_interval_features()
    cardinalities = sorted({int(row["cardinality"]) for row in interval_rows})
    labels = (
        "minkowski_3_plus_1",
        "de_sitter_flat_patch_3_plus_1",
        "total_chain_negative",
        "random_order_negative",
    )
    reference: dict[str, Any] = {}
    centroids_by_cardinality: dict[int, dict[str, list[float]]] = {}
    confusion = {
        label: {prediction: 0 for prediction in labels} for label in labels
    }
    for cardinality in cardinalities:
        reference_row: dict[str, Any] = {}
        centroids: dict[str, list[float]] = {}
        for label in labels:
            training = [
                _fixed_feature(
                    geometry=label, cardinality=cardinality, seed=seed
                )
                for seed in MATCHED_TRAIN_SEEDS
            ]
            centroids[label] = _mean_vector(training)
            reference_row[label] = {
                "training_feature_sha256s": [
                    row["feature_sha256"] for row in training
                ],
                "centroid": centroids[label],
            }
        for label in labels:
            for seed in MATCHED_HELDOUT_SEEDS:
                heldout = _fixed_feature(
                    geometry=label, cardinality=cardinality, seed=seed
                )
                prediction, _ = _classify(heldout, centroids)
                confusion[label][prediction] += 1
        reference[str(cardinality)] = reference_row
        centroids_by_cardinality[cardinality] = centroids
    predictions = {label: 0 for label in labels}
    distance_rows: dict[str, list[float]] = {label: [] for label in labels}
    for interval in interval_rows:
        label, distances = _classify(
            interval, centroids_by_cardinality[int(interval["cardinality"])]
        )
        predictions[label] += 1
        for reference_label, distance in distances.items():
            distance_rows[reference_label].append(distance)
    low, high = EXPLORATORY_4D_ORDERING_FRACTION_BAND
    in_band = sum(
        low <= float(row["ordering_fraction"]) <= high
        for row in interval_rows
    )
    manifold = {"minkowski_3_plus_1", "de_sitter_flat_patch_3_plus_1"}
    controls_distinguished = bool(
        confusion["total_chain_negative"]["total_chain_negative"]
        == len(cardinalities) * len(MATCHED_HELDOUT_SEEDS)
        and confusion["random_order_negative"]["random_order_negative"]
        == len(cardinalities) * len(MATCHED_HELDOUT_SEEDS)
        and all(
            sum(confusion[label][prediction] for prediction in manifold)
            == len(cardinalities) * len(MATCHED_HELDOUT_SEEDS)
            for label in manifold
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
        "inclusive_cardinality_range": [min(cardinalities), max(cardinalities)],
        "total_order_interval_count": sum(
            row["is_total_order"] for row in interval_rows
        ),
        "ordering_fraction_quantiles": _summary(
            [float(row["ordering_fraction"]) for row in interval_rows]
        ),
        "height_to_cardinality_quantiles": _summary(
            [float(row["height_to_cardinality"]) for row in interval_rows]
        ),
        "exploratory_4d_ordering_fraction_band": [low, high],
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
            label: _summary(values) for label, values in distance_rows.items()
        },
        "matched_cardinality_reference_centroids": reference,
        "heldout_classifier_confusion": confusion,
        "heldout_control_scope": (
            "distinguishes_the_two_manifold_reference_controls_as_a_group_"
            "from_chain_and_random_negatives_not_minkowski_from_de_sitter"
        ),
        "heldout_negative_controls_distinguished": controls_distinguished,
        "physical_or_manifold_similarity_claimed": False,
        "refinement_family_inference_allowed": False,
    }


def _references() -> dict[str, Any]:
    dimensions: dict[str, Any] = {}
    for dimension in DIMS:
        rows: list[dict[str, Any]] = []
        for seed in SEEDS:
            nodes, edges = _sprinkling(dimension, seed)
            stats = _stats(nodes, edges)
            estimate = _invert(float(stats["global_ordering_fraction"]))
            rows.append(
                {
                    "seed": seed,
                    "event_count": stats["event_count"],
                    "ordering_fraction": stats["global_ordering_fraction"],
                    "estimated_dimension": estimate,
                }
            )
        estimates = [float(row["estimated_dimension"]) for row in rows if row["estimated_dimension"] is not None]
        mean = sum(estimates) / len(estimates)
        dimensions[str(dimension)] = {
            "runs": rows,
            "mean_estimated_dimension": mean,
            "absolute_error": abs(mean - dimension),
            "calibration_within_one_dimension": abs(mean - dimension) <= 1.0,
        }
    chain_nodes = [f"chain-{index:03d}" for index in range(64)]
    chain_edges = [(chain_nodes[index], chain_nodes[index + 1]) for index in range(63)]
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
        "nested_poisson_density_controls": _nested_controls(),
        "nonmanifold_controls": {
            "total_chain": _stats(chain_nodes, chain_edges),
            "eight_disconnected_record_chains": _stats(forest_nodes, forest_edges),
            "random_dag": _stats(random_nodes, random_edges),
        },
    }


def _local_domain_diagnostic() -> dict[str, Any]:
    receipt = json.loads(LOCAL_RECEIPT.read_text(encoding="ascii"))
    with gzip.open(LOCAL_ARRAYS, "rb") as stream:
        arrays = np.load(io.BytesIO(stream.read()))
        chart = np.asarray(arrays["chart"], dtype=float)
        causal_pair_sample = np.asarray(
            arrays["causal_pairs"], dtype=np.int64
        )
        if "direct_ancestry_edges" not in arrays.files:
            _fail(
                "local bundle lacks exact direct ancestry; capped pair samples "
                "cannot define the full poset"
            )
        direct_ancestry = np.asarray(
            arrays["direct_ancestry_edges"], dtype=np.int64
        )
    specs = receipt["array_bundle_binding"]["array_specs"]
    if _sha(chart.tolist()) != specs["chart"]["value_sha256"]:
        _fail("local chart binding mismatch")
    if _sha(causal_pair_sample.tolist()) != specs["causal_pairs"][
        "value_sha256"
    ]:
        _fail("local causal-pair sample binding mismatch")
    if _sha(direct_ancestry.tolist()) != specs["direct_ancestry_edges"][
        "value_sha256"
    ]:
        _fail("local direct-ancestry binding mismatch")
    count = int(receipt["event_count"])
    if chart.shape != (count, 4):
        _fail("local chart shape mismatch")
    nodes = [f"local-event-{index:06d}" for index in range(count)]
    if direct_ancestry.shape != (int(receipt["ancestry_edge_count"]), 2):
        _fail("local direct-ancestry shape mismatch")
    edges = [
        (nodes[int(left)], nodes[int(right)]) for left, right in direct_ancestry
    ]
    stats = _stats(nodes, edges)
    if stats["comparable_pair_count"] != int(receipt["causal_pair_total"]):
        _fail("local exact closure disagrees with causal_pair_total")
    depths, depth_counts = np.unique(chart[:, 0], return_counts=True)
    interval_ready = bool(
        stats["adequate_dimension_interval_count"] >= MIN_SAMPLE
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
            str(float(depth)): int(item_count)
            for depth, item_count in zip(depths, depth_counts, strict=True)
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
            f"the frozen size floor {MIN_INTERVAL}. This does not establish "
            "OPH similarity to any reference spacetime."
        ),
    }


def verify_receipt(path: Path | str = DEFAULT_RECEIPT) -> dict[str, Any]:
    receipt = json.loads(Path(path).read_text(encoding="ascii"))
    if receipt.get("schema") != EXPECTED_SCHEMA:
        _fail("unexpected schema")
    body = {key: value for key, value in receipt.items() if key != "report_sha256"}
    if _sha(body) != receipt.get("report_sha256"):
        _fail("report hash mismatch")
    expected_config = {
        "minimum_dimension_interval_size": MIN_INTERVAL,
        "minimum_interval_sample_count": MIN_SAMPLE,
        "reference_dimensions": list(DIMS),
        "reference_seeds": list(SEEDS),
        "reference_mean_poisson_count": MEAN_COUNT,
        "density_control_poisson_mean_counts": list(DENSITY_CONTROL_MEANS),
        "density_control_reference_dimension": DENSITY_CONTROL_DIMENSION,
        "flrw_geometry": "spatially_flat_de_sitter_conformal_patch",
        "flrw_scale_factor": "a(eta)=-1/(H*eta)",
        "flrw_hubble": FLRW_HUBBLE,
        "flrw_diamond_eta_endpoints": [FLRW_ETA_START, FLRW_ETA_END],
        "flrw_volume_density": "a(eta)^4*deta*d^3x",
        "causal_order": "conformal_light_cone",
        "density_control_coupling": "nested_poisson_thinning_from_mean_256",
        "matched_profile_m_max": MATCHED_PROFILE_M_MAX,
        "matched_training_seeds": list(MATCHED_TRAIN_SEEDS),
        "matched_heldout_seeds": list(MATCHED_HELDOUT_SEEDS),
        "random_order_edge_probability": RANDOM_ORDER_EDGE_PROBABILITY,
        "random_order_model": "balanced_two_layer_bipartite_bernoulli_order",
        "exploratory_4d_ordering_fraction_band": list(
            EXPLORATORY_4D_ORDERING_FRACTION_BAND
        ),
        "rng": "numpy_generator_pcg64_v1",
        "numpy_version": np.__version__,
        "poisson_sampler": "numpy.random.Generator.poisson",
    }
    if receipt.get("frozen_config") != expected_config:
        _fail("frozen config mismatch")
    binding = receipt.get("source_binding") or {}
    if binding.get("source_receipt_path") != (
        "data/causal_order/source_derived_causal_order_receipt.json"
    ):
        _fail("unexpected source path")
    source = json.loads((ROOT / binding["source_receipt_path"]).read_text(encoding="ascii"))
    source_body = {key: value for key, value in source.items() if key != "report_sha256"}
    if _sha(source_body) != source.get("report_sha256"):
        _fail("source receipt hash mismatch")
    if binding.get("source_report_sha256") != source.get("report_sha256"):
        _fail("source report binding mismatch")
    if binding.get("generated_edges_sha256") != source.get("generated_edges_sha256"):
        _fail("source edge hash binding mismatch")
    if binding.get("event_carrier_scope") != source.get("event_carrier_scope") or (
        binding.get("event_carrier_scope")
        != "observer_instrumentation_history_over_source_state_snapshots"
    ):
        _fail("source event-carrier scope mismatch")
    if (
        binding.get("underlying_repair_transactions_promoted_as_events") is not False
        or source.get("underlying_repair_transactions_promoted_as_events") is not False
    ):
        _fail("repair-event promotion boundary mismatch")
    nodes = [str(event["event_key"]) for event in source["semantic_events"]]
    edges = [
        (str(row["parent_event_id"]), str(row["child_event_id"]))
        for row in source["generated_edges"]
    ]
    projection = {"nodes": sorted(set(nodes)), "edges": sorted(set(edges))}
    if binding.get("semantic_poset_sha256") != _sha(projection):
        _fail("semantic poset binding mismatch")
    if binding.get("semantic_event_keys_sha256") != _sha(sorted(set(nodes))):
        _fail("semantic event-key binding mismatch")
    if binding.get("semantic_events_sha256") != _sha(source["semantic_events"]):
        _fail("semantic event-material binding mismatch")
    if binding.get("observer_event_log_sha256") != source.get(
        "observer_event_log_sha256"
    ):
        _fail("observer event-log binding mismatch")
    observer_material = source.get("observer_log_material") or {}
    if source.get("observer_event_log_sha256") != observer_material.get(
        "event_log_sha256"
    ) or source.get("observer_event_log_sha256") != _raw_sha(
        observer_material.get("events")
    ):
        _fail("embedded observer event-log material hash mismatch")
    source_stats = _stats(nodes, edges)
    if receipt.get("source_statistics") != source_stats:
        _fail("source statistics mismatch")
    repair_only = source.get("repair_only_event_carrier_control") or {}
    expected_carrier_controls = {
        "observer_instrumentation_history_classification": (
            "EVALUATED_FINITE_ORDER_NOT_COMPLETE_PHYSICAL_EVENT_CARRIER"
        ),
        "repair_only_control_sha256": _sha(repair_only),
        "repair_only_event_count": repair_only.get("repair_event_count"),
        "repair_only_versioned_provenance_edge_count": repair_only.get(
            "versioned_provenance_edge_count"
        ),
        "repair_only_classification": repair_only.get("classification"),
        "required_physical_event_carrier_change": repair_only.get(
            "required_model_change"
        ),
    }
    if receipt.get("event_carrier_selection_controls") != expected_carrier_controls:
        _fail("event-carrier selection control mismatch")
    if (
        repair_only.get("classification")
        != "REPAIR_ONLY_EVENT_CARRIER_IS_ANTICHAIN"
        or repair_only.get("versioned_provenance_edge_count") != 0
    ):
        _fail("repair-only antichain control mismatch")
    local_diagnostic = _local_domain_diagnostic()
    if receipt.get("existing_local_domain_diagnostic") != local_diagnostic:
        _fail("local-domain diagnostic mismatch")
    comparison = _matched_comparison()
    if receipt.get("single_cutoff_matched_interval_comparison") != comparison:
        _fail("single-cutoff matched-interval comparison mismatch")
    references = _references()
    if receipt.get("reference_controls") != references:
        _fail("reference-control replay mismatch")
    calibration = all(
        row["calibration_within_one_dimension"]
        for row in references["minkowski_alexandrov_sprinklings"].values()
    )
    mutated_projection = {
        "nodes": projection["nodes"],
        "edges": projection["edges"][:-1],
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
    controls = {
        "event_and_edge_order_permutation_invariant": _stats(
            reversed(nodes), reversed(edges)
        )
        == source_stats,
        "duplicate_semantic_id_input_is_set_idempotent": _stats(
            [*nodes, nodes[0]], edges
        )
        == source_stats,
        "source_receipt_hash_chain_replays": bool(
            source["report_sha256"] == _sha(source_body)
            and source["generated_edges_sha256"]
            == _sha(source["generated_edges"])
            and source["observer_event_log_sha256"]
            == observer_material["event_log_sha256"]
            == _raw_sha(observer_material["events"])
            and binding["semantic_event_keys_sha256"]
            == _sha(sorted(set(nodes)))
            and binding["semantic_events_sha256"]
            == _sha(source["semantic_events"])
        ),
        "source_edge_mutation_changes_bound_poset_hash": bool(
            _sha(mutated_projection) != _sha(projection)
            and _sha(source["generated_edges"][:-1])
            != source["generated_edges_sha256"]
            and mutated_source["report_sha256"]
            != binding["source_report_sha256"]
            and mutated_source["generated_edges_sha256"]
            != binding["generated_edges_sha256"]
            and _sha(mutated_projection) != binding["semantic_poset_sha256"]
        ),
        "minkowski_dimension_calibration_within_one_dimension": calibration,
        "synthetic_minkowski_and_de_sitter_nested_inclusions_certified": all(
            row["all_nested_inclusion_couplings_certified"]
            for row in references["nested_poisson_density_controls"][
                "geometries"
            ].values()
        ),
        "flat_myrheim_meyer_excluded_from_flrw_controls": all(
            level["flat_myrheim_meyer_dimension_estimate"] is None
            and level["flat_myrheim_meyer_status"]
            == "NOT_APPLIED_CURVED_FLRW_REFERENCE_CONTROL"
            for run in references["nested_poisson_density_controls"][
                "geometries"
            ]["de_sitter_flat_patch_3_plus_1"]["runs"]
            for level in run["levels"]
        ),
        "matched_interval_heldout_controls_distinguish_manifold_group_from_negatives": bool(
            comparison["heldout_negative_controls_distinguished"]
        ),
        "single_cutoff_result_replays_exact_interval_population": bool(
            comparison["adequate_interval_count"] > 0
            and sum(comparison["nearest_reference_counts"].values())
            == comparison["adequate_interval_count"]
            and 0
            <= comparison["interval_count_in_exploratory_4d_band"]
            <= comparison["adequate_interval_count"]
        ),
        "chain_and_record_forest_controls_distinguished": bool(
            references["nonmanifold_controls"]["total_chain"]["width"] == 1
            and references["nonmanifold_controls"][
                "eight_disconnected_record_chains"
            ]["weak_component_count"]
            == 8
        ),
    }
    if receipt.get("invariance_controls") != controls:
        _fail("invariance controls mismatch")
    if receipt.get("controls_fail_closed") is not all(controls.values()):
        _fail("controls_fail_closed mismatch")
    enough = source_stats["adequate_dimension_interval_count"] >= MIN_SAMPLE
    expected_source_status = (
        "EXPLORATORY_COMPARISON_READY_NOT_CONFIRMED"
        if enough
        else "INCONCLUSIVE__INSUFFICIENT_CERTIFIED_INTERVAL_SIZE"
    )
    if receipt.get("source_control_status") != expected_source_status:
        _fail("source-control status mismatch")
    if receipt.get("status") != comparison["result"]:
        _fail("single-cutoff result status mismatch")
    if receipt.get("CAUSET_DIAGNOSTIC_PIPELINE_REPRODUCTION_RECEIPT") is not all(
        controls.values()
    ):
        _fail("exploratory receipt flag mismatch")
    if receipt.get("OPH_CAUSAL_SET_SIMILARITY_RECEIPT") is not False:
        _fail("OPH causal-set similarity receipt must remain false")
    if receipt.get("CAUSET_MANIFOLDLIKE_RECEIPT") is not False:
        _fail("manifoldlike receipt must remain false")
    if receipt.get("physical_promotion_allowed") is not False:
        _fail("physical promotion must remain false")
    if receipt.get("held_out_confirmation_status") != "NOT_RUN_EXPLORATORY_ONLY":
        _fail("held-out boundary mismatch")
    if receipt.get("refinement_invariance_status") != (
        "NOT_EVALUATED_NO_CERTIFIED_REFINEMENT_MAP_IN_SOURCE_RECEIPT"
    ):
        _fail("OPH refinement boundary mismatch")
    if receipt.get("oph_refinement_family_comparison") != {
        "status": "NOT_EVALUATED_NO_CERTIFIED_OPH_REFINEMENT_MAP_OR_FAMILY",
        "source_control_too_small_for_interval_comparison": not enough,
        "local_single_cutoff_comparison_completed": True,
        "local_single_cutoff_result": comparison["result"],
        "certified_oph_refinement_map_or_family_available": False,
        "similarity_claimed": False,
    }:
        _fail("OPH refinement-family comparison boundary mismatch")
    if receipt.get("flrw_reference_control_status") != (
        "IMPLEMENTED_REPLAYED_DE_SITTER_FLAT_PATCH_SPECIAL_FLRW_REFERENCE_ONLY"
    ):
        _fail("FLRW reference-control status mismatch")
    return {
        "verified": True,
        "status": comparison["result"],
        "source_control_status": expected_source_status,
        "source_event_count": source_stats["event_count"],
        "source_height": source_stats["height"],
        "source_maximum_interval_size": source_stats["maximum_interval_size"],
        "local_control_status": local_diagnostic["status"],
        "local_event_count": local_diagnostic["statistics"]["event_count"],
        "local_adequate_interval_count": local_diagnostic["statistics"][
            "adequate_dimension_interval_count"
        ],
    }


def _summary_line(result: Mapping[str, Any]) -> str:
    return (
        "verified: "
        f"source_status={result['source_control_status']} "
        f"source_n={result['source_event_count']} "
        f"source_height={result['source_height']} "
        f"source_max_interval={result['source_maximum_interval_size']}; "
        f"local_status={result['local_control_status']} "
        f"local_n={result['local_event_count']} "
        f"local_adequate_intervals={result['local_adequate_interval_count']} "
        f"comparison={result['status']}"
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", default=str(DEFAULT_RECEIPT))
    args = parser.parse_args()
    try:
        result = verify_receipt(args.receipt)
    except IndependentCausetVerificationError as error:
        print(f"REFUSED: {error}")
        return 1
    print(_summary_line(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
