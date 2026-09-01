"""Independent verifier for the rank-3/FCC causal-order fixture.

The producer is intentionally not imported.  This verifier rebuilds the FCC
diamonds and explicit propagation/repair parent DAG, but derives each thinned
order a second way from the closed-form FCC causal reachability condition.  It
also independently replays the hash marks, nested restrictions, raw layered
negative, fixed-cardinality Minkowski controls, interval profiles, exact width,
summary comparisons, the additional depth-24 negative, exact cubic/quartic
growth controls, payload binding, and all nonclaim firewalls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = (
    ROOT / "data" / "causal_order" / "rank3_fcc_causet_compatibility_receipt.json"
)
EXPECTED_SCHEMA = "oph.rank3-fcc-causal-order-compatibility.v2"
EXPECTED_ARTIFACT = (
    "CONSTRUCTIVE_RANK3_PLUS_TIME_CAUSAL_ORDER_COMPATIBILITY_FIXTURE"
)
EXPECTED_DEPTHS = (8, 12, 16, 20)
ADDITIONAL_EXTRAPOLATION_DEPTH = 24
FINE_DIAMOND_GROWTH_N = tuple(range(1, 11))
EXPECTED_SEEDS = (901, 902, 903)
EXPECTED_MINKOWSKI_SEEDS = (1901, 1902)
EXPECTED_MARK_DOMAIN = "oph.rank3-fcc-causet-compatibility.mark.v1"
PROFILE_M_MAX = 15
RAW_NEGATIVE_DEPTH = 10
FORMULA_DEPTH = 4
MARK_NUMERATOR = 1
MARK_DENOMINATOR = 50
SENSITIVITY_DENOMINATORS = (25, 40, 50, 60, 75, 100)
SENSITIVITY_SEED = 901
BALL_RADII = tuple(range(21))
POSITIVE_AXES: tuple[tuple[int, int, int], ...] = (
    (1, 1, 0),
    (1, -1, 0),
    (1, 0, 1),
    (1, 0, -1),
    (0, 1, 1),
    (0, 1, -1),
)
DIRECTIONS = POSITIVE_AXES + tuple((-x, -y, -z) for x, y, z in POSITIVE_AXES)
ORIGIN = (0, 0, 0)


class IndependentRank3FCCVerificationError(RuntimeError):
    """Raised when any independent replay clause fails."""


def _fail(message: str) -> None:
    raise IndependentRank3FCCVerificationError(message)


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


def _sequence_sha(rows: Iterable[object]) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"oph.ordered-row-sequence.v1\0")
    count = 0
    for row in rows:
        encoded = _canonical_bytes(row)
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
        count += 1
    hasher.update(count.to_bytes(8, "big"))
    return "sha256:" + hasher.hexdigest()


def _add(
    left: tuple[int, int, int], right: tuple[int, int, int]
) -> tuple[int, int, int]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _sub(
    left: tuple[int, int, int], right: tuple[int, int, int]
) -> tuple[int, int, int]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _identity(
    kind: str,
    phase: int,
    site: tuple[int, int, int],
    axis: int | None = None,
) -> str:
    suffix = "" if axis is None else f":a{axis}"
    return (
        f"{kind}:q{phase:+04d}:x{site[0]:+04d}:y{site[1]:+04d}:z{site[2]:+04d}{suffix}"
    )


def _incident(
    sites: set[tuple[int, int, int]],
) -> set[tuple[tuple[int, int, int], int]]:
    result: set[tuple[tuple[int, int, int], int]] = set()
    for site in sites:
        for axis, direction in enumerate(POSITIVE_AXES):
            result.add((site, axis))
            result.add((_sub(site, direction), axis))
    return result


def _fine_graph(depth: int) -> dict[str, Any]:
    balls = [{ORIGIN}]
    for _ in range(depth):
        prior = balls[-1]
        balls.append(
            prior
            | {_add(site, direction) for site in prior for direction in DIRECTIONS}
        )
    local = [balls[min(time, depth - time)] for time in range(depth + 1)]
    seams = [
        _incident(balls[time]) & _incident(balls[depth - time - 1])
        for time in range(depth)
    ]
    ids: list[str] = []
    desc: list[tuple[int, int, tuple[int, int, int], tuple[int, int, int]]] = []
    keys: list[tuple[object, ...]] = []
    parents: list[tuple[int, ...]] = []
    position: dict[tuple[object, ...], int] = {}
    for time in range(depth + 1):
        phase = 2 * time - depth
        for site in sorted(local[time]):
            position[("L", time, site)] = len(ids)
            ids.append(_identity("L", phase, site))
            desc.append((0, time, site, site))
            keys.append(("L", time, site))
            parents.append(())
        if time < depth:
            for site, axis in sorted(seams[time]):
                endpoint = _add(site, POSITIVE_AXES[axis])
                position[("R", time, site, axis)] = len(ids)
                ids.append(_identity("R", phase + 1, site, axis))
                desc.append((1, time, site, endpoint))
                keys.append(("R", time, site, axis))
                parents.append(())
    for time in range(depth + 1):
        if time:
            for site in local[time]:
                direct: list[int] = []
                for axis, direction in enumerate(POSITIVE_AXES):
                    for seam in ((site, axis), (_sub(site, direction), axis)):
                        if seam in seams[time - 1]:
                            direct.append(position[("R", time - 1, *seam)])
                parents[position[("L", time, site)]] = tuple(sorted(direct))
        if time < depth:
            for site, axis in seams[time]:
                endpoint = _add(site, POSITIVE_AXES[axis])
                parents[position[("R", time, site, axis)]] = tuple(
                    sorted(
                        position[("L", time, endpoint_site)]
                        for endpoint_site in (site, endpoint)
                        if endpoint_site in local[time]
                    )
                )
    local_count = sum(row[0] == 0 for row in desc)
    return {
        "depth": depth,
        "ids": ids,
        "descriptors": desc,
        "keys": keys,
        "parents": parents,
        "identity_set": set(ids),
        "event_count": len(ids),
        "local_count": local_count,
        "seam_count": len(ids) - local_count,
        "edge_count": sum(len(row) for row in parents),
        "carrier_sha": _sequence_sha(ids),
        "edge_sha": _sequence_sha(
            (ids[parent], ids[child])
            for child, row in enumerate(parents)
            for parent in row
        ),
    }


def _ball_growth() -> dict[str, Any]:
    balls = [{ORIGIN}]
    for _ in range(max(BALL_RADII)):
        prior = balls[-1]
        balls.append(
            prior
            | {_add(site, direction) for site in prior for direction in DIRECTIONS}
        )
    rows: list[dict[str, Any]] = []
    exact = True
    for radius in BALL_RADII:
        numerator = 10 * radius**3 + 15 * radius**2 + 11 * radius + 3
        expected = numerator // 3
        measured = len(balls[radius])
        exact &= numerator % 3 == 0 and expected == measured
        rows.append(
            {
                "radius": radius,
                "carrier_ball_count": measured,
                "count_over_radius_cubed": (
                    None if radius == 0 else measured / radius**3
                ),
            }
        )
    positive = rows[1:]
    fitted = float(
        np.polyfit(
            np.log([float(row["radius"]) for row in positive]),
            np.log([float(row["carrier_ball_count"]) for row in positive]),
            1,
        )[0]
    )
    return {
        "rows": rows,
        "exact_ball_polynomial": "(10*r^3 + 15*r^2 + 11*r + 3)/3",
        "exact_polynomial_degree": 3,
        "leading_cubic_coefficient": 10.0 / 3.0,
        "finite_radius_log_log_fit_exponent": fitted,
        "all_rows_match_exact_polynomial": exact,
        "classification": "POLYNOMIAL_RANK_3_CARRIER_BALL_GROWTH",
    }


def _fine_diamond_growth() -> dict[str, Any]:
    seam_rows: list[dict[str, Any]] = []
    all_seams_exact = True
    for radius in range(max(FINE_DIAMOND_GROWTH_N)):
        balls = [{ORIGIN}]
        for _ in range(radius):
            prior = balls[-1]
            balls.append(
                prior
                | {_add(site, direction) for site in prior for direction in DIRECTIONS}
            )
        measured = len(_incident(balls[-1]))
        expected = 20 * radius**3 + 48 * radius**2 + 40 * radius + 12
        all_seams_exact &= measured == expected
        seam_rows.append(
            {
                "radius": radius,
                "oriented_incident_seam_count": measured,
                "expected_oriented_incident_seam_count": expected,
            }
        )

    diamond_rows: list[dict[str, Any]] = []
    all_diamonds_exact = True
    for n in FINE_DIAMOND_GROWTH_N:
        depth = 2 * n
        balls = [{ORIGIN}]
        for _ in range(depth):
            prior = balls[-1]
            balls.append(
                prior
                | {_add(site, direction) for site in prior for direction in DIRECTIONS}
            )
        local_count = sum(
            len(balls[min(time, depth - time)]) for time in range(depth + 1)
        )
        seam_count = sum(
            len(_incident(balls[time]) & _incident(balls[depth - time - 1]))
            for time in range(depth)
        )
        measured = local_count + seam_count
        numerator = 35 * n**4 + 46 * n**3 + 22 * n**2 + 11 * n + 3
        expected = numerator // 3
        exact = numerator % 3 == 0 and measured == expected
        all_diamonds_exact &= exact
        diamond_rows.append(
            {
                "half_depth_n": n,
                "imposed_depth_D": depth,
                "geometric_local_event_count": local_count,
                "geometric_seam_event_count": seam_count,
                "fine_event_count": measured,
                "expected_fine_event_count": expected,
                "exact": exact,
            }
        )
    return {
        "fcc_ball_formula": "B(r)=(10*r^3 + 15*r^2 + 11*r + 3)/3",
        "oriented_incident_seam_formula": "S(r)=20*r^3 + 48*r^2 + 40*r + 12",
        "fine_diamond_decomposition": (
            "N(2*n)=B(n)+2*sum_{r=0}^{n-1}B(r)"
            "+2*sum_{r=0}^{n-1}S(r)"
        ),
        "fine_diamond_formula_checked_on_bounded_grid": (
            "N(2*n)=(35*n^4 + 46*n^3 + 22*n^2 + 11*n + 3)/3"
        ),
        "exact_polynomial_degree_in_imposed_half_depth": 4,
        "checked_half_depth_n": list(FINE_DIAMOND_GROWTH_N),
        "oriented_seam_rows": seam_rows,
        "fine_diamond_rows": diamond_rows,
        "all_oriented_seam_counts_exact": all_seams_exact,
        "all_fine_diamond_counts_exact": all_diamonds_exact,
        "general_all_n_machine_proof_received": False,
        "poset_height_used_as_independent_variable": False,
        "physical_four_volume_claimed": False,
        "temporal_layering_imposed": True,
        "classification": (
            "BOUNDED_N1_TO_N10_QUARTIC_FINE_DIAMOND_COUNT_MATCH_UNDER_IMPOSED_TEMPORAL_LAYERING"
        ),
    }


def _marked(
    identity: str,
    seed: int,
    *,
    numerator: int = MARK_NUMERATOR,
    denominator: int = MARK_DENOMINATOR,
) -> bool:
    digest = hashlib.sha256(
        f"{EXPECTED_MARK_DOMAIN}\0{seed}\0{identity}".encode("ascii")
    ).digest()
    threshold = (2**64 * numerator) // denominator
    return int.from_bytes(digest[:8], "big") < threshold


def _fcc_distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> int:
    delta = (abs(left[0] - right[0]), abs(left[1] - right[1]), abs(left[2] - right[2]))
    return max(max(delta), sum(delta) // 2)


def _before(
    left: tuple[int, int, tuple[int, int, int], tuple[int, int, int]],
    right: tuple[int, int, tuple[int, int, int], tuple[int, int, int]],
) -> bool:
    lk, lt, la, lb = left
    rk, rt, ra, rb = right
    if 2 * rt + rk <= 2 * lt + lk:
        return False
    limit = rt - lt - lk
    if limit < 0:
        return False
    starts = (la,) if lk == 0 else (la, lb)
    ends = (ra,) if rk == 0 else (ra, rb)
    return min(_fcc_distance(a, b) for a in starts for b in ends) <= limit


def _geometric_selected_order(
    graph: Mapping[str, Any],
    seed: int,
    *,
    numerator: int = MARK_NUMERATOR,
    denominator: int = MARK_DENOMINATOR,
) -> tuple[list[str], list[int], list[int]]:
    raw = [
        index
        for index, identity in enumerate(graph["ids"])
        if _marked(
            identity, seed, numerator=numerator, denominator=denominator
        )
    ]
    ids = [graph["ids"][index] for index in raw]
    descriptors = [graph["descriptors"][index] for index in raw]
    ancestors = [0] * len(raw)
    for future, right in enumerate(descriptors):
        bits = 0
        for past, left in enumerate(descriptors[:future]):
            if _before(left, right):
                bits |= 1 << past
        ancestors[future] = bits
    return ids, ancestors, raw


def _provenance_closure(graph: Mapping[str, Any]) -> tuple[list[str], list[int]]:
    ancestors = [0] * int(graph["event_count"])
    for child, direct in enumerate(graph["parents"]):
        bits = 0
        for parent in direct:
            bits |= ancestors[parent] | (1 << parent)
        ancestors[child] = bits
    return list(graph["ids"]), ancestors


def _bits(bits0: int) -> Iterable[int]:
    bits = bits0
    while bits:
        bit = bits & -bits
        bits -= bit
        yield bit.bit_length() - 1


def _descendants(ancestors: Sequence[int]) -> list[int]:
    descendants = [0] * len(ancestors)
    for future, row in enumerate(ancestors):
        for past in _bits(row):
            descendants[past] |= 1 << future
    return descendants


def _width(ancestors: Sequence[int]) -> int:
    count = len(ancestors)
    successors = [[] for _ in range(count)]
    for future, row in enumerate(ancestors):
        for past in _bits(row):
            successors[past].append(future)
    left_match = [-1] * count
    right_match = [-1] * count
    distance = [-1] * count

    def bfs() -> bool:
        queue: deque[int] = deque()
        found = False
        for left in range(count):
            if left_match[left] < 0:
                distance[left] = 0
                queue.append(left)
            else:
                distance[left] = -1
        while queue:
            left = queue.popleft()
            for right in successors[left]:
                partner = right_match[right]
                if partner < 0:
                    found = True
                elif distance[partner] < 0:
                    distance[partner] = distance[left] + 1
                    queue.append(partner)
        return found

    def dfs(left: int) -> bool:
        for right in successors[left]:
            partner = right_match[right]
            if partner < 0 or (
                distance[partner] == distance[left] + 1 and dfs(partner)
            ):
                left_match[left] = right
                right_match[right] = left
                return True
        distance[left] = -1
        return False

    matching = 0
    while bfs():
        for left in range(count):
            if left_match[left] < 0 and dfs(left):
                matching += 1
    return count - matching


def _components(ancestors: Sequence[int]) -> list[int]:
    count = len(ancestors)
    parent = list(range(count))
    size = [1] * count

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a == b:
            return
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]

    for future, row in enumerate(ancestors):
        for past in _bits(row):
            union(past, future)
    return sorted(
        (size[node] for node in range(count) if find(node) == node), reverse=True
    )


def _mm_fraction(dimension: float) -> float:
    return math.exp(
        math.lgamma(dimension + 1.0)
        + math.lgamma(dimension / 2.0)
        - math.log(2.0)
        - math.lgamma(1.5 * dimension)
    )


def _invert_mm(fraction: float) -> float | None:
    if not 0.0 < fraction < 1.0:
        return None
    low, high = 1.01, 20.0
    if not _mm_fraction(high) <= fraction <= _mm_fraction(low):
        return None
    for _ in range(96):
        mid = 0.5 * (low + high)
        if _mm_fraction(mid) > fraction:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _stats(
    ids: Sequence[str], ancestors: Sequence[int], *, width: bool
) -> dict[str, Any]:
    count = len(ids)
    descendants = _descendants(ancestors)
    comparable = sum(row.bit_count() for row in ancestors)
    heights = [1] * count
    histogram = [0] * (PROFILE_M_MAX + 1)
    for future, row in enumerate(ancestors):
        heights[future] = 1 + max((heights[past] for past in _bits(row)), default=0)
        for past in _bits(row):
            interior = (descendants[past] & ancestors[future]).bit_count()
            if interior <= PROFILE_M_MAX:
                histogram[interior] += 1
    if not histogram[0]:
        _fail("recomputed order has no links")
    ordering = 2.0 * comparable / (count * (count - 1))
    height = max(heights)
    components = _components(ancestors)
    return {
        "event_count": count,
        "comparable_pair_count": comparable,
        "ordering_fraction": ordering,
        "myrheim_meyer_dimension_candidate": _invert_mm(ordering),
        "myrheim_meyer_status": "CONSTRUCTIVE_FLAT_DIAMOND_CONTROL_ONLY",
        "height": height,
        "width": _width(ancestors) if width else None,
        "width_status": "EXACT" if width else "NOT_EVALUATED_CONTROL_ONLY",
        "height_over_fourth_root_count": height / count**0.25,
        "count_over_height_fourth_power": count / height**4,
        "weak_component_count": len(components),
        "weak_component_sizes": components,
        "interval_abundance_Nm": histogram,
        "normalized_Nm_over_N0": [value / histogram[0] for value in histogram],
        "carrier_sha256": _sequence_sha(ids),
        "induced_order_sha256": _sequence_sha(
            (ids[past], ids[future])
            for future, row in enumerate(ancestors)
            for past in _bits(row)
        ),
    }


def _relation_set(ids: Sequence[str], ancestors: Sequence[int]) -> set[tuple[str, str]]:
    return {
        (ids[past], ids[future])
        for future, row in enumerate(ancestors)
        for past in _bits(row)
    }


def _spatial_ball(rng: np.random.Generator, limit: float) -> np.ndarray:
    while True:
        direction = rng.normal(size=3)
        norm = float(np.linalg.norm(direction))
        if norm > 0.0:
            break
    radius = limit * float(rng.random()) ** (1.0 / 3.0)
    return direction * (radius / norm)


def _minkowski(cardinality: int, seed: int) -> tuple[list[str], list[int], str]:
    rng = np.random.Generator(np.random.PCG64(seed + 10_000 * cardinality))
    points: list[np.ndarray] = []
    while len(points) < cardinality:
        time = float(rng.uniform(-0.5, 0.5))
        limit = 0.5 - abs(time)
        if float(rng.random()) <= (2.0 * limit) ** 3:
            points.append(np.asarray([time, *_spatial_ball(rng, limit)], dtype=float))
    array = np.asarray(points, dtype=float)
    array = array[np.argsort(array[:, 0], kind="stable")]
    ancestors = [0] * cardinality
    for future in range(1, cardinality):
        dt = array[future, 0] - array[:future, 0]
        dx = array[future, 1:] - array[:future, 1:]
        causal = (dt > 0.0) & (dt * dt >= np.einsum("ij,ij->i", dx, dx))
        bits = 0
        for past in np.flatnonzero(causal):
            bits |= 1 << int(past)
        ancestors[future] = bits
    return (
        [f"minkowski-{index:06d}" for index in range(cardinality)],
        ancestors,
        _sha(array.tolist()),
    )


def _assert_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        _fail(f"{label} mismatch")


def _median(values: Sequence[float | int]) -> float:
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _centroid(rows: Sequence[Mapping[str, Any]], field: str) -> Any:
    values = [row[field] for row in rows]
    if isinstance(values[0], list):
        return [
            sum(float(value[index]) for value in values) / len(values)
            for index in range(len(values[0]))
        ]
    return sum(float(value) for value in values) / len(values)


def _rms(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)) / len(left))


def _resource_check(graph: Mapping[str, Any]) -> dict[str, Any]:
    def rid(prefix: str, time: int, site: tuple[int, int, int], tail: str) -> str:
        return f"{prefix}:t{time}:x{site[0]}:y{site[1]}:z{site[2]}:{tail}"

    reads: list[list[str]] = []
    writes: list[list[str]] = []
    for key in graph["keys"]:
        kind, time, site = str(key[0]), int(key[1]), key[2]
        if kind == "L":
            writes.append(
                [rid("local", time, site, f"port-{p:02d}") for p in range(12)]
            )
            event_reads: list[str] = []
            if time:
                for axis, direction in enumerate(POSITIVE_AXES):
                    event_reads.append(
                        rid("repair-left", time - 1, site, f"axis-{axis}")
                    )
                    event_reads.append(
                        rid(
                            "repair-right",
                            time - 1,
                            _sub(site, direction),
                            f"axis-{axis}",
                        )
                    )
            reads.append(event_reads)
        else:
            axis = int(key[3])
            endpoint = _add(site, POSITIVE_AXES[axis])
            reads.append(
                [
                    rid("local", time, site, f"port-{axis:02d}"),
                    rid("local", time, endpoint, f"port-{axis + 6:02d}"),
                ]
            )
            writes.append(
                [
                    rid("repair-left", time, site, f"axis-{axis}"),
                    rid("repair-right", time, site, f"axis-{axis}"),
                ]
            )
    writer: dict[str, int] = {}
    duplicates = 0
    for event, resources in enumerate(writes):
        for resource in resources:
            duplicates += resource in writer
            writer[resource] = event
    roots = mismatches = edges = 0
    for child, resources in enumerate(reads):
        generated: set[int] = set()
        for resource in resources:
            if resource in writer:
                generated.add(writer[resource])
            else:
                roots += 1
        edges += len(generated)
        mismatches += generated != set(graph["parents"][child])
    return {
        "depth": FORMULA_DEPTH,
        "event_count": graph["event_count"],
        "written_version_resource_count": len(writer),
        "distinguished_boundary_root_read_count": roots,
        "duplicate_writer_count": duplicates,
        "generated_direct_edge_count": edges,
        "declared_optimized_direct_edge_count": graph["edge_count"],
        "parent_set_mismatch_count": mismatches,
        "exact": duplicates == 0 and mismatches == 0,
    }


def verify_receipt(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="ascii"))
    _assert_equal("schema", report.get("schema"), EXPECTED_SCHEMA)
    _assert_equal("artifact", report.get("artifact_type"), EXPECTED_ARTIFACT)
    body = {key: value for key, value in report.items() if key != "payload_sha256"}
    _assert_equal("payload", report.get("payload_sha256"), _sha(body))
    expected_config = {
        "depths": list(EXPECTED_DEPTHS),
        "additional_out_of_family_extrapolation_depth": (
            ADDITIONAL_EXTRAPOLATION_DEPTH
        ),
        "fine_diamond_growth_half_depth_n": list(FINE_DIAMOND_GROWTH_N),
        "thinning_seeds": list(EXPECTED_SEEDS),
        "mark_domain": EXPECTED_MARK_DOMAIN,
        "mark_bits": 64,
        "thinning_probability_exact": {
            "numerator": MARK_NUMERATOR,
            "denominator": MARK_DENOMINATOR,
        },
        "post_hoc_thinning_sensitivity_denominators": list(
            SENSITIVITY_DENOMINATORS
        ),
        "post_hoc_thinning_sensitivity_seed": SENSITIVITY_SEED,
        "profile_m_max": PROFILE_M_MAX,
        "raw_layered_negative_depth": RAW_NEGATIVE_DEPTH,
        "provenance_formula_check_depth": FORMULA_DEPTH,
        "minkowski_control_seeds": list(EXPECTED_MINKOWSKI_SEEDS),
        "carrier_ball_growth_radii": list(BALL_RADII),
        "numpy_version": np.__version__,
        "numpy_rng_algorithms": (
            "numpy.random.Generator(np.random.PCG64); uniform, normal"
        ),
    }
    _assert_equal("complete frozen config", report.get("frozen_config"), expected_config)
    _assert_equal(
        "status",
        report.get("status"),
        "BOUNDED_RANK3_PLUS_TIME_ARCHITECTURE_COMPATIBILITY_FIXTURE_PASSED",
    )
    _assert_equal(
        "epistemic status",
        report.get("epistemic_status"),
        "POST_HOC_EXPLORATORY_CONSTRUCTIVE_CONTROL",
    )
    _assert_equal(
        "held-out status",
        report.get("held_out_confirmation_status"),
        "NOT_RUN_POST_HOC_EXPLORATORY_ONLY",
    )
    if report.get("statistical_significance_claimed") is not False:
        _fail("post-hoc control claims statistical significance")
    _assert_equal(
        "control scope",
        report.get("control_scope"),
        (
            "An imposed FCC rank-three/twelve-port gluing and imposed temporal "
            "layering with source-derived versioned propagation/repair order "
            "and deterministic sparse induced-order restriction. The positive "
            "receipt is only for this bounded architecture fixture; finite "
            "causet-likeness similarity is explicitly not received."
        ),
    )
    _assert_equal(
        "complete gluing definition",
        report.get("gluing_definition"),
        {
            "carrier_lattice": "face_centred_cubic_even_parity_lattice",
            "positive_axis_vectors": [list(row) for row in POSITIVE_AXES],
            "port_direction_count": len(DIRECTIONS),
            "local_rank": 3,
            "direction_polytope": "cuboctahedral",
            "symmetry_group": "O_h_not_A5",
            "global_gluing_imposed": True,
            "temporal_layering_imposed": True,
            "exact_oph_icosahedral_axes_used": False,
            "lorentz_invariance_claimed": False,
        },
    )
    _assert_equal(
        "complete versioned provenance semantics",
        report.get("versioned_provenance_semantics"),
        {
            "local_event": (
                "reads the latest twelve incident port versions and atomically "
                "writes twelve propagated versions"
            ),
            "seam_event": (
                "reads the two same-round endpoint versions and atomically "
                "writes their repaired successors"
            ),
            "seam_value_scope": (
                "versioned exchange/provenance topology only; no endpoint values "
                "or arithmetic mismatch-descent witness is bound"
            ),
            "simultaneous_phase_events_are_unordered": True,
            "causal_order": "transitive_closure_of_read_after_write_edges",
            "thinned_order": (
                "induced order on retained events; reachability through omitted "
                "fine events is preserved"
            ),
            "bounded_diamond_missing_reads": (
                "classified as distinguished boundary-root resources"
            ),
        },
    )
    threshold = (2**64 * MARK_NUMERATOR) // MARK_DENOMINATOR
    _assert_equal(
        "complete count-density calibration",
        report.get("count_density_calibration"),
        {
            "definition": (
                "retain a fine event iff the first 64 bits of its domain-separated "
                "identity hash are below floor(2^64/50)"
            ),
            "configured_target_retained_fraction": MARK_NUMERATOR / MARK_DENOMINATOR,
            "implemented_integer_threshold": threshold,
            "implemented_threshold_fraction": threshold / 2**64,
            "mark_semantics": (
                "deterministic domain-separated pseudorandom hash threshold; "
                "not a source of physical or true randomness"
            ),
            "true_random_process_claimed": False,
            "bernoulli_physical_process_claimed": False,
            "physical_volume_calibration_claimed": False,
            "poisson_physical_process_claimed": False,
        },
    )
    expected_next_controls = [
        "derive rather than impose a locality-preserving global gluing",
        (
            "construct an A5-compatible direction refinement family whose "
            "directions densify and isotropize on S2"
        ),
        "derive the event-thinning or coarse-observable rule from OPH dynamics",
        "certify faithful-embedding and manifoldlikeness criteria on a refinement family",
        "calibrate retained counts to physical four-volume independently",
    ]
    _assert_equal(
        "required next controls", report.get("required_next_controls"), expected_next_controls
    )
    nested = report.get("nested_thinned_family", {})
    _assert_equal(
        "nested family keys",
        sorted(nested),
        sorted(
            (
                "levels_by_seed",
                "level_summaries",
                "fine_diamonds_are_nested",
                "retained_carriers_are_nested",
                "induced_orders_restrict_exactly",
            )
        ),
    )
    for key in (
        "fine_diamonds_are_nested",
        "retained_carriers_are_nested",
        "induced_orders_restrict_exactly",
    ):
        if nested.get(key) is not True:
            _fail(f"nested-family wrapper clause is not true: {key}")
    for key in (
        "CURRENT_RANDOM_FEDERATION_SELECTS_THIS_GLUING_RECEIPT",
        "EXACT_A5_S2_DIRECTION_COMPATIBILITY_RECEIPT",
        "PHYSICAL_CAUSAL_SET_RECEIPT",
        "FINITE_CAUSAL_SET_LIKENESS_SIMILARITY_RECEIPT",
        "FAITHFUL_EMBEDDING_RECEIPT",
        "MANIFOLDLIKENESS_RECEIPT",
        "PHYSICAL_DIMENSION_3_PLUS_1_DERIVATION_RECEIPT",
        "FOURTH_POWER_HEIGHT_SCALING_RECEIPT",
        "MATCHED_FINITE_PROFILE_CONVERGENCE_RECEIPT",
        "PHYSICAL_VOLUME_CALIBRATION_RECEIPT",
        "LORENTZIAN_MANIFOLD_RECEIPT",
        "CONTINUUM_LIMIT_RECEIPT",
        "ARITHMETIC_MISMATCH_DESCENT_RECEIPT",
        "physical_promotion_allowed",
    ):
        if report.get(key) is not False:
            _fail(f"nonclaim firewall promoted: {key}")
    graphs = [_fine_graph(depth) for depth in EXPECTED_DEPTHS]
    if not all(
        a["identity_set"] <= b["identity_set"] for a, b in zip(graphs, graphs[1:])
    ):
        _fail("fine diamonds are not nested")

    receipt_by_seed = report["nested_thinned_family"]["levels_by_seed"]
    recomputed_levels: dict[int, list[dict[str, Any]]] = {}
    all_carriers = all_orders = True
    for seed in EXPECTED_SEEDS:
        runs: list[dict[str, Any]] = []
        carriers: list[set[str]] = []
        relations: list[set[tuple[str, str]]] = []
        for graph in graphs:
            ids, ancestors, raw = _geometric_selected_order(graph, seed)
            statistics = _stats(ids, ancestors, width=True)
            local_count = sum(graph["descriptors"][index][0] == 0 for index in raw)
            run = {
                "depth": graph["depth"],
                "fine_event_count": graph["event_count"],
                "fine_local_event_count": graph["local_count"],
                "fine_seam_event_count": graph["seam_count"],
                "fine_direct_edge_count": graph["edge_count"],
                "fine_carrier_sha256": graph["carrier_sha"],
                "fine_direct_edge_sha256": graph["edge_sha"],
                "retained_local_event_count": local_count,
                "retained_seam_event_count": len(ids) - local_count,
                "realized_retained_fraction": len(ids) / graph["event_count"],
                "statistics": statistics,
            }
            runs.append(run)
            carriers.append(set(ids))
            relations.append(_relation_set(ids, ancestors))
        carrier_checks = [carriers[i] <= carriers[i + 1] for i in range(3)]
        order_checks = [
            relations[i]
            == {
                edge
                for edge in relations[i + 1]
                if edge[0] in carriers[i] and edge[1] in carriers[i]
            }
            for i in range(3)
        ]
        all_carriers &= all(carrier_checks)
        all_orders &= all(order_checks)
        for index, run in enumerate(runs):
            run["nested_from_previous_carrier_inclusion"] = (
                True if index == 0 else carrier_checks[index - 1]
            )
            run["nested_from_previous_induced_order_restriction"] = (
                True if index == 0 else order_checks[index - 1]
            )
        _assert_equal(f"seed {seed} levels", runs, receipt_by_seed[str(seed)])
        recomputed_levels[seed] = runs

    raw_graph = _fine_graph(RAW_NEGATIVE_DEPTH)
    raw_ids, raw_ancestors = _provenance_closure(raw_graph)
    raw_stats = _stats(raw_ids, raw_ancestors, width=False)
    flat_profile = [
        math.gamma(m + 0.5) / (math.gamma(0.5) * math.gamma(m + 1.0))
        for m in range(PROFILE_M_MAX + 1)
    ]
    flat_expected = {
        "normalized_Nm_over_N0": flat_profile,
        "formula": "Gamma(m+1/2)/(Gamma(1/2)*Gamma(m+1))",
        "general_dimension_formula": ("Gamma(m+2/d)/(Gamma(2/d)*Gamma(m+1))"),
        "status": "LARGE_CARDINALITY_FLAT_3_PLUS_1_REFERENCE_PROFILE",
        "finite_cardinality_profile_claimed": False,
        "reference": {
            "authors": "Lisa Glaser and Sumati Surya",
            "title": "Towards a Definition of Locality in a Manifoldlike Causal Set",
            "doi": "10.1103/PhysRevD.88.124026",
            "arxiv": "1309.3403",
        },
    }
    _assert_equal(
        "flat 4d asymptotic reference",
        report["flat_4d_interval_abundance_asymptotic"],
        flat_expected,
    )

    extension_graph = _fine_graph(ADDITIONAL_EXTRAPOLATION_DEPTH)
    extension_runs: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        ids, ancestors, raw = _geometric_selected_order(extension_graph, seed)
        statistics = _stats(ids, ancestors, width=False)
        depth20_statistics = recomputed_levels[seed][-1]["statistics"]
        depth20_rms = _rms(
            depth20_statistics["normalized_Nm_over_N0"], flat_profile
        )
        extension_rms = _rms(statistics["normalized_Nm_over_N0"], flat_profile)
        extension_runs.append(
            {
                "seed": seed,
                "fine_event_count": extension_graph["event_count"],
                "fine_carrier_sha256": extension_graph["carrier_sha"],
                "fine_direct_edge_sha256": extension_graph["edge_sha"],
                "retained_event_count": len(ids),
                "retained_raw_index_sha256": _sequence_sha(raw),
                "statistics": statistics,
                "depth20_profile_rms_to_flat_4d_asymptotic": depth20_rms,
                "depth24_profile_rms_to_flat_4d_asymptotic": extension_rms,
                "depth24_normalized_N1_over_N0": statistics[
                    "normalized_Nm_over_N0"
                ][1],
                "profile_rms_worse_than_depth20": extension_rms > depth20_rms,
            }
        )
    all_extension_worse = all(
        bool(row["profile_rms_worse_than_depth20"]) for row in extension_runs
    )
    extension_expected = {
        "selection_status": (
            "POST_SELECTION_ADVERSARIAL_DEPTH_EXTENSION_NOT_PART_OF_FROZEN_PASS_FAMILY"
        ),
        "independent_seed_holdout": False,
        "depth_held_out_from_positive_family": True,
        "depth": ADDITIONAL_EXTRAPOLATION_DEPTH,
        "same_hash_seeds_as_frozen_family": list(EXPECTED_SEEDS),
        "runs": extension_runs,
        "flat_4d_asymptotic_N1_over_N0": flat_profile[1],
        "all_seed_profile_rms_worse_than_depth20": all_extension_worse,
        "classification": (
            "NEGATIVE__DEPTH24_REVERSES_D8_TO_D20_PROFILE_TREND_FOR_EVERY_SEED"
        ),
        "convergence_evidence_received": False,
    }
    _assert_equal(
        "additional depth extrapolation",
        report["additional_out_of_family_extrapolation_control"],
        extension_expected,
    )

    sensitivity_rows: list[dict[str, Any]] = []
    for denominator in SENSITIVITY_DENOMINATORS:
        ids, ancestors, raw = _geometric_selected_order(
            graphs[-1],
            SENSITIVITY_SEED,
            numerator=1,
            denominator=denominator,
        )
        statistics = _stats(ids, ancestors, width=False)
        sensitivity_rows.append(
            {
                "denominator": denominator,
                "configured_retained_fraction": 1.0 / denominator,
                "retained_raw_index_sha256": _sequence_sha(raw),
                "statistics": statistics,
                "profile_rms_to_flat_4d_asymptotic": _rms(
                    statistics["normalized_Nm_over_N0"], flat_profile
                ),
            }
        )
    sensitivity_r = [
        float(row["statistics"]["ordering_fraction"]) for row in sensitivity_rows
    ]
    sensitivity_mm = [
        float(row["statistics"]["myrheim_meyer_dimension_candidate"])
        for row in sensitivity_rows
    ]
    sensitivity_profile = [
        float(row["profile_rms_to_flat_4d_asymptotic"])
        for row in sensitivity_rows
    ]
    sensitivity_expected = {
        "status": "POST_HOC_NON_PASS_GATE_SENSITIVITY_CONTROL",
        "depth": EXPECTED_DEPTHS[-1],
        "seed": SENSITIVITY_SEED,
        "numerator": 1,
        "denominators": list(SENSITIVITY_DENOMINATORS),
        "rows": sensitivity_rows,
        "ordering_fraction_range": [min(sensitivity_r), max(sensitivity_r)],
        "myrheim_meyer_candidate_range": [min(sensitivity_mm), max(sensitivity_mm)],
        "profile_rms_range": [min(sensitivity_profile), max(sensitivity_profile)],
        "myrheim_meyer_is_algebraic_reexpression_of_ordering_fraction": True,
        "profile_rms_not_constant_on_tested_grid": (
            max(sensitivity_profile) > min(sensitivity_profile)
        ),
        "profile_rms_max_over_min": max(sensitivity_profile)
        / min(sensitivity_profile),
        "robustness_beyond_tested_grid_claimed": False,
        "pass_gate": False,
        "classification": (
            "POST_HOC_GRID_SHOWS_COARSE_ORDER_FRACTION_RANGE_AND_PROFILE_TUNING_SENSITIVITY"
        ),
    }
    _assert_equal(
        "post-hoc thinning sensitivity",
        report["post_hoc_thinning_denominator_sensitivity"],
        sensitivity_expected,
    )

    def rms(a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))

    raw_expected = {
        "depth": RAW_NEGATIVE_DEPTH,
        "statistics": raw_stats,
        "profile_rms_to_flat_4d_asymptotic": rms(
            raw_stats["normalized_Nm_over_N0"], flat_profile
        ),
        "classification": "RAW_LAYERED_FINE_DAG_FAILS_4D_INTERVAL_ABUNDANCE_PROFILE",
    }
    _assert_equal("raw layered negative", report["raw_layered_negative"], raw_expected)

    formula_graph = _fine_graph(FORMULA_DEPTH)
    formula_ids, formula_ancestors = _provenance_closure(formula_graph)
    mismatch = 0
    for future, right in enumerate(formula_graph["descriptors"]):
        for past, left in enumerate(formula_graph["descriptors"][:future]):
            mismatch += bool(formula_ancestors[future] & (1 << past)) != _before(
                left, right
            )
    formula_expected = {
        "depth": FORMULA_DEPTH,
        "event_count": len(formula_ids),
        "pair_count_checked": len(formula_ids) * (len(formula_ids) - 1) // 2,
        "mismatch_count": mismatch,
        "exact": mismatch == 0,
    }
    _assert_equal(
        "formula control", report["provenance_formula_control"], formula_expected
    )
    _assert_equal(
        "resource control",
        report["versioned_resource_provenance_control"],
        _resource_check(formula_graph),
    )
    _assert_equal(
        "carrier ball growth", report["carrier_ball_growth_control"], _ball_growth()
    )
    expected_fine_growth = _fine_diamond_growth()
    bounded_growth_by_depth = {
        int(row["imposed_depth_D"]): int(row["fine_event_count"])
        for row in expected_fine_growth["fine_diamond_rows"]
    }
    expected_fine_growth["frozen_family_counts_match_constructed_fine_dags"] = all(
        int(graph["event_count"]) == bounded_growth_by_depth[int(graph["depth"])]
        for graph in graphs
    )
    _assert_equal(
        "fine diamond growth",
        report["fine_diamond_growth_control"],
        expected_fine_growth,
    )

    control_bundle = report["matched_minkowski_3_plus_1_controls"]
    _assert_equal(
        "Minkowski control ensemble",
        {key: value for key, value in control_bundle.items() if key != "levels"},
        {
            "ensemble": (
                "fixed-cardinality iid-uniform (binomial) sprinkling in a flat "
                "Alexandrov interval; equivalent to Poisson sprinkling conditioned "
                "on total cardinality"
            ),
            "poisson_cardinality_fluctuations_present": False,
            "run_count_per_level": len(EXPECTED_MINKOWSKI_SEEDS),
            "uncertainty_or_significance_estimate_claimed": False,
        },
    )
    controls = control_bundle["levels"]
    matched_expected: list[dict[str, Any]] = []
    for index, control in enumerate(controls):
        level_stats = [
            recomputed_levels[seed][index]["statistics"] for seed in EXPECTED_SEEDS
        ]
        cardinality = int(
            round(
                sum(int(row["event_count"]) for row in level_stats) / len(level_stats)
            )
        )
        runs: list[dict[str, Any]] = []
        for seed in EXPECTED_MINKOWSKI_SEEDS:
            ids, ancestors, coordinates = _minkowski(cardinality, seed)
            runs.append(
                {
                    "seed": seed,
                    "coordinates_sha256": coordinates,
                    "statistics": _stats(ids, ancestors, width=False),
                }
            )
        control_stats = [row["statistics"] for row in runs]
        expected_control = {
            "depth": EXPECTED_DEPTHS[index],
            "cardinality": cardinality,
            "runs": runs,
            "ordering_fraction_centroid": _centroid(control_stats, "ordering_fraction"),
            "height_over_fourth_root_count_centroid": _centroid(
                control_stats, "height_over_fourth_root_count"
            ),
            "normalized_Nm_over_N0_centroid": _centroid(
                control_stats, "normalized_Nm_over_N0"
            ),
        }
        _assert_equal(f"Minkowski control at level {index}", control, expected_control)
        matched_expected.append(expected_control)

    level_summaries: list[dict[str, Any]] = []
    for index, depth in enumerate(EXPECTED_DEPTHS):
        rows = [recomputed_levels[seed][index]["statistics"] for seed in EXPECTED_SEEDS]
        counts = [int(row["event_count"]) for row in rows]
        heights = [int(row["height"]) for row in rows]
        profile = _centroid(rows, "normalized_Nm_over_N0")
        matched_profile = matched_expected[index]["normalized_Nm_over_N0_centroid"]
        level_summaries.append(
            {
                "depth": depth,
                "fine_event_count": graphs[index]["event_count"],
                "retained_event_count_minimum": min(counts),
                "retained_event_count_median": _median(counts),
                "retained_event_count_maximum": max(counts),
                "height_minimum": min(heights),
                "height_median": _median(heights),
                "height_maximum": max(heights),
                "ordering_fraction_minimum": min(
                    float(row["ordering_fraction"]) for row in rows
                ),
                "ordering_fraction_maximum": max(
                    float(row["ordering_fraction"]) for row in rows
                ),
                "myrheim_meyer_candidate_minimum": min(
                    float(row["myrheim_meyer_dimension_candidate"]) for row in rows
                ),
                "myrheim_meyer_candidate_maximum": max(
                    float(row["myrheim_meyer_dimension_candidate"]) for row in rows
                ),
                "height_over_fourth_root_count_minimum": min(
                    float(row["height_over_fourth_root_count"]) for row in rows
                ),
                "height_over_fourth_root_count_maximum": max(
                    float(row["height_over_fourth_root_count"]) for row in rows
                ),
                "normalized_Nm_over_N0_centroid": profile,
                "profile_rms_to_flat_4d_asymptotic": _rms(profile, flat_profile),
                "profile_rms_to_matched_minkowski_centroid": _rms(
                    profile, matched_profile
                ),
            }
        )
    _assert_equal(
        "level summaries",
        report["nested_thinned_family"]["level_summaries"],
        level_summaries,
    )
    scaling = level_summaries[-3:]
    exponent = float(
        np.polyfit(
            np.log([float(row["height_median"]) for row in scaling]),
            np.log([float(row["retained_event_count_median"]) for row in scaling]),
            1,
        )[0]
    )
    expected_scaling = {
        "fit_levels": [int(row["depth"]) for row in scaling],
        "log_count_vs_log_height_exponent": exponent,
        "target_exponent": 4.0,
        "absolute_deviation_from_target": abs(exponent - 4.0),
        "largest_level_height_over_fourth_root_count_range": [
            level_summaries[-1]["height_over_fourth_root_count_minimum"],
            level_summaries[-1]["height_over_fourth_root_count_maximum"],
        ],
        "matched_minkowski_centroid": matched_expected[-1][
            "height_over_fourth_root_count_centroid"
        ],
        "fourth_power_trend_demonstrated": False,
        "normalization_match_claimed": False,
    }
    _assert_equal(
        "height/count scaling", report["height_count_scaling_control"], expected_scaling
    )
    final = level_summaries[-1]
    rms_trend = [
        float(row["profile_rms_to_flat_4d_asymptotic"]) for row in level_summaries
    ]
    matched_rms_trend = [
        float(row["profile_rms_to_matched_minkowski_centroid"])
        for row in level_summaries
    ]
    d20_control_between_run_rms = _rms(
        matched_expected[-1]["runs"][0]["statistics"]["normalized_Nm_over_N0"],
        matched_expected[-1]["runs"][1]["statistics"]["normalized_Nm_over_N0"],
    )
    d20_constructive_to_control_rms = matched_rms_trend[-1]
    d20_distance_ratio = d20_constructive_to_control_rms / d20_control_between_run_rms
    expected_profile_interpretation = {
        "rms_to_large_cardinality_flat_4d_reference_by_depth": rms_trend,
        "rms_to_matched_finite_control_by_depth": matched_rms_trend,
        "depth20_constructive_to_matched_control_centroid_rms": (
            d20_constructive_to_control_rms
        ),
        "depth20_matched_control_between_run_rms": d20_control_between_run_rms,
        "depth20_constructive_to_control_spread_ratio": d20_distance_ratio,
        "depth24_profile_rms_to_flat_4d_reference_by_seed": [
            row["depth24_profile_rms_to_flat_4d_asymptotic"]
            for row in extension_runs
        ],
        "d8_to_d20_monotone_segment_used_as_convergence_evidence": False,
        "finite_profile_similarity_received": False,
        "interpretation": (
            "The additional depth-24 control worsens the asymptotic-profile "
            "RMS for every frozen seed, falsifying use of the depth-8--20 "
            "monotone segment as convergence evidence. At depth 20 the "
            "constructive-to-matched-control distance also greatly exceeds "
            "the two-control spread. No finite causet-likeness similarity, "
            "profile convergence, or manifoldlikeness receipt is issued."
        ),
    }
    _assert_equal(
        "profile comparison interpretation",
        report["profile_comparison_interpretation"],
        expected_profile_interpretation,
    )
    expected_checks = {
        "fine_diamonds_are_nested": True,
        "retained_carriers_are_nested": all_carriers,
        "induced_orders_restrict_exactly": all_orders,
        "provenance_reachability_formula_exact_at_bounded_control": formula_expected[
            "exact"
        ],
        "versioned_resources_regenerate_direct_parents_exactly": report[
            "versioned_resource_provenance_control"
        ]["exact"],
        "carrier_balls_match_exact_cubic_growth": report["carrier_ball_growth_control"][
            "all_rows_match_exact_polynomial"
        ],
        "bounded_n1_to_n10_fine_diamond_counts_match_quartic_in_imposed_depth": (
            report["fine_diamond_growth_control"][
                "all_oriented_seam_counts_exact"
            ]
            and report["fine_diamond_growth_control"][
                "all_fine_diamond_counts_exact"
            ]
            and report["fine_diamond_growth_control"][
                "frozen_family_counts_match_constructed_fine_dags"
            ]
        ),
        "largest_level_ordering_fraction_in_bounded_4d_control_band": (
            float(final["ordering_fraction_minimum"]) >= 0.08
            and float(final["ordering_fraction_maximum"]) <= 0.12
        ),
        "largest_level_myrheim_meyer_candidates_within_half_dimension": (
            float(final["myrheim_meyer_candidate_minimum"]) >= 3.5
            and float(final["myrheim_meyer_candidate_maximum"]) <= 4.5
        ),
    }
    _assert_equal(
        "compatibility checks", report["compatibility_checks"], expected_checks
    )
    _assert_equal(
        "compatibility receipt boolean",
        report["CONSTRUCTIVE_ARCHITECTURE_COMPATIBILITY_RECEIPT"],
        all(expected_checks.values()),
    )
    expected_diagnostics = {
        "all_depth24_seed_profiles_worse_than_depth20": all_extension_worse,
        "depth20_constructive_profile_farther_from_control_centroid_than_control_spread": (
            d20_constructive_to_control_rms > d20_control_between_run_rms
        ),
        "bounded_count_height_log_slope_between_three_and_five": (
            3.0 <= exponent <= 5.0
        ),
        "height_fourth_root_normalization_matches_largest_finite_control": False,
    }
    _assert_equal(
        "exploratory diagnostics",
        report["exploratory_diagnostics_not_pass_gates"],
        expected_diagnostics,
    )

    calibration = report.get("count_density_calibration", {})
    if calibration.get("true_random_process_claimed") is not False:
        _fail("hash marks are misclassified as true randomness")
    if calibration.get("bernoulli_physical_process_claimed") is not False:
        _fail("hash marks are promoted to a physical Bernoulli process")
    resource = report.get("versioned_resource_provenance_control", {})
    if int(resource.get("distinguished_boundary_root_read_count", 0)) <= 0:
        _fail("bounded-diamond boundary roots were not classified")
    if not all(report.get("compatibility_checks", {}).values()):
        _fail("one or more compatibility checks is false")
    for key in (
        "CURRENT_RANDOM_FEDERATION_SELECTS_THIS_GLUING_RECEIPT",
        "EXACT_A5_S2_DIRECTION_COMPATIBILITY_RECEIPT",
        "PHYSICAL_CAUSAL_SET_RECEIPT",
        "FINITE_CAUSAL_SET_LIKENESS_SIMILARITY_RECEIPT",
        "FAITHFUL_EMBEDDING_RECEIPT",
        "MANIFOLDLIKENESS_RECEIPT",
        "PHYSICAL_DIMENSION_3_PLUS_1_DERIVATION_RECEIPT",
        "FOURTH_POWER_HEIGHT_SCALING_RECEIPT",
        "MATCHED_FINITE_PROFILE_CONVERGENCE_RECEIPT",
        "PHYSICAL_VOLUME_CALIBRATION_RECEIPT",
        "LORENTZIAN_MANIFOLD_RECEIPT",
        "CONTINUUM_LIMIT_RECEIPT",
        "ARITHMETIC_MISMATCH_DESCENT_RECEIPT",
        "physical_promotion_allowed",
    ):
        if report.get(key) is not False:
            _fail(f"nonclaim firewall promoted: {key}")
    if not all_carriers or not all_orders:
        _fail("independent nested restriction failed")
    return {
        "verified": True,
        "schema": EXPECTED_SCHEMA,
        "payload_sha256": report["payload_sha256"],
        "independent_order_algorithm": "closed_form_fcc_reachability_not_producer_dag_propagation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    print(json.dumps(verify_receipt(args.receipt), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
