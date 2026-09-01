"""Constructive rank-3/FCC causal-order compatibility fixture.

This module asks a deliberately narrower question than the physical source
capture: can a twelve-port, rank-three local carrier architecture support a
finite order with a bounded subset of the coarse order statistics associated
with a 3+1-dimensional causal diamond?  It imposes a coherent
face-centred-cubic (FCC) gluing, alternates
atomic twelve-port propagation commits with atomic seam-repair commits, and
derives the order from the resulting versioned read-after-write DAG.

The imposed carrier matches the cubic spatial-ball formula at every checked
radius 0--20, while its centred fine-diamond event counts match a quartic in
the imposed temporal half-depth for the checked grid n=1--10.  The fine DAG is
regular and layered, so it is also tested as a
negative.  A sparse deterministic identity-hash/Bernoulli-style restriction
is then applied and the *induced*
order is retained: reachability may pass through omitted fine events.  Shared
marks on centred nested diamonds certify carrier inclusion and induced-order
restriction exactly.  Matched fixed-cardinality (binomial) sprinklings of a
flat 3+1 Minkowski Alexandrov interval are generated as calibration controls.
These are equivalent to Poisson sprinklings only after conditioning on total
cardinality.  An additional depth-24 extrapolation control reverses the
depth-8--20 interval-profile trend for every frozen hash seed, and the finite
depth-20 profile is much farther from its matched control centroid than the
two matched controls are from one another.  Consequently no finite
causet-likeness or profile-convergence receipt is issued.

The FCC directions are the twelve cuboctahedral face diagonals and have O_h,
not A5, symmetry.  This is therefore an imposed rank-3/twelve-port
compatibility construction, not a derivation from the current random
federation or the exact OPH icosahedral/S2 direction data.  It does not prove a
physical causal set, faithful embedding, manifoldlikeness, dimension, volume,
Lorentz invariance, Lorentzian geometry, or a continuum limit.  Both the FCC
gluing and temporal layering are imposed.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "data" / "causal_order" / "rank3_fcc_causet_compatibility_receipt.json"
)
SCHEMA = "oph.rank3-fcc-causal-order-compatibility.v2"
ARTIFACT_TYPE = "CONSTRUCTIVE_RANK3_PLUS_TIME_CAUSAL_ORDER_COMPATIBILITY_FIXTURE"

DEPTHS = (8, 12, 16, 20)
ADDITIONAL_EXTRAPOLATION_DEPTH = 24
FINE_DIAMOND_GROWTH_N = tuple(range(1, 11))
THINNING_SEEDS = (901, 902, 903)
THINNING_NUMERATOR = 1
THINNING_DENOMINATOR = 50
THINNING_SENSITIVITY_DENOMINATORS = (25, 40, 50, 60, 75, 100)
THINNING_SENSITIVITY_SEED = 901
PROFILE_M_MAX = 15
RAW_LAYERED_NEGATIVE_DEPTH = 10
PROVENANCE_FORMULA_CHECK_DEPTH = 4
MINKOWSKI_CONTROL_SEEDS = (1901, 1902)
BALL_GROWTH_RADII = tuple(range(21))

# Six positive FCC axes and their negatives give twelve cuboctahedral
# directions.  Each positive axis labels one oriented seam family; every site
# is incident to the outgoing and incoming member of all six families.
POSITIVE_FCC_AXES: tuple[tuple[int, int, int], ...] = (
    (1, 1, 0),
    (1, -1, 0),
    (1, 0, 1),
    (1, 0, -1),
    (0, 1, 1),
    (0, 1, -1),
)
FCC_DIRECTIONS = POSITIVE_FCC_AXES + tuple(
    (-x, -y, -z) for x, y, z in POSITIVE_FCC_AXES
)
ORIGIN = (0, 0, 0)
MARK_DOMAIN = "oph.rank3-fcc-causet-compatibility.mark.v1"


class Rank3FCCCompatibilityError(RuntimeError):
    """Raised when a constructive-control invariant fails."""


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
    """Hash an ordered row stream without materialising one huge JSON list."""

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
    return tuple(left[index] + right[index] for index in range(3))  # type: ignore[return-value]


def _sub(
    left: tuple[int, int, int], right: tuple[int, int, int]
) -> tuple[int, int, int]:
    return tuple(left[index] - right[index] for index in range(3))  # type: ignore[return-value]


def _event_id(
    kind: str,
    centred_phase: int,
    site: tuple[int, int, int],
    axis: int | None = None,
) -> str:
    suffix = "" if axis is None else f":a{axis}"
    return (
        f"{kind}:q{centred_phase:+04d}:"
        f"x{site[0]:+04d}:y{site[1]:+04d}:z{site[2]:+04d}{suffix}"
    )


def _balls(depth: int) -> list[set[tuple[int, int, int]]]:
    balls = [{ORIGIN}]
    for _ in range(depth):
        prior = balls[-1]
        balls.append(
            prior
            | {_add(site, direction) for site in prior for direction in FCC_DIRECTIONS}
        )
    return balls


def _incident_seams(
    sites: set[tuple[int, int, int]],
) -> set[tuple[tuple[int, int, int], int]]:
    seams: set[tuple[tuple[int, int, int], int]] = set()
    for site in sites:
        for axis, direction in enumerate(POSITIVE_FCC_AXES):
            seams.add((site, axis))
            seams.add((_sub(site, direction), axis))
    return seams


def _fine_dag(depth: int) -> dict[str, Any]:
    """Build one centred causal diamond from versioned propagation/repair.

    A local event reads the latest twelve endpoint resources and atomically
    writes twelve propagated versions.  A seam event reads the two same-round
    endpoint versions and atomically writes their repaired successors.  The
    parent lists below are precisely those read-after-write dependencies,
    restricted to the convex diamond between the two distinguished local
    boundary events.
    """

    if depth <= 0 or depth % 2:
        raise Rank3FCCCompatibilityError("control depths must be positive and even")
    balls = _balls(depth)
    local_layers = [balls[min(time, depth - time)] for time in range(depth + 1)]
    seam_layers = [
        _incident_seams(balls[time]) & _incident_seams(balls[depth - time - 1])
        for time in range(depth)
    ]

    identities: list[str] = []
    keys: list[tuple[object, ...]] = []
    descriptors: list[tuple[int, int, tuple[int, int, int], tuple[int, int, int]]] = []
    parents: list[tuple[int, ...]] = []
    index: dict[tuple[object, ...], int] = {}
    for time in range(depth + 1):
        centred_phase = 2 * time - depth
        for site in sorted(local_layers[time]):
            index[("L", time, site)] = len(identities)
            keys.append(("L", time, site))
            identities.append(_event_id("L", centred_phase, site))
            descriptors.append((0, time, site, site))
            parents.append(())
        if time == depth:
            continue
        centred_phase += 1
        for site, axis in sorted(seam_layers[time]):
            endpoint = _add(site, POSITIVE_FCC_AXES[axis])
            index[("R", time, site, axis)] = len(identities)
            keys.append(("R", time, site, axis))
            identities.append(_event_id("R", centred_phase, site, axis))
            descriptors.append((1, time, site, endpoint))
            parents.append(())

    for time in range(depth + 1):
        if time:
            for site in local_layers[time]:
                direct: list[int] = []
                for axis, direction in enumerate(POSITIVE_FCC_AXES):
                    for seam in ((site, axis), (_sub(site, direction), axis)):
                        if seam in seam_layers[time - 1]:
                            direct.append(index[("R", time - 1, *seam)])
                parents[index[("L", time, site)]] = tuple(sorted(direct))
        if time == depth:
            continue
        for site, axis in seam_layers[time]:
            endpoint = _add(site, POSITIVE_FCC_AXES[axis])
            direct = [
                index[("L", time, candidate)]
                for candidate in (site, endpoint)
                if candidate in local_layers[time]
            ]
            parents[index[("R", time, site, axis)]] = tuple(sorted(direct))

    if any(parent >= child for child, row in enumerate(parents) for parent in row):
        raise Rank3FCCCompatibilityError("fine provenance graph is not topological")
    local_count = sum(1 for row in descriptors if row[0] == 0)
    return {
        "depth": depth,
        "identities": identities,
        "keys": keys,
        "descriptors": descriptors,
        "parents": parents,
        "identity_set": set(identities),
        "event_count": len(identities),
        "local_event_count": local_count,
        "seam_event_count": len(identities) - local_count,
        "direct_edge_count": sum(len(row) for row in parents),
        "carrier_sha256": _sequence_sha(identities),
        "direct_edge_sha256": _sequence_sha(
            (identities[parent], identities[child])
            for child, row in enumerate(parents)
            for parent in row
        ),
    }


def _mark_below_threshold(
    identity: str,
    seed: int,
    *,
    numerator: int = THINNING_NUMERATOR,
    denominator: int = THINNING_DENOMINATOR,
) -> bool:
    digest = hashlib.sha256(
        f"{MARK_DOMAIN}\0{seed}\0{identity}".encode("ascii")
    ).digest()
    mark = int.from_bytes(digest[:8], "big")
    threshold = (2**64 * numerator) // denominator
    return mark < threshold


def _induced_ancestors(
    dag: Mapping[str, Any],
    seed: int,
    *,
    retain_all: bool = False,
    thinning_numerator: int = THINNING_NUMERATOR,
    thinning_denominator: int = THINNING_DENOMINATOR,
) -> tuple[list[str], list[int], list[int]]:
    identities = list(dag["identities"])
    parents = list(dag["parents"])
    retained_raw = [
        index
        for index, identity in enumerate(identities)
        if retain_all
        or _mark_below_threshold(
            identity,
            seed,
            numerator=thinning_numerator,
            denominator=thinning_denominator,
        )
    ]
    selected_position = [-1] * len(identities)
    for position, raw_index in enumerate(retained_raw):
        selected_position[raw_index] = position
    propagated = [0] * len(identities)
    selected_ancestors = [0] * len(retained_raw)
    for raw_index, direct in enumerate(parents):
        bits = 0
        for parent in direct:
            bits |= propagated[parent]
            position = selected_position[parent]
            if position >= 0:
                bits |= 1 << position
        propagated[raw_index] = bits
        position = selected_position[raw_index]
        if position >= 0:
            selected_ancestors[position] = bits
    return (
        [identities[index] for index in retained_raw],
        selected_ancestors,
        retained_raw,
    )


def _descendants(ancestors: Sequence[int]) -> list[int]:
    descendants = [0] * len(ancestors)
    for future, bits0 in enumerate(ancestors):
        bits = bits0
        while bits:
            bit = bits & -bits
            past = bit.bit_length() - 1
            bits -= bit
            descendants[past] |= 1 << future
    return descendants


def _width(ancestors: Sequence[int]) -> int:
    """Exact width from a Hopcroft--Karp maximum matching."""

    count = len(ancestors)
    successors: list[list[int]] = [[] for _ in range(count)]
    for future, bits0 in enumerate(ancestors):
        bits = bits0
        while bits:
            bit = bits & -bits
            past = bit.bit_length() - 1
            bits -= bit
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
        matching += sum(
            1 for left in range(count) if left_match[left] < 0 and dfs(left)
        )
    return count - matching


def _component_sizes(ancestors: Sequence[int]) -> list[int]:
    count = len(ancestors)
    parent = list(range(count))
    size = [1] * count

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left == right:
            return
        if size[left] < size[right]:
            left, right = right, left
        parent[right] = left
        size[left] += size[right]

    for future, bits0 in enumerate(ancestors):
        bits = bits0
        while bits:
            bit = bits & -bits
            bits -= bit
            union(bit.bit_length() - 1, future)
    return sorted(
        (size[node] for node in range(count) if find(node) == node), reverse=True
    )


def myrheim_meyer_fraction(dimension: float) -> float:
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
    if not myrheim_meyer_fraction(high) <= fraction <= myrheim_meyer_fraction(low):
        return None
    for _ in range(96):
        midpoint = (low + high) / 2.0
        if myrheim_meyer_fraction(midpoint) > fraction:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def _statistics(
    identities: Sequence[str],
    ancestors: Sequence[int],
    *,
    compute_width: bool = True,
) -> dict[str, Any]:
    count = len(identities)
    if count < 2:
        raise Rank3FCCCompatibilityError("thinned carrier is too small")
    descendants = _descendants(ancestors)
    comparable = sum(bits.bit_count() for bits in ancestors)
    heights = [1] * count
    for future, bits0 in enumerate(ancestors):
        bits = bits0
        maximum = 0
        while bits:
            bit = bits & -bits
            bits -= bit
            maximum = max(maximum, heights[bit.bit_length() - 1])
        heights[future] = maximum + 1
    histogram = [0] * (PROFILE_M_MAX + 1)
    for future, bits0 in enumerate(ancestors):
        bits = bits0
        while bits:
            bit = bits & -bits
            past = bit.bit_length() - 1
            bits -= bit
            interior = (descendants[past] & ancestors[future]).bit_count()
            if interior <= PROFILE_M_MAX:
                histogram[interior] += 1
    if not histogram[0]:
        raise Rank3FCCCompatibilityError("retained order has no links")
    profile = [value / histogram[0] for value in histogram]
    height = max(heights)
    ordering_fraction = 2.0 * comparable / (count * (count - 1))
    components = _component_sizes(ancestors)
    return {
        "event_count": count,
        "comparable_pair_count": comparable,
        "ordering_fraction": ordering_fraction,
        "myrheim_meyer_dimension_candidate": invert_myrheim_meyer_fraction(
            ordering_fraction
        ),
        "myrheim_meyer_status": "CONSTRUCTIVE_FLAT_DIAMOND_CONTROL_ONLY",
        "height": height,
        "width": _width(ancestors) if compute_width else None,
        "width_status": "EXACT" if compute_width else "NOT_EVALUATED_CONTROL_ONLY",
        "height_over_fourth_root_count": height / count**0.25,
        "count_over_height_fourth_power": count / height**4,
        "weak_component_count": len(components),
        "weak_component_sizes": components,
        "interval_abundance_Nm": histogram,
        "normalized_Nm_over_N0": profile,
        "carrier_sha256": _sequence_sha(identities),
        "induced_order_sha256": _sequence_sha(
            (identities[past], identities[future])
            for future, bits0 in enumerate(ancestors)
            for past in _set_bit_indices(bits0)
        ),
    }


def _set_bit_indices(bits0: int) -> Iterable[int]:
    bits = bits0
    while bits:
        bit = bits & -bits
        bits -= bit
        yield bit.bit_length() - 1


def _relation_set(
    identities: Sequence[str], ancestors: Sequence[int]
) -> set[tuple[str, str]]:
    return {
        (identities[past], identities[future])
        for future, bits in enumerate(ancestors)
        for past in _set_bit_indices(bits)
    }


def _fcc_distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> int:
    delta = [abs(left[index] - right[index]) for index in range(3)]
    return max(max(delta), sum(delta) // 2)


def _descriptor_before(
    left: tuple[int, int, tuple[int, int, int], tuple[int, int, int]],
    right: tuple[int, int, tuple[int, int, int], tuple[int, int, int]],
) -> bool:
    left_kind, left_time, left_a, left_b = left
    right_kind, right_time, right_a, right_b = right
    if 2 * right_time + right_kind <= 2 * left_time + left_kind:
        return False
    step_limit = right_time - left_time - left_kind
    if step_limit < 0:
        return False
    starts = (left_a,) if left_kind == 0 else (left_a, left_b)
    ends = (right_a,) if right_kind == 0 else (right_a, right_b)
    return min(_fcc_distance(start, end) for start in starts for end in ends) <= (
        step_limit
    )


def _provenance_formula_check() -> dict[str, Any]:
    dag = _fine_dag(PROVENANCE_FORMULA_CHECK_DEPTH)
    identities, ancestors, _ = _induced_ancestors(dag, 0, retain_all=True)
    descriptors = list(dag["descriptors"])
    mismatch_count = 0
    for future, right in enumerate(descriptors):
        for past, left in enumerate(descriptors[:future]):
            generated = bool(ancestors[future] & (1 << past))
            if generated != _descriptor_before(left, right):
                mismatch_count += 1
    return {
        "depth": PROVENANCE_FORMULA_CHECK_DEPTH,
        "event_count": len(identities),
        "pair_count_checked": len(identities) * (len(identities) - 1) // 2,
        "mismatch_count": mismatch_count,
        "exact": mismatch_count == 0,
    }


def _resource_id(prefix: str, time: int, site: tuple[int, int, int], tail: str) -> str:
    return f"{prefix}:t{time}:x{site[0]}:y{site[1]}:z{site[2]}:{tail}"


def _versioned_resource_provenance_check() -> dict[str, Any]:
    """Regenerate bounded direct parents from explicit versioned resources."""

    dag = _fine_dag(PROVENANCE_FORMULA_CHECK_DEPTH)
    keys = list(dag["keys"])
    writes: list[list[str]] = []
    reads: list[list[str]] = []
    for key in keys:
        kind = str(key[0])
        time = int(key[1])
        site = key[2]
        if not isinstance(site, tuple):
            raise Rank3FCCCompatibilityError("malformed fine event key")
        if kind == "L":
            writes.append(
                [
                    _resource_id("local", time, site, f"port-{port:02d}")
                    for port in range(12)
                ]
            )
            event_reads: list[str] = []
            if time:
                for axis, direction in enumerate(POSITIVE_FCC_AXES):
                    event_reads.append(
                        _resource_id("repair-left", time - 1, site, f"axis-{axis}")
                    )
                    incoming = _sub(site, direction)
                    event_reads.append(
                        _resource_id("repair-right", time - 1, incoming, f"axis-{axis}")
                    )
            reads.append(event_reads)
            continue
        axis = int(key[3])
        endpoint = _add(site, POSITIVE_FCC_AXES[axis])
        reads.append(
            [
                _resource_id("local", time, site, f"port-{axis:02d}"),
                _resource_id("local", time, endpoint, f"port-{axis + 6:02d}"),
            ]
        )
        writes.append(
            [
                _resource_id("repair-left", time, site, f"axis-{axis}"),
                _resource_id("repair-right", time, site, f"axis-{axis}"),
            ]
        )

    writer_of: dict[str, int] = {}
    duplicate_writer_count = 0
    for event, resources in enumerate(writes):
        for resource in resources:
            if resource in writer_of:
                duplicate_writer_count += 1
            writer_of[resource] = event
    missing_root_reads = 0
    mismatch_count = 0
    generated_edge_count = 0
    for child, resources in enumerate(reads):
        generated: set[int] = set()
        for resource in resources:
            writer = writer_of.get(resource)
            if writer is None:
                missing_root_reads += 1
            else:
                generated.add(writer)
        generated_edge_count += len(generated)
        if generated != set(dag["parents"][child]):
            mismatch_count += 1
    return {
        "depth": PROVENANCE_FORMULA_CHECK_DEPTH,
        "event_count": dag["event_count"],
        "written_version_resource_count": len(writer_of),
        "distinguished_boundary_root_read_count": missing_root_reads,
        "duplicate_writer_count": duplicate_writer_count,
        "generated_direct_edge_count": generated_edge_count,
        "declared_optimized_direct_edge_count": dag["direct_edge_count"],
        "parent_set_mismatch_count": mismatch_count,
        "exact": duplicate_writer_count == 0 and mismatch_count == 0,
    }


def _sample_spatial_ball(rng: np.random.Generator, radius_limit: float) -> np.ndarray:
    while True:
        direction = rng.normal(size=3)
        norm = float(np.linalg.norm(direction))
        if norm > 0.0:
            break
    radius = radius_limit * float(rng.random()) ** (1.0 / 3.0)
    return direction * (radius / norm)


def _sample_minkowski_diamond(
    cardinality: int, seed: int
) -> tuple[list[str], list[int], str]:
    rng = np.random.Generator(np.random.PCG64(seed + 10_000 * int(cardinality)))
    points: list[np.ndarray] = []
    while len(points) < cardinality:
        time = float(rng.uniform(-0.5, 0.5))
        radius_limit = 0.5 - abs(time)
        if float(rng.random()) > (2.0 * radius_limit) ** 3:
            continue
        space = _sample_spatial_ball(rng, radius_limit)
        points.append(np.asarray([time, *space], dtype=float))
    array = np.asarray(points, dtype=float)
    array = array[np.argsort(array[:, 0], kind="stable")]
    ancestors = [0] * cardinality
    for future in range(cardinality):
        if not future:
            continue
        delta_time = array[future, 0] - array[:future, 0]
        delta_space = array[future, 1:] - array[:future, 1:]
        causal = (delta_time > 0.0) & (
            delta_time * delta_time >= np.einsum("ij,ij->i", delta_space, delta_space)
        )
        bits = 0
        for past in np.flatnonzero(causal):
            bits |= 1 << int(past)
        ancestors[future] = bits
    identities = [f"minkowski-{index:06d}" for index in range(cardinality)]
    coordinate_hash = _sha(array.tolist())
    return identities, ancestors, coordinate_hash


def _profile_rms(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(
        sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True))
        / len(left)
    )


def _centroid(rows: Sequence[Mapping[str, Any]], field: str) -> Any:
    values = [row[field] for row in rows]
    if isinstance(values[0], list):
        return [
            sum(float(value[index]) for value in values) / len(values)
            for index in range(len(values[0]))
        ]
    return sum(float(value) for value in values) / len(values)


def _median(values: Sequence[float | int]) -> float:
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


def _raw_layered_negative() -> dict[str, Any]:
    dag = _fine_dag(RAW_LAYERED_NEGATIVE_DEPTH)
    identities, ancestors, _ = _induced_ancestors(dag, 0, retain_all=True)
    statistics = _statistics(identities, ancestors, compute_width=False)
    asymptotic = _flat_four_profile()
    return {
        "depth": RAW_LAYERED_NEGATIVE_DEPTH,
        "statistics": statistics,
        "profile_rms_to_flat_4d_asymptotic": _profile_rms(
            statistics["normalized_Nm_over_N0"], asymptotic
        ),
        "classification": ("RAW_LAYERED_FINE_DAG_FAILS_4D_INTERVAL_ABUNDANCE_PROFILE"),
    }


def _flat_four_profile() -> list[float]:
    return [
        math.gamma(value + 0.5) / (math.gamma(0.5) * math.gamma(value + 1.0))
        for value in range(PROFILE_M_MAX + 1)
    ]


def _carrier_ball_growth_control() -> dict[str, Any]:
    balls = _balls(max(BALL_GROWTH_RADII))
    rows: list[dict[str, Any]] = []
    exact = True
    for radius in BALL_GROWTH_RADII:
        expected_numerator = 10 * radius**3 + 15 * radius**2 + 11 * radius + 3
        expected = expected_numerator // 3
        measured = len(balls[radius])
        exact &= expected_numerator % 3 == 0 and measured == expected
        rows.append(
            {
                "radius": radius,
                "carrier_ball_count": measured,
                "count_over_radius_cubed": (
                    None if radius == 0 else measured / radius**3
                ),
            }
        )
    positive = [row for row in rows if int(row["radius"]) > 0]
    fitted_exponent = float(
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
        "finite_radius_log_log_fit_exponent": fitted_exponent,
        "all_rows_match_exact_polynomial": exact,
        "classification": "POLYNOMIAL_RANK_3_CARRIER_BALL_GROWTH",
    }


def _fine_diamond_growth_control() -> dict[str, Any]:
    """Check the imposed-depth event-count polynomial on n=1,...,10."""

    seam_rows: list[dict[str, Any]] = []
    all_seams_exact = True
    for radius in range(max(FINE_DIAMOND_GROWTH_N)):
        ball = _balls(radius)[radius]
        measured = len(_incident_seams(ball))
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
    for half_depth in FINE_DIAMOND_GROWTH_N:
        depth = 2 * half_depth
        balls = _balls(depth)
        local_count = sum(
            len(balls[min(time, depth - time)]) for time in range(depth + 1)
        )
        seam_count = sum(
            len(
                _incident_seams(balls[time])
                & _incident_seams(balls[depth - time - 1])
            )
            for time in range(depth)
        )
        measured = local_count + seam_count
        numerator = (
            35 * half_depth**4
            + 46 * half_depth**3
            + 22 * half_depth**2
            + 11 * half_depth
            + 3
        )
        expected = numerator // 3
        exact = numerator % 3 == 0 and measured == expected
        all_diamonds_exact &= exact
        diamond_rows.append(
            {
                "half_depth_n": half_depth,
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


@functools.lru_cache(maxsize=1)
def produce_rank3_fcc_causet_compatibility_receipt() -> dict[str, Any]:
    dags = [_fine_dag(depth) for depth in DEPTHS]
    raw_inclusions = all(
        left["identity_set"] <= right["identity_set"]
        for left, right in zip(dags, dags[1:])
    )
    levels_by_seed: dict[int, list[dict[str, Any]]] = {}
    all_nested_carriers = True
    all_induced_orders = True
    for seed in THINNING_SEEDS:
        runs: list[dict[str, Any]] = []
        retained_sets: list[set[str]] = []
        relation_sets: list[set[tuple[str, str]]] = []
        for dag in dags:
            identities, ancestors, raw_indices = _induced_ancestors(dag, seed)
            statistics = _statistics(identities, ancestors)
            retained = set(identities)
            relations = _relation_set(identities, ancestors)
            retained_sets.append(retained)
            relation_sets.append(relations)
            descriptors = list(dag["descriptors"])
            local_count = sum(descriptors[index][0] == 0 for index in raw_indices)
            runs.append(
                {
                    "depth": dag["depth"],
                    "fine_event_count": dag["event_count"],
                    "fine_local_event_count": dag["local_event_count"],
                    "fine_seam_event_count": dag["seam_event_count"],
                    "fine_direct_edge_count": dag["direct_edge_count"],
                    "fine_carrier_sha256": dag["carrier_sha256"],
                    "fine_direct_edge_sha256": dag["direct_edge_sha256"],
                    "retained_local_event_count": local_count,
                    "retained_seam_event_count": len(identities) - local_count,
                    "realized_retained_fraction": len(identities) / dag["event_count"],
                    "statistics": statistics,
                }
            )
        carrier_checks = [
            retained_sets[index] <= retained_sets[index + 1]
            for index in range(len(retained_sets) - 1)
        ]
        order_checks = [
            relation_sets[index]
            == {
                relation
                for relation in relation_sets[index + 1]
                if relation[0] in retained_sets[index]
                and relation[1] in retained_sets[index]
            }
            for index in range(len(relation_sets) - 1)
        ]
        all_nested_carriers &= all(carrier_checks)
        all_induced_orders &= all(order_checks)
        for index, run in enumerate(runs):
            run["nested_from_previous_carrier_inclusion"] = (
                True if index == 0 else carrier_checks[index - 1]
            )
            run["nested_from_previous_induced_order_restriction"] = (
                True if index == 0 else order_checks[index - 1]
            )
        levels_by_seed[seed] = runs

    level_summaries: list[dict[str, Any]] = []
    matched_controls: list[dict[str, Any]] = []
    flat_profile = _flat_four_profile()
    for level_index, depth in enumerate(DEPTHS):
        rows = [
            levels_by_seed[seed][level_index]["statistics"] for seed in THINNING_SEEDS
        ]
        event_counts = [int(row["event_count"]) for row in rows]
        heights = [int(row["height"]) for row in rows]
        profile_centroid = _centroid(rows, "normalized_Nm_over_N0")
        matched_cardinality = int(round(sum(event_counts) / len(event_counts)))
        control_rows: list[dict[str, Any]] = []
        for seed in MINKOWSKI_CONTROL_SEEDS:
            identities, ancestors, coordinates_sha256 = _sample_minkowski_diamond(
                matched_cardinality, seed
            )
            control_rows.append(
                {
                    "seed": seed,
                    "coordinates_sha256": coordinates_sha256,
                    "statistics": _statistics(
                        identities, ancestors, compute_width=False
                    ),
                }
            )
        control_statistics = [row["statistics"] for row in control_rows]
        matched_profile_centroid = _centroid(
            control_statistics, "normalized_Nm_over_N0"
        )
        matched_controls.append(
            {
                "depth": depth,
                "cardinality": matched_cardinality,
                "runs": control_rows,
                "ordering_fraction_centroid": _centroid(
                    control_statistics, "ordering_fraction"
                ),
                "height_over_fourth_root_count_centroid": _centroid(
                    control_statistics, "height_over_fourth_root_count"
                ),
                "normalized_Nm_over_N0_centroid": matched_profile_centroid,
            }
        )
        level_summaries.append(
            {
                "depth": depth,
                "fine_event_count": dags[level_index]["event_count"],
                "retained_event_count_minimum": min(event_counts),
                "retained_event_count_median": _median(event_counts),
                "retained_event_count_maximum": max(event_counts),
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
                "normalized_Nm_over_N0_centroid": profile_centroid,
                "profile_rms_to_flat_4d_asymptotic": _profile_rms(
                    profile_centroid, flat_profile
                ),
                "profile_rms_to_matched_minkowski_centroid": _profile_rms(
                    profile_centroid, matched_profile_centroid
                ),
            }
        )

    extension_dag = _fine_dag(ADDITIONAL_EXTRAPOLATION_DEPTH)
    extension_runs: list[dict[str, Any]] = []
    for seed in THINNING_SEEDS:
        identities, ancestors, raw_indices = _induced_ancestors(extension_dag, seed)
        statistics = _statistics(identities, ancestors, compute_width=False)
        depth20_statistics = levels_by_seed[seed][-1]["statistics"]
        depth20_rms = _profile_rms(
            depth20_statistics["normalized_Nm_over_N0"], flat_profile
        )
        extension_rms = _profile_rms(
            statistics["normalized_Nm_over_N0"], flat_profile
        )
        extension_runs.append(
            {
                "seed": seed,
                "fine_event_count": extension_dag["event_count"],
                "fine_carrier_sha256": extension_dag["carrier_sha256"],
                "fine_direct_edge_sha256": extension_dag["direct_edge_sha256"],
                "retained_event_count": len(identities),
                "retained_raw_index_sha256": _sequence_sha(raw_indices),
                "statistics": statistics,
                "depth20_profile_rms_to_flat_4d_asymptotic": depth20_rms,
                "depth24_profile_rms_to_flat_4d_asymptotic": extension_rms,
                "depth24_normalized_N1_over_N0": statistics[
                    "normalized_Nm_over_N0"
                ][1],
                "profile_rms_worse_than_depth20": extension_rms > depth20_rms,
            }
        )
    all_extension_rms_worse = all(
        bool(run["profile_rms_worse_than_depth20"]) for run in extension_runs
    )

    thinning_sensitivity_rows: list[dict[str, Any]] = []
    for denominator in THINNING_SENSITIVITY_DENOMINATORS:
        identities, ancestors, raw_indices = _induced_ancestors(
            dags[-1],
            THINNING_SENSITIVITY_SEED,
            thinning_numerator=1,
            thinning_denominator=denominator,
        )
        statistics = _statistics(identities, ancestors, compute_width=False)
        thinning_sensitivity_rows.append(
            {
                "denominator": denominator,
                "configured_retained_fraction": 1.0 / denominator,
                "retained_raw_index_sha256": _sequence_sha(raw_indices),
                "statistics": statistics,
                "profile_rms_to_flat_4d_asymptotic": _profile_rms(
                    statistics["normalized_Nm_over_N0"], flat_profile
                ),
            }
        )
    sensitivity_ordering_fractions = [
        float(row["statistics"]["ordering_fraction"])
        for row in thinning_sensitivity_rows
    ]
    sensitivity_mm_candidates = [
        float(row["statistics"]["myrheim_meyer_dimension_candidate"])
        for row in thinning_sensitivity_rows
    ]
    sensitivity_profile_rms = [
        float(row["profile_rms_to_flat_4d_asymptotic"])
        for row in thinning_sensitivity_rows
    ]

    raw_negative = _raw_layered_negative()
    formula_control = _provenance_formula_check()
    resource_control = _versioned_resource_provenance_check()
    ball_growth_control = _carrier_ball_growth_control()
    fine_diamond_growth_control = _fine_diamond_growth_control()
    bounded_growth_by_depth = {
        int(row["imposed_depth_D"]): int(row["fine_event_count"])
        for row in fine_diamond_growth_control["fine_diamond_rows"]
    }
    fine_diamond_growth_control[
        "frozen_family_counts_match_constructed_fine_dags"
    ] = all(
        int(dag["event_count"]) == bounded_growth_by_depth[int(dag["depth"])]
        for dag in dags
    )
    final = level_summaries[-1]
    scaling_rows = level_summaries[-3:]
    height_count_scaling_exponent = float(
        np.polyfit(
            np.log([float(row["height_median"]) for row in scaling_rows]),
            np.log([float(row["retained_event_count_median"]) for row in scaling_rows]),
            1,
        )[0]
    )
    rms_trend = [
        float(row["profile_rms_to_flat_4d_asymptotic"]) for row in level_summaries
    ]
    matched_rms_trend = [
        float(row["profile_rms_to_matched_minkowski_centroid"])
        for row in level_summaries
    ]
    depth20_matched_control_run_profile_rms = _profile_rms(
        matched_controls[-1]["runs"][0]["statistics"]["normalized_Nm_over_N0"],
        matched_controls[-1]["runs"][1]["statistics"]["normalized_Nm_over_N0"],
    )
    depth20_constructive_to_matched_rms = matched_rms_trend[-1]
    depth20_profile_distance_ratio = (
        depth20_constructive_to_matched_rms
        / depth20_matched_control_run_profile_rms
    )
    compatibility_checks = {
        "fine_diamonds_are_nested": raw_inclusions,
        "retained_carriers_are_nested": all_nested_carriers,
        "induced_orders_restrict_exactly": all_induced_orders,
        "provenance_reachability_formula_exact_at_bounded_control": (
            formula_control["exact"]
        ),
        "versioned_resources_regenerate_direct_parents_exactly": resource_control[
            "exact"
        ],
        "carrier_balls_match_exact_cubic_growth": ball_growth_control[
            "all_rows_match_exact_polynomial"
        ],
        "bounded_n1_to_n10_fine_diamond_counts_match_quartic_in_imposed_depth": (
            fine_diamond_growth_control["all_oriented_seam_counts_exact"]
            and fine_diamond_growth_control["all_fine_diamond_counts_exact"]
            and fine_diamond_growth_control[
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
    exploratory_diagnostics = {
        "all_depth24_seed_profiles_worse_than_depth20": all_extension_rms_worse,
        "depth20_constructive_profile_farther_from_control_centroid_than_control_spread": (
            depth20_constructive_to_matched_rms
            > depth20_matched_control_run_profile_rms
        ),
        "bounded_count_height_log_slope_between_three_and_five": (
            3.0 <= height_count_scaling_exponent <= 5.0
        ),
        "height_fourth_root_normalization_matches_largest_finite_control": False,
    }
    compatibility_passed = all(compatibility_checks.values())
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "artifact_type": ARTIFACT_TYPE,
        "status": (
            "BOUNDED_RANK3_PLUS_TIME_ARCHITECTURE_COMPATIBILITY_FIXTURE_PASSED"
            if compatibility_passed
            else "CONSTRUCTIVE_COMPATIBILITY_CONTROL_FAILED"
        ),
        "epistemic_status": "POST_HOC_EXPLORATORY_CONSTRUCTIVE_CONTROL",
        "held_out_confirmation_status": "NOT_RUN_POST_HOC_EXPLORATORY_ONLY",
        "statistical_significance_claimed": False,
        "control_scope": (
            "An imposed FCC rank-three/twelve-port gluing and imposed temporal "
            "layering with source-derived versioned propagation/repair order "
            "and deterministic sparse induced-order restriction. The positive "
            "receipt is only for this bounded architecture fixture; finite "
            "causet-likeness similarity is explicitly not received."
        ),
        "frozen_config": {
            "depths": list(DEPTHS),
            "additional_out_of_family_extrapolation_depth": (
                ADDITIONAL_EXTRAPOLATION_DEPTH
            ),
            "fine_diamond_growth_half_depth_n": list(FINE_DIAMOND_GROWTH_N),
            "thinning_seeds": list(THINNING_SEEDS),
            "mark_domain": MARK_DOMAIN,
            "mark_bits": 64,
            "thinning_probability_exact": {
                "numerator": THINNING_NUMERATOR,
                "denominator": THINNING_DENOMINATOR,
            },
            "post_hoc_thinning_sensitivity_denominators": list(
                THINNING_SENSITIVITY_DENOMINATORS
            ),
            "post_hoc_thinning_sensitivity_seed": THINNING_SENSITIVITY_SEED,
            "profile_m_max": PROFILE_M_MAX,
            "raw_layered_negative_depth": RAW_LAYERED_NEGATIVE_DEPTH,
            "provenance_formula_check_depth": PROVENANCE_FORMULA_CHECK_DEPTH,
            "minkowski_control_seeds": list(MINKOWSKI_CONTROL_SEEDS),
            "carrier_ball_growth_radii": list(BALL_GROWTH_RADII),
            "numpy_version": np.__version__,
            "numpy_rng_algorithms": (
                "numpy.random.Generator(np.random.PCG64); uniform, normal"
            ),
        },
        "gluing_definition": {
            "carrier_lattice": "face_centred_cubic_even_parity_lattice",
            "positive_axis_vectors": [list(row) for row in POSITIVE_FCC_AXES],
            "port_direction_count": len(FCC_DIRECTIONS),
            "local_rank": 3,
            "direction_polytope": "cuboctahedral",
            "symmetry_group": "O_h_not_A5",
            "global_gluing_imposed": True,
            "temporal_layering_imposed": True,
            "exact_oph_icosahedral_axes_used": False,
            "lorentz_invariance_claimed": False,
        },
        "versioned_provenance_semantics": {
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
        "count_density_calibration": {
            "definition": (
                "retain a fine event iff the first 64 bits of its domain-separated "
                "identity hash are below floor(2^64/50)"
            ),
            "configured_target_retained_fraction": THINNING_NUMERATOR
            / THINNING_DENOMINATOR,
            "implemented_integer_threshold": (
                (2**64 * THINNING_NUMERATOR) // THINNING_DENOMINATOR
            ),
            "implemented_threshold_fraction": (
                ((2**64 * THINNING_NUMERATOR) // THINNING_DENOMINATOR) / 2**64
            ),
            "mark_semantics": (
                "deterministic domain-separated pseudorandom hash threshold; "
                "not a source of physical or true randomness"
            ),
            "true_random_process_claimed": False,
            "bernoulli_physical_process_claimed": False,
            "physical_volume_calibration_claimed": False,
            "poisson_physical_process_claimed": False,
        },
        "post_hoc_thinning_denominator_sensitivity": {
            "status": "POST_HOC_NON_PASS_GATE_SENSITIVITY_CONTROL",
            "depth": DEPTHS[-1],
            "seed": THINNING_SENSITIVITY_SEED,
            "numerator": 1,
            "denominators": list(THINNING_SENSITIVITY_DENOMINATORS),
            "rows": thinning_sensitivity_rows,
            "ordering_fraction_range": [
                min(sensitivity_ordering_fractions),
                max(sensitivity_ordering_fractions),
            ],
            "myrheim_meyer_candidate_range": [
                min(sensitivity_mm_candidates),
                max(sensitivity_mm_candidates),
            ],
            "profile_rms_range": [
                min(sensitivity_profile_rms),
                max(sensitivity_profile_rms),
            ],
            "myrheim_meyer_is_algebraic_reexpression_of_ordering_fraction": True,
            "profile_rms_not_constant_on_tested_grid": (
                max(sensitivity_profile_rms) > min(sensitivity_profile_rms)
            ),
            "profile_rms_max_over_min": (
                max(sensitivity_profile_rms) / min(sensitivity_profile_rms)
            ),
            "robustness_beyond_tested_grid_claimed": False,
            "pass_gate": False,
            "classification": (
                "POST_HOC_GRID_SHOWS_COARSE_ORDER_FRACTION_RANGE_AND_PROFILE_TUNING_SENSITIVITY"
            ),
        },
        "carrier_ball_growth_control": ball_growth_control,
        "fine_diamond_growth_control": fine_diamond_growth_control,
        "provenance_formula_control": formula_control,
        "versioned_resource_provenance_control": resource_control,
        "raw_layered_negative": raw_negative,
        "nested_thinned_family": {
            "levels_by_seed": {
                str(seed): levels_by_seed[seed] for seed in THINNING_SEEDS
            },
            "level_summaries": level_summaries,
            "fine_diamonds_are_nested": raw_inclusions,
            "retained_carriers_are_nested": all_nested_carriers,
            "induced_orders_restrict_exactly": all_induced_orders,
        },
        "additional_out_of_family_extrapolation_control": {
            "selection_status": (
                "POST_SELECTION_ADVERSARIAL_DEPTH_EXTENSION_NOT_PART_OF_FROZEN_PASS_FAMILY"
            ),
            "independent_seed_holdout": False,
            "depth_held_out_from_positive_family": True,
            "depth": ADDITIONAL_EXTRAPOLATION_DEPTH,
            "same_hash_seeds_as_frozen_family": list(THINNING_SEEDS),
            "runs": extension_runs,
            "flat_4d_asymptotic_N1_over_N0": flat_profile[1],
            "all_seed_profile_rms_worse_than_depth20": all_extension_rms_worse,
            "classification": (
                "NEGATIVE__DEPTH24_REVERSES_D8_TO_D20_PROFILE_TREND_FOR_EVERY_SEED"
            ),
            "convergence_evidence_received": False,
        },
        "flat_4d_interval_abundance_asymptotic": {
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
        },
        "matched_minkowski_3_plus_1_controls": {
            "ensemble": (
                "fixed-cardinality iid-uniform (binomial) sprinkling in a flat "
                "Alexandrov interval; equivalent to Poisson sprinkling conditioned "
                "on total cardinality"
            ),
            "poisson_cardinality_fluctuations_present": False,
            "run_count_per_level": len(MINKOWSKI_CONTROL_SEEDS),
            "uncertainty_or_significance_estimate_claimed": False,
            "levels": matched_controls,
        },
        "height_count_scaling_control": {
            "fit_levels": [int(row["depth"]) for row in scaling_rows],
            "log_count_vs_log_height_exponent": height_count_scaling_exponent,
            "target_exponent": 4.0,
            "absolute_deviation_from_target": abs(height_count_scaling_exponent - 4.0),
            "largest_level_height_over_fourth_root_count_range": [
                final["height_over_fourth_root_count_minimum"],
                final["height_over_fourth_root_count_maximum"],
            ],
            "matched_minkowski_centroid": matched_controls[-1][
                "height_over_fourth_root_count_centroid"
            ],
            "fourth_power_trend_demonstrated": False,
            "normalization_match_claimed": False,
        },
        "profile_comparison_interpretation": {
            "rms_to_large_cardinality_flat_4d_reference_by_depth": rms_trend,
            "rms_to_matched_finite_control_by_depth": matched_rms_trend,
            "depth20_constructive_to_matched_control_centroid_rms": (
                depth20_constructive_to_matched_rms
            ),
            "depth20_matched_control_between_run_rms": (
                depth20_matched_control_run_profile_rms
            ),
            "depth20_constructive_to_control_spread_ratio": (
                depth20_profile_distance_ratio
            ),
            "depth24_profile_rms_to_flat_4d_reference_by_seed": [
                run["depth24_profile_rms_to_flat_4d_asymptotic"]
                for run in extension_runs
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
        },
        "compatibility_checks": compatibility_checks,
        "exploratory_diagnostics_not_pass_gates": exploratory_diagnostics,
        "CONSTRUCTIVE_ARCHITECTURE_COMPATIBILITY_RECEIPT": compatibility_passed,
        "CURRENT_RANDOM_FEDERATION_SELECTS_THIS_GLUING_RECEIPT": False,
        "EXACT_A5_S2_DIRECTION_COMPATIBILITY_RECEIPT": False,
        "PHYSICAL_CAUSAL_SET_RECEIPT": False,
        "FINITE_CAUSAL_SET_LIKENESS_SIMILARITY_RECEIPT": False,
        "FAITHFUL_EMBEDDING_RECEIPT": False,
        "MANIFOLDLIKENESS_RECEIPT": False,
        "PHYSICAL_DIMENSION_3_PLUS_1_DERIVATION_RECEIPT": False,
        "FOURTH_POWER_HEIGHT_SCALING_RECEIPT": False,
        "MATCHED_FINITE_PROFILE_CONVERGENCE_RECEIPT": False,
        "PHYSICAL_VOLUME_CALIBRATION_RECEIPT": False,
        "LORENTZIAN_MANIFOLD_RECEIPT": False,
        "CONTINUUM_LIMIT_RECEIPT": False,
        "ARITHMETIC_MISMATCH_DESCENT_RECEIPT": False,
        "physical_promotion_allowed": False,
        "required_next_controls": [
            "derive rather than impose a locality-preserving global gluing",
            (
                "construct an A5-compatible direction refinement family whose "
                "directions densify and isotropize on S2"
            ),
            "derive the event-thinning or coarse-observable rule from OPH dynamics",
            "certify faithful-embedding and manifoldlikeness criteria on a refinement family",
            "calibrate retained counts to physical four-volume independently",
        ],
    }
    report["payload_sha256"] = _sha(report)
    return report


def write_receipt(path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    report = produce_rank3_fcc_causet_compatibility_receipt()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = write_receipt(arguments.output)
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
