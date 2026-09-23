"""Independent replay of the source-net causal-limit receipt.

The producer is not imported.  The golden orbit, the site graphs, the graph
distances and the interval statistics of the three families at q = 5, 8, 13
are rebuilt here in the sqrt(5) basis with dense all-pairs metric decisions,
scipy shortest paths and explicit event enumeration.  The RER structural
fields (neighbour digest, edge counts, probes, centre intervals, source
records) are recomputed and compared with the values embedded in the receipt's
cross-check blocks.  Levels above q = 13 are checked for internal consistency:
counts against shell counts, exact ordering fractions against pair counts,
stratified estimates against their strata rows, seeds, bounds, clocks and
continuum references.  File pins and the schema are checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
from scipy import integrate
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

ROOT = Path(__file__).resolve().parents[1]
RER_ROOT = Path(os.environ.get("OPH_RER_ROOT", str(ROOT.parent / "reverse-engineering-reality")))
OUTPUT = ROOT / "data/exact/source_net_causal_limit_receipt.json"
SCHEMA = "oph.exact.source-net-causal-limit.v1"
REBUILT_LEVELS = (5, 6, 7)
LEVELS = (5, 6, 7, 8, 9, 10, 11)
DIMENSIONS = (3, 2, 1)
SAMPLE_SEED = 20260909
SAMPLE_MINIMUM = 2000
LOCAL_PINS = (
    "oph_exact/source_net.py",
    "oph_exact/verify_source_net_independent.py",
    "tests/test_exact_source_net.py",
)
RER_PINS = (
    "code/causal_refinement/source_net_causet.py",
    "code/causal_refinement/source_net_causet_receipt.json",
    "paper/tex_fragments/SOURCE_NET_CAUSAL_LIMIT.tex",
    "Lean/Geometry/SourceNetCausalCone.lean",
)
RER_RECEIPT_SHA256 = "c0f790ad383039186371b7a8020f190371fa533ad634010f01f9c271f1545681"
EXPECTED_SCOPE = {
    "population_supplied": True,
    "read_law_supplied": True,
    "tick_supplied": True,
    "one_event_per_site_and_layer_supplied": True,
    "exact_gram_metric_edge_decisions": True,
    "exact_graph_distance_order": True,
    "native_repair_selected": False,
    "physical_clock_or_spacetime_identified": False,
    "finite_runs_demonstrate_asymptotic_limit": False,
    "poisson_sprinkling": False,
    "dimension_statistic_used_as_acceptance": False,
    "read_write_traces_executed": False,
}
REFERENCE_FRACTION = {1: Fraction(1, 2), 2: Fraction(8, 35), 3: Fraction(1, 10)}
UNIT_BALL = {1: 2.0, 2: math.pi, 3: 4.0 * math.pi / 3.0}
FLOAT_TOLERANCE = 1e-9


class SourceNetVerificationError(ValueError):
    pass


def require(condition, label: str) -> None:
    if not condition:
        raise SourceNetVerificationError(label)


def close(x, y, label: str, tolerance: float = FLOAT_TOLERANCE) -> None:
    require(isinstance(x, (int, float)) and isinstance(y, (int, float)), label + ": numeric")
    require(abs(float(x) - float(y)) <= tolerance * max(1.0, abs(float(y))), f"{label}: {x} vs {y}")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key " + key)
        result[key] = value
    return result


def _forbidden(token):
    raise SourceNetVerificationError("non-finite JSON token " + token)


def packed(x) -> bytes:
    return (json.dumps(x, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def hashed(x) -> str:
    return hashlib.sha256(packed(x)).hexdigest()


def load(path: Path = OUTPUT) -> dict:
    data = Path(path).read_bytes()
    require(len(data) <= 8_000_000, "receipt size limit")
    receipt = json.loads(data.decode("ascii"), object_pairs_hook=_pairs, parse_constant=_forbidden)
    require(packed(receipt) == data, "receipt bytes are not canonical JSON")
    return receipt


# --------------------------------------------------------------------------
# sqrt(5) basis: (a, b) means a + b*sqrt(5)
# --------------------------------------------------------------------------


def sgn(x) -> int:
    a, b = x
    if b == 0:
        return (a > 0) - (a < 0)
    if a == 0:
        return (b > 0) - (b < 0)
    if a > 0 and b > 0:
        return 1
    if a < 0 and b < 0:
        return -1
    d = a * a - 5 * b * b
    return ((d > 0) - (d < 0)) * (1 if a > 0 else -1)


def sgn_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    d = a * a - 5 * b * b
    both = np.where((a > 0) & (b > 0), 1, np.where((a < 0) & (b < 0), -1, 0))
    mixed = np.sign(d) * np.where(a > 0, 1, -1)
    return np.where(b == 0, np.sign(a), np.where(a == 0, np.sign(b), np.where(both != 0, both, mixed)))


def to_phi_strings(x) -> list[str]:
    """(a, b) in the sqrt(5) basis as RER's [m, n] strings for m + n*phi."""
    a, b = x
    return [str(Fraction(a) - Fraction(b)), str(Fraction(2) * Fraction(b))]


def golden_floors(q: int) -> list[int]:
    floors = []
    for b in range(q):
        f = 0
        while sgn((b - 2 * (f + 1), b)) >= 0:
            f += 1
        floors.append(f)
    return floors


def fib(n: int) -> tuple[int, int]:
    require(type(n) is int and n >= 3, "level index")
    seq = [0, 1]
    for _ in range(n):
        seq.append(seq[-1] + seq[-2])
    return seq[n], seq[n + 1]


def orbit_twice(q: int) -> list[tuple[int, int]]:
    """2*xi_b = (b - 2 floor(b phi)) + b sqrt(5)."""
    return [(b - 2 * f, b) for b, f in enumerate(golden_floors(q))]


def family_geometry(q: int, dim: int):
    """Sites, dense squared metric (4|s-t|^2 in the sqrt(5) basis), neighbour lists."""
    two_xi = orbit_twice(q)
    a = np.array([v[0] for v in two_xi], dtype=np.int64)
    b = np.array([v[1] for v in two_xi], dtype=np.int64)
    sites = np.array(list(product(range(q), repeat=dim)), dtype=np.int64).reshape(-1, dim)
    count = q ** dim
    D4a = np.zeros((count, count), dtype=np.int64)
    D4b = np.zeros((count, count), dtype=np.int64)
    for axis in range(dim):
        da = a[sites[:, axis]][:, None] - a[sites[:, axis]][None, :]
        db = b[sites[:, axis]][:, None] - b[sites[:, axis]][None, :]
        D4a += da * da + 5 * db * db
        D4b += 2 * da * db
    adjacency = sgn_array(q * D4a - 4, q * D4b) <= 0
    require(bool(np.all(np.diag(adjacency))), "same-site read missing")
    require(bool(np.all(adjacency == adjacency.T)), "asymmetric neighbour relation")
    neighbors = [np.flatnonzero(adjacency[i]).tolist() for i in range(count)]
    return two_xi, sites, D4a, D4b, adjacency, neighbors


def graph_distances(adjacency: np.ndarray) -> np.ndarray:
    csr = csr_matrix(adjacency.astype(np.int8))
    dist = shortest_path(csr, method="D", unweighted=True, directed=False)
    require(bool(np.all(np.isfinite(dist))), "disconnected site graph")
    return dist.astype(np.int64)


def ceil_sqrt(q: int) -> int:
    r = math.isqrt(q)
    return r + (r * r < q)


def rational_root_upper(x, denominator: int = 10 ** 6) -> Fraction:
    """Smallest grid rational whose square is at least x = (a, b) in the sqrt(5) basis."""
    if sgn(x) == 0:
        return Fraction(0)
    lo, hi = 0, denominator
    while sgn((Fraction(hi, denominator) ** 2 - x[0], -x[1])) < 0:
        hi *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if sgn((Fraction(mid, denominator) ** 2 - x[0], -x[1])) >= 0:
            hi = mid
        else:
            lo = mid
    return Fraction(hi, denominator)


def phi_value(strings) -> float:
    m, n = Fraction(strings[0]), Fraction(strings[1])
    return float(m) + float(n) * (1.0 + math.sqrt(5.0)) / 2.0


# --------------------------------------------------------------------------
# Continuum references (independent quadrature)
# --------------------------------------------------------------------------


def diamond_volume(dim: int, tau: float) -> float:
    return UNIT_BALL[dim] * tau ** (dim + 1) / (2 ** dim * (dim + 1))


def mm_fraction(d: float) -> float:
    return math.gamma(d + 1.0) * math.gamma(d / 2.0) / (2.0 * math.gamma(1.5 * d))


def mm_dimension(f: float):
    if not 0.0 < f < 1.0 or f > mm_fraction(1.01) or f < mm_fraction(20.0):
        return None
    lo, hi = 1.01, 20.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if mm_fraction(mid) > f:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _cap(r: float, d: float) -> float:
    """Volume of the spherical cap of the ball of radius r beyond a plane at distance d."""
    if d >= r:
        return 0.0
    h = r - d
    return math.pi * h * h * (3.0 * r - h) / 3.0


def _wedge(r: float, d1: float, d2: float) -> float:
    """Volume of {p in ball(r): p_1 >= d1, p_2 >= d2}, d1, d2 >= 0."""
    if d1 * d1 + d2 * d2 >= r * r:
        return 0.0
    x1 = math.sqrt(r * r - d2 * d2)

    def segment(x):
        rho2 = r * r - x * x
        if rho2 <= d2 * d2:
            return 0.0
        rho = math.sqrt(rho2)
        return rho2 * math.acos(d2 / rho) - d2 * math.sqrt(rho2 - d2 * d2)

    value, _ = integrate.quad(segment, d1, x1, limit=100)
    return value


def _segment_area(r: float, d: float) -> float:
    if d >= r:
        return 0.0
    return r * r * math.acos(d / r) - d * math.sqrt(r * r - d * d)


def _corner_area(r: float, d1: float, d2: float) -> float:
    if d1 * d1 + d2 * d2 >= r * r:
        return 0.0
    x1 = math.sqrt(r * r - d2 * d2)
    value, _ = integrate.quad(lambda x: math.sqrt(max(r * r - x * x, 0.0)) - d2, d1, x1, limit=100)
    return value


def ball_box(center, r: float, dim: int) -> float:
    """Ball of radius r about center inside [0,1]^dim by inclusion-exclusion."""
    if r <= 0.0:
        return 0.0
    faces = [(c, 1.0 - c) for c in center]
    if dim == 1:
        return max(0.0, min(center[0] + r, 1.0) - max(center[0] - r, 0.0))
    if dim == 2:
        area = math.pi * r * r
        for lo, hi in faces:
            area -= _segment_area(r, lo) + _segment_area(r, hi)
        for d1 in faces[0]:
            for d2 in faces[1]:
                area += _corner_area(r, d1, d2)
        return area
    volume = 4.0 * math.pi * r ** 3 / 3.0
    for lo, hi in faces:
        volume -= _cap(r, lo) + _cap(r, hi)
    for i in range(3):
        for j in range(i + 1, 3):
            for d1 in faces[i]:
                for d2 in faces[j]:
                    volume += _wedge(r, d1, d2)
    corner = min(math.sqrt(d1 * d1 + d2 * d2 + d3 * d3)
                 for d1 in faces[0] for d2 in faces[1] for d3 in faces[2])
    require(r < corner, "ball reaches a cube corner; inclusion-exclusion incomplete")
    return volume


def clipped_volume(center, T: float, dim: int) -> float:
    """2 * integral_0^{T/2} ball_box(r) dr by composite Simpson with face breakpoints."""
    R = T / 2.0
    breaks = sorted({0.0, R} | {v for c in center for v in (c, 1.0 - c) if 0.0 < v < R}
                    | {math.sqrt(u * u + v * v) for u in [w for c in center for w in (c, 1.0 - c)]
                       for v in [w for c in center for w in (c, 1.0 - c)] if 0.0 < math.sqrt(u * u + v * v) < R})
    total = 0.0
    for lo, hi in zip(breaks, breaks[1:]):
        xs = np.linspace(lo, hi, 401)
        ys = np.array([ball_box(center, float(x), dim) for x in xs])
        total += float(integrate.simpson(ys, x=xs))
    return 2.0 * total


def ellipsoid_buffer(px, py, T: float) -> float:
    px, py = np.asarray(px, float), np.asarray(py, float)
    mid = (px + py) / 2.0
    delta = py - px
    ell = float(np.linalg.norm(delta))
    require(ell < T, "moving tips are not timelike")
    u = delta / ell
    b2 = (T * T - ell * ell) / 4.0
    w = np.sqrt((T / 2.0) ** 2 * u ** 2 + b2 * (1.0 - u ** 2))
    return float(min(np.min(mid - w), np.min(1.0 - mid - w)))


# --------------------------------------------------------------------------
# Family checks
# --------------------------------------------------------------------------


def strict_pairs(events: list[tuple[int, int]], dist: np.ndarray) -> int:
    """Ordered pairs (e, f), e strictly below f, by explicit enumeration."""
    layer = np.array([e[0] for e in events], dtype=np.int64)
    site = np.array([e[1] for e in events], dtype=np.int64)
    gap = layer[None, :] - layer[:, None]
    ordered = (gap > 0) & (dist[site][:, site] <= gap)
    return int(np.count_nonzero(ordered))


def check_ordering_block(row: dict, C: Fraction, N: int, dim: int, label: str) -> None:
    if N < 2:
        require(row["ordering_fraction"] is None, label + ": tiny interval ordering fraction")
        return
    f = Fraction(2) * C / (N * (N - 1))
    require(Fraction(row["ordering_fraction"]) == f, label + ": ordering fraction")
    close(row["ordering_fraction_float"], float(f), label + ": ordering fraction float")
    close(row["distance_to_reference"], float(f - REFERENCE_FRACTION[dim]), label + ": distance to reference")
    est = mm_dimension(float(f))
    if est is None:
        require(row["myrheim_meyer_dimension"] is None, label + ": dimension outside the inversion range")
    else:
        close(row["myrheim_meyer_dimension"], est, label + ": Myrheim-Meyer dimension", 1e-7)


def check_estimate(row: dict, key: str, seed: int, populations: dict, label: str) -> Fraction:
    est = row[key]
    strata = est["strata"]
    require(len(strata) >= 1, label + ": strata")
    total = Fraction(0)
    variance = 0.0
    size = 0
    seen = set()
    for stratum in strata:
        skey = tuple(stratum["stratum"]) if isinstance(stratum["stratum"], list) else stratum["stratum"]
        require(skey not in seen, label + ": duplicate stratum")
        seen.add(skey)
        Nh, nh, S, Q = stratum["population"], stratum["sample"], stratum["sum"], stratum["sum_of_squares"]
        require(all(type(v) is int for v in (Nh, nh, S, Q)), label + ": integer stratum fields")
        require(1 <= nh <= Nh, label + ": stratum sample size")
        require(nh >= 2 or nh == Nh, label + ": stratum below two starts")
        require(Q * nh >= S * S, label + ": sum of squares below the mean square")
        if populations is not None:
            require(skey in populations and populations[skey] == Nh, label + f": stratum {skey} population")
        total += Fraction(Nh * S, nh)
        s2 = (Q - S * S / nh) / (nh - 1) if nh > 1 else 0.0
        variance += Nh * Nh * (1.0 - nh / Nh) * s2 / nh
        size += nh
    if populations is not None:
        require(set(populations) == seen, label + ": strata cover the support")
    require(est["sample_size"] == size and size >= SAMPLE_MINIMUM, label + ": sample size")
    require(Fraction(est["value"]) == total, label + ": stratified estimate arithmetic")
    close(est["float"], float(total), label + ": estimate float")
    close(est["standard_error"], math.sqrt(variance), label + ": standard error", 1e-9)
    return total


def check_family(fam: dict, rebuilt: bool, expected_dim: int, n: int) -> dict:
    dim = fam["dimension"]
    require(dim == expected_dim and fam["fibonacci_index"] == n, "family census")
    q, p = fib(n)
    require(fam["q"] == q and fam["p"] == p, "Fibonacci pair")
    K = ceil_sqrt(q)
    require(fam["layer_steps"] == K, "layer count")
    count = q ** dim
    require(fam["site_count"] == count and fam["record_count"] == count, "site count")
    label = f"q={q} dim={dim}"

    two_xi = orbit_twice(q)
    perm = [(b * p) % q for b in range(q)]
    xi = [(Fraction(x[0], 2), Fraction(x[1], 2)) for x in two_xi]
    require(fam["orbit_Qphi"] == [to_phi_strings(x) for x in xi], label + ": orbit")
    require(fam["grid_permutation"] == perm, label + ": grid permutation")
    order = sorted(range(q), key=lambda i: perm[i])

    def minus(x, y):
        return (x[0] - y[0], x[1] - y[1])

    def times(c, x):
        return (c * x[0], c * x[1])

    def mult(x, y):
        return (x[0] * y[0] + 5 * x[1] * y[1], x[0] * y[1] + x[1] * y[0])

    radius = minus((1, 0), xi[order[-1]])
    for a_, b_ in zip(order, order[1:]):
        half = times(Fraction(1, 2), minus(xi[b_], xi[a_]))
        if sgn(minus(half, radius)) > 0:
            radius = half
    h2 = times(dim, mult(radius, radius))
    require(fam["one_dimensional_fill_over_L_Qphi"] == to_phi_strings(radius), label + ": covering radius")
    require(fam["whole_cube_fill_h_squared_over_L2_Qphi"] == to_phi_strings(h2), label + ": h squared")
    ratio = rational_root_upper(times(q, h2))
    require(fam["h_over_a_upper"] == str(ratio), label + ": h/a upper bound")
    inner = max(Fraction(0), 1 - 2 * ratio)
    require(fam["certified_inner_speed_lower"] == str(inner), label + ": inner speed")
    require(fam["positive_inner_cone"] == (sgn(minus((1, 0), times(4 * q, h2))) > 0), label + ": inner cone flag")
    require(fam["radius_squared_over_L2"] == str(Fraction(1, q)), label + ": edge radius")
    require(fam["layer_time_equals_radius"] is True, label + ": layer clock")
    h_float = math.sqrt(float(h2[0]) + float(h2[1]) * math.sqrt(5.0))
    close(fam["covering_radius_h_over_L"], h_float, label + ": covering radius float")
    close(fam["h_over_a"], math.sqrt(q) * h_float, label + ": h over a")
    close(fam["edge_radius_a_over_L"], 1.0 / math.sqrt(q), label + ": edge radius float")
    if dim == 3:
        close(fam["assignment_error_H_over_L"], 2.0 * math.sqrt(3.0) / q, label + ": H_q")
        require(fam["quadrature_displacement_H_squared_over_L2"] == str(Fraction(12, q * q)), label + ": H squared")
        require(fam["word_length_bound"] == 27 * (q - 1), label + ": word bound")
    else:
        require(fam["assignment_error_H_over_L"] is None, label + ": H_q only in three dimensions")

    center_axis = 0
    half = (Fraction(1, 2), Fraction(0))
    for b in range(1, q):
        d, old = minus(xi[b], half), minus(xi[center_axis], half)
        if sgn(minus(mult(d, d), mult(old, old))) < 0:
            center_axis = b
    center = sum(center_axis * q ** (dim - 1 - i) for i in range(dim))
    require(fam["centre_axis"] == center_axis and fam["intervention_source_id"] == center, label + ": centre")
    clearance = xi[center_axis]
    if sgn(minus(minus((1, 0), clearance), clearance)) < 0:
        clearance = minus((1, 0), clearance)
    require(fam["clearance_over_L_Qphi"] == to_phi_strings(clearance), label + ": clearance")
    clearance_float = float(clearance[0]) + float(clearance[1]) * math.sqrt(5.0)
    density = q ** (dim + 0.5)
    close(fam["event_density_times_L_to_dim_plus_one"], density, label + ": density")
    require(fam["exact_width"] == count and fam["exact_height_in_events"] == K + 1, label + ": width/height")

    shell = fam["shell_counts_from_centre"]
    require(type(shell) is list and len(shell) == K + 1 and all(type(v) is int and v >= 0 for v in shell),
            label + ": shell counts")
    require(shell[0] == 1, label + ": centre shell")
    xi_float = [float(x[0]) + float(x[1]) * math.sqrt(5.0) for x in xi]
    center_position = [xi_float[center_axis]] * dim
    a_q = 1.0 / math.sqrt(q)
    H_q = 2.0 * math.sqrt(3.0) / q if dim == 3 else None
    h_over_a = math.sqrt(q) * h_float

    # Structural rebuild at the small levels.
    dist = None
    if rebuilt:
        _, sites, D4a, D4b, adjacency, neighbors = family_geometry(q, dim)
        dist = graph_distances(adjacency)
        degrees = adjacency.sum(axis=1)
        require(fam["neighbors_including_wait_sha256"] == hashed(neighbors), label + ": neighbour digest")
        require(fam["undirected_spatial_edges"] == (int(degrees.sum()) - count) // 2, label + ": edges")
        require(fam["minimum_neighbor_count_including_wait"] == int(degrees.min()), label + ": min degree")
        require(fam["maximum_neighbor_count_including_wait"] == int(degrees.max()), label + ": max degree")
        probes = []
        for start in sorted({0, center, count - 1}):
            for k in range(1, K + 1):
                outer = sgn_array(q * D4a[start] - 4 * k * k, q * D4b[start]) <= 0
                reached = dist[start] <= k
                require(not np.any(reached & ~outer), label + ": reached outside the outer cone")
                # Python integers: the inner-speed denominator squared exceeds int64 under the sign rule.
                den2, num2 = inner.denominator ** 2, inner.numerator ** 2
                for i in np.flatnonzero(~reached).tolist():
                    small = sgn((q * int(D4a[start, i]) * den2 - 4 * k * k * num2,
                                 q * int(D4b[start, i]) * den2)) <= 0
                    require(not small, label + ": inner cone unreachable")
                missing = np.flatnonzero(outer & ~reached).tolist()
                probes.append({"start": start, "layers": k, "reachable_count": int(reached.sum()),
                               "reachable_ids_sha256": hashed(np.flatnonzero(reached).tolist()),
                               "outer_cone_violations": 0, "certified_inner_cone_misses": 0,
                               "exact_finite_cone_missing_count": len(missing),
                               "missing_ids_sha256": hashed(missing)})
        require(fam["reachability_probes"] == probes, label + ": reachability probes")
        require(shell == np.bincount(dist[center], minlength=K + 1)[:K + 1].tolist(), label + ": shell counts")
        if dim == 3:
            records, costs = [], []
            floors = golden_floors(q)
            for b in sites.tolist():
                a_ = [-floors[i] for i in b]
                z = [b[1] - a_[0], b[1] + a_[0], b[2] - a_[1], b[2] + a_[1], b[0] - a_[2], b[0] + a_[2]]
                require(sum(z) == 2 * sum(b), label + ": record coordinate sum")
                currents = [-a_[2] - b[0], a_[0] + b[0] + b[2], b[0] + b[1] - b[2],
                            -a_[0] - b[0] + b[2], a_[2] + b[1] - b[2], a_[1] + b[2]]
                records.append([b, z, currents])
                costs.append(sum(map(abs, currents)))
            require(fam["source_records_sha256"] == hashed(records), label + ": source records")
            require(fam["source_record_examples"] == [records[i] for i in sorted({0, center, count - 1})],
                    label + ": record examples")
            require(fam["maximum_word_length"] == max(costs) and fam["sum_word_lengths"] == sum(costs),
                    label + ": word lengths")
            require(max(costs) <= 27 * (q - 1), label + ": word bound violated")
    else:
        for name in ("neighbors_including_wait_sha256", "source_records_sha256"):
            if name in fam:
                require(type(fam[name]) is str and len(fam[name]) == 64, label + ": digest field")
        probes = fam["reachability_probes"]
        centre_reach = {row["layers"]: row["reachable_count"] for row in probes if row["start"] == center}
        for k in range(1, K + 1):
            require(centre_reach.get(k) == sum(shell[:k + 1]), label + ": centre probe against shells")
        for row in probes:
            require(row["outer_cone_violations"] == 0 and row["certified_inner_cone_misses"] == 0,
                    label + ": cone sandwich flags")

    # Vertical intervals for every k <= K.
    intervals = fam["vertical_intervals"]
    require(len(intervals) == K, label + ": vertical interval census")
    exact_vertical = fam["vertical_pair_counting"] == "exact_all_pairs"
    require(fam["vertical_pair_counting"] in ("exact_all_pairs", "stratified_sample"), label + ": pair counting mode")
    support = sum(shell[:K // 2 + 1])
    require(fam["vertical_support_site_count"] == support, label + ": vertical support")
    vertical_events = None
    if rebuilt:
        require(exact_vertical, label + ": rebuilt levels are exact")
        alpha = dist[center]
    histogram = fam["vertical_strict_pair_histogram"]
    if exact_vertical:
        require(type(histogram) is list and len(histogram) == K // 2 + 1, label + ": histogram shape")
    else:
        require(histogram is None, label + ": no histogram for sampled levels")
        require(fam["vertical_sample"]["seed"] == SAMPLE_SEED + 1000 * dim + q, label + ": vertical seed")
    Cs = {}
    for row, k in zip(intervals, range(1, K + 1)):
        rl = f"{label} k={k}"
        require(row["layers"] == k, rl + ": layer label")
        per_layer = [sum(shell[a] for a in range(K + 1) if a <= min(j, k - j)) for j in range(k + 1)]
        require(row["counts_by_layer"] == per_layer, rl + ": counts by layer")
        N = sum(per_layer)
        require(row["inclusive_event_count"] == N, rl + ": inclusive count")
        inside = sgn(minus(times(4 * q, mult(clearance, clearance)), (k * k, 0))) >= 0
        require(row["continuum_diamond_inside_cube"] == inside, rl + ": inside flag")
        T = k * a_q
        close(row["model_time_T_over_L"], T, rl + ": T")
        volume = diamond_volume(dim, T)
        normalized = N / density
        close(row["normalized_count"], normalized, rl + ": normalized count")
        close(row["continuum_diamond_volume"], volume, rl + ": continuum volume")
        close(row["deviation_from_continuum"], normalized - volume, rl + ": deviation")
        close(row["relative_deviation_from_continuum"], normalized / volume - 1.0, rl + ": relative deviation")
        clipped = clipped_volume(center_position, T, dim)
        close(row["clipped_diamond_volume"], clipped, rl + ": clipped volume", 1e-6)
        close(row["relative_deviation_from_clipped"], normalized / row["clipped_diamond_volume"] - 1.0,
              rl + ": relative deviation from clipped")
        if dim == 3:
            require(row["count_over_L4_times_sqrt_q"] == str(Fraction(N, q ** 3)), rl + ": RER count ratio")
            require(row["diamond_volume_over_pi_L4"] == str(Fraction(k ** 4, 24 * q * q)), rl + ": RER volume ratio")
            delta = a_q
            bound = 4.0 * math.pi * (T + delta) * (T / 2.0 + H_q) ** 2 * (H_q + T * h_over_a) \
                + math.pi * T ** 3 * delta / 2.0
            close(row["volume_error_bound"], bound, rl + ": volume bound")
            close(row["volume_error_bound_relative"], bound / volume, rl + ": relative bound")
            require(row["bound_hypothesis_ball_inside_cube"] == (clearance_float >= T / 2.0 + H_q),
                    rl + ": bound hypothesis")
            require(row["actual_deviation_within_bound"] == (abs(normalized - volume) <= bound),
                    rl + ": deviation within bound")
        else:
            require(row["count_over_L4_times_sqrt_q"] is None and row["diamond_volume_over_pi_L4"] is None,
                    rl + ": RER ratios only in three dimensions")
        if exact_vertical:
            require(row["pair_counting"] == "exact_all_pairs", rl + ": pair counting label")
            C = row["strict_pair_count"]
            require(type(C) is int and C >= 0, rl + ": strict pair count")
            require(row["ordering_fraction_standard_error"] is None, rl + ": exact rows carry no error")
            if rebuilt:
                events = [(j, s) for j in range(k + 1) for s in range(count) if alpha[s] <= min(j, k - j)]
                require(len(events) == N, rl + ": event enumeration")
                require(strict_pairs(events, dist) == C, rl + ": strict pairs by enumeration")
            C = Fraction(C)
        else:
            require(row["pair_counting"] == "stratified_sample", rl + ": pair counting label")
            populations = {a: shell[a] for a in range(K // 2 + 1) if shell[a] > 0}
            C = check_estimate(row, "strict_pair_count_estimate", fam["vertical_sample"]["seed"], populations, rl)
            se = row["strict_pair_count_estimate"]["standard_error"]
            close(row["ordering_fraction_standard_error"], 2.0 * se / (N * (N - 1)), rl + ": fraction error")
        check_ordering_block(row, C, N, dim, rl)
        Cs[k] = (C, N)
    if exact_vertical:
        for k in range(1, K + 1):
            total = 0
            for a in range(K // 2 + 1):
                for b in range(K // 2 + 1):
                    for d in range(K + 2):
                        w = sum(1 for j in range(a, k - a + 1) for jj in range(b, k - b + 1)
                                if jj - j >= max(d, 1))
                        total += histogram[a][b][d] * w
            require(total == Cs[k][0], f"{label} k={k}: histogram against pair count")

    # Moving-tip interval.
    moving = fam["moving_tip_interval"]
    if moving is not None:
        ml = label + " moving"
        require(moving["layers"] == K, ml + ": layers")
        s_ = moving["rank_shift"]
        rc = perm[center_axis]
        require(1 <= s_ and rc - s_ >= 0 and rc + s_ < q, ml + ": rank shift window")
        xl, yl = order[rc - s_], order[rc + s_]
        require(moving["x_axis_label"] == xl and moving["y_axis_label"] == yl, ml + ": tip labels")
        x = sum(xl * q ** (dim - 1 - i) for i in range(dim))
        y = sum(yl * q ** (dim - 1 - i) for i in range(dim))
        require(moving["x_site"] == x and moving["y_site"] == y, ml + ": tip sites")
        diff = minus(xi[yl], xi[xl])
        ell2 = times(dim, mult(diff, diff))
        require(moving["tip_separation_squared_over_L2_Qphi"] == to_phi_strings(ell2), ml + ": separation")
        require(sgn(minus(times(q, ell2), (K * K, 0))) < 0, ml + ": timelike tips")
        T = K * a_q
        ell2_float = float(ell2[0]) + float(ell2[1]) * math.sqrt(5.0)
        tau = math.sqrt(T * T - ell2_float)
        close(moving["tip_separation_over_T"], math.sqrt(ell2_float) / T, ml + ": separation over T")
        close(moving["proper_duration_over_L"], tau, ml + ": proper duration")
        buffer = ellipsoid_buffer([xi_float[xl]] * dim, [xi_float[yl]] * dim, T)
        close(moving["spatial_buffer_over_L"], buffer, ml + ": buffer", 1e-9)
        require(buffer > 0.0, ml + ": positive buffer")
        rule = moving["tip_selection_rule"]
        require(rule in ("buffer_at_least_H_q", "buffer_positive"), ml + ": selection rule")
        require(moving["spatial_buffer_at_least_H_q"] == (dim == 3 and buffer >= H_q), ml + ": buffer flag")
        if rule == "buffer_at_least_H_q":
            require(dim == 3 and buffer >= H_q, ml + ": rule against buffer")
        # Smaller shifts fail the rule that fired; for the positive rule in three
        # dimensions every smaller shift also fails the H_q rule.
        for s2 in range(1, s_):
            xl2, yl2 = order[rc - s2], order[rc + s2]
            diff2 = minus(xi[yl2], xi[xl2])
            e2 = times(dim, mult(diff2, diff2))
            if sgn(minus(times(q, e2), (K * K, 0))) >= 0:
                continue
            buf2 = ellipsoid_buffer([xi_float[xl2]] * dim, [xi_float[yl2]] * dim, T)
            floor_ = H_q if rule == "buffer_at_least_H_q" else 0.0
            require(not (buf2 > 0.0 and buf2 >= floor_), ml + f": smaller shift {s2} satisfies the rule")
        N = moving["inclusive_event_count"]
        per_layer = moving["counts_by_layer"]
        require(type(per_layer) is list and len(per_layer) == K + 1 and sum(per_layer) == N, ml + ": layer counts")
        require(per_layer[0] == 1 and per_layer[-1] == 1, ml + ": tips are single events")
        volume = diamond_volume(dim, tau)
        normalized = N / density
        close(moving["normalized_count"], normalized, ml + ": normalized count")
        close(moving["continuum_diamond_volume"], volume, ml + ": continuum volume")
        close(moving["deviation_from_continuum"], normalized - volume, ml + ": deviation")
        close(moving["relative_deviation_from_continuum"], normalized / volume - 1.0, ml + ": relative deviation")
        if dim == 3:
            delta = a_q
            bound = 8.0 * math.pi * (T + delta) * (T + H_q) ** 2 * (H_q + 2.0 * T * h_over_a) \
                + 4.0 * math.pi * T ** 3 * delta
            close(moving["volume_error_bound"], bound, ml + ": general bound")
            close(moving["volume_error_bound_relative"], bound / volume, ml + ": relative general bound")
            require(moving["actual_deviation_within_bound"] == (abs(normalized - volume) <= bound),
                    ml + ": deviation within bound")
        if rebuilt:
            alpha_m, gamma_m = dist[x], dist[y]
            counts = [int(np.count_nonzero((alpha_m <= j) & (gamma_m <= K - j))) for j in range(K + 1)]
            require(counts == per_layer, ml + ": counts by layer by enumeration")
            require(moving["support_site_count"] == int(np.count_nonzero(alpha_m + gamma_m <= K)), ml + ": support")
        if moving["pair_counting"] == "exact_all_pairs":
            C = moving["strict_pair_count"]
            require(type(C) is int and C >= 0, ml + ": strict pair count")
            require(moving["ordering_fraction_standard_error"] is None, ml + ": exact rows carry no error")
            if rebuilt:
                events = [(j, s) for j in range(K + 1) for s in range(count)
                          if alpha_m[s] <= j and gamma_m[s] <= K - j]
                require(strict_pairs(events, dist) == C, ml + ": strict pairs by enumeration")
            C = Fraction(C)
        else:
            require(moving["pair_counting"] == "stratified_sample", ml + ": pair counting label")
            require(not rebuilt, ml + ": rebuilt levels are exact")
            seed = fam["moving_sample"]["seed"]
            require(seed == SAMPLE_SEED + 1000 * dim + q + 500, ml + ": moving seed")
            C = check_estimate(moving, "strict_pair_count_estimate", seed, None, ml)
            populations = sum(r["population"] for r in moving["strict_pair_count_estimate"]["strata"])
            require(populations == moving["support_site_count"], ml + ": strata cover the moving support")
            for r in moving["strict_pair_count_estimate"]["strata"]:
                a_, g_ = r["stratum"]
                require(0 <= a_ and 0 <= g_ and a_ + g_ <= K, ml + ": stratum inside the support")
            se = moving["strict_pair_count_estimate"]["standard_error"]
            close(moving["ordering_fraction_standard_error"], 2.0 * se / (N * (N - 1)), ml + ": fraction error")
        check_ordering_block(moving, C, N, dim, ml)

    # Count clock.
    clock = fam["count_clock"]
    Kp = K // 2
    NI, NJ = Cs[K][1], Cs[Kp][1]
    require(clock["reference_layers"] == Kp and clock["interval_layers"] == K, label + ": clock layers")
    require(clock["interval_count"] == NI and clock["reference_count"] == NJ, label + ": clock counts")
    root = 1.0 / (dim + 1)
    require(clock["clock_exponent"] == f"1/{dim + 1}", label + ": clock exponent")
    close(clock["count_clock"], (NI / NJ) ** root, label + ": count clock")
    close(clock["model_time_ratio"], K / Kp, label + ": model time ratio")
    close(clock["relative_deviation"], (NI / NJ) ** root / (K / Kp) - 1.0, label + ": clock deviation")
    if dim == 3:
        EI = intervals[K - 1]["volume_error_bound"] * density
        EJ = intervals[Kp - 1]["volume_error_bound"] * density
        close(clock["count_error_interval"], EI, label + ": count error I")
        close(clock["count_error_reference"], EJ, label + ": count error J")
        close(clock["enclosure_lower"], (max(NI - EI, 0.0) / (NJ + EJ)) ** 0.25, label + ": enclosure lower")
        if NJ > EJ:
            close(clock["enclosure_upper"], ((NI + EI) / (NJ - EJ)) ** 0.25, label + ": enclosure upper")
        else:
            require(clock["enclosure_upper"] is None, label + ": enclosure upper undefined")
        require(clock["enclosure_hypotheses_satisfied"] == (
            intervals[K - 1]["bound_hypothesis_ball_inside_cube"]
            and intervals[Kp - 1]["bound_hypothesis_ball_inside_cube"] and NJ > EJ), label + ": enclosure flag")
    return {"q": q, "dimension": dim, "rebuilt": rebuilt,
            "ordering_fraction": intervals[K - 1]["ordering_fraction_float"],
            "myrheim_meyer_dimension": intervals[K - 1]["myrheim_meyer_dimension"]}


# --------------------------------------------------------------------------
# Receipt-level checks
# --------------------------------------------------------------------------


def check_cross_check(fam: dict, rer_level: dict | None) -> None:
    block = fam["rer_cross_check"]
    label = f"q={fam['q']} cross-check"
    require(block["fibonacci_index"] == fam["fibonacci_index"] and block["q"] == fam["q"], label + ": level")
    fields = block["fields"]
    require(len(fields) >= 20, label + ": compared field census")
    for name, entry in fields.items():
        require(name in fam, label + f": unknown field {name}")
        require(entry["agree"] is True, label + f": field {name} disagrees")
        if "rer" in entry:
            require(packed(entry["rer"]) == packed(fam[name]), label + f": embedded RER value of {name}")
        if rer_level is not None:
            require(packed(rer_level[name]) == packed(fam[name]), label + f": theory receipt value of {name}")
    require(block["all_agree"] is True, label + ": all_agree")


def verify(receipt: dict) -> dict:
    require(type(receipt) is dict, "receipt object")
    require(receipt.get("schema") == SCHEMA, "schema")
    require(receipt.get("scope") == EXPECTED_SCOPE, "scope block")
    require(type(receipt.get("claim_boundary")) is str and len(receipt["claim_boundary"]) > 100, "claim boundary")
    require(receipt["source_scale_L_squared_Qphi"] == ["12/5", "-4/5"], "source scale")
    require(receipt["reference_fractions"] == {"4": "1/10", "3": "8/35", "2": "1/2"}, "reference fractions")
    counting = receipt["pair_counting"]
    require(counting["sample_size_minimum"] == SAMPLE_MINIMUM and counting["sample_seed_base"] == SAMPLE_SEED,
            "sampling declaration")
    require(type(counting["exact_support_limit"]) is int and counting["exact_support_limit"] > 0, "exact limit")
    pins = receipt["source_pins"]
    for rel in LOCAL_PINS:
        require(pins.get(rel) == hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), "pin " + rel)
    rer_present = all((RER_ROOT / rel).is_file() for rel in RER_PINS)
    rer_levels = None
    if rer_present:
        for rel in RER_PINS:
            require(pins.get("reverse-engineering-reality/" + rel)
                    == hashlib.sha256((RER_ROOT / rel).read_bytes()).hexdigest(), "theory pin " + rel)
        rer_bytes = (RER_ROOT / RER_PINS[1]).read_bytes()
        require(hashlib.sha256(rer_bytes).hexdigest() == RER_RECEIPT_SHA256, "theory receipt digest")
        rer_levels = {lv["fibonacci_index"]: lv for lv in json.loads(rer_bytes)["levels"]}
    require(receipt["rer_cross_check"]["receipt_sha256"] == RER_RECEIPT_SHA256, "pinned theory receipt digest")
    require(receipt["rer_cross_check"]["levels_compared"] == list(REBUILT_LEVELS), "compared levels")
    require(set(pins) == set(LOCAL_PINS) | {"reverse-engineering-reality/" + p for p in RER_PINS}, "pin census")

    levels = receipt["levels"]
    require([lv["fibonacci_index"] for lv in levels] == list(LEVELS), "level census")
    rows = []
    all_agree = []
    for lv in levels:
        n = lv["fibonacci_index"]
        q, _ = fib(n)
        require(lv["q"] == q, "level q")
        families = lv["families"]
        require([f["dimension"] for f in families] == list(DIMENSIONS), "family census")
        for dim, fam in zip(DIMENSIONS, families):
            rows.append(check_family(fam, rebuilt=n in REBUILT_LEVELS, expected_dim=dim, n=n))
            if dim == 3 and n in REBUILT_LEVELS:
                require("rer_cross_check" in fam, "missing cross-check block")
                check_cross_check(fam, None if rer_levels is None else rer_levels[n])
                all_agree.append(fam["rer_cross_check"]["all_agree"])
            else:
                require("rer_cross_check" not in fam, "unexpected cross-check block")
    require(receipt["rer_cross_check"]["all_agree"] is (bool(all_agree) and all(all_agree)), "cross-check summary")
    return {"accepted": True, "schema": SCHEMA, "rebuilt_levels": list(REBUILT_LEVELS),
            "theory_pins_checked": rer_present, "families": rows,
            "receipt_sha256": hashlib.sha256(packed(receipt)).hexdigest()}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(verify(load(args.receipt)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
