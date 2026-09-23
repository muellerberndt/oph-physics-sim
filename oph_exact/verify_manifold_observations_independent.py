"""Independent replay of the manifold-observations receipt.

The producer is not imported.  The golden orbit, the site graphs, the graph
distances and every diamond of the three families at q = 8 and q = 13 are
rebuilt here with an own sign rule for a + b*phi, dense all-pairs
neighbour decisions, scipy shortest paths and explicit event enumeration.
Dimension readouts, the abundance profiles (between-count matrix), the
Benincasa-Dowker action, the link-direction multipoles (complex harmonics
from scipy) and the count clocks of those levels are recomputed exactly.
Levels above q = 13 are checked for internal consistency: ordering fractions
against pair counts, sampled estimates against their strata rows, ratios
against counts, summaries against rows, clocks against counts, volumes and
bounds against their formulas, the trend table against the family blocks.
The schema, the scope block, the certificate rows and the file pins are
checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.special import sph_harm_y

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/exact/manifold_observations_receipt.json"
SCHEMA = "oph.exact.manifold-observations.v1"
REBUILT_LEVELS = (6, 7)
LEVELS = (6, 7, 8, 9)
DIMENSIONS = (3, 2, 1)
PINS = (
    "oph_exact/manifold_observations.py",
    "oph_exact/verify_manifold_observations_independent.py",
    "tests/test_exact_manifold_observations.py",
    "oph_exact/source_net.py",
    "data/exact/source_net_causal_limit_receipt.json",
)
EXPECTED_SCOPE = {
    "population_supplied": True,
    "read_law_supplied": True,
    "tick_supplied": True,
    "one_event_per_site_and_layer_supplied": True,
    "exact_gram_metric_edge_decisions": True,
    "exact_graph_distance_order": True,
    "readouts_from_order_and_cardinality_only": True,
    "sprinkling_references_are_imported_controls": True,
    "native_repair_selected": False,
    "physical_clock_or_spacetime_identified": False,
    "finite_runs_demonstrate_asymptotic_limit": False,
    "dimension_statistic_used_as_acceptance": False,
    "hawking_king_mccarthy_malament_computed": False,
    "hawking_king_mccarthy_malament_declared": True,
    "physical_event_interpretation_addressed": False,
}
CERTIFICATE_STATUS = {
    "C1_physical_event_interpretation": "not addressed",
    "C2_dense_isotropic_link_directions": "computed",
    "C3_order_preserving_refinement": "computed",
    "C4_calibrated_density_law": "computed",
    "C5_independent_dimension_and_topology_tests": "computed",
    "C6_distinguishing_lorentzian_geometry_with_uniqueness_control": "declared",
}
REFERENCE_FRACTION = {1: Fraction(1, 2), 2: Fraction(8, 35), 3: Fraction(1, 10)}
UNIT_BALL = {1: 2.0, 2: math.pi, 3: 4.0 * math.pi / 3.0}
PHI = (1.0 + math.sqrt(5.0)) / 2.0
MAX_M = 15
LOG2_BINS = 16
BINS = MAX_M + 1 + LOG2_BINS
FLOAT_TOLERANCE = 1e-9


class ManifoldObservationsVerificationError(ValueError):
    pass


def require(condition, label: str) -> None:
    if not condition:
        raise ManifoldObservationsVerificationError(label)


def close(x, y, label: str, tolerance: float = FLOAT_TOLERANCE) -> None:
    require(isinstance(x, (int, float)) and isinstance(y, (int, float)), label + ": numeric")
    require(abs(float(x) - float(y)) <= tolerance * max(1.0, abs(float(y))), f"{label}: {x} vs {y}")


def rounded(x: float) -> float:
    return float(f"{float(x):.12g}")


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
    raise ManifoldObservationsVerificationError("non-finite JSON token " + token)


def packed(x) -> bytes:
    return (json.dumps(x, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def load(path: Path = OUTPUT) -> dict:
    data = Path(path).read_bytes()
    require(len(data) <= 40_000_000, "receipt size limit")
    receipt = json.loads(data.decode("ascii"), object_pairs_hook=_pairs, parse_constant=_forbidden)
    require(packed(receipt) == data, "receipt bytes are not canonical JSON")
    return receipt


# --------------------------------------------------------------------------
# Own golden arithmetic: x = a + b*phi with integers a, b
# --------------------------------------------------------------------------


def sign_phi(a: int, b: int) -> int:
    """Sign of a + b*phi: with c = 2a + b the value is (c + b*sqrt5)/2."""
    c = 2 * a + b
    if b == 0:
        return (c > 0) - (c < 0)
    if b > 0 and c >= 0:
        return 1
    if b < 0 and c <= 0:
        return -1
    # c and b have opposite signs; |c| against |b| sqrt5 decides
    return (1 if c > 0 else -1) if c * c > 5 * b * b else (1 if b > 0 else -1)


def sign_phi_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    c = 2 * a + b
    big = c * c > 5 * b * b
    opposite = np.where(big, np.sign(c), np.sign(b))
    out = np.where(b == 0, np.sign(c), np.where((b > 0) & (c >= 0), 1, np.where((b < 0) & (c <= 0), -1, opposite)))
    return out.astype(np.int64)


def fib(n: int) -> tuple[int, int]:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a, b


def golden_floor(b: int) -> int:
    """floor(b*phi) = floor((b + sqrt(5 b^2))/2) = (b + isqrt(5 b^2)) // 2."""
    return (b + math.isqrt(5 * b * b)) // 2


def orbit_pairs(q: int) -> tuple[np.ndarray, np.ndarray]:
    """xi_b = b*phi - floor(b*phi) = m_b + b*phi with m_b = -floor(b*phi)."""
    m = np.array([-golden_floor(b) for b in range(q)], dtype=np.int64)
    return m, np.arange(q, dtype=np.int64)


def ceil_sqrt(q: int) -> int:
    r = math.isqrt(q)
    return r + (r * r < q)


class Rebuilt:
    """Dense rebuild of one family: neighbours, distances, positions."""

    def __init__(self, n: int, dim: int) -> None:
        q, p = fib(n)
        self.n, self.dim, self.q, self.p = n, dim, q, p
        self.K = ceil_sqrt(q)
        self.m, self.b = orbit_pairs(q)
        self.xi = self.m + self.b * PHI
        self.count = q ** dim
        self.sites = np.array(list(product(range(q), repeat=dim)), dtype=np.int64)
        require(all(int(sum(s[i] * q ** (dim - 1 - i) for i in range(dim))) == k for k, s in enumerate(self.sites)),
                "site indexing")
        # squared differences per axis pair: (dm + db*phi)^2 = dm^2 + db^2 + (2 dm db + db^2) phi
        dm = self.m[:, None] - self.m[None, :]
        db = self.b[:, None] - self.b[None, :]
        A1 = dm * dm + db * db
        B1 = 2 * dm * db + db * db
        A = np.zeros((self.count, self.count), dtype=np.int64)
        B = np.zeros((self.count, self.count), dtype=np.int64)
        for axis in range(dim):
            lab = self.sites[:, axis]
            A += A1[lab[:, None], lab[None, :]]
            B += B1[lab[:, None], lab[None, :]]
        self.adjacent = sign_phi_array(q * A - 1, q * B) <= 0
        require(bool(np.all(np.diag(self.adjacent))), "same-site read")
        graph = csr_matrix(self.adjacent.astype(np.int8))
        self.dist = shortest_path(graph, unweighted=True, directed=False).astype(np.int64)
        require(int(self.dist.max()) < 10 ** 6, "connected")
        self.positions = self.xi[self.sites]
        self.clearance = np.min(np.minimum(self.positions, 1.0 - self.positions), axis=1)
        self.a_q = 1.0 / math.sqrt(q)
        self.density = q ** (dim + 0.5)
        self.perm = [(bb * p) % q for bb in range(q)]
        centre_axis = min(range(q), key=lambda bb: (abs(self.xi[bb] - 0.5), bb))
        self.centre_axis = centre_axis
        self.centre = int(sum(centre_axis * q ** (dim - 1 - i) for i in range(dim)))
        ordered = sorted(range(q), key=lambda bb: self.perm[bb])
        eta = self.xi[ordered]
        r_q = max(0.5 * float(np.max(np.diff(eta))), 1.0 - float(eta[-1]))
        self.h_over_a = math.sqrt(q * dim * r_q * r_q)
        self.H_q = 2.0 * math.sqrt(3.0) / q if dim == 3 else None

    def events(self, x: int, y: int, K: int) -> tuple[np.ndarray, np.ndarray]:
        alpha = self.dist[x]
        gamma = self.dist[:, y]
        site, layer = [], []
        for s in range(self.count):
            for j in range(int(alpha[s]), K - int(gamma[s]) + 1):
                site.append(s)
                layer.append(j)
        return np.array(site, dtype=np.int64), np.array(layer, dtype=np.int64)

    def future_matrix(self, site: np.ndarray, layer: np.ndarray) -> np.ndarray:
        dl = layer[None, :] - layer[:, None]
        return (dl > 0) & (self.dist[site[:, None], site[None, :]] <= dl)


# --------------------------------------------------------------------------
# Own statistics
# --------------------------------------------------------------------------


def mm_fraction(d: float) -> float:
    return math.gamma(d + 1.0) * math.gamma(d / 2.0) / (2.0 * math.gamma(1.5 * d))


def mm_dimension(f: float):
    if not 0.0 < f < 1.0:
        return None
    lo, hi = 1.01, 20.0
    if not (mm_fraction(hi) <= f <= mm_fraction(lo)):
        return None
    for _ in range(96):
        mid = (lo + hi) / 2.0
        if mm_fraction(mid) > f:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def diamond_volume(dim: int, tau: float) -> float:
    return UNIT_BALL[dim] * tau ** (dim + 1) / (2 ** dim * (dim + 1))


def vertical_bound(T, delta, H, hoa) -> float:
    return 4.0 * math.pi * (T + delta) * (T / 2.0 + H) ** 2 * (H + T * hoa) + math.pi * T ** 3 * delta / 2.0


def general_bound(T, delta, H, hoa) -> float:
    return 8.0 * math.pi * (T + delta) * (T + H) ** 2 * (H + 2.0 * T * hoa) + 4.0 * math.pi * T ** 3 * delta


def profile_bins(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.int64)
    out = np.empty(len(m), dtype=np.int64)
    for i, v in enumerate(m.tolist()):
        out[i] = v if v <= MAX_M else MAX_M + 1 + min(v.bit_length() - 5, LOG2_BINS - 1)
    return out


def profile_from_future(F: np.ndarray) -> np.ndarray:
    prod = F.astype(np.float64) @ F.astype(np.float64)
    between = np.rint(prod[F]).astype(np.int64)
    return np.bincount(profile_bins(between), minlength=BINS)


def bd_action(N: int, counts) -> float:
    return 4.0 / math.sqrt(6.0) * (N - counts[0] + 9 * counts[1] - 16 * counts[2] + 8 * counts[3])


def sphere_powers(unit: np.ndarray) -> list[float]:
    """4 pi sum_m |mean Y_lm|^2 with scipy's complex harmonics, l = 0..4."""
    theta = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
    phi = np.arctan2(unit[:, 1], unit[:, 0])
    out = []
    for l in range(5):
        power = 0.0
        for m in range(-l, l + 1):
            power += abs(np.mean(sph_harm_y(l, m, theta, phi))) ** 2
        out.append(4.0 * math.pi * power)
    return out


def circle_powers(unit: np.ndarray) -> list[float]:
    theta = np.arctan2(unit[:, 1], unit[:, 0])
    out = [1.0]
    for m in range(1, 5):
        out.append(2.0 * float(abs(np.mean(np.exp(1j * m * theta))) ** 2))
    return out


def summary(values) -> dict:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"count": 0, "mean": None, "sd": None, "min": None, "max": None}
    arr = np.array(vals)
    return {"count": len(vals), "mean": rounded(arr.mean()),
            "sd": rounded(arr.std(ddof=1)) if len(vals) > 1 else None,
            "min": rounded(arr.min()), "max": rounded(arr.max())}


def check_summary(block: dict, values, label: str) -> None:
    mine = summary(values)
    require(block["count"] == mine["count"], label + ": count")
    for key in ("mean", "sd", "min", "max"):
        if mine[key] is None:
            require(block[key] is None, label + f": {key} None")
        else:
            close(block[key], mine[key], label + f": {key}")


def check_ratios(counts, ratios, label: str) -> None:
    require(len(ratios) == len(counts), label + ": ratio length")
    n0 = counts[0]
    for c, r in zip(counts, ratios):
        if n0 == 0:
            require(r is None, label + ": ratio with zero links")
        else:
            close(r, c / n0, label + ": ratio")


def stratified(rows: list, width: int) -> tuple[list[Fraction], list[float]]:
    estimate = [Fraction(0)] * width
    variance = [0.0] * width
    for r in rows:
        Nh, nh = r["population"], r["sample"]
        require(1 <= nh <= Nh, "stratum sample size")
        require(len(r["sums"]) == width and len(r["sums_of_squares"]) == width, "stratum vector width")
        for k in range(width):
            S, Q = r["sums"][k], r["sums_of_squares"][k]
            estimate[k] += Fraction(Nh * S, nh)
            if nh > 1:
                variance[k] += Nh * Nh * (1.0 - nh / Nh) * ((Q - S * S / nh) / (nh - 1)) / nh
    return estimate, [math.sqrt(v) for v in variance]


# --------------------------------------------------------------------------
# Row checks
# --------------------------------------------------------------------------


def check_ordering(row: dict, C: int, N: int, dim: int, label: str) -> None:
    if N < 2:
        require(row["ordering_fraction"] is None and row["myrheim_meyer_dimension"] is None, label + ": degenerate")
        return
    f = Fraction(2 * C, N * (N - 1))
    require(Fraction(row["ordering_fraction"]) == f, label + ": ordering fraction")
    close(row["ordering_fraction_float"], float(f), label + ": ordering fraction float")
    close(row["distance_to_reference"], float(f - REFERENCE_FRACTION[dim]), label + ": distance to reference")
    est = mm_dimension(float(f))
    if est is None:
        require(row["myrheim_meyer_dimension"] is None, label + ": dimension None")
    else:
        close(row["myrheim_meyer_dimension"], est, label + ": dimension")


def check_estimate_block(row: dict, N: int, C: int, label: str) -> None:
    est = row["strict_pair_count_estimate"]
    value, se = stratified(est["strata"], 1)
    require(Fraction(est["estimate"]) == value[0], label + ": stratified estimate")
    close(est["estimate_float"], float(value[0]), label + ": estimate float")
    close(est["standard_error"], se[0], label + ": standard error")
    require(est["sample_size"] == sum(r["sample"] for r in est["strata"]), label + ": sample size")
    require(sum(r["population"] for r in est["strata"]) == row["support_site_count"], label + ": strata census")
    close(est["ordering_fraction_estimate"], float(2 * value[0] / (N * (N - 1))), label + ": fraction estimate")
    close(est["ordering_fraction_standard_error"], 2.0 * se[0] / (N * (N - 1)), label + ": fraction se")
    close(est["relative_deviation_from_exact"], float(value[0]) / C - 1.0, label + ": relative deviation")
    require(est["within_three_standard_errors"] is (abs(float(value[0]) - C) <= 3.0 * se[0]), label + ": three se flag")
    require(abs(float(value[0]) - C) <= 4.0 * se[0], label + ": sampled estimate outside four standard errors")


def check_vertical_row(row: dict, fam: dict, label: str, rebuilt: Rebuilt | None) -> None:
    dim, q = fam["dimension"], fam["q"]
    Kp = row["layers"]
    N = row["inclusive_event_count"]
    require(N == sum(row["counts_by_layer"]) and len(row["counts_by_layer"]) == Kp + 1, label + ": layer counts")
    require(row["pair_counting"] == "exact_all_pairs", label + ": counting mode")
    C = row["strict_pair_count"]
    check_ordering(row, C, N, dim, label)
    if N > 12000:
        require("strict_pair_count_estimate" in row, label + ": missing sampled estimate")
        check_estimate_block(row, N, C, label)
    a_q = 1.0 / math.sqrt(q)
    T = Kp * a_q
    density = q ** (dim + 0.5)
    close(row["model_time_T_over_L"], T, label + ": T")
    close(row["proper_duration_over_L"], T, label + ": tau")
    close(row["normalized_count"], N / density, label + ": normalized count")
    volume = diamond_volume(dim, T)
    close(row["continuum_diamond_volume"], volume, label + ": volume")
    close(row["relative_deviation_from_continuum"], N / density / volume - 1.0, label + ": relative deviation")
    close(row["count_volume_coefficient"], N / density / T ** (dim + 1), label + ": coefficient")
    close(row["count_volume_coefficient_reference"], UNIT_BALL[dim] / (2 ** dim * (dim + 1)), label + ": coefficient ref")
    clearance = row["tip_clearance_over_L"]
    close(row["spatial_buffer_over_L"], clearance - T / 2.0, label + ": buffer")
    require(row["continuum_diamond_inside_cube"] is (clearance >= T / 2.0), label + ": inside flag")
    if dim == 3:
        H = fam["assignment_error_H_over_L"]
        bound = vertical_bound(T, a_q, H, fam["h_over_a"])
        close(row["volume_error_bound"], bound, label + ": bound")
        close(row["volume_error_bound_relative"], bound / volume, label + ": bound relative")
        require(row["bound_hypothesis_ball_inside_cube"] is (clearance >= T / 2.0 + H), label + ": bound hypothesis")
        require(row["actual_deviation_within_bound"] is (abs(N / density - volume) <= bound), label + ": within bound")
    tip = row["tip"]
    site = tip["site"]
    labels = [(site // q ** (dim - 1 - i)) % q for i in range(dim)]
    require(tip["labels"] == labels, label + ": tip labels")
    if rebuilt is not None:
        require(list(tip["rank_offsets_from_centre"]) == [rebuilt.perm[b] - rebuilt.perm[rebuilt.centre_axis]
                                                            for b in labels], label + ": rank offsets")
        for v, w in zip(tip["position_over_L"], rebuilt.positions[site]):
            close(v, w, label + ": tip position")
        close(clearance, float(rebuilt.clearance[site]), label + ": clearance")
        site_ev, layer_ev = rebuilt.events(site, site, Kp)
        require(len(site_ev) == N, label + ": rebuilt event count")
        require([int(np.count_nonzero(layer_ev == j)) for j in range(Kp + 1)] == row["counts_by_layer"],
                label + ": rebuilt layer counts")
        F = rebuilt.future_matrix(site_ev, layer_ev)
        require(int(F.sum()) == C, label + ": rebuilt strict pairs")
        support = int(np.count_nonzero(2 * rebuilt.dist[site] <= Kp))
        require(support == row["support_site_count"], label + ": rebuilt support")


def check_moving_row(row: dict, fam: dict, label: str, rebuilt: Rebuilt | None, centre_rows: dict) -> None:
    dim, q = fam["dimension"], fam["q"]
    Kp = row["layers"]
    N = row["inclusive_event_count"]
    require(N == sum(row["counts_by_layer"]) and len(row["counts_by_layer"]) == Kp + 1, label + ": layer counts")
    C = row["strict_pair_count"]
    check_ordering(row, C, N, dim, label)
    if N > 12000:
        check_estimate_block(row, N, C, label)
    a_q = 1.0 / math.sqrt(q)
    T = Kp * a_q
    density = q ** (dim + 0.5)
    ell2_pair = [Fraction(v) for v in row["tip_separation_squared_over_L2_Qphi"]]
    ell2 = float(ell2_pair[0]) + float(ell2_pair[1]) * PHI
    ell = math.sqrt(ell2)
    tau = math.sqrt(T * T - ell2)
    close(row["tip_separation_over_T"], ell / T, label + ": ell over T")
    close(row["rapidity"], math.atanh(ell / T), label + ": rapidity")
    close(row["model_time_T_over_L"], T, label + ": T")
    close(row["proper_duration_over_L"], tau, label + ": tau")
    volume = diamond_volume(dim, tau)
    close(row["continuum_diamond_volume"], volume, label + ": volume")
    close(row["normalized_count"], N / density, label + ": normalized")
    close(row["relative_deviation_from_continuum"], N / density / volume - 1.0, label + ": relative deviation")
    close(row["count_volume_coefficient"], N / density / tau ** (dim + 1), label + ": coefficient")
    require(row["spatial_buffer_positive"] is (row["spatial_buffer_over_L"] > 0.0), label + ": buffer flag")
    if dim == 3:
        H = fam["assignment_error_H_over_L"]
        require(row["spatial_buffer_at_least_H_q"] is (row["spatial_buffer_over_L"] >= H), label + ": H_q flag")
        bound = general_bound(T, a_q, H, fam["h_over_a"])
        close(row["volume_error_bound"], bound, label + ": bound")
        close(row["volume_error_bound_relative"], bound / volume, label + ": bound relative")
        require(row["actual_deviation_within_bound"] is (abs(N / density - volume) <= bound), label + ": within bound")
    else:
        require(row["spatial_buffer_at_least_H_q"] is False, label + ": H_q flag in a control")
    x, y = row["x_site"], row["y_site"]
    require(x == sum(row["x_axis_label"] * q ** (dim - 1 - i) for i in range(dim)), label + ": x site")
    require(y == sum(row["y_axis_label"] * q ** (dim - 1 - i) for i in range(dim)), label + ": y site")
    against = row["against_vertical"]
    ref = centre_rows[against["reference_layers"]]
    close(against["ordering_fraction_difference"], row["ordering_fraction_float"] - ref["ordering_fraction_float"],
          label + ": fraction difference")
    if row["myrheim_meyer_dimension"] is not None and ref["myrheim_meyer_dimension"] is not None:
        close(against["dimension_difference"], row["myrheim_meyer_dimension"] - ref["myrheim_meyer_dimension"],
              label + ": dimension difference")
    close(against["count_volume_coefficient_ratio"], row["count_volume_coefficient"] / ref["count_volume_coefficient"],
          label + ": coefficient ratio")
    if rebuilt is not None:
        rc = rebuilt.perm[rebuilt.centre_axis]
        xl, yl = row["x_axis_label"], row["y_axis_label"]
        s = row["rank_shift"]
        require(rebuilt.perm[xl] == rc - s and rebuilt.perm[yl] == rc + s, label + ": rank shift")
        # dim * (dm1 + db1 phi)^2 with dm1 = m_y - m_x, db1 = b_y - b_x
        dm1, db1 = int(rebuilt.m[yl] - rebuilt.m[xl]), int(rebuilt.b[yl] - rebuilt.b[xl])
        a_val, b_val = dim * (dm1 * dm1 + db1 * db1), dim * (2 * dm1 * db1 + db1 * db1)
        require(ell2_pair == [Fraction(a_val), Fraction(b_val)], label + ": exact separation")
        site_ev, layer_ev = rebuilt.events(x, y, Kp)
        require(len(site_ev) == N, label + ": rebuilt event count")
        require([int(np.count_nonzero(layer_ev == j)) for j in range(Kp + 1)] == row["counts_by_layer"],
                label + ": rebuilt layer counts")
        F = rebuilt.future_matrix(site_ev, layer_ev)
        require(int(F.sum()) == C, label + ": rebuilt strict pairs")


def check_family(fam: dict, n: int, dim: int, receipt: dict) -> dict:
    q, p = fib(n)
    label = f"n={n} dim={dim}"
    require(fam["dimension"] == dim and fam["fibonacci_index"] == n and fam["q"] == q and fam["p"] == p,
            label + ": header")
    K = ceil_sqrt(q)
    require(fam["layer_steps"] == K and fam["site_count"] == q ** dim, label + ": schedule")
    a_q = 1.0 / math.sqrt(q)
    close(fam["edge_radius_a_over_L"], a_q, label + ": a_q")
    close(fam["model_time_T_over_L"], K * a_q, label + ": T")
    close(fam["event_density_times_L_to_dim_plus_one"], q ** (dim + 0.5), label + ": density")
    if dim == 3:
        close(fam["assignment_error_H_over_L"], 2.0 * math.sqrt(3.0) / q, label + ": H_q")
    else:
        require(fam["assignment_error_H_over_L"] is None, label + ": H_q None")
    rebuilt = Rebuilt(n, dim) if n in REBUILT_LEVELS else None
    if rebuilt is not None:
        require(fam["centre_axis"] == rebuilt.centre_axis and fam["centre_site"] == rebuilt.centre, label + ": centre")
        close(fam["h_over_a"], rebuilt.h_over_a, label + ": h over a")
        require(fam["centre_support_site_count"] == int(np.count_nonzero(2 * rebuilt.dist[rebuilt.centre] <= K)),
                label + ": centre support")

    centre = fam["centre_vertical_intervals"]
    require([r["layers"] for r in centre] == list(range(1, K + 1)), label + ": centre census")
    centre_rows = {}
    for row in centre:
        require(row["tip"]["site"] == fam["centre_site"] and row["tip"]["role"] == "centre", label + ": centre tip")
        check_vertical_row(row, fam, label + f" centre k={row['layers']}", rebuilt)
        centre_rows[row["layers"]] = row

    hom = fam["homogeneity"]
    sizes = [k for k in (K, K - 1, K - 2) if k >= 2]
    require([b["layers"] for b in hom["sizes"]] == sizes, label + ": homogeneity sizes")
    all_dims = []
    for block in hom["sizes"]:
        k = block["layers"]
        rows = block["diamonds"]
        require(block["diamond_count"] == len(rows) >= 1, label + ": diamond count")
        require(packed(rows[0]) == packed(centre_rows[k]), label + f": homogeneity centre row k={k}")
        seen = {rows[0]["tip"]["site"]}
        for i, row in enumerate(rows[1:], start=1):
            require(row["layers"] == k, label + ": homogeneity layers")
            require(row["tip"]["site"] not in seen and row["tip"]["role"].startswith("direction"), label + ": tip role")
            seen.add(row["tip"]["site"])
            T = k * a_q
            require(row["spatial_buffer_over_L"] > 0.0, label + ": off-centre tip buffer")
            if block["tip_rule"] == "buffer_at_least_H_q":
                require(row["spatial_buffer_over_L"] >= fam["assignment_error_H_over_L"], label + ": H_q tip rule")
            check_vertical_row(row, fam, label + f" tip {row['tip']['site']} k={k}", rebuilt)
        if len(rows) > 1:
            require(block["tip_rule"] in ("buffer_at_least_H_q", "buffer_positive"), label + ": tip rule")
        dims = [r["myrheim_meyer_dimension"] for r in rows]
        check_summary(block["dimension_summary"], dims, label + f": dimension summary k={k}")
        check_summary(block["off_centre_dimension_summary"], dims[1:], label + f": off-centre summary k={k}")
        check_summary(block["ordering_fraction_summary"], [r["ordering_fraction_float"] for r in rows],
                      label + f": fraction summary k={k}")
        check_summary(block["counts_summary"], [r["inclusive_event_count"] for r in rows], label + f": counts summary k={k}")
        all_dims += dims
    check_summary(hom["all_sizes_dimension_summary"], all_dims, label + ": all sizes summary")

    boosts = fam["boosts"]
    Kb = boosts["layers"]
    require(2 <= Kb <= K, label + ": boost layers")
    if rebuilt is not None:
        cl = float(rebuilt.clearance[rebuilt.centre])
        require(cl >= Kb * a_q / 2.0 or Kb == 2, label + ": boost layers inside cube")
        require(Kb == K or cl < (Kb + 1) * a_q / 2.0, label + ": boost layers maximal")
    moving_rows = list(boosts["diamonds"])
    for row in moving_rows:
        require(row["layers"] == Kb, label + ": moving layers")
        require(row["against_vertical"]["reference_layers"] == Kb, label + ": moving reference")
        check_moving_row(row, fam, label + f" moving shift {row['rank_shift']}", rebuilt, centre_rows)
    sn = boosts["source_net_moving_diamond"]
    if sn is not None:
        require(sn["layers"] == K and sn["against_vertical"]["reference_layers"] == K, label + ": source-net moving layers")
        check_moving_row(sn, fam, label + f" source-net moving shift {sn['rank_shift']}", rebuilt, centre_rows)
    if dim == 1:
        require(moving_rows == [] and sn is None, label + ": one-dimensional control has no moving diamonds")
    vref = boosts["vertical_reference"]
    require(vref["layers"] == Kb and vref["inclusive_event_count"] == centre_rows[Kb]["inclusive_event_count"],
            label + ": vertical reference")
    check_summary(boosts["ordering_fraction_summary_with_vertical"],
                  [centre_rows[Kb]["ordering_fraction_float"]] + [r["ordering_fraction_float"] for r in moving_rows],
                  label + ": boost fraction summary")
    check_summary(boosts["dimension_summary_with_vertical"],
                  [centre_rows[Kb]["myrheim_meyer_dimension"]] + [r["myrheim_meyer_dimension"] for r in moving_rows],
                  label + ": boost dimension summary")
    check_summary(boosts["count_volume_coefficient_summary_with_vertical"],
                  [centre_rows[Kb]["count_volume_coefficient"]] + [r["count_volume_coefficient"] for r in moving_rows],
                  label + ": boost coefficient summary")

    ab = fam["abundance"]
    N = centre_rows[K]["inclusive_event_count"]
    C = centre_rows[K]["strict_pair_count"]
    require(ab["event_count"] == N and ab["max_m"] == MAX_M, label + ": abundance header")
    if ab["mode"] == "exact_between_count_matrix":
        counts = ab["counts"]
        require(len(counts) == MAX_M + 1 and len(ab["log2_bins_above_max_m"]) == LOG2_BINS, label + ": profile width")
        require(sum(counts) + sum(ab["log2_bins_above_max_m"]) == ab["total_related_pairs"] == C, label + ": profile total")
        check_ratios(counts, ab["ratios"], label + ": abundance ratios")
        close(ab["links_fraction_of_related_pairs"], counts[0] / C, label + ": links fraction")
        for v, c in zip(ab["log2_bins_fraction_of_related_pairs"], ab["log2_bins_above_max_m"]):
            close(v, c / C, label + ": log2 fraction")
        if rebuilt is not None:
            site_ev, layer_ev = rebuilt.events(rebuilt.centre, rebuilt.centre, K)
            hist = profile_from_future(rebuilt.future_matrix(site_ev, layer_ev))
            require(hist[:MAX_M + 1].tolist() == counts and hist[MAX_M + 1:].tolist() == ab["log2_bins_above_max_m"],
                    label + ": rebuilt abundance profile")
    else:
        require(ab["mode"] == "stratified_cluster_sample_of_lower_events" and N > 8000, label + ": abundance mode")
        est, se = stratified(ab["sample"]["strata_rows"], BINS)
        est_f = [float(v) for v in est]
        for v, w in zip(ab["counts_estimate"], est_f[:MAX_M + 1]):
            close(v, w, label + ": counts estimate")
        for v, w in zip(ab["counts_standard_error"], se[:MAX_M + 1]):
            close(v, w, label + ": counts se")
        for v, w in zip(ab["log2_bins_above_max_m_estimate"], est_f[MAX_M + 1:]):
            close(v, w, label + ": log2 estimate")
        total = sum(est_f)
        total_se = math.sqrt(sum(v * v for v in se))
        close(ab["total_related_pairs_estimate"], total, label + ": total estimate")
        close(ab["total_related_pairs_standard_error"], total_se, label + ": total se")
        require(ab["exact_total_related_pairs"] == C, label + ": exact total")
        close(ab["total_relative_deviation_from_exact"], total / C - 1.0, label + ": total deviation")
        require(ab["total_within_three_standard_errors"] is (abs(total - C) <= 3.0 * total_se), label + ": total flag")
        require(abs(total - C) <= 4.0 * total_se, label + ": sampled profile total outside four standard errors")
        require(ab["sample"]["sample_size"] == sum(r["sample"] for r in ab["sample"]["strata_rows"]), label + ": sample size")
        population = sum(r["population"] for r in ab["sample"]["strata_rows"])
        require(population == N - centre_rows[K]["counts_by_layer"][K], label + ": lower-event population")
        check_ratios(ab["counts_estimate"], ab["ratios"], label + ": sampled ratios")
    for st, key in ab["reference_keys"].items():
        ref = receipt["sprinkling_references"][key]
        require(ref["spacetime_dimension"] == int(st) and ref["event_count"] == N, label + ": reference key")
        comparison = ab["comparison"][st]
        require([c["m"] for c in comparison] == list(range(MAX_M + 1)), label + ": comparison census")
        for c in comparison:
            m = c["m"]
            require(c["family"] == ab["ratios"][m] and c["reference_mean"] == ref["ratios_mean"][m]
                    and c["reference_sd"] == ref["ratios_sd"][m], label + ": comparison values")
            if c["family"] is None or c["reference_mean"] is None or not c["reference_sd"]:
                require(c["pull"] is None, label + ": pull None")
            else:
                close(c["pull"], (c["family"] - c["reference_mean"]) / c["reference_sd"], label + ": pull")

    action = fam["action"]
    if dim == 3:
        counts = ab["counts"] if ab["mode"] == "exact_between_count_matrix" else ab["counts_estimate"]
        require(action["N"] == N and action["N_1_links"] == counts[0] and action["N_2"] == int(counts[1])
                and action["N_3"] == int(counts[2]) and action["N_4"] == int(counts[3]), label + ": action inputs")
        bracket = N - int(counts[0]) + 9 * int(counts[1]) - 16 * int(counts[2]) + 8 * int(counts[3])
        require(action["bracket"] == bracket, label + ": action bracket")
        close(action["action"], 4.0 / math.sqrt(6.0) * bracket, label + ": action")
        close(action["action_over_N"], 4.0 / math.sqrt(6.0) * bracket / N, label + ": action over N")
        ref4 = receipt["sprinkling_references"][ab["reference_keys"]["4"]]
        require(packed(action["reference_3_plus_1"]["action"]) == packed(ref4["benincasa_dowker_action"]),
                label + ": action reference")
        require("(4/sqrt 6)" in action["convention"] and "(i+1)-element" in action["convention"], label + ": convention")
    else:
        require(action is None, label + ": action in a control")

    ld = fam["link_directions"]
    if dim == 1:
        require(ld is None, label + ": link directions in one dimension")
    else:
        require(ld["directed_link_count"] == 2 * ld["undirected_spatial_edges"], label + ": link count parity")
        powers = ld["power_relative_to_degree_zero"]
        require(len(powers) == 4, label + ": multipole census")
        uni = ld["uniform_reference"]
        require(uni["direction_count"] == ld["directed_link_count"] and uni["seeds"] >= 8, label + ": uniform reference")
        for l in range(1, 5):
            expected = (2 * l + 1) / ld["directed_link_count"] if dim == 3 else 2.0 / ld["directed_link_count"]
            close(uni["expected_power_uniform"][l], expected, label + f": uniform expectation l={l}")
            require(0.2 * expected <= uni["power_mean"][l] <= 5.0 * expected, label + f": uniform reference level l={l}")
            close(ld["power_over_uniform_mean"][l - 1], powers[l - 1] / uni["power_mean"][l], label + ": power over uniform")
        if dim == 3:
            require(powers[0] < 1e-20 and powers[2] < 1e-20, label + ": odd degrees vanish")
        if rebuilt is not None:
            src, dst = np.nonzero(rebuilt.adjacent & ~np.eye(rebuilt.count, dtype=bool))
            vec = rebuilt.positions[dst] - rebuilt.positions[src]
            unit = vec / np.linalg.norm(vec, axis=1)[:, None]
            mine = sphere_powers(unit) if dim == 3 else circle_powers(unit)
            require(len(src) == ld["directed_link_count"], label + ": rebuilt link count")
            for l in range(1, 5):
                close(powers[l - 1], mine[l], label + f": rebuilt multipole l={l}", tolerance=1e-7)
            disp = np.stack([rebuilt.m[rebuilt.sites[dst, ax]] - rebuilt.m[rebuilt.sites[src, ax]] for ax in range(dim)]
                            + [rebuilt.b[rebuilt.sites[dst, ax]] - rebuilt.b[rebuilt.sites[src, ax]] for ax in range(dim)], axis=1)
            require(len(np.unique(disp, axis=0)) == ld["distinct_displacement_count"], label + ": distinct displacements")
            require(len(np.unique(np.round(unit, 9), axis=0)) == ld["distinct_direction_count"], label + ": distinct directions")

    clocks = fam["count_clocks"]
    root = 1.0 / (dim + 1)
    density = q ** (dim + 0.5)
    nested = clocks["nested_vertical"]
    require([r["interval_layers"] for r in nested] == [k for k in range(K, 0, -1) if k // 2 >= 1], label + ": nested census")
    for r in nested:
        Kp, Kr = r["interval_layers"], r["reference_layers"]
        require(Kr == Kp // 2, label + ": reference layers")
        NI, NJ = centre_rows[Kp]["inclusive_event_count"], centre_rows[Kr]["inclusive_event_count"]
        require(r["interval_count"] == NI and r["reference_count"] == NJ, label + ": clock counts")
        close(r["count_clock_fourth_root"], (NI / NJ) ** 0.25, label + ": fourth root")
        close(r["count_clock_dimension_root"], (NI / NJ) ** root, label + ": dimension root")
        close(r["model_time_ratio"], Kp / Kr, label + ": model ratio")
        close(r["relative_deviation_fourth_root"], (NI / NJ) ** 0.25 / (Kp / Kr) - 1.0, label + ": clock deviation")
        close(r["relative_deviation_dimension_root"], (NI / NJ) ** root / (Kp / Kr) - 1.0, label + ": clock deviation d")
        if dim == 3:
            EI = centre_rows[Kp]["volume_error_bound"] * density
            EJ = centre_rows[Kr]["volume_error_bound"] * density
            close(r["count_error_interval"], EI, label + ": EI")
            close(r["count_error_reference"], EJ, label + ": EJ")
            close(r["enclosure_lower"], (max(NI - EI, 0.0) / (NJ + EJ)) ** 0.25, label + ": enclosure lower")
            if NJ > EJ:
                close(r["enclosure_upper"], ((NI + EI) / (NJ - EJ)) ** 0.25, label + ": enclosure upper")
            else:
                require(r["enclosure_upper"] is None, label + ": enclosure upper None")
            require(r["enclosure_hypotheses_satisfied"] is (
                centre_rows[Kp]["bound_hypothesis_ball_inside_cube"] and centre_rows[Kr]["bound_hypothesis_ball_inside_cube"]
                and NJ > EJ), label + ": enclosure hypotheses")
    all_moving = moving_rows + ([sn] if sn is not None else [])
    require(len(clocks["moving_against_vertical"]) == len(all_moving), label + ": moving clock census")
    for r, M in zip(clocks["moving_against_vertical"], all_moving):
        ref = centre_rows[M["against_vertical"]["reference_layers"]]
        NI, NJ = M["inclusive_event_count"], ref["inclusive_event_count"]
        require(r["rank_shift"] == M["rank_shift"] and r["layers"] == M["layers"] and r["interval_count"] == NI
                and r["reference_count"] == NJ and r["reference_layers"] == ref["layers"], label + ": moving clock row")
        ratio = M["proper_duration_over_L"] / ref["model_time_T_over_L"]
        close(r["proper_duration_ratio"], ratio, label + ": proper ratio")
        close(r["count_clock_fourth_root"], (NI / NJ) ** 0.25, label + ": moving fourth root")
        close(r["relative_deviation_fourth_root"], (NI / NJ) ** 0.25 / ratio - 1.0, label + ": moving deviation")
        if dim == 3:
            EI = M["volume_error_bound"] * density
            EJ = ref["volume_error_bound"] * density
            close(r["count_error_interval"], EI, label + ": moving EI")
            close(r["enclosure_lower"], (max(NI - EI, 0.0) / (NJ + EJ)) ** 0.25, label + ": moving enclosure lower")
            require(r["enclosure_hypotheses_satisfied"] is (M["spatial_buffer_positive"]
                                                            and ref["bound_hypothesis_ball_inside_cube"] and NJ > EJ),
                    label + ": moving enclosure hypotheses")

    cross = fam["source_net_cross_check"]
    require(cross is not None and cross["all_agree"] is True, label + ": source-net cross-check")
    require(all(v for e in cross["fields"].values() for v in e.values()), label + ": cross-check fields")
    return {"n": n, "q": q, "dimension": dim, "rebuilt": rebuilt is not None,
            "centre_dimension": centre_rows[K]["myrheim_meyer_dimension"],
            "diamonds_checked": len(all_dims) + len(all_moving)}


def check_reference(key: str, ref: dict, label: str) -> None:
    st = ref["spacetime_dimension"]
    N = ref["event_count"]
    require(key == f"spacetime_{st}_events_{N}", label + ": key")
    require(ref["seeds"] >= 8 and len(ref["per_seed"]) == ref["seeds"], label + ": seeds")
    dim = st - 1
    ratios = []
    fractions, dims, actions, actions_over_n, links = [], [], [], [], []
    log2 = []
    for k, row in enumerate(ref["per_seed"]):
        require(row["seed"] == ref["seed_base"] + k, label + ": seed sequence")
        C = row["strict_pair_count"]
        check_ordering(row, C, N, dim, label + f" seed {k}")
        prof = row["profile"]
        if prof["mode"] == "exact_between_count_matrix":
            counts = prof["counts"]
            require(sum(counts) + sum(prof["log2_bins_above_max_m"]) == prof["total_related_pairs"] == C,
                    label + ": reference profile total")
            total = C
            bins_above = prof["log2_bins_above_max_m"]
        else:
            require(prof["mode"] == "stratified_cluster_sample_of_lower_points" and N > 8000, label + ": reference mode")
            est, se = stratified(prof["sample"]["strata_rows"], BINS)
            est_f = [float(v) for v in est]
            counts = prof["counts_estimate"]
            for v, w in zip(counts, est_f[:MAX_M + 1]):
                close(v, w, label + ": reference counts estimate")
            for v, w in zip(prof["counts_standard_error"], se[:MAX_M + 1]):
                close(v, w, label + ": reference counts se")
            total = sum(est_f)
            close(prof["total_related_pairs_estimate"], total, label + ": reference total")
            require(prof["exact_total_related_pairs"] == C, label + ": reference exact total")
            total_se = math.sqrt(sum(v * v for v in se))
            require(prof["total_within_three_standard_errors"] is (abs(total - C) <= 3.0 * total_se),
                    label + ": reference total flag")
            require(abs(total - C) <= 4.0 * total_se, label + ": reference total outside four standard errors")
            require(sum(r["population"] for r in prof["sample"]["strata_rows"]) == N, label + ": reference strata census")
            bins_above = prof["log2_bins_above_max_m_estimate"]
        check_ratios(counts, row["ratios"], label + ": reference ratios")
        close(row["links_fraction_of_related_pairs"], counts[0] / total, label + ": reference links fraction")
        for v, c in zip(row["log2_bins_fraction_of_related_pairs"], bins_above):
            close(v, c / total, label + ": reference log2 fraction")
        ratios.append(row["ratios"])
        log2.append(row["log2_bins_fraction_of_related_pairs"])
        fractions.append(row["ordering_fraction_float"])
        dims.append(row["myrheim_meyer_dimension"])
        links.append(row["links_fraction_of_related_pairs"])
        if dim == 3:
            bd = row["benincasa_dowker"]
            close(bd["action"], bd_action(N, counts), label + ": reference action")
            close(bd["action_over_N"], bd_action(N, counts) / N, label + ": reference action over N")
            actions.append(bd["action"])
            actions_over_n.append(bd["action_over_N"])
    arr = np.array(ratios, dtype=float)
    for m in range(MAX_M + 1):
        close(ref["ratios_mean"][m], float(arr[:, m].mean()), label + f": ratios mean m={m}")
        close(ref["ratios_sd"][m], float(arr[:, m].std(ddof=1)), label + f": ratios sd m={m}")
    for k, v in enumerate(np.mean(np.array(log2, dtype=float), axis=0)):
        close(ref["log2_bins_fraction_mean"][k], float(v), label + ": log2 mean")
    check_summary(ref["ordering_fraction"], fractions, label + ": reference fraction summary")
    check_summary(ref["myrheim_meyer_dimension"], dims, label + ": reference dimension summary")
    check_summary(ref["links_fraction_of_related_pairs"], links, label + ": reference links summary")
    if dim == 3:
        check_summary(ref["benincasa_dowker_action"], actions, label + ": reference action summary")
        check_summary(ref["benincasa_dowker_action_over_N"], actions_over_n, label + ": reference action/N summary")
    require(abs(ref["myrheim_meyer_dimension"]["mean"] - st) < (1.0 if N < 100 else 0.5),
            label + ": reference dimension near its spacetime dimension")


def check_trend(receipt: dict) -> None:
    trend = receipt["trend_table"]
    rows = trend["three_dimensional_family"]
    require([r["q"] for r in rows] == [lv["q"] for lv in receipt["levels"]], "trend census")
    previous = None
    for lv, row in zip(receipt["levels"], rows):
        fam = next(f for f in lv["families"] if f["dimension"] == 3)
        K = fam["layer_steps"]
        top = fam["centre_vertical_intervals"][K - 1]
        require(row["layers"] == K and row["centre_event_count"] == top["inclusive_event_count"], "trend counts")
        require(row["centre_ordering_fraction"] == top["ordering_fraction_float"]
                and row["centre_dimension"] == top["myrheim_meyer_dimension"], "trend fraction")
        dev = abs(top["ordering_fraction_float"] - 0.1)
        close(row["absolute_deviation_from_one_tenth"], dev, "trend deviation")
        if previous is None:
            require(row["deviation_ratio_to_previous_level"] is None, "trend first ratio")
        else:
            close(row["deviation_ratio_to_previous_level"], dev / previous, "trend ratio")
        previous = dev
        for b in fam["homogeneity"]["sizes"]:
            h = row["homogeneity_dimension_by_size"][str(b["layers"])]
            require(h["mean"] == b["dimension_summary"]["mean"] and h["sd"] == b["dimension_summary"]["sd"], "trend homogeneity")
        require(row["boost_layers"] == fam["boosts"]["layers"], "trend boost layers")
        require(packed(row["boost_dimension_with_vertical"]) == packed(fam["boosts"]["dimension_summary_with_vertical"]),
                "trend boosts")
        require(row["abundance_ratios_family_m1_m2_m3"] == fam["abundance"]["ratios"][1:4], "trend abundance")
        ref4 = receipt["sprinkling_references"][fam["abundance"]["reference_keys"]["4"]]
        require(row["abundance_ratios_sprinkling_3_plus_1_m1_m2_m3"] == ref4["ratios_mean"][1:4], "trend reference")
        require(row["action_over_N_family"] == fam["action"]["action_over_N"], "trend action")
        require(row["link_multipole_power_l1_to_l4"] == fam["link_directions"]["power_relative_to_degree_zero"],
                "trend multipoles")
        require(row["distinct_link_directions"] == fam["link_directions"]["distinct_direction_count"], "trend directions")
        require(row["nested_clock_relative_deviation"]
                == fam["count_clocks"]["nested_vertical"][0]["relative_deviation_fourth_root"], "trend clock")
        check_summary(row["moving_clock_relative_deviation"],
                      [r["relative_deviation_fourth_root"] for r in fam["count_clocks"]["moving_against_vertical"]],
                      "trend moving clock")
    require("do not demonstrate" in trend["statement"], "trend statement")


def verify(receipt: dict) -> dict:
    require(type(receipt) is dict, "receipt object")
    require(receipt.get("schema") == SCHEMA, "schema")
    require(receipt.get("scope") == EXPECTED_SCOPE, "scope block")
    require(type(receipt.get("claim_boundary")) is str and len(receipt["claim_boundary"]) > 200, "claim boundary")
    rows_cert = receipt["certificate_rows"]
    require(set(rows_cert) == set(CERTIFICATE_STATUS), "certificate row census")
    for key, status in CERTIFICATE_STATUS.items():
        require(rows_cert[key]["status"] == status, "certificate status " + key)
        require(type(rows_cert[key]["items"]) is list and type(rows_cert[key]["note"]) is str, "certificate row " + key)
    pins = receipt["source_pins"]
    require(set(pins) == set(PINS), "pin census")
    for rel in PINS:
        require(pins[rel] == hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), "pin " + rel)
    require(receipt["reference_fractions"] == {"4": "1/10", "3": "8/35", "2": "1/2"}, "reference fractions")
    conv = receipt["conventions"]
    require(conv["abundance_profile"]["max_m"] == MAX_M and "(i+1)-element" in conv["benincasa_dowker"], "conventions")
    levels = receipt["levels"]
    require([lv["fibonacci_index"] for lv in levels] == list(LEVELS), "level census")
    results = []
    for lv in levels:
        n = lv["fibonacci_index"]
        require(lv["q"] == fib(n)[0], "level q")
        require([f["dimension"] for f in lv["families"]] == list(DIMENSIONS), "family census")
        for dim, fam in zip(DIMENSIONS, lv["families"]):
            results.append(check_family(fam, n, dim, receipt))
    for key, ref in receipt["sprinkling_references"].items():
        check_reference(key, ref, "reference " + key)
    used = {k for lv in levels for f in lv["families"] for k in f["abundance"]["reference_keys"].values()}
    require(used == set(receipt["sprinkling_references"]), "reference census")
    check_trend(receipt)
    return {"accepted": True, "schema": SCHEMA, "rebuilt_levels": list(REBUILT_LEVELS), "families": results,
            "receipt_sha256": hashlib.sha256(packed(receipt)).hexdigest()}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(verify(load(args.receipt)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
