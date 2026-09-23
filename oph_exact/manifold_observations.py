"""Observations of the causal manifold from inside the source-net family (lane A-obs).

An observer inside the layered source-record family of
``oph_exact/source_net.py`` has access to two things: the event order
``(j, s) <= (j', t)`` iff ``d(s, t) <= j' - j`` (graph distance on the exact
site graph, waiting included) and event cardinalities.  This lane records
what such an observer reads from order and counts alone, per Fibonacci level
``q`` in ``{8, 13, 21, 34}`` for the three-dimensional family and for the
two- and one-dimensional golden controls:

1. dimension homogeneity: the inverted Myrheim-Meyer dimension on interior
   diamonds of several sizes (``K' = K, K-1, K-2`` layers) and locations
   (centre and off-centre tips admitted by the buffer rule);
2. boost covariance: ordering fraction, dimension and count law on
   moving-tip diamonds of several rapidities against the vertical diamond;
3. the interval abundance profile ``N_m/N_0`` (``N_m`` = number of pairs
   ``x < y`` with exactly ``m`` events strictly between, ``N_0`` = links)
   against fixed-count sprinklings of flat diamonds in ``3+1`` and lower
   dimensions;
4. the Benincasa-Dowker action on the family diamond and on the sprinkling
   reference at matched event count;
5. isotropy of link directions: multipole power of the empirical link
   direction distribution on the source ``S^2`` against a uniform reference;
6. count clocks ``(|I|/|J|)^(1/4)`` for nested vertical diamonds and for
   vertical against moving-tip diamonds, with the finite enclosure of the
   theory where its hypotheses hold;
7. a refinement trend table over ``q``;
8. a map from each observable to the source-causal continuum-certificate row
   it addresses.

Pair counts are exact whenever the interval's support has at most
``EXACT_SUPPORT_LIMIT`` sites (every diamond at ``q <= 34``); intervals with
more than ``SAMPLED_EVENT_THRESHOLD`` events additionally carry a stratified
per-start estimate with a declared seed, so the sampled instrument is
calibrated against the exact count.  The population, the read law, the
layer duration and one event per site and layer are supplied, as in the
theory.  Sprinklings are imported controls.  No physical clock or spacetime
is identified; finite runs do not demonstrate the asymptotic limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
import time
from fractions import Fraction
from math import atanh, pi, sqrt
from pathlib import Path

import numpy as np

from oph_exact.source_net import (
    bfs,
    build_site_graph,
    canonical,
    ceil_sqrt,
    cone_masks,
    diamond_volume,
    digest,
    ellipsoid_buffer,
    fibonacci,
    file_sha256,
    general_bound,
    metric_tables,
    moving_weights,
    orbit,
    phi_encode,
    phi_float,
    phi_scale,
    phi_sign,
    phi_square,
    phi_sub,
    rounded,
    vertical_bound,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/exact/manifold_observations_receipt.json"
SOURCE_NET_RECEIPT = ROOT / "data/exact/source_net_causal_limit_receipt.json"
SCHEMA = "oph.exact.manifold-observations.v1"
LEVELS = (6, 7, 8, 9)
OPTIONAL_LEVEL = 10
DIMENSIONS = (3, 2, 1)
EXACT_SUPPORT_LIMIT = 20000
SAMPLED_EVENT_THRESHOLD = 12000
EXACT_PROFILE_LIMIT = 8000
START_SAMPLE_SIZE = 2000
PROFILE_SAMPLE_SIZE = 240
SPRINKLING_PROFILE_SAMPLE_SIZE = 120
SPRINKLING_SEEDS = 8
DIRECTION_SEEDS = 8
SAMPLE_SEED = 20260909
POOL_MINIMUM_SITES = 4000
BATCH_SIZE = 48
PROFILE_MAX_M = 15
PROFILE_LOG2_BINS = 16
PROFILE_BINS = PROFILE_MAX_M + 1 + PROFILE_LOG2_BINS
MAX_MULTIPOLE = 4
RAPIDITY_TARGETS = (0.2, 0.4, 0.6, 0.8)
REFERENCE_SPACETIME_DIMENSIONS = {3: (4, 3), 2: (3, 4), 1: (2, 3)}
REDUCED_DIRECTIONS_FROM_Q = 34
PINS = (
    "oph_exact/manifold_observations.py",
    "oph_exact/verify_manifold_observations_independent.py",
    "tests/test_exact_manifold_observations.py",
    "oph_exact/source_net.py",
    "data/exact/source_net_causal_limit_receipt.json",
)
SCOPE = {
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
CLAIM_BOUNDARY = (
    "Finite readouts of the declared source-record family (q^3 golden sites on the "
    "rank-three source Gram metric, complete-neighbour reads inside a_q = L/sqrt(q), "
    "one event per site and layer, layer duration a_q/c) at q <= 34, obtained from the "
    "event order and event cardinalities alone: dimension readouts on interior diamonds "
    "of several sizes and locations, moving-tip diamonds of several rapidities, interval "
    "abundance profiles, the Benincasa-Dowker action, link-direction multipoles and count "
    "clocks. Fixed-count sprinklings of flat diamonds are imported comparison controls. "
    "The readouts are finite diagnostics of the supplied law; they do not select the "
    "population or the read law from native repairs, identify a physical clock or "
    "spacetime, demonstrate the asymptotic limit of the theory's propositions, or "
    "compute the Hawking-King-McCarthy and Malament uniqueness statements, which are "
    "declared citations."
)
BD_CONVENTION = (
    "S4(C) = (4/sqrt 6) (N - N_1 + 9 N_2 - 16 N_3 + 8 N_4) with N the number of events "
    "and N_i the number of (i+1)-element inclusive order intervals (N_1 = links); in "
    "terms of the between-count profile of this receipt N_i = profile[i-1]. Benincasa "
    "and Dowker, Phys. Rev. Lett. 104, 181301 (2010), eq. (14) states the bracket up to "
    "the order-one prefactor 4/sqrt 6 of their operator B, eq. (2); layer L_i of B has "
    "n(x,y) = i-1 elements between."
)
CERTIFICATE_ROWS = {
    "C1_physical_event_interpretation": {
        "status": "not addressed",
        "items": [],
        "note": "The physical-event interpretation of retained commits and the adequacy argument "
                "are outside the readouts of this lane.",
    },
    "C2_dense_isotropic_link_directions": {
        "status": "computed",
        "items": ["link_directions"],
        "note": "Multipole power of the empirical link directions on the source S^2 per q against a "
                "uniform reference of the same cardinality; the number of distinct directions per q.",
    },
    "C3_order_preserving_refinement": {
        "status": "computed",
        "items": ["trend_table"],
        "note": "Nested Fibonacci levels q = 8, 13, 21, 34 of one construction with the per-q trend "
                "of every observable; the convergence rate is reported as measured, no limit is "
                "demonstrated.",
    },
    "C4_calibrated_density_law": {
        "status": "computed",
        "items": ["count_clocks", "boosts"],
        "note": "Count clocks against model-time and proper-duration ratios, and the count-volume "
                "coefficient N/(rho tau^4) of vertical and moving-tip diamonds; the count-volume law "
                "itself is the source-net receipt's statistic.",
    },
    "C5_independent_dimension_and_topology_tests": {
        "status": "computed",
        "items": ["homogeneity", "abundance", "action"],
        "note": "Myrheim-Meyer dimension on diamonds of several sizes and locations, interval "
                "abundance profiles against fixed-count sprinklings, and the Benincasa-Dowker action; "
                "thickened-antichain topology or homology is work in progress.",
    },
    "C6_distinguishing_lorentzian_geometry_with_uniqueness_control": {
        "status": "declared",
        "items": [],
        "note": "Hawking, King and McCarthy (1976) and Malament (1977) are declared citations for the "
                "uniqueness control of a distinguishing limit; nothing of that row is computed here.",
    },
}
REFERENCE_FRACTION = {1: Fraction(1, 2), 2: Fraction(8, 35), 3: Fraction(1, 10)}


# --------------------------------------------------------------------------
# Myrheim-Meyer dimension (same bisection as the source-net lane)
# --------------------------------------------------------------------------


def mm_fraction(dimension: float) -> float:
    """Expected ordering fraction of a flat Alexandrov interval in ``dimension`` spacetime dimensions."""
    return math.exp(math.lgamma(dimension + 1.0) + math.lgamma(dimension / 2.0)
                    - math.log(2.0) - math.lgamma(1.5 * dimension))


def mm_dimension(fraction: float):
    if not 0.0 < fraction < 1.0:
        return None
    low, high = 1.01, 20.0
    if not (mm_fraction(high) <= fraction <= mm_fraction(low)):
        return None
    for _ in range(96):
        mid = (low + high) / 2.0
        if mm_fraction(mid) > fraction:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def ordering_block(C: Fraction, N: int, dim: int) -> dict:
    if N < 2:
        return {"ordering_fraction": None, "ordering_fraction_float": None,
                "distance_to_reference": None, "myrheim_meyer_dimension": None}
    f = Fraction(2) * C / (N * (N - 1))
    est = mm_dimension(float(f))
    return {"ordering_fraction": str(f), "ordering_fraction_float": rounded(float(f)),
            "distance_to_reference": rounded(float(f - REFERENCE_FRACTION[dim])),
            "myrheim_meyer_dimension": None if est is None else rounded(est)}


# --------------------------------------------------------------------------
# Real spherical harmonics (orthonormal for the uniform probability measure)
# --------------------------------------------------------------------------


def sphere_harmonics(n: np.ndarray) -> list[np.ndarray]:
    """Real harmonics up to degree 4 on unit vectors, scaled by sqrt(4 pi).

    Returned as a list over l = 0..4 of arrays (2l+1, len(n)); the functions
    are orthonormal for the uniform probability measure on S^2, so the l = 0
    entry is the constant 1 and the empirical power in degree l is
    ``sum_m mean(Y_lm)^2``.
    """
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    x2, y2, z2 = x * x, y * y, z * z
    s = math.sqrt(4.0 * pi)
    l0 = [np.ones_like(x) * (0.5 / math.sqrt(pi))]
    c1 = 0.5 * math.sqrt(3.0 / pi)
    l1 = [c1 * y, c1 * z, c1 * x]
    l2 = [0.5 * math.sqrt(15.0 / pi) * x * y, 0.5 * math.sqrt(15.0 / pi) * y * z,
          0.25 * math.sqrt(5.0 / pi) * (3.0 * z2 - 1.0), 0.5 * math.sqrt(15.0 / pi) * x * z,
          0.25 * math.sqrt(15.0 / pi) * (x2 - y2)]
    l3 = [0.25 * math.sqrt(35.0 / (2.0 * pi)) * (3.0 * x2 - y2) * y,
          0.5 * math.sqrt(105.0 / pi) * x * y * z,
          0.25 * math.sqrt(21.0 / (2.0 * pi)) * y * (5.0 * z2 - 1.0),
          0.25 * math.sqrt(7.0 / pi) * (5.0 * z2 - 3.0) * z,
          0.25 * math.sqrt(21.0 / (2.0 * pi)) * x * (5.0 * z2 - 1.0),
          0.25 * math.sqrt(105.0 / pi) * (x2 - y2) * z,
          0.25 * math.sqrt(35.0 / (2.0 * pi)) * (x2 - 3.0 * y2) * x]
    l4 = [0.75 * math.sqrt(35.0 / pi) * x * y * (x2 - y2),
          0.75 * math.sqrt(35.0 / (2.0 * pi)) * (3.0 * x2 - y2) * y * z,
          0.75 * math.sqrt(5.0 / pi) * x * y * (7.0 * z2 - 1.0),
          0.75 * math.sqrt(5.0 / (2.0 * pi)) * y * z * (7.0 * z2 - 3.0),
          (3.0 / 16.0) * math.sqrt(1.0 / pi) * (35.0 * z2 * z2 - 30.0 * z2 + 3.0),
          0.75 * math.sqrt(5.0 / (2.0 * pi)) * x * z * (7.0 * z2 - 3.0),
          (3.0 / 8.0) * math.sqrt(5.0 / pi) * (x2 - y2) * (7.0 * z2 - 1.0),
          0.75 * math.sqrt(35.0 / (2.0 * pi)) * (x2 - 3.0 * y2) * x * z,
          (3.0 / 16.0) * math.sqrt(35.0 / pi) * (x2 * (x2 - 3.0 * y2) - y2 * (3.0 * x2 - y2))]
    return [np.stack(block) * s for block in (l0, l1, l2, l3, l4)]


def circle_harmonics(theta: np.ndarray) -> list[np.ndarray]:
    """Orthonormal Fourier modes on S^1 for the uniform probability measure, m = 0..4."""
    blocks = [np.ones((1, len(theta)))]
    for m in range(1, MAX_MULTIPOLE + 1):
        blocks.append(np.stack([math.sqrt(2.0) * np.cos(m * theta), math.sqrt(2.0) * np.sin(m * theta)]))
    return blocks


class MultipoleAccumulator:
    """Streams direction batches into the sums of the orthonormal harmonics."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.count = 0
        self.sums = None

    def add(self, vectors: np.ndarray) -> None:
        if len(vectors) == 0:
            return
        norms = np.linalg.norm(vectors, axis=1)
        keep = norms > 0.0
        unit = vectors[keep] / norms[keep, None]
        if self.dim == 3:
            blocks = sphere_harmonics(unit)
        else:
            blocks = circle_harmonics(np.arctan2(unit[:, 1], unit[:, 0]))
        sums = [b.sum(axis=1) for b in blocks]
        if self.sums is None:
            self.sums = sums
        else:
            self.sums = [a + b for a, b in zip(self.sums, sums)]
        self.count += int(keep.sum())

    def powers(self) -> list[float]:
        """Power per degree l = 0..4 relative to the constant mode (which is 1)."""
        if self.count == 0:
            return [None] * (MAX_MULTIPOLE + 1)
        return [float(np.sum((s / self.count) ** 2)) for s in self.sums]


def multipole_powers(vectors: np.ndarray, dim: int = 3) -> list[float]:
    acc = MultipoleAccumulator(dim)
    acc.add(np.asarray(vectors, dtype=float))
    return acc.powers()


def uniform_direction_reference(count: int, dim: int, seeds: int = DIRECTION_SEEDS,
                                seed_base: int = SAMPLE_SEED) -> dict:
    """Mean and standard deviation of the powers of ``count`` uniform directions over ``seeds`` seeds."""
    rows = []
    for k in range(seeds):
        rng = np.random.default_rng(seed_base + 7000 + 100 * dim + k)
        acc = MultipoleAccumulator(dim)
        done = 0
        while done < count:
            take = min(count - done, 2_000_000)
            acc.add(rng.standard_normal((take, dim)))
            done += take
        rows.append(acc.powers())
    arr = np.array(rows, dtype=float)
    return {"seeds": seeds, "seed_base": seed_base + 7000 + 100 * dim, "direction_count": count,
            "power_mean": [rounded(v) for v in arr.mean(axis=0)],
            "power_sd": [rounded(v) for v in arr.std(axis=0, ddof=1)] if seeds > 1 else None,
            "expected_power_uniform": [1.0] + [rounded((2 * l + 1) / count if dim == 3 else 2.0 / count)
                                               for l in range(1, MAX_MULTIPOLE + 1)]}


# --------------------------------------------------------------------------
# Fixed-count sprinkling of a flat diamond (imported control)
# --------------------------------------------------------------------------


def sprinkle_diamond(count: int, dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``count`` points uniform in the flat diamond of unit duration in ``dim + 1`` dimensions.

    Tips at (t, x) = (0, 0) and (1, 0); the section at time t is the ball of
    radius min(t, 1 - t).  Times are drawn from the section-volume law
    (cumulative probability proportional to r^(dim+1) on each half), positions
    uniformly in the section ball (uniform direction, radius U^(1/dim)).
    """
    rng = np.random.default_rng(seed)
    half = rng.random(count) < 0.5
    r = 0.5 * rng.random(count) ** (1.0 / (dim + 1))
    t = np.where(half, r, 1.0 - r)
    direction = rng.standard_normal((count, dim))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    radius = r * rng.random(count) ** (1.0 / dim)
    x = direction * radius[:, None]
    order = np.argsort(t, kind="stable")
    return t[order], x[order]


def sprinkling_future_rows(t: np.ndarray, x: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Boolean block F[rows, :] of the strict causal relation of a sprinkling."""
    dt = t[None, :] - t[rows, None]
    dx = x[None, :, :] - x[rows, None, :]
    return dt > np.sqrt(np.einsum("ijk,ijk->ij", dx, dx))


# --------------------------------------------------------------------------
# Interval abundance profiles and the Benincasa-Dowker action
# --------------------------------------------------------------------------


def profile_bins(m: np.ndarray) -> np.ndarray:
    """Bin index of a between count: ``m`` itself for ``m <= 15``, then log2 bins ``[2^k, 2^(k+1))``, ``k >= 4``."""
    m = np.asarray(m, dtype=np.int64)
    large = np.maximum(m, 16)
    k = np.floor(np.log2(large.astype(np.float64))).astype(np.int64) - 4
    return np.where(m <= PROFILE_MAX_M, m, PROFILE_MAX_M + 1 + np.minimum(k, PROFILE_LOG2_BINS - 1))


def profile_histogram(m: np.ndarray) -> np.ndarray:
    return np.bincount(profile_bins(m), minlength=PROFILE_BINS)


LOG2_BIN_LABELS = [f"[{2 ** (k + 4)},{2 ** (k + 5)})" for k in range(PROFILE_LOG2_BINS - 1)] + [f"[{2 ** (PROFILE_LOG2_BINS + 3)},inf)"]


def between_profile_from_future(F: np.ndarray) -> dict:
    """Between-count profile of a strict future matrix ``F`` (bool, N x N).

    ``N_m`` is the number of pairs ``x < y`` with exactly ``m`` events strictly
    between; the between count is ``(F @ F)[x, y]``, computed in row blocks in
    float32 (exact below 2^24).  Returns the counts for ``m = 0..15``, the
    log2-binned counts above 15 and the total number of related pairs.
    """
    N = F.shape[0]
    Ff = F.astype(np.float32)
    counts = np.zeros(PROFILE_BINS, dtype=np.int64)
    block = max(1, min(N, 2048))
    for lo in range(0, N, block):
        prod = Ff[lo:lo + block] @ Ff
        related = F[lo:lo + block]
        counts += profile_histogram(np.rint(prod[related]).astype(np.int64))
    return {"max_m": PROFILE_MAX_M, "counts": counts[:PROFILE_MAX_M + 1].tolist(),
            "log2_bins_above_max_m": counts[PROFILE_MAX_M + 1:].tolist(),
            "total_related_pairs": int(counts.sum())}


def profile_ratios(counts) -> list:
    n0 = counts[0]
    return [None if n0 == 0 else rounded(c / n0) for c in counts]


def benincasa_dowker_action(N: int, counts) -> dict:
    """Four-dimensional Benincasa-Dowker action, see ``BD_CONVENTION``.

    ``counts[i-1]`` is the number of pairs with ``i-1`` events between, i.e.
    the number ``N_i`` of ``(i+1)``-element inclusive order intervals.
    """
    N1, N2, N3, N4 = (int(counts[i]) for i in range(4))
    bracket = N - N1 + 9 * N2 - 16 * N3 + 8 * N4
    action = 4.0 / sqrt(6.0) * bracket
    return {"N": int(N), "N_1_links": N1, "N_2": N2, "N_3": N3, "N_4": N4,
            "bracket": int(bracket), "action": rounded(action),
            "action_over_N": rounded(action / N) if N else None}


def event_table(a: np.ndarray, g: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Events ``(j, s)`` of the diamond: support index and layer, in support order then layer order."""
    a = np.asarray(a, dtype=np.int64)
    g = np.asarray(g, dtype=np.int64)
    lengths = K - a - g + 1
    if np.any(lengths < 1):
        raise ValueError("event table needs support sites with alpha + gamma <= K")
    site = np.repeat(np.arange(len(a)), lengths)
    starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
    layer = np.arange(len(site)) - starts + np.repeat(a, lengths)
    return site, layer


def strict_future_matrix(D: np.ndarray, site: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """``F[x, y]`` iff ``layer_y > layer_x`` and ``D[site_x, site_y] <= layer_y - layer_x``."""
    N = len(site)
    F = np.zeros((N, N), dtype=bool)
    layer16 = layer.astype(np.int16)
    block = max(1, min(N, 1024))
    for lo in range(0, N, block):
        dl = layer16[None, :] - layer16[lo:lo + block, None]
        Dxy = D[site[lo:lo + block][:, None], site[None, :]].astype(np.int16)
        F[lo:lo + block] = (dl > 0) & (Dxy <= dl)
    return F


def exact_profile(D: np.ndarray, a: np.ndarray, g: np.ndarray, K: int) -> dict:
    site, layer = event_table(a, g, K)
    F = strict_future_matrix(D, site, layer)
    out = between_profile_from_future(F)
    out["event_count"] = int(len(site))
    return out


def between_counts_from_lower_event(D: np.ndarray, gamma: np.ndarray, K: int, j: int, i: int) -> np.ndarray:
    """Histogram over ``m`` of the upper partners of the lower event ``(j, support[i])``.

    For an upper event ``(j', t)`` the number of events strictly between is
    ``sum_u max(0, min(j'-1, j'-D[t,u]) - max(j+1, j+D[s,u]) + 1)``; the sum
    runs over the sub-diamond support ``{u : D[s,u] + gamma_u <= K - j}``,
    outside of which every term vanishes.  Bins follow ``profile_bins``.
    """
    ds = D[i].astype(np.int16)
    sub = np.flatnonzero(ds + gamma <= K - j)
    ds_sub = ds[sub]
    g_sub = gamma[sub].astype(np.int16)
    lo_u = np.maximum(j + 1, j + ds_sub).astype(np.int16)
    hist = np.zeros(PROFILE_BINS, dtype=np.int64)
    for jp in range(j + 1, K + 1):
        valid = (jp >= j + ds_sub) & (jp <= K - g_sub)
        ts = sub[valid]
        if len(ts) == 0:
            continue
        chunk = max(1, min(len(ts), 20_000_000 // max(1, len(sub))))
        for lo in range(0, len(ts), chunk):
            Dtu = D[ts[lo:lo + chunk]][:, sub].astype(np.int16)
            hi = np.minimum(jp - 1, jp - Dtu)
            cnt = np.maximum(0, hi - lo_u[None, :] + 1).sum(axis=1)
            hist += profile_histogram(cnt)
    return hist


def sprinkling_between_counts(t: np.ndarray, x: np.ndarray, i: int) -> np.ndarray:
    """Histogram over ``m`` of the future partners of point ``i`` of a sprinkling."""
    future = np.flatnonzero(sprinkling_future_rows(t, x, np.array([i]))[0])
    if len(future) == 0:
        return np.zeros(PROFILE_BINS, dtype=np.int64)
    tf, xf = t[future], x[future]
    counts = np.zeros(len(future), dtype=np.int64)
    chunk = max(1, min(len(future), 4_000_000 // max(1, len(future))))
    for lo in range(0, len(future), chunk):
        dt = tf[None, :] - tf[lo:lo + chunk, None]
        dx = xf[None, :, :] - xf[lo:lo + chunk, None, :]
        counts += (dt > np.sqrt(np.einsum("ijk,ijk->ij", dx, dx))).sum(axis=0)
    return profile_histogram(counts)


# --------------------------------------------------------------------------
# Stratified sampling without replacement (Horvitz-Thompson totals)
# --------------------------------------------------------------------------


def allocate(strata: dict, size: int, seed: int, minimum: int = 1) -> dict:
    """Proportional allocation over sorted strata, at least ``minimum`` members each, seeded."""
    rng = np.random.default_rng(seed)
    total = sum(len(v) for v in strata.values())
    chosen = {}
    for key in sorted(strata):
        members = np.sort(np.asarray(strata[key], dtype=np.int64))
        want = max(min(len(members), minimum), -(-size * len(members) // total))
        want = min(len(members), want)
        chosen[key] = np.sort(rng.choice(members, size=want, replace=False))
    return chosen


def stratified_totals(strata: dict, chosen: dict, values: dict, width: int) -> dict:
    """Stratified estimate of vector totals; ``values[member]`` is an int vector of length ``width``."""
    estimate = [Fraction(0)] * width
    variance = np.zeros(width, dtype=float)
    rows = []
    sample = 0
    for key in sorted(strata):
        Nh = len(strata[key])
        picked = chosen[key]
        nh = len(picked)
        vals = np.stack([np.asarray(values[int(m)], dtype=np.int64) for m in picked])
        S = vals.sum(axis=0)
        Q = (vals.astype(np.float64) ** 2).sum(axis=0)
        for k in range(width):
            estimate[k] += Fraction(Nh * int(S[k]), nh)
        if nh > 1:
            s2 = (Q - S.astype(np.float64) ** 2 / nh) / (nh - 1)
            variance += Nh * Nh * (1.0 - nh / Nh) * s2 / nh
        sample += nh
        rows.append({"stratum": list(key) if isinstance(key, tuple) else key, "population": Nh,
                     "sample": nh, "sums": [int(v) for v in S], "sums_of_squares": [int(v) for v in Q]})
    return {"estimate": [str(e) for e in estimate], "estimate_float": [rounded(float(e)) for e in estimate],
            "standard_error": [rounded(sqrt(v)) for v in variance], "sample_size": sample, "strata": rows}


# --------------------------------------------------------------------------
# Worker pool: memory-mapped graph, per-diamond contexts, three task kinds
# --------------------------------------------------------------------------

_G: dict = {}


def _worker_init(payload: dict) -> None:
    _G.clear()
    _G.update(payload)
    _G["contexts"] = {}
    if "indptr_path" in payload:
        _G["indptr"] = np.load(payload["indptr_path"], mmap_mode="r")
        _G["indices"] = np.load(payload["indices_path"], mmap_mode="r")


def _context(key: str) -> dict:
    contexts = _G.setdefault("contexts", {})
    if key not in contexts:
        if len(contexts) >= 2:
            contexts.clear()
        with np.load(key, allow_pickle=False) as z:
            contexts[key] = {k: z[k] for k in z.files}
        d_path = str(contexts[key].get("D_path", ""))
        if d_path:
            contexts[key]["D"] = np.load(d_path, mmap_mode="r")
    return contexts[key]


def save_context(path: Path, **arrays) -> str:
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
    return str(path)


def _distance_rows(task) -> tuple[int, np.ndarray]:
    """Pruned graph distances from a block of support sites to every support site.

    From ``s`` with ``alpha_s = d(x, s)`` the search runs ``K - alpha_s`` layers
    and, at layer ``m``, keeps only sites within ``(K - alpha_s - m) a_q`` of
    the upper tip ``y``: every shortest path from ``s`` to a target ``t`` with
    ``d(s, t) <= K - alpha_s - gamma_t`` (the pairs that can be ordered inside
    the diamond) lies in those balls, so the distances are exact on such
    pairs and never smaller than the true distance elsewhere.  Entries are
    clipped to ``K + 1``.
    """
    key, lo, hi = task
    c = _context(key)
    K = int(c["K"])
    n = int(c["n"])
    x = int(c["x"])
    y = int(c["y"])
    alpha, gamma, cone, support = c["alpha"], c["gamma"], c["cone_y"], c["support"]
    indptr, indices = _G["indptr"], _G["indices"]
    out = np.full((hi - lo, len(support)), K + 1, dtype=np.int8)
    for i in range(lo, hi):
        s = int(support[i])
        a_s = int(alpha[s])
        cap = K - a_s
        if cap > 0:
            allowed = [cone[K - a_s - m] for m in range(cap)]
            dist = bfs(indptr, indices, s, n, cap=cap, allowed=allowed)
        else:
            dist = np.full(n, -1, dtype=np.int16)
            dist[s] = 0
        dist[x] = a_s
        dist[y] = int(gamma[s])
        row = dist[support].astype(np.int64)
        row[(row < 0) | (row > K)] = K + 1
        out[i - lo] = row
    return lo, out


def _profile_rows(task) -> tuple[int, np.ndarray]:
    """Between-count histograms of a block of sampled lower events of a family diamond."""
    key, lo, events = task
    c = _context(key)
    D, gamma, K = c["D"], c["gamma"], int(c["K"])
    out = np.zeros((len(events), PROFILE_BINS), dtype=np.int64)
    for e, (j, i) in enumerate(events):
        out[e] = between_counts_from_lower_event(D, gamma, K, int(j), int(i))
    return lo, out


def _sprinkling_rows(task) -> tuple[int, np.ndarray]:
    """Between-count histograms of a block of sampled points of a sprinkling."""
    key, lo, points = task
    c = _context(key)
    t, x = c["t"], c["x"]
    out = np.zeros((len(points), PROFILE_BINS), dtype=np.int64)
    for e, i in enumerate(points):
        out[e] = sprinkling_between_counts(t, x, int(i))
    return lo, out


def run_tasks(pool, function, tasks):
    return pool.imap_unordered(function, tasks) if pool is not None else map(function, tasks)


# --------------------------------------------------------------------------
# Family geometry (one level, one spatial dimension)
# --------------------------------------------------------------------------

UNIT_BALL_VOLUME = {1: 2.0, 2: pi, 3: 4.0 * pi / 3.0}


def volume_coefficient_reference(dim: int) -> float:
    """Flat-diamond volume over tau^(dim+1): pi/24 in 3+1, pi/12 in 2+1, 1/2 in 1+1."""
    return UNIT_BALL_VOLUME[dim] / (2 ** dim * (dim + 1))


def tip_directions(dim: int, q: int) -> list[tuple[int, ...]]:
    """Directions of the off-centre tips: both signs per axis and both diagonals below ``q = 34``, one sign above."""
    signs = (1, -1) if q < REDUCED_DIRECTIONS_FROM_Q else (1,)
    out = []
    for i in range(dim):
        for sign in signs:
            v = [0] * dim
            v[i] = sign
            out.append(tuple(v))
    for sign in signs:
        out.append(tuple([sign] * dim))
    return out


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class Family:
    """Exact site graph, positions and derived constants of one golden population."""

    def __init__(self, n: int, dim: int) -> None:
        q, p = fibonacci(n)
        self.n, self.dim, self.q, self.p = n, dim, q, p
        self.K = ceil_sqrt(q)
        self.indptr, self.indices, self.values, self.A, self.B, self.sites = build_site_graph(q, dim)
        self.count = q ** dim
        self.degrees = np.diff(self.indptr)
        if int(self.degrees.min()) < 1:
            raise ValueError("a site lost its same-site read")
        self.a_q = 1.0 / sqrt(q)
        self.density = q ** (dim + 0.5)
        self.xi = np.array([phi_float(v) for v in self.values])
        self.positions = self.xi[self.sites]
        self.clearance_site = np.min(np.minimum(self.positions, 1.0 - self.positions), axis=1)
        self.perm = [(b * p) % q for b in range(q)]
        self.ordered = sorted(range(q), key=lambda b: self.perm[b])
        centre_axis = 0
        for b in range(1, q):
            if phi_sign(phi_sub(phi_square(phi_sub(self.values[b], (Fraction(1, 2), 0))),
                                phi_square(phi_sub(self.values[centre_axis], (Fraction(1, 2), 0))))) < 0:
                centre_axis = b
        self.centre_axis = centre_axis
        self.centre = self.site_of([centre_axis] * dim)
        radii = [phi_sub((1, 0), self.values[self.ordered[-1]])]
        radii += [phi_scale(Fraction(1, 2), phi_sub(self.values[b], self.values[a]))
                  for a, b in zip(self.ordered, self.ordered[1:])]
        radius = radii[0]
        for r in radii[1:]:
            if phi_sign(phi_sub(r, radius)) > 0:
                radius = r
        self.h_squared = phi_scale(dim, phi_square(radius))
        self.h_over_a = sqrt(q * phi_float(self.h_squared))
        self.H_q = 2.0 * sqrt(3.0) / q if dim == 3 else None
        self.payload = None
        self.pool = None

    def site_of(self, labels) -> int:
        return int(sum(int(b) * self.q ** (self.dim - 1 - i) for i, b in enumerate(labels)))

    def labels_of(self, site: int) -> list[int]:
        return [int(v) for v in self.sites[site]]

    def open_pool(self, processes: int, workdir: Path) -> None:
        payload = {"K": self.K, "n": self.count}
        tag = f"{self.dim}_{self.q}"
        if processes > 1 and self.count >= POOL_MINIMUM_SITES:
            np.save(workdir / f"indptr_{tag}.npy", self.indptr)
            np.save(workdir / f"indices_{tag}.npy", self.indices)
            payload["indptr_path"] = str(workdir / f"indptr_{tag}.npy")
            payload["indices_path"] = str(workdir / f"indices_{tag}.npy")
        else:
            payload["indptr"], payload["indices"] = self.indptr, self.indices
        _worker_init(payload)
        self.payload = payload
        if "indptr_path" in payload:
            self.pool = mp.get_context("spawn").Pool(processes, initializer=_worker_init, initargs=(payload,))

    def close_pool(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def distances(self, x: int, y: int, K: int, workdir: Path, tag: str) -> dict:
        """Support, tip distances and the pruned support distance matrix of the diamond (0,x)-(K,y)."""
        alpha = bfs(self.indptr, self.indices, x, self.count)
        gamma = alpha if y == x else bfs(self.indptr, self.indices, y, self.count)
        if int(alpha.min()) < 0 or int(gamma.min()) < 0:
            raise ValueError("disconnected site graph")
        support = np.flatnonzero(alpha.astype(np.int64) + gamma.astype(np.int64) <= K)
        if len(support) > EXACT_SUPPORT_LIMIT:
            raise ValueError(f"support of {len(support)} sites exceeds the exact limit")
        At, Bt = metric_tables(self.q, self.A, self.B, self.sites, y)
        cone_y = cone_masks(self.q, At, Bt, K)
        key = save_context(workdir / f"ctx_{tag}.npz", K=K, n=self.count, x=x, y=y, alpha=alpha,
                           gamma=gamma, cone_y=cone_y, support=support)
        m = len(support)
        D = np.empty((m, m), dtype=np.int8)
        tasks = [(key, lo, min(lo + BATCH_SIZE, m)) for lo in range(0, m, BATCH_SIZE)]
        for lo, rows in run_tasks(self.pool, _distance_rows, tasks):
            D[lo:lo + len(rows)] = rows
        return {"x": int(x), "y": int(y), "K": int(K), "alpha": alpha, "gamma": gamma, "support": support,
                "a": alpha[support].astype(np.int64), "g": gamma[support].astype(np.int64), "D": D,
                "key": key}


# --------------------------------------------------------------------------
# Diamond statistics from a support distance matrix
# --------------------------------------------------------------------------

_WEIGHTS: dict = {}


def weights(K: int) -> np.ndarray:
    if K not in _WEIGHTS:
        _WEIGHTS[K] = moving_weights(K)
    return _WEIGHTS[K]


def pair_counts_by_start(D: np.ndarray, a: np.ndarray, g: np.ndarray, K: int, rows) -> np.ndarray:
    """Exact number of ordered pairs whose lower event sits on each start site of ``rows``."""
    W = weights(K)
    rows = np.asarray(rows, dtype=np.int64)
    out = np.zeros(len(rows), dtype=np.int64)
    block = max(1, min(len(rows), max(1, 4_000_000 // max(1, len(a)))))
    for lo in range(0, len(rows), block):
        r = rows[lo:lo + block]
        blk = np.minimum(D[r].astype(np.int64), K + 1)
        out[lo:lo + block] = W[a[r, None], g[r, None], a[None, :], g[None, :], blk].sum(axis=1)
    return out


def pair_count(D: np.ndarray, a: np.ndarray, g: np.ndarray, K: int) -> int:
    return int(pair_counts_by_start(D, a, g, K, np.arange(len(a))).sum())


def restrict(dd: dict, Kp: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Support indices, tip distances and the distance block of the ``Kp``-layer sub-diamond."""
    keep = np.flatnonzero(dd["a"] + dd["g"] <= Kp)
    if Kp == dd["K"]:
        return keep, dd["a"], dd["g"], dd["D"]
    return keep, dd["a"][keep], dd["g"][keep], dd["D"][np.ix_(keep, keep)]


def diamond_row(dd: dict, Kp: int, dim: int, seed: int) -> tuple[dict, dict]:
    """Counts, exact strict pairs, ordering fraction and dimension; sampled estimate above the threshold."""
    keep, a, g, D = restrict(dd, Kp)
    N = int((Kp - a - g + 1).sum())
    per_layer = [int(np.count_nonzero((a <= j) & (j <= Kp - g))) for j in range(Kp + 1)]
    C = pair_count(D, a, g, Kp)
    row = {"layers": Kp, "inclusive_event_count": N, "counts_by_layer": per_layer,
           "support_site_count": int(len(keep)), "pair_counting": "exact_all_pairs",
           "strict_pair_count": C}
    row.update(ordering_block(Fraction(C), N, dim))
    if N > SAMPLED_EVENT_THRESHOLD:
        strata: dict = {}
        for i in range(len(a)):
            strata.setdefault((int(a[i]), int(g[i])), []).append(i)
        chosen = allocate(strata, START_SAMPLE_SIZE, seed, minimum=2)
        starts = np.sort(np.concatenate(list(chosen.values())))
        vals = pair_counts_by_start(D, a, g, Kp, starts)
        est = stratified_totals(strata, chosen, {int(s): [int(v)] for s, v in zip(starts, vals)}, 1)
        value = Fraction(est["estimate"][0])
        row["strict_pair_count_estimate"] = {
            "seed": seed, "estimate": str(value), "estimate_float": rounded(float(value)),
            "standard_error": est["standard_error"][0], "sample_size": est["sample_size"],
            "sampled_starts_sha256": digest([int(s) for s in starts]), "strata": est["strata"],
            "ordering_fraction_estimate": rounded(float(2 * value / (N * (N - 1)))),
            "ordering_fraction_standard_error": rounded(2.0 * est["standard_error"][0] / (N * (N - 1))),
            "relative_deviation_from_exact": rounded(float(value) / C - 1.0),
            "within_three_standard_errors": bool(abs(float(value) - C) <= 3.0 * est["standard_error"][0])}
    return row, {"keep": keep, "a": a, "g": g, "D": D}


def vertical_extras(fam: Family, row: dict, tip: int, Kp: int) -> None:
    dim = fam.dim
    T = Kp * fam.a_q
    N = row["inclusive_event_count"]
    clearance = float(fam.clearance_site[tip])
    volume = diamond_volume(dim, T)
    normalized = N / fam.density
    row.update({"model_time_T_over_L": rounded(T), "proper_duration_over_L": rounded(T),
                "normalized_count": rounded(normalized), "continuum_diamond_volume": rounded(volume),
                "relative_deviation_from_continuum": rounded(normalized / volume - 1.0),
                "count_volume_coefficient": rounded(normalized / T ** (dim + 1)),
                "count_volume_coefficient_reference": rounded(volume_coefficient_reference(dim)),
                "tip_clearance_over_L": rounded(clearance),
                "spatial_buffer_over_L": rounded(clearance - T / 2.0),
                "continuum_diamond_inside_cube": bool(clearance >= T / 2.0)})
    if dim == 3:
        bound = vertical_bound(T, fam.a_q, fam.H_q, fam.h_over_a)
        row.update({"volume_error_bound": rounded(bound), "volume_error_bound_relative": rounded(bound / volume),
                    "bound_hypothesis_ball_inside_cube": bool(clearance >= T / 2.0 + fam.H_q),
                    "actual_deviation_within_bound": bool(abs(normalized - volume) <= bound)})


def moving_extras(fam: Family, row: dict, cand: dict, K: int) -> None:
    dim = fam.dim
    T = K * fam.a_q
    ell2 = phi_float(cand["ell2"])
    ell = sqrt(ell2)
    tau = sqrt(T * T - ell2)
    N = row["inclusive_event_count"]
    volume = diamond_volume(dim, tau)
    normalized = N / fam.density
    row.update({"rank_shift": cand["shift"], "x_axis_label": cand["x_axis"], "y_axis_label": cand["y_axis"],
                "x_site": cand["x"], "y_site": cand["y"],
                "tip_separation_squared_over_L2_Qphi": phi_encode(cand["ell2"]),
                "tip_separation_over_T": rounded(ell / T), "rapidity": rounded(atanh(ell / T)),
                "model_time_T_over_L": rounded(T), "proper_duration_over_L": rounded(tau),
                "spatial_buffer_over_L": rounded(cand["buffer"]),
                "spatial_buffer_positive": bool(cand["buffer"] > 0.0),
                "spatial_buffer_at_least_H_q": bool(fam.H_q is not None and cand["buffer"] >= fam.H_q),
                "normalized_count": rounded(normalized), "continuum_diamond_volume": rounded(volume),
                "relative_deviation_from_continuum": rounded(normalized / volume - 1.0),
                "count_volume_coefficient": rounded(normalized / tau ** (dim + 1)),
                "count_volume_coefficient_reference": rounded(volume_coefficient_reference(dim))})
    if dim == 3:
        bound = general_bound(T, fam.a_q, fam.H_q, fam.h_over_a)
        row.update({"volume_error_bound": rounded(bound), "volume_error_bound_relative": rounded(bound / volume),
                    "actual_deviation_within_bound": bool(abs(normalized - volume) <= bound)})


def summary(values: list) -> dict:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"count": 0, "mean": None, "sd": None, "min": None, "max": None}
    arr = np.array(vals)
    return {"count": len(vals), "mean": rounded(arr.mean()),
            "sd": rounded(arr.std(ddof=1)) if len(vals) > 1 else None,
            "min": rounded(arr.min()), "max": rounded(arr.max())}


# --------------------------------------------------------------------------
# Tips: centre, off-centre admissible tips, moving-tip candidates
# --------------------------------------------------------------------------


def tip_meta(fam: Family, site: int, role: str) -> dict:
    labels = fam.labels_of(site)
    return {"site": int(site), "labels": labels, "role": role,
            "rank_offsets_from_centre": [fam.perm[b] - fam.perm[fam.centre_axis] for b in labels],
            "position_over_L": [rounded(v) for v in fam.positions[site]],
            "clearance_over_L": rounded(float(fam.clearance_site[site]))}


def admissible_tips(fam: Family, Kp: int) -> tuple[str, int, list[tuple[int, tuple[int, ...]]]]:
    """Off-centre tips whose vertical ``Kp``-layer diamond has the declared buffer inside the cube.

    Rule order as in the source-net lane: buffer at least ``H_q`` when at
    least two sites satisfy it (three dimensions), else positive buffer.  One tip per
    direction of ``tip_directions``: the admissible site with the largest
    displacement from the centre along that direction (ties: smallest
    transverse displacement, then smallest site index); the centre and
    duplicates are dropped.
    """
    T = Kp * fam.a_q
    buffer = fam.clearance_site - T / 2.0
    rules = [("buffer_at_least_H_q", fam.H_q)] if fam.dim == 3 else []
    rules.append(("buffer_positive", 0.0))
    admissible = np.zeros(0, dtype=np.int64)
    rule = "none"
    for rule, floor_ in rules:
        admissible = np.flatnonzero((buffer > 0.0) & (buffer >= floor_))
        if len(admissible) >= 2:
            break
    if len(admissible) == 0:
        return "none", 0, []
    rel = fam.positions[admissible] - fam.positions[fam.centre]
    tips: list[tuple[int, tuple[int, ...]]] = []
    for u in tip_directions(fam.dim, fam.q):
        un = np.array(u, dtype=float)
        un /= np.linalg.norm(un)
        proj = rel @ un
        trans = np.linalg.norm(rel - proj[:, None] * un[None, :], axis=1)
        best = int(admissible[np.lexsort((admissible, np.round(trans, 12), -np.round(proj, 12)))[0]])
        if best != fam.centre and best not in [site for site, _ in tips]:
            tips.append((best, u))
    return rule, int(len(admissible)), tips


def boost_layers(fam: Family) -> int:
    """Largest layer count whose vertical diamond at the centre lies inside the cube (at least 2)."""
    Kb = fam.K
    while Kb > 2 and float(fam.clearance_site[fam.centre]) < Kb * fam.a_q / 2.0:
        Kb -= 1
    return Kb


def moving_candidates(fam: Family, K: int) -> tuple[list[dict], dict | None]:
    """Tips symmetric about the centre on the rank diagonal with a positive buffer at ``K`` layers.

    The second value is the source-net lane's choice at that layer count:
    the smallest shift with buffer at least ``H_q`` (three dimensions), else
    the smallest shift with a positive buffer.
    """
    dim, q = fam.dim, fam.q
    T = K * fam.a_q
    rc = fam.perm[fam.centre_axis]
    cands = []
    for s in range(1, q):
        lo, hi = rc - s, rc + s
        if lo < 0 or hi >= q:
            break
        xl, yl = fam.ordered[lo], fam.ordered[hi]
        ell2 = phi_scale(dim, (int(fam.A[xl][yl]), int(fam.B[xl][yl])))
        if phi_sign(phi_sub(phi_scale(q, ell2), (K * K, 0))) >= 0:
            continue
        buffer = ellipsoid_buffer([fam.xi[xl]] * dim, [fam.xi[yl]] * dim, T, dim)
        if buffer <= 0.0:
            continue
        cands.append({"shift": s, "x_axis": xl, "y_axis": yl, "x": fam.site_of([xl] * dim),
                      "y": fam.site_of([yl] * dim), "ell2": ell2, "buffer": buffer,
                      "ell_over_T": sqrt(phi_float(ell2)) / T})
    source_net_choice = None
    rules = [("buffer_at_least_H_q", fam.H_q), ("buffer_positive", 0.0)] if dim == 3 else [("buffer_positive", 0.0)]
    for rule, floor_ in rules:
        for c in cands:
            if c["buffer"] >= floor_:
                source_net_choice = dict(c, rule=rule)
                break
        if source_net_choice is not None:
            break
    return cands, source_net_choice


def select_moving(cands: list[dict], source_net_choice: dict | None) -> list[dict]:
    chosen = {}
    for target in RAPIDITY_TARGETS:
        if cands:
            best = min(cands, key=lambda c: (abs(c["ell_over_T"] - target), c["shift"]))
            chosen.setdefault(best["shift"], dict(best, selection=[]))["selection"].append(f"ell_over_T_target_{target}")
    if source_net_choice is not None:
        chosen.setdefault(source_net_choice["shift"], dict(source_net_choice, selection=[]))["selection"].append(
            "source_net_rule_" + source_net_choice["rule"])
    return [chosen[s] for s in sorted(chosen)]


# --------------------------------------------------------------------------
# Abundance of the family diamond and the sprinkling references
# --------------------------------------------------------------------------


def family_abundance(fam: Family, dd: dict, seed: int, workdir: Path, exact_pairs: int) -> dict:
    K = fam.K
    a, g, D = dd["a"], dd["g"], dd["D"]
    N = int((K - a - g + 1).sum())
    if N <= EXACT_PROFILE_LIMIT:
        prof = exact_profile(D, a, g, K)
        if prof["total_related_pairs"] != exact_pairs:
            raise ValueError("between-count matrix and pair count disagree")
        return {"mode": "exact_between_count_matrix", "event_count": N, "max_m": PROFILE_MAX_M,
                "counts": prof["counts"], "log2_bins_above_max_m": prof["log2_bins_above_max_m"],
                "log2_bin_labels": LOG2_BIN_LABELS,
                "log2_bins_fraction_of_related_pairs": [rounded(v / prof["total_related_pairs"])
                                                        for v in prof["log2_bins_above_max_m"]],
                "links_fraction_of_related_pairs": rounded(prof["counts"][0] / prof["total_related_pairs"]),
                "total_related_pairs": prof["total_related_pairs"], "ratios": profile_ratios(prof["counts"])}
    m = len(a)
    strata: dict = {}
    for i in range(m):
        for j in range(int(a[i]), K - int(g[i])):
            strata.setdefault((j, int(a[i])), []).append(j * m + i)
    chosen = allocate(strata, PROFILE_SAMPLE_SIZE, seed, minimum=1)
    members = np.sort(np.concatenate(list(chosen.values())))
    events = [(int(e // m), int(e % m)) for e in members]
    d_path = workdir / f"D_{fam.dim}_{fam.q}.npy"
    np.save(d_path, D)
    key = save_context(workdir / f"prof_{fam.dim}_{fam.q}.npz", K=K, gamma=g, D_path=str(d_path))
    values = {}
    tasks = [(key, lo, events[lo:lo + 4]) for lo in range(0, len(events), 4)]
    for lo, rows in run_tasks(fam.pool, _profile_rows, tasks):
        for e, hist in zip(members[lo:lo + len(rows)], rows):
            values[int(e)] = hist
    est = stratified_totals(strata, chosen, values, PROFILE_BINS)
    counts = est["estimate_float"][:PROFILE_MAX_M + 1]
    total = float(sum(Fraction(v) for v in est["estimate"]))
    total_se = sqrt(sum(v * v for v in est["standard_error"]))
    return {"mode": "stratified_cluster_sample_of_lower_events", "event_count": N, "max_m": PROFILE_MAX_M,
            "sample": {"seed": seed, "sample_size": est["sample_size"], "strata": "(layer j, alpha_s) of the lower event; top-layer events excluded (no partners)",
                       "sampled_events_sha256": digest([int(e) for e in members]), "strata_rows": est["strata"]},
            "counts_estimate": counts, "counts_standard_error": est["standard_error"][:PROFILE_MAX_M + 1],
            "log2_bins_above_max_m_estimate": est["estimate_float"][PROFILE_MAX_M + 1:],
            "log2_bins_above_max_m_standard_error": est["standard_error"][PROFILE_MAX_M + 1:],
            "log2_bin_labels": LOG2_BIN_LABELS,
            "log2_bins_fraction_of_related_pairs": [rounded(v / total) for v in est["estimate_float"][PROFILE_MAX_M + 1:]],
            "links_fraction_of_related_pairs": rounded(counts[0] / total),
            "total_related_pairs_estimate": rounded(total),
            "total_related_pairs_standard_error": rounded(total_se),
            "exact_total_related_pairs": exact_pairs,
            "total_relative_deviation_from_exact": rounded(total / exact_pairs - 1.0),
            "total_within_three_standard_errors": bool(abs(total - exact_pairs) <= 3.0 * total_se),
            "ratios": profile_ratios(counts)}


def sprinkling_reference(N: int, dim: int, workdir: Path, pool, seeds: int = SPRINKLING_SEEDS) -> dict:
    """Fixed-count sprinklings of the flat diamond in ``dim + 1`` dimensions at ``N`` points."""
    rows = []
    for k in range(seeds):
        seed = SAMPLE_SEED + 9000 + 1000 * dim + k
        t, x = sprinkle_diamond(N, dim, seed)
        block = max(1, min(N, 4_000_000 // max(1, N)))
        if N <= EXACT_PROFILE_LIMIT:
            F = np.zeros((N, N), dtype=bool)
            for lo in range(0, N, block):
                F[lo:lo + block] = sprinkling_future_rows(t, x, np.arange(lo, min(lo + block, N)))
            C = int(F.sum())
            prof = between_profile_from_future(F)
            counts = prof["counts"]
            profile = {"mode": "exact_between_count_matrix", "counts": counts,
                       "log2_bins_above_max_m": prof["log2_bins_above_max_m"],
                       "total_related_pairs": prof["total_related_pairs"]}
        else:
            C = 0
            for lo in range(0, N, block):
                C += int(sprinkling_future_rows(t, x, np.arange(lo, min(lo + block, N))).sum())
            bands = 8
            strata = {b: list(range(b * N // bands, (b + 1) * N // bands)) for b in range(bands)}
            chosen = allocate(strata, SPRINKLING_PROFILE_SAMPLE_SIZE, seed + 1, minimum=2)
            points = np.sort(np.concatenate(list(chosen.values())))
            key = save_context(workdir / f"spr_{dim}_{N}_{k}.npz", t=t, x=x)
            values = {}
            tasks = [(key, lo, points[lo:lo + 4]) for lo in range(0, len(points), 4)]
            for lo, hist_rows in run_tasks(pool, _sprinkling_rows, tasks):
                for p_, hist in zip(points[lo:lo + len(hist_rows)], hist_rows):
                    values[int(p_)] = hist
            est = stratified_totals(strata, chosen, values, PROFILE_BINS)
            counts = est["estimate_float"][:PROFILE_MAX_M + 1]
            total = float(sum(Fraction(v) for v in est["estimate"]))
            total_se = sqrt(sum(v * v for v in est["standard_error"]))
            profile = {"mode": "stratified_cluster_sample_of_lower_points",
                       "sample": {"seed": seed + 1, "sample_size": est["sample_size"],
                                  "strata": f"{bands} equal-count time bands", "strata_rows": est["strata"],
                                  "sampled_points_sha256": digest([int(p_) for p_ in points])},
                       "counts_estimate": counts, "counts_standard_error": est["standard_error"][:PROFILE_MAX_M + 1],
                       "log2_bins_above_max_m_estimate": est["estimate_float"][PROFILE_MAX_M + 1:],
                       "log2_bins_above_max_m_standard_error": est["standard_error"][PROFILE_MAX_M + 1:],
                       "total_related_pairs_estimate": rounded(total),
                       "total_related_pairs_standard_error": rounded(total_se),
                       "exact_total_related_pairs": C,
                       "total_relative_deviation_from_exact": rounded(total / C - 1.0),
                       "total_within_three_standard_errors": bool(abs(total - C) <= 3.0 * total_se)}
        bins_above = profile.get("log2_bins_above_max_m", profile.get("log2_bins_above_max_m_estimate"))
        total_pairs = profile.get("total_related_pairs", profile.get("total_related_pairs_estimate"))
        row = {"seed": seed, "strict_pair_count": C, "profile": profile, "ratios": profile_ratios(counts),
               "log2_bins_fraction_of_related_pairs": [rounded(v / total_pairs) for v in bins_above],
               "links_fraction_of_related_pairs": rounded(counts[0] / total_pairs)}
        row.update(ordering_block(Fraction(C), N, dim))
        if dim == 3:
            row["benincasa_dowker"] = benincasa_dowker_action(N, counts)
        rows.append(row)
        _log(f"  sprinkling {dim}+1 N={N} seed {k}: f={row['ordering_fraction_float']} dim={row['myrheim_meyer_dimension']}")
    ratios = np.array([[np.nan if v is None else v for v in r["ratios"]] for r in rows], dtype=float)
    out = {"spacetime_dimension": dim + 1, "event_count": N, "seeds": seeds, "seed_base": SAMPLE_SEED + 9000 + 1000 * dim,
           "diamond": "flat Alexandrov diamond of unit duration, tips (0,0) and (1,0); fixed point count",
           "log2_bin_labels": LOG2_BIN_LABELS,
           "profile_mode": rows[0]["profile"]["mode"], "per_seed": rows,
           "ratios_mean": [rounded(v) for v in np.nanmean(ratios, axis=0)],
           "ratios_sd": [rounded(v) for v in np.nanstd(ratios, axis=0, ddof=1)] if seeds > 1 else None,
           "log2_bins_fraction_mean": [rounded(v) for v in np.mean(
               [r["log2_bins_fraction_of_related_pairs"] for r in rows], axis=0)],
           "links_fraction_of_related_pairs": summary([r["links_fraction_of_related_pairs"] for r in rows]),
           "ordering_fraction": summary([r["ordering_fraction_float"] for r in rows]),
           "myrheim_meyer_dimension": summary([r["myrheim_meyer_dimension"] for r in rows])}
    if dim == 3:
        out["benincasa_dowker_action"] = summary([r["benincasa_dowker"]["action"] for r in rows])
        out["benincasa_dowker_action_over_N"] = summary([r["benincasa_dowker"]["action_over_N"] for r in rows])
    return out


# --------------------------------------------------------------------------
# Link directions, count clocks, cross-check
# --------------------------------------------------------------------------


def link_directions(fam: Family) -> dict | None:
    """Multipole power of the directed link directions (all reads between consecutive layers)."""
    dim, q = fam.dim, fam.q
    if dim < 2:
        return None
    acc = MultipoleAccumulator(dim)
    mval = np.array([v[0] for v in fam.values], dtype=np.int64)
    base = 2 * q + 1
    keys_seen = []
    chunk = max(1, 4_000_000 // max(1, int(fam.degrees.max())))
    for lo in range(0, fam.count, chunk):
        hi = min(lo + chunk, fam.count)
        rows = np.repeat(np.arange(lo, hi), fam.degrees[lo:hi])
        cols = fam.indices[fam.indptr[lo]:fam.indptr[hi]].astype(np.int64)
        mask = cols != rows
        rows, cols = rows[mask], cols[mask]
        acc.add(fam.positions[cols] - fam.positions[rows])
        key = np.zeros(len(rows), dtype=np.int64)
        for axis in range(dim):
            bs, bt = fam.sites[rows, axis].astype(np.int64), fam.sites[cols, axis].astype(np.int64)
            axis_id = (mval[bt] - mval[bs] + q) * base + (bt - bs + q)
            key = key * (base * base) + axis_id
        keys_seen.append(np.unique(key))
    distinct = np.unique(np.concatenate(keys_seen)) if keys_seen else np.zeros(0, dtype=np.int64)
    vectors = np.zeros((len(distinct), dim))
    rem = distinct.copy()
    for axis in range(dim - 1, -1, -1):
        axis_id = rem % (base * base)
        rem //= base * base
        dm = axis_id // base - q
        db = axis_id % base - q
        vectors[:, axis] = dm + db * (1.0 + sqrt(5.0)) / 2.0
    unit = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    directions = len(np.unique(np.round(unit, 9), axis=0))
    powers = acc.powers()
    reference = uniform_direction_reference(acc.count, dim)
    return {"directed_link_count": acc.count, "undirected_spatial_edges": acc.count // 2,
            "distinct_displacement_count": int(len(distinct)), "distinct_direction_count": int(directions),
            "harmonics": "real spherical harmonics l = 1..4" if dim == 3 else "Fourier modes m = 1..4 on S^1",
            "power_relative_to_degree_zero": [rounded(v) for v in powers[1:]],
            "uniform_reference": reference,
            "power_over_uniform_mean": [rounded(p / u) if u else None
                                        for p, u in zip(powers[1:], reference["power_mean"][1:])],
            "odd_degrees_vanish_by_two_way_reads": True}


def enclosure(NI: int, EI: float, NJ: int, EJ: float, root: float) -> dict:
    lower = (max(NI - EI, 0.0) / (NJ + EJ)) ** root
    upper = ((NI + EI) / (NJ - EJ)) ** root if NJ > EJ else None
    return {"enclosure_lower": rounded(lower), "enclosure_upper": None if upper is None else rounded(upper),
            "count_error_interval": rounded(EI), "count_error_reference": rounded(EJ)}


def count_clocks(fam: Family, centre_rows: dict, moving_rows: list) -> dict:
    dim = fam.dim
    root = 1.0 / (dim + 1)
    nested = []
    for Kp in sorted(centre_rows, reverse=True):
        Kr = Kp // 2
        if Kr < 1:
            continue
        I, J = centre_rows[Kp], centre_rows[Kr]
        NI, NJ = I["inclusive_event_count"], J["inclusive_event_count"]
        row = {"interval_layers": Kp, "reference_layers": Kr, "interval_count": NI, "reference_count": NJ,
               "count_clock_fourth_root": rounded((NI / NJ) ** 0.25),
               "count_clock_dimension_root": rounded((NI / NJ) ** root),
               "model_time_ratio": rounded(Kp / Kr),
               "relative_deviation_fourth_root": rounded((NI / NJ) ** 0.25 / (Kp / Kr) - 1.0),
               "relative_deviation_dimension_root": rounded((NI / NJ) ** root / (Kp / Kr) - 1.0)}
        if dim == 3:
            EI = I["volume_error_bound"] * fam.density
            EJ = J["volume_error_bound"] * fam.density
            row.update(enclosure(NI, EI, NJ, EJ, 0.25))
            row["enclosure_hypotheses_satisfied"] = bool(I["bound_hypothesis_ball_inside_cube"]
                                                         and J["bound_hypothesis_ball_inside_cube"] and NJ > EJ)
        nested.append(row)
    moving = []
    for M in moving_rows:
        ref = centre_rows[M["against_vertical"]["reference_layers"]]
        NI, NJ = M["inclusive_event_count"], ref["inclusive_event_count"]
        ratio = M["proper_duration_over_L"] / ref["model_time_T_over_L"]
        row = {"rank_shift": M["rank_shift"], "layers": M["layers"], "interval_count": NI, "reference_count": NJ,
               "reference_layers": ref["layers"], "proper_duration_ratio": rounded(ratio),
               "count_clock_fourth_root": rounded((NI / NJ) ** 0.25),
               "count_clock_dimension_root": rounded((NI / NJ) ** root),
               "relative_deviation_fourth_root": rounded((NI / NJ) ** 0.25 / ratio - 1.0),
               "relative_deviation_dimension_root": rounded((NI / NJ) ** root / ratio - 1.0)}
        if dim == 3:
            EI = M["volume_error_bound"] * fam.density
            EJ = ref["volume_error_bound"] * fam.density
            row.update(enclosure(NI, EI, NJ, EJ, 0.25))
            row["enclosure_hypotheses_satisfied"] = bool(M["spatial_buffer_positive"]
                                                         and ref["bound_hypothesis_ball_inside_cube"] and NJ > EJ)
        moving.append(row)
    return {"definition": "(|I|/|J|)^(1/4) from inclusive interval cardinalities; the dimension root uses 1/(dim+1)",
            "nested_vertical": nested, "moving_against_vertical": moving}


def source_net_family(n: int, dim: int) -> dict | None:
    data = json.loads(SOURCE_NET_RECEIPT.read_bytes())
    for lv in data["levels"]:
        if lv["fibonacci_index"] == n:
            for fam in lv["families"]:
                if fam["dimension"] == dim:
                    return fam
    return None


def source_net_cross_check(n: int, dim: int, centre_rows: dict, moving_rows: list) -> dict | None:
    theirs = source_net_family(n, dim)
    if theirs is None:
        return None
    fields = {}
    for row in theirs["vertical_intervals"]:
        k = row["layers"]
        mine = centre_rows.get(k)
        if mine is None:
            continue
        entry = {"inclusive_event_count": mine["inclusive_event_count"] == row["inclusive_event_count"],
                 "counts_by_layer": mine["counts_by_layer"] == row["counts_by_layer"]}
        if row.get("pair_counting") == "exact_all_pairs":
            entry["strict_pair_count"] = mine["strict_pair_count"] == row["strict_pair_count"]
        fields[f"vertical_{k}"] = entry
    m = theirs.get("moving_tip_interval")
    if m is not None:
        for row in moving_rows:
            if (row["layers"] == m["layers"] and row["rank_shift"] == m["rank_shift"]
                    and row["x_site"] == m["x_site"] and row["y_site"] == m["y_site"]):
                entry = {"inclusive_event_count": row["inclusive_event_count"] == m["inclusive_event_count"],
                         "counts_by_layer": row["counts_by_layer"] == m["counts_by_layer"]}
                if m.get("pair_counting") == "exact_all_pairs":
                    entry["strict_pair_count"] = row["strict_pair_count"] == m["strict_pair_count"]
                fields["moving"] = entry
    return {"fields": fields, "all_agree": all(v for e in fields.values() for v in e.values())}


# --------------------------------------------------------------------------
# One family at one level
# --------------------------------------------------------------------------


def reference_key(spacetime_dim: int, N: int) -> str:
    return f"spacetime_{spacetime_dim}_events_{N}"


def compare_profiles(family_ratios: list, reference: dict) -> list:
    out = []
    for m, (fr, mean, sd) in enumerate(zip(family_ratios, reference["ratios_mean"], reference["ratios_sd"])):
        pull = None if (fr is None or mean is None or not sd) else rounded((fr - mean) / sd)
        out.append({"m": m, "family": fr, "reference_mean": mean, "reference_sd": sd, "pull": pull})
    return out


def build_family(n: int, dim: int, processes: int, workdir: Path, references: dict) -> dict:
    t0 = time.time()
    fam = Family(n, dim)
    q, K = fam.q, fam.K
    seed_base = SAMPLE_SEED + 1000 * dim + q
    fam.open_pool(processes, workdir)
    try:
        dd_c = fam.distances(fam.centre, fam.centre, K, workdir, f"c_{dim}_{q}")
        centre_rows = {}
        for k in range(1, K + 1):
            row, _ = diamond_row(dd_c, k, dim, seed_base + k)
            vertical_extras(fam, row, fam.centre, k)
            row["tip"] = tip_meta(fam, fam.centre, "centre")
            centre_rows[k] = row
        t1 = time.time()
        _log(f"level n={n} q={q} dim={dim}: centre support {len(dd_c['support'])} sites in {t1 - t0:.1f}s")

        sizes = [k for k in (K, K - 1, K - 2) if k >= 2]
        tips_by_size = {k: admissible_tips(fam, k) for k in sizes}
        tip_K: dict = {}
        for k in sizes:
            for site, _direction in tips_by_size[k][2]:
                tip_K[site] = max(tip_K.get(site, 0), k)
        tip_rows = {}
        for site in sorted(tip_K):
            dd = fam.distances(site, site, tip_K[site], workdir, f"t_{dim}_{q}_{site}")
            for k in sizes:
                for tip_site, direction in tips_by_size[k][2]:
                    if tip_site == site:
                        row, _ = diamond_row(dd, k, dim, seed_base + 100000 * k + site)
                        vertical_extras(fam, row, site, k)
                        row["tip"] = tip_meta(fam, site, "direction " + ",".join(map(str, direction)))
                        tip_rows[(site, k)] = row
        size_blocks = []
        all_dims = []
        for k in sizes:
            rule, admissible_count, tips = tips_by_size[k]
            rows = [centre_rows[k]] + [tip_rows[(site, k)] for site, _d in tips]
            dims = [r["myrheim_meyer_dimension"] for r in rows]
            fracs = [r["ordering_fraction_float"] for r in rows]
            size_blocks.append({"layers": k, "tip_rule": rule, "admissible_site_count": admissible_count,
                                "diamond_count": len(rows), "diamonds": rows,
                                "dimension_summary": summary(dims), "ordering_fraction_summary": summary(fracs),
                                "off_centre_dimension_summary": summary(dims[1:]),
                                "counts_summary": summary([r["inclusive_event_count"] for r in rows])})
            all_dims += dims
        homogeneity = {"sizes": size_blocks, "all_sizes_dimension_summary": summary(all_dims),
                       "tip_directions": [list(u) for u in tip_directions(dim, q)]}
        t2 = time.time()
        _log(f"  homogeneity: {len(tip_K)} off-centre tips in {t2 - t1:.1f}s")

        def moving_diamond(cand: dict, Kp: int, tag: str) -> dict | None:
            alpha_x = bfs(fam.indptr, fam.indices, cand["x"], fam.count)
            gamma_y = bfs(fam.indptr, fam.indices, cand["y"], fam.count)
            support = int(np.count_nonzero(alpha_x.astype(np.int64) + gamma_y.astype(np.int64) <= Kp))
            if support > EXACT_SUPPORT_LIMIT:
                skipped.append({"rank_shift": cand["shift"], "layers": Kp, "support_site_count": support,
                                "reason": "support above the exact limit"})
                return None
            if int(gamma_y[cand["x"]]) > Kp:
                skipped.append({"rank_shift": cand["shift"], "layers": Kp, "support_site_count": support,
                                "reason": "tips at graph distance above the layer count"})
                return None
            dd = fam.distances(cand["x"], cand["y"], Kp, workdir, tag)
            row, _ = diamond_row(dd, Kp, dim, seed_base + 5000 + 100 * Kp + cand["shift"])
            if row["ordering_fraction_float"] is None:
                skipped.append({"rank_shift": cand["shift"], "layers": Kp, "support_site_count": support,
                                "reason": "fewer than two events"})
                return None
            moving_extras(fam, row, cand, Kp)
            row["selection"] = cand.get("selection", [])
            vref = centre_rows[Kp]
            row["against_vertical"] = {
                "reference_layers": Kp,
                "ordering_fraction_difference": rounded(row["ordering_fraction_float"] - vref["ordering_fraction_float"]),
                "dimension_difference": (None if row["myrheim_meyer_dimension"] is None
                                         or vref["myrheim_meyer_dimension"] is None
                                         else rounded(row["myrheim_meyer_dimension"] - vref["myrheim_meyer_dimension"])),
                "count_volume_coefficient_ratio": rounded(row["count_volume_coefficient"] / vref["count_volume_coefficient"])}
            return row

        Kb = boost_layers(fam)
        cands, _ = moving_candidates(fam, Kb)
        cands_K, sn_choice = moving_candidates(fam, K)
        moving_rows = []
        skipped = []
        sn_row = None
        if dim >= 2:
            for cand in select_moving(cands, None):
                row = moving_diamond(cand, Kb, f"m_{dim}_{q}_{Kb}_{cand['shift']}")
                if row is not None:
                    moving_rows.append(row)
            if sn_choice is not None:
                sn_choice["selection"] = ["source_net_rule_" + sn_choice["rule"]]
                sn_row = moving_diamond(sn_choice, K, f"m_{dim}_{q}_{K}_{sn_choice['shift']}")
        ref = centre_rows[Kb]
        boosts = {"layers": Kb,
                  "layer_rule": "largest layer count whose vertical diamond at the centre lies inside the cube",
                  "tip_rule": "tips symmetric about the centre on the rank diagonal with a positive buffer; "
                              "shifts closest to the declared tip-separation targets",
                  "targets_ell_over_T": list(RAPIDITY_TARGETS),
                  "candidates": [{"rank_shift": c["shift"], "tip_separation_over_T": rounded(c["ell_over_T"]),
                                  "spatial_buffer_over_L": rounded(c["buffer"])} for c in cands],
                  "skipped": skipped, "diamonds": moving_rows,
                  "vertical_reference": {"layers": Kb, "inclusive_event_count": ref["inclusive_event_count"],
                                         "ordering_fraction_float": ref["ordering_fraction_float"],
                                         "myrheim_meyer_dimension": ref["myrheim_meyer_dimension"],
                                         "count_volume_coefficient": ref["count_volume_coefficient"]},
                  "ordering_fraction_summary_with_vertical": summary(
                      [ref["ordering_fraction_float"]] + [r["ordering_fraction_float"] for r in moving_rows]),
                  "dimension_summary_with_vertical": summary(
                      [ref["myrheim_meyer_dimension"]] + [r["myrheim_meyer_dimension"] for r in moving_rows]),
                  "count_volume_coefficient_summary_with_vertical": summary(
                      [ref["count_volume_coefficient"]] + [r["count_volume_coefficient"] for r in moving_rows]),
                  "source_net_moving_diamond": sn_row}
        t3 = time.time()
        _log(f"  boosts: {len(moving_rows)} moving diamonds at {Kb} layers in {t3 - t2:.1f}s")

        abundance = family_abundance(fam, dd_c, seed_base + 77, workdir, centre_rows[K]["strict_pair_count"])
        N = abundance["event_count"]
        abundance["reference_keys"] = {}
        abundance["comparison"] = {}
        for st in REFERENCE_SPACETIME_DIMENSIONS[dim]:
            key = reference_key(st, N)
            if key not in references:
                references[key] = sprinkling_reference(N, st - 1, workdir, fam.pool)
            abundance["reference_keys"][str(st)] = key
            abundance["comparison"][str(st)] = compare_profiles(abundance["ratios"], references[key])
        del dd_c["D"]
        action = None
        if dim == 3:
            counts = abundance["counts"] if abundance["mode"].startswith("exact") else abundance["counts_estimate"]
            action = benincasa_dowker_action(N, counts)
            action["mode"] = "exact" if abundance["mode"].startswith("exact") else "from_sampled_profile_estimates"
            r4 = references[reference_key(4, N)]
            action["reference_3_plus_1"] = {"action": r4["benincasa_dowker_action"],
                                            "action_over_N": r4["benincasa_dowker_action_over_N"]}
            action["convention"] = BD_CONVENTION
            action["reading"] = ("scalar-curvature reading diagnostic only: the flat sprinkling mean is a "
                                 "boundary contribution and the family's small-interval statistics move the value")
        t4 = time.time()
        _log(f"  abundance {abundance['mode']} N={N} in {t4 - t3:.1f}s")

        directions = link_directions(fam)
        t5 = time.time()
        all_moving = moving_rows + ([sn_row] if sn_row is not None else [])
        clocks = count_clocks(fam, centre_rows, all_moving)
        cross = source_net_cross_check(n, dim, centre_rows, all_moving)
        if cross is not None and not cross["all_agree"]:
            raise ValueError(f"source-net cross-check disagrees at n={n} dim={dim}: {cross}")
        _log(f"  directions in {t5 - t4:.1f}s; family total {t5 - t0:.1f}s")
    finally:
        fam.close_pool()
    return {"dimension": dim, "fibonacci_index": n, "q": q, "p": fam.p, "layer_steps": K,
            "site_count": fam.count, "centre_axis": fam.centre_axis, "centre_site": fam.centre,
            "edge_radius_a_over_L": rounded(fam.a_q), "model_time_T_over_L": rounded(K * fam.a_q),
            "assignment_error_H_over_L": None if fam.H_q is None else rounded(fam.H_q),
            "h_over_a": rounded(fam.h_over_a),
            "event_density_times_L_to_dim_plus_one": rounded(fam.density),
            "centre_support_site_count": int(len(dd_c["support"])),
            "centre_vertical_intervals": [centre_rows[k] for k in range(1, K + 1)],
            "homogeneity": homogeneity, "boosts": boosts, "abundance": abundance, "action": action,
            "link_directions": directions, "count_clocks": clocks, "source_net_cross_check": cross}


# --------------------------------------------------------------------------
# Trend table and the receipt
# --------------------------------------------------------------------------


def trend_table(levels: list, references: dict) -> dict:
    rows = []
    previous = None
    for lv in levels:
        fam = next(f for f in lv["families"] if f["dimension"] == 3)
        K = fam["layer_steps"]
        top = fam["centre_vertical_intervals"][K - 1]
        deviation = abs(top["ordering_fraction_float"] - 0.1)
        ref4 = references[fam["abundance"]["reference_keys"]["4"]]
        hom = {str(b["layers"]): {"mean": b["dimension_summary"]["mean"], "sd": b["dimension_summary"]["sd"],
                                   "count": b["dimension_summary"]["count"]} for b in fam["homogeneity"]["sizes"]}
        boosts = fam["boosts"]
        clocks = fam["count_clocks"]
        moving_dev = [r["relative_deviation_fourth_root"] for r in clocks["moving_against_vertical"]]
        row = {"q": fam["q"], "layers": K, "centre_event_count": top["inclusive_event_count"],
               "centre_ordering_fraction": top["ordering_fraction_float"],
               "centre_dimension": top["myrheim_meyer_dimension"],
               "absolute_deviation_from_one_tenth": rounded(deviation),
               "deviation_ratio_to_previous_level": None if previous in (None, 0.0) else rounded(deviation / previous),
               "volume_error_bound_relative": top.get("volume_error_bound_relative"),
               "homogeneity_dimension_by_size": hom,
               "all_sizes_dimension": fam["homogeneity"]["all_sizes_dimension_summary"],
               "boost_layers": boosts["layers"],
               "boost_dimension_with_vertical": boosts["dimension_summary_with_vertical"],
               "boost_ordering_fraction_with_vertical": boosts["ordering_fraction_summary_with_vertical"],
               "boost_count_volume_coefficient_with_vertical": boosts["count_volume_coefficient_summary_with_vertical"],
               "abundance_mode": fam["abundance"]["mode"],
               "abundance_ratios_family_m1_m2_m3": fam["abundance"]["ratios"][1:4],
               "abundance_ratios_sprinkling_3_plus_1_m1_m2_m3": ref4["ratios_mean"][1:4],
               "action_over_N_family": None if fam["action"] is None else fam["action"]["action_over_N"],
               "action_over_N_sprinkling_3_plus_1": ref4["benincasa_dowker_action_over_N"]["mean"],
               "link_multipole_power_l1_to_l4": fam["link_directions"]["power_relative_to_degree_zero"],
               "link_multipole_power_over_uniform": fam["link_directions"]["power_over_uniform_mean"],
               "distinct_link_directions": fam["link_directions"]["distinct_direction_count"],
               "nested_clock_relative_deviation": clocks["nested_vertical"][0]["relative_deviation_fourth_root"],
               "moving_clock_relative_deviation": summary(moving_dev)}
        rows.append(row)
        previous = deviation
    controls = []
    for lv in levels:
        for fam in lv["families"]:
            if fam["dimension"] == 3:
                continue
            K = fam["layer_steps"]
            top = fam["centre_vertical_intervals"][K - 1]
            controls.append({"dimension": fam["dimension"], "q": fam["q"], "layers": K,
                             "centre_ordering_fraction": top["ordering_fraction_float"],
                             "centre_dimension": top["myrheim_meyer_dimension"],
                             "all_sizes_dimension": fam["homogeneity"]["all_sizes_dimension_summary"],
                             "nested_clock_relative_deviation_dimension_root":
                                 fam["count_clocks"]["nested_vertical"][0]["relative_deviation_dimension_root"]})
    return {"three_dimensional_family": rows, "controls": controls,
            "statement": "Finite levels q = 8, 13, 21, 34 of one nested construction; the per-level values "
                         "and their ratios are the measured convergence; the finite runs do not demonstrate "
                         "the asymptotic limit."}


def build(levels=LEVELS, dimensions=DIMENSIONS, processes: int | None = None) -> dict:
    processes = processes or max(1, min(6, os.cpu_count() or 1))
    rows = []
    references: dict = {}
    with tempfile.TemporaryDirectory(prefix="oph_manifold_obs_") as tmp:
        workdir = Path(tmp)
        for n in levels:
            families = [build_family(n, dim, processes, workdir, references) for dim in dimensions]
            rows.append({"fibonacci_index": n, "q": fibonacci(n)[0], "families": families})
    pins = {p: file_sha256(ROOT / p) for p in PINS}
    return {"schema": SCHEMA, "scope": SCOPE, "claim_boundary": CLAIM_BOUNDARY,
            "certificate_rows": CERTIFICATE_ROWS,
            "conventions": {
                "event_order": "(j,s) <= (j',t) iff graph distance d(s,t) <= j'-j on the exact site graph, waiting included",
                "finite_schedule": "K_q = ceil(sqrt q); T_q = K_q L/sqrt q; density rho_q = q^(dim+1/2)/L^(dim+1) with c = 1",
                "diamond": "inclusive interval between (0,x) and (K',y): events (j,s) with d(x,s) <= j <= K' - d(s,y)",
                "pair_counting": {"exact_support_limit": EXACT_SUPPORT_LIMIT,
                                  "sampled_event_threshold": SAMPLED_EVENT_THRESHOLD,
                                  "start_sample_size": START_SAMPLE_SIZE,
                                  "rule": "exact all pairs on every diamond whose support has at most the exact "
                                          "limit of sites; diamonds above the event threshold carry in addition a "
                                          "stratified per-start estimate (strata (d(x,s), d(s,y)), proportional "
                                          "allocation, at least two starts per stratum, without replacement)"},
                "abundance_profile": {"definition": "N_m = number of pairs x < y with exactly m events strictly "
                                                    "between; N_0 = number of links (cover relations)",
                                      "max_m": PROFILE_MAX_M, "exact_profile_limit_events": EXACT_PROFILE_LIMIT,
                                      "sampled_profile": {"family_lower_event_sample": PROFILE_SAMPLE_SIZE,
                                                          "sprinkling_lower_point_sample": SPRINKLING_PROFILE_SAMPLE_SIZE,
                                                          "estimator": "stratified cluster sample: every upper partner "
                                                                       "of a sampled lower event is counted exactly"}},
                "benincasa_dowker": BD_CONVENTION,
                "link_directions": "every read (j,s) -> (j+1,t) with t != s is a cover relation; its direction is "
                                   "the unit vector of xi_t - xi_s in the source-Gram readback; power in degree l "
                                   "is sum_m mean(Y_lm)^2 in the basis orthonormal for the uniform probability "
                                   "measure (degree zero equals 1); uniform reference of equal cardinality",
                "count_clock": "(|I|/|J|)^(1/4) from inclusive cardinalities; enclosure from the theory's finite "
                               "volume bounds where their hypotheses hold",
                "sprinkling": "fixed point count in the flat diamond of unit duration; seeds declared per reference",
                "seed_base": SAMPLE_SEED, "sprinkling_seeds": SPRINKLING_SEEDS, "direction_seeds": DIRECTION_SEEDS,
                "rapidity_targets_ell_over_T": list(RAPIDITY_TARGETS)},
            "reference_fractions": {str(d + 1): str(REFERENCE_FRACTION[d]) for d in DIMENSIONS},
            "levels": rows,
            "sprinkling_references": {k: references[k] for k in sorted(references)},
            "trend_table": trend_table(rows, references),
            "source_pins": pins}


def repin(path: Path = OUTPUT) -> bytes:
    receipt = json.loads(path.read_bytes())
    receipt["source_pins"] = {p: file_sha256(ROOT / p) for p in PINS}
    return canonical(receipt)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rebuild and write the receipt")
    parser.add_argument("--check", action="store_true", help="rebuild and compare with the receipt")
    parser.add_argument("--repin", action="store_true", help="refresh the file pins of the committed receipt")
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--levels", type=int, nargs="*", default=None)
    parser.add_argument("--dimensions", type=int, nargs="*", default=None)
    args = parser.parse_args(argv)
    if args.repin:
        data = repin()
        OUTPUT.write_bytes(data)
        print("MANIFOLD_OBSERVATIONS_REPINNED", len(data), hashlib.sha256(data).hexdigest())
        return
    t0 = time.time()
    data = canonical(build(levels=tuple(args.levels) if args.levels else LEVELS,
                           dimensions=tuple(args.dimensions) if args.dimensions else DIMENSIONS,
                           processes=args.processes))
    _log(f"build wall time {time.time() - t0:.1f}s")
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_bytes(data)
    if args.check and OUTPUT.read_bytes() != data:
        raise ValueError("manifold-observations receipt is stale")
    print("MANIFOLD_OBSERVATIONS_REPLAYED", len(data), hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
