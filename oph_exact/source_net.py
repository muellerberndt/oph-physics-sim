"""Source-net causal limit on the golden conservative source-record populations.

This lane replays the r2039 construction of
``code/causal_refinement/source_net_causet.py`` in the theory repository with
integer-pair arithmetic in ``Q(phi)``: golden orbit values
``xi_b = b*phi - floor(b*phi)`` stored as ``(m, b)`` meaning ``m + b*phi``, the
conservative source record ``z(b)`` and its integer current section, the
``q^3`` product population on the rank-three source Gram metric, the
complete-neighbour read inside the radius ``a_q = L/sqrt(q)`` decided by the
exact sign rule, the same-site read (waiting), and the layer schedule
``K_q = ceil(sqrt q)``.  The event order of the layered read law is
``(j, s) <= (j', t)`` iff ``d(s, t) <= j' - j`` with ``d`` the graph distance
on the site graph, waiting included.

Per level the lane adds interior-diamond statistics: inclusive event counts
against ``pi c^3 T^4 / 24`` with the paper's finite error bounds, the
ordering fraction ``2C/(N(N-1))`` against the four-dimensional Myrheim-Meyer
value ``1/10``, the inverted Myrheim-Meyer dimension, a moving-tip diamond
against its continuum volume, and the count-clock ratio of two vertical
diamonds.  One- and two-dimensional golden populations under the same read
law and layer rule are the dimension controls.

The population, the read law, the layer duration ``Delta_q = a_q / c`` and one
counted event per site and layer are supplied, as in the paper.  Native
repair does not select them; no physical clock or spacetime is identified;
finite runs do not demonstrate the asymptotic limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from fractions import Fraction
from math import acos, asin, isqrt, pi, sqrt
from pathlib import Path

import numpy as np
from scipy import integrate

from oph_fpe.bulk.causet_likeness import (
    invert_myrheim_meyer_fraction,
    myrheim_meyer_fraction,
)

ROOT = Path(__file__).resolve().parents[1]
RER_ROOT = Path(os.environ.get("OPH_RER_ROOT", str(ROOT.parent / "reverse-engineering-reality")))
OUTPUT = ROOT / "data/exact/source_net_causal_limit_receipt.json"
SCHEMA = "oph.exact.source-net-causal-limit.v1"
LEVELS = (5, 6, 7, 8, 9, 10, 11)
RER_LEVELS = (5, 6, 7)
DIMENSIONS = (3, 2, 1)
EXACT_SUPPORT_LIMIT = 20000
SAMPLE_SIZE = 2000
SAMPLE_SEED = 20260909
POOL_MINIMUM_SITES = 4000
BATCH_SIZE = 48
DIGEST_ROWS = 2000
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
RER_RECEIPT = "code/causal_refinement/source_net_causet_receipt.json"
RER_RECEIPT_SHA256 = "c0f790ad383039186371b7a8020f190371fa533ad634010f01f9c271f1545681"
RER_COMPARED_FIELDS = (
    "q", "p", "layer_steps", "orbit_Qphi", "grid_permutation", "record_count",
    "source_records_sha256", "maximum_word_length", "sum_word_lengths",
    "word_length_bound", "whole_cube_fill_h_squared_over_L2_Qphi",
    "one_dimensional_fill_over_L_Qphi", "h_over_a_upper",
    "certified_inner_speed_lower", "positive_inner_cone",
    "neighbors_including_wait_sha256", "undirected_spatial_edges",
    "minimum_neighbor_count_including_wait", "maximum_neighbor_count_including_wait",
    "intervention_source_id", "reachability_probes", "center_intervals",
    "source_record_examples", "quadrature_displacement_H_squared_over_L2",
    "radius_squared_over_L2", "layer_time_equals_radius", "exact_width",
    "exact_height_in_events",
)
SCOPE = {
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
CLAIM_BOUNDARY = (
    "Finite replay of the declared source-record family: q^3 golden sites on "
    "the rank-three source Gram metric, complete-neighbour reads inside "
    "a_q = L/sqrt(q), one event per site and layer, layer duration a_q/c. "
    "Counts, ordering fractions and count clocks are finite diagnostics of "
    "that supplied law at q <= 89. They do not select the population or the "
    "read law from native repairs, identify a physical clock or spacetime, "
    "or demonstrate the asymptotic limit of the paper's propositions."
)
UNIT_BALL_VOLUME = {1: 2.0, 2: pi, 3: 4.0 * pi / 3.0}
REFERENCE_FRACTION = {1: Fraction(1, 2), 2: Fraction(8, 35), 3: Fraction(1, 10)}


# --------------------------------------------------------------------------
# Canonical JSON and hashes
# --------------------------------------------------------------------------


def canonical(x) -> bytes:
    return (json.dumps(x, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def digest(x) -> str:
    return hashlib.sha256(canonical(x)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rounded(x: float) -> float:
    """Twelve significant digits; derived statistics only."""
    return float(f"{float(x):.12g}")


# --------------------------------------------------------------------------
# Q(phi) arithmetic: (a, b) means a + b*phi with rational coefficients.
# --------------------------------------------------------------------------


def phi_sign(x) -> int:
    """Exact sign of a + b*phi, mirroring the theory producer's rule."""
    a, b = x
    c = 2 * a + b
    if b == 0:
        return (c > 0) - (c < 0)
    if c >= 0 and b > 0:
        return 1
    if c <= 0 and b < 0:
        return -1
    d = c * c - 5 * b * b
    return ((d > 0) - (d < 0)) * (1 if c > 0 else -1)


def phi_add(x, y):
    return (x[0] + y[0], x[1] + y[1])


def phi_sub(x, y):
    return (x[0] - y[0], x[1] - y[1])


def phi_scale(c, x):
    return (c * x[0], c * x[1])


def phi_square(x):
    a, b = x
    return (a * a + b * b, 2 * a * b + b * b)


def phi_float(x) -> float:
    return float(x[0]) + float(x[1]) * (1.0 + sqrt(5.0)) / 2.0


def phi_encode(x) -> list[str]:
    return [str(Fraction(v)) for v in x]


def phi_sign_array(a, b) -> np.ndarray:
    """Vectorized exact sign of a + b*phi for int64 arrays (values below 2^31)."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    c = 2 * a + b
    d = c * c - 5 * b * b
    mixed = np.sign(d) * np.where(c > 0, 1, -1)
    return np.where(b == 0, np.sign(c),
                    np.where((c >= 0) & (b > 0), 1,
                             np.where((c <= 0) & (b < 0), -1, mixed))).astype(np.int8)


def root_upper(x, denominator: int = 10 ** 6) -> Fraction:
    """Rational upper bound for sqrt(a + b*phi) on a fixed grid, proved by squaring."""
    if phi_sign(x) < 0:
        raise ValueError("negative radicand")
    if phi_sign(x) == 0:
        return Fraction(0)
    lo, hi = 0, denominator
    while phi_sign(phi_sub((Fraction(hi, denominator) ** 2, 0), x)) < 0:
        hi *= 2
    while hi - lo > 1:
        mid = (hi + lo) // 2
        if phi_sign(phi_sub((Fraction(mid, denominator) ** 2, 0), x)) >= 0:
            hi = mid
        else:
            lo = mid
    return Fraction(hi, denominator)


# --------------------------------------------------------------------------
# Golden orbit, source records, site graph
# --------------------------------------------------------------------------


def fibonacci(n: int) -> tuple[int, int]:
    if type(n) is not int or n < 3:
        raise ValueError("finite levels are integers of at least 3")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a, b


def ceil_sqrt(q: int) -> int:
    return isqrt(q) + (isqrt(q) ** 2 < q)


def orbit(q: int) -> list[tuple[int, int]]:
    """xi_b = b*phi - floor(b*phi) as (m, b) with m = -floor(b*phi)."""
    return [(-((b + isqrt(5 * b * b)) // 2), b) for b in range(q)]


def axis_tables(values) -> tuple[np.ndarray, np.ndarray]:
    """Squared differences (xi_i - xi_j)^2 = A + B*phi, in units of L^2."""
    m = np.array([v[0] for v in values], dtype=np.int64)
    b = np.array([v[1] for v in values], dtype=np.int64)
    da = m[:, None] - m[None, :]
    db = b[:, None] - b[None, :]
    return da * da + db * db, 2 * da * db + db * db


def source_control(a, b):
    return (b[1] - a[0], b[1] + a[0], b[2] - a[1], b[2] + a[1], b[0] - a[2], b[0] + a[2])


def source_currents(z):
    if sum(z) % 2:
        raise ValueError("source current section requires an even-sum integer six-vector")
    h = sum(z) // 2
    return (-z[5], h - z[0], h - z[2] - z[3], h - z[1] - z[4] - z[5], h - z[2] - z[3] - z[4], z[3])


def source_records(q: int, values, sites: np.ndarray) -> tuple[list, list[int]]:
    """RER's [labels, z(b), currents] records for every site of the cube."""
    m = np.array([v[0] for v in values], dtype=np.int64)
    b = sites.astype(np.int64)
    a = m[b]
    z = np.stack([b[:, 1] - a[:, 0], b[:, 1] + a[:, 0], b[:, 2] - a[:, 1],
                  b[:, 2] + a[:, 1], b[:, 0] - a[:, 2], b[:, 0] + a[:, 2]], axis=1)
    total = z.sum(axis=1)
    if np.any(total % 2):
        raise ValueError("nonconservative source record")
    h = total // 2
    currents = np.stack([-z[:, 5], h - z[:, 0], h - z[:, 2] - z[:, 3],
                         h - z[:, 1] - z[:, 4] - z[:, 5], h - z[:, 2] - z[:, 3] - z[:, 4],
                         z[:, 3]], axis=1)
    lengths = np.abs(currents).sum(axis=1)
    records = [[bb, zz, cc] for bb, zz, cc in zip(b.tolist(), z.tolist(), currents.tolist())]
    return records, lengths.tolist()


def site_coordinates(q: int, dim: int) -> np.ndarray:
    """All labels of [0,q)^dim in product order; index = sum b_i q^(dim-1-i)."""
    return np.indices((q,) * dim).reshape(dim, -1).T.copy()


def build_site_graph(q: int, dim: int, chunk: int = 256):
    """CSR neighbour lists including the same-site read, rows sorted ascending.

    Two sites are neighbours iff q*(sum of squared coordinate differences) <= 1
    in Q(phi), decided by the exact sign rule.  Per-axis candidates prune the
    box; the decision on the box is the full three-term sum.
    """
    values = orbit(q)
    A, B = axis_tables(values)
    within = phi_sign_array(q * A - 1, q * B) <= 0
    width = int(within.sum(axis=1).max())
    cand = np.full((q, width), q, dtype=np.int64)
    for x in range(q):
        idx = np.flatnonzero(within[x])
        cand[x, :len(idx)] = idx
    valid = cand < q
    ci = np.where(valid, cand, 0)
    rows_ = np.arange(q)[:, None]
    Ax, Bx = A[rows_, ci], B[rows_, ci]
    sites = site_coordinates(q, dim)
    n = q ** dim
    powers = [q ** (dim - 1 - i) for i in range(dim)]
    indptr = np.zeros(n + 1, dtype=np.int64)
    pieces = []
    position = 0
    for lo in range(0, n, chunk):
        S = sites[lo:lo + chunk]
        Asum = np.zeros((len(S),) + (width,) * dim, dtype=np.int64)
        Bsum = np.zeros_like(Asum)
        ok = np.ones_like(Asum, dtype=bool)
        for axis in range(dim):
            shape = (len(S),) + tuple(width if i == axis else 1 for i in range(dim))
            Asum += Ax[S[:, axis]].reshape(shape)
            Bsum += Bx[S[:, axis]].reshape(shape)
            ok &= valid[S[:, axis]].reshape(shape)
        ok &= phi_sign_array(q * Asum - 1, q * Bsum) <= 0
        found = np.nonzero(ok)
        rows = found[0]
        nb = np.zeros(len(rows), dtype=np.int64)
        for axis in range(dim):
            nb += ci[S[rows, axis], found[1 + axis]] * powers[axis]
        counts = np.bincount(rows, minlength=len(S))
        pieces.append(nb.astype(np.int32))
        indptr[lo + 1:lo + 1 + len(S)] = position + np.cumsum(counts)
        position += int(counts.sum())
    indices = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int32)
    # indptr is int64, so the number of entries may exceed 2**31; the entries are site ids.
    if n >= 2 ** 31:
        raise ValueError("site ids exceed the int32 neighbour index range")
    return indptr, indices, values, A, B, sites


def neighbor_rows(indptr, indices, lo: int, hi: int) -> bytes:
    parts = []
    for i in range(lo, hi):
        parts.append("[" + ",".join(map(str, indices[indptr[i]:indptr[i + 1]].tolist())) + "]")
    return ",".join(parts).encode("ascii")


def _digest_rows(bounds):
    return neighbor_rows(_G["indptr"], _G["indices"], bounds[0], bounds[1])


def neighbor_digest(n: int, pool=None) -> str:
    """sha256 of the canonical JSON of the neighbour lists, streamed."""
    bounds = [(lo, min(lo + DIGEST_ROWS, n)) for lo in range(0, n, DIGEST_ROWS)]
    h = hashlib.sha256()
    h.update(b"[")
    parts = pool.imap(_digest_rows, bounds) if pool is not None else map(_digest_rows, bounds)
    for i, part in enumerate(parts):
        if i:
            h.update(b",")
        h.update(part)
    h.update(b"]\n")
    return h.hexdigest()


# --------------------------------------------------------------------------
# Breadth-first search with exact cone pruning
# --------------------------------------------------------------------------


FRONTIER_CHUNK = 1 << 26  # neighbour-list entries gathered per step of a frontier expansion (0.5 GB of int64 positions)


def _ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Concatenated index ranges [starts_i, starts_i + lengths_i), lengths > 0; int64 positions."""
    cl = np.cumsum(lengths)
    total = int(cl[-1])
    idx = np.ones(total, dtype=np.int64)
    idx[0] = starts[0]
    if len(starts) > 1:
        idx[cl[:-1]] = starts[1:] - (starts[:-1] + lengths[:-1]) + 1
    np.cumsum(idx, out=idx)
    return idx


def _expand(indptr: np.ndarray, indices: np.ndarray, frontier: np.ndarray, mask: np.ndarray) -> None:
    """Mark in ``mask`` every neighbour of the frontier sites, gathering at most FRONTIER_CHUNK entries at a time.

    The neighbour table may hold more than 2**31 entries, so every position is int64, and the
    frontier is expanded in slices so that no gather materializes the whole table.
    """
    st = np.asarray(indptr[frontier], dtype=np.int64)
    ln = np.asarray(indptr[frontier + 1], dtype=np.int64) - st
    cl = np.cumsum(ln)
    lo = 0
    while lo < len(frontier):
        base = int(cl[lo - 1]) if lo else 0
        hi = max(int(np.searchsorted(cl, base + FRONTIER_CHUNK, side="right")), lo + 1)
        mask[indices[_ranges(st[lo:hi], ln[lo:hi])]] = True
        lo = hi


def bfs(indptr, indices, start: int, n: int, cap: int | None = None, allowed=None) -> np.ndarray:
    """Graph distances from ``start``; -1 beyond ``cap`` layers.

    ``allowed[m]`` restricts the frontier expanded at layer ``m``; callers use
    it with the cone regions that contain every shortest path to the targets
    they count, so the returned distances are exact for those targets and
    never smaller than the true distance elsewhere.
    """
    dist = np.full(n, -1, dtype=np.int16)
    dist[start] = 0
    frontier = np.array([start], dtype=np.int64)
    layers = n if cap is None else cap
    for m in range(layers):
        if allowed is not None:
            frontier = frontier[allowed[m][frontier]]
        if frontier.size == 0:
            break
        mask = np.zeros(n, dtype=bool)
        _expand(indptr, indices, frontier, mask)
        mask &= dist < 0
        frontier = np.flatnonzero(mask)
        if frontier.size == 0:
            break
        dist[frontier] = m + 1
    return dist


def metric_tables(q: int, A, B, sites: np.ndarray, site: int):
    """Exact squared metric distance (A + B*phi, units L^2) from ``site`` to all sites."""
    dim = sites.shape[1]
    coords = sites[site]
    At = np.zeros(len(sites), dtype=np.int64)
    Bt = np.zeros(len(sites), dtype=np.int64)
    for axis in range(dim):
        At += A[coords[axis]][sites[:, axis]]
        Bt += B[coords[axis]][sites[:, axis]]
    return At, Bt


def cone_masks(q: int, At, Bt, K: int) -> np.ndarray:
    """cone[k][t] iff q*|t - site|^2 <= k^2, i.e. |t - site| <= k*a_q, exactly."""
    return np.stack([phi_sign_array(q * At - k * k, q * Bt) <= 0 for k in range(K + 1)])


# --------------------------------------------------------------------------
# Layer-pair weights and per-start pair counts
# --------------------------------------------------------------------------


def gap_table(K: int) -> np.ndarray:
    """G[delta, j, j'] = 1 iff j' - j >= max(delta, 1); delta = K+1 is the unordered bin."""
    j = np.arange(K + 1)
    diff = j[None, :] - j[:, None]
    return np.stack([(diff >= max(d, 1)) for d in range(K + 2)]).astype(np.int64)


def vertical_weights(K: int) -> np.ndarray:
    """W[k, alpha, beta, delta]: layer pairs (j, j') of the k-layer vertical diamond."""
    G = gap_table(K)
    j = np.arange(K + 1)
    a = np.arange(K + 1)
    out = np.zeros((K + 1, K + 1, K + 1, K + 2), dtype=np.int64)
    for k in range(K + 1):
        J = ((a[:, None] <= j[None, :]) & (j[None, :] <= k - a[:, None])).astype(np.int64)
        out[k] = np.einsum("aj,bl,djl->abd", J, J, G)
    return out


def moving_weights(K: int) -> np.ndarray:
    """W[alpha_s, gamma_s, alpha_t, gamma_t, delta] for tips (0, x), (K, y)."""
    G = gap_table(K)
    j = np.arange(K + 1)
    a = np.arange(K + 1)
    J = ((a[:, None, None] <= j[None, None, :]) & (j[None, None, :] <= K - a[None, :, None])).astype(np.int64)
    return np.einsum("agj,bhl,djl->agbhd", J, J, G)


_G: dict = {}


def _worker_init(payload: dict) -> None:
    _G.clear()
    _G.update(payload)
    _G["contexts"] = {}
    if "indptr_path" in payload:
        _G["indptr"] = np.load(payload["indptr_path"], mmap_mode="r")
        _G["indices"] = np.load(payload["indices_path"], mmap_mode="r")


def save_context(path: Path, **arrays) -> str:
    """Interval context (anchors, cones, supports, weights) for the per-start runs."""
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
    return str(path)


def _context(key: str) -> dict:
    contexts = _G.setdefault("contexts", {})
    if key not in contexts:
        with np.load(key, allow_pickle=False) as z:
            contexts[key] = {k: z[k] for k in z.files}
    return contexts[key]


def _run_starts(task) -> dict:
    """Exact per-start ordered-pair counts for the vertical and moving intervals.

    ``alpha`` is the graph distance from the anchor ``x`` of the interval,
    ``gamma`` the distance to the upper tip ``y`` (moving intervals only).
    """
    key, starts = task
    c = _context(key)
    K = int(c["K"])
    n = int(c["n"])
    indptr, indices = _G["indptr"], _G["indices"]
    alpha_all = c["alpha"]
    x = int(c["x"])
    y = int(c["y"])
    gamma_all = c["gamma"] if y >= 0 else None
    cone_x = c["cone_x"]
    cone_y = c["cone_y"] if y >= 0 else None
    vs, va = c["vertical_sites"], c["vertical_alpha"]
    ms, ma, mg = c["moving_sites"], c["moving_alpha"], c["moving_gamma"]
    Wv, Wm = c["vertical_weights"], c["moving_weights"]
    bins = K + 2
    hist = np.zeros((K + 1, K + 1, bins), dtype=np.int64)
    vertical = np.zeros((len(starts), K + 1), dtype=np.int64)
    moving = np.zeros(len(starts), dtype=np.int64)
    for i, s in enumerate(starts):
        s = int(s)
        alpha_s = int(alpha_all[s])
        gamma_s = int(gamma_all[s]) if gamma_all is not None else 0
        cap = K - alpha_s - 1
        if cap > 0:
            allowed = [cone_x[K - alpha_s - m] if cone_y is None
                       else (cone_x[K - alpha_s - m] | cone_y[K - alpha_s - m]) for m in range(cap)]
            dist = bfs(indptr, indices, s, n, cap=cap, allowed=allowed)
        else:
            dist = np.full(n, -1, dtype=np.int16)
            dist[s] = 0
        dist[x] = alpha_s
        if y >= 0:
            dist[y] = gamma_s
        if len(vs):
            dv = dist[vs].astype(np.int64)
            dv[dv < 0] = K + 1
            hv = np.bincount(va * bins + dv, minlength=(K + 1) * bins).reshape(K + 1, bins)
            hist[alpha_s] += hv
            vertical[i] = np.einsum("bd,kbd->k", hv, Wv[:, alpha_s])
        if len(ms) and alpha_s + gamma_s <= K:
            dm = dist[ms].astype(np.int64)
            dm[dm < 0] = K + 1
            hm = np.bincount((ma * (K + 1) + mg) * bins + dm,
                             minlength=(K + 1) * (K + 1) * bins).reshape(K + 1, K + 1, bins)
            moving[i] = int((hm * Wm[alpha_s, gamma_s]).sum())
    return {"starts": [int(s) for s in starts], "vertical": vertical, "moving": moving, "hist": hist}


# --------------------------------------------------------------------------
# Continuum references
# --------------------------------------------------------------------------


def diamond_volume(dim: int, tau: float) -> float:
    """Volume of the flat diamond of proper duration tau in dim+1 dimensions (c = 1)."""
    return UNIT_BALL_VOLUME[dim] * tau ** (dim + 1) / (2 ** dim * (dim + 1))


def _segment(d: float, r: float) -> float:
    """Area of the disc of radius r beyond a chord at distance d >= 0 from the centre."""
    if d >= r:
        return 0.0
    return r * r * acos(d / r) - d * sqrt(r * r - d * d)


def _corner(d1: float, d2: float, r: float) -> float:
    """Area of {p in disc(r): p_1 >= d1, p_2 >= d2} for d1, d2 >= 0."""
    if d1 * d1 + d2 * d2 >= r * r:
        return 0.0
    x1 = sqrt(r * r - d2 * d2)

    def F(x):
        return x * sqrt(max(r * r - x * x, 0.0)) / 2.0 + r * r * asin(min(x / r, 1.0)) / 2.0

    return F(x1) - F(d1) - d2 * (x1 - d1)


def ball_box_volume(center, r: float, dim: int) -> float:
    """Volume of the ball of radius r about ``center`` inside the unit cube [0,1]^dim."""
    if r <= 0.0:
        return 0.0
    if dim == 1:
        return max(0.0, min(center[0] + r, 1.0) - max(center[0] - r, 0.0))
    if dim == 2:
        d = [center[0], 1.0 - center[0], center[1], 1.0 - center[1]]
        area = pi * r * r - sum(_segment(v, r) for v in d)
        for i in (0, 1):
            for j in (2, 3):
                area += _corner(d[i], d[j], r)
        return area
    lo, hi = max(-r, -center[2]), min(r, 1.0 - center[2])
    if hi <= lo:
        return 0.0
    value, _ = integrate.quad(lambda z: ball_box_volume(center[:2], sqrt(max(r * r - z * z, 0.0)), 2),
                              lo, hi, limit=200)
    return value


def clipped_diamond_volume(center, T: float, dim: int) -> float:
    """Volume of the vertical diamond of duration T about ``center`` clipped to the cube."""
    breaks = sorted({min(v, T / 2.0) for c in center for v in (c, 1.0 - c)} | {T / 2.0})
    breaks = [v for v in breaks if 0.0 < v < T / 2.0]
    value, _ = integrate.quad(lambda r: ball_box_volume(center, r, dim), 0.0, T / 2.0,
                              points=breaks or None, limit=200)
    return 2.0 * value


def ellipsoid_buffer(px, py, T: float, dim: int) -> float:
    """Smallest face clearance of the diamond's spatial projection inside the cube.

    The projection of the diamond with tips (0, x), (T, y) is the ellipsoid
    |z - x| + |z - y| <= T with semi-axis T/2 along the tips and
    sqrt(T^2 - l^2)/2 transversally.
    """
    px, py = np.asarray(px, dtype=float), np.asarray(py, dtype=float)
    mid = (px + py) / 2.0
    delta = py - px
    ell = float(np.linalg.norm(delta))
    if ell >= T:
        return float("-inf")
    u = delta / ell if ell > 0 else np.zeros(dim)
    b2 = (T * T - ell * ell) / 4.0
    w = np.sqrt((T / 2.0) ** 2 * u ** 2 + b2 * (1.0 - u ** 2))
    return float(min(np.min(mid - w), np.min(1.0 - mid - w)))


def vertical_bound(T: float, delta: float, H: float, h_over_a: float) -> float:
    """Eq. source-net-volume-error with c = 1 (units of L^4)."""
    return 4.0 * pi * (T + delta) * (T / 2.0 + H) ** 2 * (H + T * h_over_a) + pi * T ** 3 * delta / 2.0


def general_bound(T: float, delta: float, H: float, h_over_a: float) -> float:
    """Eq. source-net-general-volume-error with c = 1 (units of L^4)."""
    return 8.0 * pi * (T + delta) * (T + H) ** 2 * (H + 2.0 * T * h_over_a) + 4.0 * pi * T ** 3 * delta


def ordering_summary(C: Fraction, N: int, dim: int) -> dict:
    if N < 2:
        return {"ordering_fraction": None, "ordering_fraction_float": None,
                "distance_to_reference": None, "myrheim_meyer_dimension": None}
    f = Fraction(2) * C / (N * (N - 1))
    ref = REFERENCE_FRACTION[dim]
    dim_est = invert_myrheim_meyer_fraction(float(f))
    return {"ordering_fraction": str(f), "ordering_fraction_float": rounded(float(f)),
            "distance_to_reference": rounded(float(f - ref)),
            "myrheim_meyer_dimension": None if dim_est is None else rounded(dim_est)}


# --------------------------------------------------------------------------
# Stratified sampling of start sites
# --------------------------------------------------------------------------


def allocate_sample(strata: dict, seed: int) -> dict:
    """Proportional allocation with at least two starts per stratum, seeded."""
    rng = np.random.default_rng(seed)
    total = sum(len(v) for v in strata.values())
    chosen = {}
    for key in sorted(strata):
        members = np.sort(np.asarray(strata[key], dtype=np.int64))
        size = min(len(members), max(min(len(members), 2), -(-SAMPLE_SIZE * len(members) // total)))
        chosen[key] = np.sort(rng.choice(members, size=size, replace=False))
    return chosen


def stratified_estimate(strata: dict, chosen: dict, values: dict) -> dict:
    """Horvitz-Thompson estimate under stratified sampling without replacement."""
    estimate = Fraction(0)
    variance = 0.0
    rows = []
    sample_size = 0
    for key in sorted(strata):
        Nh = len(strata[key])
        picked = chosen[key]
        nh = len(picked)
        vals = [int(values[int(s)]) for s in picked]
        S = sum(vals)
        Q = sum(v * v for v in vals)
        estimate += Fraction(Nh * S, nh)
        s2 = (Q - S * S / nh) / (nh - 1) if nh > 1 else 0.0
        variance += Nh * Nh * (1.0 - nh / Nh) * s2 / nh
        sample_size += nh
        rows.append({"stratum": list(key) if isinstance(key, tuple) else key,
                     "population": Nh, "sample": nh, "sum": S, "sum_of_squares": Q})
    return {"value": str(estimate), "float": rounded(float(estimate)),
            "standard_error": rounded(sqrt(variance)), "sample_size": sample_size,
            "strata": rows}


# --------------------------------------------------------------------------
# One family (dimension) at one level
# --------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def build_family(n: int, dim: int, processes: int = 1, workdir: Path | None = None,
                 exact_support_limit: int = EXACT_SUPPORT_LIMIT) -> dict:
    t0 = time.time()
    q, p = fibonacci(n)
    K = ceil_sqrt(q)
    indptr, indices, values, A, B, sites = build_site_graph(q, dim)
    count = q ** dim
    degrees = np.diff(indptr)
    if int(degrees.min()) < 1:
        raise ValueError("a site lost its same-site read")
    perm = [(b * p) % q for b in range(q)]
    ordered = sorted(range(q), key=lambda b: perm[b])
    for b in range(q):
        error = phi_sub(values[b], (Fraction(perm[b], q), 0))
        if phi_sign(phi_sub((Fraction(1, q * q), 0), phi_square(error))) <= 0:
            raise ValueError("golden orbit left its rational cell")
    radii = [phi_sub((1, 0), values[ordered[-1]])]
    radii += [phi_scale(Fraction(1, 2), phi_sub(values[b], values[a])) for a, b in zip(ordered, ordered[1:])]
    radius = radii[0]
    for r in radii[1:]:
        if phi_sign(phi_sub(r, radius)) > 0:
            radius = r
    h_squared = phi_scale(dim, phi_square(radius))
    ratio_upper = root_upper(phi_scale(q, h_squared))
    inner_speed = max(Fraction(0), 1 - 2 * ratio_upper)
    center_axis = 0
    for b in range(1, q):
        if phi_sign(phi_sub(phi_square(phi_sub(values[b], (Fraction(1, 2), 0))),
                            phi_square(phi_sub(values[center_axis], (Fraction(1, 2), 0))))) < 0:
            center_axis = b
    center = sum(center_axis * q ** (dim - 1 - i) for i in range(dim))
    clearance = values[center_axis]
    if phi_sign(phi_sub(phi_sub((1, 0), clearance), clearance)) < 0:
        clearance = phi_sub((1, 0), clearance)

    # Graph for the workers: memory-mapped files for the pool, arrays in-process.
    pool = None
    payload = {"K": K, "n": count}
    tag = f"{dim}_{q}"
    if processes > 1 and count >= POOL_MINIMUM_SITES:
        np.save(workdir / f"indptr_{tag}.npy", indptr)
        np.save(workdir / f"indices_{tag}.npy", indices)
        payload["indptr_path"] = str(workdir / f"indptr_{tag}.npy")
        payload["indices_path"] = str(workdir / f"indices_{tag}.npy")
    else:
        payload["indptr"], payload["indices"] = indptr, indices
    _worker_init(payload)
    if "indptr_path" in payload:
        pool = mp.get_context("spawn").Pool(processes, initializer=_worker_init, initargs=(payload,))
    try:
        neighbor_sha = neighbor_digest(count, pool)
        t_graph = time.time() - t0

        # Probes from the corner, the centre and the last site, as in RER.
        probes = []
        distance_from = {}
        for start in sorted({0, center, count - 1}):
            dist = bfs(indptr, indices, start, count)
            if int(dist.min()) < 0:
                raise ValueError("disconnected site graph")
            distance_from[start] = dist
            At, Bt = metric_tables(q, A, B, sites, start)
            cones = cone_masks(q, At, Bt, K)
            for k in range(1, K + 1):
                reached = dist <= k
                outer = int(np.count_nonzero(reached & ~cones[k]))
                missing = np.flatnonzero(cones[k] & ~reached)
                inner_misses = 0
                if inner_speed > 0:
                    bound = k * k * inner_speed * inner_speed
                    for i in missing.tolist():
                        if phi_sign(phi_sub(phi_scale(q, (int(At[i]), int(Bt[i]))), (bound, 0))) <= 0:
                            inner_misses += 1
                if outer or inner_misses:
                    raise ValueError("cone sandwich violated")
                reachable = np.flatnonzero(reached).tolist()
                probes.append({"start": start, "layers": k, "reachable_count": len(reachable),
                               "reachable_ids_sha256": digest(reachable),
                               "outer_cone_violations": outer, "certified_inner_cone_misses": inner_misses,
                               "exact_finite_cone_missing_count": int(len(missing)),
                               "missing_ids_sha256": digest(missing.tolist())})
        alpha = distance_from[center]
        shell = np.bincount(alpha, minlength=K + 1)[:K + 1]

        # Geometry in units of L.
        a_q = 1.0 / sqrt(q)
        delta_q = a_q
        H_q = 2.0 * sqrt(3.0) / q if dim == 3 else None
        h_over_a = sqrt(q * phi_float(h_squared))
        density = q ** (dim + 0.5)
        xi = np.array([phi_float(v) for v in values])
        center_position = [xi[center_axis]] * dim
        clearance_float = phi_float(clearance)

        # Moving-tip diamond: tips symmetric about the centre on the rank diagonal.
        rank_center = perm[center_axis]
        moving = None
        rules = [("buffer_at_least_H_q", H_q), ("buffer_positive", 0.0)] if dim == 3 else [("buffer_positive", 0.0)]
        for rule, floor_ in rules:
            for s in range(1, q):
                lo, hi = rank_center - s, rank_center + s
                if lo < 0 or hi >= q:
                    break
                xl, yl = ordered[lo], ordered[hi]
                ell2 = phi_scale(dim, (int(A[xl][yl]), int(B[xl][yl])))
                if phi_sign(phi_sub(phi_scale(q, ell2), (K * K, 0))) >= 0:
                    continue
                buffer = ellipsoid_buffer([xi[xl]] * dim, [xi[yl]] * dim, K * a_q, dim)
                if buffer > 0.0 and buffer >= floor_:
                    moving = {"rule": rule, "shift": s, "x_axis": xl, "y_axis": yl,
                              "x": sum(xl * q ** (dim - 1 - i) for i in range(dim)),
                              "y": sum(yl * q ** (dim - 1 - i) for i in range(dim)),
                              "ell2": ell2, "buffer": buffer}
                    break
            if moving is not None:
                break
        if moving is not None:
            moving_x = moving["x"]
            alpha_m = distance_from[moving_x] if moving_x in distance_from else bfs(indptr, indices, moving_x, count)
            gamma_m = bfs(indptr, indices, moving["y"], count)
            At_x, Bt_x = metric_tables(q, A, B, sites, moving_x)
            At_y, Bt_y = metric_tables(q, A, B, sites, moving["y"])
            cone_mx = cone_masks(q, At_x, Bt_x, K)
            cone_my = cone_masks(q, At_y, Bt_y, K)
        # The vertical diamond is anchored at the centre, the moving diamond at
        # its own lower tip; each interval runs with its own anchor context.
        t1 = time.time()
        vertical_sites = np.flatnonzero(alpha <= K // 2)
        Wv = vertical_weights(K)
        Wm = moving_weights(K)
        At_c, Bt_c = metric_tables(q, A, B, sites, center)
        cone_c = cone_masks(q, At_c, Bt_c, K)
        empty = np.zeros(0, dtype=np.int64)

        def run(key: str, starts: np.ndarray) -> dict:
            results = {}
            hist = np.zeros((K + 1, K + 1, K + 2), dtype=np.int64)
            if len(starts) == 0:
                return {"values": results, "hist": hist}
            tasks = [(key, starts[i:i + BATCH_SIZE]) for i in range(0, len(starts), BATCH_SIZE)]
            outputs = pool.imap_unordered(_run_starts, tasks) if pool is not None else map(_run_starts, tasks)
            for out in outputs:
                hist += out["hist"]
                for i, s_ in enumerate(out["starts"]):
                    results[s_] = (out["vertical"][i].tolist(), int(out["moving"][i]))
            return {"values": results, "hist": hist}

        # Vertical diamonds at the centre (all k <= K share the K-support run).
        vertical_key = save_context(
            workdir / f"vertical_{tag}.npz", K=K, n=count, alpha=alpha, gamma=empty, cone_x=cone_c,
            cone_y=empty, x=center, y=-1, vertical_sites=vertical_sites,
            vertical_alpha=alpha[vertical_sites].astype(np.int64), moving_sites=empty,
            moving_alpha=empty, moving_gamma=empty, vertical_weights=Wv, moving_weights=Wm)
        vertical_exact = len(vertical_sites) <= exact_support_limit
        if vertical_exact:
            v_run = run(vertical_key, vertical_sites)
            v_sample = None
        else:
            strata = {int(a_): vertical_sites[alpha[vertical_sites] == a_].tolist()
                      for a_ in range(K // 2 + 1) if np.any(alpha[vertical_sites] == a_)}
            v_sample = allocate_sample(strata, SAMPLE_SEED + 1000 * dim + q)
            v_starts = np.sort(np.concatenate([v for v in v_sample.values()]))
            v_run = run(vertical_key, v_starts)
        t_vertical = time.time() - t1

        # Moving-tip diamond.
        t2 = time.time()
        m_run = None
        m_sample = None
        moving_sites = np.zeros(0, dtype=np.int64)
        if moving is not None:
            moving_sites = np.flatnonzero(alpha_m + gamma_m <= K)
            moving_key = save_context(
                workdir / f"moving_{tag}.npz", K=K, n=count, alpha=alpha_m, gamma=gamma_m,
                cone_x=cone_mx, cone_y=cone_my, x=moving_x, y=moving["y"], vertical_sites=empty,
                vertical_alpha=empty, moving_sites=moving_sites,
                moving_alpha=alpha_m[moving_sites].astype(np.int64),
                moving_gamma=gamma_m[moving_sites].astype(np.int64),
                vertical_weights=Wv, moving_weights=Wm)
            moving_exact = len(moving_sites) <= exact_support_limit
            if moving_exact:
                m_run = run(moving_key, moving_sites)
            else:
                keys = {}
                for s_ in moving_sites.tolist():
                    keys.setdefault((int(alpha_m[s_]), int(gamma_m[s_])), []).append(s_)
                m_sample = allocate_sample(keys, SAMPLE_SEED + 1000 * dim + q + 500)
                m_starts = np.sort(np.concatenate([v for v in m_sample.values()]))
                m_run = run(moving_key, m_starts)
        t_moving = time.time() - t2
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # Vertical statistics for every k <= K.
    intervals = []
    for k in range(1, K + 1):
        per_layer = [int(np.count_nonzero(alpha <= min(j, k - j))) for j in range(k + 1)]
        N = sum(per_layer)
        T = k * a_q
        inside = phi_sign(phi_sub(phi_scale(4 * q, phi_square(clearance)), (k * k, 0))) >= 0
        volume = diamond_volume(dim, T)
        clipped = clipped_diamond_volume(center_position, T, dim)
        normalized = N / density
        row = {"layers": k, "inclusive_event_count": N, "counts_by_layer": per_layer,
               "continuum_diamond_inside_cube": inside,
               "count_over_L4_times_sqrt_q": str(Fraction(N, q ** dim)) if dim == 3 else None,
               "diamond_volume_over_pi_L4": str(Fraction(k ** 4, 24 * q * q)) if dim == 3 else None,
               "model_time_T_over_L": rounded(T),
               "normalized_count": rounded(normalized),
               "continuum_diamond_volume": rounded(volume),
               "clipped_diamond_volume": rounded(clipped),
               "deviation_from_continuum": rounded(normalized - volume),
               "relative_deviation_from_continuum": rounded(normalized / volume - 1.0),
               "relative_deviation_from_clipped": rounded(normalized / clipped - 1.0)}
        if dim == 3:
            bound = vertical_bound(T, delta_q, H_q, h_over_a)
            row.update({"volume_error_bound": rounded(bound),
                        "volume_error_bound_relative": rounded(bound / volume),
                        "bound_hypothesis_ball_inside_cube": bool(clearance_float >= T / 2.0 + H_q),
                        "actual_deviation_within_bound": bool(abs(normalized - volume) <= bound)})
        if vertical_exact:
            C = Fraction(int(sum(v[0][k] for v in v_run["values"].values())))
            row["pair_counting"] = "exact_all_pairs"
            row["strict_pair_count"] = int(C)
            row["ordering_fraction_standard_error"] = None
        else:
            est = stratified_estimate({a_: strata[a_] for a_ in strata},
                                      v_sample, {s_: v[0][k] for s_, v in v_run["values"].items()})
            C = Fraction(est["value"])
            row["pair_counting"] = "stratified_sample"
            row["strict_pair_count_estimate"] = est
            row["ordering_fraction_standard_error"] = (rounded(2.0 * est["standard_error"] / (N * (N - 1)))
                                                       if N > 1 else None)
        row.update(ordering_summary(C, N, dim))
        intervals.append(row)
    histogram = v_run["hist"][:K // 2 + 1, :K // 2 + 1, :].tolist() if vertical_exact else None

    # Moving-tip statistics.
    moving_row = None
    if moving is not None:
        T = K * a_q
        ell2 = phi_float(moving["ell2"])
        tau = sqrt(T * T - ell2)
        per_layer = [int(np.count_nonzero((alpha_m <= j) & (gamma_m <= K - j))) for j in range(K + 1)]
        N = sum(per_layer)
        volume = diamond_volume(dim, tau)
        normalized = N / density
        moving_row = {"layers": K, "tip_selection_rule": moving["rule"], "rank_shift": moving["shift"],
                      "x_axis_label": moving["x_axis"], "y_axis_label": moving["y_axis"],
                      "x_site": moving["x"], "y_site": moving["y"],
                      "tip_separation_squared_over_L2_Qphi": phi_encode(moving["ell2"]),
                      "tip_separation_over_T": rounded(sqrt(ell2) / T),
                      "proper_duration_over_L": rounded(tau),
                      "spatial_buffer_over_L": rounded(moving["buffer"]),
                      "spatial_buffer_at_least_H_q": bool(H_q is not None and moving["buffer"] >= H_q),
                      "inclusive_event_count": N, "counts_by_layer": per_layer,
                      "support_site_count": int(len(moving_sites)),
                      "normalized_count": rounded(normalized),
                      "continuum_diamond_volume": rounded(volume),
                      "deviation_from_continuum": rounded(normalized - volume),
                      "relative_deviation_from_continuum": rounded(normalized / volume - 1.0)}
        if dim == 3:
            bound = general_bound(T, delta_q, H_q, h_over_a)
            moving_row.update({"volume_error_bound": rounded(bound),
                               "volume_error_bound_relative": rounded(bound / volume),
                               "actual_deviation_within_bound": bool(abs(normalized - volume) <= bound)})
        if m_sample is None:
            C = Fraction(int(sum(v[1] for v in m_run["values"].values())))
            moving_row["pair_counting"] = "exact_all_pairs"
            moving_row["strict_pair_count"] = int(C)
            moving_row["ordering_fraction_standard_error"] = None
        else:
            est = stratified_estimate(keys, m_sample, {s_: v[1] for s_, v in m_run["values"].items()})
            C = Fraction(est["value"])
            moving_row["pair_counting"] = "stratified_sample"
            moving_row["strict_pair_count_estimate"] = est
            moving_row["ordering_fraction_standard_error"] = (
                rounded(2.0 * est["standard_error"] / (N * (N - 1))) if N > 1 else None)
        moving_row.update(ordering_summary(C, N, dim))

    # Count clock between the K-layer and the floor(K/2)-layer vertical diamonds.
    # The volume clock of a flat diamond in dim+1 dimensions is the (dim+1)-th root of the
    # count ratio: the paper's fourth root for the source net, cube and square roots for the
    # two- and one-dimensional control populations.
    Kp = K // 2
    root = 1.0 / (dim + 1)
    NI, NJ = intervals[K - 1]["inclusive_event_count"], intervals[Kp - 1]["inclusive_event_count"]
    clock = {"reference_layers": Kp, "interval_layers": K,
             "interval_count": NI, "reference_count": NJ, "clock_exponent": f"1/{dim + 1}",
             "count_clock": rounded((NI / NJ) ** root), "model_time_ratio": rounded(K / Kp),
             "relative_deviation": rounded((NI / NJ) ** root / (K / Kp) - 1.0)}
    if dim == 3:
        EI = intervals[K - 1]["volume_error_bound"] * density
        EJ = intervals[Kp - 1]["volume_error_bound"] * density
        lower = (max(NI - EI, 0.0) / (NJ + EJ)) ** 0.25
        upper = ((NI + EI) / (NJ - EJ)) ** 0.25 if NJ > EJ else None
        clock.update({"enclosure_lower": rounded(lower),
                      "enclosure_upper": None if upper is None else rounded(upper),
                      "enclosure_hypotheses_satisfied": bool(
                          intervals[K - 1]["bound_hypothesis_ball_inside_cube"]
                          and intervals[Kp - 1]["bound_hypothesis_ball_inside_cube"] and NJ > EJ),
                      "count_error_interval": rounded(EI), "count_error_reference": rounded(EJ)})

    family = {"dimension": dim, "fibonacci_index": n, "q": q, "p": p, "layer_steps": K,
              "site_count": count, "record_count": count,
              "orbit_Qphi": [phi_encode(v) for v in values], "grid_permutation": perm,
              "centre_axis": center_axis, "intervention_source_id": center,
              "clearance_over_L_Qphi": phi_encode(clearance),
              "one_dimensional_fill_over_L_Qphi": phi_encode(radius),
              "whole_cube_fill_h_squared_over_L2_Qphi": phi_encode(h_squared),
              "h_over_a_upper": str(ratio_upper), "h_over_a": rounded(h_over_a),
              "covering_radius_h_over_L": rounded(sqrt(phi_float(h_squared))),
              "assignment_error_H_over_L": None if H_q is None else rounded(H_q),
              "edge_radius_a_over_L": rounded(a_q),
              "certified_inner_speed_lower": str(inner_speed),
              "positive_inner_cone": phi_sign(phi_sub((1, 0), phi_scale(4 * q, h_squared))) > 0,
              "radius_squared_over_L2": str(Fraction(1, q)), "layer_time_equals_radius": True,
              "event_density_times_L_to_dim_plus_one": rounded(density),
              "neighbors_including_wait_sha256": neighbor_sha,
              "undirected_spatial_edges": int((int(degrees.sum()) - count) // 2),
              "minimum_neighbor_count_including_wait": int(degrees.min()),
              "maximum_neighbor_count_including_wait": int(degrees.max()),
              "exact_width": count, "exact_height_in_events": K + 1,
              "reachability_probes": probes,
              "shell_counts_from_centre": shell.tolist(),
              "vertical_support_site_count": int(len(vertical_sites)),
              "vertical_pair_counting": "exact_all_pairs" if vertical_exact else "stratified_sample",
              "vertical_strict_pair_histogram": histogram,
              "vertical_intervals": intervals,
              "center_intervals": [{"layers": r["layers"], "inclusive_event_count": r["inclusive_event_count"],
                                    "counts_by_layer": r["counts_by_layer"],
                                    "continuum_diamond_inside_cube": r["continuum_diamond_inside_cube"],
                                    "count_over_L4_times_sqrt_q": r["count_over_L4_times_sqrt_q"],
                                    "diamond_volume_over_pi_L4": r["diamond_volume_over_pi_L4"],
                                    "comparison_role": "finite_diagnostic_not_dimension_fit_or_convergence_test"}
                                   for r in intervals] if dim == 3 else None,
              "moving_tip_interval": moving_row,
              "count_clock": clock}
    if dim == 3:
        records, lengths = source_records(q, values, sites)
        family.update({"source_records_sha256": digest(records),
                       "source_record_examples": [records[i] for i in sorted({0, center, count - 1})],
                       "maximum_word_length": int(max(lengths)), "sum_word_lengths": int(sum(lengths)),
                       "word_length_bound": 27 * (q - 1),
                       "quadrature_displacement_H_squared_over_L2": str(Fraction(12, q * q))})
    if v_sample is not None:
        family["vertical_sample"] = {"seed": SAMPLE_SEED + 1000 * dim + q,
                                     "sampled_starts_sha256": digest(sorted(int(s) for s in v_starts)),
                                     "sample_size": int(len(v_starts))}
    if m_sample is not None:
        family["moving_sample"] = {"seed": SAMPLE_SEED + 1000 * dim + q + 500,
                                   "sampled_starts_sha256": digest(sorted(int(s) for s in m_starts)),
                                   "sample_size": int(len(m_starts))}
    _log(f"level n={n} q={q} dim={dim}: sites={count} nnz={int(degrees.sum())} "
         f"graph+digest={t_graph:.1f}s vertical={t_vertical:.1f}s "
         f"({'exact' if vertical_exact else 'sampled'}, {len(vertical_sites)} support sites) "
         f"moving={t_moving:.1f}s total={time.time() - t0:.1f}s")
    return family


# --------------------------------------------------------------------------
# RER cross-check and the receipt
# --------------------------------------------------------------------------


def rer_receipt() -> dict:
    path = RER_ROOT / RER_RECEIPT
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != RER_RECEIPT_SHA256:
        raise ValueError(f"RER receipt digest {sha} differs from the pinned {RER_RECEIPT_SHA256}")
    return json.loads(data)


def cross_check(family: dict, rer_level: dict) -> dict:
    fields = {}
    for name in RER_COMPARED_FIELDS:
        replay = family[name]
        theirs = rer_level[name]
        fields[name] = {"agree": canonical(replay) == canonical(theirs)}
        if name not in ("reachability_probes", "center_intervals", "orbit_Qphi", "source_record_examples"):
            fields[name]["rer"] = theirs
    return {"fibonacci_index": rer_level["fibonacci_index"], "q": rer_level["q"],
            "fields": fields, "all_agree": all(v["agree"] for v in fields.values())}


def build(levels=LEVELS, dimensions=DIMENSIONS, processes: int | None = None,
          exact_support_limit: int = EXACT_SUPPORT_LIMIT) -> dict:
    processes = processes or max(1, min(10, os.cpu_count() or 1))
    theirs = {lv["fibonacci_index"]: lv for lv in rer_receipt()["levels"]}
    rows = []
    checks = []
    with tempfile.TemporaryDirectory(prefix="oph_source_net_") as tmp:
        workdir = Path(tmp)
        for n in levels:
            families = []
            for dim in dimensions:
                fam = build_family(n, dim, processes=processes, workdir=workdir,
                                   exact_support_limit=exact_support_limit)
                if dim == 3 and n in theirs:
                    fam["rer_cross_check"] = cross_check(fam, theirs[n])
                    checks.append(fam["rer_cross_check"]["all_agree"])
                families.append(fam)
            rows.append({"fibonacci_index": n, "q": fibonacci(n)[0], "families": families})
    pins = {p: file_sha256(ROOT / p) for p in LOCAL_PINS}
    pins.update({"reverse-engineering-reality/" + p: file_sha256(RER_ROOT / p) for p in RER_PINS})
    return {"schema": SCHEMA, "scope": SCOPE, "claim_boundary": CLAIM_BOUNDARY,
            "source_scale_L_squared_Qphi": ["12/5", "-4/5"],
            "source_scale_L_over_one": "2/sqrt(phi+2)",
            "window": "[0,L]^dim in the source-Gram coordinate witness; dim = 3 is the source net, 2 and 1 are controls",
            "event_order": "(j,s) <= (j',t) iff graph distance d(s,t) <= j'-j on the site graph, waiting included",
            "sampling_measure": "L^dim/q^dim per site, layer weight a_q = L/sqrt(q); rho_q = q^(dim+1/2)/L^(dim+1) with c = 1",
            "finite_schedule": "K_q = ceil(sqrt(q)); T_q = K_q L/sqrt(q); reference clock diamond has floor(K_q/2) layers",
            "moving_tips": "tips symmetric about the centre on the rank diagonal, smallest shift with the declared buffer rule",
            "pair_counting": {"exact_support_limit": exact_support_limit, "sample_size_minimum": SAMPLE_SIZE,
                              "sample_seed_base": SAMPLE_SEED,
                              "allocation": "proportional to stratum size, at least two starts per stratum, without replacement",
                              "strata": "vertical: d(centre, s); moving: (d(x, s), d(s, y))"},
            "reference_fractions": {str(d + 1): str(REFERENCE_FRACTION[d]) for d in DIMENSIONS},
            "levels": rows,
            "rer_cross_check": {"receipt": "reverse-engineering-reality/" + RER_RECEIPT,
                                "receipt_sha256": RER_RECEIPT_SHA256,
                                "levels_compared": [n for n in levels if n in theirs and 3 in dimensions],
                                "all_agree": bool(checks) and all(checks)},
            "source_pins": pins}


def repin(path: Path = OUTPUT) -> bytes:
    receipt = json.loads(path.read_bytes())
    pins = {p: file_sha256(ROOT / p) for p in LOCAL_PINS}
    pins.update({"reverse-engineering-reality/" + p: file_sha256(RER_ROOT / p) for p in RER_PINS})
    receipt["source_pins"] = pins
    return canonical(receipt)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rebuild and write the receipt")
    parser.add_argument("--check", action="store_true", help="rebuild and compare with the receipt")
    parser.add_argument("--repin", action="store_true", help="refresh the file pins of the committed receipt")
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--levels", type=int, nargs="*", default=None)
    parser.add_argument("--exact-support-limit", type=int, default=EXACT_SUPPORT_LIMIT,
                        help="largest support (in sites) counted by exact all-pairs searches; larger supports are sampled")
    args = parser.parse_args(argv)
    if args.repin:
        data = repin()
        OUTPUT.write_bytes(data)
        print("SOURCE_NET_CAUSAL_LIMIT_REPINNED", len(data), hashlib.sha256(data).hexdigest())
        return
    data = canonical(build(levels=tuple(args.levels) if args.levels else LEVELS, processes=args.processes,
                           exact_support_limit=args.exact_support_limit))
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_bytes(data)
    if args.check and OUTPUT.read_bytes() != data:
        raise ValueError("source-net causal-limit receipt is stale")
    print("SOURCE_NET_CAUSAL_LIMIT_REPLAYED", len(data), hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
