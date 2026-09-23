"""The carrier stack realizes the source-net population (lane A-real).

One exact twelve-port carrier per site ``b`` of ``[0, q)^3`` holds the golden
conservative source record ``z(b)`` of the r2039 source-net construction as
its twelve integer port loads.  Each carrier reads its own position from its
loads through the exact rank-three response of the flagship theorem
``thm:rank-three``: ``x = 2 P_slow N`` with ``P_slow`` the projector onto the
``5 - sqrt5`` band of the seam Laplacian, so that ``x`` is the signed sum
``sum_k z_k v_{p_k}`` of the labeled generators ``v_p = 2 P_slow e_p`` whose
Gram matrix is ``G = 4 P_slow``.  Carriers find their metric neighbours from
those readbacks alone (``||x(b) - x(b')|| <= a_q``, ``a_q = L/sqrt q``,
decided exactly in ``Q(sqrt5)``), read each other for ``K_q = ceil(sqrt q)``
rounds under the RER trace law ``q_i(0) = i + 1``,
``q_i(j) = 1 + sum`` of the neighbours' previous values (the same-site read
included), and log every read as an authenticated read-after-write event.
The causal order is generated from that log by the read-after-write rule of
``oph_fpe/bulk/source_derived_causal_order.py`` (``generated_provenance_edges``:
an edge runs from the writer of a register version to every event that reads
it; one writer per version; no declared parents, no layer labels as input) and
its transitive closure; the interior-diamond statistics are recomputed on the
generated order.

What the carriers produce: positions, the neighbour graph, the reads, the
provenance order.  What is supplied: the load placement, the read law, the
round rule, one event per site and round, and the fixed population.  Not
claimed: selection of the population by native repair, a physical clock,
spacetime, the continuum limit.

Cross-checks: the readback metric equals ``L^2`` times the source metric of
``oph_exact/source_net.py`` exactly (``L^2 = 2 - (2/5) sqrt5``); the neighbour
digest equals the source-net digest at every level; the RER audit chains and
layer digests are reproduced at ``q = 5, 8, 13``; the provenance order equals
the layered order on the centre interval at every level; the ``+1``
intervention at the centre changes exactly the future cone.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from fractions import Fraction
from math import exp, isqrt, lgamma, log
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from oph_exact import carrier
from oph_exact import source_net

ROOT = Path(__file__).resolve().parents[1]
RER_ROOT = Path(os.environ.get("OPH_RER_ROOT", str(ROOT.parent / "reverse-engineering-reality")))
OUTPUT = ROOT / "data/exact/carrier_source_net_receipt.json"
LOG_DIR = ROOT / "data/exact/carrier_source_net_logs"
SCHEMA = "oph.exact.carrier-source-net.v1"
LOG_SCHEMA = "oph.exact.carrier-source-net-log.v1"
LEVELS = (5, 8, 13, 21, 34, 55, 89)
RER_LEVELS = (5, 8, 13)
STORED_LOG_LEVELS = (5, 8)
FIBONACCI_INDEX = {3: 4, 5: 5, 8: 6, 13: 7, 21: 8, 34: 9, 55: 10, 89: 11, 144: 12}
POSITIVE_PORTS = (0, 1, 4, 5, 8, 9)
PREFILTER_MARGIN = 1e-9
METRIC_IDENTITY_ALL_PAIRS_LIMIT = 10000
METRIC_IDENTITY_SAMPLE = 1_000_000
METRIC_IDENTITY_SEED = 20260909
EXACT_SAMPLE_COUNT = 8
BATCH_SIZE = 64
DIGEST_ROWS = 2000
CHUNK_ENTRIES = 20_000_000  # neighbour-list entries handled per chunk by the streaming passes
ID_BYTES = 4
VERSION_BYTES = 2
LOCAL_PINS = (
    "oph_exact/carrier_source_net.py",
    "oph_exact/verify_carrier_source_net_independent.py",
    "tests/test_exact_carrier_source_net.py",
    "oph_exact/carrier.py",
    "oph_exact/source_net.py",
    "data/exact/source_net_causal_limit_receipt.json",
)
RER_PINS = (
    "code/causal_refinement/source_net_causet.py",
    "code/causal_refinement/source_net_causet_receipt.json",
    "paper/tex_fragments/SOURCE_NET_CAUSAL_LIMIT.tex",
)
RER_RECEIPT = "code/causal_refinement/source_net_causet_receipt.json"
RER_RECEIPT_SHA256 = "c0f790ad383039186371b7a8020f190371fa533ad634010f01f9c271f1545681"
SOURCE_NET_RECEIPT = "data/exact/source_net_causal_limit_receipt.json"
# The declared frame of the bridge receipt (data/repair_closure/port_gram_completion_bridge_receipt.json,
# exact_signed_module_completion.raw_generator_coordinates_qsqrt5): raw generator of positive port p_k,
# in the basis {1, sqrt5}; its unit vector is the paper's source axis k.
GENERATOR_FRAME_QSQRT5 = (
    ((-1, 0), (Fraction(1, 2), Fraction(1, 2)), (0, 0)),
    ((1, 0), (Fraction(1, 2), Fraction(1, 2)), (0, 0)),
    ((0, 0), (-1, 0), (Fraction(1, 2), Fraction(1, 2))),
    ((0, 0), (1, 0), (Fraction(1, 2), Fraction(1, 2))),
    ((Fraction(1, 2), Fraction(1, 2)), (0, 0), (-1, 0)),
    ((Fraction(1, 2), Fraction(1, 2)), (0, 0), (1, 0)),
)
SCOPE = {
    "supplied": {
        "load_placement": True,
        "read_law": True,
        "round_rule": True,
        "one_event_per_site_and_round": True,
        "fixed_population": True,
        "source_records_z_of_b": True,
        "edge_radius_a_q": True,
    },
    "produced_by_carriers": {
        "positions": True,
        "neighbour_graph": True,
        "reads": True,
        "provenance_order": True,
    },
    "not_claimed": {
        "population_selected_by_native_repair": False,
        "physical_clock": False,
        "spacetime": False,
        "continuum_limit": False,
        "dynamic_population_under_repairs": False,
    },
    "exact_readback_metric_identity": True,
    "exact_neighbour_decisions_in_Qsqrt5": True,
    "read_write_traces_executed": True,
    "full_traces_stored": False,
    "order_generated_from_log_without_declared_parents": True,
}
CLAIM_BOUNDARY = (
    "Finite realization of the r2039 source-net population by exact twelve-port "
    "carriers at q = 5, 8, 13, 21, 34, 55, 89: each carrier holds the integer record z(b) as "
    "port loads, reads its position from those loads through the rank-three response "
    "x = 2 P_slow N, finds its metric neighbours from the readbacks by an exact "
    "Q(sqrt5) decision, reads its neighbours for K_q rounds, and the causal order is "
    "generated from the read-after-write provenance of those reads with no declared "
    "parent lists and no layer labels as input. The population, the load placement, "
    "the read law, the round rule, one event per site and round and the edge radius "
    "are supplied, as in the paper. The population is held fixed during the reads. "
    "Nothing here selects the population by native repair, identifies a physical "
    "clock or spacetime, or demonstrates the continuum limit."
)


# --------------------------------------------------------------------------
# Canonical JSON, digests, exact arithmetic helpers
# --------------------------------------------------------------------------


def canonical(x) -> bytes:
    return (json.dumps(x, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def digest(x) -> str:
    return hashlib.sha256(canonical(x)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rounded(x: float) -> float:
    """Twelve significant digits; derived statistics only (source_net convention)."""
    return float(f"{float(x):.12g}")


def q5s(x) -> list[str]:
    """Q(sqrt5) value (a, b) meaning a + b*sqrt5 as two rational strings."""
    return [str(Fraction(x[0])), str(Fraction(x[1]))]


def sign_qsqrt5_array(a, b) -> np.ndarray:
    """Vectorized exact sign of a + b*sqrt5 for int64 arrays (|a|, |b| below 2^31)."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    d = a * a - 5 * b * b
    mixed = np.sign(d) * np.where(a > 0, 1, -1)
    return np.where(b == 0, np.sign(a),
                    np.where((a >= 0) & (b > 0), 1,
                             np.where((a <= 0) & (b < 0), -1, mixed))).astype(np.int8)


def byte_length(v: int) -> int:
    """Minimal unsigned big-endian byte length of a positive integer value."""
    return max(1, (int(v).bit_length() + 7) // 8)


def myrheim_meyer_fraction(dimension: float) -> float:
    """Expected ordering fraction of a flat interval (the convention of oph_fpe.bulk.causet_likeness)."""
    return exp(lgamma(dimension + 1.0) + lgamma(dimension / 2.0) - log(2.0) - lgamma(3.0 * dimension / 2.0))


def invert_myrheim_meyer_fraction(fraction: float):
    """Bisection of the causet_likeness convention, reproduced so that the staged module is not imported."""
    if not 0.0 < fraction < 1.0:
        return None
    low, high = 1.01, 20.0
    if not (myrheim_meyer_fraction(high) <= fraction <= myrheim_meyer_fraction(low)):
        return None
    for _ in range(96):
        midpoint = (low + high) / 2.0
        if myrheim_meyer_fraction(midpoint) > fraction:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# The carrier side: records as loads, readback, generator frame
# --------------------------------------------------------------------------


def golden_floor(b: int) -> int:
    """floor(b*phi) exactly: (b + isqrt(5 b^2)) // 2."""
    return (b + isqrt(5 * b * b)) // 2


def source_records(q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sites of [0,q)^3 in product order, the records z(b) and the current sections.

    ``z(b) = (b2 - m1, b2 + m1, b3 - m2, b3 + m2, b1 - m3, b1 + m3)`` with
    ``m_i = -floor(b_i phi)`` (RER ``source_control``); the current section is
    RER ``source_currents``.  Every coordinate sum is ``2(b1 + b2 + b3)``.
    """
    m = np.array([-golden_floor(b) for b in range(q)], dtype=np.int64)
    sites = np.indices((q, q, q)).reshape(3, -1).T.astype(np.int64)
    a = m[sites]
    z = np.stack([sites[:, 1] - a[:, 0], sites[:, 1] + a[:, 0], sites[:, 2] - a[:, 1],
                  sites[:, 2] + a[:, 1], sites[:, 0] - a[:, 2], sites[:, 0] + a[:, 2]], axis=1)
    if np.any(z.sum(axis=1) % 2):
        raise ValueError("a source record left the even-sum sublattice")
    h = z.sum(axis=1) // 2
    currents = np.stack([-z[:, 5], h - z[:, 0], h - z[:, 2] - z[:, 3], h - z[:, 1] - z[:, 4] - z[:, 5],
                         h - z[:, 2] - z[:, 3] - z[:, 4], z[:, 3]], axis=1)
    return sites, z, currents


def place_loads(z: np.ndarray) -> np.ndarray:
    """Declared conservative split: ``N_{p_k} = max(z_k, 0)``, ``N_{-p_k} = max(-z_k, 0)``."""
    anti = carrier.antipode()
    loads = np.zeros((len(z), carrier.PORT_COUNT), dtype=np.int64)
    for k, p in enumerate(POSITIVE_PORTS):
        loads[:, p] = np.maximum(z[:, k], 0)
        loads[:, anti[p]] = np.maximum(-z[:, k], 0)
    return loads


def antipodal_odd_quotient(loads: np.ndarray) -> np.ndarray:
    """``Z^12 -> Z^12/{x_p = x_{-p}} = Z^6``: ``(N_{p_k} - N_{-p_k})_k``."""
    anti = carrier.antipode()
    return np.stack([loads[:, p] - loads[:, anti[p]] for p in POSITIVE_PORTS], axis=1)


def gram_sign_pattern() -> np.ndarray:
    """``sigma_kl = +1`` at port distance one, ``-1`` at distance two, ``0`` on the diagonal."""
    dist = carrier.graph_distance()
    sig = np.zeros((6, 6), dtype=np.int64)
    for k in range(6):
        for l in range(6):
            if k != l:
                d = int(dist[POSITIVE_PORTS[k], POSITIVE_PORTS[l]])
                if d not in (1, 2):
                    raise AssertionError("positive ports are not pairwise non-antipodal")
                sig[k, l] = 1 if d == 1 else -1
    return sig


def gram6_exact() -> list[list[carrier.QS5]]:
    g = carrier.intrinsic_gram_exact()
    return [[g[p][r] for r in POSITIVE_PORTS] for p in POSITIVE_PORTS]


def generator_frame_is_isometric() -> bool:
    """``a_k . a_l / (phi + 2) = G6_kl`` exactly for the declared raw generators."""
    g6 = gram6_exact()
    norm = carrier.q5(Fraction(5, 2), Fraction(1, 2))  # phi + 2
    inv = carrier.q5_inv(norm)
    for k in range(6):
        for l in range(6):
            dot = carrier.q5(0)
            for c in range(3):
                dot = carrier.q5_add(dot, carrier.q5_mul(carrier.q5(*GENERATOR_FRAME_QSQRT5[k][c]),
                                                          carrier.q5(*GENERATOR_FRAME_QSQRT5[l][c])))
            if carrier.q5_mul(dot, inv) != g6[k][l]:
                return False
    return True


def readback_generators() -> np.ndarray:
    """``v_p = 2 P_slow e_p`` for the six positive ports, rows of a (6, 12) float array."""
    p2 = 2.0 * carrier.slow_band_projector()
    return np.stack([p2[:, p] for p in POSITIVE_PORTS], axis=0)


def readback(loads: np.ndarray) -> np.ndarray:
    """``x = 2 P_slow N`` in the twelve-port space (float)."""
    return loads.astype(float) @ (2.0 * carrier.slow_band_projector()).T


def chart_basis() -> np.ndarray:
    """An orthonormal basis of ``range(P_slow)``: isometric chart for the prefilter only."""
    w, v = np.linalg.eigh(carrier.slow_band_projector())
    basis = v[:, w > 0.5]
    if basis.shape[1] != 3:
        raise AssertionError("slow band is not three-dimensional")
    return basis


def exact_readback(load_row) -> list[carrier.QS5]:
    """``x = 2 P_slow N`` exactly in Q(sqrt5) for one carrier."""
    p = carrier.slow_band_projector_exact()
    out = []
    for r in range(carrier.PORT_COUNT):
        acc = carrier.q5(0)
        for c in range(carrier.PORT_COUNT):
            if load_row[c]:
                acc = carrier.q5_add(acc, carrier.q5_mul(p[r][c], carrier.q5(2 * int(load_row[c]))))
        out.append(acc)
    return out


def exact_norm_squared(x: list[carrier.QS5]) -> carrier.QS5:
    acc = carrier.q5(0)
    for v in x:
        acc = carrier.q5_add(acc, carrier.q5_mul(v, v))
    return acc


def source_squared_distance_qsqrt5(sites: np.ndarray, i: int, j: int) -> carrier.QS5:
    """Source-net squared distance in units of L^2, ``A + B phi = (A + B/2) + (B/2) sqrt5``."""
    A = 0
    B = 0
    for axis in range(3):
        bi, bj = int(sites[i, axis]), int(sites[j, axis])
        dm = -golden_floor(bi) + golden_floor(bj)
        db = bi - bj
        A += dm * dm + db * db
        B += 2 * dm * db + db * db
    return carrier.q5(Fraction(A) + Fraction(B, 2), Fraction(B, 2))


L_SQUARED_QSQRT5: carrier.QS5 = carrier.q5(2, Fraction(-2, 5))  # 12/5 - (4/5) phi


def exact_sample_checks(sites: np.ndarray, z: np.ndarray, loads: np.ndarray, sample: list[int]) -> dict:
    """Exact Q(sqrt5) checks on a sample of carriers: Gram identity, paper contraction, metric identity."""
    g6 = gram6_exact()
    xs = {i: exact_readback(loads[i]) for i in sample}
    gram_identity = True
    contraction = True
    for i in sample:
        zi = [int(v) for v in z[i]]
        quad = carrier.q5(0)
        for k in range(6):
            for l in range(6):
                quad = carrier.q5_add(quad, carrier.q5_mul(carrier.q5(zi[k] * zi[l]), g6[k][l]))
        if exact_norm_squared(xs[i]) != quad:
            gram_identity = False
        # Position in the declared frame: sum_k z_k a_k / 2 = (xi_1, xi_2, xi_3) as m + b phi.
        for c in range(3):
            acc = carrier.q5(0)
            for k in range(6):
                acc = carrier.q5_add(acc, carrier.q5_mul(carrier.q5(zi[k]), carrier.q5(*GENERATOR_FRAME_QSQRT5[k][c])))
            acc = carrier.q5_mul(acc, carrier.q5(Fraction(1, 2)))
            b = int(sites[i, c])
            xi = carrier.q5(Fraction(-golden_floor(b)) + Fraction(b, 2), Fraction(b, 2))  # m + b phi
            if acc != xi:
                contraction = False
    metric = True
    pairs = 0
    for a_ in range(len(sample)):
        for b_ in range(a_ + 1, len(sample)):
            i, j = sample[a_], sample[b_]
            diff = [carrier.q5_sub(xs[i][r], xs[j][r]) for r in range(carrier.PORT_COUNT)]
            lhs = exact_norm_squared(diff)
            rhs = carrier.q5_mul(L_SQUARED_QSQRT5, source_squared_distance_qsqrt5(sites, i, j))
            pairs += 1
            if lhs != rhs:
                metric = False
    return {"sample_sites": [int(i) for i in sample], "exact_gram_identity": gram_identity,
            "exact_position_equals_paper_contraction": contraction,
            "exact_metric_identity_pairs": pairs, "exact_metric_identity": metric}


def metric_identity_integer_form(dz: np.ndarray, sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``10 ||dx||^2 = 10 S2 + 2 X sqrt5`` with ``S2 = sum dz_k^2``, ``X = sum_{k != l} sigma_kl dz_k dz_l``."""
    s2 = (dz * dz).sum(axis=1)
    x = np.einsum("mk,kl,ml->m", dz, sig, dz)
    return 10 * s2, 2 * x


def source_metric_integer_form(sites: np.ndarray, i: np.ndarray, j: np.ndarray, floors: np.ndarray):
    """``10 L^2 |xi_i - xi_j|^2 = 20 A + (8 B - 4 A) sqrt5`` from the per-axis tables."""
    A = np.zeros(len(i), dtype=np.int64)
    B = np.zeros(len(i), dtype=np.int64)
    for axis in range(3):
        bi, bj = sites[i, axis], sites[j, axis]
        dm = -floors[bi] + floors[bj]
        db = bi - bj
        A += dm * dm + db * db
        B += 2 * dm * db + db * db
    return 20 * A, 8 * B - 4 * A


def metric_identity_all_or_sampled(sites: np.ndarray, z: np.ndarray, sig: np.ndarray, q: int) -> dict:
    n = len(z)
    floors = np.array([golden_floor(b) for b in range(q)], dtype=np.int64)

    def blocks():
        if n <= METRIC_IDENTITY_ALL_PAIRS_LIMIT:
            rows = max(1, 2_000_000 // n)
            for lo in range(0, n, rows):
                i = np.repeat(np.arange(lo, min(lo + rows, n), dtype=np.int64), n)
                j = np.tile(np.arange(n, dtype=np.int64), min(rows, n - lo))
                keep = j > i
                yield i[keep], j[keep]
        else:
            rng = np.random.default_rng(METRIC_IDENTITY_SEED + q)
            for lo in range(0, METRIC_IDENTITY_SAMPLE, 1_000_000):
                size = min(1_000_000, METRIC_IDENTITY_SAMPLE - lo)
                yield (rng.integers(0, n, size, dtype=np.int64), rng.integers(0, n, size, dtype=np.int64))

    mode = "all_pairs" if n <= METRIC_IDENTITY_ALL_PAIRS_LIMIT else "sampled_pairs"
    ok = True
    checked = 0
    for i, j in blocks():
        la, lb = metric_identity_integer_form(z[i] - z[j], sig)
        ra, rb = source_metric_integer_form(sites, i, j, floors)
        ok = ok and bool(np.array_equal(la, ra) and np.array_equal(lb, rb))
        checked += len(i)
    return {"mode": mode, "pairs_checked": int(checked), "exact": ok,
            "seed": None if mode == "all_pairs" else METRIC_IDENTITY_SEED + q}


# --------------------------------------------------------------------------
# Neighbours from readbacks
# --------------------------------------------------------------------------


def neighbour_decision(dz: np.ndarray, q: int, sig: np.ndarray) -> np.ndarray:
    """Exact ``||dx||^2 <= L^2/q``: sign((5 q S2 - 10) + (q X + 2) sqrt5) <= 0."""
    s2 = (dz * dz).sum(axis=1)
    x = np.einsum("mk,kl,ml->m", dz, sig, dz)
    return sign_qsqrt5_array(5 * q * s2 - 10, q * x + 2) <= 0


def find_neighbours(x3: np.ndarray, z: np.ndarray, q: int, sig: np.ndarray) -> dict:
    """Candidate pairs from the readback chart inside ``a_q (1 + margin)``, each decided exactly.

    The candidates are proposed per chunk of carriers by ball queries on the
    float chart (every candidate pair is met twice, once from each endpoint);
    every candidate is decided exactly in ``Q(sqrt5)`` from the integer records.
    The counts are over unordered candidate pairs.  The CSR rows are ascending
    and include the same-site read.
    """
    n = len(z)
    l2 = 2.0 - 2.0 / 5.0 ** 0.5
    a_q = (l2 / q) ** 0.5
    radius = a_q * (1.0 + PREFILTER_MARGIN)
    tree = cKDTree(x3)
    expected_degree = max(1, int(4.19 * q ** 1.5))
    chunk = max(64, min(n, CHUNK_ENTRIES // expected_degree))
    indptr = np.zeros(n + 1, dtype=np.int64)
    pieces = []
    candidates = accepted_pairs = borderline = 0
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        lists = tree.query_ball_point(x3[lo:hi], radius)
        lengths = np.fromiter((len(l) for l in lists), dtype=np.int64, count=hi - lo)
        j = np.concatenate([np.asarray(l, dtype=np.int64) for l in lists]) if lengths.sum() else np.zeros(0, dtype=np.int64)
        i = np.repeat(np.arange(lo, hi, dtype=np.int64), lengths)
        other = i != j
        io, jo = i[other], j[other]
        upper = jo > io
        candidates += int(np.count_nonzero(upper))
        ok = neighbour_decision(z[io] - z[jo], q, sig)
        accepted_pairs += int(np.count_nonzero(ok & upper))
        d = np.linalg.norm(x3[io[upper]] - x3[jo[upper]], axis=1)
        borderline += int(np.count_nonzero(np.abs(d - a_q) <= a_q * PREFILTER_MARGIN))
        keep_i = np.concatenate([io[ok], np.arange(lo, hi, dtype=np.int64)])
        keep_j = np.concatenate([jo[ok], np.arange(lo, hi, dtype=np.int64)])
        order = np.lexsort((keep_j, keep_i))
        keep_i, keep_j = keep_i[order], keep_j[order]
        counts = np.bincount(keep_i - lo, minlength=hi - lo)
        indptr[lo + 1:hi + 1] = indptr[lo] + np.cumsum(counts)
        pieces.append(keep_j.astype(np.int32))
    indices = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int32)
    return {"indptr": indptr, "indices": indices, "candidate_pairs": int(candidates),
            "accepted_pairs": int(accepted_pairs), "rejected_candidates": int(candidates - accepted_pairs),
            "borderline_candidates": borderline}


def neighbour_digest(indptr: np.ndarray, indices: np.ndarray) -> str:
    """sha256 of the canonical JSON of the neighbour lists (source_net / RER format), streamed."""
    n = len(indptr) - 1
    h = hashlib.sha256()
    h.update(b"[")
    for lo in range(0, n, DIGEST_ROWS):
        hi = min(lo + DIGEST_ROWS, n)
        parts = ["[" + ",".join(map(str, indices[indptr[i]:indptr[i + 1]].tolist())) + "]" for i in range(lo, hi)]
        if lo:
            h.update(b",")
        h.update(",".join(parts).encode("ascii"))
    h.update(b"]\n")
    return h.hexdigest()


# --------------------------------------------------------------------------
# Layered reads with an authenticated read-after-write log
# --------------------------------------------------------------------------


def run_reads(indptr: np.ndarray, indices: np.ndarray, rounds: int, intervention: int | None = None,
              chain: bool = True) -> dict:
    """Execute ``K`` rounds of complete-neighbour reads under the RER trace law.

    Round 0 writes the seed ``q_i(0) = i + 1`` (``+1`` at the intervention site).
    In round ``j >= 1`` every carrier reads version ``j`` of every neighbour
    register (written in round ``j - 1``, the same-site read included) and
    writes version ``j + 1`` of its own register with value ``1 + sum``.  The
    audit chain is RER's: ``chain = sha256(chain || canonical(material))`` with
    ``material = [[j, i], reads, [i, j + 1, [j, i], value]]`` and
    ``reads = [[r, j, [j - 1, r], value_read], ...]`` in neighbour order.

    The neighbour rows are streamed from the CSR arrays and every read record
    ``[r, j, [j - 1, r], value_read]`` is formatted once per round, so the
    memory stays linear in the number of carriers rather than in the number
    of reads.
    """
    n = len(indptr) - 1
    ch = bytes(32)
    layer_hashes, layer_sums, values = [], [], []
    prev = None
    sha = hashlib.sha256
    starts = indptr[:-1].tolist()
    ends = indptr[1:].tolist()
    for j in range(rounds + 1):
        if j == 0:
            cur = [1 + i + (1 if i == intervention else 0) for i in range(n)]
            if chain:
                for i in range(n):
                    ch = sha(ch + b"[[0,%d],[],[%d,1,[0,%d],%d]]\n" % (i, i, i, cur[i])).digest()
        else:
            cur = [0] * n
            piece = None
            if chain:
                piece = ["[%d,%d,[%d,%d],%d]" % (r, j, j - 1, r, v) for r, v in enumerate(prev)]
            get_prev = prev.__getitem__
            for i in range(n):
                row = indices[starts[i]:ends[i]].tolist()
                value = 1 + sum(map(get_prev, row))
                cur[i] = value
                if chain:
                    material = "[[%d,%d],[%s],[%d,%d,[%d,%d],%d]]\n" % (
                        j, i, ",".join(map(piece.__getitem__, row)), i, j + 1, j, i, value)
                    ch = sha(ch + material.encode("ascii")).digest()
        layer_hashes.append(digest(cur))
        layer_sums.append(sum(cur))
        values.append(cur)
        prev = cur
    return {"audit_trace_sha256": ch.hex(), "layer_value_sha256": layer_hashes, "layer_value_sums": layer_sums,
            "event_count": (rounds + 1) * n, "authenticated_read_count": rounds * int(indptr[-1]),
            "values": values}


class EventLog:
    """The read records of the log, one chunk per round ``j >= 1``.

    Each read is ``(reader site, register, version)``; the reader event is the
    round-``j`` event of the reader site and the version read is ``j``.  The
    records are listed in execution order (readers ascending, registers in
    neighbour order), which is the order the audit chain digests them in.
    ``forward``/``backward`` yield whole rounds; ``chunks`` yields the same
    records of one round in consecutive reader ranges, for the streaming passes.
    """

    def __init__(self, indptr: np.ndarray, indices: np.ndarray, rounds: int) -> None:
        n = len(indptr) - 1
        self.n = n
        self.rounds = rounds
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int32)
        degree = max(1, int(self.indptr[-1]) // max(1, n))
        self.chunk_rows = max(1, min(n, CHUNK_ENTRIES // degree))

    def round(self, j: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        readers = np.repeat(np.arange(self.n, dtype=np.int32), np.diff(self.indptr))
        return readers, self.indices, np.full(len(readers), j, dtype=np.int16)

    def chunks(self):
        """``(lo, hi, readers, registers)`` over consecutive reader ranges, in execution order."""
        for lo in range(0, self.n, self.chunk_rows):
            hi = min(lo + self.chunk_rows, self.n)
            st, en = int(self.indptr[lo]), int(self.indptr[hi])
            readers = np.repeat(np.arange(lo, hi, dtype=np.int32), np.diff(self.indptr[lo:hi + 1]))
            yield lo, hi, readers, self.indices[st:en]

    def forward(self):
        for j in range(1, self.rounds + 1):
            yield (j, *self.round(j))

    def backward(self):
        for j in range(self.rounds, 0, -1):
            yield (j, *self.round(j))


def write_records(values: list[list[int]]) -> list[tuple[int, int, int]]:
    """``(register, version, value)`` for every event, in execution order."""
    out = []
    for j, cur in enumerate(values):
        out.extend((i, j + 1, v) for i, v in enumerate(cur))
    return out


# --------------------------------------------------------------------------
# Provenance-derived order
# --------------------------------------------------------------------------


class ProvenanceError(ValueError):
    pass


def provenance_edges(n: int, rounds: int, values: list[list[int]], log: EventLog) -> dict:
    """Generate the causal edges from the log by the read-after-write rule.

    Rule (``oph_fpe/bulk/source_derived_causal_order.py::generated_provenance_edges``,
    reimplemented): every register version has exactly one writer; an edge
    runs from the writer event of a version to every event that reads it; a
    read of an unwritten version is an error (the seeds are the only events
    without parents); an event never reads its own committed version.  Event
    ids are the ordinal positions in the log.  Layer labels are outputs: the
    derived longest-path rank of every event is reported and compared with
    the round it was executed in.  The value witness of the paper's rule
    (the value read equals the value the writer committed) holds by
    construction here: versions are immutable and single-copy, and every
    value read is committed by the audit chain.

    The records of each round are consumed in consecutive reader ranges, so
    the working memory is linear in the chunk size, not in the read count.
    """
    writer_of = np.full((n, rounds + 2), -1, dtype=np.int64)
    for event_id, (register, version, _value) in enumerate(write_records(values)):
        if writer_of[register, version] != -1:
            raise ProvenanceError("register version has two writers")
        writer_of[register, version] = event_id
    rank = np.zeros((rounds + 1) * n, dtype=np.int64)
    edge_count = 0
    chronological = True
    rank_matches_round = True
    relation_identical = True
    first_digests = None
    round_relations = []
    for j in range(1, rounds + 1):
        digests = []
        for lo, hi, readers, registers in log.chunks():
            parents = writer_of[registers, j]
            if np.any(parents < 0):
                raise ProvenanceError("read of an unwritten register version")
            children = j * n + readers.astype(np.int64)
            if np.any(parents == children):
                raise ProvenanceError("event reads its own committed version")
            chronological = chronological and bool(np.all(parents < children))
            edge_count += int(len(parents))
            # Longest-path rank: 1 + max over parents, grouped by child (records are grouped by reader).
            starts = np.flatnonzero(np.r_[True, readers[1:] != readers[:-1]])
            child_ids = children[starts]
            rank[child_ids] = np.maximum.reduceat(rank[parents], starts) + 1
            rank_matches_round = rank_matches_round and bool(np.all(rank[child_ids] == j))
            h = hashlib.sha256()
            h.update(np.ascontiguousarray(readers, dtype=np.int32).tobytes())
            h.update(np.ascontiguousarray(registers, dtype=np.int32).tobytes())
            digests.append(h.hexdigest())
        if first_digests is None:
            first_digests = digests
        else:
            relation_identical = relation_identical and digests == first_digests
        round_relations.append(j)
    return {"edge_count": edge_count, "single_writer": True, "all_reads_resolved": True,
            "parents_precede_children": chronological, "derived_rank_equals_round": rank_matches_round,
            "read_relation_identical_across_rounds": relation_identical,
            "rounds_seen": round_relations, "rank": rank,
            "relation": (log.indptr, log.indices)}


def _ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    cl = np.cumsum(lengths)
    total = int(cl[-1])
    idx = np.ones(total, dtype=np.int64)
    idx[0] = starts[0]
    if len(starts) > 1:
        idx[cl[:-1]] = starts[1:] - (starts[:-1] + lengths[:-1]) + 1
    np.cumsum(idx, out=idx)
    return idx


def bfs(indptr: np.ndarray, indices: np.ndarray, start: int, n: int, cap: int | None = None,
        alpha: np.ndarray | None = None, budget: int | None = None) -> np.ndarray:
    """Distances in the read relation from ``start``; ``-1`` beyond ``cap`` layers.

    With ``alpha`` (distances from the centre) and ``budget``, the frontier at
    layer ``m`` is restricted to ``alpha <= budget - m``.  Every site ``u`` on
    a shortest path from ``start`` to a target ``t`` with
    ``alpha_t <= budget - d(start, t)`` satisfies
    ``alpha_u <= alpha_t + d(u, t) <= budget - d(start, u)``, so the returned
    distances are exact for those targets (the pruning of
    ``oph_exact/source_net.py::bfs`` with graph-distance balls).
    """
    dist = np.full(n, -1, dtype=np.int16)
    dist[start] = 0
    frontier = np.array([start], dtype=np.int64)
    layers = n if cap is None else cap
    for m in range(layers):
        if alpha is not None:
            frontier = frontier[alpha[frontier] <= budget - m]
        if frontier.size == 0:
            break
        st = indptr[frontier]
        ln = indptr[frontier + 1] - st
        idx = _ranges(st.astype(np.int64), ln.astype(np.int64))
        mask = np.zeros(n, dtype=bool)
        mask[indices[idx]] = True
        mask &= dist < 0
        frontier = np.flatnonzero(mask)
        if frontier.size == 0:
            break
        dist[frontier] = m + 1
    return dist


def cones_from_log(n: int, rounds: int, centre: int, log: EventLog) -> dict:
    """Future cone of the centre's seed event and past cone of its last event, from the log records.

    A round-``j`` event is in the future cone iff one of its reads names a
    register whose version-``j`` writer is in the cone; it is in the past cone
    iff a cone event of round ``j + 1`` reads the version it wrote.
    """
    future = [np.array([centre], dtype=np.int64)]
    current = np.zeros(n, dtype=bool)
    current[centre] = True
    for _j in range(1, rounds + 1):
        nxt = np.zeros(n, dtype=bool)
        for _lo, _hi, readers, registers in log.chunks():
            hit = current[registers]
            nxt[readers[hit]] = True
        current = nxt
        future.append(np.flatnonzero(current).astype(np.int64))
    past = [None] * (rounds + 1)
    past[rounds] = np.array([centre], dtype=np.int64)
    current = np.zeros(n, dtype=bool)
    current[centre] = True
    for j in range(rounds, 0, -1):
        prv = np.zeros(n, dtype=bool)
        for _lo, _hi, readers, registers in log.chunks():
            hit = current[readers]
            prv[registers[hit]] = True
        current = prv
        past[j - 1] = np.flatnonzero(current).astype(np.int64)
    return {"future": future, "past": past}


_G: dict = {}


def _worker_init(payload: dict) -> None:
    _G.clear()
    _G.update(payload)
    if "indptr_path" in payload:
        _G["indptr"] = np.load(payload["indptr_path"], mmap_mode="r")
        _G["indices"] = np.load(payload["indices_path"], mmap_mode="r")
        _G["alpha"] = np.load(payload["alpha_path"], mmap_mode="r")


def _histograms(starts) -> np.ndarray:
    """Per-start histograms ``h[alpha_s][alpha_t][d(s, t)]`` over support sites ``t``."""
    K = int(_G["K"])
    n = int(_G["n"])
    centre = int(_G["centre"])
    indptr, indices, alpha = _G["indptr"], _G["indices"], _G["alpha"]
    support = _G["support"]
    a_sup = _G["support_alpha"]
    bins = K + 2
    H = np.zeros((K + 1, K + 1, bins), dtype=np.int64)
    for s in starts:
        s = int(s)
        a_s = int(alpha[s])
        cap = K - a_s - 1
        if cap > 0:
            dist = bfs(indptr, indices, s, n, cap=cap, alpha=alpha, budget=K - a_s)
        else:
            dist = np.full(n, -1, dtype=np.int16)
            dist[s] = 0
        dist[centre] = a_s
        d = dist[support].astype(np.int64)
        d[d < 0] = K + 1
        H[a_s] += np.bincount(a_sup * bins + d, minlength=(K + 1) * bins).reshape(K + 1, bins)
    return H


def strict_pair_counts(H: np.ndarray, K: int) -> dict[int, int]:
    """``C_k = sum_s sum_{j, j'} sum_{alpha_t <= min(j', k - j')} sum_{d <= j' - j} h_s``."""
    counts = {}
    for k in range(1, K + 1):
        C = 0
        for a_s in range(k // 2 + 1):
            for j in range(a_s, k - a_s + 1):
                for jp in range(j + 1, k + 1):
                    amax = min(jp, k - jp)
                    C += int(H[a_s, :amax + 1, :jp - j + 1].sum())
        counts[k] = C
    return counts


# --------------------------------------------------------------------------
# One level
# --------------------------------------------------------------------------


def centre_site(q: int) -> tuple[int, int]:
    """RER's centre: the label whose orbit value is nearest to 1/2 (smallest label on ties)."""
    values = source_net.orbit(q)
    centre_axis = 0
    half = (Fraction(1, 2), 0)
    for b in range(1, q):
        if source_net.phi_sign(source_net.phi_sub(source_net.phi_square(source_net.phi_sub(values[b], half)),
                                                  source_net.phi_square(source_net.phi_sub(values[centre_axis], half)))) < 0:
            centre_axis = b
    return centre_axis, (centre_axis * q + centre_axis) * q + centre_axis


def layered_interval_digest(alpha: np.ndarray, k: int) -> tuple[list[int], str]:
    events = []
    counts = []
    for j in range(k + 1):
        sites = np.flatnonzero(alpha <= min(j, k - j)).tolist()
        counts.append(len(sites))
        events.extend([j, s] for s in sites)
    return counts, digest(events)


def sample_sites(n: int, centre: int) -> list[int]:
    picks = {0, centre, n - 1}
    step = max(1, n // EXACT_SAMPLE_COUNT)
    for i in range(0, n, step):
        picks.add(i)
    return sorted(picks)[:EXACT_SAMPLE_COUNT + 3]


def build_level(q: int, frozen: dict, rer_level: dict | None, processes: int = 1,
                workdir: Path | None = None, keep_log: bool = False) -> dict:
    t0 = time.time()
    n = q ** 3
    K = source_net.ceil_sqrt(q)
    centre_axis, centre = centre_site(q)
    sig = gram_sign_pattern()

    # 1. Population: records as loads; round trip through the antipodal-odd quotient.
    sites, z, currents = source_records(q)
    loads = place_loads(z)
    round_trip = bool(np.array_equal(antipodal_odd_quotient(loads), z))
    records = [[b, zz, cc] for b, zz, cc in zip(sites.tolist(), z.tolist(), currents.tolist())]
    records_sha = digest(records)
    lengths = np.abs(currents).sum(axis=1)
    frozen_records_sha = frozen["source_records_sha256"]

    # 2. Readback by the carrier; metric identity.
    x12 = readback(loads)
    gens = readback_generators()
    signed_sum = z.astype(float) @ gens
    readback_equals_signed_sum = bool(np.max(np.abs(x12 - signed_sum)) < 1e-9)
    x3 = x12 @ chart_basis()
    exact_checks = exact_sample_checks(sites, z, loads, sample_sites(n, centre))
    identity = metric_identity_all_or_sampled(sites, z, sig, q)
    t_readback = time.time() - t0

    # 3. Neighbours from readbacks.
    t1 = time.time()
    nb = find_neighbours(x3, z, q, sig)
    indptr, indices = nb["indptr"], nb["indices"]
    degrees = np.diff(indptr)
    nb_sha = neighbour_digest(indptr, indices)
    t_neighbours = time.time() - t1

    # 4. Layered reads: forward run and the centre +1 intervention, both hash-chained.
    t2 = time.time()
    forward = run_reads(indptr, indices, K)
    intervention = run_reads(indptr, indices, K, intervention=centre)
    t_reads = time.time() - t2

    # 5. Provenance order from the log records.
    t3 = time.time()
    log = EventLog(indptr, indices, K)
    prov = provenance_edges(n, K, forward["values"], log)
    rel_indptr, rel_indices = prov["relation"]
    relation_sha = neighbour_digest(rel_indptr, rel_indices)
    cones = cones_from_log(n, K, centre, log)
    interval_events = []
    interval_counts = []
    for j in range(K + 1):
        both = np.intersect1d(cones["future"][j], cones["past"][j]).tolist()
        interval_counts.append(len(both))
        interval_events.extend([j, s] for s in both)
    interval_sha = digest(interval_events)
    alpha = bfs(rel_indptr, rel_indices, centre, n).astype(np.int64)
    if int(alpha.min()) < 0:
        raise ValueError("read relation is disconnected")
    layered_counts, layered_sha = layered_interval_digest(alpha, K)
    # The layered order of source_net.py on its own site graph (the comparison target).
    sn_indptr, sn_indices, *_ = source_net.build_site_graph(q, 3)
    sn_alpha = source_net.bfs(sn_indptr, sn_indices, centre, n).astype(np.int64)
    sn_counts, sn_sha = layered_interval_digest(sn_alpha, K)
    future_counts = [int(len(f)) for f in cones["future"]]
    future_sha = [digest([int(v) for v in f]) for f in cones["future"]]
    t_prov = time.time() - t3

    # 6. Strict pairs on the provenance order (per-start pruned searches on the common relation).
    t4 = time.time()
    support = np.flatnonzero(alpha <= K // 2)
    payload = {"K": K, "n": n, "centre": centre, "support": support, "support_alpha": alpha[support]}
    pool = None
    if processes > 1 and len(support) >= 512:
        np.save(workdir / f"indptr_{q}.npy", rel_indptr)
        np.save(workdir / f"indices_{q}.npy", rel_indices)
        np.save(workdir / f"alpha_{q}.npy", alpha)
        payload.update({"indptr_path": str(workdir / f"indptr_{q}.npy"),
                        "indices_path": str(workdir / f"indices_{q}.npy"),
                        "alpha_path": str(workdir / f"alpha_{q}.npy")})
        pool = mp.get_context("spawn").Pool(processes, initializer=_worker_init, initargs=(payload,))
    else:
        payload.update({"indptr": rel_indptr, "indices": rel_indices, "alpha": alpha})
    _worker_init(payload)
    H = np.zeros((K + 1, K + 1, K + 2), dtype=np.int64)
    try:
        tasks = [support[i:i + BATCH_SIZE] for i in range(0, len(support), BATCH_SIZE)]
        outputs = pool.imap_unordered(_histograms, tasks) if pool is not None else map(_histograms, tasks)
        for h in outputs:
            H += h
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    pairs = strict_pair_counts(H, K)
    t_pairs = time.time() - t4

    # 7. Intervention support per round against the future cone, RER and source_net.
    support_rows = []
    frozen_probes = {(p["start"], p["layers"]): p for p in frozen["reachability_probes"]}
    rer_response = {r["layer"]: r for r in rer_level["intervention_response"]} if rer_level else None
    for j in range(K + 1):
        changed = np.flatnonzero(np.array([a != b for a, b in zip(intervention["values"][j], forward["values"][j])]))
        deltas = [b - a for a, b in zip(forward["values"][j], intervention["values"][j])]
        row = {"round": j, "support_count": int(len(changed)), "support_ids_sha256": digest(changed.tolist()),
               "positive_integer_delta_sum": int(sum(deltas)), "positive_integer_delta_maximum": int(max(deltas)),
               "all_deltas_nonnegative": bool(min(deltas) >= 0),
               "equals_future_cone": bool(len(changed) == len(cones["future"][j]) and np.array_equal(changed, cones["future"][j]))}
        if j >= 1:
            probe = frozen_probes.get((centre, j))
            row["equals_source_net_reachable_ids"] = (None if probe is None
                                                     else bool(probe["reachable_ids_sha256"] == row["support_ids_sha256"]))
        if rer_response is not None:
            rr = rer_response[j]
            row["equals_rer_support"] = bool(rr["support_ids_sha256"] == row["support_ids_sha256"]
                                             and rr["support_count"] == row["support_count"]
                                             and rr["positive_integer_delta_sum"] == row["positive_integer_delta_sum"]
                                             and rr["positive_integer_delta_maximum"] == row["positive_integer_delta_maximum"])
        support_rows.append(row)

    # 8. Manifold readouts on the provenance order.
    frozen_rows = {r["layers"]: r for r in frozen["vertical_intervals"]}
    intervals = []
    for k in range(1, K + 1):
        counts, _sha = layered_interval_digest(alpha, k)
        N = sum(counts)
        C = pairs[k]
        fr = frozen_rows[k]
        row = {"layers": k, "inclusive_event_count": N, "counts_by_layer": counts, "strict_pair_count": C}
        if N >= 2:
            f = Fraction(2 * C, N * (N - 1))
            dim_est = invert_myrheim_meyer_fraction(float(f))
            row.update({"ordering_fraction": str(f), "ordering_fraction_float": rounded(float(f)),
                        "distance_to_one_tenth": rounded(float(f - Fraction(1, 10))),
                        "myrheim_meyer_dimension": None if dim_est is None else rounded(dim_est)})
        else:
            row.update({"ordering_fraction": None, "ordering_fraction_float": None,
                        "distance_to_one_tenth": None, "myrheim_meyer_dimension": None})
        row["equals_source_net"] = bool(
            fr["inclusive_event_count"] == N and fr["counts_by_layer"] == counts
            and fr.get("pair_counting") == "exact_all_pairs" and fr["strict_pair_count"] == C
            and fr["ordering_fraction"] == row["ordering_fraction"]
            and fr["myrheim_meyer_dimension"] == row["myrheim_meyer_dimension"])
        intervals.append(row)
    Kp = K // 2
    NI, NJ = intervals[K - 1]["inclusive_event_count"], intervals[Kp - 1]["inclusive_event_count"]
    clock = {"reference_layers": Kp, "interval_layers": K, "interval_count": NI, "reference_count": NJ,
             "count_clock": rounded((NI / NJ) ** 0.25), "model_time_ratio": rounded(K / Kp),
             "relative_deviation": rounded((NI / NJ) ** 0.25 / (K / Kp) - 1.0)}
    fc = frozen["count_clock"]
    clock["equals_source_net"] = bool(all(fc[key] == clock[key] for key in
                                          ("reference_layers", "interval_layers", "interval_count", "reference_count",
                                           "count_clock", "model_time_ratio", "relative_deviation")))

    # 9. Operation costs under the declared byte model.
    deg64 = degrees.astype(np.int64)
    cost_rows = []
    total_reads = total_writes = total_read_bytes = total_write_bytes = 0
    for j in range(K + 1):
        cur = forward["values"][j]
        write_bytes = sum(ID_BYTES + VERSION_BYTES + byte_length(v) for v in cur)
        if j == 0:
            reads = 0
            read_bytes = 0
        else:
            prev_len = np.array([byte_length(v) for v in forward["values"][j - 1]], dtype=np.int64)
            reads = int(deg64.sum())
            read_bytes = int((deg64 * (2 * ID_BYTES + VERSION_BYTES + prev_len)).sum())
        cost_rows.append({"round": j, "reads": reads, "writes": n, "read_bytes": read_bytes,
                          "write_bytes": int(write_bytes), "total_bytes": read_bytes + int(write_bytes),
                          "maximum_value_bytes": max(byte_length(v) for v in cur)})
        total_reads += reads
        total_writes += n
        total_read_bytes += read_bytes
        total_write_bytes += int(write_bytes)

    # RER cross-check of the trace digests.
    rer_block = None
    if rer_level is not None:
        fwd = rer_level["forward_execution"]
        itv = rer_level["center_plus_one_intervention"]
        rer_block = {
            "fibonacci_index": rer_level["fibonacci_index"],
            "forward_audit_trace_equal": bool(fwd["audit_trace_sha256"] == forward["audit_trace_sha256"]),
            "forward_layer_values_equal": bool(fwd["layer_value_sha256"] == forward["layer_value_sha256"]),
            "forward_layer_sums_equal": bool(fwd["layer_value_sums"] == forward["layer_value_sums"]),
            "forward_counts_equal": bool(fwd["event_count"] == forward["event_count"]
                                         and fwd["authenticated_read_count"] == forward["authenticated_read_count"]),
            "intervention_audit_trace_equal": bool(itv["audit_trace_sha256"] == intervention["audit_trace_sha256"]),
            "intervention_layer_values_equal": bool(itv["layer_value_sha256"] == intervention["layer_value_sha256"]),
            "intervention_layer_sums_equal": bool(itv["layer_value_sums"] == intervention["layer_value_sums"]),
            "neighbour_digest_equal": bool(rer_level["neighbors_including_wait_sha256"] == nb_sha),
            "source_records_equal": bool(rer_level["source_records_sha256"] == records_sha),
            "intervention_source_equal": bool(rer_level["intervention_source_id"] == centre),
            "response_supports_equal": bool(all(r["equals_rer_support"] for r in support_rows)),
            "rer_forward_audit_trace_sha256": fwd["audit_trace_sha256"],
            "rer_intervention_audit_trace_sha256": itv["audit_trace_sha256"],
        }
        rer_block["all_agree"] = bool(all(v for k_, v in rer_block.items()
                                          if isinstance(v, bool)))

    stored_log = None
    if keep_log:
        stored_log = store_log(q, K, n, indptr, indices, forward["values"])

    level = {
        "q": q, "fibonacci_index": FIBONACCI_INDEX[q], "site_count": n, "carrier_count": n, "rounds": K,
        "centre_axis": centre_axis, "centre_site": centre,
        "records": {"source_records_sha256": records_sha, "equals_source_net": bool(records_sha == frozen_records_sha),
                    "maximum_word_length": int(lengths.max()), "sum_word_lengths": int(lengths.sum()),
                    "word_length_bound": 27 * (q - 1), "word_lengths_within_bound": bool(lengths.max() <= 27 * (q - 1)),
                    "record_examples": [records[i] for i in sorted({0, centre, n - 1})]},
        "load_placement": {"round_trip_exact": round_trip, "total_load": int(loads.sum()),
                           "maximum_port_load": int(loads.max()), "all_loads_nonnegative": bool(loads.min() >= 0),
                           "load_sum_equals_record_l1": bool(int(loads.sum()) == int(np.abs(z).sum()))},
        "readback": {"float_readback_equals_signed_generator_sum": readback_equals_signed_sum,
                     "exact_sample": exact_checks, "metric_identity": identity},
        "neighbours": {"candidate_pairs": nb["candidate_pairs"], "accepted_pairs": nb["accepted_pairs"],
                       "rejected_candidates": nb["rejected_candidates"],
                       "borderline_candidates": nb["borderline_candidates"],
                       "undirected_spatial_edges": nb["accepted_pairs"],
                       "minimum_neighbor_count_including_wait": int(degrees.min()),
                       "maximum_neighbor_count_including_wait": int(degrees.max()),
                       "neighbors_including_wait_sha256": nb_sha,
                       "equals_source_net_digest": bool(nb_sha == frozen["neighbors_including_wait_sha256"]),
                       "equals_source_net_edge_count": bool(nb["accepted_pairs"] == frozen["undirected_spatial_edges"])},
        "reads": {"forward": {k_: forward[k_] for k_ in ("audit_trace_sha256", "layer_value_sha256", "layer_value_sums",
                                                         "event_count", "authenticated_read_count")},
                  "intervention": {k_: intervention[k_] for k_ in ("audit_trace_sha256", "layer_value_sha256",
                                                                   "layer_value_sums", "event_count",
                                                                   "authenticated_read_count")},
                  "rer_cross_check": rer_block},
        "provenance": {"edge_count": prov["edge_count"], "single_writer": prov["single_writer"],
                       "all_reads_resolved": prov["all_reads_resolved"],
                       "parents_precede_children": prov["parents_precede_children"],
                       "derived_rank_equals_round": prov["derived_rank_equals_round"],
                       "read_relation_identical_across_rounds": prov["read_relation_identical_across_rounds"],
                       "read_relation_sha256": relation_sha,
                       "read_relation_equals_neighbour_digest": bool(relation_sha == nb_sha),
                       "future_cone_counts": future_counts, "future_cone_sha256": future_sha,
                       "centre_interval": {"inclusive_event_count": int(sum(interval_counts)),
                                           "counts_by_round": interval_counts,
                                           "event_set_sha256": interval_sha,
                                           "layered_order_counts": layered_counts,
                                           "layered_order_event_set_sha256": layered_sha,
                                           "source_net_layered_counts": sn_counts,
                                           "source_net_layered_event_set_sha256": sn_sha,
                                           "equals_layered_order": bool(interval_sha == layered_sha),
                                           "equals_source_net_layered_order": bool(interval_sha == sn_sha),
                                           "strict_pair_count": pairs[K],
                                           "equals_source_net_strict_pair_count": bool(
                                               frozen_rows[K]["strict_pair_count"] == pairs[K])},
                       "shell_counts_from_centre": np.bincount(alpha, minlength=K + 1)[:K + 1].tolist(),
                       "support_site_count": int(len(support))},
        "intervention": {"site": centre, "rounds": support_rows,
                         "equals_future_cone_all_rounds": bool(all(r["equals_future_cone"] for r in support_rows)),
                         "equals_source_net_reachable_ids_all_rounds": bool(all(
                             r.get("equals_source_net_reachable_ids") for r in support_rows[1:]))},
        "manifold": {"intervals": intervals, "count_clock": clock,
                     "all_intervals_equal_source_net": bool(all(r["equals_source_net"] for r in intervals))},
        "operation_costs": {"per_round": cost_rows, "total_reads": total_reads, "total_writes": total_writes,
                            "total_read_bytes": total_read_bytes, "total_write_bytes": total_write_bytes,
                            "total_bytes": total_read_bytes + total_write_bytes,
                            "sum_word_lengths": int(lengths.sum()), "maximum_word_length": int(lengths.max()),
                            "reads_per_round_equal_twice_edges_plus_sites": bool(
                                all(r["reads"] == 2 * nb["accepted_pairs"] + n for r in cost_rows[1:]))},
        "stored_log": stored_log,
    }
    _log(f"q={q}: sites={n} K={K} readback={t_readback:.1f}s neighbours={t_neighbours:.1f}s "
         f"reads={t_reads:.1f}s provenance={t_prov:.1f}s pairs={t_pairs:.1f}s total={time.time() - t0:.1f}s")
    return level


def store_log(q: int, K: int, n: int, indptr: np.ndarray, indices: np.ndarray, values: list[list[int]]) -> dict:
    """Write the complete event log of one level as gzip-compressed canonical JSON."""
    events = []
    for j in range(K + 1):
        for i in range(n):
            reads = [] if j == 0 else [[int(r), j] for r in indices[indptr[i]:indptr[i + 1]].tolist()]
            events.append({"event": [j, i], "write": [i, j + 1, str(values[j][i])], "reads": reads})
    payload = {"schema": LOG_SCHEMA, "q": q, "rounds": K, "site_count": n,
               "read_record": "[register, version]; the reader event is the enclosing event",
               "write_record": "[register, version, value]; values as decimal strings",
               "seed_law": "q_i(0) = i + 1", "read_law": "q_i(j) = 1 + sum of the values read", "events": events}
    data = canonical(payload)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"q{q}_event_log.json.gz"
    with gzip.GzipFile(path, "wb", mtime=0) as fh:
        fh.write(data)
    return {"path": str(path.relative_to(ROOT)), "uncompressed_sha256": hashlib.sha256(data).hexdigest(),
            "uncompressed_bytes": len(data), "event_count": len(events)}


# --------------------------------------------------------------------------
# Receipt
# --------------------------------------------------------------------------


def rer_receipt() -> dict:
    path = RER_ROOT / RER_RECEIPT
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != RER_RECEIPT_SHA256:
        raise ValueError(f"RER receipt digest {sha} differs from the pinned {RER_RECEIPT_SHA256}")
    return json.loads(data)


def source_net_families() -> dict[int, dict]:
    receipt = json.loads((ROOT / SOURCE_NET_RECEIPT).read_bytes())
    return {fam["q"]: fam for lv in receipt["levels"] for fam in lv["families"] if fam["dimension"] == 3}


def carrier_block() -> dict:
    g6 = gram6_exact()
    return {
        "ports": carrier.PORT_COUNT, "seams": carrier.SEAM_COUNT, "faces": carrier.FACE_COUNT,
        "rotations": carrier.ROTATION_COUNT, "antipode": list(carrier.antipode()),
        "positive_port_basis": list(POSITIVE_PORTS),
        "axis_to_port": {str(k): {"positive": p, "negative": carrier.antipode()[p]} for k, p in enumerate(POSITIVE_PORTS)},
        "gram6_qsqrt5": [[q5s(v) for v in row] for row in g6],
        "gram_sign_pattern": gram_sign_pattern().tolist(),
        "gram_entries_by_port_distance": {"0": "1", "1": "1/sqrt5", "2": "-1/sqrt5", "3": "-1"},
        "generator_frame_qsqrt5": [[q5s(v) for v in row] for row in GENERATOR_FRAME_QSQRT5],
        "generator_frame_source": "data/repair_closure/port_gram_completion_bridge_receipt.json "
                                  "exact_signed_module_completion.raw_generator_coordinates_qsqrt5",
        "generator_frame_is_isometric_for_gram6": generator_frame_is_isometric(),
        "load_placement_rule": "axis k of z(b) sits on the antipodal pair (p_k, -p_k) with p_k = positive_port_basis[k]; "
                               "N_{p_k} = max(z_k, 0), N_{-p_k} = max(-z_k, 0); antipodal-odd quotient N_{p_k} - N_{-p_k} = z_k",
        "readback_law": "x = 2 P_slow N = sum_k z_k v_{p_k}, v_p = 2 P_slow e_p, <v_p, v_q> = 4 (P_slow)_pq = G_pq",
        "chart_role": "an orthonormal basis of range(P_slow) is used only to propose candidate neighbours; "
                      "every decision is the exact Q(sqrt5) sign of 5 q ||dx||^2 - 5 L^2",
        "carrier_self_checks": carrier.check_carrier(),
    }


def build(levels=LEVELS, processes: int | None = None, keep_logs: bool = True) -> dict:
    processes = processes or max(1, min(6, os.cpu_count() or 1))
    frozen = source_net_families()
    theirs = {lv["q"]: lv for lv in rer_receipt()["levels"]}
    rows = []
    with tempfile.TemporaryDirectory(prefix="oph_carrier_source_net_") as tmp:
        workdir = Path(tmp)
        for q in levels:
            rows.append(build_level(q, frozen[q], theirs.get(q) if q in RER_LEVELS else None,
                                    processes=processes, workdir=workdir,
                                    keep_log=keep_logs and q in STORED_LOG_LEVELS))
    pins = {p: file_sha256(ROOT / p) for p in LOCAL_PINS}
    pins.update({"reverse-engineering-reality/" + p: file_sha256(RER_ROOT / p) for p in RER_PINS})
    for row in rows:
        if row["stored_log"] is not None:
            pins[row["stored_log"]["path"]] = file_sha256(ROOT / row["stored_log"]["path"])
    rer_levels = [r["q"] for r in rows if r["reads"]["rer_cross_check"] is not None]
    return {
        "schema": SCHEMA, "scope": SCOPE, "claim_boundary": CLAIM_BOUNDARY,
        "carrier": carrier_block(),
        "readback_metric": {
            "identity": "||x(b) - x(b')||^2 = (z(b) - z(b'))^T G6 (z(b) - z(b')) = ||s(b) - s(b')||^2 with "
                        "s(b) = L (xi_b1, xi_b2, xi_b3) the paper's source-axis contraction",
            "scale_to_paper_position_s": "1",
            "scale_to_source_net_units_L_squared_qsqrt5": q5s(L_SQUARED_QSQRT5),
            "scale_to_source_net_units_L_squared_Qphi": ["12/5", "-4/5"],
            "discrepancy": "none: the identity is exact in Q(sqrt5) on every checked pair",
            "neighbour_decision": "||x(b) - x(b')||^2 <= L^2/q iff sign((5 q S2 - 10) + (q X + 2) sqrt5) <= 0, "
                                  "S2 = sum_k dz_k^2, X = sum_{k != l} sigma_kl dz_k dz_l",
        },
        "read_law": "q_i(0) = i + 1 (+1 at the intervention site); q_i(j) = 1 + sum of version-j values of every "
                    "neighbour register including i, for j = 1..K_q, K_q = ceil(sqrt q)",
        "round_rule": "one event per carrier and round; round j reads version j and writes version j + 1",
        "edge_radius": "a_q = L/sqrt(q) on the readback metric",
        "provenance_rule": "edge (writer of register version v, reader of version v); one writer per version; "
                           "seeds have no parents; no declared parent lists, no layer labels as input "
                           "(oph_fpe/bulk/source_derived_causal_order.py::generated_provenance_edges, reimplemented)",
        "byte_model": {"site_id_bytes": ID_BYTES, "version_bytes": VERSION_BYTES,
                       "value_bytes": "minimal unsigned big-endian length of the integer value",
                       "read_record_bytes": "reader id + register id + version + value read",
                       "write_record_bytes": "register id + version + value written"},
        "audit_chain": "RER source_net_causet.trace: chain = sha256(chain || canonical([[j, i], reads, "
                       "[i, j + 1, [j, i], value])), reads = [[r, j, [j - 1, r], value_read], ...]",
        "dynamic_population_note": (
            "The population is held fixed during the reads, the paper's hypothesis of keeping the full "
            "population at each layer. A population whose records move by canonical repairs during the "
            "reads is work in progress: every accepted seam repair changes the loads by the repair mean, "
            "so each readback x = 2 P_slow N moves by 2 P_slow of the load change, the neighbour decision "
            "||x(b) - x(b')|| <= a_q is re-decided at every round, and the read relation, hence the "
            "generated order, is time dependent. The antipodal-even part of a repair leaves the readback "
            "fixed; only its antipodal-odd part moves the position."),
        "levels": rows,
        "rer_cross_check": {"receipt": "reverse-engineering-reality/" + RER_RECEIPT,
                            "receipt_sha256": RER_RECEIPT_SHA256, "levels_compared": rer_levels,
                            "all_agree": bool(rer_levels) and all(
                                r["reads"]["rer_cross_check"]["all_agree"] for r in rows
                                if r["reads"]["rer_cross_check"] is not None)},
        "source_net_cross_check": {
            "receipt": SOURCE_NET_RECEIPT,
            "neighbour_digests_equal": bool(all(r["neighbours"]["equals_source_net_digest"] for r in rows)),
            "records_equal": bool(all(r["records"]["equals_source_net"] for r in rows)),
            "centre_intervals_equal": bool(all(r["provenance"]["centre_interval"]["equals_source_net_layered_order"]
                                               and r["provenance"]["centre_interval"]["equals_source_net_strict_pair_count"]
                                               for r in rows)),
            "manifold_readouts_equal": bool(all(r["manifold"]["all_intervals_equal_source_net"]
                                                and r["manifold"]["count_clock"]["equals_source_net"] for r in rows)),
            "intervention_supports_equal": bool(all(r["intervention"]["equals_source_net_reachable_ids_all_rounds"]
                                                    for r in rows)),
        },
        "source_pins": pins,
    }


def repin(path: Path = OUTPUT) -> bytes:
    receipt = json.loads(path.read_bytes())
    pins = {p: file_sha256(ROOT / p) for p in LOCAL_PINS}
    pins.update({"reverse-engineering-reality/" + p: file_sha256(RER_ROOT / p) for p in RER_PINS})
    for row in receipt["levels"]:
        if row.get("stored_log"):
            pins[row["stored_log"]["path"]] = file_sha256(ROOT / row["stored_log"]["path"])
    receipt["source_pins"] = pins
    return canonical(receipt)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rebuild and write the receipt")
    parser.add_argument("--check", action="store_true", help="rebuild and compare with the receipt")
    parser.add_argument("--repin", action="store_true", help="refresh the file pins of the committed receipt")
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--levels", type=int, nargs="*", default=None)
    parser.add_argument("--no-logs", action="store_true", help="do not write the compressed event logs")
    args = parser.parse_args(argv)
    if args.repin:
        data = repin()
        OUTPUT.write_bytes(data)
        print("CARRIER_SOURCE_NET_REPINNED", len(data), hashlib.sha256(data).hexdigest())
        return
    receipt = build(levels=tuple(args.levels) if args.levels else LEVELS, processes=args.processes,
                    keep_logs=not args.no_logs)
    data = canonical(receipt)
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_bytes(data)
    if args.check and OUTPUT.read_bytes() != data:
        raise ValueError("carrier source-net receipt is stale")
    print("CARRIER_SOURCE_NET_REALIZED", len(data), hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
