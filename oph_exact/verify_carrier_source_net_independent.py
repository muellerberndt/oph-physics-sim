"""Independent replay of the carrier source-net receipt (lane A-real).

The producer is not imported, and neither is the shared carrier module.  The
twelve-port incidence is rebuilt here from a copy of the twenty oriented
faces; the slow-band projector is the numpy eigenprojector of the seam
Laplacian, checked against the exact ``Q(sqrt5)`` matrix with entries
``{1, 1/sqrt5, -1/sqrt5, -1}/4`` by port distance (idempotent, trace three,
Laplacian eigenmatrix for ``5 - sqrt5``); the golden records, the declared
load placement, the readback, a dense all-pairs exact neighbour decision, the
RER trace law with its audit chain, a dictionary-based read-after-write
provenance rule with explicit descendant sets, and the interval readouts are
all rebuilt at ``q = 5`` and ``q = 8`` and compared with the receipt exactly.
The stored event logs of those levels are replayed on their own.  Levels
above ``q = 8`` are checked for internal consistency and against the frozen
source-net receipt.  File pins, the pinned theory receipt and the schema are
checked.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from fractions import Fraction
from math import exp, isqrt, lgamma, log
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RER_ROOT = Path(os.environ.get("OPH_RER_ROOT", str(ROOT.parent / "reverse-engineering-reality")))
OUTPUT = ROOT / "data/exact/carrier_source_net_receipt.json"
SCHEMA = "oph.exact.carrier-source-net.v1"
LOG_SCHEMA = "oph.exact.carrier-source-net-log.v1"
LEVELS = (5, 8, 13, 21, 34, 55, 89)
REBUILT_LEVELS = (5, 8)
STORED_LOG_LEVELS = (5, 8)
FIBONACCI_INDEX = {5: 5, 8: 6, 13: 7, 21: 8, 34: 9, 55: 10, 89: 11, 144: 12}
POSITIVE_PORTS = (0, 1, 4, 5, 8, 9)
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
RER_RECEIPT_SHA256 = "c0f790ad383039186371b7a8020f190371fa533ad634010f01f9c271f1545681"
SOURCE_NET_RECEIPT = "data/exact/source_net_causal_limit_receipt.json"
ID_BYTES = 4
VERSION_BYTES = 2
FLOAT_TOLERANCE = 1e-12
# Copy of oph_fpe.dynamics.self_readback_repair_closure.ORIENTED_BASE_FACES.
ORIENTED_FACES = (
    (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
    (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
    (8, 6, 7), (9, 8, 1),
)
GENERATOR_FRAME = (
    ((-1, 0), (Fraction(1, 2), Fraction(1, 2)), (0, 0)),
    ((1, 0), (Fraction(1, 2), Fraction(1, 2)), (0, 0)),
    ((0, 0), (-1, 0), (Fraction(1, 2), Fraction(1, 2))),
    ((0, 0), (1, 0), (Fraction(1, 2), Fraction(1, 2))),
    ((Fraction(1, 2), Fraction(1, 2)), (0, 0), (-1, 0)),
    ((Fraction(1, 2), Fraction(1, 2)), (0, 0), (1, 0)),
)
EXPECTED_SCOPE = {
    "supplied": {"load_placement": True, "read_law": True, "round_rule": True,
                 "one_event_per_site_and_round": True, "fixed_population": True,
                 "source_records_z_of_b": True, "edge_radius_a_q": True},
    "produced_by_carriers": {"positions": True, "neighbour_graph": True, "reads": True, "provenance_order": True},
    "not_claimed": {"population_selected_by_native_repair": False, "physical_clock": False, "spacetime": False,
                    "continuum_limit": False, "dynamic_population_under_repairs": False},
    "exact_readback_metric_identity": True,
    "exact_neighbour_decisions_in_Qsqrt5": True,
    "read_write_traces_executed": True,
    "full_traces_stored": False,
    "order_generated_from_log_without_declared_parents": True,
}


class CarrierSourceNetVerificationError(ValueError):
    pass


def require(condition, label: str) -> None:
    if not condition:
        raise CarrierSourceNetVerificationError(label)


# --------------------------------------------------------------------------
# Strict canonical JSON
# --------------------------------------------------------------------------


def _pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise CarrierSourceNetVerificationError("duplicate key " + key)
        out[key] = value
    return out


def _forbidden(token):
    raise CarrierSourceNetVerificationError("non-finite JSON token " + token)


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


def rounded(x: float) -> float:
    return float(f"{float(x):.12g}")


# --------------------------------------------------------------------------
# Q(sqrt5): (a, b) means a + b*sqrt5 with rational a, b
# --------------------------------------------------------------------------


def r5(a, b=0):
    return (Fraction(a), Fraction(b))


def r5_add(x, y):
    return (x[0] + y[0], x[1] + y[1])


def r5_sub(x, y):
    return (x[0] - y[0], x[1] - y[1])


def r5_mul(x, y):
    return (x[0] * y[0] + 5 * x[1] * y[1], x[0] * y[1] + x[1] * y[0])


def r5_sign(x) -> int:
    a, b = x
    if b == 0:
        return (a > 0) - (a < 0)
    if a >= 0 and b > 0:
        return 1
    if a <= 0 and b < 0:
        return -1
    d = a * a - 5 * b * b
    return ((d > 0) - (d < 0)) * (1 if a > 0 else -1)


def r5_strings(x) -> list[str]:
    return [str(Fraction(x[0])), str(Fraction(x[1]))]


def sign_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    d = a * a - 5 * b * b
    mixed = np.sign(d) * np.where(a > 0, 1, -1)
    return np.where(b == 0, np.sign(a), np.where((a >= 0) & (b > 0), 1, np.where((a <= 0) & (b < 0), -1, mixed)))


L_SQUARED = r5(2, Fraction(-2, 5))


# --------------------------------------------------------------------------
# Own carrier incidence and slow-band projector
# --------------------------------------------------------------------------


class Carrier:
    def __init__(self) -> None:
        seams = sorted({tuple(sorted((f[i], f[(i + 1) % 3]))) for f in ORIENTED_FACES for i in range(3)})
        require(len(seams) == 30, "thirty seams from the faces")
        self.seams = seams
        adjacency = np.zeros((12, 12), dtype=np.int64)
        for i, j in seams:
            adjacency[i, j] = adjacency[j, i] = 1
        require(np.all(adjacency.sum(axis=1) == 5), "five seams per port")
        self.adjacency = adjacency
        self.laplacian = 5 * np.eye(12, dtype=np.int64) - adjacency
        dist = np.full((12, 12), -1, dtype=np.int64)
        for p in range(12):
            dist[p, p] = 0
            frontier = [p]
            d = 0
            while frontier:
                d += 1
                nxt = []
                for u in frontier:
                    for v in range(12):
                        if adjacency[u, v] and dist[p, v] < 0:
                            dist[p, v] = d
                            nxt.append(v)
                frontier = nxt
        require(int(dist.max()) == 3 and int(dist.min()) == 0, "port diameter three")
        self.distance = dist
        anti = []
        for p in range(12):
            far = [v for v in range(12) if dist[p, v] == 3]
            require(len(far) == 1, "unique antipode")
            anti.append(far[0])
        require(all(anti[anti[p]] == p and anti[p] != p for p in range(12)), "antipode is a free involution")
        self.antipode = tuple(anti)
        # Numerical eigenprojector of the slow band 5 - sqrt5.
        w, v = np.linalg.eigh(self.laplacian.astype(float))
        slow = np.abs(w - (5.0 - 5.0 ** 0.5)) < 1e-9
        require(int(slow.sum()) == 3, "slow band multiplicity three")
        basis = v[:, slow]
        self.p_slow = basis @ basis.T
        self.chart = basis
        # Exact matrix by distance and its Q(sqrt5) checks.
        by_distance = {0: r5(1), 1: r5(0, Fraction(1, 5)), 2: r5(0, Fraction(-1, 5)), 3: r5(-1)}
        gram = [[by_distance[int(dist[p, q])] for q in range(12)] for p in range(12)]
        p_exact = [[r5_mul(gram[p][q], r5(Fraction(1, 4))) for q in range(12)] for p in range(12)]
        square = [[self._dot(p_exact, p_exact, p, q) for q in range(12)] for p in range(12)]
        require(square == p_exact, "exact projector is idempotent")
        trace = r5(0)
        for p in range(12):
            trace = r5_add(trace, p_exact[p][p])
        require(trace == r5(3), "exact projector has trace three")
        lap = [[r5(int(self.laplacian[p, q])) for q in range(12)] for p in range(12)]
        lp = [[self._dot(lap, p_exact, p, q) for q in range(12)] for p in range(12)]
        mu = r5(5, -1)
        require(all(lp[p][q] == r5_mul(mu, p_exact[p][q]) for p in range(12) for q in range(12)),
                "exact projector is the 5 - sqrt5 eigenmatrix")
        numeric = np.array([[float(x[0]) + float(x[1]) * 5.0 ** 0.5 for x in row] for row in p_exact])
        require(np.max(np.abs(numeric - self.p_slow)) < 1e-12, "numpy eigenprojector equals the exact matrix")
        self.gram_exact = gram
        self.sign = np.zeros((6, 6), dtype=np.int64)
        for k in range(6):
            for l in range(6):
                if k != l:
                    d = int(dist[POSITIVE_PORTS[k], POSITIVE_PORTS[l]])
                    require(d in (1, 2), "positive ports pairwise non-antipodal")
                    self.sign[k, l] = 1 if d == 1 else -1
        self.gram6 = [[gram[p][q] for q in POSITIVE_PORTS] for p in POSITIVE_PORTS]

    @staticmethod
    def _dot(x, y, p, q):
        acc = r5(0)
        for t in range(12):
            acc = r5_add(acc, r5_mul(x[p][t], y[t][q]))
        return acc

    def frame_is_isometric(self) -> bool:
        inv = r5(Fraction(5, 2), Fraction(1, 2))
        norm = inv[0] * inv[0] - 5 * inv[1] * inv[1]
        inv = (inv[0] / norm, -inv[1] / norm)
        for k in range(6):
            for l in range(6):
                dot = r5(0)
                for c in range(3):
                    dot = r5_add(dot, r5_mul(r5(*GENERATOR_FRAME[k][c]), r5(*GENERATOR_FRAME[l][c])))
                if r5_mul(dot, inv) != self.gram6[k][l]:
                    return False
        return True


# --------------------------------------------------------------------------
# Own population, readback and neighbours
# --------------------------------------------------------------------------


def golden_floor(b: int) -> int:
    return (b + isqrt(5 * b * b)) // 2


def ceil_sqrt(q: int) -> int:
    return isqrt(q) + (isqrt(q) ** 2 < q)


def records(q: int):
    sites = [(a, b, c) for a in range(q) for b in range(q) for c in range(q)]
    z = []
    currents = []
    for b in sites:
        m = [-golden_floor(v) for v in b]
        zz = (b[1] - m[0], b[1] + m[0], b[2] - m[1], b[2] + m[1], b[0] - m[2], b[0] + m[2])
        require(sum(zz) % 2 == 0, "record even sum")
        h = sum(zz) // 2
        cc = (-zz[5], h - zz[0], h - zz[2] - zz[3], h - zz[1] - zz[4] - zz[5], h - zz[2] - zz[3] - zz[4], zz[3])
        z.append(zz)
        currents.append(cc)
    return sites, np.array(z, dtype=np.int64), np.array(currents, dtype=np.int64)


def centre_of(q: int) -> tuple[int, int]:
    """Nearest orbit value to 1/2, smallest label on ties (RER rule), in the sqrt5 basis."""
    best = 0
    best_gap = None
    for b in range(q):
        xi = r5(Fraction(-golden_floor(b)) + Fraction(b, 2), Fraction(b, 2))
        gap = r5_sub(xi, r5(Fraction(1, 2)))
        gap = r5_mul(gap, gap)
        if best_gap is None or r5_sign(r5_sub(gap, best_gap)) < 0:
            best, best_gap = b, gap
    return best, (best * q + best) * q + best


def place(z: np.ndarray, antipode) -> np.ndarray:
    loads = np.zeros((len(z), 12), dtype=np.int64)
    for k, p in enumerate(POSITIVE_PORTS):
        loads[:, p] = np.maximum(z[:, k], 0)
        loads[:, antipode[p]] = np.maximum(-z[:, k], 0)
    return loads


def quotient(loads: np.ndarray, antipode) -> np.ndarray:
    return np.stack([loads[:, p] - loads[:, antipode[p]] for p in POSITIVE_PORTS], axis=1)


def source_distance(sites, i: int, j: int):
    """Squared source distance in units of L^2 in the sqrt5 basis."""
    A = B = 0
    for axis in range(3):
        bi, bj = sites[i][axis], sites[j][axis]
        dm = -golden_floor(bi) + golden_floor(bj)
        db = bi - bj
        A += dm * dm + db * db
        B += 2 * dm * db + db * db
    return r5(Fraction(A) + Fraction(B, 2), Fraction(B, 2))


def exact_readback(loads_row, p_exact_gram) -> list:
    """x = 2 P N exactly, P = G/4, so x_r = (1/2) sum_c G_rc N_c."""
    out = []
    for r in range(12):
        acc = r5(0)
        for c in range(12):
            if loads_row[c]:
                acc = r5_add(acc, r5_mul(p_exact_gram[r][c], r5(Fraction(int(loads_row[c]), 2))))
        out.append(acc)
    return out


def all_pairs_neighbours(z: np.ndarray, q: int, sign: np.ndarray) -> list[list[int]]:
    """Dense exact decision on every pair; rows ascending, same-site read included."""
    n = len(z)
    rows = []
    for i in range(n):
        dz = z[i][None, :] - z
        s2 = (dz * dz).sum(axis=1)
        x = np.einsum("mk,kl,ml->m", dz, sign, dz)
        ok = sign_array(5 * q * s2 - 10, q * x + 2) <= 0
        rows.append(np.flatnonzero(ok).tolist())
        require(i in rows[-1], "same-site read")
    return rows


# --------------------------------------------------------------------------
# Own trace, log and provenance rule
# --------------------------------------------------------------------------


def trace(rows: list[list[int]], K: int, intervention=None) -> dict:
    n = len(rows)
    chain = "0" * 64
    prev = None
    layer_hashes, layer_sums, values, events = [], [], [], []
    for j in range(K + 1):
        cur = [None] * n
        for i in range(n):
            reads = [] if j == 0 else [[r, j, [j - 1, r], prev[r]] for r in rows[i]]
            value = 1 + i + (i == intervention) if j == 0 else 1 + sum(r[3] for r in reads)
            cur[i] = value
            material = [[j, i], reads, [i, j + 1, [j, i], value]]
            chain = hashlib.sha256(bytes.fromhex(chain) + packed(material)).hexdigest()
            events.append({"event": [j, i], "write": [i, j + 1, str(value)], "reads": [[r, j] for r in rows[i]] if j else []})
        layer_hashes.append(hashed(cur))
        layer_sums.append(sum(cur))
        values.append(cur)
        prev = cur
    return {"audit_trace_sha256": chain, "layer_value_sha256": layer_hashes, "layer_value_sums": layer_sums,
            "event_count": (K + 1) * n, "authenticated_read_count": K * sum(map(len, rows)),
            "values": values, "events": events}


def provenance(events: list[dict]) -> dict:
    """Read-after-write rule on a list of log events (execution order); event ids are positions."""
    writer_of = {}
    for e, ev in enumerate(events):
        key = (int(ev["write"][0]), int(ev["write"][1]))
        require(key not in writer_of, "single writer per register version")
        writer_of[key] = e
    parents = []
    for e, ev in enumerate(events):
        ps = []
        for r, v in ev["reads"]:
            key = (int(r), int(v))
            require(key in writer_of, "read of an unwritten version")
            p = writer_of[key]
            require(p != e, "self read of own version")
            require(p < e, "parent precedes child in the log")
            ps.append(p)
        parents.append(ps)
    n_events = len(events)
    rank = [0] * n_events
    children = [[] for _ in range(n_events)]
    for e, ps in enumerate(parents):
        if ps:
            rank[e] = 1 + max(rank[p] for p in ps)
        for p in ps:
            children[p].append(e)
    descendants = [set() for _ in range(n_events)]
    for e in range(n_events - 1, -1, -1):
        acc = set()
        for c in children[e]:
            acc.add(c)
            acc |= descendants[c]
        descendants[e] = acc
    return {"parents": parents, "children": children, "rank": rank, "descendants": descendants,
            "edge_count": sum(map(len, parents))}


def mm_fraction(d: float) -> float:
    return exp(lgamma(d + 1.0) + lgamma(d / 2.0) - log(2.0) - lgamma(3.0 * d / 2.0))


def mm_dimension(f: float):
    if not 0.0 < f < 1.0:
        return None
    low, high = 1.01, 20.0
    if not (mm_fraction(high) <= f <= mm_fraction(low)):
        return None
    for _ in range(96):
        mid = (low + high) / 2.0
        if mm_fraction(mid) > f:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def byte_length(v: int) -> int:
    return max(1, (int(v).bit_length() + 7) // 8)


def interval_readout(prov: dict, n: int, K: int, centre: int, k: int) -> dict:
    """Interval between the seed event of the centre and its round-k event, from descendant sets."""
    seed = centre
    top = k * n + centre
    future = prov["descendants"][seed] | {seed}
    members = sorted(e for e in future if e == top or top in prov["descendants"][e])
    inside = set(members)
    counts = [0] * (k + 1)
    for e in members:
        counts[prov["rank"][e]] += 1
    C = sum(len(prov["descendants"][e] & inside) for e in members)
    N = len(members)
    row = {"layers": k, "inclusive_event_count": N, "counts_by_layer": counts, "strict_pair_count": C,
           "events": [[prov["rank"][e], e - prov["rank"][e] * n] for e in members]}
    if N >= 2:
        f = Fraction(2 * C, N * (N - 1))
        dim = mm_dimension(float(f))
        row.update({"ordering_fraction": str(f), "ordering_fraction_float": rounded(float(f)),
                    "distance_to_one_tenth": rounded(float(f - Fraction(1, 10))),
                    "myrheim_meyer_dimension": None if dim is None else rounded(dim)})
    else:
        row.update({"ordering_fraction": None, "ordering_fraction_float": None,
                    "distance_to_one_tenth": None, "myrheim_meyer_dimension": None})
    return row


# --------------------------------------------------------------------------
# Level checks
# --------------------------------------------------------------------------


def frozen_families() -> dict[int, dict]:
    receipt = json.loads((ROOT / SOURCE_NET_RECEIPT).read_bytes())
    return {fam["q"]: fam for lv in receipt["levels"] for fam in lv["families"] if fam["dimension"] == 3}


def check_rebuilt_level(level: dict, car: Carrier, frozen: dict, rer_level: dict | None) -> dict:
    q = level["q"]
    n = q ** 3
    K = ceil_sqrt(q)
    require(level["site_count"] == n and level["carrier_count"] == n and level["rounds"] == K, "level census " + str(q))
    centre_axis, centre = centre_of(q)
    require(level["centre_axis"] == centre_axis and level["centre_site"] == centre, "centre " + str(q))
    sites, z, currents = records(q)
    rec = [[list(b), zz, cc] for b, zz, cc in zip(sites, z.tolist(), currents.tolist())]
    require(level["records"]["source_records_sha256"] == hashed(rec), "records digest " + str(q))
    require(level["records"]["source_records_sha256"] == frozen["source_records_sha256"], "records vs source net")
    lengths = np.abs(currents).sum(axis=1)
    require(level["records"]["maximum_word_length"] == int(lengths.max())
            and level["records"]["sum_word_lengths"] == int(lengths.sum())
            and level["records"]["word_length_bound"] == 27 * (q - 1), "word lengths " + str(q))
    require(level["records"]["record_examples"] == [rec[i] for i in sorted({0, centre, n - 1})], "record examples")

    loads = place(z, car.antipode)
    require(np.array_equal(quotient(loads, car.antipode), z), "own placement round trip")
    lp = level["load_placement"]
    require(lp["round_trip_exact"] is True and lp["total_load"] == int(loads.sum())
            and lp["maximum_port_load"] == int(loads.max()) and lp["all_loads_nonnegative"] is True
            and lp["load_sum_equals_record_l1"] is True and int(loads.sum()) == int(np.abs(z).sum()), "load placement " + str(q))

    # Readback: exact 12-vector on the receipt's sample; metric identity on all pairs.
    rb = level["readback"]
    require(rb["float_readback_equals_signed_generator_sum"] is True, "float readback flag")
    sample = rb["exact_sample"]["sample_sites"]
    require(all(0 <= s < n for s in sample) and {0, centre, n - 1} <= set(sample), "sample sites")
    xs = {i: exact_readback(loads[i], car.gram_exact) for i in sample}
    for i in sample:
        quad = r5(0)
        for k in range(6):
            for l in range(6):
                quad = r5_add(quad, r5_mul(r5(int(z[i, k]) * int(z[i, l])), car.gram6[k][l]))
        norm = r5(0)
        for v in xs[i]:
            norm = r5_add(norm, r5_mul(v, v))
        require(norm == quad, "exact gram identity")
        for c in range(3):
            acc = r5(0)
            for k in range(6):
                acc = r5_add(acc, r5_mul(r5(int(z[i, k])), r5(*GENERATOR_FRAME[k][c])))
            acc = r5_mul(acc, r5(Fraction(1, 2)))
            b = sites[i][c]
            require(acc == r5(Fraction(-golden_floor(b)) + Fraction(b, 2), Fraction(b, 2)), "paper contraction")
    pairs = 0
    for a_ in range(len(sample)):
        for b_ in range(a_ + 1, len(sample)):
            i, j = sample[a_], sample[b_]
            diff = [r5_sub(xs[i][r], xs[j][r]) for r in range(12)]
            lhs = r5(0)
            for v in diff:
                lhs = r5_add(lhs, r5_mul(v, v))
            require(lhs == r5_mul(L_SQUARED, source_distance(sites, i, j)), "exact metric identity on sample")
            pairs += 1
    require(rb["exact_sample"] == {"sample_sites": sample, "exact_gram_identity": True,
                                   "exact_position_equals_paper_contraction": True,
                                   "exact_metric_identity_pairs": pairs, "exact_metric_identity": True}, "exact sample block")
    # All pairs, integer forms: 10 S2 + 2 X sqrt5 == 20 A + (8 B - 4 A) sqrt5.
    floors = np.array([golden_floor(b) for b in range(q)], dtype=np.int64)
    site_arr = np.array(sites, dtype=np.int64)
    checked = 0
    for i in range(n):
        dz = z[i][None, :] - z[i + 1:]
        s2 = (dz * dz).sum(axis=1)
        x = np.einsum("mk,kl,ml->m", dz, car.sign, dz)
        A = np.zeros(n - i - 1, dtype=np.int64)
        B = np.zeros(n - i - 1, dtype=np.int64)
        for axis in range(3):
            bi = site_arr[i, axis]
            bj = site_arr[i + 1:, axis]
            dm = -floors[bi] + floors[bj]
            db = bi - bj
            A += dm * dm + db * db
            B += 2 * dm * db + db * db
        require(np.array_equal(10 * s2, 20 * A) and np.array_equal(2 * x, 8 * B - 4 * A), "metric identity all pairs")
        checked += n - i - 1
    require(rb["metric_identity"] == {"mode": "all_pairs", "pairs_checked": checked, "exact": True, "seed": None},
            "metric identity block " + str(q))

    # Neighbours by a dense exact decision.
    rows = all_pairs_neighbours(z, q, car.sign)
    nb_sha = hashed(rows)
    nb = level["neighbours"]
    edges = (sum(map(len, rows)) - n) // 2
    require(nb["neighbors_including_wait_sha256"] == nb_sha, "neighbour digest " + str(q))
    require(nb["neighbors_including_wait_sha256"] == frozen["neighbors_including_wait_sha256"], "neighbour digest vs source net")
    require(nb["undirected_spatial_edges"] == edges == nb["accepted_pairs"] == frozen["undirected_spatial_edges"], "edge count")
    require(nb["candidate_pairs"] >= nb["accepted_pairs"] and nb["rejected_candidates"] == nb["candidate_pairs"] - nb["accepted_pairs"],
            "candidate arithmetic")
    require(nb["minimum_neighbor_count_including_wait"] == min(map(len, rows))
            and nb["maximum_neighbor_count_including_wait"] == max(map(len, rows)), "neighbour counts")
    require(nb["equals_source_net_digest"] is True and nb["equals_source_net_edge_count"] is True, "neighbour flags")
    require(type(nb["borderline_candidates"]) is int and nb["borderline_candidates"] >= 0, "borderline count")

    # Reads: forward and intervention with the RER chain.
    fwd = trace(rows, K)
    itv = trace(rows, K, intervention=centre)
    for name, mine in (("forward", fwd), ("intervention", itv)):
        theirs = level["reads"][name]
        for key in ("audit_trace_sha256", "layer_value_sha256", "layer_value_sums", "event_count", "authenticated_read_count"):
            require(theirs[key] == mine[key], f"{name} {key} " + str(q))
    if rer_level is not None:
        require(rer_level["forward_execution"]["audit_trace_sha256"] == fwd["audit_trace_sha256"]
                and rer_level["forward_execution"]["layer_value_sha256"] == fwd["layer_value_sha256"]
                and rer_level["center_plus_one_intervention"]["audit_trace_sha256"] == itv["audit_trace_sha256"]
                and rer_level["center_plus_one_intervention"]["layer_value_sha256"] == itv["layer_value_sha256"]
                and rer_level["neighbors_including_wait_sha256"] == nb_sha
                and rer_level["intervention_source_id"] == centre, "theory receipt digests " + str(q))
    block = level["reads"]["rer_cross_check"]
    require(block is not None and block["all_agree"] is True and block["fibonacci_index"] == FIBONACCI_INDEX[q], "rer block")
    require(block["rer_forward_audit_trace_sha256"] == fwd["audit_trace_sha256"]
            and block["rer_intervention_audit_trace_sha256"] == itv["audit_trace_sha256"], "embedded theory digests")
    if rer_level is not None:
        require(block["rer_forward_audit_trace_sha256"] == rer_level["forward_execution"]["audit_trace_sha256"], "embedded vs theory")

    # Provenance from the own log.
    prov = provenance(fwd["events"])
    pv = level["provenance"]
    require(pv["edge_count"] == prov["edge_count"] == K * sum(map(len, rows)), "edge count " + str(q))
    require(all(prov["rank"][e] == e // n for e in range(len(fwd["events"]))), "derived rank equals round")
    for key in ("single_writer", "all_reads_resolved", "parents_precede_children", "derived_rank_equals_round",
                "read_relation_identical_across_rounds", "read_relation_equals_neighbour_digest"):
        require(pv[key] is True, "provenance flag " + key)
    require(pv["read_relation_sha256"] == nb_sha, "read relation digest")
    # Per-round read relation from the log, compared with the neighbour rows.
    for j in range(1, K + 1):
        rel = [[r for r, v in fwd["events"][j * n + i]["reads"]] for i in range(n)]
        require(rel == rows, "read relation per round")
    future = [sorted({centre})]
    for j in range(1, K + 1):
        future.append(sorted(e - j * n for e in prov["descendants"][centre] if prov["rank"][e] == j))
    require(pv["future_cone_counts"] == [len(f) for f in future]
            and pv["future_cone_sha256"] == [hashed(f) for f in future], "future cone " + str(q))
    readouts = {k: interval_readout(prov, n, K, centre, k) for k in range(1, K + 1)}
    ci = pv["centre_interval"]
    top = readouts[K]
    require(ci["inclusive_event_count"] == top["inclusive_event_count"] and ci["counts_by_round"] == top["counts_by_layer"]
            and ci["event_set_sha256"] == hashed(top["events"]) and ci["strict_pair_count"] == top["strict_pair_count"],
            "centre interval " + str(q))
    # Layered order from graph distances on the own rows.
    alpha = [-1] * n
    alpha[centre] = 0
    queue = [centre]
    for u in queue:
        for v in rows[u]:
            if alpha[v] < 0:
                alpha[v] = alpha[u] + 1
                queue.append(v)
    require(min(alpha) >= 0, "connected")
    layered = [[j, s] for j in range(K + 1) for s in range(n) if alpha[s] <= min(j, K - j)]
    require(ci["layered_order_event_set_sha256"] == hashed(layered) == ci["event_set_sha256"], "layered order digest")
    require(ci["layered_order_counts"] == top["counts_by_layer"] == ci["source_net_layered_counts"], "layered counts")
    require(ci["source_net_layered_event_set_sha256"] == ci["event_set_sha256"], "source net layered digest")
    require(ci["equals_layered_order"] is True and ci["equals_source_net_layered_order"] is True
            and ci["equals_source_net_strict_pair_count"] is True, "interval flags")
    require(frozen["vertical_intervals"][K - 1]["strict_pair_count"] == top["strict_pair_count"], "pairs vs source net")
    shells = [sum(1 for a in alpha if a == d) for d in range(K + 1)]
    require(pv["shell_counts_from_centre"] == shells and pv["support_site_count"] == sum(1 for a in alpha if a <= K // 2),
            "shells " + str(q))

    # Intervention rows.
    iv = level["intervention"]
    require(iv["site"] == centre and len(iv["rounds"]) == K + 1, "intervention census")
    probes = {(p["start"], p["layers"]): p for p in frozen["reachability_probes"]}
    response = {r["layer"]: r for r in rer_level["intervention_response"]} if rer_level else None
    for j, row in enumerate(iv["rounds"]):
        changed = [i for i in range(n) if itv["values"][j][i] != fwd["values"][j][i]]
        deltas = [itv["values"][j][i] - fwd["values"][j][i] for i in range(n)]
        expected = {"round": j, "support_count": len(changed), "support_ids_sha256": hashed(changed),
                    "positive_integer_delta_sum": sum(deltas), "positive_integer_delta_maximum": max(deltas),
                    "all_deltas_nonnegative": min(deltas) >= 0, "equals_future_cone": changed == future[j]}
        if j >= 1:
            expected["equals_source_net_reachable_ids"] = probes[(centre, j)]["reachable_ids_sha256"] == hashed(changed)
        expected["equals_rer_support"] = True if response is None else (
            response[j]["support_ids_sha256"] == hashed(changed) and response[j]["support_count"] == len(changed)
            and response[j]["positive_integer_delta_sum"] == sum(deltas)
            and response[j]["positive_integer_delta_maximum"] == max(deltas))
        require(row == expected, f"intervention round {j} at q={q}")
        require(changed == future[j], "intervention equals future cone")
    require(iv["equals_future_cone_all_rounds"] is True and iv["equals_source_net_reachable_ids_all_rounds"] is True,
            "intervention flags")

    # Manifold readouts.
    frozen_rows = {r["layers"]: r for r in frozen["vertical_intervals"]}
    for row in level["manifold"]["intervals"]:
        mine = readouts[row["layers"]]
        expected = {k: v for k, v in mine.items() if k != "events"}
        fr = frozen_rows[row["layers"]]
        expected["equals_source_net"] = (fr["inclusive_event_count"] == mine["inclusive_event_count"]
                                         and fr["counts_by_layer"] == mine["counts_by_layer"]
                                         and fr["strict_pair_count"] == mine["strict_pair_count"]
                                         and fr["ordering_fraction"] == mine["ordering_fraction"]
                                         and fr["myrheim_meyer_dimension"] == mine["myrheim_meyer_dimension"])
        require(row == expected, f"manifold interval {row['layers']} at q={q}")
        require(expected["equals_source_net"] is True, "readout vs source net")
    check_clock(level, frozen)
    require(level["manifold"]["all_intervals_equal_source_net"] is True, "manifold flag")

    # Operation costs.
    oc = level["operation_costs"]
    rows_cost = []
    degrees = np.array([len(r) for r in rows], dtype=np.int64)
    for j in range(K + 1):
        cur = fwd["values"][j]
        wb = sum(ID_BYTES + VERSION_BYTES + byte_length(v) for v in cur)
        if j == 0:
            reads, rbytes = 0, 0
        else:
            prev_len = np.array([byte_length(v) for v in fwd["values"][j - 1]], dtype=np.int64)
            reads = int(degrees.sum())
            rbytes = int((degrees * (2 * ID_BYTES + VERSION_BYTES + prev_len)).sum())
        rows_cost.append({"round": j, "reads": reads, "writes": n, "read_bytes": rbytes, "write_bytes": wb,
                          "total_bytes": rbytes + wb, "maximum_value_bytes": max(byte_length(v) for v in cur)})
    require(oc["per_round"] == rows_cost, "operation cost rows " + str(q))
    check_cost_totals(oc, n, edges, int(lengths.sum()), int(lengths.max()))

    # Stored log.
    stored = level["stored_log"]
    require(stored is not None, "stored log expected")
    path = ROOT / stored["path"]
    require(path.is_file() and stored["path"] == f"data/exact/carrier_source_net_logs/q{q}_event_log.json.gz", "stored log path")
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    require(hashlib.sha256(raw).hexdigest() == stored["uncompressed_sha256"] and len(raw) == stored["uncompressed_bytes"],
            "stored log digest")
    log = json.loads(raw.decode("ascii"), object_pairs_hook=_pairs, parse_constant=_forbidden)
    require(packed(log) == raw, "stored log canonical")
    require(log["schema"] == LOG_SCHEMA and log["q"] == q and log["rounds"] == K and log["site_count"] == n
            and stored["event_count"] == len(log["events"]) == (K + 1) * n, "stored log header")
    require(log["events"] == fwd["events"], "stored log equals the own log")
    prov_stored = provenance(log["events"])
    top_stored = interval_readout(prov_stored, n, K, centre, K)
    require(hashed(top_stored["events"]) == ci["event_set_sha256"]
            and top_stored["strict_pair_count"] == ci["strict_pair_count"], "order from the stored log alone")
    return {"q": q, "rebuilt": True, "neighbours_sha256": nb_sha, "strict_pair_count": top["strict_pair_count"]}


def check_clock(level: dict, frozen: dict) -> None:
    K = level["rounds"]
    Kp = K // 2
    rows = {r["layers"]: r for r in level["manifold"]["intervals"]}
    NI, NJ = rows[K]["inclusive_event_count"], rows[Kp]["inclusive_event_count"]
    clock = level["manifold"]["count_clock"]
    expected = {"reference_layers": Kp, "interval_layers": K, "interval_count": NI, "reference_count": NJ,
                "count_clock": rounded((NI / NJ) ** 0.25), "model_time_ratio": rounded(K / Kp),
                "relative_deviation": rounded((NI / NJ) ** 0.25 / (K / Kp) - 1.0)}
    fc = frozen["count_clock"]
    expected["equals_source_net"] = all(fc[k] == expected[k] for k in expected)
    require(clock == expected and expected["equals_source_net"] is True, "count clock " + str(level["q"]))


def check_cost_totals(oc: dict, n: int, edges: int, word_sum: int, word_max: int) -> None:
    per = oc["per_round"]
    require(oc["total_reads"] == sum(r["reads"] for r in per) and oc["total_writes"] == sum(r["writes"] for r in per)
            and oc["total_read_bytes"] == sum(r["read_bytes"] for r in per)
            and oc["total_write_bytes"] == sum(r["write_bytes"] for r in per)
            and oc["total_bytes"] == oc["total_read_bytes"] + oc["total_write_bytes"], "cost totals")
    require(oc["sum_word_lengths"] == word_sum and oc["maximum_word_length"] == word_max, "word lengths in costs")
    require(all(r["reads"] == 2 * edges + n for r in per[1:]) and per[0]["reads"] == 0
            and oc["reads_per_round_equal_twice_edges_plus_sites"] is True, "reads per round")
    require(all(r["total_bytes"] == r["read_bytes"] + r["write_bytes"] and r["writes"] == n for r in per), "cost rows")


def check_large_level(level: dict, frozen: dict) -> dict:
    """Internal consistency and agreement with the frozen source-net receipt at q >= 13."""
    q = level["q"]
    n = q ** 3
    K = ceil_sqrt(q)
    require(level["site_count"] == n and level["carrier_count"] == n and level["rounds"] == K
            and level["fibonacci_index"] == FIBONACCI_INDEX[q], "level census " + str(q))
    centre_axis, centre = centre_of(q)
    require(level["centre_axis"] == centre_axis == frozen["centre_axis"]
            and level["centre_site"] == centre == frozen["intervention_source_id"], "centre " + str(q))
    rc = level["records"]
    require(rc["source_records_sha256"] == frozen["source_records_sha256"] and rc["equals_source_net"] is True
            and rc["maximum_word_length"] == frozen["maximum_word_length"]
            and rc["sum_word_lengths"] == frozen["sum_word_lengths"] and rc["word_length_bound"] == 27 * (q - 1)
            and rc["word_lengths_within_bound"] is True, "records " + str(q))
    sites, z, currents = records(q)
    rec = [[list(b), zz, cc] for b, zz, cc in zip(sites, z.tolist(), currents.tolist())]
    require(rc["source_records_sha256"] == hashed(rec), "records digest " + str(q))
    require(rc["record_examples"] == [rec[i] for i in sorted({0, centre, n - 1})], "record examples " + str(q))
    lp = level["load_placement"]
    require(lp["round_trip_exact"] is True and lp["all_loads_nonnegative"] is True and lp["load_sum_equals_record_l1"] is True
            and lp["total_load"] == int(np.abs(z).sum()) and lp["maximum_port_load"] == int(np.abs(z).max()), "loads " + str(q))
    rb = level["readback"]
    require(rb["float_readback_equals_signed_generator_sum"] is True and rb["exact_sample"]["exact_gram_identity"] is True
            and rb["exact_sample"]["exact_position_equals_paper_contraction"] is True
            and rb["exact_sample"]["exact_metric_identity"] is True and rb["metric_identity"]["exact"] is True
            and rb["metric_identity"]["pairs_checked"] > 0, "readback flags " + str(q))
    mi = rb["metric_identity"]
    if n <= 10000:
        require(mi["mode"] == "all_pairs" and mi["pairs_checked"] == n * (n - 1) // 2 and mi["seed"] is None, "all pairs mode")
    else:
        require(mi["mode"] == "sampled_pairs" and mi["seed"] == 20260909 + q, "sampled mode")
    nb = level["neighbours"]
    require(nb["neighbors_including_wait_sha256"] == frozen["neighbors_including_wait_sha256"]
            and nb["undirected_spatial_edges"] == frozen["undirected_spatial_edges"] == nb["accepted_pairs"]
            and nb["minimum_neighbor_count_including_wait"] == frozen["minimum_neighbor_count_including_wait"]
            and nb["maximum_neighbor_count_including_wait"] == frozen["maximum_neighbor_count_including_wait"]
            and nb["equals_source_net_digest"] is True and nb["equals_source_net_edge_count"] is True
            and nb["rejected_candidates"] == nb["candidate_pairs"] - nb["accepted_pairs"], "neighbours " + str(q))
    edges = nb["accepted_pairs"]
    reads_per_round = 2 * edges + n
    for name in ("forward", "intervention"):
        tr = level["reads"][name]
        require(len(tr["layer_value_sha256"]) == K + 1 and len(tr["layer_value_sums"]) == K + 1
                and tr["event_count"] == (K + 1) * n and tr["authenticated_read_count"] == K * reads_per_round
                and len(tr["audit_trace_sha256"]) == 64, name + " trace census " + str(q))
    fwd_sums, itv_sums = level["reads"]["forward"]["layer_value_sums"], level["reads"]["intervention"]["layer_value_sums"]
    require(fwd_sums[0] == n * (n + 1) // 2 and itv_sums[0] == fwd_sums[0] + 1, "seed sums " + str(q))
    if q == 13:
        require(level["reads"]["rer_cross_check"] is not None and level["reads"]["rer_cross_check"]["all_agree"] is True
                and level["reads"]["rer_cross_check"]["fibonacci_index"] == 7, "theory block at q = 13")
    else:
        require(level["reads"]["rer_cross_check"] is None, "no theory block above q = 13")
    pv = level["provenance"]
    for key in ("single_writer", "all_reads_resolved", "parents_precede_children", "derived_rank_equals_round",
                "read_relation_identical_across_rounds", "read_relation_equals_neighbour_digest"):
        require(pv[key] is True, "provenance flag " + key + " " + str(q))
    require(pv["edge_count"] == K * reads_per_round and pv["read_relation_sha256"] == nb["neighbors_including_wait_sha256"],
            "provenance census " + str(q))
    shells = pv["shell_counts_from_centre"]
    require(shells == frozen["shell_counts_from_centre"] and sum(shells) == n and len(shells) == K + 1, "shells " + str(q))
    cumulative = [sum(shells[:j + 1]) for j in range(K + 1)]
    require(pv["future_cone_counts"] == cumulative and len(pv["future_cone_sha256"]) == K + 1, "future cone " + str(q))
    probes = {(p["start"], p["layers"]): p for p in frozen["reachability_probes"]}
    for j in range(1, K + 1):
        require(pv["future_cone_sha256"][j] == probes[(centre, j)]["reachable_ids_sha256"]
                and pv["future_cone_counts"][j] == probes[(centre, j)]["reachable_count"], "future cone vs probes " + str(q))
    require(pv["support_site_count"] == frozen["vertical_support_site_count"] == sum(shells[:K // 2 + 1]), "support " + str(q))
    ci = pv["centre_interval"]
    counts = [sum(shells[:min(j, K - j) + 1]) for j in range(K + 1)]
    require(ci["counts_by_round"] == counts == ci["layered_order_counts"] == ci["source_net_layered_counts"]
            and ci["inclusive_event_count"] == sum(counts) == frozen["vertical_intervals"][K - 1]["inclusive_event_count"],
            "interval counts " + str(q))
    require(ci["event_set_sha256"] == ci["layered_order_event_set_sha256"] == ci["source_net_layered_event_set_sha256"]
            and ci["equals_layered_order"] is True and ci["equals_source_net_layered_order"] is True, "interval digests " + str(q))
    require(frozen["vertical_pair_counting"] == "exact_all_pairs"
            and ci["strict_pair_count"] == frozen["vertical_intervals"][K - 1]["strict_pair_count"]
            and ci["equals_source_net_strict_pair_count"] is True, "strict pairs " + str(q))
    iv = level["intervention"]
    require(iv["site"] == centre and len(iv["rounds"]) == K + 1 and iv["equals_future_cone_all_rounds"] is True
            and iv["equals_source_net_reachable_ids_all_rounds"] is True, "intervention flags " + str(q))
    for j, row in enumerate(iv["rounds"]):
        require(row["round"] == j and row["support_count"] == cumulative[j] and row["support_ids_sha256"] == pv["future_cone_sha256"][j]
                and row["equals_future_cone"] is True and row["all_deltas_nonnegative"] is True
                and row["positive_integer_delta_sum"] == itv_sums[j] - fwd_sums[j]
                and row["positive_integer_delta_maximum"] <= row["positive_integer_delta_sum"]
                and ("equals_rer_support" in row) == (q == 13)
                and row.get("equals_rer_support", True) is True, "intervention row " + str((q, j)))
        if j >= 1:
            require(row["equals_source_net_reachable_ids"] is True, "intervention vs probes " + str((q, j)))
    frozen_rows = {r["layers"]: r for r in frozen["vertical_intervals"]}
    for row in level["manifold"]["intervals"]:
        k = row["layers"]
        fr = frozen_rows[k]
        kc = [sum(shells[:min(j, k - j) + 1]) for j in range(k + 1)]
        N = sum(kc)
        C = row["strict_pair_count"]
        require(row["counts_by_layer"] == kc == fr["counts_by_layer"] and row["inclusive_event_count"] == N == fr["inclusive_event_count"]
                and C == fr["strict_pair_count"] and row["equals_source_net"] is True, "manifold row " + str((q, k)))
        if N >= 2:
            f = Fraction(2 * C, N * (N - 1))
            dim = mm_dimension(float(f))
            require(row["ordering_fraction"] == str(f) == fr["ordering_fraction"]
                    and row["ordering_fraction_float"] == rounded(float(f))
                    and row["distance_to_one_tenth"] == rounded(float(f - Fraction(1, 10)))
                    and row["myrheim_meyer_dimension"] == (None if dim is None else rounded(dim)) == fr["myrheim_meyer_dimension"],
                    "manifold arithmetic " + str((q, k)))
    check_clock(level, frozen)
    require(level["manifold"]["all_intervals_equal_source_net"] is True, "manifold flag " + str(q))
    oc = level["operation_costs"]
    require(len(oc["per_round"]) == K + 1, "cost census " + str(q))
    check_cost_totals(oc, n, edges, frozen["sum_word_lengths"], frozen["maximum_word_length"])
    for j, row in enumerate(oc["per_round"]):
        require(row["round"] == j and row["write_bytes"] >= n * (ID_BYTES + VERSION_BYTES + 1)
                and (row["read_bytes"] == 0 if j == 0 else row["read_bytes"] >= reads_per_round * (2 * ID_BYTES + VERSION_BYTES + 1)),
                "cost row " + str((q, j)))
    require(level["stored_log"] is None, "no stored log above q = 8")
    return {"q": q, "rebuilt": False, "strict_pair_count": ci["strict_pair_count"]}


# --------------------------------------------------------------------------
# Receipt
# --------------------------------------------------------------------------


def check_carrier_block(block: dict, car: Carrier) -> None:
    require(block["ports"] == 12 and block["seams"] == 30 and block["faces"] == 20 and block["rotations"] == 60, "carrier census")
    require(block["antipode"] == list(car.antipode), "antipode")
    require(block["positive_port_basis"] == list(POSITIVE_PORTS), "positive port basis")
    require(block["axis_to_port"] == {str(k): {"positive": p, "negative": car.antipode[p]} for k, p in enumerate(POSITIVE_PORTS)},
            "axis to port")
    require(block["gram6_qsqrt5"] == [[r5_strings(v) for v in row] for row in car.gram6], "gram6")
    require(block["gram_sign_pattern"] == car.sign.tolist(), "sign pattern")
    require(block["gram_entries_by_port_distance"] == {"0": "1", "1": "1/sqrt5", "2": "-1/sqrt5", "3": "-1"}, "gram entries")
    require(block["generator_frame_qsqrt5"] == [[r5_strings(v) for v in row] for row in GENERATOR_FRAME], "generator frame")
    require(block["generator_frame_is_isometric_for_gram6"] is True and car.frame_is_isometric(), "frame isometry")
    require(all(v is True for v in block["carrier_self_checks"].values()), "carrier self checks")
    for key in ("load_placement_rule", "readback_law", "chart_role", "generator_frame_source"):
        require(type(block[key]) is str and len(block[key]) > 20, "carrier text " + key)


def verify(receipt: dict) -> dict:
    require(type(receipt) is dict, "receipt object")
    require(receipt.get("schema") == SCHEMA, "schema")
    require(receipt.get("scope") == EXPECTED_SCOPE, "scope block")
    require(type(receipt.get("claim_boundary")) is str and len(receipt["claim_boundary"]) > 200, "claim boundary")
    car = Carrier()
    check_carrier_block(receipt["carrier"], car)
    rm = receipt["readback_metric"]
    require(rm["scale_to_paper_position_s"] == "1" and rm["scale_to_source_net_units_L_squared_qsqrt5"] == ["2", "-2/5"]
            and rm["scale_to_source_net_units_L_squared_Qphi"] == ["12/5", "-4/5"], "readback scale")
    require(rm["discrepancy"].startswith("none"), "readback discrepancy")
    require(receipt["byte_model"]["site_id_bytes"] == ID_BYTES and receipt["byte_model"]["version_bytes"] == VERSION_BYTES, "byte model")
    for key in ("read_law", "round_rule", "edge_radius", "provenance_rule", "audit_chain", "dynamic_population_note"):
        require(type(receipt[key]) is str and len(receipt[key]) > 20, "text " + key)
    require("work in progress" in receipt["dynamic_population_note"], "dynamic population note")

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
        rer_levels = {lv["q"]: lv for lv in json.loads(rer_bytes)["levels"]}
    require(receipt["rer_cross_check"]["receipt_sha256"] == RER_RECEIPT_SHA256, "pinned theory receipt digest")
    require(receipt["rer_cross_check"]["levels_compared"] == [5, 8, 13] and receipt["rer_cross_check"]["all_agree"] is True,
            "theory cross-check summary")
    expected_pins = set(LOCAL_PINS) | {"reverse-engineering-reality/" + p for p in RER_PINS}
    expected_pins |= {f"data/exact/carrier_source_net_logs/q{q}_event_log.json.gz" for q in STORED_LOG_LEVELS}
    require(set(pins) == expected_pins, "pin census")
    for q in STORED_LOG_LEVELS:
        rel = f"data/exact/carrier_source_net_logs/q{q}_event_log.json.gz"
        require(pins[rel] == hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), "log pin " + rel)
    frozen = frozen_families()
    levels = receipt["levels"]
    require([lv["q"] for lv in levels] == list(LEVELS), "level census")
    rows = []
    for lv in levels:
        q = lv["q"]
        if q in REBUILT_LEVELS:
            rows.append(check_rebuilt_level(lv, car, frozen[q], None if rer_levels is None else rer_levels[q]))
        else:
            rows.append(check_large_level(lv, frozen[q]))
            if q == 13:
                block = lv["reads"]["rer_cross_check"]
                require(block is not None and block["all_agree"] is True, "theory block at q = 13")
                if rer_levels is not None:
                    fwd = rer_levels[13]["forward_execution"]
                    itv = rer_levels[13]["center_plus_one_intervention"]
                    require(lv["reads"]["forward"]["audit_trace_sha256"] == fwd["audit_trace_sha256"]
                            and lv["reads"]["forward"]["layer_value_sha256"] == fwd["layer_value_sha256"]
                            and lv["reads"]["forward"]["layer_value_sums"] == fwd["layer_value_sums"]
                            and lv["reads"]["intervention"]["audit_trace_sha256"] == itv["audit_trace_sha256"]
                            and lv["reads"]["intervention"]["layer_value_sha256"] == itv["layer_value_sha256"],
                            "theory digests at q = 13")
                    response = {r["layer"]: r for r in rer_levels[13]["intervention_response"]}
                    for row in lv["intervention"]["rounds"]:
                        require(row["support_ids_sha256"] == response[row["round"]]["support_ids_sha256"]
                                and row["equals_rer_support"] is True, "theory supports at q = 13")
    sn = receipt["source_net_cross_check"]
    require(sn["receipt"] == SOURCE_NET_RECEIPT and all(v is True for k, v in sn.items() if k != "receipt"), "source net summary")
    return {"accepted": True, "schema": SCHEMA, "rebuilt_levels": list(REBUILT_LEVELS),
            "theory_pins_checked": rer_present, "levels": rows,
            "receipt_sha256": hashlib.sha256(packed(receipt)).hexdigest()}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(verify(load(args.receipt)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
