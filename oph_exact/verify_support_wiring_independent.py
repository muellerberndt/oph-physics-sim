"""Independent verifier for ``data/exact/support_wiring_receipt.json``.

This module does not import ``oph_exact.support_wiring``.  It rebuilds the
twelve-neighbour cell graph from the tower faces, derives the antipode from
the carrier seams, routes the ports with the production assignment on its own
graph, counts antipodal consistency itself, replays one provenance schedule at
the smallest level in a plain Python loop with the declared rule and seed,
applies its own read-after-write rule to the stored log, recomputes the
interval statistics of the smallest level with Python-integer bitsets, checks
the exact single-carrier slow band and one W12 kernel with its own propagation,
recomputes the expected terminal hashes of the schedules from the loads, and
re-verifies the file pins.  Every failure raises ``VerificationError``.

Run: ``python -m oph_exact.verify_support_wiring_independent [path]``.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph

from oph_fpe.core.icosahedral import build_geodesic_icosahedral_tower
from oph_fpe.core.screen_ports import assign_echosahedral_ports
from oph_fpe.dynamics.self_readback_repair_closure import exact_reference_edges

SCHEMA = "oph.exact.support-wiring.v1"
LOG_SCHEMA = "oph.exact.support-wiring-log.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT = REPO_ROOT / "data" / "exact" / "support_wiring_receipt.json"
SOURCE_NET_RECEIPT = REPO_ROOT / "data" / "exact" / "source_net_causal_limit_receipt.json"
PORTS = 12
LOAD_SEED_BASE = 20260909
LOAD_MAX = 5
KERNEL_TOL = 1e-9
MM_2PLUS1 = 8.0 / 35.0
MM_3PLUS1 = 0.1


class VerificationError(AssertionError):
    """Raised when the receipt fails an independent check."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def _sha_arrays(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        arr = np.ascontiguousarray(arr)
        h.update(str(arr.dtype).encode("ascii"))
        h.update(str(arr.shape).encode("ascii"))
        h.update(arr.tobytes())
    return h.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _close(a: float, b: float, rel: float = 1e-9, abs_tol: float = 0.0) -> bool:
    return abs(float(a) - float(b)) <= abs_tol + rel * max(abs(float(a)), abs(float(b)))


def _rounded(value: float) -> float:
    out = float(f"{float(value):.12g}")
    return 0.0 if out == 0.0 else out


# --------------------------------------------------------------------------
# Own carrier data
# --------------------------------------------------------------------------


def carrier_seams() -> list[tuple[int, int]]:
    seams = [tuple(int(v) for v in e) for e in exact_reference_edges()]
    _require(len(seams) == 30, "thirty carrier seams")
    return seams


def carrier_antipode() -> list[int]:
    adj = {p: set() for p in range(PORTS)}
    for i, j in carrier_seams():
        adj[i].add(j)
        adj[j].add(i)
    anti = []
    for p in range(PORTS):
        dist = {p: 0}
        frontier = [p]
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            frontier = nxt
        far = [q for q in range(PORTS) if dist[q] == 3]
        _require(len(far) == 1, "distance-three partner unique")
        anti.append(far[0])
    _require(all(anti[anti[p]] == p and anti[p] != p for p in range(PORTS)), "antipode is a fixed-point-free involution")
    return anti


def carrier_laplacian() -> np.ndarray:
    lap = 5.0 * np.eye(PORTS)
    for i, j in carrier_seams():
        lap[i, j] -= 1.0
        lap[j, i] -= 1.0
    return lap


def slow_band_projector() -> np.ndarray:
    w, v = np.linalg.eigh(carrier_laplacian())
    target = 5.0 - math.sqrt(5.0)
    cols = [k for k in range(PORTS) if abs(w[k] - target) < 1e-9]
    _require(len(cols) == 3, "slow band has multiplicity three")
    u = v[:, cols]
    return u @ u.T


def single_carrier_kernel(n: int) -> np.ndarray:
    """``12 C_n / tr C_n`` from the band decomposition of ``T = I - L/60``."""

    w, v = np.linalg.eigh(carrier_laplacian())
    q = np.eye(PORTS) - np.ones((PORTS, PORTS)) / PORTS
    slow = 1.0 - (5.0 - math.sqrt(5.0)) / 60.0
    kernel = np.zeros((PORTS, PORTS))
    for k in range(PORTS):
        if w[k] < 1e-9:
            continue
        factor = ((1.0 - w[k] / 60.0) / slow) ** (2 * n)
        kernel += factor * np.outer(v[:, k], v[:, k])
    kernel = q @ kernel @ q
    return PORTS * kernel / np.trace(kernel)


def myrheim_meyer(d: float) -> float:
    return math.exp(math.lgamma(d + 1.0) + math.lgamma(d / 2.0) - math.log(2.0) - math.lgamma(1.5 * d))


def invert_myrheim_meyer(f: float) -> float | None:
    if not 0.0 < f < 1.0:
        return None
    lo, hi = 1.01, 20.0
    if not (myrheim_meyer(hi) <= f <= myrheim_meyer(lo)):
        return None
    for _ in range(96):
        mid = 0.5 * (lo + hi)
        if myrheim_meyer(mid) > f:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# Own wiring
# --------------------------------------------------------------------------


@lru_cache(maxsize=8)
def wiring(level: int) -> "Wiring":
    return Wiring(level)


class Wiring:
    def __init__(self, level: int) -> None:
        tower = build_geodesic_icosahedral_tower(level)
        mesh = tower.levels[level]
        faces = np.asarray(mesh.faces, dtype=np.int64)
        n = faces.shape[0]
        _require(n == 20 * 4**level, "cell count")
        points = np.sum(np.asarray(mesh.vertices)[faces], axis=1)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        vertex_faces: dict[int, list[int]] = {}
        for f, face in enumerate(faces.tolist()):
            for v in face:
                vertex_faces.setdefault(v, []).append(f)
        pairs: dict[tuple[int, int], int] = {}
        for members in vertex_faces.values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = sorted((members[i], members[j]))
                    pairs[(a, b)] = pairs.get((a, b), 0) + 1
        keys = sorted(pairs)
        self.level = level
        self.carriers = n
        self.points = points
        self.left = np.asarray([k[0] for k in keys], dtype=np.int64)
        self.right = np.asarray([k[1] for k in keys], dtype=np.int64)
        self.shared = np.asarray([pairs[k] for k in keys], dtype=np.int64)
        self.degree = np.bincount(np.concatenate([self.left, self.right]), minlength=n)
        self.pentagonal = np.any(faces < 12, axis=1)
        port_map = assign_echosahedral_ports(self.left, self.right, n, points=points)
        _require(port_map.routing_mode == "icosahedral_directional_assignment" and port_map.overflow_count == 0, "production routing accepted")
        self.left_port = port_map.left_port.astype(np.int64)
        self.right_port = port_map.right_port.astype(np.int64)
        slots = np.concatenate([self.left * PORTS + self.left_port, self.right * PORTS + self.right_port])
        _require(np.unique(slots).size == slots.size, "ports distinct per cell")
        used = np.zeros((n, PORTS), dtype=bool)
        used[self.left, self.left_port] = True
        used[self.right, self.right_port] = True
        self.unglued = np.full(n, -1, dtype=np.int64)
        for cell in np.flatnonzero(self.degree < PORTS):
            free = np.flatnonzero(~used[cell])
            _require(free.size == PORTS - self.degree[cell], "unglued count")
            self.unglued[cell] = int(free[0])
        self.alignment = np.asarray(port_map.directional_alignment, dtype=float)
        local = np.asarray(carrier_seams(), dtype=np.int64)
        base = (np.arange(n, dtype=np.int64) * PORTS)[:, None]
        intra_a = (base + local[None, :, 0]).ravel()
        intra_b = (base + local[None, :, 1]).ravel()
        self.intra = int(intra_a.size)
        self.seam_a = np.concatenate([intra_a, self.left * PORTS + self.left_port]).astype(np.int64)
        self.seam_b = np.concatenate([intra_b, self.right * PORTS + self.right_port]).astype(np.int64)
        self.ports = PORTS * n
        adjacency = sparse.coo_matrix((np.ones(self.seam_a.size), (self.seam_a, self.seam_b)), shape=(self.ports, self.ports))
        adjacency = (adjacency + adjacency.T).tocsr()
        self.laplacian = (sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency).tocsr()
        count, labels = csgraph.connected_components(adjacency, directed=False)
        first = np.full(count, self.ports, dtype=np.int64)
        np.minimum.at(first, labels, np.arange(self.ports, dtype=np.int64))
        relabel = np.empty(count, dtype=np.int64)
        relabel[np.argsort(first, kind="stable")] = np.arange(count)
        self.component = relabel[labels].astype(np.int64)
        self.sizes = np.bincount(self.component, minlength=count).astype(np.int64)
        self.components = int(count)

    def loads(self) -> np.ndarray:
        return np.random.default_rng(LOAD_SEED_BASE + self.level).integers(0, LOAD_MAX + 1, size=self.ports).astype(np.int64)

    def expected_mean_hash(self, loads: np.ndarray) -> str:
        totals = np.bincount(self.component, weights=loads.astype(float), minlength=self.components)
        q = np.rint(totals[self.component]).astype(np.int64)
        return _sha({"canonicalizer": "component_lattice_snap", "component_sizes": self.sizes.tolist(), "component_of_port": self.component.tolist(), "q": q.tolist()})

    def expected_integer_hash(self, loads: np.ndarray) -> str:
        totals = np.bincount(self.component, weights=loads.astype(float), minlength=self.components)
        entries = []
        for c in range(self.components):
            m = int(self.sizes[c])
            q, r = divmod(int(round(float(totals[c]))), m)
            multiset = []
            if m - r:
                multiset.append([q, m - r])
            if r:
                multiset.append([q + 1, r])
            entries.append([m, multiset])
        return _sha({"canonicalizer": "component_multiset", "components": entries})

    def integer_minimum(self, loads: np.ndarray) -> int:
        totals = np.bincount(self.component, weights=loads.astype(float), minlength=self.components)
        v = 0
        for c in range(self.components):
            m = int(self.sizes[c])
            q, r = divmod(int(round(float(totals[c]))), m)
            v += (m - r) * q * q + r * (q + 1) * (q + 1)
        return int(v)

    def kernel(self, cell: int, steps: list[int]) -> dict[int, np.ndarray]:
        D = 2.0 * self.seam_a.size / self.carriers
        y = np.zeros((self.ports, PORTS))
        q = np.eye(PORTS) - np.ones((PORTS, PORTS)) / PORTS
        y[cell * PORTS : (cell + 1) * PORTS, :] = q
        out = {}
        target = {2 * n: n for n in steps}
        for step in range(1, max(target) + 1):
            y = y - (self.laplacian @ y) / D
            y /= np.max(np.abs(y))
            if step in target:
                c = q @ y[cell * PORTS : (cell + 1) * PORTS, :]
                c = 0.5 * (c + c.T)
                out[target[step]] = PORTS * c / np.trace(c)
        return out


# --------------------------------------------------------------------------
# Own replay of the provenance schedule (plain Python)
# --------------------------------------------------------------------------


def replay_schedule(w: Wiring, rounds: int, seed: int) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(seed))
    x = w.loads().tolist()
    a = w.seam_a.tolist()
    b = w.seam_b.tolist()
    n_seams = len(a)
    intra = w.intra
    v = sum(t * t for t in x)
    v0 = v
    seams_out: list[int] = []
    outcomes: list[int] = []
    per_round = []
    violations = 0
    for _ in range(rounds):
        seq = rng.permutation(n_seams).tolist()
        coin = rng.integers(0, 2, size=n_seams).tolist()
        v_before = v
        waits = swaps = descents = 0
        for s, c in zip(seq, coin):
            i = a[s]
            j = b[s]
            xi = x[i]
            xj = x[j]
            tot = xi + xj
            lo = tot // 2
            hi = tot - lo
            ni = hi if c else lo
            nj = tot - ni
            d = xi - xj
            if d == 0:
                o = 0
            elif d == 1 or d == -1:
                o = 0 if ni == xi else 1
            else:
                o = 2
                v += ni * ni + nj * nj - xi * xi - xj * xj
            if o:
                x[i] = ni
                x[j] = nj
            if s >= intra:
                seams_out.append(s - intra)
                outcomes.append(o)
                if o == 0:
                    waits += 1
                elif o == 1:
                    swaps += 1
                else:
                    descents += 1
        if v > v_before:
            violations += 1
        per_round.append((waits, swaps, descents, v))
    return {
        "seam": np.asarray(seams_out, dtype=np.int32),
        "outcome": np.asarray(outcomes, dtype=np.int8),
        "v_initial": v0,
        "v_terminal": v,
        "v_state": sum(t * t for t in x),
        "violations": violations,
        "per_round": per_round,
        "state_sha256": _sha_arrays(np.asarray(x, dtype=np.int64)),
    }


def provenance_rule(cell_a: list[int], cell_b: list[int], writes: list[bool], carriers: int) -> dict[str, list[int]]:
    last = [-1] * carriers
    version = [0] * carriers
    out: dict[str, list[int]] = {k: [] for k in ("parent_a", "parent_b", "version_read_a", "version_read_b", "version_written_a", "version_written_b")}
    for t, (p, q, wr) in enumerate(zip(cell_a, cell_b, writes)):
        out["parent_a"].append(last[p])
        out["parent_b"].append(last[q])
        out["version_read_a"].append(version[p])
        out["version_read_b"].append(version[q])
        if wr:
            version[p] += 1
            version[q] += 1
            last[p] = t
            last[q] = t
        out["version_written_a"].append(version[p])
        out["version_written_b"].append(version[q])
    return out


def interval_statistics(pa: list[int], pb: list[int], bottom: int, top: int) -> dict[str, Any]:
    """Alexandrov interval statistics with Python-integer bitsets."""

    fwd = [False] * (top + 1)
    fwd[bottom] = True
    for t in range(bottom + 1, top + 1):
        fwd[t] = (pa[t] >= bottom and fwd[pa[t]]) or (pb[t] >= bottom and fwd[pb[t]])
    bwd = [False] * (top + 1)
    bwd[top] = True
    for t in range(top, bottom - 1, -1):
        if bwd[t]:
            if pa[t] >= bottom:
                bwd[pa[t]] = True
            if pb[t] >= bottom:
                bwd[pb[t]] = True
    ids = [t for t in range(bottom, top + 1) if fwd[t] and bwd[t]]
    local = {t: k for k, t in enumerate(ids)}
    n = len(ids)
    past = [0] * n
    depth = [0] * n
    comparable = 0
    for k, t in enumerate(ids):
        bits = 0
        d = 0
        for p in (pa[t], pb[t]):
            if p in local:
                bits |= past[local[p]]
                d = max(d, depth[local[p]])
        comparable += bits.bit_count()
        past[k] = bits | (1 << k)
        depth[k] = d + 1
    layers: dict[int, int] = {}
    for d in depth:
        layers[d] = layers.get(d, 0) + 1
    return {"events": n, "strict_pair_count": comparable, "height_events": max(depth) if depth else 0, "layer_width_max": max(layers.values()) if layers else 0, "ids": ids}


# --------------------------------------------------------------------------
# The verification
# --------------------------------------------------------------------------


def verify(path: Path = DEFAULT_RECEIPT, *, replay: bool = True) -> dict[str, Any]:
    receipt = json.loads(Path(path).read_text(encoding="ascii"))
    _require(receipt.get("schema") == SCHEMA, "schema")
    _require(Path(path).read_text(encoding="ascii") == _canonical(receipt), "receipt is canonical JSON")
    report: dict[str, Any] = {}

    pins = receipt["pins"]
    for name, digest in pins.items():
        file_path = REPO_ROOT / name
        _require(file_path.exists(), f"pinned file missing: {name}")
        _require(hashlib.sha256(file_path.read_bytes()).hexdigest() == digest, f"pin mismatch: {name}")
    report["pins_verified"] = len(pins)

    scope = receipt["scope"]
    _require(all(v is False for v in scope["not_claimed"].values()), "scope: not-claimed items are false")
    _require("work in progress" in " ".join(receipt["open_items"]), "open items carry the work-in-progress marker")
    _require(all(receipt["conventions"]["carrier"]["self_checks"].values()), "carrier self checks")
    anti = carrier_antipode()
    _require(receipt["conventions"]["carrier"]["antipode"] == anti, "antipode")

    # Wiring blocks.
    wirings: dict[int, Wiring] = {}
    for key, block in receipt["wiring"].items():
        level = int(key[1:])
        w = wiring(level)
        wirings[level] = w
        _require(block["carriers"] == w.carriers and block["glued_pairs"] == int(w.left.size), f"{key}: pair count")
        _require(block["edge_adjacent_pairs"] == int((w.shared == 2).sum()) and block["vertex_adjacent_pairs"] == int((w.shared == 1).sum()), f"{key}: adjacency split")
        hist = {str(k): int(v) for k, v in zip(*np.unique(w.degree, return_counts=True))}
        _require(block["neighbour_count_histogram"] == hist, f"{key}: neighbour histogram")
        _require(block["pentagonal_adjacent_cells"] == int(w.pentagonal.sum()) == 60, f"{key}: pentagonal-adjacent cells")
        _require(block["cells_with_eleven_neighbours_are_pentagonal_adjacent"] is True and np.array_equal(w.degree < PORTS, w.pentagonal), f"{key}: eleven-neighbour cells")
        _require(block["unglued_ports"] == int((w.unglued >= 0).sum()) == 60, f"{key}: unglued ports")
        _require(block["unglued_port_histogram"] == np.bincount(w.unglued[w.unglued >= 0], minlength=PORTS).tolist(), f"{key}: unglued histogram")
        _require(block["unglued_ports_by_cell_sha256"] == _sha([[int(c), int(p)] for c, p in enumerate(w.unglued.tolist()) if p >= 0]), f"{key}: unglued cells")
        _require(block["assignment_acceptance"]["accepted"] is True, f"{key}: acceptance")
        _require(block["port_pairs_sha256"] == _sha(np.stack([w.left, w.left_port, w.right, w.right_port], axis=1).tolist()), f"{key}: port pairs")
        usage = np.bincount(np.concatenate([w.left_port, w.right_port]), minlength=PORTS).tolist()
        _require(block["port_usage_histogram"] == usage, f"{key}: port usage")
        consistent = np.asarray(anti)[w.left_port] == w.right_port
        ac = block["antipodal_consistency"]
        _require(ac["consistent_pairs"] == int(consistent.sum()), f"{key}: antipodal count")
        _require(_close(ac["fraction"], consistent.mean(), 1e-9), f"{key}: antipodal fraction")
        _require(_close(ac["fraction_edge_adjacent"], consistent[w.shared == 2].mean(), 1e-9), f"{key}: antipodal fraction (edge)")
        _require(_close(ac["fraction_vertex_adjacent"], consistent[w.shared == 1].mean(), 1e-9), f"{key}: antipodal fraction (vertex)")
        _require(_close(block["alignment"]["min"], w.alignment.min(), 1e-6) and _close(block["alignment"]["mean"], w.alignment.mean(), 1e-6), f"{key}: alignment")
        angle = np.degrees(np.arccos(np.clip(np.sum(w.points[w.left] * w.points[w.right], axis=1), -1.0, 1.0)))
        _require(_close(block["neighbour_angle_deg"]["mean"], angle.mean(), 1e-6), f"{key}: neighbour angle")
        _require(0.0 < block["three_port_production_gluing"]["identical_port_fraction"] <= 1.0, f"{key}: three-port comparison")
    report["wiring_levels"] = sorted(wirings)

    # Dynamics blocks: expected hashes from the loads alone, verdict consistency.
    for key, block in receipt["dynamics"].items():
        level = int(key[1:])
        w = wirings[level]
        loads = w.loads()
        f = block["federation"]
        _require(f["carriers"] == w.carriers and f["ports"] == w.ports and f["seams"] == int(w.seam_a.size), f"{key}: federation counts")
        _require(f["intra_seams"] == w.intra and f["glued_seams"] == int(w.left.size) and f["components"] == w.components, f"{key}: seam split")
        _require(block["loads"]["sha256"] == _sha(loads.tolist()) and block["loads"]["total"] == int(loads.sum()), f"{key}: loads")
        mf = block["mean_law_float"]
        _require(mf["expected_terminal_hash"] == w.expected_mean_hash(loads), f"{key}: expected mean hash")
        _require(mf["strict_descent_violations_total"] == 0 and mf["descent_ledger_ok_all"], f"{key}: float descent")
        hashes = {e["terminal_quotient_hash"] for e in mf["entries"]}
        _require(mf["unique_terminal_hash_count"] == len(hashes), f"{key}: float hash count")
        if mf["all_terminated"]:
            _require(hashes == {mf["expected_terminal_hash"]}, f"{key}: float terminal hash equals component mean")
        il = block["integer_law"]
        _require(il["quotient_hash_equals_expected_multiset"] and {e["quotient_hash"] for e in il["entries"]} == {w.expected_integer_hash(loads)}, f"{key}: integer quotient")
        _require(all(e["descent_functional_minimum"] == w.integer_minimum(loads) for e in il["entries"]), f"{key}: integer minimum")
        _require(il["unit_transfer_decrement_identity_violations_total"] == 0 and il["descent_ledger_exact_all"], f"{key}: integer descent")
        so = block["synchronous_operator"]
        _require(so["laplacian_lambda_2"] > 0 and _close(so["laplacian_lambda_2_times_N"], so["laplacian_lambda_2"] * w.carriers, 1e-4), f"{key}: gap")

    # Slow band: exact single-carrier shares and one W12 kernel at the smallest level.
    p_slow = slow_band_projector()
    iso = receipt["slow_band"]["isolated_reference"]
    _require(iso["kernel_matches_carrier_within_1e-12"] and iso["gram_within_1e-12_at_n_300"], "isolated reference flags")
    for n_str, share in iso["exact_single_carrier_share"].items():
        k = single_carrier_kernel(int(n_str))
        own = float(np.trace(p_slow @ k @ p_slow) / np.trace(k))
        _require(_close(share, own, 1e-7), f"exact single-carrier share at n = {n_str}")
        _require(_close(iso["slow_band_share"][n_str], own, 1e-7), f"isolated share at n = {n_str}")
    gram = 4.0 * p_slow
    _require(np.max(np.abs(single_carrier_kernel(300) - gram)) < 1e-9, "single-carrier kernel limit is 4 P_slow")
    smallest = min(wirings)
    block = receipt["slow_band"][f"L{smallest}"]
    steps = [int(n) for n in block["steps"]]
    w = wirings[smallest]
    checked = 0
    for row in block["per_cell"][:2]:
        own = w.kernel(int(row["cell"]), steps)
        for entry in row["per_step"]:
            k = own[int(entry["n"])]
            share = float(np.trace(p_slow @ k @ p_slow) / np.trace(k))
            _require(_close(entry["slow_band_share"], share, 0.0, KERNEL_TOL), f"L{smallest}: W12 slow-band share of cell {row['cell']} at n = {entry['n']}")
            eig = np.sort(np.linalg.eigvalsh(k))[::-1][:4]
            _require(np.max(np.abs(np.asarray(entry["top_four_eigenvalues"]) - eig)) < 1e-6, f"L{smallest}: W12 kernel eigenvalues of cell {row['cell']}")
            checked += 1
    report["w12_kernels_checked"] = checked
    for key, blk in receipt["slow_band"].items():
        if not key.startswith("L"):
            continue
        for n_str, summary in blk["summary"].items():
            s = summary["slow_band_share"]
            _require(0.0 <= s["min"] <= s["median"] <= s["max"] <= 1.0 + 1e-12, f"{key}: share range at n = {n_str}")

    # Provenance: stored log at the smallest level, own replay, own rule, own intervals.
    for key, block in receipt["provenance"].items():
        level = int(key[1:])
        w = wirings[level]
        _require(block["external_seams"] == int(w.left.size) and block["carriers"] == w.carriers, f"{key}: system counts")
        _require(block["descent"]["ledger_exact"] and block["descent"]["round_violations"] == 0 and block["descent"]["terminal_equals_minimum"], f"{key}: descent")
        _require(block["descent"]["balanced_class_minimum"] == w.integer_minimum(w.loads()), f"{key}: balanced minimum")
        for variant in ("primary", "control_every_attempt_writes"):
            for tip in block[variant]["tips"]:
                for row in tip["ladder"]:
                    n = row["events"]
                    if n >= 2 and row["ordering_fraction"] is not None:
                        f = 2.0 * row["strict_pair_count"] / (n * (n - 1))
                        _require(_close(row["ordering_fraction"], f, 1e-9), f"{key}: ordering fraction arithmetic")
                        dim = invert_myrheim_meyer(f)
                        if dim is None:
                            _require(row["myrheim_meyer_dimension"] is None, f"{key}: dimension outside the inversion range")
                        else:
                            _require(_close(row["myrheim_meyer_dimension"], dim, 1e-7), f"{key}: dimension inversion")
                        _require(_close(row["deviation_from_2plus1"], f - MM_2PLUS1, 1e-7, 1e-12), f"{key}: deviation from 2+1")
        stored = block["stored_log"]
        if stored is None:
            continue
        log_path = REPO_ROOT / stored["path"]
        _require(log_path.exists(), f"{key}: stored log missing")
        with np.load(log_path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        meta = json.loads(str(arrays["meta"]))
        _require(meta["schema"] == LOG_SCHEMA and meta["seed"] == block["schedule"]["seed"] and meta["rounds"] == block["schedule"]["rounds_total"], f"{key}: log metadata")
        _require(np.array_equal(arrays["ext_left"], w.left.astype(np.int32)) and np.array_equal(arrays["ext_right"], w.right.astype(np.int32)), f"{key}: log seam endpoints equal the own graph")
        _require(_sha_arrays(arrays["seam"].astype(np.int32), arrays["outcome"].astype(np.int8)) == stored["content_sha256"] == block["log_content_sha256"], f"{key}: log content hash")
        _require(_sha_arrays(arrays["version_read_a"], arrays["version_read_b"], arrays["version_written_a"], arrays["version_written_b"]) == stored["versions_sha256"], f"{key}: log version hash")
        seam = arrays["seam"].astype(np.int64)
        outcome = arrays["outcome"].astype(np.int64)
        T = int(seam.size)
        _require(T == block["attempts_logged"] == block["schedule"]["rounds_total"] * int(w.left.size), f"{key}: attempt count")
        if replay:
            own = replay_schedule(w, block["schedule"]["rounds_total"], block["schedule"]["seed"])
            _require(np.array_equal(own["seam"].astype(np.int64), seam), f"{key}: replayed seam sequence")
            _require(np.array_equal(own["outcome"].astype(np.int64), outcome), f"{key}: replayed outcomes")
            _require(own["v_terminal"] == own["v_state"] == block["descent"]["terminal"] and own["v_initial"] == block["descent"]["initial"], f"{key}: replayed descent functional")
            _require(own["violations"] == 0 and own["state_sha256"] == block["terminal_state_sha256"], f"{key}: replayed terminal state")
            for r, (waits, swaps, descents, v) in enumerate(own["per_round"]):
                row = block["per_round"][r]
                _require((row["external_waits"], row["external_swaps"], row["external_descents"], row["descent_functional_after_round"]) == (waits, swaps, descents, v), f"{key}: per-round counts at round {r}")
            report[f"{key}_replayed_attempts"] = T
        cell_a = w.left[seam].tolist()
        cell_b = w.right[seam].tolist()
        rule = provenance_rule(cell_a, cell_b, (outcome > 0).tolist(), w.carriers)
        for name in ("version_read_a", "version_read_b", "version_written_a", "version_written_b"):
            _require(np.array_equal(np.asarray(rule[name], dtype=np.int64), arrays[name].astype(np.int64)), f"{key}: own provenance rule reproduces {name}")
        intervals = 0
        for variant, writes in (("primary", (outcome > 0).tolist()), ("control_every_attempt_writes", [True] * T)):
            own_rule = rule if variant == "primary" else provenance_rule(cell_a, cell_b, writes, w.carriers)
            pa, pb = own_rule["parent_a"], own_rule["parent_b"]
            for tip in block[variant]["tips"]:
                cell = tip["tip_cell"]
                touches = [t for t in range(T) if cell_a[t] == cell or cell_b[t] == cell]
                base = (block["schedule"]["burn_in_rounds"] + block["schedule"]["logged_rounds"] // 4) * int(w.left.size)
                bottom = next(t for t in touches if writes[t] and t >= base)
                _require(bottom == tip["bottom_attempt"], f"{key}/{variant}: bottom tip")
                for row in tip["ladder"]:
                    if row["events"] > 5000:
                        continue
                    own_stats = interval_statistics(pa, pb, bottom, row["top_attempt"])
                    _require(own_stats["events"] == row["events"], f"{key}/{variant}: interval size at separation {row['ladder_delta']}")
                    _require(own_stats["strict_pair_count"] == row["strict_pair_count"], f"{key}/{variant}: strict pair count at separation {row['ladder_delta']}")
                    _require(own_stats["height_events"] == row["height_events"] and own_stats["layer_width_max"] == row["layer_width_max"], f"{key}/{variant}: height and width")
                    intervals += 1
        report[f"{key}_intervals_recomputed"] = intervals

    # Source-net comparison against the source-net receipt.
    sn = receipt["source_net_comparison"]
    top = sn["vertical_intervals"][-1]
    if SOURCE_NET_RECEIPT.exists():
        stored = json.loads(SOURCE_NET_RECEIPT.read_text(encoding="ascii"))
        found = False
        for level in stored["levels"]:
            if level["q"] != sn["q"]:
                continue
            for family in level["families"]:
                if family["dimension"] != sn["dimension_of_population"]:
                    continue
                row = family["vertical_intervals"][-1]
                _require(row["inclusive_event_count"] == top["events"] and row["strict_pair_count"] == top["strict_pair_count"], "source-net top interval")
                _require(_close(row["ordering_fraction_float"], top["ordering_fraction"], 1e-9), "source-net ordering fraction")
                found = True
        _require(found and sn["reproduces_source_net_receipt_top_interval"] is True, "source-net receipt row")
    for row in sn["vertical_intervals"]:
        n = row["events"]
        f = 2.0 * row["strict_pair_count"] / (n * (n - 1))
        _require(_close(row["ordering_fraction"], f, 1e-9), "source-net ordering arithmetic")
        _require(row["height_events"] == row["layers"] + 1, "source-net height equals layers + 1")

    # Side-by-side rows and verdicts.
    _require(len(receipt["side_by_side"]) == 1 + len(receipt["provenance"]) + len(receipt["depth"]), "side-by-side row count")
    _require(_close(receipt["side_by_side"][0]["ordering_fraction"], top["ordering_fraction"], 1e-12), "side-by-side source-net row")
    for name, value in receipt["verdicts"].items():
        _require(value is True, f"verdict false: {name}")
    report["ok"] = True
    return report


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    path = Path(argv[0]) if argv else DEFAULT_RECEIPT
    report = verify(path)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
