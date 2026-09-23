"""Independent verifier for ``data/exact/federation_canonical_mean_receipt.json``.

This module does not import ``oph_exact.federation``.  It rebuilds the small
rungs from the committed carrier incidence and the production port routing,
uses its own mean law, its own Laplacian, its own component means, its own
canonical quotients and its own eigendecompositions, replays a declared
subset of the asynchronous schedules with a plain sequential loop, and
re-verifies the file pins.  Every failure raises ``VerificationError``.

Run: ``python -m oph_exact.verify_federation_independent [path]``.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as sparse_linalg

from oph_fpe.core.icosahedral import build_geodesic_icosahedral_tower, geodesic_icosahedral_patch_arrays
from oph_fpe.core.screen_ports import assign_echosahedral_ports
from oph_fpe.dynamics.self_readback_repair_closure import exact_reference_edges

SCHEMA = "oph.exact.federation-canonical-mean.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT = REPO_ROOT / "data" / "exact" / "federation_canonical_mean_receipt.json"
PORTS = 12
GLUINGS = ("isolated", "port_pair")
LOAD_SEED_BASE = 20260909
SCHEDULE_SEED_BASE = 909000
FLOAT_THRESHOLD = 1e-18
REPLAY_FLOAT = {"L0/isolated": 2, "L0/port_pair": 2, "L1/isolated": 1, "L1/port_pair": 1}
REPLAY_INTEGER = {"L0/isolated": 2, "L0/port_pair": 2, "L1/isolated": 1, "L1/port_pair": 1}
KERNEL_TOL = 1e-9


class VerificationError(AssertionError):
    """Raised when the receipt fails an independent check."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _close(a: float, b: float, rel: float = 1e-9, abs_tol: float = 0.0) -> bool:
    return abs(float(a) - float(b)) <= abs_tol + rel * max(abs(float(a)), abs(float(b)))


# --------------------------------------------------------------------------
# Independent federation
# --------------------------------------------------------------------------


class Federation:
    def __init__(self, level: int, gluing: str) -> None:
        points, left, right = geodesic_icosahedral_patch_arrays(level, patch_basis="cells")
        n = int(points.shape[0])
        _require(n == 20 * 4**level, "cell count")
        local = np.asarray(exact_reference_edges(), dtype=np.int64)
        _require(local.shape == (30, 2), "thirty carrier seams")
        base = (np.arange(n, dtype=np.int64) * PORTS)[:, None]
        a = (base + local[None, :, 0]).ravel()
        b = (base + local[None, :, 1]).ravel()
        self.level = level
        self.gluing = gluing
        self.carriers = n
        self.ports = PORTS * n
        self.intra = int(a.size)
        if gluing == "isolated":
            self.pairs = np.zeros((0, 4), dtype=np.int64)
        else:
            pm = assign_echosahedral_ports(left, right, n, points=points)
            _require(pm.overflow_count == 0, "production routing overflow")
            lp = pm.left_port.astype(np.int64)
            rp = pm.right_port.astype(np.int64)
            self.pairs = np.stack([left.astype(np.int64), lp, right.astype(np.int64), rp], axis=1)
            self.alignment = np.asarray(pm.directional_alignment, dtype=float)
            a = np.concatenate([a, left * PORTS + lp])
            b = np.concatenate([b, right * PORTS + rp])
        self.a = a.astype(np.int64)
        self.b = b.astype(np.int64)
        self.seams = int(self.a.size)
        adjacency = sparse.coo_matrix((np.ones(self.seams), (self.a, self.b)), shape=(self.ports, self.ports))
        adjacency = (adjacency + adjacency.T).tocsr()
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        self.laplacian = (sparse.diags(degree) - adjacency).tocsr()
        adjacency.sort_indices()
        self.neighbours = tuple(
            tuple(int(v) for v in adjacency.indices[adjacency.indptr[p] : adjacency.indptr[p + 1]]) for p in range(self.ports)
        )
        count, labels = csgraph.connected_components(adjacency, directed=False)
        first = np.full(count, self.ports, dtype=np.int64)
        np.minimum.at(first, labels, np.arange(self.ports, dtype=np.int64))
        relabel = np.empty(count, dtype=np.int64)
        relabel[np.argsort(first, kind="stable")] = np.arange(count)
        self.component = relabel[labels].astype(np.int64)
        self.sizes = np.bincount(self.component, minlength=count).astype(np.int64)
        self.components = int(count)
        faces = build_geodesic_icosahedral_tower(level).levels[level].faces
        self.pentagonal = np.any(np.asarray(faces) < 12, axis=1)
        self.denominator = Fraction(2 * self.seams, n)

    def loads(self) -> np.ndarray:
        return np.random.default_rng(LOAD_SEED_BASE + self.level).integers(0, 6, size=self.ports).astype(np.int64)

    def phi(self, x: np.ndarray) -> float:
        d = x[self.a] - x[self.b]
        return float(np.dot(d, d))

    def component_totals(self, loads: np.ndarray) -> np.ndarray:
        return np.bincount(self.component, weights=loads.astype(float), minlength=self.components).astype(np.int64)

    def expected_mean_hash(self, loads: np.ndarray) -> str:
        totals = self.component_totals(loads)
        return _sha(
            {
                "canonicalizer": "component_lattice_snap",
                "component_sizes": self.sizes.tolist(),
                "component_of_port": self.component.tolist(),
                "q": totals[self.component].tolist(),
            }
        )

    def snapped_hash(self, x: np.ndarray) -> tuple[str, float]:
        m = self.sizes[self.component].astype(float)
        scaled = m * x
        q = np.rint(scaled)
        return (
            _sha(
                {
                    "canonicalizer": "component_lattice_snap",
                    "component_sizes": self.sizes.tolist(),
                    "component_of_port": self.component.tolist(),
                    "q": q.astype(np.int64).tolist(),
                }
            ),
            float(np.max(np.abs(scaled - q))),
        )

    def expected_integer_hash(self, loads: np.ndarray) -> str:
        totals = self.component_totals(loads)
        entries = []
        for c in range(self.components):
            m = int(self.sizes[c])
            q, r = divmod(int(totals[c]), m)
            multiset = []
            if m - r:
                multiset.append([q, m - r])
            if r:
                multiset.append([q + 1, r])
            entries.append([m, multiset])
        return _sha({"canonicalizer": "component_multiset", "components": entries})

    def integer_hash(self, x: np.ndarray) -> str:
        entries = []
        for c in range(self.components):
            block = np.sort(x[self.component == c])
            values, counts = np.unique(block, return_counts=True)
            entries.append([int(block.size), [[int(v), int(k)] for v, k in zip(values, counts)]])
        return _sha({"canonicalizer": "component_multiset", "components": entries})

    def integer_v_min(self, loads: np.ndarray) -> int:
        totals = self.component_totals(loads)
        v = 0
        for c in range(self.components):
            m = int(self.sizes[c])
            q, r = divmod(int(totals[c]), m)
            v += (m - r) * q * q + r * (q + 1) * (q + 1)
        return v


def replay_float(fed: Federation, seed: int, max_sweeps: int = 100_000) -> dict[str, Any]:
    loads = fed.loads()
    x = loads.astype(float).tolist()
    al = fed.a.tolist()
    bl = fed.b.tolist()
    nbrs = fed.neighbours
    rng = np.random.default_rng(seed)
    seams = fed.seams
    waits = 0
    sweep = 0
    phi = fed.phi(np.asarray(x))
    v0 = float(np.dot(np.asarray(x), np.asarray(x)))
    ledger = 0.0
    phi_up = 0
    first_raise = None
    attempt = 0
    while phi >= FLOAT_THRESHOLD and sweep < max_sweeps:
        for s in rng.integers(0, seams, size=seams, dtype=np.int64).tolist():
            attempt += 1
            i = al[s]
            j = bl[s]
            xi = x[i]
            xj = x[j]
            d = xi - xj
            if d == 0.0:
                waits += 1
                continue
            ledger += 0.5 * (d * d)
            # Seam-sum change from the pre-move state: exact Phi difference of the move.
            phi_before = None
            if first_raise is None:
                ni = 0.0
                for k in nbrs[i]:
                    ni += x[k]
                nj = 0.0
                for k in nbrs[j]:
                    nj += x[k]
                degi = float(len(nbrs[i]))
                degj = float(len(nbrs[j]))
                delta = -d * ((degi * xi - ni) - (degj * xj - nj)) + (d * d) * ((degi + degj + 2.0) * 0.25)
                if delta > 0.0:
                    phi_before = fed.phi(np.asarray(x))
                    first_raise = {"attempt": attempt, "seam": s, "x_i": xi, "x_j": xj, "N_i": ni, "N_j": nj, "delta_phi": delta}
            else:
                ni = 0.0
                for k in nbrs[i]:
                    ni += x[k]
                nj = 0.0
                for k in nbrs[j]:
                    nj += x[k]
                degi = float(len(nbrs[i]))
                degj = float(len(nbrs[j]))
                delta = -d * ((degi * xi - ni) - (degj * xj - nj)) + (d * d) * ((degi + degj + 2.0) * 0.25)
            if delta > 0.0:
                phi_up += 1
            m = 0.5 * (xi + xj)
            x[i] = m
            x[j] = m
            if phi_before is not None:
                # The direct seam sum rises across the witnessed move.
                _require(fed.phi(np.asarray(x)) > phi_before, "phi counterexample: direct seam sum did not rise")
        sweep += 1
        phi = fed.phi(np.asarray(x))
    arr = np.asarray(x)
    means = fed.component_totals(loads).astype(float) / fed.sizes
    h, residual = fed.snapped_hash(arr)
    return {
        "sweeps": sweep,
        "attempts": sweep * seams,
        "waits": waits,
        "phi": phi,
        "terminated": phi < FLOAT_THRESHOLD,
        "hash": h,
        "deviation": float(np.max(np.abs(arr - means[fed.component]))),
        "ledger_ok": abs((v0 - float(np.dot(arr, arr))) - ledger) / max(v0, 1.0) < 1e-9,
        "phi_raising_moves": phi_up,
        "first_raise": first_raise,
    }


def replay_integer(fed: Federation, seed: int, max_sweeps: int = 100_000) -> dict[str, Any]:
    loads = fed.loads()
    x = loads.tolist()
    al = fed.a.tolist()
    bl = fed.b.tolist()
    rng = np.random.default_rng(seed)
    seams = fed.seams
    v = int(np.dot(loads, loads))
    v_min = fed.integer_v_min(loads)
    descents = swaps = waits = transfers = 0
    first = 0 if v == v_min else -1
    sweep = 0
    while first < 0 and sweep < max_sweeps:
        seq = rng.integers(0, seams, size=seams, dtype=np.int64).tolist()
        coin = rng.integers(0, 2, size=seams, dtype=np.int64).tolist()
        for t, (s, c) in enumerate(zip(seq, coin)):
            i = al[s]
            j = bl[s]
            xi = x[i]
            xj = x[j]
            tot = xi + xj
            lo = tot // 2
            hi = tot - lo
            ni = hi if c else lo
            nj = tot - ni
            d = xi - xj
            if d == 0:
                waits += 1
                continue
            if abs(d) == 1:
                if ni == xi:
                    waits += 1
                    continue
                swaps += 1
            else:
                descents += 1
                ad = abs(d)
                change = ni * ni + nj * nj - xi * xi - xj * xj
                # Composition of floor(|d|/2) unit transfers with mismatches |d|, |d|-2, ...: each lowers V by 2(d_k - 1).
                expected = -sum(2 * (ad - 2 * k - 1) for k in range(ad // 2))
                _require(change == expected and change < 0, "integer descent move: unit-transfer decrement identity")
                transfers += ad // 2
                v += change
            x[i] = ni
            x[j] = nj
            if first < 0 and v == v_min:
                first = sweep * seams + t + 1
        sweep += 1
    arr = np.asarray(x, dtype=np.int64)
    d = np.abs(arr[fed.a] - arr[fed.b])
    return {
        "sweeps": sweep,
        "attempts_to_balanced_class": first,
        "descents": descents,
        "swaps": swaps,
        "waits": waits,
        "odd_tie_seams": int(np.count_nonzero(d == 1)),
        "unit_transfers": transfers,
        "hash": fed.integer_hash(arr),
        "v_ok": v == int(np.dot(arr, arr)),
    }


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------


def carrier_laplacian() -> np.ndarray:
    lap = np.zeros((PORTS, PORTS))
    for i, j in exact_reference_edges():
        lap[i, j] -= 1.0
        lap[j, i] -= 1.0
        lap[i, i] += 1.0
        lap[j, j] += 1.0
    return lap


def isolated_kernel_from_eigendecomposition(n: int) -> np.ndarray:
    """``K_n`` of ``T = I - L/60`` with band ratios relative to the slowest nonconstant mode."""

    values, vectors = np.linalg.eigh(carrier_laplacian())
    damping = 1.0 - values / 60.0
    order = np.argsort(values)
    nonconstant = order[1:]
    slowest = damping[nonconstant[0]]
    kernel = np.zeros((PORTS, PORTS))
    trace = 0.0
    for k in nonconstant:
        share = (damping[k] / slowest) ** (2 * n)
        v = vectors[:, k]
        kernel += share * np.outer(v, v)
        trace += share
    return PORTS * kernel / trace


def isolated_kernel_from_power(n: int) -> np.ndarray:
    t = np.eye(PORTS) - carrier_laplacian() / 60.0
    q = np.eye(PORTS) - np.ones((PORTS, PORTS)) / PORTS
    c = q @ np.linalg.matrix_power(t, 2 * n) @ q
    return PORTS * c / np.trace(c)


def glued_kernels_dense(fed: Federation, cell: int, steps: list[int]) -> dict[int, np.ndarray]:
    """Dense ``T_fed^{2n}`` block readback for a small glued rung."""

    t = np.eye(fed.ports) - fed.laplacian.toarray() / float(fed.denominator)
    q = np.eye(PORTS) - np.ones((PORTS, PORTS)) / PORTS
    block = slice(cell * PORTS, (cell + 1) * PORTS)
    out = {}
    powers: dict[int, np.ndarray] = {}
    for n in steps:
        p = 2 * n
        # binary powering with a small cache
        result = np.eye(fed.ports)
        base = t.copy()
        e = p
        while e:
            if e & 1:
                result = result @ base
            e >>= 1
            if e:
                base = base @ base
        powers[n] = result
        r = result[block, block]
        c = q @ r @ q
        c = 0.5 * (c + c.T)
        out[n] = PORTS * c / np.trace(c)
    return out


def slow_band_projector_from_eigh() -> np.ndarray:
    values, vectors = np.linalg.eigh(carrier_laplacian())
    target = 5.0 - math.sqrt(5.0)
    columns = vectors[:, np.abs(values - target) < 1e-9]
    _require(columns.shape[1] == 3, "slow band has multiplicity three")
    return columns @ columns.T


# --------------------------------------------------------------------------
# Exact spot checks (fractions)
# --------------------------------------------------------------------------


def exact_spot_checks(fed: Federation, loads: np.ndarray) -> None:
    x = [Fraction(int(v)) for v in loads.tolist()]
    al = fed.a.tolist()
    bl = fed.b.tolist()
    comp = fed.component.tolist()
    sizes = fed.sizes.tolist()

    def project(y: list[Fraction]) -> list[Fraction]:
        totals = [Fraction(0)] * fed.components
        for p, c in enumerate(comp):
            totals[c] += y[p]
        return [totals[c] / sizes[c] for c in comp]

    pi = project(x)
    rng = np.random.default_rng(7)
    y = list(x)
    for _ in range(300):
        e = int(rng.integers(0, fed.seams))
        i, j = al[e], bl[e]
        d = y[i] - y[j]
        m = (y[i] + y[j]) / 2
        z = list(y)
        z[i] = m
        z[j] = m
        _require(sum(v * v for v in y) - sum(v * v for v in z) == d * d / 2, "exact descent identity")
        _require(sum(z) == sum(y), "exact conservation")
        y = z
    _require(project(y) == pi, "terminal quotient invariance along a walk")
    for e in range(0, fed.seams, max(1, fed.seams // 60)):
        z = list(x)
        i, j = al[e], bl[e]
        m = (z[i] + z[j]) / 2
        z[i] = m
        z[j] = m
        _require(project(z) == pi, "single move fixes the terminal quotient")


def phi_witness_check(witness: dict[str, Any]) -> None:
    seams = exact_reference_edges()
    x = [Fraction(v) for v in witness["state"]]
    a, b = witness["seam"]
    y = list(x)
    m = (y[a] + y[b]) / 2
    y[a] = m
    y[b] = m

    def phi(z: list[Fraction]) -> Fraction:
        return sum((z[i] - z[j]) ** 2 for i, j in seams)

    _require(str(phi(x)) == witness["phi_before"] and str(phi(y)) == witness["phi_after"], "phi witness values")
    _require(phi(y) > phi(x) and witness["phi_rises"] is True, "phi witness rises")
    v = lambda z: sum(t * t for t in z)
    _require(v(x) - v(y) == (x[a] - x[b]) ** 2 / 2, "witness descent identity")


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def verify(path: Path = DEFAULT_RECEIPT, *, replay: bool = True) -> dict[str, Any]:
    receipt = json.loads(Path(path).read_text(encoding="ascii"))
    _require(receipt.get("schema") == SCHEMA, "schema")
    text = Path(path).read_text(encoding="ascii")
    _require(text == _canonical(receipt), "receipt is canonical JSON")
    report: dict[str, Any] = {"rungs": {}}

    # Pins.
    pins = receipt["pins"]
    for name, digest in pins.items():
        file_path = REPO_ROOT / name
        _require(file_path.exists(), f"pinned file missing: {name}")
        _require(hashlib.sha256(file_path.read_bytes()).hexdigest() == digest, f"pin mismatch: {name}")
    report["pins_verified"] = len(pins)

    # Conventions and witness.
    phi_witness_check(receipt["conventions"]["phi_single_move_witness"])
    _require(all(receipt["conventions"]["carrier"]["self_checks"].values()), "carrier self checks")
    _require(receipt["scope"]["claimed"]["physical_position"] is False, "scope: physical position must not be claimed")
    _require(receipt["scope"]["claimed"]["cofinal_gluing"] is False, "scope: cofinal gluing must not be claimed")
    _require("work in progress" in " ".join(receipt["open_items"]), "open items carry the work-in-progress marker")
    note = receipt["production_law_note"]
    _require(note["report_path"].endswith("finite_consensus_replay_report.json"), "production law report path")

    p_slow = slow_band_projector_from_eigh()
    for key, block in receipt["rungs"].items():
        level = int(block["level"])
        gluing = block["gluing"]
        _require(key == f"L{level}/{gluing}", "rung key")
        fed = Federation(level, gluing)
        loads = fed.loads()
        f = block["federation"]
        _require(f["carriers"] == fed.carriers and f["ports"] == fed.ports and f["seams"] == fed.seams, f"{key}: counts")
        _require(f["intra_seams"] == fed.intra and f["inter_seams"] == fed.pairs.shape[0], f"{key}: seam split")
        _require(f["components"] == fed.components, f"{key}: components")
        _require(f["component_size_min"] == int(fed.sizes.min()) and f["component_size_max"] == int(fed.sizes.max()), f"{key}: sizes")
        _require(f["pentagonal_cells"] == int(fed.pentagonal.sum()), f"{key}: pentagonal cells")
        g = f["gluing_receipt"]
        if gluing == "port_pair":
            _require(g["port_pairs_sha256"] == _sha(fed.pairs.tolist()), f"{key}: port pair hash")
            if "port_pairs" in g:
                _require(g["port_pairs"] == fed.pairs.tolist(), f"{key}: explicit port pairs")
            usage = np.bincount(np.concatenate([fed.pairs[:, 1], fed.pairs[:, 3]]), minlength=PORTS).tolist()
            _require(g["port_usage_histogram"] == usage, f"{key}: port usage")
            _require(g["declared_convention"] is True and g["source_derived"] is False, f"{key}: gluing labelled declared")
            _require(_close(g["alignment_min"], float(fed.alignment.min()), 1e-6), f"{key}: alignment min")
        # loads
        _require(block["loads"]["sha256"] == _sha(loads.tolist()), f"{key}: loads hash")
        _require(block["loads"]["total"] == int(loads.sum()), f"{key}: load total")
        _require(block["loads"]["component_totals_sha256"] == _sha(fed.component_totals(loads).tolist()), f"{key}: component totals")
        # synchronous operator
        so = block["synchronous_operator"]
        _require(Fraction(so["D"]) == fed.denominator, f"{key}: D")
        _require(so["per_attempt_denominator"] == 2 * fed.seams, f"{key}: per-attempt denominator")
        if gluing == "isolated":
            _require(fed.denominator == 60 and so["isolated_block_equals_carrier_repair_mean_exact"] is True, f"{key}: block")
            lap_block = fed.laplacian[:PORTS, :PORTS].toarray()
            _require(np.array_equal(lap_block, carrier_laplacian()), f"{key}: isolated block Laplacian")
            eig = np.linalg.eigvalsh(carrier_laplacian())
            lam2, lam_max = float(eig[1]), float(eig[-1])
        elif fed.ports <= 4096:
            eig = np.linalg.eigvalsh(fed.laplacian.toarray())
            lam2, lam_max = float(eig[fed.components]), float(eig[-1])
        else:
            lam2 = float(np.sort(sparse_linalg.eigsh(fed.laplacian, k=fed.components + 1, sigma=-1e-3, which="LM", return_eigenvectors=False))[-1])
            lam_max = float(sparse_linalg.eigsh(fed.laplacian, k=1, which="LA", return_eigenvectors=False)[0])
        _require(_close(so["laplacian_lambda_2"], lam2, 1e-6, 1e-9), f"{key}: lambda_2 {so['laplacian_lambda_2']} vs {lam2}")
        _require(_close(so["laplacian_lambda_max"], lam_max, 1e-6, 1e-9), f"{key}: lambda_max")
        rung_report: dict[str, Any] = {"lambda_2": lam2}

        # Mean law, float.
        expected_mean = fed.expected_mean_hash(loads)
        mf = block["mean_law_float"]
        _require(mf["expected_terminal_hash"] == expected_mean, f"{key}: expected mean hash")
        _require(len(mf["entries"]) == mf["schedules"], f"{key}: schedule count")
        for idx, entry in enumerate(mf["entries"]):
            seed = SCHEDULE_SEED_BASE + 1000 * level + 100 * GLUINGS.index(gluing) + idx
            _require(entry["seed"] == seed, f"{key}: seed rule")
            _require(entry["terminated"] is True, f"{key}: float schedule {idx} did not terminate")
            _require(entry["terminal_quotient_hash"] == expected_mean, f"{key}: float schedule {idx} terminal hash")
            _require(entry["phi_terminal"] < FLOAT_THRESHOLD, f"{key}: terminal phi")
            _require(entry["attempts"] == entry["sweeps"] * fed.seams, f"{key}: attempts")
            _require(entry["strict_descent_violations"] == 0 and entry["descent_ledger_relative_error_below_1e-9"], f"{key}: descent")
            _require(entry["max_abs_deviation_from_component_mean"] < 1e-9, f"{key}: float deviation")
            _require(entry["lattice_residual_max"] < 0.5, f"{key}: lattice snap unambiguous")
            _require(entry["lattice_snap_unambiguous"] is True and entry["terminal_quotient_hash"] is not None, f"{key}: snap flag and hash present")
            _require(entry["centered_norm_terminal"] < 1e-12 and entry["centered_norm_initial"] > 0, f"{key}: centered norm")
            _require(_close(entry["descent_functional_minimum"], float(np.sum(fed.component_totals(loads).astype(float) ** 2 / fed.sizes)), 1e-9), f"{key}: V_min")
            _require(_close(entry["descent_functional_initial"], float(np.dot(loads, loads)), 1e-12), f"{key}: V_0")
            ce = entry["phi_raising_counterexample"]
            if entry["phi_raising_moves"] > 0:
                _require(ce is not None, f"{key}: counterexample present")
                i, j = ce["ports"]
                _require(fed.a[ce["seam"]] == i and fed.b[ce["seam"]] == j, f"{key}: counterexample seam")
                _require(ce["deg_i"] == len(fed.neighbours[i]) and ce["deg_j"] == len(fed.neighbours[j]), f"{key}: counterexample degrees")
                d = ce["x_i"] - ce["x_j"]
                delta = -d * ((ce["deg_i"] * ce["x_i"] - ce["neighbour_sum_i"]) - (ce["deg_j"] * ce["x_j"] - ce["neighbour_sum_j"])) + (d * d) * ((ce["deg_i"] + ce["deg_j"] + 2.0) * 0.25)
                _require(delta > 0 and _close(delta, ce["delta_phi"], 1e-6, 1e-9), f"{key}: counterexample delta formula")
        _require(mf["unique_terminal_hash_count"] == 1 and mf["terminal_hash_equals_expected"], f"{key}: unique hash")
        if replay and key in REPLAY_FLOAT:
            for idx in range(REPLAY_FLOAT[key]):
                entry = mf["entries"][idx]
                own = replay_float(fed, entry["seed"])
                _require(own["sweeps"] == entry["sweeps"] and own["waits"] == entry["waits"], f"{key}: replay {idx} sweeps/waits")
                _require(_close(own["phi"], entry["phi_terminal"], 1e-9), f"{key}: replay {idx} phi")
                _require(own["hash"] == entry["terminal_quotient_hash"], f"{key}: replay {idx} hash")
                _require(own["ledger_ok"], f"{key}: replay {idx} ledger")
                _require(own["phi_raising_moves"] == entry["phi_raising_moves"], f"{key}: replay {idx} phi-raising count")
                ce = entry["phi_raising_counterexample"]
                fr = own["first_raise"]
                if fr is not None or ce is not None:
                    _require(fr is not None and ce is not None, f"{key}: replay {idx} counterexample presence")
                    _require(fr["attempt"] == ce["attempt"] and fr["seam"] == ce["seam"], f"{key}: replay {idx} counterexample position")
                    _require(_close(fr["delta_phi"], ce["delta_phi"], 1e-9, 1e-12), f"{key}: replay {idx} counterexample delta")
            rung_report["float_replays"] = REPLAY_FLOAT[key]

        # Mean law, exact.
        if "mean_law_exact" in block:
            me = block["mean_law_exact"]
            _require(me["expected_terminal_hash"] == expected_mean, f"{key}: exact expected hash")
            for entry in me["entries"]:
                _require(entry["terminated"] and entry["phi_terminal_at_most_1e-18_exact"], f"{key}: exact termination")
                _require(entry["descent_identity_exact"] and entry["total_conservation_exact"] and entry["component_conservation_exact"], f"{key}: exact identities")
                _require(entry["terminal_quotient_hash"] == expected_mean, f"{key}: exact hash")
                _require(entry["lattice_snap_unambiguous"], f"{key}: exact snap")
                _require(entry["float_vs_exact_max_abs_deviation"] < 1e-12, f"{key}: float tracks exact")
                lo, hi = entry["float_over_exact_phi_ratio_range"]
                _require(0.99 <= lo <= hi <= 1.01, f"{key}: exact trigger margin")
                if entry["descent_identity_sample_step"] == 1:
                    _require(entry["global_descent_identity_exact"] is True, f"{key}: global exact ledger")
            _require(me["unique_terminal_hash_count"] == 1, f"{key}: exact unique hash")
            # The float and exact runs share seeds, so their sweep counts agree.
            float_by_seed = {e["seed"]: e for e in mf["entries"]}
            for entry in me["entries"]:
                _require(float_by_seed[entry["seed"]]["sweeps"] == entry["sweeps"], f"{key}: exact/float sweep agreement")
            exact_spot_checks(fed, loads)
            rung_report["exact_spot_checks"] = True

        # Integer law.
        il = block["integer_law"]
        expected_int = fed.expected_integer_hash(loads)
        _require(il["expected_terminal_hash"] == expected_int, f"{key}: expected integer hash")
        for entry in il["entries"]:
            _require(entry["terminated"] and entry["quotient_hash"] == expected_int, f"{key}: integer schedule hash")
            _require(entry["descent_ledger_exact"] and entry["strict_descent_violations"] == 0, f"{key}: integer descent")
            _require(entry["max_seam_difference_at_termination"] <= 1, f"{key}: balanced class")
            _require(entry["descent_functional_minimum"] == fed.integer_v_min(loads), f"{key}: V_min")
            _require(entry["descent_functional_terminal"] == entry["descent_functional_minimum"], f"{key}: terminal V is minimal")
            _require(entry["unit_transfer_decrement_identity_violations"] == 0, f"{key}: unit transfer identity")
            _require(entry["descent_functional_initial"] == int(np.dot(loads, loads)), f"{key}: integer V_0")
        _require(il["unique_terminal_hash_count"] == 1, f"{key}: integer unique hash")
        if replay and key in REPLAY_INTEGER:
            for idx in range(REPLAY_INTEGER[key]):
                entry = il["entries"][idx]
                own = replay_integer(fed, entry["seed"])
                for field in ("descents", "swaps", "waits", "attempts_to_balanced_class", "unit_transfers"):
                    _require(own[field] == entry[field], f"{key}: integer replay {idx} {field}")
                _require(own["odd_tie_seams"] == entry["odd_tie_seams_at_termination"], f"{key}: integer replay odd ties")
                _require(own["hash"] == entry["quotient_hash"] and own["v_ok"], f"{key}: integer replay hash")
            rung_report["integer_replays"] = REPLAY_INTEGER[key]

        # Local confluence block.
        if "local_confluence" in block:
            lc = block["local_confluence"]
            _require(lc["disjoint_pairs_commute_exactly"] == lc["disjoint_pairs_checked"], f"{key}: disjoint commutation")
            _require(lc["conflicting_pairs_joined_by_aggregate_component_mean"] == lc["conflicting_pairs_checked"], f"{key}: aggregate join")
            _require(lc["conflicting_pairs_same_terminal_quotient"] == lc["conflicting_pairs_checked"], f"{key}: terminal join")
            _require(lc["single_moves_fixing_terminal_quotient"] == fed.seams, f"{key}: all moves fix the quotient")
            _require(lc["descent_identity_exact"] == lc["descent_identity_moves_checked"], f"{key}: descent identity walk")

        # Response kernel.
        rk = block["response_kernel"]
        steps = rk["steps"]
        cells = rk["sampled_cells"]
        _require(len(cells) >= 8, f"{key}: at least eight sampled carriers")
        _require(rk["pentagonal_cells_in_sample"] == sum(bool(fed.pentagonal[c]) for c in cells) >= 1, f"{key}: pentagonal sample")
        if gluing == "isolated":
            _require(rk["isolated_kernel_matches_carrier_within_1e-12"] is True, f"{key}: isolated kernel flag")
            for n in steps:
                own = isolated_kernel_from_eigendecomposition(n)
                eig_own = np.sort(np.linalg.eigvalsh(own))[::-1]
                share_own = float(np.trace(p_slow @ own @ p_slow) / np.trace(own))
                if n <= 30:
                    power = isolated_kernel_from_power(n)
                    _require(np.max(np.abs(power - own)) < 1e-10, f"{key}: power form vs eigendecomposition at n={n}")
                for cell_entry in rk["per_cell"]:
                    row = next(r for r in cell_entry["per_step"] if r["n"] == n)
                    _require(np.max(np.abs(np.asarray(row["eigenvalues"]) - eig_own)) < KERNEL_TOL, f"{key}: isolated eigenvalues n={n}")
                    _require(abs(row["slow_band_share"] - share_own) < KERNEL_TOL, f"{key}: isolated share n={n}")
            limit = np.sort(np.linalg.eigvalsh(isolated_kernel_from_eigendecomposition(300)))[::-1]
            _require(np.allclose(limit[:3], 4.0, atol=1e-9) and np.allclose(limit[3:], 0.0, atol=1e-9), f"{key}: gram limit")
            _require(rk["max_abs_deviation_from_intrinsic_gram_at_n_300"] < 1e-9, f"{key}: gram deviation")
        elif fed.ports <= 1024:
            for cell_entry in rk["per_cell"][:4]:
                cell = int(cell_entry["cell"])
                own = glued_kernels_dense(fed, cell, steps)
                for row in cell_entry["per_step"]:
                    k = own[int(row["n"])]
                    eig_own = np.sort(np.linalg.eigvalsh(k))[::-1]
                    _require(np.max(np.abs(np.asarray(row["eigenvalues"]) - eig_own)) < 1e-7, f"{key}: glued eigenvalues cell {cell} n={row['n']}")
                    share_own = float(np.trace(p_slow @ k @ p_slow) / np.trace(k))
                    _require(abs(row["slow_band_share"] - share_own) < 1e-7, f"{key}: glued share cell {cell} n={row['n']}")
            rung_report["glued_kernel_dense_recomputed_cells"] = min(4, len(rk["per_cell"]))
        # Summary consistency.
        summary = rk["summary"]
        for n in steps:
            shares = [next(r for r in c["per_step"] if r["n"] == n)["slow_band_share"] for c in rk["per_cell"]]
            _require(_close(summary[str(n)]["slow_band_share"]["min"], min(shares), 1e-6, 1e-9), f"{key}: share summary min n={n}")
            _require(_close(summary[str(n)]["slow_band_share"]["max"], max(shares), 1e-6, 1e-9), f"{key}: share summary max n={n}")
        report["rungs"][key] = rung_report

    # Verdict flags follow from the rung data.
    verdicts = receipt["verdicts"]
    _require(all(verdicts.values()), f"verdict flags: {verdicts}")
    rows = {row["rung"]: row for row in receipt["summary"]}
    _require(set(rows) == set(receipt["rungs"]), "summary rows cover the rungs")
    for key, block in receipt["rungs"].items():
        _require(rows[key]["float_unique_terminal_hash_count"] == block["mean_law_float"]["unique_terminal_hash_count"], "summary float hash count")
        _require(rows[key]["integer_unique_quotient_hash_count"] == block["integer_law"]["unique_terminal_hash_count"], "summary integer hash count")
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
