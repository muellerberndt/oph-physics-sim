"""Independent verifier for ``oph_exact.federation_huge`` receipts.

It shares nothing with the engine but the pinned carrier seam template and the archive's
canonical-JSON convention.  From a run directory it checks:

* the cached geometry: cell count ``20 * 4^L``, every port glued at most once, the component
  count recomputed from the cell graph, the port-pairs digest recomputed from the inter table;
* the loads rule and, from the loads alone, the balanced-class minimum, the expected component
  multiset hash and the mean-law minimum;
* every integer schedule: the terminal state's digest, dtype and range, membership in the
  balanced class, ``V`` recomputed, conservation, the component multiset hash recomputed and
  equal to the expected hash, the ledger monotone and ending at the minimum, the draw digests
  re-derived from ``default_rng(seed)`` with single-call draws (every sweep unless limited), and
  an independent layered replay from the last retained checkpoint to the terminal state through
  every intermediate state digest;
* every mean schedule: the terminal digest, ``Phi``, ``V``, the deviation from the component
  mean, the lattice residual and the withheld-hash rule, conservation, and the draw digests;
* the response kernels: symmetry, trace twelve, the slow-band share recomputed from the
  matrix, and, for a sample of cells, the kernel at the two smallest step counts recomputed by
  a separate dense propagation on the local ball;
* the module pins against the repository files when they are present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph

from oph_exact import carrier

ROOT = Path(__file__).resolve().parents[1]
PORTS = 12
INTRA = 30
LOAD_MAX = 5
LOAD_SEED_BASE = 20260909
SCHEDULE_SEED_BASE = 909000
LATTICE_SNAP_MARGIN = 0.25
FLOAT_PHI_THRESHOLD = 1e-18
CHUNK = 1 << 24


class VerificationError(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def canonical(x: Any) -> bytes:
    return (json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha256_of(x: Any) -> str:
    return hashlib.sha256(canonical(x)).hexdigest()


def array_sha256(a: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=dtype)).astype(np.dtype(dtype).newbyteorder("<"), copy=False).tobytes(order="C")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sig(x: float, digits: int) -> float:
    return 0.0 if x == 0 else float(f"{float(x):.{digits}g}")


# --------------------------------------------------------------------------
# Geometry read independently
# --------------------------------------------------------------------------


class Geo:
    def __init__(self, cache_dir: Path, level: int) -> None:
        self.level = level
        self.dir = Path(cache_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.carriers = 20 * 4**level
        self.ports = PORTS * self.carriers
        self.inter = np.asarray(np.load(self.dir / "inter.npy"), dtype=np.int64)
        self.inter_count = int(self.inter.shape[0])
        self.intra_count = INTRA * self.carriers
        self.seams = self.intra_count + self.inter_count
        self.template = np.asarray(carrier.seams(), dtype=np.int64)
        self.ia = self.inter[:, 0] * PORTS + self.inter[:, 1]
        self.ib = self.inter[:, 2] * PORTS + self.inter[:, 3]
        adjacency = sparse.coo_matrix((np.ones(self.inter_count), (self.inter[:, 0], self.inter[:, 2])), shape=(self.carriers, self.carriers))
        self.components, label = csgraph.connected_components(adjacency, directed=False)
        first = np.unique(label, return_index=True)[1]
        relabel = np.empty(self.components, dtype=np.int64)
        relabel[np.argsort(first)] = np.arange(self.components)
        self.cell_component = relabel[label]
        self.port_component = np.repeat(self.cell_component, PORTS)

    def endpoints(self, seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        seq = np.asarray(seq, dtype=np.int64)
        a = np.empty(seq.size, dtype=np.int64)
        b = np.empty(seq.size, dtype=np.int64)
        intra = seq < self.intra_count
        c, t = np.divmod(seq[intra], INTRA)
        a[intra] = c * PORTS + self.template[t, 0]
        b[intra] = c * PORTS + self.template[t, 1]
        k = seq[~intra] - self.intra_count
        a[~intra] = self.ia[k]
        b[~intra] = self.ib[k]
        return a, b

    def phi(self, x: np.ndarray) -> float:
        xf = np.asarray(x, dtype=np.float64)
        blocks = xf.reshape(self.carriers, PORTS)
        total = 0.0
        for t in range(INTRA):
            d = blocks[:, self.template[t, 0]] - blocks[:, self.template[t, 1]]
            total += float(np.dot(d, d))
        for lo in range(0, self.inter_count, CHUNK):
            d = xf[self.ia[lo:lo + CHUNK]] - xf[self.ib[lo:lo + CHUNK]]
            total += float(np.dot(d, d))
        return total


def check_geometry(geo: Geo, receipt: dict[str, Any]) -> dict[str, Any]:
    g = receipt["geometry"]
    require(receipt["carriers"] == geo.carriers, "carrier count is not 20 * 4^L")
    require(receipt["ports"] == geo.ports and receipt["seams"] == geo.seams and receipt["inter_seams"] == geo.inter_count, "port or seam counts differ from the cached geometry")
    for name, entry in g["arrays"].items():
        require(file_sha256(geo.dir / f"{name}.npy") == entry["sha256"], f"geometry array {name} digest differs")
    slots = np.concatenate([geo.ia, geo.ib])
    require(np.unique(slots).size == slots.size, "a port is glued more than once")
    require(np.all(geo.inter[:, 1] < PORTS) and np.all(geo.inter[:, 3] < PORTS), "port index out of range")
    require(np.all(geo.inter[:, 0] != geo.inter[:, 2]), "an inter seam joins a carrier to itself")
    require(sha256_of(geo.inter.tolist()) == g["gluing"]["port_pairs_sha256"], "port pairs digest differs")
    require(receipt["components"] == geo.components, "component count differs")
    cached = np.asarray(np.load(geo.dir / "cell_component.npy"), dtype=np.int64)
    require(np.array_equal(cached, geo.cell_component), "cached component labels differ from the recomputed ones")
    require(g["gluing"]["glued_ports_per_carrier"] == 3 and geo.inter_count * 2 == 3 * geo.carriers, "every carrier must glue exactly three ports")
    return {"carriers": geo.carriers, "inter_seams": geo.inter_count, "components": geo.components}


# --------------------------------------------------------------------------
# Loads and expectations
# --------------------------------------------------------------------------


def loads_of(level: int, ports: int) -> np.ndarray:
    return np.random.default_rng(LOAD_SEED_BASE + level).integers(0, LOAD_MAX + 1, size=ports, dtype=np.int64)


def component_data(geo: Geo, loads: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = np.bincount(geo.port_component, minlength=geo.components).astype(np.int64)
    total = np.bincount(geo.port_component, weights=loads, minlength=geo.components).astype(np.int64)
    q, r = np.divmod(total, size)
    return size, total, q, r


def multiset_hash(size: np.ndarray, values_counts: list[list[list[int]]]) -> str:
    return sha256_of({"canonicalizer": "component_multiset", "components": [[int(s), vc] for s, vc in zip(size.tolist(), values_counts)]})


def expected_hash(size: np.ndarray, q: np.ndarray, r: np.ndarray) -> str:
    entries = []
    for s, qq, rr in zip(size.tolist(), q.tolist(), r.tolist()):
        vc = []
        if s - rr > 0:
            vc.append([int(qq), int(s - rr)])
        if rr > 0:
            vc.append([int(qq + 1), int(rr)])
        entries.append(vc)
    return multiset_hash(size, entries)


def state_multiset_hash(geo: Geo, x: np.ndarray) -> str:
    entries = []
    xi = np.asarray(x, dtype=np.int64)
    for c in range(geo.components):
        values, counts = np.unique(xi[geo.port_component == c], return_counts=True)
        entries.append([[int(v), int(k)] for v, k in zip(values, counts)])
    size = np.bincount(geo.port_component, minlength=geo.components)
    return multiset_hash(size, entries)


# --------------------------------------------------------------------------
# Draws and layered replay
# --------------------------------------------------------------------------


def draw(rng: np.random.Generator, seams: int, coins: bool) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    seq = rng.integers(0, seams, size=seams, dtype=np.int64)
    digests = [array_sha256(seq, "int64")]
    coin = None
    if coins:
        coin = rng.integers(0, 2, size=seams, dtype=np.int64)
        digests.append(array_sha256(coin, "int64"))
    return seq, coin, digests


def replay_integer_sweep(x: np.ndarray, geo: Geo, seq: np.ndarray, coin: np.ndarray, chunk: int = 65536) -> dict[str, int]:
    """Sequential semantics of one integer sweep in dependency layers of pairwise disjoint seams."""

    ports = x.size
    descents = swaps = waits = transfers = 0
    big = np.int64(1) << 62
    earliest = np.full(ports, big, dtype=np.int64)  # allocated once; touched entries are reset per layer
    for start in range(0, seq.size, chunk):
        ids = seq[start:start + chunk]
        ea, eb = geo.endpoints(ids)
        ec = coin[start:start + chunk]
        r = ids.size
        remaining = np.arange(r, dtype=np.int64)
        while remaining.size:
            ra = ea[remaining]
            rb = eb[remaining]
            m = remaining.size
            cat = np.empty(2 * m, dtype=np.int64)
            cat[0::2] = ra
            cat[1::2] = rb
            order = np.argsort(cat, kind="stable")
            sorted_ports = cat[order]
            starts = np.r_[True, sorted_ports[1:] != sorted_ports[:-1]]
            touched = sorted_ports[starts]
            earliest[touched] = (order // 2)[starts]
            pos = np.arange(m, dtype=np.int64)
            ready = (earliest[ra] == pos) & (earliest[rb] == pos)
            earliest[touched] = big
            la = ra[ready]
            lb = rb[ready]
            lc = ec[remaining[ready]]
            xa = x[la]
            xb = x[lb]
            tot = xa + xb
            lo = tot >> 1
            hi = tot - lo
            na = np.where(lc == 1, hi, lo)
            nb = tot - na
            d = xa - xb
            ad = np.abs(d)
            unit = ad == 1
            wait = (d == 0) | (unit & (na == xa))
            swap = unit & ~wait
            desc = ad >= 2
            waits += int(np.count_nonzero(wait))
            swaps += int(np.count_nonzero(swap))
            descents += int(np.count_nonzero(desc))
            transfers += int(np.sum((ad >> 1)[desc]))
            x[la] = na
            x[lb] = nb
            remaining = remaining[~ready]
    return {"descents": descents, "swaps": swaps, "waits": waits, "unit_transfers": transfers}


# --------------------------------------------------------------------------
# Schedules
# --------------------------------------------------------------------------


def check_integer_schedule(geo: Geo, loads: np.ndarray, size, q, r, v_min: int, expected: str, path: Path, entry: dict[str, Any],
                           *, draw_sweeps: int | None, replay_max_sweeps: int) -> dict[str, Any]:
    sched = json.loads((path / "schedule.json").read_text())
    for key in ("seed", "sweeps", "quotient_hash", "attempts_to_balanced_class", "descents", "swaps", "waits", "unit_transfers", "V_terminal"):
        require(sched[key] == entry[key], f"{path.name}: receipt entry and schedule.json disagree on {key}")
    seed = int(sched["seed"])
    require(sched["terminated"], f"{path.name}: schedule did not terminate")
    x = np.load(path / sched["terminal_state"]["path"])
    require(str(x.dtype) == "int8" and x.shape == (geo.ports,), f"{path.name}: terminal state dtype or shape")
    require(array_sha256(x, "int8") == sched["terminal_state"]["sha256"], f"{path.name}: terminal state digest")
    xi = x.astype(np.int64)
    require(int(xi.min()) >= 0 and int(xi.max()) <= LOAD_MAX + 1, f"{path.name}: terminal readings outside [0, 6]")
    v = int(np.dot(xi, xi))
    require(v == sched["V_terminal"] == v_min == sched["V_minimum"], f"{path.name}: V at termination is not the balanced minimum")
    require(int(xi.sum()) == int(loads.sum()), f"{path.name}: total not conserved")
    for c in range(geo.components):
        vals = xi[geo.port_component == c]
        require(np.all((vals == q[c]) | (vals == q[c] + 1)) and int(np.count_nonzero(vals == q[c] + 1)) == int(r[c]), f"{path.name}: component {c} not in the balanced class")
    require(state_multiset_hash(geo, xi) == sched["quotient_hash"] == expected, f"{path.name}: quotient hash differs from the recomputed or expected hash")
    ledger = sched["V_ledger"]
    require(ledger[0] == int(np.dot(loads, loads)) == sched["V_initial"], f"{path.name}: ledger start")
    require(all(a >= b for a, b in zip(ledger, ledger[1:])), f"{path.name}: ledger not monotone")
    require(ledger[-1] == v_min and len(ledger) == sched["sweeps"] + 1, f"{path.name}: ledger end or length")
    digests = sched["state_sha256_per_sweep"]
    sweeps = int(sched["sweeps"])
    require(len(digests) == sched["sweeps"] + 1 and digests[0] == array_sha256(loads.astype(np.int8), "int8") and digests[-1] == sched["terminal_state"]["sha256"], f"{path.name}: state digest chain endpoints")
    first = int(sched["attempts_to_balanced_class"])
    require((sweeps - 1) * geo.seams < first <= sweeps * geo.seams, f"{path.name}: attempts_to_balanced_class outside the last sweep")
    require(len(sched["draw_sha256_per_sweep"]) == sweeps, f"{path.name}: draw digest count")
    # checkpoints
    checkpoint = None  # the latest retained checkpoint strictly before the terminal sweep
    for c in sched.get("checkpoints", []):
        p = path / c["path"]
        if p.exists():
            xc = np.load(p)
            require(array_sha256(xc, "int8") == c["sha256"] == digests[c["sweep"]], f"{path.name}: checkpoint {c['sweep']} digest")
            if c["sweep"] < sweeps and (checkpoint is None or c["sweep"] > checkpoint[0]):
                checkpoint = (int(c["sweep"]), xc.astype(np.int64))
    # draws (sequential stream) and replay after the checkpoint
    limit = sweeps if draw_sweeps is None else min(draw_sweeps, sweeps)
    if sweeps <= replay_max_sweeps:
        replay_from = 0
    elif checkpoint is not None and sweeps - checkpoint[0] <= replay_max_sweeps:
        replay_from = checkpoint[0]
    else:
        replay_from = None
    if replay_from is not None and limit < sweeps:
        limit = sweeps  # the replay needs the whole stream
    rng = np.random.default_rng(seed)
    state = loads.astype(np.int64).copy() if replay_from == 0 else (checkpoint[1].copy() if replay_from is not None else None)
    counts = {"descents": 0, "swaps": 0, "waits": 0, "unit_transfers": 0}
    replayed = 0
    for s in range(limit):
        seq, coin, d = draw(rng, geo.seams, coins=True)
        require(d == sched["draw_sha256_per_sweep"][s], f"{path.name}: draw digests of sweep {s + 1} differ")
        if replay_from is not None and s >= replay_from:
            c = replay_integer_sweep(state, geo, seq, coin)
            require(array_sha256(state.astype(np.int8), "int8") == digests[s + 1], f"{path.name}: replayed state after sweep {s + 1} differs")
            for k in counts:
                counts[k] += c[k]
            replayed += 1
    if replay_from == 0:
        for k in counts:
            require(counts[k] == sched[k], f"{path.name}: replayed {k} differ")
    return {"seed": seed, "sweeps": sweeps, "draw_sweeps_checked": limit, "replayed_sweeps": replayed, "replay_from": replay_from}


def check_mean_schedule(geo: Geo, loads: np.ndarray, size, total, path: Path, entry: dict[str, Any], *, draw_sweeps: int | None) -> dict[str, Any]:
    sched = json.loads((path / "schedule.json").read_text())
    for key in ("seed", "sweeps", "phi_terminal", "V_terminal", "lattice_snap_unambiguous", "terminal_quotient_hash"):
        require(sched[key] == entry[key], f"{path.name}: receipt entry and schedule.json disagree on {key}")
    x = np.load(path / sched["terminal_state"]["path"])
    require(str(x.dtype) == "float64" and x.shape == (geo.ports,), f"{path.name}: terminal dtype or shape")
    require(array_sha256(x, "float64") == sched["terminal_state"]["sha256"], f"{path.name}: terminal digest")
    phi = geo.phi(x)
    require(_sig(phi, 12) == sched["phi_terminal"] or abs(phi - sched["phi_terminal"]) <= 1e-9 * max(phi, 1.0), f"{path.name}: Phi recomputed differs")
    v = float(np.dot(x, x))
    require(abs(v - sched["V_terminal"]) <= 1e-9 * max(v, 1.0), f"{path.name}: V recomputed differs")
    require(abs(float(x.sum()) - float(loads.sum())) <= 1e-6 * max(float(loads.sum()), 1.0), f"{path.name}: total drifted")
    mean = (total.astype(np.float64) / size.astype(np.float64))[geo.port_component]
    dev = float(np.max(np.abs(x - mean)))
    require(abs(dev - sched["max_abs_deviation_from_component_mean"]) <= 1e-5 * max(dev, 1e-12) + 1e-12, f"{path.name}: deviation differs")
    scaled = size.astype(np.float64)[geo.port_component] * x
    residual = float(np.max(np.abs(scaled - np.rint(scaled))))
    snap = residual < LATTICE_SNAP_MARGIN
    require(snap == sched["lattice_snap_unambiguous"], f"{path.name}: lattice snap flag differs")
    require((sched["terminal_quotient_hash"] is not None) == snap, f"{path.name}: terminal hash must be present exactly when the snap is unambiguous")
    require(sched["terminated"] == (phi < FLOAT_PHI_THRESHOLD), f"{path.name}: termination flag differs from Phi")
    require(sched["descent_ledger_relative_error_below_1e-9"] == (sched["descent_ledger_relative_error"] < 1e-9), f"{path.name}: ledger flag")
    require(sched["strict_descent_violations"] == 0, f"{path.name}: strict descent violations recorded")
    limit = sched["sweeps"] if draw_sweeps is None else min(draw_sweeps, sched["sweeps"])
    rng = np.random.default_rng(int(sched["seed"]))
    for s in range(limit):
        _seq, _c, d = draw(rng, geo.seams, coins=False)
        require(d == sched["draw_sha256_per_sweep"][s], f"{path.name}: draw digest of sweep {s + 1} differs")
    return {"seed": sched["seed"], "sweeps": sched["sweeps"], "draw_sweeps_checked": limit, "phi_terminal": sched["phi_terminal"], "snap": snap}


# --------------------------------------------------------------------------
# Kernels, recomputed on a dense local ball
# --------------------------------------------------------------------------


def dense_local_kernel(geo: Geo, cell: int, steps: Sequence[int]) -> dict[int, np.ndarray]:
    last = 2 * max(steps)
    radius = last + 1
    cells = {cell}
    frontier = {cell}
    neigh: dict[int, set[int]] = {}
    left, right = geo.inter[:, 0], geo.inter[:, 2]
    for _ in range(radius):
        mask = np.isin(left, list(frontier)) | np.isin(right, list(frontier))
        nxt = set(left[mask].tolist()) | set(right[mask].tolist())
        nxt -= cells
        if not nxt:
            break
        cells |= nxt
        frontier = nxt
    cells_sorted = sorted(cells)
    index = {c: k for k, c in enumerate(cells_sorted)}
    n = len(cells_sorted) * PORTS
    L = np.zeros((n, n))
    for k in range(len(cells_sorted)):
        for t in range(INTRA):
            a = k * PORTS + int(geo.template[t, 0])
            b = k * PORTS + int(geo.template[t, 1])
            L[a, a] += 1; L[b, b] += 1; L[a, b] -= 1; L[b, a] -= 1
    mask = np.isin(left, cells_sorted) & np.isin(right, cells_sorted)
    for row in geo.inter[mask]:
        a = index[int(row[0])] * PORTS + int(row[1])
        b = index[int(row[2])] * PORTS + int(row[3])
        L[a, a] += 1; L[b, b] += 1; L[a, b] -= 1; L[b, a] -= 1
    D = 2 * geo.seams / geo.carriers
    T = np.eye(n) - L / D
    q = np.eye(PORTS) - np.ones((PORTS, PORTS)) / PORTS
    y = np.zeros((n, PORTS))
    block = slice(index[cell] * PORTS, (index[cell] + 1) * PORTS)
    y[block, :] = q
    out = {}
    target = {2 * s: s for s in steps}
    for step in range(1, last + 1):
        y = T @ y
        scale = np.max(np.abs(y))
        if scale > 0:
            y /= scale
        if step in target:
            k = q @ y[block, :]
            k = 0.5 * (k + k.T)
            out[target[step]] = PORTS * k / np.trace(k)
    return out


def check_kernels(geo: Geo, receipt: dict[str, Any], sample: int) -> dict[str, Any]:
    rk = receipt["response_kernels"]
    steps = [int(s) for s in rk["steps"]]
    p_slow = carrier.slow_band_projector()
    worst_share = 0.0
    for entry in rk["cells"]:
        for s in [int(k) for k in entry["kernels"]]:
            k = np.asarray(entry["kernels"][str(s)])
            require(k.shape == (PORTS, PORTS) and np.allclose(k, k.T, atol=1e-12), f"kernel of cell {entry['cell']} at n={s} not symmetric")
            require(abs(np.trace(k) - PORTS) < 1e-9, f"kernel of cell {entry['cell']} at n={s} trace differs from 12")
            share = float(np.trace(p_slow @ k @ p_slow) / np.trace(k))
            worst_share = max(worst_share, abs(share - entry["slow_band_share"][str(s)]))
    require(worst_share < 1e-9, "slow-band share recomputed from a kernel differs")
    small = sorted(steps)[:2]
    worst = 0.0
    checked = 0
    for entry in rk["cells"][:sample]:
        mine = dense_local_kernel(geo, int(entry["cell"]), small)
        for s in small:
            worst = max(worst, float(np.max(np.abs(mine[s] - np.asarray(entry["kernels"][str(s)])))))
        checked += 1
    require(worst < 1e-10, f"recomputed kernels differ by {worst:.3e}")
    for s, summary in rk["glued_slow_band_share"].items():
        shares = [e["slow_band_share"][s] for e in rk["cells"]]
        require(abs(summary["min"] - _sig(min(shares), 6)) < 1e-9 and abs(summary["max"] - _sig(max(shares), 6)) < 1e-9, f"slow-band summary at n={s}")
    for s, summary in rk.get("glued_slow_band_share_long_steps", {}).items():
        shares = [e["slow_band_share"][s] for e in rk["cells"] if s in e["slow_band_share"]]
        require(len(shares) == summary["cells"] and abs(summary["min"] - _sig(min(shares), 6)) < 1e-9 and abs(summary["max"] - _sig(max(shares), 6)) < 1e-9, f"long-step slow-band summary at n={s}")
    return {"cells": len(rk["cells"]), "recomputed_cells": checked, "max_abs_kernel_difference": worst, "steps_recomputed": small}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def verify(run_dir: Path, *, cache_dir: Path | None = None, draw_sweeps: int | None = None, replay_max_sweeps: int = 8,
           kernel_sample: int = 4, schedules: Sequence[int] | None = None, repo_root: Path | None = ROOT, log=print) -> dict[str, Any]:
    run_dir = Path(run_dir)
    receipt = json.loads((run_dir / "receipt.json").read_text())
    require(receipt["schema"] == "oph.exact.federation-huge.v1", "unexpected schema")
    level = int(receipt["level"])
    cache = Path(cache_dir) if cache_dir is not None else Path(receipt["geometry"]["cache_dir"])
    geo = Geo(cache, level)
    result: dict[str, Any] = {"level": level, "run_dir": str(run_dir)}
    result["geometry"] = check_geometry(geo, receipt)
    log(f"geometry ok: {geo.carriers} carriers, {geo.inter_count} inter seams, {geo.components} component(s)")
    loads = loads_of(level, geo.ports)
    require(receipt["seeds"]["loads"]["load_max"] == LOAD_MAX, "load range")
    size, total, q, r = component_data(geo, loads)
    v_min = int(np.sum((size - r) * q * q + r * (q + 1) ** 2))
    exp = expected_hash(size, q, r)
    require(receipt["expected"]["expected_integer_quotient_hash"] == exp, "expected hash differs from the loads-derived hash")
    require(receipt["expected"]["balanced_minimum"] == v_min, "balanced minimum differs")
    require(receipt["expected"]["component_count"] == geo.components, "expected component count")
    mean_min = float(sum(Fraction(int(t) ** 2, int(m)) for t, m in zip(total.tolist(), size.tolist())))
    require(abs(receipt["expected"]["mean_minimum"] - mean_min) <= 1e-9 * max(mean_min, 1.0), "mean minimum differs")
    result["expected"] = {"hash": exp, "balanced_minimum": v_min}
    # integer schedules
    entries = receipt["integer_law"]["entries"]
    require(len(entries) == receipt["integer_law"]["schedules"], "integer schedule count")
    seeds_declared = [SCHEDULE_SEED_BASE + 1000 * level + 100 + k for k in range(len(entries))]
    require([e["seed"] for e in entries] == seeds_declared == receipt["seeds"]["schedules"]["integer"], "integer schedule seeds do not follow the rule")
    require(receipt["integer_law"]["all_terminated"] and receipt["integer_law"]["quotient_hash_equals_expected_all"], "integer summary flags")
    require(receipt["integer_law"]["unique_quotient_hash_count"] == len({e["quotient_hash"] for e in entries}) == 1, "unique hash count")
    require(receipt["integer_law"]["violations_total"] == 0 and receipt["integer_law"]["conservation_exact_all"], "integer violations or conservation")
    chosen = range(len(entries)) if schedules is None else [k for k in schedules if k < len(entries)]
    result["integer"] = []
    for k in chosen:
        e = entries[k]
        res = check_integer_schedule(geo, loads, size, q, r, v_min, exp, run_dir / f"integer_{e['seed']}", e, draw_sweeps=draw_sweeps, replay_max_sweeps=replay_max_sweeps)
        log(f"integer seed {e['seed']}: ok, {res['sweeps']} sweeps, draws checked {res['draw_sweeps_checked']}, replayed {res['replayed_sweeps']} sweep(s) from {res['replay_from']}")
        result["integer"].append(res)
    # mean schedules
    result["mean"] = []
    for e in receipt["mean_law_float"]["entries"]:
        res = check_mean_schedule(geo, loads, size, total, run_dir / f"mean_{e['seed']}", e, draw_sweeps=draw_sweeps)
        log(f"mean seed {e['seed']}: ok, {res['sweeps']} sweeps, Phi {res['phi_terminal']:.6g}, snap {res['snap']}")
        result["mean"].append(res)
    m = receipt["mean_law_float"]
    require(m["ambiguous_terminal_count"] == sum(1 for e in m["entries"] if e["terminal_quotient_hash"] is None), "ambiguous terminal count")
    # kernels
    result["kernels"] = check_kernels(geo, receipt, kernel_sample)
    log(f"kernels ok: {result['kernels']['cells']} cells, {result['kernels']['recomputed_cells']} recomputed at n={result['kernels']['steps_recomputed']}, max diff {result['kernels']['max_abs_kernel_difference']:.2e}")
    # pins
    pins = {}
    if repo_root is not None:
        for rel, digest in receipt["module_pins"].items():
            p = Path(repo_root) / rel
            if p.is_file():
                pins[rel] = file_sha256(p) == digest
        require(all(pins.values()), f"module pins differ: {[k for k, v in pins.items() if not v]}")
    result["module_pins_checked"] = pins
    result["verdict"] = "PASS"
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--draw-sweeps", type=int, default=None, help="check only this many leading sweeps' draw digests (default all)")
    parser.add_argument("--replay-max-sweeps", type=int, default=8)
    parser.add_argument("--kernel-sample", type=int, default=4)
    parser.add_argument("--schedules", type=int, nargs="*", default=None)
    parser.add_argument("--no-pins", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        result = verify(args.run_dir, cache_dir=args.cache, draw_sweeps=args.draw_sweeps, replay_max_sweeps=args.replay_max_sweeps,
                        kernel_sample=args.kernel_sample, schedules=args.schedules, repo_root=None if args.no_pins else ROOT)
    except VerificationError as error:
        print(f"FAIL: {error}")
        return 1
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
