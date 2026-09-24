"""Lean settlement engine for large glued federations (tower levels seven and beyond).

Every declared convention is the one of ``oph_exact.federation`` and
``oph_exact.federation_archive`` and is unchanged:

* carriers are the ``20 * 4^L`` cells of the geodesic icosahedral tower, twelve ports each;
  the thirty intra-carrier seams of ``carrier.seams()`` come first, carrier-major, then the
  inter-carrier seams of the production gluing (``assign_echosahedral_ports`` on the cell-dual
  graph) in dual-edge order; seam ``s`` therefore addresses ports by arithmetic when
  ``s < 30 N`` and by the inter-seam table otherwise;
* loads ``default_rng(20260909 + level).integers(0, 6)``; schedule seeds
  ``909000 + 1000 level + 100 gluing_index + k`` with the production gluing at index one;
* one sweep is ``|S|`` seam indices drawn i.i.d. with replacement from PCG64
  (``integers(0, |S|, size=|S|, dtype=int64)``), followed for the integer law by
  ``integers(0, 2, size=|S|, dtype=int64)`` tie coins, applied in draw order;
* the mean law replaces both endpoints of a seam by their mean; the integer law is the nearest
  agreement with the ceiling on the first endpoint when the coin is one; ``V = sum x^2``;
  the integer law terminates at the exact attempt at which ``V`` reaches the balanced-class
  minimum, and the sweep in progress is completed; the mean law runs under a sweep budget;
* the terminal integer hash is the component multiset; the mean law's lattice snap is a
  certificate only when unambiguous and is withheld otherwise.

What differs from the reference lane is representation, chosen so that memory stays linear in
the number of carriers: an ``int8`` state for the integer law and ``float64`` for the mean law,
no neighbour lists or Laplacian, the inter-seam table as two ``int32`` arrays shared through
memory-mapped files, draws taken in contiguous chunks that reproduce the single-call stream bit
for bit and are digested incrementally into the archive's per-sweep draw digests, a state digest
after every sweep as a checkpoint, and one process per schedule.  The per-carrier response
kernel is propagated on the ball of graph radius ``2n + 1`` around the probed carrier, which is
exact because the operator moves one hop per step.

Receipts are written as JSON beside ``.npy`` arrays under an output directory; the
independent verifier ``oph_exact/verify_federation_huge_independent.py`` reads them.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph

from oph_exact import carrier
from oph_exact import federation as F
from oph_exact.federation_archive import (
    array_sha256,
    balanced_minimum,
    component_expectation,
    expected_multiset_hash,
    kernel_sample_cells,
    vertex_cells,
)
from oph_fpe.core.icosahedral import geodesic_icosahedral_patch_arrays
from oph_fpe.core.screen_ports import assign_echosahedral_ports

ROOT = Path(__file__).resolve().parents[1]
PORTS = 12
INTRA = carrier.SEAM_COUNT  # 30
SCHEMA = "oph.exact.federation-huge.v1"
LOAD_SEED_BASE = F.LOAD_SEED_BASE
LOAD_MAX = F.LOAD_MAX
SCHEDULE_SEED_BASE = F.SCHEDULE_SEED_BASE
GLUING_INDEX = 1  # port_pair
FLOAT_PHI_THRESHOLD = F.FLOAT_PHI_THRESHOLD
LATTICE_SNAP_MARGIN = F.LATTICE_SNAP_MARGIN
DRAW_CHUNK = 1 << 24  # draws per chunk (128 MB of int64)
DEFAULT_SCHEDULES = 16
KERNEL_STEPS = (1, 5, 30, 100)
PINNED_MODULES = ("oph_fpe/core/icosahedral.py", "oph_fpe/core/screen_ports.py", "oph_exact/carrier.py",
                  "oph_exact/federation.py", "oph_exact/federation_archive.py", "oph_exact/federation_huge.py")


# --------------------------------------------------------------------------
# Canonical JSON and digests
# --------------------------------------------------------------------------


def canonical(x: Any) -> bytes:
    return (json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha256_of(x: Any) -> str:
    return hashlib.sha256(canonical(x)).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _round(x: float, digits: int = 12) -> float:
    r = round(float(x), digits)
    return 0.0 if r == 0 else r


def _sig(x: float, digits: int) -> float:
    return 0.0 if x == 0 else float(f"{float(x):.{digits}g}")


# --------------------------------------------------------------------------
# Geometry: the tower cells, the production gluing, the components
# --------------------------------------------------------------------------


def geometry_dir(cache: Path, level: int) -> Path:
    return Path(cache) / f"L{level}"


def build_geometry(level: int, cache: Path, log=print) -> dict[str, Any]:
    """Build (once) and cache the gluing of one level with the pinned production modules."""

    out = geometry_dir(cache, level)
    meta_path = out / "meta.json"
    if meta_path.is_file():
        return json.loads(meta_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    points, left, right = geodesic_icosahedral_patch_arrays(level, patch_basis="cells")
    carriers = int(points.shape[0])
    if carriers != 20 * 4**level:
        raise AssertionError("cell count differs from 20 * 4^L")
    port_map = assign_echosahedral_ports(left, right, carriers, points=points)
    if port_map.routing_mode != "icosahedral_directional_assignment":
        raise AssertionError("production geometric routing was not selected")
    if port_map.overflow_count != 0:
        raise AssertionError("production routing overflowed a twelve-slot carrier")
    lp = port_map.left_port.astype(np.int64)
    rp = port_map.right_port.astype(np.int64)
    inter = np.stack([left.astype(np.int64), lp, right.astype(np.int64), rp], axis=1)
    slots = np.concatenate([left * PORTS + lp, right * PORTS + rp])
    if np.unique(slots).size != slots.size:
        raise AssertionError("a carrier port is glued more than once")
    antipode = np.asarray(carrier.antipode(), dtype=np.int64)
    alignment = np.asarray(port_map.directional_alignment, dtype=float)
    usage = np.bincount(np.concatenate([lp, rp]), minlength=PORTS)
    # components of the cell graph (= components of the port graph, every carrier being connected)
    m = inter.shape[0]
    adjacency = sparse.coo_matrix((np.ones(m), (inter[:, 0], inter[:, 2])), shape=(carriers, carriers))
    components, cell_label = csgraph.connected_components(adjacency, directed=False)
    order = np.unique(cell_label, return_index=True)[1]
    relabel = np.empty(components, dtype=np.int64)
    relabel[np.argsort(order)] = np.arange(components)
    cell_label = relabel[cell_label].astype(np.int32)
    pent = F._pentagonal_cells(level)
    vtx = vertex_cells(level)
    np.save(out / "inter.npy", inter.astype(np.int32))
    np.save(out / "cell_component.npy", cell_label)
    np.save(out / "pentagonal.npy", pent.astype(bool))
    np.save(out / "vertex_cells.npy", vtx.astype(np.int32))
    np.save(out / "cell_points.npy", np.asarray(points, dtype=np.float64))
    meta = {
        "schema": "oph.exact.federation-huge.geometry.v1",
        "level": level,
        "carriers": carriers,
        "ports": PORTS * carriers,
        "intra_seams": INTRA * carriers,
        "inter_seams": int(m),
        "seams": INTRA * carriers + int(m),
        "components": int(components),
        "gluing": {
            "mode": "port_pair",
            "convention": "oph_fpe.core.screen_ports.assign_echosahedral_ports(left, right, N, points=cell_centres)",
            "routing_mode": port_map.routing_mode,
            "glued_ports_per_carrier": 3,
            "antipodal_consistent_fraction": _round(float(np.mean(antipode[lp] == rp))),
            "alignment_min": _round(float(alignment.min())),
            "alignment_mean": _round(float(alignment.mean())),
            "port_usage_histogram": usage.tolist(),
            "port_pairs_sha256": sha256_of(inter.tolist()),
            "inter_seam_array_sha256": array_sha256(inter, "int32"),
            "local_frame_hash": port_map.local_frame_hash,
            "local_frame_hash_note": "platform-specific float fingerprint; the port pairs digest is the portable identity",
            "source_derived": False,
            "declared_convention": True,
        },
        "seam_order": "intra seams carrier-major in carrier.seams() order (seam s: carrier s // 30, template s % 30), "
                      "then inter seams in dual-edge order; both endpoints of inter seam k are 12 * cell + port",
        "pentagonal_cells": int(pent.sum()),
        "arrays": {name: {"sha256": file_sha256(out / f"{name}.npy"), "bytes": (out / f"{name}.npy").stat().st_size}
                   for name in ("inter", "cell_component", "pentagonal", "vertex_cells", "cell_points")},
        "module_pins": {p: file_sha256(ROOT / p) for p in PINNED_MODULES if (ROOT / p).is_file()},
        "seconds": round(time.perf_counter() - t0, 3),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    log(f"geometry L{level}: {carriers} carriers, {m} inter seams, {components} component(s), {meta['seconds']} s")
    return meta


class Geometry:
    """Memory-mapped access to one level's cached gluing."""

    def __init__(self, cache: Path, level: int) -> None:
        out = geometry_dir(cache, level)
        self.meta = json.loads((out / "meta.json").read_text())
        self.level = level
        self.carriers = int(self.meta["carriers"])
        self.ports = PORTS * self.carriers
        self.intra_count = INTRA * self.carriers
        self.inter = np.load(out / "inter.npy", mmap_mode="r")
        self.inter_count = int(self.inter.shape[0])
        self.seams = self.intra_count + self.inter_count
        self.cell_component = np.load(out / "cell_component.npy", mmap_mode="r")
        self.pentagonal = np.load(out / "pentagonal.npy")
        self.vertex_cells = np.load(out / "vertex_cells.npy")
        self.template = np.asarray(carrier.seams(), dtype=np.int64)
        # contiguous int32 endpoint arrays of the inter seams (port ids)
        inter = np.asarray(self.inter, dtype=np.int64)
        self.ia = np.ascontiguousarray(inter[:, 0] * PORTS + inter[:, 1], dtype=np.int32)
        self.ib = np.ascontiguousarray(inter[:, 2] * PORTS + inter[:, 3], dtype=np.int32)
        self.port_component = np.repeat(np.asarray(self.cell_component, dtype=np.int32), PORTS)
        self.components = int(self.meta["components"])
        self._inter64: np.ndarray | None = None
        self.mean_denominator = Fraction(2 * self.seams, self.carriers)

    @property
    def inter64(self) -> np.ndarray:
        if self._inter64 is None:
            self._inter64 = np.asarray(self.inter, dtype=np.int64)
        return self._inter64

    def seam_endpoints(self, seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Port ids of the seams ``seq`` (int64), intra by arithmetic, inter by table."""

        seq = np.asarray(seq, dtype=np.int64)
        intra = seq < self.intra_count
        a = np.empty(seq.size, dtype=np.int64)
        b = np.empty(seq.size, dtype=np.int64)
        s = seq[intra]
        c, t = np.divmod(s, INTRA)
        a[intra] = c * PORTS + self.template[t, 0]
        b[intra] = c * PORTS + self.template[t, 1]
        k = seq[~intra] - self.intra_count
        a[~intra] = self.ia[k]
        b[~intra] = self.ib[k]
        return a, b

    def seam_differences_sq_sum(self, x: np.ndarray) -> float:
        """``Phi(x) = sum over seams (x_i - x_j)^2`` without materializing all seams at once."""

        xf = np.asarray(x, dtype=np.float64)
        blocks = xf.reshape(self.carriers, PORTS)
        phi = 0.0
        for t in range(INTRA):
            d = blocks[:, self.template[t, 0]] - blocks[:, self.template[t, 1]]
            phi += float(np.dot(d, d))
        for lo in range(0, self.inter_count, DRAW_CHUNK):
            d = xf[self.ia[lo:lo + DRAW_CHUNK]] - xf[self.ib[lo:lo + DRAW_CHUNK]]
            phi += float(np.dot(d, d))
        return phi

    def seam_abs_differences(self, x: np.ndarray) -> tuple[int, int]:
        """``(odd tie seams, maximum seam difference)`` of an integer state."""

        xi = np.asarray(x, dtype=np.int64)
        blocks = xi.reshape(self.carriers, PORTS)
        odd = 0
        biggest = 0
        for t in range(INTRA):
            d = np.abs(blocks[:, self.template[t, 0]] - blocks[:, self.template[t, 1]])
            odd += int(np.count_nonzero(d == 1))
            biggest = max(biggest, int(d.max()) if d.size else 0)
        for lo in range(0, self.inter_count, DRAW_CHUNK):
            d = np.abs(xi[self.ia[lo:lo + DRAW_CHUNK]] - xi[self.ib[lo:lo + DRAW_CHUNK]])
            odd += int(np.count_nonzero(d == 1))
            biggest = max(biggest, int(d.max()) if d.size else 0)
        return odd, biggest


def initial_loads(level: int, ports: int) -> np.ndarray:
    return F.initial_loads(level, ports).astype(np.int8)


def schedule_seed(level: int, index: int) -> int:
    return SCHEDULE_SEED_BASE + 1000 * level + 100 * GLUING_INDEX + index


# --------------------------------------------------------------------------
# Native kernel
# --------------------------------------------------------------------------


_KERNEL_SOURCE = r"""
#include <stdint.h>

/* Seam s < intra_count: carrier s / 30, template s % 30; otherwise the inter table. */
#define ENDPOINTS(s) \
    int64_t i, j; \
    if ((s) < intra_count) { \
        const int64_t c = (s) / 30; const int64_t t = (s) - 30 * c; \
        i = 12 * c + tpl_a[t]; j = 12 * c + tpl_b[t]; \
    } else { const int64_t k = (s) - intra_count; i = ia[k]; j = ib[k]; }

/* Integer nearest agreement on an int8 state.  counts = {descents, swaps, waits, unit
   transfers, decrement identity violations}.  Returns the index (within this call) of the
   first move after which V equals v_min, or -1. */
int64_t oph_huge_integer(int8_t *x, const int8_t *tpl_a, const int8_t *tpl_b,
                         const int32_t *ia, const int32_t *ib, int64_t intra_count,
                         const int32_t *seq, const int8_t *coin, int64_t n,
                         int64_t *counts, int64_t *v, int64_t v_min)
{
    int64_t first = -1;
    int64_t vv = v[0];
    int64_t descents = 0, swaps = 0, waits = 0, transfers = 0, violations = 0;
    for (int64_t t = 0; t < n; ++t) {
        const int64_t s = seq[t];
        ENDPOINTS(s)
        const int64_t xi = x[i];
        const int64_t xj = x[j];
        const int64_t tot = xi + xj;
        const int64_t lo = tot >> 1;
        const int64_t hi = tot - lo;
        const int64_t ni = coin[t] ? hi : lo;
        const int64_t nj = tot - ni;
        const int64_t d = xi - xj;
        if (d == 0) { ++waits; continue; }
        if (d == 1 || d == -1) {
            if (ni == xi) { ++waits; continue; }
            ++swaps;
        } else {
            const int64_t ad = d < 0 ? -d : d;
            const int64_t change = ni * ni + nj * nj - xi * xi - xj * xj;
            if (change != -((ad * ad - (ad & 1)) / 2)) ++violations;
            transfers += ad >> 1;
            ++descents;
            vv += change;
        }
        x[i] = (int8_t)ni;
        x[j] = (int8_t)nj;
        if (first < 0 && vv == v_min) first = t;
    }
    counts[0] += descents; counts[1] += swaps; counts[2] += waits;
    counts[3] += transfers; counts[4] += violations;
    v[0] = vv;
    return first;
}

/* Mean law on a float64 state.  counts = {waits, descent violations}; ledger accumulates
   (x_i - x_j)^2 / 2 over non-wait moves. */
void oph_huge_mean(double *x, const int8_t *tpl_a, const int8_t *tpl_b,
                   const int32_t *ia, const int32_t *ib, int64_t intra_count,
                   const int32_t *seq, int64_t n, double *ledger, int64_t *counts)
{
    int64_t waits = 0, violations = 0;
    double led = 0.0;
    for (int64_t t = 0; t < n; ++t) {
        const int64_t s = seq[t];
        ENDPOINTS(s)
        const double xi = x[i];
        const double xj = x[j];
        const double d = xi - xj;
        if (d == 0.0) { ++waits; continue; }
        const double half_sq = 0.5 * (d * d);
        if (!(half_sq > 0.0)) ++violations;
        led += half_sq;
        const double m = 0.5 * (xi + xj);
        x[i] = m;
        x[j] = m;
    }
    ledger[0] += led;
    counts[0] += waits;
    counts[1] += violations;
}
"""

_NATIVE: dict[str, Any] = {"lib": None, "tried": False, "error": None}


def native_kernel() -> Any:
    """Compile once (``-O2 -ffp-contract=off``) and load the kernel."""

    if _NATIVE["tried"]:
        if _NATIVE["lib"] is None:
            raise RuntimeError(f"native kernel unavailable: {_NATIVE['error']}")
        return _NATIVE["lib"]
    _NATIVE["tried"] = True
    digest = hashlib.sha256(_KERNEL_SOURCE.encode("utf-8")).hexdigest()[:16]
    directory = Path(os.environ.get("OPH_EXACT_KERNEL_DIR") or Path(tempfile.gettempdir()) / "oph_exact_kernels")
    directory.mkdir(parents=True, exist_ok=True)
    suffix = ".dll" if sys.platform == "win32" else ".so"
    library = directory / f"oph_huge_kernel_{digest}{suffix}"
    try:
        if not library.exists():
            compiler = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
            if compiler is None:
                raise RuntimeError("no C compiler on PATH")
            source = directory / f"oph_huge_kernel_{digest}.c"
            source.write_text(_KERNEL_SOURCE, encoding="utf-8")
            staging = directory / f"oph_huge_kernel_{digest}.{os.getpid()}{suffix}"
            subprocess.run([compiler, "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-o", str(staging), str(source)],
                           check=True, capture_output=True)
            os.replace(staging, library)
        lib = ctypes.CDLL(str(library))
        i8p = ctypes.POINTER(ctypes.c_int8)
        i32p = ctypes.POINTER(ctypes.c_int32)
        i64p = ctypes.POINTER(ctypes.c_int64)
        f64p = ctypes.POINTER(ctypes.c_double)
        lib.oph_huge_integer.argtypes = [i8p, i8p, i8p, i32p, i32p, ctypes.c_int64, i32p, i8p, ctypes.c_int64, i64p, i64p, ctypes.c_int64]
        lib.oph_huge_integer.restype = ctypes.c_int64
        lib.oph_huge_mean.argtypes = [f64p, i8p, i8p, i32p, i32p, ctypes.c_int64, i32p, ctypes.c_int64, f64p, i64p]
        lib.oph_huge_mean.restype = None
        _NATIVE["lib"] = lib
        return lib
    except Exception as error:  # pragma: no cover - environment dependent
        _NATIVE["error"] = f"{type(error).__name__}: {error}"
        raise RuntimeError(f"native kernel unavailable: {_NATIVE['error']}") from error


def kernel_source_sha256() -> str:
    return hashlib.sha256(_KERNEL_SOURCE.encode("utf-8")).hexdigest()


def _ptr(array: np.ndarray, ctype):
    return array.ctypes.data_as(ctypes.POINTER(ctype))


# --------------------------------------------------------------------------
# Draws: the archive's stream, taken in chunks
# --------------------------------------------------------------------------


def draw_sweep(rng: np.random.Generator, seams: int, coins: bool, chunk: int = DRAW_CHUNK) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """One sweep's draws, chunked but stream-identical to ``integers(0, |S|, size=|S|)`` then coins.

    Returns the seam indices as int32, the coins as int8 (or None) and the archive's per-sweep
    draw digests ``[sha256(seq int64 bytes), sha256(coin int64 bytes)]``.
    """

    seq = np.empty(seams, dtype=np.int32)
    h_seq = hashlib.sha256()
    for lo in range(0, seams, chunk):
        part = rng.integers(0, seams, size=min(chunk, seams - lo), dtype=np.int64)
        h_seq.update(part.astype("<i8", copy=False).tobytes())
        seq[lo:lo + part.size] = part
    coin = None
    digests = [h_seq.hexdigest()]
    if coins:
        coin = np.empty(seams, dtype=np.int8)
        h_coin = hashlib.sha256()
        for lo in range(0, seams, chunk):
            part = rng.integers(0, 2, size=min(chunk, seams - lo), dtype=np.int64)
            h_coin.update(part.astype("<i8", copy=False).tobytes())
            coin[lo:lo + part.size] = part
        digests.append(h_coin.hexdigest())
    return seq, coin, digests


# --------------------------------------------------------------------------
# Expected terminal data from the loads alone
# --------------------------------------------------------------------------


def expectation(geo: Geometry, loads: np.ndarray) -> dict[str, Any]:
    comp = component_expectation(geo.port_component, np.asarray(loads, dtype=np.int64))
    return {
        "component_count": int(comp["size"].size),
        "component_sizes": comp["size"].tolist() if comp["size"].size <= 64 else None,
        "expected_integer_quotient_hash": expected_multiset_hash(comp["size"], comp["q"], comp["r"]),
        "balanced_minimum": balanced_minimum(comp["size"], comp["q"], comp["r"]),
        "component_total": comp["total"].tolist() if comp["size"].size <= 64 else None,
        "mean_minimum": float(sum(Fraction(int(t) ** 2, int(m)) for t, m in zip(comp["total"].tolist(), comp["size"].tolist()))),
        "_arrays": comp,
    }


def integer_quotient_hash(geo: Geometry, x: np.ndarray) -> str:
    labels = geo.port_component
    xi = np.asarray(x, dtype=np.int64)
    entries = []
    for c in range(geo.components):
        mask = labels == c
        values, counts = np.unique(xi[mask], return_counts=True)
        entries.append([int(mask.sum()), [[int(v), int(k)] for v, k in zip(values, counts)]])
    return sha256_of({"canonicalizer": "component_multiset", "components": entries})


# --------------------------------------------------------------------------
# One schedule of the integer law
# --------------------------------------------------------------------------


def run_integer(geo: Geometry, loads: np.ndarray, seed: int, out: Path, *, max_sweeps: int = 400_000,
                checkpoint_every: int = 8, keep_checkpoints: int = 2, log=print) -> dict[str, Any]:
    lib = native_kernel()
    x = np.ascontiguousarray(np.asarray(loads, dtype=np.int8).copy())
    if np.any(x < 0) or np.any(x > LOAD_MAX):
        raise ValueError("loads outside the declared range")
    tpl_a = np.ascontiguousarray(geo.template[:, 0], dtype=np.int8)
    tpl_b = np.ascontiguousarray(geo.template[:, 1], dtype=np.int8)
    exp = expectation(geo, loads)
    v_min = int(exp["balanced_minimum"])
    v = np.array([int(np.dot(x.astype(np.int64), x.astype(np.int64)))], dtype=np.int64)
    v0 = int(v[0])
    counts = np.zeros(5, dtype=np.int64)
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    ledger = [v0]
    per_sweep = []
    draw_digests = []
    state_digests = [array_sha256(x, "int8")]
    checkpoints = []
    sweep = 0
    first_attempt = 0 if v0 == v_min else -1
    started = time.perf_counter()
    while first_attempt < 0 and sweep < max_sweeps:
        seq, coin, digests = draw_sweep(rng, geo.seams, coins=True)
        before = counts.copy()
        first = int(lib.oph_huge_integer(_ptr(x, ctypes.c_int8), _ptr(tpl_a, ctypes.c_int8), _ptr(tpl_b, ctypes.c_int8),
                                         _ptr(geo.ia, ctypes.c_int32), _ptr(geo.ib, ctypes.c_int32), geo.intra_count,
                                         _ptr(seq, ctypes.c_int32), _ptr(coin, ctypes.c_int8), geo.seams,
                                         _ptr(counts, ctypes.c_int64), _ptr(v, ctypes.c_int64), v_min))
        if first >= 0:
            first_attempt = sweep * geo.seams + first + 1
        sweep += 1
        delta = counts - before
        ledger.append(int(v[0]))
        state_digests.append(array_sha256(x, "int8"))
        draw_digests.append(digests)
        per_sweep.append({"sweep": sweep, "V": int(v[0]), "descents": int(delta[0]), "swaps": int(delta[1]),
                          "waits": int(delta[2]), "unit_transfers": int(delta[3]), "violations": int(delta[4])})
        if sweep % checkpoint_every == 0:
            path = out / f"state_sweep{sweep:05d}.npy"
            np.save(path, x)
            checkpoints.append({"sweep": sweep, "path": path.name, "sha256": state_digests[-1]})
            for old in checkpoints[:-keep_checkpoints]:
                p = out / old["path"]
                if p.exists():
                    p.unlink()
                    old["retained"] = False
        if sweep % 8 == 0 or first_attempt >= 0:
            log(f"  seed {seed}: sweep {sweep} V={int(v[0])} (min {v_min}) {time.perf_counter() - started:.0f}s")
    v_state = int(np.dot(x.astype(np.int64), x.astype(np.int64)))
    odd, biggest = geo.seam_abs_differences(x)
    terminal_path = out / "terminal_state.npy"
    np.save(terminal_path, x)
    total = int(x.astype(np.int64).sum())
    result = {
        "seed": int(seed),
        "terminated": bool(first_attempt >= 0),
        "sweeps": sweep,
        "attempts": sweep * geo.seams,
        "attempts_to_balanced_class": int(first_attempt),
        "descents": int(counts[0]), "swaps": int(counts[1]), "waits": int(counts[2]),
        "unit_transfers": int(counts[3]),
        "unit_transfer_decrement_identity_violations": int(counts[4]),
        "strict_descent_violations": int(counts[4]),
        "V_initial": v0, "V_minimum": v_min, "V_terminal": int(v[0]),
        "descent_ledger_exact": bool(int(v[0]) == v_state),
        "conservation_exact": bool(total == int(np.asarray(loads, dtype=np.int64).sum())),
        "V_ledger": ledger,
        "per_sweep": per_sweep,
        "draw_sha256_per_sweep": draw_digests,
        "state_sha256_per_sweep": state_digests,
        "checkpoints": [c for c in checkpoints if c.get("retained", True)],
        "odd_tie_seams_at_termination": odd,
        "max_seam_difference_at_termination": biggest,
        "quotient_hash": integer_quotient_hash(geo, x),
        "quotient_hash_equals_expected": None,
        "terminal_state": {"path": terminal_path.name, "dtype": "int8", "sha256": state_digests[-1]},
        "seconds": round(time.perf_counter() - started, 3),
    }
    result["quotient_hash_equals_expected"] = bool(result["quotient_hash"] == exp["expected_integer_quotient_hash"])
    (out / "schedule.json").write_bytes(canonical(result))
    return result


# --------------------------------------------------------------------------
# One budgeted schedule of the mean law
# --------------------------------------------------------------------------


def run_mean(geo: Geometry, loads: np.ndarray, seed: int, out: Path, *, sweeps_budget: int, log=print) -> dict[str, Any]:
    lib = native_kernel()
    x = np.ascontiguousarray(np.asarray(loads, dtype=np.float64).copy())
    tpl_a = np.ascontiguousarray(geo.template[:, 0], dtype=np.int8)
    tpl_b = np.ascontiguousarray(geo.template[:, 1], dtype=np.int8)
    exp = expectation(geo, loads)
    comp = exp["_arrays"]
    mean_of_port = (comp["total"].astype(np.float64) / comp["size"].astype(np.float64))[geo.port_component]
    v_min = float(exp["mean_minimum"])
    ledger = np.zeros(1, dtype=np.float64)
    counts = np.zeros(2, dtype=np.int64)
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    phi0 = geo.seam_differences_sq_sum(x)
    v0 = float(np.dot(x, x))
    phi = phi0
    v = v0
    v_ledger = [_round(v0)]
    phi_trace = [_sig(phi0, 12)]
    per_sweep = []
    draw_digests = []
    sweep = 0
    terminated = phi < FLOAT_PHI_THRESHOLD
    started = time.perf_counter()
    while not terminated and sweep < sweeps_budget:
        seq, _coin, digests = draw_sweep(rng, geo.seams, coins=False)
        before_ledger = float(ledger[0])
        before = counts.copy()
        lib.oph_huge_mean(_ptr(x, ctypes.c_double), _ptr(tpl_a, ctypes.c_int8), _ptr(tpl_b, ctypes.c_int8),
                          _ptr(geo.ia, ctypes.c_int32), _ptr(geo.ib, ctypes.c_int32), geo.intra_count,
                          _ptr(seq, ctypes.c_int32), geo.seams, _ptr(ledger, ctypes.c_double), _ptr(counts, ctypes.c_int64))
        sweep += 1
        phi = geo.seam_differences_sq_sum(x)
        v_new = float(np.dot(x, x))
        increment = float(ledger[0]) - before_ledger
        drop = v - v_new
        per_sweep.append({"sweep": sweep, "V": _round(v_new), "Phi": _sig(phi, 12), "ledger_increment": _round(increment),
                          "ledger_relative_error": _sig(abs(drop - increment) / max(v0, 1.0), 6),
                          "waits": int(counts[0] - before[0]), "violations": int(counts[1] - before[1])})
        v = v_new
        v_ledger.append(_round(v))
        phi_trace.append(_sig(phi, 12))
        draw_digests.append(digests)
        terminated = phi < FLOAT_PHI_THRESHOLD
        if sweep % 8 == 0 or terminated:
            log(f"  seed {seed}: float sweep {sweep} Phi={phi:.6g} V-V_min={v - v_min:.6g} {time.perf_counter() - started:.0f}s")
    deviation = float(np.max(np.abs(x - mean_of_port)))
    scaled = comp["size"].astype(np.float64)[geo.port_component] * x
    residual = float(np.max(np.abs(scaled - np.rint(scaled))))
    snap_ok = bool(residual < LATTICE_SNAP_MARGIN)
    quotient_hash = None
    if snap_ok:
        q = np.rint(scaled).astype(np.int64)
        quotient_hash = sha256_of({"canonicalizer": "component_lattice_snap", "component_sizes": comp["size"].tolist(),
                                   "component_of_port_sha256": array_sha256(geo.port_component, "int32"), "q_sha256": array_sha256(q, "int64")})
    total_error = abs((v0 - v) - float(ledger[0])) / max(v0, 1.0)
    np.save(out / "terminal_state.npy", x)
    result = {
        "seed": int(seed), "terminated": bool(terminated), "sweeps": sweep, "attempts": sweep * geo.seams,
        "waits": int(counts[0]), "strict_descent_violations": int(counts[1]),
        "phi_initial": _sig(phi0, 12), "phi_terminal": _sig(phi, 12), "phi_trace": phi_trace,
        "V_initial": _round(v0), "V_terminal": _round(v), "V_minimum": _round(v_min), "V_ledger": v_ledger,
        "descent_ledger_relative_error": _sig(total_error, 6), "descent_ledger_relative_error_below_1e-9": bool(total_error < 1e-9),
        "per_sweep": per_sweep, "draw_sha256_per_sweep": draw_digests,
        "max_abs_deviation_from_component_mean": _sig(deviation, 6),
        "lattice_residual_max": _sig(residual, 3), "lattice_snap_unambiguous": snap_ok,
        "terminal_quotient_hash": quotient_hash,
        "terminal_quotient_hash_note": "component-lattice snap payload with the component labels and q as array digests "
                                       "(the reference lane embeds the lists); withheld unless unambiguous",
        "terminal_state": {"path": "terminal_state.npy", "dtype": "float64", "sha256": array_sha256(x, "float64")},
        "seconds": round(time.perf_counter() - started, 3),
    }
    (out / "schedule.json").write_bytes(canonical(result))
    return result


# --------------------------------------------------------------------------
# Response kernels on local balls
# --------------------------------------------------------------------------


def cell_adjacency(geo: Geometry) -> sparse.csr_matrix:
    inter = geo.inter64
    m = inter.shape[0]
    data = np.ones(2 * m, dtype=np.int8)
    rows = np.concatenate([inter[:, 0], inter[:, 2]])
    cols = np.concatenate([inter[:, 2], inter[:, 0]])
    return sparse.csr_matrix((data, (rows, cols)), shape=(geo.carriers, geo.carriers))


def local_ball(adj: sparse.csr_matrix, cell: int, radius: int) -> np.ndarray:
    """Cells within graph distance ``radius`` of ``cell`` on the cell graph."""

    reached = np.zeros(adj.shape[0], dtype=bool)
    reached[cell] = True
    frontier = np.array([cell])
    for _ in range(radius):
        nxt = np.unique(adj[frontier].indices)
        nxt = nxt[~reached[nxt]]
        if nxt.size == 0:
            break
        reached[nxt] = True
        frontier = nxt
    return np.flatnonzero(reached)


def response_kernels_local(geo: Geometry, adj: sparse.csr_matrix, cell: int, steps: Sequence[int]) -> dict[str, Any]:
    """``K_n`` of one carrier by propagating the centered impulses on the ball of radius ``2 max(n) + 1``.

    The operator moves one hop per step, so the readback at the probed carrier after ``2n`` steps
    is exactly the whole-federation readback (``federation.response_kernels``).
    """

    last = 2 * max(steps)
    cells = local_ball(adj, cell, last + 1)
    index = {int(c): k for k, c in enumerate(cells.tolist())}
    n_local = cells.size * PORTS
    # local Laplacian: intra seams of every ball cell, inter seams with both ends in the ball
    template = geo.template
    rows = []
    cols = []
    for t in range(INTRA):
        a = np.arange(cells.size) * PORTS + template[t, 0]
        b = np.arange(cells.size) * PORTS + template[t, 1]
        rows.append(a); cols.append(b)
    inter = geo.inter64
    in_ball = np.zeros(geo.carriers, dtype=bool)
    in_ball[cells] = True
    mask = in_ball[inter[:, 0]] & in_ball[inter[:, 2]]
    sub = inter[mask]
    la = np.array([index[int(c)] for c in sub[:, 0]], dtype=np.int64) * PORTS + sub[:, 1]
    lb = np.array([index[int(c)] for c in sub[:, 2]], dtype=np.int64) * PORTS + sub[:, 3]
    rows.append(la); cols.append(lb)
    r = np.concatenate(rows); c = np.concatenate(cols)
    adjacency = sparse.coo_matrix((np.ones(r.size), (r, c)), shape=(n_local, n_local))
    adjacency = (adjacency + adjacency.T).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = (sparse.diags(degree) - adjacency).tocsr()
    D = float(geo.mean_denominator)
    block = slice(index[cell] * PORTS, (index[cell] + 1) * PORTS)
    y = np.zeros((n_local, PORTS))
    q = np.eye(PORTS) - np.ones((PORTS, PORTS)) / PORTS
    y[block, :] = q
    target = {2 * n: n for n in steps}
    out = {}
    for step in range(1, last + 1):
        y = y - (laplacian @ y) / D
        scale = np.max(np.abs(y))
        if scale > 0:
            y /= scale
        if step in target:
            k = q @ y[block, :]
            k = 0.5 * (k + k.T)
            out[target[step]] = PORTS * k / np.trace(k)
    p_slow = carrier.slow_band_projector()
    return {"cell": int(cell), "ball_cells": int(cells.size), "ball_ports": int(n_local),
            "kernels": {str(n): out[n].tolist() for n in steps},
            "slow_band_share": {str(n): _sig(float(np.trace(p_slow @ out[n] @ p_slow) / np.trace(out[n])), 12) for n in steps},
            "top_eigenvalues": {str(n): [_sig(float(v), 12) for v in np.sort(np.linalg.eigvalsh(out[n]))[::-1][:4]] for n in steps}}


def isolated_kernel_reference(steps: Sequence[int]) -> dict[str, Any]:
    p_slow = carrier.slow_band_projector()
    ref = {}
    for n in steps:
        k = carrier.normalized_response_kernel(n)
        ref[str(n)] = {"slow_band_share": _sig(float(np.trace(p_slow @ k @ p_slow) / np.trace(k)), 12),
                       "max_abs_difference_to_4P_slow": _sig(float(np.max(np.abs(k - 4.0 * p_slow))), 6)}
    return ref


# --------------------------------------------------------------------------
# Whole level
# --------------------------------------------------------------------------


def _schedule_task(args: tuple) -> dict[str, Any]:
    cache, level, law, seed, out, budget = args
    geo = Geometry(cache, level)
    loads = initial_loads(level, geo.ports)
    if law == "integer":
        return run_integer(geo, loads, seed, Path(out), log=lambda m: print(m, flush=True))
    return run_mean(geo, loads, seed, Path(out), sweeps_budget=budget, log=lambda m: print(m, flush=True))


_KERNEL_GEO: dict[str, Any] = {}


def _kernel_task(args: tuple) -> dict[str, Any]:
    cache, level, cell, steps = args
    key = f"{cache}:{level}"
    if _KERNEL_GEO.get("key") != key:
        geo = Geometry(cache, level)
        _KERNEL_GEO.update({"key": key, "geo": geo, "adj": cell_adjacency(geo)})
    return response_kernels_local(_KERNEL_GEO["geo"], _KERNEL_GEO["adj"], int(cell), steps)


def build(level: int, out: Path, cache: Path, *, schedules: int = DEFAULT_SCHEDULES, workers: int = 1,
          float_sweeps: int = 64, float_schedules: int = 1, kernel_cells: int = 64,
          kernel_steps: Sequence[int] = KERNEL_STEPS, long_steps: Sequence[int] = (300,), long_cells: int = 4,
          log=print) -> dict[str, Any]:
    wall0 = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    meta = build_geometry(level, cache, log=log)
    geo = Geometry(cache, level)
    loads = initial_loads(level, geo.ports)
    exp = expectation(geo, loads)
    log(f"L{level}: {geo.carriers} carriers, {geo.seams} seams, expected hash {exp['expected_integer_quotient_hash'][:16]}, V_min {exp['balanced_minimum']}")
    seeds = [schedule_seed(level, k) for k in range(schedules)]
    tasks = [(cache, level, "integer", seed, out / f"integer_{seed}", 0) for seed in seeds]
    tasks += [(cache, level, "mean", schedule_seed(level, k), out / f"mean_{schedule_seed(level, k)}", float_sweeps) for k in range(float_schedules)]
    t1 = time.perf_counter()
    if workers > 1 and len(tasks) > 1:
        import multiprocessing as mp
        native_kernel()
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.map(_schedule_task, tasks, chunksize=1)
    else:
        results = [_schedule_task(t) for t in tasks]
    t_schedules = time.perf_counter() - t1
    integer = [r for r in results if "quotient_hash" in r]
    floats = [r for r in results if "phi_trace" in r]
    for r in integer + floats:
        r.pop("per_sweep", None)  # kept in the per-schedule schedule.json
    hashes = sorted({r["quotient_hash"] for r in integer})
    # kernels on local balls
    t2 = time.perf_counter()
    cells = kernel_sample_cells(level, geo.carriers, geo.vertex_cells, kernel_cells)
    kernel_tasks = [(cache, level, c, tuple(kernel_steps) + (tuple(long_steps) if k < long_cells else ())) for k, c in enumerate(cells)]
    if workers > 1 and len(kernel_tasks) > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(workers, len(kernel_tasks))) as pool:
            kernels = pool.map(_kernel_task, kernel_tasks, chunksize=1)
    else:
        kernels = [_kernel_task(t) for t in kernel_tasks]
    for k, entry in zip(kernels, kernels):
        log(f"  kernel cell {entry['cell']}: ball {entry['ball_cells']} cells, steps {list(entry['kernels'].keys())}")
    log(f"  kernels: {len(kernels)} cells in {time.perf_counter() - t2:.0f}s")
    shares = {str(n): [k["slow_band_share"][str(n)] for k in kernels] for n in kernel_steps}
    long_shares = {str(n): [k["slow_band_share"][str(n)] for k in kernels[:long_cells] if str(n) in k["slow_band_share"]] for n in long_steps}
    t_kernels = time.perf_counter() - t2
    receipt = {
        "schema": SCHEMA,
        "level": level,
        "carriers": geo.carriers, "ports": geo.ports, "seams": geo.seams, "inter_seams": geo.inter_count,
        "components": geo.components,
        "geometry": {k: v for k, v in meta.items() if k not in ("arrays",)} | {"arrays": meta["arrays"], "cache_dir": str(geometry_dir(cache, level))},
        "laws": {
            "integer_law": "carrier.integer_nearest_agreement: s = x_a + x_b, lo = floor(s/2), hi = s - lo; the first endpoint receives hi when the coin is 1; "
                           "d = 0 wait; |d| = 1 wait or swap (V unchanged); |d| >= 2 descent lowering V by (d^2 - (d mod 2))/2",
            "mean_law": "E_e = I - b_e b_e^T / 2: both endpoints replaced by 0.5 (x_i + x_j)",
            "descent_functional": "V(x) = sum_p x_p^2",
            "termination_potential": "Phi(x) = sum_seams (x_i - x_j)^2",
            "integer_terminal_class": "balanced class: readings in {q, q+1} per component with V = V_min",
            "integer_canonicalizer": "component multiset, components ordered by lowest port; sha256 of canonical JSON",
            "float_termination": f"first sweep end with Phi < {FLOAT_PHI_THRESHOLD:g}, within the declared sweep budget",
            "mean_denominator": str(geo.mean_denominator),
        },
        "rng": {"bit_generator": "PCG64", "construction": "numpy.random.default_rng(seed)", "numpy_version": np.__version__,
                "integer_law_draw": "per sweep: seq = integers(0, |S|, size=|S|, dtype=int64), then coin = integers(0, 2, size=|S|, dtype=int64); "
                                    f"drawn in contiguous chunks of {DRAW_CHUNK} that reproduce the single-call stream, digested incrementally",
                "mean_law_draw": "per sweep: seq = integers(0, |S|, size=|S|, dtype=int64)"},
        "seeds": {"loads": {"rule": "numpy.random.default_rng(20260909 + level).integers(0, 6, size=ports)", "load_max": LOAD_MAX},
                  "schedules": {"rule": "909000 + 1000 * level + 100 * gluing_index + schedule_index, gluing_index port_pair = 1",
                                "integer": seeds, "mean": [schedule_seed(level, k) for k in range(float_schedules)]},
                  "kernel_sample": {"rule": "one cell per pentagonal vertex (lowest cell index), then default_rng(20260909 + 17 * level).choice(remaining)", "cells": cells}},
        "expected": {k: v for k, v in exp.items() if k != "_arrays"},
        "integer_law": {
            "schedules": len(integer),
            "all_terminated": bool(all(r["terminated"] for r in integer)),
            "unique_quotient_hash_count": len(hashes),
            "quotient_hash_equals_expected_all": bool(all(r["quotient_hash_equals_expected"] for r in integer)),
            "sweeps": {"min": min(r["sweeps"] for r in integer), "max": max(r["sweeps"] for r in integer)},
            "attempts_to_balanced_class": {"min": min(r["attempts_to_balanced_class"] for r in integer), "max": max(r["attempts_to_balanced_class"] for r in integer)},
            "unit_transfers": {"min": min(r["unit_transfers"] for r in integer), "max": max(r["unit_transfers"] for r in integer)},
            "violations_total": int(sum(r["unit_transfer_decrement_identity_violations"] for r in integer)),
            "conservation_exact_all": bool(all(r["conservation_exact"] for r in integer)),
            "max_seam_difference_at_termination": max(r["max_seam_difference_at_termination"] for r in integer),
            "entries": integer,
        },
        "mean_law_float": {
            "schedules": len(floats), "sweep_budget": float_sweeps,
            "terminated_all": bool(all(r["terminated"] for r in floats)) if floats else None,
            "phi_initial": floats[0]["phi_initial"] if floats else None,
            "phi_terminal_max": max(r["phi_terminal"] for r in floats) if floats else None,
            "descent_ledger_ok_all": bool(all(r["descent_ledger_relative_error_below_1e-9"] for r in floats)) if floats else None,
            "strict_descent_violations_total": int(sum(r["strict_descent_violations"] for r in floats)),
            "max_abs_deviation_from_component_mean_max": max(r["max_abs_deviation_from_component_mean"] for r in floats) if floats else None,
            "lattice_snap_unambiguous_all": bool(all(r["lattice_snap_unambiguous"] for r in floats)) if floats else None,
            "ambiguous_terminal_count": int(sum(1 for r in floats if r["terminal_quotient_hash"] is None)),
            "entries": floats,
        },
        "response_kernels": {
            "definition": "probe carrier c with the twelve centered impulses Q = I - J/12; propagate y <- y - (L y)/D for 2n steps with a per-step "
                          "rescale by max|y|; read back the c-block; C_n = Q y_block Q symmetrised; K_n = 12 C_n / tr C_n; computed on the ball of "
                          "graph radius 2n + 1 around c, which is exact because the operator moves one hop per step",
            "steps": list(kernel_steps),
            "long_steps": list(long_steps), "long_step_cells": int(min(long_cells, len(cells))),
            "slow_band_share_definition": "tr(P_slow K_n P_slow) / tr(K_n) with P_slow the spectral projector of the carrier seam Laplacian onto the 5 - sqrt5 band",
            "isolated_reference": isolated_kernel_reference(tuple(kernel_steps) + tuple(long_steps)),
            "glued_slow_band_share": {n: {"min": _sig(min(v), 6), "median": _sig(float(np.median(v)), 6), "max": _sig(max(v), 6)} for n, v in shares.items()},
            "glued_slow_band_share_long_steps": {n: {"min": _sig(min(v), 6), "median": _sig(float(np.median(v)), 6), "max": _sig(max(v), 6), "cells": len(v)} for n, v in long_shares.items() if v},
            "cells": kernels,
        },
        "lambda_2": {"status": "not computed: shift-invert timed out at level six; a deflated Krylov method is future work"},
        "known_boundaries": {"source_derived_gluing": False, "gluing_is_declared_convention": True, "refinement_limit": False,
                             "continuum_limit": False, "field_attachment": False, "physical_identification": False,
                             "physical_clock": False, "cofinal_gluing": False, "float_law_terminated": bool(all(r["terminated"] for r in floats)) if floats else None},
        "platform": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": sys.platform, "machine": os.uname().machine},
        "engine": {"native_kernel_source_sha256": kernel_source_sha256(), "compile_flags": "-O2 -ffp-contract=off", "workers": workers},
        "module_pins": {p: file_sha256(ROOT / p) for p in PINNED_MODULES if (ROOT / p).is_file()},
        "timings_seconds": {"geometry": meta.get("seconds"), "schedules": round(t_schedules, 3), "kernels": round(t_kernels, 3),
                            "total": round(time.perf_counter() - wall0, 3)},
    }
    (out / "receipt.json").write_bytes(canonical(receipt))
    log(f"L{level}: integer {len(integer)} schedules, {len(hashes)} hash(es), expected {'MATCH' if receipt['integer_law']['quotient_hash_equals_expected_all'] else 'DIFFERS'}, "
        f"sweeps {receipt['integer_law']['sweeps']}, glued slow-band share medians {[receipt['response_kernels']['glued_slow_band_share'][str(n)]['median'] for n in kernel_steps]}, "
        f"{receipt['timings_seconds']['total']:.0f} s")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("geometry", help="build and cache the gluing of a level")
    g.add_argument("--level", type=int, required=True)
    g.add_argument("--cache", type=Path, required=True)
    r = sub.add_parser("run", help="run the settlement of a level and write the receipt")
    r.add_argument("--level", type=int, required=True)
    r.add_argument("--cache", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--schedules", type=int, default=DEFAULT_SCHEDULES)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--float-sweeps", type=int, default=64)
    r.add_argument("--float-schedules", type=int, default=1)
    r.add_argument("--kernel-cells", type=int, default=64)
    r.add_argument("--kernel-steps", type=int, nargs="*", default=list(KERNEL_STEPS))
    r.add_argument("--long-steps", type=int, nargs="*", default=[300])
    r.add_argument("--long-cells", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "geometry":
        build_geometry(args.level, args.cache)
        return 0
    build(args.level, args.out, args.cache, schedules=args.schedules, workers=args.workers, float_sweeps=args.float_sweeps,
          float_schedules=args.float_schedules, kernel_cells=args.kernel_cells, kernel_steps=tuple(args.kernel_steps),
          long_steps=tuple(args.long_steps), long_cells=args.long_cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
