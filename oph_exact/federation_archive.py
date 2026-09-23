"""Evidence archive of the exact federation at one tower level (lane L1 archive).

Builds a self-describing run directory for the canonical seam-mean law on
``N = 20 * 4^L`` twelve-port carriers glued by the production ``port_pair``
convention of ``oph_exact.federation``:

``config.json``
    level, counts, the gluing declaration verbatim from ``federation.py``,
    laws, named seed streams, budgets, RNG algorithm and byte model.
``primitives.npz``
    seam endpoint arrays (carrier, port) for intra and inter seams, the
    initial integer loads, the port-graph component labels and the expected
    terminal multiset per component.
``integer_law_<gluing>.json`` / ``integer_terminal_<gluing>.npz``
    sixteen shuffled integer-law schedules to the balanced class with the
    per-sweep ``V`` ledger and one full terminal load vector each.
``float_law_port_pair.json`` / ``float_terminal_port_pair.npz``
    one budgeted asynchronous float schedule with the per-sweep ``V`` ledger.
``float_law_isolated.json``
    sixteen terminated float schedules of the isolated control.
``spectrum_port_pair.json`` / ``fiedler_port_pair.npz``
    the port-graph gap ``lambda_2`` with its eigenvector.
``kernel_readout.json`` / ``kernel_matrices.npz``
    per-carrier normalized centered response kernels, isolated and glued.
``archive_manifest.json``, ``verify_archive.py``, ``README.md``
    the byte binding, the simulator-free verifier and the report.

The verifier is embedded in this module (``VERIFIER_SOURCE``) and written into
every archive; it imports numpy and the standard library only.

Build: ``python -m oph_exact.federation_archive --level 6 --out runs/exact_federation_L6_canonical_20260909``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

from oph_exact import carrier
from oph_exact import federation as F
from oph_fpe.core.icosahedral import build_geodesic_icosahedral_tower

ARCHIVE_SCHEMA = "oph.exact_federation_archive.v1"
DEFAULT_ARCHIVE_ID = "exact_federation_L6_canonical_20260909"
SIMULATOR_REPOSITORY = "https://github.com/muellerberndt/oph-physics-sim"
LANE_RECEIPT_PATH = F.RECEIPT_PATH
BUILDER_PATH = Path(__file__).resolve()
KERNEL_SAMPLE_DEFAULT = 64
FLOAT_SWEEP_BUDGET_DEFAULT = 256
LAMBDA2_TIMEOUT_DEFAULT = 900.0
LAMBDA_MAX_TOL = 1e-6
LAMBDA_MAX_MAXITER = 60  # ARPACK restarts of ncv = 40, a bounded estimate of the nearly degenerate top of the spectrum
SINGLE_THREAD_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
KERNEL_SAMPLE_SEED_BASE = F.LOAD_SEED_BASE  # stream "kernel_sample": base + 17 * level
CONTROL_FILES = ("README.md", "archive_manifest.json", "verify_archive.py")
PORTS = F.PORTS

CONFIG_JSON = "config.json"
PRIMITIVES_NPZ = "primitives.npz"
INTEGER_JSON = {"port_pair": "integer_law_port_pair.json", "isolated": "integer_law_isolated.json"}
INTEGER_NPZ = {"port_pair": "integer_terminal_port_pair.npz", "isolated": "integer_terminal_isolated.npz"}
FLOAT_JSON = "float_law_port_pair.json"
FLOAT_NPZ = "float_terminal_port_pair.npz"
FLOAT_ISOLATED_JSON = "float_law_isolated.json"
SPECTRUM_JSON = "spectrum_port_pair.json"
FIEDLER_NPZ = "fiedler_port_pair.npz"
KERNEL_JSON = "kernel_readout.json"
KERNEL_NPZ = "kernel_matrices.npz"

canonical_json = F.canonical_json
sha256_of = F.sha256_of
sha256_file = F.sha256_file


def array_sha256(values: np.ndarray, dtype: str) -> str:
    """Digest of the C-order little-endian bytes of ``values`` in ``dtype``."""

    arr = np.ascontiguousarray(np.asarray(values), dtype=np.dtype(dtype).newbyteorder("<"))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _round(value: float, digits: int = 12) -> float:
    out = float(round(float(value), digits))
    return 0.0 if out == 0.0 else out


def _sig(value: float, digits: int = 12) -> float:
    return F._sig(value, digits)


def write_json(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value), encoding="ascii")


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="ascii"))


# --------------------------------------------------------------------------
# Geometry and primitives
# --------------------------------------------------------------------------


def vertex_cells(level: int) -> np.ndarray:
    """Cells touching each of the twelve pentagonal vertices, shape (12, 5)."""

    faces = np.asarray(build_geodesic_icosahedral_tower(level).levels[level].faces)
    rows = []
    for v in range(12):
        cells = np.flatnonzero(np.any(faces == v, axis=1))
        if cells.size != 5:
            raise AssertionError(f"vertex {v} touches {cells.size} cells")
        rows.append(cells)
    return np.asarray(rows, dtype=np.int32)


def kernel_sample_cells(level: int, carriers: int, vertex_table: np.ndarray, count: int) -> list[int]:
    """One cell per pentagonal vertex (lowest index), then a seeded draw of the rest."""

    count = min(count, carriers)
    chosen = sorted({int(row.min()) for row in vertex_table})[:count]
    rng = np.random.default_rng(KERNEL_SAMPLE_SEED_BASE + 17 * level)
    pool = np.setdiff1d(np.arange(carriers, dtype=np.int64), np.asarray(chosen, dtype=np.int64))
    extra = rng.choice(pool, size=count - len(chosen), replace=False) if count > len(chosen) else np.zeros(0, dtype=np.int64)
    return sorted(set(chosen) | {int(c) for c in extra})


def component_expectation(labels: np.ndarray, loads: np.ndarray) -> dict[str, np.ndarray]:
    """Sizes, totals and the balanced-class split ``(q, r)`` per component."""

    count = int(labels.max()) + 1
    size = np.bincount(labels, minlength=count).astype(np.int64)
    total = np.bincount(labels, weights=np.asarray(loads, dtype=float), minlength=count)
    total = np.rint(total).astype(np.int64)
    q = total // size
    r = total - q * size
    return {"size": size, "total": total, "q": q, "r": r}


def expected_multiset_hash(size: np.ndarray, q: np.ndarray, r: np.ndarray) -> str:
    entries = []
    for m, qq, rr in zip(size.tolist(), q.tolist(), r.tolist()):
        multiset = []
        if m - rr:
            multiset.append([qq, m - rr])
        if rr:
            multiset.append([qq + 1, rr])
        entries.append([m, multiset])
    return sha256_of({"canonicalizer": "component_multiset", "components": entries})


def balanced_minimum(size: np.ndarray, q: np.ndarray, r: np.ndarray) -> int:
    return int(np.sum((size - r) * q * q + r * (q + 1) * (q + 1)))


def build_primitives(level: int) -> dict[str, Any]:
    """Both federations, the loads and every primitive array of the archive."""

    fed = F.build_federation(level, "port_pair")
    iso = F.build_federation(level, "isolated")
    carriers = fed.carriers
    loads = F.initial_loads(level, fed.ports)
    template = np.asarray(carrier.seams(), dtype=np.int64)
    intra_carrier = np.repeat(np.arange(carriers, dtype=np.int64), carrier.SEAM_COUNT)
    intra_port_a = np.tile(template[:, 0], carriers)
    intra_port_b = np.tile(template[:, 1], carriers)
    if not np.array_equal(fed.seam_a[: fed.intra_count], intra_carrier * PORTS + intra_port_a):
        raise AssertionError("intra seam order differs from the carrier-major template")
    if not np.array_equal(fed.seam_b[: fed.intra_count], intra_carrier * PORTS + intra_port_b):
        raise AssertionError("intra seam order differs from the carrier-major template")
    inter = fed.inter_pairs.astype(np.int64)
    if not np.array_equal(fed.seam_a[fed.intra_count :], inter[:, 0] * PORTS + inter[:, 1]):
        raise AssertionError("inter seam order differs from the dual-edge order")
    if not np.array_equal(fed.seam_b[fed.intra_count :], inter[:, 2] * PORTS + inter[:, 3]):
        raise AssertionError("inter seam order differs from the dual-edge order")
    if not np.array_equal(iso.seam_a, fed.seam_a[: fed.intra_count]) or not np.array_equal(iso.seam_b, fed.seam_b[: fed.intra_count]):
        raise AssertionError("isolated seams differ from the intra seams of the glued federation")
    iso_labels = np.arange(fed.ports, dtype=np.int64) // PORTS
    if not np.array_equal(iso.component_of_port, iso_labels):
        raise AssertionError("isolated components differ from carriers")
    glued = component_expectation(fed.component_of_port, loads)
    isolated = component_expectation(iso_labels, loads)
    if expected_multiset_hash(glued["size"], glued["q"], glued["r"]) != fed.expected_integer_quotient_hash(loads):
        raise AssertionError("expected multiset hash differs from federation.py")
    if expected_multiset_hash(isolated["size"], isolated["q"], isolated["r"]) != iso.expected_integer_quotient_hash(loads):
        raise AssertionError("expected isolated multiset hash differs from federation.py")
    if balanced_minimum(glued["size"], glued["q"], glued["r"]) != fed.integer_minimum_descent(loads):
        raise AssertionError("balanced minimum differs from federation.py")
    vertices = vertex_cells(level)
    arrays = {
        "carrier_seam_template": template.astype(np.int8),
        "intra_seam_carrier": intra_carrier.astype(np.int32),
        "intra_seam_port_a": intra_port_a.astype(np.uint8),
        "intra_seam_port_b": intra_port_b.astype(np.uint8),
        "inter_seam": inter.astype(np.int32),
        "initial_loads": loads.reshape(carriers, PORTS).astype(np.int8),
        "component_of_port": fed.component_of_port.astype(np.int32),
        "component_size": glued["size"],
        "component_total": glued["total"],
        "component_q": glued["q"],
        "component_r": glued["r"],
        "isolated_component_total": isolated["total"],
        "isolated_component_q": isolated["q"],
        "isolated_component_r": isolated["r"],
        "pentagonal_cell": fed.pentagonal_cell.astype(bool),
        "vertex_cells": vertices,
    }
    return {"fed": fed, "iso": iso, "loads": loads, "arrays": arrays, "glued": glued, "isolated": isolated, "vertex_cells": vertices}


ARRAY_DTYPES = {
    "carrier_seam_template": "int8",
    "intra_seam_carrier": "int32",
    "intra_seam_port_a": "uint8",
    "intra_seam_port_b": "uint8",
    "inter_seam": "int32",
    "initial_loads": "int8",
    "component_of_port": "int32",
    "component_size": "int64",
    "component_total": "int64",
    "component_q": "int64",
    "component_r": "int64",
    "isolated_component_total": "int64",
    "isolated_component_q": "int64",
    "isolated_component_r": "int64",
    "pentagonal_cell": "bool",
    "vertex_cells": "int32",
}


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    """Write a compressed npz and return per-array dtype, shape and digest."""

    np.savez_compressed(path, **arrays)
    return {
        name: {"dtype": str(np.asarray(v).dtype), "shape": list(np.asarray(v).shape), "sha256": array_sha256(v, str(np.asarray(v).dtype))}
        for name, v in arrays.items()
    }


def seam_arrays_from_primitives(prim: dict[str, np.ndarray], gluing: str) -> tuple[np.ndarray, np.ndarray, int]:
    """``(seam_a, seam_b, ports)`` in archive order from the primitive arrays."""

    carriers = int(prim["initial_loads"].shape[0])
    ports = carriers * PORTS
    a = prim["intra_seam_carrier"].astype(np.int64) * PORTS + prim["intra_seam_port_a"].astype(np.int64)
    b = prim["intra_seam_carrier"].astype(np.int64) * PORTS + prim["intra_seam_port_b"].astype(np.int64)
    if gluing == "port_pair":
        inter = prim["inter_seam"].astype(np.int64)
        a = np.concatenate([a, inter[:, 0] * PORTS + inter[:, 1]])
        b = np.concatenate([b, inter[:, 2] * PORTS + inter[:, 3]])
    return np.ascontiguousarray(a), np.ascontiguousarray(b), ports


def load_primitives(archive: Path) -> dict[str, np.ndarray]:
    with np.load(archive / PRIMITIVES_NPZ, allow_pickle=False) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def laplacian_from_seams(a: np.ndarray, b: np.ndarray, ports: int) -> sparse.csr_matrix:
    """Identical construction to ``federation.build_federation``."""

    adjacency = sparse.coo_matrix((np.ones(a.size, dtype=float), (a, b)), shape=(ports, ports))
    adjacency = (adjacency + adjacency.T).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return (sparse.diags(degree) - adjacency).tocsr()


def mismatch_potential(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    d = x[a] - x[b]
    return float(np.dot(d, d))


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def build_config(level: int, prim_bundle: dict[str, Any], *, archive_id: str, schedules: int, float_sweeps: int,
                 kernel_cells: Sequence[int], engine: str, workers: int, lambda2_timeout: float) -> dict[str, Any]:
    fed = prim_bundle["fed"]
    iso = prim_bundle["iso"]
    gluing_receipt = {k: v for k, v in fed.gluing_receipt.items() if k != "port_pairs"}
    seeds = {g: [F.schedule_seed(level, g, s) for s in range(schedules)] for g in F.GLUINGS}
    return {
        "schema": ARCHIVE_SCHEMA,
        "archive_id": archive_id,
        "level": level,
        "carriers": fed.carriers,
        "ports_per_carrier": PORTS,
        "ports": fed.ports,
        "seams": {"intra": fed.intra_count, "inter": fed.inter_count, "total": fed.seams, "isolated_total": iso.seams,
                  "order": "intra seams carrier-major in carrier.seams() order (seam s: carrier s // 30, template s % 30), then inter seams in dual-edge order"},
        "carrier_seam_template": [list(map(int, s)) for s in carrier.seams()],
        "carrier_antipode": list(carrier.antipode()),
        "components": {"port_pair": fed.components, "isolated": iso.components},
        "pentagonal_cells": int(np.count_nonzero(fed.pentagonal_cell)),
        "gluing": {
            "production": gluing_receipt,
            "isolated": {k: v for k, v in iso.gluing_receipt.items()},
            "mean_denominator": {"port_pair": str(fed.mean_denominator), "isolated": str(iso.mean_denominator)},
        },
        "laws": {
            "mean_law": "E_e = I - b_e b_e^T / 2 on seam e: both endpoints replaced by their mean 0.5 * (x_i + x_j)",
            "descent_functional": "V(x) = sum_p x_p^2; the seam mean is an orthogonal projection, so V drops by exactly (x_i - x_j)^2 / 2 per non-wait move",
            "termination_potential": "Phi(x) = sum_seams (x_i - x_j)^2; Phi = 0 is consensus; Phi is not monotone along single moves",
            "synchronous_operator": "T_fed = I - L_fed / D with D = 2|S|/N (one attempt per carrier per tick); isolated blocks equal I - L_ico/60",
            "integer_law": (
                "carrier.integer_nearest_agreement: with s = x_a + x_b, lo = floor(s/2), hi = s - lo; the first endpoint a of seam s "
                "(seam_a) receives hi when the tie coin is 1 and lo when it is 0, the second endpoint b receives s minus that; "
                "d = x_a - x_b: d = 0 is a wait; |d| = 1 is a wait when the placement equals the current state and a swap otherwise "
                "(V unchanged); |d| >= 2 is a descent"
            ),
            "integer_descent": (
                "a descent with mismatch |d| >= 2 is the composition of floor(|d|/2) conservative unit transfers with mismatches "
                "|d|, |d| - 2, ...; each lowers V by exactly 2(d_k - 1), so the move lowers V by (d^2 - (d mod 2)) / 2"
            ),
            "integer_terminal_class": "balanced class: every reading in {q, q+1} per component with q = floor(total/size) and (total mod size) copies of q+1; V = V_min",
            "integer_canonicalizer": "multiset of readings per component, components ordered by lowest port; sha256 of canonical JSON {canonicalizer: component_multiset, components: [[size, [[value, count], ...]], ...]}",
            "mean_canonicalizer": "component-lattice snap q_p = round(m_c x_p); sha256 of canonical JSON {canonicalizer: component_lattice_snap, component_sizes, component_of_port, q}",
            "float_termination": f"first sweep end with Phi < {F.FLOAT_PHI_THRESHOLD}",
            "unique_fixed_point": (
                "the component mean is the unique fixed point of every schedule of the mean law: each seam move fixes the component "
                "totals (conservation) and the terminal set is the seam equalizer, which by linearity is the span of the component "
                "indicators; verified exactly in rational arithmetic at levels 0 and 1 in the lane receipt pinned by the manifest"
            ),
        },
        "seeds": {
            "loads": {"stream": "loads", "rule": f"numpy.random.default_rng({F.LOAD_SEED_BASE} + level).integers(0, {F.LOAD_MAX + 1}, size=ports)", "seed": F.LOAD_SEED_BASE + level, "load_max": F.LOAD_MAX},
            "schedules": {
                "rule": f"{F.SCHEDULE_SEED_BASE} + 1000 * level + 100 * gluing_index + schedule_index with gluing_index isolated = 0, port_pair = 1",
                "port_pair": seeds["port_pair"],
                "isolated": seeds["isolated"],
                "float_port_pair": [seeds["port_pair"][0]],
                "float_isolated": seeds["isolated"],
            },
            "kernel_sample": {"stream": "kernel_sample", "rule": f"one cell per pentagonal vertex (lowest cell index), then default_rng({KERNEL_SAMPLE_SEED_BASE} + 17 * level).choice(remaining cells, size=count - 12, replace=False)", "seed": KERNEL_SAMPLE_SEED_BASE + 17 * level, "cells": [int(c) for c in kernel_cells]},
        },
        "schedule_count": schedules,
        "budgets": {
            "integer_max_sweeps": F.MAX_SWEEPS_DEFAULT,
            "float_port_pair_sweeps": float_sweeps,
            "float_isolated_max_sweeps": F.MAX_SWEEPS_DEFAULT,
            "lambda2_timeout_seconds": lambda2_timeout,
            "kernel_steps": list(F.KERNEL_STEPS),
            "kernel_sample_count": len(kernel_cells),
            "workers": workers,
        },
        "rng": {
            "generator": "numpy.random.Generator",
            "bit_generator": "PCG64",
            "construction": "numpy.random.default_rng(seed)",
            "numpy_version": np.__version__,
            "integer_law_draw": "per sweep: seq = integers(0, |S|, size=|S|, dtype=int64), then coin = integers(0, 2, size=|S|, dtype=int64)",
            "mean_law_draw": "per sweep: seq = integers(0, |S|, size=|S|, dtype=int64)",
            "sweep": "one sweep is |S| attempts applied in draw order",
        },
        "byte_model": {
            "arrays": "numpy .npz (zip, deflate); little-endian, C order; dtypes declared per array in the JSON files",
            "array_digest": "sha256 of the C-order little-endian bytes of the array in its declared dtype",
            "json_digest": "sha256 of canonical JSON: sorted keys, separators (',', ':'), ascii, one trailing newline",
            "file_digest": "sha256 of the file bytes",
            "floats": "IEEE-754 binary64; the native kernel is compiled with -ffp-contract=off and applies the same operations as the reference engines",
            "float_rounding_in_json": "12 significant digits for Phi, V and kernel data",
        },
        "engine": {"name": engine, "native_kernel_source_sha256": hashlib.sha256(F._KERNEL_SOURCE.encode("utf-8")).hexdigest(), "native_error": F._NATIVE["error"]},
        "platform": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(), "machine": platform.machine()},
    }


# --------------------------------------------------------------------------
# Integer law to termination, with the per-sweep V ledger
# --------------------------------------------------------------------------


def integer_schedule(graph: F.PortGraph, seams: int, loads: np.ndarray, seed: int, v_min: int, *, engine: str,
                     max_sweeps: int = F.MAX_SWEEPS_DEFAULT, draw_hashes: bool = False) -> dict[str, Any]:
    move = F.INTEGER_ENGINES[engine]
    x = np.asarray(loads, dtype=np.int64).copy()
    rng = np.random.default_rng(seed)
    v = int(np.dot(x, x))
    v0 = v
    tally = F.IntegerTally()
    ledger = [v]
    per_sweep = []
    hashes = []
    sweep = 0
    first_attempt = 0 if v == v_min else -1
    started = time.perf_counter()
    while first_attempt < 0 and sweep < max_sweeps:
        seq = rng.integers(0, seams, size=seams, dtype=np.int64)
        coin = rng.integers(0, 2, size=seams, dtype=np.int64)
        if draw_hashes:
            hashes.append([array_sha256(seq, "int64"), array_sha256(coin, "int64")])
        before = (tally.descents, tally.swaps, tally.waits, tally.unit_transfers, tally.decrement_violations)
        v, first = move(x, graph, seq, coin, v, v_min, tally)
        if first >= 0:
            first_attempt = sweep * seams + first + 1
        sweep += 1
        direct = int(np.dot(x, x))
        if direct != v:
            raise AssertionError("integer ledger differs from the state")
        after = (tally.descents, tally.swaps, tally.waits, tally.unit_transfers, tally.decrement_violations)
        ledger.append(v)
        per_sweep.append({"sweep": sweep, "V": v, "descents": after[0] - before[0], "swaps": after[1] - before[1],
                          "waits": after[2] - before[2], "unit_transfers": after[3] - before[3], "violations": after[4] - before[4]})
    d = np.abs(x[graph.a] - x[graph.b])
    result = {
        "seed": int(seed),
        "terminated": bool(first_attempt >= 0),
        "sweeps": sweep,
        "attempts": sweep * seams,
        "attempts_to_balanced_class": int(first_attempt),
        "descents": tally.descents,
        "swaps": tally.swaps,
        "waits": tally.waits,
        "unit_transfers": tally.unit_transfers,
        "unit_transfer_decrement_identity_violations": tally.decrement_violations,
        "strict_descent_violations": tally.decrement_violations,
        "V_initial": v0,
        "V_minimum": int(v_min),
        "V_terminal": v,
        "V_ledger": ledger,
        "per_sweep": per_sweep,
        "odd_tie_seams_at_termination": int(np.count_nonzero(d == 1)),
        "max_seam_difference_at_termination": int(d.max()) if d.size else 0,
        "seconds": round(time.perf_counter() - started, 3),
        "state": x,
    }
    if draw_hashes:
        result["draw_sha256_per_sweep"] = hashes
    return result


def integer_quotient_hash(labels: np.ndarray, x: np.ndarray) -> str:
    """The multiset of readings per component (components ordered by label)."""

    x = np.asarray(x, dtype=np.int64)
    order = np.argsort(labels, kind="stable")
    comps = labels[order]
    vals = x[order]
    starts = np.flatnonzero(np.r_[True, comps[1:] != comps[:-1]])
    stops = np.r_[starts[1:], comps.size]
    entries = []
    for start, stop in zip(starts, stops):
        values, counts = np.unique(vals[start:stop], return_counts=True)
        entries.append([int(stop - start), [[int(v), int(c)] for v, c in zip(values, counts)]])
    return sha256_of({"canonicalizer": "component_multiset", "components": entries})


def run_integer_law(archive: Path, gluing: str, fed: F.ExactFederation, loads: np.ndarray, labels: np.ndarray,
                    expectation: dict[str, np.ndarray], seeds: Sequence[int], *, engine: str, log) -> dict[str, Any]:
    v_min = balanced_minimum(expectation["size"], expectation["q"], expectation["r"])
    expected_hash = expected_multiset_hash(expectation["size"], expectation["q"], expectation["r"])
    entries = []
    states = {}
    for index, seed in enumerate(seeds):
        result = integer_schedule(fed.graph, fed.seams, loads, seed, v_min, engine=engine, draw_hashes=(index == 0))
        state = result.pop("state")
        if state.min() < np.iinfo(np.int8).min or state.max() > np.iinfo(np.int8).max:
            raise AssertionError("terminal loads do not fit int8")
        state8 = state.astype(np.int8)
        quotient = integer_quotient_hash(labels, state)
        if quotient != fed.integer_quotient_hash(state):
            raise AssertionError("archive quotient hash differs from federation.py")
        result["quotient_hash"] = quotient
        result["quotient_hash_equals_expected"] = bool(quotient == expected_hash)
        result["terminal_array"] = f"schedule_{index:02d}"
        result["terminal_sha256"] = array_sha256(state8, "int8")
        result["terminal_dtype"] = "int8"
        entries.append(result)
        states[f"schedule_{index:02d}"] = state8
        log(f"integer {gluing} schedule {index}: seed {seed}, {result['sweeps']} sweeps, {result['attempts_to_balanced_class']} attempts to the balanced class, {result['seconds']} s")
    hashes = sorted({e["quotient_hash"] for e in entries})
    if not (len(hashes) == 1 and hashes[0] == expected_hash):
        raise AssertionError(f"integer law {gluing}: quotient hashes {hashes} differ from the expected multiset hash")
    if not all(e["terminated"] for e in entries):
        raise AssertionError("an integer schedule did not terminate")
    save_npz(archive / INTEGER_NPZ[gluing], states)
    block = {
        "schema": ARCHIVE_SCHEMA,
        "law": "integer",
        "gluing": gluing,
        "seams": fed.seams,
        "components": fed.components,
        "engine": engine,
        "termination": "first attempt after which V equals the balanced-class minimum; the sweep in progress is completed",
        "expected_quotient_hash": expected_hash,
        "unique_quotient_hash_count": len(hashes),
        "quotient_hash_equals_expected_all": True,
        "V_initial": int(entries[0]["V_initial"]),
        "V_minimum": int(v_min),
        "terminal_arrays_file": INTEGER_NPZ[gluing],
        "sweeps": {"min": min(e["sweeps"] for e in entries), "max": max(e["sweeps"] for e in entries)},
        "attempts_to_balanced_class": {"min": min(e["attempts_to_balanced_class"] for e in entries), "max": max(e["attempts_to_balanced_class"] for e in entries)},
        "unit_transfers": {"min": min(e["unit_transfers"] for e in entries), "max": max(e["unit_transfers"] for e in entries)},
        "violations_total": int(sum(e["unit_transfer_decrement_identity_violations"] for e in entries)),
        "schedules": entries,
    }
    write_json(archive / INTEGER_JSON[gluing], block)
    return block


# --------------------------------------------------------------------------
# Mean law: one budgeted schedule under the production gluing (worker entry)
# --------------------------------------------------------------------------


def float_schedule_budgeted(a: np.ndarray, b: np.ndarray, ports: int, loads: np.ndarray, seed: int, sweeps: int,
                            labels: np.ndarray, *, engine: str) -> tuple[dict[str, Any], np.ndarray]:
    graph = F.PortGraph.build(a, b, ports)
    move = F.MEAN_ENGINES[engine]
    seams = int(a.size)
    x = np.asarray(loads, dtype=np.float64).copy()
    rng = np.random.default_rng(seed)
    phi0 = mismatch_potential(x, a, b)
    v0 = float(np.dot(x, x))
    tally = F.MeanTally()
    per_sweep = []
    v_prev = v0
    ledger_prev = 0.0
    counts_prev = (0, 0, 0)
    ledger_ok = True
    started = time.perf_counter()
    first_raise_sweep = None
    for sweep in range(1, sweeps + 1):
        seq = rng.integers(0, seams, size=seams, dtype=np.int64)
        before = tally.first_raise is None
        move(x, graph, seq, tally)
        if before and tally.first_raise is not None:
            first_raise_sweep = sweep - 1
        v = float(np.dot(x, x))
        phi = mismatch_potential(x, a, b)
        increment = tally.ledger - ledger_prev
        ledger_prev = tally.ledger
        counts = (tally.waits, tally.descent_violations, tally.phi_raising_moves)
        rel = abs((v_prev - v) - increment) / max(v0, 1.0)
        ledger_ok = ledger_ok and rel < 1e-9
        per_sweep.append({"sweep": sweep, "V": _sig(v), "Phi": _sig(phi), "ledger_increment": _sig(increment),
                          "ledger_relative_error": _sig(rel, 3), "waits": counts[0] - counts_prev[0],
                          "violations": counts[1] - counts_prev[1], "phi_raising_moves": counts[2] - counts_prev[2]})
        counts_prev = counts
        v_prev = v
    seconds = time.perf_counter() - started
    means = np.bincount(labels, weights=x, minlength=int(labels.max()) + 1) / np.bincount(labels, minlength=int(labels.max()) + 1)
    residual = x - means[labels]
    attempts = sweeps * seams
    counterexample = None
    if tally.first_raise is not None:
        fr = tally.first_raise
        counterexample = {"attempt": int(first_raise_sweep * seams + fr["index"] + 1), "seam": int(fr["seam"]), "ports": [int(a[fr["seam"]]), int(b[fr["seam"]])],
                          "x_i": _sig(fr["x_i"]), "x_j": _sig(fr["x_j"]), "neighbour_sum_i": _sig(fr["N_i"]), "neighbour_sum_j": _sig(fr["N_j"]),
                          "deg_i": int(fr["deg_i"]), "deg_j": int(fr["deg_j"]), "delta_phi": _sig(fr["delta_phi"])}
    v_min = float(np.sum(np.bincount(labels, weights=np.asarray(loads, dtype=float)) ** 2 / np.bincount(labels)))
    block = {
        "seed": int(seed),
        "engine": engine,
        "terminated": bool(phi < F.FLOAT_PHI_THRESHOLD),
        "sweeps": sweeps,
        "attempts": attempts,
        "waits": tally.waits,
        "non_wait_moves": attempts - tally.waits,
        "Phi_initial": _sig(phi0),
        "Phi_terminal": _sig(phi),
        "V_initial": _sig(v0),
        "V_terminal": _sig(v_prev),
        "V_minimum": _sig(v_min),
        "centered_norm_initial": _sig(v0 - v_min),
        "centered_norm_terminal": _sig(v_prev - v_min),
        "descent_ledger_total": _sig(tally.ledger),
        "descent_ledger_relative_error_below_1e-9_every_sweep": bool(ledger_ok),
        "strict_descent_violations": tally.descent_violations,
        "phi_raising_moves": tally.phi_raising_moves,
        "phi_raising_fraction_of_non_wait_moves": _sig(tally.phi_raising_moves / max(1, attempts - tally.waits), 4),
        "phi_raising_counterexample": counterexample,
        "max_abs_deviation_from_component_mean": _sig(float(np.max(np.abs(residual))), 6),
        "total_initial": int(np.sum(loads)),
        "total_terminal": _sig(float(np.sum(x))),
        "seconds": round(seconds, 3),
        "attempts_per_second": _sig(attempts / max(seconds, 1e-9), 4),
        "per_sweep": per_sweep,
    }
    return block, x


def float_worker_main(archive: Path, seed: int, sweeps: int, engine: str) -> None:
    prim = load_primitives(archive)
    a, b, ports = seam_arrays_from_primitives(prim, "port_pair")
    loads = prim["initial_loads"].astype(np.int64).ravel()
    labels = prim["component_of_port"].astype(np.int64)
    engine = F.resolve_engine(engine, ports)
    block, x = float_schedule_budgeted(a, b, ports, loads, seed, sweeps, labels, engine=engine)
    digests = save_npz(archive / FLOAT_NPZ, {"terminal_state": x})
    block.update({
        "schema": ARCHIVE_SCHEMA,
        "law": "mean_law_float",
        "gluing": "port_pair",
        "seams": int(a.size),
        "budget": "declared sweep budget; the schedule is budgeted, its termination at this rung is work in progress",
        "terminal_state_file": FLOAT_NPZ,
        "terminal_state_array": "terminal_state",
        "terminal_state_dtype": "float64",
        "terminal_state_sha256": digests["terminal_state"]["sha256"],
        "threshold": F.FLOAT_PHI_THRESHOLD,
    })
    write_json(archive / FLOAT_JSON, block)


def run_float_isolated(archive: Path, iso: F.ExactFederation, loads: np.ndarray, seeds: Sequence[int], *, engine: str, log) -> dict[str, Any]:
    """Sixteen terminated float schedules of the isolated control (lane function)."""

    expected = iso.expected_terminal_quotient_hash(loads)
    entries = []
    for index, seed in enumerate(seeds):
        started = time.perf_counter()
        result = F._strip(F.run_mean_law(iso, loads, seed, engine=engine))
        result["seconds"] = round(time.perf_counter() - started, 3)
        entries.append(result)
        log(f"float isolated schedule {index}: seed {seed}, {result['sweeps']} sweeps, terminated {result['terminated']}, {result['seconds']} s")
    hashes = sorted({e["terminal_quotient_hash"] for e in entries})
    if not (len(hashes) == 1 and hashes[0] == expected and all(e["terminated"] for e in entries)):
        raise AssertionError("isolated float control did not reach the expected terminal hash on every schedule")
    block = {
        "schema": ARCHIVE_SCHEMA,
        "law": "mean_law_float",
        "gluing": "isolated",
        "seams": iso.seams,
        "components": iso.components,
        "engine": engine,
        "threshold": F.FLOAT_PHI_THRESHOLD,
        "termination": "first sweep end with Phi < threshold",
        "expected_terminal_quotient_hash": expected,
        "unique_terminal_hash_count": len(hashes),
        "terminal_hash_equals_expected_all": True,
        "all_terminated": True,
        "strict_descent_violations_total": int(sum(e["strict_descent_violations"] for e in entries)),
        "descent_ledger_ok_all": bool(all(e["descent_ledger_relative_error_below_1e-9"] for e in entries)),
        "component_mean_within_1e-9_all": bool(all(e["component_mean_within_1e-9"] for e in entries)),
        "sweeps": {"min": min(e["sweeps"] for e in entries), "max": max(e["sweeps"] for e in entries)},
        "schedules": entries,
    }
    write_json(archive / FLOAT_ISOLATED_JSON, block)
    return block


# --------------------------------------------------------------------------
# Spectrum of the glued port graph
# --------------------------------------------------------------------------


def spectrum(a: np.ndarray, b: np.ndarray, ports: int, components: int) -> tuple[dict[str, Any], np.ndarray]:
    lap = laplacian_from_seams(a, b, ports)
    started = time.perf_counter()
    if ports <= 4096:
        values, vectors = np.linalg.eigh(lap.toarray())
        order = np.argsort(values)
        lam2 = float(values[order[components]]) if components < ports else 0.0
        vec = vectors[:, order[components]]
        lam_max = float(values[order[-1]])
        method = "dense_eigh"
        lam_max_note = "dense_eigh"
    else:
        sigma = -1e-3
        shifted = (lap - sigma * sparse.identity(ports, format="csr")).tocsc()
        lu = sparse_linalg.splu(shifted, permc_spec="MMD_AT_PLUS_A")
        op = sparse_linalg.LinearOperator(lap.shape, matvec=lu.solve, dtype=np.float64)
        values, vectors = sparse_linalg.eigsh(lap, k=components + 1, sigma=sigma, which="LM", OPinv=op, return_eigenvectors=True)
        order = np.argsort(values)
        lam2 = float(values[order[-1]])
        vec = vectors[:, order[-1]]
        try:
            lam_max = float(sparse_linalg.eigsh(lap, k=1, which="LA", tol=LAMBDA_MAX_TOL, ncv=40, maxiter=LAMBDA_MAX_MAXITER, return_eigenvectors=False)[0])
            lam_max_note = f"ARPACK eigsh k=1 which=LA tol={LAMBDA_MAX_TOL} (converged)"
        except sparse_linalg.ArpackNoConvergence as error:
            lam_max = float(error.eigenvalues[-1]) if len(error.eigenvalues) else float("nan")
            lam_max_note = f"ARPACK eigsh k=1 which=LA tol={LAMBDA_MAX_TOL} (maxiter {LAMBDA_MAX_MAXITER} restarts reached; partial estimate, a lower bound of lambda_max)"
        method = f"sparse_shift_invert(sigma=-1e-3, SuperLU permc_spec=MMD_AT_PLUS_A, ARPACK eigsh k=components+1 which=LM) for lambda_2; {lam_max_note} for lambda_max"
    seconds = time.perf_counter() - started
    vec = np.asarray(vec, dtype=np.float64)
    vec = vec / np.linalg.norm(vec)
    residual = float(np.linalg.norm(lap @ vec - lam2 * vec))
    rayleigh = float(vec @ (lap @ vec))
    denominator = 2 * a.size / (ports // PORTS)
    block = {
        "definition": "L_fed = D - A of the port graph (intra and inter seams); T_fed = I - L_fed/D with D = 2|S|/N",
        "method": method,
        "laplacian_lambda_2": _sig(lam2),
        "laplacian_lambda_max": _sig(lam_max),
        "lambda_max_method": lam_max_note,
        "D": _sig(denominator),
        "t_fed_second_eigenvalue": _sig(1.0 - lam2 / denominator),
        "t_fed_smallest_eigenvalue": _sig(1.0 - lam_max / denominator),
        "attempts_per_e_fold_of_slowest_mode": _sig(2 * a.size / lam2, 6) if lam2 > 0 else None,
        "eigenvector_file": FIEDLER_NPZ,
        "eigenvector_array": "lambda_2_eigenvector",
        "eigenvector_residual_norm": _sig(residual, 3),
        "eigenvector_rayleigh_quotient": _sig(rayleigh),
        "eigenvector_mean": _sig(float(vec.mean()), 3),
        "isolated_lambda_2_exact": "5 - sqrt(5) (single-carrier seam Laplacian)",
        "isolated_lambda_2": _sig(5.0 - 5.0**0.5),
        "seconds": round(seconds, 3),
    }
    return block, vec


# --------------------------------------------------------------------------
# Per-carrier response kernels
# --------------------------------------------------------------------------


class _KernelHost:
    """Duck-typed stand-in for ``ExactFederation`` inside ``federation.response_kernels``."""

    def __init__(self, laplacian: sparse.csr_matrix, denominator: float) -> None:
        self.laplacian = laplacian
        self.denominator = float(denominator)
        self.ports = int(laplacian.shape[0])

    def apply_synchronous(self, y: np.ndarray) -> np.ndarray:
        return y - (self.laplacian @ y) / self.denominator


def kernel_worker(archive: str, cells: Sequence[int]) -> list[tuple[int, np.ndarray]]:
    """Glued kernels ``K_n`` (stacked over ``KERNEL_STEPS``) for a chunk of cells."""

    prim = load_primitives(Path(archive))
    a, b, ports = seam_arrays_from_primitives(prim, "port_pair")
    host = _KernelHost(laplacian_from_seams(a, b, ports), 2 * a.size / (ports // PORTS))
    out = []
    for cell in cells:
        kernels = F.response_kernels(host, int(cell))
        out.append((int(cell), np.stack([kernels[n] for n in F.KERNEL_STEPS])))
    return out


def isolated_kernels(prim: dict[str, np.ndarray], cell: int) -> np.ndarray:
    """Isolated readout of one carrier from its thirty archived intra seams."""

    rows = slice(cell * carrier.SEAM_COUNT, (cell + 1) * carrier.SEAM_COUNT)
    if not np.all(prim["intra_seam_carrier"][rows] == cell):
        raise AssertionError("intra seam rows are not carrier-major")
    a = prim["intra_seam_port_a"][rows].astype(np.int64)
    b = prim["intra_seam_port_b"][rows].astype(np.int64)
    host = _KernelHost(laplacian_from_seams(a, b, PORTS), 2 * a.size / 1)
    kernels = F.response_kernels(host, 0)
    return np.stack([kernels[n] for n in F.KERNEL_STEPS])


def kernel_rows(stack: np.ndarray, p_slow: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for n, k in zip(F.KERNEL_STEPS, stack):
        eig = np.sort(np.linalg.eigvalsh(k))[::-1]
        rows.append({
            "n": int(n),
            "eigenvalues": [_sig(v) for v in eig],
            "slow_band_share": _sig(float(np.trace(p_slow @ k @ p_slow) / np.trace(k))),
            "rank_1e-9": int(np.linalg.matrix_rank(k, tol=1e-9)),
        })
    return rows


def assemble_kernels(archive: Path, prim: dict[str, np.ndarray], cells: Sequence[int], glued: dict[int, np.ndarray]) -> dict[str, Any]:
    p_slow = carrier.slow_band_projector()
    gram = carrier.intrinsic_gram()
    vertices_of_cell: dict[int, list[int]] = {}
    for v, row in enumerate(prim["vertex_cells"].tolist()):
        for c in row:
            vertices_of_cell.setdefault(int(c), []).append(int(v))
    per_cell = []
    glued_stack = []
    iso_stack = []
    iso_dev = 0.0
    iso_gram_dev = 0.0
    shares = {n: [] for n in F.KERNEL_STEPS}
    tops = {n: [] for n in F.KERNEL_STEPS}
    for cell in cells:
        g = glued[int(cell)]
        iso = isolated_kernels(prim, int(cell))
        for n, k in zip(F.KERNEL_STEPS, iso):
            iso_dev = max(iso_dev, float(np.max(np.abs(k - carrier.normalized_response_kernel(n)))))
        iso_gram_dev = max(iso_gram_dev, float(np.max(np.abs(iso[-1] - gram))))
        g_rows = kernel_rows(g, p_slow)
        for row in g_rows:
            shares[row["n"]].append(row["slow_band_share"])
            tops[row["n"]].append(row["eigenvalues"][:4])
        per_cell.append({"cell": int(cell), "pentagonal": bool(prim["pentagonal_cell"][int(cell)]), "vertices": vertices_of_cell.get(int(cell), []),
                         "glued": g_rows, "isolated": kernel_rows(iso, p_slow)})
        glued_stack.append(g)
        iso_stack.append(iso)
    if iso_dev >= 1e-12 or iso_gram_dev >= 1e-12:
        raise AssertionError(f"isolated kernels deviate from the exact carrier kernel: {iso_dev}, {iso_gram_dev}")
    digests = save_npz(archive / KERNEL_NPZ, {"cells": np.asarray(cells, dtype=np.int32), "steps": np.asarray(F.KERNEL_STEPS, dtype=np.int32),
                                                "glued": np.stack(glued_stack), "isolated": np.stack(iso_stack)})
    summary = {}
    for n in F.KERNEL_STEPS:
        s = np.asarray(shares[n])
        t = np.asarray(tops[n])
        summary[str(n)] = {"slow_band_share": {"min": _sig(s.min(), 9), "median": _sig(float(np.median(s)), 9), "max": _sig(s.max(), 9)},
                           "top_four_eigenvalues_median": [_sig(v, 9) for v in np.median(t, axis=0)],
                           "top_four_eigenvalues_min": [_sig(v, 9) for v in t.min(axis=0)],
                           "top_four_eigenvalues_max": [_sig(v, 9) for v in t.max(axis=0)]}
    block = {
        "schema": ARCHIVE_SCHEMA,
        "definition": (
            "probe port p of carrier c with a unit impulse, apply T_fed 2n times, read carrier c: column p of R_n; "
            "C_n = Q R_n Q with Q = I - J/12; K_n = 12 C_n / tr C_n; computed by propagating the twelve centered impulses "
            "with a common rescaling per step (federation.response_kernels)"
        ),
        "steps": list(F.KERNEL_STEPS),
        "cells": [int(c) for c in cells],
        "pentagonal_cells_in_sample": int(sum(bool(prim["pentagonal_cell"][int(c)]) for c in cells)),
        "vertices_covered": sorted({v for c in cells for v in vertices_of_cell.get(int(c), [])}),
        "slow_band_projector": "P_slow: spectral projector of the carrier seam Laplacian onto the 5 - sqrt5 band (trace 3)",
        "slow_band_share_definition": "tr(P_slow K_n P_slow) / tr(K_n)",
        "isolated_readout": "the isolated federation is block diagonal, so the readout of carrier c is the propagation of its own 12 x 12 block I - L_ico/60 built from the archived intra seams of c",
        "isolated_max_abs_deviation_from_carrier_kernel": _sig(iso_dev, 3),
        "isolated_kernel_matches_carrier_within_1e-12": True,
        "isolated_max_abs_deviation_from_4_P_slow_at_n_300": _sig(iso_gram_dev, 3),
        "green_pattern_slow_share_reference": _sig(F.green_pattern_slow_share_reference(), 9),
        "matrices_file": KERNEL_NPZ,
        "matrices": digests,
        "summary_glued": summary,
        "per_cell": per_cell,
    }
    write_json(archive / KERNEL_JSON, block)
    return block


def _chunks(items: Sequence[int], count: int) -> list[list[int]]:
    count = max(1, min(count, len(items)))
    return [list(items[i::count]) for i in range(count)]


def spectrum_worker_main(archive: Path) -> None:
    prim = load_primitives(archive)
    a, b, ports = seam_arrays_from_primitives(prim, "port_pair")
    components = int(prim["component_size"].size)
    block, vec = spectrum(a, b, ports, components)
    digests = save_npz(archive / FIEDLER_NPZ, {"lambda_2_eigenvector": vec})
    block.update({"schema": ARCHIVE_SCHEMA, "gluing": "port_pair", "components": components, "eigenvector_sha256": digests["lambda_2_eigenvector"]["sha256"]})
    write_json(archive / SPECTRUM_JSON, block)


def _worker_command(kind: str, archive: Path, **extra: Any) -> list[str]:
    cmd = [sys.executable, "-m", "oph_exact.federation_archive", "--worker", kind, "--archive", str(archive)]
    for key, value in extra.items():
        cmd += [f"--{key.replace('_', '-')}", str(value)]
    return cmd


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def source_block() -> dict[str, Any]:
    return {
        "repository": SIMULATOR_REPOSITORY,
        "revision": "uncommitted-working-tree",
        "commit": None,
        "commit_note": "filled in by the maintainer after committing; the module digests below identify the working tree that built the archive",
        "modules": {"oph_exact/federation.py": sha256_file(F.PRODUCER_PATH), "oph_exact/carrier.py": sha256_file(F.CARRIER_PATH)},
        "builder": {"oph_exact/federation_archive.py": sha256_file(BUILDER_PATH)},
        "lane_receipt": {"path": str(LANE_RECEIPT_PATH.relative_to(F.REPO_ROOT)), "sha256": sha256_file(LANE_RECEIPT_PATH) if LANE_RECEIPT_PATH.exists() else None,
                         "reading": "L0..L3 receipt: exact single-move descent, exact conservation, exact invariance of the terminal quotient, exact dyadic termination at L0/L1"},
    }


def inventory(archive: Path) -> tuple[list[dict[str, Any]], str, int]:
    rows = []
    lines = []
    total = 0
    for path in sorted(p for p in archive.iterdir() if p.is_file() and p.name not in CONTROL_FILES):
        size = path.stat().st_size
        digest = sha256_file(path)
        rows.append({"path": path.name, "bytes": size, "sha256": digest})
        lines.append(f"{digest}  {size}  {path.name}\n")
        total += size
    return rows, hashlib.sha256("".join(lines).encode("utf-8")).hexdigest(), total


def write_manifest(archive: Path, config: dict[str, Any], blocks: dict[str, Any], timings: dict[str, float]) -> dict[str, Any]:
    rows, inventory_digest, total = inventory(archive)
    integer = blocks["integer_port_pair"]
    iso_int = blocks["integer_isolated"]
    flt = blocks.get("float_port_pair")
    flt_iso = blocks["float_isolated"]
    spec = blocks.get("spectrum")
    kern = blocks["kernel"]
    manifest = {
        "schema": ARCHIVE_SCHEMA,
        "archive_id": config["archive_id"],
        "source": source_block(),
        "curated_archive": {
            "file_count": len(rows),
            "total_bytes": total,
            "inventory_sha256": inventory_digest,
            "is_complete_copy_of_original_run": True,
            "selection": "configuration, primitive arrays, every terminal load vector, per-sweep ledgers, the float terminal state, the port-graph gap with its eigenvector, and the per-carrier kernel matrices",
        },
        "run_snapshot": {
            "level": config["level"],
            "carriers": config["carriers"],
            "ports": config["ports"],
            "seams": config["seams"],
            "components": config["components"],
            "gluing": config["gluing"]["production"]["mode"],
            "integer_law_port_pair": {k: integer[k] for k in ("expected_quotient_hash", "unique_quotient_hash_count", "quotient_hash_equals_expected_all", "sweeps", "attempts_to_balanced_class", "unit_transfers", "violations_total", "V_initial", "V_minimum")} | {"schedules": len(integer["schedules"])},
            "integer_law_isolated": {k: iso_int[k] for k in ("expected_quotient_hash", "unique_quotient_hash_count", "quotient_hash_equals_expected_all", "sweeps", "attempts_to_balanced_class", "violations_total", "V_initial", "V_minimum")} | {"schedules": len(iso_int["schedules"])},
            "mean_law_float_port_pair": ({k: flt[k] for k in ("seed", "sweeps", "attempts", "terminated", "Phi_initial", "Phi_terminal", "V_initial", "V_terminal", "V_minimum", "strict_descent_violations", "descent_ledger_relative_error_below_1e-9_every_sweep", "max_abs_deviation_from_component_mean", "terminal_state_sha256")} if flt else None),
            "mean_law_float_isolated": {k: flt_iso[k] for k in ("expected_terminal_quotient_hash", "unique_terminal_hash_count", "terminal_hash_equals_expected_all", "all_terminated", "sweeps", "strict_descent_violations_total")} | {"schedules": len(flt_iso["schedules"])},
            "spectrum_port_pair": ({k: spec[k] for k in ("laplacian_lambda_2", "laplacian_lambda_max", "method", "eigenvector_residual_norm", "eigenvector_sha256")} if spec else {"status": "skipped: shift-invert exceeded the declared timeout; work in progress"}),
            "response_kernel": {"cells": len(kern["cells"]), "pentagonal_cells_in_sample": kern["pentagonal_cells_in_sample"], "vertices_covered": kern["vertices_covered"],
                                "slow_band_share_median_glued": {n: kern["summary_glued"][n]["slow_band_share"]["median"] for n in kern["summary_glued"]},
                                "isolated_max_abs_deviation_from_carrier_kernel": kern["isolated_max_abs_deviation_from_carrier_kernel"],
                                "isolated_max_abs_deviation_from_4_P_slow_at_n_300": kern["isolated_max_abs_deviation_from_4_P_slow_at_n_300"]},
        },
        "exact_verification": {
            "standalone_verifier": "verify_archive.py",
            "verifier_sha256": sha256_file(archive / "verify_archive.py"),
            "imports_simulator": False,
            "checks": [
                "every inventory digest and byte count",
                "carrier seam template and the carrier-major intra seam arrays",
                "port-graph components rebuilt from the seam arrays by union-find",
                "expected terminal multisets and their hash recomputed from the initial loads",
                "every archived terminal vector: multiset, hash, seam differences at most one unit, V = V_min",
                "schedule 0 of the integer law replayed from its seed with numpy PCG64 and the declared move rule",
                "V ledger arithmetic: every unit transfer lowers V by 2(d - 1)",
                "float ledger arithmetic, terminal state V, Phi and conservation",
                "lambda_2 eigenvector residual and Rayleigh quotient from the seam arrays",
                "isolated kernel limit recomputed for two sample carriers by direct propagation with rescaling",
                "eigenvalues and slow-band shares recomputed from the archived kernel matrices",
            ],
        },
        "known_boundaries": {
            "gluing_is_declared_convention": True,
            "source_derived_gluing": False,
            "float_law_terminated_at_this_rung": bool(flt["terminated"]) if flt else False,
            "float_law_unique_fixed_point_by_linearity_and_conservation": True,
            "lambda_2_ordering_from_archived_shift_invert_computation": bool(spec is not None),
            "physical_identification": False,
            "physical_position_or_length": False,
            "refinement_limit": False,
            "continuum_limit": False,
            "field_attachment": False,
        },
        "timings_seconds": {k: round(v, 3) for k, v in timings.items()},
        "wall_clock_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inventory": rows,
    }
    write_json(archive / "archive_manifest.json", manifest)
    return manifest


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def build_archive(level: int, out: Path, *, archive_id: str | None = None, schedules: int = F.SCHEDULES,
                  float_sweeps: int = FLOAT_SWEEP_BUDGET_DEFAULT, kernel_count: int = KERNEL_SAMPLE_DEFAULT,
                  workers: int = 6, engine: str = "auto", lambda2_timeout: float = LAMBDA2_TIMEOUT_DEFAULT,
                  log=print) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    archive_id = archive_id or out.name
    timings: dict[str, float] = {}
    wall0 = time.perf_counter()
    t0 = time.perf_counter()
    bundle = build_primitives(level)
    fed, iso, loads = bundle["fed"], bundle["iso"], bundle["loads"]
    engine = F.resolve_engine(engine, fed.ports)
    F.native_kernel()
    cells = kernel_sample_cells(level, fed.carriers, bundle["vertex_cells"], kernel_count)
    digests = save_npz(out / PRIMITIVES_NPZ, bundle["arrays"])
    config = build_config(level, bundle, archive_id=archive_id, schedules=schedules, float_sweeps=float_sweeps,
                          kernel_cells=cells, engine=engine, workers=workers, lambda2_timeout=lambda2_timeout)
    config["primitive_arrays"] = {"file": PRIMITIVES_NPZ, "arrays": digests}
    write_json(out / CONFIG_JSON, config)
    timings["build_and_primitives"] = time.perf_counter() - t0
    log(f"level {level}: {fed.carriers} carriers, {fed.seams} seams ({fed.inter_count} inter), engine {engine}, {timings['build_and_primitives']:.1f} s")
    seeds = config["seeds"]["schedules"]
    blocks: dict[str, Any] = {}
    os.environ.update(SINGLE_THREAD_ENV)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(F.REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    t_kernel = time.perf_counter()
    chunks = kernel_chunk_count(cells, max(1, workers - 1))
    runner = _KernelChunkRunner(out, chunks, max(1, workers - 1), log)
    runner.start()
    t1 = time.perf_counter()
    float_proc = subprocess.Popen(_worker_command("float", out, seed=seeds["port_pair"][0], sweeps=float_sweeps, engine=engine), env=env, cwd=str(F.REPO_ROOT))
    if float_proc.wait() != 0:
        raise RuntimeError("float worker failed")
    timings["float_port_pair"] = time.perf_counter() - t1
    blocks["float_port_pair"] = load_json(out / FLOAT_JSON)
    log(f"float port_pair: {float_sweeps} sweeps, Phi {blocks['float_port_pair']['Phi_initial']} -> {blocks['float_port_pair']['Phi_terminal']}, {timings['float_port_pair']:.1f} s")
    t1 = time.perf_counter()
    spec_proc = subprocess.Popen(_worker_command("spectrum", out), env=env, cwd=str(F.REPO_ROOT))
    try:
        code = spec_proc.wait(timeout=lambda2_timeout)
    except subprocess.TimeoutExpired:
        spec_proc.kill()
        spec_proc.wait()
        code = None
    timings["spectrum"] = time.perf_counter() - t1
    if code == 0:
        blocks["spectrum"] = load_json(out / SPECTRUM_JSON)
        log(f"spectrum: lambda_2 {blocks['spectrum']['laplacian_lambda_2']}, {timings['spectrum']:.1f} s")
    else:
        for name in (SPECTRUM_JSON, FIEDLER_NPZ):
            (out / name).unlink(missing_ok=True)
        blocks["spectrum"] = None
        log(f"spectrum skipped after {timings['spectrum']:.1f} s (exit {code})")
    t1 = time.perf_counter()
    blocks["integer_port_pair"] = run_integer_law(out, "port_pair", fed, loads, fed.component_of_port, bundle["glued"], seeds["port_pair"], engine=engine, log=log)
    timings["integer_port_pair"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    blocks["integer_isolated"] = run_integer_law(out, "isolated", iso, loads, iso.component_of_port, bundle["isolated"], seeds["isolated"], engine=engine, log=log)
    timings["integer_isolated"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    blocks["float_isolated"] = run_float_isolated(out, iso, loads, seeds["isolated"], engine=engine, log=log)
    timings["float_isolated"] = time.perf_counter() - t1
    runner.join()
    if runner.error is not None:
        raise runner.error
    timings["kernel_glued"] = time.perf_counter() - t_kernel
    t1 = time.perf_counter()
    blocks["kernel"] = _assemble_from_chunks(out, cells, chunks)
    timings["kernel_assemble"] = time.perf_counter() - t1
    log(f"kernel: {len(cells)} cells, glued slow-band share medians {[blocks['kernel']['summary_glued'][str(n)]['slow_band_share']['median'] for n in F.KERNEL_STEPS]}, {timings['kernel_glued']:.1f} s")
    (out / "verify_archive.py").write_text(VERIFIER_SOURCE, encoding="utf-8")
    timings["total_before_verify"] = time.perf_counter() - wall0
    manifest = write_manifest(out, config, blocks, timings)
    t1 = time.perf_counter()
    # Absolute path: with a relative --out the script path would otherwise be resolved inside cwd=out a second time.
    verify = subprocess.run([sys.executable, "-I", str((out / "verify_archive.py").resolve())], capture_output=True, text=True, cwd=str(out))
    verify_seconds = time.perf_counter() - t1
    output = (verify.stdout + verify.stderr).strip()
    log(f"verifier exit {verify.returncode} in {verify_seconds:.1f} s: {output.splitlines()[-1] if output else ''}")
    (out / "README.md").write_text(render_readme(config, manifest, blocks, output, verify.returncode, verify_seconds), encoding="utf-8")
    return {"config": config, "manifest": manifest, "blocks": blocks, "verify_returncode": verify.returncode, "verify_output": output}


def _load_blocks(out: Path) -> dict[str, Any]:
    return {
        "integer_port_pair": load_json(out / INTEGER_JSON["port_pair"]),
        "integer_isolated": load_json(out / INTEGER_JSON["isolated"]),
        "float_port_pair": load_json(out / FLOAT_JSON) if (out / FLOAT_JSON).exists() else None,
        "float_isolated": load_json(out / FLOAT_ISOLATED_JSON),
        "spectrum": load_json(out / SPECTRUM_JSON) if (out / SPECTRUM_JSON).exists() else None,
        "kernel": load_json(out / KERNEL_JSON),
    }


KERNEL_CHUNK_DIR = ".kernel_chunks"


def kernel_chunk_worker_main(archive: Path, chunk: int, chunks: int) -> None:
    """Compute one chunk of the declared kernel sample and write it to the chunk directory."""

    config = load_json(archive / CONFIG_JSON)
    cells = [int(c) for c in config["seeds"]["kernel_sample"]["cells"]]
    part = _chunks(cells, chunks)[chunk]
    results = kernel_worker(str(archive), part)
    directory = archive / KERNEL_CHUNK_DIR
    directory.mkdir(exist_ok=True)
    staging = directory / f"chunk_{chunk:02d}.{os.getpid()}.tmp.npz"
    np.savez(staging, cells=np.asarray([c for c, _ in results], dtype=np.int32), stacks=np.stack([st for _, st in results]))
    os.replace(staging, directory / f"chunk_{chunk:02d}.npz")


class _KernelChunkRunner(threading.Thread):
    """Keeps ``workers`` kernel-chunk subprocesses running until every chunk file exists (resumable)."""

    def __init__(self, out: Path, chunks: int, workers: int, log) -> None:
        super().__init__(daemon=True)
        self.out = Path(out)
        self.chunks = chunks
        self.workers = max(1, workers)
        self.log = log
        self.error: Exception | None = None
        self.started_at = time.perf_counter()

    def run(self) -> None:
        directory = self.out / KERNEL_CHUNK_DIR
        directory.mkdir(exist_ok=True)
        env = dict(os.environ)
        env.update(SINGLE_THREAD_ENV)
        env["PYTHONPATH"] = str(F.REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        pending = [i for i in range(self.chunks) if not (directory / f"chunk_{i:02d}.npz").exists()]
        self.log(f"kernel phase: {self.chunks} chunks, {len(pending)} pending, {self.workers} workers")
        running: dict[int, subprocess.Popen] = {}
        try:
            while pending or running:
                while pending and len(running) < self.workers:
                    i = pending.pop(0)
                    running[i] = subprocess.Popen(_worker_command("kernel", self.out, chunk=i, chunks=self.chunks), env=env, cwd=str(F.REPO_ROOT))
                time.sleep(1.0)
                for i, proc in list(running.items()):
                    code = proc.poll()
                    if code is None:
                        continue
                    del running[i]
                    if code != 0:
                        raise RuntimeError(f"kernel chunk {i} failed with exit code {code}")
                    self.log(f"kernel chunk {i} done at {time.perf_counter() - self.started_at:.1f} s")
        except Exception as error:  # surfaced by the caller after join()
            self.error = error
            for proc in running.values():
                proc.kill()


def _assemble_from_chunks(out: Path, cells: Sequence[int], chunks: int) -> dict[str, Any]:
    directory = Path(out) / KERNEL_CHUNK_DIR
    glued: dict[int, np.ndarray] = {}
    for i in range(chunks):
        with np.load(directory / f"chunk_{i:02d}.npz", allow_pickle=False) as data:
            for cell, stack in zip(data["cells"].tolist(), data["stacks"]):
                glued[int(cell)] = np.asarray(stack)
    if set(glued) != set(int(c) for c in cells):
        raise RuntimeError("kernel chunks do not cover the declared sample")
    block = assemble_kernels(Path(out), load_primitives(Path(out)), cells, glued)
    for path in directory.iterdir():
        path.unlink()
    directory.rmdir()
    return block


def kernel_chunk_count(cells: Sequence[int], workers: int) -> int:
    return max(1, min(2 * max(1, workers), len(cells)))


def kernel_phase(out: Path, *, workers: int = 5, log=print) -> dict[str, Any]:
    """Glued kernels for the declared sample of an existing archive (resume path), then the assembly."""

    out = Path(out)
    config = load_json(out / CONFIG_JSON)
    cells = [int(c) for c in config["seeds"]["kernel_sample"]["cells"]]
    chunks = kernel_chunk_count(cells, workers)
    started = time.perf_counter()
    runner = _KernelChunkRunner(out, chunks, workers, log)
    runner.start()
    runner.join()
    if runner.error is not None:
        raise runner.error
    block = _assemble_from_chunks(out, cells, chunks)
    log(f"kernel: {len(cells)} cells, glued slow-band share medians {[block['summary_glued'][str(n)]['slow_band_share']['median'] for n in F.KERNEL_STEPS]}, {time.perf_counter() - started:.1f} s")
    return block


def refresh_archive(out: Path, *, log=print) -> dict[str, Any]:
    """Re-bind an existing archive: verifier, manifest, verifier run and README from the files on disk."""

    out = Path(out)
    config = load_json(out / CONFIG_JSON)
    manifest_path = out / "archive_manifest.json"
    timings = dict(load_json(manifest_path).get("timings_seconds", {})) if manifest_path.exists() else {}
    blocks = _load_blocks(out)
    if blocks["spectrum"] is not None and "seconds" in blocks["spectrum"]:
        timings["spectrum"] = float(blocks["spectrum"]["seconds"])
    (out / "verify_archive.py").write_text(VERIFIER_SOURCE, encoding="utf-8")
    manifest = write_manifest(out, config, blocks, timings)
    t1 = time.perf_counter()
    # Absolute path: with a relative --out the script path would otherwise be resolved inside cwd=out a second time.
    verify = subprocess.run([sys.executable, "-I", str((out / "verify_archive.py").resolve())], capture_output=True, text=True, cwd=str(out))
    verify_seconds = time.perf_counter() - t1
    output = (verify.stdout + verify.stderr).strip()
    log(f"verifier exit {verify.returncode} in {verify_seconds:.1f} s: {output.splitlines()[-1] if output else ''}")
    (out / "README.md").write_text(render_readme(config, manifest, blocks, output, verify.returncode, verify_seconds), encoding="utf-8")
    return {"config": config, "manifest": manifest, "blocks": blocks, "verify_returncode": verify.returncode, "verify_output": output}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", type=int, default=6)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--archive-id", default=None)
    parser.add_argument("--schedules", type=int, default=F.SCHEDULES)
    parser.add_argument("--float-sweeps", type=int, default=FLOAT_SWEEP_BUDGET_DEFAULT)
    parser.add_argument("--kernel-cells", type=int, default=KERNEL_SAMPLE_DEFAULT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--engine", default="auto", choices=("auto", "native", "numpy", "python"))
    parser.add_argument("--lambda2-timeout", type=float, default=LAMBDA2_TIMEOUT_DEFAULT)
    parser.add_argument("--worker", default=None, choices=("float", "spectrum", "kernel"))
    parser.add_argument("--chunk", type=int, default=None)
    parser.add_argument("--chunks", type=int, default=None)
    parser.add_argument("--refresh", action="store_true", help="re-bind an existing archive (--archive DIR): verifier, manifest, verifier run, README")
    parser.add_argument("--kernels-only", action="store_true", help="recompute the glued kernels of an existing archive (--archive DIR) and refresh it")
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sweeps", type=int, default=None)
    args = parser.parse_args(argv)
    if args.worker == "float":
        float_worker_main(args.archive, args.seed, args.sweeps, args.engine)
        return 0
    if args.worker == "spectrum":
        spectrum_worker_main(args.archive)
        return 0
    if args.worker == "kernel":
        kernel_chunk_worker_main(args.archive, args.chunk, args.chunks)
        return 0
    if args.kernels_only:
        kernel_phase(args.archive or args.out, workers=args.workers)
        result = refresh_archive(args.archive or args.out)
        return 0 if result["verify_returncode"] == 0 else 1
    if args.refresh:
        result = refresh_archive(args.archive or args.out)
        return 0 if result["verify_returncode"] == 0 else 1
    out = args.out or (F.REPO_ROOT / "runs" / DEFAULT_ARCHIVE_ID)
    started = time.perf_counter()
    result = build_archive(args.level, out, archive_id=args.archive_id, schedules=args.schedules, float_sweeps=args.float_sweeps,
                           kernel_count=args.kernel_cells, workers=args.workers, engine=args.engine, lambda2_timeout=args.lambda2_timeout)
    print(f"archive at {out}: {result['manifest']['curated_archive']['file_count']} files, {result['manifest']['curated_archive']['total_bytes']} bytes, "
          f"verifier exit {result['verify_returncode']}, {time.perf_counter() - started:.1f} s")
    return 0 if result["verify_returncode"] == 0 else 1


# --------------------------------------------------------------------------
# README
# --------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_readme(config: dict[str, Any], manifest: dict[str, Any], blocks: dict[str, Any], verifier_output: str,
                  returncode: int, verify_seconds: float) -> str:
    level = config["level"]
    n = config["carriers"]
    seams = config["seams"]
    integer = blocks["integer_port_pair"]
    iso_int = blocks["integer_isolated"]
    flt = blocks.get("float_port_pair")
    flt_iso = blocks["float_isolated"]
    spec = blocks.get("spectrum")
    kern = blocks["kernel"]
    src = manifest["source"]
    steps = [str(s) for s in F.KERNEL_STEPS]
    share_line = ", ".join(f"{kern['summary_glued'][s]['slow_band_share']['median']:.4f}" for s in steps)
    share_min = ", ".join(f"{kern['summary_glued'][s]['slow_band_share']['min']:.4f}" for s in steps)
    share_max = ", ".join(f"{kern['summary_glued'][s]['slow_band_share']['max']:.4f}" for s in steps)
    iso_shares = ", ".join(f"{row['slow_band_share']:.4f}" for row in kern["per_cell"][0]["isolated"])
    top300 = ", ".join(f"{v:.4g}" for v in kern["summary_glued"]["300"]["top_four_eigenvalues_median"])
    sweeps_int = integer["sweeps"]
    attempts_int = integer["attempts_to_balanced_class"]
    lines = []
    lines.append(f"# Exact federation at level {level}: {n:,} carriers under the production gluing\n")
    lines.append(
        f"This directory is the evidence archive `{config['archive_id']}` of the canonical seam-mean law on the "
        f"level-{level} cell rung of the geodesic icosahedral tower: {n:,} twelve-port carriers, {config['ports']:,} ports, "
        f"{seams['intra']:,} intra-carrier seams and {seams['inter']:,} inter-carrier seams of the declared `port_pair` "
        f"gluing (three glued ports per carrier). It was produced by `oph_exact/federation_archive.py` of "
        f"[`oph-physics-sim`]({src['repository']}) from an uncommitted working tree; the manifest records the revision as "
        f"`{src['revision']}` together with the digests of `oph_exact/federation.py` "
        f"(`{src['modules']['oph_exact/federation.py'][:16]}`) and `oph_exact/carrier.py` "
        f"(`{src['modules']['oph_exact/carrier.py'][:16]}`), and the maintainer fills in the commit after committing. "
        f"Configuration, seed streams, primitive arrays, terminal vectors, ledgers and kernel matrices are bound by "
        f"`archive_manifest.json`.\n"
    )
    lines.append("## Result\n")
    lines.append(
        f"Integer law, production gluing, {len(integer['schedules'])} shuffled schedules to the balanced class: "
        f"{sweeps_int['min']} to {sweeps_int['max']} sweeps ({attempts_int['min']:,} to {attempts_int['max']:,} attempts; one sweep "
        f"is {seams['total']:,} attempts), {integer['unit_transfers']['min']:,} to {integer['unit_transfers']['max']:,} unit transfers per "
        f"schedule, {integer['violations_total']} violations of the unit-transfer decrement identity, `V` from {integer['V_initial']:,} to the "
        f"balanced minimum {integer['V_minimum']:,} on every schedule, and one terminal quotient hash equal to the expected multiset hash "
        f"computed from the initial loads alone:\n"
    )
    lines.append(f"```text\nsha256:{integer['expected_quotient_hash']}\n```\n")
    lines.append("| schedule | seed | sweeps | attempts to the balanced class | descents | swaps | waits | unit transfers | odd-tie seams |")
    lines.append("|-|-|-|-|-|-|-|-|-|")
    for index, s in enumerate(integer["schedules"]):
        lines.append(
            f"| {index} | {s['seed']} | {s['sweeps']} | {s['attempts_to_balanced_class']:,} | {s['descents']:,} | {s['swaps']:,} | "
            f"{s['waits']:,} | {s['unit_transfers']:,} | {s['odd_tie_seams_at_termination']:,} |"
        )
    lines.append("")
    lines.append(
        f"The port graph has {config['components']['port_pair']} component{'s' if config['components']['port_pair'] != 1 else ''}, so the balanced class is the set of load vectors with every "
        f"reading in {{q, q+1}} per component; the terminal vectors of the {len(integer['schedules'])} schedules differ as vectors (different odd-tie placements) and agree "
        f"as multisets, and every seam differs by at most one unit at termination. The isolated control ({n:,} components of twelve ports) "
        f"reaches its balanced class in {iso_int['sweeps']['min']} to {iso_int['sweeps']['max']} sweeps with one quotient hash equal to its "
        f"expected multiset hash and {iso_int['violations_total']} identity violations.\n"
    )
    if flt:
        lines.append(
            f"Mean law (float), production gluing, one asynchronous schedule (seed {flt['seed']}) with a declared budget of {flt['sweeps']} sweeps "
            f"({flt['attempts']:,} attempts): `Phi` from {_fmt(flt['Phi_initial'])} to {_fmt(flt['Phi_terminal'])}, `V` from {_fmt(flt['V_initial'])} to "
            f"{_fmt(flt['V_terminal'])} against the component-mean minimum {_fmt(flt['V_minimum'])}, {flt['strict_descent_violations']} strict-descent "
            f"violations over {flt['non_wait_moves']:,} non-wait moves, the per-sweep ledger `V_before - V_after = sum (x_i - x_j)^2 / 2` within "
            f"1e-9 relative on every sweep, {flt['phi_raising_fraction_of_non_wait_moves']:.4f} of the non-wait moves raising `Phi` while `V` drops "
            f"on every one, and a maximal deviation from the component mean of {_fmt(flt['max_abs_deviation_from_component_mean'])} at the end of "
            f"the budget. The schedule is budgeted; its termination at this rung is work in progress. The component mean is the unique fixed point "
            f"of every schedule: each seam move conserves the component totals and the fixed set of the law is the seam equalizer, which by "
            f"linearity is the span of the component indicators. The lane receipt pinned by the manifest "
            f"(`{src['lane_receipt']['path']}`, `{(src['lane_receipt']['sha256'] or '')[:16]}`) verifies this exactly at levels 0 and 1 "
            f"in rational arithmetic.\n"
        )
    lines.append(
        f"The isolated float control terminates: {len(flt_iso['schedules'])} schedules reach `Phi < 1e-18` in {flt_iso['sweeps']['min']} to "
        f"{flt_iso['sweeps']['max']} sweeps with one terminal hash equal to the hash of the exact component means and "
        f"{flt_iso['strict_descent_violations_total']} strict-descent violations.\n"
    )
    if spec:
        lines.append(
            f"Synchronous expectation operator `T_fed = I - L_fed/D`, `D = {_fmt(spec['D'])}`: the port-graph gap is "
            f"`lambda_2 = {_fmt(spec['laplacian_lambda_2'])}` ({spec['method']}), `lambda_max = {_fmt(spec['laplacian_lambda_max'])}`, "
            f"{_fmt(spec['attempts_per_e_fold_of_slowest_mode'])} attempts per e-fold of the slowest mode. The archived eigenvector has residual "
            f"`|L v - lambda_2 v| = {_fmt(spec['eigenvector_residual_norm'])}` at unit norm and Rayleigh quotient {_fmt(spec['eigenvector_rayleigh_quotient'])}. "
            f"The isolated gap is `5 - sqrt(5)` exactly.\n"
        )
    else:
        lines.append("The port-graph gap `lambda_2` at this rung exceeded the declared shift-invert timeout and is work in progress.\n")
    lines.append(
        f"Per-carrier response kernel `K_n = 12 C_n / tr C_n` for {len(kern['cells'])} sampled carriers ({kern['pentagonal_cells_in_sample']} touching "
        f"the pentagonal vertices, vertices {kern['vertices_covered']}) at `n = {', '.join(steps)}`. Isolated: slow-band share {iso_shares}, equal to the "
        f"exact carrier kernel within {_fmt(kern['isolated_max_abs_deviation_from_carrier_kernel'])} at every `n` and to the intrinsic Gram "
        f"`4 P_slow` within {_fmt(kern['isolated_max_abs_deviation_from_4_P_slow_at_n_300'])} at `n = 300`. Glued (the measurement of the declared "
        f"convention): slow-band share median {share_line} (min {share_min}; max {share_max}), median top-four eigenvalues at `n = 300` "
        f"{top300}; the isotropic Green's-pattern reference is {kern['green_pattern_slow_share_reference']:.4f}.\n"
    )
    lines.append("## Independent replay\n")
    lines.append(
        "`verify_archive.py` imports numpy and the standard library only. It checks every manifest digest, rebuilds the port-graph "
        "components from the seam arrays, recomputes the expected multisets and their hash from the initial loads, checks every terminal "
        "vector against its multiset and every seam for a difference of at most one unit, replays schedule 0 of the integer law from its "
        "archived seed, checks the `V` ledger arithmetic of every schedule, the float ledger and terminal state, the eigenvector residual "
        "of `lambda_2`, and recomputes the isolated kernel limit for two sample carriers by direct propagation with rescaling.\n"
    )
    lines.append(
        "The replayed move rule (identical to `carrier.integer_nearest_agreement` as `federation.py` declares it): seam `s` has the archived "
        "endpoints `a_s` (first) and `b_s` (second); a sweep draws `seq = integers(0, |S|, size=|S|, dtype=int64)` and then "
        "`coin = integers(0, 2, size=|S|, dtype=int64)` from `numpy.random.default_rng(seed)` (PCG64) and applies the attempts in draw order. "
        "For attempt `t` with `s = seq[t]`, total `T = x_a + x_b`, `lo = floor(T/2)`, `hi = T - lo`: endpoint `a` receives `hi` when "
        "`coin[t] = 1` and `lo` when `coin[t] = 0`, endpoint `b` receives the rest. With `d = x_a - x_b`: `d = 0` is a wait; `|d| = 1` is a wait "
        "when the placement equals the current state and a swap otherwise (`V` unchanged); `|d| >= 2` is a descent lowering `V` by "
        "`(d^2 - (d mod 2)) / 2`, the sum of `2(d_k - 1)` over its `floor(|d|/2)` unit transfers with mismatches `|d|, |d| - 2, ...`. The schedule "
        "ends with the sweep in which `V` first equals the balanced minimum.\n"
    )
    lines.append("With Python 3 and NumPy installed:\n\n```bash\npython3 verify_archive.py\n```\n")
    lines.append(f"Output of the run recorded at build time (exit code {returncode}, {verify_seconds:.1f} s):\n\n```text\n{verifier_output}\n```\n")
    lines.append("## Known boundaries\n")
    lines.append(
        "The gluing is a declared convention (the production routing of `oph_fpe.core.screen_ports.assign_echosahedral_ports`, recorded "
        "verbatim in `config.json`); the source-derived gluing is work in progress, and the slow-band numbers under gluing are measurements "
        "of that convention. The float law's termination at this rung is budgeted; the terminal quotient of every schedule is the component "
        "mean by the theorem stated above, verified exactly at the small rungs of the lane receipt. The ordering of `lambda_2` as the second "
        "eigenvalue comes from the archived eigenvalue computation; the verifier certifies an eigenvalue within the residual of the archived "
        "value with an eigenvector orthogonal to the component indicators. No physical identification is made: physical position, physical length, the "
        "refinement limit, the continuum limit and any field attachment are outside the archive.\n"
    )
    lines.append("## Reproduce\n")
    lines.append(
        f"In a clone of the simulator with the two module digests above:\n\n```bash\npython3 -m pip install -e '.[dev]'\n"
        f"python3 -m oph_exact.federation_archive --level {level} --out runs/{config['archive_id']} --workers {config['budgets']['workers']} "
        f"--float-sweeps {config['budgets']['float_port_pair_sweeps']} --kernel-cells {config['budgets']['kernel_sample_count']}\n```\n\n"
        f"Seeds: loads `{config['seeds']['loads']['seed']}`, schedules `{config['seeds']['schedules']['rule']}`, kernel sample "
        f"`{config['seeds']['kernel_sample']['seed']}`. Reproductions compare the archived array digests and disclose platform or dependency changes "
        f"(numpy {config['platform']['numpy']}, scipy {config['platform']['scipy']}, Python {config['platform']['python']}).\n"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Embedded simulator-free verifier (written into every archive)
# --------------------------------------------------------------------------

VERIFIER_SOURCE = r'''#!/usr/bin/env python3
"""Fail-closed verifier for an exact-federation evidence archive.

Imports numpy and the standard library only.  Checks every manifest digest,
rebuilds the port-graph components from the seam arrays, recomputes the
expected terminal multisets from the initial loads, checks every archived
terminal vector, replays schedule 0 of the integer law from its seed with
numpy PCG64 and the declared move rule, checks the V ledgers, the float
terminal state, the lambda_2 eigenvector and the isolated kernel limit.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
CONTROL_FILES = {"README.md", "archive_manifest.json", "verify_archive.py"}
SCHEMA = "oph.exact_federation_archive.v1"
PORTS = 12
SEAMS_PER_CARRIER = 30
KERNEL_STEPS = (1, 5, 30, 100, 300)
INTEGER_JSON = {"port_pair": "integer_law_port_pair.json", "isolated": "integer_law_isolated.json"}
INTEGER_NPZ = {"port_pair": "integer_terminal_port_pair.npz", "isolated": "integer_terminal_isolated.npz"}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray, dtype: str) -> str:
    arr = np.ascontiguousarray(np.asarray(values), dtype=np.dtype(dtype).newbyteorder("<"))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def load_json(name: str) -> Any:
    with (ROOT / name).open("r", encoding="ascii") as handle:
        return json.load(handle)


def load_npz(name: str) -> dict[str, np.ndarray]:
    with np.load(ROOT / name, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def close(a: float, b: float, rel: float, absolute: float = 0.0) -> bool:
    return abs(float(a) - float(b)) <= rel * max(abs(float(a)), abs(float(b))) + absolute


# --------------------------------------------------------------------------
# Inventory
# --------------------------------------------------------------------------


def verify_inventory(manifest: dict[str, Any]) -> None:
    expected = {row["path"]: row for row in manifest["inventory"]}
    actual = {path.name for path in ROOT.iterdir() if path.is_file() and path.name not in CONTROL_FILES}
    require(actual == set(expected), "archive inventory names differ")
    lines = []
    total = 0
    for name in sorted(expected):
        row = expected[name]
        path = ROOT / name
        size = path.stat().st_size
        digest = sha256_file(path)
        require(size == row["bytes"], f"byte count mismatch: {name}")
        require(digest == row["sha256"], f"SHA-256 mismatch: {name}")
        total += size
        lines.append(f"{digest}  {size}  {name}\n")
    curated = manifest["curated_archive"]
    require(len(expected) == curated["file_count"], "file count mismatch")
    require(total == curated["total_bytes"], "total byte count mismatch")
    require(hashlib.sha256("".join(lines).encode("utf-8")).hexdigest() == curated["inventory_sha256"], "inventory digest mismatch")
    require(sha256_file(ROOT / "verify_archive.py") == manifest["exact_verification"]["verifier_sha256"], "verifier digest mismatch")


# --------------------------------------------------------------------------
# Geometry, components and expected multisets
# --------------------------------------------------------------------------


def multiset_hash(size: np.ndarray, q: np.ndarray, r: np.ndarray) -> str:
    entries = []
    for m, qq, rr in zip(size.tolist(), q.tolist(), r.tolist()):
        multiset = []
        if m - rr:
            multiset.append([qq, m - rr])
        if rr:
            multiset.append([qq + 1, rr])
        entries.append([m, multiset])
    return sha256_json({"canonicalizer": "component_multiset", "components": entries})


def components_by_union_find(carriers: int, inter: np.ndarray) -> np.ndarray:
    parent = list(range(carriers))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for u, v in zip(inter[:, 0].tolist(), inter[:, 2].tolist()):
        ru, rv = find(u), find(v)
        if ru != rv:
            if ru < rv:
                parent[rv] = ru
            else:
                parent[ru] = rv
    roots = np.asarray([find(i) for i in range(carriers)], dtype=np.int64)
    distinct = np.unique(roots)  # sorted, so components are ordered by their lowest carrier
    relabel = np.empty(carriers, dtype=np.int64)
    relabel[distinct] = np.arange(distinct.size)
    return relabel[roots]


def verify_geometry(config: dict[str, Any], prim: dict[str, np.ndarray]) -> dict[str, Any]:
    template = prim["carrier_seam_template"].astype(np.int64)
    require(template.shape == (SEAMS_PER_CARRIER, 2), "carrier seam template shape")
    require(bool(np.all(template[:, 0] < template[:, 1])) and template.min() >= 0 and template.max() < PORTS, "template ports")
    require(len({(int(i), int(j)) for i, j in template}) == SEAMS_PER_CARRIER, "duplicate template seams")
    require(template.tolist() == config["carrier_seam_template"], "template differs from config")
    require(bool(np.all(np.bincount(template.ravel(), minlength=PORTS) == 5)), "template degree")
    adjacency = np.zeros((PORTS, PORTS), dtype=bool)
    adjacency[template[:, 0], template[:, 1]] = True
    adjacency |= adjacency.T
    reach = np.zeros(PORTS, dtype=bool)
    reach[0] = True
    for _ in range(PORTS):
        reach |= adjacency[reach].any(axis=0)
    require(bool(reach.all()), "template is not connected")
    carriers = int(config["carriers"])
    ports = carriers * PORTS
    require(prim["initial_loads"].shape == (carriers, PORTS), "initial loads shape")
    require(config["ports"] == ports, "port count")
    require(np.array_equal(prim["intra_seam_carrier"], np.repeat(np.arange(carriers), SEAMS_PER_CARRIER)), "intra seam carriers")
    require(np.array_equal(prim["intra_seam_port_a"], np.tile(template[:, 0], carriers)), "intra seam ports a")
    require(np.array_equal(prim["intra_seam_port_b"], np.tile(template[:, 1], carriers)), "intra seam ports b")
    inter = prim["inter_seam"].astype(np.int64)
    m = int(inter.shape[0])
    require(inter.shape == (m, 4), "inter seam shape")
    require(config["seams"]["intra"] == carriers * SEAMS_PER_CARRIER and config["seams"]["inter"] == m, "seam counts")
    require(config["seams"]["total"] == carriers * SEAMS_PER_CARRIER + m, "total seam count")
    require(inter[:, [0, 2]].min() >= 0 and inter[:, [0, 2]].max() < carriers, "inter seam carriers")
    require(inter[:, [1, 3]].min() >= 0 and inter[:, [1, 3]].max() < PORTS, "inter seam ports")
    require(bool(np.all(inter[:, 0] != inter[:, 2])), "inter seam inside one carrier")
    slots = np.concatenate([inter[:, 0] * PORTS + inter[:, 1], inter[:, 2] * PORTS + inter[:, 3]])
    require(np.unique(slots).size == slots.size, "a carrier port is glued more than once")
    glued = np.bincount(np.concatenate([inter[:, 0], inter[:, 2]]), minlength=carriers)
    production = config["gluing"]["production"]
    require(bool(np.all(glued == production["glued_ports_per_carrier"])), "glued ports per carrier")
    usage = np.bincount(np.concatenate([inter[:, 1], inter[:, 3]]), minlength=PORTS)
    require(usage.tolist() == production["port_usage_histogram"], "port usage histogram")
    require(sha256_json(inter.tolist()) == production["port_pairs_sha256"], "port pair digest")
    loads = prim["initial_loads"].astype(np.int64).ravel()
    require(loads.min() >= 0 and loads.max() <= config["seeds"]["loads"]["load_max"], "load range")
    rng = np.random.default_rng(config["seeds"]["loads"]["seed"])
    require(np.array_equal(rng.integers(0, config["seeds"]["loads"]["load_max"] + 1, size=ports), loads), "initial loads differ from the declared seed stream")
    carrier_labels = components_by_union_find(carriers, inter)
    labels = np.repeat(carrier_labels, PORTS)
    require(np.array_equal(labels, prim["component_of_port"].astype(np.int64)), "component labels")
    count = int(labels.max()) + 1
    require(config["components"]["port_pair"] == count, "component count")
    size = np.bincount(labels, minlength=count).astype(np.int64)
    total = np.rint(np.bincount(labels, weights=loads.astype(float), minlength=count)).astype(np.int64)
    q = total // size
    r = total - q * size
    for name, arr in (("component_size", size), ("component_total", total), ("component_q", q), ("component_r", r)):
        require(np.array_equal(prim[name].astype(np.int64), arr), f"{name} differs")
    iso_labels = np.arange(ports, dtype=np.int64) // PORTS
    iso_size = np.full(carriers, PORTS, dtype=np.int64)
    iso_total = loads.reshape(carriers, PORTS).sum(axis=1)
    iso_q = iso_total // PORTS
    iso_r = iso_total - iso_q * PORTS
    for name, arr in (("isolated_component_total", iso_total), ("isolated_component_q", iso_q), ("isolated_component_r", iso_r)):
        require(np.array_equal(prim[name].astype(np.int64), arr), f"{name} differs")
    require(config["components"]["isolated"] == carriers, "isolated component count")
    intra_a = prim["intra_seam_carrier"].astype(np.int64) * PORTS + prim["intra_seam_port_a"].astype(np.int64)
    intra_b = prim["intra_seam_carrier"].astype(np.int64) * PORTS + prim["intra_seam_port_b"].astype(np.int64)
    seam_a = np.concatenate([intra_a, inter[:, 0] * PORTS + inter[:, 1]])
    seam_b = np.concatenate([intra_b, inter[:, 2] * PORTS + inter[:, 3]])
    return {
        "carriers": carriers, "ports": ports, "loads": loads, "template": template,
        "port_pair": {"labels": labels, "size": size, "total": total, "q": q, "r": r, "seam_a": seam_a, "seam_b": seam_b},
        "isolated": {"labels": iso_labels, "size": iso_size, "total": iso_total, "q": iso_q, "r": iso_r, "seam_a": intra_a, "seam_b": intra_b},
    }
'''

VERIFIER_SOURCE += r'''

# --------------------------------------------------------------------------
# Integer law: terminal vectors, ledgers and the replay of schedule 0
# --------------------------------------------------------------------------


def unit_transfer_decrements(max_mismatch: int) -> np.ndarray:
    """``decrement[d]`` = sum of 2(d_k - 1) over the unit transfers of a descent with mismatch d."""

    table = np.zeros(max_mismatch + 1, dtype=np.int64)
    for d in range(2, max_mismatch + 1):
        total = 0
        dk = d
        while dk >= 2:
            total += 2 * (dk - 1)
            dk -= 2
        require(total == (d * d - (d % 2)) // 2, "closed form of the unit-transfer ledger")
        table[d] = total
    return table


def apply_sweep(x: np.ndarray, seam_a: np.ndarray, seam_b: np.ndarray, seq: np.ndarray, coin: np.ndarray, v: int, v_min: int,
                decrement: np.ndarray, chunk: int = 65536) -> tuple[tuple[int, int, int, int], int, int]:
    """Sequential semantics of one sweep, applied in dependency layers of pairwise disjoint seams."""

    ports = x.size
    descents = swaps = waits = transfers = 0
    first = -1
    for start in range(0, seq.size, chunk):
        ids = seq[start : start + chunk]
        ea = seam_a[ids]
        eb = seam_b[ids]
        ec = coin[start : start + chunk]
        r = ids.size
        delta = np.zeros(r, dtype=np.int64)
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
            earliest = np.full(ports, m, dtype=np.int64)
            earliest[sorted_ports[starts]] = (order // 2)[starts]
            pos = np.arange(m, dtype=np.int64)
            ready = (earliest[ra] == pos) & (earliest[rb] == pos)
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
            change = na * na + nb * nb - xa * xa - xb * xb
            require(bool(np.all(change[desc] == -decrement[ad[desc]])), "a descent move violates the unit-transfer decrement identity")
            require(bool(np.all(change[~desc] == 0)), "a wait or swap changed V")
            waits += int(np.count_nonzero(wait))
            swaps += int(np.count_nonzero(swap))
            descents += int(np.count_nonzero(desc))
            transfers += int(np.sum((ad >> 1)[desc]))
            x[la] = na
            x[lb] = nb
            delta[remaining[ready]] = change
            remaining = remaining[~ready]
        running = v + np.cumsum(delta)
        if first < 0:
            hit = np.flatnonzero(running == v_min)
            if hit.size:
                first = start + int(hit[0])
        v = int(running[-1])
    return (descents, swaps, waits, transfers), v, first


def replay_schedule_zero(config: dict[str, Any], geometry: dict[str, Any], gluing: str, schedule: dict[str, Any], archived: np.ndarray) -> None:
    g = geometry[gluing]
    seam_a, seam_b = g["seam_a"], g["seam_b"]
    seams = int(seam_a.size)
    loads = geometry["loads"]
    x = loads.astype(np.int64).copy()
    v = int(np.dot(x, x))
    v_min = int(schedule["V_minimum"])
    decrement = unit_transfer_decrements(int(loads.max() - loads.min()))
    rng = np.random.default_rng(int(schedule["seed"]))
    draw_hashes = schedule.get("draw_sha256_per_sweep")
    require(draw_hashes is not None and len(draw_hashes) == schedule["sweeps"], "schedule 0 carries one draw digest per sweep")
    first_attempt = 0 if v == v_min else -1
    sweep = 0
    while first_attempt < 0 and sweep < schedule["sweeps"]:
        seq = rng.integers(0, seams, size=seams, dtype=np.int64)
        coin = rng.integers(0, 2, size=seams, dtype=np.int64)
        require(
            [array_sha256(seq, "int64"), array_sha256(coin, "int64")] == draw_hashes[sweep],
            f"RNG stream differs at sweep {sweep} (numpy {np.__version__} against archived numpy {config['rng']['numpy_version']})",
        )
        counts, v, first = apply_sweep(x, seam_a, seam_b, seq, coin, v, v_min, decrement)
        if first >= 0:
            first_attempt = sweep * seams + first + 1
        sweep += 1
        row = schedule["per_sweep"][sweep - 1]
        require(v == schedule["V_ledger"][sweep], f"replayed V differs at sweep {sweep}")
        require(v == int(np.dot(x, x)), f"replayed ledger differs from the state at sweep {sweep}")
        require(counts == (row["descents"], row["swaps"], row["waits"], row["unit_transfers"]), f"replayed move counts differ at sweep {sweep}")
    require(sweep == schedule["sweeps"] and first_attempt == schedule["attempts_to_balanced_class"], "replayed termination differs")
    require(np.array_equal(x, archived.astype(np.int64)), "replayed terminal vector differs from the archived vector")
    require(array_sha256(x.astype(np.int8), "int8") == schedule["terminal_sha256"], "replayed terminal digest differs")


def verify_integer_law(config: dict[str, Any], geometry: dict[str, Any], gluing: str, *, replay: bool) -> dict[str, Any]:
    block = load_json(INTEGER_JSON[gluing])
    arrays = load_npz(INTEGER_NPZ[gluing])
    g = geometry[gluing]
    labels, size, q, r = g["labels"], g["size"], g["q"], g["r"]
    seam_a, seam_b = g["seam_a"], g["seam_b"]
    seams = int(seam_a.size)
    loads = geometry["loads"]
    schedules = block["schedules"]
    require(block["gluing"] == gluing and block["seams"] == seams, "integer block identity")
    require(len(schedules) == config["schedule_count"], "integer schedule count")
    require([s["seed"] for s in schedules] == config["seeds"]["schedules"][gluing], "integer schedule seeds")
    expected = multiset_hash(size, q, r)
    require(block["expected_quotient_hash"] == expected, "expected multiset hash")
    v0 = int(np.dot(loads, loads))
    v_min = int(np.sum((size - r) * q * q + r * (q + 1) * (q + 1)))
    require(block["V_initial"] == v0 and block["V_minimum"] == v_min, "V endpoints of the integer block")
    dmax = int(loads.max() - loads.min())
    decrement = unit_transfer_decrements(dmax)
    q_of = q[labels]
    for index, s in enumerate(schedules):
        tag = f"{gluing} schedule {index}"
        require(s["terminated"] is True, f"{tag} did not terminate")
        require(s["unit_transfer_decrement_identity_violations"] == 0 and s["strict_descent_violations"] == 0, f"{tag} violations")
        require(s["quotient_hash"] == expected and s["quotient_hash_equals_expected"] is True, f"{tag} quotient hash")
        require(s["V_initial"] == v0 and s["V_minimum"] == v_min and s["V_terminal"] == v_min, f"{tag} V endpoints")
        ledger = s["V_ledger"]
        rows = s["per_sweep"]
        require(len(ledger) == s["sweeps"] + 1 and len(rows) == s["sweeps"], f"{tag} ledger length")
        require(ledger[0] == v0 and ledger[-1] == v_min, f"{tag} ledger endpoints")
        require(s["attempts"] == s["sweeps"] * seams, f"{tag} attempts")
        require(sum(row["descents"] for row in rows) == s["descents"], f"{tag} descent total")
        require(sum(row["swaps"] for row in rows) == s["swaps"], f"{tag} swap total")
        require(sum(row["waits"] for row in rows) == s["waits"], f"{tag} wait total")
        require(sum(row["unit_transfers"] for row in rows) == s["unit_transfers"], f"{tag} unit transfer total")
        require(sum(row["violations"] for row in rows) == 0, f"{tag} per-sweep violations")
        for k, row in enumerate(rows):
            require(row["descents"] + row["swaps"] + row["waits"] == seams, f"{tag} sweep {k + 1} attempt count")
            drop = ledger[k] - ledger[k + 1]
            require(drop >= 0, f"{tag} sweep {k + 1} raises V")
            require((2 * row["unit_transfers"] <= drop <= 2 * (dmax - 1) * row["unit_transfers"]) if dmax >= 2 else drop == 0, f"{tag} sweep {k + 1} ledger bounds")
            require(row["unit_transfers"] >= row["descents"], f"{tag} sweep {k + 1} transfers below descents")
        first = s["attempts_to_balanced_class"]
        require((first == 0 and s["sweeps"] == 0) or ((s["sweeps"] - 1) * seams < first <= s["sweeps"] * seams), f"{tag} terminating attempt lies outside the last sweep")
        x8 = arrays[s["terminal_array"]]
        require(x8.dtype == np.int8 and x8.shape == (geometry["ports"],), f"{tag} terminal array shape")
        require(array_sha256(x8, "int8") == s["terminal_sha256"], f"{tag} terminal digest")
        x = x8.astype(np.int64)
        require(x.min() >= loads.min() and x.max() <= loads.max(), f"{tag} terminal range")
        require(int(np.dot(x, x)) == v_min, f"{tag} terminal V")
        require(np.array_equal(np.rint(np.bincount(labels, weights=x.astype(float), minlength=size.size)).astype(np.int64), g["total"]), f"{tag} conservation")
        require(bool(np.all((x == q_of) | (x == q_of + 1))), f"{tag} terminal reading outside the balanced class")
        high = np.bincount(labels, weights=(x == q_of + 1).astype(float), minlength=size.size)
        require(np.array_equal(np.rint(high).astype(np.int64), r), f"{tag} balanced split")
        low = size - np.rint(high).astype(np.int64)
        require(multiset_hash(size, q, np.rint(high).astype(np.int64)) == expected and bool(np.all(low + np.rint(high).astype(np.int64) == size)), f"{tag} terminal multiset hash")
        d = np.abs(x[seam_a] - x[seam_b])
        require(int(d.max()) <= 1 and int(d.max()) == s["max_seam_difference_at_termination"], f"{tag} seam difference")
        require(int(np.count_nonzero(d == 1)) == s["odd_tie_seams_at_termination"], f"{tag} odd-tie seam count")
    require(block["unique_quotient_hash_count"] == 1 and block["quotient_hash_equals_expected_all"] is True, "integer block verdict")
    if replay:
        started = time.perf_counter()
        replay_schedule_zero(config, geometry, gluing, schedules[0], arrays[schedules[0]["terminal_array"]])
        block["_replay_seconds"] = time.perf_counter() - started
    return block
'''

VERIFIER_SOURCE += r'''

# --------------------------------------------------------------------------
# Mean law: the budgeted float schedule and the isolated control
# --------------------------------------------------------------------------


def verify_float_law(config: dict[str, Any], geometry: dict[str, Any]) -> dict[str, Any] | None:
    if not (ROOT / "float_law_port_pair.json").exists():
        return None
    block = load_json("float_law_port_pair.json")
    state = load_npz("float_terminal_port_pair.npz")[block["terminal_state_array"]]
    g = geometry["port_pair"]
    seam_a, seam_b, labels = g["seam_a"], g["seam_b"], g["labels"]
    seams = int(seam_a.size)
    loads = geometry["loads"]
    require(block["gluing"] == "port_pair" and block["seams"] == seams, "float block identity")
    require(block["seed"] == config["seeds"]["schedules"]["float_port_pair"][0], "float seed")
    require(block["sweeps"] == config["budgets"]["float_port_pair_sweeps"] and block["attempts"] == block["sweeps"] * seams, "float budget")
    require(state.dtype == np.float64 and state.shape == (geometry["ports"],), "float terminal state shape")
    require(array_sha256(state, "float64") == block["terminal_state_sha256"], "float terminal digest")
    require(bool(np.all(np.isfinite(state))), "float terminal state is not finite")
    d0 = loads[seam_a] - loads[seam_b]
    phi0 = int(np.dot(d0, d0))
    v0 = int(np.dot(loads, loads))
    require(close(block["Phi_initial"], phi0, 1e-9) and close(block["V_initial"], v0, 1e-9), "float initial Phi and V")
    d = state[seam_a] - state[seam_b]
    phi = float(np.dot(d, d))
    v = float(np.dot(state, state))
    require(close(block["Phi_terminal"], phi, 1e-6), "float terminal Phi")
    require(close(block["V_terminal"], v, 1e-9), "float terminal V")
    require(close(block["total_terminal"], float(np.sum(state)), 1e-9) and close(float(np.sum(state)), float(np.sum(loads)), 1e-9), "float conservation")
    count = int(labels.max()) + 1
    totals = np.bincount(labels, weights=loads.astype(float), minlength=count)
    sizes = np.bincount(labels, minlength=count)
    v_min = float(np.sum(totals * totals / sizes))
    require(close(block["V_minimum"], v_min, 1e-9), "float V minimum")
    means = np.bincount(labels, weights=state, minlength=count) / sizes
    deviation = float(np.max(np.abs(state - means[labels])))
    require(close(block["max_abs_deviation_from_component_mean"], deviation, 1e-5, 1e-9), "float deviation from the component mean")
    rows = block["per_sweep"]
    require(len(rows) == block["sweeps"], "float per-sweep rows")
    previous = float(v0)
    ledger = 0.0
    waits = violations = raising = 0
    for k, row in enumerate(rows):
        require(row["sweep"] == k + 1, "float sweep index")
        require(row["V"] <= previous * (1 + 1e-12), f"float V rises at sweep {k + 1}")
        require(abs((previous - row["V"]) - row["ledger_increment"]) <= 1e-9 * v0 + 1e-6 * abs(row["ledger_increment"]), f"float ledger arithmetic at sweep {k + 1}")
        require(row["violations"] == 0, f"float strict-descent violation at sweep {k + 1}")
        require(0 <= row["waits"] <= seams, f"float waits at sweep {k + 1}")
        previous = row["V"]
        ledger += row["ledger_increment"]
        waits += row["waits"]
        violations += row["violations"]
        raising += row["phi_raising_moves"]
    require(close(previous, v, 1e-9), "float ledger end differs from the terminal state")
    require(abs((v0 - v) - ledger) <= 1e-8 * v0, "float total ledger")
    require(block["waits"] == waits and block["strict_descent_violations"] == violations == 0 and block["phi_raising_moves"] == raising, "float totals")
    require(block["non_wait_moves"] == block["attempts"] - waits, "float non-wait moves")
    require(block["terminated"] is (phi < block["threshold"]), "float termination flag")
    return block


def verify_float_isolated(config: dict[str, Any], geometry: dict[str, Any]) -> dict[str, Any]:
    block = load_json("float_law_isolated.json")
    g = geometry["isolated"]
    carriers = geometry["carriers"]
    schedules = block["schedules"]
    require(block["gluing"] == "isolated" and block["seams"] == int(g["seam_a"].size), "isolated float identity")
    require(len(schedules) == config["schedule_count"], "isolated float schedule count")
    require([s["seed"] for s in schedules] == config["seeds"]["schedules"]["float_isolated"], "isolated float seeds")
    payload = {
        "canonicalizer": "component_lattice_snap",
        "component_sizes": g["size"].tolist(),
        "component_of_port": g["labels"].tolist(),
        "q": g["total"][g["labels"]].tolist(),
    }
    expected = sha256_json(payload)
    require(block["expected_terminal_quotient_hash"] == expected, "isolated float expected hash")
    for index, s in enumerate(schedules):
        tag = f"isolated float schedule {index}"
        require(s["terminated"] is True and s["phi_terminal"] < block["threshold"], f"{tag} termination")
        require(s["terminal_quotient_hash"] == expected, f"{tag} terminal hash")
        require(s["strict_descent_violations"] == 0, f"{tag} violations")
        require(s["descent_ledger_relative_error_below_1e-9"] is True, f"{tag} ledger")
        require(s["component_mean_within_1e-9"] is True and s["max_abs_deviation_from_component_mean"] < 1e-9, f"{tag} deviation")
        require(s["lattice_residual_max"] < 0.5, f"{tag} lattice snap margin")
        require(s["attempts"] == s["sweeps"] * block["seams"], f"{tag} attempts")
    require(block["unique_terminal_hash_count"] == 1 and block["terminal_hash_equals_expected_all"] is True and block["all_terminated"] is True, "isolated float verdict")
    require(block["components"] == carriers, "isolated float components")
    return block


# --------------------------------------------------------------------------
# Spectrum
# --------------------------------------------------------------------------


def carrier_laplacian(template: np.ndarray) -> np.ndarray:
    lap = np.zeros((PORTS, PORTS))
    for i, j in template.tolist():
        lap[i, j] -= 1.0
        lap[j, i] -= 1.0
        lap[i, i] += 1.0
        lap[j, j] += 1.0
    return lap


def verify_spectrum(config: dict[str, Any], geometry: dict[str, Any]) -> dict[str, Any] | None:
    template = geometry["template"]
    iso_values = np.linalg.eigvalsh(carrier_laplacian(template))
    require(abs(iso_values[1] - (5.0 - 5.0**0.5)) < 1e-12, "isolated gap differs from 5 - sqrt(5)")
    if not (ROOT / "spectrum_port_pair.json").exists():
        return None
    block = load_json("spectrum_port_pair.json")
    vec = load_npz("fiedler_port_pair.npz")[block["eigenvector_array"]]
    g = geometry["port_pair"]
    seam_a, seam_b, labels = g["seam_a"], g["seam_b"], g["labels"]
    ports = geometry["ports"]
    require(vec.dtype == np.float64 and vec.shape == (ports,), "eigenvector shape")
    require(array_sha256(vec, "float64") == block["eigenvector_sha256"], "eigenvector digest")
    norm = float(np.linalg.norm(vec))
    require(abs(norm - 1.0) < 1e-9, "eigenvector norm")
    degree = np.bincount(seam_a, minlength=ports) + np.bincount(seam_b, minlength=ports)
    lv = degree * vec - (np.bincount(seam_a, weights=vec[seam_b], minlength=ports) + np.bincount(seam_b, weights=vec[seam_a], minlength=ports))
    lam = float(block["laplacian_lambda_2"])
    residual = float(np.linalg.norm(lv - lam * vec))
    require(residual < 1e-8, f"eigenvector residual {residual}")
    dv = vec[seam_a] - vec[seam_b]
    rayleigh = float(np.dot(dv, dv))
    require(close(rayleigh, lam, 1e-8, 1e-14), "Rayleigh quotient differs from lambda_2")
    require(close(block["eigenvector_rayleigh_quotient"], rayleigh, 1e-8, 1e-14), "archived Rayleigh quotient")
    count = int(labels.max()) + 1
    require(float(np.max(np.abs(np.bincount(labels, weights=vec, minlength=count)))) < 1e-8, "eigenvector is not orthogonal to the component indicators")
    denominator = 2 * seam_a.size / geometry["carriers"]
    require(close(block["D"], denominator, 1e-12), "D")
    require(close(block["t_fed_second_eigenvalue"], 1.0 - lam / denominator, 1e-9), "second eigenvalue of T_fed")
    require(lam > 0 and block["laplacian_lambda_max"] >= lam, "spectral ordering")
    require(block["laplacian_lambda_max"] <= 2 * float(degree.max()) + 1e-9, "lambda_max above the degree bound")
    require(close(block["isolated_lambda_2"], iso_values[1], 1e-9), "isolated lambda_2")
    block["_residual"] = residual
    return block
'''

VERIFIER_SOURCE += r'''

# --------------------------------------------------------------------------
# Response kernels
# --------------------------------------------------------------------------


def slow_band_projector(template: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(carrier_laplacian(template))
    band = np.abs(values - (5.0 - 5.0**0.5)) < 1e-9
    require(int(np.count_nonzero(band)) == 3, "slow band multiplicity")
    v = vectors[:, band]
    return v @ v.T


def isolated_kernels_by_propagation(prim: dict[str, np.ndarray], cell: int) -> dict[int, np.ndarray]:
    """Direct matrix powers with rescaling on the carrier's own block, from the archived intra seams."""

    rows = slice(cell * SEAMS_PER_CARRIER, (cell + 1) * SEAMS_PER_CARRIER)
    require(bool(np.all(prim["intra_seam_carrier"][rows] == cell)), "intra seam rows of the sample carrier")
    template = np.stack([prim["intra_seam_port_a"][rows], prim["intra_seam_port_b"][rows]], axis=1).astype(np.int64)
    lap = carrier_laplacian(template)
    q = np.eye(PORTS) - np.ones((PORTS, PORTS)) / PORTS
    y = q.copy()
    out = {}
    target = {2 * n: n for n in KERNEL_STEPS}
    for step in range(1, max(target) + 1):
        y = y - (lap @ y) / 60.0
        scale = np.max(np.abs(y))
        if scale > 0:
            y /= scale
        if step in target:
            c = q @ y
            c = 0.5 * (c + c.T)
            out[target[step]] = PORTS * c / np.trace(c)
    return out


def verify_kernels(config: dict[str, Any], geometry: dict[str, Any], prim: dict[str, np.ndarray]) -> dict[str, Any]:
    block = load_json("kernel_readout.json")
    mats = load_npz("kernel_matrices.npz")
    cells = [int(c) for c in block["cells"]]
    require(cells == config["seeds"]["kernel_sample"]["cells"] and mats["cells"].tolist() == cells, "kernel sample cells")
    require(mats["steps"].tolist() == list(KERNEL_STEPS) and block["steps"] == list(KERNEL_STEPS), "kernel steps")
    for name in ("cells", "steps", "glued", "isolated"):
        entry = block["matrices"][name]
        require(mats[name].shape == tuple(entry["shape"]) and str(mats[name].dtype) == entry["dtype"], f"kernel array {name} shape")
        require(array_sha256(mats[name], entry["dtype"]) == entry["sha256"], f"kernel array {name} digest")
    require(mats["glued"].shape == (len(cells), len(KERNEL_STEPS), PORTS, PORTS), "glued kernel shape")
    p_slow = slow_band_projector(geometry["template"])
    gram = 4.0 * p_slow
    pentagonal = prim["pentagonal_cell"]
    vertex_cells = prim["vertex_cells"].astype(np.int64)
    require(vertex_cells.shape == (12, 5), "vertex cell table")
    covered = sorted({int(v) for v in range(12) for c in cells if c in vertex_cells[v].tolist()})
    require(covered == block["vertices_covered"], "vertices covered by the sample")
    require(block["pentagonal_cells_in_sample"] == int(sum(bool(pentagonal[c]) for c in cells)), "pentagonal cells in the sample")
    iso_dev = 0.0
    gram_dev = 0.0
    for i, entry in enumerate(block["per_cell"]):
        cell = int(entry["cell"])
        require(cell == cells[i] and entry["pentagonal"] == bool(pentagonal[cell]), f"per-cell identity {i}")
        for kind in ("glued", "isolated"):
            for j, (n, row) in enumerate(zip(KERNEL_STEPS, entry[kind])):
                k = mats[kind][i, j]
                require(row["n"] == n, "kernel step order")
                require(float(np.max(np.abs(k - k.T))) < 1e-12, f"{kind} kernel not symmetric (cell {cell}, n {n})")
                require(abs(float(np.trace(k)) - PORTS) < 1e-9, f"{kind} kernel trace (cell {cell}, n {n})")
                eig = np.sort(np.linalg.eigvalsh(k))[::-1]
                require(float(eig[-1]) > -1e-9, f"{kind} kernel not positive semidefinite (cell {cell}, n {n})")
                require(np.allclose(eig, np.asarray(row["eigenvalues"]), atol=1e-8, rtol=0), f"{kind} eigenvalues (cell {cell}, n {n})")
                share = float(np.trace(p_slow @ k @ p_slow) / np.trace(k))
                require(abs(share - row["slow_band_share"]) < 1e-8 and -1e-9 <= share <= 1 + 1e-9, f"{kind} slow-band share (cell {cell}, n {n})")
                require(int(np.linalg.matrix_rank(k, tol=1e-9)) == row["rank_1e-9"], f"{kind} rank (cell {cell}, n {n})")
                if kind == "isolated" and n == 300:
                    gram_dev = max(gram_dev, float(np.max(np.abs(k - gram))))
        require(float(np.max(np.abs(mats["isolated"][i] - mats["isolated"][0]))) < 1e-12, f"isolated kernels differ between carriers (cell {cell})")
    require(gram_dev < 1e-12, f"isolated K_300 differs from 4 P_slow by {gram_dev}")
    samples = [cells[0], cells[-1]] if len(cells) > 1 else [cells[0]]
    for cell in samples:
        i = cells.index(cell)
        recomputed = isolated_kernels_by_propagation(prim, cell)
        for j, n in enumerate(KERNEL_STEPS):
            dev = float(np.max(np.abs(recomputed[n] - mats["isolated"][i, j])))
            iso_dev = max(iso_dev, dev)
            require(dev < 1e-9, f"recomputed isolated kernel differs (cell {cell}, n {n}): {dev}")
            eig = np.sort(np.linalg.eigvalsh(recomputed[n]))[::-1]
            require(np.allclose(eig, np.asarray(block["per_cell"][i]["isolated"][j]["eigenvalues"]), atol=1e-8, rtol=0), f"recomputed isolated eigenvalues (cell {cell}, n {n})")
        limit = np.sort(np.linalg.eigvalsh(recomputed[300]))[::-1]
        require(np.allclose(limit[:3], 4.0, atol=1e-12) and np.allclose(limit[3:], 0.0, atol=1e-12), f"isolated limit eigenvalues (cell {cell})")
    block["_samples"] = samples
    block["_recompute_deviation"] = iso_dev
    block["_gram_deviation"] = gram_dev
    return block


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    started = time.perf_counter()
    try:
        manifest = load_json("archive_manifest.json")
        require(manifest["schema"] == SCHEMA, "archive schema")
        config = load_json("config.json")
        require(config["schema"] == SCHEMA and config["archive_id"] == manifest["archive_id"], "config identity")
        verify_inventory(manifest)
        prim = load_npz("primitives.npz")
        for name, entry in config["primitive_arrays"]["arrays"].items():
            require(name in prim and list(prim[name].shape) == entry["shape"] and str(prim[name].dtype) == entry["dtype"], f"primitive array {name}")
            require(array_sha256(prim[name], entry["dtype"]) == entry["sha256"], f"primitive array digest {name}")
        geometry = verify_geometry(config, prim)
        snapshot = manifest["run_snapshot"]
        require(snapshot["level"] == config["level"] and snapshot["carriers"] == geometry["carriers"], "snapshot identity")
        integer = verify_integer_law(config, geometry, "port_pair", replay=True)
        isolated = verify_integer_law(config, geometry, "isolated", replay=False)
        require(snapshot["integer_law_port_pair"]["expected_quotient_hash"] == integer["expected_quotient_hash"], "snapshot integer hash")
        require(snapshot["integer_law_isolated"]["expected_quotient_hash"] == isolated["expected_quotient_hash"], "snapshot isolated hash")
        flt = verify_float_law(config, geometry)
        flt_iso = verify_float_isolated(config, geometry)
        spectrum = verify_spectrum(config, geometry)
        kernels = verify_kernels(config, geometry, prim)
        if spectrum is not None:
            require(close(snapshot["spectrum_port_pair"]["laplacian_lambda_2"], spectrum["laplacian_lambda_2"], 1e-12), "snapshot lambda_2")
        require(manifest["known_boundaries"]["gluing_is_declared_convention"] is True and manifest["known_boundaries"]["physical_identification"] is False, "boundary flags")
    except (OSError, KeyError, TypeError, ValueError, IndexError, VerificationError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    seconds = time.perf_counter() - started
    print(
        f"PASS: {geometry['carriers']} carriers, {int(geometry['port_pair']['seam_a'].size)} seams, {int(geometry['port_pair']['size'].size)} component(s); "
        f"inventory and primitive digests verified; integer law port_pair {len(integer['schedules'])} schedules in "
        f"{integer['sweeps']['min']}..{integer['sweeps']['max']} sweeps with one multiset hash {integer['expected_quotient_hash'][:16]} "
        f"(schedule 0 replayed from seed {integer['schedules'][0]['seed']} in {integer['_replay_seconds']:.1f} s, terminal vector identical); "
        f"integer law isolated {len(isolated['schedules'])} schedules verified; "
        + (f"float port_pair {flt['sweeps']} sweeps, Phi {flt['Phi_initial']:.6g} -> {flt['Phi_terminal']:.6g}, V ledger exact to 1e-9, 0 violations; " if flt else "float port_pair absent; ")
        + f"float isolated {len(flt_iso['schedules'])} schedules terminated at the component mean; "
        + (f"lambda_2 {spectrum['laplacian_lambda_2']:.6g} with eigenvector residual {spectrum['_residual']:.2e}; " if spectrum else "lambda_2 absent; ")
        + f"kernels: {len(kernels['cells'])} cells, isolated limit recomputed for carriers {kernels['_samples']} within {kernels['_recompute_deviation']:.1e}, "
        f"K_300 = 4 P_slow within {kernels['_gram_deviation']:.1e}; {seconds:.1f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


if __name__ == "__main__":
    raise SystemExit(main())
