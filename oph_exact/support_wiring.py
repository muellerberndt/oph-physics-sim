"""Full S2 support wiring: twelve ports, A5 carriers, canonical repair, provenance order.

Lane A-wire of the exact-carrier package.  Every architectural element of the
declared substrate is in the run at once:

* twelve-port A5 icosahedral carriers (``oph_exact.carrier``) on the cells of
  the geodesic icosahedral tower at level ``L`` (``20 * 4^L`` carriers);
* the full S2 wiring ``W12``: every cell is glued to its twelve neighbouring
  cells (three edge-adjacent and nine vertex-adjacent) through the production
  geometric port assignment of ``oph_fpe.core.screen_ports``; the sixty cells
  touching one of the twelve pentagonal vertices have eleven neighbours and
  keep one port unglued;
* the canonical seam-mean law on the thirty intra-carrier seams and on the
  glued seams, the flagship descent functional ``V = sum x^2``, the integer
  nearest-agreement law with fixed protected totals;
* records as cumulative port loads, positions as each carrier's own
  rank-three readback ``x = 2 P_slow N`` (theorem ``thm:rank-three``);
* the provenance order generated from the log of the glued-seam reads by the
  authenticated read-after-write rule (an event depends on the events that
  wrote the record versions it read; no declared parents).

The receipt answers one question: with everything wired, what does an observer
read from the provenance order of the reads, and how does it compare with the
record-metric route of the source net (``oph_exact.source_net``)?  The
expected reading for a single-level S2 wiring is a ``2+1`` order
(Myrheim--Meyer ordering fraction ``8/35``, dimension three); the receipt
states what is measured, under two event placements (the writer's own
readback and the cell centre on the screen), for one tower level and for a
three-level federation glued by the committed join transport.

Receipt: ``data/exact/support_wiring_receipt.json`` written by
``python -m oph_exact.support_wiring --write`` and checked by ``--check``.
Independent verifier: ``oph_exact/verify_support_wiring_independent.py``.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as sparse_linalg
from scipy.optimize import linear_sum_assignment

from oph_exact import carrier
from oph_exact import federation
from oph_exact import source_net
from oph_fpe.bulk.causet_likeness import invert_myrheim_meyer_fraction, myrheim_meyer_fraction
from oph_fpe.core.icosahedral import build_geodesic_icosahedral_tower
from oph_fpe.core.screen_ports import assign_echosahedral_ports

SCHEMA = "oph.exact.support-wiring.v1"
LOG_SCHEMA = "oph.exact.support-wiring-log.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = REPO_ROOT / "data" / "exact" / "support_wiring_receipt.json"
LOG_DIR = REPO_ROOT / "data" / "exact" / "support_wiring_logs"
PRODUCER_PATH = Path(__file__).resolve()
VERIFIER_PATH = PRODUCER_PATH.parent / "verify_support_wiring_independent.py"
TEST_PATH = REPO_ROOT / "tests" / "test_exact_support_wiring.py"
PIN_FILES = (
    ("oph_exact/support_wiring.py", PRODUCER_PATH),
    ("oph_exact/verify_support_wiring_independent.py", VERIFIER_PATH),
    ("tests/test_exact_support_wiring.py", TEST_PATH),
    ("oph_exact/carrier.py", PRODUCER_PATH.parent / "carrier.py"),
    ("oph_exact/federation.py", PRODUCER_PATH.parent / "federation.py"),
    ("oph_exact/source_net.py", PRODUCER_PATH.parent / "source_net.py"),
)

PORTS = carrier.PORT_COUNT
LEVELS = (3, 4, 5)
DEPTH_LEVELS = (3, 4)  # depth federation (L-2, L-1, L); the L = 5 federation (26,880 carriers) is outside the shared-machine memory budget
SCHEDULES = 16
FLOAT_TERMINATION_LEVELS = (3,)
FLOAT_BUDGET_SWEEPS = {4: 64, 5: 32}
FLOAT_BUDGET_SCHEDULES = 4
KERNEL_STEPS = (1, 5, 30, 100, 300)
KERNEL_SAMPLE = {3: 64, 4: 64, 5: 32}  # carriers per level in the slow-band sample (budget: L = 5 propagation cost)
KERNEL_BATCH = 8
SCHEDULE_SEED_BASE = 919000
PROVENANCE_SEED_BASE = 929000
DEPTH_SEED_BASE = 939000
TIP_SEED_BASE = 949000
BURN_IN_ROUNDS = 16
LOGGED_ROUNDS = {3: 8, 4: 16, 5: 32}
DEPTH_LOGGED_ROUNDS = {3: 8, 4: 8, 5: 8}  # the coarsest level's front is four times faster in angle
LADDER_SLACK_ROUNDS = 2
DELTA_LADDER = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0)
TIP_INTERIOR = 3
TIP_PENTAGONAL = 3
DEPTH_TIPS_PER_LEVEL = 2
CONTROL_LADDER_FRACTION = 0.25  # the control's causal front is about four times faster
EXACT_PAIR_LIMIT = 60_000
PAIR_BLOCK = 8192
SAMPLE_SOURCES = 2048
INTERIOR_ANGLE_DEG = 60.0
GROWTH_MIN_DELTA = 2.0
DEPTH_GROWTH_MIN_DELTA = 1.5
STORED_LOG_LEVELS = (3,)
SOURCE_NET_Q = 13
SOURCE_NET_DIM = 3
LOAD_MAX = federation.LOAD_MAX
MM_2PLUS1 = Fraction(8, 35)
MM_3PLUS1 = Fraction(1, 10)


# --------------------------------------------------------------------------
# Canonical JSON, hashing, rounding
# --------------------------------------------------------------------------


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"


def sha256_of(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_arrays(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        arr = np.ascontiguousarray(arr)
        h.update(str(arr.dtype).encode("ascii"))
        h.update(str(arr.shape).encode("ascii"))
        h.update(arr.tobytes())
    return h.hexdigest()


def rounded(value: float | None) -> float | None:
    """Twelve significant digits; ``None`` passes through."""

    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite value in receipt")
    out = float(f"{value:.12g}")
    return 0.0 if out == 0.0 else out


def stats(values: Sequence[float]) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"min": None, "median": None, "max": None, "mean": None}
    return {
        "min": rounded(arr.min()),
        "median": rounded(float(np.median(arr))),
        "max": rounded(arr.max()),
        "mean": rounded(float(arr.mean())),
    }


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Geometry: the twelve-neighbour cell graph and the port assignment
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CellGeometry:
    level: int
    faces: np.ndarray
    vertices: np.ndarray
    points: np.ndarray  # unit cell centres
    areas: np.ndarray
    pentagonal_cell: np.ndarray
    parent: np.ndarray | None
    expectation_weight: np.ndarray | None


@lru_cache(maxsize=8)
def cell_geometry(level: int) -> CellGeometry:
    tower = build_geodesic_icosahedral_tower(level)
    mesh = tower.levels[level]
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=float)
    points = np.sum(vertices[faces], axis=1)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    parent = None
    weights = None
    if level > 0:
        refinement = tower.cell_refinements[level - 1]
        parent = np.asarray(refinement.child_to_parent, dtype=np.int64)
        weights = np.asarray(refinement.conditional_expectation_weights, dtype=float)
    for arr in (faces, vertices, points):
        arr.setflags(write=False)
    return CellGeometry(
        level=level,
        faces=faces,
        vertices=vertices,
        points=points,
        areas=np.asarray(mesh.spherical_face_areas, dtype=float),
        pentagonal_cell=np.any(faces < 12, axis=1),
        parent=parent,
        expectation_weight=weights,
    )


def neighbour_pairs(level: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unordered cell pairs sharing at least one mesh vertex.

    Returns ``(left, right, shared)`` with ``left < right`` sorted
    lexicographically and ``shared`` the number of shared vertices (two for
    edge-adjacent cells, one for vertex-adjacent cells).
    """

    geometry = cell_geometry(level)
    faces = geometry.faces
    n = faces.shape[0]
    incidence = sparse.coo_matrix(
        (np.ones(3 * n), (np.repeat(np.arange(n, dtype=np.int64), 3), faces.ravel())),
        shape=(n, geometry.vertices.shape[0]),
    ).tocsr()
    shared = (incidence @ incidence.T).tocoo()
    mask = (shared.row < shared.col) & (shared.data > 0)
    left = shared.row[mask].astype(np.int64)
    right = shared.col[mask].astype(np.int64)
    count = shared.data[mask].astype(np.int64)
    order = np.lexsort((right, left))
    return left[order], right[order], count[order]


def local_frames(points: np.ndarray) -> np.ndarray:
    """The production local surface frames (``tangent_x, tangent_y, normal`` as columns)."""

    normals = np.asarray(points, dtype=float).copy()
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-15)
    reference = np.tile(np.asarray([0.0, 0.0, 1.0]), (normals.shape[0], 1))
    reference[np.abs(normals[:, 2]) > 0.9] = np.asarray([1.0, 0.0, 0.0])
    tangent_x = np.cross(reference, normals)
    tangent_x /= np.maximum(np.linalg.norm(tangent_x, axis=1, keepdims=True), 1.0e-15)
    tangent_y = np.cross(normals, tangent_x)
    return np.stack((tangent_x, tangent_y, normals), axis=2)


@lru_cache(maxsize=1)
def port_directions() -> np.ndarray:
    """The twelve icosahedral port directions of the carrier's regular frame.

    These are the base icosahedron vertices of ``oph_fpe.core.icosahedral``;
    the carrier seams of ``oph_exact.carrier`` are exactly the edges between
    them and ``carrier.antipode()`` is the geometric antipode.
    """

    directions = np.asarray(build_geodesic_icosahedral_tower(0).levels[0].vertices, dtype=float)
    seams = set(carrier.seams())
    for i in range(PORTS):
        for j in range(i + 1, PORTS):
            adjacent = np.dot(directions[i], directions[j]) > 0.4
            if adjacent != ((i, j) in seams):
                raise AssertionError("port direction template is inconsistent with carrier.seams()")
    antipode = carrier.antipode()
    if np.max(np.linalg.norm(directions + directions[list(antipode)], axis=1)) > 1e-12:
        raise AssertionError("port direction template is inconsistent with carrier.antipode()")
    directions.setflags(write=False)
    return directions


def optimal_port_assignment(points: np.ndarray, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Float64 optimal matching of neighbour directions to port directions per cell.

    Same frames and templates as the production rule; every cell solves one
    rectangular assignment maximizing the total alignment.  Returns
    ``(left_port, right_port, total_alignment)``.
    """

    n = points.shape[0]
    frames = local_frames(points)
    directions = port_directions()
    endpoint_nodes = np.concatenate([left, right])
    endpoint_other = np.concatenate([right, left])
    vectors = points[endpoint_other] - points[endpoint_nodes]
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1.0e-15)
    local = np.einsum("ni,nij->nj", vectors, frames[endpoint_nodes])
    scores = local @ directions.T
    ports = np.full(endpoint_nodes.size, -1, dtype=np.int64)
    order = np.argsort(endpoint_nodes, kind="stable")
    sorted_nodes = endpoint_nodes[order]
    starts = np.searchsorted(sorted_nodes, np.arange(n), side="left")
    stops = np.searchsorted(sorted_nodes, np.arange(n), side="right")
    total = 0.0
    for cell in range(n):
        ids = order[starts[cell] : stops[cell]]
        rows, cols = linear_sum_assignment(-scores[ids])
        ports[ids[rows]] = cols
        total += float(scores[ids[rows], cols].sum())
    if np.any(ports < 0):
        raise AssertionError("optimal matching left an endpoint without a port")
    return ports[: left.size], ports[left.size :], total


@dataclass(frozen=True)
class Wiring:
    level: int
    carriers: int
    left: np.ndarray
    right: np.ndarray
    shared: np.ndarray
    left_port: np.ndarray
    right_port: np.ndarray
    unglued_port: np.ndarray  # per cell, -1 when all twelve ports are glued
    degree: np.ndarray
    hop_angle_deg: float
    receipt: dict[str, Any]

    @property
    def pairs(self) -> np.ndarray:
        return np.stack([self.left, self.left_port, self.right, self.right_port], axis=1)

    @property
    def edges(self) -> int:
        return int(self.left.size)


@lru_cache(maxsize=8)
def build_wiring(level: int) -> Wiring:
    geometry = cell_geometry(level)
    points = geometry.points
    n = points.shape[0]
    left, right, shared = neighbour_pairs(level)
    degree = np.bincount(np.concatenate([left, right]), minlength=n)
    port_map = assign_echosahedral_ports(left, right, n, points=points)
    accepted = {
        "routing_mode": port_map.routing_mode,
        "overflow_count": int(port_map.overflow_count),
    }
    lp = port_map.left_port.astype(np.int64)
    rp = port_map.right_port.astype(np.int64)
    slots = np.concatenate([left * PORTS + lp, right * PORTS + rp])
    bijection = bool(np.unique(slots).size == slots.size)
    accepted["ports_distinct_per_cell"] = bijection
    accepted["accepted"] = bool(port_map.routing_mode == "icosahedral_directional_assignment" and port_map.overflow_count == 0 and bijection)
    if not accepted["accepted"]:
        raise AssertionError("the production geometric port assignment does not accept the twelve-neighbour graph")
    used = np.zeros((n, PORTS), dtype=bool)
    used[left, lp] = True
    used[right, rp] = True
    unglued = np.full(n, -1, dtype=np.int64)
    for cell in np.flatnonzero(degree < PORTS):
        free = np.flatnonzero(~used[cell])
        if free.size != PORTS - degree[cell]:
            raise AssertionError("unglued port count differs from the missing neighbour count")
        unglued[cell] = int(free[0])
    antipode = np.asarray(carrier.antipode(), dtype=np.int64)
    consistent = antipode[lp] == rp
    edge_adjacent = shared == 2
    alignment = np.asarray(port_map.directional_alignment, dtype=float)
    olp, orp, optimal_total = optimal_port_assignment(points, left, right)
    agree = (olp == lp) & (orp == rp)
    angle = np.degrees(np.arccos(np.clip(np.sum(points[left] * points[right], axis=1), -1.0, 1.0)))
    production_three = federation.build_federation(level, "port_pair")
    three = production_three.inter_pairs
    three_key = {(int(a), int(b)): (int(p), int(q)) for a, p, b, q in three.tolist()}
    same = 0
    checked = 0
    for a, p, b, q, kind in zip(left.tolist(), lp.tolist(), right.tolist(), rp.tolist(), edge_adjacent.tolist()):
        if not kind:
            continue
        checked += 1
        if three_key.get((a, b)) == (p, q):
            same += 1
    if checked != three.shape[0]:
        raise AssertionError("edge-adjacent pairs differ from the production cell-dual graph")
    unglued_hist = np.bincount(unglued[unglued >= 0], minlength=PORTS)
    receipt = {
        "wiring": "W12",
        "cell_graph": (
            "cells of the geodesic icosahedral tower at this level; two cells are neighbours when they share a mesh "
            "vertex (three edge-adjacent and nine vertex-adjacent neighbours for a cell whose vertices all have "
            "degree six; the sixty cells touching one of the twelve pentagonal vertices have eleven neighbours)"
        ),
        "carriers": int(n),
        "glued_pairs": int(left.size),
        "edge_adjacent_pairs": int(edge_adjacent.sum()),
        "vertex_adjacent_pairs": int((~edge_adjacent).sum()),
        "neighbour_count_histogram": {str(k): int(v) for k, v in zip(*np.unique(degree, return_counts=True))},
        "pentagonal_adjacent_cells": int(geometry.pentagonal_cell.sum()),
        "cells_with_eleven_neighbours_are_pentagonal_adjacent": bool(np.array_equal(degree < PORTS, geometry.pentagonal_cell)),
        "unglued_ports": int((unglued >= 0).sum()),
        "unglued_port_histogram": unglued_hist.tolist(),
        "unglued_ports_by_cell_sha256": sha256_of([[int(c), int(p)] for c, p in enumerate(unglued.tolist()) if p >= 0]),
        "assignment_rule": (
            "oph_fpe.core.screen_ports.assign_echosahedral_ports(left, right, N, points=cell_centres) on the "
            "twelve-neighbour graph: each endpoint is routed to the port whose icosahedral template direction, read "
            "in the cell's local tangent frame (tangent_x = normalize(reference x normal) with reference (0,0,1), "
            "or (1,0,0) when |normal_z| > 0.9; tangent_y = normal x tangent_x), has the largest float32 inner "
            "product with the unit chord to the neighbour centre; collisions inside a cell are repaired by an exact "
            "maximum-alignment assignment over all of that cell's endpoints, so every cell ends with distinct ports "
            "and an eleven-neighbour cell leaves the least aligned port unglued"
        ),
        "assignment_acceptance": accepted,
        "port_template": "base icosahedron vertices (oph_fpe.core.icosahedral level 0), identical to the carrier incidence of oph_exact.carrier",
        "port_usage_histogram": np.bincount(np.concatenate([lp, rp]), minlength=PORTS).tolist(),
        "alignment": {"min": rounded(alignment.min()), "mean": rounded(alignment.mean()), "max": rounded(alignment.max())},
        "alignment_total_production": rounded(alignment.sum()),
        "alignment_total_float64_optimal_matching": rounded(optimal_total),
        "production_equals_float64_optimal_matching_fraction": rounded(agree.mean()),
        "antipodal_consistency": {
            "definition": "a glued pair ((a, p), (b, q)) is antipodally consistent when q equals carrier.antipode()[p]",
            "consistent_pairs": int(consistent.sum()),
            "fraction": rounded(consistent.mean()),
            "fraction_edge_adjacent": rounded(consistent[edge_adjacent].mean()),
            "fraction_vertex_adjacent": rounded(consistent[~edge_adjacent].mean()),
        },
        "three_port_production_gluing": {
            "convention": "federation.build_federation(level, 'port_pair'): the same rule on the cell-dual graph (three edge neighbours)",
            "inter_seams": int(three.shape[0]),
            "antipodal_consistent_fraction": production_three.gluing_receipt["antipodal_consistent_fraction"],
            "port_usage_histogram": production_three.gluing_receipt["port_usage_histogram"],
            "edge_adjacent_pairs_with_identical_ports_under_w12": int(same),
            "edge_adjacent_pairs_checked": int(checked),
            "identical_port_fraction": rounded(same / checked),
        },
        "neighbour_angle_deg": {"min": rounded(angle.min()), "mean": rounded(angle.mean()), "max": rounded(angle.max())},
        "local_frame_hash": port_map.local_frame_hash,
        "port_pairs_sha256": sha256_of(np.stack([left, lp, right, rp], axis=1).tolist()),
    }
    return Wiring(
        level=level,
        carriers=n,
        left=left,
        right=right,
        shared=shared,
        left_port=lp,
        right_port=rp,
        unglued_port=unglued,
        degree=degree,
        hop_angle_deg=float(angle.mean()),
        receipt=receipt,
    )


# --------------------------------------------------------------------------
# Federation objects (the L1 machinery on the W12 seam set)
# --------------------------------------------------------------------------


def _intra_seams(carriers: int) -> tuple[np.ndarray, np.ndarray]:
    local = np.asarray(carrier.seams(), dtype=np.int64)
    base = (np.arange(carriers, dtype=np.int64) * PORTS)[:, None]
    return (base + local[None, :, 0]).ravel(), (base + local[None, :, 1]).ravel()


def _components(seam_a: np.ndarray, seam_b: np.ndarray, ports: int) -> tuple[np.ndarray, np.ndarray, sparse.csr_matrix]:
    adjacency = sparse.coo_matrix((np.ones(seam_a.size), (seam_a, seam_b)), shape=(ports, ports))
    adjacency = (adjacency + adjacency.T).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = (sparse.diags(degree) - adjacency).tocsr()
    count, labels = csgraph.connected_components(adjacency, directed=False)
    first = np.full(count, ports, dtype=np.int64)
    np.minimum.at(first, labels, np.arange(ports, dtype=np.int64))
    relabel = np.empty(count, dtype=np.int64)
    relabel[np.argsort(first, kind="stable")] = np.arange(count, dtype=np.int64)
    labels = relabel[labels].astype(np.int64)
    return labels, np.bincount(labels, minlength=count).astype(np.int64), laplacian


@lru_cache(maxsize=8)
def build_w12_federation(level: int) -> federation.ExactFederation:
    wiring = build_wiring(level)
    n = wiring.carriers
    intra_a, intra_b = _intra_seams(n)
    inter = wiring.pairs
    seam_a = np.concatenate([intra_a, inter[:, 0] * PORTS + inter[:, 1]]).astype(np.int64)
    seam_b = np.concatenate([intra_b, inter[:, 2] * PORTS + inter[:, 3]]).astype(np.int64)
    ports = PORTS * n
    labels, sizes, laplacian = _components(seam_a, seam_b, ports)
    seam_a.setflags(write=False)
    seam_b.setflags(write=False)
    return federation.ExactFederation(
        level=level,
        gluing="w12",
        carriers=n,
        seam_a=seam_a,
        seam_b=seam_b,
        graph=federation.PortGraph.build(seam_a, seam_b, ports),
        intra_count=int(intra_a.size),
        inter_pairs=inter,
        component_of_port=labels,
        component_sizes=sizes,
        laplacian=laplacian,
        pentagonal_cell=cell_geometry(level).pentagonal_cell,
        gluing_receipt=wiring.receipt,
    )


def initial_loads(level: int, ports: int) -> np.ndarray:
    """The L1 lane's seeded loads in ``{0..5}`` for the same level."""

    return federation.initial_loads(level, ports)


def schedule_seed(level: int, index: int) -> int:
    return SCHEDULE_SEED_BASE + 1000 * level + index


def synchronous_operator(fed: federation.ExactFederation) -> dict[str, Any]:
    lap = fed.laplacian
    ports = fed.ports
    D = fed.mean_denominator
    v0 = np.random.default_rng(7).standard_normal(ports)
    if ports <= 4096:
        eig = np.linalg.eigvalsh(lap.toarray())
        lam2 = float(eig[fed.components])
        lam_max = float(eig[-1])
        method = "dense_eigvalsh"
    else:
        low = sparse_linalg.eigsh(lap, k=fed.components + 1, sigma=-1e-3, which="LM", return_eigenvectors=False, v0=v0)
        lam2 = float(np.sort(low)[-1])
        lam_max = float(sparse_linalg.eigsh(lap, k=1, which="LA", return_eigenvectors=False, v0=v0)[0])
        method = "sparse_shift_invert"
    return {
        "definition": "T_fed = I - L_fed/D with D = 2|S|/N (one attempt per carrier per tick)",
        "D": str(D),
        "laplacian_lambda_2": rounded(round(lam2, 10)),
        "laplacian_lambda_2_times_N": rounded(round(lam2 * fed.carriers, 6)),
        "laplacian_lambda_max": rounded(round(lam_max, 10)),
        "spectrum_method": method,
        "attempts_per_e_fold_of_slowest_mode": rounded(float(f"{2 * fed.seams / lam2:.6g}")) if lam2 > 0 else None,
        "sweeps_per_e_fold_of_slowest_mode": rounded(float(f"{2 / lam2:.6g}")) if lam2 > 0 else None,
        "three_port_reference_lambda_2_times_N": "0.88 (federation receipt, L = 0..3)",
    }


# --------------------------------------------------------------------------
# Schedules (float mean law and integer law) through the L1 engines
# --------------------------------------------------------------------------


def _strip(entry: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in entry.items() if k != "state"}


def run_mean_law(fed: federation.ExactFederation, loads: np.ndarray, seed: int, *, max_sweeps: int, threshold: float = federation.FLOAT_PHI_THRESHOLD) -> dict[str, Any]:
    """One asynchronous float schedule of the seam-mean law to ``Phi < threshold``.

    The schedule and the termination rule are those of ``federation.run_mean_law``
    (uniform over all seams with replacement, one sweep = ``|S|`` attempts,
    ``numpy.random.default_rng(seed).integers(0, |S|, size=|S|)`` per sweep);
    the moves are applied by the lean native kernel (mean, ``V`` ledger, waits).
    """

    lib = native_kernel()
    x = np.ascontiguousarray(loads, dtype=np.float64).copy()
    a = np.ascontiguousarray(fed.seam_a, dtype=np.int64)
    b = np.ascontiguousarray(fed.seam_b, dtype=np.int64)
    rng = np.random.default_rng(seed)
    seams = fed.seams
    phi0 = fed.mismatch_potential(x)
    v0 = fed.descent_functional(x)
    v_min = fed.mean_minimum_descent(loads)
    ledger = np.zeros(1, dtype=np.float64)
    counts = np.zeros(2, dtype=np.int64)
    sweep = 0
    phi = phi0
    trace = [[0, float(f"{phi0:.12g}")]]
    while not phi < threshold and sweep < max_sweeps:
        seq = np.ascontiguousarray(rng.integers(0, seams, size=seams, dtype=np.int64))
        lib.oph_wire_mean(_ptr(x, ctypes.c_double), _ptr(a, ctypes.c_int64), _ptr(b, ctypes.c_int64), _ptr(seq, ctypes.c_int64), seams, _ptr(ledger, ctypes.c_double), _ptr(counts, ctypes.c_int64))
        sweep += 1
        phi = fed.mismatch_potential(x)
        if sweep & (sweep - 1) == 0:
            trace.append([sweep, float(f"{phi:.12g}")])
    if trace[-1][0] != sweep:
        trace.append([sweep, float(f"{phi:.12g}")])
    v_end = fed.descent_functional(x)
    ledger_error = abs((v0 - v_end) - float(ledger[0])) / max(v0, 1.0)
    means = fed.component_means(loads)
    residual = x - means[fed.component_of_port]
    deviation = float(np.max(np.abs(residual)))
    quotient_hash, lattice_residual = fed.terminal_quotient_hash(x)
    attempts = sweep * seams
    return {
        "seed": int(seed),
        "terminated": bool(phi < threshold),
        "sweeps": int(sweep),
        "attempts": int(attempts),
        "waits": int(counts[0]),
        "non_wait_moves": int(attempts - counts[0]),
        "phi_initial": float(f"{phi0:.12g}"),
        "phi_terminal": float(f"{phi:.12g}"),
        "phi_trace": trace,
        "descent_functional_initial": rounded(v0),
        "descent_functional_terminal": rounded(v_end),
        "descent_functional_minimum": rounded(v_min),
        "descent_ledger_relative_error_below_1e-9": bool(ledger_error < 1e-9),
        "strict_descent_violations": int(counts[1]),
        "max_abs_deviation_from_component_mean": float(f"{deviation:.3g}"),
        "component_mean_within_1e-9": bool(deviation < 1e-9),
        "lattice_residual_max": float(f"{lattice_residual:.3g}"),
        "terminal_quotient_hash": quotient_hash,
    }


def _schedule_task(task: dict[str, Any]) -> dict[str, Any]:
    fed = build_w12_federation(task["level"])
    loads = initial_loads(task["level"], fed.ports)
    started = time.perf_counter()
    if task["law"] == "float":
        result = run_mean_law(fed, loads, task["seed"], max_sweeps=task["max_sweeps"])
    else:
        result = _strip(federation.run_integer_law(fed, loads, task["seed"], engine=task["engine"], max_sweeps=task["max_sweeps"]))
    return {"task": task, "result": result, "seconds": time.perf_counter() - started}


def _kernel_task(task: dict[str, Any]) -> dict[str, Any]:
    fed = build_w12_federation(task["level"])
    started = time.perf_counter()
    kernels = response_kernels_batch(fed, task["cells"])
    rows = [kernel_rows(fed, cell, kernels[cell]) for cell in task["cells"]]
    return {"task": task, "rows": rows, "seconds": time.perf_counter() - started}


def _spectrum_task(task: dict[str, Any]) -> dict[str, Any]:
    fed = build_w12_federation(task["level"])
    started = time.perf_counter()
    return {"task": task, "spectrum": synchronous_operator(fed), "seconds": time.perf_counter() - started}


def _dispatch(task: dict[str, Any]) -> dict[str, Any]:
    if task["kind"] == "schedule":
        return _schedule_task(task)
    if task["kind"] == "kernel":
        return _kernel_task(task)
    if task["kind"] == "spectrum":
        return _spectrum_task(task)
    raise ValueError(task["kind"])


class TaskRunner:
    """Runs pool tasks in worker processes while the caller keeps working."""

    def __init__(self, workers: int) -> None:
        self.workers = workers
        self.pool = None
        if workers > 1:
            import concurrent.futures as futures
            import multiprocessing as mp

            federation.native_kernel()
            self.pool = futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))

    def submit(self, tasks: list[dict[str, Any]]) -> Any:
        if self.pool is None:
            return [_dispatch(t) for t in tasks]
        return self.pool.map(_dispatch, tasks, chunksize=1)

    @staticmethod
    def collect(handle: Any) -> list[dict[str, Any]]:
        return list(handle)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None


def run_tasks(tasks: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    runner = TaskRunner(workers)
    try:
        return runner.collect(runner.submit(tasks))
    finally:
        runner.close()


def aggregate_float(entries: list[dict[str, Any]], fed: federation.ExactFederation, loads: np.ndarray, budget: int | None) -> dict[str, Any]:
    entries = sorted(entries, key=lambda e: e["seed"])
    expected = fed.expected_terminal_quotient_hash(loads)
    hashes = sorted({e["terminal_quotient_hash"] for e in entries})
    block = {
        "schedules": len(entries),
        "schedule": "uniform over all seams with replacement, one sweep = |S| attempts (the L1 schedule law, applied by the lean native seam-mean kernel)",
        "termination": "first sweep end with Phi < 1e-18" if budget is None else f"declared budget of {budget} sweeps",
        "all_terminated": bool(all(e["terminated"] for e in entries)),
        "unique_terminal_hash_count": len(hashes),
        "terminal_hashes_identical_across_schedules": bool(len(hashes) == 1),
        "terminal_hash_equals_component_mean": bool(len(hashes) == 1 and hashes[0] == expected),
        "expected_terminal_hash": expected,
        "attempts": stats([e["attempts"] for e in entries]),
        "strict_descent_violations_total": int(sum(e["strict_descent_violations"] for e in entries)),
        "descent_ledger_ok_all": bool(all(e["descent_ledger_relative_error_below_1e-9"] for e in entries)),
        "waits": stats([e["waits"] for e in entries]),
        "max_abs_deviation_from_component_mean_max": rounded(max(e["max_abs_deviation_from_component_mean"] for e in entries)),
        "component_mean_within_1e-9_all": bool(all(e["component_mean_within_1e-9"] for e in entries)),
        "entries": entries,
    }
    if budget is not None:
        block["phi_initial"] = entries[0]["phi_initial"]
        block["phi_terminal"] = stats([e["phi_terminal"] for e in entries])
    return block


def aggregate_integer(entries: list[dict[str, Any]], fed: federation.ExactFederation, loads: np.ndarray) -> dict[str, Any]:
    entries = sorted(entries, key=lambda e: e["seed"])
    expected = fed.expected_integer_quotient_hash(loads)
    hashes = sorted({e["quotient_hash"] for e in entries})
    return {
        "schedules": len(entries),
        "termination": "first attempt after which V equals the balanced-class minimum (federation.run_integer_law)",
        "all_terminated": bool(all(e["terminated"] for e in entries)),
        "unique_quotient_hash_count": len(hashes),
        "quotient_hash_equals_expected_multiset": bool(len(hashes) == 1 and hashes[0] == expected),
        "attempts_to_balanced_class": stats([e["attempts_to_balanced_class"] for e in entries]),
        "sweeps": stats([e["sweeps"] for e in entries]),
        "unit_transfer_decrement_identity_violations_total": int(sum(e["unit_transfer_decrement_identity_violations"] for e in entries)),
        "descent_ledger_exact_all": bool(all(e["descent_ledger_exact"] for e in entries)),
        "odd_tie_seams_at_termination": stats([e["odd_tie_seams_at_termination"] for e in entries]),
        "entries": entries,
    }


# --------------------------------------------------------------------------
# Slow band under W12
# --------------------------------------------------------------------------


def response_kernels_batch(fed: federation.ExactFederation, cells: Sequence[int], steps: Sequence[int] = KERNEL_STEPS) -> dict[int, dict[int, np.ndarray]]:
    """``K_n = 12 C_n / tr C_n`` for several carriers propagated together.

    Identical arithmetic to ``federation.response_kernels`` per column block
    (each carrier's twelve centered impulses are rescaled by their own
    maximum every step).
    """

    ports = fed.ports
    cells = [int(c) for c in cells]
    y = np.zeros((ports, PORTS * len(cells)))
    q = np.eye(PORTS) - np.ones((PORTS, PORTS)) / PORTS
    for k, cell in enumerate(cells):
        y[cell * PORTS : (cell + 1) * PORTS, k * PORTS : (k + 1) * PORTS] = q
    target = {2 * n: n for n in steps}
    last = max(target)
    out: dict[int, dict[int, np.ndarray]] = {cell: {} for cell in cells}
    for step in range(1, last + 1):
        y = fed.apply_synchronous(y)
        for k in range(len(cells)):
            block = y[:, k * PORTS : (k + 1) * PORTS]
            scale = np.max(np.abs(block))
            if scale > 0:
                block /= scale
        if step in target:
            for k, cell in enumerate(cells):
                c = q @ y[cell * PORTS : (cell + 1) * PORTS, k * PORTS : (k + 1) * PORTS]
                c = 0.5 * (c + c.T)
                out[cell][target[step]] = PORTS * c / np.trace(c)
    return out


def kernel_rows(fed: federation.ExactFederation, cell: int, kernels: dict[int, np.ndarray]) -> dict[str, Any]:
    p_slow = carrier.slow_band_projector()
    gram = carrier.intrinsic_gram()
    rows = []
    for n in KERNEL_STEPS:
        k = kernels[n]
        eig = np.sort(np.linalg.eigvalsh(k))[::-1]
        share = float(np.trace(p_slow @ k @ p_slow) / np.trace(k))
        rows.append(
            {
                "n": n,
                "slow_band_share": rounded(round(share, 9)),
                "top_four_eigenvalues": [rounded(round(float(v), 9)) for v in eig[:4]],
                "rank_1e-9": int(np.linalg.matrix_rank(k, tol=1e-9)),
                "max_abs_deviation_from_intrinsic_gram": rounded(float(f"{np.max(np.abs(k - gram)):.4g}")),
                "max_abs_deviation_from_isolated_kernel": rounded(float(f"{np.max(np.abs(k - carrier.normalized_response_kernel(n))):.4g}")),
            }
        )
    return {"cell": int(cell), "pentagonal": bool(fed.pentagonal_cell[cell]), "per_step": rows}


def kernel_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for index, n in enumerate(KERNEL_STEPS):
        shares = [r["per_step"][index]["slow_band_share"] for r in rows]
        tops = np.asarray([r["per_step"][index]["top_four_eigenvalues"] for r in rows])
        summary[str(n)] = {
            "slow_band_share": stats(shares),
            "top_four_eigenvalues_median": [rounded(v) for v in np.median(tops, axis=0)],
            "rank_1e-9_histogram": {str(k): int(v) for k, v in zip(*np.unique([r["per_step"][index]["rank_1e-9"] for r in rows], return_counts=True))},
        }
    return summary


def sample_kernel_cells(fed: federation.ExactFederation, count: int | None = None) -> list[int]:
    return federation.sample_cells(fed, KERNEL_SAMPLE.get(fed.level, 64) if count is None else count)


def isolated_reference(level: int = 3, cells: Sequence[int] = (0, 41, 700, 1279)) -> dict[str, Any]:
    fed = federation.build_federation(level, "isolated")
    kernels = response_kernels_batch(fed, cells)
    dev = 0.0
    gram_dev = 0.0
    shares = {n: [] for n in KERNEL_STEPS}
    p_slow = carrier.slow_band_projector()
    for cell in cells:
        for n in KERNEL_STEPS:
            k = kernels[cell][n]
            dev = max(dev, float(np.max(np.abs(k - carrier.normalized_response_kernel(n)))))
            shares[n].append(float(np.trace(p_slow @ k @ p_slow) / np.trace(k)))
        gram_dev = max(gram_dev, float(np.max(np.abs(kernels[cell][300] - carrier.intrinsic_gram()))))
    return {
        "level": level,
        "cells": [int(c) for c in cells],
        "max_abs_deviation_from_carrier_kernel": rounded(float(f"{dev:.3g}")),
        "kernel_matches_carrier_within_1e-12": bool(dev < 1e-12),
        "max_abs_deviation_from_intrinsic_gram_at_n_300": rounded(float(f"{gram_dev:.3g}")),
        "gram_within_1e-12_at_n_300": bool(gram_dev < 1e-12),
        "slow_band_share": {str(n): rounded(round(float(np.median(shares[n])), 9)) for n in KERNEL_STEPS},
        "exact_single_carrier_share": {
            str(n): rounded(round(float(np.trace(p_slow @ carrier.normalized_response_kernel(n) @ p_slow) / np.trace(carrier.normalized_response_kernel(n))), 9))
            for n in KERNEL_STEPS
        },
    }


def three_port_reference(level: int, cells: Sequence[int]) -> dict[str, Any]:
    fed = federation.build_federation(level, "port_pair")
    rows = []
    for start in range(0, len(cells), KERNEL_BATCH):
        batch = list(cells[start : start + KERNEL_BATCH])
        kernels = response_kernels_batch(fed, batch)
        rows.extend(kernel_rows(fed, cell, kernels[cell]) for cell in batch)
    return {"level": level, "gluing": "port_pair (three of twelve ports)", "sampled_cells": [int(c) for c in cells], "summary": kernel_summary(rows)}


# --------------------------------------------------------------------------
# Native kernel for the provenance run and the order statistics
# --------------------------------------------------------------------------

_KERNEL_SOURCE = r"""
#include <stdint.h>

/* Integer nearest-agreement law over an explicit schedule of seam indices.
   outcome[t]: 0 wait, 1 swap (unit-difference seam, coin placed the ceiling on
   the other endpoint), 2 descent (|d| >= 2).  When snap_slot[t] >= 0 the
   twelve loads of both endpoint carriers after the move are copied to
   snap[slot * 2 * ppc ...].  Returns the change of V = sum x^2. */
int64_t oph_wire_integer(int64_t *x, const int64_t *a, const int64_t *b, const int64_t *seq,
                         const int64_t *coin, int64_t n, int8_t *outcome,
                         const int64_t *snap_slot, int8_t *snap, int64_t ppc)
{
    int64_t dv = 0;
    for (int64_t t = 0; t < n; ++t) {
        const int64_t s = seq[t];
        const int64_t i = a[s], j = b[s];
        const int64_t xi = x[i], xj = x[j];
        const int64_t tot = xi + xj;
        const int64_t lo = tot >> 1, hi = tot - lo;
        const int64_t ni = coin[t] ? hi : lo, nj = tot - ni;
        const int64_t d = xi - xj;
        int8_t o;
        if (d == 0) o = 0;
        else if (d == 1 || d == -1) o = (ni == xi) ? 0 : 1;
        else o = 2;
        if (o) {
            if (o == 2) dv += ni * ni + nj * nj - xi * xi - xj * xj;
            x[i] = ni;
            x[j] = nj;
        }
        outcome[t] = o;
        if (snap_slot != 0) {
            const int64_t slot = snap_slot[t];
            if (slot >= 0) {
                int8_t *dst = snap + slot * 2 * ppc;
                const int64_t ca = (i / ppc) * ppc, cb = (j / ppc) * ppc;
                for (int64_t k = 0; k < ppc; ++k) {
                    dst[k] = (int8_t)x[ca + k];
                    dst[ppc + k] = (int8_t)x[cb + k];
                }
            }
        }
    }
    return dv;
}

/* Seam-mean law over a schedule: both endpoints replaced by their mean.
   counts = {waits, descent violations}; ledger accumulates (x_i - x_j)^2 / 2. */
void oph_wire_mean(double *x, const int64_t *a, const int64_t *b, const int64_t *seq, int64_t n,
                   double *ledger, int64_t *counts)
{
    int64_t waits = 0, violations = 0;
    double led = 0.0;
    for (int64_t t = 0; t < n; ++t) {
        const int64_t s = seq[t];
        const int64_t i = a[s], j = b[s];
        const double xi = x[i], xj = x[j];
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

/* Forward reachability from lo (reach[lo] preset) over parents pa, pb (< t). */
void oph_wire_forward(const int64_t *pa, const int64_t *pb, int64_t lo, int64_t hi, uint8_t *reach)
{
    for (int64_t t = lo + 1; t <= hi; ++t) {
        uint8_t r = 0;
        if (pa[t] >= lo && reach[pa[t]]) r = 1;
        else if (pb[t] >= lo && reach[pb[t]]) r = 1;
        reach[t] = r;
    }
}

/* Backward reachability from hi (reach[hi] preset). */
void oph_wire_backward(const int64_t *pa, const int64_t *pb, int64_t lo, int64_t hi, uint8_t *reach)
{
    for (int64_t t = hi; t >= lo; --t) {
        if (!reach[t]) continue;
        if (pa[t] >= lo) reach[pa[t]] = 1;
        if (pb[t] >= lo) reach[pb[t]] = 1;
    }
}

/* Ancestor bitsets over a DAG in topological order with CSR parents (local ids).
   slot[v] >= 0 marks the sources (bit position); bits has n * words entries,
   zeroed by the caller.  ancestors[v] = popcount(row v).  Returns their sum. */
int64_t oph_wire_ancestors(int64_t n, int64_t words, const int64_t *indptr, const int64_t *indices,
                           const int64_t *slot, uint64_t *bits, int64_t *ancestors)
{
    int64_t total = 0;
    for (int64_t v = 0; v < n; ++v) {
        uint64_t *row = bits + v * words;
        for (int64_t k = indptr[v]; k < indptr[v + 1]; ++k) {
            const uint64_t *src = bits + indices[k] * words;
            for (int64_t w = 0; w < words; ++w) row[w] |= src[w];
        }
        const int64_t s = slot[v];
        if (s >= 0) row[s >> 6] |= (uint64_t)1 << (s & 63);
        int64_t c = 0;
        for (int64_t w = 0; w < words; ++w) c += __builtin_popcountll(row[w]);
        ancestors[v] = c;
        total += c;
    }
    return total;
}

/* counts[bit] += 1 for every row containing the bit. */
void oph_wire_column_counts(int64_t n, int64_t words, const uint64_t *bits, int64_t *counts)
{
    for (int64_t v = 0; v < n; ++v) {
        const uint64_t *row = bits + v * words;
        for (int64_t w = 0; w < words; ++w) {
            uint64_t x = row[w];
            while (x) {
                const int k = __builtin_ctzll(x);
                counts[w * 64 + k] += 1;
                x &= x - 1;
            }
        }
    }
}

/* Cover test for parent edges whose parent p has slot[p] >= 0: the edge
   (p -> v) is a cover unless another parent of v has p among its ancestors. */
void oph_wire_covers(int64_t n, int64_t words, const int64_t *indptr, const int64_t *indices,
                     const int64_t *slot, const uint64_t *bits, uint8_t *is_cover)
{
    for (int64_t v = 0; v < n; ++v) {
        for (int64_t k = indptr[v]; k < indptr[v + 1]; ++k) {
            const int64_t p = indices[k];
            const int64_t s = slot[p];
            if (s < 0) continue;
            uint8_t cover = 1;
            for (int64_t m = indptr[v]; m < indptr[v + 1]; ++m) {
                const int64_t q = indices[m];
                if (q == p) continue;
                if (bits[q * words + (s >> 6)] & ((uint64_t)1 << (s & 63))) { cover = 0; break; }
            }
            is_cover[k] = cover;
        }
    }
}

/* Longest chain ending at v, counted in events. */
void oph_wire_depth(int64_t n, const int64_t *indptr, const int64_t *indices, int64_t *depth)
{
    for (int64_t v = 0; v < n; ++v) {
        int64_t best = 0;
        for (int64_t k = indptr[v]; k < indptr[v + 1]; ++k) {
            const int64_t d = depth[indices[k]];
            if (d > best) best = d;
        }
        depth[v] = best + 1;
    }
}
"""

_NATIVE: dict[str, Any] = {"lib": None, "tried": False, "error": None}


def _kernel_dir() -> Path:
    root = os.environ.get("OPH_EXACT_KERNEL_DIR")
    return Path(root) if root else Path(tempfile.gettempdir()) / "oph_exact_kernels"


def native_kernel() -> Any:
    """Compile (once) and load the wiring kernel; raises when no compiler is available."""

    if _NATIVE["tried"]:
        if _NATIVE["lib"] is None:
            raise RuntimeError(f"native wiring kernel unavailable: {_NATIVE['error']}")
        return _NATIVE["lib"]
    _NATIVE["tried"] = True
    try:
        digest = hashlib.sha256(_KERNEL_SOURCE.encode("utf-8")).hexdigest()[:16]
        directory = _kernel_dir()
        directory.mkdir(parents=True, exist_ok=True)
        suffix = ".dll" if sys.platform == "win32" else ".so"
        library = directory / f"oph_wire_kernel_{digest}{suffix}"
        if not library.exists():
            compiler = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
            if compiler is None:
                raise RuntimeError("no C compiler on PATH")
            source = directory / f"oph_wire_kernel_{digest}.c"
            source.write_text(_KERNEL_SOURCE, encoding="utf-8")
            staging = directory / f"oph_wire_kernel_{digest}.{os.getpid()}{suffix}"
            subprocess.run([compiler, "-O2", "-shared", "-fPIC", "-o", str(staging), str(source)], check=True, capture_output=True)
            os.replace(staging, library)
        lib = ctypes.CDLL(str(library))
        i64 = ctypes.c_int64
        i64p = ctypes.POINTER(ctypes.c_int64)
        i8p = ctypes.POINTER(ctypes.c_int8)
        u8p = ctypes.POINTER(ctypes.c_uint8)
        u64p = ctypes.POINTER(ctypes.c_uint64)
        lib.oph_wire_integer.argtypes = [i64p, i64p, i64p, i64p, i64p, i64, i8p, i64p, i8p, i64]
        lib.oph_wire_integer.restype = i64
        lib.oph_wire_mean.argtypes = [ctypes.POINTER(ctypes.c_double), i64p, i64p, i64p, i64, ctypes.POINTER(ctypes.c_double), i64p]
        lib.oph_wire_mean.restype = None
        lib.oph_wire_forward.argtypes = [i64p, i64p, i64, i64, u8p]
        lib.oph_wire_forward.restype = None
        lib.oph_wire_backward.argtypes = [i64p, i64p, i64, i64, u8p]
        lib.oph_wire_backward.restype = None
        lib.oph_wire_ancestors.argtypes = [i64, i64, i64p, i64p, i64p, u64p, i64p]
        lib.oph_wire_ancestors.restype = i64
        lib.oph_wire_column_counts.argtypes = [i64, i64, u64p, i64p]
        lib.oph_wire_column_counts.restype = None
        lib.oph_wire_covers.argtypes = [i64, i64, i64p, i64p, i64p, u64p, u8p]
        lib.oph_wire_covers.restype = None
        lib.oph_wire_depth.argtypes = [i64, i64p, i64p, i64p]
        lib.oph_wire_depth.restype = None
        _NATIVE["lib"] = lib
        return lib
    except Exception as error:  # pragma: no cover - environment dependent
        _NATIVE["error"] = f"{type(error).__name__}: {error}"
        raise


def _ptr(array: np.ndarray, ctype: Any) -> Any:
    return array.ctypes.data_as(ctypes.POINTER(ctype))


# --------------------------------------------------------------------------
# Event systems: seams, carriers, screen positions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EventSystem:
    """All seams of a federation with the external (event-generating) seams last."""

    name: str
    carriers: int
    seam_a: np.ndarray
    seam_b: np.ndarray
    intra_count: int
    ext_left: np.ndarray  # carrier of the first endpoint of each external seam
    ext_right: np.ndarray
    ext_kind: np.ndarray  # 0 W12 glued seam, 1 join seam
    cell_points: np.ndarray  # unit vector per carrier
    cell_level: np.ndarray  # tower level per carrier
    cell_local: np.ndarray  # cell index inside its level
    pentagonal_cell: np.ndarray
    hop_angle_deg: float  # finest-level mean neighbour angle
    levels: tuple[int, ...]
    description: str

    @property
    def ports(self) -> int:
        return PORTS * self.carriers

    @property
    def seams(self) -> int:
        return int(self.seam_a.size)

    @property
    def external(self) -> int:
        return int(self.ext_left.size)

    def seam_midpoints(self) -> np.ndarray:
        mid = self.cell_points[self.ext_left] + self.cell_points[self.ext_right]
        return mid / np.linalg.norm(mid, axis=1, keepdims=True)


def single_level_system(level: int) -> EventSystem:
    wiring = build_wiring(level)
    fed = build_w12_federation(level)
    geometry = cell_geometry(level)
    return EventSystem(
        name=f"W12/L{level}",
        carriers=wiring.carriers,
        seam_a=fed.seam_a,
        seam_b=fed.seam_b,
        intra_count=fed.intra_count,
        ext_left=wiring.left,
        ext_right=wiring.right,
        ext_kind=np.zeros(wiring.edges, dtype=np.int64),
        cell_points=geometry.points,
        cell_level=np.full(wiring.carriers, level, dtype=np.int64),
        cell_local=np.arange(wiring.carriers, dtype=np.int64),
        pentagonal_cell=geometry.pentagonal_cell,
        hop_angle_deg=wiring.hop_angle_deg,
        levels=(level,),
        description="single tower level with the W12 wiring; the events are the glued-seam repair attempts",
    )


def depth_system(top_level: int) -> EventSystem:
    """Levels ``(L-2, L-1, L)`` with the committed join transport as declared inter-level seams.

    Every child port ``k`` is glued to its parent's port ``k`` (twelve join
    seams per child cell).  The seam law on a join seam is the same mean
    (float) or nearest-agreement (integer) law as on every other seam, so the
    stationary state of a parent port is the plain average of its four
    children's ports; the committed conditional expectation weights are the
    spherical area fractions of the children, whose deviation from ``1/4`` is
    recorded, and enter no repair here.  Each seam is attempted once per
    round.
    """

    levels = tuple(range(top_level - 2, top_level + 1))
    offsets = {}
    total = 0
    for level in levels:
        offsets[level] = total
        total += 20 * 4**level
    intra_a, intra_b = _intra_seams(total)
    ext_left = []
    ext_right = []
    ext_kind = []
    port_a = []
    port_b = []
    points = []
    cell_level = []
    cell_local = []
    pentagonal = []
    weight_deviation = 0.0
    for level in levels:
        wiring = build_wiring(level)
        geometry = cell_geometry(level)
        off = offsets[level]
        ext_left.append(wiring.left + off)
        ext_right.append(wiring.right + off)
        ext_kind.append(np.zeros(wiring.edges, dtype=np.int64))
        port_a.append(wiring.left_port)
        port_b.append(wiring.right_port)
        points.append(geometry.points)
        cell_level.append(np.full(wiring.carriers, level, dtype=np.int64))
        cell_local.append(np.arange(wiring.carriers, dtype=np.int64))
        pentagonal.append(geometry.pentagonal_cell)
        if level > levels[0]:
            parent = geometry.parent
            weight_deviation = max(weight_deviation, float(np.max(np.abs(geometry.expectation_weight - 0.25))))
            child = np.arange(wiring.carriers, dtype=np.int64)
            for k in range(PORTS):
                ext_left.append(child + off)
                ext_right.append(parent + offsets[level - 1])
                ext_kind.append(np.ones(child.size, dtype=np.int64))
                port_a.append(np.full(child.size, k, dtype=np.int64))
                port_b.append(np.full(child.size, k, dtype=np.int64))
    ext_left_arr = np.concatenate(ext_left)
    ext_right_arr = np.concatenate(ext_right)
    pa = np.concatenate(port_a)
    pb = np.concatenate(port_b)
    seam_a = np.concatenate([intra_a, ext_left_arr * PORTS + pa]).astype(np.int64)
    seam_b = np.concatenate([intra_b, ext_right_arr * PORTS + pb]).astype(np.int64)
    return EventSystem(
        name=f"depth/L{top_level}",
        carriers=total,
        seam_a=seam_a,
        seam_b=seam_b,
        intra_count=int(intra_a.size),
        ext_left=ext_left_arr,
        ext_right=ext_right_arr,
        ext_kind=np.concatenate(ext_kind),
        cell_points=np.concatenate(points),
        cell_level=np.concatenate(cell_level),
        cell_local=np.concatenate(cell_local),
        pentagonal_cell=np.concatenate(pentagonal),
        hop_angle_deg=build_wiring(top_level).hop_angle_deg,
        levels=levels,
        description=(
            f"levels {levels} with the W12 wiring on every level and twelve join seams per child cell "
            f"(child port k glued to parent port k); maximal deviation of the committed area weights from 1/4: "
            f"{weight_deviation:.4g}; the events are the W12 glued-seam and join-seam repair attempts"
        ),
    )


# --------------------------------------------------------------------------
# The provenance run
# --------------------------------------------------------------------------


@dataclass
class ProvenanceLog:
    system: EventSystem
    rounds: int
    seed: int
    seam: np.ndarray  # external seam index per attempt (execution order)
    outcome: np.ndarray  # int8: 0 wait, 1 swap, 2 descent
    round_of: np.ndarray
    per_round: list[dict[str, Any]]
    v_initial: int
    v_terminal: int
    v_minimum: int
    ledger_exact: bool
    descent_violations: int
    terminal_state_sha256: str
    loads_sha256: str
    balanced_round: int | None
    snapshots: np.ndarray | None = None  # (slots, 24) int8 after the move, or None
    snapshot_slot: np.ndarray | None = None  # attempt id -> slot or -1

    @property
    def attempts(self) -> int:
        return int(self.seam.size)

    @property
    def cell_a(self) -> np.ndarray:
        return self.system.ext_left[self.seam]

    @property
    def cell_b(self) -> np.ndarray:
        return self.system.ext_right[self.seam]

    def content_sha256(self) -> str:
        return sha256_arrays(self.seam.astype(np.int32), self.outcome.astype(np.int8))


def run_provenance(system: EventSystem, rounds: int, seed: int, loads: np.ndarray, snapshot_ids: np.ndarray | None = None) -> ProvenanceLog:
    """Integer law over ``rounds`` random-permutation sweeps of all seams, logging the external attempts.

    Schedule: ``rng = numpy.random.Generator(numpy.random.PCG64(seed))``; per
    round ``seq = rng.permutation(|S|)`` then ``coin = rng.integers(0, 2,
    size=|S|)``.  Intra-carrier seams are repaired within the round and are
    carrier-internal (no event); every external attempt is logged in execution
    order with its outcome.
    """

    lib = native_kernel()
    a = np.ascontiguousarray(system.seam_a, dtype=np.int64)
    b = np.ascontiguousarray(system.seam_b, dtype=np.int64)
    x = np.ascontiguousarray(loads, dtype=np.int64).copy()
    if np.any(x < 0) or np.any(x > 127):
        raise ValueError("loads must fit the int8 snapshot range")
    n_seams = system.seams
    ext = system.external
    intra = system.intra_count
    rng = np.random.Generator(np.random.PCG64(seed))
    v = int(np.dot(x, x))
    v0 = v
    labels, sizes, _ = _components(a, b, system.ports)
    totals = np.bincount(labels, weights=x.astype(float), minlength=sizes.size)
    v_min = 0
    for c in range(sizes.size):
        m = int(sizes[c])
        q, r = divmod(int(round(float(totals[c]))), m)
        v_min += (m - r) * q * q + r * (q + 1) * (q + 1)
    slot_of = None
    snapshots = None
    if snapshot_ids is not None:
        slot_of = np.full(rounds * ext, -1, dtype=np.int64)
        slot_of[np.asarray(snapshot_ids, dtype=np.int64)] = np.arange(len(snapshot_ids), dtype=np.int64)
        snapshots = np.zeros((len(snapshot_ids), 2 * PORTS), dtype=np.int8)
    seams_out = np.empty(rounds * ext, dtype=np.int32)
    outcomes = np.empty(rounds * ext, dtype=np.int8)
    per_round = []
    balanced_round = None
    violations = 0
    null_i64 = ctypes.POINTER(ctypes.c_int64)()
    null_i8 = ctypes.POINTER(ctypes.c_int8)()
    for r in range(rounds):
        seq = np.ascontiguousarray(rng.permutation(n_seams), dtype=np.int64)
        coin = np.ascontiguousarray(rng.integers(0, 2, size=n_seams), dtype=np.int64)
        outcome = np.zeros(n_seams, dtype=np.int8)
        is_ext = seq >= intra
        if slot_of is not None:
            position = np.cumsum(is_ext) - 1
            global_id = r * ext + position
            snap_slot = np.ascontiguousarray(np.where(is_ext, slot_of[np.where(is_ext, global_id, 0)], -1), dtype=np.int64)
            slot_ptr = _ptr(snap_slot, ctypes.c_int64)
            snap_ptr = _ptr(snapshots, ctypes.c_int8)
        else:
            slot_ptr = null_i64
            snap_ptr = null_i8
        dv = lib.oph_wire_integer(
            _ptr(x, ctypes.c_int64), _ptr(a, ctypes.c_int64), _ptr(b, ctypes.c_int64), _ptr(seq, ctypes.c_int64),
            _ptr(coin, ctypes.c_int64), n_seams, _ptr(outcome, ctypes.c_int8), slot_ptr, snap_ptr, PORTS,
        )
        v_before = v
        v += int(dv)
        if v > v_before:
            violations += 1
        ext_out = outcome[is_ext]
        seams_out[r * ext : (r + 1) * ext] = (seq[is_ext] - intra).astype(np.int32)
        outcomes[r * ext : (r + 1) * ext] = ext_out
        all_out = outcome
        per_round.append(
            {
                "round": r,
                "external_waits": int(np.count_nonzero(ext_out == 0)),
                "external_swaps": int(np.count_nonzero(ext_out == 1)),
                "external_descents": int(np.count_nonzero(ext_out == 2)),
                "external_write_fraction": rounded(round(float(np.mean(ext_out > 0)), 6)),
                "intra_write_fraction": rounded(round(float(np.mean(all_out[~is_ext] > 0)), 6)),
                "descent_functional_after_round": int(v),
            }
        )
        if balanced_round is None and v == v_min:
            balanced_round = r
    v_state = int(np.dot(x, x))
    return ProvenanceLog(
        system=system,
        rounds=rounds,
        seed=seed,
        seam=seams_out,
        outcome=outcomes,
        round_of=(np.arange(rounds * ext, dtype=np.int64) // ext),
        per_round=per_round,
        v_initial=v0,
        v_terminal=v_state,
        v_minimum=int(v_min),
        ledger_exact=bool(v == v_state),
        descent_violations=int(violations),
        terminal_state_sha256=sha256_arrays(x.astype(np.int64)),
        loads_sha256=sha256_of(np.asarray(loads).tolist()),
        balanced_round=balanced_round,
        snapshots=snapshots,
        snapshot_slot=slot_of,
    )


# --------------------------------------------------------------------------
# The read-after-write provenance rule
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    parent_a: np.ndarray  # writer of the version read on the first carrier, -1 for the initial version
    parent_b: np.ndarray
    version_read_a: np.ndarray
    version_read_b: np.ndarray
    writes: np.ndarray  # bool per attempt

    @property
    def version_written_a(self) -> np.ndarray:
        return self.version_read_a + self.writes.astype(np.int64)

    @property
    def version_written_b(self) -> np.ndarray:
        return self.version_read_b + self.writes.astype(np.int64)


def provenance_from_log(cell_a: np.ndarray, cell_b: np.ndarray, writes: np.ndarray, carriers: int) -> Provenance:
    """The precedence generators from the log alone.

    Every carrier record starts at version zero (a distinguished root with no
    writer).  Attempt ``t`` reads both endpoint records at their current
    versions; when it writes, both versions advance by one and ``t`` becomes
    the writer of the new versions.  The direct parents of ``t`` are the
    writers of the two versions it read.
    """

    T = int(cell_a.size)
    writes = np.asarray(writes, dtype=bool)
    ends = np.concatenate([cell_a, cell_b]).astype(np.int64)
    tt = np.concatenate([np.arange(T, dtype=np.int64), np.arange(T, dtype=np.int64)])
    order = np.lexsort((tt, ends))
    e_s = ends[order]
    t_s = tt[order]
    start = np.r_[True, e_s[1:] != e_s[:-1]]
    gid = np.cumsum(start) - 1
    marker = np.where(writes[t_s], t_s, -1)
    key = marker + gid * (T + 1)
    inclusive = np.maximum.accumulate(key) - gid * (T + 1)
    exclusive = np.empty_like(inclusive)
    exclusive[1:] = inclusive[:-1]
    exclusive[start] = -1
    wcount = np.cumsum(writes[t_s].astype(np.int64))
    group_first = np.flatnonzero(start)
    base = np.repeat(wcount[group_first] - writes[t_s][group_first].astype(np.int64), np.diff(np.r_[group_first, 2 * T]))
    inclusive_count = wcount - base
    read_version = inclusive_count - writes[t_s].astype(np.int64)
    prev = np.empty(2 * T, dtype=np.int64)
    prev[order] = exclusive
    version = np.empty(2 * T, dtype=np.int64)
    version[order] = read_version
    return Provenance(parent_a=prev[:T], parent_b=prev[T:], version_read_a=version[:T], version_read_b=version[T:], writes=writes)


def provenance_rule_reference(cell_a: Sequence[int], cell_b: Sequence[int], writes: Sequence[bool], carriers: int) -> dict[str, list[int]]:
    """Plain sequential statement of the rule (the reference used by the tests)."""

    last_writer = [-1] * carriers
    version = [0] * carriers
    out = {"parent_a": [], "parent_b": [], "version_read_a": [], "version_read_b": [], "version_written_a": [], "version_written_b": []}
    for t, (a, b, w) in enumerate(zip(cell_a, cell_b, writes)):
        out["parent_a"].append(last_writer[a])
        out["parent_b"].append(last_writer[b])
        out["version_read_a"].append(version[a])
        out["version_read_b"].append(version[b])
        if w:
            version[a] += 1
            version[b] += 1
            last_writer[a] = t
            last_writer[b] = t
        out["version_written_a"].append(version[a])
        out["version_written_b"].append(version[b])
    return out


# --------------------------------------------------------------------------
# Intervals and their statistics
# --------------------------------------------------------------------------


def reach_forward(prov: Provenance, lo: int, hi: int) -> np.ndarray:
    lib = native_kernel()
    reach = np.zeros(hi + 1, dtype=np.uint8)
    reach[lo] = 1
    pa = np.ascontiguousarray(prov.parent_a, dtype=np.int64)
    pb = np.ascontiguousarray(prov.parent_b, dtype=np.int64)
    lib.oph_wire_forward(_ptr(pa, ctypes.c_int64), _ptr(pb, ctypes.c_int64), int(lo), int(hi), _ptr(reach, ctypes.c_uint8))
    return reach


def reach_backward(prov: Provenance, lo: int, hi: int) -> np.ndarray:
    lib = native_kernel()
    reach = np.zeros(hi + 1, dtype=np.uint8)
    reach[hi] = 1
    pa = np.ascontiguousarray(prov.parent_a, dtype=np.int64)
    pb = np.ascontiguousarray(prov.parent_b, dtype=np.int64)
    lib.oph_wire_backward(_ptr(pa, ctypes.c_int64), _ptr(pb, ctypes.c_int64), int(lo), int(hi), _ptr(reach, ctypes.c_uint8))
    return reach


def interval_ids(prov: Provenance, bottom: int, top: int) -> np.ndarray:
    fwd = reach_forward(prov, bottom, top)
    bwd = reach_backward(prov, bottom, top)
    return np.flatnonzero(fwd.astype(bool) & bwd.astype(bool)).astype(np.int64)


def local_parents(prov: Provenance, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CSR parents (local indices) of the interval nodes in topological order."""

    lo = int(ids[0])
    local = np.full(int(ids[-1]) - lo + 1, -1, dtype=np.int64)
    local[ids - lo] = np.arange(ids.size, dtype=np.int64)
    pa = prov.parent_a[ids]
    pb = prov.parent_b[ids]
    la = np.where(pa >= lo, local[np.where(pa >= lo, pa - lo, 0)], -1)
    lb = np.where(pb >= lo, local[np.where(pb >= lo, pb - lo, 0)], -1)
    same = la == lb
    lb = np.where(same, -1, lb)
    counts = (la >= 0).astype(np.int64) + (lb >= 0).astype(np.int64)
    indptr = np.zeros(ids.size + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)
    indices = np.empty(int(indptr[-1]), dtype=np.int64)
    pos = indptr[:-1].copy()
    mask = la >= 0
    indices[pos[mask]] = la[mask]
    pos[mask] += 1
    mask = lb >= 0
    indices[pos[mask]] = lb[mask]
    return indptr, indices


def dag_pairs(indptr: np.ndarray, indices: np.ndarray, node_round: np.ndarray, *, exact_limit: int = EXACT_PAIR_LIMIT, sample_seed: int = 0) -> dict[str, Any]:
    """Comparable pairs of a DAG in topological order (exact or stratified estimate)."""

    lib = native_kernel()
    n = int(indptr.size - 1)
    indptr = np.ascontiguousarray(indptr, dtype=np.int64)
    indices = np.ascontiguousarray(indices, dtype=np.int64)
    ancestors_total = np.zeros(n, dtype=np.int64)
    descendants = np.zeros(n, dtype=np.int64)
    edge_cover = np.zeros(indices.size, dtype=np.uint8)
    if n <= exact_limit:
        total = 0
        for start in range(0, n, PAIR_BLOCK):
            stop = min(n, start + PAIR_BLOCK)
            slot = np.full(n, -1, dtype=np.int64)
            slot[start:stop] = np.arange(stop - start, dtype=np.int64)
            words = (stop - start + 63) // 64
            bits = np.zeros(n * words, dtype=np.uint64)
            anc = np.zeros(n, dtype=np.int64)
            total += int(lib.oph_wire_ancestors(n, words, _ptr(indptr, ctypes.c_int64), _ptr(indices, ctypes.c_int64), _ptr(slot, ctypes.c_int64), _ptr(bits, ctypes.c_uint64), _ptr(anc, ctypes.c_int64)))
            ancestors_total += anc
            counts = np.zeros(words * 64, dtype=np.int64)
            lib.oph_wire_column_counts(n, words, _ptr(bits, ctypes.c_uint64), _ptr(counts, ctypes.c_int64))
            descendants[start:stop] = counts[: stop - start]
            lib.oph_wire_covers(n, words, _ptr(indptr, ctypes.c_int64), _ptr(indices, ctypes.c_int64), _ptr(slot, ctypes.c_int64), _ptr(bits, ctypes.c_uint64), _ptr(edge_cover, ctypes.c_uint8))
        comparable = total - n  # strict pairs
        return {
            "pair_counting": "exact_all_pairs",
            "strict_pair_count": int(comparable),
            "strict_pair_count_standard_error": None,
            "ancestors": ancestors_total - 1,
            "descendants": descendants - 1,
            "edge_is_cover": edge_cover.astype(bool),
            "sample_size": None,
        }
    rng = np.random.Generator(np.random.PCG64(sample_seed))
    rounds = np.asarray(node_round, dtype=np.int64)
    strata = {int(r): np.flatnonzero(rounds == r) for r in np.unique(rounds)}
    chosen = {}
    for r in sorted(strata):
        members = strata[r]
        size = min(members.size, max(min(members.size, 2), -(-SAMPLE_SOURCES * members.size // n)))
        chosen[r] = np.sort(rng.choice(members, size=size, replace=False))
    sampled = np.sort(np.concatenate(list(chosen.values())))
    slot = np.full(n, -1, dtype=np.int64)
    slot[sampled] = np.arange(sampled.size, dtype=np.int64)
    words = (sampled.size + 63) // 64
    bits = np.zeros(n * words, dtype=np.uint64)
    anc = np.zeros(n, dtype=np.int64)
    lib.oph_wire_ancestors(n, words, _ptr(indptr, ctypes.c_int64), _ptr(indices, ctypes.c_int64), _ptr(slot, ctypes.c_int64), _ptr(bits, ctypes.c_uint64), _ptr(anc, ctypes.c_int64))
    counts = np.zeros(words * 64, dtype=np.int64)
    lib.oph_wire_column_counts(n, words, _ptr(bits, ctypes.c_uint64), _ptr(counts, ctypes.c_int64))
    future = counts[: sampled.size] - 1  # strict descendants of each sampled source
    estimate = Fraction(0)
    variance = 0.0
    for r in sorted(strata):
        Nh = int(strata[r].size)
        picked = chosen[r]
        nh = int(picked.size)
        vals = future[slot[picked]].astype(np.int64)
        S = int(vals.sum())
        Q = int(np.dot(vals, vals))
        estimate += Fraction(Nh * S, nh)
        s2 = (Q - S * S / nh) / (nh - 1) if nh > 1 else 0.0
        variance += Nh * Nh * (1.0 - nh / Nh) * s2 / nh
    descendants[sampled] = future + 1
    return {
        "pair_counting": "stratified_sample_by_round",
        "strict_pair_count": int(round(float(estimate))),
        "strict_pair_count_standard_error": rounded(math.sqrt(variance)),
        "ancestors": anc - 1,
        "descendants": descendants - 1,
        "edge_is_cover": None,
        "sample_size": int(sampled.size),
    }


def dag_depth(indptr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    lib = native_kernel()
    n = int(indptr.size - 1)
    depth = np.zeros(n, dtype=np.int64)
    lib.oph_wire_depth(n, _ptr(np.ascontiguousarray(indptr, dtype=np.int64), ctypes.c_int64), _ptr(np.ascontiguousarray(indices, dtype=np.int64), ctypes.c_int64), _ptr(depth, ctypes.c_int64))
    return depth


def ordering_summary(pairs: int, n: int) -> dict[str, Any]:
    if n < 2:
        return {"ordering_fraction": None, "myrheim_meyer_dimension": None, "deviation_from_2plus1": None, "deviation_from_3plus1": None}
    f = Fraction(2 * int(pairs), n * (n - 1))
    dim = invert_myrheim_meyer_fraction(float(f))
    return {
        "ordering_fraction": rounded(float(f)),
        "ordering_fraction_exact": str(f),
        "myrheim_meyer_dimension": rounded(dim) if dim is not None else None,
        "deviation_from_2plus1": rounded(float(f - MM_2PLUS1)),
        "deviation_from_3plus1": rounded(float(f - MM_3PLUS1)),
    }


def angular_distance_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.sum(u * v, axis=-1), -1.0, 1.0)))


def readback_positions(loads12: np.ndarray) -> np.ndarray:
    """Isometric coordinates of ``x = 2 P_slow N``: ``X = V^T N`` with ``V`` the port directions.

    ``|X| = |2 P_slow N|`` since ``G = 4 P_slow = V V^T``; the constant mode drops out.
    """

    return np.asarray(loads12, dtype=float) @ port_directions()


def link_statistics(
    ids: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    edge_cover: np.ndarray | None,
    node_round: np.ndarray,
    screen_position: np.ndarray,
    hop_angle_deg: float,
    readback_position: np.ndarray | None,
    node_level: np.ndarray | None,
) -> dict[str, Any]:
    child = np.repeat(np.arange(ids.size, dtype=np.int64), np.diff(indptr))
    parent = indices
    if child.size == 0:
        return {"parent_edges": 0}
    dt = node_round[child] - node_round[parent]
    angle = angular_distance_deg(screen_position[child], screen_position[parent])
    hops = angle / hop_angle_deg
    # tangent-plane direction of the child seen from the parent
    p = screen_position[parent]
    c = screen_position[child]
    tangent = c - np.sum(c * p, axis=1, keepdims=True) * p
    norm = np.linalg.norm(tangent, axis=1)
    moving = norm > 1e-12
    unit = np.zeros_like(tangent)
    unit[moving] = tangent[moving] / norm[moving, None]
    frames = local_frames(p[moving]) if np.any(moving) else None
    resultant = None
    if frames is not None and moving.sum() > 0:
        u = np.einsum("ni,nij->nj", unit[moving], frames)[:, :2]
        resultant = float(np.linalg.norm(u.mean(axis=0)))
    out: dict[str, Any] = {
        "parent_edges": int(child.size),
        "cover_fraction": rounded(float(edge_cover.mean())) if edge_cover is not None else None,
        "same_round_fraction": rounded(float(np.mean(dt == 0))),
        "round_step": stats(dt),
        "screen_placement": {
            "definition": "event at the normalized midpoint of its two endpoint cell centres on S2; displacement = great-circle angle in units of the finest level's mean neighbour angle",
            "hop_angle_deg": rounded(hop_angle_deg),
            "displacement_hops": stats(hops),
            "displacement_hops_quantiles_10_50_90": [rounded(v) for v in np.quantile(hops, [0.1, 0.5, 0.9])],
            "zero_displacement_fraction": rounded(float(np.mean(hops < 1e-9))),
            "hops_per_round_when_round_step_positive": rounded(float(np.mean(hops[dt > 0] / dt[dt > 0]))) if np.any(dt > 0) else None,
            "tangent_direction_mean_resultant_length": rounded(resultant) if resultant is not None else None,
        },
    }
    if edge_cover is not None and edge_cover.any():
        out["screen_placement"]["cover_displacement_hops"] = stats(hops[edge_cover])
    if node_level is not None:
        step = node_level[child] - node_level[parent]
        out["level_step"] = {
            "fraction_same_level": rounded(float(np.mean(step == 0))),
            "fraction_child_to_parent_level": rounded(float(np.mean(step < 0))),
            "fraction_parent_to_child_level": rounded(float(np.mean(step > 0))),
        }
    if readback_position is not None:
        d = readback_position[child] - readback_position[parent]
        length = np.linalg.norm(d, axis=1)
        moving = length > 1e-12
        resultant3 = float(np.linalg.norm((d[moving] / length[moving, None]).mean(axis=0))) if moving.any() else None
        out["readback_placement"] = {
            "definition": (
                "event at the mean of its two endpoint carriers' own readbacks x = 2 P_slow N at the written "
                "versions, in the isometric coordinates X = V^T N (units: one port unit along a port direction)"
            ),
            "displacement": stats(length),
            "displacement_quantiles_10_50_90": [rounded(v) for v in np.quantile(length, [0.1, 0.5, 0.9])],
            "zero_displacement_fraction": rounded(float(np.mean(length < 1e-9))),
            "direction_mean_resultant_length": rounded(resultant3) if resultant3 is not None else None,
            "position_norm": stats(np.linalg.norm(readback_position, axis=1)),
        }
    return out


def interval_report(
    system: EventSystem,
    log: ProvenanceLog,
    prov: Provenance,
    bottom: int,
    top: int,
    tip_cell: int,
    *,
    sample_seed: int,
    with_placement: bool,
) -> dict[str, Any]:
    ids = interval_ids(prov, bottom, top)
    n = int(ids.size)
    node_round = log.round_of[ids]
    indptr, indices = local_parents(prov, ids)
    pairs = dag_pairs(indptr, indices, node_round, sample_seed=sample_seed)
    depth = dag_depth(indptr, indices)
    layer_sizes = np.bincount(depth)
    round_sizes = np.bincount(node_round - node_round.min())
    mid = system.seam_midpoints()
    positions = mid[log.seam[ids]]
    angle = angular_distance_deg(positions, system.cell_points[tip_cell][None, :])
    levels = system.cell_level[log.cell_a[ids]]
    readback = None
    if with_placement and log.snapshots is not None:
        slots = log.snapshot_slot[ids]
        if np.any(slots < 0):
            raise AssertionError("interval event without a readback snapshot")
        snap = log.snapshots[slots].astype(float)
        readback = 0.5 * (readback_positions(snap[:, :PORTS]) + readback_positions(snap[:, PORTS:]))
    order = ordering_summary(pairs["strict_pair_count"], n)
    report = {
        "bottom_attempt": int(bottom),
        "top_attempt": int(top),
        "bottom_round": int(log.round_of[bottom]),
        "top_round": int(log.round_of[top]),
        "round_separation": rounded((top - bottom) / system.external),
        "events": n,
        "events_per_round_max": int(round_sizes.max()),
        "height_events": int(depth.max()),
        "layer_width_max": int(layer_sizes.max()),
        "height_layers": int(layer_sizes.size - 1),
        "waist_angle_deg_max": rounded(float(angle.max())),
        "waist_angle_deg_mean": rounded(float(angle.mean())),
        "interior": bool(angle.max() <= INTERIOR_ANGLE_DEG),
        "events_by_level": {str(int(l)): int(c) for l, c in zip(*np.unique(levels, return_counts=True))},
        "events_by_seam_kind": {"w12": int(np.count_nonzero(system.ext_kind[log.seam[ids]] == 0)), "join": int(np.count_nonzero(system.ext_kind[log.seam[ids]] == 1))},
        "pair_counting": pairs["pair_counting"],
        "strict_pair_count": pairs["strict_pair_count"],
        "strict_pair_count_standard_error": pairs["strict_pair_count_standard_error"],
        "sample_size": pairs["sample_size"],
        "ancestor_count_mean": rounded(float(np.mean(pairs["ancestors"][pairs["ancestors"] >= 0]))) if pairs["pair_counting"] == "exact_all_pairs" else None,
        "links": link_statistics(ids, indptr, indices, pairs["edge_is_cover"], node_round, positions, system.hop_angle_deg, readback, levels if len(system.levels) > 1 else None),
    }
    report.update(order)
    if pairs["strict_pair_count_standard_error"] is not None and n > 1:
        report["ordering_fraction_standard_error"] = rounded(2.0 * pairs["strict_pair_count_standard_error"] / (n * (n - 1)))
    return report


def cell_attempts(log: ProvenanceLog, cell: int) -> np.ndarray:
    """Attempt ids touching ``cell`` in execution order."""

    return np.flatnonzero((log.cell_a == cell) | (log.cell_b == cell)).astype(np.int64)


def first_write_on_cell(touch_ids: np.ndarray, writes: np.ndarray, at_or_after: int) -> int | None:
    candidates = touch_ids[writes[touch_ids] & (touch_ids >= at_or_after)]
    return int(candidates[0]) if candidates.size else None


def front_speed(system: EventSystem, log: ProvenanceLog, prov: Provenance, bottom: int, tip_cell: int, rounds_ahead: int) -> dict[str, Any]:
    hi = min(log.attempts - 1, bottom + rounds_ahead * system.external)
    fwd = reach_forward(prov, bottom, hi)
    ids = np.flatnonzero(fwd)
    mid = system.seam_midpoints()
    angle = angular_distance_deg(mid[log.seam[ids]], system.cell_points[tip_cell][None, :])
    offset = log.round_of[ids] - log.round_of[bottom]
    per_round = []
    for k in range(rounds_ahead + 1):
        m = offset == k
        if m.any():
            per_round.append({"rounds_ahead": k, "events": int(m.sum()), "max_angle_deg": rounded(float(angle[m].max())), "mean_angle_deg": rounded(float(angle[m].mean()))})
    ks = np.asarray([r["rounds_ahead"] for r in per_round if 1 <= r["rounds_ahead"] and r["max_angle_deg"] < 150.0], dtype=float)
    vals = np.asarray([r["max_angle_deg"] for r in per_round if 1 <= r["rounds_ahead"] and r["max_angle_deg"] < 150.0], dtype=float)
    slope = None
    if ks.size >= 2:
        slope = float(np.polyfit(ks, vals, 1)[0])
    return {
        "definition": "future of the bottom tip; per round ahead the maximal great-circle angle of reached events from the tip cell centre",
        "per_round": per_round,
        "front_speed_deg_per_round": rounded(slope) if slope is not None else None,
        "front_speed_hops_per_round": rounded(slope / system.hop_angle_deg) if slope is not None else None,
    }


def growth_fit(rows: list[dict[str, Any]], min_delta: float = GROWTH_MIN_DELTA) -> dict[str, Any]:
    pts = [(r["round_separation"], r["events"]) for r in rows if r["interior"] and r["round_separation"] >= min_delta and r["events"] > 1]
    all_pts = [(r["round_separation"], r["events"]) for r in rows if r["round_separation"] >= min_delta and r["events"] > 1]
    local = []
    for (d0, n0), (d1, n1) in zip(all_pts, all_pts[1:]):
        if d1 > d0 and n1 > 0 and n0 > 0:
            local.append({"from": rounded(d0), "to": rounded(d1), "local_exponent": rounded(math.log(n1 / n0) / math.log(d1 / d0))})
    out = {
        "definition": f"least-squares slope of log(events) against log(round separation) over interior intervals with separation >= {min_delta} rounds; 1 + spatial dimension is the expected exponent of a manifold-like order",
        "fit_points": [[rounded(d), int(n)] for d, n in pts],
        "exponent": None,
        "local_exponents": local,
    }
    if len(pts) >= 2:
        x = np.log([d for d, _ in pts])
        y = np.log([n for _, n in pts])
        out["exponent"] = rounded(float(np.polyfit(x, y, 1)[0]))
    return out


def select_tips(system: EventSystem, seed: int, interior: int, pentagonal: int, level: int | None = None) -> list[int]:
    rng = np.random.Generator(np.random.PCG64(seed))
    mask = np.ones(system.carriers, dtype=bool) if level is None else (system.cell_level == level)
    pent = np.flatnonzero(mask & system.pentagonal_cell)
    inner = np.flatnonzero(mask & ~system.pentagonal_cell)
    picks = [int(v) for v in rng.choice(inner, size=min(interior, inner.size), replace=False)]
    picks += [int(v) for v in rng.choice(pent, size=min(pentagonal, pent.size), replace=False)]
    return picks


def ladder_deltas(logged_rounds: int, fraction: float = 0.75) -> list[float]:
    return [d for d in DELTA_LADDER if d <= fraction * logged_rounds + 1e-9]


def snapshot_union(log: ProvenanceLog, prov: Provenance, tips: Sequence[int], logged_rounds: int) -> np.ndarray:
    """Union over tips of the largest ladder interval (every smaller ladder interval is a subset)."""

    G = log.system.external
    base = (BURN_IN_ROUNDS + logged_rounds // 4) * G
    union = []
    for cell in tips:
        touch = cell_attempts(log, cell)
        bottom = first_write_on_cell(touch, prov.writes, base)
        if bottom is None:
            continue
        for delta in reversed(ladder_deltas(logged_rounds)):
            top = first_write_on_cell(touch, prov.writes, bottom + int(round(delta * G)))
            if top is not None:
                union.append(interval_ids(prov, bottom, top))
                break
    return np.unique(np.concatenate(union)) if union else np.zeros(0, dtype=np.int64)


def _interval_summary(rows: list[dict[str, Any]], separation: float | None) -> dict[str, Any]:
    return {
        "round_separation": separation if separation is not None else stats([r["round_separation"] for r in rows]),
        "events": stats([r["events"] for r in rows]),
        "ordering_fraction": stats([r["ordering_fraction"] for r in rows if r["ordering_fraction"] is not None]),
        "myrheim_meyer_dimension": stats([r["myrheim_meyer_dimension"] for r in rows if r["myrheim_meyer_dimension"] is not None]),
        "height_events": stats([r["height_events"] for r in rows]),
        "layer_width_max": stats([r["layer_width_max"] for r in rows]),
        "waist_angle_deg_max": stats([r["waist_angle_deg_max"] for r in rows]),
        "interior_all": bool(all(r["interior"] for r in rows)),
        "pair_counting": sorted({r["pair_counting"] for r in rows}),
    }


def provenance_block(
    system: EventSystem,
    log: ProvenanceLog,
    tips: list[int],
    *,
    variant: str,
    logged_rounds: int,
    with_placement: bool,
) -> dict[str, Any]:
    """Order statistics for one write convention."""

    if variant == "value_changing_writes":
        writes = log.outcome > 0
        rule = "an attempt writes both records when it changes a value (unit transfer or swap); a wait reads only"
        fraction = 0.75
    elif variant == "every_attempt_writes":
        writes = np.ones(log.attempts, dtype=bool)
        rule = "control: every attempt writes both records (the certified comparison counts as a write)"
        fraction = CONTROL_LADDER_FRACTION
    else:
        raise ValueError(variant)
    prov = provenance_from_log(log.cell_a, log.cell_b, writes, system.carriers)
    G = system.external
    R = logged_rounds
    base = (BURN_IN_ROUNDS + R // 4) * G
    per_tip = []
    for k, cell in enumerate(tips):
        touch = cell_attempts(log, cell)
        bottom = first_write_on_cell(touch, prov.writes, base)
        if bottom is None:
            raise AssertionError("no writing attempt on the tip carrier after the burn-in")
        rows = []
        main = None
        row_of_top: dict[int, dict[str, Any]] = {}
        for delta in ladder_deltas(R, fraction):
            top = first_write_on_cell(touch, prov.writes, bottom + int(round(delta * G)))
            if top is None:
                continue
            if top not in row_of_top:
                row = interval_report(system, log, prov, bottom, top, cell, sample_seed=log.seed + 97 * k + int(10 * delta), with_placement=with_placement)
                row["ladder_delta"] = rounded(delta)
                rows.append(row)
                row_of_top[top] = row
            if abs(delta - R / 2) < 1e-9:
                main = row_of_top[top]
        interior_rows = [r for r in rows if r["interior"]]
        largest_interior = max(interior_rows, key=lambda r: r["events"]) if interior_rows else None
        per_tip.append(
            {
                "tip_cell": int(cell),
                "tip_level": int(system.cell_level[cell]),
                "tip_pentagonal": bool(system.pentagonal_cell[cell]),
                "bottom_attempt": int(bottom),
                "bottom_round": int(log.round_of[bottom]),
                "main_interval": main,
                "largest_interior_interval": largest_interior,
                "ladder": rows,
                "growth": growth_fit(rows, DEPTH_GROWTH_MIN_DELTA if len(system.levels) > 1 else GROWTH_MIN_DELTA),
                "front": front_speed(system, log, prov, bottom, cell, min(int(math.ceil(fraction * R)), log.rounds - 1 - int(log.round_of[bottom]))),
            }
        )
    mains = [t["main_interval"] for t in per_tip if t["main_interval"] is not None]
    largest = [t["largest_interior_interval"] for t in per_tip if t["largest_interior_interval"] is not None]
    exps = [t["growth"]["exponent"] for t in per_tip if t["growth"]["exponent"] is not None]
    speeds = [t["front"]["front_speed_hops_per_round"] for t in per_tip if t["front"]["front_speed_hops_per_round"] is not None]
    roots = int(np.count_nonzero(prov.parent_a < 0)) + int(np.count_nonzero(prov.parent_b < 0))
    block = {
        "variant": variant,
        "write_rule": rule,
        "events": int(np.count_nonzero(writes)),
        "attempts": int(log.attempts),
        "root_reads": roots,
        "tips": per_tip,
        "summary": {
            "main_interval": _interval_summary(mains, rounded(R / 2)) if mains else None,
            "largest_interior_interval": _interval_summary(largest, None) if largest else None,
            "growth_exponent": stats(exps),
            "front_speed_hops_per_round": stats(speeds),
            "expected_2plus1": {"ordering_fraction": rounded(float(MM_2PLUS1)), "dimension": 3, "growth_exponent": 3},
        },
    }
    return block


def provenance_level_block(system: EventSystem, logged_rounds: int, seed: int, tips: list[int], *, store_log: bool, timings: dict[str, float] | None = None) -> dict[str, Any]:
    timings = {} if timings is None else timings
    rounds = BURN_IN_ROUNDS + logged_rounds + LADDER_SLACK_ROUNDS
    loads = federation.initial_loads(system.levels[-1], system.ports)
    t0 = time.perf_counter()
    log = run_provenance(system, rounds, seed, loads)
    prov = provenance_from_log(log.cell_a, log.cell_b, log.outcome > 0, system.carriers)
    union_ids = snapshot_union(log, prov, tips, logged_rounds)
    # Second pass: the same schedule with readback snapshots for every event of the sampled intervals.
    log2 = run_provenance(system, rounds, seed, loads, snapshot_ids=union_ids)
    if log2.content_sha256() != log.content_sha256() or log2.terminal_state_sha256 != log.terminal_state_sha256:
        raise AssertionError("the snapshot pass diverged from the logging pass")
    timings[f"{system.name}/run"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    primary = provenance_block(system, log2, tips, variant="value_changing_writes", logged_rounds=logged_rounds, with_placement=True)
    control = provenance_block(system, log2, tips, variant="every_attempt_writes", logged_rounds=logged_rounds, with_placement=False)
    timings[f"{system.name}/order"] = time.perf_counter() - t0
    stored = None
    if store_log:
        stored = store_provenance_log(system, log, prov)
    write_fraction = [r["external_write_fraction"] for r in log.per_round]
    return {
        "system": system.name,
        "description": system.description,
        "levels": list(system.levels),
        "carriers": int(system.carriers),
        "seams": int(system.seams),
        "intra_seams": int(system.intra_count),
        "external_seams": int(system.external),
        "external_seams_by_kind": {"w12": int(np.count_nonzero(system.ext_kind == 0)), "join": int(np.count_nonzero(system.ext_kind == 1))},
        "schedule": {
            "rule": "numpy.random.Generator(numpy.random.PCG64(seed)); per round seq = rng.permutation(|S|) then coin = rng.integers(0, 2, size=|S|); every seam once per round in that order; odd totals place the ceiling on the first endpoint when coin = 1",
            "seed": int(seed),
            "rounds_total": int(rounds),
            "burn_in_rounds": int(BURN_IN_ROUNDS),
            "logged_rounds": int(logged_rounds),
            "ladder_slack_rounds": int(LADDER_SLACK_ROUNDS),
            "tip_rounds": [int(BURN_IN_ROUNDS + logged_rounds // 4), int(BURN_IN_ROUNDS + 3 * logged_rounds // 4)],
            "ladder_round_separations": [rounded(d) for d in ladder_deltas(logged_rounds)],
        },
        "loads": {"rule": f"federation.initial_loads({system.levels[-1]}, ports): default_rng({federation.LOAD_SEED_BASE} + {system.levels[-1]}).integers(0, {LOAD_MAX + 1}, size=ports)", "sha256": log.loads_sha256},
        "descent": {
            "functional": "V = sum_p x_p^2",
            "initial": int(log.v_initial),
            "terminal": int(log.v_terminal),
            "balanced_class_minimum": int(log.v_minimum),
            "ledger_exact": bool(log.ledger_exact),
            "round_violations": int(log.descent_violations),
            "balanced_class_reached_at_round": log.balanced_round,
            "terminal_equals_minimum": bool(log.v_terminal == log.v_minimum),
        },
        "terminal_state_sha256": log.terminal_state_sha256,
        "log_content_sha256": log.content_sha256(),
        "attempts_logged": int(log.attempts),
        "per_round": log.per_round,
        "external_write_fraction_logged_window": stats(write_fraction[BURN_IN_ROUNDS:]),
        "provenance_rule": (
            "read-after-write: every carrier record starts at version 0 (root); an attempt reads both endpoint records at "
            "their current versions and, when it writes, advances both versions and becomes their writer; the direct "
            "parents of an attempt are the writers of the versions it read; precedence is the transitive closure; no "
            "declared parents, no round labels enter the rule"
        ),
        "tip_cells": [int(c) for c in tips],
        "primary": primary,
        "control_every_attempt_writes": control,
        "stored_log": stored,
    }


def store_provenance_log(system: EventSystem, log: ProvenanceLog, prov: Provenance) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{system.name.replace('/', '_')}_provenance_log.npz"
    meta = {
        "schema": LOG_SCHEMA,
        "system": system.name,
        "levels": list(system.levels),
        "seed": int(log.seed),
        "rounds": int(log.rounds),
        "external_seams": int(system.external),
        "columns": "seam (external seam index), outcome (0 wait, 1 swap, 2 descent), version_read_a/b, version_written_a/b; round = attempt // external_seams; carriers = ext_left[seam], ext_right[seam]",
    }
    arrays = {
        "seam": log.seam.astype(np.int32),
        "outcome": log.outcome.astype(np.int8),
        "version_read_a": prov.version_read_a.astype(np.int32),
        "version_read_b": prov.version_read_b.astype(np.int32),
        "version_written_a": prov.version_written_a.astype(np.int32),
        "version_written_b": prov.version_written_b.astype(np.int32),
        "ext_left": system.ext_left.astype(np.int32),
        "ext_right": system.ext_right.astype(np.int32),
        "meta": np.array(canonical_json(meta)),
    }
    np.savez_compressed(path, **arrays)
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "schema": LOG_SCHEMA,
        "attempts": int(log.attempts),
        "content_sha256": log.content_sha256(),
        "versions_sha256": sha256_arrays(arrays["version_read_a"], arrays["version_read_b"], arrays["version_written_a"], arrays["version_written_b"]),
    }


# --------------------------------------------------------------------------
# Source-net comparison (record-metric route, q = 13)
# --------------------------------------------------------------------------


def source_net_block(q: int = SOURCE_NET_Q, dim: int = SOURCE_NET_DIM) -> dict[str, Any]:
    indptr, indices, values, A, B, sites = source_net.build_site_graph(q, dim)
    n = q**dim
    K = source_net.ceil_sqrt(q)
    center_axis = 0
    half = (Fraction(1, 2), 0)
    for b in range(1, q):
        lhs = source_net.phi_square(source_net.phi_sub(values[b], half))
        rhs = source_net.phi_square(source_net.phi_sub(values[center_axis], half))
        if source_net.phi_sign(source_net.phi_sub(lhs, rhs)) < 0:
            center_axis = b
    center = sum(center_axis * q ** (dim - 1 - i) for i in range(dim))
    alpha = source_net.bfs(indptr, indices, center, n)
    xi = np.array([source_net.phi_float(v) for v in values])
    coords = xi[sites]  # (n, dim) in units of L
    a_q = 1.0 / math.sqrt(q)
    rows = []
    for k in range(2, K + 1):
        layer_nodes = []
        for j in range(k + 1):
            s = np.flatnonzero(alpha <= min(j, k - j))
            layer_nodes.append(s)
        node_layer = np.concatenate([np.full(s.size, j, dtype=np.int64) for j, s in enumerate(layer_nodes)])
        node_site = np.concatenate(layer_nodes)
        N = int(node_site.size)
        offsets = np.cumsum([0] + [s.size for s in layer_nodes])
        ptr = [0]
        idx = []
        for j in range(k + 1):
            if j == 0:
                for _ in layer_nodes[0]:
                    ptr.append(len(idx))
                continue
            prev = layer_nodes[j - 1]
            lookup = np.full(n, -1, dtype=np.int64)
            lookup[prev] = np.arange(prev.size) + offsets[j - 1]
            for s in layer_nodes[j]:
                nb = indices[indptr[s] : indptr[s + 1]]
                loc = lookup[nb]
                loc = loc[loc >= 0]
                idx.extend(int(v) for v in np.sort(loc))
                ptr.append(len(idx))
        indptr_l = np.asarray(ptr, dtype=np.int64)
        indices_l = np.asarray(idx, dtype=np.int64)
        pairs = dag_pairs(indptr_l, indices_l, node_layer, sample_seed=q)
        depth = dag_depth(indptr_l, indices_l)
        child = np.repeat(np.arange(N), np.diff(indptr_l))
        disp = np.linalg.norm(coords[node_site[child]] - coords[node_site[indices_l]], axis=1) / a_q
        direction = coords[node_site[child]] - coords[node_site[indices_l]]
        length = np.linalg.norm(direction, axis=1)
        moving = length > 1e-12
        resultant = float(np.linalg.norm((direction[moving] / length[moving, None]).mean(axis=0))) if moving.any() else None
        row = {
            "layers": k,
            "events": N,
            "counts_by_layer": [int(s.size) for s in layer_nodes],
            "height_events": int(depth.max()),
            "layer_width_max": int(max(s.size for s in layer_nodes)),
            "pair_counting": pairs["pair_counting"],
            "strict_pair_count": pairs["strict_pair_count"],
            "links": {
                "parent_edges": int(child.size),
                "cover_fraction": rounded(float(pairs["edge_is_cover"].mean())) if pairs["edge_is_cover"] is not None else None,
                "same_round_fraction": 0.0,
                "source_chart_placement": {
                    "definition": "event (j, s) at the source coordinate xi(s) of its site (units of a_q = L/sqrt q); every link advances one layer",
                    "displacement_a_q": stats(disp),
                    "displacement_quantiles_10_50_90": [rounded(v) for v in np.quantile(disp, [0.1, 0.5, 0.9])],
                    "zero_displacement_fraction": rounded(float(np.mean(disp < 1e-9))),
                    "direction_mean_resultant_length": rounded(resultant) if resultant is not None else None,
                },
            },
        }
        row.update(ordering_summary(pairs["strict_pair_count"], N))
        rows.append(row)
    growth = []
    for r0, r1 in zip(rows, rows[1:]):
        growth.append({"from": r0["layers"], "to": r1["layers"], "local_exponent": rounded(math.log(r1["events"] / r0["events"]) / math.log(r1["layers"] / r0["layers"]))})
    fit = None
    if len(rows) >= 2:
        fit = rounded(float(np.polyfit(np.log([r["layers"] for r in rows]), np.log([r["events"] for r in rows]), 1)[0]))
    receipt_row = None
    receipt_path = REPO_ROOT / "data" / "exact" / "source_net_causal_limit_receipt.json"
    if receipt_path.exists():
        stored = json.loads(receipt_path.read_text(encoding="ascii"))
        for level in stored["levels"]:
            if level["q"] != q:
                continue
            for family in level["families"]:
                if family["dimension"] != dim:
                    continue
                top = family["vertical_intervals"][-1]
                receipt_row = {
                    "layers": top["layers"],
                    "inclusive_event_count": top["inclusive_event_count"],
                    "strict_pair_count": top.get("strict_pair_count"),
                    "ordering_fraction_float": top["ordering_fraction_float"],
                    "myrheim_meyer_dimension": top["myrheim_meyer_dimension"],
                }
    reproduced = bool(receipt_row is not None and receipt_row["inclusive_event_count"] == rows[-1]["events"] and receipt_row["strict_pair_count"] == rows[-1]["strict_pair_count"])
    return {
        "route": "record metric (source net): q^3 golden sites on the rank-three source Gram metric, complete-neighbour reads inside a_q, one event per site and layer, layer duration a_q/c",
        "q": q,
        "dimension_of_population": dim,
        "layer_steps": K,
        "sites": n,
        "centre_site": int(center),
        "order": "(j, s) <= (j', t) iff graph distance d(s, t) <= j' - j (waiting included); parents of (j, t) are (j - 1, s) for every neighbour s of t including t itself",
        "vertical_intervals": rows,
        "growth": {"local_exponents": growth, "exponent_all_layers": fit, "note": "thin ladder (layers 2..K); even and odd layer counts alternate around the limit, see the source-net receipt"},
        "source_net_receipt_row": receipt_row,
        "reproduces_source_net_receipt_top_interval": reproduced,
        "expected_3plus1": {"ordering_fraction": rounded(float(MM_3PLUS1)), "dimension": 4},
    }


# --------------------------------------------------------------------------
# Receipt assembly
# --------------------------------------------------------------------------


def file_pins() -> dict[str, str | None]:
    return {name: (sha256_file(path) if Path(path).exists() else None) for name, path in PIN_FILES}


def dynamics_tasks(level: int, *, engine: str, schedules: int) -> list[dict[str, Any]]:
    """Pool tasks of one level: the spectrum, the kernel batches (largest first), the schedules."""

    fed = build_w12_federation(level)
    tasks: list[dict[str, Any]] = [{"kind": "spectrum", "level": level}]
    cells = sample_kernel_cells(fed)
    for start in range(0, len(cells), KERNEL_BATCH):
        tasks.append({"kind": "kernel", "level": level, "cells": cells[start : start + KERNEL_BATCH]})
    if level in FLOAT_TERMINATION_LEVELS:
        for s in range(schedules):
            tasks.append({"kind": "schedule", "level": level, "law": "float", "seed": schedule_seed(level, s), "engine": engine, "max_sweeps": federation.MAX_SWEEPS_DEFAULT})
    else:
        for s in range(min(schedules, FLOAT_BUDGET_SCHEDULES)):
            tasks.append({"kind": "schedule", "level": level, "law": "float", "seed": schedule_seed(level, s), "engine": engine, "max_sweeps": FLOAT_BUDGET_SWEEPS[level]})
    for s in range(schedules):
        tasks.append({"kind": "schedule", "level": level, "law": "integer", "seed": schedule_seed(level, s), "engine": engine, "max_sweeps": federation.MAX_SWEEPS_DEFAULT})
    return tasks


def dynamics_block(level: int, outcomes: list[dict[str, Any]], *, timings: dict[str, float]) -> tuple[dict[str, Any], dict[str, Any]]:
    fed = build_w12_federation(level)
    loads = initial_loads(level, fed.ports)
    outcomes = [o for o in outcomes if o["task"]["level"] == level]
    sync = next(o["spectrum"] for o in outcomes if o["task"]["kind"] == "spectrum")
    floats = [o["result"] for o in outcomes if o["task"]["kind"] == "schedule" and o["task"]["law"] == "float"]
    ints = [o["result"] for o in outcomes if o["task"]["kind"] == "schedule" and o["task"]["law"] == "integer"]
    cells = sample_kernel_cells(fed)
    rows = []
    for o in outcomes:
        if o["task"]["kind"] == "kernel":
            rows.extend(o["rows"])
            timings[f"L{level}/kernel/{o['task']['cells'][0]}"] = o["seconds"]
        elif o["task"]["kind"] == "schedule":
            timings[f"L{level}/{o['task']['law']}/{o['task']['seed']}"] = o["seconds"]
        else:
            timings[f"L{level}/synchronous"] = o["seconds"]
    rows.sort(key=lambda r: r["cell"])
    budget = None if level in FLOAT_TERMINATION_LEVELS else FLOAT_BUDGET_SWEEPS[level]
    float_block = aggregate_float(floats, fed, loads, budget)
    if budget is not None and sync["laplacian_lambda_2"]:
        lam2 = sync["laplacian_lambda_2"]
        phi_end = float_block["phi_terminal"]["median"]
        if phi_end and phi_end > federation.FLOAT_PHI_THRESHOLD:
            float_block["extrapolated_attempts_to_threshold"] = rounded(float(f"{floats[0]['attempts'] + math.log(phi_end / federation.FLOAT_PHI_THRESHOLD) * fed.seams / lam2:.4g}"))
    dynamics = {
        "level": level,
        "federation": {
            "carriers": fed.carriers,
            "ports": fed.ports,
            "seams": fed.seams,
            "intra_seams": fed.intra_count,
            "glued_seams": fed.inter_count,
            "components": fed.components,
            "component_size_min": int(fed.component_sizes.min()),
            "component_size_max": int(fed.component_sizes.max()),
        },
        "loads": {"rule": f"federation.initial_loads(level, ports): default_rng({federation.LOAD_SEED_BASE} + level).integers(0, {LOAD_MAX + 1}, size=ports)", "sha256": sha256_of(loads.tolist()), "total": int(loads.sum())},
        "schedule_seeds": f"{SCHEDULE_SEED_BASE} + 1000*level + schedule_index",
        "synchronous_operator": sync,
        "mean_law_float": float_block,
        "integer_law": aggregate_integer(ints, fed, loads),
    }
    slow = {
        "level": level,
        "definition": "K_n = 12 C_n / tr C_n from the twelve centered impulses of one carrier propagated by T_fed for 2n steps with per-step rescaling (federation.response_kernels); slow-band share tr(P_slow K_n P_slow) / tr K_n",
        "steps": list(KERNEL_STEPS),
        "sampled_cells": [int(c) for c in cells],
        "pentagonal_cells_in_sample": int(sum(bool(fed.pentagonal_cell[c]) for c in cells)),
        "summary": kernel_summary(rows),
        "per_cell": rows,
    }
    return dynamics, slow


def _side_row(route: str, summary: dict[str, Any], expected: str) -> dict[str, Any]:
    main = summary["main_interval"] or summary["largest_interior_interval"]
    interior = summary["largest_interior_interval"]
    return {
        "route": route,
        "main_interval_round_separation": main["round_separation"] if main else None,
        "events": main["events"]["median"] if main else None,
        "ordering_fraction": main["ordering_fraction"]["median"] if main else None,
        "myrheim_meyer_dimension": main["myrheim_meyer_dimension"]["median"] if main else None,
        "largest_interior_events": interior["events"]["median"] if interior else None,
        "largest_interior_ordering_fraction": interior["ordering_fraction"]["median"] if interior else None,
        "largest_interior_myrheim_meyer_dimension": interior["myrheim_meyer_dimension"]["median"] if interior else None,
        "growth_exponent": summary["growth_exponent"]["median"],
        "front_speed_hops_per_round": summary["front_speed_hops_per_round"]["median"],
        "height_events": main["height_events"]["median"] if main else None,
        "layer_width_max": main["layer_width_max"]["median"] if main else None,
        "expected": expected,
    }


def build_receipt(*, levels: Sequence[int] = LEVELS, depth_levels: Sequence[int] = DEPTH_LEVELS, engine: str = "auto", workers: int = 1, schedules: int = SCHEDULES, timings: dict[str, float] | None = None, store_logs: bool = True) -> dict[str, Any]:
    timings = {} if timings is None else timings
    engine = federation.resolve_engine(engine)
    carrier_checks = carrier.check_carrier()
    wiring: dict[str, Any] = {}
    dynamics: dict[str, Any] = {}
    slow_band: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    depth: dict[str, Any] = {}
    tasks: list[dict[str, Any]] = []
    for level in levels:
        t0 = time.perf_counter()
        w = build_wiring(level)
        timings[f"L{level}/wiring"] = time.perf_counter() - t0
        wiring[f"L{level}"] = w.receipt
        _log(f"L{level}: wiring built ({w.edges} glued pairs)")
        tasks.extend(dynamics_tasks(level, engine=engine, schedules=schedules))
    # The pool tasks (spectra, kernels, schedules) run in worker processes while
    # this process derives the provenance orders.
    runner = TaskRunner(workers)
    t_pool = time.perf_counter()
    try:
        handle = runner.submit(tasks)
        for level in levels:
            t0 = time.perf_counter()
            system = single_level_system(level)
            tips = select_tips(system, TIP_SEED_BASE + level, TIP_INTERIOR, TIP_PENTAGONAL)
            provenance[f"L{level}"] = provenance_level_block(system, LOGGED_ROUNDS[level], PROVENANCE_SEED_BASE + level, tips, store_log=store_logs and level in STORED_LOG_LEVELS, timings=timings)
            timings[f"L{level}/provenance"] = time.perf_counter() - t0
            _log(f"L{level}: provenance done ({timings[f'L{level}/provenance']:.1f} s)")
        for level in depth_levels:
            t0 = time.perf_counter()
            system = depth_system(level)
            tips = select_tips(system, DEPTH_SEED_BASE + level, DEPTH_TIPS_PER_LEVEL, 0, level=level - 1)
            tips += select_tips(system, DEPTH_SEED_BASE + 100 + level, DEPTH_TIPS_PER_LEVEL, 0, level=level)
            depth[f"L{level}"] = provenance_level_block(system, DEPTH_LOGGED_ROUNDS[level], DEPTH_SEED_BASE + level, tips, store_log=False, timings=timings)
            timings[f"depth L{level}"] = time.perf_counter() - t0
            _log(f"depth L{level}: done ({timings[f'depth L{level}']:.1f} s)")
        t0 = time.perf_counter()
        slow_band["isolated_reference"] = isolated_reference()
        slow_band["three_port_reference_L3"] = three_port_reference(3, sample_kernel_cells(federation.build_federation(3, "port_pair")))
        timings["kernel references"] = time.perf_counter() - t0
        outcomes = runner.collect(handle)
    finally:
        runner.close()
    timings["pool"] = time.perf_counter() - t_pool
    for level in levels:
        d, s = dynamics_block(level, outcomes, timings=timings)
        dynamics[f"L{level}"] = d
        slow_band[f"L{level}"] = s
        _log(f"L{level}: dynamics and slow band assembled")
    t0 = time.perf_counter()
    source = source_net_block()
    timings["source net"] = time.perf_counter() - t0
    side_by_side = []
    top_source = source["vertical_intervals"][-1]
    side_by_side.append(
        {
            "route": "record metric (source net, q = 13, k = 4 layers)",
            "events": top_source["events"],
            "ordering_fraction": top_source["ordering_fraction"],
            "myrheim_meyer_dimension": top_source["myrheim_meyer_dimension"],
            "growth_exponent": source["growth"]["exponent_all_layers"],
            "height_events": top_source["height_events"],
            "layer_width_max": top_source["layer_width_max"],
            "expected": "3+1 (ordering fraction 1/10, dimension 4)",
        }
    )
    for key, block in provenance.items():
        side_by_side.append(_side_row(f"S2 wiring W12, single level {key}, provenance order of the glued reads", block["primary"]["summary"], "2+1 (ordering fraction 8/35, dimension 3)"))
    for key, block in depth.items():
        side_by_side.append(_side_row(f"S2 wiring W12 with depth, levels {block['levels']}, provenance order of the glued and join reads", block["primary"]["summary"], "depth three is thin; reported as measured"))
    wiring_ok = all(w["assignment_acceptance"]["accepted"] and w["unglued_ports"] == 60 for w in wiring.values())
    float_ok = all(d["mean_law_float"]["strict_descent_violations_total"] == 0 and d["mean_law_float"]["descent_ledger_ok_all"] for d in dynamics.values())
    confluence_ok = all(d["mean_law_float"]["terminal_hash_equals_component_mean"] for d in dynamics.values() if d["level"] in FLOAT_TERMINATION_LEVELS)
    integer_ok = all(d["integer_law"]["quotient_hash_equals_expected_multiset"] and d["integer_law"]["unit_transfer_decrement_identity_violations_total"] == 0 for d in dynamics.values())
    provenance_ok = all(p["descent"]["ledger_exact"] and p["descent"]["round_violations"] == 0 and p["descent"]["terminal_equals_minimum"] for p in list(provenance.values()) + list(depth.values()))
    receipt = {
        "schema": SCHEMA,
        "lane": "A-wire: full S2 support wiring, provenance order of the reads",
        "question": (
            "with every architectural element in the run (twelve-port A5 icosahedral carriers, the geodesic S2 screen at "
            "tower level L with all twelve ports wired to the twelve neighbouring cells, the canonical repair law, records "
            "as cumulative port loads, positions as each carrier's own rank-three readback), what does an observer read "
            "from the provenance order of the reads, and how does it compare with the record-metric route of the source net?"
        ),
        "claim_boundary": (
            "Finite receipt on N = 20 * 4^L carriers, L = 3, 4, 5, and on three-level federations. Supplied: the W12 wiring "
            "convention (every cell glued to its twelve vertex-sharing neighbours through the production geometric port "
            "assignment), the join convention (child port k to parent port k), the schedules, the initial loads. Produced: "
            "the repairs, the provenance order generated by the read-after-write rule from the log alone, the readbacks. "
            "Not claimed: selection of the wiring by the axioms (the open (M1) item), a physical clock, the continuum "
            "limit, physical position or length, field attachment. The ordering fractions, dimensions and growth exponents "
            "are finite diagnostics of the declared wiring."
        ),
        "scope": {
            "supplied": {
                "wiring_convention": "W12 through the production geometric port assignment on the twelve-neighbour cell graph",
                "port_assignment": "oph_fpe.core.screen_ports.assign_echosahedral_ports (accepted the twelve-neighbour graph)",
                "join_convention": "child port k glued to parent port k, plain mean/nearest-agreement seam law, area weights recorded",
                "schedule": "random permutation of all seams per round (provenance runs); uniform with replacement (L1 engines)",
                "initial_loads": "seeded integers in {0..5} shared with the L1 lane per level",
                "write_convention": "value-changing repairs write; the every-attempt-writes control is reported",
            },
            "produced": {
                "repairs": True,
                "provenance_order": True,
                "readbacks": True,
                "slow_band_under_w12": True,
                "confluence_receipt": True,
            },
            "not_claimed": {
                "wiring_selected_by_axioms_M1": False,
                "physical_clock": False,
                "continuum_limit": False,
                "physical_position": False,
                "physical_length": False,
                "field_attachment": False,
            },
        },
        "conventions": {
            "carrier": {"ports": PORTS, "seams": carrier.SEAM_COUNT, "antipode": list(carrier.antipode()), "self_checks": carrier_checks},
            "readback": "x = 2 P_slow N; isometric coordinates X = V^T N with V the twelve port directions (G = 4 P_slow = V V^T)",
            "descent_functional": "V = sum_p x_p^2 (flagship); the integer nearest-agreement law lowers V by (d^2 - (d mod 2))/2 per descent and by zero per swap",
            "events": "one event per external (glued or join) seam repair attempt; intra-carrier repairs are carrier-internal",
            "interval": "Alexandrov interval [e, f] = {g : e <= g <= f} in the provenance order; counts include both tips",
            "ordering_fraction": "2C/(N(N-1)) with C the number of strictly comparable pairs",
            "myrheim_meyer": "inverted with oph_fpe.bulk.causet_likeness.invert_myrheim_meyer_fraction",
            "interior": f"an interval is interior when every event lies within {INTERIOR_ANGLE_DEG} degrees of the tip cell centre",
            "pair_counting": f"exact all pairs up to {EXACT_PAIR_LIMIT} events (blocked ancestor bitsets), stratified sample of {SAMPLE_SOURCES} sources by round above",
            "float_rounding": "twelve significant digits",
        },
        "wiring": wiring,
        "dynamics": dynamics,
        "slow_band": slow_band,
        "provenance": provenance,
        "depth": depth,
        "source_net_comparison": source,
        "side_by_side": side_by_side,
        "verdicts": {
            "carrier_self_checks": bool(all(carrier_checks.values())),
            "production_assignment_accepts_w12_all_levels": bool(wiring_ok),
            "float_law_zero_descent_violations_all_levels": bool(float_ok),
            "confluence_one_terminal_hash_equal_component_mean_L3": bool(confluence_ok),
            "integer_law_unique_quotient_all_levels": bool(integer_ok),
            "provenance_runs_descent_exact_and_balanced": bool(provenance_ok),
            "isolated_kernel_equals_4P_slow_within_1e-12": bool(slow_band["isolated_reference"]["kernel_matches_carrier_within_1e-12"] and slow_band["isolated_reference"]["gram_within_1e-12_at_n_300"]),
            "source_net_top_interval_reproduced": bool(source["reproduces_source_net_receipt_top_interval"]),
        },
        "open_items": [
            "selection of the wiring by the axioms (M1) is work in progress; W12 is a declared convention",
            "the refinement limit of the provenance readout across levels is work in progress",
        ],
        "pins": file_pins(),
    }
    return receipt


def write_receipt(receipt: dict[str, Any], path: Path = RECEIPT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(receipt), encoding="ascii")


def load_receipt(path: Path = RECEIPT_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="ascii"))


def check_receipt(path: Path = RECEIPT_PATH, **kwargs: Any) -> None:
    stored = load_receipt(path)
    rebuilt = build_receipt(store_logs=False, **kwargs)
    rebuilt["pins"] = stored["pins"]
    for block in rebuilt["provenance"].values():
        block["stored_log"] = stored["provenance"][f"L{block['levels'][-1]}"]["stored_log"]
    if canonical_json(stored) != canonical_json(rebuilt):
        raise RuntimeError(f"stale receipt: {path} differs from an in-memory rebuild")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="build and write the receipt")
    parser.add_argument("--check", action="store_true", help="rebuild and compare with the stored receipt (stale raises)")
    parser.add_argument("--levels", type=int, nargs="*", default=list(LEVELS))
    parser.add_argument("--depth-levels", type=int, nargs="*", default=list(DEPTH_LEVELS))
    parser.add_argument("--schedules", type=int, default=SCHEDULES)
    parser.add_argument("--engine", default="auto", choices=("auto", "native", "numpy", "python"))
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--receipt", type=Path, default=RECEIPT_PATH)
    args = parser.parse_args(argv)
    if not (args.write or args.check):
        parser.print_help()
        return 1
    timings: dict[str, float] = {}
    started = time.perf_counter()
    kwargs = dict(levels=tuple(args.levels), depth_levels=tuple(args.depth_levels), engine=args.engine, workers=args.workers, schedules=args.schedules, timings=timings)
    if args.check:
        check_receipt(args.receipt, **kwargs)
        print(f"receipt fresh: {args.receipt} ({time.perf_counter() - started:.1f} s)")
        return 0
    receipt = build_receipt(**kwargs)
    write_receipt(receipt, args.receipt)
    total = time.perf_counter() - started
    summary_dir = REPO_ROOT / "runs" / "exact_support_wiring_receipt"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "timing.json").write_text(
        canonical_json(
            {
                "engine": federation.resolve_engine(args.engine),
                "workers": args.workers,
                "total_seconds": round(total, 2),
                "timings_seconds": {k: round(v, 3) for k, v in timings.items()},
                "wall_clock_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        ),
        encoding="ascii",
    )
    print(f"wrote {args.receipt} in {total:.1f} s; verdicts: {receipt['verdicts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
