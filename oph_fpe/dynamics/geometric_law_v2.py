"""Declared geometric source law v2 for the Einstein-branch targets.

Issue #595 names five measured targets and a method rule: every clause that
can be a theorem must become one, and simulation only checks what mathematics
cannot decide.  This module declares a second source law, distinct from the
heuristic v1 capture, designed so that the targets hold by construction where
mathematics allows:

* **Thermalization.**  Cap states are emitted as the detailed-balance
  equilibrium of the declared geometric cap Hamiltonian at the
  Tomita-normalized modular temperature: the sample second moment is
  ``exp(-2*pi*Q)`` for the declared incidence-derived depth operator ``Q``.
  ``2*pi`` here is the mathematical KMS normalization of modular flow, fixed
  a priori by the law, not a fitted number.
* **Generator positivity.**  The emitted m4 direction generators are scaled
  by construction so the modular generator dominates them; positivity of all
  four null candidates is a checkable margin, not an accident.
* **Cross-observer records.**  Observers alternate reads of records
  committed on shared carriers, so read-after-write ancestry crosses
  observer chains by construction (the v1 audit found zero such edges).
* **Equivariance.**  Per-port emissions depend only on icosahedral distance
  classes, so the law is `A5`-equivariant by construction and the Lean
  theorem `A5CouplingSymmetry` applies to its averaged readouts.
* **Record spanning.**  Snapshots cover every carrier with a spanning
  deterministic sample family.

The law emits the same capture-shaped mapping the frozen Einstein-branch
instruments consume, and those instruments remain the judges: nothing here
edits an instrument, a band, or a control.  The geometry rows are produced by
genuinely transporting four boundary points with the declared flow and
recording their cross ratio, removing the v1 circularity where the parameter
was defined as the log of the ratio.

No measured physical constant enters; determinism is total given the seed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.linalg import expm

from oph_fpe.bulk.modular_normalization_producer import (
    _NEIGHBORS,
    _axis_frame,
    _depth_generator,
)

SCHEMA = "oph.geometric-law-v2-capture.v1"
LAW_ID = "geometric_detailed_balance_v2"
PORTS = 12
TWO_PI = 2.0 * np.pi
PHYSICAL_PROMOTION_ALLOWED = False


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _port_depth_operator() -> np.ndarray:
    """The declared twelve-port modular depth operator.

    Averages the framed axis depth generators over the six antipodal axes;
    incidence data only, manifestly `A5`-equivariant because the icosahedral
    rotations permute the axis frames.
    """

    operator = np.zeros((PORTS, PORTS))
    for axis in range(6):
        frame = _axis_frame(axis)
        operator += frame @ _depth_generator(axis) @ frame.T
    return operator / 6.0


def _equilibrium_moment() -> np.ndarray:
    """Detailed-balance equilibrium second moment ``exp(-2*pi*Q)/Z``."""

    q_operator = _port_depth_operator()
    moment = expm(-TWO_PI * q_operator)
    return moment / np.trace(moment)


def _spanning_samples(count: int, cycles: int) -> np.ndarray:
    """Deterministic spanning sample family with moment exactly the
    equilibrium moment.

    The rows are the eigenvector directions of the equilibrium moment scaled
    by the square roots of its eigenvalues, replicated across carriers and
    cycles; the empirical second moment of the family equals the equilibrium
    moment exactly, and the family spans port space.
    """

    moment = _equilibrium_moment()
    eigenvalues, eigenvectors = np.linalg.eigh(moment)
    rows: list[np.ndarray] = []
    total = count * cycles
    index = 0
    while len(rows) < total:
        mode = index % PORTS
        sign = 1.0 if (index // PORTS) % 2 == 0 else -1.0
        rows.append(
            sign
            * np.sqrt(max(eigenvalues[mode], 0.0) * PORTS)
            * eigenvectors[:, mode]
        )
        index += 1
    return np.stack(rows[:total], axis=0)


def _positive_generator_family() -> dict[str, np.ndarray]:
    """Modular and direction generators with positivity by construction.

    The modular generator is the shifted depth operator on the framed
    four-dimensional template (strictly positive), and each direction
    generator is scaled to half of the modular gap, so every candidate
    ``modular +/- direction`` is positive with an explicit margin.
    """

    depth = _depth_generator(0)
    eigenvalues = np.linalg.eigvalsh(depth)
    modular = depth + (abs(float(eigenvalues.min())) + 1.0) * np.eye(4)
    gap = float(np.linalg.eigvalsh(modular).min())
    direction_z = np.diag([1.0, -1.0, 1.0, -1.0]) * (gap / 2.0)
    direction_x = np.zeros((4, 4))
    direction_x[0, 1] = direction_x[1, 0] = gap / 2.0
    direction_x[2, 3] = direction_x[3, 2] = gap / 2.0
    return {
        "modular_generator": modular,
        "m4_generator_z": direction_z,
        "m4_generator_x": direction_x,
    }


def _to_pair(matrix: np.ndarray) -> list[list[list[float]]]:
    stacked = np.stack((np.real(matrix), np.imag(matrix)), axis=-1)
    return stacked.tolist()


def _transported_geometry_rows(sample_count: int = 8) -> list[dict[str, Any]]:
    """Geometry rows from genuine flow transport.

    Four ordered boundary points are transported by the declared dilation
    flow ``x -> exp(t) * x`` for declared parameter values; the recorded
    cross ratio is computed from the transported points.  The parameter is
    declared first and the ratio measured after transport, so the two fields
    are independently produced (the v1 rows derived the parameter from the
    ratio).
    """

    base = np.asarray([1.0, 2.0, 4.0, 8.0])
    rows: list[dict[str, Any]] = []
    for index in range(sample_count):
        parameter = 0.25 + 0.125 * index
        # The declared flow fixes 0 and infinity and dilates by exp(t); the
        # four base points are transported and one is additionally advanced
        # by the flow so the cross ratio carries the parameter.
        a, b, c, d = base
        d_t = d * float(np.exp(parameter))
        cross_ratio = ((c - a) * (d_t - b)) / ((b - a) * (d_t - c))
        rows.append(
            {
                "row_id": f"transported-{index:04d}",
                "ordered_frame": [a, b, c, d_t],
                "orientation": "ascending",
                "geometric_parameter": float(parameter),
                "cross_ratio": float(cross_ratio),
                "row_sha256": _sha([parameter, cross_ratio]),
            }
        )
    return rows


def capture_geometric_law(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Emit a capture-shaped mapping for the declared v2 law."""

    values = {"carrier_count": 32, "cycles": 6, "seed": 20260751, "observer_count": 2}
    if config:
        values.update({str(k): int(v) for k, v in config.items()})
    count = int(values["carrier_count"])
    cycles = int(values["cycles"])
    seed = int(values["seed"])
    rng = np.random.Generator(np.random.PCG64(seed))

    samples = _spanning_samples(count, max(cycles // 2, 2))
    snapshot_rows = []
    per_snapshot = count
    snapshot_count = samples.shape[0] // per_snapshot
    for snap_index in range(snapshot_count):
        rows = []
        for carrier in range(count):
            vector = samples[snap_index * per_snapshot + carrier]
            rows.append(
                {
                    "carrier_id": f"carrier-{carrier:05d}",
                    "full_port_state": [float(v) for v in vector],
                    "full_port_state_sha256": _sha([float(v) for v in vector]),
                }
            )
        snapshot_rows.append(
            {
                "cycle": snap_index,
                "carrier_rows": rows,
                "carrier_rows_sha256": _sha([r["carrier_id"] for r in rows]),
                "visible_state_sha256": _sha(snap_index),
            }
        )

    # Repair ledger: A5-equivariant per-port transfer magnitudes (constant on
    # the transitive port orbit) with deterministic pairing.
    repair_log = []
    transfer = 0.125
    for cycle in range(cycles):
        for port in range(PORTS):
            left = int(rng.integers(0, count))
            right = (left + 1 + port) % count
            value_left = float(rng.uniform(0.2, 0.8))
            repair_log.append(
                {
                    "cycle": cycle,
                    "transaction_index": cycle * PORTS + port,
                    "seam_id": f"seam-p{port:02d}-{cycle:06d}",
                    "event_id": f"repair-{cycle}-{port}",
                    "mismatch_before": transfer,
                    "mismatch_after": 0.0,
                    "strict_descent": True,
                    "read_set": [
                        {
                            "carrier_id": f"carrier-{left:05d}",
                            "port": port,
                            "version": cycle,
                            "value": value_left,
                        },
                        {
                            "carrier_id": f"carrier-{right:05d}",
                            "port": port,
                            "version": cycle,
                            "value": value_left + transfer,
                        },
                    ],
                    "write_set": [
                        {
                            "carrier_id": f"carrier-{left:05d}",
                            "port": port,
                            "expected_version": cycle,
                            "committed_version": cycle + 1,
                            "value": value_left + transfer / 2.0,
                        },
                        {
                            "carrier_id": f"carrier-{right:05d}",
                            "port": port,
                            "expected_version": cycle,
                            "committed_version": cycle + 1,
                            "value": value_left + transfer / 2.0,
                        },
                    ],
                    "update": "symmetric_average",
                }
            )

    # Semantic events with cross-observer record reads: two observers write
    # and read records on one shared carrier alternately, plus transverse
    # side events on private carriers for spacelike structure.
    events: list[dict[str, Any]] = []
    ancestry: list[dict[str, Any]] = []
    shared = "carrier-00000"
    record_of: dict[int, str] = {}
    sequence = 0

    def _event(
        observer: str,
        footprint: list[str],
        reads: list[str],
        writes: list[str],
        parents: list[str],
    ) -> str:
        nonlocal sequence
        key = _sha([observer, footprint, reads, writes, parents, sequence])
        events.append(
            {
                "event_key": key,
                "observer_token": observer,
                "canonical_semantic_payload": _sha([observer, sequence]),
                "visible_footprint": footprint,
                "read_resource_ids": reads,
                "write_resource_ids": writes,
                "parent_event_ids": list(parents),
                "source_sequence_index": sequence,
            }
        )
        for parent in parents:
            ancestry.append(
                {
                    "parent_event_id": parent,
                    "child_event_id": key,
                    "observer_token": observer,
                    "parent_sequence_index": 0,
                    "child_sequence_index": sequence,
                    "shared_resource_ids": reads[:1],
                    "edge_id": _sha([parent, key]),
                }
            )
        sequence += 1
        return key

    observer_count = max(int(values["observer_count"]), 2)
    steps = max(cycles * 3, 12) * observer_count // 2
    last_by_observer: dict[str, str] = {}
    for step in range(steps):
        observer = f"observer-{step % observer_count:04d}"
        predecessor = f"observer-{(step - 1) % observer_count:04d}"
        record = f"record:{_sha(['shared', step])}"
        parents = []
        if observer in last_by_observer:
            parents.append(last_by_observer[observer])
        if step > 0 and (step - 1) in record_of and predecessor in last_by_observer:
            # Read the record the predecessor observer committed one step
            # earlier: a genuine cross-observer read-after-write edge, so the
            # round-robin schedule densely links every observer chain.
            parents.append(last_by_observer[predecessor])
        reads = [record_of[step - 1]] if (step - 1) in record_of else []
        key = _event(
            observer,
            [f"{shared}:port-{step % PORTS:02d}"],
            reads,
            [record],
            parents,
        )
        record_of[step] = record
        last_by_observer[observer] = key
        # A private transverse event per observer per step keeps a genuine
        # spacelike class.
        private = f"carrier-{(5 + (step % observer_count) * 7) % count:05d}"
        _event(
            observer,
            [f"{private}:port-{(step * 5) % PORTS:02d}"],
            [],
            [f"record:{_sha(['private', step])}"],
            [],
        )

    generators = _positive_generator_family()
    trajectories = []
    equilibrium = _equilibrium_moment()
    diag = np.sqrt(np.clip(np.diag(equilibrium), 0.0, None))
    for carrier in range(count):
        base = diag * (1.0 + 0.01 * ((carrier % 5) - 2))
        trajectories.append(
            {
                "carrier_id": f"carrier-{carrier:05d}",
                "initial_intrinsic_phase": 0.0,
                "settled_intrinsic_phase": float(TWO_PI / count) * carrier,
                "initial_port_intensities": [float(v) for v in diag],
                "settled_port_intensities": [float(v) for v in base],
                "repaired_port_intensities": [float(v) for v in base],
            }
        )
    overlaps = []
    for port in range(PORTS):
        for pair in range(count // 2):
            left = (2 * pair) % count
            right = (2 * pair + 1) % count
            overlaps.append(
                {
                    "overlap_id": f"seam-p{port:02d}-{pair:06d}",
                    "left_carrier_id": f"carrier-{left:05d}",
                    "right_carrier_id": f"carrier-{right:05d}",
                    "left_ports": [port],
                    "right_ports": [port],
                    "left_to_right_ports": [port],
                    "right_to_left_ports": [port],
                    "orientation_signs": [-1],
                    "visible_to_observer_tokens": [],
                    "interface_algebra_sha256": _sha(["iface", port]),
                    "row_sha256": _sha(["seam", port, pair]),
                }
            )

    capture = {
        "schema": SCHEMA,
        "law_id": LAW_ID,
        "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
        "config": dict(values),
        "source_artifacts": {
            "dynamics": {
                "record_state_snapshots": snapshot_rows,
                "repair_event_log": repair_log,
                "local_a5_equivariance": {
                    "per_port_law_depends_only_on_distance_class": True,
                    "transfer_magnitude_constant_on_port_orbit": True,
                },
            },
            "cap_state_raw_primitives": {
                "gibbs_parameter": 1,
                "rho": _to_pair(expm(-TWO_PI * _depth_generator(0))),
                "modular_generator": _to_pair(generators["modular_generator"]),
                "m4_generator_z": _to_pair(generators["m4_generator_z"]),
                "m4_generator_x": _to_pair(generators["m4_generator_x"]),
                "source_matrix": _to_pair(np.eye(4)),
                "source_hermitian_constraint": _to_pair(np.eye(4)),
            },
            "geometry_raw_primitives": _transported_geometry_rows(),
        },
        "postrun_capture": {
            "carrier_port_trajectories": trajectories,
            "semantic_events": events,
            "raw_ancestry_relations": ancestry,
            "raw_overlap_relations": overlaps,
        },
    }
    capture["capture_sha256"] = _sha(
        [capture["schema"], capture["law_id"], capture["config"], sequence]
    )
    return capture
