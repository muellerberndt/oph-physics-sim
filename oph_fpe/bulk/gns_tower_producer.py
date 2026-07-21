"""Issue-574 producer: cyclicity, modular intersections, and cofinal rates.

The Einstein branch consumes a common GNS tower: cap algebras represented on
one Hilbert space with a cyclic separating vector, modular operators for cap
pairs that agree on a common core, and positive null generators assembling
into translations with future-cone spectrum.  Issue #574 asks for a finite
certified construction with named failure controls.

Everything here is exact finite linear algebra.  The Hilbert space is the
Hilbert-Schmidt space of the twelve-port matrix algebra with inner product
``<a, b> = Tr(rho a* b)`` for the empirical faithful state ``rho`` built from
record-snapshot port vectors (no temperature input, as in the issue-573
producer).  Cap algebras are port-support block subalgebras; the GNS cyclic
vector is the unit; cyclicity and separation residuals are computed by exact
projection onto ``A Omega`` and ``A' Omega``; the finite modular operator of
each cap algebra is solved from the closure of ``S a Omega = a* Omega`` on
``A Omega``; and the modular-intersection comparison compresses the pair
operators onto the intersection core.

Verdicts are recorded fail-closed.  The instrument does not assume the source
dynamics satisfies any clause; each clause is measured, the negative controls
must fail, and a NOT_ATTAINED verdict is an empirical statement about this
source at this cutoff.  No physical promotion follows from any output.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.bulk.physical_h3_kms_source_capture import capture_physical_source
from oph_fpe.bulk.modular_normalization_producer import (
    _NEIGHBORS,
    _snapshot_samples,
)

SCHEMA = "oph.gns-tower-producer.v1"
PHYSICAL_PROMOTION_ALLOWED = False
PORTS = 12
REGULARIZER = 1.0e-6
CYCLICITY_TOLERANCE = 1.0e-6
SUPPORT_FLOOR = 1.0e-12


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256_value(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _empirical_density(capture: Mapping[str, Any]) -> np.ndarray:
    samples = _snapshot_samples(capture)
    moment = samples.T @ samples / samples.shape[0]
    moment = moment + REGULARIZER * np.eye(PORTS)
    return moment / np.trace(moment)


def cap_ports(axis: int) -> tuple[int, ...]:
    """The cap of one antipodal axis: the axis port plus its neighbor ring."""

    return tuple(sorted((axis, *_NEIGHBORS[axis])))


def _block_basis(ports: Sequence[int]) -> list[np.ndarray]:
    """Matrix-unit basis of the port-support block subalgebra plus identity.

    The algebra is ``B(support) (+) C * I_complement``: all matrix units on
    the supported ports, together with the identity on the complement so the
    algebra is unital in the ambient twelve-port algebra.
    """

    basis: list[np.ndarray] = []
    complement = [port for port in range(PORTS) if port not in ports]
    if complement:
        identity_complement = np.zeros((PORTS, PORTS), dtype=complex)
        for port in complement:
            identity_complement[port, port] = 1.0
        basis.append(identity_complement)
    for row in ports:
        for column in ports:
            unit = np.zeros((PORTS, PORTS), dtype=complex)
            unit[row, column] = 1.0
            basis.append(unit)
    return basis


def _gns_gram(basis: Sequence[np.ndarray], rho: np.ndarray) -> np.ndarray:
    count = len(basis)
    gram = np.zeros((count, count), dtype=complex)
    for i in range(count):
        for j in range(count):
            gram[i, j] = np.trace(rho @ basis[i].conj().T @ basis[j])
    return gram


def _projection_residual(
    target: np.ndarray,
    basis: Sequence[np.ndarray],
    rho: np.ndarray,
) -> float:
    """GNS distance from ``target`` to the span of ``basis`` acting on Omega."""

    gram = _gns_gram(basis, rho)
    overlaps = np.asarray(
        [np.trace(rho @ b.conj().T @ target) for b in basis], dtype=complex
    )
    coefficients, *_ = np.linalg.lstsq(gram, overlaps, rcond=None)
    norm_sq = float(np.real(np.trace(rho @ target.conj().T @ target)))
    projected_sq = float(np.real(np.vdot(overlaps, coefficients)))
    residual_sq = max(norm_sq - projected_sq, 0.0)
    return float(np.sqrt(residual_sq) / max(np.sqrt(norm_sq), 1.0e-30))


def _test_frame(seed: int = 574) -> list[np.ndarray]:
    """Fixed countable separating test frame, declared once.

    Deterministic pseudo-random Hermitian test operators plus the twelve
    diagonal units; the frame is declared by seed and never adapted to any
    measured residual.
    """

    generator = np.random.Generator(np.random.PCG64(seed))
    frame: list[np.ndarray] = []
    for port in range(PORTS):
        unit = np.zeros((PORTS, PORTS), dtype=complex)
        unit[port, port] = 1.0
        frame.append(unit)
    for _ in range(8):
        raw = generator.standard_normal((PORTS, PORTS)) + 1j * generator.standard_normal(
            (PORTS, PORTS)
        )
        frame.append((raw + raw.conj().T) / 2.0)
    return frame


def _commutant_basis(ports: Sequence[int]) -> list[np.ndarray]:
    """Commutant of the block algebra: full block on the complement plus
    scalars on the support block."""

    complement = [port for port in range(PORTS) if port not in ports]
    basis: list[np.ndarray] = []
    support_identity = np.zeros((PORTS, PORTS), dtype=complex)
    for port in ports:
        support_identity[port, port] = 1.0
    basis.append(support_identity)
    for row in complement:
        for column in complement:
            unit = np.zeros((PORTS, PORTS), dtype=complex)
            unit[row, column] = 1.0
            basis.append(unit)
    return basis


def _modular_matrix(
    basis: Sequence[np.ndarray], rho: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Finite modular operator of the subalgebra on its GNS subspace.

    Solves ``S a Omega = a* Omega`` in the coefficient coordinates of the
    declared basis and returns ``(Delta, Gram)`` with ``Delta = S^dagger S``
    expressed in the same coordinates (with respect to the Gram metric).
    """

    count = len(basis)
    gram = _gns_gram(basis, rho)
    star = np.zeros((count, count), dtype=complex)
    for j in range(count):
        adjoint = basis[j].conj().T
        overlaps = np.asarray(
            [np.trace(rho @ b.conj().T @ adjoint) for b in basis], dtype=complex
        )
        coefficients, *_ = np.linalg.lstsq(gram, overlaps, rcond=None)
        star[:, j] = coefficients
    gram_inv = np.linalg.pinv(gram)
    delta = gram_inv @ star.conj().T @ gram @ star
    return delta, gram


def _compress_to_core(
    delta: np.ndarray,
    basis: Sequence[np.ndarray],
    core_basis: Sequence[np.ndarray],
    rho: np.ndarray,
) -> np.ndarray:
    """Compress a coefficient-space operator onto the intersection core."""

    gram = _gns_gram(basis, rho)
    cross = np.zeros((len(basis), len(core_basis)), dtype=complex)
    for i, b in enumerate(basis):
        for j, c in enumerate(core_basis):
            cross[i, j] = np.trace(rho @ b.conj().T @ c)
    embed, *_ = np.linalg.lstsq(gram, cross, rcond=None)
    core_gram = _gns_gram(core_basis, rho)
    core_gram_inv = np.linalg.pinv(core_gram)
    return core_gram_inv @ embed.conj().T @ gram @ delta @ embed


def _null_generator_report(capture: Mapping[str, Any]) -> dict[str, Any]:
    """Assemble candidate null translations from the source m4 generators.

    The capture's m4 generator pair supplies two Hermitian directions.  The
    candidate light-cone combinations ``K +/- G`` are tested for positive
    spectrum; four translations require both orientations of two directions.
    The verdict records how many of the four candidates are positive.
    """

    primitives = capture["source_artifacts"]["cap_state_raw_primitives"]

    def _complex(matrix: Any) -> np.ndarray:
        raw = np.asarray(matrix, dtype=float)
        return raw[..., 0] + 1j * raw[..., 1]

    generator_z = _complex(primitives["m4_generator_z"])
    generator_x = _complex(primitives["m4_generator_x"])
    modular = _complex(primitives["modular_generator"])
    candidates = {
        "modular_plus_z": modular + generator_z,
        "modular_minus_z": modular - generator_z,
        "modular_plus_x": modular + generator_x,
        "modular_minus_x": modular - generator_x,
    }
    rows = {}
    positive_count = 0
    for name, matrix in candidates.items():
        eigenvalues = np.linalg.eigvalsh((matrix + matrix.conj().T) / 2.0)
        positive = bool(eigenvalues.min() >= -1.0e-10)
        rows[name] = {
            "minimum_eigenvalue": float(eigenvalues.min()),
            "positive": positive,
        }
        positive_count += int(positive)
    return {
        "candidates": rows,
        "positive_candidate_count": positive_count,
        "future_cone_spectrum_attained": bool(positive_count == 4),
    }


def _tower_level_report(
    capture: Mapping[str, Any],
    *,
    noncyclic_control: bool = False,
    identity_inclusion_control: bool = False,
    support_collapse_control: bool = False,
    disjoint_intersection_control: bool = False,
) -> dict[str, Any]:
    rho = _empirical_density(capture)
    if noncyclic_control:
        # A rank-deficient vector state destroys separation and cyclicity.
        projector = np.zeros((PORTS, PORTS))
        projector[0, 0] = 1.0
        rho = projector
    frame = _test_frame()

    cap_a = cap_ports(0)
    cap_b = cap_ports(5) if not disjoint_intersection_control else (5, 7, 8)
    if support_collapse_control:
        cap_a = ()
    basis_a = (
        [np.eye(PORTS, dtype=complex)]
        if identity_inclusion_control
        else _block_basis(cap_a)
    )
    basis_b = _block_basis(cap_b)
    ambient = _block_basis(tuple(range(PORTS)))

    cyclicity = [
        _projection_residual(test, ambient, rho) for test in frame
    ]
    cap_cyclicity = [
        _projection_residual(test, basis_a, rho) for test in frame
    ]
    commutant_a = _commutant_basis(cap_a)
    separation = [
        _projection_residual(test, commutant_a, rho) for test in frame
    ]
    eigenvalues = np.linalg.eigvalsh(rho)
    support_floor = float(eigenvalues.min())

    intersection_ports = tuple(sorted(set(cap_a) & set(cap_b)))
    intersection_nonempty = bool(intersection_ports)
    modular_residual = None
    if intersection_nonempty and not (
        noncyclic_control or support_collapse_control or identity_inclusion_control
    ):
        core_basis = _block_basis(intersection_ports)
        delta_a, _ = _modular_matrix(basis_a, rho)
        delta_b, _ = _modular_matrix(basis_b, rho)
        delta_core, _ = _modular_matrix(core_basis, rho)
        compressed_a = _compress_to_core(delta_a, basis_a, core_basis, rho)
        compressed_b = _compress_to_core(delta_b, basis_b, core_basis, rho)
        scale = max(np.linalg.norm(delta_core), 1.0e-30)
        modular_residual = float(
            max(
                np.linalg.norm(compressed_a - delta_core) / scale,
                np.linalg.norm(compressed_b - delta_core) / scale,
            )
        )
    return {
        "ambient_cyclicity_max_residual": float(max(cyclicity)),
        "cap_cyclicity_max_residual": float(max(cap_cyclicity)),
        "separation_max_residual": float(max(separation)),
        "support_floor": support_floor,
        "intersection_ports": list(intersection_ports),
        "intersection_nonempty": intersection_nonempty,
        "modular_intersection_residual": modular_residual,
    }


def produce_gns_tower_report(
    *,
    config: Mapping[str, Any] | None = None,
    refinement_config: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Produce the finite GNS-tower report with clause verdicts and controls."""

    main_config = dict(
        {"carrier_count": 32, "cycles": 6, "seed": 20260751}
        if config is None
        else config
    )
    fine_config = dict(
        {"carrier_count": 64, "cycles": 6, "seed": 20260751}
        if refinement_config is None
        else refinement_config
    )
    capture = capture_physical_source(main_config)
    capture_fine = capture_physical_source(fine_config)
    frame_freeze = _sha256_value("test_frame_seed_574")

    main = _tower_level_report(capture)
    fine = _tower_level_report(capture_fine)
    null_assembly = _null_generator_report(capture)

    blockers: list[str] = []
    cyclic_ok = bool(
        main["ambient_cyclicity_max_residual"] < CYCLICITY_TOLERANCE
        and fine["ambient_cyclicity_max_residual"] < CYCLICITY_TOLERANCE
    )
    if not cyclic_ok:
        blockers.append("ambient_cyclicity_residual_above_tolerance")
    separating_ok = bool(main["support_floor"] > SUPPORT_FLOOR)
    if not separating_ok:
        blockers.append("state_support_floor_collapsed")
    intersection_ok = bool(
        main["intersection_nonempty"]
        and main["modular_intersection_residual"] is not None
    )
    if not intersection_ok:
        blockers.append("modular_intersection_not_constructed")
    modular_agreement = (
        main["modular_intersection_residual"]
        if main["modular_intersection_residual"] is not None
        else float("inf")
    )
    fine_agreement = (
        fine["modular_intersection_residual"]
        if fine["modular_intersection_residual"] is not None
        else float("inf")
    )
    intersection_converged = bool(
        modular_agreement < 0.5 and fine_agreement <= modular_agreement * 1.5
    )
    if not intersection_converged:
        blockers.append("modular_intersection_residual_not_converged")
    if not null_assembly["future_cone_spectrum_attained"]:
        blockers.append("future_cone_spectrum_not_attained")

    controls = {
        "support_collapse": _tower_level_report(
            capture, support_collapse_control=True
        ),
        "identity_inclusion": _tower_level_report(
            capture, identity_inclusion_control=True
        ),
        "inconsistent_intersection": _tower_level_report(
            capture, disjoint_intersection_control=True
        ),
        "noncyclic_vector": _tower_level_report(capture, noncyclic_control=True),
    }
    control_rows: dict[str, dict[str, Any]] = {}
    controls_fail_closed = True
    checks = {
        "support_collapse": lambda row: row["cap_cyclicity_max_residual"]
        > main["cap_cyclicity_max_residual"] + 1.0e-9,
        "identity_inclusion": lambda row: row["cap_cyclicity_max_residual"]
        > main["cap_cyclicity_max_residual"] + 1.0e-9,
        "inconsistent_intersection": lambda row: not row["intersection_nonempty"],
        "noncyclic_vector": lambda row: row["support_floor"] <= SUPPORT_FLOOR,
    }
    for name, row in controls.items():
        failed = bool(checks[name](row))
        control_rows[name] = {"control_failure_detected": failed}
        controls_fail_closed = controls_fail_closed and failed
    # Fifth control: one broken generator relation.  Perturb one m4 generator
    # and require the null-assembly verdict to change or a candidate to lose
    # positivity margin.
    perturbed = json.loads(json.dumps(capture, default=float))
    raw = np.asarray(
        perturbed["source_artifacts"]["cap_state_raw_primitives"]["m4_generator_z"],
        dtype=float,
    )
    raw[0, 0, 0] += 10.0
    perturbed["source_artifacts"]["cap_state_raw_primitives"]["m4_generator_z"] = (
        raw.tolist()
    )
    broken = _null_generator_report(perturbed)
    broken_changed = bool(
        broken["candidates"] != null_assembly["candidates"]
    )
    control_rows["broken_generator_relation"] = {
        "control_failure_detected": broken_changed
    }
    controls_fail_closed = controls_fail_closed and broken_changed
    if not controls_fail_closed:
        blockers.append("negative_control_did_not_fail")

    verdict = "ATTAINED" if not blockers else "NOT_ATTAINED"
    report = {
        "schema": SCHEMA,
        "issue": 574,
        "physical_promotion_allowed": PHYSICAL_PROMOTION_ALLOWED,
        "main_config": main_config,
        "refinement_config": fine_config,
        "capture_sha256": capture["capture_sha256"],
        "refinement_capture_sha256": capture_fine["capture_sha256"],
        "test_frame_freeze": frame_freeze,
        "tower_levels": {"main": main, "refined": fine},
        "null_generator_assembly": null_assembly,
        "negative_controls": control_rows,
        "controls_fail_closed": bool(controls_fail_closed),
        "clause_verdicts": {
            "cyclicity_uniform_on_test_frame": cyclic_ok,
            "state_separating_support_floor": separating_ok,
            "modular_intersection_constructed": intersection_ok,
            "modular_intersection_converged": intersection_converged,
            "future_cone_spectrum": bool(
                null_assembly["future_cone_spectrum_attained"]
            ),
        },
        "verdict": verdict,
        "GNS_TOWER_CLAUSES_RECEIPT": bool(verdict == "ATTAINED"),
        "blockers": sorted(set(blockers)),
        "claim_boundary": (
            "Finite issue-574 instrument: exact GNS, modular, and intersection "
            "linear algebra on the twelve-port block algebras with the "
            "empirical faithful state. Clause verdicts are measured, not "
            "assumed, and a NOT_ATTAINED verdict is an empirical result about "
            "this source at this cutoff. The instrument does not construct "
            "the continuum second-quantized tower, the C ell^(4+eta) small- "
            "diamond tails, or any physical translation group, and no "
            "physical promotion follows from any output."
        ),
    }
    if output_path is not None:
        Path(output_path).write_text(_canonical_json(report), encoding="utf-8")
    return report
