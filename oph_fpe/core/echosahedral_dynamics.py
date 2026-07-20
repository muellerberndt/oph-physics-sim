"""Target-blind local dynamics for a twelve-port echosahedral carrier.

The regular icosahedral incidence is used only as a hidden local coupling
presentation.  No global S2/H3 coordinates or clock candidate enter this
module.  The finite unitary channel is an exact A5 intertwiner, so independently
reorienting a carrier commutes with its local propagation.

This is a carrier-level construction, not a BW/KMS clock.  Its dimensionless
evolution parameter has no physical angular normalization, and all physical
clock/emergence receipts remain false.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
from scipy.linalg import expm

from oph_fpe.core.icosahedral import (
    build_geodesic_icosahedral_tower,
    icosahedral_a5_port_permutations,
)


@dataclass(frozen=True)
class LocalRecurrentCarrierState:
    """Complex twelve-channel state for a batch of local carriers."""

    amplitudes: np.ndarray
    intrinsic_phase: np.ndarray

    def __post_init__(self) -> None:
        amplitudes = np.array(self.amplitudes, dtype=np.complex128, copy=True)
        amplitudes = _validated_amplitudes(amplitudes)
        phase = np.array(self.intrinsic_phase, dtype=float, copy=True)
        if phase.shape != (amplitudes.shape[0],) or not np.all(np.isfinite(phase)):
            raise ValueError("intrinsic_phase must contain one finite value per carrier")
        if np.any((phase < 0.0) | (phase >= 1.0)):
            raise ValueError("intrinsic_phase must be represented in [0, 1)")
        amplitudes.setflags(write=False)
        phase.setflags(write=False)
        object.__setattr__(self, "amplitudes", amplitudes)
        object.__setattr__(self, "intrinsic_phase", phase)

    @property
    def carrier_count(self) -> int:
        return int(self.amplitudes.shape[0])


def reference_icosahedral_coupling() -> np.ndarray:
    """Return the 12x12 combinatorial Laplacian of the local carrier."""

    base = build_geodesic_icosahedral_tower(0).levels[0]
    adjacency = np.zeros((12, 12), dtype=float)
    edges = np.asarray(base.edges, dtype=np.int64)
    adjacency[edges[:, 0], edges[:, 1]] = 1.0
    adjacency[edges[:, 1], edges[:, 0]] = 1.0
    return np.diag(np.sum(adjacency, axis=1)) - adjacency


def initialize_local_recurrent_carriers(
    carrier_count: int,
    *,
    seed: int,
) -> LocalRecurrentCarrierState:
    """Initialize normalized source states without a candidate clock scale."""

    count = int(carrier_count)
    if count <= 0:
        raise ValueError("carrier_count must be positive")
    rng = np.random.default_rng(int(seed))
    amplitudes = rng.normal(size=(count, 12)) + 1j * rng.normal(size=(count, 12))
    norms = np.linalg.norm(amplitudes, axis=1, keepdims=True)
    amplitudes = np.asarray(amplitudes / norms, dtype=np.complex128)
    phase = rng.random(count, dtype=np.float64)
    return LocalRecurrentCarrierState(amplitudes=amplitudes, intrinsic_phase=phase)


def propagate_local_recurrent_carriers(
    state: LocalRecurrentCarrierState,
    *,
    intrinsic_step: float,
    coupling_strength: float = 1.0,
) -> LocalRecurrentCarrierState:
    """Apply one A5-equivariant local unitary and an R/Z phase increment."""

    amplitudes = _validated_amplitudes(state.amplitudes)
    step = float(intrinsic_step)
    strength = float(coupling_strength)
    if not math.isfinite(step) or not math.isfinite(strength):
        raise ValueError("intrinsic_step and coupling_strength must be finite")
    coupling = reference_icosahedral_coupling()
    unitary = expm(-1j * step * strength * coupling)
    propagated = amplitudes @ unitary.T
    phase = np.mod(np.asarray(state.intrinsic_phase, dtype=float) + step, 1.0)
    if phase.shape != (amplitudes.shape[0],) or not np.all(np.isfinite(phase)):
        raise ValueError("intrinsic_phase must contain one finite value per carrier")
    return LocalRecurrentCarrierState(
        amplitudes=np.asarray(propagated, dtype=np.complex128),
        intrinsic_phase=phase,
    )


def local_port_statistics(state: LocalRecurrentCarrierState) -> np.ndarray:
    """Return quotient-visible per-port intensities; hidden XYZ is not consulted."""

    amplitudes = _validated_amplitudes(state.amplitudes)
    return np.asarray(np.abs(amplitudes) ** 2, dtype=float)


def local_a5_dynamics_report(
    *,
    intrinsic_step: float = 0.137,
    coupling_strength: float = 1.0,
    tolerance: float = 5.0e-12,
) -> dict[str, Any]:
    """Recompute A5 covariance, unitarity, spectrum, and clock firewalls."""

    step = float(intrinsic_step)
    strength = float(coupling_strength)
    if not all(math.isfinite(value) for value in (step, strength, tolerance)):
        raise ValueError("audit parameters must be finite")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    coupling = reference_icosahedral_coupling()
    unitary = expm(-1j * step * strength * coupling)
    unitary_one_period_later = expm(-1j * (step + 1.0) * strength * coupling)
    identity = np.eye(12, dtype=np.complex128)
    unitarity_residual = float(
        np.linalg.norm(unitary.conj().T @ unitary - identity, ord="fro")
    )
    commutator_residuals: list[float] = []
    equivariance_residuals: list[float] = []
    probe = initialize_local_recurrent_carriers(3, seed=0xA5).amplitudes
    for permutation in icosahedral_a5_port_permutations():
        matrix = _permutation_matrix(permutation)
        commutator_residuals.append(
            float(np.linalg.norm(matrix @ coupling - coupling @ matrix, ord="fro"))
        )
        left = (probe @ matrix.T) @ unitary.T
        right = (probe @ unitary.T) @ matrix.T
        equivariance_residuals.append(float(np.linalg.norm(left - right, ord="fro")))
    eigenvalues = np.linalg.eigvalsh(coupling)
    multiplicities = _spectral_multiplicities(eigenvalues)
    maximum_commutator = max(commutator_residuals, default=math.inf)
    maximum_equivariance = max(equivariance_residuals, default=math.inf)
    nontriviality_norm = float(np.linalg.norm(unitary - identity, ord="fro"))
    period_descent_residual = float(
        np.linalg.norm(unitary_one_period_later - unitary, ord="fro")
    )
    phase_probe = LocalRecurrentCarrierState(
        amplitudes=np.eye(1, 12, dtype=np.complex128),
        intrinsic_phase=np.asarray([0.95], dtype=float),
    )
    phase_after = propagate_local_recurrent_carriers(
        phase_probe,
        intrinsic_step=0.1,
        coupling_strength=strength,
    ).intrinsic_phase
    phase_register_wrap_receipt = bool(
        phase_after.shape == (1,) and abs(float(phase_after[0]) - 0.05) <= tolerance
    )
    dynamics_descends_to_r_mod_z = period_descent_residual <= tolerance
    nontrivial = nontriviality_norm > tolerance
    receipt = bool(
        unitarity_residual <= tolerance
        and maximum_commutator <= tolerance
        and maximum_equivariance <= tolerance
        and sorted(row["multiplicity"] for row in multiplicities) == [1, 3, 3, 5]
    )
    payload = {
        "schema": "oph.echosahedral_local_recurrent_dynamics.v1",
        "carrier_port_count": 12,
        "local_edge_count": 30,
        "local_face_count": 20,
        "a5_action_count": len(icosahedral_a5_port_permutations()),
        "coupling": "combinatorial_icosahedral_graph_laplacian",
        "evolution": "finite_unitary_exp_minus_i_u_gL",
        "intrinsic_phase_space": "R_mod_Z",
        "intrinsic_step": step,
        "coupling_strength": strength,
        "coupling_spectrum": multiplicities,
        "unitarity_residual": unitarity_residual,
        "maximum_a5_coupling_commutator_residual": maximum_commutator,
        "maximum_a5_propagation_equivariance_residual": maximum_equivariance,
        "propagation_nontriviality_norm": nontriviality_norm,
        "one_phase_period_propagator_residual": period_descent_residual,
        "hidden_xyz_coordinates_used_by_dynamics": False,
        "global_support_chart_used_by_dynamics": False,
        "candidate_clock_scale_used_by_dynamics": False,
        "LOCAL_A5_EQUIVARIANT_PROPAGATION_RECEIPT": receipt,
        "LOCAL_NONTRIVIAL_REVERSIBLE_PROPAGATION_RECEIPT": bool(
            receipt and nontrivial
        ),
        "LOCAL_PHASE_REGISTER_MOD_ONE_RECEIPT": phase_register_wrap_receipt,
        "LOCAL_DYNAMICS_DESCENDS_TO_R_MOD_Z_RECEIPT": bool(
            receipt and nontrivial and dynamics_descends_to_r_mod_z
        ),
        "LOCAL_INTRINSIC_R_MOD_Z_RECURRENCE_RECEIPT": False,
        "PHYSICAL_2PI_CLOCK_SELECTION_RECEIPT": False,
        "BW_KMS_CLOCK_RECEIPT": False,
        "PHYSICAL_H3_KMS_EMERGENCE_RECEIPT": False,
        "claim_boundary": (
            "This certifies a target-blind finite local A5-equivariant unitary "
            "carrier channel and separately audits a mod-one phase register. The "
            "generic propagator does not descend to that circle, and neither object "
            "chooses radians, 2pi, a KMS temperature, an H3 frame, or physical time."
        ),
    }
    payload["dynamics_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _validated_amplitudes(value: np.ndarray) -> np.ndarray:
    amplitudes = np.asarray(value, dtype=np.complex128)
    if amplitudes.ndim != 2 or amplitudes.shape[1] != 12:
        raise ValueError("carrier amplitudes must have shape (carrier_count, 12)")
    if not np.all(np.isfinite(amplitudes.real)) or not np.all(
        np.isfinite(amplitudes.imag)
    ):
        raise ValueError("carrier amplitudes must be finite")
    return amplitudes


def _permutation_matrix(permutation: tuple[int, ...]) -> np.ndarray:
    matrix = np.zeros((12, 12), dtype=float)
    matrix[np.asarray(permutation, dtype=np.int64), np.arange(12)] = 1.0
    return matrix


def _spectral_multiplicities(eigenvalues: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in np.asarray(eigenvalues, dtype=float):
        if rows and abs(value - float(rows[-1]["eigenvalue"])) <= 1.0e-10:
            rows[-1]["multiplicity"] += 1
        else:
            rows.append({"eigenvalue": float(value), "multiplicity": 1})
    return rows


__all__ = [
    "LocalRecurrentCarrierState",
    "initialize_local_recurrent_carriers",
    "local_a5_dynamics_report",
    "local_port_statistics",
    "propagate_local_recurrent_carriers",
    "reference_icosahedral_coupling",
]
