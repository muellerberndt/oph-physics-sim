from __future__ import annotations

from typing import Any, Sequence

import numpy as np


PLANCK_H_J_S = 6.62607015e-34
GEV_JOULE = 1.602176634e-10


def source_clock_receipt(
    hamiltonian: Sequence[Sequence[float | complex]],
    *,
    frequency_hz: float,
    perturbation_norm_bound: float = 0.0,
    selected_levels: tuple[int, int] = (0, 1),
    source_packet_verified: bool = False,
    no_target_ancestry: bool = False,
    frequency_role: str = "unit_convention",
    tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    """Certify a finite clock gap and conditionally attach an SI energy scale.

    The eigenvalue calculation is a numerical receipt. It becomes a source-clock
    receipt only when the Hamiltonian packet and its ancestry are independently
    verified. A measured calibration frequency never promotes the source lane.
    """

    matrix = np.asarray(hamiltonian, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("clock Hamiltonian must be a square matrix of dimension at least two")
    if frequency_hz <= 0.0:
        raise ValueError("clock frequency must be positive")
    if perturbation_norm_bound < 0.0:
        raise ValueError("perturbation norm bound must be nonnegative")

    hermitian_residual = float(np.linalg.norm(matrix - matrix.conj().T, ord=2))
    self_adjoint = hermitian_residual <= tolerance
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.conj().T) / 2.0)
    i, j = selected_levels
    if not (0 <= i < j < len(eigenvalues)):
        raise ValueError("selected clock levels must be ordered valid eigenvalue indices")

    gap = float(eigenvalues[j] - eigenvalues[i])
    gap_error = 2.0 * float(perturbation_norm_bound)
    gap_interval = [gap - gap_error, gap + gap_error]
    positive_gap = gap_interval[0] > 0.0
    outside = [
        abs(float(eigenvalues[k] - eigenvalues[level]))
        for level in (i, j)
        for k in range(len(eigenvalues))
        if k not in (i, j)
    ]
    outside_separation = min(outside) if outside else float("inf")
    labels_isolated = outside_separation > gap_error

    scale_interval_joule = None
    central_scale_joule = None
    scale_interval_gev = None
    central_scale_gev = None
    if positive_gap:
        central_scale_joule = PLANCK_H_J_S * frequency_hz / gap
        scale_interval_joule = [
            PLANCK_H_J_S * frequency_hz / gap_interval[1],
            PLANCK_H_J_S * frequency_hz / gap_interval[0],
        ]
        central_scale_gev = central_scale_joule / GEV_JOULE
        scale_interval_gev = [value / GEV_JOULE for value in scale_interval_joule]

    source_frequency_eligible = frequency_role in {"unit_convention", "source_emitted"}
    declared_candidate = bool(
        source_packet_verified is True
        and no_target_ancestry is True
        and source_frequency_eligible
        and self_adjoint
        and positive_gap
        and labels_isolated
    )
    blockers = []
    if not source_packet_verified:
        blockers.append("source_clock_packet_not_verified")
    if not no_target_ancestry:
        blockers.append("source_clock_no_target_ancestry_not_verified")
    if not source_frequency_eligible:
        blockers.append("clock_frequency_is_measurement_calibration")
    if not self_adjoint:
        blockers.append("clock_hamiltonian_not_self_adjoint")
    if not positive_gap:
        blockers.append("clock_gap_not_strictly_positive_under_perturbation")
    if not labels_isolated:
        blockers.append("clock_level_labels_not_isolated")
    blockers.extend(
        [
            "finite_matrix_gap_is_only_a_wzh0_numeric_primitive",
            "operational_source_clock_envelope_not_replayed",
            "runtime_subject_and_same_root_binding_not_verified",
        ]
    )

    return {
        "schema": "oph_wzh_source_clock_receipt_v1",
        "matrix_dimension": int(matrix.shape[0]),
        "hermitian_residual": hermitian_residual,
        "self_adjoint": self_adjoint,
        "eigenvalues": [float(value) for value in eigenvalues],
        "selected_levels": [i, j],
        "dimensionless_gap": gap,
        "gap_interval": gap_interval,
        "gap_perturbation_bound": gap_error,
        "labels_isolated": labels_isolated,
        "frequency_hz": float(frequency_hz),
        "frequency_role": frequency_role,
        "E_star_joule": central_scale_joule,
        "E_star_interval_joule": scale_interval_joule,
        "E_star_GeV": central_scale_gev,
        "E_star_interval_GeV": scale_interval_gev,
        "declared_candidate_conditions_met": declared_candidate,
        "finite_gap_numerical_control_receipt": bool(
            self_adjoint and positive_gap and labels_isolated
        ),
        "source_clock_receipt": False,
        "promotion_allowed": False,
        "blockers": sorted(set(blockers)),
        "claim_boundary": (
            "The gap and unit conversion are WZH0 numerical primitives. This helper "
            "cannot promote an operational source clock from caller flags; a resolved "
            "same-root clock/transition packet and independent verifier are required."
        ),
    }
