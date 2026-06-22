from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import expm


ETA = np.diag([-1.0, 1.0, 1.0, 1.0])


def levi_civita(i: int, j: int, k: int) -> int:
    values = [int(i), int(j), int(k)]
    if len(set(values)) < 3:
        return 0
    inversions = sum(values[a] > values[b] for a in range(3) for b in range(a + 1, 3))
    return -1 if inversions % 2 else 1


def rotation_generators() -> list[np.ndarray]:
    generators: list[np.ndarray] = []
    for axis in range(3):
        matrix = np.zeros((4, 4), dtype=float)
        for row in range(3):
            for col in range(3):
                matrix[row + 1, col + 1] = -float(levi_civita(axis, row, col))
        generators.append(matrix)
    return generators


def boost_generators() -> list[np.ndarray]:
    generators: list[np.ndarray] = []
    for axis in range(3):
        matrix = np.zeros((4, 4), dtype=float)
        matrix[0, axis + 1] = 1.0
        matrix[axis + 1, 0] = 1.0
        generators.append(matrix)
    return generators


def lorentz_generators() -> tuple[list[np.ndarray], list[np.ndarray]]:
    return rotation_generators(), boost_generators()


def metric_preservation_error(transform: np.ndarray) -> float:
    transform = np.asarray(transform, dtype=float)
    return float(np.max(np.abs(transform.T @ ETA @ transform - ETA)))


def algebra_membership_error(generator: np.ndarray) -> float:
    generator = np.asarray(generator, dtype=float)
    return float(np.max(np.abs(generator.T @ ETA + ETA @ generator)))


def lorentz_algebra_report(*, sample_rapidity: float = 0.37, sample_angle: float = 0.41) -> dict[str, Any]:
    rotations, boosts = lorentz_generators()
    generators = [*rotations, *boosts]
    membership_errors = [algebra_membership_error(generator) for generator in generators]
    jj_errors: list[float] = []
    jk_errors: list[float] = []
    kk_errors: list[float] = []
    for i in range(3):
        for j in range(3):
            jj = rotations[i] @ rotations[j] - rotations[j] @ rotations[i]
            jk = rotations[i] @ boosts[j] - boosts[j] @ rotations[i]
            kk = boosts[i] @ boosts[j] - boosts[j] @ boosts[i]
            jj_target = sum(levi_civita(i, j, k) * rotations[k] for k in range(3))
            jk_target = sum(levi_civita(i, j, k) * boosts[k] for k in range(3))
            kk_target = -sum(levi_civita(i, j, k) * rotations[k] for k in range(3))
            jj_errors.append(float(np.max(np.abs(jj - jj_target))))
            jk_errors.append(float(np.max(np.abs(jk - jk_target))))
            kk_errors.append(float(np.max(np.abs(kk - kk_target))))
    boost = expm(float(sample_rapidity) * boosts[0])
    rotation = expm(float(sample_angle) * rotations[2])
    composed = boost @ rotation
    metric_errors = [
        metric_preservation_error(boost),
        metric_preservation_error(rotation),
        metric_preservation_error(composed),
    ]
    null = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
        ],
        dtype=float,
    )
    transformed_null = null @ composed.T
    null_norms = -transformed_null[:, 0] ** 2 + np.sum(transformed_null[:, 1:] ** 2, axis=1)
    origin = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    boost_tangents = np.vstack([boost_gen @ origin for boost_gen in boosts])
    tangent_rank = int(np.linalg.matrix_rank(boost_tangents[:, 1:]))
    max_commutator_error = float(max([*jj_errors, *jk_errors, *kk_errors], default=0.0))
    max_membership_error = float(max(membership_errors, default=0.0))
    max_metric_error = float(max(metric_errors, default=0.0))
    max_null_error = float(np.max(np.abs(null_norms))) if null_norms.size else 0.0
    receipt = bool(
        max_commutator_error < 1.0e-12
        and max_membership_error < 1.0e-12
        and max_metric_error < 1.0e-12
        and max_null_error < 1.0e-12
        and tangent_rank == 3
    )
    return {
        "mode": "so_3_1_lorentz_algebra_chart_verifier",
        "minkowski_signature": "(-,+,+,+)",
        "group": "SO+(3,1)",
        "conformal_boundary_group": "Conf+(S2) ~= PSL(2,C)",
        "spatial_homogeneous_space": "H3 = SO+(3,1)/SO(3)",
        "group_dimension": 6,
        "stabilizer_group": "SO(3)",
        "stabilizer_dimension": 3,
        "rotation_generator_count": 3,
        "boost_generator_count": 3,
        "h3_spatial_dimension_from_boost_orbit": tangent_rank,
        "spatial_dimension_derivation": "dim SO+(3,1)-dim SO(3)=6-3=3",
        "max_algebra_membership_error": max_membership_error,
        "max_commutator_error": max_commutator_error,
        "max_metric_preservation_error": max_metric_error,
        "max_null_cone_preservation_error": max_null_error,
        "commutator_relations": {
            "[J_i,J_j]": "epsilon_ijk J_k",
            "[J_i,K_j]": "epsilon_ijk K_k",
            "[K_i,K_j]": "-epsilon_ijk J_k",
        },
        "lorentz_algebra_receipt": receipt,
        "claim_boundary": (
            "finite algebraic chart-level verifier for the paper-side Lorentz branch. "
            "It verifies the so(3,1) generator relations, null-cone preservation, and "
            "3D H3 boost orbit. It does not by itself populate the bulk with observer "
            "records, matter defects, or CMB predictions."
        ),
    }
