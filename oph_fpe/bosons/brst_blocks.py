from __future__ import annotations

from itertools import permutations
from typing import Any, Sequence

import numpy as np


def decode_complex(value: float | int | complex | Sequence[float]) -> complex:
    if isinstance(value, complex):
        return value
    if isinstance(value, (float, int)):
        return complex(value)
    if len(value) != 2:
        raise ValueError("complex values must be scalars or [real, imaginary] pairs")
    return complex(float(value[0]), float(value[1]))


def decode_matrix_polynomial(raw: Sequence[Sequence[Sequence[Any]]]) -> np.ndarray:
    """Decode coefficient matrices ordered from s^0 through s^degree."""

    coefficients = np.asarray(
        [[[decode_complex(value) for value in row] for row in matrix] for matrix in raw],
        dtype=np.complex128,
    )
    if coefficients.ndim != 3:
        raise ValueError("matrix polynomial must have shape (degree+1, n, n)")
    if coefficients.shape[1] != coefficients.shape[2] or coefficients.shape[1] == 0:
        raise ValueError("matrix polynomial coefficient matrices must be nonempty and square")
    return coefficients


def evaluate_block(coefficients: np.ndarray, s_value: complex) -> np.ndarray:
    result = np.zeros(coefficients.shape[1:], dtype=np.complex128)
    for matrix in coefficients[::-1]:
        result = result * s_value + matrix
    return result


def determinant_polynomial(coefficients: np.ndarray) -> np.ndarray:
    """Return ascending coefficients of det(sum_k A_k s^k).

    The direct permutation expansion is intentionally small and transparent;
    boson mixing blocks in this backend are expected to be low-dimensional.
    """

    dimension = coefficients.shape[1]
    determinant = np.zeros(dimension * (coefficients.shape[0] - 1) + 1, dtype=np.complex128)
    for permutation in permutations(range(dimension)):
        inversions = sum(
            permutation[i] > permutation[j]
            for i in range(dimension)
            for j in range(i + 1, dimension)
        )
        term = np.asarray([1.0 + 0.0j])
        for row, column in enumerate(permutation):
            term = np.polynomial.polynomial.polymul(term, coefficients[:, row, column])
        determinant[: len(term)] += (-1 if inversions % 2 else 1) * term
    while len(determinant) > 1 and abs(determinant[-1]) < 1.0e-14:
        determinant = determinant[:-1]
    return determinant


def block_structure_receipt(
    raw_coefficients: Sequence[Sequence[Sequence[Any]]],
    *,
    block_id: str,
    block_kind: str,
    source_kernel_verified: bool = False,
    ward_identity_verified: bool = False,
    slavnov_taylor_verified: bool = False,
    nielsen_identity_verified: bool = False,
    tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    coefficients = decode_matrix_polynomial(raw_coefficients)
    determinant = determinant_polynomial(coefficients)
    roots = np.polynomial.polynomial.polyroots(determinant)
    photon_factor = bool(block_kind == "neutral" and abs(determinant[0]) <= tolerance)
    identities = bool(
        ward_identity_verified is True
        and slavnov_taylor_verified is True
        and nielsen_identity_verified is True
    )
    declared_candidate = bool(
        source_kernel_verified is True
        and identities
        and (block_kind != "neutral" or photon_factor)
    )
    blockers = []
    if not source_kernel_verified:
        blockers.append("source_two_point_kernel_not_verified")
    if not identities:
        blockers.append("brst_ward_nielsen_identity_packet_incomplete")
    if block_kind == "neutral" and not photon_factor:
        blockers.append("ward_protected_photon_factor_not_verified")
    blockers.extend(
        [
            "matrix_polynomial_backend_is_wzh0_synthetic_control",
            "caller_identity_flags_are_nonpromoting",
            "artifact_resolving_brst_checker_not_implemented",
        ]
    )
    return {
        "schema": "oph_wzh_brst_block_receipt_v1",
        "block_id": block_id,
        "block_kind": block_kind,
        "matrix_dimension": int(coefficients.shape[1]),
        "polynomial_degree": int(len(determinant) - 1),
        "determinant_coefficients": [[float(value.real), float(value.imag)] for value in determinant],
        "roots": [[float(value.real), float(value.imag)] for value in roots],
        "ward_photon_factor_at_s_zero": photon_factor,
        "source_kernel_verified": bool(source_kernel_verified),
        "identity_packet_verified": identities,
        "declared_candidate_conditions_met": declared_candidate,
        "matrix_polynomial_control_receipt": bool(
            coefficients.size and np.all(np.isfinite(coefficients))
        ),
        "brst_block_receipt": False,
        "promotion_allowed": False,
        "blockers": sorted(set(blockers)),
        "claim_boundary": (
            "This matrix-polynomial path is an unconditionally nonpromoting WZH0 "
            "control. Caller declarations cannot certify a physical inverse two-point "
            "block; production requires resolved analytic kernels and an independent "
            "BRST/Ward/Slavnov-Taylor/Nielsen checker."
        ),
    }
