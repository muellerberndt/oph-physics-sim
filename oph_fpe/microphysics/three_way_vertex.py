from __future__ import annotations

import numpy as np

from oph_fpe.claims import (
    DECLARED_SHAPE_SUBSTRATE_WITNESS,
    SHAPE_VERTEX_SCATTERING_RECEIPT,
)


def three_way_scattering_matrix(dtype=float) -> np.ndarray:
    """Return the declared three-way Shape vertex rule S = (2/3)J - I."""

    base_dtype = np.dtype(dtype)
    if np.issubdtype(base_dtype, np.complexfloating):
        eye_dtype = np.complex128
    else:
        eye_dtype = base_dtype
    j = np.ones((3, 3), dtype=eye_dtype)
    return (2.0 / 3.0) * j - np.eye(3, dtype=eye_dtype)


def scatter_vertex(incoming: np.ndarray) -> np.ndarray:
    incoming = np.asarray(incoming)
    if incoming.shape[-1] != 3:
        raise ValueError("three-way vertex requires last dimension size 3")
    matrix = three_way_scattering_matrix(dtype=incoming.dtype)
    return incoming @ matrix.T


def vertex_power(values: np.ndarray) -> float:
    values = np.asarray(values)
    return float(np.sum(np.abs(values) ** 2))


def scattering_receipt(tol: float = 1.0e-12) -> dict:
    matrix = three_way_scattering_matrix()
    single = np.array([1.0, 0.0, 0.0])
    scattered = matrix @ single
    cyclic = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    eigvals = np.linalg.eigvalsh(matrix)
    checks = {
        "symmetric": bool(np.allclose(matrix, matrix.T, atol=tol)),
        "orthogonal": bool(np.allclose(matrix.T @ matrix, np.eye(3), atol=tol)),
        "involution_S2_equals_I": bool(np.allclose(matrix @ matrix, np.eye(3), atol=tol)),
        "single_channel_rule": bool(np.allclose(scattered, [-1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0], atol=tol)),
        "power_conserved_single_channel": bool(abs(vertex_power(single) - vertex_power(scattered)) < tol),
        "commutes_with_cyclic_permutation": bool(np.allclose(matrix @ cyclic, cyclic @ matrix, atol=tol)),
        "eigenvalues": eigvals.tolist(),
    }
    return {
        "receipt": SHAPE_VERTEX_SCATTERING_RECEIPT,
        "receipt_name": SHAPE_VERTEX_SCATTERING_RECEIPT,
        "matrix": matrix.tolist(),
        "checks": checks,
        "passed": all(value for key, value in checks.items() if key != "eigenvalues"),
        "claim_level": DECLARED_SHAPE_SUBSTRATE_WITNESS,
        "neutral_oph_bulk_claim": False,
        "physical_cmb_prediction": False,
    }
