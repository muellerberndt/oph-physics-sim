from __future__ import annotations

from typing import Any

import numpy as np

from oph_fpe.bulk.cap_geometry import RoundCap


def minkowski_dot(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Minkowski inner product with signature (-,+,+,+)."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return -left[..., 0] * right[..., 0] + np.sum(left[..., 1:] * right[..., 1:], axis=-1)


def cap_normal(cap: RoundCap) -> np.ndarray:
    """Represent a round S2 cap by its de Sitter normal in R^{3,1}."""

    cap = cap.normalized()
    sin_theta = max(float(np.sin(cap.theta0)), 1e-12)
    normal = np.empty(4, dtype=float)
    normal[0] = float(np.cos(cap.theta0)) / sin_theta
    normal[1:] = cap.axis / sin_theta
    return normal


def cap_normals(caps: list[RoundCap]) -> np.ndarray:
    if not caps:
        return np.zeros((0, 4), dtype=float)
    return np.vstack([cap_normal(cap) for cap in caps])


def boundary_null_directions(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return np.column_stack([np.ones(points.shape[0], dtype=float), points])


def cap_boundary_residual(points: np.ndarray, cap: RoundCap) -> np.ndarray:
    return minkowski_dot(boundary_null_directions(points), cap_normal(cap))


def cap_gram_matrix(caps: list[RoundCap]) -> np.ndarray:
    normals = cap_normals(caps)
    if normals.size == 0:
        return np.zeros((0, 0), dtype=float)
    return -np.outer(normals[:, 0], normals[:, 0]) + normals[:, 1:] @ normals[:, 1:].T


def cap_normal_report(caps: list[RoundCap]) -> dict[str, Any]:
    normals = cap_normals(caps)
    norms = minkowski_dot(normals, normals) if normals.size else np.zeros(0, dtype=float)
    gram = cap_gram_matrix(caps)
    max_norm_error = float(np.max(np.abs(norms - 1.0))) if norms.size else 0.0
    return {
        "mode": "s2_cap_de_sitter_normals",
        "cap_count": len(caps),
        "minkowski_signature": "(-,+,+,+)",
        "max_unit_normal_error": max_norm_error,
        "unit_normal_receipt": bool(max_norm_error < 1e-10),
        "gram_min": float(np.min(gram)) if gram.size else None,
        "gram_max": float(np.max(gram)) if gram.size else None,
        "claim_boundary": (
            "finite algebraic bridge from round S2 caps to Lorentzian de Sitter cap normals; "
            "not a record-populated spatial bulk claim"
        ),
    }

