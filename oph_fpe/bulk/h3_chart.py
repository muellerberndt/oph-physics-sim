from __future__ import annotations

from typing import Any

import numpy as np

from oph_fpe.bulk.cap_normals import cap_normals, minkowski_dot
from oph_fpe.bulk.cap_geometry import RoundCap


def h3_origin() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def h3_point_from_tangent(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,):
        raise ValueError("H3 tangent vector must have shape (3,)")
    radius = float(np.linalg.norm(vector))
    if radius < 1e-15:
        return h3_origin()
    direction = vector / radius
    point = np.empty(4, dtype=float)
    point[0] = np.cosh(radius)
    point[1:] = np.sinh(radius) * direction
    return point


def h3_tangent_from_point(point: np.ndarray) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    if point.shape != (4,):
        raise ValueError("H3 point must have shape (4,)")
    radius = float(np.arccosh(max(float(point[0]), 1.0)))
    if radius < 1e-15:
        return np.zeros(3, dtype=float)
    sinh_radius = max(float(np.sinh(radius)), 1e-15)
    return point[1:] * (radius / sinh_radius)


def random_h3_points(count: int, *, seed: int = 1, radius: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tangents = rng.normal(size=(int(count), 3))
    tangents *= float(radius) / np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)
    scales = rng.random(int(count))[:, None] ** (1.0 / 3.0)
    return np.vstack([h3_point_from_tangent(row) for row in tangents * scales])


def h3_distance_matrix(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("H3 points must have shape (N, 4)")
    gram = -(-np.outer(points[:, 0], points[:, 0]) + points[:, 1:] @ points[:, 1:].T)
    gram = np.maximum(gram, 1.0)
    distance = np.arccosh(gram)
    np.fill_diagonal(distance, 0.0)
    return distance


def h3_halfspace_profile(points: np.ndarray, normals: np.ndarray, *, softness: float = 0.15) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    signed = -np.outer(points[:, 0], normals[:, 0]) + points[:, 1:] @ normals[:, 1:].T
    width = max(float(softness), 1e-9)
    return 1.0 / (1.0 + np.exp(-np.clip(signed / width, -60.0, 60.0)))


def h3_chart_report(caps: list[RoundCap]) -> dict[str, Any]:
    normals = cap_normals(caps)
    origin = h3_origin()
    origin_norm = float(minkowski_dot(origin, origin))
    normal_norms = minkowski_dot(normals, normals) if normals.size else np.zeros(0, dtype=float)
    max_normal_error = float(np.max(np.abs(normal_norms - 1.0))) if normal_norms.size else 0.0
    receipt = bool(
        len(caps) > 0
        and normals.shape == (len(caps), 4)
        and abs(origin_norm + 1.0) < 1e-12
        and max_normal_error < 1e-10
    )
    return {
        "mode": "canonical_conformal_h3_spatial_chart",
        "homogeneous_space": "SO+(3,1)/SO(3)",
        "model": "H3 hyperboloid {X: eta(X,X)=-1, X0>0}",
        "minkowski_signature": "(-,+,+,+)",
        "spatial_dimension": 3,
        "lorentz_group_dimension": 6,
        "stabilizer_group": "SO(3)",
        "stabilizer_dimension": 3,
        "spatial_dimension_derivation": "dim SO+(3,1)-dim SO(3)=6-3=3",
        "origin_norm": origin_norm,
        "cap_count": len(caps),
        "max_cap_normal_unit_error": max_normal_error,
        "conformal_h3_spatial_chart_receipt": receipt,
        "blockers": [] if receipt else (["missing_caps"] if not caps else ["invalid_cap_normals"]),
        "record_population_receipt": False,
        "claim_boundary": (
            "canonical 3D spatial chart implied by the conformal/Lorentz cap branch; "
            "record and defect population requires a separate populated-bulk gate"
        ),
    }
