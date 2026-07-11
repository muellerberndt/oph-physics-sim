from __future__ import annotations

from dataclasses import dataclass
from math import tan
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree

from oph_fpe.constants.oph_pixel import P_STAR, cap_area_planck, cap_entropy_capacity
from oph_fpe.core.graph import fibonacci_sphere_points


@dataclass(frozen=True)
class RoundCap:
    axis: np.ndarray
    theta0: float
    tangent: np.ndarray
    collar_width: float = 0.03

    def normalized(self) -> "RoundCap":
        axis = _unit(self.axis)
        tangent = np.asarray(self.tangent, dtype=float)
        tangent = tangent - axis * float(np.dot(axis, tangent))
        if np.linalg.norm(tangent) < 1e-12:
            tangent = _fallback_tangent(axis)
        return RoundCap(axis=axis, theta0=float(self.theta0), tangent=_unit(tangent), collar_width=float(self.collar_width))


def cap_weights(points: np.ndarray, cap: RoundCap, soft: bool = True) -> np.ndarray:
    cap = cap.normalized()
    threshold = np.cos(cap.theta0)
    signed = points @ cap.axis - threshold
    if not soft:
        return (signed >= 0.0).astype(float)
    width = max(float(cap.collar_width), 1e-6)
    logits = np.clip(signed / width, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def collar_mask(points: np.ndarray, cap: RoundCap) -> np.ndarray:
    cap = cap.normalized()
    threshold = np.cos(cap.theta0)
    return np.abs(points @ cap.axis - threshold) <= max(cap.collar_width, 1e-6)


def cap_ordered_frame_points(cap: RoundCap) -> tuple[np.ndarray, np.ndarray]:
    """Return the ordered BW boundary frame selected by ``cap.tangent``.

    With the stereographic/Mobius convention used by :func:`lambda_cap`,
    ``p_minus`` is the attracting fixed point for positive ``s`` and is sent
    to zero by the paper's frame map; ``p_plus`` is the repelling point sent to
    infinity.  Making this pair explicit prevents a tangent direction from
    being mistaken for an unordered numerical convenience.
    """

    normalized = cap.normalized()
    boundary_center = np.cos(normalized.theta0) * normalized.axis
    boundary_offset = np.sin(normalized.theta0) * normalized.tangent
    p_minus = _unit(boundary_center + boundary_offset)
    p_plus = _unit(boundary_center - boundary_offset)
    return p_minus, p_plus


def lambda_cap(points: np.ndarray, cap: RoundCap, s: float) -> np.ndarray:
    """Apply the cap-preserving conformal flow lambda_C(s) on S2.

    The implementation uses a cap-local stereographic disk coordinate. The cap
    axis is the local north pole, the cap boundary is normalized to |w|=1, and
    w -> (w+a)/(a*w+1), a=tanh(s/2), preserves the disk and the ordered
    boundary pair returned by :func:`cap_ordered_frame_points`.  For positive
    ``s``, ``p_minus`` is attracting and ``p_plus`` is repelling, matching
    ``h(p_minus)=0``, ``h(p_plus)=infinity``, and ``h -> exp(-s) h``.
    """

    cap = cap.normalized()
    e1 = cap.tangent
    e2 = _unit(np.cross(cap.axis, e1))
    north = cap.axis
    local_x = points @ e1
    local_y = points @ e2
    local_z = np.clip(points @ north, -1.0, 1.0)
    z = (local_x + 1j * local_y) / np.maximum(1.0 + local_z, 1e-15)
    rho = max(tan(cap.theta0 / 2.0), 1e-12)
    w = z / rho
    a = np.tanh(float(s) / 2.0)
    w_next = (w + a) / (a * w + 1.0)
    z_next = rho * w_next
    mag2 = np.square(z_next.real) + np.square(z_next.imag)
    denom = 1.0 + mag2
    u = 2.0 * z_next.real / denom
    v = 2.0 * z_next.imag / denom
    zcoord = (1.0 - mag2) / denom
    mapped = u[:, None] * e1 + v[:, None] * e2 + zcoord[:, None] * north
    return mapped / np.maximum(np.linalg.norm(mapped, axis=1, keepdims=True), 1e-15)


def interpolation_map(
    points: np.ndarray,
    query_points: np.ndarray,
    k: int = 8,
    tree: cKDTree | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    tree = tree or cKDTree(points)
    k = max(1, min(int(k), points.shape[0]))
    distances, indices = tree.query(query_points, k=k)
    if k == 1:
        indices = indices[:, None]
        distances = distances[:, None]
    weights = 1.0 / np.maximum(distances, 1e-12)
    exact = distances <= 1e-12
    if np.any(exact):
        weights = np.where(exact, 1.0, 0.0)
    weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-15)
    return indices.astype(np.int64), weights.astype(float)


def apply_interpolation(values: np.ndarray, indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.sum(values[indices] * weights, axis=1)


def pullback_field(
    points: np.ndarray,
    values: np.ndarray,
    cap: RoundCap,
    s: float,
    k: int = 8,
    tree: cKDTree | None = None,
) -> np.ndarray:
    mapped = lambda_cap(points, cap, s)
    indices, weights = interpolation_map(points, mapped, k=k, tree=tree)
    return apply_interpolation(values, indices, weights)


def sample_caps(
    points: np.ndarray,
    count: int,
    theta_values: Iterable[float],
    seed: int,
    collar_width: float = 0.03,
) -> list[RoundCap]:
    theta_list = [float(value) for value in theta_values]
    if not theta_list:
        theta_list = [0.55, 0.75, 1.0]
    axes = fibonacci_sphere_points(max(1, int(count)))
    rng = np.random.default_rng(seed)
    caps: list[RoundCap] = []
    for index, axis in enumerate(axes):
        tangent = rng.normal(size=3)
        tangent = tangent - axis * float(np.dot(axis, tangent))
        if np.linalg.norm(tangent) < 1e-12:
            tangent = _fallback_tangent(axis)
        caps.append(
            RoundCap(
                axis=_unit(axis),
                theta0=theta_list[index % len(theta_list)],
                tangent=_unit(tangent),
                collar_width=collar_width,
            )
        )
    return caps


def cap_geometry_report(
    points: np.ndarray,
    caps: list[RoundCap],
    *,
    cell_area_planck: float | np.ndarray = P_STAR,
    cell_entropy: float | np.ndarray = P_STAR / 4.0,
) -> dict[str, object]:
    rows = []
    for index, cap in enumerate(caps):
        p_minus, p_plus = cap_ordered_frame_points(cap)
        hard_weights = cap_weights(points, cap, soft=False)
        soft_weights = cap_weights(points, cap, soft=True)
        collar = collar_mask(points, cap)
        rows.append(
            {
                "cap_index": index,
                "theta0": cap.theta0,
                "frame_p_minus": p_minus.tolist(),
                "frame_p_plus": p_plus.tolist(),
                "frame_ordering": "p_minus_attracting_for_positive_s",
                "hard_count": int(np.sum(hard_weights)),
                "hard_fraction": float(np.mean(hard_weights)),
                "collar_count": int(np.sum(collar)),
                "hard_cap_area_planck": cap_area_planck(hard_weights, cell_area_planck),
                "hard_cap_entropy_capacity": cap_entropy_capacity(hard_weights, cell_entropy),
                "soft_cap_area_planck": cap_area_planck(soft_weights, cell_area_planck),
                "soft_cap_entropy_capacity": cap_entropy_capacity(soft_weights, cell_entropy),
            }
        )
    return {
        "cap_count": len(caps),
        "weight_measure": "cell_entropy_capacity",
        "P_usage": "cap area/capacity only; lambda_C geometry is unchanged",
        "caps": rows,
    }


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-15:
        raise ValueError("zero vector cannot be normalized")
    return vector / norm


def _fallback_tangent(axis: np.ndarray) -> np.ndarray:
    candidate = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(candidate, axis))) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0])
    tangent = candidate - axis * float(np.dot(axis, candidate))
    return _unit(tangent)
