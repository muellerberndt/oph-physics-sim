from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.optimize import least_squares

from oph_fpe.bulk.cap_geometry import RoundCap
from oph_fpe.bulk.cap_normals import cap_normal, cap_normals, minkowski_dot
from oph_fpe.core.graph import fibonacci_sphere_points


@dataclass(frozen=True)
class H3ProfileFit:
    point: np.ndarray
    spatial: np.ndarray
    mean_squared_error: float
    max_abs_error: float
    nfev: int
    restart_count: int

    def to_json(self) -> dict[str, Any]:
        return {
            "point": [float(value) for value in self.point],
            "spatial": [float(value) for value in self.spatial],
            "mean_squared_error": float(self.mean_squared_error),
            "max_abs_error": float(self.max_abs_error),
            "nfev": int(self.nfev),
            "restart_count": int(self.restart_count),
        }


def h3_point_from_ball(spatial: np.ndarray) -> np.ndarray:
    """Return the H3 hyperboloid point X=(sqrt(1+|x|^2), x)."""

    spatial = np.asarray(spatial, dtype=float)
    if spatial.shape != (3,):
        raise ValueError("spatial coordinate must have shape (3,)")
    point = np.empty(4, dtype=float)
    point[0] = float(np.sqrt(1.0 + np.dot(spatial, spatial)))
    point[1:] = spatial
    return point


def verify_h3_point(point: np.ndarray, *, atol: float = 1.0e-9) -> bool:
    point = np.asarray(point, dtype=float)
    return bool(point.shape == (4,) and point[0] > 0.0 and abs(float(minkowski_dot(point, point)) + 1.0) < atol)


def sample_round_caps(
    axis_count: int,
    theta_values: Iterable[float],
    *,
    seed: int = 1,
    collar_width: float = 0.03,
) -> list[RoundCap]:
    """Sample a product family of S2 cap axes and opening angles."""

    theta_list = [float(value) for value in theta_values]
    if not theta_list:
        theta_list = [0.35, 0.55, 0.75, 1.0, 1.25]
    axes = fibonacci_sphere_points(max(1, int(axis_count)))
    rng = np.random.default_rng(seed)
    caps: list[RoundCap] = []
    for axis in axes:
        tangent = rng.normal(size=3)
        tangent = tangent - axis * float(np.dot(axis, tangent))
        if np.linalg.norm(tangent) < 1.0e-12:
            tangent = _fallback_tangent(axis)
        for theta in theta_list:
            caps.append(
                RoundCap(
                    axis=axis,
                    theta0=float(theta),
                    tangent=tangent,
                    collar_width=float(collar_width),
                ).normalized()
            )
    return caps


def sample_h3_spatial_points(count: int, *, radius: float = 1.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(int(count), 3))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1.0e-12)
    radii = float(radius) * rng.random(int(count))[:, None] ** (1.0 / 3.0)
    return directions * radii


def cap_profile_for_h3_point(point: np.ndarray, normals: np.ndarray, *, softness: float = 0.25) -> np.ndarray:
    """Support-visible cap-response profile for an H3 point.

    The profile is a softened sign of eta(X,n_C). This is the minimal finite
    bridge: bulk points are represented by cap responses, not by boundary
    observer coordinates.
    """

    width = max(float(softness), 1.0e-12)
    signed = minkowski_dot(np.asarray(point, dtype=float), np.asarray(normals, dtype=float))
    return np.tanh(np.clip(signed / width, -60.0, 60.0))


def fit_h3_point_from_cap_profile(
    profile: np.ndarray,
    normals: np.ndarray,
    *,
    softness: float = 0.25,
    radius: float = 2.0,
    restarts: int = 24,
    seed: int = 1,
    max_nfev: int = 200,
) -> H3ProfileFit:
    profile = np.asarray(profile, dtype=float)
    normals = np.asarray(normals, dtype=float)
    if normals.ndim != 2 or normals.shape[1] != 4:
        raise ValueError("normals must have shape (N, 4)")
    if profile.shape != (normals.shape[0],):
        raise ValueError("profile length must match cap-normal count")

    rng = np.random.default_rng(seed)
    starts = [np.zeros(3, dtype=float)]
    starts.extend(sample_h3_spatial_points(max(0, int(restarts) - 1), radius=radius, seed=seed + 17))

    def residual(spatial: np.ndarray) -> np.ndarray:
        point = h3_point_from_ball(spatial)
        return cap_profile_for_h3_point(point, normals, softness=softness) - profile

    best_result = None
    best_error = float("inf")
    for start in starts:
        jittered = np.asarray(start, dtype=float)
        if np.linalg.norm(jittered) < 1.0e-12 and len(starts) > 1:
            jittered = jittered + 1.0e-6 * rng.normal(size=3)
        result = least_squares(
            residual,
            jittered,
            bounds=(-float(radius), float(radius)),
            max_nfev=int(max_nfev),
        )
        error = float(np.mean(np.square(residual(result.x))))
        if error < best_error:
            best_error = error
            best_result = result

    if best_result is None:
        raise RuntimeError("least-squares fit did not run")
    final_residual = residual(best_result.x)
    return H3ProfileFit(
        point=h3_point_from_ball(best_result.x),
        spatial=np.asarray(best_result.x, dtype=float),
        mean_squared_error=float(np.mean(np.square(final_residual))),
        max_abs_error=float(np.max(np.abs(final_residual))) if final_residual.size else 0.0,
        nfev=int(best_result.nfev),
        restart_count=int(len(starts)),
    )


def best_s2_boundary_profile_error(profile: np.ndarray, caps: list[RoundCap]) -> float:
    """Best boundary-point cap-incidence error over sampled cap axes.

    This control asks whether a boundary S2 point can explain the profile as
    well as an H3 cap-profile point.
    """

    profile = np.asarray(profile, dtype=float)
    if not caps:
        return float("inf")
    axes = np.vstack([cap.normalized().axis for cap in caps])
    unique_axes = _unique_rows_rounded(axes)
    errors: list[float] = []
    thresholds = np.asarray([np.cos(cap.normalized().theta0) for cap in caps], dtype=float)
    cap_axes = np.vstack([cap.normalized().axis for cap in caps])
    for axis in unique_axes:
        hard = np.where(cap_axes @ axis >= thresholds, 1.0, -1.0)
        errors.append(float(np.mean(np.square(hard - profile))))
    return float(min(errors)) if errors else float("inf")


def caps_to_h3_minimal_receipt(
    *,
    axis_count: int = 64,
    theta_values: Iterable[float] = (0.35, 0.55, 0.75, 1.0, 1.25),
    object_count: int = 16,
    object_radius: float = 1.0,
    fit_radius: float = 2.0,
    softness: float = 0.25,
    restarts: int = 24,
    seed: int = 1,
    max_median_error: float = 0.02,
) -> dict[str, Any]:
    """Run the minimal theorem-aligned cap-profile -> H3 reconstruction smoke."""

    caps = sample_round_caps(axis_count, theta_values, seed=seed)
    normals = cap_normals(caps)
    normal_norms = minkowski_dot(normals, normals) if normals.size else np.zeros(0, dtype=float)
    max_normal_error = float(np.max(np.abs(normal_norms - 1.0))) if normal_norms.size else 0.0
    planted_spatial = sample_h3_spatial_points(object_count, radius=object_radius, seed=seed + 101)
    planted_points = np.vstack([h3_point_from_ball(row) for row in planted_spatial])

    fit_errors: list[float] = []
    max_errors: list[float] = []
    shuffled_errors: list[float] = []
    s2_errors: list[float] = []
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed + 202)
    for index, point in enumerate(planted_points):
        profile = cap_profile_for_h3_point(point, normals, softness=softness)
        fit = fit_h3_point_from_cap_profile(
            profile,
            normals,
            softness=softness,
            radius=fit_radius,
            restarts=restarts,
            seed=seed + 1000 + index,
        )
        shuffled_profile = profile[rng.permutation(profile.size)]
        shuffled_fit = fit_h3_point_from_cap_profile(
            shuffled_profile,
            normals,
            softness=softness,
            radius=fit_radius,
            restarts=max(6, restarts // 2),
            seed=seed + 2000 + index,
            max_nfev=120,
        )
        s2_error = best_s2_boundary_profile_error(profile, caps)
        fit_errors.append(fit.mean_squared_error)
        max_errors.append(fit.max_abs_error)
        shuffled_errors.append(shuffled_fit.mean_squared_error)
        s2_errors.append(s2_error)
        rows.append(
            {
                "object_index": int(index),
                "planted_point": [float(value) for value in point],
                "fit": fit.to_json(),
                "shuffled_profile_fit_mse": float(shuffled_fit.mean_squared_error),
                "best_s2_boundary_profile_mse": float(s2_error),
            }
        )

    median_error = float(np.median(fit_errors)) if fit_errors else float("inf")
    median_shuffled = float(np.median(shuffled_errors)) if shuffled_errors else float("inf")
    median_s2 = float(np.median(s2_errors)) if s2_errors else float("inf")
    h3_beats_shuffled = bool(median_error < 0.25 * median_shuffled)
    h3_beats_s2_boundary = bool(median_error < 0.25 * median_s2)
    passed = bool(
        max_normal_error < 1.0e-9
        and median_error < float(max_median_error)
        and h3_beats_shuffled
        and h3_beats_s2_boundary
    )
    return {
        "mode": "e0_caps_to_h3_minimal",
        "receipt": "S2_CAP_PROFILE_TO_H3_RECEIPT",
        "S2_CAP_PROFILE_TO_H3_RECEIPT": passed,
        "passed": passed,
        "axis_count": int(axis_count),
        "cap_count": int(len(caps)),
        "object_count": int(object_count),
        "softness": float(softness),
        "object_radius": float(object_radius),
        "fit_radius": float(fit_radius),
        "max_cap_normal_unit_error": max_normal_error,
        "cap_normal_checks_pass": bool(max_normal_error < 1.0e-9),
        "median_reconstruction_mse": median_error,
        "median_max_abs_error": float(np.median(max_errors)) if max_errors else None,
        "median_shuffled_profile_mse": median_shuffled,
        "median_s2_boundary_profile_mse": median_s2,
        "h3_beats_shuffled": h3_beats_shuffled,
        "h3_beats_s2_boundary": h3_beats_s2_boundary,
        "sample_rows": rows[: min(len(rows), 16)],
        "claim_boundary": (
            "Minimal theorem-aligned geometry smoke: H3 bulk points are reconstructed from "
            "S2 cap-response profiles. This is not full OPH repair dynamics, not particles, "
            "not CMB, and not a populated observer-object bulk receipt."
        ),
    }


def _fallback_tangent(axis: np.ndarray) -> np.ndarray:
    candidate = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(candidate, axis))) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0], dtype=float)
    tangent = candidate - axis * float(np.dot(axis, candidate))
    return tangent / max(float(np.linalg.norm(tangent)), 1.0e-12)


def _unique_rows_rounded(values: np.ndarray, decimals: int = 12) -> np.ndarray:
    rounded = np.round(np.asarray(values, dtype=float), decimals=decimals)
    unique = np.unique(rounded, axis=0)
    return unique / np.maximum(np.linalg.norm(unique, axis=1, keepdims=True), 1.0e-12)
