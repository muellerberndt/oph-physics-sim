from __future__ import annotations

from typing import Any

import numpy as np


def shape_only_spectrum_proxy(
    sim_spectrum: list[dict[str, float]],
    target_spectrum: list[dict[str, float]],
    *,
    sample_count: int = 128,
) -> dict[str, Any]:
    sim = [(float(row["ell"]), float(row.get("D_ell", 0.0))) for row in sim_spectrum if float(row["ell"]) >= 2]
    target = [
        (float(row["ell"]), float(row.get("D_ell", 0.0)))
        for row in target_spectrum
        if float(row.get("D_ell", 0.0)) > 0.0
    ]
    if len(sim) < 2 or len(target) < 2:
        return {"usable": False, "reason": "not_enough_spectrum_points"}
    grid = np.linspace(0.0, 1.0, int(sample_count))
    sim_curve = _standardize(np.interp(grid, *_normalize_curve(sim)))
    target_curve = _standardize(np.interp(grid, *_normalize_curve(target)))
    amplitude = float(np.dot(sim_curve, target_curve) / max(float(np.dot(sim_curve, sim_curve)), 1e-12))
    residual = amplitude * sim_curve - target_curve
    return {
        "usable": True,
        "shape_correlation": _corr(sim_curve, target_curve),
        "normalized_rmse": float(np.sqrt(np.mean(residual * residual))),
        "best_fit_amplitude": amplitude,
        "sample_count": int(sample_count),
    }


def control_separation_score(field_report: dict[str, Any]) -> float:
    comparison = field_report.get("control_comparison", {})
    value = comparison.get("min_relative_l2_delta")
    if value is None:
        return 0.0
    return float(value)


def _normalize_curve(rows: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([row[0] for row in rows], dtype=float)
    y = np.asarray([row[1] for row in rows], dtype=float)
    span = max(float(np.max(x) - np.min(x)), 1e-12)
    return (x - float(np.min(x))) / span, y


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values - float(np.mean(values))
    scale = float(np.std(values))
    if scale < 1e-12:
        return np.zeros_like(values)
    return values / scale


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)
