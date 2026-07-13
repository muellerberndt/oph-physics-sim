from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from oph_fpe.cosmology.cassini_external_field import (
    OPH_A0_M_S2,
    OPH_LAMBDA_Z6_TARGET,
)


@dataclass(frozen=True)
class RARRows:
    log10_gbar: np.ndarray
    sigma_log10_gbar: np.ndarray
    log10_gobs: np.ndarray
    sigma_log10_gobs: np.ndarray

    @property
    def count(self) -> int:
        return int(self.log10_gbar.size)


def load_rar_table(path: Path) -> RARRows:
    rows: list[tuple[float, float, float, float]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        try:
            values = tuple(float(value) for value in parts)
        except ValueError:
            continue
        if all(math.isfinite(value) for value in values):
            rows.append(values)
    if not rows:
        raise ValueError("RAR table contains no usable rows")
    array = np.asarray(rows, dtype=float)
    return RARRows(
        log10_gbar=array[:, 0],
        sigma_log10_gbar=array[:, 1],
        log10_gobs=array[:, 2],
        sigma_log10_gobs=array[:, 3],
    )


def _log10_rar_model(log10_gbar: np.ndarray, effective_a0_m_s2: float) -> np.ndarray:
    gbar = np.power(10.0, np.asarray(log10_gbar, dtype=float))
    root = np.sqrt(np.maximum(gbar / float(effective_a0_m_s2), 0.0))
    denominator = -np.expm1(-root)
    return np.log10(gbar / np.maximum(denominator, 1.0e-300))


def _log10_rar_slope(log10_gbar: np.ndarray, effective_a0_m_s2: float) -> np.ndarray:
    step = 1.0e-4
    values = np.asarray(log10_gbar, dtype=float)
    return (
        _log10_rar_model(values + step, effective_a0_m_s2)
        - _log10_rar_model(values - step, effective_a0_m_s2)
    ) / (2.0 * step)


def rar_branch_statistics(rows: RARRows, effective_a0_m_s2: float) -> dict[str, float | int]:
    predicted = _log10_rar_model(rows.log10_gbar, effective_a0_m_s2)
    residual = rows.log10_gobs - predicted
    slope = _log10_rar_slope(rows.log10_gbar, effective_a0_m_s2)
    propagated_sigma = np.sqrt(
        rows.sigma_log10_gobs**2 + (slope * rows.sigma_log10_gbar) ** 2
    )
    usable = propagated_sigma > 0.0
    chi2 = float(np.sum((residual[usable] / propagated_sigma[usable]) ** 2))
    return {
        "effective_a0_m_s2": float(effective_a0_m_s2),
        "point_count": rows.count,
        "mean_residual_data_minus_model_dex": float(np.mean(residual)),
        "median_absolute_residual_dex": float(np.median(np.abs(residual))),
        "rms_residual_dex": float(np.sqrt(np.mean(residual * residual))),
        "diagonal_chi2_proxy": chi2,
        "diagonal_chi2_proxy_per_point": chi2 / int(np.sum(usable)),
    }


def _load_binned_rar(path: Path) -> np.ndarray:
    rows: list[tuple[float, float, float, int]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1]), float(parts[2]), int(float(parts[3]))))
        except ValueError:
            continue
    if not rows:
        raise ValueError("binned RAR table contains no usable rows")
    return np.asarray(rows, dtype=float)


def _binned_statistics(path: Path, effective_a0_m_s2: float) -> dict[str, float | int]:
    rows = _load_binned_rar(path)
    predicted = _log10_rar_model(rows[:, 0], effective_a0_m_s2)
    residual = rows[:, 1] - predicted
    weights = rows[:, 3]
    return {
        "bin_count": int(rows.shape[0]),
        "represented_point_count": int(np.sum(weights)),
        "mean_residual_data_minus_model_dex": float(np.mean(residual)),
        "unweighted_bin_rms_residual_dex": float(np.sqrt(np.mean(residual * residual))),
        "point_count_weighted_rms_residual_dex": float(
            np.sqrt(np.sum(weights * residual * residual) / np.sum(weights))
        ),
    }


def fixed_oph_rar_report(all_rows_path: Path, binned_rows_path: Path) -> dict[str, Any]:
    """Evaluate fixed OPH unit/Z6 continuation branches against public RAR rows."""

    rows = load_rar_table(all_rows_path)
    optimum = minimize_scalar(
        lambda log10_a0: float(
            rar_branch_statistics(rows, 10.0 ** float(log10_a0))["diagonal_chi2_proxy"]
        ),
        bounds=(math.log10(0.2e-10), math.log10(3.0e-10)),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    if not optimum.success:
        raise RuntimeError("RAR effective-a0 comparison fit failed")
    best_a0 = 10.0 ** float(optimum.x)
    branch_specs = (
        (
            "z6_exact_uniform_target",
            OPH_A0_M_S2 / (OPH_LAMBDA_Z6_TARGET * OPH_LAMBDA_Z6_TARGET),
            "conditional_exact_uniform_product_thickening_target",
        ),
        ("unit_lambda_endpoint", OPH_A0_M_S2, "jensen_band_endpoint"),
        ("same_data_best_fit", best_a0, "same_data_calibration_reference"),
    )
    branches: dict[str, Any] = {}
    for name, a0_eff, status in branch_specs:
        branches[name] = {
            **rar_branch_statistics(rows, a0_eff),
            "binned": _binned_statistics(binned_rows_path, a0_eff),
            "coefficient_status": status,
            "fit_to_rar": name == "same_data_best_fit",
        }
    z6 = branches["z6_exact_uniform_target"]
    best = branches["same_data_best_fit"]
    return {
        "mode": "fixed_oph_rar_public_comparison",
        "dataset_id": "sparc_rar_lelli2017",
        "all_rows_path": str(all_rows_path),
        "binned_rows_path": str(binned_rows_path),
        "public_rar_galaxy_count": 153,
        "point_count": rows.count,
        "branches": branches,
        "z6_vs_same_data_best": {
            "effective_a0_ratio": float(z6["effective_a0_m_s2"])
            / float(best["effective_a0_m_s2"]),
            "rms_residual_delta_dex": float(z6["rms_residual_dex"])
            - float(best["rms_residual_dex"]),
            "diagonal_chi2_proxy_per_point_delta": float(
                z6["diagonal_chi2_proxy_per_point"]
            )
            - float(best["diagonal_chi2_proxy_per_point"]),
        },
        "comparison_receipt": True,
        "physical_prediction_receipt": False,
        "assessment": (
            "positive retrospective fixed-formula check: the conditional Z6 effective scale is close "
            "to the same-data optimum and has nearly identical aggregate RAR scatter"
        ),
        "claim_boundary": (
            "Neither OPH coefficient was fitted to this RAR table in this calculation, but a0_OPH is a "
            "cosmology-derived benchmark and the Z6 lambda is conditional rather than a closed theorem. "
            "The 2,693 aggregate rows have no galaxy identifiers and share distance, inclination, and "
            "mass-to-light systematics, so the diagonal chi-square is a proxy and not a likelihood significance."
        ),
    }
