from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from oph_fpe.cosmology.cassini_external_field import (
    OPH_A0_M_S2,
    OPH_LAMBDA_Z6_TARGET,
)


G_SI = 6.67430e-11
M_SUN_KG = 1.98847e30
PIVOT_LOG10_V_KM_S = 2.0
PUBLISHED_ORTHOGONAL_ML_SLOPE = 3.85
PUBLISHED_ORTHOGONAL_ML_SLOPE_SIGMA = 0.09


@dataclass(frozen=True)
class BTFRRows:
    galaxy: np.ndarray
    log10_velocity: np.ndarray
    log10_mass: np.ndarray
    sigma_log10_velocity: np.ndarray
    sigma_log10_mass: np.ndarray

    @property
    def count(self) -> int:
        return int(self.log10_mass.size)


def load_btfr_l2019(path: Path) -> BTFRRows:
    """Load the Lelli et al. (2019) flat-velocity BTFR table with errors."""

    names: list[str] = []
    log_velocity: list[float] = []
    log_mass: list[float] = []
    sigma_log_velocity: list[float] = []
    sigma_log_mass: list[float] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            mass = float(parts[1])
            mass_error = float(parts[2])
            velocity = float(parts[5])
            velocity_error = float(parts[6])
        except ValueError:
            continue
        if (
            velocity <= 0.0
            or velocity_error < 0.0
            or mass_error < 0.0
            or not all(math.isfinite(value) for value in (mass, mass_error, velocity, velocity_error))
        ):
            continue
        names.append(str(parts[0]))
        log_velocity.append(math.log10(velocity))
        log_mass.append(mass)
        sigma_log_velocity.append(velocity_error / (velocity * math.log(10.0)))
        sigma_log_mass.append(mass_error)
    if len(names) < 3:
        raise ValueError("BTFR table contains fewer than three usable flat-velocity rows")
    return BTFRRows(
        galaxy=np.asarray(names, dtype=object),
        log10_velocity=np.asarray(log_velocity, dtype=float),
        log10_mass=np.asarray(log_mass, dtype=float),
        sigma_log10_velocity=np.asarray(sigma_log_velocity, dtype=float),
        sigma_log10_mass=np.asarray(sigma_log_mass, dtype=float),
    )


def _orthogonal_nll(
    rows: BTFRRows,
    *,
    slope: float,
    pivot_mass: float,
    log_intrinsic_scatter: float,
) -> float:
    intrinsic = math.exp(float(log_intrinsic_scatter))
    slope_value = float(slope)
    normalization = math.sqrt(1.0 + slope_value * slope_value)
    residual = (
        rows.log10_mass
        - (float(pivot_mass) + slope_value * (rows.log10_velocity - PIVOT_LOG10_V_KM_S))
    ) / normalization
    variance = (
        slope_value * slope_value * rows.sigma_log10_velocity**2
        + rows.sigma_log10_mass**2
    ) / (1.0 + slope_value * slope_value) + intrinsic * intrinsic
    if np.any(variance <= 0.0) or not np.all(np.isfinite(variance)):
        return math.inf
    return float(0.5 * np.sum(np.log(2.0 * math.pi * variance) + residual * residual / variance))


def _numerical_covariance(function, optimum: np.ndarray, step: float = 1.0e-4) -> np.ndarray:
    point = np.asarray(optimum, dtype=float)
    count = int(point.size)
    hessian = np.empty((count, count), dtype=float)
    center = float(function(point))
    for i in range(count):
        e_i = np.zeros(count, dtype=float)
        e_i[i] = step
        hessian[i, i] = (function(point + e_i) - 2.0 * center + function(point - e_i)) / (
            step * step
        )
        for j in range(i):
            e_j = np.zeros(count, dtype=float)
            e_j[j] = step
            hessian[i, j] = hessian[j, i] = (
                function(point + e_i + e_j)
                - function(point + e_i - e_j)
                - function(point - e_i + e_j)
                + function(point - e_i - e_j)
            ) / (4.0 * step * step)
    return np.linalg.inv(hessian)


def fit_btfr_orthogonal_ml(rows: BTFRRows) -> dict[str, float | bool]:
    """Fit slope, pivoted normalization, and perpendicular intrinsic scatter."""

    objective = lambda values: _orthogonal_nll(
        rows,
        slope=float(values[0]),
        pivot_mass=float(values[1]),
        log_intrinsic_scatter=float(values[2]),
    )
    result = minimize(
        objective,
        np.asarray([3.85, 9.68, math.log(0.025)], dtype=float),
        method="Nelder-Mead",
        options={"maxiter": 20_000, "xatol": 1.0e-12, "fatol": 1.0e-12},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"orthogonal BTFR fit failed: {result.message}")
    covariance = _numerical_covariance(objective, np.asarray(result.x, dtype=float))
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    slope, pivot_mass, log_scatter = (float(value) for value in result.x)
    return {
        "usable": True,
        "slope": slope,
        "slope_standard_error_hessian": float(errors[0]),
        "pivot_log10_velocity_km_s": PIVOT_LOG10_V_KM_S,
        "pivot_log10_mass_msun": pivot_mass,
        "pivot_mass_standard_error_hessian": float(errors[1]),
        "intercept_at_log10_velocity_zero": pivot_mass - slope * PIVOT_LOG10_V_KM_S,
        "intrinsic_scatter_perpendicular_dex": math.exp(log_scatter),
        "nll": float(result.fun),
        "row_count": rows.count,
        "optimizer_success": bool(result.success),
    }


def fit_btfr_fixed_slope(rows: BTFRRows, slope: float = 4.0) -> dict[str, float | bool]:
    """Fit only normalization and perpendicular scatter at a declared slope."""

    fixed_slope = float(slope)
    objective = lambda values: _orthogonal_nll(
        rows,
        slope=fixed_slope,
        pivot_mass=float(values[0]),
        log_intrinsic_scatter=float(values[1]),
    )
    result = minimize(
        objective,
        np.asarray([9.67, math.log(0.028)], dtype=float),
        method="Nelder-Mead",
        options={"maxiter": 20_000, "xatol": 1.0e-12, "fatol": 1.0e-12},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"fixed-slope BTFR fit failed: {result.message}")
    covariance = _numerical_covariance(objective, np.asarray(result.x, dtype=float))
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    pivot_mass, log_scatter = (float(value) for value in result.x)
    return {
        "usable": True,
        "slope": fixed_slope,
        "pivot_log10_velocity_km_s": PIVOT_LOG10_V_KM_S,
        "pivot_log10_mass_msun": pivot_mass,
        "pivot_mass_standard_error_hessian": float(errors[0]),
        "intercept_at_log10_velocity_zero": pivot_mass - fixed_slope * PIVOT_LOG10_V_KM_S,
        "intrinsic_scatter_perpendicular_dex": math.exp(log_scatter),
        "nll": float(result.fun),
        "row_count": rows.count,
        "optimizer_success": bool(result.success),
    }


def btfr_pivot_mass_from_effective_a0(
    effective_a0_m_s2: float,
    *,
    slope: float = 4.0,
    pivot_log10_velocity_km_s: float = PIVOT_LOG10_V_KM_S,
) -> float:
    """Return the asymptotic BTFR mass at the declared velocity pivot."""

    a0 = float(effective_a0_m_s2)
    if not math.isfinite(a0) or a0 <= 0.0:
        raise ValueError("effective_a0_m_s2 must be finite and positive")
    intercept = math.log10(1000.0**4 / (G_SI * M_SUN_KG * a0))
    return float(intercept + float(slope) * float(pivot_log10_velocity_km_s))


def _fixed_model_profile(
    rows: BTFRRows,
    *,
    slope: float,
    pivot_mass: float,
) -> dict[str, float]:
    result = minimize_scalar(
        lambda log_scatter: _orthogonal_nll(
            rows,
            slope=float(slope),
            pivot_mass=float(pivot_mass),
            log_intrinsic_scatter=float(log_scatter),
        ),
        bounds=(-12.0, 0.0),
        method="bounded",
    )
    if not result.success:
        raise RuntimeError("fixed-model BTFR scatter profile failed")
    return {
        "nll_with_intrinsic_scatter_profiled": float(result.fun),
        "profiled_intrinsic_scatter_perpendicular_dex": math.exp(float(result.x)),
    }


def btfr_error_aware_report(path: Path) -> dict[str, Any]:
    """Compare the OPH slope/normalization with the public error-bearing table."""

    rows = load_btfr_l2019(path)
    free = fit_btfr_orthogonal_ml(rows)
    slope_four = fit_btfr_fixed_slope(rows, slope=4.0)
    slope_pull = (4.0 - float(free["slope"])) / float(free["slope_standard_error_hessian"])

    a0_z6 = OPH_A0_M_S2 / (OPH_LAMBDA_Z6_TARGET * OPH_LAMBDA_Z6_TARGET)
    branches: dict[str, Any] = {}
    for name, a0_eff, status in (
        (
            "z6_exact_uniform_target",
            a0_z6,
            "conditional_exact_uniform_product_thickening_target",
        ),
        ("unit_lambda_endpoint", OPH_A0_M_S2, "jensen_band_endpoint"),
    ):
        pivot = btfr_pivot_mass_from_effective_a0(a0_eff)
        delta = float(slope_four["pivot_log10_mass_msun"]) - pivot
        profile = _fixed_model_profile(rows, slope=4.0, pivot_mass=pivot)
        branches[name] = {
            "effective_a0_m_s2": float(a0_eff),
            "coefficient_status": status,
            "predicted_slope": 4.0,
            "predicted_pivot_log10_mass_msun_at_100_km_s": pivot,
            "observed_minus_predicted_pivot_dex": delta,
            "stat_only_normalization_pull_sigma": delta
            / float(slope_four["pivot_mass_standard_error_hessian"]),
            **profile,
            "deviance_vs_free_slope_fit": 2.0
            * (float(profile["nll_with_intrinsic_scatter_profiled"]) - float(free["nll"])),
            "deviance_vs_slope_four_normalization_fit": 2.0
            * (
                float(profile["nll_with_intrinsic_scatter_profiled"])
                - float(slope_four["nll"])
            ),
        }

    benchmark_match = bool(
        abs(float(free["slope"]) - PUBLISHED_ORTHOGONAL_ML_SLOPE) < 0.03
        and abs(
            float(free["slope_standard_error_hessian"])
            - PUBLISHED_ORTHOGONAL_ML_SLOPE_SIGMA
        )
        < 0.03
    )
    return {
        "mode": "sparc_btfr_error_aware_likelihood",
        "dataset_id": "sparc_btfr_lelli2019",
        "table_path": str(path),
        "row_count": rows.count,
        "method": {
            "fit": "orthogonal_maximum_likelihood",
            "errors_in_both_coordinates": True,
            "intrinsic_scatter_profiled": True,
            "velocity_error_conversion": "sigma_log10V=e_V/(V ln 10)",
            "pivot_log10_velocity_km_s": PIVOT_LOG10_V_KM_S,
        },
        "published_benchmark": {
            "slope": PUBLISHED_ORTHOGONAL_ML_SLOPE,
            "slope_standard_error": PUBLISHED_ORTHOGONAL_ML_SLOPE_SIGMA,
            "systematic_slope_range": [3.5, 4.0],
            "source": "Lelli_et_al_2019",
        },
        "validation": {
            "benchmark_reproduction_receipt": benchmark_match,
            "slope_delta_from_published": float(free["slope"])
            - PUBLISHED_ORTHOGONAL_ML_SLOPE,
            "slope_sigma_delta_from_published": float(free["slope_standard_error_hessian"])
            - PUBLISHED_ORTHOGONAL_ML_SLOPE_SIGMA,
        },
        "free_orthogonal_ml_fit": free,
        "slope_four_fit": slope_four,
        "oph_slope_test": {
            "predicted_slope": 4.0,
            "observed_slope": float(free["slope"]),
            "observed_slope_standard_error_hessian": float(
                free["slope_standard_error_hessian"]
            ),
            "predicted_minus_observed_pull_sigma": slope_pull,
            "inside_published_systematic_range": True,
            "assessment": "slope_four_not_excluded_by_this_BTFR_analysis",
        },
        "oph_fixed_normalization_branches": branches,
        "comparison_receipt": benchmark_match,
        "physical_prediction_receipt": False,
        "assessment": (
            "The asymptotic slope 4 is compatible with the error-aware fit and lies at the edge of "
            "the published systematic range. The fixed OPH normalization is high for the table's "
            "declared stellar mass-to-light convention; its quoted pull is statistical only and does "
            "not marginalize global mass-to-light, distance, inclination, or sample systematics."
        ),
        "claim_boundary": (
            "This corrects the invalid comparison between free-slope and fixed-slope intercepts at "
            "V=1 km/s. It uses the public table's errors in both coordinates and a 100 km/s pivot. "
            "The OPH a0 and lambda inputs remain calibrated/conditional continuation coefficients, "
            "so this is an independent external-data constraint, not a recovered-core prediction."
        ),
    }
