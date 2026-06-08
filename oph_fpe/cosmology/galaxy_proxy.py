from __future__ import annotations

from typing import Any

import numpy as np

from oph_fpe.claims import PROXY, STATIC_GALAXY_LAW_RECEIPT, with_claim_metadata


DEFAULT_A0_OPH = 1.2e-10


def nu_oph(x: np.ndarray | float, lambda_collar: float) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    lam = float(lambda_collar)
    root = np.sqrt(np.maximum(values, 0.0))
    denominator = 1.0 - np.exp(-lam * root)
    return 1.0 / np.maximum(denominator, 1e-15)


def rar_curve(
    g_baryon: np.ndarray | float,
    *,
    a0_oph: float = DEFAULT_A0_OPH,
    lambda_collar: float = 1.0,
) -> np.ndarray:
    gb = np.asarray(g_baryon, dtype=float)
    x = gb / max(float(a0_oph), 1e-30)
    return nu_oph(x, lambda_collar) * gb


def effective_a0(a0_oph: float = DEFAULT_A0_OPH, lambda_collar: float = 1.0) -> float:
    lam = max(float(lambda_collar), 1e-15)
    return float(a0_oph) / (lam * lam)


def fit_lambda_collar(
    g_baryon: np.ndarray,
    g_observed: np.ndarray,
    *,
    a0_oph: float = DEFAULT_A0_OPH,
    grid: np.ndarray | None = None,
) -> dict[str, Any]:
    gb = np.asarray(g_baryon, dtype=float)
    go = np.asarray(g_observed, dtype=float)
    mask = (gb > 0.0) & (go > 0.0) & np.isfinite(gb) & np.isfinite(go)
    if int(np.sum(mask)) < 3:
        return {"usable": False, "reason": "not_enough_positive_acceleration_points"}
    gb = gb[mask]
    go = go[mask]
    candidates = np.asarray(grid if grid is not None else np.linspace(0.2, 4.0, 256), dtype=float)
    residuals = []
    log_observed = np.log(go)
    for lam in candidates:
        predicted = rar_curve(gb, a0_oph=a0_oph, lambda_collar=float(lam))
        residuals.append(float(np.mean((np.log(np.maximum(predicted, 1e-300)) - log_observed) ** 2)))
    best_index = int(np.argmin(residuals))
    return {
        "usable": True,
        "lambda_collar": float(candidates[best_index]),
        "log_mse": float(residuals[best_index]),
        "candidate_count": int(candidates.size),
    }


def btfr_summary(baryonic_mass: np.ndarray, flat_velocity: np.ndarray) -> dict[str, Any]:
    mass = np.asarray(baryonic_mass, dtype=float)
    velocity = np.asarray(flat_velocity, dtype=float)
    mask = (mass > 0.0) & (velocity > 0.0) & np.isfinite(mass) & np.isfinite(velocity)
    if int(np.sum(mask)) < 3:
        return {"usable": False, "reason": "not_enough_positive_mass_velocity_points"}
    log_m = np.log10(mass[mask])
    log_v = np.log10(velocity[mask])
    slope, intercept = np.polyfit(log_v, log_m, deg=1)
    residual = log_m - (slope * log_v + intercept)
    return {
        "usable": True,
        "slope_logM_vs_logV": float(slope),
        "intercept_logM_vs_logV": float(intercept),
        "rms_dex": float(np.sqrt(np.mean(residual * residual))),
        "galaxy_count": int(np.sum(mask)),
    }


def disk_potential_residual(observed_velocity: np.ndarray, proxy_velocity: np.ndarray) -> dict[str, Any]:
    observed = np.asarray(observed_velocity, dtype=float)
    proxy = np.asarray(proxy_velocity, dtype=float)
    mask = (observed > 0.0) & np.isfinite(observed) & np.isfinite(proxy)
    if int(np.sum(mask)) == 0:
        return {"usable": False, "reason": "not_enough_velocity_points"}
    rel = (proxy[mask] - observed[mask]) / np.maximum(observed[mask], 1e-12)
    return {
        "usable": True,
        "mean_fractional_residual": float(np.mean(rel)),
        "rms_fractional_residual": float(np.sqrt(np.mean(rel * rel))),
        "point_count": int(np.sum(mask)),
    }


def galaxy_proxy_receipt(
    *,
    g_baryon: np.ndarray | None = None,
    g_observed: np.ndarray | None = None,
    baryonic_mass: np.ndarray | None = None,
    flat_velocity: np.ndarray | None = None,
    observed_velocity: np.ndarray | None = None,
    proxy_velocity: np.ndarray | None = None,
    a0_oph: float = DEFAULT_A0_OPH,
    lambda_collar: float = 1.0,
) -> dict[str, Any]:
    if g_baryon is None:
        gb = np.logspace(-13, -8, 64)
    else:
        gb = np.asarray(g_baryon, dtype=float)
    rar = rar_curve(gb, a0_oph=a0_oph, lambda_collar=lambda_collar)
    lambda_fit = (
        fit_lambda_collar(gb, np.asarray(g_observed, dtype=float), a0_oph=a0_oph)
        if g_observed is not None
        else {"usable": False, "reason": "observed_rar_not_provided"}
    )
    btfr = (
        btfr_summary(np.asarray(baryonic_mass, dtype=float), np.asarray(flat_velocity, dtype=float))
        if baryonic_mass is not None and flat_velocity is not None
        else {"usable": False, "reason": "btfr_data_not_provided"}
    )
    disk = (
        disk_potential_residual(np.asarray(observed_velocity, dtype=float), np.asarray(proxy_velocity, dtype=float))
        if observed_velocity is not None and proxy_velocity is not None
        else {"usable": False, "reason": "disk_velocity_data_not_provided"}
    )
    report = {
        "mode": "oph_galaxy_rar_btfr_proxy",
        "GALAXY_PROXY_RECEIPT": True,
        "receipt": True,
        "a0_oph": float(a0_oph),
        "lambda_collar_declared": float(lambda_collar),
        "a0_eff": effective_a0(a0_oph, lambda_collar),
        "rar_curve": [
            {"g_baryon": float(left), "g_observed_proxy": float(right)}
            for left, right in zip(gb, rar, strict=False)
        ],
        "lambda_collar_estimate": lambda_fit,
        "btfr": btfr,
        "disk_potential_residual": disk,
        "physical_claim": False,
        "claim_boundary": (
            "measurement-facing galaxy proxy for OPH RAR/BTFR continuation surfaces. It is not a "
            "SPARC fit unless external SPARC rows are supplied, not a populated-bulk proof, and not "
            "a replacement for a baryonic mass-model likelihood"
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=PROXY,
        receipt=STATIC_GALAXY_LAW_RECEIPT,
        physical_claim=False,
        observable_id="oph_static_galaxy_proxy",
        fit_objective="rar_btfr_formula_bookkeeping",
    )
