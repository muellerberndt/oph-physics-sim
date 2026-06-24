from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import gammaln


ScreenSpectrumModel = Literal["EXACT_GAMMA_CONFORMAL", "FRACTIONAL_LAPLACIAN_ASYMPTOTIC"]


@dataclass(frozen=True)
class ScreenSpectrumParams:
    A_q: float = 1.0
    theta: float = 0.0
    mu: float = 0.0
    model: ScreenSpectrumModel = "FRACTIONAL_LAPLACIAN_ASYMPTOTIC"


def screen_precision_eigenvalue(
    ell: float | np.ndarray,
    theta: float,
    *,
    model: ScreenSpectrumModel = "EXACT_GAMMA_CONFORMAL",
    mu: float = 0.0,
) -> float | np.ndarray:
    """Canonical scalar screen precision eigenvalue.

    ``EXACT_GAMMA_CONFORMAL`` is the theorem-side thin-shell gamma family and
    recovers ``ell*(ell+1)`` at theta=0. ``FRACTIONAL_LAPLACIAN_ASYMPTOTIC`` is
    the older asymptotic model; it is close at high ell but not an exact
    low-multipole theorem.
    """

    ell_arr = np.asarray(ell, dtype=float)
    if model == "EXACT_GAMMA_CONFORMAL":
        values = np.exp(gammaln(ell_arr + 2.0 + 0.5 * float(theta)) - gammaln(ell_arr - 0.5 * float(theta)))
    elif model == "FRACTIONAL_LAPLACIAN_ASYMPTOTIC":
        base = np.maximum(ell_arr * (ell_arr + 1.0) + float(mu) ** 2, 1.0e-300)
        values = base ** (1.0 + 0.5 * float(theta))
    else:
        raise ValueError(f"unknown screen spectrum model: {model}")
    if np.isscalar(ell):
        return float(values)
    return values


def screen_cl(
    ell: float | np.ndarray,
    params: ScreenSpectrumParams | None = None,
    *,
    A_q: float | None = None,
    theta: float | None = None,
    mu: float | None = None,
    model: ScreenSpectrumModel | None = None,
) -> float | np.ndarray:
    """Canonical screen covariance ``C_l^q = A_q / Lambda_l(theta)``."""

    params = params or ScreenSpectrumParams()
    amp = float(params.A_q if A_q is None else A_q)
    tilt = float(params.theta if theta is None else theta)
    mass = float(params.mu if mu is None else mu)
    family = params.model if model is None else model
    values = amp / screen_precision_eigenvalue(ell, tilt, model=family, mu=mass)
    if np.isscalar(ell):
        return float(values)
    return values


def screen_d_ell(ell: float | np.ndarray, c_ell: float | np.ndarray) -> float | np.ndarray:
    ell_arr = np.asarray(ell, dtype=float)
    c_arr = np.asarray(c_ell, dtype=float)
    values = ell_arr * (ell_arr + 1.0) * c_arr / (2.0 * np.pi)
    if np.isscalar(ell) and np.isscalar(c_ell):
        return float(values)
    return values


def red_tilt_slope_check(
    *,
    theta: float,
    ell_min: int = 30,
    ell_max: int = 1000,
    model: ScreenSpectrumModel = "FRACTIONAL_LAPLACIAN_ASYMPTOTIC",
) -> dict[str, float | bool]:
    """Regression check: for theta>0, D_l must have negative log slope."""

    ell = np.arange(int(ell_min), int(ell_max), dtype=float)
    c_ell = screen_cl(ell, A_q=1.0, theta=float(theta), model=model)
    d_ell = screen_d_ell(ell, c_ell)
    slope = float(np.polyfit(np.log(ell), np.log(d_ell), 1)[0])
    return {
        "theta": float(theta),
        "model": model,
        "log_D_ell_slope": slope,
        "red_tilt_sign_receipt": bool(float(theta) <= 0.0 or slope < 0.0),
    }
