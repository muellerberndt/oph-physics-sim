from __future__ import annotations

import numpy as np


def cl_oph_screen(
    ell: np.ndarray,
    *,
    A: float = 1.0,
    eta: float = 0.035,
    mu: float = 0.0,
    ell_cap: float = 3000.0,
) -> np.ndarray:
    """Analytic OPH-CET screen covariance used as a CMB fossil diagnostic."""

    ell_values = np.asarray(ell, dtype=float)
    denom = np.maximum(ell_values * (ell_values + 1.0) + float(mu) * float(mu), 1e-30)
    base = float(A) / np.power(denom, 1.0 - float(eta) / 2.0)
    window = np.exp(-ell_values * (ell_values + 1.0) / (float(ell_cap) * float(ell_cap)))
    return base * window * window


def apply_low_l_repair_suppression(
    cl: np.ndarray,
    ell: np.ndarray,
    *,
    q_ir: float = 0.15,
    ell_ir: float = 5.0,
) -> np.ndarray:
    """Apply a low-ell finite-repair suppression factor."""

    ell_values = np.asarray(ell, dtype=float)
    values = np.asarray(cl, dtype=float)
    scale = float(ell_ir) * (float(ell_ir) + 1.0)
    envelope = np.exp(-ell_values * (ell_values + 1.0) / max(scale, 1e-30))
    return values * (1.0 - float(q_ir) * envelope)


def apply_parity_term(
    cl: np.ndarray,
    ell: np.ndarray,
    *,
    eps_p: float = 0.0,
    ell_p: float = 30.0,
) -> np.ndarray:
    """Apply a decaying even/odd parity modulation."""

    ell_values = np.asarray(ell, dtype=float)
    values = np.asarray(cl, dtype=float)
    parity = 1.0 + float(eps_p) * ((-1.0) ** ell_values.astype(int)) * np.exp(
        -ell_values / max(float(ell_p), 1e-30)
    )
    return values * parity
