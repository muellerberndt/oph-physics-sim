from __future__ import annotations

import numpy as np


def screen_cl_to_primordial_modulation(
    k: np.ndarray,
    *,
    D_star_mpc: float,
    ell_grid: np.ndarray,
    cl_oph: np.ndarray,
    cl_base: np.ndarray,
) -> np.ndarray:
    """Map an OPH screen C_l ratio to a primordial-power modulation F(k)."""

    k_values = np.asarray(k, dtype=float)
    ell_values = np.asarray(ell_grid, dtype=float)
    ratio = np.asarray(cl_oph, dtype=float) / np.maximum(np.asarray(cl_base, dtype=float), 1e-30)
    ell_of_k = np.maximum(k_values * float(D_star_mpc), 2.0)
    return np.interp(ell_of_k, ell_values, ratio, left=float(ratio[0]), right=1.0)


def primordial_power(
    k: np.ndarray,
    *,
    A_s: float,
    n_s: float,
    k0: float,
    F_oph: np.ndarray,
) -> np.ndarray:
    """Apply an OPH fossil modulation to a standard primordial power law."""

    k_values = np.asarray(k, dtype=float)
    modulation = np.asarray(F_oph, dtype=float)
    return float(A_s) * np.power(k_values / float(k0), float(n_s) - 1.0) * modulation
