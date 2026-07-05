from __future__ import annotations

import numpy as np

from oph_fpe.cosmology.oph_constants import OPHConstants

KM_S_MPC_TO_GYR_INV = 1.022712165045695e-3


def E_of_a(
    a: float | np.ndarray,
    Omega_m: float,
    Omega_r: float,
    Omega_lambda: float,
) -> float | np.ndarray:
    """Dimensionless H(a)/H0 for the background used by the diagnostic kernel."""

    a_arr = np.asarray(a, dtype=float)
    return np.sqrt(Omega_r / a_arr**4 + Omega_m / a_arr**3 + Omega_lambda)


def H_phys_Gyr_inv(
    a: float | np.ndarray,
    H0_km_s_Mpc: float,
    Omega_m: float,
    Omega_r: float,
    Omega_lambda: float,
) -> float | np.ndarray:
    """Physical H(a) in Gyr^-1."""

    return float(H0_km_s_Mpc) * KM_S_MPC_TO_GYR_INV * E_of_a(a, Omega_m, Omega_r, Omega_lambda)


def W_k(k_hMpc: float | np.ndarray, kA_hMpc: float) -> float | np.ndarray:
    """Minimal scalar finite-collar k-window."""

    k = np.asarray(k_hMpc, dtype=float)
    k_a = max(float(kA_hMpc), 1.0e-30)
    return k**2 / (k**2 + k_a**2)


def W_a(
    a: float | np.ndarray,
    tau_rec_Gyr: float,
    q_A: float,
    H0_km_s_Mpc: float,
    Omega_m: float,
    Omega_r: float,
    Omega_lambda: float,
) -> float | np.ndarray:
    """Late repair-activation window for the Boltzmann-lite branch."""

    gamma_rec = 1.0 / max(float(tau_rec_Gyr), 1.0e-30)
    h_phys = H_phys_Gyr_inv(a, H0_km_s_Mpc, Omega_m, Omega_r, Omega_lambda)
    numerator = gamma_rec * float(q_A)
    return numerator / (numerator + h_phys)


def B_A_z6_poisson_five_of_seven(
    k_hMpc: float | np.ndarray,
    a: float | np.ndarray,
    *,
    kA_hMpc: float,
    tau_rec_Gyr: float,
    q_A: float = 1.0,
    H0_km_s_Mpc: float = 67.4,
    Omega_m: float = 0.315905207,
    Omega_r: float = 9.2e-5,
    Omega_lambda: float = 0.6840,
    constants: OPHConstants = OPHConstants(),
) -> float | np.ndarray:
    """OPH z6/Poisson five-of-seven weak-lensing response kernel.

    B_A(k,a) = 1 - (5/7)(1 - lambda_collar) W_k(k) W_a(a).
    The exp(-P/24) value is the exact-uniform/product-thickening diagnostic
    target; finite-thickness runs must promote it through
    UNIFORM_PRODUCT_THICKENING_EXACT before treating it as a physical local
    coefficient.
    """

    return 1.0 - constants.epsilon_A_wl * W_k(k_hMpc, kA_hMpc) * W_a(
        a,
        tau_rec_Gyr,
        q_A,
        H0_km_s_Mpc,
        Omega_m,
        Omega_r,
        Omega_lambda,
    )


def B_A_minimal_one_pole(
    k_hMpc: float | np.ndarray,
    a: float | np.ndarray,
    *,
    kA_hMpc: float,
    tau_rec_Gyr: float,
    q_A: float = 1.0,
    H0_km_s_Mpc: float = 67.4,
    Omega_m: float = 0.315905207,
    Omega_r: float = 9.2e-5,
    Omega_lambda: float = 0.6840,
    constants: OPHConstants = OPHConstants(),
) -> float | np.ndarray:
    """Minimal OPH finite-collar response before survey projection.

    B_A(k,a) = 1 - eta_A W_k(k) W_a(a), with eta_A = 1 - lambda_collar.
    The exact-uniform target uses lambda_collar = exp(-P/24); generic
    finite-thickness lanes should read lambda_collar from a scalar profile
    integral until the exact-value gate closes.
    This is the microphysics-side kernel target. A weak-lensing scalar number
    is obtained only after projecting this window through a survey response.
    """

    return 1.0 - constants.reserve * W_k(k_hMpc, kA_hMpc) * W_a(
        a,
        tau_rec_Gyr,
        q_A,
        H0_km_s_Mpc,
        Omega_m,
        Omega_r,
        Omega_lambda,
    )


def compressed_projection_fraction(
    observed_amplitude: float,
    baseline_amplitude: float,
    constants: OPHConstants = OPHConstants(),
) -> float:
    """Return Pi_L from L_OPH/L_0 = 1 - eta_A Pi_L."""

    eta = constants.reserve
    if eta <= 0.0:
        raise ValueError("eta_A must be positive")
    return float((1.0 - float(observed_amplitude) / float(baseline_amplitude)) / eta)


def projected_amplitude_ratio(pi_L: float, constants: OPHConstants = OPHConstants()) -> float:
    """Return L_OPH/L_0 for a normalized projection fraction Pi_L."""

    return 1.0 - constants.reserve * float(pi_L)


def projected_amplitude(
    baseline_amplitude: float,
    pi_L: float,
    constants: OPHConstants = OPHConstants(),
) -> float:
    """Return L_OPH = L_0[1 - eta_A Pi_L]."""

    return float(baseline_amplitude) * projected_amplitude_ratio(pi_L, constants)


def normalized_projection_average(window: np.ndarray, response_weight: np.ndarray) -> float:
    """Compute Pi_L = <W>_K for a discrete normalized or unnormalized response kernel."""

    w = np.asarray(window, dtype=float)
    k = np.asarray(response_weight, dtype=float)
    if w.shape != k.shape:
        raise ValueError("window and response_weight must have the same shape")
    mass = float(np.sum(k))
    if mass <= 0.0:
        raise ValueError("response_weight has zero mass")
    return float(np.sum(k * w) / mass)


def apply_projected_wl_selector(S8: float, constants: OPHConstants = OPHConstants()) -> float:
    """Compressed-scorecard projection with the exact-uniform collar target."""

    return float(S8) * constants.R_wl
