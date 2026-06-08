from __future__ import annotations

from collections.abc import Callable
from typing import Any


ScalarFn = Callable[..., float] | float


def anomaly_background_rhs(
    a: float,
    rho_A: float,
    *,
    H: ScalarFn,
    Gamma: ScalarFn,
    rho_A_eq: ScalarFn,
) -> float:
    """Newtonian/conformal-time background RHS for the OPH anomaly scaffold.

    rho_A' + 3 H rho_A = -a Gamma (rho_A - rho_A_eq)

    This is a standalone equation audit helper. Nonzero Gamma is diagnostic
    until a compensating exchange sector closes total energy-momentum.
    """

    a_value = float(a)
    rho_value = float(rho_A)
    h_value = _eval(H, a_value)
    gamma_value = _eval(Gamma, a_value)
    eq_value = _eval(rho_A_eq, a_value)
    return float(-3.0 * h_value * rho_value - a_value * gamma_value * (rho_value - eq_value))


def anomaly_perturbation_rhs(
    eta: float,
    y: tuple[float, float],
    *,
    k: float,
    a: ScalarFn,
    Hconf: ScalarFn,
    Phi: ScalarFn,
    Phi_prime: ScalarFn,
    Psi: ScalarFn,
    delta_b: ScalarFn,
    rho_A: ScalarFn,
    rho_A_eq: ScalarFn,
    Gamma: Callable[[float, float], float] | float,
    B_A: Callable[[float, float], float] | float,
) -> tuple[float, float]:
    """Conformal Newtonian gauge perturbation RHS for the anomaly scaffold.

    delta_A' = -theta_A + 3 Phi' - a Gamma q_A (delta_A - B_A delta_b)
    theta_A' = -H theta_A + k^2 Psi

    This is not a physical prediction until Gamma/B_A/rho_A/rho_A_eq are supplied
    by a non-fit OPH parent and gauge/energy-exchange receipts pass.
    """

    delta_A, theta_A = (float(y[0]), float(y[1]))
    eta_value = float(eta)
    k_value = float(k)
    a_value = _eval(a, eta_value, k_value)
    rho_value = max(_eval(rho_A, eta_value, k_value), 1.0e-300)
    rho_eq_value = _eval(rho_A_eq, eta_value, k_value)
    gamma_value = _eval_k_a(Gamma, k_value, a_value)
    b_value = _eval_k_a(B_A, k_value, a_value)
    q_A = rho_eq_value / rho_value
    delta_prime = (
        -theta_A
        + 3.0 * _eval(Phi_prime, eta_value, k_value)
        - a_value
        * gamma_value
        * q_A
        * (delta_A - b_value * _eval(delta_b, eta_value, k_value))
    )
    theta_prime = -_eval(Hconf, eta_value, k_value) * theta_A + k_value * k_value * _eval(Psi, eta_value, k_value)
    return float(delta_prime), float(theta_prime)


def exchange_closure_report(*, gamma_enabled: bool, compensating_sector_declared: bool) -> dict[str, Any]:
    closed = (not bool(gamma_enabled)) or bool(compensating_sector_declared)
    return {
        "gamma_enabled": bool(gamma_enabled),
        "compensating_sector_declared": bool(compensating_sector_declared),
        "energy_momentum_exchange_closed": bool(closed),
        "physical_exchange_ready": bool(closed and bool(compensating_sector_declared)),
        "claim_boundary": (
            "Gamma_rec=0 is conservation-safe for the CDM-limit branch. Nonzero Gamma_rec "
            "requires a declared compensating sector before physical CMB use."
        ),
    }


def _eval(value: ScalarFn, *args: float) -> float:
    if callable(value):
        return float(value(*args))
    return float(value)


def _eval_k_a(value: Callable[[float, float], float] | float, k: float, a: float) -> float:
    if callable(value):
        return float(value(float(k), float(a)))
    return float(value)
