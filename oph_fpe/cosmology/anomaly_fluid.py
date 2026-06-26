from __future__ import annotations

from collections.abc import Callable, Sequence
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
    delta_Gamma: ScalarFn = 0.0,
    branch: str = "GENERAL_PARALLEL_EXCHANGE",
    tracking_tolerance: float = 1.0e-9,
) -> tuple[float, float]:
    """Conformal Newtonian gauge perturbation RHS for the anomaly scaffold.

    GENERAL_PARALLEL_EXCHANGE:
    delta_A' = -theta_A + 3 Phi' - a Gamma [q_A (delta_A - B_A delta_b)
      + (1 - q_A)(Psi + delta_Gamma)]

    TRACKING_BACKGROUND_PARALLEL_EXCHANGE:
    the same equation with an enforced q_A = 1 background.

    theta_A' = -H theta_A + k^2 Psi

    This is not a physical prediction until Gamma/B_A/rho_A/rho_A_eq are supplied
    by a non-fit OPH parent and gauge/energy-exchange receipts pass.
    """

    delta_A, theta_A = (float(y[0]), float(y[1]))
    eta_value = float(eta)
    k_value = float(k)
    a_value = _eval(a, eta_value, k_value)
    rho_value = _eval(rho_A, eta_value, k_value)
    if rho_value <= 0.0:
        raise ValueError("rho_A must be positive for the physical anomaly perturbation branch")
    rho_eq_value = _eval(rho_A_eq, eta_value, k_value)
    gamma_value = _eval_k_a(Gamma, k_value, a_value)
    b_value = _eval_k_a(B_A, k_value, a_value)
    q_A = rho_eq_value / rho_value
    branch_name = str(branch).upper()
    if branch_name in {"TRACKING_BACKGROUND_PARALLEL_EXCHANGE", "TRACKING_BACKGROUND_PRESSURELESS_PARALLEL_EXCHANGE"}:
        if abs(1.0 - q_A) > float(tracking_tolerance):
            raise ValueError("tracking background branch requires rho_A_eq / rho_A close to one")
    elif branch_name != "GENERAL_PARALLEL_EXCHANGE":
        raise ValueError(f"unknown anomaly perturbation branch: {branch}")
    delta_gamma_value = _eval(delta_Gamma, eta_value, k_value)
    psi_value = _eval(Psi, eta_value, k_value)
    exchange_term = q_A * (delta_A - b_value * _eval(delta_b, eta_value, k_value))
    if branch_name == "GENERAL_PARALLEL_EXCHANGE":
        exchange_term += (1.0 - q_A) * (psi_value + delta_gamma_value)
    delta_prime = (
        -theta_A
        + 3.0 * _eval(Phi_prime, eta_value, k_value)
        - a_value * gamma_value * exchange_term
    )
    theta_prime = -_eval(Hconf, eta_value, k_value) * theta_A + k_value * k_value * psi_value
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


def covariant_exchange_current(
    *,
    scalar_rate: float,
    u_mu: Sequence[float],
    force_mu: Sequence[float] | None = None,
    metric: Sequence[Sequence[float]] | None = None,
    tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    """Construct the local-frame exchange current Q^mu = Q u^mu + F^mu.

    The report is deliberately small: it verifies unit timelike normalization
    for the supplied four-velocity and orthogonality of any force-like piece.
    """

    g = _metric(metric)
    u = _vector4(u_mu, name="u_mu")
    force = _vector4(force_mu if force_mu is not None else (0.0, 0.0, 0.0, 0.0), name="force_mu")
    rate = float(scalar_rate)
    tol = float(tolerance)
    u_norm_residual = abs(_quadratic(g, u) + 1.0)
    force_orthogonality_residual = abs(_bilinear(g, u, force))
    current = [rate * value + force[index] for index, value in enumerate(u)]
    receipt = bool(
        _finite(rate)
        and all(_finite(value) for value in current)
        and u_norm_residual <= tol
        and force_orthogonality_residual <= tol
    )
    blockers: list[str] = []
    if not _finite(rate):
        blockers.append("exchange_scalar_rate_not_finite")
    if u_norm_residual > tol:
        blockers.append("exchange_four_velocity_not_unit_timelike")
    if force_orthogonality_residual > tol:
        blockers.append("exchange_force_not_spatial_in_local_frame")
    return {
        "EXCHANGE_CURRENT_RECEIPT": receipt,
        "scalar_rate": rate,
        "u_mu": u,
        "force_mu": force,
        "Q_mu": current,
        "u_norm_residual": u_norm_residual,
        "force_orthogonality_residual": force_orthogonality_residual,
        "blockers": blockers,
        "claim_boundary": (
            "Local covariant exchange-current receipt. A nonzero anomaly exchange is physical only "
            "when paired with a recipient current and total exchange closes."
        ),
    }


def covariant_exchange_closure_report(
    *,
    anomaly_scalar_rate: float,
    recipient_scalar_rate: float,
    u_mu: Sequence[float],
    anomaly_force_mu: Sequence[float] | None = None,
    recipient_force_mu: Sequence[float] | None = None,
    metric: Sequence[Sequence[float]] | None = None,
    tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    anomaly = covariant_exchange_current(
        scalar_rate=anomaly_scalar_rate,
        u_mu=u_mu,
        force_mu=anomaly_force_mu,
        metric=metric,
        tolerance=tolerance,
    )
    recipient = covariant_exchange_current(
        scalar_rate=recipient_scalar_rate,
        u_mu=u_mu,
        force_mu=recipient_force_mu,
        metric=metric,
        tolerance=tolerance,
    )
    total = [
        float(anomaly["Q_mu"][index]) + float(recipient["Q_mu"][index])
        for index in range(4)
    ]
    residual = max(abs(value) for value in total)
    closed = bool(
        anomaly["EXCHANGE_CURRENT_RECEIPT"]
        and recipient["EXCHANGE_CURRENT_RECEIPT"]
        and residual <= float(tolerance)
    )
    blockers = [
        *list(anomaly.get("blockers") or []),
        *[f"recipient_{blocker}" for blocker in recipient.get("blockers") or []],
    ]
    if residual > float(tolerance):
        blockers.append("total_exchange_current_not_closed")
    return {
        "EXCHANGE_CURRENT_CLOSURE_RECEIPT": closed,
        "anomaly_current": anomaly,
        "recipient_current": recipient,
        "Q_A_plus_Q_R_mu": total,
        "exchange_current_residual": residual,
        "blockers": blockers,
        "claim_boundary": (
            "Covariant repair-exchange closure receipt for nabla_a T_A^{ab}=Q_A^b and "
            "nabla_a T_R^{ab}=Q_R^b. Nonzero exchange is promotable only when Q_A^b+Q_R^b=0."
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


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _metric(value: Sequence[Sequence[float]] | None) -> list[list[float]]:
    if value is None:
        return [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    rows = [[float(item) for item in row] for row in value]
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("metric must be a 4x4 matrix")
    return rows


def _vector4(value: Sequence[float], *, name: str) -> list[float]:
    vector = [float(item) for item in value]
    if len(vector) != 4:
        raise ValueError(f"{name} must have four components")
    return vector


def _quadratic(metric: Sequence[Sequence[float]], vector: Sequence[float]) -> float:
    return _bilinear(metric, vector, vector)


def _bilinear(metric: Sequence[Sequence[float]], left: Sequence[float], right: Sequence[float]) -> float:
    return float(
        sum(float(left[i]) * float(metric[i][j]) * float(right[j]) for i in range(4) for j in range(4))
    )
