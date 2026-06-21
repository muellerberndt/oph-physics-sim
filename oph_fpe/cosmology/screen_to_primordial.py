from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ThinShellLift:
    """Exact thin-shell screen-to-primordial power-law lift parameters.

    The shell model is a conditional lift, not a generic inversion of screen
    data. `D_star` must use the inverse unit of `k_pivot`.
    """

    A_zeta: float
    theta: float
    k_pivot: float
    D_star: float
    Z_star: float = 1.0

    def as_jsonable(self) -> dict[str, float]:
        return asdict(self)


def shell_precision_eigenvalue(ell: float | np.ndarray, theta: float) -> float | np.ndarray:
    """Exact thin-shell gamma-ratio precision eigenvalue.

    For theta=0 this reduces to ell*(ell+1). The expression is evaluated in
    log-gamma form so moderately large multipoles remain stable.
    """

    ell_arr = np.asarray(ell, dtype=float)
    values = np.exp(
        np.vectorize(math.lgamma)(ell_arr + 2.0 + 0.5 * float(theta))
        - np.vectorize(math.lgamma)(ell_arr - 0.5 * float(theta))
    )
    if np.isscalar(ell):
        return float(values)
    return values


def shell_action_amplitude(params: ThinShellLift) -> float:
    """Return A_q^shell for the exact thin-shell theorem."""

    return float(
        math.pi**1.5
        * float(params.Z_star) ** 2
        * float(params.A_zeta)
        * (float(params.k_pivot) * float(params.D_star)) ** float(params.theta)
        * math.exp(math.lgamma(1.0 + 0.5 * float(params.theta)) - math.lgamma(1.5 + 0.5 * float(params.theta)))
    )


def exact_thin_shell_cl(ell: float | np.ndarray, params: ThinShellLift) -> float | np.ndarray:
    """Exact projected screen C_ell for a thin shell and 3D power law."""

    amplitude = shell_action_amplitude(params)
    values = amplitude / shell_precision_eigenvalue(ell, params.theta)
    if np.isscalar(ell):
        return float(values)
    return values


def fractional_screen_cl(ell: float | np.ndarray, A_q: float, theta: float) -> float | np.ndarray:
    """Intrinsic fractional screen spectrum used before a radial lift."""

    ell_arr = np.asarray(ell, dtype=float)
    values = float(A_q) / (ell_arr * (ell_arr + 1.0)) ** (1.0 + 0.5 * float(theta))
    if np.isscalar(ell):
        return float(values)
    return values


def A_zeta_from_shell_action_amplitude(
    A_q_shell: float,
    theta: float,
    k_pivot: float,
    D_star: float,
    Z_star: float = 1.0,
) -> float:
    """Invert the exact thin-shell amplitude relation."""

    return float(
        float(A_q_shell)
        * math.exp(math.lgamma(1.5 + 0.5 * float(theta)) - math.lgamma(1.0 + 0.5 * float(theta)))
        / (math.pi**1.5 * float(Z_star) ** 2 * (float(k_pivot) * float(D_star)) ** float(theta))
    )


def exact_shell_gamma_lift_receipt(
    *,
    A_q_shell: float,
    theta: float,
    k_pivot: float,
    D_star: float,
    Z_star: float = 1.0,
    W_star_hash: str | None,
    Z_star_source: str | None,
    D_star_source: str | None,
    bessel_kernel_hash: str | None,
    kernel_rank: int | None,
    kernel_condition_number: float | None,
    nullspace_dimension: int | None,
    radial_prior: str | None,
    finite_width_bound_by_l: dict[str, float] | None = None,
    forward_projection_residual: float | None,
    A_q_source: str | None,
    residual_tolerance: float = 1.0e-8,
    ell_equals_kD_scaffold_only: bool = False,
) -> dict[str, Any]:
    """Validate the theorem-side thin-shell lift receipt.

    This receipt intentionally does not accept the old ell=kD interpolation as
    evidence. It computes A_zeta only when the exact shell model and forward
    projection checks are declared.
    """

    blockers: list[str] = []
    if ell_equals_kD_scaffold_only:
        blockers.append("ell_equals_kD_scaffold_only")
    if not W_star_hash:
        blockers.append("W_star_hash_missing")
    if not Z_star_source or not _finite_positive(Z_star):
        blockers.append("Z_star_missing")
    if not D_star_source or not _finite_positive(D_star):
        blockers.append("D_star_missing")
    if not bessel_kernel_hash:
        blockers.append("bessel_kernel_hash_missing")
    if kernel_rank is None or int(kernel_rank) <= 0:
        blockers.append("kernel_rank_missing")
    if kernel_condition_number is None or not np.isfinite(float(kernel_condition_number)):
        blockers.append("kernel_condition_number_missing")
    if nullspace_dimension is None or int(nullspace_dimension) < 0:
        blockers.append("nullspace_dimension_missing")
    if not radial_prior:
        blockers.append("radial_prior_missing")
    if forward_projection_residual is None or not np.isfinite(float(forward_projection_residual)):
        blockers.append("forward_projection_residual_missing")
    elif float(forward_projection_residual) > float(residual_tolerance):
        blockers.append("forward_projection_residual_above_tolerance")
    if not A_q_source or not _finite_positive(A_q_shell):
        blockers.append("A_q_shell_missing")
    if not _finite_positive(k_pivot):
        blockers.append("k_pivot_missing")
    if not np.isfinite(float(theta)):
        blockers.append("theta_missing")

    A_zeta = None
    if not blockers:
        A_zeta = A_zeta_from_shell_action_amplitude(
            A_q_shell=float(A_q_shell),
            theta=float(theta),
            k_pivot=float(k_pivot),
            D_star=float(D_star),
            Z_star=float(Z_star),
        )

    passed = not blockers
    return {
        "receipt": "SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT",
        "passed": passed,
        "lift_model": "thin_shell_power_law",
        "operator_family_match": "exact_shell_gamma",
        "W_star_hash": W_star_hash,
        "Z_star": float(Z_star) if _finite_positive(Z_star) else Z_star,
        "Z_star_source": Z_star_source,
        "D_star": float(D_star) if _finite_positive(D_star) else D_star,
        "D_star_source": D_star_source,
        "bessel_kernel_hash": bessel_kernel_hash,
        "kernel_rank": kernel_rank,
        "kernel_condition_number": kernel_condition_number,
        "nullspace_dimension": nullspace_dimension,
        "radial_prior": radial_prior,
        "finite_width_bound_by_l": finite_width_bound_by_l or {},
        "forward_projection_residual": forward_projection_residual,
        "forward_projection_residual_tolerance": float(residual_tolerance),
        "A_q_source": A_q_source,
        "A_q_shell": float(A_q_shell) if _finite_positive(A_q_shell) else A_q_shell,
        "theta": float(theta) if np.isfinite(float(theta)) else theta,
        "k_pivot": float(k_pivot) if _finite_positive(k_pivot) else k_pivot,
        "A_zeta_derived": A_zeta,
        "ell_equals_kD_scaffold_only": bool(ell_equals_kD_scaffold_only),
        "blockers": blockers,
        "claim_boundary": (
            "Conditional screen-to-primordial lift receipt. Passing this receipt licenses only the "
            "declared thin-shell power-law lift; physical TT/TE/EE spectra still require transfer and "
            "likelihood receipts."
        ),
    }


def transfer_firewall_receipt(
    *,
    screen_spectrum_derived: bool,
    primordial_lift_derived: bool,
    rho_A_finite_derived: bool,
    B_A_k_a_finite_derived: bool,
    Gamma_rec_k_a_finite_derived: bool,
    cdm_limit_regression_passed: bool,
    recombination_inputs_ready: bool,
    official_likelihood_ready: bool,
) -> dict[str, Any]:
    gates = {
        "screen_spectrum_derived": bool(screen_spectrum_derived),
        "primordial_lift_derived": bool(primordial_lift_derived),
        "rho_A_finite_derived": bool(rho_A_finite_derived),
        "B_A_k_a_finite_derived": bool(B_A_k_a_finite_derived),
        "Gamma_rec_k_a_finite_derived": bool(Gamma_rec_k_a_finite_derived),
        "cdm_limit_regression_passed": bool(cdm_limit_regression_passed),
        "recombination_inputs_ready": bool(recombination_inputs_ready),
        "official_likelihood_ready": bool(official_likelihood_ready),
    }
    physical = all(gates.values())
    return {
        "receipt": "TRANSFER_FIREWALL_RECEIPT",
        "passed": physical,
        **gates,
        "physical_cmb_prediction": physical,
        "claim_boundary": (
            "TT/TE/EE promotion is the conjunction of the primordial lift, finite source kernels, "
            "CDM-limit, recombination, and official-likelihood gates."
        ),
    }


def _finite_positive(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)) and float(value) > 0.0)
    except (TypeError, ValueError):
        return False
