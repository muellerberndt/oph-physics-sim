from __future__ import annotations

import numpy as np

from oph_fpe.cosmology.screen_to_primordial import (
    ThinShellLift,
    exact_shell_gamma_lift_receipt,
    exact_thin_shell_cl,
)


def screen_cl_to_primordial_modulation(
    k: np.ndarray,
    *,
    D_star_mpc: float,
    ell_grid: np.ndarray,
    cl_oph: np.ndarray,
    cl_base: np.ndarray,
) -> np.ndarray:
    """Map a screen C_l ratio to a diagnostic ell=kD primordial modulation.

    This is intentionally a scaffold. It must not be used as a passed
    screen-to-primordial lift receipt; exact lifts should use the Bessel/gamma
    shell functions below or a finite-window Bessel-kernel certificate.
    """

    k_values = np.asarray(k, dtype=float)
    ell_values = np.asarray(ell_grid, dtype=float)
    ratio = np.asarray(cl_oph, dtype=float) / np.maximum(np.asarray(cl_base, dtype=float), 1e-30)
    ell_of_k = np.maximum(k_values * float(D_star_mpc), 2.0)
    return np.interp(ell_of_k, ell_values, ratio, left=float(ratio[0]), right=1.0)


def exact_thin_shell_primordial_bridge(
    ell: np.ndarray,
    params: ThinShellLift,
) -> dict[str, np.ndarray]:
    """Return exact thin-shell screen power from a primordial power law."""

    ell_values = np.asarray(ell, dtype=float)
    return {
        "ell": ell_values,
        "C_ell_q": np.asarray(exact_thin_shell_cl(ell_values, params), dtype=float),
    }


def thin_shell_lift_receipt_from_screen_amplitude(
    *,
    A_q_shell: float,
    theta: float,
    k_pivot: float,
    D_star: float,
    W_star_hash: str | None,
    Z_star: float = 1.0,
    Z_star_source: str | None,
    D_star_source: str | None,
    bessel_kernel_hash: str | None,
    kernel_rank: int | None,
    kernel_condition_number: float | None,
    nullspace_dimension: int | None,
    radial_prior: str | None,
    forward_projection_residual: float | None,
    A_q_source: str | None,
) -> dict:
    """Build the issue-330 exact-shell lift receipt from screen amplitude."""

    return exact_shell_gamma_lift_receipt(
        A_q_shell=A_q_shell,
        theta=theta,
        k_pivot=k_pivot,
        D_star=D_star,
        Z_star=Z_star,
        W_star_hash=W_star_hash,
        Z_star_source=Z_star_source,
        D_star_source=D_star_source,
        bessel_kernel_hash=bessel_kernel_hash,
        kernel_rank=kernel_rank,
        kernel_condition_number=kernel_condition_number,
        nullspace_dimension=nullspace_dimension,
        radial_prior=radial_prior,
        forward_projection_residual=forward_projection_residual,
        A_q_source=A_q_source,
        ell_equals_kD_scaffold_only=False,
    )


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
