"""Source-side mathematics and fail-closed receipts for the OPH screen spectrum.

The functions in this module do not fit sky data and do not run a transfer
solver. A conditional formula becomes a passed source receipt only when its
finite source evidence is supplied explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eigh, null_space
from scipy.special import digamma, gammaln, spherical_jn

from oph_fpe.constants.oph_pixel import P_STAR

PHI = (1.0 + math.sqrt(5.0)) / 2.0


class SourceSpectrumInputError(ValueError):
    """A source-side input violates the conditional theorem contract."""


RadialLiftInputError = SourceSpectrumInputError


@dataclass(frozen=True)
class EdgeCenterTilt:
    P: float
    theta: float
    n_s: float
    kappa_rep_edge: float
    full_collar_generator_density: float


@dataclass(frozen=True)
class SourceAmplitude:
    A_q: float
    E_src: float
    mode_count: int
    sample_count: int
    total_weight: float
    min_precision_eigenvalue: float
    max_precision_eigenvalue: float


@dataclass(frozen=True)
class PrimordialLift:
    A_q: float
    theta: float
    A_zeta: float
    k_pivot: float
    R_star: float
    Z_q: float
    source_pivot: bool
    kR_pivot: float
    conversion_factor: float


@dataclass(frozen=True)
class RadialNullSpace:
    shape: tuple[int, int]
    rank: int
    nullity: int
    singular_values: tuple[float, ...]
    rank_threshold: float
    condition_number_nonzero: float
    null_basis: tuple[tuple[float, ...], ...]

    @property
    def effective_threshold(self) -> float:
        """Compatibility name used by the standalone SCR330 reference."""
        return self.rank_threshold


@dataclass(frozen=True)
class WindowBound:
    """Certified deviation of a finite radial window from a thin shell."""

    ell: int
    theta: float
    I_ell_theta: float
    J_ell_theta: float
    eta: float
    shell_norm: float
    absolute_cl_bound: float
    relative_to_shell_cl_bound: float


@dataclass(frozen=True)
class PriorContinuation:
    """Minimum-Q continuation and its explicit resolution/null operators."""

    p: tuple[float, ...]
    residual: tuple[float, ...]
    residual_norm: float
    objective: float
    resolution: tuple[tuple[float, ...], ...]
    null_projector: tuple[tuple[float, ...], ...]
    effective_rank: int


@dataclass(frozen=True)
class DilationReceipt:
    """Finite safe-band consequence of the physical dilation intertwiner."""

    theta: float
    scale_ratios: tuple[float, ...]
    max_absolute_log_residual: float
    rms_log_residual: float
    passed: bool
    tolerance: float
    evaluated_pairs: int


SCR330_SCHEMA_VERSION = "scr330-radial-v2"
SCR330_MAX_NUMERICAL_TOLERANCE = 1.0e-6
RADIAL_RECEIPTS = frozenset(
    {
        "SCR330_SOURCE_SHELL_EMBEDDING_RECEIPT",
        "SCR330_PHYSICAL_MODE_BASIS_RECEIPT",
        "SCR330_RADIAL_DILATION_INTERTWINER_RECEIPT",
        "SCR330_THIN_SHELL_MELLIN_LIFT_RECEIPT",
        "SCR330_FINITE_WINDOW_KERNEL_RECEIPT",
        "SCR330_RADIAL_NULL_REPORT",
        "SCR330_RADIAL_FORWARD_RESIDUAL_RECEIPT",
        "SCR330_RADIAL_TOMOGRAPHY_RECEIPT",
        "SCR330_RADIAL_PROMOTION_RECEIPT",
        "SCR330_TRANSFER_FIREWALL_RECEIPT",
    }
)

_SOURCE_DAG_NODE_KINDS = frozenset(
    {
        "boundary",
        "certificate",
        "clock",
        "collar",
        "constant",
        "embedding",
        "finite_lattice",
        "geometry",
        "kernel",
        "maxent",
        "mode_basis",
        "operator",
        "parameter",
        "proof",
        "radial",
        "refinement",
        "release",
        "simulation",
        "source",
        "spectrum",
        "theorem",
    }
)
_DOWNSTREAM_DAG_NODE_KINDS = frozenset(
    {
        "boltzmann",
        "class_output",
        "ee_spectrum",
        "foreground",
        "lensing",
        "nuisance_parameter",
        "recombination",
        "te_spectrum",
        "transfer_output",
        "tt_spectrum",
    }
)
_MEASUREMENT_ID_TOKEN = re.compile(
    r"(?:^|[^a-z0-9])"
    r"(?:act|calibrated|data|fit|likelihood|measurement|measurements|observed|"
    r"observation|planck|posterior|target|wmap)"
    r"(?:$|[^a-z0-9])"
)

# Compatibility for the first source-screen draft.  Returned receipts always
# use the canonical SCR330 identifier so they validate against the v2 schema.
_LEGACY_RECEIPT_ALIASES = {
    "GEOMETRIC_Q": "SCR330_SOURCE_SHELL_EMBEDDING_RECEIPT",
    "COLLAR_MAXENT_SOURCE": "SCR330_SOURCE_SHELL_EMBEDDING_RECEIPT",
    "RELEASE_AMPLITUDE": "SCR330_SOURCE_SHELL_EMBEDDING_RECEIPT",
    "RESERVE_GENERATOR_TILT": "SCR330_SOURCE_SHELL_EMBEDDING_RECEIPT",
    "CONFORMAL_PRECISION": "SCR330_SOURCE_SHELL_EMBEDDING_RECEIPT",
    "SCREEN_SPECTRUM": "SCR330_SOURCE_SHELL_EMBEDDING_RECEIPT",
    "THIN_SHELL_RADIAL_LIFT": "SCR330_THIN_SHELL_MELLIN_LIFT_RECEIPT",
    "RADIAL_NULL_REPORT": "SCR330_RADIAL_NULL_REPORT",
    "TRANSFER_FIREWALL": "SCR330_TRANSFER_FIREWALL_RECEIPT",
}


def edge_center_tilt(
    *,
    P: float = P_STAR,
    full_collar_generator_density: float,
    orientation_halves: int = 2,
) -> EdgeCenterTilt:
    """Apply the conditional edge-center theorem to an emitted generator density.

    The function requires the full-collar density as input. It does not turn the
    expected value ``P/24`` into source evidence.
    """
    P = _finite_positive(P, "P")
    rho_full = _finite_positive(
        full_collar_generator_density, "full_collar_generator_density"
    )
    if orientation_halves != 2:
        raise SourceSpectrumInputError("the source branch requires two orientation halves")
    theta = rho_full / float(orientation_halves)
    if P <= PHI:
        raise SourceSpectrumInputError("P must exceed phi on the selected pixel branch")
    return EdgeCenterTilt(
        P=P,
        theta=theta,
        n_s=1.0 - theta,
        kappa_rep_edge=theta / (P - PHI),
        full_collar_generator_density=rho_full,
    )


def theta_from_step_survival(survival: float, scale_ratio: float) -> float:
    """Convert one finite survival factor to its exact semigroup exponent."""
    survival = float(survival)
    scale_ratio = float(scale_ratio)
    if not (0.0 < survival <= 1.0):
        raise SourceSpectrumInputError("survival must lie in (0, 1]")
    if not (math.isfinite(scale_ratio) and scale_ratio > 1.0):
        raise SourceSpectrumInputError("scale_ratio must be finite and greater than one")
    return -math.log(survival) / math.log(scale_ratio)


def mellin_spherical_bessel_square(ell: int, theta: float) -> float:
    r"""Return ``I_l(theta) = int dln(x) x^-theta j_l(x)^2``.

    The gamma-function expression is exact on its absolute-convergence strip
    ``-2 < theta < 2*ell``.  The common retained scalar band begins at
    ``ell=2``, for which ``(-2, 4)`` is sufficient.
    """
    ell = _integer_at_least(ell, 1, "ell")
    theta = _finite(theta, "theta")
    if not (-2.0 < theta < 2.0 * ell):
        raise SourceSpectrumInputError(
            f"Mellin integral requires -2 < theta < 2*ell; got ell={ell}, theta={theta}"
        )
    log_value = (
        0.5 * math.log(math.pi)
        - math.log(4.0)
        + gammaln(1.0 + 0.5 * theta)
        - gammaln(1.5 + 0.5 * theta)
        + gammaln(ell - 0.5 * theta)
        - gammaln(ell + 2.0 + 0.5 * theta)
    )
    value = math.exp(log_value)
    if not (math.isfinite(value) and value > 0.0):
        raise SourceSpectrumInputError(
            "Mellin integral evaluated to a nonpositive or nonfinite value"
        )
    return value


def derivative_mellin_norm(ell: int, theta: float) -> float:
    r"""Return ``J_l(theta) = int dln(x) x^(2-theta) j_l'(x)^2``.

    Integration by parts with the spherical-Bessel equation gives
    ``J_l(theta) = I_l(theta-2) - [l(l+1)-theta(theta+1)/2] I_l(theta)``.
    """
    ell = _integer_at_least(ell, 1, "ell")
    theta = _finite(theta, "theta")
    if not (0.0 < theta < 2.0 * ell):
        raise SourceSpectrumInputError(
            "derivative Mellin norm requires 0 < theta < 2*ell"
        )
    value = mellin_spherical_bessel_square(ell, theta - 2.0) - (
        ell * (ell + 1.0) - 0.5 * theta * (theta + 1.0)
    ) * mellin_spherical_bessel_square(ell, theta)
    if value < 0.0 and abs(value) < 1.0e-13:
        value = 0.0
    if not (math.isfinite(value) and value >= 0.0):
        raise SourceSpectrumInputError(
            "derived derivative Mellin norm is negative or nonfinite"
        )
    return value


def conformal_precision_eigenvalue(
    ell: ArrayLike, theta: float
) -> NDArray[np.float64] | float:
    """Exact gamma-ratio precision on the retained scalar modes."""
    theta = float(theta)
    if not (-2.0 < theta < 4.0):
        raise SourceSpectrumInputError("theta must lie in (-2, 4)")
    ell_array = np.asarray(ell, dtype=float)
    if (
        np.any(~np.isfinite(ell_array))
        or np.any(ell_array < 2.0)
        or np.any(ell_array != np.floor(ell_array))
    ):
        raise SourceSpectrumInputError(
            "ell must contain finite integer multipoles at least two"
        )
    values = np.exp(
        gammaln(ell_array + 2.0 + 0.5 * theta)
        - gammaln(ell_array - 0.5 * theta)
    )
    return float(values) if np.isscalar(ell) else values


def screen_cl(
    ell: ArrayLike, A_q: float, theta: float
) -> NDArray[np.float64] | float:
    """Return the exact source-screen angular covariance."""
    amplitude = _finite_positive(A_q, "A_q")
    values = amplitude / conformal_precision_eigenvalue(ell, theta)
    return float(values) if np.isscalar(ell) else np.asarray(values, dtype=float)


def screen_gamma_ratio_cl(
    ell: ArrayLike, A_q: float, theta: float
) -> NDArray[np.float64] | float:
    """Canonical SCR330 name for :func:`screen_cl`."""
    return screen_cl(ell, A_q, theta)


def source_amplitude_from_samples(
    q_samples: ArrayLike,
    mass_matrix: ArrayLike,
    precision: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    symmetry_tolerance: float = 1.0e-10,
) -> SourceAmplitude:
    """Compute ``A_q = E[q^T M K q] / d`` from source-release samples."""
    q = np.asarray(q_samples, dtype=float)
    if q.ndim == 1:
        q = q[None, :]
    if q.ndim != 2 or not q.size or np.any(~np.isfinite(q)):
        raise SourceSpectrumInputError("q_samples must be a finite nonempty matrix")
    mode_count = int(q.shape[1])
    mass = np.asarray(mass_matrix, dtype=float)
    operator = np.asarray(precision, dtype=float)
    if mass.shape != (mode_count, mode_count) or operator.shape != mass.shape:
        raise SourceSpectrumInputError("mass and precision dimensions must match the modes")
    if not np.allclose(mass, mass.T, rtol=0.0, atol=symmetry_tolerance):
        raise SourceSpectrumInputError("mass_matrix must be symmetric")
    quadratic = mass @ operator
    if not np.allclose(quadratic, quadratic.T, rtol=0.0, atol=symmetry_tolerance):
        raise SourceSpectrumInputError("precision must be self-adjoint in the mass metric")
    if float(np.min(np.linalg.eigvalsh(mass))) <= 0.0:
        raise SourceSpectrumInputError("mass_matrix must be positive definite")
    eigenvalues = eigh(quadratic, mass, eigvals_only=True, check_finite=True)
    if float(np.min(eigenvalues)) <= 0.0:
        raise SourceSpectrumInputError("precision must be positive on retained modes")
    normalized_weights, total_weight = _weights(weights, int(q.shape[0]))
    sample_quadratics = np.einsum("ni,ij,nj->n", q, quadratic, q, optimize=True)
    if np.any(sample_quadratics < -1.0e-9):
        raise SourceSpectrumInputError("negative quadratic energy encountered")
    mean_quadratic = float(
        np.dot(normalized_weights, np.maximum(sample_quadratics, 0.0))
    )
    amplitude = mean_quadratic / float(mode_count)
    if not (math.isfinite(amplitude) and amplitude > 0.0):
        raise SourceSpectrumInputError("derived A_q must be positive and finite")
    return SourceAmplitude(
        A_q=amplitude,
        E_src=0.5 * mean_quadratic,
        mode_count=mode_count,
        sample_count=int(q.shape[0]),
        total_weight=total_weight,
        min_precision_eigenvalue=float(np.min(eigenvalues)),
        max_precision_eigenvalue=float(np.max(eigenvalues)),
    )


def primordial_amplitude_from_screen(
    A_q: float,
    theta: float,
    *,
    R_star: float,
    k_pivot: float | None = None,
    Z_q: float = 1.0,
) -> PrimordialLift:
    """Apply the exact thin-shell Bessel amplitude identity."""
    A_q = _finite_positive(A_q, "A_q")
    R_star = _finite_positive(R_star, "R_star")
    Z_q = _finite_positive(Z_q, "Z_q")
    theta = float(theta)
    if not (-2.0 < theta < 4.0):
        raise SourceSpectrumInputError("theta must lie in (-2, 4)")
    source_pivot = k_pivot is None
    pivot = 1.0 / R_star if k_pivot is None else _finite_positive(k_pivot, "k_pivot")
    log_factor = (
        gammaln(1.5 + 0.5 * theta)
        - gammaln(1.0 + 0.5 * theta)
        - 1.5 * math.log(math.pi)
        - 2.0 * math.log(Z_q)
        - theta * math.log(pivot * R_star)
    )
    factor = math.exp(log_factor)
    return PrimordialLift(
        A_q=A_q,
        theta=theta,
        A_zeta=A_q * factor,
        k_pivot=pivot,
        R_star=R_star,
        Z_q=Z_q,
        source_pivot=source_pivot,
        kR_pivot=pivot * R_star,
        conversion_factor=factor,
    )


def screen_amplitude_from_primordial(
    A_zeta: float,
    theta: float,
    *,
    Z_q: float,
    R_star: float,
    k_pivot: float,
) -> float:
    """Invert the exact thin-shell source-amplitude conversion."""
    A_zeta = _finite_positive(A_zeta, "A_zeta")
    Z_q = _finite_positive(Z_q, "Z_q")
    R_star = _finite_positive(R_star, "R_star")
    k_pivot = _finite_positive(k_pivot, "k_pivot")
    theta = _finite(theta, "theta")
    if not (-2.0 < theta < 4.0):
        raise SourceSpectrumInputError("theta must lie in (-2, 4)")
    log_factor = (
        1.5 * math.log(math.pi)
        + 2.0 * math.log(Z_q)
        + theta * math.log(k_pivot * R_star)
        + gammaln(1.0 + 0.5 * theta)
        - gammaln(1.5 + 0.5 * theta)
    )
    return A_zeta * math.exp(log_factor)


def primordial_amplitude_log_sensitivity(theta: float, kR_pivot: float) -> float:
    """Return ``d_theta ln(A_zeta/A_q)`` at fixed scale and normalization."""
    theta = _finite(theta, "theta")
    kR_pivot = _finite_positive(kR_pivot, "kR_pivot")
    if not (-2.0 < theta < 4.0):
        raise SourceSpectrumInputError("theta must lie in (-2, 4)")
    return float(
        -math.log(kR_pivot)
        + 0.5
        * (
            digamma(1.5 + 0.5 * theta)
            - digamma(1.0 + 0.5 * theta)
        )
    )


def thin_shell_cl(
    ell: ArrayLike,
    A_zeta: float,
    theta: float,
    *,
    R_star: float,
    k_pivot: float | None = None,
    Z_q: float = 1.0,
) -> NDArray[np.float64] | float:
    """Project the one-dimensional source power family onto a thin shell."""
    A_zeta = _finite_positive(A_zeta, "A_zeta")
    R_star = _finite_positive(R_star, "R_star")
    pivot = 1.0 / R_star if k_pivot is None else _finite_positive(k_pivot, "k_pivot")
    Z_q = _finite_positive(Z_q, "Z_q")
    theta = float(theta)
    if not (-2.0 < theta < 4.0):
        raise SourceSpectrumInputError("theta must lie in (-2, 4)")
    log_A_q = (
        1.5 * math.log(math.pi)
        + 2.0 * math.log(Z_q)
        + math.log(A_zeta)
        + theta * math.log(pivot * R_star)
        + gammaln(1.0 + 0.5 * theta)
        - gammaln(1.5 + 0.5 * theta)
    )
    return screen_cl(ell, math.exp(log_A_q), theta)


def thin_shell_powerlaw_cl(
    ell: ArrayLike,
    A_zeta: float,
    theta: float,
    *,
    Z_q: float,
    R_star: float,
    k_pivot: float,
) -> NDArray[np.float64] | float:
    """Canonical SCR330 name for the exact thin-shell power-law lift."""
    return thin_shell_cl(
        ell,
        A_zeta,
        theta,
        R_star=R_star,
        k_pivot=k_pivot,
        Z_q=Z_q,
    )


def normalized_radial_window(
    radii: ArrayLike, weights: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and normalize a nonnegative discrete radial window."""
    radius = np.asarray(radii, dtype=float)
    raw_weights = np.asarray(weights, dtype=float)
    if radius.ndim != 1 or raw_weights.shape != radius.shape or radius.size == 0:
        raise SourceSpectrumInputError(
            "radii and weights must be nonempty matching vectors"
        )
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise SourceSpectrumInputError("radii must be positive and finite")
    if np.any(~np.isfinite(raw_weights)) or np.any(raw_weights < 0.0):
        raise SourceSpectrumInputError(
            "radial weights must be finite and nonnegative"
        )
    total = float(np.sum(raw_weights))
    if total <= 0.0:
        raise SourceSpectrumInputError("radial weights must have positive total")
    return radius, raw_weights / total


def window_transfer(
    ell: int,
    k: ArrayLike,
    radii: ArrayLike,
    weights: ArrayLike,
) -> NDArray[np.float64]:
    r"""Return ``Psi_l(k) = sum_i w_i j_l(k r_i)``."""
    ell = _integer_at_least(ell, 0, "ell")
    k_values = np.asarray(k, dtype=float)
    if (
        k_values.ndim != 1
        or k_values.size == 0
        or np.any(~np.isfinite(k_values))
        or np.any(k_values <= 0.0)
    ):
        raise SourceSpectrumInputError("k must be a nonempty positive finite vector")
    radius, normalized_weights = normalized_radial_window(radii, weights)
    return np.asarray(
        spherical_jn(ell, np.outer(k_values, radius)) @ normalized_weights,
        dtype=float,
    )


def window_powerlaw_cl_quadrature(
    ell: int,
    A_zeta: float,
    theta: float,
    *,
    Z_q: float,
    k_pivot: float,
    k: ArrayLike,
    dlnk_weights: ArrayLike,
    radii: ArrayLike,
    radial_weights: ArrayLike,
) -> float:
    """Evaluate the finite radial-window forward projection without fitting."""
    ell = _integer_at_least(ell, 0, "ell")
    A_zeta = _finite_positive(A_zeta, "A_zeta")
    Z_q = _finite_positive(Z_q, "Z_q")
    k_pivot = _finite_positive(k_pivot, "k_pivot")
    theta = _finite(theta, "theta")
    k_values = np.asarray(k, dtype=float)
    quadrature_weights = np.asarray(dlnk_weights, dtype=float)
    if (
        k_values.ndim != 1
        or quadrature_weights.shape != k_values.shape
        or k_values.size == 0
    ):
        raise SourceSpectrumInputError(
            "k and dlnk_weights must be nonempty matching vectors"
        )
    if np.any(~np.isfinite(k_values)) or np.any(k_values <= 0.0):
        raise SourceSpectrumInputError("k must be positive and finite")
    if (
        np.any(~np.isfinite(quadrature_weights))
        or np.any(quadrature_weights <= 0.0)
    ):
        raise SourceSpectrumInputError(
            "dlnk_weights must be positive and finite"
        )
    transfer = window_transfer(ell, k_values, radii, radial_weights)
    power = A_zeta * (k_values / k_pivot) ** (-theta)
    value = 4.0 * math.pi * Z_q**2 * float(
        np.dot(quadrature_weights, power * transfer**2)
    )
    return _finite_positive(value, "projected C_ell")


def finite_window_stability_bound(
    ell: int,
    theta: float,
    *,
    A_zeta: float,
    Z_q: float,
    k_pivot: float,
    R_star: float,
    radii: ArrayLike,
    radial_weights: ArrayLike,
) -> WindowBound:
    """Return the SCR330 Hilbert-space bound for a finite radial window."""
    ell = _integer_at_least(ell, 1, "ell")
    theta = _finite(theta, "theta")
    if not (0.0 < theta < 2.0 * ell):
        raise SourceSpectrumInputError(
            "window stability theorem requires 0 < theta < 2*ell"
        )
    A_zeta = _finite_positive(A_zeta, "A_zeta")
    Z_q = _finite_positive(Z_q, "Z_q")
    k_pivot = _finite_positive(k_pivot, "k_pivot")
    R_star = _finite_positive(R_star, "R_star")
    radius, weights = normalized_radial_window(radii, radial_weights)
    mellin = mellin_spherical_bessel_square(ell, theta)
    derivative_norm = derivative_mellin_norm(ell, theta)
    exponent = 0.5 * theta
    eta = (2.0 * math.sqrt(derivative_norm) / theta) * float(
        np.dot(weights, np.abs(radius**exponent - R_star**exponent))
    )
    shell_norm = R_star**exponent * math.sqrt(mellin)
    prefactor = 4.0 * math.pi * Z_q**2 * A_zeta * k_pivot**theta
    absolute_bound = prefactor * eta * (2.0 * shell_norm + eta)
    shell_cl_value = prefactor * shell_norm**2
    return WindowBound(
        ell=ell,
        theta=theta,
        I_ell_theta=mellin,
        J_ell_theta=derivative_norm,
        eta=eta,
        shell_norm=shell_norm,
        absolute_cl_bound=absolute_bound,
        relative_to_shell_cl_bound=absolute_bound / shell_cl_value,
    )


def radial_projection_matrix(
    ell: ArrayLike,
    k: ArrayLike,
    dlnk_weights: ArrayLike,
    *,
    Z_q: float,
    radii: ArrayLike,
    radial_weights: ArrayLike,
) -> NDArray[np.float64]:
    r"""Build ``A_lj = 4*pi*Z_q^2*w_j*|Psi_l(k_j)|^2``."""
    raw_ell_values = np.asarray(ell, dtype=float)
    k_values = np.asarray(k, dtype=float)
    quadrature_weights = np.asarray(dlnk_weights, dtype=float)
    Z_q = _finite_positive(Z_q, "Z_q")
    if (
        raw_ell_values.ndim != 1
        or raw_ell_values.size == 0
        or np.any(~np.isfinite(raw_ell_values))
        or np.any(raw_ell_values < 0)
        or np.any(raw_ell_values != np.floor(raw_ell_values))
    ):
        raise SourceSpectrumInputError(
            "ell must be a nonempty vector of nonnegative integers"
        )
    ell_values = raw_ell_values.astype(int)
    if (
        k_values.ndim != 1
        or quadrature_weights.shape != k_values.shape
        or k_values.size == 0
    ):
        raise SourceSpectrumInputError(
            "k and dlnk_weights must be nonempty matching vectors"
        )
    if np.any(~np.isfinite(k_values)) or np.any(k_values <= 0.0):
        raise SourceSpectrumInputError("k must be positive and finite")
    if (
        np.any(~np.isfinite(quadrature_weights))
        or np.any(quadrature_weights <= 0.0)
    ):
        raise SourceSpectrumInputError(
            "dlnk_weights must be positive and finite"
        )
    matrix = np.empty((ell_values.size, k_values.size), dtype=float)
    for index, ell_value in enumerate(ell_values):
        transfer = window_transfer(
            int(ell_value), k_values, radii, radial_weights
        )
        matrix[index, :] = (
            4.0 * math.pi * Z_q**2 * quadrature_weights * transfer**2
        )
    return matrix


def radial_kernel_matrix(
    ell: ArrayLike,
    k: ArrayLike,
    *,
    R_star: float,
    dlnk_weights: ArrayLike | None = None,
    Z_q: float = 1.0,
) -> NDArray[np.float64]:
    """Backward-compatible thin-shell wrapper around the SCR330 operator."""
    k_values = np.asarray(k, dtype=float)
    if dlnk_weights is None:
        weights = (
            np.ones(1, dtype=float)
            if k_values.size == 1
            else np.gradient(np.log(k_values))
        )
    else:
        weights = np.asarray(dlnk_weights, dtype=float)
    return radial_projection_matrix(
        ell,
        k_values,
        weights,
        Z_q=Z_q,
        radii=[_finite_positive(R_star, "R_star")],
        radial_weights=[1.0],
    )


def radial_null_space(matrix: ArrayLike, *, rtol: float = 1.0e-12) -> RadialNullSpace:
    """Report the unrestricted radial kernel's right null space."""
    operator = np.asarray(matrix, dtype=float)
    if operator.ndim != 2 or not operator.size or np.any(~np.isfinite(operator)):
        raise SourceSpectrumInputError("matrix must be finite, nonempty, and two-dimensional")
    rtol = _finite_positive(rtol, "rtol")
    singular_values = np.linalg.svd(operator, compute_uv=False)
    threshold = rtol * float(singular_values[0])
    rank = int(np.sum(singular_values > threshold))
    positive = singular_values[singular_values > threshold]
    basis = null_space(operator, rcond=rtol).T
    return RadialNullSpace(
        shape=(int(operator.shape[0]), int(operator.shape[1])),
        rank=rank,
        nullity=int(operator.shape[1] - rank),
        singular_values=tuple(float(value) for value in singular_values),
        rank_threshold=float(threshold),
        condition_number_nonzero=float(positive[0] / positive[-1]) if positive.size else math.inf,
        null_basis=tuple(tuple(float(value) for value in row) for row in basis),
    )


def radial_null_space_report(
    matrix: ArrayLike, *, rtol: float = 1.0e-12
) -> RadialNullSpace:
    """Canonical SCR330 name for :func:`radial_null_space`."""
    return radial_null_space(matrix, rtol=rtol)


def minimum_prior_continuation(
    matrix: ArrayLike,
    screen_cl_values: ArrayLike,
    *,
    prior_center: ArrayLike,
    prior_precision: ArrayLike,
    rtol: float = 1.0e-12,
) -> PriorContinuation:
    r"""Return the minimum-Q representative satisfying ``A p = C``.

    This is explicitly a ``PRIOR_CONTINUATION`` diagnostic and never a
    source-derived E4 uniqueness receipt.
    """
    operator = np.asarray(matrix, dtype=float)
    covariance = np.asarray(screen_cl_values, dtype=float)
    center = np.asarray(prior_center, dtype=float)
    precision = np.asarray(prior_precision, dtype=float)
    rtol = _finite_positive(rtol, "rtol")
    if (
        operator.ndim != 2
        or np.any(~np.isfinite(operator))
        or covariance.shape != (operator.shape[0],)
        or center.shape != (operator.shape[1],)
        or np.any(~np.isfinite(covariance))
        or np.any(~np.isfinite(center))
    ):
        raise SourceSpectrumInputError(
            "matrix, screen covariance, and prior center dimensions must match"
        )
    if (
        precision.shape != (operator.shape[1], operator.shape[1])
        or np.any(~np.isfinite(precision))
    ):
        raise SourceSpectrumInputError(
            "prior_precision must be a finite square matrix"
        )
    if not np.allclose(precision, precision.T, rtol=0.0, atol=1.0e-12):
        raise SourceSpectrumInputError("prior_precision must be symmetric")
    if float(np.min(np.linalg.eigvalsh(precision))) <= 0.0:
        raise SourceSpectrumInputError(
            "prior_precision must be positive definite"
        )
    inverse_precision = np.linalg.inv(precision)
    gram = operator @ inverse_precision @ operator.T
    gram_pinv = np.linalg.pinv(gram, rcond=rtol, hermitian=True)
    resolution = inverse_precision @ operator.T @ gram_pinv @ operator
    continuation = center + inverse_precision @ operator.T @ gram_pinv @ (
        covariance - operator @ center
    )
    residual = covariance - operator @ continuation
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm > 100.0 * rtol * max(float(np.linalg.norm(covariance)), 1.0):
        raise SourceSpectrumInputError(
            "exact constraints are inconsistent at the requested tolerance"
        )
    displacement = continuation - center
    null_projector = np.eye(operator.shape[1], dtype=float) - resolution
    return PriorContinuation(
        p=tuple(float(value) for value in continuation),
        residual=tuple(float(value) for value in residual),
        residual_norm=residual_norm,
        objective=0.5 * float(displacement @ precision @ displacement),
        resolution=tuple(
            tuple(float(value) for value in row) for row in resolution
        ),
        null_projector=tuple(
            tuple(float(value) for value in row) for row in null_projector
        ),
        effective_rank=radial_null_space(operator, rtol=rtol).rank,
    )


def source_powerlaw(
    k: ArrayLike,
    A_zeta: float,
    theta: float,
    k_pivot: float,
) -> NDArray[np.float64]:
    """Return the one-dimensional source-derived radial power family."""
    k_values = np.asarray(k, dtype=float)
    if (
        k_values.ndim != 1
        or k_values.size == 0
        or np.any(~np.isfinite(k_values))
        or np.any(k_values <= 0.0)
    ):
        raise SourceSpectrumInputError("k must be a nonempty positive finite vector")
    return _finite_positive(A_zeta, "A_zeta") * (
        k_values / _finite_positive(k_pivot, "k_pivot")
    ) ** (-_finite(theta, "theta"))


def forward_residual(
    matrix: ArrayLike,
    spectrum: ArrayLike,
    screen_cl_values: ArrayLike,
) -> dict[str, Any]:
    """Compute a signed forward residual without refitting source parameters."""
    operator = np.asarray(matrix, dtype=float)
    power = np.asarray(spectrum, dtype=float)
    covariance = np.asarray(screen_cl_values, dtype=float)
    if (
        operator.ndim != 2
        or np.any(~np.isfinite(operator))
        or power.shape != (operator.shape[1],)
        or covariance.shape != (operator.shape[0],)
        or np.any(~np.isfinite(power))
        or np.any(~np.isfinite(covariance))
    ):
        raise SourceSpectrumInputError(
            "matrix, spectrum, and screen covariance dimensions must match"
        )
    predicted = operator @ power
    residual = covariance - predicted
    absolute = float(np.linalg.norm(residual))
    denominator = max(float(np.linalg.norm(covariance)), 1.0e-300)
    return {
        "predicted": predicted.tolist(),
        "residual": residual.tolist(),
        "absolute_l2_residual": absolute,
        "relative_l2_residual": absolute / denominator,
    }


def source_family_forward_residual(
    matrix: ArrayLike,
    screen_cl_values: ArrayLike,
    theta: float,
    k: ArrayLike,
    *,
    k_pivot: float,
    A_zeta: float,
) -> dict[str, Any]:
    """Evaluate the declared one-dimensional source family without fitting."""
    operator = np.asarray(matrix, dtype=float)
    observed_screen = np.asarray(screen_cl_values, dtype=float)
    k_values = np.asarray(k, dtype=float)
    if (
        operator.ndim != 2
        or observed_screen.shape != (operator.shape[0],)
        or k_values.shape != (operator.shape[1],)
        or np.any(~np.isfinite(operator))
        or np.any(~np.isfinite(observed_screen))
        or np.any(~np.isfinite(k_values))
        or np.any(k_values <= 0.0)
    ):
        raise SourceSpectrumInputError("matrix, screen covariance, and k dimensions are inconsistent")
    pivot = _finite_positive(k_pivot, "k_pivot")
    source_amplitude = _finite_positive(A_zeta, "A_zeta")
    report = forward_residual(
        operator,
        source_powerlaw(k_values, source_amplitude, theta, pivot),
        observed_screen,
    )
    return {**report, "source_family_dimension": 1}


def dilation_intertwiner_receipt(
    k: ArrayLike,
    delta_zeta_sq: ArrayLike,
    theta: float,
    *,
    scale_ratios: Sequence[float],
    tolerance: float,
) -> DilationReceipt:
    r"""Test ``Delta^2(bk) = b^-theta Delta^2(k)`` on a safe band.

    This is the finite diagonal consequence of ``D U = U R``.  It does not by
    itself prove the source embedding commutative square, covariance
    naturality, strong convergence, or the uniform operator bound; those stay
    explicit fields in the SCR330 receipt payload.
    """
    k_values = np.asarray(k, dtype=float)
    power = np.asarray(delta_zeta_sq, dtype=float)
    theta = _finite(theta, "theta")
    tolerance = _finite_positive(tolerance, "tolerance")
    if k_values.ndim != 1 or power.shape != k_values.shape or k_values.size < 3:
        raise SourceSpectrumInputError(
            "k and delta_zeta_sq must be matching vectors with at least three points"
        )
    if (
        np.any(~np.isfinite(k_values))
        or np.any(k_values <= 0.0)
        or np.any(np.diff(k_values) <= 0.0)
    ):
        raise SourceSpectrumInputError(
            "k must be strictly increasing, positive, and finite"
        )
    if np.any(~np.isfinite(power)) or np.any(power <= 0.0):
        raise SourceSpectrumInputError(
            "delta_zeta_sq must be positive and finite"
        )
    log_k = np.log(k_values)
    log_power = np.log(power)
    residuals: list[float] = []
    evaluated_ratios: list[float] = []
    for raw_ratio in scale_ratios:
        ratio = _finite_positive(raw_ratio, "scale_ratio")
        if math.isclose(ratio, 1.0):
            continue
        shifted = log_k + math.log(ratio)
        mask = (shifted >= log_k[0]) & (shifted <= log_k[-1])
        if not np.any(mask):
            continue
        interpolated = np.interp(shifted[mask], log_k, log_power)
        local = interpolated - log_power[mask] + theta * math.log(ratio)
        residuals.extend(float(value) for value in local)
        evaluated_ratios.append(ratio)
    if not residuals:
        raise SourceSpectrumInputError(
            "no scale-ratio pair lies inside the safe k band"
        )
    residual_array = np.asarray(residuals, dtype=float)
    max_residual = float(np.max(np.abs(residual_array)))
    return DilationReceipt(
        theta=theta,
        scale_ratios=tuple(evaluated_ratios),
        max_absolute_log_residual=max_residual,
        rms_log_residual=float(np.sqrt(np.mean(residual_array**2))),
        passed=bool(max_residual <= tolerance),
        tolerance=tolerance,
        evaluated_pairs=int(residual_array.size),
    )


def approximate_dilation_shape_bound(
    k: ArrayLike,
    epsilon_log_slope: ArrayLike,
    *,
    k_pivot: float,
) -> NDArray[np.float64]:
    r"""Integrate a bound on ``|d ln Delta^2/d ln k + theta|``."""
    k_values = np.asarray(k, dtype=float)
    epsilon = np.asarray(epsilon_log_slope, dtype=float)
    k_pivot = _finite_positive(k_pivot, "k_pivot")
    if k_values.ndim != 1 or epsilon.shape != k_values.shape or k_values.size < 2:
        raise SourceSpectrumInputError(
            "k and epsilon_log_slope must be matching vectors"
        )
    if (
        np.any(~np.isfinite(k_values))
        or np.any(k_values <= 0.0)
        or np.any(np.diff(k_values) <= 0.0)
    ):
        raise SourceSpectrumInputError(
            "k must be strictly increasing, positive, and finite"
        )
    if np.any(~np.isfinite(epsilon)) or np.any(epsilon < 0.0):
        raise SourceSpectrumInputError(
            "epsilon_log_slope must be finite and nonnegative"
        )
    if not (k_values[0] <= k_pivot <= k_values[-1]):
        raise SourceSpectrumInputError("k_pivot must lie inside the k grid")
    log_k = np.log(k_values)
    pivot_log = math.log(k_pivot)
    grid = np.unique(np.concatenate([log_k, np.array([pivot_log])]))
    epsilon_grid = np.interp(grid, log_k, epsilon)
    increments = 0.5 * (epsilon_grid[:-1] + epsilon_grid[1:]) * np.diff(grid)
    cumulative = np.concatenate([[0.0], np.cumsum(increments)])
    pivot_index = int(
        np.where(np.isclose(grid, pivot_log, rtol=0.0, atol=1.0e-14))[0][0]
    )
    return np.interp(log_k, grid, np.abs(cumulative - cumulative[pivot_index]))


def build_radial_receipt(
    *,
    receipt: str,
    passed: bool,
    claim_tier: str,
    source_dag: Mapping[str, Any],
    blockers: Iterable[str] = (),
    physical_tt_te_ee_claim: bool = False,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an exact, fail-closed ``scr330-radial-v2`` receipt."""
    if receipt not in RADIAL_RECEIPTS:
        raise SourceSpectrumInputError(f"unknown radial receipt: {receipt}")
    if claim_tier not in {"E0", "E1", "E2", "E3", "E4", "E5"}:
        raise SourceSpectrumInputError("claim_tier must be E0 through E5")
    if not isinstance(source_dag, Mapping):
        raise SourceSpectrumInputError("source_dag must be a mapping")

    blocker_set = {str(value) for value in blockers}
    declared_pass = passed
    if not isinstance(passed, bool):
        declared_pass = False
        blocker_set.add("passed_flag_not_boolean")
    if not isinstance(physical_tt_te_ee_claim, bool):
        physical_tt_te_ee_claim = False
        blocker_set.add("physical_tt_te_ee_claim_not_boolean")
    nodes = source_dag.get("nodes")
    dag_populated = isinstance(nodes, list) and bool(nodes)
    clean_ancestry = dag_populated and not _dag_has_blacklisted_ancestor(source_dag)
    if not dag_populated:
        blocker_set.add("source_dag_empty_or_invalid")
    if not clean_ancestry:
        blocker_set.add("measurement_fit_or_likelihood_ancestor")
    if claim_tier != "E5" and _dag_has_downstream_transfer_ancestor(source_dag):
        blocker_set.add("transfer_or_observable_ancestor_before_E5")
    if physical_tt_te_ee_claim and claim_tier != "E5":
        blocker_set.add("tt_te_ee_claim_before_E5")
    if receipt == "SCR330_TRANSFER_FIREWALL_RECEIPT" and claim_tier != "E5":
        blocker_set.add("downstream_transfer_requires_E5")
    blocker_set.update(_receipt_payload_blockers(receipt, payload))
    if receipt == "SCR330_RADIAL_PROMOTION_RECEIPT":
        if claim_tier != "E4":
            blocker_set.add("radial_primordial_promotion_requires_E4")

    canonical = json.dumps(
        source_dag, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    result: dict[str, Any] = {
        "schema_version": SCR330_SCHEMA_VERSION,
        "receipt": receipt,
        "passed": bool(declared_pass and not blocker_set),
        "claim_tier": claim_tier,
        "source_dag_hash": "sha256:" + hashlib.sha256(canonical).hexdigest(),
        "no_measurement_fit_likelihood_ancestor": bool(clean_ancestry),
        "physical_tt_te_ee_claim": bool(physical_tt_te_ee_claim),
        "blockers": sorted(blocker_set),
    }
    if payload is not None:
        result["payload"] = dict(payload)
    return result


def _receipt_payload_blockers(
    receipt: str,
    payload: Mapping[str, Any] | None,
) -> set[str]:
    if receipt == "SCR330_RADIAL_PROMOTION_RECEIPT":
        return _promotion_payload_blockers(payload)
    if not isinstance(payload, Mapping):
        return {"receipt_evidence_payload_missing"}

    if receipt == "SCR330_SOURCE_SHELL_EMBEDDING_RECEIPT":
        evidence = payload.get("source_shell_embedding")
        blockers: set[str] = set()
        if not _is_sha256(payload.get("source_embedding_hash")):
            blockers.add("source_embedding_hash_invalid")
        if not _is_positive_number(payload.get("R_star")):
            blockers.add("source_shell_radius_invalid")
        if payload.get("background_curvature_status") not in {
            "FlatExact",
            "FlatAssumed",
            "OpenCurved",
            "ClosedCurved",
        }:
            blockers.add("source_shell_background_unresolved")
        if not _valid_bounded_evidence(evidence, required_true=("source_derived", "refinement_natural")):
            blockers.add("source_shell_embedding_evidence_invalid")
        return blockers

    if receipt == "SCR330_PHYSICAL_MODE_BASIS_RECEIPT":
        evidence = payload.get("physical_mode_basis")
        blockers = set()
        if not isinstance(payload.get("physical_mode_basis_id"), str) or not str(
            payload.get("physical_mode_basis_id", "")
        ).strip():
            blockers.add("physical_mode_basis_id_invalid")
        if not _is_sha256(payload.get("physical_mode_basis_hash")):
            blockers.add("physical_mode_basis_hash_invalid")
        if not _valid_bounded_evidence(
            evidence,
            required_true=("source_derived", "gauge_independent", "refinement_converged"),
        ):
            blockers.add("physical_mode_basis_evidence_invalid")
        return blockers

    if receipt == "SCR330_RADIAL_DILATION_INTERTWINER_RECEIPT":
        dilation = payload.get("dilation_intertwiner")
        blockers = set()
        if not _is_sha256(payload.get("source_embedding_hash")):
            blockers.add("source_embedding_hash_invalid")
        if not isinstance(payload.get("physical_mode_basis_id"), str) or not str(
            payload.get("physical_mode_basis_id", "")
        ).strip():
            blockers.add("physical_mode_basis_id_invalid")
        if not isinstance(dilation, Mapping) or not _valid_dilation_payload(dilation):
            blockers.add("dilation_intertwiner_incomplete_or_invalid")
        else:
            if dilation.get("passed") is not True:
                blockers.add("dilation_intertwiner_not_passed")
            if dilation.get("finite_to_continuum_passed") is not True:
                blockers.add("dilation_finite_to_continuum_not_passed")
        return blockers

    if receipt == "SCR330_THIN_SHELL_MELLIN_LIFT_RECEIPT":
        return (
            set()
            if _valid_mellin_lift(payload.get("mellin_lift"))
            else {"thin_shell_mellin_evidence_invalid"}
        )

    if receipt == "SCR330_FINITE_WINDOW_KERNEL_RECEIPT":
        blockers = set()
        if not _is_sha256(payload.get("window_hash")):
            blockers.add("finite_window_hash_invalid")
        if not _valid_finite_window(payload.get("finite_window")):
            blockers.add("finite_window_evidence_invalid")
        return blockers

    if receipt == "SCR330_RADIAL_NULL_REPORT":
        radial_svd = payload.get("radial_svd")
        blockers = set()
        if not _valid_radial_svd(radial_svd):
            blockers.add("radial_svd_incomplete_or_invalid")
        elif not (
            _is_sha256(radial_svd.get("right_null_basis_hash"))
            or _is_sha256(radial_svd.get("null_basis_hash"))
        ):
            blockers.add("radial_right_null_basis_hash_missing")
        if isinstance(radial_svd, Mapping) and not _is_sha256(
            radial_svd.get("resolution_kernel_hash")
        ):
            blockers.add("radial_resolution_kernel_hash_missing")
        return blockers

    if receipt == "SCR330_RADIAL_FORWARD_RESIDUAL_RECEIPT":
        forward = payload.get("forward_residual")
        if not _valid_forward_residual(forward):
            return {"radial_forward_residual_invalid"}
        blockers = set()
        if forward.get("passed") is not True:
            blockers.add("radial_forward_residual_not_passed")
        if forward.get("held_out") is not True:
            blockers.add("radial_forward_residual_not_held_out")
        if float(forward["absolute_l2_residual"]) > float(forward["tolerance"]):
            blockers.add("radial_forward_residual_exceeds_tolerance")
        return blockers

    if receipt == "SCR330_RADIAL_TOMOGRAPHY_RECEIPT":
        return _tomography_blockers(payload.get("radial_tomography"))

    if receipt == "SCR330_TRANSFER_FIREWALL_RECEIPT":
        blockers = set()
        for key in (
            "upstream_radial_receipt_hash",
            "transfer_source_hash",
            "solver_assumption_hash",
        ):
            if not _is_sha256(payload.get(key)):
                blockers.add(f"transfer_firewall_invalid:{key}")
        if payload.get("upstream_claim_tier") != "E4":
            blockers.add("transfer_firewall_upstream_not_E4")
        if payload.get("no_back_edge_to_E4_source") is not True:
            blockers.add("transfer_firewall_back_edge_not_excluded")
        return blockers

    return {"unknown_receipt_evidence_contract"}


def write_radial_receipt(path: Path, **kwargs: Any) -> dict[str, Any]:
    """Build and write one canonical SCR330 receipt."""
    receipt = build_radial_receipt(**kwargs)
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        destination = destination / "scr330_radial_receipt.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def build_receipt(
    *,
    receipt: str,
    claimed_pass: bool,
    claim_tier: str,
    source_dag: dict[str, Any],
    blockers: Iterable[str] = (),
    physical_tt_te_ee_claim: bool = False,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper returning canonical SCR330 receipts."""
    canonical_receipt = _LEGACY_RECEIPT_ALIASES.get(receipt, receipt)
    return build_radial_receipt(
        receipt=canonical_receipt,
        passed=claimed_pass,
        claim_tier=claim_tier,
        source_dag=source_dag,
        blockers=blockers,
        physical_tt_te_ee_claim=physical_tt_te_ee_claim,
        payload=payload,
    )


def _promotion_payload_blockers(
    payload: Mapping[str, Any] | None,
) -> set[str]:
    if not isinstance(payload, Mapping):
        return {"radial_promotion_payload_missing"}
    blockers: set[str] = set()
    branch = payload.get("radial_branch")
    if branch not in {"SOURCE_DILATION", "RADIAL_TOMOGRAPHY", "PRIOR_CONTINUATION"}:
        blockers.add("radial_uniqueness_branch_missing_or_invalid")
        return blockers
    if branch == "PRIOR_CONTINUATION":
        blockers.add("prior_continuation_is_not_source_derived_E4")
        return blockers

    common = {
        "source_embedding_hash",
        "physical_mode_basis_id",
        "background_curvature_status",
        "Z_q",
        "theta",
        "A_q",
        "A_zeta",
        "k_pivot",
        "R_star",
        "radial_svd",
        "forward_residual",
    }
    for key in sorted(common - set(payload)):
        blockers.add(f"radial_promotion_missing:{key}")
    if not _is_sha256(payload.get("source_embedding_hash")):
        blockers.add("source_embedding_hash_invalid")
    if not isinstance(payload.get("physical_mode_basis_id"), str) or not str(
        payload.get("physical_mode_basis_id", "")
    ).strip():
        blockers.add("physical_mode_basis_id_invalid")
    curvature = payload.get("background_curvature_status")
    if curvature not in {
        "FlatExact",
        "FlatAssumed",
        "OpenCurved",
        "ClosedCurved",
        "Unresolved",
    }:
        blockers.add("background_curvature_status_invalid")
    if curvature == "Unresolved":
        blockers.add("background_curvature_unresolved")
    for key in ("Z_q", "A_q", "A_zeta", "k_pivot", "R_star"):
        if not _is_positive_number(payload.get(key)):
            blockers.add(f"radial_promotion_invalid:{key}")
    theta = payload.get("theta")
    if not _is_number(theta) or not (-2.0 < float(theta) < 4.0):
        blockers.add("radial_promotion_invalid:theta")

    radial_svd = payload.get("radial_svd")
    if not _valid_radial_svd(radial_svd):
        blockers.add("radial_svd_incomplete_or_invalid")
    elif not (
        _is_sha256(radial_svd.get("right_null_basis_hash"))
        or _is_sha256(radial_svd.get("null_basis_hash"))
    ):
        blockers.add("radial_right_null_basis_hash_missing")
    if isinstance(radial_svd, Mapping) and not _is_sha256(
        radial_svd.get("resolution_kernel_hash")
    ):
        blockers.add("radial_resolution_kernel_hash_missing")

    forward = payload.get("forward_residual")
    if not _valid_forward_residual(forward) or forward.get("passed") is not True:
        blockers.add("radial_forward_residual_not_passed")
    elif (
        forward.get("held_out") is not True
        or float(forward["absolute_l2_residual"]) > float(forward["tolerance"])
    ):
        blockers.add("radial_forward_residual_not_held_out_or_out_of_tolerance")

    if branch == "SOURCE_DILATION":
        dilation = payload.get("dilation_intertwiner")
        if not isinstance(dilation, Mapping):
            blockers.add("dilation_intertwiner_missing")
        else:
            if not _valid_dilation_payload(dilation):
                blockers.add("dilation_intertwiner_incomplete_or_invalid")
            if dilation.get("passed") is not True:
                blockers.add("dilation_intertwiner_not_passed")
            if dilation.get("finite_to_continuum_passed") is not True:
                blockers.add("dilation_finite_to_continuum_not_passed")
        if "mellin_lift" not in payload and "finite_window" not in payload:
            blockers.add("mellin_or_finite_window_receipt_missing")
        elif not (
            _valid_mellin_lift(payload.get("mellin_lift"))
            or _valid_finite_window(payload.get("finite_window"))
        ):
            blockers.add("mellin_or_finite_window_receipt_invalid")
    elif branch == "RADIAL_TOMOGRAPHY":
        blockers.update(_tomography_blockers(payload.get("radial_tomography")))
    return blockers


def _valid_bounded_evidence(
    value: Any,
    *,
    required_true: Sequence[str],
) -> bool:
    if not isinstance(value, Mapping):
        return False
    residual = value.get("max_residual")
    tolerance = value.get("tolerance")
    return bool(
        all(value.get(key) is True for key in required_true)
        and value.get("passed") is True
        and _is_nonnegative_number(residual)
        and _is_nonnegative_number(tolerance)
        and float(residual) <= float(tolerance)
    )


def _valid_mellin_lift(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    ell_min = value.get("ell_min")
    ell_max = value.get("ell_max")
    strip = value.get("convergence_strip")
    residual = value.get("max_arithmetic_residual")
    tolerance = value.get("tolerance")
    return bool(
        _is_positive_integer(ell_min)
        and int(ell_min) >= 2
        and _is_positive_integer(ell_max)
        and int(ell_max) >= int(ell_min)
        and isinstance(strip, (list, tuple))
        and len(strip) == 2
        and all(_is_number(item) for item in strip)
        and float(strip[0]) <= -2.0
        and float(strip[1]) >= 4.0
        and _is_positive_number(value.get("A_zeta_over_A_q"))
        and _is_nonnegative_number(residual)
        and _is_nonnegative_number(tolerance)
        and float(residual) <= float(tolerance)
        and value.get("passed") is True
    )


def _valid_finite_window(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    eta = value.get("eta_by_ell")
    bounds = value.get("absolute_bound_by_ell")
    error = value.get("quadrature_relative_error")
    tolerance = value.get("tolerance")
    return bool(
        _is_positive_number(value.get("mean_radius"))
        and _is_nonnegative_number(value.get("variance"))
        and isinstance(eta, Mapping)
        and bool(eta)
        and all(_is_nonnegative_number(item) for item in eta.values())
        and isinstance(bounds, Mapping)
        and bool(bounds)
        and all(_is_nonnegative_number(item) for item in bounds.values())
        and _is_nonnegative_number(error)
        and _is_nonnegative_number(tolerance)
        and float(error) <= float(tolerance)
        and value.get("bound_verified") is True
        and value.get("passed") is True
    )


def _tomography_blockers(value: Any) -> set[str]:
    if not isinstance(value, Mapping):
        return {"radial_tomography_missing"}
    blockers: set[str] = set()
    if not _is_sha256(value.get("cross_covariance_hash")):
        blockers.add("radial_tomography_cross_covariance_missing")
    for key in (
        "positive_multiplication_spectrum",
        "refinement_converged",
        "held_out_reconstruction_passed",
    ):
        if value.get(key) is not True:
            blockers.add(f"radial_tomography_not_passed:{key}")
    tolerance = value.get("tolerance")
    if not _is_nonnegative_number(tolerance):
        blockers.add("radial_tomography_invalid:tolerance")
    for key in ("hankel_unitarity_residual", "off_diagonal_k_leakage"):
        residual = value.get(key)
        if not _is_nonnegative_number(residual):
            blockers.add(f"radial_tomography_invalid:{key}")
        elif _is_nonnegative_number(tolerance) and float(residual) > float(tolerance):
            blockers.add(f"radial_tomography_exceeds_tolerance:{key}")
    return blockers


def _valid_radial_svd(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    shape = value.get("shape")
    rank = value.get("rank")
    nullity = value.get("nullity")
    singular = value.get("singular_values")
    threshold = value.get("rank_threshold")
    return bool(
        isinstance(shape, (list, tuple))
        and len(shape) == 2
        and all(_is_positive_integer(item) for item in shape)
        and _is_nonnegative_integer(rank)
        and _is_nonnegative_integer(nullity)
        and int(rank) + int(nullity) == int(shape[1])
        and isinstance(singular, (list, tuple))
        and all(_is_nonnegative_number(item) for item in singular)
        and _is_nonnegative_number(threshold)
    )


def _valid_forward_residual(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and all(
            _is_nonnegative_number(value.get(key))
            for key in (
                "absolute_l2_residual",
                "relative_l2_residual",
                "tolerance",
            )
        )
        and isinstance(value.get("passed"), bool)
    )


def _valid_dilation_payload(value: Mapping[str, Any]) -> bool:
    array_keys = (
        "scale_ratios",
        "source_embedding_commutator_norms",
        "screen_covariance_naturality_residual_norms",
        "physical_operator_residual_norms",
        "strong_covariance_cauchy_residuals",
        "strong_dilation_cauchy_residuals",
    )
    if any(
        not isinstance(value.get(key), (list, tuple)) or not value.get(key)
        for key in array_keys
    ):
        return False
    if not all(_is_positive_number(item) for item in value["scale_ratios"]):
        return False
    for key in array_keys[1:]:
        if not all(_is_nonnegative_number(item) for item in value[key]):
            return False
    return bool(
        _is_nonnegative_number(value.get("uniform_covariance_norm_bound"))
        and _is_nonnegative_number(value.get("max_absolute_log_residual"))
        and _is_positive_number(value.get("tolerance"))
        and isinstance(value.get("finite_to_continuum_passed"), bool)
        and isinstance(value.get("passed"), bool)
    )


def _dag_has_blacklisted_ancestor(dag: Mapping[str, Any]) -> bool:
    nodes = dag.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return True
    forbidden = (
        "measurement",
        "fit",
        "likelihood",
        "posterior",
        "planck_shape",
        "target_calibrated_proxy",
        "calibrated_to_data",
        "metadata_unknown",
    )
    for node in nodes:
        if not isinstance(node, Mapping):
            return True
        if any(bool(node.get(key)) for key in forbidden):
            return True
        declared_kinds = {
            str(node.get(key, "")).strip().lower()
            for key in ("kind", "source_kind", "type", "role")
        }
        if declared_kinds.intersection(forbidden):
            return True
    return False


def _dag_has_downstream_transfer_ancestor(dag: Mapping[str, Any]) -> bool:
    nodes = dag.get("nodes")
    if not isinstance(nodes, list):
        return True
    downstream = (
        "transfer_output",
        "boltzmann",
        "camb",
        "class_output",
        "tt_spectrum",
        "te_spectrum",
        "ee_spectrum",
        "lensing",
        "recombination",
        "foreground",
        "nuisance_parameter",
    )
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        if any(bool(node.get(key)) for key in downstream):
            return True
        declared_kinds = {
            str(node.get(key, "")).strip().lower()
            for key in ("kind", "source_kind", "type", "role")
        }
        if declared_kinds.intersection(downstream):
            return True
    return False


def _dag_has_target_ancestor(dag: dict[str, Any]) -> bool:
    """Compatibility alias for the stricter SCR330 ancestry audit."""
    return _dag_has_blacklisted_ancestor(dag)


def _weights(weights: ArrayLike | None, count: int) -> tuple[NDArray[np.float64], float]:
    if weights is None:
        return np.full(count, 1.0 / count), float(count)
    values = np.asarray(weights, dtype=float)
    if values.shape != (count,) or np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise SourceSpectrumInputError("weights must be finite, nonnegative, and match samples")
    total = float(np.sum(values))
    if total <= 0.0:
        raise SourceSpectrumInputError("weights must have positive total")
    return values / total, total


def _finite_positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise SourceSpectrumInputError(f"{name} must be positive")
    return result


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise SourceSpectrumInputError(f"{name} must be finite")
    return result


def _integer_at_least(value: int, minimum: int, name: str) -> int:
    if isinstance(value, bool):
        raise SourceSpectrumInputError(f"{name} must be an integer")
    result = int(value)
    if result != value or result < minimum:
        raise SourceSpectrumInputError(
            f"{name} must be an integer at least {minimum}"
        )
    return result


def _is_number(value: Any) -> bool:
    return bool(
        not isinstance(value, bool)
        and isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
    )


def _is_positive_number(value: Any) -> bool:
    return _is_number(value) and float(value) > 0.0


def _is_nonnegative_number(value: Any) -> bool:
    return _is_number(value) and float(value) >= 0.0


def _is_positive_integer(value: Any) -> bool:
    return bool(
        not isinstance(value, bool)
        and isinstance(value, (int, np.integer))
        and int(value) > 0
    )


def _is_nonnegative_integer(value: Any) -> bool:
    return bool(
        not isinstance(value, bool)
        and isinstance(value, (int, np.integer))
        and int(value) >= 0
    )


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return bool(
        len(digest) == 64
        and digest != "0" * 64
        and all(character in "0123456789abcdef" for character in digest)
    )
