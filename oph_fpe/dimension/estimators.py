"""Dimension estimators: heat-kernel spectral dimension and Weyl-law fit.

Exploratory, non-evidential.  Every pin in this module is declared in
DESIGN.md sections 4 and 7 before any run: sigma grid, window rule, probe
and Lanczos counts, Weyl eigencount and rank floor, dense cap, seeds, and
the zero-mode cut.  The estimator code is the same for calibration graphs
and tower configurations.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigh_tridiagonal

SIGMA_GRID = np.logspace(-3.0, 4.0, 71)
DENSE_CAP = 2000
HUTCHINSON_PROBES = 64
LANCZOS_STEPS = 120
WEYL_EIGENCOUNT = 200
WEYL_RANK_FLOOR = 8
WEYL_SHIFT = -1.0e-6
ZERO_MODE_CUT = 1.0e-9
SIGMA_LO = 4.0
WINDOW_MIN_POINTS = 5
HUTCHINSON_SEED = 20260820
EIGSH_SEED = 20260821


def dense_spectrum(matrix: sp.spmatrix) -> np.ndarray:
    """Full ascending spectrum of a dense-cap operator, clipped at zero."""

    dense = np.asarray(matrix.todense(), dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(dense)
    return np.clip(eigenvalues, 0.0, None)


def return_probability_from_spectrum(
    eigenvalues: np.ndarray, sigmas: np.ndarray
) -> np.ndarray:
    """Exact ``P(sigma) = (1/N) sum exp(-sigma lambda)``."""

    exponents = -np.outer(sigmas, eigenvalues)
    return np.exp(exponents).sum(axis=1) / float(eigenvalues.size)


def _lanczos_exp_curve(
    matrix: sp.csr_matrix,
    probe: np.ndarray,
    kernel_basis: np.ndarray | None,
    steps: int,
    sigmas: np.ndarray,
) -> np.ndarray:
    """One probe's curve ``z^T exp(-sigma L) z`` by Lanczos quadrature.

    Full reorthogonalization at every step, plus reorthogonalization against
    the exact kernel basis (the probe is deflated before entry; the
    per-step projection controls floating-point drift back into the
    kernel).  Ritz values are clipped at zero from below.
    """

    node_count = probe.size
    norm_squared = float(probe @ probe)
    if norm_squared == 0.0:
        return np.zeros(sigmas.size, dtype=np.float64)
    vector = probe / np.sqrt(norm_squared)
    basis = np.zeros((node_count, steps), dtype=np.float64)
    alphas: list[float] = []
    betas: list[float] = []
    for step in range(steps):
        basis[:, step] = vector
        work = matrix @ vector
        alpha = float(vector @ work)
        alphas.append(alpha)
        work = work - alpha * vector
        if step > 0:
            work = work - betas[-1] * basis[:, step - 1]
        active = basis[:, : step + 1]
        work = work - active @ (active.T @ work)
        if kernel_basis is not None and kernel_basis.size:
            work = work - kernel_basis @ (kernel_basis.T @ work)
        beta = float(np.linalg.norm(work))
        if step == steps - 1 or beta <= 1.0e-13 * max(1.0, abs(alpha)):
            break
        betas.append(beta)
        vector = work / beta
    ritz_values, ritz_vectors = eigh_tridiagonal(
        np.asarray(alphas, dtype=np.float64), np.asarray(betas, dtype=np.float64)
    )
    ritz_values = np.clip(ritz_values, 0.0, None)
    first_components_squared = ritz_vectors[0, :] ** 2
    exponents = np.exp(-np.outer(sigmas, ritz_values))
    return norm_squared * (exponents @ first_components_squared)


def return_probability_stochastic(
    matrix: sp.csr_matrix,
    kernel_basis: np.ndarray,
    sigmas: np.ndarray,
    *,
    probes: int = HUTCHINSON_PROBES,
    steps: int = LANCZOS_STEPS,
    seed: int = HUTCHINSON_SEED,
) -> np.ndarray:
    """Kernel-deflated Hutchinson estimate of ``P(sigma)``.

    Rademacher probes are projected orthogonal to the exact kernel basis;
    the exact kernel contribution ``kernel_dim / N`` is added in closed
    form.  With the deflation, the estimator targets
    ``tr[exp(-sigma L) (I - P_kernel)]`` exactly.
    """

    node_count = matrix.shape[0]
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    kernel_dim = int(kernel_basis.shape[1]) if kernel_basis.size else 0
    accumulator = np.zeros(sigmas.size, dtype=np.float64)
    for _ in range(int(probes)):
        probe = generator.integers(0, 2, size=node_count).astype(np.float64) * 2.0 - 1.0
        if kernel_dim:
            probe = probe - kernel_basis @ (kernel_basis.T @ probe)
        accumulator += _lanczos_exp_curve(matrix, probe, kernel_basis, steps, sigmas)
    trace_estimate = float(kernel_dim) + accumulator / float(probes)
    return trace_estimate / float(node_count)


def spectral_dimension_curve(sigmas: np.ndarray, p_return: np.ndarray) -> np.ndarray:
    """Centered log-log differences; the two endpoints carry ``nan``."""

    log_sigma = np.log(sigmas)
    log_p = np.log(p_return)
    curve = np.full(sigmas.size, np.nan, dtype=np.float64)
    curve[1:-1] = -2.0 * (log_p[2:] - log_p[:-2]) / (log_sigma[2:] - log_sigma[:-2])
    return curve


def window_statistics(
    sigmas: np.ndarray,
    ds_curve: np.ndarray,
    lambda_2: float | None,
) -> dict:
    """Pinned window rule: ``[SIGMA_LO, 1 / lambda_2]``, median statistic."""

    if lambda_2 is None or lambda_2 <= 0.0:
        return {
            "window_degenerate": True,
            "window_point_count": 0,
            "window_sigma_lo": None,
            "window_sigma_hi": None,
            "d_s_median": None,
            "d_s_window_min": None,
            "d_s_window_max": None,
        }
    sigma_hi = 1.0 / float(lambda_2)
    mask = (sigmas >= SIGMA_LO) & (sigmas <= sigma_hi) & np.isfinite(ds_curve)
    count = int(np.count_nonzero(mask))
    values = ds_curve[mask]
    return {
        "window_degenerate": bool(count < WINDOW_MIN_POINTS),
        "window_point_count": count,
        "window_sigma_lo": float(SIGMA_LO),
        "window_sigma_hi": float(sigma_hi),
        "d_s_median": float(np.median(values)) if count else None,
        "d_s_window_min": float(np.min(values)) if count else None,
        "d_s_window_max": float(np.max(values)) if count else None,
    }


def _pinned_start_vector(node_count: int, seed: int = EIGSH_SEED) -> np.ndarray:
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    vector = generator.standard_normal(node_count)
    return vector / np.linalg.norm(vector)


def low_spectrum(
    matrix: sp.csr_matrix,
    *,
    eigencount: int = WEYL_EIGENCOUNT,
    seed: int = EIGSH_SEED,
) -> np.ndarray:
    """The smallest eigenvalues, ascending: dense path under the cap,
    shift-invert ``eigsh`` above it."""

    node_count = matrix.shape[0]
    k = min(int(eigencount), node_count - 2)
    if node_count <= DENSE_CAP:
        return dense_spectrum(matrix)[:k]
    values = spla.eigsh(
        matrix,
        k=k,
        sigma=WEYL_SHIFT,
        which="LM",
        v0=_pinned_start_vector(node_count, seed),
        return_eigenvectors=False,
    )
    return np.clip(np.sort(values), 0.0, None)


def largest_eigenvalue(
    matrix: sp.csr_matrix, *, seed: int = EIGSH_SEED
) -> float:
    """The largest eigenvalue (dense under the cap, ``eigsh`` above)."""

    node_count = matrix.shape[0]
    if node_count <= DENSE_CAP:
        return float(dense_spectrum(matrix)[-1])
    value = spla.eigsh(
        matrix,
        k=1,
        which="LA",
        v0=_pinned_start_vector(node_count, seed),
        return_eigenvectors=False,
    )
    return float(value[0])


def weyl_fit(low_eigenvalues: np.ndarray, expected_kernel_dim: int | None = None) -> dict:
    """Weyl-law dimension from the counting function over the low spectrum.

    Nonzero eigenvalues ranked ``i = 1..M`` define ``N(lambda_i) = i``;
    the least-squares slope of ``ln i`` against ``ln lambda_i`` over ranks
    ``[WEYL_RANK_FLOOR, M]`` gives ``d_weyl = 2 * slope``.
    """

    eigenvalues = np.sort(np.asarray(low_eigenvalues, dtype=np.float64))
    kernel_dim = int(np.count_nonzero(eigenvalues <= ZERO_MODE_CUT))
    nonzero = eigenvalues[eigenvalues > ZERO_MODE_CUT]
    result: dict = {
        "kernel_dim": kernel_dim,
        "lambda_2": float(nonzero[0]) if nonzero.size else None,
        "kernel_dim_matches_components": (
            None if expected_kernel_dim is None else bool(kernel_dim == expected_kernel_dim)
        ),
    }
    if expected_kernel_dim is not None and kernel_dim != expected_kernel_dim:
        raise ValueError(
            "kernel dimension from the low spectrum "
            f"({kernel_dim}) does not match the construction component count "
            f"({expected_kernel_dim})"
        )
    ranks = np.arange(1, nonzero.size + 1, dtype=np.float64)
    usable = ranks >= WEYL_RANK_FLOOR
    if int(np.count_nonzero(usable)) < 2:
        result.update({"d_weyl": None, "weyl_rank_range": None})
        return result
    log_rank = np.log(ranks[usable])
    log_lambda = np.log(nonzero[usable])
    slope = np.polyfit(log_lambda, log_rank, 1)[0]
    result.update(
        {
            "d_weyl": float(2.0 * slope),
            "weyl_rank_range": [int(WEYL_RANK_FLOOR), int(nonzero.size)],
        }
    )
    return result


def measure_dimensions(
    matrix: sp.csr_matrix,
    kernel_basis: np.ndarray,
    *,
    expected_kernel_dim: int | None = None,
    probes: int = HUTCHINSON_PROBES,
    steps: int = LANCZOS_STEPS,
    hutchinson_seed: int = HUTCHINSON_SEED,
    eigsh_seed: int = EIGSH_SEED,
    force_stochastic: bool = False,
) -> dict:
    """Both estimators on one operator, with the pinned path selection.

    Dense path is primary under ``DENSE_CAP`` nodes; the stochastic path is
    primary above it.  ``force_stochastic`` runs the stochastic path on a
    dense-cap operator for the cross-check suite.
    """

    node_count = matrix.shape[0]
    sigmas = SIGMA_GRID
    use_dense = node_count <= DENSE_CAP and not force_stochastic
    if use_dense:
        spectrum = dense_spectrum(matrix)
        p_return = return_probability_from_spectrum(spectrum, sigmas)
        low_eigenvalues = spectrum[: min(WEYL_EIGENCOUNT, node_count - 2)]
        lambda_max = float(spectrum[-1])
        estimator_path = "dense_eigvalsh"
    else:
        p_return = return_probability_stochastic(
            matrix,
            kernel_basis,
            sigmas,
            probes=probes,
            steps=steps,
            seed=hutchinson_seed,
        )
        low_eigenvalues = low_spectrum(matrix, seed=eigsh_seed)
        lambda_max = largest_eigenvalue(matrix, seed=eigsh_seed)
        estimator_path = "hutchinson_lanczos_quadrature"
    weyl = weyl_fit(low_eigenvalues, expected_kernel_dim=expected_kernel_dim)
    ds_curve = spectral_dimension_curve(sigmas, p_return)
    window = window_statistics(sigmas, ds_curve, weyl["lambda_2"])
    return {
        "node_count": int(node_count),
        "estimator_path": estimator_path,
        "probes": int(probes) if not use_dense else None,
        "lanczos_steps": int(steps) if not use_dense else None,
        "hutchinson_seed": int(hutchinson_seed) if not use_dense else None,
        "lambda_max": lambda_max,
        "p_return": p_return,
        "d_s_curve": ds_curve,
        **weyl,
        **window,
    }
