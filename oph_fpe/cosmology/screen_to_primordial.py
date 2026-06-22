from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.special import gammaln, spherical_jn


PRIMORDIAL_RECEIPT_ALIASES = {
    "SOURCE_STRESS_EIGENCLOCK_RECEIPT": "source_stress_eigenclock_receipt",
    "TIME_ORIENTATION_HOLONOMY_RECEIPT": "time_orientation_holonomy_receipt",
    "UNIFORM_DENSITY_SLICE_RECEIPT": "uniform_density_slice_receipt",
    "FINITE_GEOMETRIC_VOLUME_RECEIPT": "finite_geometric_volume_receipt",
    "DELTA_N_CURVATURE_RECEIPT": "delta_n_curvature_receipt",
    "TOTAL_STRESS_CLOSURE_RECEIPT": "total_stress_closure_receipt",
    "TOTAL_ENERGY_FRAME_RECEIPT": "total_energy_frame_receipt",
    "SINGLE_CLOCK_NORMAL_FORM_RECEIPT": "single_clock_normal_form_receipt",
    "ENTROPY_REPAIR_GAP_RECEIPT": "entropy_repair_gap_receipt",
    "CURVATURE_EVOLUTION_RECEIPT": "curvature_evolution_receipt",
    "FREEZE_LIMIT_RECEIPT": "freeze_limit_receipt",
    "ADIABATIC_MODE_RECEIPT": "adiabatic_mode_receipt",
    "ISOCURVATURE_BOUND_REPORT": "isocurvature_bound_receipt",
    "PRIMORDIAL_PHASE_COHERENCE_RECEIPT": "primordial_phase_coherence_receipt",
    "SCALAR_RG_NATURALITY_RECEIPT": "scalar_rg_naturality_receipt",
    "SCALAR_EIGENVALUE_ISOLATION_RECEIPT": "scalar_eigenvalue_isolation_receipt",
    "CONFORMAL_INTERTWINER_RECEIPT": "conformal_intertwiner_receipt",
    "SPATIAL_CURVATURE_BRANCH_RECEIPT": "spatial_curvature_branch_receipt",
    "RADIAL_PRIOR_DECLARATION_RECEIPT": "radial_prior_declaration_receipt",
    "RADIAL_NULL_SPACE_REPORT_RECEIPT": "radial_null_space_report_receipt",
    "EXACT_BESSEL_FORWARD_RECEIPT": "exact_bessel_forward_receipt",
    "FORWARD_PROJECTION_RESIDUAL_RECEIPT": "forward_projection_residual_receipt",
    "PRIMORDIAL_AMPLITUDE_RECEIPT": "primordial_amplitude_receipt",
    "PRIMORDIAL_SPECTRUM_SOURCE_ONLY_RECEIPT": "primordial_spectrum_source_only_receipt",
}

PRIMORDIAL_BLOCKER_STATUS = {
    "source_stress_eigenclock_receipt": ("SOURCE_CLOCK_UNPROVEN",),
    "uniform_density_slice_receipt": ("UNIFORM_DENSITY_SLICE_UNPROVEN",),
    "finite_geometric_volume_receipt": ("GEOMETRIC_VOLUME_READOUT_UNPROVEN",),
    "freeze_limit_receipt": ("FREEZE_LIMIT_UNPROVEN",),
    "scalar_rg_naturality_receipt": ("SCALAR_RG_MAP_UNDEFINED",),
    "scalar_eigenvalue_isolation_receipt": ("SCALAR_RG_EIGENMODE_DEGENERATE", "SCALAR_RG_GAP_UNPROVEN"),
    "conformal_intertwiner_receipt": ("CONFORMAL_NATURALITY_UNPROVEN",),
    "spatial_curvature_branch_receipt": ("SPATIAL_CURVATURE_UNRESOLVED",),
    "radial_prior_declaration_receipt": ("RADIAL_PRIOR_UNDECLARED",),
    "forward_projection_residual_receipt": ("RADIAL_FORWARD_RESIDUAL_FAILED",),
    "adiabatic_mode_receipt": ("ADIABATIC_RANK_ONE_FAILED",),
}


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


def source_only_quotient_screen_scalar(
    values: np.ndarray | list[float],
    axes: np.ndarray | list[list[float]] | None = None,
    *,
    weights: np.ndarray | list[float] | None = None,
) -> dict[str, Any]:
    """Remove the screen monopole and dipole from a quotient-visible scalar.

    The returned field is a source-side screen scalar candidate. It is not a
    primordial curvature field unless the later source-stress, single-clock,
    repair-gap, mode-purity, radial-lift, and residual receipts pass.
    """

    field = _as_1d(values, "values")
    if field.size == 0:
        raise ValueError("values must be non-empty")
    sample_weights = _weights_like(weights, field.size)
    design_columns = [np.ones(field.size, dtype=float)]
    valid_axes = axes is not None
    axis_matrix = None
    if axes is not None:
        axis_matrix = np.asarray(axes, dtype=float)
        valid_axes = bool(axis_matrix.shape == (field.size, 3) and np.all(np.isfinite(axis_matrix)))
        if valid_axes:
            norms = np.linalg.norm(axis_matrix, axis=1)
            valid_axes = bool(np.all(norms > 1.0e-12))
            if valid_axes:
                axis_matrix = axis_matrix / norms[:, None]
                design_columns.extend([axis_matrix[:, index] for index in range(3)])
    design = np.column_stack(design_columns)
    sqrt_w = np.sqrt(sample_weights)
    coeff, *_ = np.linalg.lstsq(design * sqrt_w[:, None], field * sqrt_w, rcond=None)
    fitted = design @ coeff
    residual = field - fitted
    centered = field - _weighted_mean(field, sample_weights)
    centered_variance = _weighted_mean(centered * centered, sample_weights)
    residual_variance = _weighted_mean(residual * residual, sample_weights)
    return {
        "field_type": "SCREEN_CURVATURE_CANDIDATE",
        "source_only": True,
        "sample_count": int(field.size),
        "background_removed": True,
        "dipole_removed": bool(valid_axes),
        "monopole": float(coeff[0]),
        "dipole_vector": [float(value) for value in coeff[1:4]] if valid_axes else None,
        "centered_variance": float(centered_variance),
        "monopole_dipole_residual_variance": float(residual_variance),
        "dipole_removed_fraction": float(
            np.clip(1.0 - residual_variance / max(centered_variance, 1.0e-300), -1.0e6, 1.0)
        ),
        "values": [float(value) for value in residual],
        "claim_boundary": (
            "Low-mode-removed quotient-visible screen scalar. This is source-only screen instrumentation, "
            "not a primordial spectrum or CMB prediction."
        ),
    }


def shell_precision_eigenvalue(ell: float | np.ndarray, theta: float) -> float | np.ndarray:
    """Exact thin-shell gamma-ratio precision eigenvalue.

    For theta=0 this reduces to ell*(ell+1). The expression is evaluated in
    log-gamma form so moderately large multipoles remain stable.
    """

    ell_arr = np.asarray(ell, dtype=float)
    values = np.exp(gammaln(ell_arr + 2.0 + 0.5 * float(theta)) - gammaln(ell_arr - 0.5 * float(theta)))
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


def bessel_window_response(
    ell: np.ndarray | list[float],
    k: np.ndarray | list[float],
    *,
    radius: float | None = None,
    radial_nodes: np.ndarray | list[float] | None = None,
    radial_weights: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Return Psi_l(k) for a thin shell or declared radial window.

    For a thin shell, ``Psi_l(k)=j_l(k R)``. For a finite window, the caller
    supplies quadrature nodes and weights for ``int dr W(r) j_l(k r)``.
    """

    ell_arr = _as_1d(ell, "ell")
    k_arr = _positive_1d(k, "k")
    if radius is not None:
        r_value = float(radius)
        if not np.isfinite(r_value) or r_value <= 0.0:
            raise ValueError("radius must be positive")
        return spherical_jn(ell_arr[:, None].astype(int), k_arr[None, :] * r_value)
    if radial_nodes is None or radial_weights is None:
        raise ValueError("provide radius or radial_nodes/radial_weights")
    nodes = _positive_1d(radial_nodes, "radial_nodes")
    weights = _as_1d(radial_weights, "radial_weights")
    if nodes.size != weights.size:
        raise ValueError("radial_nodes and radial_weights must have the same length")
    if not np.all(np.isfinite(weights)):
        raise ValueError("radial_weights must be finite")
    response = np.zeros((ell_arr.size, k_arr.size), dtype=float)
    for index, l_value in enumerate(ell_arr.astype(int)):
        response[index, :] = np.sum(
            weights[:, None] * spherical_jn(l_value, k_arr[None, :] * nodes[:, None]),
            axis=0,
        )
    return response


def bessel_projection_matrix(
    ell: np.ndarray | list[float],
    k: np.ndarray | list[float],
    *,
    radius: float | None = None,
    radial_nodes: np.ndarray | list[float] | None = None,
    radial_weights: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Return the finite quadrature matrix for C_l = K_lj Delta_zeta^2(k_j)."""

    k_arr = _positive_1d(k, "k")
    response = bessel_window_response(
        ell,
        k_arr,
        radius=radius,
        radial_nodes=radial_nodes,
        radial_weights=radial_weights,
    )
    return 4.0 * math.pi * np.square(response) * _log_trapezoid_weights(k_arr)[None, :]


def project_primordial_to_screen(
    k: np.ndarray | list[float],
    delta_zeta2: np.ndarray | list[float],
    ell: np.ndarray | list[float],
    *,
    radius: float | None = None,
    radial_nodes: np.ndarray | list[float] | None = None,
    radial_weights: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Project a declared radial primordial spectrum to screen C_l."""

    spectrum = _as_1d(delta_zeta2, "delta_zeta2")
    matrix = bessel_projection_matrix(
        ell,
        k,
        radius=radius,
        radial_nodes=radial_nodes,
        radial_weights=radial_weights,
    )
    if spectrum.size != matrix.shape[1]:
        raise ValueError("delta_zeta2 length must match k length")
    if np.any(~np.isfinite(spectrum)):
        raise ValueError("delta_zeta2 must be finite")
    return matrix @ spectrum


def screen_to_radial_lift_report(
    *,
    ell: np.ndarray | list[float],
    screen_cl: np.ndarray | list[float],
    k: np.ndarray | list[float],
    radial_prior_delta_zeta2: np.ndarray | list[float] | None = None,
    radius: float | None = None,
    radial_nodes: np.ndarray | list[float] | None = None,
    radial_weights: np.ndarray | list[float] | None = None,
    radial_prior_declared: bool = False,
    source_only_screen_scalar: bool = False,
    theorem_gate: bool = False,
    source_stress_eigenclock_receipt: bool = False,
    time_orientation_holonomy_receipt: bool = False,
    uniform_density_slice_receipt: bool = False,
    finite_geometric_volume_receipt: bool = False,
    delta_n_curvature_receipt: bool = False,
    total_stress_closure_receipt: bool = False,
    total_energy_frame_receipt: bool = False,
    single_clock_normal_form_receipt: bool = False,
    entropy_repair_gap_receipt: bool = False,
    curvature_freezeout_receipt: bool = False,
    freeze_limit_receipt: bool = False,
    scalar_rg_naturality_receipt: bool = False,
    scalar_eigenvalue_isolation_receipt: bool = False,
    conformal_intertwiner_receipt: bool = False,
    spatial_curvature_branch_receipt: bool = False,
    adiabatic_mode_receipt: bool = False,
    isocurvature_bound_receipt: bool = False,
    primordial_phase_coherence_receipt: bool = False,
    no_observation_ancestry_receipt: bool = False,
    residual_tolerance: float = 1.0e-6,
    rank_tolerance: float | None = None,
) -> dict[str, Any]:
    """Emit the exact Bessel lift artifact and hard promotion gate.

    This function computes the forward projection residual and linear
    non-identifiability of a declared radial prior. It keeps the receipt false
    unless the paper theorem gate and all source-side simulator gates are true.
    """

    ell_arr = _as_1d(ell, "ell")
    screen = _as_1d(screen_cl, "screen_cl")
    if ell_arr.size != screen.size:
        raise ValueError("ell and screen_cl must have the same length")
    k_arr = _positive_1d(k, "k")
    matrix = bessel_projection_matrix(
        ell_arr,
        k_arr,
        radius=radius,
        radial_nodes=radial_nodes,
        radial_weights=radial_weights,
    )
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    tol = (
        float(rank_tolerance)
        if rank_tolerance is not None
        else max(matrix.shape) * np.finfo(float).eps * (float(singular_values[0]) if singular_values.size else 1.0)
    )
    effective_rank = int(np.sum(singular_values > tol))
    nullspace_dimension = int(max(0, matrix.shape[1] - effective_rank))
    projected = None
    residual_by_ell = None
    residual_l2 = None
    residual_linf = None
    positivity = None
    prior_row_count = 0
    prior_values = None
    if radial_prior_delta_zeta2 is not None:
        prior_values = _as_1d(radial_prior_delta_zeta2, "radial_prior_delta_zeta2")
        if prior_values.size != matrix.shape[1]:
            raise ValueError("radial_prior_delta_zeta2 length must match k length")
        projected = matrix @ prior_values
        residual = screen - projected
        norm = max(float(np.linalg.norm(screen)), 1.0e-300)
        residual_l2 = float(np.linalg.norm(residual) / norm)
        residual_linf = float(np.max(np.abs(residual)) / max(float(np.max(np.abs(screen))), 1.0e-300))
        residual_by_ell = [
            {
                "ell": float(ell_arr[index]),
                "screen_C_ell": float(screen[index]),
                "projected_C_ell": float(projected[index]),
                "residual": float(residual[index]),
            }
            for index in range(ell_arr.size)
        ]
        positivity = bool(np.all(prior_values >= 0.0))
        prior_row_count = int(prior_values.size)

    radial_prior_receipt = bool(radial_prior_declared)
    forward_residual_receipt = bool(residual_l2 is not None and residual_l2 <= float(residual_tolerance))
    radial_null_space_receipt = True
    exact_bessel_forward_receipt = bool(forward_residual_receipt and (radius is not None or radial_nodes is not None))
    primordial_amplitude_receipt = bool(radial_prior_receipt and positivity and prior_values is not None)
    curvature_evolution_receipt = bool(curvature_freezeout_receipt)
    gates = {
        "theorem_gate": bool(theorem_gate),
        "source_only_screen_scalar": bool(source_only_screen_scalar),
        "source_stress_eigenclock_receipt": bool(source_stress_eigenclock_receipt),
        "time_orientation_holonomy_receipt": bool(time_orientation_holonomy_receipt),
        "uniform_density_slice_receipt": bool(uniform_density_slice_receipt),
        "finite_geometric_volume_receipt": bool(finite_geometric_volume_receipt),
        "delta_n_curvature_receipt": bool(delta_n_curvature_receipt),
        "total_stress_closure_receipt": bool(total_stress_closure_receipt),
        "total_energy_frame_receipt": bool(total_energy_frame_receipt),
        "single_clock_normal_form_receipt": bool(single_clock_normal_form_receipt),
        "entropy_repair_gap_receipt": bool(entropy_repair_gap_receipt),
        "curvature_evolution_receipt": curvature_evolution_receipt,
        "curvature_freezeout_receipt": bool(curvature_freezeout_receipt),
        "freeze_limit_receipt": bool(freeze_limit_receipt),
        "scalar_rg_naturality_receipt": bool(scalar_rg_naturality_receipt),
        "scalar_eigenvalue_isolation_receipt": bool(scalar_eigenvalue_isolation_receipt),
        "conformal_intertwiner_receipt": bool(conformal_intertwiner_receipt),
        "spatial_curvature_branch_receipt": bool(spatial_curvature_branch_receipt),
        "adiabatic_mode_receipt": bool(adiabatic_mode_receipt),
        "isocurvature_bound_receipt": bool(isocurvature_bound_receipt),
        "primordial_phase_coherence_receipt": bool(primordial_phase_coherence_receipt),
        "radial_prior_declaration_receipt": radial_prior_receipt,
        "radial_null_space_report_receipt": radial_null_space_receipt,
        "exact_bessel_forward_receipt": exact_bessel_forward_receipt,
        "no_observation_ancestry_receipt": bool(no_observation_ancestry_receipt),
        "positive_radial_prior": bool(positivity) if positivity is not None else False,
        "forward_projection_residual_receipt": forward_residual_receipt,
        "primordial_amplitude_receipt": primordial_amplitude_receipt,
    }
    gates["primordial_spectrum_source_only_receipt"] = bool(all(gates.values()))
    blockers = [name for name, passed in gates.items() if not passed]
    receipt = not blockers
    receipt_aliases = {
        receipt_name: bool(gates[gate_name])
        for receipt_name, gate_name in PRIMORDIAL_RECEIPT_ALIASES.items()
    }
    hard_block_statuses = [
        status
        for gate_name, statuses in PRIMORDIAL_BLOCKER_STATUS.items()
        if not bool(gates.get(gate_name, False))
        for status in statuses
    ]
    if blockers:
        hard_block_statuses.append("PRIMORDIAL_PROMOTION_BLOCKED")
    primordial_parameters = _primordial_parameter_metadata(
        k=k_arr,
        delta_zeta2=prior_values,
        enabled=receipt,
    )
    return {
        "mode": "screen_to_radial_lift_artifact_v0",
        "receipt": "SCREEN_TO_RADIAL_LIFT_RECEIPT",
        "SCREEN_TO_RADIAL_LIFT_RECEIPT": receipt,
        "SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT": receipt,
        "SOURCE_ONLY_PRIMORDIAL_SPECTRUM_RECEIPT": receipt,
        **receipt_aliases,
        "field_status": "PRIMORDIAL_CURVATURE" if receipt else "CONDITIONAL_RADIAL_CONTINUATION",
        "primordial_promotion_status": "RECEIPT_PASSED" if receipt else "RECEIPT_GATED",
        "projection": {
            "formula": "C_ell = 4 pi int dlnk Delta_zeta^2(k) |Psi_ell(k)|^2",
            "window": "thin_shell" if radius is not None else "radial_window",
            "radius": float(radius) if radius is not None else None,
            "ell_min": float(np.min(ell_arr)),
            "ell_max": float(np.max(ell_arr)),
            "ell_count": int(ell_arr.size),
            "k_min": float(np.min(k_arr)),
            "k_max": float(np.max(k_arr)),
            "k_count": int(matrix.shape[1]),
        },
        "radial_prior": {
            "declared": bool(radial_prior_declared),
            "row_count": prior_row_count,
            "positive": positivity,
            "status": "declared_source_prior" if radial_prior_declared else "diagnostic_or_missing_prior",
        },
        "radial_null_space": {
            "singular_values": [float(value) for value in singular_values],
            "effective_rank": effective_rank,
            "rank_tolerance": float(tol),
            "nullspace_dimension": nullspace_dimension,
            "analytic_statement": (
                "A finite screen C_l vector constrains only the row space of the Bessel projection. "
                "Any component in the radial null space is invisible without an additional source prior "
                "or nested-shell covariance."
            ),
        },
        "forward_projection_residual": {
            "available": residual_l2 is not None,
            "l2_relative": residual_l2,
            "linf_relative": residual_linf,
            "tolerance": float(residual_tolerance),
            "passed": bool(residual_l2 is not None and residual_l2 <= float(residual_tolerance)),
            "rows": residual_by_ell or [],
        },
        "gates": gates,
        "blockers": blockers,
        "hard_block_statuses": hard_block_statuses,
        "UNRESTRICTED_RADIAL_NULL_SPACE": "INFINITE_DIMENSIONAL",
        "source_only_primordial_parameters": primordial_parameters,
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "Exact Bessel screen-to-radial lift artifact. It reports the declared projection kernel, "
            "radial null space, and forward residual. A_s, n_s, running, isocurvature, phase coherence, "
            "and TT/TE/EE predictions stay gated until every source and lift receipt passes."
        ),
    }


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
        "SCREEN_TO_RADIAL_LIFT_RECEIPT": passed,
        "SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT": passed,
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
    source_provenance_receipt: bool = False,
    pooled_source_reducer_receipt: bool = False,
) -> dict[str, Any]:
    gates = {
        "screen_spectrum_derived": bool(screen_spectrum_derived),
        "primordial_lift_derived": bool(primordial_lift_derived),
        "source_provenance_receipt": bool(source_provenance_receipt),
        "pooled_source_reducer_receipt": bool(pooled_source_reducer_receipt),
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
            "TT/TE/EE promotion is the conjunction of the primordial lift, source-provenance, "
            "pooled-reducer, finite source kernels, CDM-limit, recombination, and official-likelihood gates."
        ),
    }


def _finite_positive(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)) and float(value) > 0.0)
    except (TypeError, ValueError):
        return False


def _as_1d(values: np.ndarray | list[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _positive_1d(values: np.ndarray | list[float], name: str) -> np.ndarray:
    arr = _as_1d(values, name)
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be positive")
    return arr


def _weights_like(values: np.ndarray | list[float] | None, size: int) -> np.ndarray:
    if values is None:
        return np.ones(size, dtype=float)
    weights = _as_1d(values, "weights")
    if weights.size != size:
        raise ValueError("weights length must match values length")
    if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        raise ValueError("weights must be non-negative with positive total")
    return weights


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * values) / np.sum(weights))


def _log_trapezoid_weights(k: np.ndarray) -> np.ndarray:
    x = np.log(k)
    if x.size == 1:
        return np.ones(1, dtype=float)
    weights = np.zeros_like(x)
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    if np.any(weights <= 0.0):
        raise ValueError("k must be strictly increasing")
    return weights


def _primordial_parameter_metadata(
    *,
    k: np.ndarray,
    delta_zeta2: np.ndarray | None,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled or delta_zeta2 is None:
        return {
            "A_s": None,
            "n_s": None,
            "alpha_s": None,
            "pivot_k": None,
            "status": "withheld_until_SCREEN_TO_RADIAL_LIFT_RECEIPT",
        }
    if np.any(delta_zeta2 <= 0.0):
        return {
            "A_s": None,
            "n_s": None,
            "alpha_s": None,
            "pivot_k": None,
            "status": "withheld_nonpositive_radial_prior",
        }
    x = np.log(k)
    y = np.log(delta_zeta2)
    pivot = float(k[len(k) // 2])
    xp = math.log(pivot)
    degree = min(2, max(1, k.size - 1))
    coeff = np.polyfit(x - xp, y, degree)
    if degree == 1:
        slope = float(coeff[0])
        intercept = float(coeff[1])
        running = 0.0
    else:
        slope = float(coeff[1])
        intercept = float(coeff[2])
        running = float(2.0 * coeff[0])
    return {
        "A_s": float(math.exp(intercept)),
        "n_s": float(1.0 + slope),
        "alpha_s": running,
        "pivot_k": pivot,
        "status": "source_only_after_all_lift_gates",
    }
