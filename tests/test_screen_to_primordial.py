from __future__ import annotations

import math

import numpy as np

from oph_fpe.cosmology.screen_to_primordial import (
    ThinShellLift,
    A_zeta_from_shell_action_amplitude,
    bessel_projection_matrix,
    exact_shell_gamma_lift_receipt,
    exact_thin_shell_cl,
    fractional_screen_cl,
    project_primordial_to_screen,
    screen_to_radial_lift_report,
    shell_action_amplitude,
    shell_precision_eigenvalue,
    source_only_quotient_screen_scalar,
    transfer_firewall_receipt,
)


def test_exact_thin_shell_theta_zero_identity():
    params = ThinShellLift(A_zeta=1.0, theta=0.0, k_pivot=1.0, D_star=1.0, Z_star=1.0)

    for ell in [2, 3, 10, 64]:
        expected = 2.0 * math.pi / (ell * (ell + 1.0))
        assert math.isclose(exact_thin_shell_cl(ell, params), expected, rel_tol=1.0e-12)
        assert math.isclose(shell_precision_eigenvalue(ell, 0.0), ell * (ell + 1.0), rel_tol=1.0e-12)


def test_shell_amplitude_inverse_roundtrip():
    params = ThinShellLift(A_zeta=2.3e-9, theta=0.035, k_pivot=0.05, D_star=13_800.0, Z_star=0.7)
    A_q = shell_action_amplitude(params)

    recovered = A_zeta_from_shell_action_amplitude(
        A_q,
        theta=params.theta,
        k_pivot=params.k_pivot,
        D_star=params.D_star,
        Z_star=params.Z_star,
    )

    assert math.isclose(recovered, params.A_zeta, rel_tol=1.0e-12)


def test_gamma_shell_approaches_fractional_screen_at_high_ell():
    theta = 0.035
    A_q = 1.7
    ell = np.asarray([500.0, 1000.0, 2000.0])
    params = ThinShellLift(
        A_zeta=A_zeta_from_shell_action_amplitude(A_q, theta, 0.05, 13_800.0),
        theta=theta,
        k_pivot=0.05,
        D_star=13_800.0,
    )

    shell = exact_thin_shell_cl(ell, params)
    frac = fractional_screen_cl(ell, A_q, theta)

    assert np.max(np.abs(shell / frac - 1.0)) < 1.0e-5


def test_lift_receipt_derives_a_zeta_only_for_exact_kernel():
    receipt = exact_shell_gamma_lift_receipt(
        A_q_shell=1.2,
        theta=0.02,
        k_pivot=0.05,
        D_star=13_800.0,
        W_star_hash="sha256:window",
        Z_star=1.0,
        Z_star_source="finite_freezeout",
        D_star_source="finite_distance_certificate",
        bessel_kernel_hash="sha256:bessel",
        kernel_rank=32,
        kernel_condition_number=4.0,
        nullspace_dimension=0,
        radial_prior="thin_shell_power_law",
        forward_projection_residual=1.0e-10,
        A_q_source="scalar_release_energy",
    )

    assert receipt["passed"] is True
    assert receipt["SCREEN_TO_RADIAL_LIFT_RECEIPT"] is True
    assert receipt["operator_family_match"] == "exact_shell_gamma"
    assert receipt["A_zeta_derived"] > 0.0
    assert receipt["ell_equals_kD_scaffold_only"] is False


def test_source_only_screen_scalar_removes_background_and_dipole():
    axes = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    values = 3.0 + axes @ np.asarray([0.5, -0.25, 0.75]) + np.asarray([0.2, 0.2, -0.1, -0.1, 0.0, 0.0])

    report = source_only_quotient_screen_scalar(values, axes)

    assert report["field_type"] == "SCREEN_CURVATURE_CANDIDATE"
    assert report["source_only"] is True
    assert report["background_removed"] is True
    assert report["dipole_removed"] is True
    assert abs(np.mean(report["values"])) < 1.0e-12
    assert np.linalg.norm(np.asarray(report["values"]) @ axes) < 1.0e-12


def test_exact_bessel_projection_round_trips_declared_prior():
    ell = np.arange(2, 10, dtype=float)
    k = np.geomspace(1.0e-4, 1.0e-1, 48)
    prior = 2.0e-9 * (k / 0.05) ** -0.035
    screen = project_primordial_to_screen(k, prior, ell, radius=13_800.0)

    report = screen_to_radial_lift_report(
        ell=ell,
        screen_cl=screen,
        k=k,
        radial_prior_delta_zeta2=prior,
        radius=13_800.0,
        radial_prior_declared=True,
        source_only_screen_scalar=True,
        theorem_gate=True,
        source_stress_eigenclock_receipt=True,
        time_orientation_holonomy_receipt=True,
        uniform_density_slice_receipt=True,
        finite_geometric_volume_receipt=True,
        delta_n_curvature_receipt=True,
        total_stress_closure_receipt=True,
        total_energy_frame_receipt=True,
        single_clock_normal_form_receipt=True,
        entropy_repair_gap_receipt=True,
        curvature_freezeout_receipt=True,
        freeze_limit_receipt=True,
        scalar_rg_naturality_receipt=True,
        scalar_eigenvalue_isolation_receipt=True,
        conformal_intertwiner_receipt=True,
        spatial_curvature_branch_receipt=True,
        adiabatic_mode_receipt=True,
        isocurvature_bound_receipt=True,
        primordial_phase_coherence_receipt=True,
        no_observation_ancestry_receipt=True,
        residual_tolerance=1.0e-12,
    )

    assert report["SCREEN_TO_RADIAL_LIFT_RECEIPT"] is True
    assert report["SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT"] is True
    assert report["PRIMORDIAL_SPECTRUM_SOURCE_ONLY_RECEIPT"] is True
    assert report["TOTAL_STRESS_CLOSURE_RECEIPT"] is True
    assert report["SCALAR_EIGENVALUE_ISOLATION_RECEIPT"] is True
    assert report["SPATIAL_CURVATURE_BRANCH_RECEIPT"] is True
    assert report["forward_projection_residual"]["l2_relative"] < 1.0e-12
    assert report["source_only_primordial_parameters"]["A_s"] > 0.0
    assert abs(report["source_only_primordial_parameters"]["n_s"] - 0.965) < 1.0e-12


def test_radial_null_space_injection_is_projection_invisible():
    ell = np.arange(2, 6, dtype=float)
    k = np.geomspace(1.0e-4, 2.0e-1, 12)
    matrix = bessel_projection_matrix(ell, k, radius=13_800.0)
    _, _, vh = np.linalg.svd(matrix)
    null_vector = vh[-1]
    assert np.linalg.norm(matrix @ null_vector) < 1.0e-14

    base = np.full(k.shape, 1.0e-9)
    perturbed = base + 1.0e-11 * null_vector

    assert np.allclose(matrix @ base, matrix @ perturbed, atol=1.0e-20, rtol=1.0e-10)


def test_transfer_firewall_requires_all_physical_gates():
    receipt = transfer_firewall_receipt(
        screen_spectrum_derived=True,
        primordial_lift_derived=True,
        rho_A_finite_derived=True,
        B_A_k_a_finite_derived=True,
        Gamma_rec_k_a_finite_derived=True,
        cdm_limit_regression_passed=True,
        recombination_inputs_ready=False,
        official_likelihood_ready=True,
    )

    assert receipt["passed"] is False
    assert receipt["physical_cmb_prediction"] is False


def test_screen_to_radial_lift_reports_round1_block_status_names():
    ell = np.arange(2, 6, dtype=float)
    k = np.geomspace(1.0e-4, 1.0e-1, 8)
    prior = np.full_like(k, 1.0e-9)
    screen = project_primordial_to_screen(k, prior, ell, radius=13_800.0)

    report = screen_to_radial_lift_report(
        ell=ell,
        screen_cl=screen,
        k=k,
        radial_prior_delta_zeta2=prior,
        radius=13_800.0,
        radial_prior_declared=False,
        source_only_screen_scalar=False,
        theorem_gate=False,
    )

    assert report["SCREEN_TO_RADIAL_LIFT_RECEIPT"] is False
    assert report["PRIMORDIAL_SPECTRUM_SOURCE_ONLY_RECEIPT"] is False
    assert report["RADIAL_NULL_SPACE_REPORT_RECEIPT"] is True
    assert report["UNRESTRICTED_RADIAL_NULL_SPACE"] == "INFINITE_DIMENSIONAL"
    assert "SOURCE_CLOCK_UNPROVEN" in report["hard_block_statuses"]
    assert "SCALAR_RG_EIGENMODE_DEGENERATE" in report["hard_block_statuses"]
    assert "RADIAL_PRIOR_UNDECLARED" in report["hard_block_statuses"]
    assert "PRIMORDIAL_PROMOTION_BLOCKED" in report["hard_block_statuses"]
