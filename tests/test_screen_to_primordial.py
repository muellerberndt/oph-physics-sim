from __future__ import annotations

import math

import numpy as np

from oph_fpe.cosmology.screen_to_primordial import (
    ThinShellLift,
    A_zeta_from_shell_action_amplitude,
    exact_shell_gamma_lift_receipt,
    exact_thin_shell_cl,
    fractional_screen_cl,
    shell_action_amplitude,
    shell_precision_eigenvalue,
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
    assert receipt["operator_family_match"] == "exact_shell_gamma"
    assert receipt["A_zeta_derived"] > 0.0
    assert receipt["ell_equals_kD_scaffold_only"] is False


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
