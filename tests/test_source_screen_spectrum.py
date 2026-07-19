from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from oph_fpe.cosmology.source_screen_spectrum import (
    P_STAR,
    PHI,
    RADIAL_RECEIPTS,
    SCR330_SCHEMA_VERSION,
    approximate_dilation_shape_bound,
    build_receipt,
    build_radial_receipt,
    conformal_precision_eigenvalue,
    derivative_mellin_norm,
    dilation_intertwiner_receipt,
    edge_center_tilt,
    finite_window_stability_bound,
    mellin_spherical_bessel_square,
    minimum_prior_continuation,
    primordial_amplitude_from_screen,
    radial_kernel_matrix,
    radial_null_space,
    radial_projection_matrix,
    screen_cl,
    source_powerlaw,
    source_family_forward_residual,
    source_amplitude_from_samples,
    theta_from_step_survival,
    thin_shell_cl,
    window_powerlaw_cl_quadrature,
    write_radial_receipt,
)


SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas/cosmology/source_screen_spectrum_receipt.schema.json"
)


def test_edge_center_tilt_requires_emitted_density() -> None:
    result = edge_center_tilt(full_collar_generator_density=P_STAR / 24.0)
    assert math.isclose(result.theta, P_STAR / 48.0, abs_tol=1.0e-15)
    assert math.isclose(result.kappa_rep_edge, result.theta / (P_STAR - PHI))
    assert not math.isclose(result.kappa_rep_edge, math.e, rel_tol=1.0e-3)


def test_finite_survival_is_distinct_from_generator_density() -> None:
    density = P_STAR / 48.0
    exponent = theta_from_step_survival(1.0 - density, math.e)
    assert not math.isclose(exponent, density, rel_tol=1.0e-4)


def test_gamma_precision_reduces_to_laplacian() -> None:
    ell = np.arange(2, 30, dtype=float)
    np.testing.assert_allclose(
        conformal_precision_eigenvalue(ell, 0.0),
        ell * (ell + 1.0),
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_mellin_identity_and_derivative_norm() -> None:
    for ell in range(2, 10):
        expected = 1.0 / (2.0 * ell * (ell + 1.0))
        assert math.isclose(
            mellin_spherical_bessel_square(ell, 0.0),
            expected,
            rel_tol=2.0e-14,
        )
        assert derivative_mellin_norm(ell, 0.5) > 0.0


def test_source_amplitude_has_gaussian_factor_of_two() -> None:
    q = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = source_amplitude_from_samples(q, np.eye(2), np.diag([2.0, 5.0]))
    assert math.isclose(result.A_q, 2.0 * result.E_src / result.mode_count)


def test_source_pivot_amplitude_conversion() -> None:
    result = primordial_amplitude_from_screen(
        1.0, P_STAR / 48.0, R_star=23.0
    )
    assert result.source_pivot
    assert math.isclose(result.A_zeta, 0.16080676040273595, rel_tol=2.0e-14)


def test_thin_shell_projection_matches_screen_family() -> None:
    theta = P_STAR / 48.0
    amplitude = 0.37
    radius = 1.7
    lift = primordial_amplitude_from_screen(amplitude, theta, R_star=radius)
    ell = np.array([2.0, 3.0, 10.0])
    np.testing.assert_allclose(
        thin_shell_cl(ell, lift.A_zeta, theta, R_star=radius),
        screen_cl(ell, amplitude, theta),
        rtol=2.0e-14,
        atol=0.0,
    )


def test_finite_window_bound_contains_quadrature_difference() -> None:
    ell = 2
    theta = 0.8
    A_zeta = 1.7
    radius = 1.0
    radii = np.array([0.97, 1.0, 1.03])
    radial_weights = np.array([0.25, 0.5, 0.25])
    log_k = np.linspace(-15.0, 15.0, 60_001)
    k = np.exp(log_k)
    dlnk = np.gradient(log_k)
    finite_window = window_powerlaw_cl_quadrature(
        ell,
        A_zeta,
        theta,
        Z_q=1.0,
        k_pivot=1.0,
        k=k,
        dlnk_weights=dlnk,
        radii=radii,
        radial_weights=radial_weights,
    )
    shell = thin_shell_cl(
        ell,
        A_zeta,
        theta,
        R_star=radius,
        k_pivot=1.0,
        Z_q=1.0,
    )
    bound = finite_window_stability_bound(
        ell,
        theta,
        A_zeta=A_zeta,
        Z_q=1.0,
        k_pivot=1.0,
        R_star=radius,
        radii=radii,
        radial_weights=radial_weights,
    )
    assert abs(finite_window - shell) <= bound.absolute_cl_bound + 1.0e-5 * shell


def test_radial_projection_generalizes_thin_shell_wrapper() -> None:
    ell = np.arange(2, 7)
    k = np.geomspace(1.0e-2, 10.0, 12)
    weights = np.gradient(np.log(k))
    general = radial_projection_matrix(
        ell,
        k,
        weights,
        Z_q=1.0,
        radii=[1.0],
        radial_weights=[1.0],
    )
    legacy = radial_kernel_matrix(
        ell,
        k,
        R_star=1.0,
        dlnk_weights=weights,
    )
    np.testing.assert_allclose(general, legacy, rtol=0.0, atol=0.0)


def test_unrestricted_radial_map_reports_null_space() -> None:
    matrix = radial_kernel_matrix(
        np.arange(2, 7), np.geomspace(1.0e-2, 10.0, 12), R_star=1.0
    )
    report = radial_null_space(matrix)
    assert report.nullity >= 7
    assert report.rank_threshold > 0.0


def test_minimum_prior_continuation_exposes_resolution_and_null_projectors() -> None:
    matrix = np.array(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]
    )
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    covariance = matrix @ truth
    result = minimum_prior_continuation(
        matrix,
        covariance,
        prior_center=np.zeros(4),
        prior_precision=np.diag([1.0, 2.0, 3.0, 4.0]),
    )
    continuation = np.asarray(result.p)
    resolution = np.asarray(result.resolution)
    null_projector = np.asarray(result.null_projector)
    np.testing.assert_allclose(matrix @ continuation, covariance, atol=1.0e-12)
    np.testing.assert_allclose(resolution @ resolution, resolution, atol=1.0e-12)
    np.testing.assert_allclose(matrix @ null_projector, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(resolution + null_projector, np.eye(4), atol=1.0e-12)


def test_source_family_forward_residual_does_not_fit() -> None:
    k = np.geomspace(1.0e-2, 10.0, 12)
    matrix = radial_kernel_matrix(np.arange(2, 7), k, R_star=1.0)
    theta = P_STAR / 48.0
    source_power = 0.4 * k ** (-theta)
    screen_values = matrix @ source_power
    report = source_family_forward_residual(
        matrix,
        screen_values,
        theta,
        k,
        k_pivot=1.0,
        A_zeta=0.4,
    )
    assert report["source_family_dimension"] == 1
    assert report["relative_l2_residual"] < 1.0e-14


def test_dilation_intertwiner_accepts_powerlaw_and_rejects_wiggle() -> None:
    k = np.geomspace(1.0e-4, 1.0e4, 5000)
    theta = 0.23
    power = source_powerlaw(k, 2.0, theta, 1.0)
    good = dilation_intertwiner_receipt(
        k,
        power,
        theta,
        scale_ratios=[1.1, 1.7, 2.0],
        tolerance=2.0e-9,
    )
    assert good.passed
    wiggle = power * np.exp(0.03 * np.sin(3.0 * np.log(k)))
    bad = dilation_intertwiner_receipt(
        k,
        wiggle,
        theta,
        scale_ratios=[1.7, 2.0],
        tolerance=1.0e-3,
    )
    assert not bad.passed


def test_approximate_dilation_shape_bound_integrates_log_slope_error() -> None:
    k = np.geomspace(0.1, 10.0, 101)
    bound = approximate_dilation_shape_bound(
        k, np.full_like(k, 0.02), k_pivot=1.0
    )
    np.testing.assert_allclose(
        bound,
        0.02 * np.abs(np.log(k)),
        rtol=2.0e-3,
        atol=2.0e-5,
    )


def test_receipt_fails_closed() -> None:
    receipt = build_receipt(
        receipt="RELEASE_AMPLITUDE",
        claimed_pass=True,
        claim_tier="E4",
        source_dag={"nodes": [{"id": "source"}, {"id": "sky", "measurement": True}]},
    )
    assert receipt["passed"] is False
    assert receipt["schema_version"] == SCR330_SCHEMA_VERSION
    assert receipt["receipt"] == "SCR330_SOURCE_SHELL_EMBEDDING_RECEIPT"
    assert receipt["no_measurement_fit_likelihood_ancestor"] is False
    assert "measurement_fit_or_likelihood_ancestor" in receipt["blockers"]


def test_transfer_claim_requires_E5() -> None:
    receipt = build_receipt(
        receipt="TRANSFER_FIREWALL",
        claimed_pass=True,
        claim_tier="E4",
        source_dag={"nodes": [{"id": "source"}]},
        physical_tt_te_ee_claim=True,
    )
    assert receipt["passed"] is False
    assert "downstream_transfer_requires_E5" in receipt["blockers"]
    assert "tt_te_ee_claim_before_E5" in receipt["blockers"]
    _validate_schema(receipt)


def test_transfer_receipt_can_pass_only_at_E5() -> None:
    digest = "sha256:" + "a" * 64
    receipt = build_radial_receipt(
        receipt="SCR330_TRANSFER_FIREWALL_RECEIPT",
        passed=True,
        claim_tier="E5",
        source_dag={"nodes": [{"id": "source"}]},
        physical_tt_te_ee_claim=True,
        payload={
            "upstream_radial_receipt_hash": digest,
            "transfer_source_hash": digest,
            "solver_assumption_hash": digest,
            "upstream_claim_tier": "E4",
            "no_back_edge_to_E4_source": True,
        },
    )
    assert receipt["passed"] is True
    _validate_schema(receipt)


def test_promotion_missing_payload_and_prior_continuation_fail_closed() -> None:
    clean_dag = {"nodes": [{"id": "source"}, {"id": "mode_embedding"}]}
    missing = build_radial_receipt(
        receipt="SCR330_RADIAL_PROMOTION_RECEIPT",
        passed=True,
        claim_tier="E4",
        source_dag=clean_dag,
    )
    assert missing["passed"] is False
    assert "radial_promotion_payload_missing" in missing["blockers"]

    prior = build_radial_receipt(
        receipt="SCR330_RADIAL_PROMOTION_RECEIPT",
        passed=True,
        claim_tier="E4",
        source_dag=clean_dag,
        payload={"radial_branch": "PRIOR_CONTINUATION"},
    )
    assert prior["passed"] is False
    assert "prior_continuation_is_not_source_derived_E4" in prior["blockers"]
    _validate_schema(prior)


def test_no_scr330_receipt_can_pass_from_a_caller_boolean_without_evidence() -> None:
    for receipt_name in sorted(RADIAL_RECEIPTS):
        claim_tier = "E5" if receipt_name == "SCR330_TRANSFER_FIREWALL_RECEIPT" else "E4"
        report = build_radial_receipt(
            receipt=receipt_name,
            passed=True,
            claim_tier=claim_tier,
            source_dag={"nodes": [{"id": "source"}]},
        )
        assert report["passed"] is False, receipt_name
        assert report["blockers"], receipt_name
        _validate_schema(report)


def test_complete_source_dilation_promotion_validates_v2_schema() -> None:
    receipt = build_radial_receipt(
        receipt="SCR330_RADIAL_PROMOTION_RECEIPT",
        passed=True,
        claim_tier="E4",
        source_dag={"nodes": [{"id": "source"}, {"id": "mode_embedding"}]},
        payload=_source_dilation_payload(),
    )
    assert receipt["passed"] is True
    assert receipt["blockers"] == []
    _validate_schema(receipt)


def test_placeholder_zero_hash_cannot_promote_radial_source() -> None:
    payload = _source_dilation_payload()
    payload["source_embedding_hash"] = "sha256:" + "0" * 64

    receipt = build_radial_receipt(
        receipt="SCR330_RADIAL_PROMOTION_RECEIPT",
        passed=True,
        claim_tier="E4",
        source_dag={"nodes": [{"id": "source"}]},
        payload=payload,
    )

    assert receipt["passed"] is False
    assert "source_embedding_hash_invalid" in receipt["blockers"]


def test_source_dag_must_be_populated_and_transfer_free_at_E4() -> None:
    empty = build_radial_receipt(
        receipt="SCR330_RADIAL_DILATION_INTERTWINER_RECEIPT",
        passed=True,
        claim_tier="E4",
        source_dag={"nodes": []},
    )
    assert empty["passed"] is False
    assert "source_dag_empty_or_invalid" in empty["blockers"]

    downstream = build_radial_receipt(
        receipt="SCR330_RADIAL_DILATION_INTERTWINER_RECEIPT",
        passed=True,
        claim_tier="E4",
        source_dag={"nodes": [{"id": "source"}, {"id": "camb", "camb": True}]},
    )
    assert downstream["passed"] is False
    assert "transfer_or_observable_ancestor_before_E5" in downstream["blockers"]

    measurement_kind = build_radial_receipt(
        receipt="SCR330_RADIAL_NULL_REPORT",
        passed=True,
        claim_tier="E3",
        source_dag={"nodes": [{"id": "target", "kind": "measurement"}]},
    )
    assert measurement_kind["passed"] is False
    assert "measurement_fit_or_likelihood_ancestor" in measurement_kind["blockers"]


def test_receipt_rejects_truthy_nonboolean_pass_flag() -> None:
    receipt = build_radial_receipt(
        receipt="SCR330_RADIAL_DILATION_INTERTWINER_RECEIPT",
        passed="true",  # type: ignore[arg-type]
        claim_tier="E4",
        source_dag={"nodes": [{"id": "source"}]},
    )
    assert receipt["passed"] is False
    assert "passed_flag_not_boolean" in receipt["blockers"]
    _validate_schema(receipt)


def test_write_canonical_radial_receipt(tmp_path: Path) -> None:
    digest = "sha256:" + "a" * 64
    result = write_radial_receipt(
        tmp_path,
        receipt="SCR330_RADIAL_NULL_REPORT",
        passed=True,
        claim_tier="E3",
        source_dag={"nodes": [{"id": "finite_source"}]},
        payload={
            "radial_svd": {
                "shape": [2, 3],
                "rank": 2,
                "nullity": 1,
                "singular_values": [1.0, 0.5],
                "rank_threshold": 1.0e-12,
                "condition_number_nonzero": 2.0,
                "right_null_basis_hash": digest,
                "resolution_kernel_hash": digest,
            }
        },
    )

    assert result["schema_version"] == "scr330-radial-v2"
    assert result["passed"] is True
    assert (tmp_path / "scr330_radial_receipt.json").exists()


def _source_dilation_payload() -> dict[str, object]:
    digest = "sha256:" + "a" * 64
    return {
        "radial_branch": "SOURCE_DILATION",
        "source_embedding_hash": digest,
        "physical_mode_basis_id": "safe-dlnk-basis-v1",
        "background_curvature_status": "FlatExact",
        "Z_q": 1.0,
        "theta": P_STAR / 48.0,
        "A_q": 1.0,
        "A_zeta": 0.16080676040273595,
        "k_pivot": 1.0,
        "R_star": 1.0,
        "dilation_intertwiner": {
            "scale_ratios": [1.1, 1.7],
            "source_embedding_commutator_norms": [0.0],
            "screen_covariance_naturality_residual_norms": [0.0],
            "physical_operator_residual_norms": [0.0],
            "strong_covariance_cauchy_residuals": [0.0],
            "strong_dilation_cauchy_residuals": [0.0],
            "uniform_covariance_norm_bound": 1.0,
            "finite_to_continuum_passed": True,
            "off_band_leakage": 0.0,
            "max_absolute_log_residual": 0.0,
            "rms_log_residual": 0.0,
            "tolerance": 1.0e-9,
            "passed": True,
        },
        "mellin_lift": {
            "ell_min": 2,
            "ell_max": 20,
            "convergence_strip": [-2.0, 4.0],
            "A_zeta_over_A_q": 0.16080676040273595,
            "max_arithmetic_residual": 0.0,
            "tolerance": 1.0e-12,
            "passed": True,
        },
        "radial_svd": {
            "shape": [2, 3],
            "rank": 2,
            "nullity": 1,
            "singular_values": [1.0, 0.5],
            "rank_threshold": 1.0e-12,
            "condition_number_nonzero": 2.0,
            "right_null_basis_hash": digest,
            "resolution_kernel_hash": digest,
        },
        "forward_residual": {
            "absolute_l2_residual": 0.0,
            "relative_l2_residual": 0.0,
            "tolerance": 1.0e-9,
            "passed": True,
            "held_out": True,
        },
    }


def _validate_schema(receipt: dict[str, object]) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(receipt)
