from __future__ import annotations

import json
import math

import pytest

from oph_fpe.bulk.bw_certificate_308 import issue308_bw_certificate_report
from oph_fpe.bulk.theorem_contract import finite_oph_theorem_contract_report
from oph_fpe.claims import (
    BW_SAME_TOWER_INPUTS_RECEIPT,
    ISSUE_308_BW_CERTIFICATE_RECEIPT,
    MGNS1_CERTIFICATE_RECEIPT,
    SUPPORT_VISIBLE_BW_THEOREM_APPLICABLE_RECEIPT,
)


def _primitive_bwrec() -> dict:
    spatial_normal = math.sqrt(1.0 + 0.1**2)
    boundary_z = 0.1 / spatial_normal
    boundary_x = math.sqrt(1.0 - boundary_z**2)
    return {
        "finite_cap_source_role": "geometric_support_flow",
        "finite_cap_source_artifact_hash": f"sha256:{'1' * 64}",
        "finite_cap_tower_id": "tower-v1",
        "finite_cap_tower_hash": f"sha256:{'2' * 64}",
        "cap_normal": [0.1, 0.0, 0.0, spatial_normal],
        "cap_normal_norm_residual": 0.0,
        "cap_orientation": "interior_positive",
        "cap_radius_margin": 0.25,
        "cap_boundary_incidence_residual": 0.0,
        "cap_sign_violation": 0.0,
        "cap_mesh_error": 1.0e-4,
        "point_mesh_error": 1.0e-4,
        "refinement_normal_error": 0.0,
        "frame_p_minus": [boundary_x, 0.0, boundary_z],
        "frame_p_plus": [-boundary_x, 0.0, boundary_z],
        "frame_boundary_residual": 0.0,
        "frame_separation": 2.0 * boundary_x,
        "frame_ordering": "p_minus_attracting_for_positive_s",
        "frame_orientation_witness": True,
        "cap_inclusion_matrix": [[1, 0], [0, 1]],
        "strict_inclusion_margin": 0.1,
        "order_refinement_error": 0.0,
        "support_isotony_failures": 0,
        "support_separation_margin": 0.1,
        "support_covariance_residual_T": 1.0e-7,
        "support_kernel_residual": 0.0,
        "sector_scope": "PRIME_GEOMETRIC_SUPPORT_VISIBLE",
        "flow_identity_residual": 0.0,
        "flow_group_residual_T": 1.0e-7,
        "flow_inverse_residual_T": 1.0e-7,
        "flow_equi_continuity_bound": 2.0,
        "cap_anchor_residual": 0.0,
        "frame_fixed_point_residual": 0.0,
        "cross_ratio_holdout_max": 1.0e-7,
        "quartet_separation_min": 0.25,
        "cross_ratio_anchor_condition": 3.0,
        "orientation_witness": True,
        "geometric_parameter_convention": "h_C(z) -> e^{-s} h_C(z)",
        "kms_comparison_state_id": "finite-state-r128",
        "kms_comparison_state_hash": f"sha256:{'6' * 64}",
        "kms_matrix_element_residual_T": 1.0e-7,
        "kms_strip_bound": 10.0,
        "kms_residual_beta_2pi": 1.0e-7,
        "geometric_flow_nontrivial": True,
        "wrong_beta_interval": [1.0, 10.0],
        "wrong_beta_gap_delta": 0.05,
        "geometric_generator_noncentrality": 0.2,
        "generator_distance_beta_2pi": 1.0e-7,
        "total_308_error_envelope": 5.0e-7,
        "error_envelope_samples": [1.0e-5, 5.0e-7],
        "error_envelope_refinement_levels": [64, 128],
        "error_envelope_refinement_witness": True,
    }


def _primitive_mgns1() -> dict:
    return {
        "certificate_kind": "MGNS-1",
        "source_role": "algebra_state_tower",
        "source_artifact_hash": f"sha256:{'4' * 64}",
        "tower_id": "tower-v1",
        "tower_hash": f"sha256:{'2' * 64}",
        "fixed_local_algebra_ids": ["A64", "A128"],
        "fine_to_coarse_embedding_residual_T": 1.0e-7,
        "state_restriction_residual_T": 1.0e-7,
        "expectation_idempotence_residual_T": 1.0e-7,
        "expectation_state_preservation_residual_T": 1.0e-7,
        "comparison_map_isometry_residual_T": 1.0e-7,
        "cyclic_vector_residual_T": 1.0e-7,
        "separating_vector_margin": 0.1,
        "state_vector_compatibility_residual_T": 1.0e-7,
        "density_matrix_trace_residual_T": 1.0e-7,
        "regularizer_eta": 1.0e-5,
        "regularization_schedule": [1.0e-3, 1.0e-4],
        "state_ids_by_level": ["finite-state-r64", "finite-state-r128"],
        "state_fingerprints_by_level": [f"sha256:{'5' * 64}", f"sha256:{'6' * 64}"],
        "mixed_gns_cauchy_residual_T": 1.0e-7,
        "negative_time_residual_T": 1.0e-7,
        "matrix_element_residual_T": 1.0e-7,
        "modular_identity_residual": 1.0e-7,
        "modular_group_residual_T": 1.0e-7,
        "modular_inverse_residual_T": 1.0e-7,
        "modular_support_covariance_residual_T": 1.0e-7,
        "cap_family_uniformity_bound_T": 1.0e-7,
        "cofinal_cauchy_modulus_T": 1.0e-7,
    }


def test_issue308_finite_cap_certificate_does_not_make_bw_applicable_alone() -> None:
    report = issue308_bw_certificate_report({"BWRec_r": _primitive_bwrec(), "bw_passed": False})

    assert report["tier"] == "FC3"
    assert report[ISSUE_308_BW_CERTIFICATE_RECEIPT] is True
    assert report["issue_308_finite_cap_bw_certificate_receipt"] is True
    assert report[MGNS1_CERTIFICATE_RECEIPT] is False
    assert report[BW_SAME_TOWER_INPUTS_RECEIPT] is False
    assert report[SUPPORT_VISIBLE_BW_THEOREM_APPLICABLE_RECEIPT] is False
    assert report["ignored_caller_pass_fields"]["bw_passed"] is False
    assert all(row["passed"] for row in report["clauses"].values())
    assert report["nonclaims"]["canonical_h3_reconstruction"] is False
    assert report["nonclaims"]["record_populated_h3"] is False


def test_issue308_bw_applicability_requires_complete_independent_same_tower_mgns1() -> None:
    report = issue308_bw_certificate_report(
        {"BWRec_r": _primitive_bwrec(), "MGNS1Rec_r": _primitive_mgns1()}
    )

    assert report[ISSUE_308_BW_CERTIFICATE_RECEIPT] is True
    assert report[MGNS1_CERTIFICATE_RECEIPT] is True
    assert report[BW_SAME_TOWER_INPUTS_RECEIPT] is True
    assert report[SUPPORT_VISIBLE_BW_THEOREM_APPLICABLE_RECEIPT] is True


def test_issue308_repeated_rho_diagnostic_is_not_mgns1() -> None:
    mgns1 = _primitive_mgns1()
    mgns1["state_fingerprints_by_level"] = [f"sha256:{'5' * 64}"] * 2

    report = issue308_bw_certificate_report(
        {"BWRec_r": _primitive_bwrec(), "MGNS1Rec_r": mgns1}
    )

    assert report[ISSUE_308_BW_CERTIFICATE_RECEIPT] is True
    assert report[MGNS1_CERTIFICATE_RECEIPT] is False
    assert report["mgns1"]["details"]["repeated_rho_diagnostic"] is True
    assert report[SUPPORT_VISIBLE_BW_THEOREM_APPLICABLE_RECEIPT] is False


def test_issue308_rejects_complete_mgns1_from_a_different_tower() -> None:
    mgns1 = _primitive_mgns1()
    mgns1["tower_id"] = "tower-v2"

    report = issue308_bw_certificate_report(
        {"BWRec_r": _primitive_bwrec(), "MGNS1Rec_r": mgns1}
    )

    assert report[MGNS1_CERTIFICATE_RECEIPT] is True
    assert report[BW_SAME_TOWER_INPUTS_RECEIPT] is False
    assert report[SUPPORT_VISIBLE_BW_THEOREM_APPLICABLE_RECEIPT] is False


def test_issue308_bw_certificate_ignores_producer_supplied_pass_boolean() -> None:
    report = issue308_bw_certificate_report(
        {
            "bw_passed": True,
            "tier": "FC3",
            ISSUE_308_BW_CERTIFICATE_RECEIPT: True,
            "cap_normal": [0.0, 0.0, 0.0, 1.0],
        }
    )

    assert report["tier"] != "FC3"
    assert report[ISSUE_308_BW_CERTIFICATE_RECEIPT] is False
    assert "bw_passed" in report["ignored_caller_pass_fields"]
    assert "tier" in report["ignored_caller_pass_fields"]
    assert report["clauses"]["C4_geometric_support_flow"]["passed"] is False


def test_theorem_contract_surfaces_issue308_bw_certificate_from_primitive_file(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "bw_rec_308.json").write_text(json.dumps({"BWRec_r": _primitive_bwrec()}), encoding="utf-8")

    report = finite_oph_theorem_contract_report(run)

    assert report["issue_308_bw_certificate"]["finite_cap_receipt"] is True
    assert report["issue_308_bw_certificate"]["theorem_applicable"] is False
    assert report["issue_308_bw_certificate"]["tier"] == "FC3"
    assert report[ISSUE_308_BW_CERTIFICATE_RECEIPT] is True


def test_issue308_rejects_supplied_zero_residual_for_invalid_cap_normal() -> None:
    fields = _primitive_bwrec()
    fields["cap_normal"] = [100.0, 0.0, 0.0, 0.0]
    fields["cap_normal_norm_residual"] = 0.0

    report = issue308_bw_certificate_report({"BWRec_r": fields})

    assert report[ISSUE_308_BW_CERTIFICATE_RECEIPT] is False
    clause = report["clauses"]["C1_cap_normal_refinement"]
    assert clause["passed"] is False
    assert clause["details"]["cap_normal_computed_norm_residual"] == 10001.0


def test_issue308_rejects_negative_inclusion_margin_and_malformed_interval() -> None:
    fields = _primitive_bwrec()
    fields["strict_inclusion_margin"] = -999.0
    fields["wrong_beta_interval"] = [10.0, 1.0]

    report = issue308_bw_certificate_report({"BWRec_r": fields})

    assert report[ISSUE_308_BW_CERTIFICATE_RECEIPT] is False
    assert report["clauses"]["C3_prime_support_visible_cap_net"]["passed"] is False
    assert report["clauses"]["C6_wrong_normalization_and_nontriviality"]["passed"] is False


def test_issue308_rejects_negative_or_nonfinite_error_envelope_samples() -> None:
    fields = _primitive_bwrec()
    fields["error_envelope_samples"] = [1.0e-5, -5.0e-7]

    report = issue308_bw_certificate_report({"BWRec_r": fields})

    assert report[ISSUE_308_BW_CERTIFICATE_RECEIPT] is False
    assert report["error_envelope"]["passed"] is False


def test_issue308_downgrades_without_validated_error_envelope_refinement() -> None:
    fields = _primitive_bwrec()
    fields.pop("error_envelope_refinement_levels")
    fields.pop("error_envelope_refinement_witness")

    report = issue308_bw_certificate_report({"BWRec_r": fields})

    assert report[ISSUE_308_BW_CERTIFICATE_RECEIPT] is False
    assert report["tier"] == "FC2"
    assert report["error_envelope"]["passed"] is False
    assert report["error_envelope"]["details"]["error_envelope_refinement_validated"] is False


def test_issue308_rejects_truthy_nonliteral_and_empty_witnesses() -> None:
    fields = _primitive_bwrec()
    fields["frame_orientation_witness"] = "true"
    fields["orientation_witness"] = {"passed": True}
    fields["geometric_flow_nontrivial"] = 1
    fields["kms_comparison_state_id"] = []
    fields["kms_comparison_state_hash"] = "   "
    fields["cap_inclusion_matrix"] = []
    fields["error_envelope_refinement_witness"] = "true"

    report = issue308_bw_certificate_report({"BWRec_r": fields})

    assert report[ISSUE_308_BW_CERTIFICATE_RECEIPT] is False
    assert report["clauses"]["C2_bw_frame"]["passed"] is False
    assert report["clauses"]["C3_prime_support_visible_cap_net"]["passed"] is False
    assert report["clauses"]["C4_geometric_support_flow"]["passed"] is False
    assert report["clauses"]["C5_geometric_2pi_kms_comparison"]["passed"] is False
    assert report["clauses"]["C6_wrong_normalization_and_nontriviality"]["passed"] is False
    assert report["error_envelope"]["passed"] is False


def test_issue308_rejects_malformed_typed_primitive_witnesses() -> None:
    fields = _primitive_bwrec()
    fields["frame_p_minus"] = [1.0, 0.0]
    fields["frame_p_plus"] = [1.0, 0.0, float("nan")]
    fields["cap_orientation"] = ["interior_positive"]
    fields["kms_matrix_element_residual_T"] = True
    fields["flow_equi_continuity_bound"] = -1.0
    fields["cross_ratio_anchor_condition"] = 0.0
    fields["wrong_beta_interval"] = [True, 10.0]

    report = issue308_bw_certificate_report({"BWRec_r": fields})

    assert report[ISSUE_308_BW_CERTIFICATE_RECEIPT] is False
    assert report["clauses"]["C1_cap_normal_refinement"]["passed"] is False
    assert report["clauses"]["C2_bw_frame"]["passed"] is False
    assert report["clauses"]["C4_geometric_support_flow"]["passed"] is False
    assert report["clauses"]["C5_geometric_2pi_kms_comparison"]["passed"] is False
    assert report["clauses"]["C6_wrong_normalization_and_nontriviality"]["passed"] is False


def test_issue308_recomputes_frame_boundary_incidence_instead_of_trusting_zero_residual() -> None:
    fields = _primitive_bwrec()
    fields["frame_p_minus"] = [1.0, 0.0, 0.0]
    fields["frame_p_plus"] = [-1.0, 0.0, 0.0]
    fields["frame_boundary_residual"] = 0.0
    fields["frame_separation"] = 2.0

    report = issue308_bw_certificate_report({"BWRec_r": fields})

    clause = report["clauses"]["C2_bw_frame"]
    assert report[ISSUE_308_BW_CERTIFICATE_RECEIPT] is False
    assert clause["passed"] is False
    assert clause["details"]["maximum_cap_boundary_incidence"] == pytest.approx(0.1)
    assert clause["details"]["frame_points_on_cap_boundary"] is False


def test_issue308_recomputes_frame_separation_instead_of_trusting_declared_margin() -> None:
    fields = _primitive_bwrec()
    fields["frame_p_plus"] = list(fields["frame_p_minus"])
    fields["frame_boundary_residual"] = 0.0
    fields["frame_separation"] = 999.0

    report = issue308_bw_certificate_report({"BWRec_r": fields})

    clause = report["clauses"]["C2_bw_frame"]
    assert report[ISSUE_308_BW_CERTIFICATE_RECEIPT] is False
    assert clause["passed"] is False
    assert clause["details"]["computed_separation"] == 0.0
    assert clause["details"]["frame_points_distinct_nondegenerate"] is False
    assert clause["details"]["frame_separation_recomputed"] is False


def test_issue308_rejects_nonunit_or_unordered_frame_points() -> None:
    fields = _primitive_bwrec()
    fields["frame_p_minus"] = [2.0 * value for value in fields["frame_p_minus"]]
    fields["frame_ordering"] = "unordered"

    report = issue308_bw_certificate_report({"BWRec_r": fields})

    clause = report["clauses"]["C2_bw_frame"]
    assert report[ISSUE_308_BW_CERTIFICATE_RECEIPT] is False
    assert clause["passed"] is False
    assert clause["details"]["frame_p_minus_finite_unit_s2"] is False
    assert clause["details"]["frame_ordering"] is False


def test_issue308_requires_paper_interior_positive_orientation() -> None:
    fields = _primitive_bwrec()
    fields["cap_orientation"] = "interior_negative"

    report = issue308_bw_certificate_report({"BWRec_r": fields})

    assert report[ISSUE_308_BW_CERTIFICATE_RECEIPT] is False
    assert report["clauses"]["C1_cap_normal_refinement"]["passed"] is False


@pytest.mark.parametrize(
    "convention",
    [
        "geometric",
        "h_C(z) -> e^{+s} h_C(z)",
        "h_C(z) -> exp(s) h_C(z)",
        "exp(-s)",
    ],
)
def test_issue308_rejects_nonpaper_geometric_flow_conventions(convention) -> None:
    fields = _primitive_bwrec()
    fields["geometric_parameter_convention"] = convention

    report = issue308_bw_certificate_report({"BWRec_r": fields})

    assert report[ISSUE_308_BW_CERTIFICATE_RECEIPT] is False
    assert report["clauses"]["C5_geometric_2pi_kms_comparison"]["passed"] is False
