from __future__ import annotations

import json

from oph_fpe.bulk.bw_certificate_308 import issue308_bw_certificate_report
from oph_fpe.bulk.theorem_contract import finite_oph_theorem_contract_report
from oph_fpe.claims import ISSUE_308_BW_CERTIFICATE_RECEIPT


def _primitive_bwrec() -> dict:
    return {
        "cap_normal": [0.1, 0.0, 0.0, 1.004987562],
        "cap_normal_norm_residual": 0.0,
        "cap_orientation": "interior_positive",
        "cap_radius_margin": 0.25,
        "cap_boundary_incidence_residual": 0.0,
        "cap_sign_violation": 0.0,
        "cap_mesh_error": 1.0e-4,
        "point_mesh_error": 1.0e-4,
        "refinement_normal_error": 0.0,
        "frame_p_minus": [1.0, 0.0, 0.0],
        "frame_p_plus": [-1.0, 0.0, 0.0],
        "frame_boundary_residual": 0.0,
        "frame_separation": 1.5,
        "frame_orientation_witness": True,
        "cap_inclusion_matrix": [[1, 0], [0, 1]],
        "strict_inclusion_margin": 0.1,
        "order_refinement_error": 0.0,
        "support_isotony_failures": 0,
        "support_separation_margin": 0.1,
        "support_covariance_residual_T": 1.0e-7,
        "support_kernel_residual": 0.0,
        "sector_scope": "PRIME_GEOMETRIC_SUPPORT_VISIBLE",
        "test_tower_id": "tower-v1",
        "test_tower_hash": "sha256:abc",
        "state_embedding_residual": 0.0,
        "regularizer_eta": 1.0e-5,
        "physical_reference_trace_distance": 0.0,
        "fixed_local_modular_bound_T": 1.0e-7,
        "mixed_gns_cauchy_residual_T": 1.0e-7,
        "negative_time_residual_T": 1.0e-7,
        "matrix_element_residual_T": 1.0e-7,
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


def test_issue308_bw_certificate_recomputes_bw3_from_primitive_fields() -> None:
    report = issue308_bw_certificate_report({"BWRec_r": _primitive_bwrec(), "bw_passed": False})

    assert report["tier"] == "BW3"
    assert report[ISSUE_308_BW_CERTIFICATE_RECEIPT] is True
    assert report["issue_308_finite_cap_bw_certificate_receipt"] is True
    assert report["ignored_caller_pass_fields"]["bw_passed"] is False
    assert all(row["passed"] for row in report["clauses"].values())
    assert report["nonclaims"]["canonical_h3_reconstruction"] is False
    assert report["nonclaims"]["record_populated_h3"] is False


def test_issue308_bw_certificate_ignores_producer_supplied_pass_boolean() -> None:
    report = issue308_bw_certificate_report(
        {
            "bw_passed": True,
            "tier": "BW3",
            ISSUE_308_BW_CERTIFICATE_RECEIPT: True,
            "cap_normal": [0.0, 0.0, 0.0, 1.0],
        }
    )

    assert report["tier"] != "BW3"
    assert report[ISSUE_308_BW_CERTIFICATE_RECEIPT] is False
    assert "bw_passed" in report["ignored_caller_pass_fields"]
    assert "tier" in report["ignored_caller_pass_fields"]
    assert report["clauses"]["C6_geometric_rigidity"]["passed"] is False


def test_theorem_contract_surfaces_issue308_bw_certificate_from_primitive_file(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "bw_rec_308.json").write_text(json.dumps({"BWRec_r": _primitive_bwrec()}), encoding="utf-8")

    report = finite_oph_theorem_contract_report(run)

    assert report["issue_308_bw_certificate"]["receipt"] is True
    assert report["issue_308_bw_certificate"]["tier"] == "BW3"
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
    assert report["clauses"]["C8_wrong_normalization_and_nontriviality"]["passed"] is False


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
    assert report["tier"] == "BW2"
    assert report["error_envelope"]["passed"] is False
    assert report["error_envelope"]["details"]["error_envelope_refinement_validated"] is False


def test_issue308_rejects_truthy_nonliteral_and_empty_witnesses() -> None:
    fields = _primitive_bwrec()
    fields["frame_orientation_witness"] = "true"
    fields["orientation_witness"] = {"passed": True}
    fields["geometric_flow_nontrivial"] = 1
    fields["test_tower_id"] = []
    fields["test_tower_hash"] = "   "
    fields["cap_inclusion_matrix"] = []
    fields["error_envelope_refinement_witness"] = "true"

    report = issue308_bw_certificate_report({"BWRec_r": fields})

    assert report[ISSUE_308_BW_CERTIFICATE_RECEIPT] is False
    assert report["clauses"]["C2_bw_frame"]["passed"] is False
    assert report["clauses"]["C3_prime_support_visible_cap_net"]["passed"] is False
    assert report["clauses"]["C4_modular_reference_tower"]["passed"] is False
    assert report["clauses"]["C6_geometric_rigidity"]["passed"] is False
    assert report["clauses"]["C8_wrong_normalization_and_nontriviality"]["passed"] is False
    assert report["error_envelope"]["passed"] is False


def test_issue308_rejects_malformed_typed_primitive_witnesses() -> None:
    fields = _primitive_bwrec()
    fields["frame_p_minus"] = [1.0, 0.0]
    fields["frame_p_plus"] = [1.0, 0.0, float("nan")]
    fields["cap_orientation"] = ["interior_positive"]
    fields["regularizer_eta"] = True
    fields["flow_equi_continuity_bound"] = -1.0
    fields["cross_ratio_anchor_condition"] = 0.0
    fields["wrong_beta_interval"] = [True, 10.0]

    report = issue308_bw_certificate_report({"BWRec_r": fields})

    assert report[ISSUE_308_BW_CERTIFICATE_RECEIPT] is False
    assert report["clauses"]["C1_cap_normal_refinement"]["passed"] is False
    assert report["clauses"]["C2_bw_frame"]["passed"] is False
    assert report["clauses"]["C4_modular_reference_tower"]["passed"] is False
    assert report["clauses"]["C6_geometric_rigidity"]["passed"] is False
    assert report["clauses"]["C8_wrong_normalization_and_nontriviality"]["passed"] is False
