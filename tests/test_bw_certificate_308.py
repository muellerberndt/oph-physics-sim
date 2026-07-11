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
