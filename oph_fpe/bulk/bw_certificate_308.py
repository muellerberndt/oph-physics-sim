from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from oph_fpe.claims import (
    BRANCH_INSTANTIATION_SANITY,
    BW_SAME_TOWER_INPUTS_RECEIPT,
    ISSUE_308_BW_CERTIFICATE_RECEIPT,
    MGNS1_CERTIFICATE_RECEIPT,
    SUPPORT_VISIBLE_BW_THEOREM_APPLICABLE_RECEIPT,
    with_claim_metadata,
)


DEFAULT_THRESHOLDS = {
    "residual_tol": 1.0e-6,
    "mesh_tol": 1.0e-3,
    "frame_separation_min": 1.0e-3,
    "support_margin_min": 1.0e-6,
    "quartet_separation_min": 1.0e-3,
    "wrong_beta_gap_min": 1.0e-6,
    "strip_bound_max": 1.0e12,
    "error_envelope_tol": 1.0e-6,
}

PRIME_SCOPE = "PRIME_GEOMETRIC_SUPPORT_VISIBLE"

_IGNORED_PASS_KEYS = {
    "bw_passed",
    "bw3_passed",
    "BW3",
    "issue_308_finite_cap_bw_certificate_receipt",
    ISSUE_308_BW_CERTIFICATE_RECEIPT,
    "tier",
}

MGNS1_REQUIRED_FIELDS = (
    "certificate_kind",
    "source_role",
    "source_artifact_hash",
    "tower_id",
    "tower_hash",
    "fixed_local_algebra_ids",
    "fine_to_coarse_embedding_residual_T",
    "state_restriction_residual_T",
    "expectation_idempotence_residual_T",
    "expectation_state_preservation_residual_T",
    "comparison_map_isometry_residual_T",
    "cyclic_vector_residual_T",
    "separating_vector_margin",
    "state_vector_compatibility_residual_T",
    "density_matrix_trace_residual_T",
    "regularizer_eta",
    "regularization_schedule",
    "state_ids_by_level",
    "state_fingerprints_by_level",
    "mixed_gns_cauchy_residual_T",
    "negative_time_residual_T",
    "matrix_element_residual_T",
    "modular_identity_residual",
    "modular_group_residual_T",
    "modular_inverse_residual_T",
    "modular_support_covariance_residual_T",
    "cap_family_uniformity_bound_T",
    "cofinal_cauchy_modulus_T",
)


def issue308_bw_certificate_report(
    payload: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Recompute the issue #308 BW tier from primitive receipt fields.

    This deliberately ignores caller-provided pass/tier booleans. A renderer
    cap, fitted boost, cap-ID permutation, or producer-supplied `bw_passed` flag
    is at most BW0/BW1 unless the primitive certificate fields are present.
    """

    limits = {**DEFAULT_THRESHOLDS, **dict(thresholds or {})}
    fields = _primitive_fields(payload)
    ignored = {key: payload.get(key) for key in _IGNORED_PASS_KEYS if key in payload}

    clauses = {
        "C1_cap_normal_refinement": _cap_normal_refinement(fields, limits),
        "C2_bw_frame": _bw_frame(fields, limits),
        "C3_prime_support_visible_cap_net": _prime_support_visible_cap_net(fields, limits),
        "C4_geometric_support_flow": _geometric_rigidity(fields, limits),
        "C5_geometric_2pi_kms_comparison": _geometric_2pi_kms(fields, limits),
        "C6_wrong_normalization_and_nontriviality": _wrong_normalization(fields, limits),
    }
    envelope = _error_envelope(fields, limits)
    clause_pass = all(row["passed"] for row in clauses.values())
    any_evidence = any(row["evidence_present"] for row in clauses.values()) or envelope["evidence_present"]

    if clause_pass and envelope["passed"]:
        tier = "FC3"
    elif clause_pass or (sum(1 for row in clauses.values() if row["passed"]) >= 6 and envelope["evidence_present"]):
        tier = "FC2"
    elif any_evidence:
        tier = "FC1"
    else:
        tier = "FC0"

    finite_cap_receipt = tier == "FC3"
    mgns1 = _mgns1_certificate(_mgns1_fields(payload), limits)
    same_tower = _same_tower_input_pair(fields, _mgns1_fields(payload), mgns1)
    theorem_applicable = bool(
        finite_cap_receipt
        and mgns1[MGNS1_CERTIFICATE_RECEIPT]
        and same_tower[BW_SAME_TOWER_INPUTS_RECEIPT]
    )
    report = {
        "mode": "issue_308_finite_cap_bw_and_mgns1_pair_audit",
        "tier": tier,
        ISSUE_308_BW_CERTIFICATE_RECEIPT: finite_cap_receipt,
        "issue_308_finite_cap_bw_certificate_receipt": finite_cap_receipt,
        MGNS1_CERTIFICATE_RECEIPT: mgns1[MGNS1_CERTIFICATE_RECEIPT],
        BW_SAME_TOWER_INPUTS_RECEIPT: same_tower[BW_SAME_TOWER_INPUTS_RECEIPT],
        SUPPORT_VISIBLE_BW_THEOREM_APPLICABLE_RECEIPT: theorem_applicable,
        "support_visible_bw_theorem_applicable": theorem_applicable,
        "clauses": clauses,
        "error_envelope": envelope,
        "mgns1": mgns1,
        "same_tower_input_pair": same_tower,
        "primitive_field_count": len(fields),
        "ignored_caller_pass_fields": ignored,
        "promotion_tiers": {
            "FC0": "no finite cap-flow certificate evidence",
            "FC1": "partial finite cap-flow evidence",
            "FC2": "six finite clauses or their refinement envelope remain incomplete",
            "FC3": "complete FiniteCapBWCertificate only",
        },
        "nonclaims": {
            "bare_finite_consensus_implies_cap_bw_certificate": False,
            "finite_cap_certificate_alone_implies_bw": False,
            "repeated_rho_diagnostic_is_mgns1": False,
            "canonical_h3_reconstruction": False,
            "record_populated_h3": False,
        },
        "claim_boundary": (
            "FiniteCapBWCertificate contains only the six geometric, support-flow, orientation, "
            "continuity, KMS-comparison, and normalization clauses. Support-visible BW is theorem-"
            "applicable only when a separately verified complete MGNS-1 algebra-state package is "
            "bound to the same tower. A repeated-rho diagnostic is not MGNS-1."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=BRANCH_INSTANTIATION_SANITY,
        receipt=ISSUE_308_BW_CERTIFICATE_RECEIPT,
        physical_claim=False,
        observable_id="issue_308_finite_cap_bw_certificate",
        fit_objective="theorem_308_finite_cap_bw_certificate_fields",
    )


def write_issue308_bw_certificate_report(source: Path | Mapping[str, Any], out: Path | None = None) -> dict[str, Any]:
    payload = _read_json(source) if isinstance(source, Path) else dict(source)
    report = issue308_bw_certificate_report(payload)
    out_path = out
    if out_path is None and isinstance(source, Path):
        out_path = source.with_name("issue_308_bw_certificate_report.json")
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _primitive_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    for key in (
        "FiniteCapBWRec_r",
        "finite_cap_bw_rec",
        "BWRec_r",
        "bw_rec",
        "bwrec",
        "issue_308_primitive_fields",
    ):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    excluded = {*_IGNORED_PASS_KEYS, "MGNS1Rec_r", "mgns1", "mgns1_rec"}
    return {key: value for key, value in payload.items() if key not in excluded}


def _mgns1_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("MGNS1Rec_r", "mgns1", "mgns1_rec"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _cap_normal_refinement(fields: Mapping[str, Any], limits: Mapping[str, float]) -> dict[str, Any]:
    missing = []
    cap_normal = fields.get("cap_normal")
    computed_norm_residual = _cap_normal_norm_residual(cap_normal)
    if computed_norm_residual is None:
        missing.append("cap_normal")
    supplied_norm_residual = _number(fields.get("cap_normal_norm_residual"))
    checks = {
        "cap_normal_norm_residual": bool(
            computed_norm_residual is not None
            and computed_norm_residual <= limits["residual_tol"]
            and supplied_norm_residual is not None
            and abs(supplied_norm_residual) <= limits["residual_tol"]
            and abs(supplied_norm_residual - computed_norm_residual) <= limits["residual_tol"]
        ),
        "cap_boundary_incidence_residual": _le_abs(
            fields, "cap_boundary_incidence_residual", limits["residual_tol"], missing
        ),
        "cap_sign_violation": _le_abs(fields, "cap_sign_violation", limits["residual_tol"], missing),
        "cap_mesh_error": _le_abs(fields, "cap_mesh_error", limits["mesh_tol"], missing),
        "point_mesh_error": _le_abs(fields, "point_mesh_error", limits["mesh_tol"], missing),
        "refinement_normal_error": _le_abs(fields, "refinement_normal_error", limits["residual_tol"], missing),
        "cap_radius_margin": _ge(fields, "cap_radius_margin", limits["frame_separation_min"], missing),
    }
    if not checks["cap_normal_norm_residual"]:
        missing.append("cap_normal_minkowski_norm_recomputation")
    checks["cap_orientation"] = _enum_string(
        fields,
        "cap_orientation",
        {"interior_positive"},
        missing,
    )
    stage = _stage(
        all(checks.values()) and not missing,
        "cap-normal density, nondegeneracy, boundary incidence, and refinement compatibility",
        missing,
        checks,
    )
    stage["details"].update(
        {
            "cap_normal_computed_norm_residual": computed_norm_residual,
            "cap_normal_supplied_norm_residual": supplied_norm_residual,
        }
    )
    return stage


def _bw_frame(fields: Mapping[str, Any], limits: Mapping[str, float]) -> dict[str, Any]:
    missing = []
    geometry = _frame_geometry_report(fields)
    supplied_boundary_residual = _number(fields.get("frame_boundary_residual"))
    supplied_separation = _number(fields.get("frame_separation"))
    boundary_residual_consistent = bool(
        supplied_boundary_residual is not None
        and supplied_boundary_residual >= 0.0
        and geometry["computed_boundary_residual"] is not None
        and abs(supplied_boundary_residual - geometry["computed_boundary_residual"])
        <= limits["residual_tol"]
    )
    separation_consistent = bool(
        supplied_separation is not None
        and geometry["computed_separation"] is not None
        and abs(supplied_separation - geometry["computed_separation"]) <= limits["residual_tol"]
    )
    checks = {
        "cap_normal_finite_unit_spacelike": bool(
            geometry["cap_normal_norm_residual"] is not None
            and geometry["cap_normal_norm_residual"] <= limits["residual_tol"]
        ),
        "frame_p_minus_finite_unit_s2": bool(
            geometry["p_minus_unit_residual"] is not None
            and geometry["p_minus_unit_residual"] <= limits["residual_tol"]
        ),
        "frame_p_plus_finite_unit_s2": bool(
            geometry["p_plus_unit_residual"] is not None
            and geometry["p_plus_unit_residual"] <= limits["residual_tol"]
        ),
        "frame_points_on_cap_boundary": bool(
            geometry["maximum_cap_boundary_incidence"] is not None
            and geometry["maximum_cap_boundary_incidence"] <= limits["residual_tol"]
        ),
        "frame_boundary_residual_recomputed": boundary_residual_consistent,
        "frame_points_distinct_nondegenerate": bool(
            geometry["computed_separation"] is not None
            and geometry["computed_separation"] >= limits["frame_separation_min"]
        ),
        "frame_separation_recomputed": separation_consistent,
        "frame_ordering": _enum_string(
            fields,
            "frame_ordering",
            {"p_minus_attracting_for_positive_s"},
            missing,
        ),
        "frame_orientation_witness": _literal_true(fields, "frame_orientation_witness", missing),
    }
    for key, passed in checks.items():
        if not passed:
            missing.append(key)
    stage = _stage(all(checks.values()) and not missing, "ordered nondegenerate BW boundary frame", missing, checks)
    stage["details"].update(
        {
            **geometry,
            "supplied_frame_boundary_residual": supplied_boundary_residual,
            "supplied_frame_separation": supplied_separation,
        }
    )
    return stage


def _prime_support_visible_cap_net(fields: Mapping[str, Any], limits: Mapping[str, float]) -> dict[str, Any]:
    missing = []
    checks = {
        "cap_inclusion_matrix": _finite_nonempty_matrix(fields, "cap_inclusion_matrix", missing),
        "strict_inclusion_margin": _ge(
            fields, "strict_inclusion_margin", limits["support_margin_min"], missing
        ),
        "order_refinement_error": _le_abs(fields, "order_refinement_error", limits["residual_tol"], missing),
        "support_isotony_failures": _le_abs(fields, "support_isotony_failures", limits["residual_tol"], missing),
        "support_separation_margin": _ge(fields, "support_separation_margin", limits["support_margin_min"], missing),
        "support_covariance_residual_T": _le_abs(
            fields, "support_covariance_residual_T", limits["residual_tol"], missing
        ),
        "support_kernel_residual": _le_abs(fields, "support_kernel_residual", limits["residual_tol"], missing),
        "sector_scope": str(fields.get("sector_scope", "")) == PRIME_SCOPE,
    }
    if not checks["sector_scope"]:
        missing.append("sector_scope=PRIME_GEOMETRIC_SUPPORT_VISIBLE")
    return _stage(
        all(checks.values()) and not missing,
        "prime support-visible cap net with support order separation and covariance",
        missing,
        checks,
    )


def _mgns1_certificate(fields: Mapping[str, Any], limits: Mapping[str, float]) -> dict[str, Any]:
    missing: list[str] = []
    algebra_ids = fields.get("fixed_local_algebra_ids")
    algebra_ids_valid = bool(
        isinstance(algebra_ids, (list, tuple))
        and len(algebra_ids) >= 2
        and all(isinstance(value, str) and value.strip() for value in algebra_ids)
        and len(set(algebra_ids)) == len(algebra_ids)
    )
    if not algebra_ids_valid:
        missing.append("fixed_local_algebra_ids")

    fingerprints = fields.get("state_fingerprints_by_level")
    fingerprints_valid = bool(
        isinstance(fingerprints, (list, tuple))
        and len(fingerprints) >= 2
        and all(_strict_sha256(value) for value in fingerprints)
    )
    repeated_rho = bool(fingerprints_valid and len(set(fingerprints)) == 1)
    if not fingerprints_valid:
        missing.append("state_fingerprints_by_level")
    if repeated_rho:
        missing.append("repeated_state_across_levels_is_diagnostic_not_mgns1")

    state_ids = fields.get("state_ids_by_level")
    state_ids_valid = bool(
        isinstance(state_ids, (list, tuple))
        and fingerprints_valid
        and len(state_ids) == len(fingerprints)
        and all(isinstance(value, str) and value.strip() for value in state_ids)
        and len(set(state_ids)) == len(state_ids)
    )
    if not state_ids_valid:
        missing.append("state_ids_by_level")

    schedule_valid = _strict_regularization_schedule(fields.get("regularization_schedule"))
    if not schedule_valid:
        missing.append("regularization_schedule")

    checks = {
        "certificate_kind": fields.get("certificate_kind") == "MGNS-1",
        "source_role": fields.get("source_role") == "algebra_state_tower",
        "source_artifact_hash": _strict_sha256(fields.get("source_artifact_hash")),
        "tower_id": _nonempty_string(fields, "tower_id", missing),
        "tower_hash": _strict_sha256(fields.get("tower_hash")),
        "fixed_local_algebra_ids": algebra_ids_valid,
        "fine_to_coarse_embedding_residual_T": _le_abs(
            fields, "fine_to_coarse_embedding_residual_T", limits["residual_tol"], missing
        ),
        "state_restriction_residual_T": _le_abs(
            fields, "state_restriction_residual_T", limits["residual_tol"], missing
        ),
        "expectation_idempotence_residual_T": _le_abs(
            fields, "expectation_idempotence_residual_T", limits["residual_tol"], missing
        ),
        "expectation_state_preservation_residual_T": _le_abs(
            fields, "expectation_state_preservation_residual_T", limits["residual_tol"], missing
        ),
        "comparison_map_isometry_residual_T": _le_abs(
            fields, "comparison_map_isometry_residual_T", limits["residual_tol"], missing
        ),
        "cyclic_vector_residual_T": _le_abs(
            fields, "cyclic_vector_residual_T", limits["residual_tol"], missing
        ),
        "separating_vector_margin": _ge(
            fields, "separating_vector_margin", limits["support_margin_min"], missing
        ),
        "state_vector_compatibility_residual_T": _le_abs(
            fields, "state_vector_compatibility_residual_T", limits["residual_tol"], missing
        ),
        "density_matrix_trace_residual_T": _le_abs(
            fields, "density_matrix_trace_residual_T", limits["residual_tol"], missing
        ),
        "regularizer_eta": _positive(fields, "regularizer_eta", missing),
        "regularization_schedule": schedule_valid,
        "state_ids_by_level": state_ids_valid,
        "state_fingerprints_by_level": bool(fingerprints_valid and not repeated_rho),
        "mixed_gns_cauchy_residual_T": _le_abs(
            fields, "mixed_gns_cauchy_residual_T", limits["residual_tol"], missing
        ),
        "negative_time_residual_T": _le_abs(
            fields, "negative_time_residual_T", limits["residual_tol"], missing
        ),
        "matrix_element_residual_T": _le_abs(
            fields, "matrix_element_residual_T", limits["residual_tol"], missing
        ),
        "modular_identity_residual": _le_abs(
            fields, "modular_identity_residual", limits["residual_tol"], missing
        ),
        "modular_group_residual_T": _le_abs(
            fields, "modular_group_residual_T", limits["residual_tol"], missing
        ),
        "modular_inverse_residual_T": _le_abs(
            fields, "modular_inverse_residual_T", limits["residual_tol"], missing
        ),
        "modular_support_covariance_residual_T": _le_abs(
            fields, "modular_support_covariance_residual_T", limits["residual_tol"], missing
        ),
        "cap_family_uniformity_bound_T": _le_abs(
            fields, "cap_family_uniformity_bound_T", limits["residual_tol"], missing
        ),
        "cofinal_cauchy_modulus_T": _le_abs(
            fields, "cofinal_cauchy_modulus_T", limits["residual_tol"], missing
        ),
    }
    for key in ("certificate_kind", "source_role", "source_artifact_hash", "tower_hash"):
        if not checks[key]:
            missing.append(key)
    passed = bool(set(fields) == set(MGNS1_REQUIRED_FIELDS) and all(checks.values()) and not missing)
    if set(fields) != set(MGNS1_REQUIRED_FIELDS):
        missing.append("mgns1_primitive_field_key_set_mismatch")
    return {
        MGNS1_CERTIFICATE_RECEIPT: passed,
        "passed": passed,
        "evidence_present": bool(fields),
        "meaning": (
            "complete modular algebra-state representation with comparison maps, compact-time "
            "control, support covariance, cap-family uniformity, and a cofinal Cauchy modulus"
        ),
        "missing_or_blocking_evidence": sorted(set(missing)),
        "details": {
            **checks,
            "repeated_rho_diagnostic": repeated_rho,
            "required_primitive_fields": list(MGNS1_REQUIRED_FIELDS),
        },
    }


def _same_tower_input_pair(
    finite_fields: Mapping[str, Any],
    mgns_fields: Mapping[str, Any],
    mgns_report: Mapping[str, Any],
) -> dict[str, Any]:
    finite_tower_id = finite_fields.get("finite_cap_tower_id")
    finite_tower_hash = finite_fields.get("finite_cap_tower_hash")
    finite_source_hash = finite_fields.get("finite_cap_source_artifact_hash")
    mgns_tower_id = mgns_fields.get("tower_id")
    mgns_tower_hash = mgns_fields.get("tower_hash")
    mgns_source_hash = mgns_fields.get("source_artifact_hash")
    state_ids = mgns_fields.get("state_ids_by_level")
    state_hashes = mgns_fields.get("state_fingerprints_by_level")
    comparison_state_pairs = set(
        zip(state_ids, state_hashes)
        if isinstance(state_ids, (list, tuple))
        and isinstance(state_hashes, (list, tuple))
        and len(state_ids) == len(state_hashes)
        else ()
    )
    comparison_state_id = finite_fields.get("kms_comparison_state_id")
    comparison_state_hash = finite_fields.get("kms_comparison_state_hash")
    comparison_state_bound = bool(
        isinstance(comparison_state_id, str)
        and _strict_sha256(comparison_state_hash)
        and (comparison_state_id, comparison_state_hash) in comparison_state_pairs
    )
    checks = {
        "finite_cap_source_role": finite_fields.get("finite_cap_source_role")
        == "geometric_support_flow",
        "finite_cap_source_artifact_hash": _strict_sha256(finite_source_hash),
        "finite_cap_tower_id": isinstance(finite_tower_id, str) and bool(finite_tower_id.strip()),
        "finite_cap_tower_hash": _strict_sha256(finite_tower_hash),
        "mgns1_complete": mgns_report.get(MGNS1_CERTIFICATE_RECEIPT) is True,
        "tower_id_equal": finite_tower_id == mgns_tower_id,
        "tower_hash_equal": finite_tower_hash == mgns_tower_hash,
        "kms_comparison_state_bound_to_mgns1": comparison_state_bound,
        "independent_source_artifacts": bool(
            _strict_sha256(finite_source_hash)
            and _strict_sha256(mgns_source_hash)
            and finite_source_hash != mgns_source_hash
        ),
    }
    passed = all(checks.values())
    return {
        BW_SAME_TOWER_INPUTS_RECEIPT: passed,
        "passed": passed,
        "checks": checks,
        "finite_cap_tower_id": finite_tower_id,
        "mgns1_tower_id": mgns_tower_id,
        "missing_or_blocking_evidence": sorted(key for key, value in checks.items() if not value),
    }


def _geometric_rigidity(fields: Mapping[str, Any], limits: Mapping[str, float]) -> dict[str, Any]:
    missing = []
    checks = {
        "flow_identity_residual": _le_abs(fields, "flow_identity_residual", limits["residual_tol"], missing),
        "flow_group_residual_T": _le_abs(fields, "flow_group_residual_T", limits["residual_tol"], missing),
        "flow_inverse_residual_T": _le_abs(fields, "flow_inverse_residual_T", limits["residual_tol"], missing),
        "flow_equi_continuity_bound": _nonnegative(fields, "flow_equi_continuity_bound", missing),
        "cap_anchor_residual": _le_abs(fields, "cap_anchor_residual", limits["residual_tol"], missing),
        "frame_fixed_point_residual": _le_abs(
            fields, "frame_fixed_point_residual", limits["residual_tol"], missing
        ),
        "cross_ratio_holdout_max": _le_abs(fields, "cross_ratio_holdout_max", limits["residual_tol"], missing),
        "quartet_separation_min": _ge(fields, "quartet_separation_min", limits["quartet_separation_min"], missing),
        "cross_ratio_anchor_condition": _positive(fields, "cross_ratio_anchor_condition", missing),
        "orientation_witness": _literal_true(fields, "orientation_witness", missing),
    }
    return _stage(
        all(checks.values()) and not missing,
        "held-out complex cross-ratio rigidity, frame preservation, and orientation",
        missing,
        checks,
    )


def _geometric_2pi_kms(fields: Mapping[str, Any], limits: Mapping[str, float]) -> dict[str, Any]:
    missing = []
    raw_convention = fields.get("geometric_parameter_convention")
    convention = raw_convention if isinstance(raw_convention, str) else ""
    convention_ok = _negative_geometric_parameter_convention(convention)
    if not convention_ok:
        missing.append("geometric_parameter_convention")
    checks = {
        "geometric_parameter_convention": convention_ok,
        "kms_comparison_state_id": _nonempty_string(
            fields, "kms_comparison_state_id", missing
        ),
        "kms_comparison_state_hash": _strict_sha256(
            fields.get("kms_comparison_state_hash")
        ),
        "kms_matrix_element_residual_T": _le_abs(
            fields, "kms_matrix_element_residual_T", limits["residual_tol"], missing
        ),
        "kms_strip_bound": _le_abs(fields, "kms_strip_bound", limits["strip_bound_max"], missing),
        "kms_residual_beta_2pi": _le_abs(fields, "kms_residual_beta_2pi", limits["residual_tol"], missing),
    }
    if not checks["kms_comparison_state_hash"]:
        missing.append("kms_comparison_state_hash")
    return _stage(
        all(checks.values()) and not missing,
        (
            "independently normalized geometric 2pi-KMS comparison against an explicitly "
            "identified finite state; this clause does not certify its algebra-state tower"
        ),
        missing,
        checks,
    )


def _wrong_normalization(fields: Mapping[str, Any], limits: Mapping[str, float]) -> dict[str, Any]:
    missing = []
    nontrivial = _literal_true(fields, "geometric_flow_nontrivial", missing)
    interval = _valid_interval(fields.get("wrong_beta_interval"))
    if not interval:
        missing.append("wrong_beta_interval_finite_ordered_pair")
    gap = _ge(fields, "wrong_beta_gap_delta", limits["wrong_beta_gap_min"], [])
    generator = (
        _ge(fields, "geometric_generator_noncentrality", limits["support_margin_min"], [])
        and _le_abs(fields, "generator_distance_beta_2pi", limits["residual_tol"], [])
    )
    if not (gap or generator):
        missing.append("wrong_beta_gap_delta_or_type_I_generator_distance")
    checks = {
        "geometric_flow_nontrivial": nontrivial,
        "wrong_beta_interval": interval,
        "wrong_beta_gap_or_generator_bound": bool(gap or generator),
    }
    return _stage(
        all(checks.values()) and not missing,
        "wrong-normalization separation plus nontrivial geometric flow",
        missing,
        checks,
    )


def _error_envelope(fields: Mapping[str, Any], limits: Mapping[str, float]) -> dict[str, Any]:
    missing = []
    envelope = _number(fields.get("total_308_error_envelope"))
    samples = fields.get("error_envelope_samples")
    levels = fields.get("error_envelope_refinement_levels")
    refinement_witness = _literal_true(
        fields,
        "error_envelope_refinement_witness",
        missing,
    )
    if envelope is None:
        missing.append("total_308_error_envelope")
    sample_ok = False
    levels_ok = False
    values: list[float] = []
    refinement_levels: list[int] = []
    if not isinstance(samples, (list, tuple)) or len(samples) < 2:
        missing.append("error_envelope_samples_at_least_two_levels")
    else:
        parsed = [_number(value) for value in samples]
        values = [float(value) for value in parsed if value is not None]
        sample_ok = bool(
            len(values) == len(parsed)
            and all(value >= 0.0 for value in values)
            and all(values[index + 1] <= values[index] for index in range(len(values) - 1))
            and any(values[index + 1] < values[index] for index in range(len(values) - 1))
            and values[-1] <= limits["error_envelope_tol"]
            and envelope is not None
            and values[-1] <= envelope + 1.0e-15
        )
        if not sample_ok:
            missing.append("validated_nonincreasing_error_envelope_refinement")
    if not isinstance(levels, (list, tuple)) or len(levels) != len(values):
        missing.append("error_envelope_refinement_levels")
    else:
        levels_ok = bool(
            len(levels) >= 2
            and all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in levels)
            and all(levels[index + 1] > levels[index] for index in range(len(levels) - 1))
        )
        if levels_ok:
            refinement_levels = [int(value) for value in levels]
        else:
            missing.append("strictly_increasing_positive_integer_refinement_levels")
    refinement_validated = bool(refinement_witness and sample_ok and levels_ok)
    passed = bool(
        envelope is not None
        and envelope >= 0.0
        and envelope <= limits["error_envelope_tol"]
        and refinement_validated
    )
    return {
        "passed": bool(passed),
        "evidence_present": envelope is not None or samples is not None,
        "meaning": (
            "combined issue-308 error envelope is below threshold and is backed by a literal, "
            "typed, strictly refined multi-level error-envelope witness"
        ),
        "missing_or_blocking_evidence": sorted(set(missing)),
        "details": {
            "total_308_error_envelope": envelope,
            "error_envelope_samples_nonincreasing": sample_ok,
            "error_envelope_samples": values,
            "error_envelope_refinement_levels": refinement_levels,
            "error_envelope_refinement_witness": refinement_witness,
            "error_envelope_refinement_validated": refinement_validated,
            "threshold": limits["error_envelope_tol"],
        },
    }


def _cap_normal_norm_residual(value: Any) -> float | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    components = [_number(component) for component in value]
    if any(component is None for component in components):
        return None
    t, x, y, z = (float(component) for component in components)
    return abs((-t * t + x * x + y * y + z * z) - 1.0)


def _frame_geometry_report(fields: Mapping[str, Any]) -> dict[str, float | None]:
    normal = _finite_vector_values(fields.get("cap_normal"), 4)
    p_minus = _finite_vector_values(fields.get("frame_p_minus"), 3)
    p_plus = _finite_vector_values(fields.get("frame_p_plus"), 3)
    normal_residual = _cap_normal_norm_residual(normal)
    minus_unit_residual = _s2_unit_residual(p_minus)
    plus_unit_residual = _s2_unit_residual(p_plus)
    incidences: list[float] = []
    if normal is not None:
        for point in (p_minus, p_plus):
            if point is not None:
                incidences.append(
                    abs(-normal[0] + sum(normal[index + 1] * point[index] for index in range(3)))
                )
    maximum_incidence = max(incidences) if len(incidences) == 2 else None
    separation = None
    if p_minus is not None and p_plus is not None:
        separation = math.sqrt(sum((p_minus[index] - p_plus[index]) ** 2 for index in range(3)))
    residual_components = [
        value
        for value in (minus_unit_residual, plus_unit_residual, maximum_incidence)
        if value is not None
    ]
    computed_boundary_residual = max(residual_components) if len(residual_components) == 3 else None
    return {
        "cap_normal_norm_residual": normal_residual,
        "p_minus_unit_residual": minus_unit_residual,
        "p_plus_unit_residual": plus_unit_residual,
        "maximum_cap_boundary_incidence": maximum_incidence,
        "computed_boundary_residual": computed_boundary_residual,
        "computed_separation": separation,
    }


def _finite_vector_values(value: Any, size: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        return None
    parsed = [_number(component) for component in value]
    if any(component is None for component in parsed):
        return None
    return [float(component) for component in parsed if component is not None]


def _s2_unit_residual(point: list[float] | None) -> float | None:
    if point is None:
        return None
    return abs(sum(component * component for component in point) - 1.0)


def _negative_geometric_parameter_convention(value: str) -> bool:
    compact = "".join(value.lower().split()).replace("→", "->").replace("↦", "->")
    mapping_explicit = "->" in compact or "\\mapsto" in compact
    negative_scale_explicit = any(
        token in compact
        for token in ("e^{-s}", "e^(-s)", "e^-s", "exp(-s)")
    )
    positive_scale_present = any(
        token in compact
        for token in ("e^{+s}", "e^{s}", "e^(+s)", "e^+s", "exp(+s)", "exp(s)")
    )
    return bool(
        compact.count("h") >= 2
        and mapping_explicit
        and negative_scale_explicit
        and not positive_scale_present
    )


def _valid_interval(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    lower = _number(value[0])
    upper = _number(value[1])
    return bool(lower is not None and upper is not None and lower < upper)


def _stage(passed: bool, meaning: str, missing: list[str], details: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "evidence_present": bool(details) and len(missing) < len(details),
        "meaning": meaning,
        "missing_or_blocking_evidence": sorted(set(missing)),
        "details": details,
    }


def _nonempty_string(fields: Mapping[str, Any], key: str, missing: list[str]) -> bool:
    value = fields.get(key)
    if not isinstance(value, str) or not value.strip():
        missing.append(key)
        return False
    return True


def _enum_string(fields: Mapping[str, Any], key: str, allowed: set[str], missing: list[str]) -> bool:
    value = fields.get(key)
    if not isinstance(value, str) or value not in allowed:
        missing.append(key)
        return False
    return True


def _finite_nonempty_matrix(fields: Mapping[str, Any], key: str, missing: list[str]) -> bool:
    value = fields.get(key)
    valid = False
    if isinstance(value, (list, tuple)) and value:
        rows = list(value)
        widths = [len(row) for row in rows if isinstance(row, (list, tuple))]
        valid = bool(
            len(widths) == len(rows)
            and widths
            and widths[0] > 0
            and all(width == widths[0] for width in widths)
            and all(_number(component) is not None for row in rows for component in row)
        )
    if not valid:
        missing.append(key)
    return valid


def _literal_true(fields: Mapping[str, Any], key: str, missing: list[str]) -> bool:
    value = fields.get(key)
    valid = isinstance(value, bool) and value is True
    if not valid:
        missing.append(key)
    return valid


def _nonnegative(fields: Mapping[str, Any], key: str, missing: list[str]) -> bool:
    value = _number(fields.get(key))
    if value is None or value < 0.0:
        missing.append(key)
        return False
    return True


def _positive(fields: Mapping[str, Any], key: str, missing: list[str]) -> bool:
    value = _number(fields.get(key))
    if value is None or value <= 0.0:
        missing.append(key)
        return False
    return True


def _le_abs(fields: Mapping[str, Any], key: str, limit: float, missing: list[str]) -> bool:
    value = _number(fields.get(key))
    if value is None:
        missing.append(key)
        return False
    return abs(value) <= float(limit)


def _ge(fields: Mapping[str, Any], key: str, limit: float, missing: list[str]) -> bool:
    value = _number(fields.get(key))
    if value is None:
        missing.append(key)
        return False
    return value >= float(limit)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _strict_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"sha256:[0-9a-f]{64}", str(value or "")))


def _strict_regularization_schedule(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    parsed = [_number(item) for item in value]
    if any(item is None or item <= 0.0 for item in parsed):
        return False
    numbers = [float(item) for item in parsed if item is not None]
    return all(numbers[index + 1] < numbers[index] for index in range(len(numbers) - 1))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}
