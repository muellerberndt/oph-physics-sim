from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from oph_fpe.claims import BRANCH_INSTANTIATION_SANITY, ISSUE_308_BW_CERTIFICATE_RECEIPT, with_claim_metadata


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
        "C4_modular_reference_tower": _modular_reference_tower(fields, limits),
        "C5_mixed_gns_and_support_covariance": _mixed_gns_and_support_covariance(fields, limits),
        "C6_geometric_rigidity": _geometric_rigidity(fields, limits),
        "C7_geometric_2pi_kms": _geometric_2pi_kms(fields, limits),
        "C8_wrong_normalization_and_nontriviality": _wrong_normalization(fields, limits),
    }
    envelope = _error_envelope(fields, limits)
    clause_pass = all(row["passed"] for row in clauses.values())
    any_evidence = any(row["evidence_present"] for row in clauses.values()) or envelope["evidence_present"]

    if clause_pass and envelope["passed"]:
        tier = "BW3"
    elif clause_pass or (sum(1 for row in clauses.values() if row["passed"]) >= 6 and envelope["evidence_present"]):
        tier = "BW2"
    elif any_evidence:
        tier = "BW1"
    else:
        tier = "BW0"

    receipt = tier == "BW3"
    report = {
        "mode": "issue_308_finite_cap_bw_certificate_audit",
        "tier": tier,
        ISSUE_308_BW_CERTIFICATE_RECEIPT: receipt,
        "issue_308_finite_cap_bw_certificate_receipt": receipt,
        "clauses": clauses,
        "error_envelope": envelope,
        "primitive_field_count": len(fields),
        "ignored_caller_pass_fields": ignored,
        "promotion_tiers": {
            "BW0": "visual or finite diagnostic",
            "BW1": "one-regulator finite hypothesis witness",
            "BW2": "refinement-compatible certificate sequence with vanishing-envelope evidence",
            "BW3": "Theorem 308 finite cap-net BW certificate applicable",
        },
        "nonclaims": {
            "bare_finite_consensus_implies_cap_bw_certificate": False,
            "canonical_h3_reconstruction": False,
            "record_populated_h3": False,
        },
        "claim_boundary": (
            "Issue #308 closes only FiniteCapBWCertificate => support-visible BW_S2. A renderer "
            "cap, fitted boost, finite cap-ID permutation, or coefficient near 2*pi is not a BW3 "
            "receipt. BW3 is recomputed from primitive cap-normal, frame, support-order, cross-ratio, "
            "mixed-GNS, geometric KMS, wrong-scale, and error-envelope fields."
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
    for key in ("BWRec_r", "bw_rec", "bwrec", "issue_308_primitive_fields"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {key: value for key, value in payload.items() if key not in _IGNORED_PASS_KEYS}


def _cap_normal_refinement(fields: Mapping[str, Any], limits: Mapping[str, float]) -> dict[str, Any]:
    missing = []
    cap_normal = fields.get("cap_normal")
    if not (isinstance(cap_normal, list) and len(cap_normal) == 4):
        missing.append("cap_normal")
    checks = {
        "cap_normal_norm_residual": _le_abs(fields, "cap_normal_norm_residual", limits["residual_tol"], missing),
        "cap_boundary_incidence_residual": _le_abs(
            fields, "cap_boundary_incidence_residual", limits["residual_tol"], missing
        ),
        "cap_sign_violation": _le_abs(fields, "cap_sign_violation", limits["residual_tol"], missing),
        "cap_mesh_error": _le_abs(fields, "cap_mesh_error", limits["mesh_tol"], missing),
        "point_mesh_error": _le_abs(fields, "point_mesh_error", limits["mesh_tol"], missing),
        "refinement_normal_error": _le_abs(fields, "refinement_normal_error", limits["residual_tol"], missing),
        "cap_radius_margin": _ge(fields, "cap_radius_margin", limits["frame_separation_min"], missing),
    }
    checks["cap_orientation"] = _present(fields, "cap_orientation", missing)
    return _stage(
        all(checks.values()) and not missing,
        "cap-normal density, nondegeneracy, boundary incidence, and refinement compatibility",
        missing,
        checks,
    )


def _bw_frame(fields: Mapping[str, Any], limits: Mapping[str, float]) -> dict[str, Any]:
    missing = []
    checks = {
        "frame_p_minus": _present(fields, "frame_p_minus", missing),
        "frame_p_plus": _present(fields, "frame_p_plus", missing),
        "frame_boundary_residual": _le_abs(fields, "frame_boundary_residual", limits["residual_tol"], missing),
        "frame_separation": _ge(fields, "frame_separation", limits["frame_separation_min"], missing),
        "frame_orientation_witness": _truthy(fields, "frame_orientation_witness", missing),
    }
    return _stage(all(checks.values()) and not missing, "ordered nondegenerate BW boundary frame", missing, checks)


def _prime_support_visible_cap_net(fields: Mapping[str, Any], limits: Mapping[str, float]) -> dict[str, Any]:
    missing = []
    checks = {
        "cap_inclusion_matrix": _present(fields, "cap_inclusion_matrix", missing),
        "strict_inclusion_margin": _present(fields, "strict_inclusion_margin", missing),
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


def _modular_reference_tower(fields: Mapping[str, Any], limits: Mapping[str, float]) -> dict[str, Any]:
    missing = []
    checks = {
        "test_tower_id": _present(fields, "test_tower_id", missing),
        "test_tower_hash": _present(fields, "test_tower_hash", missing),
        "state_embedding_residual": _le_abs(fields, "state_embedding_residual", limits["residual_tol"], missing),
        "regularizer_eta": _present(fields, "regularizer_eta", missing),
        "physical_reference_trace_distance": _le_abs(
            fields, "physical_reference_trace_distance", limits["residual_tol"], missing
        ),
        "fixed_local_modular_bound_T": _le_abs(
            fields, "fixed_local_modular_bound_T", limits["residual_tol"], missing
        ),
    }
    return _stage(all(checks.values()) and not missing, "refinement-compatible modular reference tower", missing, checks)


def _mixed_gns_and_support_covariance(fields: Mapping[str, Any], limits: Mapping[str, float]) -> dict[str, Any]:
    missing = []
    checks = {
        "mixed_gns_cauchy_residual_T": _le_abs(
            fields, "mixed_gns_cauchy_residual_T", limits["residual_tol"], missing
        ),
        "negative_time_residual_T": _le_abs(fields, "negative_time_residual_T", limits["residual_tol"], missing),
        "matrix_element_residual_T": _le_abs(fields, "matrix_element_residual_T", limits["residual_tol"], missing),
        "support_covariance_residual_T": _le_abs(
            fields, "support_covariance_residual_T", limits["residual_tol"], missing
        ),
    }
    return _stage(
        all(checks.values()) and not missing,
        "quadratic mixed-GNS convergence with inverse-time and support-covariance control",
        missing,
        checks,
    )


def _geometric_rigidity(fields: Mapping[str, Any], limits: Mapping[str, float]) -> dict[str, Any]:
    missing = []
    checks = {
        "flow_identity_residual": _le_abs(fields, "flow_identity_residual", limits["residual_tol"], missing),
        "flow_group_residual_T": _le_abs(fields, "flow_group_residual_T", limits["residual_tol"], missing),
        "flow_inverse_residual_T": _le_abs(fields, "flow_inverse_residual_T", limits["residual_tol"], missing),
        "flow_equi_continuity_bound": _finite(fields, "flow_equi_continuity_bound", missing),
        "cap_anchor_residual": _le_abs(fields, "cap_anchor_residual", limits["residual_tol"], missing),
        "frame_fixed_point_residual": _le_abs(
            fields, "frame_fixed_point_residual", limits["residual_tol"], missing
        ),
        "cross_ratio_holdout_max": _le_abs(fields, "cross_ratio_holdout_max", limits["residual_tol"], missing),
        "quartet_separation_min": _ge(fields, "quartet_separation_min", limits["quartet_separation_min"], missing),
        "cross_ratio_anchor_condition": _finite(fields, "cross_ratio_anchor_condition", missing),
        "orientation_witness": _truthy(fields, "orientation_witness", missing),
    }
    return _stage(
        all(checks.values()) and not missing,
        "held-out complex cross-ratio rigidity, frame preservation, and orientation",
        missing,
        checks,
    )


def _geometric_2pi_kms(fields: Mapping[str, Any], limits: Mapping[str, float]) -> dict[str, Any]:
    missing = []
    convention = str(fields.get("geometric_parameter_convention", ""))
    convention_ok = any(token in convention for token in ("e^{-s}", "e^-s", "exp(-s)", "geometric"))
    if not convention_ok:
        missing.append("geometric_parameter_convention")
    checks = {
        "geometric_parameter_convention": convention_ok,
        "kms_strip_bound": _le_abs(fields, "kms_strip_bound", limits["strip_bound_max"], missing),
        "kms_residual_beta_2pi": _le_abs(fields, "kms_residual_beta_2pi", limits["residual_tol"], missing),
    }
    return _stage(
        all(checks.values()) and not missing,
        "independently normalized geometric 2pi-KMS strip residual",
        missing,
        checks,
    )


def _wrong_normalization(fields: Mapping[str, Any], limits: Mapping[str, float]) -> dict[str, Any]:
    missing = []
    nontrivial = _truthy(fields, "geometric_flow_nontrivial", missing)
    interval = _present(fields, "wrong_beta_interval", missing)
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
    if envelope is None:
        missing.append("total_308_error_envelope")
    sample_ok = True
    if samples is not None:
        values = [_number(value) for value in samples if _number(value) is not None]
        sample_ok = bool(values) and all(values[i + 1] <= values[i] for i in range(len(values) - 1))
    passed = envelope is not None and abs(envelope) <= limits["error_envelope_tol"] and sample_ok
    return {
        "passed": bool(passed),
        "evidence_present": envelope is not None or samples is not None,
        "meaning": "combined issue-308 error envelope is below threshold and samples are nonincreasing if supplied",
        "missing_or_blocking_evidence": missing,
        "details": {
            "total_308_error_envelope": envelope,
            "error_envelope_samples_nonincreasing": sample_ok,
            "threshold": limits["error_envelope_tol"],
        },
    }


def _stage(passed: bool, meaning: str, missing: list[str], details: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "evidence_present": bool(details) and len(missing) < len(details),
        "meaning": meaning,
        "missing_or_blocking_evidence": sorted(set(missing)),
        "details": details,
    }


def _present(fields: Mapping[str, Any], key: str, missing: list[str]) -> bool:
    if key not in fields or fields.get(key) in (None, ""):
        missing.append(key)
        return False
    return True


def _truthy(fields: Mapping[str, Any], key: str, missing: list[str]) -> bool:
    if key not in fields:
        missing.append(key)
        return False
    value = fields.get(key)
    return bool(value) and str(value).lower() not in {"false", "0", "no", "none"}


def _finite(fields: Mapping[str, Any], key: str, missing: list[str]) -> bool:
    value = _number(fields.get(key))
    if value is None:
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
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}
