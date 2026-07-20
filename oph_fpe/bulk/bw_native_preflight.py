"""Strict native finite-cap plus MGNS-1 payload verifier.

The input is deliberately a bundle of primitive simulator artifacts, not a
bundle of producer supplied pass flags. Each of the six finite-cap clauses and
the separate MGNS-1 package have exact, independently hashed field contracts.
Support-visible BW is applicable only when both certificates pass on the same
tower.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from oph_fpe.bulk.bw_certificate_308 import (
    MGNS1_REQUIRED_FIELDS,
    issue308_bw_certificate_report,
)
from oph_fpe.claims import (
    BW_SAME_TOWER_INPUTS_RECEIPT,
    ISSUE_308_BW_CERTIFICATE_RECEIPT,
    MGNS1_CERTIFICATE_RECEIPT,
    SUPPORT_VISIBLE_BW_THEOREM_APPLICABLE_RECEIPT,
)

BW_NATIVE_SCHEMA_VERSION = "oph_bw_native_finite_cap_mgns1_payload_v2"
REQUIRED_BW_CLAUSE_IDS = tuple(f"C{index}" for index in range(1, 7))

_TOP_LEVEL_KEYS = {
    "schema_version",
    "producer_kind",
    "source_kind",
    "antecedent_hash",
    "clauses",
    "mgns1",
}
_CLAUSE_KEYS = {"antecedent_hash", "primitive_artifact_hash", "primitive_fields"}
_FORBIDDEN_ASSERTION_KEYS = {
    "passed",
    "pass",
    "tier",
    "bw_passed",
    "bw3_passed",
    "receipt",
}

# The overlap between C1 and C2 for ``cap_normal`` is intentional. Duplicate
# primitive fields must be byte-equivalent across clauses.
BW_PRIMITIVE_FIELD_CONTRACT: dict[str, tuple[str, ...]] = {
    "C1": (
        "finite_cap_source_role",
        "finite_cap_source_artifact_hash",
        "finite_cap_tower_id",
        "finite_cap_tower_hash",
        "cap_normal",
        "cap_normal_norm_residual",
        "cap_orientation",
        "cap_radius_margin",
        "cap_boundary_incidence_residual",
        "cap_sign_violation",
        "cap_mesh_error",
        "point_mesh_error",
        "refinement_normal_error",
    ),
    "C2": (
        "cap_normal",
        "frame_p_minus",
        "frame_p_plus",
        "frame_boundary_residual",
        "frame_separation",
        "frame_ordering",
        "frame_orientation_witness",
    ),
    "C3": (
        "cap_inclusion_matrix",
        "strict_inclusion_margin",
        "order_refinement_error",
        "support_isotony_failures",
        "support_separation_margin",
        "support_covariance_residual_T",
        "support_kernel_residual",
        "sector_scope",
    ),
    "C4": (
        "flow_identity_residual",
        "flow_group_residual_T",
        "flow_inverse_residual_T",
        "flow_equi_continuity_bound",
        "cap_anchor_residual",
        "frame_fixed_point_residual",
        "cross_ratio_holdout_max",
        "quartet_separation_min",
        "cross_ratio_anchor_condition",
        "orientation_witness",
    ),
    "C5": (
        "geometric_parameter_convention",
        "kms_comparison_state_id",
        "kms_comparison_state_hash",
        "kms_matrix_element_residual_T",
        "kms_strip_bound",
        "kms_residual_beta_2pi",
    ),
    "C6": (
        "geometric_flow_nontrivial",
        "wrong_beta_interval",
        "wrong_beta_gap_delta",
        "geometric_generator_noncentrality",
        "generator_distance_beta_2pi",
        "total_308_error_envelope",
        "error_envelope_samples",
        "error_envelope_refinement_levels",
        "error_envelope_refinement_witness",
    ),
}

_ISSUE308_TO_BW = {
    "C1": "C1_cap_normal_refinement",
    "C2": "C2_bw_frame",
    "C3": "C3_prime_support_visible_cap_net",
    "C4": "C4_geometric_support_flow",
    "C5": "C5_geometric_2pi_kms_comparison",
    "C6": "C6_wrong_normalization_and_nontriviality",
}


def canonical_payload_hash(value: Any) -> str:
    """Return a deterministic SHA-256 identifier for a JSON-compatible value."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def native_bw_pair_report(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Recompute the six finite-cap clauses and the separate MGNS-1 package.

    Caller-provided pass/tier/receipt fields are schema violations.  Missing,
    extra, replay-derived, fixture-derived, unhashed, or antecedent-mismatched
    clause payloads fail closed before numerical predicates are considered.
    """

    raw = dict(payload) if isinstance(payload, Mapping) else {}
    conformance_blockers: list[str] = []
    scientific_failures: list[str] = []
    keys = set(raw)
    if keys != _TOP_LEVEL_KEYS:
        conformance_blockers.append("native_bw_top_level_key_set_mismatch")
    if raw.get("schema_version") != BW_NATIVE_SCHEMA_VERSION:
        conformance_blockers.append("native_bw_schema_version_mismatch")
    if raw.get("producer_kind") != "native_simulator":
        conformance_blockers.append("native_bw_producer_is_not_native_simulator")
    if raw.get("source_kind") != "physical_source_generation":
        conformance_blockers.append("native_bw_source_is_fixture_replay_or_nonphysical")
    antecedent_hash = str(raw.get("antecedent_hash") or "")
    if not _strict_sha256(antecedent_hash):
        conformance_blockers.append("native_bw_antecedent_hash_missing_or_malformed")
    if _forbidden_assertion_paths(raw):
        conformance_blockers.append(
            "native_bw_contains_caller_asserted_pass_tier_or_receipt"
        )

    raw_clauses = raw.get("clauses")
    clauses = dict(raw_clauses) if isinstance(raw_clauses, Mapping) else {}
    if set(clauses) != set(REQUIRED_BW_CLAUSE_IDS):
        conformance_blockers.append(
            "native_finite_cap_clause_key_set_must_be_exactly_c1_through_c6"
        )

    merged_fields: dict[str, Any] = {}
    wrapper_status: dict[str, dict[str, Any]] = {}
    for clause_id in REQUIRED_BW_CLAUSE_IDS:
        row = dict(clauses.get(clause_id)) if isinstance(clauses.get(clause_id), Mapping) else {}
        row_blockers: list[str] = []
        if set(row) != _CLAUSE_KEYS:
            row_blockers.append("clause_wrapper_key_set_mismatch")
        if str(row.get("antecedent_hash") or "") != antecedent_hash or not _strict_sha256(
            row.get("antecedent_hash")
        ):
            row_blockers.append("clause_antecedent_hash_mismatch")
        primitive_fields = (
            dict(row.get("primitive_fields"))
            if isinstance(row.get("primitive_fields"), Mapping)
            else {}
        )
        required_fields = set(BW_PRIMITIVE_FIELD_CONTRACT[clause_id])
        if set(primitive_fields) != required_fields:
            row_blockers.append("primitive_field_key_set_mismatch")
        supplied_hash = str(row.get("primitive_artifact_hash") or "")
        computed_hash = None
        try:
            computed_hash = canonical_payload_hash(primitive_fields)
        except (TypeError, ValueError):
            row_blockers.append("primitive_fields_not_canonical_json")
        if not _strict_sha256(supplied_hash) or supplied_hash != computed_hash:
            row_blockers.append("primitive_artifact_hash_mismatch")
        for field_name, field_value in primitive_fields.items():
            if field_name in merged_fields and not _canonical_equal(
                merged_fields[field_name], field_value
            ):
                row_blockers.append(f"conflicting_shared_primitive:{field_name}")
            else:
                merged_fields[field_name] = field_value
        wrapper_status[clause_id] = {
            "wrapper_valid": not row_blockers,
            "blockers": sorted(set(row_blockers)),
            "primitive_artifact_hash": supplied_hash or None,
            "primitive_field_names": sorted(primitive_fields),
        }
        conformance_blockers.extend(f"{clause_id}:{item}" for item in row_blockers)

    raw_mgns1 = raw.get("mgns1")
    mgns1_wrapper = dict(raw_mgns1) if isinstance(raw_mgns1, Mapping) else {}
    mgns1_blockers: list[str] = []
    if set(mgns1_wrapper) != _CLAUSE_KEYS:
        mgns1_blockers.append("mgns1_wrapper_key_set_mismatch")
    if str(mgns1_wrapper.get("antecedent_hash") or "") != antecedent_hash or not _strict_sha256(
        mgns1_wrapper.get("antecedent_hash")
    ):
        mgns1_blockers.append("mgns1_antecedent_hash_mismatch")
    mgns1_fields = (
        dict(mgns1_wrapper.get("primitive_fields"))
        if isinstance(mgns1_wrapper.get("primitive_fields"), Mapping)
        else {}
    )
    if set(mgns1_fields) != set(MGNS1_REQUIRED_FIELDS):
        mgns1_blockers.append("mgns1_primitive_field_key_set_mismatch")
    supplied_mgns1_hash = str(mgns1_wrapper.get("primitive_artifact_hash") or "")
    computed_mgns1_hash = None
    try:
        computed_mgns1_hash = canonical_payload_hash(mgns1_fields)
    except (TypeError, ValueError):
        mgns1_blockers.append("mgns1_primitive_fields_not_canonical_json")
    if not _strict_sha256(supplied_mgns1_hash) or supplied_mgns1_hash != computed_mgns1_hash:
        mgns1_blockers.append("mgns1_primitive_artifact_hash_mismatch")
    conformance_blockers.extend(mgns1_blockers)

    issue308 = issue308_bw_certificate_report(
        {"BWRec_r": merged_fields, "MGNS1Rec_r": mgns1_fields}
    )
    recomputed: dict[str, dict[str, Any]] = {}
    for clause_id in REQUIRED_BW_CLAUSE_IDS:
        source_row = dict(issue308.get("clauses", {}).get(_ISSUE308_TO_BW[clause_id], {}))
        numerical_pass = source_row.get("passed") is True
        if clause_id == "C6":
            numerical_pass = numerical_pass and issue308.get("error_envelope", {}).get("passed") is True
        passed = bool(wrapper_status[clause_id]["wrapper_valid"] and numerical_pass)
        recomputed[clause_id] = {
            **wrapper_status[clause_id],
            "passed": passed,
            "numerical_predicates_recomputed": True,
            "numerical_evidence": source_row,
            **(
                {"error_envelope": issue308.get("error_envelope", {})}
                if clause_id == "C6"
                else {}
            ),
        }
        if not numerical_pass:
            scientific_failures.append(
                f"{clause_id}:recomputed_clause_predicates_failed"
            )

    mgns1_numerical_pass = issue308.get(MGNS1_CERTIFICATE_RECEIPT) is True
    mgns1_status = {
        "wrapper_valid": not mgns1_blockers,
        "blockers": sorted(set(mgns1_blockers)),
        "primitive_artifact_hash": supplied_mgns1_hash or None,
        "primitive_field_names": sorted(mgns1_fields),
        "passed": bool(not mgns1_blockers and mgns1_numerical_pass),
        "numerical_predicates_recomputed": True,
        "numerical_evidence": issue308.get("mgns1", {}),
    }
    if not mgns1_numerical_pass:
        scientific_failures.append("MGNS1:recomputed_certificate_predicates_failed")
    if issue308.get(BW_SAME_TOWER_INPUTS_RECEIPT) is not True:
        scientific_failures.append("PAIR:finite_cap_and_mgns1_not_independent_same_tower_inputs")

    conformance_blockers = list(dict.fromkeys(conformance_blockers))
    scientific_failures = list(dict.fromkeys(scientific_failures))
    blockers = [*conformance_blockers, *scientific_failures]
    conformance_receipt = not conformance_blockers
    receipt = bool(
        conformance_receipt
        and not scientific_failures
        and issue308.get(ISSUE_308_BW_CERTIFICATE_RECEIPT) is True
        and issue308.get(MGNS1_CERTIFICATE_RECEIPT) is True
        and issue308.get(BW_SAME_TOWER_INPUTS_RECEIPT) is True
        and issue308.get(SUPPORT_VISIBLE_BW_THEOREM_APPLICABLE_RECEIPT) is True
        and all(row["passed"] for row in recomputed.values())
    )
    return {
        "schema_version": "oph_bw_native_finite_cap_mgns1_verification_v2",
        "source_schema_version": raw.get("schema_version"),
        "antecedent_hash": antecedent_hash or None,
        "native_payload_conformance_receipt": conformance_receipt,
        "native_payload_receipt": receipt,
        "clauses": recomputed,
        "mgns1": mgns1_status,
        "finite_cap_bw_certificate_receipt": issue308.get(
            ISSUE_308_BW_CERTIFICATE_RECEIPT
        )
        is True,
        "mgns1_certificate_receipt": issue308.get(MGNS1_CERTIFICATE_RECEIPT) is True,
        "same_tower_inputs_receipt": issue308.get(BW_SAME_TOWER_INPUTS_RECEIPT) is True,
        "support_visible_bw_theorem_applicable": issue308.get(
            SUPPORT_VISIBLE_BW_THEOREM_APPLICABLE_RECEIPT
        )
        is True,
        "required_clause_ids": list(REQUIRED_BW_CLAUSE_IDS),
        "conformance_blockers": conformance_blockers,
        "scientific_failures": scientific_failures,
        "scientific_outcome": (
            "VALID_PASS"
            if receipt
            else "VALID_FAIL"
            if conformance_receipt
            else "INSTRUMENT_INVALID"
        ),
        "blockers": blockers,
        "ignored_producer_pass_flags": False,
        "recomputed_issue308_tier": issue308.get("tier"),
        "claim_boundary": (
            "C1-C6 certify only finite geometry, support flow, and normalization. The separate "
            "MGNS-1 payload certifies the modular algebra-state representation. The support-visible "
            "BW theorem is applicable only when both pass with distinct source artifacts on one "
            "tower. Repeated-rho diagnostics cannot satisfy MGNS-1."
        ),
    }


def native_bw01_bw08_report(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compatibility entry point for callers using the former function name."""

    return native_bw_pair_report(payload)


def _forbidden_assertion_paths(value: Any, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.lower()
            if (
                normalized in _FORBIDDEN_ASSERTION_KEYS
                or normalized.endswith("_passed")
                or normalized.endswith("_receipt")
            ):
                hits.append(f"{path}.{key}")
            hits.extend(_forbidden_assertion_paths(child, f"{path}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            hits.extend(_forbidden_assertion_paths(child, f"{path}[{index}]"))
    return hits


def _strict_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"sha256:[0-9a-f]{64}", str(value or "")))


def _canonical_equal(left: Any, right: Any) -> bool:
    try:
        return canonical_payload_hash(left) == canonical_payload_hash(right)
    except (TypeError, ValueError):
        return False


__all__ = [
    "BW_NATIVE_SCHEMA_VERSION",
    "BW_PRIMITIVE_FIELD_CONTRACT",
    "REQUIRED_BW_CLAUSE_IDS",
    "canonical_payload_hash",
    "native_bw_pair_report",
    "native_bw01_bw08_report",
]
