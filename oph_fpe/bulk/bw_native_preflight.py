"""Strict native BW01--BW08 payload verifier for the physical campaign.

The input is deliberately a bundle of primitive simulator artifacts, not a
bundle of producer supplied pass flags.  Each of the eight paper clauses has
an exact field contract and an independently hashed primitive payload.  The
numerical predicates are recomputed by the existing issue-308 verifier and
are exposed here under the unambiguous BW01--BW08 names.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from oph_fpe.bulk.bw_certificate_308 import issue308_bw_certificate_report
from oph_fpe.claims import ISSUE_308_BW_CERTIFICATE_RECEIPT

BW_NATIVE_SCHEMA_VERSION = "oph_bw_native_payload_v1"
REQUIRED_BW_CLAUSE_IDS = tuple(f"BW{index:02d}" for index in range(1, 9))

_TOP_LEVEL_KEYS = {
    "schema_version",
    "producer_kind",
    "source_kind",
    "antecedent_hash",
    "clauses",
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

# The overlap between BW01/BW02 (cap_normal) and BW03/BW05
# (support_covariance_residual_T) is intentional.  Duplicate primitive fields
# must be byte-equivalent across clauses or the aggregate is invalid.
BW_PRIMITIVE_FIELD_CONTRACT: dict[str, tuple[str, ...]] = {
    "BW01": (
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
    "BW02": (
        "cap_normal",
        "frame_p_minus",
        "frame_p_plus",
        "frame_boundary_residual",
        "frame_separation",
        "frame_ordering",
        "frame_orientation_witness",
    ),
    "BW03": (
        "cap_inclusion_matrix",
        "strict_inclusion_margin",
        "order_refinement_error",
        "support_isotony_failures",
        "support_separation_margin",
        "support_covariance_residual_T",
        "support_kernel_residual",
        "sector_scope",
    ),
    "BW04": (
        "test_tower_id",
        "test_tower_hash",
        "state_embedding_residual",
        "regularizer_eta",
        "physical_reference_trace_distance",
        "fixed_local_modular_bound_T",
    ),
    "BW05": (
        "mixed_gns_cauchy_residual_T",
        "negative_time_residual_T",
        "matrix_element_residual_T",
        "support_covariance_residual_T",
    ),
    "BW06": (
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
    "BW07": (
        "geometric_parameter_convention",
        "kms_strip_bound",
        "kms_residual_beta_2pi",
    ),
    "BW08": (
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
    "BW01": "C1_cap_normal_refinement",
    "BW02": "C2_bw_frame",
    "BW03": "C3_prime_support_visible_cap_net",
    "BW04": "C4_modular_reference_tower",
    "BW05": "C5_mixed_gns_and_support_covariance",
    "BW06": "C6_geometric_rigidity",
    "BW07": "C7_geometric_2pi_kms",
    "BW08": "C8_wrong_normalization_and_nontriviality",
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


def native_bw01_bw08_report(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Recompute the eight native BW clauses from exact primitive payloads.

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
            "native_bw_clause_key_set_must_be_exactly_bw01_through_bw08"
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

    issue308 = issue308_bw_certificate_report({"BWRec_r": merged_fields})
    recomputed: dict[str, dict[str, Any]] = {}
    for clause_id in REQUIRED_BW_CLAUSE_IDS:
        source_row = dict(issue308.get("clauses", {}).get(_ISSUE308_TO_BW[clause_id], {}))
        numerical_pass = source_row.get("passed") is True
        if clause_id == "BW08":
            numerical_pass = numerical_pass and issue308.get("error_envelope", {}).get("passed") is True
        passed = bool(wrapper_status[clause_id]["wrapper_valid"] and numerical_pass)
        recomputed[clause_id] = {
            **wrapper_status[clause_id],
            "passed": passed,
            "numerical_predicates_recomputed": True,
            "numerical_evidence": source_row,
            **(
                {"error_envelope": issue308.get("error_envelope", {})}
                if clause_id == "BW08"
                else {}
            ),
        }
        if not numerical_pass:
            scientific_failures.append(
                f"{clause_id}:recomputed_clause_predicates_failed"
            )

    conformance_blockers = list(dict.fromkeys(conformance_blockers))
    scientific_failures = list(dict.fromkeys(scientific_failures))
    blockers = [*conformance_blockers, *scientific_failures]
    conformance_receipt = not conformance_blockers
    receipt = bool(
        conformance_receipt
        and not scientific_failures
        and issue308.get(ISSUE_308_BW_CERTIFICATE_RECEIPT) is True
        and all(row["passed"] for row in recomputed.values())
    )
    return {
        "schema_version": "oph_bw_native_verification_v1",
        "source_schema_version": raw.get("schema_version"),
        "antecedent_hash": antecedent_hash or None,
        "native_payload_conformance_receipt": conformance_receipt,
        "native_payload_receipt": receipt,
        "clauses": recomputed,
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
            "BW01--BW08 are recomputed from exact, hashed native primitive artifacts. "
            "Producer pass flags, fixtures, replay, missing clauses, extra clauses, and "
            "antecedent mismatches cannot satisfy the conformance gate. A complete, "
            "well-provenanced payload whose recomputed clause predicate is false is a "
            "VALID_FAIL scientific outcome, not an invalid instrument."
        ),
    }


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
    "native_bw01_bw08_report",
]
