"""Fail-closed W/Z claim-lane requirements aggregator.

The aggregator is a consumer, not a scientific producer.  It accepts one
on-disk production-envelope bundle manifest, invokes the production verifier,
and evaluates the exact external/native conjunctions without trusting caller
booleans or stored reports.  The current generic production verifier is P0
inventory infrastructure and deliberately cannot replay scientific claims;
therefore its envelopes remain non-promoting here.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


REQUIREMENTS_REPORT_SCHEMA = "oph.physical-wz.requirements-audit.v1"
REQUIREMENTS_REPORT_ARTIFACT_TYPE = "OPH_PHYSICAL_WZ_REQUIREMENTS_AUDIT"
EXPOSURE_CLASSIFICATION = "post_exposure_validation"
ADMITTED_SCIENTIFIC_EVIDENCE_CLASS = "INDEPENDENT_SCIENTIFIC_REPLAY"

# Visualization/control packets may use these labels, but no spelling in this
# family is scientific evidence.  This is intentionally broader than the
# current ``DEMO_ASSUMPTION`` mode so a future force-receipt UI cannot acquire
# evidentiary meaning by choosing a nearby label.
NONSCIENTIFIC_LABEL_TOKENS = (
    "ASSUMPTION",
    "DEMO",
    "FORCED",
    "FROZEN_TARGET",
    "MOCK",
    "PLACEHOLDER",
    "SYNTHETIC",
    "TARGET_VALUE",
    "VISUALIZATION",
    "WZH0",
)


class ClaimLane(str, Enum):
    OPH_CHART_ONLY = "OPH_CHART_ONLY"
    EXTERNAL_SM_EFT_VALIDATION = "EXTERNAL_SM_EFT_VALIDATION"
    OPH_NATIVE_DIMENSIONLESS = "OPH_NATIVE_DIMENSIONLESS"
    OPH_NATIVE_PHYSICAL = "OPH_NATIVE_PHYSICAL"
    TARGET_COMPARISON_ONLY = "TARGET_COMPARISON_ONLY"


class ClaimScope(str, Enum):
    W_ONLY = "W_ONLY"
    Z_ONLY = "Z_ONLY"
    WZ = "WZ"
    WZH = "WZH"


class RequirementStatus(str, Enum):
    PASS = "PASS"
    OPEN = "OPEN"
    UNRESOLVED = "UNRESOLVED"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


CHART_ONLY_GATES = ("CHART_COORDINATES_1",)

EXTERNAL_COMMON_GATES = (
    "IMPORTED_MINKOWSKI_CHART",
    "IMPORTED_SM_EFT_ACTION",
    "ANOMALY_AND_PERTURBATIVE_BRST_1",
    "FULL_YUKAWA_1",
    "EFT_MATCHING_1",
    "RULE_EQUIVALENCE_1",
    "RENORMALIZATION_ST_1",
    "FJ_DIRECT_1",
    "FJ_CONVERTED_1",
    "FJ_EQUIVALENCE_1",
    "GENERAL_GAUGE_BRST_1",
    "WARD_ST_NIELSEN_1",
    "POLE_SERIES_1",
    "ANALYTIC_CONTINUATION_1",
    "RUNTIME_SUBJECT_BINDING_1",
    "TARGET_FIREWALL_1",
)

NATIVE_DIMENSIONLESS_COMMON_GATES = (
    "EVENT_GEOMETRY_OR_EXPLICIT_NATIVE_LOCAL_CHART",
    "OPH_SM_EFT_ACTION_1",
    "OPH_SOURCE_PARAMETER_JET_1",
    "OPH_SOURCE_TO_SM_MATCHING_1",
    "FULL_YUKAWA_1",
    "SOURCE_LAW_1",
    "SOURCE_COVARIANCE_1",
    "NO_TARGET_ANCESTRY_1",
    # Source-independent QFT stages are retained, but every envelope must be
    # regenerated in the selected native lane and bound to its native family.
    "ANOMALY_AND_PERTURBATIVE_BRST_1",
    "RULE_EQUIVALENCE_1",
    "RENORMALIZATION_ST_1",
    "FJ_DIRECT_1",
    "FJ_CONVERTED_1",
    "FJ_EQUIVALENCE_1",
    "GENERAL_GAUGE_BRST_1",
    "WARD_ST_NIELSEN_1",
    "POLE_SERIES_1",
    "ANALYTIC_CONTINUATION_1",
    "RUNTIME_SUBJECT_BINDING_1",
    "TARGET_FIREWALL_1",
)

NATIVE_PHYSICAL_EXTRA_GATES = (
    "SOURCE_CLOCK_1",
    "POLE_UNCERTAINTY_FREEZE_1",
)

W_POLE_GATE = "PHYSICAL_CURRENT_POLE_W_1"
Z_POLE_GATE = "PHYSICAL_CURRENT_POLE_Z_1"
H_POLE_GATE = "PHYSICAL_SCALAR_POLE_H_1"

TARGET_COMPARISON_GATES = (
    "TARGET_FIREWALL_1",
    "TARGET_COMPARISON_IMMUTABILITY_1",
)

ALL_GATE_IDS = tuple(
    dict.fromkeys(
        (
            *CHART_ONLY_GATES,
            *EXTERNAL_COMMON_GATES,
            *NATIVE_DIMENSIONLESS_COMMON_GATES,
            *NATIVE_PHYSICAL_EXTRA_GATES,
            W_POLE_GATE,
            Z_POLE_GATE,
            H_POLE_GATE,
            *TARGET_COMPARISON_GATES,
        )
    )
)

FAMILY_FIELDS = (
    "source_root_hash",
    "branch_id",
    "freeze_id",
    "action_ast_hash",
    "field_census_hash",
    "scheme_hash",
    "fj_convention_hash",
    "term_mask_hash",
    "analytic_sheet_hash",
    "units_basis_hash",
)

PHYSICAL_RECEIPT_KEYS = (
    "EXTERNAL_W_POLE_VALIDATION_RECEIPT",
    "EXTERNAL_Z_POLE_VALIDATION_RECEIPT",
    "EXTERNAL_H_POLE_VALIDATION_RECEIPT",
    "EXTERNAL_WZ_QFT_RECEIPT",
    "OPH_NATIVE_DIMENSIONLESS_W_RECEIPT",
    "OPH_NATIVE_DIMENSIONLESS_Z_RECEIPT",
    "OPH_NATIVE_DIMENSIONLESS_H_RECEIPT",
    "OPH_NATIVE_DIMENSIONLESS_RECEIPT",
    "OPH_NATIVE_PHYSICAL_W_RECEIPT",
    "OPH_NATIVE_PHYSICAL_Z_RECEIPT",
    "OPH_NATIVE_PHYSICAL_H_RECEIPT",
    "OPH_NATIVE_PHYSICAL_RECEIPT",
    "PHYSICAL_WZ_PROMOTION_ALLOWED",
)


def verify_physical_wz_requirements(
    production_bundle_manifest: str | Path | None = None,
    *,
    lane: ClaimLane | str = ClaimLane.OPH_CHART_ONLY,
    scope: ClaimScope | str = ClaimScope.WZ,
) -> dict[str, Any]:
    """Recompute one exclusive W/Z lane audit from a production manifest.

    ``production_bundle_manifest`` is either absent or an on-disk path.  A
    mapping, stored verification report, or gate dictionary is rejected.
    """

    parsed_lane, lane_blocker = _parse_enum(lane, ClaimLane, "claim_lane")
    parsed_scope, scope_blocker = _parse_enum(scope, ClaimScope, "claim_scope")
    input_blockers = [
        blocker for blocker in (lane_blocker, scope_blocker) if blocker is not None
    ]
    if parsed_lane is None or parsed_scope is None:
        return _invalid_report(
            lane=parsed_lane,
            scope=parsed_scope,
            blockers=input_blockers,
        )

    production_report: dict[str, Any] | None = None
    production_valid = False
    production_supplied = production_bundle_manifest is not None
    if production_bundle_manifest is not None:
        if isinstance(production_bundle_manifest, Mapping):
            input_blockers.append(
                "production_bundle_manifest_must_be_an_on_disk_path_not_a_mapping"
            )
        else:
            production_report = _verify_production_manifest(
                production_bundle_manifest
            )
            marker_check = _production_report_marker_check(production_report)
            if marker_check:
                input_blockers.extend(marker_check)
            else:
                production_valid = bool(
                    production_report.get("inventory_replay_passed") is True
                )
                if not production_valid:
                    input_blockers.extend(
                        str(value)
                        for value in production_report.get("blockers") or ()
                    )

    normalized_rows, row_blockers = _normalize_production_rows(
        production_report if production_valid else None
    )
    input_blockers.extend(row_blockers)
    required_gates = _required_gate_ids(parsed_lane, parsed_scope)
    family = _family_audit(
        normalized_rows,
        required_gate_ids=required_gates,
        lane=parsed_lane,
        scope=parsed_scope,
    )
    input_blockers.extend(family["blockers"])

    gate_rows: dict[str, dict[str, Any]] = {}
    for gate_id in ALL_GATE_IDS:
        if gate_id not in required_gates:
            gate_rows[gate_id] = {
                "status": RequirementStatus.NOT_APPLICABLE.value,
                "passed": False,
                "required": False,
                "envelope_ids": [],
                "blockers": [],
            }
            continue
        gate_rows[gate_id] = _evaluate_gate(
            gate_id,
            normalized_rows=normalized_rows,
            lane=parsed_lane,
            scope=parsed_scope,
            production_supplied=production_supplied,
            production_valid=production_valid,
            family_passed=family["passed"],
        )

    required_statuses = [gate_rows[gate_id]["status"] for gate_id in required_gates]
    lane_status = _conjunction_status(required_statuses)
    if input_blockers:
        lane_status = RequirementStatus.FAIL.value

    species = _species_requirements(parsed_scope)
    receipts = _compute_receipts(
        lane=parsed_lane,
        scope=parsed_scope,
        gates=gate_rows,
        family_passed=family["passed"],
    )
    # An ordinary OPEN downstream dependency may retain useful partial
    # receipts.  An integrity error may not: mixed or malformed evidence must
    # not leave scientific receipt bits true in a FAIL report.
    if input_blockers:
        receipts = {key: False for key in PHYSICAL_RECEIPT_KEYS}
    # This protects the current P0 production verifier boundary even if a
    # producer places PASS-like strings in its envelopes.  Only a future strict
    # scientific replay can make the per-gate evaluator return PASS.
    promotion_allowed = bool(
        parsed_lane is ClaimLane.OPH_NATIVE_PHYSICAL
        and lane_status == RequirementStatus.PASS.value
        and receipts["OPH_NATIVE_PHYSICAL_RECEIPT"]
    )
    receipts["PHYSICAL_WZ_PROMOTION_ALLOWED"] = promotion_allowed
    lane_receipts = {
        candidate.value: False for candidate in ClaimLane
    }
    if lane_status == RequirementStatus.PASS.value:
        lane_receipts[parsed_lane.value] = True

    diagnostic_ids = sorted(
        row["envelope_id"]
        for row in normalized_rows
        if row["diagnostic"] is True
    )
    blockers = sorted(
        set(
            input_blockers
            + [
                blocker
                for gate_id in required_gates
                for blocker in gate_rows[gate_id]["blockers"]
            ]
        )
    )
    open_requirements = [
        gate_id
        for gate_id in required_gates
        if gate_rows[gate_id]["status"]
        in {RequirementStatus.OPEN.value, RequirementStatus.UNRESOLVED.value}
    ]
    report = {
        "schema": REQUIREMENTS_REPORT_SCHEMA,
        "artifact_type": REQUIREMENTS_REPORT_ARTIFACT_TYPE,
        "selected_lane": parsed_lane.value,
        "selected_scope": parsed_scope.value,
        "exclusive_lane_count": 1,
        "allowed_lanes": [value.value for value in ClaimLane],
        "allowed_scopes": [value.value for value in ClaimScope],
        "exposure_classification": EXPOSURE_CLASSIFICATION,
        "prospective_prediction": False,
        "required_species": species,
        "required_gate_ids": list(required_gates),
        "gates": gate_rows,
        "family_audit": family,
        "production_bundle": {
            "supplied": production_supplied,
            "strict_inventory_replayed": production_valid,
            "manifest_path": (
                None
                if production_report is None
                else production_report.get("manifest_path")
            ),
            "manifest_sha256": (
                None
                if production_report is None
                else production_report.get("manifest_sha256")
            ),
            "bundle_id": (
                None
                if production_report is None
                else production_report.get("bundle_id")
            ),
            "scientific_replay_passed": bool(
                production_report is not None
                and production_report.get("scientific_replay_passed") is True
            ),
            "promotion_allowed": bool(
                production_report is not None
                and production_report.get("promotion_allowed") is True
            ),
        },
        "diagnostic_envelope_ids": diagnostic_ids,
        "WZH0_SYNTHETIC_CONTROL_NONPROMOTING": True,
        "SCALE_CAMPAIGN_ALLOWED": False,
        "scale_authorization": False,
        "lane_receipts": lane_receipts,
        "receipts": receipts,
        **receipts,
        "status": lane_status,
        "passed": lane_status == RequirementStatus.PASS.value,
        "promotion_allowed": promotion_allowed,
        "open_requirements": open_requirements,
        "blockers": blockers,
        "claim_boundary": (
            "This report only aggregates strict production-envelope replays. "
            "The current generic envelope verifier is inventory-only, WZH0 is "
            "nonpromoting, DEMO_ASSUMPTION/visualization envelopes are never "
            "scientific evidence, external validation never promotes a native "
            "claim, and all W/Z targets are classified as post-exposure validation."
        ),
    }
    return report


def _verify_production_manifest(path: str | Path) -> dict[str, Any]:
    try:
        from oph_fpe.evidence.production_envelope import (
            verify_production_bundle_manifest,
        )

        result = verify_production_bundle_manifest(path)
        return dict(result) if isinstance(result, Mapping) else {
            "schema": None,
            "artifact_type": None,
            "inventory_replay_passed": False,
            "blockers": ["production_verifier_returned_non_mapping"],
        }
    except (ImportError, OSError, TypeError, ValueError) as exc:
        return {
            "schema": None,
            "artifact_type": None,
            "inventory_replay_passed": False,
            "blockers": [
                f"production_manifest_replay_failed:{type(exc).__name__}:{exc}"
            ],
        }


def _production_report_marker_check(report: Mapping[str, Any]) -> list[str]:
    try:
        from oph_fpe.evidence.production_envelope import (
            PRODUCTION_BUNDLE_REPORT_ARTIFACT_TYPE,
            PRODUCTION_BUNDLE_REPORT_SCHEMA,
        )
    except ImportError:
        return ["production_envelope_verifier_unavailable"]
    blockers: list[str] = []
    if report.get("schema") != PRODUCTION_BUNDLE_REPORT_SCHEMA:
        blockers.append("production_report_schema_mismatch")
    if report.get("artifact_type") != PRODUCTION_BUNDLE_REPORT_ARTIFACT_TYPE:
        blockers.append("production_report_artifact_type_mismatch")
    return blockers


def _normalize_production_rows(
    report: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if report is None:
        return [], []
    raw = report.get("envelopes")
    if not isinstance(raw, Mapping):
        return [], ["production_report_envelopes_must_be_mapping"]
    order = report.get("envelope_order")
    if not isinstance(order, list) or not all(isinstance(value, str) for value in order):
        return [], ["production_report_envelope_order_invalid"]
    if set(order) != set(raw) or len(order) != len(set(order)):
        return [], ["production_report_envelope_order_mismatch"]
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for envelope_id in order:
        value = raw.get(envelope_id)
        if not isinstance(value, Mapping):
            blockers.append(f"production_envelope_row_not_mapping:{envelope_id}")
            continue
        normalized, row_blockers = _normalize_one_envelope(envelope_id, value)
        blockers.extend(row_blockers)
        if normalized is not None:
            rows.append(normalized)
    return rows, blockers


def _normalize_one_envelope(
    envelope_id: str,
    value: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Normalize the frozen production report row without trusting its claims."""

    shared = value.get("shared_contract_hashes")
    if not isinstance(shared, Mapping):
        return None, [f"production_envelope_shared_contract_hashes_invalid:{envelope_id}"]
    artifact_id = _first_text(value.get("artifact_id"))
    stage_id = _first_text(value.get("stage_id"))
    lane = _first_text(value.get("claim_lane"))
    scope = _first_text(value.get("claim_scope"))
    profile = _first_text(value.get("profile"))
    receipt_type = _first_text(value.get("receipt_type"))
    family = {
        "source_root_hash": _first_text(value.get("source_root_hash")),
        "branch_id": _first_text(value.get("branch_id")),
        "freeze_id": _first_text(value.get("freeze_id")),
        **{
            field: _first_text(shared.get(field))
            for field in FAMILY_FIELDS
            if field not in {"source_root_hash", "branch_id", "freeze_id"}
        },
    }
    blockers: list[str] = []
    required_identity_fields = [
        ("artifact_id", artifact_id),
        ("stage_id", stage_id),
        ("claim_lane", lane),
        ("claim_scope", scope),
        ("profile", profile),
        ("receipt_type", receipt_type),
        ("source_root_hash", family["source_root_hash"]),
        ("branch_id", family["branch_id"]),
        ("freeze_id", family["freeze_id"]),
    ]
    if profile == "WZ_SOURCE_TO_POLE":
        required_identity_fields.extend(
            (field, family[field])
            for field in FAMILY_FIELDS
            if field not in {"source_root_hash", "branch_id", "freeze_id"}
        )
    for field_name, field_value in required_identity_fields:
        if field_value is None:
            blockers.append(f"production_envelope_missing_{field_name}:{envelope_id}")
    if blockers:
        return None, blockers
    if artifact_id != envelope_id:
        return None, [f"production_envelope_key_artifact_id_mismatch:{envelope_id}"]
    status = _first_text(value.get("producer_status"))
    evidence_class = _first_text(value.get("evidence_class"))
    diagnostic_label = f"{profile}:{receipt_type}:{evidence_class}:{status}".upper()
    diagnostic = bool(
        profile is not None
        and any(
            token in diagnostic_label
            for token in ("DIAGNOSTIC", *NONSCIENTIFIC_LABEL_TOKENS)
        )
    )
    raw_blockers = value.get("blockers")
    if not isinstance(raw_blockers, list) or any(
        not isinstance(item, str) for item in raw_blockers
    ):
        return None, [f"production_envelope_blockers_invalid:{envelope_id}"]
    normalized = {
        "envelope_id": envelope_id,
        "stage_id": stage_id,
        "claim_lane": lane,
        "claim_scope": scope,
        "profile": profile,
        "receipt_type": receipt_type,
        "status": status,
        "diagnostic": diagnostic,
        "inventory_replay_passed": value.get("inventory_replay_passed") is True,
        "scientific_replay_passed": value.get("scientific_replay_passed") is True,
        "promotion_allowed": value.get("promotion_allowed") is True,
        "producer_status_trusted": value.get("producer_status_trusted") is True,
        "evidence_class": evidence_class,
        "family": family,
        "blockers": list(raw_blockers),
    }
    return normalized, []


def _family_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    required_gate_ids: Sequence[str],
    lane: ClaimLane,
    scope: ClaimScope,
) -> dict[str, Any]:
    relevant = [row for row in rows if row.get("stage_id") in required_gate_ids]
    blockers: list[str] = []
    baseline: dict[str, Any] | None = None
    # One manifest is one exclusive W/Z claim lane and scope.  Extra rows from
    # another lane/scope/family are an integrity failure even when their stages
    # are not required by the selected scope.
    for row in rows:
        if row.get("stage_id") not in ALL_GATE_IDS:
            blockers.append(f"unknown_stage_envelope:{row.get('envelope_id')}")
        if row.get("claim_lane") != lane.value:
            blockers.append(f"wrong_lane_envelope:{row.get('envelope_id')}")
        if row.get("claim_scope") != scope.value:
            blockers.append(f"wrong_scope_envelope:{row.get('envelope_id')}")
        family = row.get("family")
        if not isinstance(family, Mapping):
            blockers.append(f"missing_family:{row.get('envelope_id')}")
            continue
        if baseline is None:
            baseline = dict(family)
        elif dict(family) != baseline:
            blockers.append(f"mixed_receipt_family:{row.get('envelope_id')}")
    return {
        "passed": not blockers,
        "relevant_envelope_count": len(relevant),
        "baseline": baseline,
        "source_family_immutable": bool(relevant and not blockers),
        "blockers": sorted(set(blockers)),
    }


def _evaluate_gate(
    gate_id: str,
    *,
    normalized_rows: Sequence[Mapping[str, Any]],
    lane: ClaimLane,
    scope: ClaimScope,
    production_supplied: bool,
    production_valid: bool,
    family_passed: bool,
) -> dict[str, Any]:
    matches = [row for row in normalized_rows if row.get("stage_id") == gate_id]
    blockers: list[str] = []
    if not production_supplied:
        status = RequirementStatus.OPEN
    elif not production_valid:
        status = RequirementStatus.FAIL
        blockers.append("production_bundle_inventory_invalid")
    elif len(matches) == 0:
        status = RequirementStatus.OPEN
    elif len(matches) > 1:
        status = RequirementStatus.FAIL
        blockers.append("duplicate_stage_envelopes")
    else:
        row = matches[0]
        if row.get("claim_lane") != lane.value:
            status = RequirementStatus.FAIL
            blockers.append("envelope_claim_lane_mismatch")
        elif row.get("claim_scope") != scope.value:
            status = RequirementStatus.FAIL
            blockers.append("envelope_claim_scope_mismatch")
        elif not family_passed:
            status = RequirementStatus.FAIL
            blockers.append("envelope_family_mismatch")
        elif row.get("inventory_replay_passed") is not True:
            status = RequirementStatus.FAIL
            blockers.append("envelope_inventory_replay_failed")
        elif row.get("profile") != "WZ_SOURCE_TO_POLE":
            status = RequirementStatus.UNRESOLVED
            blockers.append("non_wz_production_profile_is_nonpromoting")
        elif row.get("blockers"):
            status = RequirementStatus.FAIL
            blockers.append("scientific_envelope_contains_blockers")
        elif row.get("diagnostic") is True:
            status = RequirementStatus.UNRESOLVED
            blockers.append("diagnostic_envelope_is_nonpromoting")
        elif row.get("status") == RequirementStatus.FAIL.value:
            status = RequirementStatus.FAIL
            blockers.append("scientific_envelope_declares_failure")
        elif row.get("status") == RequirementStatus.UNRESOLVED.value:
            status = RequirementStatus.UNRESOLVED
        elif (
            row.get("status") == RequirementStatus.PASS.value
            and row.get("scientific_replay_passed") is True
            and row.get("promotion_allowed") is True
            and row.get("producer_status_trusted") is True
            and row.get("evidence_class") == ADMITTED_SCIENTIFIC_EVIDENCE_CLASS
        ):
            status = RequirementStatus.PASS
        else:
            # P0 inventory evidence, including a producer-authored PASS, is an
            # open scientific requirement rather than a physical pass.
            status = RequirementStatus.OPEN
            blockers.append("independent_scientific_replay_not_available")
    return {
        "status": status.value,
        "passed": status is RequirementStatus.PASS,
        "required": True,
        "envelope_ids": [str(row.get("envelope_id")) for row in matches],
        "blockers": blockers,
    }


def _required_gate_ids(lane: ClaimLane, scope: ClaimScope) -> tuple[str, ...]:
    if lane is ClaimLane.OPH_CHART_ONLY:
        return CHART_ONLY_GATES
    if lane is ClaimLane.TARGET_COMPARISON_ONLY:
        return TARGET_COMPARISON_GATES
    if lane is ClaimLane.EXTERNAL_SM_EFT_VALIDATION:
        common = EXTERNAL_COMMON_GATES
    elif lane is ClaimLane.OPH_NATIVE_DIMENSIONLESS:
        common = NATIVE_DIMENSIONLESS_COMMON_GATES
    else:
        common = (
            *NATIVE_DIMENSIONLESS_COMMON_GATES,
            *NATIVE_PHYSICAL_EXTRA_GATES,
        )
    return tuple(dict.fromkeys((*common, *_scope_pole_gates(scope))))


def _scope_pole_gates(scope: ClaimScope) -> tuple[str, ...]:
    if scope is ClaimScope.W_ONLY:
        return (W_POLE_GATE,)
    if scope is ClaimScope.Z_ONLY:
        return (Z_POLE_GATE,)
    if scope is ClaimScope.WZ:
        return (W_POLE_GATE, Z_POLE_GATE)
    return (W_POLE_GATE, Z_POLE_GATE, H_POLE_GATE)


def _species_requirements(scope: ClaimScope) -> dict[str, bool]:
    return {
        "W": scope in {ClaimScope.W_ONLY, ClaimScope.WZ, ClaimScope.WZH},
        "Z": scope in {ClaimScope.Z_ONLY, ClaimScope.WZ, ClaimScope.WZH},
        "H": scope is ClaimScope.WZH,
    }


def _compute_receipts(
    *,
    lane: ClaimLane,
    scope: ClaimScope,
    gates: Mapping[str, Mapping[str, Any]],
    family_passed: bool,
) -> dict[str, bool]:
    receipts = {key: False for key in PHYSICAL_RECEIPT_KEYS}
    species = _species_requirements(scope)
    if lane is ClaimLane.EXTERNAL_SM_EFT_VALIDATION:
        common = _all_pass(gates, EXTERNAL_COMMON_GATES) and family_passed
        receipts["EXTERNAL_W_POLE_VALIDATION_RECEIPT"] = bool(
            species["W"] and common and gates[W_POLE_GATE]["passed"] is True
        )
        receipts["EXTERNAL_Z_POLE_VALIDATION_RECEIPT"] = bool(
            species["Z"] and common and gates[Z_POLE_GATE]["passed"] is True
        )
        receipts["EXTERNAL_H_POLE_VALIDATION_RECEIPT"] = bool(
            species["H"] and common and gates[H_POLE_GATE]["passed"] is True
        )
        receipts["EXTERNAL_WZ_QFT_RECEIPT"] = _scope_receipt_conjunction(
            receipts,
            species,
            prefix="EXTERNAL",
        )
    if lane in {
        ClaimLane.OPH_NATIVE_DIMENSIONLESS,
        ClaimLane.OPH_NATIVE_PHYSICAL,
    }:
        common = _all_pass(gates, NATIVE_DIMENSIONLESS_COMMON_GATES) and family_passed
        for boson, gate_id in (("W", W_POLE_GATE), ("Z", Z_POLE_GATE), ("H", H_POLE_GATE)):
            receipts[f"OPH_NATIVE_DIMENSIONLESS_{boson}_RECEIPT"] = bool(
                species[boson] and common and gates[gate_id]["passed"] is True
            )
        receipts["OPH_NATIVE_DIMENSIONLESS_RECEIPT"] = _scope_receipt_conjunction(
            receipts,
            species,
            prefix="OPH_NATIVE_DIMENSIONLESS",
        )
    if lane is ClaimLane.OPH_NATIVE_PHYSICAL:
        units = _all_pass(gates, NATIVE_PHYSICAL_EXTRA_GATES)
        for boson in ("W", "Z", "H"):
            receipts[f"OPH_NATIVE_PHYSICAL_{boson}_RECEIPT"] = bool(
                species[boson]
                and units
                and receipts[f"OPH_NATIVE_DIMENSIONLESS_{boson}_RECEIPT"]
            )
        receipts["OPH_NATIVE_PHYSICAL_RECEIPT"] = _scope_receipt_conjunction(
            receipts,
            species,
            prefix="OPH_NATIVE_PHYSICAL",
        )
    return receipts


def _scope_receipt_conjunction(
    receipts: Mapping[str, bool],
    species: Mapping[str, bool],
    *,
    prefix: str,
) -> bool:
    return all(
        not required or receipts.get(f"{prefix}_{boson}_POLE_VALIDATION_RECEIPT", False)
        if prefix == "EXTERNAL"
        else not required or receipts.get(f"{prefix}_{boson}_RECEIPT", False)
        for boson, required in species.items()
    )


def _all_pass(
    gates: Mapping[str, Mapping[str, Any]], gate_ids: Sequence[str]
) -> bool:
    return all(gates[gate_id].get("status") == RequirementStatus.PASS.value for gate_id in gate_ids)


def _conjunction_status(statuses: Sequence[str]) -> str:
    if any(value == RequirementStatus.FAIL.value for value in statuses):
        return RequirementStatus.FAIL.value
    if any(value == RequirementStatus.UNRESOLVED.value for value in statuses):
        return RequirementStatus.UNRESOLVED.value
    if any(value == RequirementStatus.OPEN.value for value in statuses):
        return RequirementStatus.OPEN.value
    if statuses and all(value == RequirementStatus.PASS.value for value in statuses):
        return RequirementStatus.PASS.value
    return RequirementStatus.NOT_APPLICABLE.value


def _parse_enum(
    value: Any,
    enum_type: type[Enum],
    field_name: str,
) -> tuple[Any | None, str | None]:
    if isinstance(value, enum_type):
        return value, None
    if isinstance(value, str):
        try:
            return enum_type(value), None
        except ValueError:
            pass
    return None, f"{field_name}_must_select_exactly_one_allowed_value"


def _first_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _invalid_report(
    *,
    lane: ClaimLane | None,
    scope: ClaimScope | None,
    blockers: Sequence[str],
) -> dict[str, Any]:
    receipts = {key: False for key in PHYSICAL_RECEIPT_KEYS}
    return {
        "schema": REQUIREMENTS_REPORT_SCHEMA,
        "artifact_type": REQUIREMENTS_REPORT_ARTIFACT_TYPE,
        "selected_lane": None if lane is None else lane.value,
        "selected_scope": None if scope is None else scope.value,
        "exclusive_lane_count": 0 if lane is None else 1,
        "allowed_lanes": [value.value for value in ClaimLane],
        "allowed_scopes": [value.value for value in ClaimScope],
        "exposure_classification": EXPOSURE_CLASSIFICATION,
        "prospective_prediction": False,
        "SCALE_CAMPAIGN_ALLOWED": False,
        "scale_authorization": False,
        "gates": {
            gate_id: {
                "status": RequirementStatus.NOT_APPLICABLE.value,
                "passed": False,
                "required": False,
                "envelope_ids": [],
                "blockers": [],
            }
            for gate_id in ALL_GATE_IDS
        },
        "receipts": receipts,
        **receipts,
        "status": RequirementStatus.FAIL.value,
        "passed": False,
        "promotion_allowed": False,
        "blockers": sorted(set(blockers)),
        "claim_boundary": "Invalid lane or scope selection; no claim is admitted.",
    }
