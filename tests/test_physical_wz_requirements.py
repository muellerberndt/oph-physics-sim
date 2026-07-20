from __future__ import annotations

from typing import Any

import pytest

from oph_fpe.bosons import physical_wz_requirements as wz


def _family(seed: str = "a") -> dict[str, str]:
    digest = f"sha256:{seed * 64}"
    return {
        "source_root_hash": digest,
        "branch_id": f"branch-{seed}",
        "freeze_id": f"freeze-{seed}",
        "action_ast_hash": digest,
        "field_census_hash": digest,
        "scheme_hash": digest,
        "fj_convention_hash": digest,
        "term_mask_hash": digest,
        "analytic_sheet_hash": digest,
        "units_basis_hash": digest,
    }


def _envelope_row(
    stage_id: str,
    *,
    lane: wz.ClaimLane,
    scope: wz.ClaimScope,
    family: dict[str, str] | None = None,
    profile: str = "WZ_SOURCE_TO_POLE",
    receipt_type: str = "PHYSICAL_WZ_STAGE_RECEIPT",
    status: str = "PASS",
    scientific: bool = True,
    promotion: bool = True,
) -> dict[str, Any]:
    resolved_family = family or _family()
    shared_fields = {
        field: resolved_family[field]
        for field in wz.FAMILY_FIELDS
        if field not in {"source_root_hash", "branch_id", "freeze_id"}
    }
    return {
        "artifact_id": "assigned-after-construction",
        "stage_id": stage_id,
        "profile": profile,
        "receipt_type": receipt_type,
        "claim_lane": lane.value,
        "claim_scope": scope.value,
        "source_root_hash": resolved_family["source_root_hash"],
        "branch_id": resolved_family["branch_id"],
        "freeze_id": resolved_family["freeze_id"],
        "producer_status": status,
        "producer_status_trusted": True,
        "shared_contract_hashes": shared_fields,
        "inventory_replay_passed": True,
        "scientific_replay_passed": scientific,
        "promotion_allowed": promotion,
        "evidence_class": wz.ADMITTED_SCIENTIFIC_EVIDENCE_CLASS,
        "blockers": [],
    }


def _production_report(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for envelope_id, row in rows.items():
        row["artifact_id"] = envelope_id
    return {
        "schema": "test-production-schema",
        "artifact_type": "test-production-artifact",
        "manifest_path": "manifest.json",
        "manifest_sha256": "sha256:" + "f" * 64,
        "bundle_id": "test-production-bundle",
        "envelope_order": list(rows),
        "envelopes": rows,
        "inventory_replay_passed": True,
        "scientific_replay_passed": all(
            row["scientific_replay_passed"] for row in rows.values()
        ),
        "promotion_allowed": all(row["promotion_allowed"] for row in rows.values()),
        "blockers": [],
    }


def _install_verified_report(
    monkeypatch: pytest.MonkeyPatch,
    report: dict[str, Any],
) -> None:
    monkeypatch.setattr(wz, "_verify_production_manifest", lambda _path: report)
    monkeypatch.setattr(wz, "_production_report_marker_check", lambda _report: [])


def _passing_report_for(
    lane: wz.ClaimLane,
    scope: wz.ClaimScope,
    *,
    omit: set[str] | None = None,
) -> dict[str, Any]:
    required = wz._required_gate_ids(lane, scope)
    rows = {
        f"envelope-{index:02d}": _envelope_row(
            gate_id,
            lane=lane,
            scope=scope,
        )
        for index, gate_id in enumerate(required)
        if gate_id not in (omit or set())
    }
    return _production_report(rows)


def test_default_empty_audit_is_open_and_every_physical_promotion_is_false() -> None:
    report = wz.verify_physical_wz_requirements()

    assert len(wz.ClaimLane) == 5
    assert set(value.value for value in wz.ClaimLane) == {
        "OPH_CHART_ONLY",
        "EXTERNAL_SM_EFT_VALIDATION",
        "OPH_NATIVE_DIMENSIONLESS",
        "OPH_NATIVE_PHYSICAL",
        "TARGET_COMPARISON_ONLY",
    }
    assert report["status"] == wz.RequirementStatus.OPEN.value
    assert report["passed"] is False
    assert report["promotion_allowed"] is False
    assert report["exposure_classification"] == "post_exposure_validation"
    assert report["prospective_prediction"] is False
    assert report["SCALE_CAMPAIGN_ALLOWED"] is False
    assert all(report[key] is False for key in wz.PHYSICAL_RECEIPT_KEYS)
    assert report["WZH0_SYNTHETIC_CONTROL_NONPROMOTING"] is True


def test_all_true_caller_mapping_cannot_forge_a_production_bundle() -> None:
    forged = {
        "inventory_replay_passed": True,
        "scientific_replay_passed": True,
        "promotion_allowed": True,
        "status": "PASS",
        **{gate_id: True for gate_id in wz.ALL_GATE_IDS},
    }

    report = wz.verify_physical_wz_requirements(
        forged,  # type: ignore[arg-type]
        lane=wz.ClaimLane.OPH_NATIVE_PHYSICAL,
        scope=wz.ClaimScope.WZ,
    )

    assert report["status"] == wz.RequirementStatus.FAIL.value
    assert report["promotion_allowed"] is False
    assert all(report[key] is False for key in wz.PHYSICAL_RECEIPT_KEYS)
    assert any("on_disk_path" in blocker for blocker in report["blockers"])


@pytest.mark.parametrize(
    "bad_lane",
    [
        ["OPH_CHART_ONLY", "OPH_NATIVE_PHYSICAL"],
        {"OPH_CHART_ONLY": True, "OPH_NATIVE_PHYSICAL": True},
        "UNKNOWN_LANE",
    ],
)
def test_lane_selection_is_exactly_one_allowed_enum(bad_lane: Any) -> None:
    report = wz.verify_physical_wz_requirements(lane=bad_lane)

    assert report["exclusive_lane_count"] == 0
    assert report["status"] == wz.RequirementStatus.FAIL.value
    assert report["promotion_allowed"] is False


def test_h_is_not_a_hidden_wz_prerequisite_but_is_required_for_wzh() -> None:
    wz_report = wz.verify_physical_wz_requirements(
        lane=wz.ClaimLane.EXTERNAL_SM_EFT_VALIDATION,
        scope=wz.ClaimScope.WZ,
    )
    wzh_report = wz.verify_physical_wz_requirements(
        lane=wz.ClaimLane.EXTERNAL_SM_EFT_VALIDATION,
        scope=wz.ClaimScope.WZH,
    )

    assert wz.H_POLE_GATE not in wz_report["required_gate_ids"]
    assert (
        wz_report["gates"][wz.H_POLE_GATE]["status"]
        == wz.RequirementStatus.NOT_APPLICABLE.value
    )
    assert wz.H_POLE_GATE in wzh_report["required_gate_ids"]
    assert (
        wzh_report["gates"][wz.H_POLE_GATE]["status"]
        == wz.RequirementStatus.OPEN.value
    )


def test_w_and_z_pole_gates_are_separate() -> None:
    w_only = wz.verify_physical_wz_requirements(
        lane=wz.ClaimLane.EXTERNAL_SM_EFT_VALIDATION,
        scope=wz.ClaimScope.W_ONLY,
    )
    z_only = wz.verify_physical_wz_requirements(
        lane=wz.ClaimLane.EXTERNAL_SM_EFT_VALIDATION,
        scope=wz.ClaimScope.Z_ONLY,
    )

    assert wz.W_POLE_GATE in w_only["required_gate_ids"]
    assert wz.Z_POLE_GATE not in w_only["required_gate_ids"]
    assert wz.Z_POLE_GATE in z_only["required_gate_ids"]
    assert wz.W_POLE_GATE not in z_only["required_gate_ids"]


def test_mixed_root_family_fails_even_when_each_row_claims_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane = wz.ClaimLane.EXTERNAL_SM_EFT_VALIDATION
    scope = wz.ClaimScope.W_ONLY
    report = _passing_report_for(lane, scope)
    second_id = report["envelope_order"][1]
    family_b = _family("b")
    report["envelopes"][second_id]["source_root_hash"] = family_b[
        "source_root_hash"
    ]
    _install_verified_report(monkeypatch, report)

    audit = wz.verify_physical_wz_requirements(
        "production-manifest.json",
        lane=lane,
        scope=scope,
    )

    assert audit["family_audit"]["passed"] is False
    assert audit["status"] == wz.RequirementStatus.FAIL.value
    assert audit["EXTERNAL_W_POLE_VALIDATION_RECEIPT"] is False
    assert any("mixed_receipt_family" in blocker for blocker in audit["blockers"])


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("claim_lane", "OPH_NATIVE_PHYSICAL", "wrong_lane_envelope"),
        ("claim_scope", "WZH", "wrong_scope_envelope"),
    ],
)
def test_unrequired_extra_envelope_cannot_mix_lane_or_scope(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    needle: str,
) -> None:
    lane = wz.ClaimLane.EXTERNAL_SM_EFT_VALIDATION
    scope = wz.ClaimScope.W_ONLY
    report = _passing_report_for(lane, scope)
    report["envelopes"]["unrequired-extra"] = _envelope_row(
        "UNRELATED_STAGE",
        lane=lane,
        scope=scope,
    )
    report["envelopes"]["unrequired-extra"]["artifact_id"] = "unrequired-extra"
    report["envelopes"]["unrequired-extra"][field] = value
    report["envelope_order"].append("unrequired-extra")
    _install_verified_report(monkeypatch, report)

    audit = wz.verify_physical_wz_requirements(
        "production-manifest.json", lane=lane, scope=scope
    )

    assert audit["status"] == wz.RequirementStatus.FAIL.value
    assert audit["promotion_allowed"] is False
    assert all(audit[key] is False for key in wz.PHYSICAL_RECEIPT_KEYS)
    assert any(needle in blocker for blocker in audit["blockers"])


def test_unknown_stage_cannot_hide_beside_a_passing_native_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane = wz.ClaimLane.OPH_NATIVE_PHYSICAL
    scope = wz.ClaimScope.W_ONLY
    report = _passing_report_for(lane, scope)
    report["envelopes"]["unknown-stage"] = _envelope_row(
        "UNKNOWN_STAGE",
        lane=lane,
        scope=scope,
    )
    report["envelopes"]["unknown-stage"]["artifact_id"] = "unknown-stage"
    report["envelope_order"].append("unknown-stage")
    _install_verified_report(monkeypatch, report)

    audit = wz.verify_physical_wz_requirements(
        "production-manifest.json", lane=lane, scope=scope
    )

    assert audit["status"] == wz.RequirementStatus.FAIL.value
    assert audit["promotion_allowed"] is False
    assert all(audit[key] is False for key in wz.PHYSICAL_RECEIPT_KEYS)
    assert any("unknown_stage_envelope" in blocker for blocker in audit["blockers"])


@pytest.mark.parametrize(
    "receipt_type",
    [
        "WZH0_DIAGNOSTIC",
        "DEMO_ASSUMPTION",
        "FORCED_RECEIPT",
        "FROZEN_TARGET_VALUE",
        "SYNTHETIC_PLACEHOLDER",
    ],
)
def test_diagnostic_envelope_is_unresolved_and_wzh0_cannot_promote(
    monkeypatch: pytest.MonkeyPatch,
    receipt_type: str,
) -> None:
    lane = wz.ClaimLane.EXTERNAL_SM_EFT_VALIDATION
    scope = wz.ClaimScope.W_ONLY
    report = _passing_report_for(lane, scope)
    first_id = report["envelope_order"][0]
    report["envelopes"][first_id]["receipt_type"] = receipt_type
    report["envelopes"][first_id]["scientific_replay_passed"] = False
    report["envelopes"][first_id]["promotion_allowed"] = False
    _install_verified_report(monkeypatch, report)

    audit = wz.verify_physical_wz_requirements(
        "production-manifest.json",
        lane=lane,
        scope=scope,
    )

    assert audit["status"] == wz.RequirementStatus.UNRESOLVED.value
    assert audit["passed"] is False
    assert audit["promotion_allowed"] is False
    assert audit["SCALE_CAMPAIGN_ALLOWED"] is False
    assert first_id in audit["diagnostic_envelope_ids"]
    assert audit["WZH0_SYNTHETIC_CONTROL_NONPROMOTING"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scientific_replay_passed", 1),
        ("promotion_allowed", "true"),
        ("producer_status_trusted", 1),
    ],
)
def test_truthy_non_boolean_scientific_markers_cannot_pass(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    lane = wz.ClaimLane.OPH_NATIVE_PHYSICAL
    scope = wz.ClaimScope.W_ONLY
    report = _passing_report_for(lane, scope)
    first_id = report["envelope_order"][0]
    report["envelopes"][first_id][field] = value
    _install_verified_report(monkeypatch, report)

    audit = wz.verify_physical_wz_requirements(
        "production-manifest.json", lane=lane, scope=scope
    )

    assert audit["status"] == wz.RequirementStatus.OPEN.value
    assert audit["PHYSICAL_WZ_PROMOTION_ALLOWED"] is False
    assert audit["promotion_allowed"] is False


def test_inventory_only_evidence_class_cannot_be_promoted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane = wz.ClaimLane.OPH_NATIVE_PHYSICAL
    scope = wz.ClaimScope.W_ONLY
    report = _passing_report_for(lane, scope)
    first_id = report["envelope_order"][0]
    report["envelopes"][first_id]["evidence_class"] = "IMMUTABLE_INVENTORY_ONLY"
    _install_verified_report(monkeypatch, report)

    audit = wz.verify_physical_wz_requirements(
        "production-manifest.json", lane=lane, scope=scope
    )

    assert audit["status"] == wz.RequirementStatus.OPEN.value
    assert audit["promotion_allowed"] is False


def test_scientific_row_with_blockers_cannot_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane = wz.ClaimLane.OPH_NATIVE_PHYSICAL
    scope = wz.ClaimScope.W_ONLY
    report = _passing_report_for(lane, scope)
    first_id = report["envelope_order"][0]
    report["envelopes"][first_id]["blockers"] = ["checker_disagreement"]
    _install_verified_report(monkeypatch, report)

    audit = wz.verify_physical_wz_requirements(
        "production-manifest.json", lane=lane, scope=scope
    )

    assert audit["status"] == wz.RequirementStatus.FAIL.value
    assert audit["promotion_allowed"] is False
    assert "scientific_envelope_contains_blockers" in audit["gates"][
        report["envelopes"][first_id]["stage_id"]
    ]["blockers"]


def test_malformed_unrequired_row_forces_all_receipts_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane = wz.ClaimLane.OPH_NATIVE_PHYSICAL
    scope = wz.ClaimScope.W_ONLY
    report = _passing_report_for(lane, scope)
    report["envelopes"]["bad-extra"] = {"artifact_id": "bad-extra"}
    report["envelope_order"].append("bad-extra")
    _install_verified_report(monkeypatch, report)

    audit = wz.verify_physical_wz_requirements(
        "production-manifest.json", lane=lane, scope=scope
    )

    assert audit["status"] == wz.RequirementStatus.FAIL.value
    assert audit["promotion_allowed"] is False
    assert all(audit[key] is False for key in wz.PHYSICAL_RECEIPT_KEYS)


def test_native_physical_requires_dimensionless_clock_and_uncertainty_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane = wz.ClaimLane.OPH_NATIVE_PHYSICAL
    scope = wz.ClaimScope.W_ONLY
    report = _passing_report_for(lane, scope, omit={"SOURCE_CLOCK_1"})
    _install_verified_report(monkeypatch, report)

    open_audit = wz.verify_physical_wz_requirements(
        "production-manifest.json",
        lane=lane,
        scope=scope,
    )

    assert open_audit["OPH_NATIVE_DIMENSIONLESS_W_RECEIPT"] is True
    assert open_audit["OPH_NATIVE_DIMENSIONLESS_RECEIPT"] is True
    assert open_audit["OPH_NATIVE_PHYSICAL_W_RECEIPT"] is False
    assert open_audit["OPH_NATIVE_PHYSICAL_RECEIPT"] is False
    assert open_audit["status"] == wz.RequirementStatus.OPEN.value

    _install_verified_report(monkeypatch, _passing_report_for(lane, scope))
    passing_audit = wz.verify_physical_wz_requirements(
        "production-manifest.json",
        lane=lane,
        scope=scope,
    )

    assert passing_audit["OPH_NATIVE_DIMENSIONLESS_W_RECEIPT"] is True
    assert passing_audit["OPH_NATIVE_PHYSICAL_W_RECEIPT"] is True
    assert passing_audit["OPH_NATIVE_PHYSICAL_RECEIPT"] is True
    assert passing_audit["PHYSICAL_WZ_PROMOTION_ALLOWED"] is True
    assert passing_audit["exposure_classification"] == "post_exposure_validation"


def test_external_success_never_flips_native_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane = wz.ClaimLane.EXTERNAL_SM_EFT_VALIDATION
    scope = wz.ClaimScope.WZ
    _install_verified_report(monkeypatch, _passing_report_for(lane, scope))

    audit = wz.verify_physical_wz_requirements(
        "production-manifest.json",
        lane=lane,
        scope=scope,
    )

    assert audit["status"] == wz.RequirementStatus.PASS.value
    assert audit["EXTERNAL_W_POLE_VALIDATION_RECEIPT"] is True
    assert audit["EXTERNAL_Z_POLE_VALIDATION_RECEIPT"] is True
    assert audit["EXTERNAL_WZ_QFT_RECEIPT"] is True
    assert audit["OPH_NATIVE_DIMENSIONLESS_RECEIPT"] is False
    assert audit["OPH_NATIVE_PHYSICAL_RECEIPT"] is False
    assert audit["PHYSICAL_WZ_PROMOTION_ALLOWED"] is False


def test_target_comparison_is_post_exposure_and_cannot_promote_source_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane = wz.ClaimLane.TARGET_COMPARISON_ONLY
    scope = wz.ClaimScope.WZ
    _install_verified_report(monkeypatch, _passing_report_for(lane, scope))

    audit = wz.verify_physical_wz_requirements(
        "production-manifest.json",
        lane=lane,
        scope=scope,
    )

    assert audit["status"] == wz.RequirementStatus.PASS.value
    assert audit["lane_receipts"]["TARGET_COMPARISON_ONLY"] is True
    assert audit["family_audit"]["source_family_immutable"] is True
    assert audit["exposure_classification"] == "post_exposure_validation"
    assert audit["prospective_prediction"] is False
    assert audit["promotion_allowed"] is False
    assert all(audit[key] is False for key in wz.PHYSICAL_RECEIPT_KEYS)


def test_only_literal_pass_satisfies_a_gate_status() -> None:
    assert wz._conjunction_status(["PASS"]) == "PASS"
    assert wz._conjunction_status(["OPEN"]) == "OPEN"
    assert wz._conjunction_status(["UNRESOLVED"]) == "UNRESOLVED"
    assert wz._conjunction_status(["FAIL", "PASS"]) == "FAIL"
    assert wz._conjunction_status(["NOT_APPLICABLE"]) == "NOT_APPLICABLE"
