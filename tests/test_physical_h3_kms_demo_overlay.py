from __future__ import annotations

from copy import deepcopy

from oph_fpe.viz.physical_h3_kms_demo_overlay import (
    DEMO_STATUS,
    PHYSICAL_H3_KMS_STAGE_IDS,
    build_physical_h3_kms_demo_overlay,
)
from oph_fpe.viz.screen_a5_ladder import build_screen_a5_ladder_payload


def _physical_snapshot() -> dict:
    stages = {}
    for index, stage_id in enumerate(PHYSICAL_H3_KMS_STAGE_IDS):
        if index < 4:
            gate_status = "PASS"
            scientific_status = "NOT_EVALUATED"
            passed = True
        elif index == 4:
            gate_status = "PASS"
            scientific_status = "VALID_FAIL"
            passed = True
        else:
            gate_status = "NOT_EVALUATED"
            scientific_status = "NOT_EVALUATED"
            passed = False
        stages[stage_id] = {
            "gate_status": gate_status,
            "scientific_status": scientific_status,
            "passed": passed,
            "blockers": [f"raw-blocker-{index}"],
        }
    return {
        "schema": "oph_physical_h3_kms_preflight_v3",
        "mode": "physical_h3_kms_campaign_preflight",
        "campaign_status": "INCOMPLETE",
        "physical_promotion_allowed": False,
        "retirement_counting_allowed": False,
        "stages": stages,
    }


def test_force_all_nudges_only_the_display_overlay_and_preserves_raw_snapshot() -> None:
    snapshot = _physical_snapshot()
    before = deepcopy(snapshot)

    overlay = build_physical_h3_kms_demo_overlay(
        snapshot,
        demo_enabled=True,
        force_all_missing=True,
    )

    assert snapshot == before
    assert overlay["physicalSnapshot"] == before
    assert overlay["physicalSnapshotDigestBefore"] == overlay[
        "physicalSnapshotDigestAfter"
    ]
    assert overlay["physicalSnapshotDigestPreserved"] is True
    assert overlay["physicalSnapshotTrusted"] is True
    assert overlay["stageOrder"] == list(PHYSICAL_H3_KMS_STAGE_IDS)
    assert overlay["forcedStageIds"] == list(PHYSICAL_H3_KMS_STAGE_IDS)
    assert overlay["displayComplete"] is True

    for stage_id, raw_stage, row in zip(
        PHYSICAL_H3_KMS_STAGE_IDS,
        before["stages"].values(),
        overlay["stageNodes"],
        strict=True,
    ):
        assert row["stageId"] == stage_id
        assert row["physicalGateStatus"] == raw_stage["gate_status"]
        assert row["physicalScientificStatus"] == raw_stage["scientific_status"]
        assert row["physicalPassed"] is False
        assert row["demoNudgeApplied"] is True
        assert row["displayStatus"] == DEMO_STATUS
        assert row["displayComplete"] is True
        assert row["displayData"]
        assert len(row["fieldProvenance"]) == len(row["displayData"])
        assert all(
            provenance["epistemicStatus"] == DEMO_STATUS
            and provenance["physicalReceiptEligible"] is False
            and provenance["targetAncestryEligible"] is False
            for provenance in row["fieldProvenance"]
        )

    assert overlay["displayGuards"] == {
        "scientific_receipts_unchanged": True,
        "physical_stage_statuses_unchanged": True,
        "physical_snapshot_mutated": False,
        "promotion_allowed": False,
        "retirement_counting_allowed": False,
        "SCALE_CAMPAIGN_ALLOWED": False,
        "target_ancestry_eligible": False,
        "may_write_campaign_artifacts": False,
    }
    assert overlay["receipts"]["PHYSICAL_PROMOTION_RECEIPT"] is False
    assert overlay["receipts"]["BRANCH_RETIREMENT_AUTHORIZED"] is False
    assert overlay["receipts"]["SCALE_CAMPAIGN_ALLOWED"] is False


def test_real_valid_pass_is_computed_while_only_missing_stages_are_nudged() -> None:
    snapshot = _physical_snapshot()
    passed_id = PHYSICAL_H3_KMS_STAGE_IDS[6]
    snapshot["stages"][passed_id] = {
        "gate_status": "PASS",
        "scientific_status": "VALID_PASS",
        "passed": True,
        "blockers": [],
    }

    overlay = build_physical_h3_kms_demo_overlay(
        snapshot,
        demo_enabled=True,
        force_all_missing=True,
    )
    by_id = {row["stageId"]: row for row in overlay["stageNodes"]}

    assert passed_id not in overlay["forcedStageIds"]
    assert by_id[passed_id]["physicalPassed"] is True
    assert by_id[passed_id]["demoNudgeApplied"] is False
    assert by_id[passed_id]["displayStatus"] == "COMPUTED"
    assert by_id[passed_id]["displayData"] == {}
    assert by_id[passed_id]["fieldProvenance"] == []
    assert overlay["displayComplete"] is True


def test_disabled_or_untrusted_overlay_never_self_promotes() -> None:
    untrusted = _physical_snapshot()
    untrusted["schema"] = "passed-looking-untyped-object"
    for row in untrusted["stages"].values():
        row.update(
            {"gate_status": "PASS", "scientific_status": "VALID_PASS", "passed": True}
        )

    disabled = build_physical_h3_kms_demo_overlay(
        untrusted,
        demo_enabled=False,
        force_all_missing=True,
    )
    forced = build_physical_h3_kms_demo_overlay(
        untrusted,
        demo_enabled=True,
        force_all_missing=True,
    )

    assert disabled["enabled"] is False
    assert disabled["forcedStageIds"] == []
    assert disabled["displayComplete"] is False
    assert forced["physicalSnapshotTrusted"] is False
    assert all(row["physicalPassed"] is False for row in forced["stageNodes"])
    assert forced["displayComplete"] is True
    assert forced["receipts"]["PHYSICAL_PROMOTION_RECEIPT"] is False
    assert forced["receipts"]["BRANCH_RETIREMENT_AUTHORIZED"] is False


def test_screen_ladder_embeds_overlay_without_changing_a5_or_h3_physical_data() -> None:
    snapshot = _physical_snapshot()
    before = deepcopy(snapshot)

    ladder = build_screen_a5_ladder_payload(
        physical_h3_kms_snapshot=snapshot,
        demo_config={"enabled": True, "forceAllStages": True},
    )
    overlay = ladder["physicalH3KmsDemoOverlay"]

    assert snapshot == before
    assert overlay["physicalSnapshot"] == before
    assert overlay["displayComplete"] is True
    assert ladder["a5ToSm"]["displayComplete"] is True
    assert ladder["receipts"]["promotion_allowed"] is False
    assert ladder["receipts"]["SCALE_CAMPAIGN_ALLOWED"] is False
    assert overlay["receipts"]["PHYSICAL_PROMOTION_RECEIPT"] is False
    assert overlay["receipts"]["BRANCH_RETIREMENT_AUTHORIZED"] is False

