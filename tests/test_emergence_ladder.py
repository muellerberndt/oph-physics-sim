from __future__ import annotations

import copy
import json
from pathlib import Path

from oph_fpe.emergence_ladder import (
    EMERGENCE_LADDER_SCHEMA_VERSION,
    STAGE_SPECS,
    audit_emergence_ladder,
    canonical_receipt_keys,
    validate_emergence_ladder_report,
    write_emergence_ladder_report,
)


def _write_receipts(run_dir: Path, name: str, receipts: dict[str, object]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / name).write_text(json.dumps({"receipts": receipts}), encoding="utf-8")


def _canonical_keys_for(*stage_ids: str) -> set[str]:
    selected = set(stage_ids)
    return {
        requirement.receipt_keys[0]
        for stage in STAGE_SPECS
        if stage.stage_id in selected
        for requirement in stage.requirements
    }


def test_complete_caller_boolean_fixture_cannot_forge_c0(tmp_path: Path) -> None:
    _write_receipts(
        tmp_path,
        "complete_receipts.json",
        {key: True for key in canonical_receipt_keys()},
    )

    report = audit_emergence_ladder(tmp_path)

    assert report["schema_version"] == EMERGENCE_LADDER_SCHEMA_VERSION
    assert report["dag"]["stages"]["A4"]["passed"] is True
    assert report["dag"]["stages"]["C0"]["passed"] is False
    assert report["overall_receipts"]["OPH_EMERGENCE_LADDER_RECEIPT"] is False
    assert report["overall_claim_status"] == "missing"
    assert report["blockers"]
    assert all(report["policy_checks"].values())
    assert validate_emergence_ladder_report(report) == []
    assert report["dag"]["stages"]["SM7"]["source_report_paths"] == [
        "complete_receipts.json"
    ]


def test_a5_alone_cannot_promote_standard_model(tmp_path: Path) -> None:
    upstream = _canonical_keys_for("A0", "A1", "A2", "A3", "A4", "C0", "A5")
    receipts = {key: True for key in upstream}
    # These broad/legacy labels are deliberately not primitive ladder evidence.
    receipts["FINITE_A5_STRUCTURAL_RECEIPT"] = True
    receipts["SM_ADJOINT_CHARACTER_MATCH_RECEIPT"] = True
    receipts["PHYSICAL_STANDARD_MODEL_PROMOTION_RECEIPT"] = True
    _write_receipts(tmp_path, "a5_only.json", receipts)

    report = audit_emergence_ladder(tmp_path)
    stages = report["dag"]["stages"]

    assert stages["C0"]["passed"] is False
    assert stages["A5"]["passed"] is False
    assert stages["SM0"]["passed"] is False
    assert stages["SM0"]["claim_status"] == "missing"
    assert stages["SM7"]["passed"] is False
    assert (
        report["overall_receipts"]["OPH_STANDARD_MODEL_EMERGENCE_LADDER_RECEIPT"]
        is False
    )


def test_bare_consensus_cannot_bypass_geometry_or_gravity_dependencies(
    tmp_path: Path,
) -> None:
    downstream = _canonical_keys_for(
        "G0",
        "G1",
        "G2",
        "G3",
        "G4",
        "GR0",
        "GR1",
        "GR2",
        "GR3",
        "GR4",
        "GR5",
        "GR6",
        "GR7",
    )
    receipts = {key: True for key in downstream}
    receipts.update(
        {
            "FINITE_CONSENSUS_THEOREM_RECEIPT": True,
            "PRODUCTION_GRAVITY_RECEIPT": True,
            "PHYSICAL_GRAVITY_PREDICTION_RECEIPT": True,
        }
    )
    _write_receipts(tmp_path, "bare_consensus.json", receipts)

    report = audit_emergence_ladder(tmp_path)
    stages = report["dag"]["stages"]

    assert stages["G0"]["passed"] is False
    assert stages["G0"]["claim_status"] == "conditional"
    assert "dependency_not_computed:C0" in stages["G0"]["blockers"]
    assert stages["GR7"]["passed"] is False
    assert report["overall_receipts"]["OPH_GRAVITY_EMERGENCE_LADDER_RECEIPT"] is False


def test_h3_frame_fiber_cannot_substitute_for_semantic_event_manifold(
    tmp_path: Path,
) -> None:
    stage_ids = ("A0", "A1", "A2", "A3", "A4", "C0", "G0", "G1", "G2", "G3")
    receipts = {key: True for key in _canonical_keys_for(*stage_ids)}
    receipts.update(
        {
            "EVENT_MANIFOLD_3P1D_RECEIPT": True,
            "OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT": True,
        }
    )
    _write_receipts(tmp_path, "h3_is_not_event_base.json", receipts)

    report = audit_emergence_ladder(tmp_path)
    stages = report["dag"]["stages"]

    assert stages["C0"]["passed"] is False
    assert stages["G3"]["passed"] is False
    assert stages["G4"]["passed"] is False
    assert stages["G4"]["claim_status"] == "missing"
    assert stages["GR0"]["passed"] is False
    assert report["overall_receipts"]["OPH_GEOMETRY_EMERGENCE_LADDER_RECEIPT"] is False


def test_contradictory_or_truthy_non_boolean_receipts_fail_closed(
    tmp_path: Path,
) -> None:
    complete = {key: True for key in canonical_receipt_keys()}
    _write_receipts(tmp_path, "positive.json", complete)
    _write_receipts(
        tmp_path,
        "contradiction.json",
        {
            "A5_EQUIVARIANT_REFINEMENT_RECEIPT": False,
            "A5_Z6_GLOBAL_FORM_THEOREM_APPLICATION_RECEIPT": 1,
        },
    )

    report = audit_emergence_ladder(tmp_path)
    stages = report["dag"]["stages"]

    assert stages["A5"]["passed"] is False
    assert "contradictory_receipts:a5_equivariance" in stages["A5"]["blockers"]
    assert stages["SM5"]["passed"] is False
    assert any(
        blocker.startswith("non_boolean_receipt:global_theorem_application:")
        for blocker in stages["SM5"]["blockers"]
    )


def test_writer_emits_schema_and_cannot_self_certify_on_reaudit(tmp_path: Path) -> None:
    _write_receipts(
        tmp_path,
        "complete_receipts.json",
        {key: True for key in canonical_receipt_keys()},
    )
    schema_path = tmp_path / "emergence_ladder.schema.json"

    first = write_emergence_ladder_report(tmp_path, schema_path=schema_path)
    second = audit_emergence_ladder(tmp_path)

    assert first["overall_receipts"]["OPH_EMERGENCE_LADDER_RECEIPT"] is False
    assert second["overall_receipts"]["OPH_EMERGENCE_LADDER_RECEIPT"] is False
    assert second["source_inventory"]["ignored_self_artifacts"] == [
        "emergence_ladder.schema.json",
        "emergence_ladder_report.json",
    ]

    tampered = copy.deepcopy(second)
    tampered["dag"]["stages"]["G4"]["dependencies"] = []
    assert "G4:dependency_list_mismatch" in validate_emergence_ladder_report(tampered)
