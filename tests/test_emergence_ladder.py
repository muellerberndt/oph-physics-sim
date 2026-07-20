from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from oph_fpe.common_source_tower import C0_RECEIPT_KEYS
from oph_fpe.core.echosahedral_federation import (
    EchosahedralFederation,
    ExternalBoundaryBundle,
    ObserverSupport,
    interface_algebra_sha256,
    reference_echosahedral_carrier,
    reference_federation_instrument_bundle,
)
from oph_fpe.emergence_ladder import (
    COMMON_SOURCE_REPORT_ARTIFACT_TYPE,
    EMERGENCE_LADDER_SCHEMA_VERSION,
    STAGE_SPECS,
    audit_emergence_ladder,
    canonical_receipt_keys,
    validate_emergence_ladder_report,
    write_emergence_ladder_report,
)
from oph_fpe.repair.transaction import (
    build_repair_replay_envelope,
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


def _valid_repair_artifact() -> dict[str, object]:
    zero = {"terms": [], "constant": 0, "transform": "identity"}
    components = {
        name: dict(zero)
        for name in ("record", "sector", "holonomy", "local_constraint")
    }
    components["overlap"] = {
        "terms": [{"register": "x", "coefficient": 1}],
        "constant": 0,
        "transform": "absolute",
    }
    return build_repair_replay_envelope(
        initial_state={"x": 2},
        initial_versions={"x": 0},
        mismatch_evaluator={
            "kind": "exact_affine_ledger_v1",
            "components": components,
            "physical_auxiliary": {},
        },
        proposals=[
            {
                "proposal_id": "verified",
                "transition_kind": "STRICT_REPAIR",
                "proposal_class": "EXACT_SPLICE",
                "collar": {
                    "collar_id": "c",
                    "visible_read_set": ["x"],
                    "writable_registers": ["x"],
                    "protected_boundary": [],
                    "sector_registers": [],
                    "record_registers": [],
                    "checkpoint_registers": [],
                    "interior_registers": ["x"],
                    "carrier_ids": [],
                    "seam_ids": [],
                    "forbidden_target_fields": [],
                },
                "declared_read_set": ["x"],
                "recovery": {
                    "kind": "literal_updates_v1",
                    "updates": {"x": 1},
                },
                "inverse_updates": {},
                "source_parameters": {},
                "parent_event_ids": [],
            }
        ],
    )


def _canonical_singleton_federation_bundle() -> dict[str, object]:
    carrier = reference_echosahedral_carrier("c0")
    algebra_hash = interface_algebra_sha256({"algebra": "test", "version": 1})
    federation = EchosahedralFederation(
        federation_id="singleton",
        carriers=(carrier,),
        seams=(),
        external_boundaries=(
            ExternalBoundaryBundle(
                boundary_id="external",
                carrier_id="c0",
                ports=tuple(range(12)),
                boundary_condition="open_external",
                boundary_algebra_sha256=algebra_hash,
            ),
        ),
        observer_supports=(
            ObserverSupport(
                observer_token="observer",
                carrier_ids=frozenset({"c0"}),
                visible_seam_ids=frozenset(),
                record_algebra_sha256=algebra_hash,
                checkpoint_cut_sha256=algebra_hash,
            ),
        ),
    )
    return reference_federation_instrument_bundle(federation)


def test_all_name_caller_booleans_cannot_promote_any_stage(tmp_path: Path) -> None:
    _write_receipts(
        tmp_path,
        "complete_receipts.json",
        {key: True for key in canonical_receipt_keys()},
    )

    report = audit_emergence_ladder(tmp_path)

    assert report["schema_version"] == EMERGENCE_LADDER_SCHEMA_VERSION
    assert all(
        stage["passed"] is False
        for stage in report["dag"]["stages"].values()
    )
    assert report["dag"]["stages"]["A0"]["source_report_paths"] == []
    assert report["dag"]["stages"]["C0"]["source_report_paths"] == []
    assert report["overall_receipts"]["OPH_EMERGENCE_LADDER_RECEIPT"] is False
    assert report["overall_claim_status"] == "missing"
    assert report["blockers"]
    assert all(report["policy_checks"].values())
    assert validate_emergence_ladder_report(report) == []
    assert report["source_inventory"]["registered_artifacts"] == []
    unadmitted = report["source_inventory"]["unadmitted_json_reports"]
    assert [row["report_path"] for row in unadmitted] == [
        "complete_receipts.json"
    ]
    assert {
        row["receipt_key"]
        for row in unadmitted[0]["claimed_receipt_booleans"]
    } == {
        key for key in canonical_receipt_keys() if key.endswith("_RECEIPT")
    }


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
    assert stages["G0"]["claim_status"] == "missing"
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


def test_unregistered_false_or_truthy_receipt_names_remain_diagnostic_only(
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
    assert any(
        blocker.startswith("missing_receipt:a5_equivariance:")
        for blocker in stages["A5"]["blockers"]
    )
    assert stages["SM5"]["passed"] is False
    assert stages["SM5"]["claim_status"] == "missing"
    assert report["source_inventory"]["registered_artifacts"] == []
    assert {
        row["report_path"]
        for row in report["source_inventory"]["unadmitted_json_reports"]
    } == {"positive.json", "contradiction.json"}


def test_registered_federation_replay_does_not_admit_extra_named_booleans(
    tmp_path: Path,
) -> None:
    bundle = _canonical_singleton_federation_bundle()
    bundle["PATCH_LOCAL_STATE_RECEIPT"] = True
    bundle["PATCH_READBACK_RECEIPT"] = True
    (tmp_path / "federation.json").write_text(json.dumps(bundle), encoding="utf-8")

    report = audit_emergence_ladder(tmp_path)
    a0 = report["dag"]["stages"]["A0"]

    assert a0["evidence"]["canonical_federation_bundle"]["passed"] is True
    assert a0["evidence"]["carrier_conformance"]["passed"] is True
    assert a0["evidence"]["federation_sewing"]["passed"] is True
    assert a0["evidence"]["local_state"]["passed"] is False
    assert a0["evidence"]["readback"]["passed"] is False
    assert a0["passed"] is False
    assert report["source_inventory"]["registered_artifacts"][0][
        "registry_id"
    ] == "canonical_echosahedral_federation_bundle_v1"


def test_valid_c0_verifier_output_plus_all_other_booleans_cannot_promote(
    tmp_path: Path, monkeypatch
) -> None:
    source_path = tmp_path / "source_report.json"
    source_path.write_text(
        json.dumps({"artifact_type": COMMON_SOURCE_REPORT_ARTIFACT_TYPE}),
        encoding="utf-8",
    )
    _write_receipts(
        tmp_path,
        "caller_booleans.json",
        {key: True for key in canonical_receipt_keys()},
    )
    commitment = "a" * 64

    def replay(_path: Path) -> dict[str, object]:
        return {
            "passed": True,
            "blockers": [],
            "recomputed_report": {
                **{key: True for key in C0_RECEIPT_KEYS},
                "computed_bundle_commitment": commitment,
                "artifact_verification": {"rows": []},
            },
        }

    monkeypatch.setattr(
        "oph_fpe.emergence_ladder.verify_common_source_tower_report_file",
        replay,
    )
    report = audit_emergence_ladder(tmp_path)

    c0 = report["dag"]["stages"]["C0"]
    assert all(item["passed"] is True for item in c0["evidence"].values())
    assert c0["passed"] is False
    assert "dependency_not_computed:A4" in c0["blockers"]
    assert all(
        stage["passed"] is False
        for stage in report["dag"]["stages"].values()
    )
    assert report["source_inventory"]["unadmitted_json_reports"][0][
        "report_path"
    ] == "caller_booleans.json"


def test_repair_artifact_is_replayed_and_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    artifact = _valid_repair_artifact()
    (valid_dir / "repair.json").write_text(json.dumps(artifact), encoding="utf-8")

    valid = audit_emergence_ladder(valid_dir)
    a2 = valid["dag"]["stages"]["A2"]
    assert all(item["passed"] is True for item in a2["evidence"].values())
    assert a2["passed"] is False
    assert "dependency_not_computed:A1" in a2["blockers"]
    assert valid["source_inventory"]["input_errors"] == []

    tampered_dir = tmp_path / "tampered"
    tampered_dir.mkdir()
    tampered = copy.deepcopy(artifact)
    tampered["initial_state"]["x"] = 99
    (tampered_dir / "repair.json").write_text(
        json.dumps(tampered), encoding="utf-8"
    )

    rejected = audit_emergence_ladder(tampered_dir)
    rejected_a2 = rejected["dag"]["stages"]["A2"]
    assert rejected_a2["evidence"]["repair_artifact_integrity"]["passed"] is False
    assert rejected_a2["evidence"]["transactional_repair"]["passed"] is False
    assert rejected["source_inventory"]["input_errors"]


def test_cross_artifact_stage_closes_without_a_common_source_commitment(
    tmp_path: Path,
) -> None:
    (tmp_path / "federation.json").write_text(
        json.dumps(_canonical_singleton_federation_bundle()), encoding="utf-8"
    )
    (tmp_path / "repair.json").write_text(
        json.dumps(_valid_repair_artifact()), encoding="utf-8"
    )

    report = audit_emergence_ladder(tmp_path)
    a2 = report["dag"]["stages"]["A2"]

    assert a2["common_source_binding_required"] is True
    assert a2["common_source_binding_verified"] is False
    assert set(a2["closure_source_report_paths"]) == {
        "federation.json",
        "repair.json",
    }
    assert any(
        blocker.startswith("common_source_commitment_unbound:")
        for blocker in a2["blockers"]
    )
    assert a2["passed"] is False


def test_verified_common_source_hash_and_role_bind_cross_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    payloads = {
        "federation.json": _canonical_singleton_federation_bundle(),
        "repair.json": _valid_repair_artifact(),
    }
    artifact_rows = []
    roles = {
        "federation.json": "authoritative_presentation",
        "repair.json": "repair_log",
    }
    for name, payload in payloads.items():
        raw = json.dumps(payload).encode("utf-8")
        (tmp_path / name).write_bytes(raw)
        artifact_rows.append(
            {
                "artifact_id": name.removesuffix(".json"),
                "semantic_role": roles[name],
                "actual_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
                "passed": True,
            }
        )
    (tmp_path / "source.json").write_text(
        json.dumps({"artifact_type": COMMON_SOURCE_REPORT_ARTIFACT_TYPE}),
        encoding="utf-8",
    )
    commitment = "source-bundle-commitment"

    def replay(_path: Path) -> dict[str, object]:
        return {
            "passed": True,
            "blockers": [],
            "recomputed_report": {
                **{key: True for key in C0_RECEIPT_KEYS},
                "computed_bundle_commitment": commitment,
                "artifact_verification": {"rows": artifact_rows},
            },
        }

    monkeypatch.setattr(
        "oph_fpe.emergence_ladder.verify_common_source_tower_report_file",
        replay,
    )
    report = audit_emergence_ladder(tmp_path)
    a2 = report["dag"]["stages"]["A2"]

    assert a2["common_source_binding_required"] is True
    assert a2["common_source_binding_verified"] is True
    assert set(a2["closure_source_bindings"].values()) == {commitment}
    assert not any(
        blocker.startswith("common_source_commitment_")
        for blocker in a2["blockers"]
    )
    registry_rows = {
        row["report_path"]: row
        for row in report["source_inventory"]["registered_artifacts"]
    }
    assert registry_rows["federation.json"]["source_binding_status"] == (
        "bound_by_verified_common_source_manifest"
    )
    assert registry_rows["repair.json"]["source_binding_status"] == (
        "bound_by_verified_common_source_manifest"
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
