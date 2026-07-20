from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from oph_fpe.evidence.production_envelope import (
    CANONICAL_JSON_POLICY,
    FREEZE_ANCHOR_SCHEMA,
    PRODUCTION_BUNDLE_MANIFEST_SCHEMA,
    PRODUCTION_BUNDLE_REPORT_ARTIFACT_TYPE,
    PRODUCTION_BUNDLE_REPORT_SCHEMA,
    PRODUCTION_ENVELOPE_SCHEMA,
    WZ_SHARED_HASH_FIELDS,
    WZ_SOURCE_TO_POLE,
    canonical_json_sha256,
    verify_production_bundle_manifest,
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _media(path: str) -> str:
    if path.endswith("schema.json"):
        return "application/schema+json"
    if path.endswith(".json"):
        return "application/json"
    if path.endswith(".py"):
        return "text/x-python"
    return "application/octet-stream"


def _ref(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative,
        "sha256": _sha(path),
        "byte_count": path.stat().st_size,
        "media_type": _media(relative),
    }


def _envelope(
    root: Path,
    *,
    artifact_id: str,
    stage_id: str,
    subject_path: str,
    output_path: str,
    parents: list[dict[str, object]],
) -> dict[str, object]:
    shared = {
        name: _sha(root / "contracts" / f"{name}.bin")
        for name in WZ_SHARED_HASH_FIELDS
    }
    return {
        "envelope_schema": PRODUCTION_ENVELOPE_SCHEMA,
        "profile": WZ_SOURCE_TO_POLE,
        "schema_id": "oph.fixture.output-schema",
        "schema_version": "1.0.0",
        "schema_sha256": _ref(root, "schema/output.schema.json")["sha256"],
        "schema_ref": _ref(root, "schema/output.schema.json"),
        "artifact_id": artifact_id,
        "stage_id": stage_id,
        "receipt_type": f"{stage_id}.PHYSICAL_RECEIPT",
        "claim_lane": "EXTERNAL_SM_EFT_VALIDATION",
        "claim_scope": "WZ",
        "branch_id": "branch.wz.fixture",
        "freeze_id": "freeze.wz.fixture",
        "source_root_hash": "sha256:" + "1" * 64,
        "subject_type": f"{stage_id}.runtime-subject",
        "subject_canonicalization": CANONICAL_JSON_POLICY,
        "subject_ref": _ref(root, subject_path),
        "subject_digest": canonical_json_sha256(json.loads((root / subject_path).read_text())),
        "output_ref": _ref(root, output_path),
        "output_digest": canonical_json_sha256(json.loads((root / output_path).read_text())),
        "producer": {
            "producer_id": "fixture.producer",
            "source_tree_ref": _ref(root, "code/producer.py"),
            "executable_ref": _ref(root, "bin/producer.bin"),
            "environment_lock_ref": _ref(root, "env/producer.lock"),
        },
        "checker": {
            "checker_id": "fixture.independent-checker",
            "source_tree_ref": _ref(root, "code/checker.py"),
            "executable_ref": _ref(root, "bin/checker.bin"),
            "checker_independence_class": "independent_implementation",
        },
        "freeze_anchor_ref": _ref(root, "freeze/anchor.json"),
        "parent_receipts": parents,
        "shared_contract_hashes": shared,
        "numeric_precision_and_rounding": {
            "arithmetic": "interval",
            "precision_bits": 192,
            "rounding_mode": "outward",
            "numeric_backend": "fixture-ball",
        },
        "target_firewall": {
            "source_ancestry_contains_target": False,
            "target_used_to_select_candidate": False,
            "target_used_to_tune_producer": False,
            "target_used_to_tune_checker": False,
            "target_bytes_available_to_producer": False,
            "target_bytes_available_to_checker": False,
            "comparison_process_can_mutate_bundle": False,
            "comparison_boundary": "read_only_separate_process",
            "exposure_status": "post_exposure_validation",
        },
        "status": "PASS",
        "blockers": [],
        "generated_utc": "2026-07-20T12:00:00Z",
    }


def _build_bundle(root: Path) -> Path:
    _write_json(
        root / "schema/output.schema.json",
        {"$id": "oph.fixture.output-schema", "schema_version": "1.0.0", "type": "object"},
    )
    _write_json(root / "freeze/anchor.json", {
        "schema": FREEZE_ANCHOR_SCHEMA,
        "source_root_hash": "sha256:" + "1" * 64,
        "branch_id": "branch.wz.fixture",
        "freeze_id": "freeze.wz.fixture",
        "commitment_phase": "pre_outcome",
        "frozen_utc": "2026-07-20T11:00:00Z",
    })
    for relative, raw in {
        "code/producer.py": b"PRODUCER_SOURCE\n",
        "bin/producer.bin": b"PRODUCER_EXECUTABLE\n",
        "env/producer.lock": b"fixture-env==1\n",
        "code/checker.py": b"INDEPENDENT_CHECKER_SOURCE\n",
        "bin/checker.bin": b"INDEPENDENT_CHECKER_EXECUTABLE\n",
        **{
            f"contracts/{name}.bin": f"FROZEN {name}\n".encode()
            for name in WZ_SHARED_HASH_FIELDS
        },
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    _write_json(root / "subjects/action.json", {"fields": ["W", "B", "H"], "Q": "T3+Y"})
    _write_json(root / "outputs/action.json", {"passed": True, "verified": True})
    _write_json(root / "subjects/pole.json", {"boson": "W", "sheet": "second"})
    # Deliberately all true: these producer booleans are not scientific evidence.
    _write_json(
        root / "outputs/pole.json",
        {"passed": True, "verified": True, "promotion_allowed": True, "nested": {"all": True}},
    )
    parent_envelope = _envelope(
        root,
        artifact_id="action.packet",
        stage_id="IMPORTED_SM_EFT_ACTION",
        subject_path="subjects/action.json",
        output_path="outputs/action.json",
        parents=[],
    )
    _write_json(root / "envelopes/action.json", parent_envelope)
    parent_ref = {
        "artifact_id": "action.packet",
        "envelope_ref": _ref(root, "envelopes/action.json"),
        "subject_digest": parent_envelope["subject_digest"],
        "output_digest": parent_envelope["output_digest"],
    }
    child_envelope = _envelope(
        root,
        artifact_id="pole.w.packet",
        stage_id="PHYSICAL_CURRENT_POLE_W_1",
        subject_path="subjects/pole.json",
        output_path="outputs/pole.json",
        parents=[parent_ref],
    )
    _write_json(root / "envelopes/pole.json", child_envelope)
    relative_files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    )
    files = [_ref(root, relative) for relative in relative_files]
    manifest: dict[str, object] = {
        "schema": PRODUCTION_BUNDLE_MANIFEST_SCHEMA,
        "bundle_id": "fixture.production-bundle",
        "manifest_payload_sha256": "sha256:" + "0" * 64,
        "files": files,
        "envelope_paths": ["envelopes/action.json", "envelopes/pole.json"],
    }
    _set_manifest_self_hash(manifest)
    manifest_path = root / "production_bundle_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _set_manifest_self_hash(manifest: dict[str, object]) -> None:
    unhashed = dict(manifest)
    unhashed.pop("manifest_payload_sha256", None)
    manifest["manifest_payload_sha256"] = canonical_json_sha256(unhashed)


def _refresh_file_row(manifest_path: Path, relative: str) -> None:
    manifest = json.loads(manifest_path.read_text())
    row = next(item for item in manifest["files"] if item["path"] == relative)
    row.update(_ref(manifest_path.parent, relative))
    _set_manifest_self_hash(manifest)
    _write_json(manifest_path, manifest)


def _mutate_child_envelope(manifest_path: Path, mutator) -> None:
    path = manifest_path.parent / "envelopes/pole.json"
    envelope = json.loads(path.read_text())
    mutator(envelope)
    _write_json(path, envelope)
    _refresh_file_row(manifest_path, "envelopes/pole.json")


def test_valid_inventory_and_all_true_output_remain_nonpromoting(tmp_path: Path):
    manifest = _build_bundle(tmp_path / "bundle")

    report = verify_production_bundle_manifest(manifest)

    assert report["schema"] == PRODUCTION_BUNDLE_REPORT_SCHEMA
    assert report["artifact_type"] == PRODUCTION_BUNDLE_REPORT_ARTIFACT_TYPE
    assert report["inventory_replay_passed"] is True
    assert report["scientific_replay_passed"] is False
    assert report["promotion_allowed"] is False
    assert report["passed"] is False
    assert report["status"] == "OPEN"
    assert report["envelope_order"] == ["action.packet", "pole.w.packet"]
    assert report["ignored_producer_statuses"] == {
        "action.packet": "PASS",
        "pole.w.packet": "PASS",
    }
    row = report["envelopes"]["pole.w.packet"]
    assert row["inventory_replay_passed"] is True
    assert row["producer_status_trusted"] is False
    assert row["scientific_replay_passed"] is False
    assert row["promotion_allowed"] is False


def test_real_wz_aggregator_replays_bundle_but_keeps_physics_open(tmp_path: Path):
    from oph_fpe.bosons.physical_wz_requirements import (
        PHYSICAL_RECEIPT_KEYS,
        verify_physical_wz_requirements,
    )

    manifest = _build_bundle(tmp_path / "bundle")
    report = verify_physical_wz_requirements(
        manifest,
        lane="EXTERNAL_SM_EFT_VALIDATION",
        scope="WZ",
    )

    assert report["production_bundle"]["strict_inventory_replayed"] is True
    assert report["status"] == "OPEN"
    assert report["passed"] is False
    assert report["promotion_allowed"] is False
    assert all(report["receipts"][key] is False for key in PHYSICAL_RECEIPT_KEYS)


def test_manifest_must_be_an_on_disk_file():
    report = verify_production_bundle_manifest({"schema": PRODUCTION_BUNDLE_MANIFEST_SCHEMA})

    assert report["inventory_replay_passed"] is False
    assert any("on_disk_file" in item for item in report["blockers"])


def test_path_escape_is_rejected(tmp_path: Path):
    manifest_path = _build_bundle(tmp_path / "bundle")
    manifest = json.loads(manifest_path.read_text())
    row = next(item for item in manifest["files"] if item["path"] == "subjects/pole.json")
    row["path"] = "../outside.json"
    _set_manifest_self_hash(manifest)
    _write_json(manifest_path, manifest)

    report = verify_production_bundle_manifest(manifest_path)

    assert report["inventory_replay_passed"] is False
    assert any("relative_posix_path" in item for item in report["blockers"])


def test_symlink_is_rejected_even_when_target_bytes_match(tmp_path: Path):
    manifest_path = _build_bundle(tmp_path / "bundle")
    subject = manifest_path.parent / "subjects/pole.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(subject.read_bytes())
    subject.unlink()
    subject.symlink_to(outside)

    report = verify_production_bundle_manifest(manifest_path)

    assert report["receipts"]["P0_FILE_CONTAINMENT_RECEIPT"] is False
    assert report["inventory_replay_passed"] is False


def test_hash_and_byte_count_drift_are_both_rejected(tmp_path: Path):
    manifest_path = _build_bundle(tmp_path / "bundle")
    (manifest_path.parent / "outputs/pole.json").write_text(
        '{"changed":true,"longer":[1,2,3]}\n', encoding="utf-8"
    )

    report = verify_production_bundle_manifest(manifest_path)

    assert report["receipts"]["P0_FILE_HASH_REPLAY_RECEIPT"] is False
    assert report["receipts"]["P0_FILE_BYTE_COUNT_REPLAY_RECEIPT"] is False


def test_unlisted_extra_file_is_rejected(tmp_path: Path):
    manifest_path = _build_bundle(tmp_path / "bundle")
    (manifest_path.parent / "unlisted-target-cache.txt").write_text("forbidden\n")

    report = verify_production_bundle_manifest(manifest_path)

    assert report["receipts"]["P0_EXACT_FILE_CENSUS_RECEIPT"] is False
    assert any("unlisted_extra_file" in item for item in report["inventory_blockers"])


def test_runtime_subject_digest_mismatch_is_rejected(tmp_path: Path):
    manifest_path = _build_bundle(tmp_path / "bundle")
    _mutate_child_envelope(
        manifest_path,
        lambda envelope: envelope.update({"subject_digest": "sha256:" + "f" * 64}),
    )

    report = verify_production_bundle_manifest(manifest_path)

    assert report["receipts"]["P0_RUNTIME_SUBJECT_BINDING_RECEIPT"] is False
    assert any("runtime_subject_digest_mismatch" in item for item in report["blockers"])


def test_mixed_source_root_parent_is_rejected(tmp_path: Path):
    manifest_path = _build_bundle(tmp_path / "bundle")
    _mutate_child_envelope(
        manifest_path,
        lambda envelope: envelope.update({"source_root_hash": "sha256:" + "e" * 64}),
    )

    report = verify_production_bundle_manifest(manifest_path)

    assert report["receipts"]["P0_SINGLE_SOURCE_BRANCH_FREEZE_FAMILY_RECEIPT"] is False
    assert report["receipts"]["P0_PARENT_ENVELOPE_RESOLUTION_RECEIPT"] is False


def test_declared_parent_cycle_is_rejected_even_when_hash_becomes_stale(tmp_path: Path):
    manifest_path = _build_bundle(tmp_path / "bundle")
    root = manifest_path.parent
    action_path = root / "envelopes/action.json"
    action = json.loads(action_path.read_text())
    pole = json.loads((root / "envelopes/pole.json").read_text())
    action["parent_receipts"] = [{
        "artifact_id": "pole.w.packet",
        "envelope_ref": _ref(root, "envelopes/pole.json"),
        "subject_digest": pole["subject_digest"],
        "output_digest": pole["output_digest"],
    }]
    _write_json(action_path, action)
    _refresh_file_row(manifest_path, "envelopes/action.json")

    report = verify_production_bundle_manifest(manifest_path)

    assert report["receipts"]["P0_ANCESTRY_DAG_RECEIPT"] is False
    assert "parent_envelope_ancestry_cycle" in report["blockers"]


def test_missing_parent_hash_is_rejected(tmp_path: Path):
    manifest_path = _build_bundle(tmp_path / "bundle")
    _mutate_child_envelope(
        manifest_path,
        lambda envelope: envelope["parent_receipts"][0].pop("output_digest"),
    )

    report = verify_production_bundle_manifest(manifest_path)

    assert report["receipts"]["P0_PARENT_ENVELOPE_RESOLUTION_RECEIPT"] is False
    assert any("missing_fields" in item for item in report["blockers"])


def test_producer_checker_hash_sharing_and_self_verification_are_rejected(tmp_path: Path):
    manifest_path = _build_bundle(tmp_path / "bundle")

    def mutate(envelope: dict) -> None:
        envelope["checker"]["source_tree_ref"] = envelope["producer"]["source_tree_ref"]

    _mutate_child_envelope(manifest_path, mutate)
    report = verify_production_bundle_manifest(manifest_path)

    assert report["receipts"]["P0_PRODUCER_CHECKER_SEPARATION_RECEIPT"] is False
    assert any("hash_shared" in item for item in report["blockers"])


def test_producer_cannot_name_itself_as_checker(tmp_path: Path):
    manifest_path = _build_bundle(tmp_path / "bundle")

    def mutate(envelope: dict) -> None:
        envelope["checker"]["checker_id"] = envelope["producer"]["producer_id"]

    _mutate_child_envelope(manifest_path, mutate)
    report = verify_production_bundle_manifest(manifest_path)

    assert report["receipts"]["P0_PRODUCER_CHECKER_SEPARATION_RECEIPT"] is False
    assert any("producer_and_checker_id_identical" in item for item in report["blockers"])


def test_wz_shared_hash_must_resolve_to_listed_bundle_bytes(tmp_path: Path):
    manifest_path = _build_bundle(tmp_path / "bundle")

    def mutate(envelope: dict) -> None:
        envelope["shared_contract_hashes"]["action_ast_hash"] = "sha256:" + "9" * 64

    _mutate_child_envelope(manifest_path, mutate)
    report = verify_production_bundle_manifest(manifest_path)

    assert report["receipts"]["P0_WZ_SHARED_HASH_FAMILY_RECEIPT"] is False
    assert any("do_not_resolve_to_bundle_bytes" in item for item in report["blockers"])


@pytest.mark.parametrize(
    "mutator,needle",
    [
        (
            lambda envelope: envelope["target_firewall"].update(
                {"target_used_to_tune_producer": True}
            ),
            "target_firewall_field_not_false",
        ),
        (
            lambda envelope: envelope["target_firewall"].update(
                {"target_used_to_tune_producer": 1}
            ),
            "target_firewall_field_not_false",
        ),
        (
            lambda envelope: envelope["target_firewall"].update(
                {"exposure_status": "prospective_blind"}
            ),
            "wz_requires_post_exposure_validation",
        ),
    ],
)
def test_target_firewall_and_wz_exposure_fail_closed(tmp_path: Path, mutator, needle: str):
    manifest_path = _build_bundle(tmp_path / "bundle")
    _mutate_child_envelope(manifest_path, mutator)

    report = verify_production_bundle_manifest(manifest_path)

    assert report["receipts"]["P0_TARGET_FIREWALL_DECLARATION_RECEIPT"] is False
    assert any(needle in item for item in report["blockers"])


@pytest.mark.parametrize(
    ("generated_utc", "needle"),
    [
        ("2026-07-20T10:59:59Z", "freeze_anchor_not_before_envelope_generation"),
        ("2026-99-99T12:00:00Z", "generated_utc_must_be_valid_utc"),
    ],
)
def test_freeze_anchor_has_real_pre_generation_chronology(
    tmp_path: Path,
    generated_utc: str,
    needle: str,
):
    manifest_path = _build_bundle(tmp_path / "bundle")
    _mutate_child_envelope(
        manifest_path,
        lambda envelope: envelope.update({"generated_utc": generated_utc}),
    )

    report = verify_production_bundle_manifest(manifest_path)

    assert report["receipts"]["P0_FREEZE_PREOUTCOME_ANCHOR_RECEIPT"] is False
    assert report["inventory_replay_passed"] is False
    assert any(needle in item for item in report["blockers"])


def test_same_freeze_id_cannot_hide_different_anchor_bytes(tmp_path: Path):
    manifest_path = _build_bundle(tmp_path / "bundle")
    root = manifest_path.parent
    second_anchor = {
        "schema": FREEZE_ANCHOR_SCHEMA,
        "source_root_hash": "sha256:" + "1" * 64,
        "branch_id": "branch.wz.fixture",
        "freeze_id": "freeze.wz.fixture",
        "commitment_phase": "pre_outcome",
        "frozen_utc": "2026-07-20T10:59:59Z",
    }
    _write_json(root / "freeze/anchor2.json", second_anchor)
    manifest = json.loads(manifest_path.read_text())
    manifest["files"].append(_ref(root, "freeze/anchor2.json"))
    _set_manifest_self_hash(manifest)
    _write_json(manifest_path, manifest)
    _mutate_child_envelope(
        manifest_path,
        lambda envelope: envelope.update(
            {"freeze_anchor_ref": _ref(root, "freeze/anchor2.json")}
        ),
    )

    report = verify_production_bundle_manifest(manifest_path)

    assert report["receipts"]["P0_SINGLE_SOURCE_BRANCH_FREEZE_FAMILY_RECEIPT"] is False
    assert report["inventory_replay_passed"] is False


def test_duplicate_json_key_is_rejected(tmp_path: Path):
    manifest_path = _build_bundle(tmp_path / "bundle")
    raw = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text('{"schema":"duplicate",' + raw[1:], encoding="utf-8")

    report = verify_production_bundle_manifest(manifest_path)

    assert report["inventory_replay_passed"] is False
    assert any("duplicate_json_key:schema" in item for item in report["blockers"])


@pytest.mark.parametrize("raw_value", ["NaN", "1e999"])
def test_nonfinite_json_artifact_is_rejected(tmp_path: Path, raw_value: str):
    manifest_path = _build_bundle(tmp_path / "bundle")
    root = manifest_path.parent
    output_path = root / "outputs/pole.json"
    output_path.write_text(f'{{"value":{raw_value}}}\n', encoding="utf-8")
    envelope_path = root / "envelopes/pole.json"
    envelope = json.loads(envelope_path.read_text())
    envelope["output_ref"] = _ref(root, "outputs/pole.json")
    envelope["output_digest"] = "sha256:" + "a" * 64
    _write_json(envelope_path, envelope)
    _refresh_file_row(manifest_path, "outputs/pole.json")
    _refresh_file_row(manifest_path, "envelopes/pole.json")

    report = verify_production_bundle_manifest(manifest_path)

    assert report["receipts"]["P0_DUPLICATE_KEY_AND_FINITE_JSON_RECEIPT"] is False
    assert report["promotion_allowed"] is False


def test_demo_assumption_profile_is_not_production_evidence(tmp_path: Path):
    manifest_path = _build_bundle(tmp_path / "bundle")
    _mutate_child_envelope(
        manifest_path,
        lambda envelope: envelope.update({"profile": "DEMO_ASSUMPTION"}),
    )

    report = verify_production_bundle_manifest(manifest_path)

    assert report["inventory_replay_passed"] is False
    assert report["scientific_replay_passed"] is False
    assert report["promotion_allowed"] is False
