from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from oph_fpe.viz.screen_a5_ladder import (
    a5_to_standard_model_view_contract,
    build_screen_a5_ladder_payload,
    demo_universe_view_contract,
    screen_geometry_view_contract,
)
from tools.local_visualizer.export import (
    EXPORT_SCHEMA,
    LovableExportError,
    build_lovable_export,
)


def _timeline_payload(*, demo: bool = False) -> dict:
    ladder = build_screen_a5_ladder_payload(
        demo_config={"enabled": True, "forceAllStages": True} if demo else None,
        federation_carrier_count=64,
    )
    views = {
        "screenGeometry": screen_geometry_view_contract(ladder),
        "a5ToStandardModel": a5_to_standard_model_view_contract(ladder),
        "demoUniverse": demo_universe_view_contract(ladder),
        "observerCamera": {
            "viewId": "observerCamera",
            "label": "Virtual observer camera",
            "sectionKind": "observer_camera",
            "description": "Show the finite records available to one virtual observer.",
            "dataSources": ["subjectiveObserverCameras"],
        },
    }
    return {
        "schema": "oph_universe_timeline_visualization_payload_v1",
        "schemaVersion": "oph_universe_timeline_visualization_payload_v1",
        "title": "Test universe",
        "screenA5Ladder": ladder,
        "visualizationViews": views,
        "subjectiveObserverCameras": [],
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _public_anchor() -> dict:
    return {
        "schema": "oph.public_visualization_anchor.v1",
        "public_safe": True,
        "run_id": "frozen-16k-admission-seed-20260751",
        "claim_boundary": "Structural sewing only; physical dynamics refused before RNG.",
        "physical_status": {
            "physical_dynamics_started": False,
            "physical_promotion_allowed": False,
        },
        "structural_exact": {
            "carrier_count": 16_384,
            "seam_count": 16_383,
            "a5_action_count": 60,
        },
        "seeded_disorder_disclosure": {
            "deterministic_visual_seed": 20260751,
            "random_source_state_materialized": False,
            "settling_trajectory_available": False,
            "forbidden_provenance_class": "run_anchored",
        },
    }


def test_directory_export_has_exact_contracts_census_hashes_and_view_summaries(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    safe = source / "carriers.json"
    safe.write_text('{"carrierRows":[0,1,2]}\n', encoding="utf-8")
    large = source / "atoms.bin"
    large.write_bytes(b"a" * 64)
    payload_path = source / "visualization_payload.json"
    _write_json(payload_path, _timeline_payload(demo=True))
    _write_json(
        source / "visualization_export_manifest.json",
        {
            "schema": "oph_universe_visualization_sidecars_v1",
            "files": {
                "carriers": {
                    "path": str(safe),
                    "written": True,
                    "row_count": 3,
                },
                "atoms": {"path": str(large), "written": True},
                "missing": {"path": str(source / "missing.csv"), "written": True},
                "disabled": {
                    "path": None,
                    "written": False,
                    "reason": "producer_disabled",
                },
            },
        },
    )
    out = tmp_path / "lovable-export"

    manifest = build_lovable_export(
        payload_path,
        out,
        public_anchor=_public_anchor(),
        max_sidecar_bytes=safe.stat().st_size,
    )

    assert out.is_dir()
    assert not list(out.rglob("*.zip"))
    assert manifest["schema"] == EXPORT_SCHEMA
    assert manifest["format"] == "directory"
    assert manifest["archivesEmitted"] is False
    census = manifest["sidecarCensus"]
    assert census["full"]["declaredFileCount"] == 4
    assert census["included"] == {
        "fileCount": 1,
        "byteCount": safe.stat().st_size,
    }
    assert census["omitted"]["fileCount"] == 3
    assert census["omitted"]["everyDeclaredFileAccountedFor"] is True
    assert census["withinByteCap"] is True
    assert {row["reason"] for row in manifest["omittedSidecars"]} == {
        "sidecar_byte_cap",
        "source_missing",
        "producer_disabled",
    }

    included = manifest["includedSidecars"][0]
    exported_sidecar = out / included["exportedPath"]
    assert exported_sidecar.read_bytes() == safe.read_bytes()
    assert included["sha256"] == hashlib.sha256(safe.read_bytes()).hexdigest()
    assert included["byteCount"] == safe.stat().st_size
    assert included["mediaType"] == "application/json"
    assert included["metadata"]["rowCount"] == 3

    prototype = json.loads(
        (out / "contracts/local_carrier_prototype.json").read_text(encoding="utf-8")
    )
    assert prototype["counts"] == {
        "a5ActionCount": 60,
        "antipodalPairCount": 6,
        "edgeCount": 30,
        "faceCount": 20,
        "portCount": 12,
    }
    contracts = json.loads(
        (out / "contracts/visualization_views.json").read_text(encoding="utf-8")
    )
    assert contracts["screenGeometry"] == _timeline_payload(demo=True)[
        "visualizationViews"
    ]["screenGeometry"]
    summary = json.loads((out / "payload/summary.json").read_text(encoding="utf-8"))
    assert summary["viewCount"] == 4
    assert all(row["summary"] for row in summary["views"])
    assert summary["census"]["declaredCarrierCount"] == 64
    assert summary["census"]["localCarrierPortCount"] == 12
    instructions = (out / "VISUALIZER_AGENT_INSTRUCTIONS.md").read_text(encoding="utf-8")
    assert "SCREEN_A5_SM_VISUALIZER_AGENT_BRIEF.md" in instructions
    assert "public_cinematic_story.json" in instructions
    assert "illustrative reconstruction" in instructions
    assert "must not contain receipt tables" in instructions
    assert "seeded disordered/random state" in instructions
    assert "includes its optional provenance drawer" in instructions
    assert "It must not expose raw receipts" in instructions
    story = json.loads(
        (out / "contracts/public_cinematic_story.json").read_text(encoding="utf-8")
    )
    assert story["schema"] == "oph.public-cinematic-story/1.0.0"
    assert story["primaryUx"]["gateFree"] is True
    assert story["primaryUx"]["persistentDisclosure"] == "illustrative reconstruction"
    assert story["provenanceContract"]["allowedStatuses"] == [
        "measured",
        "computed",
        "interpolated",
        "synthetic",
        "frozen",
    ]
    assert len(story["sceneData"]["a5"]["actions"]) == 60
    assert story["sceneData"]["eventsAndMatter"]["worldlineSamples"]
    assert story["sceneData"]["eventsAndMatter"]["composites"]
    assert story["sceneData"]["h3"]["eventFrames"]
    assert story["sceneData"]["federation"]["declaredCarrierCount"] == 16_384
    assert story["sceneData"]["federation"]["seededDisorderOpening"]
    assert story["structural16kAnchor"]["physicalDynamicsStarted"] is False
    serialized_story = json.dumps(story)
    assert "physicalStatus" not in serialized_story
    assert '"receipts"' not in serialized_story
    assert "physicalReceipt" not in serialized_story
    assert "physicalPromotion" not in serialized_story
    assert manifest["publicPresentation"]["primaryUxGateFree"] is True
    assert manifest["publicPresentation"]["structural16kAnchorProvided"] is True
    assert (out / "contracts/public_visualization_anchor.json").is_file()

    inventory = {row["path"]: row for row in manifest["files"]}
    for relative, row in inventory.items():
        artifact = out / relative
        assert row["byteCount"] == artifact.stat().st_size
        assert row["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert row["mediaType"]
    assert "export_manifest.json" not in inventory
    assert manifest["manifestIntegrity"][
        "selfFileInventoryExcludedBecauseSelfReferential"
    ] is True
    assert manifest["epistemicBoundary"]["physicalReceiptSnapshotImmutable"] is True
    assert manifest["epistemicBoundary"]["promotion_allowed"] is False
    assert manifest["epistemicBoundary"]["SCALE_CAMPAIGN_ALLOWED"] is False


def test_standalone_screen_a5_payload_derives_exact_primary_view_contracts(
    tmp_path: Path,
) -> None:
    ladder = build_screen_a5_ladder_payload(
        demo_config={"enabled": True, "forceAllStages": True}
    )
    out = tmp_path / "standalone-export"

    manifest = build_lovable_export(ladder, out)

    assert manifest["input"]["kind"] == "standalone_screen_a5"
    views = json.loads(
        (out / "contracts/visualization_views.json").read_text(encoding="utf-8")
    )
    assert views == {
        "a5ToStandardModel": a5_to_standard_model_view_contract(ladder),
        "demoUniverse": demo_universe_view_contract(ladder),
        "screenGeometry": screen_geometry_view_contract(ladder),
    }
    summary = json.loads((out / "payload/summary.json").read_text(encoding="utf-8"))
    assert {row["viewId"] for row in summary["views"]} == {
        "screenGeometry",
        "a5ToStandardModel",
        "demoUniverse",
    }
    assert manifest["sidecarCensus"]["full"]["declaredFileCount"] == 0


def test_payload_is_redacted_and_sensitive_structured_sidecar_is_omitted(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload = _timeline_payload()
    payload["sourcePaths"] = {"observer": "/Users/person/private/run"}
    payload["debug"] = {
        "client_secret": "do-not-export",
        "passwordHash": "also-do-not-export",
        "other": "/private/tmp/also-private",
        "wrapped_error": (
            "FileNotFoundError: '/Users/person/private/run/missing.json'"
        ),
    }
    payload_path = source / "visualization_payload.json"
    _write_json(payload_path, payload)
    sensitive = source / "otherwise-safe.json"
    _write_json(sensitive, {"rows": [], "access_token": "do-not-copy"})
    _write_json(
        source / "visualization_export_manifest.json",
        {
            "schema": "oph_universe_visualization_sidecars_v1",
            "files": {
                "unsafe_content": {"path": str(sensitive), "written": True},
            },
        },
    )
    out = tmp_path / "redacted"

    manifest = build_lovable_export(payload_path, out)

    redacted_text = (out / "payload/visualization_payload.json").read_text(
        encoding="utf-8"
    )
    redacted = json.loads(redacted_text)
    assert redacted["sourcePaths"] == "[REDACTED_BY_LOVABLE_EXPORT]"
    assert redacted["debug"]["client_secret"] == "[REDACTED_BY_LOVABLE_EXPORT]"
    assert redacted["debug"]["passwordHash"] == "[REDACTED_BY_LOVABLE_EXPORT]"
    assert redacted["debug"]["other"] == "[REDACTED_BY_LOVABLE_EXPORT]"
    assert redacted["debug"]["wrapped_error"] == "[REDACTED_BY_LOVABLE_EXPORT]"
    assert "do-not-export" not in redacted_text
    assert "/Users/person" not in redacted_text
    assert not (out / "sidecars").exists()
    assert manifest["redaction"]["applied"] is True
    assert manifest["redaction"]["coverage"] == (
        "recognized credential keys and private filesystem paths"
    )
    assert manifest["redaction"]["generalContentSecrecyGuarantee"] is False
    assert len(manifest["omittedSidecars"]) == 1
    assert manifest["omittedSidecars"][0]["reason"].startswith(
        "sensitive_content_key"
    )


def test_traversal_symlink_and_sensitive_filenames_are_never_copied(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    real = root / "real.bin"
    real.write_bytes(b"inside")
    link = root / "link.bin"
    link.symlink_to(real)
    real_dir = root / "real-dir"
    real_dir.mkdir()
    (real_dir / "nested.bin").write_bytes(b"nested")
    link_dir = root / "link-dir"
    link_dir.symlink_to(real_dir, target_is_directory=True)
    secret = root / "secret-data.json"
    secret.write_text("{}", encoding="utf-8")
    manifest_path = root / "visualization_export_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema": "oph_universe_visualization_sidecars_v1",
            "files": {
                "traversal": {"path": "../outside.bin", "written": True},
                "symlink": {"path": "link.bin", "written": True},
                "symlink_component": {
                    "path": "link-dir/nested.bin",
                    "written": True,
                },
                "secret": {"path": "secret-data.json", "written": True},
                "invalid_flag": {"path": "real.bin", "written": "true"},
            },
        },
    )
    out = tmp_path / "safe-output"

    manifest = build_lovable_export(
        _timeline_payload(),
        out,
        sidecar_manifest=manifest_path,
        sidecar_root=root,
    )

    assert manifest["includedSidecars"] == []
    assert {row["reason"] for row in manifest["omittedSidecars"]} == {
        "path_traversal",
        "symlink_not_allowed",
        "symlink_path_component",
        "sensitive_filename",
        "invalid_written_flag",
    }
    serialized = json.dumps(manifest)
    assert str(outside) not in serialized
    assert "secret-data.json" not in serialized


def test_unrecognized_manifest_and_demo_promotion_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    bad_manifest = root / "visualization_export_manifest.json"
    _write_json(bad_manifest, {"schema": "not-recognized", "files": {}})
    first_out = tmp_path / "first"

    with pytest.raises(LovableExportError, match="unrecognized_sidecar_manifest_schema"):
        build_lovable_export(
            _timeline_payload(),
            first_out,
            sidecar_manifest=bad_manifest,
            sidecar_root=root,
        )
    assert not first_out.exists()

    unsafe = build_screen_a5_ladder_payload(
        demo_config={"enabled": True, "forceAllStages": True}
    )
    unsafe["receipts"]["promotion_allowed"] = True
    second_out = tmp_path / "second"
    with pytest.raises(LovableExportError, match="demo_assumption_cannot_promote"):
        build_lovable_export(unsafe, second_out)
    assert not second_out.exists()


def test_existing_destination_and_payload_symlink_are_rejected(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(LovableExportError, match="output_directory_must_not_exist"):
        build_lovable_export(_timeline_payload(), existing)

    real_payload = tmp_path / "visualization_payload.json"
    _write_json(real_payload, _timeline_payload())
    linked_payload = tmp_path / "linked-payload.json"
    linked_payload.symlink_to(real_payload)
    with pytest.raises(LovableExportError, match="payload_symlink_not_allowed"):
        build_lovable_export(linked_payload, tmp_path / "unused")


@pytest.mark.parametrize("mutation, error", [
    (
        lambda ladder: ladder["localCarrier"]["ports"].pop(),
        "exact_contract_icosahedral_row_counts_invalid",
    ),
    (
        lambda ladder: ladder["localCarrier"]["a5"]["actions"][1].update(
            {"portPermutation": ladder["localCarrier"]["a5"]["actions"][0]["portPermutation"]}
        ),
        "exact_contract_duplicate_a5_action",
    ),
    (
        lambda ladder: ladder["a5ToSm"]["stageEdges"].pop(),
        "exact_contract_stage_edges_invalid",
    ),
])
def test_exact_contract_validation_fails_closed(
    tmp_path: Path,
    mutation,
    error: str,
) -> None:
    ladder = build_screen_a5_ladder_payload()
    mutation(ladder)
    with pytest.raises(LovableExportError, match=error):
        build_lovable_export(ladder, tmp_path / "invalid")


def test_hidden_demo_marker_cannot_promote_and_cap_requires_exact_int(
    tmp_path: Path,
) -> None:
    ladder = build_screen_a5_ladder_payload()
    ladder["a5ToSm"]["stageNodes"][0]["visualizationOnly"] = True
    ladder["a5ToSm"]["stageNodes"][0]["promotion_allowed"] = True
    with pytest.raises(LovableExportError, match="demo_assumption_cannot_promote"):
        build_lovable_export(ladder, tmp_path / "unsafe")
    for value in ("64", 64.0, True):
        with pytest.raises(
            LovableExportError,
            match="sidecar_byte_cap_must_be_nonnegative_integer",
        ):
            build_lovable_export(
                build_screen_a5_ladder_payload(),
                tmp_path / f"bad-cap-{type(value).__name__}",
                max_sidecar_bytes=value,  # type: ignore[arg-type]
            )

    unsafe_anchor = _public_anchor()
    unsafe_anchor["physical_status"]["physical_dynamics_started"] = True
    with pytest.raises(
        LovableExportError,
        match="public_visualization_anchor_boundary_invalid",
    ):
        build_lovable_export(
            build_screen_a5_ladder_payload(),
            tmp_path / "bad-anchor",
            public_anchor=unsafe_anchor,
        )


def test_sidecar_declared_hashes_are_verified_and_names_reencoded(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    first_dir = root / "first"
    second_dir = root / "second"
    first_dir.mkdir(parents=True)
    second_dir.mkdir()
    first = first_dir / "rows.json"
    second = second_dir / "rows.json"
    first.write_text('{"rows":[1]}\n', encoding="utf-8")
    second.write_text('{"rows":[2]}\n', encoding="utf-8")
    first_digest = hashlib.sha256(first.read_bytes()).hexdigest()
    manifest_path = root / "visualization_export_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema": "oph_universe_visualization_sidecars_v1",
            "files": {
                "private-looking/logical-label": {
                    "path": "first/rows.json",
                    "written": True,
                    "byte_count": first.stat().st_size,
                    "sha256": first_digest,
                },
                "same-basename": {
                    "path": "second/rows.json",
                    "written": True,
                    "byte_count": second.stat().st_size,
                    "sha256": "0" * 64,
                },
                "disabled-secret-reason": {
                    "path": None,
                    "written": False,
                    "reason": "arbitrary private explanation",
                },
            },
        },
    )
    out = tmp_path / "verified"
    report = build_lovable_export(
        build_screen_a5_ladder_payload(),
        out,
        sidecar_manifest=manifest_path,
        sidecar_root=root,
    )

    assert len(report["includedSidecars"]) == 1
    assert report["includedSidecars"][0]["logicalName"].startswith("dataset-")
    assert "private-looking" not in json.dumps(report)
    assert {row["reason"] for row in report["omittedSidecars"]} == {
        "declared_sha256_mismatch",
        "producer_reported_not_written",
    }
