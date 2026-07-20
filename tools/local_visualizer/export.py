"""Build a bounded, auditable Lovable visualizer handoff directory.

The local visualizer can page the complete source data directly, while this
module creates a portable renderer handoff.  It deliberately emits a directory
rather than an archive so a visualizer agent can inspect individual contracts
without unpacking a monolithic bundle.

Only files declared by the recognized visualization-sidecar manifest can be
copied.  The payload is canonicalized and redacted, sidecars are confined to a
declared root, and every declared sidecar is accounted for as included or
omitted.  Demo-assumption state is retained for rendering but can never promote
physical receipts or authorize a scale campaign.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import mimetypes
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence


EXPORT_SCHEMA = "oph.lovable-visualizer-export/1.0.0"
SUMMARY_SCHEMA = "oph.lovable-visualizer-summary/1.0.0"
PUBLIC_STORY_SCHEMA = "oph.public-cinematic-story/1.0.0"
PROVENANCE_SCHEMA = "oph.visualization-provenance/1.0.0"
PROVENANCE_STATUSES = ("measured", "computed", "interpolated", "synthetic", "frozen")
PUBLIC_DISCLOSURE = "illustrative reconstruction"
RECOGNIZED_SIDECAR_SCHEMA = "oph_universe_visualization_sidecars_v1"
DEFAULT_SIDECAR_BYTE_CAP = 64_000_000
ALLOWED_SIDECAR_SUFFIXES = frozenset(
    {".json", ".jsonl", ".csv", ".bin", ".npy", ".npz"}
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DETAILED_BRIEF = PROJECT_ROOT / "docs" / "SCREEN_A5_SM_VISUALIZER_AGENT_BRIEF.md"

REDACTION_MARKER = "[REDACTED_BY_LOVABLE_EXPORT]"
UNSAFE_PATH_MARKER = "[REDACTED_UNSAFE_PATH]"
SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "auth_token",
        "authorization",
        "password",
        "passphrase",
        "private_key",
        "secret_key",
        "client_secret",
        "session_token",
        "cookie",
        "set_cookie",
    }
)
SENSITIVE_FILENAMES = frozenset(
    {
        ".env",
        ".netrc",
        "credentials",
        "credentials.json",
        "secrets.json",
        "id_rsa",
        "id_ed25519",
    }
)
SENSITIVE_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx", ".kdbx"})
PRIVATE_PATH_KEYS = frozenset(
    {
        "source_paths",
        "source_path",
        "payload_path",
        "manifest_path",
        "bundle_dir",
        "viewer_path",
        "instructions_path",
        "data_root",
        "sidecar_root",
    }
)
PUBLIC_METADATA_KEYS = {
    "row_count": "rowCount",
    "dtype": "dtype",
    "layout": "layout",
    "schema": "dataSchema",
    "frame_count": "frameCount",
    "field_name": "fieldName",
}
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_EMBEDDED_POSIX_PRIVATE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9])/(?:Users|home|private|tmp|var|etc|root|opt|mnt|workspace|srv)/"
)
_EMBEDDED_WINDOWS_PRIVATE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"
)
_SAFE_COMPONENT_RE = re.compile(r"[^a-zA-Z0-9._-]+")


class LovableExportError(ValueError):
    """Raised when an export cannot satisfy its safety contract."""


@dataclass(frozen=True)
class _SidecarDeclaration:
    logical_name: str
    raw_path: str | None
    written: bool | None
    reason: str | None
    metadata: Mapping[str, Any]


def build_lovable_export(
    payload: Mapping[str, Any] | Path | str,
    out_dir: Path | str,
    *,
    sidecar_manifest: Path | str | None = None,
    sidecar_root: Path | str | None = None,
    public_anchor: Mapping[str, Any] | Path | str | None = None,
    max_sidecar_bytes: int = DEFAULT_SIDECAR_BYTE_CAP,
    discover_adjacent_manifest: bool = True,
) -> dict[str, Any]:
    """Write one deterministic, non-archive visualizer handoff.

    ``payload`` may be a complete universe-timeline payload or a standalone
    ``screenA5Ladder`` payload.  The destination must not already exist.  If a
    sidecar manifest is supplied (or discovered next to a payload file), only
    its declared files are considered and the aggregate copied sidecar bytes
    never exceed ``max_sidecar_bytes``.
    """

    cap = _validated_cap(max_sidecar_bytes)
    destination = Path(out_dir).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise LovableExportError("output_directory_must_not_exist")
    if not destination.name or destination.name in {".", ".."}:
        raise LovableExportError("invalid_output_directory")

    loaded, source_path, input_digest = _load_payload(payload)
    input_kind, ladder, source_views = _payload_contracts(loaded)
    _assert_demo_assumption_boundary(ladder)

    redactions: list[dict[str, str]] = []
    redacted_payload = _redact_json(loaded, redactions=redactions)
    if input_kind == "standalone_screen_a5":
        redacted_ladder = _json_roundtrip(redacted_payload)
    else:
        embedded_ladder = redacted_payload.get("screenA5Ladder")
        if not isinstance(embedded_ladder, Mapping):
            raise LovableExportError("screen_a5_ladder_lost_during_redaction")
        redacted_ladder = _json_roundtrip(embedded_ladder)
    redacted_views = _redact_json(
        source_views,
        redactions=redactions,
        path="$.visualizationViews",
    )
    loaded_anchor, anchor_digest = _load_public_visualization_anchor(public_anchor)
    redacted_anchor = (
        _redact_json(
            loaded_anchor,
            redactions=redactions,
            path="$.publicVisualizationAnchor",
        )
        if loaded_anchor is not None
        else None
    )
    redactions = _deduplicated_redactions(redactions)
    redacted_payload = _annotate_visualization_records(
        redacted_payload,
        source_digest=input_digest,
    )
    redacted_ladder = _annotate_visualization_records(
        redacted_ladder,
        source_digest=input_digest,
        path="$.screenA5Ladder",
    )
    local_carrier = redacted_ladder.get("localCarrier")
    if not isinstance(local_carrier, Mapping):
        raise LovableExportError("screen_a5_local_carrier_missing")

    manifest_path = _resolve_optional_manifest(
        sidecar_manifest=sidecar_manifest,
        source_payload_path=source_path,
        discover_adjacent=discover_adjacent_manifest,
    )
    root_path = _resolve_sidecar_root(manifest_path, sidecar_root)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    staging = temp_parent / "export"
    staging.mkdir()
    try:
        _write_json(staging / "payload" / "visualization_payload.json", redacted_payload)
        _write_json(staging / "contracts" / "screen_a5_ladder.json", redacted_ladder)
        _write_json(
            staging / "contracts" / "local_carrier_prototype.json",
            local_carrier,
        )
        _write_json(
            staging / "contracts" / "visualization_views.json",
            redacted_views,
        )
        public_story = _public_cinematic_story(
            payload=redacted_payload,
            ladder=redacted_ladder,
            source_digest=input_digest,
            public_anchor=redacted_anchor,
            anchor_digest=anchor_digest,
        )
        _write_json(
            staging / "contracts" / "public_cinematic_story.json",
            public_story,
        )
        if redacted_anchor is not None:
            _write_json(
                staging / "contracts" / "public_visualization_anchor.json",
                redacted_anchor,
            )

        included, omitted, sidecar_full = _copy_manifested_sidecars(
            manifest_path=manifest_path,
            sidecar_root=root_path,
            staging=staging,
            byte_cap=cap,
        )
        sidecar_census = _sidecar_census(sidecar_full, included, omitted, cap)
        view_summaries = _view_summaries(redacted_views)
        summary = _build_export_summary(
            payload=redacted_payload,
            ladder=redacted_ladder,
            input_kind=input_kind,
            view_summaries=view_summaries,
            sidecar_census=sidecar_census,
            redactions=redactions,
        )
        _write_json(staging / "payload" / "summary.json", summary)

        documentation = staging / "documentation"
        documentation.mkdir(parents=True, exist_ok=True)
        if not DETAILED_BRIEF.is_file() or DETAILED_BRIEF.is_symlink():
            raise LovableExportError("detailed_visualizer_brief_missing_or_unsafe")
        shutil.copyfile(
            DETAILED_BRIEF,
            documentation / "SCREEN_A5_SM_VISUALIZER_AGENT_BRIEF.md",
        )
        (staging / "VISUALIZER_AGENT_INSTRUCTIONS.md").write_text(
            _agent_instructions(view_summaries),
            encoding="utf-8",
        )

        output_files = _inventory_files(staging)
        boundary = _epistemic_boundary()
        export_manifest: dict[str, Any] = {
            "schema": EXPORT_SCHEMA,
            "format": "directory",
            "archivesEmitted": False,
            "input": {
                "kind": input_kind,
                "payloadSchema": redacted_payload.get("schemaVersion")
                or redacted_payload.get("schema"),
                "canonicalInputSha256": input_digest,
                "sourcePathExported": False,
            },
            "canonicalPayload": "payload/visualization_payload.json",
            "canonicalSummary": "payload/summary.json",
            "prototypeContracts": {
                "screenA5Ladder": "contracts/screen_a5_ladder.json",
                "localIcosahedralCarrier": "contracts/local_carrier_prototype.json",
                "visualizationViews": "contracts/visualization_views.json",
                "publicCinematicStory": "contracts/public_cinematic_story.json",
                **(
                    {
                        "publicVisualizationAnchor": (
                            "contracts/public_visualization_anchor.json"
                        )
                    }
                    if redacted_anchor is not None
                    else {}
                ),
            },
            "publicPresentation": {
                "profile": "PUBLIC_CINEMATIC_ILLUSTRATIVE_RECONSTRUCTION",
                "primaryUxGateFree": True,
                "storyContract": "contracts/public_cinematic_story.json",
                "persistentDisclosure": PUBLIC_DISCLOSURE,
                "optionalProvenanceDrawer": True,
                "allowedProvenanceStatuses": list(PROVENANCE_STATUSES),
                "structural16kAnchorProvided": redacted_anchor is not None,
                "structural16kAnchorSha256": anchor_digest,
            },
            "visualizerAgentInstructions": "VISUALIZER_AGENT_INSTRUCTIONS.md",
            "detailedBrief": (
                "documentation/SCREEN_A5_SM_VISUALIZER_AGENT_BRIEF.md"
            ),
            "redaction": {
                "applied": bool(redactions),
                "count": len(redactions),
                "locations": redactions,
                "sourceFilesystemPathsExported": False,
                "coverage": "recognized credential keys and private filesystem paths",
                "generalContentSecrecyGuarantee": False,
            },
            "sidecarManifest": {
                "recognizedSchema": RECOGNIZED_SIDECAR_SCHEMA,
                "present": manifest_path is not None,
                "sourcePathExported": False,
            },
            "sidecarCensus": sidecar_census,
            "includedSidecars": included,
            "omittedSidecars": omitted,
            "files": output_files,
            "epistemicBoundary": boundary,
            "manifestIntegrity": {
                "algorithm": "sha256",
                "scope": "canonical JSON manifest with manifestBodySha256 absent",
                "selfFileInventoryExcludedBecauseSelfReferential": True,
            },
        }
        body_bytes = _canonical_json_bytes(export_manifest)
        export_manifest["manifestIntegrity"]["manifestBodySha256"] = hashlib.sha256(
            body_bytes
        ).hexdigest()
        _write_json(staging / "export_manifest.json", export_manifest)

        # Defensive assertion: the exporter never changes a caller-owned mapping.
        after_loaded, _, after_digest = _load_payload(payload)
        if input_digest != after_digest or loaded != after_loaded:
            raise LovableExportError("input_payload_changed_during_export")

        staging.rename(destination)
        temp_parent.rmdir()
        return json.loads((destination / "export_manifest.json").read_text(encoding="utf-8"))
    except Exception:
        if temp_parent.exists():
            shutil.rmtree(temp_parent)
        raise


def _load_payload(
    payload: Mapping[str, Any] | Path | str,
) -> tuple[dict[str, Any], Path | None, str]:
    source_path: Path | None = None
    if isinstance(payload, Mapping):
        raw = _canonical_json_bytes(payload)
        parsed = _load_json_bytes(raw)
    else:
        source_path = Path(payload).expanduser().absolute()
        if source_path.is_symlink():
            raise LovableExportError("payload_symlink_not_allowed")
        if _looks_sensitive_filename(source_path.name):
            raise LovableExportError("sensitive_payload_filename")
        if source_path.suffix.lower() != ".json":
            raise LovableExportError("payload_must_be_json")
        if not source_path.is_file():
            raise LovableExportError("payload_file_missing")
        try:
            parsed = _load_json_bytes(source_path.read_bytes())
        except OSError as exc:
            raise LovableExportError("payload_unreadable") from exc
        raw = _canonical_json_bytes(parsed)
    if not isinstance(parsed, dict):
        raise LovableExportError("payload_root_must_be_object")
    return parsed, source_path, hashlib.sha256(raw).hexdigest()


def _load_public_visualization_anchor(
    anchor: Mapping[str, Any] | Path | str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if anchor is None:
        return None, None
    if isinstance(anchor, Mapping):
        parsed = _json_roundtrip(anchor)
    else:
        path = Path(anchor).expanduser().absolute()
        if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".json":
            raise LovableExportError("public_visualization_anchor_missing_or_unsafe")
        parsed = _load_json_bytes(path.read_bytes())
    if not isinstance(parsed, dict):
        raise LovableExportError("public_visualization_anchor_must_be_object")
    if parsed.get("schema") != "oph.public_visualization_anchor.v1":
        raise LovableExportError("public_visualization_anchor_schema_invalid")
    if parsed.get("public_safe") is not True:
        raise LovableExportError("public_visualization_anchor_not_public_safe")
    physical = parsed.get("physical_status")
    structural = parsed.get("structural_exact")
    disorder = parsed.get("seeded_disorder_disclosure")
    if (
        not isinstance(physical, Mapping)
        or physical.get("physical_dynamics_started") is not False
        or physical.get("physical_promotion_allowed") is not False
        or not isinstance(structural, Mapping)
        or structural.get("carrier_count") != 16_384
        or structural.get("a5_action_count") != 60
        or not isinstance(disorder, Mapping)
        or disorder.get("random_source_state_materialized") is not False
        or disorder.get("settling_trajectory_available") is not False
        or disorder.get("forbidden_provenance_class") != "run_anchored"
    ):
        raise LovableExportError("public_visualization_anchor_boundary_invalid")
    raw = _canonical_json_bytes(parsed)
    return parsed, hashlib.sha256(raw).hexdigest()


def _load_json_bytes(raw: bytes) -> Any:
    def reject_constant(value: str) -> None:
        raise LovableExportError(f"nonfinite_json_value:{value}")

    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise LovableExportError(f"duplicate_json_key:{key}")
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LovableExportError("invalid_utf8_json") from exc


def _payload_contracts(
    payload: Mapping[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    embedded = payload.get("screenA5Ladder")
    if isinstance(embedded, Mapping):
        ladder = _json_roundtrip(embedded)
        views_value = payload.get("visualizationViews")
        views = _json_roundtrip(views_value) if isinstance(views_value, Mapping) else {}
        missing = {
            "screenGeometry",
            "a5ToStandardModel",
            "demoUniverse",
        } - set(views)
        if missing:
            views.update(_derived_screen_views(ladder, only=missing))
        kind = "universe_timeline"
    elif all(
        isinstance(payload.get(key), Mapping)
        for key in ("localCarrier", "a5ToSm")
    ):
        ladder = _json_roundtrip(payload)
        views = _derived_screen_views(ladder)
        kind = "standalone_screen_a5"
    else:
        raise LovableExportError("payload_missing_screen_a5_contract")
    _validate_exact_screen_a5_contract(ladder, views)
    return kind, ladder, views


def _validate_exact_screen_a5_contract(
    ladder: Mapping[str, Any],
    views: Mapping[str, Any],
) -> None:
    """Fail closed before the export calls geometry or the stage DAG exact."""

    from oph_fpe.gauge.physical_a5_sm_requirements import (
        STAGE_DAG_EDGES,
        STAGE_IDS,
    )

    carrier = ladder.get("localCarrier")
    if not isinstance(carrier, Mapping):
        raise LovableExportError("exact_contract_local_carrier_missing")
    counts = carrier.get("counts")
    expected_counts = {
        "portCount": 12,
        "edgeCount": 30,
        "faceCount": 20,
        "antipodalPairCount": 6,
        "a5ActionCount": 60,
    }
    if not isinstance(counts, Mapping) or any(
        counts.get(key) != expected for key, expected in expected_counts.items()
    ):
        raise LovableExportError("exact_contract_icosahedral_counts_invalid")

    ports = carrier.get("ports")
    edges = carrier.get("edges")
    faces = carrier.get("faces")
    antipodes = carrier.get("antipodes")
    rows = (ports, edges, faces, antipodes)
    if not all(isinstance(value, list) for value in rows):
        raise LovableExportError("exact_contract_icosahedral_rows_missing")
    assert isinstance(ports, list)
    assert isinstance(edges, list)
    assert isinstance(faces, list)
    assert isinstance(antipodes, list)
    if tuple(map(len, rows)) != (12, 30, 20, 6):
        raise LovableExportError("exact_contract_icosahedral_row_counts_invalid")
    port_ids = [row.get("portId") for row in ports if isinstance(row, Mapping)]
    if (
        len(port_ids) != 12
        or len(set(port_ids)) != 12
        or any(not value for value in port_ids)
    ):
        raise LovableExportError("exact_contract_port_ids_invalid")
    port_set = set(port_ids)
    edge_pairs: set[frozenset[Any]] = set()
    degree = {port_id: 0 for port_id in port_ids}
    numeric_edges: set[frozenset[int]] = set()
    for row in edges:
        ids = row.get("portIds") if isinstance(row, Mapping) else None
        if not isinstance(ids, list) or len(ids) != 2 or len(set(ids)) != 2:
            raise LovableExportError("exact_contract_edge_invalid")
        if not set(ids) <= port_set:
            raise LovableExportError("exact_contract_edge_unknown_port")
        pair = frozenset(ids)
        if pair in edge_pairs:
            raise LovableExportError("exact_contract_duplicate_edge")
        edge_pairs.add(pair)
        numeric_edges.add(frozenset(port_ids.index(port_id) for port_id in ids))
        for port_id in ids:
            degree[port_id] += 1
    if set(degree.values()) != {5}:
        raise LovableExportError("exact_contract_port_degree_invalid")

    face_sets: set[frozenset[Any]] = set()
    for row in faces:
        ids = row.get("portIds") if isinstance(row, Mapping) else None
        if not isinstance(ids, list) or len(ids) != 3 or len(set(ids)) != 3:
            raise LovableExportError("exact_contract_face_invalid")
        if not set(ids) <= port_set:
            raise LovableExportError("exact_contract_face_unknown_port")
        triangle = frozenset(ids)
        if triangle in face_sets:
            raise LovableExportError("exact_contract_duplicate_face")
        face_sets.add(triangle)
        pairs = ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0]))
        if any(frozenset(pair) not in edge_pairs for pair in pairs):
            raise LovableExportError("exact_contract_face_not_bounded_by_edges")

    antipode_ports: list[Any] = []
    for row in antipodes:
        ids = row.get("portIds") if isinstance(row, Mapping) else None
        if (
            not isinstance(ids, list)
            or len(ids) != 2
            or len(set(ids)) != 2
            or not set(ids) <= port_set
            or frozenset(ids) in edge_pairs
            or row.get("fixedPointFree") is not True
        ):
            raise LovableExportError("exact_contract_antipode_invalid")
        antipode_ports.extend(ids)
    if len(antipode_ports) != 12 or set(antipode_ports) != port_set:
        raise LovableExportError("exact_contract_antipode_partition_invalid")

    a5 = carrier.get("a5")
    if not isinstance(a5, Mapping) or a5.get("order") != 60:
        raise LovableExportError("exact_contract_a5_order_invalid")
    actions = a5.get("actions")
    sectors = a5.get("sectors")
    if not isinstance(actions, list) or len(actions) != 60:
        raise LovableExportError("exact_contract_a5_action_count_invalid")
    canonical_indices = set(range(12))
    action_ids: set[Any] = set()
    permutations: set[tuple[Any, ...]] = set()
    for row in actions:
        if not isinstance(row, Mapping):
            raise LovableExportError("exact_contract_a5_action_invalid")
        action_id = row.get("actionId")
        permutation = row.get("portPermutation")
        if not action_id or action_id in action_ids:
            raise LovableExportError("exact_contract_a5_action_id_invalid")
        if not isinstance(permutation, list) or set(permutation) != canonical_indices:
            raise LovableExportError("exact_contract_a5_permutation_invalid")
        encoded = tuple(permutation)
        if encoded in permutations:
            raise LovableExportError("exact_contract_duplicate_a5_action")
        transformed_edges = {
            frozenset(permutation[index] for index in pair)
            for pair in numeric_edges
        }
        if transformed_edges != numeric_edges:
            raise LovableExportError("exact_contract_a5_action_breaks_edges")
        action_ids.add(action_id)
        permutations.add(encoded)
    expected_sectors = {"1": 1, "3": 3, "3-prime": 3, "5": 5}
    if not isinstance(sectors, list) or {
        row.get("sectorId"): row.get("dimension")
        for row in sectors
        if isinstance(row, Mapping)
    } != expected_sectors:
        raise LovableExportError("exact_contract_a5_sectors_invalid")

    a5_to_sm = ladder.get("a5ToSm")
    nodes = a5_to_sm.get("stageNodes") if isinstance(a5_to_sm, Mapping) else None
    stage_edges = (
        a5_to_sm.get("stageEdges") if isinstance(a5_to_sm, Mapping) else None
    )
    if not isinstance(nodes, list) or [
        row.get("stageId") for row in nodes if isinstance(row, Mapping)
    ] != list(STAGE_IDS):
        raise LovableExportError("exact_contract_stage_ids_invalid")
    actual_edges = [
        (row.get("sourceStageId"), row.get("targetStageId"))
        for row in stage_edges or []
        if isinstance(row, Mapping)
    ]
    if not isinstance(stage_edges, list) or actual_edges != list(STAGE_DAG_EDGES):
        raise LovableExportError("exact_contract_stage_edges_invalid")
    for required in ("screenGeometry", "a5ToStandardModel", "demoUniverse"):
        view = views.get(required)
        if not isinstance(view, Mapping) or view.get("viewId") != required:
            raise LovableExportError(f"exact_contract_view_invalid:{required}")


def _derived_screen_views(
    ladder: Mapping[str, Any],
    *,
    only: set[str] | None = None,
) -> dict[str, Any]:
    # Reuse the simulator's authoritative builders instead of copying their
    # renderer contract into this exporter and allowing the two to drift.
    from oph_fpe.viz.screen_a5_ladder import (
        a5_to_standard_model_view_contract,
        demo_universe_view_contract,
        screen_geometry_view_contract,
    )

    builders = {
        "screenGeometry": screen_geometry_view_contract,
        "a5ToStandardModel": a5_to_standard_model_view_contract,
        "demoUniverse": demo_universe_view_contract,
    }
    selected = set(builders) if only is None else set(only)
    return {
        name: builders[name](ladder)
        for name in builders
        if name in selected
    }


def _assert_demo_assumption_boundary(ladder: Mapping[str, Any]) -> None:
    demo_marker_keys = {
        "forcedfordisplay",
        "forceallstages",
        "postexposuredisplaycandidate",
        "visualizationonly",
    }
    demo_container_keys = {
        "demoselection",
        "frozentargets",
        "frozendisplaytarget",
    }
    demo_active = False
    for _path, key, value in _walk_mapping_values(ladder):
        normalized = _normalize_key(key).replace("_", "")
        if isinstance(value, str) and "demo_assumption" in value.lower():
            demo_active = True
        if normalized in demo_marker_keys and value is True:
            demo_active = True
        if normalized in demo_container_keys and value not in (None, False, {}, []):
            demo_active = True
    if not demo_active:
        return
    unsafe_keys = {
        "promotion_allowed",
        "promotionallowed",
        "scale_campaign_allowed",
        "scalecampaignallowed",
        "target_ancestry_eligible",
        "targetancestryeligible",
    }
    for path, key, value in _walk_mapping_values(ladder):
        if _normalize_key(key) in unsafe_keys and value is True:
            raise LovableExportError(
                f"demo_assumption_cannot_promote_physical_claim:{path}.{key}"
            )


def _walk_mapping_values(
    value: Any,
    path: str = "$.screenA5Ladder",
) -> Iterable[tuple[str, str, Any]]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield path, str(key), nested
            yield from _walk_mapping_values(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from _walk_mapping_values(nested, f"{path}[{index}]")


def _redact_json(
    value: Any,
    *,
    redactions: list[dict[str, str]],
    path: str = "$",
) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, nested in value.items():
            if not isinstance(raw_key, str):
                raise LovableExportError("json_object_keys_must_be_strings")
            child_path = f"{path}.{raw_key}"
            normalized = _normalize_key(raw_key)
            if _is_sensitive_key(normalized):
                result[raw_key] = REDACTION_MARKER
                redactions.append({"path": child_path, "reason": "sensitive_key"})
            elif normalized in PRIVATE_PATH_KEYS:
                result[raw_key] = REDACTION_MARKER
                redactions.append({"path": child_path, "reason": "private_source_path"})
            else:
                result[raw_key] = _redact_json(
                    nested,
                    redactions=redactions,
                    path=child_path,
                )
        return result
    if isinstance(value, list):
        return [
            _redact_json(item, redactions=redactions, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, str) and _looks_like_private_path(value):
        redactions.append({"path": path, "reason": "absolute_filesystem_path"})
        return REDACTION_MARKER
    if isinstance(value, float) and not math.isfinite(value):
        raise LovableExportError("nonfinite_payload_number")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise LovableExportError(f"non_json_payload_value:{type(value).__name__}")


def _resolve_optional_manifest(
    *,
    sidecar_manifest: Path | str | None,
    source_payload_path: Path | None,
    discover_adjacent: bool,
) -> Path | None:
    if sidecar_manifest is not None:
        return Path(sidecar_manifest).expanduser().absolute()
    if source_payload_path is not None and discover_adjacent:
        candidate = source_payload_path.parent / "visualization_export_manifest.json"
        if candidate.exists() or candidate.is_symlink():
            return candidate.absolute()
    return None


def _resolve_sidecar_root(
    manifest_path: Path | None,
    sidecar_root: Path | str | None,
) -> Path | None:
    if manifest_path is None:
        if sidecar_root is not None:
            raise LovableExportError("sidecar_root_requires_manifest")
        return None
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise LovableExportError("sidecar_manifest_missing_or_symlink")
    if _looks_sensitive_filename(manifest_path.name):
        raise LovableExportError("sensitive_manifest_filename")
    root = (
        Path(sidecar_root).expanduser().absolute()
        if sidecar_root is not None
        else manifest_path.parent.absolute()
    )
    if root.is_symlink() or not root.is_dir():
        raise LovableExportError("sidecar_root_missing_or_symlink")
    resolved_root = root.resolve(strict=True)
    try:
        manifest_path.resolve(strict=True).relative_to(resolved_root)
    except ValueError as exc:
        raise LovableExportError("sidecar_manifest_outside_root") from exc
    return resolved_root


def _copy_manifested_sidecars(
    *,
    manifest_path: Path | None,
    sidecar_root: Path | None,
    staging: Path,
    byte_cap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[_SidecarDeclaration]]:
    if manifest_path is None or sidecar_root is None:
        return [], [], []
    manifest = _load_json_bytes(manifest_path.read_bytes())
    if not isinstance(manifest, Mapping):
        raise LovableExportError("sidecar_manifest_root_must_be_object")
    if manifest.get("schema") != RECOGNIZED_SIDECAR_SCHEMA:
        raise LovableExportError("unrecognized_sidecar_manifest_schema")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise LovableExportError("sidecar_manifest_files_must_be_object")
    declarations = sorted(
        _manifest_declarations(files),
        key=lambda row: (row.logical_name, row.raw_path or ""),
    )
    included: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    included_bytes = 0
    seen_sources: set[Path] = set()
    for declaration in declarations:
        base = {"logicalName": _public_logical_name(declaration.logical_name)}
        if "written" in declaration.metadata and not isinstance(
            declaration.metadata.get("written"), bool
        ):
            omitted.append(
                {
                    **base,
                    "sourcePath": _safe_manifest_locator(declaration.raw_path),
                    "reason": "invalid_written_flag",
                }
            )
            continue
        if declaration.written is False:
            omitted.append(
                {
                    **base,
                    "sourcePath": _safe_manifest_locator(declaration.raw_path),
                    "reason": _safe_declared_reason(declaration.reason),
                }
            )
            continue
        if not declaration.raw_path:
            omitted.append({**base, "sourcePath": None, "reason": "source_path_missing"})
            continue
        if _raw_path_has_traversal(declaration.raw_path):
            omitted.append(
                {**base, "sourcePath": UNSAFE_PATH_MARKER, "reason": "path_traversal"}
            )
            continue
        candidate = Path(declaration.raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = manifest_path.parent / candidate
        candidate = candidate.absolute()
        safe, locator, rejection = _admit_sidecar(candidate, sidecar_root)
        if safe is None:
            omitted.append(
                {
                    **base,
                    "sourcePath": locator,
                    "reason": _public_rejection_reason(rejection),
                }
            )
            continue
        if safe in seen_sources:
            omitted.append(
                {**base, "sourcePath": locator, "reason": "duplicate_source_path"}
            )
            continue
        seen_sources.add(safe)
        byte_count = safe.stat().st_size
        declared_byte_count = declaration.metadata.get(
            "byte_count", declaration.metadata.get("byteCount")
        )
        if declared_byte_count is not None and type(declared_byte_count) is not int:
            omitted.append(
                {
                    **base,
                    "sourcePath": locator,
                    "sourceByteCount": byte_count,
                    "reason": "invalid_declared_byte_count",
                }
            )
            continue
        if declared_byte_count is not None and declared_byte_count != byte_count:
            omitted.append(
                {
                    **base,
                    "sourcePath": locator,
                    "sourceByteCount": byte_count,
                    "reason": "declared_byte_count_mismatch",
                }
            )
            continue
        declared_sha256 = declaration.metadata.get("sha256")
        if declared_sha256 is not None and (
            not isinstance(declared_sha256, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", declared_sha256) is None
        ):
            omitted.append(
                {
                    **base,
                    "sourcePath": locator,
                    "sourceByteCount": byte_count,
                    "reason": "invalid_declared_sha256",
                }
            )
            continue
        if included_bytes + byte_count > byte_cap:
            omitted.append(
                {
                    **base,
                    "sourcePath": locator,
                    "sourceByteCount": byte_count,
                    "reason": "sidecar_byte_cap",
                }
            )
            continue
        content_rejection = _unsafe_sidecar_content(safe)
        if content_rejection is not None:
            omitted.append(
                {
                    **base,
                    "sourcePath": locator,
                    "sourceByteCount": byte_count,
                    "reason": _public_rejection_reason(content_rejection),
                }
            )
            continue
        logical_component = _public_logical_name(declaration.logical_name)
        source_tag = hashlib.sha256(locator.encode("utf-8")).hexdigest()[:12]
        target_name = f"{_safe_component(safe.stem)}-{source_tag}{safe.suffix.lower()}"
        target = staging / "sidecars" / logical_component / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        digest, copied_bytes = _copy_and_hash(safe, target)
        if copied_bytes != byte_count or safe.stat().st_size != byte_count:
            target.unlink(missing_ok=True)
            omitted.append(
                {
                    **base,
                    "sourcePath": locator,
                    "sourceByteCount": byte_count,
                    "reason": "source_changed_during_export",
                }
            )
            continue
        if (
            isinstance(declared_sha256, str)
            and digest.lower() != declared_sha256.lower()
        ):
            target.unlink(missing_ok=True)
            omitted.append(
                {
                    **base,
                    "sourcePath": locator,
                    "sourceByteCount": byte_count,
                    "reason": "declared_sha256_mismatch",
                }
            )
            continue
        included_bytes += copied_bytes
        included.append(
            {
                **base,
                "sourcePath": locator,
                "exportedPath": target.relative_to(staging).as_posix(),
                "byteCount": copied_bytes,
                "sha256": digest,
                "mediaType": _media_type(target),
                "metadata": _public_metadata(declaration.metadata),
            }
        )
    return included, omitted, declarations


def _manifest_declarations(
    files: Mapping[str, Any],
) -> Iterable[_SidecarDeclaration]:
    def walk(logical_name: str, value: Any) -> Iterable[_SidecarDeclaration]:
        if isinstance(value, Mapping):
            is_file_entry = "path" in value or "written" in value
            if is_file_entry:
                raw_path = value.get("path")
                yield _SidecarDeclaration(
                    logical_name=logical_name,
                    raw_path=raw_path if isinstance(raw_path, str) and raw_path else None,
                    written=value.get("written") if isinstance(value.get("written"), bool) else None,
                    reason=str(value.get("reason")) if value.get("reason") is not None else None,
                    metadata=value,
                )
            for key, nested in value.items():
                if key in {"path", "source", "payload_path"}:
                    continue
                if isinstance(nested, (Mapping, list)):
                    yield from walk(f"{logical_name}.{key}", nested)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                yield from walk(f"{logical_name}.{index}", nested)

    for key, value in files.items():
        yield from walk(str(key), value)


def _admit_sidecar(
    candidate: Path,
    root: Path,
) -> tuple[Path | None, str, str | None]:
    if candidate.is_symlink():
        return None, UNSAFE_PATH_MARKER, "symlink_not_allowed"
    try:
        lexical_relative = candidate.relative_to(root)
    except ValueError:
        return None, UNSAFE_PATH_MARKER, "outside_sidecar_root"
    cursor = root
    for part in lexical_relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            return None, lexical_relative.as_posix(), "symlink_path_component"
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return None, _safe_manifest_locator(candidate.name), "source_missing"
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return None, UNSAFE_PATH_MARKER, "outside_sidecar_root"
    if not resolved.is_file():
        return None, relative.as_posix(), "source_not_regular_file"
    if _looks_sensitive_relative_path(relative):
        return None, "[REDACTED_SENSITIVE_FILENAME]", "sensitive_filename"
    if resolved.suffix.lower() not in ALLOWED_SIDECAR_SUFFIXES:
        return None, relative.as_posix(), "unsupported_sidecar_type"
    return resolved, relative.as_posix(), None


def _unsafe_sidecar_content(path: Path) -> str | None:
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            return _structured_content_rejection(_load_json_bytes(path.read_bytes()))
        if suffix == ".jsonl":
            with path.open("rb") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    if not raw_line.strip():
                        continue
                    rejection = _structured_content_rejection(_load_json_bytes(raw_line))
                    if rejection:
                        return f"{rejection}:line_{line_number}"
        if suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for field in reader.fieldnames or []:
                    if _is_sensitive_key(_normalize_key(field)):
                        return f"sensitive_content_key:{field}"
                for row_number, row in enumerate(reader, start=2):
                    for field, value in row.items():
                        if value is not None and _looks_like_private_path(value):
                            return f"private_path_content:{field}:row_{row_number}"
    except (OSError, UnicodeDecodeError, csv.Error, LovableExportError):
        return "unreadable_or_invalid_structured_sidecar"
    return None


def _structured_content_rejection(value: Any, path: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = _normalize_key(str(key))
            if _is_sensitive_key(normalized):
                return f"sensitive_content_key:{path}.{key}"
            if normalized in PRIVATE_PATH_KEYS:
                return f"private_path_content:{path}.{key}"
            found = _structured_content_rejection(nested, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found = _structured_content_rejection(nested, f"{path}[{index}]")
            if found:
                return found
    elif isinstance(value, str) and _looks_like_private_path(value):
        return f"private_path_content:{path}"
    return None


def _sidecar_census(
    declarations: Sequence[_SidecarDeclaration],
    included: Sequence[Mapping[str, Any]],
    omitted: Sequence[Mapping[str, Any]],
    cap: int,
) -> dict[str, Any]:
    included_bytes = sum(int(row.get("byteCount", 0)) for row in included)
    known_omitted_bytes = sum(int(row.get("sourceByteCount", 0)) for row in omitted)
    return {
        "full": {
            "declaredFileCount": len(declarations),
            "declaredWrittenFileCount": sum(row.written is not False for row in declarations),
            "knownByteCount": included_bytes + known_omitted_bytes,
        },
        "included": {
            "fileCount": len(included),
            "byteCount": included_bytes,
        },
        "omitted": {
            "fileCount": len(omitted),
            "knownByteCount": known_omitted_bytes,
            "everyDeclaredFileAccountedFor": len(declarations) == len(included) + len(omitted),
        },
        "byteCap": cap,
        "withinByteCap": included_bytes <= cap,
        "selectionOrder": "logical_name_then_declared_path",
    }


def _annotate_visualization_records(
    value: Any,
    *,
    source_digest: str,
    path: str = "$",
    list_item: bool = False,
) -> Any:
    """Add a normalized, deterministic provenance envelope to list records."""

    if isinstance(value, Mapping):
        result = {
            str(key): _annotate_visualization_records(
                nested,
                source_digest=source_digest,
                path=f"{path}.{key}",
                list_item=False,
            )
            for key, nested in value.items()
            if key != "visualizationProvenance"
        }
        existing = value.get("visualizationProvenance")
        record_like = list_item or any(
            key in value
            for key in (
                "recordId",
                "traceId",
                "stageId",
                "carrierId",
                "portId",
                "edgeId",
                "faceId",
                "actionId",
                "sectorId",
                "seamId",
                "candidateId",
                "targetId",
                "viewId",
            )
        )
        if record_like:
            result["visualizationProvenance"] = _provenance_envelope(
                value,
                source_digest=source_digest,
                path=path,
                existing=existing if isinstance(existing, Mapping) else None,
            )
        return result
    if isinstance(value, list):
        return [
            _annotate_visualization_records(
                nested,
                source_digest=source_digest,
                path=f"{path}[{index}]",
                list_item=isinstance(nested, Mapping),
            )
            for index, nested in enumerate(value)
        ]
    return value


def _provenance_envelope(
    record: Mapping[str, Any],
    *,
    source_digest: str,
    path: str,
    existing: Mapping[str, Any] | None,
) -> dict[str, Any]:
    status = _provenance_status(record, existing)
    source_refs = record.get("sourceRefs")
    refs = [str(ref) for ref in source_refs] if isinstance(source_refs, list) else []
    envelope: dict[str, Any] = {
        "schema": PROVENANCE_SCHEMA,
        "status": status,
        "sourceRecordPath": path,
        "sourcePayloadSha256": source_digest,
        "sourceRefs": refs,
        "deterministic": status != "measured",
    }
    if status == "measured":
        envelope["measurementRefs"] = refs
    elif status == "computed":
        envelope["method"] = str(
            record.get("computationMethod") or "exported_simulator_record"
        )
    elif status == "interpolated":
        envelope.update(
            {
                "method": str(
                    record.get("interpolationMethod")
                    or "deterministic_linear_between_exported_samples"
                ),
                "parentRecordIds": refs,
                "weights": list(record.get("interpolationWeights") or []),
            }
        )
    elif status == "synthetic":
        envelope["generator"] = {
            "id": str(record.get("generatorId") or "oph-public-story-synthesis"),
            "version": "1",
            "seed": source_digest[:16],
            "deterministicIndex": int(
                hashlib.sha256(path.encode("utf-8")).hexdigest()[:12], 16
            ),
        }
    elif status == "frozen":
        envelope.update(
            {
                "sourceTarget": refs,
                "targetExposure": str(
                    record.get("targetExposure") or "post_exposure_display_only"
                ),
            }
        )
    return envelope


def _provenance_status(
    record: Mapping[str, Any],
    existing: Mapping[str, Any] | None,
) -> str:
    if existing is not None and existing.get("status") in PROVENANCE_STATUSES:
        return str(existing["status"])
    declared = str(record.get("provenanceClass") or "").lower()
    declared_map = {
        "measured": "measured",
        "run_anchored": "computed",
        "computed": "computed",
        "computed_exact": "computed",
        "interpolated": "interpolated",
        "synthetic": "synthetic",
        "frozen": "frozen",
        "frozen_reference": "frozen",
    }
    if declared in declared_map:
        return declared_map[declared]
    if record.get("measurementDerived") is True:
        return "measured"
    if record.get("interpolated") is True or record.get("interpolationMethod"):
        return "interpolated"
    if (
        record.get("postExposureDisplayCandidate") is True
        or record.get("targetExposure") == "post_exposure_display_only"
        or record.get("frozenForDisplay") is True
    ):
        return "frozen"
    if (
        record.get("trajectoryAssumed") is True
        or record.get("visualizationOnly") is True
        or str(record.get("epistemicStatus") or "").upper() == "DEMO_ASSUMPTION"
    ):
        return "synthetic"
    return "computed"


_PUBLIC_GATE_FIELDS = frozenset(
    {
        "physicalStatus",
        "physicalPassed",
        "physicalPromotion",
        "physicalReceipt",
        "promotion_allowed",
        "receipts",
        "SCALE_CAMPAIGN_ALLOWED",
        "target_ancestry_eligible",
        "watermark",
        "status",
        "epistemicStatus",
        "visualizationOnly",
        "displayStatus",
        "displayComplete",
        "forcedForDisplay",
    }
)


def _strip_public_gate_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_public_gate_fields(nested)
            for key, nested in value.items()
            if key not in _PUBLIC_GATE_FIELDS
        }
    if isinstance(value, list):
        return [_strip_public_gate_fields(row) for row in value]
    return value


def _public_cinematic_story(
    *,
    payload: Mapping[str, Any],
    ladder: Mapping[str, Any],
    source_digest: str,
    public_anchor: Mapping[str, Any] | None,
    anchor_digest: str | None,
) -> dict[str, Any]:
    demo = ladder.get("demoUniverse")
    segments_value = demo.get("segments") if isinstance(demo, Mapping) else None
    segment_records: dict[str, list[Any]] = {}
    if isinstance(segments_value, list):
        for segment in segments_value:
            if not isinstance(segment, Mapping):
                continue
            segment_id = segment.get("segmentId")
            records = segment.get("records")
            if isinstance(segment_id, str) and isinstance(records, list):
                segment_records[segment_id] = records
    sm_records = segment_records.get("forced_sm_catalogue_and_interactions", [])
    sm_by_kind: dict[str, list[Any]] = {}
    for row in sm_records:
        if isinstance(row, Mapping):
            sm_by_kind.setdefault(str(row.get("recordKind") or "other"), []).append(row)
    gravity_records = segment_records.get("gravity_response", [])
    h3_frames = [
        row
        for row in gravity_records
        if isinstance(row, Mapping)
        and row.get("recordKind") == "h3_event_display_frame"
    ]
    if not h3_frames:
        gravity_sources = [
            row
            for row in gravity_records
            if isinstance(row, Mapping)
            and str(row.get("recordId") or "").startswith("gravity-response-")
        ]
        count = max(1, len(gravity_sources))
        h3_frames = [
            {
                "recordId": f"public-h3-synthetic-{index:02d}",
                "recordKind": "h3_event_display_frame",
                "frameIndex": index,
                "poincareBallPosition": [
                    round(0.72 * math.cos(2.0 * math.pi * index / count), 12),
                    round(0.18 * math.sin(4.0 * math.pi * index / count), 12),
                    round(0.72 * math.sin(2.0 * math.pi * index / count), 12),
                ],
                "eventConeRadiusProxy": round(0.08 + 0.3 * index / count, 12),
                "gravityResponseRef": source.get("recordId"),
                "provenanceClass": "synthetic",
                "generatorId": "oph-public-h3-fallback-v1",
                "sourceRefs": [source.get("recordId")],
                "synthesisReason": "no exported H3 display frame; deterministic visual bridge from gravity response",
            }
            for index, source in enumerate(gravity_sources)
        ]
    actor_records = list(sm_by_kind.get("particle_actor", []))
    composite_records = [
        row
        for row in actor_records
        if isinstance(row, Mapping)
        and row.get("actorClass") in {"composite_baryon", "composite_atom"}
    ]
    if not any(
        str(row.get("speciesId") or "").lower() == "proton"
        for row in composite_records
    ):
        actor_ids = {
            row.get("actorId")
            for row in actor_records
            if isinstance(row, Mapping)
        }
        constituents = [
            actor_id
            for actor_id in (
                "particle-actor-up-000",
                "particle-actor-up-000",
                "particle-actor-down-000",
            )
            if actor_id in actor_ids
        ]
        if constituents:
            composite_records.append(
                {
                    "recordId": "public-proton-synthetic-000",
                    "recordKind": "particle_actor",
                    "actorId": "public-proton-synthetic-000",
                    "actorClass": "composite_baryon",
                    "speciesId": "proton",
                    "constituentActorIds": constituents,
                    "provenanceClass": "synthetic",
                    "generatorId": "oph-public-proton-composite-v1",
                    "sourceRefs": list(dict.fromkeys(constituents)),
                    "synthesisReason": "visual composite assembled after emergence from exported quark actor records",
                }
            )
    stage_nodes = (
        ladder.get("a5ToSm", {}).get("stageNodes", [])
        if isinstance(ladder.get("a5ToSm"), Mapping)
        else []
    )
    stage_narrative = [
        {
            "stageId": row.get("stageId"),
            "allDependencies": row.get("allDependencies", []),
            "anyDependencyGroups": row.get("anyDependencyGroups", []),
            "routeIds": row.get("routeIds", []),
            "storyCaption": row.get("claimBoundary"),
        }
        for row in stage_nodes
        if isinstance(row, Mapping)
    ]
    story: dict[str, Any] = {
        "schema": PUBLIC_STORY_SCHEMA,
        "profile": "PUBLIC_CINEMATIC_ILLUSTRATIVE_RECONSTRUCTION",
        "structural16kAnchor": {
            "provided": public_anchor is not None,
            "canonicalSha256": anchor_digest,
            "schema": public_anchor.get("schema") if public_anchor else None,
            "runId": public_anchor.get("run_id") if public_anchor else None,
            "exactCarrierCount": (
                public_anchor.get("structural_exact", {}).get("carrier_count")
                if public_anchor
                else None
            ),
            "exactSeamCount": (
                public_anchor.get("structural_exact", {}).get("seam_count")
                if public_anchor
                else None
            ),
            "physicalDynamicsStarted": (
                public_anchor.get("physical_status", {}).get(
                    "physical_dynamics_started"
                )
                if public_anchor
                else None
            ),
            "settlingTrajectoryAvailable": (
                public_anchor.get("seeded_disorder_disclosure", {}).get(
                    "settling_trajectory_available"
                )
                if public_anchor
                else None
            ),
            "claimBoundary": (
                public_anchor.get("claim_boundary") if public_anchor else None
            ),
        },
        "primaryUx": {
            "gateFree": True,
            "forbiddenPrimaryComponents": [
                "receipt table",
                "PASS/OPEN checklist",
                "gate matrix",
                "force-stage toggles",
                "red/amber blocker dashboard",
            ],
            "persistentDisclosure": PUBLIC_DISCLOSURE,
            "disclosureStyle": "quiet, legible, persistent, never modal",
            "optionalProvenanceDrawer": True,
        },
        "provenanceContract": {
            "schema": PROVENANCE_SCHEMA,
            "allowedStatuses": list(PROVENANCE_STATUSES),
            "recordField": "visualizationProvenance",
            "interpolation": {
                "allowed": True,
                "method": "deterministic linear interpolation between adjacent exported samples of the same stable actor",
                "requiredMetadata": [
                    "status=interpolated",
                    "parentRecordIds",
                    "weights",
                    "method",
                ],
            },
            "synthesis": {
                "allowedOnlyWhenSampleMissing": True,
                "anchorPriority": "use real frozen 16k records wherever they exist; the refused pre-RNG 16k run supplies no settling trajectory, so any settling samples remain synthetic rather than measured or run-anchored",
                "emergenceBoundary": "synthetic particle-family imagery may begin only after the repair-to-emergence transition",
                "requiredMetadata": [
                    "status=synthetic",
                    "stable generated record ID",
                    "generator.id",
                    "generator.version",
                    "generator.seed",
                    "generator.deterministicIndex",
                ],
            },
            "frozen": {
                "requiredMetadata": [
                    "status=frozen",
                    "sourceTarget",
                    "targetExposure=post_exposure_display_only",
                ]
            },
        },
        "chapters": [
            {
                "chapterId": "federation_awakes",
                "title": "A federation of self-reading icosahedra",
                "sceneRef": "sceneData.federation",
                "direction": "Consume sceneData.federation.seededDisorderOpening and open wide on its deterministic seeded disorder across thousands of exact 12-port carriers. Replay provenance-tagged exported light/readback iterations across exported seams and visibly converge toward the fixed-point observable normal form. Because the 16k run refused before RNG, label this settling trajectory synthetic. Do not begin with a pre-solved or prepopulated screen.",
            },
            {
                "chapterId": "inside_one_carrier",
                "title": "Inside one carrier",
                "sceneRef": "sceneData.exactCarrier",
                "direction": "Fly into one exact 12-vertex, 30-edge, 20-face carrier; expose ports, antipodes, faces, and readback without substituting a decorative polyhedron.",
            },
            {
                "chapterId": "repair",
                "title": "Mismatch becomes repair",
                "sceneRef": "sceneData.repair",
                "direction": "Turn overlap mismatch into a visible repair transaction and a settling normal form, retaining record ancestry.",
            },
            {
                "chapterId": "a5_action",
                "title": "The sixty motions of A5",
                "sceneRef": "sceneData.a5",
                "direction": "Browse and animate all 60 exported port permutations, then resolve the 1 + 3 + 3-prime + 5 sectors.",
            },
            {
                "chapterId": "standard_model_ladder",
                "title": "From sectors to the Standard-Model catalogue",
                "sceneRef": "sceneData.standardModel",
                "direction": "Use a clean dependency story with the Q2 alternatives shown as a fork; do not render receipt states in the main scene.",
            },
            {
                "chapterId": "events_and_matter",
                "title": "Particles, interactions, and atoms",
                "sceneRef": "sceneData.eventsAndMatter",
                "direction": "Join actors, worldline samples, interactions, and atom constituents by stable IDs. Never move the camera as if it were a particle.",
            },
            {
                "chapterId": "h3_emergence",
                "title": "Event geometry opens into H3",
                "sceneRef": "sceneData.h3",
                "direction": "Transition from linked repair/event records into the exported H3 chart and finite event frames.",
            },
            {
                "chapterId": "gravity",
                "title": "Matter shapes the observer-visible geometry",
                "sceneRef": "sceneData.gravity",
                "direction": "Animate the exported matter-to-gravity-response joins as curvature-response imagery, preserving their provenance status.",
            },
            {
                "chapterId": "cosmology",
                "title": "From finite readback to a sky",
                "sceneRef": "sceneData.cosmology",
                "direction": "Move through expansion and CMB layers using exported samples, deterministic interpolation, and explicit provenance.",
            },
            {
                "chapterId": "observer_frame_finale",
                "title": "One observer frame",
                "sceneRef": "sceneData.observerFinale",
                "direction": "Finish inside an observer-frame spacetime: modular clock visible, particle families moving, atoms assembled, and gravity responding around them.",
            },
        ],
        "sceneData": {
            "federation": {
                "declaredCarrierCount": (
                    public_anchor.get("structural_exact", {}).get("carrier_count")
                    if public_anchor
                    else ladder.get("federation", {}).get("declaredCarrierCount")
                ),
                "structuralSeamCount": (
                    public_anchor.get("structural_exact", {}).get("seam_count")
                    if public_anchor
                    else len(ladder.get("federation", {}).get("seams", []))
                ),
                "deterministicVisualSeed": (
                    public_anchor.get("seeded_disorder_disclosure", {}).get(
                        "deterministic_visual_seed"
                    )
                    if public_anchor
                    else ladder.get("demoUniverse", {}).get("proceduralSeed")
                ),
                "seededDisorderOpening": (
                    demo.get("seededDisorderOpening", {})
                    if isinstance(demo, Mapping)
                    else {}
                ),
                "carrierInstances": ladder.get("federation", {}).get(
                    "carrierInstances", []
                ),
                "seams": ladder.get("federation", {}).get("seams", []),
                "lightReadbackSamples": segment_records.get(
                    "carrier_light_readback_settling", []
                ),
            },
            "exactCarrier": ladder.get("localCarrier", {}),
            "repair": {
                "bridge": ladder.get("observerRepairBridge", {}),
                "settlingSamples": segment_records.get("repair_fixed_point", []),
            },
            "a5": {
                "actions": ladder.get("localCarrier", {}).get("a5", {}).get(
                    "actions", []
                ),
                "sectors": ladder.get("localCarrier", {}).get("a5", {}).get(
                    "sectors", []
                ),
            },
            "standardModel": {
                "stageNarrative": stage_narrative,
                "species": sm_by_kind.get("particle_species", []),
            },
            "eventsAndMatter": {
                "actors": actor_records,
                "composites": composite_records,
                "worldlineSamples": sm_by_kind.get("particle_worldline_sample", []),
                "interactionFamilies": sm_by_kind.get("interaction_family", []),
                "interactionEvents": sm_by_kind.get("interaction_event", []),
                "atoms": segment_records.get("finite_atom_census", []),
                "traceability": demo.get("traceability", {})
                if isinstance(demo, Mapping)
                else {},
            },
            "h3": {
                "eventFrames": h3_frames,
                "smallUniverse": payload.get("smallUniverse", {}),
                "emergentCurvedSpacetime": payload.get(
                    "emergentCurvedSpacetime", {}
                ),
            },
            "gravity": segment_records.get("gravity_response", []),
            "cosmology": {
                "frames": segment_records.get("cosmology", []),
                "cmbComparison": payload.get("cmbComparison", {}),
            },
            "observerFinale": {
                "cameraFrames": segment_records.get(
                    "virtual_observer_camera", []
                ),
                "observerModularTime": payload.get("observerModularTime", {}),
                "traceability": demo.get("traceability", {})
                if isinstance(demo, Mapping)
                else {},
                "featuredCompositeRecords": composite_records,
            },
        },
    }
    clean = _strip_public_gate_fields(story)
    return _annotate_visualization_records(
        clean,
        source_digest=source_digest,
        path="$.publicCinematicStory",
    )


def _build_export_summary(
    *,
    payload: Mapping[str, Any],
    ladder: Mapping[str, Any],
    input_kind: str,
    view_summaries: Sequence[Mapping[str, Any]],
    sidecar_census: Mapping[str, Any],
    redactions: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    federation = ladder.get("federation") if isinstance(ladder.get("federation"), Mapping) else {}
    demo_universe = (
        ladder.get("demoUniverse")
        if isinstance(ladder.get("demoUniverse"), Mapping)
        else {}
    )
    finite_census = (
        demo_universe.get("finiteCensus")
        if isinstance(demo_universe.get("finiteCensus"), Mapping)
        else {}
    )
    local_carrier = (
        ladder.get("localCarrier")
        if isinstance(ladder.get("localCarrier"), Mapping)
        else {}
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "inputKind": input_kind,
        "payloadSchema": payload.get("schemaVersion") or payload.get("schema"),
        "title": payload.get("title") or "OPH screen-to-emergent-universe visualizer",
        "brief": (
            "A cinematic illustrative reconstruction follows a finite, self-reading "
            "federation of local icosahedral carriers through repair, all sixty A5 "
            "actions, the Standard-Model dependency story, particle events, H3, gravity, "
            "cosmology, and one observer-frame finale."
        ),
        "viewCount": len(view_summaries),
        "views": list(view_summaries),
        "census": {
            "topLevelPayloadSectionCount": len(payload),
            "declaredCarrierCount": _nonnegative_int(federation.get("declaredCarrierCount")),
            "renderedCarrierPrototypeCount": 1 if local_carrier else 0,
            "localCarrierPortCount": _nested_count(local_carrier, "ports", "portCount"),
            "localCarrierEdgeCount": _nested_count(local_carrier, "edges", "edgeCount"),
            "localCarrierFaceCount": _nested_count(local_carrier, "faces", "faceCount"),
            "demoAtomCount": _nonnegative_int(finite_census.get("atomCount")),
            "addressingRule": (
                "Use finite carrier/atom IDs and request visible chunks. A census means every "
                "object is addressable, not that every object must be drawn in one frame."
            ),
        },
        "sidecarCensus": sidecar_census,
        "redactionCount": len(redactions),
        "epistemicBoundary": _epistemic_boundary(),
    }


def _view_summaries(views: Mapping[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for key, raw in views.items():
        if not isinstance(raw, Mapping):
            continue
        view_id = str(raw.get("viewId") or key)
        label = str(raw.get("label") or _humanize(view_id))
        section_kind = str(raw.get("sectionKind") or "visualization_contract")
        description = raw.get("description")
        if not isinstance(description, str) or not description.strip():
            description = (
                f"{label} renders the {section_kind.replace('_', ' ')} view from its "
                "declared data sources and receipt boundary."
            )
        summaries.append(
            {
                "viewId": view_id,
                "label": label,
                "sectionKind": section_kind,
                "summary": description.strip(),
                "contractRef": f"contracts/visualization_views.json#/{key}",
            }
        )
    return summaries


def _agent_instructions(view_summaries: Sequence[Mapping[str, Any]]) -> str:
    view_lines = "\n".join(
        f"- `{row['viewId']}` — {row['summary']}" for row in view_summaries
    )
    return f"""# Instructions for the Lovable public visualizer agent

Start with the [detailed screen/A5-to-SM brief](documentation/SCREEN_A5_SM_VISUALIZER_AGENT_BRIEF.md).
Its **public cinematic profile** is authoritative for this handoff. Instructions
in that document about receipt tables, gate matrices, force toggles, PASS/OPEN
badges, and blocker dashboards apply only to the separate local analyzer.

## Data entry points

- [Primary public cinematic story](contracts/public_cinematic_story.json)
- [Validated public 16k structural anchor, when supplied](contracts/public_visualization_anchor.json)
- [Canonical redacted payload](payload/visualization_payload.json)
- [Concise payload and view summary](payload/summary.json)
- [Exact screen/A5 ladder](contracts/screen_a5_ladder.json)
- [Exact local icosahedral prototype](contracts/local_carrier_prototype.json)
- [Exact visualization view contracts](contracts/visualization_views.json)
- [Export provenance and omission ledger](export_manifest.json)

Do not look for a ZIP. This handoff is an ordinary directory. Files under
`sidecars/` are an optional byte-capped subset of the full local dataset. The
manifest records the complete declared census and a reason for every omission.

The public app must consume `public_cinematic_story.json`. The optional
provenance drawer may show stable record IDs, hashes, source references, and
provenance labels only. It must not expose raw receipts, blockers, gate state,
PASS/OPEN status, or force controls. Those remain exclusive to the separate
local analyzer.

## Public experience

Build one continuous, cinematic OPH story:

1. Open on a **seeded disordered/random state**, not a pre-solved screen, across
   the full visible federation of exact 12-port icosahedral compute carriers.
   Replay provenance-tagged exported light/readback iterations through each
   visible carrier and across exported seam links until the fixed-point
   observable normal form becomes visible. The current 16k physical run refused
   before RNG, so its absent settling trajectory cannot be called measured or
   run-anchored; use deterministic samples tagged `synthetic`.
2. Zoom into one exact 12-port / 30-edge / 20-face carrier and make ports,
   antipodes, faces, and local self-reading legible.
3. Follow overlap mismatch through observer repair and a stable normal form.
4. Let the viewer browse or animate **all 60 exported A5 actions**, then resolve
   the `1 + 3 + 3-prime + 5` sectors.
5. Unfold the A5-to-SM dependency story, particle catalogue, interaction events,
   worldlines, and atoms using ID joins from the story contract.
6. Transition through H3 event geometry, gravity-response imagery, and
   cosmology.
7. End inside the observer-frame spacetime with the modular clock visible and
   particle families, composite matter/atoms, and gravity all in motion.

Use cinematic camera motion, depth, glow, restrained labels, and smooth chapter
transitions. Keep performance bounded with instancing, level of detail, visible
windows, pausing while hidden, and a composed static frame for reduced motion.

The public app must not contain receipt tables, gate checklists, PASS/OPEN
columns, force-stage controls, or red/amber blocker panels anywhere. This
prohibition includes its optional provenance drawer. Keep the exact lowercase phrase
`{PUBLIC_DISCLOSURE}` subtly visible at all times. The collapsed provenance
drawer is limited to record IDs, hashes, source references, and provenance
labels.

## Record and sample provenance

Every supplied scene record carries `visualizationProvenance.status`, whose only
allowed values are `measured`, `computed`, `interpolated`, `synthetic`, and
`frozen`. Preserve that envelope through transforms and joins.

Missing in-between visual samples may be generated only as deterministic linear
interpolation between adjacent samples of the same stable actor. Give each
interpolated sample a stable ID, both parent IDs, weights, method, and status
`interpolated`. If an essential sample has no valid parents, deterministic
model-based synthesis is allowed: give it a stable ID, generator ID/version,
seed, deterministic index, source refs, and status `synthetic`. Never relabel a
frozen target as computed or measured. Never infer `measured` without an
explicit measurement source.

Prefer real frozen 16k records as quantitative anchors wherever they actually
exist. The refused pre-RNG 16k run contributes no physical settling samples;
do not relabel a synthetic settling sequence as measured or run-anchored. Any
model-based particle-family reconstruction must approximate available 16k
records as closely as the exported joins allow, must start only after the explicit
repair-to-emergence transition, and must remain tagged `synthetic` (or
`interpolated` for valid same-actor in-betweens). Do not visually seed particles,
atoms, or a solved geometry into the opening federation.

## Supplied view summaries

These summaries are available to the provenance drawer and chapter authoring;
they are not a requirement for permanent technical panels beside the canvas.

{view_lines}

## Immutable scientific boundary

This public treatment is an illustrative reconstruction. It never alters
physical receipts, authorizes scale, or creates target ancestry. The renderer
may interpolate or synthesize missing *visual samples* under the provenance
rules above; it may not fabricate receipt data or claim that omitted sidecars
were present. Use finite procedural addressing, paging, and level-of-detail
rendering for large censuses.
"""


def _epistemic_boundary() -> dict[str, Any]:
    return {
        "modeDefault": "RECEIPT",
        "demoMode": "DEMO_ASSUMPTION",
        "physicalReceiptSnapshotImmutable": True,
        "scientificReceiptsChangedByExport": False,
        "forcedDisplayValuesArePhysicalReceipts": False,
        "promotion_allowed": False,
        "SCALE_CAMPAIGN_ALLOWED": False,
        "target_ancestry_eligible": False,
        "demoWritesToSimulator": False,
        "frozenTargets": "post_exposure_display_only",
    }


def _inventory_files(staging: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(staging.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(staging).as_posix()
        rows.append(
            {
                "path": relative,
                "byteCount": path.stat().st_size,
                "sha256": _sha256_file(path),
                "mediaType": _media_type(path),
            }
        )
    return rows


def _copy_and_hash(source: Path, target: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    with source.open("rb") as reader, target.open("xb") as writer:
        while True:
            chunk = reader.read(1024 * 1024)
            if not chunk:
                break
            writer.write(chunk)
            digest.update(chunk)
            count += len(chunk)
    return digest.hexdigest(), count


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(value))


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LovableExportError("value_is_not_canonical_json") from exc


def _json_roundtrip(value: Any) -> Any:
    return _load_json_bytes(_canonical_json_bytes(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _media_type(path: Path) -> str:
    overrides = {
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".csv": "text/csv",
        ".bin": "application/octet-stream",
        ".npy": "application/octet-stream",
        ".npz": "application/octet-stream",
        ".md": "text/markdown",
    }
    return overrides.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or (
        "application/octet-stream"
    )


def _validated_cap(value: int) -> int:
    if type(value) is not int:
        raise LovableExportError("sidecar_byte_cap_must_be_nonnegative_integer")
    if value < 0:
        raise LovableExportError("sidecar_byte_cap_must_be_nonnegative_integer")
    return value


def _normalize_key(value: str) -> str:
    snake_hint = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    return re.sub(r"[^a-z0-9]+", "_", snake_hint.lower()).strip("_")


def _is_sensitive_key(normalized: str) -> bool:
    segments = set(normalized.split("_"))
    sensitive_segments = {
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "passphrase",
        "password",
        "secret",
        "token",
    }
    return (
        normalized in SENSITIVE_KEYS
        or bool(segments & sensitive_segments)
        or normalized.endswith("_private_key")
    )


def _looks_like_private_path(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if stripped.lower().startswith("file://"):
        return True
    if stripped.startswith(("/", "~/")) or bool(_WINDOWS_ABSOLUTE_RE.match(stripped)):
        return True
    # Diagnostics frequently wrap an absolute source path in prose, for
    # example ``FileNotFoundError: '/Users/name/workspace/file.json'``. A
    # start-of-string check alone would leak that path while claiming the
    # export was redacted. Recognize common private filesystem roots wherever
    # they occur, while avoiding ordinary https://host/path URLs.
    return bool(
        _EMBEDDED_POSIX_PRIVATE_PATH_RE.search(stripped)
        or _EMBEDDED_WINDOWS_PRIVATE_PATH_RE.search(stripped)
    )


def _raw_path_has_traversal(value: str) -> bool:
    normalized = value.replace("\\", "/")
    return ".." in Path(normalized).parts or "\x00" in value


def _looks_sensitive_filename(name: str) -> bool:
    lower = name.lower()
    if lower in SENSITIVE_FILENAMES or lower.startswith(".env"):
        return True
    if any(
        token in lower
        for token in (
            "credential",
            "private-key",
            "private_key",
            "secret",
            "password",
            "access-token",
            "access_token",
            "auth-token",
            "auth_token",
        )
    ):
        return True
    return Path(lower).suffix in SENSITIVE_SUFFIXES


def _looks_sensitive_relative_path(relative: Path) -> bool:
    for part in relative.parts:
        if part.startswith(".") or _looks_sensitive_filename(part):
            return True
    return False


def _safe_manifest_locator(raw: str | Path | None) -> str | None:
    if raw is None:
        return None
    value = str(raw)
    if _looks_like_private_path(value) or _raw_path_has_traversal(value):
        return UNSAFE_PATH_MARKER
    if _looks_sensitive_filename(Path(value).name):
        return "[REDACTED_SENSITIVE_FILENAME]"
    return value


def _deduplicated_redactions(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    unique = {
        (str(row.get("path", "")), str(row.get("reason", "")))
        for row in rows
    }
    return [
        {"path": path, "reason": reason}
        for path, reason in sorted(unique)
    ]


def _safe_component(value: str) -> str:
    safe = _SAFE_COMPONENT_RE.sub("-", value).strip(".-_").lower()
    if not safe:
        safe = "sidecar"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{safe[:72]}-{digest}"


def _public_logical_name(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"dataset-{digest}"


def _safe_declared_reason(value: str | None) -> str:
    allowed = {
        "producer_disabled",
        "source_not_written",
        "not_requested",
        "not_available",
    }
    return value if value in allowed else "producer_reported_not_written"


def _public_rejection_reason(value: str | None) -> str:
    if not value:
        return "unsafe_sidecar"
    prefixes = {
        "private_path_content": "private_path_content",
        "sensitive_content_key": "sensitive_content_key",
        "unreadable_or_invalid_structured_sidecar": (
            "unreadable_or_invalid_structured_sidecar"
        ),
    }
    for prefix, public in prefixes.items():
        if value.startswith(prefix):
            return public
    controlled = {
        "outside_sidecar_root",
        "path_traversal",
        "source_missing",
        "source_not_regular_file",
        "sensitive_filename",
        "symlink_not_allowed",
        "symlink_path_component",
        "unsupported_sidecar_type",
    }
    return value if value in controlled else "unsafe_sidecar"


def _public_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        public: value[source]
        for source, public in PUBLIC_METADATA_KEYS.items()
        if isinstance(value.get(source), (str, int, float, bool))
    }


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _nested_count(value: Mapping[str, Any], list_key: str, count_key: str) -> int:
    rows = value.get(list_key)
    if isinstance(rows, list):
        return len(rows)
    counts = value.get("counts")
    if isinstance(counts, Mapping):
        return _nonnegative_int(counts.get(count_key))
    return 0


def _humanize(value: str) -> str:
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", value).replace("_", " ")
    return spaced[:1].upper() + spaced[1:]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a directory-based Lovable visualizer handoff."
    )
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sidecar-manifest", type=Path)
    parser.add_argument("--sidecar-root", type=Path)
    parser.add_argument("--public-anchor", type=Path)
    parser.add_argument(
        "--max-sidecar-bytes",
        type=int,
        default=DEFAULT_SIDECAR_BYTE_CAP,
    )
    parser.add_argument(
        "--no-discover-adjacent-manifest",
        action="store_true",
    )
    args = parser.parse_args(argv)
    report = build_lovable_export(
        args.payload,
        args.out_dir,
        sidecar_manifest=args.sidecar_manifest,
        sidecar_root=args.sidecar_root,
        public_anchor=args.public_anchor,
        max_sidecar_bytes=args.max_sidecar_bytes,
        discover_adjacent_manifest=not args.no_discover_adjacent_manifest,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
