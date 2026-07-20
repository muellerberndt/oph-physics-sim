"""Fail-closed verifier for the issue-572 common-domain source tower.

The manifest handled here is evidence, not authority.  Receipt booleans in a
manifest are ignored.  The verifier resolves regular files below one bundle
root, recomputes their hashes, replays a small allowlisted evaluator language,
rebuilds the typed provenance DAG, and executes the numerical splice,
refinement, realization, gauge, and repair-schedule checks.

No manifest-provided code or command is executed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np


MANIFEST_SCHEMA = "oph.typed-common-domain-source-tower.manifest.v1"
REPORT_SCHEMA = "oph.typed-common-domain-source-tower.verification.v1"
REPORT_ARTIFACT_TYPE = "oph_typed_common_domain_source_tower_verification"
DEFAULT_MANIFEST_NAME = "common_domain_source_tower_manifest.json"
DEFAULT_REPORT_NAME = "common_domain_source_tower_verification.json"

COMMON_DOMAIN_SOURCE_TOWER_RECEIPT = "COMMON_DOMAIN_SOURCE_TOWER_RECEIPT"
ECHOSAHEDRAL_TO_ABSTRACT_PATCH_NET_REALIZATION_RECEIPT = (
    "ECHOSAHEDRAL_TO_ABSTRACT_PATCH_NET_REALIZATION_RECEIPT"
)
SOURCE_TOWER_PROVENANCE_GRAPH_RECEIPT = "SOURCE_TOWER_PROVENANCE_GRAPH_RECEIPT"
SOURCE_TOWER_NO_TARGET_PATH_RECEIPT = "SOURCE_TOWER_NO_TARGET_PATH_RECEIPT"
SOURCE_TOWER_REFINEMENT_COMMUTATION_RECEIPT = (
    "SOURCE_TOWER_REFINEMENT_COMMUTATION_RECEIPT"
)
SOURCE_TOWER_CROSS_SOURCE_SPLICE_REJECTION_RECEIPT = (
    "SOURCE_TOWER_CROSS_SOURCE_SPLICE_REJECTION_RECEIPT"
)
ARRAY_CHANNEL_REALIZATION_DIAGNOSTIC_RECEIPT = (
    "ARRAY_CHANNEL_REALIZATION_DIAGNOSTIC_RECEIPT"
)
DECLARED_TARGET_PATH_FIREWALL_DIAGNOSTIC_RECEIPT = (
    "DECLARED_TARGET_PATH_FIREWALL_DIAGNOSTIC_RECEIPT"
)

C0_RECEIPT_KEYS = frozenset(
    {
        COMMON_DOMAIN_SOURCE_TOWER_RECEIPT,
        ECHOSAHEDRAL_TO_ABSTRACT_PATCH_NET_REALIZATION_RECEIPT,
        SOURCE_TOWER_PROVENANCE_GRAPH_RECEIPT,
        SOURCE_TOWER_NO_TARGET_PATH_RECEIPT,
        SOURCE_TOWER_REFINEMENT_COMMUTATION_RECEIPT,
        SOURCE_TOWER_CROSS_SOURCE_SPLICE_REJECTION_RECEIPT,
    }
)

REQUIRED_ROLES = (
    "authoritative_presentation",
    "quotient_normal_form",
    "physical_coarse_maps",
    "protected_boundary",
    "repair_log",
    "cap_algebras",
    "state",
    "modular_data",
    "semantic_event_graph",
    "null_charges",
    "stress",
    "entropy",
    "scale",
)

REFINEMENT_ROLES = (
    "cap_algebras",
    "state",
    "modular_data",
    "semantic_event_graph",
    "null_charges",
    "stress",
    "entropy",
    "scale",
)

REALIZATION_CHANNELS = (
    "accessible_algebras",
    "port_restrictions",
    "records",
    "repairs",
    "checkpoints",
    "semantic_event_history",
    "physical_quotient",
)

FORBIDDEN_TARGET_SEMANTICS = frozenset(
    {
        "expected_normalization",
        "target_signature",
        "target_coupling",
        "target_vacuum",
        "target_scale",
        "einstein_conclusion",
    }
)

ALLOWED_ARTIFACT_CLASSES = frozenset(
    {
        "source_primitive",
        "authoritative_source",
        "readout",
        "typed_arrow",
        "evaluator",
        "configuration",
        "seed",
        "forbidden_target",
        "negative_control",
    }
)
ALLOWED_FORMATS = frozenset({"json", "npy", "npz", "raw"})
ALLOWED_OPERATIONS = frozenset(
    {"identity", "npz_extract", "numpy_matmul", "json_pointer"}
)
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_REPAIR_SCHEDULES = 720

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "bundle_id",
        "artifacts",
        "role_bindings",
        "provenance_processes",
        "splice_controls",
        "refinement_squares",
        "realization",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "artifact_id",
        "path",
        "format",
        "artifact_class",
        "semantic_role",
        "sha256",
        "array_key",
    }
)
_PROCESS_FIELDS = frozenset(
    {
        "process_id",
        "data_input_artifact_ids",
        "output_artifact_id",
        "evaluator_artifact_id",
        "configuration_artifact_id",
        "seed_artifact_id",
    }
)

_REQUIRED_ROLE_CLASSES = {
    "authoritative_presentation": "authoritative_source",
    "physical_coarse_maps": "typed_arrow",
    **{
        role: "readout"
        for role in REQUIRED_ROLES
        if role not in {"authoritative_presentation", "physical_coarse_maps"}
    },
}


class ManifestError(ValueError):
    """Strict manifest parsing failure."""


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_id: str
    path: str
    format: str
    artifact_class: str
    semantic_role: str
    sha256: str
    array_key: str | None = None


@dataclass(frozen=True)
class ProcessSpec:
    process_id: str
    data_input_artifact_ids: tuple[str, ...]
    output_artifact_id: str
    evaluator_artifact_id: str
    configuration_artifact_id: str
    seed_artifact_id: str


@dataclass(frozen=True)
class TypedSourceTowerManifest:
    path: Path
    base_dir: Path
    bundle_id: str
    artifacts: dict[str, ArtifactSpec]
    role_bindings: dict[str, str]
    processes: tuple[ProcessSpec, ...]
    splice_controls: Mapping[str, Any]
    refinement_squares: tuple[Mapping[str, Any], ...]
    realization: Mapping[str, Any]
    raw_sha256: str


class _ArtifactStore:
    def __init__(self, manifest: TypedSourceTowerManifest):
        self.manifest = manifest
        self._values: dict[str, Any] = {}
        self._paths: dict[str, Path] = {}
        self.rows: dict[str, dict[str, Any]] = {}
        self.blockers: list[str] = []

    def verify_all(self) -> None:
        for artifact_id in sorted(self.manifest.artifacts):
            self._verify_one(artifact_id)

    def value(self, artifact_id: str) -> Any:
        if artifact_id not in self._values:
            self._verify_one(artifact_id)
        if artifact_id not in self._values:
            raise ManifestError(f"artifact_unavailable:{artifact_id}")
        return self._values[artifact_id]

    def actual_sha256(self, artifact_id: str) -> str | None:
        return self.rows.get(artifact_id, {}).get("actual_sha256")

    def valid(self, artifact_id: str) -> bool:
        return self.rows.get(artifact_id, {}).get("passed") is True

    def _verify_one(self, artifact_id: str) -> None:
        if artifact_id in self.rows:
            return
        spec = self.manifest.artifacts[artifact_id]
        row: dict[str, Any] = {
            "artifact_id": artifact_id,
            "declared_path": spec.path,
            "format": spec.format,
            "artifact_class": spec.artifact_class,
            "semantic_role": spec.semantic_role,
            "declared_sha256": spec.sha256,
            "passed": False,
            "blockers": [],
        }
        blockers: list[str] = row["blockers"]
        try:
            relative = Path(spec.path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ManifestError("path_is_absolute_or_contains_parent_traversal")
            unresolved = self.manifest.base_dir / relative
            if unresolved.is_symlink():
                raise ManifestError("artifact_path_is_symlink")
            resolved = unresolved.resolve(strict=True)
            resolved.relative_to(self.manifest.base_dir.resolve(strict=True))
            if not resolved.is_file():
                raise ManifestError("artifact_path_is_not_regular_file")
            size = resolved.stat().st_size
            if size > MAX_ARTIFACT_BYTES:
                raise ManifestError("artifact_exceeds_size_limit")
            raw = resolved.read_bytes()
            actual = "sha256:" + hashlib.sha256(raw).hexdigest()
            row["byte_length"] = size
            row["actual_sha256"] = actual
            if actual != spec.sha256:
                raise ManifestError("declared_sha256_mismatch")
            value = _decode_artifact(spec, raw, resolved)
            _validate_finite_value(value)
            row["value_metadata"] = _value_metadata(value)
            row["decoded_value_sha256"] = _decoded_value_sha256(value)
            row["resolved_relative_path"] = resolved.relative_to(
                self.manifest.base_dir.resolve(strict=True)
            ).as_posix()
            row["passed"] = True
            self._values[artifact_id] = value
            self._paths[artifact_id] = resolved
        except (OSError, ValueError, ManifestError) as exc:
            blockers.append(str(exc))
            self.blockers.append(f"artifact:{artifact_id}:{exc}")
        self.rows[artifact_id] = row


def verify_common_domain_source_tower(
    manifest: str | Path | Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Strictly verify one common-domain source-tower manifest."""

    try:
        parsed = _parse_manifest(manifest, base_dir=base_dir)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ManifestError) as exc:
        return _incomplete_report(
            blocker=f"manifest_parse_failed:{type(exc).__name__}:{exc}",
            manifest_path=str(manifest) if not isinstance(manifest, Mapping) else None,
        )

    try:
        store = _ArtifactStore(parsed)
        store.verify_all()
        # Keep the live graph private to this invocation.  Only its deterministic
        # ledger is serialized, so reports are portable JSON rather than pickled
        # Python authority objects.
        provenance_internal = _verify_provenance(parsed, store)
        no_target = _verify_no_target_paths(parsed, store, provenance_internal)
        refinement = _verify_refinement(parsed, store, provenance_internal)
        splices = _verify_splice_controls(parsed, store, provenance_internal)
        realization = _verify_realization(parsed, store, provenance_internal)
    except Exception as exc:  # malformed evidence must fail closed, never abort a campaign
        return _incomplete_report(
            blocker=f"verification_failed:{type(exc).__name__}:{exc}",
            manifest_path=parsed.path.name,
        )
    provenance = {
        key: value
        for key, value in provenance_internal.items()
        if key != "graph"
    }

    provenance_receipt = bool(
        not store.blockers and provenance["passed"] is True
    )
    declared_no_target_receipt = bool(
        provenance_receipt and no_target["passed"] is True
    )
    # A manifest DAG plus a token scan can reject declared target paths, but it
    # cannot establish semantic noninterference of arbitrary generator code
    # (encoded constants, aliases, or target-derived matrices).  Keep the
    # physical no-target receipt false until generator dependencies are replayed
    # under a stronger code/provenance firewall.
    no_target_receipt = False
    refinement_receipt = bool(
        provenance_receipt and refinement["passed"] is True
    )
    splice_receipt = bool(
        provenance_receipt and splices["passed"] is True
    )
    array_realization_receipt = bool(
        provenance_receipt
        and realization.get("array_channel_contract_passed") is True
    )
    # The legacy N x 12 channel test below verifies typed numerical arrows, but
    # it does not reconstruct an EchosahedralFederation, its seams/collars,
    # higher overlaps, records, checkpoints, or quotient transition law.  Keep
    # the physical realization receipt false until a replayed federation bundle
    # and those channel-specific preservation maps are bound into this manifest.
    realization_receipt = False
    common_receipt = bool(
        provenance_receipt
        and declared_no_target_receipt
        and provenance["required_roles_share_one_source"] is True
        and provenance["all_required_readouts_reconstructed"] is True
    )
    blockers = sorted(
        set(
            store.blockers
            + provenance["blockers"]
            + no_target["blockers"]
            + refinement["blockers"]
            + splices["blockers"]
            + realization["blockers"]
            + [
                "generator_code_dependency_firewall_not_bound",
                "encoded_or_aliased_target_noninterference_not_proved",
            ]
        )
    )
    report = {
        "schema": REPORT_SCHEMA,
        "artifact_type": REPORT_ARTIFACT_TYPE,
        "issue": 572,
        "manifest_path": parsed.path.name,
        "manifest_sha256": parsed.raw_sha256,
        "bundle_id": parsed.bundle_id,
        "computed_bundle_commitment": _bundle_commitment(parsed, store),
        "verifier_module_sha256": _module_sha256(),
        "artifact_verification": {
            "rows": [store.rows[key] for key in sorted(store.rows)],
            "all_declared_hashes_recomputed": not store.blockers,
        },
        "provenance": provenance,
        "target_free_source_path": no_target,
        "refinement_commutation": refinement,
        "cross_source_splice_controls": splices,
        "echosahedral_abstract_realization": realization,
        COMMON_DOMAIN_SOURCE_TOWER_RECEIPT: common_receipt,
        ARRAY_CHANNEL_REALIZATION_DIAGNOSTIC_RECEIPT: array_realization_receipt,
        DECLARED_TARGET_PATH_FIREWALL_DIAGNOSTIC_RECEIPT: (
            declared_no_target_receipt
        ),
        ECHOSAHEDRAL_TO_ABSTRACT_PATCH_NET_REALIZATION_RECEIPT: realization_receipt,
        SOURCE_TOWER_PROVENANCE_GRAPH_RECEIPT: provenance_receipt,
        SOURCE_TOWER_NO_TARGET_PATH_RECEIPT: no_target_receipt,
        SOURCE_TOWER_REFINEMENT_COMMUTATION_RECEIPT: refinement_receipt,
        SOURCE_TOWER_CROSS_SOURCE_SPLICE_REJECTION_RECEIPT: splice_receipt,
        "receipt": bool(
            common_receipt
            and realization_receipt
            and provenance_receipt
            and no_target_receipt
            and refinement_receipt
            and splice_receipt
        ),
        "blockers": blockers,
        "claim_boundary": (
            "Issue-572 finite common-domain provenance and commuting-map verification. "
            "It authenticates one typed array/readout source tower under the pinned built-in "
            "evaluator threat model. The N-by-12 channel diagnostic is not a physical "
            "EchosahedralFederation realization: seams, collars, higher overlaps, semantic "
            "records/checkpoints, and quotient dynamics must be independently replayed and "
            "bound first. The declared target-path diagnostic likewise does not prove "
            "semantic noninterference of arbitrary generator code. It is not an Einstein "
            "equation, continuum, gravity, or "
            "Standard-Model promotion receipt. Origin claims beyond local hashes require "
            "signed build/transparency attestation."
        ),
    }
    report["verification_report_sha256"] = _canonical_report_hash(report)
    return report


def write_common_domain_source_tower_report(
    manifest: str | Path,
    out: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest).resolve(strict=True)
    report = verify_common_domain_source_tower(manifest_path)
    destination = (
        Path(out)
        if out is not None
        else manifest_path.parent / DEFAULT_REPORT_NAME
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def verify_common_domain_source_tower_report(
    report: Mapping[str, Any],
    *,
    report_path: str | Path,
) -> dict[str, Any]:
    """Re-run the referenced manifest; never trust stored C0 booleans."""

    candidate = dict(report)
    blockers: list[str] = []
    if candidate.get("schema") != REPORT_SCHEMA:
        blockers.append("report_schema_mismatch")
    if candidate.get("artifact_type") != REPORT_ARTIFACT_TYPE:
        blockers.append("report_artifact_type_mismatch")
    manifest_name = candidate.get("manifest_path")
    if not isinstance(manifest_name, str) or not manifest_name:
        return {"passed": False, "blockers": [*blockers, "manifest_path_missing"]}
    relative = Path(manifest_name)
    if relative.is_absolute() or ".." in relative.parts:
        return {"passed": False, "blockers": [*blockers, "manifest_path_unsafe"]}
    source = Path(report_path).resolve().parent / relative
    recomputed = verify_common_domain_source_tower(source)
    for key in sorted(C0_RECEIPT_KEYS):
        if candidate.get(key) is not recomputed.get(key):
            blockers.append(f"stored_receipt_mismatch:{key}")
    for key in (
        "manifest_sha256",
        "computed_bundle_commitment",
        "verifier_module_sha256",
        "verification_report_sha256",
    ):
        if candidate.get(key) != recomputed.get(key):
            blockers.append(f"stored_verification_field_mismatch:{key}")
    if not _values_equal(candidate, recomputed):
        blockers.append("stored_report_not_exact_verifier_output")
    return {
        "passed": not blockers and recomputed.get("receipt") is True,
        "blockers": blockers + list(recomputed.get("blockers") or []),
        "recomputed_report": recomputed,
    }


def verify_common_source_tower_report(
    report: Mapping[str, Any],
    *,
    report_path: str | Path,
) -> dict[str, Any]:
    """Public strict C0 validator used by downstream ladder consumers.

    This deliberately returns the full validation result rather than a bare
    truthy object.  Consumers must require ``result["passed"] is True``.
    """

    return verify_common_domain_source_tower_report(
        report,
        report_path=report_path,
    )


def verify_common_source_tower_report_file(
    report_path: str | Path,
) -> dict[str, Any]:
    """Strictly decode and replay one on-disk C0 verification artifact."""

    candidate_path = Path(report_path)
    try:
        if candidate_path.is_symlink():
            raise ManifestError("report_path_is_symlink")
        resolved = candidate_path.resolve(strict=True)
        if not resolved.is_file():
            raise ManifestError("report_path_is_not_regular_file")
        raw = resolved.read_bytes()
        if len(raw) > MAX_JSON_BYTES:
            raise ManifestError("report_exceeds_json_size_limit")
        candidate = _strict_json_loads(raw)
        if not isinstance(candidate, Mapping):
            raise ManifestError("report_root_must_be_object")
        return verify_common_source_tower_report(
            candidate,
            report_path=resolved,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ManifestError,
    ) as exc:
        return {
            "passed": False,
            "blockers": [f"report_parse_failed:{type(exc).__name__}:{exc}"],
            "recomputed_report": None,
        }


def _parse_manifest(
    manifest: str | Path | Mapping[str, Any],
    *,
    base_dir: str | Path | None,
) -> TypedSourceTowerManifest:
    if isinstance(manifest, Mapping):
        if base_dir is None:
            raise ManifestError("base_dir_required_for_mapping_manifest")
        raw_payload = dict(manifest)
        encoded = _canonical_json_bytes(raw_payload)
        path = Path(base_dir).resolve() / DEFAULT_MANIFEST_NAME
        root = Path(base_dir).resolve(strict=True)
    else:
        path = Path(manifest).resolve(strict=True)
        raw = path.read_bytes()
        if len(raw) > MAX_JSON_BYTES:
            raise ManifestError("manifest_exceeds_json_size_limit")
        raw_payload = _strict_json_loads(raw)
        encoded = raw
        root = path.parent.resolve(strict=True)
    if not isinstance(raw_payload, Mapping):
        raise ManifestError("manifest_root_must_be_object")
    unknown = set(raw_payload) - _TOP_LEVEL_FIELDS
    missing = _TOP_LEVEL_FIELDS - set(raw_payload)
    if unknown or missing:
        raise ManifestError(
            f"manifest_fields_invalid:unknown={sorted(unknown)}:missing={sorted(missing)}"
        )
    if raw_payload.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError("manifest_schema_mismatch")
    bundle_id = raw_payload.get("bundle_id")
    if not isinstance(bundle_id, str) or not bundle_id:
        raise ManifestError("bundle_id_missing")

    artifact_rows = raw_payload.get("artifacts")
    if not isinstance(artifact_rows, list) or not artifact_rows:
        raise ManifestError("artifacts_must_be_nonempty_array")
    artifacts: dict[str, ArtifactSpec] = {}
    for index, row in enumerate(artifact_rows):
        if not isinstance(row, Mapping):
            raise ManifestError(f"artifact_{index}_must_be_object")
        unknown = set(row) - _ARTIFACT_FIELDS
        required = _ARTIFACT_FIELDS - {"array_key"}
        missing = required - set(row)
        if unknown or missing:
            raise ManifestError(
                f"artifact_{index}_fields_invalid:unknown={sorted(unknown)}:missing={sorted(missing)}"
            )
        artifact_id = row.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ManifestError(f"artifact_{index}_id_missing")
        if artifact_id in artifacts:
            raise ManifestError(f"duplicate_artifact_id:{artifact_id}")
        format_name = row.get("format")
        artifact_class = row.get("artifact_class")
        semantic_role = row.get("semantic_role")
        declared_hash = row.get("sha256")
        declared_path = row.get("path")
        if not isinstance(declared_path, str) or not declared_path:
            raise ManifestError(f"artifact_path_missing:{artifact_id}")
        if format_name not in ALLOWED_FORMATS:
            raise ManifestError(f"unsupported_artifact_format:{artifact_id}")
        if artifact_class not in ALLOWED_ARTIFACT_CLASSES:
            raise ManifestError(f"unsupported_artifact_class:{artifact_id}")
        if not isinstance(semantic_role, str) or not semantic_role:
            raise ManifestError(f"artifact_semantic_role_missing:{artifact_id}")
        if not isinstance(declared_hash, str) or not _SHA256_RE.fullmatch(declared_hash):
            raise ManifestError(f"artifact_sha256_invalid:{artifact_id}")
        array_key = row.get("array_key")
        if array_key is not None and not isinstance(array_key, str):
            raise ManifestError(f"array_key_invalid:{artifact_id}")
        artifacts[artifact_id] = ArtifactSpec(
            artifact_id=artifact_id,
            path=declared_path,
            format=str(format_name),
            artifact_class=str(artifact_class),
            semantic_role=semantic_role,
            sha256=declared_hash,
            array_key=array_key,
        )

    bindings_raw = raw_payload.get("role_bindings")
    if not isinstance(bindings_raw, Mapping) or set(bindings_raw) != set(REQUIRED_ROLES):
        raise ManifestError("role_bindings_must_match_exact_required_role_set")
    bindings = {str(key): str(value) for key, value in bindings_raw.items()}
    if len(set(bindings.values())) != len(bindings):
        raise ManifestError("required_roles_must_bind_distinct_artifacts")
    for role, artifact_id in bindings.items():
        if artifact_id not in artifacts:
            raise ManifestError(f"role_binding_unknown_artifact:{role}:{artifact_id}")
        if artifacts[artifact_id].semantic_role != role:
            raise ManifestError(f"role_binding_semantic_mismatch:{role}:{artifact_id}")
        expected_class = _REQUIRED_ROLE_CLASSES[role]
        if artifacts[artifact_id].artifact_class != expected_class:
            raise ManifestError(
                f"role_binding_class_mismatch:{role}:{artifact_id}:{expected_class}"
            )

    process_rows = raw_payload.get("provenance_processes")
    if not isinstance(process_rows, list) or not process_rows:
        raise ManifestError("provenance_processes_must_be_nonempty_array")
    processes: list[ProcessSpec] = []
    process_ids: set[str] = set()
    for index, row in enumerate(process_rows):
        if not isinstance(row, Mapping):
            raise ManifestError(f"process_{index}_must_be_object")
        unknown = set(row) - _PROCESS_FIELDS
        missing = _PROCESS_FIELDS - set(row)
        if unknown or missing:
            raise ManifestError(
                f"process_{index}_fields_invalid:unknown={sorted(unknown)}:missing={sorted(missing)}"
            )
        process_id = row.get("process_id")
        inputs = row.get("data_input_artifact_ids")
        if not isinstance(process_id, str) or not process_id:
            raise ManifestError(f"process_{index}_id_missing")
        if process_id in process_ids:
            raise ManifestError(f"duplicate_process_id:{process_id}")
        if not isinstance(inputs, list) or not inputs or not all(
            isinstance(value, str) and value for value in inputs
        ):
            raise ManifestError(f"process_{index}_inputs_invalid")
        if len(inputs) != len(set(inputs)):
            raise ManifestError(f"process_{index}_duplicate_inputs")
        typed_references = {
            field: row.get(field)
            for field in (
                "output_artifact_id",
                "evaluator_artifact_id",
                "configuration_artifact_id",
                "seed_artifact_id",
            )
        }
        if not all(isinstance(value, str) and value for value in typed_references.values()):
            raise ManifestError(f"process_{index}_typed_references_invalid")
        process_ids.add(process_id)
        processes.append(
            ProcessSpec(
                process_id=process_id,
                data_input_artifact_ids=tuple(inputs),
                output_artifact_id=typed_references["output_artifact_id"],
                evaluator_artifact_id=typed_references["evaluator_artifact_id"],
                configuration_artifact_id=typed_references["configuration_artifact_id"],
                seed_artifact_id=typed_references["seed_artifact_id"],
            )
        )

    squares = raw_payload.get("refinement_squares")
    if not isinstance(squares, list):
        raise ManifestError("refinement_squares_must_be_array")
    splice_controls = raw_payload.get("splice_controls")
    realization = raw_payload.get("realization")
    if not isinstance(splice_controls, Mapping):
        raise ManifestError("splice_controls_must_be_object")
    if not isinstance(realization, Mapping):
        raise ManifestError("realization_must_be_object")
    return TypedSourceTowerManifest(
        path=path,
        base_dir=root,
        bundle_id=bundle_id,
        artifacts=artifacts,
        role_bindings=bindings,
        processes=tuple(processes),
        splice_controls=splice_controls,
        refinement_squares=tuple(squares),
        realization=realization,
        raw_sha256="sha256:" + hashlib.sha256(encoded).hexdigest(),
    )


def _verify_provenance(
    manifest: TypedSourceTowerManifest,
    store: _ArtifactStore,
) -> dict[str, Any]:
    blockers: list[str] = []
    graph = nx.DiGraph()
    graph.add_nodes_from(manifest.artifacts)
    output_process: dict[str, str] = {}
    process_rows: list[dict[str, Any]] = []
    for process in manifest.processes:
        row_blockers: list[str] = []
        references = (
            *process.data_input_artifact_ids,
            process.output_artifact_id,
            process.evaluator_artifact_id,
            process.configuration_artifact_id,
            process.seed_artifact_id,
        )
        unknown = [value for value in references if value not in manifest.artifacts]
        if unknown:
            row_blockers.append(f"unknown_artifacts:{sorted(set(unknown))}")
        if process.output_artifact_id in output_process:
            row_blockers.append("output_has_multiple_producers")
        else:
            output_process[process.output_artifact_id] = process.process_id
        if not unknown:
            expected_classes = {
                process.evaluator_artifact_id: "evaluator",
                process.configuration_artifact_id: "configuration",
                process.seed_artifact_id: "seed",
            }
            for artifact_id, expected_class in expected_classes.items():
                if manifest.artifacts[artifact_id].artifact_class != expected_class:
                    row_blockers.append(
                        f"typed_provenance_class_mismatch:{artifact_id}:{expected_class}"
                    )
            for source in (
                *process.data_input_artifact_ids,
                process.evaluator_artifact_id,
                process.configuration_artifact_id,
                process.seed_artifact_id,
            ):
                graph.add_edge(source, process.output_artifact_id, process_id=process.process_id)
        reconstructed = False
        contract_hash = None
        if not row_blockers and all(store.valid(value) for value in references):
            try:
                evaluator = store.value(process.evaluator_artifact_id)
                config = store.value(process.configuration_artifact_id)
                seed = store.value(process.seed_artifact_id)
                if not isinstance(evaluator, Mapping):
                    raise ManifestError("evaluator_artifact_must_be_json_object")
                if set(evaluator) != {"schema", "operation", "seed_policy"}:
                    raise ManifestError("evaluator_schema_fields_not_exact")
                if evaluator.get("schema") != "oph.source-tower-evaluator.v1":
                    raise ManifestError("evaluator_schema_mismatch")
                operation = evaluator.get("operation")
                if operation not in ALLOWED_OPERATIONS:
                    raise ManifestError("evaluator_operation_not_allowlisted")
                if evaluator.get("seed_policy") != "bound_no_random_draws":
                    raise ManifestError("evaluator_seed_policy_invalid")
                if not isinstance(config, Mapping):
                    raise ManifestError("configuration_artifact_must_be_json_object")
                if not isinstance(seed, Mapping) or set(seed) != {"seed"} or type(seed["seed"]) is not int:
                    raise ManifestError("seed_artifact_must_contain_exact_integer_seed")
                inputs = {
                    artifact_id: store.value(artifact_id)
                    for artifact_id in process.data_input_artifact_ids
                }
                expected = _execute_evaluator(str(operation), inputs, config)
                reconstructed = _values_equal(
                    expected,
                    store.value(process.output_artifact_id),
                )
                if not reconstructed:
                    raise ManifestError("independent_reconstruction_mismatch")
                contract_hash = _process_contract_hash(process, store)
            except (ValueError, ManifestError) as exc:
                row_blockers.append(str(exc))
        else:
            row_blockers.append("process_references_invalid_artifact")
        blockers.extend(f"process:{process.process_id}:{value}" for value in row_blockers)
        process_rows.append(
            {
                "process_id": process.process_id,
                "output_artifact_id": process.output_artifact_id,
                "reconstructed": reconstructed,
                "process_contract_sha256": contract_hash,
                "passed": not row_blockers,
                "blockers": row_blockers,
            }
        )

    try:
        dag = nx.is_directed_acyclic_graph(graph)
    except nx.NetworkXError:
        dag = False
    if not dag:
        blockers.append("provenance_graph_is_not_acyclic")
    main = manifest.role_bindings["authoritative_presentation"]
    if manifest.artifacts[main].artifact_class != "authoritative_source":
        blockers.append("authoritative_presentation_class_mismatch")
    missing_producers = [
        artifact_id
        for artifact_id in manifest.role_bindings.values()
        if artifact_id not in output_process
    ]
    if missing_producers:
        blockers.append(f"required_role_outputs_missing_producer:{sorted(missing_producers)}")
    source_rows: dict[str, bool] = {}
    if dag:
        for role, artifact_id in manifest.role_bindings.items():
            source_rows[role] = bool(
                artifact_id == main or nx.has_path(graph, main, artifact_id)
            )
    else:
        source_rows = {role: False for role in REQUIRED_ROLES}
    all_reconstructed = all(
        output_process.get(artifact_id)
        and next(
            (
                row["reconstructed"]
                for row in process_rows
                if row["process_id"] == output_process[artifact_id]
            ),
            False,
        )
        for artifact_id in manifest.role_bindings.values()
    )
    same_source = all(source_rows.values())
    if not same_source:
        blockers.append("required_roles_do_not_share_authoritative_source_ancestry")
    if not all_reconstructed:
        blockers.append("not_all_required_readouts_independently_reconstructed")
    main_descendants = sorted(nx.descendants(graph, main)) if dag else []
    protected_ancestors: set[str] = set()
    if dag:
        for artifact_id in manifest.role_bindings.values():
            protected_ancestors.update(nx.ancestors(graph, artifact_id))
            protected_ancestors.add(artifact_id)
    reconstructed_outputs = sorted(
        row["output_artifact_id"]
        for row in process_rows
        if row["reconstructed"] is True
    )
    authoritative_roots = sorted(
        artifact_id
        for artifact_id, spec in manifest.artifacts.items()
        if spec.artifact_class == "authoritative_source"
    )
    primitive_ancestors = (
        {
            artifact_id: sorted(
                ancestor
                for ancestor in nx.ancestors(graph, artifact_id)
                if manifest.artifacts[ancestor].artifact_class == "source_primitive"
                and graph.in_degree(ancestor) == 0
            )
            for artifact_id in authoritative_roots
        }
        if dag
        else {artifact_id: [] for artifact_id in authoritative_roots}
    )
    if main not in reconstructed_outputs or not primitive_ancestors.get(main):
        blockers.append(
            "authoritative_presentation_not_reconstructed_from_root_source_primitive"
        )
    return {
        "passed": not blockers,
        "blockers": blockers,
        "graph": graph,
        "main_source_artifact_id": main,
        "role_artifact_ids": dict(sorted(manifest.role_bindings.items())),
        "role_source_ancestry": source_rows,
        "main_source_descendant_artifact_ids": main_descendants,
        "protected_source_ancestry_artifact_ids": sorted(protected_ancestors),
        "required_roles_share_one_source": same_source,
        "all_required_readouts_reconstructed": all_reconstructed,
        "reconstructed_output_artifact_ids": reconstructed_outputs,
        "authoritative_source_primitive_ancestors": primitive_ancestors,
        "dag_acyclic": dag,
        "process_rows": process_rows,
        "provenance_edge_count": graph.number_of_edges(),
        "provenance_edges": [
            {
                "from_artifact_id": source,
                "to_artifact_id": target,
                "process_id": str(attributes.get("process_id")),
            }
            for source, target, attributes in sorted(
                graph.edges(data=True),
                key=lambda row: (str(row[0]), str(row[1]), str(row[2].get("process_id"))),
            )
        ],
    }


def _verify_no_target_paths(
    manifest: TypedSourceTowerManifest,
    store: _ArtifactStore,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    graph = provenance.get("graph")
    if not isinstance(graph, nx.DiGraph) or not nx.is_directed_acyclic_graph(graph):
        return {"passed": False, "blockers": ["provenance_graph_unavailable"]}
    main = manifest.role_bindings["authoritative_presentation"]
    # Protect the complete main-source evidence cone, not only the thirteen
    # named role tips.  Otherwise a forbidden target could be smuggled into a
    # refinement or realization artifact that happens not to be a role binding.
    protected = {
        main,
        *nx.descendants(graph, main),
        *manifest.role_bindings.values(),
    }
    forbidden_nodes = {
        artifact_id
        for artifact_id, spec in manifest.artifacts.items()
        if spec.artifact_class == "forbidden_target"
        or spec.semantic_role in FORBIDDEN_TARGET_SEMANTICS
    }
    target_paths: list[dict[str, str]] = []
    for source in sorted(forbidden_nodes):
        for target in sorted(protected):
            if source == target or nx.has_path(graph, source, target):
                target_paths.append({"source": source, "target": target})
    if target_paths:
        blockers.append("forbidden_target_has_path_to_protected_source_readout")

    ancestor_ids: set[str] = set()
    for target in protected:
        ancestor_ids.update(nx.ancestors(graph, target))
        ancestor_ids.add(target)
    hidden_hits: list[dict[str, str]] = []
    for artifact_id in sorted(ancestor_ids):
        if not store.valid(artifact_id):
            continue
        value = store.value(artifact_id)
        for token in sorted(FORBIDDEN_TARGET_SEMANTICS):
            if _json_contains_token(value, token):
                hidden_hits.append({"artifact_id": artifact_id, "token": token})
    if hidden_hits:
        blockers.append("forbidden_target_token_hidden_in_source_ancestor")
    return {
        "passed": not blockers,
        "blockers": blockers,
        "main_source_artifact_id": main,
        "forbidden_target_nodes": sorted(forbidden_nodes),
        "target_to_protected_paths": target_paths,
        "hidden_target_token_hits": hidden_hits,
    }


def _verify_refinement(
    manifest: TypedSourceTowerManifest,
    store: _ArtifactStore,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    graph = provenance.get("graph")
    main = provenance.get("main_source_artifact_id")
    rows: list[dict[str, Any]] = []
    coverage: set[str] = set()
    required_fields = {
        "square_id",
        "readout_role",
        "fine_source_artifact_id",
        "coarse_source_artifact_id",
        "source_coarse_map_artifact_id",
        "fine_readout_artifact_id",
        "coarse_readout_artifact_id",
        "fine_readout_operator_artifact_id",
        "coarse_readout_operator_artifact_id",
        "readout_coarse_map_artifact_id",
        "error_envelope_artifact_id",
    }
    seen_ids: set[str] = set()
    for index, raw in enumerate(manifest.refinement_squares):
        row_blockers: list[str] = []
        if not isinstance(raw, Mapping) or set(raw) != required_fields:
            rows.append(
                {
                    "index": index,
                    "passed": False,
                    "blockers": ["refinement_square_fields_not_exact"],
                }
            )
            blockers.append(f"refinement_square_{index}_fields_not_exact")
            continue
        square_id = str(raw["square_id"])
        role = str(raw["readout_role"])
        if not square_id or square_id in seen_ids:
            row_blockers.append("square_id_missing_or_duplicate")
        seen_ids.add(square_id)
        if role not in REFINEMENT_ROLES:
            row_blockers.append("readout_role_not_required_refinement_role")
        elif raw["fine_readout_artifact_id"] != manifest.role_bindings[role]:
            row_blockers.append("fine_readout_does_not_match_role_binding")
        ids = [str(raw[field]) for field in required_fields if field.endswith("artifact_id")]
        unknown = [artifact_id for artifact_id in ids if artifact_id not in manifest.artifacts]
        if unknown:
            row_blockers.append(f"unknown_artifacts:{sorted(set(unknown))}")
        numeric_ids = [
            str(raw[field])
            for field in required_fields
            if field.endswith("artifact_id") and field != "error_envelope_artifact_id"
        ]
        if not unknown and isinstance(graph, nx.DiGraph) and isinstance(main, str):
            non_source = [
                artifact_id
                for artifact_id in numeric_ids
                if artifact_id != main and not nx.has_path(graph, main, artifact_id)
            ]
            if non_source:
                row_blockers.append(f"refinement_artifacts_not_main_source_derived:{non_source}")
        residuals: dict[str, float] = {}
        bound = None
        mode = None
        if not row_blockers and all(store.valid(value) for value in ids):
            try:
                envelope = _parse_error_envelope(
                    store.value(str(raw["error_envelope_artifact_id"])),
                    manifest,
                    store,
                )
                mode = envelope["mode"]
                bound = envelope["absolute_tolerance"]
                x_f = _vector(store.value(str(raw["fine_source_artifact_id"])))
                x_c = _vector(store.value(str(raw["coarse_source_artifact_id"])))
                coarse = _matrix(store.value(str(raw["source_coarse_map_artifact_id"])))
                y_f = _vector(store.value(str(raw["fine_readout_artifact_id"])))
                y_c = _vector(store.value(str(raw["coarse_readout_artifact_id"])))
                a_f = _matrix(store.value(str(raw["fine_readout_operator_artifact_id"])))
                a_c = _matrix(store.value(str(raw["coarse_readout_operator_artifact_id"])))
                down = _matrix(store.value(str(raw["readout_coarse_map_artifact_id"])))
                residuals = {
                    "coarse_source_state": _same_shape_residual(coarse @ x_f, x_c),
                    "fine_readout_state": _same_shape_residual(a_f @ x_f, y_f),
                    "coarse_readout_state": _same_shape_residual(a_c @ x_c, y_c),
                    "coarsened_readout_state": _same_shape_residual(down @ y_f, y_c),
                    "operator_square": _same_shape_residual(
                        down @ a_f, a_c @ coarse
                    ),
                }
                if any(value > bound for value in residuals.values()):
                    row_blockers.append("commutation_residual_exceeds_declared_envelope")
            except (ValueError, ManifestError) as exc:
                row_blockers.append(str(exc))
        else:
            row_blockers.append("refinement_artifact_hash_or_lineage_failure")
        if not row_blockers:
            coverage.add(role)
        blockers.extend(f"refinement:{square_id}:{value}" for value in row_blockers)
        rows.append(
            {
                "square_id": square_id,
                "readout_role": role,
                "envelope_mode": mode,
                "absolute_tolerance": bound,
                "residuals": residuals,
                "passed": not row_blockers,
                "blockers": row_blockers,
            }
        )
    missing = sorted(set(REFINEMENT_ROLES) - coverage)
    if missing:
        blockers.append(f"refinement_role_coverage_missing:{missing}")
    return {
        "passed": not blockers,
        "blockers": blockers,
        "required_role_coverage": list(REFINEMENT_ROLES),
        "verified_role_coverage": sorted(coverage),
        "rows": rows,
    }


def _verify_splice_controls(
    manifest: TypedSourceTowerManifest,
    store: _ArtifactStore,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    controls = manifest.splice_controls
    if set(controls) != {"cap_state", "stress_entropy"}:
        return {
            "passed": False,
            "blockers": ["splice_controls_must_be_exactly_cap_state_and_stress_entropy"],
            "rows": [],
        }
    graph = provenance.get("graph")
    main = provenance.get("main_source_artifact_id")
    rows: list[dict[str, Any]] = []
    specs = {
        "cap_state": ("cap_algebras", "state", "foreign_state_artifact_id"),
        "stress_entropy": ("stress", "entropy", "foreign_entropy_artifact_id"),
    }
    for control_name, (left_role, right_role, foreign_field) in specs.items():
        control = controls.get(control_name)
        row_blockers: list[str] = []
        if not isinstance(control, Mapping) or set(control) != {
            foreign_field,
            "foreign_source_anchor_artifact_id",
        }:
            row_blockers.append("splice_control_fields_not_exact")
            foreign_id = ""
            foreign_anchor = ""
        else:
            foreign_id = str(control[foreign_field])
            foreign_anchor = str(control["foreign_source_anchor_artifact_id"])
        ids = [foreign_id, foreign_anchor]
        if any(value not in manifest.artifacts for value in ids):
            row_blockers.append("splice_control_references_unknown_artifact")
        left_id = manifest.role_bindings[left_role]
        right_id = manifest.role_bindings[right_role]
        baseline_accepted = False
        spliced_accepted = True
        foreign_is_lookalike = False
        different_root_hash = False
        independent_source_primitives = False
        if (
            not row_blockers
            and isinstance(graph, nx.DiGraph)
            and isinstance(main, str)
            and all(store.valid(value) for value in (left_id, right_id, *ids))
        ):
            baseline_accepted = _share_anchor(graph, main, (left_id, right_id))
            spliced_accepted = any(
                _share_anchor(graph, anchor, (left_id, foreign_id))
                for anchor in (main, foreign_anchor)
            )
            foreign_is_lookalike = _values_equal(
                store.value(foreign_id), store.value(right_id)
            )
            different_root_hash = (
                store.actual_sha256(foreign_anchor) != store.actual_sha256(main)
            )
            independent_source_primitives = _source_roots_are_independent(
                provenance,
                store,
                main,
                foreign_anchor,
            )
            if manifest.artifacts[foreign_anchor].artifact_class != "authoritative_source":
                row_blockers.append("foreign_anchor_is_not_authoritative_source")
            reconstructed = set(
                provenance.get("reconstructed_output_artifact_ids") or []
            )
            primitive_ancestors = provenance.get(
                "authoritative_source_primitive_ancestors"
            )
            if (
                foreign_anchor not in reconstructed
                or not isinstance(primitive_ancestors, Mapping)
                or not primitive_ancestors.get(foreign_anchor)
            ):
                row_blockers.append(
                    "foreign_anchor_not_independently_reconstructed_from_source_primitive"
                )
            if manifest.artifacts[foreign_id].semantic_role != right_role:
                row_blockers.append("foreign_splice_semantic_role_mismatch")
            if not _share_anchor(graph, foreign_anchor, (foreign_id,)):
                row_blockers.append("foreign_artifact_not_derived_from_foreign_anchor")
            if not baseline_accepted:
                row_blockers.append("baseline_pair_does_not_share_main_source")
            if spliced_accepted:
                row_blockers.append("cross_source_splice_was_not_rejected")
            if not foreign_is_lookalike:
                row_blockers.append("foreign_negative_control_is_not_value_lookalike")
            if not different_root_hash:
                row_blockers.append("foreign_source_commitment_not_distinct")
            if not independent_source_primitives:
                row_blockers.append("foreign_root_source_primitives_not_independent")
        else:
            row_blockers.append("splice_control_artifact_or_provenance_failure")
        blockers.extend(f"splice:{control_name}:{value}" for value in row_blockers)
        rows.append(
            {
                "control": control_name,
                "baseline_same_source_gate_accepted": baseline_accepted,
                "rehashed_foreign_lookalike": foreign_is_lookalike,
                "foreign_source_commitment_distinct": different_root_hash,
                "root_source_primitives_independent": independent_source_primitives,
                "spliced_same_source_gate_accepted": spliced_accepted,
                "splice_rejected": not spliced_accepted,
                "passed": not row_blockers,
                "blockers": row_blockers,
            }
        )
    return {"passed": not blockers, "blockers": blockers, "rows": rows}


def _verify_realization(
    manifest: TypedSourceTowerManifest,
    store: _ArtifactStore,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    raw = manifest.realization
    required_fields = {
        "patch_count",
        "ports_per_patch",
        "channels",
        "gauge_relabeling_control",
        "repair_schedule_control",
        "lookalike_arrow_control",
    }
    if set(raw) != required_fields:
        return {
            "passed": False,
            "blockers": ["realization_fields_not_exact"],
        }
    patch_count = raw.get("patch_count")
    ports = raw.get("ports_per_patch")
    if type(patch_count) is not int or patch_count < 1:
        blockers.append("realization_patch_count_invalid")
    if ports != 12:
        blockers.append("realization_requires_exactly_twelve_local_ports")
    graph = provenance.get("graph")
    main = provenance.get("main_source_artifact_id")
    channels = raw.get("channels")
    channel_rows: list[dict[str, Any]] = []
    if not isinstance(channels, Mapping) or set(channels) != set(REALIZATION_CHANNELS):
        blockers.append("realization_channels_do_not_match_required_set")
        channels = {}
    for channel_name in REALIZATION_CHANNELS:
        channel = channels.get(channel_name)
        row_blockers: list[str] = []
        if not isinstance(channel, Mapping) or set(channel) != {
            "concrete_artifact_id",
            "abstract_artifact_id",
            "arrow_artifact_id",
            "arrow_kind",
            "error_envelope_artifact_id",
        }:
            row_blockers.append("channel_fields_not_exact")
            channel_rows.append(
                {"channel": channel_name, "passed": False, "blockers": row_blockers}
            )
            blockers.extend(f"realization:{channel_name}:{value}" for value in row_blockers)
            continue
        ids = [
            str(channel["concrete_artifact_id"]),
            str(channel["abstract_artifact_id"]),
            str(channel["arrow_artifact_id"]),
            str(channel["error_envelope_artifact_id"]),
        ]
        if any(value not in manifest.artifacts for value in ids):
            row_blockers.append("channel_references_unknown_artifact")
        source_ids = ids[:3]
        if not row_blockers and isinstance(graph, nx.DiGraph) and isinstance(main, str):
            if not all(_share_anchor(graph, main, (artifact_id,)) for artifact_id in source_ids):
                row_blockers.append("channel_objects_not_main_source_derived")
        residual = None
        bound = None
        if not row_blockers and all(store.valid(value) for value in ids):
            try:
                concrete = np.asarray(store.value(ids[0]))
                if concrete.ndim < 2 or concrete.shape[:2] != (patch_count, 12):
                    raise ManifestError("concrete_channel_is_not_N_by_12_echosahedral_state")
                abstract = np.asarray(store.value(ids[1]))
                mapped = _apply_typed_arrow(
                    str(channel["arrow_kind"]),
                    store.value(ids[2]),
                    concrete,
                )
                envelope = _parse_error_envelope(store.value(ids[3]), manifest, store)
                bound = envelope["absolute_tolerance"]
                residual = _same_shape_residual(mapped, abstract)
                if residual > bound:
                    row_blockers.append("realization_channel_residual_exceeds_envelope")
            except (ValueError, ManifestError) as exc:
                row_blockers.append(str(exc))
        else:
            row_blockers.append("channel_artifact_hash_or_lineage_failure")
        blockers.extend(f"realization:{channel_name}:{value}" for value in row_blockers)
        channel_rows.append(
            {
                "channel": channel_name,
                "concrete_shape": list(np.asarray(store.value(ids[0])).shape)
                if ids and ids[0] in manifest.artifacts and store.valid(ids[0])
                else None,
                "residual": residual,
                "absolute_tolerance": bound,
                "passed": not row_blockers,
                "blockers": row_blockers,
            }
        )

    gauge = _verify_gauge_control(
        raw.get("gauge_relabeling_control"),
        manifest,
        store,
        graph,
        main,
        patch_count,
    )
    schedule = _verify_schedule_control(
        raw.get("repair_schedule_control"),
        manifest,
        store,
        graph,
        main,
        patch_count,
    )
    lookalike = _verify_lookalike_arrow_control(
        raw.get("lookalike_arrow_control"),
        channels,
        manifest,
        store,
        graph,
        main,
        provenance,
    )
    blockers.extend(gauge["blockers"])
    blockers.extend(schedule["blockers"])
    blockers.extend(lookalike["blockers"])
    array_channel_contract_passed = not blockers
    physical_blockers = [
        "typed_echosahedral_federation_bundle_not_bound",
        "seam_collar_and_higher_overlap_preservation_not_replayed",
        "record_checkpoint_and_quotient_transition_preservation_not_replayed",
    ]
    return {
        "passed": False,
        "array_channel_contract_passed": array_channel_contract_passed,
        "blockers": [*blockers, *physical_blockers],
        "patch_count": patch_count,
        "ports_per_patch": ports,
        "materialized_local_port_coordinate_count": (
            patch_count * 12 if type(patch_count) is int and patch_count > 0 else 0
        ),
        "channel_rows": channel_rows,
        "gauge_relabeling_control": gauge,
        "repair_schedule_control": schedule,
        "typed_arrow_lookalike_control": lookalike,
        "claim_boundary": (
            "The verified N-by-12 arrays and typed arrows are a numerical channel "
            "realization diagnostic only. They do not instantiate or preserve the typed "
            "carrier federation required by the physical realization receipt."
        ),
    }


def _verify_gauge_control(
    raw: Any,
    manifest: TypedSourceTowerManifest,
    store: _ArtifactStore,
    graph: Any,
    main: Any,
    patch_count: Any,
) -> dict[str, Any]:
    blockers: list[str] = []
    required = {
        "concrete_artifact_id",
        "quotient_arrow_artifact_id",
        "quotient_arrow_kind",
        "reference_quotient_artifact_id",
        "generator_artifact_ids",
        "error_envelope_artifact_id",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        return {"passed": False, "blockers": ["gauge_control_fields_not_exact"]}
    generator_ids = raw.get("generator_artifact_ids")
    if not isinstance(generator_ids, list) or len(generator_ids) < 2:
        return {"passed": False, "blockers": ["at_least_two_gauge_generators_required"]}
    ids = [
        str(raw["concrete_artifact_id"]),
        str(raw["quotient_arrow_artifact_id"]),
        str(raw["reference_quotient_artifact_id"]),
        str(raw["error_envelope_artifact_id"]),
        *[str(value) for value in generator_ids],
    ]
    if any(value not in manifest.artifacts for value in ids):
        return {"passed": False, "blockers": ["gauge_control_unknown_artifact"]}
    if not isinstance(graph, nx.DiGraph) or not isinstance(main, str) or not all(
        _share_anchor(graph, main, (artifact_id,)) for artifact_id in ids if artifact_id != ids[3]
    ):
        blockers.append("gauge_control_objects_not_main_source_derived")
    residuals: list[float] = []
    nonidentity = 0
    generated_group_order = 0
    if not blockers and all(store.valid(value) for value in ids):
        try:
            from oph_fpe.core.icosahedral import icosahedral_a5_port_permutations

            concrete = np.asarray(store.value(ids[0]))
            if concrete.ndim < 2 or concrete.shape[:2] != (patch_count, 12):
                raise ManifestError("gauge_control_concrete_state_is_not_N_by_12")
            arrow = store.value(ids[1])
            reference = np.asarray(store.value(ids[2]))
            envelope = _parse_error_envelope(store.value(ids[3]), manifest, store)
            exact_a5 = set(icosahedral_a5_port_permutations())
            accepted_generators: list[tuple[int, ...]] = []
            for generator_id in ids[4:]:
                raw_permutation = np.asarray(store.value(generator_id))
                if not np.issubdtype(raw_permutation.dtype, np.integer):
                    raise ManifestError("gauge_generator_dtype_is_not_integer")
                if np.any(raw_permutation < 0) or np.any(raw_permutation >= 12):
                    raise ManifestError("gauge_generator_index_out_of_range")
                permutation = raw_permutation.astype(np.int64, copy=False)
                if permutation.shape != (12,) or sorted(permutation.tolist()) != list(range(12)):
                    raise ManifestError("gauge_generator_is_not_port_permutation")
                permutation_tuple = tuple(int(value) for value in permutation)
                if permutation_tuple not in exact_a5:
                    raise ManifestError("gauge_generator_is_not_exact_icosahedral_A5_rotation")
                accepted_generators.append(permutation_tuple)
                if not np.array_equal(permutation, np.arange(12)):
                    nonidentity += 1
                inverse = np.argsort(permutation)
                transformed = concrete[:, inverse, ...]
                quotient = _apply_typed_arrow(
                    str(raw["quotient_arrow_kind"]), arrow, transformed
                )
                residuals.append(_same_shape_residual(quotient, reference))
            if nonidentity < 2:
                raise ManifestError("two_nonidentity_gauge_generators_required")
            generated_group_order = len(_generated_permutation_group(accepted_generators))
            if generated_group_order != 60:
                raise ManifestError("declared_gauge_generators_do_not_generate_exact_A5")
            if any(value > envelope["absolute_tolerance"] for value in residuals):
                raise ManifestError("gauge_relabeling_changes_quotient_output")
        except (ValueError, ManifestError) as exc:
            blockers.append(str(exc))
    else:
        blockers.append("gauge_control_artifact_failure")
    return {
        "passed": not blockers,
        "blockers": blockers,
        "generator_count": len(generator_ids),
        "nonidentity_generator_count": nonidentity,
        "generated_group_order": generated_group_order,
        "quotient_residuals": residuals,
    }


def _verify_schedule_control(
    raw: Any,
    manifest: TypedSourceTowerManifest,
    store: _ArtifactStore,
    graph: Any,
    main: Any,
    patch_count: Any,
) -> dict[str, Any]:
    blockers: list[str] = []
    required = {
        "initial_state_artifact_id",
        "repair_linear_artifact_id",
        "repair_bias_artifact_id",
        "quotient_arrow_artifact_id",
        "quotient_arrow_kind",
        "reference_quotient_artifact_id",
        "error_envelope_artifact_id",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        return {"passed": False, "blockers": ["schedule_control_fields_not_exact"]}
    ids = [str(raw[field]) for field in required if field.endswith("artifact_id")]
    if any(value not in manifest.artifacts for value in ids):
        return {"passed": False, "blockers": ["schedule_control_unknown_artifact"]}
    envelope_id = str(raw["error_envelope_artifact_id"])
    source_ids = [value for value in ids if value != envelope_id]
    if not isinstance(graph, nx.DiGraph) or not isinstance(main, str) or not all(
        _share_anchor(graph, main, (artifact_id,)) for artifact_id in source_ids
    ):
        blockers.append("schedule_control_objects_not_main_source_derived")
    residuals: list[float] = []
    schedule_count = 0
    if not blockers and all(store.valid(value) for value in ids):
        try:
            initial = np.asarray(store.value(str(raw["initial_state_artifact_id"])), dtype=float)
            if initial.ndim < 2 or initial.shape[:2] != (patch_count, 12):
                raise ManifestError("repair_initial_state_is_not_N_by_12")
            linear = np.asarray(
                store.value(str(raw["repair_linear_artifact_id"])), dtype=float
            )
            bias = np.asarray(
                store.value(str(raw["repair_bias_artifact_id"])), dtype=float
            )
            flat = initial.reshape(-1)
            if (
                linear.ndim != 3
                or bias.ndim != 2
                or linear.shape[0] != bias.shape[0]
                or linear.shape[1:] != (flat.size, flat.size)
                or bias.shape[1] != flat.size
                or not (2 <= linear.shape[0] <= 6)
            ):
                raise ManifestError("repair_operation_shapes_or_count_invalid")
            schedule_count = math.factorial(int(linear.shape[0]))
            if schedule_count > MAX_REPAIR_SCHEDULES:
                raise ManifestError("repair_schedule_enumeration_limit_exceeded")
            arrow = store.value(str(raw["quotient_arrow_artifact_id"]))
            reference = np.asarray(store.value(str(raw["reference_quotient_artifact_id"])))
            envelope = _parse_error_envelope(store.value(envelope_id), manifest, store)
            for order in itertools.permutations(range(linear.shape[0])):
                state = flat.copy()
                for operation in order:
                    state = linear[operation] @ state + bias[operation]
                quotient = _apply_typed_arrow(
                    str(raw["quotient_arrow_kind"]),
                    arrow,
                    state.reshape(initial.shape),
                )
                residuals.append(_same_shape_residual(quotient, reference))
            if any(value > envelope["absolute_tolerance"] for value in residuals):
                raise ManifestError("repair_schedule_changes_quotient_output")
        except (ValueError, ManifestError) as exc:
            blockers.append(str(exc))
    else:
        blockers.append("schedule_control_artifact_failure")
    return {
        "passed": not blockers,
        "blockers": blockers,
        "enumerated_schedule_count": schedule_count,
        "maximum_quotient_residual": max(residuals, default=None),
    }


def _verify_lookalike_arrow_control(
    raw: Any,
    channels: Mapping[str, Any],
    manifest: TypedSourceTowerManifest,
    store: _ArtifactStore,
    graph: Any,
    main: Any,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    required = {
        "channel",
        "foreign_arrow_artifact_id",
        "foreign_source_anchor_artifact_id",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        return {"passed": False, "blockers": ["lookalike_control_fields_not_exact"]}
    channel_name = str(raw["channel"])
    channel = channels.get(channel_name)
    if not isinstance(channel, Mapping) or set(channel) != {
        "concrete_artifact_id",
        "abstract_artifact_id",
        "arrow_artifact_id",
        "arrow_kind",
        "error_envelope_artifact_id",
    }:
        return {"passed": False, "blockers": ["lookalike_control_channel_unknown"]}
    true_arrow = str(channel["arrow_artifact_id"])
    foreign_arrow = str(raw["foreign_arrow_artifact_id"])
    foreign_anchor = str(raw["foreign_source_anchor_artifact_id"])
    ids = [true_arrow, foreign_arrow, foreign_anchor]
    if any(value not in manifest.artifacts for value in ids):
        return {"passed": False, "blockers": ["lookalike_control_unknown_artifact"]}
    numerically_identical = False
    lineage_rejected = False
    if (
        isinstance(graph, nx.DiGraph)
        and isinstance(main, str)
        and all(store.valid(value) for value in ids)
    ):
        numerically_identical = _values_equal(
            store.value(true_arrow), store.value(foreign_arrow)
        )
        foreign_lineage = _share_anchor(graph, foreign_anchor, (foreign_arrow,))
        main_lineage = _share_anchor(graph, main, (foreign_arrow,))
        lineage_rejected = bool(foreign_lineage and not main_lineage)
        if manifest.artifacts[foreign_anchor].artifact_class != "authoritative_source":
            blockers.append("lookalike_foreign_anchor_not_authoritative")
        reconstructed = set(provenance.get("reconstructed_output_artifact_ids") or [])
        primitive_ancestors = provenance.get(
            "authoritative_source_primitive_ancestors"
        )
        if (
            foreign_anchor not in reconstructed
            or not isinstance(primitive_ancestors, Mapping)
            or not primitive_ancestors.get(foreign_anchor)
        ):
            blockers.append(
                "lookalike_foreign_anchor_not_independently_reconstructed_from_source_primitive"
            )
        if not numerically_identical:
            blockers.append("foreign_arrow_is_not_numerical_lookalike")
        if not lineage_rejected:
            blockers.append("foreign_lookalike_arrow_not_rejected_by_lineage")
        if store.actual_sha256(main) == store.actual_sha256(foreign_anchor):
            blockers.append("lookalike_foreign_source_commitment_not_distinct")
        if not _source_roots_are_independent(
            provenance,
            store,
            main,
            foreign_anchor,
        ):
            blockers.append("lookalike_foreign_root_source_primitives_not_independent")
    else:
        blockers.append("lookalike_control_artifact_or_provenance_failure")
    return {
        "passed": not blockers,
        "blockers": blockers,
        "channel": channel_name,
        "numerically_identical_arrow": numerically_identical,
        "replacement_rejected_by_source_lineage": lineage_rejected,
    }


def _execute_evaluator(
    operation: str,
    inputs: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Any:
    if operation == "identity":
        if set(config) != set() or len(inputs) != 1:
            raise ManifestError("identity_evaluator_requires_empty_config_and_one_input")
        return next(iter(inputs.values()))
    if operation == "npz_extract":
        if set(config) != {"input_artifact_id", "key"}:
            raise ManifestError("npz_extract_config_fields_not_exact")
        artifact_id = config["input_artifact_id"]
        key = config["key"]
        if set(inputs) != {artifact_id} or not isinstance(key, str):
            raise ManifestError("npz_extract_config_reference_invalid")
        source = inputs[artifact_id]
        if not isinstance(source, Mapping) or key not in source:
            raise ManifestError("npz_extract_key_missing")
        return source[key]
    if operation == "numpy_matmul":
        if set(config) != {"matrix_artifact_id", "value_artifact_id"}:
            raise ManifestError("numpy_matmul_config_fields_not_exact")
        matrix_id = config["matrix_artifact_id"]
        value_id = config["value_artifact_id"]
        if set(inputs) != {matrix_id, value_id}:
            raise ManifestError("numpy_matmul_inputs_do_not_match_config")
        return _matrix(inputs[matrix_id]) @ _vector(inputs[value_id])
    if operation == "json_pointer":
        if set(config) != {"input_artifact_id", "pointer"}:
            raise ManifestError("json_pointer_config_fields_not_exact")
        artifact_id = config["input_artifact_id"]
        pointer = config["pointer"]
        if set(inputs) != {artifact_id} or not isinstance(pointer, str):
            raise ManifestError("json_pointer_inputs_invalid")
        return _resolve_json_pointer(inputs[artifact_id], pointer)
    raise ManifestError("evaluator_operation_not_allowlisted")


def _apply_typed_arrow(kind: str, arrow: Any, concrete: np.ndarray) -> np.ndarray:
    values = np.asarray(concrete)
    if kind == "dense_linear":
        matrix = _matrix(arrow)
        return matrix @ values.reshape(-1)
    if kind == "port_weighted_sum":
        weights = _vector(arrow)
        if weights.shape != (12,) or values.ndim < 2 or values.shape[1] != 12:
            raise ManifestError("port_weighted_sum_shape_mismatch")
        return np.tensordot(values, weights, axes=(1, 0))
    if kind == "echosahedral_indexed_gather":
        if not isinstance(arrow, Mapping) or set(arrow) != {"node_indices", "port_indices"}:
            raise ManifestError("indexed_gather_arrow_requires_node_and_port_indices")
        raw_node = np.asarray(arrow["node_indices"])
        raw_port = np.asarray(arrow["port_indices"])
        if not np.issubdtype(raw_node.dtype, np.integer) or not np.issubdtype(
            raw_port.dtype, np.integer
        ):
            raise ManifestError("indexed_gather_indices_must_have_integer_dtype")
        if (
            np.any(raw_node < 0)
            or np.any(raw_node >= values.shape[0])
            or np.any(raw_port < 0)
            or np.any(raw_port >= 12)
        ):
            raise ManifestError("indexed_gather_index_out_of_range")
        node = raw_node.astype(np.int64, copy=False)
        port = raw_port.astype(np.int64, copy=False)
        if node.shape != port.shape or node.ndim != 1:
            raise ManifestError("indexed_gather_index_shapes_mismatch")
        return values[node, port]
    raise ManifestError("typed_arrow_kind_not_allowlisted")


def _parse_error_envelope(
    value: Any,
    manifest: TypedSourceTowerManifest,
    store: _ArtifactStore,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError("error_envelope_must_be_json_object")
    mode = value.get("mode")
    if mode == "exact" and set(value) == {"mode"}:
        return {"mode": "exact", "absolute_tolerance": 0.0}
    if mode == "absolute" and set(value) == {
        "mode",
        "absolute_tolerance",
        "units",
        "derivation_artifact_id",
    }:
        tolerance = value["absolute_tolerance"]
        derivation = value["derivation_artifact_id"]
        if (
            type(tolerance) not in {int, float}
            or not math.isfinite(float(tolerance))
            or float(tolerance) < 0.0
            or not isinstance(value["units"], str)
            or not value["units"]
            or not isinstance(derivation, str)
            or derivation not in manifest.artifacts
            or not store.valid(derivation)
            or manifest.artifacts[derivation].artifact_class != "configuration"
        ):
            raise ManifestError("absolute_error_envelope_invalid")
        return {"mode": "absolute", "absolute_tolerance": float(tolerance)}
    raise ManifestError("error_envelope_fields_or_mode_invalid")


def _decode_artifact(spec: ArtifactSpec, raw: bytes, path: Path) -> Any:
    if spec.format == "raw":
        return raw
    if spec.format == "json":
        if len(raw) > MAX_JSON_BYTES:
            raise ManifestError("json_artifact_exceeds_size_limit")
        return _strict_json_loads(raw)
    if spec.format == "npy":
        value = np.load(path, allow_pickle=False)
        if not isinstance(value, np.ndarray):
            raise ManifestError("npy_artifact_did_not_decode_array")
        return value
    if spec.format == "npz":
        with np.load(path, allow_pickle=False) as archive:
            if spec.array_key is not None:
                if spec.array_key not in archive.files:
                    raise ManifestError("npz_array_key_missing")
                return np.asarray(archive[spec.array_key])
            return {key: np.asarray(archive[key]) for key in sorted(archive.files)}
    raise ManifestError("unsupported_artifact_format")


def _strict_json_loads(raw: bytes | str) -> Any:
    def pairs_hook(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ManifestError(f"duplicate_json_key:{key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ManifestError(f"nonfinite_json_constant:{value}")

    return json.loads(raw, object_pairs_hook=pairs_hook, parse_constant=reject_constant)


def _validate_finite_value(value: Any) -> None:
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ManifestError("object_dtype_array_forbidden")
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise ManifestError("nonfinite_numeric_array_forbidden")
    elif isinstance(value, Mapping):
        for child in value.values():
            _validate_finite_value(child)
    elif isinstance(value, list):
        for child in value:
            _validate_finite_value(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ManifestError("nonfinite_json_number_forbidden")


def _value_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "all_finite": bool(
                not np.issubdtype(value.dtype, np.number) or np.all(np.isfinite(value))
            ),
        }
    if isinstance(value, Mapping):
        return {"type": "mapping", "keys": sorted(str(key) for key in value)}
    if isinstance(value, bytes):
        return {"type": "bytes", "byte_length": len(value)}
    return {"type": type(value).__name__}


def _decoded_value_sha256(value: Any) -> str:
    """Hash a decoded value with explicit type, dtype, shape, and key framing."""

    digest = hashlib.sha256()

    def update(candidate: Any) -> None:
        if isinstance(candidate, np.ndarray):
            array = np.ascontiguousarray(candidate)
            digest.update(b"ndarray\0")
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(b"\0")
            digest.update(_canonical_json_bytes(list(array.shape)))
            digest.update(b"\0")
            digest.update(array.tobytes(order="C"))
            return
        if isinstance(candidate, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(candidate, key=lambda item: str(item)):
                encoded_key = str(key).encode("utf-8")
                digest.update(len(encoded_key).to_bytes(8, "big"))
                digest.update(encoded_key)
                update(candidate[key])
            return
        if isinstance(candidate, list):
            digest.update(b"list\0")
            digest.update(len(candidate).to_bytes(8, "big"))
            for child in candidate:
                update(child)
            return
        if isinstance(candidate, bytes):
            digest.update(b"bytes\0")
            digest.update(len(candidate).to_bytes(8, "big"))
            digest.update(candidate)
            return
        digest.update(b"json-scalar\0")
        digest.update(_canonical_json_bytes(candidate))

    update(value)
    return "sha256:" + digest.hexdigest()


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            a = np.asarray(left)
            b = np.asarray(right)
        except (TypeError, ValueError):
            return False
        return bool(a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _values_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def _vector(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ManifestError("expected_finite_numeric_vector")
    return array


def _matrix(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ManifestError("expected_finite_numeric_matrix")
    return array


def _max_abs(value: Any) -> float:
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise ManifestError("nonfinite_residual")
    return float(np.max(np.abs(array))) if array.size else 0.0


def _same_shape_residual(left: Any, right: Any) -> float:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        raise ManifestError(
            f"residual_shape_mismatch:{left_array.shape}:{right_array.shape}"
        )
    return _max_abs(left_array - right_array)


def _share_anchor(graph: nx.DiGraph, anchor: str, artifacts: Sequence[str]) -> bool:
    return all(
        artifact == anchor or nx.has_path(graph, anchor, artifact)
        for artifact in artifacts
    )


def _source_roots_are_independent(
    provenance: Mapping[str, Any],
    store: _ArtifactStore,
    main_anchor: str,
    foreign_anchor: str,
) -> bool:
    ancestors = provenance.get("authoritative_source_primitive_ancestors")
    if not isinstance(ancestors, Mapping):
        return False
    main_roots = set(ancestors.get(main_anchor) or [])
    foreign_roots = set(ancestors.get(foreign_anchor) or [])
    if not main_roots or not foreign_roots or main_roots & foreign_roots:
        return False
    main_hashes = {store.actual_sha256(artifact_id) for artifact_id in main_roots}
    foreign_hashes = {
        store.actual_sha256(artifact_id) for artifact_id in foreign_roots
    }
    return bool(None not in main_hashes | foreign_hashes and main_hashes.isdisjoint(foreign_hashes))


def _generated_permutation_group(
    generators: Sequence[tuple[int, ...]],
) -> set[tuple[int, ...]]:
    """Return the finite closure of twelve-port permutations."""

    identity = tuple(range(12))
    closure = {identity}
    frontier = [identity]
    while frontier:
        left = frontier.pop()
        for right in generators:
            composed = tuple(left[right[index]] for index in range(12))
            if composed not in closure:
                closure.add(composed)
                frontier.append(composed)
    return closure


def _resolve_json_pointer(value: Any, pointer: str) -> Any:
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ManifestError("json_pointer_must_start_with_slash")
    current = value
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            raise ManifestError("json_pointer_not_found")
    return current


def _json_contains_token(value: Any, token: str) -> bool:
    normalized = token.lower()
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            return any(normalized in str(item).lower() for item in value.reshape(-1))
        return False
    if isinstance(value, Mapping):
        return any(
            normalized in str(key).lower() or _json_contains_token(child, token)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_json_contains_token(child, token) for child in value)
    if isinstance(value, str):
        return normalized in value.lower()
    if isinstance(value, bytes):
        return normalized.encode("utf-8") in value.lower()
    return False


def _process_contract_hash(process: ProcessSpec, store: _ArtifactStore) -> str:
    payload = {
        "process_id": process.process_id,
        "inputs": {
            artifact_id: store.actual_sha256(artifact_id)
            for artifact_id in process.data_input_artifact_ids
        },
        "output": store.actual_sha256(process.output_artifact_id),
        "evaluator": store.actual_sha256(process.evaluator_artifact_id),
        "configuration": store.actual_sha256(process.configuration_artifact_id),
        "seed": store.actual_sha256(process.seed_artifact_id),
    }
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _bundle_commitment(
    manifest: TypedSourceTowerManifest,
    store: _ArtifactStore,
) -> str:
    payload = {
        "schema": MANIFEST_SCHEMA,
        "bundle_id": manifest.bundle_id,
        "manifest_sha256": manifest.raw_sha256,
        "artifacts": {
            artifact_id: store.actual_sha256(artifact_id)
            for artifact_id in sorted(manifest.artifacts)
        },
        "role_bindings": manifest.role_bindings,
    }
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _module_sha256() -> str:
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _canonical_report_hash(report: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in report.items()
        if key not in {"verification_report_sha256", "provenance"}
    }
    # Provenance contains a live NetworkX graph; include its serializable ledger.
    provenance = report.get("provenance")
    if isinstance(provenance, Mapping):
        payload["provenance"] = {
            key: value for key, value in provenance.items() if key != "graph"
        }
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _incomplete_report(*, blocker: str, manifest_path: str | None) -> dict[str, Any]:
    report = {
        "schema": REPORT_SCHEMA,
        "artifact_type": REPORT_ARTIFACT_TYPE,
        "issue": 572,
        "manifest_path": manifest_path,
        COMMON_DOMAIN_SOURCE_TOWER_RECEIPT: False,
        ARRAY_CHANNEL_REALIZATION_DIAGNOSTIC_RECEIPT: False,
        DECLARED_TARGET_PATH_FIREWALL_DIAGNOSTIC_RECEIPT: False,
        ECHOSAHEDRAL_TO_ABSTRACT_PATCH_NET_REALIZATION_RECEIPT: False,
        SOURCE_TOWER_PROVENANCE_GRAPH_RECEIPT: False,
        SOURCE_TOWER_NO_TARGET_PATH_RECEIPT: False,
        SOURCE_TOWER_REFINEMENT_COMMUTATION_RECEIPT: False,
        SOURCE_TOWER_CROSS_SOURCE_SPLICE_REJECTION_RECEIPT: False,
        "receipt": False,
        "blockers": [blocker],
        "claim_boundary": (
            "No common-domain source-tower receipt is available without a complete, "
            "strictly parsed and independently reconstructed issue-572 manifest."
        ),
    }
    report["verification_report_sha256"] = _canonical_report_hash(report)
    return report
