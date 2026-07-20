"""Primitive-artifact verifier for finite operational self-reading observers.

This module verifies a deliberately small claim.  Given one on-disk,
content-addressed manifest, it replays a connected observer support, committed
record writes and their causal readbacks, a frozen prediction control, a local
feedback ablation, and checkpoint continuation.  Manifest booleans are neither
accepted nor consulted.

The receipt is an implementation/finite-model receipt.  It does not establish
physical geometry, a physical clock, gravity, or Standard-Model emergence.
Parent source and transaction artifacts are hash-bound antecedents; this
verifier does not recursively prove the claims made by those parents.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from oph_fpe.core.echosahedral_federation import (
    verify_reference_federation_instrument_bundle,
)
from oph_fpe.ontology import audit_source_packet, canonical_hash
from oph_fpe.ontology.firewall import classify_forbidden_field
from oph_fpe.repair.transaction import verify_repair_replay_envelope


MANIFEST_SCHEMA = "oph.operational-observer.manifest.v3"
REPORT_SCHEMA = "oph.operational-observer.verification.v3"
REPORT_ARTIFACT_TYPE = "oph_operational_observer_verification"
TRANSACTION_PARENT_ENVELOPE_SCHEMA = (
    "oph.operational-observer.transaction-parent-envelope.v1"
)

A3_RECORD_COMMIT_REPLAY_RECEIPT = "A3_RECORD_COMMIT_REPLAY_RECEIPT"
A3_READ_AFTER_WRITE_ANCESTRY_RECEIPT = (
    "A3_READ_AFTER_WRITE_ANCESTRY_RECEIPT"
)
A3_READBACK_PREDICTION_CONTROL_RECEIPT = (
    "A3_READBACK_PREDICTION_CONTROL_RECEIPT"
)
A4_CONNECTED_OBSERVER_SUPPORT_RECEIPT = (
    "A4_CONNECTED_OBSERVER_SUPPORT_RECEIPT"
)
A4_BOUNDED_INTERFACE_RECEIPT = "A4_BOUNDED_INTERFACE_RECEIPT"
A4_FEEDBACK_ABLATION_RECEIPT = "A4_FEEDBACK_ABLATION_RECEIPT"
A4_CHECKPOINT_CONTINUATION_REPLAY_RECEIPT = (
    "A4_CHECKPOINT_CONTINUATION_REPLAY_RECEIPT"
)
OBSERVER_ARTIFACT_INTEGRITY_RECEIPT = "OBSERVER_ARTIFACT_INTEGRITY_RECEIPT"
OBSERVER_CONTRACT_BINDING_RECEIPT = "OBSERVER_CONTRACT_BINDING_RECEIPT"
OBSERVER_SOURCE_FIREWALL_RECEIPT = "OBSERVER_SOURCE_FIREWALL_RECEIPT"
OBSERVER_RECORD_COMMIT_PROVENANCE_RECEIPT = (
    "OBSERVER_RECORD_COMMIT_PROVENANCE_RECEIPT"
)
OBSERVER_OUTCOME_PRODUCER_PROVENANCE_RECEIPT = (
    "OBSERVER_OUTCOME_PRODUCER_PROVENANCE_RECEIPT"
)
OPERATIONAL_SELF_READING_OBSERVER_RECEIPT = (
    "OPERATIONAL_SELF_READING_OBSERVER_RECEIPT"
)

FINITE_RECEIPT_KEYS = frozenset(
    {
        A3_RECORD_COMMIT_REPLAY_RECEIPT,
        A3_READ_AFTER_WRITE_ANCESTRY_RECEIPT,
        A3_READBACK_PREDICTION_CONTROL_RECEIPT,
        A4_CONNECTED_OBSERVER_SUPPORT_RECEIPT,
        A4_BOUNDED_INTERFACE_RECEIPT,
        A4_FEEDBACK_ABLATION_RECEIPT,
        A4_CHECKPOINT_CONTINUATION_REPLAY_RECEIPT,
        OBSERVER_ARTIFACT_INTEGRITY_RECEIPT,
        OBSERVER_CONTRACT_BINDING_RECEIPT,
        OBSERVER_SOURCE_FIREWALL_RECEIPT,
        OBSERVER_RECORD_COMMIT_PROVENANCE_RECEIPT,
        OBSERVER_OUTCOME_PRODUCER_PROVENANCE_RECEIPT,
        OPERATIONAL_SELF_READING_OBSERVER_RECEIPT,
    }
)

PHYSICAL_NONCLAIM_KEYS = (
    "PHYSICAL_PREDICTIVE_INDEPENDENCE_RECEIPT",
    "PHYSICAL_GEOMETRY_RECEIPT",
    "INDEPENDENT_PHYSICAL_CLOCK_RECEIPT",
    "GRAVITY_EMERGENCE_RECEIPT",
    "STANDARD_MODEL_EMERGENCE_RECEIPT",
)

REQUIRED_ROLES = (
    "source_bundle_receipt",
    "federation_bundle_receipt",
    "canonical_repair_artifact",
    "transaction_parent_envelope",
    "evaluator",
    "configuration",
    "seed",
    "federation_support",
    "semantic_trace",
    "checkpoint",
    "source_features",
    "source_features_binding",
    "record_commit_provenance",
    "outcome_provenance",
    "frozen_control",
)

_ALLOWED_FORMATS = frozenset({"json", "npy", "npz"})
_EXPECTED_FORMATS = {
    "source_bundle_receipt": "json",
    "federation_bundle_receipt": "json",
    "canonical_repair_artifact": "json",
    "transaction_parent_envelope": "json",
    "evaluator": "json",
    "configuration": "json",
    "seed": "json",
    "federation_support": "json",
    "semantic_trace": "json",
    "checkpoint": "json",
    "source_features": "npy",
    "source_features_binding": "json",
    "record_commit_provenance": "json",
    "outcome_provenance": "json",
    "frozen_control": "npz",
}
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_ARRAY_ELEMENTS = 2_000_000
_MAX_EVENTS = 100_000
_PORT_COUNT = 12

_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "bundle_id",
        "source_bundle_receipt_hash",
        "federation_bundle_receipt_hash",
        "canonical_repair_artifact_hash",
        "transaction_parent_envelope_hash",
        "evaluator_artifact_id",
        "configuration_artifact_id",
        "seed_artifact_id",
        "contract_binding_sha256",
        "artifacts",
        "role_bindings",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "artifact_id",
        "path",
        "format",
        "role",
        "sha256",
        "contract_binding_sha256",
    }
)


class ObserverManifestError(ValueError):
    """A manifest or primitive artifact violates the finite contract."""


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    artifact_id: str
    path: str
    format: str
    role: str
    sha256: str
    contract_binding_sha256: str


@dataclass(frozen=True, slots=True)
class OperationalObserverManifest:
    path: Path
    base_dir: Path
    raw_sha256: str
    bundle_id: str
    source_bundle_receipt_hash: str
    federation_bundle_receipt_hash: str
    canonical_repair_artifact_hash: str
    transaction_parent_envelope_hash: str
    evaluator_artifact_id: str
    configuration_artifact_id: str
    seed_artifact_id: str
    contract_binding_sha256: str
    artifacts: Mapping[str, ArtifactSpec]
    role_bindings: Mapping[str, str]


class _ArtifactStore:
    def __init__(self, manifest: OperationalObserverManifest):
        self.manifest = manifest
        self.rows: dict[str, dict[str, Any]] = {}
        self.values: dict[str, Any] = {}
        self.raw_hashes: dict[str, str] = {}
        self.blockers: list[str] = []

    def verify_all(self) -> None:
        for artifact_id in sorted(self.manifest.artifacts):
            self._verify_one(artifact_id)

    def value_for_role(self, role: str) -> Any:
        artifact_id = self.manifest.role_bindings[role]
        if artifact_id not in self.values:
            raise ObserverManifestError(f"artifact_unavailable:{artifact_id}")
        return self.values[artifact_id]

    def hash_for_role(self, role: str) -> str:
        artifact_id = self.manifest.role_bindings[role]
        if artifact_id not in self.raw_hashes:
            raise ObserverManifestError(f"artifact_hash_unavailable:{artifact_id}")
        return self.raw_hashes[artifact_id]

    def _verify_one(self, artifact_id: str) -> None:
        spec = self.manifest.artifacts[artifact_id]
        row: dict[str, Any] = {
            "artifact_id": artifact_id,
            "role": spec.role,
            "format": spec.format,
            "declared_path": spec.path,
            "declared_sha256": spec.sha256,
            "passed": False,
            "blockers": [],
        }
        blockers: list[str] = row["blockers"]
        try:
            relative = Path(spec.path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ObserverManifestError(
                    "path_is_absolute_or_contains_parent_traversal"
                )
            unresolved = self.manifest.base_dir / relative
            if unresolved.is_symlink():
                raise ObserverManifestError("artifact_path_is_symlink")
            resolved = unresolved.resolve(strict=True)
            resolved.relative_to(self.manifest.base_dir.resolve(strict=True))
            if not resolved.is_file():
                raise ObserverManifestError("artifact_path_is_not_regular_file")
            raw = resolved.read_bytes()
            if len(raw) > _MAX_ARTIFACT_BYTES:
                raise ObserverManifestError("artifact_exceeds_size_limit")
            actual = _raw_sha256(raw)
            row["actual_sha256"] = actual
            row["byte_length"] = len(raw)
            if actual != spec.sha256:
                raise ObserverManifestError("declared_sha256_mismatch")
            value = _decode_artifact(spec, raw)
            row["decoded_value_sha256"] = _decoded_value_hash(value)
            row["value_metadata"] = _value_metadata(value)
            row["resolved_relative_path"] = resolved.relative_to(
                self.manifest.base_dir.resolve(strict=True)
            ).as_posix()
            row["passed"] = True
            self.values[artifact_id] = value
            self.raw_hashes[artifact_id] = actual
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            zipfile.BadZipFile,
            ObserverManifestError,
            TypeError,
            ValueError,
        ) as exc:
            blockers.append(str(exc))
            self.blockers.append(f"artifact:{artifact_id}:{exc}")
        self.rows[artifact_id] = row


def compute_observer_contract_binding(
    *,
    bundle_id: str,
    source_bundle_receipt_hash: str,
    federation_bundle_receipt_hash: str,
    canonical_repair_artifact_hash: str,
    transaction_parent_envelope_hash: str,
    evaluator_artifact_id: str,
    evaluator_sha256: str,
    evaluator_id: str,
    configuration_artifact_id: str,
    configuration_sha256: str,
    configuration_id: str,
    seed_artifact_id: str,
    seed_sha256: str,
    seed_id: str,
    run_seed: int,
    shuffle_seed: int,
) -> str:
    """Return the canonical parent/configuration binding for one bundle."""

    material = {
        "schema": "oph.operational-observer.contract-binding.v3",
        "bundle_id": bundle_id,
        "source_bundle_receipt_hash": source_bundle_receipt_hash,
        "federation_bundle_receipt_hash": federation_bundle_receipt_hash,
        "canonical_repair_artifact_hash": canonical_repair_artifact_hash,
        "transaction_parent_envelope_hash": transaction_parent_envelope_hash,
        "evaluator": {
            "artifact_id": evaluator_artifact_id,
            "sha256": evaluator_sha256,
            "evaluator_id": evaluator_id,
        },
        "configuration": {
            "artifact_id": configuration_artifact_id,
            "sha256": configuration_sha256,
            "configuration_id": configuration_id,
        },
        "seed": {
            "artifact_id": seed_artifact_id,
            "sha256": seed_sha256,
            "seed_id": seed_id,
            "run_seed": run_seed,
            "shuffle_seed": shuffle_seed,
        },
    }
    return canonical_hash(material, domain="oph.operational-observer.binding.v3")


def semantic_observer_event_id(
    event_without_id: Mapping[str, Any],
    *,
    observer_id: str,
    contract_binding_sha256: str,
) -> str:
    """Compute executor-independent identity for one semantic trace event."""

    if "event_id" in event_without_id:
        raise ValueError("event_without_id must not contain event_id")
    return canonical_hash(
        {
            "observer_id": observer_id,
            "contract_binding_sha256": contract_binding_sha256,
            "event": dict(event_without_id),
        },
        domain="oph.operational-observer.semantic-event.v1",
    )


def frozen_shuffle_permutation(length: int, shuffle_seed: int) -> np.ndarray:
    """Version-stable SHA-256 Fisher--Yates permutation."""

    if type(length) is not int or length < 0:
        raise ValueError("length must be a nonnegative integer")
    if type(shuffle_seed) is not int or shuffle_seed < 0:
        raise ValueError("shuffle_seed must be a nonnegative integer")
    result = list(range(length))
    for upper in range(length - 1, 0, -1):
        material = f"oph-observer-shuffle-v1:{shuffle_seed}:{upper}".encode("ascii")
        choice = int.from_bytes(hashlib.sha256(material).digest(), "big") % (upper + 1)
        result[upper], result[choice] = result[choice], result[upper]
    return np.asarray(result, dtype=np.int64)


def compute_outcome_generator_precommitment(
    *,
    source_primitive_commitment: str,
    generator_id: str,
    outcome_secret_commitment: str,
    action_modulus: int,
    sample_count: int,
) -> str:
    """Commit the outcome producer before any prediction trace is supplied."""

    _sha256_text(source_primitive_commitment, "source_primitive_commitment")
    if generator_id != "sha256_source_feature_counter_v1":
        raise ValueError("outcome generator is not allowlisted")
    _sha256_text(outcome_secret_commitment, "outcome_secret_commitment")
    _bounded_integer(
        action_modulus,
        field="action_modulus",
        minimum=2,
        maximum=2**31 - 1,
    )
    _bounded_integer(
        sample_count,
        field="sample_count",
        minimum=1,
        maximum=_MAX_ARRAY_ELEMENTS,
    )
    return canonical_hash(
        {
            "generator_id": generator_id,
            "source_primitive_commitment": source_primitive_commitment,
            "outcome_secret_commitment": outcome_secret_commitment,
            "action_modulus": action_modulus,
            "sample_count": sample_count,
        },
        domain="oph.operational-observer.outcome-generator-precommitment.v1",
    )


def frozen_source_outcomes(
    source_features: Sequence[int] | np.ndarray,
    *,
    source_primitive_commitment: str,
    outcome_secret: str,
    action_modulus: int,
) -> np.ndarray:
    """Replay the fixed outcome producer without accepting prediction inputs."""

    _sha256_text(source_primitive_commitment, "source_primitive_commitment")
    if not isinstance(outcome_secret, str) or not outcome_secret:
        raise ValueError("outcome_secret must be a non-empty reveal string")
    _bounded_integer(
        action_modulus,
        field="action_modulus",
        minimum=2,
        maximum=2**31 - 1,
    )
    features = _integer_vector(np.asarray(source_features), "source_features")
    outcomes: list[int] = []
    for index, feature in enumerate(features.tolist()):
        digest = canonical_hash(
            {
                "source_primitive_commitment": source_primitive_commitment,
                "outcome_secret": outcome_secret,
                "sample_index": index,
                "source_feature": int(feature),
            },
            domain="oph.operational-observer.outcome-sample.v1",
        ).removeprefix("sha256:")
        outcomes.append(int(digest, 16) % action_modulus)
    return np.asarray(outcomes, dtype=np.int64)


def compute_outcome_secret_commitment(outcome_secret: str) -> str:
    if not isinstance(outcome_secret, str) or not outcome_secret:
        raise ValueError("outcome_secret must be a non-empty reveal string")
    return canonical_hash(
        {"outcome_secret": outcome_secret},
        domain="oph.operational-observer.outcome-secret-commitment.v1",
    )


def compute_prediction_phase_commitment(events: Sequence[Mapping[str, Any]]) -> str:
    """Commit record/prediction events before the outcome secret is revealed."""

    rows: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("prediction phase events must be objects")
        if event.get("kind") not in {"RECORD_COMMIT", "READBACK_PREDICTION"}:
            continue
        rows.append(dict(event))
    if not rows:
        raise ValueError("prediction phase contains no record or prediction events")
    return canonical_hash(
        rows, domain="oph.operational-observer.prediction-phase-commitment.v1"
    )


def verify_operational_observer_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Strictly replay one on-disk operational-observer evidence bundle."""

    try:
        manifest = _parse_manifest(manifest_path)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ObserverManifestError,
        TypeError,
        ValueError,
    ) as exc:
        return _incomplete_report(f"manifest_parse_failed:{type(exc).__name__}:{exc}")

    store = _ArtifactStore(manifest)
    store.verify_all()
    if store.blockers:
        return _assemble_report(
            manifest=manifest,
            store=store,
            contract={"passed": False, "blockers": ["artifact_integrity_failed"]},
            firewall={"passed": False, "blockers": ["artifact_integrity_failed"]},
            support=_failed_component("artifact_integrity_failed"),
            record_provenance=_failed_component("artifact_integrity_failed"),
            trace=_failed_component("artifact_integrity_failed"),
            outcome_provenance=_failed_component("artifact_integrity_failed"),
            prediction=_failed_component("artifact_integrity_failed"),
            feedback=_failed_component("artifact_integrity_failed"),
            checkpoint=_failed_component("artifact_integrity_failed"),
        )

    try:
        contract = _verify_contract(manifest, store)
        firewall = _verify_semantic_firewall(manifest, store)
        support = _verify_federation_support(
            store.value_for_role("federation_support"),
            federation_parent=store.value_for_role("federation_bundle_receipt"),
            config=store.value_for_role("configuration"),
            expected_binding=contract.get("evidence_binding"),
        )
        record_provenance = _verify_record_commit_provenance(
            store.value_for_role("record_commit_provenance"),
            source_parent=store.value_for_role("source_bundle_receipt"),
            expected_binding=contract.get("evidence_binding"),
        )
        trace = _verify_trace(
            store.value_for_role("semantic_trace"),
            support=support,
            config=store.value_for_role("configuration"),
            source_features=store.value_for_role("source_features"),
            expected_binding=contract.get("evidence_binding"),
            canonical_repair_artifact_hash=(
                manifest.canonical_repair_artifact_hash
            ),
            record_provenance=record_provenance,
        )
        outcome_provenance = _verify_outcome_provenance(
            store.value_for_role("outcome_provenance"),
            source_parent=store.value_for_role("source_bundle_receipt"),
            source_features=store.value_for_role("source_features"),
            source_features_artifact_id=manifest.role_bindings["source_features"],
            source_features_hash=store.hash_for_role("source_features"),
            source_features_decoded_hash=store.rows[
                manifest.role_bindings["source_features"]
            ].get("decoded_value_sha256"),
            trace=trace,
            prediction_side_artifacts={
                role: store.value_for_role(role)
                for role in (
                    "evaluator",
                    "configuration",
                    "seed",
                    "record_commit_provenance",
                    "semantic_trace",
                )
            },
            seed=store.value_for_role("seed"),
            config=store.value_for_role("configuration"),
            expected_binding=contract.get("evidence_binding"),
        )
        prediction = _verify_prediction_control(
            trace=trace,
            control=store.value_for_role("frozen_control"),
            outcome_provenance=outcome_provenance,
            seed=store.value_for_role("seed"),
            config=store.value_for_role("configuration"),
            expected_binding=contract.get("evidence_binding"),
        )
        feedback = _verify_feedback_ablation(
            trace=trace,
            config=store.value_for_role("configuration"),
        )
        checkpoint = _verify_checkpoint(
            store.value_for_role("checkpoint"),
            trace=trace,
            support=support,
            config=store.value_for_role("configuration"),
            expected_binding=contract.get("evidence_binding"),
        )
    except Exception as exc:  # malformed evidence fails closed, never aborts a run
        failure = _failed_component(
            f"verification_failed:{type(exc).__name__}:{exc}"
        )
        contract = locals().get("contract", failure)
        firewall = locals().get("firewall", failure)
        support = locals().get("support", failure)
        record_provenance = locals().get("record_provenance", failure)
        trace = locals().get("trace", failure)
        outcome_provenance = locals().get("outcome_provenance", failure)
        prediction = locals().get("prediction", failure)
        feedback = locals().get("feedback", failure)
        checkpoint = locals().get("checkpoint", failure)

    return _assemble_report(
        manifest=manifest,
        store=store,
        contract=contract,
        firewall=firewall,
        support=support,
        record_provenance=record_provenance,
        trace=trace,
        outcome_provenance=outcome_provenance,
        prediction=prediction,
        feedback=feedback,
        checkpoint=checkpoint,
    )


def write_operational_observer_report(
    manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Verify and write the canonical JSON report."""

    report = verify_operational_observer_manifest(manifest_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parse_manifest(path_value: str | Path) -> OperationalObserverManifest:
    if isinstance(path_value, Mapping):
        raise ObserverManifestError("manifest_must_be_an_on_disk_path")
    candidate = Path(path_value)
    if candidate.is_symlink():
        raise ObserverManifestError("manifest_path_is_symlink")
    path = candidate.resolve(strict=True)
    if not path.is_file():
        raise ObserverManifestError("manifest_path_is_not_regular_file")
    raw = path.read_bytes()
    if len(raw) > _MAX_JSON_BYTES:
        raise ObserverManifestError("manifest_exceeds_size_limit")
    payload = _strict_json_loads(raw)
    if not isinstance(payload, Mapping):
        raise ObserverManifestError("manifest_root_must_be_object")
    _require_exact_fields(payload, _MANIFEST_FIELDS, context="manifest")
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ObserverManifestError("manifest_schema_mismatch")

    bundle_id = _nonempty_text(payload.get("bundle_id"), "bundle_id")
    source_hash = _sha256_text(
        payload.get("source_bundle_receipt_hash"),
        "source_bundle_receipt_hash",
    )
    federation_hash = _sha256_text(
        payload.get("federation_bundle_receipt_hash"),
        "federation_bundle_receipt_hash",
    )
    repair_hash = _sha256_text(
        payload.get("canonical_repair_artifact_hash"),
        "canonical_repair_artifact_hash",
    )
    transaction_envelope_hash = _sha256_text(
        payload.get("transaction_parent_envelope_hash"),
        "transaction_parent_envelope_hash",
    )
    contract_hash = _sha256_text(
        payload.get("contract_binding_sha256"), "contract_binding_sha256"
    )
    evaluator_artifact_id = _nonempty_text(
        payload.get("evaluator_artifact_id"), "evaluator_artifact_id"
    )
    configuration_artifact_id = _nonempty_text(
        payload.get("configuration_artifact_id"), "configuration_artifact_id"
    )
    seed_artifact_id = _nonempty_text(
        payload.get("seed_artifact_id"), "seed_artifact_id"
    )

    artifact_rows = payload.get("artifacts")
    if not isinstance(artifact_rows, list) or not artifact_rows:
        raise ObserverManifestError("artifacts_must_be_nonempty_array")
    artifacts: dict[str, ArtifactSpec] = {}
    paths: set[str] = set()
    roles: set[str] = set()
    for index, row in enumerate(artifact_rows):
        if not isinstance(row, Mapping):
            raise ObserverManifestError(f"artifact_{index}_must_be_object")
        _require_exact_fields(row, _ARTIFACT_FIELDS, context=f"artifact_{index}")
        artifact_id = _nonempty_text(row.get("artifact_id"), "artifact_id")
        if artifact_id in artifacts:
            raise ObserverManifestError(f"duplicate_artifact_id:{artifact_id}")
        role = _nonempty_text(row.get("role"), "role")
        if role not in REQUIRED_ROLES:
            raise ObserverManifestError(f"unknown_artifact_role:{role}")
        if role in roles:
            raise ObserverManifestError(f"duplicate_artifact_role:{role}")
        format_name = row.get("format")
        if format_name not in _ALLOWED_FORMATS:
            raise ObserverManifestError(f"unsupported_artifact_format:{artifact_id}")
        if format_name != _EXPECTED_FORMATS[role]:
            raise ObserverManifestError(
                f"artifact_format_role_mismatch:{role}:{format_name}"
            )
        relative_path = _nonempty_text(row.get("path"), "path")
        if relative_path in paths:
            raise ObserverManifestError(f"duplicate_artifact_path:{relative_path}")
        suffix = Path(relative_path).suffix.lower().removeprefix(".")
        if suffix != format_name:
            raise ObserverManifestError(
                f"artifact_extension_format_mismatch:{artifact_id}"
            )
        declared_contract = _sha256_text(
            row.get("contract_binding_sha256"),
            f"artifact_contract_binding:{artifact_id}",
        )
        if declared_contract != contract_hash:
            raise ObserverManifestError(
                f"artifact_contract_binding_mismatch:{artifact_id}"
            )
        artifacts[artifact_id] = ArtifactSpec(
            artifact_id=artifact_id,
            path=relative_path,
            format=str(format_name),
            role=role,
            sha256=_sha256_text(row.get("sha256"), f"sha256:{artifact_id}"),
            contract_binding_sha256=declared_contract,
        )
        paths.add(relative_path)
        roles.add(role)
    if roles != set(REQUIRED_ROLES):
        raise ObserverManifestError(
            "artifact_roles_must_equal_exact_required_role_set"
        )

    bindings = payload.get("role_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(REQUIRED_ROLES):
        raise ObserverManifestError(
            "role_bindings_must_equal_exact_required_role_set"
        )
    normalized_bindings: dict[str, str] = {}
    for role in REQUIRED_ROLES:
        artifact_id = _nonempty_text(bindings.get(role), f"role_binding:{role}")
        if artifact_id not in artifacts:
            raise ObserverManifestError(f"role_binding_unknown_artifact:{role}")
        if artifacts[artifact_id].role != role:
            raise ObserverManifestError(f"role_binding_role_mismatch:{role}")
        normalized_bindings[role] = artifact_id
    if len(set(normalized_bindings.values())) != len(REQUIRED_ROLES):
        raise ObserverManifestError("role_bindings_must_be_distinct")
    if normalized_bindings["evaluator"] != evaluator_artifact_id:
        raise ObserverManifestError("evaluator_artifact_id_binding_mismatch")
    if normalized_bindings["configuration"] != configuration_artifact_id:
        raise ObserverManifestError("configuration_artifact_id_binding_mismatch")
    if normalized_bindings["seed"] != seed_artifact_id:
        raise ObserverManifestError("seed_artifact_id_binding_mismatch")

    return OperationalObserverManifest(
        path=path,
        base_dir=path.parent.resolve(strict=True),
        raw_sha256=_raw_sha256(raw),
        bundle_id=bundle_id,
        source_bundle_receipt_hash=source_hash,
        federation_bundle_receipt_hash=federation_hash,
        canonical_repair_artifact_hash=repair_hash,
        transaction_parent_envelope_hash=transaction_envelope_hash,
        evaluator_artifact_id=evaluator_artifact_id,
        configuration_artifact_id=configuration_artifact_id,
        seed_artifact_id=seed_artifact_id,
        contract_binding_sha256=contract_hash,
        artifacts=artifacts,
        role_bindings=normalized_bindings,
    )


def _verify_contract(
    manifest: OperationalObserverManifest,
    store: _ArtifactStore,
) -> dict[str, Any]:
    blockers: list[str] = []
    source = store.value_for_role("source_bundle_receipt")
    federation = store.value_for_role("federation_bundle_receipt")
    repair = store.value_for_role("canonical_repair_artifact")
    transaction_envelope = store.value_for_role("transaction_parent_envelope")
    evaluator = store.value_for_role("evaluator")
    config = store.value_for_role("configuration")
    seed = store.value_for_role("seed")
    if not isinstance(source, Mapping):
        blockers.append("source_bundle_receipt_must_be_json_object")
    if not isinstance(federation, Mapping):
        blockers.append("federation_bundle_receipt_must_be_json_object")
    if not isinstance(repair, Mapping):
        blockers.append("canonical_repair_artifact_must_be_json_object")
    if not isinstance(transaction_envelope, Mapping):
        blockers.append("transaction_parent_envelope_must_be_json_object")
    if (
        store.hash_for_role("source_bundle_receipt")
        != manifest.source_bundle_receipt_hash
    ):
        blockers.append("source_bundle_parent_hash_mismatch")
    if (
        store.hash_for_role("federation_bundle_receipt")
        != manifest.federation_bundle_receipt_hash
    ):
        blockers.append("federation_bundle_parent_hash_mismatch")
    if (
        store.hash_for_role("canonical_repair_artifact")
        != manifest.canonical_repair_artifact_hash
    ):
        blockers.append("canonical_repair_parent_hash_mismatch")
    if (
        store.hash_for_role("transaction_parent_envelope")
        != manifest.transaction_parent_envelope_hash
    ):
        blockers.append("transaction_parent_envelope_hash_mismatch")

    envelope_expected_fields = {
        "schema",
        "source_bundle_receipt_hash",
        "federation_bundle_receipt_hash",
        "canonical_repair_artifact_hash",
    }
    if isinstance(transaction_envelope, Mapping):
        try:
            _require_exact_fields(
                transaction_envelope,
                envelope_expected_fields,
                context="transaction_parent_envelope",
            )
            if transaction_envelope.get("schema") != TRANSACTION_PARENT_ENVELOPE_SCHEMA:
                raise ObserverManifestError("transaction_parent_envelope_schema_mismatch")
            envelope_bindings = {
                "source_bundle_receipt_hash": manifest.source_bundle_receipt_hash,
                "federation_bundle_receipt_hash": (
                    manifest.federation_bundle_receipt_hash
                ),
                "canonical_repair_artifact_hash": (
                    manifest.canonical_repair_artifact_hash
                ),
            }
            for field, expected in envelope_bindings.items():
                actual = _sha256_text(transaction_envelope.get(field), field)
                if actual != expected:
                    blockers.append(f"transaction_parent_envelope_mismatch:{field}")
        except ObserverManifestError as exc:
            blockers.append(str(exc))

    federation_verification = (
        verify_reference_federation_instrument_bundle(federation)
        if isinstance(federation, Mapping)
        else {}
    )
    for receipt_key in (
        "INSTRUMENT_BUNDLE_SCHEMA_RECEIPT",
        "ECHOSAHEDRAL_CARRIER_CONFORMANCE",
        "FEDERATION_SEWING_RECEIPT",
    ):
        if federation_verification.get(receipt_key) is not True:
            blockers.append(f"federation_parent_verification_failed:{receipt_key}")
    federation_firewall = federation_verification.get("presentation_firewall")
    if not isinstance(federation_firewall, Mapping) or federation_firewall.get(
        "CARRIER_PRESENTATION_FIREWALL_RECEIPT"
    ) is not True:
        blockers.append("federation_parent_presentation_firewall_failed")
    repair_verification = (
        verify_repair_replay_envelope(repair)
        if isinstance(repair, Mapping)
        else {}
    )
    for receipt_key in (
        "REPAIR_REPLAY_ENVELOPE_INTEGRITY_RECEIPT",
        "REPAIR_ARTIFACT_INTEGRITY_RECEIPT",
        "COMPLETE_READ_SET_RECEIPT",
        "CONFLICT_COMPONENT_SUPPORT_RECEIPT",
        "ATOMIC_UNION_REVALIDATION_RECEIPT",
        "TRANSACTIONAL_REPAIR_RECEIPT",
    ):
        if repair_verification.get(receipt_key) is not True:
            blockers.append(f"canonical_repair_verification_failed:{receipt_key}")

    evaluator_expected = {
        "schema",
        "evaluator_id",
        "prediction_rule",
        "action_rule",
        "shuffle_rule",
        "checkpoint_rule",
    }
    config_expected = {
        "schema",
        "configuration_id",
        "action_modulus",
        "neutral_feedback",
        "max_interface_ports",
        "minimum_prediction_advantage_count",
    }
    seed_expected = {"schema", "seed_id", "run_seed", "shuffle_seed"}
    try:
        if not isinstance(evaluator, Mapping):
            raise ObserverManifestError("evaluator_must_be_json_object")
        _require_exact_fields(evaluator, evaluator_expected, context="evaluator")
        if evaluator.get("schema") != "oph.operational-observer.evaluator.v1":
            raise ObserverManifestError("evaluator_schema_mismatch")
        expected_rules = {
            "prediction_rule": "single_committed_record_value_v1",
            "action_rule": "modular_feedback_v1",
            "shuffle_rule": "sha256_fisher_yates_v1",
            "checkpoint_rule": "exact_suffix_replay_v1",
        }
        for field, expected in expected_rules.items():
            if evaluator.get(field) != expected:
                raise ObserverManifestError(f"evaluator_{field}_not_allowlisted")
        evaluator_id = _nonempty_text(evaluator.get("evaluator_id"), "evaluator_id")

        if not isinstance(config, Mapping):
            raise ObserverManifestError("configuration_must_be_json_object")
        _require_exact_fields(config, config_expected, context="configuration")
        if config.get("schema") != "oph.operational-observer.configuration.v2":
            raise ObserverManifestError("configuration_schema_mismatch")
        configuration_id = _nonempty_text(
            config.get("configuration_id"), "configuration_id"
        )
        action_modulus = _bounded_integer(
            config.get("action_modulus"),
            field="action_modulus",
            minimum=2,
            maximum=2**31 - 1,
        )
        neutral_feedback = _bounded_integer(
            config.get("neutral_feedback"),
            field="neutral_feedback",
            minimum=0,
            maximum=action_modulus - 1,
        )
        max_interface_ports = _bounded_integer(
            config.get("max_interface_ports"),
            field="max_interface_ports",
            minimum=1,
            maximum=1_000_000,
        )
        minimum_advantage = _bounded_integer(
            config.get("minimum_prediction_advantage_count"),
            field="minimum_prediction_advantage_count",
            minimum=1,
            maximum=1_000_000,
        )
        if not isinstance(seed, Mapping):
            raise ObserverManifestError("seed_must_be_json_object")
        _require_exact_fields(seed, seed_expected, context="seed")
        if seed.get("schema") != "oph.operational-observer.seed.v1":
            raise ObserverManifestError("seed_schema_mismatch")
        seed_id = _nonempty_text(seed.get("seed_id"), "seed_id")
        run_seed = _bounded_integer(
            seed.get("run_seed"), field="run_seed", minimum=0, maximum=2**63 - 1
        )
        shuffle_seed = _bounded_integer(
            seed.get("shuffle_seed"),
            field="shuffle_seed",
            minimum=0,
            maximum=2**63 - 1,
        )
        expected_binding = compute_observer_contract_binding(
            bundle_id=manifest.bundle_id,
            source_bundle_receipt_hash=manifest.source_bundle_receipt_hash,
            federation_bundle_receipt_hash=(
                manifest.federation_bundle_receipt_hash
            ),
            canonical_repair_artifact_hash=(
                manifest.canonical_repair_artifact_hash
            ),
            transaction_parent_envelope_hash=(
                manifest.transaction_parent_envelope_hash
            ),
            evaluator_artifact_id=manifest.evaluator_artifact_id,
            evaluator_sha256=store.hash_for_role("evaluator"),
            evaluator_id=evaluator_id,
            configuration_artifact_id=manifest.configuration_artifact_id,
            configuration_sha256=store.hash_for_role("configuration"),
            configuration_id=configuration_id,
            seed_artifact_id=manifest.seed_artifact_id,
            seed_sha256=store.hash_for_role("seed"),
            seed_id=seed_id,
            run_seed=run_seed,
            shuffle_seed=shuffle_seed,
        )
        if expected_binding != manifest.contract_binding_sha256:
            blockers.append("computed_contract_binding_mismatch")
        evidence_binding = {
            "source_bundle_receipt_hash": manifest.source_bundle_receipt_hash,
            "federation_bundle_receipt_hash": (
                manifest.federation_bundle_receipt_hash
            ),
            "canonical_repair_artifact_hash": (
                manifest.canonical_repair_artifact_hash
            ),
            "transaction_parent_envelope_hash": (
                manifest.transaction_parent_envelope_hash
            ),
            "evaluator_artifact_id": manifest.evaluator_artifact_id,
            "configuration_artifact_id": manifest.configuration_artifact_id,
            "seed_artifact_id": manifest.seed_artifact_id,
            "evaluator_id": evaluator_id,
            "configuration_id": configuration_id,
            "seed_id": seed_id,
            "contract_binding_sha256": expected_binding,
        }
        for role in REQUIRED_ROLES:
            if (
                manifest.artifacts[manifest.role_bindings[role]].contract_binding_sha256
                != expected_binding
            ):
                blockers.append(f"artifact_contract_binding_mismatch:{role}")
        for role in (
            "federation_support",
            "semantic_trace",
            "checkpoint",
            "source_features_binding",
            "record_commit_provenance",
            "outcome_provenance",
        ):
            evidence = store.value_for_role(role)
            if not isinstance(evidence, Mapping):
                raise ObserverManifestError(f"{role}_must_be_json_object")
            _require_binding(evidence.get("binding"), evidence_binding)
        frozen_control = store.value_for_role("frozen_control")
        if not isinstance(frozen_control, Mapping):
            raise ObserverManifestError("frozen_control_must_be_npz_mapping")
        _require_npz_binding(frozen_control, evidence_binding)

        feature_descriptor = store.value_for_role("source_features_binding")
        _require_exact_fields(
            feature_descriptor,
            {
                "schema",
                "binding",
                "source_features_artifact_id",
                "source_features_sha256",
                "source_features_decoded_sha256",
            },
            context="source_features_binding",
        )
        if feature_descriptor.get("schema") != (
            "oph.operational-observer.source-features-binding.v1"
        ):
            raise ObserverManifestError("source_features_binding_schema_mismatch")
        feature_artifact_id = manifest.role_bindings["source_features"]
        if feature_descriptor.get("source_features_artifact_id") != feature_artifact_id:
            raise ObserverManifestError("source_features_artifact_id_mismatch")
        if feature_descriptor.get("source_features_sha256") != store.hash_for_role(
            "source_features"
        ):
            raise ObserverManifestError("source_features_raw_hash_binding_mismatch")
        if feature_descriptor.get("source_features_decoded_sha256") != store.rows[
            feature_artifact_id
        ].get("decoded_value_sha256"):
            raise ObserverManifestError("source_features_decoded_hash_binding_mismatch")
    except ObserverManifestError as exc:
        blockers.append(str(exc))
        expected_binding = None
        evaluator_id = None
        configuration_id = None
        seed_id = None
        run_seed = None
        shuffle_seed = None
        action_modulus = None
        neutral_feedback = None
        max_interface_ports = None
        minimum_advantage = None
        evidence_binding = None

    return {
        "passed": not blockers,
        "blockers": sorted(set(blockers)),
        "source_bundle_receipt_hash": manifest.source_bundle_receipt_hash,
        "federation_bundle_receipt_hash": manifest.federation_bundle_receipt_hash,
        "canonical_repair_artifact_hash": manifest.canonical_repair_artifact_hash,
        "transaction_parent_envelope_hash": (
            manifest.transaction_parent_envelope_hash
        ),
        "expected_binding": expected_binding,
        "declared_binding": manifest.contract_binding_sha256,
        "evidence_binding": evidence_binding,
        "evaluator_id": evaluator_id,
        "configuration_id": configuration_id,
        "seed_id": seed_id,
        "run_seed": run_seed,
        "shuffle_seed": shuffle_seed,
        "action_modulus": action_modulus,
        "neutral_feedback": neutral_feedback,
        "max_interface_ports": max_interface_ports,
        "minimum_prediction_advantage_count": minimum_advantage,
        "federation_parent_replayed": all(
            federation_verification.get(key) is True
            for key in (
                "INSTRUMENT_BUNDLE_SCHEMA_RECEIPT",
                "ECHOSAHEDRAL_CARRIER_CONFORMANCE",
                "FEDERATION_SEWING_RECEIPT",
            )
        )
        and isinstance(federation_firewall, Mapping)
        and federation_firewall.get("CARRIER_PRESENTATION_FIREWALL_RECEIPT") is True,
        "canonical_repair_parent_replayed": all(
            repair_verification.get(key) is True
            for key in (
                "REPAIR_REPLAY_ENVELOPE_INTEGRITY_RECEIPT",
                "REPAIR_ARTIFACT_INTEGRITY_RECEIPT",
                "COMPLETE_READ_SET_RECEIPT",
                "CONFLICT_COMPONENT_SUPPORT_RECEIPT",
                "ATOMIC_UNION_REVALIDATION_RECEIPT",
                "TRANSACTIONAL_REPAIR_RECEIPT",
            )
        ),
        "canonical_repair_commit_id": repair_verification.get("commit_id"),
        "source_parent_recursively_verified": False,
        "parent_claims_recursively_verified": False,
    }


def _verify_semantic_firewall(
    manifest: OperationalObserverManifest,
    store: _ArtifactStore,
) -> dict[str, Any]:
    blockers: list[str] = []
    finding_rows: list[dict[str, str]] = []
    for role in (
        "evaluator",
        "configuration",
        "seed",
        "transaction_parent_envelope",
        "federation_support",
        "semantic_trace",
        "checkpoint",
        "source_features_binding",
        "outcome_provenance",
    ):
        report = audit_source_packet(store.value_for_role(role))
        for finding in report.findings:
            finding_rows.append({"role": role, **finding.to_jsonable()})
            blockers.append(f"forbidden_semantic_field:{role}:{finding.path}")
    for role in ("source_features", "frozen_control"):
        value = store.value_for_role(role)
        if isinstance(value, Mapping):
            for key in value:
                category = classify_forbidden_field(str(key))
                if category is not None:
                    blockers.append(f"forbidden_array_field:{role}:{key}:{category}")
                    finding_rows.append(
                        {
                            "role": role,
                            "path": f"$.{key}",
                            "category": category,
                            "field": str(key),
                        }
                    )
    return {
        "passed": not blockers,
        "blockers": sorted(set(blockers)),
        "findings": sorted(
            finding_rows,
            key=lambda row: (
                row.get("role", ""),
                row.get("path", ""),
                row.get("category", ""),
            ),
        ),
        "semantic_identity_excludes_execution_metadata": not blockers,
        "source_features_exclude_downstream_targets": not blockers,
    }


def _verify_federation_support(
    payload: Any,
    *,
    federation_parent: Any,
    config: Any,
    expected_binding: Any,
) -> dict[str, Any]:
    blockers: list[str] = []
    expected_fields = {
        "schema",
        "binding",
        "federation_id",
        "observer_id",
        "carriers",
        "seams",
        "external_boundaries",
        "observer_support",
    }
    try:
        if not isinstance(payload, Mapping):
            raise ObserverManifestError("federation_support_must_be_json_object")
        _require_exact_fields(payload, expected_fields, context="federation_support")
        if payload.get("schema") != "oph.operational-observer.federation-support.v1":
            raise ObserverManifestError("federation_support_schema_mismatch")
        _require_binding(payload.get("binding"), expected_binding)
        federation_id = _nonempty_text(payload.get("federation_id"), "federation_id")
        observer_id = _nonempty_text(payload.get("observer_id"), "observer_id")
        carrier_values = payload.get("carriers")
        if not isinstance(carrier_values, list) or not carrier_values:
            raise ObserverManifestError("carriers_must_be_nonempty_array")
        carriers = tuple(_nonempty_text(value, "carrier_id") for value in carrier_values)
        if len(set(carriers)) != len(carriers):
            raise ObserverManifestError("carrier_ids_must_be_unique")
        carrier_set = set(carriers)
        if not isinstance(federation_parent, Mapping):
            raise ObserverManifestError("federation_parent_must_be_json_object")
        if federation_parent.get("federation_id") != federation_id:
            raise ObserverManifestError("support_federation_id_parent_mismatch")
        parent_carriers = federation_parent.get("carrier_ids")
        if (
            not isinstance(parent_carriers, list)
            or not all(isinstance(value, str) for value in parent_carriers)
            or set(parent_carriers) != carrier_set
            or len(parent_carriers) != len(carriers)
        ):
            raise ObserverManifestError("support_carrier_set_parent_mismatch")

        seam_rows = payload.get("seams")
        if not isinstance(seam_rows, list):
            raise ObserverManifestError("seams_must_be_array")
        seam_ids: set[str] = set()
        seams: list[dict[str, Any]] = []
        used_ports: dict[str, set[int]] = {carrier: set() for carrier in carriers}
        for index, row in enumerate(seam_rows):
            if not isinstance(row, Mapping):
                raise ObserverManifestError(f"seam_{index}_must_be_object")
            _require_exact_fields(
                row,
                {"seam_id", "left_carrier", "right_carrier", "left_ports", "right_ports"},
                context=f"seam_{index}",
            )
            seam_id = _nonempty_text(row.get("seam_id"), "seam_id")
            if seam_id in seam_ids:
                raise ObserverManifestError(f"duplicate_seam_id:{seam_id}")
            left = _nonempty_text(row.get("left_carrier"), "left_carrier")
            right = _nonempty_text(row.get("right_carrier"), "right_carrier")
            if left not in carrier_set or right not in carrier_set or left == right:
                raise ObserverManifestError(f"invalid_seam_endpoints:{seam_id}")
            left_ports = _port_tuple(row.get("left_ports"), f"{seam_id}:left_ports")
            right_ports = _port_tuple(row.get("right_ports"), f"{seam_id}:right_ports")
            if len(left_ports) != len(right_ports):
                raise ObserverManifestError(f"seam_port_arity_mismatch:{seam_id}")
            if used_ports[left] & set(left_ports) or used_ports[right] & set(right_ports):
                raise ObserverManifestError(f"seam_port_reused:{seam_id}")
            used_ports[left].update(left_ports)
            used_ports[right].update(right_ports)
            seam_ids.add(seam_id)
            seams.append(
                {
                    "seam_id": seam_id,
                    "left_carrier": left,
                    "right_carrier": right,
                    "left_ports": left_ports,
                    "right_ports": right_ports,
                }
            )
        parent_seam_rows = federation_parent.get("seams")
        if not isinstance(parent_seam_rows, list):
            raise ObserverManifestError("federation_parent_seams_must_be_array")
        parent_seams: dict[str, tuple[str, str, tuple[int, ...], tuple[int, ...]]] = {}
        for row in parent_seam_rows:
            if not isinstance(row, Mapping):
                raise ObserverManifestError("federation_parent_seam_must_be_object")
            seam_id = _nonempty_text(row.get("seam_id"), "parent_seam_id")
            parent_seams[seam_id] = (
                _nonempty_text(row.get("left_carrier_id"), "left_carrier_id"),
                _nonempty_text(row.get("right_carrier_id"), "right_carrier_id"),
                _port_tuple(row.get("left_ports"), f"parent:{seam_id}:left_ports"),
                _port_tuple(row.get("right_ports"), f"parent:{seam_id}:right_ports"),
            )
        declared_seams = {
            row["seam_id"]: (
                row["left_carrier"],
                row["right_carrier"],
                row["left_ports"],
                row["right_ports"],
            )
            for row in seams
        }
        if declared_seams != parent_seams:
            raise ObserverManifestError("support_seams_parent_mismatch")

        boundary_rows = payload.get("external_boundaries")
        if not isinstance(boundary_rows, list):
            raise ObserverManifestError("external_boundaries_must_be_array")
        external_ports: dict[str, set[int]] = {carrier: set() for carrier in carriers}
        seen_boundary_carriers: set[str] = set()
        for index, row in enumerate(boundary_rows):
            if not isinstance(row, Mapping):
                raise ObserverManifestError(f"boundary_{index}_must_be_object")
            _require_exact_fields(row, {"carrier_id", "ports"}, context=f"boundary_{index}")
            carrier = _nonempty_text(row.get("carrier_id"), "carrier_id")
            if carrier not in carrier_set or carrier in seen_boundary_carriers:
                raise ObserverManifestError(f"invalid_boundary_carrier:{carrier}")
            ports = set(_port_tuple(row.get("ports"), f"boundary:{carrier}" , allow_empty=True))
            if used_ports[carrier] & ports:
                raise ObserverManifestError(f"boundary_seam_port_overlap:{carrier}")
            external_ports[carrier] = ports
            seen_boundary_carriers.add(carrier)
        if seen_boundary_carriers != carrier_set:
            raise ObserverManifestError("every_carrier_requires_one_boundary_declaration")
        for carrier in carriers:
            if used_ports[carrier] | external_ports[carrier] != set(range(_PORT_COUNT)):
                raise ObserverManifestError(f"carrier_ports_not_exactly_partitioned:{carrier}")
        parent_boundary_rows = federation_parent.get("external_boundaries")
        if not isinstance(parent_boundary_rows, list):
            raise ObserverManifestError(
                "federation_parent_external_boundaries_must_be_array"
            )
        parent_external_ports: dict[str, set[int]] = {
            carrier: set() for carrier in carriers
        }
        for row in parent_boundary_rows:
            if not isinstance(row, Mapping):
                raise ObserverManifestError(
                    "federation_parent_boundary_must_be_object"
                )
            carrier = _nonempty_text(row.get("carrier_id"), "parent_boundary_carrier")
            if carrier not in carrier_set:
                raise ObserverManifestError(
                    "federation_parent_boundary_unknown_carrier"
                )
            parent_external_ports[carrier].update(
                _port_tuple(
                    row.get("ports"),
                    f"parent_boundary:{carrier}",
                    allow_empty=True,
                )
            )
        if parent_external_ports != external_ports:
            raise ObserverManifestError("support_external_boundaries_parent_mismatch")

        support = payload.get("observer_support")
        if not isinstance(support, Mapping):
            raise ObserverManifestError("observer_support_must_be_object")
        _require_exact_fields(
            support,
            {"carrier_ids", "visible_seam_ids", "interface_ports"},
            context="observer_support",
        )
        support_values = support.get("carrier_ids")
        if not isinstance(support_values, list) or not support_values:
            raise ObserverManifestError("observer_support_carriers_must_be_nonempty_array")
        support_carriers = tuple(
            _nonempty_text(value, "support_carrier") for value in support_values
        )
        if len(set(support_carriers)) != len(support_carriers):
            raise ObserverManifestError("observer_support_carriers_must_be_unique")
        support_set = set(support_carriers)
        if not support_set <= carrier_set:
            raise ObserverManifestError("observer_support_contains_unknown_carrier")
        parent_support_rows = federation_parent.get("observer_supports")
        if not isinstance(parent_support_rows, list):
            raise ObserverManifestError("federation_parent_supports_must_be_array")
        matching_parent_supports = [
            row
            for row in parent_support_rows
            if isinstance(row, Mapping) and row.get("observer_token") == observer_id
        ]
        if len(matching_parent_supports) != 1:
            raise ObserverManifestError("observer_support_parent_token_mismatch")
        parent_support = matching_parent_supports[0]
        if set(parent_support.get("carrier_ids") or []) != support_set:
            raise ObserverManifestError("observer_support_carriers_parent_mismatch")

        adjacency = {carrier: set() for carrier in support_carriers}
        crossing: set[str] = set()
        computed_interface: set[tuple[str, int]] = set()
        for seam in seams:
            left_in = seam["left_carrier"] in support_set
            right_in = seam["right_carrier"] in support_set
            if left_in and right_in:
                adjacency[seam["left_carrier"]].add(seam["right_carrier"])
                adjacency[seam["right_carrier"]].add(seam["left_carrier"])
            elif left_in:
                crossing.add(seam["seam_id"])
                computed_interface.update(
                    (seam["left_carrier"], port) for port in seam["left_ports"]
                )
            elif right_in:
                crossing.add(seam["seam_id"])
                computed_interface.update(
                    (seam["right_carrier"], port) for port in seam["right_ports"]
                )
        visited: set[str] = set()
        frontier = [support_carriers[0]]
        while frontier:
            carrier = frontier.pop()
            if carrier in visited:
                continue
            visited.add(carrier)
            frontier.extend(sorted(adjacency[carrier] - visited))
        connected = visited == support_set
        if not connected:
            blockers.append("observer_support_is_not_connected")

        visible_values = support.get("visible_seam_ids")
        if not isinstance(visible_values, list) or not all(
            isinstance(value, str) and value for value in visible_values
        ):
            raise ObserverManifestError("visible_seam_ids_must_be_string_array")
        if len(set(visible_values)) != len(visible_values) or set(visible_values) != crossing:
            blockers.append("visible_seam_ids_do_not_equal_support_cut")
        if set(parent_support.get("visible_seam_ids") or []) != set(visible_values):
            blockers.append("visible_seam_ids_parent_support_mismatch")
        for carrier in support_carriers:
            computed_interface.update(
                (carrier, port) for port in external_ports[carrier]
            )
        interface_values = support.get("interface_ports")
        if not isinstance(interface_values, list):
            raise ObserverManifestError("interface_ports_must_be_array")
        declared_interface: set[tuple[str, int]] = set()
        for index, row in enumerate(interface_values):
            if (
                not isinstance(row, list)
                or len(row) != 2
                or not isinstance(row[0], str)
                or type(row[1]) is not int
                or row[1] < 0
                or row[1] >= _PORT_COUNT
            ):
                raise ObserverManifestError(f"interface_port_{index}_invalid")
            declared_interface.add((row[0], row[1]))
        if len(declared_interface) != len(interface_values):
            blockers.append("interface_ports_are_not_unique")
        if declared_interface != computed_interface:
            blockers.append("declared_interface_does_not_equal_support_boundary")
        max_interface = _bounded_integer(
            config.get("max_interface_ports") if isinstance(config, Mapping) else None,
            field="max_interface_ports",
            minimum=1,
            maximum=1_000_000,
        )
        bounded = len(computed_interface) <= max_interface
        if not bounded:
            blockers.append("support_interface_exceeds_frozen_bound")
    except ObserverManifestError as exc:
        blockers.append(str(exc))
        federation_id = None
        observer_id = None
        carriers = ()
        support_carriers = ()
        seam_ids = set()
        computed_interface = set()
        connected = False
        bounded = False

    return {
        "passed": not blockers,
        "blockers": sorted(set(blockers)),
        "federation_id": federation_id,
        "observer_id": observer_id,
        "carrier_ids": list(carriers),
        "support_carrier_ids": list(support_carriers),
        "seam_count": len(seam_ids),
        "interface_port_count": len(computed_interface),
        "connected": connected,
        "bounded_interface": bounded,
        "federation_parent_binding_passed": not any(
            "parent_" in blocker or "_parent" in blocker for blocker in blockers
        ),
        "local_port_count": _PORT_COUNT,
    }


def _verify_record_commit_provenance(
    payload: Any,
    *,
    source_parent: Any,
    expected_binding: Any,
) -> dict[str, Any]:
    """Replay every record write from a primitive RECORD_COMMIT envelope."""

    blockers: list[str] = []
    rows: list[dict[str, Any]] = []
    by_event_id: dict[str, dict[str, Any]] = {}
    try:
        if not isinstance(payload, Mapping):
            raise ObserverManifestError(
                "record_commit_provenance_must_be_json_object"
            )
        _require_exact_fields(
            payload,
            {"schema", "binding", "entries"},
            context="record_commit_provenance",
        )
        if payload.get("schema") != (
            "oph.operational-observer.record-commit-provenance.v1"
        ):
            raise ObserverManifestError(
                "record_commit_provenance_schema_mismatch"
            )
        _require_binding(payload.get("binding"), expected_binding)
        if not isinstance(source_parent, Mapping):
            raise ObserverManifestError("source_parent_must_be_json_object")
        source_commitment = _sha256_text(
            source_parent.get("primitive_commitment"),
            "source_parent_primitive_commitment",
        )
        entries = payload.get("entries")
        if (
            not isinstance(entries, list)
            or not entries
            or len(entries) > _MAX_EVENTS
        ):
            raise ObserverManifestError(
                "record_commit_provenance_entries_must_be_bounded_nonempty_array"
            )
        for index, entry in enumerate(entries):
            context = f"record_commit_provenance_entry_{index}"
            if not isinstance(entry, Mapping):
                raise ObserverManifestError(f"{context}_must_be_object")
            _require_exact_fields(
                entry,
                {
                    "observer_event_id",
                    "record_id",
                    "record_register",
                    "value",
                    "replay_envelope",
                },
                context=context,
            )
            observer_event_id = _sha256_text(
                entry.get("observer_event_id"), f"{context}:observer_event_id"
            )
            if observer_event_id in by_event_id:
                raise ObserverManifestError(
                    f"duplicate_record_provenance_event:{observer_event_id}"
                )
            record_id = _nonempty_text(entry.get("record_id"), "record_id")
            record_register = _nonempty_text(
                entry.get("record_register"), "record_register"
            )
            value = entry.get("value")
            if isinstance(value, bool) or not isinstance(value, int):
                raise ObserverManifestError(
                    f"record_provenance_value_must_be_integer:{record_id}"
                )
            replay_envelope = entry.get("replay_envelope")
            if not isinstance(replay_envelope, Mapping):
                raise ObserverManifestError(
                    f"record_replay_envelope_must_be_object:{record_id}"
                )
            proposals = replay_envelope.get("proposals")
            if not isinstance(proposals, list) or not proposals:
                raise ObserverManifestError(
                    f"record_replay_proposals_missing:{record_id}"
                )
            for proposal in proposals:
                if not isinstance(proposal, Mapping) or proposal.get(
                    "source_parameters"
                ) != {"source_primitive_commitment": source_commitment}:
                    raise ObserverManifestError(
                        f"record_replay_source_parent_mismatch:{record_id}"
                    )
            replay = verify_repair_replay_envelope(replay_envelope)
            if replay.get("REPAIR_REPLAY_ENVELOPE_INTEGRITY_RECEIPT") is not True:
                blockers.extend(
                    f"record_replay_failed:{record_id}:{reason}"
                    for reason in replay.get("failure_reasons", [])
                )
            if replay.get("RECORD_COMMIT_REPLAY_RECEIPT") is not True:
                blockers.append(f"record_transition_not_replayed:{record_id}")
            state_changes = replay.get("state_changes")
            if (
                not isinstance(state_changes, Mapping)
                or set(state_changes) != {record_register}
            ):
                blockers.append(f"record_replay_write_set_mismatch:{record_id}")
                change = None
            else:
                change = state_changes[record_register]
            appended_exact = False
            if isinstance(change, Mapping):
                before = change.get("before")
                after = change.get("after")
                appended_exact = bool(
                    type(before) is list
                    and type(after) is list
                    and len(after) == len(before) + 1
                    and after[: len(before)] == before
                    and after[-1] == {"record_id": record_id, "value": value}
                )
            if not appended_exact:
                blockers.append(f"record_replay_appended_value_mismatch:{record_id}")
            receipt = replay.get("receipt")
            parent_event_ids = (
                list(receipt.get("parent_event_ids") or [])
                if isinstance(receipt, Mapping)
                else []
            )
            semantic_record_event_id = replay.get("semantic_record_event_id")
            artifact_hash = replay.get("artifact_hash")
            if not isinstance(semantic_record_event_id, str) or not isinstance(
                artifact_hash, str
            ):
                blockers.append(f"record_replay_identity_missing:{record_id}")
                semantic_event = None
                envelope_hash = None
            else:
                semantic_event = f"sha256:{semantic_record_event_id}"
                envelope_hash = f"sha256:{artifact_hash}"
            row = {
                "observer_event_id": observer_event_id,
                "record_id": record_id,
                "record_register": record_register,
                "value": value,
                "replay_envelope_sha256": envelope_hash,
                "semantic_record_event_id": semantic_event,
                "parent_event_ids": parent_event_ids,
                "commit_id": replay.get("commit_id"),
                "passed": bool(
                    replay.get("RECORD_COMMIT_REPLAY_RECEIPT") is True
                    and appended_exact
                    and semantic_event is not None
                    and envelope_hash is not None
                ),
            }
            rows.append(row)
            by_event_id[observer_event_id] = row
    except ObserverManifestError as exc:
        blockers.append(str(exc))
        source_commitment = None
        rows = []
        by_event_id = {}

    return {
        "passed": bool(rows) and not blockers and all(row["passed"] for row in rows),
        "blockers": sorted(set(blockers)),
        "source_primitive_commitment": source_commitment,
        "entry_count": len(rows),
        "entries": rows,
        "entry_by_observer_event_id": by_event_id,
        "primitive_record_replay_only": True,
    }


def _find_forbidden_string_values(
    value: Any,
    *,
    forbidden: set[str],
    path: str,
) -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            findings.extend(
                _find_forbidden_string_values(
                    child,
                    forbidden=forbidden,
                    path=f"{path}.{key}",
                )
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            findings.extend(
                _find_forbidden_string_values(
                    child,
                    forbidden=forbidden,
                    path=f"{path}[{index}]",
                )
            )
    elif isinstance(value, str) and any(token in value for token in forbidden):
        findings.append(path)
    return findings


def _verify_outcome_provenance(
    payload: Any,
    *,
    source_parent: Any,
    source_features: Any,
    source_features_artifact_id: str,
    source_features_hash: str,
    source_features_decoded_hash: Any,
    trace: Mapping[str, Any],
    prediction_side_artifacts: Mapping[str, Any],
    seed: Any,
    config: Any,
    expected_binding: Any,
) -> dict[str, Any]:
    """Recompute outcomes from a frozen source-only producer."""

    blockers: list[str] = []
    outcomes = np.asarray([], dtype=np.int64)
    try:
        if not isinstance(payload, Mapping):
            raise ObserverManifestError("outcome_provenance_must_be_json_object")
        expected_fields = {
            "schema",
            "binding",
            "generator_id",
            "source_parent_primitive_commitment",
            "source_generator_precommitment",
            "source_features_artifact_id",
            "source_features_sha256",
            "source_features_decoded_sha256",
            "outcome_secret",
            "outcome_secret_commitment",
            "prediction_phase_commitment",
            "action_modulus",
            "sample_count",
            "generated_outcomes_commitment",
        }
        _require_exact_fields(payload, expected_fields, context="outcome_provenance")
        if payload.get("schema") != (
            "oph.operational-observer.outcome-provenance.v1"
        ):
            raise ObserverManifestError("outcome_provenance_schema_mismatch")
        _require_binding(payload.get("binding"), expected_binding)
        if not isinstance(source_parent, Mapping):
            raise ObserverManifestError("source_parent_must_be_json_object")
        source_commitment = _sha256_text(
            source_parent.get("primitive_commitment"),
            "source_parent_primitive_commitment",
        )
        if payload.get("source_parent_primitive_commitment") != source_commitment:
            raise ObserverManifestError("outcome_source_parent_commitment_mismatch")
        if not isinstance(seed, Mapping) or not isinstance(config, Mapping):
            raise ObserverManifestError("outcome_seed_or_configuration_unavailable")
        generator_id = _nonempty_text(payload.get("generator_id"), "generator_id")
        if generator_id != "sha256_source_feature_counter_v1":
            raise ObserverManifestError("outcome_generator_not_allowlisted")
        _nonempty_text(seed.get("seed_id"), "seed_id")
        outcome_secret = _nonempty_text(
            payload.get("outcome_secret"), "outcome_secret"
        )
        secret_commitment = compute_outcome_secret_commitment(outcome_secret)
        if payload.get("outcome_secret_commitment") != secret_commitment:
            raise ObserverManifestError("outcome_secret_reveal_mismatch")
        if source_parent.get("outcome_secret_commitment") != secret_commitment:
            raise ObserverManifestError("source_parent_secret_commitment_mismatch")
        leaked_paths: list[str] = []
        for role, artifact in prediction_side_artifacts.items():
            leaked_paths.extend(
                _find_forbidden_string_values(
                    artifact,
                    forbidden={outcome_secret, secret_commitment},
                    path=f"$.{role}",
                )
            )
        if leaked_paths:
            raise ObserverManifestError(
                "outcome_secret_or_commitment_leaked_to_prediction_side:"
                + ",".join(sorted(leaked_paths))
            )
        modulus = _bounded_integer(
            config.get("action_modulus"),
            field="action_modulus",
            minimum=2,
            maximum=2**31 - 1,
        )
        if payload.get("action_modulus") != modulus:
            raise ObserverManifestError("outcome_action_modulus_mismatch")
        features = _source_stimuli(source_features, expected_binding)
        sample_count = len(features)
        if payload.get("sample_count") != sample_count:
            raise ObserverManifestError("outcome_sample_count_mismatch")
        if payload.get("source_features_artifact_id") != source_features_artifact_id:
            raise ObserverManifestError("outcome_source_features_artifact_id_mismatch")
        if payload.get("source_features_sha256") != source_features_hash:
            raise ObserverManifestError("outcome_source_features_raw_hash_mismatch")
        if payload.get("source_features_decoded_sha256") != (
            source_features_decoded_hash
        ):
            raise ObserverManifestError("outcome_source_features_decoded_hash_mismatch")
        precommitment = compute_outcome_generator_precommitment(
            source_primitive_commitment=source_commitment,
            generator_id=generator_id,
            outcome_secret_commitment=secret_commitment,
            action_modulus=modulus,
            sample_count=sample_count,
        )
        if payload.get("source_generator_precommitment") != precommitment:
            raise ObserverManifestError("outcome_provenance_precommitment_mismatch")
        if source_parent.get("outcome_generator_precommitment") != precommitment:
            raise ObserverManifestError("source_parent_outcome_precommitment_mismatch")
        outcomes = frozen_source_outcomes(
            features,
            source_primitive_commitment=source_commitment,
            outcome_secret=outcome_secret,
            action_modulus=modulus,
        )
        trace_events = trace.get("events") if isinstance(trace, Mapping) else None
        if not isinstance(trace_events, list):
            raise ObserverManifestError("verified_prediction_trace_unavailable")
        prediction_phase_commitment = compute_prediction_phase_commitment(
            trace_events
        )
        if payload.get("prediction_phase_commitment") != (
            prediction_phase_commitment
        ):
            raise ObserverManifestError("prediction_phase_commitment_mismatch")
        commitment = canonical_hash(
            outcomes.tolist(),
            domain="oph.operational-observer.generated-outcomes.v1",
        )
        if payload.get("generated_outcomes_commitment") != commitment:
            raise ObserverManifestError("generated_outcomes_commitment_mismatch")
    except (ObserverManifestError, ValueError) as exc:
        blockers.append(str(exc))
        source_commitment = None
        precommitment = None
        secret_commitment = None
        prediction_phase_commitment = None
        commitment = None
        sample_count = 0
        outcomes = np.asarray([], dtype=np.int64)

    return {
        "passed": not blockers and sample_count > 0,
        "blockers": sorted(set(blockers)),
        "source_primitive_commitment": source_commitment,
        "source_generator_precommitment": precommitment,
        "outcome_secret_commitment": secret_commitment,
        "prediction_phase_commitment": prediction_phase_commitment,
        "generated_outcomes_commitment": commitment,
        "sample_count": sample_count,
        "generated_outcomes": outcomes.tolist(),
        "generator_inputs": [
            "source_primitive_commitment",
            "source_features",
            "outcome_secret_reveal",
            "sample_index",
            "action_modulus",
        ],
        "outcome_generator_prediction_or_trace_inputs_consumed": False,
        "protocol_verifier_consumes_prediction_phase_commitment": True,
        "prediction_phase_committed_before_reveal_by_protocol": bool(
            not blockers and prediction_phase_commitment
        ),
        "prediction_side_secret_noninterference_replayed": not blockers,
        "temporal_external_timestamp_claim": False,
        "claim_tier": "SYNTHETIC_COMMIT_REVEAL_PROTOCOL_MECHANICS",
    }


def _verify_trace(
    payload: Any,
    *,
    support: Mapping[str, Any],
    config: Any,
    source_features: Any,
    expected_binding: Any,
    canonical_repair_artifact_hash: str,
    record_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    try:
        if not isinstance(payload, Mapping):
            raise ObserverManifestError("semantic_trace_must_be_json_object")
        _require_exact_fields(
            payload,
            {"schema", "binding", "observer_id", "events"},
            context="semantic_trace",
        )
        if payload.get("schema") != "oph.operational-observer.semantic-trace.v1":
            raise ObserverManifestError("semantic_trace_schema_mismatch")
        _require_binding(payload.get("binding"), expected_binding)
        observer_id = _nonempty_text(payload.get("observer_id"), "observer_id")
        if observer_id != support.get("observer_id"):
            raise ObserverManifestError("trace_observer_id_support_mismatch")
        event_values = payload.get("events")
        if (
            not isinstance(event_values, list)
            or not event_values
            or len(event_values) > _MAX_EVENTS
        ):
            raise ObserverManifestError("events_must_be_bounded_nonempty_array")
        if not isinstance(config, Mapping):
            raise ObserverManifestError("configuration_unavailable")
        modulus = _bounded_integer(
            config.get("action_modulus"),
            field="action_modulus",
            minimum=2,
            maximum=2**31 - 1,
        )
        stimuli = _source_stimuli(source_features, expected_binding)
        support_carriers = set(support.get("support_carrier_ids") or [])
        if not support_carriers:
            raise ObserverManifestError("verified_support_unavailable")
        if record_provenance.get("passed") is not True:
            raise ObserverManifestError("record_commit_provenance_unavailable")
        provenance_by_event = record_provenance.get(
            "entry_by_observer_event_id"
        )
        if not isinstance(provenance_by_event, Mapping):
            raise ObserverManifestError("record_commit_provenance_index_unavailable")

        normalized_events: list[dict[str, Any]] = []
        event_by_id: dict[str, dict[str, Any]] = {}
        ancestors: dict[str, set[str]] = {}
        records: dict[str, int] = {}
        record_origins: dict[str, str] = {}
        snapshots: list[dict[str, Any]] = []
        predictions: list[dict[str, int | str]] = []
        actions: list[dict[str, int | str]] = []
        prediction_events: dict[str, dict[str, Any]] = {}
        record_commit_count = 0
        used_record_provenance: set[str] = set()
        ancestry_passed = True

        for index, raw_event in enumerate(event_values):
            if not isinstance(raw_event, Mapping):
                raise ObserverManifestError(f"event_{index}_must_be_object")
            kind = raw_event.get("kind")
            common = {"kind", "event_id", "parents", "carrier_id"}
            if kind == "RECORD_COMMIT":
                expected_fields = common | {
                    "record_id",
                    "value",
                    "commit_status",
                    "canonical_repair_artifact_hash",
                    "record_replay_envelope_sha256",
                    "semantic_record_event_id",
                }
            elif kind == "READBACK_PREDICTION":
                expected_fields = common | {
                    "reads",
                    "feature_index",
                    "prediction",
                }
            elif kind == "LOCAL_ACTION":
                expected_fields = common | {
                    "reads",
                    "feature_index",
                    "feedback_event_id",
                    "action",
                }
            else:
                raise ObserverManifestError(f"event_{index}_kind_not_allowlisted")
            _require_exact_fields(raw_event, expected_fields, context=f"event_{index}")
            event = dict(raw_event)
            event_id = _sha256_text(event.get("event_id"), f"event_{index}_id")
            if event_id in event_by_id:
                raise ObserverManifestError(f"duplicate_event_id:{event_id}")
            parents_raw = event.get("parents")
            if not isinstance(parents_raw, list) or not all(
                isinstance(value, str) and _SHA256_RE.fullmatch(value)
                for value in parents_raw
            ):
                raise ObserverManifestError(f"event_{index}_parents_invalid")
            if len(set(parents_raw)) != len(parents_raw):
                raise ObserverManifestError(f"event_{index}_parents_duplicate")
            if parents_raw != sorted(parents_raw):
                raise ObserverManifestError(f"event_{index}_parents_not_canonical")
            if any(parent not in event_by_id for parent in parents_raw):
                raise ObserverManifestError(f"event_{index}_parent_not_prior")
            carrier_id = _nonempty_text(event.get("carrier_id"), "carrier_id")
            if carrier_id not in support_carriers:
                raise ObserverManifestError(f"event_{index}_outside_observer_support")
            without_id = {key: value for key, value in event.items() if key != "event_id"}
            expected_id = semantic_observer_event_id(
                without_id,
                observer_id=observer_id,
                contract_binding_sha256=str(
                    expected_binding["contract_binding_sha256"]
                ),
            )
            if event_id != expected_id:
                raise ObserverManifestError(f"event_{index}_semantic_id_mismatch")
            event_ancestors: set[str] = set(parents_raw)
            for parent in parents_raw:
                event_ancestors.update(ancestors[parent])
            ancestors[event_id] = event_ancestors

            if kind == "RECORD_COMMIT":
                record_id = _nonempty_text(event.get("record_id"), "record_id")
                if record_id in records:
                    raise ObserverManifestError(f"record_overwrite_not_allowed:{record_id}")
                value = _bounded_integer(
                    event.get("value"),
                    field=f"record_value:{record_id}",
                    minimum=0,
                    maximum=modulus - 1,
                )
                if event.get("commit_status") != "COMMITTED":
                    raise ObserverManifestError(f"record_not_committed:{record_id}")
                if (
                    event.get("canonical_repair_artifact_hash")
                    != canonical_repair_artifact_hash
                ):
                    raise ObserverManifestError(
                        f"record_canonical_repair_parent_mismatch:{record_id}"
                    )
                provenance = provenance_by_event.get(event_id)
                if not isinstance(provenance, Mapping):
                    raise ObserverManifestError(
                        f"record_commit_has_no_primitive_replay:{record_id}"
                    )
                if provenance.get("record_id") != record_id:
                    raise ObserverManifestError(
                        f"record_provenance_record_id_mismatch:{record_id}"
                    )
                if provenance.get("value") != value:
                    raise ObserverManifestError(
                        f"record_provenance_value_mismatch:{record_id}"
                    )
                if provenance.get("parent_event_ids") != parents_raw:
                    raise ObserverManifestError(
                        f"record_provenance_parent_mismatch:{record_id}"
                    )
                if event.get("record_replay_envelope_sha256") != provenance.get(
                    "replay_envelope_sha256"
                ):
                    raise ObserverManifestError(
                        f"record_replay_envelope_hash_mismatch:{record_id}"
                    )
                if event.get("semantic_record_event_id") != provenance.get(
                    "semantic_record_event_id"
                ):
                    raise ObserverManifestError(
                        f"semantic_record_event_id_mismatch:{record_id}"
                    )
                if provenance.get("passed") is not True:
                    raise ObserverManifestError(
                        f"record_primitive_replay_failed:{record_id}"
                    )
                used_record_provenance.add(event_id)
                records[record_id] = value
                record_origins[record_id] = event_id
                record_commit_count += 1
            else:
                read_values = event.get("reads")
                if (
                    not isinstance(read_values, list)
                    or len(read_values) != 1
                    or not isinstance(read_values[0], str)
                ):
                    raise ObserverManifestError(
                        f"event_{index}_requires_exactly_one_record_read"
                    )
                record_id = read_values[0]
                if record_id not in records:
                    raise ObserverManifestError(f"read_before_write:{record_id}")
                if record_origins[record_id] not in event_ancestors:
                    ancestry_passed = False
                    blockers.append(f"record_write_not_causal_ancestor:{record_id}")
                feature_index = _bounded_integer(
                    event.get("feature_index"),
                    field="feature_index",
                    minimum=0,
                    maximum=len(stimuli) - 1,
                )
                if kind == "READBACK_PREDICTION":
                    prediction = _bounded_integer(
                        event.get("prediction"),
                        field="prediction",
                        minimum=0,
                        maximum=modulus - 1,
                    )
                    if prediction != records[record_id]:
                        blockers.append(
                            f"prediction_not_replayed_from_record:{event_id}"
                        )
                    row = {
                        "event_id": event_id,
                        "feature_index": feature_index,
                        "prediction": prediction,
                    }
                    predictions.append(row)
                    prediction_events[event_id] = row
                else:
                    feedback_event_id = _sha256_text(
                        event.get("feedback_event_id"), "feedback_event_id"
                    )
                    if feedback_event_id not in prediction_events:
                        raise ObserverManifestError(
                            f"feedback_prediction_not_prior:{feedback_event_id}"
                        )
                    if feedback_event_id not in event_ancestors:
                        blockers.append(
                            f"feedback_prediction_not_causal_ancestor:{event_id}"
                        )
                    prediction_row = prediction_events[feedback_event_id]
                    if prediction_row["feature_index"] != feature_index:
                        blockers.append(f"feedback_feature_index_mismatch:{event_id}")
                    action = _bounded_integer(
                        event.get("action"),
                        field="action",
                        minimum=0,
                        maximum=modulus - 1,
                    )
                    expected_action = (
                        int(stimuli[feature_index])
                        + int(prediction_row["prediction"])
                    ) % modulus
                    if action != expected_action:
                        blockers.append(f"local_action_replay_mismatch:{event_id}")
                    actions.append(
                        {
                            "event_id": event_id,
                            "feature_index": feature_index,
                            "feedback_event_id": feedback_event_id,
                            "action": action,
                            "expected_action": expected_action,
                        }
                    )
            event_by_id[event_id] = event
            normalized_events.append(event)
            snapshots.append(
                {
                    "record_state": dict(sorted(records.items())),
                    "record_origins": dict(sorted(record_origins.items())),
                }
            )

        prediction_indices = [int(row["feature_index"]) for row in predictions]
        action_indices = [int(row["feature_index"]) for row in actions]
        expected_indices = list(range(len(stimuli)))
        if sorted(prediction_indices) != expected_indices or len(
            set(prediction_indices)
        ) != len(prediction_indices):
            blockers.append("predictions_do_not_cover_each_feature_exactly_once")
        if sorted(action_indices) != expected_indices or len(set(action_indices)) != len(
            action_indices
        ):
            blockers.append("actions_do_not_cover_each_feature_exactly_once")
        if used_record_provenance != set(provenance_by_event):
            blockers.append("record_provenance_trace_coverage_mismatch")
        replay_passed = not any(
            value.startswith(
                (
                    "record_",
                    "semantic_record_",
                    "prediction_not_replayed",
                    "local_action_replay",
                    "feedback_prediction_not_causal",
                    "feedback_feature_index",
                )
            )
            for value in blockers
        )
    except ObserverManifestError as exc:
        blockers.append(str(exc))
        observer_id = None
        normalized_events = []
        ancestors = {}
        snapshots = []
        predictions = []
        actions = []
        record_commit_count = 0
        ancestry_passed = False
        replay_passed = False
        records = {}
        record_origins = {}
        stimuli = np.asarray([], dtype=np.int64)

    return {
        "passed": not blockers,
        "blockers": sorted(set(blockers)),
        "observer_id": observer_id,
        "event_count": len(normalized_events),
        "record_commit_count": record_commit_count,
        "prediction_count": len(predictions),
        "local_action_count": len(actions),
        "record_commit_replay_passed": bool(record_commit_count and replay_passed),
        "read_after_write_ancestry_passed": bool(
            ancestry_passed and (predictions or actions)
        ),
        "event_ids": [event["event_id"] for event in normalized_events],
        "events": normalized_events,
        "ancestors": {key: sorted(value) for key, value in ancestors.items()},
        "snapshots": snapshots,
        "predictions": predictions,
        "actions": actions,
        "final_record_state": dict(sorted(records.items())),
        "final_record_origins": dict(sorted(record_origins.items())),
        "stimuli": stimuli.tolist(),
    }


def _verify_prediction_control(
    *,
    trace: Mapping[str, Any],
    control: Any,
    outcome_provenance: Mapping[str, Any],
    seed: Any,
    config: Any,
    expected_binding: Any,
) -> dict[str, Any]:
    blockers: list[str] = []
    try:
        if not isinstance(control, Mapping):
            raise ObserverManifestError("frozen_control_must_be_npz_mapping")
        expected_keys = {
            "schema",
            *_binding_fields(),
            "permutation",
            "shuffled_predictions",
        }
        if set(control) != expected_keys:
            raise ObserverManifestError("frozen_control_fields_not_exact")
        if _npz_scalar_text(control["schema"], "control_schema") != (
            "oph.operational-observer.frozen-control.v2"
        ):
            raise ObserverManifestError("frozen_control_schema_mismatch")
        _require_npz_binding(control, expected_binding)
        if not isinstance(seed, Mapping):
            raise ObserverManifestError("seed_unavailable")
        shuffle_seed = _bounded_integer(
            seed.get("shuffle_seed"),
            field="shuffle_seed",
            minimum=0,
            maximum=2**63 - 1,
        )
        predictions_by_index = {
            int(row["feature_index"]): int(row["prediction"])
            for row in trace.get("predictions") or []
        }
        length = len(predictions_by_index)
        if sorted(predictions_by_index) != list(range(length)):
            raise ObserverManifestError("prediction_indices_not_contiguous")
        predictions = np.asarray(
            [predictions_by_index[index] for index in range(length)],
            dtype=np.int64,
        )
        if outcome_provenance.get("passed") is not True:
            raise ObserverManifestError("outcome_provenance_unavailable")
        outcomes = _integer_vector(
            np.asarray(outcome_provenance.get("generated_outcomes")),
            "generated_outcomes",
        )
        permutation = _integer_vector(control["permutation"], "permutation")
        stored_shuffled = _integer_vector(
            control["shuffled_predictions"], "shuffled_predictions"
        )
        if not (len(outcomes) == len(permutation) == len(stored_shuffled) == length):
            raise ObserverManifestError("frozen_control_vector_length_mismatch")
        if not isinstance(config, Mapping):
            raise ObserverManifestError("configuration_unavailable")
        action_modulus = _bounded_integer(
            config.get("action_modulus"),
            field="action_modulus",
            minimum=2,
            maximum=2**31 - 1,
        )
        if np.any(outcomes < 0) or np.any(outcomes >= action_modulus):
            raise ObserverManifestError("heldout_outcomes_outside_frozen_alphabet")
        outcome_commitment = canonical_hash(
            outcomes.tolist(),
            domain="oph.operational-observer.generated-outcomes.v1",
        )
        if outcome_commitment != outcome_provenance.get(
            "generated_outcomes_commitment"
        ):
            raise ObserverManifestError(
                "generated_outcomes_provenance_commitment_mismatch"
            )
        expected_permutation = frozen_shuffle_permutation(length, shuffle_seed)
        permutation_exact = np.array_equal(permutation, expected_permutation)
        if not permutation_exact:
            blockers.append("frozen_permutation_seed_replay_mismatch")
        expected_shuffled = predictions[expected_permutation]
        shuffled_exact = np.array_equal(stored_shuffled, expected_shuffled)
        if not shuffled_exact:
            blockers.append("stored_shuffled_predictions_mismatch")
        direct_matches = int(np.sum(predictions == outcomes))
        shuffled_matches = int(np.sum(expected_shuffled == outcomes))
        advantage = direct_matches - shuffled_matches
        minimum_advantage = _bounded_integer(
            config.get("minimum_prediction_advantage_count"),
            field="minimum_prediction_advantage_count",
            minimum=1,
            maximum=1_000_000,
        )
        if advantage < minimum_advantage:
            blockers.append("readback_prediction_does_not_beat_frozen_shuffle")
    except ObserverManifestError as exc:
        blockers.append(str(exc))
        direct_matches = 0
        shuffled_matches = 0
        advantage = 0
        minimum_advantage = None
        permutation_exact = False
        shuffled_exact = False
        length = 0

    return {
        "passed": not blockers,
        "blockers": sorted(set(blockers)),
        "sample_count": length,
        "direct_match_count": direct_matches,
        "shuffled_match_count": shuffled_matches,
        "prediction_advantage_count": advantage,
        "minimum_prediction_advantage_count": minimum_advantage,
        "frozen_permutation_replayed": permutation_exact,
        "shuffled_predictions_replayed": shuffled_exact,
        "outcomes_replayed_from_source_only_producer": bool(
            outcome_provenance.get("passed") is True
        ),
        "statistical_population_claim": False,
    }


def _verify_feedback_ablation(
    *,
    trace: Mapping[str, Any],
    config: Any,
) -> dict[str, Any]:
    blockers: list[str] = []
    try:
        if not isinstance(config, Mapping):
            raise ObserverManifestError("configuration_unavailable")
        modulus = _bounded_integer(
            config.get("action_modulus"),
            field="action_modulus",
            minimum=2,
            maximum=2**31 - 1,
        )
        neutral = _bounded_integer(
            config.get("neutral_feedback"),
            field="neutral_feedback",
            minimum=0,
            maximum=modulus - 1,
        )
        stimuli = [int(value) for value in trace.get("stimuli") or []]
        actions = trace.get("actions") or []
        if not actions:
            raise ObserverManifestError("no_local_actions_for_ablation")
        changed_rows: list[dict[str, Any]] = []
        for row in actions:
            feature_index = int(row["feature_index"])
            actual = int(row["action"])
            ablated = (stimuli[feature_index] + neutral) % modulus
            if actual != ablated:
                changed_rows.append(
                    {
                        "event_id": row["event_id"],
                        "actual_action": actual,
                        "ablated_action": ablated,
                    }
                )
        if not changed_rows:
            blockers.append("feedback_ablation_changes_no_later_local_action")
    except (ObserverManifestError, KeyError, TypeError, ValueError) as exc:
        blockers.append(str(exc))
        changed_rows = []
        actions = []
    return {
        "passed": not blockers,
        "blockers": sorted(set(blockers)),
        "local_action_count": len(actions),
        "changed_action_count": len(changed_rows),
        "changed_actions": changed_rows,
        "ablation_is_local_and_frozen": True,
    }


def _verify_checkpoint(
    payload: Any,
    *,
    trace: Mapping[str, Any],
    support: Mapping[str, Any],
    config: Any,
    expected_binding: Any,
) -> dict[str, Any]:
    blockers: list[str] = []
    expected_fields = {
        "schema",
        "binding",
        "observer_id",
        "continuation_observer_id",
        "cut_event_id",
        "next_event_index",
        "record_state",
        "record_origins",
        "semantic_history_root",
        "committed_suffix_event_ids",
        "continuation_state_hash",
        "checkpoint_hash",
    }
    try:
        if not isinstance(payload, Mapping):
            raise ObserverManifestError("checkpoint_must_be_json_object")
        _require_exact_fields(payload, expected_fields, context="checkpoint")
        if payload.get("schema") != "oph.operational-observer.checkpoint.v1":
            raise ObserverManifestError("checkpoint_schema_mismatch")
        _require_binding(payload.get("binding"), expected_binding)
        observer_id = _nonempty_text(payload.get("observer_id"), "observer_id")
        continuation_id = _nonempty_text(
            payload.get("continuation_observer_id"), "continuation_observer_id"
        )
        if observer_id != trace.get("observer_id") or observer_id != support.get(
            "observer_id"
        ):
            raise ObserverManifestError("checkpoint_observer_mismatch")
        if continuation_id != observer_id:
            raise ObserverManifestError(
                "narrow_verifier_requires_same_semantic_observer_continuation"
            )
        event_ids = list(trace.get("event_ids") or [])
        cut_event_id = _sha256_text(payload.get("cut_event_id"), "cut_event_id")
        if cut_event_id not in event_ids:
            raise ObserverManifestError("checkpoint_cut_event_not_in_trace")
        cut_index = event_ids.index(cut_event_id)
        next_index = _bounded_integer(
            payload.get("next_event_index"),
            field="next_event_index",
            minimum=1,
            maximum=len(event_ids),
        )
        if next_index != cut_index + 1 or next_index >= len(event_ids):
            raise ObserverManifestError("checkpoint_next_event_index_mismatch")
        snapshots = trace.get("snapshots") or []
        expected_snapshot = snapshots[cut_index]
        record_state = _integer_state_mapping(payload.get("record_state"), "record_state")
        origins = _hash_state_mapping(payload.get("record_origins"), "record_origins")
        if record_state != expected_snapshot["record_state"]:
            blockers.append("checkpoint_record_state_mismatch")
        if origins != expected_snapshot["record_origins"]:
            blockers.append("checkpoint_record_origins_mismatch")
        expected_history_root = canonical_hash(
            event_ids[:next_index], domain="oph.operational-observer.history-root.v1"
        )
        history_root = _sha256_text(
            payload.get("semantic_history_root"), "semantic_history_root"
        )
        if history_root != expected_history_root:
            blockers.append("checkpoint_history_root_mismatch")
        suffix_ids_raw = payload.get("committed_suffix_event_ids")
        if not isinstance(suffix_ids_raw, list) or not all(
            isinstance(value, str) and _SHA256_RE.fullmatch(value)
            for value in suffix_ids_raw
        ):
            raise ObserverManifestError("committed_suffix_event_ids_invalid")
        suffix_ids = event_ids[next_index:]
        if suffix_ids_raw != suffix_ids:
            blockers.append("checkpoint_suffix_commitment_mismatch")

        checkpoint_material = {
            key: value for key, value in payload.items() if key != "checkpoint_hash"
        }
        expected_checkpoint_hash = canonical_hash(
            checkpoint_material,
            domain="oph.operational-observer.checkpoint.v1",
        )
        checkpoint_hash = _sha256_text(
            payload.get("checkpoint_hash"), "checkpoint_hash"
        )
        if checkpoint_hash != expected_checkpoint_hash:
            blockers.append("checkpoint_hash_mismatch")

        replay = _replay_suffix_from_checkpoint(
            events=(trace.get("events") or [])[next_index:],
            initial_records=record_state,
            initial_origins=origins,
            ancestors=trace.get("ancestors") or {},
            stimuli=trace.get("stimuli") or [],
            config=config,
        )
        blockers.extend(replay["blockers"])
        expected_continuation_hash = canonical_hash(
            {
                "record_state": replay["final_record_state"],
                "suffix_predictions": replay["predictions"],
                "suffix_actions": replay["actions"],
                "last_event_id": suffix_ids[-1] if suffix_ids else cut_event_id,
            },
            domain="oph.operational-observer.continuation-state.v1",
        )
        declared_continuation_hash = _sha256_text(
            payload.get("continuation_state_hash"), "continuation_state_hash"
        )
        if declared_continuation_hash != expected_continuation_hash:
            blockers.append("checkpoint_continuation_state_hash_mismatch")
        if replay["final_record_state"] != trace.get("final_record_state"):
            blockers.append("checkpoint_suffix_final_state_differs_from_full_replay")
        later_action = any(
            event.get("kind") == "LOCAL_ACTION"
            for event in (trace.get("events") or [])[next_index:]
        )
        if not later_action:
            blockers.append("checkpoint_has_no_later_local_action")
    except (ObserverManifestError, IndexError, KeyError, TypeError, ValueError) as exc:
        blockers.append(str(exc))
        checkpoint_hash = None
        expected_checkpoint_hash = None
        cut_event_id = None
        next_index = None
        replay = {"passed": False, "blockers": [str(exc)]}

    return {
        "passed": not blockers,
        "blockers": sorted(set(blockers)),
        "checkpoint_hash": checkpoint_hash,
        "expected_checkpoint_hash": expected_checkpoint_hash,
        "cut_event_id": cut_event_id,
        "next_event_index": next_index,
        "suffix_replay_passed": replay.get("passed") is True,
        "semantic_continuation_only": True,
        "executor_process_identity_used": False,
    }


def _replay_suffix_from_checkpoint(
    *,
    events: Sequence[Mapping[str, Any]],
    initial_records: Mapping[str, int],
    initial_origins: Mapping[str, str],
    ancestors: Mapping[str, Sequence[str]],
    stimuli: Sequence[int],
    config: Any,
) -> dict[str, Any]:
    blockers: list[str] = []
    records = dict(initial_records)
    origins = dict(initial_origins)
    predictions_by_event: dict[str, dict[str, Any]] = {}
    predictions: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    if not isinstance(config, Mapping):
        return _failed_component("configuration_unavailable")
    modulus = int(config["action_modulus"])
    transaction_hashes = {
        str(event.get("canonical_repair_artifact_hash"))
        for event in events
        if event.get("kind") == "RECORD_COMMIT"
    }
    if len(transaction_hashes) > 1:
        blockers.append("suffix_contains_multiple_transaction_parents")
    for event in events:
        kind = event["kind"]
        event_id = event["event_id"]
        if kind == "RECORD_COMMIT":
            record_id = event["record_id"]
            if record_id in records or event.get("commit_status") != "COMMITTED":
                blockers.append(f"suffix_record_commit_invalid:{record_id}")
                continue
            records[record_id] = int(event["value"])
            origins[record_id] = event_id
            continue
        record_id = event["reads"][0]
        if record_id not in records:
            blockers.append(f"suffix_read_missing_checkpoint_or_write:{record_id}")
            continue
        if origins[record_id] not in set(ancestors.get(event_id, ())):
            blockers.append(f"suffix_read_origin_not_ancestor:{record_id}")
        index = int(event["feature_index"])
        if kind == "READBACK_PREDICTION":
            prediction = int(event["prediction"])
            if prediction != records[record_id]:
                blockers.append(f"suffix_prediction_replay_mismatch:{event_id}")
            row = {
                "event_id": event_id,
                "feature_index": index,
                "prediction": prediction,
            }
            predictions.append(row)
            predictions_by_event[event_id] = row
        else:
            feedback_id = event["feedback_event_id"]
            feedback = predictions_by_event.get(feedback_id)
            if feedback is None:
                blockers.append(f"suffix_feedback_prediction_unavailable:{feedback_id}")
                continue
            expected = (int(stimuli[index]) + int(feedback["prediction"])) % modulus
            if int(event["action"]) != expected:
                blockers.append(f"suffix_action_replay_mismatch:{event_id}")
            actions.append(
                {
                    "event_id": event_id,
                    "feature_index": index,
                    "action": int(event["action"]),
                }
            )
    return {
        "passed": not blockers,
        "blockers": blockers,
        "final_record_state": dict(sorted(records.items())),
        "predictions": predictions,
        "actions": actions,
    }


def _assemble_report(
    *,
    manifest: OperationalObserverManifest,
    store: _ArtifactStore,
    contract: Mapping[str, Any],
    firewall: Mapping[str, Any],
    support: Mapping[str, Any],
    record_provenance: Mapping[str, Any],
    trace: Mapping[str, Any],
    outcome_provenance: Mapping[str, Any],
    prediction: Mapping[str, Any],
    feedback: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    integrity = not store.blockers and all(
        row.get("passed") is True for row in store.rows.values()
    )
    contract_pass = contract.get("passed") is True
    firewall_pass = firewall.get("passed") is True
    support_common = integrity and contract_pass and firewall_pass
    receipts = {
        OBSERVER_ARTIFACT_INTEGRITY_RECEIPT: integrity,
        OBSERVER_CONTRACT_BINDING_RECEIPT: bool(integrity and contract_pass),
        OBSERVER_SOURCE_FIREWALL_RECEIPT: bool(integrity and firewall_pass),
        OBSERVER_RECORD_COMMIT_PROVENANCE_RECEIPT: bool(
            support_common and record_provenance.get("passed") is True
        ),
        OBSERVER_OUTCOME_PRODUCER_PROVENANCE_RECEIPT: bool(
            support_common and outcome_provenance.get("passed") is True
        ),
        A4_CONNECTED_OBSERVER_SUPPORT_RECEIPT: bool(
            support_common and support.get("connected") is True
        ),
        A4_BOUNDED_INTERFACE_RECEIPT: bool(
            support_common and support.get("bounded_interface") is True
        ),
        A3_RECORD_COMMIT_REPLAY_RECEIPT: bool(
            support_common
            and record_provenance.get("passed") is True
            and trace.get("record_commit_replay_passed") is True
        ),
        A3_READ_AFTER_WRITE_ANCESTRY_RECEIPT: bool(
            support_common and trace.get("read_after_write_ancestry_passed") is True
        ),
        A3_READBACK_PREDICTION_CONTROL_RECEIPT: bool(
            support_common and trace.get("passed") is True and prediction.get("passed") is True
            and outcome_provenance.get("passed") is True
        ),
        A4_FEEDBACK_ABLATION_RECEIPT: bool(
            support_common and trace.get("passed") is True and feedback.get("passed") is True
        ),
        A4_CHECKPOINT_CONTINUATION_REPLAY_RECEIPT: bool(
            support_common and trace.get("passed") is True and checkpoint.get("passed") is True
        ),
    }
    receipts[OPERATIONAL_SELF_READING_OBSERVER_RECEIPT] = all(receipts.values())
    blockers = sorted(
        set(
            store.blockers
            + list(contract.get("blockers") or [])
            + list(firewall.get("blockers") or [])
            + list(support.get("blockers") or [])
            + list(record_provenance.get("blockers") or [])
            + list(trace.get("blockers") or [])
            + list(outcome_provenance.get("blockers") or [])
            + list(prediction.get("blockers") or [])
            + list(feedback.get("blockers") or [])
            + list(checkpoint.get("blockers") or [])
        )
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "artifact_type": REPORT_ARTIFACT_TYPE,
        "manifest_path": manifest.path.name,
        "manifest_sha256": manifest.raw_sha256,
        "bundle_id": manifest.bundle_id,
        "verifier_module_sha256": _module_sha256(),
        "artifact_verification": {
            "rows": [store.rows[key] for key in sorted(store.rows)],
            "all_required_artifacts_present_and_rehashed": integrity,
            "deletion_sensitive_required_role_set": True,
            "cross_source_splice_binding_replayed": contract_pass,
        },
        "contract_binding": dict(contract),
        "semantic_firewall": dict(firewall),
        "federation_support": dict(support),
        "record_commit_provenance": dict(record_provenance),
        "record_and_trace_replay": dict(trace),
        "outcome_provenance": dict(outcome_provenance),
        "frozen_prediction_control": dict(prediction),
        "feedback_ablation": dict(feedback),
        "checkpoint_continuation": dict(checkpoint),
        **receipts,
        **{key: False for key in PHYSICAL_NONCLAIM_KEYS},
        "receipt": receipts[OPERATIONAL_SELF_READING_OBSERVER_RECEIPT],
        "claim_tier": "IMPLEMENTATION",
        "verdict": (
            "VALID_PASS"
            if receipts[OPERATIONAL_SELF_READING_OBSERVER_RECEIPT]
            else "INVALID_INSTRUMENT"
        ),
        "blockers": blockers,
        "claim_boundary": (
            "Exact finite replay of one content-addressed operational observer bundle. "
            "The canonical federation and proof-carrying repair parents are replayed, "
            "and the prediction/outcome split is a synthetic commit-reveal protocol "
            "without an external temporal attestation. The source parent remains "
            "conditional on a downstream common-source verifier. This receipt does not "
            "promote physical predictive independence, geometry, an independently "
            "derived clock, gravity, or Standard-Model emergence."
        ),
    }
    report["verification_report_sha256"] = canonical_hash(
        report, domain="oph.operational-observer.verification-report.v3"
    )
    return report


def _incomplete_report(blocker: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "artifact_type": REPORT_ARTIFACT_TYPE,
        "manifest_path": None,
        "manifest_sha256": None,
        "bundle_id": None,
        "verifier_module_sha256": _module_sha256(),
        **{key: False for key in FINITE_RECEIPT_KEYS},
        **{key: False for key in PHYSICAL_NONCLAIM_KEYS},
        "receipt": False,
        "claim_tier": "IMPLEMENTATION",
        "verdict": "INVALID_INSTRUMENT",
        "blockers": [blocker],
        "claim_boundary": (
            "Manifest parsing failed; no operational or physical claim is admitted."
        ),
    }
    report["verification_report_sha256"] = canonical_hash(
        report, domain="oph.operational-observer.verification-report.v3"
    )
    return report


def _failed_component(blocker: str) -> dict[str, Any]:
    return {"passed": False, "blockers": [blocker]}


def _require_binding(value: Any, expected_binding: Any) -> None:
    if not isinstance(value, Mapping):
        raise ObserverManifestError("evidence_binding_must_be_object")
    expected_fields = _binding_fields()
    _require_exact_fields(value, expected_fields, context="evidence_binding")
    if not isinstance(expected_binding, Mapping) or dict(value) != dict(
        expected_binding
    ):
        raise ObserverManifestError("evidence_contract_binding_mismatch")
    for field in (
        "source_bundle_receipt_hash",
        "federation_bundle_receipt_hash",
        "canonical_repair_artifact_hash",
        "transaction_parent_envelope_hash",
        "contract_binding_sha256",
    ):
        _sha256_text(value.get(field), field)
    for field in (
        "evaluator_artifact_id",
        "configuration_artifact_id",
        "seed_artifact_id",
        "evaluator_id",
        "configuration_id",
        "seed_id",
    ):
        _nonempty_text(value.get(field), field)


def _binding_fields() -> set[str]:
    return {
        "source_bundle_receipt_hash",
        "federation_bundle_receipt_hash",
        "canonical_repair_artifact_hash",
        "transaction_parent_envelope_hash",
        "evaluator_artifact_id",
        "configuration_artifact_id",
        "seed_artifact_id",
        "evaluator_id",
        "configuration_id",
        "seed_id",
        "contract_binding_sha256",
    }


def _require_npz_binding(value: Mapping[str, np.ndarray], expected_binding: Any) -> None:
    binding = {
        field: _npz_scalar_text(value[field], field) for field in _binding_fields()
    }
    _require_binding(binding, expected_binding)


def _source_stimuli(value: Any, expected_binding: Any) -> np.ndarray:
    del expected_binding  # the separately hashed JSON descriptor binds the NPY
    return _integer_vector(value, "stimulus")


def _decode_artifact(spec: ArtifactSpec, raw: bytes) -> Any:
    if spec.format == "json":
        if len(raw) > _MAX_JSON_BYTES:
            raise ObserverManifestError("json_artifact_exceeds_size_limit")
        value = _strict_json_loads(raw)
        _validate_json_finite(value)
        return value
    if spec.format == "npy":
        value = np.load(io.BytesIO(raw), allow_pickle=False)
        if not isinstance(value, np.ndarray):
            raise ObserverManifestError("npy_did_not_decode_to_array")
        _validate_array(value)
        return value
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ObserverManifestError("npz_contains_duplicate_array_names")
        result = {name: np.asarray(archive[name]) for name in archive.files}
    for array in result.values():
        _validate_array(array)
    return result


def _strict_json_loads(raw: bytes) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ObserverManifestError(f"duplicate_json_key:{key}")
            result[key] = value
        return result

    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs_hook)


def _validate_json_finite(value: Any) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ObserverManifestError("json_contains_nonfinite_number")
        return
    if isinstance(value, list):
        for child in value:
            _validate_json_finite(child)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise ObserverManifestError("json_keys_must_be_nonempty_strings")
            _validate_json_finite(child)
        return
    raise ObserverManifestError("json_contains_unsupported_value")


def _validate_array(value: np.ndarray) -> None:
    if value.size > _MAX_ARRAY_ELEMENTS:
        raise ObserverManifestError("array_exceeds_element_limit")
    if value.dtype.hasobject or value.dtype.kind in {"O", "V"}:
        raise ObserverManifestError("object_or_void_arrays_are_forbidden")
    if value.dtype.kind in {"f", "c"} and not np.all(np.isfinite(value)):
        raise ObserverManifestError("array_contains_nonfinite_value")
    if value.dtype.kind not in {"b", "i", "u", "f", "c", "U", "S"}:
        raise ObserverManifestError(f"unsupported_array_dtype:{value.dtype}")


def _decoded_value_hash(value: Any) -> str:
    return canonical_hash(_decoded_jsonable(value), domain="oph.primitive-artifact.value.v1")


def _decoded_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "c":
            data = [
                [float(item.real), float(item.imag)]
                for item in value.reshape(-1).tolist()
            ]
        elif value.dtype.kind == "S":
            data = [item.hex() for item in value.reshape(-1).tolist()]
        else:
            data = value.reshape(-1).tolist()
        return {"dtype": value.dtype.str, "shape": list(value.shape), "data": data}
    if isinstance(value, Mapping):
        return {str(key): _decoded_jsonable(child) for key, child in sorted(value.items())}
    if isinstance(value, list):
        return [_decoded_jsonable(child) for child in value]
    return value


def _value_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        return {"kind": "ndarray", "dtype": value.dtype.str, "shape": list(value.shape)}
    if isinstance(value, Mapping) and all(isinstance(child, np.ndarray) for child in value.values()):
        return {
            "kind": "npz",
            "arrays": {
                str(key): {"dtype": child.dtype.str, "shape": list(child.shape)}
                for key, child in sorted(value.items())
            },
        }
    return {"kind": "json", "root_type": type(value).__name__}


def _integer_vector(value: Any, field: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 1 or value.dtype.kind not in {"i", "u"}:
        raise ObserverManifestError(f"{field}_must_be_one_dimensional_integer_array")
    if value.size == 0:
        raise ObserverManifestError(f"{field}_must_be_nonempty")
    if value.dtype.kind == "u" and np.any(value > np.iinfo(np.int64).max):
        raise ObserverManifestError(f"{field}_exceeds_int64")
    return np.asarray(value, dtype=np.int64)


def _npz_scalar_text(value: Any, field: str) -> str:
    if not isinstance(value, np.ndarray) or value.shape != () or value.dtype.kind != "U":
        raise ObserverManifestError(f"{field}_must_be_unicode_scalar_array")
    return _nonempty_text(value.item(), field)


def _integer_state_mapping(value: Any, field: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ObserverManifestError(f"{field}_must_be_object")
    result: dict[str, int] = {}
    for key, child in value.items():
        result[_nonempty_text(key, f"{field}_key")] = _bounded_integer(
            child, field=f"{field}:{key}", minimum=0, maximum=2**31 - 1
        )
    return dict(sorted(result.items()))


def _hash_state_mapping(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ObserverManifestError(f"{field}_must_be_object")
    result = {
        _nonempty_text(key, f"{field}_key"): _sha256_text(child, f"{field}:{key}")
        for key, child in value.items()
    }
    return dict(sorted(result.items()))


def _port_tuple(value: Any, field: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ObserverManifestError(f"{field}_must_be_integer_array")
    ports = tuple(
        _bounded_integer(item, field=field, minimum=0, maximum=_PORT_COUNT - 1)
        for item in value
    )
    if len(set(ports)) != len(ports):
        raise ObserverManifestError(f"{field}_ports_must_be_unique")
    return ports


def _bounded_integer(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise ObserverManifestError(
            f"{field}_must_be_integer_in_closed_range_{minimum}_{maximum}"
        )
    return value


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ObserverManifestError(f"{field}_must_be_nonempty_string")
    return value


def _sha256_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ObserverManifestError(f"{field}_must_be_sha256_digest")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    *,
    context: str,
) -> None:
    unknown = set(value) - set(expected)
    missing = set(expected) - set(value)
    if unknown or missing:
        raise ObserverManifestError(
            f"{context}_fields_invalid:unknown={sorted(unknown)}:missing={sorted(missing)}"
        )


def _raw_sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _module_sha256() -> str:
    try:
        return _raw_sha256(Path(__file__).read_bytes())
    except OSError:
        return "sha256:" + "0" * 64
