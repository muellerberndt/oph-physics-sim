"""Separate execution logs, semantic events, record algebra, and checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from ._canonical import (
    FrozenMap,
    canonical_hash,
    canonical_json,
    require_nonempty,
    require_sha256,
)
from .firewall import require_no_presentation_fields
from .observers import ObserverToken


@dataclass(frozen=True, slots=True)
class ExecutionLogEntry:
    """Debug/provenance metadata that is never part of semantic event identity."""

    worker_id: str
    queue_index: int
    retry_count: int
    wall_clock_ns: int
    process_id: str
    trace_uuid: str
    message: str

    def __post_init__(self) -> None:
        require_nonempty(self.worker_id, field_name="worker_id")
        require_nonempty(self.process_id, field_name="process_id")
        require_nonempty(self.trace_uuid, field_name="trace_uuid")
        if self.queue_index < 0 or self.retry_count < 0 or self.wall_clock_ns < 0:
            raise ValueError("execution counters must be nonnegative")

    @property
    def execution_log_hash(self) -> str:
        return canonical_hash(self.to_jsonable(), domain="oph.execution-log.v1")

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema": "oph.execution-log-entry.v1",
            "worker_id": self.worker_id,
            "queue_index": self.queue_index,
            "retry_count": self.retry_count,
            "wall_clock_ns": self.wall_clock_ns,
            "process_id": self.process_id,
            "trace_uuid": self.trace_uuid,
            "message": self.message,
            "semantic_evidence": False,
            "physical_promotion_receipt": False,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_jsonable())


@dataclass(frozen=True, slots=True)
class SemanticEvent:
    """Observer-visible causal event with executor-independent identity."""

    canonical_payload: FrozenMap | Mapping[str, Any]
    observer_token: ObserverToken
    visible_footprint: tuple[str, ...]
    semantic_parents: tuple[str, ...]
    event_id: str = ""

    def __post_init__(self) -> None:
        payload = FrozenMap.from_mapping(self.canonical_payload)
        require_no_presentation_fields(
            payload, context="SemanticEvent.canonical_payload"
        )
        object.__setattr__(self, "canonical_payload", payload)
        if not isinstance(self.observer_token, ObserverToken):
            raise TypeError("observer_token must be an ObserverToken")
        if isinstance(self.visible_footprint, (str, bytes)):
            raise TypeError("visible_footprint must be a sequence, not text")
        footprint = tuple(sorted(set(self.visible_footprint)))
        if any(not isinstance(item, str) or not item for item in footprint):
            raise ValueError("visible_footprint entries must be nonempty strings")
        object.__setattr__(self, "visible_footprint", footprint)
        if isinstance(self.semantic_parents, (str, bytes)):
            raise TypeError("semantic_parents must be a sequence, not text")
        parents = tuple(sorted(set(self.semantic_parents)))
        for parent in parents:
            require_sha256(parent, field_name="semantic_parent")
        object.__setattr__(self, "semantic_parents", parents)
        expected = canonical_hash(
            self._identity_material(), domain="oph.semantic-event-id.v1"
        )
        if self.event_id:
            require_sha256(self.event_id, field_name="event_id")
            if self.event_id != expected:
                raise ValueError(
                    "event_id includes noncanonical or mismatched material"
                )
        else:
            object.__setattr__(self, "event_id", expected)

    def _identity_material(self) -> dict[str, Any]:
        # Deliberately excludes worker, queue, retry, wall-clock, UUID, and all
        # other ExecutionLogEntry fields.
        return {
            "canonical_payload": self.canonical_payload,
            "observer_token": self.observer_token.token_hash,
            "visible_footprint": self.visible_footprint,
            "semantic_parents": self.semantic_parents,
        }

    @property
    def physical_event_receipt(self) -> bool:
        return False

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema": "oph.semantic-event.v1",
            **self._identity_material(),
            "event_id": self.event_id,
            "physical_event_receipt": False,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_jsonable())


@dataclass(frozen=True, slots=True)
class RecordAlgebra:
    """Stable re-readable algebraic record surface, not a persistent field."""

    algebra_id: str
    projector_ids: tuple[str, ...]
    central_observables: FrozenMap | Mapping[str, Any]
    protected_record_hashes: tuple[str, ...]
    algebra_hash: str = ""

    def __post_init__(self) -> None:
        require_nonempty(self.algebra_id, field_name="algebra_id")
        if isinstance(self.projector_ids, (str, bytes)):
            raise TypeError("projector_ids must be a sequence, not text")
        projectors = tuple(sorted(set(self.projector_ids)))
        if any(not isinstance(value, str) or not value for value in projectors):
            raise ValueError("projector_ids must be nonempty strings")
        object.__setattr__(self, "projector_ids", projectors)
        observables = FrozenMap.from_mapping(self.central_observables)
        require_no_presentation_fields(
            observables, context="RecordAlgebra.central_observables"
        )
        object.__setattr__(self, "central_observables", observables)
        if isinstance(self.protected_record_hashes, (str, bytes)):
            raise TypeError("protected_record_hashes must be a sequence, not text")
        record_hashes = tuple(sorted(set(self.protected_record_hashes)))
        for record_hash in record_hashes:
            require_sha256(record_hash, field_name="protected_record_hash")
        object.__setattr__(self, "protected_record_hashes", record_hashes)
        expected = canonical_hash(self._hash_material(), domain="oph.record-algebra.v1")
        if self.algebra_hash:
            require_sha256(self.algebra_hash, field_name="algebra_hash")
            if self.algebra_hash != expected:
                raise ValueError("algebra_hash does not match record algebra")
        else:
            object.__setattr__(self, "algebra_hash", expected)

    def _hash_material(self) -> dict[str, Any]:
        return {
            "algebra_id": self.algebra_id,
            "projector_ids": self.projector_ids,
            "central_observables": self.central_observables,
            "protected_record_hashes": self.protected_record_hashes,
        }

    @property
    def physical_record_receipt(self) -> bool:
        return False

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema": "oph.record-algebra.v1",
            **self._hash_material(),
            "algebra_hash": self.algebra_hash,
            "physical_record_receipt": False,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_jsonable())


@dataclass(frozen=True, slots=True)
class ProjectorDiagnostic:
    """Primitive finite projector checks; passing is not a physical record claim."""

    projector_id: str
    idempotence_error: float
    hermiticity_error: float
    centrality_error: float
    idempotence_tolerance: float
    hermiticity_tolerance: float
    centrality_tolerance: float
    primitive_payload_hash: str
    verifier_hash: str

    def __post_init__(self) -> None:
        require_nonempty(self.projector_id, field_name="projector_id")
        require_sha256(self.primitive_payload_hash, field_name="primitive_payload_hash")
        require_sha256(self.verifier_hash, field_name="verifier_hash")
        for field_name in (
            "idempotence_error",
            "hermiticity_error",
            "centrality_error",
            "idempotence_tolerance",
            "hermiticity_tolerance",
            "centrality_tolerance",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and nonnegative")

    @property
    def passed(self) -> bool:
        return (
            self.idempotence_error <= self.idempotence_tolerance
            and self.hermiticity_error <= self.hermiticity_tolerance
            and self.centrality_error <= self.centrality_tolerance
        )

    @property
    def physical_record_receipt(self) -> bool:
        return False

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema": "oph.projector-diagnostic.v1",
            "projector_id": self.projector_id,
            "primitive_payload_hash": self.primitive_payload_hash,
            "verifier_hash": self.verifier_hash,
            "errors": {
                "idempotence": self.idempotence_error,
                "hermiticity": self.hermiticity_error,
                "centrality": self.centrality_error,
            },
            "tolerances": {
                "idempotence": self.idempotence_tolerance,
                "hermiticity": self.hermiticity_tolerance,
                "centrality": self.centrality_tolerance,
            },
            "passed": self.passed,
            "physical_record_receipt": False,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_jsonable())


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Semantic continuation data, separate from executor replay metadata."""

    observer_token: ObserverToken
    quotient_hash: str
    semantic_history_root: str
    continuation_data: FrozenMap | Mapping[str, Any]
    checkpoint_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.observer_token, ObserverToken):
            raise TypeError("observer_token must be an ObserverToken")
        require_sha256(self.quotient_hash, field_name="quotient_hash")
        require_sha256(self.semantic_history_root, field_name="semantic_history_root")
        data = FrozenMap.from_mapping(self.continuation_data)
        require_no_presentation_fields(data, context="Checkpoint.continuation_data")
        object.__setattr__(self, "continuation_data", data)
        expected = canonical_hash(self._hash_material(), domain="oph.checkpoint.v1")
        if self.checkpoint_hash:
            require_sha256(self.checkpoint_hash, field_name="checkpoint_hash")
            if self.checkpoint_hash != expected:
                raise ValueError("checkpoint_hash does not match continuation data")
        else:
            object.__setattr__(self, "checkpoint_hash", expected)

    def _hash_material(self) -> dict[str, Any]:
        return {
            "observer_token": self.observer_token.token_hash,
            "quotient_hash": self.quotient_hash,
            "semantic_history_root": self.semantic_history_root,
            "continuation_data": self.continuation_data,
        }

    @property
    def physical_continuation_receipt(self) -> bool:
        return False

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema": "oph.checkpoint.v1",
            **self._hash_material(),
            "checkpoint_hash": self.checkpoint_hash,
            "physical_continuation_receipt": False,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_jsonable())
