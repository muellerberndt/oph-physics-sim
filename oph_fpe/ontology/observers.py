"""Semantic observer registry tokens and evidence-bearing continuation arrows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

from ._canonical import canonical_hash, canonical_json, require_nonempty, require_sha256


class ObserverKind(str, Enum):
    SEMANTIC_PATCH = "SEMANTIC_PATCH"
    COMPOSITE_FEDERATION = "COMPOSITE_FEDERATION"
    EXTERNAL_REGISTERED = "EXTERNAL_REGISTERED"


class ContinuationKind(str, Enum):
    CONTINUATION = "CONTINUATION"
    SPLIT = "SPLIT"
    MERGE = "MERGE"


@dataclass(frozen=True, slots=True)
class ObserverToken:
    """Semantic registry identity, independent of process or carrier identity."""

    kind: ObserverKind
    birth_event: str
    lineage_root: str
    registry_namespace: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ObserverKind):
            raise TypeError("kind must be an ObserverKind")
        require_sha256(self.birth_event, field_name="birth_event")
        require_sha256(self.lineage_root, field_name="lineage_root")
        require_nonempty(self.registry_namespace, field_name="registry_namespace")

    @property
    def token_hash(self) -> str:
        return canonical_hash(self.to_jsonable(), domain="oph.observer-token.v1")

    @property
    def physical_identity_receipt(self) -> bool:
        return False

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema": "oph.observer-token.v1",
            "kind": self.kind.value,
            "birth_event": self.birth_event,
            "lineage_root": self.lineage_root,
            "registry_namespace": self.registry_namespace,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_jsonable())


@dataclass(frozen=True, slots=True)
class ContinuationArrow:
    """An explicitly evidenced registry arrow; not identity by object reuse."""

    source_observer: ObserverToken
    target_observer: ObserverToken
    source_checkpoint: str
    target_checkpoint: str
    continuation_error: float
    evidence_hash: str
    kind: ContinuationKind = ContinuationKind.CONTINUATION

    def __post_init__(self) -> None:
        if not isinstance(self.source_observer, ObserverToken):
            raise TypeError("source_observer must be an ObserverToken")
        if not isinstance(self.target_observer, ObserverToken):
            raise TypeError("target_observer must be an ObserverToken")
        if not isinstance(self.kind, ContinuationKind):
            raise TypeError("kind must be a ContinuationKind")
        require_sha256(self.source_checkpoint, field_name="source_checkpoint")
        require_sha256(self.target_checkpoint, field_name="target_checkpoint")
        require_sha256(self.evidence_hash, field_name="evidence_hash")
        if not math.isfinite(self.continuation_error) or self.continuation_error < 0.0:
            raise ValueError("continuation_error must be finite and nonnegative")

    @property
    def arrow_hash(self) -> str:
        return canonical_hash(self.to_jsonable(), domain="oph.continuation-arrow.v1")

    @property
    def physical_continuation_receipt(self) -> bool:
        return False

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema": "oph.continuation-arrow.v1",
            "kind": self.kind.value,
            "source_observer": self.source_observer.token_hash,
            "target_observer": self.target_observer.token_hash,
            "source_checkpoint": self.source_checkpoint,
            "target_checkpoint": self.target_checkpoint,
            "continuation_error": self.continuation_error,
            "evidence_hash": self.evidence_hash,
            "physical_continuation_receipt": False,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_jsonable())
