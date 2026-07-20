"""Strict state strata for presentation, semantics, quotient, and normal form."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ._canonical import (
    FrozenMap,
    canonical_hash,
    canonical_json,
    require_nonempty,
    require_sha256,
)
from .firewall import require_no_presentation_fields


STATE_SCHEMA = "oph.simulator-state-ontology.v1"


class FiberStatus(str, Enum):
    UNREALIZABLE = "UNREALIZABLE"
    UNIQUE = "UNIQUE"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class PresentationState:
    """Implementation storage; no field in this class is automatically physical."""

    carrier_states: FrozenMap | Mapping[str, Any]
    seam_states: FrozenMap | Mapping[str, Any]
    scheduler_state: FrozenMap | Mapping[str, Any]
    worker_state: FrozenMap | Mapping[str, Any]
    rng_state: FrozenMap | Mapping[str, Any]
    provenance: FrozenMap | Mapping[str, Any]

    def __post_init__(self) -> None:
        for field_name in (
            "carrier_states",
            "seam_states",
            "scheduler_state",
            "worker_state",
            "rng_state",
            "provenance",
        ):
            object.__setattr__(
                self,
                field_name,
                FrozenMap.from_mapping(getattr(self, field_name)),
            )

    @property
    def presentation_hash(self) -> str:
        return canonical_hash(self.to_jsonable(), domain="oph.presentation-state.v1")

    @property
    def physical_promotion_receipt(self) -> bool:
        return False

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema": STATE_SCHEMA,
            "state_type": "PRESENTATION_STATE",
            "carrier_states": self.carrier_states.to_jsonable(),
            "seam_states": self.seam_states.to_jsonable(),
            "scheduler_state": self.scheduler_state.to_jsonable(),
            "worker_state": self.worker_state.to_jsonable(),
            "rng_state": self.rng_state.to_jsonable(),
            "provenance": self.provenance.to_jsonable(),
            "physical_promotion_receipt": False,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_jsonable())


@dataclass(frozen=True, slots=True)
class SemanticCarrierState:
    """Finite carrier variables that can affect declared interface behavior."""

    carrier_id: str
    accessible_state: FrozenMap | Mapping[str, Any]
    interface_states: FrozenMap | Mapping[str, Any]
    record_state: FrozenMap | Mapping[str, Any]
    checkpoint_state: FrozenMap | Mapping[str, Any]
    sector_state: FrozenMap | Mapping[str, Any]

    def __post_init__(self) -> None:
        require_nonempty(self.carrier_id, field_name="carrier_id")
        semantic_payload: dict[str, Any] = {}
        for field_name in (
            "accessible_state",
            "interface_states",
            "record_state",
            "checkpoint_state",
            "sector_state",
        ):
            frozen = FrozenMap.from_mapping(getattr(self, field_name))
            object.__setattr__(self, field_name, frozen)
            semantic_payload[field_name] = frozen
        require_no_presentation_fields(semantic_payload, context="SemanticCarrierState")

    @property
    def semantic_hash(self) -> str:
        return canonical_hash(
            self.to_jsonable(), domain="oph.semantic-carrier-state.v1"
        )

    @property
    def physical_promotion_receipt(self) -> bool:
        return False

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema": STATE_SCHEMA,
            "state_type": "SEMANTIC_CARRIER_STATE",
            "carrier_id": self.carrier_id,
            "accessible_state": self.accessible_state.to_jsonable(),
            "interface_states": self.interface_states.to_jsonable(),
            "record_state": self.record_state.to_jsonable(),
            "checkpoint_state": self.checkpoint_state.to_jsonable(),
            "sector_state": self.sector_state.to_jsonable(),
            "physical_promotion_receipt": False,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_jsonable())


@dataclass(frozen=True, slots=True)
class QuotientState:
    """Content-addressed state after hidden presentation redundancies are removed."""

    canonical_interface_data: bytes
    protected_records: bytes
    sector_invariants: bytes
    semantic_history_root: str
    quotient_hash: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "canonical_interface_data",
            "protected_records",
            "sector_invariants",
        ):
            if not isinstance(getattr(self, field_name), bytes):
                raise TypeError(f"{field_name} must be bytes")
        require_sha256(self.semantic_history_root, field_name="semantic_history_root")
        expected = canonical_hash(self._hash_material(), domain="oph.quotient-state.v1")
        if self.quotient_hash:
            require_sha256(self.quotient_hash, field_name="quotient_hash")
            if self.quotient_hash != expected:
                raise ValueError("quotient_hash does not match quotient-visible state")
        else:
            object.__setattr__(self, "quotient_hash", expected)

    def _hash_material(self) -> dict[str, Any]:
        return {
            "canonical_interface_data": self.canonical_interface_data,
            "protected_records": self.protected_records,
            "sector_invariants": self.sector_invariants,
            "semantic_history_root": self.semantic_history_root,
        }

    @property
    def physical_promotion_receipt(self) -> bool:
        return False

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema": STATE_SCHEMA,
            "state_type": "QUOTIENT_STATE",
            **self._hash_material(),
            "quotient_hash": self.quotient_hash,
            "physical_promotion_receipt": False,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_jsonable())


@dataclass(frozen=True, slots=True)
class NormalFormState:
    """A certified settled quotient state, without a physical selector claim."""

    quotient_state: QuotientState
    fiber_status: FiberStatus
    normalizer_contract_id: str
    normalizer_receipt_hash: str
    normal_form_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.quotient_state, QuotientState):
            raise TypeError("quotient_state must be a QuotientState")
        if not isinstance(self.fiber_status, FiberStatus):
            raise TypeError("fiber_status must be a FiberStatus")
        require_nonempty(
            self.normalizer_contract_id, field_name="normalizer_contract_id"
        )
        require_sha256(
            self.normalizer_receipt_hash, field_name="normalizer_receipt_hash"
        )
        expected = canonical_hash(
            self._hash_material(), domain="oph.normal-form-state.v1"
        )
        if self.normal_form_hash:
            require_sha256(self.normal_form_hash, field_name="normal_form_hash")
            if self.normal_form_hash != expected:
                raise ValueError("normal_form_hash does not match normal-form material")
        else:
            object.__setattr__(self, "normal_form_hash", expected)

    def _hash_material(self) -> dict[str, Any]:
        return {
            "quotient_hash": self.quotient_state.quotient_hash,
            "fiber_status": self.fiber_status.value,
            "normalizer_contract_id": self.normalizer_contract_id,
            "normalizer_receipt_hash": self.normalizer_receipt_hash,
        }

    @property
    def physical_promotion_receipt(self) -> bool:
        return False

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema": STATE_SCHEMA,
            "state_type": "NORMAL_FORM_STATE",
            **self._hash_material(),
            "normal_form_hash": self.normal_form_hash,
            "physical_promotion_receipt": False,
            "nonclaims": [
                "event_manifold",
                "h3_geometry",
                "physical_ensemble",
                "physical_vacuum",
                "probability_law",
            ],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_jsonable())
