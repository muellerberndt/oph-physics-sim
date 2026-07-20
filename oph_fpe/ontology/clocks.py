"""Four disjoint clock domains with no implicit conversion path."""

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
from .observers import ObserverToken


@dataclass(frozen=True, slots=True)
class ExecutionClockReading:
    """Wall/CPU/queue telemetry; never semantic or physical time."""

    process_id: str
    wall_clock_ns: int
    cpu_time_ns: int
    queue_delay_ns: int

    def __post_init__(self) -> None:
        require_nonempty(self.process_id, field_name="process_id")
        if min(self.wall_clock_ns, self.cpu_time_ns, self.queue_delay_ns) < 0:
            raise ValueError("execution clock values must be nonnegative")

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema": "oph.execution-clock.v1",
            "clock_domain": "EXECUTION",
            "process_id": self.process_id,
            "wall_clock_ns": self.wall_clock_ns,
            "cpu_time_ns": self.cpu_time_ns,
            "queue_delay_ns": self.queue_delay_ns,
            "physical_clock_receipt": False,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_jsonable())


@dataclass(frozen=True, slots=True)
class RepairOrderReading:
    """Causal order of committed transactions, not elapsed time."""

    commit_id: str
    parent_commit_ids: tuple[str, ...]
    causal_depth: int

    def __post_init__(self) -> None:
        require_sha256(self.commit_id, field_name="commit_id")
        if isinstance(self.parent_commit_ids, (str, bytes)):
            raise TypeError("parent_commit_ids must be a sequence, not text")
        parents = tuple(sorted(set(self.parent_commit_ids)))
        for parent in parents:
            require_sha256(parent, field_name="parent_commit_id")
        object.__setattr__(self, "parent_commit_ids", parents)
        if self.causal_depth < 0:
            raise ValueError("causal_depth must be nonnegative")

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema": "oph.repair-order.v1",
            "clock_domain": "REPAIR_ORDER",
            "commit_id": self.commit_id,
            "parent_commit_ids": list(self.parent_commit_ids),
            "causal_depth": self.causal_depth,
            "physical_clock_receipt": False,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_jsonable())


@dataclass(frozen=True, slots=True)
class SemanticOrderReading:
    """Observer-readable causal event order, distinct from repair scheduling."""

    observer_token: ObserverToken
    event_id: str
    parent_event_ids: tuple[str, ...]
    observer_causal_depth: int

    def __post_init__(self) -> None:
        if not isinstance(self.observer_token, ObserverToken):
            raise TypeError("observer_token must be an ObserverToken")
        require_sha256(self.event_id, field_name="event_id")
        if isinstance(self.parent_event_ids, (str, bytes)):
            raise TypeError("parent_event_ids must be a sequence, not text")
        parents = tuple(sorted(set(self.parent_event_ids)))
        for parent in parents:
            require_sha256(parent, field_name="parent_event_id")
        object.__setattr__(self, "parent_event_ids", parents)
        if self.observer_causal_depth < 0:
            raise ValueError("observer_causal_depth must be nonnegative")

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema": "oph.semantic-order.v1",
            "clock_domain": "SEMANTIC_ORDER",
            "observer_token": self.observer_token.token_hash,
            "event_id": self.event_id,
            "parent_event_ids": list(self.parent_event_ids),
            "observer_causal_depth": self.observer_causal_depth,
            "physical_clock_receipt": False,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_jsonable())


@dataclass(frozen=True, slots=True)
class OperationalClockReading:
    """Independently calibrated readout; a receipt must promote it separately."""

    observer_token: ObserverToken
    checkpoint_hash: str
    distribution: FrozenMap | Mapping[str, Any]
    calibration_id: str
    calibration_receipt_hash: str
    affine_scale: float
    affine_offset: float
    residual_bound: float

    def __post_init__(self) -> None:
        if not isinstance(self.observer_token, ObserverToken):
            raise TypeError("observer_token must be an ObserverToken")
        require_sha256(self.checkpoint_hash, field_name="checkpoint_hash")
        require_nonempty(self.calibration_id, field_name="calibration_id")
        require_sha256(
            self.calibration_receipt_hash, field_name="calibration_receipt_hash"
        )
        object.__setattr__(
            self, "distribution", FrozenMap.from_mapping(self.distribution)
        )
        for field_name in ("affine_scale", "affine_offset", "residual_bound"):
            if not math.isfinite(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be finite")
        if self.affine_scale == 0.0:
            raise ValueError("affine_scale must be nonzero")
        if self.residual_bound < 0.0:
            raise ValueError("residual_bound must be nonnegative")

    @property
    def reading_hash(self) -> str:
        return canonical_hash(
            self.to_jsonable(), domain="oph.operational-clock-reading.v1"
        )

    @property
    def physical_clock_receipt(self) -> bool:
        # A typed calibration reference is necessary but not itself sufficient.
        return False

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema": "oph.operational-clock.v1",
            "clock_domain": "OPERATIONAL",
            "observer_token": self.observer_token.token_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "distribution": self.distribution,
            "calibration_id": self.calibration_id,
            "calibration_receipt_hash": self.calibration_receipt_hash,
            "affine_scale": self.affine_scale,
            "affine_offset": self.affine_offset,
            "residual_bound": self.residual_bound,
            "physical_clock_receipt": False,
            "nonclaims": ["modular_time", "repair_time", "spacetime_time"],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_jsonable())
