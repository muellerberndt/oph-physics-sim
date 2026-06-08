from __future__ import annotations

from dataclasses import dataclass

from oph_universe.arrow.records import RecordAlgebra
from oph_universe.arrow.schemas import stable_hash


@dataclass(frozen=True)
class Checkpoint:
    checkpoint_id: str
    t: int
    record_algebra: RecordAlgebra
    accessible_state_hash: str
    external_interface_hash: str
    schedule_class_hash: str
    provenance_bundle_hash: str
    macrostate_id: str
    s_of_bits: float
    s_max_bits: float
    hidden_export_budget_bits: float
    approximation_error_bits: float = 0.0


def checkpoint_equivalent(a: Checkpoint, b: Checkpoint) -> bool:
    return (
        a.record_algebra.algebra_id == b.record_algebra.algebra_id
        and a.accessible_state_hash == b.accessible_state_hash
        and a.external_interface_hash == b.external_interface_hash
        and a.schedule_class_hash == b.schedule_class_hash
        and a.provenance_bundle_hash == b.provenance_bundle_hash
    )


def future_law_signature(checkpoint: Checkpoint, continuation_law_id: str) -> str:
    return stable_hash(
        {
            "record_algebra": checkpoint.record_algebra.algebra_id,
            "accessible": checkpoint.accessible_state_hash,
            "interface": checkpoint.external_interface_hash,
            "schedule": checkpoint.schedule_class_hash,
            "provenance": checkpoint.provenance_bundle_hash,
            "law": continuation_law_id,
        }
    )

