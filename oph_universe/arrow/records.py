from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RecordEvent:
    record_id: str
    t: int
    patch_ids: tuple[int, ...]
    source_id: str
    payload_bits: float
    value_hash: str
    decoder_id: str | None
    epsilon: float
    parent_record_ids: tuple[str, ...]
    provenance_ids: tuple[str, ...]
    substrate_id: str | None
    committed: bool = True


@dataclass
class RecordAlgebra:
    algebra_id: str
    t: int
    events: dict[str, RecordEvent]

    def atom_count(self) -> int:
        return max(1, 2 ** len([event for event in self.events.values() if event.committed]))

    def capacity_bits(self) -> float:
        return math.log2(max(1, self.atom_count()))

    def payload_bits(self) -> float:
        seen_sources: set[tuple[str, str, str | None]] = set()
        total = 0.0
        for event in sorted(self.events.values(), key=lambda item: item.record_id):
            if not event.committed:
                continue
            key = (event.source_id, event.value_hash, event.decoder_id)
            if key in seen_sources:
                continue
            seen_sources.add(key)
            total += max(0.0, float(event.payload_bits))
        return total

    def append(self, event: RecordEvent) -> "RecordAlgebra":
        if event.record_id in self.events:
            raise ValueError(f"record already exists: {event.record_id}")
        self.events[event.record_id] = event
        return self


def record_tower_capacity(algebras: list[RecordAlgebra]) -> list[float]:
    return [alg.capacity_bits() for alg in algebras]

