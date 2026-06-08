from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RecordTowerPoint:
    t: int
    record_capacity_bits: float
    atom_count: int
    payload_bits: float


def infer_record_arrow(tower: list[RecordTowerPoint]) -> Literal["forward", "reverse", "ambiguous"]:
    if len(tower) < 2:
        return "ambiguous"
    ordered = sorted(tower, key=lambda point: point.t)
    payload = [point.payload_bits for point in ordered]
    capacity = [point.record_capacity_bits for point in ordered]
    forward = _strict_non_decreasing(payload) and _strict_non_decreasing(capacity)
    reverse = _strict_non_decreasing(list(reversed(payload))) and _strict_non_decreasing(list(reversed(capacity)))
    if forward and not reverse:
        return "forward"
    if reverse and not forward:
        return "reverse"
    return "ambiguous"


def record_reversal_erasure_cost_bits(later_payload_bits: float, earlier_payload_bits: float) -> float:
    return max(0.0, float(later_payload_bits) - float(earlier_payload_bits))


def _strict_non_decreasing(values: list[float]) -> bool:
    return all(float(b) > float(a) for a, b in zip(values, values[1:]))

