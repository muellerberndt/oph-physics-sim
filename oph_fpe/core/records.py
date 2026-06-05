from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from oph_fpe.core.patchnet import PatchNet


@dataclass(frozen=True)
class RecordEvent:
    cycle: int
    node: int
    packet: tuple[Any, ...]
    stable_count: int


def update_records(net: PatchNet, cycle: int, commit_cycles: int) -> list[RecordEvent]:
    events: list[RecordEvent] = []
    for node, state in net.states.items():
        packet = state.observer_packet()
        if packet == state.candidate_record:
            state.stable_count += 1
        else:
            state.candidate_record = packet
            state.stable_count = 1
        if state.stable_count >= commit_cycles and state.record != packet:
            state.record = packet
            state.commit_count += 1
            events.append(RecordEvent(cycle=cycle, node=node, packet=packet, stable_count=state.stable_count))
    return events
