from __future__ import annotations

from collections.abc import Iterable

from oph_fpe.small_universe.model import SmallUniverse, State, edge_key, phi


REPAIR_BACKEND = "parent_copy_calibration_v1"


def enabled_repairs(universe: SmallUniverse, state: State) -> list[int]:
    out: list[int] = []
    for node, parent in sorted(universe.parent.items()):
        expected = state[parent] ^ universe.offsets[edge_key(parent, node)]
        if state[node] != expected:
            out.append(node)
    return out


def apply_repair(universe: SmallUniverse, state: State, node: int) -> State:
    parent = universe.parent[node]
    expected = state[parent] ^ universe.offsets[edge_key(parent, node)]
    if state[node] == expected:
        return state
    new_state = list(state)
    new_state[node] = expected
    return tuple(new_state)


def repair_transition_rows(universe: SmallUniverse, states: Iterable[State]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for state in states:
        before_phi = phi(universe, state)
        for node in enabled_repairs(universe, state):
            next_state = apply_repair(universe, state, node)
            after_phi = phi(universe, next_state)
            rows.append(
                {
                    "state": list(state),
                    "node": int(node),
                    "parent": int(universe.parent[node]),
                    "next_state": list(next_state),
                    "phi_before": int(before_phi),
                    "phi_after": int(after_phi),
                    "delta_phi": int(after_phi - before_phi),
                    "accepted_by_generator": after_phi < before_phi,
                }
            )
    return rows
