from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import networkx as nx

from oph_fpe.core.patchnet import PatchNet


@dataclass(frozen=True)
class HolonomyDefect:
    cycle_key: tuple[int, ...]
    ordered_cycle: tuple[int, ...]
    holonomy: Hashable
    holonomy_label: str
    support_size: int


def cycle_holonomy(net: PatchNet, ordered_cycle: list[int] | tuple[int, ...]) -> Hashable:
    if len(ordered_cycle) < 3:
        raise ValueError("cycle holonomy requires at least three nodes")
    product = net.group.identity
    nodes = tuple(ordered_cycle)
    for index, left in enumerate(nodes):
        right = nodes[(index + 1) % len(nodes)]
        if right not in net.states[left].gauges:
            raise ValueError(f"cycle uses non-edge {left}->{right}")
        product = net.group.multiply(product, net.states[left].gauges[right])
    return product


def scan_holonomy_defects(net: PatchNet, max_cycles: int = 512) -> list[HolonomyDefect]:
    if net.graph.number_of_edges() > max_cycles * 8:
        cycles = _sample_triangle_cycles(net.graph, max_cycles)
    else:
        cycles = nx.cycle_basis(net.graph)
    defects: list[HolonomyDefect] = []
    for cycle in cycles[:max_cycles]:
        hol = cycle_holonomy(net, cycle)
        if hol != net.group.identity:
            defects.append(
                HolonomyDefect(
                    cycle_key=tuple(sorted(cycle)),
                    ordered_cycle=tuple(cycle),
                    holonomy=hol,
                    holonomy_label=net.group.label(hol),
                    support_size=len(cycle),
                )
            )
    return defects


def _sample_triangle_cycles(graph: nx.Graph, max_cycles: int) -> list[list[int]]:
    cycles: list[list[int]] = []
    seen: set[tuple[int, int, int]] = set()
    for node in graph.nodes:
        neighbors = sorted(graph.neighbors(node))
        for idx, left in enumerate(neighbors):
            for right in neighbors[idx + 1 :]:
                if graph.has_edge(left, right):
                    key = tuple(sorted((node, left, right)))
                    if key in seen:
                        continue
                    seen.add(key)
                    cycles.append([node, left, right])
                    if len(cycles) >= max_cycles:
                        return cycles
    if cycles:
        return cycles
    return nx.cycle_basis(graph)[:max_cycles]
