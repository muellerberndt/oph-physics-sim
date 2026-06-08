from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ProvenanceNode:
    node_id: str
    t: int
    kind: Literal["source", "record", "memory", "environment", "hidden"]
    capacity_bits: float
    payload_bits: float
    visible: bool


@dataclass(frozen=True)
class ProvenanceEdge:
    src: str
    dst: str
    capacity_bits: float
    entropy_cost_bits: float = 0.0


@dataclass
class ProvenanceDAG:
    nodes: dict[str, ProvenanceNode] = field(default_factory=dict)
    edges: list[ProvenanceEdge] = field(default_factory=list)

    def ancestors_of(self, node_id: str) -> set[str]:
        reverse: dict[str, list[str]] = {}
        for edge in self.edges:
            reverse.setdefault(edge.dst, []).append(edge.src)
        seen: set[str] = set()
        queue = deque(reverse.get(node_id, []))
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            queue.extend(reverse.get(current, []))
        return seen

    def cut_capacity_bits(self, source_ids: set[str], target_ids: set[str]) -> float:
        source_ids = set(source_ids)
        target_ids = set(target_ids)
        if not source_ids or not target_ids:
            return 0.0
        capacity: dict[tuple[str, str], float] = {}
        nodes = set(self.nodes)
        super_source = "__source__"
        super_sink = "__sink__"
        nodes.update({super_source, super_sink})
        for edge in self.edges:
            capacity[(edge.src, edge.dst)] = capacity.get((edge.src, edge.dst), 0.0) + max(0.0, float(edge.capacity_bits))
            capacity.setdefault((edge.dst, edge.src), 0.0)
        big = sum(max(0.0, float(edge.capacity_bits)) for edge in self.edges) + 1.0
        for src in source_ids:
            capacity[(super_source, src)] = big
            capacity.setdefault((src, super_source), 0.0)
        for dst in target_ids:
            capacity[(dst, super_sink)] = big
            capacity.setdefault((super_sink, dst), 0.0)
        return _max_flow(capacity, super_source, super_sink)


def provenance_bottleneck_ok(
    dag: ProvenanceDAG,
    source_ids: set[str],
    target_ids: set[str],
    required_payload_bits: float,
) -> bool:
    return dag.cut_capacity_bits(source_ids, target_ids) + 1e-9 >= float(required_payload_bits)


def _max_flow(capacity: dict[tuple[str, str], float], source: str, sink: str) -> float:
    residual = dict(capacity)
    adjacency: dict[str, set[str]] = {}
    for (u, v), cap in residual.items():
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)
        residual.setdefault((v, u), 0.0)
    flow = 0.0
    while True:
        parent: dict[str, str | None] = {source: None}
        queue = deque([source])
        while queue and sink not in parent:
            u = queue.popleft()
            for v in sorted(adjacency.get(u, ())):
                if v not in parent and residual.get((u, v), 0.0) > 1e-12:
                    parent[v] = u
                    queue.append(v)
        if sink not in parent:
            return flow
        bottleneck = float("inf")
        v = sink
        while parent[v] is not None:
            u = parent[v]
            bottleneck = min(bottleneck, residual[(u, v)])
            v = u
        v = sink
        while parent[v] is not None:
            u = parent[v]
            residual[(u, v)] -= bottleneck
            residual[(v, u)] = residual.get((v, u), 0.0) + bottleneck
            v = u
        flow += bottleneck

