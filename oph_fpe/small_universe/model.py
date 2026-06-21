from __future__ import annotations

import hashlib
import itertools
import json
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Any

import networkx as nx
import numpy as np


State = tuple[int, ...]
Edge = tuple[int, int]


@dataclass(frozen=True)
class SmallUniverse:
    graph: nx.Graph
    root: int
    parent: dict[int, int]
    depth: dict[int, int]
    target: State
    offsets: dict[Edge, int]
    weights: dict[Edge, int]
    tree_edges: frozenset[Edge]
    flipped_edge: Edge | None = None

    @property
    def patch_count(self) -> int:
        return int(self.graph.number_of_nodes())

    @property
    def edge_count(self) -> int:
        return int(self.graph.number_of_edges())


def edge_key(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)


def stable_hash(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default).encode()
    return hashlib.sha256(payload).hexdigest()


def build_icosa12_universe(seed: int, *, frustrate: bool = False) -> SmallUniverse:
    graph = nx.icosahedral_graph()
    root = 0
    parent: dict[int, int] = {}
    depth: dict[int, int] = {root: 0}
    queue: deque[int] = deque([root])
    while queue:
        u = queue.popleft()
        for v in sorted(graph.neighbors(u)):
            if v in depth:
                continue
            depth[v] = depth[u] + 1
            parent[v] = u
            queue.append(v)

    rng = np.random.default_rng(seed)
    target = tuple([0] + [int(x) for x in rng.integers(0, 2, size=graph.number_of_nodes() - 1)])
    offsets = {edge_key(u, v): target[u] ^ target[v] for u, v in graph.edges()}
    tree_edges = frozenset(edge_key(p, v) for v, p in parent.items())

    flipped_edge: Edge | None = None
    if frustrate:
        non_tree_edges = sorted(set(offsets) - set(tree_edges))
        flipped_edge = non_tree_edges[seed % len(non_tree_edges)]
        offsets[flipped_edge] ^= 1

    max_depth = max(depth.values())
    base = 64
    weights: dict[Edge, int] = {}
    for edge in offsets:
        if edge in tree_edges:
            child = edge[0] if parent.get(edge[0]) == edge[1] else edge[1]
            weights[edge] = base ** (max_depth - depth[child] + 1)
        else:
            weights[edge] = 1

    return SmallUniverse(
        graph=graph,
        root=root,
        parent=parent,
        depth=depth,
        target=target,
        offsets=offsets,
        weights=weights,
        tree_edges=tree_edges,
        flipped_edge=flipped_edge,
    )


def mismatch(universe: SmallUniverse, state: State, edge: Edge) -> int:
    a, b = edge
    return int((state[a] ^ state[b]) != universe.offsets[edge])


def phi(universe: SmallUniverse, state: State) -> int:
    return sum(universe.weights[edge] * mismatch(universe, state, edge) for edge in universe.offsets)


def globally_consistent(universe: SmallUniverse, state: State) -> bool:
    return all(mismatch(universe, state, edge) == 0 for edge in universe.offsets)


def all_root_pinned_states(universe: SmallUniverse) -> Iterable[State]:
    for tail in itertools.product((0, 1), repeat=universe.patch_count - 1):
        yield (0,) + tail


def cycle_holonomies(universe: SmallUniverse) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cycle in nx.cycle_basis(universe.graph, root=universe.root):
        holonomy = 0
        edges: list[Edge] = []
        for a, b in zip(cycle, cycle[1:] + cycle[:1]):
            edge = edge_key(a, b)
            edges.append(edge)
            holonomy ^= universe.offsets[edge]
        rows.append(
            {
                "cycle": [int(x) for x in cycle],
                "edges": [[int(a), int(b)] for a, b in edges],
                "holonomy_z2": int(holonomy),
            }
        )
    return rows


def manifest(universe: SmallUniverse, *, repair_backend: str, seed: int) -> dict[str, Any]:
    out = {
        "schema": "small_oph_universe_manifest_v1",
        "seed": int(seed),
        "graph": "networkx.icosahedral_graph",
        "patch_count": universe.patch_count,
        "edge_count": universe.edge_count,
        "visible_interface_alphabet": "Z2",
        "root_boundary": universe.root,
        "repair_backend": repair_backend,
        "parent": {str(k): int(v) for k, v in sorted(universe.parent.items())},
        "depth": {str(k): int(v) for k, v in sorted(universe.depth.items())},
        "target": list(universe.target),
        "offsets": _edge_map(universe.offsets),
        "weights": _edge_map(universe.weights),
        "tree_edges": [[int(a), int(b)] for a, b in sorted(universe.tree_edges)],
        "flipped_edge": list(universe.flipped_edge) if universe.flipped_edge is not None else None,
        "claim_boundary": (
            "Fixed-cutoff 12-patch Z2 finite-consensus calibration universe. "
            "This manifest does not encode modular flow, H3 coordinates, particles, or CMB data."
        ),
    }
    out["model_sha256"] = stable_hash(out)
    return out


def state_id(state: State) -> str:
    return "".join(str(int(bit)) for bit in state)


def _edge_map(values: dict[Edge, int]) -> dict[str, int]:
    return {f"{a}-{b}": int(value) for (a, b), value in sorted(values.items())}


def _json_default(value: object) -> object:
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot JSON encode {type(value).__name__}")
