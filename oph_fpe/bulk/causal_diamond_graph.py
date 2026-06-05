from __future__ import annotations

from typing import Any

import numpy as np

from oph_fpe.claims import DEMO, with_claim_metadata


def causal_diamond_graph_report(
    times: np.ndarray,
    causal_edges: np.ndarray,
    *,
    source: int,
    target: int,
) -> dict[str, Any]:
    time_values = np.asarray(times, dtype=float)
    edges = np.asarray(causal_edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("causal_edges must have shape (edge_count, 2)")
    src = int(source)
    dst = int(target)
    future = _reachable(src, edges, forward=True)
    past = _reachable(dst, edges, forward=False)
    diamond = sorted(future & past)
    time_ordered = all(time_values[left] <= time_values[right] + 1e-12 for left, right in edges)
    report = {
        "mode": "finite_causal_diamond_graph",
        "CAUSAL_DIAMOND_GRAPH_RECEIPT": bool(diamond) and time_ordered,
        "receipt": bool(diamond) and time_ordered,
        "source": src,
        "target": dst,
        "node_count": int(time_values.shape[0]),
        "edge_count": int(edges.shape[0]),
        "diamond_node_count": len(diamond),
        "diamond_nodes": diamond,
        "time_ordered_edges": bool(time_ordered),
        "claim_boundary": (
            "finite causal-diamond graph diagnostic for support-visible event ordering; "
            "not a continuum Lorentzian manifold theorem"
        ),
    }
    return with_claim_metadata(report, claim_level=DEMO, receipt="CAUSAL_DIAMOND_GRAPH_RECEIPT")


def _reachable(start: int, edges: np.ndarray, *, forward: bool) -> set[int]:
    adjacency: dict[int, list[int]] = {}
    for left, right in edges:
        a, b = (int(left), int(right)) if forward else (int(right), int(left))
        adjacency.setdefault(a, []).append(b)
    seen = {int(start)}
    stack = [int(start)]
    while stack:
        node = stack.pop()
        for next_node in adjacency.get(node, []):
            if next_node not in seen:
                seen.add(next_node)
                stack.append(next_node)
    return seen
