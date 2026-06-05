from __future__ import annotations

import networkx as nx
import numpy as np


def laplacian_embedding(graph: nx.Graph, dimensions: int = 3) -> dict[int, list[float]]:
    nodes = list(graph.nodes)
    if not nodes:
        return {}
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, dtype=float)
    degree = np.diag(np.sum(adjacency, axis=1))
    laplacian = degree - adjacency
    values, vectors = np.linalg.eigh(laplacian)
    order = np.argsort(values)
    coords = vectors[:, order[1 : dimensions + 1]]
    if coords.shape[1] < dimensions:
        coords = np.pad(coords, ((0, 0), (0, dimensions - coords.shape[1])))
    return {
        int(node): [float(value) for value in coords[row, :dimensions]]
        for row, node in enumerate(nodes)
    }
