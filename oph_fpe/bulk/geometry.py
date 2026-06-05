from __future__ import annotations

import networkx as nx
import numpy as np

from oph_fpe.core.patchnet import PatchNet


def graph_distance_matrix(graph: nx.Graph) -> tuple[list[int], np.ndarray]:
    nodes = list(graph.nodes)
    index = {node: pos for pos, node in enumerate(nodes)}
    distances = np.full((len(nodes), len(nodes)), np.inf, dtype=float)
    for node in nodes:
        distances[index[node], index[node]] = 0.0
        lengths = nx.single_source_shortest_path_length(graph, node)
        for other, distance in lengths.items():
            distances[index[node], index[other]] = float(distance)
    return nodes, distances


def record_feature_matrix(net: PatchNet) -> tuple[list[int], np.ndarray]:
    nodes = list(net.graph.nodes)
    max_degree = max((net.graph.degree[node] for node in nodes), default=0)
    width = 3 + max_degree
    features = np.zeros((len(nodes), width), dtype=float)
    element_index = {element: idx for idx, element in enumerate(net.group.elements)}
    for row, node in enumerate(nodes):
        state = net.states[node]
        features[row, 0] = state.hidden
        features[row, 1] = state.scalar
        features[row, 2] = state.stable_count
        for col, neighbor in enumerate(sorted(state.ports)[:max_degree], start=3):
            features[row, col] = element_index[state.ports[neighbor]]
    return nodes, features


def feature_distance_matrix(features: np.ndarray) -> np.ndarray:
    diff = features[:, None, :] - features[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))
