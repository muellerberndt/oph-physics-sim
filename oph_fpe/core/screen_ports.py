from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def echosahedral_port_names(ports_per_patch: int = 12) -> list[str]:
    return [f"P{index}" for index in range(int(ports_per_patch))]


@dataclass(frozen=True)
class ScreenPortMap:
    left_port: np.ndarray
    right_port: np.ndarray
    ports_per_patch: int
    overflow_count: int
    node_degree: np.ndarray

    def as_jsonable(self, *, sample_edges: int = 64) -> dict[str, Any]:
        sample_count = min(int(sample_edges), int(self.left_port.size))
        port_names = echosahedral_port_names(self.ports_per_patch)
        return {
            "mode": "explicit_named_echosahedral_ports",
            "ports_per_patch": int(self.ports_per_patch),
            "port_names": port_names,
            "edge_count": int(self.left_port.size),
            "overflow_count": int(self.overflow_count),
            "overflow_fraction": float(self.overflow_count / max(1, int(self.left_port.size) * 2)),
            "degree_min": int(np.min(self.node_degree)) if self.node_degree.size else 0,
            "degree_mean": float(np.mean(self.node_degree)) if self.node_degree.size else 0.0,
            "degree_max": int(np.max(self.node_degree)) if self.node_degree.size else 0,
            "sample_edge_ports": [
                {
                    "edge_index": index,
                    "left_port": port_names[int(self.left_port[index])],
                    "right_port": port_names[int(self.right_port[index])],
                }
                for index in range(sample_count)
            ],
            "claim_boundary": (
                "explicit finite P0..P11 port assignment for the screen regulator; overflow "
                "marks edges that exceed the local echosahedral port budget"
            ),
        }


def assign_echosahedral_ports(
    left: np.ndarray,
    right: np.ndarray,
    patch_count: int,
    *,
    ports_per_patch: int = 12,
) -> ScreenPortMap:
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    ports = max(1, int(ports_per_patch))
    node_degree = np.bincount(np.concatenate([left, right]), minlength=int(patch_count))
    endpoint_nodes = np.empty(left.size * 2, dtype=np.int64)
    endpoint_nodes[0::2] = left
    endpoint_nodes[1::2] = right
    if endpoint_nodes.size:
        order = np.argsort(endpoint_nodes, kind="stable")
        sorted_nodes = endpoint_nodes[order]
        group_start = np.empty(sorted_nodes.size, dtype=bool)
        group_start[0] = True
        group_start[1:] = sorted_nodes[1:] != sorted_nodes[:-1]
        starts = np.flatnonzero(group_start)
        counts = np.diff(np.append(starts, sorted_nodes.size))
        local_rank_sorted = np.arange(sorted_nodes.size, dtype=np.int64) - np.repeat(starts, counts)
        local_rank = np.empty_like(local_rank_sorted)
        local_rank[order] = local_rank_sorted
        overflow = int(np.sum(local_rank >= ports))
        left_port = (local_rank[0::2] % ports).astype(np.int16)
        right_port = (local_rank[1::2] % ports).astype(np.int16)
    else:
        left_port = np.zeros(0, dtype=np.int16)
        right_port = np.zeros(0, dtype=np.int16)
        overflow = 0
    return ScreenPortMap(
        left_port=left_port,
        right_port=right_port,
        ports_per_patch=ports,
        overflow_count=int(overflow),
        node_degree=node_degree,
    )
