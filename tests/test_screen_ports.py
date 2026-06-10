from __future__ import annotations

import numpy as np

from oph_fpe.core.screen_ports import assign_echosahedral_ports, echosahedral_port_names


def test_echosahedral_port_names_are_explicit():
    assert echosahedral_port_names(12) == [f"P{index}" for index in range(12)]


def test_assign_echosahedral_ports_reports_overflow():
    left = np.array([0, 0, 0, 0])
    right = np.array([1, 2, 3, 4])
    port_map = assign_echosahedral_ports(left, right, patch_count=5, ports_per_patch=3)
    report = port_map.as_jsonable(sample_edges=2)

    assert port_map.left_port.tolist() == [0, 1, 2, 0]
    assert port_map.overflow_count == 1
    assert report["port_names"] == ["P0", "P1", "P2"]
    assert report["sample_edge_ports"][0]["left_port"] == "P0"


def test_assign_echosahedral_ports_matches_sequential_endpoint_order():
    left = np.array([0, 0, 1, 2, 0, 2, 1], dtype=np.int64)
    right = np.array([1, 2, 2, 3, 3, 0, 3], dtype=np.int64)
    ports = 2
    port_map = assign_echosahedral_ports(left, right, patch_count=4, ports_per_patch=ports)

    counters = np.zeros(4, dtype=np.int64)
    expected_left: list[int] = []
    expected_right: list[int] = []
    overflow = 0
    for a, b in zip(left, right, strict=False):
        left_count = counters[int(a)]
        right_count = counters[int(b)]
        overflow += int(left_count >= ports) + int(right_count >= ports)
        expected_left.append(int(left_count % ports))
        expected_right.append(int(right_count % ports))
        counters[int(a)] += 1
        counters[int(b)] += 1

    assert port_map.left_port.tolist() == expected_left
    assert port_map.right_port.tolist() == expected_right
    assert port_map.overflow_count == overflow
    assert port_map.node_degree.tolist() == [4, 3, 4, 3]
