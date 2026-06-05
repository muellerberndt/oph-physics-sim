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
