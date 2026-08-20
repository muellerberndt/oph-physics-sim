"""Production icosahedral_tower carrier family: receipts, guards, configs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oph_fpe.core.array_geometry import (
    ICOSAHEDRAL_TOWER_FAMILY,
    array_screen_geometry_from_config,
    icosahedral_patch_degree_histogram,
    icosahedral_tower_carrier_verification,
    resolve_icosahedral_tower_rung,
)
from oph_fpe.core.icosahedral import (
    UnsupportedIcosahedralPatchCount,
    build_geodesic_icosahedral_tower,
    supported_icosahedral_count,
)
from oph_fpe.core.screen_ports import assign_echosahedral_ports
from oph_fpe.experiments import load_config
from oph_fpe.scale.array_screen import _knn_edges
from oph_fpe.scale.bw_array import _repairs_per_cycle_from_config, run_bw_array_config


TOWER_CONFIG_RUNGS = {
    "icosa_smoke.yml": (3, 1_280),
    "icosa_20k.yml": (5, 20_480),
    "icosa_82k.yml": (6, 81_920),
    "icosa_328k.yml": (7, 327_680),
    "icosa_1310k.yml": (8, 1_310_720),
}


def _tower_geometry(level: int):
    return array_screen_geometry_from_config(
        {"family": ICOSAHEDRAL_TOWER_FAMILY, "refinement_level": level},
        knn_builder=_knn_edges,
    )


def test_exact_rung_counts_per_level():
    for level in range(9):
        assert supported_icosahedral_count(level, "cells") == 20 * 4**level
        assert supported_icosahedral_count(level, "vertices") == 10 * 4**level + 2
    for level in range(4):
        geometry = _tower_geometry(level)
        assert geometry.patch_count == 20 * 4**level
        assert geometry.edge_count == 30 * 4**level
        assert geometry.report["actual_patch_count"] == 20 * 4**level
    for filename, (level, cells) in TOWER_CONFIG_RUNGS.items():
        assert supported_icosahedral_count(level, "cells") == cells, filename


def test_level_zero_structure_is_exactly_the_icosahedron():
    base = build_geodesic_icosahedral_tower(0).levels[0]
    assert base.vertex_count == 12
    assert base.edge_count == 30
    assert base.face_count == 20
    assert base.euler_characteristic == 2

    geometry = _tower_geometry(0)
    report = geometry.report
    assert geometry.patch_count == 20
    assert geometry.edge_count == 30
    assert report["BASE_CARRIER_IS_REGULAR_ICOSAHEDRON_RECEIPT"] is True
    assert report["carrier_semantics"]["base_complex_at_level_0"] == {
        "vertices": 12,
        "edges": 30,
        "faces": 20,
    }
    level_zero = report["levels"][0]
    assert level_zero["vertex_count"] == 12
    assert level_zero["cell_count"] == 20
    euler = (
        level_zero["vertex_count"]
        - 30 * 4 ** level_zero["refinement_level"]
        + level_zero["cell_count"]
    )
    assert euler == 2


def test_degree_histograms_are_exact_per_level():
    for level in range(4):
        cells = 20 * 4**level
        report = _tower_geometry(level).report
        assert report["carrier_degree_histogram"] == {"3": cells}
        assert report["CARRIER_ADJACENCY_DEGREE_HISTOGRAM_RECEIPT"] is True
        assert icosahedral_patch_degree_histogram(level, patch_basis="cells") == {
            3: cells
        }
        vertex_histogram = icosahedral_patch_degree_histogram(
            level, patch_basis="vertices"
        )
        if level == 0:
            assert vertex_histogram == {5: 12}
        else:
            assert vertex_histogram == {5: 12, 6: 10 * 4**level - 10}


def test_a5_and_tower_receipts_are_true_and_conflation_is_false():
    report = _tower_geometry(2).report
    assert report["TRUE_ICOSAHEDRAL_REFINEMENT_TOWER_RECEIPT"] is True
    assert report["A5_EQUIVARIANT_REFINEMENT_RECEIPT"] is True
    assert report["NESTED_ICOSAHEDRAL_LINEAGE_RECEIPT"] is True
    assert report["TWELVE_PERSISTENT_COMBINATORIAL_DEFECT_PORTS_RECEIPT"] is True
    assert report["TRUE_TOWER_CARRIER_VERIFICATION_RECEIPT"] is True
    assert report["CARRIER_SUPPORT_CONFLATION_PRESENT_IN_LEGACY_ENGINE"] is False
    assert report["PORT_SLOT_DEGREE_RECONCILIATION_RECEIPT"] is True
    # Unearned certificates stay false on the production family.
    assert report["PAPER_MULTIRESOLUTION_REGULATOR_CERTIFICATE"] is False
    assert report["CARRIER_TO_SUPPORT_REALIZATION_RECEIPT"] is False
    assert report["PHYSICAL_A5_PORT_EMERGENCE_RECEIPT"] is False


def test_two_builds_are_identical():
    first = _tower_geometry(2)
    build_geodesic_icosahedral_tower.cache_clear()
    second = _tower_geometry(2)
    assert np.array_equal(first.points, second.points)
    assert np.array_equal(first.edge_left, second.edge_left)
    assert np.array_equal(first.edge_right, second.edge_right)
    assert json.dumps(first.report, sort_keys=True) == json.dumps(
        second.report, sort_keys=True
    )


def test_port_slot_reconciliation_against_true_adjacency():
    geometry = _tower_geometry(2)
    port_map = assign_echosahedral_ports(
        geometry.edge_left,
        geometry.edge_right,
        geometry.patch_count,
        ports_per_patch=12,
        points=geometry.points,
    )
    assert port_map.overflow_count == 0
    assert np.all(port_map.node_degree == 3)
    routed = np.zeros((geometry.patch_count, 12), dtype=bool)
    routed[np.asarray(geometry.edge_left), np.asarray(port_map.left_port)] = True
    routed[np.asarray(geometry.edge_right), np.asarray(port_map.right_port)] = True
    # Every routed endpoint occupies a distinct local slot: three per patch.
    assert int(routed.sum()) == 2 * geometry.edge_count
    assert np.all(routed.sum(axis=1) == 3)
    payload = port_map.as_jsonable()
    assert payload["unused_port_slots"] == 9 * geometry.patch_count
    assert payload["degree_min"] == payload["degree_max"] == 3
    assert payload["routing_mode"] == "icosahedral_directional_assignment"
    assert payload["GEOMETRIC_LOCAL_PORT_ROUTING_RECEIPT"] is True
    assert payload["REFERENCE_SINGLETON_FEDERATION_SEWING_RECEIPT"] is True


def test_mutation_guards_fail_closed():
    geometry = _tower_geometry(2)
    left = np.array(geometry.edge_left)
    right = np.array(geometry.edge_right)

    baseline = icosahedral_tower_carrier_verification(
        geometry.points, left, right, refinement_level=2
    )
    assert baseline["TRUE_TOWER_CARRIER_VERIFICATION_RECEIPT"] is True

    order = np.random.default_rng(20260820).permutation(left.size)
    shuffled = icosahedral_tower_carrier_verification(
        geometry.points, left[order], right[order], refinement_level=2
    )
    assert shuffled["TRUE_TOWER_CARRIER_VERIFICATION_RECEIPT"] is False

    dropped = icosahedral_tower_carrier_verification(
        geometry.points, left[:-1], right[:-1], refinement_level=2
    )
    assert dropped["TRUE_TOWER_CARRIER_VERIFICATION_RECEIPT"] is False
    assert dropped["seam_count_exact"] is False

    rewired = left.copy()
    rewired[0] = (rewired[0] + 7) % geometry.patch_count
    rewired_report = icosahedral_tower_carrier_verification(
        geometry.points, rewired, right, refinement_level=2
    )
    assert rewired_report["TRUE_TOWER_CARRIER_VERIFICATION_RECEIPT"] is False
    assert rewired_report["degree_histogram_exact"] is False

    wrong_points = np.array(geometry.points)
    wrong_points[0] = wrong_points[1]
    moved = icosahedral_tower_carrier_verification(
        wrong_points, left, right, refinement_level=2
    )
    assert moved["TRUE_TOWER_CARRIER_VERIFICATION_RECEIPT"] is False


def test_rung_resolver_rejects_non_carrier_requests():
    with pytest.raises(ValueError, match="vertex"):
        resolve_icosahedral_tower_rung(
            {"family": ICOSAHEDRAL_TOWER_FAMILY, "patch_basis": "vertices", "refinement_level": 2}
        )
    with pytest.raises(UnsupportedIcosahedralPatchCount):
        resolve_icosahedral_tower_rung(
            {"family": ICOSAHEDRAL_TOWER_FAMILY, "patch_count": 4096}
        )
    with pytest.raises(ValueError, match="nominal_patch_count"):
        resolve_icosahedral_tower_rung(
            {"family": ICOSAHEDRAL_TOWER_FAMILY, "refinement_level": 5, "patch_count": 20000}
        )
    with pytest.raises(ValueError, match="exact"):
        resolve_icosahedral_tower_rung(
            {
                "family": ICOSAHEDRAL_TOWER_FAMILY,
                "patch_count": 20480,
                "patch_count_policy": "nearest",
            }
        )
    with pytest.raises(ValueError, match="refinement_level or an exact"):
        resolve_icosahedral_tower_rung({"family": ICOSAHEDRAL_TOWER_FAMILY})


def test_every_tower_config_validates():
    for filename, (level, cells) in TOWER_CONFIG_RUNGS.items():
        config = load_config(Path("configs") / filename)
        graph = config["graph"]
        assert graph["family"] == ICOSAHEDRAL_TOWER_FAMILY, filename
        rung = resolve_icosahedral_tower_rung(graph)
        assert rung["refinement_level"] == level, filename
        assert rung["patch_count"] == cells, filename
        assert rung["patch_count"] == int(graph["patch_count"]), filename
        assert rung["ICOSAHEDRAL_TOWER_RUNG_RESOLUTION_RECEIPT"] is True, filename
        assert config["screen"]["ports_per_patch"] == 12, filename
        repairs = _repairs_per_cycle_from_config(
            config["dynamics"],
            patch_count=rung["patch_count"],
            edge_count=rung["seam_count"],
        )
        assert repairs == rung["patch_count"], filename
        assert repairs <= rung["seam_count"], filename


def test_smoke_config_builds_the_declared_carrier():
    config = load_config(Path("configs") / "icosa_smoke.yml")
    geometry = array_screen_geometry_from_config(
        config["graph"], knn_builder=_knn_edges
    )
    assert geometry.patch_count == 1_280
    assert geometry.edge_count == 1_920
    assert geometry.report["TRUE_ICOSAHEDRAL_REFINEMENT_TOWER_RECEIPT"] is True
    assert geometry.report["carrier_degree_histogram"] == {"3": 1_280}


def test_legacy_family_still_works_and_stamps_false():
    geometry = array_screen_geometry_from_config(
        {"family": "fibonacci_sphere", "patch_count": 64, "neighbors": 4},
        knn_builder=_knn_edges,
    )
    report = geometry.report
    assert geometry.patch_count == 64
    assert report["geometry_family"] == "legacy_fibonacci_knn_control"
    assert report["TRUE_ICOSAHEDRAL_REFINEMENT_TOWER_RECEIPT"] is False
    assert report["A5_EQUIVARIANT_REFINEMENT_RECEIPT"] is False
    assert report["CARRIER_SUPPORT_CONFLATION_PRESENT_IN_LEGACY_ENGINE"] is True


def test_bw_array_engine_runs_on_the_tower_carrier(tmp_path: Path):
    config = load_config(Path("configs") / "icosa_smoke.yml")
    config = dict(config)
    config["run_id"] = "icosa_tower_engine_receipt_check"
    config["graph"] = dict(
        config["graph"], refinement_level=2, patch_count=320
    )
    config["dynamics"] = dict(config["dynamics"], cycles=6, repairs_per_cycle=480)
    config["bw"] = dict(config["bw"], cap_count=4, times=[0.025], n_jobs=2)
    config["cosmology"] = {
        "angular_power": {
            "ell_max": 8,
            "pair_samples": 2048,
            "controls": ["shuffled_field"],
        }
    }

    result = run_bw_array_config(config, tmp_path)
    run_path = Path(result["path"])
    geometry_report = json.loads(
        (run_path / "icosahedral_federation_geometry_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert geometry_report["geometry_family"] == ICOSAHEDRAL_TOWER_FAMILY
    assert geometry_report["actual_patch_count"] == 320
    assert geometry_report["TRUE_ICOSAHEDRAL_REFINEMENT_TOWER_RECEIPT"] is True
    assert geometry_report["A5_EQUIVARIANT_REFINEMENT_RECEIPT"] is True
    assert (
        geometry_report["CARRIER_SUPPORT_CONFLATION_PRESENT_IN_LEGACY_ENGINE"] is False
    )
    assert geometry_report["PORT_SLOT_DEGREE_RECONCILIATION_RECEIPT"] is True
    assert geometry_report["carrier_degree_histogram"] == {"3": 320}

    ports_report = json.loads(
        (run_path / "screen_ports.json").read_text(encoding="utf-8")
    )
    assert ports_report["overflow_count"] == 0
    assert ports_report["degree_min"] == ports_report["degree_max"] == 3
    assert ports_report["unused_port_slots"] == 9 * 320

    patch_state_report = json.loads(
        (run_path / "echosahedral_patch_state_report.json").read_text(encoding="utf-8")
    )
    assert (
        patch_state_report["ECHOSAHEDRAL_PATCH_STATE_INSTANTIATION_RECEIPT"] is True
    )
    assert (
        patch_state_report["unrouted_exposed_or_reserved_port_slot_count"]
        == 9 * 320
    )
