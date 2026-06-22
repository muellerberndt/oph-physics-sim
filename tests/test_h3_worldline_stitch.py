from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

from oph_fpe.bulk.h3_worldline_stitch import (
    H3_STITCH_AMBIGUOUS,
    H3_STITCH_CERTIFIED,
    H3_STITCH_REJECTED,
    h3_distance,
    h3_point_from_spatial_point,
    h3_worldline_stitch_certificate_report,
    lorentz_matrix_residual,
    write_h3_worldline_stitch_certificate_report,
)
from oph_fpe.measurement_pack import export_measurement_pack


def test_h3_distance_is_lorentz_natural_under_boost() -> None:
    left = h3_point_from_spatial_point([0.1, 0.0, 0.0])
    right = h3_point_from_spatial_point([0.7, 0.2, 0.0])
    rapidity = 0.4
    boost = np.array(
        [
            [math.cosh(rapidity), math.sinh(rapidity), 0.0, 0.0],
            [math.sinh(rapidity), math.cosh(rapidity), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    assert lorentz_matrix_residual(boost) < 1.0e-12
    assert h3_distance(left, right) == pytest_approx(h3_distance(boost @ left, boost @ right))


def test_h3_worldline_stitch_certificate_passes_and_ignores_record_ids() -> None:
    artifact = _valid_artifact()
    report = h3_worldline_stitch_certificate_report(artifact)

    relabeled = copy.deepcopy(artifact)
    relabeled["candidate_edges"][0]["record_id"] = "changed_left"
    relabeled["candidate_edges"][0]["right_record_id"] = "changed_right"
    relabeled_report = h3_worldline_stitch_certificate_report(relabeled)

    assert report["terminal_status"] == H3_STITCH_CERTIFIED
    assert report["round1_terminal_status"] == "H3_WORLDLINE_STITCH_CERTIFIED"
    assert report["H3_WORLDLINE_STITCH_CERTIFIED"] is True
    assert report["h3_worldline_stitch_certificate_receipt"] is True
    assert report["certified_assignment_gap"] == pytest_approx(0.3)
    assert relabeled_report["terminal_status"] == H3_STITCH_CERTIFIED
    assert relabeled_report["certified_assignment_gap"] == report["certified_assignment_gap"]
    assert relabeled_report["metric_rows"] == report["metric_rows"]


def test_h3_worldline_stitch_certificate_returns_ambiguous_for_equal_costs() -> None:
    artifact = _valid_artifact()
    artifact["assignment"]["runner_up_cost_lower"] = artifact["assignment"]["proposed_cost_upper"]

    report = h3_worldline_stitch_certificate_report(artifact)

    assert report["terminal_status"] == H3_STITCH_AMBIGUOUS
    assert "assignment_cost_gap_nonpositive" in report["blockers"]
    assert report["h3_worldline_stitch_certificate_receipt"] is False


def test_h3_worldline_stitch_certificate_rejects_id_based_or_euclidean_matching() -> None:
    artifact = _valid_artifact()
    artifact["assignment"]["uses_record_ids"] = True
    artifact["candidate_edges"][0]["metric"] = "euclidean_h3SpatialPoint"

    report = h3_worldline_stitch_certificate_report(artifact)

    assert report["terminal_status"] == H3_STITCH_REJECTED
    assert "record_ids_used_in_admissibility_or_cost" in report["blockers"]
    assert "candidate_edge_0_uses_euclidean_h3_distance" in report["blockers"]


def test_h3_worldline_stitch_cli_writer_and_measurement_pack(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    source = run / "h3_worldline_stitch_source.json"
    out = run / "h3_worldline_stitch_certificate_report.json"
    source.write_text(json.dumps(_valid_artifact()), encoding="utf-8")

    report = write_h3_worldline_stitch_certificate_report(source, out)
    pack = export_measurement_pack([run], tmp_path / "pack")

    assert report["terminal_status"] == H3_STITCH_CERTIFIED
    assert (tmp_path / "pack" / "h3_worldline_stitch_certificate_report.json").exists()
    assert pack["claims"]["h3_worldline_stitch_status"] == H3_STITCH_CERTIFIED
    assert pack["claims"]["h3_worldline_stitch_certificate_receipt"] is True


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value, abs=1.0e-12)


def _valid_artifact() -> dict:
    identity = np.eye(4).tolist()
    predicted = h3_point_from_spatial_point([0.12, 0.0, 0.0]).tolist()
    observed = h3_point_from_spatial_point([0.14, 0.0, 0.0]).tolist()
    return {
        "atlas": {
            "model": "H3 hyperboloid in R^{1,3}",
            "R_H": 1.0,
            "points": [{"id": "p0", "X": predicted}, {"id": "p1", "X": observed}],
            "chart_transitions": [{"id": "id", "matrix": identity}],
        },
        "clock": {
            "common_time_line": True,
            "orientation_preserving": True,
            "max_error": 0.01,
            "time_order_margin": 0.08,
        },
        "extraction": {
            "detector_hash": "sha256:test",
            "thresholds_frozen": True,
            "chart_naturality_residual": 0.0,
        },
        "descent": {
            "overlap_graph_present": True,
            "same_component_join_margin": 0.2,
            "distinct_component_separation_margin": 0.25,
            "support_error": 0.01,
            "triple_overlap_cocycle_residual": 0.0,
        },
        "crossing": {
            "real_interface_contact": True,
            "transverse": True,
            "normal_velocity_lower_bound": 0.2,
            "signed_distance_margin": 0.1,
        },
        "transport": {
            "common_chart": True,
            "sector_continuity": True,
            "gauge_transport_continuity": True,
            "holonomy_compared_covariantly": True,
            "connector_declared": True,
            "transport_residual": 0.0,
        },
        "assignment": {
            "complete_graph": True,
            "one_to_one": True,
            "appearance_disappearance_penalties": True,
            "uses_record_ids": False,
            "distinct_shard_ids": ["left_shard", "right_shard"],
            "proposed_cost_upper": 0.1,
            "runner_up_cost_lower": 0.4,
            "required_gap": 0.05,
        },
        "refinement": {
            "coarse_fine_pair": True,
            "Q_sr_present": True,
            "contracted_graph_isomorphic": True,
            "eta_sr": 0.02,
            "coarse_gap": 0.1,
        },
        "interaction": {
            "free_propagation_slab": True,
            "interaction_required": False,
        },
        "candidate_edges": [
            {
                "left": "l0",
                "right": "r0",
                "record_id": "left_record",
                "right_record_id": "right_record",
                "metric": "h3_hyperboloid_geodesic",
                "predicted_h3_point": predicted,
                "observed_h3_point": observed,
            }
        ],
        "hashes": {"source_hash": "sha256:source", "configuration_hash": "sha256:config"},
    }
