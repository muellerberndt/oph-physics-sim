from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

from oph_fpe.bulk.h3_worldline_stitch import (
    H3_CERTIFICATE_INCOMPLETE,
    H3_STITCH_AMBIGUOUS,
    H3_STITCH_CERTIFIED,
    H3_STITCH_REJECTED,
    H3_PRIMITIVE_SCHEMA,
    h3_distance,
    h3_point_from_spatial_point,
    h3_worldline_stitch_primitive_hash,
    h3_worldline_stitch_certificate_report,
    lorentz_matrix_residual,
    write_h3_worldline_stitch_certificate_report,
)
from oph_fpe.evidence.hashes import stable_json_hash
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
    assert report["primitive_provenance_verified"] is True
    assert report["assignment_replay"]["record_id_independence_recomputed"] is True
    assert report["certified_assignment_gap"] > 1.0
    assert relabeled_report["terminal_status"] == H3_STITCH_CERTIFIED
    assert relabeled_report["certified_assignment_gap"] == report["certified_assignment_gap"]
    assert relabeled_report["metric_rows"] == report["metric_rows"]


def test_h3_worldline_stitch_certificate_returns_ambiguous_for_equal_costs() -> None:
    artifact = _valid_artifact()
    same = h3_point_from_spatial_point([0.2, 0.0, 0.0]).tolist()
    for edge in artifact["candidate_edges"]:
        edge["predicted_h3_point"] = same
        edge["observed_h3_point"] = same
        edge["h3_geodesic_cost"] = 0.0
    artifact["assignment"]["proposed_cost_upper"] = 0.0
    artifact["assignment"]["runner_up_cost_lower"] = 0.0
    artifact["assignment"]["multiple_optima"] = True
    _rebind(artifact)

    report = h3_worldline_stitch_certificate_report(artifact)

    assert report["terminal_status"] == H3_STITCH_AMBIGUOUS
    assert "assignment_cost_gap_nonpositive" in report["blockers"]
    assert report["h3_worldline_stitch_certificate_receipt"] is False


def test_h3_worldline_stitch_certificate_rejects_id_based_or_euclidean_matching() -> None:
    artifact = _valid_artifact()
    artifact["assignment"]["uses_record_ids"] = True
    artifact["candidate_edges"][0]["metric"] = "euclidean_h3SpatialPoint"
    _rebind(artifact)

    report = h3_worldline_stitch_certificate_report(artifact)

    assert report["terminal_status"] == H3_STITCH_REJECTED
    assert "record_ids_used_in_admissibility_or_cost" in report["blockers"]
    assert "candidate_edge_0_uses_euclidean_h3_distance" in report["blockers"]


def test_h3_worldline_stitch_rejects_assertions_without_primitive_producer_evidence() -> None:
    artifact = _valid_artifact()
    artifact.pop("primitive_provenance")
    artifact["candidate_edges"] = []
    artifact["h3_metric_receipt"] = True

    report = h3_worldline_stitch_certificate_report(artifact)

    assert report["terminal_status"] == H3_CERTIFICATE_INCOMPLETE
    assert report["h3_worldline_stitch_certificate_receipt"] is False
    assert "h3_primitive_producer_provenance_missing" in report["blockers"]
    assert "candidate_edge_h3_metric_primitives_missing" in report["blockers"]


def test_h3_worldline_stitch_requires_exact_booleans() -> None:
    artifact = _valid_artifact()
    artifact["assignment"]["complete_graph"] = "true"
    artifact["crossing"]["transverse"] = 1
    _rebind(artifact)

    report = h3_worldline_stitch_certificate_report(artifact)

    assert report["terminal_status"] == H3_STITCH_REJECTED
    assert "complete_graph_not_exact_boolean" in report["blockers"]
    assert "transverse_not_exact_boolean" in report["blockers"]


def test_h3_worldline_stitch_recomputes_complete_graph_and_assignment() -> None:
    artifact = _valid_artifact()
    artifact["candidate_edges"].pop()
    _rebind(artifact)

    report = h3_worldline_stitch_certificate_report(artifact)

    assert report["terminal_status"] == H3_STITCH_REJECTED
    assert "candidate_graph_not_complete_bipartite_from_primitives" in report["blockers"]


def test_h3_worldline_stitch_rejects_forged_assignment_bounds() -> None:
    artifact = _valid_artifact()
    artifact["assignment"]["proposed_cost_upper"] = 0.0
    artifact["assignment"]["runner_up_cost_lower"] = 100.0
    _rebind(artifact)

    report = h3_worldline_stitch_certificate_report(artifact)

    assert report["terminal_status"] == H3_STITCH_REJECTED
    assert "assignment_winner_cost_upper_below_recomputed_winner" in report["blockers"]
    assert "assignment_runner_cost_lower_above_recomputed_runner_up" in report["blockers"]


def test_h3_worldline_stitch_rejects_unbound_primitive_or_source_mutation() -> None:
    artifact = _valid_artifact()
    artifact["candidate_edges"][0]["observed_h3_point"] = h3_point_from_spatial_point(
        [0.4, 0.0, 0.0]
    ).tolist()
    artifact["primitive_provenance"]["source_manifest"]["run_id"] = "mutated"

    report = h3_worldline_stitch_certificate_report(artifact)

    assert report["terminal_status"] == H3_STITCH_REJECTED
    assert "h3_primitive_payload_hash_mismatch" in report["blockers"]
    assert "h3_primitive_source_hash_mismatch" in report["blockers"]


def test_h3_worldline_stitch_rejects_empty_selected_continuation() -> None:
    artifact = _valid_artifact()
    artifact["assignment"]["selected_pairs"] = []
    _rebind(artifact)

    report = h3_worldline_stitch_certificate_report(artifact)

    assert report["h3_worldline_stitch_certificate_receipt"] is False
    assert "selected_assignment_pair_primitives_missing" in report["blockers"]


def test_h3_worldline_stitch_rejects_nonfinite_h3_primitives() -> None:
    artifact = _valid_artifact()
    artifact["candidate_edges"][0]["predicted_h3_point"][0] = float("nan")
    _rebind(artifact)

    report = h3_worldline_stitch_certificate_report(artifact)

    assert report["h3_worldline_stitch_certificate_receipt"] is False
    assert "candidate_edge_0_h3_distance_invalid" in report["blockers"]


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
    left_points = {
        "l0": h3_point_from_spatial_point([0.10, 0.0, 0.0]).tolist(),
        "l1": h3_point_from_spatial_point([1.00, 0.0, 0.0]).tolist(),
    }
    right_points = {
        "r0": h3_point_from_spatial_point([0.12, 0.0, 0.0]).tolist(),
        "r1": h3_point_from_spatial_point([0.98, 0.0, 0.0]).tolist(),
    }
    edge_rows = []
    for left, predicted in left_points.items():
        for right, observed in right_points.items():
            edge_rows.append(
                {
                    "left": left,
                    "right": right,
                    "left_shard_id": "left_shard",
                    "right_shard_id": "right_shard",
                    "record_id": f"{left}_record",
                    "right_record_id": f"{right}_record",
                    "metric": "h3_hyperboloid_geodesic",
                    "predicted_h3_point": predicted,
                    "observed_h3_point": observed,
                    "h3_geodesic_cost": h3_distance(predicted, observed),
                }
            )
    winner = h3_distance(left_points["l0"], right_points["r0"]) + h3_distance(
        left_points["l1"], right_points["r1"]
    )
    runner = h3_distance(left_points["l0"], right_points["r1"]) + h3_distance(
        left_points["l1"], right_points["r0"]
    )
    artifact = {
        "atlas": {
            "model": "H3 hyperboloid in R^{1,3}",
            "R_H": 1.0,
            "points": [
                {"id": key, "X": point}
                for key, point in {**left_points, **right_points}.items()
            ],
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
            "selected_pairs": [
                {"left": "l0", "right": "r0"},
                {"left": "l1", "right": "r1"},
            ],
            "appearance_penalties": [
                {"right": "r0", "cost": 2.0},
                {"right": "r1", "cost": 2.0},
            ],
            "disappearance_penalties": [
                {"left": "l0", "cost": 2.0},
                {"left": "l1", "cost": 2.0},
            ],
            "proposed_cost_upper": winner,
            "runner_up_cost_lower": runner,
            "required_gap": 0.05,
        },
        "refinement": {
            "coarse_fine_pair": True,
            "Q_sr_present": True,
            "contracted_graph_isomorphic": True,
            "eta_sr": 0.02,
            "coarse_gap": runner - winner,
        },
        "interaction": {
            "free_propagation_slab": True,
            "interaction_required": False,
        },
        "candidate_edges": edge_rows,
    }
    source_manifest = {
        "run_id": "test_measured_interface",
        "left_shard_id": "left_shard",
        "right_shard_id": "right_shard",
    }
    configuration = {"curvature_radius": 1.0, "assignment_replay_limit": 7}
    artifact["primitive_provenance"] = {
        "schema": H3_PRIMITIVE_SCHEMA,
        "source_kind": "measured_cross_shard_interface",
        "producer": "external_h3_interface_measurement_fixture",
        "producer_version": "1",
        "source_manifest": source_manifest,
        "configuration": configuration,
        "source_hash": stable_json_hash(source_manifest),
        "configuration_hash": stable_json_hash(configuration),
    }
    _rebind(artifact)
    return artifact


def _rebind(artifact: dict) -> None:
    provenance = artifact["primitive_provenance"]
    provenance["source_hash"] = stable_json_hash(provenance["source_manifest"])
    provenance["configuration_hash"] = stable_json_hash(provenance["configuration"])
    provenance["primitive_hash"] = h3_worldline_stitch_primitive_hash(artifact)
    artifact["hashes"] = {
        "source_hash": provenance["source_hash"],
        "configuration_hash": provenance["configuration_hash"],
    }
