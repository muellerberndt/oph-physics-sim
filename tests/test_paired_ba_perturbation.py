from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oph_fpe.bulk.cap_geometry import RoundCap
from oph_fpe.bulk.neutral_bulk import overlap_native_neutral_control_report
from oph_fpe.bulk.modular_response_kernel import _simulate_cap_collar_perturb_resettle
from oph_fpe.core.graph import fibonacci_sphere_points
from oph_fpe.cosmology.comparable_data import comparable_data_report
from oph_fpe.cosmology.paired_ba_perturbation import (
    paired_perturb_resettle_b_a_report,
    write_paired_perturb_resettle_b_a_report,
)
from oph_fpe.gauge.covariant_overlap import transform_local_frames
from oph_fpe.observers.subjective import observer_view_rows
from oph_fpe.scale.array_screen import _knn_edges, _node_signature


def _toy_state(patch_count: int = 96):
    points = fibonacci_sphere_points(patch_count)
    left, right = _knn_edges(points, 4)
    group_order = 6
    port_left = (left % group_order).astype(np.int16)
    port_right = (right % group_order).astype(np.int16)
    signature = _node_signature(port_left, port_right, left, right, patch_count)
    degree = np.bincount(np.concatenate([left, right]), minlength=patch_count).astype(float)
    raw_fields = {
        "record_signature": signature.astype(float),
        "stable_count": np.arange(patch_count, dtype=float) % 8.0,
        "committed_mask": np.ones(patch_count, dtype=float),
        "repair_load": np.zeros(patch_count, dtype=float),
        "local_mismatch_density": np.zeros(patch_count, dtype=float),
        "cumulative_repair_load": np.linspace(0.1, 1.1, patch_count),
        "s3_sector_class": np.mod(signature, 6).astype(np.int64),
        "s3_class_density": np.mod(signature, 3).astype(float),
    }
    graph = {
        "left": left,
        "right": right,
        "port_left": port_left,
        "port_right": port_right,
        "gauge": np.zeros(left.size, dtype=np.int16),
        "group_name": "Z6",
        "group_order": group_order,
        "patch_count": patch_count,
        "degree": np.maximum(degree, 1.0),
    }
    cap = RoundCap(
        axis=np.array([0.0, 0.0, 1.0]),
        theta0=0.8,
        tangent=np.array([1.0, 0.0, 0.0]),
        collar_width=0.12,
    )
    return points, [cap], raw_fields, graph


def test_paired_perturb_resettle_b_a_emits_real_rerun_rows_but_keeps_gate_closed():
    points, caps, raw_fields, graph = _toy_state()

    report = paired_perturb_resettle_b_a_report(
        points,
        caps,
        raw_fields,
        graph,
        a_grid=[1.0],
        times=[0.05],
        modes_per_cap_time=1,
        controls=["no_perturbation", "no_repair_load_channel"],
        repair_steps=2,
        repairs_per_step=16,
        seed=42,
    )

    assert report["rows"]
    assert report["control_rows"]
    assert report["primary_parent_source"] == "paired_cap_collar_perturb_resettle_rerun"
    assert report["readiness"]["checks"]["real_baryon_perturbation_runs_present"] is True
    assert report["readiness"]["checks"]["report_backed_surrogate_parent"] is False
    assert report["B_A_PARENT_RECEIPT"] is False
    assert report["physical_cmb_prediction"] is False
    assert report["source_intervention_schema"]
    assert report["rows"][0]["physical_source_intervention"] is False
    assert report["rows"][0]["source_intervention"]["source_vector_id"] == "ANOMALY_FRAME_BARYON_CONTRAST_PROXY"
    assert report["rows"][0]["delivered_source_difference"] == 2.0 * report["rows"][0]["delivered_source_half_step"]
    assert "common_source_functional_receipt" in report["readiness"]["missing_gates"]
    assert {row["control"] for row in report["control_rows"]} == {"no_perturbation", "no_repair_load_channel"}
    assert all(row["B_A_mean"] == 0.0 for row in report["control_rows"])


def test_write_paired_perturb_resettle_b_a_report_uses_b_a_parent_contract(tmp_path: Path):
    points, caps, raw_fields, graph = _toy_state()

    report = write_paired_perturb_resettle_b_a_report(
        tmp_path,
        points,
        caps,
        raw_fields,
        graph,
        a_grid=[1.0],
        times=[0.05],
        modes_per_cap_time=1,
        controls=["no_perturbation"],
        repair_steps=1,
        repairs_per_step=8,
        seed=17,
    )

    assert (tmp_path / "paired_b_a_perturbation_report.json").exists()
    assert (tmp_path / "b_a_parent_report.json").exists()
    assert (tmp_path / "paired_b_a_perturbation_rows.csv").exists()
    assert json.loads((tmp_path / "b_a_parent_report.json").read_text(encoding="utf-8"))["mode"] == report["mode"]

    comparable = comparable_data_report([tmp_path])
    lane = comparable["measurement_lanes"]["oph_B_A_parent_finite_difference"]
    assert lane["run_count"] == 1
    assert lane["real_baryon_perturbation_run_count"] == 1
    assert lane["primary_parent_source_counts"]["paired_cap_collar_perturb_resettle_rerun"] == 1


def test_paired_probe_attaches_real_observer_response_but_marks_cap_ancestry_diagnostic():
    points, caps, raw_fields, graph = _toy_state()
    observer_views = [
        {
            "view_type": "patch_observer",
            "observer_id": start,
            "support_nodes": list(range(start, min(start + 12, points.shape[0]))),
        }
        for start in range(0, 84, 12)
    ]

    report = paired_perturb_resettle_b_a_report(
        points,
        caps,
        raw_fields,
        graph,
        observer_views=observer_views,
        a_grid=[1.0],
        times=[0.05],
        modes_per_cap_time=1,
        controls=["no_perturbation"],
        repair_steps=2,
        repairs_per_step=16,
        seed=42,
    )

    assert report["paired_perturbation_response_producer_receipt"] is True
    assert report["observer_geometry_attachment"]["receipt"] is True
    assert report["observer_response_no_perturbation_control_separation_receipt"] is True
    assert all(row["paired_perturbation_response_tensor"] for row in observer_views)
    assert all(row["paired_perturbation_response_producer_receipt"] for row in observer_views)
    assert all(
        row["paired_perturbation_response_provenance"]["strict_neutral_eligible"] is False
        for row in observer_views
    )


def test_s3_paired_probe_and_local_packet_geometry_are_frame_invariant():
    points, caps, raw_fields, graph = _toy_state(48)
    graph = {
        **graph,
        "group_name": "S3",
        "gauge": np.zeros(graph["left"].size, dtype=np.int16),
        "production_sector_repair_enabled": True,
        "production_sector_repair_config": {
            "enabled": True,
            "mode": "repair_coupled_group_compose",
            "probability": 1.0,
        },
    }
    frames = np.random.default_rng(5).integers(0, 6, size=points.shape[0], dtype=np.int16)
    transformed_left, transformed_right, transformed_gauge = transform_local_frames(
        graph["port_left"],
        graph["port_right"],
        graph["gauge"],
        graph["left"],
        graph["right"],
        frames,
        group_name="S3",
        group_order=6,
    )
    transformed_graph = {
        **graph,
        "port_left": transformed_left,
        "port_right": transformed_right,
        "gauge": transformed_gauge,
    }
    kwargs = {
        "scale": 2.0 * np.pi,
        "time_value": 0.1,
        "perturb_strength": 1.0,
        "perturb_budget_mode": "fixed_fraction",
        "fixed_perturb_fraction": 0.4,
        "perturb_selection_mode": "lambda_collar_generator",
        "repair_steps": 2,
        "repairs_per_step": 16,
        "seed": 9,
    }
    original = _simulate_cap_collar_perturb_resettle(points, caps[0], raw_fields, graph, **kwargs)
    transformed = _simulate_cap_collar_perturb_resettle(
        points,
        caps[0],
        raw_fields,
        transformed_graph,
        **kwargs,
    )
    for field in (
        "record_signature",
        "stable_count",
        "committed_mask",
        "repair_load",
        "local_mismatch_density",
        "cumulative_repair_load",
    ):
        np.testing.assert_allclose(original[field], transformed[field])
    assert original["_gauge_covariant_probe_receipt"] is True
    assert original["_sector_repair_replayed"] is True
    assert original["_sector_repair_move_calls"] > 0
    assert original["_sector_edges_changed"] > 0
    assert original["_intervention_support_hash"] == transformed["_intervention_support_hash"]

    entropy = np.ones(points.shape[0], dtype=float)
    area = np.ones(points.shape[0], dtype=float)
    first_views = observer_view_rows(
        points,
        raw_fields=original,
        visible_fields={},
        caps=[],
        times=[0.1],
        cell_area_planck=area,
        cell_entropy=entropy,
        sample_count=12,
        neighborhood_size=8,
        seed=3,
        edge_left=graph["left"],
        edge_right=graph["right"],
    )
    second_views = observer_view_rows(
        points,
        raw_fields=transformed,
        visible_fields={},
        caps=[],
        times=[0.1],
        cell_area_planck=area,
        cell_entropy=entropy,
        sample_count=12,
        neighborhood_size=8,
        seed=3,
        edge_left=graph["left"],
        edge_right=graph["right"],
    )
    assert [row["locality_preserving_packet_feature_vector"] for row in first_views] == [
        row["locality_preserving_packet_feature_vector"] for row in second_views
    ]
    assert [row["measured_overlap_correspondences"] for row in first_views] == [
        row["measured_overlap_correspondences"] for row in second_views
    ]
    overlap_report = overlap_native_neutral_control_report(first_views, seed=7, max_model_points=12)
    assert overlap_report["overlap_evidence"]["source"] == "literal_support_intersection_v1"
    assert overlap_report["overlap_evidence"]["available"] is True
    assert overlap_report["presentation_invariance_control"]["serialization_row_permutation_applied"] is True
    assert overlap_report["presentation_invariance_control"]["receipt"] is True


def test_paired_probe_reuses_a_grid_dynamics_and_enforces_preflight_budget():
    points, caps, raw_fields, graph = _toy_state(48)
    graph = {
        **graph,
        "group_name": "S3",
        "gauge": np.zeros(graph["left"].size, dtype=np.int16),
        "production_sector_repair_enabled": True,
        "production_sector_repair_config": {
            "enabled": True,
            "mode": "repair_coupled_group_compose",
            "probability": 1.0,
        },
    }
    kwargs = {
        "a_grid": [0.01, 0.1],
        "times": [0.05],
        "modes_per_cap_time": 1,
        "controls": ["no_perturbation"],
        "repair_steps": 1,
        "repairs_per_step": 8,
        "reuse_dynamics_across_a_grid": True,
        "seed": 91,
    }

    report = paired_perturb_resettle_b_a_report(
        points,
        caps,
        raw_fields,
        graph,
        max_full_graph_simulations=2,
        **kwargs,
    )

    budget = report["full_graph_simulation_budget"]
    assert report["execution_status"] == "completed"
    assert report["production_move_contract"]["sector_repair_replayed"] is True
    assert budget["requested_without_a_grid_reuse"] == 4
    assert budget["planned_with_a_grid_reuse"] == 2
    assert budget["executed_full_graph_simulations"] == 2
    assert budget["a_grid_cache_hits"] == 2
    assert len(report["rows"]) == 2
    assert all(row["gauge_covariant_centered_probe_receipt"] for row in report["rows"])

    skipped = paired_perturb_resettle_b_a_report(
        points,
        caps,
        raw_fields,
        graph,
        max_full_graph_simulations=1,
        **kwargs,
    )
    assert skipped["execution_status"] == "skipped"
    assert skipped["rows"] == []
    assert skipped["full_graph_simulation_budget"]["executed_full_graph_simulations"] == 0
    assert "paired_full_graph_simulation_budget_exceeded" in skipped["execution_blockers"]


def test_paired_probe_parallel_execution_is_deterministic_and_sequentially_equivalent():
    points, caps, raw_fields, graph = _toy_state(64)
    graph = {
        **graph,
        "group_name": "S3",
        "gauge": np.zeros(graph["left"].size, dtype=np.int16),
        "production_sector_repair_enabled": True,
        "production_sector_repair_config": {
            "enabled": True,
            "mode": "repair_coupled_group_compose",
            "probability": 1.0,
        },
    }
    kwargs = {
        "a_grid": [0.01, 0.1],
        "times": [0.05],
        "modes_per_cap_time": 1,
        "controls": ["no_perturbation", "phase_shuffled_baryon_mode"],
        "repair_steps": 1,
        "repairs_per_step": 8,
        "reuse_dynamics_across_a_grid": True,
        "max_full_graph_simulations": 4,
        "seed": 119,
    }

    sequential = paired_perturb_resettle_b_a_report(
        points,
        caps,
        raw_fields,
        graph,
        full_graph_n_jobs=1,
        **kwargs,
    )
    parallel = paired_perturb_resettle_b_a_report(
        points,
        caps,
        raw_fields,
        graph,
        full_graph_n_jobs=2,
        **kwargs,
    )
    repeated = paired_perturb_resettle_b_a_report(
        points,
        caps,
        raw_fields,
        graph,
        full_graph_n_jobs=2,
        **kwargs,
    )

    assert parallel["rows"] == sequential["rows"] == repeated["rows"]
    assert (
        parallel["control_rows"]
        == sequential["control_rows"]
        == repeated["control_rows"]
    )
    assert parallel["full_graph_simulation_budget"] == sequential[
        "full_graph_simulation_budget"
    ]
    receipt = parallel["parallel_execution"]
    assert receipt["requested_n_jobs"] == 2
    assert receipt["effective_n_jobs"] == 2
    assert receipt["max_in_flight_full_graph_states"] == 2
    assert receipt["ordered_result_assembly"] is True
