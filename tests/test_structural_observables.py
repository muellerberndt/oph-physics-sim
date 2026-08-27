from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oph_fpe.cosmology.structural_observables import (
    CLASSIFICATION,
    SCHEMA,
    _adjacency_lists,
    _antipodal_parity_report,
    _basis_invariant_band_cubic_rows,
    _correlation_horizon_report,
    _defect_dynamics_report,
    _phase_transition_report,
    _return_probability,
    structural_observables_report,
    write_structural_observables_report,
)


def _octahedral_history() -> dict[str, np.ndarray]:
    points = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    left: list[int] = []
    right: list[int] = []
    for first in range(points.shape[0]):
        for second in range(first + 1, points.shape[0]):
            if not np.allclose(points[first], -points[second]):
                left.append(first)
                right.append(second)
    cycles = np.arange(6, dtype=float)
    state = np.asarray(
        [
            [3, -1, -1, -1, -1, -1],
            [3, 2, 1, -1, -1, -1],
            [2, 2, 2, 0, -1, -1],
            [1, 2, 3, 1, 0, -1],
            [0, 1, 2, 2, 1, 0],
            [-1, 0, 1, 2, 2, 1],
        ],
        dtype=float,
    )
    velocity = np.vstack([np.zeros(6), np.diff(state, axis=0)])
    commits = np.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=float,
    )
    edge_count = len(left)
    defects = np.zeros((6, edge_count), dtype=float)
    defects[0:2, 0] = 1.0
    defects[1:4, 1] = 1.0
    defects[3:5, 4] = 1.0
    defects[4:, 7] = 1.0
    delta = np.zeros_like(state)
    delta[1, 0] = 1.0
    delta[2, [0, 2, 4]] = 0.5
    delta[3, :] = 0.25
    delta[4:, :] = 0.1
    return {
        "points": points,
        "edge_left": np.asarray(left, dtype=np.int64),
        "edge_right": np.asarray(right, dtype=np.int64),
        "cycles": cycles,
        "state_frames": state,
        "velocity_frames": velocity,
        "commit_frames": commits,
        "defect_frames": defects,
        "paired_delta_frames": delta,
        "quench_values": np.linspace(1.0, 0.0, 6),
    }


def test_full_structural_report_is_target_blind_and_covers_diagnostics() -> None:
    payload = _octahedral_history()
    report = structural_observables_report(
        **payload,
        thresholds=(0.25, 0.5, 0.75),
        max_graph_hops=3,
        spectral_modes=6,
        seed=4,
    )

    assert report["schema"] == SCHEMA
    assert report["classification"] == CLASSIFICATION
    assert report["epistemic_gates"]["target_data_read"] is False
    assert report["epistemic_gates"]["physical_early_universe_claim"] is False
    carrier = report["carrier_geometry"]
    assert carrier["graph_betti_0"] == 1
    assert carrier["graph_betti_1_cycle_rank"] == 7
    assert carrier["carrier_is_time_dependent"] is False
    assert carrier["spectral_dimension_curve"]
    assert carrier["volume_growth"]["rows"]
    assert report["field_correlation_horizon"]["available"] is True
    correlation_row = report["field_correlation_horizon"]["rows"][0]
    assert "correlation_length_first_moment_hops" not in correlation_row
    assert "largest_null_excess_threshold_crossing_hop" in correlation_row
    assert report["field_correlation_horizon"]["formal_significance_claim"] is False
    assert report["paired_difference_front"]["available"] is True
    assert report["paired_difference_front"]["causal_intervention_claim"] is False
    assert report["phase_transition_proxies"]["rows"]
    assert report["phase_transition_proxies"]["susceptibility"]["available"] is False
    assert report["phase_transition_proxies"]["binder_cumulant"]["available"] is False
    assert "spatial_susceptibility_proxy" not in report["phase_transition_proxies"]["rows"][0]
    assert report["phase_transition_proxies"]["quench_scaling_inputs"]["available"] is True
    topology = report["morphology_and_graph_topology"]
    assert topology["threshold_rows"]
    assert topology["zero_dimensional_superlevel_persistence"]
    for row in topology["threshold_rows"]:
        assert row["active_graph_betti_1_cycle_rank"] >= 0
        assert row["active_graph_euler_characteristic"] == (
            row["active_graph_betti_0"] - row["active_graph_betti_1_cycle_rank"]
        )
    for frame in topology["zero_dimensional_superlevel_persistence"]:
        assert all(bar["persistence"] >= 0.0 for bar in frame["top_finite_bars"])
    assert topology["higher_betti_numbers_available"] is False
    defects = report["defect_dynamics"]
    assert defects["available"] is True
    assert defects["edge_activity_episodes"]["episode_count"] >= 4
    assert report["graph_field_statistics"]["graph_spectrum"]["rows"]
    cubic = report["graph_field_statistics"][
        "basis_invariant_low_band_cubic_couplings"
    ]
    assert cubic["rows"]
    assert cubic["ensemble_bispectrum_estimate"] is False
    assert "low_mode_bispectrum_proxy" not in report["graph_field_statistics"]
    symmetry = report["symmetry_diagnostics"]
    assert symmetry["antipodal_parity"]["available"] is True
    assert symmetry["preferred_axis"]["preferred_axis_detection_claim"] is False
    assert symmetry["chirality"]["available"] is False
    assert report["observer_record_diagnostics"]["record_production_arrow_claim"] is True


def test_optional_diagnostics_fail_closed_when_arrays_are_absent() -> None:
    payload = _octahedral_history()
    report = structural_observables_report(
        payload["points"],
        payload["edge_left"],
        payload["edge_right"],
        payload["cycles"],
        payload["state_frames"],
        spectral_modes=4,
    )

    assert report["paired_difference_front"]["available"] is False
    assert report["defect_dynamics"]["available"] is False
    assert report["scalar_velocity_diagnostics"]["available"] is False
    assert report["observer_record_diagnostics"]["available"] is False
    assert report["phase_transition_proxies"]["quench_scaling_inputs"]["available"] is False
    assert report["symmetry_diagnostics"]["chirality"]["available"] is False


def test_writer_preserves_machine_inputs_and_report(tmp_path: Path) -> None:
    payload = _octahedral_history()
    report = write_structural_observables_report(tmp_path, **payload, spectral_modes=6)

    report_path = tmp_path / "structural_observables_report.json"
    inputs_path = tmp_path / "structural_observables_inputs.npz"
    manifest_path = tmp_path / "structural_observables_manifest.json"
    assert report_path.is_file()
    assert inputs_path.is_file()
    assert manifest_path.is_file()
    assert json.loads(report_path.read_text())["schema"] == SCHEMA
    manifest = json.loads(manifest_path.read_text())
    assert manifest["target_data_included"] is False
    assert len(manifest["artifacts"]) == 2
    assert all(len(row["sha256"]) == 64 for row in manifest["artifacts"])
    with np.load(inputs_path, allow_pickle=False) as arrays:
        assert set(payload).issubset(arrays.files)
        np.testing.assert_allclose(arrays["state_frames"], payload["state_frames"])
    assert report["input_summary"]["frame_count"] == 6


def test_invalid_shapes_and_duplicate_edges_are_rejected() -> None:
    payload = _octahedral_history()
    with pytest.raises(ValueError, match="state_frames has shape"):
        structural_observables_report(
            payload["points"],
            payload["edge_left"],
            payload["edge_right"],
            payload["cycles"],
            payload["state_frames"][:, :-1],
        )
    with pytest.raises(ValueError, match="duplicate undirected edges"):
        structural_observables_report(
            payload["points"],
            np.append(payload["edge_left"], payload["edge_left"][0]),
            np.append(payload["edge_right"], payload["edge_right"][0]),
            payload["cycles"],
            payload["state_frames"],
        )
    with pytest.raises(ValueError, match="thresholds must be finite"):
        structural_observables_report(
            payload["points"],
            payload["edge_left"],
            payload["edge_right"],
            payload["cycles"],
            payload["state_frames"],
            thresholds=(0.5, float("nan")),
        )


def test_isolated_node_stays_put_in_fixed_carrier_return_probability() -> None:
    points = np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    report = structural_observables_report(
        points,
        np.asarray([0]),
        np.asarray([1]),
        np.asarray([0.0, 1.0]),
        np.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 0.0]]),
        diffusion_steps=(1, 2),
        spectral_modes=3,
    )

    carrier = report["carrier_geometry"]
    assert carrier["graph_betti_0"] == 2
    assert carrier["lazy_random_walk_return_probability"][0]["probability"] == pytest.approx(2.0 / 3.0)
    assert carrier["intrinsic_curvature"]["available"] is False


def test_single_spatial_frame_is_not_labeled_susceptibility_or_binder() -> None:
    state = np.asarray(
        [
            np.ones(12),
            np.asarray([1.0, -1.0] * 6),
        ]
    )
    report = _phase_transition_report(state, np.asarray([0.0, 1.0]), None)

    assert report["susceptibility"]["available"] is False
    assert report["binder_cumulant"]["available"] is False
    assert report["rows"][0]["spatial_fluctuation_shape_u4"] is None
    assert report["rows"][1]["spatial_fluctuation_shape_u4"] == pytest.approx(
        2.0 / 3.0
    )
    assert all(
        "susceptibility" not in key and "binder" not in key
        for row in report["rows"]
        for key in row
    )


def test_complete_eigenspace_cubic_couplings_are_basis_rotation_invariant() -> None:
    rng = np.random.default_rng(123)
    basis, _ = np.linalg.qr(rng.normal(size=(18, 7)))
    centered = rng.normal(size=18)
    centered -= np.mean(centered)
    bands = (
        {"start": 0, "stop": 3},
        {"start": 3, "stop": 7},
    )
    first = _basis_invariant_band_cubic_rows(centered, basis, bands)

    q_first, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q_second, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    rotated = basis.copy()
    rotated[:, :3] = basis[:, :3] @ q_first
    rotated[:, 3:7] = basis[:, 3:7] @ q_second
    second = _basis_invariant_band_cubic_rows(centered, rotated, bands)

    assert [row["mode_ranges_stop_exclusive"] for row in first] == [
        row["mode_ranges_stop_exclusive"] for row in second
    ]
    np.testing.assert_allclose(
        [row["mean_projected_field_product"] for row in first],
        [row["mean_projected_field_product"] for row in second],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        [row["rms_normalized_cubic_coupling"] for row in first],
        [row["rms_normalized_cubic_coupling"] for row in second],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_defect_episode_summary_marks_boundary_censoring() -> None:
    points = np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    left = np.asarray([0], dtype=np.int64)
    right = np.asarray([1], dtype=np.int64)
    cycles = np.asarray([0.0, 2.0, 5.0, 9.0])
    report = _defect_dynamics_report(
        np.ones((4, 1)), cycles, points, left, right
    )
    episodes = report["edge_activity_episodes"]

    assert episodes["episode_count"] == 1
    assert episodes["left_censored_count"] == 1
    assert episodes["right_censored_count"] == 1
    assert episodes["doubly_censored_count"] == 1
    assert episodes["completed_uncensored_run_lengths_frames"]["count"] == 0
    assert episodes["observed_active_span_cycles_lower_bound"]["minimum"] == 9.0
    assert episodes["exact_lifetime_distribution_available"] is False


def test_antipodal_fluctuation_parity_is_invariant_to_constant_offset() -> None:
    unit = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    state = np.asarray([[1.0, -1.0, 2.0, 0.0]])
    cycles = np.asarray([0.0])
    frames = np.asarray([0], dtype=np.int64)
    first = _antipodal_parity_report(unit, state, cycles, frames)
    shifted = _antipodal_parity_report(unit, state + 100.0, cycles, frames)

    assert first["global_spatial_mean_removed_before_parity_split"] is True
    assert first["rows"][0]["odd_fraction"] == pytest.approx(
        shifted["rows"][0]["odd_fraction"]
    )


def test_stochastic_return_probability_reports_reproducible_standard_error() -> None:
    node_count = 257
    left = np.arange(node_count, dtype=np.int64)
    right = np.roll(left, -1)
    adjacency = _adjacency_lists(node_count, left, right)
    first, first_error = _return_probability(
        adjacency, [1, 2, 4], np.random.default_rng(9), probe_count=8
    )
    second, second_error = _return_probability(
        adjacency, [1, 2, 4], np.random.default_rng(9), probe_count=8
    )

    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first_error, second_error)
    assert all(value is not None and value >= 0.0 for value in first_error)


def test_iid_field_does_not_acquire_an_absolute_noise_floor_length() -> None:
    node_count = 512
    left = np.arange(node_count, dtype=np.int64)
    right = np.roll(left, -1)
    adjacency = _adjacency_lists(node_count, left, right)
    state = np.random.default_rng(123).normal(size=(1, node_count))
    report = _correlation_horizon_report(
        state,
        np.asarray([0.0]),
        adjacency,
        frame_indices=np.asarray([0], dtype=np.int64),
        max_hops=8,
        anchor_count=128,
        null_draws=16,
        rng=np.random.default_rng(17),
    )
    row = report["rows"][0]

    assert row["largest_null_excess_threshold_crossing_hop"] == 0
    assert row["null_and_threshold_excess_weighted_hop_heuristic"] is None
    assert report["correlation_length_estimate_available"] is False
    assert report["formal_significance_claim"] is False
