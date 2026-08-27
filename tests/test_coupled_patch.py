from __future__ import annotations

import json

import numpy as np
import pytest

from oph_fpe.dynamics.coupled_patch import (
    CLASSIFICATION,
    CoupledPatchConfig,
    LocalizedIntervention,
    run_collision_counterfactual,
    run_paired_counterfactual,
    simulate_coupled_patch,
    write_coupled_patch_run,
)


def _ring_graph(node_count: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = 2.0 * np.pi * np.arange(node_count) / node_count
    points = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(node_count)))
    left = np.arange(node_count, dtype=np.int64)
    right = np.roll(left, -1)
    return points, left, right


def _small_config(**overrides: object) -> CoupledPatchConfig:
    values: dict[str, object] = {
        "cycles": 12,
        "dt": 0.04,
        "seed": 7123,
        "state_bound": 2.0,
        "velocity_bound": 4.0,
        "state_levels": 32769,
        "velocity_levels": 32769,
        "initial_state_scale": 0.08,
        "initial_velocity_scale": 0.01,
        "coupling": 0.3,
        "quartic": 0.5,
        "mass2_start": 0.4,
        "mass2_end": -0.5,
        "damping": 0.15,
        "noise_amplitude": 0.01,
        "record_threshold": 0.4,
        "record_persistence": 2,
        "record_amplitude": 0.8,
        "feedback_strength": 0.2,
        "defect_threshold": 0.5,
        "snapshot_stride": 1,
    }
    values.update(overrides)
    return CoupledPatchConfig(**values)


def test_deterministic_finite_replay_and_contract() -> None:
    points, left, right = _ring_graph()
    config = _small_config()

    first = simulate_coupled_patch(points, left, right, config)
    second = simulate_coupled_patch(points, left, right, config)
    permuted = simulate_coupled_patch(
        points, right[::-1], left[::-1], config
    )

    for name in (
        "points",
        "left",
        "right",
        "cycles",
        "state_frames",
        "velocity_frames",
        "record_frames",
        "commit_frames",
        "defect_frames",
        "feedback_force_frames",
        "mass2_frames",
        "intervention_mask",
        "intervention_delta",
    ):
        assert np.array_equal(getattr(first, name), getattr(second, name)), name
        assert np.array_equal(getattr(first, name), getattr(permuted, name)), name

    assert first.state_frames.shape == (config.cycles + 1, points.shape[0])
    assert first.defect_frames.shape == (config.cycles + 1, left.size)
    assert np.max(np.abs(first.state_frames)) <= config.state_bound
    assert np.max(np.abs(first.velocity_frames)) <= config.velocity_bound
    assert first.provenance["classification"] == CLASSIFICATION
    assert first.provenance["physical_interpretation_allowed"] is False
    assert first.provenance["target_data_consumed"] is False
    assert first.provenance["finite_local_state"] is True
    assert first.provenance["point_coordinates_drive_dynamics"] is False
    assert first.provenance["numerical_checks"]["stability_bound_pass"] is True
    assert first.provenance["numerical_checks"][
        "deterministic_stiffness_step_guard_pass"
    ] is True
    assert first.provenance["numerical_checks"]["numerical_acceptance_pass"] is True
    assert first.provenance["conditional_read_radius_hops"] == 1
    assert first.provenance["exogenous_spatially_uniform_quench"] is True
    assert first.provenance["autonomous_local_clock"] is False
    assert first.provenance["finite_grid"]["rounding_rule"].endswith(
        "rint_ties_to_even"
    )
    assert first.provenance["saved_history"]["all_transition_frames_saved"] is True
    assert first.provenance["noise_stream_sha256"] == second.provenance["noise_stream_sha256"]
    expected_defects = np.abs(
        first.state_frames[:, first.left] - first.state_frames[:, first.right]
    ) >= config.defect_threshold
    assert np.array_equal(first.defect_frames, expected_defects)
    assert first.intervention_masks.shape == (0, points.shape[0])
    assert first.intervention_deltas.shape == (0, points.shape[0], 2)
    state_step = 2.0 * config.state_bound / (config.state_levels - 1)
    state_indices = (first.state_frames + config.state_bound) / state_step
    assert np.allclose(state_indices, np.rint(state_indices), atol=1.0e-10)


def test_paired_intervention_is_same_noise_and_graph_local() -> None:
    points, left, right = _ring_graph()
    config = _small_config(
        cycles=7,
        dt=0.05,
        initial_state_scale=0.0,
        initial_velocity_scale=0.0,
        noise_amplitude=0.0,
        coupling=0.6,
        quartic=0.2,
        mass2_start=0.0,
        mass2_end=0.0,
        record_threshold=1.5,
        feedback_strength=0.0,
    )
    event = LocalizedIntervention(
        center_node=0,
        cycle=2,
        radius_hops=0,
        state_delta=0.8,
    )

    pair = run_paired_counterfactual(points, left, right, config, event)

    assert pair.receipt["same_seed"] is True
    assert pair.receipt["same_process_noise_draws"] is True
    assert pair.receipt["same_initial_state"] is True
    assert pair.receipt["all_interventions_effective_after_quantization"] is True
    assert pair.receipt["realized_intervention_count"] == 1
    assert np.all(pair.state_delta_frames[:3] == 0.0)
    first_affected_frame = pair.state_delta_frames[3]
    allowed = np.zeros(points.shape[0], dtype=bool)
    allowed[[0, 1, points.shape[0] - 1]] = True
    assert np.any(first_affected_frame[allowed] != 0.0)
    assert np.all(first_affected_frame[~allowed] == 0.0)
    assert np.array_equal(
        np.flatnonzero(pair.intervened.intervention_mask), np.array([0])
    )
    assert pair.intervened.intervention_delta[0, 0] != 0.0
    assert not np.any(pair.control.intervention_mask)
    distance = np.minimum(np.arange(points.shape[0]), points.shape[0] - np.arange(points.shape[0]))
    for frame_index, cycle in enumerate(pair.control.cycles):
        changed = (pair.state_delta_frames[frame_index] != 0.0) | (
            pair.velocity_delta_frames[frame_index] != 0.0
        )
        allowed = distance <= max(0, int(cycle) - event.cycle)
        if cycle <= event.cycle:
            allowed[:] = False
        assert not np.any(changed & ~allowed)


def test_multiple_impulses_and_collision_residual_contract() -> None:
    points, left, right = _ring_graph()
    config = _small_config(cycles=16, noise_amplitude=0.02)
    events_a = (
        LocalizedIntervention(
            center_node=0, cycle=2, radius_hops=0, velocity_delta=0.5
        ),
        LocalizedIntervention(
            center_node=1, cycle=4, radius_hops=1, state_delta=0.2
        ),
    )
    events_b = (
        LocalizedIntervention(
            center_node=5, cycle=2, radius_hops=0, velocity_delta=-0.5
        ),
    )

    collision = run_collision_counterfactual(
        points, left, right, config, events_a, events_b
    )

    assert collision.receipt["residual_formula"] == "AB-A-B+baseline"
    assert collision.receipt["same_process_noise_draws"] is True
    assert collision.a.intervention_cycles.tolist() == [2, 4]
    assert collision.a.intervention_masks.shape == (2, points.shape[0])
    assert collision.a.intervention_deltas.shape == (2, points.shape[0], 2)
    assert collision.ab.intervention_cycles.tolist() == [2, 4, 2]
    expected = (
        collision.ab.state_frames
        - collision.a.state_frames
        - collision.b.state_frames
        + collision.baseline.state_frames
    )
    assert np.array_equal(collision.state_nonlinear_residual_frames, expected)
    assert np.all(collision.state_nonlinear_residual_frames[:3] == 0.0)
    assert collision.receipt["physical_interpretation_allowed"] is False
    actuator = collision.receipt["actuator_diagnostics"]
    assert actuator["saturation_detected"] is False
    assert actuator["same_cycle_requested_support_overlap"] is False


def test_collision_receipt_exposes_actuator_overlap_and_saturation() -> None:
    points, left, right = _ring_graph(7)
    initial_state = np.zeros(points.shape[0])
    initial_state[0] = 1.5
    config = _small_config(
        cycles=2,
        dt=0.05,
        state_levels=9,
        velocity_levels=9,
        initial_state_scale=0.0,
        initial_velocity_scale=0.0,
        noise_amplitude=0.0,
        coupling=0.0,
        quartic=0.0,
        mass2_start=0.0,
        mass2_end=0.0,
        damping=0.0,
        record_threshold=1.9,
        feedback_strength=0.0,
    )
    event = LocalizedIntervention(
        center_node=0, cycle=0, radius_hops=1, state_delta=1.0
    )

    collision = run_collision_counterfactual(
        points,
        left,
        right,
        config,
        event,
        event,
        initial_state=initial_state,
        initial_velocity=np.zeros(points.shape[0]),
    )

    actuator = collision.receipt["actuator_diagnostics"]
    assert actuator["same_cycle_requested_support_overlap"] is True
    assert actuator["same_cycle_overlap_rows"][0]["overlap_node_count"] == 3
    assert actuator["saturation_detected"] is True
    assert actuator["aggregate_realized_kick_additive"] is False
    assert actuator["aggregate_residual_nonzero_node_count"] == 1
    assert actuator["clean_dynamical_residual_interpretation"] is False
    assert collision.ab.provenance["numerical_checks"]["numerical_acceptance_pass"] is False


def test_committed_record_first_causes_a_later_local_write() -> None:
    points, left, right = _ring_graph(4)
    initial_state = np.full(4, 0.8)
    initial_velocity = np.zeros(4)
    common = dict(
        cycles=4,
        dt=0.1,
        state_bound=2.0,
        velocity_bound=4.0,
        state_levels=65537,
        velocity_levels=65537,
        initial_state_scale=0.0,
        initial_velocity_scale=0.0,
        coupling=0.0,
        quartic=0.1,
        mass2_start=0.0,
        mass2_end=0.0,
        damping=0.0,
        noise_amplitude=0.0,
        record_threshold=0.2,
        record_persistence=1,
        record_amplitude=1.0,
        defect_threshold=0.4,
    )
    feedback = CoupledPatchConfig(**common, feedback_strength=1.0)
    no_feedback = CoupledPatchConfig(**common, feedback_strength=0.0)

    driven = simulate_coupled_patch(
        points,
        left,
        right,
        feedback,
        initial_state=initial_state,
        initial_velocity=initial_velocity,
    )
    control = simulate_coupled_patch(
        points,
        left,
        right,
        no_feedback,
        initial_state=initial_state,
        initial_velocity=initial_velocity,
    )

    assert np.all(driven.commit_frames[1])
    assert np.all(driven.feedback_force_frames[1] == 0.0)
    assert np.any(driven.feedback_force_frames[2] != 0.0)
    assert np.array_equal(driven.state_frames[1], control.state_frames[1])
    assert np.any(driven.state_frames[2] != control.state_frames[2])
    assert driven.provenance["readback_nonzero_write_count"] > 0
    assert driven.provenance["readback_nonzero_force_count"] > 0
    assert driven.provenance["readback_quantized_any_write_count"] > 0
    assert driven.provenance["readback_caused_later_quantized_write"] is True
    assert driven.provenance["record_readback_first_affects_next_transition"] is True


def test_nonzero_readback_force_is_not_misreported_as_a_quantized_write() -> None:
    points, left, right = _ring_graph(4)
    result = simulate_coupled_patch(
        points,
        left,
        right,
        CoupledPatchConfig(
            cycles=3,
            dt=0.1,
            state_bound=2.0,
            velocity_bound=4.0,
            state_levels=9,
            velocity_levels=9,
            initial_state_scale=0.0,
            initial_velocity_scale=0.0,
            coupling=0.0,
            quartic=0.0,
            mass2_start=0.0,
            mass2_end=0.0,
            damping=0.0,
            noise_amplitude=0.0,
            record_threshold=0.2,
            record_persistence=1,
            record_amplitude=1.0,
            feedback_strength=0.01,
            defect_threshold=0.5,
        ),
        initial_state=np.full(4, 0.5),
        initial_velocity=np.zeros(4),
    )

    assert result.provenance["readback_nonzero_force_count"] > 0
    assert result.provenance["readback_quantized_any_write_count"] == 0
    assert result.provenance["readback_nonzero_write_count"] == 0
    assert result.provenance["readback_caused_later_quantized_write"] is False


def test_stride_keeps_initial_and_final_snapshots() -> None:
    points, left, right = _ring_graph()
    result = simulate_coupled_patch(
        points, left, right, _small_config(cycles=10, snapshot_stride=4)
    )
    assert np.array_equal(result.cycles, np.array([0, 4, 8, 10]))
    assert result.state_frames.shape[0] == 4
    history = result.provenance["saved_history"]
    assert history["all_transition_frames_saved"] is False
    assert history["semantics"].startswith("sampled_transition_history")
    assert history["omitted_transition_count"] == 7

    linear_ablation = simulate_coupled_patch(
        points,
        left,
        right,
        _small_config(cycles=3, quartic=0.0, feedback_strength=0.0),
    )
    assert linear_ablation.config["quartic"] == 0.0


def test_unstable_timestep_and_bad_graph_fail_closed() -> None:
    points, left, right = _ring_graph()
    with pytest.raises(ValueError, match="stability bound"):
        simulate_coupled_patch(
            points,
            left,
            right,
            _small_config(dt=1.0, coupling=3.0, state_bound=3.0),
        )

    duplicate_left = np.append(left, left[0])
    duplicate_right = np.append(right, right[0])
    with pytest.raises(ValueError, match="duplicate"):
        simulate_coupled_patch(
            points, duplicate_left, duplicate_right, _small_config()
        )


def test_strict_graph_intervention_and_initial_array_validation() -> None:
    points, left, right = _ring_graph()
    config = _small_config(cycles=3)

    with pytest.raises(ValueError, match="integer typed"):
        simulate_coupled_patch(points, left.astype(float), right, config)
    with pytest.raises(ValueError, match="one-dimensional"):
        simulate_coupled_patch(points, left.reshape(3, 3), right, config)
    with pytest.raises(ValueError, match="center_node must be an integer"):
        simulate_coupled_patch(
            points,
            left,
            right,
            config,
            intervention=LocalizedIntervention(True, 0, state_delta=0.5),
        )
    with pytest.raises(ValueError, match="cycle must be an integer"):
        simulate_coupled_patch(
            points,
            left,
            right,
            config,
            intervention=LocalizedIntervention(0, 1.5, state_delta=0.5),
        )
    with pytest.raises(ValueError, match=r"shape \(9,\)"):
        simulate_coupled_patch(
            points,
            left,
            right,
            config,
            initial_state=np.zeros((3, 3)),
        )
    with pytest.raises(ValueError, match="non-boolean"):
        simulate_coupled_patch(
            points,
            left,
            right,
            config,
            initial_state=np.zeros(9, dtype=bool),
        )


def test_subgrid_intervention_fails_instead_of_issuing_false_receipt() -> None:
    points, left, right = _ring_graph(4)
    config = _small_config(
        cycles=2,
        dt=0.01,
        state_levels=3,
        velocity_levels=3,
        initial_state_scale=0.0,
        initial_velocity_scale=0.0,
        noise_amplitude=0.0,
        coupling=0.0,
        quartic=0.0,
        mass2_start=0.0,
        mass2_end=0.0,
        record_threshold=1.5,
        feedback_strength=0.0,
    )
    with pytest.raises(ValueError, match="exact no-op"):
        run_paired_counterfactual(
            points,
            left,
            right,
            config,
            LocalizedIntervention(0, 0, state_delta=0.1),
        )


def test_finite_grid_midpoints_use_declared_ties_to_even_rule() -> None:
    points, left, right = _ring_graph(4)
    result = simulate_coupled_patch(
        points,
        left,
        right,
        _small_config(
            cycles=1,
            dt=0.01,
            state_levels=5,
            velocity_levels=9,
            initial_state_scale=0.0,
            initial_velocity_scale=0.0,
            noise_amplitude=0.0,
            coupling=0.0,
            quartic=0.0,
            mass2_start=0.0,
            mass2_end=0.0,
            record_threshold=1.9,
            feedback_strength=0.0,
        ),
        initial_state=np.array([-1.5, -0.5, 0.5, 1.5]),
        initial_velocity=np.zeros(4),
    )

    assert np.array_equal(result.state_frames[0], np.array([-2.0, 0.0, 0.0, 2.0]))
    assert result.provenance["finite_grid"]["state_spacing"] == 1.0


def test_writer_emits_lossless_arrays_manifest_and_boundary(tmp_path) -> None:
    points, left, right = _ring_graph()
    result = simulate_coupled_patch(points, left, right, _small_config(cycles=5))

    manifest = write_coupled_patch_run(result, tmp_path)

    assert manifest["classification"] == CLASSIFICATION
    assert manifest["physical_interpretation_allowed"] is False
    assert manifest["files"]["coupled_patch_frames.npz"]["sha256"].startswith(
        "sha256:"
    )
    on_disk = json.loads((tmp_path / "coupled_patch_manifest.json").read_text())
    assert on_disk["provenance"]["target_data_consumed"] is False
    arrays = np.load(tmp_path / "coupled_patch_frames.npz", allow_pickle=False)
    assert np.array_equal(arrays["state_frames"], result.state_frames)
    assert np.array_equal(arrays["intervention_delta"], result.intervention_delta)
    readme = (tmp_path / "README.md").read_text()
    assert "INTERNAL_DIAGNOSTIC_ONLY" in readme
    assert "No\narray is identified with CMB temperature" in readme
    assert "ties-to-even" in readme
    assert "not thereby a topological defect" in readme
