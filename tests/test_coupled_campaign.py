from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from oph_fpe.core.icosahedral import geodesic_icosahedral_patch_arrays
from oph_fpe.cosmology.coupled_campaign import (
    DIAGNOSTIC_KEYS,
    _causal_front_receipt,
    _collision_aggregate_report,
    _collision_interventions,
    _diffusion_steps,
    _kernel_config,
    _load_config,
    _quench_scaling_report,
    _run_summary,
)
from oph_fpe.dynamics.coupled_patch import (
    CoupledPatchConfig,
    LocalizedIntervention,
    run_paired_counterfactual,
)


CONFIG = Path("configs/coupled_internal_observables_v1.yml")


def _campaign() -> dict[str, object]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _ring(node_count: int = 15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angle = 2.0 * np.pi * np.arange(node_count) / node_count
    points = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(node_count)))
    left = np.arange(node_count, dtype=np.int64)
    right = np.roll(left, -1)
    return points, left, right


def _distances(
    node_count: int, left: np.ndarray, right: np.ndarray, source: int
) -> np.ndarray:
    rows = [[] for _ in range(node_count)]
    for first, second in zip(left, right, strict=True):
        rows[int(first)].append(int(second))
        rows[int(second)].append(int(first))
    distance = np.full(node_count, -1, dtype=int)
    distance[source] = 0
    frontier = [source]
    while frontier:
        following = []
        for node in frontier:
            for neighbor in rows[node]:
                if distance[neighbor] < 0:
                    distance[neighbor] = distance[node] + 1
                    following.append(neighbor)
        frontier = following
    return distance


def test_frozen_config_retains_every_update_cycle() -> None:
    campaign = _load_config(CONFIG.resolve())
    config = _kernel_config(campaign, seed=7, quench_cycles=48)

    assert config.snapshot_stride == 1
    assert campaign["classification"] == "INTERNAL_DIAGNOSTIC_ONLY"
    assert campaign["outputs"]["save_full_state_history"] is True
    assert set(campaign["diagnostics"]) == DIAGNOSTIC_KEYS
    assert _diffusion_steps(campaign["diagnostics"]["random_walk_max_steps"])[-1] == 64


def test_quantized_linear_collision_control_uses_the_same_wide_grid() -> None:
    campaign = _load_config(CONFIG.resolve())
    base = _kernel_config(campaign, seed=7, quench_cycles=48)
    row = next(
        item
        for item in campaign["campaign"]["ablations"]
        if item["name"] == "quantized_no_quartic_no_feedback"
    )
    control = _kernel_config(
        campaign,
        seed=7,
        quench_cycles=48,
        overrides={key: value for key, value in row.items() if key != "name"},
    )

    base_step = 2.0 * base.state_bound / (base.state_levels - 1)
    control_step = 2.0 * control.state_bound / (control.state_levels - 1)
    assert control.state_bound == base.state_bound == 8.0
    assert control.state_levels == base.state_levels == 65537
    assert control_step == base_step == 1.0 / 4096.0
    assert control.quartic == 0.0
    assert control.feedback_strength == 0.0


def test_collision_aggregate_withholds_term_attribution_when_quantized_control_is_nonzero() -> None:
    run_rows = [{"zero_state_and_velocity_clipping": True}]
    domain = {
        "state_bound": 8.0,
        "state_levels": 65537,
        "state_grid_spacing": 1.0 / 4096.0,
    }
    counterfactual = {
        "same_initial_state": True,
        "same_process_noise_draws": True,
        "aligned_snapshot_cycles": True,
        "actuator_diagnostics": {
            "clean_dynamical_residual_interpretation": True,
        },
    }
    report = _collision_aggregate_report(
        [
            {
                "lane": "collision_probe",
                "residual": {
                    "maximum_state_residual_l2": 0.0042,
                    "locality_before_cone_intersection_pass": True,
                    "actuator_additivity_pass": True,
                    "disjoint_injection_supports": True,
                    "controlled_internal_interaction_receipt_pass": True,
                },
                "finite_state_domain": domain,
                "counterfactual_receipt": counterfactual,
                "run_index_rows": run_rows,
            },
            {
                "lane": "ablation_quantized_no_quartic_no_feedback",
                "residual": {
                    "maximum_state_residual_l2": 0.0043,
                    "locality_before_cone_intersection_pass": True,
                    "actuator_additivity_pass": True,
                    "disjoint_injection_supports": True,
                    "controlled_internal_interaction_receipt_pass": True,
                },
                "finite_state_domain": domain,
                "counterfactual_receipt": counterfactual,
                "run_index_rows": run_rows,
            },
        ]
    )

    assert report["declared_ablation_all_arms_zero_state_and_velocity_clipping"] is True
    assert report["quantized_linear_control_residual_nonzero"] is True
    assert report["nonzero_residual_survives_quantized_linear_ablation"] is True
    assert report["quartic_or_record_feedback_attribution_available"] is False


@pytest.mark.parametrize("failure", ["clipping", "locality", "domain"])
def test_collision_aggregate_fails_closed_for_invalid_control_packets(
    failure: str,
) -> None:
    domain = {"state_bound": 8.0, "state_levels": 65537}
    counterfactual = {
        "same_initial_state": True,
        "same_process_noise_draws": True,
        "aligned_snapshot_cycles": True,
        "actuator_diagnostics": {
            "clean_dynamical_residual_interpretation": True,
        },
    }

    def row(lane: str) -> dict[str, object]:
        return {
            "lane": lane,
            "residual": {
                "maximum_state_residual_l2": 0.004,
                "locality_before_cone_intersection_pass": True,
                "actuator_additivity_pass": True,
                "disjoint_injection_supports": True,
                "controlled_internal_interaction_receipt_pass": True,
            },
            "finite_state_domain": dict(domain),
            "counterfactual_receipt": copy.deepcopy(counterfactual),
            "run_index_rows": [{"zero_state_and_velocity_clipping": True}],
        }

    main = row("collision_probe")
    control = row("ablation_quantized_no_quartic_no_feedback")
    if failure == "clipping":
        control["run_index_rows"] = [
            {"zero_state_and_velocity_clipping": False}
        ]
    elif failure == "locality":
        control["residual"]["locality_before_cone_intersection_pass"] = False
    else:
        control["finite_state_domain"] = {
            "state_bound": 4.0,
            "state_levels": 32769,
        }

    report = _collision_aggregate_report([main, control])
    assert report["quantized_linear_null_available"] is False
    assert report["nonzero_residual_survives_quantized_linear_ablation"] is False
    assert report["quartic_or_record_feedback_attribution_available"] is False


def test_collision_zero_residual_control_remains_an_available_null() -> None:
    domain = {
        "state_bound": 8.0,
        "state_levels": 65537,
        "state_grid_spacing": 1.0 / 4096.0,
    }
    counterfactual = {
        "same_initial_state": True,
        "same_process_noise_draws": True,
        "aligned_snapshot_cycles": True,
        "actuator_diagnostics": {
            "clean_dynamical_residual_interpretation": True,
        },
    }

    def row(lane: str, peak: float) -> dict[str, object]:
        return {
            "lane": lane,
            "residual": {
                "maximum_state_residual_l2": peak,
                "locality_before_cone_intersection_pass": True,
                "actuator_additivity_pass": True,
                "disjoint_injection_supports": True,
                "controlled_internal_interaction_receipt_pass": peak > 0.0,
            },
            "finite_state_domain": dict(domain),
            "counterfactual_receipt": copy.deepcopy(counterfactual),
            "run_index_rows": [{"zero_state_and_velocity_clipping": True}],
        }

    report = _collision_aggregate_report(
        [
            row("collision_probe", 0.0042),
            row("ablation_quantized_no_quartic_no_feedback", 0.0),
        ]
    )
    assert report["quantized_linear_null_available"] is True
    assert report["quantized_linear_control_residual_nonzero"] is False
    assert report["nonzero_residual_survives_quantized_linear_ablation"] is False
    assert report["quartic_or_record_feedback_attribution_available"] is False


def test_whole_frozen_config_contract_rejects_silent_declarations(
    tmp_path: Path,
) -> None:
    campaign = _campaign()
    campaign["outputs"]["hash_every_artifact"] = False
    path = tmp_path / "bad.yml"
    path.write_text(yaml.safe_dump(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="hash_every_artifact"):
        _load_config(path)

    campaign = copy.deepcopy(_campaign())
    campaign["campaign"]["largest_gate"] = {
        "refinement_level": 6,
        "seed": 1,
        "quench_cycles": 48,
    }
    path.write_text(yaml.safe_dump(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="campaign key mismatch"):
        _load_config(path)


def test_collision_pair_uses_declared_cycle_and_exact_graph_separation() -> None:
    campaign = _campaign()
    points, left, right = geodesic_icosahedral_patch_arrays(
        2, patch_basis="cells"
    )
    config = _kernel_config(campaign, seed=7, quench_cycles=48)
    first, second = _collision_interventions(
        campaign, points, left, right, config
    )
    distance = _distances(points.shape[0], left, right, first.center_node)

    assert first.cycle == second.cycle == campaign["intervention"]["collision_pair"]["cycle"]
    assert distance[second.center_node] == campaign["intervention"]["collision_pair"]["separation_hops"]
    assert first.velocity_delta == -second.velocity_delta


def test_causal_receipt_uses_cycle_gaps_instead_of_frame_indices() -> None:
    points, left, right = _ring()
    config = CoupledPatchConfig(
        cycles=8,
        snapshot_stride=2,
        dt=0.04,
        seed=11,
        initial_state_scale=0.0,
        initial_velocity_scale=0.0,
        noise_amplitude=0.0,
        coupling=0.4,
        quartic=0.0,
        mass2_start=0.0,
        mass2_end=0.0,
        damping=0.0,
        feedback_strength=0.0,
        record_threshold=1.5,
    )
    event = LocalizedIntervention(
        center_node=0, cycle=0, radius_hops=0, state_delta=0.5
    )
    pair = run_paired_counterfactual(points, left, right, config, event)
    receipt = _causal_front_receipt(pair, event)

    assert pair.control.cycles.tolist() == [0, 2, 4, 6, 8]
    assert receipt["outside_exact_one_hop_per_update_cone_count"] == 0
    assert receipt["finite_graph_causal_contract_pass"] is True


def test_quench_aggregate_matches_defects_to_the_selected_cycle() -> None:
    rows = []
    for duration in (24, 48, 96):
        for seed_index, seed in enumerate((1, 2, 3)):
            sampled_cycle = duration + 24
            rows.append(
                {
                    "lane": "seed_and_quench_matrix",
                    "seed": seed,
                    "quench_cycles": duration,
                    "node_count": 100,
                    "phase_time_rows": [
                        {
                            "cycle": sampled_cycle,
                            "spatial_state_mean": 0.01 * seed_index,
                            "spatial_site_variance": 0.2,
                            "spatial_fluctuation_shape_u4": 0.1,
                        }
                    ],
                    "defect_fraction_time_rows": [
                        {"cycle": sampled_cycle, "defect_edge_fraction": duration / 1000.0},
                        {"cycle": 192, "defect_edge_fraction": 0.99},
                    ],
                    "final_null_excess_correlation_extent_hops": 3,
                }
            )
    report = _quench_scaling_report(
        {"model": {"quench_start_cycle": 0}}, rows
    )

    assert report["available"] is True
    assert [
        row["mean_defect_edge_fraction"] for row in report["grouped_rows"]
    ] == pytest.approx([0.024, 0.048, 0.096])
    assert all(
        row["ensemble_estimate_is_exploratory"]
        for row in report["grouped_rows"]
    )


def test_quench_aggregate_uses_realized_endpoint_before_offset() -> None:
    rows = []
    for duration in (24, 48, 96):
        nominal_end = 20 + duration
        realized_end = nominal_end + 1
        sampled_cycle = realized_end + 24
        for seed in (1, 2, 3):
            rows.append(
                {
                    "lane": "seed_and_quench_matrix",
                    "seed": seed,
                    "quench_cycles": duration,
                    "node_count": 100,
                    "phase_time_rows": [
                        {"cycle": nominal_end, "mass2": -0.719},
                        {
                            "cycle": realized_end,
                            "mass2": -0.72,
                            "spatial_state_mean": 0.01,
                            "spatial_site_variance": 0.2,
                            "spatial_fluctuation_shape_u4": 0.1,
                        },
                        {
                            "cycle": sampled_cycle,
                            "mass2": -0.72,
                            "spatial_state_mean": 0.02,
                            "spatial_site_variance": 0.3,
                            "spatial_fluctuation_shape_u4": 0.2,
                        },
                    ],
                    "defect_fraction_time_rows": [
                        {
                            "cycle": nominal_end + 24,
                            "defect_edge_fraction": 0.99,
                        },
                        {
                            "cycle": sampled_cycle,
                            "defect_edge_fraction": duration / 1000.0,
                        },
                    ],
                    "final_null_excess_correlation_extent_hops": 3,
                }
            )
    report = _quench_scaling_report(
        {
            "model": {
                "quench_start_cycle": 20,
                "mass_squared_end": -0.72,
            }
        },
        rows,
    )

    assert all(
        row["realized_quench_endpoint_resolved"] is True
        for row in report["rows"]
    )
    assert all(
        row["sampled_cycle"] == row["realized_quench_end_cycle"] + 24
        for row in report["rows"]
    )
    assert [
        row["mean_defect_edge_fraction"] for row in report["grouped_rows"]
    ] == pytest.approx([0.024, 0.048, 0.096])


def test_run_summary_carries_realized_mass2_history_into_phase_rows() -> None:
    result = SimpleNamespace(
        state_frames=np.asarray([[0.0, 0.0], [0.25, -0.25]]),
        points=np.zeros((2, 3)),
        left=np.asarray([0]),
        cycles=np.asarray([0, 1]),
        mass2_frames=np.asarray([0.7, -0.72]),
        defect_frames=np.asarray([[False], [True]]),
        commit_frames=np.asarray([[False, False], [True, False]]),
        provenance={
            "numerical_checks": {"state_clip_count": 0, "velocity_clip_count": 0}
        },
    )
    structural = {
        "phase_transition_proxies": {
            "rows": [
                {
                    "cycle": 0.0,
                    "spatial_state_mean": 0.0,
                    "spatial_site_variance": 0.0,
                },
                {
                    "cycle": 1.0,
                    "spatial_state_mean": 0.0,
                    "spatial_site_variance": 0.0625,
                },
            ]
        },
        "carrier_geometry": {"spectral_dimension_curve": []},
        "field_correlation_horizon": {"rows": []},
    }
    observer = {
        "information_dynamics": {"available": True},
        "localized_excitations": {"track_count": 0},
        "candidate_scattering_channels": {"encounter_count": 0},
        "candidate_family_clustering": {"available": False},
    }

    summary = _run_summary(
        result,
        structural,
        observer,
        lane="seed_and_quench_matrix",
        level=1,
        seed=7,
        quench_cycles=1,
        branch="baseline",
        structural_table_count=0,
    )

    assert [row["mass2"] for row in summary["phase_time_rows"]] == [0.7, -0.72]
