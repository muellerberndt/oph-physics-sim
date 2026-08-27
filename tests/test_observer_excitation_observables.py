from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oph_fpe.cosmology.observer_excitation_observables import (
    CLASSIFICATION,
    observer_excitation_observables,
    write_observer_excitation_observables,
)


def _ring_fixture() -> dict[str, np.ndarray]:
    node_count = 60
    time_count = 12
    angles = 2.0 * np.pi * np.arange(node_count) / node_count
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    left = np.arange(node_count, dtype=np.int32)
    right = np.roll(left, -1)
    cycles = np.arange(time_count, dtype=float)

    # Low-amplitude deterministic texture prevents degenerate information
    # statistics. Two high-amplitude peaks approach, merge at node 6, and
    # split. Eight distant stationary tracks supply a family-stability fixture.
    nodes = np.arange(node_count, dtype=float)
    state = np.asarray(
        [0.04 * np.sin(0.47 * nodes + 0.31 * time) for time in range(time_count)]
    )
    paths = [
        (2, 10),
        (3, 9),
        (4, 8),
        (5, 7),
        (6,),
        (5, 7),
        (4, 8),
        (3, 9),
        (2, 10),
        (1, 11),
        (0, 12),
        (59, 13),
    ]
    for time, positions in enumerate(paths):
        for position in positions:
            state[time, position] += 8.0
    stationary = [20, 25, 30, 35, 40, 45, 50, 55]
    for index, position in enumerate(stationary):
        state[:, position] += 5.0 if index < 4 else 10.0

    velocity = np.zeros_like(state)
    records = np.zeros((time_count, node_count), dtype=np.int8)
    commits = np.zeros_like(records)
    for time in range(time_count):
        extent = min(node_count, 6 * (time + 1))
        records[time, :extent] = np.where(np.arange(extent) % 2 == 0, 1, -1)
        if time >= 2:
            commits[time] = records[time - 2]
    defects = (state > 3.0).astype(np.uint8)

    # Same-seed paired response with an exactly one-hop-per-frame support
    # front. It is still only an operational graph-cycle diagnostic.
    delta = np.zeros_like(state)
    for time in range(time_count):
        distance = np.minimum(np.arange(node_count), node_count - np.arange(node_count))
        delta[time, distance <= time] = 1.0 / (time + 1)
    intervention_origin_mask = np.zeros(node_count, dtype=np.uint8)
    intervention_origin_mask[0] = 1

    return {
        "points": points,
        "left": left,
        "right": right,
        "cycles": cycles,
        "state_frames": state,
        "velocity_frames": velocity,
        "record_frames": records,
        "commit_frames": commits,
        "defect_frames": defects,
        "intervention_delta": delta,
        "intervention_origin_cycle": 0.0,
        "intervention_origin_mask": intervention_origin_mask,
        "locality_hops_per_cycle": 1.0,
    }


def test_report_covers_observer_excitation_lanes_and_fails_closed() -> None:
    arrays = _ring_fixture()
    report = observer_excitation_observables(
        **arrays,
        entropy_bins=6,
        excitation_threshold=20.0,
        max_graph_hops=8,
        seed=41,
    )

    assert report["classification"] == CLASSIFICATION
    assert report["target_data_read"] is False
    assert report["measurement_files_read"] == []

    information = report["information_dynamics"]
    assert information["conditional_entropy_rate_nats_per_step"] >= 0.0
    assert information["temporal_mutual_information_nats"] >= 0.0
    assert information["edge_mutual_information_nats"] >= 0.0
    assert len(information["time_rows"]) == arrays["cycles"].size

    arrow = report["mixing_and_record_arrow"]
    assert arrow["record_birth_count"] > 0
    assert arrow["record_loss_count"] == 0
    assert arrow["record_arrow_score"] == 1.0
    assert arrow["record_fraction_monotone"] is True

    neighborhoods = report["latent_patch_neighborhoods"]
    assert neighborhoods["mean_adjacent_neighborhood_overlap_fraction"] > 0.0
    assert neighborhoods["same_source_observer_consensus_available"] is False
    assert neighborhoods["adjacent_nonzero_signed_record_sign_agreement"] is None
    assert neighborhoods["legacy_adjacent_committed_endpoint_sign_similarity"] == 0.0
    assert len(neighborhoods["latent_smoothing_rows"]) == 9
    assert neighborhoods["homogeneity_claim_available"] is False
    assert neighborhoods["latent_graph_cone_dependence"]["physical_causality_claim"] is False
    assert report["observer_local_skies"]["deprecated_alias_for"] == (
        "latent_patch_neighborhoods"
    )

    response = report["paired_intervention_response"]
    assert response["available"] is True
    assert response["one_hop_per_frame_cone_leakage_fraction"] == 0.0
    assert response["front_slope_hops_per_cycle_proxy"] == pytest.approx(1.0)
    assert response["causal_speed_claim"] is False

    excitations = report["localized_excitations"]
    assert excitations["component_count"] > 0
    assert excitations["track_count"] >= 8
    assert excitations["particle_identification"] is False
    assert all(row["physical_particle_claim"] is False for row in excitations["track_catalog"])

    mode = report["graph_field_snapshot_proxies"]
    assert len(mode["time_rows"]) == arrays["cycles"].size
    assert mode["dispersion_fit_available"] is False
    assert mode["omega_squared_vs_graph_k_squared_slope"] is None
    assert mode["physical_dispersion_relation_claim"] is False

    scattering = report["candidate_scattering_channels"]
    assert scattering["available"] is False
    assert scattering["candidate_detection_available"] is True
    assert any(row["channel_topology"] == "2_to_2" for row in scattering["event_rows"])
    assert scattering["matched_null_control_available"] is False
    assert scattering["four_arm_nonlinear_evidence_available"] is False
    assert scattering["physical_scattering_claim"] is False

    families = report["candidate_family_clustering"]
    assert families["available"] is False
    assert families["candidate_partition_computed"] is True
    assert families["stable_candidate_partition"] is False
    assert families["one_family_null_rejected"] is False
    assert families["physical_particle_family_claim"] is False
    assert families["clustered_track_count"] >= 6

    assert all(value is False for value in report["physical_claims"].values())
    assert all(value is False for value in report["physical_promotion_gates"].values())


def test_report_is_deterministic_and_without_delta_fails_intervention_closed() -> None:
    arrays = _ring_fixture()
    arrays.pop("intervention_delta")
    first = observer_excitation_observables(
        **arrays, entropy_bins=5, excitation_threshold=20.0, seed=7
    )
    second = observer_excitation_observables(
        **arrays, entropy_bins=5, excitation_threshold=20.0, seed=7
    )

    assert first == second
    assert first["paired_intervention_response"]["available"] is False
    assert first["paired_intervention_response"]["causal_speed_claim"] is False


def test_writer_preserves_machine_readable_products(tmp_path: Path) -> None:
    arrays = _ring_fixture()
    report = write_observer_excitation_observables(
        tmp_path,
        **arrays,
        entropy_bins=6,
        excitation_threshold=20.0,
        seed=11,
    )

    expected = {
        "observer_excitation_observables.json",
        "observer_information_timeseries.csv",
        "observer_autocorrelation.csv",
        "observer_homogeneity_scale.csv",
        "observer_excitation_components.csv",
        "observer_excitation_tracks.csv",
        "observer_mode_dispersion.csv",
        "observer_scattering_candidates.csv",
        "observer_candidate_families.csv",
        "observer_excitation_analysis_arrays.npz",
        "OBSERVER_EXCITATION_OBSERVABLES.md",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    with np.load(tmp_path / "observer_excitation_analysis_arrays.npz") as payload:
        assert payload["quantized_state"].shape == arrays["state_frames"].shape
        assert payload["excitation_component_labels"].shape == arrays["state_frames"].shape
        assert payload["excitation_track_labels"].shape == arrays["state_frames"].shape
        assert payload["intervention_delta_magnitude"].shape == arrays["state_frames"].shape
        assert payload["intervention_origin_mask"].shape == (
            arrays["state_frames"].shape[1],
        )
    assert report["schema"] == "oph_observer_excitation_observables_v1"


def test_no_actual_track_encounter_produces_no_scattering_channel() -> None:
    arrays = _ring_fixture()
    state = 0.04 * np.sin(
        0.47 * np.arange(60, dtype=float)[None, :]
        + 0.31 * np.arange(12, dtype=float)[:, None]
    )
    state[:, 20] += 8.0
    arrays["state_frames"] = state
    arrays["velocity_frames"] = np.zeros_like(state)
    arrays["defect_frames"] = (state > 3.0).astype(np.uint8)
    report = observer_excitation_observables(
        **arrays, excitation_threshold=20.0, max_graph_hops=4
    )

    assert report["candidate_scattering_channels"]["available"] is False
    assert report["candidate_scattering_channels"]["candidate_detection_available"] is False
    assert report["candidate_scattering_channels"]["event_rows"] == []
    assert report["candidate_scattering_channels"]["cross_section_or_amplitude_available"] is False


def test_shape_and_cycle_validation_is_fail_fast() -> None:
    arrays = _ring_fixture()
    arrays["record_frames"] = arrays["record_frames"][:, :-1]
    with pytest.raises(ValueError, match="record_frames has shape"):
        observer_excitation_observables(**arrays)

    arrays = _ring_fixture()
    arrays["cycles"] = arrays["cycles"][::-1]
    with pytest.raises(ValueError, match="strictly increasing"):
        observer_excitation_observables(**arrays)


def test_declared_anchor_and_lifetime_controls_are_consumed() -> None:
    arrays = _ring_fixture()
    report = observer_excitation_observables(
        **arrays,
        excitation_threshold=20.0,
        excitation_min_lifetime_frames=20,
        latent_neighborhood_anchor_count=7,
    )

    assert report["latent_patch_neighborhoods"]["latent_smoothing_anchor_count"] == 7
    assert report["candidate_family_clustering"]["minimum_lifetime_frames"] == 20
    assert report["candidate_family_clustering"]["eligible_track_count"] == 0
    assert report["declared_diagnostic_settings"][
        "latent_neighborhood_anchor_count_requested"
    ] == 7


def test_edge_defects_are_preserved_and_projected_to_incident_observers() -> None:
    arrays = _ring_fixture()
    arrays["left"] = np.arange(59, dtype=np.int32)
    arrays["right"] = np.arange(1, 60, dtype=np.int32)
    edge_defects = np.zeros((12, 59), dtype=np.int8)
    edge_defects[np.arange(12), np.arange(12)] = 1
    arrays["defect_frames"] = edge_defects
    report = observer_excitation_observables(
        **arrays, excitation_threshold=20.0, max_graph_hops=3
    )

    assert report["array_semantics"]["defect_semantics"] == (
        "edge_defects_with_incident_node_projection"
    )
    defects = report["defect_dynamics"]
    assert defects["exact_defect_counts_by_frame"] == [1] * 12
    assert defects["incident_node_counts_by_frame"] == [2] * 12
    assert report["mixing_and_record_arrow"]["initial_defect_count"] == 2


def test_record_fraction_monotonicity_uses_current_records_not_ever_seen_union() -> None:
    arrays = _ring_fixture()
    records = np.zeros_like(arrays["record_frames"])
    records[0] = 1
    records[1, :15] = 1
    arrays["record_frames"] = records

    report = observer_excitation_observables(
        **arrays, excitation_threshold=20.0, max_graph_hops=3
    )
    arrow = report["mixing_and_record_arrow"]
    assert arrow["current_record_fraction_by_frame"][:3] == [1.0, 0.25, 0.0]
    assert arrow["record_fraction_monotone"] is False
    assert arrow["cumulative_ever_recorded_fraction_monotone_by_construction"] is True


def test_paired_cone_requires_declared_injection_metadata() -> None:
    arrays = _ring_fixture()
    arrays.pop("intervention_origin_cycle")
    arrays.pop("intervention_origin_mask")
    arrays.pop("locality_hops_per_cycle")

    report = observer_excitation_observables(
        **arrays, excitation_threshold=20.0, max_graph_hops=8
    )
    response = report["paired_intervention_response"]
    assert response["available"] is False
    assert response["response_support_detected"] is True
    assert set(response["missing_required_metadata"]) == {
        "intervention_origin_cycle",
        "intervention_origin_mask",
        "locality_hops_per_cycle",
    }


def test_stride_two_one_hop_per_cycle_front_has_no_false_leakage() -> None:
    arrays = _ring_fixture()
    cycles = 2.0 * np.arange(arrays["cycles"].size)
    distance = np.minimum(np.arange(60), 60 - np.arange(60))
    delta = np.zeros_like(arrays["state_frames"])
    for time_index, cycle in enumerate(cycles.astype(int).tolist()):
        delta[time_index, distance <= cycle] = 1.0 / (time_index + 1)
    arrays["cycles"] = cycles
    arrays["intervention_delta"] = delta

    report = observer_excitation_observables(
        **arrays, excitation_threshold=20.0, max_graph_hops=24
    )
    response = report["paired_intervention_response"]
    assert response["available"] is True
    assert response["declared_locality_cone_leakage_fraction"] == 0.0
    assert response["front_slope_hops_per_cycle_proxy"] == pytest.approx(1.0)
    assert response["response_rows"][1]["allowed_response_radius_hops"] == 2.0


def test_iid_noise_does_not_promote_encounters_or_candidate_families() -> None:
    node_count = 4096
    time_count = 10
    rng = np.random.default_rng(20260827)
    angles = 2.0 * np.pi * np.arange(node_count) / node_count
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    left = np.arange(node_count, dtype=np.int32)
    right = np.roll(left, -1)
    zeros = np.zeros((time_count, node_count), dtype=np.int8)

    report = observer_excitation_observables(
        points=points,
        left=left,
        right=right,
        cycles=np.arange(time_count, dtype=float),
        state_frames=rng.normal(size=(time_count, node_count)),
        velocity_frames=rng.normal(size=(time_count, node_count)),
        record_frames=zeros,
        commit_frames=zeros,
        defect_frames=zeros,
        excitation_threshold=3.0,
        max_graph_hops=3,
        seed=29,
    )

    assert report["array_semantics"]["velocity_reduction"] == "identity_scalar"
    assert report["localized_excitations"]["active_node_fraction"] < 0.03
    scattering = report["candidate_scattering_channels"]
    assert scattering["available"] is False
    assert scattering["interaction_promotion_available"] is False
    assert scattering["encounter_excess_over_null_established"] is False
    families = report["candidate_family_clustering"]
    assert families["available"] is False
    assert families["stable_candidate_partition"] is False
    assert families["one_family_null_rejected"] is False
