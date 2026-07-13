from __future__ import annotations

import numpy as np

from oph_fpe.bulk.observer_reconstruction import (
    assert_no_forbidden_keys,
    blind_observer_bulk_report,
    blind_observer_low_rank_sweep,
    bulk_reconstruction_report,
    build_blind_observer_feature_matrix,
    build_blind_observer_features,
    neutral_reconstruction_controls,
    observer_distance_matrix,
    observer_similarity_components,
    observer_similarity_matrix,
    planted_dimension_report,
    shuffled_observer_record_control,
)


def test_radial_depth_not_used_in_neutral_reconstruction():
    views = [
        {
            "view_type": "patch_observer",
            "observer_id": i,
            "committed_fraction": 1.0,
            "record_stability_mean": float(i),
            "repair_load_mean": 0.0,
            "mismatch_density_mean": 0.0,
            "visible_signature_entropy": float(i % 2),
            "visible_readout_hash": str(i % 2),
        }
        for i in range(6)
    ]
    report = bulk_reconstruction_report(views)
    assert report["bulk_3d_established"] is False
    assert report["neutral_dimension_report"]["radial_depth_used"] is False
    assert report["observer_similarity_debug_report"]["physics_claim"] is False
    assert report["observer_similarity_debug_report"]["mode"] == "neutral_summary_distance_diagnostic"
    assert report["observer_similarity_debug_report"]["receipt"] == "NEUTRAL_SUMMARY_DISTANCE_DIAGNOSTIC"
    assert report["observer_similarity_debug_report"]["claim_level"] == "debug"
    assert "component_dimension_debug_reports" in report
    assert report["controls"]["radial_depth_used"] is False
    assert "planted_3d" in report["controls"]["planted_dimensions"]
    assert "planted_s2_boundary" in report["controls"]["planted_dimensions"]
    assert "planted_h3_bulk" in report["controls"]["planted_dimensions"]
    assert report["candidate_3d_dimension_window"] is False
    assert report["blind_observer_bulk_report"]["bulk_3d_established"] is False


def test_blind_observer_features_exclude_screen_support_and_axis():
    views = []
    for i in range(10):
        views.append(
            {
                "view_type": "patch_observer",
                "observer_id": i,
                "axis": [1.0, 0.0, 0.0],
                "support_nodes": [i, i + 1],
                "record_transition_histogram": [float(i % 3), float((i + 1) % 3)],
                "checkpoint_class_transition": {"a": float(i % 2), "b": float(i % 5)},
                "perturb_resettle_signature": [float(i), float(i * i % 7)],
                "counterfactual_stability": float(i % 4) / 4.0,
                "sector_change_signature": [float(i % 6)],
                "repair_response_spectrum": [float((i * j) % 5) for j in range(4)],
            }
        )

    features, ids, metadata = build_blind_observer_features(views)
    report = blind_observer_bulk_report(views)

    assert ids == list(range(10))
    assert features.shape[0] == 10
    assert features.shape[1] > 2
    assert metadata["forbidden_input_keys_seen_but_not_used"] == ["axis", "support_nodes"]
    assert metadata["forbidden_feature_keys_used"] == []
    assert report["usable"] is True
    assert report["forbidden_feature_keys_used"] == []
    assert "low_rank_transition_chart_sweep" in report
    assert report["low_rank_transition_chart_sweep"]["physical_claim"] is False
    assert "blind_feature_group_sweep" in report
    assert report["blind_feature_group_sweep"]["physical_claim"] is False
    assert report["bulk_3d_established"] is False


def test_blind_record_transition_rank3_candidate_is_reported_without_bulk_claim():
    rng = np.random.default_rng(43)
    coords = rng.random((128, 3))
    views = []
    for i, row in enumerate(coords):
        axis = rng.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        redundant = [
            float(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[0] + row[1]),
            float(row[1] - row[2]),
            float(0.5 * row[0] - row[2]),
        ]
        views.append(
            {
                "view_type": "patch_observer",
                "observer_id": i,
                "axis": [float(value) for value in axis],
                "support_nodes": [i, (i + 1) % 128],
                "record_transition_histogram": redundant,
                "checkpoint_class_transition": {"noise": float(rng.normal())},
                "perturb_resettle_signature": [float(rng.normal()), float(rng.normal())],
                "counterfactual_stability": 0.8,
                "sector_change_signature": [float(rng.normal())],
                "repair_response_spectrum": [float(rng.normal()) for _ in range(6)],
            }
        )

    features, ids, metadata = build_blind_observer_feature_matrix(views, ("record_transition_histogram",))
    report = blind_observer_bulk_report(views)
    group_sweep = report["blind_feature_group_sweep"]

    assert ids == list(range(128))
    assert features.shape[1] == 6
    assert metadata["forbidden_input_keys_seen_but_not_used"] == ["axis", "support_nodes"]
    assert group_sweep["record_transition_rank3_receipt"] is True
    assert report["strict_blind_record_transition_3d_candidate_receipt"] is True
    assert report["bulk_3d_established"] is False


def test_blind_observer_low_rank_sweep_reports_planted_three_dimensional_candidate():
    rng = np.random.default_rng(42)
    coords = rng.normal(size=(96, 3))
    views = []
    for i, row in enumerate(coords):
        redundant = [
            float(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[0] + row[1]),
            float(row[1] - row[2]),
            float(0.5 * row[0] - row[2]),
        ]
        views.append(
            {
                "view_type": "patch_observer",
                "observer_id": i,
                "record_transition_histogram": redundant,
                "checkpoint_class_transition": {"c0": redundant[0], "c1": redundant[1]},
                "perturb_resettle_signature": redundant[2:5],
                "counterfactual_stability": 1.0,
                "sector_change_signature": [redundant[5]],
                "repair_response_spectrum": redundant[:4],
            }
        )

    report = blind_observer_low_rank_sweep(views, max_rank=6)

    assert report["usable"] is True
    assert 2.5 <= report["participation_rank"] <= 3.5
    assert report["selected_rank"] == 3
    assert report["selected_rank_report"]["candidate_3d_dimension_window"] is True
    assert report["physical_claim"] is False


def test_blind_observer_bulk_report_refuses_missing_transition_features():
    views = [{"view_type": "patch_observer", "observer_id": i, "axis": [1.0, 0.0, 0.0]} for i in range(12)]

    report = blind_observer_bulk_report(views)

    assert report["usable"] is False
    assert report["reason"] == "insufficient_blind_transition_features"
    assert report["forbidden_input_keys_seen_but_not_used"] == ["axis"]
    assert report["bulk_3d_established"] is False


def test_assert_no_forbidden_keys_rejects_s2_evidence_fields():
    try:
        assert_no_forbidden_keys({"support_nodes": [1, 2], "record_transition_histogram": [1.0]})
    except ValueError as exc:
        assert "support_nodes" in str(exc)
    else:
        raise AssertionError("expected forbidden-key rejection")


def test_observer_distance_matrix_is_zero_diagonal():
    views = [
        {"view_type": "patch_observer", "observer_id": 1, "visible_readout_hash": "a"},
        {"view_type": "patch_observer", "observer_id": 2, "visible_readout_hash": "b"},
    ]
    similarity, ids = observer_similarity_matrix(views)
    distance = observer_distance_matrix(similarity)
    assert ids == [1, 2]
    assert np.allclose(np.diag(distance), 0.0)


def test_record_family_similarity_uses_transition_affinity_without_support_overlap():
    views = [
        {
            "view_type": "patch_observer",
            "observer_id": 1,
            "support_nodes": [0, 1],
            "visible_readout_hash": "a",
            "object_packet_histogram": {"5": 0.9},
            "transition_affinity_histograms": {"checkpoint_class": {"2": 0.8}},
        },
        {
            "view_type": "patch_observer",
            "observer_id": 2,
            "support_nodes": [2, 3],
            "visible_readout_hash": "b",
            "object_packet_histogram": {"7": 0.9},
            "transition_affinity_histograms": {"checkpoint_class": {"4": 0.8}},
        },
    ]
    families = [
        {
            "object_id": "obj_a",
            "support_nodes": [100, 101],
            "record_signature": 5,
            "counterfactual_stability": 0.7,
            "transition_affinity": {"object_packet": 5, "checkpoint_class": 2},
        },
        {
            "object_id": "obj_b",
            "support_nodes": [200, 201],
            "record_signature": 7,
            "counterfactual_stability": 0.7,
            "transition_affinity": {"object_packet": 7, "checkpoint_class": 4},
        },
    ]

    components, ids = observer_similarity_components(views, families)

    assert ids == [1, 2]
    assert components["record_family"][0, 0] == 1.0
    assert components["record_family"][0, 1] < 0.5
    assert components["counterfactual"][0, 0] == 1.0
    assert components["counterfactual"][0, 1] < 0.5


def test_planted_2d_dimension():
    rng = np.random.default_rng(2)
    report = planted_dimension_report(rng.random((900, 2)))
    assert abs(report["correlation_dimension"]["estimate"] - 2.0) < 0.35


def test_planted_3d_dimension():
    rng = np.random.default_rng(3)
    report = planted_dimension_report(rng.random((900, 3)))
    assert abs(report["correlation_dimension"]["estimate"] - 3.0) < 0.45
    assert report["primary_dimension"]["estimate"] is not None


def test_planted_4d_dimension():
    rng = np.random.default_rng(4)
    report = planted_dimension_report(rng.random((900, 4)))
    assert abs(report["correlation_dimension"]["estimate"] - 4.0) < 0.65


def test_neutral_reconstruction_controls_pass_planted_dimensions():
    controls = neutral_reconstruction_controls(_varied_views(16), seed=11)

    assert controls["planted_dimensions"]["planted_2d"]["expected_failure_observed"] is True
    assert controls["planted_dimensions"]["planted_3d"]["expected_failure_observed"] is True
    assert controls["planted_dimensions"]["planted_4d"]["expected_failure_observed"] is True
    assert controls["planted_dimensions"]["planted_s2_boundary"]["expected_failure_observed"] is True
    assert controls["planted_dimensions"]["planted_h3_bulk"]["expected_failure_observed"] is True
    assert controls["shuffled_observer_records"]["expected_failure_observed"] is True
    assert controls["all_expected_failures_observed"] is True


def test_shuffled_observer_record_control_degrades_distances():
    control = shuffled_observer_record_control(_varied_views(12), seed=12)

    assert control["observer_count"] == 12
    assert control["expected_failure_observed"] is True
    assert control["mean_abs_distance_delta"] > 0.0


def _varied_views(count: int) -> list[dict]:
    return [
        {
            "view_type": "patch_observer",
            "observer_id": i,
            "committed_fraction": 1.0,
            "record_stability_mean": float(i % 5),
            "repair_load_mean": float((i * 3) % 7) / 7.0,
            "mismatch_density_mean": float((i * 5) % 11) / 11.0,
            "visible_signature_entropy": float((i * i) % 13) / 13.0,
            "visible_readout_hash": f"h{i % 4}",
            "locality_preserving_packet_feature_vector": [
                float(i) / max(1, count - 1),
                float((i * 3) % 7) / 7.0,
                float((i * i) % 11) / 11.0,
            ],
            "locality_preserving_packet_feature_schema": {
                "support_selection_carrier": "finite_patch_adjacency_bfs",
                "fields": ["repair_load", "cumulative_repair_load"],
                "excluded_hash_fields": ["record_signature", "visible_readout_hash"],
                "feature_value_coordinate_fields_used": [],
                "strict_neutral_eligible": False,
            },
        }
        for i in range(count)
    ]
