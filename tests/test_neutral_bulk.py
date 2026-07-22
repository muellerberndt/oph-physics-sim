from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from oph_fpe.cosmology.comparable_data import comparable_data_report
from oph_fpe.bulk.quotient_geometry import (
    ChannelMetricSpec,
    ProvenanceRecord,
    quotient_geometry_certificate,
)
from oph_fpe.bulk.neutral_bulk import (
    DEFAULT_NEUTRAL_WEIGHTS,
    _double_center_squared_distance,
    _measured_overlap_presentation_invariance_report,
    _overlap_feature_distance_matrix,
    _overlap_graph_rank_selection,
    _prime_geometric_selected_rank_controls,
    bounded_strict_neutral_observer_views,
    build_neutral_observer_views,
    neutral_channel_duplicate_audit,
    neutral_distance,
    neutral_distance_matrix,
    neutral_feature_matrix,
    neutral_leakage_audit,
    neutral_model_selection,
    neutral_3d_bulk_audit_report,
    neutral_independent_rank_selector_audit_report,
    overlap_native_graph_geometry_report,
    overlap_native_graph_geometry_sweep_report,
    overlap_residualized_graph_geometry_report,
    overlap_residualized_graph_geometry_sweep_report,
    overlap_native_neutral_control_report,
    neutral_profile_audit_report,
    planted_neutral_control_report,
    prime_geometric_rank_refinement_report,
    prime_geometric_rank_sweep_report,
    shuffled_neutral_control_report,
    strict_neutral_bulk_frontier_report,
    strict_neutral_bulk_receipt,
    strict_neutral_bulk_report,
    write_neutral_independent_rank_selector_audit_report,
    write_overlap_native_graph_geometry_report,
    write_overlap_native_graph_geometry_sweep_report,
    write_overlap_residualized_graph_geometry_report,
    write_overlap_residualized_graph_geometry_sweep_report,
    write_overlap_native_neutral_control_report,
    write_prime_geometric_rank_sweep_report,
    write_strict_neutral_bulk_report,
    write_strict_neutral_bulk_frontier_report,
    write_neutral_3d_bulk_audit_report,
    write_neutral_profile_audit_report,
)


def _canonical_refinement_receipt() -> dict[str, object]:
    required = [4_096, 16_384, 65_536, 262_144]
    return {
        "mode": "prime_geometric_rank_refinement_v0",
        "sizes": [{"patch_count": value} for value in required],
        "required_patch_count_ladder": required,
        "missing_required_patch_counts": [],
        "required_ladder_complete": True,
        "multi_scale": True,
        "all_control_quotient_spatial_3d_candidates": True,
        "all_candidate_s2_leakage_pass": True,
        "all_candidate_rank3_e3": True,
        "candidate_dimension_stable": True,
        "independent_rank3_selector_all": True,
        "proper_negative_control_all": True,
        "directional_h3_strict_all": True,
        "measured_overlap_geometry_all": True,
        "strict_neutral_bulk_refinement_receipt": True,
        "proof_blockers": [],
    }


def _quotient_provenance(count: int) -> list[ProvenanceRecord]:
    return [
        ProvenanceRecord(
            record_id=f"record_{index}",
            split="train",
            batch_id=f"batch_{index}",
            seed_id=f"seed_{index}",
            boundary_condition_id=f"boundary_{index}",
            trajectory_family_id=f"trajectory_{index}",
        )
        for index in range(count)
    ]


def test_build_neutral_observer_views_ignores_forbidden_geometry_fields():
    observer_views = [
        _observer_view(
            1,
            records=[1, 2, 3, 1],
            checkpoints=[4, 4, 5, 5],
            sectors=[0, 1, 1, 2],
            repairs=[2, 3, 2, 3],
            axis=[1.0, 0.0, 0.0],
            support_nodes=[10, 11],
            h3_point=[0.1, 0.2, 0.3],
            cap_axis=[0.0, 1.0, 0.0],
            radial_depth=17.0,
            modular_depth=23.0,
        )
    ]

    views = build_neutral_observer_views(observer_views)

    assert len(views) == 1
    view = views[0]
    assert view.record_transition_hist.shape == (32,)
    assert view.record_signature_hist.shape == (64,)
    assert view.object_packet_hist.shape == (64,)
    assert view.boundary_packet_hash_hist.shape == (128,)
    assert view.overlap_correspondence_hist.shape == (256,)
    assert view.checkpoint_transition_hist.shape == (32,)
    assert view.sector_transition_hist.shape == (6,)
    assert view.port_pair_lag_hist.shape == (512,)
    assert view.repair_response_hist.shape == (16,)
    assert view.repair_response_spectrum.shape == (32,)
    assert view.repair_current_tensor.shape == (128,)
    assert view.perturbation_response_tensor.shape == (128,)
    assert view.first_passage_response_hist.shape == (64,)
    assert view.modular_response_hist.shape == (64,)
    assert view.prime_geometric_modular_spectrum.shape == (64,)
    assert view.prime_geometric_control_quotient_spectrum.shape == (64,)
    assert view.support_visible_modular_spectrum.shape == (64,)
    assert view.repair_modular_spectrum.shape == (32,)
    assert view.transition_token_hist.shape == (128,)
    assert view.transition_token_persistent_hist.shape == (128,)
    assert view.transition_affinity_hist.shape == (96,)
    assert view.scalar_readout_features.shape == (6,)
    assert np.isclose(view.record_transition_hist.sum(), 1.0)

    # If forbidden geometry fields were used, this radically different copy
    # would change the neutral feature vectors. It must not.
    changed_geometry = [
        {
            **observer_views[0],
            "axis": [0.0, 0.0, 1.0],
            "support_nodes": [999],
            "h3_point": [9.0, 9.0, 9.0],
            "cap_axis": [1.0, 0.0, 0.0],
            "radial_depth": 999.0,
            "modular_depth": 999.0,
        }
    ]
    changed_views = build_neutral_observer_views(changed_geometry)

    assert np.allclose(view.record_transition_hist, changed_views[0].record_transition_hist)
    assert np.allclose(view.checkpoint_transition_hist, changed_views[0].checkpoint_transition_hist)
    assert np.allclose(view.sector_transition_hist, changed_views[0].sector_transition_hist)
    assert np.allclose(view.repair_response_hist, changed_views[0].repair_response_hist)
    assert np.allclose(view.record_signature_hist, changed_views[0].record_signature_hist)
    assert np.allclose(view.object_packet_hist, changed_views[0].object_packet_hist)
    assert np.allclose(view.boundary_packet_hash_hist, changed_views[0].boundary_packet_hash_hist)
    assert np.allclose(view.overlap_correspondence_hist, changed_views[0].overlap_correspondence_hist)
    assert np.allclose(view.port_pair_lag_hist, changed_views[0].port_pair_lag_hist)
    assert np.allclose(view.repair_current_tensor, changed_views[0].repair_current_tensor)
    assert np.allclose(view.perturbation_response_tensor, changed_views[0].perturbation_response_tensor)
    assert np.allclose(view.first_passage_response_hist, changed_views[0].first_passage_response_hist)
    assert np.allclose(view.modular_response_hist, changed_views[0].modular_response_hist)
    assert np.allclose(view.prime_geometric_modular_spectrum, changed_views[0].prime_geometric_modular_spectrum)
    assert np.allclose(
        view.prime_geometric_control_quotient_spectrum,
        changed_views[0].prime_geometric_control_quotient_spectrum,
    )
    assert np.allclose(view.support_visible_modular_spectrum, changed_views[0].support_visible_modular_spectrum)
    assert np.allclose(view.repair_modular_spectrum, changed_views[0].repair_modular_spectrum)
    assert np.allclose(view.transition_affinity_hist, changed_views[0].transition_affinity_hist)


def test_theory_required_neutral_channels_are_extracted_without_geometry():
    base = _observer_view(
        7,
        records=[1, 2, 3],
        checkpoints=[1, 2],
        sectors=[0, 1],
        repairs=[2, 3],
        axis=[1.0, 0.0, 0.0],
        support_nodes=[10, 11],
        h3_point=[0.1, 0.2, 0.3],
        cap_axis=[0.0, 1.0, 0.0],
        radial_depth=3.0,
        modular_depth=5.0,
    )
    base.update(
        {
            "local_boundary_packet_hash_histogram": {"packet-A": 2.0, "packet-B": 1.0},
            "overlap_correspondence_histogram": {"observer-7:observer-8": 3.0},
            "port_pair_lag_histogram": {"north:east:2": 4.0},
            "repair_current_tensor": [0.0, 1.5, -0.5],
            "perturbation_response_tensor": {"delta-local-port-0": 2.0, "delta-local-port-1": -1.0},
            "first_passage_time_histogram": {"3": 1.0, "5": 2.0},
        }
    )
    changed_geometry = {
        **base,
        "axis": [0.0, 0.0, 1.0],
        "support_nodes": [999],
        "h3_point": [9.0, 9.0, 9.0],
        "cap_axis": [1.0, 0.0, 0.0],
        "radial_depth": 999.0,
        "modular_depth": 999.0,
    }

    view, changed = build_neutral_observer_views([base, changed_geometry])

    assert np.isclose(view.boundary_packet_hash_hist.sum(), 1.0)
    assert np.isclose(view.overlap_correspondence_hist.sum(), 0.0)
    assert np.isclose(view.port_pair_lag_hist.sum(), 1.0)
    assert np.linalg.norm(view.repair_current_tensor) > 0.0
    assert np.linalg.norm(view.perturbation_response_tensor) == 0.0
    assert np.isclose(view.first_passage_response_hist.sum(), 1.0)
    assert np.allclose(view.boundary_packet_hash_hist, changed.boundary_packet_hash_hist)
    assert np.allclose(view.overlap_correspondence_hist, changed.overlap_correspondence_hist)
    assert np.allclose(view.port_pair_lag_hist, changed.port_pair_lag_hist)
    assert np.allclose(view.repair_current_tensor, changed.repair_current_tensor)
    assert np.allclose(view.perturbation_response_tensor, changed.perturbation_response_tensor)
    assert np.allclose(view.first_passage_response_hist, changed.first_passage_response_hist)


def test_missing_theory_channels_do_not_dilute_available_neutral_distance():
    observer_views = [
        _observer_view(i, records=[i, i + 1, i + 2], checkpoints=[i % 4], sectors=[i % 3], repairs=[i % 5])
        for i in range(8)
    ]
    views = build_neutral_observer_views(observer_views)

    record_only = neutral_distance_matrix(views, weights={"record": 1.0})
    record_plus_missing = neutral_distance_matrix(
        views,
        weights={"record": 1.0, "boundary_packet": 1.0, "repair_current_tensor": 1.0},
    )

    assert np.allclose(record_only, record_plus_missing)


def test_custom_strict_neutral_weights_reach_quotient_manifest_and_evidence_gaps():
    observer_views = [
        _observer_view(i, records=[i, i + 1], checkpoints=[i % 4], sectors=[i % 3], repairs=[i % 5])
        for i in range(8)
    ]

    report = strict_neutral_bulk_report(
        observer_views,
        weights={"record": 1.0},
        model_selection={"best_model": "H3", "h3_beats_s2": True, "h3_beats_h2_h4": True},
        controls={
            "shuffled_records_fail": True,
            "shuffled_transition_labels_fail": True,
            "planted_2d_returns_2d": True,
            "planted_3d_returns_3d": True,
            "planted_h3_returns_h3": True,
        },
        refinement=_canonical_refinement_receipt(),
    )

    manifest = report["quotient_geometry_contract"]["channel_manifest"]
    assert [row["name"] for row in manifest] == ["record"]
    assert report["strict_neutral_theory_alignment"]["theory_required_channels_present"] is False
    assert any(
        "inactive_required_neutral_channel:boundary_packet" in gap
        for gap in report["strict_neutral_theory_evidence_gaps"]
    )
    assert report["receipt"]["theory_required_channels_present"] is False
    assert report["strict_neutral_bulk"] is False


def test_strict_neutral_writer_hash_binds_primitive_observer_source(tmp_path: Path):
    observer_path = tmp_path / "observer_views.jsonl"
    observer_rows = [
        _observer_view(
            index,
            records=[index, index + 1, index + 2],
            checkpoints=[index % 4],
            sectors=[index % 3],
            repairs=[index % 5],
        )
        for index in range(8)
    ]
    observer_path.write_text(
        "\n".join(json.dumps(row) for row in observer_rows) + "\n",
        encoding="utf-8",
    )

    report = write_strict_neutral_bulk_report(
        tmp_path,
        seed=7,
        max_model_points=16,
        planted_control_points=16,
    )
    manifest = json.loads(
        (tmp_path / "strict_neutral_source_manifest.json").read_text(encoding="utf-8")
    )
    expected_hash = "sha256:" + hashlib.sha256(observer_path.read_bytes()).hexdigest()

    assert report["source_artifact"] == manifest
    assert manifest["schema"] == "strict_neutral_bulk_source_v1"
    assert manifest["observer_views_sha256"] == expected_hash
    assert manifest["analysis_parameters"] == {
        "seed": 7,
        "max_model_points": 16,
        "planted_control_points": 16,
    }
    assert manifest["refinement_input"]["primitive_replay_available"] is False


def test_strict_neutral_writer_bounds_dense_cohort_before_geometry(tmp_path: Path):
    observer_path = tmp_path / "observer_views.jsonl"
    observer_rows = [
        _observer_view(
            index,
            records=[index, index + 1, index + 2],
            checkpoints=[index % 4],
            sectors=[index % 3],
            repairs=[index % 5],
        )
        for index in range(32)
    ]
    observer_path.write_text(
        "\n".join(json.dumps(row) for row in observer_rows) + "\n",
        encoding="utf-8",
    )

    expected_rows, expected_population = bounded_strict_neutral_observer_views(
        observer_rows,
        max_observers=8,
    )
    report = write_strict_neutral_bulk_report(
        tmp_path,
        seed=7,
        max_model_points=16,
        planted_control_points=16,
        max_observers=8,
    )
    manifest = json.loads(
        (tmp_path / "strict_neutral_source_manifest.json").read_text(encoding="utf-8")
    )

    assert len(expected_rows) == 8
    assert report["observer_count"] == 8
    assert report["distance_matrix_shape"] == [8, 8]
    assert manifest["schema"] == "strict_neutral_bulk_source_v2"
    assert manifest["observer_view_row_count"] == 32
    assert manifest["analysis_parameters"]["max_observers"] == 8
    assert manifest["analysis_population"] == expected_population
    assert manifest["analysis_population"]["sampling_policy"] == (
        "deterministic_observer_id_hash_rank_v1"
    )


def test_rich_observer_visible_packets_affect_neutral_distance_without_geometry():
    base = _observer_view(1, records=[1, 2], checkpoints=[1], sectors=[0], repairs=[1])
    same_packets_new_geometry = {
        **base,
        "axis": [0.0, 0.0, 1.0],
        "support_nodes": [999, 1000],
        "h3_point": [9.0, 9.0, 9.0],
        "cap_axis": [0.0, 1.0, 0.0],
        "radial_depth": 100.0,
        "modular_depth": 200.0,
    }
    changed_packets = {
        **base,
        "record_signature_histogram": {"63": 1.0},
        "object_packet_histogram": {"7": 1.0},
        "repair_response_spectrum": [-1.0] * 32,
        "prime_geometric_modular_spectrum": [2.0] * 64,
        "prime_geometric_control_quotient_spectrum": [3.0] * 64,
        "support_visible_modular_spectrum": [-2.0] * 64,
        "repair_modular_spectrum": [1.5] * 32,
        "modular_response_histograms": {"modular_response_cluster": {"99": 1.0}},
        "transition_history_histograms": {
            **base["transition_history_histograms"],
            "local_transition_token": {"123456": 1.0},
            "local_transition_token_persistent": {"123456": 1.0},
        },
        "transition_affinity_histograms": {"record_family": {"9": 1.0}},
        "visible_signature_entropy": 9.0,
        "counterfactual_stability": 0.1,
    }

    a, b, c = build_neutral_observer_views([base, same_packets_new_geometry, changed_packets])

    assert neutral_distance(a, b) == 0.0
    # Hash/categorical packets remain readable but are non-claim-bearing in
    # the default geometry because their token bins have no locality metric.
    assert neutral_distance(a, c) == 0.0
    assert neutral_distance(a, c, weights={"record_signature": 1.0, "object_packet": 1.0}) > 0.05


def test_neutral_distance_matrix_is_symmetric_and_zero_diagonal():
    observer_views = [
        _observer_view(1, records=[1, 1, 2], checkpoints=[1], sectors=[0, 1], repairs=[1]),
        _observer_view(2, records=[2, 3, 3], checkpoints=[2], sectors=[1, 1], repairs=[2]),
        _observer_view(3, records=[7, 8, 9], checkpoints=[3], sectors=[2, 2], repairs=[3]),
    ]
    views = build_neutral_observer_views(observer_views)

    distance = neutral_distance_matrix(views)

    assert distance.shape == (3, 3)
    assert np.allclose(distance, distance.T)
    assert np.allclose(np.diag(distance), 0.0)
    assert np.all(distance[np.triu_indices(3, k=1)] >= 0.0)


def test_overlap_feature_distance_matrix_matches_literal_shared_mass_definition():
    features = np.array(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.5, 1.5, 0.0, 0.0],
            [0.0, 1.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    distance = _overlap_feature_distance_matrix(features)
    expected = np.zeros((features.shape[0], features.shape[0]), dtype=float)
    masses = np.sum(np.maximum(features, 0.0), axis=1)
    for i in range(features.shape[0]):
        for j in range(i + 1, features.shape[0]):
            denom = max(0.5 * (float(masses[i]) + float(masses[j])), 1.0e-12)
            similarity = float(np.sum(np.minimum(features[i], features[j])) / denom)
            expected[i, j] = expected[j, i] = 1.0 - max(0.0, min(1.0, similarity))

    assert np.allclose(distance, expected)
    assert np.allclose(distance, distance.T)
    assert np.allclose(np.diag(distance), 0.0)


def test_double_center_squared_distance_matches_projection_formula():
    distance = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ],
        dtype=float,
    )
    squared = distance**2
    projector = np.eye(3) - np.ones((3, 3), dtype=float) / 3.0
    expected = -0.5 * projector @ squared @ projector

    assert np.allclose(_double_center_squared_distance(squared), expected)


def test_primary_neutral_distance_uses_fixed_embedding_metric_without_rank_prefixes():
    observer_views = [
        _observer_view(1, records=[1, 1, 2], checkpoints=[1], sectors=[0, 1], repairs=[1]),
        _observer_view(2, records=[2, 3, 3], checkpoints=[2], sectors=[1, 1], repairs=[2]),
        _observer_view(3, records=[7, 8, 9], checkpoints=[3], sectors=[2, 2], repairs=[3]),
        _observer_view(4, records=[4, 5, 6], checkpoints=[4], sectors=[3, 3], repairs=[4]),
    ]
    views = build_neutral_observer_views(observer_views)

    features = neutral_feature_matrix(views)
    distance = neutral_distance_matrix(views)

    assert features.shape[0] == 4
    assert distance.shape == (4, 4)
    assert np.allclose(distance, distance.T)
    assert "modular_response" not in DEFAULT_NEUTRAL_WEIGHTS
    assert "prime_geometric_modular" not in DEFAULT_NEUTRAL_WEIGHTS
    assert "prime_geometric_control_quotient" not in DEFAULT_NEUTRAL_WEIGHTS
    assert "repair_modular" not in DEFAULT_NEUTRAL_WEIGHTS
    assert "prime_geometric_rank3" not in DEFAULT_NEUTRAL_WEIGHTS
    assert "transition_token" not in DEFAULT_NEUTRAL_WEIGHTS
    assert "support_visible_modular" not in DEFAULT_NEUTRAL_WEIGHTS


def test_duplicate_primary_channel_audit_blocks_identical_weighted_channels():
    observer_views = []
    rng = np.random.default_rng(123)
    for index in range(8):
        row = _observer_view(
            index,
            records=[index, index + 1],
            checkpoints=[index % 4],
            sectors=[index % 3],
            repairs=[index % 5],
        )
        spectrum = [float(index + j) for j in range(64)]
        row["prime_geometric_modular_spectrum"] = spectrum
        row["prime_geometric_control_quotient_spectrum"] = rng.normal(size=64).tolist()
        row["support_visible_modular_spectrum"] = spectrum
        observer_views.append(row)
    views = build_neutral_observer_views(observer_views)

    forbidden_prime_audit = neutral_channel_duplicate_audit(
        views,
        weights={"prime_geometric_modular": 1.0, "prime_geometric_control_quotient": 1.0},
    )
    diagnostic_audit = neutral_channel_duplicate_audit(
        views,
        weights={"prime_geometric_modular": 1.0, "support_visible_modular": 1.0},
    )

    assert forbidden_prime_audit["feature_ancestry_gate_pass"] is False
    assert forbidden_prime_audit["duplicate_channel_gate_pass"] is False
    assert diagnostic_audit["duplicate_channel_gate_pass"] is False
    assert diagnostic_audit["duplicate_pairs"]


def test_strict_neutral_feature_ancestry_blocks_support_visible_weights():
    observer_views = [
        _observer_view(1, records=[1, 1, 2], checkpoints=[1], sectors=[0, 1], repairs=[1]),
        _observer_view(2, records=[2, 3, 3], checkpoints=[2], sectors=[1, 1], repairs=[2]),
        _observer_view(3, records=[7, 8, 9], checkpoints=[3], sectors=[2, 2], repairs=[3]),
        _observer_view(4, records=[4, 5, 6], checkpoints=[4], sectors=[3, 3], repairs=[4]),
    ]
    views = build_neutral_observer_views(observer_views)

    audit = neutral_channel_duplicate_audit(
        views,
        weights={"record": 1.0, "prime_geometric_modular": 1.0},
    )

    assert audit["feature_ancestry_gate_pass"] is False
    assert audit["duplicate_channel_gate_pass"] is False
    assert audit["feature_ancestry_blockers"] == [
        "forbidden_strict_neutral_feature_ancestry:prime_geometric_modular:prime_geometric_response"
    ]


def test_primary_neutral_distance_is_invariant_under_transition_token_relabeling():
    observer_views = [
        _observer_view(i, records=[i, i + 1], checkpoints=[i % 4], sectors=[i % 3], repairs=[i % 5])
        for i in range(8)
    ]
    relabeled = []
    for row in observer_views:
        copy = json.loads(json.dumps(row))
        histograms = copy["transition_history_histograms"]
        token_hist = histograms.get("local_transition_token", {})
        histograms["local_transition_token"] = {
            str(900000 + int(key)): value for key, value in token_hist.items()
        }
        histograms["local_transition_token_persistent"] = {
            str(800000 + int(key)): value for key, value in token_hist.items()
        }
        relabeled.append(copy)

    original_distance = neutral_distance_matrix(build_neutral_observer_views(observer_views))
    relabeled_distance = neutral_distance_matrix(build_neutral_observer_views(relabeled))

    assert np.allclose(original_distance, relabeled_distance)


def test_neutral_leakage_audit_is_posthoc_and_flags_primary_no_geometry_use():
    observer_views = [
        _observer_view(1, records=[1, 2], checkpoints=[1], sectors=[0], repairs=[1], axis=[1.0, 0.0, 0.0]),
        _observer_view(2, records=[2, 3], checkpoints=[2], sectors=[1], repairs=[2], axis=[0.0, 1.0, 0.0]),
        _observer_view(3, records=[3, 4], checkpoints=[3], sectors=[2], repairs=[3], axis=[0.0, 0.0, 1.0]),
    ]
    views = build_neutral_observer_views(observer_views)
    distance = neutral_distance_matrix(views)

    audit = neutral_leakage_audit(distance, observer_views)

    assert audit["h3_coordinates_used"] is False
    assert audit["cap_normals_used"] is False
    assert audit["screen_axes_used_in_primary_distance"] is False


def test_neutral_leakage_audit_fails_closed_when_axes_are_missing():
    audit = neutral_leakage_audit(np.zeros((2, 2)), [{"view_type": "patch_observer"}] * 2)

    assert audit["s2_leakage_audit_available"] is False
    assert audit["s2_leakage_pass"] is False
    assert "s2_distance_correlation" in audit


def test_strict_neutral_bulk_receipt_requires_refinement():
    receipt = strict_neutral_bulk_receipt(
        dimension={"estimators_agree_3d": True},
        model_selection={
            "best_model": "H3",
            "h3_beats_s2": True,
            "h3_beats_h2_h4": True,
        },
        leakage={"s2_leakage_pass": True},
        controls={
            "shuffled_records_fail": True,
            "shuffled_transition_labels_fail": True,
            "planted_2d_returns_2d": True,
            "planted_3d_returns_3d": True,
            "planted_h3_returns_h3": True,
        },
        refinement={},
    )

    assert receipt["receipt"] == "STRICT_NEUTRAL_BULK_RECEIPT"
    assert receipt["strict_neutral_bulk"] is False
    assert receipt["physical_claim"] is False


def test_strict_neutral_bulk_receipt_can_pass_all_gates():
    quotient_contract = quotient_geometry_certificate(
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        quotient_ids=["a", "b"],
        channel_manifest=[ChannelMetricSpec(name="record")],
        jointly_separating=True,
        atlas_receipt={
            "identity_defect": 0.0,
            "inverse_defect": 0.0,
            "cocycle_defect": 0.0,
            "cycle_holonomy_defect": 0.0,
        },
        feature_receipt={
            "max_transport_defect": 0.0,
            "quotient_visible_missingness": True,
        },
        invariance_receipt={
            "gauge_distortion": 0.0,
            "port_distortion": 0.0,
            "order_distortion": 0.0,
            "schedule_distortion": 0.0,
            "partition_distortion": 0.0,
        },
        refinement_receipt=_canonical_refinement_receipt(),
        statistics_receipt={
            "ancestry_leakage_count": 0,
            "test_used_once": True,
            "positive_controls_passed": True,
            "negative_controls_passed": True,
        },
        provenance_records=_quotient_provenance(2),
    )
    receipt = strict_neutral_bulk_receipt(
        dimension={"estimators_agree_3d": True},
        model_selection={
            "best_model": "H3",
            "h3_beats_s2": True,
            "h3_beats_h2_h4": True,
        },
        leakage={"s2_leakage_pass": True},
        controls={
            "shuffled_records_fail": True,
            "shuffled_transition_labels_fail": True,
            "planted_2d_returns_2d": True,
            "planted_3d_returns_3d": True,
            "planted_h3_returns_h3": True,
        },
        refinement=_canonical_refinement_receipt(),
        quotient_geometry=quotient_contract,
        channel_audit={
            "duplicate_channel_gate_pass": True,
            "feature_ancestry_gate_pass": True,
        },
        theory_alignment={"theory_required_channels_present": True},
    )

    assert receipt["strict_neutral_bulk"] is True
    assert receipt["physical_claim"] is True
    assert receipt["QUOTIENT_GEOMETRY_CONTRACT_RECEIPT"] is True


def test_strict_neutral_bulk_receipt_rejects_truthy_strings() -> None:
    receipt = strict_neutral_bulk_receipt(
        dimension={"estimators_agree_3d": "true"},
        model_selection={
            "best_model": "H3",
            "h3_beats_s2": "true",
            "h3_beats_h2_h4": "true",
        },
        leakage={"s2_leakage_pass": "true"},
        controls={
            "shuffled_records_fail": "true",
            "shuffled_transition_labels_fail": "true",
            "planted_2d_returns_2d": "true",
            "planted_3d_returns_3d": "true",
            "planted_h3_returns_h3": "true",
        },
        refinement=_canonical_refinement_receipt(),
        quotient_geometry={"QUOTIENT_GEOMETRY_CONTRACT_RECEIPT": "true"},
        channel_audit={
            "duplicate_channel_gate_pass": "true",
            "feature_ancestry_gate_pass": "true",
        },
        theory_alignment={"theory_required_channels_present": "true"},
    )

    assert receipt["strict_neutral_bulk"] is False
    assert receipt["physical_claim"] is False


def test_strict_neutral_bulk_receipt_includes_feature_ancestry_gate() -> None:
    receipt = strict_neutral_bulk_receipt(
        dimension={"estimators_agree_3d": True},
        model_selection={
            "best_model": "H3",
            "h3_beats_s2": True,
            "h3_beats_h2_h4": True,
        },
        leakage={"s2_leakage_pass": True},
        controls={
            "shuffled_records_fail": True,
            "shuffled_transition_labels_fail": True,
            "planted_2d_returns_2d": True,
            "planted_3d_returns_3d": True,
            "planted_h3_returns_h3": True,
        },
        refinement=_canonical_refinement_receipt(),
        quotient_geometry={"QUOTIENT_GEOMETRY_CONTRACT_RECEIPT": True},
        channel_audit={
            "duplicate_channel_gate_pass": True,
            "feature_ancestry_gate_pass": False,
        },
        theory_alignment={"theory_required_channels_present": True},
    )

    assert receipt["feature_ancestry_gate_pass"] is False
    assert receipt["strict_neutral_bulk"] is False
    assert receipt["physical_claim"] is False


def test_strict_neutral_bulk_receipt_blocks_without_quotient_contract():
    receipt = strict_neutral_bulk_receipt(
        dimension={"estimators_agree_3d": True},
        model_selection={
            "best_model": "H3",
            "h3_beats_s2": True,
            "h3_beats_h2_h4": True,
        },
        leakage={"s2_leakage_pass": True},
        controls={
            "shuffled_records_fail": True,
            "shuffled_transition_labels_fail": True,
            "planted_2d_returns_2d": True,
            "planted_3d_returns_3d": True,
            "planted_h3_returns_h3": True,
        },
        refinement=_canonical_refinement_receipt(),
    )

    assert receipt["strict_neutral_bulk"] is False
    assert receipt["QUOTIENT_GEOMETRY_CONTRACT_RECEIPT"] is False


def test_dimension_gate_uses_finite_regulator_median_for_planted_3d():
    report = planted_neutral_control_report(point_count=160, seed=372, max_points=128)

    dimension = report["rows"]["planted_3d"]["dimension"]

    assert dimension["diagnostic_target"] == "neutral_record_feature_quotient_dimension"
    assert dimension["not_the_support_visible_chart_dimension"] is True
    assert report["controls"]["planted_3d_returns_3d"] is True
    assert dimension["estimators_agree_3d"] is True
    assert 2.7 <= dimension["median_dimension_estimate"] <= 3.3


def test_current_rank3_candidate_not_strict_without_refinement():
    report = strict_neutral_bulk_report(
        [_observer_view(i, records=[i, i + 1, i + 2], checkpoints=[i], sectors=[i % 3], repairs=[i % 5]) for i in range(12)],
        model_selection={"best_model": "H3", "h3_beats_s2": True, "h3_beats_h2_h4": True},
        controls={
            "shuffled_records_fail": True,
            "shuffled_transition_labels_fail": True,
            "planted_2d_returns_2d": True,
            "planted_3d_returns_3d": True,
            "planted_h3_returns_h3": True,
        },
        refinement={},
    )

    assert report["strict_neutral_bulk"] is False
    assert report["receipt"]["strict_neutral_bulk"] is False
    assert "H3 fitted points" in report["forbidden_primary_features"]
    assert "support-visible S2/H3 chart" in report["chart_boundary"]


def test_primary_features_are_locality_preserving_and_legacy_lanes_are_demoted():
    report = strict_neutral_bulk_report(
        [_observer_view(i, records=[i, i + 1], checkpoints=[i], sectors=[i % 3], repairs=[i % 5]) for i in range(8)],
        refinement={},
    )

    primary = report["primary_features"]
    assert primary == [
        "locality_preserving_packet_feature_vector",
        "measured_overlap_correspondences",
        "paired_perturbation_response_tensor",
    ]
    # Legacy hash-token/self-synthesized/repackaged lanes are demoted, not deleted.
    legacy = report["legacy_diagnostic_features"]
    assert "boundary_packet_hash_hist" in legacy
    assert "overlap_correspondence_hist_self_synthesized" in legacy
    assert "perturbation_response_tensor_repackaged" in legacy
    for name in primary:
        assert name not in legacy
    notes = report["legacy_diagnostic_feature_notes"]
    assert "self-synthesized" in notes["overlap_correspondence_hist_self_synthesized"]
    assert "not a perturb-resettle measurement" in notes["perturbation_response_tensor_repackaged"]
    assert "hash-token" in notes["boundary_packet_hash_hist"]
    primary_notes = report["primary_feature_notes"]
    assert "literal_support_intersection_v1" in primary_notes["measured_overlap_correspondences"]
    assert "hash" in primary_notes["locality_preserving_packet_feature_vector"]


def test_shuffled_neutral_control_report_emits_run_specific_controls():
    observer_views = [
        _observer_view(
            i,
            records=[i, i + 1, i + 2, i + 3],
            checkpoints=[i % 5, (i + 1) % 5],
            sectors=[i % 3, (i + 1) % 3],
            repairs=[i % 7, (i + 2) % 7],
        )
        for i in range(24)
    ]

    report = shuffled_neutral_control_report(observer_views, seed=7, max_model_points=24)

    assert report["mode"] == "strict_neutral_run_specific_shuffled_controls_v0"
    assert report["observer_count"] == 24
    assert set(report["controls"]) == {"shuffled_records_fail", "shuffled_transition_labels_fail"}
    assert set(report["rows"]) == {"shuffled_records", "shuffled_transition_labels"}
    for row in report["rows"].values():
        assert "distance_shape_correlation_to_original" in row
        assert "expected_failure_observed" in row


def test_overlap_native_neutral_control_report_uses_observer_overlap_substrate(tmp_path: Path):
    observer_views = [
        _observer_view(
            i,
            records=[i % 9, (i + 1) % 9, (i + 2) % 9, (i + 3) % 9],
            checkpoints=[i % 5, (i + 1) % 5],
            sectors=[i % 3, (i + 1) % 3],
            repairs=[i % 7, (i + 2) % 7],
        )
        for i in range(32)
    ]

    report = overlap_native_neutral_control_report(observer_views, seed=9, max_model_points=32)

    assert report["mode"] == "overlap_native_neutral_control_v0"
    assert report["observer_count"] == 32
    assert report["fundamental_operation"].startswith("Overlapping observations")
    assert set(row["control"] for row in report["control_rows"]) == {
        "degree_preserving_overlap_graph_rewire",
        "overlap_edge_weight_permutation",
        "columnwise_histogram_null",
    }
    assert report["strict_neutral_bulk"] is False
    assert report["physical_claim"] is False
    assert report["parallel_execution"]["effective_n_jobs"] == 1
    assert report["parallel_execution"]["ordered_result_assembly"] is True
    for row in report["control_rows"]:
        assert "distance_shape_correlation_to_original" in row
        assert "expected_failure_observed" in row

    run = tmp_path / "run"
    run.mkdir()
    with (run / "observer_views.jsonl").open("w", encoding="utf-8") as handle:
        for row in observer_views:
            handle.write(json.dumps(row) + "\n")
    written = write_overlap_native_neutral_control_report(run, tmp_path / "out", seed=9, max_model_points=32)

    assert written["mode"] == "overlap_native_neutral_control_v0"
    assert (tmp_path / "out" / "overlap_native_neutral_control_report.json").exists()
    assert (tmp_path / "out" / "overlap_native_neutral_control_report.md").exists()


def test_overlap_native_controls_use_ordered_bounded_parallel_refits(monkeypatch):
    observer_views = [
        _observer_view(
            i,
            records=[i % 9, (i + 1) % 9, (i + 2) % 9, (i + 3) % 9],
            checkpoints=[i % 5, (i + 1) % 5],
            sectors=[i % 3, (i + 1) % 3],
            repairs=[i % 7, (i + 2) % 7],
        )
        for i in range(12)
    ]
    monkeypatch.setenv("OPH_FPE_GRAPH_SWEEP_WORKERS", "3")

    report = overlap_native_neutral_control_report(
        observer_views,
        seed=9,
        max_model_points=12,
    )

    assert report["parallel_execution"]["effective_n_jobs"] == 3
    assert [row["control"] for row in report["control_rows"]] == [
        "degree_preserving_overlap_graph_rewire",
        "overlap_edge_weight_permutation",
        "columnwise_histogram_null",
    ]


def test_measured_overlap_presentation_invariance_ignores_external_cohort_peers():
    observer_views = [
        {
            "view_type": "patch_observer",
            "observer_id": observer_id,
            "overlap_correspondence_evidence_provenance": {
                "cross_observer_measurement": True,
                "self_histogram_synthesis": False,
            },
            "measured_overlap_correspondences": [
                {
                    "peer_observer_id": peer_id,
                    "measured_affinity": 0.75 if peer_id < 8 else 0.25,
                }
                for peer_id in range(16)
                if peer_id != observer_id
            ],
        }
        for observer_id in range(8)
    ]

    report = _measured_overlap_presentation_invariance_report(
        observer_views,
        np.random.default_rng(91),
    )

    assert report["available"] is True
    assert report["receipt"] is True
    assert report["global_observer_relabel_affinity_distortion"] <= 1.0e-12
    assert report["global_observer_relabel_distortion"] <= 1.0e-12


def test_overlap_native_graph_geometry_report_uses_observer_overlap_graph(tmp_path: Path):
    observer_views = [
        _observer_view(
            i,
            records=[i % 9, (i + 1) % 9, (i + 2) % 9, (i + 3) % 9],
            checkpoints=[i % 5, (i + 1) % 5],
            sectors=[i % 3, (i + 1) % 3],
            repairs=[i % 7, (i + 2) % 7],
        )
        for i in range(32)
    ]

    report = overlap_native_graph_geometry_report(
        observer_views,
        seed=11,
        max_model_points=32,
        k_neighbors=8,
    )

    assert report["mode"] == "overlap_native_graph_geometry_v0"
    assert report["observer_count"] == 32
    assert report["fundamental_operation"].startswith("Overlapping observations")
    assert report["graph_summary"]["edge_count"] > 0
    assert "dimension" in report
    assert "model_selection" in report
    assert "rank_selection" in report
    assert set(row["control"] for row in report["control_rows"]) == {
        "degree_preserving_overlap_graph_rewire",
        "overlap_edge_weight_permutation",
        "columnwise_histogram_null",
    }
    assert report["strict_neutral_bulk"] is False
    assert report["physical_claim"] is False

    run = tmp_path / "run"
    run.mkdir()
    with (run / "observer_views.jsonl").open("w", encoding="utf-8") as handle:
        for row in observer_views:
            handle.write(json.dumps(row) + "\n")
    written = write_overlap_native_graph_geometry_report(
        run,
        tmp_path / "graph",
        seed=11,
        max_model_points=32,
        k_neighbors=8,
    )

    assert written["mode"] == "overlap_native_graph_geometry_v0"
    assert (tmp_path / "graph" / "overlap_native_graph_geometry_report.json").exists()
    assert (tmp_path / "graph" / "overlap_native_graph_geometry_report.md").exists()


def test_overlap_native_graph_geometry_sweep_writes_cases_and_summary(tmp_path: Path):
    observer_views = [
        _observer_view(
            i,
            records=[i % 7, (i + 1) % 7, (i + 2) % 7],
            checkpoints=[i % 4, (i + 1) % 4],
            sectors=[i % 3],
            repairs=[i % 5],
        )
        for i in range(18)
    ]
    run = tmp_path / "run"
    run.mkdir()
    with (run / "observer_views.jsonl").open("w", encoding="utf-8") as handle:
        for row in observer_views:
            handle.write(json.dumps(row) + "\n")

    report = overlap_native_graph_geometry_sweep_report(
        [run],
        seeds=(3,),
        max_model_points_values=(18,),
        k_neighbor_values=(4, 6),
    )
    written = write_overlap_native_graph_geometry_sweep_report(
        [run],
        tmp_path / "sweep",
        seeds=(3,),
        max_model_points_values=(18,),
        k_neighbor_values=(4, 6),
    )

    assert report["mode"] == "overlap_native_graph_geometry_sweep_v0"
    assert report["case_count"] == 2
    assert report["strict_neutral_bulk"] is False
    assert report["physical_claim"] is False
    assert "best_case" in report
    assert report["rank_obstruction_summary"]["available"] is True
    assert report["rank_obstruction_summary"]["case_count"] == 2
    assert "dominant_largest_gap_rank" in report["rank_obstruction_summary"]
    assert report["gate_coincidence_summary"]["available"] is True
    assert report["gate_coincidence_summary"]["case_count"] == 2
    assert "spatial_h3_nontrivial_rank3_selector_count" in report["gate_coincidence_summary"]
    assert len(report["closest_strict_rows"]) == 2
    assert "missing_strict_gates" in report["closest_strict_rows"][0]
    assert "gate_status" in report["closest_strict_rows"][0]
    assert "rank3_selector" in report["closest_strict_rows"][0]
    assert "nontrivial_rank3_selector" in report["closest_strict_rows"][0]
    assert "strict_h3_candidate" in report["closest_strict_rows"][0]
    assert written["case_count"] == 2
    assert len(written["closest_strict_rows"]) == 2
    assert (tmp_path / "sweep" / "overlap_native_graph_geometry_sweep_report.json").exists()
    assert (tmp_path / "sweep" / "overlap_native_graph_geometry_sweep_report.md").exists()
    assert (tmp_path / "sweep" / "overlap_native_graph_geometry_sweep_rows.csv").exists()
    assert len(list((tmp_path / "sweep" / "overlap_graph_cases").glob("*/overlap_native_graph_geometry_report.json"))) == 2


def test_overlap_residualized_graph_geometry_sweep_writes_common_mode_diagnostics(tmp_path: Path):
    observer_views = [
        _observer_view(
            i,
            records=[i % 7, (i + 1) % 7, (i + 2) % 7],
            checkpoints=[i % 4, (i + 1) % 4],
            sectors=[i % 3],
            repairs=[i % 5],
        )
        for i in range(18)
    ]
    run = tmp_path / "run"
    run.mkdir()
    with (run / "observer_views.jsonl").open("w", encoding="utf-8") as handle:
        for row in observer_views:
            handle.write(json.dumps(row) + "\n")

    single = overlap_residualized_graph_geometry_report(
        observer_views,
        seed=3,
        max_model_points=18,
        k_neighbors=4,
        remove_modes=1,
    )
    written_single = write_overlap_residualized_graph_geometry_report(
        run,
        tmp_path / "single",
        seed=3,
        max_model_points=18,
        k_neighbors=4,
        remove_modes=1,
    )
    report = overlap_residualized_graph_geometry_sweep_report(
        [run],
        seeds=(3,),
        max_model_points_values=(18,),
        k_neighbor_values=(4, 6),
        remove_mode_values=(0, 1),
    )
    written = write_overlap_residualized_graph_geometry_sweep_report(
        [run],
        tmp_path / "residual_sweep",
        seeds=(3,),
        max_model_points_values=(18,),
        k_neighbor_values=(4, 6),
        remove_mode_values=(0, 1),
    )

    assert single["mode"] == "overlap_residualized_graph_geometry_v0"
    assert single["fundamental_operation"].startswith("Overlapping observations")
    assert single["residualization"]["method"] == "column_center_then_remove_leading_svd_modes"
    assert "raw_largest_gap_rank" in single["residualization"]
    assert single["strict_neutral_bulk"] is False
    assert single["physical_claim"] is False
    assert written_single["mode"] == "overlap_residualized_graph_geometry_v0"
    assert (tmp_path / "single" / "overlap_residualized_graph_geometry_report.json").exists()
    assert (tmp_path / "single" / "overlap_residualized_graph_geometry_report.md").exists()
    assert report["mode"] == "overlap_residualized_graph_geometry_sweep_v0"
    assert report["case_count"] == 4
    assert report["strict_neutral_bulk"] is False
    assert report["physical_claim"] is False
    assert report["rank_obstruction_summary"]["available"] is True
    assert report["rank_obstruction_summary"]["case_count"] == 4
    assert "raw_largest_gap_rank1_count" in report["rank_obstruction_summary"]
    assert report["gate_coincidence_summary"]["available"] is True
    assert report["gate_coincidence_summary"]["case_count"] == 4
    assert "spatial_h3_nontrivial_rank3_selector_count" in report["gate_coincidence_summary"]
    assert len(report["closest_strict_rows"]) == 4
    assert "missing_strict_gates" in report["closest_strict_rows"][0]
    assert "gate_status" in report["closest_strict_rows"][0]
    assert "rank3_selector" in report["closest_strict_rows"][0]
    assert "nontrivial_rank3_selector" in report["closest_strict_rows"][0]
    assert "strict_h3_candidate" in report["closest_strict_rows"][0]
    assert written["case_count"] == 4
    assert len(written["closest_strict_rows"]) == 4
    assert (tmp_path / "residual_sweep" / "overlap_residualized_graph_geometry_sweep_report.json").exists()
    assert (tmp_path / "residual_sweep" / "overlap_residualized_graph_geometry_sweep_report.md").exists()
    assert (tmp_path / "residual_sweep" / "overlap_residualized_graph_geometry_sweep_rows.csv").exists()
    assert len(
        list(
            (tmp_path / "residual_sweep" / "overlap_residual_graph_cases").glob(
                "*/overlap_residualized_graph_geometry_report.json"
            )
        )
    ) == 4


def test_neutral_model_selection_reports_required_metric_families():
    rng = np.random.default_rng(12)
    coords = rng.normal(size=(48, 3))
    distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)

    report = neutral_model_selection(distance, seed=2, max_points=48)

    assert set(report["models"]) == {"S2", "E2", "E3", "E4", "H2", "H3", "H4"}
    assert report["best_model"] in report["models"]
    assert report["fit_observer_count"] == 48
    assert report["heldout_pair_count"] > 0


def test_planted_neutral_controls_are_diagnostic_and_include_hard_flags():
    report = planted_neutral_control_report(point_count=48, seed=3, max_points=48)

    assert set(report["rows"]) == {"planted_2d", "planted_3d", "planted_4d", "planted_h3"}
    assert "planted_2d_returns_2d" in report["controls"]
    assert "planted_3d_returns_3d" in report["controls"]
    assert "planted_4d_returns_4d" in report["controls"]
    assert "planted_h3_returns_h3" in report["controls"]
    assert isinstance(report["controls"]["planted_h3_returns_h3"], bool)


def test_neutral_profile_audit_is_diagnostic_not_bulk_claim():
    observer_views = [
        _observer_view(
            i,
            records=[i, i + 1, i + 2],
            checkpoints=[i % 4],
            sectors=[i % 3],
            repairs=[i % 5],
            axis=[1.0, 0.0, 0.0],
        )
        for i in range(16)
    ]

    report = neutral_profile_audit_report(observer_views, seed=4, sample_count=16, max_model_points=16)

    assert report["mode"] == "neutral_distance_profile_audit_v0"
    assert report["observer_count"] == 16
    assert report["strict_neutral_bulk"] is False
    assert report["physical_claim"] is False
    assert {row["profile"] for row in report["profile_rows"]} == {
        "all_observer_visible",
        "strict_record_repair_only",
        "overlap_record_repair_only",
        "support_visible_prime_geometric_diagnostic",
        "scalar_only",
        "transition_core",
        "scalar_response",
        "prime_geometric_modular",
        "prime_geometric_control_quotient",
        "prime_geometric_rank3",
        "prime_geometric_rank4",
        "prime_geometric_rank8",
        "prime_geometric_rank16",
        "prime_geometric_rank32",
        "prime_geometric_control_quotient_rank3",
        "prime_geometric_control_quotient_rank4",
        "prime_geometric_control_quotient_rank8",
        "prime_geometric_control_quotient_rank16",
        "prime_geometric_control_quotient_rank32",
        "prime_geometric_modular_counterfactual",
        "prime_geometric_control_quotient_counterfactual",
        "support_visible_modular",
        "support_visible_modular_scalar",
        "repair_modular_only",
    }
    for row in report["profile_rows"]:
        assert "blockers" in row
        assert "dimension" in row
        assert "model_selection" in row


def test_prime_geometric_rank_sweep_is_diagnostic_not_bulk_claim():
    observer_views = [
        _observer_view(
            i,
            records=[i, i + 1, i + 2],
            checkpoints=[i % 4],
            sectors=[i % 3],
            repairs=[i % 5],
            axis=[1.0, 0.0, 0.0],
        )
        for i in range(18)
    ]

    report = prime_geometric_rank_sweep_report(
        observer_views,
        ranks=(2, 3, 4),
        seed=4,
        sample_count=18,
        max_model_points=18,
    )

    assert report["mode"] == "prime_geometric_rank_sweep_v0"
    assert report["observer_count"] == 18
    assert report["physical_claim"] is False
    assert [row["rank"] for row in report["rows"]] == [2, 3, 4]
    assert [row["rank"] for row in report["control_quotient_rows"]] == [2, 3, 4]
    assert [row["rank"] for row in report["coordinate_rows"]] == [2, 3, 4]
    assert [row["rank"] for row in report["control_quotient_coordinate_rows"]] == [2, 3, 4]
    assert "best_dimension_row" in report
    assert "control_quotient_best_dimension_row" in report
    assert "coordinate_best_dimension_row" in report
    assert "control_quotient_coordinate_best_dimension_row" in report
    assert "control_quotient_coordinate_best_3d_dimension_row" in report
    assert "best_3d_dimension_row" in report
    assert "coordinate_best_3d_dimension_row" in report
    assert report["regulator_control_quotient_lane"]["is_negative_control"] is False
    assert report["prime_geometric_quotient_3d_diagnostic_receipt"] in {True, False}
    assert report["prime_geometric_control_quotient_spatial_3d_candidate_receipt"] in {True, False}
    assert report["prime_geometric_strict_neutral_candidate_receipt"] is False
    assert report["selected_rank_controls"]["mode"] == "prime_geometric_selected_rank_null_controls_v0"
    assert isinstance(report["selected_rank_controls"]["control_rows"], list)
    assert "proof_blockers" in report
    assert "control_quotient_lane_is_not_a_negative_control" in report["proof_blockers"]
    for row in report["rows"] + report["coordinate_rows"]:
        assert "dimension" in row
        assert "model_selection" in row
        assert "leakage" in row
    for row in report["coordinate_rows"]:
        assert row["distance_metric"] == "median_normalized_euclidean_on_response_coordinates"
        assert "spatial_3d_ready" in row


def test_prime_selected_rank_controls_exclude_coordinate_rank3_tautology_from_gate():
    observer_views = [
        _observer_view(
            i,
            records=[i, i + 1, i + 2],
            checkpoints=[i % 4],
            sectors=[i % 3],
            repairs=[i % 5],
            axis=[1.0, 0.0, 0.0],
        )
        for i in range(24)
    ]
    neutral_views = build_neutral_observer_views(observer_views)

    report = _prime_geometric_selected_rank_controls(
        neutral_views,
        observer_views,
        target_rank=None,
        coordinate_rank=3,
        seed=44,
        max_model_points=24,
    )

    assert report["non_tautological_control_count"] == 0
    assert report["all_expected_failures_observed"] is False
    assert report["coordinate_rank3_tautology_warning"] in {True, False}


def test_prime_rank_sweep_writer_carries_independent_rank_selector(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    observer_views = [
        _observer_view(
            i,
            records=[i, i + 1, i + 2],
            checkpoints=[i % 4],
            sectors=[i % 3],
            repairs=[i % 5],
            axis=[1.0, 0.0, 0.0],
        )
        for i in range(18)
    ]
    with (run / "observer_views.jsonl").open("w", encoding="utf-8") as handle:
        for row in observer_views:
            handle.write(json.dumps(row) + "\n")
    (run / "prime_geometric_response_attachment_report.json").write_text(
        json.dumps(
            {
                "prime_geometric_control_quotient": {
                    "embedding": {
                        "rank_selection": {
                            "independent_rank3_selector_receipt": True,
                            "largest_gap_rank": 3,
                            "chord_elbow_rank": 3,
                            "effective_rank": 3.4,
                            "participation_rank": 3.1,
                            "rank3_cumulative_explained_variance": 0.71,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    report = write_prime_geometric_rank_sweep_report(
        run,
        ranks=(2, 3, 4),
        seed=4,
        sample_count=18,
        max_model_points=18,
    )

    assert report["independent_rank3_selector_receipt"] is True
    assert report["independent_rank_selection"]["control_quotient_rank3_selector_receipt"] is True
    assert (
        "requires_independent_rank_selection_rule_before_physical_interpretation"
        not in report["proof_blockers"]
    )


def test_overlap_rank_selection_carries_model_order_diagnostics():
    rng = np.random.default_rng(123)
    coords = rng.normal(size=(32, 3))
    distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    sigma = float(np.median(distance[distance > 0.0]))
    affinity = np.exp(-(distance**2) / max(sigma**2, 1.0e-12))
    np.fill_diagonal(affinity, 0.0)

    report = _overlap_graph_rank_selection(affinity)

    assert report["available"] is True
    assert "model_order_selection" in report
    assert "nontrivial_model_order_selection" in report
    assert "model_order_rank3_selector_receipt" in report
    assert "nontrivial_model_order_rank3_selector_receipt" in report
    assert report["model_order_selection"]["available"] is True
    assert report["model_order_selection"]["profile_likelihood_rank"] is not None
    assert report["model_order_selection"]["broken_stick_rank"] is not None


def test_prime_rank_refinement_candidate_is_not_strict_without_independent_rank(tmp_path: Path):
    def write_report(index: int, patch_count: int, median: float, corr: float, mle: float) -> None:
        source = tmp_path / f"source_{index}"
        source.mkdir()
        (source / "manifest.json").write_text(
            json.dumps({"patch_count": patch_count, "run_id": f"source_{index}"}),
            encoding="utf-8",
        )
        report_dir = tmp_path / f"report_{index}"
        report_dir.mkdir()
        payload = {
            "mode": "prime_geometric_rank_sweep_v0",
            "source_run_dir": str(source),
            "prime_geometric_control_quotient_spatial_3d_candidate_receipt": True,
            "independent_rank3_selector_receipt": False,
            "observer_count": 128,
            "sampled_observer_count": 128,
            "control_quotient_coordinate_best_3d_dimension_row": {
                "rank": 3,
                "dimension": {
                    "median_dimension_estimate": median,
                    "correlation_dimension": {"estimate": corr},
                    "local_mle_dimension": {"median_estimate": mle},
                },
                "model_selection": {"best_model": "E3"},
                "leakage": {
                    "s2_distance_correlation": 0.01,
                    "s2_leakage_pass": True,
                },
            },
            "proof_blockers": ["requires_independent_rank_selection_rule_before_physical_interpretation"],
        }
        (report_dir / "prime_geometric_rank_sweep_report.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    write_report(0, 65536, 2.86, 2.68, 3.04)
    write_report(1, 65536, 2.82, 2.66, 2.98)
    write_report(2, 262144, 2.83, 2.65, 3.02)
    write_report(3, 262144, 2.84, 2.67, 3.01)

    report = prime_geometric_rank_refinement_report([tmp_path])

    assert report["control_quotient_rank3_refinement_candidate_receipt"] is False
    assert report["required_ladder_complete"] is False
    assert report["missing_required_patch_counts"] == [4096, 16384]
    assert report["strict_neutral_bulk_refinement_receipt"] is False
    assert report["independent_rank3_selector_all"] is False
    assert report["candidate_dimension_stable"] is False
    assert report["candidate_dimension_drift"] < 0.1
    assert "independent_svd_rank3_selector_not_stable_or_false" in report["proof_blockers"]


def test_neutral_3d_bulk_audit_summarizes_rank3_candidate_blockers(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({"patch_count": 262144}), encoding="utf-8")
    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    refinement_dir = tmp_path / "refinement"
    refinement_dir.mkdir()
    overlap_dir = tmp_path / "overlap"
    overlap_dir.mkdir()
    sweep_payload = {
        "mode": "prime_geometric_rank_sweep_v0",
        "source_run_dir": str(source),
        "observer_count": 128,
        "sampled_observer_count": 128,
        "strict_3d_ready_count": 0,
        "coordinate_spatial_3d_ready_count": 0,
        "control_quotient_coordinate_spatial_3d_ready_count": 1,
        "prime_geometric_control_quotient_spatial_3d_candidate_receipt": True,
        "control_residualized_rank3_candidate_receipt": True,
        "control_quotient_coordinate_best_3d_dimension_row": {
            "rank": 3,
            "dimension": {"median_dimension_estimate": 2.84},
            "model_selection": {
                "best_model": "E3",
                "h3_beats_s2": True,
                "h3_beats_e3": False,
                "h3_beats_h2_h4": False,
            },
            "leakage": {"s2_distance_correlation": 0.01, "s2_leakage_pass": True},
        },
        "independent_rank_selection": {
            "prime_geometric": {"largest_gap_rank": 1, "chord_elbow_rank": 6},
            "control_quotient": {"largest_gap_rank": 53, "chord_elbow_rank": 6},
            "prime_rank3_selector_receipt": False,
            "control_quotient_rank3_selector_receipt": False,
        },
        "regulator_control_quotient_lane": {"is_negative_control": False},
        "proof_blockers": [
            "control_quotient_lane_is_not_a_negative_control",
            "requires_independent_rank_selection_rule_before_physical_interpretation",
        ],
    }
    refinement_payload = {
        "mode": "prime_geometric_rank_refinement_v0",
        "run_count": 1,
        "multi_scale": False,
        "control_quotient_rank3_refinement_candidate_receipt": True,
        "independent_rank3_selector_all": False,
        "candidate_dimension_drift": 0.015,
        "candidate_dimension_stable": True,
        "strict_neutral_bulk_refinement_receipt": False,
        "proof_blockers": [
            "independent_svd_rank3_selector_not_stable_or_false",
            "control_quotient_lane_is_not_a_negative_control",
            "directional_h3_strict_rank_gate_not_passed",
        ],
    }
    (sweep_dir / "prime_geometric_rank_sweep_report.json").write_text(
        json.dumps(sweep_payload),
        encoding="utf-8",
    )
    (refinement_dir / "prime_geometric_rank_refinement_report.json").write_text(
        json.dumps(refinement_payload),
        encoding="utf-8",
    )
    (overlap_dir / "overlap_native_neutral_control_report.json").write_text(
        json.dumps(
            {
                "mode": "overlap_native_neutral_control_v0",
                "source_run_dir": str(source),
                "observer_count": 128,
                "sampled_observer_count": 128,
                "OVERLAP_NATIVE_NEGATIVE_CONTROL_RECEIPT": True,
                "overlap_native_spatial_3d_candidate": False,
                "overlap_native_strict_h3_candidate": False,
                "blockers": ["overlap_native_distance_not_spatial_3d_candidate"],
            }
        ),
        encoding="utf-8",
    )

    report = neutral_3d_bulk_audit_report([tmp_path])
    written = write_neutral_3d_bulk_audit_report([tmp_path], tmp_path / "audit")

    assert report["control_residualized_rank3_refinement_candidate"] is True
    assert report["control_residualized_rank3_independent_selector_all"] is False
    assert report["strict_neutral_bulk_ready"] is False
    assert report["strict_neutral_bulk"] is False
    assert report["directional_strict_ready_total"] == 0
    assert report["control_quotient_candidate_count"] == 1
    assert report["overlap_native_negative_control_report_count"] == 1
    assert report["overlap_native_negative_control_receipt_count"] == 1
    assert report["overlap_native_negative_control_receipt_all"] is True
    assert "independent_svd_rank3_selector_not_stable_or_false" in report["blockers"]
    assert "control_quotient_lane_is_not_a_negative_control" in report["blockers"]
    assert report["sweeps"][0]["control_quotient_candidate"]["rank"] == 3
    assert written["mode"] == "neutral_3d_bulk_audit_v0"
    assert (tmp_path / "audit" / "neutral_3d_bulk_audit_report.json").exists()


def test_strict_neutral_bulk_frontier_summarizes_missing_receipts(tmp_path: Path):
    audit_dir = tmp_path / "audit"
    selector_dir = tmp_path / "selector"
    graph_dir = tmp_path / "graph"
    residual_dir = tmp_path / "residual"
    audit_dir.mkdir()
    selector_dir.mkdir()
    graph_dir.mkdir()
    residual_dir.mkdir()
    (audit_dir / "neutral_3d_bulk_audit_report.json").write_text(
        json.dumps(
            {
                "mode": "neutral_3d_bulk_audit_v0",
                "strict_neutral_bulk_ready": False,
                "strict_neutral_bulk": False,
                "control_residualized_rank3_refinement_candidate": True,
                "control_residualized_rank3_dimension_drift": 0.027,
                "directional_strict_ready_total": 0,
                "control_quotient_candidate_count": 4,
                "overlap_native_negative_control_report_count": 4,
                "overlap_native_negative_control_receipt_count": 4,
                "overlap_native_negative_control_receipt_all": True,
                "overlap_native_spatial_3d_candidate_count": 0,
                "refinement_summary": {
                    "candidate_dimension_stable": True,
                    "control_quotient_rank3_refinement_candidate_receipt": True,
                },
                "blockers": [
                    "independent_svd_rank3_selector_not_stable_or_false",
                    "directional_h3_strict_rank_gate_not_passed",
                ],
            }
        ),
        encoding="utf-8",
    )
    (selector_dir / "neutral_independent_rank_selector_audit_report.json").write_text(
        json.dumps(
            {
                "mode": "neutral_independent_rank_selector_audit_v0",
                "NEUTRAL_INDEPENDENT_RANK3_SELECTOR_RECEIPT": False,
                "run_count": 4,
                "control_quotient_rank3_candidate_count": 4,
                "control_quotient_rank3_selector_count": 0,
                "blockers": ["control_quotient_independent_rank3_selector_not_all_true"],
            }
        ),
        encoding="utf-8",
    )
    (graph_dir / "overlap_native_graph_geometry_report.json").write_text(
        json.dumps(
            {
                "mode": "overlap_native_graph_geometry_v0",
                "OVERLAP_NATIVE_GRAPH_GEOMETRY_RECEIPT": True,
                "overlap_graph_spatial_3d_candidate": True,
                "overlap_graph_strict_h3_candidate": False,
                "rank_selection": {
                    "rank3_selector_receipt": False,
                    "model_order_rank3_selector_receipt": True,
                    "nontrivial_model_order_rank3_selector_receipt": False,
                    "model_order_selection": {"consensus_rank": 3},
                    "nontrivial_model_order_selection": {"consensus_rank": 4},
                },
                "graph_summary": {"edge_count": 16, "component_count": 1},
                "blockers": ["overlap_graph_not_strict_h3_candidate"],
            }
        ),
        encoding="utf-8",
    )
    (residual_dir / "overlap_residualized_graph_geometry_report.json").write_text(
        json.dumps(
            {
                "mode": "overlap_residualized_graph_geometry_v0",
                "OVERLAP_RESIDUALIZED_GRAPH_GEOMETRY_RECEIPT": True,
                "overlap_residual_graph_spatial_3d_candidate": True,
                "overlap_residual_graph_strict_h3_candidate": False,
                "rank_selection": {
                    "rank3_selector_receipt": False,
                    "largest_gap_rank": 2,
                    "model_order_rank3_selector_receipt": False,
                    "nontrivial_model_order_rank3_selector_receipt": True,
                    "model_order_selection": {"consensus_rank": 2},
                    "nontrivial_model_order_selection": {"consensus_rank": 3},
                },
                "residualization": {"raw_largest_gap_rank": 1},
                "graph_summary": {"edge_count": 12, "component_count": 1},
                "blockers": ["overlap_residual_graph_not_strict_h3_candidate"],
            }
        ),
        encoding="utf-8",
    )

    report = strict_neutral_bulk_frontier_report([tmp_path])
    written = write_strict_neutral_bulk_frontier_report([tmp_path], tmp_path / "frontier")

    assert report["STRICT_NEUTRAL_BULK_FRONTIER_REPORT"] is True
    assert report["strict_neutral_bulk"] is False
    assert report["strict_neutral_bulk_ready"] is False
    assert report["control_residualized_rank3_refinement_candidate"] is True
    assert report["overlap_native_negative_control_receipt_all"] is True
    assert report["overlap_native_graph_geometry_report_count"] == 1
    assert report["overlap_native_graph_geometry_receipt_count"] == 1
    assert report["overlap_native_graph_spatial_3d_candidate_count"] == 1
    assert report["overlap_native_graph_strict_h3_candidate_count"] == 0
    assert report["overlap_native_graph_model_order_rank3_selector_count"] == 1
    assert report["overlap_native_graph_nontrivial_model_order_rank3_selector_count"] == 0
    assert report["overlap_residualized_graph_geometry_receipt_count"] == 1
    assert report["overlap_residualized_graph_spatial_3d_candidate_count"] == 1
    assert report["overlap_residualized_graph_strict_h3_candidate_count"] == 0
    assert report["overlap_residualized_graph_model_order_rank3_selector_count"] == 0
    assert report["overlap_residualized_graph_nontrivial_model_order_rank3_selector_count"] == 1
    assert report["neutral_independent_rank3_selector_receipt"] is False
    assert report["control_quotient_rank3_candidate_count"] == 4
    assert {row["gate"] for row in report["gate_rows"]} >= {
        "control_residualized_rank3_refinement_candidate",
        "independent_rank3_selector",
        "directional_h3_strict_gate",
        "overlap_native_graph_geometry",
        "overlap_native_graph_strict_h3",
        "overlap_residualized_graph_geometry",
        "overlap_residualized_graph_strict_h3",
    }
    assert "independent_svd_rank3_selector_not_stable_or_false" in report["blockers"]
    assert "overlap_graph_strict_h3_candidate_false" in report["blockers"]
    assert "overlap_residual_graph_strict_h3_candidate_false" in report["blockers"]
    assert written["mode"] == "strict_neutral_bulk_frontier_v0"
    assert (tmp_path / "frontier" / "strict_neutral_bulk_frontier_report.json").exists()
    assert (tmp_path / "frontier" / "strict_neutral_bulk_frontier_report.md").exists()


def test_neutral_independent_rank_selector_audit_keeps_rank3_candidate_diagnostic(tmp_path: Path):
    for index, effective_rank in enumerate([98.0, 154.0]):
        run = tmp_path / f"rank_sweep_{index}"
        run.mkdir()
        payload = {
            "mode": "prime_geometric_rank_sweep_v0",
            "source_run_dir": f"run_{index}",
            "observer_count": 1000,
            "sampled_observer_count": 256,
            "prime_geometric_control_quotient_spatial_3d_candidate_receipt": True,
            "control_residualized_rank3_candidate_receipt": True,
            "regulator_control_quotient_lane": {"is_negative_control": False},
            "independent_rank_selection": {
                "prime_rank3_selector_receipt": False,
                "control_quotient_rank3_selector_receipt": False,
                "prime_geometric": {
                    "largest_gap_rank": 2,
                    "chord_elbow_rank": 6 + index,
                    "effective_rank": 90.0 + index,
                    "participation_rank": 56.0 + index,
                    "rank3_cumulative_explained_variance": 0.13,
                },
                "control_quotient": {
                    "largest_gap_rank": 53,
                    "chord_elbow_rank": 6,
                    "effective_rank": effective_rank,
                    "participation_rank": 75.0,
                    "rank3_cumulative_explained_variance": 0.05,
                },
            },
            "proof_blockers": ["requires_independent_rank_selection_rule_before_physical_interpretation"],
        }
        (run / "prime_geometric_rank_sweep_report.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    report = neutral_independent_rank_selector_audit_report([tmp_path])
    written = write_neutral_independent_rank_selector_audit_report([tmp_path], tmp_path / "selector_audit")

    assert report["mode"] == "neutral_independent_rank_selector_audit_v0"
    assert report["run_count"] == 2
    assert report["NEUTRAL_INDEPENDENT_RANK3_SELECTOR_RECEIPT"] is False
    assert report["control_quotient_rank3_selector_count"] == 0
    assert report["control_quotient_rank3_candidate_count"] == 2
    assert report["control_quotient_median_effective_rank"] == 126.0
    assert "control_quotient_independent_rank3_selector_not_all_true" in report["blockers"]
    assert "control_quotient_effective_rank_not_low_dimensional" in report["blockers"]
    assert written["mode"] == "neutral_independent_rank_selector_audit_v0"
    assert (tmp_path / "selector_audit" / "neutral_independent_rank_selector_audit_report.json").exists()
    assert (tmp_path / "selector_audit" / "neutral_independent_rank_selector_audit_report.md").exists()


def test_neutral_profile_audit_appears_in_comparable_snapshot(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    observer_views = [
        _observer_view(
            i,
            records=[i, i + 1, i + 2],
            checkpoints=[i % 4],
            sectors=[i % 3],
            repairs=[i % 5],
            axis=[1.0, 0.0, 0.0],
        )
        for i in range(16)
    ]
    with (run / "observer_views.jsonl").open("w", encoding="utf-8") as handle:
        for row in observer_views:
            handle.write(json.dumps(row) + "\n")
    write_neutral_profile_audit_report(run, seed=4, sample_count=16, max_model_points=16)

    report = comparable_data_report([run])
    lane = report["measurement_lanes"]["neutral_distance_profile_audit"]

    assert lane["run_count"] == 1
    assert lane["mean_profile_count"] == 24
    assert lane["strict_3d_ready_count"] == 0


def test_write_neutral_profile_audit_can_limit_profiles(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    observer_views = [
        _observer_view(
            i,
            records=[i, i + 1, i + 2],
            checkpoints=[i % 4],
            sectors=[i % 3],
            repairs=[i % 5],
            axis=[1.0, 0.0, 0.0],
        )
        for i in range(16)
    ]
    with (run / "observer_views.jsonl").open("w", encoding="utf-8") as handle:
        for row in observer_views:
            handle.write(json.dumps(row) + "\n")

    report = write_neutral_profile_audit_report(
        run,
        seed=4,
        sample_count=16,
        max_model_points=16,
        profiles={
            "transition_core": {
                "record": 1.0,
                "checkpoint": 0.75,
                "sector": 0.75,
                "repair": 0.75,
                "transition_token": 1.0,
                "transition_token_persistent": 0.5,
            },
            "prime_geometric_rank3": {"prime_geometric_rank3": 1.0},
        },
    )

    assert [row["profile"] for row in report["profile_rows"]] == [
        "transition_core",
        "prime_geometric_rank3",
    ]


def _observer_view(
    observer_id: int,
    *,
    records: list[int],
    checkpoints: list[int],
    sectors: list[int],
    repairs: list[int],
    axis: list[float] | None = None,
    support_nodes: list[int] | None = None,
    h3_point: list[float] | None = None,
    cap_axis: list[float] | None = None,
    radial_depth: float | None = None,
    modular_depth: float | None = None,
) -> dict:
    steps = []
    count = max(len(records), len(checkpoints), len(sectors), len(repairs))
    for index in range(count):
        steps.append(
            {
                "record_family": records[index % len(records)],
                "checkpoint_class": checkpoints[index % len(checkpoints)],
                "s3_sector_class": sectors[index % len(sectors)],
                "repair_load_bucket": repairs[index % len(repairs)],
            }
        )
    return {
        "view_type": "patch_observer",
        "observer_id": observer_id,
        "axis": axis,
        "support_nodes": support_nodes or [],
        "h3_point": h3_point,
        "cap_axis": cap_axis,
        "radial_depth": radial_depth,
        "modular_depth": modular_depth,
        "transition_history_descriptor": {"steps": steps},
        "transition_history_histograms": {
            "local_transition_token": {str(records[0] * 101 + repairs[0]): 1.0},
            "local_transition_token_persistent": {str(records[0] * 101 + repairs[0]): 1.0},
        },
        "record_signature_histogram": {str(record): 1.0 / len(records) for record in records},
        "object_packet_histogram": {str((record * 7) % 83): 1.0 / len(records) for record in records},
        "repair_response_spectrum": [float((repair % 3) - 1) for repair in repairs],
        "prime_geometric_modular_spectrum": [float(((observer_id + j) % 5) - 2) for j in range(64)],
        "prime_geometric_control_quotient_spectrum": [
            float(((observer_id * 3 + j) % 7) - 3) for j in range(64)
        ],
        "support_visible_modular_spectrum": [float(((observer_id * 2 + j) % 7) - 3) for j in range(64)],
        "repair_modular_spectrum": [float(((repairs[0] + j) % 3) - 1) for j in range(32)],
        "modular_response_histograms": {
            "modular_response_cluster": {str(records[0] * 17): 1.0},
            "modular_response_component_0": {str(sectors[0] % 4): 1.0},
        },
        "transition_affinity_histograms": {
            "record_family": {str(record % 16): 1.0 / len(records) for record in records},
            "s3_sector_class": {str(sector % 3): 1.0 / len(sectors) for sector in sectors},
            "repair_load_bucket": {str(repair % 8): 1.0 / len(repairs) for repair in repairs},
        },
        "counterfactual_continuation_hist": [1.0, 2.0, 3.0],
        "record_persistence": 0.75,
        "sector_persistence": 0.5,
        "stable_fraction": 0.8,
        "committed_fraction": 1.0,
        "record_stability_mean": 3.0,
        "repair_load_mean": 1.0,
        "mismatch_density_mean": 0.0,
        "visible_signature_entropy": 1.0,
        "counterfactual_stability": 0.75,
    }
