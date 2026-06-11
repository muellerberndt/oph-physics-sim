from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oph_fpe.cosmology.comparable_data import comparable_data_report
from oph_fpe.bulk.neutral_bulk import (
    _prime_geometric_selected_rank_controls,
    build_neutral_observer_views,
    neutral_distance,
    neutral_distance_matrix,
    neutral_leakage_audit,
    neutral_model_selection,
    neutral_profile_audit_report,
    planted_neutral_control_report,
    prime_geometric_rank_refinement_report,
    prime_geometric_rank_sweep_report,
    shuffled_neutral_control_report,
    strict_neutral_bulk_receipt,
    strict_neutral_bulk_report,
    write_prime_geometric_rank_sweep_report,
    write_neutral_profile_audit_report,
)


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
    assert view.checkpoint_transition_hist.shape == (32,)
    assert view.sector_transition_hist.shape == (6,)
    assert view.repair_response_hist.shape == (16,)
    assert view.repair_response_spectrum.shape == (32,)
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
    assert np.allclose(view.modular_response_hist, changed_views[0].modular_response_hist)
    assert np.allclose(view.prime_geometric_modular_spectrum, changed_views[0].prime_geometric_modular_spectrum)
    assert np.allclose(
        view.prime_geometric_control_quotient_spectrum,
        changed_views[0].prime_geometric_control_quotient_spectrum,
    )
    assert np.allclose(view.support_visible_modular_spectrum, changed_views[0].support_visible_modular_spectrum)
    assert np.allclose(view.repair_modular_spectrum, changed_views[0].repair_modular_spectrum)
    assert np.allclose(view.transition_affinity_hist, changed_views[0].transition_affinity_hist)


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
    assert neutral_distance(a, c) > 0.05


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
        refinement={"stable_across_64k_256k_1m": False},
    )

    assert receipt["receipt"] == "STRICT_NEUTRAL_BULK_RECEIPT"
    assert receipt["strict_neutral_bulk"] is False
    assert receipt["physical_claim"] is False


def test_strict_neutral_bulk_receipt_can_pass_all_gates():
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
        refinement={"stable_across_64k_256k_1m": True},
    )

    assert receipt["strict_neutral_bulk"] is True
    assert receipt["physical_claim"] is True


def test_dimension_gate_uses_finite_regulator_median_for_planted_3d():
    report = planted_neutral_control_report(point_count=160, seed=372, max_points=128)

    dimension = report["rows"]["planted_3d"]["dimension"]

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
        refinement={"stable_across_64k_256k_1m": False},
    )

    assert report["strict_neutral_bulk"] is False
    assert report["receipt"]["strict_neutral_bulk"] is False
    assert "H3 fitted points" in report["forbidden_primary_features"]


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


def test_neutral_model_selection_reports_required_metric_families():
    rng = np.random.default_rng(12)
    coords = rng.normal(size=(48, 3))
    distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)

    report = neutral_model_selection(distance, seed=2, max_points=48)

    assert set(report["models"]) == {"S2", "E2", "E3", "H2", "H3", "H4"}
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

    assert report["control_quotient_rank3_refinement_candidate_receipt"] is True
    assert report["strict_neutral_bulk_refinement_receipt"] is False
    assert report["independent_rank3_selector_all"] is False
    assert report["candidate_dimension_stable"] is True
    assert report["candidate_dimension_drift"] < 0.1
    assert "independent_svd_rank3_selector_not_stable_or_false" in report["proof_blockers"]


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
    assert lane["mean_profile_count"] == 21
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
