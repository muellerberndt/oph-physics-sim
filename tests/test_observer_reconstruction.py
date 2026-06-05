from __future__ import annotations

import numpy as np

from oph_fpe.bulk.observer_reconstruction import (
    bulk_reconstruction_report,
    neutral_reconstruction_controls,
    observer_distance_matrix,
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
    assert report["observer_similarity_debug_report"]["mode"] == "boundary_similarity_dimension_diagnostic"
    assert "component_dimension_debug_reports" in report
    assert report["controls"]["radial_depth_used"] is False
    assert "planted_3d" in report["controls"]["planted_dimensions"]
    assert "planted_s2_boundary" in report["controls"]["planted_dimensions"]
    assert "planted_h3_bulk" in report["controls"]["planted_dimensions"]
    assert report["candidate_3d_dimension_window"] is False


def test_observer_distance_matrix_is_zero_diagonal():
    views = [
        {"view_type": "patch_observer", "observer_id": 1, "visible_readout_hash": "a"},
        {"view_type": "patch_observer", "observer_id": 2, "visible_readout_hash": "b"},
    ]
    similarity, ids = observer_similarity_matrix(views)
    distance = observer_distance_matrix(similarity)
    assert ids == [1, 2]
    assert np.allclose(np.diag(distance), 0.0)


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
        }
        for i in range(count)
    ]
