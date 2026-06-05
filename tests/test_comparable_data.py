from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.cosmology.comparable_data import comparable_data_report, write_comparable_data_package


def test_comparable_data_report_collects_h3_cmb_and_holonomy(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "manifest.json",
        {
            "run_id": "r1",
            "name": "demo",
            "patch_count": 4096,
        },
    )
    _write_json(
        run / "modular_response_h3_report.json",
        {
            "h3_bulk_candidate_receipt": True,
            "h3_fit": {
                "heldout_normalized_rmse": 0.9,
                "heldout_explained_variance": 0.1,
            },
            "s2_boundary_control": {"heldout_normalized_rmse": 1.1},
            "control_fits": {
                "shuffled_response": {"heldout_normalized_rmse": 1.0},
                "shuffled_observer_labels": {"heldout_normalized_rmse": 1.2},
                "no_perturbation": {"heldout_normalized_rmse": 1.0},
            },
            "wrong_scale_control_fits": {
                "1x": {"heldout_normalized_rmse": 1.05},
                "pi": {"heldout_normalized_rmse": 1.2},
                "4pi": {"heldout_normalized_rmse": 1.01},
            },
        },
    )
    _write_json(
        run / "cmb_lite_comparison_report.json",
        {
            "benchmark": {"label": "PlanckLite"},
            "best_shape_field": "record_signature",
            "field_comparisons": {
                "record_signature": {
                    "shape_correlation": 0.3,
                    "normalized_rmse": 0.95,
                    "peak_fraction_delta": 0.1,
                }
            },
            "physical_cmb_prediction": False,
        },
    )
    _write_json(
        run / "cl_comparison_report.json",
        {
            "freezeout_cycle": 8,
            "committed_fraction": 0.99,
            "fields": {
                "record_signature": {
                    "peak_ell": 9,
                    "total_abs_D_ell_2_plus": 1.5,
                    "control_comparison": {"min_relative_l2_delta": 0.8},
                }
            },
        },
    )
    _write_json(
        run / "array_holonomy_report.json",
        {
            "triangle_count": 100,
            "defect_fraction": 0.7,
            "cluster_count": 4,
        },
    )
    _write_json(
        run / "observer_chart_object_h3_report.json",
        {
            "object_count": 11,
            "median_h3_compactness_normalized": 0.2,
            "median_s2_boundary_compactness_normalized": 0.1,
            "median_shuffled_h3_compactness_normalized": 0.5,
            "h3_beats_shuffled_incidence": True,
            "h3_not_boundary_dominated": False,
            "observer_chart_object_h3_receipt": True,
            "observer_chart_bulk_population_receipt": False,
        },
    )

    report = comparable_data_report([tmp_path])

    assert report["run_count"] == 1
    assert report["physical_cmb_prediction"] is False
    assert report["bulk_3d_established"] is False
    assert report["measurement_lanes"]["h3_modular_response_controls"]["receipt_count"] == 1
    assert report["measurement_lanes"]["observer_chart_object_population"]["object_chart_receipt_count"] == 1
    assert report["measurement_lanes"]["observer_chart_object_population"]["bulk_population_receipt_count"] == 0
    assert report["measurement_lanes"]["planck_tt_shape_lite"]["mean_record_signature_shape_correlation"] == 0.3
    assert report["measurement_lanes"]["screen_holonomy_defect_proxy"]["mean_defect_fraction"] == 0.7


def test_write_comparable_data_package_writes_json_csv_and_markdown(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(run / "array_holonomy_report.json", {"triangle_count": 10, "defect_fraction": 0.5})

    out = tmp_path / "out"
    report = write_comparable_data_package([tmp_path], out)

    assert report["run_count"] == 1
    assert (out / "comparable_data_snapshot.json").exists()
    assert (out / "comparable_data_rows.csv").exists()
    assert (out / "comparable_data_snapshot.md").exists()
    assert "not a physical CMB prediction" in (out / "comparable_data_snapshot.md").read_text(encoding="utf-8")


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")
