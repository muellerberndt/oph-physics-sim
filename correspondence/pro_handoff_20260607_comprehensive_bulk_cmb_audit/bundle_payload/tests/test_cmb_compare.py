import json
from pathlib import Path

from oph_fpe.cosmology.cmb_compare import (
    cmb_lite_comparison_report,
    load_planck_tt_binned,
    write_cmb_lite_comparison,
)


def test_cmb_lite_compare_writes_shape_only_report(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cl_report = {
        "estimator": "spherical_harmonic",
        "ell_max": 4,
        "point_count": 128,
        "gate_report": {"allowed": True},
        "fields": {
            "record_signature": {
                "spectrum": [
                    {"ell": 0, "D_ell": 0.0},
                    {"ell": 1, "D_ell": 0.0},
                    {"ell": 2, "D_ell": 1.0},
                    {"ell": 3, "D_ell": 3.0},
                    {"ell": 4, "D_ell": 2.0},
                ]
            }
        },
    }
    (run_dir / "cl_comparison_report.json").write_text(json.dumps(cl_report), encoding="utf-8")
    benchmark = tmp_path / "planck_tt.txt"
    benchmark.write_text(
        "# l Dl -dDl +dDl BestFit\n"
        "50 1000 10 10 990\n"
        "100 3000 10 10 3010\n"
        "150 2000 10 10 2010\n",
        encoding="utf-8",
    )

    report = write_cmb_lite_comparison(run_dir, benchmark)

    assert (run_dir / "cmb_lite_comparison_report.json").exists()
    assert report["physical_cmb_prediction"] is False
    assert report["benchmark"]["row_count"] == 3
    assert report["best_shape_field"] == "record_signature"
    assert report["field_comparisons"]["record_signature"]["usable"] is True
    assert report["field_comparisons"]["record_signature"]["real_ell_physical_comparison"]["usable"] is False
    assert (
        report["field_comparisons"]["record_signature"]["real_ell_physical_comparison"]["reason"]
        == "sim_ell_range_does_not_cover_benchmark_ell_range"
    )
    assert report["field_comparisons"]["record_signature"]["overlap_ell_physical_comparison"]["usable"] is False


def test_cmb_lite_compare_reports_real_ell_overlap_without_full_range_claim():
    benchmark_rows = [
        {"ell": 50.0, "D_ell": 1.0},
        {"ell": 60.0, "D_ell": 1.5},
        {"ell": 70.0, "D_ell": 2.0},
        {"ell": 80.0, "D_ell": 2.5},
        {"ell": 90.0, "D_ell": 3.0},
        {"ell": 100.0, "D_ell": 3.5},
        {"ell": 110.0, "D_ell": 4.0},
        {"ell": 125.0, "D_ell": 3.0},
        {"ell": 150.0, "D_ell": 2.0},
        {"ell": 200.0, "D_ell": 1.0},
    ]
    cl_report = {
        "ell_max": 128,
        "fields": {
            "record_signature": {
                "spectrum": [
                    {"ell": 2, "D_ell": 0.0},
                    {"ell": 50, "D_ell": 1.0},
                    {"ell": 60, "D_ell": 1.5},
                    {"ell": 70, "D_ell": 2.0},
                    {"ell": 80, "D_ell": 2.5},
                    {"ell": 90, "D_ell": 3.0},
                    {"ell": 100, "D_ell": 3.5},
                    {"ell": 110, "D_ell": 4.0},
                    {"ell": 125, "D_ell": 3.0},
                    {"ell": 128, "D_ell": 2.5},
                ]
            }
        },
    }

    report = cmb_lite_comparison_report(cl_report, benchmark_rows)
    comparison = report["field_comparisons"]["record_signature"]

    assert comparison["real_ell_physical_comparison"]["usable"] is False
    assert comparison["overlap_ell_physical_comparison"]["usable"] is True
    assert comparison["overlap_ell_physical_comparison"]["full_benchmark_covered"] is False
    assert comparison["overlap_ell_physical_comparison"]["overlap_benchmark_count"] == 8
    assert comparison["overlap_ell_physical_comparison"]["shape_correlation"] > 0.9
    assert report["physical_cmb_prediction"] is False


def test_cmb_lite_compare_does_not_let_anticorrelated_negative_amplitude_field_win():
    benchmark_rows = [
        {"ell": 50.0, "D_ell": 1.0},
        {"ell": 100.0, "D_ell": 2.0},
        {"ell": 150.0, "D_ell": 4.0},
        {"ell": 200.0, "D_ell": 3.0},
    ]
    cl_report = {
        "ell_max": 5,
        "fields": {
            "anticorrelated": {
                "spectrum": [
                    {"ell": 2, "D_ell": 4.0},
                    {"ell": 3, "D_ell": 3.0},
                    {"ell": 4, "D_ell": 2.0},
                    {"ell": 5, "D_ell": 1.0},
                ]
            },
            "positive": {
                "spectrum": [
                    {"ell": 2, "D_ell": 1.0},
                    {"ell": 3, "D_ell": 2.0},
                    {"ell": 4, "D_ell": 4.0},
                    {"ell": 5, "D_ell": 3.0},
                ]
            },
        },
    }

    report = cmb_lite_comparison_report(cl_report, benchmark_rows)

    assert report["best_shape_field"] == "positive"
    assert report["best_positive_shape_field"] == "positive"
    assert report["best_normalized_axis_diagnostic_field"] is not None
    assert report["field_comparisons"]["anticorrelated"]["shape_correlation"] < 0.0
    assert report["field_comparisons"]["anticorrelated"]["usable_positive_shape"] is False
    assert report["field_comparisons"]["positive"]["usable_positive_shape"] is True


def test_planck_loader_skips_comments(tmp_path: Path):
    benchmark = tmp_path / "planck_tt.txt"
    benchmark.write_text("# header\n 4.7 1479.3 50 51 1461\n", encoding="utf-8")

    rows = load_planck_tt_binned(benchmark)

    assert rows == [
        {
            "ell": 4.7,
            "D_ell": 1479.3,
            "minus_dD_ell": 50.0,
            "plus_dD_ell": 51.0,
            "best_fit_D_ell": 1461.0,
        }
    ]
