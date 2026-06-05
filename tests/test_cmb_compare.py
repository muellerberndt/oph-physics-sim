import json
from pathlib import Path

from oph_fpe.cosmology.cmb_compare import load_planck_tt_binned, write_cmb_lite_comparison


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
