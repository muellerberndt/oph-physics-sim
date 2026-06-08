from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oph_fpe.cosmology.camb_adapter import compare_camb_tt_to_benchmark, write_camb_lcdm_baseline_report


def test_compare_camb_tt_to_benchmark_reports_real_multipole_metrics():
    ell = np.arange(2, 8, dtype=float)
    model = np.asarray([10.0, 20.0, 40.0, 30.0, 15.0, 8.0])
    rows = [
        {"ell": 2.0, "D_ell": 11.0, "minus_dD_ell": 1.0, "plus_dD_ell": 1.0, "best_fit_D_ell": 10.0},
        {"ell": 3.0, "D_ell": 19.0, "minus_dD_ell": 1.0, "plus_dD_ell": 1.0, "best_fit_D_ell": 20.0},
        {"ell": 4.0, "D_ell": 39.0, "minus_dD_ell": 1.0, "plus_dD_ell": 1.0, "best_fit_D_ell": 40.0},
        {"ell": 5.0, "D_ell": 31.0, "minus_dD_ell": 1.0, "plus_dD_ell": 1.0, "best_fit_D_ell": 30.0},
        {"ell": 6.0, "D_ell": 16.0, "minus_dD_ell": 1.0, "plus_dD_ell": 1.0, "best_fit_D_ell": 15.0},
        {"ell": 7.0, "D_ell": 9.0, "minus_dD_ell": 1.0, "plus_dD_ell": 1.0, "best_fit_D_ell": 8.0},
    ]

    report = compare_camb_tt_to_benchmark(ell, model, rows)

    assert report["usable"] is True
    assert report["bin_count"] == 6
    assert report["shape_correlation"] > 0.99
    assert report["amplitude_fit_chi2_per_bin"] < report["raw_chi2_per_bin"] + 1.0e-9
    assert len(report["binned_tt_comparison"]) == 6


def test_write_camb_lcdm_baseline_report_smoke(tmp_path: Path):
    pytest.importorskip("camb")
    benchmark = tmp_path / "planck_tt.txt"
    benchmark.write_text(
        "# l Dl -dDl +dDl BestFit\n"
        "50 1479.0 50 50 1461.0\n"
        "100 2955.0 65 65 2904.0\n"
        "200 5464.0 90 90 5535.0\n"
        "500 2460.0 35 35 2465.0\n"
        "1000 1050.0 20 20 1055.0\n",
        encoding="utf-8",
    )

    report = write_camb_lcdm_baseline_report(benchmark, tmp_path / "out", lmax=1200)

    assert (tmp_path / "out" / "camb_lcdm_baseline_report.json").exists()
    assert (tmp_path / "out" / "camb_lcdm_baseline_report.md").exists()
    assert (tmp_path / "out" / "camb_lcdm_tt_bins.csv").exists()
    assert report["physical_cmb_prediction"] is False
    assert report["comparison"]["usable"] is True
    assert report["receipt_thresholds"]["shape_correlation_min"] == 0.995
    assert report["software"]["camb_version"] != "not_installed"
    assert report["input_hashes"]["benchmark_sha256"]
    assert report["input_hashes"]["params_sha256"]
