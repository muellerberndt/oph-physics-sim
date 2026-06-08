from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oph_fpe.cosmology.camb_adapter import (
    compare_camb_tt_to_benchmark,
    write_camb_lcdm_baseline_report,
    write_oph_inflation_cmb_camb_report,
    write_oph_screen_camb_report,
)


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


def test_write_oph_screen_camb_report_smoke(tmp_path: Path):
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
    screen_report = tmp_path / "oph_screen_power_report.json"
    screen_report.write_text(
        """
        {
          "simulator_primordial_reference_ready": false,
          "primordial_reference_source": "phenomenological_planck_eta_target_due_to_invalid_simulator_tilt",
          "reference_screen_parameters": {
            "eta_R": 0.035,
            "n_s_proxy": 0.965,
            "N_cap_eff": 4096
          },
          "primordial_bridge": {
            "A_s": 2.1e-9,
            "excludes": ["parity_envelope", "BipoSH_off_diagonal_covariance"]
          }
        }
        """,
        encoding="utf-8",
    )

    report = write_oph_screen_camb_report(screen_report, benchmark, tmp_path / "out", lmax=1200)

    assert (tmp_path / "out" / "oph_screen_camb_report.json").exists()
    assert (tmp_path / "out" / "oph_screen_camb_report.md").exists()
    assert (tmp_path / "out" / "oph_screen_camb_tt_bins.csv").exists()
    assert report["physical_cmb_prediction"] is False
    assert report["screen_input"]["simulator_eta_R_ready"] is False
    assert report["camb"]["lambda_cdm_parameters_with_screen_ns"]["ns"] == 0.965
    assert report["comparison"]["usable"] is True


def test_write_oph_inflation_cmb_camb_report_smoke(tmp_path: Path):
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
    bridge_report = tmp_path / "oph_inflation_cmb_bridge_report.json"
    bridge_report.write_text(
        """
        {
          "mode": "oph_inflation_cmb_bridge_v0",
          "screen_spectrum_prediction": {
            "P": 1.6309682094039593,
            "theta_OPH": 0.033978504362582485,
            "n_s": 0.9660214956374176,
            "A_zeta": 2.502261321821402e-09
          },
          "cmb_success_ladder": {
            "core_numbers": {
              "v0_2_IR_bestfit_q_IR": 0.2445545067865991,
              "v0_2_IR_bestfit_ell_IR": 33.614958176528,
              "v0_3_camb_lowell_LCDM_chi2_ell2_29": 27.40723660095901,
              "v0_3_camb_lowell_IR_bestfit_chi2_ell2_29": 16.65364090607876,
              "v0_4_LCDM_PTE_R_OE_upper": 0.0107,
              "v0_4_parity_PTE_R_OE_upper": 0.4069
            }
          }
        }
        """,
        encoding="utf-8",
    )

    report = write_oph_inflation_cmb_camb_report(bridge_report, benchmark, tmp_path / "out", lmax=1200)

    assert (tmp_path / "out" / "oph_inflation_cmb_camb_report.json").exists()
    assert (tmp_path / "out" / "oph_inflation_cmb_camb_report.md").exists()
    assert (tmp_path / "out" / "oph_inflation_cmb_tt_bins.csv").exists()
    assert (tmp_path / "out" / "oph_inflation_cmb_tt_curves.csv").exists()
    assert report["measurement_comparable_cmb_curve"] is True
    assert report["physical_cmb_prediction"] is False
    assert report["oph_input"]["n_s"] == 0.9660214956374176
    assert report["oph_input"]["q_IR"] == 0.2445545067865991
    assert report["comparison"]["oph_p48_ir_v04"]["usable"] is True
    assert report["low_ell_v04_diagnostic"]["CAMB_OPH_IR_chi2_ell2_29"] == 16.65364090607876
