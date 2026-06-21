from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.cosmology.physical_cmb_output import (
    physical_cmb_output_comparison_report,
    write_physical_cmb_output_comparison_report,
)


def test_physical_cmb_output_comparison_keeps_prediction_gate_hard(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "physical_cmb_input_report.json",
        {
            "PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT": False,
            "blockers": ["A_zeta_not_finite_derived"],
        },
    )
    _write_json(
        run / "physical_cmb_promotion_audit_report.json",
        {
            "physical_cmb_promotion_ready": False,
            "official_likelihood_ready": False,
            "promotion_blockers": ["official_likelihood_not_ready"],
        },
    )
    _write_json(
        run / "camb_lcdm_baseline_report.json",
        {
            "comparison": {
                "usable": True,
                "bin_count": 83,
                "shape_correlation": 0.9998,
                "normalized_rmse": 0.02,
                "amplitude_fit_chi2_per_bin": 1.05,
                "first_peak_ell": 221.0,
                "benchmark_first_peak_ell": 220.0,
            }
        },
    )
    _write_json(
        run / "scale_compressed_cmb_camb_report.json",
        {
            "measurement_comparable_cmb_curve": True,
            "physical_cmb_prediction": False,
            "comparison": {
                "scale_compressed_ir_kernel": {
                    "usable": True,
                    "bin_count": 83,
                    "shape_correlation": 0.9999,
                    "normalized_rmse": 0.018,
                    "amplitude_fit_chi2_per_bin": 0.94,
                    "first_peak_ell": 221.0,
                    "benchmark_first_peak_ell": 220.0,
                },
                "lcdm_reference": {
                    "usable": True,
                    "bin_count": 83,
                    "shape_correlation": 0.9998,
                    "normalized_rmse": 0.021,
                    "amplitude_fit_chi2_per_bin": 1.02,
                },
            },
        },
    )
    _write_json(
        run / "finite_repair_clock_cmb_camb_report.json",
        {
            "measurement_comparable_cmb_curve": True,
            "physical_cmb_prediction": False,
            "comparison": {
                "finite_repair_clock_plus_selector_ir": {
                    "usable": True,
                    "bin_count": 83,
                    "shape_correlation": 0.99985,
                    "normalized_rmse": 0.019,
                    "amplitude_fit_chi2_per_bin": 0.98,
                }
            },
        },
    )
    (run / "scale_compressed_cmb_tt_bins.csv").write_text(
        "ell,observed_D_ell,minus_dD_ell,plus_dD_ell,scale_compressed_ir_kernel_D_ell\n"
        "50,1000,10,10,990\n"
        "80,1200,20,20,1240\n",
        encoding="utf-8",
    )

    report = physical_cmb_output_comparison_report([run])
    written = write_physical_cmb_output_comparison_report([run], tmp_path / "out")

    assert report["PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT"] is True
    assert report["USABLE_PHYSICAL_CMB_DATA_RECEIPT"] is True
    assert report["usable_physical_cmb_data_receipt"] is True
    assert "physical_cmb_best_oph_residuals.csv" in report["usable_physical_cmb_data_products"]
    assert report["PHYSICAL_CMB_PREDICTION_RECEIPT"] is False
    assert report["physical_cmb_prediction"] is False
    assert report["measurement_comparable_model_count"] == 4
    assert report["oph_diagnostic_model_count"] == 2
    assert report["best_oph_diagnostic_model"]["model_id"] == "scale_compressed_ir_kernel"
    assert report["best_oph_diagnostic_model"]["amplitude_fit_chi2_per_bin"] == 0.94
    assert report["best_oph_residual_summary"]["available"] is True
    assert report["best_oph_residual_summary"]["bin_count"] == 2
    assert report["best_oph_residual_summary"]["max_abs_sigma_residual"] == 2.0
    assert report["best_oph_residual_rows"][0]["residual_sigma"] == -1.0
    assert report["best_oph_peak_feature_summary"]["available"] is True
    assert report["best_oph_peak_feature_summary"]["peak_count"] == 1
    assert report["best_oph_peak_feature_summary"]["mean_abs_peak_ell_delta"] == 0.0
    assert report["peak_feature_rows"][0]["model_id"] == "scale_compressed_ir_kernel"
    assert report["peak_feature_rows"][0]["peak_index"] == 1
    assert report["peak_feature_rows"][0]["ell_delta"] == 0.0
    assert written["best_oph_diagnostic_model"]["model_id"] == "scale_compressed_ir_kernel"
    assert (tmp_path / "out" / "physical_cmb_output_comparison_report.json").exists()
    assert (tmp_path / "out" / "physical_cmb_output_comparison_report.md").exists()
    assert "scale_compressed_ir_kernel" in (
        tmp_path / "out" / "physical_cmb_output_comparison_rows.csv"
    ).read_text(encoding="utf-8")
    assert "residual_sigma" in (tmp_path / "out" / "physical_cmb_best_oph_residuals.csv").read_text(
        encoding="utf-8"
    )
    assert "peak_index" in (tmp_path / "out" / "physical_cmb_peak_features.csv").read_text(
        encoding="utf-8"
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
