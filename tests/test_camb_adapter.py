from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import oph_fpe.cosmology.camb_adapter as camb_adapter
from oph_fpe.cosmology.camb_adapter import (
    compare_camb_tt_to_benchmark,
    finite_repair_clock_cmb_camb_report,
    official_planck_readiness_report,
    write_camb_lcdm_baseline_report,
    write_finite_repair_clock_cmb_camb_report,
    write_official_planck_readiness_report,
    write_oph_exact_cmb_camb_report,
    write_oph_inflation_cmb_camb_report,
    write_oph_screen_camb_report,
    write_scale_compressed_cmb_camb_report,
)
from oph_fpe.cosmology.selector_elimination import ir_kernel


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


def test_finite_clock_camb_rejects_stale_true_transition_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ell = np.arange(2, 8, dtype=float)
    model = np.asarray([10.0, 20.0, 40.0, 30.0, 15.0, 8.0])
    benchmark = [
        {
            "ell": float(value),
            "D_ell": float(amplitude),
            "minus_dD_ell": 1.0,
            "plus_dD_ell": 1.0,
            "best_fit_D_ell": float(amplitude),
        }
        for value, amplitude in zip(ell, model, strict=True)
    ]
    monkeypatch.setattr(
        camb_adapter,
        "selector_elimination_report",
        lambda **_kwargs: {"cmb_ir_kernel": {"q_IR": 0.25, "ell_IR": 32.0}},
    )
    monkeypatch.setattr(
        camb_adapter,
        "_run_camb_tt",
        lambda *_args, **_kwargs: (ell, model),
    )
    monkeypatch.setattr(
        camb_adapter,
        "_run_camb_tt_custom_power",
        lambda *_args, **_kwargs: (ell, model),
    )
    stale_report = {
        "finite_transition_matrix_ready": True,
        "finite_lattice_derived": True,
        "repair_clock_certificate": True,
        "clock_normalization_certified": True,
        "clock_normalization_numeric_match": True,
        "repair_scale_hypothesis_clock_match": True,
        "state_count": 2,
        "transition_count": 48,
        "primary": {
            "finite": True,
            "irreducible": False,
            "aperiodic": False,
            "lambda_2": 1.0,
            "detailed_balance_max_abs_error": 0.0,
            "n_s_estimate": 1.0,
            "eta_R_estimate": 0.0,
            "kappa_rep_estimate": 0.0,
        },
    }

    report = finite_repair_clock_cmb_camb_report(stale_report, benchmark)

    finite_input = report["finite_repair_clock_input"]
    assert finite_input["transition_clock_eligibility"]["eligible"] is False
    assert finite_input["matrix_ready"] is False
    assert finite_input["finite_lattice_derived"] is False
    assert finite_input["repair_clock_certificate"] is False
    assert finite_input["clock_normalization_certified"] is False
    assert finite_input["clock_normalization_numeric_match"] is False
    assert finite_input["repair_scale_hypothesis_clock_match"] is False
    assert report["finite_lattice_clock_derived"] is False
    assert report["repair_clock_certificate"] is False


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


def test_write_scale_compressed_cmb_camb_report_smoke(tmp_path: Path):
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
    scale_report = tmp_path / "scale_compressed_repair_report.json"
    scale_report.write_text(
        """
        {
          "mode": "oph_scale_compressed_repair_round_branch_v0",
          "logical_repair_rounds": 24,
          "scale_compressed_operator_receipt": true,
          "repair_round_trace_receipt": true,
          "h3_preview": {
            "populated_h3_preview_receipt": true
          },
          "cmb_parameter_readouts": {
            "eta_R": 0.033978504362582485,
            "n_s": 0.9660214956374176,
            "q_IR": 0.25,
            "ell_IR": 32.0,
            "N_CRC_predicted_from_P": 4.274424586583862e122,
            "N_CRC_declared": 3.3149984974788145e122,
            "relative_error_gprime_vs_N_CRC": 0.005281676043309455
          }
        }
        """,
        encoding="utf-8",
    )

    report = write_scale_compressed_cmb_camb_report(scale_report, benchmark, tmp_path / "out", lmax=1200)

    assert (tmp_path / "out" / "scale_compressed_cmb_camb_report.json").exists()
    assert (tmp_path / "out" / "scale_compressed_cmb_camb_report.md").exists()
    assert (tmp_path / "out" / "scale_compressed_cmb_tt_bins.csv").exists()
    assert (tmp_path / "out" / "scale_compressed_cmb_tt_curves.csv").exists()
    assert report["mode"] == "oph_scale_compressed_cmb_camb_transfer_v0"
    assert report["measurement_comparable_cmb_curve"] is True
    assert report["physical_cmb_prediction"] is False
    assert report["scale_compressed_input"]["logical_repair_rounds"] == 24
    assert report["scale_compressed_input"]["scale_compressed_operator_receipt"] is True
    assert report["comparison"]["scale_compressed_ir_kernel"]["usable"] is True


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


def test_write_oph_exact_cmb_camb_report_smoke(tmp_path: Path):
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
    source = tmp_path / "cmb5"
    source.mkdir()
    (source / "OPH-CMB-Official-Likelihood-and-Finite-Patch-v1.0.md").write_text("v1", encoding="utf-8")
    (source / "finite_patch_cmb_derivations_v1_0.md").write_text("math", encoding="utf-8")
    (source / "OPH-Unique-Prediction-Gate-v0.9.md").write_text("v09", encoding="utf-8")
    (source / "OPH-CMB-Selector-Elimination-v1.5.md").write_text("v15", encoding="utf-8")
    _write_selector_status_csv(source / "selector_elimination_status_v1_5.csv")
    _write_exact_ir_csv(source / "exact_ir_kernel_values_v1_5.csv")

    report = write_oph_exact_cmb_camb_report(
        benchmark,
        tmp_path / "out",
        source_dir=source,
        lmax=1200,
    )

    assert (tmp_path / "out" / "oph_exact_cmb_camb_report.json").exists()
    assert (tmp_path / "out" / "oph_exact_cmb_camb_report.md").exists()
    assert (tmp_path / "out" / "oph_exact_cmb_tt_bins.csv").exists()
    assert (tmp_path / "out" / "oph_exact_cmb_tt_curves.csv").exists()
    assert report["mode"] == "oph_exact_cmb_camb_transfer_v1"
    assert report["measurement_comparable_cmb_curve"] is True
    assert report["physical_cmb_prediction"] is False
    assert abs(report["oph_exact_input"]["n_s"] - 0.964841143031) < 2.0e-12
    assert report["oph_exact_input"]["q_IR"] == 0.25
    assert report["oph_exact_input"]["ell_IR"] == 32.0
    assert report["oph_exact_input"]["selector_elimination_theorem_receipt"] is False
    assert report["oph_exact_input"]["selector_elimination_source_audit_receipt"] is True
    assert report["selector_elimination_v1_5"]["SOURCE_PACKET_AUDIT_RECEIPT"] is True
    assert report["comparison"]["oph_exact_ir_v10"]["usable"] is True
    assert report["source_files"]["all_core_files_present"] is True
    assert report["source_files"]["selector_v1_5_core_files_present"] is True
    assert report["official_planck_likelihood_readiness"]["official_likelihood_execution_ready"] is False


def test_write_finite_repair_clock_cmb_camb_report_smoke(tmp_path: Path):
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
    finite_clock = tmp_path / "finite_repair_transition_matrix_report.json"
    finite_clock.write_text(
        """
        {
          "mode": "oph_finite_repair_transition_clock_v0",
          "finite_transition_matrix_ready": true,
          "finite_lattice_derived": true,
          "clock_normalization_certified": false,
          "repair_clock_certificate": false,
          "primary_matrix": "reversible_empirical",
          "repair_step_time": 50.26548245743669,
          "state_count": 2,
          "transition_count": 3072,
          "primary": {
            "kappa_rep_estimate": 2.4755067024747386,
            "eta_R_estimate": 0.03201874992042351,
            "n_s_estimate": 0.9679812500795765,
            "finite": true,
            "irreducible": true,
            "aperiodic": true,
            "lambda_2": 0.2,
            "detailed_balance_max_abs_error": 0.0
          },
          "blockers": [
            "finite transition matrix does not yield kappa_rep=e under the declared repair-step time"
          ]
        }
        """,
        encoding="utf-8",
    )
    source = tmp_path / "cmb7"
    source.mkdir()
    (source / "OPH-CMB-Selector-Elimination-v1.5.md").write_text("v15", encoding="utf-8")
    _write_selector_status_csv(source / "selector_elimination_status_v1_5.csv")
    _write_exact_ir_csv(source / "exact_ir_kernel_values_v1_5.csv")

    report = write_finite_repair_clock_cmb_camb_report(
        finite_clock,
        benchmark,
        tmp_path / "out",
        source_dir=source,
        lmax=1200,
    )

    assert (tmp_path / "out" / "finite_repair_clock_cmb_camb_report.json").exists()
    assert (tmp_path / "out" / "finite_repair_clock_cmb_camb_report.md").exists()
    assert (tmp_path / "out" / "finite_repair_clock_cmb_tt_bins.csv").exists()
    assert (tmp_path / "out" / "finite_repair_clock_cmb_tt_curves.csv").exists()
    assert report["mode"] == "finite_repair_clock_cmb_camb_transfer_v0"
    assert report["measurement_comparable_cmb_curve"] is True
    assert report["finite_lattice_clock_derived"] is True
    assert report["repair_clock_certificate"] is False
    assert report["physical_cmb_prediction"] is False
    assert report["finite_repair_clock_input"]["n_s"] == 0.9679812500795765
    assert report["selector_ir_input"]["q_IR"] == 0.25
    assert report["comparison"]["finite_repair_clock_scalar_tilt"]["usable"] is True
    assert report["comparison"]["finite_repair_clock_plus_selector_ir"]["usable"] is True


def test_official_planck_readiness_report_is_gated(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "OPH_PLANCK_LIKELIHOOD_DIR",
        "PLANCK_PR3_LIKELIHOOD_DIR",
        "PLANCK_LIKELIHOOD_DIR",
        "CLIK_DATA",
    ):
        monkeypatch.delenv(name, raising=False)

    report = official_planck_readiness_report()

    assert "modules" in report
    assert report["official_planck_likelihood_data_paths_configured"] is False
    assert report["official_likelihood_execution_ready"] is False
    assert "official_planck_likelihood_data_path_not_configured" in report["blockers"]


def test_write_official_planck_readiness_report_records_configured_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    for name in (
        "PLANCK_PR3_LIKELIHOOD_DIR",
        "PLANCK_LIKELIHOOD_DIR",
        "CLIK_DATA",
    ):
        monkeypatch.delenv(name, raising=False)
    data_dir = tmp_path / "planck_pr3"
    data_dir.mkdir()
    monkeypatch.setenv("OPH_PLANCK_LIKELIHOOD_DIR", str(data_dir))

    report = write_official_planck_readiness_report(tmp_path / "out")

    assert report["official_planck_likelihood_data_paths_configured"] is True
    assert report["official_likelihood_execution_ready"] is report["official_clik_api_available"]
    assert (tmp_path / "out" / "official_planck_likelihood_readiness_report.json").exists()
    assert (tmp_path / "out" / "official_planck_likelihood_readiness_report.md").exists()


def _write_selector_status_csv(path: Path) -> None:
    rows = [
        {
            "old_selector": "S1: eta_R = e alpha sqrt(pi)",
            "v1_5_status": "not free, but still requires the repair-clock normalization theorem/certificate",
            "replacement": "repair clock",
            "what_is_closed": "alpha dependence",
            "what_remains": "kappa_rep=e",
        },
        {
            "old_selector": "S2: q_IR = 1/4 from four equipotent sectors",
            "v1_5_status": "removed as selector",
            "replacement": "affine zero-mode",
            "what_is_closed": "quarter reserve",
            "what_remains": "branch validation",
        },
        {
            "old_selector": "S3: ell_IR = 32",
            "v1_5_status": "removed as selector",
            "replacement": "visible covariance rank",
            "what_is_closed": "F+V+1",
            "what_remains": "noncollapse",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_exact_ir_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ell", "F_IR_exact_q1over4_L32"])
        writer.writeheader()
        for ell in (2, 3, 32, 220):
            writer.writerow({"ell": ell, "F_IR_exact_q1over4_L32": float(ir_kernel(float(ell)))})
