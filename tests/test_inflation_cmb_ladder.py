import csv
import json
import math
from pathlib import Path

from oph_fpe.constants.oph_pixel import P_STAR
from oph_fpe.cosmology.edge_center_clock import (
    EDGE_CENTER_CLOCK_RECEIPT,
    EDGE_CENTER_EVIDENCE_RECEIPTS,
    edge_center_clock_target,
)
from oph_fpe.cosmology.inflation_cmb_ladder import (
    cmb_success_ladder_report,
    flat_sector_selection_report,
    screen_spectrum_prediction,
    write_inflation_cmb_bridge_report,
)


def test_screen_spectrum_prediction_keeps_amplitude_and_lift_gated():
    report = screen_spectrum_prediction()
    target = edge_center_clock_target()

    assert report["selected_clock_branch"] == "edge_center_orientation_half"
    assert math.isclose(report["full_collar_derivative_target"], P_STAR / 24.0)
    assert report["orientation_halves"] == 2
    assert math.isclose(report["theta_OPH"], P_STAR / 48.0)
    assert math.isclose(report["eta_R"], target.theta)
    assert math.isclose(report["n_s"], 1.0 - P_STAR / 48.0)
    assert math.isclose(report["kappa_rep"], target.kappa_rep)
    assert abs(report["n_s"] - 0.9660214956) < 1.0e-9
    assert report[EDGE_CENTER_CLOCK_RECEIPT] is False
    assert all(report[name] is False for name in EDGE_CENTER_EVIDENCE_RECEIPTS)
    assert report["e_diagnostic_control"]["promoting"] is False
    assert report["e_diagnostic_control"]["selected"] is False
    assert math.isclose(report["A_q_cmi_upper_bound"], 4.0 * math.log(2.0) * 3.61e-11)
    assert report["A_q_energy"] is None
    assert report["A_zeta"] is None
    assert report["Sachs_Wolfe_conversion_used"] is False
    assert report["screen_to_primordial_lift_receipt"] is False
    assert report["physical_claim"] is False if "physical_claim" in report else True


def test_flat_sector_selection_reports_residual_anomaly():
    report = flat_sector_selection_report(
        omega_lambda_oph=0.68,
        omega_b0=0.05,
        omega_nu0=0.002,
        omega_r0=0.0001,
    )

    assert report["status"] == "OPEN_THEOREM"
    assert report["geometry_branch"] == "UNRESOLVED"
    assert report["selected_Omega_K"] is None
    assert report["Omega_A0"] is None
    assert report["Omega_A0_residual"] is None
    assert math.isclose(report["Omega_A0_plus_Omega_K0"], 0.2679)
    assert report["anomaly_curvature_degeneracy"] is True


def test_cmb_success_ladder_imports_v04_tables(tmp_path: Path):
    (tmp_path / "oph_cmb_success_summary_v0_4.json").write_text(
        json.dumps(
            {
                "claim_boundary": "diagnostic only",
                "success_criteria_met": ["low ell"],
                "core_numbers": {
                    "v0_2_IR_bestfit_q_IR": 0.2445545068,
                    "v0_4_LCDM_PTE_R_OE_upper": 0.0107,
                },
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        tmp_path / "01_model_selection_lowell_and_high_ell_v0_4.csv",
        [
            {
                "ell_min": 2,
                "ell_max": 29,
                "model": "CAMB_LCDM_powerlaw",
                "chi2_diag": 27.4,
                "delta_AIC_vs_LCDM": 0.0,
            },
            {
                "ell_min": 2,
                "ell_max": 29,
                "model": "OPH_IR_bestfit_lowell_q0p2446_ell33p615",
                "chi2_diag": 16.6,
                "delta_AIC_vs_LCDM": -6.7,
            },
        ],
    )
    _write_csv(
        tmp_path / "oph_lowell_fullsky_mc_summary_v0_4.csv",
        [
            {
                "model": "LCDM_bestfit_theory",
                "PTE_R_OE_upper_tail": 0.0107,
                "PTE_S_1_2_lower_tail": 0.0775,
            },
            {
                "model": "OPH_IR_plus_parity_bestfit",
                "PTE_R_OE_upper_tail": 0.2778,
                "PTE_S_1_2_lower_tail": 0.2299,
            },
        ],
    )
    _write_csv(
        tmp_path / "01_TT_TE_EE_diagonal_proxy_chi2_v0_5.csv",
        [
            {
                "spectrum": "TT",
                "ell_min": 2,
                "ell_max": 29,
                "model": "OPH_IR_bestfit_lowell_q0p2446_ell33p615",
                "delta_chi2_improvement_vs_LCDM": 10.75,
            },
            {
                "spectrum": "TE",
                "ell_min": 2,
                "ell_max": 29,
                "model": "OPH_IR_bestfit_lowell_q0p2446_ell33p615",
                "delta_chi2_improvement_vs_LCDM": 0.43,
            },
            {
                "spectrum": "EE",
                "ell_min": 2,
                "ell_max": 29,
                "model": "OPH_IR_bestfit_lowell_q0p2446_ell33p615",
                "delta_chi2_improvement_vs_LCDM": -1.83,
            },
            {
                "spectrum": "TT",
                "ell_min": 30,
                "ell_max": 1200,
                "model": "OPH_IR_bestfit_lowell_q0p2446_ell33p615",
                "delta_chi2_improvement_vs_LCDM": -0.39,
            },
        ],
    )
    _write_csv(
        tmp_path / "02_combined_lowell_TT_TE_EE_proxy_v0_5.csv",
        [
            {
                "spectra": "TT+TE+EE",
                "ell_min": 2,
                "ell_max": 29,
                "model": "OPH_IR_bestfit_lowell_q0p2446_ell33p615",
                "delta_chi2_improvement_vs_LCDM": 9.36,
            }
        ],
    )
    _write_csv(
        tmp_path / "07_success_falsifier_status_ledger_v0_5.csv",
        [
            {
                "gate": "D",
                "quantity": "EE low-l direction",
                "v0_5_status": "pressure point",
            }
        ],
    )

    report = cmb_success_ladder_report(tmp_path)

    assert report["diagnostic_cmb_data_available"] is True
    assert report["physical_cmb_prediction"] is False
    assert report["core_numbers"]["v0_2_IR_bestfit_q_IR"] == 0.2445545068
    assert report["low_ell_model_selection"]["oph_ir_bestfit_chi2"] == 16.6
    assert report["fullsky_monte_carlo"]["OPH_joint_PTE_R_OE_upper"] == 0.2778
    assert report["hard_gates_v0_5"]["TT_lowell_delta_chi2"] == 10.75
    assert report["hard_gates_v0_5"]["TE_lowell_delta_chi2"] == 0.43
    assert report["hard_gates_v0_5"]["EE_lowell_delta_chi2"] == -1.83
    assert report["hard_gates_v0_5"]["combined_TT_TE_EE_lowell_delta_chi2"] == 9.36
    assert report["hard_gates_v0_5"]["pressure_points"][0]["gate"] == "D"


def test_write_inflation_cmb_bridge_report(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "oph_cmb_success_summary_v0_4.json").write_text(
        json.dumps({"core_numbers": {}, "success_criteria_met": []}),
        encoding="utf-8",
    )
    _write_csv(source / "01_model_selection_lowell_and_high_ell_v0_4.csv", [])
    _write_csv(source / "oph_lowell_fullsky_mc_summary_v0_4.csv", [])

    report = write_inflation_cmb_bridge_report(source, tmp_path / "out")

    assert (tmp_path / "out" / "oph_inflation_cmb_bridge_report.json").exists()
    assert report["receipt_name"] == "COSMOLOGY_PERTURBATION_RECEIPT"
    assert report["physical_cmb_prediction"] is False
    assert (tmp_path / "out" / "oph_cmb_v05_status_ledger.csv").exists()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row})
    if not keys:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
