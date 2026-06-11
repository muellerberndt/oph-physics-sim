from __future__ import annotations

import json
import math
from pathlib import Path

from oph_fpe.cosmology.comparable_data import comparable_data_report
from oph_fpe.cosmology.finite_collar_projection import (
    finite_collar_cmb_projection_report,
    write_finite_collar_cmb_projection_report,
)


def _bundle() -> dict:
    return {
        "mode": "finite_collar_boltzmann_source_bundle_v0",
        "FINITE_COLLAR_BOLTZMANN_SOURCE_BUNDLE_RECEIPT": True,
        "B_A_k_a_diagnostic": {
            "rows": [
                {"a": 1.0, "k": 2.0, "k_units": "inverse_cap_opening_angle_proxy", "B_A": 0.2},
                {
                    "a": 1.0,
                    "theta0": 0.5,
                    "k": 2.0,
                    "k_units": "inverse_cap_opening_angle_proxy",
                    "B_A": -0.1,
                },
                {"a": 0.5, "theta0": 1.0, "k": 1.0, "k_units": "inverse_cap_opening_angle_proxy", "B_A": 0.05},
            ]
        },
        "rho_A_a_diagnostic": {
            "rows": [
                {"a": 1.0, "rho_A": 2.0, "rho_A_eq": 3.0},
                {"a": 0.5, "rho_A": 4.0, "rho_A_eq": 5.0},
            ]
        },
        "Gamma_rec_k_a_diagnostic": {
            "rows": [
                {"a": 1.0, "Gamma_rec_over_H": 0.25},
                {"a": 0.5, "Gamma_rec_over_H": 0.5},
            ]
        },
    }


def test_finite_collar_projection_maps_theta_to_ell_and_k_without_physical_gate():
    report = finite_collar_cmb_projection_report(_bundle(), chi_star_mpc=10_000.0, h=0.5)

    rows = report["projected_B_A_rows"]
    assert report["FINITE_COLLAR_CMB_PROJECTION_DIAGNOSTIC_RECEIPT"] is True
    assert report["PHYSICAL_K_CALIBRATION_RECEIPT"] is False
    assert rows[0]["ell_eff"] == math.pi * 2.0
    assert rows[0]["k_Mpc^-1"] == (math.pi * 2.0 + 0.5) / 10_000.0
    assert rows[0]["k_h_Mpc^-1"] == rows[0]["k_Mpc^-1"] / 0.5
    assert report["background_rows"][0]["q_A_eq_over_A"] == 5.0 / 4.0
    assert report["shape_summary"]["row_count"] == 3
    assert report["physical_cmb_prediction"] is False


def test_write_projection_report_feeds_comparable_data(tmp_path: Path):
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    (run / "finite_collar_boltzmann_bundle_report.json").write_text(json.dumps(_bundle()), encoding="utf-8")

    written = write_finite_collar_cmb_projection_report([run], out, chi_star_mpc=10_000.0, h=0.5)
    comparable = comparable_data_report([out])
    lane = comparable["measurement_lanes"]["finite_collar_cmb_projection"]

    assert (out / "finite_collar_cmb_projection_report.json").exists()
    assert (out / "finite_collar_projected_B_A_rows.csv").exists()
    assert written["FINITE_COLLAR_CMB_PROJECTION_DIAGNOSTIC_RECEIPT"] is True
    assert lane["run_count"] == 1
    assert lane["projection_receipt_count"] == 1
    assert lane["physical_k_receipt_count"] == 0
    assert lane["mean_projected_B_A_row_count"] == 3.0
    assert lane["mean_background_row_count"] == 2.0
