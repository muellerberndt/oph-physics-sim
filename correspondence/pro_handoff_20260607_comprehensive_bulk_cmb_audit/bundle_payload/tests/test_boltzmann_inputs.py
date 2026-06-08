from __future__ import annotations

from pathlib import Path

import json

from oph_fpe.cosmology.boltzmann_inputs import (
    oph_boltzmann_input_report,
    write_oph_boltzmann_input_report,
)


def test_oph_boltzmann_input_report_separates_cdm_limit_and_diagnostic_proxy():
    oph_cmb = {
        "finite_collar_parent": {
            "weighted_collar_repair_defect_R": 1.5,
            "rho_A_eq_proxy": None,
            "rho_A_eq_proxy_units": "not_available_without_parent_length_scale",
        },
        "diagnostic_kernel_proxy": {
            "a_grid": [0.01, 1.0],
            "k_grid_h_mpc_required_for_boltzmann": [0.1, 0.3],
            "kernel_proxy_rows": [
                {"theta0": 0.5, "k_proxy_inverse_theta": 2.0, "B_A_shape_proxy": -1.0},
                {"theta0": 1.0, "k_proxy_inverse_theta": 1.0, "B_A_shape_proxy": 1.0},
            ],
        },
    }
    camb = {
        "CDM_LIMIT_BOLTZMANN_RECEIPT": True,
        "camb": {
            "lambda_cdm_parameters": {
                "H0": 67.36,
                "ombh2": 0.02237,
                "omch2": 0.1200,
            }
        },
    }

    report = oph_boltzmann_input_report([oph_cmb], camb_baseline_report=camb)

    assert report["physical_cmb_prediction"] is False
    assert report["readiness"]["cdm_limit_solver_ready"] is True
    assert report["readiness"]["diagnostic_repair_exchange_table_ready"] is True
    assert report["readiness"]["physical_prediction_ready"] is False
    assert report["cdm_limit"]["row_count"] == 2
    assert report["diagnostic_repair_exchange"]["row_count"] == 4
    assert report["diagnostic_repair_exchange"]["rows"][0]["Gamma_rec_over_H_shape_proxy"] == 0.6


def test_write_oph_boltzmann_input_report_scans_run_dirs(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "oph_cmb_stress_report.json",
        {
            "finite_collar_parent": {"weighted_collar_repair_defect_R": 1.0},
            "diagnostic_kernel_proxy": {
                "a_grid": [1.0],
                "kernel_proxy_rows": [{"theta0": 1.0, "k_proxy_inverse_theta": 1.0, "B_A_shape_proxy": 0.0}],
            },
        },
    )
    _write_json(run / "camb_lcdm_baseline_report.json", {"CDM_LIMIT_BOLTZMANN_RECEIPT": True})

    report = write_oph_boltzmann_input_report([tmp_path], tmp_path / "out")

    assert report["source_report_count"] == 1
    assert (tmp_path / "out" / "oph_boltzmann_input_report.json").exists()
    assert (tmp_path / "out" / "oph_boltzmann_input_report.md").exists()
    assert (tmp_path / "out" / "oph_boltzmann_cdm_limit_rows.csv").exists()
    assert (tmp_path / "out" / "oph_boltzmann_diagnostic_repair_rows.csv").exists()


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")
