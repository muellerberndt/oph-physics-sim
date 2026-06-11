from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.cosmology.comparable_data import comparable_data_report
from oph_fpe.cosmology.finite_collar_boltzmann_bundle import (
    finite_collar_boltzmann_bundle_report,
    write_finite_collar_boltzmann_bundle_report,
)


def test_finite_collar_boltzmann_bundle_collects_source_tables_but_blocks_physical_claim(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(run / "no_data_use_receipt.json", {"NO_DATA_USE_RECEIPT": True})
    _write_json(
        run / "b_a_parent_report.json",
        {
            "mode": "paired_cap_collar_perturb_resettle_B_A_parent_v0",
            "primary_parent_source": "paired_cap_collar_perturb_resettle_rerun",
            "B_A_PAIRED_DIAGNOSTIC_RECEIPT": True,
            "readiness": {
                "B_A_PAIRED_DIAGNOSTIC_RECEIPT": True,
                "checks": {"no_cmb_data_used": True, "controls_fail": True},
            },
            "rows": [
                {
                    "a": 0.5,
                    "k_h_mpc": 0.1,
                    "k_units": "inverse_cap_opening_angle_proxy",
                    "B_A_mean": 1.25,
                    "B_A_sem": 0.02,
                    "rho_A": 7.0,
                    "rho_A_eq_plus_mean": 7.4,
                    "rho_A_eq_minus_mean": 6.6,
                    "delta_baryon": 0.01,
                    "source": "paired_cap_collar_perturb_resettle_rerun",
                }
            ],
        },
    )
    _write_json(
        run / "finite_repair_transition_matrix_report.json",
        {
            "finite_transition_matrix_ready": True,
            "primary": {"gamma_continuous": 0.125, "lambda_2": 0.882, "eta_R_estimate": 0.035},
        },
    )

    report = finite_collar_boltzmann_bundle_report([run])

    assert report["FINITE_COLLAR_BOLTZMANN_SOURCE_BUNDLE_RECEIPT"] is True
    assert report["PHYSICAL_BOLTZMANN_EXPORT_CERTIFICATE"] is False
    assert report["physical_cmb_prediction"] is False
    assert report["B_A_k_a_diagnostic"]["row_count"] == 1
    assert report["rho_A_a_diagnostic"]["row_count"] == 1
    assert report["Gamma_rec_k_a_diagnostic"]["row_count"] == 1
    assert "physical_k_units_calibrated" in report["readiness"]["physical_missing_gates"]
    assert "official_likelihood_not_ready" in report["physical_cmb_input_validation"]["blockers"]


def test_write_finite_collar_boltzmann_bundle_feeds_comparable_data(tmp_path: Path):
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_json(run / "no_data_use_receipt.json", {"NO_DATA_USE_RECEIPT": True})
    _write_json(
        run / "b_a_parent_report.json",
        {
            "B_A_PAIRED_DIAGNOSTIC_RECEIPT": True,
            "readiness": {
                "B_A_PAIRED_DIAGNOSTIC_RECEIPT": True,
                "checks": {"no_cmb_data_used": True, "controls_fail": True},
            },
            "rows": [
                {
                    "a": 1.0,
                    "k_proxy_inverse_theta": 2.0,
                    "B_A_mean": -0.5,
                    "rho_A_base": 3.0,
                    "rho_A_eq_plus_mean": 3.2,
                    "rho_A_eq_minus_mean": 2.8,
                }
            ],
        },
    )
    _write_json(
        run / "finite_repair_transition_matrix_report.json",
        {"finite_transition_matrix_ready": True, "primary": {"gamma_continuous": 0.25}},
    )

    written = write_finite_collar_boltzmann_bundle_report([run], out)
    comparable = comparable_data_report([out])
    lane = comparable["measurement_lanes"]["finite_collar_boltzmann_source_bundle"]

    assert (out / "finite_collar_boltzmann_bundle_report.json").exists()
    assert (out / "finite_collar_B_A_k_a_diagnostic.csv").exists()
    assert written["FINITE_COLLAR_BOLTZMANN_SOURCE_BUNDLE_RECEIPT"] is True
    assert lane["run_count"] == 1
    assert lane["diagnostic_bundle_receipt_count"] == 1
    assert lane["physical_certificate_count"] == 0
    assert lane["mean_B_A_row_count"] == 1.0
    assert lane["mean_rho_A_row_count"] == 1.0
    assert lane["mean_Gamma_rec_row_count"] == 1.0


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")
