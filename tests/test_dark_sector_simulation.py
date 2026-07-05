from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.cosmology.dark_sector_simulation import (
    dark_sector_simulation_plan,
    write_dark_sector_simulation_plan,
)


def test_dark_sector_plan_marks_first_missing_finite_parent(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "static_galaxy_measurement_report.json",
        {
            "STATIC_GALAXY_RAR_BTFR_RECEIPT": True,
            "OPH_STATIC_GALAXY_BRIDGE_RECEIPT": True,
        },
    )

    report = dark_sector_simulation_plan([run])

    assert report["DARK_SECTOR_SIMULATION_PLAN_RECEIPT"] is True
    assert report["DARK_SECTOR_PHYSICAL_PROMOTION_READY"] is False
    assert report["physical_dark_sector_prediction"] is False
    assert report["first_blocking_stage"] == "finite_covariant_parent"
    assert report["stage_summary"]["static_galaxy"]["gate_passed"] is True
    assert report["stage_summary"]["finite_covariant_parent"]["gate_passed"] is False
    next_item = [
        item for item in report["simulation_suggestions"] if item["stage"] == "finite_covariant_parent"
    ][0]
    assert next_item["next_blocker"] is True
    assert "finite-covariant-collar-parent" in next_item["command"]


def test_dark_sector_plan_writes_promotion_ready_audit_without_physical_prediction(tmp_path: Path) -> None:
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_json(run / "static_galaxy_measurement_report.json", {"STATIC_GALAXY_RAR_BTFR_RECEIPT": True})
    _write_json(
        run / "finite_covariant_collar_packet_parent_report.json",
        {"FINITE_COVARIANT_COLLAR_PACKET_PARENT_RECEIPT": True},
    )
    _write_json(
        run / "finite_collar_boltzmann_bundle_report.json",
        {
            "FINITE_COLLAR_BOLTZMANN_SOURCE_BUNDLE_RECEIPT": True,
            "PHYSICAL_BOLTZMANN_EXPORT_CERTIFICATE": True,
        },
    )
    _write_json(
        run / "oph_boltzmann_input_report.json",
        {
            "readiness": {
                "cdm_limit_solver_ready": True,
                "diagnostic_repair_exchange_table_ready": True,
            }
        },
    )
    _write_json(
        run / "frozen_transfer_likelihood_report.json",
        {
            "FROZEN_TRANSFER_LIKELIHOOD_CLOSURE_RECEIPT": True,
            "FROZEN_PHYSICAL_SPECTRUM_RECEIPT": True,
            "LIKELIHOOD_EVALUATED_PHYSICAL_PREDICTION_RECEIPT": True,
        },
    )
    _write_json(run / "cmb_anomaly_report.json", {"CMB_ANOMALY_DIAGNOSTIC_RECEIPT": True})

    report = write_dark_sector_simulation_plan([run], out)

    assert report["first_blocking_stage"] is None
    assert report["DARK_SECTOR_PHYSICAL_PROMOTION_READY"] is True
    assert report["physical_dark_sector_prediction"] is False
    assert report["stage_summary"]["finite_collar_boltzmann"]["physical_certificate"] is True
    assert report["stage_summary"]["finite_collar_boltzmann"]["exact_uniform_lambda_claim_ready"] is False
    assert "UNIFORM_PRODUCT_THICKENING_EXACT" in report["stage_summary"]["finite_collar_boltzmann"][
        "exact_uniform_target_blockers"
    ]
    assert report["stage_summary"]["boltzmann_inputs"]["cdm_limit_solver_ready"] is True
    assert report["stage_summary"]["frozen_likelihood"]["physical_spectrum_receipt"] is True
    assert (out / "dark_sector_simulation_plan.json").exists()
    assert (out / "dark_sector_simulation_plan.md").exists()


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")
