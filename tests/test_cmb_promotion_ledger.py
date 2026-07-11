from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.cosmology.cmb_promotion_ledger import (
    PHYSICAL_PREDICTION_PARENT_GATES,
    _frozen_parent_hashes_receipt,
    _physical_prediction_receipt,
    write_cmb_promotion_ledger_report,
)
from oph_fpe.cosmology.cosmological_scale_bridge import imported_flrw_reference_receipts
from oph_fpe.cli import main


def test_cmb_promotion_ledger_empty_run_fails_closed(tmp_path: Path):
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()

    report = write_cmb_promotion_ledger_report([run], out)

    assert report["current_claim_tier"] == "UNSTARTED_OR_INVALIDATED"
    assert report["likelihood_evaluated_physical_cmb_prediction"] is False
    assert report["readiness_gates"]["PHYSICAL_CMB_PREDICTION_RECEIPT"] is False
    assert "visual_diagnostic_receipt_missing" in report["blockers"]
    assert (out / "cmb_promotion_ledger_report.json").exists()
    assert (out / "cmb_promotion_ledger_report.md").exists()


def test_cmb_promotion_ledger_keeps_imported_flrw_diagnostic_bounded(tmp_path: Path):
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_json(run / "physical_scale_bridge_report.json", imported_flrw_reference_receipts(hash_seed="cmb02"))
    _write_json(
        run / "physical_cmb_output_comparison_report.json",
        {
            "PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT": True,
            "PHYSICAL_CMB_PREDICTION_RECEIPT": False,
            "best_oph_diagnostic_model": {"model_id": "oph_diagnostic"},
            "best_oph_residual_summary": {"available": True, "bin_count": 2},
        },
    )

    report = write_cmb_promotion_ledger_report([run], out)

    assert report["conditional_physical_scale_bridge_ready"] is True
    assert report["oph_native_geometry_ready"] is False
    assert report["readiness_gates"]["CONDITIONAL_IMPORTED_FLRW_GEOMETRY_RECEIPT"] is True
    assert report["readiness_gates"]["OPH_NATIVE_COSMO_GEOM_READ_RECEIPT"] is False
    assert report["current_claim_tier"] == "SPECTRUM_DIAGNOSTIC"
    assert report["likelihood_evaluated_physical_cmb_prediction"] is False
    assert report["readiness_gates"]["SOURCE_ONLY_FINITE_ARTIFACT_RECEIPT"] is False


def test_cmb_promotion_ledger_rejects_untrusted_terminal_prediction_boolean(tmp_path: Path):
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_json(
        run / "physical_cmb_output_comparison_report.json",
        {
            "PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT": True,
            "PHYSICAL_CMB_PREDICTION_RECEIPT": True,
            "physical_cmb_prediction": True,
        },
    )

    report = write_cmb_promotion_ledger_report([run], out)

    assert report["terminal_prediction_asserted_by_output"] is True
    assert report["current_claim_tier"] == "SPECTRUM_DIAGNOSTIC"
    assert report["likelihood_evaluated_physical_cmb_prediction"] is False
    assert report["readiness_gates"]["PHYSICAL_CMB_PREDICTION_RECEIPT"] is False
    assert report["readiness_gates"]["FROZEN_PARENT_HASHES_RECEIPT"] is False
    assert "untrusted_terminal_prediction_assertion_rejected" in report["blockers"]


def test_physical_prediction_receipt_is_derived_from_all_parent_gates():
    gates = {gate: True for gate in PHYSICAL_PREDICTION_PARENT_GATES}

    assert _physical_prediction_receipt(
        gates,
        conditional_physical_source=True,
        native_physical_source=False,
    ) is True

    gates["FROZEN_PARENT_HASHES_RECEIPT"] = False
    assert _physical_prediction_receipt(
        gates,
        conditional_physical_source=True,
        native_physical_source=False,
    ) is False


def test_frozen_parent_hashes_must_be_valid_and_match_input_contract():
    frozen = {
        "frozen_source_hash": "0" * 64,
        "frozen_solver_hash": "1" * 64,
        "frozen_likelihood_hash": "2" * 64,
    }
    physical_input = {"contract": dict(frozen)}

    assert _frozen_parent_hashes_receipt(frozen, physical_input) is True

    physical_input["contract"]["frozen_solver_hash"] = "3" * 64
    assert _frozen_parent_hashes_receipt(frozen, physical_input) is False


def test_cmb_promotion_ledger_cli(tmp_path: Path):
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_json(run / "cmb_static_plots_summary.json", {"plot_count": 1})

    assert main(["cmb-promotion-ledger", "--run-dir", str(run), "--out", str(out)]) == 0
    report = json.loads((out / "cmb_promotion_ledger_report.json").read_text(encoding="utf-8"))

    assert report["current_claim_tier"] == "VISUAL_DIAGNOSTIC"
    assert report["readiness_gates"]["VISUAL_DIAGNOSTIC_RECEIPT"] is True


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
