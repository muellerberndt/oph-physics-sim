from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.cosmology.physical_cmb_prediction import write_physical_cmb_input_report


def test_physical_cmb_input_report_blocks_current_diagnostics(tmp_path: Path):
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_json(run / "no_data_use_receipt.json", {"no_data_use_receipt": True})
    _write_json(
        run / "finite_repair_transition_matrix_report.json",
        {
            "finite_transition_matrix_ready": True,
            "eta_R_finite_lattice_derived": False,
            "primary": {"eta_R_estimate": 0.035, "gamma_continuous": 0.1},
        },
    )
    _write_json(
        run / "b_a_parent_report.json",
        {"rows": [{"k_h_mpc": 0.1, "a": 0.5, "B_A_mean": 1.0, "base_epsilon_cmi": 0.2}]},
    )

    report = write_physical_cmb_input_report([run], out)

    assert report["PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT"] is False
    assert report["physical_cmb_prediction"] is False
    assert "eta_R_not_finite_derived" in report["blockers"]
    assert "B_A_k_a_missing_or_not_finite" in report["blockers"]
    assert (out / "physical_cmb_input_contract.json").exists()


def test_physical_cmb_input_report_can_pass_only_with_finite_sources(tmp_path: Path):
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_json(run / "no_data_use_receipt.json", {"no_data_use_receipt": True})
    _write_json(
        run / "finite_repair_transition_matrix_report.json",
        {
            "finite_transition_matrix_ready": True,
            "eta_R_finite_lattice_derived": True,
            "primary": {"eta_R_estimate": 0.035, "gamma_continuous": 0.1},
        },
    )
    _write_json(
        run / "finite_certificate_report.json",
        {"theorem_grade_finite_inputs": True, "derived_outputs": {"A_zeta": 2.1e-9, "rho_A_a": [[0.5, 0.2]]}},
    )
    _write_json(
        run / "B_A_kernel_report.json",
        {"B_A_KERNEL_RECEIPT": True, "B_A_k_a": [[0.1, 0.5, 1.0]]},
    )
    _write_json(
        run / "b_a_parent_report.json",
        {"rows": [{"k_h_mpc": 0.1, "a": 0.5, "B_A_mean": 1.0, "base_epsilon_cmi": 0.2}]},
    )
    _write_json(
        run / "scale_compressed_repair_report.json",
        {
            "scale_compressed_operator_receipt": True,
            "logical_repair_rounds": 24,
            "cmb_parameter_readouts": {"q_IR": 0.25, "ell_IR": 32.0},
        },
    )
    _write_json(run / "strict_neutral_bulk_report.json", {"strict_neutral_bulk": True})
    _write_json(
        run / "oph_compressed_likelihood_report.json",
        {"official_likelihood_ready": True, "cdm_limit_regression_passed": True},
    )

    report = write_physical_cmb_input_report([run], out)

    assert report["PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT"] is True
    assert report["physical_cmb_prediction_eligible"] is True
    assert report["physical_cmb_prediction"] is False
    assert report["blockers"] == []


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")
