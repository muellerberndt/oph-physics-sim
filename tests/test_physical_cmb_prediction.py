from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.cosmology.physical_cmb_prediction import (
    write_physical_cmb_frontier_report,
    write_physical_cmb_input_no_data_use_receipt,
    write_physical_cmb_input_report,
    write_physical_cmb_promotion_audit_report,
)


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
        {
            "theorem_grade_finite_inputs": True,
            "derived_outputs": {
                "A_zeta": 2.1e-9,
                "rho_A_a": [[0.5, 0.2]],
                "screen_to_primordial_lift_receipt": True,
            },
        },
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


def test_physical_cmb_input_no_data_use_receipt_only_certifies_firewall(tmp_path: Path):
    run = tmp_path / "run"
    receipt_dir = tmp_path / "receipt"
    out = tmp_path / "out"
    run.mkdir()
    _write_json(
        run / "finite_repair_transition_matrix_report.json",
        {
            "finite_transition_matrix_ready": True,
            "eta_R_empirical_finite_lattice_derived": True,
            "clock_modes": {"empirical": {"eta_R_value": 0.12}},
            "primary": {"gamma_continuous": 0.12},
        },
    )
    _write_json(
        run / "b_a_parent_report.json",
        {
            "readiness": {"checks": {"no_cmb_data_used": True}},
            "rows": [{"k_h_mpc": 0.1, "a": 0.5, "B_A_mean": 1.0, "base_epsilon_cmi": 0.2}],
        },
    )
    _write_json(
        run / "official_planck_likelihood_readiness_report.json",
        {"official_likelihood_execution_ready": False},
    )

    receipt = write_physical_cmb_input_no_data_use_receipt([run], receipt_dir)
    report = write_physical_cmb_input_report([receipt_dir, run], out)

    assert receipt["NO_DATA_USE_RECEIPT"] is True
    assert receipt["source_status"]["official_planck_likelihood_readiness_report"]["measurement_data_used"] is False
    assert "no_data_use_receipt_false" not in report["blockers"]
    assert "B_A_k_a_missing_or_not_finite" in report["blockers"]
    assert report["physical_cmb_prediction_eligible"] is False


def test_physical_cmb_input_report_ignores_empty_shadow_receipts(tmp_path: Path):
    stale = tmp_path / "stale"
    fresh = tmp_path / "fresh"
    out = tmp_path / "out"
    stale.mkdir()
    fresh.mkdir()
    _write_json(stale / "no_data_use_receipt.json", {})
    _write_json(fresh / "no_data_use_receipt.json", {"NO_DATA_USE_RECEIPT": True})

    report = write_physical_cmb_input_report([stale, fresh], out)

    assert "no_data_use_receipt_false" not in report["blockers"]
    assert report["source_summary"]["no_data_use_receipt"]["present"] is True


def test_physical_cmb_input_report_accepts_camb_baseline_cdm_limit_gate(tmp_path: Path):
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_json(run / "no_data_use_receipt.json", {"NO_DATA_USE_RECEIPT": True})
    _write_json(run / "camb_lcdm_baseline_report.json", {"CDM_LIMIT_BOLTZMANN_RECEIPT": True})

    report = write_physical_cmb_input_report([run], out)

    assert "cdm_limit_regression_not_passed" not in report["blockers"]
    assert "official_likelihood_not_ready" in report["blockers"]
    assert report["source_summary"]["camb_lcdm_baseline_report"]["present"] is True


def test_physical_cmb_input_report_accepts_official_readiness_likelihood_gate(tmp_path: Path):
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_json(run / "no_data_use_receipt.json", {"NO_DATA_USE_RECEIPT": True})
    _write_json(run / "camb_lcdm_baseline_report.json", {"CDM_LIMIT_BOLTZMANN_RECEIPT": True})
    _write_json(
        run / "official_planck_likelihood_readiness_report.json",
        {
            "mode": "official_planck_likelihood_readiness_v0",
            "official_likelihood_execution_ready": True,
            "official_planck_likelihood_data_paths_configured": True,
            "official_clik_api_available": True,
            "camb_available": True,
            "cobaya_available": True,
            "blockers": [],
        },
    )

    report = write_physical_cmb_input_report([run], out)
    summary = report["source_summary"]["official_planck_likelihood_readiness_report"]

    assert "official_likelihood_not_ready" not in report["blockers"]
    assert "A_zeta_not_finite_derived" in report["blockers"]
    assert summary["present"] is True
    assert summary["official_likelihood_execution_ready"] is True
    assert summary["official_planck_likelihood_data_paths_configured"] is True
    assert summary["official_clik_api_available"] is True


def test_physical_cmb_input_report_uses_paired_parent_rho_A_rows(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "b_a_parent_report.json",
        {
            "mode": "paired_cap_collar_perturb_resettle_B_A_parent_v0",
            "readiness": {"checks": {"no_cmb_data_used": True}},
            "rows": [
                {"k_h_mpc": 0.1, "a": 0.5, "B_A_mean": 1.0, "rho_A": 7.5},
                {"k_h_mpc": 0.2, "a": 1.0, "B_A_mean": 2.0, "rho_A_base": 4.0},
            ],
        },
    )

    out = tmp_path / "out"
    report = write_physical_cmb_input_report([run], out)
    contract = json.loads((out / "physical_cmb_input_contract.json").read_text(encoding="utf-8"))

    assert contract["rho_A_a"] == [[0.5, 7.5], [1.0, 4.0]]
    assert report["input_status"]["B_A_k_a"]["diagnostic_value_present"] is True
    assert report["input_status"]["B_A_k_a"]["source_is_finite_cmb_source"] is False
    assert report["input_status"]["rho_A_a"]["diagnostic_value_present"] is True
    assert report["input_status"]["rho_A_a"]["source_is_finite_cmb_source"] is False
    assert report["physical_cmb_prediction"] is False
    assert "B_A_k_a_missing_or_not_finite" in report["blockers"]


def test_physical_cmb_input_report_prefers_kernel_candidate_as_diagnostic(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "B_A_kernel_report.json",
        {
            "B_A_KERNEL_CANDIDATE_RECEIPT": True,
            "B_A_KERNEL_RECEIPT": False,
            "B_A_k_a": [[0.1, 0.5, 1.0, 0.01, 4]],
        },
    )
    _write_json(
        run / "b_a_parent_report.json",
        {"rows": [{"k_h_mpc": 0.2, "a": 0.5, "B_A_mean": 2.0, "rho_A": 0.3}]},
    )

    out = tmp_path / "out"
    report = write_physical_cmb_input_report([run], out)
    contract = json.loads((out / "physical_cmb_input_contract.json").read_text())

    assert contract["B_A_k_a"] == [[0.1, 0.5, 1.0]]
    assert report["input_status"]["B_A_k_a"]["row_count"] == 1
    assert report["input_status"]["B_A_k_a"]["physical_gate_passed"] is False
    assert "B_A_k_a_missing_or_not_finite" in report["blockers"]


def test_physical_cmb_input_report_labels_proxy_finite_certificate_inputs(tmp_path: Path):
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_json(run / "no_data_use_receipt.json", {"no_data_use_receipt": True})
    _write_json(
        run / "finite_certificate_report.json",
        {
            "mode": "oph_finite_cosmology_certificate_bundle_v0",
            "finite_certificate_compiler_ready": True,
            "theorem_grade_finite_inputs": False,
            "proxy_certificate": True,
            "derived_outputs": {
                "A_zeta": 0.7,
                "rho_A_a": [[0.5, 0.2]],
            },
        },
    )

    report = write_physical_cmb_input_report([run], out)

    assert report["source_summary"]["finite_certificate_report"]["present"] is True
    assert report["source_summary"]["finite_certificate_report"]["finite_certificate_compiler_ready"] is True
    assert report["source_summary"]["finite_certificate_report"]["theorem_grade_finite_inputs"] is False
    assert report["source_summary"]["finite_certificate_report"]["proxy_certificate"] is True


def test_physical_cmb_frontier_keeps_measurement_outputs_below_prediction_gate(tmp_path: Path):
    run = tmp_path / "run"
    out = tmp_path / "frontier"
    run.mkdir()
    _write_json(run / "no_data_use_receipt.json", {"NO_DATA_USE_RECEIPT": True})
    _write_json(run / "camb_lcdm_baseline_report.json", {"CDM_LIMIT_BOLTZMANN_RECEIPT": True})
    _write_json(
        run / "official_planck_likelihood_readiness_report.json",
        {"official_likelihood_execution_ready": False},
    )
    _write_json(
        run / "finite_certificate_report.json",
        {
            "finite_certificate_compiler_ready": True,
            "theorem_grade_finite_inputs": False,
            "proxy_certificate": True,
            "derived_outputs": {"A_zeta": 2.1e-9, "rho_A_a": [[0.5, 0.2]]},
        },
    )
    _write_json(
        run / "B_A_kernel_report.json",
        {"B_A_KERNEL_CANDIDATE_RECEIPT": True, "B_A_KERNEL_RECEIPT": False, "row_count": 1},
    )
    _write_json(
        run / "B_A_kernel_refinement_report.json",
        {
            "two_scale_diagnostic_receipt": True,
            "B_A_KERNEL_REFINEMENT_CONVERGENCE_RECEIPT": False,
            "patch_counts": [4096, 16384],
        },
    )
    _write_json(
        run / "physical_cmb_output_comparison_report.json",
        {
            "PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT": True,
            "PHYSICAL_CMB_PREDICTION_RECEIPT": False,
            "measurement_comparable_model_count": 3,
            "oph_diagnostic_model_count": 2,
            "best_oph_diagnostic_model": {
                "model_id": "scale_compressed_ir_kernel",
                "amplitude_fit_chi2_per_bin": 0.94,
            },
        },
    )

    report = write_physical_cmb_frontier_report([run], out)
    gates = {row["gate"]: row["passed"] for row in report["gate_rows"]}

    assert report["PHYSICAL_CMB_FRONTIER_REPORT"] is True
    assert report["physical_cmb_output_comparison_receipt"] is True
    assert report["physical_cmb_prediction_receipt"] is False
    assert report["physical_cmb_prediction_ready"] is False
    assert gates["measurement_comparable_cmb_outputs"] is True
    assert gates["finite_theorem_A_zeta"] is False
    assert gates["finite_B_A_kernel"] is False
    assert gates["official_planck_likelihood_ready"] is False
    assert "A_zeta_not_finite_derived" in report["blockers"]
    assert "official_likelihood_not_ready" in report["blockers"]
    assert (out / "physical_cmb_frontier_report.json").exists()
    assert (out / "physical_cmb_frontier_report.md").exists()
    assert report["input_status"]["A_zeta"]["diagnostic_value_present"] is True
    assert report["input_status"]["A_zeta"]["source_is_finite_cmb_source"] is False
    assert report["input_status"]["A_zeta"]["physical_gate_passed"] is False
    assert "A_zeta_not_finite_derived" in report["blockers"]


def test_physical_cmb_input_report_labels_observed_screen_capacity(tmp_path: Path):
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_json(run / "no_data_use_receipt.json", {"no_data_use_receipt": True})
    _write_json(
        run / "screen_capacity_closure_report.json",
        {
            "mode": "oph_screen_capacity_closure_v0",
            "observed_branch_normalization": {"N_CRC": 3.3e122},
            "readiness_gates": {"observed_branch_N_scr_readout_available": True},
        },
    )

    report = write_physical_cmb_input_report([run], out)
    contract = json.loads((out / "physical_cmb_input_contract.json").read_text())

    assert contract["N_source"] == "OPH_screen_capacity_observed_branch_readout"
    assert report["physical_cmb_prediction"] is False


def test_physical_cmb_promotion_audit_names_proxy_blockers(tmp_path: Path):
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_json(run / "no_data_use_receipt.json", {"NO_DATA_USE_RECEIPT": True})
    _write_json(run / "camb_lcdm_baseline_report.json", {"CDM_LIMIT_BOLTZMANN_RECEIPT": True})
    _write_json(
        run / "finite_certificate_report.json",
        {
            "finite_certificate_compiler_ready": True,
            "finite_certificate_stack_ready": True,
            "theorem_grade_finite_inputs": False,
            "proxy_certificate": True,
            "derived_outputs": {"A_zeta": 0.7, "rho_A_a": [[0.5, 0.2]]},
        },
    )
    _write_json(
        run / "b_a_parent_report.json",
        {
            "readiness": {"checks": {"no_cmb_data_used": True}},
            "rows": [{"k_h_mpc": 0.1, "a": 0.5, "B_A_mean": 1.0, "rho_A": 0.2}],
        },
    )
    _write_json(
        run / "B_A_kernel_report.json",
        {
            "B_A_KERNEL_CANDIDATE_RECEIPT": True,
            "B_A_KERNEL_RECEIPT": False,
            "row_count": 1,
            "promotion_blockers": ["physical_check_failed_scale_calibrated_k_h_mpc"],
        },
    )
    _write_json(
        run / "B_A_kernel_refinement_report.json",
        {
            "two_scale_diagnostic_receipt": True,
            "B_A_KERNEL_REFINEMENT_CONVERGENCE_RECEIPT": False,
            "blockers": ["requires_at_least_three_patch_counts_for_refinement_convergence"],
        },
    )

    report = write_physical_cmb_promotion_audit_report([run], out)

    assert report["physical_cmb_promotion_ready"] is False
    assert report["physical_cmb_prediction"] is False
    assert report["no_data_use_receipt"] is True
    assert report["cdm_limit_regression_passed"] is True
    assert "finite_certificate_proxy_not_theorem_grade" in report["promotion_blockers"]
    assert "B_A_kernel_receipt_missing" in report["promotion_blockers"]
    assert "B_A_kernel_candidate_not_physical" in report["promotion_blockers"]
    assert "B_A_kernel_physical_check_failed_scale_calibrated_k_h_mpc" in report["promotion_blockers"]
    assert "B_A_kernel_refinement_convergence_not_passed" in report["promotion_blockers"]
    assert (
        "B_A_kernel_refinement_requires_at_least_three_patch_counts_for_refinement_convergence"
        in report["promotion_blockers"]
    )
    assert "A_zeta_diagnostic_proxy_not_physical_source" in report["promotion_blockers"]
    assert "official_likelihood_not_ready" in report["promotion_blockers"]
    assert (out / "physical_cmb_promotion_audit_report.json").exists()
    assert (out / "physical_cmb_promotion_audit_report.md").exists()


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")
