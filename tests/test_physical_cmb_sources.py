from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.cosmology.physical_cmb_prediction import write_physical_cmb_input_report
from oph_fpe.cosmology.physical_cmb_sources import write_physical_cmb_source_readiness_report


def test_physical_cmb_source_readiness_fails_closed_for_proxy_sources(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(run / "no_data_use_receipt.json", {"NO_DATA_USE_RECEIPT": True})
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
        {
            "readiness": {"checks": {"no_cmb_data_used": True}},
            "rows": [{"k_h_mpc": 0.1, "a": 0.5, "B_A_mean": 1.0, "rho_A": 0.2}],
        },
    )

    report = write_physical_cmb_source_readiness_report([run], run)
    input_report = write_physical_cmb_input_report([run], tmp_path / "input")

    assert report["finite_covariant_parent"]["parent_receipt"] is False
    assert report["finite_covariant_parent"]["stress_energy_closure_receipt"] is False
    assert report["finite_covariant_parent"]["gauge_independence_receipt"] is False
    assert report["finite_covariant_parent"]["frozen_likelihood_protocol_receipt"] is False
    assert "finite_certificate_not_theorem_grade" in report["blockers"]
    assert "B_A_kernel_receipt_missing" in report["blockers"]
    assert "official_likelihood_not_ready" in report["blockers"]
    assert input_report["PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT"] is False
    assert "finite_covariant_parent_receipt_missing" in input_report["blockers"]
    assert (run / "finite_covariant_collar_packet_parent_artifact.json").exists()
    assert (run / "finite_covariant_collar_packet_parent_report.json").exists()
    assert (run / "oph_boltzmann_input_report.json").exists()
    assert (run / "finite_collar_boltzmann_bundle_report.json").exists()


def test_physical_cmb_source_readiness_builds_parent_from_theorem_grade_sources(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(run / "no_data_use_receipt.json", {"NO_DATA_USE_RECEIPT": True})
    _write_json(
        run / "finite_repair_transition_matrix_report.json",
        {
            "finite_transition_matrix_ready": True,
            "eta_R_finite_lattice_derived": True,
            "finite_lattice_derived": True,
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
        {
            "B_A_KERNEL_RECEIPT": True,
            "B_A_k_a": [[0.1, 0.5, 1.0]],
            "physical_checks": {
                "energy_momentum_exchange_closed": True,
                "gauge_consistency_audited": True,
                "refinement_convergence_passed": True,
            },
        },
    )
    _write_json(
        run / "B_A_kernel_refinement_report.json",
        {"B_A_KERNEL_REFINEMENT_CONVERGENCE_RECEIPT": True, "patch_counts": [1024, 4096, 16384]},
    )
    _write_json(
        run / "b_a_parent_report.json",
        {
            "readiness": {
                "checks": {
                    "no_cmb_data_used": True,
                    "energy_momentum_exchange_closed": True,
                    "gauge_consistency_audited": True,
                    "refinement_convergence_passed": True,
                }
            },
            "rows": [{"k_h_mpc": 0.1, "a": 0.5, "B_A_mean": 1.0, "rho_A": 0.2}],
        },
    )
    _write_json(
        run / "scale_compressed_repair_report.json",
        {
            "scale_compressed_operator_receipt": True,
            "logical_repair_rounds": 24,
            "cmb_parameter_readouts": {"q_IR": 0.25, "ell_IR": 32.0},
        },
    )
    _write_json(
        run / "screen_capacity_closure_report.json",
        {"SCREEN_CAPACITY_CLOSURE_RECEIPT": True},
    )
    _write_json(run / "strict_neutral_bulk_report.json", {"strict_neutral_bulk": True})
    _write_json(run / "camb_lcdm_baseline_report.json", {"CDM_LIMIT_BOLTZMANN_RECEIPT": True})
    _write_json(
        run / "official_planck_likelihood_readiness_report.json",
        {
            "official_likelihood_execution_ready": True,
            "solver_hash": "sha256:solver",
            "likelihood_hash": "sha256:likelihood",
            "FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT": True,
            "blockers": [],
        },
    )

    report = write_physical_cmb_source_readiness_report([run], run)
    input_report = write_physical_cmb_input_report([run], tmp_path / "input")

    assert report["finite_covariant_parent"]["parent_receipt"] is True
    assert report["finite_covariant_parent"]["stress_energy_closure_receipt"] is True
    assert report["finite_covariant_parent"]["gauge_independence_receipt"] is True
    assert report["finite_covariant_parent"]["causal_response_receipt"] is True
    assert report["finite_covariant_parent"]["refinement_convergence_receipt"] is True
    assert report["finite_covariant_parent"]["frozen_likelihood_protocol_receipt"] is True
    assert report["blockers"] == []
    assert input_report["PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT"] is True
    assert input_report["physical_cmb_prediction_eligible"] is True
    assert input_report["physical_cmb_prediction"] is False


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")
