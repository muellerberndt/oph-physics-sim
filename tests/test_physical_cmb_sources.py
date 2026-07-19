from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.cosmology.cosmological_scale_bridge import imported_flrw_reference_receipts
from oph_fpe.cosmology.physical_cmb_prediction import write_physical_cmb_input_report
from oph_fpe.cosmology.physical_cmb_sources import (
    build_finite_covariant_parent_artifact_from_reports,
    build_finite_parent_readiness_summary_from_reports,
    write_physical_cmb_source_readiness_report,
)


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
            "rows": [
                {
                    "k_h_mpc": 0.1,
                    "k_units": "Mpc^-1",
                    "physical_calibration": True,
                    "a": 0.5,
                    "B_A_mean": 1.0,
                    "rho_A": 0.2,
                }
            ],
        },
    )

    report = write_physical_cmb_source_readiness_report([run], run)
    input_report = write_physical_cmb_input_report([run], tmp_path / "input")

    assert report["finite_covariant_parent"]["parent_receipt"] is False
    assert report["finite_covariant_parent"]["stress_energy_closure_receipt"] is False
    assert report["finite_covariant_parent"]["gauge_independence_receipt"] is False
    assert report["finite_covariant_parent"]["frozen_likelihood_protocol_receipt"] is False
    assert report["finite_covariant_parent"]["readiness_summary_written"] is True
    assert report["finite_covariant_parent"]["candidate_artifact_written"] is False
    assert "explicit_finite_covariant_parent_artifact_missing" in report["blockers"]
    assert "finite_certificate_not_theorem_grade" in report["blockers"]
    assert "B_A_kernel_receipt_missing" in report["blockers"]
    assert "official_likelihood_not_ready" in report["blockers"]
    assert "finite_transition_clock_certified" not in report["blockers"]
    assert "finite_collar_parent_theorem_grade" not in report["blockers"]
    assert "finite_collar_boltzmann_missing_physical_cmb_input_contract_passed" in report["blockers"]
    assert "finite_transition_clock_certified" in report["oph_boltzmann_input"]["diagnostic_missing_gates"]
    assert input_report["PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT"] is False
    assert "finite_covariant_parent_receipt_missing" in input_report["blockers"]
    assert (run / "finite_parent_readiness_summary.json").exists()
    assert not (run / "finite_covariant_collar_packet_parent_artifact.json").exists()
    assert (run / "oph_boltzmann_input_report.json").exists()
    assert (run / "finite_collar_boltzmann_bundle_report.json").exists()


def test_physical_cmb_source_readiness_uses_explicit_parent_report(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(run / "no_data_use_receipt.json", {"NO_DATA_USE_RECEIPT": True})
    _write_json(
        run / "finite_repair_transition_matrix_report.json",
        {
            "finite_transition_matrix_ready": True,
            "eta_R_finite_lattice_derived": True,
            "finite_lattice_derived": True,
            "state_count": 2,
            "transition_count": 48,
            "primary": {
                "eta_R_estimate": 0.035,
                "gamma_continuous": 0.1,
                "finite": True,
                "irreducible": True,
                "aperiodic": True,
                "lambda_2": 0.5,
                "detailed_balance_max_abs_error": 0.0,
            },
        },
    )
    _write_json(
        run / "finite_certificate_report.json",
        {
            "theorem_grade_finite_inputs": True,
            "RHO_A_TRANSPORT_RECEIPT": True,
            "ANOMALY_ABUNDANCE_SOURCE_RECEIPT": True,
            "RHO_A_SOURCE_RECEIPT": True,
            "rho_A_claim_label": "SOURCE_ONLY_ANOMALY_ABUNDANCE",
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
            "rows": [
                {
                    "k_h_mpc": 0.1,
                    "k_units": "Mpc^-1",
                    "physical_calibration": True,
                    "a": 0.5,
                    "B_A_mean": 1.0,
                    "rho_A": 0.2,
                }
            ],
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
        {
            "PHYSICAL_N_CLOSURE_RECEIPT": True,
            "complete_terminal_fiber_receipt": True,
            "whole_fiber_scalarization_receipt": True,
            "target_free_capacity_producer_receipt": True,
            "robust_closure_receipt": True,
            "unique_regulator_stable_slack_zero_receipt": True,
            "horizon_record_saturation_receipt": True,
            "readiness_gates": {
                "finite_correctable_public_record_evaluator_implemented": True
            },
        },
    )
    _write_json(run / "strict_neutral_bulk_report.json", {"strict_neutral_bulk": True, "freezeout_cycle": 24})
    _write_json(run / "camb_lcdm_baseline_report.json", {"CDM_LIMIT_BOLTZMANN_RECEIPT": True})
    _write_json(
        run / "official_planck_likelihood_readiness_report.json",
        {
            "official_likelihood_execution_ready": True,
            "solver_hash": _hash("1"),
            "likelihood_hash": _hash("2"),
            "FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT": True,
            "rollup": "global",
            "blockers": [],
        },
    )
    _write_json(run / "finite_covariant_collar_packet_parent_artifact.json", _parent_artifact())
    _write_clean_source_provenance(run)
    _write_physical_scale_bridge(run)
    _write_frozen_transfer(run)

    report = write_physical_cmb_source_readiness_report([run], run)
    input_report = write_physical_cmb_input_report([run], tmp_path / "input")

    assert report["finite_covariant_parent"]["parent_receipt"] is True
    assert report["finite_covariant_parent"]["stress_energy_closure_receipt"] is True
    assert report["finite_covariant_parent"]["gauge_independence_receipt"] is True
    assert report["finite_covariant_parent"]["causal_response_receipt"] is True
    assert report["finite_covariant_parent"]["refinement_convergence_receipt"] is True
    assert report["finite_covariant_parent"]["frozen_likelihood_protocol_receipt"] is False
    assert report["blockers"] == []
    assert input_report["PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT"] is True
    assert input_report["physical_cmb_prediction_eligible"] is True
    assert input_report["physical_cmb_prediction"] is False


def test_source_readiness_source_hash_ignores_likelihood_side_reports(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "finite_repair_transition_matrix_report.json",
        {"finite_transition_matrix_ready": True, "primary": {"gamma_continuous": 0.1}},
    )
    _write_json(run / "official_planck_likelihood_readiness_report.json", {"solver_hash": _hash("1")})

    artifact_a, status_a = build_finite_covariant_parent_artifact_from_reports([run])
    _write_json(
        run / "official_planck_likelihood_readiness_report.json",
        {"solver_hash": _hash("2"), "likelihood_hash": _hash("3"), "rollup": "global"},
    )
    artifact_b, status_b = build_finite_covariant_parent_artifact_from_reports([run])
    summary, summary_status = build_finite_parent_readiness_summary_from_reports([run])

    assert artifact_a["manifest"]["source_hash"] == artifact_b["manifest"]["source_hash"]
    assert status_a["source_hash"] == status_b["source_hash"]
    assert artifact_a["not_a_model_artifact"] is True
    assert status_a["candidate_artifact_written"] is False
    assert summary["Gamma_rec"] is None
    assert summary_status["readiness_summary_written"] is True


def test_source_readiness_rejects_stale_true_transition_flags(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "finite_repair_transition_matrix_report.json",
        {
            "finite_transition_matrix_ready": True,
            "finite_lattice_derived": True,
            "eta_R_finite_lattice_derived": True,
            "state_count": 2,
            "transition_count": 48,
            "primary": {
                "finite": True,
                "irreducible": False,
                "aperiodic": False,
                "lambda_2": 1.0,
                "detailed_balance_max_abs_error": None,
                "eta_R_estimate": 0.035,
                "gamma_continuous": 0.1,
            },
        },
    )

    summary, _status = build_finite_parent_readiness_summary_from_reports([run])

    assert summary["source_gate_status"]["finite_transition_ready"] is False
    assert summary["transition_clock_eligibility"]["eligible"] is False
    assert "primary_irreducible" in summary["transition_clock_eligibility"]["blockers"]


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_physical_scale_bridge(run: Path) -> None:
    _write_json(run / "physical_scale_bridge_report.json", imported_flrw_reference_receipts())


def _write_frozen_transfer(run: Path) -> None:
    _write_json(
        run / "frozen_transfer_likelihood_report.json",
        {
            "FROZEN_SOURCE_MANIFEST_RECEIPT": True,
            "SOLVER_ASSUMPTION_PIN_RECEIPT": True,
            "CMB1_CUSTOM_PARENT_CDM_LIMIT_REGRESSION_RECEIPT": True,
            "CDM_LIMIT_REGRESSION_RECEIPT": True,
            "STANDARD_MODEL_OFF_REGRESSION_RECEIPT": True,
            "BLINDED_COMPARISON_SETUP_RECEIPT": True,
            "FULL_OBSERVABLE_LIKELIHOOD_RECEIPT": True,
            "FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT": True,
            "FROZEN_TRANSFER_LIKELIHOOD_CLOSURE_RECEIPT": True,
            "official_likelihood_execution_ready": True,
            "frozen_source_hash": _hash("0"),
            "frozen_solver_hash": _hash("1"),
            "frozen_likelihood_hash": _hash("2"),
            "blockers": [],
        },
    )


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _identity4() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _parent_artifact() -> dict:
    return {
        "manifest": {
            "source_hash": _hash("0"),
            "regulator_id": "N64_eps0",
            "parent_theorem_version": "fccpp-v1",
        },
        "geometry": {
            "metric": [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "transport": {"parallel_transport_maps": [{"matrix": _identity4()}]},
        "packets": {
            "states": [
                {
                    "label": "anomaly",
                    "rho": 0.2,
                    "occupation": 1.0,
                    "invariant_weight": 1.0,
                    "mass": 1.0,
                    "momentum_local": [1.0, 0.0, 0.0, 0.0],
                    "u_mu": [1.0, 0.0, 0.0, 0.0],
                    "SM_charges": {"EM": 0.0, "color": [0.0, 0.0, 0.0], "weak": 0.0},
                },
                {
                    "label": "recipient",
                    "rho": 0.1,
                    "occupation": 1.0,
                    "invariant_weight": 1.0,
                    "mass": 1.0,
                    "momentum_local": [1.0, 0.0, 0.0, 0.0],
                    "u_mu": [1.0, 0.0, 0.0, 0.0],
                },
            ]
        },
        "reaction_channels": {"channels": [{"channel_id": "A_to_R", "transported_delta_p": [0.0, 0.0, 0.0, 0.0]}]},
        "moments": {
            "stress_moment_residual": 0.0,
            "finite_divergence_operator_residual": 0.0,
            "stress_volume_weight_residual": 0.0,
            "variational_moment_agreement_residual": 0.0,
        },
        "source_localization": {
            "source_route": "NONLINEAR_CMI_STRESS",
            "entropy_base": "nats",
            "hbar": 1.0,
            "c": 1.0,
            "length_unit": "code_length",
            "proper_diamond_radius_ell": 1.0,
            "uv_scale": 0.01,
            "collar_width": 0.1,
            "curvature_scale_bound": 100.0,
            "gradient_scale_bound": 100.0,
            "cmi_kind": "QUANTUM_EXACT",
            "S_AB": 1.4,
            "S_BD": 1.3,
            "S_B": 1.0,
            "S_ABD": 1.5,
            "CMI": 0.2,
            "DeltaK_expectation": 0.2,
            "first_variation_classification": "MARKOV_BRANCH_QUADRATIC",
            "matching_theorem_hash": _hash("3"),
            "matching_residual": 0.0,
            "modular_source_charge_nats": 0.2,
            "source_localization_residual_nats": 0.0,
            "bw_ball_normalization_residual": 0.0,
            "ell4_scaling_plateau_residual": 0.0,
            "cover_independence_residual": 0.0,
            "timelike_probe_rank": 10,
            "heldout_quadraticity_residual": 0.0,
        },
        "repair": {
            "Gamma_rec": 0.05,
            "detailed_balance_residual": 0.0,
            "PHYSICAL_CLOCK_RECEIPT": True,
            "ACTIVE_FIBER_RECEIPT": True,
            "CONSERVED_SECTOR_DECOMPOSITION_RECEIPT": True,
        },
        "stress": {
            "total_stress_divergence_residual": 0.0,
            "exchange_current_residual": 0.0,
            "recipient_stress_residual": 0.0,
            "recipient_exchange_residual": 0.0,
            "local_frame_covariance_residual": 0.0,
            "carrier_quotient_invariance_residual": 0.0,
        },
        "causal_response": {
            "characteristic_speed_bound": 0.5,
            "kinetic_matrix": [[1.0]],
            "damping_matrix": [[1.0]],
            "propagation_matrix": [[0.5]],
            "source_matrix": [[1.0]],
            "output_matrix": [[1.0]],
            "response_stability_residual": 0.0,
            "retarded_support_residual": 0.0,
            "finite_domain_residual": 0.0,
            "COMMON_PARENT_RESPONSE_POLE_RECEIPT": True,
        },
        "gauge": {
            "gauge_consistency_residual": 0.0,
            "gauge_invariant_variable_residual": 0.0,
            "independent_gauge_presentation_count": 2,
        },
        "refinement": {"convergence_residual": 0.0, "regulator_level_count": 3},
        "cdm_limit": {"cdm_limit_residual": 0.0, "cdm_operator_residual": 0.0},
        "frozen_run": {
            "source_hash": _hash("0"),
            "solver_hash": _hash("1"),
            "likelihood_hash": _hash("2"),
            "mutable_source_artifacts": False,
        },
    }


def _write_clean_source_provenance(run: Path) -> None:
    sources = {
        "eta_R": "finite_repair_transition_clock",
        "Gamma_rec": "finite_repair_transition_clock",
        "A_zeta": "finite_lattice",
        "q_IR": "scale_compressed_24_round_finite_ladder",
        "ell_IR": "scale_compressed_24_round_finite_ladder",
        "B_A_k_a": "parent_collar_finite_difference",
        "rho_A_a": "finite_lattice",
        "N_CRC": "OPH_direct_public_record_capacity",
    }
    _write_json(
        run / "cmb_source_provenance_report.json",
        {
            "nodes": [
                {
                    "node_id": quantity,
                    "quantity": quantity,
                    "source": source,
                    "source_kind": source,
                    "source_only": True,
                    "no_cmb_data_used": True,
                    "parents": [],
                }
                for quantity, source in sources.items()
            ],
            "reducers": {
                **{
                    quantity: {
                        "mode": "pooled_sufficient_statistics",
                        "pooled_sufficient_statistics": True,
                        "units_validated": True,
                        "coordinate_grid_validated": True,
                        "coverage_validated": True,
                        "duplicates_checked": True,
                        "interpolation_policy_frozen": True,
                        "covariance_validated": True,
                        "shard_local_nonlinear_average": False,
                    }
                    for quantity in sources
                    if quantity != "N_CRC"
                },
                "N_CRC": {
                    "mode": "direct_public_record_capacity",
                    "exact_public_record_capacity_evaluator": True,
                    "complete_terminal_fiber_receipt": True,
                    "whole_fiber_scalarization_receipt": True,
                    "target_free_capacity_producer_receipt": True,
                    "robust_closure_receipt": True,
                    "unique_regulator_stable_slack_zero_receipt": True,
                    "horizon_record_saturation_receipt": True,
                    "physical_N_closure_receipt": True,
                },
            },
            "global_checks": {
                "official_likelihood_rollup": "global",
                "cdm_limit_rollup": "global",
                "HERMETIC_READ_SET_RECEIPT": True,
                "SOURCE_MODEL_FREEZE_RECEIPT": True,
            },
        },
    )
