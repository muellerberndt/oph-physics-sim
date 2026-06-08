from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.cosmology.comparable_data import comparable_data_report, write_comparable_data_package


def test_comparable_data_report_collects_h3_cmb_and_holonomy(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "manifest.json",
        {
            "run_id": "r1",
            "name": "demo",
            "patch_count": 4096,
        },
    )
    _write_json(
        run / "modular_response_h3_report.json",
        {
            "h3_bulk_candidate_receipt": True,
            "h3_fit": {
                "heldout_normalized_rmse": 0.9,
                "heldout_explained_variance": 0.1,
            },
            "h3_chart_dimension_debug": {
                "point_count": 16,
                "candidate_3d_dimension_window": True,
                "dimension_estimators_agree": True,
                "correlation_dimension": {"estimate": 2.95},
                "local_mle_dimension": {"estimate": 3.05},
            },
            "s2_boundary_control": {"heldout_normalized_rmse": 1.1},
            "control_fits": {
                "shuffled_response": {"heldout_normalized_rmse": 1.0},
                "shuffled_observer_labels": {"heldout_normalized_rmse": 1.2},
                "no_perturbation": {"heldout_normalized_rmse": 1.0},
            },
            "wrong_scale_control_fits": {
                "1x": {"heldout_normalized_rmse": 1.05},
                "pi": {"heldout_normalized_rmse": 1.2},
                "4pi": {"heldout_normalized_rmse": 1.01},
            },
            "wrong_scale_feature_audit": {
                "eligible": True,
                "audited_feature_count": 10,
                "wrong_scale_win_count": 2,
                "wrong_scale_win_fraction": 0.2,
                "two_pi_h3_fit_win_fraction": 0.8,
                "red_flag_wrong_scale_wins": True,
                "winner_counts": {"2pi_h3_fit": 8, "pi": 2},
                "worst_groups": [
                    {
                        "cap_index": 1,
                        "time_index": 0,
                        "time": 0.1,
                        "observable": "checkpoint_class",
                        "feature_type": "target_distribution_delta",
                        "feature_count": 4,
                        "wrong_scale_win_fraction": 0.5,
                    }
                ],
            },
        },
    )
    _write_json(
        run / "cmb_lite_comparison_report.json",
        {
            "benchmark": {"label": "PlanckLite"},
            "best_shape_field": "record_signature",
            "field_comparisons": {
                "record_signature": {
                    "shape_correlation": 0.3,
                    "normalized_rmse": 0.95,
                    "peak_fraction_delta": 0.1,
                }
            },
            "physical_cmb_prediction": False,
        },
    )
    _write_json(
        run / "cl_comparison_report.json",
        {
            "freezeout_cycle": 8,
            "committed_fraction": 0.99,
            "fields": {
                "record_signature": {
                    "peak_ell": 9,
                    "total_abs_D_ell_2_plus": 1.5,
                    "control_comparison": {"min_relative_l2_delta": 0.8},
                }
            },
        },
    )
    _write_json(
        run / "cmb_transfer_report.json",
        {
            "diagnostic_transfer_receipt": True,
            "train_patch_count": 4096,
            "test_patch_count": 16384,
            "train": {"mean_shape_correlation": 0.8, "mean_normalized_rmse": 0.6},
            "test": {"mean_shape_correlation": 0.7, "mean_normalized_rmse": 0.65},
            "max_control_test_shape_correlation": 0.2,
            "physical_cmb_prediction": False,
        },
    )
    _write_json(
        run / "camb_lcdm_baseline_report.json",
        {
            "CDM_LIMIT_BOLTZMANN_RECEIPT": True,
            "oph_anomaly_module_ready": False,
            "physical_cmb_prediction": False,
            "comparison": {
                "shape_correlation": 0.99,
                "normalized_rmse": 0.1,
                "amplitude_fit_chi2_per_bin": 2.0,
                "best_fit_column_chi2_per_bin": 1.0,
                "first_peak_ell": 225.0,
                "benchmark_first_peak_ell": 225.0,
                "mean_absolute_fractional_error": 0.03,
            },
        },
    )
    _write_json(
        run / "oph_boltzmann_input_report.json",
        {
            "source_report_count": 2,
            "readiness": {
                "cdm_limit_solver_ready": True,
                "diagnostic_repair_exchange_table_ready": True,
                "physical_prediction_ready": False,
                "missing_gates": ["B_A_k_a_physical_emitted", "full_likelihood_ready"],
            },
            "cdm_limit": {"row_count": 4},
            "diagnostic_repair_exchange": {
                "row_count": 8,
                "rows": [
                    {"Gamma_rec_over_H_shape_proxy": 0.5, "B_A_shape_proxy": -1.0},
                    {"Gamma_rec_over_H_shape_proxy": 0.7, "B_A_shape_proxy": 1.0},
                ],
            },
        },
    )
    _write_json(
        run / "oph_cmb_stress_report.json",
        {
            "physical_prediction_readiness": {
                "boltzmann_ready": False,
                "physical_cmb_prediction_ready": False,
                "missing_gates": ["rho_A_a_emitted", "B_A_k_a_emitted"],
            },
            "finite_collar_parent": {
                "sample_count": 4,
                "weighted_collar_repair_defect_R": 1.25,
                "rho_A_eq_proxy": None,
            },
            "diagnostic_kernel_proxy": {
                "kernel_proxy_rows": [{"k_proxy_inverse_theta": 1.0}],
            },
            "COSMOLOGY_PERTURBATION_RECEIPT": False,
            "physical_cmb_prediction": False,
        },
    )
    _write_json(
        run / "galaxy_proxy_report.json",
        {
            "GALAXY_PROXY_RECEIPT": True,
            "a0_oph": 1.2e-10,
            "a0_eff": 6.0e-11,
            "lambda_collar_declared": 1.41421356237,
            "rar_curve": [{"g_baryon": 1.0e-12, "g_observed_proxy": 1.0e-11}],
            "lambda_collar_estimate": {"usable": True, "lambda_collar": 1.4},
            "btfr": {"usable": True, "slope_logM_vs_logV": 4.0},
            "disk_potential_residual": {"usable": False},
            "physical_claim": False,
        },
    )
    _write_json(
        run / "array_holonomy_report.json",
        {
            "triangle_count": 100,
            "defect_fraction": 0.7,
            "cluster_count": 4,
        },
    )
    _write_json(
        run / "defect_timeline_report.json",
        {
            "snapshot_count": 10,
            "worldline_count": 25,
            "persistent_worldline_count": 24,
            "max_observation_count": 10,
            "max_lifetime_cycles": 63,
            "persistent_worldline_precursor_receipt": True,
        },
    )
    _write_json(
        run / "defect_interaction_report.json",
        {
            "screen_transport_proxy_count": 21,
            "fusion_candidate_count": 77,
            "fusion_conservation_proxy_pass": True,
            "scattering_reproducibility_proxy_pass": True,
            "interaction_proxy_receipt": True,
        },
    )
    _write_json(
        run / "particle_likeness_report.json",
        {
            "worldline_count": 25,
            "localized_count": 25,
            "persistent_count": 24,
            "sector_stable_count": 25,
            "transportable_count": 21,
            "particle_like_count": 0,
            "particle_matter_receipt": False,
        },
    )
    _write_json(
        run / "defect_h3_worldlines_report.json",
        {
            "worldline_count": 25,
            "persistent_h3_worldline_count": 20,
            "bulk_worldline_precursor_receipt": False,
        },
    )
    _write_json(
        run / "observer_chart_object_h3_report.json",
        {
            "object_count": 11,
            "localized_object_count": 5,
            "localized_not_boundary_object_count": 4,
            "shuffle_control_count": 3,
            "shuffled_localized_object_count": 2,
            "shuffled_localized_object_p90": 6.0,
            "median_h3_compactness_normalized": 0.2,
            "median_s2_boundary_compactness_normalized": 0.1,
            "median_shuffled_h3_compactness_normalized": 0.5,
            "p10_shuffled_h3_compactness_normalized": 0.4,
            "p90_shuffled_h3_compactness_normalized": 0.7,
            "h3_beats_shuffled_incidence": True,
            "h3_not_boundary_dominated": False,
            "observer_chart_object_h3_receipt": True,
            "observer_chart_bulk_population_receipt": False,
        },
    )
    _write_json(
        run / "bulk_reconstruction_report.json",
        {
            "control_gate_passed": True,
            "candidate_3d_dimension_window": False,
            "bulk_3d_established": False,
            "dimension_estimators_agree": False,
            "neutral_dimension_report": {
                "radial_depth_used": False,
                "primary_dimension": {"estimate": None},
                "correlation_dimension": {"estimate": 2.1},
                "local_mle_dimension": {"estimate": 2.4},
            },
            "blind_observer_bulk_report": {
                "usable": True,
                "feature_width": 9,
                "s2_leakage_audit_pass": True,
                "s2_distance_correlation": 0.12,
                "candidate_3d_dimension_window": False,
                "bulk_3d_established": False,
                "neutral_dimension_report": {
                    "correlation_dimension": {"estimate": 2.2},
                    "local_mle_dimension": {"estimate": 2.3},
                },
            },
        },
    )
    _write_json(
        run / "bw_state_derived_report.json",
        {
            "BW_KMS_BRANCH_INSTANTIATION_RECEIPT": False,
            "state_mode": "cap_flow_graph_generator",
            "endogenous_modular_generator": False,
            "declared_cap_flow_generator": True,
            "normalization_declared": False,
            "generator_scale": 6.283185307179586,
            "state_selected_scale_label": "1x",
            "state_selected_2pi": False,
            "correct_beats_controls": False,
            "median": 0.75,
            "best_control": "wrong_1x_normalization",
            "best_control_median": 0.43,
            "controls": {
                "wrong_1x_normalization": {"median": 0.43},
                "wrong_pi_normalization": {"median": 0.43},
                "wrong_4pi_normalization": {"median": 1.29},
                "no_modular_flow": {"median": 0.67},
            },
        },
    )

    report = comparable_data_report([tmp_path])

    assert report["run_count"] == 1
    assert report["physical_cmb_prediction"] is False
    assert report["bulk_3d_established"] is False
    oph_cmb = report["measurement_lanes"]["oph_cmb_anomaly_stress_adapter"]
    assert oph_cmb["run_count"] == 1
    assert oph_cmb["boltzmann_ready_count"] == 0
    assert oph_cmb["mean_weighted_collar_repair_defect_R"] == 1.25
    assert report["measurement_lanes"]["h3_modular_response_controls"]["receipt_count"] == 1
    assert report["measurement_lanes"]["h3_modular_response_controls"]["h3_chart_3d_window_count"] == 1
    assert report["measurement_lanes"]["h3_modular_response_controls"]["h3_chart_dimension_agree_count"] == 1
    assert report["measurement_lanes"]["h3_modular_response_controls"]["mean_h3_chart_correlation_dimension"] == 2.95
    assert report["measurement_lanes"]["h3_modular_response_controls"]["mean_h3_chart_local_mle_dimension"] == 3.05
    assert report["measurement_lanes"]["h3_modular_response_controls"]["wrong_scale_feature_audit_count"] == 1
    assert report["measurement_lanes"]["h3_modular_response_controls"]["mean_wrong_scale_feature_win_fraction"] == 0.2
    assert report["measurement_lanes"]["h3_modular_response_controls"]["wrong_scale_red_flag_count"] == 1
    state_bw = report["measurement_lanes"]["state_derived_bw_matrix_elements"]
    assert state_bw["run_count"] == 1
    assert state_bw["endogenous_run_count"] == 0
    assert state_bw["declared_cap_flow_run_count"] == 1
    assert state_bw["direct_transition_automorphism_run_count"] == 0
    assert state_bw["selected_2pi_count"] == 0
    assert state_bw["mean_declared_cap_flow_median_residual"] == 0.75
    assert report["measurement_lanes"]["observer_chart_object_population"]["object_chart_receipt_count"] == 1
    assert report["measurement_lanes"]["observer_chart_object_population"]["bulk_population_receipt_count"] == 0
    assert report["measurement_lanes"]["observer_chart_object_population"]["mean_shuffle_control_count"] == 3.0
    assert report["measurement_lanes"]["observer_chart_object_population"]["mean_shuffled_localized_object_p90"] == 6.0
    assert report["measurement_lanes"]["observer_chart_object_population"]["mean_p90_shuffled_h3_compactness"] == 0.7
    neutral = report["measurement_lanes"]["neutral_observer_reconstruction"]
    assert neutral["run_count"] == 1
    assert neutral["blind_usable_count"] == 1
    assert neutral["blind_s2_leakage_pass_count"] == 1
    assert neutral["blind_candidate_3d_window_count"] == 0
    assert neutral["mean_blind_feature_width"] == 9.0
    assert neutral["mean_blind_s2_distance_correlation"] == 0.12
    assert report["measurement_lanes"]["planck_tt_shape_lite"]["mean_record_signature_shape_correlation"] == 0.3
    assert report["measurement_lanes"]["cmb_screen_basis_transfer"]["diagnostic_receipt_count"] == 1
    camb = report["measurement_lanes"]["camb_lcdm_baseline"]
    assert camb["run_count"] == 1
    assert camb["cdm_limit_boltzmann_receipt_count"] == 1
    assert camb["mean_shape_correlation"] == 0.99
    assert camb["mean_amplitude_fit_chi2_per_bin"] == 2.0
    boltzmann = report["measurement_lanes"]["oph_boltzmann_input_readouts"]
    assert boltzmann["run_count"] == 1
    assert boltzmann["cdm_limit_ready_count"] == 1
    assert boltzmann["diagnostic_repair_table_ready_count"] == 1
    assert boltzmann["physical_prediction_ready_count"] == 0
    assert boltzmann["mean_source_report_count"] == 2.0
    assert boltzmann["mean_gamma_rec_over_H_shape_proxy"] == 0.6
    galaxy = report["measurement_lanes"]["static_galaxy_proxy"]
    assert galaxy["proxy_receipt_count"] == 1
    assert galaxy["runs_with_external_rar_or_btfr_data"] == 1
    assert galaxy["mean_fitted_lambda_collar"] == 1.4
    assert galaxy["mean_btfr_slope"] == 4.0
    assert report["measurement_lanes"]["cmb_screen_basis_transfer"]["mean_test_shape_correlation"] == 0.7
    assert report["measurement_lanes"]["screen_holonomy_defect_proxy"]["mean_defect_fraction"] == 0.7
    defect_worldlines = report["measurement_lanes"]["defect_worldline_particle_precursors"]
    assert defect_worldlines["run_count"] == 1
    assert defect_worldlines["worldline_precursor_receipt_count"] == 1
    assert defect_worldlines["interaction_proxy_receipt_count"] == 1
    assert defect_worldlines["particle_matter_receipt_count"] == 0
    assert defect_worldlines["h3_bulk_worldline_precursor_count"] == 0
    assert defect_worldlines["mean_timeline_worldline_count"] == 25.0
    assert defect_worldlines["mean_screen_transport_proxy_count"] == 21.0
    assert defect_worldlines["mean_fusion_candidate_count"] == 77.0


def test_write_comparable_data_package_writes_json_csv_and_markdown(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(run / "array_holonomy_report.json", {"triangle_count": 10, "defect_fraction": 0.5})

    out = tmp_path / "out"
    report = write_comparable_data_package([tmp_path], out)

    assert report["run_count"] == 1
    assert (out / "comparable_data_snapshot.json").exists()
    assert (out / "comparable_data_rows.csv").exists()
    assert (out / "comparable_data_snapshot.md").exists()
    assert "not a physical CMB prediction" in (out / "comparable_data_snapshot.md").read_text(encoding="utf-8")


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")
