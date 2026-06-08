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
                "material_wrong_scale_win_count": 1,
                "material_wrong_scale_win_fraction": 0.1,
                "two_pi_h3_fit_win_fraction": 0.8,
                "material_two_pi_h3_fit_win_fraction": 0.9,
                "red_flag_wrong_scale_wins": True,
                "material_red_flag_wrong_scale_wins": True,
                "winner_counts": {"2pi_h3_fit": 8, "pi": 2},
                "material_winner_counts": {"2pi_h3_fit": 9, "pi": 1},
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
        run / "h3_refit_ensemble_report.json",
        {
            "mode": "h3_refit_seed_ensemble",
            "seed_count": 8,
            "receipt_count": 4,
            "receipt_fraction": 0.5,
            "control_separation_receipt_count": 8,
            "control_separation_receipt_fraction": 1.0,
            "candidate_receipt_count": 4,
            "candidate_receipt_fraction": 0.5,
            "candidate_3d_window_count": 0,
            "candidate_3d_window_fraction": 0.0,
            "H3_RESPONSE_ENSEMBLE_RECEIPT": True,
            "h3_response_seed_robust_receipt": True,
            "h3_chart_3d_seed_robust_receipt": False,
            "mean_heldout_normalized_rmse": 0.93,
            "mean_heldout_explained_variance": 0.13,
            "p75_material_wrong_scale_win_fraction": 0.05859375,
        },
    )
    _write_json(
        run / "caps_to_h3_minimal_report.json",
        {
            "mode": "e0_caps_to_h3_minimal",
            "S2_CAP_PROFILE_TO_H3_RECEIPT": True,
            "median_reconstruction_mse": 1.0e-30,
            "median_shuffled_profile_mse": 0.14,
            "median_s2_boundary_profile_mse": 0.34,
            "h3_beats_shuffled": True,
            "h3_beats_s2_boundary": True,
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
                    "overlap_ell_physical_comparison": {
                        "usable": True,
                        "shape_correlation": 0.42,
                        "positive_amp_normalized_rmse": 0.91,
                        "overlap_ell_min": 50.0,
                        "overlap_ell_max": 128.0,
                        "overlap_benchmark_count": 4,
                    },
                }
            },
            "physical_cmb_prediction": False,
        },
    )
    _write_json(
        run / "paper_3d_bulk_chart_report.json",
        {
            "PAPER_THEOREM_3D_BULK_CHART_RECEIPT": True,
            "paper_theorem_3d_bulk_chart_receipt": True,
            "h3_spatial_dimension_from_boost_orbit": 3,
            "finite_point_cloud_dimension_estimator_used": False,
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
        run / "static_galaxy_measurement_report.json",
        {
            "STATIC_GALAXY_RAR_BTFR_RECEIPT": True,
            "OPH_STATIC_GALAXY_BRIDGE_RECEIPT": True,
            "bridge": "static_galaxy",
            "claim_tier": "Tier1_phenomenological_continuation",
            "bulk_required": False,
            "physical_cmb_claim": False,
            "physical_matter_power_claim": False,
            "physical_claim": True,
            "dataset_row_count": 180,
            "dataset_galaxy_count": 175,
            "galaxy_count": 175,
            "measurement_galaxy_count": 175,
            "rar_galaxy_count": 1,
            "rar_point_count": 153,
            "shared_a0": 1.16e-10,
            "shared_lambda_collar": 1.0,
            "rar_scatter_dex": 0.13,
            "btfr": {
                "usable": True,
                "galaxy_count": 123,
                "slope_logM_vs_logV": 3.5,
                "intercept_logM_vs_logV": 2.7,
            },
            "btfr_prediction_from_rar_fit": {
                "predicted_slope_logM_vs_logV": 4.0,
                "predicted_intercept_logM_vs_logV": 1.8,
                "slope_delta_observed_minus_predicted": -0.5,
                "intercept_delta_observed_minus_predicted": 0.9,
                "abs_slope_delta": 0.5,
                "abs_intercept_delta_dex": 0.9,
            },
            "holdout_validation": {
                "usable": True,
                "receipt": True,
                "train_galaxy_count": 120,
                "test_galaxy_count": 55,
                "train_point_count": 2300,
                "test_point_count": 1091,
                "shared_a0": 1.17e-10,
                "shared_lambda_collar": 1.02,
                "train": {"log_acceleration_rmse_dex": 0.14},
                "test": {
                    "log_acceleration_rmse_dex": 0.16,
                    "velocity_rmse_km_s": 18.0,
                    "baryon_only_velocity_rmse_km_s": 31.0,
                    "velocity_rmse_improvement_fraction": 0.42,
                    "velocity_chi2_proxy_per_point": 9.5,
                },
            },
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
        run / "record_family_h3_report.json",
        {
            "support_count": 123,
            "cap_count": 48,
            "record_populated_h3_receipt": True,
            "record_family_h3_bulk_population_candidate": True,
            "h3_fit": {"median_residual": 0.12},
            "s2_boundary_control": {"median_residual": 0.31},
            "shuffled_cap_response_control": {"median_residual": 0.2},
        },
    )
    _write_json(
        run / "defect_cluster_h3_report.json",
        {
            "support_count": 12,
            "cap_count": 48,
            "record_populated_h3_receipt": True,
            "record_family_h3_bulk_population_candidate": True,
            "h3_fit": {"median_residual": 0.03},
            "s2_boundary_control": {"median_residual": 0.08},
            "shuffled_cap_response_control": {"median_residual": 0.25},
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
            "target_scale_control_degenerate": True,
            "degenerate_target_scale_controls": ["wrong_1x_normalization", "wrong_pi_normalization"],
            "median": 0.75,
            "best_control": "wrong_1x_normalization",
            "best_control_median": 0.43,
            "generator_scale_audit": {
                "enabled": True,
                "best_label": "minus_2pi",
                "best_scale": -6.283185307179586,
                "best_score": 0.52,
                "configured_score": 0.88,
                "configured_is_best": False,
                "two_pi_generator_is_best": False,
                "diagnosis": "opposite_sign_best",
            },
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
    assert report["measurement_lanes"]["h3_modular_response_controls"]["mean_material_wrong_scale_feature_win_fraction"] == 0.1
    assert report["measurement_lanes"]["h3_modular_response_controls"]["wrong_scale_red_flag_count"] == 1
    assert report["measurement_lanes"]["h3_modular_response_controls"]["material_wrong_scale_red_flag_count"] == 1
    h3_ensemble = report["measurement_lanes"]["h3_seed_ensemble_robustness"]
    assert h3_ensemble["run_count"] == 1
    assert h3_ensemble["response_seed_robust_count"] == 1
    assert h3_ensemble["chart_3d_seed_robust_count"] == 0
    assert h3_ensemble["mean_seed_count"] == 8.0
    assert h3_ensemble["mean_receipt_fraction"] == 0.5
    assert h3_ensemble["mean_control_separation_fraction"] == 1.0
    assert h3_ensemble["mean_candidate_receipt_fraction"] == 0.5
    assert h3_ensemble["mean_p75_material_wrong"] == 0.05859375
    minimal_caps = report["measurement_lanes"]["minimal_caps_to_h3"]
    assert minimal_caps["run_count"] == 1
    assert minimal_caps["receipt_count"] == 1
    assert minimal_caps["mean_median_reconstruction_mse"] == 1.0e-30
    assert minimal_caps["h3_beats_shuffled_count"] == 1
    assert minimal_caps["h3_beats_s2_boundary_count"] == 1
    state_bw = report["measurement_lanes"]["state_derived_bw_matrix_elements"]
    assert state_bw["run_count"] == 1
    assert state_bw["endogenous_run_count"] == 0
    assert state_bw["declared_cap_flow_run_count"] == 1
    assert state_bw["direct_transition_automorphism_run_count"] == 0
    assert state_bw["selected_2pi_count"] == 0
    assert state_bw["mean_declared_cap_flow_median_residual"] == 0.75
    assert state_bw["generator_scale_audit_count"] == 1
    assert state_bw["target_scale_control_degenerate_count"] == 1
    assert state_bw["degenerate_target_scale_control_counts"] == {
        "wrong_1x_normalization": 1,
        "wrong_pi_normalization": 1,
    }
    assert state_bw["generator_scale_audit_configured_best_count"] == 0
    assert state_bw["generator_scale_audit_two_pi_best_count"] == 0
    assert state_bw["generator_scale_audit_best_label_counts"] == {"minus_2pi": 1}
    assert state_bw["generator_scale_audit_diagnosis_counts"] == {"opposite_sign_best": 1}
    assert state_bw["mean_generator_scale_audit_best_score"] == 0.52
    assert state_bw["mean_generator_scale_audit_configured_score"] == 0.88
    lorentz = report["measurement_lanes"]["support_visible_lorentz_branch"]
    assert lorentz["support_visible_record_h3_population_count"] == 1
    assert lorentz["support_visible_defect_h3_population_count"] == 1
    assert lorentz["support_visible_h3_populated_bulk_count"] == 1
    assert lorentz["mean_record_family_h3_median_residual"] == 0.12
    assert lorentz["mean_defect_cluster_h3_median_residual"] == 0.03
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
    assert report["measurement_lanes"]["planck_tt_shape_lite"]["overlap_ell_usable_count"] == 1
    assert (
        report["measurement_lanes"]["planck_tt_shape_lite"][
            "mean_record_signature_overlap_ell_shape_correlation"
        ]
        == 0.42
    )
    assert (
        report["measurement_lanes"]["planck_tt_shape_lite"][
            "mean_record_signature_overlap_ell_normalized_rmse"
        ]
        == 0.91
    )
    assert report["measurement_lanes"]["planck_tt_shape_lite"]["mean_record_signature_overlap_ell_max"] == 128.0
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
    static_galaxy = report["measurement_lanes"]["static_galaxy_measurement_fit"]
    assert static_galaxy["bridge_receipt_count"] == 1
    assert static_galaxy["claim_tier_counts"] == {"Tier1_phenomenological_continuation": 1}
    assert static_galaxy["physical_cmb_claim_count"] == 0
    assert static_galaxy["bulk_required_count"] == 0
    assert static_galaxy["mean_measurement_galaxy_count"] == 175.0
    assert static_galaxy["mean_rar_galaxy_count"] == 1.0
    assert static_galaxy["mean_btfr_predicted_slope"] == 4.0
    assert static_galaxy["mean_btfr_slope_delta"] == -0.5
    assert static_galaxy["mean_btfr_intercept_delta"] == 0.9
    assert static_galaxy["holdout_receipt_count"] == 1
    assert static_galaxy["mean_holdout_test_galaxy_count"] == 55.0
    assert static_galaxy["mean_holdout_test_point_count"] == 1091.0
    assert static_galaxy["mean_holdout_test_log_accel_rmse"] == 0.16
    assert static_galaxy["mean_holdout_test_velocity_rmse"] == 18.0
    assert static_galaxy["mean_holdout_test_velocity_improvement"] == 0.42
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


def test_direct_transition_automorphism_ignores_archived_generator_scale_audit(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "bw_state_derived_report.json",
        {
            "BW_KMS_BRANCH_INSTANTIATION_RECEIPT": True,
            "state_mode": "transition_response_unitary",
            "direct_transition_automorphism": True,
            "endogenous_modular_generator": False,
            "declared_cap_flow_generator": False,
            "normalization_declared": True,
            "generator_scale": 6.283185307179586,
            "state_selected_scale_label": "2pi",
            "state_selected_2pi": True,
            "correct_beats_controls": True,
            "median": 1e-15,
            "best_control": "wrong_pi_normalization",
            "best_control_median": 0.7,
            "generator_scale_audit": {
                "enabled": True,
                "best_label": "minus_2pi",
                "best_scale": -6.283185307179586,
                "best_score": 1e-15,
                "configured_score": 2e-15,
                "configured_is_best": False,
                "two_pi_generator_is_best": False,
                "diagnosis": "opposite_sign_best",
            },
            "controls": {
                "wrong_1x_normalization": {"median": 1.2},
                "wrong_pi_normalization": {"median": 0.7},
                "no_modular_flow": {"median": 1.1},
            },
        },
    )

    report = comparable_data_report([tmp_path])
    state_bw = report["measurement_lanes"]["state_derived_bw_matrix_elements"]

    assert state_bw["run_count"] == 1
    assert state_bw["direct_transition_automorphism_run_count"] == 1
    assert state_bw["receipt_count"] == 1
    assert state_bw["generator_scale_audit_count"] == 0
    assert state_bw["generator_scale_audit_best_label_counts"] == {}


def test_comparable_data_collects_oph_inflation_cmb_camb_transfer(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "oph_inflation_cmb_camb_report.json",
        {
            "mode": "oph_inflation_cmb_camb_transfer_v0",
            "oph_input": {
                "n_s": 0.9660214956374176,
                "A_zeta": 2.502261321821402e-09,
                "q_IR": 0.2445545067865991,
                "ell_IR": 33.614958176528,
            },
            "comparison": {
                "camb_lcdm_powerlaw": {
                    "shape_correlation": 0.995,
                    "amplitude_fit_chi2_per_bin": 1.4,
                },
                "oph_p48_powerlaw": {
                    "shape_correlation": 0.996,
                    "amplitude_fit_chi2_per_bin": 1.35,
                },
                "oph_p48_ir_v04": {
                    "shape_correlation": 0.996,
                    "amplitude_fit_chi2_per_bin": 1.36,
                },
            },
            "acoustic_preservation": {
                "mean_abs_fractional_delta_ell_ge_50": 0.001,
            },
            "low_ell_v04_diagnostic": {
                "CAMB_LCDM_chi2_ell2_29": 27.40723660095901,
                "CAMB_OPH_IR_chi2_ell2_29": 16.65364090607876,
            },
            "measurement_comparable_cmb_curve": True,
            "physical_cmb_prediction": False,
            "screen_camb_transfer_receipt": True,
        },
    )

    report = comparable_data_report([tmp_path])
    lane = report["measurement_lanes"]["oph_inflation_cmb_camb_transfer"]

    assert lane["run_count"] == 1
    assert lane["measurement_comparable_curve_count"] == 1
    assert lane["transfer_receipt_count"] == 1
    assert lane["physical_cmb_prediction_count"] == 0
    assert lane["mean_n_s"] == 0.9660214956374176
    assert lane["mean_p48_ir_chi2_per_bin"] == 1.36
    assert lane["mean_lowell_oph_ir_chi2"] == 16.65364090607876


def test_comparable_data_collects_inflation_cmb_v05_hard_gates(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "oph_inflation_cmb_bridge_report.json",
        {
            "screen_spectrum_prediction": {
                "n_s": 0.966,
                "theta_OPH": 0.034,
                "A_zeta": 2.5e-9,
                "n_s_pull_vs_planck": 0.25,
            },
            "flat_sector_selection": {
                "selected_Omega_K": 0.0,
                "Omega_A0_residual": 0.264,
                "rho_A_over_rho_b": 5.36,
            },
            "cmb_success_ladder": {
                "diagnostic_cmb_data_available": True,
                "core_numbers": {
                    "v0_2_IR_bestfit_q_IR": 0.244,
                    "v0_2_IR_bestfit_ell_IR": 33.6,
                    "v0_3_camb_lowell_LCDM_chi2_ell2_29": 27.4,
                    "v0_3_camb_lowell_IR_bestfit_chi2_ell2_29": 16.65,
                    "v0_4_LCDM_PTE_R_OE_upper": 0.0107,
                    "v0_4_parity_PTE_R_OE_upper": 0.4069,
                },
                "hard_gates_v0_5": {
                    "TT_lowell_delta_chi2": 10.75,
                    "TE_lowell_delta_chi2": 0.435,
                    "EE_lowell_delta_chi2": -1.826,
                    "TT_high_ell_delta_chi2_30_1200": -0.388,
                    "combined_TT_TE_EE_lowell_delta_chi2": 9.363,
                    "pressure_points": [{"gate": "D"}, {"gate": "F"}],
                },
            },
            "physical_cmb_prediction": False,
        },
    )

    report = comparable_data_report([tmp_path])
    lane = report["measurement_lanes"]["oph_inflation_cmb_bridge"]

    assert lane["run_count"] == 1
    assert lane["mean_v05_TT_lowell_delta_chi2"] == 10.75
    assert lane["mean_v05_TE_lowell_delta_chi2"] == 0.435
    assert lane["mean_v05_EE_lowell_delta_chi2"] == -1.826
    assert lane["mean_v05_combined_lowell_delta_chi2"] == 9.363
    assert lane["mean_v05_pressure_point_count"] == 2


def test_comparable_data_collects_gap_release_adiabaticity_and_h0s8(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "sync_gap_report.json",
        {
            "mode": "oph_low_k_synchronization_gap_audit_v0",
            "run_count": 4,
            "aggregate": {
                "time_resolved_trace_count": 0,
                "cached_proxy_pass_count": 4,
                "time_resolved_gap_pass_count": 0,
                "mean_global_phi_gamma_per_cycle": 0.61,
                "median_global_phi_gamma_per_cycle": 0.62,
            },
            "low_k_gap_established": False,
            "same_boundary_selector_established": False,
            "inflation_replacement_ready": False,
        },
    )
    _write_json(
        run / "hot_release_report.json",
        {
            "mode": "oph_hot_maxent_release_audit_v0",
            "run_count": 4,
            "aggregate": {
                "mechanical_release_surface_count": 4,
                "collar_markov_gate_pass_count": 0,
                "hot_release_theorem_ready_count": 0,
                "median_release_cycle": 10,
                "mean_release_cycle": 10.5,
                "mean_median_epsilon_cmi": 0.38,
            },
            "hot_release_theorem_ready": False,
        },
    )
    _write_json(
        run / "adiabaticity_report.json",
        {
            "mode": "oph_same_boundary_adiabaticity_audit_v0",
            "run_count": 4,
            "aggregate": {
                "adiabaticity_proxy_pass_count": 0,
                "mean_max_entropy_residual_std": 1.4,
                "mean_min_common_clock_corr": -0.03,
            },
            "adiabaticity_established": False,
        },
    )
    _write_json(
        run / "h0s8_branch_report.json",
        {
            "mode": "oph_h0_s8_branch_diagnostic_v0",
            "flat_q_a_closure": {
                "H0_km_s_Mpc": 67.40002854274209,
                "Omega_m": 0.3154851573406727,
                "Omega_A": 0.26411417760478384,
                "Omega_Lambda_OPH": 0.684392720043898,
                "flat_sum": 1.0,
            },
            "collar_tracking": {
                "lambda_collar": 0.9343006394893864,
                "f_A": 0.8371683150836268,
                "mu_eff_source_suppression": 0.9449985770592578,
                "source_suppression_fraction": 0.05500142294074217,
            },
            "branches": {
                "A_conserved_cdm_like": {"S8": 0.828924043, "sigma8": 0.807787208},
                "B_direct_jacobi_repair": {"S8": 0.79, "growth_suppression_factor": 0.953042690305944},
                "C_matrix_gapped_jacobi": {"S8": 0.8266117319670833, "growth_suppression_factor": 0.9972104669270442},
            },
            "measurement_comparisons": {
                "Planck2018_H0": {"branch_pull_sigma": 0.074},
                "SH0ES_H0": {"branch_pull_sigma": -5.423},
                "Planck2018_S8": {"cdm_pull_sigma": -0.237},
                "weak_lensing_S8_target": {
                    "cdm_pull_sigma": 2.433,
                    "direct_jacobi_pull_sigma": 0.0,
                },
            },
            "theorem_gates": {
                "Q_A_from_finite_collar_selector": False,
                "B_A_from_parent_collar_kernel": False,
                "lambda_collar_from_P_survival": True,
                "Gamma_rec_equals_Jacobi_clock": False,
                "full_CAMB_CLASS_anomaly_module": False,
                "full_likelihood_contract": False,
            },
            "physical_prediction_ready": False,
            "physical_cmb_prediction": False,
            "physical_matter_power_prediction": False,
        },
    )

    report = comparable_data_report([tmp_path])

    sync_gap = report["measurement_lanes"]["low_k_synchronization_gap"]
    assert sync_gap["run_count"] == 1
    assert sync_gap["mean_cached_proxy_pass_count"] == 4.0
    assert sync_gap["low_k_gap_established_count"] == 0
    hot_release = report["measurement_lanes"]["hot_maxent_release"]
    assert hot_release["mean_mechanical_surface_count"] == 4.0
    assert hot_release["mean_median_release_cycle"] == 10.0
    adiabaticity = report["measurement_lanes"]["same_boundary_adiabaticity"]
    assert adiabaticity["mean_max_entropy_residual_std"] == 1.4
    assert adiabaticity["established_report_count"] == 0
    h0s8 = report["measurement_lanes"]["h0_s8_branch_diagnostic"]
    assert h0s8["mean_H0_km_s_Mpc"] == 67.40002854274209
    assert h0s8["mean_direct_jacobi_S8"] == 0.79
    assert h0s8["lambda_P_gate_count"] == 1
    assert h0s8["Q_A_gate_count"] == 0


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")
