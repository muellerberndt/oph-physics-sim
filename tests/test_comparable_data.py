from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.cosmology.comparable_data import comparable_data_report, write_comparable_data_package


def test_comparable_data_collects_maxent_green_source(tmp_path: Path):
    run = tmp_path / "maxent"
    run.mkdir()
    _write_json(
        run / "maxent_green_spectrum_report.json",
        {
            "mode": "oph_maxent_green_screen_source_v0",
            "finite_regulator": {
                "patch_count": 65536,
                "bandlimit_for_ir_receipt": True,
                "bandlimit_for_requested_ell_receipt": True,
            },
            "maxent_inverse_laplacian": {"eta0_flat_D_ell_receipt": True},
            "fractional_repair_tilt": {
                "eta_R": 0.035158856969,
                "n_s": 0.964841143031,
                "fit_eta_R_abs_error": 0.0,
                "repair_clock_certificate": False,
            },
            "selector_elimination_v1_5": {
                "theorem_side_receipt": True,
                "source_packet_audit_receipt": True,
                "q_IR": 0.25,
                "ell_IR": 32.0,
                "N_frz_proxy": 1089,
            },
            "screen_spectrum": {"ell_max": 64, "F_IR_ell2": 0.751, "F_IR_ell32": 0.908},
            "MAXENT_GREEN_SOURCE_RECEIPT": True,
            "finite_lattice_derived": False,
            "physical_cmb_prediction": False,
        },
    )

    report = comparable_data_report([tmp_path])
    lane = report["measurement_lanes"]["oph_maxent_green_screen_source"]

    assert lane["run_count"] == 1
    assert lane["source_receipt_count"] == 1
    assert lane["repair_clock_certificate_count"] == 0
    assert lane["finite_lattice_derived_count"] == 0
    assert lane["physical_cmb_prediction_count"] == 0
    assert lane["mean_n_s"] == 0.964841143031


def test_comparable_data_collects_repair_clock_kappa_audit(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "repair_clock_certificate_report.json",
        {
            "mode": "oph_repair_clock_kappa_audit_v0",
            "target": {
                "required_kappa_rep": 2.718281828459045,
                "required_eta_R": 0.035158856969,
            },
            "inputs": {
                "candidate_run_count": 3,
                "cycle_time_normalization_declared": False,
            },
            "summary": {
                "estimator_count": 6,
                "eligible_estimator_count": 0,
                "passed_estimator_count": 0,
                "median_kappa_rep_estimate": 0.42,
                "median_eta_R_estimate": 0.0054,
                "median_n_s_estimate": 0.9946,
            },
            "blockers": ["no predeclared finite repair-time normalization"],
            "finite_repair_clock_certificate": False,
            "repair_clock_certificate": False,
            "eta_R_finite_lattice_derived": False,
            "physical_cmb_prediction": False,
        },
    )

    report = comparable_data_report([run])
    lane = report["measurement_lanes"]["oph_repair_clock_kappa"]

    assert lane["run_count"] == 1
    assert lane["finite_repair_clock_certificate_count"] == 0
    assert lane["eta_R_finite_lattice_derived_count"] == 0
    assert lane["mean_estimator_count"] == 6.0
    assert lane["mean_median_kappa_rep"] == 0.42
    assert lane["target_kappa_rep"] == 2.718281828459045
    assert lane["physical_cmb_prediction"] is False


def test_comparable_data_collects_scalar_repair_semigroup_lane(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "scalar_repair_semigroup_report.json",
        {
            "mode": "oph_scalar_repair_semigroup_gap_audit_v0",
            "source": "finite_state_transition_matrix",
            "dimension": 2,
            "centered_subspace_dimension": 1,
            "semigroup": {
                "kappa_rep_estimate": 31.0,
                "eta_R_estimate": 0.4,
                "n_s_estimate": 0.6,
                "centered_gap": 0.4,
            },
            "SEMIGROUP_TARGET_RECEIPT": False,
            "repair_clock_certificate": False,
            "eligible_for_repair_clock_certificate": False,
            "finite_lattice_derived": True,
            "semigroup_controls_passed": True,
        },
    )

    report = comparable_data_report([run])
    lane = report["measurement_lanes"]["oph_scalar_repair_semigroup"]

    assert lane["run_count"] == 1
    assert lane["finite_lattice_derived_count"] == 1
    assert lane["repair_clock_certificate_count"] == 0
    assert lane["mean_kappa_rep"] == 31.0
    assert lane["source_values"] == ["finite_state_transition_matrix"]


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


def test_comparable_data_prefers_bulk_proof_over_legacy_emergence_flag(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "emergence_status_report.json",
        {
            "CHART_LEVEL_CONFORMAL_LORENTZ_RECEIPT": True,
            "bulk_3d_established": True,
            "PAPER_THEOREM_ASSISTED_H3_POPULATED_CHART_RECEIPT": True,
            "OBJECT_BULK_POPULATION_RECEIPT": True,
            "OBJECT_H3_NONBOUNDARY_POPULATION_RECEIPT": True,
        },
    )
    _write_json(
        run / "bulk_proof_certificate_report.json",
        {
            "chart_level_3p1_lorentz_kinematics_established": True,
            "theorem_assisted_h3_object_preview_established": True,
            "theorem_assisted_h3_populated_chart_established": False,
            "theorem_assisted_h3_nonboundary_population_established": False,
            "bulk_3d_established_theorem_assisted": False,
            "strict_neutral_third_person_bulk_established": False,
            "screen_cmb_proxy_available": True,
            "physical_cmb_prediction": False,
        },
    )

    report = comparable_data_report([tmp_path])
    lane = report["measurement_lanes"]["support_visible_lorentz_branch"]

    assert report["bulk_3d_established"] is False
    assert lane["theorem_assisted_h3_object_preview_count"] == 1
    assert lane["paper_theorem_assisted_h3_populated_chart_count"] == 0
    assert lane["object_h3_nonboundary_population_count"] == 0
    assert lane["object_bulk_population_count"] == 0


def test_comparable_data_separates_observer_3p1d_from_populated_h3(tmp_path: Path):
    run = tmp_path / "run"
    (run / "observer_consensus_bulk").mkdir(parents=True)
    _write_json(
        run / "observer_consensus_bulk" / "observer_consensus_bulk_readout_report.json",
        {
            "observer_like_self_reading_system_receipt": True,
            "observer_modular_time_receipt": True,
            "observer_facing_3p1d_h3_experience_receipt": True,
            "observer_facing_populated_h3_experience_receipt": False,
            "observer_h3_object_population_receipt": False,
            "theorem_assisted_consensus_3d_bulk_readout_receipt": False,
            "strict_neutral_third_person_bulk_receipt": False,
        },
    )
    _write_json(
        run / "observer_modular_experience_report.json",
        {
            "observer_modular_time_receipt": True,
            "observer_facing_3p1d_h3_experience_receipt": True,
            "observer_facing_populated_h3_experience_receipt": False,
            "observer_h3_object_population_receipt": False,
        },
    )

    report = comparable_data_report([tmp_path])
    lane = report["measurement_lanes"]["support_visible_lorentz_branch"]
    row = report["rows"][0]

    assert row["observer_facing_3p1d_h3_experience_receipt"] is True
    assert row["observer_facing_populated_h3_experience_receipt"] is False
    assert report["observer_facing_3p1d_h3_experience_count"] == 1
    assert report["observer_facing_3p1d_h3_experience_any"] is True
    assert report["theorem_assisted_observer_facing_h3_population_count"] == 0
    assert report["theorem_assisted_observer_facing_h3_population_any"] is False
    assert lane["observer_facing_3p1d_h3_experience_count"] == 1
    assert lane["observer_facing_populated_h3_experience_count"] == 0


def test_comparable_data_collects_standalone_strict_neutral_report(tmp_path: Path):
    report_path = tmp_path / "strict_neutral_seed.json"
    _write_json(
        report_path,
        {
            "mode": "strict_neutral_bulk_record_transition_audit",
            "observer_count": 128,
            "dimension": {
                "estimators_agree_3d": False,
                "median_dimension_estimate": 7.5,
                "correlation_dimension": {"estimate": 8.9},
                "local_mle_dimension": {"median": 6.1},
            },
            "model_selection": {
                "best_model": "H4",
                "selected_model": "H3",
                "h3_beats_s2": True,
                "h3_beats_h2_h4": False,
            },
            "leakage": {
                "s2_leakage_pass": False,
                "s2_distance_correlation": 0.18,
            },
            "controls": {
                "rows": [
                    {"name": "planted_2d", "control_passed": True},
                    {"name": "shuffled_records", "control_passed": True},
                ]
            },
            "strict_neutral_bulk": False,
            "blockers": [
                "neutral_dimension_estimators_do_not_agree_3d",
                "s2_leakage_audit_failed",
            ],
        },
    )

    report = comparable_data_report([report_path])
    lane = report["measurement_lanes"]["neutral_observer_reconstruction"]
    row = report["rows"][0]

    assert report["run_count"] == 1
    assert report["strict_neutral_3d_bulk_count"] == 0
    assert lane["strict_neutral_report_count"] == 1
    assert lane["strict_neutral_bulk_count"] == 0
    assert lane["strict_neutral_h3_selected_count"] == 1
    assert lane["strict_neutral_h3_best_count"] == 0
    assert lane["mean_strict_neutral_median_dimension"] == 7.5
    assert lane["mean_strict_neutral_s2_distance_correlation"] == 0.18
    assert row["strict_neutral_blockers"] == [
        "neutral_dimension_estimators_do_not_agree_3d",
        "s2_leakage_audit_failed",
    ]


def test_comparable_data_collects_standalone_strict_neutral_object_report(tmp_path: Path):
    report_path = tmp_path / "strict_neutral_object_seed.json"
    _write_json(
        report_path,
        {
            "mode": "strict_neutral_object_bulk_v0",
            "object_count": 24,
            "dimension": {
                "estimators_agree_3d": True,
                "median_dimension_estimate": 3.02,
            },
            "latent_geometry_selection": {
                "selected_model": "H3",
                "h3_selected": True,
            },
            "leakage": {
                "s2_leakage_pass": True,
                "s2_distance_correlation": 0.01,
            },
            "STRICT_NEUTRAL_OBJECT_BULK_RECEIPT": False,
            "strict_neutral_object_bulk": False,
            "blockers": ["shuffled_record_object_control_did_not_fail"],
        },
    )

    report = comparable_data_report([report_path])
    lane = report["measurement_lanes"]["neutral_observer_reconstruction"]
    row = report["rows"][0]

    assert report["run_count"] == 1
    assert lane["strict_neutral_object_report_count"] == 1
    assert lane["strict_neutral_object_bulk_count"] == 0
    assert lane["strict_neutral_object_h3_selected_count"] == 1
    assert lane["mean_strict_neutral_object_count"] == 24
    assert lane["mean_strict_neutral_object_median_dimension"] == 3.02
    assert row["strict_neutral_object_selected_model"] == "H3"
    assert row["strict_neutral_object_blockers"] == ["shuffled_record_object_control_did_not_fail"]


def test_comparable_data_collects_b_a_parent_diagnostic(tmp_path: Path):
    run = tmp_path / "ba"
    run.mkdir()
    _write_json(
        run / "b_a_parent_report.json",
        {
            "mode": "report_backed_finite_collar_B_A_parent_diagnostic_v0",
            "source_report_count": 2,
            "observer_view_source_count": 2,
            "primary_parent_source": "observer_view_finite_collar_packet_variation",
            "rows": [{"B_A_mean": 0.1}, {"B_A_mean": -0.2}],
            "control_rows": [{"control": "phase_shuffled_baryon_mode"}],
            "observer_view_rows": [{"B_A_mean": 0.1}],
            "observer_view_control_rows": [{"control": "phase_shuffled_baryon_mode"}],
            "B_A_PARENT_RECEIPT": False,
            "physical_prediction_ready": False,
            "physical_cmb_prediction": False,
            "readiness": {
                "checks": {
                    "controls_fail": False,
                    "real_baryon_perturbation_runs_present": False,
                    "finite_observer_view_parent_variation": True,
                    "refinement_convergence_passed": False,
                },
                "control_failures": {
                    "phase_shuffled_baryon_mode": False,
                    "baryon_delta_applied_after_record_freezeout": True,
                },
                "missing_gates": [
                    "controls_fail",
                    "real_baryon_perturbation_runs_present",
                ],
            },
        },
    )

    report = comparable_data_report([run])
    lane = report["measurement_lanes"]["oph_B_A_parent_finite_difference"]

    assert lane["run_count"] == 1
    assert lane["receipt_count"] == 0
    assert lane["finite_observer_view_parent_variation_count"] == 1
    assert lane["mean_source_report_count"] == 2.0
    assert lane["mean_row_count"] == 2.0
    assert lane["control_failure_counts"] == {"baryon_delta_applied_after_record_freezeout": 1}
    assert lane["missing_gate_counts"] == {
        "controls_fail": 1,
        "real_baryon_perturbation_runs_present": 1,
    }


def test_comparable_data_collects_short_neutral_profile_filename(tmp_path: Path):
    run = tmp_path / "profile"
    run.mkdir()
    _write_json(
        run / "neutral_profile_audit.json",
        {
            "mode": "neutral_distance_profile_audit_v0",
            "observer_count": 32,
            "sampled_observer_count": 16,
            "profile_rows": [
                {
                    "profile": "transition_core",
                    "strict_3d_ready": False,
                    "dimension": {
                        "correlation_dimension": {"estimate": 2.9},
                        "local_mle_dimension": {"median_estimate": 3.1},
                    },
                    "model_selection": {"best_model": "H3"},
                    "leakage": {"s2_distance_correlation": 0.01, "s2_leakage_pass": True},
                    "blockers": [],
                }
            ],
        },
    )

    report = comparable_data_report([run])
    lane = report["measurement_lanes"]["neutral_distance_profile_audit"]

    assert lane["run_count"] == 1
    assert lane["mean_profile_count"] == 1


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


def test_comparable_data_collects_oph_exact_cmb_camb_transfer(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "oph_exact_cmb_camb_report.json",
        {
            "mode": "oph_exact_cmb_camb_transfer_v1",
            "oph_exact_input": {
                "n_s": 0.964841143031,
                "eta_R": 0.035158856969,
                "q_IR": 0.25,
                "ell_IR": 32.0,
                "N_frz_proxy": 1089,
            },
            "comparison": {
                "camb_lcdm_powerlaw": {
                    "shape_correlation": 0.995,
                    "amplitude_fit_chi2_per_bin": 1.4,
                },
                "oph_exact_scalar_tilt": {
                    "shape_correlation": 0.996,
                    "amplitude_fit_chi2_per_bin": 1.35,
                },
                "oph_exact_ir_v10": {
                    "shape_correlation": 0.997,
                    "amplitude_fit_chi2_per_bin": 1.31,
                },
            },
            "acoustic_preservation": {
                "mean_abs_fractional_delta_ell_ge_50": 0.001,
            },
            "official_planck_likelihood_readiness": {
                "official_clik_api_available": False,
                "official_likelihood_execution_ready": False,
            },
            "measurement_comparable_cmb_curve": True,
            "physical_cmb_prediction": False,
            "screen_camb_transfer_receipt": True,
        },
    )

    report = comparable_data_report([tmp_path])
    lane = report["measurement_lanes"]["oph_exact_cmb_camb_transfer"]

    assert lane["run_count"] == 1
    assert lane["measurement_comparable_curve_count"] == 1
    assert lane["transfer_receipt_count"] == 1
    assert lane["official_likelihood_ready_count"] == 0
    assert lane["mean_n_s"] == 0.964841143031
    assert lane["mean_q_IR"] == 0.25
    assert lane["mean_ell_IR"] == 32.0
    assert lane["mean_ir_chi2_per_bin"] == 1.31


def test_comparable_data_collects_cmb_selector_elimination(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "oph_cmb_selector_elimination_report.json",
        {
            "mode": "oph_cmb_selector_elimination_v1_5",
            "selector_elimination": {
                "q_IR_selector_removed": True,
                "ell_IR_selector_removed": True,
                "eta_R_reduced_to_repair_clock_certificate": True,
            },
            "scalar_tilt": {
                "n_s": 0.964841143031,
                "eta_R": 0.035158856969,
                "canonical_kappa_rep_status": "certificate_pending",
            },
            "cmb_ir_kernel": {
                "q_IR": 0.25,
                "ell_IR": 32.0,
            },
            "exact_ir_kernel_csv_audit": {
                "passed": True,
                "max_abs_error": 1.0e-15,
            },
            "THEOREM_SIDE_SELECTOR_ELIMINATION_RECEIPT": True,
            "SOURCE_PACKET_AUDIT_RECEIPT": True,
            "finite_lattice_derived": False,
            "physical_cmb_prediction": False,
        },
    )

    report = comparable_data_report([tmp_path])
    lane = report["measurement_lanes"]["oph_cmb_selector_elimination_v1_5"]

    assert lane["run_count"] == 1
    assert lane["theorem_side_receipt_count"] == 1
    assert lane["source_packet_audit_receipt_count"] == 1
    assert lane["q_IR_selector_removed_count"] == 1
    assert lane["ell_IR_selector_removed_count"] == 1
    assert lane["eta_R_repair_clock_reduction_count"] == 1
    assert lane["finite_lattice_derived_count"] == 0
    assert lane["mean_q_IR"] == 0.25
    assert lane["mean_ell_IR"] == 32.0


def test_comparable_data_collects_finite_repair_clock_cmb_camb(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "finite_repair_clock_cmb_camb_report.json",
        {
            "mode": "finite_repair_clock_cmb_camb_transfer_v0",
            "measurement_comparable_cmb_curve": True,
            "screen_camb_transfer_receipt": True,
            "finite_lattice_clock_derived": True,
            "repair_clock_certificate": False,
            "selector_ir_theory_side": True,
            "physical_cmb_prediction": False,
            "finite_repair_clock_input": {
                "n_s": 0.9679812500795765,
                "eta_R": 0.03201874992042351,
                "kappa_rep": 2.4755067024747386,
            },
            "selector_ir_input": {"q_IR": 0.25, "ell_IR": 32.0},
            "comparison": {
                "camb_lcdm_powerlaw": {"shape_correlation": 0.999, "amplitude_fit_chi2_per_bin": 1.0},
                "finite_repair_clock_scalar_tilt": {
                    "shape_correlation": 0.998,
                    "amplitude_fit_chi2_per_bin": 1.1,
                },
                "finite_repair_clock_plus_selector_ir": {
                    "shape_correlation": 0.997,
                    "amplitude_fit_chi2_per_bin": 1.2,
                },
            },
            "acoustic_preservation": {"mean_abs_fractional_delta_ell_ge_50": 0.01},
        },
    )

    report = comparable_data_report([run])
    lane = report["measurement_lanes"]["finite_repair_clock_cmb_camb_transfer"]

    assert lane["run_count"] == 1
    assert lane["measurement_comparable_curve_count"] == 1
    assert lane["finite_lattice_clock_derived_count"] == 1
    assert lane["repair_clock_certificate_count"] == 0
    assert lane["physical_cmb_prediction_count"] == 0
    assert lane["mean_n_s"] == 0.9679812500795765
    assert lane["mean_kappa_rep"] == 2.4755067024747386
    assert lane["mean_scalar_shape_correlation"] == 0.998
    assert lane["mean_ir_chi2_per_bin"] == 1.2


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
                "LOCAL_POISSON_RESERVE_SURVIVAL": True,
                "SCALAR_WEIGHTED_Z6_MEAN": False,
                "UNIFORM_PRODUCT_THICKENING_EXACT": False,
                "lambda_collar_from_P_survival": False,
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
    assert h0s8["lambda_P_gate_count"] == 0
    assert h0s8["local_poisson_reserve_survival_gate_count"] == 1
    assert h0s8["scalar_weighted_z6_mean_gate_count"] == 0
    assert h0s8["uniform_product_thickening_exact_gate_count"] == 0
    assert h0s8["Q_A_gate_count"] == 0


def test_comparable_data_collects_oph_cnb_neutrino_background(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "oph_cnb_neutrino_report.json",
        {
            "mode": "oph_cnb_neutrino_background_v1",
            "oph_neutrino_mass_status": {
                "available": False,
                "sum_mnu_eV": None,
                "public_promotion_allowed": False,
            },
            "conventional_camb_baseline": {
                "sum_mnu_eV": 0.06,
                "relic_background": {
                    "N_eff": 3.044,
                    "Omega_nu_h2": 0.0006443298969072165,
                    "Omega_nu": 0.0014183665809050366,
                    "f_nu": 0.004495615153423254,
                    "small_scale_power_suppression_fraction": -0.03596492122738603,
                },
                "measurement_comparisons": {
                    "Planck2018_N_eff": {"pull_sigma": 0.3176470588235297},
                    "Planck2018_BAO_sum_mnu_bound": {"passes_bound": True},
                    "ACT_DR6_extended_sum_mnu_bound": {"passes_bound": True},
                    "DESI_DR2_LCDM_sum_mnu_bound": {"passes_bound": True},
                },
            },
            "historical_rejected_weighted_cycle_benchmark": {"included": False},
            "late_repair_projection_target": {
                "eta_A": 0.0656993605106136,
                "Pi_WL_compressed_required": 0.7147300876,
            },
            "readiness_gates": {
                "conventional_baseline_relic_background_callable": True,
                "finite_lattice_mass_derivation": False,
                "B_A_k_a_from_finite_collar_parent": False,
                "full_boltzmann_likelihood_run": False,
            },
            "measurement_comparable_now": False,
            "conventional_baseline_measurement_comparable": True,
            "finite_lattice_derived": False,
            "physical_cmb_prediction": False,
            "physical_matter_power_prediction": False,
        },
    )

    report = comparable_data_report([run])
    cnb = report["measurement_lanes"]["oph_cnb_neutrino_background"]

    assert cnb["run_count"] == 1
    assert cnb["measurement_comparable_count"] == 1
    assert cnb["finite_lattice_derived_count"] == 0
    assert cnb["background_gate_count"] == 1
    assert cnb["B_A_kernel_gate_count"] == 0
    assert cnb["planck_bao_bound_pass_count"] == 1
    assert cnb["act_bound_pass_count"] == 1
    assert cnb["desi_lcdm_bound_pass_count"] == 1
    assert cnb["oph_mass_prediction_available_count"] == 0
    assert cnb["mean_sum_mnu_eV"] is None
    assert cnb["mean_conventional_sum_mnu_eV"] == 0.06
    assert cnb["mean_Pi_WL_compressed_required"] == 0.7147300876


def test_comparable_data_collects_inflation_certificate_stack(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "inflation_certificate_report.json",
        {
            "mode": "oph_inflation_certificate_bundle_v0",
            "certificate_summary": {
                "found_count": 2,
                "passed_count": 1,
                "expected_count": 6,
                "missing_types": ["homogeneous_anomaly", "parent_collar", "repair_matrix", "boltzmann_handoff"],
            },
            "readiness_gates": {
                "scalar_release_certificate": True,
                "edge_center_certificate": False,
                "homogeneous_anomaly_certificate": False,
                "parent_collar_kernel_certificate": False,
                "repair_matrix_certificate": False,
                "boltzmann_handoff_certificate": False,
                "no_data_use_firewall": True,
            },
            "derived_outputs": {
                "scalar_release": {"A_zeta": 2.1e-9},
                "edge_center": {"n_s": 0.966},
            },
            "no_data_use_manifest": {"no_data_use_receipt": True},
            "inflation_certificate_stack_ready": False,
            "physical_cmb_prediction": False,
            "physical_matter_power_prediction": False,
        },
    )

    report = comparable_data_report([run])
    certs = report["measurement_lanes"]["oph_inflation_certificate_stack"]

    assert certs["run_count"] == 1
    assert certs["stack_ready_count"] == 0
    assert certs["scalar_release_gate_count"] == 1
    assert certs["edge_center_gate_count"] == 0
    assert certs["no_data_use_count"] == 1
    assert certs["mean_found_count"] == 2.0
    assert certs["mean_passed_count"] == 1.0
    assert certs["mean_A_zeta"] == 2.1e-9
    assert certs["mean_n_s"] == 0.966


def test_comparable_data_collects_fossil_spectrum_time_trace(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "fossil_spectrum_report.json",
        {
            "mode": "oph_fossil_spectrum_time_resolved_diagnostic_v0",
            "cycle_markers": {"freezeout_cycle": 21, "phi_zero_cycle": 15, "phi_half_cycle": 7},
            "best_target_closeness_diagnostic": {
                "field": "record_signature",
                "cycle": 12,
                "eta_R": 0.005,
                "n_s": 0.995,
                "abs_eta_R_delta_to_planck": 0.03,
            },
            "best_same_field_control_delta_to_planck": 2.0,
            "best_beats_same_field_controls": True,
            "near_scale_invariant_transient": False,
            "physical_cmb_prediction": False,
        },
    )

    report = comparable_data_report([run])
    fossil = report["measurement_lanes"]["oph_fossil_spectrum_time_trace"]

    assert fossil["run_count"] == 1
    assert fossil["near_scale_invariant_transient_count"] == 0
    assert fossil["best_beats_same_field_controls_count"] == 1
    assert fossil["best_field_counts"] == {"record_signature": 1}
    assert fossil["mean_best_cycle"] == 12.0
    assert fossil["mean_best_eta_R"] == 0.005
    assert fossil["physical_cmb_prediction"] is False


def test_comparable_data_collects_control_quotient_spatial_3d_candidate(tmp_path: Path):
    run = tmp_path / "rank_sweep"
    run.mkdir()
    _write_json(
        run / "prime_geometric_rank_sweep_report.json",
        {
            "mode": "prime_geometric_rank_sweep_v0",
            "prime_geometric_control_quotient_spatial_3d_candidate_receipt": True,
            "prime_geometric_spatial_3d_candidate_receipt": False,
            "prime_geometric_strict_neutral_candidate_receipt": False,
            "selected_rank_controls": {
                "all_expected_failures_observed": True,
                "coordinate_rank3_tautology_warning": True,
                "control_rows": [
                    {
                        "metric": "coordinate_euclidean",
                        "rank": 3,
                        "excluded_from_selected_rank_gate": True,
                        "survives": True,
                    },
                    {
                        "metric": "directional_cosine",
                        "rank": 3,
                        "excluded_from_selected_rank_gate": False,
                        "survives": False,
                    },
                ],
            },
            "control_quotient_coordinate_spatial_3d_ready_count": 1,
            "control_quotient_coordinate_best_3d_dimension_row": {
                "rank": 3,
                "dimension": {
                    "correlation_dimension": {"estimate": 2.83},
                    "local_mle_dimension": {"median_estimate": 3.02},
                },
                "model_selection": {"best_model": "E3"},
                "leakage": {
                    "s2_distance_correlation": 0.023,
                    "s2_leakage_pass": True,
                },
            },
            "proof_blockers": ["requires_refinement_stability_across_regulator_sizes"],
        },
    )

    report = comparable_data_report([run])
    lane = report["measurement_lanes"]["prime_geometric_rank_sweep"]
    row = report["rows"][0]

    assert row["prime_rank_sweep_control_quotient_spatial_3d_candidate_receipt"] is True
    assert row["prime_rank_sweep_spatial_3d_candidate_receipt"] is False
    assert row["prime_rank_sweep_strict_neutral_candidate_receipt"] is False
    assert row["prime_rank_sweep_control_quotient_coordinate_best_3d_rank"] == 3
    assert row["prime_rank_sweep_control_quotient_coordinate_best_3d_corr_dim"] == 2.83
    assert row["prime_rank_sweep_control_quotient_coordinate_best_3d_mle_dim"] == 3.02
    assert row["prime_rank_sweep_control_quotient_coordinate_best_3d_model"] == "E3"
    assert row["prime_rank_sweep_control_quotient_coordinate_best_3d_s2_leakage_corr"] == 0.023
    assert row["prime_rank_sweep_control_quotient_coordinate_best_3d_s2_leakage_pass"] is True
    assert lane["control_quotient_spatial_3d_candidate_receipt_count"] == 1
    assert lane["spatial_3d_candidate_receipt_count"] == 0
    assert lane["strict_neutral_candidate_receipt_count"] == 0
    assert lane["control_quotient_coordinate_best_3d_rank_counts"] == {"3": 1}
    assert lane["control_quotient_coordinate_best_3d_model_counts"] == {"E3": 1}
    assert lane["control_quotient_coordinate_mean_best_3d_corr_dim"] == 2.83
    assert lane["control_quotient_coordinate_mean_best_3d_mle_dim"] == 3.02
    assert lane["control_quotient_coordinate_mean_best_3d_s2_leakage_corr"] == 0.023
    assert lane["control_quotient_coordinate_best_3d_s2_leakage_pass_count"] == 1


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")
