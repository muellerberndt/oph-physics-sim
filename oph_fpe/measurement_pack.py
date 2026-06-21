from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any


def export_measurement_pack(run_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    roots = [Path(path) for path in run_dirs]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    exported: dict[str, str] = {}
    _copy_first(roots, out / "static_galaxy_measurement_report.json", exported, "static_galaxy_measurement_report.json")
    _copy_first(roots, out / "galaxy_rar_fit.csv", exported, "galaxy_rar_fit.csv")
    _copy_first(roots, out / "galaxy_btfr_fit.csv", exported, "galaxy_btfr_fit.csv")
    _copy_first(roots, out / "galaxy_rotation_residuals.csv", exported, "galaxy_rotation_residuals.csv")
    _write_cmb_screen_cl(roots, out / "cmb_screen_cl.csv")
    exported["cmb_screen_cl.csv"] = "aggregated cl_proxy.csv/cl_comparison_rows.csv rows"
    _copy_first(
        roots,
        out / "cmb_fossil_bridge_tt.csv",
        exported,
        "oph_exact_cmb_tt_curves.csv",
        "cmb_fossil_bridge_tt.csv",
    )
    _copy_first(
        roots,
        out / "cmb_fossil_bridge_params.json",
        exported,
        "cmb_fossil_bridge_params.json",
        "cmb_fossil_bridge_report.json",
        "oph_exact_cmb_camb_report.json",
    )
    _write_h3_objects(roots, out / "h3_objects.csv")
    _write_h3_defects(roots, out / "h3_defects.csv")
    _copy_first(roots, out / "shape_loop_particles.csv", exported, "shape_loop_particles.csv")
    _copy_first(roots, out / "shape_screen_cl.csv", exported, "shape_screen_cl.csv")
    _copy_first(roots, out / "shape_settling_trace.csv", exported, "shape_settling_trace.csv")
    _copy_first(roots, out / "emergence_status_report.json", exported, "emergence_status_report.json")
    _copy_first(roots, out / "receipt_ladder_report.json", exported, "receipt_ladder_report.json")
    _copy_first(roots, out / "bulk_reconstruction_report.json", exported, "bulk_reconstruction_report.json")
    _copy_first(roots, out / "cmb_lite_comparison_report.json", exported, "cmb_lite_comparison_report.json")
    _copy_first(roots, out / "cl_comparison_report.json", exported, "cl_comparison_report.json")
    _copy_first(roots, out / "particle_likeness_report.json", exported, "particle_likeness_report.json")
    _copy_first(
        roots,
        out / "controlled_defect_particle_assay_report.json",
        exported,
        "controlled_defect_particle_assay_report.json",
    )
    _copy_first(roots, out / "neutral_profile_audit_report.json", exported, "neutral_profile_audit_report.json")
    _copy_first(roots, out / "prime_geometric_rank_sweep_report.json", exported, "prime_geometric_rank_sweep_report.json")
    _copy_first(
        roots,
        out / "prime_geometric_rank_refinement_report.json",
        exported,
        "prime_geometric_rank_refinement_report.json",
    )
    _copy_preferred_report_pair(
        roots,
        out / "overlap_native_neutral_control_report.json",
        out / "overlap_native_neutral_control_report.md",
        exported,
        "overlap_native_neutral_control_report.json",
        "overlap_native_neutral_control_report.md",
    )
    _copy_preferred_report_pair(
        roots,
        out / "overlap_native_graph_geometry_report.json",
        out / "overlap_native_graph_geometry_report.md",
        exported,
        "overlap_native_graph_geometry_report.json",
        "overlap_native_graph_geometry_report.md",
    )
    _copy_preferred_report_pair(
        roots,
        out / "overlap_native_graph_geometry_sweep_report.json",
        out / "overlap_native_graph_geometry_sweep_report.md",
        exported,
        "overlap_native_graph_geometry_sweep_report.json",
        "overlap_native_graph_geometry_sweep_report.md",
    )
    _copy_first(
        roots,
        out / "overlap_native_graph_geometry_sweep_rows.csv",
        exported,
        "overlap_native_graph_geometry_sweep_rows.csv",
    )
    _copy_preferred_report_pair(
        roots,
        out / "overlap_residualized_graph_geometry_report.json",
        out / "overlap_residualized_graph_geometry_report.md",
        exported,
        "overlap_residualized_graph_geometry_report.json",
        "overlap_residualized_graph_geometry_report.md",
    )
    _copy_preferred_report_pair(
        roots,
        out / "overlap_residualized_graph_geometry_sweep_report.json",
        out / "overlap_residualized_graph_geometry_sweep_report.md",
        exported,
        "overlap_residualized_graph_geometry_sweep_report.json",
        "overlap_residualized_graph_geometry_sweep_report.md",
    )
    _copy_first(
        roots,
        out / "overlap_residualized_graph_geometry_sweep_rows.csv",
        exported,
        "overlap_residualized_graph_geometry_sweep_rows.csv",
    )
    _copy_preferred_report_pair(
        roots,
        out / "neutral_3d_bulk_audit_report.json",
        out / "neutral_3d_bulk_audit_report.md",
        exported,
        "neutral_3d_bulk_audit_report.json",
        "neutral_3d_bulk_audit_report.md",
    )
    _copy_preferred_report_pair(
        roots,
        out / "neutral_independent_rank_selector_audit_report.json",
        out / "neutral_independent_rank_selector_audit_report.md",
        exported,
        "neutral_independent_rank_selector_audit_report.json",
        "neutral_independent_rank_selector_audit_report.md",
    )
    _copy_preferred_report_pair(
        roots,
        out / "strict_neutral_bulk_frontier_report.json",
        out / "strict_neutral_bulk_frontier_report.md",
        exported,
        "strict_neutral_bulk_frontier_report.json",
        "strict_neutral_bulk_frontier_report.md",
    )
    _copy_first(roots, out / "strict_neutral_bulk_report.json", exported, "strict_neutral_bulk_report.json")
    _copy_first(
        roots,
        out / "strict_neutral_object_bulk_report.json",
        exported,
        "strict_neutral_object_bulk_report.json",
    )
    _copy_first(roots, out / "neutral_objects.jsonl", exported, "neutral_objects.jsonl")
    _copy_first(
        roots,
        out / "observer_chart_object_h3_lineage_report.json",
        exported,
        "observer_chart_object_h3_lineage_report.json",
    )
    _copy_first(
        roots,
        out / "observer_chart_object_h3_report.json",
        exported,
        "observer_chart_object_h3_report.json",
    )
    _copy_first(
        roots,
        out / "observer_chart_object_h3_transition_history_report.json",
        exported,
        "observer_chart_object_h3_transition_history_report.json",
    )
    _copy_first(
        roots,
        out / "observer_chart_object_h3_observer_transition_mixture_report.json",
        exported,
        "observer_chart_object_h3_observer_transition_mixture_report.json",
    )
    _copy_first(
        roots,
        out / "observer_chart_object_h3_scale_compressed_report.json",
        exported,
        "observer_chart_object_h3_scale_compressed_report.json",
    )
    _copy_first(
        roots,
        out / "observer_modular_experience_report.json",
        exported,
        "observer_modular_experience_report.json",
    )
    _copy_first(roots, out / "observer_views.jsonl", exported, "observer_views.jsonl")
    _copy_first(roots, out / "bulk_proof_certificate_report.json", exported, "bulk_proof_certificate_report.json")
    _copy_first(roots, out / "bulk_proof_certificate_report.md", exported, "bulk_proof_certificate_report.md")
    _copy_first(roots, out / "paper_3d_bulk_chart_report.json", exported, "paper_3d_bulk_chart_report.json")
    _copy_first(
        roots,
        out / "conformal_h3_spatial_chart_report.json",
        exported,
        "conformal_h3_spatial_chart_report.json",
    )
    _copy_first(
        roots,
        out / "transition_selection_report.json",
        exported,
        "transition_selection_report.json",
        "transition_scale_selection_report.json",
    )
    _copy_first(roots, out / "comparable_data_snapshot.json", exported, "comparable_data_snapshot.json")
    _copy_first(roots, out / "comparable_data_snapshot.md", exported, "comparable_data_snapshot.md")
    _copy_first(roots, out / "finite_certificate_report.json", exported, "finite_certificate_report.json")
    _copy_first(roots, out / "finite_certificate_manifest.json", exported, "finite_certificate_manifest.json")
    _copy_first(
        roots,
        out / "finite_repair_transition_matrix_report.json",
        exported,
        "finite_repair_transition_matrix_report.json",
    )
    _copy_first(roots, out / "finite_repair_transition_rows.csv", exported, "finite_repair_transition_rows.csv")
    _copy_first(roots, out / "finite_repair_transition_matrix.npz", exported, "finite_repair_transition_matrix.npz")
    _copy_first(roots, out / "scalar_repair_semigroup_report.json", exported, "scalar_repair_semigroup_report.json")
    _copy_first(roots, out / "repair_scale_closure_report.json", exported, "repair_scale_closure_report.json")
    _copy_first(roots, out / "repair_scale_closure_report.md", exported, "repair_scale_closure_report.md")
    _copy_first(roots, out / "repair_scale_round_depth.csv", exported, "repair_scale_round_depth.csv")
    _copy_first(roots, out / "screen_capacity_closure_report.json", exported, "screen_capacity_closure_report.json")
    _copy_first(roots, out / "screen_capacity_closure_report.md", exported, "screen_capacity_closure_report.md")
    _copy_first(roots, out / "capacity_readback_proxy_report.json", exported, "capacity_readback_proxy_report.json")
    _copy_first(roots, out / "capacity_readback_proxy_report.md", exported, "capacity_readback_proxy_report.md")
    _copy_first(roots, out / "capacity_readback_proxy_rows.csv", exported, "capacity_readback_proxy_rows.csv")
    _copy_first(roots, out / "pn_resonance_report.json", exported, "pn_resonance_report.json")
    _copy_first(roots, out / "pn_resonance_report.md", exported, "pn_resonance_report.md")
    _copy_first(
        roots,
        out / "silence_to_observation_report.json",
        exported,
        "silence_to_observation_report.json",
    )
    _copy_first(
        roots,
        out / "silence_to_observation_report.md",
        exported,
        "silence_to_observation_report.md",
    )
    _copy_first(roots, out / "kernel_dispatch_report.json", exported, "kernel_dispatch_report.json")
    _copy_first(roots, out / "kernel_dispatch_report.md", exported, "kernel_dispatch_report.md")
    _copy_first(roots, out / "positive_geometry_kernel_report.json", exported, "positive_geometry_kernel_report.json")
    _copy_first(roots, out / "positive_geometry_kernel_report.md", exported, "positive_geometry_kernel_report.md")
    _copy_first(roots, out / "positive_geometry_kernel_manifest.json", exported, "positive_geometry_kernel_manifest.json")
    _copy_first(roots, out / "positive_geometry_kernel_receipt.json", exported, "positive_geometry_kernel_receipt.json")
    _copy_first(roots, out / "oph_scale_bridge_report.json", exported, "oph_scale_bridge_report.json")
    _copy_first(roots, out / "oph_scale_bridge_report.md", exported, "oph_scale_bridge_report.md")
    _copy_first(roots, out / "no_g_clock_bridge_report.json", exported, "no_g_clock_bridge_report.json")
    _copy_first(roots, out / "no_g_clock_bridge_report.md", exported, "no_g_clock_bridge_report.md")
    _copy_first(roots, out / "parent_collar_ladder_report.json", exported, "parent_collar_ladder_report.json")
    _copy_first(roots, out / "parent_collar_ladder_report.md", exported, "parent_collar_ladder_report.md")
    _copy_first(roots, out / "repair_clock_certificate_report.json", exported, "repair_clock_certificate_report.json")
    _copy_first(roots, out / "repair_clock_certificate_report.md", exported, "repair_clock_certificate_report.md")
    _copy_first(roots, out / "repair_clock_estimators.csv", exported, "repair_clock_estimators.csv")
    _copy_first(roots, out / "oph_boltzmann_input_report.json", exported, "oph_boltzmann_input_report.json")
    _copy_first(roots, out / "oph_boltzmann_input_report.md", exported, "oph_boltzmann_input_report.md")
    _copy_first(roots, out / "oph_boltzmann_cdm_limit_rows.csv", exported, "oph_boltzmann_cdm_limit_rows.csv")
    _copy_first(
        roots,
        out / "oph_boltzmann_diagnostic_repair_rows.csv",
        exported,
        "oph_boltzmann_diagnostic_repair_rows.csv",
    )
    _copy_first(
        roots,
        out / "oph_boltzmann_b_a_parent_rows.csv",
        exported,
        "oph_boltzmann_b_a_parent_rows.csv",
    )
    _copy_first(
        roots,
        out / "oph_boltzmann_finite_repair_clock_rows.csv",
        exported,
        "oph_boltzmann_finite_repair_clock_rows.csv",
    )
    _copy_first(
        roots,
        out / "finite_collar_boltzmann_bundle_report.json",
        exported,
        "finite_collar_boltzmann_bundle_report.json",
    )
    _copy_first(
        roots,
        out / "finite_collar_boltzmann_bundle_report.md",
        exported,
        "finite_collar_boltzmann_bundle_report.md",
    )
    _copy_first(
        roots,
        out / "finite_collar_B_A_k_a_diagnostic.csv",
        exported,
        "finite_collar_B_A_k_a_diagnostic.csv",
    )
    _copy_first(
        roots,
        out / "finite_collar_rho_A_a_diagnostic.csv",
        exported,
        "finite_collar_rho_A_a_diagnostic.csv",
    )
    _copy_first(
        roots,
        out / "finite_collar_Gamma_rec_k_a_diagnostic.csv",
        exported,
        "finite_collar_Gamma_rec_k_a_diagnostic.csv",
    )
    _copy_first(
        roots,
        out / "finite_collar_cmb_projection_report.json",
        exported,
        "finite_collar_cmb_projection_report.json",
    )
    _copy_first(
        roots,
        out / "finite_collar_cmb_projection_report.md",
        exported,
        "finite_collar_cmb_projection_report.md",
    )
    _copy_first(
        roots,
        out / "finite_collar_projected_B_A_rows.csv",
        exported,
        "finite_collar_projected_B_A_rows.csv",
    )
    _copy_first(
        roots,
        out / "finite_collar_projected_background_rows.csv",
        exported,
        "finite_collar_projected_background_rows.csv",
    )
    _copy_first(roots, out / "physical_cmb_input_report.json", exported, "physical_cmb_input_report.json")
    _copy_first(roots, out / "physical_cmb_input_report.md", exported, "physical_cmb_input_report.md")
    _copy_first(
        roots,
        out / "physical_cmb_promotion_audit_report.json",
        exported,
        "physical_cmb_promotion_audit_report.json",
    )
    _copy_first(
        roots,
        out / "physical_cmb_promotion_audit_report.md",
        exported,
        "physical_cmb_promotion_audit_report.md",
    )
    _copy_first(
        roots,
        out / "physical_cmb_frontier_report.json",
        exported,
        "physical_cmb_frontier_report.json",
    )
    _copy_first(
        roots,
        out / "physical_cmb_frontier_report.md",
        exported,
        "physical_cmb_frontier_report.md",
    )
    _copy_first(
        roots,
        out / "physical_cmb_output_comparison_report.json",
        exported,
        "physical_cmb_output_comparison_report.json",
    )
    _copy_first(
        roots,
        out / "physical_cmb_output_comparison_report.md",
        exported,
        "physical_cmb_output_comparison_report.md",
    )
    _copy_first(
        roots,
        out / "physical_cmb_output_comparison_rows.csv",
        exported,
        "physical_cmb_output_comparison_rows.csv",
    )
    _copy_first(
        roots,
        out / "physical_cmb_best_oph_residuals.csv",
        exported,
        "physical_cmb_best_oph_residuals.csv",
    )
    _copy_first(
        roots,
        out / "physical_cmb_peak_features.csv",
        exported,
        "physical_cmb_peak_features.csv",
    )
    _copy_first(
        roots,
        out / "official_planck_likelihood_readiness_report.json",
        exported,
        "official_planck_likelihood_readiness_report.json",
    )
    _copy_first(
        roots,
        out / "official_planck_likelihood_readiness_report.md",
        exported,
        "official_planck_likelihood_readiness_report.md",
    )
    _copy_first(roots, out / "physical_cmb_input_contract.json", exported, "physical_cmb_input_contract.json")
    _copy_first(roots, out / "physical_cmb_input_validation.json", exported, "physical_cmb_input_validation.json")
    _copy_first(roots, out / "physical_cmb_B_A_k_a.csv", exported, "B_A_k_a.csv")
    _copy_first(roots, out / "physical_cmb_Gamma_rec_k_a.csv", exported, "Gamma_rec_k_a.csv")
    _copy_first(roots, out / "physical_cmb_rho_A_a.csv", exported, "rho_A_a.csv")
    _copy_first(
        roots,
        out / "finite_repair_clock_cmb_camb_report.json",
        exported,
        "finite_repair_clock_cmb_camb_report.json",
    )
    _copy_first(
        roots,
        out / "finite_repair_clock_cmb_camb_report.md",
        exported,
        "finite_repair_clock_cmb_camb_report.md",
    )
    _copy_first(
        roots,
        out / "finite_repair_clock_cmb_tt_bins.csv",
        exported,
        "finite_repair_clock_cmb_tt_bins.csv",
    )
    _copy_first(
        roots,
        out / "finite_repair_clock_cmb_tt_curves.csv",
        exported,
        "finite_repair_clock_cmb_tt_curves.csv",
    )
    _copy_first(roots, out / "camb_lcdm_baseline_report.json", exported, "camb_lcdm_baseline_report.json")
    _copy_first(roots, out / "camb_lcdm_baseline_report.md", exported, "camb_lcdm_baseline_report.md")
    _copy_first(roots, out / "camb_lcdm_tt_bins.csv", exported, "camb_lcdm_tt_bins.csv")
    _copy_first(roots, out / "b_a_parent_report.json", exported, "b_a_parent_report.json")
    _copy_first(roots, out / "b_a_parent_report.md", exported, "b_a_parent_report.md")
    _copy_first(roots, out / "b_a_parent_rows.csv", exported, "b_a_parent_rows.csv")
    _copy_first(roots, out / "b_a_parent_control_rows.csv", exported, "b_a_parent_control_rows.csv")
    _copy_first(roots, out / "B_A_kernel_report.json", exported, "B_A_kernel_report.json")
    _copy_first(roots, out / "B_A_kernel_report.md", exported, "B_A_kernel_report.md")
    _copy_first(roots, out / "B_A_kernel_candidate.csv", exported, "B_A_kernel_candidate.csv")
    _copy_first(roots, out / "B_A_kernel_refinement_report.json", exported, "B_A_kernel_refinement_report.json")
    _copy_first(roots, out / "B_A_kernel_refinement_report.md", exported, "B_A_kernel_refinement_report.md")
    _copy_first(roots, out / "B_A_kernel_refinement_pairs.csv", exported, "B_A_kernel_refinement_pairs.csv")
    _copy_first(
        roots,
        out / "B_A_kernel_refinement_key_pairs.csv",
        exported,
        "B_A_kernel_refinement_key_pairs.csv",
    )
    _copy_first(roots, out / "paired_b_a_perturbation_report.json", exported, "paired_b_a_perturbation_report.json")
    _copy_first(roots, out / "paired_b_a_perturbation_report.md", exported, "paired_b_a_perturbation_report.md")
    _copy_first(roots, out / "paired_b_a_perturbation_rows.csv", exported, "paired_b_a_perturbation_rows.csv")
    _copy_first(
        roots,
        out / "paired_b_a_perturbation_control_rows.csv",
        exported,
        "paired_b_a_perturbation_control_rows.csv",
    )
    _copy_first(roots, out / "b_a_parent_observer_view_rows.csv", exported, "b_a_parent_observer_view_rows.csv")
    _copy_first(
        roots,
        out / "b_a_parent_observer_view_control_rows.csv",
        exported,
        "b_a_parent_observer_view_control_rows.csv",
    )
    _copy_first(roots, out / "b_a_parent_stress_surrogate_rows.csv", exported, "b_a_parent_stress_surrogate_rows.csv")
    _copy_first(roots, out / "oph_screen_power_report.json", exported, "oph_screen_power_report.json")
    _copy_first(roots, out / "oph_screen_power_report.md", exported, "oph_screen_power_report.md")
    _copy_first(roots, out / "oph_screen_power_fit_rows.csv", exported, "oph_screen_power_fit_rows.csv")
    _copy_path(
        roots,
        out / "oph_screen_power_primordial_table.csv",
        exported,
        "screen_power/oph_primordial_power_table.csv",
        "oph_primordial_power_table.csv",
    )
    _copy_path(
        roots,
        out / "oph_screen_power_CLASS_CAMB.txt",
        exported,
        "screen_power/oph_primordial_power_CLASS_CAMB.txt",
        "oph_primordial_power_CLASS_CAMB.txt",
    )
    _copy_first(roots, out / "maxent_green_spectrum_report.json", exported, "maxent_green_spectrum_report.json")
    _copy_first(roots, out / "maxent_green_spectrum_report.md", exported, "maxent_green_spectrum_report.md")
    _copy_first(roots, out / "maxent_green_spectrum_rows.csv", exported, "maxent_green_spectrum_rows.csv")
    _copy_path(
        roots,
        out / "maxent_green_primordial_table.csv",
        exported,
        "maxent_green/oph_primordial_power_table.csv",
    )
    _copy_path(
        roots,
        out / "maxent_green_CLASS_CAMB.txt",
        exported,
        "maxent_green/oph_primordial_power_CLASS_CAMB.txt",
    )
    _copy_first(
        roots,
        out / "oph_cmb_selector_elimination_report.json",
        exported,
        "oph_cmb_selector_elimination_report.json",
    )
    _copy_first(
        roots,
        out / "oph_cmb_selector_elimination_report.md",
        exported,
        "oph_cmb_selector_elimination_report.md",
    )
    _copy_first(roots, out / "exact_ir_kernel_values_v1_5.csv", exported, "exact_ir_kernel_values_v1_5.csv")
    _copy_first(roots, out / "cmb_anomaly_report.json", exported, "cmb_anomaly_report.json")
    _copy_first(roots, out / "cmb_anomaly_report.md", exported, "cmb_anomaly_report.md")
    _copy_first(roots, out / "cmb_anomaly_rows.csv", exported, "cmb_anomaly_rows.csv")
    _copy_first(roots, out / "oph_cnb_neutrino_report.json", exported, "oph_cnb_neutrino_report.json")
    _copy_first(roots, out / "oph_cnb_neutrino_report.md", exported, "oph_cnb_neutrino_report.md")
    _copy_first(roots, out / "oph_cnb_neutrino_mass_rows.csv", exported, "oph_cnb_neutrino_mass_rows.csv")
    _copy_first(
        roots,
        out / "oph_cnb_neutrino_comparison_rows.csv",
        exported,
        "oph_cnb_neutrino_comparison_rows.csv",
    )
    _copy_first(roots, out / "oph_cnb_free_streaming_rows.csv", exported, "oph_cnb_free_streaming_rows.csv")
    _copy_first(roots, out / "h0s8_branch_report.json", exported, "h0s8_branch_report.json")
    _copy_first(roots, out / "h0s8_branch_report.md", exported, "h0s8_branch_report.md")
    _copy_first(roots, out / "h0s8_branch_rows.csv", exported, "h0s8_branch_rows.csv")
    _copy_first(
        roots,
        out / "h0s8_lane8_certificate_report.json",
        exported,
        "h0s8_lane8_certificate_report.json",
    )
    _copy_first(
        roots,
        out / "h0s8_lane8_certificate_report.md",
        exported,
        "h0s8_lane8_certificate_report.md",
    )
    _copy_first(
        roots,
        out / "oph_compressed_likelihood_report.json",
        exported,
        "oph_compressed_likelihood_report.json",
    )
    _copy_first(
        roots,
        out / "oph_compressed_likelihood_report.md",
        exported,
        "oph_compressed_likelihood_report.md",
    )
    _copy_first(roots, out / "oph_compressed_likelihood_rows.csv", exported, "oph_compressed_likelihood_rows.csv")
    _copy_first(
        roots,
        out / "oph_compressed_likelihood_scan_points.csv",
        exported,
        "oph_compressed_likelihood_scan_points.csv",
    )
    _copy_first(roots, out / "comparable_data_rows.csv", exported, "comparable_data_rows.csv")
    _copy_first(roots, out / "scale_compressed_repair_report.json", exported, "scale_compressed_repair_report.json")
    _copy_first(roots, out / "scale_compressed_repair_rounds.csv", exported, "scale_compressed_repair_rounds.csv")
    _copy_first(roots, out / "scale_compressed_h3_objects.csv", exported, "scale_compressed_h3_objects.csv")
    _copy_first(roots, out / "scale_compressed_particles.csv", exported, "scale_compressed_particles.csv")
    _copy_first(roots, out / "scale_compressed_screen_cl.csv", exported, "scale_compressed_screen_cl.csv")
    _copy_first(roots, out / "scale_compressed_cmb_camb_report.json", exported, "scale_compressed_cmb_camb_report.json")
    _copy_first(roots, out / "scale_compressed_cmb_tt_bins.csv", exported, "scale_compressed_cmb_tt_bins.csv")
    _copy_first(roots, out / "scale_compressed_cmb_tt_curves.csv", exported, "scale_compressed_cmb_tt_curves.csv")
    _copy_first(
        roots,
        out / "scale_compressed_repair_viewer.html",
        exported,
        "scale_compressed_repair_viewer.html",
    )
    _copy_first(
        roots,
        out / "scale_compressed_repair_viewer_summary.json",
        exported,
        "scale_compressed_repair_viewer_summary.json",
    )
    _copy_first(roots, out / "oph_receipt_viewer.html", exported, "oph_receipt_viewer.html")
    _copy_first(
        roots,
        out / "oph_realtime_viewer_summary.json",
        exported,
        "oph_realtime_viewer_summary.json",
    )
    _copy_first(roots, out / "object_h3_bulk_viewer.html", exported, "object_h3_bulk_viewer.html")
    _copy_first(
        roots,
        out / "object_h3_bulk_viewer_summary.json",
        exported,
        "object_h3_bulk_viewer_summary.json",
    )
    _copy_first(roots, out / "cmb_neutral_frontier_viewer.html", exported, "cmb_neutral_frontier_viewer.html")
    _copy_first(
        roots,
        out / "cmb_neutral_frontier_viewer_summary.json",
        exported,
        "cmb_neutral_frontier_viewer_summary.json",
    )
    _copy_first(roots, out / "cmb_static_plots_summary.json", exported, "cmb_static_plots_summary.json")
    _copy_first(roots, out / "physical_cmb_tt_comparison.png", exported, "physical_cmb_tt_comparison.png")
    _copy_first(
        roots,
        out / "physical_cmb_best_oph_residuals.png",
        exported,
        "physical_cmb_best_oph_residuals.png",
    )
    _copy_first(
        roots,
        out / "physical_cmb_peak_features.png",
        exported,
        "physical_cmb_peak_features.png",
    )
    _copy_first(
        roots,
        out / "strict_neutral_gate_coincidence.png",
        exported,
        "strict_neutral_gate_coincidence.png",
    )
    _copy_first(
        roots,
        out / "strict_neutral_rank_selector_diagnostics.png",
        exported,
        "strict_neutral_rank_selector_diagnostics.png",
    )
    _copy_first(
        roots,
        out / "strict_neutral_near_miss_frontier.png",
        exported,
        "strict_neutral_near_miss_frontier.png",
    )
    _copy_first(
        roots,
        out / "strict_neutral_near_miss_frontier.csv",
        exported,
        "strict_neutral_near_miss_frontier.csv",
    )
    _copy_first(roots, out / "release_code_certificate.json", exported, "release_code_certificate.json")
    _copy_first(roots, out / "parent_collar_certificate.json", exported, "parent_collar_certificate.json")
    _copy_first(roots, out / "repair_matrix_certificate.json", exported, "repair_matrix_certificate.json")
    _copy_first(roots, out / "boltzmann_export_certificate.json", exported, "boltzmann_export_certificate.json")
    _copy_first(roots, out / "no_data_use_receipt.json", exported, "no_data_use_receipt.json")
    _write_finite_certificate_outputs(roots, out / "finite_certificate_outputs.csv")
    _refresh_bulk_proof_certificate(out, exported)

    claims = _collect_claims([out] + roots)
    (out / "claims.json").write_text(json.dumps(claims, indent=2, default=str), encoding="utf-8")

    report = {
        "mode": "oph_measurement_pack_v0",
        "source_run_dirs": [str(path) for path in roots],
        "out_dir": str(out),
        "files": [],
        "claims": claims,
        "claim_boundary": (
            "Standard export pack for measurement-facing OPH-FPE diagnostics. "
            "It may include physical-data fits and internal mini-universe tables, but claims remain "
            "controlled by claims.json and the source receipt files."
        ),
    }
    (out / "measurement_pack_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "README.md").write_text(_readme(report), encoding="utf-8")
    report["files"] = sorted(path.name for path in out.iterdir() if path.is_file())
    (out / "measurement_pack_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "README.md").write_text(_readme(report), encoding="utf-8")
    return report


def _refresh_bulk_proof_certificate(out: Path, exported: dict[str, str]) -> None:
    try:
        from oph_fpe.bulk.proof_certificate import write_bulk_proof_certificate
    except Exception:
        return
    try:
        write_bulk_proof_certificate(out, out)
    except Exception:
        return
    exported["bulk_proof_certificate_report.json"] = "generated from exported receipt bundle"
    exported["bulk_proof_certificate_report.md"] = "generated from exported receipt bundle"


def _collect_claims(roots: list[Path]) -> dict[str, Any]:
    static_galaxy = _first_json(roots, "static_galaxy_measurement_report.json")
    comparable = _first_json(roots, "comparable_data_snapshot.json")
    bulk = _first_json(roots, "bulk_proof_certificate_report.json")
    exact_cmb = _first_json(roots, "oph_exact_cmb_camb_report.json")
    finite_clock_cmb = _first_json(roots, "finite_repair_clock_cmb_camb_report.json")
    camb_baseline = _first_json(roots, "camb_lcdm_baseline_report.json")
    shape = _first_json(roots, "shape_substrate_summary.json")
    finite = _first_json(roots, "finite_certificate_report.json")
    finite_transition = _first_json(roots, "finite_repair_transition_matrix_report.json")
    scalar_repair_semigroup = _first_json(roots, "scalar_repair_semigroup_report.json")
    screen_capacity = _first_json(roots, "screen_capacity_closure_report.json")
    capacity_proxy = _first_json(roots, "capacity_readback_proxy_report.json")
    repair_scale = _first_json(roots, "repair_scale_closure_report.json")
    pn_resonance = _first_json(roots, "pn_resonance_report.json")
    silence_to_observation = _first_json(roots, "silence_to_observation_report.json")
    kernel_dispatch = _first_json(roots, "kernel_dispatch_report.json")
    positive_geometry_kernel = _first_json(roots, "positive_geometry_kernel_report.json")
    scale_bridge = _first_json(roots, "oph_scale_bridge_report.json")
    no_g_clock_bridge = _first_json(roots, "no_g_clock_bridge_report.json")
    parent_collar = _first_json(roots, "parent_collar_ladder_report.json")
    repair_clock = _first_json(roots, "repair_clock_certificate_report.json")
    boltzmann_inputs = _first_json(roots, "oph_boltzmann_input_report.json")
    finite_collar_boltzmann = _first_json(roots, "finite_collar_boltzmann_bundle_report.json")
    finite_collar_projection = _first_json(roots, "finite_collar_cmb_projection_report.json")
    physical_cmb_input = _first_json(roots, "physical_cmb_input_report.json")
    physical_cmb_promotion = _first_json(roots, "physical_cmb_promotion_audit_report.json")
    physical_cmb_frontier = _first_json(roots, "physical_cmb_frontier_report.json")
    physical_cmb_output = _first_json(roots, "physical_cmb_output_comparison_report.json")
    official_likelihood_readiness = _first_json(roots, "official_planck_likelihood_readiness_report.json")
    neutral_profile = _first_json(roots, "neutral_profile_audit_report.json")
    prime_rank_refinement = _first_json(roots, "prime_geometric_rank_refinement_report.json")
    neutral_3d_bulk_audit = _first_json(roots, "neutral_3d_bulk_audit_report.json")
    overlap_neutral_control = _first_json(roots, "overlap_native_neutral_control_report.json")
    overlap_graph_geometry = _first_json(roots, "overlap_native_graph_geometry_report.json")
    overlap_graph_sweep = _first_json(roots, "overlap_native_graph_geometry_sweep_report.json")
    overlap_graph_rank_obstruction = overlap_graph_sweep.get("rank_obstruction_summary") or {}
    overlap_graph_coincidence = overlap_graph_sweep.get("gate_coincidence_summary") or {}
    overlap_residual_graph = _first_json(roots, "overlap_residualized_graph_geometry_report.json")
    overlap_residual_graph_sweep = _first_json(roots, "overlap_residualized_graph_geometry_sweep_report.json")
    overlap_residual_rank_obstruction = overlap_residual_graph_sweep.get("rank_obstruction_summary") or {}
    overlap_residual_gate_coincidence = (
        overlap_residual_graph_sweep.get("gate_coincidence_summary") or {}
    )
    neutral_rank_selector_audit = _first_json(roots, "neutral_independent_rank_selector_audit_report.json")
    strict_neutral_frontier = _first_json(roots, "strict_neutral_bulk_frontier_report.json")
    observer_modular_experience = _first_json(roots, "observer_modular_experience_report.json")
    ba_parent = _first_json(roots, "b_a_parent_report.json")
    ba_kernel = _first_json(roots, "B_A_kernel_report.json")
    ba_kernel_refinement = _first_json(roots, "B_A_kernel_refinement_report.json")
    screen_power = _first_json(roots, "oph_screen_power_report.json")
    maxent_green = _first_json(roots, "maxent_green_spectrum_report.json")
    selector_elimination = _first_json(roots, "oph_cmb_selector_elimination_report.json")
    cmb_anomaly = _first_json(roots, "cmb_anomaly_report.json")
    neutrinos = _first_json(roots, "oph_cnb_neutrino_report.json")
    h0s8 = _first_json(roots, "h0s8_branch_report.json")
    h0s8_lane8 = _first_json(roots, "h0s8_lane8_certificate_report.json")
    compressed_likelihood = _first_json(roots, "oph_compressed_likelihood_report.json")
    scale_compressed = _first_json(roots, "scale_compressed_repair_report.json")
    scale_compressed_cmb = _first_json(roots, "scale_compressed_cmb_camb_report.json")
    object_h3_viewer = _first_json(roots, "object_h3_bulk_viewer_summary.json")
    cmb_neutral_viewer = _first_json(roots, "cmb_neutral_frontier_viewer_summary.json")
    cmb_static_plots = _first_json(roots, "cmb_static_plots_summary.json")
    comparable_lorentz = comparable.get("measurement_lanes", {}).get("support_visible_lorentz_branch", {})
    comparable_neutral = comparable.get("measurement_lanes", {}).get("neutral_observer_reconstruction", {})
    screen_capacity_gates = screen_capacity.get("readiness_gates") or {}
    capacity_proxy_gates = capacity_proxy.get("readiness_gates") or {}
    repair_scale_gates = repair_scale.get("readiness_gates") or {}
    pn_resonance_gates = pn_resonance.get("readiness_gates") or {}
    silence_gates = silence_to_observation.get("readiness_gates") or {}
    kernel_dispatch_positive_geometry = (kernel_dispatch.get("kernels") or {}).get("positive_geometry") or {}
    positive_geometry_kernel_gates = positive_geometry_kernel.get("readiness_gates") or {}
    scale_bridge_gates = scale_bridge.get("readiness_gates") or {}
    scale_bridge_values = scale_bridge.get("scale_bridge") or {}
    clock_bridge_values = no_g_clock_bridge.get("clock_bridge") or {}
    clock_bridge_gates = no_g_clock_bridge.get("readiness_gates") or {}
    neutrino_gates = neutrinos.get("readiness_gates") or {}
    h0s8_gates = h0s8.get("readiness_gates") or {}
    h0s8_comparisons = h0s8.get("measurement_comparisons") or {}
    cmb_anomaly_aggregate = cmb_anomaly.get("aggregate") or {}
    neutral_profile_rows = neutral_profile.get("profile_rows", [])
    if not isinstance(neutral_profile_rows, list):
        neutral_profile_rows = []
    physical_cmb_input_status = physical_cmb_input.get("input_status") or {}
    physical_cmb_p_source = physical_cmb_input_status.get("P_source") or {}
    physical_cmb_n_source = physical_cmb_input_status.get("N_source") or {}
    physical_cmb_a_zeta = physical_cmb_input_status.get("A_zeta") or {}
    physical_cmb_b_a = physical_cmb_input_status.get("B_A_k_a") or {}
    physical_cmb_rho_a = physical_cmb_input_status.get("rho_A_a") or {}
    bulk_observer_modular_summary = bulk.get("observer_modular_experience_summary") or {}
    observer_modular_experience_blockers = bulk_observer_modular_summary.get("blockers")
    if observer_modular_experience_blockers is None:
        observer_modular_experience_blockers = observer_modular_experience.get("blockers") or []
    observer_modular_experience_source_blockers = bulk_observer_modular_summary.get("source_report_blockers")
    if observer_modular_experience_source_blockers is None:
        observer_modular_experience_source_blockers = observer_modular_experience.get("blockers") or []
    physical_cmb_frontier_gates = {
        str(row.get("gate")): bool(row.get("passed", False))
        for row in (physical_cmb_frontier.get("gate_rows") or [])
        if isinstance(row, dict) and row.get("gate") is not None
    }
    best_oph_cmb_output = physical_cmb_output.get("best_oph_diagnostic_model") or {}
    best_oph_cmb_residuals = physical_cmb_output.get("best_oph_residual_summary") or {}
    best_oph_cmb_peaks = physical_cmb_output.get("best_oph_peak_feature_summary") or {}
    physical_cmb_usable_data = bool(
        physical_cmb_output.get("USABLE_PHYSICAL_CMB_DATA_RECEIPT", False)
        or physical_cmb_output.get("usable_physical_cmb_data_receipt", False)
        or (
            physical_cmb_output.get("PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT", False)
            and best_oph_cmb_output
            and best_oph_cmb_residuals.get("available", False)
            and (
                int(best_oph_cmb_residuals.get("bin_count") or 0) > 0
                or bool(physical_cmb_output.get("best_oph_residual_rows") or [])
            )
        )
    )
    return {
        "WORKING_MINI_UNIVERSE_V0": bool(
            static_galaxy.get("STATIC_GALAXY_RAR_BTFR_RECEIPT", False)
            and (
                comparable
                or bulk.get("chart_level_3p1_lorentz_kinematics_established", False)
                or shape.get("shape_settling_receipt", False)
            )
        ),
        "static_galaxy_measurement_fit": bool(static_galaxy.get("STATIC_GALAXY_RAR_BTFR_RECEIPT", False)),
        "static_galaxy_bridge_receipt": bool(static_galaxy.get("OPH_STATIC_GALAXY_BRIDGE_RECEIPT", False)),
        "static_galaxy_physical_claim": bool(static_galaxy.get("physical_claim", False)),
        "physical_cmb_prediction": bool(
            comparable.get("physical_cmb_prediction", False)
            or bulk.get("physical_cmb_prediction", False)
            or exact_cmb.get("physical_cmb_prediction", False)
        ),
        "exact_cmb_curve_comparable": bool(
            exact_cmb.get("measurement_comparable_curve", False)
            or exact_cmb.get("measurement_comparable_cmb_curve", False)
        ),
        "finite_repair_clock_cmb_curve_comparable": bool(
            finite_clock_cmb.get("measurement_comparable_cmb_curve", False)
        ),
        "finite_repair_clock_cmb_finite_lattice_clock": bool(
            finite_clock_cmb.get("finite_lattice_clock_derived", False)
        ),
        "finite_repair_clock_cmb_physical_prediction": bool(
            finite_clock_cmb.get("physical_cmb_prediction", False)
        ),
        "camb_lcdm_cdm_limit_boltzmann_receipt": bool(
            camb_baseline.get("CDM_LIMIT_BOLTZMANN_RECEIPT", False)
        ),
        "camb_lcdm_oph_anomaly_module_ready": bool(
            camb_baseline.get("oph_anomaly_module_ready", False)
        ),
        "chart_level_3p1": bool(
            comparable.get("chart_level_3p1_any", False)
            or comparable.get("chart_level_3p1_count", 0)
            or comparable_lorentz.get("support_visible_lorentz_3p1_count", 0)
            or comparable_lorentz.get("paper_theorem_3d_bulk_chart_count", 0)
            or bulk.get("chart_level_3p1_lorentz_kinematics_established", False)
        ),
        "theorem_assisted_h3_bulk": bool(
            comparable.get("theorem_assisted_h3_bulk_any", False)
            or comparable.get("theorem_assisted_h3_bulk_count", 0)
            or comparable_lorentz.get("paper_theorem_assisted_h3_populated_chart_count", 0)
            or bulk.get("bulk_3d_established_theorem_assisted", False)
        ),
        "theorem_assisted_observer_facing_h3_population": bool(
            comparable.get("theorem_assisted_h3_bulk_any", False)
            or comparable.get("theorem_assisted_h3_bulk_count", 0)
            or comparable_lorentz.get("paper_theorem_assisted_h3_populated_chart_count", 0)
            or bulk.get("theorem_assisted_observer_facing_h3_population", False)
            or bulk.get("bulk_3d_established_theorem_assisted", False)
        ),
        "observer_facing_3p1d_h3_experience": bool(
            bulk.get("OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT", False)
            or bulk.get("theorem_assisted_observer_facing_h3_population", False)
        ),
        "observer_modular_experience_written": bool(observer_modular_experience),
        "observer_modular_time_receipt": bool(
            observer_modular_experience.get("observer_modular_time_receipt", False)
            or bulk.get("observer_modular_time_receipt", False)
        ),
        "observer_modular_time_observer_count": int(
            observer_modular_experience.get("observer_count") or 0
        ),
        "observer_modular_time_relative_time_count": int(
            observer_modular_experience.get("observer_relative_time_count") or 0
        ),
        "observer_facing_3p1d_h3_experience_receipt": bool(
            observer_modular_experience.get("OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT", False)
            or observer_modular_experience.get("observer_facing_3p1d_h3_experience_receipt", False)
            or bulk.get("OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT", False)
            or bulk.get("observer_facing_3p1d_h3_experience_receipt", False)
        ),
        "observer_modular_experience_blockers": list(
            observer_modular_experience_blockers
        ),
        "observer_modular_experience_source_blockers": list(
            observer_modular_experience_source_blockers
        ),
        "strict_neutral_bulk": bool(
            comparable.get("strict_neutral_3d_bulk_any", False)
            or comparable.get("strict_neutral_3d_bulk_count", 0)
            or bulk.get("strict_neutral_third_person_bulk_established", False)
            or strict_neutral_frontier.get("strict_neutral_bulk", False)
        ),
        "strict_blind_record_transition_3d_candidate": bool(
            comparable_neutral.get("blind_record_transition_rank3_receipt_count", 0)
        ),
        "strict_blind_record_transition_rank3_candidate": bool(
            comparable_neutral.get("blind_record_transition_rank3_receipt_count", 0)
        ),
        "neutral_profile_audit_written": bool(neutral_profile),
        "neutral_profile_strict_3d_ready_count": int(
            sum(1 for row in neutral_profile_rows if isinstance(row, dict) and row.get("strict_3d_ready"))
        ),
        "control_residualized_rank3_refinement_candidate": bool(
            prime_rank_refinement.get("control_quotient_rank3_refinement_candidate_receipt", False)
        ),
        "control_residualized_rank3_independent_selector_all": bool(
            prime_rank_refinement.get("independent_rank3_selector_all", False)
        ),
        "control_residualized_rank3_dimension_drift": prime_rank_refinement.get(
            "candidate_dimension_drift"
        ),
        "strict_neutral_bulk_refinement_receipt": bool(
            prime_rank_refinement.get("strict_neutral_bulk_refinement_receipt", False)
        ),
        "control_residualized_rank3_physical_claim": bool(prime_rank_refinement.get("physical_claim", False)),
        "neutral_3d_bulk_audit_written": bool(neutral_3d_bulk_audit),
        "neutral_3d_bulk_audit_ready": bool(
            neutral_3d_bulk_audit.get("strict_neutral_bulk_ready", False)
        ),
        "neutral_3d_bulk_audit_directional_strict_ready_total": int(
            neutral_3d_bulk_audit.get("directional_strict_ready_total") or 0
        ),
        "neutral_3d_bulk_audit_control_quotient_candidate_count": int(
            neutral_3d_bulk_audit.get("control_quotient_candidate_count") or 0
        ),
        "overlap_native_neutral_control_written": bool(overlap_neutral_control),
        "overlap_native_negative_control_receipt": bool(
            overlap_neutral_control.get("OVERLAP_NATIVE_NEGATIVE_CONTROL_RECEIPT", False)
        ),
        "overlap_native_spatial_3d_candidate": bool(
            overlap_neutral_control.get("overlap_native_spatial_3d_candidate", False)
        ),
        "overlap_native_strict_h3_candidate": bool(
            overlap_neutral_control.get("overlap_native_strict_h3_candidate", False)
        ),
        "overlap_native_graph_geometry_written": bool(overlap_graph_geometry),
        "overlap_native_graph_geometry_receipt": bool(
            overlap_graph_geometry.get("OVERLAP_NATIVE_GRAPH_GEOMETRY_RECEIPT", False)
        ),
        "overlap_native_graph_spatial_3d_candidate": bool(
            overlap_graph_geometry.get("overlap_graph_spatial_3d_candidate", False)
        ),
        "overlap_native_graph_strict_h3_candidate": bool(
            overlap_graph_geometry.get("overlap_graph_strict_h3_candidate", False)
        ),
        "overlap_native_graph_rank3_selector": bool(
            (overlap_graph_geometry.get("rank_selection") or {}).get("rank3_selector_receipt", False)
        ),
        "overlap_native_graph_sweep_written": bool(overlap_graph_sweep),
        "overlap_native_graph_sweep_receipt": bool(
            overlap_graph_sweep.get("OVERLAP_NATIVE_GRAPH_GEOMETRY_SWEEP_RECEIPT", False)
        ),
        "overlap_native_graph_sweep_case_count": int(overlap_graph_sweep.get("case_count") or 0),
        "overlap_native_graph_sweep_receipt_count": int(
            overlap_graph_sweep.get("graph_geometry_receipt_count") or 0
        ),
        "overlap_native_graph_sweep_spatial_candidates": int(
            overlap_graph_sweep.get("spatial_3d_candidate_count") or 0
        ),
        "overlap_native_graph_sweep_strict_h3_candidates": int(
            overlap_graph_sweep.get("strict_h3_candidate_count") or 0
        ),
        "overlap_native_graph_sweep_rank3_selectors": int(
            overlap_graph_sweep.get("rank3_selector_count") or 0
        ),
        "overlap_native_graph_sweep_model_order_rank3_selectors": int(
            overlap_graph_rank_obstruction.get("model_order_rank3_selector_count") or 0
        ),
        "overlap_native_graph_sweep_closest_strict_candidate_count": int(
            len(overlap_graph_sweep.get("closest_strict_rows") or [])
        ),
        "overlap_native_graph_sweep_nontrivial_rank3_selectors": int(
            overlap_graph_rank_obstruction.get("nontrivial_rank3_selector_count") or 0
        ),
        "overlap_native_graph_sweep_nontrivial_model_order_rank3_selectors": int(
            overlap_graph_rank_obstruction.get("nontrivial_model_order_rank3_selector_count") or 0
        ),
        "overlap_native_graph_sweep_spatial_h3_rank3_coincidences": int(
            overlap_graph_coincidence.get("spatial_h3_independent_rank3_selector_count") or 0
        ),
        "overlap_native_graph_sweep_spatial_h3_nontrivial_rank3_coincidences": int(
            overlap_graph_coincidence.get("spatial_h3_nontrivial_rank3_selector_count") or 0
        ),
        "overlap_native_graph_sweep_dominant_largest_gap_rank": (
            overlap_graph_rank_obstruction.get("dominant_largest_gap_rank")
        ),
        "overlap_native_graph_sweep_dominant_nontrivial_largest_gap_rank": (
            overlap_graph_rank_obstruction.get("dominant_nontrivial_largest_gap_rank")
        ),
        "overlap_native_graph_sweep_max_rank3_ev": (
            overlap_graph_rank_obstruction.get("max_rank3_cumulative_explained_variance")
        ),
        "overlap_native_graph_sweep_max_nontrivial_rank3_ev": (
            overlap_graph_rank_obstruction.get("max_nontrivial_rank3_cumulative_explained_variance")
        ),
        "overlap_native_graph_sweep_median_effective_rank": (
            overlap_graph_rank_obstruction.get("median_effective_rank")
        ),
        "overlap_native_graph_sweep_median_nontrivial_effective_rank": (
            overlap_graph_rank_obstruction.get("median_nontrivial_effective_rank")
        ),
        "overlap_native_graph_sweep_spatial_max_rank3_ev": (
            overlap_graph_rank_obstruction.get("spatial_max_rank3_cumulative_explained_variance")
        ),
        "overlap_native_graph_sweep_spatial_max_nontrivial_rank3_ev": (
            overlap_graph_rank_obstruction.get("spatial_max_nontrivial_rank3_cumulative_explained_variance")
        ),
        "overlap_native_graph_sweep_spatial_median_effective_rank": (
            overlap_graph_rank_obstruction.get("spatial_median_effective_rank")
        ),
        "overlap_native_graph_sweep_spatial_median_nontrivial_rank3_ev": (
            overlap_graph_rank_obstruction.get("spatial_median_nontrivial_rank3_cumulative_explained_variance")
        ),
        "overlap_residualized_graph_geometry_written": bool(overlap_residual_graph),
        "overlap_residualized_graph_geometry_receipt": bool(
            overlap_residual_graph.get("OVERLAP_RESIDUALIZED_GRAPH_GEOMETRY_RECEIPT", False)
        ),
        "overlap_residualized_graph_spatial_3d_candidate": bool(
            overlap_residual_graph.get("overlap_residual_graph_spatial_3d_candidate", False)
        ),
        "overlap_residualized_graph_strict_h3_candidate": bool(
            overlap_residual_graph.get("overlap_residual_graph_strict_h3_candidate", False)
        ),
        "overlap_residualized_graph_rank3_selector": bool(
            (overlap_residual_graph.get("rank_selection") or {}).get("rank3_selector_receipt", False)
        ),
        "overlap_residualized_graph_sweep_written": bool(overlap_residual_graph_sweep),
        "overlap_residualized_graph_sweep_receipt": bool(
            overlap_residual_graph_sweep.get("OVERLAP_RESIDUALIZED_GRAPH_GEOMETRY_SWEEP_RECEIPT", False)
        ),
        "overlap_residualized_graph_sweep_case_count": int(
            overlap_residual_graph_sweep.get("case_count") or 0
        ),
        "overlap_residualized_graph_sweep_receipt_count": int(
            overlap_residual_graph_sweep.get("residual_graph_receipt_count") or 0
        ),
        "overlap_residualized_graph_sweep_spatial_candidates": int(
            overlap_residual_graph_sweep.get("spatial_3d_candidate_count") or 0
        ),
        "overlap_residualized_graph_sweep_strict_h3_candidates": int(
            overlap_residual_graph_sweep.get("strict_h3_candidate_count") or 0
        ),
        "overlap_residualized_graph_sweep_rank3_selectors": int(
            overlap_residual_graph_sweep.get("rank3_selector_count") or 0
        ),
        "overlap_residualized_graph_sweep_model_order_rank3_selectors": int(
            overlap_residual_rank_obstruction.get("model_order_rank3_selector_count") or 0
        ),
        "overlap_residualized_graph_sweep_closest_strict_candidate_count": int(
            len(overlap_residual_graph_sweep.get("closest_strict_rows") or [])
        ),
        "overlap_residualized_graph_sweep_nontrivial_rank3_selectors": int(
            overlap_residual_rank_obstruction.get("nontrivial_rank3_selector_count") or 0
        ),
        "overlap_residualized_graph_sweep_nontrivial_model_order_rank3_selectors": int(
            overlap_residual_rank_obstruction.get("nontrivial_model_order_rank3_selector_count") or 0
        ),
        "overlap_residualized_graph_sweep_spatial_h3_rank3_coincidences": int(
            overlap_residual_gate_coincidence.get("spatial_h3_independent_rank3_selector_count") or 0
        ),
        "overlap_residualized_graph_sweep_spatial_h3_nontrivial_rank3_coincidences": int(
            overlap_residual_gate_coincidence.get("spatial_h3_nontrivial_rank3_selector_count") or 0
        ),
        "overlap_residualized_graph_sweep_dominant_largest_gap_rank": (
            overlap_residual_rank_obstruction.get("dominant_largest_gap_rank")
        ),
        "overlap_residualized_graph_sweep_dominant_nontrivial_largest_gap_rank": (
            overlap_residual_rank_obstruction.get("dominant_nontrivial_largest_gap_rank")
        ),
        "overlap_residualized_graph_sweep_raw_rank1_cases": (
            overlap_residual_rank_obstruction.get("raw_largest_gap_rank1_count")
        ),
        "overlap_residualized_graph_sweep_max_rank3_ev": (
            overlap_residual_rank_obstruction.get("max_rank3_cumulative_explained_variance")
        ),
        "overlap_residualized_graph_sweep_max_nontrivial_rank3_ev": (
            overlap_residual_rank_obstruction.get("max_nontrivial_rank3_cumulative_explained_variance")
        ),
        "overlap_residualized_graph_sweep_median_effective_rank": (
            overlap_residual_rank_obstruction.get("median_effective_rank")
        ),
        "overlap_residualized_graph_sweep_median_nontrivial_effective_rank": (
            overlap_residual_rank_obstruction.get("median_nontrivial_effective_rank")
        ),
        "neutral_independent_rank_selector_audit_written": bool(neutral_rank_selector_audit),
        "neutral_independent_rank3_selector_receipt": bool(
            neutral_rank_selector_audit.get("NEUTRAL_INDEPENDENT_RANK3_SELECTOR_RECEIPT", False)
        ),
        "neutral_independent_rank_selector_run_count": int(
            neutral_rank_selector_audit.get("run_count") or 0
        ),
        "neutral_independent_rank_selector_control_rank3_count": int(
            neutral_rank_selector_audit.get("control_quotient_rank3_selector_count") or 0
        ),
        "neutral_independent_rank_selector_control_candidate_count": int(
            neutral_rank_selector_audit.get("control_quotient_rank3_candidate_count") or 0
        ),
        "neutral_independent_rank_selector_control_median_effective_rank": (
            neutral_rank_selector_audit.get("control_quotient_median_effective_rank")
        ),
        "neutral_independent_rank_selector_control_median_rank3_ev": (
            neutral_rank_selector_audit.get("control_quotient_median_rank3_cumulative_explained_variance")
        ),
        "strict_neutral_bulk_frontier_written": bool(strict_neutral_frontier),
        "strict_neutral_bulk_frontier_ready": bool(
            strict_neutral_frontier.get("strict_neutral_bulk_ready", False)
        ),
        "strict_neutral_bulk_frontier_gap_count": int(
            len(strict_neutral_frontier.get("gate_gap_rows") or [])
        ),
        "strict_neutral_bulk_frontier_rank3_candidate": bool(
            strict_neutral_frontier.get("control_residualized_rank3_refinement_candidate", False)
        ),
        "strict_neutral_bulk_frontier_overlap_controls": bool(
            strict_neutral_frontier.get("overlap_native_negative_control_receipt_all", False)
        ),
        "strict_neutral_bulk_frontier_overlap_graph_receipts": int(
            strict_neutral_frontier.get("overlap_native_graph_geometry_receipt_count") or 0
        ),
        "strict_neutral_bulk_frontier_overlap_graph_spatial_candidates": int(
            strict_neutral_frontier.get("overlap_native_graph_spatial_3d_candidate_count") or 0
        ),
        "strict_neutral_bulk_frontier_overlap_graph_strict_h3_candidates": int(
            strict_neutral_frontier.get("overlap_native_graph_strict_h3_candidate_count") or 0
        ),
        "strict_neutral_bulk_frontier_overlap_graph_model_order_rank3_selectors": int(
            strict_neutral_frontier.get("overlap_native_graph_model_order_rank3_selector_count") or 0
        ),
        "strict_neutral_bulk_frontier_overlap_graph_nontrivial_model_order_rank3_selectors": int(
            strict_neutral_frontier.get(
                "overlap_native_graph_nontrivial_model_order_rank3_selector_count"
            )
            or 0
        ),
        "strict_neutral_bulk_frontier_overlap_residual_graph_receipts": int(
            strict_neutral_frontier.get("overlap_residualized_graph_geometry_receipt_count") or 0
        ),
        "strict_neutral_bulk_frontier_overlap_residual_graph_spatial_candidates": int(
            strict_neutral_frontier.get("overlap_residualized_graph_spatial_3d_candidate_count") or 0
        ),
        "strict_neutral_bulk_frontier_overlap_residual_graph_strict_h3_candidates": int(
            strict_neutral_frontier.get("overlap_residualized_graph_strict_h3_candidate_count") or 0
        ),
        "strict_neutral_bulk_frontier_overlap_residual_graph_rank3_selectors": int(
            strict_neutral_frontier.get("overlap_residualized_graph_rank3_selector_count") or 0
        ),
        "strict_neutral_bulk_frontier_overlap_residual_graph_model_order_rank3_selectors": int(
            strict_neutral_frontier.get("overlap_residualized_graph_model_order_rank3_selector_count") or 0
        ),
        "strict_neutral_bulk_frontier_overlap_residual_graph_nontrivial_model_order_rank3_selectors": int(
            strict_neutral_frontier.get(
                "overlap_residualized_graph_nontrivial_model_order_rank3_selector_count"
            )
            or 0
        ),
        "strict_neutral_bulk_frontier_independent_selector": bool(
            strict_neutral_frontier.get("neutral_independent_rank3_selector_receipt", False)
        ),
        "strict_neutral_bulk_frontier_directional_ready_total": int(
            strict_neutral_frontier.get("directional_strict_ready_total") or 0
        ),
        "production_particles": bool(
            bulk.get("production_particle_matter_receipt", False)
            or comparable.get("physical_matter_power_prediction", False)
        ),
        "finite_certificate_compiler_ready": bool(finite.get("finite_certificate_compiler_ready", False)),
        "finite_certificate_stack_ready": bool(finite.get("finite_certificate_stack_ready", False)),
        "finite_certificate_theorem_grade": bool(finite.get("theorem_grade_finite_inputs", False)),
        "finite_certificate_real_physics": bool(finite.get("real_physics_certificate", False)),
        "finite_certificate_no_data_use": bool(
            (finite.get("no_data_use_receipt") or {}).get("no_data_use_receipt", False)
        ),
        "finite_certificate_proxy_A_zeta_available": bool(
            finite.get("proxy_certificate", False)
            and (finite.get("derived_outputs") or {}).get("A_zeta") is not None
        ),
        "finite_transition_matrix_ready": bool(
            finite_transition.get("finite_transition_matrix_ready", False)
        ),
        "finite_transition_clock_certified": bool(
            finite_transition.get("clock_normalization_certified", False)
            or scalar_repair_semigroup.get("repair_clock_certificate", False)
        ),
        "finite_transition_eta_R_finite_lattice_derived": bool(
            finite_transition.get("eta_R_finite_lattice_derived", False)
            or scalar_repair_semigroup.get("eta_R_finite_lattice_derived", False)
        ),
        "screen_capacity_observed_branch_available": bool(
            screen_capacity_gates.get("observed_branch_N_scr_readout_available", False)
        ),
        "screen_capacity_finite_fixed_point_solved": bool(
            screen_capacity_gates.get("N_CRC_fixed_point_solved_from_finite_simulator", False)
        ),
        "capacity_readback_proxy_written": bool(capacity_proxy),
        "capacity_readback_proxy_row_count": int(capacity_proxy.get("row_count") or 0),
        "capacity_readback_proxy_max_observer_count": int(capacity_proxy.get("max_observer_count") or 0),
        "capacity_readback_proxy_max_terminal_proxy_count": int(
            capacity_proxy.get("max_terminal_normal_form_count_proxy") or 0
        ),
        "capacity_readback_proxy_fixed_point_solved": bool(
            capacity_proxy_gates.get("N_CRC_fixed_point_solved_from_finite_simulator", False)
        ),
        "capacity_readback_proxy_F_N_implemented": bool(
            capacity_proxy_gates.get("F_N_readback_map_implemented", False)
        ),
        "pn_resonance_written": bool(pn_resonance),
        "pn_resonance_branch_status": pn_resonance.get("branch_status"),
        "pn_resonance_numeric_replay": bool(pn_resonance.get("PN_RESONANCE_NUMERIC_REPLAY", False)),
        "pn_resonance_receipt": bool(pn_resonance.get("PN_RESONANCE_RECEIPT", False)),
        "pn_resonance_theorem_grade": bool(
            pn_resonance_gates.get("theorem_grade_pn_resonance", False)
        ),
        "pn_resonance_scale_compressed_replay_eligible": bool(
            pn_resonance_gates.get("scale_compressed_pn_resonance_replay_eligible", False)
        ),
        "pn_silence_to_observation_written": bool(silence_to_observation),
        "pn_silence_to_observation_scale_compressed_receipt": bool(
            silence_to_observation.get("scale_compressed_pn_silence_to_observation_receipt", False)
        ),
        "pn_silence_to_observation_literal_global_N": bool(
            silence_to_observation.get("literal_global_N_capacity_simulated_receipt", False)
        ),
        "pn_silence_to_observation_dynamic_detuning_controls": bool(
            silence_to_observation.get("dynamic_p_detuning_control_receipt", False)
        ),
        "pn_silence_to_observation_initial_silence": bool(
            silence_gates.get("initial_record_silence", False)
        ),
        "pn_silence_to_observation_observer_records_emerged": bool(
            silence_gates.get("observation_records_emerged", False)
        ),
        "pn_silence_to_observation_h3_object_emergence": bool(
            silence_gates.get("h3_object_emergence", False)
        ),
        "pn_silence_to_observation_N_eff": (
            (silence_to_observation.get("finite_regulator_depth") or {}).get(
                "regulator_entropy_capacity_N_eff"
            )
        ),
        "kernel_dispatch_written": bool(kernel_dispatch),
        "kernel_dispatch_routing_decision": kernel_dispatch.get("routing_decision"),
        "kernel_dispatch_generic_repair_executed": bool(kernel_dispatch.get("generic_repair_executed", False)),
        "kernel_dispatch_effective_acceleration_enabled": bool(
            kernel_dispatch.get("effective_acceleration_enabled", False)
        ),
        "kernel_dispatch_physical_observables_changed": bool(
            kernel_dispatch.get("physical_observables_changed", False)
        ),
        "kernel_dispatch_positive_geometry_status": kernel_dispatch_positive_geometry.get("dispatch_status"),
        "positive_geometry_kernel_written": bool(positive_geometry_kernel),
        "positive_geometry_kernel_verdict": (
            (positive_geometry_kernel.get("receipt") or {}).get("verdict")
        ),
        "positive_geometry_kernel_execution_mode": positive_geometry_kernel.get("execution_mode"),
        "positive_geometry_kernel_geometry_certified": bool(
            positive_geometry_kernel_gates.get("PGK_geometry_certified", False)
        ),
        "positive_geometry_kernel_acceleration_enabled": bool(
            positive_geometry_kernel_gates.get("PGK_acceleration_enabled", False)
        ),
        "positive_geometry_kernel_fallback_required": bool(
            positive_geometry_kernel_gates.get("generic_oph_repair_fallback_required", False)
        ),
        "scale_bridge_written": bool(scale_bridge),
        "scale_bridge_independent_supplied": bool(
            scale_bridge_gates.get("independent_scale_bridge_supplied", False)
        ),
        "scale_bridge_dimensionful_G_eligible": bool(
            scale_bridge_gates.get("dimensionful_G_SI_eligible", False)
        ),
        "scale_bridge_finite_simulator_derived_G_SI": bool(
            scale_bridge_gates.get("finite_simulator_derived_G_SI", False)
        ),
        "scale_bridge_B_ell_m2_inverse": scale_bridge_values.get("B_ell_m2_inverse"),
        "scale_bridge_G_SI": scale_bridge_values.get("G_SI"),
        "no_g_clock_bridge_written": bool(no_g_clock_bridge),
        "no_g_clock_bridge_receipt": bool(no_g_clock_bridge.get("NO_G_CLOCK_BRIDGE_RECEIPT", False)),
        "no_g_clock_bridge_source_predictive_G_SI": bool(
            no_g_clock_bridge.get("source_predictive_G_SI", False)
        ),
        "no_g_clock_bridge_dimensionful_G_eligible": bool(
            no_g_clock_bridge.get("dimensionful_G_SI_eligible", False)
        ),
        "no_g_clock_bridge_forbidden_dependency_path_count": int(
            clock_bridge_gates.get("forbidden_dependency_path_count") or 0
        ),
        "no_g_clock_bridge_G_SI_checksum": clock_bridge_values.get("G_SI"),
        "parent_collar_local_density_receipt": bool(
            parent_collar.get("local_recovery_density_receipt", False)
        ),
        "parent_collar_theorem_grade": bool(
            parent_collar.get("theorem_grade_parent_collar_ladder", False)
        ),
        "repair_clock_certificate": bool(repair_clock.get("repair_clock_certificate", False)),
        "repair_clock_eta_R_finite_lattice_derived": bool(
            repair_clock.get("eta_R_finite_lattice_derived", False)
        ),
        "boltzmann_input_table_written": bool(boltzmann_inputs),
        "boltzmann_input_physical_prediction": bool(
            boltzmann_inputs.get("physical_cmb_prediction", False)
            or boltzmann_inputs.get("physical_matter_power_prediction", False)
        ),
        "boltzmann_finite_repair_clock_rows_emitted": bool(
            ((boltzmann_inputs.get("readiness") or {}).get("checks") or {}).get(
                "finite_repair_clock_diagnostic_rows_emitted", False
            )
        ),
        "finite_collar_boltzmann_source_bundle": bool(
            finite_collar_boltzmann.get("FINITE_COLLAR_BOLTZMANN_SOURCE_BUNDLE_RECEIPT", False)
        ),
        "finite_collar_boltzmann_physical_certificate": bool(
            finite_collar_boltzmann.get("PHYSICAL_BOLTZMANN_EXPORT_CERTIFICATE", False)
        ),
        "finite_collar_boltzmann_physical_prediction": bool(
            finite_collar_boltzmann.get("physical_cmb_prediction", False)
            or finite_collar_boltzmann.get("physical_matter_power_prediction", False)
        ),
        "finite_collar_cmb_projection": bool(
            finite_collar_projection.get("FINITE_COLLAR_CMB_PROJECTION_DIAGNOSTIC_RECEIPT", False)
        ),
        "finite_collar_cmb_projection_physical_k": bool(
            finite_collar_projection.get("PHYSICAL_K_CALIBRATION_RECEIPT", False)
        ),
        "physical_cmb_input_contract_receipt": bool(
            physical_cmb_input.get("PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT", False)
        ),
        "physical_cmb_input_prediction_eligible": bool(
            physical_cmb_input.get("physical_cmb_prediction_eligible", False)
        ),
        "physical_cmb_input_P_source": physical_cmb_p_source.get("source"),
        "physical_cmb_input_P_source_theorem_side": bool(
            physical_cmb_p_source.get("source_is_theorem_side_constant", False)
        ),
        "physical_cmb_input_N_source": physical_cmb_n_source.get("source"),
        "physical_cmb_input_N_source_theorem_side": bool(
            physical_cmb_n_source.get("source_is_theorem_side_constant", False)
        ),
        "physical_cmb_input_A_zeta_diagnostic_present": bool(
            physical_cmb_a_zeta.get("diagnostic_value_present", False)
        ),
        "physical_cmb_input_A_zeta_physical_gate_passed": bool(
            physical_cmb_a_zeta.get("physical_gate_passed", False)
        ),
        "physical_cmb_input_B_A_diagnostic_rows": int(physical_cmb_b_a.get("row_count") or 0),
        "physical_cmb_input_B_A_physical_gate_passed": bool(
            physical_cmb_b_a.get("physical_gate_passed", False)
        ),
        "physical_cmb_input_rho_A_diagnostic_rows": int(physical_cmb_rho_a.get("row_count") or 0),
        "physical_cmb_input_rho_A_physical_gate_passed": bool(
            physical_cmb_rho_a.get("physical_gate_passed", False)
        ),
        "physical_cmb_promotion_audit_written": bool(physical_cmb_promotion),
        "physical_cmb_promotion_ready": bool(
            physical_cmb_promotion.get("physical_cmb_promotion_ready", False)
        ),
        "physical_cmb_promotion_official_likelihood_ready": bool(
            physical_cmb_promotion.get("official_likelihood_ready", False)
        ),
        "physical_cmb_promotion_blocker_count": int(
            len(physical_cmb_promotion.get("promotion_blockers") or [])
        ),
        "physical_cmb_frontier_written": bool(physical_cmb_frontier),
        "physical_cmb_frontier_ready": bool(
            physical_cmb_frontier.get("physical_cmb_prediction_ready", False)
        ),
        "physical_cmb_frontier_gate_count": int(len(physical_cmb_frontier.get("gate_rows") or [])),
        "physical_cmb_frontier_gap_count": int(len(physical_cmb_frontier.get("gate_gap_rows") or [])),
        "physical_cmb_frontier_blocker_count": int(len(physical_cmb_frontier.get("blockers") or [])),
        "physical_cmb_frontier_measurement_outputs": bool(
            physical_cmb_frontier_gates.get("measurement_comparable_cmb_outputs", False)
        ),
        "physical_cmb_frontier_finite_A_zeta": bool(
            physical_cmb_frontier_gates.get("finite_theorem_A_zeta", False)
        ),
        "physical_cmb_frontier_finite_B_A": bool(
            physical_cmb_frontier_gates.get("finite_B_A_kernel", False)
        ),
        "physical_cmb_frontier_finite_rho_A": bool(
            physical_cmb_frontier_gates.get("finite_rho_A", False)
        ),
        "physical_cmb_frontier_official_likelihood": bool(
            physical_cmb_frontier_gates.get("official_planck_likelihood_ready", False)
        ),
        "official_planck_likelihood_readiness_written": bool(official_likelihood_readiness),
        "official_planck_likelihood_execution_ready": bool(
            official_likelihood_readiness.get("official_likelihood_execution_ready", False)
        ),
        "official_planck_likelihood_data_paths_configured": bool(
            official_likelihood_readiness.get("official_planck_likelihood_data_paths_configured", False)
        ),
        "official_planck_clik_api_available": bool(
            official_likelihood_readiness.get("official_clik_api_available", False)
        ),
        "official_planck_likelihood_blocker_count": int(
            len(official_likelihood_readiness.get("blockers") or [])
        ),
        "physical_cmb_output_comparison_written": bool(physical_cmb_output),
        "physical_cmb_output_comparison_receipt": bool(
            physical_cmb_output.get("PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT", False)
        ),
        "physical_cmb_output_usable_data_receipt": bool(
            physical_cmb_usable_data
        ),
        "physical_cmb_output_prediction_receipt": bool(
            physical_cmb_output.get("PHYSICAL_CMB_PREDICTION_RECEIPT", False)
        ),
        "physical_cmb_output_measurement_comparable_model_count": int(
            physical_cmb_output.get("measurement_comparable_model_count") or 0
        ),
        "physical_cmb_output_oph_diagnostic_model_count": int(
            physical_cmb_output.get("oph_diagnostic_model_count") or 0
        ),
        "physical_cmb_output_best_oph_model": best_oph_cmb_output.get("model_id"),
        "physical_cmb_output_best_oph_chi2_per_bin": best_oph_cmb_output.get(
            "amplitude_fit_chi2_per_bin"
        ),
        "physical_cmb_output_best_oph_residual_bin_count": int(
            best_oph_cmb_residuals.get("bin_count") or 0
        ),
        "physical_cmb_output_best_oph_rms_sigma_residual": (
            best_oph_cmb_residuals.get("rms_sigma_residual")
        ),
        "physical_cmb_output_best_oph_max_abs_sigma_residual": (
            best_oph_cmb_residuals.get("max_abs_sigma_residual")
        ),
        "physical_cmb_output_best_oph_max_abs_sigma_ell": (
            best_oph_cmb_residuals.get("max_abs_sigma_ell")
        ),
        "physical_cmb_output_best_oph_peak_count": int(best_oph_cmb_peaks.get("peak_count") or 0),
        "physical_cmb_output_best_oph_mean_abs_peak_ell_delta": (
            best_oph_cmb_peaks.get("mean_abs_peak_ell_delta")
        ),
        "physical_cmb_output_best_oph_max_abs_peak_ell_delta": (
            best_oph_cmb_peaks.get("max_abs_peak_ell_delta")
        ),
        "physical_cmb_output_best_oph_mean_abs_peak_height_fractional_delta": (
            best_oph_cmb_peaks.get("mean_abs_peak_height_fractional_delta")
        ),
        "physical_cmb_output_best_oph_max_abs_peak_height_fractional_delta": (
            best_oph_cmb_peaks.get("max_abs_peak_height_fractional_delta")
        ),
        "b_a_parent_rows_emitted": bool(
            ((ba_parent.get("readiness") or {}).get("checks") or {}).get("finite_difference_rows_emitted", False)
        ),
        "b_a_parent_observer_view_variation": bool(
            ((ba_parent.get("readiness") or {}).get("checks") or {}).get("finite_observer_view_parent_variation", False)
        ),
        "b_a_parent_receipt": bool(ba_parent.get("B_A_PARENT_RECEIPT", False)),
        "b_a_parent_physical_prediction": bool(
            ba_parent.get("physical_prediction_ready", False)
            or ba_parent.get("physical_cmb_prediction", False)
            or ba_parent.get("physical_matter_power_prediction", False)
        ),
        "B_A_kernel_candidate_receipt": bool(ba_kernel.get("B_A_KERNEL_CANDIDATE_RECEIPT", False)),
        "B_A_kernel_physical_receipt": bool(ba_kernel.get("B_A_KERNEL_RECEIPT", False)),
        "B_A_kernel_row_count": int(ba_kernel.get("row_count") or 0),
        "B_A_kernel_promotion_blocker_count": int(len(ba_kernel.get("promotion_blockers") or [])),
        "B_A_kernel_refinement_two_scale_diagnostic": bool(
            ba_kernel_refinement.get("two_scale_diagnostic_receipt", False)
        ),
        "B_A_kernel_refinement_convergence_receipt": bool(
            ba_kernel_refinement.get("B_A_KERNEL_REFINEMENT_CONVERGENCE_RECEIPT", False)
        ),
        "B_A_kernel_refinement_patch_count_count": int(
            ba_kernel_refinement.get("patch_count_count") or 0
        ),
        "B_A_kernel_refinement_key_pair_row_count": int(
            ba_kernel_refinement.get("key_pair_row_count") or 0
        ),
        "B_A_kernel_refinement_key_pair_stable_fraction": (
            ba_kernel_refinement.get("key_pair_stable_fraction")
        ),
        "B_A_kernel_refinement_blocker_count": int(len(ba_kernel_refinement.get("blockers") or [])),
        "screen_power_simulator_primordial_ready": bool(
            screen_power.get("simulator_primordial_reference_ready", False)
        ),
        "maxent_green_source_receipt": bool(maxent_green.get("MAXENT_GREEN_SOURCE_RECEIPT", False)),
        "selector_elimination_theorem_side_receipt": bool(
            selector_elimination.get("THEOREM_SIDE_SELECTOR_ELIMINATION_RECEIPT", False)
        ),
        "cmb_anomaly_parity_asymmetry_proxy": bool(
            cmb_anomaly_aggregate.get("parity_more_asymmetric_than_controls_count", 0)
        ),
        "cmb_anomaly_low_power_proxy": bool(
            cmb_anomaly_aggregate.get("low_power_suppressed_vs_controls_count", 0)
        ),
        "cmb_anomaly_planck_tilt_proxy": bool(
            cmb_anomaly_aggregate.get("planck_tilt_compatible_proxy_count", 0)
        ),
        "neutrino_measurement_comparable": bool(
            neutrinos.get("measurement_comparable_now", False)
            or neutrino_gates.get("measurement_comparable_relic_background", False)
        ),
        "neutrino_finite_lattice_derived": bool(neutrinos.get("finite_lattice_derived", False)),
        "h0s8_measurement_comparable": bool(h0s8_comparisons),
        "h0s8_physical_prediction_ready": bool(h0s8.get("physical_prediction_ready", False)),
        "h0s8_lane8_values_run_derived": bool(h0s8_lane8.get("values_are_run_derived", False)),
        "h0s8_finite_kernel_gates_closed": bool(
            h0s8_gates.get("Q_A_gate", False)
            and h0s8_gates.get("B_A_gate", False)
            and h0s8_gates.get("Gamma_J_gate", False)
        ),
        "compressed_likelihood_reference": bool(compressed_likelihood),
        "repair_scale_closure_numeric_match": bool(
            repair_scale_gates.get("scale_closure_numeric_match_within_1_percent", False)
        ),
        "repair_scale_24_rounds_derived": bool(
            repair_scale_gates.get("twenty_four_round_hypothesis_derived_from_finite_selector", False)
        ),
        "repair_scale_finite_eta_R": bool(
            repair_scale_gates.get("finite_lattice_derived_eta_R", False)
        ),
        "scale_compressed_operator_receipt": bool(
            scale_compressed.get("scale_compressed_operator_receipt", False)
        ),
        "scale_compressed_populated_h3_preview": bool(
            ((scale_compressed.get("h3_preview") or {}).get("populated_h3_preview_receipt", False))
        ),
        "scale_compressed_physical_cmb_prediction": bool(
            scale_compressed.get("physical_cmb_prediction", False)
        ),
        "scale_compressed_cmb_curve_comparable": bool(
            scale_compressed_cmb.get("measurement_comparable_cmb_curve", False)
        ),
        "scale_compressed_cmb_physical_prediction": bool(
            scale_compressed_cmb.get("physical_cmb_prediction", False)
        ),
        "object_h3_bulk_viewer_written": bool(object_h3_viewer),
        "object_h3_bulk_viewer_object_count": int(object_h3_viewer.get("object_count") or 0),
        "object_h3_bulk_viewer_observer_overlap_link_count": int(
            object_h3_viewer.get("observer_overlap_link_count") or 0
        ),
        "object_h3_bulk_viewer_theorem_assisted": bool(
            object_h3_viewer.get("theorem_assisted_h3_bulk", False)
        ),
        "object_h3_bulk_viewer_strict_neutral": bool(object_h3_viewer.get("strict_neutral_bulk", False)),
        "cmb_neutral_frontier_viewer_written": bool(cmb_neutral_viewer),
        "cmb_neutral_frontier_viewer_tt_bin_count": int(cmb_neutral_viewer.get("tt_bin_count") or 0),
        "cmb_static_plots_written": bool(cmb_static_plots),
        "cmb_static_plots_best_oph_model": cmb_static_plots.get("best_oph_model"),
        "cmb_static_plots_best_oph_chi2_per_bin": cmb_static_plots.get("best_oph_chi2_per_bin"),
        "cmb_static_plots_file_count": int(len(cmb_static_plots.get("files") or [])),
        "cmb_static_plots_strict_neutral_near_miss_count": int(
            cmb_static_plots.get("strict_neutral_near_miss_count") or 0
        ),
        "cmb_static_plots_strict_neutral_best_near_miss_gate_score": (
            cmb_static_plots.get("strict_neutral_best_near_miss_gate_score")
        ),
        "cmb_static_plots_strict_neutral_best_near_miss_dimension_error": (
            cmb_static_plots.get("strict_neutral_best_near_miss_dimension_error")
        ),
        "cmb_static_plots_strict_neutral_best_near_miss_nontrivial_rank3_ev": (
            cmb_static_plots.get("strict_neutral_best_near_miss_nontrivial_rank3_ev")
        ),
        "claim_boundary": (
            "Measurement-pack claim flags are copied from source receipts. Static galaxy fits can be "
            "measurement-facing without being CMB or bulk proofs. CMB and strict neutral bulk remain "
            "false unless their source gates pass."
        ),
    }


def _copy_preferred_report_pair(
    roots: list[Path],
    json_target: Path,
    markdown_target: Path,
    exported: dict[str, str],
    json_name: str,
    markdown_name: str,
) -> None:
    json_path = _find_preferred_json(roots, (json_name,))
    if json_path is None:
        _write_missing_placeholder(json_target)
        _write_missing_placeholder(markdown_target)
        return

    shutil.copy2(json_path, json_target)
    exported[json_target.name] = str(json_path)

    markdown_path = json_path.with_name(markdown_name)
    if markdown_path.exists():
        shutil.copy2(markdown_path, markdown_target)
        exported[markdown_target.name] = str(markdown_path)
    else:
        _write_missing_placeholder(markdown_target)


def _copy_first(roots: list[Path], target: Path, exported: dict[str, str], *names: str) -> None:
    if target.suffix.lower() == ".json":
        path = _find_preferred_json(roots, names)
        if path is not None:
            shutil.copy2(path, target)
            exported[target.name] = str(path)
            return
    else:
        for name in names:
            path = _find_first(roots, name)
            if path is not None:
                shutil.copy2(path, target)
                exported[target.name] = str(path)
                return
    _write_missing_placeholder(target)


def _copy_path(roots: list[Path], target: Path, exported: dict[str, str], *relative_paths: str) -> None:
    for relative_path in relative_paths:
        for root in roots:
            path = Path(root) / relative_path
            if path.exists():
                shutil.copy2(path, target)
                exported[target.name] = str(path)
                return
    _write_missing_placeholder(target)


def _write_cmb_screen_cl(roots: list[Path], target: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in _find_all(roots, "cl_proxy.csv"):
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row = dict(row)
                row["source_run"] = path.parent.name
                rows.append(row)
    if rows:
        fieldnames = ["source_run"] + [name for name in rows[0] if name != "source_run"]
        _write_rows(target, rows, fieldnames)
        return
    copied = _find_first(roots, "cl_comparison_rows.csv") or _find_first(roots, "cmb_anomaly_rows.csv")
    if copied is not None:
        shutil.copy2(copied, target)
        return
    rows = _cl_rows_from_reports(roots)
    if rows:
        fieldnames = ["source_run", "field", "ell", "C_ell", "D_ell"]
        _write_rows(target, rows, fieldnames)
        return
    exact = _find_first(roots, "oph_exact_cmb_tt_bins.csv")
    if exact is not None:
        shutil.copy2(exact, target)
        return
    _write_missing_placeholder(target)


def _write_h3_objects(roots: list[Path], target: Path) -> None:
    report = _first_json(roots, "observer_chart_object_h3_report.json")
    rows = report.get("sample_objects") or report.get("objects") or []
    fieldnames = sorted({key for row in rows if isinstance(row, dict) for key in row}) or [
        "object_id",
        "h3_compactness",
        "s2_boundary_compactness",
    ]
    _write_rows(target, rows if isinstance(rows, list) else [], fieldnames)


def _write_h3_defects(roots: list[Path], target: Path) -> None:
    report = _first_json(roots, "defect_cluster_h3_report.json")
    rows = report.get("clusters") or report.get("sample_clusters") or []
    if not rows and report:
        rows = [
            {
                "defect_cluster_h3_support_receipt": report.get("defect_cluster_h3_support_receipt"),
                "median_residual": report.get("median_residual"),
                "cluster_count": report.get("cluster_count"),
            }
        ]
    fieldnames = sorted({key for row in rows if isinstance(row, dict) for key in row}) or [
        "cluster_id",
        "median_residual",
    ]
    _write_rows(target, rows if isinstance(rows, list) else [], fieldnames)


def _write_finite_certificate_outputs(roots: list[Path], target: Path) -> None:
    report = _first_json(roots, "finite_certificate_report.json")
    outputs = report.get("derived_outputs") if isinstance(report.get("derived_outputs"), dict) else {}
    row = {
        "finite_certificate_compiler_ready": report.get("finite_certificate_compiler_ready"),
        "finite_certificate_stack_ready": report.get("finite_certificate_stack_ready"),
        "theorem_grade_finite_inputs": report.get("theorem_grade_finite_inputs"),
        "proxy_certificate": report.get("proxy_certificate"),
        "real_physics_certificate": report.get("real_physics_certificate"),
        "physical_cmb_prediction": report.get("physical_cmb_prediction"),
        "physical_matter_power_prediction": report.get("physical_matter_power_prediction"),
        "no_data_use_receipt": (report.get("no_data_use_receipt") or {}).get("no_data_use_receipt")
        if isinstance(report.get("no_data_use_receipt"), dict)
        else None,
        "epsilon_star_bits": outputs.get("epsilon_star_bits"),
        "kappa_rel": outputs.get("kappa_rel"),
        "N_rel": outputs.get("N_rel"),
        "A_zeta": outputs.get("A_zeta"),
        "Q_A": outputs.get("Q_A"),
        "B_A_first": _first_numeric(outputs.get("B_A")),
        "Gamma_rec": outputs.get("Gamma_rec"),
        "n_s": outputs.get("n_s"),
    }
    _write_rows(target, [row] if report else [], list(row))


def _write_rows(target: Path, rows: list[Any], fieldnames: list[str]) -> None:
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            if isinstance(row, dict):
                writer.writerow({key: _cell(value) for key, value in row.items()})


def _cl_rows_from_reports(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _find_all(roots, "cl_comparison_report.json"):
        report = _read_json(path)
        fields = report.get("fields") if isinstance(report.get("fields"), dict) else {}
        for field_name, field_report in fields.items():
            if not isinstance(field_report, dict):
                continue
            spectrum = field_report.get("spectrum")
            if not isinstance(spectrum, list):
                continue
            for point in spectrum:
                if not isinstance(point, dict):
                    continue
                rows.append(
                    {
                        "source_run": path.parent.name,
                        "field": field_name,
                        "ell": point.get("ell"),
                        "C_ell": point.get("C_ell"),
                        "D_ell": point.get("D_ell"),
                    }
                )
    return rows


def _write_missing_placeholder(target: Path) -> None:
    suffix = target.suffix.lower()
    if suffix == ".json":
        target.write_text("{}\n", encoding="utf-8")
        return
    if suffix in {".md", ".txt", ".html"}:
        target.write_text("", encoding="utf-8")
        return
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write("")


def _cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _first_numeric(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (int, float)):
                return item
    return None


def _first_json(roots: list[Path], name: str) -> dict[str, Any]:
    path = _find_preferred_json(roots, (name,))
    if path is None:
        return {}
    return _read_json(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _find_first(roots: list[Path], name: str) -> Path | None:
    paths = _candidate_paths(roots, name)
    return paths[0] if paths else None


def _find_preferred_json(roots: list[Path], names: tuple[str, ...]) -> Path | None:
    candidates: list[tuple[tuple[float, ...], int, Path]] = []
    for name in names:
        for index, path in enumerate(_candidate_paths(roots, name)):
            data = _read_json(path)
            candidates.append((_json_preference_score(name, data), -index, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _candidate_paths(roots: list[Path], name: str) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        root = Path(root)
        if root.is_file() and root.name == name:
            resolved = root.resolve()
            if resolved not in seen:
                candidates.append(root)
                seen.add(resolved)
            continue
        direct = root / name
        if direct.exists():
            resolved = direct.resolve()
            if resolved not in seen:
                candidates.append(direct)
                seen.add(resolved)
        if root.exists() and root.is_dir():
            for match in sorted(root.glob(f"**/{name}")):
                resolved = match.resolve()
                if resolved not in seen:
                    candidates.append(match)
                    seen.add(resolved)
    return candidates


def _json_preference_score(name: str, data: dict[str, Any]) -> tuple[float, ...]:
    if not data:
        return (0.0,)
    if name == "finite_collar_boltzmann_bundle_report.json":
        validation = data.get("physical_cmb_input_validation") or {}
        readiness = data.get("readiness") or {}
        checks = readiness.get("checks") or {}
        source_summary = data.get("contract_source_summary") or {}
        present_sources = sum(1 for row in source_summary.values() if isinstance(row, dict) and row.get("present"))
        blockers = validation.get("blockers") or []
        return (
            100.0,
            float(bool(data.get("PHYSICAL_BOLTZMANN_EXPORT_CERTIFICATE", False))),
            float(bool(data.get("FINITE_COLLAR_BOLTZMANN_SOURCE_BUNDLE_RECEIPT", False))),
            float(bool(checks.get("no_data_use_receipt", False))),
            float(present_sources),
            -float(len(blockers)),
        )
    if name == "finite_collar_cmb_projection_report.json":
        rows = data.get("projected_B_A_rows") or []
        background = data.get("background_rows") or []
        readiness = data.get("readiness") or {}
        return (
            100.0,
            float(bool(data.get("PHYSICAL_K_CALIBRATION_RECEIPT", False))),
            float(bool(data.get("FINITE_COLLAR_CMB_PROJECTION_DIAGNOSTIC_RECEIPT", False))),
            float(bool(readiness.get("finite_collar_source_bundle_receipt", False))),
            float(len(rows)),
            float(len(background)),
        )
    if name == "physical_cmb_input_report.json":
        blockers = data.get("blockers") or []
        source_summary = data.get("source_summary") or {}
        present_sources = sum(1 for row in source_summary.values() if isinstance(row, dict) and row.get("present"))
        input_status = data.get("input_status") or {}
        p_status = input_status.get("P_source") or {}
        n_status = input_status.get("N_source") or {}
        return (
            100.0,
            float(bool(data.get("PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT", False))),
            float(bool(data.get("physical_cmb_prediction_eligible", False))),
            float(present_sources),
            float(bool(p_status.get("source"))),
            float(bool(n_status.get("source"))),
            -float(len(blockers)),
        )
    if name == "physical_cmb_promotion_audit_report.json":
        blockers = data.get("promotion_blockers") or []
        return (
            100.0,
            float(bool(data.get("physical_cmb_promotion_ready", False))),
            float(bool(data.get("physical_cmb_input_contract_receipt", False))),
            float(bool(data.get("cdm_limit_regression_passed", False))),
            -float(len(blockers)),
        )
    if name == "physical_cmb_output_comparison_report.json":
        blockers = (data.get("promotion_blockers") or []) + (data.get("contract_blockers") or [])
        residual_summary = data.get("best_oph_residual_summary") or {}
        residual_rows = data.get("best_oph_residual_rows") or []
        residual_bin_count = residual_summary.get("bin_count") or len(residual_rows)
        peak_summary = data.get("best_oph_peak_feature_summary") or {}
        peak_rows = data.get("peak_feature_rows") or []
        peak_count = peak_summary.get("peak_count") or len(peak_rows)
        return (
            100.0,
            float(bool(data.get("PHYSICAL_CMB_PREDICTION_RECEIPT", False))),
            float(bool(data.get("PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT", False))),
            float(bool(residual_summary.get("available", False))),
            float(int(residual_bin_count or 0)),
            float(bool(peak_summary.get("available", False))),
            float(int(peak_count or 0)),
            float(int(data.get("oph_diagnostic_model_count") or 0)),
            float(int(data.get("measurement_comparable_model_count") or 0)),
            -float(len(blockers)),
        )
    if name == "physical_cmb_frontier_report.json":
        blockers = data.get("blockers") or []
        gate_rows = data.get("gate_rows") or []
        gap_rows = data.get("gate_gap_rows") or []
        passed = sum(1 for row in gate_rows if isinstance(row, dict) and row.get("passed"))
        return (
            100.0,
            float(bool(data.get("physical_cmb_prediction_ready", False))),
            float(bool(data.get("physical_cmb_output_comparison_receipt", False))),
            float(bool(data.get("physical_cmb_input_contract_receipt", False))),
            float(bool(data.get("official_likelihood_ready", False))),
            float(bool(gap_rows)),
            float(passed),
            -float(len(blockers)),
        )
    if name == "official_planck_likelihood_readiness_report.json":
        blockers = data.get("blockers") or []
        return (
            100.0,
            float(bool(data.get("official_likelihood_execution_ready", False))),
            float(bool(data.get("official_planck_likelihood_data_paths_configured", False))),
            float(bool(data.get("official_clik_api_available", False))),
            float(bool(data.get("camb_available", False))),
            -float(len(blockers)),
        )
    if name == "neutral_3d_bulk_audit_report.json":
        blockers = data.get("blockers") or []
        return (
            100.0,
            float(bool(data.get("strict_neutral_bulk_ready", False))),
            float(bool(data.get("control_residualized_rank3_refinement_candidate", False))),
            float(int(data.get("overlap_native_negative_control_receipt_count") or 0)),
            float(int(data.get("overlap_native_negative_control_report_count") or 0)),
            float(int(data.get("sweep_report_count") or 0)),
            -float(len(blockers)),
        )
    if name == "overlap_native_neutral_control_report.json":
        blockers = data.get("blockers") or []
        return (
            100.0,
            float(bool(data.get("OVERLAP_NATIVE_NEGATIVE_CONTROL_RECEIPT", False))),
            float(bool(data.get("overlap_native_spatial_3d_candidate", False))),
            float(bool(data.get("overlap_native_strict_h3_candidate", False))),
            -float(len(blockers)),
        )
    if name == "overlap_native_graph_geometry_report.json":
        blockers = data.get("blockers") or []
        rank = data.get("rank_selection") or {}
        graph = data.get("graph_summary") or {}
        return (
            100.0,
            float(bool(data.get("OVERLAP_NATIVE_GRAPH_GEOMETRY_RECEIPT", False))),
            float(bool(data.get("overlap_graph_spatial_3d_candidate", False))),
            float(bool(data.get("overlap_graph_strict_h3_candidate", False))),
            float(bool(rank.get("rank3_selector_receipt", False))),
            float(int(graph.get("edge_count") or 0)),
            -float(len(blockers)),
        )
    if name == "overlap_native_graph_geometry_sweep_report.json":
        blockers = data.get("blockers") or []
        rank_obstruction = data.get("rank_obstruction_summary") or {}
        closest_rows = data.get("closest_strict_rows") or []
        return (
            100.0,
            float(bool(closest_rows)),
            float(bool(data.get("OVERLAP_NATIVE_GRAPH_GEOMETRY_SWEEP_RECEIPT", False))),
            float(bool(rank_obstruction.get("nontrivial_largest_gap_rank_counts"))),
            float(int(rank_obstruction.get("nontrivial_rank3_selector_count") or 0)),
            float(int(data.get("strict_h3_candidate_count") or 0)),
            float(int(data.get("rank3_selector_count") or 0)),
            float(int(data.get("spatial_3d_candidate_count") or 0)),
            float(int(data.get("graph_geometry_receipt_count") or 0)),
            float(int(data.get("case_count") or 0)),
            -float(len(blockers)),
        )
    if name == "overlap_residualized_graph_geometry_report.json":
        blockers = data.get("blockers") or []
        rank = data.get("rank_selection") or {}
        graph = data.get("graph_summary") or {}
        return (
            100.0,
            float(bool(data.get("OVERLAP_RESIDUALIZED_GRAPH_GEOMETRY_RECEIPT", False))),
            float(bool(data.get("overlap_residual_graph_spatial_3d_candidate", False))),
            float(bool(data.get("overlap_residual_graph_strict_h3_candidate", False))),
            float(bool(rank.get("rank3_selector_receipt", False))),
            float(int(graph.get("edge_count") or 0)),
            -float(len(blockers)),
        )
    if name == "overlap_residualized_graph_geometry_sweep_report.json":
        blockers = data.get("blockers") or []
        rank_obstruction = data.get("rank_obstruction_summary") or {}
        closest_rows = data.get("closest_strict_rows") or []
        return (
            100.0,
            float(bool(closest_rows)),
            float(bool(data.get("OVERLAP_RESIDUALIZED_GRAPH_GEOMETRY_SWEEP_RECEIPT", False))),
            float(bool(rank_obstruction.get("nontrivial_largest_gap_rank_counts"))),
            float(int(rank_obstruction.get("nontrivial_rank3_selector_count") or 0)),
            float(int(data.get("strict_h3_candidate_count") or 0)),
            float(int(data.get("rank3_selector_count") or 0)),
            float(int(data.get("spatial_3d_candidate_count") or 0)),
            float(int(data.get("residual_graph_receipt_count") or 0)),
            float(int(data.get("case_count") or 0)),
            -float(len(blockers)),
        )
    if name == "neutral_independent_rank_selector_audit_report.json":
        blockers = data.get("blockers") or []
        return (
            100.0,
            float(bool(data.get("NEUTRAL_INDEPENDENT_RANK3_SELECTOR_RECEIPT", False))),
            -float(len(blockers)),
        )
    if name == "strict_neutral_bulk_frontier_report.json":
        blockers = data.get("blockers") or []
        gate_rows = data.get("gate_rows") or []
        gap_rows = data.get("gate_gap_rows") or []
        residualized_gate_present = any(
            isinstance(row, dict) and str(row.get("gate", "")).startswith("overlap_residualized_graph")
            for row in gate_rows
        )
        residualized_report_count = int(data.get("overlap_residualized_graph_geometry_report_count") or 0)
        residualized_receipt_count = int(data.get("overlap_residualized_graph_geometry_receipt_count") or 0)
        residualized_spatial_count = int(data.get("overlap_residualized_graph_spatial_3d_candidate_count") or 0)
        residualized_strict_h3_count = int(
            data.get("overlap_residualized_graph_strict_h3_candidate_count") or 0
        )
        native_report_count = int(data.get("overlap_native_graph_geometry_report_count") or 0)
        native_receipt_count = int(data.get("overlap_native_graph_geometry_receipt_count") or 0)
        native_spatial_count = int(data.get("overlap_native_graph_spatial_3d_candidate_count") or 0)
        model_order_diagnostics_present = (
            "overlap_native_graph_model_order_rank3_selector_count" in data
            or "overlap_residualized_graph_model_order_rank3_selector_count" in data
        )
        return (
            100.0,
            float(bool(data.get("strict_neutral_bulk_ready", False))),
            float(bool(gap_rows)),
            float(bool(residualized_gate_present or residualized_report_count)),
            float(bool(model_order_diagnostics_present)),
            float(native_report_count),
            float(native_receipt_count),
            float(native_spatial_count),
            float(residualized_report_count),
            float(residualized_receipt_count),
            float(residualized_spatial_count),
            float(residualized_strict_h3_count),
            float(bool(data.get("control_residualized_rank3_refinement_candidate", False))),
            float(bool(data.get("overlap_native_negative_control_receipt_all", False))),
            float(bool(data.get("neutral_independent_rank3_selector_receipt", False))),
            float(int(data.get("directional_strict_ready_total") or 0)),
            -float(len(blockers)),
        )
    if name == "capacity_readback_proxy_report.json":
        gates = data.get("readiness_gates") or {}
        return (
            100.0,
            float(bool(gates.get("finite_regulator_rows_present", False))),
            float(int(data.get("row_count") or 0)),
            float(int(data.get("max_observer_count") or 0)),
            float(int(data.get("max_terminal_normal_form_count_proxy") or 0)),
        )
    if name == "no_g_clock_bridge_report.json":
        gates = data.get("readiness_gates") or {}
        return (
            100.0,
            float(bool(data.get("NO_G_CLOCK_BRIDGE_RECEIPT", False))),
            float(bool(data.get("source_predictive_G_SI", False))),
            -float(int(gates.get("forbidden_dependency_path_count") or 0)),
        )
    if name == "B_A_kernel_refinement_report.json":
        blockers = data.get("blockers") or []
        return (
            100.0,
            float(bool(data.get("B_A_KERNEL_REFINEMENT_CONVERGENCE_RECEIPT", False))),
            float(bool(data.get("two_scale_diagnostic_receipt", False))),
            float(int(data.get("patch_count_count") or 0)),
            float(int(data.get("key_pair_row_count") or 0)),
            -float(len(blockers)),
        )
    if name == "no_data_use_receipt.json":
        return (
            100.0,
            float(bool(data.get("NO_DATA_USE_RECEIPT", False) or data.get("no_data_use_receipt", False))),
            -float(bool(data.get("measurement_data_used_for_input_functions", False))),
        )
    return (100.0,)


def _find_all(roots: list[Path], name: str) -> list[Path]:
    matches: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        root = Path(root)
        candidates: list[Path] = []
        if root.is_file() and root.name == name:
            candidates.append(root)
        direct = root / name
        if direct.exists():
            candidates.append(direct)
        if root.exists() and root.is_dir():
            candidates.extend(sorted(root.glob(f"**/{name}")))
        for path in candidates:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                matches.append(path)
    return matches


def _readme(report: dict[str, Any]) -> str:
    claims = report.get("claims", {})
    files = "\n".join(f"- `{name}`" for name in report.get("files", []))
    return (
        "# OPH-FPE Measurement Pack\n\n"
        f"{report.get('claim_boundary')}\n\n"
        "## Claim Flags\n\n"
        f"- static galaxy measurement fit: {claims.get('static_galaxy_measurement_fit')}\n"
        f"- physical CMB prediction: {claims.get('physical_cmb_prediction')}\n"
        f"- physical CMB output comparison receipt: {claims.get('physical_cmb_output_comparison_receipt')}\n"
        f"- physical CMB output usable data receipt: {claims.get('physical_cmb_output_usable_data_receipt')}\n"
        f"- physical CMB output prediction receipt: {claims.get('physical_cmb_output_prediction_receipt')}\n"
        f"- physical CMB output OPH diagnostic models: {claims.get('physical_cmb_output_oph_diagnostic_model_count')}\n"
        f"- physical CMB output best OPH chi2/bin: {claims.get('physical_cmb_output_best_oph_chi2_per_bin')}\n"
        f"- physical CMB output best OPH residual bins: {claims.get('physical_cmb_output_best_oph_residual_bin_count')}\n"
        f"- physical CMB output best OPH RMS sigma residual: {claims.get('physical_cmb_output_best_oph_rms_sigma_residual')}\n"
        f"- physical CMB output best OPH max abs sigma residual: {claims.get('physical_cmb_output_best_oph_max_abs_sigma_residual')}\n"
        f"- physical CMB output best OPH peak count: {claims.get('physical_cmb_output_best_oph_peak_count')}\n"
        f"- physical CMB output best OPH mean abs peak ell delta: {claims.get('physical_cmb_output_best_oph_mean_abs_peak_ell_delta')}\n"
        f"- physical CMB output best OPH mean abs peak-height fractional delta: {claims.get('physical_cmb_output_best_oph_mean_abs_peak_height_fractional_delta')}\n"
        f"- exact target CMB curve comparable: {claims.get('exact_cmb_curve_comparable')}\n"
        f"- finite repair-clock CMB curve comparable: {claims.get('finite_repair_clock_cmb_curve_comparable')}\n"
        f"- finite repair-clock CMB uses finite-lattice clock: {claims.get('finite_repair_clock_cmb_finite_lattice_clock')}\n"
        f"- finite repair-clock CMB physical prediction: {claims.get('finite_repair_clock_cmb_physical_prediction')}\n"
        f"- CAMB LambdaCDM CDM-limit Boltzmann receipt: {claims.get('camb_lcdm_cdm_limit_boltzmann_receipt')}\n"
        f"- OPH CMB anomaly module ready: {claims.get('camb_lcdm_oph_anomaly_module_ready')}\n"
        f"- chart-level 3+1D: {claims.get('chart_level_3p1')}\n"
        f"- theorem-assisted H3 bulk: {claims.get('theorem_assisted_h3_bulk')}\n"
        f"- observer modular-time receipt: {claims.get('observer_modular_time_receipt')}\n"
        f"- observer modular-time observer count: {claims.get('observer_modular_time_observer_count')}\n"
        f"- observer-facing 3+1D/H3 receipt: {claims.get('observer_facing_3p1d_h3_experience_receipt')}\n"
        f"- observer-facing 3+1D/H3 blockers: {claims.get('observer_modular_experience_blockers')}\n"
        f"- strict neutral bulk: {claims.get('strict_neutral_bulk')}\n"
        f"- neutral profile audit written: {claims.get('neutral_profile_audit_written')}\n"
        f"- neutral profile strict-3D ready count: {claims.get('neutral_profile_strict_3d_ready_count')}\n"
        f"- control-residualized rank-3 refinement candidate: {claims.get('control_residualized_rank3_refinement_candidate')}\n"
        f"- control-residualized rank-3 physical claim: {claims.get('control_residualized_rank3_physical_claim')}\n"
        f"- neutral 3D bulk audit written: {claims.get('neutral_3d_bulk_audit_written')}\n"
        f"- neutral 3D bulk audit ready: {claims.get('neutral_3d_bulk_audit_ready')}\n"
        f"- overlap-native neutral control written: {claims.get('overlap_native_neutral_control_written')}\n"
        f"- overlap-native negative-control receipt: {claims.get('overlap_native_negative_control_receipt')}\n"
        f"- overlap-native spatial-3D candidate: {claims.get('overlap_native_spatial_3d_candidate')}\n"
        f"- overlap-native strict-H3 candidate: {claims.get('overlap_native_strict_h3_candidate')}\n"
        f"- overlap-native graph geometry receipt: {claims.get('overlap_native_graph_geometry_receipt')}\n"
        f"- overlap-native graph spatial-3D candidate: {claims.get('overlap_native_graph_spatial_3d_candidate')}\n"
        f"- overlap-native graph strict-H3 candidate: {claims.get('overlap_native_graph_strict_h3_candidate')}\n"
        f"- overlap-native graph rank-3 selector: {claims.get('overlap_native_graph_rank3_selector')}\n"
        f"- overlap-native graph sweep written: {claims.get('overlap_native_graph_sweep_written')}\n"
        f"- overlap-native graph sweep cases: {claims.get('overlap_native_graph_sweep_case_count')}\n"
        f"- overlap-native graph sweep spatial candidates: {claims.get('overlap_native_graph_sweep_spatial_candidates')}\n"
        f"- overlap-native graph sweep strict-H3 candidates: {claims.get('overlap_native_graph_sweep_strict_h3_candidates')}\n"
        f"- overlap-native graph sweep rank-3 selectors: {claims.get('overlap_native_graph_sweep_rank3_selectors')}\n"
        f"- overlap-native graph sweep closest strict candidates: {claims.get('overlap_native_graph_sweep_closest_strict_candidate_count')}\n"
        f"- overlap-native graph sweep nontrivial rank-3 selectors: {claims.get('overlap_native_graph_sweep_nontrivial_rank3_selectors')}\n"
        f"- overlap-native graph sweep dominant largest-gap rank: {claims.get('overlap_native_graph_sweep_dominant_largest_gap_rank')}\n"
        f"- overlap-native graph sweep dominant nontrivial largest-gap rank: {claims.get('overlap_native_graph_sweep_dominant_nontrivial_largest_gap_rank')}\n"
        f"- overlap-native graph sweep max rank-3 EV: {claims.get('overlap_native_graph_sweep_max_rank3_ev')}\n"
        f"- overlap-native graph sweep max nontrivial rank-3 EV: {claims.get('overlap_native_graph_sweep_max_nontrivial_rank3_ev')}\n"
        f"- overlap-native graph sweep median effective rank: {claims.get('overlap_native_graph_sweep_median_effective_rank')}\n"
        f"- overlap-native graph sweep median nontrivial effective rank: {claims.get('overlap_native_graph_sweep_median_nontrivial_effective_rank')}\n"
        f"- residualized overlap graph receipt: {claims.get('overlap_residualized_graph_geometry_receipt')}\n"
        f"- residualized overlap graph spatial-3D candidate: {claims.get('overlap_residualized_graph_spatial_3d_candidate')}\n"
        f"- residualized overlap graph strict-H3 candidate: {claims.get('overlap_residualized_graph_strict_h3_candidate')}\n"
        f"- residualized overlap graph rank-3 selector: {claims.get('overlap_residualized_graph_rank3_selector')}\n"
        f"- residualized overlap graph sweep written: {claims.get('overlap_residualized_graph_sweep_written')}\n"
        f"- residualized overlap graph sweep cases: {claims.get('overlap_residualized_graph_sweep_case_count')}\n"
        f"- residualized overlap graph sweep spatial candidates: {claims.get('overlap_residualized_graph_sweep_spatial_candidates')}\n"
        f"- residualized overlap graph sweep strict-H3 candidates: {claims.get('overlap_residualized_graph_sweep_strict_h3_candidates')}\n"
        f"- residualized overlap graph sweep rank-3 selectors: {claims.get('overlap_residualized_graph_sweep_rank3_selectors')}\n"
        f"- residualized overlap graph sweep closest strict candidates: {claims.get('overlap_residualized_graph_sweep_closest_strict_candidate_count')}\n"
        f"- residualized overlap graph sweep nontrivial rank-3 selectors: {claims.get('overlap_residualized_graph_sweep_nontrivial_rank3_selectors')}\n"
        f"- residualized overlap graph sweep dominant largest-gap rank: {claims.get('overlap_residualized_graph_sweep_dominant_largest_gap_rank')}\n"
        f"- residualized overlap graph sweep dominant nontrivial largest-gap rank: {claims.get('overlap_residualized_graph_sweep_dominant_nontrivial_largest_gap_rank')}\n"
        f"- residualized overlap graph sweep raw rank-1 cases: {claims.get('overlap_residualized_graph_sweep_raw_rank1_cases')}\n"
        f"- residualized overlap graph sweep max nontrivial rank-3 EV: {claims.get('overlap_residualized_graph_sweep_max_nontrivial_rank3_ev')}\n"
        f"- residualized overlap graph sweep median nontrivial effective rank: {claims.get('overlap_residualized_graph_sweep_median_nontrivial_effective_rank')}\n"
        f"- neutral independent rank-selector audit written: {claims.get('neutral_independent_rank_selector_audit_written')}\n"
        f"- neutral independent rank-3 selector receipt: {claims.get('neutral_independent_rank3_selector_receipt')}\n"
        f"- neutral independent rank-selector run count: {claims.get('neutral_independent_rank_selector_run_count')}\n"
        f"- neutral selector control-quotient rank-3 count: {claims.get('neutral_independent_rank_selector_control_rank3_count')}\n"
        f"- neutral selector control-quotient median effective rank: {claims.get('neutral_independent_rank_selector_control_median_effective_rank')}\n"
        f"- strict neutral frontier written: {claims.get('strict_neutral_bulk_frontier_written')}\n"
        f"- strict neutral frontier ready: {claims.get('strict_neutral_bulk_frontier_ready')}\n"
        f"- strict neutral frontier hard-gate gaps: {claims.get('strict_neutral_bulk_frontier_gap_count')}\n"
        f"- strict neutral frontier rank-3 candidate: {claims.get('strict_neutral_bulk_frontier_rank3_candidate')}\n"
        f"- strict neutral frontier overlap controls: {claims.get('strict_neutral_bulk_frontier_overlap_controls')}\n"
        f"- strict neutral frontier overlap graph receipts: {claims.get('strict_neutral_bulk_frontier_overlap_graph_receipts')}\n"
        f"- strict neutral frontier overlap graph spatial candidates: {claims.get('strict_neutral_bulk_frontier_overlap_graph_spatial_candidates')}\n"
        f"- strict neutral frontier overlap graph strict-H3 candidates: {claims.get('strict_neutral_bulk_frontier_overlap_graph_strict_h3_candidates')}\n"
        f"- strict neutral frontier overlap graph model-order rank-3 selectors: {claims.get('strict_neutral_bulk_frontier_overlap_graph_model_order_rank3_selectors')}\n"
        f"- strict neutral frontier residual graph receipts: {claims.get('strict_neutral_bulk_frontier_overlap_residual_graph_receipts')}\n"
        f"- strict neutral frontier residual graph spatial candidates: {claims.get('strict_neutral_bulk_frontier_overlap_residual_graph_spatial_candidates')}\n"
        f"- strict neutral frontier residual graph strict-H3 candidates: {claims.get('strict_neutral_bulk_frontier_overlap_residual_graph_strict_h3_candidates')}\n"
        f"- strict neutral frontier residual graph rank-3 selectors: {claims.get('strict_neutral_bulk_frontier_overlap_residual_graph_rank3_selectors')}\n"
        f"- strict neutral frontier residual graph model-order rank-3 selectors: {claims.get('strict_neutral_bulk_frontier_overlap_residual_graph_model_order_rank3_selectors')}\n"
        f"- strict neutral frontier independent selector: {claims.get('strict_neutral_bulk_frontier_independent_selector')}\n"
        f"- strict blind record-transition 3D candidate: {claims.get('strict_blind_record_transition_3d_candidate')}\n"
        f"- production particles: {claims.get('production_particles')}\n\n"
        f"- finite certificate stack ready: {claims.get('finite_certificate_stack_ready')}\n"
        f"- finite certificate theorem-grade: {claims.get('finite_certificate_theorem_grade')}\n"
        f"- finite certificate real physics: {claims.get('finite_certificate_real_physics')}\n\n"
        f"- finite transition matrix ready: {claims.get('finite_transition_matrix_ready')}\n"
        f"- finite transition clock certified: {claims.get('finite_transition_clock_certified')}\n"
        f"- finite transition eta_R finite-derived: {claims.get('finite_transition_eta_R_finite_lattice_derived')}\n\n"
        f"- screen-capacity observed branch: {claims.get('screen_capacity_observed_branch_available')}\n"
        f"- screen-capacity finite fixed point solved: {claims.get('screen_capacity_finite_fixed_point_solved')}\n\n"
        f"- capacity-readback proxy written: {claims.get('capacity_readback_proxy_written')}\n"
        f"- capacity-readback proxy row count: {claims.get('capacity_readback_proxy_row_count')}\n"
        f"- capacity-readback fixed point solved: {claims.get('capacity_readback_proxy_fixed_point_solved')}\n\n"
        f"- P/N resonance report written: {claims.get('pn_resonance_written')}\n"
        f"- P/N resonance branch status: {claims.get('pn_resonance_branch_status')}\n"
        f"- P/N resonance numeric replay: {claims.get('pn_resonance_numeric_replay')}\n"
        f"- P/N resonance theorem-grade receipt: {claims.get('pn_resonance_receipt')}\n\n"
        f"- P/N silence-to-observation report written: {claims.get('pn_silence_to_observation_written')}\n"
        f"- P/N silence-to-observation scale-compressed receipt: {claims.get('pn_silence_to_observation_scale_compressed_receipt')}\n"
        f"- P/N silence-to-observation literal global N: {claims.get('pn_silence_to_observation_literal_global_N')}\n"
        f"- P/N silence-to-observation dynamic controls: {claims.get('pn_silence_to_observation_dynamic_detuning_controls')}\n\n"
        f"- kernel dispatch written: {claims.get('kernel_dispatch_written')}\n"
        f"- kernel dispatch routing decision: {claims.get('kernel_dispatch_routing_decision')}\n"
        f"- kernel dispatch effective acceleration: {claims.get('kernel_dispatch_effective_acceleration_enabled')}\n"
        f"- kernel dispatch changed observables: {claims.get('kernel_dispatch_physical_observables_changed')}\n\n"
        f"- positive-geometry kernel written: {claims.get('positive_geometry_kernel_written')}\n"
        f"- positive-geometry kernel verdict: {claims.get('positive_geometry_kernel_verdict')}\n"
        f"- positive-geometry acceleration enabled: {claims.get('positive_geometry_kernel_acceleration_enabled')}\n"
        f"- positive-geometry fallback required: {claims.get('positive_geometry_kernel_fallback_required')}\n\n"
        f"- scale bridge report written: {claims.get('scale_bridge_written')}\n"
        f"- scale bridge independent supplied: {claims.get('scale_bridge_independent_supplied')}\n"
        f"- scale bridge dimensionful G eligible: {claims.get('scale_bridge_dimensionful_G_eligible')}\n"
        f"- scale bridge finite-simulator G_SI: {claims.get('scale_bridge_finite_simulator_derived_G_SI')}\n\n"
        f"- no-G clock bridge report written: {claims.get('no_g_clock_bridge_written')}\n"
        f"- no-G clock bridge receipt: {claims.get('no_g_clock_bridge_receipt')}\n"
        f"- no-G clock bridge source-predictive G_SI: {claims.get('no_g_clock_bridge_source_predictive_G_SI')}\n"
        f"- no-G clock bridge G_SI checksum: {claims.get('no_g_clock_bridge_G_SI_checksum')}\n\n"
        f"- parent-collar local density receipt: {claims.get('parent_collar_local_density_receipt')}\n"
        f"- parent-collar theorem-grade: {claims.get('parent_collar_theorem_grade')}\n"
        f"- repair-clock certificate: {claims.get('repair_clock_certificate')}\n"
        f"- repair-clock eta_R finite-derived: {claims.get('repair_clock_eta_R_finite_lattice_derived')}\n\n"
        f"- B_A kernel candidate receipt: {claims.get('B_A_kernel_candidate_receipt')}\n"
        f"- B_A kernel physical receipt: {claims.get('B_A_kernel_physical_receipt')}\n"
        f"- B_A kernel row count: {claims.get('B_A_kernel_row_count')}\n\n"
        f"- B_A kernel refinement two-scale diagnostic: {claims.get('B_A_kernel_refinement_two_scale_diagnostic')}\n"
        f"- B_A kernel refinement convergence: {claims.get('B_A_kernel_refinement_convergence_receipt')}\n"
        f"- B_A kernel refinement patch-count count: {claims.get('B_A_kernel_refinement_patch_count_count')}\n\n"
        f"- B_A kernel refinement key-pair row count: {claims.get('B_A_kernel_refinement_key_pair_row_count')}\n"
        f"- B_A kernel refinement key-pair stable fraction: {claims.get('B_A_kernel_refinement_key_pair_stable_fraction')}\n\n"
        f"- Boltzmann input table written: {claims.get('boltzmann_input_table_written')}\n"
        f"- physical CMB promotion audit written: {claims.get('physical_cmb_promotion_audit_written')}\n"
        f"- physical CMB promotion ready: {claims.get('physical_cmb_promotion_ready')}\n"
        f"- physical CMB promotion blocker count: {claims.get('physical_cmb_promotion_blocker_count')}\n"
        f"- physical CMB frontier written: {claims.get('physical_cmb_frontier_written')}\n"
        f"- physical CMB frontier ready: {claims.get('physical_cmb_frontier_ready')}\n"
        f"- physical CMB frontier gate count: {claims.get('physical_cmb_frontier_gate_count')}\n"
        f"- physical CMB frontier hard-gate gaps: {claims.get('physical_cmb_frontier_gap_count')}\n"
        f"- physical CMB frontier blocker count: {claims.get('physical_cmb_frontier_blocker_count')}\n"
        f"- physical CMB output comparison written: {claims.get('physical_cmb_output_comparison_written')}\n"
        f"- screen-power primordial reference ready: {claims.get('screen_power_simulator_primordial_ready')}\n"
        f"- MaxEnt Green source receipt: {claims.get('maxent_green_source_receipt')}\n"
        f"- selector-elimination theorem-side receipt: {claims.get('selector_elimination_theorem_side_receipt')}\n"
        f"- CMB anomaly parity proxy: {claims.get('cmb_anomaly_parity_asymmetry_proxy')}\n"
        f"- CMB anomaly low-power proxy: {claims.get('cmb_anomaly_low_power_proxy')}\n"
        f"- CMB anomaly Planck-tilt proxy: {claims.get('cmb_anomaly_planck_tilt_proxy')}\n\n"
        f"- neutrino measurement-comparable: {claims.get('neutrino_measurement_comparable')}\n"
        f"- neutrino finite-lattice-derived: {claims.get('neutrino_finite_lattice_derived')}\n"
        f"- H0/S8 measurement-comparable: {claims.get('h0s8_measurement_comparable')}\n"
        f"- H0/S8 physical prediction ready: {claims.get('h0s8_physical_prediction_ready')}\n"
        f"- H0/S8 finite-kernel gates closed: {claims.get('h0s8_finite_kernel_gates_closed')}\n"
        f"- compressed likelihood reference: {claims.get('compressed_likelihood_reference')}\n\n"
        f"- official Planck likelihood readiness written: {claims.get('official_planck_likelihood_readiness_written')}\n"
        f"- official Planck likelihood execution ready: {claims.get('official_planck_likelihood_execution_ready')}\n"
        f"- official Planck likelihood data paths configured: {claims.get('official_planck_likelihood_data_paths_configured')}\n"
        f"- official Planck clik API available: {claims.get('official_planck_clik_api_available')}\n\n"
        f"- repair-scale closure numeric match: {claims.get('repair_scale_closure_numeric_match')}\n"
        f"- repair-scale 24 rounds finite-derived: {claims.get('repair_scale_24_rounds_derived')}\n"
        f"- repair-scale eta_R finite-derived: {claims.get('repair_scale_finite_eta_R')}\n\n"
        f"- scale-compressed operator receipt: {claims.get('scale_compressed_operator_receipt')}\n"
        f"- scale-compressed populated H3 preview: {claims.get('scale_compressed_populated_h3_preview')}\n"
        f"- scale-compressed physical CMB: {claims.get('scale_compressed_physical_cmb_prediction')}\n\n"
        f"- scale-compressed CAMB curve comparable: {claims.get('scale_compressed_cmb_curve_comparable')}\n"
        f"- scale-compressed CAMB physical prediction: {claims.get('scale_compressed_cmb_physical_prediction')}\n\n"
        f"- object-H3 bulk viewer written: {claims.get('object_h3_bulk_viewer_written')}\n"
        f"- object-H3 bulk viewer object count: {claims.get('object_h3_bulk_viewer_object_count')}\n"
        f"- object-H3 bulk viewer theorem-assisted: {claims.get('object_h3_bulk_viewer_theorem_assisted')}\n\n"
        f"- CMB/neutral frontier viewer written: {claims.get('cmb_neutral_frontier_viewer_written')}\n"
        f"- CMB/neutral frontier TT bin count: {claims.get('cmb_neutral_frontier_viewer_tt_bin_count')}\n\n"
        f"- CMB static plots written: {claims.get('cmb_static_plots_written')}\n"
        f"- CMB static plots best OPH model: {claims.get('cmb_static_plots_best_oph_model')}\n"
        f"- CMB static plots file count: {claims.get('cmb_static_plots_file_count')}\n\n"
        f"- strict neutral near-miss rows: {claims.get('cmb_static_plots_strict_neutral_near_miss_count')}\n"
        f"- strict neutral best near-miss gate score: {claims.get('cmb_static_plots_strict_neutral_best_near_miss_gate_score')}\n"
        f"- strict neutral best near-miss dimension error: {claims.get('cmb_static_plots_strict_neutral_best_near_miss_dimension_error')}\n"
        f"- strict neutral best near-miss nontrivial rank-3 EV: {claims.get('cmb_static_plots_strict_neutral_best_near_miss_nontrivial_rank3_ev')}\n\n"
        "## Files\n\n"
        f"{files}\n"
    )
