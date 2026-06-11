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
    _copy_first(roots, out / "neutral_3d_bulk_audit_report.json", exported, "neutral_3d_bulk_audit_report.json")
    _copy_first(roots, out / "neutral_3d_bulk_audit_report.md", exported, "neutral_3d_bulk_audit_report.md")
    _copy_first(
        roots,
        out / "neutral_independent_rank_selector_audit_report.json",
        exported,
        "neutral_independent_rank_selector_audit_report.json",
    )
    _copy_first(
        roots,
        out / "neutral_independent_rank_selector_audit_report.md",
        exported,
        "neutral_independent_rank_selector_audit_report.md",
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
    _copy_first(roots, out / "oph_scale_bridge_report.json", exported, "oph_scale_bridge_report.json")
    _copy_first(roots, out / "oph_scale_bridge_report.md", exported, "oph_scale_bridge_report.md")
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
    repair_scale = _first_json(roots, "repair_scale_closure_report.json")
    scale_bridge = _first_json(roots, "oph_scale_bridge_report.json")
    parent_collar = _first_json(roots, "parent_collar_ladder_report.json")
    repair_clock = _first_json(roots, "repair_clock_certificate_report.json")
    boltzmann_inputs = _first_json(roots, "oph_boltzmann_input_report.json")
    finite_collar_boltzmann = _first_json(roots, "finite_collar_boltzmann_bundle_report.json")
    finite_collar_projection = _first_json(roots, "finite_collar_cmb_projection_report.json")
    physical_cmb_input = _first_json(roots, "physical_cmb_input_report.json")
    physical_cmb_promotion = _first_json(roots, "physical_cmb_promotion_audit_report.json")
    neutral_profile = _first_json(roots, "neutral_profile_audit_report.json")
    prime_rank_refinement = _first_json(roots, "prime_geometric_rank_refinement_report.json")
    neutral_3d_bulk_audit = _first_json(roots, "neutral_3d_bulk_audit_report.json")
    neutral_rank_selector_audit = _first_json(roots, "neutral_independent_rank_selector_audit_report.json")
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
    comparable_lorentz = comparable.get("measurement_lanes", {}).get("support_visible_lorentz_branch", {})
    comparable_neutral = comparable.get("measurement_lanes", {}).get("neutral_observer_reconstruction", {})
    screen_capacity_gates = screen_capacity.get("readiness_gates") or {}
    repair_scale_gates = repair_scale.get("readiness_gates") or {}
    scale_bridge_gates = scale_bridge.get("readiness_gates") or {}
    scale_bridge_values = scale_bridge.get("scale_bridge") or {}
    neutrino_gates = neutrinos.get("readiness_gates") or {}
    h0s8_gates = h0s8.get("readiness_gates") or {}
    h0s8_comparisons = h0s8.get("measurement_comparisons") or {}
    cmb_anomaly_aggregate = cmb_anomaly.get("aggregate") or {}
    neutral_profile_rows = neutral_profile.get("profile_rows", [])
    if not isinstance(neutral_profile_rows, list):
        neutral_profile_rows = []
    physical_cmb_input_status = physical_cmb_input.get("input_status") or {}
    physical_cmb_a_zeta = physical_cmb_input_status.get("A_zeta") or {}
    physical_cmb_b_a = physical_cmb_input_status.get("B_A_k_a") or {}
    physical_cmb_rho_a = physical_cmb_input_status.get("rho_A_a") or {}
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
        "strict_neutral_bulk": bool(
            comparable.get("strict_neutral_3d_bulk_any", False)
            or comparable.get("strict_neutral_3d_bulk_count", 0)
            or bulk.get("strict_neutral_third_person_bulk_established", False)
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
        "claim_boundary": (
            "Measurement-pack claim flags are copied from source receipts. Static galaxy fits can be "
            "measurement-facing without being CMB or bulk proofs. CMB and strict neutral bulk remain "
            "false unless their source gates pass."
        ),
    }


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
        return (
            100.0,
            float(bool(data.get("PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT", False))),
            float(bool(data.get("physical_cmb_prediction_eligible", False))),
            float(present_sources),
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
    if name == "neutral_3d_bulk_audit_report.json":
        blockers = data.get("blockers") or []
        return (
            100.0,
            float(bool(data.get("strict_neutral_bulk_ready", False))),
            float(bool(data.get("control_residualized_rank3_refinement_candidate", False))),
            float(int(data.get("sweep_report_count") or 0)),
            -float(len(blockers)),
        )
    if name == "neutral_independent_rank_selector_audit_report.json":
        blockers = data.get("blockers") or []
        return (
            100.0,
            float(bool(data.get("NEUTRAL_INDEPENDENT_RANK3_SELECTOR_RECEIPT", False))),
            float(int(data.get("run_count") or 0)),
            float(int(data.get("control_quotient_rank3_candidate_count") or 0)),
            -float(len(blockers)),
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
        f"- exact target CMB curve comparable: {claims.get('exact_cmb_curve_comparable')}\n"
        f"- finite repair-clock CMB curve comparable: {claims.get('finite_repair_clock_cmb_curve_comparable')}\n"
        f"- finite repair-clock CMB uses finite-lattice clock: {claims.get('finite_repair_clock_cmb_finite_lattice_clock')}\n"
        f"- finite repair-clock CMB physical prediction: {claims.get('finite_repair_clock_cmb_physical_prediction')}\n"
        f"- CAMB LambdaCDM CDM-limit Boltzmann receipt: {claims.get('camb_lcdm_cdm_limit_boltzmann_receipt')}\n"
        f"- OPH CMB anomaly module ready: {claims.get('camb_lcdm_oph_anomaly_module_ready')}\n"
        f"- chart-level 3+1D: {claims.get('chart_level_3p1')}\n"
        f"- theorem-assisted H3 bulk: {claims.get('theorem_assisted_h3_bulk')}\n"
        f"- strict neutral bulk: {claims.get('strict_neutral_bulk')}\n"
        f"- neutral profile audit written: {claims.get('neutral_profile_audit_written')}\n"
        f"- neutral profile strict-3D ready count: {claims.get('neutral_profile_strict_3d_ready_count')}\n"
        f"- control-residualized rank-3 refinement candidate: {claims.get('control_residualized_rank3_refinement_candidate')}\n"
        f"- control-residualized rank-3 physical claim: {claims.get('control_residualized_rank3_physical_claim')}\n"
        f"- neutral 3D bulk audit written: {claims.get('neutral_3d_bulk_audit_written')}\n"
        f"- neutral 3D bulk audit ready: {claims.get('neutral_3d_bulk_audit_ready')}\n"
        f"- neutral independent rank-selector audit written: {claims.get('neutral_independent_rank_selector_audit_written')}\n"
        f"- neutral independent rank-3 selector receipt: {claims.get('neutral_independent_rank3_selector_receipt')}\n"
        f"- neutral independent rank-selector run count: {claims.get('neutral_independent_rank_selector_run_count')}\n"
        f"- neutral selector control-quotient rank-3 count: {claims.get('neutral_independent_rank_selector_control_rank3_count')}\n"
        f"- neutral selector control-quotient median effective rank: {claims.get('neutral_independent_rank_selector_control_median_effective_rank')}\n"
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
        f"- scale bridge report written: {claims.get('scale_bridge_written')}\n"
        f"- scale bridge independent supplied: {claims.get('scale_bridge_independent_supplied')}\n"
        f"- scale bridge dimensionful G eligible: {claims.get('scale_bridge_dimensionful_G_eligible')}\n"
        f"- scale bridge finite-simulator G_SI: {claims.get('scale_bridge_finite_simulator_derived_G_SI')}\n\n"
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
        f"- repair-scale closure numeric match: {claims.get('repair_scale_closure_numeric_match')}\n"
        f"- repair-scale 24 rounds finite-derived: {claims.get('repair_scale_24_rounds_derived')}\n"
        f"- repair-scale eta_R finite-derived: {claims.get('repair_scale_finite_eta_R')}\n\n"
        f"- scale-compressed operator receipt: {claims.get('scale_compressed_operator_receipt')}\n"
        f"- scale-compressed populated H3 preview: {claims.get('scale_compressed_populated_h3_preview')}\n"
        f"- scale-compressed physical CMB: {claims.get('scale_compressed_physical_cmb_prediction')}\n\n"
        f"- scale-compressed CAMB curve comparable: {claims.get('scale_compressed_cmb_curve_comparable')}\n"
        f"- scale-compressed CAMB physical prediction: {claims.get('scale_compressed_cmb_physical_prediction')}\n\n"
        "## Files\n\n"
        f"{files}\n"
    )
