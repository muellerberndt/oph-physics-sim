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

    claims = _collect_claims(roots)
    (out / "claims.json").write_text(json.dumps(claims, indent=2, default=str), encoding="utf-8")

    exported: dict[str, str] = {}
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
    _copy_first(roots, out / "b_a_parent_report.json", exported, "b_a_parent_report.json")
    _copy_first(roots, out / "b_a_parent_report.md", exported, "b_a_parent_report.md")
    _copy_first(roots, out / "b_a_parent_rows.csv", exported, "b_a_parent_rows.csv")
    _copy_first(roots, out / "b_a_parent_control_rows.csv", exported, "b_a_parent_control_rows.csv")
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


def _collect_claims(roots: list[Path]) -> dict[str, Any]:
    static_galaxy = _first_json(roots, "static_galaxy_measurement_report.json")
    comparable = _first_json(roots, "comparable_data_snapshot.json")
    bulk = _first_json(roots, "bulk_proof_certificate_report.json")
    exact_cmb = _first_json(roots, "oph_exact_cmb_camb_report.json")
    shape = _first_json(roots, "shape_substrate_summary.json")
    finite = _first_json(roots, "finite_certificate_report.json")
    finite_transition = _first_json(roots, "finite_repair_transition_matrix_report.json")
    scalar_repair_semigroup = _first_json(roots, "scalar_repair_semigroup_report.json")
    screen_capacity = _first_json(roots, "screen_capacity_closure_report.json")
    repair_scale = _first_json(roots, "repair_scale_closure_report.json")
    parent_collar = _first_json(roots, "parent_collar_ladder_report.json")
    repair_clock = _first_json(roots, "repair_clock_certificate_report.json")
    boltzmann_inputs = _first_json(roots, "oph_boltzmann_input_report.json")
    ba_parent = _first_json(roots, "b_a_parent_report.json")
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
    neutrino_gates = neutrinos.get("readiness_gates") or {}
    h0s8_gates = h0s8.get("readiness_gates") or {}
    h0s8_comparisons = h0s8.get("measurement_comparisons") or {}
    cmb_anomaly_aggregate = cmb_anomaly.get("aggregate") or {}
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
    path = _find_first(roots, name)
    if path is None:
        return {}
    return _read_json(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _find_first(roots: list[Path], name: str) -> Path | None:
    for root in roots:
        root = Path(root)
        if root.is_file() and root.name == name:
            return root
        direct = root / name
        if direct.exists():
            return direct
        if root.exists() and root.is_dir():
            matches = sorted(root.glob(f"**/{name}"))
            if matches:
                return matches[0]
    return None


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
        f"- chart-level 3+1D: {claims.get('chart_level_3p1')}\n"
        f"- theorem-assisted H3 bulk: {claims.get('theorem_assisted_h3_bulk')}\n"
        f"- strict neutral bulk: {claims.get('strict_neutral_bulk')}\n"
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
        f"- parent-collar local density receipt: {claims.get('parent_collar_local_density_receipt')}\n"
        f"- parent-collar theorem-grade: {claims.get('parent_collar_theorem_grade')}\n"
        f"- repair-clock certificate: {claims.get('repair_clock_certificate')}\n"
        f"- repair-clock eta_R finite-derived: {claims.get('repair_clock_eta_R_finite_lattice_derived')}\n\n"
        f"- Boltzmann input table written: {claims.get('boltzmann_input_table_written')}\n"
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
