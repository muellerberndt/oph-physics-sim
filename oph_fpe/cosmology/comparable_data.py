from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from oph_fpe.bulk.h3_chart import h3_distance_matrix
from oph_fpe.bulk.observer_reconstruction import neutral_dimension_report_from_distance


RELEVANT_REPORTS = (
    "bw_state_derived_report.json",
    "transition_scale_selection_report.json",
    "paper_3d_bulk_chart_report.json",
    "modular_response_h3_report.json",
    "h3_refit_ensemble_report.json",
    "h3_refit_seed_ensemble_report.json",
    "caps_to_h3_minimal_report.json",
    "record_populated_h3_report.json",
    "record_family_h3_report.json",
    "defect_cluster_h3_report.json",
    "observer_chart_object_h3_report.json",
    "observer_chart_object_h3_lineage_report.json",
    "observer_chart_object_h3_transition_history_report.json",
    "observer_chart_object_h3_observer_transition_mixture_report.json",
    "observer_chart_object_h3_recomputed.json",
    "observer_consensus_report.json",
    "object_consensus_report.json",
    "bulk_reconstruction_report.json",
    "strict_neutral_bulk_report.json",
    "strict_neutral_object_bulk_report.json",
    "cmb_lite_comparison_report.json",
    "cmb_transfer_report.json",
    "oph_cmb_selector_elimination_report.json",
    "camb_lcdm_baseline_report.json",
    "oph_screen_power_report.json",
    "oph_screen_camb_report.json",
    "maxent_green_spectrum_report.json",
    "repair_clock_certificate_report.json",
    "scalar_repair_semigroup_report.json",
    "fossil_spectrum_report.json",
    "cmb_fossil_bridge_report.json",
    "oph_inflation_cmb_bridge_report.json",
    "inflation_certificate_report.json",
    "oph_inflation_cmb_camb_report.json",
    "oph_exact_cmb_camb_report.json",
    "finite_repair_clock_cmb_camb_report.json",
    "oph_unique_prediction_gate_report.json",
    "oph_cnb_neutrino_report.json",
    "finite_certificate_report.json",
    "finite_certificate_manifest.json",
    "scalar_quotient_report.json",
    "neutral_profile_audit_report.json",
    "neutral_profile_audit.json",
    "prime_geometric_rank_sweep_report.json",
    "prime_geometric_rank_refinement_report.json",
    "parent_collar_ladder_report.json",
    "paired_b_a_perturbation_report.json",
    "b_a_parent_report.json",
    "screen_capacity_closure_report.json",
    "repair_scale_closure_report.json",
    "scale_compressed_repair_report.json",
    "scale_compressed_cmb_camb_report.json",
    "cmb_parameter_derivation_report.json",
    "oph_boltzmann_input_report.json",
    "finite_collar_boltzmann_bundle_report.json",
    "finite_collar_cmb_projection_report.json",
    "oph_cmb_stress_report.json",
    "cmb_anomaly_report.json",
    "sync_gap_report.json",
    "hot_release_report.json",
    "adiabaticity_report.json",
    "h0s8_branch_report.json",
    "h0s8_lane8_certificate_report.json",
    "galaxy_proxy_report.json",
    "static_galaxy_measurement_report.json",
    "cl_comparison_report.json",
    "array_holonomy_report.json",
    "defect_timeline_report.json",
    "defect_interaction_report.json",
    "particle_likeness_report.json",
    "controlled_defect_particle_assay_report.json",
    "defect_h3_worldlines_report.json",
    "emergence_status_report.json",
    "bulk_proof_certificate_report.json",
    "shape_substrate_summary.json",
    "shape_vertex_scattering_report.json",
    "shape_dodeca_cell_report.json",
    "shape_loop_mode_report.json",
    "shape_particle_loop_report.json",
    "shape_loop_particle_report.json",
    "shape_screen_projection_report.json",
    "shape_cl_report.json",
    "shape_cmb_certificate_inputs.json",
)

H3_ENSEMBLE_FILENAMES = (
    "h3_refit_ensemble_report.json",
    "h3_refit_seed_ensemble_report.json",
)

CAPS_TO_H3_FILENAMES = (
    "caps_to_h3_minimal_report.json",
)

OBJECT_CHART_FILENAMES = (
    "observer_chart_object_h3_lineage_report.json",
    "observer_chart_object_h3_transition_history_report.json",
    "observer_chart_object_h3_observer_transition_mixture_report.json",
    "observer_chart_object_h3_recomputed.json",
    "observer_chart_object_h3_report.json",
)

CMB_LITE_FILENAMES = (
    "cmb_lite_comparison_report.json",
)

DECLARED_CAP_FLOW_STATE_MODES = frozenset({"cap_flow_graph_generator", "cap_flow_detailed_balance_kernel"})


def collect_comparable_runs(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows = [_extract_run_row(path) for path in _find_run_dirs(run_dirs)]
    return [row for row in rows if row.get("has_comparable_data")]


def comparable_data_report(run_dirs: list[Path]) -> dict[str, Any]:
    rows = collect_comparable_runs(run_dirs)
    bulk_pass_count = sum(1 for row in rows if bool(row.get("bulk_3d_established")))
    strict_neutral_bulk_count = sum(
        1
        for row in rows
        if bool(row.get("bulk_proof_strict_neutral_3d_bulk"))
        or bool(row.get("strict_neutral_bulk_receipt"))
        or bool(row.get("strict_neutral_object_bulk_receipt"))
    )
    theorem_assisted_bulk_count = sum(
        1
        for row in rows
        if bool(row.get("bulk_proof_theorem_assisted_h3_nonboundary_population"))
        or bool(row.get("bulk_proof_theorem_assisted_h3_populated_chart"))
    )
    chart_level_3p1_count = sum(1 for row in rows if bool(row.get("bulk_proof_chart_level_3p1")))
    return {
        "mode": "oph_fpe_comparable_data_snapshot",
        "run_count": len(rows),
        "measurement_lanes": {
            "support_visible_lorentz_branch": _lorentz_branch_summary(rows),
            "state_derived_bw_matrix_elements": _state_bw_summary(rows),
            "planck_tt_shape_lite": _planck_lite_summary(rows),
            "cmb_screen_basis_transfer": _cmb_transfer_summary(rows),
            "camb_lcdm_baseline": _camb_lcdm_summary(rows),
            "oph_screen_power_effective_theory": _oph_screen_power_summary(rows),
            "oph_screen_camb_transfer": _oph_screen_camb_summary(rows),
            "oph_maxent_green_screen_source": _maxent_green_summary(rows),
            "oph_repair_clock_kappa": _repair_clock_summary(rows),
            "oph_scalar_repair_semigroup": _scalar_repair_semigroup_summary(rows),
            "oph_fossil_spectrum_time_trace": _fossil_spectrum_summary(rows),
            "oph_cmb_fossil_bridge": _cmb_fossil_bridge_summary(rows),
            "oph_inflation_cmb_bridge": _oph_inflation_cmb_bridge_summary(rows),
            "oph_inflation_certificate_stack": _inflation_certificate_summary(rows),
            "oph_inflation_cmb_camb_transfer": _oph_inflation_cmb_camb_summary(rows),
            "oph_cmb_selector_elimination_v1_5": _selector_elimination_summary(rows),
            "oph_exact_cmb_camb_transfer": _oph_exact_cmb_camb_summary(rows),
            "finite_repair_clock_cmb_camb_transfer": _finite_repair_clock_cmb_camb_summary(rows),
            "oph_unique_prediction_gate": _oph_unique_prediction_summary(rows),
            "oph_cnb_neutrino_background": _oph_cnb_neutrino_summary(rows),
            "oph_finite_certificate_authority": _finite_certificate_authority_summary(rows),
            "oph_scalar_geometric_quotient": _scalar_quotient_summary(rows),
            "neutral_distance_profile_audit": _neutral_profile_audit_summary(rows),
            "prime_geometric_rank_sweep": _prime_rank_sweep_summary(rows),
            "prime_geometric_rank_refinement": _prime_rank_refinement_summary(rows),
            "oph_parent_collar_recovery_ladder": _parent_collar_ladder_summary(rows),
            "oph_B_A_parent_finite_difference": _b_a_parent_summary(rows),
            "oph_screen_capacity_closure": _screen_capacity_closure_summary(rows),
            "oph_repair_scale_closure": _repair_scale_closure_summary(rows),
            "oph_scale_compressed_repair_branch": _scale_compressed_repair_summary(rows),
            "oph_scale_compressed_cmb_camb_transfer": _scale_compressed_cmb_camb_summary(rows),
            "finite_lattice_cmb_derivation": _cmb_derivation_summary(rows),
            "oph_boltzmann_input_readouts": _oph_boltzmann_summary(rows),
            "finite_collar_boltzmann_source_bundle": _finite_collar_boltzmann_summary(rows),
            "finite_collar_cmb_projection": _finite_collar_projection_summary(rows),
            "oph_cmb_anomaly_stress_adapter": _oph_cmb_summary(rows),
            "finite_screen_cmb_anomaly_readouts": _cmb_anomaly_summary(rows),
            "low_k_synchronization_gap": _sync_gap_summary(rows),
            "hot_maxent_release": _hot_release_summary(rows),
            "same_boundary_adiabaticity": _adiabaticity_summary(rows),
            "h0_s8_branch_diagnostic": _h0s8_summary(rows),
            "shape_substrate_witness": _shape_substrate_summary(rows),
            "static_galaxy_proxy": _galaxy_proxy_summary(rows),
            "static_galaxy_measurement_fit": _static_galaxy_measurement_summary(rows),
            "minimal_caps_to_h3": _minimal_caps_to_h3_summary(rows),
            "h3_modular_response_controls": _h3_summary(rows),
            "h3_seed_ensemble_robustness": _h3_seed_ensemble_summary(rows),
            "observer_readout_consensus": _observer_readout_summary(rows),
            "observer_chart_object_population": _observer_chart_summary(rows),
            "neutral_observer_reconstruction": _neutral_reconstruction_summary(rows),
            "screen_holonomy_defect_proxy": _holonomy_summary(rows),
            "defect_worldline_particle_precursors": _defect_worldline_summary(rows),
            "controlled_defect_particle_assay": _controlled_defect_assay_summary(rows),
        },
        "rows": rows,
        "physical_cmb_prediction": False,
        "physical_matter_power_prediction": False,
        "bulk_3d_established_count": int(bulk_pass_count),
        "bulk_3d_established_any": bool(bulk_pass_count > 0),
        "theorem_assisted_h3_bulk_count": int(theorem_assisted_bulk_count),
        "theorem_assisted_h3_bulk_any": bool(theorem_assisted_bulk_count > 0),
        "theorem_assisted_observer_facing_h3_population_count": int(theorem_assisted_bulk_count),
        "theorem_assisted_observer_facing_h3_population_any": bool(theorem_assisted_bulk_count > 0),
        "observer_facing_3p1d_h3_experience_count": int(theorem_assisted_bulk_count),
        "observer_facing_3p1d_h3_experience_any": bool(theorem_assisted_bulk_count > 0),
        "chart_level_3p1_count": int(chart_level_3p1_count),
        "chart_level_3p1_any": bool(chart_level_3p1_count > 0),
        "strict_neutral_3d_bulk_count": int(strict_neutral_bulk_count),
        "strict_neutral_3d_bulk_any": bool(strict_neutral_bulk_count > 0),
        "bulk_3d_established": bool(rows and bulk_pass_count == len(rows)),
        "claim_boundary": (
            "Comparable-data snapshot for current OPH-FPE receipts. Planck comparisons are shape-only "
            "C_l diagnostics with normalized axes and amplitude rescaling. OPH-screen CAMB transfer reports "
            "are solver scaffolds unless their screen-power input is simulator-derived and passes the finite "
            "eta/readiness gates. OPH-CMB anomaly-stress "
            "and inflation/CMB bridge reports are continuation diagnostics, not finite-lattice derivations. "
            "adapter values are finite-collar parent diagnostics, not CAMB/CLASS anomaly-module outputs. "
            "H3 values are internal modular-response-vs-control receipts. Defect values are screen/collar holonomy proxies. "
            "The top-level bulk flag requires every comparable seed in the selected run set to pass a "
            "non-boundary H3 object-population gate or strict neutral third-person bulk gate. H3 preview "
            "alone is reported separately. This is not a physical CMB "
            "prediction, not a P(k), not a Boltzmann likelihood, and not a completed particle-emergence result."
        ),
    }


def write_comparable_data_package(run_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    report = comparable_data_report(run_dirs)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparable_data_snapshot.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    _write_rows_csv(out_dir / "comparable_data_rows.csv", report["rows"])
    (out_dir / "comparable_data_snapshot.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _find_run_dirs(roots: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        root = Path(root)
        if root.is_file():
            paths.add(root)
            continue
        if any((root / name).exists() for name in RELEVANT_REPORTS):
            paths.add(root)
        if root.exists():
            for name in RELEVANT_REPORTS:
                for report_path in root.glob(f"**/{name}"):
                    paths.add(_row_parent_for_report(report_path))
    return sorted(paths, key=lambda path: str(path))


def _row_parent_for_report(report_path: Path) -> Path:
    parent = Path(report_path).parent
    current = parent
    for _ in range(6):
        if (current / "manifest.json").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return parent


def _extract_run_row(run_path: Path) -> dict[str, Any]:
    standalone_report = _read_json(run_path) if run_path.is_file() else {}
    standalone_parent = run_path.parent if run_path.is_file() else run_path
    manifest = _read_json(standalone_parent / "manifest.json")
    h3 = _read_json(standalone_parent / "modular_response_h3_report.json")
    state_bw = _read_json(standalone_parent / "bw_state_derived_report.json")
    transition_selection = _read_json(standalone_parent / "transition_scale_selection_report.json")
    paper_3d_chart = _read_json(standalone_parent / "paper_3d_bulk_chart_report.json")
    h3_ensemble = _best_h3_ensemble_report(standalone_parent)
    if standalone_report.get("mode") == "h3_refit_seed_ensemble":
        h3_ensemble = standalone_report
    caps_to_h3 = _best_caps_to_h3_report(standalone_parent)
    if "S2_CAP_PROFILE_TO_H3_RECEIPT" in standalone_report:
        caps_to_h3 = standalone_report
    record_populated_h3 = _read_json(standalone_parent / "record_populated_h3_report.json")
    record_family_h3 = _read_json(standalone_parent / "record_family_h3_report.json")
    defect_cluster_h3 = _read_json(standalone_parent / "defect_cluster_h3_report.json")
    cmb = _best_cmb_lite_report(standalone_parent)
    if standalone_report.get("mode") == "cmb_lite_shape_comparison":
        cmb = standalone_report
    transfer = _read_json(standalone_parent / "cmb_transfer_report.json")
    camb_baseline = _read_json(standalone_parent / "camb_lcdm_baseline_report.json")
    screen_power = _read_json(standalone_parent / "oph_screen_power_report.json")
    screen_camb = _read_json(standalone_parent / "oph_screen_camb_report.json")
    maxent_green = _read_json(standalone_parent / "maxent_green_spectrum_report.json")
    repair_clock = _read_json(standalone_parent / "repair_clock_certificate_report.json")
    scalar_repair_semigroup = _read_json(standalone_parent / "scalar_repair_semigroup_report.json")
    fossil_spectrum = _read_json(standalone_parent / "fossil_spectrum_report.json")
    cmb_fossil_bridge = _read_json(standalone_parent / "cmb_fossil_bridge_report.json")
    inflation_cmb_bridge = _read_json(standalone_parent / "oph_inflation_cmb_bridge_report.json")
    inflation_certificates = _read_json(standalone_parent / "inflation_certificate_report.json")
    inflation_cmb_camb = _read_json(standalone_parent / "oph_inflation_cmb_camb_report.json")
    selector_elimination = _read_json(standalone_parent / "oph_cmb_selector_elimination_report.json")
    exact_cmb_camb = _read_json(standalone_parent / "oph_exact_cmb_camb_report.json")
    finite_clock_cmb_camb = _read_json(standalone_parent / "finite_repair_clock_cmb_camb_report.json")
    unique_prediction = _read_json(standalone_parent / "oph_unique_prediction_gate_report.json")
    cnb_neutrino = _read_json(standalone_parent / "oph_cnb_neutrino_report.json")
    finite_certificates = _read_json(standalone_parent / "finite_certificate_report.json")
    finite_certificate_manifest = _read_json(standalone_parent / "finite_certificate_manifest.json")
    scalar_quotient = _read_json(standalone_parent / "scalar_quotient_report.json")
    neutral_profile = _read_json(standalone_parent / "neutral_profile_audit_report.json") or _read_json(
        standalone_parent / "neutral_profile_audit.json"
    )
    prime_rank_sweep = _read_json(standalone_parent / "prime_geometric_rank_sweep_report.json")
    prime_rank_refinement = _read_json(standalone_parent / "prime_geometric_rank_refinement_report.json")
    parent_collar_ladder = _read_json(standalone_parent / "parent_collar_ladder_report.json")
    paired_b_a_parent = _read_json(standalone_parent / "paired_b_a_perturbation_report.json")
    b_a_parent = _read_json(standalone_parent / "b_a_parent_report.json")
    screen_capacity = _read_json(standalone_parent / "screen_capacity_closure_report.json")
    repair_scale = _read_json(standalone_parent / "repair_scale_closure_report.json")
    scale_compressed = _read_json(standalone_parent / "scale_compressed_repair_report.json")
    scale_compressed_cmb_camb = _read_json(standalone_parent / "scale_compressed_cmb_camb_report.json")
    cmb_derivation = _read_json(standalone_parent / "cmb_parameter_derivation_report.json")
    if standalone_report.get("mode") == "oph_screen_power_effective_theory_v0":
        screen_power = standalone_report
    if standalone_report.get("mode") == "oph_screen_camb_transfer_scaffold":
        screen_camb = standalone_report
    if standalone_report.get("mode") == "oph_maxent_green_screen_source_v0":
        maxent_green = standalone_report
    if standalone_report.get("mode") == "oph_repair_clock_kappa_audit_v0":
        repair_clock = standalone_report
    if standalone_report.get("mode") == "oph_scalar_repair_semigroup_gap_audit_v0":
        scalar_repair_semigroup = standalone_report
    if standalone_report.get("mode") == "oph_fossil_spectrum_time_resolved_diagnostic_v0":
        fossil_spectrum = standalone_report
    if standalone_report.get("mode") == "oph_cmb_fossil_bridge_diagnostic":
        cmb_fossil_bridge = standalone_report
    if standalone_report.get("mode") == "oph_inflation_cmb_bridge_v0":
        inflation_cmb_bridge = standalone_report
    if standalone_report.get("mode") == "oph_inflation_certificate_bundle_v0":
        inflation_certificates = standalone_report
    if standalone_report.get("mode") == "oph_inflation_cmb_camb_transfer_v0":
        inflation_cmb_camb = standalone_report
    if standalone_report.get("mode") == "oph_cmb_selector_elimination_v1_5":
        selector_elimination = standalone_report
    if standalone_report.get("mode") == "oph_exact_cmb_camb_transfer_v1":
        exact_cmb_camb = standalone_report
    if standalone_report.get("mode") == "finite_repair_clock_cmb_camb_transfer_v0":
        finite_clock_cmb_camb = standalone_report
    if standalone_report.get("mode") == "oph_unique_prediction_gate_v0_9":
        unique_prediction = standalone_report
    if standalone_report.get("mode") == "oph_cnb_neutrino_background_v0":
        cnb_neutrino = standalone_report
    if standalone_report.get("mode") == "oph_finite_cosmology_certificate_bundle_v0":
        finite_certificates = standalone_report
    if standalone_report.get("manifest_type") == "oph_finite_certificate_manifest":
        finite_certificate_manifest = standalone_report
    if standalone_report.get("mode") == "oph_scalar_geometric_quotient_report_v0":
        scalar_quotient = standalone_report
    if standalone_report.get("mode") == "neutral_distance_profile_audit_v0":
        neutral_profile = standalone_report
    if standalone_report.get("mode") == "prime_geometric_rank_sweep_v0":
        prime_rank_sweep = standalone_report
    if standalone_report.get("mode") == "prime_geometric_rank_refinement_v0":
        prime_rank_refinement = standalone_report
    if standalone_report.get("mode") == "oph_parent_collar_recovery_ladder_v0":
        parent_collar_ladder = standalone_report
    if standalone_report.get("mode") == "paired_cap_collar_perturb_resettle_B_A_parent_v0":
        paired_b_a_parent = standalone_report
    if standalone_report.get("mode") == "report_backed_finite_collar_B_A_parent_diagnostic_v0":
        b_a_parent = standalone_report
    if paired_b_a_parent and not b_a_parent:
        b_a_parent = paired_b_a_parent
    if standalone_report.get("mode") == "oph_screen_capacity_closure_v0":
        screen_capacity = standalone_report
    if standalone_report.get("mode") == "oph_repair_scale_closure_hypothesis_v0":
        repair_scale = standalone_report
    if standalone_report.get("mode") == "oph_scale_compressed_repair_round_branch_v0":
        scale_compressed = standalone_report
    if standalone_report.get("mode") == "oph_scale_compressed_cmb_camb_transfer_v0":
        scale_compressed_cmb_camb = standalone_report
    if standalone_report.get("mode") == "finite_lattice_cmb_parameter_derivation_audit_v0":
        cmb_derivation = standalone_report
    if not unique_prediction and inflation_cmb_bridge:
        unique_prediction = inflation_cmb_bridge.get("unique_prediction_gate_v0_9", {}) or {}
    if not selector_elimination and exact_cmb_camb:
        selector_elimination = exact_cmb_camb.get("selector_elimination_v1_5", {}) or {}
    if not selector_elimination and unique_prediction:
        selector_elimination = unique_prediction.get("selector_elimination_v1_5", {}) or {}
    boltzmann_inputs = _read_json(standalone_parent / "oph_boltzmann_input_report.json")
    finite_collar_boltzmann = _read_json(standalone_parent / "finite_collar_boltzmann_bundle_report.json")
    if standalone_report.get("mode") == "finite_collar_boltzmann_source_bundle_v0":
        finite_collar_boltzmann = standalone_report
    finite_collar_projection = _read_json(standalone_parent / "finite_collar_cmb_projection_report.json")
    if standalone_report.get("mode") == "finite_collar_cmb_projection_diagnostic_v0":
        finite_collar_projection = standalone_report
    oph_cmb = _read_json(standalone_parent / "oph_cmb_stress_report.json")
    cmb_anomaly = _read_json(standalone_parent / "cmb_anomaly_report.json")
    sync_gap = _read_json(standalone_parent / "sync_gap_report.json")
    hot_release = _read_json(standalone_parent / "hot_release_report.json")
    adiabaticity = _read_json(standalone_parent / "adiabaticity_report.json")
    h0s8 = _read_json(standalone_parent / "h0s8_branch_report.json")
    h0s8_lane8 = _read_json(standalone_parent / "h0s8_lane8_certificate_report.json")
    if standalone_report.get("mode") == "finite_screen_cmb_anomaly_diagnostics_v0":
        cmb_anomaly = standalone_report
    if standalone_report.get("mode") == "oph_low_k_synchronization_gap_audit_v0":
        sync_gap = standalone_report
    if standalone_report.get("mode") == "oph_hot_maxent_release_audit_v0":
        hot_release = standalone_report
    if standalone_report.get("mode") == "oph_same_boundary_adiabaticity_audit_v0":
        adiabaticity = standalone_report
    if standalone_report.get("mode") == "oph_h0_s8_branch_diagnostic_v0":
        h0s8 = standalone_report
    if standalone_report.get("mode") == "oph_h0_s8_lane8_certificate_stack_v0":
        h0s8_lane8 = standalone_report
    if not h0s8_lane8 and h0s8:
        h0s8_lane8 = h0s8.get("lane8_certificate_stack", {}) or {}
    galaxy = _read_json(standalone_parent / "galaxy_proxy_report.json")
    static_galaxy = _read_json(standalone_parent / "static_galaxy_measurement_report.json")
    cl = _read_json(standalone_parent / "cl_comparison_report.json")
    hol = _read_json(standalone_parent / "array_holonomy_report.json")
    timeline = _read_json(standalone_parent / "defect_timeline_report.json")
    interaction = _read_json(standalone_parent / "defect_interaction_report.json")
    particle = _read_json(standalone_parent / "particle_likeness_report.json")
    controlled_particle = _read_json(standalone_parent / "controlled_defect_particle_assay_report.json")
    defect_h3_worldlines = _read_json(standalone_parent / "defect_h3_worldlines_report.json")
    emergence = _read_json(standalone_parent / "emergence_status_report.json")
    bulk_proof = _read_json(standalone_parent / "bulk_proof_certificate_report.json")
    if standalone_report.get("mode") == "oph_3d_bulk_and_measurement_proof_certificate_v0":
        bulk_proof = standalone_report
    shape_summary = _read_json(standalone_parent / "shape_substrate_summary.json")
    shape_vertex = _read_json(standalone_parent / "shape_vertex_scattering_report.json")
    shape_cell = _read_json(standalone_parent / "shape_dodeca_cell_report.json")
    shape_settling = _read_json(standalone_parent / "shape_settling_report.json")
    shape_particle = _read_json(standalone_parent / "shape_loop_particle_report.json")
    if not shape_particle:
        shape_particle = _read_json(standalone_parent / "shape_particle_loop_report.json")
    shape_projection = _read_json(standalone_parent / "shape_screen_projection_report.json")
    shape_certificate = _read_json(standalone_parent / "shape_cmb_certificate_inputs.json")
    if standalone_report.get("mode") == "declared_shape_substrate_witness":
        shape_summary = standalone_report
    object_chart = _best_object_chart_report(standalone_parent)
    if standalone_report.get("mode") == "observer_chart_object_h3_population":
        object_chart = standalone_report
    observer_consensus = _read_json(standalone_parent / "observer_consensus_report.json")
    object_consensus = _read_json(standalone_parent / "object_consensus_report.json")
    neutral = _read_json(standalone_parent / "bulk_reconstruction_report.json")
    strict_neutral = _read_json(standalone_parent / "strict_neutral_bulk_report.json")
    if standalone_report.get("mode") == "strict_neutral_bulk_record_transition_audit":
        strict_neutral = standalone_report
    strict_neutral_object = _read_json(standalone_parent / "strict_neutral_object_bulk_report.json")
    if standalone_report.get("mode") == "strict_neutral_object_bulk_v0":
        strict_neutral_object = standalone_report

    h3_fit = h3.get("h3_fit", {}) if h3 else {}
    h3_chart_dimension = h3.get("h3_chart_dimension_debug", {}) if h3 else {}
    if h3 and not h3_chart_dimension:
        h3_chart_dimension = _h3_chart_dimension_from_fit(h3_fit)
    h3_chart_corr = h3_chart_dimension.get("correlation_dimension", {}) if isinstance(h3_chart_dimension, dict) else {}
    h3_chart_mle = h3_chart_dimension.get("local_mle_dimension", {}) if isinstance(h3_chart_dimension, dict) else {}
    s2 = h3.get("s2_boundary_control", {}) if h3 else {}
    controls = h3.get("control_fits", {}) if h3 else {}
    wrong = h3.get("wrong_scale_control_fits", {}) if h3 else {}
    wrong_audit = h3.get("wrong_scale_feature_audit", {}) if h3 else {}
    h3_stage_gates = h3.get("h3_response_stage_gates", {}) if h3 else {}
    cmb_fields = cmb.get("field_comparisons", {}) if cmb else {}
    cmb_best_name = cmb.get("best_shape_field") if cmb else None
    cmb_best = cmb_fields.get(cmb_best_name, {}) if cmb_best_name else {}
    cmb_best_positive_name = cmb.get("best_positive_shape_field") if cmb else None
    cmb_best_positive = cmb_fields.get(cmb_best_positive_name, {}) if cmb_best_positive_name else {}
    record_cmb = cmb_fields.get("record_signature", {})
    cmb_best_overlap = cmb_best.get("overlap_ell_physical_comparison", {}) if isinstance(cmb_best, dict) else {}
    cmb_best_overlap_name, cmb_best_overlap_real = _best_overlap_ell_field(cmb_fields)
    record_cmb_overlap = (
        record_cmb.get("overlap_ell_physical_comparison", {}) if isinstance(record_cmb, dict) else {}
    )
    cl_record = (cl.get("fields", {}) if cl else {}).get("record_signature", {})
    cl_record_control = cl_record.get("control_comparison", {})
    boltzmann_diagnostic_rows = (
        (boltzmann_inputs.get("diagnostic_repair_exchange", {}) or {}).get("rows", []) if boltzmann_inputs else []
    )
    neutral_dimension = neutral.get("neutral_dimension_report", {}) if neutral else {}
    blind_bulk = neutral.get("blind_observer_bulk_report", {}) if neutral else {}
    blind_dimension = blind_bulk.get("neutral_dimension_report", {}) if isinstance(blind_bulk, dict) else {}
    blind_low_rank = blind_bulk.get("low_rank_transition_chart_sweep", {}) if isinstance(blind_bulk, dict) else {}
    blind_group_sweep = blind_bulk.get("blind_feature_group_sweep", {}) if isinstance(blind_bulk, dict) else {}
    blind_record_rank3 = (
        blind_group_sweep.get("record_transition_rank3_report", {})
        if isinstance(blind_group_sweep, dict)
        else {}
    )
    blind_record_rank3_corr = (
        blind_record_rank3.get("correlation_dimension", {})
        if isinstance(blind_record_rank3, dict)
        else {}
    )
    blind_record_rank3_mle = (
        blind_record_rank3.get("local_mle_dimension", {})
        if isinstance(blind_record_rank3, dict)
        else {}
    )
    blind_low_rank_selected = (
        blind_low_rank.get("selected_rank_report", {}) if isinstance(blind_low_rank, dict) else {}
    )
    blind_low_rank_selected_corr = (
        blind_low_rank_selected.get("correlation_dimension", {})
        if isinstance(blind_low_rank_selected, dict)
        else {}
    )
    blind_low_rank_selected_mle = (
        blind_low_rank_selected.get("local_mle_dimension", {})
        if isinstance(blind_low_rank_selected, dict)
        else {}
    )
    blind_corr = blind_dimension.get("correlation_dimension", {}) if isinstance(blind_dimension, dict) else {}
    blind_mle = blind_dimension.get("local_mle_dimension", {}) if isinstance(blind_dimension, dict) else {}
    neutral_primary = neutral_dimension.get("primary_dimension", {}) if isinstance(neutral_dimension, dict) else {}
    neutral_corr = neutral_dimension.get("correlation_dimension", {}) if isinstance(neutral_dimension, dict) else {}
    neutral_mle = neutral_dimension.get("local_mle_dimension", {}) if isinstance(neutral_dimension, dict) else {}
    strict_neutral_dimension = strict_neutral.get("dimension", {}) if strict_neutral else {}
    strict_neutral_corr = (
        strict_neutral_dimension.get("correlation_dimension", {})
        if isinstance(strict_neutral_dimension, dict)
        else {}
    )
    strict_neutral_mle = (
        strict_neutral_dimension.get("local_mle_dimension", {})
        if isinstance(strict_neutral_dimension, dict)
        else {}
    )
    strict_neutral_model = strict_neutral.get("model_selection", {}) if strict_neutral else {}
    strict_neutral_leakage = strict_neutral.get("leakage", {}) if strict_neutral else {}
    strict_neutral_controls = strict_neutral.get("controls", {}) if strict_neutral else {}
    strict_neutral_control_rows = (
        strict_neutral_controls.get("rows", [])
        if isinstance(strict_neutral_controls.get("rows", []), list)
        else []
    )
    strict_neutral_object_dimension = strict_neutral_object.get("dimension", {}) if strict_neutral_object else {}
    strict_neutral_object_model = (
        strict_neutral_object.get("latent_geometry_selection", {}) if strict_neutral_object else {}
    )
    strict_neutral_object_leakage = strict_neutral_object.get("leakage", {}) if strict_neutral_object else {}
    state_bw_generator_audit = state_bw.get("generator_scale_audit", {}) if isinstance(state_bw, dict) else {}
    if bool(state_bw.get("direct_transition_automorphism", False)):
        # Older run bundles may contain generator-scale audit rows for this
        # lane, but direct-transition probes construct K from the declared
        # finite automorphism and intentionally ignore generator_scale. Treat
        # those archived audit rows as not applicable in aggregate reports.
        state_bw_generator_audit = {
            "enabled": False,
            "not_applicable": True,
            "claim_boundary": (
                "ignored during comparable-data aggregation: direct transition-response "
                "automorphism lanes do not use generator_scale"
            ),
        }
    object_chart_flags = _object_chart_robust_flags(object_chart)
    chart_level_lorentz = bool(
        emergence.get("CHART_LEVEL_CONFORMAL_LORENTZ_RECEIPT")
        or emergence.get("chart_level_conformal_lorentz_receipt")
        or emergence.get("conformal_h3_spatial_chart_receipt", False)
    )
    bw_automorphism_sanity = bool(
        emergence.get("BW_AUTOMORPHISM_SANITY_RECEIPT")
        or emergence.get("bw_automorphism_sanity_receipt")
        or (
            state_bw.get("direct_transition_automorphism", False)
            and state_bw.get("state_selected_2pi", False)
            and state_bw.get("correct_beats_controls", False)
            and transition_selection.get("primary_source") == "kms_collar_transport_response"
            and transition_selection.get("two_pi_selected", False)
            and not transition_selection.get("response_degenerate", False)
        )
    )
    endogenous_modular_generator_receipt = bool(
        emergence.get("ENDOGENOUS_MODULAR_GENERATOR_RECEIPT")
        or emergence.get("endogenous_modular_generator_receipt")
        or (
            state_bw.get("endogenous_modular_generator", False)
            and state_bw.get("state_selected_2pi", False)
            and state_bw.get("correct_beats_controls", False)
        )
    )
    support_visible_lorentz = bool(chart_level_lorentz and bw_automorphism_sanity)
    paper_chart_receipt = bool(
        paper_3d_chart.get("PAPER_THEOREM_3D_BULK_CHART_RECEIPT")
        or paper_3d_chart.get("paper_theorem_3d_bulk_chart_receipt")
        or emergence.get("PAPER_THEOREM_3D_BULK_CHART_RECEIPT")
        or emergence.get("paper_theorem_3d_bulk_chart_receipt")
    )
    record_h3_support_receipt = bool(record_family_h3.get("record_populated_h3_receipt", False))
    record_h3_bulk_candidate = bool(record_family_h3.get("record_family_h3_bulk_population_candidate", False))
    defect_h3_support_receipt = bool(defect_cluster_h3.get("record_populated_h3_receipt", False))
    support_visible_record_h3_population = bool(
        emergence.get("support_visible_record_h3_population_receipt")
        or (paper_chart_receipt and record_h3_bulk_candidate)
    )
    support_visible_defect_h3_population = bool(
        emergence.get("support_visible_defect_h3_population_receipt")
        or (
            paper_chart_receipt
            and defect_h3_support_receipt
            and bool(timeline.get("persistent_worldline_precursor_receipt", False))
        )
    )
    support_visible_h3_populated_bulk = bool(
        emergence.get("SUPPORT_VISIBLE_H3_POPULATED_BULK_RECEIPT")
        or emergence.get("support_visible_h3_populated_bulk_receipt")
        or support_visible_record_h3_population
    )
    neutral_profile_rows = (
        neutral_profile.get("profile_rows", []) if isinstance(neutral_profile.get("profile_rows", []), list) else []
    )
    neutral_profile_all = _profile_row(neutral_profile, "all_observer_visible")
    neutral_profile_scalar_response = _profile_row(neutral_profile, "scalar_response")
    neutral_profile_prime_geometric = _profile_row(neutral_profile, "prime_geometric_modular")
    neutral_profile_prime_control_quotient = _profile_row(
        neutral_profile,
        "prime_geometric_control_quotient",
    )
    neutral_profile_prime_rank3 = _profile_row(neutral_profile, "prime_geometric_rank3")
    neutral_profile_prime_rank8 = _profile_row(neutral_profile, "prime_geometric_rank8")
    neutral_profile_prime_control_quotient_rank3 = _profile_row(
        neutral_profile,
        "prime_geometric_control_quotient_rank3",
    )
    neutral_profile_prime_control_quotient_rank8 = _profile_row(
        neutral_profile,
        "prime_geometric_control_quotient_rank8",
    )
    neutral_profile_support_visible = _profile_row(neutral_profile, "support_visible_modular")

    if bulk_proof:
        object_bulk_population_receipt = bool(
            bulk_proof.get("theorem_assisted_h3_nonboundary_population_established", False)
            or bulk_proof.get("bulk_3d_established_theorem_assisted", False)
        )
        theorem_assisted_h3_object_preview_receipt = bool(
            bulk_proof.get("theorem_assisted_h3_object_preview_established", False)
        )
        object_h3_nonboundary_population_receipt = bool(
            bulk_proof.get("theorem_assisted_h3_nonboundary_population_established", False)
        )
        paper_theorem_assisted_h3_populated_chart_receipt = bool(
            bulk_proof.get("theorem_assisted_h3_populated_chart_established", False)
        )
        paper_theorem_assisted_h3_chart_precursor_receipt = bool(
            bulk_proof.get("theorem_assisted_h3_object_preview_established", False)
        )
    else:
        object_bulk_population_receipt = bool(
            emergence.get("OBJECT_BULK_POPULATION_RECEIPT")
            or emergence.get("object_bulk_population_receipt")
        )
        theorem_assisted_h3_object_preview_receipt = bool(
            emergence.get("THEOREM_ASSISTED_H3_OBJECT_PREVIEW_RECEIPT")
            or emergence.get("theorem_assisted_h3_object_preview_receipt")
            or emergence.get("observer_chart_object_h3_receipt")
        )
        object_h3_nonboundary_population_receipt = bool(
            emergence.get("OBJECT_H3_NONBOUNDARY_POPULATION_RECEIPT")
            or emergence.get("object_h3_nonboundary_population_receipt")
            or emergence.get("observer_chart_bulk_population_receipt")
        )
        paper_theorem_assisted_h3_populated_chart_receipt = bool(
            emergence.get("PAPER_THEOREM_ASSISTED_H3_POPULATED_CHART_RECEIPT")
            or emergence.get("paper_theorem_assisted_h3_populated_chart_receipt")
        )
        paper_theorem_assisted_h3_chart_precursor_receipt = bool(
            emergence.get("PAPER_THEOREM_ASSISTED_H3_CHART_PRECURSOR_RECEIPT")
            or emergence.get("paper_theorem_assisted_h3_chart_precursor_receipt")
        )

    row = {
        "run_path": str(run_path),
        "run_id": manifest.get("run_id") or run_path.name,
        "name": manifest.get("name"),
        "patch_count": manifest.get("patch_count") or (cl or {}).get("point_count"),
        "has_comparable_data": any(
            (
                state_bw,
                transition_selection,
                paper_3d_chart,
                record_populated_h3,
                record_family_h3,
                defect_cluster_h3,
                emergence,
                h3_ensemble,
                caps_to_h3,
                h3,
                cmb,
                transfer,
                camb_baseline,
                screen_power,
                screen_camb,
                maxent_green,
                repair_clock,
                scalar_repair_semigroup,
                fossil_spectrum,
                cmb_fossil_bridge,
                inflation_cmb_bridge,
                inflation_certificates,
                inflation_cmb_camb,
                selector_elimination,
                exact_cmb_camb,
                finite_clock_cmb_camb,
                unique_prediction,
                cnb_neutrino,
                finite_certificates,
                finite_certificate_manifest,
                scalar_quotient,
                neutral_profile,
                prime_rank_sweep,
                prime_rank_refinement,
                parent_collar_ladder,
                b_a_parent,
                screen_capacity,
                repair_scale,
                scale_compressed,
                scale_compressed_cmb_camb,
                cmb_derivation,
                boltzmann_inputs,
                finite_collar_boltzmann,
                finite_collar_projection,
                oph_cmb,
                cmb_anomaly,
                sync_gap,
                hot_release,
                adiabaticity,
                h0s8,
                galaxy,
                static_galaxy,
                cl,
                hol,
                timeline,
                interaction,
                particle,
                controlled_particle,
                defect_h3_worldlines,
                bulk_proof,
                shape_summary,
                shape_vertex,
                shape_cell,
                shape_settling,
                shape_particle,
                shape_projection,
                shape_certificate,
                strict_neutral,
                strict_neutral_object,
            )
        ),
        "bulk_3d_established": bool(
            (
                bulk_proof
                and (
                    bulk_proof.get("bulk_3d_established_theorem_assisted", False)
                    or bulk_proof.get("strict_neutral_third_person_bulk_established", False)
                )
            )
            or (
                not bulk_proof
                and emergence.get("bulk_3d_established", False)
                and (
                    emergence.get("OBJECT_H3_NONBOUNDARY_POPULATION_RECEIPT", False)
                    or emergence.get("observer_chart_bulk_population_receipt", False)
                    or emergence.get("strict_blind_observer_bulk_receipt", False)
                    or emergence.get("neutral_bulk_3d_established", False)
                )
            )
        ),
        "bulk_proof_certificate_written": bool(bulk_proof),
        "bulk_proof_chart_level_3p1": bool(
            bulk_proof.get("chart_level_3p1_lorentz_kinematics_established", False)
        ),
        "bulk_proof_theorem_assisted_h3_populated_chart": bool(
            bulk_proof.get("theorem_assisted_h3_populated_chart_established", False)
        ),
        "bulk_proof_theorem_assisted_h3_object_preview": bool(
            bulk_proof.get("theorem_assisted_h3_object_preview_established", False)
        ),
        "bulk_proof_theorem_assisted_h3_nonboundary_population": bool(
            bulk_proof.get("theorem_assisted_h3_nonboundary_population_established", False)
        ),
        "bulk_proof_strict_neutral_3d_bulk": bool(
            bulk_proof.get("strict_neutral_third_person_bulk_established", False)
        ),
        "bulk_proof_screen_cmb_proxy": bool(bulk_proof.get("screen_cmb_proxy_available", False)),
        "bulk_proof_physical_cmb_prediction": bool(bulk_proof.get("physical_cmb_prediction", False)),
        "bulk_proof_production_particle_matter": bool(
            bulk_proof.get("production_particle_matter_receipt", False)
        ),
        "support_visible_lorentz_3p1_kinematics_receipt": support_visible_lorentz,
        "chart_level_conformal_lorentz_receipt": chart_level_lorentz,
        "bw_automorphism_sanity_receipt": bw_automorphism_sanity,
        "endogenous_modular_generator_receipt": endogenous_modular_generator_receipt,
        "object_bulk_population_receipt": object_bulk_population_receipt,
        "theorem_assisted_h3_object_preview_receipt": theorem_assisted_h3_object_preview_receipt,
        "object_h3_nonboundary_population_receipt": object_h3_nonboundary_population_receipt,
        "paper_theorem_assisted_h3_populated_chart_receipt": paper_theorem_assisted_h3_populated_chart_receipt,
        "paper_theorem_assisted_h3_chart_precursor_receipt": paper_theorem_assisted_h3_chart_precursor_receipt,
        "paper_theorem_3d_bulk_chart_receipt": paper_chart_receipt,
        "paper_theorem_object_populated_chart_precursor_receipt": bool(
            paper_3d_chart.get("paper_theorem_object_populated_chart_precursor_receipt")
            or emergence.get("paper_theorem_object_populated_chart_precursor_receipt")
        ),
        "paper_theorem_neutral_populated_bulk_receipt": bool(
            paper_3d_chart.get("paper_theorem_neutral_populated_bulk_receipt")
            or emergence.get("paper_theorem_neutral_populated_bulk_receipt")
        ),
        "paper_theorem_h3_spatial_dimension": (
            paper_3d_chart.get("h3_spatial_dimension_from_boost_orbit")
            if paper_3d_chart
            else emergence.get("paper_theorem_3d_chart_dimension")
        ),
        "paper_theorem_finite_point_cloud_dimension_estimator_used": paper_3d_chart.get(
            "finite_point_cloud_dimension_estimator_used"
        )
        if paper_3d_chart
        else None,
        "record_populated_h3_spatial_receipt": bool(
            emergence.get("record_populated_h3_spatial_receipt")
            or record_populated_h3.get("record_populated_h3_receipt")
        ),
        "record_family_h3_support_receipt": record_h3_support_receipt,
        "record_family_h3_bulk_population_candidate": record_h3_bulk_candidate,
        "record_family_h3_support_count": record_family_h3.get("support_count") if record_family_h3 else None,
        "record_family_h3_cap_count": record_family_h3.get("cap_count") if record_family_h3 else None,
        "record_family_h3_median_residual": _nested(record_family_h3, "h3_fit", "median_residual"),
        "record_family_h3_s2_median_residual": _nested(
            record_family_h3,
            "s2_boundary_control",
            "median_residual",
        ),
        "record_family_h3_shuffled_median_residual": _nested(
            record_family_h3,
            "shuffled_cap_response_control",
            "median_residual",
        ),
        "defect_cluster_h3_support_receipt": defect_h3_support_receipt,
        "defect_cluster_h3_support_count": defect_cluster_h3.get("support_count") if defect_cluster_h3 else None,
        "defect_cluster_h3_cap_count": defect_cluster_h3.get("cap_count") if defect_cluster_h3 else None,
        "defect_cluster_h3_median_residual": _nested(defect_cluster_h3, "h3_fit", "median_residual"),
        "defect_cluster_h3_s2_median_residual": _nested(
            defect_cluster_h3,
            "s2_boundary_control",
            "median_residual",
        ),
        "defect_cluster_h3_shuffled_median_residual": _nested(
            defect_cluster_h3,
            "shuffled_cap_response_control",
            "median_residual",
        ),
        "support_visible_record_h3_population_receipt": support_visible_record_h3_population,
        "support_visible_defect_h3_population_receipt": support_visible_defect_h3_population,
        "support_visible_h3_populated_bulk_receipt": support_visible_h3_populated_bulk,
        "screen_proxy_cmb_receipt": bool(
            emergence.get("SCREEN_PROXY_CMB_RECEIPT") or emergence.get("screen_proxy_cmb_receipt", False)
        ),
        "physical_cmb_prediction": bool(cmb.get("physical_cmb_prediction", False)),
        "cmb_anomaly_written": bool(cmb_anomaly),
        "cmb_anomaly_primary_field": _nested(cmb_anomaly, "aggregate", "primary_field"),
        "cmb_anomaly_best_low_power_field": _nested(
            cmb_anomaly,
            "aggregate",
            "best_low_power_suppression_field",
        ),
        "cmb_anomaly_best_low_power_abs_fraction": _nested(
            cmb_anomaly,
            "aggregate",
            "best_low_power_abs_fraction",
        ),
        "cmb_anomaly_best_large_angle_field": _nested(
            cmb_anomaly,
            "aggregate",
            "best_large_angle_suppression_field",
        ),
        "cmb_anomaly_best_S_1_2_scalar_proxy": _nested(
            cmb_anomaly,
            "aggregate",
            "best_S_1_2_scalar_proxy",
        ),
        "cmb_anomaly_best_parity_field": _nested(cmb_anomaly, "aggregate", "best_parity_asymmetry_field"),
        "cmb_anomaly_best_parity_log_abs_deviation": _nested(
            cmb_anomaly,
            "aggregate",
            "best_parity_log_abs_deviation",
        ),
        "cmb_anomaly_best_tilt_field": _nested(cmb_anomaly, "aggregate", "best_tilt_field"),
        "cmb_anomaly_best_eta_R_estimate": _nested(cmb_anomaly, "aggregate", "best_eta_R_estimate"),
        "cmb_anomaly_best_n_s_proxy": _nested(cmb_anomaly, "aggregate", "best_n_s_proxy"),
        "cmb_anomaly_low_power_suppressed_vs_controls_count": _nested(
            cmb_anomaly,
            "aggregate",
            "low_power_suppressed_vs_controls_count",
        ),
        "cmb_anomaly_large_angle_suppressed_vs_controls_count": _nested(
            cmb_anomaly,
            "aggregate",
            "large_angle_suppressed_vs_controls_count",
        ),
        "cmb_anomaly_parity_more_asymmetric_than_controls_count": _nested(
            cmb_anomaly,
            "aggregate",
            "parity_more_asymmetric_than_controls_count",
        ),
        "cmb_anomaly_planck_tilt_compatible_proxy_count": _nested(
            cmb_anomaly,
            "aggregate",
            "planck_tilt_compatible_proxy_count",
        ),
        "cmb_anomaly_total_entropy_capacity": _nested(cmb_anomaly, "screen_capacity", "total_entropy_capacity"),
        "cmb_anomaly_ell_sqrt_capacity_proxy": _nested(
            cmb_anomaly,
            "screen_capacity",
            "ell_sqrt_patch_capacity_proxy",
        ),
        "cmb_anomaly_physical_cmb_prediction": bool(cmb_anomaly.get("physical_cmb_prediction", False)),
        "cmb_lite_report_source": cmb.get("_report_path"),
        "state_bw_written": bool(state_bw),
        "state_bw_receipt": bool(state_bw.get("BW_KMS_BRANCH_INSTANTIATION_RECEIPT", False)),
        "state_bw_state_mode": state_bw.get("state_mode"),
        "state_bw_endogenous_generator": state_bw.get("endogenous_modular_generator"),
        "state_bw_declared_cap_flow_generator": bool(
            state_bw.get("declared_cap_flow_generator")
            or state_bw.get("state_mode") in DECLARED_CAP_FLOW_STATE_MODES
        ),
        "state_bw_declared_response_density": bool(
            state_bw.get("declared_transition_response_density")
            or state_bw.get("state_mode") == "transition_response_density_log"
        ),
        "state_bw_density_log_calibration_receipt": bool(
            state_bw.get("TRANSITION_RESPONSE_DENSITY_LOG_CALIBRATION_RECEIPT", False)
        ),
        "state_bw_direct_transition_automorphism": bool(
            state_bw.get("direct_transition_automorphism")
            or state_bw.get("state_mode") == "transition_response_unitary"
        ),
        "state_bw_normalization_declared": state_bw.get("normalization_declared"),
        "state_bw_generator_scale": state_bw.get("generator_scale"),
        "state_bw_selected_scale_label": state_bw.get("state_selected_scale_label"),
        "state_bw_selected_2pi": state_bw.get("state_selected_2pi"),
        "state_bw_correct_beats_controls": state_bw.get("correct_beats_controls"),
        "state_bw_not_applicable_controls": state_bw.get("not_applicable_controls"),
        "state_bw_target_scale_control_degenerate": bool(
            state_bw.get("target_scale_control_degenerate", False)
        ),
        "state_bw_degenerate_target_scale_controls": state_bw.get("degenerate_target_scale_controls") or [],
        "state_bw_bulk_gate": emergence.get("state_derived_bw_bulk_gate"),
        "state_bw_median": state_bw.get("median"),
        "state_bw_best_control": state_bw.get("best_control"),
        "state_bw_best_control_median": state_bw.get("best_control_median"),
        "state_bw_wrong_1x_median": _nested(state_bw.get("controls", {}), "wrong_1x_normalization", "median"),
        "state_bw_wrong_pi_median": _nested(state_bw.get("controls", {}), "wrong_pi_normalization", "median"),
        "state_bw_wrong_4pi_median": _nested(state_bw.get("controls", {}), "wrong_4pi_normalization", "median"),
        "state_bw_no_modular_flow_median": _nested(state_bw.get("controls", {}), "no_modular_flow", "median"),
        "state_bw_shuffled_observables_median": _nested(
            state_bw.get("controls", {}),
            "shuffled_observables",
            "median",
        ),
        "state_bw_generator_audit_enabled": bool(state_bw_generator_audit.get("enabled", False)),
        "state_bw_generator_audit_best_label": state_bw_generator_audit.get("best_label"),
        "state_bw_generator_audit_best_scale": state_bw_generator_audit.get("best_scale"),
        "state_bw_generator_audit_best_score": state_bw_generator_audit.get("best_score"),
        "state_bw_generator_audit_configured_score": state_bw_generator_audit.get("configured_score"),
        "state_bw_generator_audit_configured_is_best": state_bw_generator_audit.get("configured_is_best"),
        "state_bw_generator_audit_two_pi_is_best": state_bw_generator_audit.get("two_pi_generator_is_best"),
        "state_bw_generator_audit_diagnosis": state_bw_generator_audit.get("diagnosis"),
        "cmb_transfer_diagnostic_receipt": transfer.get("diagnostic_transfer_receipt"),
        "cmb_transfer_train_patch_count": transfer.get("train_patch_count"),
        "cmb_transfer_test_patch_count": transfer.get("test_patch_count"),
        "cmb_transfer_train_shape_correlation": _nested(transfer, "train", "mean_shape_correlation"),
        "cmb_transfer_train_normalized_rmse": _nested(transfer, "train", "mean_normalized_rmse"),
        "cmb_transfer_test_shape_correlation": _nested(transfer, "test", "mean_shape_correlation"),
        "cmb_transfer_test_normalized_rmse": _nested(transfer, "test", "mean_normalized_rmse"),
        "cmb_transfer_max_control_test_shape_correlation": transfer.get("max_control_test_shape_correlation"),
        "cmb_transfer_test_control_gap": transfer.get("test_vs_max_control_shape_correlation_gap"),
        "cmb_transfer_bootstrap_count": _nested(transfer, "bootstrap", "bootstrap_count"),
        "cmb_transfer_bootstrap_corr_p05": _nested(transfer.get("bootstrap", {}), "test_shape_correlation", "p05"),
        "cmb_transfer_bootstrap_corr_p95": _nested(transfer.get("bootstrap", {}), "test_shape_correlation", "p95"),
        "cmb_transfer_bootstrap_rmse_p05": _nested(transfer.get("bootstrap", {}), "test_normalized_rmse", "p05"),
        "cmb_transfer_bootstrap_rmse_p95": _nested(transfer.get("bootstrap", {}), "test_normalized_rmse", "p95"),
        "cmb_transfer_physical_cmb_prediction": bool(transfer.get("physical_cmb_prediction", False)),
        "camb_lcdm_written": bool(camb_baseline),
        "camb_lcdm_boltzmann_receipt": bool(camb_baseline.get("CDM_LIMIT_BOLTZMANN_RECEIPT", False)),
        "camb_lcdm_oph_anomaly_module_ready": bool(camb_baseline.get("oph_anomaly_module_ready", False)),
        "camb_lcdm_physical_cmb_prediction": bool(camb_baseline.get("physical_cmb_prediction", False)),
        "camb_lcdm_shape_correlation": _nested(camb_baseline, "comparison", "shape_correlation"),
        "camb_lcdm_normalized_rmse": _nested(camb_baseline, "comparison", "normalized_rmse"),
        "camb_lcdm_amplitude_fit_chi2_per_bin": _nested(
            camb_baseline, "comparison", "amplitude_fit_chi2_per_bin"
        ),
        "camb_lcdm_best_fit_column_chi2_per_bin": _nested(
            camb_baseline, "comparison", "best_fit_column_chi2_per_bin"
        ),
        "camb_lcdm_first_peak_ell": _nested(camb_baseline, "comparison", "first_peak_ell"),
        "camb_lcdm_benchmark_first_peak_ell": _nested(camb_baseline, "comparison", "benchmark_first_peak_ell"),
        "camb_lcdm_mean_abs_fractional_error": _nested(
            camb_baseline, "comparison", "mean_absolute_fractional_error"
        ),
        "oph_screen_power_written": bool(screen_power),
        "oph_screen_power_source_run_count": screen_power.get("source_run_count") if screen_power else None,
        "oph_screen_power_fit_row_count": screen_power.get("fit_row_count") if screen_power else None,
        "oph_screen_power_available_fit_count": _nested(screen_power, "aggregate", "available_fit_count"),
        "oph_screen_power_best_planck_eta_field": _nested(
            screen_power,
            "aggregate",
            "best_planck_eta_diagnostic_field",
        ),
        "oph_screen_power_simulator_primordial_ready": bool(
            screen_power.get("simulator_primordial_reference_ready", False)
        ),
        "oph_screen_power_reference_source": screen_power.get("primordial_reference_source")
        if screen_power
        else None,
        "oph_screen_power_reference_eta_R": _nested(screen_power, "reference_screen_parameters", "eta_R"),
        "oph_screen_power_reference_n_s": _nested(screen_power, "reference_screen_parameters", "n_s_proxy"),
        "oph_screen_power_record_eta_R": _nested(
            screen_power,
            "aggregate",
            "field_summary",
            "record_signature",
            "median_eta_R",
        ),
        "oph_screen_power_stable_eta_R": _nested(
            screen_power,
            "aggregate",
            "field_summary",
            "stable_count",
            "median_eta_R",
        ),
        "oph_screen_power_physical_cmb_prediction": bool(screen_power.get("physical_cmb_prediction", False)),
        "oph_screen_camb_written": bool(screen_camb),
        "oph_screen_camb_receipt": bool(screen_camb.get("screen_camb_transfer_receipt", False)),
        "oph_screen_camb_reference_source": _nested(screen_camb, "screen_input", "reference_source"),
        "oph_screen_camb_simulator_eta_ready": _nested(screen_camb, "screen_input", "simulator_eta_R_ready"),
        "oph_screen_camb_eta_R": _nested(screen_camb, "screen_input", "eta_R"),
        "oph_screen_camb_n_s": _nested(screen_camb, "screen_input", "n_s_proxy"),
        "oph_screen_camb_shape_correlation": _nested(screen_camb, "comparison", "shape_correlation"),
        "oph_screen_camb_normalized_rmse": _nested(screen_camb, "comparison", "normalized_rmse"),
        "oph_screen_camb_amplitude_fit_chi2_per_bin": _nested(
            screen_camb,
            "comparison",
            "amplitude_fit_chi2_per_bin",
        ),
        "oph_screen_camb_first_peak_ell": _nested(screen_camb, "comparison", "first_peak_ell"),
        "oph_screen_camb_benchmark_first_peak_ell": _nested(
            screen_camb,
            "comparison",
            "benchmark_first_peak_ell",
        ),
        "oph_screen_camb_mean_abs_fractional_error": _nested(
            screen_camb,
            "comparison",
            "mean_absolute_fractional_error",
        ),
        "oph_screen_camb_physical_cmb_prediction": bool(screen_camb.get("physical_cmb_prediction", False)),
        "oph_maxent_green_written": bool(maxent_green),
        "oph_maxent_green_receipt": bool(maxent_green.get("MAXENT_GREEN_SOURCE_RECEIPT", False)),
        "oph_maxent_green_flat_green_receipt": bool(
            _nested(maxent_green, "maxent_inverse_laplacian", "eta0_flat_D_ell_receipt")
        ),
        "oph_maxent_green_ir_receipt": bool(
            _nested(maxent_green, "selector_elimination_v1_5", "theorem_side_receipt")
        ),
        "oph_maxent_green_source_packet_audit": bool(
            _nested(maxent_green, "selector_elimination_v1_5", "source_packet_audit_receipt")
        ),
        "oph_maxent_green_repair_clock_certificate": bool(
            _nested(maxent_green, "fractional_repair_tilt", "repair_clock_certificate")
        ),
        "oph_maxent_green_bandlimit_for_ir": bool(
            _nested(maxent_green, "finite_regulator", "bandlimit_for_ir_receipt")
        ),
        "oph_maxent_green_bandlimit_for_requested_ell": bool(
            _nested(maxent_green, "finite_regulator", "bandlimit_for_requested_ell_receipt")
        ),
        "oph_maxent_green_patch_count": _nested(maxent_green, "finite_regulator", "patch_count"),
        "oph_maxent_green_ell_max": _nested(maxent_green, "screen_spectrum", "ell_max"),
        "oph_maxent_green_eta_R": _nested(maxent_green, "fractional_repair_tilt", "eta_R"),
        "oph_maxent_green_n_s": _nested(maxent_green, "fractional_repair_tilt", "n_s"),
        "oph_maxent_green_fit_eta_R_error": _nested(
            maxent_green,
            "fractional_repair_tilt",
            "fit_eta_R_abs_error",
        ),
        "oph_maxent_green_q_IR": _nested(maxent_green, "selector_elimination_v1_5", "q_IR"),
        "oph_maxent_green_ell_IR": _nested(maxent_green, "selector_elimination_v1_5", "ell_IR"),
        "oph_maxent_green_N_frz_proxy": _nested(maxent_green, "selector_elimination_v1_5", "N_frz_proxy"),
        "oph_maxent_green_mean_F_IR_ell2_29": _nested(maxent_green, "screen_spectrum", "mean_F_IR_ell2_29"),
        "oph_maxent_green_F_IR_ell2": _nested(maxent_green, "screen_spectrum", "F_IR_ell2"),
        "oph_maxent_green_F_IR_ell32": _nested(maxent_green, "screen_spectrum", "F_IR_ell32"),
        "oph_maxent_green_finite_lattice_derived": bool(maxent_green.get("finite_lattice_derived", False)),
        "oph_maxent_green_physical_cmb_prediction": bool(maxent_green.get("physical_cmb_prediction", False)),
        "repair_clock_written": bool(repair_clock),
        "repair_clock_certificate": bool(repair_clock.get("repair_clock_certificate", False)),
        "repair_clock_finite_certificate": bool(repair_clock.get("finite_repair_clock_certificate", False)),
        "repair_clock_eta_finite_lattice_derived": bool(
            repair_clock.get("eta_R_finite_lattice_derived", False)
        ),
        "repair_clock_physical_cmb_prediction": bool(repair_clock.get("physical_cmb_prediction", False)),
        "repair_clock_candidate_run_count": _nested(repair_clock, "inputs", "candidate_run_count"),
        "repair_clock_estimator_count": _nested(repair_clock, "summary", "estimator_count"),
        "repair_clock_eligible_estimator_count": _nested(repair_clock, "summary", "eligible_estimator_count"),
        "repair_clock_passed_estimator_count": _nested(repair_clock, "summary", "passed_estimator_count"),
        "repair_clock_median_kappa_rep": _nested(repair_clock, "summary", "median_kappa_rep_estimate"),
        "repair_clock_median_eta_R": _nested(repair_clock, "summary", "median_eta_R_estimate"),
        "repair_clock_median_n_s": _nested(repair_clock, "summary", "median_n_s_estimate"),
        "repair_clock_target_kappa_rep": _nested(repair_clock, "target", "required_kappa_rep"),
        "repair_clock_target_eta_R": _nested(repair_clock, "target", "required_eta_R"),
        "repair_clock_cycle_time_normalization_declared": bool(
            _nested(repair_clock, "inputs", "cycle_time_normalization_declared")
        ),
        "repair_clock_blocker_count": len(repair_clock.get("blockers", [])) if repair_clock else None,
        "scalar_repair_semigroup_written": bool(scalar_repair_semigroup),
        "scalar_repair_semigroup_target_receipt": bool(
            scalar_repair_semigroup.get("SEMIGROUP_TARGET_RECEIPT", False)
        ),
        "scalar_repair_semigroup_repair_clock_certificate": bool(
            scalar_repair_semigroup.get("repair_clock_certificate", False)
        ),
        "scalar_repair_semigroup_eligible_for_certificate": bool(
            scalar_repair_semigroup.get("eligible_for_repair_clock_certificate", False)
        ),
        "scalar_repair_semigroup_finite_lattice_derived": bool(
            scalar_repair_semigroup.get("finite_lattice_derived", False)
        ),
        "scalar_repair_semigroup_source": scalar_repair_semigroup.get("source"),
        "scalar_repair_semigroup_dimension": scalar_repair_semigroup.get("dimension"),
        "scalar_repair_semigroup_centered_dim": scalar_repair_semigroup.get("centered_subspace_dimension"),
        "scalar_repair_semigroup_kappa_rep": _nested(
            scalar_repair_semigroup,
            "semigroup",
            "kappa_rep_estimate",
        ),
        "scalar_repair_semigroup_eta_R": _nested(
            scalar_repair_semigroup,
            "semigroup",
            "eta_R_estimate",
        ),
        "scalar_repair_semigroup_n_s": _nested(
            scalar_repair_semigroup,
            "semigroup",
            "n_s_estimate",
        ),
        "scalar_repair_semigroup_centered_gap": _nested(
            scalar_repair_semigroup,
            "semigroup",
            "centered_gap",
        ),
        "scalar_repair_semigroup_controls_passed": bool(
            scalar_repair_semigroup.get("semigroup_controls_passed", False)
        ),
        "scalar_repair_semigroup_transition_required_step_time": _nested(
            scalar_repair_semigroup,
            "transition_matrix_certificate",
            "required_repair_step_time_for_kappa_e",
        ),
        "scalar_repair_semigroup_transition_primary_lambda_2": _nested(
            scalar_repair_semigroup,
            "transition_matrix_certificate",
            "primary_lambda_2",
        ),
        "scalar_repair_semigroup_transition_clock_certified": bool(
            _nested(
                scalar_repair_semigroup,
                "transition_matrix_certificate",
                "clock_normalization_certified",
            )
        ),
        "fossil_spectrum_written": bool(fossil_spectrum),
        "fossil_spectrum_near_scale_invariant_transient": bool(
            fossil_spectrum.get("near_scale_invariant_transient", False)
        ),
        "fossil_spectrum_best_beats_controls": bool(fossil_spectrum.get("best_beats_same_field_controls", False)),
        "fossil_spectrum_best_field": _nested(fossil_spectrum, "best_target_closeness_diagnostic", "field"),
        "fossil_spectrum_best_cycle": _nested(fossil_spectrum, "best_target_closeness_diagnostic", "cycle"),
        "fossil_spectrum_best_eta_R": _nested(fossil_spectrum, "best_target_closeness_diagnostic", "eta_R"),
        "fossil_spectrum_best_n_s": _nested(fossil_spectrum, "best_target_closeness_diagnostic", "n_s"),
        "fossil_spectrum_best_abs_eta_delta": _nested(
            fossil_spectrum,
            "best_target_closeness_diagnostic",
            "abs_eta_R_delta_to_planck",
        ),
        "fossil_spectrum_best_control_abs_eta_delta": fossil_spectrum.get(
            "best_same_field_control_delta_to_planck"
        )
        if fossil_spectrum
        else None,
        "fossil_spectrum_freezeout_cycle": _nested(fossil_spectrum, "cycle_markers", "freezeout_cycle"),
        "fossil_spectrum_phi_zero_cycle": _nested(fossil_spectrum, "cycle_markers", "phi_zero_cycle"),
        "fossil_spectrum_phi_half_cycle": _nested(fossil_spectrum, "cycle_markers", "phi_half_cycle"),
        "fossil_spectrum_physical_cmb_prediction": bool(fossil_spectrum.get("physical_cmb_prediction", False)),
        "cmb_fossil_bridge_written": bool(cmb_fossil_bridge),
        "cmb_fossil_bridge_receipt": bool(cmb_fossil_bridge.get("receipt")),
        "cmb_fossil_bridge_physical_cmb_prediction": bool(
            cmb_fossil_bridge.get("physical_cmb_prediction", False)
        ),
        "cmb_fossil_bridge_shape_correlation": _nested(
            cmb_fossil_bridge,
            "benchmark_score",
            "shape_correlation",
        ),
        "cmb_fossil_bridge_normalized_rmse": _nested(
            cmb_fossil_bridge,
            "benchmark_score",
            "normalized_rmse",
        ),
        "cmb_fossil_bridge_eta": _nested(cmb_fossil_bridge, "parameters", "eta"),
        "cmb_fossil_bridge_q_ir": _nested(cmb_fossil_bridge, "parameters", "q_ir"),
        "cmb_fossil_bridge_ell_ir": _nested(cmb_fossil_bridge, "parameters", "ell_ir"),
        "oph_inflation_cmb_bridge_written": bool(inflation_cmb_bridge),
        "oph_inflation_cmb_bridge_physical_cmb_prediction": bool(
            inflation_cmb_bridge.get("physical_cmb_prediction", False)
        ),
        "oph_inflation_cmb_n_s": _nested(inflation_cmb_bridge, "screen_spectrum_prediction", "n_s"),
        "oph_inflation_cmb_theta_OPH": _nested(
            inflation_cmb_bridge, "screen_spectrum_prediction", "theta_OPH"
        ),
        "oph_inflation_cmb_A_zeta": _nested(inflation_cmb_bridge, "screen_spectrum_prediction", "A_zeta"),
        "oph_inflation_cmb_n_s_pull": _nested(
            inflation_cmb_bridge, "screen_spectrum_prediction", "n_s_pull_vs_planck"
        ),
        "oph_inflation_cmb_Omega_K": _nested(
            inflation_cmb_bridge, "flat_sector_selection", "selected_Omega_K"
        ),
        "oph_inflation_cmb_Omega_A0": _nested(
            inflation_cmb_bridge, "flat_sector_selection", "Omega_A0_residual"
        ),
        "oph_inflation_cmb_rho_A_over_rho_b": _nested(
            inflation_cmb_bridge, "flat_sector_selection", "rho_A_over_rho_b"
        ),
        "oph_inflation_cmb_v04_available": _nested(
            inflation_cmb_bridge, "cmb_success_ladder", "diagnostic_cmb_data_available"
        ),
        "oph_inflation_cmb_v04_q_IR": _nested(
            inflation_cmb_bridge,
            "cmb_success_ladder",
            "core_numbers",
            "v0_2_IR_bestfit_q_IR",
        ),
        "oph_inflation_cmb_v04_ell_IR": _nested(
            inflation_cmb_bridge,
            "cmb_success_ladder",
            "core_numbers",
            "v0_2_IR_bestfit_ell_IR",
        ),
        "oph_inflation_cmb_v04_camb_lcdm_lowell_chi2": _nested(
            inflation_cmb_bridge,
            "cmb_success_ladder",
            "core_numbers",
            "v0_3_camb_lowell_LCDM_chi2_ell2_29",
        ),
        "oph_inflation_cmb_v04_camb_oph_ir_lowell_chi2": _nested(
            inflation_cmb_bridge,
            "cmb_success_ladder",
            "core_numbers",
            "v0_3_camb_lowell_IR_bestfit_chi2_ell2_29",
        ),
        "oph_inflation_cmb_v04_lcdm_roe_pte": _nested(
            inflation_cmb_bridge,
            "cmb_success_ladder",
            "core_numbers",
            "v0_4_LCDM_PTE_R_OE_upper",
        ),
        "oph_inflation_cmb_v04_oph_parity_roe_pte": _nested(
            inflation_cmb_bridge,
            "cmb_success_ladder",
            "core_numbers",
            "v0_4_parity_PTE_R_OE_upper",
        ),
        "oph_inflation_cmb_v05_TT_lowell_delta_chi2": _nested(
            inflation_cmb_bridge,
            "cmb_success_ladder",
            "hard_gates_v0_5",
            "TT_lowell_delta_chi2",
        ),
        "oph_inflation_cmb_v05_TE_lowell_delta_chi2": _nested(
            inflation_cmb_bridge,
            "cmb_success_ladder",
            "hard_gates_v0_5",
            "TE_lowell_delta_chi2",
        ),
        "oph_inflation_cmb_v05_EE_lowell_delta_chi2": _nested(
            inflation_cmb_bridge,
            "cmb_success_ladder",
            "hard_gates_v0_5",
            "EE_lowell_delta_chi2",
        ),
        "oph_inflation_cmb_v05_TT_high_ell_delta_chi2_30_1200": _nested(
            inflation_cmb_bridge,
            "cmb_success_ladder",
            "hard_gates_v0_5",
            "TT_high_ell_delta_chi2_30_1200",
        ),
        "oph_inflation_cmb_v05_combined_lowell_delta_chi2": _nested(
            inflation_cmb_bridge,
            "cmb_success_ladder",
            "hard_gates_v0_5",
            "combined_TT_TE_EE_lowell_delta_chi2",
        ),
        "oph_inflation_cmb_v05_pressure_point_count": len(
            _nested(
                inflation_cmb_bridge,
                "cmb_success_ladder",
                "hard_gates_v0_5",
                "pressure_points",
            )
            or []
        )
        if inflation_cmb_bridge
        else None,
        "inflation_certificates_written": bool(inflation_certificates),
        "inflation_certificates_stack_ready": bool(
            inflation_certificates.get("inflation_certificate_stack_ready", False)
        ),
        "inflation_certificates_physical_cmb_prediction": bool(
            inflation_certificates.get("physical_cmb_prediction", False)
        ),
        "inflation_certificates_physical_matter_power_prediction": bool(
            inflation_certificates.get("physical_matter_power_prediction", False)
        ),
        "inflation_certificates_found_count": _nested(
            inflation_certificates,
            "certificate_summary",
            "found_count",
        ),
        "inflation_certificates_passed_count": _nested(
            inflation_certificates,
            "certificate_summary",
            "passed_count",
        ),
        "inflation_certificates_expected_count": _nested(
            inflation_certificates,
            "certificate_summary",
            "expected_count",
        ),
        "inflation_certificates_missing_count": len(
            _nested(inflation_certificates, "certificate_summary", "missing_types") or []
        )
        if inflation_certificates
        else None,
        "inflation_certificates_no_data_use": bool(
            _nested(inflation_certificates, "no_data_use_manifest", "no_data_use_receipt")
        ),
        "inflation_certificates_scalar_release_gate": bool(
            _nested(inflation_certificates, "readiness_gates", "scalar_release_certificate")
        ),
        "inflation_certificates_edge_center_gate": bool(
            _nested(inflation_certificates, "readiness_gates", "edge_center_certificate")
        ),
        "inflation_certificates_homogeneous_anomaly_gate": bool(
            _nested(inflation_certificates, "readiness_gates", "homogeneous_anomaly_certificate")
        ),
        "inflation_certificates_parent_collar_gate": bool(
            _nested(inflation_certificates, "readiness_gates", "parent_collar_kernel_certificate")
        ),
        "inflation_certificates_repair_matrix_gate": bool(
            _nested(inflation_certificates, "readiness_gates", "repair_matrix_certificate")
        ),
        "inflation_certificates_boltzmann_handoff_gate": bool(
            _nested(inflation_certificates, "readiness_gates", "boltzmann_handoff_certificate")
        ),
        "inflation_certificates_A_zeta": _nested(
            inflation_certificates,
            "derived_outputs",
            "scalar_release",
            "A_zeta",
        ),
        "inflation_certificates_n_s": _nested(
            inflation_certificates,
            "derived_outputs",
            "edge_center",
            "n_s",
        ),
        "inflation_certificates_Q_A": _nested(
            inflation_certificates,
            "derived_outputs",
            "homogeneous_anomaly",
            "Q_A_last",
        ),
        "inflation_certificates_mean_B_A": _nested(
            inflation_certificates,
            "derived_outputs",
            "parent_collar",
            "mean_B_A",
        ),
        "inflation_certificates_mean_Gamma_rec": _nested(
            inflation_certificates,
            "derived_outputs",
            "repair_matrix",
            "mean_Gamma_rec",
        ),
        "oph_unique_prediction_written": bool(unique_prediction),
        "oph_unique_prediction_measurement_comparable": bool(
            unique_prediction.get("measurement_comparable_now", False)
        ),
        "oph_unique_prediction_finite_lattice_derived": bool(
            unique_prediction.get("finite_lattice_derived", False)
        ),
        "oph_unique_prediction_physical_cmb_prediction": bool(
            unique_prediction.get("physical_cmb_prediction", False)
        ),
        "oph_unique_n_s": _nested(unique_prediction, "scalar_tilt", "n_s"),
        "oph_unique_eta_R": _nested(unique_prediction, "scalar_tilt", "eta_R"),
        "oph_unique_n_s_pull": _nested(unique_prediction, "scalar_tilt", "pull_vs_planck_sigma"),
        "oph_unique_q_IR": _nested(unique_prediction, "cmb_ir_kernel", "q_IR"),
        "oph_unique_ell_IR": _nested(unique_prediction, "cmb_ir_kernel", "ell_IR"),
        "oph_unique_N_frz_proxy": _nested(unique_prediction, "cmb_ir_kernel", "N_frz_proxy"),
        "oph_unique_parity_R_OE_TT_2_29": _nested(
            unique_prediction,
            "parity_envelope",
            "predicted_R_OE_TT_2_29",
        ),
        "oph_unique_sum_mnu_eV": _nested(unique_prediction, "neutrino_cosmology", "sum_mnu_eV"),
        "oph_unique_neutrino_f_nu": _nested(unique_prediction, "neutrino_cosmology", "f_nu"),
        "oph_unique_small_scale_neutrino_suppression": _nested(
            unique_prediction,
            "neutrino_cosmology",
            "small_scale_power_suppression_fraction",
        ),
        "oph_cnb_neutrino_written": bool(cnb_neutrino),
        "oph_cnb_neutrino_measurement_comparable": bool(cnb_neutrino.get("measurement_comparable_now", False)),
        "oph_cnb_neutrino_finite_lattice_derived": bool(cnb_neutrino.get("finite_lattice_derived", False)),
        "oph_cnb_neutrino_physical_cmb_prediction": bool(cnb_neutrino.get("physical_cmb_prediction", False)),
        "oph_cnb_neutrino_physical_matter_power_prediction": bool(
            cnb_neutrino.get("physical_matter_power_prediction", False)
        ),
        "oph_cnb_N_eff": _nested(cnb_neutrino, "relic_background", "N_eff"),
        "oph_cnb_sum_mnu_eV": _nested(cnb_neutrino, "oph_neutrino_branch", "sum_mnu_eV"),
        "oph_cnb_m_lightest_eV": _nested(cnb_neutrino, "oph_neutrino_branch", "m_lightest_eV"),
        "oph_cnb_Omega_nu_h2": _nested(cnb_neutrino, "relic_background", "Omega_nu_h2"),
        "oph_cnb_Omega_nu": _nested(cnb_neutrino, "relic_background", "Omega_nu"),
        "oph_cnb_f_nu": _nested(cnb_neutrino, "relic_background", "f_nu"),
        "oph_cnb_small_scale_suppression": _nested(
            cnb_neutrino,
            "relic_background",
            "small_scale_power_suppression_fraction",
        ),
        "oph_cnb_planck_neff_pull_sigma": _nested(
            cnb_neutrino,
            "measurement_comparisons",
            "Planck2018_N_eff",
            "pull_sigma",
        ),
        "oph_cnb_planck_bao_sum_mnu_pass": bool(
            _nested(cnb_neutrino, "measurement_comparisons", "Planck2018_BAO_sum_mnu_bound", "passes_bound")
        ),
        "oph_cnb_act_sum_mnu_pass": bool(
            _nested(cnb_neutrino, "measurement_comparisons", "ACT_DR6_extended_sum_mnu_bound", "passes_bound")
        ),
        "oph_cnb_desi_lcdm_sum_mnu_pass": bool(
            _nested(cnb_neutrino, "measurement_comparisons", "DESI_DR2_LCDM_sum_mnu_bound", "passes_bound")
        ),
        "oph_cnb_eta_A": _nested(cnb_neutrino, "late_repair_projection_target", "eta_A"),
        "oph_cnb_Pi_WL_compressed_required": _nested(
            cnb_neutrino,
            "late_repair_projection_target",
            "Pi_WL_compressed_required",
        ),
        "oph_cnb_five_of_seven_kernel_callable": bool(
            _nested(cnb_neutrino, "readiness_gates", "z6_poisson_five_of_seven_kernel_callable")
        ),
        "oph_cnb_five_of_seven_projection_gate": bool(
            _nested(cnb_neutrino, "readiness_gates", "z6_poisson_five_of_seven_compressed_projection")
        ),
        "oph_cnb_five_of_seven_pi_wl": _nested(
            cnb_neutrino,
            "late_repair_projection_target",
            "z6_poisson_five_of_seven",
            "pi_wl",
        ),
        "oph_cnb_five_of_seven_epsilon_A": _nested(
            cnb_neutrino,
            "late_repair_projection_target",
            "z6_poisson_five_of_seven",
            "epsilon_A_wl",
        ),
        "oph_cnb_five_of_seven_S8_projected": _nested(
            cnb_neutrino,
            "late_repair_projection_target",
            "z6_poisson_five_of_seven",
            "S8_projected_from_cdm_branch",
        ),
        "oph_cnb_five_of_seven_pull_sigma": _nested(
            cnb_neutrino,
            "late_repair_projection_target",
            "z6_poisson_five_of_seven",
            "pull_sigma_reference",
        ),
        "oph_cnb_background_gate": bool(
            _nested(cnb_neutrino, "readiness_gates", "measurement_comparable_relic_background")
        ),
        "oph_cnb_mass_derivation_gate": bool(
            _nested(cnb_neutrino, "readiness_gates", "finite_lattice_mass_derivation")
        ),
        "oph_cnb_B_A_kernel_gate": bool(
            _nested(cnb_neutrino, "readiness_gates", "B_A_k_a_from_finite_collar_parent")
        ),
        "oph_cnb_likelihood_gate": bool(
            _nested(cnb_neutrino, "readiness_gates", "full_boltzmann_likelihood_run")
        ),
        "finite_certificates_written": bool(finite_certificates or finite_certificate_manifest),
        "finite_certificates_stack_ready": bool(
            finite_certificates.get("finite_certificate_stack_ready", False)
            or finite_certificate_manifest.get("finite_certificate_stack_ready", False)
        ),
        "finite_certificates_compiler_ready": bool(
            finite_certificates.get("finite_certificate_compiler_ready", False)
            or finite_certificate_manifest.get("finite_certificate_compiler_ready", False)
            or finite_certificates.get("finite_certificate_stack_ready", False)
            or finite_certificate_manifest.get("finite_certificate_stack_ready", False)
        ),
        "finite_certificates_theorem_grade_inputs": bool(
            finite_certificates.get("theorem_grade_finite_inputs", False)
            or finite_certificate_manifest.get("theorem_grade_finite_inputs", False)
        ),
        "finite_certificates_proxy_certificate": bool(
            finite_certificates.get("proxy_certificate", False)
            or finite_certificate_manifest.get("proxy_certificate", False)
        ),
        "finite_certificates_no_data_use": bool(
            _nested(finite_certificates, "no_data_use_receipt", "no_data_use_receipt")
            or finite_certificate_manifest.get("no_data_use_receipt", False)
        ),
        "finite_certificates_release_code_gate": bool(
            _nested(finite_certificates, "readiness_gates", "release_code_certificate")
            or _nested(finite_certificate_manifest, "readiness_gates", "release_code_certificate")
        ),
        "finite_certificates_parent_collar_gate": bool(
            _nested(finite_certificates, "readiness_gates", "parent_collar_certificate")
            or _nested(finite_certificate_manifest, "readiness_gates", "parent_collar_certificate")
        ),
        "finite_certificates_repair_matrix_gate": bool(
            _nested(finite_certificates, "readiness_gates", "repair_matrix_certificate")
            or _nested(finite_certificate_manifest, "readiness_gates", "repair_matrix_certificate")
        ),
        "finite_certificates_boltzmann_export_gate": bool(
            _nested(finite_certificates, "readiness_gates", "boltzmann_export_certificate")
            or _nested(finite_certificate_manifest, "readiness_gates", "boltzmann_export_certificate")
        ),
        "finite_certificates_physical_cmb_prediction": bool(
            finite_certificates.get("physical_cmb_prediction", False)
        ),
        "finite_certificates_physical_matter_power_prediction": bool(
            finite_certificates.get("physical_matter_power_prediction", False)
        ),
        "finite_certificates_real_physics_certificate": bool(
            finite_certificates.get("real_physics_certificate", False)
        ),
        "finite_certificates_A_zeta": _nested(finite_certificates, "derived_outputs", "A_zeta"),
        "finite_certificates_n_s": _nested(finite_certificates, "derived_outputs", "n_s"),
        "finite_certificates_Q_A": _nested(finite_certificates, "derived_outputs", "Q_A"),
        "finite_certificates_Gamma_rec": _nested(finite_certificates, "derived_outputs", "Gamma_rec"),
        "finite_certificates_B_A": _first_numeric(_nested(finite_certificates, "derived_outputs", "B_A")),
        "b_a_parent_written": bool(b_a_parent),
        "b_a_parent_source_report_count": b_a_parent.get("source_report_count") if b_a_parent else None,
        "b_a_parent_observer_view_source_count": (
            b_a_parent.get("observer_view_source_count") if b_a_parent else None
        ),
        "b_a_parent_primary_parent_source": b_a_parent.get("primary_parent_source") if b_a_parent else None,
        "b_a_parent_row_count": len(b_a_parent.get("rows") or []) if b_a_parent else None,
        "b_a_parent_control_row_count": len(b_a_parent.get("control_rows") or []) if b_a_parent else None,
        "b_a_parent_observer_view_row_count": (
            len(b_a_parent.get("observer_view_rows") or []) if b_a_parent else None
        ),
        "b_a_parent_observer_view_control_row_count": (
            len(b_a_parent.get("observer_view_control_rows") or []) if b_a_parent else None
        ),
        "b_a_parent_receipt": bool(b_a_parent.get("B_A_PARENT_RECEIPT", False)),
        "b_a_parent_paired_diagnostic_receipt": bool(
            b_a_parent.get("B_A_PAIRED_DIAGNOSTIC_RECEIPT", False)
            or _nested(b_a_parent, "readiness", "B_A_PAIRED_DIAGNOSTIC_RECEIPT")
            or _nested(b_a_parent, "readiness", "checks", "paired_B_A_diagnostic_receipt")
        ),
        "b_a_parent_physical_prediction_ready": bool(b_a_parent.get("physical_prediction_ready", False)),
        "b_a_parent_physical_cmb_prediction": bool(b_a_parent.get("physical_cmb_prediction", False)),
        "b_a_parent_controls_fail": _nested(b_a_parent, "readiness", "checks", "controls_fail"),
        "b_a_parent_real_baryon_perturbation_runs_present": _nested(
            b_a_parent,
            "readiness",
            "checks",
            "real_baryon_perturbation_runs_present",
        ),
        "b_a_parent_finite_observer_view_parent_variation": _nested(
            b_a_parent,
            "readiness",
            "checks",
            "finite_observer_view_parent_variation",
        ),
        "b_a_parent_refinement_convergence_passed": _nested(
            b_a_parent,
            "readiness",
            "checks",
            "refinement_convergence_passed",
        ),
        "b_a_parent_control_failures": _nested(b_a_parent, "readiness", "control_failures"),
        "b_a_parent_missing_gates": _nested(b_a_parent, "readiness", "missing_gates") or [],
        "scalar_quotient_written": bool(scalar_quotient),
        "scalar_quotient_receipt": bool(scalar_quotient.get("SCALAR_QUOTIENT_RECEIPT", False)),
        "scalar_quotient_finite_ready": bool(
            scalar_quotient.get("finite_lattice_cmb_scalar_release_ready", False)
        ),
        "scalar_quotient_observer_count": scalar_quotient.get("observer_count"),
        "scalar_quotient_patch_count": scalar_quotient.get("patch_count"),
        "scalar_quotient_packet_alphabet_size": scalar_quotient.get("scalar_packet_alphabet_size"),
        "scalar_quotient_packet_entropy_bits": scalar_quotient.get("scalar_packet_entropy_bits"),
        "scalar_quotient_n_s": _nested(scalar_quotient, "edge_center_readout", "n_s_P_over_48"),
        "scalar_quotient_theta_OPH": _nested(scalar_quotient, "edge_center_readout", "theta_OPH_P_over_48"),
        "scalar_quotient_target_ell_IR": _nested(scalar_quotient, "active_angular_levels", "target_ell_IR"),
        "scalar_quotient_observer_level_proxy": _nested(
            scalar_quotient,
            "active_angular_levels",
            "observer_level_proxy_floor_sqrt_observers",
        ),
        "scalar_quotient_patch_capacity_level_proxy": _nested(
            scalar_quotient,
            "active_angular_levels",
            "patch_capacity_level_proxy_floor_sqrt_patches",
        ),
        "scalar_quotient_active_33_level_clause": bool(
            _nested(scalar_quotient, "readiness_gates", "active_33_level_freezeout_clause")
        ),
        "scalar_quotient_theorem_grade_release": bool(
            _nested(scalar_quotient, "readiness_gates", "theorem_grade_scalar_release_code")
        ),
        "scalar_quotient_physical_cmb_prediction": bool(scalar_quotient.get("physical_cmb_prediction", False)),
        "neutral_profile_audit_written": bool(neutral_profile),
        "neutral_profile_count": len(neutral_profile_rows),
        "neutral_profile_strict_3d_ready_count": sum(
            1 for item in neutral_profile_rows if bool(item.get("strict_3d_ready", False))
        ),
        "neutral_profile_sampled_observer_count": neutral_profile.get("sampled_observer_count"),
        "neutral_profile_all_corr_dim": _nested(
            neutral_profile_all,
            "dimension",
            "correlation_dimension",
            "estimate",
        ),
        "neutral_profile_all_mle_dim": _nested(
            neutral_profile_all,
            "dimension",
            "local_mle_dimension",
            "median_estimate",
        ),
        "neutral_profile_all_best_model": _nested(neutral_profile_all, "model_selection", "best_model"),
        "neutral_profile_all_s2_leakage_corr": _nested(neutral_profile_all, "leakage", "s2_distance_correlation"),
        "neutral_profile_all_s2_leakage_pass": _nested(neutral_profile_all, "leakage", "s2_leakage_pass"),
        "neutral_profile_scalar_response_corr_dim": _nested(
            neutral_profile_scalar_response,
            "dimension",
            "correlation_dimension",
            "estimate",
        ),
        "neutral_profile_scalar_response_mle_dim": _nested(
            neutral_profile_scalar_response,
            "dimension",
            "local_mle_dimension",
            "median_estimate",
        ),
        "neutral_profile_scalar_response_best_model": _nested(
            neutral_profile_scalar_response,
            "model_selection",
            "best_model",
        ),
        "neutral_profile_scalar_response_s2_leakage_corr": _nested(
            neutral_profile_scalar_response,
            "leakage",
            "s2_distance_correlation",
        ),
        "neutral_profile_scalar_response_s2_leakage_pass": _nested(
            neutral_profile_scalar_response,
            "leakage",
            "s2_leakage_pass",
        ),
        "neutral_profile_prime_geometric_corr_dim": _nested(
            neutral_profile_prime_geometric,
            "dimension",
            "correlation_dimension",
            "estimate",
        ),
        "neutral_profile_prime_geometric_mle_dim": _nested(
            neutral_profile_prime_geometric,
            "dimension",
            "local_mle_dimension",
            "median_estimate",
        ),
        "neutral_profile_prime_geometric_best_model": _nested(
            neutral_profile_prime_geometric,
            "model_selection",
            "best_model",
        ),
        "neutral_profile_prime_geometric_s2_leakage_corr": _nested(
            neutral_profile_prime_geometric,
            "leakage",
            "s2_distance_correlation",
        ),
        "neutral_profile_prime_geometric_s2_leakage_pass": _nested(
            neutral_profile_prime_geometric,
            "leakage",
            "s2_leakage_pass",
        ),
        "neutral_profile_prime_control_quotient_corr_dim": _nested(
            neutral_profile_prime_control_quotient,
            "dimension",
            "correlation_dimension",
            "estimate",
        ),
        "neutral_profile_prime_control_quotient_mle_dim": _nested(
            neutral_profile_prime_control_quotient,
            "dimension",
            "local_mle_dimension",
            "median_estimate",
        ),
        "neutral_profile_prime_control_quotient_best_model": _nested(
            neutral_profile_prime_control_quotient,
            "model_selection",
            "best_model",
        ),
        "neutral_profile_prime_control_quotient_s2_leakage_corr": _nested(
            neutral_profile_prime_control_quotient,
            "leakage",
            "s2_distance_correlation",
        ),
        "neutral_profile_prime_control_quotient_s2_leakage_pass": _nested(
            neutral_profile_prime_control_quotient,
            "leakage",
            "s2_leakage_pass",
        ),
        "neutral_profile_prime_rank3_corr_dim": _nested(
            neutral_profile_prime_rank3,
            "dimension",
            "correlation_dimension",
            "estimate",
        ),
        "neutral_profile_prime_rank3_mle_dim": _nested(
            neutral_profile_prime_rank3,
            "dimension",
            "local_mle_dimension",
            "median_estimate",
        ),
        "neutral_profile_prime_rank3_best_model": _nested(
            neutral_profile_prime_rank3,
            "model_selection",
            "best_model",
        ),
        "neutral_profile_prime_rank3_s2_leakage_corr": _nested(
            neutral_profile_prime_rank3,
            "leakage",
            "s2_distance_correlation",
        ),
        "neutral_profile_prime_control_quotient_rank3_corr_dim": _nested(
            neutral_profile_prime_control_quotient_rank3,
            "dimension",
            "correlation_dimension",
            "estimate",
        ),
        "neutral_profile_prime_control_quotient_rank3_mle_dim": _nested(
            neutral_profile_prime_control_quotient_rank3,
            "dimension",
            "local_mle_dimension",
            "median_estimate",
        ),
        "neutral_profile_prime_control_quotient_rank3_best_model": _nested(
            neutral_profile_prime_control_quotient_rank3,
            "model_selection",
            "best_model",
        ),
        "neutral_profile_prime_control_quotient_rank3_s2_leakage_corr": _nested(
            neutral_profile_prime_control_quotient_rank3,
            "leakage",
            "s2_distance_correlation",
        ),
        "neutral_profile_prime_rank8_corr_dim": _nested(
            neutral_profile_prime_rank8,
            "dimension",
            "correlation_dimension",
            "estimate",
        ),
        "neutral_profile_prime_rank8_mle_dim": _nested(
            neutral_profile_prime_rank8,
            "dimension",
            "local_mle_dimension",
            "median_estimate",
        ),
        "neutral_profile_prime_rank8_best_model": _nested(
            neutral_profile_prime_rank8,
            "model_selection",
            "best_model",
        ),
        "neutral_profile_prime_rank8_s2_leakage_corr": _nested(
            neutral_profile_prime_rank8,
            "leakage",
            "s2_distance_correlation",
        ),
        "neutral_profile_prime_control_quotient_rank8_corr_dim": _nested(
            neutral_profile_prime_control_quotient_rank8,
            "dimension",
            "correlation_dimension",
            "estimate",
        ),
        "neutral_profile_prime_control_quotient_rank8_mle_dim": _nested(
            neutral_profile_prime_control_quotient_rank8,
            "dimension",
            "local_mle_dimension",
            "median_estimate",
        ),
        "neutral_profile_prime_control_quotient_rank8_best_model": _nested(
            neutral_profile_prime_control_quotient_rank8,
            "model_selection",
            "best_model",
        ),
        "neutral_profile_prime_control_quotient_rank8_s2_leakage_corr": _nested(
            neutral_profile_prime_control_quotient_rank8,
            "leakage",
            "s2_distance_correlation",
        ),
        "neutral_profile_support_visible_corr_dim": _nested(
            neutral_profile_support_visible,
            "dimension",
            "correlation_dimension",
            "estimate",
        ),
        "neutral_profile_support_visible_mle_dim": _nested(
            neutral_profile_support_visible,
            "dimension",
            "local_mle_dimension",
            "median_estimate",
        ),
        "neutral_profile_support_visible_best_model": _nested(
            neutral_profile_support_visible,
            "model_selection",
            "best_model",
        ),
        "neutral_profile_support_visible_s2_leakage_corr": _nested(
            neutral_profile_support_visible,
            "leakage",
            "s2_distance_correlation",
        ),
        "neutral_profile_support_visible_s2_leakage_pass": _nested(
            neutral_profile_support_visible,
            "leakage",
            "s2_leakage_pass",
        ),
        "prime_rank_sweep_written": bool(prime_rank_sweep),
        "prime_rank_sweep_quotient_3d_diagnostic_receipt": bool(
            prime_rank_sweep.get("PRIME_GEOMETRIC_QUOTIENT_3D_DIAGNOSTIC_RECEIPT", False)
            or prime_rank_sweep.get("prime_geometric_quotient_3d_diagnostic_receipt", False)
        ),
        "prime_rank_sweep_spatial_3d_candidate_receipt": bool(
            prime_rank_sweep.get("prime_geometric_spatial_3d_candidate_receipt", False)
        ),
        "prime_rank_sweep_control_quotient_spatial_3d_candidate_receipt": bool(
            prime_rank_sweep.get("prime_geometric_control_quotient_spatial_3d_candidate_receipt", False)
        ),
        "prime_rank_sweep_strict_neutral_candidate_receipt": bool(
            prime_rank_sweep.get("prime_geometric_strict_neutral_candidate_receipt", False)
        ),
        "prime_rank_sweep_proof_blockers": prime_rank_sweep.get("proof_blockers", []),
        "prime_rank_sweep_control_quotient_is_negative_control": _nested(
            prime_rank_sweep,
            "regulator_control_quotient_lane",
            "is_negative_control",
        ),
        "prime_rank_sweep_selected_controls_written": bool(
            _nested(prime_rank_sweep, "selected_rank_controls", "control_rows")
        ),
        "prime_rank_sweep_selected_controls_all_failed": bool(
            _nested(prime_rank_sweep, "selected_rank_controls", "all_expected_failures_observed")
        ),
        "prime_rank_sweep_coordinate_rank3_tautology_warning": bool(
            _nested(prime_rank_sweep, "selected_rank_controls", "coordinate_rank3_tautology_warning")
        ),
        "prime_rank_sweep_selected_directional_control_survive_count": _selected_rank_control_count(
            prime_rank_sweep,
            metric="directional_cosine",
            survives=True,
        ),
        "prime_rank_sweep_selected_directional_control_fail_count": _selected_rank_control_count(
            prime_rank_sweep,
            metric="directional_cosine",
            survives=False,
        ),
        "prime_rank_sweep_selected_coordinate_control_survive_count": _selected_rank_control_count(
            prime_rank_sweep,
            metric="coordinate_euclidean",
            survives=True,
        ),
        "prime_rank_sweep_selected_coordinate_control_fail_count": _selected_rank_control_count(
            prime_rank_sweep,
            metric="coordinate_euclidean",
            survives=False,
        ),
        "prime_rank_sweep_rank_count": len(prime_rank_sweep.get("rows", []))
        if isinstance(prime_rank_sweep.get("rows", []), list)
        else 0,
        "prime_rank_sweep_strict_3d_ready_count": int(prime_rank_sweep.get("strict_3d_ready_count", 0) or 0),
        "prime_rank_sweep_dimension_3d_window_count": int(
            prime_rank_sweep.get("dimension_3d_window_count", 0) or 0
        ),
        "prime_rank_sweep_best_rank": _nested(prime_rank_sweep, "best_dimension_row", "rank"),
        "prime_rank_sweep_best_corr_dim": _nested(
            prime_rank_sweep,
            "best_dimension_row",
            "dimension",
            "correlation_dimension",
            "estimate",
        ),
        "prime_rank_sweep_best_mle_dim": _nested(
            prime_rank_sweep,
            "best_dimension_row",
            "dimension",
            "local_mle_dimension",
            "median_estimate",
        ),
        "prime_rank_sweep_best_model": _nested(
            prime_rank_sweep,
            "best_dimension_row",
            "model_selection",
            "best_model",
        ),
        "prime_rank_sweep_best_s2_leakage_corr": _nested(
            prime_rank_sweep,
            "best_dimension_row",
            "leakage",
            "s2_distance_correlation",
        ),
        "prime_rank_sweep_best_3d_rank": _nested(prime_rank_sweep, "best_3d_dimension_row", "rank"),
        "prime_rank_sweep_best_3d_corr_dim": _nested(
            prime_rank_sweep,
            "best_3d_dimension_row",
            "dimension",
            "correlation_dimension",
            "estimate",
        ),
        "prime_rank_sweep_best_3d_mle_dim": _nested(
            prime_rank_sweep,
            "best_3d_dimension_row",
            "dimension",
            "local_mle_dimension",
            "median_estimate",
        ),
        "prime_rank_sweep_best_3d_model": _nested(
            prime_rank_sweep,
            "best_3d_dimension_row",
            "model_selection",
            "best_model",
        ),
        "prime_rank_sweep_coordinate_rank_count": len(prime_rank_sweep.get("coordinate_rows", []))
        if isinstance(prime_rank_sweep.get("coordinate_rows", []), list)
        else 0,
        "prime_rank_sweep_coordinate_spatial_3d_ready_count": int(
            prime_rank_sweep.get("coordinate_spatial_3d_ready_count", 0) or 0
        ),
        "prime_rank_sweep_coordinate_dimension_3d_window_count": int(
            prime_rank_sweep.get("coordinate_dimension_3d_window_count", 0) or 0
        ),
        "prime_rank_sweep_coordinate_best_rank": _nested(
            prime_rank_sweep,
            "coordinate_best_dimension_row",
            "rank",
        ),
        "prime_rank_sweep_coordinate_best_corr_dim": _nested(
            prime_rank_sweep,
            "coordinate_best_dimension_row",
            "dimension",
            "correlation_dimension",
            "estimate",
        ),
        "prime_rank_sweep_coordinate_best_mle_dim": _nested(
            prime_rank_sweep,
            "coordinate_best_dimension_row",
            "dimension",
            "local_mle_dimension",
            "median_estimate",
        ),
        "prime_rank_sweep_coordinate_best_model": _nested(
            prime_rank_sweep,
            "coordinate_best_dimension_row",
            "model_selection",
            "best_model",
        ),
        "prime_rank_sweep_coordinate_best_s2_leakage_corr": _nested(
            prime_rank_sweep,
            "coordinate_best_dimension_row",
            "leakage",
            "s2_distance_correlation",
        ),
        "prime_rank_sweep_coordinate_best_3d_rank": _nested(
            prime_rank_sweep,
            "coordinate_best_3d_dimension_row",
            "rank",
        ),
        "prime_rank_sweep_coordinate_best_3d_corr_dim": _nested(
            prime_rank_sweep,
            "coordinate_best_3d_dimension_row",
            "dimension",
            "correlation_dimension",
            "estimate",
        ),
        "prime_rank_sweep_coordinate_best_3d_mle_dim": _nested(
            prime_rank_sweep,
            "coordinate_best_3d_dimension_row",
            "dimension",
            "local_mle_dimension",
            "median_estimate",
        ),
        "prime_rank_sweep_coordinate_best_3d_model": _nested(
            prime_rank_sweep,
            "coordinate_best_3d_dimension_row",
            "model_selection",
            "best_model",
        ),
        "prime_rank_sweep_control_quotient_rank_count": len(prime_rank_sweep.get("control_quotient_rows", []))
        if isinstance(prime_rank_sweep.get("control_quotient_rows", []), list)
        else 0,
        "prime_rank_sweep_control_quotient_strict_3d_ready_count": int(
            prime_rank_sweep.get("control_quotient_strict_3d_ready_count", 0) or 0
        ),
        "prime_rank_sweep_control_quotient_dimension_3d_window_count": int(
            prime_rank_sweep.get("control_quotient_dimension_3d_window_count", 0) or 0
        ),
        "prime_rank_sweep_control_quotient_best_rank": _nested(
            prime_rank_sweep,
            "control_quotient_best_dimension_row",
            "rank",
        ),
        "prime_rank_sweep_control_quotient_best_corr_dim": _nested(
            prime_rank_sweep,
            "control_quotient_best_dimension_row",
            "dimension",
            "correlation_dimension",
            "estimate",
        ),
        "prime_rank_sweep_control_quotient_best_mle_dim": _nested(
            prime_rank_sweep,
            "control_quotient_best_dimension_row",
            "dimension",
            "local_mle_dimension",
            "median_estimate",
        ),
        "prime_rank_sweep_control_quotient_best_model": _nested(
            prime_rank_sweep,
            "control_quotient_best_dimension_row",
            "model_selection",
            "best_model",
        ),
        "prime_rank_sweep_control_quotient_best_s2_leakage_corr": _nested(
            prime_rank_sweep,
            "control_quotient_best_dimension_row",
            "leakage",
            "s2_distance_correlation",
        ),
        "prime_rank_sweep_control_quotient_coordinate_rank_count": len(
            prime_rank_sweep.get("control_quotient_coordinate_rows", [])
        )
        if isinstance(prime_rank_sweep.get("control_quotient_coordinate_rows", []), list)
        else 0,
        "prime_rank_sweep_control_quotient_coordinate_spatial_3d_ready_count": int(
            prime_rank_sweep.get("control_quotient_coordinate_spatial_3d_ready_count", 0) or 0
        ),
        "prime_rank_sweep_control_quotient_coordinate_dimension_3d_window_count": int(
            prime_rank_sweep.get("control_quotient_coordinate_dimension_3d_window_count", 0) or 0
        ),
        "prime_rank_sweep_control_quotient_coordinate_best_rank": _nested(
            prime_rank_sweep,
            "control_quotient_coordinate_best_dimension_row",
            "rank",
        ),
        "prime_rank_sweep_control_quotient_coordinate_best_corr_dim": _nested(
            prime_rank_sweep,
            "control_quotient_coordinate_best_dimension_row",
            "dimension",
            "correlation_dimension",
            "estimate",
        ),
        "prime_rank_sweep_control_quotient_coordinate_best_mle_dim": _nested(
            prime_rank_sweep,
            "control_quotient_coordinate_best_dimension_row",
            "dimension",
            "local_mle_dimension",
            "median_estimate",
        ),
        "prime_rank_sweep_control_quotient_coordinate_best_model": _nested(
            prime_rank_sweep,
            "control_quotient_coordinate_best_dimension_row",
            "model_selection",
            "best_model",
        ),
        "prime_rank_sweep_control_quotient_coordinate_best_s2_leakage_corr": _nested(
            prime_rank_sweep,
            "control_quotient_coordinate_best_dimension_row",
            "leakage",
            "s2_distance_correlation",
        ),
        "prime_rank_sweep_control_quotient_coordinate_best_3d_rank": _nested(
            prime_rank_sweep,
            "control_quotient_coordinate_best_3d_dimension_row",
            "rank",
        ),
        "prime_rank_sweep_control_quotient_coordinate_best_3d_corr_dim": _nested(
            prime_rank_sweep,
            "control_quotient_coordinate_best_3d_dimension_row",
            "dimension",
            "correlation_dimension",
            "estimate",
        ),
        "prime_rank_sweep_control_quotient_coordinate_best_3d_mle_dim": _nested(
            prime_rank_sweep,
            "control_quotient_coordinate_best_3d_dimension_row",
            "dimension",
            "local_mle_dimension",
            "median_estimate",
        ),
        "prime_rank_sweep_control_quotient_coordinate_best_3d_model": _nested(
            prime_rank_sweep,
            "control_quotient_coordinate_best_3d_dimension_row",
            "model_selection",
            "best_model",
        ),
        "prime_rank_sweep_control_quotient_coordinate_best_3d_s2_leakage_corr": _nested(
            prime_rank_sweep,
            "control_quotient_coordinate_best_3d_dimension_row",
            "leakage",
            "s2_distance_correlation",
        ),
        "prime_rank_sweep_control_quotient_coordinate_best_3d_s2_leakage_pass": _nested(
            prime_rank_sweep,
            "control_quotient_coordinate_best_3d_dimension_row",
            "leakage",
            "s2_leakage_pass",
        ),
        "prime_rank_refinement_written": bool(prime_rank_refinement),
        "prime_rank_refinement_candidate_receipt": bool(
            prime_rank_refinement.get("control_quotient_rank3_refinement_candidate_receipt", False)
        ),
        "prime_rank_refinement_strict_neutral_receipt": bool(
            prime_rank_refinement.get("strict_neutral_bulk_refinement_receipt", False)
        ),
        "prime_rank_refinement_run_count": int(prime_rank_refinement.get("run_count", 0) or 0),
        "prime_rank_refinement_multi_scale": bool(prime_rank_refinement.get("multi_scale", False)),
        "prime_rank_refinement_all_candidates": bool(
            prime_rank_refinement.get("all_control_quotient_spatial_3d_candidates", False)
        ),
        "prime_rank_refinement_all_candidate_s2_leakage_pass": bool(
            prime_rank_refinement.get("all_candidate_s2_leakage_pass", False)
        ),
        "prime_rank_refinement_all_candidate_rank3_e3": bool(
            prime_rank_refinement.get("all_candidate_rank3_e3", False)
        ),
        "prime_rank_refinement_dimension_drift": prime_rank_refinement.get("candidate_dimension_drift"),
        "prime_rank_refinement_dimension_stable": bool(
            prime_rank_refinement.get("candidate_dimension_stable", False)
        ),
        "prime_rank_refinement_independent_rank3_all": bool(
            prime_rank_refinement.get("independent_rank3_selector_all", False)
        ),
        "prime_rank_refinement_patch_counts": _rank_refinement_patch_counts(prime_rank_refinement),
        "prime_rank_refinement_size_median_dimensions": _rank_refinement_size_medians(prime_rank_refinement),
        "prime_rank_refinement_proof_blockers": prime_rank_refinement.get("proof_blockers", []),
        "parent_collar_ladder_written": bool(parent_collar_ladder),
        "parent_collar_ladder_compiler_ready": bool(parent_collar_ladder.get("compiler_ready", False)),
        "parent_collar_ladder_regulator_ready": bool(
            parent_collar_ladder.get("regulator_ladder_ready", False)
        ),
        "parent_collar_ladder_scaling_pass": bool(
            parent_collar_ladder.get("parent_collar_recovery_ladder_receipt", False)
        ),
        "parent_collar_ladder_local_density_receipt": bool(
            parent_collar_ladder.get("local_recovery_density_receipt", False)
        ),
        "parent_collar_ladder_strict_local_density_receipt": bool(
            parent_collar_ladder.get("strict_local_recovery_density_receipt", False)
        ),
        "parent_collar_ladder_theorem_grade": bool(
            parent_collar_ladder.get("theorem_grade_parent_collar_ladder", False)
        ),
        "parent_collar_ladder_strict_cap_family_matched": bool(
            _nested(parent_collar_ladder, "cap_family", "strict_cap_family_matched")
        ),
        "parent_collar_ladder_unique_theta_family_matched": bool(
            _nested(parent_collar_ladder, "cap_family", "unique_theta_family_matched")
        ),
        "parent_collar_ladder_physical_cmb_prediction": bool(
            parent_collar_ladder.get("physical_cmb_prediction", False)
        ),
        "parent_collar_ladder_report_count": parent_collar_ladder.get("report_count"),
        "parent_collar_ladder_row_count": parent_collar_ladder.get("row_count"),
        "parent_collar_ladder_patch_count_count": len(parent_collar_ladder.get("patch_counts", []) or [])
        if parent_collar_ladder
        else None,
        "parent_collar_ladder_slope": _nested(parent_collar_ladder, "scaling", "slope"),
        "parent_collar_ladder_density_slope": _nested(parent_collar_ladder, "local_density_scaling", "slope"),
        "parent_collar_ladder_final_p90_epsilon_cmi": _final_parent_collar_value(
            parent_collar_ladder,
            "p90_epsilon_cmi",
        ),
        "parent_collar_ladder_final_p90_epsilon_cmi_per_collar_patch": _final_parent_collar_value(
            parent_collar_ladder,
            "p90_epsilon_cmi_per_collar_patch",
        ),
        "parent_collar_ladder_final_p90_r_fr": _final_parent_collar_value(
            parent_collar_ladder,
            "p90_r_fr_bound",
        ),
        "screen_capacity_written": bool(screen_capacity),
        "screen_capacity_input_mode": _nested(
            screen_capacity,
            "observed_branch_normalization",
            "input_mode",
        ),
        "screen_capacity_N_patch_bare_ratio": _nested(
            screen_capacity,
            "observed_branch_normalization",
            "N_patch_bare_radius_squared_ratio",
        ),
        "screen_capacity_N_scr": _nested(screen_capacity, "observed_branch_normalization", "N_scr_entropy_capacity"),
        "screen_capacity_Lambda_lP2": _nested(screen_capacity, "observed_branch_normalization", "Lambda_lP2"),
        "screen_capacity_P_cell_count": _nested(
            screen_capacity,
            "observed_branch_normalization",
            "N_cells_if_tiled_by_local_P_cells",
        ),
        "screen_capacity_observed_readout_gate": bool(
            _nested(screen_capacity, "readiness_gates", "observed_branch_N_scr_readout_available")
        ),
        "screen_capacity_F_N_implemented": bool(
            _nested(screen_capacity, "readiness_gates", "F_N_readback_map_implemented")
        ),
        "screen_capacity_fixed_point_solved": bool(
            _nested(screen_capacity, "readiness_gates", "N_CRC_fixed_point_solved_from_finite_simulator")
        ),
        "screen_capacity_physical_cmb_prediction": bool(screen_capacity.get("physical_cmb_prediction", False)),
        "repair_scale_written": bool(repair_scale),
        "repair_scale_local_P_available": bool(
            _nested(repair_scale, "readiness_gates", "local_P_closure_available")
        ),
        "repair_scale_N_CRC_declared": bool(_nested(repair_scale, "readiness_gates", "global_N_CRC_declared")),
        "repair_scale_numeric_match_1pct": bool(
            _nested(repair_scale, "readiness_gates", "scale_closure_numeric_match_within_1_percent")
        ),
        "repair_scale_24_rounds_declared": bool(
            _nested(repair_scale, "readiness_gates", "twenty_four_round_hypothesis_declared")
        ),
        "repair_scale_24_rounds_derived": bool(
            _nested(repair_scale, "readiness_gates", "twenty_four_round_hypothesis_derived_from_finite_selector")
        ),
        "repair_scale_finite_lattice_derived_eta_R": bool(
            _nested(repair_scale, "readiness_gates", "finite_lattice_derived_eta_R")
        ),
        "repair_scale_physical_cmb_prediction": bool(repair_scale.get("physical_cmb_prediction", False)),
        "repair_scale_bulk_3d_established": bool(repair_scale.get("bulk_3d_established", False)),
        "repair_scale_rounds": _nested(repair_scale, "hypothesis", "declared_repair_rounds_m"),
        "repair_scale_capacity_exponent": _nested(repair_scale, "hypothesis", "capacity_exponent_2m"),
        "repair_scale_gprime": _nested(repair_scale, "closure_outputs", "local_repair_contraction_abs_gprime"),
        "repair_scale_q_round": _nested(repair_scale, "closure_outputs", "scale_factor_per_round_q"),
        "repair_scale_length_ratio": _nested(repair_scale, "closure_outputs", "length_ratio_after_m_rounds"),
        "repair_scale_N_implied_by_ansatz": (
            _nested(repair_scale, "closure_outputs", "capacity_implied_by_declared_repair_depth_ansatz")
            or _nested(repair_scale, "closure_outputs", "capacity_predicted_from_local_P")
        ),
        "repair_scale_N_pred": (
            _nested(repair_scale, "closure_outputs", "capacity_implied_by_declared_repair_depth_ansatz")
            or _nested(repair_scale, "closure_outputs", "capacity_predicted_from_local_P")
        ),
        "repair_scale_N_CRC": _nested(repair_scale, "global_capacity_inputs", "N_CRC"),
        "repair_scale_rel_error": _nested(
            repair_scale,
            "closure_outputs",
            "relative_error_ansatz_capacity_vs_declared_N_CRC",
        )
        or _nested(
            repair_scale,
            "closure_outputs",
            "relative_error_gprime_vs_N_CRC_closure",
        ),
        "repair_scale_eta_R": _nested(repair_scale, "closure_outputs", "eta_R_p_over_2m"),
        "repair_scale_n_s": _nested(repair_scale, "closure_outputs", "n_s_p_over_2m"),
        "repair_scale_1m_depth_rounds": _round_depth_for_patch_count(repair_scale, 1_048_576),
        "scale_compressed_written": bool(scale_compressed),
        "scale_compressed_operator_receipt": bool(
            scale_compressed.get("scale_compressed_operator_receipt", False)
        ),
        "scale_compressed_round_trace_receipt": bool(
            scale_compressed.get("repair_round_trace_receipt", False)
        ),
        "scale_compressed_populated_h3_preview": bool(
            _nested(scale_compressed, "h3_preview", "populated_h3_preview_receipt")
        ),
        "scale_compressed_cap_profile_receipt": bool(
            _nested(scale_compressed, "h3_preview", "cap_profile_receipt")
        ),
        "scale_compressed_strict_neutral_bulk": bool(
            _nested(scale_compressed, "h3_preview", "strict_neutral_third_person_bulk_established")
        ),
        "scale_compressed_particle_preview": bool(
            _nested(scale_compressed, "particle_preview", "particle_preview_receipt")
        ),
        "scale_compressed_production_particle_matter": bool(
            _nested(scale_compressed, "particle_preview", "production_particle_matter_receipt")
        ),
        "scale_compressed_physical_cmb_prediction": bool(scale_compressed.get("physical_cmb_prediction", False)),
        "scale_compressed_rounds": scale_compressed.get("logical_repair_rounds") if scale_compressed else None,
        "scale_compressed_eta_R": _nested(scale_compressed, "cmb_parameter_readouts", "eta_R"),
        "scale_compressed_n_s": _nested(scale_compressed, "cmb_parameter_readouts", "n_s"),
        "scale_compressed_q_IR": _nested(scale_compressed, "cmb_parameter_readouts", "q_IR"),
        "scale_compressed_ell_IR": _nested(scale_compressed, "cmb_parameter_readouts", "ell_IR"),
        "scale_compressed_N_CRC_implied_by_ansatz": _nested(
            scale_compressed,
            "cmb_parameter_readouts",
            "N_CRC_implied_by_declared_repair_depth_ansatz",
        )
        or _nested(
            scale_compressed,
            "cmb_parameter_readouts",
            "N_CRC_predicted_from_P",
        ),
        "scale_compressed_N_CRC_pred": _nested(
            scale_compressed,
            "cmb_parameter_readouts",
            "N_CRC_implied_by_declared_repair_depth_ansatz",
        )
        or _nested(
            scale_compressed,
            "cmb_parameter_readouts",
            "N_CRC_predicted_from_P",
        ),
        "scale_compressed_N_CRC_declared": _nested(
            scale_compressed,
            "cmb_parameter_readouts",
            "N_CRC_declared",
        ),
        "scale_compressed_rel_error": _nested(
            scale_compressed,
            "cmb_parameter_readouts",
            "relative_error_gprime_vs_N_CRC",
        ),
        "scale_compressed_object_count": _nested(scale_compressed, "h3_preview", "object_count"),
        "scale_compressed_particle_count": _nested(scale_compressed, "particle_preview", "particle_worldline_count"),
        "scale_compressed_cl_ell_max": _nested(scale_compressed, "cmb_screen_scaffold", "ell_max"),
        "scale_compressed_cmb_camb_written": bool(scale_compressed_cmb_camb),
        "scale_compressed_cmb_camb_curve_comparable": bool(
            scale_compressed_cmb_camb.get("measurement_comparable_cmb_curve", False)
        ),
        "scale_compressed_cmb_camb_transfer_receipt": bool(
            scale_compressed_cmb_camb.get("screen_camb_transfer_receipt", False)
        ),
        "scale_compressed_cmb_camb_physical_cmb_prediction": bool(
            scale_compressed_cmb_camb.get("physical_cmb_prediction", False)
        ),
        "scale_compressed_cmb_camb_rounds": _nested(
            scale_compressed_cmb_camb,
            "scale_compressed_input",
            "logical_repair_rounds",
        ),
        "scale_compressed_cmb_camb_operator_receipt": _nested(
            scale_compressed_cmb_camb,
            "scale_compressed_input",
            "scale_compressed_operator_receipt",
        ),
        "scale_compressed_cmb_camb_h3_preview_receipt": _nested(
            scale_compressed_cmb_camb,
            "scale_compressed_input",
            "populated_h3_preview_receipt",
        ),
        "scale_compressed_cmb_camb_eta_R": _nested(scale_compressed_cmb_camb, "scale_compressed_input", "eta_R"),
        "scale_compressed_cmb_camb_n_s": _nested(scale_compressed_cmb_camb, "scale_compressed_input", "n_s"),
        "scale_compressed_cmb_camb_q_IR": _nested(scale_compressed_cmb_camb, "scale_compressed_input", "q_IR"),
        "scale_compressed_cmb_camb_ell_IR": _nested(
            scale_compressed_cmb_camb,
            "scale_compressed_input",
            "ell_IR",
        ),
        "scale_compressed_cmb_camb_lcdm_shape_correlation": _nested(
            scale_compressed_cmb_camb,
            "comparison",
            "camb_lcdm_powerlaw",
            "shape_correlation",
        ),
        "scale_compressed_cmb_camb_scalar_shape_correlation": _nested(
            scale_compressed_cmb_camb,
            "comparison",
            "scale_compressed_scalar_tilt",
            "shape_correlation",
        ),
        "scale_compressed_cmb_camb_ir_shape_correlation": _nested(
            scale_compressed_cmb_camb,
            "comparison",
            "scale_compressed_ir_kernel",
            "shape_correlation",
        ),
        "scale_compressed_cmb_camb_lcdm_chi2_per_bin": _nested(
            scale_compressed_cmb_camb,
            "comparison",
            "camb_lcdm_powerlaw",
            "amplitude_fit_chi2_per_bin",
        ),
        "scale_compressed_cmb_camb_scalar_chi2_per_bin": _nested(
            scale_compressed_cmb_camb,
            "comparison",
            "scale_compressed_scalar_tilt",
            "amplitude_fit_chi2_per_bin",
        ),
        "scale_compressed_cmb_camb_ir_chi2_per_bin": _nested(
            scale_compressed_cmb_camb,
            "comparison",
            "scale_compressed_ir_kernel",
            "amplitude_fit_chi2_per_bin",
        ),
        "scale_compressed_cmb_camb_acoustic_mean_abs_delta": _nested(
            scale_compressed_cmb_camb,
            "acoustic_preservation",
            "mean_abs_fractional_delta_ell_ge_50",
        ),
        "cmb_derivation_written": bool(cmb_derivation),
        "cmb_derivation_ready": bool(cmb_derivation.get("finite_lattice_cmb_parameters_ready", False)),
        "cmb_derivation_run_count": cmb_derivation.get("run_count") if cmb_derivation else None,
        "cmb_derivation_mean_eta_R": _nested(cmb_derivation, "aggregate", "mean_best_eta_R_estimate"),
        "cmb_derivation_mean_eta_R_abs_error": _nested(cmb_derivation, "aggregate", "mean_eta_R_abs_error"),
        "cmb_derivation_mean_q_IR": _nested(cmb_derivation, "aggregate", "mean_best_q_IR_proxy"),
        "cmb_derivation_mean_ell_IR": _nested(cmb_derivation, "aggregate", "mean_best_ell_IR_proxy"),
        "cmb_derivation_mean_scalar_quotient_theta_OPH": _nested(
            cmb_derivation,
            "aggregate",
            "mean_scalar_quotient_theta_OPH",
        ),
        "cmb_derivation_mean_scalar_quotient_n_s": _nested(
            cmb_derivation,
            "aggregate",
            "mean_scalar_quotient_n_s",
        ),
        "cmb_derivation_mean_scalar_quotient_n_s_abs_error": _nested(
            cmb_derivation,
            "aggregate",
            "mean_scalar_quotient_n_s_abs_error",
        ),
        "cmb_derivation_mean_scalar_quotient_observer_count": _nested(
            cmb_derivation,
            "aggregate",
            "mean_scalar_quotient_observer_count",
        ),
        "cmb_derivation_mean_median_epsilon_cmi": _nested(
            cmb_derivation,
            "aggregate",
            "mean_median_epsilon_cmi",
        ),
        "oph_inflation_cmb_camb_written": bool(inflation_cmb_camb),
        "oph_inflation_cmb_camb_curve_comparable": bool(
            inflation_cmb_camb.get("measurement_comparable_cmb_curve", False)
        ),
        "oph_inflation_cmb_camb_physical_cmb_prediction": bool(
            inflation_cmb_camb.get("physical_cmb_prediction", False)
        ),
        "oph_inflation_cmb_camb_transfer_receipt": bool(
            inflation_cmb_camb.get("screen_camb_transfer_receipt", False)
        ),
        "oph_inflation_cmb_camb_n_s": _nested(inflation_cmb_camb, "oph_input", "n_s"),
        "oph_inflation_cmb_camb_A_zeta": _nested(inflation_cmb_camb, "oph_input", "A_zeta"),
        "oph_inflation_cmb_camb_q_IR": _nested(inflation_cmb_camb, "oph_input", "q_IR"),
        "oph_inflation_cmb_camb_ell_IR": _nested(inflation_cmb_camb, "oph_input", "ell_IR"),
        "oph_inflation_cmb_camb_lcdm_shape_correlation": _nested(
            inflation_cmb_camb,
            "comparison",
            "camb_lcdm_powerlaw",
            "shape_correlation",
        ),
        "oph_inflation_cmb_camb_p48_shape_correlation": _nested(
            inflation_cmb_camb,
            "comparison",
            "oph_p48_powerlaw",
            "shape_correlation",
        ),
        "oph_inflation_cmb_camb_p48_ir_shape_correlation": _nested(
            inflation_cmb_camb,
            "comparison",
            "oph_p48_ir_v04",
            "shape_correlation",
        ),
        "oph_inflation_cmb_camb_unique_n_s": _nested(inflation_cmb_camb, "oph_unique_input", "n_s"),
        "oph_inflation_cmb_camb_unique_q_IR": _nested(inflation_cmb_camb, "oph_unique_input", "q_IR"),
        "oph_inflation_cmb_camb_unique_ell_IR": _nested(inflation_cmb_camb, "oph_unique_input", "ell_IR"),
        "oph_inflation_cmb_camb_unique_shape_correlation": _nested(
            inflation_cmb_camb,
            "comparison",
            "oph_unique_ir_v09",
            "shape_correlation",
        ),
        "oph_inflation_cmb_camb_lcdm_chi2_per_bin": _nested(
            inflation_cmb_camb,
            "comparison",
            "camb_lcdm_powerlaw",
            "amplitude_fit_chi2_per_bin",
        ),
        "oph_inflation_cmb_camb_p48_chi2_per_bin": _nested(
            inflation_cmb_camb,
            "comparison",
            "oph_p48_powerlaw",
            "amplitude_fit_chi2_per_bin",
        ),
        "oph_inflation_cmb_camb_p48_ir_chi2_per_bin": _nested(
            inflation_cmb_camb,
            "comparison",
            "oph_p48_ir_v04",
            "amplitude_fit_chi2_per_bin",
        ),
        "oph_inflation_cmb_camb_unique_chi2_per_bin": _nested(
            inflation_cmb_camb,
            "comparison",
            "oph_unique_ir_v09",
            "amplitude_fit_chi2_per_bin",
        ),
        "oph_inflation_cmb_camb_acoustic_mean_abs_delta": _nested(
            inflation_cmb_camb,
            "acoustic_preservation",
            "mean_abs_fractional_delta_ell_ge_50",
        ),
        "oph_inflation_cmb_camb_unique_acoustic_mean_abs_delta": _nested(
            inflation_cmb_camb,
            "unique_acoustic_preservation",
            "mean_abs_fractional_delta_ell_ge_50",
        ),
        "oph_inflation_cmb_camb_lowell_lcdm_chi2": _nested(
            inflation_cmb_camb,
            "low_ell_v04_diagnostic",
            "CAMB_LCDM_chi2_ell2_29",
        ),
        "oph_inflation_cmb_camb_lowell_oph_ir_chi2": _nested(
            inflation_cmb_camb,
            "low_ell_v04_diagnostic",
            "CAMB_OPH_IR_chi2_ell2_29",
        ),
        "oph_selector_elimination_written": bool(selector_elimination),
        "oph_selector_elimination_theorem_receipt": bool(
            selector_elimination.get("THEOREM_SIDE_SELECTOR_ELIMINATION_RECEIPT")
            or selector_elimination.get("theorem_side_receipt")
        ),
        "oph_selector_elimination_source_audit_receipt": bool(
            selector_elimination.get("SOURCE_PACKET_AUDIT_RECEIPT")
            or selector_elimination.get("source_packet_audit_receipt")
        ),
        "oph_selector_q_IR_removed": bool(
            _nested(selector_elimination, "selector_elimination", "q_IR_selector_removed")
            or selector_elimination.get("q_IR_selector_removed")
        ),
        "oph_selector_ell_IR_removed": bool(
            _nested(selector_elimination, "selector_elimination", "ell_IR_selector_removed")
            or selector_elimination.get("ell_IR_selector_removed")
        ),
        "oph_selector_eta_R_reduced_to_repair_clock": bool(
            _nested(selector_elimination, "selector_elimination", "eta_R_reduced_to_repair_clock_certificate")
            or selector_elimination.get("eta_R_reduced_to_repair_clock_certificate")
        ),
        "oph_selector_finite_lattice_derived": bool(selector_elimination.get("finite_lattice_derived", False)),
        "oph_selector_physical_cmb_prediction": bool(selector_elimination.get("physical_cmb_prediction", False)),
        "oph_selector_n_s": (
            _nested(selector_elimination, "scalar_tilt", "n_s")
            or _nested(exact_cmb_camb, "oph_exact_input", "n_s")
        ),
        "oph_selector_eta_R": (
            _nested(selector_elimination, "scalar_tilt", "eta_R")
            or _nested(exact_cmb_camb, "oph_exact_input", "eta_R")
        ),
        "oph_selector_q_IR": (
            _nested(selector_elimination, "cmb_ir_kernel", "q_IR")
            or _nested(selector_elimination, "selector_elimination", "q_IR")
            or _nested(exact_cmb_camb, "oph_exact_input", "q_IR")
        ),
        "oph_selector_ell_IR": (
            _nested(selector_elimination, "cmb_ir_kernel", "ell_IR")
            or _nested(selector_elimination, "selector_elimination", "ell_IR")
            or _nested(exact_cmb_camb, "oph_exact_input", "ell_IR")
        ),
        "oph_selector_kernel_csv_passed": bool(
            _nested(selector_elimination, "exact_ir_kernel_csv_audit", "passed")
        ),
        "oph_selector_kernel_csv_max_abs_error": _nested(
            selector_elimination,
            "exact_ir_kernel_csv_audit",
            "max_abs_error",
        ),
        "oph_selector_kappa_rep_status": _nested(selector_elimination, "scalar_tilt", "canonical_kappa_rep_status"),
        "oph_exact_cmb_camb_written": bool(exact_cmb_camb),
        "oph_exact_cmb_camb_curve_comparable": bool(
            exact_cmb_camb.get("measurement_comparable_cmb_curve", False)
        ),
        "oph_exact_cmb_camb_physical_cmb_prediction": bool(
            exact_cmb_camb.get("physical_cmb_prediction", False)
        ),
        "oph_exact_cmb_camb_transfer_receipt": bool(
            exact_cmb_camb.get("screen_camb_transfer_receipt", False)
        ),
        "oph_exact_cmb_camb_n_s": _nested(exact_cmb_camb, "oph_exact_input", "n_s"),
        "oph_exact_cmb_camb_eta_R": _nested(exact_cmb_camb, "oph_exact_input", "eta_R"),
        "oph_exact_cmb_camb_q_IR": _nested(exact_cmb_camb, "oph_exact_input", "q_IR"),
        "oph_exact_cmb_camb_ell_IR": _nested(exact_cmb_camb, "oph_exact_input", "ell_IR"),
        "oph_exact_cmb_camb_N_frz_proxy": _nested(exact_cmb_camb, "oph_exact_input", "N_frz_proxy"),
        "oph_exact_cmb_camb_lcdm_shape_correlation": _nested(
            exact_cmb_camb,
            "comparison",
            "camb_lcdm_powerlaw",
            "shape_correlation",
        ),
        "oph_exact_cmb_camb_scalar_shape_correlation": _nested(
            exact_cmb_camb,
            "comparison",
            "oph_exact_scalar_tilt",
            "shape_correlation",
        ),
        "oph_exact_cmb_camb_ir_shape_correlation": _nested(
            exact_cmb_camb,
            "comparison",
            "oph_exact_ir_v10",
            "shape_correlation",
        ),
        "oph_exact_cmb_camb_lcdm_chi2_per_bin": _nested(
            exact_cmb_camb,
            "comparison",
            "camb_lcdm_powerlaw",
            "amplitude_fit_chi2_per_bin",
        ),
        "oph_exact_cmb_camb_scalar_chi2_per_bin": _nested(
            exact_cmb_camb,
            "comparison",
            "oph_exact_scalar_tilt",
            "amplitude_fit_chi2_per_bin",
        ),
        "oph_exact_cmb_camb_ir_chi2_per_bin": _nested(
            exact_cmb_camb,
            "comparison",
            "oph_exact_ir_v10",
            "amplitude_fit_chi2_per_bin",
        ),
        "oph_exact_cmb_camb_acoustic_mean_abs_delta": _nested(
            exact_cmb_camb,
            "acoustic_preservation",
            "mean_abs_fractional_delta_ell_ge_50",
        ),
        "oph_exact_cmb_camb_official_clik_ready": _nested(
            exact_cmb_camb,
            "official_planck_likelihood_readiness",
            "official_clik_api_available",
        ),
        "oph_exact_cmb_camb_official_likelihood_ready": _nested(
            exact_cmb_camb,
            "official_planck_likelihood_readiness",
            "official_likelihood_execution_ready",
        ),
        "finite_clock_cmb_camb_written": bool(finite_clock_cmb_camb),
        "finite_clock_cmb_camb_curve_comparable": bool(
            finite_clock_cmb_camb.get("measurement_comparable_cmb_curve", False)
        ),
        "finite_clock_cmb_camb_physical_cmb_prediction": bool(
            finite_clock_cmb_camb.get("physical_cmb_prediction", False)
        ),
        "finite_clock_cmb_camb_transfer_receipt": bool(
            finite_clock_cmb_camb.get("screen_camb_transfer_receipt", False)
        ),
        "finite_clock_cmb_camb_finite_lattice_clock_derived": bool(
            finite_clock_cmb_camb.get("finite_lattice_clock_derived", False)
        ),
        "finite_clock_cmb_camb_repair_clock_certificate": bool(
            finite_clock_cmb_camb.get("repair_clock_certificate", False)
        ),
        "finite_clock_cmb_camb_clock_numeric_match": bool(
            _nested(
                finite_clock_cmb_camb,
                "finite_repair_clock_input",
                "clock_normalization_numeric_match",
            )
        ),
        "finite_clock_cmb_camb_repair_scale_hypothesis_match": bool(
            _nested(
                finite_clock_cmb_camb,
                "finite_repair_clock_input",
                "repair_scale_hypothesis_clock_match",
            )
        ),
        "finite_clock_cmb_camb_clock_source_theorem_grade": bool(
            _nested(
                finite_clock_cmb_camb,
                "finite_repair_clock_input",
                "clock_normalization_source_status",
                "theorem_grade",
            )
        ),
        "finite_clock_cmb_camb_clock_source": _nested(
            finite_clock_cmb_camb,
            "finite_repair_clock_input",
            "clock_normalization_source",
        ),
        "finite_clock_cmb_camb_selector_ir_theory_side": bool(
            finite_clock_cmb_camb.get("selector_ir_theory_side", False)
        ),
        "finite_clock_cmb_camb_n_s": _nested(
            finite_clock_cmb_camb,
            "finite_repair_clock_input",
            "n_s",
        ),
        "finite_clock_cmb_camb_eta_R": _nested(
            finite_clock_cmb_camb,
            "finite_repair_clock_input",
            "eta_R",
        ),
        "finite_clock_cmb_camb_kappa_rep": _nested(
            finite_clock_cmb_camb,
            "finite_repair_clock_input",
            "kappa_rep",
        ),
        "finite_clock_cmb_camb_q_IR": _nested(finite_clock_cmb_camb, "selector_ir_input", "q_IR"),
        "finite_clock_cmb_camb_ell_IR": _nested(finite_clock_cmb_camb, "selector_ir_input", "ell_IR"),
        "finite_clock_cmb_camb_lcdm_shape_correlation": _nested(
            finite_clock_cmb_camb,
            "comparison",
            "camb_lcdm_powerlaw",
            "shape_correlation",
        ),
        "finite_clock_cmb_camb_scalar_shape_correlation": _nested(
            finite_clock_cmb_camb,
            "comparison",
            "finite_repair_clock_scalar_tilt",
            "shape_correlation",
        ),
        "finite_clock_cmb_camb_ir_shape_correlation": _nested(
            finite_clock_cmb_camb,
            "comparison",
            "finite_repair_clock_plus_selector_ir",
            "shape_correlation",
        ),
        "finite_clock_cmb_camb_lcdm_chi2_per_bin": _nested(
            finite_clock_cmb_camb,
            "comparison",
            "camb_lcdm_powerlaw",
            "amplitude_fit_chi2_per_bin",
        ),
        "finite_clock_cmb_camb_scalar_chi2_per_bin": _nested(
            finite_clock_cmb_camb,
            "comparison",
            "finite_repair_clock_scalar_tilt",
            "amplitude_fit_chi2_per_bin",
        ),
        "finite_clock_cmb_camb_ir_chi2_per_bin": _nested(
            finite_clock_cmb_camb,
            "comparison",
            "finite_repair_clock_plus_selector_ir",
            "amplitude_fit_chi2_per_bin",
        ),
        "finite_clock_cmb_camb_acoustic_mean_abs_delta": _nested(
            finite_clock_cmb_camb,
            "acoustic_preservation",
            "mean_abs_fractional_delta_ell_ge_50",
        ),
        "oph_boltzmann_input_written": bool(boltzmann_inputs),
        "oph_boltzmann_source_report_count": boltzmann_inputs.get("source_report_count") if boltzmann_inputs else None,
        "oph_boltzmann_cdm_limit_ready": _nested(boltzmann_inputs, "readiness", "cdm_limit_solver_ready"),
        "oph_boltzmann_diagnostic_ready": _nested(
            boltzmann_inputs, "readiness", "diagnostic_repair_exchange_table_ready"
        ),
        "oph_boltzmann_physical_prediction_ready": _nested(
            boltzmann_inputs, "readiness", "physical_prediction_ready"
        ),
        "oph_boltzmann_cdm_row_count": _nested(boltzmann_inputs, "cdm_limit", "row_count"),
        "oph_boltzmann_diagnostic_row_count": _nested(
            boltzmann_inputs, "diagnostic_repair_exchange", "row_count"
        ),
        "oph_boltzmann_missing_gate_count": len(_nested(boltzmann_inputs, "readiness", "missing_gates") or [])
        if boltzmann_inputs
        else None,
        "oph_boltzmann_mean_gamma_proxy": _mean(
            row.get("Gamma_rec_over_H_shape_proxy") for row in boltzmann_diagnostic_rows
        ),
        "oph_boltzmann_mean_B_A_shape_proxy": _mean(row.get("B_A_shape_proxy") for row in boltzmann_diagnostic_rows),
        "finite_collar_boltzmann_written": bool(finite_collar_boltzmann),
        "finite_collar_boltzmann_bundle_receipt": bool(
            finite_collar_boltzmann.get("FINITE_COLLAR_BOLTZMANN_SOURCE_BUNDLE_RECEIPT", False)
        ),
        "finite_collar_boltzmann_physical_certificate": bool(
            finite_collar_boltzmann.get("PHYSICAL_BOLTZMANN_EXPORT_CERTIFICATE", False)
        ),
        "finite_collar_boltzmann_physical_cmb_prediction": bool(
            finite_collar_boltzmann.get("physical_cmb_prediction", False)
        ),
        "finite_collar_boltzmann_b_a_row_count": _nested(
            finite_collar_boltzmann, "B_A_k_a_diagnostic", "row_count"
        ),
        "finite_collar_boltzmann_rho_row_count": _nested(
            finite_collar_boltzmann, "rho_A_a_diagnostic", "row_count"
        ),
        "finite_collar_boltzmann_gamma_row_count": _nested(
            finite_collar_boltzmann, "Gamma_rec_k_a_diagnostic", "row_count"
        ),
        "finite_collar_boltzmann_no_data": _nested(
            finite_collar_boltzmann, "readiness", "checks", "no_data_use_receipt"
        ),
        "finite_collar_boltzmann_physical_missing_gate_count": len(
            _nested(finite_collar_boltzmann, "readiness", "physical_missing_gates") or []
        )
        if finite_collar_boltzmann
        else None,
        "finite_collar_boltzmann_diagnostic_missing_gates": _nested(
            finite_collar_boltzmann, "readiness", "diagnostic_missing_gates"
        )
        or [],
        "finite_collar_boltzmann_physical_missing_gates": _nested(
            finite_collar_boltzmann, "readiness", "physical_missing_gates"
        )
        or [],
        "finite_collar_boltzmann_mean_B_A": _nested(
            finite_collar_boltzmann, "readiness", "mean_abs_B_A"
        ),
        "finite_collar_boltzmann_mean_Gamma_rec": _nested(
            finite_collar_boltzmann, "readiness", "mean_Gamma_rec_over_H"
        ),
        "finite_collar_projection_written": bool(finite_collar_projection),
        "finite_collar_projection_receipt": bool(
            finite_collar_projection.get("FINITE_COLLAR_CMB_PROJECTION_DIAGNOSTIC_RECEIPT", False)
        ),
        "finite_collar_projection_physical_k_receipt": bool(
            finite_collar_projection.get("PHYSICAL_K_CALIBRATION_RECEIPT", False)
        ),
        "finite_collar_projection_physical_cmb_prediction": bool(
            finite_collar_projection.get("physical_cmb_prediction", False)
        ),
        "finite_collar_projection_row_count": len(finite_collar_projection.get("projected_B_A_rows") or [])
        if finite_collar_projection
        else None,
        "finite_collar_projection_background_row_count": len(
            finite_collar_projection.get("background_rows") or []
        )
        if finite_collar_projection
        else None,
        "finite_collar_projection_ell_min": _nested(finite_collar_projection, "shape_summary", "ell_min"),
        "finite_collar_projection_ell_max": _nested(finite_collar_projection, "shape_summary", "ell_max"),
        "finite_collar_projection_mean_B_A": _nested(finite_collar_projection, "shape_summary", "mean_B_A"),
        "finite_collar_projection_mean_abs_B_A": _nested(
            finite_collar_projection, "shape_summary", "mean_abs_B_A"
        ),
        "finite_collar_projection_positive_fraction": _nested(
            finite_collar_projection, "shape_summary", "positive_fraction"
        ),
        "finite_collar_projection_log_slope": _nested(
            finite_collar_projection, "shape_summary", "log_abs_B_A_vs_log_ell_slope"
        ),
        "finite_collar_projection_largest_scale_B_A": _nested(
            finite_collar_projection, "shape_summary", "largest_scale_mean_B_A"
        ),
        "finite_collar_projection_smallest_scale_B_A": _nested(
            finite_collar_projection, "shape_summary", "smallest_scale_mean_B_A"
        ),
        "oph_cmb_adapter_written": bool(oph_cmb),
        "oph_cmb_boltzmann_ready": _nested(oph_cmb, "physical_prediction_readiness", "boltzmann_ready"),
        "oph_cmb_physical_prediction_ready": _nested(
            oph_cmb,
            "physical_prediction_readiness",
            "physical_cmb_prediction_ready",
        ),
        "oph_cmb_physical_cmb_prediction": bool(oph_cmb.get("physical_cmb_prediction", False)),
        "oph_cmb_cosmology_perturbation_receipt": bool(oph_cmb.get("COSMOLOGY_PERTURBATION_RECEIPT", False)),
        "oph_cmb_parent_sample_count": _nested(oph_cmb, "finite_collar_parent", "sample_count"),
        "oph_cmb_parent_R": _nested(oph_cmb, "finite_collar_parent", "weighted_collar_repair_defect_R"),
        "oph_cmb_parent_rho_A_eq_proxy": _nested(oph_cmb, "finite_collar_parent", "rho_A_eq_proxy"),
        "oph_cmb_kernel_proxy_row_count": len(
            _nested(oph_cmb, "diagnostic_kernel_proxy", "kernel_proxy_rows") or []
        )
        if oph_cmb
        else None,
        "oph_cmb_missing_gate_count": len(
            _nested(oph_cmb, "physical_prediction_readiness", "missing_gates") or []
        )
        if oph_cmb
        else None,
        "sync_gap_written": bool(sync_gap),
        "sync_gap_source_run_count": sync_gap.get("run_count") if sync_gap else None,
        "sync_gap_low_k_established": bool(sync_gap.get("low_k_gap_established", False)),
        "sync_gap_same_boundary_selector_established": bool(
            sync_gap.get("same_boundary_selector_established", False)
        ),
        "sync_gap_inflation_replacement_ready": bool(sync_gap.get("inflation_replacement_ready", False)),
        "sync_gap_time_resolved_trace_count": _nested(sync_gap, "aggregate", "time_resolved_trace_count"),
        "sync_gap_cached_proxy_pass_count": _nested(sync_gap, "aggregate", "cached_proxy_pass_count"),
        "sync_gap_time_resolved_gap_pass_count": _nested(sync_gap, "aggregate", "time_resolved_gap_pass_count"),
        "sync_gap_mean_global_phi_gamma": _nested(sync_gap, "aggregate", "mean_global_phi_gamma_per_cycle"),
        "sync_gap_median_global_phi_gamma": _nested(sync_gap, "aggregate", "median_global_phi_gamma_per_cycle"),
        "sync_gap_residual_candidate_receipt": _sync_gap_residual_candidate_value(sync_gap, "candidate_receipt"),
        "sync_gap_residual_candidate_field": _sync_gap_residual_candidate_value(sync_gap, "field"),
        "sync_gap_residual_candidate_positive_gamma_fraction": _sync_gap_residual_candidate_value(
            sync_gap, "positive_gamma_fraction"
        ),
        "sync_gap_residual_candidate_control_separation_fraction": _sync_gap_residual_candidate_value(
            sync_gap, "control_separation_positive_fraction"
        ),
        "sync_gap_residual_candidate_median_gamma": _sync_gap_residual_candidate_value(
            sync_gap, "median_gamma_per_cycle"
        ),
        "sync_gap_residual_candidate_median_control_gap": _sync_gap_residual_candidate_value(
            sync_gap, "median_target_minus_max_control_gamma"
        ),
        "hot_release_written": bool(hot_release),
        "hot_release_source_run_count": hot_release.get("run_count") if hot_release else None,
        "hot_release_theorem_ready": bool(hot_release.get("hot_release_theorem_ready", False)),
        "hot_release_mechanical_surface_count": _nested(
            hot_release,
            "aggregate",
            "mechanical_release_surface_count",
        ),
        "hot_release_collar_gate_pass_count": _nested(
            hot_release,
            "aggregate",
            "collar_markov_gate_pass_count",
        ),
        "hot_release_theorem_ready_count": _nested(
            hot_release,
            "aggregate",
            "hot_release_theorem_ready_count",
        ),
        "hot_release_median_release_cycle": _nested(hot_release, "aggregate", "median_release_cycle"),
        "hot_release_mean_release_cycle": _nested(hot_release, "aggregate", "mean_release_cycle"),
        "hot_release_mean_median_epsilon_cmi": _nested(hot_release, "aggregate", "mean_median_epsilon_cmi"),
        "adiabaticity_written": bool(adiabaticity),
        "adiabaticity_source_run_count": adiabaticity.get("run_count") if adiabaticity else None,
        "adiabaticity_established": bool(adiabaticity.get("adiabaticity_established", False)),
        "adiabaticity_proxy_pass_count": _nested(
            adiabaticity,
            "aggregate",
            "adiabaticity_proxy_pass_count",
        ),
        "adiabaticity_mean_max_entropy_residual_std": _nested(
            adiabaticity,
            "aggregate",
            "mean_max_entropy_residual_std",
        ),
        "adiabaticity_mean_min_common_clock_corr": _nested(
            adiabaticity,
            "aggregate",
            "mean_min_common_clock_corr",
        ),
        "h0s8_written": bool(h0s8),
        "h0s8_H0_km_s_Mpc": _nested(h0s8, "flat_q_a_closure", "H0_km_s_Mpc"),
        "h0s8_Omega_m": _nested(h0s8, "flat_q_a_closure", "Omega_m"),
        "h0s8_Omega_A": _nested(h0s8, "flat_q_a_closure", "Omega_A"),
        "h0s8_Omega_Lambda": _nested(h0s8, "flat_q_a_closure", "Omega_Lambda_OPH"),
        "h0s8_flat_sum": _nested(h0s8, "flat_q_a_closure", "flat_sum"),
        "h0s8_lambda_collar": _nested(h0s8, "collar_tracking", "lambda_collar"),
        "h0s8_f_A": _nested(h0s8, "collar_tracking", "f_A"),
        "h0s8_mu_eff_source_suppression": _nested(h0s8, "collar_tracking", "mu_eff_source_suppression"),
        "h0s8_source_suppression_fraction": _nested(h0s8, "collar_tracking", "source_suppression_fraction"),
        "h0s8_cdm_like_S8": _nested(h0s8, "branches", "A_conserved_cdm_like", "S8"),
        "h0s8_cdm_like_sigma8": _nested(h0s8, "branches", "A_conserved_cdm_like", "sigma8"),
        "h0s8_direct_jacobi_S8": _nested(h0s8, "branches", "B_direct_jacobi_repair", "S8"),
        "h0s8_direct_jacobi_growth": _nested(
            h0s8,
            "branches",
            "B_direct_jacobi_repair",
            "growth_suppression_factor",
        ),
        "h0s8_matrix_gap_S8": _nested(h0s8, "branches", "C_matrix_gapped_jacobi", "S8"),
        "h0s8_matrix_gap_growth": _nested(
            h0s8,
            "branches",
            "C_matrix_gapped_jacobi",
            "growth_suppression_factor",
        ),
        "h0s8_planck_H0_pull_sigma": _nested(
            h0s8,
            "measurement_comparisons",
            "Planck2018_H0",
            "branch_pull_sigma",
        ),
        "h0s8_shoes_H0_pull_sigma": _nested(
            h0s8,
            "measurement_comparisons",
            "SH0ES_H0",
            "branch_pull_sigma",
        ),
        "h0s8_planck_S8_cdm_pull_sigma": _nested(
            h0s8,
            "measurement_comparisons",
            "Planck2018_S8",
            "cdm_pull_sigma",
        ),
        "h0s8_weak_lensing_cdm_pull_sigma": _nested(
            h0s8,
            "measurement_comparisons",
            "weak_lensing_S8_target",
            "cdm_pull_sigma",
        ),
        "h0s8_direct_jacobi_weak_lensing_pull_sigma": _nested(
            h0s8,
            "measurement_comparisons",
            "weak_lensing_S8_target",
            "direct_jacobi_pull_sigma",
        ),
        "h0s8_Q_A_gate": bool(_nested(h0s8, "theorem_gates", "Q_A_from_finite_collar_selector")),
        "h0s8_B_A_gate": bool(_nested(h0s8, "theorem_gates", "B_A_from_parent_collar_kernel")),
        "h0s8_lambda_P_gate": bool(_nested(h0s8, "theorem_gates", "lambda_collar_from_P_survival")),
        "h0s8_Gamma_J_gate": bool(_nested(h0s8, "theorem_gates", "Gamma_rec_equals_Jacobi_clock")),
        "h0s8_camb_class_gate": bool(_nested(h0s8, "theorem_gates", "full_CAMB_CLASS_anomaly_module")),
        "h0s8_likelihood_gate": bool(_nested(h0s8, "theorem_gates", "full_likelihood_contract")),
        "h0s8_lane8_written": bool(h0s8_lane8),
        "h0s8_lane8_values_are_run_derived": bool(h0s8_lane8.get("values_are_run_derived", False)),
        "h0s8_lane8_i0_bits": _nested(h0s8_lane8, "low_entropy_certificate", "i0_bits"),
        "h0s8_lane8_b0_bits": _nested(h0s8_lane8, "low_entropy_certificate", "b0_bits"),
        "h0s8_lane8_p0_bits": _nested(h0s8_lane8, "low_entropy_certificate", "p0_bits"),
        "h0s8_lane8_a0_bits": _nested(h0s8_lane8, "low_entropy_certificate", "a0_bits"),
        "h0s8_lane8_low_entropy_gap_bits": _nested(
            h0s8_lane8,
            "low_entropy_certificate",
            "low_entropy_gap_bits",
        ),
        "h0s8_lane8_fake_gamma_margin_bits": _nested(
            h0s8_lane8,
            "fake_history_certificate",
            "gamma_margin_bits",
        ),
        "h0s8_lane8_fake_probability_bound": _nested(
            h0s8_lane8,
            "fake_history_certificate",
            "probability_bound",
        ),
        "h0s8_lane8_payload_gate": bool(
            _nested(h0s8_lane8, "theorem_gates", "audited_record_payload_lower_bound")
        ),
        "h0s8_lane8_fake_suppression_gate": bool(
            _nested(h0s8_lane8, "theorem_gates", "fake_trial_suppression")
        ),
        "h0s8_lane8_selector_gate": bool(
            _nested(h0s8_lane8, "theorem_gates", "selector_dominance_margin_positive")
        ),
        "h0s8_lane8_refinement_gate": bool(
            _nested(h0s8_lane8, "theorem_gates", "refinement_stability")
        ),
        "h0s8_lane8_certificate_ready": bool(
            _nested(h0s8_lane8, "theorem_gates", "low_entropy_ancestry_certificate_ready")
        ),
        "h0s8_physical_prediction_ready": bool(h0s8.get("physical_prediction_ready", False)),
        "h0s8_physical_cmb_prediction": bool(h0s8.get("physical_cmb_prediction", False)),
        "h0s8_physical_matter_power_prediction": bool(
            h0s8.get("physical_matter_power_prediction", False)
        ),
        "galaxy_proxy_written": bool(galaxy),
        "galaxy_proxy_receipt": bool(galaxy.get("GALAXY_PROXY_RECEIPT", False)),
        "galaxy_proxy_a0_oph": galaxy.get("a0_oph"),
        "galaxy_proxy_a0_eff": galaxy.get("a0_eff"),
        "galaxy_proxy_lambda_collar_declared": galaxy.get("lambda_collar_declared"),
        "galaxy_proxy_rar_point_count": len(galaxy.get("rar_curve", [])) if galaxy else None,
        "galaxy_proxy_lambda_fit_usable": _nested(galaxy, "lambda_collar_estimate", "usable"),
        "galaxy_proxy_lambda_fit": _nested(galaxy, "lambda_collar_estimate", "lambda_collar"),
        "galaxy_proxy_btfr_usable": _nested(galaxy, "btfr", "usable"),
        "galaxy_proxy_btfr_slope": _nested(galaxy, "btfr", "slope_logM_vs_logV"),
        "galaxy_proxy_disk_usable": _nested(galaxy, "disk_potential_residual", "usable"),
        "galaxy_proxy_physical_claim": bool(galaxy.get("physical_claim", False)),
        "static_galaxy_written": bool(static_galaxy),
        "static_galaxy_receipt": bool(static_galaxy.get("STATIC_GALAXY_RAR_BTFR_RECEIPT", False)),
        "static_galaxy_bridge_receipt": bool(static_galaxy.get("OPH_STATIC_GALAXY_BRIDGE_RECEIPT", False)),
        "static_galaxy_bridge": static_galaxy.get("bridge"),
        "static_galaxy_claim_tier": static_galaxy.get("claim_tier"),
        "static_galaxy_bulk_required": static_galaxy.get("bulk_required"),
        "static_galaxy_physical_cmb_claim": bool(static_galaxy.get("physical_cmb_claim", False)),
        "static_galaxy_physical_matter_power_claim": bool(
            static_galaxy.get("physical_matter_power_claim", False)
        ),
        "static_galaxy_physical_claim": bool(static_galaxy.get("physical_claim", False)),
        "static_galaxy_dataset_row_count": static_galaxy.get("dataset_row_count"),
        "static_galaxy_dataset_galaxy_count": static_galaxy.get("dataset_galaxy_count"),
        "static_galaxy_galaxy_count": static_galaxy.get("galaxy_count"),
        "static_galaxy_measurement_galaxy_count": static_galaxy.get("measurement_galaxy_count"),
        "static_galaxy_rar_galaxy_count": static_galaxy.get("rar_galaxy_count"),
        "static_galaxy_rar_galaxy_support_count": static_galaxy.get("rar_galaxy_support_count"),
        "static_galaxy_rar_point_count": static_galaxy.get("rar_point_count"),
        "static_galaxy_shared_a0": static_galaxy.get("shared_a0"),
        "static_galaxy_shared_lambda_collar": static_galaxy.get("shared_lambda_collar"),
        "static_galaxy_rar_scatter_dex": static_galaxy.get("rar_scatter_dex"),
        "static_galaxy_btfr_usable": _nested(static_galaxy, "btfr", "usable"),
        "static_galaxy_btfr_galaxy_count": _nested(static_galaxy, "btfr", "galaxy_count"),
        "static_galaxy_btfr_slope": _nested(static_galaxy, "btfr", "slope_logM_vs_logV"),
        "static_galaxy_btfr_intercept": _nested(static_galaxy, "btfr", "intercept_logM_vs_logV"),
        "static_galaxy_btfr_predicted_slope": _nested(
            static_galaxy, "btfr_prediction_from_rar_fit", "predicted_slope_logM_vs_logV"
        ),
        "static_galaxy_btfr_predicted_intercept": _nested(
            static_galaxy, "btfr_prediction_from_rar_fit", "predicted_intercept_logM_vs_logV"
        ),
        "static_galaxy_btfr_slope_delta": _nested(
            static_galaxy, "btfr_prediction_from_rar_fit", "slope_delta_observed_minus_predicted"
        ),
        "static_galaxy_btfr_intercept_delta": _nested(
            static_galaxy, "btfr_prediction_from_rar_fit", "intercept_delta_observed_minus_predicted"
        ),
        "static_galaxy_btfr_abs_slope_delta": _nested(
            static_galaxy, "btfr_prediction_from_rar_fit", "abs_slope_delta"
        ),
        "static_galaxy_btfr_abs_intercept_delta": _nested(
            static_galaxy, "btfr_prediction_from_rar_fit", "abs_intercept_delta_dex"
        ),
        "static_galaxy_holdout_usable": _nested(static_galaxy, "holdout_validation", "usable"),
        "static_galaxy_holdout_receipt": bool(_nested(static_galaxy, "holdout_validation", "receipt")),
        "static_galaxy_holdout_train_galaxy_count": _nested(
            static_galaxy, "holdout_validation", "train_galaxy_count"
        ),
        "static_galaxy_holdout_test_galaxy_count": _nested(
            static_galaxy, "holdout_validation", "test_galaxy_count"
        ),
        "static_galaxy_holdout_train_point_count": _nested(
            static_galaxy, "holdout_validation", "train_point_count"
        ),
        "static_galaxy_holdout_test_point_count": _nested(static_galaxy, "holdout_validation", "test_point_count"),
        "static_galaxy_holdout_shared_a0": _nested(static_galaxy, "holdout_validation", "shared_a0"),
        "static_galaxy_holdout_shared_lambda_collar": _nested(
            static_galaxy, "holdout_validation", "shared_lambda_collar"
        ),
        "static_galaxy_holdout_train_log_accel_rmse": _nested(
            static_galaxy, "holdout_validation", "train", "log_acceleration_rmse_dex"
        ),
        "static_galaxy_holdout_test_log_accel_rmse": _nested(
            static_galaxy, "holdout_validation", "test", "log_acceleration_rmse_dex"
        ),
        "static_galaxy_holdout_test_velocity_rmse": _nested(
            static_galaxy, "holdout_validation", "test", "velocity_rmse_km_s"
        ),
        "static_galaxy_holdout_test_baryon_velocity_rmse": _nested(
            static_galaxy, "holdout_validation", "test", "baryon_only_velocity_rmse_km_s"
        ),
        "static_galaxy_holdout_test_velocity_improvement": _nested(
            static_galaxy, "holdout_validation", "test", "velocity_rmse_improvement_fraction"
        ),
        "static_galaxy_holdout_test_chi2_proxy": _nested(
            static_galaxy, "holdout_validation", "test", "velocity_chi2_proxy_per_point"
        ),
        "h3_receipt": bool(h3.get("h3_bulk_candidate_receipt", h3.get("MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT", False))),
        "h3_control_separation_receipt": bool(
            h3.get("H3_RESPONSE_CONTROL_SEPARATION_RECEIPT", False)
            or h3.get("h3_control_separation_receipt", False)
            or h3_stage_gates.get("H3_RESPONSE_CONTROL_SEPARATION_RECEIPT", False)
            or h3_stage_gates.get("intermediate_control_separation_receipt", False)
        ),
        "h3_signal_gate": bool(h3_stage_gates.get("signal_gate", False)),
        "h3_geometry_gate": bool(h3_stage_gates.get("geometry_gate", False)),
        "h3_aggregate_wrong_scale_gate": bool(h3_stage_gates.get("aggregate_wrong_scale_gate", False)),
        "h3_material_feature_gate": bool(h3_stage_gates.get("material_feature_gate", False)),
        "h3_channel_mode": h3.get("channel_mode")
        or h3_fit.get("channel_mode")
        or ("time_observable_class_legacy" if h3 else None),
        "h3_channel_count": h3_fit.get("channel_count"),
        "h3_profile_mode": h3.get("profile_mode")
        or h3_fit.get("profile_mode")
        or ("static_halfspace_legacy" if h3 else None),
        "h3_rmse": h3_fit.get("heldout_normalized_rmse"),
        "h3_explained_variance": h3_fit.get("heldout_explained_variance"),
        "h3_chart_dimension_point_count": h3_chart_dimension.get("point_count") if h3_chart_dimension else None,
        "h3_chart_candidate_3d_dimension_window": h3_chart_dimension.get("candidate_3d_dimension_window")
        if h3_chart_dimension
        else None,
        "h3_chart_dimension_estimators_agree": h3_chart_dimension.get("dimension_estimators_agree")
        if h3_chart_dimension
        else None,
        "h3_chart_correlation_dimension_estimate": _dimension_estimate_or_none(
            h3_chart_corr.get("estimate") if isinstance(h3_chart_corr, dict) else None
        ),
        "h3_chart_local_mle_dimension_estimate": _dimension_estimate_or_none(
            h3_chart_mle.get("estimate") if isinstance(h3_chart_mle, dict) else None
        ),
        "s2_boundary_rmse": s2.get("heldout_normalized_rmse"),
        "shuffled_response_rmse": _nested(controls, "shuffled_response", "heldout_normalized_rmse"),
        "shuffled_observer_labels_rmse": _nested(controls, "shuffled_observer_labels", "heldout_normalized_rmse"),
        "no_perturbation_rmse": _nested(controls, "no_perturbation", "heldout_normalized_rmse"),
        "wrong_1x_rmse": _nested(wrong, "1x", "heldout_normalized_rmse"),
        "wrong_pi_rmse": _nested(wrong, "pi", "heldout_normalized_rmse"),
        "wrong_4pi_rmse": _nested(wrong, "4pi", "heldout_normalized_rmse"),
        "h3_wrong_scale_audit_eligible": bool(wrong_audit.get("eligible", False)),
        "h3_wrong_scale_audited_feature_count": wrong_audit.get("audited_feature_count"),
        "h3_wrong_scale_win_count": wrong_audit.get("wrong_scale_win_count"),
        "h3_wrong_scale_win_fraction": wrong_audit.get("wrong_scale_win_fraction"),
        "h3_material_wrong_scale_win_count": wrong_audit.get("material_wrong_scale_win_count"),
        "h3_material_wrong_scale_win_fraction": wrong_audit.get("material_wrong_scale_win_fraction"),
        "h3_two_pi_fit_win_fraction": wrong_audit.get("two_pi_h3_fit_win_fraction"),
        "h3_material_two_pi_fit_win_fraction": wrong_audit.get("material_two_pi_h3_fit_win_fraction"),
        "h3_wrong_scale_red_flag": bool(wrong_audit.get("red_flag_wrong_scale_wins", False)),
        "h3_material_wrong_scale_red_flag": bool(wrong_audit.get("material_red_flag_wrong_scale_wins", False)),
        "h3_wrong_scale_winner_counts": wrong_audit.get("winner_counts"),
        "h3_material_wrong_scale_winner_counts": wrong_audit.get("material_winner_counts"),
        "h3_wrong_scale_worst_group": (wrong_audit.get("worst_groups") or [None])[0],
        "h3_seed_ensemble_written": bool(h3_ensemble),
        "h3_seed_ensemble_seed_count": h3_ensemble.get("seed_count"),
        "h3_seed_ensemble_receipt_count": h3_ensemble.get("receipt_count"),
        "h3_seed_ensemble_receipt_fraction": h3_ensemble.get("receipt_fraction"),
        "h3_seed_ensemble_control_separation_fraction": h3_ensemble.get("control_separation_receipt_fraction"),
        "h3_seed_ensemble_candidate_receipt_fraction": h3_ensemble.get("candidate_receipt_fraction"),
        "h3_seed_ensemble_dim3_count": h3_ensemble.get("candidate_3d_window_count"),
        "h3_seed_ensemble_dim3_fraction": h3_ensemble.get("candidate_3d_window_fraction"),
        "h3_response_seed_robust_receipt": bool(h3_ensemble.get("h3_response_seed_robust_receipt", False)),
        "h3_chart_3d_seed_robust_receipt": bool(h3_ensemble.get("h3_chart_3d_seed_robust_receipt", False)),
        "h3_seed_ensemble_mean_nrmse": h3_ensemble.get("mean_heldout_normalized_rmse"),
        "h3_seed_ensemble_mean_ev": h3_ensemble.get("mean_heldout_explained_variance"),
        "h3_seed_ensemble_p75_material_wrong": h3_ensemble.get("p75_material_wrong_scale_win_fraction"),
        "minimal_caps_to_h3_written": bool(caps_to_h3),
        "minimal_caps_to_h3_receipt": bool(caps_to_h3.get("S2_CAP_PROFILE_TO_H3_RECEIPT", False)),
        "minimal_caps_to_h3_median_reconstruction_mse": caps_to_h3.get("median_reconstruction_mse"),
        "minimal_caps_to_h3_median_shuffled_profile_mse": caps_to_h3.get("median_shuffled_profile_mse"),
        "minimal_caps_to_h3_median_s2_boundary_profile_mse": caps_to_h3.get("median_s2_boundary_profile_mse"),
        "minimal_caps_to_h3_beats_shuffled": caps_to_h3.get("h3_beats_shuffled"),
        "minimal_caps_to_h3_beats_s2_boundary": caps_to_h3.get("h3_beats_s2_boundary"),
        "observer_readout_observer_count": observer_consensus.get("observer_count"),
        "observer_readout_pair_count": observer_consensus.get("pair_count"),
        "observer_readout_global_committed_fraction": observer_consensus.get("global_committed_fraction"),
        "observer_readout_median_overlap_jaccard": observer_consensus.get("median_overlap_jaccard"),
        "observer_readout_median_signature_similarity": observer_consensus.get("median_signature_histogram_similarity"),
        "observer_readout_p10_signature_similarity": observer_consensus.get("p10_signature_histogram_similarity"),
        "observer_object_consensus_object_count": object_consensus.get("object_count"),
        "observer_object_consensus_persistent_object_count": object_consensus.get("persistent_object_count"),
        "observer_object_consensus_median_overlap_agreement": object_consensus.get("median_overlap_agreement"),
        "observer_object_consensus_p10_overlap_agreement": object_consensus.get("p10_overlap_agreement"),
        "observer_object_consensus_median_counterfactual_stability": object_consensus.get(
            "median_counterfactual_stability"
        ),
        "observer_object_bad_record_rewrite_detected": object_consensus.get("bad_record_rewrite_detected"),
        "observer_chart_object_receipt": object_chart_flags.get("chart_receipt"),
        "observer_chart_report_source": object_chart.get("_report_path"),
        "observer_chart_postprocess_recomputed": object_chart.get("postprocess_recomputed"),
        "observer_chart_postprocess_incidence_mode": object_chart.get("postprocess_incidence_mode")
        or object_chart.get("incidence_mode"),
        "observer_chart_object_median_receipt": object_chart_flags.get("chart_median_receipt"),
        "observer_chart_modular_response_h3_control_separation_receipt": object_chart.get(
            "modular_response_h3_control_separation_receipt"
        ),
        "observer_chart_localized_object_precursor_receipt": object_chart_flags.get("localized_precursor_receipt"),
        "observer_chart_localized_object_median_precursor_receipt": object_chart_flags.get(
            "localized_median_precursor_receipt"
        ),
        "observer_chart_bulk_population_receipt": object_chart.get("observer_chart_bulk_population_receipt"),
        "observer_chart_localized_nonboundary_bulk_population_receipt": object_chart.get(
            "localized_nonboundary_bulk_population_receipt"
        ),
        "observer_chart_object_count": object_chart.get("object_count"),
        "observer_chart_localized_object_count": object_chart.get("localized_object_count"),
        "observer_chart_localized_not_boundary_object_count": object_chart.get("localized_not_boundary_object_count"),
        "observer_chart_shuffle_control_count": object_chart.get("shuffle_control_count"),
        "observer_chart_shuffled_localized_object_count": object_chart.get("shuffled_localized_object_count"),
        "observer_chart_shuffled_localized_object_p90": object_chart.get("shuffled_localized_object_p90"),
        "observer_chart_shuffled_localized_not_boundary_object_count": object_chart.get(
            "shuffled_localized_not_boundary_object_count"
        ),
        "observer_chart_shuffled_localized_not_boundary_object_p90": object_chart.get(
            "shuffled_localized_not_boundary_object_p90"
        ),
        "observer_chart_h3_compactness": object_chart.get("median_h3_compactness_normalized"),
        "observer_chart_s2_compactness": object_chart.get("median_s2_boundary_compactness_normalized"),
        "observer_chart_shuffled_h3_compactness": object_chart.get("median_shuffled_h3_compactness_normalized"),
        "observer_chart_p10_shuffled_h3_compactness": object_chart.get(
            "p10_shuffled_h3_compactness_normalized"
        ),
        "observer_chart_p90_shuffled_h3_compactness": object_chart.get(
            "p90_shuffled_h3_compactness_normalized"
        ),
        "observer_chart_h3_compactness_margin_vs_median_shuffle": object_chart.get(
            "h3_compactness_margin_vs_median_shuffle"
        ),
        "observer_chart_h3_compactness_margin_vs_p10_shuffle": object_chart.get(
            "h3_compactness_margin_vs_p10_shuffle"
        ),
        "observer_chart_h3_compactness_margin_vs_s2_boundary": object_chart.get(
            "h3_compactness_margin_vs_s2_boundary"
        ),
        "observer_chart_compactness_distribution_control_receipt": object_chart.get(
            "compactness_distribution_control_receipt"
        ),
        "observer_chart_compactness_distribution_population_receipt": object_chart.get(
            "compactness_distribution_population_receipt"
        ),
        "observer_chart_localized_count_saturation_warning": object_chart.get("localized_count_saturation_warning"),
        "observer_chart_localized_not_boundary_count_saturation_warning": object_chart.get(
            "localized_not_boundary_count_saturation_warning"
        ),
        "observer_chart_h3_beats_shuffled": object_chart_flags.get("h3_beats_shuffled"),
        "observer_chart_h3_beats_shuffled_robust": object_chart_flags.get("h3_beats_shuffled_robust"),
        "observer_chart_h3_not_boundary_dominated": object_chart.get("h3_not_boundary_dominated"),
        "neutral_reconstruction_written": bool(neutral),
        "neutral_radial_depth_used": neutral_dimension.get("radial_depth_used") if neutral_dimension else None,
        "neutral_control_gate_passed": neutral.get("control_gate_passed") if neutral else None,
        "neutral_candidate_3d_dimension_window": neutral.get("candidate_3d_dimension_window") if neutral else None,
        "neutral_bulk_3d_established": neutral.get("bulk_3d_established") if neutral else None,
        "neutral_dimension_estimators_agree": neutral.get("dimension_estimators_agree") if neutral else None,
        "neutral_primary_dimension_estimate": neutral_primary.get("estimate") if isinstance(neutral_primary, dict) else None,
        "neutral_correlation_dimension_estimate": neutral_corr.get("estimate") if isinstance(neutral_corr, dict) else None,
        "neutral_local_mle_dimension_estimate": neutral_mle.get("estimate") if isinstance(neutral_mle, dict) else None,
        "strict_neutral_report_written": bool(strict_neutral),
        "strict_neutral_bulk_receipt": bool(strict_neutral.get("strict_neutral_bulk", False)),
        "strict_neutral_observer_count": strict_neutral.get("observer_count") if strict_neutral else None,
        "strict_neutral_estimators_agree_3d": (
            strict_neutral_dimension.get("estimators_agree_3d")
            if isinstance(strict_neutral_dimension, dict)
            else None
        ),
        "strict_neutral_median_dimension_estimate": (
            strict_neutral_dimension.get("median_dimension_estimate")
            if isinstance(strict_neutral_dimension, dict)
            else None
        ),
        "strict_neutral_correlation_dimension_estimate": (
            strict_neutral_corr.get("estimate")
            if isinstance(strict_neutral_corr, dict)
            else None
        ),
        "strict_neutral_local_mle_dimension_estimate": (
            strict_neutral_mle.get("median")
            if isinstance(strict_neutral_mle, dict)
            else None
        ),
        "strict_neutral_selected_model": (
            strict_neutral_model.get("selected_model", strict_neutral_model.get("best_model"))
            if strict_neutral_model
            else None
        ),
        "strict_neutral_best_model": strict_neutral_model.get("best_model") if strict_neutral_model else None,
        "strict_neutral_h3_beats_s2": strict_neutral_model.get("h3_beats_s2") if strict_neutral_model else None,
        "strict_neutral_h3_beats_h2_h4": (
            strict_neutral_model.get("h3_beats_h2_h4") if strict_neutral_model else None
        ),
        "strict_neutral_s2_leakage_pass": (
            strict_neutral_leakage.get("s2_leakage_pass") if strict_neutral_leakage else None
        ),
        "strict_neutral_s2_distance_correlation": (
            strict_neutral_leakage.get("s2_distance_correlation") if strict_neutral_leakage else None
        ),
        "strict_neutral_control_pass_count": sum(
            1 for control in strict_neutral_control_rows if control.get("control_passed")
        ),
        "strict_neutral_control_count": len(strict_neutral_control_rows),
        "strict_neutral_blockers": strict_neutral.get("blockers", []) if strict_neutral else [],
        "strict_neutral_object_report_written": bool(strict_neutral_object),
        "strict_neutral_object_bulk_receipt": bool(
            strict_neutral_object.get("STRICT_NEUTRAL_OBJECT_BULK_RECEIPT", False)
            or strict_neutral_object.get("strict_neutral_object_bulk", False)
        ),
        "strict_neutral_object_count": strict_neutral_object.get("object_count") if strict_neutral_object else None,
        "strict_neutral_object_estimators_agree_3d": (
            strict_neutral_object_dimension.get("estimators_agree_3d")
            if isinstance(strict_neutral_object_dimension, dict)
            else None
        ),
        "strict_neutral_object_median_dimension_estimate": (
            strict_neutral_object_dimension.get("median_dimension_estimate")
            if isinstance(strict_neutral_object_dimension, dict)
            else None
        ),
        "strict_neutral_object_selected_model": (
            strict_neutral_object_model.get("selected_model") if strict_neutral_object_model else None
        ),
        "strict_neutral_object_h3_selected": (
            strict_neutral_object_model.get("h3_selected") if strict_neutral_object_model else None
        ),
        "strict_neutral_object_s2_leakage_pass": (
            strict_neutral_object_leakage.get("s2_leakage_pass") if strict_neutral_object_leakage else None
        ),
        "strict_neutral_object_s2_distance_correlation": (
            strict_neutral_object_leakage.get("s2_distance_correlation")
            if strict_neutral_object_leakage
            else None
        ),
        "strict_neutral_object_blockers": strict_neutral_object.get("blockers", []) if strict_neutral_object else [],
        "blind_bulk_usable": blind_bulk.get("usable") if blind_bulk else None,
        "blind_bulk_feature_width": blind_bulk.get("feature_width") if blind_bulk else None,
        "blind_bulk_s2_leakage_audit_pass": blind_bulk.get("s2_leakage_audit_pass") if blind_bulk else None,
        "blind_bulk_s2_distance_correlation": blind_bulk.get("s2_distance_correlation") if blind_bulk else None,
        "blind_bulk_candidate_3d_dimension_window": blind_bulk.get("candidate_3d_dimension_window") if blind_bulk else None,
        "blind_low_rank_usable": blind_low_rank.get("usable") if blind_low_rank else None,
        "blind_low_rank_participation_rank": blind_low_rank.get("participation_rank") if blind_low_rank else None,
        "blind_low_rank_entropy_rank": blind_low_rank.get("entropy_rank") if blind_low_rank else None,
        "blind_low_rank_selected_rank": blind_low_rank.get("selected_rank") if blind_low_rank else None,
        "blind_low_rank_selected_rank_3d_candidate_receipt": (
            blind_low_rank.get("selected_rank_3d_candidate_receipt") if blind_low_rank else None
        ),
        "blind_low_rank_selected_candidate_3d_window": (
            blind_low_rank_selected.get("candidate_3d_dimension_window")
            if isinstance(blind_low_rank_selected, dict)
            else None
        ),
        "blind_low_rank_selected_s2_leakage_audit_pass": (
            blind_low_rank_selected.get("s2_leakage_audit_pass")
            if isinstance(blind_low_rank_selected, dict)
            else None
        ),
        "blind_low_rank_selected_s2_distance_correlation": (
            blind_low_rank_selected.get("s2_distance_correlation")
            if isinstance(blind_low_rank_selected, dict)
            else None
        ),
        "blind_low_rank_selected_correlation_dimension_estimate": (
            blind_low_rank_selected_corr.get("estimate") if isinstance(blind_low_rank_selected_corr, dict) else None
        ),
        "blind_low_rank_selected_local_mle_dimension_estimate": (
            blind_low_rank_selected_mle.get("estimate") if isinstance(blind_low_rank_selected_mle, dict) else None
        ),
        "blind_record_transition_rank3_receipt": (
            blind_group_sweep.get("record_transition_rank3_receipt") if isinstance(blind_group_sweep, dict) else None
        ),
        "blind_record_transition_rank3_candidate_3d_window": (
            blind_record_rank3.get("candidate_3d_dimension_window")
            if isinstance(blind_record_rank3, dict)
            else None
        ),
        "blind_record_transition_rank3_dimension_estimators_agree": (
            blind_record_rank3.get("dimension_estimators_agree")
            if isinstance(blind_record_rank3, dict)
            else None
        ),
        "blind_record_transition_rank3_s2_leakage_audit_pass": (
            blind_record_rank3.get("s2_leakage_audit_pass") if isinstance(blind_record_rank3, dict) else None
        ),
        "blind_record_transition_rank3_s2_distance_correlation": (
            blind_record_rank3.get("s2_distance_correlation") if isinstance(blind_record_rank3, dict) else None
        ),
        "blind_record_transition_rank3_correlation_dimension_estimate": (
            blind_record_rank3_corr.get("estimate") if isinstance(blind_record_rank3_corr, dict) else None
        ),
        "blind_record_transition_rank3_local_mle_dimension_estimate": (
            blind_record_rank3_mle.get("estimate") if isinstance(blind_record_rank3_mle, dict) else None
        ),
        "blind_bulk_3d_established": blind_bulk.get("bulk_3d_established") if blind_bulk else None,
        "blind_bulk_correlation_dimension_estimate": (
            blind_corr.get("estimate") if isinstance(blind_corr, dict) else None
        ),
        "blind_bulk_local_mle_dimension_estimate": blind_mle.get("estimate") if isinstance(blind_mle, dict) else None,
        "cmb_benchmark": (cmb.get("benchmark", {}) if cmb else {}).get("label"),
        "cmb_best_field": cmb_best_name,
        "cmb_best_shape_correlation": cmb_best.get("shape_correlation"),
        "cmb_best_normalized_rmse": cmb_best.get("normalized_rmse"),
        "cmb_best_peak_fraction_delta": cmb_best.get("peak_fraction_delta"),
        "cmb_best_positive_field": cmb_best_positive_name,
        "cmb_best_positive_shape_correlation": cmb_best_positive.get("shape_correlation"),
        "cmb_best_positive_normalized_rmse": cmb_best_positive.get("normalized_rmse"),
        "cmb_best_positive_peak_fraction_delta": cmb_best_positive.get("peak_fraction_delta"),
        "cmb_best_overlap_ell_usable": cmb_best_overlap.get("usable"),
        "cmb_best_overlap_ell_shape_correlation": cmb_best_overlap.get("shape_correlation"),
        "cmb_best_overlap_ell_normalized_rmse": cmb_best_overlap.get("positive_amp_normalized_rmse"),
        "cmb_best_overlap_ell_min": cmb_best_overlap.get("overlap_ell_min"),
        "cmb_best_overlap_ell_max": cmb_best_overlap.get("overlap_ell_max"),
        "cmb_best_overlap_benchmark_count": cmb_best_overlap.get("overlap_benchmark_count"),
        "cmb_best_real_ell_overlap_field": cmb_best_overlap_name,
        "cmb_best_real_ell_overlap_shape_correlation": cmb_best_overlap_real.get("shape_correlation"),
        "cmb_best_real_ell_overlap_normalized_rmse": cmb_best_overlap_real.get("positive_amp_normalized_rmse"),
        "cmb_best_real_ell_overlap_min": cmb_best_overlap_real.get("overlap_ell_min"),
        "cmb_best_real_ell_overlap_max": cmb_best_overlap_real.get("overlap_ell_max"),
        "cmb_best_real_ell_overlap_benchmark_count": cmb_best_overlap_real.get("overlap_benchmark_count"),
        "record_signature_shape_correlation": record_cmb.get("shape_correlation"),
        "record_signature_normalized_rmse": record_cmb.get("normalized_rmse"),
        "record_signature_peak_fraction_delta": record_cmb.get("peak_fraction_delta"),
        "record_signature_overlap_ell_usable": record_cmb_overlap.get("usable"),
        "record_signature_overlap_ell_shape_correlation": record_cmb_overlap.get("shape_correlation"),
        "record_signature_overlap_ell_normalized_rmse": record_cmb_overlap.get("positive_amp_normalized_rmse"),
        "record_signature_overlap_ell_min": record_cmb_overlap.get("overlap_ell_min"),
        "record_signature_overlap_ell_max": record_cmb_overlap.get("overlap_ell_max"),
        "record_signature_overlap_benchmark_count": record_cmb_overlap.get("overlap_benchmark_count"),
        "record_signature_peak_ell": cl_record.get("peak_ell"),
        "record_signature_total_abs_D_ell": cl_record.get("total_abs_D_ell_2_plus"),
        "record_signature_control_min_l2_delta": cl_record_control.get("min_relative_l2_delta"),
        "freezeout_cycle": cl.get("freezeout_cycle") if cl else None,
        "committed_fraction": cl.get("committed_fraction") if cl else None,
        "holonomy_triangle_count": hol.get("triangle_count") if hol else None,
        "holonomy_defect_fraction": hol.get("defect_fraction") if hol else None,
        "holonomy_cluster_count": hol.get("cluster_count") if hol else None,
        "defect_timeline_written": bool(timeline),
        "defect_timeline_snapshot_count": timeline.get("snapshot_count") if timeline else None,
        "defect_timeline_worldline_count": timeline.get("worldline_count") if timeline else None,
        "defect_timeline_persistent_worldline_count": timeline.get("persistent_worldline_count") if timeline else None,
        "defect_timeline_max_observation_count": timeline.get("max_observation_count") if timeline else None,
        "defect_timeline_max_lifetime_cycles": timeline.get("max_lifetime_cycles") if timeline else None,
        "defect_worldline_precursor_receipt": timeline.get("persistent_worldline_precursor_receipt")
        if timeline
        else None,
        "defect_interaction_written": bool(interaction),
        "defect_screen_transport_proxy_count": interaction.get("screen_transport_proxy_count")
        if interaction
        else None,
        "defect_fusion_candidate_count": interaction.get("fusion_candidate_count") if interaction else None,
        "defect_fusion_conservation_proxy_pass": interaction.get("fusion_conservation_proxy_pass")
        if interaction
        else None,
        "defect_scattering_reproducibility_proxy_pass": interaction.get("scattering_reproducibility_proxy_pass")
        if interaction
        else None,
        "defect_interaction_proxy_receipt": interaction.get("interaction_proxy_receipt") if interaction else None,
        "particle_likeness_written": bool(particle),
        "particle_likeness_worldline_count": particle.get("worldline_count") if particle else None,
        "particle_likeness_localized_count": particle.get("localized_count") if particle else None,
        "particle_likeness_persistent_count": particle.get("persistent_count") if particle else None,
        "particle_likeness_sector_stable_count": particle.get("sector_stable_count") if particle else None,
        "particle_likeness_transportable_count": particle.get("transportable_count") if particle else None,
        "particle_like_count": particle.get("particle_like_count") if particle else None,
        "particle_matter_receipt": particle.get("particle_matter_receipt") if particle else None,
        "controlled_defect_assay_written": bool(controlled_particle),
        "controlled_defect_assay_inverse_identity_pass": controlled_particle.get("s3_inverse_identity_pass")
        if controlled_particle
        else None,
        "controlled_defect_assay_interaction_receipt": controlled_particle.get("interaction_proxy_receipt")
        if controlled_particle
        else None,
        "controlled_defect_assay_fusion_pass": controlled_particle.get("fusion_conservation_proxy_pass")
        if controlled_particle
        else None,
        "controlled_defect_assay_detector_positive": controlled_particle.get("particle_detector_positive_receipt")
        if controlled_particle
        else None,
        "controlled_defect_assay_particle_like_count": controlled_particle.get("particle_like_count")
        if controlled_particle
        else None,
        "controlled_defect_assay_physical_particle_emergence": controlled_particle.get(
            "physical_particle_emergence"
        )
        if controlled_particle
        else None,
        "shape_substrate_report_written": bool(shape_summary),
        "shape_vertex_scattering_receipt": bool(
            shape_summary.get("shape_vertex_scattering_receipt") or shape_vertex.get("passed")
        ),
        "shape_dodeca_cell_receipt": bool(shape_summary.get("shape_dodeca_cell_receipt") or shape_cell.get("passed")),
        "shape_settling_receipt": bool(shape_summary.get("shape_settling_receipt") or shape_settling.get("passed")),
        "shape_loop_particle_receipt": bool(
            shape_summary.get("shape_loop_particle_receipt") or shape_particle.get("passed")
        ),
        "shape_screen_projection_receipt": bool(
            shape_summary.get("shape_screen_projection_receipt") or shape_projection.get("passed")
        ),
        "shape_selector_elimination_target_input_receipt": bool(
            shape_summary.get("shape_selector_elimination_target_input_receipt")
            or shape_certificate.get("selector_elimination_target_input_receipt")
        ),
        "shape_cmb_certificate_input_receipt": bool(
            shape_summary.get("shape_cmb_certificate_input_receipt") or shape_certificate.get("passed")
        ),
        "shape_phi_drop_fraction": (
            shape_summary.get("shape_phi_drop_fraction")
            if shape_summary
            else shape_settling.get("phi_drop_fraction")
        ),
        "shape_loop_particle_count": (
            shape_summary.get("shape_loop_particle_count")
            if shape_summary
            else shape_particle.get("persistent_loop_particle_count")
        ),
        "shape_q_IR_candidate": (
            shape_summary.get("shape_q_IR_candidate") if shape_summary else shape_certificate.get("q_IR_candidate")
        ),
        "shape_q_IR_runtime_zero_mode": (
            shape_summary.get("shape_q_IR_runtime_zero_mode")
            if shape_summary
            else shape_certificate.get("q_IR_runtime_zero_mode")
        ),
        "shape_ell_IR_candidate": (
            shape_summary.get("shape_ell_IR_candidate") if shape_summary else shape_certificate.get("ell_IR_candidate")
        ),
        "shape_ell_IR_runtime_covariance_rank": (
            shape_summary.get("shape_ell_IR_runtime_covariance_rank")
            if shape_summary
            else _nested(shape_certificate, "ell_IR_runtime_covariance", "visible_covariance_rank")
        ),
        "shape_eta_R_candidate": (
            shape_summary.get("shape_eta_R_candidate") if shape_summary else shape_certificate.get("eta_R_candidate")
        ),
        "shape_planck_lite_shape_correlation": shape_summary.get("shape_planck_lite_shape_correlation"),
        "shape_planck_lite_normalized_rmse": shape_summary.get("shape_planck_lite_normalized_rmse"),
        "shape_physical_cmb_prediction": bool(
            shape_summary.get("physical_cmb_prediction", False)
            or shape_projection.get("physical_cmb_prediction", False)
            or shape_certificate.get("physical_cmb_prediction", False)
        ),
        "shape_neutral_oph_bulk_claim": bool(
            shape_summary.get("neutral_oph_bulk_claim", False)
            or shape_projection.get("neutral_oph_bulk_claim", False)
        ),
        "shape_declared_3d_substrate": bool(
            shape_summary.get("declared_3d_substrate", False)
            or shape_projection.get("declared_3d_substrate", False)
        ),
        "defect_h3_worldline_written": bool(defect_h3_worldlines),
        "defect_h3_worldline_count": defect_h3_worldlines.get("worldline_count") if defect_h3_worldlines else None,
        "defect_h3_persistent_worldline_count": defect_h3_worldlines.get("persistent_h3_worldline_count")
        if defect_h3_worldlines
        else None,
        "defect_h3_bulk_worldline_precursor_receipt": defect_h3_worldlines.get("bulk_worldline_precursor_receipt")
        if defect_h3_worldlines
        else None,
    }
    return row


def _lorentz_branch_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [
        row
        for row in rows
        if any(
            row.get(name)
            for name in (
                "chart_level_conformal_lorentz_receipt",
                "bw_automorphism_sanity_receipt",
                "support_visible_lorentz_3p1_kinematics_receipt",
                "endogenous_modular_generator_receipt",
                "paper_theorem_3d_bulk_chart_receipt",
                "support_visible_h3_populated_bulk_receipt",
                "theorem_assisted_h3_object_preview_receipt",
                "object_h3_nonboundary_population_receipt",
                "paper_theorem_assisted_h3_chart_precursor_receipt",
                "paper_theorem_assisted_h3_populated_chart_receipt",
                "object_bulk_population_receipt",
                "bulk_proof_certificate_written",
            )
        )
    ]
    return {
        "run_count": len(usable),
        "chart_level_conformal_lorentz_count": sum(
            1 for row in usable if row.get("chart_level_conformal_lorentz_receipt")
        ),
        "bw_automorphism_sanity_count": sum(1 for row in usable if row.get("bw_automorphism_sanity_receipt")),
        "support_visible_lorentz_3p1_count": sum(
            1 for row in usable if row.get("support_visible_lorentz_3p1_kinematics_receipt")
        ),
        "endogenous_modular_generator_count": sum(
            1 for row in usable if row.get("endogenous_modular_generator_receipt")
        ),
        "paper_theorem_3d_bulk_chart_count": sum(
            1 for row in usable if row.get("paper_theorem_3d_bulk_chart_receipt")
        ),
        "paper_theorem_object_populated_chart_precursor_count": sum(
            1 for row in usable if row.get("paper_theorem_object_populated_chart_precursor_receipt")
        ),
        "paper_theorem_neutral_populated_bulk_count": sum(
            1 for row in usable if row.get("paper_theorem_neutral_populated_bulk_receipt")
        ),
        "support_visible_record_h3_population_count": sum(
            1 for row in usable if row.get("support_visible_record_h3_population_receipt")
        ),
        "support_visible_defect_h3_population_count": sum(
            1 for row in usable if row.get("support_visible_defect_h3_population_receipt")
        ),
        "support_visible_h3_populated_bulk_count": sum(
            1 for row in usable if row.get("support_visible_h3_populated_bulk_receipt")
        ),
        "mean_record_family_h3_median_residual": _mean(
            row.get("record_family_h3_median_residual") for row in usable
        ),
        "mean_defect_cluster_h3_median_residual": _mean(
            row.get("defect_cluster_h3_median_residual") for row in usable
        ),
        "mean_paper_theorem_h3_spatial_dimension": _mean(
            row.get("paper_theorem_h3_spatial_dimension") for row in usable
        ),
        "paper_theorem_assisted_h3_chart_precursor_count": sum(
            1 for row in usable if row.get("paper_theorem_assisted_h3_chart_precursor_receipt")
        ),
        "paper_theorem_assisted_h3_populated_chart_count": sum(
            1 for row in usable if row.get("paper_theorem_assisted_h3_populated_chart_receipt")
        ),
        "theorem_assisted_h3_object_preview_count": sum(
            1 for row in usable if row.get("theorem_assisted_h3_object_preview_receipt")
        ),
        "object_h3_nonboundary_population_count": sum(
            1 for row in usable if row.get("object_h3_nonboundary_population_receipt")
        ),
        "object_bulk_population_count": sum(1 for row in usable if row.get("object_bulk_population_receipt")),
        "bulk_proof_certificate_count": sum(1 for row in usable if row.get("bulk_proof_certificate_written")),
        "bulk_proof_chart_level_3p1_count": sum(1 for row in usable if row.get("bulk_proof_chart_level_3p1")),
        "bulk_proof_theorem_assisted_h3_object_preview_count": sum(
            1 for row in usable if row.get("bulk_proof_theorem_assisted_h3_object_preview")
        ),
        "bulk_proof_theorem_assisted_h3_nonboundary_population_count": sum(
            1 for row in usable if row.get("bulk_proof_theorem_assisted_h3_nonboundary_population")
        ),
        "bulk_proof_theorem_assisted_h3_populated_chart_count": sum(
            1 for row in usable if row.get("bulk_proof_theorem_assisted_h3_populated_chart")
        ),
        "bulk_proof_strict_neutral_3d_bulk_count": sum(
            1 for row in usable if row.get("bulk_proof_strict_neutral_3d_bulk")
        ),
        "bulk_proof_screen_cmb_proxy_count": sum(1 for row in usable if row.get("bulk_proof_screen_cmb_proxy")),
        "bulk_proof_physical_cmb_prediction_count": sum(
            1 for row in usable if row.get("bulk_proof_physical_cmb_prediction")
        ),
        "bulk_proof_production_particle_matter_count": sum(
            1 for row in usable if row.get("bulk_proof_production_particle_matter")
        ),
        "screen_proxy_cmb_count": sum(1 for row in usable if row.get("screen_proxy_cmb_receipt")),
        "bulk_3d_established_count": sum(1 for row in usable if row.get("bulk_3d_established")),
        "interpretation": (
            "Splits the paper-side chart-level Lorentz receipt from later strengthening receipts. "
            "A support-visible Lorentz count means conformal H3 chart plus direct BW/KMS automorphism sanity; "
            "a paper-theorem 3D bulk chart count means the simulator has verified the exact chart route "
            "Conf+(S2) -> SO+(3,1) -> H3 with boost-orbit dimension 3, rather than fitting a finite "
            "point cloud dimension. A support-visible populated-H3 count means persistent record-family "
            "support profiles populate that H3 chart under S2-boundary and shuffled-cap controls. "
            "Defect-cluster support in H3 is reported separately as a matter/particle precursor. "
            "a theorem-assisted object-preview count means the Lorentz chart, BW automorphism sanity, "
            "intermediate H3 response control separation, and localized observer-object H3 preview are all present. "
            "A nonboundary population count is stricter and requires the boundary leakage audit to pass. Both remain separate "
            "from the strict finite endogenous-generator bulk proof and from particles/physical cosmology. "
            "The bulk-proof certificate counts make this split explicit: chart-level 3+1D, theorem-assisted "
            "H3 preview, nonboundary H3 population, strict neutral bulk, screen-CMB proxy, physical CMB, and production particles are "
            "separate gates."
        ),
    }


def _planck_lite_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("cmb_best_normalized_rmse") is not None]
    overlap_usable = [row for row in usable if row.get("record_signature_overlap_ell_usable")]
    best_overlap_usable = [row for row in usable if row.get("cmb_best_real_ell_overlap_field")]
    return {
        "run_count": len(usable),
        "best_field_counts": _counts(row.get("cmb_best_field") for row in usable),
        "best_positive_field_counts": _counts(row.get("cmb_best_positive_field") for row in usable),
        "best_real_ell_overlap_field_counts": _counts(
            row.get("cmb_best_real_ell_overlap_field") for row in best_overlap_usable
        ),
        "mean_best_shape_correlation": _mean(row.get("cmb_best_shape_correlation") for row in usable),
        "mean_best_normalized_rmse": _mean(row.get("cmb_best_normalized_rmse") for row in usable),
        "mean_best_positive_shape_correlation": _mean(
            row.get("cmb_best_positive_shape_correlation") for row in usable
        ),
        "mean_best_positive_normalized_rmse": _mean(
            row.get("cmb_best_positive_normalized_rmse") for row in usable
        ),
        "mean_record_signature_shape_correlation": _mean(
            row.get("record_signature_shape_correlation") for row in usable
        ),
        "mean_record_signature_normalized_rmse": _mean(row.get("record_signature_normalized_rmse") for row in usable),
        "overlap_ell_usable_count": len(overlap_usable),
        "mean_record_signature_overlap_ell_shape_correlation": _mean(
            row.get("record_signature_overlap_ell_shape_correlation") for row in overlap_usable
        ),
        "mean_record_signature_overlap_ell_normalized_rmse": _mean(
            row.get("record_signature_overlap_ell_normalized_rmse") for row in overlap_usable
        ),
        "mean_record_signature_overlap_ell_min": _mean(
            row.get("record_signature_overlap_ell_min") for row in overlap_usable
        ),
        "mean_record_signature_overlap_ell_max": _mean(
            row.get("record_signature_overlap_ell_max") for row in overlap_usable
        ),
        "mean_record_signature_overlap_benchmark_count": _mean(
            row.get("record_signature_overlap_benchmark_count") for row in overlap_usable
        ),
        "best_real_ell_overlap_usable_count": len(best_overlap_usable),
        "mean_best_real_ell_overlap_shape_correlation": _mean(
            row.get("cmb_best_real_ell_overlap_shape_correlation") for row in best_overlap_usable
        ),
        "mean_best_real_ell_overlap_normalized_rmse": _mean(
            row.get("cmb_best_real_ell_overlap_normalized_rmse") for row in best_overlap_usable
        ),
        "mean_best_real_ell_overlap_min": _mean(
            row.get("cmb_best_real_ell_overlap_min") for row in best_overlap_usable
        ),
        "mean_best_real_ell_overlap_max": _mean(
            row.get("cmb_best_real_ell_overlap_max") for row in best_overlap_usable
        ),
        "mean_best_real_ell_overlap_benchmark_count": _mean(
            row.get("cmb_best_real_ell_overlap_benchmark_count") for row in best_overlap_usable
        ),
        "physical_cmb_prediction": False,
        "interpretation": (
            "Currently useful as a negative/diagnostic Planck-lite screen comparison. The normalized-axis "
            "diagnostic is not physical ell-space; the overlap-ell diagnostic scores only measured Planck "
            "bins covered by the finite screen. Neither is a likelihood or physical CMB prediction."
        ),
    }


def _state_bw_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("state_bw_written")]
    declared_cap_flow = [row for row in usable if row.get("state_bw_declared_cap_flow_generator")]
    declared_response_density = [row for row in usable if row.get("state_bw_declared_response_density")]
    direct_transition = [row for row in usable if row.get("state_bw_direct_transition_automorphism")]
    endogenous = [
        row
        for row in usable
        if (
            row.get("state_bw_endogenous_generator")
            and not row.get("state_bw_declared_cap_flow_generator")
            and not row.get("state_bw_declared_response_density")
        )
    ]
    if not usable:
        interpretation = "No state-derived BW matrix-element reports were found in the selected run set."
    elif endogenous and not any(row.get("state_bw_correct_beats_controls") for row in endogenous):
        interpretation = (
            "Strict endogenous cap/collar matrix-element probes are present, but they do not yet beat "
            "wrong-normalization/no-flow controls. This blocks the endogenous modular-generator strengthening "
            "receipt, not the chart-level Lorentz receipt reported in the separate Lorentz branch lane."
        )
    elif direct_transition and not endogenous:
        interpretation = (
            "Only direct transition-automorphism BW receipts pass in this run set. Those are useful finite "
            "support-visible sanity receipts for the declared 2pi transport. They are counted with the "
            "conformal H3 chart in the separate Lorentz branch lane; they are not endogenous observer-record "
            "modular generators and do not establish a populated bulk."
        )
    elif declared_response_density and not endogenous:
        interpretation = (
            "Declared transition-response density-log calibration receipts are present. This verifies that "
            "finite rho -> -log(rho) transport can recover a declared 2pi cap transition, but it is still "
            "not an endogenous observer-record modular generator and does not establish a populated bulk."
        )
    elif any(row.get("state_bw_receipt") for row in usable):
        interpretation = (
            "At least one state-derived BW matrix-element receipt passed. This is still a finite-regulator "
            "diagnostic and needs refinement scaling before any Lorentz/bulk claim."
        )
    else:
        interpretation = (
            "State-derived BW reports are present but do not pass the full receipt. Treat them as diagnostic "
            "until correct 2pi transport beats all implemented controls."
        )
    return {
        "run_count": len(usable),
        "endogenous_run_count": len(endogenous),
        "declared_cap_flow_run_count": len(declared_cap_flow),
        "declared_response_density_run_count": len(declared_response_density),
        "direct_transition_automorphism_run_count": len(direct_transition),
        "receipt_count": sum(1 for row in usable if row.get("state_bw_receipt")),
        "density_log_calibration_receipt_count": sum(
            1 for row in usable if row.get("state_bw_density_log_calibration_receipt")
        ),
        "endogenous_correct_beats_controls_count": sum(
            1 for row in endogenous if row.get("state_bw_correct_beats_controls")
        ),
        "selected_2pi_count": sum(1 for row in usable if row.get("state_bw_selected_2pi")),
        "mean_median_residual": _mean(row.get("state_bw_median") for row in usable),
        "mean_endogenous_median_residual": _mean(row.get("state_bw_median") for row in endogenous),
        "mean_declared_cap_flow_median_residual": _mean(row.get("state_bw_median") for row in declared_cap_flow),
        "mean_declared_response_density_median_residual": _mean(
            row.get("state_bw_median") for row in declared_response_density
        ),
        "mean_direct_transition_median_residual": _mean(row.get("state_bw_median") for row in direct_transition),
        "mean_wrong_1x_median": _mean(row.get("state_bw_wrong_1x_median") for row in usable),
        "mean_wrong_pi_median": _mean(row.get("state_bw_wrong_pi_median") for row in usable),
        "mean_no_modular_flow_median": _mean(row.get("state_bw_no_modular_flow_median") for row in usable),
        "best_control_counts": _counts(row.get("state_bw_best_control") for row in usable if row.get("state_bw_best_control")),
        "state_mode_counts": _counts(row.get("state_bw_state_mode") for row in usable if row.get("state_bw_state_mode")),
        "generator_scale_counts": _counts(
            f"{float(row.get('state_bw_generator_scale')):.6g}"
            for row in usable
            if isinstance(row.get("state_bw_generator_scale"), (int, float))
        ),
        "target_scale_control_degenerate_count": sum(
            1 for row in usable if row.get("state_bw_target_scale_control_degenerate")
        ),
        "degenerate_target_scale_control_counts": _counts(
            control
            for row in usable
            for control in (row.get("state_bw_degenerate_target_scale_controls") or [])
        ),
        "generator_scale_audit_count": sum(1 for row in usable if row.get("state_bw_generator_audit_enabled")),
        "generator_scale_audit_configured_best_count": sum(
            1 for row in usable if row.get("state_bw_generator_audit_configured_is_best")
        ),
        "generator_scale_audit_two_pi_best_count": sum(
            1 for row in usable if row.get("state_bw_generator_audit_two_pi_is_best")
        ),
        "generator_scale_audit_best_label_counts": _counts(
            row.get("state_bw_generator_audit_best_label")
            for row in usable
            if row.get("state_bw_generator_audit_enabled") and row.get("state_bw_generator_audit_best_label")
        ),
        "generator_scale_audit_diagnosis_counts": _counts(
            row.get("state_bw_generator_audit_diagnosis")
            for row in usable
            if row.get("state_bw_generator_audit_enabled") and row.get("state_bw_generator_audit_diagnosis")
        ),
        "mean_generator_scale_audit_best_score": _mean(
            row.get("state_bw_generator_audit_best_score")
            for row in usable
            if row.get("state_bw_generator_audit_enabled")
        ),
        "mean_generator_scale_audit_configured_score": _mean(
            row.get("state_bw_generator_audit_configured_score")
            for row in usable
            if row.get("state_bw_generator_audit_enabled")
        ),
        "interpretation": interpretation,
    }


def _object_chart_robust_flags(object_chart: dict[str, Any]) -> dict[str, Any]:
    if not object_chart:
        return {
            "chart_receipt": None,
            "chart_median_receipt": None,
            "localized_precursor_receipt": None,
            "localized_median_precursor_receipt": None,
            "h3_beats_shuffled": None,
            "h3_beats_shuffled_robust": None,
        }
    pass_ratio = _float_or(object_chart.get("pass_ratio"), 1.0)
    object_count = int(object_chart.get("object_count") or 0)
    min_objects = int(object_chart.get("min_objects") or 0)
    median_h3 = _float_or_none(object_chart.get("median_h3_compactness_normalized"))
    median_shuffle = _float_or_none(object_chart.get("median_shuffled_h3_compactness_normalized"))
    p10_shuffle = _float_or_none(object_chart.get("p10_shuffled_h3_compactness_normalized"))
    h3_beats = object_chart.get("h3_beats_shuffled_incidence")
    if h3_beats is None and median_h3 is not None and median_shuffle is not None:
        h3_beats = bool(median_h3 < pass_ratio * median_shuffle)
    h3_beats_robust = object_chart.get("h3_beats_shuffled_incidence_robust")
    if h3_beats_robust is None:
        if median_h3 is not None and p10_shuffle is not None:
            h3_beats_robust = bool(median_h3 < pass_ratio * p10_shuffle)
        else:
            h3_beats_robust = h3_beats
    eligible = bool(object_count >= min_objects)
    chart_median = object_chart.get("observer_chart_object_h3_median_receipt")
    if chart_median is None:
        chart_median = object_chart.get("observer_chart_object_h3_receipt")
    chart_receipt = object_chart.get("observer_chart_object_h3_receipt")
    if object_chart.get("observer_chart_object_h3_median_receipt") is None and p10_shuffle is not None:
        chart_receipt = bool(eligible and h3_beats_robust)

    localized_count = int(object_chart.get("localized_not_boundary_object_count") or 0)
    min_localized = int(object_chart.get("min_localized_objects") or 1)
    shuffled_localized = _float_or_none(object_chart.get("shuffled_localized_object_count"))
    shuffled_localized_p90 = _float_or_none(object_chart.get("shuffled_localized_object_p90"))
    median_precursor = object_chart.get("localized_object_median_precursor_receipt")
    if median_precursor is None:
        median_precursor = object_chart.get("localized_object_precursor_receipt")
    localized_precursor = object_chart.get("localized_object_precursor_receipt")
    if object_chart.get("localized_object_median_precursor_receipt") is None and shuffled_localized_p90 is not None:
        localized_precursor = bool(localized_count >= min_localized and localized_count > shuffled_localized_p90)
    elif localized_precursor is None and shuffled_localized is not None:
        localized_precursor = bool(localized_count >= min_localized and localized_count > shuffled_localized)
    return {
        "chart_receipt": chart_receipt,
        "chart_median_receipt": chart_median,
        "localized_precursor_receipt": localized_precursor,
        "localized_median_precursor_receipt": median_precursor,
        "h3_beats_shuffled": h3_beats,
        "h3_beats_shuffled_robust": h3_beats_robust,
    }


def _cmb_transfer_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("cmb_transfer_test_shape_correlation") is not None]
    return {
        "run_count": len(usable),
        "diagnostic_receipt_count": sum(1 for row in usable if row.get("cmb_transfer_diagnostic_receipt")),
        "mean_train_shape_correlation": _mean(row.get("cmb_transfer_train_shape_correlation") for row in usable),
        "mean_train_normalized_rmse": _mean(row.get("cmb_transfer_train_normalized_rmse") for row in usable),
        "mean_test_shape_correlation": _mean(row.get("cmb_transfer_test_shape_correlation") for row in usable),
        "mean_test_normalized_rmse": _mean(row.get("cmb_transfer_test_normalized_rmse") for row in usable),
        "mean_max_control_test_shape_correlation": _mean(
            row.get("cmb_transfer_max_control_test_shape_correlation") for row in usable
        ),
        "mean_test_control_gap": _mean(row.get("cmb_transfer_test_control_gap") for row in usable),
        "mean_bootstrap_corr_p05": _mean(row.get("cmb_transfer_bootstrap_corr_p05") for row in usable),
        "mean_bootstrap_corr_p95": _mean(row.get("cmb_transfer_bootstrap_corr_p95") for row in usable),
        "mean_bootstrap_rmse_p05": _mean(row.get("cmb_transfer_bootstrap_rmse_p05") for row in usable),
        "mean_bootstrap_rmse_p95": _mean(row.get("cmb_transfer_bootstrap_rmse_p95") for row in usable),
        "physical_cmb_prediction": False,
        "interpretation": (
            "Cross-scale screen-basis transfer is the strongest current Planck-facing diagnostic. "
            "It fits weights using observed Planck TT shape and checks transfer across patch counts against "
            "target/field-shuffle controls, so it is useful evidence about screen-statistic structure but not "
            "a physical CMB prediction."
        ),
    }


def _camb_lcdm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("camb_lcdm_written")]
    return {
        "run_count": len(usable),
        "cdm_limit_boltzmann_receipt_count": sum(
            1 for row in usable if row.get("camb_lcdm_boltzmann_receipt")
        ),
        "oph_anomaly_module_ready_count": sum(
            1 for row in usable if row.get("camb_lcdm_oph_anomaly_module_ready")
        ),
        "mean_shape_correlation": _mean(row.get("camb_lcdm_shape_correlation") for row in usable),
        "mean_normalized_rmse": _mean(row.get("camb_lcdm_normalized_rmse") for row in usable),
        "mean_amplitude_fit_chi2_per_bin": _mean(
            row.get("camb_lcdm_amplitude_fit_chi2_per_bin") for row in usable
        ),
        "mean_best_fit_column_chi2_per_bin": _mean(
            row.get("camb_lcdm_best_fit_column_chi2_per_bin") for row in usable
        ),
        "mean_first_peak_ell": _mean(row.get("camb_lcdm_first_peak_ell") for row in usable),
        "mean_benchmark_first_peak_ell": _mean(
            row.get("camb_lcdm_benchmark_first_peak_ell") for row in usable
        ),
        "mean_abs_fractional_error": _mean(row.get("camb_lcdm_mean_abs_fractional_error") for row in usable),
        "physical_cmb_prediction": False,
        "interpretation": (
            "External LambdaCDM CAMB baseline for CDM-limit/Boltzmann plumbing. This lane should match "
            "the local Planck TT benchmark reasonably well and gives us a regression target for any future "
            "OPH anomaly module. It is not an OPH prediction because all cosmological parameters are "
            "external defaults and no repair/anomaly source term is injected."
        ),
    }


def _oph_screen_power_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("oph_screen_power_written")]
    return {
        "run_count": len(usable),
        "simulator_primordial_ready_count": sum(
            1 for row in usable if row.get("oph_screen_power_simulator_primordial_ready")
        ),
        "physical_cmb_prediction_count": sum(
            1 for row in usable if row.get("oph_screen_power_physical_cmb_prediction")
        ),
        "mean_source_run_count": _mean(row.get("oph_screen_power_source_run_count") for row in usable),
        "mean_fit_row_count": _mean(row.get("oph_screen_power_fit_row_count") for row in usable),
        "mean_available_fit_count": _mean(row.get("oph_screen_power_available_fit_count") for row in usable),
        "best_planck_eta_field_counts": _counts(
            row.get("oph_screen_power_best_planck_eta_field")
            for row in usable
            if row.get("oph_screen_power_best_planck_eta_field")
        ),
        "reference_source_counts": _counts(
            row.get("oph_screen_power_reference_source")
            for row in usable
            if row.get("oph_screen_power_reference_source")
        ),
        "mean_reference_eta_R": _mean(row.get("oph_screen_power_reference_eta_R") for row in usable),
        "mean_reference_n_s": _mean(row.get("oph_screen_power_reference_n_s") for row in usable),
        "mean_record_signature_eta_R": _mean(row.get("oph_screen_power_record_eta_R") for row in usable),
        "mean_stable_count_eta_R": _mean(row.get("oph_screen_power_stable_eta_R") for row in usable),
        "physical_cmb_prediction": False,
        "interpretation": (
            "OPH screen-power effective-theory bridge. This is where finite screen C_l fields are converted "
            "into eta_R/n_s and a primordial-table scaffold. If simulator_primordial_ready_count is zero, "
            "the exported reference table is either a phenomenological Planck-target fallback or a deliberately "
            "labeled failed finite diagnostic, not a simulator-ready primordial prediction."
        ),
    }


def _oph_screen_camb_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("oph_screen_camb_written")]
    return {
        "run_count": len(usable),
        "screen_camb_transfer_receipt_count": sum(1 for row in usable if row.get("oph_screen_camb_receipt")),
        "simulator_eta_ready_count": sum(1 for row in usable if row.get("oph_screen_camb_simulator_eta_ready")),
        "physical_cmb_prediction_count": sum(
            1 for row in usable if row.get("oph_screen_camb_physical_cmb_prediction")
        ),
        "reference_source_counts": _counts(
            row.get("oph_screen_camb_reference_source")
            for row in usable
            if row.get("oph_screen_camb_reference_source")
        ),
        "mean_eta_R": _mean(row.get("oph_screen_camb_eta_R") for row in usable),
        "mean_n_s": _mean(row.get("oph_screen_camb_n_s") for row in usable),
        "mean_shape_correlation": _mean(row.get("oph_screen_camb_shape_correlation") for row in usable),
        "mean_normalized_rmse": _mean(row.get("oph_screen_camb_normalized_rmse") for row in usable),
        "mean_amplitude_fit_chi2_per_bin": _mean(
            row.get("oph_screen_camb_amplitude_fit_chi2_per_bin") for row in usable
        ),
        "mean_first_peak_ell": _mean(row.get("oph_screen_camb_first_peak_ell") for row in usable),
        "mean_benchmark_first_peak_ell": _mean(
            row.get("oph_screen_camb_benchmark_first_peak_ell") for row in usable
        ),
        "mean_abs_fractional_error": _mean(
            row.get("oph_screen_camb_mean_abs_fractional_error") for row in usable
        ),
        "physical_cmb_prediction": False,
        "interpretation": (
            "CAMB TT transfer from the OPH screen-power scalar scaffold. A transfer receipt means CAMB "
            "plumbing and Planck TT comparison are working. It is a physical OPH prediction only when the "
            "input screen report is simulator-derived, passes the simulator_eta_ready/readiness gates, and "
            "the downstream finite-certificate gates pass."
        ),
    }


def _maxent_green_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("oph_maxent_green_written")]
    return {
        "run_count": len(usable),
        "source_receipt_count": sum(1 for row in usable if row.get("oph_maxent_green_receipt")),
        "flat_green_receipt_count": sum(1 for row in usable if row.get("oph_maxent_green_flat_green_receipt")),
        "ir_theorem_receipt_count": sum(1 for row in usable if row.get("oph_maxent_green_ir_receipt")),
        "source_packet_audit_count": sum(1 for row in usable if row.get("oph_maxent_green_source_packet_audit")),
        "repair_clock_certificate_count": sum(
            1 for row in usable if row.get("oph_maxent_green_repair_clock_certificate")
        ),
        "bandlimit_for_ir_count": sum(1 for row in usable if row.get("oph_maxent_green_bandlimit_for_ir")),
        "bandlimit_for_requested_ell_count": sum(
            1 for row in usable if row.get("oph_maxent_green_bandlimit_for_requested_ell")
        ),
        "finite_lattice_derived_count": sum(
            1 for row in usable if row.get("oph_maxent_green_finite_lattice_derived")
        ),
        "physical_cmb_prediction_count": sum(
            1 for row in usable if row.get("oph_maxent_green_physical_cmb_prediction")
        ),
        "mean_patch_count": _mean(row.get("oph_maxent_green_patch_count") for row in usable),
        "mean_ell_max": _mean(row.get("oph_maxent_green_ell_max") for row in usable),
        "mean_eta_R": _mean(row.get("oph_maxent_green_eta_R") for row in usable),
        "mean_n_s": _mean(row.get("oph_maxent_green_n_s") for row in usable),
        "mean_fit_eta_R_error": _mean(row.get("oph_maxent_green_fit_eta_R_error") for row in usable),
        "mean_q_IR": _mean(row.get("oph_maxent_green_q_IR") for row in usable),
        "mean_ell_IR": _mean(row.get("oph_maxent_green_ell_IR") for row in usable),
        "mean_N_frz_proxy": _mean(row.get("oph_maxent_green_N_frz_proxy") for row in usable),
        "mean_F_IR_ell2": _mean(row.get("oph_maxent_green_F_IR_ell2") for row in usable),
        "mean_F_IR_ell32": _mean(row.get("oph_maxent_green_F_IR_ell32") for row in usable),
        "physical_cmb_prediction": False,
        "interpretation": (
            "Paper-side finite-screen CMB source certificate. This lane verifies the MaxEnt "
            "inverse-Laplacian screen covariance, selector-eliminated q_IR/ell_IR target, and a CAMB/CLASS "
            "primordial-table scaffold. The finite repair-clock certificate kappa_rep=e, finite freezeout "
            "emission, parity/BipoSH covariance, and official likelihood gates remain separate."
        ),
    }


def _repair_clock_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("repair_clock_written")]
    return {
        "run_count": len(usable),
        "finite_repair_clock_certificate_count": sum(
            1 for row in usable if row.get("repair_clock_finite_certificate")
        ),
        "repair_clock_certificate_count": sum(1 for row in usable if row.get("repair_clock_certificate")),
        "eta_R_finite_lattice_derived_count": sum(
            1 for row in usable if row.get("repair_clock_eta_finite_lattice_derived")
        ),
        "cycle_time_normalization_declared_count": sum(
            1 for row in usable if row.get("repair_clock_cycle_time_normalization_declared")
        ),
        "mean_candidate_run_count": _mean(row.get("repair_clock_candidate_run_count") for row in usable),
        "mean_estimator_count": _mean(row.get("repair_clock_estimator_count") for row in usable),
        "mean_eligible_estimator_count": _mean(
            row.get("repair_clock_eligible_estimator_count") for row in usable
        ),
        "mean_passed_estimator_count": _mean(row.get("repair_clock_passed_estimator_count") for row in usable),
        "mean_median_kappa_rep": _mean(row.get("repair_clock_median_kappa_rep") for row in usable),
        "mean_median_eta_R": _mean(row.get("repair_clock_median_eta_R") for row in usable),
        "mean_median_n_s": _mean(row.get("repair_clock_median_n_s") for row in usable),
        "target_kappa_rep": _first_non_null(row.get("repair_clock_target_kappa_rep") for row in usable),
        "target_eta_R": _first_non_null(row.get("repair_clock_target_eta_R") for row in usable),
        "mean_blocker_count": _mean(row.get("repair_clock_blocker_count") for row in usable),
        "physical_cmb_prediction": False,
        "interpretation": (
            "Finite repair-clock audit for the exact OPH CMB scalar tilt eta_R=kappa_rep*(P-phi). "
            "This is the remaining dynamic certificate after q_IR and ell_IR selector elimination. "
            "Diagnostic trace fits are kept visible, but the lane does not certify a physical CMB "
            "prediction unless theorem-grade eligible rows derive kappa_rep=e under declared controls."
        ),
    }


def _scalar_repair_semigroup_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("scalar_repair_semigroup_written")]
    return {
        "run_count": len(usable),
        "semigroup_target_receipt_count": sum(
            1 for row in usable if row.get("scalar_repair_semigroup_target_receipt")
        ),
        "repair_clock_certificate_count": sum(
            1 for row in usable if row.get("scalar_repair_semigroup_repair_clock_certificate")
        ),
        "eligible_for_certificate_count": sum(
            1 for row in usable if row.get("scalar_repair_semigroup_eligible_for_certificate")
        ),
        "finite_lattice_derived_count": sum(
            1 for row in usable if row.get("scalar_repair_semigroup_finite_lattice_derived")
        ),
        "source_values": sorted(
            {
                str(row.get("scalar_repair_semigroup_source"))
                for row in usable
                if row.get("scalar_repair_semigroup_source") is not None
            }
        ),
        "mean_dimension": _mean(row.get("scalar_repair_semigroup_dimension") for row in usable),
        "mean_centered_subspace_dimension": _mean(row.get("scalar_repair_semigroup_centered_dim") for row in usable),
        "mean_kappa_rep": _mean(row.get("scalar_repair_semigroup_kappa_rep") for row in usable),
        "mean_eta_R": _mean(row.get("scalar_repair_semigroup_eta_R") for row in usable),
        "mean_n_s": _mean(row.get("scalar_repair_semigroup_n_s") for row in usable),
        "mean_centered_gap": _mean(row.get("scalar_repair_semigroup_centered_gap") for row in usable),
        "controls_passed_count": sum(
            1 for row in usable if row.get("scalar_repair_semigroup_controls_passed")
        ),
        "mean_required_repair_step_time_for_kappa_e": _mean(
            row.get("scalar_repair_semigroup_transition_required_step_time") for row in usable
        ),
        "mean_primary_lambda_2": _mean(
            row.get("scalar_repair_semigroup_transition_primary_lambda_2") for row in usable
        ),
        "transition_clock_certified_count": sum(
            1 for row in usable if row.get("scalar_repair_semigroup_transition_clock_certified")
        ),
        "physical_cmb_prediction": False,
        "interpretation": (
            "Scalar repair-semigroup gap lane for the exact OPH CMB clock. A declared Euler target "
            "only checks the algebraic route to kappa_rep=e. A finite support-visible transition "
            "matrix is stronger, but it still certifies the CMB clock only if its declared repair-step "
            "normalization yields kappa_rep=e under the report controls."
        ),
    }


def _fossil_spectrum_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("fossil_spectrum_written")]
    return {
        "run_count": len(usable),
        "near_scale_invariant_transient_count": sum(
            1 for row in usable if row.get("fossil_spectrum_near_scale_invariant_transient")
        ),
        "best_beats_same_field_controls_count": sum(
            1 for row in usable if row.get("fossil_spectrum_best_beats_controls")
        ),
        "physical_cmb_prediction_count": sum(
            1 for row in usable if row.get("fossil_spectrum_physical_cmb_prediction")
        ),
        "best_field_counts": _counts(row.get("fossil_spectrum_best_field") for row in usable),
        "mean_best_cycle": _mean(row.get("fossil_spectrum_best_cycle") for row in usable),
        "mean_best_eta_R": _mean(row.get("fossil_spectrum_best_eta_R") for row in usable),
        "mean_best_n_s": _mean(row.get("fossil_spectrum_best_n_s") for row in usable),
        "mean_best_abs_eta_delta": _mean(row.get("fossil_spectrum_best_abs_eta_delta") for row in usable),
        "mean_best_control_abs_eta_delta": _mean(
            row.get("fossil_spectrum_best_control_abs_eta_delta") for row in usable
        ),
        "mean_freezeout_cycle": _mean(row.get("fossil_spectrum_freezeout_cycle") for row in usable),
        "mean_phi_zero_cycle": _mean(row.get("fossil_spectrum_phi_zero_cycle") for row in usable),
        "mean_phi_half_cycle": _mean(row.get("fossil_spectrum_phi_half_cycle") for row in usable),
        "physical_cmb_prediction": False,
        "interpretation": (
            "Time-resolved fossil-spectrum diagnostic from finite harmonic traces. A near-scale-invariant "
            "transient is only a clue unless a paper-derived freezeout rule selects that cycle before "
            "measurement comparison and controls/refinement pass."
        ),
    }


def _cmb_fossil_bridge_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("cmb_fossil_bridge_written")]
    physical_count = sum(1 for row in usable if row.get("cmb_fossil_bridge_physical_cmb_prediction"))
    return {
        "run_count": len(usable),
        "bridge_receipt_count": sum(1 for row in usable if row.get("cmb_fossil_bridge_receipt")),
        "physical_cmb_prediction_count": physical_count,
        "mean_shape_correlation": _mean(row.get("cmb_fossil_bridge_shape_correlation") for row in usable),
        "mean_normalized_rmse": _mean(row.get("cmb_fossil_bridge_normalized_rmse") for row in usable),
        "mean_eta": _mean(row.get("cmb_fossil_bridge_eta") for row in usable),
        "mean_q_ir": _mean(row.get("cmb_fossil_bridge_q_ir") for row in usable),
        "mean_ell_ir": _mean(row.get("cmb_fossil_bridge_ell_ir") for row in usable),
        "physical_cmb_prediction": bool(physical_count > 0),
        "interpretation": (
            "CMB1 OPH-CET fossil bridge: analytic observer-consensus screen covariance mapped to a "
            "measurement-facing TT diagnostic. This is not a finite-lattice physical CMB prediction "
            "unless the later OPH kernel and Boltzmann certificate gates pass."
        ),
    }


def _oph_inflation_cmb_bridge_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("oph_inflation_cmb_bridge_written")]
    return {
        "run_count": len(usable),
        "v04_diagnostic_available_count": sum(1 for row in usable if row.get("oph_inflation_cmb_v04_available")),
        "physical_cmb_prediction_count": sum(
            1 for row in usable if row.get("oph_inflation_cmb_bridge_physical_cmb_prediction")
        ),
        "mean_n_s": _mean(row.get("oph_inflation_cmb_n_s") for row in usable),
        "mean_theta_OPH": _mean(row.get("oph_inflation_cmb_theta_OPH") for row in usable),
        "mean_A_zeta": _mean(row.get("oph_inflation_cmb_A_zeta") for row in usable),
        "mean_n_s_pull_vs_planck": _mean(row.get("oph_inflation_cmb_n_s_pull") for row in usable),
        "mean_Omega_K": _mean(row.get("oph_inflation_cmb_Omega_K") for row in usable),
        "mean_Omega_A0": _mean(row.get("oph_inflation_cmb_Omega_A0") for row in usable),
        "mean_rho_A_over_rho_b": _mean(row.get("oph_inflation_cmb_rho_A_over_rho_b") for row in usable),
        "mean_q_IR": _mean(row.get("oph_inflation_cmb_v04_q_IR") for row in usable),
        "mean_ell_IR": _mean(row.get("oph_inflation_cmb_v04_ell_IR") for row in usable),
        "mean_camb_lcdm_lowell_chi2": _mean(
            row.get("oph_inflation_cmb_v04_camb_lcdm_lowell_chi2") for row in usable
        ),
        "mean_camb_oph_ir_lowell_chi2": _mean(
            row.get("oph_inflation_cmb_v04_camb_oph_ir_lowell_chi2") for row in usable
        ),
        "mean_lcdm_roe_pte": _mean(row.get("oph_inflation_cmb_v04_lcdm_roe_pte") for row in usable),
        "mean_oph_parity_roe_pte": _mean(
            row.get("oph_inflation_cmb_v04_oph_parity_roe_pte") for row in usable
        ),
        "mean_v05_TT_lowell_delta_chi2": _mean(
            row.get("oph_inflation_cmb_v05_TT_lowell_delta_chi2") for row in usable
        ),
        "mean_v05_TE_lowell_delta_chi2": _mean(
            row.get("oph_inflation_cmb_v05_TE_lowell_delta_chi2") for row in usable
        ),
        "mean_v05_EE_lowell_delta_chi2": _mean(
            row.get("oph_inflation_cmb_v05_EE_lowell_delta_chi2") for row in usable
        ),
        "mean_v05_TT_high_ell_delta_chi2_30_1200": _mean(
            row.get("oph_inflation_cmb_v05_TT_high_ell_delta_chi2_30_1200") for row in usable
        ),
        "mean_v05_combined_lowell_delta_chi2": _mean(
            row.get("oph_inflation_cmb_v05_combined_lowell_delta_chi2") for row in usable
        ),
        "mean_v05_pressure_point_count": _mean(
            row.get("oph_inflation_cmb_v05_pressure_point_count") for row in usable
        ),
        "physical_cmb_prediction": False,
        "interpretation": (
            "Continuation bridge imported from the Pro CMB/inflation notes. It supplies P/48 screen-spectrum "
            "numbers, zero-holonomy flat-sector bookkeeping, and measured Planck low-ell/CAMB diagnostic "
            "ladders including v0.5 TT/TE/EE hard-gate proxies. It is not a finite-lattice derivation of "
            "those numbers and not an official Planck likelihood."
        ),
    }


def _inflation_certificate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("inflation_certificates_written")]
    return {
        "run_count": len(usable),
        "stack_ready_count": sum(1 for row in usable if row.get("inflation_certificates_stack_ready")),
        "physical_cmb_prediction_count": sum(
            1 for row in usable if row.get("inflation_certificates_physical_cmb_prediction")
        ),
        "physical_matter_power_prediction_count": sum(
            1 for row in usable if row.get("inflation_certificates_physical_matter_power_prediction")
        ),
        "no_data_use_count": sum(1 for row in usable if row.get("inflation_certificates_no_data_use")),
        "scalar_release_gate_count": sum(1 for row in usable if row.get("inflation_certificates_scalar_release_gate")),
        "edge_center_gate_count": sum(1 for row in usable if row.get("inflation_certificates_edge_center_gate")),
        "homogeneous_anomaly_gate_count": sum(
            1 for row in usable if row.get("inflation_certificates_homogeneous_anomaly_gate")
        ),
        "parent_collar_gate_count": sum(1 for row in usable if row.get("inflation_certificates_parent_collar_gate")),
        "repair_matrix_gate_count": sum(1 for row in usable if row.get("inflation_certificates_repair_matrix_gate")),
        "boltzmann_handoff_gate_count": sum(
            1 for row in usable if row.get("inflation_certificates_boltzmann_handoff_gate")
        ),
        "mean_found_count": _mean(row.get("inflation_certificates_found_count") for row in usable),
        "mean_passed_count": _mean(row.get("inflation_certificates_passed_count") for row in usable),
        "mean_expected_count": _mean(row.get("inflation_certificates_expected_count") for row in usable),
        "mean_missing_count": _mean(row.get("inflation_certificates_missing_count") for row in usable),
        "mean_A_zeta": _mean(row.get("inflation_certificates_A_zeta") for row in usable),
        "mean_n_s": _mean(row.get("inflation_certificates_n_s") for row in usable),
        "mean_Q_A": _mean(row.get("inflation_certificates_Q_A") for row in usable),
        "mean_B_A": _mean(row.get("inflation_certificates_mean_B_A") for row in usable),
        "mean_Gamma_rec": _mean(row.get("inflation_certificates_mean_Gamma_rec") for row in usable),
        "physical_cmb_prediction": False,
        "interpretation": (
            "Finite certificate stack for the inflation-free OPH cosmology branch. This lane is the "
            "artifact contract for A_zeta, n_s, Q_A, B_A(k,a), Gamma_rec(k,a), and Boltzmann handoff. "
            "The current report is a physical prediction only when all finite certificates and the "
            "no-data-use firewall pass before likelihood comparison."
        ),
    }


def _oph_inflation_cmb_camb_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("oph_inflation_cmb_camb_written")]
    return {
        "run_count": len(usable),
        "measurement_comparable_curve_count": sum(
            1 for row in usable if row.get("oph_inflation_cmb_camb_curve_comparable")
        ),
        "transfer_receipt_count": sum(1 for row in usable if row.get("oph_inflation_cmb_camb_transfer_receipt")),
        "physical_cmb_prediction_count": sum(
            1 for row in usable if row.get("oph_inflation_cmb_camb_physical_cmb_prediction")
        ),
        "mean_n_s": _mean(row.get("oph_inflation_cmb_camb_n_s") for row in usable),
        "mean_A_zeta": _mean(row.get("oph_inflation_cmb_camb_A_zeta") for row in usable),
        "mean_q_IR": _mean(row.get("oph_inflation_cmb_camb_q_IR") for row in usable),
        "mean_ell_IR": _mean(row.get("oph_inflation_cmb_camb_ell_IR") for row in usable),
        "mean_lcdm_shape_correlation": _mean(
            row.get("oph_inflation_cmb_camb_lcdm_shape_correlation") for row in usable
        ),
        "mean_p48_shape_correlation": _mean(
            row.get("oph_inflation_cmb_camb_p48_shape_correlation") for row in usable
        ),
        "mean_p48_ir_shape_correlation": _mean(
            row.get("oph_inflation_cmb_camb_p48_ir_shape_correlation") for row in usable
        ),
        "mean_unique_n_s": _mean(row.get("oph_inflation_cmb_camb_unique_n_s") for row in usable),
        "mean_unique_q_IR": _mean(row.get("oph_inflation_cmb_camb_unique_q_IR") for row in usable),
        "mean_unique_ell_IR": _mean(row.get("oph_inflation_cmb_camb_unique_ell_IR") for row in usable),
        "mean_unique_shape_correlation": _mean(
            row.get("oph_inflation_cmb_camb_unique_shape_correlation") for row in usable
        ),
        "mean_lcdm_chi2_per_bin": _mean(
            row.get("oph_inflation_cmb_camb_lcdm_chi2_per_bin") for row in usable
        ),
        "mean_p48_chi2_per_bin": _mean(
            row.get("oph_inflation_cmb_camb_p48_chi2_per_bin") for row in usable
        ),
        "mean_p48_ir_chi2_per_bin": _mean(
            row.get("oph_inflation_cmb_camb_p48_ir_chi2_per_bin") for row in usable
        ),
        "mean_unique_chi2_per_bin": _mean(
            row.get("oph_inflation_cmb_camb_unique_chi2_per_bin") for row in usable
        ),
        "mean_acoustic_abs_delta": _mean(
            row.get("oph_inflation_cmb_camb_acoustic_mean_abs_delta") for row in usable
        ),
        "mean_unique_acoustic_abs_delta": _mean(
            row.get("oph_inflation_cmb_camb_unique_acoustic_mean_abs_delta") for row in usable
        ),
        "mean_lowell_lcdm_chi2": _mean(
            row.get("oph_inflation_cmb_camb_lowell_lcdm_chi2") for row in usable
        ),
        "mean_lowell_oph_ir_chi2": _mean(
            row.get("oph_inflation_cmb_camb_lowell_oph_ir_chi2") for row in usable
        ),
        "physical_cmb_prediction": False,
        "interpretation": (
            "Direct CAMB TT transfer for the OPH P/48 screen spectrum plus imported v0.4 IR kernel. "
            "This lane emits an actual CMB curve and compares it with the local Planck TT benchmark table. "
            "It remains a continuation diagnostic until the finite lattice derives A_zeta, q_IR, ell_IR, "
            "and angular covariance from cap/collar microphysics."
        ),
    }


def _selector_elimination_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("oph_selector_elimination_written")]
    return {
        "run_count": len(usable),
        "theorem_side_receipt_count": sum(
            1 for row in usable if row.get("oph_selector_elimination_theorem_receipt")
        ),
        "source_packet_audit_receipt_count": sum(
            1 for row in usable if row.get("oph_selector_elimination_source_audit_receipt")
        ),
        "q_IR_selector_removed_count": sum(1 for row in usable if row.get("oph_selector_q_IR_removed")),
        "ell_IR_selector_removed_count": sum(1 for row in usable if row.get("oph_selector_ell_IR_removed")),
        "eta_R_repair_clock_reduction_count": sum(
            1 for row in usable if row.get("oph_selector_eta_R_reduced_to_repair_clock")
        ),
        "finite_lattice_derived_count": sum(1 for row in usable if row.get("oph_selector_finite_lattice_derived")),
        "physical_cmb_prediction_count": sum(
            1 for row in usable if row.get("oph_selector_physical_cmb_prediction")
        ),
        "kernel_csv_passed_count": sum(1 for row in usable if row.get("oph_selector_kernel_csv_passed")),
        "mean_kernel_csv_max_abs_error": _mean(row.get("oph_selector_kernel_csv_max_abs_error") for row in usable),
        "mean_n_s": _mean(row.get("oph_selector_n_s") for row in usable),
        "mean_eta_R": _mean(row.get("oph_selector_eta_R") for row in usable),
        "mean_q_IR": _mean(row.get("oph_selector_q_IR") for row in usable),
        "mean_ell_IR": _mean(row.get("oph_selector_ell_IR") for row in usable),
        "kappa_rep_status_counts": _counts(
            row.get("oph_selector_kappa_rep_status") for row in usable if row.get("oph_selector_kappa_rep_status")
        ),
        "physical_cmb_prediction": False,
        "interpretation": (
            "v1.5 OPH-CMB selector-elimination lane. q_IR and ell_IR are counted as theorem-side target "
            "derivations when their receipts pass; eta_R is only reduced to a repair-clock certificate "
            "until the finite lattice derives kappa_rep=e. This lane strengthens the exact target branch "
            "but does not by itself make the CAMB curve a physical finite-lattice CMB prediction."
        ),
    }


def _oph_exact_cmb_camb_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("oph_exact_cmb_camb_written")]
    return {
        "run_count": len(usable),
        "measurement_comparable_curve_count": sum(
            1 for row in usable if row.get("oph_exact_cmb_camb_curve_comparable")
        ),
        "transfer_receipt_count": sum(1 for row in usable if row.get("oph_exact_cmb_camb_transfer_receipt")),
        "physical_cmb_prediction_count": sum(
            1 for row in usable if row.get("oph_exact_cmb_camb_physical_cmb_prediction")
        ),
        "official_clik_ready_count": sum(1 for row in usable if row.get("oph_exact_cmb_camb_official_clik_ready")),
        "official_likelihood_ready_count": sum(
            1 for row in usable if row.get("oph_exact_cmb_camb_official_likelihood_ready")
        ),
        "selector_theorem_receipt_count": sum(
            1 for row in usable if row.get("oph_selector_elimination_theorem_receipt")
        ),
        "selector_source_audit_receipt_count": sum(
            1 for row in usable if row.get("oph_selector_elimination_source_audit_receipt")
        ),
        "mean_n_s": _mean(row.get("oph_exact_cmb_camb_n_s") for row in usable),
        "mean_eta_R": _mean(row.get("oph_exact_cmb_camb_eta_R") for row in usable),
        "mean_q_IR": _mean(row.get("oph_exact_cmb_camb_q_IR") for row in usable),
        "mean_ell_IR": _mean(row.get("oph_exact_cmb_camb_ell_IR") for row in usable),
        "mean_N_frz_proxy": _mean(row.get("oph_exact_cmb_camb_N_frz_proxy") for row in usable),
        "mean_lcdm_shape_correlation": _mean(row.get("oph_exact_cmb_camb_lcdm_shape_correlation") for row in usable),
        "mean_scalar_shape_correlation": _mean(
            row.get("oph_exact_cmb_camb_scalar_shape_correlation") for row in usable
        ),
        "mean_ir_shape_correlation": _mean(row.get("oph_exact_cmb_camb_ir_shape_correlation") for row in usable),
        "mean_lcdm_chi2_per_bin": _mean(row.get("oph_exact_cmb_camb_lcdm_chi2_per_bin") for row in usable),
        "mean_scalar_chi2_per_bin": _mean(row.get("oph_exact_cmb_camb_scalar_chi2_per_bin") for row in usable),
        "mean_ir_chi2_per_bin": _mean(row.get("oph_exact_cmb_camb_ir_chi2_per_bin") for row in usable),
        "mean_acoustic_abs_delta": _mean(row.get("oph_exact_cmb_camb_acoustic_mean_abs_delta") for row in usable),
        "physical_cmb_prediction": False,
        "interpretation": (
            "Native CAMB TT transfer for the exact OPH CMB target branch updated with the v1.5 "
            "selector-elimination receipts: q_IR and ell_IR are theorem-side target counts, while "
            "n_s still depends on the pending kappa_rep=e repair-clock certificate. This is "
            "measurement-comparable Boltzmann output, but remains a target/continuation diagnostic until "
            "finite lattice derivation and official Planck likelihood/map-space gates pass."
        ),
    }


def _finite_repair_clock_cmb_camb_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("finite_clock_cmb_camb_written")]
    return {
        "run_count": len(usable),
        "measurement_comparable_curve_count": sum(
            1 for row in usable if row.get("finite_clock_cmb_camb_curve_comparable")
        ),
        "transfer_receipt_count": sum(1 for row in usable if row.get("finite_clock_cmb_camb_transfer_receipt")),
        "finite_lattice_clock_derived_count": sum(
            1 for row in usable if row.get("finite_clock_cmb_camb_finite_lattice_clock_derived")
        ),
        "repair_clock_certificate_count": sum(
            1 for row in usable if row.get("finite_clock_cmb_camb_repair_clock_certificate")
        ),
        "clock_numeric_match_count": sum(
            1 for row in usable if row.get("finite_clock_cmb_camb_clock_numeric_match")
        ),
        "repair_scale_hypothesis_match_count": sum(
            1 for row in usable if row.get("finite_clock_cmb_camb_repair_scale_hypothesis_match")
        ),
        "clock_source_theorem_grade_count": sum(
            1 for row in usable if row.get("finite_clock_cmb_camb_clock_source_theorem_grade")
        ),
        "clock_source_counts": _counts(
            row.get("finite_clock_cmb_camb_clock_source") for row in usable
        ),
        "selector_ir_theory_side_count": sum(
            1 for row in usable if row.get("finite_clock_cmb_camb_selector_ir_theory_side")
        ),
        "physical_cmb_prediction_count": sum(
            1 for row in usable if row.get("finite_clock_cmb_camb_physical_cmb_prediction")
        ),
        "mean_n_s": _mean(row.get("finite_clock_cmb_camb_n_s") for row in usable),
        "mean_eta_R": _mean(row.get("finite_clock_cmb_camb_eta_R") for row in usable),
        "mean_kappa_rep": _mean(row.get("finite_clock_cmb_camb_kappa_rep") for row in usable),
        "mean_q_IR": _mean(row.get("finite_clock_cmb_camb_q_IR") for row in usable),
        "mean_ell_IR": _mean(row.get("finite_clock_cmb_camb_ell_IR") for row in usable),
        "mean_lcdm_shape_correlation": _mean(
            row.get("finite_clock_cmb_camb_lcdm_shape_correlation") for row in usable
        ),
        "mean_scalar_shape_correlation": _mean(
            row.get("finite_clock_cmb_camb_scalar_shape_correlation") for row in usable
        ),
        "mean_ir_shape_correlation": _mean(
            row.get("finite_clock_cmb_camb_ir_shape_correlation") for row in usable
        ),
        "mean_lcdm_chi2_per_bin": _mean(row.get("finite_clock_cmb_camb_lcdm_chi2_per_bin") for row in usable),
        "mean_scalar_chi2_per_bin": _mean(
            row.get("finite_clock_cmb_camb_scalar_chi2_per_bin") for row in usable
        ),
        "mean_ir_chi2_per_bin": _mean(row.get("finite_clock_cmb_camb_ir_chi2_per_bin") for row in usable),
        "mean_acoustic_abs_delta": _mean(
            row.get("finite_clock_cmb_camb_acoustic_mean_abs_delta") for row in usable
        ),
        "physical_cmb_prediction": False,
        "interpretation": (
            "Native CAMB TT transfer using the finite transition-matrix repair clock emitted by the simulator. "
            "This is the strongest current simulator-derived CMB curve lane: n_s comes from finite repair data. "
            "It remains non-physical until the exact repair-clock certificate, selector finite-register "
            "certificate, physical source/k calibration, and official likelihood gates pass."
        ),
    }


def _oph_unique_prediction_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("oph_unique_prediction_written")]
    return {
        "run_count": len(usable),
        "measurement_comparable_count": sum(
            1 for row in usable if row.get("oph_unique_prediction_measurement_comparable")
        ),
        "finite_lattice_derived_count": sum(
            1 for row in usable if row.get("oph_unique_prediction_finite_lattice_derived")
        ),
        "physical_cmb_prediction_count": sum(
            1 for row in usable if row.get("oph_unique_prediction_physical_cmb_prediction")
        ),
        "mean_n_s": _mean(row.get("oph_unique_n_s") for row in usable),
        "mean_eta_R": _mean(row.get("oph_unique_eta_R") for row in usable),
        "mean_n_s_pull": _mean(row.get("oph_unique_n_s_pull") for row in usable),
        "mean_q_IR": _mean(row.get("oph_unique_q_IR") for row in usable),
        "mean_ell_IR": _mean(row.get("oph_unique_ell_IR") for row in usable),
        "mean_N_frz_proxy": _mean(row.get("oph_unique_N_frz_proxy") for row in usable),
        "mean_parity_R_OE_TT_2_29": _mean(row.get("oph_unique_parity_R_OE_TT_2_29") for row in usable),
        "mean_sum_mnu_eV": _mean(row.get("oph_unique_sum_mnu_eV") for row in usable),
        "mean_neutrino_f_nu": _mean(row.get("oph_unique_neutrino_f_nu") for row in usable),
        "mean_small_scale_neutrino_suppression": _mean(
            row.get("oph_unique_small_scale_neutrino_suppression") for row in usable
        ),
        "physical_cmb_prediction": False,
        "interpretation": (
            "Current v0.9 OPH-only public-comparison target lane: alpha-linked scalar tilt, exact IR "
            "kernel, parity envelope, neutrino mass sum, and compressed dark-sector rows. These are "
            "measurement-comparable targets, not finite-lattice derivations unless the derivation audit passes."
        ),
    }


def _oph_cnb_neutrino_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("oph_cnb_neutrino_written")]
    return {
        "run_count": len(usable),
        "measurement_comparable_count": sum(
            1 for row in usable if row.get("oph_cnb_neutrino_measurement_comparable")
        ),
        "finite_lattice_derived_count": sum(
            1 for row in usable if row.get("oph_cnb_neutrino_finite_lattice_derived")
        ),
        "physical_cmb_prediction_count": sum(
            1 for row in usable if row.get("oph_cnb_neutrino_physical_cmb_prediction")
        ),
        "physical_matter_power_prediction_count": sum(
            1 for row in usable if row.get("oph_cnb_neutrino_physical_matter_power_prediction")
        ),
        "background_gate_count": sum(1 for row in usable if row.get("oph_cnb_background_gate")),
        "mass_derivation_gate_count": sum(1 for row in usable if row.get("oph_cnb_mass_derivation_gate")),
        "B_A_kernel_gate_count": sum(1 for row in usable if row.get("oph_cnb_B_A_kernel_gate")),
        "five_of_seven_kernel_callable_count": sum(
            1 for row in usable if row.get("oph_cnb_five_of_seven_kernel_callable")
        ),
        "five_of_seven_projection_count": sum(
            1 for row in usable if row.get("oph_cnb_five_of_seven_projection_gate")
        ),
        "likelihood_gate_count": sum(1 for row in usable if row.get("oph_cnb_likelihood_gate")),
        "planck_bao_bound_pass_count": sum(1 for row in usable if row.get("oph_cnb_planck_bao_sum_mnu_pass")),
        "act_bound_pass_count": sum(1 for row in usable if row.get("oph_cnb_act_sum_mnu_pass")),
        "desi_lcdm_bound_pass_count": sum(1 for row in usable if row.get("oph_cnb_desi_lcdm_sum_mnu_pass")),
        "mean_N_eff": _mean(row.get("oph_cnb_N_eff") for row in usable),
        "mean_sum_mnu_eV": _mean(row.get("oph_cnb_sum_mnu_eV") for row in usable),
        "mean_m_lightest_eV": _mean(row.get("oph_cnb_m_lightest_eV") for row in usable),
        "mean_Omega_nu_h2": _mean(row.get("oph_cnb_Omega_nu_h2") for row in usable),
        "mean_f_nu": _mean(row.get("oph_cnb_f_nu") for row in usable),
        "mean_small_scale_suppression": _mean(row.get("oph_cnb_small_scale_suppression") for row in usable),
        "mean_planck_neff_pull_sigma": _mean(row.get("oph_cnb_planck_neff_pull_sigma") for row in usable),
        "mean_eta_A": _mean(row.get("oph_cnb_eta_A") for row in usable),
        "mean_Pi_WL_compressed_required": _mean(row.get("oph_cnb_Pi_WL_compressed_required") for row in usable),
        "mean_five_of_seven_pi_wl": _mean(row.get("oph_cnb_five_of_seven_pi_wl") for row in usable),
        "mean_five_of_seven_epsilon_A": _mean(row.get("oph_cnb_five_of_seven_epsilon_A") for row in usable),
        "mean_five_of_seven_S8_projected": _mean(
            row.get("oph_cnb_five_of_seven_S8_projected") for row in usable
        ),
        "mean_five_of_seven_pull_sigma": _mean(
            row.get("oph_cnb_five_of_seven_pull_sigma") for row in usable
        ),
        "physical_cmb_prediction": False,
        "interpretation": (
            "Standalone OPH-CnuB relic-neutrino background lane. The weighted-cycle neutrino masses are "
            "measurement-comparable through standard relic-neutrino cosmology, while finite-lattice mass "
            "derivation, B_A(k,a), and full Boltzmann/likelihood gates remain closed."
        ),
    }


def _finite_certificate_authority_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("finite_certificates_written")]
    return {
        "run_count": len(usable),
        "compiler_ready_count": sum(1 for row in usable if row.get("finite_certificates_compiler_ready")),
        "stack_ready_count": sum(1 for row in usable if row.get("finite_certificates_stack_ready")),
        "theorem_grade_finite_inputs_count": sum(
            1 for row in usable if row.get("finite_certificates_theorem_grade_inputs")
        ),
        "proxy_certificate_count": sum(1 for row in usable if row.get("finite_certificates_proxy_certificate")),
        "no_data_use_count": sum(1 for row in usable if row.get("finite_certificates_no_data_use")),
        "release_code_gate_count": sum(1 for row in usable if row.get("finite_certificates_release_code_gate")),
        "parent_collar_gate_count": sum(1 for row in usable if row.get("finite_certificates_parent_collar_gate")),
        "repair_matrix_gate_count": sum(1 for row in usable if row.get("finite_certificates_repair_matrix_gate")),
        "boltzmann_export_gate_count": sum(1 for row in usable if row.get("finite_certificates_boltzmann_export_gate")),
        "real_physics_certificate_count": sum(
            1 for row in usable if row.get("finite_certificates_real_physics_certificate")
        ),
        "physical_cmb_prediction_count": sum(
            1 for row in usable if row.get("finite_certificates_physical_cmb_prediction")
        ),
        "physical_matter_power_prediction_count": sum(
            1 for row in usable if row.get("finite_certificates_physical_matter_power_prediction")
        ),
        "mean_A_zeta": _mean(row.get("finite_certificates_A_zeta") for row in usable),
        "mean_n_s": _mean(row.get("finite_certificates_n_s") for row in usable),
        "mean_Q_A": _mean(row.get("finite_certificates_Q_A") for row in usable),
        "mean_B_A": _mean(row.get("finite_certificates_B_A") for row in usable),
        "mean_Gamma_rec": _mean(row.get("finite_certificates_Gamma_rec") for row in usable),
        "physical_cmb_prediction": False,
        "interpretation": (
            "Finite OPH cosmology certificate authority. This lane computes A_zeta, Q_A, B_A(k,a), "
            "Gamma_rec, and a Boltzmann handoff contract from finite release/collar/repair inputs. "
            "Compiler-ready means certificate artifacts are internally consistent, not theorem-grade. "
            "Toy/proxy inputs validate the compiler only; physical CMB and matter-power gates remain "
            "closed until real simulator regulator data and cold-limit solver receipts pass the firewall."
        ),
    }


def _scalar_quotient_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("scalar_quotient_written")]
    return {
        "run_count": len(usable),
        "scalar_quotient_receipt_count": sum(1 for row in usable if row.get("scalar_quotient_receipt")),
        "finite_ready_count": sum(1 for row in usable if row.get("scalar_quotient_finite_ready")),
        "active_33_level_clause_count": sum(
            1 for row in usable if row.get("scalar_quotient_active_33_level_clause")
        ),
        "theorem_grade_release_count": sum(
            1 for row in usable if row.get("scalar_quotient_theorem_grade_release")
        ),
        "physical_cmb_prediction_count": sum(
            1 for row in usable if row.get("scalar_quotient_physical_cmb_prediction")
        ),
        "mean_observer_count": _mean(row.get("scalar_quotient_observer_count") for row in usable),
        "mean_patch_count": _mean(row.get("scalar_quotient_patch_count") for row in usable),
        "mean_packet_alphabet_size": _mean(
            row.get("scalar_quotient_packet_alphabet_size") for row in usable
        ),
        "mean_packet_entropy_bits": _mean(row.get("scalar_quotient_packet_entropy_bits") for row in usable),
        "mean_n_s": _mean(row.get("scalar_quotient_n_s") for row in usable),
        "mean_theta_OPH": _mean(row.get("scalar_quotient_theta_OPH") for row in usable),
        "mean_target_ell_IR": _mean(row.get("scalar_quotient_target_ell_IR") for row in usable),
        "mean_observer_level_proxy": _mean(
            row.get("scalar_quotient_observer_level_proxy") for row in usable
        ),
        "mean_patch_capacity_level_proxy": _mean(
            row.get("scalar_quotient_patch_capacity_level_proxy") for row in usable
        ),
        "physical_cmb_prediction": False,
        "interpretation": (
            "Finite observer-visible scalar/geometric quotient lane. This is the direct instrumentation "
            "needed before CMB promotion: scalar packets, center-free scalar screen fields, P/48 readout, "
            "and the 33-level freezeout clause are audited separately. A scalar quotient receipt is not "
            "a physical CMB prediction unless finite-ready, theorem-grade release, parent-collar, repair, "
            "and likelihood gates also pass."
        ),
    }


def _neutral_profile_audit_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("neutral_profile_audit_written")]
    return {
        "run_count": len(usable),
        "strict_3d_ready_count": sum(
            int(row.get("neutral_profile_strict_3d_ready_count") or 0) for row in usable
        ),
        "mean_profile_count": _mean(row.get("neutral_profile_count") for row in usable),
        "mean_sampled_observer_count": _mean(row.get("neutral_profile_sampled_observer_count") for row in usable),
        "all_observer_visible_best_model_counts": _counts(
            row.get("neutral_profile_all_best_model") for row in usable
        ),
        "scalar_response_best_model_counts": _counts(
            row.get("neutral_profile_scalar_response_best_model") for row in usable
        ),
        "prime_geometric_best_model_counts": _counts(
            row.get("neutral_profile_prime_geometric_best_model") for row in usable
        ),
        "prime_control_quotient_best_model_counts": _counts(
            row.get("neutral_profile_prime_control_quotient_best_model") for row in usable
        ),
        "prime_rank3_best_model_counts": _counts(
            row.get("neutral_profile_prime_rank3_best_model") for row in usable
        ),
        "prime_rank8_best_model_counts": _counts(
            row.get("neutral_profile_prime_rank8_best_model") for row in usable
        ),
        "prime_control_quotient_rank3_best_model_counts": _counts(
            row.get("neutral_profile_prime_control_quotient_rank3_best_model") for row in usable
        ),
        "prime_control_quotient_rank8_best_model_counts": _counts(
            row.get("neutral_profile_prime_control_quotient_rank8_best_model") for row in usable
        ),
        "support_visible_best_model_counts": _counts(
            row.get("neutral_profile_support_visible_best_model") for row in usable
        ),
        "mean_all_corr_dim": _mean(row.get("neutral_profile_all_corr_dim") for row in usable),
        "mean_all_mle_dim": _mean(row.get("neutral_profile_all_mle_dim") for row in usable),
        "mean_all_s2_leakage_corr": _mean(row.get("neutral_profile_all_s2_leakage_corr") for row in usable),
        "all_s2_leakage_pass_count": sum(
            1 for row in usable if row.get("neutral_profile_all_s2_leakage_pass")
        ),
        "mean_scalar_response_corr_dim": _mean(
            row.get("neutral_profile_scalar_response_corr_dim") for row in usable
        ),
        "mean_scalar_response_mle_dim": _mean(
            row.get("neutral_profile_scalar_response_mle_dim") for row in usable
        ),
        "mean_scalar_response_s2_leakage_corr": _mean(
            row.get("neutral_profile_scalar_response_s2_leakage_corr") for row in usable
        ),
        "scalar_response_s2_leakage_pass_count": sum(
            1 for row in usable if row.get("neutral_profile_scalar_response_s2_leakage_pass")
        ),
        "mean_prime_geometric_corr_dim": _mean(
            row.get("neutral_profile_prime_geometric_corr_dim") for row in usable
        ),
        "mean_prime_geometric_mle_dim": _mean(
            row.get("neutral_profile_prime_geometric_mle_dim") for row in usable
        ),
        "mean_prime_geometric_s2_leakage_corr": _mean(
            row.get("neutral_profile_prime_geometric_s2_leakage_corr") for row in usable
        ),
        "prime_geometric_s2_leakage_pass_count": sum(
            1 for row in usable if row.get("neutral_profile_prime_geometric_s2_leakage_pass")
        ),
        "mean_prime_control_quotient_corr_dim": _mean(
            row.get("neutral_profile_prime_control_quotient_corr_dim") for row in usable
        ),
        "mean_prime_control_quotient_mle_dim": _mean(
            row.get("neutral_profile_prime_control_quotient_mle_dim") for row in usable
        ),
        "mean_prime_control_quotient_s2_leakage_corr": _mean(
            row.get("neutral_profile_prime_control_quotient_s2_leakage_corr") for row in usable
        ),
        "prime_control_quotient_s2_leakage_pass_count": sum(
            1 for row in usable if row.get("neutral_profile_prime_control_quotient_s2_leakage_pass")
        ),
        "mean_prime_rank3_corr_dim": _mean(
            row.get("neutral_profile_prime_rank3_corr_dim") for row in usable
        ),
        "mean_prime_rank3_mle_dim": _mean(
            row.get("neutral_profile_prime_rank3_mle_dim") for row in usable
        ),
        "mean_prime_rank3_s2_leakage_corr": _mean(
            row.get("neutral_profile_prime_rank3_s2_leakage_corr") for row in usable
        ),
        "mean_prime_control_quotient_rank3_corr_dim": _mean(
            row.get("neutral_profile_prime_control_quotient_rank3_corr_dim") for row in usable
        ),
        "mean_prime_control_quotient_rank3_mle_dim": _mean(
            row.get("neutral_profile_prime_control_quotient_rank3_mle_dim") for row in usable
        ),
        "mean_prime_control_quotient_rank3_s2_leakage_corr": _mean(
            row.get("neutral_profile_prime_control_quotient_rank3_s2_leakage_corr") for row in usable
        ),
        "mean_prime_rank8_corr_dim": _mean(
            row.get("neutral_profile_prime_rank8_corr_dim") for row in usable
        ),
        "mean_prime_rank8_mle_dim": _mean(
            row.get("neutral_profile_prime_rank8_mle_dim") for row in usable
        ),
        "mean_prime_rank8_s2_leakage_corr": _mean(
            row.get("neutral_profile_prime_rank8_s2_leakage_corr") for row in usable
        ),
        "mean_prime_control_quotient_rank8_corr_dim": _mean(
            row.get("neutral_profile_prime_control_quotient_rank8_corr_dim") for row in usable
        ),
        "mean_prime_control_quotient_rank8_mle_dim": _mean(
            row.get("neutral_profile_prime_control_quotient_rank8_mle_dim") for row in usable
        ),
        "mean_prime_control_quotient_rank8_s2_leakage_corr": _mean(
            row.get("neutral_profile_prime_control_quotient_rank8_s2_leakage_corr") for row in usable
        ),
        "mean_support_visible_corr_dim": _mean(
            row.get("neutral_profile_support_visible_corr_dim") for row in usable
        ),
        "mean_support_visible_mle_dim": _mean(
            row.get("neutral_profile_support_visible_mle_dim") for row in usable
        ),
        "mean_support_visible_s2_leakage_corr": _mean(
            row.get("neutral_profile_support_visible_s2_leakage_corr") for row in usable
        ),
        "support_visible_s2_leakage_pass_count": sum(
            1 for row in usable if row.get("neutral_profile_support_visible_s2_leakage_pass")
        ),
        "interpretation": (
            "Bounded diagnostic over neutral-distance feature profiles. It identifies whether the "
            "observer-visible packet is overcomplete, undercomplete, or still screen-leaky. It does "
            "not establish strict neutral 3D bulk."
        ),
    }


def _prime_rank_sweep_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("prime_rank_sweep_written")]
    return {
        "run_count": len(usable),
        "quotient_3d_diagnostic_receipt_count": sum(
            1 for row in usable if row.get("prime_rank_sweep_quotient_3d_diagnostic_receipt")
        ),
        "spatial_3d_candidate_receipt_count": sum(
            1 for row in usable if row.get("prime_rank_sweep_spatial_3d_candidate_receipt")
        ),
        "control_quotient_spatial_3d_candidate_receipt_count": sum(
            1
            for row in usable
            if row.get("prime_rank_sweep_control_quotient_spatial_3d_candidate_receipt")
        ),
        "strict_neutral_candidate_receipt_count": sum(
            1 for row in usable if row.get("prime_rank_sweep_strict_neutral_candidate_receipt")
        ),
        "selected_controls_written_count": sum(
            1 for row in usable if row.get("prime_rank_sweep_selected_controls_written")
        ),
        "selected_controls_all_failed_count": sum(
            1 for row in usable if row.get("prime_rank_sweep_selected_controls_all_failed")
        ),
        "coordinate_rank3_tautology_warning_count": sum(
            1 for row in usable if row.get("prime_rank_sweep_coordinate_rank3_tautology_warning")
        ),
        "selected_directional_control_survive_count": sum(
            int(row.get("prime_rank_sweep_selected_directional_control_survive_count") or 0)
            for row in usable
        ),
        "selected_directional_control_fail_count": sum(
            int(row.get("prime_rank_sweep_selected_directional_control_fail_count") or 0)
            for row in usable
        ),
        "selected_coordinate_control_survive_count": sum(
            int(row.get("prime_rank_sweep_selected_coordinate_control_survive_count") or 0)
            for row in usable
        ),
        "selected_coordinate_control_fail_count": sum(
            int(row.get("prime_rank_sweep_selected_coordinate_control_fail_count") or 0)
            for row in usable
        ),
        "control_quotient_negative_control_counts": _counts(
            row.get("prime_rank_sweep_control_quotient_is_negative_control") for row in usable
        ),
        "mean_rank_count": _mean(row.get("prime_rank_sweep_rank_count") for row in usable),
        "strict_3d_ready_count": sum(
            int(row.get("prime_rank_sweep_strict_3d_ready_count") or 0) for row in usable
        ),
        "dimension_3d_window_count": sum(
            int(row.get("prime_rank_sweep_dimension_3d_window_count") or 0) for row in usable
        ),
        "best_rank_counts": _counts(row.get("prime_rank_sweep_best_rank") for row in usable),
        "best_model_counts": _counts(row.get("prime_rank_sweep_best_model") for row in usable),
        "mean_best_corr_dim": _mean(row.get("prime_rank_sweep_best_corr_dim") for row in usable),
        "mean_best_mle_dim": _mean(row.get("prime_rank_sweep_best_mle_dim") for row in usable),
        "mean_best_s2_leakage_corr": _mean(row.get("prime_rank_sweep_best_s2_leakage_corr") for row in usable),
        "best_3d_rank_counts": _counts(row.get("prime_rank_sweep_best_3d_rank") for row in usable),
        "best_3d_model_counts": _counts(row.get("prime_rank_sweep_best_3d_model") for row in usable),
        "mean_best_3d_corr_dim": _mean(row.get("prime_rank_sweep_best_3d_corr_dim") for row in usable),
        "mean_best_3d_mle_dim": _mean(row.get("prime_rank_sweep_best_3d_mle_dim") for row in usable),
        "coordinate_spatial_3d_ready_count": sum(
            int(row.get("prime_rank_sweep_coordinate_spatial_3d_ready_count") or 0) for row in usable
        ),
        "coordinate_dimension_3d_window_count": sum(
            int(row.get("prime_rank_sweep_coordinate_dimension_3d_window_count") or 0) for row in usable
        ),
        "coordinate_best_rank_counts": _counts(
            row.get("prime_rank_sweep_coordinate_best_rank") for row in usable
        ),
        "coordinate_best_model_counts": _counts(
            row.get("prime_rank_sweep_coordinate_best_model") for row in usable
        ),
        "coordinate_mean_best_corr_dim": _mean(
            row.get("prime_rank_sweep_coordinate_best_corr_dim") for row in usable
        ),
        "coordinate_mean_best_mle_dim": _mean(
            row.get("prime_rank_sweep_coordinate_best_mle_dim") for row in usable
        ),
        "coordinate_mean_best_s2_leakage_corr": _mean(
            row.get("prime_rank_sweep_coordinate_best_s2_leakage_corr") for row in usable
        ),
        "coordinate_best_3d_rank_counts": _counts(
            row.get("prime_rank_sweep_coordinate_best_3d_rank") for row in usable
        ),
        "coordinate_best_3d_model_counts": _counts(
            row.get("prime_rank_sweep_coordinate_best_3d_model") for row in usable
        ),
        "coordinate_mean_best_3d_corr_dim": _mean(
            row.get("prime_rank_sweep_coordinate_best_3d_corr_dim") for row in usable
        ),
        "coordinate_mean_best_3d_mle_dim": _mean(
            row.get("prime_rank_sweep_coordinate_best_3d_mle_dim") for row in usable
        ),
        "control_quotient_strict_3d_ready_count": sum(
            int(row.get("prime_rank_sweep_control_quotient_strict_3d_ready_count") or 0)
            for row in usable
        ),
        "control_quotient_dimension_3d_window_count": sum(
            int(row.get("prime_rank_sweep_control_quotient_dimension_3d_window_count") or 0)
            for row in usable
        ),
        "control_quotient_best_rank_counts": _counts(
            row.get("prime_rank_sweep_control_quotient_best_rank") for row in usable
        ),
        "control_quotient_best_model_counts": _counts(
            row.get("prime_rank_sweep_control_quotient_best_model") for row in usable
        ),
        "control_quotient_mean_best_corr_dim": _mean(
            row.get("prime_rank_sweep_control_quotient_best_corr_dim") for row in usable
        ),
        "control_quotient_mean_best_mle_dim": _mean(
            row.get("prime_rank_sweep_control_quotient_best_mle_dim") for row in usable
        ),
        "control_quotient_mean_best_s2_leakage_corr": _mean(
            row.get("prime_rank_sweep_control_quotient_best_s2_leakage_corr") for row in usable
        ),
        "control_quotient_coordinate_spatial_3d_ready_count": sum(
            int(row.get("prime_rank_sweep_control_quotient_coordinate_spatial_3d_ready_count") or 0)
            for row in usable
        ),
        "control_quotient_coordinate_dimension_3d_window_count": sum(
            int(row.get("prime_rank_sweep_control_quotient_coordinate_dimension_3d_window_count") or 0)
            for row in usable
        ),
        "control_quotient_coordinate_best_rank_counts": _counts(
            row.get("prime_rank_sweep_control_quotient_coordinate_best_rank") for row in usable
        ),
        "control_quotient_coordinate_best_model_counts": _counts(
            row.get("prime_rank_sweep_control_quotient_coordinate_best_model") for row in usable
        ),
        "control_quotient_coordinate_mean_best_corr_dim": _mean(
            row.get("prime_rank_sweep_control_quotient_coordinate_best_corr_dim") for row in usable
        ),
        "control_quotient_coordinate_mean_best_mle_dim": _mean(
            row.get("prime_rank_sweep_control_quotient_coordinate_best_mle_dim") for row in usable
        ),
        "control_quotient_coordinate_mean_best_s2_leakage_corr": _mean(
            row.get("prime_rank_sweep_control_quotient_coordinate_best_s2_leakage_corr") for row in usable
        ),
        "control_quotient_coordinate_best_3d_rank_counts": _counts(
            row.get("prime_rank_sweep_control_quotient_coordinate_best_3d_rank") for row in usable
        ),
        "control_quotient_coordinate_best_3d_model_counts": _counts(
            row.get("prime_rank_sweep_control_quotient_coordinate_best_3d_model") for row in usable
        ),
        "control_quotient_coordinate_mean_best_3d_corr_dim": _mean(
            row.get("prime_rank_sweep_control_quotient_coordinate_best_3d_corr_dim") for row in usable
        ),
        "control_quotient_coordinate_mean_best_3d_mle_dim": _mean(
            row.get("prime_rank_sweep_control_quotient_coordinate_best_3d_mle_dim") for row in usable
        ),
        "control_quotient_coordinate_mean_best_3d_s2_leakage_corr": _mean(
            row.get("prime_rank_sweep_control_quotient_coordinate_best_3d_s2_leakage_corr")
            for row in usable
        ),
        "control_quotient_coordinate_best_3d_s2_leakage_pass_count": sum(
            1 for row in usable if row.get("prime_rank_sweep_control_quotient_coordinate_best_3d_s2_leakage_pass")
        ),
        "physical_claim": False,
        "interpretation": (
            "Diagnostic low-rank quotient sweep over prime-geometric modular response and its "
            "finite-regulator control quotient. Directional/cosine rows test angular response structure; "
            "coordinate rows test whether response coordinates already carry a neutral 3D spatial chart. "
            "The control quotient is a target-response quotient with regulator-control directions removed, "
            "not a shuffled/null negative control. These are diagnostics only until controls and refinement pass."
        ),
    }


def _prime_rank_refinement_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("prime_rank_refinement_written")]
    dimension_drifts = [
        row.get("prime_rank_refinement_dimension_drift")
        for row in usable
        if isinstance(row.get("prime_rank_refinement_dimension_drift"), (int, float))
    ]
    return {
        "run_count": len(usable),
        "candidate_receipt_count": sum(
            1 for row in usable if row.get("prime_rank_refinement_candidate_receipt")
        ),
        "strict_neutral_receipt_count": sum(
            1 for row in usable if row.get("prime_rank_refinement_strict_neutral_receipt")
        ),
        "multi_scale_count": sum(1 for row in usable if row.get("prime_rank_refinement_multi_scale")),
        "all_candidate_count": sum(1 for row in usable if row.get("prime_rank_refinement_all_candidates")),
        "all_candidate_s2_leakage_pass_count": sum(
            1 for row in usable if row.get("prime_rank_refinement_all_candidate_s2_leakage_pass")
        ),
        "all_candidate_rank3_e3_count": sum(
            1 for row in usable if row.get("prime_rank_refinement_all_candidate_rank3_e3")
        ),
        "dimension_stable_count": sum(
            1 for row in usable if row.get("prime_rank_refinement_dimension_stable")
        ),
        "independent_rank3_all_count": sum(
            1 for row in usable if row.get("prime_rank_refinement_independent_rank3_all")
        ),
        "mean_source_run_count": _mean(row.get("prime_rank_refinement_run_count") for row in usable),
        "mean_dimension_drift": _mean(dimension_drifts),
        "max_dimension_drift": max(dimension_drifts) if dimension_drifts else None,
        "patch_count_sets": _counts(row.get("prime_rank_refinement_patch_counts") for row in usable),
        "size_median_dimension_sets": _counts(
            row.get("prime_rank_refinement_size_median_dimensions") for row in usable
        ),
        "blocker_counts": _flattened_counts(row.get("prime_rank_refinement_proof_blockers") for row in usable),
        "physical_claim": False,
        "interpretation": (
            "Cross-regulator diagnostic for the control-quotient coordinate rank-3 window. A candidate "
            "receipt means the supplied rank-3/E3 control-quotient window is stable across patch counts "
            "and passes the S2-leakage gate. It is not a strict neutral 3D-bulk proof because the "
            "independent SVD rank selector, proper negative-control replacement, and directional H3 "
            "strict gates remain separate blockers."
        ),
    }


def _parent_collar_ladder_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("parent_collar_ladder_written")]
    return {
        "run_count": len(usable),
        "compiler_ready_count": sum(1 for row in usable if row.get("parent_collar_ladder_compiler_ready")),
        "regulator_ready_count": sum(1 for row in usable if row.get("parent_collar_ladder_regulator_ready")),
        "scaling_pass_count": sum(1 for row in usable if row.get("parent_collar_ladder_scaling_pass")),
        "local_density_receipt_count": sum(
            1 for row in usable if row.get("parent_collar_ladder_local_density_receipt")
        ),
        "strict_local_density_receipt_count": sum(
            1 for row in usable if row.get("parent_collar_ladder_strict_local_density_receipt")
        ),
        "theorem_grade_count": sum(1 for row in usable if row.get("parent_collar_ladder_theorem_grade")),
        "strict_cap_family_count": sum(
            1 for row in usable if row.get("parent_collar_ladder_strict_cap_family_matched")
        ),
        "unique_theta_family_count": sum(
            1 for row in usable if row.get("parent_collar_ladder_unique_theta_family_matched")
        ),
        "physical_cmb_prediction_count": sum(
            1 for row in usable if row.get("parent_collar_ladder_physical_cmb_prediction")
        ),
        "mean_report_count": _mean(row.get("parent_collar_ladder_report_count") for row in usable),
        "mean_row_count": _mean(row.get("parent_collar_ladder_row_count") for row in usable),
        "mean_patch_count_count": _mean(row.get("parent_collar_ladder_patch_count_count") for row in usable),
        "mean_log_cmi_slope": _mean(row.get("parent_collar_ladder_slope") for row in usable),
        "mean_log_cmi_density_slope": _mean(row.get("parent_collar_ladder_density_slope") for row in usable),
        "mean_final_p90_epsilon_cmi": _mean(
            row.get("parent_collar_ladder_final_p90_epsilon_cmi") for row in usable
        ),
        "mean_final_p90_epsilon_cmi_per_collar_patch": _mean(
            row.get("parent_collar_ladder_final_p90_epsilon_cmi_per_collar_patch") for row in usable
        ),
        "mean_final_p90_r_fr": _mean(row.get("parent_collar_ladder_final_p90_r_fr") for row in usable),
        "physical_cmb_prediction": False,
        "interpretation": (
            "Finite parent-collar recovery ladder diagnostic. This aggregates cached diagonal "
            "collar-Markov CMI/Fawzi-Renner proxy reports across regulator sizes. It is the right "
            "input lane for future finite CMB certificates, but it is not theorem-grade unless the "
            "recovery errors improve with refinement and pass conservative cold-limit thresholds."
        ),
    }


def _b_a_parent_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("b_a_parent_written")]
    return {
        "run_count": len(usable),
        "receipt_count": sum(1 for row in usable if row.get("b_a_parent_receipt")),
        "paired_diagnostic_receipt_count": sum(
            1 for row in usable if row.get("b_a_parent_paired_diagnostic_receipt")
        ),
        "physical_prediction_ready_count": sum(
            1 for row in usable if row.get("b_a_parent_physical_prediction_ready")
        ),
        "physical_cmb_prediction_count": sum(
            1 for row in usable if row.get("b_a_parent_physical_cmb_prediction")
        ),
        "controls_fail_count": sum(1 for row in usable if row.get("b_a_parent_controls_fail")),
        "real_baryon_perturbation_run_count": sum(
            1 for row in usable if row.get("b_a_parent_real_baryon_perturbation_runs_present")
        ),
        "finite_observer_view_parent_variation_count": sum(
            1 for row in usable if row.get("b_a_parent_finite_observer_view_parent_variation")
        ),
        "refinement_convergence_passed_count": sum(
            1 for row in usable if row.get("b_a_parent_refinement_convergence_passed")
        ),
        "primary_parent_source_counts": _counts(row.get("b_a_parent_primary_parent_source") for row in usable),
        "mean_source_report_count": _mean(row.get("b_a_parent_source_report_count") for row in usable),
        "mean_observer_view_source_count": _mean(
            row.get("b_a_parent_observer_view_source_count") for row in usable
        ),
        "mean_row_count": _mean(row.get("b_a_parent_row_count") for row in usable),
        "mean_control_row_count": _mean(row.get("b_a_parent_control_row_count") for row in usable),
        "mean_observer_view_row_count": _mean(
            row.get("b_a_parent_observer_view_row_count") for row in usable
        ),
        "control_failure_counts": _dict_true_counts(
            row.get("b_a_parent_control_failures") for row in usable
        ),
        "missing_gate_counts": _list_counts(row.get("b_a_parent_missing_gates") for row in usable),
        "physical_cmb_prediction": False,
        "interpretation": (
            "Finite-difference B_A(k,a) parent diagnostic from observer-visible cap/collar packet "
            "variation. The paired diagnostic receipt means real plus/minus perturb-resettle reruns, "
            "sign stability, and expected control failures are present. It is still not a physical "
            "Boltzmann kernel unless k/a units are calibrated, energy exchange is closed, and "
            "refinement passes."
        ),
    }


def _screen_capacity_closure_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("screen_capacity_written")]
    return {
        "run_count": len(usable),
        "observed_readout_count": sum(1 for row in usable if row.get("screen_capacity_observed_readout_gate")),
        "F_N_implemented_count": sum(1 for row in usable if row.get("screen_capacity_F_N_implemented")),
        "fixed_point_solved_count": sum(1 for row in usable if row.get("screen_capacity_fixed_point_solved")),
        "physical_cmb_prediction_count": sum(
            1 for row in usable if row.get("screen_capacity_physical_cmb_prediction")
        ),
        "input_mode_counts": _counts(row.get("screen_capacity_input_mode") for row in usable),
        "mean_N_patch_bare_ratio": _mean(row.get("screen_capacity_N_patch_bare_ratio") for row in usable),
        "mean_N_scr": _mean(row.get("screen_capacity_N_scr") for row in usable),
        "mean_Lambda_lP2": _mean(row.get("screen_capacity_Lambda_lP2") for row in usable),
        "mean_P_cell_count": _mean(row.get("screen_capacity_P_cell_count") for row in usable),
        "physical_cmb_prediction": False,
        "interpretation": (
            "Global cosmic record/screen-capacity closure lane. It gives the observed-branch "
            "de Sitter entropy-capacity normalization and Lambda*l_P^2 readout, while keeping ordinary "
            "finite patch counts as numerical regulators until the OPH readback map F(N) is implemented."
        ),
    }


def _repair_scale_closure_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("repair_scale_written")]
    return {
        "run_count": len(usable),
        "local_P_available_count": sum(1 for row in usable if row.get("repair_scale_local_P_available")),
        "N_CRC_declared_count": sum(1 for row in usable if row.get("repair_scale_N_CRC_declared")),
        "numeric_match_within_1_percent_count": sum(
            1 for row in usable if row.get("repair_scale_numeric_match_1pct")
        ),
        "twenty_four_rounds_declared_count": sum(
            1 for row in usable if row.get("repair_scale_24_rounds_declared")
        ),
        "twenty_four_rounds_derived_count": sum(
            1 for row in usable if row.get("repair_scale_24_rounds_derived")
        ),
        "finite_lattice_derived_eta_R_count": sum(
            1 for row in usable if row.get("repair_scale_finite_lattice_derived_eta_R")
        ),
        "physical_cmb_prediction_count": sum(
            1 for row in usable if row.get("repair_scale_physical_cmb_prediction")
        ),
        "bulk_3d_established_count": sum(1 for row in usable if row.get("repair_scale_bulk_3d_established")),
        "mean_rounds": _mean(row.get("repair_scale_rounds") for row in usable),
        "mean_capacity_exponent": _mean(row.get("repair_scale_capacity_exponent") for row in usable),
        "mean_gprime": _mean(row.get("repair_scale_gprime") for row in usable),
        "mean_q_round": _mean(row.get("repair_scale_q_round") for row in usable),
        "mean_length_ratio": _mean(row.get("repair_scale_length_ratio") for row in usable),
        "mean_N_implied_by_ansatz": _mean(row.get("repair_scale_N_implied_by_ansatz") for row in usable),
        "mean_N_pred": _mean(row.get("repair_scale_N_pred") for row in usable),
        "mean_N_CRC": _mean(row.get("repair_scale_N_CRC") for row in usable),
        "mean_relative_error": _mean(row.get("repair_scale_rel_error") for row in usable),
        "mean_eta_R": _mean(row.get("repair_scale_eta_R") for row in usable),
        "mean_n_s": _mean(row.get("repair_scale_n_s") for row in usable),
        "mean_1m_effective_round_depth": _mean(row.get("repair_scale_1m_depth_rounds") for row in usable),
        "physical_cmb_prediction": False,
        "interpretation": (
            "Maarten/OPH 24-round scale-closure hypothesis. This lane predicts a local repair contraction "
            "|g'(P)|=alpha/phi^2, a capacity implied by the declared repair-depth ansatz, and n_s=1-P/48. "
            "It explains finite-regulator shallow depth, but does not derive N from P alone, does not set a "
            "dimensionful SI scale, does not derive the 24 rounds from a finite selector, and does not "
            "establish populated 3D bulk or physical CMB."
        ),
    }


def _scale_compressed_repair_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("scale_compressed_written")]
    return {
        "run_count": len(usable),
        "operator_receipt_count": sum(1 for row in usable if row.get("scale_compressed_operator_receipt")),
        "round_trace_receipt_count": sum(1 for row in usable if row.get("scale_compressed_round_trace_receipt")),
        "cap_profile_receipt_count": sum(1 for row in usable if row.get("scale_compressed_cap_profile_receipt")),
        "populated_h3_preview_count": sum(1 for row in usable if row.get("scale_compressed_populated_h3_preview")),
        "strict_neutral_bulk_count": sum(1 for row in usable if row.get("scale_compressed_strict_neutral_bulk")),
        "particle_preview_count": sum(1 for row in usable if row.get("scale_compressed_particle_preview")),
        "production_particle_matter_count": sum(
            1 for row in usable if row.get("scale_compressed_production_particle_matter")
        ),
        "physical_cmb_prediction_count": sum(
            1 for row in usable if row.get("scale_compressed_physical_cmb_prediction")
        ),
        "mean_rounds": _mean(row.get("scale_compressed_rounds") for row in usable),
        "mean_eta_R": _mean(row.get("scale_compressed_eta_R") for row in usable),
        "mean_n_s": _mean(row.get("scale_compressed_n_s") for row in usable),
        "mean_q_IR": _mean(row.get("scale_compressed_q_IR") for row in usable),
        "mean_ell_IR": _mean(row.get("scale_compressed_ell_IR") for row in usable),
        "mean_N_CRC_implied_by_ansatz": _mean(
            row.get("scale_compressed_N_CRC_implied_by_ansatz") for row in usable
        ),
        "mean_N_CRC_pred": _mean(row.get("scale_compressed_N_CRC_pred") for row in usable),
        "mean_N_CRC_declared": _mean(row.get("scale_compressed_N_CRC_declared") for row in usable),
        "mean_relative_error": _mean(row.get("scale_compressed_rel_error") for row in usable),
        "mean_object_count": _mean(row.get("scale_compressed_object_count") for row in usable),
        "mean_particle_worldline_count": _mean(row.get("scale_compressed_particle_count") for row in usable),
        "mean_cl_ell_max": _mean(row.get("scale_compressed_cl_ell_max") for row in usable),
        "physical_cmb_prediction": False,
        "interpretation": (
            "Logical 24-round scale-compressed repair branch. This lane makes the repair-depth hypothesis "
            "computable now: it emits a round trace, OPH CMB scalar parameters, a screen C_l scaffold, "
            "and a populated H3 preview. It is not a literal finite-patch proof, not strict neutral bulk, "
            "and not a physical CMB prediction."
        ),
    }


def _scale_compressed_cmb_camb_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("scale_compressed_cmb_camb_written")]
    return {
        "run_count": len(usable),
        "measurement_comparable_curve_count": sum(
            1 for row in usable if row.get("scale_compressed_cmb_camb_curve_comparable")
        ),
        "transfer_receipt_count": sum(
            1 for row in usable if row.get("scale_compressed_cmb_camb_transfer_receipt")
        ),
        "physical_cmb_prediction_count": sum(
            1 for row in usable if row.get("scale_compressed_cmb_camb_physical_cmb_prediction")
        ),
        "operator_receipt_count": sum(
            1 for row in usable if row.get("scale_compressed_cmb_camb_operator_receipt")
        ),
        "h3_preview_receipt_count": sum(
            1 for row in usable if row.get("scale_compressed_cmb_camb_h3_preview_receipt")
        ),
        "mean_rounds": _mean(row.get("scale_compressed_cmb_camb_rounds") for row in usable),
        "mean_eta_R": _mean(row.get("scale_compressed_cmb_camb_eta_R") for row in usable),
        "mean_n_s": _mean(row.get("scale_compressed_cmb_camb_n_s") for row in usable),
        "mean_q_IR": _mean(row.get("scale_compressed_cmb_camb_q_IR") for row in usable),
        "mean_ell_IR": _mean(row.get("scale_compressed_cmb_camb_ell_IR") for row in usable),
        "mean_lcdm_shape_correlation": _mean(
            row.get("scale_compressed_cmb_camb_lcdm_shape_correlation") for row in usable
        ),
        "mean_scalar_shape_correlation": _mean(
            row.get("scale_compressed_cmb_camb_scalar_shape_correlation") for row in usable
        ),
        "mean_ir_shape_correlation": _mean(
            row.get("scale_compressed_cmb_camb_ir_shape_correlation") for row in usable
        ),
        "mean_lcdm_chi2_per_bin": _mean(
            row.get("scale_compressed_cmb_camb_lcdm_chi2_per_bin") for row in usable
        ),
        "mean_scalar_chi2_per_bin": _mean(
            row.get("scale_compressed_cmb_camb_scalar_chi2_per_bin") for row in usable
        ),
        "mean_ir_chi2_per_bin": _mean(
            row.get("scale_compressed_cmb_camb_ir_chi2_per_bin") for row in usable
        ),
        "mean_acoustic_abs_delta": _mean(
            row.get("scale_compressed_cmb_camb_acoustic_mean_abs_delta") for row in usable
        ),
        "physical_cmb_prediction": False,
        "interpretation": (
            "CAMB TT transfer for the logical 24-round scale-compressed repair branch. This is the "
            "current closest measurement-comparable CMB artifact tied to the repair-depth simulator "
            "readouts. It remains non-physical until the 24-round selector, amplitude, anomaly-sector, "
            "and official likelihood/map-space gates are closed."
        ),
    }


def _cmb_derivation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("cmb_derivation_written")]
    return {
        "run_count": len(usable),
        "ready_count": sum(1 for row in usable if row.get("cmb_derivation_ready")),
        "mean_source_run_count": _mean(row.get("cmb_derivation_run_count") for row in usable),
        "mean_eta_R": _mean(row.get("cmb_derivation_mean_eta_R") for row in usable),
        "mean_eta_R_abs_error": _mean(row.get("cmb_derivation_mean_eta_R_abs_error") for row in usable),
        "mean_q_IR": _mean(row.get("cmb_derivation_mean_q_IR") for row in usable),
        "mean_ell_IR": _mean(row.get("cmb_derivation_mean_ell_IR") for row in usable),
        "mean_scalar_quotient_theta_OPH": _mean(
            row.get("cmb_derivation_mean_scalar_quotient_theta_OPH") for row in usable
        ),
        "mean_scalar_quotient_n_s": _mean(
            row.get("cmb_derivation_mean_scalar_quotient_n_s") for row in usable
        ),
        "mean_scalar_quotient_n_s_abs_error": _mean(
            row.get("cmb_derivation_mean_scalar_quotient_n_s_abs_error") for row in usable
        ),
        "mean_scalar_quotient_observer_count": _mean(
            row.get("cmb_derivation_mean_scalar_quotient_observer_count") for row in usable
        ),
        "mean_median_epsilon_cmi": _mean(
            row.get("cmb_derivation_mean_median_epsilon_cmi") for row in usable
        ),
        "physical_cmb_prediction": False,
        "interpretation": (
            "Audit of whether current finite lattice reports derive the CMB target parameters from "
            "cap/collar/screen receipts. A ready count of zero means comparable target numbers exist, "
            "but the finite simulator has not yet earned a physical CMB prediction claim."
        ),
    }


def _oph_boltzmann_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("oph_boltzmann_input_written")]
    return {
        "run_count": len(usable),
        "cdm_limit_ready_count": sum(1 for row in usable if row.get("oph_boltzmann_cdm_limit_ready")),
        "diagnostic_repair_table_ready_count": sum(
            1 for row in usable if row.get("oph_boltzmann_diagnostic_ready")
        ),
        "physical_prediction_ready_count": sum(
            1 for row in usable if row.get("oph_boltzmann_physical_prediction_ready")
        ),
        "mean_source_report_count": _mean(row.get("oph_boltzmann_source_report_count") for row in usable),
        "mean_cdm_row_count": _mean(row.get("oph_boltzmann_cdm_row_count") for row in usable),
        "mean_diagnostic_row_count": _mean(row.get("oph_boltzmann_diagnostic_row_count") for row in usable),
        "mean_missing_gate_count": _mean(row.get("oph_boltzmann_missing_gate_count") for row in usable),
        "mean_gamma_rec_over_H_shape_proxy": _mean(row.get("oph_boltzmann_mean_gamma_proxy") for row in usable),
        "mean_B_A_shape_proxy": _mean(row.get("oph_boltzmann_mean_B_A_shape_proxy") for row in usable),
        "physical_cmb_prediction": False,
        "interpretation": (
            "Machine-readable bridge from OPH-CMB diagnostics to future CAMB/CLASS inputs. The CDM-limit "
            "rows are solver-ready only as an external LambdaCDM regression target. The repair-exchange rows "
            "are finite-collar proxy shapes and remain gated against physical prediction language."
        ),
    }


def _finite_collar_boltzmann_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("finite_collar_boltzmann_written")]
    return {
        "run_count": len(usable),
        "diagnostic_bundle_receipt_count": sum(
            1 for row in usable if row.get("finite_collar_boltzmann_bundle_receipt")
        ),
        "physical_certificate_count": sum(
            1 for row in usable if row.get("finite_collar_boltzmann_physical_certificate")
        ),
        "physical_cmb_prediction_count": sum(
            1 for row in usable if row.get("finite_collar_boltzmann_physical_cmb_prediction")
        ),
        "no_data_use_count": sum(1 for row in usable if row.get("finite_collar_boltzmann_no_data")),
        "mean_B_A_row_count": _mean(row.get("finite_collar_boltzmann_b_a_row_count") for row in usable),
        "mean_rho_A_row_count": _mean(row.get("finite_collar_boltzmann_rho_row_count") for row in usable),
        "mean_Gamma_rec_row_count": _mean(row.get("finite_collar_boltzmann_gamma_row_count") for row in usable),
        "mean_physical_missing_gate_count": _mean(
            row.get("finite_collar_boltzmann_physical_missing_gate_count") for row in usable
        ),
        "mean_abs_B_A": _mean(row.get("finite_collar_boltzmann_mean_B_A") for row in usable),
        "mean_abs_Gamma_rec_over_H": _mean(row.get("finite_collar_boltzmann_mean_Gamma_rec") for row in usable),
        "diagnostic_missing_gate_counts": _list_counts(
            row.get("finite_collar_boltzmann_diagnostic_missing_gates") for row in usable
        ),
        "physical_missing_gate_counts": _list_counts(
            row.get("finite_collar_boltzmann_physical_missing_gates") for row in usable
        ),
        "physical_cmb_prediction": False,
        "interpretation": (
            "Consolidated finite-collar source bundle for the OPH Boltzmann/CMB bridge. A diagnostic "
            "receipt means rho_A(a), rho_A,eq(a), B_A(k,a), and Gamma_rec(k,a) source-side tables exist "
            "behind a no-data firewall. A physical certificate still requires calibrated k/a units, "
            "energy-exchange closure, refinement convergence, strict freezeout/bulk gates, CDM-limit "
            "regression, and an official likelihood-ready path."
        ),
    }


def _finite_collar_projection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("finite_collar_projection_written")]
    return {
        "run_count": len(usable),
        "projection_receipt_count": sum(1 for row in usable if row.get("finite_collar_projection_receipt")),
        "physical_k_receipt_count": sum(
            1 for row in usable if row.get("finite_collar_projection_physical_k_receipt")
        ),
        "physical_cmb_prediction_count": sum(
            1 for row in usable if row.get("finite_collar_projection_physical_cmb_prediction")
        ),
        "mean_projected_B_A_row_count": _mean(row.get("finite_collar_projection_row_count") for row in usable),
        "mean_background_row_count": _mean(row.get("finite_collar_projection_background_row_count") for row in usable),
        "mean_ell_min": _mean(row.get("finite_collar_projection_ell_min") for row in usable),
        "mean_ell_max": _mean(row.get("finite_collar_projection_ell_max") for row in usable),
        "mean_B_A": _mean(row.get("finite_collar_projection_mean_B_A") for row in usable),
        "mean_abs_B_A": _mean(row.get("finite_collar_projection_mean_abs_B_A") for row in usable),
        "mean_positive_fraction": _mean(
            row.get("finite_collar_projection_positive_fraction") for row in usable
        ),
        "mean_log_abs_B_A_vs_log_ell_slope": _mean(
            row.get("finite_collar_projection_log_slope") for row in usable
        ),
        "mean_largest_scale_B_A": _mean(
            row.get("finite_collar_projection_largest_scale_B_A") for row in usable
        ),
        "mean_smallest_scale_B_A": _mean(
            row.get("finite_collar_projection_smallest_scale_B_A") for row in usable
        ),
        "physical_cmb_prediction": False,
        "interpretation": (
            "External-fiducial ell/k projection of finite-collar B_A rows. This is useful for "
            "measurement-facing plots and Boltzmann plumbing, but it does not close the physical k "
            "calibration gate because chi_* and h are declared comparison geometry rather than derived "
            "from the finite OPH screen-to-bulk scale theorem."
        ),
    }


def _oph_cmb_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("oph_cmb_adapter_written")]
    return {
        "run_count": len(usable),
        "boltzmann_ready_count": sum(1 for row in usable if row.get("oph_cmb_boltzmann_ready")),
        "physical_prediction_ready_count": sum(
            1 for row in usable if row.get("oph_cmb_physical_prediction_ready")
        ),
        "cosmology_perturbation_receipt_count": sum(
            1 for row in usable if row.get("oph_cmb_cosmology_perturbation_receipt")
        ),
        "mean_parent_sample_count": _mean(row.get("oph_cmb_parent_sample_count") for row in usable),
        "mean_weighted_collar_repair_defect_R": _mean(row.get("oph_cmb_parent_R") for row in usable),
        "mean_rho_A_eq_proxy": _mean(row.get("oph_cmb_parent_rho_A_eq_proxy") for row in usable),
        "mean_kernel_proxy_row_count": _mean(row.get("oph_cmb_kernel_proxy_row_count") for row in usable),
        "mean_missing_gate_count": _mean(row.get("oph_cmb_missing_gate_count") for row in usable),
        "physical_cmb_prediction": False,
        "interpretation": (
            "OPH-CMB adapter lane from the CMB writeup. It preserves standard photon-baryon "
            "recombination physics and exposes the OPH anomaly sector quantities needed by a "
            "future CAMB/CLASS module. Current values are finite-collar parent diagnostics, not "
            "theorem-grade rho_A(a), Gamma_rec, or B_A(k,a), so the Boltzmann gate remains closed."
        ),
    }


def _cmb_anomaly_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("cmb_anomaly_written")]
    return {
        "run_count": len(usable),
        "primary_field_counts": _counts(row.get("cmb_anomaly_primary_field") for row in usable),
        "best_low_power_field_counts": _counts(row.get("cmb_anomaly_best_low_power_field") for row in usable),
        "best_large_angle_field_counts": _counts(row.get("cmb_anomaly_best_large_angle_field") for row in usable),
        "best_parity_field_counts": _counts(row.get("cmb_anomaly_best_parity_field") for row in usable),
        "best_tilt_field_counts": _counts(row.get("cmb_anomaly_best_tilt_field") for row in usable),
        "mean_best_low_power_abs_fraction": _mean(
            row.get("cmb_anomaly_best_low_power_abs_fraction") for row in usable
        ),
        "mean_best_S_1_2_scalar_proxy": _mean(row.get("cmb_anomaly_best_S_1_2_scalar_proxy") for row in usable),
        "mean_best_parity_log_abs_deviation": _mean(
            row.get("cmb_anomaly_best_parity_log_abs_deviation") for row in usable
        ),
        "mean_best_eta_R_estimate": _mean(row.get("cmb_anomaly_best_eta_R_estimate") for row in usable),
        "mean_best_n_s_proxy": _mean(row.get("cmb_anomaly_best_n_s_proxy") for row in usable),
        "mean_low_power_suppressed_vs_controls_count": _mean(
            row.get("cmb_anomaly_low_power_suppressed_vs_controls_count") for row in usable
        ),
        "mean_large_angle_suppressed_vs_controls_count": _mean(
            row.get("cmb_anomaly_large_angle_suppressed_vs_controls_count") for row in usable
        ),
        "mean_parity_more_asymmetric_than_controls_count": _mean(
            row.get("cmb_anomaly_parity_more_asymmetric_than_controls_count") for row in usable
        ),
        "mean_planck_tilt_compatible_proxy_count": _mean(
            row.get("cmb_anomaly_planck_tilt_compatible_proxy_count") for row in usable
        ),
        "mean_total_entropy_capacity": _mean(row.get("cmb_anomaly_total_entropy_capacity") for row in usable),
        "mean_ell_sqrt_capacity_proxy": _mean(row.get("cmb_anomaly_ell_sqrt_capacity_proxy") for row in usable),
        "physical_cmb_prediction_count": sum(
            1 for row in usable if row.get("cmb_anomaly_physical_cmb_prediction")
        ),
        "interpretation": (
            "Finite-screen anomaly readout from actual C_l receipts. It targets the OPH CMB bridge questions "
            "about scale tilt, low multipoles, odd/even parity, large-angle correlations, and finite screen "
            "capacity. It is screen-only: no photon-baryon transfer, no official Planck likelihood, and no "
            "3D-bulk proof."
        ),
    }


def _sync_gap_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("sync_gap_written")]
    return {
        "run_count": len(usable),
        "source_run_count": _mean(row.get("sync_gap_source_run_count") for row in usable),
        "low_k_gap_established_count": sum(1 for row in usable if row.get("sync_gap_low_k_established")),
        "same_boundary_selector_count": sum(
            1 for row in usable if row.get("sync_gap_same_boundary_selector_established")
        ),
        "inflation_replacement_ready_count": sum(
            1 for row in usable if row.get("sync_gap_inflation_replacement_ready")
        ),
        "mean_time_resolved_trace_count": _mean(
            row.get("sync_gap_time_resolved_trace_count") for row in usable
        ),
        "mean_cached_proxy_pass_count": _mean(row.get("sync_gap_cached_proxy_pass_count") for row in usable),
        "mean_time_resolved_gap_pass_count": _mean(
            row.get("sync_gap_time_resolved_gap_pass_count") for row in usable
        ),
        "residual_field_candidate_receipt_count": sum(
            1 for row in usable if row.get("sync_gap_residual_candidate_receipt")
        ),
        "residual_field_counts": _counts(
            row.get("sync_gap_residual_candidate_field") for row in usable
        ),
        "mean_residual_field_positive_gamma_fraction": _mean(
            row.get("sync_gap_residual_candidate_positive_gamma_fraction") for row in usable
        ),
        "mean_residual_field_control_separation_fraction": _mean(
            row.get("sync_gap_residual_candidate_control_separation_fraction") for row in usable
        ),
        "mean_residual_field_median_gamma": _mean(
            row.get("sync_gap_residual_candidate_median_gamma") for row in usable
        ),
        "mean_residual_field_median_control_gap": _mean(
            row.get("sync_gap_residual_candidate_median_control_gap") for row in usable
        ),
        "mean_global_phi_gamma_per_cycle": _mean(row.get("sync_gap_mean_global_phi_gamma") for row in usable),
        "median_global_phi_gamma_per_cycle": _mean(
            row.get("sync_gap_median_global_phi_gamma") for row in usable
        ),
        "interpretation": (
            "Low-k synchronization-gap audit for OPH inflation-replacement questions. Cached final spectra "
            "and global Phi decay are useful diagnostics. Residual-field candidates report majority-mode "
            "repair/mismatch decay, but a true horizon-coherence/gap claim requires the stricter all-mode "
            "time-resolved harmonic repair gate and controls."
        ),
    }


def _hot_release_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("hot_release_written")]
    return {
        "run_count": len(usable),
        "source_run_count": _mean(row.get("hot_release_source_run_count") for row in usable),
        "theorem_ready_report_count": sum(1 for row in usable if row.get("hot_release_theorem_ready")),
        "mean_mechanical_surface_count": _mean(
            row.get("hot_release_mechanical_surface_count") for row in usable
        ),
        "mean_collar_gate_pass_count": _mean(row.get("hot_release_collar_gate_pass_count") for row in usable),
        "mean_theorem_ready_count": _mean(row.get("hot_release_theorem_ready_count") for row in usable),
        "mean_median_release_cycle": _mean(row.get("hot_release_median_release_cycle") for row in usable),
        "mean_release_cycle": _mean(row.get("hot_release_mean_release_cycle") for row in usable),
        "mean_median_epsilon_cmi": _mean(row.get("hot_release_mean_median_epsilon_cmi") for row in usable),
        "interpretation": (
            "Hot-MaxEnt release audit. A mechanical release surface means repair and record commitment "
            "settle together; it becomes theorem-ready only when collar Markov/recovery errors and the "
            "physical unit/charge map also pass."
        ),
    }


def _adiabaticity_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("adiabaticity_written")]
    return {
        "run_count": len(usable),
        "source_run_count": _mean(row.get("adiabaticity_source_run_count") for row in usable),
        "established_report_count": sum(1 for row in usable if row.get("adiabaticity_established")),
        "mean_proxy_pass_count": _mean(row.get("adiabaticity_proxy_pass_count") for row in usable),
        "mean_max_entropy_residual_std": _mean(
            row.get("adiabaticity_mean_max_entropy_residual_std") for row in usable
        ),
        "mean_min_common_clock_corr": _mean(
            row.get("adiabaticity_mean_min_common_clock_corr") for row in usable
        ),
        "interpretation": (
            "Same-boundary adiabaticity/isocurvature proxy. Current channels are observer-visible record "
            "fields, not physical photon/baryon/neutrino fluids, so this lane remains a gate audit until "
            "the Boltzmann species map exists."
        ),
    }


def _h0s8_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("h0s8_written")]
    return {
        "run_count": len(usable),
        "physical_prediction_ready_count": sum(1 for row in usable if row.get("h0s8_physical_prediction_ready")),
        "physical_cmb_prediction_count": sum(1 for row in usable if row.get("h0s8_physical_cmb_prediction")),
        "physical_matter_power_prediction_count": sum(
            1 for row in usable if row.get("h0s8_physical_matter_power_prediction")
        ),
        "Q_A_gate_count": sum(1 for row in usable if row.get("h0s8_Q_A_gate")),
        "B_A_gate_count": sum(1 for row in usable if row.get("h0s8_B_A_gate")),
        "lambda_P_gate_count": sum(1 for row in usable if row.get("h0s8_lambda_P_gate")),
        "Gamma_J_gate_count": sum(1 for row in usable if row.get("h0s8_Gamma_J_gate")),
        "CAMB_CLASS_gate_count": sum(1 for row in usable if row.get("h0s8_camb_class_gate")),
        "likelihood_gate_count": sum(1 for row in usable if row.get("h0s8_likelihood_gate")),
        "lane8_certificate_count": sum(1 for row in usable if row.get("h0s8_lane8_written")),
        "lane8_run_derived_count": sum(1 for row in usable if row.get("h0s8_lane8_values_are_run_derived")),
        "lane8_certificate_ready_count": sum(1 for row in usable if row.get("h0s8_lane8_certificate_ready")),
        "lane8_payload_gate_count": sum(1 for row in usable if row.get("h0s8_lane8_payload_gate")),
        "lane8_fake_suppression_gate_count": sum(
            1 for row in usable if row.get("h0s8_lane8_fake_suppression_gate")
        ),
        "lane8_selector_gate_count": sum(1 for row in usable if row.get("h0s8_lane8_selector_gate")),
        "lane8_refinement_gate_count": sum(1 for row in usable if row.get("h0s8_lane8_refinement_gate")),
        "mean_H0_km_s_Mpc": _mean(row.get("h0s8_H0_km_s_Mpc") for row in usable),
        "mean_Omega_m": _mean(row.get("h0s8_Omega_m") for row in usable),
        "mean_Omega_A": _mean(row.get("h0s8_Omega_A") for row in usable),
        "mean_Omega_Lambda": _mean(row.get("h0s8_Omega_Lambda") for row in usable),
        "mean_flat_sum": _mean(row.get("h0s8_flat_sum") for row in usable),
        "mean_lambda_collar": _mean(row.get("h0s8_lambda_collar") for row in usable),
        "mean_f_A": _mean(row.get("h0s8_f_A") for row in usable),
        "mean_mu_eff_source_suppression": _mean(
            row.get("h0s8_mu_eff_source_suppression") for row in usable
        ),
        "mean_source_suppression_fraction": _mean(
            row.get("h0s8_source_suppression_fraction") for row in usable
        ),
        "mean_cdm_like_S8": _mean(row.get("h0s8_cdm_like_S8") for row in usable),
        "mean_cdm_like_sigma8": _mean(row.get("h0s8_cdm_like_sigma8") for row in usable),
        "mean_direct_jacobi_S8": _mean(row.get("h0s8_direct_jacobi_S8") for row in usable),
        "mean_direct_jacobi_growth": _mean(row.get("h0s8_direct_jacobi_growth") for row in usable),
        "mean_matrix_gap_S8": _mean(row.get("h0s8_matrix_gap_S8") for row in usable),
        "mean_matrix_gap_growth": _mean(row.get("h0s8_matrix_gap_growth") for row in usable),
        "mean_planck_H0_pull_sigma": _mean(row.get("h0s8_planck_H0_pull_sigma") for row in usable),
        "mean_shoes_H0_pull_sigma": _mean(row.get("h0s8_shoes_H0_pull_sigma") for row in usable),
        "mean_planck_S8_cdm_pull_sigma": _mean(row.get("h0s8_planck_S8_cdm_pull_sigma") for row in usable),
        "mean_weak_lensing_cdm_pull_sigma": _mean(
            row.get("h0s8_weak_lensing_cdm_pull_sigma") for row in usable
        ),
        "mean_direct_jacobi_weak_lensing_pull_sigma": _mean(
            row.get("h0s8_direct_jacobi_weak_lensing_pull_sigma") for row in usable
        ),
        "mean_lane8_i0_bits": _mean(row.get("h0s8_lane8_i0_bits") for row in usable),
        "mean_lane8_low_entropy_gap_bits": _mean(
            row.get("h0s8_lane8_low_entropy_gap_bits") for row in usable
        ),
        "mean_lane8_fake_gamma_margin_bits": _mean(
            row.get("h0s8_lane8_fake_gamma_margin_bits") for row in usable
        ),
        "mean_lane8_fake_probability_bound": _mean(
            row.get("h0s8_lane8_fake_probability_bound") for row in usable
        ),
        "interpretation": (
            "H0/S8 branch diagnostic from the OPH cosmology notes. The numbers are measurement-facing "
            "branch consequences. Lane-8 certificate fields expose record/provenance closure gates, but "
            "they become finite-lattice predictions only when Q_A, B_A(k,a), the repair clock, "
            "CAMB/CLASS anomaly module, likelihood gates, and run-derived Lane-8 certificate values are closed."
        ),
    }


def _shape_substrate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("shape_substrate_report_written")]
    return {
        "run_count": len(usable),
        "vertex_scattering_receipt_count": sum(1 for row in usable if row.get("shape_vertex_scattering_receipt")),
        "dodeca_cell_receipt_count": sum(1 for row in usable if row.get("shape_dodeca_cell_receipt")),
        "shape_settling_receipt_count": sum(1 for row in usable if row.get("shape_settling_receipt")),
        "loop_particle_receipt_count": sum(1 for row in usable if row.get("shape_loop_particle_receipt")),
        "screen_projection_receipt_count": sum(1 for row in usable if row.get("shape_screen_projection_receipt")),
        "selector_elimination_target_input_count": sum(
            1 for row in usable if row.get("shape_selector_elimination_target_input_receipt")
        ),
        "cmb_certificate_input_count": sum(1 for row in usable if row.get("shape_cmb_certificate_input_receipt")),
        "declared_3d_substrate_count": sum(1 for row in usable if row.get("shape_declared_3d_substrate")),
        "neutral_oph_bulk_claim_count": sum(1 for row in usable if row.get("shape_neutral_oph_bulk_claim")),
        "physical_cmb_prediction_count": sum(1 for row in usable if row.get("shape_physical_cmb_prediction")),
        "mean_q_IR_candidate": _mean(row.get("shape_q_IR_candidate") for row in usable),
        "mean_q_IR_runtime_zero_mode": _mean(row.get("shape_q_IR_runtime_zero_mode") for row in usable),
        "mean_ell_IR_candidate": _mean(row.get("shape_ell_IR_candidate") for row in usable),
        "mean_ell_IR_runtime_covariance_rank": _mean(
            row.get("shape_ell_IR_runtime_covariance_rank") for row in usable
        ),
        "mean_eta_R_candidate": _mean(row.get("shape_eta_R_candidate") for row in usable),
        "mean_phi_drop_fraction": _mean(row.get("shape_phi_drop_fraction") for row in usable),
        "mean_loop_particle_count": _mean(row.get("shape_loop_particle_count") for row in usable),
        "mean_planck_lite_shape_correlation": _mean(
            row.get("shape_planck_lite_shape_correlation") for row in usable
        ),
        "mean_planck_lite_normalized_rmse": _mean(
            row.get("shape_planck_lite_normalized_rmse") for row in usable
        ),
        "physical_cmb_prediction": False,
        "claim_boundary": "Declared Alex/Shape substrate witness. Not neutral OPH bulk proof.",
        "interpretation": (
            "This lane tests declared dodecahedral three-way scattering, loop modes, loop particles, "
            "screen-projection diagnostics, and selector-elimination target inputs such as q_IR=1/4 and "
            "ell_IR=32. It is useful as a controlled substrate witness and C_l diagnostic source, but it must "
            "not be counted as neutral OPH bulk emergence or a physical CMB prediction."
        ),
    }


def _galaxy_proxy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("galaxy_proxy_written")]
    data_rows = [
        row
        for row in usable
        if row.get("galaxy_proxy_lambda_fit_usable")
        or row.get("galaxy_proxy_btfr_usable")
        or row.get("galaxy_proxy_disk_usable")
    ]
    return {
        "run_count": len(usable),
        "proxy_receipt_count": sum(1 for row in usable if row.get("galaxy_proxy_receipt")),
        "runs_with_external_rar_or_btfr_data": len(data_rows),
        "rar_lambda_fit_count": sum(1 for row in usable if row.get("galaxy_proxy_lambda_fit_usable")),
        "btfr_fit_count": sum(1 for row in usable if row.get("galaxy_proxy_btfr_usable")),
        "disk_residual_count": sum(1 for row in usable if row.get("galaxy_proxy_disk_usable")),
        "mean_a0_oph": _mean(row.get("galaxy_proxy_a0_oph") for row in usable),
        "mean_a0_eff": _mean(row.get("galaxy_proxy_a0_eff") for row in usable),
        "mean_lambda_collar_declared": _mean(row.get("galaxy_proxy_lambda_collar_declared") for row in usable),
        "mean_fitted_lambda_collar": _mean(row.get("galaxy_proxy_lambda_fit") for row in usable),
        "mean_btfr_slope": _mean(row.get("galaxy_proxy_btfr_slope") for row in usable),
        "physical_galaxy_prediction": False,
        "interpretation": (
            "Static galaxy lane for OPH RAR/BTFR continuation diagnostics. Current run-bundle outputs are "
            "proxy surfaces unless external SPARC/RAR/BTFR rows are supplied, so this is measurement-facing "
            "scaffolding rather than a publication-grade galaxy likelihood."
        ),
    }


def _static_galaxy_measurement_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("static_galaxy_written")]
    return {
        "run_count": len(usable),
        "receipt_count": sum(1 for row in usable if row.get("static_galaxy_receipt")),
        "bridge_receipt_count": sum(1 for row in usable if row.get("static_galaxy_bridge_receipt")),
        "physical_claim_count": sum(1 for row in usable if row.get("static_galaxy_physical_claim")),
        "physical_cmb_claim_count": sum(1 for row in usable if row.get("static_galaxy_physical_cmb_claim")),
        "bulk_required_count": sum(1 for row in usable if row.get("static_galaxy_bulk_required")),
        "claim_tier_counts": _counts(
            row.get("static_galaxy_claim_tier") for row in usable if row.get("static_galaxy_claim_tier")
        ),
        "mean_dataset_row_count": _mean(row.get("static_galaxy_dataset_row_count") for row in usable),
        "mean_dataset_galaxy_count": _mean(row.get("static_galaxy_dataset_galaxy_count") for row in usable),
        "mean_galaxy_count": _mean(row.get("static_galaxy_galaxy_count") for row in usable),
        "mean_measurement_galaxy_count": _mean(
            row.get("static_galaxy_measurement_galaxy_count") for row in usable
        ),
        "mean_rar_galaxy_count": _mean(row.get("static_galaxy_rar_galaxy_count") for row in usable),
        "mean_rar_galaxy_support_count": _mean(
            row.get("static_galaxy_rar_galaxy_support_count") for row in usable
        ),
        "mean_rar_point_count": _mean(row.get("static_galaxy_rar_point_count") for row in usable),
        "mean_shared_a0": _mean(row.get("static_galaxy_shared_a0") for row in usable),
        "mean_shared_lambda_collar": _mean(row.get("static_galaxy_shared_lambda_collar") for row in usable),
        "mean_rar_scatter_dex": _mean(row.get("static_galaxy_rar_scatter_dex") for row in usable),
        "btfr_fit_count": sum(1 for row in usable if row.get("static_galaxy_btfr_usable")),
        "mean_btfr_galaxy_count": _mean(row.get("static_galaxy_btfr_galaxy_count") for row in usable),
        "mean_btfr_slope": _mean(row.get("static_galaxy_btfr_slope") for row in usable),
        "mean_btfr_intercept": _mean(row.get("static_galaxy_btfr_intercept") for row in usable),
        "mean_btfr_predicted_slope": _mean(row.get("static_galaxy_btfr_predicted_slope") for row in usable),
        "mean_btfr_predicted_intercept": _mean(
            row.get("static_galaxy_btfr_predicted_intercept") for row in usable
        ),
        "mean_btfr_slope_delta": _mean(row.get("static_galaxy_btfr_slope_delta") for row in usable),
        "mean_btfr_intercept_delta": _mean(row.get("static_galaxy_btfr_intercept_delta") for row in usable),
        "mean_btfr_abs_slope_delta": _mean(row.get("static_galaxy_btfr_abs_slope_delta") for row in usable),
        "mean_btfr_abs_intercept_delta": _mean(
            row.get("static_galaxy_btfr_abs_intercept_delta") for row in usable
        ),
        "holdout_usable_count": sum(1 for row in usable if row.get("static_galaxy_holdout_usable")),
        "holdout_receipt_count": sum(1 for row in usable if row.get("static_galaxy_holdout_receipt")),
        "mean_holdout_train_galaxy_count": _mean(
            row.get("static_galaxy_holdout_train_galaxy_count") for row in usable
        ),
        "mean_holdout_test_galaxy_count": _mean(
            row.get("static_galaxy_holdout_test_galaxy_count") for row in usable
        ),
        "mean_holdout_train_point_count": _mean(
            row.get("static_galaxy_holdout_train_point_count") for row in usable
        ),
        "mean_holdout_test_point_count": _mean(
            row.get("static_galaxy_holdout_test_point_count") for row in usable
        ),
        "mean_holdout_shared_a0": _mean(row.get("static_galaxy_holdout_shared_a0") for row in usable),
        "mean_holdout_shared_lambda_collar": _mean(
            row.get("static_galaxy_holdout_shared_lambda_collar") for row in usable
        ),
        "mean_holdout_train_log_accel_rmse": _mean(
            row.get("static_galaxy_holdout_train_log_accel_rmse") for row in usable
        ),
        "mean_holdout_test_log_accel_rmse": _mean(
            row.get("static_galaxy_holdout_test_log_accel_rmse") for row in usable
        ),
        "mean_holdout_test_velocity_rmse": _mean(
            row.get("static_galaxy_holdout_test_velocity_rmse") for row in usable
        ),
        "mean_holdout_test_baryon_velocity_rmse": _mean(
            row.get("static_galaxy_holdout_test_baryon_velocity_rmse") for row in usable
        ),
        "mean_holdout_test_velocity_improvement": _mean(
            row.get("static_galaxy_holdout_test_velocity_improvement") for row in usable
        ),
        "mean_holdout_test_chi2_proxy": _mean(
            row.get("static_galaxy_holdout_test_chi2_proxy") for row in usable
        ),
        "interpretation": (
            "External static-galaxy OPH-CET bridge lane. It calibrates the OPH continuation law on "
            "RAR acceleration rows with free a0/lambda_collar and reports the asymptotic BTFR implied "
            "by that fit when BTFR rows exist. The optional SPARC mass-model holdout splits by galaxy "
            "and fits only shared a0/lambda_collar with fixed M/L assumptions. It is measurement-facing "
            "static phenomenology; it does not require the populated-bulk gate and is not a physical "
            "CMB/P(k) prediction."
        ),
    }


def _h3_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("h3_rmse") is not None]
    receipt_count = sum(1 for row in usable if row.get("h3_receipt"))
    h3_rmse = _mean(row.get("h3_rmse") for row in usable)
    wrong_pi_rmse = _mean(row.get("wrong_pi_rmse") for row in usable)
    audit_rows = [row for row in usable if row.get("h3_wrong_scale_audit_eligible")]
    wrong_scale_win_fraction = _mean(row.get("h3_wrong_scale_win_fraction") for row in audit_rows)
    material_wrong_scale_win_fraction = _mean(
        row.get("h3_material_wrong_scale_win_fraction") for row in audit_rows
    )
    if not usable:
        interpretation = "No H3 modular-response diagnostics were found in the selected run set."
    elif receipt_count == 0:
        interpretation = (
            "H3 modular-response diagnostics are present, but no run passed the full receipt. "
            "The current failure mode can beat S2/shuffle/no-flow controls while wrong-scale "
            "controls still fit better, so this is not bulk evidence."
        )
    elif wrong_scale_win_fraction is not None and wrong_scale_win_fraction > 0.0:
        interpretation = (
            "Some H3 receipts are present, but strict feature-level wrong-scale audits still find "
            "cap/time/observable cells where a wrong normalization is competitive. Treat this as an "
            "unstable precursor until the worst-group audit clears."
        )
    elif wrong_pi_rmse is not None and h3_rmse is not None and wrong_pi_rmse < h3_rmse:
        interpretation = (
            "Some H3 receipts are present, but the aggregate wrong-pi control remains competitive. "
            "Treat this as an unstable precursor until wrong-normalization separation improves."
        )
    else:
        interpretation = (
            "H3 modular-response residuals pass at least some receipts and beat the aggregate wrong-pi "
            "control, but this is still an internal diagnostic rather than a physical bulk claim."
        )
    return {
        "run_count": len(usable),
        "receipt_count": receipt_count,
        "control_separation_receipt_count": sum(1 for row in usable if row.get("h3_control_separation_receipt")),
        "mean_h3_rmse": h3_rmse,
        "mean_h3_explained_variance": _mean(row.get("h3_explained_variance") for row in usable),
        "channel_modes": _count_values(row.get("h3_channel_mode") for row in usable),
        "profile_modes": _count_values(row.get("h3_profile_mode") for row in usable),
        "mean_h3_channel_count": _mean(row.get("h3_channel_count") for row in usable),
        "h3_chart_3d_window_count": sum(1 for row in usable if row.get("h3_chart_candidate_3d_dimension_window")),
        "h3_chart_dimension_agree_count": sum(1 for row in usable if row.get("h3_chart_dimension_estimators_agree")),
        "signal_gate_count": sum(1 for row in usable if row.get("h3_signal_gate")),
        "geometry_gate_count": sum(1 for row in usable if row.get("h3_geometry_gate")),
        "aggregate_wrong_scale_gate_count": sum(1 for row in usable if row.get("h3_aggregate_wrong_scale_gate")),
        "material_feature_gate_count": sum(1 for row in usable if row.get("h3_material_feature_gate")),
        "mean_h3_chart_correlation_dimension": _mean(
            row.get("h3_chart_correlation_dimension_estimate") for row in usable
        ),
        "mean_h3_chart_local_mle_dimension": _mean(row.get("h3_chart_local_mle_dimension_estimate") for row in usable),
        "mean_s2_boundary_rmse": _mean(row.get("s2_boundary_rmse") for row in usable),
        "mean_shuffled_response_rmse": _mean(row.get("shuffled_response_rmse") for row in usable),
        "mean_wrong_pi_rmse": wrong_pi_rmse,
        "wrong_scale_feature_audit_count": len(audit_rows),
        "mean_wrong_scale_feature_win_fraction": wrong_scale_win_fraction,
        "mean_material_wrong_scale_feature_win_fraction": material_wrong_scale_win_fraction,
        "mean_two_pi_h3_feature_win_fraction": _mean(row.get("h3_two_pi_fit_win_fraction") for row in audit_rows),
        "mean_material_two_pi_h3_feature_win_fraction": _mean(
            row.get("h3_material_two_pi_fit_win_fraction") for row in audit_rows
        ),
        "wrong_scale_red_flag_count": sum(1 for row in audit_rows if row.get("h3_wrong_scale_red_flag")),
        "material_wrong_scale_red_flag_count": sum(
            1 for row in audit_rows if row.get("h3_material_wrong_scale_red_flag")
        ),
        "worst_wrong_scale_groups": [
            row.get("h3_wrong_scale_worst_group")
            for row in audit_rows
            if row.get("h3_wrong_scale_worst_group")
        ][:8],
        "interpretation": interpretation,
    }


def _h3_seed_ensemble_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("h3_seed_ensemble_seed_count") is not None]
    return {
        "run_count": len(usable),
        "response_seed_robust_count": sum(1 for row in usable if row.get("h3_response_seed_robust_receipt")),
        "chart_3d_seed_robust_count": sum(1 for row in usable if row.get("h3_chart_3d_seed_robust_receipt")),
        "mean_seed_count": _mean(row.get("h3_seed_ensemble_seed_count") for row in usable),
        "mean_receipt_fraction": _mean(row.get("h3_seed_ensemble_receipt_fraction") for row in usable),
        "mean_control_separation_fraction": _mean(
            row.get("h3_seed_ensemble_control_separation_fraction") for row in usable
        ),
        "mean_candidate_receipt_fraction": _mean(
            row.get("h3_seed_ensemble_candidate_receipt_fraction") for row in usable
        ),
        "mean_dim3_fraction": _mean(row.get("h3_seed_ensemble_dim3_fraction") for row in usable),
        "mean_heldout_normalized_rmse": _mean(row.get("h3_seed_ensemble_mean_nrmse") for row in usable),
        "mean_heldout_explained_variance": _mean(row.get("h3_seed_ensemble_mean_ev") for row in usable),
        "mean_p75_material_wrong": _mean(row.get("h3_seed_ensemble_p75_material_wrong") for row in usable),
        "interpretation": (
            "Cached H3 seed-ensemble robustness. This prevents accepting a single lucky H3 candidate "
            "sample as bulk evidence. The response ensemble receipt uses aggregate control separation, "
            "median explained variance, and p75 material wrong-scale fraction; per-seed candidate receipts "
            "remain diagnostic because their single-seed gate is stricter. This is still not a populated "
            "bulk receipt."
        ),
    }


def _minimal_caps_to_h3_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("minimal_caps_to_h3_written")]
    return {
        "run_count": len(usable),
        "receipt_count": sum(1 for row in usable if row.get("minimal_caps_to_h3_receipt")),
        "mean_median_reconstruction_mse": _mean(
            row.get("minimal_caps_to_h3_median_reconstruction_mse") for row in usable
        ),
        "mean_median_shuffled_profile_mse": _mean(
            row.get("minimal_caps_to_h3_median_shuffled_profile_mse") for row in usable
        ),
        "mean_median_s2_boundary_profile_mse": _mean(
            row.get("minimal_caps_to_h3_median_s2_boundary_profile_mse") for row in usable
        ),
        "h3_beats_shuffled_count": sum(1 for row in usable if row.get("minimal_caps_to_h3_beats_shuffled")),
        "h3_beats_s2_boundary_count": sum(1 for row in usable if row.get("minimal_caps_to_h3_beats_s2_boundary")),
        "claim_boundary": "S2 cap-profile to H3 geometry calibration only; not a dynamics or populated-bulk claim.",
    }


def _observer_readout_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("observer_readout_pair_count") is not None]
    object_usable = [row for row in rows if row.get("observer_object_consensus_object_count") is not None]
    bad_rewrite_count = sum(1 for row in object_usable if row.get("observer_object_bad_record_rewrite_detected"))
    return {
        "run_count": len(usable),
        "mean_observer_count": _mean(row.get("observer_readout_observer_count") for row in usable),
        "mean_pair_count": _mean(row.get("observer_readout_pair_count") for row in usable),
        "mean_global_committed_fraction": _mean(
            row.get("observer_readout_global_committed_fraction") for row in usable
        ),
        "mean_median_overlap_jaccard": _mean(row.get("observer_readout_median_overlap_jaccard") for row in usable),
        "mean_median_signature_similarity": _mean(
            row.get("observer_readout_median_signature_similarity") for row in usable
        ),
        "mean_p10_signature_similarity": _mean(row.get("observer_readout_p10_signature_similarity") for row in usable),
        "object_consensus_run_count": len(object_usable),
        "mean_object_consensus_count": _mean(
            row.get("observer_object_consensus_object_count") for row in object_usable
        ),
        "mean_persistent_object_count": _mean(
            row.get("observer_object_consensus_persistent_object_count") for row in object_usable
        ),
        "mean_object_overlap_agreement": _mean(
            row.get("observer_object_consensus_median_overlap_agreement") for row in object_usable
        ),
        "mean_object_p10_overlap_agreement": _mean(
            row.get("observer_object_consensus_p10_overlap_agreement") for row in object_usable
        ),
        "mean_counterfactual_stability": _mean(
            row.get("observer_object_consensus_median_counterfactual_stability") for row in object_usable
        ),
        "bad_record_rewrite_detected_count": bad_rewrite_count,
        "interpretation": (
            "Observer-facing readout metrics report local objectivity across overlapping views. "
            "They are useful for OPH subjective-perspective diagnostics, but they are not a "
            "dimension estimator and do not establish a third-person 3D bulk."
        ),
    }


def _observer_chart_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("observer_chart_h3_compactness") is not None]
    h3_beats_count = sum(1 for row in usable if row.get("observer_chart_h3_beats_shuffled"))
    h3_beats_robust_count = sum(1 for row in usable if row.get("observer_chart_h3_beats_shuffled_robust"))
    boundary_count = sum(1 for row in usable if row.get("observer_chart_h3_not_boundary_dominated"))
    if h3_beats_robust_count:
        interpretation = (
            "At least some observer-facing object populations beat the robust shuffled observer-object envelope. "
            "A bulk claim still requires the object-chart receipt, H3 localization, non-boundary-dominated "
            "compactness, and repeated-seed controls."
        )
    elif h3_beats_count:
        interpretation = (
            "At least some observer-facing object populations beat median shuffled observer-object incidence, "
            "but not the robust shuffle envelope. This is only a weak precursor, not populated-bulk evidence."
        )
    else:
        interpretation = (
            "Observer-facing object populations are being emitted, but they do not yet beat shuffled "
            "observer-object incidence. This is a controlled negative for populated-bulk emergence."
        )
    return {
        "run_count": len(usable),
        "object_chart_receipt_count": sum(1 for row in usable if row.get("observer_chart_object_receipt")),
        "modular_response_h3_control_separation_receipt_count": sum(
            1 for row in usable if row.get("observer_chart_modular_response_h3_control_separation_receipt")
        ),
        "localized_object_precursor_receipt_count": sum(
            1 for row in usable if row.get("observer_chart_localized_object_precursor_receipt")
        ),
        "bulk_population_receipt_count": sum(1 for row in usable if row.get("observer_chart_bulk_population_receipt")),
        "localized_nonboundary_bulk_population_receipt_count": sum(
            1 for row in usable if row.get("observer_chart_localized_nonboundary_bulk_population_receipt")
        ),
        "object_chart_median_receipt_count": sum(1 for row in usable if row.get("observer_chart_object_median_receipt")),
        "mean_object_count": _mean(row.get("observer_chart_object_count") for row in usable),
        "mean_localized_object_count": _mean(row.get("observer_chart_localized_object_count") for row in usable),
        "mean_localized_not_boundary_object_count": _mean(
            row.get("observer_chart_localized_not_boundary_object_count") for row in usable
        ),
        "mean_shuffle_control_count": _mean(row.get("observer_chart_shuffle_control_count") for row in usable),
        "mean_shuffled_localized_object_count": _mean(
            row.get("observer_chart_shuffled_localized_object_count") for row in usable
        ),
        "mean_shuffled_localized_not_boundary_object_count": _mean(
            row.get("observer_chart_shuffled_localized_not_boundary_object_count") for row in usable
        ),
        "mean_shuffled_localized_not_boundary_object_p90": _mean(
            row.get("observer_chart_shuffled_localized_not_boundary_object_p90") for row in usable
        ),
        "mean_shuffled_localized_object_p90": _mean(
            row.get("observer_chart_shuffled_localized_object_p90") for row in usable
        ),
        "mean_h3_compactness": _mean(row.get("observer_chart_h3_compactness") for row in usable),
        "mean_s2_boundary_compactness": _mean(row.get("observer_chart_s2_compactness") for row in usable),
        "mean_shuffled_h3_compactness": _mean(row.get("observer_chart_shuffled_h3_compactness") for row in usable),
        "mean_p10_shuffled_h3_compactness": _mean(
            row.get("observer_chart_p10_shuffled_h3_compactness") for row in usable
        ),
        "mean_p90_shuffled_h3_compactness": _mean(
            row.get("observer_chart_p90_shuffled_h3_compactness") for row in usable
        ),
        "mean_h3_margin_vs_median_shuffle": _mean(
            row.get("observer_chart_h3_compactness_margin_vs_median_shuffle") for row in usable
        ),
        "mean_h3_margin_vs_p10_shuffle": _mean(
            row.get("observer_chart_h3_compactness_margin_vs_p10_shuffle") for row in usable
        ),
        "mean_h3_margin_vs_s2_boundary": _mean(
            row.get("observer_chart_h3_compactness_margin_vs_s2_boundary") for row in usable
        ),
        "h3_beats_shuffled_count": h3_beats_count,
        "h3_beats_shuffled_robust_count": h3_beats_robust_count,
        "h3_not_boundary_dominated_count": boundary_count,
        "compactness_distribution_control_receipt_count": sum(
            1 for row in usable if row.get("observer_chart_compactness_distribution_control_receipt")
        ),
        "compactness_distribution_population_receipt_count": sum(
            1 for row in usable if row.get("observer_chart_compactness_distribution_population_receipt")
        ),
        "localized_count_saturation_warning_count": sum(
            1 for row in usable if row.get("observer_chart_localized_count_saturation_warning")
        ),
        "localized_not_boundary_count_saturation_warning_count": sum(
            1 for row in usable if row.get("observer_chart_localized_not_boundary_count_saturation_warning")
        ),
        "interpretation": interpretation,
    }


def _neutral_reconstruction_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [
        row
        for row in rows
        if row.get("neutral_reconstruction_written")
        or row.get("strict_neutral_report_written")
        or row.get("strict_neutral_object_report_written")
    ]
    return {
        "run_count": len(usable),
        "legacy_neutral_reconstruction_count": sum(
            1 for row in usable if row.get("neutral_reconstruction_written")
        ),
        "strict_neutral_report_count": sum(1 for row in usable if row.get("strict_neutral_report_written")),
        "strict_neutral_bulk_count": sum(1 for row in usable if row.get("strict_neutral_bulk_receipt")),
        "strict_neutral_object_report_count": sum(
            1 for row in usable if row.get("strict_neutral_object_report_written")
        ),
        "strict_neutral_object_bulk_count": sum(
            1 for row in usable if row.get("strict_neutral_object_bulk_receipt")
        ),
        "strict_neutral_object_h3_selected_count": sum(
            1 for row in usable if row.get("strict_neutral_object_h3_selected")
        ),
        "strict_neutral_object_estimators_agree_3d_count": sum(
            1 for row in usable if row.get("strict_neutral_object_estimators_agree_3d")
        ),
        "mean_strict_neutral_object_count": _mean(
            row.get("strict_neutral_object_count") for row in usable
        ),
        "mean_strict_neutral_object_median_dimension": _mean(
            row.get("strict_neutral_object_median_dimension_estimate") for row in usable
        ),
        "strict_neutral_object_selected_model_counts": _counts(
            row.get("strict_neutral_object_selected_model") for row in usable
        ),
        "strict_neutral_object_blocker_counts": _flattened_counts(
            row.get("strict_neutral_object_blockers") for row in usable
        ),
        "strict_neutral_estimators_agree_3d_count": sum(
            1 for row in usable if row.get("strict_neutral_estimators_agree_3d")
        ),
        "strict_neutral_s2_leakage_pass_count": sum(
            1 for row in usable if row.get("strict_neutral_s2_leakage_pass")
        ),
        "strict_neutral_h3_selected_count": sum(
            1 for row in usable if row.get("strict_neutral_selected_model") == "H3"
        ),
        "strict_neutral_h3_best_count": sum(
            1 for row in usable if row.get("strict_neutral_best_model") == "H3"
        ),
        "mean_strict_neutral_observer_count": _mean(
            row.get("strict_neutral_observer_count") for row in usable
        ),
        "mean_strict_neutral_median_dimension": _mean(
            row.get("strict_neutral_median_dimension_estimate") for row in usable
        ),
        "mean_strict_neutral_correlation_dimension": _mean(
            row.get("strict_neutral_correlation_dimension_estimate") for row in usable
        ),
        "mean_strict_neutral_local_mle_dimension": _mean(
            row.get("strict_neutral_local_mle_dimension_estimate") for row in usable
        ),
        "mean_strict_neutral_s2_distance_correlation": _mean(
            row.get("strict_neutral_s2_distance_correlation") for row in usable
        ),
        "radial_depth_used_count": sum(1 for row in usable if row.get("neutral_radial_depth_used")),
        "control_gate_passed_count": sum(1 for row in usable if row.get("neutral_control_gate_passed")),
        "candidate_3d_window_count": sum(1 for row in usable if row.get("neutral_candidate_3d_dimension_window")),
        "bulk_3d_established_count": sum(1 for row in usable if row.get("neutral_bulk_3d_established")),
        "estimators_agree_count": sum(1 for row in usable if row.get("neutral_dimension_estimators_agree")),
        "mean_primary_dimension": _mean(row.get("neutral_primary_dimension_estimate") for row in usable),
        "mean_correlation_dimension": _mean(row.get("neutral_correlation_dimension_estimate") for row in usable),
        "mean_local_mle_dimension": _mean(row.get("neutral_local_mle_dimension_estimate") for row in usable),
        "blind_usable_count": sum(1 for row in usable if row.get("blind_bulk_usable")),
        "blind_s2_leakage_pass_count": sum(1 for row in usable if row.get("blind_bulk_s2_leakage_audit_pass")),
        "blind_candidate_3d_window_count": sum(
            1 for row in usable if row.get("blind_bulk_candidate_3d_dimension_window")
        ),
        "blind_low_rank_usable_count": sum(1 for row in usable if row.get("blind_low_rank_usable")),
        "blind_low_rank_selected_3d_candidate_count": sum(
            1 for row in usable if row.get("blind_low_rank_selected_rank_3d_candidate_receipt")
        ),
        "blind_low_rank_selected_candidate_3d_window_count": sum(
            1 for row in usable if row.get("blind_low_rank_selected_candidate_3d_window")
        ),
        "blind_low_rank_selected_s2_leakage_pass_count": sum(
            1 for row in usable if row.get("blind_low_rank_selected_s2_leakage_audit_pass")
        ),
        "blind_record_transition_rank3_receipt_count": sum(
            1 for row in usable if row.get("blind_record_transition_rank3_receipt")
        ),
        "blind_record_transition_rank3_candidate_3d_window_count": sum(
            1 for row in usable if row.get("blind_record_transition_rank3_candidate_3d_window")
        ),
        "blind_record_transition_rank3_estimators_agree_count": sum(
            1 for row in usable if row.get("blind_record_transition_rank3_dimension_estimators_agree")
        ),
        "blind_record_transition_rank3_s2_leakage_pass_count": sum(
            1 for row in usable if row.get("blind_record_transition_rank3_s2_leakage_audit_pass")
        ),
        "mean_blind_low_rank_participation_rank": _mean(
            row.get("blind_low_rank_participation_rank") for row in usable
        ),
        "mean_blind_low_rank_entropy_rank": _mean(row.get("blind_low_rank_entropy_rank") for row in usable),
        "mean_blind_low_rank_selected_rank": _mean(row.get("blind_low_rank_selected_rank") for row in usable),
        "mean_blind_low_rank_selected_correlation_dimension": _mean(
            row.get("blind_low_rank_selected_correlation_dimension_estimate") for row in usable
        ),
        "mean_blind_low_rank_selected_local_mle_dimension": _mean(
            row.get("blind_low_rank_selected_local_mle_dimension_estimate") for row in usable
        ),
        "mean_blind_low_rank_selected_s2_distance_correlation": _mean(
            row.get("blind_low_rank_selected_s2_distance_correlation") for row in usable
        ),
        "mean_blind_record_transition_rank3_correlation_dimension": _mean(
            row.get("blind_record_transition_rank3_correlation_dimension_estimate") for row in usable
        ),
        "mean_blind_record_transition_rank3_local_mle_dimension": _mean(
            row.get("blind_record_transition_rank3_local_mle_dimension_estimate") for row in usable
        ),
        "mean_blind_record_transition_rank3_s2_distance_correlation": _mean(
            row.get("blind_record_transition_rank3_s2_distance_correlation") for row in usable
        ),
        "blind_bulk_3d_established_count": sum(1 for row in usable if row.get("blind_bulk_3d_established")),
        "mean_blind_feature_width": _mean(row.get("blind_bulk_feature_width") for row in usable),
        "mean_blind_s2_distance_correlation": _mean(row.get("blind_bulk_s2_distance_correlation") for row in usable),
        "mean_blind_correlation_dimension": _mean(
            row.get("blind_bulk_correlation_dimension_estimate") for row in usable
        ),
        "mean_blind_local_mle_dimension": _mean(row.get("blind_bulk_local_mle_dimension_estimate") for row in usable),
        "interpretation": (
            "Neutral observer-record reconstruction debug lane. It must report radial_depth_used=false "
            "and pass planted/shuffled controls before any dimension value is interpreted. The strict "
            "neutral report lane is the current hard gate for third-person bulk claims; it keeps its "
            "blockers separate from theorem-assisted H3 chart receipts. The strict blind lane excludes "
            "S2 axes, support nodes, cap membership, and radial/modular-depth priors. "
            "The low-rank sweep diagnoses whether overcomplete observer-transition features hide a "
            "3D continuation, but it is not used to force a bulk claim because rank selection can bias "
            "dimension estimates. The predeclared record-transition rank-3 sublane is reported "
            "separately because it is closer to the paper's observer-continuation object, but a 3D "
            "window there is still not a physical bulk claim unless the BW/object/refinement gates pass too."
        ),
    }


def _holonomy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("holonomy_triangle_count") is not None]
    return {
        "run_count": len(usable),
        "mean_defect_fraction": _mean(row.get("holonomy_defect_fraction") for row in usable),
        "mean_cluster_count": _mean(row.get("holonomy_cluster_count") for row in usable),
        "interpretation": (
            "Screen/collar S3 holonomy defect statistics are comparable across runs and controls. They are "
            "not matter-particle observables until persistent worldlines pass in a neutral 3D reconstruction."
        ),
    }


def _defect_worldline_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [
        row
        for row in rows
        if row.get("defect_timeline_written")
        or row.get("defect_interaction_written")
        or row.get("particle_likeness_written")
        or row.get("defect_h3_worldline_written")
    ]
    return {
        "run_count": len(usable),
        "timeline_run_count": sum(1 for row in usable if row.get("defect_timeline_written")),
        "interaction_run_count": sum(1 for row in usable if row.get("defect_interaction_written")),
        "particle_likeness_run_count": sum(1 for row in usable if row.get("particle_likeness_written")),
        "h3_worldline_run_count": sum(1 for row in usable if row.get("defect_h3_worldline_written")),
        "worldline_precursor_receipt_count": sum(1 for row in usable if row.get("defect_worldline_precursor_receipt")),
        "interaction_proxy_receipt_count": sum(1 for row in usable if row.get("defect_interaction_proxy_receipt")),
        "particle_matter_receipt_count": sum(1 for row in usable if row.get("particle_matter_receipt")),
        "h3_bulk_worldline_precursor_count": sum(
            1 for row in usable if row.get("defect_h3_bulk_worldline_precursor_receipt")
        ),
        "mean_timeline_worldline_count": _mean(row.get("defect_timeline_worldline_count") for row in usable),
        "mean_persistent_worldline_count": _mean(
            row.get("defect_timeline_persistent_worldline_count") for row in usable
        ),
        "mean_max_observation_count": _mean(row.get("defect_timeline_max_observation_count") for row in usable),
        "mean_max_lifetime_cycles": _mean(row.get("defect_timeline_max_lifetime_cycles") for row in usable),
        "mean_screen_transport_proxy_count": _mean(
            row.get("defect_screen_transport_proxy_count") for row in usable
        ),
        "mean_fusion_candidate_count": _mean(row.get("defect_fusion_candidate_count") for row in usable),
        "mean_particle_like_count": _mean(row.get("particle_like_count") for row in usable),
        "interpretation": (
            "Persistent S3 screen/collar defect worldlines and interaction proxies are particle precursors. "
            "They are not matter particles until they localize in a neutral 3D reconstruction and pass "
            "particle-likeness gates."
        ),
    }


def _controlled_defect_assay_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("controlled_defect_assay_written")]
    return {
        "run_count": len(usable),
        "inverse_identity_pass_count": sum(
            1 for row in usable if row.get("controlled_defect_assay_inverse_identity_pass")
        ),
        "interaction_proxy_receipt_count": sum(
            1 for row in usable if row.get("controlled_defect_assay_interaction_receipt")
        ),
        "fusion_conservation_pass_count": sum(
            1 for row in usable if row.get("controlled_defect_assay_fusion_pass")
        ),
        "detector_positive_count": sum(
            1 for row in usable if row.get("controlled_defect_assay_detector_positive")
        ),
        "physical_particle_emergence_count": sum(
            1 for row in usable if row.get("controlled_defect_assay_physical_particle_emergence")
        ),
        "mean_particle_like_count": _mean(
            row.get("controlled_defect_assay_particle_like_count") for row in usable
        ),
        "interpretation": (
            "Controlled planted S3 inverse-defect assay. Positive counts validate the particle-gate "
            "logic on known transport/fusion/scattering structure. They do not count as spontaneous "
            "matter emergence in production OPH-FPE dynamics."
        ),
    }


def _h3_chart_dimension_from_fit(h3_fit: dict[str, Any]) -> dict[str, Any]:
    points = np.asarray(h3_fit.get("fitted_h3_points", []), dtype=float)
    if points.ndim != 2 or points.shape[0] < 8 or points.shape[1] != 4:
        return {}
    dimension = neutral_dimension_report_from_distance(h3_distance_matrix(points))
    corr = dimension.get("correlation_dimension", {}).get("estimate")
    mle = dimension.get("local_mle_dimension", {}).get("estimate")
    candidate = bool(
        isinstance(corr, (int, float))
        and isinstance(mle, (int, float))
        and np.isfinite(float(corr))
        and np.isfinite(float(mle))
        and 2.7 <= float(corr) <= 3.3
        and 2.7 <= float(mle) <= 3.3
    )
    return {
        "mode": "h3_chart_dimension_debug",
        "point_count": int(points.shape[0]),
        "candidate_3d_dimension_window": candidate,
        "dimension_estimators_agree": bool(dimension.get("dimension_estimators_agree", False)),
        "correlation_dimension": dimension.get("correlation_dimension"),
        "local_mle_dimension": dimension.get("local_mle_dimension"),
        "claim_boundary": "computed from existing fitted H3 points for comparable-data diagnostics only",
    }


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(report: dict[str, Any]) -> str:
    lanes = report["measurement_lanes"]
    lorentz = lanes["support_visible_lorentz_branch"]
    state_bw = lanes["state_derived_bw_matrix_elements"]
    h3 = lanes["h3_modular_response_controls"]
    observer_readout = lanes["observer_readout_consensus"]
    object_chart = lanes["observer_chart_object_population"]
    neutral = lanes["neutral_observer_reconstruction"]
    planck = lanes["planck_tt_shape_lite"]
    transfer = lanes["cmb_screen_basis_transfer"]
    camb = lanes["camb_lcdm_baseline"]
    screen_power = lanes["oph_screen_power_effective_theory"]
    screen_camb = lanes["oph_screen_camb_transfer"]
    maxent_green = lanes["oph_maxent_green_screen_source"]
    repair_clock = lanes["oph_repair_clock_kappa"]
    scalar_semigroup = lanes["oph_scalar_repair_semigroup"]
    inflation_cmb = lanes["oph_inflation_cmb_bridge"]
    inflation_certificates = lanes["oph_inflation_certificate_stack"]
    finite_certificates = lanes["oph_finite_certificate_authority"]
    scalar_quotient = lanes["oph_scalar_geometric_quotient"]
    neutral_profile = lanes["neutral_distance_profile_audit"]
    prime_rank_sweep = lanes["prime_geometric_rank_sweep"]
    prime_rank_refinement = lanes["prime_geometric_rank_refinement"]
    parent_collar = lanes["oph_parent_collar_recovery_ladder"]
    b_a_parent = lanes["oph_B_A_parent_finite_difference"]
    screen_capacity = lanes["oph_screen_capacity_closure"]
    repair_scale = lanes["oph_repair_scale_closure"]
    scale_compressed = lanes["oph_scale_compressed_repair_branch"]
    scale_compressed_camb = lanes["oph_scale_compressed_cmb_camb_transfer"]
    inflation_cmb_camb = lanes["oph_inflation_cmb_camb_transfer"]
    selector_elimination = lanes["oph_cmb_selector_elimination_v1_5"]
    exact_cmb_camb = lanes["oph_exact_cmb_camb_transfer"]
    finite_clock_cmb = lanes["finite_repair_clock_cmb_camb_transfer"]
    unique_prediction = lanes["oph_unique_prediction_gate"]
    cnb_neutrino = lanes["oph_cnb_neutrino_background"]
    cmb_derivation = lanes["finite_lattice_cmb_derivation"]
    boltzmann = lanes["oph_boltzmann_input_readouts"]
    finite_collar_boltzmann = lanes["finite_collar_boltzmann_source_bundle"]
    finite_collar_projection = lanes["finite_collar_cmb_projection"]
    oph_cmb = lanes["oph_cmb_anomaly_stress_adapter"]
    cmb_anomaly = lanes["finite_screen_cmb_anomaly_readouts"]
    sync_gap = lanes["low_k_synchronization_gap"]
    hot_release = lanes["hot_maxent_release"]
    adiabaticity = lanes["same_boundary_adiabaticity"]
    h0s8 = lanes["h0_s8_branch_diagnostic"]
    shape = lanes["shape_substrate_witness"]
    galaxy = lanes["static_galaxy_proxy"]
    static_galaxy = lanes["static_galaxy_measurement_fit"]
    minimal_caps = lanes["minimal_caps_to_h3"]
    h3_seed_ensemble = lanes["h3_seed_ensemble_robustness"]
    hol = lanes["screen_holonomy_defect_proxy"]
    defect_worldlines = lanes["defect_worldline_particle_precursors"]
    controlled_defect = lanes["controlled_defect_particle_assay"]
    if lorentz.get("bulk_proof_theorem_assisted_h3_populated_chart_count", 0) or lorentz.get(
        "support_visible_h3_populated_bulk_count", 0
    ):
        current_answer = (
            "Yes: the current simulator emits diagnostic, measurement-facing values and a "
            "theorem-assisted populated-H3 chart receipt on the paper route. No: it does not "
            "yet emit a physical CMB prediction, physical P(k), strict neutral third-person "
            "bulk reconstruction, or particle spectrum."
        )
    else:
        current_answer = (
            "Yes: the current simulator can emit diagnostic, measurement-facing values. No: it "
            "does not yet emit a physical CMB prediction, physical P(k), populated 3D bulk, or "
            "particle spectrum."
        )
    lines = [
        "# OPH-FPE Comparable Data Snapshot",
        "",
        report["claim_boundary"],
        "",
        "## Current Answer",
        "",
        current_answer,
        "",
        f"- comparable rows: {report['run_count']}",
        f"- theorem-assisted H3 bulk any/count: {report['theorem_assisted_h3_bulk_any']} / {report['theorem_assisted_h3_bulk_count']}",
        f"- chart-level 3+1D any/count: {report['chart_level_3p1_any']} / {report['chart_level_3p1_count']}",
        f"- strict neutral 3D bulk any/count: {report['strict_neutral_3d_bulk_any']} / {report['strict_neutral_3d_bulk_count']}",
        f"- all selected rows bulk-established: {report['bulk_3d_established']}",
        f"- exact OPH target n_s: {_fmt(exact_cmb_camb.get('mean_n_s'))}",
        f"- exact OPH target CAMB IR shape correlation: {_fmt(exact_cmb_camb.get('mean_ir_shape_correlation'))}",
        f"- exact OPH target CAMB IR chi2/bin: {_fmt(exact_cmb_camb.get('mean_ir_chi2_per_bin'))}",
        f"- finite transition-clock n_s: {_fmt(scalar_semigroup.get('mean_n_s'))}",
        f"- finite transition-clock kappa_rep: {_fmt(scalar_semigroup.get('mean_kappa_rep'))}",
        f"- finite transition-clock certified count: {scalar_semigroup.get('transition_clock_certified_count')}",
        f"- finite transition required repair-step time for kappa=e: {_fmt(scalar_semigroup.get('mean_required_repair_step_time_for_kappa_e'))}",
        f"- screen Planck-lite best shape correlation: {_fmt(planck.get('mean_best_shape_correlation'))}",
        f"- physical CMB prediction: {report['physical_cmb_prediction']}",
        "",
        "## Support-Visible Lorentz Branch",
        "",
        f"- runs with Lorentz split receipts: {lorentz['run_count']}",
        f"- chart-level conformal Lorentz receipts: {lorentz['chart_level_conformal_lorentz_count']}",
        f"- BW automorphism sanity receipts: {lorentz['bw_automorphism_sanity_count']}",
        f"- support-visible 3+1D Lorentz receipts: {lorentz['support_visible_lorentz_3p1_count']}",
        f"- endogenous modular-generator receipts: {lorentz['endogenous_modular_generator_count']}",
        f"- paper-theorem 3D bulk-chart receipts: {lorentz['paper_theorem_3d_bulk_chart_count']}",
        f"- paper-theorem H3 spatial dimension mean: {_fmt(lorentz['mean_paper_theorem_h3_spatial_dimension'])}",
        f"- paper-theorem object-populated chart precursors: {lorentz['paper_theorem_object_populated_chart_precursor_count']}",
        f"- paper-theorem neutral populated bulk receipts: {lorentz['paper_theorem_neutral_populated_bulk_count']}",
        f"- support-visible record H3 population receipts: {lorentz['support_visible_record_h3_population_count']}",
        f"- support-visible defect H3 population receipts: {lorentz['support_visible_defect_h3_population_count']}",
        f"- support-visible populated-H3 bulk receipts: {lorentz['support_visible_h3_populated_bulk_count']}",
        f"- mean record-family H3 residual: {_fmt(lorentz['mean_record_family_h3_median_residual'])}",
        f"- mean defect-cluster H3 residual: {_fmt(lorentz['mean_defect_cluster_h3_median_residual'])}",
        f"- paper-theorem-assisted H3 chart precursor receipts: {lorentz['paper_theorem_assisted_h3_chart_precursor_count']}",
        f"- paper-theorem-assisted populated H3 chart receipts: {lorentz['paper_theorem_assisted_h3_populated_chart_count']}",
        f"- theorem-assisted H3 object-preview receipts: {lorentz['theorem_assisted_h3_object_preview_count']}",
        f"- object H3 nonboundary population receipts: {lorentz['object_h3_nonboundary_population_count']}",
        f"- object bulk-population receipts: {lorentz['object_bulk_population_count']}",
        f"- bulk-proof certificates: {lorentz['bulk_proof_certificate_count']}",
        f"- bulk-proof chart-level 3+1D counts: {lorentz['bulk_proof_chart_level_3p1_count']}",
        f"- bulk-proof theorem-assisted H3 preview counts: {lorentz['bulk_proof_theorem_assisted_h3_object_preview_count']}",
        f"- bulk-proof theorem-assisted nonboundary H3 counts: {lorentz['bulk_proof_theorem_assisted_h3_nonboundary_population_count']}",
        f"- bulk-proof legacy theorem-assisted populated H3 counts: {lorentz['bulk_proof_theorem_assisted_h3_populated_chart_count']}",
        f"- bulk-proof strict neutral 3D counts: {lorentz['bulk_proof_strict_neutral_3d_bulk_count']}",
        f"- bulk-proof screen-CMB proxy counts: {lorentz['bulk_proof_screen_cmb_proxy_count']}",
        f"- bulk-proof physical CMB counts: {lorentz['bulk_proof_physical_cmb_prediction_count']}",
        f"- bulk-proof production particle counts: {lorentz['bulk_proof_production_particle_matter_count']}",
        f"- screen-proxy CMB receipts: {lorentz['screen_proxy_cmb_count']}",
        f"- bulk-established count: {lorentz['bulk_3d_established_count']}",
        f"- interpretation: {lorentz['interpretation']}",
        "",
        "## State-Derived BW Matrix Elements",
        "",
        f"- runs with state-derived BW reports: {state_bw['run_count']}",
        f"- endogenous-generator runs: {state_bw['endogenous_run_count']}",
        f"- declared cap-flow generator runs: {state_bw['declared_cap_flow_run_count']}",
        f"- declared response-density runs: {state_bw['declared_response_density_run_count']}",
        f"- direct transition-automorphism runs: {state_bw['direct_transition_automorphism_run_count']}",
        f"- BW/KMS receipts: {state_bw['receipt_count']}",
        f"- density-log calibration receipts: {state_bw['density_log_calibration_receipt_count']}",
        f"- endogenous correct-beats-controls count: {state_bw['endogenous_correct_beats_controls_count']}",
        f"- selected-2pi count: {state_bw['selected_2pi_count']}",
        f"- state-mode counts: {state_bw['state_mode_counts']}",
        f"- generator-scale counts: {state_bw['generator_scale_counts']}",
        f"- best-control counts: {state_bw['best_control_counts']}",
        f"- target-scale degenerate-control runs: {state_bw['target_scale_control_degenerate_count']}",
        f"- degenerate target-scale control counts: {state_bw['degenerate_target_scale_control_counts']}",
        f"- generator-scale audit runs: {state_bw['generator_scale_audit_count']}",
        f"- generator audit configured-best count: {state_bw['generator_scale_audit_configured_best_count']}",
        f"- generator audit 2pi-best count: {state_bw['generator_scale_audit_two_pi_best_count']}",
        f"- generator audit best-label counts: {state_bw['generator_scale_audit_best_label_counts']}",
        f"- generator audit diagnosis counts: {state_bw['generator_scale_audit_diagnosis_counts']}",
        f"- mean generator-audit best score: {_fmt(state_bw['mean_generator_scale_audit_best_score'])}",
        f"- mean generator-audit configured score: {_fmt(state_bw['mean_generator_scale_audit_configured_score'])}",
        f"- mean median residual: {_fmt(state_bw['mean_median_residual'])}",
        f"- mean endogenous median residual: {_fmt(state_bw['mean_endogenous_median_residual'])}",
        f"- mean declared cap-flow median residual: {_fmt(state_bw['mean_declared_cap_flow_median_residual'])}",
        f"- mean declared response-density median residual: {_fmt(state_bw['mean_declared_response_density_median_residual'])}",
        f"- mean direct transition median residual: {_fmt(state_bw['mean_direct_transition_median_residual'])}",
        f"- mean wrong-1x median: {_fmt(state_bw['mean_wrong_1x_median'])}",
        f"- mean wrong-pi median: {_fmt(state_bw['mean_wrong_pi_median'])}",
        f"- mean no-modular-flow median: {_fmt(state_bw['mean_no_modular_flow_median'])}",
        f"- interpretation: {state_bw['interpretation']}",
        "",
        "## Declared Shape Substrate Witness",
        "",
        f"- Shape reports: {shape['run_count']}",
        f"- vertex-scattering receipts: {shape['vertex_scattering_receipt_count']}",
        f"- dodeca cell receipts: {shape['dodeca_cell_receipt_count']}",
        f"- settling receipts: {shape['shape_settling_receipt_count']}",
        f"- loop-particle receipts: {shape['loop_particle_receipt_count']}",
        f"- screen-projection receipts: {shape['screen_projection_receipt_count']}",
        f"- selector-elimination target input receipts: {shape['selector_elimination_target_input_count']}",
        f"- CMB-certificate input receipts: {shape['cmb_certificate_input_count']}",
        f"- declared-3D-substrate reports: {shape['declared_3d_substrate_count']}",
        f"- neutral OPH bulk claims: {shape['neutral_oph_bulk_claim_count']}",
        f"- physical-CMB prediction claims: {shape['physical_cmb_prediction_count']}",
        f"- mean q_IR target: {_fmt(shape['mean_q_IR_candidate'])}",
        f"- mean q_IR runtime zero-mode: {_fmt(shape['mean_q_IR_runtime_zero_mode'])}",
        f"- mean ell_IR target: {_fmt(shape['mean_ell_IR_candidate'])}",
        f"- mean ell_IR runtime covariance rank: {_fmt(shape['mean_ell_IR_runtime_covariance_rank'])}",
        f"- mean Phi drop fraction: {_fmt(shape['mean_phi_drop_fraction'])}",
        f"- mean loop-particle count: {_fmt(shape['mean_loop_particle_count'])}",
        f"- mean Planck-lite shape correlation: {_fmt(shape['mean_planck_lite_shape_correlation'])}",
        f"- mean Planck-lite RMSE: {_fmt(shape['mean_planck_lite_normalized_rmse'])}",
        f"- interpretation: {shape['interpretation']}",
        "",
        "## Static Galaxy Proxy",
        "",
        f"- galaxy proxy reports: {galaxy['run_count']}",
        f"- proxy receipts: {galaxy['proxy_receipt_count']}",
        f"- runs with external RAR/BTFR/disk data: {galaxy['runs_with_external_rar_or_btfr_data']}",
        f"- RAR lambda-fit count: {galaxy['rar_lambda_fit_count']}",
        f"- BTFR fit count: {galaxy['btfr_fit_count']}",
        f"- disk residual count: {galaxy['disk_residual_count']}",
        f"- mean declared a0_OPH: {_fmt(galaxy['mean_a0_oph'])}",
        f"- mean effective a0: {_fmt(galaxy['mean_a0_eff'])}",
        f"- mean declared lambda_collar: {_fmt(galaxy['mean_lambda_collar_declared'])}",
        f"- mean fitted lambda_collar: {_fmt(galaxy['mean_fitted_lambda_collar'])}",
        f"- mean BTFR slope: {_fmt(galaxy['mean_btfr_slope'])}",
        f"- interpretation: {galaxy['interpretation']}",
        "",
        "## Static Galaxy Measurement Fit",
        "",
        f"- external fit reports: {static_galaxy['run_count']}",
        f"- RAR/BTFR receipts: {static_galaxy['receipt_count']}",
        f"- OPH-CET bridge receipts: {static_galaxy['bridge_receipt_count']}",
        f"- claim-tier counts: {static_galaxy['claim_tier_counts']}",
        f"- physical-claim fit reports: {static_galaxy['physical_claim_count']}",
        f"- physical-CMB claim reports: {static_galaxy['physical_cmb_claim_count']}",
        f"- bulk-required reports: {static_galaxy['bulk_required_count']}",
        f"- mean dataset galaxy count: {_fmt(static_galaxy['mean_dataset_galaxy_count'])}",
        f"- mean measurement galaxy count: {_fmt(static_galaxy['mean_measurement_galaxy_count'])}",
        f"- mean RAR galaxy count: {_fmt(static_galaxy['mean_rar_galaxy_count'])}",
        f"- mean RAR galaxy support count: {_fmt(static_galaxy['mean_rar_galaxy_support_count'])}",
        f"- mean RAR point count: {_fmt(static_galaxy['mean_rar_point_count'])}",
        f"- mean RAR scatter dex: {_fmt(static_galaxy['mean_rar_scatter_dex'])}",
        f"- mean BTFR slope: {_fmt(static_galaxy['mean_btfr_slope'])}",
        f"- mean BTFR predicted slope from RAR fit: {_fmt(static_galaxy['mean_btfr_predicted_slope'])}",
        f"- mean BTFR slope delta: {_fmt(static_galaxy['mean_btfr_slope_delta'])}",
        f"- mean BTFR intercept delta dex: {_fmt(static_galaxy['mean_btfr_intercept_delta'])}",
        f"- holdout usable reports: {static_galaxy['holdout_usable_count']}",
        f"- holdout receipts: {static_galaxy['holdout_receipt_count']}",
        f"- mean holdout train/test galaxies: {_fmt(static_galaxy['mean_holdout_train_galaxy_count'])} / {_fmt(static_galaxy['mean_holdout_test_galaxy_count'])}",
        f"- mean holdout train/test points: {_fmt(static_galaxy['mean_holdout_train_point_count'])} / {_fmt(static_galaxy['mean_holdout_test_point_count'])}",
        f"- mean holdout shared a0: {_fmt(static_galaxy['mean_holdout_shared_a0'])}",
        f"- mean holdout shared lambda_collar: {_fmt(static_galaxy['mean_holdout_shared_lambda_collar'])}",
        f"- mean holdout train log-accel RMSE dex: {_fmt(static_galaxy['mean_holdout_train_log_accel_rmse'])}",
        f"- mean holdout test log-accel RMSE dex: {_fmt(static_galaxy['mean_holdout_test_log_accel_rmse'])}",
        f"- mean holdout test velocity RMSE km/s: {_fmt(static_galaxy['mean_holdout_test_velocity_rmse'])}",
        f"- mean holdout baryon-only velocity RMSE km/s: {_fmt(static_galaxy['mean_holdout_test_baryon_velocity_rmse'])}",
        f"- mean holdout velocity RMSE improvement: {_fmt(static_galaxy['mean_holdout_test_velocity_improvement'])}",
        f"- interpretation: {static_galaxy['interpretation']}",
        "",
        "## Minimal Caps-To-H3 Calibration",
        "",
        f"- calibration reports: {minimal_caps['run_count']}",
        f"- calibration receipts: {minimal_caps['receipt_count']}",
        f"- mean median reconstruction MSE: {_fmt(minimal_caps['mean_median_reconstruction_mse'])}",
        f"- mean shuffled profile MSE: {_fmt(minimal_caps['mean_median_shuffled_profile_mse'])}",
        f"- mean S2-boundary profile MSE: {_fmt(minimal_caps['mean_median_s2_boundary_profile_mse'])}",
        f"- H3 beats shuffled count: {minimal_caps['h3_beats_shuffled_count']}",
        f"- H3 beats S2-boundary count: {minimal_caps['h3_beats_s2_boundary_count']}",
        f"- claim boundary: {minimal_caps['claim_boundary']}",
        "",
        "## H3 Seed Ensemble Robustness",
        "",
        f"- ensemble reports: {h3_seed_ensemble['run_count']}",
        f"- response seed-robust receipts: {h3_seed_ensemble['response_seed_robust_count']}",
        f"- chart-3D seed-robust receipts: {h3_seed_ensemble['chart_3d_seed_robust_count']}",
        f"- mean seed count: {_fmt(h3_seed_ensemble['mean_seed_count'])}",
        f"- mean single-seed receipt fraction: {_fmt(h3_seed_ensemble['mean_receipt_fraction'])}",
        f"- mean control-separation fraction: {_fmt(h3_seed_ensemble['mean_control_separation_fraction'])}",
        f"- mean candidate receipt fraction: {_fmt(h3_seed_ensemble['mean_candidate_receipt_fraction'])}",
        f"- mean 3D-window fraction: {_fmt(h3_seed_ensemble['mean_dim3_fraction'])}",
        f"- mean heldout normalized RMSE: {_fmt(h3_seed_ensemble['mean_heldout_normalized_rmse'])}",
        f"- mean heldout explained variance: {_fmt(h3_seed_ensemble['mean_heldout_explained_variance'])}",
        f"- mean p75 material wrong-scale fraction: {_fmt(h3_seed_ensemble['mean_p75_material_wrong'])}",
        f"- interpretation: {h3_seed_ensemble['interpretation']}",
        "",
        "## H3 Modular-Response Receipt",
        "",
        f"- runs with H3 fits: {h3['run_count']}",
        f"- H3 receipts: {h3['receipt_count']}",
        f"- H3 control-separation precursor receipts: {h3['control_separation_receipt_count']}",
        f"- mean H3 RMSE: {_fmt(h3['mean_h3_rmse'])}",
        f"- mean H3 explained variance: {_fmt(h3['mean_h3_explained_variance'])}",
        f"- H3 channel modes: {h3['channel_modes']}",
        f"- H3 profile modes: {h3['profile_modes']}",
        f"- mean H3 channel count: {_fmt(h3['mean_h3_channel_count'])}",
        f"- H3-chart 3D-window count: {h3['h3_chart_3d_window_count']}",
        f"- H3-chart estimator-agreement count: {h3['h3_chart_dimension_agree_count']}",
        f"- staged signal gate count: {h3['signal_gate_count']}",
        f"- staged geometry gate count: {h3['geometry_gate_count']}",
        f"- staged aggregate wrong-scale gate count: {h3['aggregate_wrong_scale_gate_count']}",
        f"- staged material feature gate count: {h3['material_feature_gate_count']}",
        f"- mean H3-chart correlation dimension: {_fmt(h3['mean_h3_chart_correlation_dimension'])}",
        f"- mean H3-chart local-MLE dimension: {_fmt(h3['mean_h3_chart_local_mle_dimension'])}",
        f"- mean S2-boundary RMSE: {_fmt(h3['mean_s2_boundary_rmse'])}",
        f"- mean shuffled-response RMSE: {_fmt(h3['mean_shuffled_response_rmse'])}",
        f"- strict wrong-scale feature audits: {h3['wrong_scale_feature_audit_count']}",
        f"- mean wrong-scale feature win fraction: {_fmt(h3['mean_wrong_scale_feature_win_fraction'])}",
        f"- mean 2pi/H3 feature win fraction: {_fmt(h3['mean_two_pi_h3_feature_win_fraction'])}",
        f"- wrong-scale red-flag runs: {h3['wrong_scale_red_flag_count']}",
        f"- mean material wrong-scale win fraction: {_fmt(h3['mean_material_wrong_scale_feature_win_fraction'])}",
        f"- mean material 2pi/H3 win fraction: {_fmt(h3['mean_material_two_pi_h3_feature_win_fraction'])}",
        f"- material wrong-scale red-flag runs: {h3['material_wrong_scale_red_flag_count']}",
        f"- worst wrong-scale group sample: {h3['worst_wrong_scale_groups'][0] if h3['worst_wrong_scale_groups'] else 'n/a'}",
        "",
        "## Observer Readout Consensus",
        "",
        f"- runs with observer readout reports: {observer_readout['run_count']}",
        f"- mean observer count: {_fmt(observer_readout['mean_observer_count'])}",
        f"- mean overlap pair count: {_fmt(observer_readout['mean_pair_count'])}",
        f"- mean global committed fraction: {_fmt(observer_readout['mean_global_committed_fraction'])}",
        f"- mean median overlap Jaccard: {_fmt(observer_readout['mean_median_overlap_jaccard'])}",
        f"- mean median signature similarity: {_fmt(observer_readout['mean_median_signature_similarity'])}",
        f"- mean p10 signature similarity: {_fmt(observer_readout['mean_p10_signature_similarity'])}",
        f"- mean object overlap agreement: {_fmt(observer_readout['mean_object_overlap_agreement'])}",
        f"- mean object p10 overlap agreement: {_fmt(observer_readout['mean_object_p10_overlap_agreement'])}",
        f"- mean counterfactual stability: {_fmt(observer_readout['mean_counterfactual_stability'])}",
        f"- bad record rewrite detections: {observer_readout['bad_record_rewrite_detected_count']}",
        "",
        "## Observer-Chart Object Population",
        "",
        f"- runs with object-chart reports: {object_chart['run_count']}",
        f"- object-chart receipts: {object_chart['object_chart_receipt_count']}",
        f"- object-chart median receipts: {object_chart['object_chart_median_receipt_count']}",
        f"- object-chart H3 control-separation receipts: {object_chart['modular_response_h3_control_separation_receipt_count']}",
        f"- localized-object precursor receipts: {object_chart['localized_object_precursor_receipt_count']}",
        f"- bulk-population receipts: {object_chart['bulk_population_receipt_count']}",
        f"- mean object count: {_fmt(object_chart['mean_object_count'])}",
        f"- mean localized object count: {_fmt(object_chart['mean_localized_object_count'])}",
        f"- mean localized non-boundary object count: {_fmt(object_chart['mean_localized_not_boundary_object_count'])}",
        f"- mean shuffle control count: {_fmt(object_chart['mean_shuffle_control_count'])}",
        f"- mean shuffled localized object count: {_fmt(object_chart['mean_shuffled_localized_object_count'])}",
        f"- mean shuffled localized object p90: {_fmt(object_chart['mean_shuffled_localized_object_p90'])}",
        f"- mean H3 compactness: {_fmt(object_chart['mean_h3_compactness'])}",
        f"- mean S2-boundary compactness: {_fmt(object_chart['mean_s2_boundary_compactness'])}",
        f"- mean shuffled-H3 compactness: {_fmt(object_chart['mean_shuffled_h3_compactness'])}",
        f"- mean p10 shuffled-H3 compactness: {_fmt(object_chart['mean_p10_shuffled_h3_compactness'])}",
        f"- mean p90 shuffled-H3 compactness: {_fmt(object_chart['mean_p90_shuffled_h3_compactness'])}",
        f"- mean H3 margin vs median shuffle: {_fmt(object_chart['mean_h3_margin_vs_median_shuffle'])}",
        f"- mean H3 margin vs p10 shuffle: {_fmt(object_chart['mean_h3_margin_vs_p10_shuffle'])}",
        f"- mean H3 margin vs S2 boundary: {_fmt(object_chart['mean_h3_margin_vs_s2_boundary'])}",
        f"- H3 beats median shuffle count: {object_chart['h3_beats_shuffled_count']}",
        f"- H3 beats robust shuffle count: {object_chart['h3_beats_shuffled_robust_count']}",
        f"- compactness-distribution control receipts: {object_chart['compactness_distribution_control_receipt_count']}",
        f"- compactness-distribution strict population receipts: {object_chart['compactness_distribution_population_receipt_count']}",
        f"- localized-count saturation warnings: {object_chart['localized_count_saturation_warning_count']}",
        f"- localized non-boundary saturation warnings: {object_chart['localized_not_boundary_count_saturation_warning_count']}",
        "",
        "## Neutral Observer Reconstruction",
        "",
        f"- runs with neutral reconstruction reports: {neutral['run_count']}",
        f"- legacy neutral reconstruction reports: {neutral['legacy_neutral_reconstruction_count']}",
        f"- strict-neutral reports: {neutral['strict_neutral_report_count']}",
        f"- strict-neutral bulk receipts: {neutral['strict_neutral_bulk_count']}",
        f"- strict-neutral H3 selected count: {neutral['strict_neutral_h3_selected_count']}",
        f"- strict-neutral H3 best count: {neutral['strict_neutral_h3_best_count']}",
        f"- strict-neutral estimators-agree-3D count: {neutral['strict_neutral_estimators_agree_3d_count']}",
        f"- strict-neutral S2-leakage-pass count: {neutral['strict_neutral_s2_leakage_pass_count']}",
        f"- mean strict-neutral observer count: {_fmt(neutral['mean_strict_neutral_observer_count'])}",
        f"- mean strict-neutral median dimension: {_fmt(neutral['mean_strict_neutral_median_dimension'])}",
        f"- mean strict-neutral correlation dimension: {_fmt(neutral['mean_strict_neutral_correlation_dimension'])}",
        f"- mean strict-neutral local-MLE dimension: {_fmt(neutral['mean_strict_neutral_local_mle_dimension'])}",
        f"- mean strict-neutral S2-distance correlation: {_fmt(neutral['mean_strict_neutral_s2_distance_correlation'])}",
        f"- radial-depth-used count: {neutral['radial_depth_used_count']}",
        f"- control-gate passed count: {neutral['control_gate_passed_count']}",
        f"- candidate 3D-window count: {neutral['candidate_3d_window_count']}",
        f"- bulk-established count: {neutral['bulk_3d_established_count']}",
        f"- estimators-agree count: {neutral['estimators_agree_count']}",
        f"- mean primary dimension: {_fmt(neutral['mean_primary_dimension'])}",
        f"- mean correlation dimension: {_fmt(neutral['mean_correlation_dimension'])}",
        f"- mean local-MLE dimension: {_fmt(neutral['mean_local_mle_dimension'])}",
        f"- blind usable count: {neutral['blind_usable_count']}",
        f"- blind S2-leakage-pass count: {neutral['blind_s2_leakage_pass_count']}",
        f"- blind candidate 3D-window count: {neutral['blind_candidate_3d_window_count']}",
        f"- blind low-rank usable count: {neutral['blind_low_rank_usable_count']}",
        f"- blind low-rank selected 3D-candidate count: {neutral['blind_low_rank_selected_3d_candidate_count']}",
        f"- blind low-rank selected 3D-window count: {neutral['blind_low_rank_selected_candidate_3d_window_count']}",
        f"- blind low-rank selected S2-leakage-pass count: {neutral['blind_low_rank_selected_s2_leakage_pass_count']}",
        f"- blind record-transition rank-3 receipt count: {neutral['blind_record_transition_rank3_receipt_count']}",
        f"- blind record-transition rank-3 3D-window count: {neutral['blind_record_transition_rank3_candidate_3d_window_count']}",
        f"- blind record-transition rank-3 estimator-agreement count: {neutral['blind_record_transition_rank3_estimators_agree_count']}",
        f"- blind record-transition rank-3 S2-leakage-pass count: {neutral['blind_record_transition_rank3_s2_leakage_pass_count']}",
        f"- blind bulk-established count: {neutral['blind_bulk_3d_established_count']}",
        f"- mean blind feature width: {_fmt(neutral['mean_blind_feature_width'])}",
        f"- mean blind S2-distance correlation: {_fmt(neutral['mean_blind_s2_distance_correlation'])}",
        f"- mean blind correlation dimension: {_fmt(neutral['mean_blind_correlation_dimension'])}",
        f"- mean blind local-MLE dimension: {_fmt(neutral['mean_blind_local_mle_dimension'])}",
        f"- mean blind low-rank participation rank: {_fmt(neutral['mean_blind_low_rank_participation_rank'])}",
        f"- mean blind low-rank entropy rank: {_fmt(neutral['mean_blind_low_rank_entropy_rank'])}",
        f"- mean blind low-rank selected rank: {_fmt(neutral['mean_blind_low_rank_selected_rank'])}",
        f"- mean blind low-rank selected correlation dimension: {_fmt(neutral['mean_blind_low_rank_selected_correlation_dimension'])}",
        f"- mean blind low-rank selected local-MLE dimension: {_fmt(neutral['mean_blind_low_rank_selected_local_mle_dimension'])}",
        f"- mean blind low-rank selected S2-distance correlation: {_fmt(neutral['mean_blind_low_rank_selected_s2_distance_correlation'])}",
        f"- mean blind record-transition rank-3 correlation dimension: {_fmt(neutral['mean_blind_record_transition_rank3_correlation_dimension'])}",
        f"- mean blind record-transition rank-3 local-MLE dimension: {_fmt(neutral['mean_blind_record_transition_rank3_local_mle_dimension'])}",
        f"- mean blind record-transition rank-3 S2-distance correlation: {_fmt(neutral['mean_blind_record_transition_rank3_s2_distance_correlation'])}",
        "",
        "## Planck-Lite Screen C_l Shape",
        "",
        f"- runs with CMB-lite comparisons: {planck['run_count']}",
        f"- best-field counts: {planck['best_field_counts']}",
        f"- best-positive-field counts: {planck['best_positive_field_counts']}",
        f"- best real-ell overlap field counts: {planck['best_real_ell_overlap_field_counts']}",
        f"- mean best-field correlation: {_fmt(planck['mean_best_shape_correlation'])}",
        f"- mean best-field RMSE: {_fmt(planck['mean_best_normalized_rmse'])}",
        f"- mean best-positive-field correlation: {_fmt(planck['mean_best_positive_shape_correlation'])}",
        f"- mean best-positive-field RMSE: {_fmt(planck['mean_best_positive_normalized_rmse'])}",
        f"- mean record-signature correlation: {_fmt(planck['mean_record_signature_shape_correlation'])}",
        f"- mean record-signature RMSE: {_fmt(planck['mean_record_signature_normalized_rmse'])}",
        f"- real-ell overlap usable runs: {planck['overlap_ell_usable_count']}",
        f"- mean record-signature overlap-ell correlation: {_fmt(planck['mean_record_signature_overlap_ell_shape_correlation'])}",
        f"- mean record-signature overlap-ell RMSE: {_fmt(planck['mean_record_signature_overlap_ell_normalized_rmse'])}",
        f"- mean overlap ell range: [{_fmt(planck['mean_record_signature_overlap_ell_min'])}, {_fmt(planck['mean_record_signature_overlap_ell_max'])}]",
        f"- mean overlap benchmark bins: {_fmt(planck['mean_record_signature_overlap_benchmark_count'])}",
        f"- best real-ell overlap usable runs: {planck['best_real_ell_overlap_usable_count']}",
        f"- mean best real-ell overlap correlation: {_fmt(planck['mean_best_real_ell_overlap_shape_correlation'])}",
        f"- mean best real-ell overlap RMSE: {_fmt(planck['mean_best_real_ell_overlap_normalized_rmse'])}",
        f"- mean best real-ell overlap range: [{_fmt(planck['mean_best_real_ell_overlap_min'])}, {_fmt(planck['mean_best_real_ell_overlap_max'])}]",
        f"- mean best real-ell overlap benchmark bins: {_fmt(planck['mean_best_real_ell_overlap_benchmark_count'])}",
        "",
        "## Finite-Screen CMB Anomaly Readouts",
        "",
        f"- anomaly reports: {cmb_anomaly['run_count']}",
        f"- primary-field counts: {cmb_anomaly['primary_field_counts']}",
        f"- best low-power field counts: {cmb_anomaly['best_low_power_field_counts']}",
        f"- best large-angle field counts: {cmb_anomaly['best_large_angle_field_counts']}",
        f"- best parity field counts: {cmb_anomaly['best_parity_field_counts']}",
        f"- best tilt field counts: {cmb_anomaly['best_tilt_field_counts']}",
        f"- mean best low-power fraction: {_fmt(cmb_anomaly['mean_best_low_power_abs_fraction'])}",
        f"- mean best S1/2 scalar proxy: {_fmt(cmb_anomaly['mean_best_S_1_2_scalar_proxy'])}",
        f"- mean best parity log deviation: {_fmt(cmb_anomaly['mean_best_parity_log_abs_deviation'])}",
        f"- mean best eta_R: {_fmt(cmb_anomaly['mean_best_eta_R_estimate'])}",
        f"- mean best n_s proxy: {_fmt(cmb_anomaly['mean_best_n_s_proxy'])}",
        f"- mean low-power-suppressed field count: {_fmt(cmb_anomaly['mean_low_power_suppressed_vs_controls_count'])}",
        f"- mean large-angle-suppressed field count: {_fmt(cmb_anomaly['mean_large_angle_suppressed_vs_controls_count'])}",
        f"- mean parity-asymmetric field count: {_fmt(cmb_anomaly['mean_parity_more_asymmetric_than_controls_count'])}",
        f"- mean Planck-tilt-compatible field count: {_fmt(cmb_anomaly['mean_planck_tilt_compatible_proxy_count'])}",
        f"- mean total entropy capacity: {_fmt(cmb_anomaly['mean_total_entropy_capacity'])}",
        f"- mean sqrt-N ell capacity proxy: {_fmt(cmb_anomaly['mean_ell_sqrt_capacity_proxy'])}",
        f"- physical-CMB prediction reports: {cmb_anomaly['physical_cmb_prediction_count']}",
        f"- interpretation: {cmb_anomaly['interpretation']}",
        "",
        "## CMB Screen-Basis Transfer",
        "",
        f"- transfer reports: {transfer['run_count']}",
        f"- diagnostic receipts: {transfer['diagnostic_receipt_count']}",
        f"- mean train correlation: {_fmt(transfer['mean_train_shape_correlation'])}",
        f"- mean train RMSE: {_fmt(transfer['mean_train_normalized_rmse'])}",
        f"- mean test correlation: {_fmt(transfer['mean_test_shape_correlation'])}",
        f"- mean test RMSE: {_fmt(transfer['mean_test_normalized_rmse'])}",
        f"- mean max control test correlation: {_fmt(transfer['mean_max_control_test_shape_correlation'])}",
        f"- mean test-control correlation gap: {_fmt(transfer['mean_test_control_gap'])}",
        f"- mean bootstrap test correlation [p05, p95]: [{_fmt(transfer['mean_bootstrap_corr_p05'])}, {_fmt(transfer['mean_bootstrap_corr_p95'])}]",
        f"- mean bootstrap test RMSE [p05, p95]: [{_fmt(transfer['mean_bootstrap_rmse_p05'])}, {_fmt(transfer['mean_bootstrap_rmse_p95'])}]",
        f"- interpretation: {transfer['interpretation']}",
        "",
        "## CAMB LambdaCDM Baseline",
        "",
        f"- baseline reports: {camb['run_count']}",
        f"- CDM-limit Boltzmann receipts: {camb['cdm_limit_boltzmann_receipt_count']}",
        f"- OPH anomaly-module-ready reports: {camb['oph_anomaly_module_ready_count']}",
        f"- mean shape correlation: {_fmt(camb['mean_shape_correlation'])}",
        f"- mean normalized RMSE: {_fmt(camb['mean_normalized_rmse'])}",
        f"- mean amplitude-fit chi2/bin: {_fmt(camb['mean_amplitude_fit_chi2_per_bin'])}",
        f"- mean best-fit-column chi2/bin: {_fmt(camb['mean_best_fit_column_chi2_per_bin'])}",
        f"- mean first peak ell: {_fmt(camb['mean_first_peak_ell'])}",
        f"- mean benchmark first peak ell: {_fmt(camb['mean_benchmark_first_peak_ell'])}",
        f"- mean absolute fractional error: {_fmt(camb['mean_abs_fractional_error'])}",
        f"- interpretation: {camb['interpretation']}",
        "",
        "## OPH Screen-Power Effective Theory",
        "",
        f"- screen-power reports: {screen_power['run_count']}",
        f"- simulator primordial-ready reports: {screen_power['simulator_primordial_ready_count']}",
        f"- physical-CMB prediction reports: {screen_power['physical_cmb_prediction_count']}",
        f"- mean source run count: {_fmt(screen_power['mean_source_run_count'])}",
        f"- mean fit row count: {_fmt(screen_power['mean_fit_row_count'])}",
        f"- mean available fit count: {_fmt(screen_power['mean_available_fit_count'])}",
        f"- best Planck-eta field counts: {screen_power['best_planck_eta_field_counts']}",
        f"- reference-source counts: {screen_power['reference_source_counts']}",
        f"- mean reference eta_R: {_fmt(screen_power['mean_reference_eta_R'])}",
        f"- mean reference n_s: {_fmt(screen_power['mean_reference_n_s'])}",
        f"- mean simulator record-signature eta_R: {_fmt(screen_power['mean_record_signature_eta_R'])}",
        f"- mean simulator stable-count eta_R: {_fmt(screen_power['mean_stable_count_eta_R'])}",
        f"- interpretation: {screen_power['interpretation']}",
        "",
        "## OPH Screen CAMB Transfer",
        "",
        f"- OPH-screen CAMB reports: {screen_camb['run_count']}",
        f"- transfer receipts: {screen_camb['screen_camb_transfer_receipt_count']}",
        f"- simulator eta-ready reports: {screen_camb['simulator_eta_ready_count']}",
        f"- physical-CMB prediction reports: {screen_camb['physical_cmb_prediction_count']}",
        f"- reference-source counts: {screen_camb['reference_source_counts']}",
        f"- mean eta_R: {_fmt(screen_camb['mean_eta_R'])}",
        f"- mean n_s: {_fmt(screen_camb['mean_n_s'])}",
        f"- mean shape correlation: {_fmt(screen_camb['mean_shape_correlation'])}",
        f"- mean normalized RMSE: {_fmt(screen_camb['mean_normalized_rmse'])}",
        f"- mean amplitude-fit chi2/bin: {_fmt(screen_camb['mean_amplitude_fit_chi2_per_bin'])}",
        f"- mean first peak ell: {_fmt(screen_camb['mean_first_peak_ell'])}",
        f"- mean benchmark first peak ell: {_fmt(screen_camb['mean_benchmark_first_peak_ell'])}",
        f"- mean absolute fractional error: {_fmt(screen_camb['mean_abs_fractional_error'])}",
        f"- interpretation: {screen_camb['interpretation']}",
        "",
        "## OPH MaxEnt Green-Spectrum Source",
        "",
        f"- source reports: {maxent_green['run_count']}",
        f"- source receipts: {maxent_green['source_receipt_count']}",
        f"- flat-Green receipts: {maxent_green['flat_green_receipt_count']}",
        f"- IR theorem receipts: {maxent_green['ir_theorem_receipt_count']}",
        f"- source-packet audits: {maxent_green['source_packet_audit_count']}",
        f"- repair-clock certificates: {maxent_green['repair_clock_certificate_count']}",
        f"- bandlimit-for-IR counts: {maxent_green['bandlimit_for_ir_count']}",
        f"- bandlimit-for-requested-ell counts: {maxent_green['bandlimit_for_requested_ell_count']}",
        f"- finite-lattice-derived reports: {maxent_green['finite_lattice_derived_count']}",
        f"- physical-CMB prediction reports: {maxent_green['physical_cmb_prediction_count']}",
        f"- mean eta_R: {_fmt(maxent_green['mean_eta_R'])}",
        f"- mean n_s: {_fmt(maxent_green['mean_n_s'])}",
        f"- mean q_IR: {_fmt(maxent_green['mean_q_IR'])}",
        f"- mean ell_IR: {_fmt(maxent_green['mean_ell_IR'])}",
        f"- mean N_frz proxy: {_fmt(maxent_green['mean_N_frz_proxy'])}",
        f"- mean F_IR(ell=2): {_fmt(maxent_green['mean_F_IR_ell2'])}",
        f"- mean F_IR(ell=32): {_fmt(maxent_green['mean_F_IR_ell32'])}",
        f"- interpretation: {maxent_green['interpretation']}",
        "",
        "## OPH Repair-Clock Kappa Audit",
        "",
        f"- repair-clock reports: {repair_clock['run_count']}",
        f"- finite repair-clock certificates: {repair_clock['finite_repair_clock_certificate_count']}",
        f"- eta_R finite-lattice-derived reports: {repair_clock['eta_R_finite_lattice_derived_count']}",
        f"- cycle-time normalization declared reports: {repair_clock['cycle_time_normalization_declared_count']}",
        f"- mean candidate run count: {_fmt(repair_clock['mean_candidate_run_count'])}",
        f"- mean estimator count: {_fmt(repair_clock['mean_estimator_count'])}",
        f"- mean eligible estimator count: {_fmt(repair_clock['mean_eligible_estimator_count'])}",
        f"- mean passed estimator count: {_fmt(repair_clock['mean_passed_estimator_count'])}",
        f"- mean median kappa_rep estimate: {_fmt(repair_clock['mean_median_kappa_rep'])}",
        f"- target kappa_rep: {_fmt(repair_clock['target_kappa_rep'])}",
        f"- mean median eta_R estimate: {_fmt(repair_clock['mean_median_eta_R'])}",
        f"- target eta_R: {_fmt(repair_clock['target_eta_R'])}",
        f"- mean blocker count: {_fmt(repair_clock['mean_blocker_count'])}",
        f"- interpretation: {repair_clock['interpretation']}",
        "",
        "## OPH Scalar Repair-Semigroup Gap",
        "",
        f"- semigroup reports: {scalar_semigroup['run_count']}",
        f"- algebraic target receipts: {scalar_semigroup['semigroup_target_receipt_count']}",
        f"- repair-clock certificates: {scalar_semigroup['repair_clock_certificate_count']}",
        f"- eligible certificate reports: {scalar_semigroup['eligible_for_certificate_count']}",
        f"- finite-lattice-derived reports: {scalar_semigroup['finite_lattice_derived_count']}",
        f"- sources: {scalar_semigroup['source_values']}",
        f"- mean dimension: {_fmt(scalar_semigroup['mean_dimension'])}",
        f"- mean centered dimension: {_fmt(scalar_semigroup['mean_centered_subspace_dimension'])}",
        f"- mean kappa_rep: {_fmt(scalar_semigroup['mean_kappa_rep'])}",
        f"- mean eta_R: {_fmt(scalar_semigroup['mean_eta_R'])}",
        f"- mean n_s: {_fmt(scalar_semigroup['mean_n_s'])}",
        f"- mean centered gap: {_fmt(scalar_semigroup['mean_centered_gap'])}",
        f"- mean primary lambda_2: {_fmt(scalar_semigroup['mean_primary_lambda_2'])}",
        f"- mean required repair-step time for kappa=e: {_fmt(scalar_semigroup['mean_required_repair_step_time_for_kappa_e'])}",
        f"- transition clock certified reports: {scalar_semigroup['transition_clock_certified_count']}",
        f"- controls-passed reports: {scalar_semigroup['controls_passed_count']}",
        f"- interpretation: {scalar_semigroup['interpretation']}",
        "",
        "## OPH Inflation/CMB Bridge",
        "",
        f"- bridge reports: {inflation_cmb['run_count']}",
        f"- v0.4 public-data diagnostic reports: {inflation_cmb['v04_diagnostic_available_count']}",
        f"- physical-CMB prediction reports: {inflation_cmb['physical_cmb_prediction_count']}",
        f"- mean n_s = 1 - P/48: {_fmt(inflation_cmb['mean_n_s'])}",
        f"- mean theta_OPH = P/48: {_fmt(inflation_cmb['mean_theta_OPH'])}",
        f"- mean A_zeta: {_fmt(inflation_cmb['mean_A_zeta'])}",
        f"- mean n_s pull vs Planck target: {_fmt(inflation_cmb['mean_n_s_pull_vs_planck'])}",
        f"- mean selected Omega_K: {_fmt(inflation_cmb['mean_Omega_K'])}",
        f"- mean residual Omega_A0: {_fmt(inflation_cmb['mean_Omega_A0'])}",
        f"- mean rho_A/rho_b: {_fmt(inflation_cmb['mean_rho_A_over_rho_b'])}",
        f"- mean OPH IR q_IR: {_fmt(inflation_cmb['mean_q_IR'])}",
        f"- mean OPH IR ell_IR: {_fmt(inflation_cmb['mean_ell_IR'])}",
        f"- mean CAMB LCDM low-ell chi2: {_fmt(inflation_cmb['mean_camb_lcdm_lowell_chi2'])}",
        f"- mean CAMB OPH-IR low-ell chi2: {_fmt(inflation_cmb['mean_camb_oph_ir_lowell_chi2'])}",
        f"- mean LCDM R_OE upper-tail PTE: {_fmt(inflation_cmb['mean_lcdm_roe_pte'])}",
        f"- mean OPH parity R_OE upper-tail PTE: {_fmt(inflation_cmb['mean_oph_parity_roe_pte'])}",
        f"- v0.5 TT low-ell delta chi2: {_fmt(inflation_cmb['mean_v05_TT_lowell_delta_chi2'])}",
        f"- v0.5 TE low-ell delta chi2: {_fmt(inflation_cmb['mean_v05_TE_lowell_delta_chi2'])}",
        f"- v0.5 EE low-ell delta chi2: {_fmt(inflation_cmb['mean_v05_EE_lowell_delta_chi2'])}",
        f"- v0.5 TT high-ell delta chi2, ell=30..1200: {_fmt(inflation_cmb['mean_v05_TT_high_ell_delta_chi2_30_1200'])}",
        f"- v0.5 combined TT+TE+EE low-ell delta chi2: {_fmt(inflation_cmb['mean_v05_combined_lowell_delta_chi2'])}",
        f"- v0.5 pressure/not-yet-run gate count: {_fmt(inflation_cmb['mean_v05_pressure_point_count'])}",
        f"- interpretation: {inflation_cmb['interpretation']}",
        "",
        "## OPH Inflation Certificate Stack",
        "",
        f"- certificate reports: {inflation_certificates['run_count']}",
        f"- stack-ready reports: {inflation_certificates['stack_ready_count']}",
        f"- physical-CMB prediction reports: {inflation_certificates['physical_cmb_prediction_count']}",
        f"- physical matter-power prediction reports: {inflation_certificates['physical_matter_power_prediction_count']}",
        f"- no-data-use firewall count: {inflation_certificates['no_data_use_count']}",
        f"- scalar/edge/anomaly/parent/repair/boltzmann gates: {inflation_certificates['scalar_release_gate_count']} / {inflation_certificates['edge_center_gate_count']} / {inflation_certificates['homogeneous_anomaly_gate_count']} / {inflation_certificates['parent_collar_gate_count']} / {inflation_certificates['repair_matrix_gate_count']} / {inflation_certificates['boltzmann_handoff_gate_count']}",
        f"- mean found/passed/expected certificates: {_fmt(inflation_certificates['mean_found_count'])} / {_fmt(inflation_certificates['mean_passed_count'])} / {_fmt(inflation_certificates['mean_expected_count'])}",
        f"- mean missing certificate count: {_fmt(inflation_certificates['mean_missing_count'])}",
        f"- mean A_zeta: {_fmt(inflation_certificates['mean_A_zeta'])}",
        f"- mean n_s: {_fmt(inflation_certificates['mean_n_s'])}",
        f"- mean Q_A: {_fmt(inflation_certificates['mean_Q_A'])}",
        f"- mean B_A: {_fmt(inflation_certificates['mean_B_A'])}",
        f"- mean Gamma_rec: {_fmt(inflation_certificates['mean_Gamma_rec'])}",
        f"- interpretation: {inflation_certificates['interpretation']}",
        "",
        "## OPH Finite Certificate Authority",
        "",
        f"- finite certificate bundles: {finite_certificates['run_count']}",
        f"- compiler-ready bundles: {finite_certificates['compiler_ready_count']}",
        f"- legacy stack-ready bundles: {finite_certificates['stack_ready_count']}",
        f"- theorem-grade finite-input bundles: {finite_certificates['theorem_grade_finite_inputs_count']}",
        f"- proxy-certificate bundles: {finite_certificates['proxy_certificate_count']}",
        f"- no-data-use receipts: {finite_certificates['no_data_use_count']}",
        f"- release-code gates: {finite_certificates['release_code_gate_count']}",
        f"- parent-collar gates: {finite_certificates['parent_collar_gate_count']}",
        f"- repair-matrix gates: {finite_certificates['repair_matrix_gate_count']}",
        f"- Boltzmann-export gates: {finite_certificates['boltzmann_export_gate_count']}",
        f"- real-physics certificate bundles: {finite_certificates['real_physics_certificate_count']}",
        f"- physical-CMB prediction reports: {finite_certificates['physical_cmb_prediction_count']}",
        f"- mean A_zeta: {_fmt(finite_certificates['mean_A_zeta'])}",
        f"- mean n_s: {_fmt(finite_certificates['mean_n_s'])}",
        f"- mean Q_A: {_fmt(finite_certificates['mean_Q_A'])}",
        f"- mean B_A: {_fmt(finite_certificates['mean_B_A'])}",
        f"- mean Gamma_rec: {_fmt(finite_certificates['mean_Gamma_rec'])}",
        f"- interpretation: {finite_certificates['interpretation']}",
        "",
        "## OPH Scalar Geometric Quotient",
        "",
        f"- scalar quotient reports: {scalar_quotient['run_count']}",
        f"- scalar quotient receipts: {scalar_quotient['scalar_quotient_receipt_count']}",
        f"- finite-ready scalar releases: {scalar_quotient['finite_ready_count']}",
        f"- active 33-level clauses: {scalar_quotient['active_33_level_clause_count']}",
        f"- theorem-grade release codes: {scalar_quotient['theorem_grade_release_count']}",
        f"- physical-CMB prediction reports: {scalar_quotient['physical_cmb_prediction_count']}",
        f"- mean n_s: {_fmt(scalar_quotient['mean_n_s'])}",
        f"- mean observer level proxy: {_fmt(scalar_quotient['mean_observer_level_proxy'])}",
        f"- mean patch capacity level proxy: {_fmt(scalar_quotient['mean_patch_capacity_level_proxy'])}",
        f"- interpretation: {scalar_quotient['interpretation']}",
        "",
        "## Neutral Distance Profile Audit",
        "",
        f"- profile audit reports: {neutral_profile['run_count']}",
        f"- strict-ready profile count: {neutral_profile['strict_3d_ready_count']}",
        f"- all-profile best models: {neutral_profile['all_observer_visible_best_model_counts']}",
        f"- scalar-response best models: {neutral_profile['scalar_response_best_model_counts']}",
        f"- prime-geometric best models: {neutral_profile['prime_geometric_best_model_counts']}",
        f"- prime control-quotient best models: {neutral_profile['prime_control_quotient_best_model_counts']}",
        f"- prime rank-3 best models: {neutral_profile['prime_rank3_best_model_counts']}",
        f"- prime rank-8 best models: {neutral_profile['prime_rank8_best_model_counts']}",
        f"- prime control-quotient rank-3 best models: {neutral_profile['prime_control_quotient_rank3_best_model_counts']}",
        f"- prime control-quotient rank-8 best models: {neutral_profile['prime_control_quotient_rank8_best_model_counts']}",
        f"- support-visible best models: {neutral_profile['support_visible_best_model_counts']}",
        f"- mean all-profile corr/local-MLE dimension: {_fmt(neutral_profile['mean_all_corr_dim'])} / {_fmt(neutral_profile['mean_all_mle_dim'])}",
        f"- mean all-profile S2 leakage corr: {_fmt(neutral_profile['mean_all_s2_leakage_corr'])}",
        f"- mean scalar-response corr/local-MLE dimension: {_fmt(neutral_profile['mean_scalar_response_corr_dim'])} / {_fmt(neutral_profile['mean_scalar_response_mle_dim'])}",
        f"- mean scalar-response S2 leakage corr: {_fmt(neutral_profile['mean_scalar_response_s2_leakage_corr'])}",
        f"- mean prime-geometric corr/local-MLE dimension: {_fmt(neutral_profile['mean_prime_geometric_corr_dim'])} / {_fmt(neutral_profile['mean_prime_geometric_mle_dim'])}",
        f"- mean prime-geometric S2 leakage corr: {_fmt(neutral_profile['mean_prime_geometric_s2_leakage_corr'])}",
        f"- mean prime control-quotient corr/local-MLE dimension: {_fmt(neutral_profile['mean_prime_control_quotient_corr_dim'])} / {_fmt(neutral_profile['mean_prime_control_quotient_mle_dim'])}",
        f"- mean prime control-quotient S2 leakage corr: {_fmt(neutral_profile['mean_prime_control_quotient_s2_leakage_corr'])}",
        f"- mean prime rank-3 corr/local-MLE dimension: {_fmt(neutral_profile['mean_prime_rank3_corr_dim'])} / {_fmt(neutral_profile['mean_prime_rank3_mle_dim'])}",
        f"- mean prime rank-3 S2 leakage corr: {_fmt(neutral_profile['mean_prime_rank3_s2_leakage_corr'])}",
        f"- mean prime control-quotient rank-3 corr/local-MLE dimension: {_fmt(neutral_profile['mean_prime_control_quotient_rank3_corr_dim'])} / {_fmt(neutral_profile['mean_prime_control_quotient_rank3_mle_dim'])}",
        f"- mean prime control-quotient rank-3 S2 leakage corr: {_fmt(neutral_profile['mean_prime_control_quotient_rank3_s2_leakage_corr'])}",
        f"- mean prime rank-8 corr/local-MLE dimension: {_fmt(neutral_profile['mean_prime_rank8_corr_dim'])} / {_fmt(neutral_profile['mean_prime_rank8_mle_dim'])}",
        f"- mean prime rank-8 S2 leakage corr: {_fmt(neutral_profile['mean_prime_rank8_s2_leakage_corr'])}",
        f"- mean prime control-quotient rank-8 corr/local-MLE dimension: {_fmt(neutral_profile['mean_prime_control_quotient_rank8_corr_dim'])} / {_fmt(neutral_profile['mean_prime_control_quotient_rank8_mle_dim'])}",
        f"- mean prime control-quotient rank-8 S2 leakage corr: {_fmt(neutral_profile['mean_prime_control_quotient_rank8_s2_leakage_corr'])}",
        f"- mean support-visible corr/local-MLE dimension: {_fmt(neutral_profile['mean_support_visible_corr_dim'])} / {_fmt(neutral_profile['mean_support_visible_mle_dim'])}",
        f"- mean support-visible S2 leakage corr: {_fmt(neutral_profile['mean_support_visible_s2_leakage_corr'])}",
        f"- interpretation: {neutral_profile['interpretation']}",
        "",
        "## Prime-Geometric Rank Sweep",
        "",
        f"- rank-sweep reports: {prime_rank_sweep['run_count']}",
        f"- quotient 3D diagnostic receipts: {prime_rank_sweep['quotient_3d_diagnostic_receipt_count']}",
        f"- spatial 3D candidate receipts: {prime_rank_sweep['spatial_3d_candidate_receipt_count']}",
        f"- control-quotient spatial 3D candidate receipts: {prime_rank_sweep['control_quotient_spatial_3d_candidate_receipt_count']}",
        f"- strict neutral candidate receipts: {prime_rank_sweep['strict_neutral_candidate_receipt_count']}",
        f"- selected-rank controls written: {prime_rank_sweep['selected_controls_written_count']}",
        f"- selected-rank controls all failed: {prime_rank_sweep['selected_controls_all_failed_count']}",
        f"- coordinate rank-3 tautology warnings: {prime_rank_sweep['coordinate_rank3_tautology_warning_count']}",
        f"- selected directional control survive/fail counts: {prime_rank_sweep['selected_directional_control_survive_count']} / {prime_rank_sweep['selected_directional_control_fail_count']}",
        f"- selected coordinate control survive/fail counts: {prime_rank_sweep['selected_coordinate_control_survive_count']} / {prime_rank_sweep['selected_coordinate_control_fail_count']}",
        f"- strict-ready ranks: {prime_rank_sweep['strict_3d_ready_count']}",
        f"- dimension-window ranks: {prime_rank_sweep['dimension_3d_window_count']}",
        f"- best-rank counts: {prime_rank_sweep['best_rank_counts']}",
        f"- best-model counts: {prime_rank_sweep['best_model_counts']}",
        f"- mean best-rank corr/local-MLE dimension: {_fmt(prime_rank_sweep['mean_best_corr_dim'])} / {_fmt(prime_rank_sweep['mean_best_mle_dim'])}",
        f"- mean best-rank S2 leakage corr: {_fmt(prime_rank_sweep['mean_best_s2_leakage_corr'])}",
        f"- best 3D-window ranks: {prime_rank_sweep['best_3d_rank_counts']}",
        f"- best 3D-window models: {prime_rank_sweep['best_3d_model_counts']}",
        f"- mean best 3D-window corr/local-MLE dimension: {_fmt(prime_rank_sweep['mean_best_3d_corr_dim'])} / {_fmt(prime_rank_sweep['mean_best_3d_mle_dim'])}",
        f"- coordinate spatial-3D-ready ranks: {prime_rank_sweep['coordinate_spatial_3d_ready_count']}",
        f"- coordinate dimension-window ranks: {prime_rank_sweep['coordinate_dimension_3d_window_count']}",
        f"- coordinate best-rank counts: {prime_rank_sweep['coordinate_best_rank_counts']}",
        f"- coordinate best-model counts: {prime_rank_sweep['coordinate_best_model_counts']}",
        f"- coordinate mean best-rank corr/local-MLE dimension: {_fmt(prime_rank_sweep['coordinate_mean_best_corr_dim'])} / {_fmt(prime_rank_sweep['coordinate_mean_best_mle_dim'])}",
        f"- coordinate mean best-rank S2 leakage corr: {_fmt(prime_rank_sweep['coordinate_mean_best_s2_leakage_corr'])}",
        f"- coordinate best 3D-window ranks: {prime_rank_sweep['coordinate_best_3d_rank_counts']}",
        f"- coordinate best 3D-window models: {prime_rank_sweep['coordinate_best_3d_model_counts']}",
        f"- coordinate mean best 3D-window corr/local-MLE dimension: {_fmt(prime_rank_sweep['coordinate_mean_best_3d_corr_dim'])} / {_fmt(prime_rank_sweep['coordinate_mean_best_3d_mle_dim'])}",
        f"- control quotient marked as negative control: {prime_rank_sweep['control_quotient_negative_control_counts']}",
        f"- control-quotient strict-ready ranks: {prime_rank_sweep['control_quotient_strict_3d_ready_count']}",
        f"- control-quotient dimension-window ranks: {prime_rank_sweep['control_quotient_dimension_3d_window_count']}",
        f"- control-quotient best-rank counts: {prime_rank_sweep['control_quotient_best_rank_counts']}",
        f"- control-quotient best-model counts: {prime_rank_sweep['control_quotient_best_model_counts']}",
        f"- control-quotient mean best-rank corr/local-MLE dimension: {_fmt(prime_rank_sweep['control_quotient_mean_best_corr_dim'])} / {_fmt(prime_rank_sweep['control_quotient_mean_best_mle_dim'])}",
        f"- control-quotient mean best-rank S2 leakage corr: {_fmt(prime_rank_sweep['control_quotient_mean_best_s2_leakage_corr'])}",
        f"- control-quotient coordinate spatial-3D-ready ranks: {prime_rank_sweep['control_quotient_coordinate_spatial_3d_ready_count']}",
        f"- control-quotient coordinate dimension-window ranks: {prime_rank_sweep['control_quotient_coordinate_dimension_3d_window_count']}",
        f"- control-quotient coordinate best-rank counts: {prime_rank_sweep['control_quotient_coordinate_best_rank_counts']}",
        f"- control-quotient coordinate best-model counts: {prime_rank_sweep['control_quotient_coordinate_best_model_counts']}",
        f"- control-quotient coordinate mean best-rank corr/local-MLE dimension: {_fmt(prime_rank_sweep['control_quotient_coordinate_mean_best_corr_dim'])} / {_fmt(prime_rank_sweep['control_quotient_coordinate_mean_best_mle_dim'])}",
        f"- control-quotient coordinate mean best-rank S2 leakage corr: {_fmt(prime_rank_sweep['control_quotient_coordinate_mean_best_s2_leakage_corr'])}",
        f"- control-quotient coordinate best 3D-window ranks: {prime_rank_sweep['control_quotient_coordinate_best_3d_rank_counts']}",
        f"- control-quotient coordinate best 3D-window models: {prime_rank_sweep['control_quotient_coordinate_best_3d_model_counts']}",
        f"- control-quotient coordinate mean best 3D-window corr/local-MLE dimension: {_fmt(prime_rank_sweep['control_quotient_coordinate_mean_best_3d_corr_dim'])} / {_fmt(prime_rank_sweep['control_quotient_coordinate_mean_best_3d_mle_dim'])}",
        f"- control-quotient coordinate mean best 3D-window S2 leakage corr: {_fmt(prime_rank_sweep['control_quotient_coordinate_mean_best_3d_s2_leakage_corr'])}",
        f"- control-quotient coordinate best 3D-window S2 leakage-pass count: {prime_rank_sweep['control_quotient_coordinate_best_3d_s2_leakage_pass_count']}",
        f"- interpretation: {prime_rank_sweep['interpretation']}",
        "",
        "## Prime-Geometric Rank Refinement",
        "",
        f"- refinement reports: {prime_rank_refinement['run_count']}",
        f"- source run count mean: {_fmt(prime_rank_refinement['mean_source_run_count'])}",
        f"- multi-scale reports: {prime_rank_refinement['multi_scale_count']}",
        f"- rank-3 refinement candidate receipts: {prime_rank_refinement['candidate_receipt_count']}",
        f"- strict neutral refinement receipts: {prime_rank_refinement['strict_neutral_receipt_count']}",
        f"- all-candidate reports: {prime_rank_refinement['all_candidate_count']}",
        f"- all-candidate S2-leakage-pass reports: {prime_rank_refinement['all_candidate_s2_leakage_pass_count']}",
        f"- all-candidate rank-3/E3 reports: {prime_rank_refinement['all_candidate_rank3_e3_count']}",
        f"- dimension-stable reports: {prime_rank_refinement['dimension_stable_count']}",
        f"- independent rank-3 selector reports: {prime_rank_refinement['independent_rank3_all_count']}",
        f"- mean/max candidate dimension drift: {_fmt(prime_rank_refinement['mean_dimension_drift'])} / {_fmt(prime_rank_refinement['max_dimension_drift'])}",
        f"- patch-count sets: {prime_rank_refinement['patch_count_sets']}",
        f"- size median dimension sets: {prime_rank_refinement['size_median_dimension_sets']}",
        f"- blocker counts: {prime_rank_refinement['blocker_counts']}",
        f"- interpretation: {prime_rank_refinement['interpretation']}",
        "",
        "## OPH Parent-Collar Recovery Ladder",
        "",
        f"- parent-collar ladder reports: {parent_collar['run_count']}",
        f"- compiler-ready reports: {parent_collar['compiler_ready_count']}",
        f"- regulator-ladder-ready reports: {parent_collar['regulator_ready_count']}",
        f"- scaling-pass reports: {parent_collar['scaling_pass_count']}",
        f"- local recovery-density reports: {parent_collar['local_density_receipt_count']}",
        f"- strict local recovery-density reports: {parent_collar['strict_local_density_receipt_count']}",
        f"- theorem-grade reports: {parent_collar['theorem_grade_count']}",
        f"- strict cap-family reports: {parent_collar['strict_cap_family_count']}",
        f"- unique theta-family reports: {parent_collar['unique_theta_family_count']}",
        f"- physical-CMB prediction reports: {parent_collar['physical_cmb_prediction_count']}",
        f"- mean collar report count: {_fmt(parent_collar['mean_report_count'])}",
        f"- mean cap-row count: {_fmt(parent_collar['mean_row_count'])}",
        f"- mean regulator patch-count count: {_fmt(parent_collar['mean_patch_count_count'])}",
        f"- mean log p90-CMI vs log-N slope: {_fmt(parent_collar['mean_log_cmi_slope'])}",
        f"- mean log p90-CMI/collar vs log-N slope: {_fmt(parent_collar['mean_log_cmi_density_slope'])}",
        f"- mean final p90 epsilon_CMI: {_fmt(parent_collar['mean_final_p90_epsilon_cmi'])}",
        f"- mean final p90 epsilon_CMI/collar: {_fmt(parent_collar['mean_final_p90_epsilon_cmi_per_collar_patch'])}",
        f"- mean final p90 r_FR: {_fmt(parent_collar['mean_final_p90_r_fr'])}",
        f"- interpretation: {parent_collar['interpretation']}",
        "",
        "## OPH B_A Parent Finite Difference",
        "",
        f"- B_A parent reports: {b_a_parent['run_count']}",
        f"- B_A parent receipts: {b_a_parent['receipt_count']}",
        f"- paired finite diagnostic receipts: {b_a_parent['paired_diagnostic_receipt_count']}",
        f"- physical-prediction-ready reports: {b_a_parent['physical_prediction_ready_count']}",
        f"- physical-CMB prediction reports: {b_a_parent['physical_cmb_prediction_count']}",
        f"- controls-fail reports: {b_a_parent['controls_fail_count']}",
        f"- real baryon perturbation rerun reports: {b_a_parent['real_baryon_perturbation_run_count']}",
        f"- finite observer-view parent variation reports: {b_a_parent['finite_observer_view_parent_variation_count']}",
        f"- refinement convergence reports: {b_a_parent['refinement_convergence_passed_count']}",
        f"- primary parent source counts: {b_a_parent['primary_parent_source_counts']}",
        f"- mean source report count: {_fmt(b_a_parent['mean_source_report_count'])}",
        f"- mean observer-view source count: {_fmt(b_a_parent['mean_observer_view_source_count'])}",
        f"- mean B_A rows: {_fmt(b_a_parent['mean_row_count'])}",
        f"- mean B_A control rows: {_fmt(b_a_parent['mean_control_row_count'])}",
        f"- control failure counts: {b_a_parent['control_failure_counts']}",
        f"- missing gate counts: {b_a_parent['missing_gate_counts']}",
        f"- interpretation: {b_a_parent['interpretation']}",
        "",
        "## OPH Screen-Capacity Closure",
        "",
        f"- screen-capacity reports: {screen_capacity['run_count']}",
        f"- observed-branch readout reports: {screen_capacity['observed_readout_count']}",
        f"- F(N) implemented reports: {screen_capacity['F_N_implemented_count']}",
        f"- finite-simulator fixed-point solved reports: {screen_capacity['fixed_point_solved_count']}",
        f"- physical-CMB prediction reports: {screen_capacity['physical_cmb_prediction_count']}",
        f"- input-mode counts: {screen_capacity['input_mode_counts']}",
        f"- mean N_patch bare ratio: {_fmt(screen_capacity['mean_N_patch_bare_ratio'])}",
        f"- mean N_scr entropy capacity: {_fmt(screen_capacity['mean_N_scr'])}",
        f"- mean Lambda l_P^2: {_fmt(screen_capacity['mean_Lambda_lP2'])}",
        f"- mean local P-cell count for N_scr: {_fmt(screen_capacity['mean_P_cell_count'])}",
        f"- interpretation: {screen_capacity['interpretation']}",
        "",
        "## OPH Repair-Scale Closure",
        "",
        f"- repair-scale reports: {repair_scale['run_count']}",
        f"- numeric match within 1% reports: {repair_scale['numeric_match_within_1_percent_count']}",
        f"- declared 24-round reports: {repair_scale['twenty_four_rounds_declared_count']}",
        f"- finite-selector-derived 24-round reports: {repair_scale['twenty_four_rounds_derived_count']}",
        f"- finite-lattice-derived eta_R reports: {repair_scale['finite_lattice_derived_eta_R_count']}",
        f"- physical-CMB prediction reports: {repair_scale['physical_cmb_prediction_count']}",
        f"- populated-3D-bulk reports: {repair_scale['bulk_3d_established_count']}",
        f"- mean |g'(P)|: {_fmt(repair_scale['mean_gprime'])}",
        f"- mean q per round: {_fmt(repair_scale['mean_q_round'])}",
        f"- mean N_CRC implied by repair-depth ansatz: {_fmt(repair_scale['mean_N_implied_by_ansatz'])}",
        f"- mean declared N_CRC: {_fmt(repair_scale['mean_N_CRC'])}",
        f"- mean n_s = 1-P/48: {_fmt(repair_scale['mean_n_s'])}",
        f"- mean 1M effective repair-round depth: {_fmt(repair_scale['mean_1m_effective_round_depth'])}",
        f"- interpretation: {repair_scale['interpretation']}",
        "",
        "## OPH Scale-Compressed Repair Branch",
        "",
        f"- compressed branch reports: {scale_compressed['run_count']}",
        f"- operator receipts: {scale_compressed['operator_receipt_count']}",
        f"- round-trace receipts: {scale_compressed['round_trace_receipt_count']}",
        f"- cap-profile H3 receipts: {scale_compressed['cap_profile_receipt_count']}",
        f"- populated-H3 preview reports: {scale_compressed['populated_h3_preview_count']}",
        f"- strict neutral bulk reports: {scale_compressed['strict_neutral_bulk_count']}",
        f"- particle preview reports: {scale_compressed['particle_preview_count']}",
        f"- production particle-matter reports: {scale_compressed['production_particle_matter_count']}",
        f"- physical-CMB prediction reports: {scale_compressed['physical_cmb_prediction_count']}",
        f"- mean rounds: {_fmt(scale_compressed['mean_rounds'])}",
        f"- mean eta_R: {_fmt(scale_compressed['mean_eta_R'])}",
        f"- mean n_s: {_fmt(scale_compressed['mean_n_s'])}",
        f"- mean q_IR: {_fmt(scale_compressed['mean_q_IR'])}",
        f"- mean ell_IR: {_fmt(scale_compressed['mean_ell_IR'])}",
        f"- mean object count: {_fmt(scale_compressed['mean_object_count'])}",
        f"- mean particle worldline count: {_fmt(scale_compressed['mean_particle_worldline_count'])}",
        f"- interpretation: {scale_compressed['interpretation']}",
        "",
        "## OPH Scale-Compressed CMB CAMB Transfer",
        "",
        f"- CAMB transfer reports: {scale_compressed_camb['run_count']}",
        f"- measurement-comparable curve reports: {scale_compressed_camb['measurement_comparable_curve_count']}",
        f"- transfer receipts: {scale_compressed_camb['transfer_receipt_count']}",
        f"- operator receipts: {scale_compressed_camb['operator_receipt_count']}",
        f"- H3 preview receipts: {scale_compressed_camb['h3_preview_receipt_count']}",
        f"- physical-CMB prediction reports: {scale_compressed_camb['physical_cmb_prediction_count']}",
        f"- mean rounds: {_fmt(scale_compressed_camb['mean_rounds'])}",
        f"- mean eta_R: {_fmt(scale_compressed_camb['mean_eta_R'])}",
        f"- mean n_s: {_fmt(scale_compressed_camb['mean_n_s'])}",
        f"- mean q_IR: {_fmt(scale_compressed_camb['mean_q_IR'])}",
        f"- mean ell_IR: {_fmt(scale_compressed_camb['mean_ell_IR'])}",
        f"- mean LCDM shape correlation: {_fmt(scale_compressed_camb['mean_lcdm_shape_correlation'])}",
        f"- mean scale-compressed scalar shape correlation: {_fmt(scale_compressed_camb['mean_scalar_shape_correlation'])}",
        f"- mean scale-compressed IR shape correlation: {_fmt(scale_compressed_camb['mean_ir_shape_correlation'])}",
        f"- mean LCDM chi2/bin: {_fmt(scale_compressed_camb['mean_lcdm_chi2_per_bin'])}",
        f"- mean scale-compressed scalar chi2/bin: {_fmt(scale_compressed_camb['mean_scalar_chi2_per_bin'])}",
        f"- mean scale-compressed IR chi2/bin: {_fmt(scale_compressed_camb['mean_ir_chi2_per_bin'])}",
        f"- mean acoustic |delta|: {_fmt(scale_compressed_camb['mean_acoustic_abs_delta'])}",
        f"- interpretation: {scale_compressed_camb['interpretation']}",
        "",
        "## OPH Inflation/CMB CAMB Transfer",
        "",
        f"- CAMB transfer reports: {inflation_cmb_camb['run_count']}",
        f"- measurement-comparable curve reports: {inflation_cmb_camb['measurement_comparable_curve_count']}",
        f"- transfer receipts: {inflation_cmb_camb['transfer_receipt_count']}",
        f"- physical-CMB prediction reports: {inflation_cmb_camb['physical_cmb_prediction_count']}",
        f"- mean n_s: {_fmt(inflation_cmb_camb['mean_n_s'])}",
        f"- mean A_zeta: {_fmt(inflation_cmb_camb['mean_A_zeta'])}",
        f"- mean q_IR: {_fmt(inflation_cmb_camb['mean_q_IR'])}",
        f"- mean ell_IR: {_fmt(inflation_cmb_camb['mean_ell_IR'])}",
        f"- mean LCDM high-ell shape correlation: {_fmt(inflation_cmb_camb['mean_lcdm_shape_correlation'])}",
        f"- mean OPH P/48 high-ell shape correlation: {_fmt(inflation_cmb_camb['mean_p48_shape_correlation'])}",
        f"- mean OPH P/48+IR high-ell shape correlation: {_fmt(inflation_cmb_camb['mean_p48_ir_shape_correlation'])}",
        f"- mean LCDM high-ell chi2/bin: {_fmt(inflation_cmb_camb['mean_lcdm_chi2_per_bin'])}",
        f"- mean OPH P/48 high-ell chi2/bin: {_fmt(inflation_cmb_camb['mean_p48_chi2_per_bin'])}",
        f"- mean OPH P/48+IR high-ell chi2/bin: {_fmt(inflation_cmb_camb['mean_p48_ir_chi2_per_bin'])}",
        f"- mean acoustic |delta|: {_fmt(inflation_cmb_camb['mean_acoustic_abs_delta'])}",
        f"- imported low-ell LCDM chi2: {_fmt(inflation_cmb_camb['mean_lowell_lcdm_chi2'])}",
        f"- imported low-ell OPH-IR chi2: {_fmt(inflation_cmb_camb['mean_lowell_oph_ir_chi2'])}",
        f"- mean unique v0.9 n_s: {_fmt(inflation_cmb_camb['mean_unique_n_s'])}",
        f"- mean unique v0.9 q_IR: {_fmt(inflation_cmb_camb['mean_unique_q_IR'])}",
        f"- mean unique v0.9 ell_IR: {_fmt(inflation_cmb_camb['mean_unique_ell_IR'])}",
        f"- mean unique v0.9 high-ell shape correlation: {_fmt(inflation_cmb_camb['mean_unique_shape_correlation'])}",
        f"- mean unique v0.9 chi2/bin: {_fmt(inflation_cmb_camb['mean_unique_chi2_per_bin'])}",
        f"- mean unique v0.9 acoustic |delta|: {_fmt(inflation_cmb_camb['mean_unique_acoustic_abs_delta'])}",
        f"- interpretation: {inflation_cmb_camb['interpretation']}",
        "",
        "## OPH CMB Selector Elimination v1.5",
        "",
        f"- selector-elimination reports: {selector_elimination['run_count']}",
        f"- theorem-side receipts: {selector_elimination['theorem_side_receipt_count']}",
        f"- source-packet audit receipts: {selector_elimination['source_packet_audit_receipt_count']}",
        f"- q_IR selector-removed count: {selector_elimination['q_IR_selector_removed_count']}",
        f"- ell_IR selector-removed count: {selector_elimination['ell_IR_selector_removed_count']}",
        f"- eta_R repair-clock reduction count: {selector_elimination['eta_R_repair_clock_reduction_count']}",
        f"- finite-lattice-derived reports: {selector_elimination['finite_lattice_derived_count']}",
        f"- physical-CMB prediction reports: {selector_elimination['physical_cmb_prediction_count']}",
        f"- exact-kernel CSV pass count: {selector_elimination['kernel_csv_passed_count']}",
        f"- mean exact-kernel CSV max error: {_fmt(selector_elimination['mean_kernel_csv_max_abs_error'])}",
        f"- mean n_s: {_fmt(selector_elimination['mean_n_s'])}",
        f"- mean eta_R: {_fmt(selector_elimination['mean_eta_R'])}",
        f"- mean q_IR: {_fmt(selector_elimination['mean_q_IR'])}",
        f"- mean ell_IR: {_fmt(selector_elimination['mean_ell_IR'])}",
        f"- kappa_rep status counts: {selector_elimination['kappa_rep_status_counts']}",
        f"- interpretation: {selector_elimination['interpretation']}",
        "",
        "## OPH Exact CMB CAMB Transfer",
        "",
        f"- CAMB transfer reports: {exact_cmb_camb['run_count']}",
        f"- measurement-comparable curve reports: {exact_cmb_camb['measurement_comparable_curve_count']}",
        f"- transfer receipts: {exact_cmb_camb['transfer_receipt_count']}",
        f"- physical-CMB prediction reports: {exact_cmb_camb['physical_cmb_prediction_count']}",
        f"- official clik-ready reports: {exact_cmb_camb['official_clik_ready_count']}",
        f"- official likelihood-ready reports: {exact_cmb_camb['official_likelihood_ready_count']}",
        f"- embedded selector theorem receipts: {exact_cmb_camb['selector_theorem_receipt_count']}",
        f"- embedded selector source-audit receipts: {exact_cmb_camb['selector_source_audit_receipt_count']}",
        f"- mean exact n_s: {_fmt(exact_cmb_camb['mean_n_s'])}",
        f"- mean exact eta_R: {_fmt(exact_cmb_camb['mean_eta_R'])}",
        f"- mean exact q_IR: {_fmt(exact_cmb_camb['mean_q_IR'])}",
        f"- mean exact ell_IR: {_fmt(exact_cmb_camb['mean_ell_IR'])}",
        f"- mean exact N_frz proxy: {_fmt(exact_cmb_camb['mean_N_frz_proxy'])}",
        f"- mean LCDM shape correlation: {_fmt(exact_cmb_camb['mean_lcdm_shape_correlation'])}",
        f"- mean exact scalar shape correlation: {_fmt(exact_cmb_camb['mean_scalar_shape_correlation'])}",
        f"- mean exact IR shape correlation: {_fmt(exact_cmb_camb['mean_ir_shape_correlation'])}",
        f"- mean LCDM chi2/bin: {_fmt(exact_cmb_camb['mean_lcdm_chi2_per_bin'])}",
        f"- mean exact scalar chi2/bin: {_fmt(exact_cmb_camb['mean_scalar_chi2_per_bin'])}",
        f"- mean exact IR chi2/bin: {_fmt(exact_cmb_camb['mean_ir_chi2_per_bin'])}",
        f"- mean acoustic |delta|: {_fmt(exact_cmb_camb['mean_acoustic_abs_delta'])}",
        f"- interpretation: {exact_cmb_camb['interpretation']}",
        "",
        "## Finite Repair-Clock CMB CAMB Transfer",
        "",
        f"- CAMB transfer reports: {finite_clock_cmb['run_count']}",
        f"- measurement-comparable curve reports: {finite_clock_cmb['measurement_comparable_curve_count']}",
        f"- transfer receipts: {finite_clock_cmb['transfer_receipt_count']}",
        f"- finite-lattice clock reports: {finite_clock_cmb['finite_lattice_clock_derived_count']}",
        f"- exact repair-clock certificate reports: {finite_clock_cmb['repair_clock_certificate_count']}",
        f"- selector IR theory-side reports: {finite_clock_cmb['selector_ir_theory_side_count']}",
        f"- physical-CMB prediction reports: {finite_clock_cmb['physical_cmb_prediction_count']}",
        f"- mean finite-clock n_s: {_fmt(finite_clock_cmb['mean_n_s'])}",
        f"- mean finite-clock eta_R: {_fmt(finite_clock_cmb['mean_eta_R'])}",
        f"- mean finite-clock kappa_rep: {_fmt(finite_clock_cmb['mean_kappa_rep'])}",
        f"- mean q_IR: {_fmt(finite_clock_cmb['mean_q_IR'])}",
        f"- mean ell_IR: {_fmt(finite_clock_cmb['mean_ell_IR'])}",
        f"- mean finite-clock scalar shape correlation: {_fmt(finite_clock_cmb['mean_scalar_shape_correlation'])}",
        f"- mean finite-clock IR shape correlation: {_fmt(finite_clock_cmb['mean_ir_shape_correlation'])}",
        f"- mean finite-clock scalar chi2/bin: {_fmt(finite_clock_cmb['mean_scalar_chi2_per_bin'])}",
        f"- mean finite-clock IR chi2/bin: {_fmt(finite_clock_cmb['mean_ir_chi2_per_bin'])}",
        f"- mean acoustic |delta|: {_fmt(finite_clock_cmb['mean_acoustic_abs_delta'])}",
        f"- interpretation: {finite_clock_cmb['interpretation']}",
        "",
        "## OPH Unique Prediction Gate v0.9",
        "",
        f"- target reports: {unique_prediction['run_count']}",
        f"- measurement-comparable target reports: {unique_prediction['measurement_comparable_count']}",
        f"- finite-lattice-derived target reports: {unique_prediction['finite_lattice_derived_count']}",
        f"- physical-CMB prediction reports: {unique_prediction['physical_cmb_prediction_count']}",
        f"- mean n_s: {_fmt(unique_prediction['mean_n_s'])}",
        f"- mean eta_R: {_fmt(unique_prediction['mean_eta_R'])}",
        f"- mean Planck n_s pull: {_fmt(unique_prediction['mean_n_s_pull'])}",
        f"- mean q_IR: {_fmt(unique_prediction['mean_q_IR'])}",
        f"- mean ell_IR: {_fmt(unique_prediction['mean_ell_IR'])}",
        f"- mean N_frz proxy: {_fmt(unique_prediction['mean_N_frz_proxy'])}",
        f"- mean parity R_OE TT(2..29): {_fmt(unique_prediction['mean_parity_R_OE_TT_2_29'])}",
        f"- mean sum m_nu eV: {_fmt(unique_prediction['mean_sum_mnu_eV'])}",
        f"- mean neutrino f_nu: {_fmt(unique_prediction['mean_neutrino_f_nu'])}",
        f"- mean small-scale neutrino suppression: {_fmt(unique_prediction['mean_small_scale_neutrino_suppression'])}",
        f"- interpretation: {unique_prediction['interpretation']}",
        "",
        "## OPH-CnuB Neutrino Background",
        "",
        f"- neutrino reports: {cnb_neutrino['run_count']}",
        f"- measurement-comparable reports: {cnb_neutrino['measurement_comparable_count']}",
        f"- finite-lattice-derived reports: {cnb_neutrino['finite_lattice_derived_count']}",
        f"- physical-CMB prediction reports: {cnb_neutrino['physical_cmb_prediction_count']}",
        f"- physical matter-power prediction reports: {cnb_neutrino['physical_matter_power_prediction_count']}",
        f"- background/mass/B_A/likelihood gate counts: {cnb_neutrino['background_gate_count']} / {cnb_neutrino['mass_derivation_gate_count']} / {cnb_neutrino['B_A_kernel_gate_count']} / {cnb_neutrino['likelihood_gate_count']}",
        f"- five-of-seven kernel/projection counts: {cnb_neutrino['five_of_seven_kernel_callable_count']} / {cnb_neutrino['five_of_seven_projection_count']}",
        f"- Planck+BAO / ACT / DESI strict sum-mnu pass counts: {cnb_neutrino['planck_bao_bound_pass_count']} / {cnb_neutrino['act_bound_pass_count']} / {cnb_neutrino['desi_lcdm_bound_pass_count']}",
        f"- mean N_eff: {_fmt(cnb_neutrino['mean_N_eff'])}",
        f"- mean sum m_nu eV: {_fmt(cnb_neutrino['mean_sum_mnu_eV'])}",
        f"- mean lightest mass eV: {_fmt(cnb_neutrino['mean_m_lightest_eV'])}",
        f"- mean Omega_nu h2: {_fmt(cnb_neutrino['mean_Omega_nu_h2'])}",
        f"- mean f_nu: {_fmt(cnb_neutrino['mean_f_nu'])}",
        f"- mean small-scale suppression: {_fmt(cnb_neutrino['mean_small_scale_suppression'])}",
        f"- mean Planck N_eff pull sigma: {_fmt(cnb_neutrino['mean_planck_neff_pull_sigma'])}",
        f"- mean eta_A: {_fmt(cnb_neutrino['mean_eta_A'])}",
        f"- mean compressed Pi_WL target: {_fmt(cnb_neutrino['mean_Pi_WL_compressed_required'])}",
        f"- mean five-of-seven Pi_WL: {_fmt(cnb_neutrino['mean_five_of_seven_pi_wl'])}",
        f"- mean five-of-seven epsilon_A: {_fmt(cnb_neutrino['mean_five_of_seven_epsilon_A'])}",
        f"- mean five-of-seven projected S8: {_fmt(cnb_neutrino['mean_five_of_seven_S8_projected'])}",
        f"- mean five-of-seven S8 pull sigma: {_fmt(cnb_neutrino['mean_five_of_seven_pull_sigma'])}",
        f"- interpretation: {cnb_neutrino['interpretation']}",
        "",
        "## Finite Lattice CMB Derivation Audit",
        "",
        f"- derivation reports: {cmb_derivation['run_count']}",
        f"- ready reports: {cmb_derivation['ready_count']}",
        f"- mean source run count: {_fmt(cmb_derivation['mean_source_run_count'])}",
        f"- mean finite eta_R: {_fmt(cmb_derivation['mean_eta_R'])}",
        f"- mean eta_R absolute error: {_fmt(cmb_derivation['mean_eta_R_abs_error'])}",
        f"- mean scalar-quotient n_s: {_fmt(cmb_derivation['mean_scalar_quotient_n_s'])}",
        f"- mean scalar-quotient theta_OPH: {_fmt(cmb_derivation['mean_scalar_quotient_theta_OPH'])}",
        f"- mean scalar-quotient n_s target error: {_fmt(cmb_derivation['mean_scalar_quotient_n_s_abs_error'])}",
        f"- mean finite q_IR: {_fmt(cmb_derivation['mean_q_IR'])}",
        f"- mean finite ell_IR: {_fmt(cmb_derivation['mean_ell_IR'])}",
        f"- mean collar median CMI: {_fmt(cmb_derivation['mean_median_epsilon_cmi'])}",
        f"- interpretation: {cmb_derivation['interpretation']}",
        "",
        "## OPH Boltzmann Input Readouts",
        "",
        f"- input reports: {boltzmann['run_count']}",
        f"- CDM-limit-ready reports: {boltzmann['cdm_limit_ready_count']}",
        f"- diagnostic repair-table-ready reports: {boltzmann['diagnostic_repair_table_ready_count']}",
        f"- physical-prediction-ready reports: {boltzmann['physical_prediction_ready_count']}",
        f"- mean source report count: {_fmt(boltzmann['mean_source_report_count'])}",
        f"- mean CDM row count: {_fmt(boltzmann['mean_cdm_row_count'])}",
        f"- mean diagnostic row count: {_fmt(boltzmann['mean_diagnostic_row_count'])}",
        f"- mean missing gate count: {_fmt(boltzmann['mean_missing_gate_count'])}",
        f"- mean Gamma_rec/H shape proxy: {_fmt(boltzmann['mean_gamma_rec_over_H_shape_proxy'])}",
        f"- mean B_A shape proxy: {_fmt(boltzmann['mean_B_A_shape_proxy'])}",
        f"- interpretation: {boltzmann['interpretation']}",
        "",
        "## Finite-Collar Boltzmann Source Bundle",
        "",
        f"- source-bundle reports: {finite_collar_boltzmann['run_count']}",
        f"- diagnostic bundle receipts: {finite_collar_boltzmann['diagnostic_bundle_receipt_count']}",
        f"- physical export certificates: {finite_collar_boltzmann['physical_certificate_count']}",
        f"- physical-CMB prediction reports: {finite_collar_boltzmann['physical_cmb_prediction_count']}",
        f"- no-data-use reports: {finite_collar_boltzmann['no_data_use_count']}",
        f"- mean B_A rows: {_fmt(finite_collar_boltzmann['mean_B_A_row_count'])}",
        f"- mean rho_A rows: {_fmt(finite_collar_boltzmann['mean_rho_A_row_count'])}",
        f"- mean Gamma_rec rows: {_fmt(finite_collar_boltzmann['mean_Gamma_rec_row_count'])}",
        f"- mean physical missing gate count: {_fmt(finite_collar_boltzmann['mean_physical_missing_gate_count'])}",
        f"- mean |B_A|: {_fmt(finite_collar_boltzmann['mean_abs_B_A'])}",
        f"- mean |Gamma_rec/H|: {_fmt(finite_collar_boltzmann['mean_abs_Gamma_rec_over_H'])}",
        f"- diagnostic missing gate counts: {finite_collar_boltzmann['diagnostic_missing_gate_counts']}",
        f"- physical missing gate counts: {finite_collar_boltzmann['physical_missing_gate_counts']}",
        f"- interpretation: {finite_collar_boltzmann['interpretation']}",
        "",
        "## Finite-Collar CMB Projection",
        "",
        f"- projection reports: {finite_collar_projection['run_count']}",
        f"- projection receipts: {finite_collar_projection['projection_receipt_count']}",
        f"- physical-k calibration receipts: {finite_collar_projection['physical_k_receipt_count']}",
        f"- physical-CMB prediction reports: {finite_collar_projection['physical_cmb_prediction_count']}",
        f"- mean projected B_A rows: {_fmt(finite_collar_projection['mean_projected_B_A_row_count'])}",
        f"- mean background rows: {_fmt(finite_collar_projection['mean_background_row_count'])}",
        f"- mean ell range: {_fmt(finite_collar_projection['mean_ell_min'])} .. {_fmt(finite_collar_projection['mean_ell_max'])}",
        f"- mean B_A: {_fmt(finite_collar_projection['mean_B_A'])}",
        f"- mean |B_A|: {_fmt(finite_collar_projection['mean_abs_B_A'])}",
        f"- mean B_A positive fraction: {_fmt(finite_collar_projection['mean_positive_fraction'])}",
        f"- mean log |B_A| / log ell slope: {_fmt(finite_collar_projection['mean_log_abs_B_A_vs_log_ell_slope'])}",
        f"- mean largest-scale B_A: {_fmt(finite_collar_projection['mean_largest_scale_B_A'])}",
        f"- mean smallest-scale B_A: {_fmt(finite_collar_projection['mean_smallest_scale_B_A'])}",
        f"- interpretation: {finite_collar_projection['interpretation']}",
        "",
        "## Low-k Synchronization Gap",
        "",
        f"- gap audit reports: {sync_gap['run_count']}",
        f"- mean source run count: {_fmt(sync_gap['source_run_count'])}",
        f"- low-k gap established reports: {sync_gap['low_k_gap_established_count']}",
        f"- same-boundary selector reports: {sync_gap['same_boundary_selector_count']}",
        f"- inflation-replacement-ready reports: {sync_gap['inflation_replacement_ready_count']}",
        f"- mean time-resolved trace count: {_fmt(sync_gap['mean_time_resolved_trace_count'])}",
        f"- mean cached proxy pass count: {_fmt(sync_gap['mean_cached_proxy_pass_count'])}",
        f"- mean time-resolved gap pass count: {_fmt(sync_gap['mean_time_resolved_gap_pass_count'])}",
        f"- mean global Phi gamma/cycle: {_fmt(sync_gap['mean_global_phi_gamma_per_cycle'])}",
        f"- interpretation: {sync_gap['interpretation']}",
        "",
        "## Hot MaxEnt Release",
        "",
        f"- release audit reports: {hot_release['run_count']}",
        f"- mean source run count: {_fmt(hot_release['source_run_count'])}",
        f"- theorem-ready release reports: {hot_release['theorem_ready_report_count']}",
        f"- mean mechanical release surfaces: {_fmt(hot_release['mean_mechanical_surface_count'])}",
        f"- mean collar-gate passes: {_fmt(hot_release['mean_collar_gate_pass_count'])}",
        f"- mean theorem-ready source runs: {_fmt(hot_release['mean_theorem_ready_count'])}",
        f"- mean median release cycle: {_fmt(hot_release['mean_median_release_cycle'])}",
        f"- mean collar median CMI: {_fmt(hot_release['mean_median_epsilon_cmi'])}",
        f"- interpretation: {hot_release['interpretation']}",
        "",
        "## Same-Boundary Adiabaticity",
        "",
        f"- adiabaticity audit reports: {adiabaticity['run_count']}",
        f"- mean source run count: {_fmt(adiabaticity['source_run_count'])}",
        f"- established reports: {adiabaticity['established_report_count']}",
        f"- mean proxy pass count: {_fmt(adiabaticity['mean_proxy_pass_count'])}",
        f"- mean max entropy residual std: {_fmt(adiabaticity['mean_max_entropy_residual_std'])}",
        f"- mean min common-clock correlation: {_fmt(adiabaticity['mean_min_common_clock_corr'])}",
        f"- interpretation: {adiabaticity['interpretation']}",
        "",
        "## H0/S8 Branch Diagnostic",
        "",
        f"- H0/S8 reports: {h0s8['run_count']}",
        f"- physical-prediction-ready reports: {h0s8['physical_prediction_ready_count']}",
        f"- Q_A/B_A/Gamma_J gate counts: {h0s8['Q_A_gate_count']} / {h0s8['B_A_gate_count']} / {h0s8['Gamma_J_gate_count']}",
        f"- lambda(P) gate count: {h0s8['lambda_P_gate_count']}",
        f"- CAMB/CLASS and likelihood gate counts: {h0s8['CAMB_CLASS_gate_count']} / {h0s8['likelihood_gate_count']}",
        f"- Lane-8 certificate / run-derived / ready counts: {h0s8['lane8_certificate_count']} / {h0s8['lane8_run_derived_count']} / {h0s8['lane8_certificate_ready_count']}",
        f"- Lane-8 payload/fake/selector/refinement gate counts: {h0s8['lane8_payload_gate_count']} / {h0s8['lane8_fake_suppression_gate_count']} / {h0s8['lane8_selector_gate_count']} / {h0s8['lane8_refinement_gate_count']}",
        f"- mean H0 km/s/Mpc: {_fmt(h0s8['mean_H0_km_s_Mpc'])}",
        f"- mean Omega_m: {_fmt(h0s8['mean_Omega_m'])}",
        f"- mean Omega_A: {_fmt(h0s8['mean_Omega_A'])}",
        f"- mean lambda_collar: {_fmt(h0s8['mean_lambda_collar'])}",
        f"- mean source suppression fraction: {_fmt(h0s8['mean_source_suppression_fraction'])}",
        f"- mean CDM-like S8: {_fmt(h0s8['mean_cdm_like_S8'])}",
        f"- mean direct-Jacobi S8: {_fmt(h0s8['mean_direct_jacobi_S8'])}",
        f"- mean matrix-gap S8: {_fmt(h0s8['mean_matrix_gap_S8'])}",
        f"- mean Lane-8 I0 bits: {_fmt(h0s8['mean_lane8_i0_bits'])}",
        f"- mean Lane-8 low-entropy gap bits: {_fmt(h0s8['mean_lane8_low_entropy_gap_bits'])}",
        f"- mean Lane-8 fake gamma margin bits: {_fmt(h0s8['mean_lane8_fake_gamma_margin_bits'])}",
        f"- mean Planck H0 pull sigma: {_fmt(h0s8['mean_planck_H0_pull_sigma'])}",
        f"- mean SH0ES H0 pull sigma: {_fmt(h0s8['mean_shoes_H0_pull_sigma'])}",
        f"- mean weak-lensing CDM S8 pull sigma: {_fmt(h0s8['mean_weak_lensing_cdm_pull_sigma'])}",
        f"- interpretation: {h0s8['interpretation']}",
        "",
        "## OPH-CMB Anomaly-Stress Adapter",
        "",
        f"- adapter reports: {oph_cmb['run_count']}",
        f"- Boltzmann-ready reports: {oph_cmb['boltzmann_ready_count']}",
        f"- physical-prediction-ready reports: {oph_cmb['physical_prediction_ready_count']}",
        f"- cosmology perturbation receipts: {oph_cmb['cosmology_perturbation_receipt_count']}",
        f"- mean finite-collar sample count: {_fmt(oph_cmb['mean_parent_sample_count'])}",
        f"- mean weighted collar repair defect R: {_fmt(oph_cmb['mean_weighted_collar_repair_defect_R'])}",
        f"- mean rho_A_eq proxy: {_fmt(oph_cmb['mean_rho_A_eq_proxy'])}",
        f"- mean diagnostic kernel rows: {_fmt(oph_cmb['mean_kernel_proxy_row_count'])}",
        f"- mean missing gate count: {_fmt(oph_cmb['mean_missing_gate_count'])}",
        f"- interpretation: {oph_cmb['interpretation']}",
        "",
        "## Screen Holonomy Defect Proxy",
        "",
        f"- runs with holonomy reports: {hol['run_count']}",
        f"- mean defect fraction: {_fmt(hol['mean_defect_fraction'])}",
        f"- mean cluster count: {_fmt(hol['mean_cluster_count'])}",
        "",
        "## Defect Worldline / Particle Precursors",
        "",
        f"- runs with defect-worldline diagnostics: {defect_worldlines['run_count']}",
        f"- timeline reports: {defect_worldlines['timeline_run_count']}",
        f"- interaction reports: {defect_worldlines['interaction_run_count']}",
        f"- particle-likeness reports: {defect_worldlines['particle_likeness_run_count']}",
        f"- H3 worldline reports: {defect_worldlines['h3_worldline_run_count']}",
        f"- screen-worldline precursor receipts: {defect_worldlines['worldline_precursor_receipt_count']}",
        f"- interaction proxy receipts: {defect_worldlines['interaction_proxy_receipt_count']}",
        f"- H3 bulk-worldline precursor receipts: {defect_worldlines['h3_bulk_worldline_precursor_count']}",
        f"- particle-matter receipts: {defect_worldlines['particle_matter_receipt_count']}",
        f"- mean worldline count: {_fmt(defect_worldlines['mean_timeline_worldline_count'])}",
        f"- mean persistent worldline count: {_fmt(defect_worldlines['mean_persistent_worldline_count'])}",
        f"- mean max observation count: {_fmt(defect_worldlines['mean_max_observation_count'])}",
        f"- mean max lifetime cycles: {_fmt(defect_worldlines['mean_max_lifetime_cycles'])}",
        f"- mean screen transport proxy count: {_fmt(defect_worldlines['mean_screen_transport_proxy_count'])}",
        f"- mean fusion candidate count: {_fmt(defect_worldlines['mean_fusion_candidate_count'])}",
        f"- mean particle-like count: {_fmt(defect_worldlines['mean_particle_like_count'])}",
        "",
        "## Controlled Defect Particle Assay",
        "",
        f"- controlled assay reports: {controlled_defect['run_count']}",
        f"- inverse-identity passes: {controlled_defect['inverse_identity_pass_count']}",
        f"- interaction proxy receipts: {controlled_defect['interaction_proxy_receipt_count']}",
        f"- fusion-conservation passes: {controlled_defect['fusion_conservation_pass_count']}",
        f"- detector-positive receipts: {controlled_defect['detector_positive_count']}",
        f"- physical particle-emergence receipts: {controlled_defect['physical_particle_emergence_count']}",
        f"- mean controlled particle-like count: {_fmt(controlled_defect['mean_particle_like_count'])}",
        f"- interpretation: {controlled_defect['interpretation']}",
        "",
        "## Output Files",
        "",
        "- `comparable_data_snapshot.json`",
        "- `comparable_data_rows.csv`",
        "- `comparable_data_snapshot.md`",
        "",
    ]
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return {}
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _find_named_reports(root: Path, filenames: tuple[str, ...]) -> list[Path]:
    root = Path(root)
    reports: list[Path] = []
    if root.is_file():
        if root.name in filenames:
            reports.append(root)
        return reports
    for name in filenames:
        direct = root / name
        if direct.exists():
            reports.append(direct)
        if root.exists():
            reports.extend(root.glob(f"**/{name}"))
    seen: set[str] = set()
    unique: list[Path] = []
    for path in reports:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _best_h3_ensemble_report(root: Path) -> dict[str, Any]:
    reports = [_read_json(path) for path in _find_named_reports(root, H3_ENSEMBLE_FILENAMES)]
    reports = [report for report in reports if report]
    if not reports:
        return {}
    return max(
        reports,
        key=lambda row: (
            _float_or(row.get("control_separation_receipt_fraction"), 0.0),
            _float_or(row.get("median_heldout_explained_variance"), -1.0e9),
            -_float_or(row.get("p75_material_wrong_scale_win_fraction"), 1.0e9),
        ),
    )


def _best_caps_to_h3_report(root: Path) -> dict[str, Any]:
    reports = [_read_json(path) for path in _find_named_reports(root, CAPS_TO_H3_FILENAMES)]
    reports = [report for report in reports if report]
    if not reports:
        return {}
    return max(
        reports,
        key=lambda row: (
            bool(row.get("S2_CAP_PROFILE_TO_H3_RECEIPT", False)),
            -_float_or(row.get("median_reconstruction_mse"), 1.0e9),
        ),
    )


def _best_object_chart_report(root: Path) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for path in _find_named_reports(root, OBJECT_CHART_FILENAMES):
        report = _read_json(path)
        if not report:
            continue
        report = dict(report)
        report["_report_path"] = str(path)
        candidates.append(report)
    if not candidates:
        return {}

    def score(report: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, float, float]:
        flags = _object_chart_robust_flags(report)
        median_h3 = _float_or_none(report.get("median_h3_compactness_normalized"))
        median_shuffle = _float_or_none(report.get("median_shuffled_h3_compactness_normalized"))
        if median_h3 is not None and median_shuffle is not None and median_shuffle > 0.0:
            h3_margin = (median_shuffle - median_h3) / median_shuffle
        else:
            h3_margin = -1.0e9
        lineage_mode = (
            report.get("postprocess_incidence_mode") == "record_sector_checkpoint_lineage"
            or report.get("incidence_mode") == "record_sector_checkpoint_lineage"
        )
        current_schema = report.get("h3_compactness_margin_vs_median_shuffle") is not None
        return (
            float(bool(lineage_mode and report.get("observer_chart_bulk_population_receipt"))),
            float(bool(report.get("observer_chart_bulk_population_receipt"))),
            float(bool(flags.get("chart_receipt"))),
            float(bool(flags.get("localized_precursor_receipt"))),
            float(bool(flags.get("h3_beats_shuffled_robust"))),
            float(bool(current_schema)),
            _float_or(report.get("localized_object_count"), 0.0),
            _float_or(report.get("object_count"), 0.0),
            h3_margin,
        )

    return max(candidates, key=score)


def _best_cmb_lite_report(root: Path) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for path in _find_named_reports(root, CMB_LITE_FILENAMES):
        report = _read_json(path)
        if not report:
            continue
        report = dict(report)
        report["_report_path"] = str(path)
        candidates.append(report)
    if not candidates:
        return {}
    return max(candidates, key=_cmb_lite_report_score)


def _cmb_lite_report_score(report: dict[str, Any]) -> tuple[float, float, float, float, float]:
    fields = report.get("field_comparisons", {})
    if not isinstance(fields, dict):
        fields = {}
    best_overlap_corr = -1.0e9
    best_overlap_rmse = 1.0e9
    best_positive_corr = -1.0e9
    best_unconstrained_rmse = 1.0e9
    for field in fields.values():
        if not isinstance(field, dict):
            continue
        overlap = field.get("overlap_ell_physical_comparison", {})
        if isinstance(overlap, dict) and overlap.get("usable") and overlap.get("usable_positive_shape", False):
            corr = _float_or(overlap.get("shape_correlation"), -1.0e9)
            rmse = _float_or(overlap.get("positive_amp_normalized_rmse"), 1.0e9)
            if corr > best_overlap_corr or (corr == best_overlap_corr and rmse < best_overlap_rmse):
                best_overlap_corr = corr
                best_overlap_rmse = rmse
        if field.get("usable_positive_shape", False):
            best_positive_corr = max(best_positive_corr, _float_or(field.get("shape_correlation"), -1.0e9))
        best_unconstrained_rmse = min(best_unconstrained_rmse, _float_or(field.get("unconstrained_normalized_rmse"), 1.0e9))
    benchmark = report.get("benchmark", {})
    simulator = report.get("simulator", {})
    benchmark_rows = _float_or(benchmark.get("row_count"), 0.0) if isinstance(benchmark, dict) else 0.0
    ell_max = _float_or(simulator.get("ell_max"), 0.0) if isinstance(simulator, dict) else 0.0
    return (
        float(best_overlap_corr > -1.0e8),
        best_overlap_corr,
        -best_overlap_rmse,
        best_positive_corr,
        -best_unconstrained_rmse + 1.0e-6 * benchmark_rows + 1.0e-9 * ell_max,
    )


def _nested(data: dict[str, Any], *keys: str) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _rank_refinement_patch_counts(report: dict[str, Any]) -> list[int]:
    sizes = report.get("sizes", []) if isinstance(report, dict) else []
    if not isinstance(sizes, list):
        return []
    patch_counts: list[int] = []
    for row in sizes:
        if not isinstance(row, dict):
            continue
        try:
            patch_counts.append(int(row.get("patch_count")))
        except (TypeError, ValueError):
            continue
    return patch_counts


def _rank_refinement_size_medians(report: dict[str, Any]) -> list[float]:
    sizes = report.get("sizes", []) if isinstance(report, dict) else []
    if not isinstance(sizes, list):
        return []
    medians: list[float] = []
    for row in sizes:
        if not isinstance(row, dict):
            continue
        value = _float_or_none(row.get("median_candidate_dimension"))
        if value is not None:
            medians.append(value)
    return medians


def _flattened_counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if isinstance(value, list):
            items = value
        elif value is None:
            items = []
        else:
            items = [value]
        for item in items:
            key = str(item)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _selected_rank_control_count(
    report: dict[str, Any],
    *,
    metric: str,
    survives: bool,
) -> int:
    rows = _nested(report, "selected_rank_controls", "control_rows")
    if not isinstance(rows, list):
        return 0
    return sum(
        1
        for row in rows
        if isinstance(row, dict)
        and row.get("metric") == metric
        and bool(row.get("candidate_survives_control", False)) is bool(survives)
    )


def _profile_row(report: dict[str, Any], profile: str) -> dict[str, Any]:
    rows = report.get("profile_rows", []) if isinstance(report, dict) else []
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and row.get("profile") == profile:
            return row
    return {}


def _round_depth_for_patch_count(report: dict[str, Any], patch_count: int) -> Any:
    if not isinstance(report, dict) or not report:
        return None
    rows = report.get("finite_regulator_round_depth", [])
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            count = int(row.get("patch_count"))
        except (TypeError, ValueError):
            continue
        if count == int(patch_count):
            return row.get("patch_count_as_capacity_depth_rounds")
    return None


def _sync_gap_residual_candidate_value(sync_gap: dict[str, Any], key: str) -> Any:
    if not isinstance(sync_gap, dict) or not sync_gap:
        return None
    candidates = [
        _nested(row, "time_resolved_gap", "residual_field_gap_candidate")
        for row in sync_gap.get("rows", [])
        if isinstance(row, dict)
    ]
    candidates = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("available")
    ]
    if not candidates:
        return None
    if key == "candidate_receipt":
        return any(bool(candidate.get(key, False)) for candidate in candidates)
    if key == "field":
        selected = [candidate for candidate in candidates if candidate.get("candidate_receipt")] or candidates
        field = selected[0].get("field")
        return str(field) if field is not None else None
    values = [candidate.get(key) for candidate in candidates if isinstance(candidate.get(key), (int, float))]
    return _mean(values)


def _final_parent_collar_value(report: dict[str, Any], key: str) -> Any:
    if not isinstance(report, dict) or not report:
        return None
    patch_counts = report.get("patch_counts") or []
    if not patch_counts:
        return None
    try:
        final_key = str(max(int(value) for value in patch_counts))
    except (TypeError, ValueError):
        return None
    return _nested(report, "by_patch_count", final_key, key)


def _first_numeric(value: Any) -> float | None:
    if isinstance(value, list):
        for item in value:
            parsed = _float_or_none(item)
            if parsed is not None:
                return parsed
        return None
    return _float_or_none(value)


def _best_overlap_ell_field(fields: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    best_name: str | None = None
    best_report: dict[str, Any] = {}
    best_score = float("-inf")
    for name, report in fields.items():
        if not isinstance(report, dict):
            continue
        overlap = report.get("overlap_ell_physical_comparison", {})
        if not isinstance(overlap, dict) or not overlap.get("usable"):
            continue
        score = _float_or_none(overlap.get("shape_correlation"))
        if score is None:
            continue
        if not bool(overlap.get("usable_positive_shape", score > 0.0)):
            continue
        if score > best_score:
            best_score = float(score)
            best_name = str(name)
            best_report = overlap
    return best_name, best_report


def _mean(values: Any) -> float | None:
    numeric = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and np.isfinite(float(value))
    ]
    return fmean(numeric) if numeric else None


def _first_non_null(values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _dimension_estimate_or_none(value: Any, *, max_dimension: float = 64.0) -> float | None:
    parsed = _float_or_none(value)
    if parsed is None or not np.isfinite(parsed):
        return None
    if parsed < 0.0 or parsed > float(max_dimension):
        return None
    return parsed


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _float_or(value: Any, default: float) -> float:
    parsed = _float_or_none(value)
    return float(default if parsed is None else parsed)


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _dict_true_counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        for key, passed in value.items():
            if bool(passed):
                counts[str(key)] = counts.get(str(key), 0) + 1
    return counts


def _list_counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if isinstance(value, str):
            counts[value] = counts.get(value, 0) + 1
        elif isinstance(value, list):
            for item in value:
                counts[str(item)] = counts.get(str(item), 0) + 1
    return counts


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.10g}"
    return "n/a"
