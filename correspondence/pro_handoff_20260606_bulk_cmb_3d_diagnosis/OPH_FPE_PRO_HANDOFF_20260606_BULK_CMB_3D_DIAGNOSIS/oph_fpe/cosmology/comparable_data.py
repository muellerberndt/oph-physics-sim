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
    "modular_response_h3_report.json",
    "h3_refit_seed_ensemble_report.json",
    "observer_chart_object_h3_report.json",
    "observer_consensus_report.json",
    "object_consensus_report.json",
    "bulk_reconstruction_report.json",
    "cmb_lite_comparison_report.json",
    "cmb_transfer_report.json",
    "camb_lcdm_baseline_report.json",
    "oph_boltzmann_input_report.json",
    "oph_cmb_stress_report.json",
    "galaxy_proxy_report.json",
    "cl_comparison_report.json",
    "array_holonomy_report.json",
    "defect_timeline_report.json",
    "defect_interaction_report.json",
    "particle_likeness_report.json",
    "defect_h3_worldlines_report.json",
    "emergence_status_report.json",
)

DECLARED_CAP_FLOW_STATE_MODES = frozenset({"cap_flow_graph_generator", "cap_flow_detailed_balance_kernel"})


def collect_comparable_runs(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows = [_extract_run_row(path) for path in _find_run_dirs(run_dirs)]
    return [row for row in rows if row.get("has_comparable_data")]


def comparable_data_report(run_dirs: list[Path]) -> dict[str, Any]:
    rows = collect_comparable_runs(run_dirs)
    return {
        "mode": "oph_fpe_comparable_data_snapshot",
        "run_count": len(rows),
        "measurement_lanes": {
            "support_visible_lorentz_branch": _lorentz_branch_summary(rows),
            "state_derived_bw_matrix_elements": _state_bw_summary(rows),
            "planck_tt_shape_lite": _planck_lite_summary(rows),
            "cmb_screen_basis_transfer": _cmb_transfer_summary(rows),
            "camb_lcdm_baseline": _camb_lcdm_summary(rows),
            "oph_boltzmann_input_readouts": _oph_boltzmann_summary(rows),
            "oph_cmb_anomaly_stress_adapter": _oph_cmb_summary(rows),
            "static_galaxy_proxy": _galaxy_proxy_summary(rows),
            "h3_modular_response_controls": _h3_summary(rows),
            "h3_seed_ensemble_robustness": _h3_seed_ensemble_summary(rows),
            "observer_readout_consensus": _observer_readout_summary(rows),
            "observer_chart_object_population": _observer_chart_summary(rows),
            "neutral_observer_reconstruction": _neutral_reconstruction_summary(rows),
            "screen_holonomy_defect_proxy": _holonomy_summary(rows),
            "defect_worldline_particle_precursors": _defect_worldline_summary(rows),
        },
        "rows": rows,
        "physical_cmb_prediction": False,
        "physical_matter_power_prediction": False,
        "bulk_3d_established": any(bool(row.get("bulk_3d_established")) for row in rows),
        "claim_boundary": (
            "Comparable-data snapshot for current OPH-FPE receipts. Planck comparisons are shape-only "
            "C_l diagnostics with normalized axes and amplitude rescaling. OPH-CMB anomaly-stress "
            "adapter values are finite-collar parent diagnostics, not CAMB/CLASS outputs. H3 values are "
            "internal modular-response-vs-control receipts. Defect values are screen/collar holonomy proxies. "
            "This is not a physical CMB prediction, not a P(k), not a Boltzmann likelihood, and not a "
            "completed 3D-bulk or particle-emergence result."
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
        if any((root / name).exists() for name in RELEVANT_REPORTS):
            paths.add(root)
        if root.exists():
            for name in RELEVANT_REPORTS:
                for report_path in root.glob(f"**/{name}"):
                    paths.add(report_path.parent)
    return sorted(paths, key=lambda path: str(path))


def _extract_run_row(run_path: Path) -> dict[str, Any]:
    manifest = _read_json(run_path / "manifest.json")
    h3 = _read_json(run_path / "modular_response_h3_report.json")
    state_bw = _read_json(run_path / "bw_state_derived_report.json")
    transition_selection = _read_json(run_path / "transition_scale_selection_report.json")
    h3_ensemble = _read_json(run_path / "h3_refit_seed_ensemble_report.json")
    cmb = _read_json(run_path / "cmb_lite_comparison_report.json")
    transfer = _read_json(run_path / "cmb_transfer_report.json")
    camb_baseline = _read_json(run_path / "camb_lcdm_baseline_report.json")
    boltzmann_inputs = _read_json(run_path / "oph_boltzmann_input_report.json")
    oph_cmb = _read_json(run_path / "oph_cmb_stress_report.json")
    galaxy = _read_json(run_path / "galaxy_proxy_report.json")
    cl = _read_json(run_path / "cl_comparison_report.json")
    hol = _read_json(run_path / "array_holonomy_report.json")
    timeline = _read_json(run_path / "defect_timeline_report.json")
    interaction = _read_json(run_path / "defect_interaction_report.json")
    particle = _read_json(run_path / "particle_likeness_report.json")
    defect_h3_worldlines = _read_json(run_path / "defect_h3_worldlines_report.json")
    emergence = _read_json(run_path / "emergence_status_report.json")
    object_chart = _read_json(run_path / "observer_chart_object_h3_report.json")
    observer_consensus = _read_json(run_path / "observer_consensus_report.json")
    object_consensus = _read_json(run_path / "object_consensus_report.json")
    neutral = _read_json(run_path / "bulk_reconstruction_report.json")

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
    cmb_fields = cmb.get("field_comparisons", {}) if cmb else {}
    cmb_best_name = cmb.get("best_shape_field") if cmb else None
    cmb_best = cmb_fields.get(cmb_best_name, {}) if cmb_best_name else {}
    cmb_best_positive_name = cmb.get("best_positive_shape_field") if cmb else None
    cmb_best_positive = cmb_fields.get(cmb_best_positive_name, {}) if cmb_best_positive_name else {}
    record_cmb = cmb_fields.get("record_signature", {})
    cl_record = (cl.get("fields", {}) if cl else {}).get("record_signature", {})
    cl_record_control = cl_record.get("control_comparison", {})
    boltzmann_diagnostic_rows = (
        (boltzmann_inputs.get("diagnostic_repair_exchange", {}) or {}).get("rows", []) if boltzmann_inputs else []
    )
    neutral_dimension = neutral.get("neutral_dimension_report", {}) if neutral else {}
    blind_bulk = neutral.get("blind_observer_bulk_report", {}) if neutral else {}
    blind_dimension = blind_bulk.get("neutral_dimension_report", {}) if isinstance(blind_bulk, dict) else {}
    blind_corr = blind_dimension.get("correlation_dimension", {}) if isinstance(blind_dimension, dict) else {}
    blind_mle = blind_dimension.get("local_mle_dimension", {}) if isinstance(blind_dimension, dict) else {}
    neutral_primary = neutral_dimension.get("primary_dimension", {}) if isinstance(neutral_dimension, dict) else {}
    neutral_corr = neutral_dimension.get("correlation_dimension", {}) if isinstance(neutral_dimension, dict) else {}
    neutral_mle = neutral_dimension.get("local_mle_dimension", {}) if isinstance(neutral_dimension, dict) else {}
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

    row = {
        "run_path": str(run_path),
        "run_id": manifest.get("run_id") or run_path.name,
        "name": manifest.get("name"),
        "patch_count": manifest.get("patch_count") or (cl or {}).get("point_count"),
        "has_comparable_data": any(
            (
                state_bw,
                transition_selection,
                emergence,
                h3,
                cmb,
                transfer,
                camb_baseline,
                boltzmann_inputs,
                oph_cmb,
                galaxy,
                cl,
                hol,
                timeline,
                interaction,
                particle,
                defect_h3_worldlines,
            )
        ),
        "bulk_3d_established": bool(emergence.get("bulk_3d_established", False)),
        "support_visible_lorentz_3p1_kinematics_receipt": support_visible_lorentz,
        "chart_level_conformal_lorentz_receipt": chart_level_lorentz,
        "bw_automorphism_sanity_receipt": bw_automorphism_sanity,
        "endogenous_modular_generator_receipt": endogenous_modular_generator_receipt,
        "object_bulk_population_receipt": bool(
            emergence.get("OBJECT_BULK_POPULATION_RECEIPT")
            or emergence.get("object_bulk_population_receipt")
            or emergence.get("observer_chart_bulk_population_receipt", False)
        ),
        "screen_proxy_cmb_receipt": bool(
            emergence.get("SCREEN_PROXY_CMB_RECEIPT") or emergence.get("screen_proxy_cmb_receipt", False)
        ),
        "physical_cmb_prediction": bool(cmb.get("physical_cmb_prediction", False)),
        "state_bw_written": bool(state_bw),
        "state_bw_receipt": bool(state_bw.get("BW_KMS_BRANCH_INSTANTIATION_RECEIPT", False)),
        "state_bw_state_mode": state_bw.get("state_mode"),
        "state_bw_endogenous_generator": state_bw.get("endogenous_modular_generator"),
        "state_bw_declared_cap_flow_generator": bool(
            state_bw.get("declared_cap_flow_generator")
            or state_bw.get("state_mode") in DECLARED_CAP_FLOW_STATE_MODES
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
        "h3_receipt": bool(h3.get("h3_bulk_candidate_receipt", h3.get("MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT", False))),
        "h3_rmse": h3_fit.get("heldout_normalized_rmse"),
        "h3_explained_variance": h3_fit.get("heldout_explained_variance"),
        "h3_chart_dimension_point_count": h3_chart_dimension.get("point_count") if h3_chart_dimension else None,
        "h3_chart_candidate_3d_dimension_window": h3_chart_dimension.get("candidate_3d_dimension_window")
        if h3_chart_dimension
        else None,
        "h3_chart_dimension_estimators_agree": h3_chart_dimension.get("dimension_estimators_agree")
        if h3_chart_dimension
        else None,
        "h3_chart_correlation_dimension_estimate": h3_chart_corr.get("estimate")
        if isinstance(h3_chart_corr, dict)
        else None,
        "h3_chart_local_mle_dimension_estimate": h3_chart_mle.get("estimate")
        if isinstance(h3_chart_mle, dict)
        else None,
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
        "h3_two_pi_fit_win_fraction": wrong_audit.get("two_pi_h3_fit_win_fraction"),
        "h3_wrong_scale_red_flag": bool(wrong_audit.get("red_flag_wrong_scale_wins", False)),
        "h3_wrong_scale_winner_counts": wrong_audit.get("winner_counts"),
        "h3_wrong_scale_worst_group": (wrong_audit.get("worst_groups") or [None])[0],
        "h3_seed_ensemble_written": bool(h3_ensemble),
        "h3_seed_ensemble_seed_count": h3_ensemble.get("seed_count"),
        "h3_seed_ensemble_receipt_count": h3_ensemble.get("receipt_count"),
        "h3_seed_ensemble_receipt_fraction": h3_ensemble.get("receipt_fraction"),
        "h3_seed_ensemble_dim3_count": h3_ensemble.get("candidate_3d_window_count"),
        "h3_seed_ensemble_dim3_fraction": h3_ensemble.get("candidate_3d_window_fraction"),
        "h3_response_seed_robust_receipt": bool(h3_ensemble.get("h3_response_seed_robust_receipt", False)),
        "h3_chart_3d_seed_robust_receipt": bool(h3_ensemble.get("h3_chart_3d_seed_robust_receipt", False)),
        "h3_seed_ensemble_mean_nrmse": h3_ensemble.get("mean_heldout_normalized_rmse"),
        "h3_seed_ensemble_mean_ev": h3_ensemble.get("mean_heldout_explained_variance"),
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
        "observer_chart_object_median_receipt": object_chart_flags.get("chart_median_receipt"),
        "observer_chart_localized_object_precursor_receipt": object_chart_flags.get("localized_precursor_receipt"),
        "observer_chart_localized_object_median_precursor_receipt": object_chart_flags.get(
            "localized_median_precursor_receipt"
        ),
        "observer_chart_bulk_population_receipt": object_chart.get("observer_chart_bulk_population_receipt"),
        "observer_chart_object_count": object_chart.get("object_count"),
        "observer_chart_localized_object_count": object_chart.get("localized_object_count"),
        "observer_chart_localized_not_boundary_object_count": object_chart.get("localized_not_boundary_object_count"),
        "observer_chart_shuffle_control_count": object_chart.get("shuffle_control_count"),
        "observer_chart_shuffled_localized_object_count": object_chart.get("shuffled_localized_object_count"),
        "observer_chart_shuffled_localized_object_p90": object_chart.get("shuffled_localized_object_p90"),
        "observer_chart_h3_compactness": object_chart.get("median_h3_compactness_normalized"),
        "observer_chart_s2_compactness": object_chart.get("median_s2_boundary_compactness_normalized"),
        "observer_chart_shuffled_h3_compactness": object_chart.get("median_shuffled_h3_compactness_normalized"),
        "observer_chart_p10_shuffled_h3_compactness": object_chart.get(
            "p10_shuffled_h3_compactness_normalized"
        ),
        "observer_chart_p90_shuffled_h3_compactness": object_chart.get(
            "p90_shuffled_h3_compactness_normalized"
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
        "blind_bulk_usable": blind_bulk.get("usable") if blind_bulk else None,
        "blind_bulk_feature_width": blind_bulk.get("feature_width") if blind_bulk else None,
        "blind_bulk_s2_leakage_audit_pass": blind_bulk.get("s2_leakage_audit_pass") if blind_bulk else None,
        "blind_bulk_s2_distance_correlation": blind_bulk.get("s2_distance_correlation") if blind_bulk else None,
        "blind_bulk_candidate_3d_dimension_window": blind_bulk.get("candidate_3d_dimension_window") if blind_bulk else None,
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
        "record_signature_shape_correlation": record_cmb.get("shape_correlation"),
        "record_signature_normalized_rmse": record_cmb.get("normalized_rmse"),
        "record_signature_peak_fraction_delta": record_cmb.get("peak_fraction_delta"),
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
                "object_bulk_population_receipt",
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
        "object_bulk_population_count": sum(1 for row in usable if row.get("object_bulk_population_receipt")),
        "screen_proxy_cmb_count": sum(1 for row in usable if row.get("screen_proxy_cmb_receipt")),
        "bulk_3d_established_count": sum(1 for row in usable if row.get("bulk_3d_established")),
        "interpretation": (
            "Splits the paper-side chart-level Lorentz receipt from later strengthening receipts. "
            "A support-visible Lorentz count means conformal H3 chart plus direct BW/KMS automorphism sanity; "
            "it does not mean populated 3D bulk, particles, or physical cosmology."
        ),
    }


def _planck_lite_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("cmb_best_normalized_rmse") is not None]
    return {
        "run_count": len(usable),
        "best_field_counts": _counts(row.get("cmb_best_field") for row in usable),
        "best_positive_field_counts": _counts(row.get("cmb_best_positive_field") for row in usable),
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
        "physical_cmb_prediction": False,
        "interpretation": (
            "Currently useful as a negative/diagnostic Planck-lite screen comparison. The best low-RMSE "
            "fields are typically anticorrelated with Planck TT; record_signature is weakly positive."
        ),
    }


def _state_bw_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("state_bw_written")]
    declared_cap_flow = [row for row in usable if row.get("state_bw_declared_cap_flow_generator")]
    direct_transition = [row for row in usable if row.get("state_bw_direct_transition_automorphism")]
    endogenous = [
        row
        for row in usable
        if row.get("state_bw_endogenous_generator") and not row.get("state_bw_declared_cap_flow_generator")
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
        "direct_transition_automorphism_run_count": len(direct_transition),
        "receipt_count": sum(1 for row in usable if row.get("state_bw_receipt")),
        "endogenous_correct_beats_controls_count": sum(
            1 for row in endogenous if row.get("state_bw_correct_beats_controls")
        ),
        "selected_2pi_count": sum(1 for row in usable if row.get("state_bw_selected_2pi")),
        "mean_median_residual": _mean(row.get("state_bw_median") for row in usable),
        "mean_endogenous_median_residual": _mean(row.get("state_bw_median") for row in endogenous),
        "mean_declared_cap_flow_median_residual": _mean(row.get("state_bw_median") for row in declared_cap_flow),
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


def _h3_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("h3_rmse") is not None]
    receipt_count = sum(1 for row in usable if row.get("h3_receipt"))
    h3_rmse = _mean(row.get("h3_rmse") for row in usable)
    wrong_pi_rmse = _mean(row.get("wrong_pi_rmse") for row in usable)
    audit_rows = [row for row in usable if row.get("h3_wrong_scale_audit_eligible")]
    wrong_scale_win_fraction = _mean(row.get("h3_wrong_scale_win_fraction") for row in audit_rows)
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
        "mean_h3_rmse": h3_rmse,
        "mean_h3_explained_variance": _mean(row.get("h3_explained_variance") for row in usable),
        "h3_chart_3d_window_count": sum(1 for row in usable if row.get("h3_chart_candidate_3d_dimension_window")),
        "h3_chart_dimension_agree_count": sum(1 for row in usable if row.get("h3_chart_dimension_estimators_agree")),
        "mean_h3_chart_correlation_dimension": _mean(
            row.get("h3_chart_correlation_dimension_estimate") for row in usable
        ),
        "mean_h3_chart_local_mle_dimension": _mean(row.get("h3_chart_local_mle_dimension_estimate") for row in usable),
        "mean_s2_boundary_rmse": _mean(row.get("s2_boundary_rmse") for row in usable),
        "mean_shuffled_response_rmse": _mean(row.get("shuffled_response_rmse") for row in usable),
        "mean_wrong_pi_rmse": wrong_pi_rmse,
        "wrong_scale_feature_audit_count": len(audit_rows),
        "mean_wrong_scale_feature_win_fraction": wrong_scale_win_fraction,
        "mean_two_pi_h3_feature_win_fraction": _mean(row.get("h3_two_pi_fit_win_fraction") for row in audit_rows),
        "wrong_scale_red_flag_count": sum(1 for row in audit_rows if row.get("h3_wrong_scale_red_flag")),
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
        "mean_dim3_fraction": _mean(row.get("h3_seed_ensemble_dim3_fraction") for row in usable),
        "mean_heldout_normalized_rmse": _mean(row.get("h3_seed_ensemble_mean_nrmse") for row in usable),
        "mean_heldout_explained_variance": _mean(row.get("h3_seed_ensemble_mean_ev") for row in usable),
        "interpretation": (
            "Cached H3 seed-ensemble robustness. This prevents accepting a single lucky H3 candidate "
            "sample as bulk evidence. A robust 3D result requires high response receipt fraction and "
            "a stable 3D chart fraction; failures keep the bulk gate closed."
        ),
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
        "localized_object_precursor_receipt_count": sum(
            1 for row in usable if row.get("observer_chart_localized_object_precursor_receipt")
        ),
        "bulk_population_receipt_count": sum(1 for row in usable if row.get("observer_chart_bulk_population_receipt")),
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
        "h3_beats_shuffled_count": h3_beats_count,
        "h3_beats_shuffled_robust_count": h3_beats_robust_count,
        "h3_not_boundary_dominated_count": boundary_count,
        "interpretation": interpretation,
    }


def _neutral_reconstruction_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("neutral_reconstruction_written")]
    return {
        "run_count": len(usable),
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
            "blind lane excludes S2 axes, support nodes, cap membership, and radial/modular-depth priors. "
            "A 3D window here is still not a physical bulk claim unless the BW/object gates pass too."
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
    boltzmann = lanes["oph_boltzmann_input_readouts"]
    oph_cmb = lanes["oph_cmb_anomaly_stress_adapter"]
    galaxy = lanes["static_galaxy_proxy"]
    hol = lanes["screen_holonomy_defect_proxy"]
    defect_worldlines = lanes["defect_worldline_particle_precursors"]
    lines = [
        "# OPH-FPE Comparable Data Snapshot",
        "",
        report["claim_boundary"],
        "",
        "## Current Answer",
        "",
        "Yes: the current simulator can emit diagnostic, measurement-facing values. No: it does not yet emit a physical CMB prediction, physical P(k), populated 3D bulk, or particle spectrum.",
        "",
        "## Support-Visible Lorentz Branch",
        "",
        f"- runs with Lorentz split receipts: {lorentz['run_count']}",
        f"- chart-level conformal Lorentz receipts: {lorentz['chart_level_conformal_lorentz_count']}",
        f"- BW automorphism sanity receipts: {lorentz['bw_automorphism_sanity_count']}",
        f"- support-visible 3+1D Lorentz receipts: {lorentz['support_visible_lorentz_3p1_count']}",
        f"- endogenous modular-generator receipts: {lorentz['endogenous_modular_generator_count']}",
        f"- object bulk-population receipts: {lorentz['object_bulk_population_count']}",
        f"- screen-proxy CMB receipts: {lorentz['screen_proxy_cmb_count']}",
        f"- bulk-established count: {lorentz['bulk_3d_established_count']}",
        f"- interpretation: {lorentz['interpretation']}",
        "",
        "## State-Derived BW Matrix Elements",
        "",
        f"- runs with state-derived BW reports: {state_bw['run_count']}",
        f"- endogenous-generator runs: {state_bw['endogenous_run_count']}",
        f"- declared cap-flow generator runs: {state_bw['declared_cap_flow_run_count']}",
        f"- direct transition-automorphism runs: {state_bw['direct_transition_automorphism_run_count']}",
        f"- BW/KMS receipts: {state_bw['receipt_count']}",
        f"- endogenous correct-beats-controls count: {state_bw['endogenous_correct_beats_controls_count']}",
        f"- selected-2pi count: {state_bw['selected_2pi_count']}",
        f"- state-mode counts: {state_bw['state_mode_counts']}",
        f"- generator-scale counts: {state_bw['generator_scale_counts']}",
        f"- best-control counts: {state_bw['best_control_counts']}",
        f"- mean median residual: {_fmt(state_bw['mean_median_residual'])}",
        f"- mean endogenous median residual: {_fmt(state_bw['mean_endogenous_median_residual'])}",
        f"- mean declared cap-flow median residual: {_fmt(state_bw['mean_declared_cap_flow_median_residual'])}",
        f"- mean direct transition median residual: {_fmt(state_bw['mean_direct_transition_median_residual'])}",
        f"- mean wrong-1x median: {_fmt(state_bw['mean_wrong_1x_median'])}",
        f"- mean wrong-pi median: {_fmt(state_bw['mean_wrong_pi_median'])}",
        f"- mean no-modular-flow median: {_fmt(state_bw['mean_no_modular_flow_median'])}",
        f"- interpretation: {state_bw['interpretation']}",
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
        "## H3 Modular-Response Receipt",
        "",
        f"- runs with H3 fits: {h3['run_count']}",
        f"- H3 receipts: {h3['receipt_count']}",
        f"- mean H3 RMSE: {_fmt(h3['mean_h3_rmse'])}",
        f"- mean H3 explained variance: {_fmt(h3['mean_h3_explained_variance'])}",
        f"- H3-chart 3D-window count: {h3['h3_chart_3d_window_count']}",
        f"- H3-chart estimator-agreement count: {h3['h3_chart_dimension_agree_count']}",
        f"- mean H3-chart correlation dimension: {_fmt(h3['mean_h3_chart_correlation_dimension'])}",
        f"- mean H3-chart local-MLE dimension: {_fmt(h3['mean_h3_chart_local_mle_dimension'])}",
        f"- mean S2-boundary RMSE: {_fmt(h3['mean_s2_boundary_rmse'])}",
        f"- mean shuffled-response RMSE: {_fmt(h3['mean_shuffled_response_rmse'])}",
        f"- strict wrong-scale feature audits: {h3['wrong_scale_feature_audit_count']}",
        f"- mean wrong-scale feature win fraction: {_fmt(h3['mean_wrong_scale_feature_win_fraction'])}",
        f"- mean 2pi/H3 feature win fraction: {_fmt(h3['mean_two_pi_h3_feature_win_fraction'])}",
        f"- wrong-scale red-flag runs: {h3['wrong_scale_red_flag_count']}",
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
        f"- H3 beats median shuffle count: {object_chart['h3_beats_shuffled_count']}",
        f"- H3 beats robust shuffle count: {object_chart['h3_beats_shuffled_robust_count']}",
        "",
        "## Neutral Observer Reconstruction",
        "",
        f"- runs with neutral reconstruction reports: {neutral['run_count']}",
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
        f"- blind bulk-established count: {neutral['blind_bulk_3d_established_count']}",
        f"- mean blind feature width: {_fmt(neutral['mean_blind_feature_width'])}",
        f"- mean blind S2-distance correlation: {_fmt(neutral['mean_blind_s2_distance_correlation'])}",
        f"- mean blind correlation dimension: {_fmt(neutral['mean_blind_correlation_dimension'])}",
        f"- mean blind local-MLE dimension: {_fmt(neutral['mean_blind_local_mle_dimension'])}",
        "",
        "## Planck-Lite Screen C_l Shape",
        "",
        f"- runs with CMB-lite comparisons: {planck['run_count']}",
        f"- best-field counts: {planck['best_field_counts']}",
        f"- best-positive-field counts: {planck['best_positive_field_counts']}",
        f"- mean best-field correlation: {_fmt(planck['mean_best_shape_correlation'])}",
        f"- mean best-field RMSE: {_fmt(planck['mean_best_normalized_rmse'])}",
        f"- mean best-positive-field correlation: {_fmt(planck['mean_best_positive_shape_correlation'])}",
        f"- mean best-positive-field RMSE: {_fmt(planck['mean_best_positive_normalized_rmse'])}",
        f"- mean record-signature correlation: {_fmt(planck['mean_record_signature_shape_correlation'])}",
        f"- mean record-signature RMSE: {_fmt(planck['mean_record_signature_normalized_rmse'])}",
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
    return json.loads(path.read_text(encoding="utf-8"))


def _nested(data: dict[str, Any], first: str, second: str) -> Any:
    value = data.get(first, {})
    if not isinstance(value, dict):
        return None
    return value.get(second)


def _mean(values: Any) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return fmean(numeric) if numeric else None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or(value: Any, default: float) -> float:
    parsed = _float_or_none(value)
    return float(default if parsed is None else parsed)


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.10g}"
    return "n/a"
