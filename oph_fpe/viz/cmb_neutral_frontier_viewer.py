from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_cmb_neutral_frontier_viewer(run_dir: Path, out_path: Path | None = None) -> dict[str, Any]:
    """Write a pack-level CMB/neutral-bulk status and comparison viewer."""

    run = Path(run_dir)
    claims = _read_json(run / "claims.json")
    cmb_input = _read_json(run / "physical_cmb_input_report.json")
    cmb_promotion = _read_json(run / "physical_cmb_promotion_audit_report.json")
    cmb_frontier = _read_json(run / "physical_cmb_frontier_report.json")
    cmb_output = _read_json(run / "physical_cmb_output_comparison_report.json")
    official_readiness = _read_json(run / "official_planck_likelihood_readiness_report.json")
    neutral_audit = _read_json(run / "neutral_3d_bulk_audit_report.json")
    neutral_frontier = _read_json(run / "strict_neutral_bulk_frontier_report.json")
    overlap = _read_json(run / "overlap_native_neutral_control_report.json")
    overlap_graph = _read_json(run / "overlap_native_graph_geometry_report.json")
    overlap_graph_sweep = _read_json(run / "overlap_native_graph_geometry_sweep_report.json")
    overlap_residual_graph = _read_json(run / "overlap_residualized_graph_geometry_report.json")
    overlap_residual_graph_sweep = _read_json(run / "overlap_residualized_graph_geometry_sweep_report.json")
    object_summary = _read_json(run / "object_h3_bulk_viewer_summary.json")
    finite_bins = _read_rows(run / "finite_repair_clock_cmb_tt_bins.csv")
    scale_bins = _read_rows(run / "scale_compressed_cmb_tt_bins.csv")
    lcdm_bins = _read_rows(run / "camb_lcdm_tt_bins.csv")
    peak_feature_rows = _read_rows(run / "physical_cmb_peak_features.csv")
    overlap_rank_obstruction = (
        overlap_graph_sweep.get("rank_obstruction_summary")
        if isinstance(overlap_graph_sweep.get("rank_obstruction_summary"), dict)
        else {}
    )
    overlap_residual_rank_obstruction = (
        overlap_residual_graph_sweep.get("rank_obstruction_summary")
        if isinstance(overlap_residual_graph_sweep.get("rank_obstruction_summary"), dict)
        else {}
    )
    overlap_gate_coincidence = (
        overlap_graph_sweep.get("gate_coincidence_summary")
        if isinstance(overlap_graph_sweep.get("gate_coincidence_summary"), dict)
        else {}
    )
    overlap_residual_gate_coincidence = (
        overlap_residual_graph_sweep.get("gate_coincidence_summary")
        if isinstance(overlap_residual_graph_sweep.get("gate_coincidence_summary"), dict)
        else {}
    )

    destination = out_path or (run / "cmb_neutral_frontier_viewer.html")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _payload(
        run,
        claims,
        cmb_input,
        cmb_promotion,
        cmb_frontier,
        cmb_output,
        official_readiness,
        neutral_audit,
        neutral_frontier,
        overlap,
        overlap_graph,
        overlap_graph_sweep,
        overlap_residual_graph,
        overlap_residual_graph_sweep,
        object_summary,
        finite_bins,
        scale_bins,
        lcdm_bins,
        peak_feature_rows,
    )
    destination.write_text(_render_html(payload), encoding="utf-8")
    summary = {
        "mode": "cmb_neutral_frontier_viewer",
        "viewer_path": str(destination),
        "run_dir": str(run),
        "tt_bin_count": len(payload["ttBins"]),
        "physical_cmb_prediction": bool(claims.get("physical_cmb_prediction", False)),
        "physical_cmb_promotion_ready": bool(claims.get("physical_cmb_promotion_ready", False)),
        "physical_cmb_promotion_blocker_count": int(claims.get("physical_cmb_promotion_blocker_count") or 0),
        "physical_cmb_frontier_written": bool(cmb_frontier),
        "physical_cmb_frontier_ready": bool(cmb_frontier.get("physical_cmb_prediction_ready", False)),
        "physical_cmb_frontier_gate_count": int(len(cmb_frontier.get("gate_rows") or [])),
        "physical_cmb_frontier_gap_count": int(len(cmb_frontier.get("gate_gap_rows") or [])),
        "physical_cmb_frontier_blocker_count": int(len(cmb_frontier.get("blockers") or [])),
        "physical_cmb_output_comparison_receipt": bool(
            cmb_output.get("PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT", False)
        ),
        "physical_cmb_output_prediction_receipt": bool(
            cmb_output.get("PHYSICAL_CMB_PREDICTION_RECEIPT", False)
        ),
        "physical_cmb_output_best_oph_chi2_per_bin": (
            (cmb_output.get("best_oph_diagnostic_model") or {}).get("amplitude_fit_chi2_per_bin")
        ),
        "physical_cmb_output_best_oph_residual_bin_count": int(
            (cmb_output.get("best_oph_residual_summary") or {}).get("bin_count") or 0
        ),
        "physical_cmb_output_best_oph_rms_sigma_residual": (
            (cmb_output.get("best_oph_residual_summary") or {}).get("rms_sigma_residual")
        ),
        "physical_cmb_output_best_oph_max_abs_sigma_residual": (
            (cmb_output.get("best_oph_residual_summary") or {}).get("max_abs_sigma_residual")
        ),
        "physical_cmb_output_best_oph_peak_count": int(
            (cmb_output.get("best_oph_peak_feature_summary") or {}).get("peak_count") or 0
        ),
        "physical_cmb_output_best_oph_mean_abs_peak_ell_delta": (
            (cmb_output.get("best_oph_peak_feature_summary") or {}).get("mean_abs_peak_ell_delta")
        ),
        "physical_cmb_output_best_oph_mean_abs_peak_height_fractional_delta": (
            (cmb_output.get("best_oph_peak_feature_summary") or {}).get(
                "mean_abs_peak_height_fractional_delta"
            )
        ),
        "official_planck_likelihood_readiness_written": bool(official_readiness),
        "official_planck_likelihood_execution_ready": bool(
            official_readiness.get("official_likelihood_execution_ready", False)
        ),
        "official_planck_likelihood_data_paths_configured": bool(
            official_readiness.get("official_planck_likelihood_data_paths_configured", False)
        ),
        "official_planck_clik_api_available": bool(
            official_readiness.get("official_clik_api_available", False)
        ),
        "official_planck_likelihood_blocker_count": int(len(official_readiness.get("blockers") or [])),
        "strict_neutral_bulk": bool(claims.get("strict_neutral_bulk", False)),
        "strict_neutral_bulk_frontier_written": bool(neutral_frontier),
        "strict_neutral_bulk_frontier_ready": bool(neutral_frontier.get("strict_neutral_bulk_ready", False)),
        "strict_neutral_bulk_frontier_gate_count": int(len(neutral_frontier.get("gate_rows") or [])),
        "strict_neutral_bulk_frontier_gap_count": int(len(neutral_frontier.get("gate_gap_rows") or [])),
        "overlap_native_negative_control_receipt": bool(
            overlap.get("OVERLAP_NATIVE_NEGATIVE_CONTROL_RECEIPT", False)
        ),
        "overlap_native_spatial_3d_candidate": bool(overlap.get("overlap_native_spatial_3d_candidate", False)),
        "overlap_native_graph_geometry_receipt": bool(
            overlap_graph.get("OVERLAP_NATIVE_GRAPH_GEOMETRY_RECEIPT", False)
        ),
        "overlap_native_graph_spatial_3d_candidate": bool(
            overlap_graph.get("overlap_graph_spatial_3d_candidate", False)
        ),
        "overlap_native_graph_strict_h3_candidate": bool(
            overlap_graph.get("overlap_graph_strict_h3_candidate", False)
        ),
        "overlap_native_graph_sweep_case_count": int(overlap_graph_sweep.get("case_count") or 0),
        "overlap_native_graph_sweep_spatial_3d_candidate_count": int(
            overlap_graph_sweep.get("spatial_3d_candidate_count") or 0
        ),
        "overlap_native_graph_sweep_strict_h3_candidate_count": int(
            overlap_graph_sweep.get("strict_h3_candidate_count") or 0
        ),
        "overlap_native_graph_sweep_rank3_selector_count": int(
            overlap_graph_sweep.get("rank3_selector_count") or 0
        ),
        "overlap_native_graph_sweep_model_order_rank3_selector_count": int(
            overlap_rank_obstruction.get("model_order_rank3_selector_count") or 0
        ),
        "overlap_native_graph_sweep_closest_strict_candidate_count": int(
            len(overlap_graph_sweep.get("closest_strict_rows") or [])
        ),
        "overlap_native_graph_sweep_nontrivial_rank3_selector_count": int(
            overlap_rank_obstruction.get("nontrivial_rank3_selector_count") or 0
        ),
        "overlap_native_graph_sweep_nontrivial_model_order_rank3_selector_count": int(
            overlap_rank_obstruction.get("nontrivial_model_order_rank3_selector_count") or 0
        ),
        "overlap_native_graph_sweep_spatial_h3_rank3_coincidence_count": int(
            overlap_gate_coincidence.get("spatial_h3_independent_rank3_selector_count") or 0
        ),
        "overlap_native_graph_sweep_spatial_h3_nontrivial_rank3_coincidence_count": int(
            overlap_gate_coincidence.get("spatial_h3_nontrivial_rank3_selector_count") or 0
        ),
        "overlap_residualized_graph_sweep_case_count": int(
            overlap_residual_graph_sweep.get("case_count") or 0
        ),
        "overlap_residualized_graph_sweep_spatial_3d_candidate_count": int(
            overlap_residual_graph_sweep.get("spatial_3d_candidate_count") or 0
        ),
        "overlap_residualized_graph_sweep_strict_h3_candidate_count": int(
            overlap_residual_graph_sweep.get("strict_h3_candidate_count") or 0
        ),
        "overlap_residualized_graph_sweep_rank3_selector_count": int(
            overlap_residual_graph_sweep.get("rank3_selector_count") or 0
        ),
        "overlap_residualized_graph_sweep_model_order_rank3_selector_count": int(
            overlap_residual_rank_obstruction.get("model_order_rank3_selector_count") or 0
        ),
        "overlap_residualized_graph_sweep_closest_strict_candidate_count": int(
            len(overlap_residual_graph_sweep.get("closest_strict_rows") or [])
        ),
        "overlap_residualized_graph_sweep_nontrivial_rank3_selector_count": int(
            overlap_residual_rank_obstruction.get("nontrivial_rank3_selector_count") or 0
        ),
        "overlap_residualized_graph_sweep_nontrivial_model_order_rank3_selector_count": int(
            overlap_residual_rank_obstruction.get("nontrivial_model_order_rank3_selector_count") or 0
        ),
        "overlap_residualized_graph_sweep_spatial_h3_rank3_coincidence_count": int(
            overlap_residual_gate_coincidence.get("spatial_h3_independent_rank3_selector_count") or 0
        ),
        "overlap_residualized_graph_sweep_spatial_h3_nontrivial_rank3_coincidence_count": int(
            overlap_residual_gate_coincidence.get("spatial_h3_nontrivial_rank3_selector_count") or 0
        ),
        "theorem_assisted_h3_bulk": bool(claims.get("theorem_assisted_h3_bulk", False)),
        "object_h3_bulk_viewer_object_count": int(claims.get("object_h3_bulk_viewer_object_count") or 0),
        "object_h3_bulk_viewer_observer_overlap_link_count": int(
            claims.get("object_h3_bulk_viewer_observer_overlap_link_count") or 0
        ),
        "claim_boundary": (
            "Visualization of current measurement-comparable CMB curves, physical-CMB input gates, "
            "neutral-bulk gates, and overlap/H3 bulk status. It does not promote diagnostics to "
            "physical predictions."
        ),
    }
    (destination.parent / "cmb_neutral_frontier_viewer_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


def _payload(
    run: Path,
    claims: dict[str, Any],
    cmb_input: dict[str, Any],
    cmb_promotion: dict[str, Any],
    cmb_frontier: dict[str, Any],
    cmb_output: dict[str, Any],
    official_readiness: dict[str, Any],
    neutral_audit: dict[str, Any],
    neutral_frontier: dict[str, Any],
    overlap: dict[str, Any],
    overlap_graph: dict[str, Any],
    overlap_graph_sweep: dict[str, Any],
    overlap_residual_graph: dict[str, Any],
    overlap_residual_graph_sweep: dict[str, Any],
    object_summary: dict[str, Any],
    finite_bins: list[dict[str, str]],
    scale_bins: list[dict[str, str]],
    lcdm_bins: list[dict[str, str]],
    peak_feature_rows: list[dict[str, str]],
) -> dict[str, Any]:
    by_ell: dict[float, dict[str, float | None]] = {}

    def row_for(ell: float) -> dict[str, float | None]:
        return by_ell.setdefault(
            round(float(ell), 6),
            {
                "ell": float(ell),
                "observed": None,
                "sigma": None,
                "lcdm": None,
                "finiteRepair": None,
                "scaleCompressed": None,
            },
        )

    for row in lcdm_bins:
        ell = _float_or_none(row.get("ell"))
        if ell is None:
            continue
        out = row_for(ell)
        out["observed"] = _float_or_none(row.get("observed_D_ell"))
        out["sigma"] = _float_or_none(row.get("sigma_D_ell"))
        out["lcdm"] = _first_float(row, "amplitude_fit_camb_D_ell", "camb_D_ell", "best_fit_column_D_ell")
    for row in finite_bins:
        ell = _float_or_none(row.get("ell"))
        if ell is None:
            continue
        out = row_for(ell)
        out["observed"] = out["observed"] if out["observed"] is not None else _float_or_none(row.get("observed_D_ell"))
        out["finiteRepair"] = _first_float(
            row,
            "finite_repair_clock_plus_selector_ir_D_ell",
            "finite_repair_clock_scalar_tilt_D_ell",
            "best_fit_D_ell",
        )
    for row in scale_bins:
        ell = _float_or_none(row.get("ell"))
        if ell is None:
            continue
        out = row_for(ell)
        out["observed"] = out["observed"] if out["observed"] is not None else _float_or_none(row.get("observed_D_ell"))
        out["scaleCompressed"] = _first_float(
            row,
            "scale_compressed_ir_kernel_D_ell",
            "scale_compressed_scalar_tilt_D_ell",
            "best_fit_D_ell",
        )
    tt_bins = [
        row
        for row in sorted(by_ell.values(), key=lambda value: float(value["ell"] or 0.0))
        if row.get("observed") is not None
    ]
    input_status = cmb_input.get("input_status") if isinstance(cmb_input.get("input_status"), dict) else {}
    rank_obstruction = (
        overlap_graph_sweep.get("rank_obstruction_summary")
        if isinstance(overlap_graph_sweep.get("rank_obstruction_summary"), dict)
        else {}
    )
    residual_rank_obstruction = (
        overlap_residual_graph_sweep.get("rank_obstruction_summary")
        if isinstance(overlap_residual_graph_sweep.get("rank_obstruction_summary"), dict)
        else {}
    )
    gate_coincidence = (
        overlap_graph_sweep.get("gate_coincidence_summary")
        if isinstance(overlap_graph_sweep.get("gate_coincidence_summary"), dict)
        else {}
    )
    residual_gate_coincidence = (
        overlap_residual_graph_sweep.get("gate_coincidence_summary")
        if isinstance(overlap_residual_graph_sweep.get("gate_coincidence_summary"), dict)
        else {}
    )
    return {
        "runDir": str(run),
        "ttBins": tt_bins,
        "claims": {
            "workingMiniUniverse": bool(claims.get("WORKING_MINI_UNIVERSE_V0", False)),
            "physicalCmbPrediction": bool(claims.get("physical_cmb_prediction", False)),
            "physicalCmbPromotionReady": bool(claims.get("physical_cmb_promotion_ready", False)),
            "physicalCmbBlockerCount": int(claims.get("physical_cmb_promotion_blocker_count") or 0),
            "physicalCmbOutputComparison": bool(
                claims.get("physical_cmb_output_comparison_receipt", False)
                or cmb_output.get("PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT", False)
            ),
            "strictNeutralBulk": bool(claims.get("strict_neutral_bulk", False)),
            "strictNeutralFrontierReady": bool(
                claims.get("strict_neutral_bulk_frontier_ready", False)
                or neutral_frontier.get("strict_neutral_bulk_ready", False)
            ),
            "theoremAssistedH3Bulk": bool(claims.get("theorem_assisted_h3_bulk", False)),
            "objectCount": int(claims.get("object_h3_bulk_viewer_object_count") or 0),
            "overlapLinks": int(claims.get("object_h3_bulk_viewer_observer_overlap_link_count") or 0),
        },
        "cmb": {
            "contractReceipt": bool(cmb_input.get("PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT", False)),
            "promotionReady": bool(cmb_promotion.get("physical_cmb_promotion_ready", False)),
            "contractBlockers": cmb_promotion.get("contract_blockers") or cmb_input.get("blockers") or [],
            "promotionBlockers": cmb_promotion.get("promotion_blockers") or [],
            "frontier": {
                "written": bool(cmb_frontier),
                "ready": bool(cmb_frontier.get("physical_cmb_prediction_ready", False)),
                "comparisonReceipt": bool(cmb_frontier.get("physical_cmb_output_comparison_receipt", False)),
                "predictionReceipt": bool(cmb_frontier.get("physical_cmb_prediction_receipt", False)),
                "gateRows": cmb_frontier.get("gate_rows") or [],
                "gapRows": cmb_frontier.get("gate_gap_rows") or [],
                "blockers": cmb_frontier.get("blockers") or [],
                "nextMissingReceipts": cmb_frontier.get("next_missing_receipts") or [],
            },
            "inputStatus": {
                key: {
                    "source": value.get("source"),
                    "diagnosticValuePresent": bool(value.get("diagnostic_value_present", False)),
                    "physicalGatePassed": bool(value.get("physical_gate_passed", False)),
                    "rowCount": value.get("row_count"),
                }
                for key, value in input_status.items()
                if isinstance(value, dict)
                and key
                in {
                    "P_source",
                    "N_source",
                    "eta_R",
                    "A_zeta",
                    "q_IR",
                    "ell_IR",
                    "B_A_k_a",
                    "Gamma_rec_k_a",
                    "rho_A_a",
                    "freezeout_surface",
                }
            },
            "output": {
                "comparisonReceipt": bool(cmb_output.get("PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT", False)),
                "predictionReceipt": bool(cmb_output.get("PHYSICAL_CMB_PREDICTION_RECEIPT", False)),
                "measurementComparableModelCount": int(
                    cmb_output.get("measurement_comparable_model_count") or 0
                ),
                "ophDiagnosticModelCount": int(cmb_output.get("oph_diagnostic_model_count") or 0),
                "bestOphDiagnosticModel": cmb_output.get("best_oph_diagnostic_model") or {},
                "bestOphResidualSummary": cmb_output.get("best_oph_residual_summary") or {},
                "bestOphPeakFeatureSummary": cmb_output.get("best_oph_peak_feature_summary") or {},
                "peakFeatures": _compact_peak_feature_rows(peak_feature_rows),
                "rows": _compact_output_rows(cmb_output.get("rows")),
            },
            "officialLikelihood": {
                "written": bool(official_readiness),
                "executionReady": bool(official_readiness.get("official_likelihood_execution_ready", False)),
                "dataPathsConfigured": bool(
                    official_readiness.get("official_planck_likelihood_data_paths_configured", False)
                ),
                "clikApiAvailable": bool(official_readiness.get("official_clik_api_available", False)),
                "cambAvailable": bool(official_readiness.get("camb_available", False)),
                "cobayaAvailable": bool(official_readiness.get("cobaya_available", False)),
                "blockers": official_readiness.get("blockers") or [],
                "dataPaths": official_readiness.get("data_paths") or [],
            },
        },
        "neutral": {
            "strictNeutralBulkReady": bool(neutral_audit.get("strict_neutral_bulk_ready", False)),
            "strictNeutralFrontierReady": bool(neutral_frontier.get("strict_neutral_bulk_ready", False)),
            "strictNeutralBulk": bool(neutral_audit.get("strict_neutral_bulk", False)),
            "directionalStrictReadyTotal": int(neutral_audit.get("directional_strict_ready_total") or 0),
            "controlQuotientCandidateCount": int(neutral_audit.get("control_quotient_candidate_count") or 0),
            "overlapNativeReportCount": int(neutral_audit.get("overlap_native_negative_control_report_count") or 0),
            "overlapNativeReceiptCount": int(neutral_audit.get("overlap_native_negative_control_receipt_count") or 0),
            "overlapNativeReceipt": bool(overlap.get("OVERLAP_NATIVE_NEGATIVE_CONTROL_RECEIPT", False)),
            "overlapNativeSpatialCandidate": bool(overlap.get("overlap_native_spatial_3d_candidate", False)),
            "overlapNativeStrictH3Candidate": bool(overlap.get("overlap_native_strict_h3_candidate", False)),
            "overlapGraphReceipt": bool(overlap_graph.get("OVERLAP_NATIVE_GRAPH_GEOMETRY_RECEIPT", False)),
            "overlapGraphSpatialCandidate": bool(overlap_graph.get("overlap_graph_spatial_3d_candidate", False)),
            "overlapGraphStrictH3Candidate": bool(overlap_graph.get("overlap_graph_strict_h3_candidate", False)),
            "overlapGraphRank3Selector": bool(
                (overlap_graph.get("rank_selection") or {}).get("rank3_selector_receipt", False)
            ),
            "overlapGraphSummary": overlap_graph.get("graph_summary") or {},
            "overlapGraphBlockers": overlap_graph.get("blockers") or [],
            "overlapGraphSweep": {
                "written": bool(overlap_graph_sweep),
                "receipt": bool(overlap_graph_sweep.get("OVERLAP_NATIVE_GRAPH_GEOMETRY_SWEEP_RECEIPT", False)),
                "caseCount": int(overlap_graph_sweep.get("case_count") or 0),
                "receiptCount": int(overlap_graph_sweep.get("graph_geometry_receipt_count") or 0),
                "spatialCandidateCount": int(overlap_graph_sweep.get("spatial_3d_candidate_count") or 0),
                "strictH3CandidateCount": int(overlap_graph_sweep.get("strict_h3_candidate_count") or 0),
                "rank3SelectorCount": int(overlap_graph_sweep.get("rank3_selector_count") or 0),
                "bestCase": overlap_graph_sweep.get("best_case") or {},
                "closestStrictRows": overlap_graph_sweep.get("closest_strict_rows") or [],
                "rankObstruction": rank_obstruction,
                "gateCoincidence": gate_coincidence,
                "blockers": overlap_graph_sweep.get("blockers") or [],
            },
            "overlapResidualGraphReceipt": bool(
                overlap_residual_graph.get("OVERLAP_RESIDUALIZED_GRAPH_GEOMETRY_RECEIPT", False)
            ),
            "overlapResidualGraphSpatialCandidate": bool(
                overlap_residual_graph.get("overlap_residual_graph_spatial_3d_candidate", False)
            ),
            "overlapResidualGraphStrictH3Candidate": bool(
                overlap_residual_graph.get("overlap_residual_graph_strict_h3_candidate", False)
            ),
            "overlapResidualGraphRank3Selector": bool(
                (overlap_residual_graph.get("rank_selection") or {}).get("rank3_selector_receipt", False)
            ),
            "overlapResidualGraphSummary": overlap_residual_graph.get("graph_summary") or {},
            "overlapResidualGraphBlockers": overlap_residual_graph.get("blockers") or [],
            "overlapResidualGraphSweep": {
                "written": bool(overlap_residual_graph_sweep),
                "receipt": bool(
                    overlap_residual_graph_sweep.get(
                        "OVERLAP_RESIDUALIZED_GRAPH_GEOMETRY_SWEEP_RECEIPT",
                        False,
                    )
                ),
                "caseCount": int(overlap_residual_graph_sweep.get("case_count") or 0),
                "receiptCount": int(overlap_residual_graph_sweep.get("residual_graph_receipt_count") or 0),
                "spatialCandidateCount": int(
                    overlap_residual_graph_sweep.get("spatial_3d_candidate_count") or 0
                ),
                "strictH3CandidateCount": int(
                    overlap_residual_graph_sweep.get("strict_h3_candidate_count") or 0
                ),
                "rank3SelectorCount": int(overlap_residual_graph_sweep.get("rank3_selector_count") or 0),
                "bestCase": overlap_residual_graph_sweep.get("best_case") or {},
                "closestStrictRows": overlap_residual_graph_sweep.get("closest_strict_rows") or [],
                "rankObstruction": residual_rank_obstruction,
                "gateCoincidence": residual_gate_coincidence,
                "blockers": overlap_residual_graph_sweep.get("blockers") or [],
            },
            "blockers": neutral_audit.get("blockers") or [],
            "frontierGates": neutral_frontier.get("gate_rows") or [],
            "frontierGapRows": neutral_frontier.get("gate_gap_rows") or [],
            "frontierBlockers": neutral_frontier.get("blockers") or [],
            "nextMissingReceipts": neutral_frontier.get("next_missing_receipts") or [],
        },
        "bulk": {
            "viewer": "object_h3_bulk_viewer.html",
            "objectCount": object_summary.get("object_count"),
            "reportedObjectCount": object_summary.get("reported_object_count"),
            "overlapLinks": object_summary.get("observer_overlap_link_count"),
            "fundamentalOperation": object_summary.get("fundamental_operation"),
            "dotSemantics": object_summary.get("dot_semantics"),
            "colorEncoding": object_summary.get("color_encoding"),
        },
    }


def _compact_output_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    compact: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("measurement_comparable"):
            continue
        compact.append(
            {
                "modelId": row.get("model_id"),
                "modelRole": row.get("model_role"),
                "sourceReport": row.get("source_report"),
                "binCount": row.get("bin_count"),
                "shapeCorrelation": row.get("shape_correlation"),
                "normalizedRmse": row.get("normalized_rmse"),
                "chi2PerBin": row.get("amplitude_fit_chi2_per_bin"),
                "firstPeakAbsDelta": row.get("first_peak_abs_delta"),
            }
        )
    return compact[:12]


def _compact_peak_feature_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in rows:
        if row.get("model_role") != "oph_diagnostic":
            continue
        compact.append(
            {
                "modelId": row.get("model_id"),
                "peakIndex": _int_or_none(row.get("peak_index")),
                "observedPeakEll": _float_or_none(row.get("observed_peak_ell")),
                "modelPeakEll": _float_or_none(row.get("model_peak_ell")),
                "ellDelta": _float_or_none(row.get("ell_delta")),
                "fractionalHeightDelta": _float_or_none(row.get("fractional_D_ell_delta")),
            }
        )
    return compact[:24]


def _render_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, separators=(",", ":"), default=str)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OPH CMB and Neutral Bulk Frontier</title>
<style>
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#20242a; background:#f5f7f8; }}
header {{ padding:16px 20px; background:#ffffff; border-bottom:1px solid #d8dee4; }}
h1 {{ margin:0 0 6px; font-size:21px; }}
.sub {{ color:#5e6875; font-size:13px; line-height:1.4; }}
.metrics {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }}
.metric {{ border:1px solid #cfd7df; background:#ffffff; border-radius:6px; padding:7px 9px; font-size:12px; }}
.pass {{ border-color:#7bbb8a; background:#e7f5eb; color:#14552a; }}
.open {{ border-color:#df9a92; background:#fff0ee; color:#7a211a; }}
main {{ display:grid; grid-template-columns:1.35fr .65fr; gap:12px; padding:12px; }}
section {{ background:#ffffff; border:1px solid #d8dee4; border-radius:8px; overflow:hidden; }}
section h2 {{ margin:0; padding:10px 12px; font-size:14px; border-bottom:1px solid #d8dee4; }}
svg {{ display:block; width:100%; height:380px; background:#fbfcfd; }}
.panel {{ padding:12px; }}
.gate {{ display:grid; grid-template-columns:1fr auto; gap:8px; border-bottom:1px solid #edf0f2; padding:7px 0; font-size:13px; }}
.gate:last-child {{ border-bottom:0; }}
.status {{ font-weight:600; }}
.note {{ color:#5e6875; font-size:12px; line-height:1.45; padding:8px 12px 12px; }}
.wide {{ grid-column:1/-1; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; }}
th,td {{ text-align:left; padding:6px 8px; border-bottom:1px solid #edf0f2; vertical-align:top; }}
th {{ color:#5e6875; font-weight:600; }}
ul {{ margin:8px 0 0; padding-left:18px; }}
li {{ margin:4px 0; }}
a {{ color:#1664ad; }}
@media (max-width:960px) {{ main {{ grid-template-columns:1fr; }} .wide {{ grid-column:auto; }} }}
</style>
</head>
<body>
<header>
  <h1>OPH CMB and Neutral Bulk Frontier</h1>
  <div class="sub" id="subtitle"></div>
  <div class="metrics" id="metrics"></div>
</header>
<main>
  <section class="wide"><h2>CMB TT Comparison</h2><svg id="tt"></svg><div class="note" id="ttNote"></div></section>
  <section class="wide"><h2>CMB Output Metrics</h2><div class="panel" id="cmbOutput"></div></section>
  <section class="wide"><h2>Physical CMB Frontier</h2><div class="panel" id="cmbFrontier"></div></section>
  <section><h2>Physical CMB Input Gates</h2><div class="panel" id="cmbGates"></div></section>
  <section><h2>Official Planck Likelihood Readiness</h2><div class="panel" id="officialLikelihood"></div></section>
  <section><h2>Neutral Bulk Gates</h2><div class="panel" id="neutralGates"></div></section>
  <section><h2>CMB Blockers</h2><div class="panel" id="cmbBlockers"></div></section>
  <section><h2>3D Bulk View</h2><div class="panel" id="bulkInfo"></div></section>
</main>
<script>
const DATA = {data};
const NS = "http://www.w3.org/2000/svg";
function el(n,a={{}}){{const e=document.createElementNS(NS,n);for(const[k,v]of Object.entries(a))e.setAttribute(k,v);return e;}}
function clear(s){{while(s.firstChild)s.removeChild(s.firstChild);}}
function dims(s){{const r=s.getBoundingClientRect();return [Math.max(360,r.width),Math.max(260,r.height||380)];}}
function nums(rows,key){{return rows.map(r=>r[key]).filter(v=>typeof v==='number'&&isFinite(v));}}
function sx(rows,w){{const xs=nums(rows,'ell');const min=Math.min(...xs),max=Math.max(...xs);return x=>36+(w-58)*(x-min)/Math.max(max-min,1e-12);}}
function sy(rows,h){{const ys=['observed','lcdm','finiteRepair','scaleCompressed'].flatMap(k=>nums(rows,k));const min=Math.min(...ys),max=Math.max(...ys);return y=>24+(h-54)*(1-(y-min)/Math.max(max-min,1e-12));}}
function path(rows,key,x,y){{const pts=rows.filter(r=>typeof r[key]==='number');return pts.map((r,i)=>`${{i?'L':'M'}}${{x(r.ell).toFixed(1)}} ${{y(r[key]).toFixed(1)}}`).join(' ');}}
function drawTT(){{const svg=document.getElementById('tt');clear(svg);const rows=DATA.ttBins;const[w,h]=dims(svg);svg.setAttribute('viewBox',`0 0 ${{w}} ${{h}}`);svg.appendChild(el('rect',{{x:0,y:0,width:w,height:h,fill:'#fbfcfd'}}));if(!rows.length)return;const x=sx(rows,w),y=sy(rows,h);for(const t of [0,.25,.5,.75,1]){{const yy=24+(h-54)*t;svg.appendChild(el('line',{{x1:34,y1:yy,x2:w-18,y2:yy,stroke:'#e5e9ed','stroke-width':1}}));}}rows.forEach(r=>{{if(typeof r.observed==='number')svg.appendChild(el('circle',{{cx:x(r.ell),cy:y(r.observed),r:3.4,fill:'#20242a',opacity:.78}}));}});for(const [key,color,width] of [['lcdm','#8792a0',2],['finiteRepair','#0f8b8d',2.4],['scaleCompressed','#f2a33a',2.4]]){{const d=path(rows,key,x,y);if(d)svg.appendChild(el('path',{{d,fill:'none',stroke:color,'stroke-width':width,opacity:.9}}));}}svg.appendChild(el('text',{{x:42,y:20,fill:'#20242a','font-size':12}})).textContent='black: observed Planck bins, gray: LCDM baseline, teal/orange: OPH diagnostic transfer curves';document.getElementById('ttNote').textContent=`${{rows.length}} TT bins. These are measurement-comparable diagnostic curves; physical CMB prediction is ${{DATA.claims.physicalCmbPrediction}}.`;}}
function metric(label,value,open){{return `<span class="metric ${{open?'open':'pass'}}">${{label}}: ${{value}}</span>`;}}
function gate(label,ok,detail=''){{return `<div class="gate"><span>${{label}}<br><span class="sub">${{detail||''}}</span></span><span class="status ${{ok?'pass':'open'}}">${{ok?'pass':'open'}}</span></div>`;}}
function list(items){{return items&&items.length?`<ul>${{items.slice(0,14).map(x=>`<li><code>${{x}}</code></li>`).join('')}}</ul>`:'<div class="sub">none</div>';}}
function fmt(v){{if(typeof v==='number'&&isFinite(v))return Math.abs(v)>=100?String(Math.round(v)):v.toPrecision(4);return v??'n/a';}}
function gapRowsTable(rows){{if(!rows||!rows.length)return '<div class="sub">none</div>';return `<table><thead><tr><th>gate</th><th>missing receipt</th><th>current evidence</th><th>action</th></tr></thead><tbody>${{rows.map(r=>`<tr><td><code>${{r.gate||''}}</code></td><td>${{r.missing_receipt||'n/a'}}</td><td>${{r.current_evidence||'n/a'}}</td><td>${{r.action_surface||'n/a'}}</td></tr>`).join('')}}</tbody></table>`;}}
function closestRowsTable(rows){{if(!rows||!rows.length)return '<div class="sub">none</div>';return `<table><thead><tr><th>source</th><th>params</th><th>score</th><th>dim/model</th><th>missing strict gates</th></tr></thead><tbody>${{rows.slice(0,8).map(r=>`<tr><td>${{String(r.source_run_dir||'n/a').split('/').slice(-1)[0]}}</td><td>seed=${{r.seed??'n/a'}} max=${{r.max_model_points??'n/a'}} k=${{r.k_neighbors??'n/a'}}${{r.remove_modes!==undefined?' remove='+r.remove_modes:''}}</td><td>${{r.gate_score??'n/a'}}</td><td>${{fmt(r.median_dimension)}} / ${{r.selected_model||'n/a'}}</td><td>${{(r.missing_strict_gates||[]).map(x=>`<code>${{x}}</code>`).join(', ')||'none'}}</td></tr>`).join('')}}</tbody></table>`;}}
function cmbOutputTable(output){{const rows=output.rows||[];const peaks=output.peakFeatures||[];const best=output.bestOphDiagnosticModel||{{}};const residual=output.bestOphResidualSummary||{{}};const peakSummary=output.bestOphPeakFeatureSummary||{{}};const residualBlock=residual.available?`<div class="gate"><span>best OPH residual bins</span><span>${{residual.bin_count??'n/a'}}</span></div><div class="gate"><span>best OPH RMS sigma residual</span><span>${{fmt(residual.rms_sigma_residual)}}</span></div><div class="gate"><span>best OPH max abs sigma residual</span><span>${{fmt(residual.max_abs_sigma_residual)}} @ ell ${{fmt(residual.max_abs_sigma_ell)}}</span></div>`:`<div class="gate"><span>best OPH residual rows</span><span>not available</span></div>`;const peakBlock=peakSummary.available?`<div class="gate"><span>best OPH acoustic peak features<br><span class="sub">binned peak positions and heights, diagnostic only</span></span><span>${{peakSummary.peak_count??0}} peaks, mean |d ell| ${{fmt(peakSummary.mean_abs_peak_ell_delta)}}, mean |d height| ${{fmt(peakSummary.mean_abs_peak_height_fractional_delta)}}</span></div>`:`<div class="gate"><span>best OPH acoustic peak features</span><span>not available</span></div>`;const intro=`<div class="gate"><span>output comparison receipt</span><span>${{output.comparisonReceipt?'true':'false'}}</span></div><div class="gate"><span>physical prediction receipt</span><span>${{output.predictionReceipt?'true':'false'}}</span></div><div class="gate"><span>best OPH diagnostic</span><span>${{best.model_id||'n/a'}} / chi2/bin ${{fmt(best.amplitude_fit_chi2_per_bin)}}</span></div>${{residualBlock}}${{peakBlock}}`;const modelTable=rows.length?`<table><thead><tr><th>model</th><th>role</th><th>bins</th><th>chi2/bin</th><th>corr</th><th>peak delta</th></tr></thead><tbody>${{rows.map(r=>`<tr><td>${{r.modelId}}</td><td>${{r.modelRole}}</td><td>${{r.binCount??'n/a'}}</td><td>${{fmt(r.chi2PerBin)}}</td><td>${{fmt(r.shapeCorrelation)}}</td><td>${{fmt(r.firstPeakAbsDelta)}}</td></tr>`).join('')}}</tbody></table>`:'<div class="sub">No measurement-comparable output rows found.</div>';const peakTable=peaks.length?`<h3>OPH Peak Features</h3><table><thead><tr><th>model</th><th>peak</th><th>observed ell</th><th>model ell</th><th>d ell</th><th>d height</th></tr></thead><tbody>${{peaks.map(r=>`<tr><td>${{r.modelId}}</td><td>${{r.peakIndex??'n/a'}}</td><td>${{fmt(r.observedPeakEll)}}</td><td>${{fmt(r.modelPeakEll)}}</td><td>${{fmt(r.ellDelta)}}</td><td>${{fmt(r.fractionalHeightDelta)}}</td></tr>`).join('')}}</tbody></table>`:'';return intro+modelTable+peakTable;}}
function cmbFrontierPanel(frontier){{if(!frontier||!frontier.written)return '<div class="sub">No physical-CMB frontier report in this pack.</div>';const rows=frontier.gateRows||[];const gates=rows.map(r=>gate(r.gate,Boolean(r.passed),String(r.detail||''))).join('');return [
gate('frontier report written',Boolean(frontier.written)),
gate('physical CMB prediction ready',Boolean(frontier.ready)),
gate('output comparison receipt',Boolean(frontier.comparisonReceipt)),
gate('physical prediction receipt',Boolean(frontier.predictionReceipt)),
gates,
'<h3>Hard-gate gaps</h3>',
gapRowsTable(frontier.gapRows||[]),
'<h3>Frontier blockers</h3>',
list(frontier.blockers),
'<h3>Next missing receipts</h3>',
list((frontier.nextMissingReceipts||[]).map(r=>r.blocker||r.next_step||String(r)))
].join('');}}
function officialLikelihoodPanel(info){{const paths=info.dataPaths||[];const pathRows=paths.map(r=>`<tr><td><code>${{r.env_var}}</code></td><td>${{r.configured?'true':'false'}}</td><td>${{r.exists?'true':'false'}}</td></tr>`).join('');return [
gate('readiness report written',Boolean(info.written)),
gate('official likelihood execution ready',Boolean(info.executionReady)),
gate('official clik API available',Boolean(info.clikApiAvailable)),
gate('Planck likelihood data path configured',Boolean(info.dataPathsConfigured)),
gate('CAMB importable',Boolean(info.cambAvailable)),
gate('Cobaya importable',Boolean(info.cobayaAvailable)),
'<h3>Environment paths</h3>',
pathRows?`<table><thead><tr><th>env</th><th>configured</th><th>exists</th></tr></thead><tbody>${{pathRows}}</tbody></table>`:'<div class="sub">none</div>',
'<h3>Readiness blockers</h3>',
list(info.blockers)
].join('');}}
function rankObstruction(info){{if(!info||!info.available)return '<div class="sub">No rank-obstruction summary in this pack.</div>';const counts=info.largest_gap_rank_counts||{{}};const countRows=Object.entries(counts).map(([k,v])=>`<tr><td>${{k}}</td><td>${{v}}</td></tr>`).join('');return [
gate('primary obstruction',info.primary_obstruction!=='no_independent_rank3_selector',String(info.primary_obstruction||'n/a')),
gate('dominant largest-gap rank',String(info.dominant_largest_gap_rank)==='3',String(info.dominant_largest_gap_rank??'n/a')),
gate('dominant model-order consensus rank',String(info.dominant_model_order_consensus_rank)==='3',String(info.dominant_model_order_consensus_rank??'n/a')),
gate('rank-3 selector count',(info.rank3_selector_count||0)>0,String(info.rank3_selector_count||0)+' / '+String(info.case_count||0)),
gate('model-order rank-3 selector count',(info.model_order_rank3_selector_count||0)>0,String(info.model_order_rank3_selector_count||0)+' / '+String(info.case_count||0)),
gate('nontrivial rank-3 selector count',(info.nontrivial_rank3_selector_count||0)>0,String(info.nontrivial_rank3_selector_count||0)+' / '+String(info.case_count||0)),
gate('nontrivial model-order rank-3 selector count',(info.nontrivial_model_order_rank3_selector_count||0)>0,String(info.nontrivial_model_order_rank3_selector_count||0)+' / '+String(info.case_count||0)),
`<div class="gate"><span>max rank-3 explained variance<br><span class="sub">all cases</span></span><span>${{fmt(info.max_rank3_cumulative_explained_variance)}}</span></div>`,
`<div class="gate"><span>max nontrivial rank-3 explained variance<br><span class="sub">Perron/common mode excluded</span></span><span>${{fmt(info.max_nontrivial_rank3_cumulative_explained_variance)}}</span></div>`,
`<div class="gate"><span>median effective rank<br><span class="sub">all cases</span></span><span>${{fmt(info.median_effective_rank)}}</span></div>`,
`<div class="gate"><span>median nontrivial effective rank</span><span>${{fmt(info.median_nontrivial_effective_rank)}}</span></div>`,
`<div class="gate"><span>spatial max rank-3 explained variance</span><span>${{fmt(info.spatial_max_rank3_cumulative_explained_variance)}}</span></div>`,
`<div class="gate"><span>spatial median effective rank</span><span>${{fmt(info.spatial_median_effective_rank)}}</span></div>`,
countRows?`<table><thead><tr><th>largest gap rank</th><th>cases</th></tr></thead><tbody>${{countRows}}</tbody></table>`:''
].join('');}}
function witnessRows(rows){{if(!rows||!rows.length)return '<div class="sub">none</div>';return `<table><thead><tr><th>source</th><th>params</th><th>dim/model</th><th>rank gates</th></tr></thead><tbody>${{rows.slice(0,6).map(r=>`<tr><td>${{String(r.source_run_dir||'n/a').split('/').slice(-1)[0]}}</td><td>seed=${{r.seed??'n/a'}} max=${{r.max_model_points??'n/a'}} k=${{r.k_neighbors??'n/a'}}${{r.remove_modes!==undefined?' remove='+r.remove_modes:''}}</td><td>${{fmt(r.median_dimension)}} / ${{r.selected_model||'n/a'}}</td><td>rank3=${{r.rank3_selector?'true':'false'}} nontriv=${{r.nontrivial_rank3_selector?'true':'false'}} strict=${{r.strict_h3_candidate?'true':'false'}}</td></tr>`).join('')}}</tbody></table>`;}}
function gateCoincidence(info){{if(!info||!info.available)return '<div class="sub">No gate-coincidence summary in this pack.</div>';return [
gate('spatial-H3 geometry rows',(info.spatial_h3_geometry_count||0)>0,String(info.spatial_h3_geometry_count||0)+' / '+String(info.case_count||0)),
gate('independent rank-3 rows',(info.independent_rank3_selector_count||0)>0,String(info.independent_rank3_selector_count||0)+' / '+String(info.case_count||0)),
gate('nontrivial rank-3 rows',(info.nontrivial_rank3_selector_count||0)>0,String(info.nontrivial_rank3_selector_count||0)+' / '+String(info.case_count||0)),
gate('spatial-H3 plus independent rank-3',(info.spatial_h3_independent_rank3_selector_count||0)>0,String(info.spatial_h3_independent_rank3_selector_count||0)+' cases'),
gate('spatial-H3 plus nontrivial rank-3',(info.spatial_h3_nontrivial_rank3_selector_count||0)>0,String(info.spatial_h3_nontrivial_rank3_selector_count||0)+' cases'),
gate('strict-H3 candidates',(info.strict_h3_candidate_count||0)>0,String(info.strict_h3_candidate_count||0)+' cases'),
`<h4>Best spatial-H3 rows</h4>${{witnessRows(info.best_spatial_h3_rows||[])}}`,
`<h4>Best nontrivial rank-3 rows</h4>${{witnessRows(info.best_nontrivial_rank3_rows||[])}}`,
`<h4>Coincidence rows</h4>${{witnessRows(info.coincidence_rows||[])}}`
].join('');}}
function init(){{document.getElementById('subtitle').textContent=DATA.runDir;const c=DATA.claims;document.getElementById('metrics').innerHTML=[
metric('mini universe',c.workingMiniUniverse,!c.workingMiniUniverse),
metric('physical CMB',c.physicalCmbPrediction,!c.physicalCmbPrediction),
metric('CMB output',c.physicalCmbOutputComparison,!c.physicalCmbOutputComparison),
metric('CMB promotion',c.physicalCmbPromotionReady,!c.physicalCmbPromotionReady),
metric('strict neutral bulk',c.strictNeutralBulk,!c.strictNeutralBulk),
metric('neutral frontier',c.strictNeutralFrontierReady,!c.strictNeutralFrontierReady),
metric('theorem H3 bulk',c.theoremAssistedH3Bulk,!c.theoremAssistedH3Bulk),
metric('overlap links',c.overlapLinks,false)
].join('');drawTT();document.getElementById('cmbOutput').innerHTML=cmbOutputTable(DATA.cmb.output||{{}});document.getElementById('cmbFrontier').innerHTML=cmbFrontierPanel(DATA.cmb.frontier||{{}});document.getElementById('officialLikelihood').innerHTML=officialLikelihoodPanel(DATA.cmb.officialLikelihood||{{}});const gates=DATA.cmb.inputStatus;document.getElementById('cmbGates').innerHTML=Object.entries(gates).map(([k,v])=>gate(k,v.physicalGatePassed,`${{v.source||''}} ${{v.rowCount?`rows=${{v.rowCount}}`:''}}`)).join('');const n=DATA.neutral;const frontier=(n.frontierGates||[]).map(r=>gate(r.gate,Boolean(r.passed),r.detail)).join('');const sweep=n.overlapGraphSweep||{{}};const bestSweep=sweep.bestCase||{{}};const residualSweep=n.overlapResidualGraphSweep||{{}};const bestResidual=residualSweep.bestCase||{{}};document.getElementById('neutralGates').innerHTML=[
gate('strict neutral bulk ready',n.strictNeutralBulkReady),
gate('strict neutral frontier ready',n.strictNeutralFrontierReady),
gate('overlap-native negative control',n.overlapNativeReceipt,`${{n.overlapNativeReceiptCount}}/${{n.overlapNativeReportCount}} audit receipts`),
gate('overlap-native spatial 3D candidate',n.overlapNativeSpatialCandidate),
gate('overlap-native strict H3 candidate',n.overlapNativeStrictH3Candidate),
gate('overlap graph geometry receipt',n.overlapGraphReceipt,`edges=${{(n.overlapGraphSummary||{{}}).edge_count??'n/a'}} components=${{(n.overlapGraphSummary||{{}}).component_count??'n/a'}}`),
gate('overlap graph spatial 3D candidate',n.overlapGraphSpatialCandidate),
gate('overlap graph strict H3 candidate',n.overlapGraphStrictH3Candidate),
gate('overlap graph rank-3 selector',n.overlapGraphRank3Selector),
gate('overlap graph sweep receipts',Boolean(sweep.receipt),String(sweep.receiptCount||0)+'/'+String(sweep.caseCount||0)+' parameter cases'),
gate('overlap graph sweep spatial 3D candidates',(sweep.spatialCandidateCount||0)>0,String(sweep.spatialCandidateCount||0)+' cases'),
gate('overlap graph sweep strict H3 candidates',(sweep.strictH3CandidateCount||0)>0,String(sweep.strictH3CandidateCount||0)+' cases'),
gate('overlap graph sweep rank-3 selectors',(sweep.rank3SelectorCount||0)>0,String(sweep.rank3SelectorCount||0)+' cases'),
gate('residualized overlap graph receipt',n.overlapResidualGraphReceipt,`edges=${{(n.overlapResidualGraphSummary||{{}}).edge_count??'n/a'}} components=${{(n.overlapResidualGraphSummary||{{}}).component_count??'n/a'}}`),
gate('residualized overlap graph spatial 3D candidate',n.overlapResidualGraphSpatialCandidate),
gate('residualized overlap graph strict H3 candidate',n.overlapResidualGraphStrictH3Candidate),
gate('residualized overlap graph rank-3 selector',n.overlapResidualGraphRank3Selector),
gate('residualized graph sweep receipts',Boolean(residualSweep.receipt),String(residualSweep.receiptCount||0)+'/'+String(residualSweep.caseCount||0)+' parameter cases'),
gate('residualized graph sweep spatial 3D candidates',(residualSweep.spatialCandidateCount||0)>0,String(residualSweep.spatialCandidateCount||0)+' cases'),
gate('residualized graph sweep strict H3 candidates',(residualSweep.strictH3CandidateCount||0)>0,String(residualSweep.strictH3CandidateCount||0)+' cases'),
gate('residualized graph sweep rank-3 selectors',(residualSweep.rank3SelectorCount||0)>0,String(residualSweep.rank3SelectorCount||0)+' cases'),
gate('directional strict ready rows',n.directionalStrictReadyTotal>0,`${{n.directionalStrictReadyTotal}} rows`),
gate('control quotient candidate rows',n.controlQuotientCandidateCount>0,`${{n.controlQuotientCandidateCount}} rows`)
].join('') + '<h3>Overlap graph sweep</h3><div class="sub">best: '+(bestSweep.source_run_dir||'n/a')+' seed='+(bestSweep.seed??'n/a')+' max='+((bestSweep.max_model_points??'n/a'))+' k='+(bestSweep.k_neighbors??'n/a')+' dim='+fmt(bestSweep.median_dimension)+' model='+(bestSweep.selected_model||'n/a')+'</div>' + '<h3>Closest strict candidates</h3>' + closestRowsTable(sweep.closestStrictRows||[]) + '<h3>Rank obstruction</h3>' + rankObstruction(sweep.rankObstruction||{{}}) + '<h3>Gate coincidence</h3>' + gateCoincidence(sweep.gateCoincidence||{{}}) + list(sweep.blockers) + '<h3>Residualized overlap graph sweep</h3><div class="sub">best: '+(bestResidual.source_run_dir||'n/a')+' seed='+(bestResidual.seed??'n/a')+' max='+((bestResidual.max_model_points??'n/a'))+' k='+(bestResidual.k_neighbors??'n/a')+' remove='+(bestResidual.remove_modes??'n/a')+' dim='+fmt(bestResidual.median_dimension)+' model='+(bestResidual.selected_model||'n/a')+'</div>' + '<h3>Closest residualized strict candidates</h3>' + closestRowsTable(residualSweep.closestStrictRows||[]) + '<h3>Residualized rank obstruction</h3>' + rankObstruction(residualSweep.rankObstruction||{{}}) + '<h3>Residualized gate coincidence</h3>' + gateCoincidence(residualSweep.gateCoincidence||{{}}) + list(residualSweep.blockers) + '<h3>Overlap graph blockers</h3>' + list(n.overlapGraphBlockers) + '<h3>Residualized graph blockers</h3>' + list(n.overlapResidualGraphBlockers) + '<h3>Frontier gates</h3>' + (frontier||'<div class="sub">none</div>') + '<h3>Hard-gate gaps</h3>' + gapRowsTable(n.frontierGapRows||[]) + '<h3>Neutral blockers</h3>' + list((n.frontierBlockers&&n.frontierBlockers.length)?n.frontierBlockers:n.blockers) + '<h3>Next missing receipts</h3>' + list(n.nextMissingReceipts);document.getElementById('cmbBlockers').innerHTML='<h3>Contract blockers</h3>'+list(DATA.cmb.contractBlockers)+'<h3>Promotion blockers</h3>'+list(DATA.cmb.promotionBlockers);const b=DATA.bulk;document.getElementById('bulkInfo').innerHTML=`<p><a href="${{b.viewer}}">Open overlap/H3 3D bulk viewer</a></p><p>${{b.fundamentalOperation||''}}</p><p>${{b.dotSemantics||''}}</p><p>${{b.colorEncoding||''}}</p><div class="gate"><span>objects</span><span>${{b.objectCount??'n/a'}}</span></div><div class="gate"><span>observer-overlap links</span><span>${{b.overlapLinks??'n/a'}}</span></div>`;}}
init();
</script>
</body>
</html>
"""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _first_float(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        value = _float_or_none(row.get(key))
        if value is not None:
            return value
    return None
