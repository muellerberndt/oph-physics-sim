from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oph_fpe.claims import (
    BRANCH_INSTANTIATION_SANITY,
    BW_KMS_BRANCH_INSTANTIATION_RECEIPT,
    CHART_LORENTZ_H3_RECEIPT,
    H3_RESPONSE_CANDIDATE_RECEIPT,
    OBJECT_BULK_POPULATION_RECEIPT,
    PHYSICAL_CMB_RECEIPT,
    PROTO_PARTICLE_RECEIPT,
    RECORD_COMMIT_RECEIPT,
    REPAIR_CORE_RECEIPT,
    SCREEN_PROXY_CMB_RECEIPT,
    with_claim_metadata,
)


def bulk_proof_certificate(run_dir: Path) -> dict[str, Any]:
    """Build a tiered OPH 3D-bulk proof certificate from run receipts.

    This report intentionally separates the paper-side Lorentz/H3 chart result
    from neutral third-person reconstruction, particle emergence, and physical
    CMB prediction. The OPH papers derive the chart route from support-visible
    BW cap modular flow; finite simulation receipts then test whether observer
    records and defects populate that chart under controls.
    """

    root = Path(run_dir)
    emergence = _read_json(root / "emergence_status_report.json")
    ladder = _read_json(root / "receipt_ladder_report.json")
    paper_chart = _read_json(root / "paper_3d_bulk_chart_report.json")
    conformal_chart = _read_json(root / "conformal_h3_spatial_chart_report.json")
    transition_selection = _read_json(root / "transition_selection_report.json")
    neutral = _read_json(root / "bulk_reconstruction_report.json")
    cmb_lite = _read_json(root / "cmb_lite_comparison_report.json")
    cl = _read_json(root / "cl_comparison_report.json")
    scale_compressed = _read_json(root / "scale_compressed_repair_report.json")
    scale_compressed_cmb = _read_json(root / "scale_compressed_cmb_camb_report.json")
    scale_compressed_particle = _read_json(root / "scale_compressed_particle_report.json")
    particle = _read_json(root / "particle_likeness_report.json")
    controlled_particle = _read_json(root / "controlled_defect_particle_assay_report.json")
    object_chart_name, object_chart = _best_object_chart_report(root)

    repair_core = _ladder_passed(ladder, "R0") or _truthy(emergence, "final_phi_zero")
    record_commit = _ladder_passed(ladder, "R1") or _truthy(emergence, "records_committed")
    bw_kms = _truthy_any(
        emergence,
        "BW_KMS_DIRECT_2PI_RECEIPT",
        "state_derived_correct_beats_controls",
        "state_derived_selected_2pi",
    ) or _ladder_passed(ladder, "R2") or bool(
        paper_chart.get("bw_2pi_cap_flow_receipt", False)
        or (
            transition_selection.get("primary_source") == "kms_collar_transport_response"
            and transition_selection.get("two_pi_selected", False)
            and not transition_selection.get("response_degenerate", False)
        )
    )
    chart = _truthy_any(
        emergence,
        "PAPER_THEOREM_3D_BULK_CHART_RECEIPT",
        "CHART_LEVEL_CONFORMAL_LORENTZ_RECEIPT",
        "CHART_LORENTZ_H3_RECEIPT",
    ) or _ladder_passed(ladder, "R3") or bool(
        paper_chart.get("PAPER_THEOREM_3D_BULK_CHART_RECEIPT", False)
        or paper_chart.get("paper_theorem_3d_bulk_chart_receipt", False)
        or conformal_chart.get("conformal_h3_spatial_chart_receipt", False)
    )
    h3_response = _truthy_any(
        emergence,
        "H3_RESPONSE_CANDIDATE_RECEIPT",
        "H3_RESPONSE_CONTROL_SEPARATION_RECEIPT",
        "modular_response_h3_candidate_receipt",
    ) or _ladder_passed(ladder, "R4")
    h3_object_preview = _truthy_any(
        emergence,
        "THEOREM_ASSISTED_H3_OBJECT_PREVIEW_RECEIPT",
        "PAPER_THEOREM_ASSISTED_H3_POPULATED_CHART_RECEIPT",
        "observer_chart_object_h3_receipt",
    ) or _truthy_any(
        object_chart,
        "THEOREM_ASSISTED_H3_OBJECT_PREVIEW_RECEIPT",
        "observer_chart_object_h3_receipt",
        "observer_chart_object_h3_median_receipt",
    ) or _ladder_passed(ladder, "R5")
    object_nonboundary_population = _truthy_any(
        emergence,
        "OBJECT_H3_NONBOUNDARY_POPULATION_RECEIPT",
        "OBJECT_BULK_POPULATION_RECEIPT",
        "observer_chart_bulk_population_receipt",
    ) or _truthy_any(
        object_chart,
        "OBJECT_H3_NONBOUNDARY_POPULATION_RECEIPT",
        "OBJECT_BULK_POPULATION_RECEIPT",
        "observer_chart_bulk_population_receipt",
        "localized_nonboundary_bulk_population_receipt",
    )
    theorem_assisted_chart_preview = bool(
        chart and bw_kms and h3_response and (h3_object_preview or object_nonboundary_population)
    )
    theorem_assisted_nonboundary_population = bool(
        chart and bw_kms and h3_response and object_nonboundary_population
    )
    strict_neutral_bulk = bool(
        neutral.get("bulk_3d_established", False)
        or emergence.get("strict_blind_observer_bulk_receipt", False)
        or emergence.get("neutral_bulk_3d_established", False)
    )
    screen_cmb = bool(
        emergence.get("SCREEN_PROXY_CMB_RECEIPT", False)
        or _ladder_passed(ladder, "R7")
        or cl
        or cmb_lite
    )
    physical_cmb = bool(
        emergence.get("physical_cmb_prediction", False)
        or cmb_lite.get("physical_cmb_prediction", False)
        or cl.get("physical_cmb_prediction", False)
    )
    production_particle = bool(
        emergence.get("particle_matter_receipt", False)
        or particle.get("particle_matter_receipt", False)
        or controlled_particle.get("physical_particle_emergence", False)
    )
    scale_operator = bool(scale_compressed.get("scale_compressed_operator_receipt", False))
    scale_round_trace = bool(scale_compressed.get("repair_round_trace_receipt", False))
    scale_h3 = scale_compressed.get("h3_preview") or {}
    scale_cmb_params = scale_compressed.get("cmb_parameter_readouts") or {}
    scale_h3_preview = bool(
        scale_operator
        and scale_h3.get("populated_h3_preview_receipt", False)
        and scale_h3.get("cap_profile_receipt", False)
    )
    scale_particle_preview = bool(
        scale_compressed_particle.get("particle_preview_receipt", False)
        or ((scale_compressed.get("particle_preview") or {}).get("particle_preview_receipt", False))
    )
    scale_compressed_measurement_cmb = bool(
        scale_compressed_cmb.get("measurement_comparable_cmb_curve", False)
        and scale_compressed_cmb.get("screen_camb_transfer_receipt", False)
    )
    scale_physical_cmb = bool(
        scale_compressed.get("physical_cmb_prediction", False)
        or scale_compressed_cmb.get("physical_cmb_prediction", False)
    )
    screen_cmb = bool(screen_cmb or scale_compressed_measurement_cmb)
    physical_cmb = bool(physical_cmb or scale_physical_cmb)

    tiers = {
        "T0_finite_repair_core": _tier(
            REPAIR_CORE_RECEIPT,
            repair_core,
            "Finite overlap repair settled the declared mismatch/normal-form surface.",
        ),
        "T1_record_commit": _tier(
            RECORD_COMMIT_RECEIPT,
            record_commit,
            "Observer-facing records committed and can be read as finite record algebra data.",
        ),
        "T2_bw_kms_2pi_branch": _tier(
            BW_KMS_BRANCH_INSTANTIATION_RECEIPT,
            bw_kms,
            "Finite support-visible cap/collar branch instantiates BW/KMS 2*pi cap transport.",
        ),
        "T3_chart_lorentz_h3": _tier(
            CHART_LORENTZ_H3_RECEIPT,
            chart,
            "Paper-side conformal route Conf+(S2) -> SO+(3,1) -> H3 spatial chart is instantiated.",
        ),
        "T4_h3_response_controls": _tier(
            H3_RESPONSE_CANDIDATE_RECEIPT,
            h3_response,
            "Observer/cap response signal populates the H3 chart better than implemented controls.",
        ),
        "T5a_theorem_assisted_h3_object_preview": _tier(
            "THEOREM_ASSISTED_H3_OBJECT_PREVIEW_RECEIPT",
            theorem_assisted_chart_preview,
            "Persistent observer-facing objects can be displayed in the theorem-side H3 chart under controls.",
        ),
        "T5b_nonboundary_h3_object_population": _tier(
            OBJECT_BULK_POPULATION_RECEIPT,
            theorem_assisted_nonboundary_population,
            "Persistent observer-facing objects populate the theorem-side H3 chart and are not boundary-dominated.",
        ),
        "T5c_scale_compressed_h3_preview": _tier(
            "SCALE_COMPRESSED_H3_PREVIEW_RECEIPT",
            scale_h3_preview,
            "Scale-compressed logical repair-round branch populates the paper-side H3 chart as a preview artifact.",
        ),
        "T6_strict_neutral_third_person_bulk": _tier(
            "STRICT_NEUTRAL_3D_BULK_RECEIPT",
            strict_neutral_bulk,
            "Neutral observer-record reconstruction establishes a third-person 3D bulk without chart prior.",
        ),
        "T7_production_particles": _tier(
            PROTO_PARTICLE_RECEIPT,
            production_particle,
            "Production defects satisfy localization, transport, fusion/scattering, and bulk-worldline gates.",
        ),
        "T7a_scale_compressed_particle_preview": _tier(
            "SCALE_COMPRESSED_PARTICLE_PREVIEW_RECEIPT",
            scale_particle_preview,
            "Scale-compressed localized defect/worldline preview exists, separate from production particle matter.",
        ),
        "T8_screen_cmb_proxy": _tier(
            SCREEN_PROXY_CMB_RECEIPT,
            screen_cmb,
            "Freezeout/screen angular spectra are available for shape-level measurement comparison.",
        ),
        "T8b_scale_compressed_camb_transfer": _tier(
            "SCALE_COMPRESSED_CMB_CAMB_TRANSFER_RECEIPT",
            scale_compressed_measurement_cmb,
            "Scale-compressed OPH readouts have been passed through CAMB for a measurement-comparable TT curve.",
        ),
        "T9_physical_cmb_prediction": _tier(
            PHYSICAL_CMB_RECEIPT,
            physical_cmb,
            "A physical CMB prediction is emitted through Boltzmann/likelihood-ready finite OPH kernels.",
        ),
    }

    report = {
        "mode": "oph_3d_bulk_and_measurement_proof_certificate_v0",
        "run_path": str(root),
        "proof_tiers": tiers,
        "chart_level_3p1_lorentz_kinematics_established": bool(chart and bw_kms),
        "theorem_assisted_h3_object_preview_established": theorem_assisted_chart_preview,
        "theorem_assisted_h3_nonboundary_population_established": theorem_assisted_nonboundary_population,
        "theorem_assisted_h3_populated_chart_established": theorem_assisted_nonboundary_population,
        "scale_compressed_operator_receipt": scale_operator,
        "scale_compressed_repair_round_trace_receipt": scale_round_trace,
        "scale_compressed_h3_preview_established": scale_h3_preview,
        "scale_compressed_measurement_comparable_cmb_curve": scale_compressed_measurement_cmb,
        "scale_compressed_particle_preview_established": scale_particle_preview,
        "strict_neutral_third_person_bulk_established": strict_neutral_bulk,
        "bulk_3d_established_theorem_assisted": theorem_assisted_nonboundary_population,
        "bulk_3d_established_strict": strict_neutral_bulk,
        "screen_cmb_proxy_available": screen_cmb,
        "physical_cmb_prediction": physical_cmb,
        "production_particle_matter_receipt": production_particle,
        "selected_object_chart_report": object_chart_name,
        "selected_object_chart_incidence_mode": object_chart.get("postprocess_incidence_mode")
        or object_chart.get("incidence_mode"),
        "selected_object_chart_summary": {
            "object_count": object_chart.get("object_count"),
            "localized_object_count": object_chart.get("localized_object_count"),
            "localized_not_boundary_object_count": object_chart.get("localized_not_boundary_object_count"),
            "median_h3_compactness_normalized": object_chart.get("median_h3_compactness_normalized"),
            "median_s2_boundary_compactness_normalized": object_chart.get(
                "median_s2_boundary_compactness_normalized"
            ),
            "median_shuffled_h3_compactness_normalized": object_chart.get(
                "median_shuffled_h3_compactness_normalized"
            ),
            "h3_beats_shuffled_incidence_robust": object_chart.get("h3_beats_shuffled_incidence_robust"),
            "observer_chart_bulk_population_receipt": object_chart.get("observer_chart_bulk_population_receipt"),
        },
        "scale_compressed_summary": {
            "logical_repair_rounds": scale_compressed.get("logical_repair_rounds"),
            "eta_R": scale_cmb_params.get("eta_R"),
            "n_s": scale_cmb_params.get("n_s"),
            "q_IR": scale_cmb_params.get("q_IR"),
            "ell_IR": scale_cmb_params.get("ell_IR"),
            "N_CRC_predicted_from_P": scale_cmb_params.get("N_CRC_predicted_from_P"),
            "N_CRC_declared": scale_cmb_params.get("N_CRC_declared"),
            "relative_error_gprime_vs_N_CRC": scale_cmb_params.get("relative_error_gprime_vs_N_CRC"),
            "h3_object_count": scale_h3.get("object_count"),
            "h3_cap_count": scale_h3.get("cap_count"),
            "particle_worldline_count": (
                scale_compressed_particle.get("worldline_count")
                or ((scale_compressed.get("particle_preview") or {}).get("particle_worldline_count"))
            ),
            "camb_ir_shape_correlation": (
                ((scale_compressed_cmb.get("comparison") or {}).get("scale_compressed_ir_kernel") or {}).get(
                    "shape_correlation"
                )
            ),
            "camb_ir_normalized_rmse": (
                ((scale_compressed_cmb.get("comparison") or {}).get("scale_compressed_ir_kernel") or {}).get(
                    "normalized_rmse"
                )
            ),
            "camb_ir_chi2_per_bin": (
                ((scale_compressed_cmb.get("comparison") or {}).get("scale_compressed_ir_kernel") or {}).get(
                    "best_fit_column_chi2_per_bin"
                )
            ),
            "physical_cmb_prediction": scale_physical_cmb,
            "strict_neutral_bulk": bool(
                scale_compressed.get("strict_neutral_bulk", False)
                or scale_h3.get("strict_neutral_third_person_bulk_established", False)
            ),
        },
        "paper_chart_summary": {
            "paper_theorem_3d_bulk_chart_receipt": paper_chart.get("paper_theorem_3d_bulk_chart_receipt")
            or paper_chart.get("PAPER_THEOREM_3D_BULK_CHART_RECEIPT"),
            "chart_level_conformal_lorentz_receipt": paper_chart.get("chart_level_conformal_lorentz_receipt")
            or conformal_chart.get("conformal_h3_spatial_chart_receipt"),
            "bw_2pi_cap_flow_receipt": paper_chart.get("bw_2pi_cap_flow_receipt")
            or transition_selection.get("two_pi_selected"),
            "lorentz_group": paper_chart.get("lorentz_group"),
            "spatial_homogeneous_space": paper_chart.get("spatial_homogeneous_space"),
            "h3_spatial_dimension_from_boost_orbit": paper_chart.get("h3_spatial_dimension_from_boost_orbit")
            or ((conformal_chart.get("lorentz_algebra_report") or {}).get("h3_spatial_dimension_from_boost_orbit")),
            "h3_chart_spatial_dimension": paper_chart.get("h3_chart_spatial_dimension")
            or ((conformal_chart.get("h3_chart_report") or {}).get("spatial_dimension")),
            "finite_point_cloud_dimension_estimator_used": paper_chart.get(
                "finite_point_cloud_dimension_estimator_used", False
            ),
        },
        "paper_alignment": {
            "screen_role": "S2 is the observer-facing cap/symmetry chart, not a raw point-cloud proof of dimension.",
            "lorentz_route": "support-visible BW/KMS cap flow with s=2*pi*t gives Conf+(S2) ~= SO+(3,1).",
            "spatial_chart": "H3 is the spatial homogeneous chart SO+(3,1)/SO(3).",
            "finite_gate": "finite runs must separately show observer records/objects/defects populate that chart under controls.",
            "scale_compressed_branch": (
                "Logical scale compression can expose OPH repair-round/CMB readouts and H3 previews, but it "
                "does not replace strict neutral observer-record reconstruction."
            ),
        },
        "claim_boundary": (
            "Tiered OPH proof/readout certificate. T2-T3 establish the chart-level 3+1D Lorentz/H3 "
            "branch for this run. T5a is theorem-assisted H3 preview evidence from observer objects. "
            "T5b is stricter non-boundary H3 object population evidence. T5c is a scale-compressed "
            "logical repair-round H3 preview and is intentionally not promoted to T5b/T6. "
            "T6 is stricter neutral third-person bulk reconstruction and may remain false even when T5 passes. "
            "T8/T8b are measurement-facing screen/CAMB-transfer data only. T9 is a physical CMB prediction and remains false "
            "until finite OPH kernels feed a Boltzmann/likelihood-ready pipeline."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=BRANCH_INSTANTIATION_SANITY,
        receipt="OPH_3D_BULK_PROOF_CERTIFICATE",
        physical_claim=False,
        observable_id="tiered_3d_bulk_and_measurement_receipts",
        fit_objective="receipt_gate_summary",
    )


def write_bulk_proof_certificate(run_dir: Path, out: Path | None = None) -> dict[str, Any]:
    report = bulk_proof_certificate(run_dir)
    out_path = Path(out) if out is not None else Path(run_dir) / "bulk_proof_certificate_report.json"
    if out_path.suffix.lower() != ".json":
        out_path = out_path / "bulk_proof_certificate_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path = out_path.with_suffix(".md")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    return report


def _tier(receipt_name: str, passed: bool, description: str) -> dict[str, Any]:
    return {"receipt_name": receipt_name, "passed": bool(passed), "description": description}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _ladder_passed(ladder: dict[str, Any], key: str) -> bool:
    return bool(((ladder.get("receipts", {}) or {}).get(key, {}) or {}).get("passed", False))


def _truthy(data: dict[str, Any], key: str) -> bool:
    return bool(data.get(key, False))


def _truthy_any(data: dict[str, Any], *keys: str) -> bool:
    return any(_truthy(data, key) for key in keys)


def _best_object_chart_report(root: Path) -> tuple[str | None, dict[str, Any]]:
    names = [
        "observer_chart_object_h3_lineage_report.json",
        "observer_chart_object_h3_transition_history_report.json",
        "observer_chart_object_h3_observer_transition_mixture_report.json",
        "observer_chart_object_h3_recomputed.json",
        "observer_chart_object_h3_report.json",
    ]
    paths: list[Path] = []
    for name in names:
        path = root / name
        if path.exists():
            paths.append(path)
    for path in sorted(root.glob("observer_chart_object_h3_*_report.json")):
        if path.name not in {p.name for p in paths}:
            paths.append(path)

    candidates: list[tuple[str, dict[str, Any]]] = []
    for path in paths:
        report = _read_json(path)
        if report:
            candidates.append((path.name, report))
    if not candidates:
        return None, {}

    def score(item: tuple[str, dict[str, Any]]) -> tuple[float, float, float, float, float, float, float]:
        name, report = item
        median_h3 = _float_or(report.get("median_h3_compactness_normalized"), 1.0e9)
        median_shuffle = _float_or(report.get("median_shuffled_h3_compactness_normalized"), 1.0e-12)
        if median_shuffle > 0.0 and median_h3 < 1.0e8:
            h3_margin = (median_shuffle - median_h3) / median_shuffle
        else:
            h3_margin = -1.0e9
        lineage_or_transition = report.get("postprocess_incidence_mode") in {
            "record_sector_checkpoint_lineage",
            "transition_history",
            "observer_transition_mixture_cluster",
        } or report.get("incidence_mode") in {
            "record_sector_checkpoint_lineage",
            "transition_history",
            "observer_transition_mixture_cluster",
        }
        return (
            float(bool(report.get("observer_chart_bulk_population_receipt", False))),
            float(bool(report.get("OBJECT_H3_NONBOUNDARY_POPULATION_RECEIPT", False))),
            float(bool(report.get("THEOREM_ASSISTED_H3_OBJECT_PREVIEW_RECEIPT", False))),
            float(bool(report.get("h3_beats_shuffled_incidence_robust", False))),
            float(bool(lineage_or_transition)),
            _float_or(report.get("localized_not_boundary_object_count"), 0.0),
            h3_margin,
        )

    return max(candidates, key=score)


def _float_or(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# OPH 3D Bulk And Measurement Proof Certificate",
        "",
        f"- run: `{report['run_path']}`",
        f"- chart-level 3+1D Lorentz/H3: `{str(report['chart_level_3p1_lorentz_kinematics_established']).lower()}`",
        f"- theorem-assisted H3 object preview: `{str(report['theorem_assisted_h3_object_preview_established']).lower()}`",
        f"- theorem-assisted non-boundary H3 population: `{str(report['theorem_assisted_h3_nonboundary_population_established']).lower()}`",
        f"- scale-compressed H3 preview: `{str(report['scale_compressed_h3_preview_established']).lower()}`",
        f"- strict neutral third-person bulk: `{str(report['strict_neutral_third_person_bulk_established']).lower()}`",
        f"- screen CMB proxy available: `{str(report['screen_cmb_proxy_available']).lower()}`",
        f"- scale-compressed CAMB TT curve: `{str(report['scale_compressed_measurement_comparable_cmb_curve']).lower()}`",
        f"- physical CMB prediction: `{str(report['physical_cmb_prediction']).lower()}`",
        f"- production particle matter receipt: `{str(report['production_particle_matter_receipt']).lower()}`",
        f"- scale-compressed particle preview: `{str(report['scale_compressed_particle_preview_established']).lower()}`",
        "",
        "## Tiers",
        "",
    ]
    for name, tier in report["proof_tiers"].items():
        lines.append(f"- `{name}`: `{str(tier['passed']).lower()}` - {tier['receipt_name']}")
    scale = report.get("scale_compressed_summary") or {}
    if scale:
        lines.extend(
            [
                "",
                "## Scale-Compressed Readouts",
                "",
                f"- logical repair rounds: `{scale.get('logical_repair_rounds')}`",
                f"- n_s: `{scale.get('n_s')}`",
                f"- eta_R: `{scale.get('eta_R')}`",
                f"- q_IR: `{scale.get('q_IR')}`",
                f"- ell_IR: `{scale.get('ell_IR')}`",
                f"- H3 objects: `{scale.get('h3_object_count')}`",
                f"- particle worldlines: `{scale.get('particle_worldline_count')}`",
                f"- CAMB IR shape correlation: `{scale.get('camb_ir_shape_correlation')}`",
                f"- CAMB IR normalized RMSE: `{scale.get('camb_ir_normalized_rmse')}`",
                f"- CAMB IR chi2/bin: `{scale.get('camb_ir_chi2_per_bin')}`",
            ]
        )
    chart = report.get("paper_chart_summary") or {}
    if chart:
        lines.extend(
            [
                "",
                "## Paper Chart Readouts",
                "",
                f"- paper 3D chart receipt: `{chart.get('paper_theorem_3d_bulk_chart_receipt')}`",
                f"- chart-level conformal Lorentz: `{chart.get('chart_level_conformal_lorentz_receipt')}`",
                f"- BW 2pi cap-flow receipt: `{chart.get('bw_2pi_cap_flow_receipt')}`",
                f"- Lorentz group: `{chart.get('lorentz_group')}`",
                f"- spatial chart: `{chart.get('spatial_homogeneous_space')}`",
                f"- H3 boost-orbit dimension: `{chart.get('h3_spatial_dimension_from_boost_orbit')}`",
                f"- H3 chart dimension: `{chart.get('h3_chart_spatial_dimension')}`",
                f"- finite point-cloud dimension estimator used: `{chart.get('finite_point_cloud_dimension_estimator_used')}`",
            ]
        )
    lines.extend(["", "## Claim Boundary", "", str(report.get("claim_boundary", "")), ""])
    return "\n".join(lines)
