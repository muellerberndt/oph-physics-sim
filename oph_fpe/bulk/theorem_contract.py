from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oph_fpe.claims import (
    BRANCH_INSTANTIATION_SANITY,
    CAP_NORMAL_H3_CHART_RECEIPT,
    EINSTEIN_ALL_TIMELIKE_TENSOR_UPGRADE_RECEIPT,
    EINSTEIN_BRANCH_ENTRY_RECEIPT,
    EINSTEIN_BRANCH_ISSUE_503_RECEIPT,
    EINSTEIN_BRIDGE_DEPENDENCY_DISCHARGE_RECEIPT,
    EINSTEIN_BRIDGE_RUN_RECEIPTS_RECEIPT,
    EINSTEIN_FIXED_CAP_ENTROPY_STATIONARITY_RECEIPT,
    EINSTEIN_LAMBDA_CONSTANCY_CONSERVATION_RECEIPT,
    EINSTEIN_NEWTON_COUPLING_FORBIDDEN_INPUT_AUDIT_RECEIPT,
    EINSTEIN_NULL_STRESS_CHARGE_RECEIPT,
    EINSTEIN_SMALL_BALL_AREA_BRIDGE_RECEIPT,
    ISSUE_308_BW_CERTIFICATE_RECEIPT,
    MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT,
    OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_RECEIPT,
    OPH_EINSTEIN_BRIDGE_MANIFEST_RECEIPT,
    OPH_LORENTZ_THEOREM_FINITE_CONTRACT_RECEIPT,
    with_claim_metadata,
)
from oph_fpe.bulk.bw_certificate_308 import issue308_bw_certificate_report
from oph_fpe.bulk.cap_normal_h3_chart import cap_normal_h3_chart_report
from oph_fpe.bulk.einstein_bridge import einstein_bridge_manifest_report
from oph_fpe.bulk.modular_response_h3_localization import modular_response_h3_localization_report


def finite_oph_theorem_contract_report(run_dir: Path) -> dict[str, Any]:
    """Audit whether a run instantiated the finite OPH spacetime contract.

    The branch replay path can show the S2/BW/H3 chart route. The stronger
    paper-faithful simulation claim needs the finite observer-record hypotheses
    to be instantiated and visible in receipts. This report is intentionally a
    hard gate: missing receipts are blockers, not tuning hints.
    """

    root = Path(run_dir)
    theorem_core = _read_json(root / "theorem_core_receipts.json")
    emergence = _read_json(root / "emergence_status_report.json")
    h3 = _read_json(root / "modular_response_h3_report.json")
    object_chart = _read_json(root / "observer_chart_object_h3_report.json")
    observer = _read_json(root / "observer_modular_experience_report.json")
    chart = _read_json(root / "conformal_h3_spatial_chart_report.json")
    paper_chart = _read_json(root / "paper_3d_bulk_chart_report.json")
    neutral = _read_json(root / "bulk_reconstruction_report.json")
    neutral_audit = _read_json(root / "neutral_3d_bulk_audit_report.json")
    refinement = _read_json(root / "strict_neutral_bulk_frontier_report.json")
    einstein = _read_json(root / "einstein_branch_entry_report.json")
    explicit_einstein_bridge_manifest = _read_json(root / "einstein_bridge_manifest.json")
    einstein_bridge = explicit_einstein_bridge_manifest or einstein_bridge_manifest_report(root)
    issue308_bw = _read_issue308_bw_certificate(root)
    issue309_h3 = _read_issue309_cap_normal_h3_chart(root)
    issue310_h3loc = _read_issue310_modular_response_h3_localization(root)
    use_einstein_bridge_manifest = bool(explicit_einstein_bridge_manifest)
    refinement_summary = (
        refinement.get("refinement_summary")
        if isinstance(refinement.get("refinement_summary"), dict)
        else neutral_audit.get("refinement_summary")
        if isinstance(neutral_audit.get("refinement_summary"), dict)
        else {}
    )

    observer_rows = _observer_row_summary(root / "observer_views.jsonl")
    h3_gates = h3.get("h3_response_stage_gates") or {}
    wrong_scale = h3.get("wrong_scale_feature_audit") or {}
    object_blockers = list(object_chart.get("blockers") or [])

    finite_consensus = _truthy_any(
        theorem_core,
        "FINITE_CONSENSUS_THEOREM_RECEIPT",
        "finite_consensus_theorem_receipt",
    ) or _truthy_any(emergence, "FINITE_CONSENSUS_THEOREM_RECEIPT", "finite_consensus_theorem_receipt")
    record_algebra = bool(
        observer_rows["patch_observer_count"] > 0
        and observer_rows["rows_with_support_nodes"] > 0
        and observer_rows["rows_with_readout_hash"] > 0
        and observer_rows["rows_with_transition_histories"] > 0
    )
    endogenous_generator = _truthy_any(
        emergence,
        "ENDOGENOUS_MODULAR_GENERATOR_RECEIPT",
        "endogenous_modular_generator_receipt",
    )
    state_bw = _read_json(root / "bw_state_derived_report.json")
    inferred_clock = state_bw.get("inferred_modular_clock_fit") or {}
    kms_clock_fit = _truthy_any(
        emergence,
        "KMS_GEOMETRIC_CLOCK_FIT_RECEIPT",
        "kms_geometric_clock_fit_receipt",
    ) or _truthy_any(
        state_bw,
        "KMS_GEOMETRIC_CLOCK_FIT_RECEIPT",
        "kms_geometric_clock_fit_receipt",
    )
    declared_geometric_kms_branch = _declared_geometric_kms_branch_receipt(
        paper_chart=paper_chart,
        emergence=emergence,
        chart=chart,
    )
    support_visible_bw_covariance = bool(
        h3_gates.get("signal_gate", False)
        and h3_gates.get("geometry_gate", False)
        and h3_gates.get("aggregate_wrong_scale_gate", False)
        and h3_gates.get("material_feature_gate", False)
    )
    ordered_cut_pair_rigidity = bool(
        _truthy_any(emergence, "ordered_cut_pair_rigidity_receipt")
        or _truthy_any(chart, "ordered_cut_pair_rigidity_receipt")
        or _truthy_any(paper_chart, "ordered_cut_pair_rigidity_receipt")
        or (
            _truthy_any(paper_chart, "PAPER_THEOREM_3D_BULK_CHART_RECEIPT", "paper_theorem_3d_bulk_chart_receipt")
            and bool(paper_chart.get("bw_2pi_cap_flow_receipt", False))
            and bool(paper_chart.get("lorentz_algebra_receipt", False))
        )
        or (
            bool(chart.get("conformal_h3_spatial_chart_receipt", False))
            and bool(chart.get("lorentz_algebra_receipt", False))
            and bool((chart.get("cap_normal_report") or {}).get("unit_normal_receipt", False))
        )
    )
    lorentz_algebra_closure = bool(
        chart.get("lorentz_algebra_receipt", False)
        or paper_chart.get("lorentz_algebra_receipt", False)
    )
    refinement_naturality = bool(
        _truthy_any(refinement, "strict_neutral_bulk_refinement_receipt")
        or _truthy_any(refinement_summary, "strict_neutral_bulk_refinement_receipt")
        or _truthy_any(neutral_audit, "strict_neutral_bulk_refinement_receipt")
        or _truthy_any(emergence, "refinement_naturality_receipt")
    )
    h3_candidate = bool(h3.get("H3_RESPONSE_CANDIDATE_RECEIPT", False))
    object_population = bool(
        object_chart.get("OBJECT_BULK_POPULATION_RECEIPT", False)
        or object_chart.get("observer_chart_bulk_population_receipt", False)
        or issue310_h3loc.get(MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT, False)
    )
    observer_3p1d = bool(
        observer.get("OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT", False)
        or observer.get("observer_facing_3p1d_h3_experience_receipt", False)
    )
    neutral_bulk = bool(
        neutral.get("bulk_3d_established", False)
        or neutral_audit.get("strict_neutral_bulk", False)
        or refinement.get("strict_neutral_bulk", False)
        or emergence.get("strict_blind_observer_bulk_receipt", False)
        or emergence.get("neutral_bulk_3d_established", False)
    )
    einstein_issue_503 = bool(
        _truthy_any(
            einstein,
            EINSTEIN_BRANCH_ISSUE_503_RECEIPT,
            "issue_503_closed_receipt",
            "issue_503_branch_entry_closed_receipt",
        )
        or str(einstein.get("issue_503_status", "")).lower() in {"closed", "complete", "passed"}
    )
    einstein_e1_null_stress = _truthy_any(
        einstein,
        EINSTEIN_NULL_STRESS_CHARGE_RECEIPT,
        "E1_NULL_STRESS_CHARGE_RECEIPT",
        "null_generator_stress_charge_receipt",
    )
    einstein_e2_entropy = _truthy_any(
        einstein,
        EINSTEIN_FIXED_CAP_ENTROPY_STATIONARITY_RECEIPT,
        "E2_FIXED_CAP_ENTROPY_STATIONARITY_RECEIPT",
        "fixed_cap_entropy_stationarity_receipt",
    )
    einstein_e3_small_ball = _truthy_any(
        einstein,
        EINSTEIN_SMALL_BALL_AREA_BRIDGE_RECEIPT,
        "E3_SMALL_BALL_AREA_BRIDGE_RECEIPT",
        "small_ball_area_bridge_receipt",
    )
    einstein_e4_tensor = _truthy_any(
        einstein,
        EINSTEIN_ALL_TIMELIKE_TENSOR_UPGRADE_RECEIPT,
        "E4_ALL_TIMELIKE_TENSOR_UPGRADE_RECEIPT",
        "all_timelike_tensor_upgrade_receipt",
    )
    einstein_e5_lambda = _truthy_any(
        einstein,
        EINSTEIN_LAMBDA_CONSTANCY_CONSERVATION_RECEIPT,
        "E5_LAMBDA_CONSTANCY_CONSERVATION_RECEIPT",
        "lambda_constancy_conservation_receipt",
    )
    einstein_e6_newton = _truthy_any(
        einstein,
        EINSTEIN_NEWTON_COUPLING_FORBIDDEN_INPUT_AUDIT_RECEIPT,
        "E6_NEWTON_COUPLING_FORBIDDEN_INPUT_AUDIT_RECEIPT",
        "newton_coupling_forbidden_input_audit_receipt",
    )
    einstein_child_gates = {
        "E1_null_generator_stress_charge": einstein_e1_null_stress,
        "E2_fixed_cap_entropy_stationarity": einstein_e2_entropy,
        "E3_small_ball_area_bridge": einstein_e3_small_ball,
        "E4_all_timelike_tensor_upgrade": einstein_e4_tensor,
        "E5_lambda_constancy_conservation": einstein_e5_lambda,
        "E6_newton_coupling_forbidden_input_audit": einstein_e6_newton,
    }
    legacy_einstein_branch_entry = bool(einstein_issue_503 and all(einstein_child_gates.values()))
    manifest_child_gates = einstein_bridge.get("einstein_branch_entry_child_gates")
    if use_einstein_bridge_manifest and isinstance(manifest_child_gates, dict):
        einstein_child_gates = {name: bool(value) for name, value in manifest_child_gates.items()}
        einstein_e1_null_stress = bool(einstein_child_gates.get("E1_null_generator_stress_charge", False))
        einstein_e2_entropy = bool(einstein_child_gates.get("E2_fixed_cap_entropy_stationarity", False))
        einstein_e3_small_ball = bool(einstein_child_gates.get("E3_small_ball_area_bridge", False))
        einstein_e4_tensor = bool(einstein_child_gates.get("E4_all_timelike_tensor_upgrade", False))
        einstein_e5_lambda = bool(einstein_child_gates.get("E5_lambda_constancy_conservation", False))
        einstein_e6_newton = bool(
            einstein_child_gates.get("E6_newton_coupling_forbidden_input_audit", False)
        )
    einstein_bridge_dependency_discharge = _truthy_any(
        einstein_bridge,
        EINSTEIN_BRIDGE_DEPENDENCY_DISCHARGE_RECEIPT,
        "theorem_e0_dependency_discharge_receipt",
    )
    einstein_bridge_run_receipts = _truthy_any(
        einstein_bridge,
        EINSTEIN_BRIDGE_RUN_RECEIPTS_RECEIPT,
        "einstein_bridge_run_receipts_receipt",
        "all_required_receipts_theorem_tagged",
    )
    einstein_branch_entry = (
        _truthy_any(
            einstein_bridge,
            OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_RECEIPT,
            EINSTEIN_BRANCH_ENTRY_RECEIPT,
            "einstein_branch_entry_contract_receipt",
            "einstein_branch_entry_receipt",
        )
        if use_einstein_bridge_manifest
        else legacy_einstein_branch_entry
    )

    stages = {
        "C0_finite_consensus_theorem": _stage(
            finite_consensus,
            "finite overlap repair is replayed as a theorem-phase consensus certificate",
            missing=theorem_core.get("finite_consensus_missing_evidence")
            or (theorem_core.get("finite_consensus_theorem") or {}).get("missing_evidence")
            or emergence.get("finite_consensus_missing_evidence"),
        ),
        "L1_observer_record_algebra": _stage(
            record_algebra,
            "observer-like self-reading rows expose support, records, readback, and transition histories",
            details=observer_rows,
        ),
        "L2_endogenous_modular_generator": _stage(
            endogenous_generator,
            "modular generator is recovered from observer-record/cap state rather than supplied by branch replay",
            details={
                "state_derived_endogenous_modular_generator": emergence.get(
                    "state_derived_endogenous_modular_generator"
                ),
                "state_derived_selected_2pi": emergence.get("state_derived_selected_2pi"),
                "state_derived_correct_beats_controls": emergence.get("state_derived_correct_beats_controls"),
            },
        ),
        "L3_kms_modular_clock_fit": _stage(
            kms_clock_fit,
            "finite support-visible KMS/BW clock infers kappa ~= 2*pi against controls",
            details={
                "kappa_hat": inferred_clock.get("kappa_hat"),
                "kappa_95ci": inferred_clock.get("kappa_95ci"),
                "nearest_known_scale": inferred_clock.get("nearest_known_scale"),
                "clock_fit_blockers": inferred_clock.get("blockers", []),
                "transition_primary_source": emergence.get("transition_primary_source"),
                "transition_selected_label": emergence.get("transition_selected_label"),
                "transition_two_pi_over_best": emergence.get("transition_two_pi_over_best"),
            },
        ),
        "L3b_declared_geometric_kms_branch": _stage(
            declared_geometric_kms_branch,
            "paper geometric branch supplies the KMS collar/cap 2*pi normalization used by the Lorentz/H3 chart theorem",
            details={
                "paper_chart_declared_bw_2pi_cap_flow_receipt": paper_chart.get(
                    "declared_bw_2pi_cap_flow_receipt"
                ),
                "paper_chart_bw_2pi_cap_flow_receipt": paper_chart.get("bw_2pi_cap_flow_receipt"),
                "paper_chart_bw_2pi_cap_flow_source": paper_chart.get("bw_2pi_cap_flow_source"),
                "emergence_transition_primary_source": emergence.get("transition_primary_source"),
                "emergence_transition_selected_label": emergence.get("transition_selected_label"),
                "emergence_transition_two_pi_selected_by_primary": emergence.get(
                    "transition_two_pi_selected_by_primary"
                ),
            },
        ),
        "L4_support_visible_bw_covariance": _stage(
            support_visible_bw_covariance,
            "held-out support-visible H3 response clears signal, geometry, aggregate wrong-scale, and material feature gates",
            details={
                "signal_gate": h3_gates.get("signal_gate"),
                "geometry_gate": h3_gates.get("geometry_gate"),
                "aggregate_wrong_scale_gate": h3_gates.get("aggregate_wrong_scale_gate"),
                "material_feature_gate": h3_gates.get("material_feature_gate"),
                "material_wrong_scale_win_fraction": h3_gates.get("material_wrong_scale_win_fraction"),
                "material_wrong_scale_gate_metric": h3_gates.get("material_wrong_scale_gate_metric"),
                "material_wrong_scale_gate_value": h3_gates.get("material_wrong_scale_gate_value"),
                "material_wrong_scale_advantage_energy_fraction": h3_gates.get(
                    "material_wrong_scale_advantage_energy_fraction"
                ),
                "wrong_scale_material_margin": wrong_scale.get("material_margin"),
            },
        ),
        "L5_ordered_cut_pair_rigidity": _stage(
            ordered_cut_pair_rigidity,
            "ordered boundary cap-pair/collar rigidity is witnessed by the finite conformal H3/Lorentz chart verifier",
            details={
                "emergence_receipt": emergence.get("ordered_cut_pair_rigidity_receipt"),
                "chart_receipt": chart.get("ordered_cut_pair_rigidity_receipt"),
                "paper_chart_receipt": paper_chart.get("ordered_cut_pair_rigidity_receipt"),
                "paper_chart_receipt_inferred_from_cap_lorentz_verifier": bool(
                    paper_chart.get("PAPER_THEOREM_3D_BULK_CHART_RECEIPT", False)
                    and paper_chart.get("bw_2pi_cap_flow_receipt", False)
                    and paper_chart.get("lorentz_algebra_receipt", False)
                ),
                "conformal_chart_receipt_inferred_from_cap_normals": bool(
                    chart.get("conformal_h3_spatial_chart_receipt", False)
                    and chart.get("lorentz_algebra_receipt", False)
                    and (chart.get("cap_normal_report") or {}).get("unit_normal_receipt", False)
                ),
            },
        ),
        "L6_lorentz_algebra_closure": _stage(
            lorentz_algebra_closure,
            "the conformal chart closes the finite Lorentz algebra diagnostics",
        ),
        "L7_refinement_naturality": _stage(
            refinement_naturality,
            "chart-blind neutral quotient receipts survive regulator refinement and observer resampling",
            details={
                "strict_neutral_bulk_refinement_receipt": refinement.get(
                    "strict_neutral_bulk_refinement_receipt"
                ),
                "nested_strict_neutral_bulk_refinement_receipt": refinement_summary.get(
                    "strict_neutral_bulk_refinement_receipt"
                ),
                "candidate_dimension_stable": refinement_summary.get("candidate_dimension_stable"),
                "multi_scale": refinement_summary.get("multi_scale"),
                "proof_blockers": refinement_summary.get("proof_blockers", []),
            },
        ),
        "B1_h3_response_candidate": _stage(
            h3_candidate,
            "observer/cap response is a strict H3 candidate after material wrong-scale audit",
        ),
        "B2_observer_object_population": _stage(
            object_population,
            "observer-visible records populate the H3 chart through the issue #310 cap-response localization receipt or the older object-population gate",
            missing=object_blockers,
            details={
                "object_chart_population_receipt": object_chart.get("OBJECT_BULK_POPULATION_RECEIPT")
                or object_chart.get("observer_chart_bulk_population_receipt"),
                "issue_310_modular_response_h3_localization_receipt": issue310_h3loc.get(
                    MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT
                ),
                "issue_310_terminal_status": issue310_h3loc.get("terminal_status"),
                "issue_310_blockers": issue310_h3loc.get("blockers", []),
            },
        ),
        "B3_observer_facing_3p1d_experience": _stage(
            observer_3p1d,
            "observer-local modular time, H3 chart, and H3 response all pass",
            missing=observer.get("blockers", []),
        ),
        "B4_strict_neutral_bulk_audit": _stage(
            neutral_bulk,
            "chart-blind neutral quotient records independently reconstruct a third-person 3D bulk",
            missing=refinement.get("blockers") or neutral_audit.get("blockers"),
            details={
                "legacy_bulk_reconstruction_established": neutral.get("bulk_3d_established"),
                "neutral_3d_audit_strict_bulk": neutral_audit.get("strict_neutral_bulk"),
                "neutral_frontier_strict_bulk": refinement.get("strict_neutral_bulk"),
                "strict_neutral_bulk_ready": refinement.get("strict_neutral_bulk_ready")
                or neutral_audit.get("strict_neutral_bulk_ready"),
            },
        ),
        "E0_einstein_branch_entry_umbrella": _stage(
            einstein_branch_entry,
            (
                "E0 OPH5 Einstein bridge manifest is theorem-discharged and all required run "
                "sidecar receipts are theorem-tagged"
                if use_einstein_bridge_manifest
                else "legacy issue #503 umbrella is closed and all Einstein branch-entry child gates E1-E6 pass"
            ),
            missing=(
                list(einstein_bridge.get("einstein_branch_entry_blockers") or einstein_bridge.get("blockers") or [])
                if use_einstein_bridge_manifest
                else [
                    name
                    for name, passed in {
                        "issue_503_closed_receipt": einstein_issue_503,
                        **einstein_child_gates,
                    }.items()
                    if not passed
                ]
            ),
            details={
                "manifest_written": use_einstein_bridge_manifest,
                "manifest_receipt": bool(
                    einstein_bridge.get(OPH_EINSTEIN_BRIDGE_MANIFEST_RECEIPT, False)
                ),
                "theorem_e0_dependency_discharge_receipt": einstein_bridge_dependency_discharge,
                "einstein_bridge_run_receipts_receipt": einstein_bridge_run_receipts,
                "claim_tier": einstein_bridge.get("claim_tier"),
                "legacy_issue": 503,
                "legacy_issue_url": "https://github.com/FloatingPragma/observer-patch-holography/issues/503",
                "legacy_issue_503_status": einstein.get("issue_503_status", "open_or_unreported"),
                "legacy_source_report_written": bool(einstein),
                "child_gate_issue_map": {
                    "E1_null_generator_stress_charge": "#19",
                    "E2_fixed_cap_entropy_stationarity": "#20",
                    "E3_small_ball_area_bridge": "#21",
                    "E4_all_timelike_tensor_upgrade": "chi_nu_test/EinsteinBranch Lean core plus #21",
                    "E5_lambda_constancy_conservation": "chi_nu_test/LambdaConstancy Lean core plus #22",
                    "E6_newton_coupling_forbidden_input_audit": "#345",
                },
            },
        ),
        "E1_null_generator_stress_charge": _stage(
            einstein_e1_null_stress,
            "OPH null generator is identified with the local null-stress charge inside the branch",
        ),
        "E2_fixed_cap_entropy_stationarity": _stage(
            einstein_e2_entropy,
            "fixed-cap generalized-entropy stationarity is derived for the realized MaxEnt family",
        ),
        "E3_small_ball_area_bridge": _stage(
            einstein_e3_small_ball,
            "small-ball entropy/area bridge and fixed-volume area response are internally certified",
        ),
        "E4_all_timelike_tensor_upgrade": _stage(
            einstein_e4_tensor,
            "all-observer/all-timelike rest-frame coverage upgrades scalar data to the tensor equation",
        ),
        "E5_lambda_constancy_conservation": _stage(
            einstein_e5_lambda,
            "Bianchi identity, stress conservation, metric compatibility, and connectedness fix one Lambda",
        ),
        "E6_newton_coupling_forbidden_input_audit": _stage(
            einstein_e6_newton,
            "Einstein/Newton coupling is audited without measured G, Planck area, Lambda, or gravity-calibrated endpoints",
        ),
    }
    finite_contract = all(stages[name]["passed"] for name in (
        "C0_finite_consensus_theorem",
        "L1_observer_record_algebra",
        "L2_endogenous_modular_generator",
        "L3_kms_modular_clock_fit",
        "L4_support_visible_bw_covariance",
        "L5_ordered_cut_pair_rigidity",
        "L6_lorentz_algebra_closure",
    ))
    paper_geometric_branch_contract_stage_names = (
        "C0_finite_consensus_theorem",
        "L1_observer_record_algebra",
        "L3b_declared_geometric_kms_branch",
        "L4_support_visible_bw_covariance",
        "L5_ordered_cut_pair_rigidity",
        "L6_lorentz_algebra_closure",
    )
    paper_geometric_branch_contract = all(
        stages[name]["passed"] for name in paper_geometric_branch_contract_stage_names
    )
    observer_spacetime = bool(
        finite_contract
        and stages["B1_h3_response_candidate"]["passed"]
        and stages["B3_observer_facing_3p1d_experience"]["passed"]
    )
    populated_h3 = bool(
        observer_spacetime
        and stages["B2_observer_object_population"]["passed"]
    )
    observer_facing_consensus_bulk = populated_h3
    paper_geometric_branch_observer_spacetime = bool(
        paper_geometric_branch_contract
        and stages["B1_h3_response_candidate"]["passed"]
        and stages["B3_observer_facing_3p1d_experience"]["passed"]
    )
    paper_geometric_branch_populated_h3 = bool(
        paper_geometric_branch_observer_spacetime
        and stages["B2_observer_object_population"]["passed"]
    )
    paper_geometric_branch_consensus_bulk = paper_geometric_branch_populated_h3
    chart_blind_strict_neutral = bool(
        stages["L7_refinement_naturality"]["passed"]
        and stages["B4_strict_neutral_bulk_audit"]["passed"]
    )
    finite_contract_stage_names = (
        "C0_finite_consensus_theorem",
        "L1_observer_record_algebra",
        "L2_endogenous_modular_generator",
        "L3_kms_modular_clock_fit",
        "L4_support_visible_bw_covariance",
        "L5_ordered_cut_pair_rigidity",
        "L6_lorentz_algebra_closure",
    )
    observer_consensus_stage_names = (
        *finite_contract_stage_names,
        "B1_h3_response_candidate",
        "B2_observer_object_population",
        "B3_observer_facing_3p1d_experience",
    )
    paper_geometric_branch_consensus_stage_names = (
        *paper_geometric_branch_contract_stage_names,
        "B1_h3_response_candidate",
        "B2_observer_object_population",
        "B3_observer_facing_3p1d_experience",
    )
    chart_blind_neutral_stage_names = (
        "L7_refinement_naturality",
        "B4_strict_neutral_bulk_audit",
    )
    einstein_branch_stage_names = (
        "E0_einstein_branch_entry_umbrella",
        "E1_null_generator_stress_charge",
        "E2_fixed_cap_entropy_stationarity",
        "E3_small_ball_area_bridge",
        "E4_all_timelike_tensor_upgrade",
        "E5_lambda_constancy_conservation",
        "E6_newton_coupling_forbidden_input_audit",
    )
    blockers = [name for name in observer_consensus_stage_names if not stages[name]["passed"]]
    paper_geometric_branch_blockers = [
        name for name in paper_geometric_branch_consensus_stage_names if not stages[name]["passed"]
    ]
    chart_blind_neutral_blockers = [
        name for name in chart_blind_neutral_stage_names if not stages[name]["passed"]
    ]
    einstein_branch_blockers = [
        name for name in einstein_branch_stage_names if not stages[name]["passed"]
    ]
    all_stage_blockers = [name for name, row in stages.items() if not row["passed"]]
    report = {
        "mode": "finite_oph_theorem_contract_audit_v1",
        "run_path": str(root),
        "stages": stages,
        OPH_LORENTZ_THEOREM_FINITE_CONTRACT_RECEIPT: finite_contract,
        "finite_lorentz_theorem_contract_receipt": finite_contract,
        "paper_faithful_observer_spacetime_emergence_receipt": observer_spacetime,
        "paper_faithful_populated_h3_observer_experience_receipt": populated_h3,
        "observer_facing_consensus_3d_bulk_emergence_receipt": observer_facing_consensus_bulk,
        "paper_faithful_consensus_bulk_emergence_receipt": observer_facing_consensus_bulk,
        "paper_geometric_branch_lorentz_contract_receipt": paper_geometric_branch_contract,
        "paper_geometric_branch_observer_spacetime_emergence_receipt": (
            paper_geometric_branch_observer_spacetime
        ),
        "paper_geometric_branch_populated_h3_observer_experience_receipt": (
            paper_geometric_branch_populated_h3
        ),
        "paper_geometric_branch_consensus_bulk_emergence_receipt": (
            paper_geometric_branch_consensus_bulk
        ),
        "simulation_matches_observer_facing_oph_spacetime_bulk_prediction_receipt": (
            paper_geometric_branch_consensus_bulk
        ),
        "simulation_matches_full_oph_spacetime_bulk_prediction_receipt": observer_facing_consensus_bulk,
        "chart_blind_strict_neutral_quotient_bulk_receipt": chart_blind_strict_neutral,
        "strict_neutral_bulk_contract_receipt": chart_blind_strict_neutral,
        OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_RECEIPT: einstein_branch_entry,
        EINSTEIN_BRANCH_ENTRY_RECEIPT: einstein_branch_entry,
        "einstein_branch_entry_contract_receipt": einstein_branch_entry,
        "issue_503_einstein_branch_entry_status": einstein.get(
            "issue_503_status", "open_or_unreported"
        ),
        "einstein_bridge_manifest_written": use_einstein_bridge_manifest,
        "einstein_bridge_manifest": {
            "receipt": bool(einstein_bridge.get(OPH_EINSTEIN_BRIDGE_MANIFEST_RECEIPT, False)),
            "dependency_discharge_receipt": einstein_bridge_dependency_discharge,
            "run_receipts_receipt": einstein_bridge_run_receipts,
            "claim_tier": einstein_bridge.get("claim_tier"),
            "blockers": list(einstein_bridge.get("blockers") or []),
            "required_receipt_files": list(einstein_bridge.get("requiredReceiptFiles") or []),
            "provenance_tags": dict(einstein_bridge.get("provenanceTags") or {}),
        },
        "issue_308_bw_certificate": {
            "receipt": bool(issue308_bw.get(ISSUE_308_BW_CERTIFICATE_RECEIPT, False)),
            "tier": issue308_bw.get("tier", "BW0"),
            "report_written": bool(issue308_bw),
            "nonclaims": dict(issue308_bw.get("nonclaims") or {}),
            "primary_blockers": [
                name
                for name, row in (issue308_bw.get("clauses") or {}).items()
                if isinstance(row, dict) and not row.get("passed", False)
            ][:6],
        },
        ISSUE_308_BW_CERTIFICATE_RECEIPT: bool(issue308_bw.get(ISSUE_308_BW_CERTIFICATE_RECEIPT, False)),
        "issue_308_finite_cap_bw_certificate_receipt": bool(
            issue308_bw.get("issue_308_finite_cap_bw_certificate_receipt", False)
        ),
        "issue_309_cap_normal_h3_chart": {
            "receipt": bool(issue309_h3.get(CAP_NORMAL_H3_CHART_RECEIPT, False)),
            "terminal_status": issue309_h3.get("terminal_status"),
            "report_written": bool(issue309_h3),
            "mandatory_nonclaims": dict(issue309_h3.get("mandatory_nonclaims") or {}),
            "primary_blockers": list(issue309_h3.get("blockers") or [])[:6],
        },
        CAP_NORMAL_H3_CHART_RECEIPT: bool(issue309_h3.get(CAP_NORMAL_H3_CHART_RECEIPT, False)),
        "cap_normal_h3_chart_receipt": bool(issue309_h3.get("cap_normal_h3_chart_receipt", False)),
        "issue_310_modular_response_h3_localization": {
            "receipt": bool(issue310_h3loc.get(MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT, False)),
            "terminal_status": issue310_h3loc.get("terminal_status"),
            "report_written": bool(issue310_h3loc),
            "component_receipts": dict(issue310_h3loc.get("component_receipts") or {}),
            "primary_blockers": list(issue310_h3loc.get("blockers") or [])[:6],
        },
        MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT: bool(
            issue310_h3loc.get(MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT, False)
        ),
        "h3_modular_response_localization_receipt": bool(
            issue310_h3loc.get("h3_modular_response_localization_receipt", False)
        ),
        "blockers": blockers,
        "primary_blockers": blockers[:6],
        "paper_geometric_branch_blockers": paper_geometric_branch_blockers,
        "paper_geometric_branch_primary_blockers": paper_geometric_branch_blockers[:6],
        "chart_blind_strict_neutral_blockers": chart_blind_neutral_blockers,
        "strict_neutral_blockers": chart_blind_neutral_blockers,
        "einstein_branch_entry_blockers": einstein_branch_blockers,
        "einstein_branch_entry_primary_blockers": einstein_branch_blockers[:6],
        "einstein_branch_entry_child_gates": einstein_child_gates,
        "all_stage_blockers": all_stage_blockers,
        "observer_like_self_reading_system_receipt": bool(observer_rows["patch_observer_count"] > 0),
        "observer_row_summary": observer_rows,
        "claim_boundary": (
            "Paper-faithful finite OPH spacetime/bulk emergence audit. This is stricter than branch "
            "replay: OPH tech must instantiate observer-like self-reading systems with local state, "
            "boundaries, readback, records, feedback/repair moves, and public evidence bundles. The "
            "legacy finite Lorentz/modular L-lane stops at support-visible BW covariance, cut-pair "
            "rigidity diagnostics, Lorentz algebra closure, and an endogenous finite KMS clock fit. "
            "The issue #308 BW3 theorem receipt is stricter: it requires primitive cap-normal, frame, "
            "support-order, held-out cross-ratio, mixed-GNS, geometric KMS, wrong-scale, and envelope "
            "fields. Separately, the paper-geometric branch receipt uses the declared theorem chart "
            "instead of claiming the finite run rediscovered that normalization endogenously. The "
            "issue #309 chart receipt proves the cap-normal H3 chart; the issue #310 localization "
            "receipt is the stricter route for record-populated observer-facing H3, requiring "
            "record-conditioned responses, a compact domain, alpha>0, bounded error, and positive "
            "Delta_loc for unique finite points. The observer spacetime receipt adds the H3 response "
            "and observer-local 3+1D experience. The observer-facing consensus 3D bulk receipt adds "
            "shared record/object population in that H3 chart. The chart-blind strict neutral quotient audit is a separate "
            "stronger certificate and is reported without being required for the observer-facing 3D "
            "theorem receipt. The Einstein branch-entry receipt is separate again: the E0 paper theorem "
            "discharges the OPH5 bridge dependencies, but no run promotes production gravity unless the "
            "manifest sidecar receipts for stress, entropy, bounded interval, small ball, remainder, "
            "timelike coverage, stress closure, Lambda, Newton audit, and residual checks are explicitly closed."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=BRANCH_INSTANTIATION_SANITY,
        receipt="FINITE_OPH_THEOREM_CONTRACT_AUDIT",
        physical_claim=False,
        observable_id="finite_observer_spacetime_bulk_contract",
        fit_objective="paper_faithful_theorem_hypothesis_receipts",
    )


def write_finite_oph_theorem_contract_report(run_dir: Path, out: Path | None = None) -> dict[str, Any]:
    report = finite_oph_theorem_contract_report(run_dir)
    out_path = Path(out) if out is not None else Path(run_dir) / "finite_oph_theorem_contract_report.json"
    if out_path.suffix.lower() != ".json":
        out_path = out_path / "finite_oph_theorem_contract_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    out_path.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _observer_row_summary(path: Path) -> dict[str, Any]:
    summary = {
        "observer_view_count": 0,
        "patch_observer_count": 0,
        "cap_observer_count": 0,
        "rows_with_support_nodes": 0,
        "rows_with_readout_hash": 0,
        "rows_with_transition_histories": 0,
        "rows_with_modular_time": 0,
    }
    if not path.exists():
        return summary
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            summary["observer_view_count"] += 1
            view_type = str(row.get("view_type", ""))
            if view_type == "patch_observer":
                summary["patch_observer_count"] += 1
            elif view_type == "cap_observer":
                summary["cap_observer_count"] += 1
            if isinstance(row.get("support_nodes"), list) and row["support_nodes"]:
                summary["rows_with_support_nodes"] += 1
            if row.get("visible_readout_hash"):
                summary["rows_with_readout_hash"] += 1
            if row.get("transition_history_descriptor") or row.get("transition_history_histograms"):
                summary["rows_with_transition_histories"] += 1
            if row.get("observer_relative_times"):
                summary["rows_with_modular_time"] += 1
    return summary


def _stage(
    passed: bool,
    meaning: str,
    *,
    missing: Any = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {"passed": bool(passed), "meaning": str(meaning)}
    if missing:
        row["missing_or_blocking_evidence"] = list(missing) if isinstance(missing, list) else missing
    if details:
        row["details"] = details
    return row


def _truthy_any(data: dict[str, Any], *keys: str) -> bool:
    return any(bool(data.get(key, False)) for key in keys)


def _declared_geometric_kms_branch_receipt(
    *,
    paper_chart: dict[str, Any],
    emergence: dict[str, Any],
    chart: dict[str, Any],
) -> bool:
    chart_receipt = _truthy_any(
        paper_chart,
        "PAPER_THEOREM_3D_BULK_CHART_RECEIPT",
        "paper_theorem_3d_bulk_chart_receipt",
    ) or _truthy_any(chart, "conformal_h3_spatial_chart_receipt", "CHART_LEVEL_CONFORMAL_LORENTZ_RECEIPT")
    branch_receipt = _truthy_any(
        paper_chart,
        "declared_bw_2pi_cap_flow_receipt",
        "bw_2pi_cap_flow_receipt",
        "BW_KMS_BRANCH_REPLAY_RECEIPT",
        "BW_KMS_DIRECT_2PI_RECEIPT",
    ) or _truthy_any(
        emergence,
        "BW_KMS_BRANCH_REPLAY_RECEIPT",
        "BW_KMS_DIRECT_2PI_RECEIPT",
        "support_visible_lorentz_3p1_kinematics_receipt",
    )
    source_values = {
        str(paper_chart.get("bw_2pi_cap_flow_source", "")),
        str(emergence.get("transition_primary_source", "")),
    }
    declared_or_kms_source = bool(
        {"declared_kms_collar_transport_response", "kms_collar_transport_response"}
        & source_values
    ) or bool(paper_chart.get("declared_bw_2pi_cap_flow_receipt", False))
    selected_2pi = bool(
        paper_chart.get("bw_2pi_cap_flow_receipt", False)
        or paper_chart.get("declared_bw_2pi_cap_flow_receipt", False)
        or emergence.get("transition_two_pi_selected_by_primary", False)
        or str(emergence.get("transition_selected_label", "")) == "2pi"
    )
    return bool(chart_receipt and branch_receipt and declared_or_kms_source and selected_2pi)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _read_issue308_bw_certificate(root: Path) -> dict[str, Any]:
    for name in (
        "issue_308_bw_certificate_report.json",
        "bw_issue308_certificate_report.json",
        "finite_cap_bw_certificate_report.json",
    ):
        report = _read_json(root / name)
        if report:
            return report
    for name in ("bw_rec_308.json", "BWRec_r.json", "issue_308_primitive_fields.json"):
        payload = _read_json(root / name)
        if payload:
            return issue308_bw_certificate_report(payload)
    return {}


def _read_issue309_cap_normal_h3_chart(root: Path) -> dict[str, Any]:
    for name in (
        "cap_normal_h3_chart_report.json",
        "issue_309_cap_normal_h3_chart_report.json",
        "cap_normal_h3_chart_receipt.json",
    ):
        report = _read_json(root / name)
        if report:
            return report
    for name in (
        "cap_normal_h3_chart_source.json",
        "issue_309_cap_normal_h3_chart_source.json",
        "cap_normal_h3_primitive_fields.json",
    ):
        payload = _read_json(root / name)
        if payload:
            return cap_normal_h3_chart_report(payload)
    return {}


def _read_issue310_modular_response_h3_localization(root: Path) -> dict[str, Any]:
    for name in (
        "modular_response_h3_localization_report.json",
        "issue_310_modular_response_h3_localization_report.json",
        "h3_localization_receipt.json",
    ):
        report = _read_json(root / name)
        if report:
            return report
    for name in (
        "modular_response_h3_localization_source.json",
        "issue_310_modular_response_h3_localization_source.json",
        "h3_localization_primitive_fields.json",
    ):
        payload = _read_json(root / name)
        if payload:
            return modular_response_h3_localization_report(payload)
    return {}


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Finite OPH Theorem Contract Audit",
        "",
        f"- finite Lorentz contract: `{str(report['finite_lorentz_theorem_contract_receipt']).lower()}`",
        "- observer spacetime emergence: "
        f"`{str(report['paper_faithful_observer_spacetime_emergence_receipt']).lower()}`",
        "- populated H3 observer experience: "
        f"`{str(report['paper_faithful_populated_h3_observer_experience_receipt']).lower()}`",
        "- observer-facing consensus bulk emergence: "
        f"`{str(report['paper_faithful_consensus_bulk_emergence_receipt']).lower()}`",
        "- paper-geometric branch Lorentz contract: "
        f"`{str(report['paper_geometric_branch_lorentz_contract_receipt']).lower()}`",
        "- paper-geometric branch observer spacetime: "
        f"`{str(report['paper_geometric_branch_observer_spacetime_emergence_receipt']).lower()}`",
        "- paper-geometric branch consensus bulk: "
        f"`{str(report['paper_geometric_branch_consensus_bulk_emergence_receipt']).lower()}`",
        "- chart-blind strict neutral quotient bulk: "
        f"`{str(report['chart_blind_strict_neutral_quotient_bulk_receipt']).lower()}`",
        "- Einstein branch-entry contract: "
        f"`{str(report['einstein_branch_entry_contract_receipt']).lower()}`",
        "",
        "## Stages",
        "",
    ]
    for name, row in report["stages"].items():
        lines.append(f"- `{name}`: `{str(row['passed']).lower()}` - {row['meaning']}")
    lines.extend(["", "## Blockers", ""])
    for blocker in report["blockers"]:
        lines.append(f"- `{blocker}`")
    if report.get("paper_geometric_branch_blockers"):
        lines.extend(["", "## Paper-Geometric Branch Blockers", ""])
        for blocker in report["paper_geometric_branch_blockers"]:
            lines.append(f"- `{blocker}`")
    if report.get("einstein_branch_entry_blockers"):
        lines.extend(["", "## Einstein Branch-Entry Blockers", ""])
        for blocker in report["einstein_branch_entry_blockers"]:
            lines.append(f"- `{blocker}`")
    lines.extend(["", "## Claim Boundary", "", report["claim_boundary"], ""])
    return "\n".join(lines)
