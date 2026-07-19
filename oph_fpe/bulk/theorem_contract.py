from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

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
    FINITE_CONSENSUS_THEOREM_RECEIPT,
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
from oph_fpe.bulk.lorentz_algebra import lorentz_algebra_report
from oph_fpe.bulk.modular_response_h3_localization import modular_response_h3_localization_report
from oph_fpe.evidence.hashes import CANONICAL_HASH_SCHEMA
from oph_fpe.gauge.covariant_overlap import (
    GAUGE_COVARIANT_OVERLAP_SCHEMA,
    GAUGE_QUOTIENT_CANONICALIZER,
)
from oph_fpe.simulation_assumptions import (
    manifest_assumptions_pass,
    revalidate_simulation_assumption_manifest,
    simulation_assumption_manifest,
)


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
    # A persisted manifest is a cache/display artifact, not producer evidence.
    # Always replay the theorem-tagged sidecars so neither an old aggregate
    # report nor a hand-edited manifest can promote branch entry.
    einstein_bridge = einstein_bridge_manifest_report(root)
    issue308_bw = _read_issue308_bw_certificate(root)
    issue309_h3 = _read_issue309_cap_normal_h3_chart(root)
    issue310_h3loc = _read_issue310_modular_response_h3_localization(root)
    assumption_manifest, assumption_source = _read_simulation_assumption_manifest(root)
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
    issue308_receipt = _literal_true(issue308_bw.get(ISSUE_308_BW_CERTIFICATE_RECEIPT))
    issue309_receipt = _literal_true(issue309_h3.get(CAP_NORMAL_H3_CHART_RECEIPT))
    issue310_receipt = _literal_true(issue310_h3loc.get(MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT))

    finite_consensus, finite_consensus_validation = _validated_finite_consensus_replay(
        root,
        theorem_core,
    )
    record_algebra = bool(
        observer_rows["patch_observer_count"] > 0
        and observer_rows["rows_with_support_nodes"] > 0
        and observer_rows["rows_with_readout_hash"] > 0
        and observer_rows["rows_with_transition_histories"] > 0
    )
    state_bw = _read_json(root / "bw_state_derived_report.json")
    inferred_clock = state_bw.get("inferred_modular_clock_fit") or {}
    endogenous_generator = _validated_endogenous_modular_generator(state_bw)
    kms_clock_fit = _validated_kms_clock_fit(state_bw)
    declared_geometric_kms_branch = _declared_geometric_kms_branch_receipt(
        paper_chart=paper_chart,
        emergence=emergence,
        chart=chart,
    )
    support_visible_bw_covariance = bool(
        issue308_receipt
        and _literal_true(h3_gates.get("signal_gate"))
        and _literal_true(h3_gates.get("geometry_gate"))
        and _literal_true(h3_gates.get("aggregate_wrong_scale_gate"))
        and _literal_true(h3_gates.get("material_feature_gate"))
    )
    ordered_cut_pair_rigidity = bool(
        issue308_receipt
        and issue309_receipt
        and _issue308_clause_passed(issue308_bw, "C2_bw_frame")
        and _issue308_clause_passed(issue308_bw, "C3_prime_support_visible_cap_net")
        and _issue308_clause_passed(issue308_bw, "C6_geometric_rigidity")
    )
    local_lorentz_algebra = lorentz_algebra_report()
    lorentz_algebra_closure = bool(
        issue309_receipt
        and _literal_true(local_lorentz_algebra.get("lorentz_algebra_receipt"))
    )
    refinement_naturality = _validated_refinement_naturality(
        refinement,
        refinement_summary,
        neutral_audit,
    )
    h3_candidate = _literal_true(h3.get("H3_RESPONSE_CANDIDATE_RECEIPT"))
    object_population = bool(
        _literal_true(object_chart.get("OBJECT_BULK_POPULATION_RECEIPT"))
        or _literal_true(object_chart.get("observer_chart_bulk_population_receipt"))
        or _literal_true(issue310_h3loc.get(MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT))
    )
    observer_3p1d = bool(
        _literal_true(observer.get("OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT"))
        or _literal_true(observer.get("observer_facing_3p1d_h3_experience_receipt"))
    )
    neutral_bulk = bool(
        _literal_true(neutral.get("bulk_3d_established"))
        or _literal_true(neutral_audit.get("strict_neutral_bulk"))
        or _literal_true(refinement.get("strict_neutral_bulk"))
        or _literal_true(emergence.get("strict_blind_observer_bulk_receipt"))
        or _literal_true(emergence.get("neutral_bulk_3d_established"))
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
    if isinstance(manifest_child_gates, dict):
        einstein_child_gates = {name: _literal_true(value) for name, value in manifest_child_gates.items()}
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
    einstein_branch_entry = _truthy_any(
        einstein_bridge,
        OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_RECEIPT,
        EINSTEIN_BRANCH_ENTRY_RECEIPT,
        "einstein_branch_entry_contract_receipt",
        "einstein_branch_entry_receipt",
    )

    stages = {
        "C0_finite_consensus_theorem": _stage(
            finite_consensus,
            "finite overlap repair is replayed as a theorem-phase consensus certificate",
            missing=finite_consensus_validation.get("blockers", []),
            details=finite_consensus_validation,
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
                "source_mode": state_bw.get("mode"),
                "state_mode": state_bw.get("state_mode"),
                "row_count": state_bw.get("row_count"),
                "endogenous_modular_generator": state_bw.get("endogenous_modular_generator"),
                "endogenous_generator_non_degenerate": state_bw.get(
                    "endogenous_generator_non_degenerate"
                ),
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
            "the issue #309 chart is paired with a locally recomputed Lorentz algebra closure",
            details={
                "locally_recomputed": True,
                "max_commutator_error": local_lorentz_algebra.get("max_commutator_error"),
                "max_null_cone_preservation_error": local_lorentz_algebra.get(
                    "max_null_cone_preservation_error"
                ),
            },
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
        "T308_finite_cap_bw_certificate": _stage(
            issue308_receipt,
            "issue #308 BW3 is recomputed from primitive finite cap-net certificate fields",
            missing=[
                name
                for name, row in (issue308_bw.get("clauses") or {}).items()
                if isinstance(row, dict) and not row.get("passed", False)
            ],
            details={
                "tier": issue308_bw.get("tier", "BW0"),
                "source_report_written": bool(issue308_bw),
            },
        ),
        "T309_cap_normal_h3_chart": _stage(
            issue309_receipt,
            "issue #309 cap-normal H3 chart is recomputed from unit centers, Lorentz maps, and H3 points",
            missing=issue309_h3.get("blockers", []),
            details={
                "terminal_status": issue309_h3.get("terminal_status"),
                "source_report_written": bool(issue309_h3),
            },
        ),
        "T310_modular_response_h3_localization": _stage(
            issue310_receipt,
            "issue #310 record-conditioned H3 localization passes frame, domain, net, interval, and uniqueness checks",
            missing=issue310_h3loc.get("blockers", []),
            details={
                "terminal_status": issue310_h3loc.get("terminal_status"),
                "source_report_written": bool(issue310_h3loc),
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
            "E0 OPH5 Einstein bridge dependencies and independently replayed, theorem-tagged run sidecars",
            missing=list(
                einstein_bridge.get("einstein_branch_entry_blockers")
                or einstein_bridge.get("blockers")
                or []
            ),
            details={
                "manifest_written": use_einstein_bridge_manifest,
                "manifest_recomputed_from_sidecars": True,
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
                "legacy_aggregate_branch_entry_ignored": legacy_einstein_branch_entry,
                "persisted_manifest_branch_entry_ignored": _truthy_any(
                    explicit_einstein_bridge_manifest,
                    OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_RECEIPT,
                    EINSTEIN_BRANCH_ENTRY_RECEIPT,
                    "einstein_branch_entry_contract_receipt",
                    "einstein_branch_entry_receipt",
                ),
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
    finite_contract_stage_names = (
        "C0_finite_consensus_theorem",
        "L1_observer_record_algebra",
        "L2_endogenous_modular_generator",
        "L3_kms_modular_clock_fit",
        "L4_support_visible_bw_covariance",
        "L5_ordered_cut_pair_rigidity",
        "L6_lorentz_algebra_closure",
        "L7_refinement_naturality",
        "T308_finite_cap_bw_certificate",
    )
    finite_contract = all(stages[name]["passed"] for name in finite_contract_stage_names)
    paper_geometric_branch_contract_stage_names = (
        "C0_finite_consensus_theorem",
        "L1_observer_record_algebra",
        "L3b_declared_geometric_kms_branch",
        "L4_support_visible_bw_covariance",
        "L5_ordered_cut_pair_rigidity",
        "L6_lorentz_algebra_closure",
        "L7_refinement_naturality",
        "T308_finite_cap_bw_certificate",
        "T309_cap_normal_h3_chart",
    )
    paper_geometric_branch_contract = all(
        stages[name]["passed"] for name in paper_geometric_branch_contract_stage_names
    )
    observer_spacetime = bool(
        finite_contract
        and stages["T309_cap_normal_h3_chart"]["passed"]
        and stages["B1_h3_response_candidate"]["passed"]
        and stages["B3_observer_facing_3p1d_experience"]["passed"]
    )
    populated_h3 = bool(
        observer_spacetime
        and stages["T310_modular_response_h3_localization"]["passed"]
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
        and stages["T310_modular_response_h3_localization"]["passed"]
        and stages["B2_observer_object_population"]["passed"]
    )
    paper_geometric_branch_consensus_bulk = paper_geometric_branch_populated_h3
    chart_blind_strict_neutral = bool(
        stages["L7_refinement_naturality"]["passed"]
        and stages["B4_strict_neutral_bulk_audit"]["passed"]
    )
    observer_consensus_stage_names = (
        *finite_contract_stage_names,
        "T309_cap_normal_h3_chart",
        "B1_h3_response_candidate",
        "B3_observer_facing_3p1d_experience",
        "T310_modular_response_h3_localization",
        "B2_observer_object_population",
    )
    paper_geometric_branch_consensus_stage_names = (
        *paper_geometric_branch_contract_stage_names,
        "B1_h3_response_candidate",
        "B3_observer_facing_3p1d_experience",
        "T310_modular_response_h3_localization",
        "B2_observer_object_population",
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
    assumed_lorentz_h3 = _all_manifest_assumptions(
        assumption_manifest,
        "bw_2pi_geometric_branch",
        "h3_observer_chart",
    )
    assumed_observer_spacetime = _all_manifest_assumptions(
        assumption_manifest,
        "screen_s2",
        "bw_2pi_geometric_branch",
        "observer_modular_time_interpretation",
        "h3_observer_chart",
        "screen_observer_to_h3_camera_embedding",
        "ds4_open_slicing_background",
        "positive_cosmological_constant",
        "observer_tetrad_visualization",
    )
    assumed_populated_h3 = bool(
        assumed_observer_spacetime
        and _all_manifest_assumptions(
            assumption_manifest,
            "record_population_on_h3",
            "refinement_naturality_visualization",
        )
    )
    assumed_topological_matter = bool(
        assumed_populated_h3
        and _all_manifest_assumptions(assumption_manifest, "topological_defects_render_as_matter")
    )
    assumed_cmb_visualization = bool(
        assumed_observer_spacetime
        and _all_manifest_assumptions(
            assumption_manifest,
            "cmb_screen_to_temperature_transfer_visualization",
            "cmb_tt_reference_shape_visualization",
        )
    )
    assumed_visual_universe = bool(
        _literal_true(assumption_manifest.get("SIMULATION_ASSUMED_VISUAL_UNIVERSE_RECEIPT"))
        and assumed_topological_matter
        and assumed_cmb_visualization
    )
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
        "SIMULATION_ASSUMED_LORENTZ_H3_BRIDGE_RECEIPT": assumed_lorentz_h3,
        "simulation_assumed_lorentz_h3_bridge_receipt": assumed_lorentz_h3,
        "SIMULATION_ASSUMED_OBSERVER_SPACETIME_VISUALIZATION_RECEIPT": assumed_observer_spacetime,
        "simulation_assumed_observer_spacetime_visualization_receipt": assumed_observer_spacetime,
        "SIMULATION_ASSUMED_POPULATED_H3_VISUALIZATION_RECEIPT": assumed_populated_h3,
        "simulation_assumed_populated_h3_visualization_receipt": assumed_populated_h3,
        "SIMULATION_ASSUMED_TOPOLOGICAL_MATTER_VISUALIZATION_RECEIPT": assumed_topological_matter,
        "simulation_assumed_topological_matter_visualization_receipt": assumed_topological_matter,
        "SIMULATION_ASSUMED_CMB_VISUALIZATION_RECEIPT": assumed_cmb_visualization,
        "simulation_assumed_cmb_visualization_receipt": assumed_cmb_visualization,
        "SIMULATION_ASSUMED_VISUAL_UNIVERSE_RECEIPT": assumed_visual_universe,
        "simulation_assumed_visual_universe_receipt": assumed_visual_universe,
        "simulation_assumption_tier": {
            "enabled": _literal_true(assumption_manifest.get("enabled")),
            "profile": assumption_manifest.get("profile", "none"),
            "source": assumption_source,
            "schema": assumption_manifest.get("schema"),
            "assumptions": dict(assumption_manifest.get("assumptions") or {}),
            "missing_assumptions": list(assumption_manifest.get("missing_assumptions") or []),
            "ds4_visualization_parameters": dict(
                assumption_manifest.get("ds4_visualization_parameters") or {}
            ),
            "observer_camera_visualization_parameters": dict(
                assumption_manifest.get("observer_camera_visualization_parameters") or {}
            ),
            "cmb_visualization_parameters": dict(
                assumption_manifest.get("cmb_visualization_parameters") or {}
            ),
            "computed_theorem_receipts_unchanged": True,
        },
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
            "receipt": _literal_true(einstein_bridge.get(OPH_EINSTEIN_BRIDGE_MANIFEST_RECEIPT)),
            "dependency_discharge_receipt": einstein_bridge_dependency_discharge,
            "run_receipts_receipt": einstein_bridge_run_receipts,
            "claim_tier": einstein_bridge.get("claim_tier"),
            "blockers": list(einstein_bridge.get("blockers") or []),
            "required_receipt_files": list(einstein_bridge.get("requiredReceiptFiles") or []),
            "provenance_tags": dict(einstein_bridge.get("provenanceTags") or {}),
        },
        "issue_308_bw_certificate": {
            "receipt": issue308_receipt,
            "tier": issue308_bw.get("tier", "BW0"),
            "report_written": bool(issue308_bw),
            "nonclaims": dict(issue308_bw.get("nonclaims") or {}),
            "primary_blockers": [
                name
                for name, row in (issue308_bw.get("clauses") or {}).items()
                if isinstance(row, dict) and not row.get("passed", False)
            ][:6],
        },
        ISSUE_308_BW_CERTIFICATE_RECEIPT: issue308_receipt,
        "issue_308_finite_cap_bw_certificate_receipt": _literal_true(
            issue308_bw.get("issue_308_finite_cap_bw_certificate_receipt")
        ),
        "issue_309_cap_normal_h3_chart": {
            "receipt": issue309_receipt,
            "terminal_status": issue309_h3.get("terminal_status"),
            "report_written": bool(issue309_h3),
            "mandatory_nonclaims": dict(issue309_h3.get("mandatory_nonclaims") or {}),
            "primary_blockers": list(issue309_h3.get("blockers") or [])[:6],
        },
        CAP_NORMAL_H3_CHART_RECEIPT: issue309_receipt,
        "cap_normal_h3_chart_receipt": _literal_true(issue309_h3.get("cap_normal_h3_chart_receipt")),
        "issue_310_modular_response_h3_localization": {
            "receipt": issue310_receipt,
            "terminal_status": issue310_h3loc.get("terminal_status"),
            "report_written": bool(issue310_h3loc),
            "component_receipts": dict(issue310_h3loc.get("component_receipts") or {}),
            "primary_blockers": list(issue310_h3loc.get("blockers") or [])[:6],
        },
        MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT: issue310_receipt,
        "h3_modular_response_localization_receipt": _literal_true(
            issue310_h3loc.get("h3_modular_response_localization_receipt")
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
            "Computed finite OPH spacetime/bulk emergence audit. This is stricter than branch "
            "replay: OPH tech must instantiate observer-like self-reading systems with local state, "
            "boundaries, readback, records, feedback/repair moves, and public evidence bundles. The "
            "legacy finite Lorentz/modular L-lane stops at support-visible BW covariance, cut-pair "
            "rigidity diagnostics, Lorentz algebra closure, and an endogenous finite KMS clock fit. "
            "The issue #308 BW3 theorem receipt is stricter: it requires primitive cap-normal, frame, "
            "support-order, held-out cross-ratio, mixed-GNS, geometric KMS, wrong-scale, and envelope "
            "fields. The paper-geometric diagnostic may use the declared theorem chart, but its "
            "computed contract still requires issue #308/#309 and L7 refinement receipts. The "
            "separate SIMULATION_ASSUMED_* tier may complete a renderer scene from an explicit "
            "assumption manifest; it never changes these computed receipts. The "
            "issue #309 chart receipt proves the cap-normal H3 chart; the issue #310 localization "
            "receipt is the stricter route for record-populated observer-facing H3, requiring "
            "record-conditioned responses, a compact domain, verified epsilon-net metadata, alpha>0, bounded error, and positive "
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


def finite_consensus_replay_validation_report(run_dir: Path) -> dict[str, Any]:
    """Independently validate C0b from its primitive gauge-coupled source.

    This is the read-only proof-certificate entry point.  It deliberately does
    not read ``finite_oph_theorem_contract_report.json`` and it never writes a
    replacement report: the primitive NPZ, its manifest, the replay sidecar,
    and the theorem-core certificate are revalidated in memory.
    """

    root = Path(run_dir)
    theorem_core = _read_json(root / "theorem_core_receipts.json")
    passed, validation = _validated_finite_consensus_replay(root, theorem_core)
    return {
        "mode": "finite_consensus_primitive_bound_validation_v1",
        "passed": bool(passed),
        "validation": validation,
        "blockers": list(validation.get("blockers", [])),
        "claim_boundary": (
            "C0b is recomputed from the hash-bound primitive gauge-coupled "
            "arrays and exact replay parameters. Persisted theorem-contract "
            "stage booleans are diagnostic only."
        ),
    }


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


def _validated_finite_consensus_replay(
    root: Path,
    theorem_core: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Cross-check the computed C0b certificate against its replay artifact.

    A copied top-level boolean is never sufficient.  The gauge-coupled v3 certificate and
    independently written replay sidecar must agree on provenance hash and all
    theorem-relevant counts, while sampled accepted events must still exhibit
    strict touched descent and non-increasing global mismatch.
    """

    certificate = (
        theorem_core.get("finite_consensus_theorem")
        if isinstance(theorem_core.get("finite_consensus_theorem"), dict)
        else {}
    )
    nested_replay = (
        theorem_core.get("finite_consensus_replay")
        if isinstance(theorem_core.get("finite_consensus_replay"), dict)
        else {}
    )
    replay_path = root / "finite_consensus_replay_report.json"
    replay = _read_json(replay_path)
    evidence = replay.get("evidence") if isinstance(replay.get("evidence"), dict) else {}
    blockers: list[str] = []

    certificate_mode = certificate.get("mode")
    if certificate_mode == "finite_consensus_theorem_certificate_v2_computed_array_replay":
        blockers.append("obsolete_v2_decoupled_consensus_certificate_rejected")
    if certificate_mode != "finite_consensus_theorem_certificate_v3_computed_gauge_quotient_replay":
        blockers.append("computed_v3_gauge_quotient_consensus_certificate_missing")
    for key in (
        FINITE_CONSENSUS_THEOREM_RECEIPT,
        "finite_consensus_theorem_receipt",
        "receipt",
        "computed_replay_artifact_present",
        "computed_from_port_pair_arrays",
        "computed_from_gauge_coupled_arrays",
        "gauge_covariant_mismatch",
    ):
        if not _literal_true(certificate.get(key)):
            blockers.append(f"certificate_{key}_not_literal_true")
    if not (
        _literal_true(theorem_core.get(FINITE_CONSENSUS_THEOREM_RECEIPT))
        and _literal_true(theorem_core.get("finite_consensus_theorem_receipt"))
    ):
        blockers.append("theorem_core_consensus_receipt_mismatch")

    if not replay_path.is_file() or not replay:
        blockers.append("computed_replay_sidecar_missing")
    if replay.get("mode") != "array_port_pair_strict_consensus_replay":
        blockers.append("computed_replay_mode_invalid")
    for key in (
        "enabled",
        "receipt",
        FINITE_CONSENSUS_THEOREM_RECEIPT,
        "finite_consensus_theorem_receipt",
        "computed_from_port_pair_arrays",
        "computed_from_gauge_coupled_arrays",
        "gauge_covariant_mismatch",
    ):
        if not _literal_true(replay.get(key)):
            blockers.append(f"replay_{key}_not_literal_true")
    if nested_replay != replay:
        blockers.append("nested_replay_does_not_match_sidecar")

    source_hash = certificate.get("source_state_sha256")
    if not _sha256_receipt(source_hash):
        blockers.append("certificate_source_state_sha256_invalid")
    if replay.get("source_state_sha256") != source_hash:
        blockers.append("replay_source_state_sha256_mismatch")
    source_quotient_hash = certificate.get("source_quotient_hash")
    if not _sha256_receipt(source_quotient_hash):
        blockers.append("certificate_source_quotient_hash_invalid")
    if replay.get("source_quotient_hash") != source_quotient_hash:
        blockers.append("replay_source_quotient_hash_mismatch")
    if not _sha256_receipt(replay.get("terminal_hash")):
        blockers.append("replay_terminal_hash_invalid")
    if evidence.get("evidence_kind") != "computed_gauge_covariant_quotient_replay_v1":
        blockers.append("computed_replay_evidence_kind_invalid")
    for owner, payload in (("certificate", certificate), ("replay", replay)):
        if payload.get("gauge_quotient_canonicalizer") != GAUGE_QUOTIENT_CANONICALIZER:
            blockers.append(f"{owner}_gauge_quotient_canonicalizer_invalid")
    production_move_contract = (
        replay.get("production_move_contract")
        if isinstance(replay.get("production_move_contract"), dict)
        else {}
    )
    if certificate.get("production_move_contract") != production_move_contract:
        blockers.append("certificate_production_move_contract_mismatch")
    if production_move_contract.get("schema") != "bw_array_production_overlap_move_contract_v1":
        blockers.append("production_move_contract_schema_invalid")
    if production_move_contract.get("mismatch_definition") != GAUGE_COVARIANT_OVERLAP_SCHEMA:
        blockers.append("production_move_contract_mismatch_definition_invalid")
    if not _literal_true(production_move_contract.get("replayed_endpoint_branches")):
        blockers.append("production_endpoint_branches_not_replayed")
    if not _literal_true(production_move_contract.get("exact_production_move_set_replayed")):
        blockers.append("production_move_set_not_exactly_replayed")

    exact_counts = {
        "strict_descent_violation_count": 0,
        "accepted_phi_increase_violation_count": 0,
        "disjoint_commutation_violation_count": 0,
        "local_diamond_violation_count": 0,
        "gauge_covariance_violation_count": 0,
        "production_move_contract_violation_count": 0,
        "endpoint_branch_coverage_incomplete_count": 0,
        "endpoint_branch_confluence_violation_count": 0,
        "repair_completeness_violation_count": 0,
        "unique_terminal_quotient_hash_count": 1,
    }
    for key, expected in exact_counts.items():
        value = evidence.get(key)
        if not _strict_int(value, minimum=0) or value != expected:
            blockers.append(f"replay_{key}_invalid")
        if certificate.get(key) != value:
            blockers.append(f"certificate_{key}_mismatch")

    positive_counts = (
        "theorem_phase_event_count",
        "accepted_theorem_move_count",
        "schedule_replay_count",
        "requested_schedule_replays",
        "gauge_relabeling_check_count",
    )
    for key in positive_counts:
        value = evidence.get(key)
        if not _strict_int(value, minimum=1):
            blockers.append(f"replay_{key}_invalid")
        if certificate.get(key) != value:
            blockers.append(f"certificate_{key}_mismatch")
    if (
        _strict_int(evidence.get("schedule_replay_count"), minimum=1)
        and _strict_int(evidence.get("requested_schedule_replays"), minimum=1)
        and evidence["schedule_replay_count"] < evidence["requested_schedule_replays"]
    ):
        blockers.append("insufficient_schedule_replays")
    if replay.get("initial_phi") != evidence.get("theorem_phase_event_count"):
        blockers.append("replay_initial_phi_mismatch")

    exact_branch_check = (
        replay.get("exact_endpoint_branch_check")
        if isinstance(replay.get("exact_endpoint_branch_check"), dict)
        else {}
    )
    if certificate.get("exact_endpoint_branch_check") != exact_branch_check:
        blockers.append("certificate_exact_endpoint_branch_check_mismatch")
    if exact_branch_check.get("mode") != "exact_endpoint_branch_structural_confluence_v1":
        blockers.append("exact_endpoint_branch_check_mode_invalid")
    if not _literal_true(exact_branch_check.get("coverage_complete")):
        blockers.append("exact_endpoint_branch_coverage_incomplete")
    if not _literal_true(exact_branch_check.get("structurally_confluent")):
        blockers.append("exact_endpoint_branch_nonconfluent")
    exact_witness_count = exact_branch_check.get("structural_nonconfluence_witness_count")
    if (
        not _strict_int(exact_witness_count, minimum=0)
        or exact_witness_count != evidence.get("endpoint_branch_confluence_violation_count")
    ):
        blockers.append("exact_endpoint_branch_witness_count_mismatch")
    exact_unique_count = exact_branch_check.get("unique_terminal_quotient_hash_count")
    if (
        not _strict_int(exact_unique_count, minimum=1)
        or exact_unique_count != evidence.get("unique_terminal_quotient_hash_count")
    ):
        blockers.append("exact_endpoint_branch_terminal_count_mismatch")

    sample_events = replay.get("sample_events") if isinstance(replay.get("sample_events"), list) else []
    if certificate.get("sample_event_count") != len(sample_events):
        blockers.append("certificate_sample_event_count_mismatch")
    for index, row in enumerate(sample_events):
        if not isinstance(row, dict) or not _literal_true(row.get("accepted")):
            blockers.append(f"sample_event_{index}_not_accepted")
            continue
        touched = _finite_number(row.get("delta_touched_phi"))
        global_delta = _finite_number(row.get("delta_global_phi"))
        if touched is None or touched >= 0.0:
            blockers.append(f"sample_event_{index}_not_strictly_descending")
        if global_delta is None or global_delta > 0.0:
            blockers.append(f"sample_event_{index}_global_phi_increased")
    if certificate.get("invalid_evidence") not in ([], ()):
        blockers.append("certificate_invalid_evidence_nonempty")

    independent_replay = _independently_replay_finite_consensus_source(
        root,
        theorem_core=theorem_core,
        certificate=certificate,
        replay=replay,
    )
    blockers.extend(independent_replay["blockers"])

    details = {
        "validation_mode": "primitive_bound_v3_gauge_quotient_independent_replay",
        "certificate_mode": certificate.get("mode"),
        "replay_mode": replay.get("mode"),
        "source_state_sha256": source_hash,
        "source_quotient_hash": source_quotient_hash,
        "sidecar_path": str(replay_path),
        "sample_event_count": len(sample_events),
        "independent_source_replay": independent_replay,
        "blockers": list(dict.fromkeys(blockers)),
    }
    return not details["blockers"], details


def _independently_replay_finite_consensus_source(
    root: Path,
    *,
    theorem_core: dict[str, Any],
    certificate: dict[str, Any],
    replay: dict[str, Any],
) -> dict[str, Any]:
    """Hash-bind primitive arrays and rerun C0b instead of trusting JSON claims."""

    manifest_path = root / "finite_consensus_source_manifest.json"
    manifest = _read_json(manifest_path)
    nested_manifest = theorem_core.get("finite_consensus_source_artifact")
    blockers: list[str] = []
    if not manifest_path.is_file() or not manifest:
        blockers.append("finite_consensus_source_manifest_missing")
    if not isinstance(nested_manifest, dict) or nested_manifest != manifest:
        blockers.append("finite_consensus_nested_source_manifest_mismatch")
    if manifest.get("schema") != "finite_consensus_replay_source_v1":
        blockers.append("finite_consensus_source_manifest_schema_invalid")
    if manifest.get("hash_schema") != CANONICAL_HASH_SCHEMA:
        blockers.append("finite_consensus_source_hash_schema_invalid")
    if manifest.get("state_path") != "finite_consensus_source_state.npz":
        blockers.append("finite_consensus_source_state_path_invalid")
    state_path = root / "finite_consensus_source_state.npz"
    expected_file_hash = manifest.get("state_file_sha256")
    actual_file_hash = _file_sha256(state_path) if state_path.is_file() else None
    if not _sha256_receipt(expected_file_hash) or actual_file_hash != expected_file_hash:
        blockers.append("finite_consensus_source_state_file_hash_mismatch")

    try:
        from oph_fpe.scale import bw_array as replay_kernel
    except Exception:
        replay_kernel = None
        blockers.append("finite_consensus_replay_kernel_import_failed")
    if replay_kernel is not None:
        kernel_hash = _file_sha256(Path(replay_kernel.__file__))
        if (
            not _sha256_receipt(manifest.get("replay_kernel_file_sha256"))
            or manifest.get("replay_kernel_file_sha256") != kernel_hash
        ):
            blockers.append("finite_consensus_replay_kernel_hash_mismatch")

    arrays: dict[str, np.ndarray] = {}
    if state_path.is_file():
        try:
            with np.load(state_path, allow_pickle=False) as payload:
                required = {
                    "initial_port_left",
                    "initial_port_right",
                    "initial_gauge",
                    "edge_left",
                    "edge_right",
                }
                if set(payload.files) != required:
                    blockers.append("finite_consensus_source_array_schema_invalid")
                arrays = {
                    key: np.asarray(payload[key]).copy()
                    for key in required
                    if key in payload.files
                }
        except (OSError, ValueError, KeyError):
            blockers.append("finite_consensus_source_state_unreadable")
    if arrays:
        shapes = {array.shape for array in arrays.values()}
        if len(shapes) != 1 or any(array.ndim != 1 for array in arrays.values()):
            blockers.append("finite_consensus_source_array_shapes_invalid")
        edge_count = manifest.get("edge_count")
        if not _strict_int(edge_count, minimum=1) or next(iter(shapes), ()) != (edge_count,):
            blockers.append("finite_consensus_source_edge_count_invalid")

    group_name = manifest.get("group_name")
    group_order = manifest.get("group_order")
    replay_seed = manifest.get("replay_seed")
    replay_config = manifest.get("replay_config")
    sector_config = manifest.get("production_sector_repair_config")
    if group_name not in {"S3", "ZN"}:
        blockers.append("finite_consensus_source_group_name_invalid")
    if not _strict_int(group_order, minimum=1):
        blockers.append("finite_consensus_source_group_order_invalid")
    if not _strict_int(replay_seed, minimum=0):
        blockers.append("finite_consensus_source_replay_seed_invalid")
    if not isinstance(replay_config, dict) or not _literal_true(replay_config.get("enabled")):
        blockers.append("finite_consensus_source_replay_config_invalid")
    if not isinstance(sector_config, dict):
        blockers.append("finite_consensus_source_sector_config_invalid")

    recomputed_source_hash = None
    recomputed_quotient_hash = None
    recomputed_replay: dict[str, Any] = {}
    if not blockers and replay_kernel is not None:
        recomputed_source_hash = replay_kernel.coupled_state_hash(
            arrays["initial_port_left"],
            arrays["initial_port_right"],
            arrays["initial_gauge"],
            edge_left=arrays["edge_left"],
            edge_right=arrays["edge_right"],
            group_name=str(group_name),
            group_order=int(group_order),
        )
        recomputed_quotient_hash = replay_kernel.gauge_quotient_state_hash(
            arrays["initial_port_left"],
            arrays["initial_port_right"],
            arrays["initial_gauge"],
            edge_left=arrays["edge_left"],
            edge_right=arrays["edge_right"],
            group_name=str(group_name),
            group_order=int(group_order),
        )
        for owner, payload in (
            ("manifest", manifest),
            ("certificate", certificate),
            ("replay", replay),
        ):
            if payload.get("source_state_sha256") != recomputed_source_hash:
                blockers.append(f"finite_consensus_{owner}_source_hash_not_recomputed")
            if payload.get("source_quotient_hash") != recomputed_quotient_hash:
                blockers.append(f"finite_consensus_{owner}_quotient_hash_not_recomputed")
        recomputed_replay = replay_kernel._array_port_pair_consensus_replay_report(
            arrays["initial_port_left"],
            arrays["initial_port_right"],
            arrays["initial_gauge"],
            edge_left=arrays["edge_left"],
            edge_right=arrays["edge_right"],
            group_name=str(group_name),
            group_order=int(group_order),
            config=dict(replay_config),
            production_sector_repair_config=dict(sector_config),
            seed=int(replay_seed),
        )
        if recomputed_replay != replay:
            blockers.append("finite_consensus_independent_replay_mismatch")

    return {
        "mode": "finite_consensus_primitive_bound_independent_replay_v1",
        "manifest_path": str(manifest_path),
        "state_path": str(state_path),
        "state_file_sha256": actual_file_hash,
        "source_state_sha256": recomputed_source_hash,
        "source_quotient_hash": recomputed_quotient_hash,
        "recomputed_receipt": recomputed_replay.get("receipt"),
        "blockers": list(dict.fromkeys(blockers)),
        "passed": not blockers,
    }


def _file_sha256(path: Path) -> str | None:
    if not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _validated_endogenous_modular_generator(state_bw: dict[str, Any]) -> bool:
    row_count = state_bw.get("row_count")
    median = _finite_number(state_bw.get("median"))
    return bool(
        state_bw.get("mode") == "state_derived_modular_probe"
        and _strict_int(row_count, minimum=1)
        and median is not None
        and _literal_true(state_bw.get("endogenous_modular_generator"))
        and _literal_true(state_bw.get("endogenous_generator_non_degenerate"))
        and _literal_true(state_bw.get("ENDOGENOUS_MODULAR_GENERATOR_RECEIPT"))
        and _literal_true(state_bw.get("endogenous_modular_generator_receipt"))
        and not _literal_true(state_bw.get("direct_transition_automorphism"))
        and not _literal_true(state_bw.get("declared_cap_flow_generator"))
        and not _literal_true(state_bw.get("declared_transition_response_density"))
    )


def _validated_kms_clock_fit(state_bw: dict[str, Any]) -> bool:
    clock = (
        state_bw.get("inferred_modular_clock_fit")
        if isinstance(state_bw.get("inferred_modular_clock_fit"), dict)
        else {}
    )
    kappa = _finite_number(clock.get("kappa_hat"))
    interval = clock.get("kappa_95ci")
    ci = (
        [_finite_number(value) for value in interval]
        if isinstance(interval, (list, tuple)) and len(interval) == 2
        else [None, None]
    )
    return bool(
        _validated_endogenous_modular_generator(state_bw)
        and _literal_true(state_bw.get("KMS_GEOMETRIC_CLOCK_FIT_RECEIPT"))
        and _literal_true(state_bw.get("kms_geometric_clock_fit_receipt"))
        and clock.get("mode") == "inferred_modular_clock_fit"
        and _literal_true(clock.get("enabled"))
        and _literal_true(clock.get("receipt"))
        and _literal_true(clock.get("KMS_GEOMETRIC_CLOCK_FIT_RECEIPT"))
        and clock.get("clock_fit_selection_policy")
        == "predeclared_informative_nonstatic_carriers_else_nonstatic_carriers_no_pass_shopping"
        and _strict_int(clock.get("valid_row_count"), minimum=1)
        and _strict_int(clock.get("distinct_time_count"), minimum=3)
        and kappa is not None
        and ci[0] is not None
        and ci[1] is not None
        and ci[0] <= 2.0 * math.pi <= ci[1]
        and clock.get("nearest_known_scale") == "2pi"
        and not _literal_true(clock.get("response_degenerate"))
        and not list(clock.get("blockers") or [])
    )


def _validated_refinement_naturality(
    refinement: dict[str, Any],
    refinement_summary: dict[str, Any],
    neutral_audit: dict[str, Any],
) -> bool:
    required_ladder = {4_096, 16_384, 65_536, 262_144}
    candidates = [refinement, refinement_summary]
    if isinstance(neutral_audit.get("refinement_summary"), dict):
        candidates.append(neutral_audit["refinement_summary"])
    for row in candidates:
        if not isinstance(row, dict):
            continue
        sizes = row.get("sizes") if isinstance(row.get("sizes"), list) else []
        patch_counts = {
            item.get("patch_count")
            for item in sizes
            if isinstance(item, dict)
            and _strict_int(item.get("patch_count"), minimum=1)
        }
        declared_ladder = row.get("required_patch_count_ladder")
        declared_patch_counts = (
            {value for value in declared_ladder if _strict_int(value, minimum=1)}
            if isinstance(declared_ladder, list)
            else set()
        )
        dimension_drift = _finite_number(row.get("candidate_dimension_drift"))
        if (
            _literal_true(row.get("strict_neutral_bulk_refinement_receipt"))
            and _literal_true(row.get("multi_scale"))
            and _literal_true(row.get("candidate_dimension_stable"))
            and _literal_true(row.get("required_ladder_complete"))
            and _strict_int(row.get("run_count"), minimum=4)
            and patch_counts == required_ladder
            and declared_patch_counts == required_ladder
            and list(row.get("missing_required_patch_counts") or []) == []
            and dimension_drift is not None
            and dimension_drift <= 0.10
            and not list(row.get("proof_blockers") or [])
        ):
            return True
    return False


def _issue308_clause_passed(report: dict[str, Any], name: str) -> bool:
    row = (report.get("clauses") or {}).get(name)
    return isinstance(row, dict) and _literal_true(row.get("passed"))


def _strict_int(value: Any, *, minimum: int) -> bool:
    return type(value) is int and value >= minimum


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _sha256_receipt(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _truthy_any(data: dict[str, Any], *keys: str) -> bool:
    return any(_literal_true(data.get(key)) for key in keys)


def _literal_true(value: Any) -> bool:
    """Accept only the JSON/YAML boolean true as theorem evidence.

    Numeric one and strings such as ``"true"`` are deliberately rejected.
    Receipts are proof-carrying data, not command-line truthiness flags.
    """

    return type(value) is bool and value is True


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
    ) or _literal_true(paper_chart.get("declared_bw_2pi_cap_flow_receipt"))
    selected_2pi = bool(
        _literal_true(paper_chart.get("bw_2pi_cap_flow_receipt"))
        or _literal_true(paper_chart.get("declared_bw_2pi_cap_flow_receipt"))
        or _literal_true(emergence.get("transition_two_pi_selected_by_primary"))
        or str(emergence.get("transition_selected_label", "")) == "2pi"
    )
    return bool(chart_receipt and branch_receipt and declared_or_kms_source and selected_2pi)


def _all_manifest_assumptions(manifest: dict[str, Any], *names: str) -> bool:
    return manifest_assumptions_pass(manifest, *names)


def _read_simulation_assumption_manifest(root: Path) -> tuple[dict[str, Any], str | None]:
    explicit_path = root / "simulation_assumption_manifest.json"
    explicit = _read_json(explicit_path)
    if explicit:
        return revalidate_simulation_assumption_manifest(explicit), str(explicit_path)
    for name in ("config.yml", "config.yaml", "config.json"):
        path = root / name
        if not path.exists():
            continue
        try:
            if path.suffix == ".json":
                parsed = json.loads(path.read_text(encoding="utf-8"))
            else:
                parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, yaml.YAMLError, OSError):
            continue
        if not isinstance(parsed, dict):
            continue
        section = parsed.get("simulation_assumptions")
        if not isinstance(section, dict):
            continue
        normalized = dict(parsed)
        if not section.get("assumed") and isinstance(section.get("assumed_bridges"), list):
            bridge_names = {str(value) for value in section["assumed_bridges"]}
            assumed = {
                "bw_2pi_geometric_branch": "BW3_FINITE_CAP_CERTIFICATE" in bridge_names,
                "h3_observer_chart": "CAP_NORMAL_H3_CHART" in bridge_names,
                "record_population_on_h3": "MODULAR_RESPONSE_H3_LOCALIZATION" in bridge_names,
            }
            normalized["simulation_assumptions"] = {**section, "assumed": assumed}
        return simulation_assumption_manifest(normalized), str(path)
    return simulation_assumption_manifest({}), None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _read_issue308_bw_certificate(root: Path) -> dict[str, Any]:
    for name in ("bw_rec_308.json", "BWRec_r.json", "issue_308_primitive_fields.json"):
        payload = _read_json(root / name)
        if payload:
            return issue308_bw_certificate_report(payload)
    return _ignored_precomputed_receipt(
        root,
        (
            "issue_308_bw_certificate_report.json",
            "bw_issue308_certificate_report.json",
            "finite_cap_bw_certificate_report.json",
        ),
        ISSUE_308_BW_CERTIFICATE_RECEIPT,
        "issue_308_requires_primitive_fields_for_local_recomputation",
    )


def _read_issue309_cap_normal_h3_chart(root: Path) -> dict[str, Any]:
    for name in (
        "cap_normal_h3_chart_source.json",
        "issue_309_cap_normal_h3_chart_source.json",
        "cap_normal_h3_primitive_fields.json",
    ):
        payload = _read_json(root / name)
        if payload:
            return cap_normal_h3_chart_report(payload)
    return _ignored_precomputed_receipt(
        root,
        (
            "cap_normal_h3_chart_report.json",
            "issue_309_cap_normal_h3_chart_report.json",
            "cap_normal_h3_chart_receipt.json",
        ),
        CAP_NORMAL_H3_CHART_RECEIPT,
        "issue_309_requires_primitive_fields_for_local_recomputation",
    )


def _read_issue310_modular_response_h3_localization(root: Path) -> dict[str, Any]:
    for name in (
        "modular_response_h3_localization_source.json",
        "issue_310_modular_response_h3_localization_source.json",
        "h3_localization_primitive_fields.json",
    ):
        payload = _read_json(root / name)
        if payload:
            return modular_response_h3_localization_report(payload)
    return _ignored_precomputed_receipt(
        root,
        (
            "modular_response_h3_localization_report.json",
            "issue_310_modular_response_h3_localization_report.json",
            "h3_localization_receipt.json",
        ),
        MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT,
        "issue_310_requires_primitive_fields_for_local_recomputation",
    )


def _ignored_precomputed_receipt(
    root: Path,
    names: tuple[str, ...],
    receipt_key: str,
    blocker: str,
) -> dict[str, Any]:
    """Fail closed when only a mutable derived report is present.

    A result JSON is not its own proof.  The theorem contract independently
    reruns each verifier from primitive fields; report-only inputs are retained
    solely as diagnostics and can never raise a theorem receipt.
    """

    for name in names:
        if _read_json(root / name):
            return {
                "mode": "unverified_precomputed_report_ignored",
                receipt_key: False,
                "receipt": False,
                "terminal_status": "UNVERIFIED_PRECOMPUTED_REPORT",
                "ignored_precomputed_report": name,
                "blockers": [blocker],
            }
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
        "- explicitly assumed visual universe: "
        f"`{str(report['SIMULATION_ASSUMED_VISUAL_UNIVERSE_RECEIPT']).lower()}`",
        "- explicitly assumed CMB visualization: "
        f"`{str(report['SIMULATION_ASSUMED_CMB_VISUALIZATION_RECEIPT']).lower()}`",
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
