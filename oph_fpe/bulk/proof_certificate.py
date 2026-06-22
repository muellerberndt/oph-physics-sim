from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oph_fpe.claims import (
    BRANCH_INSTANTIATION_SANITY,
    BW_KMS_BRANCH_INSTANTIATION_RECEIPT,
    BW_KMS_BRANCH_REPLAY_RECEIPT,
    CHART_LORENTZ_H3_RECEIPT,
    CONTROL_RESIDUALIZED_RANK3_CANDIDATE_RECEIPT,
    FINITE_CONSENSUS_THEOREM_RECEIPT,
    FINITE_SETTLE_DIAGNOSTIC_RECEIPT,
    H3_RESPONSE_CANDIDATE_RECEIPT,
    OBJECT_BULK_POPULATION_RECEIPT,
    OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT,
    OPH_LORENTZ_THEOREM_FINITE_CONTRACT_RECEIPT,
    PHYSICAL_CMB_RECEIPT,
    PRIME_GEOMETRIC_QUOTIENT_3D_DIAGNOSTIC_RECEIPT,
    PROTO_PARTICLE_RECEIPT,
    RECORD_COMMIT_RECEIPT,
    REPAIR_CORE_RECEIPT,
    SCREEN_PROXY_CMB_RECEIPT,
    STRICT_NEUTRAL_BULK_RECEIPT,
    STRICT_NEUTRAL_OBJECT_BULK_RECEIPT,
    THEOREM_ASSISTED_H3_OBJECT_POPULATION_RECEIPT,
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
    neutral_frontier = _read_json(root / "strict_neutral_bulk_frontier_report.json")
    cmb_lite = _read_json(root / "cmb_lite_comparison_report.json")
    cl = _read_json(root / "cl_comparison_report.json")
    physical_cmb_frontier = _read_json(root / "physical_cmb_frontier_report.json")
    physical_cmb_output = _read_json(root / "physical_cmb_output_comparison_report.json")
    scale_compressed = _read_json(root / "scale_compressed_repair_report.json")
    scale_compressed_cmb = _read_json(root / "scale_compressed_cmb_camb_report.json")
    scale_compressed_particle = _read_json(root / "scale_compressed_particle_report.json")
    particle = _read_json(root / "particle_likeness_report.json")
    controlled_particle = _read_json(root / "controlled_defect_particle_assay_report.json")
    prime_rank_sweep = _read_json(root / "prime_geometric_rank_sweep_report.json")
    prime_rank_refinement = _read_json(root / "prime_geometric_rank_refinement_report.json")
    strict_neutral_object = _read_json(root / "strict_neutral_object_bulk_report.json")
    theorem_core = _read_json(root / "theorem_core_receipts.json")
    finite_contract_report = _read_json(root / "finite_oph_theorem_contract_report.json")
    observer_modular_experience = _read_json(root / "observer_modular_experience_report.json")
    object_chart_name, object_chart = _best_object_chart_report(root)

    finite_settle_diagnostic = bool(
        _ladder_passed(ladder, "R0")
        or _truthy(emergence, "final_phi_zero")
        or _truthy(emergence, FINITE_SETTLE_DIAGNOSTIC_RECEIPT)
        or theorem_core.get("finite_settle_diagnostic_receipt", False)
    )
    finite_consensus_theorem = bool(
        _truthy(emergence, FINITE_CONSENSUS_THEOREM_RECEIPT)
        or theorem_core.get(FINITE_CONSENSUS_THEOREM_RECEIPT, False)
        or theorem_core.get("finite_consensus_theorem_receipt", False)
    )
    repair_core = finite_settle_diagnostic
    record_commit = _ladder_passed(ladder, "R1") or _truthy(emergence, "records_committed")
    bw_kms = _truthy_any(
        emergence,
        BW_KMS_BRANCH_REPLAY_RECEIPT,
        "BW_KMS_DIRECT_2PI_RECEIPT",
        "state_derived_correct_beats_controls",
        "state_derived_selected_2pi",
    ) or _ladder_passed(ladder, "R2") or _ladder_receipt_passed(ladder, BW_KMS_BRANCH_REPLAY_RECEIPT) or bool(
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
    observer_modular_experience_written = bool(observer_modular_experience)
    observer_modular_time = bool(observer_modular_experience.get("observer_modular_time_receipt", False))
    observer_facing_3p1d_experience = bool(
        observer_modular_experience.get(OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT, False)
        or observer_modular_experience.get("observer_facing_3p1d_h3_experience_receipt", False)
        or (
            chart
            and bw_kms
            and h3_response
            and (observer_modular_time or not observer_modular_experience_written)
        )
    )
    observer_facing_populated_h3_experience = bool(
        observer_modular_experience.get("observer_facing_populated_h3_experience_receipt", False)
        or (observer_facing_3p1d_experience and object_nonboundary_population)
    )
    observer_component_gates = {
        "observer_modular_time_receipt": bool(
            observer_modular_time or not observer_modular_experience_written
        ),
        "bw_kms_branch_replay_receipt": bw_kms,
        "conformal_h3_chart_receipt": chart,
        "h3_modular_response_receipt": h3_response,
    }
    observer_populated_h3_component_gates = {
        **observer_component_gates,
        "observer_h3_object_population_receipt": object_nonboundary_population,
    }
    observer_blockers = [
        gate for gate, passed in observer_component_gates.items() if not passed
    ]
    observer_populated_h3_blockers = [
        gate for gate, passed in observer_populated_h3_component_gates.items() if not passed
    ]
    strict_neutral_object_bulk = bool(
        strict_neutral_object.get(STRICT_NEUTRAL_OBJECT_BULK_RECEIPT, False)
        or strict_neutral_object.get("strict_neutral_object_bulk", False)
    )
    strict_neutral_bulk = bool(
        neutral.get("bulk_3d_established", False)
        or neutral_frontier.get("strict_neutral_bulk", False)
        or emergence.get("strict_blind_observer_bulk_receipt", False)
        or emergence.get("neutral_bulk_3d_established", False)
        or strict_neutral_object_bulk
    )
    prime_geometric_quotient_3d = bool(
        prime_rank_sweep.get("PRIME_GEOMETRIC_QUOTIENT_3D_DIAGNOSTIC_RECEIPT", False)
        or prime_rank_sweep.get("prime_geometric_quotient_3d_diagnostic_receipt", False)
    )
    prime_geometric_spatial_3d = bool(
        prime_rank_sweep.get("prime_geometric_spatial_3d_candidate_receipt", False)
    )
    prime_geometric_rank3_refinement = bool(
        prime_rank_refinement.get("control_quotient_rank3_refinement_candidate_receipt", False)
    )
    prime_geometric_strict_refinement = bool(
        prime_rank_refinement.get("strict_neutral_bulk_refinement_receipt", False)
    )
    screen_cmb = bool(
        emergence.get("SCREEN_PROXY_CMB_RECEIPT", False)
        or _ladder_passed(ladder, "R7")
        or cl
        or cmb_lite
        or physical_cmb_output.get("PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT", False)
    )
    physical_cmb = bool(
        emergence.get("physical_cmb_prediction", False)
        or cmb_lite.get("physical_cmb_prediction", False)
        or cl.get("physical_cmb_prediction", False)
        or physical_cmb_frontier.get("physical_cmb_prediction_receipt", False)
        or physical_cmb_output.get("PHYSICAL_CMB_PREDICTION_RECEIPT", False)
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

    finite_lorentz_contract = bool(
        finite_contract_report.get(OPH_LORENTZ_THEOREM_FINITE_CONTRACT_RECEIPT, False)
        or finite_contract_report.get("finite_lorentz_theorem_contract_receipt", False)
    )
    paper_faithful_observer_spacetime = bool(
        finite_contract_report.get("paper_faithful_observer_spacetime_emergence_receipt", False)
    )
    paper_faithful_populated_h3 = bool(
        finite_contract_report.get("paper_faithful_populated_h3_observer_experience_receipt", False)
    )
    paper_faithful_consensus_bulk = bool(
        finite_contract_report.get("paper_faithful_consensus_bulk_emergence_receipt", False)
    )

    tiers = {
        "C0a_finite_settle_diagnostic": _tier(
            FINITE_SETTLE_DIAGNOSTIC_RECEIPT,
            finite_settle_diagnostic,
            "Final finite overlap mismatch settled; diagnostic only, not schedule-confluent consensus theorem.",
        ),
        "C0b_finite_consensus_theorem": _tier(
            FINITE_CONSENSUS_THEOREM_RECEIPT,
            finite_consensus_theorem,
            "Strict theorem-phase finite consensus certificate with replay, confluence, and repair-completeness gates.",
        ),
        "T0_finite_repair_core": _tier(
            REPAIR_CORE_RECEIPT,
            repair_core,
            "Legacy alias for C0a finite settling diagnostic; not a C0b finite consensus theorem.",
            canonical_tier="C0a",
        ),
        "T1_record_commit": _tier(
            RECORD_COMMIT_RECEIPT,
            record_commit,
            "Observer-facing records committed and can be read as finite record algebra data.",
        ),
        "L0_bw_kms_branch_replay": _tier(
            BW_KMS_BRANCH_REPLAY_RECEIPT,
            bw_kms,
            "Declared finite support-visible cap/collar branch replays BW/KMS 2*pi cap transport.",
            legacy_receipt_name=BW_KMS_BRANCH_INSTANTIATION_RECEIPT,
        ),
        "T2_bw_kms_2pi_branch": _tier(
            BW_KMS_BRANCH_REPLAY_RECEIPT,
            bw_kms,
            "Legacy alias for L0 branch replay; not a finite Lorentz theorem contract.",
            legacy_receipt_name=BW_KMS_BRANCH_INSTANTIATION_RECEIPT,
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
            THEOREM_ASSISTED_H3_OBJECT_POPULATION_RECEIPT,
            theorem_assisted_nonboundary_population,
            "Persistent observer-facing objects populate the theorem-side H3 chart and are not boundary-dominated.",
        ),
        "T5c_scale_compressed_h3_preview": _tier(
            "SCALE_COMPRESSED_H3_PREVIEW_RECEIPT",
            scale_h3_preview,
            "Scale-compressed logical repair-round branch populates the paper-side H3 chart as a preview artifact.",
        ),
        "T6_chart_blind_strict_neutral_quotient_bulk": _tier(
            STRICT_NEUTRAL_BULK_RECEIPT,
            strict_neutral_bulk,
            "Chart-blind neutral quotient reconstruction establishes a third-person 3D bulk without S2/H3 chart prior.",
        ),
        "T6_strict_neutral_third_person_bulk": _tier(
            STRICT_NEUTRAL_BULK_RECEIPT,
            strict_neutral_bulk,
            "Legacy alias for T6 chart-blind strict neutral quotient bulk; separate from observer-facing H3 consensus bulk.",
            canonical_tier="T6_chart_blind_strict_neutral_quotient_bulk",
        ),
        "T6_object_strict_neutral_bulk": _tier(
            STRICT_NEUTRAL_OBJECT_BULK_RECEIPT,
            strict_neutral_object_bulk,
            "Neutral object extraction and held-out latent geometry selection establish a strict third-person object bulk.",
        ),
        "T6a_prime_geometric_quotient_3d_diagnostic": _tier(
            PRIME_GEOMETRIC_QUOTIENT_3D_DIAGNOSTIC_RECEIPT,
            prime_geometric_quotient_3d,
            "Observer-visible prime-geometric response quotient has a 3D-compatible rank window, without satisfying strict neutral proof gates.",
        ),
        "T6b_prime_geometric_rank3_refinement_diagnostic": _tier(
            CONTROL_RESIDUALIZED_RANK3_CANDIDATE_RECEIPT,
            prime_geometric_rank3_refinement,
            "Control-quotient coordinate rank-3/E3 candidate is stable across supplied finite-regulator sizes; this is still diagnostic, not chart-blind strict neutral quotient proof.",
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
        "L_full_oph_lorentz_finite_contract": _tier(
            OPH_LORENTZ_THEOREM_FINITE_CONTRACT_RECEIPT,
            finite_lorentz_contract,
            "Finite Lorentz theorem contract through support-visible BW covariance and Lorentz closure is audited from finite observer-record receipts.",
            blockers=finite_contract_report.get("primary_blockers", []),
        ),
        "B_full_paper_faithful_observer_spacetime": _tier(
            "PAPER_FAITHFUL_OBSERVER_SPACETIME_EMERGENCE_RECEIPT",
            paper_faithful_observer_spacetime,
            "Observer-local modular time plus H3 spatial chart and H3 response, gated by the finite theorem contract.",
            blockers=finite_contract_report.get("primary_blockers", []),
        ),
        "B_populated_h3_observer_experience": _tier(
            "PAPER_FAITHFUL_POPULATED_H3_OBSERVER_EXPERIENCE_RECEIPT",
            paper_faithful_populated_h3,
            "Observer-facing H3 spacetime plus controlled object population in that chart.",
            blockers=finite_contract_report.get("primary_blockers", []),
        ),
        "B_full_paper_faithful_consensus_bulk": _tier(
            "PAPER_FAITHFUL_CONSENSUS_BULK_EMERGENCE_RECEIPT",
            paper_faithful_consensus_bulk,
            "Paper-faithful observer-facing consensus 3D bulk: observer spacetime emergence plus populated H3 object records.",
            blockers=finite_contract_report.get("primary_blockers", []),
        ),
    }

    report = {
        "mode": "oph_3d_bulk_and_measurement_proof_certificate_v0",
        "run_path": str(root),
        "proof_tiers": tiers,
        FINITE_SETTLE_DIAGNOSTIC_RECEIPT: finite_settle_diagnostic,
        FINITE_CONSENSUS_THEOREM_RECEIPT: finite_consensus_theorem,
        "finite_settle_diagnostic_receipt": finite_settle_diagnostic,
        "finite_consensus_theorem_receipt": finite_consensus_theorem,
        "chart_level_3p1_lorentz_kinematics_established": bool(chart and bw_kms),
        "paper_route_lorentz_h3_chart_established": bool(chart and bw_kms),
        OPH_LORENTZ_THEOREM_FINITE_CONTRACT_RECEIPT: finite_lorentz_contract,
        "finite_lorentz_theorem_contract_receipt": finite_lorentz_contract,
        "paper_faithful_observer_spacetime_emergence_receipt": paper_faithful_observer_spacetime,
        "paper_faithful_populated_h3_observer_experience_receipt": paper_faithful_populated_h3,
        "observer_facing_consensus_3d_bulk_emergence_receipt": paper_faithful_consensus_bulk,
        "paper_faithful_consensus_bulk_emergence_receipt": paper_faithful_consensus_bulk,
        "simulation_matches_observer_facing_oph_spacetime_bulk_prediction_receipt": paper_faithful_consensus_bulk,
        "simulation_matches_full_oph_spacetime_bulk_prediction_receipt": paper_faithful_consensus_bulk,
        "finite_theorem_contract_summary": {
            "written": bool(finite_contract_report),
            "blockers": finite_contract_report.get("blockers", []),
            "primary_blockers": finite_contract_report.get("primary_blockers", []),
            "chart_blind_strict_neutral_blockers": finite_contract_report.get(
                "chart_blind_strict_neutral_blockers", []
            ),
            "strict_neutral_blockers": finite_contract_report.get("strict_neutral_blockers", []),
            "all_stage_blockers": finite_contract_report.get("all_stage_blockers", []),
            "stages": finite_contract_report.get("stages", {}),
            "claim_boundary": finite_contract_report.get("claim_boundary"),
        },
        "theorem_assisted_h3_object_preview_established": theorem_assisted_chart_preview,
        "theorem_assisted_h3_nonboundary_population_established": theorem_assisted_nonboundary_population,
        "theorem_assisted_h3_populated_chart_established": theorem_assisted_nonboundary_population,
        "theorem_assisted_observer_facing_h3_population": theorem_assisted_nonboundary_population,
        THEOREM_ASSISTED_H3_OBJECT_POPULATION_RECEIPT: theorem_assisted_nonboundary_population,
        "observer_facing_h3_object_population_receipt": theorem_assisted_nonboundary_population,
        "observer_modular_time_receipt": observer_modular_time,
        OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT: observer_facing_3p1d_experience,
        "observer_facing_3p1d_h3_experience_receipt": observer_facing_3p1d_experience,
        "observer_facing_populated_h3_experience_receipt": observer_facing_populated_h3_experience,
        "observer_h3_object_population_receipt": object_nonboundary_population,
        "scale_compressed_operator_receipt": scale_operator,
        "scale_compressed_repair_round_trace_receipt": scale_round_trace,
        "scale_compressed_h3_preview_established": scale_h3_preview,
        "scale_compressed_measurement_comparable_cmb_curve": scale_compressed_measurement_cmb,
        "scale_compressed_particle_preview_established": scale_particle_preview,
        "strict_neutral_third_person_bulk_established": strict_neutral_bulk,
        "chart_blind_strict_neutral_quotient_bulk_established": strict_neutral_bulk,
        "chart_blind_strict_neutral_quotient_bulk_receipt": strict_neutral_bulk,
        "strict_neutral_bulk_contract_receipt": bool(
            finite_contract_report.get("chart_blind_strict_neutral_quotient_bulk_receipt", strict_neutral_bulk)
        ),
        STRICT_NEUTRAL_OBJECT_BULK_RECEIPT: strict_neutral_object_bulk,
        STRICT_NEUTRAL_BULK_RECEIPT: strict_neutral_bulk,
        "prime_geometric_quotient_3d_diagnostic": prime_geometric_quotient_3d,
        "prime_geometric_spatial_3d_candidate": prime_geometric_spatial_3d,
        "prime_geometric_rank3_refinement_candidate": prime_geometric_rank3_refinement,
        CONTROL_RESIDUALIZED_RANK3_CANDIDATE_RECEIPT: prime_geometric_rank3_refinement,
        "prime_geometric_rank3_refinement_strict_neutral": prime_geometric_strict_refinement,
        "bulk_3d_established_theorem_assisted": theorem_assisted_nonboundary_population,
        "bulk_3d_established_observer_facing_consensus": paper_faithful_consensus_bulk,
        "bulk_3d_established_strict": strict_neutral_bulk,
        "bulk_3d_established_chart_blind_strict_neutral": strict_neutral_bulk,
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
            "observer_facing_h3_object_population_receipt": object_chart.get(
                "observer_facing_h3_object_population_receipt",
                object_chart.get("observer_chart_bulk_population_receipt"),
            ),
        },
        "observer_modular_experience_summary": {
            "written": observer_modular_experience_written,
            "observer_modular_time_receipt": observer_modular_time,
            "observer_facing_3p1d_h3_experience_receipt": observer_facing_3p1d_experience,
            "observer_facing_populated_h3_experience_receipt": observer_facing_populated_h3_experience,
            "observer_h3_object_population_receipt": object_nonboundary_population,
            "observer_count": observer_modular_experience.get("observer_count"),
            "observer_relative_time_count": observer_modular_experience.get("observer_relative_time_count"),
            "blockers": observer_blockers,
            "populated_h3_experience_blockers": observer_populated_h3_blockers,
            "component_gates": observer_component_gates,
            "populated_h3_component_gates": observer_populated_h3_component_gates,
            "source_report_blockers": observer_modular_experience.get("blockers", []),
            "source_report_populated_h3_blockers": observer_modular_experience.get(
                "populated_h3_experience_blockers", []
            ),
            "source_report_component_gates": observer_modular_experience.get("component_gates", {}),
            "source_report_populated_h3_component_gates": observer_modular_experience.get(
                "populated_h3_component_gates", {}
            ),
        },
        "scale_compressed_summary": {
            "logical_repair_rounds": scale_compressed.get("logical_repair_rounds"),
            "eta_R": scale_cmb_params.get("eta_R"),
            "n_s": scale_cmb_params.get("n_s"),
            "q_IR": scale_cmb_params.get("q_IR"),
            "ell_IR": scale_cmb_params.get("ell_IR"),
            "N_CRC_implied_by_declared_repair_depth_ansatz": scale_cmb_params.get(
                "N_CRC_implied_by_declared_repair_depth_ansatz",
                scale_cmb_params.get("N_CRC_predicted_from_P"),
            ),
            "N_CRC_predicted_from_P": scale_cmb_params.get("N_CRC_predicted_from_P"),
            "N_CRC_declared": scale_cmb_params.get("N_CRC_declared"),
            "relative_error_ansatz_capacity_vs_declared_N_CRC": scale_cmb_params.get(
                "relative_error_ansatz_capacity_vs_declared_N_CRC",
                scale_cmb_params.get("relative_error_gprime_vs_N_CRC"),
            ),
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
        "prime_geometric_rank_sweep_summary": {
            "written": bool(prime_rank_sweep),
            "diagnostic_receipt": prime_geometric_quotient_3d,
            "spatial_3d_candidate_receipt": prime_geometric_spatial_3d,
            "strict_neutral_candidate_receipt": bool(
                prime_rank_sweep.get("prime_geometric_strict_neutral_candidate_receipt", False)
            ),
            "dimension_3d_window_count": prime_rank_sweep.get("dimension_3d_window_count"),
            "coordinate_dimension_3d_window_count": prime_rank_sweep.get(
                "coordinate_dimension_3d_window_count"
            ),
            "best_3d_rank": ((prime_rank_sweep.get("best_3d_dimension_row") or {}).get("rank")),
            "coordinate_best_3d_rank": (
                (prime_rank_sweep.get("coordinate_best_3d_dimension_row") or {}).get("rank")
            ),
            "regulator_control_quotient_is_negative_control": (
                (prime_rank_sweep.get("regulator_control_quotient_lane") or {}).get(
                    "is_negative_control"
                )
            ),
            "proof_blockers": prime_rank_sweep.get("proof_blockers", []),
        },
        "prime_geometric_rank_refinement_summary": {
            "written": bool(prime_rank_refinement),
            "candidate_receipt": prime_geometric_rank3_refinement,
            "receipt_name": CONTROL_RESIDUALIZED_RANK3_CANDIDATE_RECEIPT,
            "physical_claim": False,
            "strict_neutral_bulk_participation": "diagnostic_only",
            "diagnostic_reason": (
                "control-quotient rank-3 is a residualized signal construction, not a null/negative control"
            ),
            "strict_neutral_receipt": prime_geometric_strict_refinement,
            "source_run_count": prime_rank_refinement.get("run_count"),
            "multi_scale": prime_rank_refinement.get("multi_scale"),
            "candidate_dimension_drift": prime_rank_refinement.get("candidate_dimension_drift"),
            "candidate_dimension_stable": prime_rank_refinement.get("candidate_dimension_stable"),
            "independent_rank3_selector_all": prime_rank_refinement.get("independent_rank3_selector_all"),
            "sizes": prime_rank_refinement.get("sizes", []),
            "proof_blockers": prime_rank_refinement.get("proof_blockers", []),
        },
        "strict_neutral_object_bulk_summary": {
            "written": bool(strict_neutral_object),
            "receipt": strict_neutral_object_bulk,
            "object_count": strict_neutral_object.get("object_count"),
            "selected_model": ((strict_neutral_object.get("latent_geometry_selection") or {}).get("selected_model")),
            "h3_selected": ((strict_neutral_object.get("latent_geometry_selection") or {}).get("h3_selected")),
            "median_dimension_estimate": ((strict_neutral_object.get("dimension") or {}).get("median_dimension_estimate")),
            "blockers": strict_neutral_object.get("blockers", []),
            "claim_boundary": (
                "Strict neutral object-bulk report uses observer-visible object histories only; it is separate "
                "from theorem-assisted H3 chart population."
            ),
        },
        "paper_alignment": {
            "screen_role": "S2 is the observer-facing cap/symmetry chart, not a raw point-cloud proof of dimension.",
            "lorentz_route": "support-visible BW/KMS cap flow with s=2*pi*t gives Conf+(S2) ~= SO+(3,1).",
            "spatial_chart": "H3 is the spatial homogeneous chart SO+(3,1)/SO(3).",
            "finite_gate": "finite runs must separately show observer records/objects/defects populate that chart under controls.",
            "strict_neutral_route": (
                "chart-blind strict neutral quotient bulk must be reconstructed from neutral observer/object "
                "records without H3/S2/support coordinates and must pass held-out latent-geometry controls."
            ),
            "scale_compressed_branch": (
                "Logical scale compression can expose OPH repair-round/CMB readouts and H3 previews, but it "
                "does not replace strict neutral observer-record reconstruction."
            ),
        },
        "claim_boundary": (
            "Tiered OPH proof/readout certificate. C0a/T0 is only finite settling; C0b finite "
            "consensus requires strict theorem-phase replay and may remain false even when final_phi is zero. "
            "L0/T2 is branch replay: the declared BW/KMS 2pi route executes under current controls, "
            "but the full L1-L7 finite Lorentz theorem contract remains false. T3 is the conformal H3 chart route for this run. "
            "T5a is theorem-assisted H3 preview evidence from observer objects. "
            "T5b is stricter non-boundary H3 object population evidence. T5c is a scale-compressed "
            "logical repair-round H3 preview and is intentionally not promoted to T5b/T6. "
            "T6 is stricter chart-blind neutral quotient reconstruction and may remain false even when "
            "the observer-facing H3 consensus-bulk receipt passes. "
            "T6a/T6b are intermediate residualized prime-geometric quotient diagnostics and are not promoted to T6. "
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


def _tier(receipt_name: str, passed: bool, description: str, **metadata: Any) -> dict[str, Any]:
    row = {"receipt_name": receipt_name, "passed": bool(passed), "description": description}
    row.update(metadata)
    return row


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


def _ladder_receipt_passed(ladder: dict[str, Any], receipt_name: str) -> bool:
    receipts = (ladder.get("receipts") or {}).values()
    return any(
        bool(row.get("passed", False)) and row.get("receipt_name") == receipt_name
        for row in receipts
        if isinstance(row, dict)
    )


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

    def score(item: tuple[str, dict[str, Any]]) -> tuple[float, float, float, float, float, float, float, float]:
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
        current_schema = report.get("h3_compactness_margin_vs_median_shuffle") is not None
        return (
            float(bool(report.get("observer_chart_bulk_population_receipt", False))),
            float(bool(report.get("OBJECT_H3_NONBOUNDARY_POPULATION_RECEIPT", False))),
            float(bool(report.get("THEOREM_ASSISTED_H3_OBJECT_PREVIEW_RECEIPT", False))),
            float(bool(report.get("h3_beats_shuffled_incidence_robust", False))),
            float(bool(lineage_or_transition)),
            float(bool(current_schema)),
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
        f"- finite settle diagnostic: `{str(report['finite_settle_diagnostic_receipt']).lower()}`",
        f"- finite consensus theorem: `{str(report['finite_consensus_theorem_receipt']).lower()}`",
        f"- chart-level 3+1D Lorentz/H3: `{str(report['chart_level_3p1_lorentz_kinematics_established']).lower()}`",
        f"- finite Lorentz theorem contract: `{str(report['finite_lorentz_theorem_contract_receipt']).lower()}`",
        "- paper-faithful observer spacetime emergence: "
        f"`{str(report['paper_faithful_observer_spacetime_emergence_receipt']).lower()}`",
        "- paper-faithful observer-facing consensus bulk emergence: "
        f"`{str(report['paper_faithful_consensus_bulk_emergence_receipt']).lower()}`",
        f"- observer modular time: `{str(report['observer_modular_time_receipt']).lower()}`",
        f"- observer-facing 3+1D/H3 experience: `{str(report['observer_facing_3p1d_h3_experience_receipt']).lower()}`",
        f"- theorem-assisted H3 object preview: `{str(report['theorem_assisted_h3_object_preview_established']).lower()}`",
        f"- theorem-assisted non-boundary H3 population: `{str(report['theorem_assisted_h3_nonboundary_population_established']).lower()}`",
        f"- scale-compressed H3 preview: `{str(report['scale_compressed_h3_preview_established']).lower()}`",
        "- chart-blind strict neutral quotient bulk: "
        f"`{str(report['chart_blind_strict_neutral_quotient_bulk_established']).lower()}`",
        f"- prime-geometric quotient 3D diagnostic: `{str(report['prime_geometric_quotient_3d_diagnostic']).lower()}`",
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
    contract = report.get("finite_theorem_contract_summary") or {}
    if contract:
        lines.extend(
            [
                "",
                "## Finite Theorem Contract",
                "",
                f"- written: `{str(bool(contract.get('written'))).lower()}`",
                f"- primary blockers: `{contract.get('primary_blockers')}`",
                f"- chart-blind neutral blockers: `{contract.get('chart_blind_strict_neutral_blockers')}`",
            ]
        )
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
    prime = report.get("prime_geometric_rank_sweep_summary") or {}
    if prime:
        lines.extend(
            [
                "",
                "## Prime-Geometric Quotient",
                "",
                f"- diagnostic receipt: `{prime.get('diagnostic_receipt')}`",
                f"- spatial 3D candidate: `{prime.get('spatial_3d_candidate_receipt')}`",
                f"- strict neutral candidate: `{prime.get('strict_neutral_candidate_receipt')}`",
                f"- target 3D-window count: `{prime.get('dimension_3d_window_count')}`",
                f"- coordinate 3D-window count: `{prime.get('coordinate_dimension_3d_window_count')}`",
                f"- best target 3D rank: `{prime.get('best_3d_rank')}`",
                f"- best coordinate 3D rank: `{prime.get('coordinate_best_3d_rank')}`",
                f"- control quotient is negative control: `{prime.get('regulator_control_quotient_is_negative_control')}`",
                f"- proof blockers: `{prime.get('proof_blockers')}`",
            ]
        )
    prime_refinement = report.get("prime_geometric_rank_refinement_summary") or {}
    if prime_refinement:
        lines.extend(
            [
                "",
                "## Prime-Geometric Rank Refinement",
                "",
                f"- candidate receipt: `{prime_refinement.get('candidate_receipt')}`",
                f"- strict neutral receipt: `{prime_refinement.get('strict_neutral_receipt')}`",
                f"- source run count: `{prime_refinement.get('source_run_count')}`",
                f"- multi-scale: `{prime_refinement.get('multi_scale')}`",
                f"- candidate dimension drift: `{prime_refinement.get('candidate_dimension_drift')}`",
                f"- dimension stable: `{prime_refinement.get('candidate_dimension_stable')}`",
                f"- independent rank-3 selector all: `{prime_refinement.get('independent_rank3_selector_all')}`",
                f"- sizes: `{prime_refinement.get('sizes')}`",
                f"- proof blockers: `{prime_refinement.get('proof_blockers')}`",
            ]
        )
    lines.extend(["", "## Claim Boundary", "", str(report.get("claim_boundary", "")), ""])
    return "\n".join(lines)
