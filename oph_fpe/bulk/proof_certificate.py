from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from oph_fpe.claims import (
    BRANCH_INSTANTIATION_SANITY,
    BULK_WORLDLINE_PRECURSOR_RECEIPT,
    BW_KMS_BRANCH_INSTANTIATION_RECEIPT,
    BW_KMS_BRANCH_REPLAY_RECEIPT,
    CAP_NORMAL_H3_CHART_RECEIPT,
    CHART_LORENTZ_H3_RECEIPT,
    CLASSICAL_CARRIER_MODE_RECEIPT,
    COLORED_DECONFINEMENT_RECEIPT,
    CONTROL_RESIDUALIZED_RANK3_CANDIDATE_RECEIPT,
    EINSTEIN_BRANCH_ENTRY_RECEIPT,
    EINSTEIN_BRIDGE_DEPENDENCY_DISCHARGE_RECEIPT,
    EINSTEIN_BRIDGE_RUN_RECEIPTS_RECEIPT,
    FINITE_CONSENSUS_THEOREM_RECEIPT,
    FINITE_SETTLE_DIAGNOSTIC_RECEIPT,
    H3_RESPONSE_CANDIDATE_RECEIPT,
    MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT,
    OBJECT_BULK_POPULATION_RECEIPT,
    OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT,
    OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_RECEIPT,
    OPH_EINSTEIN_BRIDGE_MANIFEST_RECEIPT,
    OPH_LORENTZ_THEOREM_FINITE_CONTRACT_RECEIPT,
    PHYSICAL_CMB_RECEIPT,
    PHYSICAL_GRAVITY_PREDICTION_RECEIPT,
    PRIME_GEOMETRIC_QUOTIENT_3D_DIAGNOSTIC_RECEIPT,
    PROTO_PARTICLE_RECEIPT,
    PRODUCTION_PARTICLE_MATTER_RECEIPT,
    PRODUCTION_GRAVITY_RECEIPT,
    RECORD_COMMIT_RECEIPT,
    REPAIR_CORE_RECEIPT,
    SCREEN_PROXY_CMB_RECEIPT,
    STRICT_NEUTRAL_BULK_RECEIPT,
    STRICT_NEUTRAL_OBJECT_BULK_RECEIPT,
    QUANTUM_PARTICLE_RECEIPT,
    THEOREM_ASSISTED_H3_OBJECT_POPULATION_RECEIPT,
    with_claim_metadata,
)
from oph_fpe.bulk.einstein_bridge import einstein_bridge_manifest_report
from oph_fpe.bulk.particle_contract import particle_promotion_contract_report


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
    cap_normal_h3_chart = _read_json(root / "cap_normal_h3_chart_report.json")
    modular_response_h3_localization = _read_json(root / "modular_response_h3_localization_report.json")
    transition_selection = _read_json(root / "transition_selection_report.json")
    strict_neutral_report = _read_json(root / "strict_neutral_bulk_report.json")
    cmb_lite = _read_json(root / "cmb_lite_comparison_report.json")
    cl = _read_json(root / "cl_comparison_report.json")
    physical_cmb_input = _read_json(root / "physical_cmb_input_report.json")
    physical_cmb_input_validation = _read_json(root / "physical_cmb_input_validation.json")
    frozen_transfer_likelihood = _read_json(root / "frozen_transfer_likelihood_report.json")
    physical_cmb_frontier = _read_json(root / "physical_cmb_frontier_report.json")
    physical_cmb_output = _read_json(root / "physical_cmb_output_comparison_report.json")
    scale_compressed = _read_json(root / "scale_compressed_repair_report.json")
    scale_compressed_cmb = _read_json(root / "scale_compressed_cmb_camb_report.json")
    scale_compressed_particle = _read_json(root / "scale_compressed_particle_report.json")
    particle = _read_json(root / "particle_likeness_report.json")
    controlled_particle = _read_json(root / "controlled_defect_particle_assay_report.json")
    particle_contract = particle_promotion_contract_report(root)
    prime_rank_sweep = _read_json(root / "prime_geometric_rank_sweep_report.json")
    prime_rank_refinement = _read_json(root / "prime_geometric_rank_refinement_report.json")
    strict_neutral_object = _read_json(root / "strict_neutral_object_bulk_report.json")
    theorem_core = _read_json(root / "theorem_core_receipts.json")
    persisted_finite_contract_report = _read_json(
        root / "finite_oph_theorem_contract_report.json"
    )
    finite_contract_report = _independently_recomputed_finite_contract(root)
    einstein_branch_report = _read_json(root / "einstein_branch_entry_report.json")
    explicit_einstein_bridge_manifest = _read_json(root / "einstein_bridge_manifest.json")
    einstein_bridge_manifest = explicit_einstein_bridge_manifest or einstein_bridge_manifest_report(root)
    use_einstein_bridge_manifest = bool(explicit_einstein_bridge_manifest)
    observer_modular_experience = _read_json(root / "observer_modular_experience_report.json")
    object_chart_name, object_chart = _best_object_chart_report(root)

    finite_settle_diagnostic = bool(
        _ladder_passed(ladder, "R0")
        or _truthy(emergence, "final_phi_zero")
        or _truthy(emergence, FINITE_SETTLE_DIAGNOSTIC_RECEIPT)
        or _truthy(theorem_core, "finite_settle_diagnostic_receipt")
    )
    finite_consensus_declaration_diagnostic = bool(
        _truthy(emergence, FINITE_CONSENSUS_THEOREM_RECEIPT)
        or _truthy(theorem_core, FINITE_CONSENSUS_THEOREM_RECEIPT)
        or _truthy(theorem_core, "finite_consensus_theorem_receipt")
    )
    finite_consensus_validation = _finite_consensus_validation_from_contract(
        finite_contract_report
    )
    finite_consensus_theorem = _truthy(finite_consensus_validation, "passed")
    repair_core = finite_settle_diagnostic
    record_commit_declaration_diagnostic = bool(
        _ladder_passed(ladder, "R1") or _truthy(emergence, "records_committed")
    )
    bw_kms_declaration_diagnostic = _truthy_any(
        emergence,
        BW_KMS_BRANCH_REPLAY_RECEIPT,
        "BW_KMS_DIRECT_2PI_RECEIPT",
        "state_derived_correct_beats_controls",
        "state_derived_selected_2pi",
    ) or _ladder_passed(ladder, "R2") or _ladder_receipt_passed(ladder, BW_KMS_BRANCH_REPLAY_RECEIPT) or bool(
        _truthy(paper_chart, "bw_2pi_cap_flow_receipt")
        or (
            transition_selection.get("primary_source") == "kms_collar_transport_response"
            and transition_selection.get("two_pi_selected") is True
            and transition_selection.get("response_degenerate") is False
        )
    )
    chart_declaration_diagnostic = _truthy_any(
        emergence,
        "PAPER_THEOREM_3D_BULK_CHART_RECEIPT",
        "CHART_LEVEL_CONFORMAL_LORENTZ_RECEIPT",
        "CHART_LORENTZ_H3_RECEIPT",
        CAP_NORMAL_H3_CHART_RECEIPT,
    ) or _ladder_passed(ladder, "R3") or bool(
        _truthy(paper_chart, "PAPER_THEOREM_3D_BULK_CHART_RECEIPT")
        or _truthy(paper_chart, "paper_theorem_3d_bulk_chart_receipt")
        or _truthy(conformal_chart, "conformal_h3_spatial_chart_receipt")
        or _truthy(cap_normal_h3_chart, CAP_NORMAL_H3_CHART_RECEIPT)
    )
    h3_response_declaration_diagnostic = _truthy_any(
        emergence,
        "H3_RESPONSE_CANDIDATE_RECEIPT",
        "H3_RESPONSE_CONTROL_SEPARATION_RECEIPT",
        "modular_response_h3_candidate_receipt",
    ) or _ladder_passed(ladder, "R4")
    h3_object_preview_declaration_diagnostic = _truthy_any(
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
    object_population_declaration_diagnostic = _truthy_any(
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
    ) or _truthy_any(
        modular_response_h3_localization,
        MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT,
        "h3_modular_response_localization_receipt",
        "H3LOC",
    )
    # Promotion gates come only from the theorem contract recomputed above.
    # T308/T309/T310 each rerun their verifier from primitive source fields and
    # explicitly ignore mutable precomputed receipt JSON.
    record_commit = _contract_stage_passed(
        finite_contract_report,
        "L1_observer_record_algebra",
    )
    bw_kms = _contract_stage_passed(
        finite_contract_report,
        "T308_finite_cap_bw_certificate",
    )
    chart = _contract_stage_passed(
        finite_contract_report,
        "T309_cap_normal_h3_chart",
    )
    h3_response = _contract_stage_passed(
        finite_contract_report,
        "T310_modular_response_h3_localization",
    )
    h3_object_preview = h3_response
    object_nonboundary_population = h3_response
    theorem_assisted_chart_preview = bool(
        finite_consensus_theorem
        and record_commit
        and chart
        and bw_kms
        and h3_response
        and (h3_object_preview or object_nonboundary_population)
    )
    theorem_assisted_nonboundary_population = bool(
        finite_consensus_theorem
        and record_commit
        and chart
        and bw_kms
        and h3_response
        and object_nonboundary_population
    )
    observer_modular_experience_written = bool(observer_modular_experience)
    observer_modular_time = _truthy(observer_modular_experience, "observer_modular_time_receipt")
    observer_facing_h3_chart = bool(chart and bw_kms and h3_response)
    observer_history_component_gates = {
        "observer_modular_experience_written": observer_modular_experience_written,
        "observer_modular_time_receipt": observer_modular_time,
        "semantic_history_receipt": _observer_report_gate(
            observer_modular_experience,
            "semantic_history_receipt",
        ),
        "observer_clock_naturality_receipt": _observer_report_gate(
            observer_modular_experience,
            "observer_clock_naturality_receipt",
        ),
        "observer_registry_descent_receipt": _observer_report_gate(
            observer_modular_experience,
            "observer_registry_descent_receipt",
        ),
        "state_preserving_observer_algebra_receipt": _observer_report_gate(
            observer_modular_experience,
            "state_preserving_observer_algebra_receipt",
        ),
        "support_cap_chart_naturality_receipt": _observer_report_gate(
            observer_modular_experience,
            "support_cap_chart_naturality_receipt",
        ),
        "observer_facing_h3_chart_receipt": observer_facing_h3_chart,
    }
    observer_experienced_3p1d_history = all(observer_history_component_gates.values())
    observer_facing_3p1d_experience = observer_experienced_3p1d_history
    observer_facing_populated_h3_experience = bool(
        _truthy(observer_modular_experience, "observer_facing_populated_h3_experience_receipt")
        and observer_experienced_3p1d_history
        and object_nonboundary_population
    )
    observer_component_gates = {
        "observer_modular_experience_written": observer_modular_experience_written,
        "observer_modular_time_receipt": observer_modular_time,
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
    strict_neutral_object_declaration_diagnostic = _truthy_any(
        strict_neutral_object,
        STRICT_NEUTRAL_OBJECT_BULK_RECEIPT,
        "strict_neutral_object_bulk",
    )
    strict_neutral_object_validation = _independently_replayed_strict_neutral_object(
        root,
        strict_neutral_object,
    )
    strict_neutral_object_bulk = _truthy(
        strict_neutral_object_validation,
        "passed",
    )
    strict_neutral_validation = _independently_replayed_strict_neutral(
        root,
        strict_neutral_report,
        prime_rank_refinement,
    )
    strict_neutral_quotient_metric = _truthy(
        strict_neutral_validation,
        "quotient_metric_passed",
    )
    strict_neutral_bulk = _truthy(
        strict_neutral_validation,
        "strict_neutral_bulk_passed",
    )
    prime_geometric_quotient_3d = _truthy_any(
        prime_rank_sweep,
        "PRIME_GEOMETRIC_QUOTIENT_3D_DIAGNOSTIC_RECEIPT",
        "prime_geometric_quotient_3d_diagnostic_receipt",
    )
    prime_geometric_spatial_3d = _truthy(
        prime_rank_sweep,
        "prime_geometric_spatial_3d_candidate_receipt",
    )
    prime_geometric_rank3_refinement = _truthy(
        prime_rank_refinement,
        "control_quotient_rank3_refinement_candidate_receipt",
    )
    prime_geometric_strict_refinement = _truthy(
        prime_rank_refinement,
        "strict_neutral_bulk_refinement_receipt",
    )
    screen_cmb = bool(
        _truthy(emergence, "SCREEN_PROXY_CMB_RECEIPT")
        or _ladder_passed(ladder, "R7")
        or cl
        or cmb_lite
        or _truthy(physical_cmb_output, "PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT")
    )
    physical_cmb_stage1_input = _truthy(
        physical_cmb_input,
        "PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT",
    ) or _truthy(
        physical_cmb_input_validation,
        "PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT",
    )
    physical_cmb_stage2_frozen_likelihood = bool(
        _truthy(frozen_transfer_likelihood, "FROZEN_TRANSFER_LIKELIHOOD_CLOSURE_RECEIPT")
        and _truthy(frozen_transfer_likelihood, "FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT")
        and _truthy(frozen_transfer_likelihood, "FROZEN_PHYSICAL_SPECTRUM_RECEIPT")
    )
    physical_cmb_stage3_output = _truthy_any(
        emergence,
        "physical_cmb_prediction",
    ) or _truthy_any(
        cmb_lite,
        "physical_cmb_prediction",
    ) or _truthy_any(
        cl,
        "physical_cmb_prediction",
    ) or _truthy_any(
        physical_cmb_frontier,
        "physical_cmb_prediction_receipt",
    ) or _truthy_any(
        physical_cmb_output,
        "PHYSICAL_CMB_PREDICTION_RECEIPT",
    )
    physical_cmb = bool(
        physical_cmb_stage1_input
        and physical_cmb_stage2_frozen_likelihood
        and physical_cmb_stage3_output
    )
    proto_particle = particle_contract.get(BULK_WORLDLINE_PRECURSOR_RECEIPT) is True
    classical_carrier = particle_contract.get(CLASSICAL_CARRIER_MODE_RECEIPT) is True
    quantum_particle = particle_contract.get(QUANTUM_PARTICLE_RECEIPT) is True
    colored_deconfinement = particle_contract.get(COLORED_DECONFINEMENT_RECEIPT) is True
    production_particle = particle_contract.get(PRODUCTION_PARTICLE_MATTER_RECEIPT) is True
    scale_operator = _truthy(scale_compressed, "scale_compressed_operator_receipt")
    scale_round_trace = _truthy(scale_compressed, "repair_round_trace_receipt")
    scale_h3 = _as_dict(scale_compressed.get("h3_preview"))
    scale_cmb_params = _as_dict(scale_compressed.get("cmb_parameter_readouts"))
    scale_h3_preview = bool(
        scale_operator
        and _truthy(scale_h3, "populated_h3_preview_receipt")
        and _truthy(scale_h3, "cap_profile_receipt")
    )
    scale_particle_preview = _truthy(
        scale_compressed_particle,
        "particle_preview_receipt",
    ) or _truthy(
        _as_dict(scale_compressed.get("particle_preview")),
        "particle_preview_receipt",
    )
    scale_compressed_measurement_cmb = bool(
        _truthy(scale_compressed_cmb, "measurement_comparable_cmb_curve")
        and _truthy(scale_compressed_cmb, "screen_camb_transfer_receipt")
    )
    scale_physical_cmb = _truthy(
        scale_compressed,
        "physical_cmb_prediction",
    ) or _truthy(
        scale_compressed_cmb,
        "physical_cmb_prediction",
    )
    screen_cmb = bool(screen_cmb or scale_compressed_measurement_cmb)
    physical_cmb = bool(physical_cmb or (scale_physical_cmb and physical_cmb_stage1_input and physical_cmb_stage2_frozen_likelihood))

    finite_lorentz_contract = _truthy_any(
        finite_contract_report,
        OPH_LORENTZ_THEOREM_FINITE_CONTRACT_RECEIPT,
        "finite_lorentz_theorem_contract_receipt",
    )
    paper_faithful_observer_spacetime = _truthy(
        finite_contract_report,
        "paper_faithful_observer_spacetime_emergence_receipt",
    )
    paper_faithful_populated_h3 = _truthy(
        finite_contract_report,
        "paper_faithful_populated_h3_observer_experience_receipt",
    )
    paper_faithful_consensus_bulk = _truthy(
        finite_contract_report,
        "paper_faithful_consensus_bulk_emergence_receipt",
    )
    paper_geometric_branch_contract = _truthy(
        finite_contract_report,
        "paper_geometric_branch_lorentz_contract_receipt",
    )
    paper_geometric_branch_observer_spacetime = _truthy(
        finite_contract_report,
        "paper_geometric_branch_observer_spacetime_emergence_receipt",
    )
    paper_geometric_branch_populated_h3 = _truthy(
        finite_contract_report,
        "paper_geometric_branch_populated_h3_observer_experience_receipt",
    )
    paper_geometric_branch_consensus_bulk = _truthy(
        finite_contract_report,
        "paper_geometric_branch_consensus_bulk_emergence_receipt",
    )
    einstein_bridge_dependency_discharge = _truthy_any(
        einstein_bridge_manifest,
        EINSTEIN_BRIDGE_DEPENDENCY_DISCHARGE_RECEIPT,
        "theorem_e0_dependency_discharge_receipt",
    )
    einstein_bridge_run_receipts = _truthy_any(
        einstein_bridge_manifest,
        EINSTEIN_BRIDGE_RUN_RECEIPTS_RECEIPT,
        "einstein_bridge_run_receipts_receipt",
        "all_required_receipts_theorem_tagged",
    )
    manifest_einstein_branch_entry = _truthy_any(
        einstein_bridge_manifest,
        OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_RECEIPT,
        EINSTEIN_BRANCH_ENTRY_RECEIPT,
        "einstein_branch_entry_contract_receipt",
        "einstein_branch_entry_receipt",
    )
    legacy_einstein_branch_entry = _truthy_any(
        finite_contract_report,
        OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_RECEIPT,
        EINSTEIN_BRANCH_ENTRY_RECEIPT,
        "einstein_branch_entry_contract_receipt",
    )
    einstein_branch_entry = (
        manifest_einstein_branch_entry if use_einstein_bridge_manifest else legacy_einstein_branch_entry
    )
    einstein_branch_blockers = list(
        (
            einstein_bridge_manifest.get("einstein_branch_entry_blockers")
            or einstein_bridge_manifest.get("blockers")
            or []
        )
        if use_einstein_bridge_manifest
        else finite_contract_report.get("einstein_branch_entry_blockers")
        or einstein_branch_report.get("blockers")
        or [
            "E0_einstein_branch_entry_umbrella",
            "E1_null_generator_stress_charge",
            "E2_fixed_cap_entropy_stationarity",
            "E3_small_ball_area_bridge",
            "E4_all_timelike_tensor_upgrade",
            "E5_lambda_constancy_conservation",
            "E6_newton_coupling_forbidden_input_audit",
        ]
    )
    production_gravity = False
    physical_gravity_prediction = False

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
            "Paper-side conformal/cap-normal route Conf+(S2) -> SO+(3,1) -> H3 spatial chart is instantiated.",
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
            required_receipts=[
                "STRICT_NEUTRAL_OBJECT_BULK_CANDIDATE_RECEIPT",
                "STRICT_NEUTRAL_QUOTIENT_METRIC_RECEIPT",
                "STRICT_NEUTRAL_THIRD_PERSON_BULK_RECEIPT",
            ],
        ),
        "T6_strict_neutral_third_person_bulk": _tier(
            STRICT_NEUTRAL_BULK_RECEIPT,
            strict_neutral_bulk,
            "Legacy alias for T6 chart-blind strict neutral quotient bulk; separate from observer-facing H3 consensus bulk.",
            canonical_tier="T6_chart_blind_strict_neutral_quotient_bulk",
        ),
        "T6_object_strict_neutral_bulk": _tier(
            "STRICT_NEUTRAL_OBJECT_BULK_CANDIDATE_RECEIPT",
            strict_neutral_object_bulk,
            "Neutral object extraction and held-out latent geometry selection establish an object-bulk candidate only; it does not promote full strict neutral third-person bulk.",
            legacy_receipt_name=STRICT_NEUTRAL_OBJECT_BULK_RECEIPT,
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
        "P0_bulk_worldline_precursor": _tier(
            BULK_WORLDLINE_PRECURSOR_RECEIPT,
            proto_particle,
            "Localized, transported, controlled bulk proto-worldline; not yet a particle claim.",
        ),
        "P1a_classical_carrier_mode": _tier(
            CLASSICAL_CARRIER_MODE_RECEIPT,
            classical_carrier,
            "Action-level physical carrier mode with positive kinetic term, reduced Hamiltonian, and wave kernel.",
        ),
        "P1b_quantum_particle": _tier(
            QUANTUM_PARTICLE_RECEIPT,
            quantum_particle,
            "Positive-energy physical Hilbert/spectral/asymptotic particle receipt, including deconfinement for color.",
        ),
        "P1_production_particle_matter": _tier(
            PRODUCTION_PARTICLE_MATTER_RECEIPT,
            production_particle,
            "P1 is recomputed as P0 AND classical carrier mode AND quantum particle, with colored deconfinement when applicable.",
            blockers=particle_contract.get("blockers", []),
            required_receipts=[
                BULK_WORLDLINE_PRECURSOR_RECEIPT,
                CLASSICAL_CARRIER_MODE_RECEIPT,
                QUANTUM_PARTICLE_RECEIPT,
                COLORED_DECONFINEMENT_RECEIPT,
            ],
        ),
        "T7_production_particles": _tier(
            PRODUCTION_PARTICLE_MATTER_RECEIPT,
            production_particle,
            "Legacy alias for P1; producer top-level particle booleans are ignored.",
            canonical_tier="P1_production_particle_matter",
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
            "Legacy finite Lorentz contract audits observer-record modular diagnostics; issue #308 BW3 is the stricter finite cap-normal certificate receipt.",
            blockers=finite_contract_report.get("primary_blockers", []),
        ),
        "L_branch_paper_geometric_lorentz_contract": _tier(
            "PAPER_GEOMETRIC_BRANCH_LORENTZ_CONTRACT_RECEIPT",
            paper_geometric_branch_contract,
            "Paper-geometric branch contract uses the declared BW-framed theorem chart, without promoting the endogenous finite clock diagnostic or a BW3 finite certificate.",
            blockers=finite_contract_report.get("paper_geometric_branch_primary_blockers", []),
        ),
        "B_full_paper_faithful_observer_spacetime": _tier(
            "PAPER_FAITHFUL_OBSERVER_SPACETIME_EMERGENCE_RECEIPT",
            paper_faithful_observer_spacetime,
            "Observer-local modular time plus H3 spatial chart and H3 response, gated by the finite theorem contract.",
            blockers=finite_contract_report.get("primary_blockers", []),
        ),
        "B_branch_paper_geometric_observer_spacetime": _tier(
            "PAPER_GEOMETRIC_BRANCH_OBSERVER_SPACETIME_EMERGENCE_RECEIPT",
            paper_geometric_branch_observer_spacetime,
            "Observer-local modular time plus H3 spatial chart and H3 response, gated by the paper-geometric KMS branch contract.",
            blockers=finite_contract_report.get("paper_geometric_branch_primary_blockers", []),
        ),
        "B_populated_h3_observer_experience": _tier(
            "PAPER_FAITHFUL_POPULATED_H3_OBSERVER_EXPERIENCE_RECEIPT",
            paper_faithful_populated_h3,
            "Observer-facing H3 spacetime plus controlled object population in that chart.",
            blockers=finite_contract_report.get("primary_blockers", []),
        ),
        "B_branch_paper_geometric_populated_h3": _tier(
            "PAPER_GEOMETRIC_BRANCH_POPULATED_H3_OBSERVER_EXPERIENCE_RECEIPT",
            paper_geometric_branch_populated_h3,
            "Observer-facing H3 spacetime plus controlled object population, using the paper-geometric KMS branch contract.",
            blockers=finite_contract_report.get("paper_geometric_branch_primary_blockers", []),
        ),
        "B_full_paper_faithful_consensus_bulk": _tier(
            "PAPER_FAITHFUL_CONSENSUS_BULK_EMERGENCE_RECEIPT",
            paper_faithful_consensus_bulk,
            "Paper-faithful observer-facing consensus 3D bulk: observer spacetime emergence plus populated H3 object records.",
            blockers=finite_contract_report.get("primary_blockers", []),
        ),
        "B_branch_paper_geometric_consensus_bulk": _tier(
            "PAPER_GEOMETRIC_BRANCH_CONSENSUS_BULK_EMERGENCE_RECEIPT",
            paper_geometric_branch_consensus_bulk,
            "Paper-geometric observer-facing consensus 3D bulk: observer spacetime plus populated H3 object records under the declared KMS branch.",
            blockers=finite_contract_report.get("paper_geometric_branch_primary_blockers", []),
        ),
        "E0_einstein_branch_entry_contract": _tier(
            OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_RECEIPT,
            einstein_branch_entry,
            "E0 Einstein bridge manifest: OPH5 theorem provenance plus theorem-tagged run sidecar receipts for geometry, stress, entropy, bounded interval, small ball, remainder, timelike coverage, Lambda, Newton audit, and residual checks.",
            blockers=einstein_branch_blockers,
        ),
        "G2_production_gravity": _tier(
            PRODUCTION_GRAVITY_RECEIPT,
            production_gravity,
            "Production gravity remains closed unless the Einstein branch-entry contract and a production source/stress bridge both pass.",
            blockers=einstein_branch_blockers
            if not einstein_branch_entry
            else ["production_source_stress_bridge_missing"],
            required_receipts=[
                OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_RECEIPT,
                "production_source_stress_bridge_receipt",
                PRODUCTION_PARTICLE_MATTER_RECEIPT,
            ],
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
        "finite_consensus_declaration_diagnostic": finite_consensus_declaration_diagnostic,
        "finite_consensus_primitive_validation": finite_consensus_validation,
        "chart_level_3p1_lorentz_kinematics_established": bool(chart and bw_kms),
        CAP_NORMAL_H3_CHART_RECEIPT: chart,
        "cap_normal_h3_chart_receipt": chart,
        "cap_normal_h3_chart_terminal_status": _as_dict(
            finite_contract_report.get("issue_309_cap_normal_h3_chart")
        ).get("terminal_status"),
        MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT: h3_response,
        "h3_modular_response_localization_receipt": h3_response,
        "h3_modular_response_localization_terminal_status": _as_dict(
            finite_contract_report.get("issue_310_modular_response_h3_localization")
        ).get("terminal_status"),
        "paper_route_lorentz_h3_chart_established": bool(chart and bw_kms),
        OPH_LORENTZ_THEOREM_FINITE_CONTRACT_RECEIPT: finite_lorentz_contract,
        "finite_lorentz_theorem_contract_receipt": finite_lorentz_contract,
        "paper_faithful_observer_spacetime_emergence_receipt": paper_faithful_observer_spacetime,
        "paper_faithful_populated_h3_observer_experience_receipt": paper_faithful_populated_h3,
        "observer_facing_consensus_3d_bulk_emergence_receipt": paper_faithful_consensus_bulk,
        "paper_faithful_consensus_bulk_emergence_receipt": paper_faithful_consensus_bulk,
        "paper_geometric_branch_lorentz_contract_receipt": paper_geometric_branch_contract,
        "paper_geometric_branch_observer_spacetime_emergence_receipt": (
            paper_geometric_branch_observer_spacetime
        ),
        "paper_geometric_branch_populated_h3_observer_experience_receipt": (
            paper_geometric_branch_populated_h3
        ),
        "paper_geometric_branch_consensus_bulk_emergence_receipt": paper_geometric_branch_consensus_bulk,
        OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_RECEIPT: einstein_branch_entry,
        EINSTEIN_BRANCH_ENTRY_RECEIPT: einstein_branch_entry,
        "einstein_branch_entry_contract_receipt": einstein_branch_entry,
        "einstein_branch_entry_blockers": einstein_branch_blockers,
        PRODUCTION_GRAVITY_RECEIPT: production_gravity,
        "production_gravity_receipt": production_gravity,
        PHYSICAL_GRAVITY_PREDICTION_RECEIPT: physical_gravity_prediction,
        "physical_gravity_prediction": physical_gravity_prediction,
        "simulation_matches_observer_facing_oph_spacetime_bulk_prediction_receipt": (
            paper_geometric_branch_consensus_bulk
        ),
        "simulation_matches_full_oph_spacetime_bulk_prediction_receipt": paper_faithful_consensus_bulk,
        "finite_theorem_contract_summary": {
            "written": bool(persisted_finite_contract_report),
            "recomputed_in_memory": finite_contract_report.get("mode")
            == "finite_oph_theorem_contract_audit_v1",
            "recomputation_error": finite_contract_report.get("recomputation_error"),
            "blockers": finite_contract_report.get("blockers", []),
            "primary_blockers": finite_contract_report.get("primary_blockers", []),
            "paper_geometric_branch_blockers": finite_contract_report.get(
                "paper_geometric_branch_blockers", []
            ),
            "paper_geometric_branch_primary_blockers": finite_contract_report.get(
                "paper_geometric_branch_primary_blockers", []
            ),
            "chart_blind_strict_neutral_blockers": finite_contract_report.get(
                "chart_blind_strict_neutral_blockers", []
            ),
            "strict_neutral_blockers": finite_contract_report.get("strict_neutral_blockers", []),
            "einstein_branch_entry_contract_receipt": einstein_branch_entry,
            "einstein_branch_entry_blockers": einstein_branch_blockers,
            "all_stage_blockers": finite_contract_report.get("all_stage_blockers", []),
            "stages": finite_contract_report.get("stages", {}),
            "claim_boundary": finite_contract_report.get("claim_boundary"),
        },
        "persisted_finite_theorem_contract_diagnostic": {
            "written": bool(persisted_finite_contract_report),
            "mode": persisted_finite_contract_report.get("mode"),
            "finite_lorentz_contract_claim": persisted_finite_contract_report.get(
                OPH_LORENTZ_THEOREM_FINITE_CONTRACT_RECEIPT
            )
            or persisted_finite_contract_report.get(
                "finite_lorentz_theorem_contract_receipt"
            ),
            "observer_spacetime_claim": persisted_finite_contract_report.get(
                "paper_faithful_observer_spacetime_emergence_receipt"
            ),
            "consensus_bulk_claim": persisted_finite_contract_report.get(
                "paper_faithful_consensus_bulk_emergence_receipt"
            ),
            "claim_boundary": (
                "Persisted theorem-contract outputs are diagnostic only. L/B promotion "
                "uses the read-only in-memory recomputation above."
            ),
        },
        "theorem_assisted_source_validation": {
            "record_commit": record_commit,
            "finite_cap_bw_certificate": bw_kms,
            "cap_normal_h3_chart": chart,
            "record_conditioned_h3_localization": h3_response,
            "object_population_from_localization": object_nonboundary_population,
            "ignored_declaration_diagnostics": {
                "record_commit": record_commit_declaration_diagnostic,
                "bw_kms": bw_kms_declaration_diagnostic,
                "chart": chart_declaration_diagnostic,
                "h3_response": h3_response_declaration_diagnostic,
                "h3_object_preview": h3_object_preview_declaration_diagnostic,
                "object_population": object_population_declaration_diagnostic,
            },
            "claim_boundary": (
                "The theorem-assisted lane uses only in-memory T308/T309/T310 primitive "
                "verifier results; emergence, ladder, and precomputed report booleans are "
                "retained as diagnostics and cannot promote."
            ),
        },
        "theorem_assisted_h3_object_preview_established": theorem_assisted_chart_preview,
        "theorem_assisted_h3_nonboundary_population_established": theorem_assisted_nonboundary_population,
        "theorem_assisted_h3_populated_chart_established": theorem_assisted_nonboundary_population,
        "theorem_assisted_observer_facing_h3_population": theorem_assisted_nonboundary_population,
        THEOREM_ASSISTED_H3_OBJECT_POPULATION_RECEIPT: theorem_assisted_nonboundary_population,
        "observer_facing_h3_object_population_receipt": theorem_assisted_nonboundary_population,
        "observer_modular_time_receipt": observer_modular_time,
        "OBSERVER_FACING_H3_CHART_RECEIPT": observer_facing_h3_chart,
        "observer_facing_h3_chart_receipt": observer_facing_h3_chart,
        "OBSERVER_EXPERIENCED_3P1D_HISTORY_RECEIPT": observer_experienced_3p1d_history,
        "observer_experienced_3p1d_history_receipt": observer_experienced_3p1d_history,
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
        "strict_neutral_object_bulk_candidate_receipt": strict_neutral_object_bulk,
        "STRICT_NEUTRAL_OBJECT_BULK_CANDIDATE_RECEIPT": strict_neutral_object_bulk,
        "STRICT_NEUTRAL_QUOTIENT_METRIC_RECEIPT": strict_neutral_quotient_metric,
        "strict_neutral_quotient_metric_receipt": strict_neutral_quotient_metric,
        "STRICT_NEUTRAL_THIRD_PERSON_BULK_RECEIPT": strict_neutral_bulk,
        "strict_neutral_third_person_bulk_receipt": strict_neutral_bulk,
        "strict_neutral_bulk_contract_receipt": strict_neutral_bulk,
        STRICT_NEUTRAL_OBJECT_BULK_RECEIPT: strict_neutral_object_bulk,
        STRICT_NEUTRAL_BULK_RECEIPT: strict_neutral_bulk,
        "strict_neutral_object_source_validation": strict_neutral_object_validation,
        "strict_neutral_source_validation": strict_neutral_validation,
        "strict_neutral_derived_report_diagnostic": {
            "written": bool(strict_neutral_report),
            "persisted_top_level_claim": strict_neutral_report.get("strict_neutral_bulk"),
            "persisted_typed_bulk_candidate": _strict_neutral_bulk_passed(
                strict_neutral_report,
                prime_rank_refinement,
            ),
            "persisted_typed_quotient_metric_candidate": (
                _strict_neutral_quotient_metric_passed(strict_neutral_report)
            ),
            "persisted_blockers": strict_neutral_report.get("blockers", []),
            "claim_boundary": (
                "Persisted strict-neutral booleans are retained as diagnostics only; "
                "promotion uses the source replay validation above."
            ),
        },
        "prime_geometric_quotient_3d_diagnostic": prime_geometric_quotient_3d,
        "prime_geometric_spatial_3d_candidate": prime_geometric_spatial_3d,
        "prime_geometric_rank3_refinement_candidate": prime_geometric_rank3_refinement,
        CONTROL_RESIDUALIZED_RANK3_CANDIDATE_RECEIPT: prime_geometric_rank3_refinement,
        "prime_geometric_rank3_refinement_strict_neutral": prime_geometric_strict_refinement,
        "bulk_3d_established_theorem_assisted": theorem_assisted_nonboundary_population,
        "bulk_3d_established_observer_facing_consensus": paper_faithful_consensus_bulk,
        "bulk_3d_established_paper_geometric_branch_observer_facing_consensus": (
            paper_geometric_branch_consensus_bulk
        ),
        "bulk_3d_established_strict": strict_neutral_bulk,
        "bulk_3d_established_chart_blind_strict_neutral": strict_neutral_bulk,
        "screen_cmb_proxy_available": screen_cmb,
        "physical_cmb_prediction": physical_cmb,
        "CMB1_SOURCE_INPUT_CONTRACT": physical_cmb_stage1_input,
        "CMB1_FROZEN_TRANSFER_LIKELIHOOD_CLOSURE": physical_cmb_stage2_frozen_likelihood,
        "CMB2_PHYSICAL_CMB_PREDICTION_RECEIPT": physical_cmb,
        "physical_cmb_staged_contract": {
            "CMB1_SOURCE_INPUT_CONTRACT": physical_cmb_stage1_input,
            "CMB1_FROZEN_TRANSFER_LIKELIHOOD_CLOSURE": physical_cmb_stage2_frozen_likelihood,
            "CMB2_OUTPUT_ARTIFACT_PRESENT": physical_cmb_stage3_output,
            "CMB2_PHYSICAL_CMB_PREDICTION_RECEIPT": physical_cmb,
            "blockers": (
                ([] if physical_cmb_stage1_input else ["CMB1_SOURCE_INPUT_CONTRACT_missing"])
                + (
                    []
                    if physical_cmb_stage2_frozen_likelihood
                    else ["CMB1_FROZEN_TRANSFER_LIKELIHOOD_CLOSURE_missing"]
                )
                + ([] if physical_cmb_stage3_output else ["CMB2_OUTPUT_ARTIFACT_missing"])
            ),
            "claim_boundary": (
                "Physical CMB promotion is staged: source input contract first, frozen transfer/"
                "likelihood closure second, final output artifact last. Caller booleans alone do not promote."
            ),
        },
        PROTO_PARTICLE_RECEIPT: proto_particle,
        BULK_WORLDLINE_PRECURSOR_RECEIPT: proto_particle,
        "bulk_worldline_precursor_receipt": proto_particle,
        CLASSICAL_CARRIER_MODE_RECEIPT: classical_carrier,
        "classical_carrier_mode_receipt": classical_carrier,
        QUANTUM_PARTICLE_RECEIPT: quantum_particle,
        "quantum_particle_receipt": quantum_particle,
        COLORED_DECONFINEMENT_RECEIPT: colored_deconfinement,
        "colored_deconfinement_receipt": colored_deconfinement,
        PRODUCTION_PARTICLE_MATTER_RECEIPT: production_particle,
        "production_particle_matter_receipt": production_particle,
        "particle_promotion_contract_summary": {
            "evidence_file": particle_contract.get("evidence_file"),
            "evidence_sha256": particle_contract.get("evidence_sha256"),
            "candidate_id": particle_contract.get("candidate_id"),
            "candidate_kind": particle_contract.get("candidate_kind"),
            "schema_gates": particle_contract.get("schema_gates", {}),
            "provenance": particle_contract.get("provenance", {}),
            "lanes": particle_contract.get("lanes", {}),
            "ignored_caller_promotion_fields": particle_contract.get(
                "ignored_caller_promotion_fields", {}
            ),
            "ignored_legacy_producer_fields": {
                "emergence_status_report.particle_matter_receipt": emergence.get(
                    "particle_matter_receipt"
                ),
                "particle_likeness_report.particle_matter_receipt": particle.get(
                    "particle_matter_receipt"
                ),
                "controlled_defect_particle_assay_report.physical_particle_emergence": (
                    controlled_particle.get("physical_particle_emergence")
                ),
            },
            "blockers": particle_contract.get("blockers", []),
            "claim_boundary": particle_contract.get("claim_boundary"),
        },
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
            "observer_facing_h3_chart_receipt": observer_facing_h3_chart,
            "observer_experienced_3p1d_history_receipt": observer_experienced_3p1d_history,
            "observer_facing_3p1d_h3_experience_receipt": observer_facing_3p1d_experience,
            "observer_facing_populated_h3_experience_receipt": observer_facing_populated_h3_experience,
            "observer_h3_object_population_receipt": object_nonboundary_population,
            "observer_count": observer_modular_experience.get("observer_count"),
            "observer_relative_time_count": observer_modular_experience.get("observer_relative_time_count"),
            "blockers": observer_blockers,
            "populated_h3_experience_blockers": observer_populated_h3_blockers,
            "component_gates": observer_component_gates,
            "history_component_gates": observer_history_component_gates,
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
                or _as_dict(scale_compressed.get("particle_preview")).get("particle_worldline_count")
            ),
            "camb_ir_shape_correlation": (
                _as_dict(_as_dict(scale_compressed_cmb.get("comparison")).get("scale_compressed_ir_kernel")).get(
                    "shape_correlation"
                )
            ),
            "camb_ir_normalized_rmse": (
                _as_dict(_as_dict(scale_compressed_cmb.get("comparison")).get("scale_compressed_ir_kernel")).get(
                    "normalized_rmse"
                )
            ),
            "camb_ir_chi2_per_bin": (
                _as_dict(_as_dict(scale_compressed_cmb.get("comparison")).get("scale_compressed_ir_kernel")).get(
                    "best_fit_column_chi2_per_bin"
                )
            ),
            "physical_cmb_prediction": scale_physical_cmb,
            "strict_neutral_bulk": _truthy(
                scale_compressed,
                "strict_neutral_bulk",
            ) or _truthy(
                scale_h3,
                "strict_neutral_third_person_bulk_established",
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
            or _as_dict(conformal_chart.get("lorentz_algebra_report")).get("h3_spatial_dimension_from_boost_orbit"),
            "h3_chart_spatial_dimension": paper_chart.get("h3_chart_spatial_dimension")
            or _as_dict(conformal_chart.get("h3_chart_report")).get("spatial_dimension"),
            "finite_point_cloud_dimension_estimator_used": paper_chart.get(
                "finite_point_cloud_dimension_estimator_used", False
            ),
        },
        "prime_geometric_rank_sweep_summary": {
            "written": bool(prime_rank_sweep),
            "diagnostic_receipt": prime_geometric_quotient_3d,
            "spatial_3d_candidate_receipt": prime_geometric_spatial_3d,
            "strict_neutral_candidate_receipt": _truthy(
                prime_rank_sweep,
                "prime_geometric_strict_neutral_candidate_receipt",
            ),
            "dimension_3d_window_count": prime_rank_sweep.get("dimension_3d_window_count"),
            "coordinate_dimension_3d_window_count": prime_rank_sweep.get(
                "coordinate_dimension_3d_window_count"
            ),
            "best_3d_rank": _as_dict(prime_rank_sweep.get("best_3d_dimension_row")).get("rank"),
            "coordinate_best_3d_rank": (
                _as_dict(prime_rank_sweep.get("coordinate_best_3d_dimension_row")).get("rank")
            ),
            "regulator_control_quotient_is_negative_control": (
                _as_dict(prime_rank_sweep.get("regulator_control_quotient_lane")).get(
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
            "persisted_declaration_diagnostic": strict_neutral_object_declaration_diagnostic,
            "source_validation": strict_neutral_object_validation,
            "object_count": strict_neutral_object.get("object_count"),
            "selected_model": _as_dict(strict_neutral_object.get("latent_geometry_selection")).get("selected_model"),
            "h3_selected": _as_dict(strict_neutral_object.get("latent_geometry_selection")).get("h3_selected"),
            "median_dimension_estimate": _as_dict(strict_neutral_object.get("dimension")).get("median_dimension_estimate"),
            "blockers": strict_neutral_object.get("blockers", []),
            "claim_boundary": (
                "Strict neutral object-bulk report uses observer-visible object histories only; it is separate "
                "from theorem-assisted H3 chart population."
            ),
        },
        "methodological_audit_blockers": [
            {
                "component": "neutral_model_selection",
                "blocker": "pair_holdout_scored_after_full_distance_matrix_fit",
                "promotion_impact": "strict_neutral_T6_remains_closed",
                "detail": (
                    "neutral_model_selection currently fits embeddings on the complete sampled "
                    "distance matrix and only then scores a sampled pair subset called heldout. "
                    "It is not independent out-of-sample validation."
                ),
            }
        ],
        "einstein_branch_entry_summary": {
            "written": bool(finite_contract_report) or bool(einstein_branch_report) or use_einstein_bridge_manifest,
            "receipt": einstein_branch_entry,
            "manifest_written": use_einstein_bridge_manifest,
            "manifest_receipt": _truthy(
                einstein_bridge_manifest,
                OPH_EINSTEIN_BRIDGE_MANIFEST_RECEIPT,
            ),
            "theorem_e0_dependency_discharge_receipt": einstein_bridge_dependency_discharge,
            "einstein_bridge_run_receipts_receipt": einstein_bridge_run_receipts,
            "claim_tier": einstein_bridge_manifest.get("claim_tier"),
            "legacy_issue": 503,
            "legacy_issue_url": "https://github.com/FloatingPragma/observer-patch-holography/issues/503",
            "legacy_source_report_written": bool(einstein_branch_report),
            "blockers": einstein_branch_blockers,
            "child_gates": (
                einstein_bridge_manifest.get("einstein_branch_entry_child_gates", {})
                if use_einstein_bridge_manifest
                else finite_contract_report.get("einstein_branch_entry_child_gates", {})
            ),
            "provenance_tags": dict(einstein_bridge_manifest.get("provenanceTags") or {}),
            "required_receipt_files": list(einstein_bridge_manifest.get("requiredReceiptFiles") or []),
            "claim_boundary": (
                "The E0 paper theorem discharges the OPH5 recovered-core bridge dependencies. A concrete "
                "run still needs the theorem-tagged sidecar receipts in the Einstein bridge manifest. Without "
                "them, curved-spacetime and defect displays remain diagnostics, not production gravity."
            ),
        },
        "paper_alignment": {
            "screen_role": "S2 is the observer-facing cap/symmetry chart, not a raw point-cloud proof of dimension.",
            "lorentz_route": (
                "support-visible BW-framed cap automorphism with s=2*pi*t gives "
                "Conf+(S2) ~= SO+(3,1); finite BW3 evidence is the separate issue #308 BWRec audit, "
                "issue #309 cap-normal H3 chart evidence is CAP_NORMAL_H3_CHART_RECEIPT, and "
                "issue #310 record-populated H3 localization evidence is "
                "MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT."
            ),
            "spatial_chart": (
                "H3 is the observer-frame homogeneous chart SO+(3,1)/SO(3). The issue #309 receipt "
                "checks q(Omega), n_C, signed incidence, n_gC=Lambda_g n_C, and H3 future-sheet fields. "
                "The issue #310 receipt can populate observer-facing H3 only with certified "
                "localization balls from record-conditioned cap responses, compact-domain, alpha>0, "
                "bounded-error, and Delta_loc controls."
            ),
            "finite_gate": "finite runs must separately show observer records/objects/defects populate that chart under controls.",
            "strict_neutral_route": (
                "chart-blind strict neutral quotient bulk must be reconstructed from neutral observer/object "
                "records without H3/S2/support coordinates and must pass held-out latent-geometry controls."
            ),
            "scale_compressed_branch": (
                "Logical scale compression can expose OPH repair-round/CMB readouts and H3 previews, but it "
                "does not replace strict neutral observer-record reconstruction."
            ),
            "einstein_branch_entry": (
                "The OPH5 Einstein bridge is paper-discharged by E0; the simulator keeps production gravity "
                "closed until the run emits all theorem-tagged bridge sidecar receipts."
            ),
        },
        "claim_boundary": (
            "Tiered OPH proof/readout certificate. C0a/T0 is only finite settling; C0b finite "
            "consensus requires strict theorem-phase replay and may remain false even when final_phi is zero. "
            "L0/T2 is branch replay: the declared BW/KMS 2pi route executes under current controls, "
            "but the full L1-L7 finite Lorentz theorem contract remains false. T3 is the conformal/cap-normal H3 chart route for this run. "
            "T5a is theorem-assisted H3 preview evidence from observer objects. "
            "T5b is stricter non-boundary H3 object population evidence. T5c is a scale-compressed "
            "logical repair-round H3 preview and is intentionally not promoted to T5b/T6. "
            "T6 is stricter chart-blind neutral quotient reconstruction and may remain false even when "
            "the observer-facing H3 consensus-bulk receipt passes. "
            "T6a/T6b are intermediate residualized prime-geometric quotient diagnostics and are not promoted to T6. "
            "T8/T8b are measurement-facing screen/CAMB-transfer data only. T9 is a physical CMB prediction and remains false "
            "until finite OPH kernels feed a Boltzmann/likelihood-ready pipeline. E0/G2 are separate "
            "Einstein/gravity promotion gates: visual curvature, apparent attraction, or H3 object motion "
            "does not become production gravity without the #503 branch-entry contract."
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
    row = {"receipt_name": receipt_name, "passed": _is_true(passed), "description": description}
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


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _ladder_passed(ladder: dict[str, Any], key: str) -> bool:
    return _is_true(_as_dict(_as_dict(ladder.get("receipts")).get(key)).get("passed"))


def _ladder_receipt_passed(ladder: dict[str, Any], receipt_name: str) -> bool:
    receipts = _as_dict(ladder.get("receipts")).values()
    return any(
        _is_true(row.get("passed")) and row.get("receipt_name") == receipt_name
        for row in receipts
        if isinstance(row, dict)
    )


def _truthy(data: dict[str, Any], key: str) -> bool:
    return _is_true(data.get(key))


def _truthy_any(data: dict[str, Any], *keys: str) -> bool:
    return any(_truthy(data, key) for key in keys)


def _independently_recomputed_finite_contract(root: Path) -> dict[str, Any]:
    """Recompute the complete finite theorem contract without writing a report."""

    try:
        from oph_fpe.bulk.theorem_contract import finite_oph_theorem_contract_report

        report = finite_oph_theorem_contract_report(root)
    except Exception as exc:  # pragma: no cover - defensive proof boundary.
        return {
            "mode": "finite_oph_theorem_contract_recomputation_failed",
            "stages": {},
            "blockers": [
                f"finite_theorem_contract_recomputation_failed:{type(exc).__name__}:{exc}"
            ],
            "recomputation_error": f"{type(exc).__name__}:{exc}",
        }
    if not isinstance(report, dict) or report.get("mode") != "finite_oph_theorem_contract_audit_v1":
        return {
            "mode": "finite_oph_theorem_contract_recomputation_failed",
            "stages": {},
            "blockers": ["finite_theorem_contract_recomputation_mode_invalid"],
            "recomputation_error": "invalid_return_mode",
        }
    return report


def _contract_stage_passed(report: dict[str, Any], stage_name: str) -> bool:
    if report.get("mode") != "finite_oph_theorem_contract_audit_v1":
        return False
    stage = _as_dict(_as_dict(report.get("stages")).get(stage_name))
    return _truthy(stage, "passed")


def _finite_consensus_validation_from_contract(
    contract: dict[str, Any],
) -> dict[str, Any]:
    stage = _as_dict(
        _as_dict(contract.get("stages")).get("C0_finite_consensus_theorem")
    )
    details = _as_dict(stage.get("details"))
    blockers = list(details.get("blockers", []))
    if not stage:
        blockers.append("finite_consensus_stage_missing_from_recomputed_contract")
    return {
        "mode": "finite_consensus_primitive_bound_validation_v1",
        "passed": _truthy(stage, "passed"),
        "validation": details,
        "blockers": list(dict.fromkeys(blockers)),
        "source": "read_only_in_memory_finite_theorem_contract_recomputation",
    }


def _independently_replayed_strict_neutral_object(
    root: Path,
    persisted_report: dict[str, Any],
) -> dict[str, Any]:
    """Recompute the neutral-object candidate from hash-bound observer rows."""

    persisted_candidate = _truthy_any(
        persisted_report,
        STRICT_NEUTRAL_OBJECT_BULK_RECEIPT,
        "strict_neutral_object_bulk",
    )
    summary: dict[str, Any] = {
        "mode": "strict_neutral_object_primitive_bound_validation_v1",
        "persisted_candidate": persisted_candidate,
        "replay_attempted": False,
        "source_validation_passed": False,
        "passed": False,
        "blockers": [],
    }
    if not persisted_candidate:
        summary["blockers"] = ["persisted_strict_neutral_object_candidate_not_true"]
        return summary

    manifest_path = root / "strict_neutral_object_source_manifest.json"
    manifest = _read_json(manifest_path)
    blockers: list[str] = []
    if not manifest_path.is_file() or not manifest:
        blockers.append("strict_neutral_object_source_manifest_missing")
    if manifest.get("schema") != "strict_neutral_object_bulk_source_v1":
        blockers.append("strict_neutral_object_source_manifest_schema_invalid")
    if _as_dict(persisted_report.get("source_artifact")) != manifest:
        blockers.append("strict_neutral_object_nested_source_manifest_mismatch")

    observer_path = root / "observer_views.jsonl"
    if manifest.get("observer_views_path") != observer_path.name:
        blockers.append("strict_neutral_object_observer_source_path_invalid")
    actual_observer_hash = _certificate_file_sha256(observer_path)
    expected_observer_hash = manifest.get("observer_views_sha256")
    if (
        not _sha256_receipt(expected_observer_hash)
        or expected_observer_hash != actual_observer_hash
    ):
        blockers.append("strict_neutral_object_observer_source_hash_mismatch")

    parameters = _as_dict(manifest.get("analysis_parameters"))
    seed = parameters.get("seed")
    min_objects = parameters.get("min_objects")
    min_observers = parameters.get("min_observers_per_object")
    max_fraction = parameters.get("max_observer_fraction_per_object")
    max_model_points = parameters.get("max_model_points")
    heldout_fraction = parameters.get("heldout_fraction")
    for name, value, minimum, maximum in (
        ("seed", seed, 0, 2**63 - 1),
        ("min_objects", min_objects, 1, 1_000_000),
        ("min_observers_per_object", min_observers, 1, 1_000_000),
        ("max_model_points", max_model_points, 8, 4_096),
    ):
        if not _bounded_strict_int(value, minimum=minimum, maximum=maximum):
            blockers.append(f"strict_neutral_object_{name}_invalid")
    if not _bounded_finite_number(max_fraction, minimum=0.0, maximum=1.0):
        blockers.append("strict_neutral_object_max_observer_fraction_invalid")
    if not _bounded_finite_number(heldout_fraction, minimum=0.01, maximum=0.95):
        blockers.append("strict_neutral_object_heldout_fraction_invalid")

    try:
        import oph_fpe.bulk.neutral_object_bulk as object_kernel
    except Exception as exc:  # pragma: no cover - defensive proof boundary.
        object_kernel = None
        blockers.append(
            f"strict_neutral_object_kernel_import_failed:{type(exc).__name__}:{exc}"
        )
    if object_kernel is not None:
        actual_kernel_hash = _certificate_file_sha256(Path(str(object_kernel.__file__)))
        expected_kernel_hash = manifest.get("analysis_kernel_file_sha256")
        if (
            not _sha256_receipt(expected_kernel_hash)
            or expected_kernel_hash != actual_kernel_hash
        ):
            blockers.append("strict_neutral_object_kernel_hash_mismatch")
    else:
        actual_kernel_hash = None

    summary.update(
        {
            "manifest_path": str(manifest_path),
            "observer_views_path": str(observer_path),
            "observer_views_sha256": actual_observer_hash,
            "analysis_kernel_file_sha256": actual_kernel_hash,
        }
    )
    if blockers or object_kernel is None:
        summary["blockers"] = list(dict.fromkeys(blockers))
        return summary

    summary["replay_attempted"] = True
    observer_rows, row_blockers = _read_strict_jsonl(observer_path)
    declared_row_count = manifest.get("observer_view_row_count")
    if (
        not _bounded_strict_int(declared_row_count, minimum=1, maximum=100_000_000)
        or declared_row_count != len(observer_rows)
    ):
        row_blockers.append("strict_neutral_object_observer_row_count_mismatch")
    recomputed: dict[str, Any] = {}
    if not row_blockers:
        try:
            recomputed = object_kernel.strict_neutral_object_bulk_report(
                observer_rows,
                seed=int(seed),
                min_objects=int(min_objects),
                min_observers_per_object=int(min_observers),
                max_observer_fraction_per_object=float(max_fraction),
                max_model_points=int(max_model_points),
                heldout_fraction=float(heldout_fraction),
            )
        except Exception as exc:  # pragma: no cover - defensive proof boundary.
            row_blockers.append(
                f"strict_neutral_object_primitive_replay_failed:{type(exc).__name__}:{exc}"
            )
    core_matches = bool(
        recomputed
        and all(persisted_report.get(key) == value for key, value in recomputed.items())
    )
    if recomputed and not core_matches:
        row_blockers.append("strict_neutral_object_persisted_report_does_not_match_replay")
    source_passed = bool(recomputed and core_matches and not row_blockers)
    passed = bool(
        source_passed
        and _truthy(recomputed, STRICT_NEUTRAL_OBJECT_BULK_RECEIPT)
        and _truthy(recomputed, "strict_neutral_object_bulk")
        and recomputed.get("blockers") == []
    )
    summary.update(
        {
            "source_validation_passed": source_passed,
            "recomputed_candidate": _truthy(
                recomputed,
                STRICT_NEUTRAL_OBJECT_BULK_RECEIPT,
            ),
            "recomputed_object_count": recomputed.get("object_count"),
            "persisted_core_matches_replay": core_matches,
            "passed": passed,
            "blockers": list(dict.fromkeys(row_blockers)),
            "claim_boundary": (
                "The object-bulk candidate is recomputed from hash-bound observer rows. "
                "A report boolean or copied derived report cannot promote it."
            ),
        }
    )
    return summary


def _independently_replayed_strict_neutral(
    root: Path,
    persisted_report: dict[str, Any],
    persisted_refinement: dict[str, Any],
) -> dict[str, Any]:
    """Replay strict-neutral primitives without invoking any report writer.

    The persisted strict-neutral and refinement JSON files are retained for
    diagnostics and as replay-trigger hints only.  Promotion is derived from
    the hash-bound observer JSONL plus pure analysis functions.  The current
    refinement producer has no primitive replay chain, so the T6 bulk gate is
    explicitly fail-closed even when its derived JSON is internally
    self-consistent.
    """

    persisted_bulk_candidate = _strict_neutral_bulk_passed(
        persisted_report,
        persisted_refinement,
    )
    persisted_metric_candidate = _strict_neutral_quotient_metric_passed(
        persisted_report
    )
    summary: dict[str, Any] = {
        "mode": "strict_neutral_primitive_bound_validation_v1",
        "replay_attempted": False,
        "persisted_bulk_candidate": persisted_bulk_candidate,
        "persisted_quotient_metric_candidate": persisted_metric_candidate,
        "source_validation_passed": False,
        "refinement_replay_passed": False,
        "quotient_metric_passed": False,
        "strict_neutral_bulk_passed": False,
        "blockers": [],
    }
    if not (persisted_bulk_candidate or persisted_metric_candidate):
        summary["blockers"] = ["persisted_strict_neutral_candidate_not_true"]
        summary["claim_boundary"] = (
            "No derived candidate requested replay; false derived reports remain false."
        )
        return summary

    manifest_path = root / "strict_neutral_source_manifest.json"
    manifest = _read_json(manifest_path)
    source_blockers: list[str] = []
    if not manifest_path.is_file() or not manifest:
        source_blockers.append("strict_neutral_source_manifest_missing")
    source_schema = manifest.get("schema")
    if source_schema not in {
        "strict_neutral_bulk_source_v1",
        "strict_neutral_bulk_source_v2",
    }:
        source_blockers.append("strict_neutral_source_manifest_schema_invalid")
    if _as_dict(persisted_report.get("source_artifact")) != manifest:
        source_blockers.append("strict_neutral_nested_source_manifest_mismatch")

    observer_name = manifest.get("observer_views_path")
    if observer_name != "observer_views.jsonl":
        source_blockers.append("strict_neutral_observer_source_path_invalid")
    observer_path = root / "observer_views.jsonl"
    expected_observer_hash = manifest.get("observer_views_sha256")
    actual_observer_hash = _certificate_file_sha256(observer_path)
    if (
        not _sha256_receipt(expected_observer_hash)
        or expected_observer_hash != actual_observer_hash
    ):
        source_blockers.append("strict_neutral_observer_source_hash_mismatch")

    parameters = _as_dict(manifest.get("analysis_parameters"))
    seed = parameters.get("seed")
    max_model_points = parameters.get("max_model_points")
    planted_control_points = parameters.get("planted_control_points")
    max_observers = parameters.get("max_observers")
    if not _bounded_strict_int(seed, minimum=0, maximum=2**63 - 1):
        source_blockers.append("strict_neutral_seed_invalid")
    if not _bounded_strict_int(max_model_points, minimum=8, maximum=4_096):
        source_blockers.append("strict_neutral_max_model_points_invalid")
    if not _bounded_strict_int(planted_control_points, minimum=16, maximum=4_096):
        source_blockers.append("strict_neutral_planted_control_points_invalid")
    if source_schema == "strict_neutral_bulk_source_v2" and not _bounded_strict_int(
        max_observers,
        minimum=8,
        maximum=10_000_000,
    ):
        source_blockers.append("strict_neutral_max_observers_invalid")

    refinement_binding = _as_dict(manifest.get("refinement_input"))
    refinement_path = root / "prime_geometric_rank_refinement_report.json"
    if refinement_binding.get("path") != refinement_path.name:
        source_blockers.append("strict_neutral_refinement_path_invalid")
    expected_refinement_hash = refinement_binding.get("sha256")
    actual_refinement_hash = _certificate_file_sha256(refinement_path)
    if actual_refinement_hash is None:
        if expected_refinement_hash is not None:
            source_blockers.append("strict_neutral_refinement_hash_mismatch")
    elif (
        not _sha256_receipt(expected_refinement_hash)
        or expected_refinement_hash != actual_refinement_hash
    ):
        source_blockers.append("strict_neutral_refinement_hash_mismatch")

    try:
        from oph_fpe.bulk import neutral_bulk as neutral_kernel
    except Exception as exc:  # pragma: no cover - defensive proof boundary.
        neutral_kernel = None
        source_blockers.append(
            f"strict_neutral_analysis_kernel_import_failed:{type(exc).__name__}:{exc}"
        )
    if neutral_kernel is not None:
        kernel_path = Path(str(neutral_kernel.__file__))
        expected_kernel_hash = manifest.get("analysis_kernel_file_sha256")
        actual_kernel_hash = _certificate_file_sha256(kernel_path)
        if (
            not _sha256_receipt(expected_kernel_hash)
            or expected_kernel_hash != actual_kernel_hash
        ):
            source_blockers.append("strict_neutral_analysis_kernel_hash_mismatch")
    else:
        actual_kernel_hash = None

    summary.update(
        {
            "manifest_path": str(manifest_path),
            "observer_views_path": str(observer_path),
            "observer_views_sha256": actual_observer_hash,
            "analysis_kernel_file_sha256": actual_kernel_hash,
            "refinement_report_sha256": actual_refinement_hash,
            "source_blockers": list(dict.fromkeys(source_blockers)),
        }
    )
    if source_blockers or neutral_kernel is None:
        summary["blockers"] = list(dict.fromkeys(source_blockers))
        summary["claim_boundary"] = (
            "A handwritten derived report cannot promote without a hash-bound "
            "observer source and matching analysis kernel."
        )
        return summary

    summary["replay_attempted"] = True
    observer_rows, row_blockers = _read_strict_jsonl(observer_path)
    declared_row_count = manifest.get("observer_view_row_count")
    if (
        not _bounded_strict_int(declared_row_count, minimum=1, maximum=100_000_000)
        or declared_row_count != len(observer_rows)
    ):
        row_blockers.append("strict_neutral_observer_row_count_mismatch")
    recomputed: dict[str, Any] = {}
    analysis_rows = observer_rows
    if not row_blockers and source_schema == "strict_neutral_bulk_source_v2":
        try:
            analysis_rows, recomputed_population = (
                neutral_kernel.bounded_strict_neutral_observer_views(
                    observer_rows,
                    max_observers=int(max_observers),
                )
            )
            if _as_dict(manifest.get("analysis_population")) != recomputed_population:
                row_blockers.append("strict_neutral_analysis_population_mismatch")
        except Exception as exc:  # pragma: no cover - defensive proof boundary.
            row_blockers.append(
                f"strict_neutral_analysis_population_replay_failed:{type(exc).__name__}:{exc}"
            )
    if not row_blockers:
        try:
            planted = neutral_kernel.planted_neutral_control_report(
                point_count=int(planted_control_points),
                seed=int(seed) + 101,
                max_points=min(int(max_model_points), int(planted_control_points)),
            )
            shuffled = neutral_kernel.shuffled_neutral_control_report(
                analysis_rows,
                seed=int(seed) + 303,
                max_model_points=min(int(max_model_points), 96),
            )
            controls = dict(_as_dict(planted.get("controls")))
            controls.update(_as_dict(shuffled.get("controls")))
            # Deliberately omit the unverified derived refinement.  This keeps
            # the source replay useful for diagnostics while closing T6.
            recomputed = neutral_kernel.strict_neutral_bulk_report(
                analysis_rows,
                controls=controls,
                refinement={},
                seed=int(seed),
                max_model_points=int(max_model_points),
            )
        except Exception as exc:  # pragma: no cover - defensive proof boundary.
            row_blockers.append(
                f"strict_neutral_primitive_replay_failed:{type(exc).__name__}:{exc}"
            )

    primitive_keys = (
        "observer_count",
        "distance_matrix_shape",
        "neutral_metric_construction",
        "channel_audit",
        "strict_neutral_theory_alignment",
        "dimension",
        "model_selection",
        "leakage",
        "controls",
    )
    primitive_matches = {
        key: persisted_report.get(key) == recomputed.get(key)
        for key in primitive_keys
    }
    if recomputed and not all(primitive_matches.values()):
        row_blockers.append("strict_neutral_persisted_primitives_do_not_match_replay")
    source_validation_passed = bool(recomputed and not row_blockers)

    primitive_refinement_available = (
        refinement_binding.get("primitive_replay_available") is True
    )
    refinement_blockers = [
        (
            "strict_neutral_refinement_primitive_replay_verifier_not_implemented"
            if primitive_refinement_available
            else "strict_neutral_refinement_primitive_replay_unavailable"
        )
    ]
    quotient_metric_passed = bool(
        source_validation_passed
        and _strict_neutral_quotient_metric_passed(recomputed)
    )
    summary.update(
        {
            "source_validation_passed": source_validation_passed,
            "primitive_section_matches": primitive_matches,
            "recomputed_dimension": recomputed.get("dimension", {}),
            "recomputed_model_selection": recomputed.get("model_selection", {}),
            "recomputed_leakage": recomputed.get("leakage", {}),
            "recomputed_quotient_metric_candidate": _strict_neutral_quotient_metric_passed(
                recomputed
            ),
            "refinement_replay_passed": False,
            "refinement_blockers": refinement_blockers,
            "quotient_metric_passed": quotient_metric_passed,
            "strict_neutral_bulk_passed": False,
            "blockers": list(dict.fromkeys([*row_blockers, *refinement_blockers])),
            "claim_boundary": (
                "Observer-distance diagnostics are independently replayed from hash-bound JSONL. "
                "T6 remains false because the multi-scale refinement artifact has no primitive "
                "source replay chain; its self-reported booleans and hash are not proof."
            ),
        }
    )
    return summary


def _certificate_file_sha256(path: Path) -> str | None:
    if not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _sha256_receipt(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def _bounded_strict_int(value: Any, *, minimum: int, maximum: int) -> bool:
    return type(value) is int and minimum <= value <= maximum


def _bounded_finite_number(value: Any, *, minimum: float, maximum: float) -> bool:
    if type(value) not in (int, float):
        return False
    parsed = float(value)
    return math.isfinite(parsed) and minimum <= parsed <= maximum


def _read_strict_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    blockers.append(f"strict_neutral_observer_json_invalid:{line_number}")
                    continue
                if not isinstance(row, dict):
                    blockers.append(f"strict_neutral_observer_row_not_object:{line_number}")
                    continue
                rows.append(row)
    except OSError as exc:
        blockers.append(f"strict_neutral_observer_source_unreadable:{type(exc).__name__}:{exc}")
    if not rows:
        blockers.append("strict_neutral_observer_source_empty")
    return rows, blockers


def _observer_report_gate(report: dict[str, Any], key: str) -> bool:
    component_gates = report.get("component_gates") if isinstance(report.get("component_gates"), dict) else {}
    history_gates = (
        report.get("history_component_gates")
        if isinstance(report.get("history_component_gates"), dict)
        else {}
    )
    return bool(
        _is_true(report.get(key))
        or _is_true(component_gates.get(key))
        or _is_true(history_gates.get(key))
    )


def _is_true(value: Any) -> bool:
    """Accept only the JSON boolean true, never generic truthiness."""

    return type(value) is bool and value


def _canonical_refinement_passed(report: dict[str, Any]) -> bool:
    required = [4_096, 16_384, 65_536, 262_144]
    sizes = report.get("sizes")
    if not isinstance(sizes, list) or len(sizes) != len(required):
        return False
    if any(
        not isinstance(row, dict)
        or type(row.get("patch_count")) is not int
        for row in sizes
    ):
        return False
    observed = sorted(row["patch_count"] for row in sizes)
    return all(
        (
            report.get("mode") == "prime_geometric_rank_refinement_v0",
            report.get("required_patch_count_ladder") == required,
            observed == required,
            report.get("missing_required_patch_counts") == [],
            _truthy(report, "required_ladder_complete"),
            _truthy(report, "multi_scale"),
            _truthy(report, "all_control_quotient_spatial_3d_candidates"),
            _truthy(report, "all_candidate_s2_leakage_pass"),
            _truthy(report, "all_candidate_rank3_e3"),
            _truthy(report, "candidate_dimension_stable"),
            _truthy(report, "independent_rank3_selector_all"),
            _truthy(report, "proper_negative_control_all"),
            _truthy(report, "directional_h3_strict_all"),
            _truthy(report, "measured_overlap_geometry_all"),
            _truthy(report, "strict_neutral_bulk_refinement_receipt"),
            report.get("proof_blockers") == [],
        )
    )


def _strict_neutral_quotient_metric_passed(report: dict[str, Any]) -> bool:
    quotient = _as_dict(report.get("quotient_geometry_contract"))
    metric = _as_dict(quotient.get("metric"))
    return all(
        (
            _truthy(quotient, "QUOTIENT_GEOMETRY_CONTRACT_RECEIPT"),
            _truthy(quotient, "bulk_promotion_allowed"),
            _truthy(metric, "valid_pseudometric"),
            _truthy(metric, "valid_metric"),
            _truthy(metric, "triangle_checked_exact"),
            quotient.get("blockers") == [],
            metric.get("blockers") == [],
            metric.get("metric_blockers") == [],
        )
    )


def _strict_neutral_bulk_passed(
    report: dict[str, Any],
    persisted_refinement: dict[str, Any],
) -> bool:
    """Independently compose the persisted strict-neutral primitive gates."""

    if report.get("mode") != "strict_neutral_bulk_record_transition_audit":
        return False
    dimension = _as_dict(report.get("dimension"))
    model = _as_dict(report.get("model_selection"))
    leakage = _as_dict(report.get("leakage"))
    controls = _as_dict(report.get("controls"))
    refinement = _as_dict(report.get("refinement"))
    channel_audit = _as_dict(report.get("channel_audit"))
    theory = _as_dict(report.get("strict_neutral_theory_alignment"))
    receipt = _as_dict(report.get("receipt"))
    return all(
        (
            _truthy(dimension, "estimators_agree_3d"),
            model.get("best_model") == "H3",
            _truthy(model, "h3_beats_s2"),
            _truthy(model, "h3_beats_h2_h4"),
            _truthy(leakage, "s2_leakage_pass"),
            _truthy(channel_audit, "duplicate_channel_gate_pass"),
            _truthy(channel_audit, "feature_ancestry_gate_pass"),
            _truthy(theory, "theory_required_channels_present"),
            _truthy(controls, "shuffled_records_fail"),
            _truthy(controls, "shuffled_transition_labels_fail"),
            _truthy(controls, "planted_2d_returns_2d"),
            _truthy(controls, "planted_3d_returns_3d"),
            _truthy(controls, "planted_h3_returns_h3"),
            refinement == persisted_refinement,
            _canonical_refinement_passed(refinement),
            _strict_neutral_quotient_metric_passed(report),
            _as_dict(report.get("quotient_geometry_contract")).get("refinement") == refinement,
            receipt.get("receipt") == "STRICT_NEUTRAL_BULK_RECEIPT",
            _truthy(receipt, "strict_neutral_bulk"),
            _truthy(receipt, "physical_claim"),
            _truthy(report, "strict_neutral_bulk"),
            report.get("blockers") == [],
        )
    )


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
            float(_truthy(report, "observer_chart_bulk_population_receipt")),
            float(_truthy(report, "OBJECT_H3_NONBOUNDARY_POPULATION_RECEIPT")),
            float(_truthy(report, "THEOREM_ASSISTED_H3_OBJECT_PREVIEW_RECEIPT")),
            float(_truthy(report, "h3_beats_shuffled_incidence_robust")),
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
        "- paper-geometric branch observer spacetime emergence: "
        f"`{str(report['paper_geometric_branch_observer_spacetime_emergence_receipt']).lower()}`",
        "- paper-geometric branch observer-facing consensus bulk emergence: "
        f"`{str(report['paper_geometric_branch_consensus_bulk_emergence_receipt']).lower()}`",
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
                f"- paper-geometric branch blockers: `{contract.get('paper_geometric_branch_primary_blockers')}`",
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
