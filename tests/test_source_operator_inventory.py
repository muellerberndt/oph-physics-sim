from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from oph_fpe.dynamics import source_operator_inventory as inventory
from oph_fpe.dynamics import verify_source_operator_inventory_independent as independent


REPORT = inventory.build_inventory()


def _rehash(report: dict) -> None:
    payload = copy.deepcopy(report)
    payload.pop("receipt_sha256", None)
    report["receipt_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _row(report: dict, path: str) -> dict:
    return next(
        row for row in report["canonical_artifact_rows"] if row["path"] == path
    )


def test_inventory_replays_on_the_indexed_serialized_data_surface() -> None:
    verification = independent.verify(REPORT)
    assert verification["receipt"] is True, verification["reasons"]
    assert REPORT["status"] == inventory.STATUS
    assert REPORT["issue"] == 655
    assert REPORT["scope"] == (
        "Git-indexed tracked paths under data; semantic scan of current canonical "
        "simulator JSON objects excluding declared recursive ancestor and descendant "
        "outputs; legacy, imported, and external/comparison paths counted only"
    )

    catalog = REPORT["tracked_serialized_data_catalog"]
    assert catalog["path_count_including_declared_recursive_outputs"] >= 1000
    assert catalog["content_index_row_count_excluding_recursive_outputs"] >= 998
    assert catalog["provenance_counts"]["LEGACY_EARNED_RUN"] >= 895
    assert catalog["provenance_counts"]["IMPORTED_NONNATIVE"] >= 30
    assert catalog["provenance_counts"]["EXTERNAL_OR_COMPARISON_DATA"] >= 40
    assert catalog["untracked_data_paths_excluding_declared_recursive_outputs"] == []
    assert catalog["unstaged_current_canonical_inputs"] == []


def test_only_current_canonical_json_is_scanned_semantically() -> None:
    scan = REPORT["current_canonical_json_contract_scan"]
    assert scan["current_canonical_json_path_count_excluding_recursive_outputs"] == (
        len(inventory.CANONICAL_CONTRACTS) - len(inventory.DECLARED_OUTPUT_PATHS)
    )
    assert scan["recursive_output_paths_excluded"] == sorted(
        inventory.DECLARED_OUTPUT_PATHS
    )
    assert scan["registered_source_packet_rows_excluding_recursive_outputs"] == []
    assert scan["positive_promotion_signal_rows_excluding_recursive_outputs"] == []

    for label in (
        "LEGACY_EARNED_RUN",
        "IMPORTED_NONNATIVE",
        "EXTERNAL_OR_COMPARISON_DATA",
    ):
        assert REPORT["noncurrent_path_catalog"][label][
            "semantic_payloads_scanned"
        ] is False
    assert all(
        not row["path"].startswith(inventory.NONCURRENT_PREFIXES)
        for row in REPORT["canonical_artifact_rows"]
    )


def test_every_current_canonical_json_has_an_exact_contract() -> None:
    rows = REPORT["canonical_artifact_rows"]
    assert {row["path"] for row in rows} == set(inventory.CANONICAL_CONTRACTS)
    for row in rows:
        contract = inventory.CANONICAL_CONTRACTS[row["path"]]
        assert row["schema"] == contract["schema"]
        assert row["status"] == contract["status"]
        assert row["disposition"] == contract["disposition"]
        assert row["semantic_scan_excluded_as_recursive_output"] == (
            row["path"] in inventory.DECLARED_OUTPUT_PATHS
        )
        if row["path"] in inventory.DECLARED_OUTPUT_PATHS:
            assert "raw_pin" not in row
        else:
            assert row["raw_pin"]["path"] == row["path"]


def test_conditional_fz11_adapter_is_not_admitted_as_source_capability() -> None:
    row = _row(
        REPORT,
        "data/repair_closure/fz11_3d_translation_bridge_receipt.json",
    )
    assert row["schema"] == "oph.fz11-conditional-3d-translation-bridge.v1"
    assert row["status"] == (
        "CONDITIONAL_3D_SCALAR_TRANSLATION_ADAPTER_ATTAINED__"
        "CANONICAL_SOURCE_SELECTION_TIME_EVOLUTION_PHOTON_SECTOR_SCALE_FRAME_"
        "BOOST_AND_EXCLUSIVITY_OPEN"
    )
    assert row["disposition"] == (
        "CONDITIONAL_AUXILIARY_CONTINUOUS_R3_TRANSLATION_ADAPTER__"
        "NOT_SOURCE_NATIVE_NOT_PHYSICAL_NOT_COMPARISON_ELIGIBLE"
    )
    assert REPORT["bridge_admission_contract"][
        "accepted_bridge_count_excluding_recursive_outputs"
    ] == 0


def test_conditional_fz11_time_lift_is_not_a_physical_bridge() -> None:
    row = _row(
        REPORT,
        "data/repair_closure/fz11_conservative_time_lift_receipt.json",
    )
    assert row["schema"] == "oph.fz11-conservative-time-lift.v1"
    assert row["status"] == (
        "EXACT_CONSERVATIVE_TIME_LIFT_FOR_DECLARED_FZ11_OPERATOR_ATTAINED__"
        "SOURCE_B_CLOCK_LORENTZ_SECTOR_CONTINUUM_AND_SCALE_OPEN"
    )
    assert row["disposition"] == (
        "CONDITIONAL_CONSERVATIVE_TIME_LIFT__TRANSLATION_SOURCE_CLOCK_"
        "LORENTZ_SECTOR_CONTINUUM_SCALE_AND_READOUT_OPEN"
    )
    assert row["critical_bridge_evidence"] == {
        "emitted_B_source_selected": False,
        "emitted_comparison_permitted": False,
        "emitted_conditional_auxiliary_time_evolution": True,
        "emitted_continuum_limit": False,
        "emitted_generic_psd_factorizations_remain_nonunique": True,
        "emitted_lorentz_or_boost_law": False,
        "emitted_phase_norm_identified_with_physical_energy": False,
        "emitted_physical_clock_selected": False,
        "emitted_physical_field_sector_selected": False,
        "emitted_physical_readout_selected": False,
        "emitted_physical_scale_selected": False,
        "emitted_repair_tick_supplies_physical_time": False,
        "emitted_translation_action_source_selected": False,
    }
    assert REPORT["bridge_admission_contract"][
        "accepted_bridge_count_excluding_recursive_outputs"
    ] == 0


def test_port_gram_descendant_is_declared_and_excluded_from_recursive_scan() -> None:
    row = _row(REPORT, inventory.PORT_GRAM_RELATIVE_PATH)
    assert row == {
        "critical_bridge_evidence": None,
        "disposition": (
            "RECURSIVE_DESCENDANT_PORT_GRAM_COMPLETION_RECEIPT_EXCLUDED_FROM_"
            "SEMANTIC_SCAN"
        ),
        "path": inventory.PORT_GRAM_RELATIVE_PATH,
        "schema": "oph.port-gram-hausdorff-completion-bridge.v1",
        "semantic_scan_excluded_as_recursive_output": True,
        "status": (
            "EXACT_REPAIR_RESPONSE_GRAM_QUOTIENT_AND_3D_COMPLETION_ATTAINED__"
            "A1R_SIGNED_RECORD_MODULE_AND_A2R_POSITION_READBACK_PREMISES_OPEN"
        ),
    }
    exclusion = REPORT["bridge_admission_contract"][
        "recursive_descendant_receipt_exclusion"
    ]
    assert exclusion["path"] == inventory.PORT_GRAM_RELATIVE_PATH
    assert exclusion["packet_count_included_in_scan"] is False


def test_completion_action_and_load_quotient_descendants_are_cycle_excluded() -> None:
    expected = {
        inventory.PORT_GRAM_ACTION_RELATIVE_PATH: (
            "oph.port-gram-equivariant-completion-action.v1",
            "RECURSIVE_DESCENDANT_EQUIVARIANT_ACTION_RECEIPT_EXCLUDED_FROM_SEMANTIC_SCAN",
        ),
        inventory.PORT_LOAD_QUOTIENT_RELATIVE_PATH: (
            "oph.port-load-repair-gram-metric-quotient.v1",
            "RECURSIVE_DESCENDANT_LOAD_QUOTIENT_RECEIPT_EXCLUDED_FROM_SEMANTIC_SCAN",
        ),
    }
    for path, (schema, disposition) in expected.items():
        row = _row(REPORT, path)
        assert row["path"] == path
        assert row["schema"] == schema
        assert row["disposition"] == disposition
        assert row["semantic_scan_excluded_as_recursive_output"] is True
        assert row["critical_bridge_evidence"] is None
        assert "raw_pin" not in row


def test_primitive_port_dual_measure_retains_physical_attachment_boundary() -> None:
    path = "data/repair_closure/primitive_port_dual_measure_receipt.json"
    row = _row(REPORT, path)
    assert row["schema"] == "oph.primitive-port-dual-normalized-measure.v1"
    assert row["status"] == (
        "QUOTIENT_VISIBLE_NORMALIZED_PORT_DUAL_MEASURE_ATTAINED__"
        "PHYSICAL_PIXEL_AND_HOP_IDENTITIES_OPEN"
    )
    assert row["disposition"] == (
        "QUOTIENT_VISIBLE_PORT_DUAL_MEASURE__PHYSICAL_PIXEL_HOP_AND_SCALE_"
        "IDENTITIES_OPEN"
    )
    assert row["critical_bridge_evidence"] == {
        "emitted_comparison_permitted": False,
        "emitted_declared_finite_refinement_naturality": True,
        "emitted_exact_normalized_port_dual_measure": True,
        "emitted_issue_662_armed": False,
        "emitted_kappa_geom_source_selected": False,
        "emitted_physical_P_pixel_identification": False,
        "emitted_physical_prediction_promoted": False,
        "emitted_quotient_visible_port_to_support_map": True,
        "emitted_shared_geometry_physical_identity": False,
        "emitted_support_radius_hop_identification": False,
        "emitted_terminal_refinement_stage_selected": False,
    }


def test_ordered_port_diagnostic_is_indexed_with_exact_negative_evidence() -> None:
    path = "data/a2_holonomy/ordered_port_response_diagnostic_receipt.json"
    row = _row(REPORT, path)
    assert row["schema"] == "oph.ordered-port-response-diagnostic.v1"
    assert row["status"] == "ATTAINED_BOUNDED_NEGATIVE_CONTROL"
    assert row["disposition"] == (
        "TWELVE_PORT_ADJACENCY_PROPAGATION_OVERSHOOTS_TO_U12__"
        "PHYSICAL_CURRENT_SOURCE_OPEN"
    )
    evidence = row["critical_bridge_evidence"]
    assert evidence == {
        "emitted_port_count": 12,
        "emitted_propagation_generator": "minus i times L, where L = 5 I - A",
        "emitted_generated_algebra_type": "u(12)",
        "emitted_generated_algebra_real_rank": 144,
        "emitted_derived_algebra_type": "su(12)",
        "emitted_derived_algebra_real_rank": 143,
        "emitted_A1_complete_response_receipt": False,
        "emitted_A2_same_current_receipt": False,
        "emitted_physical_current_source_bridge_receipt": False,
        "emitted_u12_is_candidate_oph_current": False,
    }


def test_near_candidates_remain_separate_objects() -> None:
    charged = _row(
        REPORT, "data/common_reserve/charged_response_artifact.json"
    )["critical_bridge_evidence"]
    assert charged["emitted_support_size"] == 12
    assert charged["emitted_source_response_operator"] == (
        "negative_graph_antipode_involution"
    )
    assert charged["emitted_source_bound_impulse_readback"] is True
    assert charged["emitted_current_lift_source_selected"] is False
    assert charged["spatial_translation_binding"] == {
        "classification": "ABSENT_FROM_DECLARED_SCHEMA",
        "key": "spatial_translation_identification",
        "searched_scope": "entire_canonical_json_object",
        "occurrences": [],
    }
    assert charged["same_operator_physical_readout"]["classification"] == (
        "ABSENT_FROM_DECLARED_SCHEMA"
    )

    stage3 = _row(REPORT, "data/local_domain/stage3_receipt.json")[
        "critical_bridge_evidence"
    ]
    assert stage3["emitted_visible_edge_count"] == 11816
    assert stage3["emitted_physical_promotion_allowed"] is False
    assert stage3["vertex12_identity_bridge"]["classification"] == (
        "ABSENT_FROM_DECLARED_SCHEMA"
    )

    gap = _row(REPORT, "data/local_domain/source_gap_receipt.json")[
        "critical_bridge_evidence"
    ]
    assert gap["emitted_operator"] == (
        "signed Laplacian of the observer-visible seam complex"
    )
    assert gap["emitted_physical_promotion_allowed"] is False
    assert gap["physical_reference_transition"]["classification"] == (
        "ABSENT_FROM_DECLARED_SCHEMA"
    )

    vertex12 = _row(
        REPORT,
        "data/repair_closure/vertex12_atomic_port_transfer_receipt.json",
    )["critical_bridge_evidence"]
    assert vertex12 == {
        "operator_domain": "internal_federation_visible_port_fiber_Q^(8_times_12)",
        "emitted_source_native_internal_seam_partner_operator": True,
        "emitted_exact_symbolic_matching_and_projector_algebra": True,
        "emitted_source_native_spatial_translation": False,
        "emitted_in_process_snapshot_reread_carrier_count": 8,
        "emitted_readback_mechanism": "in_process_snapshot_lookup_digest_reread",
        "emitted_independent_persistence_readback": False,
        "emitted_independent_second_producer_readback": False,
        "emitted_physical_sector_readout": False,
        "emitted_noncollapsed_inverse_compatible_quotient_count": 0,
        "emitted_same_operator_physical_readout": False,
        "emitted_current_fixed_matching_family_has_no_qualifying_carrier_set_quotient": True,
    }

    feasibility = _row(
        REPORT,
        "data/repair_closure/vertex12_directed_transport_feasibility_receipt.json",
    )
    assert feasibility["disposition"] == (
        "CURRENT_MATCHING_COVER_OBSTRUCTED__DECLARED_ALGEBRAIC_CONTROL_"
        "NOT_SOURCE_EMITTED_OR_PHYSICAL"
    )
    assert feasibility["critical_bridge_evidence"] == {
        "emitted_antipodal_pair_count": 6,
        "emitted_semiconjugate_cover_can_satisfy_inverse_law": False,
        "emitted_algebraic_control_site_domain": "(Z/3Z)^6",
        "emitted_algebraic_control_site_count": 729,
        "emitted_algebraic_control_A5_order": 60,
        "emitted_algebraic_control_inverse_and_covariance": True,
        "emitted_control_source_transition_event": False,
        "emitted_control_repair_generated": False,
        "emitted_control_source_selected_site_completion": False,
        "emitted_control_spatial_translation": False,
        "emitted_control_physical_readout": False,
        "emitted_control_physical_prediction": False,
        "emitted_requested_source_transport_ledger": False,
        "emitted_requested_twelve_directed_maps": False,
        "emitted_requested_antipodal_inverse": False,
        "emitted_requested_A5_covariance": False,
    }

    coefficients = _row(
        REPORT,
        "data/refinement/a5_biposh_dual_operator_coefficients.json",
    )
    assert coefficients["disposition"] == (
        "FULL_BIPOSH_COEFFICIENT_BUNDLE__SUPPORTING_OPERATOR_FINGERPRINT_"
        "ONLY_NOT_PHYSICAL"
    )
    assert coefficients["critical_bridge_evidence"] == {
        "emitted_coefficient_kind": "finite stiffness-form operator fingerprint",
        "emitted_case_count": 8,
        "emitted_coefficient_count_per_case": 5929,
        "forbidden_top_level_claim_fields_checked": list(
            inventory.BIPOSH_COEFFICIENT_FORBIDDEN_CLAIM_FIELDS
        ),
        "forbidden_top_level_claim_fields_present": [],
        "emitted_supporting_bundle_has_separate_status": False,
        "emitted_physical_promotion": False,
    }

    biposh = _row(
        REPORT,
        "data/refinement/a5_biposh_dual_operator_receipt.json",
    )
    assert biposh["disposition"] == (
        "FINITE_DUAL_OPERATOR_FINGERPRINT__OPERATOR_SELECTION_CONTINUUM_"
        "AND_PHYSICAL_COVARIANCE_OPEN"
    )
    assert biposh["critical_bridge_evidence"] == {
        "emitted_base_628_operator_match": True,
        "emitted_base_labelled_face_presentation_matches_parent": True,
        "emitted_base_edge_set_matches_face_presentation": True,
        "emitted_base_equal_seam_operator_bounded_reconstructed": True,
        "emitted_continuum_residual_decided": False,
        "emitted_equal_seam_operator_source_selected": False,
        "emitted_global_frame_quotient_visible": False,
        "emitted_physical_covariance_selected": False,
        "emitted_physical_prediction": False,
        "emitted_physical_release_ensemble_selected": False,
        "emitted_physical_repair_law_selected": False,
        "emitted_promotion_allowed": False,
        "emitted_refinement_extension_source_selected": False,
        "emitted_screen_to_sky_readout_selected": False,
        "emitted_comparison_data_used": False,
    }


def test_new_conditional_receipts_have_exact_fail_closed_boundaries() -> None:
    equal_seam = _row(
        REPORT,
        "data/refinement/refined_equal_seam_source_gate_receipt.json",
    )
    assert equal_seam["disposition"] == (
        "BASE_EQUAL_SEAM_EXACT__REGISTERED_MESH_A5_ORBITS_RESIDUAL_GATED__"
        "ALL_LEVEL_SOURCE_COUNTING_AND_PHYSICAL_OPERATOR_OPEN"
    )
    assert equal_seam["critical_bridge_evidence"] == {
        "emitted_base_equal_seam_operator_selected_in_bounded_realization": True,
        "emitted_registered_mesh_a5_orbits_residual_gated": True,
        "emitted_edge_orbit_counts": [1, 2, 8, 32, 128, 512],
        "emitted_maximum_coordinate_residual": 5.688200336284365e-16,
        "emitted_coordinate_residual_gate": 5.0e-11,
        "emitted_all_registered_mesh_permutation_gates_passed": True,
        "emitted_all_edge_incidence_gates_passed": True,
        "emitted_refined_edge_alphabets_have_multiple_a5_orbits": True,
        "emitted_a5_cross_orbit_weight_selection": False,
        "emitted_canonical_a1_a3_cross_orbit_weight_source": False,
        "emitted_all_level_atomic_counting_law_source": False,
        "emitted_refinement_commuting_diagram": False,
        "emitted_continuum_equal_seam_operator": False,
        "emitted_physical_repair_law": False,
        "emitted_physical_covariance": False,
        "emitted_promotion_allowed": False,
        "emitted_comparison_data_used": False,
        "emitted_framework_wide_no_go": False,
        "emitted_fourth_axiom_logically_required": False,
        "emitted_canonical_basis_amendment_required": True,
        "emitted_unit_counting_additional_premise_until_derived": True,
        "emitted_unit_counting_derived_from_canonical_structures": False,
    }

    tail = _row(
        REPORT,
        "data/refinement/a5_biposh_continuum_tail_receipt.json",
    )
    assert tail["disposition"] == (
        "CONDITIONAL_EQUAL_SEAM_CONTINUUM_L6_NONZERO__SOURCE_SELECTION_"
        "INVERSE_TAIL_AND_PHYSICAL_TRANSFER_OPEN"
    )
    assert tail["critical_bridge_evidence"] == {
        "emitted_exact_refinement_identity": True,
        "emitted_declared_block_cauchy_limit": True,
        "emitted_conditional_stiffness_continuum_limit": True,
        "emitted_conditional_l6_nonzero_under_numerical_envelope": True,
        "emitted_conditional_interval_excludes_zero": True,
        "emitted_numerical_envelope_is_analytic_library_proof": False,
        "emitted_equal_seam_refinement_source_selected": False,
        "emitted_global_a1_a3_policy_uniqueness": False,
        "emitted_inverse_covariance_finite_diagnostic": True,
        "emitted_inverse_covariance_continuum_tail": False,
        "emitted_inverse_covariance_continuum_limit": False,
        "emitted_source_ensemble_selected": False,
        "emitted_physical_covariance": False,
        "emitted_screen_to_sky_readout": False,
        "emitted_physical_prediction": False,
        "emitted_promotion_allowed": False,
    }

    inverse = _row(
        REPORT,
        "data/refinement/a5_biposh_inverse_continuum_gate_receipt.json",
    )
    assert inverse["disposition"] == (
        "FULL_RAW_STIFFNESS_CAUCHY__INVERSE_COERCIVITY_QUOTIENT_AND_"
        "PHYSICAL_RESPONSE_OPEN"
    )
    assert inverse["critical_bridge_evidence"] == {
        "emitted_full_raw_stiffness_cauchy_limit": True,
        "emitted_uniform_continuum_coercivity": False,
        "emitted_projected_quotient_continuum_tail": False,
        "emitted_full_inverse_covariance_continuum_limit": False,
        "emitted_finite_anchor_neumann_gate": False,
        "emitted_source_ensemble_selected": False,
        "emitted_declared_ladder_primitive_alphabet": True,
        "emitted_declared_ladder_unit_counting": True,
        "emitted_declared_ladder_reaches_inverse_anchor": False,
        "emitted_first_order_refinement_readback": True,
        "emitted_full_refinement_commuting_diagram": False,
        "emitted_all_level_response_law": False,
        "emitted_operational_stiffness_observable_candidate": True,
        "emitted_physical_response_readout": False,
        "emitted_scalar_rescaling_cancellation": True,
        "emitted_rotation_equivariant_transfer": False,
        "emitted_multiplicity_one": False,
        "emitted_radial_copy_mixing_excluded": False,
        "emitted_shape_regular_coercivity_theorem": False,
        "emitted_exact_tail_arithmetic_reimplemented": True,
        "emitted_independent_harmonic_implementation": False,
        "emitted_harmonic_kernels_shared_with_producer": True,
        "emitted_physical_covariance": False,
        "emitted_physical_prediction": False,
        "emitted_promotion_allowed": False,
        "emitted_comparison_data_used": False,
    }

    all_level = _row(
        REPORT,
        "data/refinement/all_level_primitive_seam_source_receipt.json",
    )
    assert all_level["disposition"] == (
        "DECLARED_REGISTERED_LADDER_UNIT_COUNTING_BRANCH__INFINITE_TOWER_"
        "ATOMIC_RECORD_FULL_REFINEMENT_CONTINUUM_AND_PHYSICAL_BRIDGES_OPEN"
    )
    assert all_level["critical_bridge_evidence"] == {
        "emitted_registered_level_count": 6,
        "emitted_complete_primitive_event_count": 40950,
        "emitted_declared_registered_ladder_event_source": True,
        "emitted_declared_registered_ladder_unit_counting_source": True,
        "emitted_infinite_tower_event_source": False,
        "emitted_infinite_tower_unit_counting_source": False,
        "emitted_unit_counting_across_a5_orbits": True,
        "emitted_unit_counting_derived_from_canonical_a1_a3": False,
        "emitted_expected_balancing_diagnostic": True,
        "emitted_canonical_a2_pathwise_agreement": False,
        "emitted_odd_total_pathwise_exact_agreement": False,
        "emitted_issue_628_atomic_record_bridge": False,
        "emitted_complete_event_lineage": True,
        "emitted_normalized_counting_refinement_naturality": True,
        "emitted_first_order_refinement_readback": True,
        "emitted_full_refinement_commuting_diagram": False,
        "emitted_repair_semigroup_refinement_naturality": False,
        "emitted_canonical_a1_a3_force_emitter": False,
        "emitted_continuum_operator_selected": False,
        "emitted_physical_repair_law": False,
        "emitted_physical_prediction": False,
        "emitted_promotion_allowed": False,
        "emitted_comparison_data_used": False,
    }

    endpoint = _row(
        REPORT,
        "data/repair_closure/vertex12_a2_endpoint_commutator_receipt.json",
    )
    assert endpoint["disposition"] == (
        "CONDITIONAL_A2_ENDPOINT_DESCENT_AND_Z_POWER_6_FACTORIZATION__SOURCE_"
        "NATURALITY_INVERSES_DIAMONDS_FAITHFUL_ACTION_AND_PHYSICAL_"
        "TRANSLATION_OPEN"
    )
    assert endpoint["critical_bridge_evidence"] == {
        "emitted_conditional_a2_endpoint_descent_lemma": True,
        "emitted_a2_endpoint_diamonds_without_source_premise": False,
        "emitted_universal_z_power_6_factorization": True,
        "emitted_positive_axis_diamond_count": 15,
        "emitted_antipodal_inverse_count": 6,
        "emitted_current_terminal_confluence": True,
        "emitted_current_port_block_maps_bijective": False,
        "emitted_current_oriented_bijective_step_ledger": False,
        "emitted_source_endpoint_diamond_ledger": False,
        "emitted_source_a2_naturality_rows": False,
        "emitted_source_accepted_observer_quotient": False,
        "emitted_faithful_physical_z_power_6_action": False,
        "emitted_spatial_translation": False,
        "emitted_physical_prediction": False,
        "emitted_negative_issue_655_closure": False,
        "emitted_comparison_data_used": False,
    }

    constructive = _row(
        REPORT,
        "data/repair_closure/vertex12_constructive_source_law_receipt.json",
    )
    assert constructive["disposition"] == (
        "CONSTRUCTIVE_SOURCE_LAW_CONTROL_ONLY__CANONICAL_SELECTION_AND_"
        "PHYSICAL_ATTACHMENT_OPEN"
    )
    assert constructive["critical_bridge_evidence"] == {
        "emitted_constructive_source_capture_root": True,
        "emitted_accepted_surjective_quotient": True,
        "emitted_raw_step_count": 12,
        "emitted_meaning_step_count": 12,
        "emitted_A2_descent_square_count": 12,
        "emitted_quotient_inverse_count": 6,
        "emitted_endpoint_diamond_count": 15,
        "emitted_same_Q_A5_group_order": 60,
        "emitted_same_Q_A5_covariance_row_count": 720,
        "emitted_canonical_source_selection": False,
        "emitted_canonical_A1_A2_A3_derivation_claimed": False,
        "emitted_full_canonical_A1_typed_object_instantiated": False,
        "emitted_full_A2_observer_federation_functor_instantiated": False,
        "emitted_canonical_A3_maximum_entropy_selection_instantiated": False,
        "emitted_spatial_translation": False,
        "emitted_physical_readout": False,
        "emitted_physical_prediction": False,
        "emitted_advances_canonical_source_bridge": False,
        "emitted_advances_physical_bridge": False,
        "emitted_issue_closure_supported": False,
        "emitted_comparison_data_used": False,
    }


def test_expensive_tail_replay_is_cached_by_complete_content_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert inventory._tail_verification_key() == independent._tail_verification_key()
    labels = {row[0] for row in inventory._tail_verification_key()}
    assert "data/refinement/a5_biposh_continuum_tail_receipt.json" in labels
    assert "data/refinement/a5_biposh_dual_operator_receipt.json" in labels
    assert "oph_fpe/cosmology/a5_biposh_continuum_tail.py" in labels
    assert (
        "oph_fpe/cosmology/verify_a5_biposh_continuum_tail_independent.py"
        in labels
    )

    child = inventory.verify_a5_biposh_continuum_tail_independent
    cache_attribute = inventory._TAIL_VERIFICATION_CACHE_ATTRIBUTE
    monkeypatch.setattr(child, cache_attribute, {}, raising=False)
    key = (("synthetic", 1, "sha256"),)
    monkeypatch.setattr(inventory, "_tail_verification_key", lambda: key)
    monkeypatch.setattr(independent, "_tail_verification_key", lambda: key)
    calls: list[bool] = []

    def fake_verify() -> dict:
        calls.append(True)
        return {
            "schema": "oph.a5-biposh-continuum-tail.v1",
            "status": (
                "CONDITIONAL_EQUAL_SEAM_CONTINUUM_L6_NONZERO_UNDER_DECLARED_"
                "NUMERICAL_ENVELOPE__SOURCE_SELECTION_AND_PHYSICAL_TRANSFER_OPEN"
            ),
        }

    monkeypatch.setattr(child, "verify_packet", fake_verify)
    inventory._verify_tail_packet_once_per_content()
    independent._verify_tail_packet_once_per_content()
    assert calls == [True]


def test_expensive_inverse_replay_is_cached_by_complete_content_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        inventory._inverse_verification_key()
        == independent._inverse_verification_key()
    )
    labels = {row[0] for row in inventory._inverse_verification_key()}
    assert "data/refinement/a5_biposh_inverse_continuum_gate_receipt.json" in labels
    assert "data/refinement/all_level_primitive_seam_source_receipt.json" in labels
    assert "oph_fpe/cosmology/a5_biposh_inverse_continuum_gate.py" in labels
    assert (
        "oph_fpe/cosmology/verify_a5_biposh_inverse_continuum_gate_independent.py"
        in labels
    )

    child = inventory.verify_a5_biposh_inverse_continuum_gate_independent
    cache_attribute = inventory._INVERSE_VERIFICATION_CACHE_ATTRIBUTE
    monkeypatch.setattr(child, cache_attribute, {}, raising=False)
    key = (("synthetic-inverse", 1, "sha256"),)
    monkeypatch.setattr(inventory, "_inverse_verification_key", lambda: key)
    monkeypatch.setattr(independent, "_inverse_verification_key", lambda: key)
    calls: list[bool] = []

    def fake_verify() -> dict:
        calls.append(True)
        return {
            "schema": "oph.a5-biposh-inverse-continuum-gate.v1",
            "status": (
                "FULL_RAW_STIFFNESS_CAUCHY_TAIL_ATTAINED__UNIFORM_COERCIVITY_"
                "PROJECTED_QUOTIENT_AND_PHYSICAL_RESPONSE_OPEN"
            ),
        }

    monkeypatch.setattr(child, "verify_packet", fake_verify)
    inventory._verify_inverse_packet_once_per_content()
    independent._verify_inverse_packet_once_per_content()
    assert calls == [True]


def test_tail_child_verifier_rejects_rehashed_physical_promotion(
    tmp_path: Path,
) -> None:
    child = inventory.verify_a5_biposh_continuum_tail_independent
    receipt = json.loads(child.DEFAULT_RECEIPT.read_text(encoding="utf-8"))
    receipt["selection_decision"]["physical_prediction"] = True
    receipt.pop("payload_sha256", None)
    payload = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    receipt["payload_sha256"] = "sha256:" + hashlib.sha256(payload).hexdigest()
    path = tmp_path / "tampered_tail_receipt.json"
    path.write_text(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(child.VerificationError, match="selection boundary"):
        child.verify_packet(path)


@pytest.mark.parametrize(
    "claim_field",
    inventory.BIPOSH_COEFFICIENT_FORBIDDEN_CLAIM_FIELDS,
)
def test_biposh_coefficient_bundle_rejects_top_level_claim_fields(
    claim_field: str,
) -> None:
    path = "data/refinement/a5_biposh_dual_operator_coefficients.json"
    bundle = json.loads((inventory.REPOSITORY_ROOT / path).read_text("utf-8"))
    bundle[claim_field] = False
    with pytest.raises(ValueError, match="forbidden claim fields"):
        inventory._critical_evidence(path, bundle)
    with pytest.raises(ValueError, match="forbidden claim fields"):
        independent._evidence(path, bundle)


def test_admission_counts_and_boundary_are_scope_qualified() -> None:
    admission = REPORT["bridge_admission_contract"]
    assert admission["registered_packet_count_excluding_recursive_outputs"] == 0
    assert admission["true_promotion_signal_path_count_excluding_recursive_outputs"] == 0
    assert admission["accepted_bridge_count_excluding_recursive_outputs"] == 0
    assert admission["recursive_parent_bridge_receipt_exclusion"] == {
        "path": inventory.BRIDGE_RELATIVE_PATH,
        "reason": (
            "parent output embeds the current negative source packet and is "
            "excluded to avoid recursive custody"
        ),
        "packet_count_included_in_scan": False,
    }

    boundary = REPORT["epistemic_boundary"]
    assert boundary["local_spatial_or_kinetic_operators_exist"] is True
    assert boundary[
        "twelve_port_internal_seam_response_and_in_process_snapshot_reread_exist"
    ] is True
    assert boundary["claim_that_no_spatial_operator_exists"] is False
    assert boundary[
        "registered_accepted_same_domain_chain_on_scanned_surface_exists"
    ] is False
    assert boundary["unregistered_equivalent_semantics_ruled_out"] is False
    assert boundary["producer_code_or_sibling_repository_absence_claimed"] is False
    assert boundary["physical_prediction_unsealed"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "disposition",
        "critical_evidence",
        "transport_feasibility_promotion",
        "biposh_promotion",
        "equal_seam_source_promotion",
        "all_level_seam_physical_promotion",
        "continuum_tail_physical_promotion",
        "inverse_continuum_physical_promotion",
        "endpoint_commutator_physical_promotion",
        "constructive_source_law_physical_promotion",
        "schema",
        "scope",
        "issue",
        "required_chain",
        "raw_pin_path",
        "duplicated_row",
        "qualified_count",
        "recursive_exclusion",
    ],
)
def test_rehashed_semantic_mutations_fail_independent_verification(
    mutation: str,
) -> None:
    report = copy.deepcopy(REPORT)
    charged = _row(report, "data/common_reserve/charged_response_artifact.json")
    if mutation == "disposition":
        charged["disposition"] = "PROMOTED"
    elif mutation == "critical_evidence":
        charged["critical_bridge_evidence"][
            "emitted_current_lift_source_selected"
        ] = True
    elif mutation == "transport_feasibility_promotion":
        feasibility = _row(
            report,
            "data/repair_closure/vertex12_directed_transport_feasibility_receipt.json",
        )
        feasibility["critical_bridge_evidence"][
            "emitted_requested_source_transport_ledger"
        ] = True
    elif mutation == "biposh_promotion":
        biposh = _row(
            report,
            "data/refinement/a5_biposh_dual_operator_receipt.json",
        )
        biposh["critical_bridge_evidence"]["emitted_promotion_allowed"] = True
    elif mutation == "equal_seam_source_promotion":
        equal_seam = _row(
            report,
            "data/refinement/refined_equal_seam_source_gate_receipt.json",
        )
        equal_seam["critical_bridge_evidence"][
            "emitted_all_level_atomic_counting_law_source"
        ] = True
    elif mutation == "all_level_seam_physical_promotion":
        all_level = _row(
            report,
            "data/refinement/all_level_primitive_seam_source_receipt.json",
        )
        all_level["critical_bridge_evidence"][
            "emitted_physical_prediction"
        ] = True
    elif mutation == "continuum_tail_physical_promotion":
        tail = _row(
            report,
            "data/refinement/a5_biposh_continuum_tail_receipt.json",
        )
        tail["critical_bridge_evidence"]["emitted_physical_prediction"] = True
    elif mutation == "inverse_continuum_physical_promotion":
        inverse = _row(
            report,
            "data/refinement/a5_biposh_inverse_continuum_gate_receipt.json",
        )
        inverse["critical_bridge_evidence"]["emitted_physical_prediction"] = True
    elif mutation == "endpoint_commutator_physical_promotion":
        endpoint = _row(
            report,
            "data/repair_closure/vertex12_a2_endpoint_commutator_receipt.json",
        )
        endpoint["critical_bridge_evidence"][
            "emitted_faithful_physical_z_power_6_action"
        ] = True
    elif mutation == "constructive_source_law_physical_promotion":
        constructive = _row(
            report,
            "data/repair_closure/vertex12_constructive_source_law_receipt.json",
        )
        constructive["critical_bridge_evidence"][
            "emitted_advances_physical_bridge"
        ] = True
    elif mutation == "schema":
        charged["schema"] = "oph.mutated.v1"
    elif mutation == "scope":
        report["scope"] = "all files everywhere"
    elif mutation == "issue":
        report["issue"] = 0
    elif mutation == "required_chain":
        report["bridge_admission_contract"]["required_chain"].pop()
    elif mutation == "raw_pin_path":
        charged["raw_pin"]["path"] = (
            "data/local_domain/source_gap_receipt.json"
        )
    elif mutation == "duplicated_row":
        report["canonical_artifact_rows"].append(copy.deepcopy(charged))
    elif mutation == "qualified_count":
        report["bridge_admission_contract"][
            "registered_packet_count_excluding_recursive_outputs"
        ] = 1
    elif mutation == "recursive_exclusion":
        report["bridge_admission_contract"][
            "recursive_parent_bridge_receipt_exclusion"
        ]["packet_count_included_in_scan"] = True
    else:  # pragma: no cover
        raise AssertionError(mutation)
    _rehash(report)

    result = independent.verify(report)
    assert result["receipt"] is False
    assert result["status"] == "FAIL"


def test_untracked_and_unstaged_current_input_gates_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        inventory,
        "_untracked_data_paths",
        lambda: ["data/unregistered.json"],
    )
    with pytest.raises(ValueError, match="untracked data paths"):
        inventory.build_inventory()

    monkeypatch.setattr(inventory, "_untracked_data_paths", lambda: [])
    monkeypatch.setattr(
        inventory,
        "_unstaged_current_inputs",
        lambda: ["data/local_domain/source_gap_receipt.json"],
    )
    with pytest.raises(ValueError, match="unstaged current canonical inputs"):
        inventory.build_inventory()


def test_schema_status_drift_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    original_load = inventory._load_json

    def load_with_drift(path: str) -> dict:
        value = original_load(path)
        if path == "data/common_reserve/charged_response_artifact.json":
            value["schema"] = "oph.mutated.v1"
        return value

    monkeypatch.setattr(inventory, "_load_json", load_with_drift)
    with pytest.raises(ValueError, match="canonical schema/status drift"):
        inventory.build_inventory()


def test_fresh_process_verifier_observes_the_filesystem(tmp_path: Path) -> None:
    report_path = tmp_path / "source_operator_ancestry_inventory.json"
    report_path.write_text(
        json.dumps(REPORT, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "oph_fpe.dynamics.verify_source_operator_inventory_independent",
            str(report_path),
        ],
        cwd=inventory.REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["receipt"] is True

    mutated = copy.deepcopy(REPORT)
    mutated["issue"] = 0
    _rehash(mutated)
    report_path.write_text(
        json.dumps(mutated, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "oph_fpe.dynamics.verify_source_operator_inventory_independent",
            str(report_path),
        ],
        cwd=inventory.REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert json.loads(completed.stdout)["receipt"] is False


def test_writer_and_canonical_receipt_are_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "source_operator_ancestry_inventory.json"
    written = inventory.write_inventory(output)
    assert written == REPORT
    assert json.loads(output.read_text(encoding="utf-8")) == REPORT
    canonical = json.loads(inventory.OUTPUT_PATH.read_text(encoding="utf-8"))
    assert canonical == REPORT
    assert inventory.verify_inventory(canonical)["receipt"] is True
    assert independent.verify(canonical)["receipt"] is True
