from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import oph_fpe.bulk.physical_h3_kms_preflight as preflight_module
from oph_fpe.bulk.bw_certificate_308 import MGNS1_REQUIRED_FIELDS
from oph_fpe.bulk.bw_native_preflight import (
    BW_NATIVE_SCHEMA_VERSION,
    BW_PRIMITIVE_FIELD_CONTRACT,
    canonical_payload_hash,
)
from oph_fpe.bulk.physical_h3_kms_preflight import (
    CellStatus,
    DEFAULT_INSTRUMENT_VERSION,
    PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT,
    REQUIRED_CARRIER_COUNTS,
    REQUIRED_CLOCK_LABELS,
    REQUIRED_GEOMETRY_MODELS,
    REQUIRED_RUNGS,
    physical_h3_kms_preflight_report,
)
from oph_fpe.bulk.physical_h3_kms_prerun import (
    REGISTERED_HISTORICAL_16K_SOURCE_SEED,
    REGISTERED_HISTORICAL_CAMPAIGN_SHA256,
    REGISTERED_HISTORICAL_RECEIPT_BYTE_SHA256,
)
from oph_fpe.repair import build_repair_replay_envelope

SUPPORT_REGULATOR_CELL_COUNTS = (5_120, 20_480, 81_920, 327_680)


def _hash(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('utf-8')).hexdigest()}"


def _primitive_bwrec() -> dict:
    spatial_normal = math.sqrt(1.0 + 0.1**2)
    boundary_z = 0.1 / spatial_normal
    boundary_x = math.sqrt(1.0 - boundary_z**2)
    return {
        "finite_cap_source_role": "geometric_support_flow",
        "finite_cap_source_artifact_hash": f"sha256:{'1' * 64}",
        "finite_cap_tower_id": "tower-v1",
        "finite_cap_tower_hash": f"sha256:{'2' * 64}",
        "cap_normal": [0.1, 0.0, 0.0, spatial_normal],
        "cap_normal_norm_residual": 0.0,
        "cap_orientation": "interior_positive",
        "cap_radius_margin": 0.25,
        "cap_boundary_incidence_residual": 0.0,
        "cap_sign_violation": 0.0,
        "cap_mesh_error": 1.0e-4,
        "point_mesh_error": 1.0e-4,
        "refinement_normal_error": 0.0,
        "frame_p_minus": [boundary_x, 0.0, boundary_z],
        "frame_p_plus": [-boundary_x, 0.0, boundary_z],
        "frame_boundary_residual": 0.0,
        "frame_separation": 2.0 * boundary_x,
        "frame_ordering": "p_minus_attracting_for_positive_s",
        "frame_orientation_witness": True,
        "cap_inclusion_matrix": [[1, 0], [0, 1]],
        "strict_inclusion_margin": 0.1,
        "order_refinement_error": 0.0,
        "support_isotony_failures": 0,
        "support_separation_margin": 0.1,
        "support_covariance_residual_T": 1.0e-7,
        "support_kernel_residual": 0.0,
        "sector_scope": "PRIME_GEOMETRIC_SUPPORT_VISIBLE",
        "flow_identity_residual": 0.0,
        "flow_group_residual_T": 1.0e-7,
        "flow_inverse_residual_T": 1.0e-7,
        "flow_equi_continuity_bound": 2.0,
        "cap_anchor_residual": 0.0,
        "frame_fixed_point_residual": 0.0,
        "cross_ratio_holdout_max": 1.0e-7,
        "quartet_separation_min": 0.25,
        "cross_ratio_anchor_condition": 3.0,
        "orientation_witness": True,
        "geometric_parameter_convention": "h_C(z) -> e^{-s} h_C(z)",
        "kms_comparison_state_id": "finite-state-r128",
        "kms_comparison_state_hash": f"sha256:{'6' * 64}",
        "kms_matrix_element_residual_T": 1.0e-7,
        "kms_strip_bound": 10.0,
        "kms_residual_beta_2pi": 1.0e-7,
        "geometric_flow_nontrivial": True,
        "wrong_beta_interval": [1.0, 10.0],
        "wrong_beta_gap_delta": 0.05,
        "geometric_generator_noncentrality": 0.2,
        "generator_distance_beta_2pi": 1.0e-7,
        "total_308_error_envelope": 5.0e-7,
        "error_envelope_samples": [1.0e-5, 5.0e-7],
        "error_envelope_refinement_levels": [64, 128],
        "error_envelope_refinement_witness": True,
    }


def _primitive_mgns1() -> dict:
    return {
        "certificate_kind": "MGNS-1",
        "source_role": "algebra_state_tower",
        "source_artifact_hash": f"sha256:{'4' * 64}",
        "tower_id": "tower-v1",
        "tower_hash": f"sha256:{'2' * 64}",
        "fixed_local_algebra_ids": ["A64", "A128"],
        "fine_to_coarse_embedding_residual_T": 1.0e-7,
        "state_restriction_residual_T": 1.0e-7,
        "expectation_idempotence_residual_T": 1.0e-7,
        "expectation_state_preservation_residual_T": 1.0e-7,
        "comparison_map_isometry_residual_T": 1.0e-7,
        "cyclic_vector_residual_T": 1.0e-7,
        "separating_vector_margin": 0.1,
        "state_vector_compatibility_residual_T": 1.0e-7,
        "density_matrix_trace_residual_T": 1.0e-7,
        "regularizer_eta": 1.0e-5,
        "regularization_schedule": [1.0e-3, 1.0e-4],
        "state_ids_by_level": ["finite-state-r64", "finite-state-r128"],
        "state_fingerprints_by_level": [f"sha256:{'5' * 64}", f"sha256:{'6' * 64}"],
        "mixed_gns_cauchy_residual_T": 1.0e-7,
        "negative_time_residual_T": 1.0e-7,
        "matrix_element_residual_T": 1.0e-7,
        "modular_identity_residual": 1.0e-7,
        "modular_group_residual_T": 1.0e-7,
        "modular_inverse_residual_T": 1.0e-7,
        "modular_support_covariance_residual_T": 1.0e-7,
        "cap_family_uniformity_bound_T": 1.0e-7,
        "cofinal_cauchy_modulus_T": 1.0e-7,
    }


def _native_bw_payload() -> dict:
    fields = _primitive_bwrec()
    antecedent_hash = _hash("one-common-native-bw-antecedent")
    clauses = {}
    for clause_id, field_names in BW_PRIMITIVE_FIELD_CONTRACT.items():
        primitive_fields = {name: fields[name] for name in field_names}
        clauses[clause_id] = {
            "antecedent_hash": antecedent_hash,
            "primitive_artifact_hash": canonical_payload_hash(primitive_fields),
            "primitive_fields": primitive_fields,
        }
    return {
        "schema_version": BW_NATIVE_SCHEMA_VERSION,
        "producer_kind": "native_simulator",
        "source_kind": "physical_source_generation",
        "antecedent_hash": antecedent_hash,
        "clauses": clauses,
        "mgns1": {
            "antecedent_hash": antecedent_hash,
            "primitive_artifact_hash": canonical_payload_hash(_primitive_mgns1()),
            "primitive_fields": _primitive_mgns1(),
        },
    }


def test_native_bw_fixture_tracks_exact_finite_cap_and_mgns1_contract() -> None:
    finite_cap_fields = {
        field_name
        for field_names in BW_PRIMITIVE_FIELD_CONTRACT.values()
        for field_name in field_names
    }

    assert set(_primitive_bwrec()) == finite_cap_fields
    assert set(_primitive_mgns1()) == set(MGNS1_REQUIRED_FIELDS)


def _repair_replay_envelope() -> dict:
    zero = {"terms": [], "constant": 0, "transform": "identity"}
    components = {
        name: dict(zero)
        for name in ("record", "sector", "holonomy", "local_constraint")
    }
    components["overlap"] = {
        "terms": [{"register": "x", "coefficient": 1}],
        "constant": 0,
        "transform": "absolute",
    }
    return build_repair_replay_envelope(
        initial_state={"x": 2},
        initial_versions={"x": 0},
        mismatch_evaluator={
            "kind": "exact_affine_ledger_v1",
            "components": components,
            "physical_auxiliary": {},
        },
        proposals=[
            {
                "proposal_id": "preflight-repair",
                "transition_kind": "STRICT_REPAIR",
                "proposal_class": "EXACT_SPLICE",
                "collar": {
                    "collar_id": "preflight-collar",
                    "visible_read_set": ["x"],
                    "writable_registers": ["x"],
                    "protected_boundary": [],
                    "sector_registers": [],
                    "record_registers": [],
                    "checkpoint_registers": [],
                    "interior_registers": ["x"],
                    "carrier_ids": [],
                    "seam_ids": [],
                    "forbidden_target_fields": [],
                },
                "declared_read_set": ["x"],
                "recovery": {
                    "kind": "literal_updates_v1",
                    "updates": {"x": 1},
                },
                "inverse_updates": {},
                "source_parameters": {},
                "parent_event_ids": [],
            }
        ],
    )


def _conforming_bundle() -> dict:
    levels = [
        {
            "level_id": f"r{rung}",
            "patch_count": SUPPORT_REGULATOR_CELL_COUNTS[index],
            **(
                {}
                if index == 0
                else {
                    "parent_level_id": f"r{REQUIRED_RUNGS[index - 1]}",
                    "lineage_hash": f"sha256:lineage-{rung}",
                }
            ),
        }
        for index, rung in enumerate(REQUIRED_RUNGS)
    ]
    expectations = [
        {
            "fine_level_id": f"r{REQUIRED_RUNGS[index]}",
            "coarse_level_id": f"r{REQUIRED_RUNGS[index - 1]}",
            "operator_hash": f"sha256:expectation-{index}",
            "unital": True,
            "positive": True,
            "state_preserving": True,
            "cap_isotony_compatible": True,
            "noncommutative_prime_cap_expectation": True,
        }
        for index in range(1, len(REQUIRED_RUNGS))
    ]
    common_candidate_payload = {
        "intervention_row_ids": ["event-1", "event-2", "event-3"],
        "heldout_event_row_ids": ["event-2", "event-3"],
        "intervention_packet_hash": _hash("one-frozen-intervention-packet"),
        "source_trajectory_hash": _hash("one-frozen-source-trajectory"),
        "raw_response_hash": _hash("one-frozen-response-matrix"),
    }
    candidate_aggregate_hash = canonical_payload_hash(common_candidate_payload)
    candidate_values = {
        "1x": 1.0,
        "pi": math.pi,
        "2pi": 2.0 * math.pi,
        "4pi": 4.0 * math.pi,
    }
    candidates = {
        label: {
            "intervention_row_ids": ["event-1", "event-2", "event-3"],
            "heldout_event_row_ids": ["event-2", "event-3"],
            "intervention_packet_hash": common_candidate_payload["intervention_packet_hash"],
            "source_trajectory_hash": common_candidate_payload["source_trajectory_hash"],
            "raw_response_hash": common_candidate_payload["raw_response_hash"],
            "candidate_invariance_aggregate_hash": candidate_aggregate_hash,
            "candidate_scale_applied_only_in_scoring": True,
            "candidate_scale_enters_intervention": False,
            "candidate_parameter_name": "kappa",
            "candidate_value": candidate_values[label],
            "candidate_units": "dimensionless_geometric_flow_parameter",
        }
        for label in REQUIRED_CLOCK_LABELS
    }
    shared_control_hashes = {
        "heldout_event_matrix_hash": _hash("shared-heldout-event-matrix"),
        "heldout_weights_hash": _hash("shared-heldout-weights"),
        "missingness_mask_hash": _hash("shared-missingness-mask"),
        "preprocessing_hash": _hash("shared-preprocessing"),
        "source_packet_hash": common_candidate_payload["intervention_packet_hash"],
        "prediction_target_hash": _hash("shared-prediction-target"),
        "fit_protocol_hash": _hash("shared-fit-protocol"),
    }
    models = {
        model: {
            "heldout_event_row_ids": ["event-2", "event-3"],
            **shared_control_hashes,
            "effective_model_capacity": 12,
            "heldout_score": {"H3": 0.1, "S2": 0.4, "E3": 0.3, "E4": 0.5}[model],
            "optimizer_status": "CONVERGED",
            "required_rows_complete": True,
        }
        for model in REQUIRED_GEOMETRY_MODELS
    }
    seeds = [20_260_751, 20_260_752, 20_260_753]
    family_contract = {
        "instrument_commit": "abcdef0123456789",
        "container_digest": _hash("container-image"),
        "schema_versions": {"preflight": "v2", "bw": "v1", "event": "v1"},
        "source_protocol": {"generator": "target_blind_v1"},
        "feature_contract": {"prime_cap_only": True},
        "model_families": {"geometry": list(REQUIRED_GEOMETRY_MODELS)},
        "loss_functions": {"clock": "paired_holdout", "geometry": "paired_holdout"},
        "thresholds": {"clock_delta": 0.1, "absolute_residual": 0.2},
        "control_set": {
            "clock_candidates": list(REQUIRED_CLOCK_LABELS),
            "geometry_models": list(REQUIRED_GEOMETRY_MODELS),
        },
        "split_algorithm": {"name": "semantic_hash_split_v1"},
        "seed_derivation": {"name": "sha256_domain_separated_v1"},
        "rung_scaling_laws": {
            "carrier_count": "exact_federation_cardinality",
            "support_regulator": "independent_nested_geodesic_chart",
            "repairs_per_cycle": "carrier_count*repairs_per_carrier_cycle",
            "cycles": "fixed",
        },
        "archive_boundary": {
            "archived_instrument_status": "FROZEN_NO_RETUNE",
            "archived_16k_failure_preserved": True,
            "new_instrument_is_distinct_family": True,
            "archived_outcomes_used_for_threshold_selection": False,
            "historical_receipt_byte_sha256": (
                REGISTERED_HISTORICAL_RECEIPT_BYTE_SHA256
            ),
            "historical_campaign_sha256": REGISTERED_HISTORICAL_CAMPAIGN_SHA256,
            "historical_16k_source_seed": REGISTERED_HISTORICAL_16K_SOURCE_SEED,
            "historical_16k_rung": 16_384,
            "historical_16k_joint_independent_receipt": False,
            "historical_stable_branch_failure_established": False,
        },
        "retirement_rule": {
            "decisive_rungs": [16_384, 65_536, 262_144],
            "same_predeclared_failure_mode_required": True,
            "all_cells_powered_and_complete_required": True,
            "frozen_before_first_run": True,
            "failure_mode_derivation": "verified_predicate_set_sha256_v1",
        },
    }
    family_hash = canonical_payload_hash(family_contract)
    run_matrix = [
        {
            "seed": seed,
            "rung": rung,
            "carrier_count": rung,
            "replicate_id": "primary",
            "campaign_id": "physical-h3-kms-frozen-family-001",
            "instrument_version": DEFAULT_INSTRUMENT_VERSION,
            "instrument_commit": family_contract["instrument_commit"],
            "family_hash": family_hash,
            "protocol_hash": _hash("frozen-physical-h3-kms-protocol"),
            "frozen_config_hash": _hash(f"config-{seed}-{rung}"),
            "preflight": "PASS",
            "required_controls_complete": False,
            "source_hashes_complete": False,
            "powered_and_complete": False,
            "status": CellStatus.NOT_EVALUATED.value,
            "failure_mode": None,
        }
        for seed in seeds
        for rung in REQUIRED_RUNGS
    ]
    return {
        "config": {
            "seed": seeds[0],
            "source_federation": {
                "family": "federated_echosahedral_carriers",
                "carrier_count": REQUIRED_RUNGS[0],
            },
            "support_regulator": {
                "family": "nested_geodesic_icosahedral",
                "patch_basis": "cells",
                "refinement_level": 4,
            },
            "bw": {"state_mode": "prime_geometric_cap_maxent"},
        },
        "reports": {
            "source_observer": {
                "schema_version": "oph_source_repair_record_observer_contract_v2",
                "SOURCE_PATCH_ARCHITECTURE_RECEIPT": True,
                "PATCH_LOCAL_STATE_RECEIPT": True,
                "PATCH_PORT_BOUNDARY_RECEIPT": True,
                "PATCH_READBACK_RECEIPT": True,
                "PATCH_ALL_PORT_READBACK_RECEIPT": True,
                "RECORD_SIGNATURE_BINDS_ALL_LOCAL_PORT_STATE_RECEIPT": True,
                "ECHOSAHEDRAL_LOCAL_PATCH_ARCHITECTURE_RECEIPT": True,
                "ECHOSAHEDRAL_CARRIER_CONFORMANCE": True,
                "FEDERATION_SEWING_RECEIPT": True,
                "CARRIER_QUOTIENT_INVARIANCE_RECEIPT": True,
                "CARRIER_REFINEMENT_NATURALITY_RECEIPT": True,
                "TRANSACTION_VALIDATION_COMPLETE_READ_CONFLICT_SET_RECEIPT": True,
                "UNION_PAYLOAD_ATOMIC_REVALIDATION_RECEIPT": True,
                "source_generator_target_free": True,
                "source_forbidden_target_hits": [],
                "source_architecture": {
                    "bounded_patch_system": True,
                    "simulation_native_source": True,
                    "carrier_count": REQUIRED_CARRIER_COUNTS[0],
                    "local_state_dimension": 6**12,
                    "local_state_factor_count": 12,
                    "materialized_local_state_coordinate_count": (
                        REQUIRED_CARRIER_COUNTS[0] * 12
                    ),
                    "boundary_port_count": 12,
                    "carrier_family": "federated_echosahedral_patch_system",
                    "one_local_echosahedron_per_carrier": True,
                    "carrier_is_not_support_chart_cell": True,
                    "carrier_is_not_primitive_observer": True,
                    "all_local_port_readout_maps_materialized": True,
                    "all_local_port_states_bound_into_records": True,
                    "local_patch_template_hash": _hash("local-echosahedral-template"),
                    "patch_port_state_sha256": _hash("patch-by-twelve-state"),
                    "source_architecture_hash": "sha256:bounded-patch-source-architecture",
                },
                "repair_dynamics": {
                    "local_update_rule": True,
                    "uses_only_local_state_and_ports": True,
                    "target_free_rule": True,
                    "repair_event_count": 128,
                    "nonlocal_write_count": 0,
                    "repair_rule_hash": "sha256:local-target-free-repair-rule",
                    "repair_event_log_hash": "sha256:repair-event-log",
                },
                "record_observer": {
                    "observer_count": 32,
                    "committed_record_count": 96,
                    "readback_count": 80,
                    "feedback_event_count": 64,
                    "readback_changes_future_local_actions": True,
                    "records_causally_bound_to_writes": True,
                    "bounded_interface_verified": True,
                    "self_prediction_beats_shuffled_control": True,
                    "feedback_ablation_changes_future_actions": True,
                    "checkpoint_continuation_verified": True,
                    "orphan_read_count": 0,
                    "record_readback_feedback_log_hash": "sha256:record-readback-feedback-log",
                },
            },
            "refinement": {
                "mesh_family": "nested_geodesic_icosahedral",
                "nested_lineage_receipt": True,
                "conditional_expectations_receipt": True,
                "TRUE_ICOSAHEDRAL_REFINEMENT_TOWER_RECEIPT": True,
                "A5_EQUIVARIANT_REFINEMENT_RECEIPT": True,
                "PAPER_MULTIRESOLUTION_REGULATOR_CERTIFICATE": True,
                "levels": levels,
                "conditional_expectations": expectations,
            },
            "prime_geometric_state": {
                "state_mode": "prime_geometric_cap_maxent",
                "algebra_scope": "prime_geometric_cap_interior",
                "state_construction": "source_maxent_cap_state",
                "noncommutative_algebra": True,
                "commutator_norm": 0.25,
                "source_primitive_fields": [
                    "cap_incidence",
                    "cap_orientation",
                    "cap_isotony",
                    "boundary_frame",
                ],
                "surrogate_inputs": [],
                "rho": {
                    "dimension": 4,
                    "trace": 1.0,
                    "minimum_eigenvalue": 0.05,
                    "hermiticity_residual": 1.0e-12,
                    "matrix_hash": "sha256:rho-matrix-source-maxent",
                },
                "modular_generator": {
                    "construction": "negative_log_density_matrix",
                    "dimension": 4,
                    "functional_calculus_residual": 1.0e-12,
                    "noncentrality_norm": 0.4,
                    "matrix_hash": "sha256:negative-log-rho-matrix",
                },
                "mixed_gns": {
                    "constructed": True,
                    "left_right_representation": True,
                    "cyclic_separating_support": True,
                },
            },
            "independent_geometry": {
                "derivation_method": "ordered_bw_frame_cross_ratio",
                "source_primitive_fields": [
                    "cap_incidence",
                    "orientation",
                    "ordered_bw_frame",
                    "cross_ratio",
                ],
                "derivation_expression": "s=log(cross_ratio_from_incidence_and_ordered_frame)",
                "forbidden_token_hits": [],
                "target_blind_derivation": True,
                "independent_of_modular_fit": True,
                "independent_of_kms_target": True,
                "orientation_fixed_from_source": True,
                "geometric_parameter_values": [-0.75, 0.0, 0.5, 1.25],
                "geometry_source_row_ids": ["geometry-1", "geometry-2"],
                "kms_score_row_ids": ["event-2", "event-3"],
                "geometry_derivation_hash": "sha256:source-only-geometric-rapidity",
            },
            "candidate_interventions": {
                "interventions_frozen_before_candidate_scoring": True,
                "candidate_labels_frozen_before_runs": True,
                "source_intervention_target_free": True,
                "candidate_invariance_aggregate_hash": candidate_aggregate_hash,
                "candidates": candidates,
                "continuous_clock_fit": {
                    "fitted_kappa_interval": [6.2, 6.36],
                    "absolute_residual": 0.05,
                    "frozen_absolute_residual_threshold": 0.2,
                    "wrong_normalization_separation_passed": True,
                    "refinement_tail_stable": True,
                    "fit_artifact_hash": _hash("continuous-clock-fit"),
                },
                "discrete_clock_comparison": {
                    "paired_losses": {"1x": 0.8, "pi": 0.6, "2pi": 0.1, "4pi": 0.9},
                    "frozen_delta_clock": 0.1,
                    "paired_uncertainty_upper_bound": 0.05,
                    "same_rows_and_packets": True,
                    "thresholds_from_independent_synthetic_calibration": True,
                    "thresholds_frozen_before_physical_campaign": True,
                    "calibration_artifact_hash": _hash("clock-instrument-calibration"),
                },
            },
            "native_bw": _native_bw_payload(),
            "geometry_controls": {
                "models_frozen_before_holdout": True,
                "heldout_excluded_from_model_selection": True,
                "models": models,
                "paired_geometry_comparison": {
                    "loss_direction": "lower_is_better",
                    "frozen_h3_win_margin": 0.05,
                    "paired_uncertainty_upper_bound": 0.02,
                    "thresholds_frozen_before_physical_campaign": True,
                    "physical_threshold_calibration_receipt": True,
                    "calibration_artifact_hash": _hash(
                        "paired-geometry-calibration"
                    ),
                },
                "curvature_leverage": {
                    "calibration_source": "independent_synthetic_power_suite",
                    "physical_threshold_calibration_receipt": True,
                    "frozen_before_physical_campaign": True,
                    "calibration_hash": _hash("curvature-power-calibration"),
                    "registered_analysis_hash": _hash("curvature-analysis"),
                    "domain_diameter": 2.0,
                    "registered_curvature_radius": 1.0,
                    "noise_scale": 0.01,
                    "sample_count": 2_000,
                    "calibrated_power": 0.95,
                    "minimum_power": 0.9,
                    "curvature_radius_source": "independent_source",
                    "curvature_radius_frozen_before_h3_fit": True,
                    "flat_limit_excluded": True,
                    "curvature_parameter_charged_to_model_capacity": False,
                    "h3_e3_distinguishable_at_registered_effect": True,
                },
            },
            "semantic_event": {
                "event_clauses": {
                    "EVENT_E1_POPULATION": {
                        "semantic_record_germ_count": 40,
                        "certified_localization_box_count": 40,
                        "dense_population_verified": True,
                        "shrinking_box_sequence_verified": True,
                        "population_artifact_hash": _hash("event-e1-population"),
                    },
                    "EVENT_E2_SEPARATION": {
                        "distinct_germ_pair_count": 32,
                        "separated_pair_count": 32,
                        "minimum_localization_gap": 0.02,
                        "disjoint_certified_boxes_verified": True,
                        "separation_artifact_hash": _hash("event-e2-separation"),
                    },
                    "EVENT_E3_RANK_FOUR": {
                        "conditioned_spatial_response_rank": 3,
                        "independent_clock_line_rank": 1,
                        "combined_event_rank": 4,
                        "clock_line_independent_of_spatial_response": True,
                        "independent_clock_receipt": True,
                        "rank_four_artifact_hash": _hash("event-e3-rank-four"),
                    },
                    "EVENT_E4_POINCARE_COCYCLE": {
                        "overlap_transition_count": 20,
                        "lorentz_components_present": True,
                        "translation_components_present": True,
                        "poincare_cocycle_residual": 1.0e-12,
                        "connected_overlap_atlas": True,
                        "poincare_transition_artifact_hash": _hash(
                            "event-e4-poincare-transitions"
                        ),
                    },
                },
                "semantic_event_dag": {
                    "identity_fields": [
                        "canonical_semantic_payload",
                        "observer_token",
                        "visible_footprint",
                        "semantic_causal_parents",
                    ],
                    "forbidden_identity_fields_present": [],
                    "semantic_parent_edge_count": 31,
                    "acyclic": True,
                    "causal_cycle_count": 0,
                    "duplicate_semantic_event_count": 0,
                    "preassigned_metric_used_for_identity": False,
                    "semantic_event_dag_hash": _hash("semantic-event-dag"),
                },
                "causal_ancestry": {
                    "committed_read_after_write_edge_count": 31,
                    "translation_edge_ancestry_coverage_fraction": 1.0,
                    "population_used_as_reachability_surrogate": False,
                    "ancestry_artifact_hash": _hash("semantic-causal-ancestry"),
                },
                "quadratic_event_cone": {
                    "fit_row_ids": ["cone-train-1", "cone-train-2"],
                    "heldout_row_ids": ["cone-test-1", "cone-test-2"],
                    "inference_source": "semantic_event_relations",
                    "preassigned_lorentz_metric_used": False,
                    "ambient_rank": 4,
                    "negative_eigenvalue_count": 1,
                    "positive_eigenvalue_count": 3,
                    "zero_eigenvalue_count": 0,
                    "heldout_quadratic_residual": 1.0e-4,
                    "frozen_residual_threshold": 1.0e-3,
                    "time_orientation_consistent": True,
                    "cofinal_normalized_margin_lower_bound": 0.1,
                    "cofinal_tail_level_count": 3,
                    "cone_inference_artifact_hash": _hash(
                        "heldout-quadratic-event-cone"
                    ),
                },
                "stable_causality": {
                    "time_function_source": "source_derived_semantic_ancestry",
                    "minimum_causal_edge_increment": 0.02,
                    "time_function_perturbation_margin": 0.01,
                    "stable_causality_artifact_hash": _hash("stable-causality"),
                },
                "record_cauchy_completion": {
                    "refinement_cauchy_residual": 1.0e-5,
                    "frozen_residual_threshold": 1.0e-4,
                    "every_cauchy_filter_has_record_germ": True,
                    "open_image_local_degree_nonzero": True,
                    "completion_artifact_hash": _hash("record-cauchy-completion"),
                },
                "h3_role": "observer_frame_fiber",
                "event_base_role": "event_position_manifold",
                "h3_and_event_base_separate": True,
                "frame_fiber_construction_hash": _hash("h3-frame-fiber-construction"),
                "event_base_construction_hash": _hash("event-position-construction"),
            },
            "campaign": {
                "campaign_id": "physical-h3-kms-frozen-family-001",
                "instrument_version": DEFAULT_INSTRUMENT_VERSION,
                "campaign_family_hash": family_hash,
                "family_contract": family_contract,
                "seeds": seeds,
                "rungs": list(REQUIRED_RUNGS),
                "carrier_counts": list(REQUIRED_CARRIER_COUNTS),
                "replicate_ids": ["primary"],
                "frozen_before_first_run": True,
                "retune_after_freeze": False,
                "retune_events": [],
                "run_matrix": run_matrix,
            },
        },
    }


def _replay_bound_not_evaluated_ledger() -> dict:
    stage_ids = (
        "P1_nested_refinement_and_expectations",
        "P2_prime_geometric_cap_state",
        "P3_independent_geometric_parameter",
        "P4_native_bw01_bw08",
        "P6_h3_s2_e3_e4_same_holdout_and_curvature_leverage",
        "P7_semantic_event_e1_e4_and_frame_fiber_separation",
        "P8_frozen_multiseed_four_rung_campaign",
    )
    ledger = {
        stage_id: {
            "measurement_status": CellStatus.NOT_EVALUATED.value,
            "physical_gate_eligible": False,
            "not_evaluated_reasons": [f"{stage_id}_independent_producer_missing"],
            "scientific_failures": [],
            "structural_receipt": {
                "scope": "artifact_integrity_only",
                "status": "COMPLETE",
                "artifact_hash": _hash(f"structure:{stage_id}"),
                "physical_claim": False,
            },
            "instrument_receipt": {
                "scope": "independent_physical_producer",
                "status": "MISSING_REQUIRED_PRODUCER",
                "physical_claim": False,
            },
            "sensitivity_receipt": {
                "scope": "diagnostic_sensitivity_only",
                "status": "COMPLETE",
                "artifact_hash": _hash(f"sensitivity:{stage_id}"),
                "physical_gate_eligible": False,
                "diagnostic_findings": [f"{stage_id}_diagnostic_predicate_false"],
            },
        }
        for stage_id in stage_ids
    }
    ledger["P5_frozen_candidate_interventions"] = {
        "measurement_status": CellStatus.NOT_EVALUATED.value
    }
    return ledger


def _trusted_epistemic_admission(ledger: dict) -> dict:
    return {
        "gate_status": "PASS",
        "PHYSICAL_ARTIFACT_REPLAY_ADMISSION_RECEIPT": True,
        "blockers": [],
        "replay_bound_postrun_stage_epistemics_receipt": True,
        "replay_bound_postrun_stage_epistemics": ledger,
        "registered_source_capture_replay_bound": True,
        "preflight_export_hash_rows": {
            "source_observer": {"exact": True},
        },
    }


def test_replay_bound_epistemics_quarantine_diagnostics_from_science(
    monkeypatch,
) -> None:
    ledger = _replay_bound_not_evaluated_ledger()
    monkeypatch.setattr(
        preflight_module,
        "_physical_artifact_admission",
        lambda config, provenance, evidence: _trusted_epistemic_admission(ledger),
    )

    report = physical_h3_kms_preflight_report(_conforming_bundle())

    for stage_id in preflight_module._REPLAY_BOUND_NOT_EVALUATED_STAGES:
        stage = report["stages"][stage_id]
        assert stage["passed"] is False
        assert stage["gate_status"] == "NOT_EVALUATED"
        assert stage["scientific_status"] == CellStatus.NOT_EVALUATED.value
        assert stage["scientific_failure_count"] == 0
        assert stage["hard_blockers"] == []
        assert stage["evidence"]["scientific_failures"] == []
        assert stage["evidence"]["diagnostic_findings"]
        assert stage["evidence"]["epistemic_status_source"] == (
            "content_addressed_exact_postrun_replay"
        )
    assert report["scientific_failures"] == []
    campaign_stage = report["stages"]["P8_frozen_multiseed_four_rung_campaign"]
    assert campaign_stage["gate_status"] == "NOT_EVALUATED"
    assert campaign_stage["passed"] is False
    assert report["campaign_status"] == CellStatus.INCOMPLETE.value
    assert report["physical_promotion_allowed"] is False
    assert report["retirement_counting_allowed"] is False


def test_replay_bound_source_gap_keeps_p0_structural_diagnostics_visible(
    monkeypatch,
) -> None:
    ledger = _replay_bound_not_evaluated_ledger()
    monkeypatch.setattr(
        preflight_module,
        "_physical_artifact_admission",
        lambda config, provenance, evidence: _trusted_epistemic_admission(ledger),
    )
    bundle = _conforming_bundle()
    source = bundle["reports"]["source_observer"]
    source.update(
        {
            "INDEPENDENT_SUPPORT_REGULATOR_RECEIPT": True,
            "CARRIER_REFINEMENT_NATURALITY_RECEIPT": False,
            "carrier_refinement_naturality_status": "NOT_EVALUATED",
            "PHYSICAL_ECHOSAHEDRAL_FEDERATION_REALIZATION_RECEIPT": False,
            "physical_federation_status": "NOT_EVALUATED",
            "CARRIER_TO_SUPPORT_CHART_REALIZATION_RECEIPT": False,
            "carrier_to_support_realization_status": "NOT_EVALUATED",
        }
    )

    report = physical_h3_kms_preflight_report(bundle)
    p0 = report["stages"]["P0_source_dynamics_repair_record_observer"]

    assert p0["gate_status"] == "NOT_EVALUATED"
    assert p0["scientific_status"] == CellStatus.NOT_EVALUATED.value
    assert p0["scientific_failure_count"] == 0
    assert p0["hard_blockers"] == []
    assert p0["evidence"]["dependency_status"] == {
        "SOURCE_PATCH_ARCHITECTURE": True,
        "LOCAL_REPAIR_DYNAMICS": True,
        "OBSERVER_SELF_READING_RECORD_LOOP": True,
        "SOURCE_GENERATOR_TARGET_FREE": True,
    }
    assert p0["evidence"]["structural_receipt"]["status"] == "COMPLETE"
    assert set(p0["evidence"]["not_evaluated_reasons"]) == {
        "physical_echosahedral_federation_realization_not_established",
        "carrier_to_support_chart_realization_not_established",
        "carrier_refinement_naturality_not_established",
    }
    assert report["scientific_failures"] == []
    assert report["retirement_counting_allowed"] is False


def test_stage_epistemics_loader_requires_manifest_bound_postrun_bytes(
    tmp_path: Path,
) -> None:
    ledger = _replay_bound_not_evaluated_ledger()
    postrun_bytes = (
        json.dumps(
            {"stage_epistemics": ledger},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    artifact = tmp_path / "artifacts" / "postrun_report.json"
    artifact.parent.mkdir()
    artifact.write_bytes(postrun_bytes)
    manifest = {
        "artifacts": {
            "postrun_report": {
                "path": "artifacts/postrun_report.json",
                "byte_sha256": "sha256:" + hashlib.sha256(postrun_bytes).hexdigest(),
                "byte_count": len(postrun_bytes),
            }
        }
    }
    manifest_bytes = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    manifest_path = tmp_path / "replay_manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    replay_result = {
        "manifest_byte_sha256": "sha256:"
        + hashlib.sha256(manifest_bytes).hexdigest(),
        "REPLAY_MANIFEST_VERIFICATION_RECEIPT": True,
        "PRE_SOURCE_FREEZE_REPLAY_RECEIPT": True,
        "HISTORICAL_16K_ARCHIVE_BYTE_REPLAY_RECEIPT": True,
        "POSTRUN_REPORT_EXACT_REPLAY_RECEIPT": True,
        "SOURCE_CAPTURE_REPLAY_RECEIPT": True,
        "SINGLE_BUNDLE_COMMITMENT_RECEIPT": True,
        "NUMERICAL_RUNTIME_REPLAY_RECEIPT": True,
    }

    loaded, bound = preflight_module._load_replay_bound_stage_epistemics(
        manifest_path, replay_result
    )
    assert bound is True
    assert loaded == ledger

    artifact.write_bytes(postrun_bytes + b" ")
    loaded, bound = preflight_module._load_replay_bound_stage_epistemics(
        manifest_path, replay_result
    )
    assert bound is False
    assert loaded == {}


def test_caller_authored_not_evaluated_ledger_is_not_trusted() -> None:
    bundle = _conforming_bundle()
    bundle["reports"]["stage_epistemics"] = _replay_bound_not_evaluated_ledger()
    bundle["reports"]["geometry_controls"]["measurement_status"] = (
        CellStatus.NOT_EVALUATED.value
    )
    bundle["reports"]["semantic_event"]["measurement_status"] = (
        CellStatus.NOT_EVALUATED.value
    )

    report = physical_h3_kms_preflight_report(bundle)

    assert report["artifact_replay_admission"][
        "replay_bound_postrun_stage_epistemics_receipt"
    ] is False
    assert report["stages"]["P1_nested_refinement_and_expectations"][
        "gate_status"
    ] == "PASS"
    assert report["stages"]["P4_native_bw01_bw08"]["gate_status"] == "PASS"
    assert report["stages"][
        "P6_h3_s2_e3_e4_same_holdout_and_curvature_leverage"
    ]["gate_status"] == "PASS"
    assert report["stages"][
        "P7_semantic_event_e1_e4_and_frame_fiber_separation"
    ]["gate_status"] == "PASS"
    assert report["physical_promotion_allowed"] is False
    assert report["retirement_counting_allowed"] is False


def test_conforming_synthetic_bundle_is_diagnostic_only_and_cannot_self_admit() -> None:
    report = physical_h3_kms_preflight_report(_conforming_bundle())

    assert report[PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT] is False
    assert report["schema"] == "oph_physical_h3_kms_preflight_v3"
    assert report["verdict"] == "NO_GO"
    assert report["gate_status"] == "BLOCKED"
    assert report["campaign_status"] == CellStatus.INCOMPLETE.value
    assert report["physical_promotion_allowed"] is False
    assert report["retirement_counting_allowed"] is False
    assert report["diagnostic_contract_passed"] is True
    assert report["artifact_replay_admission"][
        "PHYSICAL_ARTIFACT_REPLAY_ADMISSION_RECEIPT"
    ] is False
    assert report["artifact_replay_admission"]["gate_status"] == "NOT_EVALUATED"
    assert any(
        blocker.startswith("artifact_admission:") for blocker in report["blockers"]
    )
    assert report["hard_blockers"] == []
    assert all(stage["passed"] for stage in report["stages"].values())
    assert all(
        stage["gate_status"] == "PASS" for stage in report["stages"].values()
    )
    assert report["stages"]["P4_native_bw01_bw08"]["evidence"][
        "native_payload_receipt"
    ] is True
    graph = report["dependency_graph"]
    assert graph["nodes"]["SOURCE_PATCH_ARCHITECTURE"]["passed"] is True
    assert graph["nodes"][PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT]["passed"] is False
    assert graph["nodes"]["EVENT_MANIFOLD_3P1D_RECEIPT"]["passed"] is False
    edges = {(row["from"], row["to"]) for row in graph["edges"]}
    assert ("SOURCE_PATCH_ARCHITECTURE", "LOCAL_REPAIR_DYNAMICS") in edges
    assert ("LOCAL_REPAIR_DYNAMICS", "OBSERVER_SELF_READING_RECORD_LOOP") in edges
    assert ("OBSERVER_SELF_READING_RECORD_LOOP", PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT) in edges
    assert "independent_event_position_manifold_promotion_receipt" in report[
        "downstream_blockers"
    ]["EVENT_MANIFOLD_3P1D_RECEIPT"]


def test_caller_authored_replay_booleans_cannot_self_admit() -> None:
    bundle = deepcopy(_conforming_bundle())
    bundle["reports"]["artifact_replay"] = {
        "REPLAY_MANIFEST_VERIFICATION_RECEIPT": True,
        "SOURCE_CAPTURE_REPLAY_RECEIPT": True,
        "PER_CELL_CONTROL_ARTIFACTS_REPLAY_RECEIPT": True,
        "PER_CELL_SCIENTIFIC_PREDICATES_RECOMPUTED_RECEIPT": True,
        "SINGLE_BUNDLE_COMMITMENT_RECEIPT": True,
    }

    report = physical_h3_kms_preflight_report(bundle)
    admission = report["artifact_replay_admission"]
    assert admission["PHYSICAL_ARTIFACT_REPLAY_ADMISSION_RECEIPT"] is False
    assert admission["replay_manifest_path"] is None
    assert admission["registered_source_capture_replay_bound"] is False
    assert "registered_replay_manifest_not_verified" in admission["blockers"]
    assert "stored_replay_report_is_not_exact_fresh_replay" in admission["blockers"]


def test_candidate_specific_response_matrix_fails_closed() -> None:
    bundle = deepcopy(_conforming_bundle())
    candidate = bundle["reports"]["candidate_interventions"]["candidates"]["4pi"]
    candidate["raw_response_hash"] = _hash("different-response-generated-at-4pi")

    report = physical_h3_kms_preflight_report(bundle)

    stage = report["stages"]["P5_frozen_candidate_interventions"]
    assert report[PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT] is False
    assert report["gate_status"] == "INSTRUMENT_INVALID"
    assert report["campaign_status"] == CellStatus.INSTRUMENT_INVALID.value
    assert report["physical_promotion_allowed"] is False
    assert report["retirement_counting_allowed"] is False
    assert stage["gate_status"] == "BLOCKED"
    assert stage["passed"] is False
    assert "candidate_raw_response_hashes_differ" in stage["blockers"]


def test_native_bw_requires_exact_finite_cap_mgns_pair_and_rejects_caller_pass_flag() -> None:
    bundle = deepcopy(_conforming_bundle())
    bundle["reports"]["native_bw"]["clauses"]["C6"]["passed"] = True

    report = physical_h3_kms_preflight_report(bundle)

    stage = report["stages"]["P4_native_bw01_bw08"]
    assert stage["passed"] is False
    assert "bw_native_payload_missing" in report["hard_blockers"]
    assert any("clause_wrapper_key_set_mismatch" in value for value in stage["blockers"])
    assert any("caller_asserted_pass" in value for value in stage["blockers"])


def test_native_bw_clause_deletion_fails_the_aggregate() -> None:
    bundle = deepcopy(_conforming_bundle())
    del bundle["reports"]["native_bw"]["clauses"]["C4"]

    report = physical_h3_kms_preflight_report(bundle)

    stage = report["stages"]["P4_native_bw01_bw08"]
    assert stage["passed"] is False
    assert any("exactly_c1_through_c6" in value for value in stage["blockers"])


def test_native_bw_recomputed_negative_is_valid_fail_not_invalid_instrument() -> None:
    bundle = deepcopy(_conforming_bundle())
    row = bundle["reports"]["native_bw"]["clauses"]["C5"]
    row["primitive_fields"]["kms_residual_beta_2pi"] = 100.0
    row["primitive_artifact_hash"] = canonical_payload_hash(row["primitive_fields"])

    report = physical_h3_kms_preflight_report(bundle)

    stage = report["stages"]["P4_native_bw01_bw08"]
    assert stage["passed"] is True
    assert stage["evidence"]["native_payload_conformance_receipt"] is True
    assert stage["evidence"]["native_payload_receipt"] is False
    assert stage["evidence"]["scientific_outcome"] == CellStatus.VALID_FAIL.value
    assert report[PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT] is False
    assert report["scientific_outcome"] == CellStatus.INCOMPLETE.value
    assert report["physical_promotion_allowed"] is False


def test_native_bw_repeated_rho_diagnostic_cannot_satisfy_mgns1() -> None:
    bundle = deepcopy(_conforming_bundle())
    row = bundle["reports"]["native_bw"]["mgns1"]
    repeated = f"sha256:{'5' * 64}"
    row["primitive_fields"]["state_fingerprints_by_level"] = [repeated, repeated]
    row["primitive_artifact_hash"] = canonical_payload_hash(row["primitive_fields"])

    report = physical_h3_kms_preflight_report(bundle)

    stage = report["stages"]["P4_native_bw01_bw08"]
    assert stage["evidence"]["finite_cap_bw_certificate_receipt"] is True
    assert stage["evidence"]["mgns1_certificate_receipt"] is False
    assert stage["evidence"]["support_visible_bw_theorem_applicable"] is False
    assert stage["evidence"]["scientific_outcome"] == CellStatus.VALID_FAIL.value


def test_native_bw_pair_must_bind_both_inputs_to_one_tower() -> None:
    bundle = deepcopy(_conforming_bundle())
    row = bundle["reports"]["native_bw"]["mgns1"]
    row["primitive_fields"]["tower_id"] = "tower-v2"
    row["primitive_artifact_hash"] = canonical_payload_hash(row["primitive_fields"])

    report = physical_h3_kms_preflight_report(bundle)

    stage = report["stages"]["P4_native_bw01_bw08"]
    assert stage["evidence"]["finite_cap_bw_certificate_receipt"] is True
    assert stage["evidence"]["mgns1_certificate_receipt"] is True
    assert stage["evidence"]["same_tower_inputs_receipt"] is False
    assert stage["evidence"]["support_visible_bw_theorem_applicable"] is False


def test_complete_wrong_clock_result_is_valid_fail_not_invalid_instrument() -> None:
    bundle = deepcopy(_conforming_bundle())
    fit = bundle["reports"]["candidate_interventions"]["continuous_clock_fit"]
    fit["fitted_kappa_interval"] = [12.4, 12.7]
    fit["wrong_normalization_separation_passed"] = False
    discrete = bundle["reports"]["candidate_interventions"][
        "discrete_clock_comparison"
    ]
    discrete["paired_losses"] = {"1x": 0.2, "pi": 0.3, "2pi": 0.8, "4pi": 0.1}

    report = physical_h3_kms_preflight_report(bundle)

    stage = report["stages"]["P5_frozen_candidate_interventions"]
    assert stage["passed"] is True
    assert stage["blockers"] == []
    assert stage["evidence"]["scientific_outcome"] == CellStatus.VALID_FAIL.value
    assert report[PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT] is False
    assert report["scientific_outcome"] == CellStatus.INCOMPLETE.value
    assert report["physical_promotion_allowed"] is False


def test_geometry_controls_require_finite_equal_capacity_rows_and_curvature_leverage() -> None:
    bundle = deepcopy(_conforming_bundle())
    controls = bundle["reports"]["geometry_controls"]
    controls["models"]["E3"]["heldout_score"] = float("nan")
    controls["models"]["E4"]["effective_model_capacity"] = 13
    controls.pop("curvature_leverage")

    report = physical_h3_kms_preflight_report(bundle)

    stage = report["stages"]["P6_h3_s2_e3_e4_same_holdout_and_curvature_leverage"]
    assert stage["passed"] is False
    assert "E3_heldout_score_missing_or_nonfinite" in stage["blockers"]
    assert "geometry_model_effective_capacities_are_not_matched" in stage["blockers"]
    assert "geometry_control_missing_or_nonfinite" in report["hard_blockers"]
    assert "curvature_leverage_missing" in report["hard_blockers"]
    assert report["campaign_status"] == CellStatus.INSTRUMENT_INVALID.value


def test_event_e4_requires_translation_component_not_only_lorentz_frames() -> None:
    bundle = deepcopy(_conforming_bundle())
    e4 = bundle["reports"]["semantic_event"]["event_clauses"][
        "EVENT_E4_POINCARE_COCYCLE"
    ]
    e4["translation_components_present"] = False

    report = physical_h3_kms_preflight_report(bundle)

    stage = report["stages"]["P7_semantic_event_e1_e4_and_frame_fiber_separation"]
    assert stage["passed"] is True
    assert "EVENT_E4_translation_components_missing" in stage["evidence"][
        "scientific_failures"
    ]
    assert stage["evidence"]["scientific_outcome"] == CellStatus.VALID_FAIL.value
    assert "event_manifold_e1_e4_missing" not in report["hard_blockers"]
    assert report["physical_promotion_allowed"] is False
    assert report["retirement_counting_allowed"] is False


def test_declared_valid_pass_cells_cannot_enable_promotion_without_replay() -> None:
    bundle = deepcopy(_conforming_bundle())
    for row in bundle["reports"]["campaign"]["run_matrix"]:
        row["status"] = CellStatus.VALID_PASS.value
        row["required_controls_complete"] = True
        row["source_hashes_complete"] = True
        row["powered_and_complete"] = True

    report = physical_h3_kms_preflight_report(bundle)

    assert report[PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT] is False
    assert report["reported_campaign_cell_aggregate_status"] == CellStatus.VALID_PASS.value
    assert report["diagnostic_campaign_status"] == CellStatus.VALID_PASS.value
    assert report["campaign_status"] == CellStatus.INCOMPLETE.value
    assert report["physical_promotion_allowed"] is False
    assert report["retirement_counting_allowed"] is False


def test_declared_stable_valid_fail_cannot_retire_without_replay() -> None:
    bundle = deepcopy(_conforming_bundle())
    for row in bundle["reports"]["campaign"]["run_matrix"]:
        row["status"] = CellStatus.VALID_FAIL.value
        row["required_controls_complete"] = True
        row["source_hashes_complete"] = True
        row["powered_and_complete"] = True
        row["failure_mode"] = "CLOCK_2PI_MARGIN_NOT_MET"
        row["failure_evidence_hash"] = _hash(
            f"clock-failure-{row['seed']}-{row['rung']}"
        )

    report = physical_h3_kms_preflight_report(bundle)

    assert report[PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT] is False
    assert report["reported_campaign_cell_aggregate_status"] == CellStatus.VALID_FAIL.value
    assert report["diagnostic_campaign_status"] == CellStatus.VALID_FAIL.value
    assert report["campaign_status"] == CellStatus.INCOMPLETE.value
    assert report["physical_promotion_allowed"] is False
    assert report["retirement_counting_allowed"] is False


def test_incomplete_cell_never_becomes_valid_fail_or_retirement() -> None:
    bundle = deepcopy(_conforming_bundle())
    for row in bundle["reports"]["campaign"]["run_matrix"]:
        row["status"] = CellStatus.VALID_FAIL.value
        row["required_controls_complete"] = True
        row["source_hashes_complete"] = True
        row["powered_and_complete"] = True
        row["failure_mode"] = "CLOCK_2PI_MARGIN_NOT_MET"
        row["failure_evidence_hash"] = _hash(
            f"clock-failure-{row['seed']}-{row['rung']}"
        )
    bundle["reports"]["campaign"]["run_matrix"][0]["status"] = CellStatus.INCOMPLETE.value

    report = physical_h3_kms_preflight_report(bundle)

    assert report["campaign_status"] == CellStatus.INCOMPLETE.value
    assert report["physical_promotion_allowed"] is False
    assert report["retirement_counting_allowed"] is False


def test_family_hash_mismatch_invalidates_cells_instead_of_counting_as_failure() -> None:
    bundle = deepcopy(_conforming_bundle())
    bundle["reports"]["campaign"]["campaign_family_hash"] = _hash("wrong-family")

    report = physical_h3_kms_preflight_report(bundle)

    assert report["campaign_status"] == CellStatus.INSTRUMENT_INVALID.value
    assert "frozen_config_family_mismatch" in report["hard_blockers"]
    assert report["physical_promotion_allowed"] is False
    assert report["retirement_counting_allowed"] is False


def test_legacy_pass_flags_cannot_replace_prime_geometric_primitives() -> None:
    bundle = deepcopy(_conforming_bundle())
    bundle["reports"]["prime_geometric_state"] = {
        "ENDOGENOUS_MODULAR_GENERATOR_RECEIPT": True,
        "KMS_GEOMETRIC_CLOCK_FIT_RECEIPT": True,
        "state_mode": "history_koopman_generator_state",
    }

    report = physical_h3_kms_preflight_report(bundle)

    stage = report["stages"]["P2_prime_geometric_cap_state"]
    assert stage["passed"] is False
    assert "configured_state_mode_is_record_history_or_declared_surrogate" in stage["blockers"]
    assert "rho_dimension_missing_or_trivial" in stage["blockers"]
    assert "modular_generator_is_not_negative_log_rho" in stage["blockers"]


def test_broken_readback_feedback_loop_blocks_geometry_gate_and_dependency_graph() -> None:
    bundle = deepcopy(_conforming_bundle())
    observer = bundle["reports"]["source_observer"]["record_observer"]
    observer["feedback_event_count"] = 0
    observer["readback_changes_future_local_actions"] = False

    report = physical_h3_kms_preflight_report(bundle)

    stage = report["stages"]["P0_source_dynamics_repair_record_observer"]
    graph = report["dependency_graph"]
    assert report[PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT] is False
    assert "OBSERVER_SELF_READING_RECORD_LOOP_dependency_not_discharged" in stage["blockers"]
    assert graph["nodes"]["OBSERVER_SELF_READING_RECORD_LOOP"]["passed"] is False
    assert PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT in report["downstream_blockers"][
        "EVENT_MANIFOLD_3P1D_RECEIPT"
    ]


def test_current_16k_config_is_rejected_before_an_important_campaign() -> None:
    repo = Path(__file__).resolve().parents[1]
    config = repo / "configs" / "k1_population_transfer_16k_dense_ladder.yml"

    report = physical_h3_kms_preflight_report(config)

    assert report[PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT] is False
    assert report["gate_status"] == "NOT_EVALUATED"
    assert all(
        stage["gate_status"] == "NOT_EVALUATED"
        for stage in report["stages"].values()
    )
    assert all(
        stage["scientific_status"] == CellStatus.NOT_EVALUATED.value
        for stage in report["stages"].values()
    )
    assert report["scientific_outcome"] == CellStatus.INCOMPLETE.value
    assert report["realized_report_names"] == []
    source = report["stages"]["P0_source_dynamics_repair_record_observer"]
    refinement = report["stages"]["P1_nested_refinement_and_expectations"]
    state = report["stages"]["P2_prime_geometric_cap_state"]
    native_bw = report["stages"]["P4_native_bw01_bw08"]
    interventions = report["stages"]["P5_frozen_candidate_interventions"]
    campaign = report["stages"]["P8_frozen_multiseed_four_rung_campaign"]
    semantic = report["stages"][
        "P7_semantic_event_e1_e4_and_frame_fiber_separation"
    ]
    assert "SOURCE_PATCH_ARCHITECTURE_dependency_not_discharged" in source["blockers"]
    assert "support_regulator_config_is_missing_or_not_icosahedral" in refinement[
        "blockers"
    ]
    assert "configured_state_mode_is_record_history_or_declared_surrogate" in state["blockers"]
    assert native_bw["passed"] is False
    assert "source_intervention_not_proven_target_free" in interventions["blockers"]
    assert campaign["blockers"] == ["frozen_campaign_manifest_missing"]
    assert not any(semantic["evidence"]["event_clause_status"].values())
    assert report["hard_blockers"] == []
    assert report["physical_promotion_allowed"] is False
    assert report["retirement_counting_allowed"] is False


def test_h3_cannot_stand_in_for_the_event_position_base() -> None:
    bundle = deepcopy(_conforming_bundle())
    semantic = bundle["reports"]["semantic_event"]
    semantic["event_base_role"] = "h3_spatial_chart"
    semantic["h3_and_event_base_separate"] = False

    report = physical_h3_kms_preflight_report(bundle)

    stage = report["stages"]["P7_semantic_event_e1_e4_and_frame_fiber_separation"]
    assert stage["passed"] is False
    assert "event_base_not_typed_as_event_position_manifold" in stage["blockers"]
    assert "h3_frame_fiber_and_event_base_not_separated" in stage["blockers"]


def test_carrier_rung_cannot_be_silently_replaced_by_s2_mesh_cell_count() -> None:
    bundle = deepcopy(_conforming_bundle())
    campaign = bundle["reports"]["campaign"]
    campaign["carrier_counts"][0] = SUPPORT_REGULATOR_CELL_COUNTS[0]
    campaign["run_matrix"][0]["carrier_count"] = SUPPORT_REGULATOR_CELL_COUNTS[0]

    report = physical_h3_kms_preflight_report(bundle)

    stage = report["stages"]["P8_frozen_multiseed_four_rung_campaign"]
    assert stage["passed"] is False
    assert "exact_carrier_counts_mismatch" in stage["blockers"]
    assert "run_matrix_0_carrier_count_mismatch" in stage["blockers"]
    assert "frozen_config_family_mismatch" in report["hard_blockers"]


def test_source_federation_and_support_regulator_must_be_separate_configs() -> None:
    bundle = deepcopy(_conforming_bundle())
    bundle["config"]["graph"] = bundle["config"].pop("support_regulator")
    bundle["config"].pop("source_federation")

    report = physical_h3_kms_preflight_report(bundle)

    source = report["stages"]["P0_source_dynamics_repair_record_observer"]
    refinement = report["stages"]["P1_nested_refinement_and_expectations"]
    assert source["passed"] is False
    assert "source_federation_family_missing_or_mismatched" in source["blockers"]
    assert "source_federation_carrier_count_missing" in source["blockers"]
    assert refinement["passed"] is False
    assert "support_regulator_config_is_missing_or_not_icosahedral" in refinement[
        "blockers"
    ]


def test_preflight_admits_only_primitive_replayed_repair_artifacts() -> None:
    envelope_bundle = deepcopy(_conforming_bundle())
    envelope = _repair_replay_envelope()
    envelope_bundle["reports"]["repair_receipt"] = envelope

    replayed = physical_h3_kms_preflight_report(envelope_bundle)
    assert replayed["artifact_replay_admission"][
        "registered_transaction_replay_bound"
    ] is True
    assert replayed[PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT] is False

    raw_receipt_bundle = deepcopy(_conforming_bundle())
    raw_receipt_bundle["reports"]["repair_receipt"] = envelope[
        "expected_receipt"
    ]
    rejected = physical_h3_kms_preflight_report(raw_receipt_bundle)
    assert rejected["artifact_replay_admission"][
        "registered_transaction_replay_bound"
    ] is False
    assert rejected["artifact_replay_admission"]["registered_replay_summaries"][
        "transaction"
    ]["REPAIR_REPLAY_ENVELOPE_INTEGRITY_RECEIPT"] is False
