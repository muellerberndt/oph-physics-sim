from __future__ import annotations

from copy import deepcopy

import pytest

import oph_fpe.bulk.physical_h3_kms_campaign as campaign_module
import oph_fpe.bulk.physical_h3_kms_postrun as postrun_module
from oph_fpe.bulk.physical_h3_kms_postrun import (
    PREREGISTRATION_ENVELOPE_SCHEMA,
    PostrunCaptureError,
    compute_postrun_reports,
    curvature_calibration_commitment,
)
from oph_fpe.bulk.physical_h3_kms_preflight import physical_h3_kms_preflight_report
from oph_fpe.bulk.physical_h3_kms_prerun import (
    ALLOWED_CHECKER_IDS,
    ALLOWED_PRODUCER_IDS,
    CELL_CONFIG_SCHEMA,
    PLAN_SCHEMA,
    REGISTERED_HISTORICAL_16K_SOURCE_SEED,
    REGISTERED_HISTORICAL_CAMPAIGN_SHA256,
    REGISTERED_HISTORICAL_RECEIPT_BYTE_SHA256,
    SCHEMA_VERSION,
    canonical_sha256,
    physical_h3_kms_prerun_report,
    physical_h3_kms_source_inputs,
)
from oph_fpe.bulk.physical_h3_kms_source_capture import (
    TYPED_CLOCK_PAIR_CONTRACT_SCHEMA,
    TYPED_CLOCK_PAIR_INPUT_SCHEMA,
    capture_physical_source,
)


def _hash(label: str) -> str:
    return canonical_sha256({"label": label})


def test_not_evaluated_assessment_is_deterministic_and_non_promotional() -> None:
    kwargs = {
        "reason": "missing_independent_test_producer",
        "structural_complete": True,
        "structural_artifact_hash": _hash("structure"),
        "instrument_producer_available": False,
        "sensitivity_complete": True,
        "sensitivity_artifact_hash": _hash("sensitivity"),
        "diagnostic_findings": ["diagnostic_predicate_false"],
    }
    first = postrun_module._not_evaluated_stage_assessment(**kwargs)
    second = postrun_module._not_evaluated_stage_assessment(**kwargs)
    assert first == second
    assert first["measurement_status"] == "NOT_EVALUATED"
    assert first["physical_gate_eligible"] is False
    assert first["scientific_failures"] == []
    assert first["sensitivity_receipt"]["diagnostic_findings"] == [
        "diagnostic_predicate_false"
    ]
    assert first["sensitivity_receipt"]["physical_gate_eligible"] is False


def test_target_encoded_curvature_power_producer_is_not_executable() -> None:
    assert not hasattr(postrun_module, "_independent_curvature_power")


def test_target_firewall_iterative_scan_preserves_key_and_value_semantics() -> None:
    value = {
        "candidate.scale": "KMS",
        "foo_receipt": False,
        "nested": [
            {"selected-model": "2pi"},
            {"preferred-geometry": "innocent"},
        ],
        "safe": "not-a-target-token",
    }

    assert postrun_module._target_leak_hits(value) == sorted(
        {
            "$.candidate.scale:target_token",
            "$.candidate.scale:target_value",
            "$.foo_receipt:caller_assertion",
            "$.nested[0].selected-model:caller_assertion",
            "$.nested[0].selected-model:target_value",
            "$.nested[1].preferred-geometry:target_token",
        }
    )


def test_exact_small_p6_p7_diagnostics_do_not_manufacture_science_failures() -> None:
    source = capture_physical_source(
        {
            "carrier_count": 8,
            "rung": 8,
            "seed": 1729,
            "cycles": 4,
            "record_commit_cycles": 2,
            "propagation_steps": 2,
            "observer_count": 2,
            "observer_support_size": 2,
            "observer_samples": 4,
            "checkpoint_interval": 2,
            "support_refinement_level": 1,
            "geometry_sample_count": 8,
        }
    )
    raw = source["postrun_capture"]
    plan = {
        "thresholds": {
            "geometry_win_margin_min": 0.05,
            "curvature_minimum_power": 0.9,
        },
        "calibrations": {
            "geometry_calibration_sha256": _hash("small-geometry"),
            "curvature_calibration_sha256": _hash("small-curvature"),
            "independent_of_campaign_source_seeds": True,
            "frozen_before_source_capture": True,
            "physical_threshold_calibration_receipt": False,
        },
        "archive_boundary": {
            "frozen_before_source_capture": True,
            "retune_after_freeze": False,
        },
        "split_contract": {
            "algorithm_id": "semantic_hash_split_v1",
            "assignment_salt_sha256": _hash("small-split"),
            "holdout_fraction": 0.25,
            "derivation": "semantic_event_id_hash_threshold_v1",
            "heldout_ids_materialized_before_capture": False,
        },
        "current_cell": {"seed": 1729, "rung": 8, "replicate_id": "primary"},
    }
    geometry, geometry_stats = postrun_module._geometry_control_report(
        raw["geometry_control_rows"],
        plan,
        {"geometry_control_rows": canonical_sha256(raw["geometry_control_rows"])},
    )
    event, event_stats = postrun_module._semantic_event_report(
        raw["semantic_events"],
        raw["raw_overlap_relations"],
        raw["raw_ancestry_relations"],
        plan,
    )

    assert geometry["measurement_status"] == "NOT_EVALUATED"
    assert geometry["scientific_failures"] == []
    assert geometry_stats["scientific_failures"] == []
    assert geometry["diagnostic_findings"]
    assert event["measurement_status"] == "NOT_EVALUATED"
    assert event["scientific_failures"] == []
    assert event_stats["scientific_failures"] == []
    assert event["diagnostic_findings"]
    pair_scope = event["pairwise_diagnostic_scope"]
    event_count = len(raw["semantic_events"])
    assert pair_scope["full_pair_census"] is True
    assert pair_scope["population_pair_count"] == event_count * (event_count - 1) // 2
    assert pair_scope["evaluated_pair_count"] == pair_scope["population_pair_count"]
    assert event["quadratic_event_cone"]["relation_scope"] == (
        "exact_postcapture_diagnostic_census"
    )


def test_event_pair_rank_inversion_matches_exact_lexicographic_pairs() -> None:
    for event_count in range(2, 24):
        expected = [
            (left, right)
            for left in range(event_count)
            for right in range(left + 1, event_count)
        ]
        observed = [
            postrun_module._pair_rank_to_indices(rank, event_count)
            for rank in range(len(expected))
        ]
        assert observed == expected


def test_bounded_event_pair_sample_is_deterministic_unique_and_non_promotional() -> None:
    event_count = 300
    keys = [f"event-{index:04d}" for index in range(event_count)]
    computed_ids = {key: _hash(key) for key in keys}
    plan = {
        "split_contract": {
            "assignment_salt_sha256": _hash("bounded-pair-split"),
        }
    }

    first_pairs, first_scope = postrun_module._diagnostic_event_pair_sample(
        keys,
        computed_ids,
        plan,
    )
    second_pairs, second_scope = postrun_module._diagnostic_event_pair_sample(
        keys,
        computed_ids,
        plan,
    )

    assert first_pairs == second_pairs
    assert first_scope == second_scope
    assert len(first_pairs) == postrun_module._P7_MAX_DIAGNOSTIC_PAIR_COUNT
    assert len({rank for rank, _, _ in first_pairs}) == len(first_pairs)
    assert all(0 <= left < right < event_count for _, left, right in first_pairs)
    assert first_scope["full_pair_census"] is False
    assert first_scope["scope"] == "postcapture_p7_diagnostic_only"
    assert first_scope["physical_gate_eligible"] is False
    assert first_scope["diagnostic_scope_preregistered_as_physical_instrument"] is False


def _preregistration() -> dict:
    rungs = [4_096, 16_384, 65_536, 262_144]
    seeds = [101, 202, 303]
    support_counts = {str(rung): int(1.25 * rung) for rung in rungs}
    observer_counts = dict(zip(map(str, rungs), [32, 64, 128, 256], strict=True))
    producers = {
        role: {
            "producer_id": next(iter(values)),
            "source_code_sha256": _hash(f"producer:{role}"),
        }
        for role, values in ALLOWED_PRODUCER_IDS.items()
    }
    checkers = {
        role: {
            "checker_id": next(iter(values)),
            "checker_code_sha256": _hash(f"checker:{role}"),
        }
        for role, values in ALLOWED_CHECKER_IDS.items()
    }
    scaling = {
        "carrier_count_law": "exact_rung_cardinality_v1",
        "support_regulator_law": "first_icosahedral_cell_count_at_or_above_rung_v1",
        "support_counts_by_rung": support_counts,
        "observer_scaling": {
            "law_id": "power_law_ceil_v1",
            "coefficient": 0.5,
            "exponent": 0.5,
            "minimum": 32,
            "maximum": 4096,
            "counts_by_rung": observer_counts,
        },
        "cycles": 160,
        "repair_fraction_per_cycle": 0.0625,
        "record_commit_cycles": 12,
    }

    def cell_config(seed: int, rung: int) -> dict:
        level = {4_096: 4, 16_384: 5, 65_536: 6, 262_144: 7}[rung]
        return {
            "schema": CELL_CONFIG_SCHEMA,
            "cell": {"seed": seed, "rung": rung, "replicate_id": "primary"},
            "source_federation": {
                "family": "federated_echosahedral_carriers",
                "carrier_count": rung,
                "ports_per_carrier": 12,
                "local_template": "regular_icosahedron_12_30_20_antipode_a5_v1",
                **producers["source_federation"],
            },
            "support_regulator": {
                "family": "nested_geodesic_icosahedral",
                "patch_basis": "cells",
                "refinement_level": level,
                "patch_count": support_counts[str(rung)],
                "drives_source_seams": False,
                "drives_source_repairs": False,
                **producers["support_regulator"],
            },
            "source_generator": {
                **producers["source_dynamics"],
                "state_space": "normalized_complex_amplitude_in_C12",
                "rng_family": "numpy_generator_pcg64_v1",
                "initialization_distribution": "normalized_complex_gaussian_v1",
                "intrinsic_phase_distribution": "uniform_unit_interval_v1",
                "propagation_steps": 4,
                "intrinsic_step": 0.137,
                "coupling_strength": 1.0,
                "geometry_sample_count": 32,
            },
            "repair_dynamics": {
                **producers["source_dynamics"],
                "cycles": scaling["cycles"],
                "repair_fraction_per_cycle": scaling["repair_fraction_per_cycle"],
                "record_commit_cycles": scaling["record_commit_cycles"],
                "seam_update_rule": "disjoint_single_port_endpoint_arithmetic_mean_v1",
            },
            "observer_capture": {
                **producers["observer_capture"],
                "observer_count": observer_counts[str(rung)],
                "support_size": 2,
                "samples_per_observer": 6,
                "prediction_control": "semantic_hash_shuffle_v1",
                "feedback_enabled": True,
                "checkpoint_interval": 8,
            },
        }

    current = {"seed": 101, "rung": 4_096, "replicate_id": "primary"}
    run_matrix = []
    requested = None
    for seed in seeds:
        for rung in rungs:
            config = cell_config(seed, rung)
            run_matrix.append(
                {
                    "cell": config["cell"],
                    "cell_config": config,
                    "config_sha256": canonical_sha256(config),
                    "status": "NOT_EVALUATED",
                }
            )
            if config["cell"] == current:
                requested = deepcopy(config)
    plan = {
        "schema": PLAN_SCHEMA,
        "campaign_id": "physical-h3-kms-postrun-test-001",
        "instrument_version": "physical-h3-kms-v2",
        "instrument_commit_sha256": _hash("instrument"),
        "seeds": seeds,
        "rungs": rungs,
        "replicate_ids": ["primary"],
        "clock_candidates": ["1x", "pi", "2pi", "4pi"],
        "geometry_models": ["H3", "S2", "E3", "E4"],
        "thresholds": {
            "clock_absolute_residual_max": 0.2,
            "clock_win_margin_min": 0.1,
            "geometry_win_margin_min": 0.05,
            "curvature_minimum_power": 0.9,
        },
        "calibrations": {
            "clock_calibration_sha256": _hash("clock-calibration"),
            "geometry_calibration_sha256": _hash("geometry-calibration"),
            "curvature_calibration_sha256": curvature_calibration_commitment(),
            "independent_of_campaign_source_seeds": True,
            "frozen_before_source_capture": True,
            "physical_threshold_calibration_receipt": False,
        },
        "split_contract": {
            "algorithm_id": "semantic_hash_split_v1",
            "assignment_salt_sha256": _hash("semantic-split-salt"),
            "holdout_fraction": 0.25,
            "derivation": "semantic_event_id_hash_threshold_v1",
            "heldout_ids_materialized_before_capture": False,
        },
        "scaling_contract": scaling,
        "archive_boundary": {
            "frozen_before_source_capture": True,
            "retune_after_freeze": False,
            "archived_16k_failure_preserved": True,
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
        "producer_registry": producers,
        "checker_registry": checkers,
        "run_matrix": run_matrix,
        "current_cell": current,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    assert requested is not None
    return {"schema": SCHEMA_VERSION, "config": requested, "plan": plan}


def _envelope(preregistration: dict) -> dict:
    report = physical_h3_kms_prerun_report(preregistration)
    assert report["admission_status"] == "VALID_PASS"
    return {
        "schema": PREREGISTRATION_ENVELOPE_SCHEMA,
        "preregistration": preregistration,
        "preregistration_report": report,
        "preregistration_sha256": canonical_sha256(preregistration),
    }


@pytest.fixture(scope="module")
def artifacts() -> dict:
    preregistration = _preregistration()
    envelope = _envelope(preregistration)
    source = capture_physical_source(physical_h3_kms_source_inputs(preregistration))
    reports = compute_postrun_reports(source, envelope)
    return {
        "preregistration": preregistration,
        "envelope": envelope,
        "source": source,
        "reports": reports,
    }


def _available_clock_pair_input(
    artifacts: dict,
    *,
    group_count: int = 64,
    refinement_levels: tuple[int, ...] = (3, 4),
) -> dict:
    raw = artifacts["source"]["postrun_capture"]
    interventions = raw["intervention_rows"]
    responses = raw["response_rows"]
    modular_source_hash = _hash("independent-modular-source-field")
    geometric_source_hash = _hash("independent-geometric-source-field")
    contract = {
        "schema": TYPED_CLOCK_PAIR_CONTRACT_SCHEMA,
        "status": "AVAILABLE",
        "join_key_fields": [
            "intervention_id",
            "event_id",
            "observer_or_cap_id",
            "refinement_level",
            "trajectory_group_id",
        ],
        "group_key_fields": [
            "source_seed",
            "observer_or_cap_id",
            "trajectory_group_id",
        ],
        "minimum_refinement_level_count": 2,
        "modular_transport_producer_id": "independent-modular-transport-v1",
        "modular_transport_producer_code_sha256": _hash("modular-code"),
        "modular_transport_source_field_sha256": modular_source_hash,
        "geometric_flow_producer_id": "independent-geometric-flow-v1",
        "geometric_flow_producer_code_sha256": _hash("geometric-code"),
        "geometric_flow_source_field_sha256": geometric_source_hash,
        "source_fixed_oriented_frame_incidence_required": True,
        "scoring_constants_absent": True,
        "unavailable_reason": None,
    }
    modular_rows = []
    geometric_rows = []
    source_seed = artifacts["preregistration"]["plan"]["current_cell"]["seed"]
    for group_index in range(group_count):
        response = responses[group_index % len(responses)]
        intervention = interventions[group_index % len(interventions)]
        trajectory_group_id = f"trajectory-{group_index:04d}"
        for level in refinement_levels:
            common = {
                "intervention_id": intervention["row_id"],
                "event_id": response["record_event_id"],
                "observer_or_cap_id": response["observer_token"],
                "refinement_level": level,
                "trajectory_group_id": trajectory_group_id,
                "source_seed": source_seed,
            }
            transport_time = 0.2 + 0.013 * group_index + 0.007 * level
            modular_material = {
                "row_id": f"modular-{group_index:04d}-{level}",
                **common,
                "modular_transport_time": transport_time,
                "producer_source_field_sha256": modular_source_hash,
            }
            modular_rows.append(
                {**modular_material, "row_sha256": canonical_sha256(modular_material)}
            )
            # This arbitrary slope is deliberately not one of the frozen
            # evaluator constants; the fixture exercises plumbing, not a pass.
            geometric_parameter = (
                1.75 * transport_time
                + 0.0005 * ((group_index % 5) - 2)
                + 0.0001 * level
            )
            geometric_material = {
                "row_id": f"geometric-{group_index:04d}-{level}",
                **common,
                "geometric_flow_parameter": geometric_parameter,
                "producer_source_field_sha256": geometric_source_hash,
                "oriented_frame_incidence_sha256": _hash(
                    f"oriented-frame:{group_index}:{level}"
                ),
            }
            geometric_rows.append(
                {
                    **geometric_material,
                    "row_sha256": canonical_sha256(geometric_material),
                }
            )
    return {
        "schema": TYPED_CLOCK_PAIR_INPUT_SCHEMA,
        "contract": contract,
        "modular_transport_rows": modular_rows,
        "geometric_flow_rows": geometric_rows,
    }


def test_registered_postrun_is_target_blind_and_candidate_paired(artifacts: dict) -> None:
    reports = artifacts["reports"]
    candidates = reports["candidate_interventions"]["candidates"]
    assert set(candidates) == {"1x", "pi", "2pi", "4pi"}
    assert len({row["raw_response_hash"] for row in candidates.values()}) == 1
    assert len({row["source_trajectory_hash"] for row in candidates.values()}) == 1
    assert all(row["candidate_scale_enters_intervention"] is False for row in candidates.values())
    assert reports["candidate_interventions"]["measurement_status"] == "NOT_EVALUATED"
    pair_contract = reports["candidate_interventions"]["typed_clock_pair_contract"]
    assert pair_contract["required_collections"] == [
        "modular_transport_rows",
        "geometric_flow_rows",
    ]
    assert pair_contract["required_fields"] == [
        "modular_transport_time",
        "geometric_flow_parameter",
    ]
    assert pair_contract["available"] is False
    assert pair_contract["contract_status"] == "UNAVAILABLE"
    assert pair_contract["minimum_refinement_level_count"] == 2
    assert pair_contract["intensity_delta_used_as_modular_time"] is False
    assert pair_contract["one_field_synthesized_from_the_other"] is False
    discrete = reports["candidate_interventions"]["discrete_clock_comparison"]
    assert discrete["threshold_fixture_commitment_valid"] is True
    assert discrete["thresholds_from_independent_synthetic_calibration"] is False
    assert discrete["physical_calibration_replay_bound"] is False
    source_text = repr(artifacts["source"]["postrun_capture"]).lower()
    assert "event_position" not in source_text
    assert "h3_frame" not in source_text
    assert "selected_model" not in source_text


def test_typed_clock_pairs_use_separate_producers_and_grouped_refinement(
    artifacts: dict,
) -> None:
    raw = artifacts["source"]["postrun_capture"]
    pair_input = _available_clock_pair_input(artifacts)
    report, stats = postrun_module._candidate_report(
        raw["carrier_port_trajectories"],
        raw["intervention_rows"],
        raw["response_rows"],
        pair_input,
        artifacts["preregistration"]["plan"],
    )

    assert report["measurement_status"] == "NOT_EVALUATED"
    assert report["physical_gate_eligible"] is False
    assert report["not_evaluated_reasons"] == [
        "replay_bound_independent_physical_clock_calibration_missing"
    ]
    assert report["scientific_failures"] == []
    assert report["typed_clock_pair_contract"]["available"] is True
    grouped = report["grouped_inference"]
    assert grouped["train_group_count"] >= 2
    assert grouped["holdout_group_count"] >= 2
    assert grouped["row_split_within_group"] is False
    assert grouped["equal_weight_per_holdout_group"] is True
    assert report["continuous_clock_fit"]["actual_refinement_levels"] == [3, 4]
    assert [
        row["refinement_level"]
        for row in report["continuous_clock_fit"]["tail_level_fits"]
    ] == [3, 4]
    assert report["discrete_clock_comparison"][
        "thresholds_from_independent_synthetic_calibration"
    ] is False
    assert report["discrete_clock_comparison"][
        "physical_calibration_replay_bound"
    ] is False
    assert stats["evaluated"] is False
    assert stats["typed_producers_available"] is True
    assert stats["sensitivity_complete"] is True
    assert stats["diagnostic_findings"]

    heldout_pair_ids = set(
        report["candidates"]["1x"]["heldout_event_row_ids"]
    )
    group_members: dict[str, set[str]] = {}
    for row in report["primitive_typed_clock_pair_rows"]:
        group_members.setdefault(row["trajectory_group_sha256"], set()).add(
            row["pair_row_id"]
        )
    assert all(
        members <= heldout_pair_ids or members.isdisjoint(heldout_pair_ids)
        for members in group_members.values()
    )


def test_typed_clock_contract_rejects_shared_producer_and_single_level(
    artifacts: dict,
) -> None:
    raw = artifacts["source"]["postrun_capture"]
    plan = artifacts["preregistration"]["plan"]
    shared_producer = _available_clock_pair_input(artifacts)
    shared_producer["contract"]["geometric_flow_producer_id"] = shared_producer[
        "contract"
    ]["modular_transport_producer_id"]
    with pytest.raises(PostrunCaptureError, match="producer_ids_not_disjoint"):
        postrun_module._candidate_report(
            raw["carrier_port_trajectories"],
            raw["intervention_rows"],
            raw["response_rows"],
            shared_producer,
            plan,
        )

    one_level = _available_clock_pair_input(
        artifacts, refinement_levels=(3,)
    )
    with pytest.raises(PostrunCaptureError, match="refinement_levels_insufficient"):
        postrun_module._candidate_report(
            raw["carrier_port_trajectories"],
            raw["intervention_rows"],
            raw["response_rows"],
            one_level,
            plan,
        )


def test_already_verified_source_path_is_exactly_equivalent(artifacts: dict) -> None:
    trusted = postrun_module._compute_postrun_reports_from_verified_source(
        artifacts["source"],
        artifacts["envelope"],
    )

    assert trusted == artifacts["reports"]
    assert canonical_sha256(trusted) == canonical_sha256(artifacts["reports"])


def test_postrun_separates_diagnostics_from_physical_measurements(
    artifacts: dict,
) -> None:
    source = artifacts["source"]
    reports = artifacts["reports"]
    epistemics = reports["stage_epistemics"]
    assert set(epistemics) == {
        "P1_nested_refinement_and_expectations",
        "P2_prime_geometric_cap_state",
        "P3_independent_geometric_parameter",
        "P4_native_bw01_bw08",
        "P5_frozen_candidate_interventions",
        "P6_h3_s2_e3_e4_same_holdout_and_curvature_leverage",
        "P7_semantic_event_e1_e4_and_frame_fiber_separation",
        "P8_frozen_multiseed_four_rung_campaign",
    }
    for stage in epistemics.values():
        assert stage["measurement_status"] == "NOT_EVALUATED"
        assert stage["physical_gate_eligible"] is False
        assert stage["not_evaluated_reasons"]
        assert stage["scientific_failures"] == []
        assert stage["structural_receipt"]["scope"] == "artifact_integrity_only"
        assert stage["instrument_receipt"]["status"] == (
            "MISSING_REQUIRED_PRODUCER"
        )
        assert stage["sensitivity_receipt"]["physical_gate_eligible"] is False

    p4 = reports["native_bw_diagnostic_verification"]
    assert p4["measurement_status"] == "NOT_EVALUATED"
    assert p4["physical_gate_eligible"] is False
    assert p4["scientific_failures"] == []
    assert p4["structural_conformance_complete"] is False
    assert p4["unavailable_input_contract_complete"] is True
    assert reports["native_bw"]["status"] == "UNAVAILABLE"
    assert reports["native_bw"]["clauses"] == {}
    assert "analytic_strip_kms_residual" in reports["native_bw"][
        "missing_producers"
    ]
    assert "central_minimized_weighted_generator_distance" in reports[
        "native_bw"
    ]["missing_producers"]

    p6 = reports["geometry_controls"]
    assert p6["measurement_status"] == "NOT_EVALUATED"
    assert p6["physical_gate_eligible"] is False
    comparison = p6["paired_geometry_comparison"]
    assert comparison["equal_footing_candidate_inputs"] is True
    assert comparison["feature_target_disjoint_receipt"] is True
    assert comparison["predictor_response_field_intersection"] == []
    assert comparison["row_split_within_group"] is False
    assert comparison["physical_threshold_calibration_receipt"] is False
    assert len({row["preprocessing_hash"] for row in p6["models"].values()}) == 1
    assert len({row["fit_protocol_hash"] for row in p6["models"].values()}) == 1
    assert len(
        {row["candidate_design_matrix_hash"] for row in p6["models"].values()}
    ) >= 2
    assert p6["curvature_leverage"]["calibration_source"] == (
        "legacy_synthetic_power_suite_quarantined"
    )
    assert p6["curvature_leverage"]["calibrated_power"] is None
    assert p6["scientific_failures"] == []

    p7 = reports["semantic_event"]
    assert p7["measurement_status"] == "NOT_EVALUATED"
    assert p7["physical_gate_eligible"] is False
    assert p7["EVENT_MANIFOLD_3P1D_RECEIPT"] is False
    assert p7["event_clauses"]["EVENT_E3_RANK_FOUR"][
        "independent_clock_receipt"
    ] is False
    assert p7["event_clauses"]["EVENT_E3_RANK_FOUR"][
        "physical_independent_clock_receipt"
    ] is False
    assert p7["quadratic_event_cone"]["coordinate_source"] == (
        "postcapture_diagnostic_construction"
    )
    assert p7["quadratic_event_cone"]["inference_source"] == (
        "UNAVAILABLE_NO_INDEPENDENT_EVENT_CHART"
    )
    assert p7["quadratic_event_cone"]["cone_inference_artifact_hash"] is None
    assert p7["physical_event_chart_input"]["status"] == "UNAVAILABLE"
    assert p7["semantic_event_dag"]["quotient_canonical_identity_receipt"] is False
    assert p7["causal_ancestry"]["generic_event_token_used_as_resource_witness"] is False
    assert p7["event_base_role"] == "presentation_layout_only_not_event_manifold"
    assert p7["scientific_failures"] == []

    assert reports["postrun_scientific_failures"] == []
    assert reports["postrun_not_evaluated_reasons"]
    assert {
        "physical_echosahedral_federation_realization_not_established",
        "carrier_to_support_chart_realization_not_established",
        "carrier_refinement_naturality_not_established",
    }.issubset(reports["postrun_not_evaluated_reasons"])
    audit = physical_h3_kms_preflight_report(
        {"config": source["config"], "reports": {**source["reports"], **reports}}
    )
    p5 = audit["stages"]["P5_frozen_candidate_interventions"]
    assert p5["passed"] is False
    assert p5["gate_status"] == "NOT_EVALUATED"
    assert p5["evidence"]["scientific_outcome"] == "NOT_EVALUATED"
    assert p5["scientific_failure_count"] == 0
    assert p5["evidence"]["scientific_failures"] == []
    assert audit["physical_promotion_allowed"] is False


def test_p0_physical_source_gaps_are_part_of_cell_completion(
    artifacts: dict,
) -> None:
    source = deepcopy(artifacts["source"])
    expected = {
        "physical_echosahedral_federation_realization_not_established",
        "carrier_to_support_chart_realization_not_established",
        "carrier_refinement_naturality_not_established",
    }
    assert set(postrun_module._p0_not_evaluated_reasons(source)) == expected

    report = source["reports"]["source_observer"]
    for receipt, _reason in postrun_module._P0_PHYSICAL_SOURCE_REQUIREMENTS:
        report[receipt] = True
    assert postrun_module._p0_not_evaluated_reasons(source) == []


def test_old_mapping_only_synthetic_capture_is_rejected() -> None:
    with pytest.raises(PostrunCaptureError, match="only registered"):
        compute_postrun_reports(
            {"schema": "oph.physical_h3_kms.target_blind_capture.v1"},
            _envelope(_preregistration()),
        )


def test_preregistration_report_digest_and_cell_binding_are_enforced(
    artifacts: dict,
) -> None:
    bad_report = deepcopy(artifacts["envelope"])
    bad_report["preregistration_report"]["SOURCE_CAPTURE_ALLOWED"] = False
    with pytest.raises(PostrunCaptureError, match="report is not an exact replay"):
        compute_postrun_reports(artifacts["source"], bad_report)

    bad_digest = deepcopy(artifacts["envelope"])
    bad_digest["preregistration_sha256"] = _hash("wrong-preregistration")
    with pytest.raises(PostrunCaptureError, match="digest mismatch"):
        compute_postrun_reports(artifacts["source"], bad_digest)


def test_source_tamper_fails_replay_and_campaign_updates_only_current_cell(
    artifacts: dict,
) -> None:
    tampered = deepcopy(artifacts["source"])
    tampered["postrun_capture"]["response_rows"][0]["raw_response"] += 1.0
    with pytest.raises(PostrunCaptureError, match="source capture replay failed"):
        compute_postrun_reports(tampered, artifacts["envelope"])

    rows = artifacts["reports"]["campaign"]["run_matrix"]
    assert len(rows) == 12
    current = [row for row in rows if row["status"] != "NOT_EVALUATED"]
    assert len(current) == 1
    assert current[0]["status"] == "INCOMPLETE"
    assert current[0]["required_controls_complete"] is False
    assert current[0]["powered_and_complete"] is False
    assert all(row["carrier_count"] == row["rung"] for row in rows)
    campaign = artifacts["reports"]["campaign"]
    assert campaign["campaign_status"] == "INCOMPLETE"
    assert campaign["physical_promotion_allowed"] is False
    assert campaign["branch_retirement_authorized"] is False
    assert campaign["stable_failure_rule_satisfied"] is False
    assert campaign["scientific_failures"] == []


def test_campaign_family_and_protocol_hashes_ignore_only_cell_selector() -> None:
    four_k = campaign_module.build_frozen_campaign(current_rung=4_096)[
        "preregistration"
    ]
    sixty_four_k = campaign_module.build_frozen_campaign(current_rung=65_536)[
        "preregistration"
    ]

    reports = [
        postrun_module._campaign_manifest(
            preregistration["plan"],
            preregistration,
            {"source": _hash("same-source-component-contract")},
            scientific_failures=[],
            evaluation_complete=False,
        )
        for preregistration in (four_k, sixty_four_k)
    ]

    assert reports[0]["execution_plan_sha256"] != reports[1][
        "execution_plan_sha256"
    ]
    assert reports[0]["frozen_campaign_family_sha256"] == reports[1][
        "frozen_campaign_family_sha256"
    ]
    assert reports[0]["campaign_family_hash"] == reports[1][
        "campaign_family_hash"
    ]
    assert {
        row["protocol_hash"]
        for report in reports
        for row in report["run_matrix"]
    } == {reports[0]["run_matrix"][0]["protocol_hash"]}
