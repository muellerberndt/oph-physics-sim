from __future__ import annotations

import copy
import hashlib
import json
import re

import numpy as np
import pytest

from oph_fpe.bulk.physical_h3_kms_preflight import (
    physical_h3_kms_preflight_report,
)
from oph_fpe.bulk.physical_h3_kms_source_capture import (
    POSTRUN_CAPTURE_SCHEMA,
    _normalize_config,
    capture_physical_source,
    verify_physical_source_capture,
    write_physical_source_capture,
)


SMALL_CONFIG = {
    "carrier_count": 4,
    "seed": 1729,
    "propagation_steps": 2,
    "cycles": 16,
    "repair_fraction_per_cycle": 0.0625,
    "record_commit_cycles": 4,
    "observer_count": 2,
    "observer_support_size": 2,
    "observer_samples": 4,
    "prediction_control": "semantic_hash_shuffle_v1",
    "feedback_enabled": True,
    "checkpoint_interval": 3,
    "support_refinement_level": 1,
    "geometry_sample_count": 4,
    "rung": 4,
    "replicate_id": "primary",
    "preregistered_plan_sha256": "sha256:" + "a" * 64,
    "intrinsic_step": 0.137,
    "coupling_strength": 1.0,
    "state_space": "normalized_complex_amplitude_in_C12",
    "rng_family": "numpy_generator_pcg64_v1",
    "initialization_distribution": "normalized_complex_gaussian_v1",
    "intrinsic_phase_distribution": "uniform_unit_interval_v1",
    "seam_update_rule": "disjoint_single_port_endpoint_arithmetic_mean_v1",
}


def _sha(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def test_capture_is_deterministic_replayable_and_has_real_local_activity() -> None:
    first = capture_physical_source(SMALL_CONFIG)
    second = capture_physical_source(SMALL_CONFIG)

    assert first == second
    assert verify_physical_source_capture(first)["SOURCE_CAPTURE_REPLAY_RECEIPT"]
    source = first["reports"]["source_observer"]
    dynamics = first["source_artifacts"]["dynamics"]
    assert source["SOURCE_PATCH_ARCHITECTURE_RECEIPT"]
    assert source["source_architecture"]["carrier_count"] == 4
    assert source["source_architecture"]["boundary_port_count"] == 12
    assert source["source_architecture"]["local_state_space"] == (
        "normalized_complex_amplitude_in_C12"
    )
    assert source["source_architecture"]["local_state_dimension"] == 12
    assert source["source_architecture"]["local_state_real_coordinate_count"] == 24
    assert source["record_observer"]["observer_count"] == 2
    assert dynamics["initial_maximum_unit_norm_residual"] <= 1.0e-12
    assert dynamics["repair_mismatch_before"] > 0.0
    assert dynamics["repair_mismatch_after"] <= 1.0e-12
    assert dynamics["repair_event_count"] == 4 * 12 // 2
    assert len(dynamics["repair_event_log"]) == dynamics["repair_event_count"]
    assert dynamics["REPAIR_ORDER_REPLAY_EXACT_RECEIPT"]
    assert dynamics["REPAIR_IDEMPOTENCE_REPLAY_EXACT_RECEIPT"]
    assert dynamics["REPAIR_TERMINAL_FIXED_POINT_RECEIPT"]
    assert source["source_architecture"]["all_twelve_ports_sewn_once_per_carrier"]
    assert source["INDEPENDENT_SUPPORT_REGULATOR_RECEIPT"]
    assert source["independent_support_regulator_scope"] == (
        "construction_order_and_source_topology_only"
    )
    assert source["PHYSICAL_INDEPENDENT_SUPPORT_REGULATOR_RECEIPT"] is False
    assert source["SUPPORT_REGULATOR_STRUCTURAL_DIAGNOSTIC_RECEIPT"]
    assert source["CARRIER_REFINEMENT_NATURALITY_RECEIPT"] is False
    assert source["carrier_refinement_naturality_status"] == "NOT_EVALUATED"
    assert source["PHYSICAL_ECHOSAHEDRAL_FEDERATION_REALIZATION_RECEIPT"] is False
    assert source["CARRIER_TO_SUPPORT_CHART_REALIZATION_RECEIPT"] is False


def test_constructed_m4_state_is_not_promoted_to_physical_or_maxent() -> None:
    capture = capture_physical_source(SMALL_CONFIG)
    state = capture["reports"]["prime_geometric_state"]

    assert state["state_mode"] == "constructed_m4_amplitude_gibbs_diagnostic"
    assert capture["config"]["bw"]["state_mode"] == state["state_mode"]
    assert state["algebra_scope"] == "abstract_M4_matrix_diagnostic"
    assert state["state_construction"] == (
        "first_16_presentation_ordered_amplitudes_hermitian_gibbs_map"
    )
    assert state["state_status"] == "CONSTRUCTED_DIAGNOSTIC_ONLY"
    assert state["PHYSICAL_PRIME_GEOMETRIC_CAP_STATE_RECEIPT"] is False
    assert state["MAXIMUM_ENTROPY_STATE_DERIVATION_RECEIPT"] is False
    assert state["SOURCE_SELECTION_A5_QUOTIENT_INVARIANCE_RECEIPT"] is False
    assert state["noncommutative_algebra"]
    assert state["noncommutative_algebra_scope"] == (
        "abstract_M4_matrix_algebra_only"
    )
    assert state["source_selection_rule"] == (
        "first_16_flattened_terminal_complex_amplitudes"
    )
    assert state["surrogate_inputs"] == [
        "presentation_ordered_first_16_terminal_complex_amplitudes"
    ]


def test_refinement_keeps_geometry_receipts_but_quarantines_repeated_rho() -> None:
    capture = capture_physical_source(SMALL_CONFIG)
    refinement = capture["reports"]["refinement"]

    assert refinement["nested_lineage_receipt"]
    assert refinement["TRUE_ICOSAHEDRAL_REFINEMENT_TOWER_RECEIPT"]
    assert refinement["A5_EQUIVARIANT_REFINEMENT_RECEIPT"]
    assert refinement["COMMUTATIVE_CELL_REFINEMENT_DIAGNOSTIC_RECEIPT"]
    assert refinement["CONSTRUCTED_REPEATED_RHO_IDENTITY_RECEIPT"]
    assert refinement["conditional_expectations_receipt"] is False
    assert refinement["PAPER_MULTIRESOLUTION_REGULATOR_CERTIFICATE"] is False
    assert refinement["conditional_expectation_status"] == (
        "NOT_ESTABLISHED_REPEATED_CONSTRUCTED_RHO_ONLY"
    )
    assert "same_constructed_rho_repeated_on_every_child_fiber" in refinement[
        "certificate_blockers"
    ]
    for row in refinement["conditional_expectations"]:
        assert row["unital"]
        assert row["positive"]
        assert row["left_inverse_residual"] <= 1.0e-11
        assert row["state_preserving"] is False
        assert row["cap_isotony_compatible"] is False
        assert row["noncommutative_prime_cap_expectation"] is False


def test_source_derived_geometry_is_target_blind_but_not_independent() -> None:
    capture = capture_physical_source(SMALL_CONFIG)
    geometry = capture["reports"]["independent_geometry"]

    assert geometry["target_blind_derivation"]
    assert geometry["independent_of_modular_fit"]
    assert geometry["independent_of_kms_target"]
    assert geometry["source_primitive_fields"] == [
        "initial_port_intensities",
        "deterministic_index_schedule",
    ]
    assert geometry["source_phase"] == "pre_intervention_initial_state"
    assert geometry["derivation_scope"] == "same_source_self_derived_diagnostic"
    assert geometry["orientation_fixed_from_source"] is False
    assert geometry["orientation_status"] == "CONSTRUCTED_BY_ASCENDING_SORT"
    assert geometry["INDEPENDENT_GEOMETRY_PRODUCER_RECEIPT"] is False
    assert geometry["independent_geometry_producer_status"] == (
        "NOT_INDEPENDENT_SAME_SOURCE_CAPTURE"
    )
    assert geometry["kms_score_row_ids"] == []
    assert geometry["heldout_control_row_ids"]
    assert capture["reports"]["source_observer"][
        "source_forbidden_target_hits"
    ] == []


def test_semantic_event_keys_are_explicitly_presentation_bound() -> None:
    capture = capture_physical_source(SMALL_CONFIG)
    source = capture["reports"]["source_observer"]
    semantic = capture["postrun_capture"]["semantic_events"]
    raw_event_ids = {
        row["event_id"]
        for row in capture["source_artifacts"]["observer_log"]["events"]
    }

    assert source["CARRIER_QUOTIENT_INVARIANCE_RECEIPT"]
    assert source["carrier_quotient_invariance_scope"] == (
        "federation_topology_only_excludes_semantic_event_identity"
    )
    assert source["SEMANTIC_EVENT_A5_QUOTIENT_INVARIANCE_RECEIPT"] is False
    assert source["semantic_event_identity_status"] == (
        "PRESENTATION_BOUND_DIAGNOSTIC_KEY_ONLY"
    )
    assert "presentation_port_index" in source["semantic_event_identity_basis"]
    assert {row["event_key"] for row in semantic} == raw_event_ids


def test_federation_is_connected_all_port_and_support_independent() -> None:
    capture = capture_physical_source(SMALL_CONFIG)
    bundle = capture["source_artifacts"]["federation_bundle"]
    seams = bundle["seams"]
    occupancy: dict[str, set[int]] = {
        carrier_id: set() for carrier_id in bundle["carrier_ids"]
    }
    adjacency: dict[str, set[str]] = {
        carrier_id: set() for carrier_id in bundle["carrier_ids"]
    }
    for seam in seams:
        left = seam["left_carrier_id"]
        right = seam["right_carrier_id"]
        occupancy[left].update(seam["left_ports"])
        occupancy[right].update(seam["right_ports"])
        adjacency[left].add(right)
        adjacency[right].add(left)
    assert len(seams) == SMALL_CONFIG["carrier_count"] * 12 // 2
    assert bundle["external_boundaries"] == []
    assert all(ports == set(range(12)) for ports in occupancy.values())
    reached = {bundle["carrier_ids"][0]}
    frontier = list(reached)
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency[node] - reached:
            reached.add(neighbor)
            frontier.append(neighbor)
    assert reached == set(bundle["carrier_ids"])

    wider_support = capture_physical_source(
        {**SMALL_CONFIG, "observer_support_size": 3}
    )
    assert wider_support["source_artifacts"]["federation_bundle"]["seams"] == seams


def test_frozen_cycles_execute_real_versioned_atomic_repair() -> None:
    dynamics = capture_physical_source(SMALL_CONFIG)["source_artifacts"]["dynamics"]
    assert dynamics["cycles"] == SMALL_CONFIG["cycles"]
    assert len(dynamics["repair_cycle_ledger"]) == SMALL_CONFIG["cycles"]
    assert dynamics["repair_count_per_cycle"] == 2
    assert dynamics["record_commit_schedule"] == [3, 7, 11, 15]
    assert len(dynamics["record_state_snapshots"]) == 4
    assert dynamics["repair_noop_count"] > 0
    assert dynamics["repair_event_count"] + dynamics["repair_noop_count"] == (
        SMALL_CONFIG["cycles"] * dynamics["repair_count_per_cycle"]
    )
    assert dynamics["TRANSACTION_VALIDATION_COMPLETE_READ_CONFLICT_SET_RECEIPT"]
    assert dynamics["UNION_PAYLOAD_ATOMIC_REVALIDATION_RECEIPT"]
    for event in dynamics["repair_event_log"]:
        assert event["strict_descent"]
        assert event["mismatch_after"] < event["mismatch_before"]
        assert len(event["read_set"]) == len(event["write_set"]) == 2
        for read, write in zip(event["read_set"], event["write_set"], strict=True):
            assert write["expected_version"] == read["version"]
            assert write["committed_version"] == read["version"] + 1


def test_repair_digest_is_complete_when_event_examples_are_bounded() -> None:
    capture = capture_physical_source(
        {
            **SMALL_CONFIG,
            "carrier_count": 64,
            "rung": 64,
            "observer_support_size": 4,
        }
    )
    dynamics = capture["source_artifacts"]["dynamics"]
    assert dynamics["repair_event_count"] == 64 * 12 // 2
    assert dynamics["repair_event_examples_complete"] is False
    assert len(dynamics["repair_event_log"]) == dynamics["repair_event_example_limit"]
    assert verify_physical_source_capture(capture)["SOURCE_CAPTURE_REPLAY_RECEIPT"]


def test_records_bind_full_c12_and_checkpoint_restarts_from_saved_prefix() -> None:
    capture = capture_physical_source(SMALL_CONFIG)
    observer = capture["source_artifacts"]["observer_log"]
    records = [row for row in observer["events"] if row["kind"] == "RECORD_COMMIT"]
    reads = [row for row in observer["events"] if row["kind"] == "READBACK"]
    assert observer["checkpoint_replay_exact"]
    assert observer["checkpoint"]["requested_checkpoint_interval"] == 3
    assert observer["checkpoint"]["cut_unit_index"] == 3
    assert observer["checkpoint"]["saved_continuation_state"]
    assert all(len(row["full_port_state"]) == 12 for row in records)
    assert all(row["full_port_state_sha256"] == _sha(row["full_port_state"]) for row in records)
    assert all(row["record_signature_matches_source_field"] for row in reads)
    assert observer["full_port_record_signature_count"] == len(records)
    assert observer["readback_source_signature_match_count"] == len(reads)

    semantic = capture["postrun_capture"]["semantic_events"]
    assert all(
        "sequence_index" not in row["canonical_semantic_payload"]
        for row in semantic
    )
    record_semantic = next(
        row
        for row in semantic
        if row["canonical_semantic_payload"]["event_kind"] == "RECORD_COMMIT"
    )
    assert len(record_semantic["visible_footprint"]) == 12


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("cycles", 17),
        ("repair_fraction_per_cycle", 0.125),
        ("record_commit_cycles", 5),
        ("checkpoint_interval", 2),
    ],
)
def test_every_behavior_affecting_frozen_value_changes_capture(
    key: str, value: object
) -> None:
    baseline = capture_physical_source(SMALL_CONFIG)
    changed = capture_physical_source({**SMALL_CONFIG, key: value})
    assert changed["input_config"][key] == value
    assert changed["capture_sha256"] != baseline["capture_sha256"]


def test_capture_rejects_target_or_scoring_configuration() -> None:
    with pytest.raises(ValueError, match="unknown source-capture config fields"):
        capture_physical_source({**SMALL_CONFIG, "target_scale": 6.28})
    with pytest.raises(ValueError, match="unknown source-capture config fields"):
        capture_physical_source({**SMALL_CONFIG, "candidate": "preferred"})
    with pytest.raises(ValueError, match="unknown source-capture config fields"):
        capture_physical_source({**SMALL_CONFIG, "postrun_capture": {}})
    with pytest.raises(ValueError, match="scoring or model label"):
        capture_physical_source({**SMALL_CONFIG, "replicate_id": "h3"})

    capture = capture_physical_source(SMALL_CONFIG)
    source = capture["reports"]["source_observer"]
    assert source["source_generator_target_free"]
    assert source["source_forbidden_target_hits"] == []


def test_hash_and_exact_replay_detect_tampering() -> None:
    capture = capture_physical_source(SMALL_CONFIG)
    tampered = copy.deepcopy(capture)
    tampered["source_artifacts"]["dynamics"]["seed"] += 1

    verification = verify_physical_source_capture(tampered)
    assert not verification["SOURCE_CAPTURE_REPLAY_RECEIPT"]
    assert "capture_sha256_mismatch" in verification["blockers"]
    assert "capture_is_not_exact_replay_output" in verification["blockers"]


def test_postrun_capture_is_strict_neutral_registered_source_material() -> None:
    capture = capture_physical_source(SMALL_CONFIG)
    postrun = capture["postrun_capture"]
    expected_components = {
        "registration",
        "carrier_port_trajectories",
        "intervention_rows",
        "response_rows",
        "clock_pair_input",
        "geometry_samples",
        "geometry_control_rows",
        "semantic_events",
        "raw_overlap_relations",
        "raw_ancestry_relations",
    }
    assert set(postrun) == {
        "schema",
        *expected_components,
        "declared_hashes",
        "primitive_root_sha256",
    }
    assert postrun["schema"] == POSTRUN_CAPTURE_SCHEMA
    assert set(postrun["declared_hashes"]) == expected_components
    for name in expected_components:
        assert postrun["declared_hashes"][name] == _sha(postrun[name])
    assert postrun["primitive_root_sha256"] == _sha(
        {"schema": POSTRUN_CAPTURE_SCHEMA, "components": postrun["declared_hashes"]}
    )
    assert capture["source_hashes"]["postrun_capture"] == postrun[
        "primitive_root_sha256"
    ]

    clock_pair = postrun["clock_pair_input"]
    assert clock_pair["contract"]["status"] == "UNAVAILABLE"
    assert clock_pair["contract"]["join_key_fields"] == [
        "intervention_id",
        "event_id",
        "observer_or_cap_id",
        "refinement_level",
        "trajectory_group_id",
    ]
    assert clock_pair["contract"]["group_key_fields"] == [
        "source_seed",
        "observer_or_cap_id",
        "trajectory_group_id",
    ]
    assert clock_pair["contract"]["minimum_refinement_level_count"] == 2
    assert clock_pair["modular_transport_rows"] == []
    assert clock_pair["geometric_flow_rows"] == []
    assert all(
        "modular_transport_time" not in row
        and "geometric_flow_parameter" not in row
        for row in postrun["response_rows"]
    )
    assert all(
        row["predictor_source_phase"] == "pre_intervention_initial_state"
        and row["response_source_phase"]
        == "post_repair_minus_initial_response"
        and row["predictor_response_field_intersection"] == []
        for row in postrun["geometry_control_rows"]
    )
    assert all(
        not any(
            resource.startswith("event:")
            for resource in [
                *row["read_resource_ids"],
                *row["write_resource_ids"],
            ]
        )
        for row in postrun["semantic_events"]
    )
    assert all(
        row["shared_resource_ids"]
        for row in postrun["raw_ancestry_relations"]
    )

    registration = postrun["registration"]
    source_inputs = {
        "carrier_count": 4,
        "seed": 1729,
        "rung": 4,
        "replicate_id": "primary",
        "preregistered_plan_sha256": "sha256:" + "a" * 64,
        "propagation_steps": 2,
        "intrinsic_step": 0.137,
        "coupling_strength": 1.0,
        "state_space": "normalized_complex_amplitude_in_C12",
        "rng_family": "numpy_generator_pcg64_v1",
        "initialization_distribution": "normalized_complex_gaussian_v1",
        "intrinsic_phase_distribution": "uniform_unit_interval_v1",
        "seam_update_rule": (
            "disjoint_single_port_endpoint_arithmetic_mean_v1"
        ),
        "cycles": 16,
        "repair_fraction_per_cycle": 0.0625,
        "record_commit_cycles": 4,
        "observer_count": 2,
        "observer_support_size": 2,
        "observer_samples": 4,
        "prediction_control": "semantic_hash_shuffle_v1",
        "feedback_enabled": True,
        "checkpoint_interval": 3,
        "support_refinement_level": 1,
        "geometry_sample_count": 4,
    }
    assert registration == {
        "schema": "oph.physical-source-capture.registration.v1",
        "seed": 1729,
        "rung": 4,
        "replicate_id": "primary",
        "carrier_count": 4,
        "support_regulator_count": 80,
        "support_refinement_level": 1,
        "observer_count": 2,
        "observer_support_size": 2,
        "preregistered_plan_sha256": "sha256:" + "a" * 64,
        "source_inputs": source_inputs,
        "source_inputs_sha256": _sha(source_inputs),
    }

    forbidden_keys = {
        "candidate",
        "candidates",
        "candidate_labels",
        "event_position",
        "geometries",
        "h3_frame",
        "lorentz",
        "model",
        "models",
        "relation",
        "selected_model",
        "selected_scale",
        "target",
        "translation",
    }
    forbidden_labels = {"1x", "pi", "2pi", "4pi", "h3", "s2", "e3", "e4", "lorentz", "causal", "spacelike", "null"}

    def audit(value: object) -> None:
        if isinstance(value, dict):
            assert not (set(value) & forbidden_keys)
            for item in value.values():
                audit(item)
        elif isinstance(value, list):
            for item in value:
                audit(item)
        elif isinstance(value, str):
            for label in forbidden_labels:
                assert re.search(
                    rf"(?<![a-z0-9]){re.escape(label)}(?![a-z0-9])",
                    value.lower(),
                ) is None

    for component in expected_components:
        audit(postrun[component])

    assert capture_physical_source(registration["source_inputs"]) == capture


def test_postrun_responses_are_recomputed_from_source_trajectories() -> None:
    capture = capture_physical_source(SMALL_CONFIG)
    postrun = capture["postrun_capture"]
    trajectories = {
        row["carrier_id"]: row for row in postrun["carrier_port_trajectories"]
    }
    records = {
        row["event_id"]: row
        for row in capture["source_artifacts"]["observer_log"]["events"]
        if row["kind"] == "RECORD_COMMIT"
    }
    for row in postrun["response_rows"]:
        trajectory = trajectories[row["carrier_id"]]
        record = records[row["record_event_id"]]
        port = row["port"]
        expected = (
            record["full_port_state"][port]
            - trajectory["initial_port_intensities"][port]
        )
        assert row["raw_response"] == pytest.approx(expected, abs=2.0e-15)
        assert row["repaired_port_intensity"] == record["port_value"]
        assert row["settled_port_intensity"] == trajectory[
            "settled_port_intensities"
        ][port]


def test_declared_pcg64_complex_gaussian_initializer_is_executable() -> None:
    capture = capture_physical_source(SMALL_CONFIG)
    actual = capture["source_artifacts"]["dynamics"]["initial_port_amplitudes"]
    rng = np.random.Generator(np.random.PCG64(SMALL_CONFIG["seed"]))
    expected = rng.normal(size=(4, 12)) + 1j * rng.normal(size=(4, 12))
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    reconstructed = np.asarray(
        [[complex(real, imag) for real, imag in row] for row in actual]
    )

    assert reconstructed == pytest.approx(expected, abs=1.0e-14)


def test_observer_parent_chains_are_independent_and_actions_are_recomputed() -> None:
    capture = capture_physical_source(SMALL_CONFIG)
    events = capture["source_artifacts"]["observer_log"]["events"]
    by_observer: dict[str, list[dict]] = {}
    for row in events:
        by_observer.setdefault(row["observer_token"], []).append(row)

    for observer_rows in by_observer.values():
        records = [row for row in observer_rows if row["kind"] == "RECORD_COMMIT"]
        feedback = [row for row in observer_rows if row["kind"] == "LOCAL_FEEDBACK"]
        assert records[0]["parents"] == []
        for index in range(1, len(records)):
            assert records[index]["parents"] == [feedback[index - 1]["event_id"]]
            assert records[index]["port"] == feedback[index - 1][
                "observed_action_material_next_port"
            ]
        for row in feedback:
            assert row["observed_action_recomputed_from_record"]
            assert row["predicted_action_material_sha256"] == row[
                "observed_recomputation_material_sha256"
            ]
            assert row["predicted_action"] == row["observed_action"]
            assert row["predicted_action_material_next_port"] == row[
                "observed_action_material_next_port"
            ]
        # The ablation is judged on the concrete action, not on a hash whose
        # domain label was changed by construction. Individual records may be
        # insensitive modulo eleven, but the observer history must not be.
        assert any(
            row["observed_action_material_next_port"]
            != row["ablated_action_material_next_port"]
            for row in feedback
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("carrier_count", "4"),
        ("seed", True),
        ("propagation_steps", 2.0),
        ("observer_count", "2"),
        ("support_refinement_level", 1.0),
    ],
)
def test_integer_config_fields_are_strict(key: str, value: object) -> None:
    with pytest.raises(TypeError, match="exact integer"):
        capture_physical_source({**SMALL_CONFIG, key: value})


@pytest.mark.parametrize(
    ("key", "value", "error_type"),
    [
        ("intrinsic_step", "0.137", TypeError),
        ("intrinsic_step", True, TypeError),
        ("intrinsic_step", float("nan"), ValueError),
        ("coupling_strength", float("inf"), ValueError),
        ("coupling_strength", 0.0, ValueError),
    ],
)
def test_continuous_source_inputs_are_strict_and_finite(
    key: str,
    value: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        capture_physical_source({**SMALL_CONFIG, key: value})


@pytest.mark.parametrize(
    "key",
    [
        "state_space",
        "rng_family",
        "initialization_distribution",
        "intrinsic_phase_distribution",
        "seam_update_rule",
    ],
)
def test_declarative_source_inputs_must_match_executable_literals(key: str) -> None:
    with pytest.raises(ValueError, match="registered executable literal"):
        capture_physical_source({**SMALL_CONFIG, key: "substituted"})


def test_frozen_ladder_bounds_are_representable_without_allocating_large_run() -> None:
    normalized = _normalize_config(
        {
            **SMALL_CONFIG,
            "carrier_count": 262_144,
            "rung": 262_144,
            "support_refinement_level": 7,
        }
    )
    assert normalized["carrier_count"] == 262_144
    assert normalized["support_refinement_level"] == 7
    with pytest.raises(ValueError, match="rung must equal"):
        _normalize_config({**SMALL_CONFIG, "rung": 16})


def test_nested_postrun_tampering_is_rejected_by_exact_replay() -> None:
    capture = capture_physical_source(SMALL_CONFIG)
    tampered = copy.deepcopy(capture)
    tampered["postrun_capture"]["response_rows"][0]["raw_response"] += 0.5

    verification = verify_physical_source_capture(tampered)
    assert not verification["SOURCE_CAPTURE_REPLAY_RECEIPT"]
    assert "capture_sha256_mismatch" in verification["blockers"]
    assert "capture_is_not_exact_replay_output" in verification["blockers"]


def test_capture_keeps_constructed_p1_p3_diagnostics_fail_closed() -> None:
    preflight = physical_h3_kms_preflight_report(
        capture_physical_source(SMALL_CONFIG)
    )
    p0 = preflight["stages"]["P0_source_dynamics_repair_record_observer"]
    # The local carrier/repair/observer instrument is structurally complete,
    # while the distinct physical carrier-to-support realization remains an
    # explicit NOT_EVALUATED field.  A mapping fixture cannot promote either.
    assert p0["passed"]
    assert p0["gate_status"] == "PASS"
    assert p0["scientific_status"] == "NOT_EVALUATED"
    assert p0["evidence"]["carrier_refinement_naturality_receipt"] is False
    assert p0["evidence"]["physical_federation_realization_receipt"] is False
    assert p0["evidence"]["carrier_to_support_realization_receipt"] is False
    refinement = preflight["stages"][
        "P1_nested_refinement_and_expectations"
    ]
    state = preflight["stages"]["P2_prime_geometric_cap_state"]
    geometry = preflight["stages"]["P3_independent_geometric_parameter"]
    for stage in (refinement, state, geometry):
        assert stage["passed"] is False
        assert stage["gate_status"] == "BLOCKED"
        assert stage["scientific_status"] == "NOT_EVALUATED"
    assert "full_noncommutative_multiresolution_certificate_missing" in refinement[
        "blockers"
    ]
    assert "state_is_not_source_maxent_cap_state" in state["blockers"]
    assert "geometry_and_kms_row_partitions_missing" in geometry["blockers"]
    assert not preflight["stages"]["P4_native_bw01_bw08"]["passed"]
    assert not preflight["PHYSICAL_H3_KMS_PREFLIGHT_RECEIPT"]


def test_writer_emits_exact_replayable_capture(tmp_path) -> None:
    output_path = tmp_path / "source-capture.json"
    written = write_physical_source_capture(output_path, SMALL_CONFIG)

    assert output_path.is_file()
    assert verify_physical_source_capture(written)["SOURCE_CAPTURE_REPLAY_RECEIPT"]
