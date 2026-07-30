from __future__ import annotations

import copy
from fractions import Fraction
import json

from oph_fpe.dynamics import self_readback_repair_closure as closure


REPORT = closure.self_readback_repair_closure_report()


def test_exact_directed_seams_form_a_combinatorial_torsor() -> None:
    torsor = closure.exact_directed_seam_torsor()

    assert torsor["incidence_automorphism_count"] == 120
    assert torsor["orientation_preserving_rotation_count"] == 60
    assert torsor["directed_event_atom_count"] == 60
    assert torsor["directed_seam_orbit_size"] == 60
    assert torsor["reference_stabilizer_order"] == 1
    assert torsor["action_closed"] is True
    assert torsor["action_simply_transitive"] is True
    assert torsor["floating_coordinate_matching_used"] is False
    assert torsor["floating_geometry_source_used"] is False
    assert torsor["exact_oriented_face_count"] == 20
    assert torsor["exact_oriented_faces_distinct"] is True
    assert torsor["every_seam_has_two_incident_faces"] is True
    assert torsor["incident_face_directions_are_opposite"] is True
    assert len(closure.ORIENTED_BASE_FACES) == 20


def test_every_protected_state_through_total_four_is_enumerated() -> None:
    expected = {0: 1, 1: 12, 2: 78, 3: 364, 4: 1365}

    for total, count in expected.items():
        states = closure.enumerate_protected_sector(total)
        assert len(states) == count
        assert len(set(states)) == count
        assert all(len(state) == 12 for state in states)
        assert all(sum(state) == total for state in states)
        assert all(value >= 0 for state in states for value in state)


def test_public_readback_hides_candidate_identity_and_transition_weights() -> None:
    first = closure.CandidateLaw(
        "opaque_name_one",
        "directed_balanced",
        "direct_uniform",
    )
    second = closure.CandidateLaw(
        "unrelated_opaque_name",
        "directed_balanced",
        "direct_uniform",
    )

    first_readback = closure.public_constraint_readback(first)
    second_readback = closure.public_constraint_readback(second)
    encoded = json.dumps(first_readback, sort_keys=True).lower()

    assert first_readback == second_readback
    assert "law_id" not in encoded
    assert "local_policy" not in encoded
    assert "clock_ratio" not in encoded
    assert "transition_probability" not in encoded
    assert "held_out" not in encoded
    assert first_readback["primitive_probe_count"] == 1820 * 60
    assert (
        closure.reconstruct_from_constraint_readback(first_readback)
        is not None
    )

    tainted = copy.deepcopy(first_readback)
    tainted["candidate_law_id"] = "leak"
    assert closure.reconstruct_from_constraint_readback(tainted) is None


def test_directed_balancing_reconstructs_at_every_tested_clock() -> None:
    reference = closure.candidate_transition_rows(
        closure.CandidateLaw(
            "reference",
            "directed_balanced",
            "direct_uniform",
        )
    )
    for clock in (Fraction(1, 4), Fraction(1, 2), Fraction(1)):
        candidate = closure.CandidateLaw(
            f"clock_{clock}",
            "directed_balanced",
            "direct_uniform",
            clock,
        )
        readback = closure.public_constraint_readback(candidate)
        comparison = closure.generator_ray_comparison(
            closure.candidate_transition_rows(candidate),
            reference,
        )

        assert closure.reconstruct_from_constraint_readback(readback) is not None
        assert comparison["equivalent"] is True
        assert comparison["clock_ratio"] == str(clock)


def test_direction_convention_is_quotiented_but_dynamics_is_not_fit_back() -> None:
    reference = closure.candidate_transition_rows(
        closure.CandidateLaw(
            "reference",
            "directed_balanced",
            "direct_uniform",
        )
    )
    reversed_convention = closure.CandidateLaw(
        "reversed",
        "reverse_directed_balanced",
        "direct_uniform",
    )
    incomplete_mixing = closure.CandidateLaw(
        "nearest",
        "nearest_keep_high_side",
        "direct_uniform",
    )

    assert closure.generator_ray_comparison(
        closure.candidate_transition_rows(reversed_convention),
        reference,
    )["equivalent"]

    readback = closure.public_constraint_readback(incomplete_mixing)
    assert closure.reconstruct_from_constraint_readback(readback) is not None
    comparison = closure.generator_ray_comparison(
        closure.candidate_transition_rows(incomplete_mixing),
        reference,
    )
    assert comparison["equivalent"] is False
    assert comparison["reason"] == "nonuniform_generator_ratio"


def test_adversarial_controls_fail_the_clause_that_they_violate() -> None:
    controls = {
        "one_unit": closure.CandidateLaw(
            "one_unit",
            "one_unit_descent",
            "direct_uniform",
        ),
        "edge_bias": closure.CandidateLaw(
            "edge_bias",
            "directed_balanced",
            "direct_edge_biased",
        ),
        "radius_two": closure.CandidateLaw(
            "radius_two",
            "directed_balanced",
            "distance_two_uniform",
        ),
        "port_bias": closure.CandidateLaw(
            "port_bias",
            "lower_index_gets_ceiling",
            "direct_uniform",
        ),
    }
    readbacks = {
        name: closure.public_constraint_readback(candidate)
        for name, candidate in controls.items()
    }

    assert (
        readbacks["one_unit"][
            "all_outcomes_land_in_nearest_agreement_shell"
        ]
        is False
    )
    assert readbacks["edge_bias"]["event_atom_schedule_is_uniform"] is False
    assert (
        readbacks["radius_two"][
            "event_atom_support_is_exact_directed_seam_torsor"
        ]
        is False
    )
    assert readbacks["port_bias"]["local_rule_is_presentation_covariant"] is False
    assert readbacks["port_bias"]["endpoint_reversal_covariant"] is False
    assert (
        closure.reconstruct_from_constraint_readback(readbacks["one_unit"])
        is None
    )
    assert (
        closure.reconstruct_from_constraint_readback(readbacks["port_bias"])
        is None
    )
    # Support and schedule are source-fixed. Observed biased or wrong-radius
    # candidates therefore reach the held-out comparison instead of being
    # filtered by their own reported weights.
    assert (
        closure.reconstruct_from_constraint_readback(readbacks["edge_bias"])
        is not None
    )
    assert (
        closure.reconstruct_from_constraint_readback(readbacks["radius_two"])
        is not None
    )


def test_bounded_coupled_state_generator_closure_is_exact() -> None:
    rows = REPORT["coupled_state_generator_closure"]["sector_rows"]

    assert [row["state_count"] for row in rows] == [1, 12, 78, 364, 1365]
    assert [row["closed_class_size"] for row in rows] == [1, 12, 66, 220, 495]
    assert all(row["closed_component_count"] == 1 for row in rows)
    assert all(row["closed_class_is_exact_balanced_shell"] for row in rows)
    assert all(row["all_states_reach_closed_class"] for row in rows)
    assert all(
        row["energy_nonincreasing_on_every_positive_transition"]
        for row in rows
    )
    assert all(row["closed_class_exactly_doubly_stochastic"] for row in rows)
    assert all(row["unique_stationary_state"] for row in rows)
    mean_bridge = REPORT["exact_conditional_mean_bridge"]
    assert mean_bridge["all_probed_states_exact_identity_verified"] is True
    assert mean_bridge["states_checked"] == 1820
    assert mean_bridge["coordinate_expectations_checked"] == 1820 * 12
    assert mean_bridge["full_integer_transition_kernel_is_linear"] is False
    assert (
        mean_bridge["one_atom_restriction"]["exact_identity_verified"]
        is True
    )
    assert REPORT[closure.BOUNDED_SELF_READBACK_RECEIPT] is True
    assert REPORT[closure.BOUNDED_COUPLED_CLOSURE_RECEIPT] is False


def test_two_event_table_rejects_correlated_uniform_marginals() -> None:
    path_law = REPORT["two_event_schedule_closure"]

    assert path_law["seam_attempt_count"] == 30
    assert path_law["directed_completion_label_count"] == 60
    assert path_law["iid_ordered_attempt_pair_count"] == 900
    assert path_law["iid_probability_per_ordered_attempt_pair"] == "1/900"
    assert path_law["one_attempt_marginal_probability"] == "1/30"
    assert path_law["odd_tie_probability_per_completion"] == "1/2"
    assert path_law["iid_path_kernel_equals_P_squared"] is True
    assert (
        path_law["repeat_same_control_has_same_one_attempt_marginal"]
        is True
    )
    assert path_law["repeat_same_control_differs_from_P_squared"] is True
    assert path_law["repeat_same_differing_matrix_entries"] > 0
    assert (
        path_law["iid_two_step_sha256"]
        != path_law["repeat_same_two_step_sha256"]
    )
    free_words = REPORT["conditional_free_event_word_law"]
    assert free_words["uniform_word_law_factorizes_for_every_length"] is True
    assert free_words["canonical_a3_alone_implies_markovity"] is False
    assert free_words["proposed_a1r_a2r_temporal_clauses_required"] is True
    assert REPORT[closure.CONDITIONAL_FREE_WORD_LAW_RECEIPT] is True


def test_signed_cube_exercises_the_general_progress_construction() -> None:
    signed = REPORT["bounded_signed_state_control"]

    assert signed["coordinate_alphabet"] == [-1, 0, 1]
    assert signed["states_checked"] == 3**12
    assert signed["protected_totals_covered"] == list(range(-12, 13))
    assert signed["balanced_shell_states"] == 8191
    assert (
        signed["non_shell_states_with_strict_progress_word"]
        == 3**12 - 8191
    )
    assert signed["maximum_events_before_strict_descent"] <= 3
    assert signed["all_non_shell_states_have_strict_progress_word"] is True


def test_one_unit_control_has_many_absorbing_terminal_states() -> None:
    control = REPORT["adversarial_controls"]["one_unit_rule"]

    assert control["constraint_reconstruction_eligible"] is False
    assert control["fixed_modulo_clock"] is False
    assert control["absorbing_state_count_by_total"] == {
        "0": 1,
        "1": 12,
        "2": 66,
        "3": 220,
        "4": 495,
    }


def test_total_twelve_diagnostics_reach_the_sourced_singleton() -> None:
    total_twelve = REPORT["distinguished_total_twelve_sector"]

    assert total_twelve["protected_total"] == 12
    assert total_twelve["full_nonnegative_composition_count"] == 1_352_078
    assert total_twelve["minimum_energy"] == 12
    assert total_twelve["next_energy_floor"] == 14
    assert total_twelve["minimum_shell_cardinality"] == 1
    assert total_twelve["unique_minimum_state"] == [1] * 12
    assert total_twelve["source_packet_imported_or_hash_verified_here"] is True
    assert total_twelve["source_projection"]["verified"] is True
    assert total_twelve["source_projection"]["source_issue"] == 628
    assert total_twelve["source_projection"]["source_generation"] == (
        "twelve atomic +1 writes at port p00"
    )
    assert total_twelve["source_projection"]["termination_lyapunov"] == (
        "load square V(N) = sum_i N_i^2"
    )
    exhaustive = total_twelve["exhaustive_progress_audit"]
    assert exhaustive["states_checked"] == 1_352_078
    assert exhaustive["unique_minimum_state_count"] == 1
    assert (
        exhaustive["nonminimum_states_with_strict_descent_path"]
        == 1_352_077
    )
    assert exhaustive["maximum_events_before_strict_descent"] <= 3
    assert exhaustive["all_nonminimum_states_have_strict_descent_path"] is True
    assert exhaustive["old_one_unit_absorbing_state_count"] == 303
    assert exhaustive["old_one_unit_nonglobal_absorbing_state_count"] == 302
    assert (
        exhaustive["canonical_unique_closed_class_follows_from_full_support"]
        is True
    )
    assert (
        total_twelve[
            "all_nonnegative_states_reach_unique_minimum_almost_surely_under_uniform_iid_attempt_law"
        ]
        is True
    )
    assert total_twelve["reconstructed_law_progress_only"] is True
    assert (
        total_twelve["candidate_conformance_exhausted_on_total_twelve"]
        is False
    )
    assert (
        total_twelve["all_seeded_trajectories_settled_to_unique_minimum"]
        is True
    )
    assert len(total_twelve["trajectory_rows"]) == 5
    assert all(
        row["settled_to_all_ones"]
        and row["final_state"] == [1] * 12
        and row["final_energy"] == 12
        for row in total_twelve["trajectory_rows"]
    )


def test_report_has_one_fixed_ray_class_and_keeps_global_claims_false() -> None:
    suite = REPORT["candidate_suite"]

    assert REPORT[closure.DIRECTED_SEAM_TORSOR_RECEIPT] is True
    assert REPORT[closure.BOUNDED_SELF_READBACK_RECEIPT] is True
    assert REPORT[closure.BOUNDED_COUPLED_CLOSURE_RECEIPT] is False
    assert suite["fixed_candidate_count"] == 6
    assert suite["fixed_generator_ray_class_count"] == 1
    assert REPORT[closure.GLOBAL_POLICY_RECEIPT] is False
    assert REPORT[closure.FULL_SELF_READBACK_RECEIPT] is False
    assert REPORT[closure.PHYSICAL_REPAIR_RECEIPT] is False
    assert REPORT["source_inputs"]["laboratory_data_used"] is False
    assert REPORT["source_inputs"]["downstream_target_used"] is False
    custody = REPORT["anti_circular_reconstruction"]
    assert custody["observed_event_support_used_to_construct_law"] is False
    assert custody["observed_schedule_weights_used_to_construct_law"] is False
    assert (
        REPORT["adversarial_controls"][
            "biased_schedule_reaches_held_out_scoring"
        ]
        is True
    )
    assert (
        REPORT["adversarial_controls"][
            "biased_schedule_rejected_by_held_out_generator"
        ]
        is True
    )
    assert (
        REPORT["adversarial_controls"][
            "radius_two_reaches_held_out_scoring"
        ]
        is True
    )
    assert (
        REPORT["adversarial_controls"][
            "radius_two_rejected_by_held_out_generator"
        ]
        is True
    )


def test_stored_reference_receipt_is_byte_exact_and_verifies() -> None:
    stored = json.loads(
        closure.REFERENCE_REPORT_PATH.read_text(encoding="utf-8")
    )

    assert stored == REPORT
    assert closure.REFERENCE_REPORT_PATH.read_bytes() == (
        json.dumps(REPORT, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    verification = closure.verify_self_readback_repair_closure_report(stored)
    assert verification["receipt"] is True
    assert verification["producer_replay"] is True
    assert verification["independent_implementation"] is False


def test_verifier_rejects_mutation_promotion_and_malformed_payload() -> None:
    assert closure.verify_self_readback_repair_closure_report(REPORT)[
        "receipt"
    ]

    mutated = copy.deepcopy(REPORT)
    mutated["exact_conditional_mean_bridge"]["identity"] = "P=I"
    result = closure.verify_self_readback_repair_closure_report(mutated)
    assert result["receipt"] is False
    assert "payload_hash_mismatch" in result["reasons"]
    assert "producer_replay_mismatch" in result["reasons"]

    promoted = copy.deepcopy(REPORT)
    promoted[closure.PHYSICAL_REPAIR_RECEIPT] = True
    body = {
        key: value
        for key, value in promoted.items()
        if key != "certificate_payload_sha256"
    }
    promoted["certificate_payload_sha256"] = closure._sha256_json(body)
    result = closure.verify_self_readback_repair_closure_report(promoted)
    assert result["receipt"] is False
    assert "forbidden_scope_promotion" in result["reasons"]

    promoted = copy.deepcopy(REPORT)
    promoted[closure.BOUNDED_COUPLED_CLOSURE_RECEIPT] = True
    body = {
        key: value
        for key, value in promoted.items()
        if key != "certificate_payload_sha256"
    }
    promoted["certificate_payload_sha256"] = closure._sha256_json(body)
    result = closure.verify_self_readback_repair_closure_report(promoted)
    assert result["receipt"] is False
    assert "forbidden_scope_promotion" in result["reasons"]

    malformed = copy.deepcopy(REPORT)
    malformed["source_inputs"]["total_state_count"] = float("nan")
    result = closure.verify_self_readback_repair_closure_report(malformed)
    assert result["receipt"] is False
    assert "malformed_or_noncanonical_payload" in result["reasons"]

    result = closure.verify_self_readback_repair_closure_report([])  # type: ignore[arg-type]
    assert result["receipt"] is False
    assert "malformed_or_noncanonical_payload" in result["reasons"]


def test_cli_output_is_byte_deterministic_and_verifies(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    verification = tmp_path / "verification.json"

    assert closure.main(["--output", str(first)]) == 0
    assert closure.main(["--output", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()
    assert closure.main(
        [
            "--verify",
            str(first),
            "--output",
            str(verification),
        ]
    ) == 0
    result = json.loads(verification.read_text(encoding="utf-8"))
    assert result["receipt"] is True
