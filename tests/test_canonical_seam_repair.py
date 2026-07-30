from __future__ import annotations

import copy
from fractions import Fraction
import json

from oph_fpe.dynamics import canonical_seam_repair as cert


REPORT = cert.canonical_seam_repair_certificate()
EDGES = cert.reference_edges()


def test_certificate_is_exactly_recomputable_and_scope_bounded() -> None:
    verification = cert.verify_canonical_seam_repair_certificate(REPORT)

    assert verification["receipt"] is True
    assert REPORT["schema"] == "oph.canonical_seam_repair_certificate.v1"
    assert REPORT[cert.CANONICAL_SEAM_EXPECTATION_RECEIPT] is True
    assert REPORT[cert.CONDITIONAL_UNIFORM_SEAM_SCHEDULE_RECEIPT] is True
    assert REPORT[cert.LAPLACIAN_REPAIR_GENERATOR_RECEIPT] is True
    assert REPORT[cert.FINITE_REPAIR_CONVERGENCE_RECEIPT] is True
    assert REPORT[cert.GLOBAL_A1_A3_POLICY_UNIQUENESS_RECEIPT] is False
    assert REPORT[
        cert.COUPLED_STATE_GENERATOR_UNIQUENESS_RECEIPT
    ] is False
    assert REPORT[
        cert.FULL_SELF_READING_UNIVERSE_CLOSURE_RECEIPT
    ] is False
    assert REPORT[cert.FULL_REFINEMENT_SEMIGROUP_RECEIPT] is False
    assert REPORT[cert.PHYSICAL_REPAIR_LAW_RECEIPT] is False
    assert REPORT["source_inputs"]["laboratory_data_used"] is False
    assert REPORT["source_inputs"][
        "strictly_combinatorial_rotation_enumeration_certified"
    ] is False
    assert verification["producer_replay"] is True
    assert verification["independent_implementation"] is False
    assert REPORT["declared_complete_grammar"][
        "not_a_complete_grammar_of_every_a1_a3_repair_instrument"
    ]


def test_all_thirty_seam_maps_are_exact_conditional_expectations() -> None:
    assert len(EDGES) == 30
    checks = REPORT["conditional_expectation"]["exact_checks"]
    assert checks["edges_checked"] == 30
    assert checks["matrix_entries_checked"] == 30 * 12 * 12

    identity = cert._identity_fraction(12)
    for edge in EDGES:
        expectation = cert.edge_conditional_expectation(edge)
        assert cert._fraction_multiply(expectation, expectation) == expectation
        assert cert._fraction_transpose(expectation) == expectation
        assert all(sum(row, Fraction()) == 1 for row in expectation)
        assert all(
            sum(expectation[row][column] for row in range(12)) == 1
            for column in range(12)
        )
        left, right = edge
        assert expectation[left] == expectation[right]
        for port in range(12):
            if port not in edge:
                assert expectation[port] == identity[port]


def test_endpoint_grammar_has_one_nontrivial_idempotent_retraction() -> None:
    classification = REPORT["conditional_expectation"]["classification"]

    assert classification["general_endpoint_block"] == "[[a, 1-a], [1-a, a]]"
    assert classification["unique_nonidentity_solution"] == "a=1/2"
    assert classification[
        "unique_nonidentity_solution_is_conditional_expectation"
    ]
    assert classification["idempotent_solutions"] == [
        {
            "diagonal": "1",
            "off_diagonal": "0",
            "mismatch_eigenvalue": "1",
            "idempotent": True,
            "range_is_agreement_equalizer": False,
            "disposition": "identity_no_repair",
        },
        {
            "diagonal": "1/2",
            "off_diagonal": "1/2",
            "mismatch_eigenvalue": "0",
            "idempotent": True,
            "range_is_agreement_equalizer": True,
            "disposition": "selected_nontrivial_retraction",
        },
    ]


def test_uniform_schedule_gives_laplacian_generator_exactly() -> None:
    expectations = {
        edge: cert.edge_conditional_expectation(edge)
        for edge in EDGES
    }
    laplacian = cert.graph_laplacian(12, EDGES)
    expected = cert._fraction_subtract(
        cert._identity_fraction(12),
        cert._fraction_scale(
            cert._as_fraction_matrix(laplacian),
            Fraction(1, 60),
        ),
    )

    assert cert._mean_operator(expectations) == expected
    schedule = REPORT["conditional_schedule"]
    assert schedule["selected_probability_per_seam"] == "1/30"
    assert schedule["full_support"] is True
    assert schedule["proper_rotation_count"] == 60
    assert schedule["port_orbit_size"] == 12
    assert schedule["first_seam_orbit_size"] == 30
    assert schedule["first_directed_seam_orbit_size"] == 60
    assert schedule["uniform_reference_derived_from_bare_a1_alone"] is False
    assert schedule["complete_move_simplex_required"] is True
    assert schedule["conditional_expectation_covariance_squares_checked"] == 1800
    classification = REPORT["laplacian_generator"][
        "generator_orbit_classification"
    ]
    assert classification["general_covariant_supported_matrix"] == "a I + b A"
    assert classification["conservation_equation"] == "a + 5 b = 0"
    assert classification["conclusion"] == (
        "G = -b L, unique up to the common rate b"
    )


def test_laplacian_spectrum_and_contraction_are_exactly_typed() -> None:
    spectrum = REPORT["laplacian_generator"]["spectrum"]

    assert spectrum["verified_entrywise_over_integers"] is True
    assert spectrum["adjacency_trace_powers_zero_through_three"] == [
        12,
        0,
        60,
        120,
    ]
    assert spectrum["laplacian_bands"] == [
        {"eigenvalue": "0", "multiplicity": 1},
        {"eigenvalue": "5-sqrt(5)", "multiplicity": 3},
        {"eigenvalue": "6", "multiplicity": 5},
        {"eigenvalue": "5+sqrt(5)", "multiplicity": 3},
    ]
    assert spectrum["expected_step_eigenvalues"] == [
        "1",
        "(55+sqrt(5))/60",
        "9/10",
        "(55-sqrt(5))/60",
    ]
    energy = REPORT["convergence_and_conservation"]["quadratic_disagreement"]
    assert energy["conserved_total_preserved"] is True
    assert energy["external_record_or_checkpoint_instrument_certified"] is False
    assert energy["strict_unless_consensus"] is True
    assert energy["exact_integer_probe_count"] == 30 * 7 * 7


def test_entropy_claim_is_backed_by_exact_double_stochasticity() -> None:
    entropy = REPORT["convergence_and_conservation"]["entropy"]

    assert entropy["majorization"] == "E_e p is majorized by p"
    assert entropy["shannon_entropy"] == "H(E_e p) >= H(p)"
    assert entropy["signed_load_entropy_claimed"] is False

    edge = EDGES[0]
    expectation = cert.edge_conditional_expectation(edge)
    probability = [Fraction(1, 12)] * 12
    probability[edge[0]] = Fraction(1, 6)
    probability[edge[1]] = Fraction(0)
    averaged = cert.apply_fraction_matrix(expectation, probability)
    assert sum(averaged, Fraction()) == 1
    assert averaged[edge[0]] == averaged[edge[1]] == Fraction(1, 12)


def test_atomic_integer_lift_preserves_total_and_averages_in_expectation() -> None:
    even = cert.atomic_integer_expectation_lift(5, 1)
    odd = cert.atomic_integer_expectation_lift(4, 1)

    assert even == ((Fraction(1), (3, 3)),)
    assert odd == (
        (Fraction(1, 2), (2, 3)),
        (Fraction(1, 2), (3, 2)),
    )
    assert all(sum(outcome) == 5 for _, outcome in odd)
    assert sum(
        (weight * outcome[0] for weight, outcome in odd),
        Fraction(),
    ) == Fraction(5, 2)
    boundary = REPORT["convergence_and_conservation"]["atomic_integer_lift"]
    assert boundary["odd_total_pathwise_exact_agreement"] is False
    assert boundary["exact_a2_agreement_retraction"] is False
    assert boundary["range"] == "nearest_balanced_integer_shell"
    assert boundary["expectation_equals_rational_conditional_expectation"] is True
    assert "issue 628" in boundary["issue_628_relation"]


def test_countermodels_identify_each_load_bearing_clause() -> None:
    controls = REPORT["countermodels_and_boundaries"]

    partial = controls["partial_average"]
    assert partial["idempotent"] is False
    assert partial["uniform_generator"] == "-L/120"

    biased = controls["biased_schedule"]
    assert biased["normalized"] is True
    assert biased["a5_covariant"] is False
    assert biased["first_probability"] == "2/31"
    assert biased["rotated_probability"] == "1/31"

    radius_two = controls["radius_two_average"]
    assert radius_two["a5_covariant"] is True
    assert radius_two["strict_seam_local"] is False
    assert radius_two["exact_polynomial_relation"] == (
        "2 A_distance2 = A^2 - 2 A - 5 I"
    )
    assert radius_two["band_action"][
        "adjacency_plus_sqrt5_triplet"
    ] == "5+sqrt(5)"

    atomic = controls["nonlinear_atomic_unit_transfer"]
    assert atomic["sum_preserving"] is True
    assert atomic["linear"] is False
    assert atomic["F_4_0"] != atomic["two_times_F_2_0"]


def test_refinement_has_first_order_readback_but_no_semigroup_intertwiner() -> None:
    boundary = REPORT["refinement_boundary"]

    assert boundary["full_refinement_dynamics_promoted"] is False
    assert len(boundary["rows"]) == 2
    assert [
        (row["coarse_vertex_count"], row["fine_vertex_count"])
        for row in boundary["rows"]
    ] == [(12, 42), (42, 162)]
    for row in boundary["rows"]:
        assert row["restriction_after_interpolation_is_identity"] is True
        assert row["fine_to_coarse_seam_count_ratio"] == "4"
        assert row["raw_laplacian_first_order_identity_exact"] is True
        assert row["per_total_attempt_first_order_identity_exact"] is True
        assert row["raw_laplacian_first_order_inherited_readback"] == (
            "Q Delta_f J = (1/2) Delta_c"
        )
        assert row["per_total_attempt_first_order_inherited_readback"] == (
            "Q K_f J = (1/8) K_c"
        )
        assert row["per_total_attempt_strong_intertwiner"] is False
        assert row["per_total_attempt_strong_intertwiner_witness"][
            "nonzero_entry_count"
        ] > 0
        assert row[
            "per_total_attempt_second_order_semigroup_condition"
        ] is False
        assert row["per_total_attempt_second_order_witness"][
            "nonzero_entry_count"
        ] > 0


def test_verifier_rejects_payload_and_scope_promotion_mutations() -> None:
    mutated = copy.deepcopy(REPORT)
    mutated["laplacian_generator"]["uniform_expected_step"] = "T = I"
    result = cert.verify_canonical_seam_repair_certificate(mutated)
    assert result["receipt"] is False
    assert "payload_hash_mismatch" in result["reasons"]
    assert "producer_replay_mismatch" in result["reasons"]

    promoted = copy.deepcopy(REPORT)
    promoted[cert.PHYSICAL_REPAIR_LAW_RECEIPT] = True
    body = {
        key: value
        for key, value in promoted.items()
        if key != "certificate_payload_sha256"
    }
    promoted["certificate_payload_sha256"] = cert._payload_sha256(body)
    result = cert.verify_canonical_seam_repair_certificate(promoted)
    assert result["receipt"] is False
    assert "forbidden_scope_promotion" in result["reasons"]
    assert "producer_replay_mismatch" in result["reasons"]

    coupled_promoted = copy.deepcopy(REPORT)
    coupled_promoted[
        cert.COUPLED_STATE_GENERATOR_UNIQUENESS_RECEIPT
    ] = True
    coupled_body = {
        key: value
        for key, value in coupled_promoted.items()
        if key != "certificate_payload_sha256"
    }
    coupled_promoted["certificate_payload_sha256"] = cert._payload_sha256(
        coupled_body
    )
    result = cert.verify_canonical_seam_repair_certificate(coupled_promoted)
    assert result["receipt"] is False
    assert "forbidden_scope_promotion" in result["reasons"]
    assert "producer_replay_mismatch" in result["reasons"]

    malformed = copy.deepcopy(REPORT)
    malformed["source_inputs"]["port_count"] = float("nan")
    result = cert.verify_canonical_seam_repair_certificate(malformed)
    assert result["receipt"] is False
    assert "malformed_or_noncanonical_payload" in result["reasons"]

    wrong_top_level = cert.verify_canonical_seam_repair_certificate([])  # type: ignore[arg-type]
    assert wrong_top_level["receipt"] is False
    assert "malformed_or_noncanonical_payload" in wrong_top_level["reasons"]


def test_cli_writes_and_verifies_deterministic_json(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    verification_path = tmp_path / "verification.json"

    assert cert.main(["--output", str(first)]) == 0
    assert cert.main(["--output", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()
    assert cert.main(
        [
            "--verify",
            str(first),
            "--output",
            str(verification_path),
        ]
    ) == 0
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    assert verification["receipt"] is True
