from __future__ import annotations

from fractions import Fraction

import pytest

from oph_fpe.algebra import (
    verify_diagonal_record_event_algebra,
    verify_exact_diagonal_partition_pinching,
)


def test_exact_record_partition_applies_lueders_and_pinching_theorems() -> None:
    audit = verify_diagonal_record_event_algebra(
        ("red", "red", "blue"),
        (Fraction(1, 6), Fraction(1, 3), Fraction(1, 2)),
        selected_event="red",
        state_space_complete=True,
    )

    assert audit.exact_projective_partition is True
    assert audit.projections_pairwise_orthogonal is True
    assert audit.projections_sum_to_identity is True
    assert audit.selected_born_weight == Fraction(1, 2)
    assert audit.lueders_state_weights == (
        Fraction(1, 3),
        Fraction(2, 3),
        Fraction(0),
    )
    assert audit.lueders_output_is_state is True
    assert audit.selected_event_certain_after_update is True
    assert audit.lueders_update_idempotent is True
    assert audit.lueders_fixed_iff_born_weight_one is True
    assert audit.partition_pinching_exact_range_is_commutant is True
    assert audit.conditional_mathematical_receipt is True
    assert audit.state_space_complete_declared is True
    assert audit.external_record_census_authenticated is False
    assert audit.theorem_archive_hash_bundle_authenticated is False
    assert audit.partition_pinching_completely_positive_receipt is False
    assert audit.representative_selection_receipt is False
    assert audit.repair_receipt is False
    assert audit.physical_promotion is False
    assert audit.scale_authorized is False


def test_zero_weight_event_cannot_enter_typed_lueders_interface() -> None:
    audit = verify_diagonal_record_event_algebra(
        ("present", "zero"),
        (1, 0),
        selected_event="zero",
    )

    assert audit.selected_born_weight == 0
    assert audit.selected_weight_nonzero is False
    assert audit.lueders_state_interface_applicable is False
    assert audit.lueders_state_weights is None
    assert audit.conditional_mathematical_receipt is False


def test_fixed_state_equivalence_needs_no_extra_nonzero_guard() -> None:
    certain = verify_diagonal_record_event_algebra(
        ("yes", "no"),
        (1, 0),
        selected_event="yes",
    )

    assert certain.selected_born_weight == 1
    assert certain.lueders_state_weights == (Fraction(1), Fraction(0))
    assert certain.lueders_fixed_iff_born_weight_one is True
    assert certain.conditional_mathematical_receipt is True


def test_exact_pinching_removes_cross_block_entries_and_replays_geometry() -> None:
    audit = verify_exact_diagonal_partition_pinching(
        ("a", "a", "b"),
        (
            (1, 2, 3),
            (4, 5, 6),
            (7, 8, 9),
        ),
    )

    assert audit.pinched_operator == (
        (Fraction(1), Fraction(2), Fraction(0)),
        (Fraction(4), Fraction(5), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(9)),
    )
    assert audit.trace_preserved is True
    assert audit.idempotent is True
    assert audit.fixed_by_pinching is False
    assert audit.commutes_with_every_event_projector is False
    assert audit.fixed_iff_in_commutant is True
    assert audit.hilbert_schmidt_pythagoras is True
    assert audit.hilbert_schmidt_contraction is True
    assert audit.physical_promotion is False


def test_block_diagonal_operator_is_exact_fixed_point_and_commutant_member() -> None:
    audit = verify_exact_diagonal_partition_pinching(
        (0, 0, 1),
        ((1, 2, 0), (3, 4, 0), (0, 0, 5)),
    )

    assert audit.fixed_by_pinching is True
    assert audit.commutes_with_every_event_projector is True
    assert audit.fixed_iff_in_commutant is True


@pytest.mark.parametrize(
    ("labels", "weights", "message"),
    [
        ((), (), "nonempty"),
        (("a",), (Fraction(1, 2), Fraction(1, 2)), "length"),
        (("a",), (1.0,), "never float"),
    ],
)
def test_malformed_exact_event_inputs_fail_closed(labels, weights, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        verify_diagonal_record_event_algebra(labels, weights)


def test_selected_event_must_exist() -> None:
    with pytest.raises(ValueError, match="not present"):
        verify_diagonal_record_event_algebra(("a",), (1,), selected_event="b")
