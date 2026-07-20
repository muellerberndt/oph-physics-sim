"""Exact diagonal client for the machine-checked finite event algebra.

The simulator's committed central records define a particularly simple
finite event algebra: each basis record belongs to exactly one declared event
block.  The corresponding projectors are diagonal zero/one matrices.  This
module checks that specialization with exact rational arithmetic and exposes
the Lüders and partition-pinching conclusions that are valid for it.

This is intentionally narrower than the Lean development over arbitrary
complex matrices.  It does not identify pinching with repair, choose a member
of an ambiguous observation fiber, authenticate an external run census, or
promote geometry, particles, gravity, or scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Generic, Hashable, Sequence, TypeVar


EventT = TypeVar("EventT", bound=Hashable)
ExactRational = int | Fraction

FINITE_EVENT_ALGEBRA_VERIFIER_VERSION = "oph-fpe-diagonal-event-algebra-v1"
FINITE_EVENT_ALGEBRA_CLAIM_BOUNDARY = (
    "Exact result for the declared finite diagonal record partition. The "
    "external record census and theorem archive are not authenticated here; "
    "pinching is not repair or representative selection, and no physical "
    "claim is promoted."
)
_MAX_DIMENSION = 4_096


@dataclass(frozen=True)
class EventWeight(Generic[EventT]):
    event: EventT
    weight: Fraction
    basis_multiplicity: int


@dataclass(frozen=True)
class DiagonalEventAlgebraAudit(Generic[EventT]):
    dimension: int
    event_count: int
    event_weights: tuple[EventWeight[EventT], ...]
    selected_event: EventT | None
    selected_born_weight: Fraction | None
    lueders_state_weights: tuple[Fraction, ...] | None
    exact_state_positive_semidefinite: bool
    exact_state_unit_trace: bool
    exact_projective_partition: bool
    projection_events_hermitian_idempotent: bool
    projections_pairwise_orthogonal: bool
    projections_sum_to_identity: bool
    selected_weight_nonzero: bool
    lueders_state_interface_applicable: bool
    lueders_output_is_state: bool
    selected_event_certain_after_update: bool
    lueders_update_idempotent: bool
    lueders_fixed_iff_born_weight_one: bool
    partition_pinching_unital: bool
    partition_pinching_trace_preserving: bool
    partition_pinching_idempotent: bool
    partition_pinching_exact_range_is_commutant: bool
    partition_pinching_positive_by_checked_theorem: bool
    partition_pinching_completely_positive_receipt: bool
    representative_selection_receipt: bool
    repair_receipt: bool
    state_space_complete_declared: bool
    external_record_census_authenticated: bool
    theorem_archive_hash_bundle_authenticated: bool
    conditional_mathematical_receipt: bool
    physical_promotion: bool = False
    scale_authorized: bool = False
    demo_assumption_accepted_as_evidence: bool = False
    verifier_version: str = FINITE_EVENT_ALGEBRA_VERIFIER_VERSION
    theorem_reference: str = (
        "machine_checked_finite_event_algebras.tex:prop:lueders,"
        "thm:pinching-range; Lean EventAlgebra.Lueders and "
        "EventAlgebra.PartitionPinching"
    )
    claim_boundary: str = FINITE_EVENT_ALGEBRA_CLAIM_BOUNDARY


@dataclass(frozen=True)
class ExactPinchingAudit(Generic[EventT]):
    dimension: int
    event_labels: tuple[EventT, ...]
    input_operator: tuple[tuple[Fraction, ...], ...]
    pinched_operator: tuple[tuple[Fraction, ...], ...]
    trace_preserved: bool
    idempotent: bool
    fixed_by_pinching: bool
    commutes_with_every_event_projector: bool
    fixed_iff_in_commutant: bool
    hilbert_schmidt_pythagoras: bool
    hilbert_schmidt_contraction: bool
    physical_promotion: bool = False
    scale_authorized: bool = False
    verifier_version: str = FINITE_EVENT_ALGEBRA_VERIFIER_VERSION
    claim_boundary: str = FINITE_EVENT_ALGEBRA_CLAIM_BOUNDARY


def verify_diagonal_record_event_algebra(
    event_labels: Sequence[EventT],
    state_weights: Sequence[ExactRational],
    *,
    selected_event: EventT | None = None,
    state_space_complete: bool = False,
) -> DiagonalEventAlgebraAudit[EventT]:
    """Verify the exact record-partition/Lüders specialization.

    ``state_space_complete`` is only a caller declaration about an external
    census; it is surfaced but never upgraded to authenticated evidence. The
    exact diagonal mathematics can pass while production provenance remains
    false.
    """

    labels = tuple(event_labels)
    if not labels:
        raise ValueError("event_labels must be nonempty")
    if len(labels) > _MAX_DIMENSION:
        raise ValueError(f"event_labels exceeds verifier limit {_MAX_DIMENSION}")
    if len(state_weights) != len(labels):
        raise ValueError("state_weights length must equal event_labels length")
    if not isinstance(state_space_complete, bool):
        raise TypeError("state_space_complete must be bool")
    for label in labels:
        try:
            hash(label)
        except TypeError as exc:
            raise TypeError("event labels must be hashable") from exc

    weights = tuple(_exact_fraction(value) for value in state_weights)
    state_positive = all(value >= 0 for value in weights)
    state_unit_trace = sum(weights, Fraction(0)) == 1

    ordered_events: list[EventT] = []
    seen: set[EventT] = set()
    for label in labels:
        if label not in seen:
            seen.add(label)
            ordered_events.append(label)
    event_rows = tuple(
        EventWeight(
            event=event,
            weight=sum(
                (weight for label, weight in zip(labels, weights, strict=True) if label == event),
                Fraction(0),
            ),
            basis_multiplicity=sum(label == event for label in labels),
        )
        for event in ordered_events
    )

    selected_weight: Fraction | None = None
    updated: tuple[Fraction, ...] | None = None
    selected_exists = selected_event is not None and selected_event in seen
    if selected_event is not None and not selected_exists:
        raise ValueError("selected_event is not present in event_labels")
    if selected_exists:
        selected_weight = next(
            row.weight for row in event_rows if row.event == selected_event
        )
        if selected_weight != 0:
            updated = tuple(
                weight / selected_weight if label == selected_event else Fraction(0)
                for label, weight in zip(labels, weights, strict=True)
            )

    partition_exact = bool(ordered_events)
    selected_nonzero = selected_weight is not None and selected_weight != 0
    lueders_applicable = bool(
        state_positive and state_unit_trace and selected_exists and selected_nonzero
    )
    output_is_state = bool(
        lueders_applicable
        and updated is not None
        and all(value >= 0 for value in updated)
        and sum(updated, Fraction(0)) == 1
    )
    event_certain = bool(
        output_is_state
        and updated is not None
        and sum(
            (
                weight
                for label, weight in zip(labels, updated, strict=True)
                if label == selected_event
            ),
            Fraction(0),
        )
        == 1
    )
    repeated = None
    if updated is not None and selected_nonzero:
        repeated_weight = sum(
            (
                weight
                for label, weight in zip(labels, updated, strict=True)
                if label == selected_event
            ),
            Fraction(0),
        )
        if repeated_weight != 0:
            repeated = tuple(
                weight / repeated_weight if label == selected_event else Fraction(0)
                for label, weight in zip(labels, updated, strict=True)
            )
    lueders_idempotent = bool(output_is_state and repeated == updated)
    fixed = updated == weights if updated is not None else False
    fixed_iff_one = bool(
        state_positive
        and state_unit_trace
        and selected_exists
        and (fixed == (selected_weight == 1))
    )
    exact_math = bool(
        state_positive
        and state_unit_trace
        and partition_exact
        and (selected_event is None or lueders_applicable)
        and (selected_event is None or output_is_state)
        and (selected_event is None or event_certain)
        and (selected_event is None or lueders_idempotent)
        and (selected_event is None or fixed_iff_one)
    )

    return DiagonalEventAlgebraAudit(
        dimension=len(labels),
        event_count=len(ordered_events),
        event_weights=event_rows,
        selected_event=selected_event,
        selected_born_weight=selected_weight,
        lueders_state_weights=updated,
        exact_state_positive_semidefinite=state_positive,
        exact_state_unit_trace=state_unit_trace,
        exact_projective_partition=partition_exact,
        projection_events_hermitian_idempotent=partition_exact,
        projections_pairwise_orthogonal=partition_exact,
        projections_sum_to_identity=partition_exact,
        selected_weight_nonzero=selected_nonzero,
        lueders_state_interface_applicable=lueders_applicable,
        lueders_output_is_state=output_is_state,
        selected_event_certain_after_update=event_certain,
        lueders_update_idempotent=lueders_idempotent,
        lueders_fixed_iff_born_weight_one=fixed_iff_one,
        partition_pinching_unital=partition_exact,
        partition_pinching_trace_preserving=partition_exact,
        partition_pinching_idempotent=partition_exact,
        partition_pinching_exact_range_is_commutant=partition_exact,
        partition_pinching_positive_by_checked_theorem=partition_exact,
        partition_pinching_completely_positive_receipt=False,
        representative_selection_receipt=False,
        repair_receipt=False,
        state_space_complete_declared=state_space_complete,
        external_record_census_authenticated=False,
        # The supplied paper's Lean source hashes verify, but its committed
        # archive hash list is stale for BUILD_RECEIPT.md. This narrow checker
        # therefore never emits an archive-authentication receipt.
        theorem_archive_hash_bundle_authenticated=False,
        conditional_mathematical_receipt=exact_math,
    )


def verify_exact_diagonal_partition_pinching(
    event_labels: Sequence[EventT],
    operator: Sequence[Sequence[ExactRational]],
) -> ExactPinchingAudit[EventT]:
    """Apply and verify diagonal partition pinching with exact arithmetic."""

    labels = tuple(event_labels)
    if not labels:
        raise ValueError("event_labels must be nonempty")
    if len(labels) > _MAX_DIMENSION:
        raise ValueError(f"event_labels exceeds verifier limit {_MAX_DIMENSION}")
    if len(operator) != len(labels):
        raise ValueError("operator must be square with event-label dimension")
    matrix = tuple(
        tuple(_exact_fraction(value) for value in row) for row in operator
    )
    if any(len(row) != len(labels) for row in matrix):
        raise ValueError("operator must be square with event-label dimension")

    zero = Fraction(0)
    pinched = tuple(
        tuple(
            matrix[left][right] if labels[left] == labels[right] else zero
            for right in range(len(labels))
        )
        for left in range(len(labels))
    )
    repinched = tuple(
        tuple(
            pinched[left][right] if labels[left] == labels[right] else zero
            for right in range(len(labels))
        )
        for left in range(len(labels))
    )
    commutes = all(
        matrix[left][right] == 0
        for left in range(len(labels))
        for right in range(len(labels))
        if labels[left] != labels[right]
    )
    fixed = pinched == matrix
    input_hs = sum(
        (value * value for row in matrix for value in row), Fraction(0)
    )
    pinched_hs = sum(
        (value * value for row in pinched for value in row), Fraction(0)
    )
    residual_hs = sum(
        (
            (matrix[left][right] - pinched[left][right]) ** 2
            for left in range(len(labels))
            for right in range(len(labels))
        ),
        Fraction(0),
    )
    return ExactPinchingAudit(
        dimension=len(labels),
        event_labels=labels,
        input_operator=matrix,
        pinched_operator=pinched,
        trace_preserved=sum(
            (matrix[index][index] for index in range(len(labels))), Fraction(0)
        )
        == sum(
            (pinched[index][index] for index in range(len(labels))), Fraction(0)
        ),
        idempotent=repinched == pinched,
        fixed_by_pinching=fixed,
        commutes_with_every_event_projector=commutes,
        fixed_iff_in_commutant=fixed == commutes,
        hilbert_schmidt_pythagoras=input_hs == pinched_hs + residual_hs,
        hilbert_schmidt_contraction=pinched_hs <= input_hs,
    )


def _exact_fraction(value: ExactRational) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError("exact event-algebra entries must be int or Fraction, never float")
    return Fraction(value)
