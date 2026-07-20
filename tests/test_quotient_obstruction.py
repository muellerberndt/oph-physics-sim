from __future__ import annotations

from fractions import Fraction

import pytest

from oph_fpe.quotient.obstruction import (
    AbelianEdge,
    FiberStatus,
    KernelKind,
    audit_abelian_cycle_holonomy,
    classify_boundary_fiber,
    exhaust_accepted_repair_paths,
    verify_quotient_lumpability,
)


def test_complete_boundary_fiber_trichotomy_and_incomplete_unknown() -> None:
    states = (("b0", 0), ("b1", 0), ("b1", 1), ("b2", 0))

    def boundary(state: tuple[str, int]) -> str:
        return state[0]

    def consistent(state: tuple[str, int]) -> bool:
        return state != ("b2", 0)

    unique = classify_boundary_fiber(
        states,
        boundary_value="b0",
        boundary_map=boundary,
        is_consistent=consistent,
    )
    ambiguous = classify_boundary_fiber(
        states,
        boundary_value="b1",
        boundary_map=boundary,
        is_consistent=consistent,
    )
    unrealizable = classify_boundary_fiber(
        states,
        boundary_value="b2",
        boundary_map=boundary,
        is_consistent=consistent,
    )
    unknown = classify_boundary_fiber(
        states,
        boundary_value="b0",
        boundary_map=boundary,
        is_consistent=consistent,
        enumeration_complete=False,
    )

    assert unique.status is FiberStatus.UNIQUE
    assert unique.witnesses == (("b0", 0),)
    assert ambiguous.status is FiberStatus.AMBIGUOUS
    assert ambiguous.consistent_state_count == 2
    assert unrealizable.status is FiberStatus.UNREALIZABLE
    assert unknown.status is FiberStatus.UNKNOWN
    assert unknown.exact_finite_classification is False
    assert all(
        report.physical_promotion is False
        for report in (unique, ambiguous, unrealizable, unknown)
    )


def test_parity_frustrated_triangle_has_exact_nonzero_holonomy() -> None:
    edges = (
        AbelianEdge("ab", "a", "b", 1),
        AbelianEdge("bc", "b", "c", 1),
        AbelianEdge("ca", "c", "a", 1),
    )

    audit = audit_abelian_cycle_holonomy(("a", "b", "c"), edges, modulus=2)

    assert len(audit.cycles) == 1
    assert audit.cycles[0].holonomy == 1
    assert audit.cycles[0].obstructed is True
    assert audit.global_extension_exists is False
    assert audit.mathematical_obstruction_receipt is False
    assert audit.physical_promotion is False


def test_fractional_flat_connection_vanishes_on_fundamental_cycle() -> None:
    edges = (
        AbelianEdge("ab", "a", "b", Fraction(1, 2)),
        AbelianEdge("bc", "b", "c", Fraction(1, 3)),
        AbelianEdge("ac", "a", "c", Fraction(5, 6)),
    )

    audit = audit_abelian_cycle_holonomy(("a", "b", "c"), edges)

    assert audit.cycles[0].raw_holonomy == 0
    assert audit.global_extension_exists is True
    assert audit.mathematical_obstruction_receipt is True
    assert audit.cycle_basis_complete is True


def test_empty_holonomy_model_is_rejected_as_vacuous() -> None:
    with pytest.raises(ValueError, match="vertices must be nonempty"):
        audit_abelian_cycle_holonomy((), ())


def test_float_holonomy_input_is_rejected_instead_of_rounded() -> None:
    with pytest.raises(TypeError, match="int or Fraction"):
        audit_abelian_cycle_holonomy(
            ("a", "b"),
            (AbelianEdge("ab", "a", "b", 0.5),),  # type: ignore[arg-type]
        )


def test_exact_probability_kernel_can_lump_despite_representative_targets() -> None:
    states = (("a", "w1"), ("a", "w2"), ("b", "w1"), ("b", "w2"))
    kernel = {
        states[0]: {states[2]: 1},
        states[1]: {states[3]: 1},
        states[2]: {states[0]: Fraction(1, 2), states[1]: Fraction(1, 2)},
        states[3]: {states[1]: 1},
    }

    audit = verify_quotient_lumpability(
        states,
        kernel,
        quotient_map=lambda state: state[0],
        kernel_kind=KernelKind.PROBABILITY,
        kernel_dependency_fields={"semantic_interface"},
    )

    assert audit.aggregate_equality is True
    assert audit.metadata_firewall_passed is True
    assert audit.quotient_lumpable is True
    assert audit.nontrivial_presentation_fiber_count == 2
    assert audit.physical_promotion is False


def test_non_lumpable_presentation_kernel_exposes_representative_defect() -> None:
    states = (("a", "w1"), ("a", "w2"), ("b", "w1"), ("b", "w2"))
    kernel = {
        states[0]: {states[2]: 1},
        states[1]: {states[0]: 1},
        states[2]: {states[2]: 1},
        states[3]: {states[3]: 1},
    }

    audit = verify_quotient_lumpability(
        states,
        kernel,
        quotient_map=lambda state: state[0],
    )

    assert audit.aggregate_equality is False
    assert audit.quotient_lumpable is False
    assert len(audit.defects) == 1
    assert audit.defects[0].source_quotient == "a"


@pytest.mark.parametrize(
    "hidden_field",
    ["worker_id", "scheduler.queue_position", "retry_count", "executor_uuid"],
)
def test_worker_or_scheduler_dependency_blocks_lumpability_receipt(
    hidden_field: str,
) -> None:
    states = (("q", "p0"), ("q", "p1"))
    kernel = {states[0]: {states[0]: 1}, states[1]: {states[1]: 1}}

    audit = verify_quotient_lumpability(
        states,
        kernel,
        quotient_map=lambda _state: "q",
        kernel_dependency_fields={hidden_field},
    )

    assert audit.aggregate_equality is True
    assert audit.metadata_firewall_passed is False
    assert audit.forbidden_dependency_fields == (hidden_field,)
    assert audit.quotient_lumpable is False


def test_probability_rows_require_exact_normalization() -> None:
    with pytest.raises(ValueError, match="does not sum exactly to one"):
        verify_quotient_lumpability(
            ("a", "b"),
            {"a": {"b": Fraction(1, 3)}, "b": {"b": 1}},
            quotient_map=lambda state: state,
        )


def test_exact_rate_kernel_uses_the_same_representative_aggregation() -> None:
    states = (("q", "p0"), ("q", "p1"), ("r", "p0"), ("r", "p1"))
    kernel = {
        states[0]: {states[2]: Fraction(3, 2)},
        states[1]: {states[3]: Fraction(3, 2)},
        states[2]: {},
        states[3]: {},
    }

    audit = verify_quotient_lumpability(
        states,
        kernel,
        quotient_map=lambda state: state[0],
        kernel_kind=KernelKind.RATE,
    )

    assert audit.kernel_kind is KernelKind.RATE
    assert audit.quotient_lumpable is True


def test_repair_exhaustor_distinguishes_presentation_from_quotient_endpoint() -> None:
    states = ("start", "left", "right", "end0", "end1")
    repairs = {
        "start": ("left", "right"),
        "left": ("end0",),
        "right": ("end1",),
        "end0": (),
        "end1": (),
    }

    audit = exhaust_accepted_repair_paths(
        states,
        repairs,
        quotient_map=lambda state: "normal" if state.startswith("end") else state,
        initial_states=("start",),
    )

    assert audit.termination_certified is True
    assert audit.presentation_schedule_independent is False
    assert audit.quotient_schedule_independent is True
    assert audit.quotient_confluent is True
    assert audit.initial_audits[0].normal_form_status is FiberStatus.UNIQUE
    assert audit.physical_promotion is False


def test_schedule_dependent_endpoints_are_reported_as_ambiguous() -> None:
    states = ("start", "left_end", "right_end")
    repairs = {
        "start": ("left_end", "right_end"),
        "left_end": (),
        "right_end": (),
    }

    audit = exhaust_accepted_repair_paths(
        states,
        repairs,
        quotient_map=lambda state: state,
        initial_states=("start",),
    )

    assert audit.termination_certified is True
    assert audit.quotient_schedule_independent is False
    assert audit.quotient_confluent is False
    assert audit.ambiguous_initial_states == ("start",)
    assert audit.initial_audits[0].normal_form_status is FiberStatus.AMBIGUOUS


def test_reachable_repair_cycle_is_an_exact_nontermination_witness() -> None:
    audit = exhaust_accepted_repair_paths(
        ("a", "b", "done"),
        {"a": ("b",), "b": ("a", "done"), "done": ()},
        quotient_map=lambda state: state,
        initial_states=("a",),
    )

    assert audit.graph_terminating is False
    assert audit.termination_certified is False
    assert audit.nonterminating_initial_states == ("a",)
    assert audit.initial_audits[0].cycle_witness_paths == (("a", "b", "a"),)
    assert audit.initial_audits[0].normal_form_status is FiberStatus.UNKNOWN


def test_incomplete_repair_relation_cannot_certify_observed_unique_endpoint() -> None:
    audit = exhaust_accepted_repair_paths(
        ("start", "done"),
        {"start": ("done",), "done": ()},
        quotient_map=lambda state: state,
        initial_states=("start",),
        repair_relation_complete=False,
    )

    assert audit.graph_terminating is True
    assert audit.exact_exhaustion is False
    assert audit.termination_certified is False
    assert audit.quotient_confluent is False
    assert audit.initial_audits[0].normal_form_status is FiberStatus.UNKNOWN


def test_selected_initial_state_cannot_hide_an_unexamined_repair_cycle() -> None:
    audit = exhaust_accepted_repair_paths(
        ("good", "cycle-a", "cycle-b"),
        {
            "good": (),
            "cycle-a": ("cycle-b",),
            "cycle-b": ("cycle-a",),
        },
        quotient_map=lambda state: state,
        initial_states=("good",),
    )

    assert audit.termination_certified is True
    assert audit.initial_state_coverage_complete is False
    assert audit.mathematical_repair_receipt is False


def test_scheduler_dependency_blocks_otherwise_unique_repair_receipt() -> None:
    audit = exhaust_accepted_repair_paths(
        ("start", "done"),
        {"start": ("done",), "done": ()},
        quotient_map=lambda state: state,
        initial_states=("start",),
        transition_dependency_fields={"scheduler_id"},
    )

    assert audit.termination_certified is True
    assert audit.forbidden_dependency_fields == ("scheduler_id",)
    assert audit.quotient_schedule_independent is False
    assert audit.mathematical_repair_receipt is False
