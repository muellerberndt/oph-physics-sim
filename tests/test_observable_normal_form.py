from __future__ import annotations

from fractions import Fraction

import pytest

from oph_fpe.quotient import (
    recognize_conditional_resampling_kernel,
    verify_observation_determined_normal_form,
)


def _two_bit_repairs():
    states = ((False, False), (False, True), (True, False), (True, True))
    repairs = {
        (False, False): (),
        (False, True): ((False, False),),
        (True, False): (),
        (True, True): ((True, False),),
    }
    return states, repairs


def test_two_bit_system_replays_cross_source_lean_theorem_conditionally() -> None:
    states, repairs = _two_bit_repairs()

    audit = verify_observation_determined_normal_form(
        states,
        repairs,
        observation_map=lambda state: state[0],
        is_consistent=lambda state: state[1] is False,
        quotient_map=lambda state: state,
        state_space_complete=True,
        repair_relation_complete=True,
    )

    assert audit.observation_preserving is True
    assert audit.normal_forms_equal_consistent_set is True
    assert audit.boundary_identifies_modulo is True
    assert audit.cross_source_endpoint_unique_modulo is True
    assert audit.all_sources_weakly_normalizing is True
    assert audit.all_schedules_terminate is True
    assert audit.theorem_hypotheses_verified_on_declared_table is True
    assert audit.theorem_equivalence_replayed is True
    assert audit.conditional_mathematical_receipt is True
    assert audit.external_completeness_authenticated is False
    assert audit.physical_promotion is False
    assert audit.scale_authorized is False
    assert audit.demo_assumption_accepted_as_evidence is False


def test_same_source_confluence_does_not_hide_coarse_cross_source_failure() -> None:
    states, repairs = _two_bit_repairs()

    audit = verify_observation_determined_normal_form(
        states,
        repairs,
        observation_map=lambda _state: "coarse",
        is_consistent=lambda state: state[1] is False,
        quotient_map=lambda state: state,
        state_space_complete=True,
        repair_relation_complete=True,
    )

    assert audit.observation_preserving is True
    assert audit.normal_forms_equal_consistent_set is True
    assert audit.boundary_identifies_modulo is False
    assert audit.boundary_identification_defects
    assert audit.cross_source_endpoint_unique_modulo is False
    assert audit.cross_source_endpoint_defects
    assert audit.theorem_equivalence_replayed is True
    assert audit.conditional_mathematical_receipt is False


def test_same_source_multiple_terminal_quotients_are_in_cross_source_check() -> None:
    audit = verify_observation_determined_normal_form(
        ("start", "left", "right"),
        {"start": ("left", "right"), "left": (), "right": ()},
        observation_map=lambda _state: "same",
        is_consistent=lambda state: state != "start",
        quotient_map=lambda state: state,
        state_space_complete=True,
        repair_relation_complete=True,
    )

    assert audit.cross_source_endpoint_unique_modulo is False
    assert any(
        row.left_source == row.right_source == "start"
        for row in audit.cross_source_endpoint_defects
    )


def test_observation_leak_blocks_singleton_fiber_shortcut() -> None:
    states = ("start", "intended", "escaped")
    observations = {"start": 0, "intended": 0, "escaped": 1}
    audit = verify_observation_determined_normal_form(
        states,
        {"start": ("escaped",), "intended": (), "escaped": ()},
        observation_map=observations.__getitem__,
        is_consistent=lambda state: state != "start",
        quotient_map=lambda state: state,
        state_space_complete=True,
        repair_relation_complete=True,
    )

    assert audit.observation_preserving is False
    assert audit.observation_leaks[0].source == "start"
    assert audit.theorem_hypotheses_verified_on_declared_table is False
    assert audit.conditional_mathematical_receipt is False


def test_completeness_defaults_fail_closed_despite_positive_finite_table() -> None:
    states, repairs = _two_bit_repairs()
    audit = verify_observation_determined_normal_form(
        states,
        repairs,
        observation_map=lambda state: state[0],
        is_consistent=lambda state: state[1] is False,
        quotient_map=lambda state: state,
    )

    assert audit.boundary_identifies_modulo is True
    assert audit.cross_source_endpoint_unique_modulo is True
    assert audit.theorem_hypotheses_verified_on_declared_table is False
    assert audit.conditional_mathematical_receipt is False


@pytest.mark.parametrize("field", ["scheduler_id", "target_geometry"])
def test_presentation_or_target_dependency_blocks_theorem_application(
    field: str,
) -> None:
    states, repairs = _two_bit_repairs()
    audit = verify_observation_determined_normal_form(
        states,
        repairs,
        observation_map=lambda state: state[0],
        is_consistent=lambda state: state[1] is False,
        quotient_map=lambda state: state,
        state_space_complete=True,
        repair_relation_complete=True,
        dependency_fields=(field,),
    )

    assert audit.dependency_firewall_passed is False
    assert audit.forbidden_dependency_fields == (field,)
    assert audit.conditional_mathematical_receipt is False


def test_cycle_prevents_all_schedule_settlement_receipt() -> None:
    audit = verify_observation_determined_normal_form(
        ("a", "b", "done"),
        {"a": ("b",), "b": ("a", "done"), "done": ()},
        observation_map=lambda _state: "same",
        is_consistent=lambda state: state == "done",
        quotient_map=lambda _state: "one",
        state_space_complete=True,
        repair_relation_complete=True,
    )

    assert audit.all_sources_weakly_normalizing is True
    assert audit.all_schedules_terminate is False
    assert audit.theorem_equivalence_replayed is True
    assert audit.conditional_mathematical_receipt is False


def test_exact_r1_r2_r3_recognize_weighted_fiber_resampling() -> None:
    states = ("a0", "a1", "b")
    weights = {
        "a0": Fraction(1, 6),
        "a1": Fraction(1, 3),
        "b": Fraction(1, 2),
    }
    kernel = {
        "a0": {"a0": Fraction(1, 3), "a1": Fraction(2, 3)},
        "a1": {"a0": Fraction(1, 3), "a1": Fraction(2, 3)},
        "b": {"b": 1},
    }
    observations = {"a0": "a", "a1": "a", "b": "b"}

    audit = recognize_conditional_resampling_kernel(
        states,
        kernel,
        weights=weights,
        observation_map=observations.__getitem__,
    )

    assert audit.weights_normalized is True
    assert audit.kernel_nonnegative is True
    assert audit.kernel_row_stochastic is True
    assert audit.r1_fiber_supported is True
    assert audit.r2_fiber_rows_constant is True
    assert audit.r3_weighted_detailed_balance is True
    assert audit.explicit_formula_match is True
    assert audit.exact_table_recognition_receipt is True
    assert audit.external_kernel_provenance_authenticated is False
    assert audit.representative_selection_receipt is False
    assert audit.spectral_gap_receipt is False
    assert audit.convergence_rate_receipt is False
    assert audit.physical_promotion is False


def test_r1_rejects_transition_across_observation_fibers() -> None:
    audit = recognize_conditional_resampling_kernel(
        ("a", "b"),
        {"a": {"b": 1}, "b": {"a": 1}},
        weights={"a": Fraction(1, 2), "b": Fraction(1, 2)},
        observation_map=lambda state: state,
    )

    assert audit.r1_fiber_supported is False
    assert audit.r3_weighted_detailed_balance is True
    assert audit.exact_table_recognition_receipt is False


def test_r2_rejects_representative_dependent_rows_inside_one_fiber() -> None:
    audit = recognize_conditional_resampling_kernel(
        ("x", "y"),
        {"x": {"x": 1}, "y": {"y": 1}},
        weights={"x": Fraction(1, 2), "y": Fraction(1, 2)},
        observation_map=lambda _state: "same",
    )

    assert audit.r1_fiber_supported is True
    assert audit.r2_fiber_rows_constant is False
    assert audit.r3_weighted_detailed_balance is True
    assert audit.row_defects[0].differing_targets == ("x", "y")
    assert audit.exact_table_recognition_receipt is False


def test_r3_rejects_wrong_weight_ratio_even_when_rows_are_constant() -> None:
    audit = recognize_conditional_resampling_kernel(
        ("x", "y"),
        {
            "x": {"x": Fraction(1, 2), "y": Fraction(1, 2)},
            "y": {"x": Fraction(1, 2), "y": Fraction(1, 2)},
        },
        weights={"x": Fraction(1, 3), "y": Fraction(2, 3)},
        observation_map=lambda _state: "same",
    )

    assert audit.r1_fiber_supported is True
    assert audit.r2_fiber_rows_constant is True
    assert audit.r3_weighted_detailed_balance is False
    assert audit.exact_table_recognition_receipt is False


def test_global_weight_normalization_is_required_by_manuscript_contract() -> None:
    audit = recognize_conditional_resampling_kernel(
        ("x", "y"),
        {
            "x": {"x": Fraction(1, 3), "y": Fraction(2, 3)},
            "y": {"x": Fraction(1, 3), "y": Fraction(2, 3)},
        },
        weights={"x": 1, "y": 2},
        observation_map=lambda _state: "same",
    )

    assert audit.r1_fiber_supported is True
    assert audit.r2_fiber_rows_constant is True
    assert audit.r3_weighted_detailed_balance is True
    assert audit.explicit_formula_match is True
    assert audit.weights_normalized is False
    assert audit.exact_table_recognition_receipt is False


def test_float_inputs_are_rejected_instead_of_tolerance_checked() -> None:
    with pytest.raises(TypeError, match="never float"):
        recognize_conditional_resampling_kernel(
            ("x",),
            {"x": {"x": 1.0}},  # type: ignore[arg-type]
            weights={"x": 1},
            observation_map=lambda state: state,
        )

