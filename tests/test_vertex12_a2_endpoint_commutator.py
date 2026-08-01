from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from oph_fpe.dynamics import vertex12_a2_endpoint_commutator as producer
from oph_fpe.dynamics import (
    verify_vertex12_a2_endpoint_commutator_independent as independent,
)


def _sha(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _rehash(report: dict) -> dict:
    result = copy.deepcopy(report)
    result.pop("receipt_sha256", None)
    result["receipt_sha256"] = _sha(result)
    return result


@pytest.fixture(scope="module")
def receipt() -> dict:
    return producer.produce_receipt()


def test_exact_A2_endpoint_theorem_keeps_the_missing_premise_explicit(
    receipt: dict,
) -> None:
    assert producer.verify_receipt(receipt)["receipt"] is True
    assert receipt["status"] == producer.STATUS
    theorem = receipt["a2_endpoint_commutator_theorem"]
    assert (
        theorem["all_state_descent_statement"]["exact_conditional_implication"] is True
    )
    assert (
        theorem[
            "a2_forces_pairwise_endpoint_diamonds_without_a_source_or_repair_premise"
        ]
        is False
    )
    assert (
        theorem["a2_can_supply_quotient_descent_after_A1_accepts_and_types_each_step"]
        is True
    )
    assert (
        theorem["single_start_state_statement"]["global_quotient_commutation_follows"]
        is False
    )
    assert (
        theorem["underlying_path_equality_statement"][
            "hidden_kernel_or_holonomy_can_remain"
        ]
        is True
    )


def test_controls_separate_local_coincidence_descent_and_path_equality(
    receipt: dict,
) -> None:
    controls = receipt["exact_finite_controls"]
    local = controls["local_endpoint_coincidence_without_global_commutation"]
    assert local["equal_ordered_endpoint_start_states"] == [3]
    assert local["unequal_ordered_endpoint_start_states"] == [0, 1, 2]
    assert local["global_commutation_attained"] is False

    descent = controls["raw_commutation_without_quotient_descent"]
    assert descent["raw_ordered_maps_equal_on_every_state"] is True
    assert descent["first_step_quotient_congruence_violation_pairs"] == [[0, 1], [2, 3]]
    assert descent["quotient_port_action_defined"] is False

    holonomy = controls["visible_terminal_confluence_without_path_equality"]
    assert holonomy["raw_state_object"] == "H_3(F_3)"
    assert holonomy["raw_state_count"] == 27
    assert holonomy["observer_quotient_class_count"] == 9
    assert holonomy["members_per_observer_class"] == [3]
    assert holonomy["both_steps_descend_to_observer_quotient"] is True
    assert holonomy["visible_ordered_endpoint_equality_count"] == 27
    assert holonomy["raw_ordered_endpoint_equality_count"] == 0
    assert holonomy["central_commutator_shift_modulo_three"] == 1
    assert holonomy["identity_start_witness"] == {
        "first_then_second_raw_endpoint": [1, 1, 1],
        "second_then_first_raw_endpoint": [1, 1, 0],
        "shared_visible_endpoint": [1, 1],
    }


def test_current_source_has_exact_repair_confluence_without_oriented_steps(
    receipt: dict,
) -> None:
    control = receipt["exact_current_source_repair_control"]
    assert control["source_data_dimension"] == 96
    assert control["repair_event_count"] == 48
    assert control["every_carrier_port_coordinate_written_once"] is True
    assert control["unordered_port_block_pair_count"] == 66
    assert control["commuting_port_block_pair_count"] == 66
    assert control["each_eight_coordinate_matching_projector_rank"] == 4
    assert control["each_ninety_six_coordinate_port_block_map_rank"] == 92
    assert control["each_port_block_map_bijective"] is False
    assert control["full_twelve_port_repair_projector_rank"] == 48
    assert control["terminal_confluence_on_actual_source_field_attained"] is True
    assert control["underlying_event_word_equality_attained"] is False
    witness = control["distinct_event_words_same_terminal_operator_witness"]
    assert witness["event_words_equal"] is False
    assert witness["terminal_state_operators_equal"] is True
    shared = control["shared_eight_carrier_reinterpretation"]
    assert shared["unordered_pair_count"] == 66
    assert shared["commuting_pair_count"] == 23
    assert shared["noncommuting_pair_count"] == 43
    assert (
        shared["discarding_the_port_coordinate_preserves_source_operator_semantics"]
        is False
    )
    assert control["oriented_bijective_port_step_ledger_attained"] is False
    assert control["universal_z_power_6_translation_action_attained"] is False


def test_universal_factorization_is_exact_but_not_a_faithful_realization(
    receipt: dict,
) -> None:
    theorem = receipt["universal_abelian_port_factorization"]
    assert theorem["oriented_port_count"] == 12
    assert theorem["antipodal_axis_count"] == 6
    assert theorem["antipodal_relation_matrix_rank_over_Q"] == 6
    assert theorem["free_abelian_rank_after_antipodal_relations"] == 6
    assert theorem["positive_axis_commutator_diamond_count"] == 15
    assert theorem["all_oriented_nonantipodal_pair_count"] == 60
    assert (
        theorem[
            "fifteen_positive_diamonds_plus_inverse_law_imply_all_sixty_oriented_diamonds"
        ]
        is True
    )
    assert len(theorem["commutator_relation_irredundancy_witnesses"]) == 15
    assert len(theorem["inverse_relation_irredundancy_witnesses"]) == 6
    assert (
        theorem["every_positive_commutator_relation_is_individually_necessary"] is True
    )
    assert theorem["every_antipodal_inverse_relation_is_individually_necessary"] is True
    conditional = theorem["conditional_theorem"]
    assert conditional["accepted_meaning_action_factors_through_z_power_6"] is True
    assert (
        conditional[
            "physical_quotient_action_factors_through_z_power_6_if_Q_is_separately_identified_as_physical"
        ]
        is True
    )
    assert conditional["Q_identified_with_physical_support_by_this_theorem"] is False
    assert conditional["accepted_meaning_action_is_forced_to_be_faithful"] is False
    boundary = theorem["finite_ledger_faithfulness_boundary"]
    assert boundary["finite_Q_can_carry_a_faithful_z_power_6_action"] is False
    assert (
        boundary["finite_z3_power_6_control_is_faithful_universal_z_power_6_action"]
        is False
    )
    assert (
        boundary["finite_Z3_power_6_control_obeys_extra_relations"]
        == "3e_i=0 for every axis"
    )


def test_minimum_source_ledger_is_actionable_and_fail_closed(receipt: dict) -> None:
    ledger = receipt["minimum_source_emitted_port_step_ledger"]
    checks = ledger["factorization_checks"]
    assert checks["antipodal_inverse_family_count"] == 6
    assert checks["positive_axis_endpoint_diamond_family_count"] == 15
    assert checks["negative_orientation_diamonds_need_separate_rows"] is False
    assert len(ledger["source_capture_fields"]) == 6
    assert len(ledger["a1_a2_typing_fields"]) == 3
    assert len(ledger["faithful_universal_z_power_6_addendum"]) == 3
    assert len(ledger["issue_655_geometry_addendum"]) == 3
    assert len(ledger["physicalization_addendum"]) == 3
    for field in (
        "current_source_packet_supplies_complete_port_step_maps",
        "current_source_packet_supplies_accepted_observer_quotient",
        "current_source_packet_supplies_A2_naturality_rows",
        "current_source_packet_supplies_endpoint_diamonds",
        "current_source_packet_supplies_faithful_z_power_6_action",
        "algebraic_Z3_power_6_control_counts_as_source_ledger",
        "spatial_or_physical_promotion_allowed",
    ):
        assert ledger[field] is False
    assert receipt["comparison_data_read"] is False
    disposition = receipt["issue_655_disposition"]
    assert disposition["negative_closure_supported"] is False
    assert disposition["source_producer_narrowed"] is True
    assert len(disposition["minimum_next_source_packet"]) == 3


def test_independent_verifier_reconstructs_without_importing_producer(
    receipt: dict,
) -> None:
    verification = independent.verify_report(receipt)
    assert verification["receipt"] is True
    assert verification["finite_controls_independently_reconstructed"] is True
    assert verification["universal_presentation_independently_reconstructed"] is True
    assert verification["source_engine_independently_reimplemented"] is False
    tree = ast.parse(Path(independent.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert "oph_fpe.dynamics.vertex12_a2_endpoint_commutator" not in imported


@pytest.mark.parametrize(
    "mutation",
    (
        "extra_top_level_field",
        "status",
        "local_control",
        "descent_control",
        "holonomy_control",
        "current_source_control",
        "factor_rank",
        "diamond_count",
        "faithfulness_promotion",
        "source_ledger_promotion",
        "spatial_promotion",
        "upstream_pin",
        "implementation_pin",
        "claim_boundary",
    ),
)
def test_rehashed_semantic_mutations_fail_both_verifiers(
    receipt: dict, mutation: str
) -> None:
    changed = copy.deepcopy(receipt)
    if mutation == "extra_top_level_field":
        changed["undeclared"] = True
    elif mutation == "status":
        changed["status"] = "ATTAINED"
    elif mutation == "local_control":
        changed["exact_finite_controls"][
            "local_endpoint_coincidence_without_global_commutation"
        ]["global_commutation_attained"] = True
    elif mutation == "descent_control":
        changed["exact_finite_controls"]["raw_commutation_without_quotient_descent"][
            "quotient_port_action_defined"
        ] = True
    elif mutation == "holonomy_control":
        changed["exact_finite_controls"][
            "visible_terminal_confluence_without_path_equality"
        ]["central_commutator_shift_modulo_three"] = 0
    elif mutation == "current_source_control":
        changed["exact_current_source_repair_control"][
            "oriented_bijective_port_step_ledger_attained"
        ] = True
    elif mutation == "factor_rank":
        changed["universal_abelian_port_factorization"][
            "free_abelian_rank_after_antipodal_relations"
        ] = 5
    elif mutation == "diamond_count":
        changed["universal_abelian_port_factorization"][
            "positive_axis_commutator_diamond_count"
        ] = 14
    elif mutation == "faithfulness_promotion":
        changed["attainment"]["faithful_physical_z_power_6_action"] = True
    elif mutation == "source_ledger_promotion":
        changed["minimum_source_emitted_port_step_ledger"][
            "current_source_packet_supplies_endpoint_diamonds"
        ] = True
    elif mutation == "spatial_promotion":
        changed["minimum_source_emitted_port_step_ledger"][
            "spatial_or_physical_promotion_allowed"
        ] = True
    elif mutation == "upstream_pin":
        changed["upstream_feasibility_packet"]["pin"]["sha256"] = "sha256:" + "0" * 64
    elif mutation == "implementation_pin":
        changed["implementation_pins"]["producer"]["sha256"] = "sha256:" + "0" * 64
    elif mutation == "claim_boundary":
        changed["claim_boundary"] = "A2 forces spatial translations"
    else:  # pragma: no cover
        raise AssertionError(mutation)
    changed = _rehash(changed)
    assert producer.verify_receipt(changed)["receipt"] is False
    assert independent.verify_report(changed)["receipt"] is False


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("issue",), True),
        (("comparison_data_read",), 0),
        (
            (
                "universal_abelian_port_factorization",
                "positive_axis_commutator_diamond_count",
            ),
            15.0,
        ),
        (("attainment", "physical_prediction"), 0),
    ),
)
def test_json_type_confusion_mutations_fail_closed(
    receipt: dict, path: tuple[str, ...], replacement: object
) -> None:
    changed = copy.deepcopy(receipt)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    changed = _rehash(changed)
    assert producer.verify_receipt(changed)["receipt"] is False
    assert independent.verify_report(changed)["receipt"] is False


def test_duplicate_nested_json_keys_are_rejected(tmp_path: Path, receipt: dict) -> None:
    rendered = json.dumps(receipt, sort_keys=True)
    rendered = rendered.replace(
        '"raw_state_count": 27,',
        '"raw_state_count": 27, "raw_state_count": 27,',
        1,
    )
    path = tmp_path / "duplicate.json"
    path.write_text(rendered, encoding="utf-8")
    with pytest.raises((producer.EndpointCommutatorError, ValueError)):
        producer._load_json(path)
    with pytest.raises((independent.IndependentVerificationError, ValueError)):
        independent._load_json(path)


def test_committed_receipt_is_semantically_current(receipt: dict) -> None:
    committed = json.loads(producer.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert committed == receipt
