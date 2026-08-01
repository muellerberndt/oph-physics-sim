from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from oph_fpe.dynamics import vertex12_directed_transport_feasibility as producer
from oph_fpe.dynamics import (
    verify_vertex12_directed_transport_feasibility_independent as independent,
)


def _sha(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _rehash(report: dict) -> dict:
    result = copy.deepcopy(report)
    result.pop("receipt_sha256", None)
    result["receipt_sha256"] = _sha(result)
    return result


@pytest.fixture(scope="module")
def receipt() -> dict:
    return producer.produce_receipt()


def test_current_source_matchings_have_exact_semiconjugacy_obstruction(
    receipt: dict,
) -> None:
    assert producer.verify_receipt(receipt)["receipt"] is True
    assert receipt["status"] == producer.STATUS
    theorem = receipt["exact_semiconjugacy_obstruction"]
    assert theorem["carrier_count"] == 8
    assert theorem["port_count"] == 12
    assert theorem["antipodal_pair_count"] == 6
    assert theorem["source_repair_event_count"] == 48
    assert theorem["source_repair_events_per_port"] == 4
    assert len(theorem["pair_rows"]) == 6
    assert theorem["all_source_matchings_are_fixed_point_free_involutions"] is True
    assert (
        theorem[
            "all_six_antipodal_source_matchings_differ_from_required_inverse"
        ]
        is True
    )
    assert theorem["semiconjugate_noncollapsed_site_cover_can_satisfy_inverse_law"] is False
    for row in theorem["pair_rows"]:
        witness = row["first_differing_carrier_index"]
        assert row["S_antipode_p_at_witness"] == row["S_antipode_p"][witness]
        assert row["inverse_S_p_at_witness"] == row["inverse_S_p"][witness]
        assert row["S_antipode_p_at_witness"] != row["inverse_S_p_at_witness"]
        assert len(row["source_repair_event_ids_for_p"]) == 4
        assert len(row["source_repair_event_ids_for_antipode_p"]) == 4


def test_free_abelian_control_proves_the_algebra_is_consistent(receipt: dict) -> None:
    control = receipt["algebraic_transport_positive_control"]
    assert control["schema"] == "oph.vertex12-free-abelian-translation-grammar-control.v1"
    assert control["universal_site_object"] == (
        "Z[P]/(e_antipode_p+e_p) isomorphic to Z^6"
    )
    assert control["site_domain"] == "(Z/3Z)^6"
    assert control["site_count"] == 729
    assert control["modulus"] == 3
    assert len(control["antipodal_axis_pairs"]) == 6
    assert len(control["port_rows"]) == 12
    assert control["proper_A5_port_action_order"] == 60
    assert control["site_A5_action_is_faithful"] is True
    assert control["site_A5_action_is_closed"] is True
    assert control["all_twelve_site_maps_are_bijections"] is True
    assert control["all_six_antipodal_pairs_are_exact_inverses"] is True
    assert control["exact_covariance_checks"] == 720
    assert control["exact_covariance_attained"] is True
    assert control["independent_axis_sign_choice_count"] == 64
    assert control["axis_sign_presentation_inverse_checks"] == 768
    assert control["axis_sign_presentation_covariance_checks"] == 46080
    assert (
        control["all_axis_sign_presentations_preserve_inverse_and_covariance"]
        is True
    )
    assert (
        control["axis_sign_presentation_changes_algebraic_equivalence_class"]
        is False
    )
    assert (
        control[
            "port_geometry_derived_after_declared_free_abelian_completion"
        ]
        is True
    )
    assert control["source_transition_event_emitted"] is False
    assert control["repair_generated"] is False
    assert control["source_selected_site_completion"] is False
    assert control["spatial_translation"] is False
    assert control["physical_readout"] is False
    assert control["physical_prediction"] is False


def test_missing_source_ledger_and_physical_boundary_remain_explicit(
    receipt: dict,
) -> None:
    requested = receipt["requested_directed_transport_ledger"]
    assert requested["schema"] == "oph.vertex12-directed-transport-ledger.v1"
    assert requested["attained_from_current_source_emissions"] is False
    assert requested["twelve_event_emitted_directed_maps_attained"] is False
    assert requested["exact_T_antipode_p_equals_inverse_T_p_attained"] is False
    assert requested["site_A5_action_and_exact_covariance_attained"] is False
    assert requested["inverse_and_A5_covariance_equations_algebraically_satisfiable"] is True
    assert requested["positive_control_is_source_transport_ledger"] is False
    contract = receipt["minimal_source_producer_contract"]
    assert (
        contract[
            "new_or_non_semiconjugate_source_emitted_oriented_law_required_for_issue_contract"
        ]
        is True
    )
    assert (
        contract[
            "inverse_and_A5_covariance_are_jointly_consistent_in_declared_control"
        ]
        is True
    )
    assert (
        contract[
            "source_selection_of_a_site_completion_and_event_provenance_remain_required"
        ]
        is True
    )
    scope = receipt["scope_boundary"]
    assert (
        scope[
            "rules_out_surjective_semiconjugate_covers_or_"
            "extensions_of_the_current_matchings"
        ]
        is True
    )
    assert scope["rules_out_a_new_source_emitted_oriented_transport_law"] is False
    assert scope["rules_out_non_semiconjugate_linear_or_state_space_transport"] is False
    assert scope["spatial_translation_attained"] is False
    assert scope["physical_sector_readout_attained"] is False
    assert scope["physical_prediction_unsealed"] is False
    assert receipt["comparison_data_read"] is False


def test_independent_verifier_reconstructs_without_importing_producer(
    receipt: dict,
) -> None:
    result = independent.verify_report(receipt)
    assert result["receipt"] is True
    assert result["packet_analysis_independently_reimplemented"] is True
    assert result["source_engine_independently_reimplemented"] is False
    tree = ast.parse(Path(independent.__file__).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert "oph_fpe.dynamics.vertex12_directed_transport_feasibility" not in imported


@pytest.mark.parametrize(
    "mutation",
    (
        "extra_top_level_field",
        "status",
        "obstruction_witness",
        "obstruction_promotion",
        "control_transport_digest",
        "control_source_promotion",
        "control_physical_promotion",
        "requested_ledger_promotion",
        "required_event_field",
        "upstream_pin",
        "implementation_pin",
        "claim_boundary",
    ),
)
def test_rehashed_serialized_mutations_fail_closed(
    receipt: dict, mutation: str
) -> None:
    changed = copy.deepcopy(receipt)
    if mutation == "extra_top_level_field":
        changed["undeclared"] = True
    elif mutation == "status":
        changed["status"] = "ATTAINED"
    elif mutation == "obstruction_witness":
        changed["exact_semiconjugacy_obstruction"]["pair_rows"][0][
            "first_differing_carrier_index"
        ] = 7
    elif mutation == "obstruction_promotion":
        changed["exact_semiconjugacy_obstruction"][
            "semiconjugate_noncollapsed_site_cover_can_satisfy_inverse_law"
        ] = True
    elif mutation == "control_transport_digest":
        changed["algebraic_transport_positive_control"][
            "transport_permutation_family_sha256"
        ] = "sha256:" + "0" * 64
    elif mutation == "control_source_promotion":
        changed["algebraic_transport_positive_control"][
            "source_transition_event_emitted"
        ] = True
    elif mutation == "control_physical_promotion":
        changed["algebraic_transport_positive_control"]["spatial_translation"] = True
    elif mutation == "requested_ledger_promotion":
        changed["requested_directed_transport_ledger"][
            "attained_from_current_source_emissions"
        ] = True
    elif mutation == "required_event_field":
        changed["minimal_source_producer_contract"]["required_fields"].pop()
    elif mutation == "upstream_pin":
        changed["upstream_source_packet"]["pin"]["sha256"] = "sha256:" + "0" * 64
    elif mutation == "implementation_pin":
        changed["implementation_pins"]["producer"]["sha256"] = "sha256:" + "0" * 64
    elif mutation == "claim_boundary":
        changed["claim_boundary"] = "spatial transport attained"
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
        (("algebraic_transport_positive_control", "site_count"), 729.0),
        (("scope_boundary", "physical_prediction_unsealed"), 0),
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
    rendered = rendered.replace('"site_count": 729,', '"site_count": 729, "site_count": 729,', 1)
    path = tmp_path / "duplicate.json"
    path.write_text(rendered, encoding="utf-8")
    with pytest.raises((producer.FeasibilityError, ValueError)):
        producer._load_json(path)
    with pytest.raises(ValueError):
        independent._load_json(path)


def test_committed_receipt_is_semantically_current(receipt: dict) -> None:
    committed = json.loads(producer.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert committed == receipt
