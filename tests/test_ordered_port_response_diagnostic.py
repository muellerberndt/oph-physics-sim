from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from oph_fpe.gauge import ordered_port_response_diagnostic as diagnostic
from oph_fpe.gauge import (
    verify_ordered_port_response_diagnostic_independent as independent,
)


def _rehash(receipt: dict, *, source: bool = False) -> dict:
    if source:
        receipt["source_projection_sha256"] = diagnostic.canonical_sha256(
            receipt["source_projection"]
        )
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = diagnostic.canonical_sha256(receipt)
    return receipt


def test_committed_receipt_replays_and_verifies_independently() -> None:
    committed = json.loads(diagnostic.DEFAULT_RECEIPT.read_text(encoding="utf-8"))
    generated = diagnostic.produce_receipt()

    assert committed == generated
    diagnostic.verify_receipt(committed)
    result = independent.verify(committed)
    assert result == {
        "schema": ("oph.ordered-port-response-diagnostic-independent-verification.v1"),
        "verdict": "VALID",
        "first_order_real_rank": 12,
        "derived_rank_before_propagation": 0,
        "generated_algebra_real_rank": 144,
        "derived_algebra_real_rank": 143,
        "a5_covariance_checks": 720,
        "inverse_port_checks": 12,
        "physical_current_source_bridge_receipt": False,
    }


def test_obvious_lift_is_an_exact_bounded_negative_control() -> None:
    receipt = diagnostic.produce_receipt()
    direct = receipt["direct_port_response"]
    propagated = receipt["propagation_adjoined_response"]

    assert direct["primitive_generator_count"] == 12
    assert direct["first_order_real_rank"] == 12
    assert direct["unordered_port_pair_count"] == 66
    assert direct["nonzero_direct_commutator_count"] == 0
    assert direct["derived_algebra_real_rank_before_propagation"] == 0
    assert direct["algebra_type"] == "u(1)^12"

    assert propagated["edge_mixed_response_nonzero_count"] == 30
    assert propagated["all_unordered_port_pairs_reached"] == 66
    assert propagated["generated_algebra_real_rank"] == 144
    assert propagated["generated_algebra_type"] == "u(12)"
    assert propagated["derived_algebra_real_rank"] == 143
    assert propagated["derived_algebra_type"] == "su(12)"
    assert propagated["center_dimension"] == 1
    assert len(propagated["edge_witnesses"]) == 30
    assert len(propagated["path_witnesses"]) == 66


def test_corrected_source_gate_is_separate_from_achieved_control() -> None:
    receipt = diagnostic.produce_receipt()
    gate = receipt["corrected_source_acceptance_gate"]
    interpretation = receipt["scientific_interpretation"]

    assert gate == {
        "expected_first_order_real_rank": 12,
        "expected_derived_algebra_real_rank": 11,
        "expected_center_dimension": 1,
        "center_condition": (
            "the constant linear combination of the twelve port generators "
            "spans the one-dimensional center"
        ),
        "obvious_diagonal_lift_satisfies_gate": False,
        "failure_before_propagation": "derived rank 0 rather than 11",
        "failure_after_propagation": "derived rank 143 rather than 11",
    }
    assert interpretation == {
        "u12_is_candidate_oph_current": False,
        "only_obvious_diagonal_port_lift_rejected": True,
        "issue_566_closed": False,
    }
    assert receipt["receipts"] == {
        "BOUNDED_ORDERED_PORT_RESPONSE_DIAGNOSTIC_RECEIPT": True,
        "A1_COMPLETE_TWELVE_DIMENSIONAL_RESPONSE_RECEIPT": False,
        "A2_SAME_CURRENT_HOLONOMY_RECEIPT": False,
        "PHYSICAL_CURRENT_SOURCE_BRIDGE_RECEIPT": False,
    }
    assert "does not close issue #566" in receipt["claim_boundary"]


def test_a5_and_inverse_port_actions_are_exact() -> None:
    receipt = diagnostic.produce_receipt()
    covariance = receipt["a5_covariance_audit"]
    inverse = receipt["inverse_port_response_audit"]

    assert covariance == {
        "proper_action_count": 60,
        "port_generator_conjugation_checks": 720,
        "propagation_invariance_checks": 60,
        "all_checks_exact": True,
    }
    assert inverse["operator"] == "R = -J"
    assert inverse["projector_permutation_checks"] == 12
    assert inverse["adds_continuous_tangent_direction"] is False
    assert inverse["reduces_generated_u12_algebra"] is False


def test_source_projection_is_target_free_and_pinned() -> None:
    receipt = diagnostic.produce_receipt()
    source = receipt["source_projection"]

    assert source["target_labels_used"] is False
    assert source["laboratory_data_used"] is False
    assert source["conditional_current_fixture_used"] is False
    assert receipt["target_firewall"]["forbidden_source_hits"] == []
    assert receipt["calculation_audit"] == {
        "matrix_domain": "Gaussian integer anti-Hermitian matrices",
        "rank_domain": "exact rational row reduction",
        "floating_point_rank_threshold_used": False,
        "matrix_exponential_evaluated": False,
    }
    assert set(source["source_files"]) == {
        "oph_fpe/core/echosahedral_dynamics.py",
        "oph_fpe/core/icosahedral.py",
    }
    assert receipt["source_projection_sha256"] == diagnostic.canonical_sha256(source)


def _mutate_first_order_rank(receipt: dict) -> None:
    receipt["direct_port_response"]["first_order_real_rank"] = 11


def _mutate_full_basis(receipt: dict) -> None:
    receipt["propagation_adjoined_response"]["full_basis_sha256"] = "sha256:0"


def _mutate_action(receipt: dict) -> None:
    receipt["source_projection"]["proper_actions_sha256"] = "sha256:0"


def _mutate_antipode(receipt: dict) -> None:
    receipt["source_projection"]["distance_three_map"][0] = 0


def _mutate_edge(receipt: dict) -> None:
    receipt["source_projection"]["carrier_edges"][0] = [0, 11]


@pytest.mark.parametrize(
    ("mutator", "source"),
    [
        (_mutate_first_order_rank, False),
        (_mutate_full_basis, False),
        (_mutate_action, True),
        (_mutate_antipode, True),
        (_mutate_edge, True),
    ],
)
def test_independent_verifier_rejects_structural_mutations(
    mutator, source: bool
) -> None:
    receipt = copy.deepcopy(diagnostic.produce_receipt())
    mutator(receipt)
    _rehash(receipt, source=source)

    with pytest.raises(independent.VerificationError):
        independent.verify(receipt)


def test_independent_verifier_rejects_target_and_promotion_mutations() -> None:
    target = copy.deepcopy(diagnostic.produce_receipt())
    target["source_projection"]["primitive_path"] += " standard_model"
    _rehash(target, source=True)
    with pytest.raises(independent.VerificationError, match="target firewall"):
        independent.verify(target)

    promoted = copy.deepcopy(diagnostic.produce_receipt())
    promoted["receipts"]["PHYSICAL_CURRENT_SOURCE_BRIDGE_RECEIPT"] = True
    _rehash(promoted)
    with pytest.raises(independent.VerificationError, match="forbidden promotion"):
        independent.verify(promoted)

    candidate = copy.deepcopy(diagnostic.produce_receipt())
    candidate["scientific_interpretation"]["u12_is_candidate_oph_current"] = True
    _rehash(candidate)
    with pytest.raises(independent.VerificationError, match="promoted"):
        independent.verify(candidate)


def test_receipt_hash_mutation_fails_before_recomputation(tmp_path: Path) -> None:
    receipt = copy.deepcopy(diagnostic.produce_receipt())
    receipt["status"] = "MUTATED"

    with pytest.raises(diagnostic.DiagnosticError, match="receipt hash"):
        diagnostic.verify_receipt(receipt)
    with pytest.raises(independent.VerificationError, match="receipt hash"):
        independent.verify(receipt)

    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert path.exists()
