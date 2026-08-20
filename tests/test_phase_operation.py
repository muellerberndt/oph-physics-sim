"""Exact positive and mutation controls for the declared PR-04 phase
operation architecture module.

Guarded fail-closed mutation families: a wrong lift coefficient fails the
projector identity, a tampered outcome entry fails the cross-repository
replay, a state with a nonzero off-diagonal imaginary part shifts the phase
frequency off 1/2, and a window violation is detected by the committed
integer receipt clause.
"""

from __future__ import annotations

import copy
import json
from fractions import Fraction

import pytest

from oph_fpe.core.charged_response import canonical_sha256
from oph_fpe.quantum import phase_operation as po


REFERENCE_AVAILABLE = po.DEFAULT_REFERENCE_RECEIPT.is_file()

cross_repo = pytest.mark.skipif(
    not REFERENCE_AVAILABLE,
    reason="the committed reference receipt is not checked out",
)


@pytest.fixture(scope="module")
def receipt() -> dict:
    return po.build_receipt()


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads(po.DEFAULT_REFERENCE_RECEIPT.read_text(encoding="utf-8"))


def test_declared_operation_is_pauli_y_projector() -> None:
    operation = po.declared_phase_operation()
    assert operation == po.PAULI_Y_PLUS_PROJECTOR
    assert po.mdagger(operation) == operation
    assert po.mmul(operation, operation) == operation
    assert po.encode_matrix(operation) == [
        [["1/2", "0", "0", "0"], ["0", "0", "-1/2", "0"]],
        [["0", "0", "1/2", "0"], ["1/2", "0", "0", "0"]],
    ]


def test_irrep_generation_matches_committed_literals() -> None:
    irrep = po.standard_irrep()
    assert len(irrep) == 6
    assert irrep[0] == po.IDENTITY
    threecycle = po._real_matrix(po.COMMITTED_IRREP_ENTRIES[3])
    assert irrep[3] == threecycle


def test_context_web_matches_committed_projector_orbit() -> None:
    projectors = po.context_projectors()
    assert projectors[0] == po.RECORD_PROJECTOR
    assert projectors[1] == po.RECORD_PROJECTOR
    rotated = po.mat(
        (
            (Fraction(1, 4), po.C3(po.Q3(Fraction(0), Fraction(-1, 4)))),
            (po.C3(po.Q3(Fraction(0), Fraction(-1, 4))), Fraction(3, 4)),
        )
    )
    assert projectors[3] == rotated
    commutator = po.record_commutator(projectors)
    expected = po.mat(
        (
            (0, po.C3(po.Q3(Fraction(0), Fraction(1, 4)))),
            (po.C3(po.Q3(Fraction(0), Fraction(-1, 4))), 0),
        )
    )
    assert commutator == expected


def test_outcome_table_on_committed_state() -> None:
    state = po.record_diagonal_state(*po.RUN_COUNTS)
    rows = po.outcome_table(state)
    expected = [
        ("web_diagonal", "111/179", 1, 179, [111, 68]),
        ("web_conjugated_0", "111/179", 1, 179, [111, 68]),
        ("web_conjugated_1", "111/179", 1, 179, [111, 68]),
        ("web_conjugated_2", "315/716", 4, 716, [315, 401]),
        ("web_conjugated_3", "315/716", 4, 716, [315, 401]),
        ("web_conjugated_4", "315/716", 4, 716, [315, 401]),
        ("web_conjugated_5", "315/716", 4, 716, [315, 401]),
        ("phase", "1/2", 2, 358, [179, 179]),
    ]
    assert [
        (
            row["context"],
            row["born_weight"],
            row["run_mass_multiple"],
            row["mass"],
            row["counts"],
        )
        for row in rows
    ] == expected
    for row in rows:
        assert row["counts"][0] + row["counts"][1] == row["mass"]
        assert row["mass"] == row["run_mass_multiple"] * po.RUN_MASS


def test_receipt_is_deterministic_and_self_hashed(receipt: dict) -> None:
    assert po.build_receipt() == receipt
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    assert receipt["receipt_sha256"] == canonical_sha256(body)


def test_receipt_stamps_declared_provenance(receipt: dict) -> None:
    provenance = receipt["provenance"]
    assert provenance["operation_origin"] == "declared_architecture_operation"
    assert provenance["derived_from_source_dynamics"] is False
    assert "never derived" in provenance["statement"]
    assert "never measured" in provenance["statement"]
    decision = receipt["recorded_decision"]
    assert decision["register_row"] == "PR-04"
    assert decision["disposition"] == "axiomatize"
    assert decision["date"] == "2026-08-18"
    assert "declared architecture operation" in decision["statement"]
    assert "No sampling" in receipt["semantics"]
    boundaries = receipt["boundaries"]
    assert "no instrument is frozen" in boundaries["instrument_scope"]
    assert "no run is evidential" in boundaries["instrument_scope"]
    assert "OL-C5 status is untouched" in boundaries["ol_c5_scope"]
    assert receipt["phase_receipt_window"]["satisfied"] is True


@cross_repo
def test_cross_repo_replay_matches_committed_reference(
    receipt: dict, reference: dict
) -> None:
    report = po.replay_against_reference(receipt, reference)
    assert report["contexts_replayed"] == 8
    assert report["phase_counts"] == [179, 179]
    assert report["phase_window_lhs"] == 0
    assert report["phase_window_rhs"] == 3869527488
    for row, reference_row in zip(receipt["contexts"], reference["contexts"]):
        for key in (
            "context",
            "effect",
            "born_weight",
            "run_mass_multiple",
            "mass",
            "counts",
        ):
            assert row[key] == reference_row[key]
    assert (
        receipt["declared_operation"]["matrix"]
        == reference["declared_operation"]["matrix"]
    )
    assert (
        receipt["committed_state"]["matrix"]
        == reference["committed_state"]["matrix"]
    )


@cross_repo
def test_tampered_phase_count_fails_replay(receipt: dict, reference: dict) -> None:
    tampered = copy.deepcopy(reference)
    tampered["contexts"][-1]["counts"] = [180, 178]
    with pytest.raises(po.PhaseOperationError, match="REPLAY_CONTEXT"):
        po.replay_against_reference(receipt, tampered)


@cross_repo
def test_tampered_rotated_count_fails_replay(receipt: dict, reference: dict) -> None:
    tampered = copy.deepcopy(reference)
    tampered["contexts"][3]["counts"] = [316, 400]
    with pytest.raises(po.PhaseOperationError, match="REPLAY_CONTEXT"):
        po.replay_against_reference(receipt, tampered)


@cross_repo
def test_tampered_window_integers_fail_replay(
    receipt: dict, reference: dict
) -> None:
    tampered = copy.deepcopy(reference)
    tampered["phase_receipt_window"]["lhs"] = 1
    with pytest.raises(po.PhaseOperationError, match="REPLAY_WINDOW"):
        po.replay_against_reference(receipt, tampered)


@cross_repo
def test_tampered_operation_matrix_fails_replay(
    receipt: dict, reference: dict
) -> None:
    tampered = copy.deepcopy(reference)
    tampered["declared_operation"]["matrix"][0][1] = ["0", "0", "1/2", "0"]
    with pytest.raises(po.PhaseOperationError, match="REPLAY_OPERATION"):
        po.replay_against_reference(receipt, tampered)


def test_wrong_lift_coefficient_fails_projector_identity() -> None:
    commutator = po.record_commutator()
    wrong = po.phase_lift(commutator, coefficient=po.Q3.of(1))
    assert po.mmul(wrong, wrong) != wrong
    with pytest.raises(po.PhaseOperationError, match="PHASE_PROJECTOR"):
        po.declared_phase_operation(coefficient=po.Q3.of(1))


def test_sign_flipped_coefficient_fails_committed_literal() -> None:
    flipped = po.Q3(Fraction(0), Fraction(-2, 3))
    minus_y = po.phase_lift(po.record_commutator(), coefficient=flipped)
    assert po.mmul(minus_y, minus_y) == minus_y
    assert po.mdagger(minus_y) == minus_y
    with pytest.raises(po.PhaseOperationError, match="PHASE_PROJECTOR"):
        po.declared_phase_operation(coefficient=flipped)


def test_off_diagonal_imaginary_part_shifts_phase_frequency() -> None:
    operation = po.declared_phase_operation()
    half = Fraction(1, 2)
    assert po.born_weight(po.committed_core_state(0, 0), operation) == half
    for y in (Fraction(1, 8), Fraction(-1, 12), Fraction(12, 25)):
        weight = po.born_weight(po.committed_core_state(0, y), operation)
        assert weight == half - y
        assert weight != half
    rows = po.outcome_table(po.committed_core_state(0, Fraction(1, 8)))
    phase_row = rows[-1]
    assert phase_row["born_weight"] == "3/8"
    assert phase_row["counts"] == [537, 895]
    assert phase_row["mass"] == 1432


def test_real_off_diagonal_part_leaves_phase_but_fails_rotated_rationality() -> None:
    state = po.committed_core_state(Fraction(1, 10), 0)
    operation = po.declared_phase_operation()
    assert po.born_weight(state, operation) == Fraction(1, 2)
    rotated = po.context_projectors()[3]
    with pytest.raises(po.PhaseOperationError, match="WEIGHT_SQRT3"):
        po.born_weight(state, rotated)


def test_window_violation_detected() -> None:
    assert po.check_phase_window([179, 179]) == (0, 3869527488)
    with pytest.raises(po.PhaseOperationError, match="WINDOW_VIOLATION"):
        po.check_phase_window([179, 1])
    with pytest.raises(po.PhaseOperationError, match="WINDOW_POSITIVITY"):
        po.check_phase_window([0, 179])
    with pytest.raises(po.PhaseOperationError, match="WINDOW_POSITIVITY"):
        po.check_phase_window([358, 0])
    with pytest.raises(po.PhaseOperationError, match="WINDOW_SHAPE"):
        po.check_phase_window([179])


def test_committed_core_positivity_gate_matches_window_bound() -> None:
    with pytest.raises(po.PhaseOperationError, match="STATE_POSITIVITY"):
        po.committed_core_state(0, Fraction(1, 2))
    with pytest.raises(po.PhaseOperationError, match="STATE_POSITIVITY"):
        po.committed_core_state(Fraction(1, 2), 0)
    inside = po.committed_core_state(0, Fraction(12, 25))
    rows = po.outcome_table(inside)
    po.check_phase_window(rows[-1]["counts"])
    assert 4 * po.RUN_COUNTS[0] * po.RUN_COUNTS[1] == po.WINDOW_RHS_COEFFICIENT
    assert po.RUN_MASS**2 == po.WINDOW_LHS_COEFFICIENT


def test_record_diagonal_state_gates() -> None:
    with pytest.raises(po.PhaseOperationError, match="STATE_COUNTS"):
        po.record_diagonal_state(-1, 180)
    with pytest.raises(po.PhaseOperationError, match="STATE_COUNTS"):
        po.record_diagonal_state(0, 0)
    with pytest.raises(po.PhaseOperationError, match="OUTCOME_POSITIVITY"):
        po.outcome_table(po.record_diagonal_state(179, 0))
