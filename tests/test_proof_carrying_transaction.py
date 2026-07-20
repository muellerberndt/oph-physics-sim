from __future__ import annotations

# Tests for the proof-carrying repair contract (the legacy transaction tests
# remain in test_transactional_repair.py).

from decimal import Decimal
from fractions import Fraction

import pytest

from oph_fpe.repair.transaction import (
    MismatchLedger,
    ProposalClass,
    RepairCollar,
    Snapshot,
    TransactionalRepairEngine,
    TransitionKind,
    conflict_components,
    proposals_conflict,
    verify_repair_receipt_artifact,
)


def _absolute_mismatch(state, collar: RepairCollar) -> MismatchLedger:
    total = sum(
        abs(state[ref])
        for ref in collar.visible_read_set
        if isinstance(state[ref], int) and not isinstance(state[ref], bool)
    )
    return MismatchLedger(overlap=total)


def _constant_mismatch(_state, _collar: RepairCollar) -> MismatchLedger:
    return MismatchLedger()


def _collar(
    collar_id: str,
    *,
    reads=("x",),
    writes=("x",),
    protected=(),
    sectors=(),
    records=(),
    checkpoints=(),
) -> RepairCollar:
    return RepairCollar(
        collar_id=collar_id,
        visible_read_set=frozenset(reads),
        writable_registers=frozenset(writes),
        protected_boundary=frozenset(protected),
        sector_registers=frozenset(sectors),
        record_registers=frozenset(records),
        checkpoint_registers=frozenset(checkpoints),
    )


def _strict(
    engine: TransactionalRepairEngine,
    *,
    proposal_id: str,
    collar: RepairCollar,
    reads,
    recovery,
    proposal_class=ProposalClass.EXACT_SPLICE,
    source_parameters=None,
):
    return engine.prepare(
        proposal_id=proposal_id,
        transition_kind=TransitionKind.STRICT_REPAIR,
        proposal_class=proposal_class,
        collar=collar,
        declared_read_set=reads,
        recovery=recovery,
        source_parameters=source_parameters,
    )


def _check(receipt, name: str) -> bool:
    return next(item.passed for item in receipt.checks if item.name == name)


def test_mismatch_ledger_is_exact_and_lexicographic() -> None:
    before = MismatchLedger(
        overlap=2,
        local_constraint=Decimal("0.5"),
        physical_auxiliary={"tie": Fraction(3, 7)},
    )
    after = MismatchLedger(
        overlap=1,
        local_constraint=Fraction(1, 2),
        physical_auxiliary={"tie": Fraction(3, 7)},
    )
    assert after.strictly_descends_from(before)
    assert after.overlap == Fraction(1)
    with pytest.raises(TypeError, match="exact number"):
        MismatchLedger(overlap=0.1)
    with pytest.raises(ValueError, match="non-negative"):
        MismatchLedger(overlap=-1)


def test_snapshot_is_deeply_immutable_and_versioned() -> None:
    state = {"x": {"nested": [1, 2]}}
    snapshot = Snapshot.capture(state, {"x": 4}, {"x"})
    state["x"]["nested"].append(3)
    extracted = snapshot.value("x")
    extracted["nested"].append(99)
    assert snapshot.value("x") == {"nested": [1, 2]}
    assert snapshot.version("x") == 4
    assert len(snapshot.snapshot_hash) == 64


def test_strict_repair_commits_with_proof_carrying_receipt() -> None:
    engine = TransactionalRepairEngine(
        {"x": 3, "boundary": 7, "sector": "A", "other": 11},
        mismatch_evaluator=_absolute_mismatch,
    )
    collar = _collar(
        "c",
        protected=("boundary",),
        sectors=("sector",),
    )
    proposal = _strict(
        engine,
        proposal_id="p",
        collar=collar,
        reads={"x", "boundary", "sector"},
        recovery=lambda state: {"x": state["x"] - 1},
    )
    receipt = engine.commit(proposal)
    assert receipt.committed
    assert receipt.verdict == "VALID_PASS"
    assert receipt.physical_repair_receipt
    assert receipt.transition_event_emitted
    assert not receipt.semantic_record_written
    assert receipt.mismatch_before == MismatchLedger(overlap=3)
    assert receipt.mismatch_after == MismatchLedger(overlap=2)
    assert engine.state == {"x": 2, "boundary": 7, "sector": "A", "other": 11}
    assert engine.versions == {"x": 1, "boundary": 0, "sector": 0, "other": 0}
    assert _check(receipt, "complete_read_set")
    assert _check(receipt, "unlisted_registers_unchanged")
    assert _check(receipt, "atomic_union_revalidated")


def test_stale_snapshot_is_rejected_without_mutation() -> None:
    engine = TransactionalRepairEngine({"x": 2}, mismatch_evaluator=_absolute_mismatch)
    collar = _collar("c")
    stale = _strict(
        engine,
        proposal_id="stale",
        collar=collar,
        reads={"x"},
        recovery=lambda state: {"x": state["x"] - 2},
    )
    first = _strict(
        engine,
        proposal_id="first",
        collar=collar,
        reads={"x"},
        recovery=lambda state: {"x": state["x"] - 1},
    )
    assert engine.commit(first).committed
    receipt = engine.commit(stale)
    assert not receipt.committed
    assert not _check(receipt, "snapshot_current")
    assert engine.state["x"] == 1


def test_observed_but_undeclared_dependency_is_rejected() -> None:
    engine = TransactionalRepairEngine(
        {"x": 2, "hidden": 1}, mismatch_evaluator=_absolute_mismatch
    )
    proposal = _strict(
        engine,
        proposal_id="incomplete",
        collar=_collar("c"),
        reads={"x"},
        recovery=lambda state: {"x": state["x"] - state["hidden"]},
    )
    receipt = engine.commit(proposal)
    assert not receipt.committed
    assert not _check(receipt, "complete_read_set")
    assert "hidden" in receipt.failure_reasons[0]
    assert engine.state == {"x": 2, "hidden": 1}


def test_score_protected_and_write_registers_must_all_be_snapshotted() -> None:
    engine = TransactionalRepairEngine(
        {"x": 2, "boundary": 3}, mismatch_evaluator=_absolute_mismatch
    )
    collar = _collar("c", protected=("boundary",))
    proposal = _strict(
        engine,
        proposal_id="missing-protected",
        collar=collar,
        reads={"x"},
        recovery=lambda state: {"x": state["x"] - 1},
    )
    receipt = engine.commit(proposal)
    assert not receipt.committed
    assert not _check(receipt, "complete_read_set")


def test_write_outside_declared_collar_is_rejected() -> None:
    engine = TransactionalRepairEngine(
        {"x": 2, "y": 1}, mismatch_evaluator=_absolute_mismatch
    )
    proposal = _strict(
        engine,
        proposal_id="nonlocal",
        collar=_collar("c", reads=("x",), writes=("x",)),
        reads={"x", "y"},
        recovery=lambda state: {"y": state["y"] - 1},
    )
    receipt = engine.commit(proposal)
    assert not receipt.committed
    assert not _check(receipt, "write_locality")
    assert engine.state == {"x": 2, "y": 1}


@pytest.mark.parametrize("invariant", ["protected", "sector"])
def test_protected_and_sector_mutations_are_rejected(invariant: str) -> None:
    kwargs = {"protected": ("x",)} if invariant == "protected" else {"sectors": ("x",)}
    engine = TransactionalRepairEngine({"x": 2}, mismatch_evaluator=_absolute_mismatch)
    proposal = _strict(
        engine,
        proposal_id=invariant,
        collar=_collar("c", **kwargs),
        reads={"x"},
        recovery=lambda state: {"x": state["x"] - 1},
    )
    receipt = engine.commit(proposal)
    assert not receipt.committed
    check_name = (
        "protected_boundary_preserved"
        if invariant == "protected"
        else "sector_preserved"
    )
    assert not _check(receipt, check_name)
    assert engine.state["x"] == 2


@pytest.mark.parametrize(
    "source_parameters",
    [
        {"target_beta": "2pi"},
        {"nested": {"candidate_model": "H3"}},
        {"preferred_geometry": "H3"},
    ],
)
def test_downstream_target_fields_are_rejected(source_parameters) -> None:
    engine = TransactionalRepairEngine({"x": 2}, mismatch_evaluator=_absolute_mismatch)
    proposal = _strict(
        engine,
        proposal_id="leak",
        collar=_collar("c"),
        reads={"x"},
        recovery=lambda state: {"x": state["x"] - 1},
        source_parameters=source_parameters,
    )
    receipt = engine.commit(proposal)
    assert not receipt.committed
    assert not _check(receipt, "target_free_source")


def test_diagnostic_heuristic_cannot_get_physical_repair_receipt() -> None:
    engine = TransactionalRepairEngine({"x": 2}, mismatch_evaluator=_absolute_mismatch)
    proposal = _strict(
        engine,
        proposal_id="heuristic",
        collar=_collar("c"),
        reads={"x"},
        recovery=lambda state: {"x": state["x"] - 1},
        proposal_class=ProposalClass.DIAGNOSTIC_HEURISTIC,
    )
    receipt = engine.commit(proposal)
    assert receipt.committed
    assert receipt.diagnostic_only
    assert not receipt.physical_repair_receipt
    assert not receipt.semantic_record_written
    assert receipt.claim_tier == "REGULATOR_DIAGNOSTIC"


def test_reversible_propagation_requires_exact_inverse_and_writes_no_record() -> None:
    engine = TransactionalRepairEngine(
        {"x": 1, "log": ()}, mismatch_evaluator=_constant_mismatch
    )
    collar = _collar("u", reads=("x",), writes=("x",), records=("log",))
    proposal = engine.prepare(
        proposal_id="u",
        transition_kind=TransitionKind.REVERSIBLE_PROPAGATION,
        proposal_class=ProposalClass.PHYSICAL_CARRIER_RESPONSE,
        collar=collar,
        declared_read_set={"x", "log"},
        recovery=lambda state: {"x": state["x"] + 1},
        inverse_updates={"x": 1},
    )
    receipt = engine.commit(proposal)
    assert receipt.committed
    assert engine.state["x"] == 2
    assert not receipt.semantic_record_written
    assert not receipt.physical_repair_receipt

    bad_engine = TransactionalRepairEngine(
        {"x": 1}, mismatch_evaluator=_constant_mismatch
    )
    bad = bad_engine.prepare(
        proposal_id="bad-u",
        transition_kind=TransitionKind.REVERSIBLE_PROPAGATION,
        proposal_class=ProposalClass.PHYSICAL_CARRIER_RESPONSE,
        collar=_collar("u"),
        declared_read_set={"x"},
        recovery=lambda state: {"x": state["x"] + 1},
        inverse_updates={"x": 99},
    )
    assert not bad_engine.commit(bad).committed
    assert bad_engine.state["x"] == 1


def test_reversible_propagation_rejects_record_mutation_even_with_inverse() -> None:
    engine = TransactionalRepairEngine(
        {"log": ()}, mismatch_evaluator=_constant_mismatch
    )
    collar = _collar("u", reads=("log",), writes=("log",), records=("log",))
    proposal = engine.prepare(
        proposal_id="bad-record-u",
        transition_kind=TransitionKind.REVERSIBLE_PROPAGATION,
        proposal_class=ProposalClass.EXACT_SPLICE,
        collar=collar,
        declared_read_set={"log"},
        recovery=lambda state: {"log": (*state["log"], "event")},
        inverse_updates={"log": ()},
    )
    receipt = engine.commit(proposal)
    assert not receipt.committed
    assert engine.state["log"] == ()


def test_record_commit_is_append_only_and_not_a_strict_repair() -> None:
    engine = TransactionalRepairEngine(
        {"log": ("birth",), "checkpoint": 0}, mismatch_evaluator=_constant_mismatch
    )
    collar = _collar(
        "record",
        reads=("log",),
        writes=("log", "checkpoint"),
        protected=("log",),
        records=("log",),
        checkpoints=("checkpoint",),
    )
    proposal = engine.prepare(
        proposal_id="record",
        transition_kind=TransitionKind.RECORD_COMMIT,
        proposal_class=ProposalClass.EXACT_SPLICE,
        collar=collar,
        declared_read_set={"log", "checkpoint"},
        recovery=lambda state: {
            "log": (*state["log"], "repair-1"),
            "checkpoint": state["checkpoint"] + 1,
        },
    )
    receipt = engine.commit(proposal)
    assert receipt.committed
    assert receipt.semantic_record_written
    assert not receipt.physical_repair_receipt
    assert engine.state == {"log": ("birth", "repair-1"), "checkpoint": 1}

    replacement_engine = TransactionalRepairEngine(
        {"log": ("birth",)}, mismatch_evaluator=_constant_mismatch
    )
    replacement = replacement_engine.prepare(
        proposal_id="replace",
        transition_kind=TransitionKind.RECORD_COMMIT,
        proposal_class=ProposalClass.EXACT_SPLICE,
        collar=_collar(
            "record",
            reads=("log",),
            writes=("log",),
            protected=("log",),
            records=("log",),
        ),
        declared_read_set={"log"},
        recovery=lambda _state: {"log": ("replacement",)},
    )
    assert not replacement_engine.commit(replacement).committed


def test_rollback_restores_exact_prestate_but_is_not_a_repair_receipt() -> None:
    engine = TransactionalRepairEngine(
        {"x": 2, "other": 9}, mismatch_evaluator=_absolute_mismatch
    )
    repair = _strict(
        engine,
        proposal_id="repair",
        collar=_collar("c"),
        reads={"x"},
        recovery=lambda state: {"x": state["x"] - 1},
    )
    committed = engine.commit(repair)
    assert committed.committed
    rollback = engine.rollback(committed.commit_id)
    assert rollback.committed
    assert rollback.transition_kind is TransitionKind.ROLLBACK
    assert not rollback.physical_repair_receipt
    assert not rollback.semantic_record_written
    assert engine.state == {"x": 2, "other": 9}
    assert engine.versions["x"] == 2


def test_rollback_rejects_when_later_state_is_current() -> None:
    engine = TransactionalRepairEngine({"x": 3}, mismatch_evaluator=_absolute_mismatch)
    first = engine.commit(
        _strict(
            engine,
            proposal_id="first",
            collar=_collar("c"),
            reads={"x"},
            recovery=lambda state: {"x": state["x"] - 1},
        )
    )
    second = engine.commit(
        _strict(
            engine,
            proposal_id="second",
            collar=_collar("c"),
            reads={"x"},
            recovery=lambda state: {"x": state["x"] - 1},
        )
    )
    assert second.committed
    rollback = engine.rollback(first.commit_id)
    assert not rollback.committed
    assert engine.state["x"] == 1


def _nonadditive_mismatch(state, _collar: RepairCollar) -> MismatchLedger:
    if state["x"] == 0 and state["y"] == 0:
        return MismatchLedger(overlap=10)
    return MismatchLedger(overlap=state["x"] + state["y"])


def test_individually_descending_conflicts_are_revalidated_and_rejected_as_union() -> (
    None
):
    engine = TransactionalRepairEngine(
        {"x": 1, "y": 1}, mismatch_evaluator=_nonadditive_mismatch
    )
    collar_x = _collar("cx", reads=("x", "y"), writes=("x",))
    collar_y = _collar("cy", reads=("x", "y"), writes=("y",))
    left = _strict(
        engine,
        proposal_id="left",
        collar=collar_x,
        reads={"x", "y"},
        recovery=lambda state: {"x": state["x"] - 1},
    )
    right = _strict(
        engine,
        proposal_id="right",
        collar=collar_y,
        reads={"x", "y"},
        recovery=lambda state: {"y": state["y"] - 1},
    )
    assert engine.assess(left).admissible
    assert engine.assess(right).admissible
    assert proposals_conflict(left, right)
    receipts = engine.commit_batch([right, left])
    assert len(receipts) == 1
    assert not receipts[0].committed
    assert not _check(receipts[0], "transition_contract")
    assert not _check(receipts[0], "atomic_union_revalidated")
    assert engine.state == {"x": 1, "y": 1}


def test_cross_proposal_protected_mutation_fails_atomic_union() -> None:
    engine = TransactionalRepairEngine(
        {"x": 2, "b": 2}, mismatch_evaluator=_absolute_mismatch
    )
    writer = _strict(
        engine,
        proposal_id="writer",
        collar=_collar("writer", reads=("b",), writes=("b",)),
        reads={"b"},
        recovery=lambda state: {"b": state["b"] - 1},
    )
    observer = _strict(
        engine,
        proposal_id="observer",
        collar=_collar("observer", reads=("x", "b"), writes=("x",), protected=("b",)),
        reads={"x", "b"},
        recovery=lambda state: {"x": state["x"] - 1},
    )
    assert engine.assess(writer).admissible
    receipt = engine.commit_batch([writer, observer])[0]
    assert not receipt.committed
    assert not _check(receipt, "protected_boundary_preserved")
    assert engine.state == {"x": 2, "b": 2}


def test_conflict_graph_uses_read_write_intersections() -> None:
    engine = TransactionalRepairEngine(
        {"x": 2, "y": 2, "z": 2}, mismatch_evaluator=_absolute_mismatch
    )
    left = _strict(
        engine,
        proposal_id="left",
        collar=_collar("left", reads=("x",), writes=("x",)),
        reads={"x"},
        recovery=lambda state: {"x": state["x"] - 1},
    )
    bridge = _strict(
        engine,
        proposal_id="bridge",
        collar=_collar("bridge", reads=("x", "y"), writes=("y",)),
        reads={"x", "y"},
        recovery=lambda state: {"y": state["y"] - 1},
    )
    separate = _strict(
        engine,
        proposal_id="separate",
        collar=_collar("separate", reads=("z",), writes=("z",)),
        reads={"z"},
        recovery=lambda state: {"z": state["z"] - 1},
    )
    components = conflict_components([separate, bridge, left])
    assert [[proposal.proposal_id for proposal in item] for item in components] == [
        ["bridge", "left"],
        ["separate"],
    ]


def test_disjoint_schedule_order_does_not_change_endpoint_or_receipts() -> None:
    def run(order):
        engine = TransactionalRepairEngine(
            {"x": 2, "y": 2, "untouched": 5}, mismatch_evaluator=_absolute_mismatch
        )
        proposals = {
            "x": _strict(
                engine,
                proposal_id="a-x",
                collar=_collar("x", reads=("x",), writes=("x",)),
                reads={"x"},
                recovery=lambda state: {"x": state["x"] - 1},
            ),
            "y": _strict(
                engine,
                proposal_id="b-y",
                collar=_collar("y", reads=("y",), writes=("y",)),
                reads={"y"},
                recovery=lambda state: {"y": state["y"] - 1},
            ),
        }
        receipts = engine.commit_batch([proposals[key] for key in order])
        return engine.state, [receipt.commit_id for receipt in receipts]

    forward = run(["x", "y"])
    reverse = run(["y", "x"])
    assert forward == reverse
    assert forward[0] == {"x": 1, "y": 1, "untouched": 5}


def test_controlled_exploration_is_fail_closed_without_budget_verifier() -> None:
    engine = TransactionalRepairEngine({"x": 1}, mismatch_evaluator=_absolute_mismatch)
    proposal = engine.prepare(
        proposal_id="explore",
        transition_kind=TransitionKind.CONTROLLED_EXPLORATION,
        proposal_class=ProposalClass.DIAGNOSTIC_HEURISTIC,
        collar=_collar("c"),
        declared_read_set={"x"},
        recovery=lambda state: {"x": state["x"] + 1},
    )
    receipt = engine.commit(proposal)
    assert not receipt.committed
    assert "typed budget" in " ".join(receipt.failure_reasons)


def test_strict_repair_cannot_mutate_checkpoint_or_record_state() -> None:
    engine = TransactionalRepairEngine(
        {"x": 2, "checkpoint": 0}, mismatch_evaluator=_absolute_mismatch
    )
    collar = _collar(
        "strict",
        reads=("x",),
        writes=("x", "checkpoint"),
        checkpoints=("checkpoint",),
    )
    proposal = _strict(
        engine,
        proposal_id="checkpoint-leak",
        collar=collar,
        reads={"x", "checkpoint"},
        recovery=lambda state: {
            "x": state["x"] - 1,
            "checkpoint": state["checkpoint"] + 1,
        },
    )
    receipt = engine.commit(proposal)
    assert not receipt.committed
    assert not _check(receipt, "transition_contract")
    assert engine.state == {"x": 2, "checkpoint": 0}


def test_strict_repair_cannot_trade_record_or_holonomy_violation_for_overlap() -> None:
    def invalid_priority_score(state, _collar):
        return MismatchLedger(
            record=state["x"],
            holonomy=state["holonomy"],
            overlap=10 - state["x"],
        )

    engine = TransactionalRepairEngine(
        {"x": 2, "holonomy": 0}, mismatch_evaluator=invalid_priority_score
    )
    proposal = _strict(
        engine,
        proposal_id="violation-trade",
        collar=_collar("c", reads=("x", "holonomy"), writes=("x",)),
        reads={"x", "holonomy"},
        recovery=lambda state: {"x": state["x"] - 1},
    )
    receipt = engine.commit(proposal)
    assert not receipt.committed
    assert not _check(receipt, "transition_contract")
    assert engine.state["x"] == 2


def test_hidden_mismatch_reads_cannot_evade_conflict_or_read_completeness() -> None:
    def hidden_nonlinear_score(state, _collar):
        # Both registers affect the acceptance score even when a collar falsely
        # declares only one. The tracked evaluator must discover that lie.
        penalty = 10 if state["x"] == 0 and state["y"] == 0 else 0
        return MismatchLedger(overlap=state["x"] + state["y"] + penalty)

    engine = TransactionalRepairEngine(
        {"x": 1, "y": 1}, mismatch_evaluator=hidden_nonlinear_score
    )
    left = _strict(
        engine,
        proposal_id="left-hidden",
        collar=_collar("left", reads=("x",), writes=("x",)),
        reads={"x"},
        recovery=lambda state: {"x": state["x"] - 1},
    )
    right = _strict(
        engine,
        proposal_id="right-hidden",
        collar=_collar("right", reads=("y",), writes=("y",)),
        reads={"y"},
        recovery=lambda state: {"y": state["y"] - 1},
    )
    assert not proposals_conflict(left, right)
    receipts = engine.commit_batch([left, right])
    assert len(receipts) == 2
    assert all(not receipt.committed for receipt in receipts)
    assert all(
        not _check(receipt, "mismatch_read_trace_complete") for receipt in receipts
    )
    assert all(not receipt.COMPLETE_READ_SET_RECEIPT for receipt in receipts)
    assert engine.state == {"x": 1, "y": 1}


def test_register_and_nested_mapping_keys_are_never_lossily_coerced() -> None:
    with pytest.raises(TypeError, match="lossy coercion"):
        TransactionalRepairEngine(
            {1: 2, "1": 3},  # type: ignore[dict-item]
            mismatch_evaluator=_constant_mismatch,
        )
    with pytest.raises(TypeError, match="lossy key coercion"):
        TransactionalRepairEngine(
            {"x": {1: "numeric", "1": "string"}},
            mismatch_evaluator=_constant_mismatch,
        )
    with pytest.raises(TypeError, match="register references"):
        _collar("bad", reads=(1,), writes=("x",))  # type: ignore[arg-type]


def test_serialized_artifact_verifier_recomputes_receipt_booleans_and_hashes() -> None:
    engine = TransactionalRepairEngine({"x": 2}, mismatch_evaluator=_absolute_mismatch)
    receipt = engine.commit(
        _strict(
            engine,
            proposal_id="verified",
            collar=_collar("c"),
            reads={"x"},
            recovery=lambda state: {"x": state["x"] - 1},
        )
    )
    report = verify_repair_receipt_artifact(receipt.as_dict())
    assert report["REPAIR_ARTIFACT_INTEGRITY_RECEIPT"]
    assert report["COMPLETE_READ_SET_RECEIPT"]
    assert report["CONFLICT_COMPONENT_SUPPORT_RECEIPT"]
    assert report["ATOMIC_UNION_REVALIDATION_RECEIPT"]
    assert report["TRANSACTIONAL_REPAIR_RECEIPT"]
    assert report["commit_id"] == receipt.commit_id

    forged = receipt.as_dict()
    forged["score_observed_reads"] = ["hidden_target"]
    forged["COMPLETE_READ_SET_RECEIPT"] = True
    forged["TRANSACTIONAL_REPAIR_RECEIPT"] = True
    forged_report = verify_repair_receipt_artifact(forged)
    assert not forged_report["REPAIR_ARTIFACT_INTEGRITY_RECEIPT"]
    assert not forged_report["TRANSACTIONAL_REPAIR_RECEIPT"]


def test_rejected_receipt_cannot_be_promoted_by_flipping_serialized_flags() -> None:
    engine = TransactionalRepairEngine({"x": 1}, mismatch_evaluator=_absolute_mismatch)
    rejected = engine.commit(
        _strict(
            engine,
            proposal_id="no-descent",
            collar=_collar("c"),
            reads={"x"},
            recovery=lambda state: {"x": state["x"] + 1},
        )
    )
    artifact = rejected.as_dict()
    artifact["physical_repair_receipt"] = True
    artifact["TRANSACTIONAL_REPAIR_RECEIPT"] = True
    artifact["ATOMIC_UNION_REVALIDATION_RECEIPT"] = True
    report = verify_repair_receipt_artifact(artifact)
    assert not report["REPAIR_ARTIFACT_INTEGRITY_RECEIPT"]
    assert not report["TRANSACTIONAL_REPAIR_RECEIPT"]


def test_incompatible_writes_in_one_component_are_rejected() -> None:
    engine = TransactionalRepairEngine({"x": 3}, mismatch_evaluator=_absolute_mismatch)
    collar = _collar("c")
    one = _strict(
        engine,
        proposal_id="one",
        collar=collar,
        reads={"x"},
        recovery=lambda state: {"x": state["x"] - 1},
    )
    two = _strict(
        engine,
        proposal_id="two",
        collar=collar,
        reads={"x"},
        recovery=lambda state: {"x": state["x"] - 2},
    )
    receipt = engine.commit_batch([one, two])[0]
    assert not receipt.committed
    assert not _check(receipt, "compatible_component_writes")
    assert engine.state["x"] == 3
