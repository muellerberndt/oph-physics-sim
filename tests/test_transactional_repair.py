from __future__ import annotations

from oph_fpe.consensus import (
    aggregate_component,
    atomic_diamond_report,
    classify_union_repair,
    commit_transaction,
    conflict_components,
    exhaustive_transition_graph_report,
    gauge_relabeling_invariance_report,
    prepare_transaction,
    selected_fiber_branch_elimination_report,
    transactional_repair_receipt,
    transactions_conflict,
    validation_support,
)


def test_stale_transaction_is_not_a_repair_step():
    state = {"root": 0, "a": 1}
    tx = prepare_transaction(state, tx_id="repair_a", read_set={"root", "a"}, payload={"a": 0})

    result = commit_transaction(
        {"root": 1, "a": 1},
        tx,
        measure=_measure,
        boundary=_boundary,
    )

    assert result.committed is False
    assert result.status == "STALE_SNAPSHOT"
    assert result.state == {"root": 1, "a": 1}


def test_in_place_state_mutation_cannot_rewrite_prepared_snapshot():
    state = {"root": 0, "a": [2]}
    tx = prepare_transaction(state, tx_id="repair_a", read_set={"a"}, payload={"a": [0]})
    state["a"][0] = 1

    result = commit_transaction(
        state,
        tx,
        measure=lambda row: row["a"][0],
        boundary=_boundary,
    )

    assert result.committed is False
    assert result.status == "STALE_SNAPSHOT"
    assert tx.snapshot_dict == {"a": [2]}


def test_versioned_transaction_aborts_on_stale_read_version():
    state = {"root": 0, "a": 1}
    tx = prepare_transaction(
        state,
        tx_id="repair_a",
        read_set={"root", "a"},
        payload={"a": 0},
        versions={"root": 7, "a": 3},
    )

    result = commit_transaction(
        state,
        tx,
        measure=_measure,
        boundary=_boundary,
        versions={"root": 7, "a": 4},
    )

    assert result.committed is False
    assert result.status == "STALE_READ_VERSION"


def test_duplicate_delivery_replays_as_stale_snapshot():
    state = {"root": 0, "a": 1}
    tx = prepare_transaction(state, tx_id="repair_a", read_set={"a"}, payload={"a": 0})

    first = commit_transaction(state, tx, measure=_measure, boundary=_boundary)
    replay = commit_transaction(first.state, tx, measure=_measure, boundary=_boundary)

    assert first.committed is True
    assert replay.committed is False
    assert replay.status == "STALE_SNAPSHOT"


def test_empty_transaction_batch_cannot_certify_atomic_descent():
    state = {"root": 0, "a": 1}

    report = transactional_repair_receipt(
        state,
        [],
        measure=_measure,
        boundary=_boundary,
    )

    assert report["SEAM_ATOMIC_COMMIT_RECEIPT"] is False
    assert report["SEAM_REPAIR_DESCENT_RECEIPT"] is False
    assert report["receipt"] is False
    assert report["primitive_transaction_count"] == 0
    assert report["nonempty_transaction_set"] is False


def test_disjoint_transactions_commute_by_atomic_diamond():
    state = {"root": 0, "a": 1, "b": 1}
    left = prepare_transaction(state, tx_id="repair_a", read_set={"a"}, payload={"a": 0})
    right = prepare_transaction(state, tx_id="repair_b", read_set={"b"}, payload={"b": 0})

    report = atomic_diamond_report(state, left, right, measure=_measure, boundary=_boundary)

    assert report["DISTRIBUTED_LOCAL_DIAMOND_RECEIPT"] is True
    assert report["left_then_right"] == {"root": 0, "a": 0, "b": 0}
    assert report["right_then_left"] == {"root": 0, "a": 0, "b": 0}


def test_duplicate_transaction_id_cannot_certify_a_conflicting_diamond():
    state = {"root": 0, "a": 2}
    left = prepare_transaction(state, tx_id="duplicate", read_set={"a"}, payload={"a": 1})
    right = prepare_transaction(state, tx_id="duplicate", read_set={"a"}, payload={"a": 0})

    report = atomic_diamond_report(state, left, right, measure=_measure, boundary=_boundary)

    assert report["duplicate_transaction_id"] is True
    assert report["DISTRIBUTED_LOCAL_DIAMOND_RECEIPT"] is False
    assert report["receipt"] is False
    assert report["status"] == "CONFLICT_REQUIRES_AGGREGATE_TRANSACTION"


def test_conflicting_repairs_are_aggregated_before_commit():
    state = {"root": 0, "a": 1, "b": 1}
    left = prepare_transaction(state, tx_id="repair_a", read_set={"a"}, payload={"a": 0})
    right = prepare_transaction(state, tx_id="repair_b_after_a", read_set={"a", "b"}, payload={"b": 0})

    assert transactions_conflict(left, right) is True
    components = conflict_components([right, left])
    assert [[tx.tx_id for tx in component] for component in components] == [["repair_a", "repair_b_after_a"]]

    aggregate = aggregate_component(state, components[0], tx_id="component")
    result = commit_transaction(state, aggregate, measure=_measure, boundary=_boundary)
    assert result.committed is True
    assert result.state == {"root": 0, "a": 0, "b": 0}

    receipt = transactional_repair_receipt(state, [left, right], measure=_measure, boundary=_boundary)
    assert receipt["SEAM_ATOMIC_COMMIT_RECEIPT"] is True
    assert receipt["SEAM_REPAIR_DESCENT_RECEIPT"] is True
    assert receipt["component_count"] == 1


def test_aggregate_rejects_incompatible_overlapping_writes() -> None:
    state = {"root": 0, "a": 2}
    left = prepare_transaction(state, tx_id="left", read_set={"a"}, payload={"a": 1})
    right = prepare_transaction(state, tx_id="right", read_set={"a"}, payload={"a": 0})

    try:
        aggregate_component(state, [left, right])
    except ValueError as exc:
        assert "incompatible aggregate writes" in str(exc)
    else:
        raise AssertionError("incompatible overlapping writes must be rejected")

    receipt = transactional_repair_receipt(
        state,
        [left, right],
        measure=_measure,
        boundary=_boundary,
    )
    assert receipt["SEAM_ATOMIC_COMMIT_RECEIPT"] is False
    assert receipt["aggregate_results"][0]["status"] == "INCOMPATIBLE_OVERLAPPING_WRITES"


def test_stale_primitive_cannot_be_laundered_through_a_fresh_aggregate() -> None:
    prepared_state = {"root": 0, "a": 2}
    stale = prepare_transaction(
        prepared_state,
        tx_id="stale",
        read_set={"a"},
        payload={"a": 0},
    )
    current_state = {"root": 0, "a": 1}

    direct = commit_transaction(current_state, stale, measure=_measure, boundary=_boundary)
    assert direct.committed is False
    assert direct.status == "STALE_SNAPSHOT"

    try:
        aggregate_component(current_state, [stale])
    except ValueError as exc:
        assert "is not fresh (STALE_SNAPSHOT)" in str(exc)
    else:
        raise AssertionError("a stale primitive transaction must not be re-prepared by aggregation")

    receipt = transactional_repair_receipt(
        current_state,
        [stale],
        measure=_measure,
        boundary=_boundary,
    )
    assert receipt["SEAM_ATOMIC_COMMIT_RECEIPT"] is False
    assert receipt["SEAM_REPAIR_DESCENT_RECEIPT"] is False
    assert receipt["aggregate_results"][0]["status"] == "STALE_PRIMITIVE_SNAPSHOT"
    assert receipt["final_state"] == current_state


def test_validation_support_includes_every_changed_mismatch_endpoint():
    supports = [{"a", "b"}, {"b", "c"}, {"d", "e"}]

    assert validation_support({"b"}, supports) == frozenset({"a", "b", "c"})
    assert validation_support({"d"}, supports) == frozenset({"d", "e"})


def test_selected_fiber_eliminates_nontrivial_same_boundary_candidates():
    candidates = [
        {"r": 0, "a": 1, "b": 0},
        {"r": 0, "a": 0, "b": 0},
        {"r": 0, "a": 0, "b": 1},
    ]

    report = selected_fiber_branch_elimination_report(
        boundary_value="zero",
        root="r",
        root_value=0,
        parents={"a": "r", "b": "a"},
        extension_maps={"a": lambda parent, _boundary_value: parent, "b": lambda parent, _boundary_value: parent},
        candidates=candidates,
        consistency_check=lambda row: row["r"] == row["a"] == row["b"],
        normalizer=lambda _row: {"r": 0, "a": 0, "b": 0},
    )

    assert report["SELECTED_FIBER_NONTRIVIAL_ELIMINATION_RECEIPT"] is True
    assert report["SAME_BOUNDARY_MULTISTART_CONFLUENCE_RECEIPT"] is True
    assert report["QUOTIENT_NORMAL_FORM_CANONICAL_HASH_RECEIPT"] is True
    assert report["inconsistent_candidate_count"] == 2


def test_ambiguous_union_repair_is_not_hash_selected():
    report = classify_union_repair(
        [{"x": 0}, {"x": 1}],
        consistency_check=lambda _row: True,
        score=lambda _row: 0,
    )

    assert report["status"] == "AMBIGUOUS_UNION_REPAIR"
    assert report["AMBIGUOUS_UNION_REPAIR"] is True
    assert report["selected"] is None
    assert report["ambiguity_witnesses"] == [{"x": 0}, {"x": 1}]


def test_obstructed_union_repair_does_not_force_settling():
    report = classify_union_repair(
        [{"x": 0}],
        consistency_check=lambda _row: False,
        score=lambda _row: 0,
    )

    assert report["status"] == "OBSTRUCTED"
    assert report["OBSTRUCTED"] is True
    assert report["selected"] is None


def test_obstructed_selected_fiber_reports_failed_chord_check():
    report = selected_fiber_branch_elimination_report(
        boundary_value="zero",
        root="r",
        root_value=0,
        parents={"a": "r"},
        extension_maps={"a": lambda parent, _boundary_value: parent},
        candidates=[{"r": 0, "a": 0}],
        consistency_check=lambda _row: False,
        normalizer=lambda row: row,
    )

    assert report["status"] == "OBSTRUCTED"
    assert report["OBSTRUCTED"] is True
    assert report["SELECTED_FIBER_NONTRIVIAL_ELIMINATION_RECEIPT"] is False


def test_exhaustive_graph_rejects_terminating_fork():
    states = [{"x": "a"}, {"x": "b"}, {"x": "c"}]

    def transitions(state):
        if state["x"] == "a":
            return [{"x": "b"}, {"x": "c"}]
        return []

    report = exhaustive_transition_graph_report(
        states,
        transitions,
        measure=lambda state: {"a": 1, "b": 0, "c": 0}[state["x"]],
        consistency_check=lambda state: state["x"] in {"b", "c"},
    )

    assert report["EXHAUSTIVE_TRANSACTION_GRAPH_RECEIPT"] is False
    assert report["local_diamond_violation_count"] == 1
    assert report["unique_terminal_violation_count"] == 1


def test_exhaustive_graph_rejects_non_descending_residual_step():
    states = [{"x": 1}, {"x": 0}]

    def transitions(state):
        if state["x"] == 1:
            return [{"x": 0}]
        return []

    report = exhaustive_transition_graph_report(
        states,
        transitions,
        measure=lambda _state: 0,
        consistency_check=lambda state: state["x"] == 0,
    )

    assert report["EXHAUSTIVE_TRANSACTION_GRAPH_RECEIPT"] is False
    assert report["strict_descent_violation_count"] == 1


def test_exhaustive_graph_requires_declared_carrier_closure():
    report = exhaustive_transition_graph_report(
        [{"x": 1}],
        lambda state: [{"x": 0}] if state["x"] == 1 else [],
        measure=lambda state: state["x"],
        consistency_check=lambda state: state["x"] == 0,
    )

    assert report["EXHAUSTIVE_TRANSACTION_GRAPH_RECEIPT"] is False
    assert report["DISTRIBUTED_REPAIR_COMPLETENESS_RECEIPT"] is False
    assert report["carrier_closure_violation_count"] == 1
    assert report["discovered_undeclared_target_count"] == 1


def test_exhaustive_graph_checks_consistency_implies_terminal():
    states = [{"x": 1}, {"x": 0}]
    report = exhaustive_transition_graph_report(
        states,
        lambda state: [{"x": 0}] if state["x"] == 1 else [],
        measure=lambda state: state["x"],
        consistency_check=lambda _state: True,
    )

    assert report["EXHAUSTIVE_TRANSACTION_GRAPH_RECEIPT"] is False
    assert report["consistent_nonterminal_violation_count"] == 1


def test_exhaustive_graph_accepts_closed_exact_normal_form_carrier():
    states = [{"x": 1}, {"x": 0}]
    report = exhaustive_transition_graph_report(
        states,
        lambda state: [{"x": 0}] if state["x"] == 1 else [],
        measure=lambda state: state["x"],
        consistency_check=lambda state: state["x"] == 0,
    )

    assert report["EXHAUSTIVE_TRANSACTION_GRAPH_RECEIPT"] is True
    assert report["DISTRIBUTED_REPAIR_COMPLETENESS_RECEIPT"] is True
    assert report["carrier_closure_violation_count"] == 0
    assert report["consistent_nonterminal_violation_count"] == 0


def test_exhaustive_graph_rejects_empty_declared_carrier():
    report = exhaustive_transition_graph_report(
        [],
        lambda _state: [],
        measure=lambda _state: 0,
        consistency_check=lambda _state: True,
    )

    assert report["EXHAUSTIVE_TRANSACTION_GRAPH_RECEIPT"] is False
    assert report["DECLARED_CARRIER_CLOSURE_RECEIPT"] is False
    assert report["DISTRIBUTED_LOCAL_DIAMOND_RECEIPT"] is False
    assert report["DISTRIBUTED_REPAIR_COMPLETENESS_RECEIPT"] is False


def test_gauge_duplicates_have_identical_quotient_transitions():
    representatives = [
        {"x": 1, "gauge": "left"},
        {"x": 1, "gauge": "right"},
        {"x": 0, "gauge": "left"},
    ]

    report = gauge_relabeling_invariance_report(
        representatives,
        quotient_map=lambda row: {"x": row["x"]},
        quotient_transition_fn=lambda row: [{"x": 0}] if row["x"] else [],
    )

    assert report["GAUGE_RELABELING_QUOTIENT_INVARIANCE_RECEIPT"] is True
    assert report["representative_pair_count"] == 1
    assert report["violation_count"] == 0


def test_gauge_invariance_receipt_rejects_vacuous_representative_set():
    report = gauge_relabeling_invariance_report(
        [{"x": 1, "gauge": "only"}],
        quotient_map=lambda row: {"x": row["x"]},
        quotient_transition_fn=lambda _row: [{"x": 0}],
    )

    assert report["GAUGE_RELABELING_QUOTIENT_INVARIANCE_RECEIPT"] is False
    assert report["nonvacuous_representative_comparison"] is False
    assert report["representative_pair_count"] == 0


def test_gauge_invariance_checks_representative_dynamics_before_quotienting():
    representatives = [
        {"x": 1, "gauge": "left"},
        {"x": 1, "gauge": "right"},
    ]

    report = gauge_relabeling_invariance_report(
        representatives,
        quotient_map=lambda row: {"x": row["x"]},
        quotient_transition_fn=lambda row: [
            {"x": 0, "gauge": row["gauge"]}
            if row["gauge"] == "left"
            else {"x": 2, "gauge": row["gauge"]}
        ],
    )

    assert report["GAUGE_RELABELING_QUOTIENT_INVARIANCE_RECEIPT"] is False
    assert report["receipt"] is False
    assert report["violation_count"] == 1


def _measure(state):
    return (int(state.get("a", 0)) + int(state.get("b", 0)),)


def _boundary(state):
    return state.get("root", 0)
