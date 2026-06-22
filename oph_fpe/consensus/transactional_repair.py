from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from oph_fpe.claims import RECOVERED_CORE, with_claim_metadata

State = dict[str, Any]
MeasureFn = Callable[[Mapping[str, Any]], Any]
BoundaryFn = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class Transaction:
    """Prepared finite quotient repair transaction."""

    tx_id: str
    read_set: frozenset[str]
    write_set: frozenset[str]
    snapshot: tuple[tuple[str, Any], ...]
    payload: tuple[tuple[str, Any], ...]
    read_versions: tuple[tuple[str, Any], ...] = ()

    @property
    def snapshot_dict(self) -> dict[str, Any]:
        return dict(self.snapshot)

    @property
    def payload_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    @property
    def read_versions_dict(self) -> dict[str, Any]:
        return dict(self.read_versions)


@dataclass(frozen=True)
class CommitResult:
    status: str
    committed: bool
    state: State
    before_measure: Any
    after_measure: Any
    before_boundary: Any
    after_boundary: Any
    reason: str | None = None


def prepare_transaction(
    state: Mapping[str, Any],
    *,
    tx_id: str,
    read_set: Iterable[str],
    payload: Mapping[str, Any],
    versions: Mapping[str, Any] | None = None,
) -> Transaction:
    reads = frozenset(str(key) for key in read_set)
    writes = frozenset(str(key) for key in payload)
    missing_reads = sorted(key for key in reads if key not in state)
    if missing_reads:
        raise KeyError(f"read set contains missing registers: {missing_reads}")
    if not writes:
        raise ValueError("transaction payload must write at least one register")
    snapshot = tuple(sorted((key, state[key]) for key in reads))
    payload_items = tuple(sorted((str(key), value) for key, value in payload.items()))
    version_items: tuple[tuple[str, Any], ...] = ()
    if versions is not None:
        missing_versions = sorted(key for key in reads if key not in versions)
        if missing_versions:
            raise KeyError(f"version map is missing read registers: {missing_versions}")
        version_items = tuple(sorted((key, versions[key]) for key in reads))
    return Transaction(
        tx_id=str(tx_id),
        read_set=reads,
        write_set=writes,
        snapshot=snapshot,
        payload=payload_items,
        read_versions=version_items,
    )


def apply_payload(state: Mapping[str, Any], tx: Transaction) -> State:
    next_state = dict(state)
    next_state.update(tx.payload_dict)
    return next_state


def commit_transaction(
    state: Mapping[str, Any],
    tx: Transaction,
    *,
    measure: MeasureFn,
    boundary: BoundaryFn,
    versions: Mapping[str, Any] | None = None,
) -> CommitResult:
    before = dict(state)
    before_measure = measure(before)
    before_boundary = boundary(before)
    snapshot = tx.snapshot_dict
    read_versions = tx.read_versions_dict
    if read_versions and versions is None:
        return CommitResult(
            status="MISSING_READ_VERSIONS",
            committed=False,
            state=before,
            before_measure=before_measure,
            after_measure=before_measure,
            before_boundary=before_boundary,
            after_boundary=before_boundary,
            reason="transaction carries read versions but no current version map was supplied",
        )
    for key, expected_version in read_versions.items():
        if versions is None or versions.get(key) != expected_version:
            return CommitResult(
                status="STALE_READ_VERSION",
                committed=False,
                state=before,
                before_measure=before_measure,
                after_measure=before_measure,
                before_boundary=before_boundary,
                after_boundary=before_boundary,
                reason=f"register {key!r} version changed since prepare",
            )
    for key, expected in snapshot.items():
        if before.get(key) != expected:
            return CommitResult(
                status="STALE_SNAPSHOT",
                committed=False,
                state=before,
                before_measure=before_measure,
                after_measure=before_measure,
                before_boundary=before_boundary,
                after_boundary=before_boundary,
                reason=f"register {key!r} changed since prepare",
            )
    after = apply_payload(before, tx)
    after_boundary = boundary(after)
    after_measure = measure(after)
    if after_boundary != before_boundary:
        return CommitResult(
            status="BOUNDARY_CHANGED",
            committed=False,
            state=before,
            before_measure=before_measure,
            after_measure=after_measure,
            before_boundary=before_boundary,
            after_boundary=after_boundary,
            reason="transaction changed protected boundary/sector data",
        )
    if not _strictly_descends(after_measure, before_measure):
        return CommitResult(
            status="NO_DESCENT",
            committed=False,
            state=before,
            before_measure=before_measure,
            after_measure=after_measure,
            before_boundary=before_boundary,
            after_boundary=after_boundary,
            reason="transaction did not strictly lower the declared measure",
        )
    return CommitResult(
        status="COMMITTED",
        committed=True,
        state=after,
        before_measure=before_measure,
        after_measure=after_measure,
        before_boundary=before_boundary,
        after_boundary=after_boundary,
    )


def transactions_conflict(left: Transaction, right: Transaction) -> bool:
    return bool(
        left.write_set & (right.read_set | right.write_set)
        or right.write_set & (left.read_set | left.write_set)
    )


def conflict_components(transactions: Sequence[Transaction]) -> list[list[Transaction]]:
    txs = sorted(transactions, key=lambda tx: tx.tx_id)
    remaining = set(range(len(txs)))
    components: list[list[Transaction]] = []
    while remaining:
        start = min(remaining)
        stack = [start]
        group: set[int] = set()
        remaining.remove(start)
        while stack:
            index = stack.pop()
            group.add(index)
            neighbors = [
                other
                for other in sorted(remaining)
                if transactions_conflict(txs[index], txs[other])
            ]
            for other in neighbors:
                remaining.remove(other)
                stack.append(other)
        components.append([txs[index] for index in sorted(group, key=lambda idx: txs[idx].tx_id)])
    return components


def aggregate_component(
    state: Mapping[str, Any],
    component: Sequence[Transaction],
    *,
    tx_id: str | None = None,
) -> Transaction:
    if not component:
        raise ValueError("cannot aggregate an empty conflict component")
    ordered = sorted(component, key=lambda tx: tx.tx_id)
    read_set: set[str] = set()
    payload: dict[str, Any] = {}
    for tx in ordered:
        read_set.update(tx.read_set)
        payload.update(tx.payload_dict)
    return prepare_transaction(
        state,
        tx_id=tx_id or "+".join(tx.tx_id for tx in ordered),
        read_set=read_set | set(payload),
        payload=payload,
    )


def atomic_diamond_report(
    state: Mapping[str, Any],
    left: Transaction,
    right: Transaction,
    *,
    measure: MeasureFn,
    boundary: BoundaryFn,
) -> dict[str, Any]:
    if transactions_conflict(left, right):
        same_component = left.tx_id == right.tx_id
        report = {
            "mode": "atomic_conflict_component_diamond",
            "same_conflict_component": same_component,
            "disjoint_components": False,
            "DISTRIBUTED_LOCAL_DIAMOND_RECEIPT": same_component,
            "receipt": same_component,
            "status": "SAME_COMPONENT" if same_component else "CONFLICT_REQUIRES_AGGREGATE_TRANSACTION",
            "claim_boundary": (
                "Conflicting primitive repairs do not commit independently; they must be replaced by one "
                "canonical aggregate transaction before a local-diamond claim is meaningful."
            ),
        }
        return with_claim_metadata(report, claim_level=RECOVERED_CORE, receipt="DISTRIBUTED_LOCAL_DIAMOND_RECEIPT")

    left_first = commit_transaction(state, left, measure=measure, boundary=boundary)
    if not left_first.committed:
        return _diamond_failure("left_first_failed", left_first)
    left_then_right = commit_transaction(left_first.state, right, measure=measure, boundary=boundary)
    if not left_then_right.committed:
        return _diamond_failure("left_then_right_failed", left_then_right)

    right_first = commit_transaction(state, right, measure=measure, boundary=boundary)
    if not right_first.committed:
        return _diamond_failure("right_first_failed", right_first)
    right_then_left = commit_transaction(right_first.state, left, measure=measure, boundary=boundary)
    if not right_then_left.committed:
        return _diamond_failure("right_then_left_failed", right_then_left)

    same_terminal = left_then_right.state == right_then_left.state
    report = {
        "mode": "atomic_conflict_component_diamond",
        "same_conflict_component": False,
        "disjoint_components": True,
        "DISTRIBUTED_LOCAL_DIAMOND_RECEIPT": same_terminal,
        "receipt": same_terminal,
        "left_then_right": left_then_right.state,
        "right_then_left": right_then_left.state,
        "normal_form_hash": canonical_state_hash(left_then_right.state) if same_terminal else None,
        "claim_boundary": (
            "Receipt for exact disjoint transaction commutation on a finite quotient state; it is not "
            "evidence that conflicting primitive repairs may commit separately."
        ),
    }
    return with_claim_metadata(report, claim_level=RECOVERED_CORE, receipt="DISTRIBUTED_LOCAL_DIAMOND_RECEIPT")


def validation_support(write_set: Iterable[str], measure_supports: Iterable[Iterable[str]]) -> frozenset[str]:
    writes = frozenset(str(key) for key in write_set)
    support: set[str] = set()
    for term_support in measure_supports:
        term = {str(key) for key in term_support}
        if term & writes:
            support.update(term)
    return frozenset(support)


def selected_fiber_branch_elimination_report(
    *,
    boundary_value: Any,
    root: str,
    root_value: Any,
    parents: Mapping[str, str],
    extension_maps: Mapping[str, Callable[[Any, Any], Any]],
    candidates: Iterable[Mapping[str, Any]],
    consistency_check: Callable[[Mapping[str, Any]], bool],
    normalizer: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_rows = [dict(row) for row in candidates]
    selected = _functional_extension(
        root=root,
        root_value=root_value,
        parents=parents,
        extension_maps=extension_maps,
        boundary_value=boundary_value,
    )
    selected_consistent = bool(consistency_check(selected))
    normal_forms = [dict(normalizer(row)) for row in candidate_rows]
    selected_hash = canonical_state_hash(selected) if selected_consistent else None
    normal_hashes = [canonical_state_hash(row) for row in normal_forms]
    inconsistent_candidates = [row for row in candidate_rows if not bool(consistency_check(row))]
    same_normal = bool(normal_hashes and selected_hash and set(normal_hashes) == {selected_hash})
    nontrivial = len(candidate_rows) >= 2 and bool(inconsistent_candidates)
    selected_receipt = bool(nontrivial and selected_consistent and same_normal)
    status = "SELECTED_FIBER_ELIMINATED" if selected_receipt else "OBSTRUCTED" if not selected_consistent else "INCOMPLETE_EVIDENCE"
    report = {
        "mode": "selected_fiber_branch_elimination_v1",
        "status": status,
        "OBSTRUCTED": status == "OBSTRUCTED",
        "SELECTED_FIBER_NONTRIVIAL_ELIMINATION_RECEIPT": selected_receipt,
        "SAME_BOUNDARY_MULTISTART_CONFLUENCE_RECEIPT": bool(selected_consistent and same_normal),
        "QUOTIENT_NORMAL_FORM_CANONICAL_HASH_RECEIPT": bool(selected_consistent and selected_hash),
        "receipt": selected_receipt,
        "boundary_value": boundary_value,
        "candidate_count": len(candidate_rows),
        "inconsistent_candidate_count": len(inconsistent_candidates),
        "selected_consistent": selected_consistent,
        "selected_extension": selected,
        "normal_form_hash": selected_hash,
        "normal_form_hashes": normal_hashes,
        "inconsistent_candidate_witnesses": inconsistent_candidates[:2],
        "claim_boundary": (
            "Finite selected-fiber check: multiple same-boundary candidates are eliminated only because "
            "the functional extension and consistency checks pick one quotient normal form. The hash is "
            "an evidence receipt for equality after quotienting, not a selector among distinct endpoints."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=RECOVERED_CORE,
        receipt="SELECTED_FIBER_NONTRIVIAL_ELIMINATION_RECEIPT",
    )


def classify_union_repair(
    candidates: Iterable[Mapping[str, Any]],
    *,
    consistency_check: Callable[[Mapping[str, Any]], bool],
    score: Callable[[Mapping[str, Any]], Any],
) -> dict[str, Any]:
    rows = [dict(row) for row in candidates if bool(consistency_check(row))]
    minimizers: list[State] = []
    if not rows:
        status = "OBSTRUCTED"
        selected: State | None = None
    else:
        best_score = min(score(row) for row in rows)
        minimizers = [row for row in rows if score(row) == best_score]
        unique_hashes = {canonical_state_hash(row) for row in minimizers}
        if len(unique_hashes) > 1:
            status = "AMBIGUOUS_UNION_REPAIR"
            selected = None
        else:
            status = "UNIQUE_UNION_REPAIR"
            selected = minimizers[0]
    report = {
        "mode": "union_repair_classifier_v1",
        "status": status,
        "AMBIGUOUS_UNION_REPAIR": status == "AMBIGUOUS_UNION_REPAIR",
        "OBSTRUCTED": status == "OBSTRUCTED",
        "selected": selected,
        "candidate_count": len(rows),
        "minimizer_count": len(minimizers),
        "ambiguity_witnesses": minimizers[:2] if status == "AMBIGUOUS_UNION_REPAIR" else [],
        "claim_boundary": (
            "A normal-form hash may confirm equality of quotient states. It must not choose between "
            "two physically distinct consistent minimizing quotient states."
        ),
    }
    return with_claim_metadata(report, claim_level=RECOVERED_CORE, receipt="AMBIGUOUS_UNION_REPAIR")


def transactional_repair_receipt(
    state: Mapping[str, Any],
    transactions: Sequence[Transaction],
    *,
    measure: MeasureFn,
    boundary: BoundaryFn,
) -> dict[str, Any]:
    components = conflict_components(transactions)
    aggregate_results = []
    current = dict(state)
    for index, component in enumerate(components):
        aggregate = aggregate_component(current, component, tx_id=f"component_{index}")
        result = commit_transaction(current, aggregate, measure=measure, boundary=boundary)
        aggregate_results.append(
            {
                "component_index": index,
                "primitive_transaction_ids": [tx.tx_id for tx in component],
                "status": result.status,
                "committed": result.committed,
                "before_measure": result.before_measure,
                "after_measure": result.after_measure,
            }
        )
        if result.committed:
            current = result.state
    all_committed = all(row["committed"] for row in aggregate_results)
    report = {
        "mode": "transactional_repair_receipt_v1",
        "SEAM_REPAIR_DESCENT_RECEIPT": all_committed,
        "SEAM_ATOMIC_COMMIT_RECEIPT": all_committed and len(components) <= len(transactions),
        "receipt": all_committed,
        "component_count": len(components),
        "aggregate_results": aggregate_results,
        "final_state": current,
        "normal_form_hash": canonical_state_hash(current),
        "claim_boundary": (
            "Exact finite transaction receipt for quotient repair. It validates descent and atomic "
            "component commits, but repair completeness and selected-fiber uniqueness require their own checks."
        ),
    }
    return with_claim_metadata(report, claim_level=RECOVERED_CORE, receipt="SEAM_ATOMIC_COMMIT_RECEIPT")


def exhaustive_transition_graph_report(
    states: Iterable[Mapping[str, Any]],
    transition_fn: Callable[[Mapping[str, Any]], Iterable[Mapping[str, Any]]],
    *,
    measure: MeasureFn,
    consistency_check: Callable[[Mapping[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    rows = [dict(state) for state in states]
    by_hash = {canonical_state_hash(state): state for state in rows}
    adjacency: dict[str, list[str]] = {}
    edge_rows: list[dict[str, Any]] = []
    for state in rows:
        state_hash = canonical_state_hash(state)
        adjacency[state_hash] = []
        for target in transition_fn(state):
            target_state = dict(target)
            target_hash = canonical_state_hash(target_state)
            by_hash.setdefault(target_hash, target_state)
            adjacency[state_hash].append(target_hash)
            before_measure = measure(state)
            after_measure = measure(target_state)
            edge_rows.append(
                {
                    "source": state_hash,
                    "target": target_hash,
                    "before_measure": before_measure,
                    "after_measure": after_measure,
                    "strict_descent": _strictly_descends(after_measure, before_measure),
                }
            )
    for state_hash in by_hash:
        adjacency.setdefault(state_hash, [])

    strict_descent_violations = [row for row in edge_rows if not row["strict_descent"]]
    local_diamond_violations: list[dict[str, Any]] = []
    for state_hash, successors in adjacency.items():
        unique_successors = sorted(set(successors))
        for left_index, left in enumerate(unique_successors):
            for right in unique_successors[left_index + 1 :]:
                if not (_reachable_hashes(left, adjacency) & _reachable_hashes(right, adjacency)):
                    local_diamond_violations.append({"source": state_hash, "left": left, "right": right})

    terminal_violations: list[dict[str, Any]] = []
    terminal_consistency_violations: list[dict[str, Any]] = []
    for state_hash in sorted(by_hash):
        reachable = _reachable_hashes(state_hash, adjacency)
        terminals = sorted(node for node in reachable if not adjacency.get(node))
        if len(terminals) != 1:
            terminal_violations.append({"source": state_hash, "terminal_count": len(terminals), "terminals": terminals})
        if consistency_check is not None:
            for terminal in terminals:
                if not bool(consistency_check(by_hash[terminal])):
                    terminal_consistency_violations.append({"source": state_hash, "terminal": terminal})

    nontrivial_sccs = [
        component
        for component in _strongly_connected_components(adjacency)
        if len(component) > 1 or any(node in adjacency.get(node, []) for node in component)
    ]
    receipt = bool(
        not strict_descent_violations
        and not local_diamond_violations
        and not terminal_violations
        and not terminal_consistency_violations
        and not nontrivial_sccs
    )
    report = {
        "mode": "exhaustive_transaction_graph_report_v1",
        "EXHAUSTIVE_TRANSACTION_GRAPH_RECEIPT": receipt,
        "DISTRIBUTED_LOCAL_DIAMOND_RECEIPT": not local_diamond_violations,
        "DISTRIBUTED_REPAIR_COMPLETENESS_RECEIPT": not terminal_consistency_violations and not terminal_violations,
        "SEAM_REPAIR_DESCENT_RECEIPT": not strict_descent_violations,
        "receipt": receipt,
        "state_count": len(by_hash),
        "edge_count": len(edge_rows),
        "strict_descent_violation_count": len(strict_descent_violations),
        "local_diamond_violation_count": len(local_diamond_violations),
        "nontrivial_scc_count": len(nontrivial_sccs),
        "unique_terminal_violation_count": len(terminal_violations),
        "terminal_consistency_violation_count": len(terminal_consistency_violations),
        "strict_descent_violations": strict_descent_violations[:4],
        "local_diamond_violations": local_diamond_violations[:4],
        "terminal_violations": terminal_violations[:4],
        "claim_boundary": (
            "Exhaustive finite quotient-state transition graph check for small exported nets. It rejects "
            "terminating forks, non-descending edges, nontrivial SCCs, and inconsistent terminal states."
        ),
    }
    return with_claim_metadata(report, claim_level=RECOVERED_CORE, receipt="EXHAUSTIVE_TRANSACTION_GRAPH_RECEIPT")


def gauge_relabeling_invariance_report(
    representatives: Iterable[Mapping[str, Any]],
    *,
    quotient_map: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    quotient_transition_fn: Callable[[Mapping[str, Any]], Iterable[Mapping[str, Any]]],
) -> dict[str, Any]:
    grouped: dict[str, list[State]] = {}
    for representative in representatives:
        quotient_state = dict(quotient_map(dict(representative)))
        grouped.setdefault(canonical_state_hash(quotient_state), []).append(dict(representative))

    violations: list[dict[str, Any]] = []
    for quotient_hash, reps in grouped.items():
        transition_hash_sets = []
        for rep in reps:
            quotient_state = dict(quotient_map(rep))
            transition_hash_sets.append(
                sorted(canonical_state_hash(dict(next_state)) for next_state in quotient_transition_fn(quotient_state))
            )
        if any(row != transition_hash_sets[0] for row in transition_hash_sets[1:]):
            violations.append({"quotient_hash": quotient_hash, "representative_count": len(reps)})
    receipt = not violations
    report = {
        "mode": "gauge_relabeling_invariance_report_v1",
        "GAUGE_RELABELING_QUOTIENT_INVARIANCE_RECEIPT": receipt,
        "receipt": receipt,
        "quotient_class_count": len(grouped),
        "violation_count": len(violations),
        "violations": violations[:4],
        "claim_boundary": (
            "Finite representative-layer check: gauge/carrier relabelings must canonicalize to the same "
            "quotient transitions and normal-form serialization."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=RECOVERED_CORE,
        receipt="GAUGE_RELABELING_QUOTIENT_INVARIANCE_RECEIPT",
    )


def canonical_state_hash(state: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(state), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _strictly_descends(after: Any, before: Any) -> bool:
    try:
        return bool(after < before)
    except TypeError as exc:
        raise TypeError(f"measure values must be comparable, got {before!r} and {after!r}") from exc


def _diamond_failure(stage: str, result: CommitResult) -> dict[str, Any]:
    report = {
        "mode": "atomic_conflict_component_diamond",
        "DISTRIBUTED_LOCAL_DIAMOND_RECEIPT": False,
        "receipt": False,
        "status": stage,
        "failed_commit_status": result.status,
        "reason": result.reason,
    }
    return with_claim_metadata(report, claim_level=RECOVERED_CORE, receipt="DISTRIBUTED_LOCAL_DIAMOND_RECEIPT")


def _reachable_hashes(start: str, adjacency: Mapping[str, Sequence[str]]) -> set[str]:
    seen: set[str] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacency.get(node, ()))
    return seen


def _strongly_connected_components(adjacency: Mapping[str, Sequence[str]]) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    components: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for successor in adjacency.get(node, ()):
            if successor not in indices:
                visit(successor)
                lowlinks[node] = min(lowlinks[node], lowlinks[successor])
            elif successor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[successor])
        if lowlinks[node] == indices[node]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            components.append(sorted(component))

    for node in sorted(adjacency):
        if node not in indices:
            visit(node)
    return components


def _functional_extension(
    *,
    root: str,
    root_value: Any,
    parents: Mapping[str, str],
    extension_maps: Mapping[str, Callable[[Any, Any], Any]],
    boundary_value: Any,
) -> State:
    state: State = {str(root): root_value}
    pending = {str(child): str(parent) for child, parent in parents.items()}
    while pending:
        progressed = False
        for child, parent in list(pending.items()):
            if parent not in state:
                continue
            if child not in extension_maps:
                raise KeyError(f"missing extension map for child {child!r}")
            state[child] = extension_maps[child](state[parent], boundary_value)
            del pending[child]
            progressed = True
        if not progressed:
            raise ValueError("parent map is not a rooted tree reachable from the declared root")
    return state
