"""Exact obstruction oracles for tiny, explicitly enumerated models.

The routines in this module deliberately stop at the mathematical finite-model
boundary.  In particular, a zero graph holonomy, a singleton supplied fiber,
or a lumpable supplied kernel is not a receipt for a physical carrier law,
continuum limit, H3 geometry, KMS clock, or Standard-Model emergence.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Callable, Generic, Hashable, Iterable, Mapping, Sequence, TypeVar

from oph_fpe.ontology.states import FiberStatus


StateT = TypeVar("StateT", bound=Hashable)
BoundaryT = TypeVar("BoundaryT")
VertexT = TypeVar("VertexT", bound=Hashable)
QuotientT = TypeVar("QuotientT", bound=Hashable)

ExactInput = int | Fraction

FINITE_MODEL_CLAIM_BOUNDARY = (
    "Exact only for the explicitly supplied finite state space, graph, or kernel. "
    "It does not authenticate completeness of an external simulator run and cannot "
    "promote any physical-emergence claim."
)


class KernelKind(str, Enum):
    """Supported exact presentation-kernel conventions."""

    PROBABILITY = "PROBABILITY"
    RATE = "RATE"


@dataclass(frozen=True)
class FiberClassification(Generic[StateT, BoundaryT]):
    boundary_value: BoundaryT
    status: FiberStatus
    matching_state_count: int
    consistent_state_count: int
    witnesses: tuple[StateT, ...]
    enumeration_complete: bool
    exact_finite_classification: bool
    physical_promotion: bool = False
    claim_boundary: str = FINITE_MODEL_CLAIM_BOUNDARY


@dataclass(frozen=True)
class AbelianEdge(Generic[VertexT]):
    """Oriented graph edge carrying an additive abelian interface value.

    ``value`` is interpreted as ``potential[target] - potential[source]``.
    """

    edge_id: str
    source: VertexT
    target: VertexT
    value: ExactInput


@dataclass(frozen=True)
class FundamentalCycle:
    chord_edge_id: str
    oriented_edge_terms: tuple[tuple[str, int], ...]
    raw_holonomy: Fraction
    holonomy: Fraction
    obstructed: bool


@dataclass(frozen=True)
class HolonomyAudit(Generic[VertexT]):
    vertex_count: int
    edge_count: int
    connected_component_count: int
    tree_edge_ids: tuple[str, ...]
    cycles: tuple[FundamentalCycle, ...]
    modulus: Fraction | None
    cycle_basis_complete: bool
    global_extension_exists: bool
    mathematical_obstruction_receipt: bool
    physical_promotion: bool = False
    claim_boundary: str = FINITE_MODEL_CLAIM_BOUNDARY


@dataclass(frozen=True)
class LumpabilityDefect(Generic[StateT, QuotientT]):
    source_quotient: QuotientT
    left_representative: StateT
    right_representative: StateT
    left_aggregates: tuple[tuple[QuotientT, Fraction], ...]
    right_aggregates: tuple[tuple[QuotientT, Fraction], ...]


@dataclass(frozen=True)
class LumpabilityAudit(Generic[StateT, QuotientT]):
    kernel_kind: KernelKind
    presentation_state_count: int
    quotient_state_count: int
    nontrivial_presentation_fiber_count: int
    aggregate_equality: bool
    metadata_firewall_passed: bool
    forbidden_dependency_fields: tuple[str, ...]
    defects: tuple[LumpabilityDefect[StateT, QuotientT], ...]
    quotient_lumpable: bool
    mathematical_lumpability_receipt: bool
    physical_promotion: bool = False
    claim_boundary: str = FINITE_MODEL_CLAIM_BOUNDARY


@dataclass(frozen=True)
class RepairInitialAudit(Generic[StateT, QuotientT]):
    initial_state: StateT
    terminal_paths: tuple[tuple[StateT, ...], ...]
    cycle_witness_paths: tuple[tuple[StateT, ...], ...]
    terminal_states: tuple[StateT, ...]
    terminal_quotients: tuple[QuotientT, ...]
    normal_form_status: FiberStatus


@dataclass(frozen=True)
class RepairPathAudit(Generic[StateT, QuotientT]):
    state_count: int
    transition_count: int
    initial_state_count: int
    initial_state_coverage_complete: bool
    state_space_complete: bool
    repair_relation_complete: bool
    exact_exhaustion: bool
    graph_terminating: bool
    termination_certified: bool
    presentation_schedule_independent: bool
    quotient_schedule_independent: bool
    quotient_confluent: bool
    ambiguous_initial_states: tuple[StateT, ...]
    nonterminating_initial_states: tuple[StateT, ...]
    forbidden_dependency_fields: tuple[str, ...]
    initial_audits: tuple[RepairInitialAudit[StateT, QuotientT], ...]
    mathematical_repair_receipt: bool
    physical_promotion: bool = False
    claim_boundary: str = FINITE_MODEL_CLAIM_BOUNDARY


def classify_boundary_fiber(
    states: Sequence[StateT],
    *,
    boundary_value: BoundaryT,
    boundary_map: Callable[[StateT], BoundaryT],
    is_consistent: Callable[[StateT], bool],
    enumeration_complete: bool = True,
) -> FiberClassification[StateT, BoundaryT]:
    """Classify one boundary fiber in an explicitly enumerated finite model.

    If the supplied list is known to be incomplete, the result is ``UNKNOWN``
    regardless of how many candidate extensions happened to be observed.
    """

    finite_states = _distinct_hashable_sequence(states, label="states")
    matching = tuple(
        state for state in finite_states if boundary_map(state) == boundary_value
    )
    witnesses = tuple(state for state in matching if bool(is_consistent(state)))

    if not enumeration_complete:
        status = FiberStatus.UNKNOWN
    elif not witnesses:
        status = FiberStatus.UNREALIZABLE
    elif len(witnesses) == 1:
        status = FiberStatus.UNIQUE
    else:
        status = FiberStatus.AMBIGUOUS

    return FiberClassification(
        boundary_value=boundary_value,
        status=status,
        matching_state_count=len(matching),
        consistent_state_count=len(witnesses),
        witnesses=witnesses,
        enumeration_complete=bool(enumeration_complete),
        exact_finite_classification=bool(enumeration_complete),
    )


def audit_abelian_cycle_holonomy(
    vertices: Sequence[VertexT],
    edges: Sequence[AbelianEdge[VertexT]],
    *,
    modulus: ExactInput | None = None,
) -> HolonomyAudit[VertexT]:
    """Audit all fundamental-cycle holonomies with exact arithmetic.

    Edges are processed in supplied order to build a deterministic spanning
    forest.  Every non-tree edge closes one fundamental cycle.  Vanishing on
    this basis is equivalent to vanishing on every cycle for an additive
    abelian graph connection.
    """

    finite_vertices = _distinct_hashable_sequence(vertices, label="vertices")
    if not finite_vertices:
        raise ValueError("vertices must be nonempty for a nonvacuous holonomy audit")
    vertex_set = set(finite_vertices)
    finite_edges = tuple(edges)
    edge_ids: set[str] = set()
    exact_values: list[Fraction] = []
    for edge in finite_edges:
        if not isinstance(edge, AbelianEdge):
            raise TypeError("edges must contain AbelianEdge instances")
        if not isinstance(edge.edge_id, str) or not edge.edge_id:
            raise ValueError("every edge_id must be a nonempty string")
        if edge.edge_id in edge_ids:
            raise ValueError(f"duplicate edge_id: {edge.edge_id!r}")
        edge_ids.add(edge.edge_id)
        if edge.source not in vertex_set or edge.target not in vertex_set:
            raise ValueError(f"edge {edge.edge_id!r} has an endpoint outside vertices")
        exact_values.append(
            _exact_fraction(edge.value, label=f"edge {edge.edge_id!r} value")
        )

    exact_modulus = (
        None if modulus is None else _exact_fraction(modulus, label="modulus")
    )
    if exact_modulus is not None and exact_modulus <= 0:
        raise ValueError("modulus must be strictly positive")

    parents = {vertex: vertex for vertex in finite_vertices}
    ranks = {vertex: 0 for vertex in finite_vertices}
    adjacency: dict[VertexT, list[tuple[VertexT, int, int]]] = {
        vertex: [] for vertex in finite_vertices
    }
    tree_indices: list[int] = []
    chord_indices: list[int] = []

    def find(vertex: VertexT) -> VertexT:
        root = vertex
        while parents[root] != root:
            root = parents[root]
        while parents[vertex] != vertex:
            next_vertex = parents[vertex]
            parents[vertex] = root
            vertex = next_vertex
        return root

    def union(left: VertexT, right: VertexT) -> bool:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return False
        if ranks[left_root] < ranks[right_root]:
            left_root, right_root = right_root, left_root
        parents[right_root] = left_root
        if ranks[left_root] == ranks[right_root]:
            ranks[left_root] += 1
        return True

    for index, edge in enumerate(finite_edges):
        if union(edge.source, edge.target):
            tree_indices.append(index)
            adjacency[edge.source].append((edge.target, index, +1))
            adjacency[edge.target].append((edge.source, index, -1))
        else:
            chord_indices.append(index)

    cycles: list[FundamentalCycle] = []
    for chord_index in chord_indices:
        chord = finite_edges[chord_index]
        path_terms = _tree_path_terms(
            start=chord.target,
            goal=chord.source,
            adjacency=adjacency,
        )
        oriented_terms = ((chord.edge_id, +1),) + tuple(
            (finite_edges[edge_index].edge_id, sign) for edge_index, sign in path_terms
        )
        raw_holonomy = exact_values[chord_index] + sum(
            (sign * exact_values[edge_index] for edge_index, sign in path_terms),
            start=Fraction(0),
        )
        holonomy = (
            raw_holonomy if exact_modulus is None else raw_holonomy % exact_modulus
        )
        cycles.append(
            FundamentalCycle(
                chord_edge_id=chord.edge_id,
                oriented_edge_terms=oriented_terms,
                raw_holonomy=raw_holonomy,
                holonomy=holonomy,
                obstructed=holonomy != 0,
            )
        )

    component_count = len({find(vertex) for vertex in finite_vertices})
    extension_exists = all(not cycle.obstructed for cycle in cycles)
    return HolonomyAudit(
        vertex_count=len(finite_vertices),
        edge_count=len(finite_edges),
        connected_component_count=component_count,
        tree_edge_ids=tuple(finite_edges[index].edge_id for index in tree_indices),
        cycles=tuple(cycles),
        modulus=exact_modulus,
        cycle_basis_complete=True,
        global_extension_exists=extension_exists,
        mathematical_obstruction_receipt=extension_exists,
    )


def verify_quotient_lumpability(
    presentation_states: Sequence[StateT],
    presentation_kernel: Mapping[StateT, Mapping[StateT, ExactInput]],
    *,
    quotient_map: Callable[[StateT], QuotientT],
    kernel_kind: KernelKind | str = KernelKind.PROBABILITY,
    kernel_dependency_fields: Iterable[str] = (),
) -> LumpabilityAudit[StateT, QuotientT]:
    """Verify strong quotient lumpability by exact representative comparison.

    A declared dependency on worker or scheduler presentation metadata blocks
    the receipt even if one sampled kernel happens to have equal aggregates.
    """

    states = _distinct_hashable_sequence(
        presentation_states, label="presentation_states"
    )
    if not states:
        raise ValueError("presentation_states must be nonempty")
    state_set = set(states)
    if set(presentation_kernel) != state_set:
        missing = state_set - set(presentation_kernel)
        extra = set(presentation_kernel) - state_set
        raise ValueError(
            f"kernel rows must match states exactly; missing={missing}, extra={extra}"
        )

    kind = KernelKind(kernel_kind)
    quotient_by_state = {state: quotient_map(state) for state in states}
    for quotient in quotient_by_state.values():
        _require_hashable(quotient, label="quotient state")
    quotient_order = tuple(dict.fromkeys(quotient_by_state.values()))

    exact_kernel: dict[StateT, dict[StateT, Fraction]] = {}
    for source in states:
        row = presentation_kernel[source]
        if not isinstance(row, Mapping):
            raise TypeError("each kernel row must be a mapping")
        if not set(row).issubset(state_set):
            raise ValueError(
                f"kernel row for {source!r} targets a state outside the enumeration"
            )
        exact_row: dict[StateT, Fraction] = {}
        for target, value in row.items():
            weight = _exact_fraction(value, label="kernel weight")
            if weight < 0:
                raise ValueError("kernel weights/rates must be nonnegative")
            exact_row[target] = weight
        if (
            kind is KernelKind.PROBABILITY
            and sum(exact_row.values(), start=Fraction(0)) != 1
        ):
            raise ValueError(
                f"probability row for {source!r} does not sum exactly to one"
            )
        exact_kernel[source] = exact_row

    aggregates: dict[StateT, tuple[tuple[QuotientT, Fraction], ...]] = {}
    for source in states:
        row_totals = {quotient: Fraction(0) for quotient in quotient_order}
        for target, weight in exact_kernel[source].items():
            row_totals[quotient_by_state[target]] += weight
        aggregates[source] = tuple(
            (quotient, row_totals[quotient]) for quotient in quotient_order
        )

    representatives: dict[QuotientT, list[StateT]] = {
        quotient: [] for quotient in quotient_order
    }
    for state in states:
        representatives[quotient_by_state[state]].append(state)

    defects: list[LumpabilityDefect[StateT, QuotientT]] = []
    for source_quotient, fiber in representatives.items():
        if len(fiber) < 2:
            continue
        reference = fiber[0]
        for representative in fiber[1:]:
            if aggregates[representative] != aggregates[reference]:
                defects.append(
                    LumpabilityDefect(
                        source_quotient=source_quotient,
                        left_representative=reference,
                        right_representative=representative,
                        left_aggregates=aggregates[reference],
                        right_aggregates=aggregates[representative],
                    )
                )

    dependency_fields = tuple(
        sorted({str(field) for field in kernel_dependency_fields})
    )
    forbidden_fields = tuple(
        field
        for field in dependency_fields
        if _is_forbidden_presentation_dependency(field)
    )
    aggregate_equality = not defects
    metadata_firewall_passed = not forbidden_fields
    quotient_lumpable = aggregate_equality and metadata_firewall_passed
    return LumpabilityAudit(
        kernel_kind=kind,
        presentation_state_count=len(states),
        quotient_state_count=len(quotient_order),
        nontrivial_presentation_fiber_count=sum(
            len(fiber) > 1 for fiber in representatives.values()
        ),
        aggregate_equality=aggregate_equality,
        metadata_firewall_passed=metadata_firewall_passed,
        forbidden_dependency_fields=forbidden_fields,
        defects=tuple(defects),
        quotient_lumpable=quotient_lumpable,
        mathematical_lumpability_receipt=quotient_lumpable,
    )


def exhaust_accepted_repair_paths(
    states: Sequence[StateT],
    accepted_repairs: Mapping[StateT, Iterable[StateT]],
    *,
    quotient_map: Callable[[StateT], QuotientT],
    initial_states: Sequence[StateT] | None = None,
    state_space_complete: bool = True,
    repair_relation_complete: bool = True,
    transition_dependency_fields: Iterable[str] = (),
    max_paths: int = 100_000,
) -> RepairPathAudit[StateT, QuotientT]:
    """Exhaust all simple accepted-repair paths in a tiny finite system.

    A repeated state is recorded as a nontermination witness.  For finite
    systems, absence of such a reachable cycle certifies termination.  Under
    termination, a unique terminal quotient for every requested initial state
    certifies schedule-independent quotient normal forms on this finite model.
    """

    finite_states = _distinct_hashable_sequence(states, label="states")
    if not finite_states:
        raise ValueError("states must be nonempty")
    state_set = set(finite_states)
    if set(accepted_repairs) != state_set:
        missing = state_set - set(accepted_repairs)
        extra = set(accepted_repairs) - state_set
        raise ValueError(
            f"accepted-repair rows must match states exactly; missing={missing}, extra={extra}"
        )
    if isinstance(max_paths, bool) or not isinstance(max_paths, int) or max_paths <= 0:
        raise ValueError("max_paths must be a positive integer")

    successors: dict[StateT, tuple[StateT, ...]] = {}
    for source in finite_states:
        row = _distinct_hashable_sequence(
            tuple(accepted_repairs[source]),
            label=f"accepted successors of {source!r}",
        )
        unknown = set(row) - state_set
        if unknown:
            raise ValueError(
                f"accepted repair from {source!r} leaves the state space: {unknown}"
            )
        successors[source] = row

    initials = (
        finite_states
        if initial_states is None
        else _distinct_hashable_sequence(initial_states, label="initial_states")
    )
    unknown_initials = set(initials) - state_set
    if not initials:
        raise ValueError("initial_states must be nonempty")
    if unknown_initials:
        raise ValueError(
            f"initial states are outside the state space: {unknown_initials}"
        )
    initial_coverage_complete = set(initials) == state_set

    quotient_by_state = {state: quotient_map(state) for state in finite_states}
    for quotient in quotient_by_state.values():
        _require_hashable(quotient, label="quotient state")

    explored_path_count = 0
    initial_audits: list[RepairInitialAudit[StateT, QuotientT]] = []
    for initial in initials:
        terminal_paths: list[tuple[StateT, ...]] = []
        cycle_paths: list[tuple[StateT, ...]] = []

        def visit(current: StateT, path: tuple[StateT, ...]) -> None:
            nonlocal explored_path_count
            next_states = successors[current]
            if not next_states:
                explored_path_count += 1
                if explored_path_count > max_paths:
                    raise RuntimeError(
                        "accepted-repair path budget exceeded; no partial receipt emitted"
                    )
                terminal_paths.append(path)
                return
            for target in next_states:
                explored_path_count += 1
                if explored_path_count > max_paths:
                    raise RuntimeError(
                        "accepted-repair path budget exceeded; no partial receipt emitted"
                    )
                next_path = path + (target,)
                if target in path:
                    cycle_paths.append(next_path)
                else:
                    visit(target, next_path)

        visit(initial, (initial,))
        terminal_states = tuple(dict.fromkeys(path[-1] for path in terminal_paths))
        terminal_quotients = tuple(
            dict.fromkeys(quotient_by_state[state] for state in terminal_states)
        )
        complete = bool(state_space_complete and repair_relation_complete)
        if not complete or cycle_paths or not terminal_quotients:
            normal_form_status = FiberStatus.UNKNOWN
        elif len(terminal_quotients) == 1:
            normal_form_status = FiberStatus.UNIQUE
        else:
            normal_form_status = FiberStatus.AMBIGUOUS
        initial_audits.append(
            RepairInitialAudit(
                initial_state=initial,
                terminal_paths=tuple(terminal_paths),
                cycle_witness_paths=tuple(cycle_paths),
                terminal_states=terminal_states,
                terminal_quotients=terminal_quotients,
                normal_form_status=normal_form_status,
            )
        )

    complete = bool(state_space_complete and repair_relation_complete)
    graph_terminating = all(not audit.cycle_witness_paths for audit in initial_audits)
    termination_certified = complete and graph_terminating
    presentation_independent = termination_certified and all(
        len(audit.terminal_states) == 1 for audit in initial_audits
    )
    quotient_endpoint_unique = termination_certified and all(
        len(audit.terminal_quotients) == 1 for audit in initial_audits
    )
    dependency_fields = tuple(
        sorted({str(field) for field in transition_dependency_fields})
    )
    forbidden_fields = tuple(
        field
        for field in dependency_fields
        if _is_forbidden_presentation_dependency(field)
    )
    quotient_independent = quotient_endpoint_unique and not forbidden_fields
    ambiguous_initials = tuple(
        audit.initial_state
        for audit in initial_audits
        if len(audit.terminal_quotients) > 1
    )
    nonterminating_initials = tuple(
        audit.initial_state for audit in initial_audits if audit.cycle_witness_paths
    )

    return RepairPathAudit(
        state_count=len(finite_states),
        transition_count=sum(len(row) for row in successors.values()),
        initial_state_count=len(initials),
        initial_state_coverage_complete=initial_coverage_complete,
        state_space_complete=bool(state_space_complete),
        repair_relation_complete=bool(repair_relation_complete),
        exact_exhaustion=complete,
        graph_terminating=graph_terminating,
        termination_certified=termination_certified,
        presentation_schedule_independent=presentation_independent,
        quotient_schedule_independent=quotient_independent,
        quotient_confluent=quotient_independent,
        ambiguous_initial_states=ambiguous_initials,
        nonterminating_initial_states=nonterminating_initials,
        forbidden_dependency_fields=forbidden_fields,
        initial_audits=tuple(initial_audits),
        mathematical_repair_receipt=bool(
            initial_coverage_complete
            and termination_certified
            and quotient_independent
        ),
    )


def _tree_path_terms(
    *,
    start: VertexT,
    goal: VertexT,
    adjacency: Mapping[VertexT, Sequence[tuple[VertexT, int, int]]],
) -> tuple[tuple[int, int], ...]:
    if start == goal:
        return ()
    queue: deque[VertexT] = deque([start])
    predecessor: dict[VertexT, tuple[VertexT, int, int] | None] = {start: None}
    while queue:
        current = queue.popleft()
        for neighbor, edge_index, sign in adjacency[current]:
            if neighbor in predecessor:
                continue
            predecessor[neighbor] = (current, edge_index, sign)
            if neighbor == goal:
                queue.clear()
                break
            queue.append(neighbor)
    if goal not in predecessor:
        raise RuntimeError(
            "fundamental-cycle chord endpoints are not joined by the spanning forest"
        )

    reversed_terms: list[tuple[int, int]] = []
    cursor = goal
    while cursor != start:
        entry = predecessor[cursor]
        if entry is None:
            raise RuntimeError("invalid spanning-forest predecessor chain")
        parent, edge_index, sign = entry
        reversed_terms.append((edge_index, sign))
        cursor = parent
    reversed_terms.reverse()
    return tuple(reversed_terms)


def _exact_fraction(value: ExactInput, *, label: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError(f"{label} must be an int or Fraction, never a float/bool")
    return Fraction(value)


def _distinct_hashable_sequence(
    values: Sequence[StateT],
    *,
    label: str,
) -> tuple[StateT, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a finite sequence")
    result = tuple(values)
    seen: set[StateT] = set()
    for value in result:
        _require_hashable(value, label=label)
        if value in seen:
            raise ValueError(f"{label} contains duplicate value {value!r}")
        seen.add(value)
    return result


def _require_hashable(value: object, *, label: str) -> None:
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(f"{label} values must be hashable") from exc


def _is_forbidden_presentation_dependency(field: str) -> bool:
    normalized = field.strip().lower().replace("-", "_")
    tokens = tuple(part for part in normalized.replace(".", "_").split("_") if part)
    return any(
        token
        in {
            "worker",
            "scheduler",
            "schedule",
            "queue",
            "process",
            "thread",
            "retry",
            "attempt",
            "executor",
            "runtime",
            "wall",
            "memory",
            "storage",
            "uuid",
            "pid",
            "rng",
        }
        for token in tokens
    )
