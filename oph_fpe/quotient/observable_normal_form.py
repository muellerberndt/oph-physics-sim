"""Exact finite checks for observation-determined normal forms.

The checks in this module are executable clients of the finite theorem kernels
proved in ``ObservableNormalForms``.  They deliberately certify only a closed,
explicitly enumerated table supplied by the caller:

* ``verify_observation_determined_normal_form`` checks the load-bearing
  hypotheses and both sides of the cross-source endpoint theorem; and
* ``recognize_conditional_resampling_kernel`` checks the exact R1--R3 matrix
  receipt for observation-fiber averaging.

Neither checker authenticates that a table exhausts an external simulator run,
that a kernel was produced independently of the recognition formula, or that
the finite objects are physical.  Consequently every returned report keeps
physical promotion, Standard-Model emergence, geometry, gravity, and scale
authorization false even when the finite theorem application succeeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Callable, Generic, Hashable, Iterable, Mapping, Sequence, TypeVar

from oph_fpe.ontology.firewall import classify_forbidden_field
from oph_fpe.quotient.obstruction import exhaust_accepted_repair_paths


StateT = TypeVar("StateT", bound=Hashable)
ObservationT = TypeVar("ObservationT", bound=Hashable)
QuotientT = TypeVar("QuotientT", bound=Hashable)

ExactInput = int | Fraction

OBSERVABLE_NORMAL_FORM_VERIFIER_VERSION = "oph-fpe-observable-normal-form-v1"
CONDITIONAL_RESAMPLING_VERIFIER_VERSION = (
    "oph-fpe-conditional-resampling-recognition-v1"
)
FINITE_THEOREM_CLAIM_BOUNDARY = (
    "Conditional exact result for the explicitly declared finite table only. "
    "External enumeration completeness and producer independence are not "
    "authenticated, and no physical or emergence claim is promoted."
)

_MAX_STATES = 4_096
_MAX_TRANSITIONS = 100_000


@dataclass(frozen=True)
class ObservationLeak(Generic[StateT, ObservationT]):
    source: StateT
    target: StateT
    source_observation: ObservationT
    target_observation: ObservationT


@dataclass(frozen=True)
class BoundaryIdentificationDefect(Generic[StateT, ObservationT, QuotientT]):
    observation: ObservationT
    left_state: StateT
    right_state: StateT
    left_quotient: QuotientT
    right_quotient: QuotientT


@dataclass(frozen=True)
class CrossSourceEndpointDefect(Generic[StateT, ObservationT, QuotientT]):
    observation: ObservationT
    left_source: StateT
    right_source: StateT
    left_endpoint: StateT
    right_endpoint: StateT
    left_quotient: QuotientT
    right_quotient: QuotientT


@dataclass(frozen=True)
class ObservableNormalFormAudit(Generic[StateT, ObservationT, QuotientT]):
    state_count: int
    transition_count: int
    terminal_states: tuple[StateT, ...]
    consistent_states: tuple[StateT, ...]
    state_space_complete_declared: bool
    repair_relation_complete_declared: bool
    table_closed_under_declared_relation: bool
    external_completeness_authenticated: bool
    dependency_firewall_passed: bool
    forbidden_dependency_fields: tuple[str, ...]
    observation_preserving: bool
    observation_leaks: tuple[ObservationLeak[StateT, ObservationT], ...]
    normal_forms_equal_consistent_set: bool
    boundary_identifies_modulo: bool
    boundary_identification_defects: tuple[
        BoundaryIdentificationDefect[StateT, ObservationT, QuotientT], ...
    ]
    cross_source_endpoint_unique_modulo: bool
    cross_source_endpoint_defects: tuple[
        CrossSourceEndpointDefect[StateT, ObservationT, QuotientT], ...
    ]
    all_sources_weakly_normalizing: bool
    all_schedules_terminate: bool
    theorem_hypotheses_verified_on_declared_table: bool
    theorem_equivalence_replayed: bool
    conditional_mathematical_receipt: bool
    physical_promotion: bool = False
    scale_authorized: bool = False
    demo_assumption_accepted_as_evidence: bool = False
    verifier_version: str = OBSERVABLE_NORMAL_FORM_VERIFIER_VERSION
    theorem_reference: str = (
        "observable_normal_forms.tex:thm:cross-source-modulo; "
        "Lean boundaryIdentifiesModulo_iff_observerEndpointUniqueModulo"
    )
    claim_boundary: str = FINITE_THEOREM_CLAIM_BOUNDARY


@dataclass(frozen=True)
class KernelEntryDefect(Generic[StateT]):
    check: str
    left_state: StateT
    right_state: StateT
    actual: Fraction
    expected: Fraction


@dataclass(frozen=True)
class KernelRowDefect(Generic[StateT]):
    check: str
    left_state: StateT
    right_state: StateT
    differing_targets: tuple[StateT, ...]


@dataclass(frozen=True)
class ConditionalResamplingAudit(Generic[StateT, ObservationT]):
    state_count: int
    observation_fiber_count: int
    weights_strictly_positive: bool
    weights_normalized: bool
    kernel_nonnegative: bool
    kernel_row_stochastic: bool
    r1_fiber_supported: bool
    r2_fiber_rows_constant: bool
    r3_weighted_detailed_balance: bool
    explicit_formula_match: bool
    entry_defects: tuple[KernelEntryDefect[StateT], ...]
    row_defects: tuple[KernelRowDefect[StateT], ...]
    theorem_hypotheses_verified_on_declared_table: bool
    exact_table_recognition_receipt: bool
    external_kernel_provenance_authenticated: bool = False
    representative_selection_receipt: bool = False
    spectral_gap_receipt: bool = False
    convergence_rate_receipt: bool = False
    physical_promotion: bool = False
    scale_authorized: bool = False
    demo_assumption_accepted_as_evidence: bool = False
    verifier_version: str = CONDITIONAL_RESAMPLING_VERIFIER_VERSION
    theorem_reference: str = (
        "observable_normal_forms.tex:thm:fiber-conditional-expectation; "
        "Lean kernel_eq_conditionalResamplingKernel_iff_recognition"
    )
    claim_boundary: str = FINITE_THEOREM_CLAIM_BOUNDARY


def verify_observation_determined_normal_form(
    states: Sequence[StateT],
    accepted_repairs: Mapping[StateT, Iterable[StateT]],
    *,
    observation_map: Callable[[StateT], ObservationT],
    is_consistent: Callable[[StateT], bool],
    quotient_map: Callable[[StateT], QuotientT],
    state_space_complete: bool = False,
    repair_relation_complete: bool = False,
    dependency_fields: Iterable[str] = (),
    max_paths: int = 100_000,
) -> ObservableNormalFormAudit[StateT, ObservationT, QuotientT]:
    """Replay the finite cross-source observable-normal-form certificate.

    The completeness flags are assumptions about the declared universe rather
    than evidence about an external run.  They therefore default to ``False``
    and are surfaced verbatim in the result.  Even a successful conditional
    receipt cannot be registered as physical evidence by this module.
    """

    finite_states = _distinct_states(states)
    if not finite_states:
        raise ValueError("states must be nonempty")
    if len(finite_states) > _MAX_STATES:
        raise ValueError(f"states exceeds the finite verifier limit {_MAX_STATES}")
    if not isinstance(state_space_complete, bool):
        raise TypeError("state_space_complete must be bool")
    if not isinstance(repair_relation_complete, bool):
        raise TypeError("repair_relation_complete must be bool")

    state_set = set(finite_states)
    if set(accepted_repairs) != state_set:
        missing = state_set - set(accepted_repairs)
        extra = set(accepted_repairs) - state_set
        raise ValueError(
            "accepted-repair rows must match states exactly; "
            f"missing={missing}, extra={extra}"
        )
    successors: dict[StateT, tuple[StateT, ...]] = {}
    transition_count = 0
    for source in finite_states:
        row = _distinct_iterable(accepted_repairs[source], label="repair row")
        outside = set(row) - state_set
        if outside:
            raise ValueError(
                f"accepted repair from {source!r} leaves declared states: {outside}"
            )
        successors[source] = row
        transition_count += len(row)
    if transition_count > _MAX_TRANSITIONS:
        raise ValueError(
            f"accepted repairs exceeds the verifier limit {_MAX_TRANSITIONS}"
        )

    observations = {state: observation_map(state) for state in finite_states}
    quotients = {state: quotient_map(state) for state in finite_states}
    for value in (*observations.values(), *quotients.values()):
        _require_hashable(value, label="observation/quotient")

    consistent_states = tuple(
        state for state in finite_states if bool(is_consistent(state))
    )
    terminal_states = tuple(state for state in finite_states if not successors[state])

    observation_leaks = tuple(
        ObservationLeak(
            source=source,
            target=target,
            source_observation=observations[source],
            target_observation=observations[target],
        )
        for source in finite_states
        for target in successors[source]
        if observations[source] != observations[target]
    )
    observation_preserving = not observation_leaks
    normal_forms_equal_consistent = set(terminal_states) == set(consistent_states)

    boundary_defects: list[
        BoundaryIdentificationDefect[StateT, ObservationT, QuotientT]
    ] = []
    for left, right in combinations(consistent_states, 2):
        if observations[left] != observations[right]:
            continue
        if quotients[left] != quotients[right]:
            boundary_defects.append(
                BoundaryIdentificationDefect(
                    observation=observations[left],
                    left_state=left,
                    right_state=right,
                    left_quotient=quotients[left],
                    right_quotient=quotients[right],
                )
            )
    boundary_identifies = not boundary_defects

    repair_audit = exhaust_accepted_repair_paths(
        finite_states,
        successors,
        quotient_map=quotient_map,
        initial_states=finite_states,
        state_space_complete=state_space_complete,
        repair_relation_complete=repair_relation_complete,
        transition_dependency_fields=(),
        max_paths=max_paths,
    )
    audit_by_source = {
        row.initial_state: row for row in repair_audit.initial_audits
    }
    cross_source_defects: list[
        CrossSourceEndpointDefect[StateT, ObservationT, QuotientT]
    ] = []
    for left_index, left_source in enumerate(finite_states):
        for right_source in finite_states[left_index:]:
            if observations[left_source] != observations[right_source]:
                continue
            left_endpoints = audit_by_source[left_source].terminal_states
            right_endpoints = audit_by_source[right_source].terminal_states
            for left_endpoint in left_endpoints:
                for right_endpoint in right_endpoints:
                    if quotients[left_endpoint] != quotients[right_endpoint]:
                        cross_source_defects.append(
                            CrossSourceEndpointDefect(
                                observation=observations[left_source],
                                left_source=left_source,
                                right_source=right_source,
                                left_endpoint=left_endpoint,
                                right_endpoint=right_endpoint,
                                left_quotient=quotients[left_endpoint],
                                right_quotient=quotients[right_endpoint],
                            )
                        )
    cross_source_unique = not cross_source_defects
    weakly_normalizing = all(
        bool(row.terminal_paths) for row in repair_audit.initial_audits
    )
    all_schedules_terminate = repair_audit.graph_terminating

    dependency_names = _dependency_names(dependency_fields)
    forbidden_dependencies = tuple(
        name for name in dependency_names if classify_forbidden_field(name) is not None
    )
    dependency_firewall = not forbidden_dependencies
    declared_complete = state_space_complete and repair_relation_complete
    theorem_hypotheses = bool(
        declared_complete
        and observation_preserving
        and normal_forms_equal_consistent
        and dependency_firewall
    )
    theorem_equivalence = bool(
        theorem_hypotheses and boundary_identifies == cross_source_unique
    )
    conditional_receipt = bool(
        theorem_equivalence
        and boundary_identifies
        and cross_source_unique
        and weakly_normalizing
        and all_schedules_terminate
    )

    return ObservableNormalFormAudit(
        state_count=len(finite_states),
        transition_count=transition_count,
        terminal_states=terminal_states,
        consistent_states=consistent_states,
        state_space_complete_declared=state_space_complete,
        repair_relation_complete_declared=repair_relation_complete,
        table_closed_under_declared_relation=True,
        external_completeness_authenticated=False,
        dependency_firewall_passed=dependency_firewall,
        forbidden_dependency_fields=forbidden_dependencies,
        observation_preserving=observation_preserving,
        observation_leaks=observation_leaks,
        normal_forms_equal_consistent_set=normal_forms_equal_consistent,
        boundary_identifies_modulo=boundary_identifies,
        boundary_identification_defects=tuple(boundary_defects),
        cross_source_endpoint_unique_modulo=cross_source_unique,
        cross_source_endpoint_defects=tuple(cross_source_defects),
        all_sources_weakly_normalizing=weakly_normalizing,
        all_schedules_terminate=all_schedules_terminate,
        theorem_hypotheses_verified_on_declared_table=theorem_hypotheses,
        theorem_equivalence_replayed=theorem_equivalence,
        conditional_mathematical_receipt=conditional_receipt,
    )


def recognize_conditional_resampling_kernel(
    states: Sequence[StateT],
    kernel: Mapping[StateT, Mapping[StateT, ExactInput]],
    *,
    weights: Mapping[StateT, ExactInput],
    observation_map: Callable[[StateT], ObservationT],
) -> ConditionalResamplingAudit[StateT, ObservationT]:
    """Recognize the exact finite fiber-resampling kernel using R1--R3.

    Floats are rejected.  The supplied kernel must be an exact stochastic
    matrix and the supplied strictly positive weights must sum to one.  The
    verifier independently evaluates R1, R2, R3, and the implied entry formula.
    It cannot determine whether the producer constructed ``kernel`` from that
    target formula, so producer independence remains false in the report.
    """

    finite_states = _distinct_states(states)
    if not finite_states:
        raise ValueError("states must be nonempty")
    if len(finite_states) > _MAX_STATES:
        raise ValueError(f"states exceeds the finite verifier limit {_MAX_STATES}")
    state_set = set(finite_states)
    if set(weights) != state_set:
        raise ValueError("weights keys must match states exactly")
    if set(kernel) != state_set:
        raise ValueError("kernel rows must match states exactly")

    exact_weights = {
        state: _exact_fraction(weights[state], label=f"weight[{state!r}]")
        for state in finite_states
    }
    if any(value <= 0 for value in exact_weights.values()):
        raise ValueError("every state weight must be strictly positive")
    weights_normalized = (
        sum(exact_weights.values(), start=Fraction(0)) == Fraction(1)
    )

    exact_kernel: dict[StateT, dict[StateT, Fraction]] = {}
    kernel_nonnegative = True
    kernel_stochastic = True
    for source in finite_states:
        raw_row = kernel[source]
        if not isinstance(raw_row, Mapping):
            raise TypeError("every kernel row must be a mapping")
        outside = set(raw_row) - state_set
        if outside:
            raise ValueError(
                f"kernel row for {source!r} leaves declared states: {outside}"
            )
        row = {target: Fraction(0) for target in finite_states}
        for target, value in raw_row.items():
            row[target] = _exact_fraction(
                value, label=f"kernel[{source!r}][{target!r}]"
            )
        if any(value < 0 for value in row.values()):
            kernel_nonnegative = False
        if sum(row.values(), start=Fraction(0)) != 1:
            kernel_stochastic = False
        exact_kernel[source] = row

    observations = {state: observation_map(state) for state in finite_states}
    for observation in observations.values():
        _require_hashable(observation, label="observation")
    fiber_values = tuple(dict.fromkeys(observations.values()))
    fibers = {
        observation: tuple(
            state
            for state in finite_states
            if observations[state] == observation
        )
        for observation in fiber_values
    }

    entry_defects: list[KernelEntryDefect[StateT]] = []
    row_defects: list[KernelRowDefect[StateT]] = []

    for source in finite_states:
        for target in finite_states:
            actual = exact_kernel[source][target]
            if observations[source] != observations[target] and actual != 0:
                entry_defects.append(
                    KernelEntryDefect(
                        check="R1_FIBER_SUPPORT",
                        left_state=source,
                        right_state=target,
                        actual=actual,
                        expected=Fraction(0),
                    )
                )
    r1 = not any(row.check == "R1_FIBER_SUPPORT" for row in entry_defects)

    for left, right in combinations(finite_states, 2):
        if observations[left] != observations[right]:
            continue
        differing = tuple(
            target
            for target in finite_states
            if exact_kernel[left][target] != exact_kernel[right][target]
        )
        if differing:
            row_defects.append(
                KernelRowDefect(
                    check="R2_FIBER_ROW_CONSTANT",
                    left_state=left,
                    right_state=right,
                    differing_targets=differing,
                )
            )
    r2 = not row_defects

    for left in finite_states:
        for right in finite_states:
            left_value = exact_weights[left] * exact_kernel[left][right]
            right_value = exact_weights[right] * exact_kernel[right][left]
            if left_value != right_value:
                entry_defects.append(
                    KernelEntryDefect(
                        check="R3_WEIGHTED_DETAILED_BALANCE",
                        left_state=left,
                        right_state=right,
                        actual=left_value,
                        expected=right_value,
                    )
                )
    r3 = not any(
        row.check == "R3_WEIGHTED_DETAILED_BALANCE" for row in entry_defects
    )

    for source in finite_states:
        fiber = fibers[observations[source]]
        fiber_mass = sum(
            (exact_weights[state] for state in fiber), start=Fraction(0)
        )
        for target in finite_states:
            expected = (
                exact_weights[target] / fiber_mass
                if observations[target] == observations[source]
                else Fraction(0)
            )
            actual = exact_kernel[source][target]
            if actual != expected:
                entry_defects.append(
                    KernelEntryDefect(
                        check="EXPLICIT_FIBER_FORMULA",
                        left_state=source,
                        right_state=target,
                        actual=actual,
                        expected=expected,
                    )
                )
    formula_match = not any(
        row.check == "EXPLICIT_FIBER_FORMULA" for row in entry_defects
    )
    hypotheses = bool(
        weights_normalized and kernel_nonnegative and kernel_stochastic
    )
    recognition = bool(hypotheses and r1 and r2 and r3 and formula_match)

    return ConditionalResamplingAudit(
        state_count=len(finite_states),
        observation_fiber_count=len(fibers),
        weights_strictly_positive=True,
        weights_normalized=weights_normalized,
        kernel_nonnegative=kernel_nonnegative,
        kernel_row_stochastic=kernel_stochastic,
        r1_fiber_supported=r1,
        r2_fiber_rows_constant=r2,
        r3_weighted_detailed_balance=r3,
        explicit_formula_match=formula_match,
        entry_defects=tuple(entry_defects),
        row_defects=tuple(row_defects),
        theorem_hypotheses_verified_on_declared_table=hypotheses,
        exact_table_recognition_receipt=recognition,
    )


def _exact_fraction(value: ExactInput, *, label: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError(f"{label} must be int or Fraction, never float/bool")
    return Fraction(value)


def _distinct_states(states: Sequence[StateT]) -> tuple[StateT, ...]:
    if isinstance(states, (str, bytes)) or not isinstance(states, Sequence):
        raise TypeError("states must be a finite sequence")
    result = tuple(states)
    seen: set[StateT] = set()
    for state in result:
        _require_hashable(state, label="state")
        if state in seen:
            raise ValueError(f"states contains duplicate {state!r}")
        seen.add(state)
    return result


def _distinct_iterable(values: Iterable[StateT], *, label: str) -> tuple[StateT, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must not be a string")
    result = tuple(values)
    seen: set[StateT] = set()
    for value in result:
        _require_hashable(value, label=label)
        if value in seen:
            raise ValueError(f"{label} contains duplicate {value!r}")
        seen.add(value)
    return result


def _require_hashable(value: object, *, label: str) -> None:
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(f"{label} values must be hashable") from exc


def _dependency_names(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("dependency_fields must be an iterable of field names")
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise TypeError("dependency field names must be nonempty strings")
        result.add(value.strip())
    return tuple(sorted(result))


__all__ = [
    "BoundaryIdentificationDefect",
    "ConditionalResamplingAudit",
    "CrossSourceEndpointDefect",
    "FINITE_THEOREM_CLAIM_BOUNDARY",
    "KernelEntryDefect",
    "KernelRowDefect",
    "ObservableNormalFormAudit",
    "ObservationLeak",
    "recognize_conditional_resampling_kernel",
    "verify_observation_determined_normal_form",
]
