"""Fail-closed proof-carrying transactional repair.

This module implements the finite transaction boundary recommended by the OPH
simulator contract.  It does not decide that a proposal is physically sourced;
it proves the narrower, but necessary, facts that a committed update was read
complete, fresh, collar local, protected/sector preserving, target free and
accepted under the transition-specific exact rule.

The implementation intentionally has no imports from geometry, H3/KMS, gauge,
cosmology, or campaign code.  Lower-level repair proposals therefore cannot
inspect downstream candidate labels or targets through this API.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from typing import Any, TypeAlias

RegisterRef: TypeAlias = str
RegisterVersion: TypeAlias = int
ExactInput: TypeAlias = int | Fraction | Decimal
State: TypeAlias = Mapping[RegisterRef, Any]

REPAIR_CONTRACT_VERSION = "repair-contract-r1"
REPAIR_VERIFIER_VERSION = "oph-fpe-transaction-verifier-v1"
REPAIR_REPLAY_ENVELOPE_SCHEMA = "oph.repair.replay-envelope.v1"
REPAIR_REPLAY_ENVELOPE_ARTIFACT_TYPE = "TRANSACTIONAL_REPAIR_REPLAY_ENVELOPE"
REPAIR_REPLAY_VERIFIER_VERSION = "oph-fpe-primitive-replay-verifier-v1"

_REPAIR_REPLAY_SCOPE = "finite primitive transactional-repair replay"
_REPAIR_REPLAY_NONCLAIMS = [
    "physical source provenance",
    "probability or vacuum selection",
    "H3 or event geometry",
    "KMS clock normalization",
    "continuum or Standard Model emergence",
]


class TransitionKind(str, Enum):
    """Semantically distinct transition laws.

    Only ``STRICT_REPAIR`` is subjected to strict lexicographic mismatch
    descent.  Other kinds have separate contracts and can never inherit that
    receipt merely because they happen to lower a diagnostic.
    """

    STRICT_REPAIR = "STRICT_REPAIR"
    REVERSIBLE_PROPAGATION = "REVERSIBLE_PROPAGATION"
    RECORD_COMMIT = "RECORD_COMMIT"
    CONTROLLED_EXPLORATION = "CONTROLLED_EXPLORATION"
    GAUGE_STUTTER = "GAUGE_STUTTER"
    ROLLBACK = "ROLLBACK"


# A compatibility alias for callers that used the draft contract's name.
TransactionKind = TransitionKind


class ProposalClass(str, Enum):
    """Declared source mechanism for a proposal."""

    EXACT_SPLICE = "EXACT_SPLICE"
    CONDITIONAL_EXPECTATION = "CONDITIONAL_EXPECTATION"
    PETZ_RECOVERY = "PETZ_RECOVERY"
    FAWZI_RENNER_RECOVERY = "FAWZI_RENNER_RECOVERY"
    CSP_COMPLETION = "CSP_COMPLETION"
    PHYSICAL_CARRIER_RESPONSE = "PHYSICAL_CARRIER_RESPONSE"
    DIAGNOSTIC_HEURISTIC = "DIAGNOSTIC_HEURISTIC"


def _exact(value: ExactInput, *, field_name: str) -> Fraction:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an exact number, not bool")
    if isinstance(value, Fraction):
        result = value
    elif isinstance(value, int):
        result = Fraction(value)
    elif isinstance(value, Decimal):
        result = Fraction(value)
    else:
        raise TypeError(
            f"{field_name} must be an exact number (int, Fraction, or Decimal); "
            f"received {type(value).__name__}"
        )
    if result < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return result


@dataclass(frozen=True)
class MismatchLedger:
    """An exact, typed mismatch vector with no hidden scalar weights.

    The strict order is deliberately explicit:

    ``record, sector, holonomy, overlap, local_constraint, auxiliaries``.

    Auxiliary names must agree before and after a transaction.  This prevents
    a proposal from manufacturing a new objective during acceptance.
    """

    overlap: ExactInput = Fraction(0)
    local_constraint: ExactInput = Fraction(0)
    sector: ExactInput = Fraction(0)
    record: ExactInput = Fraction(0)
    holonomy: ExactInput = Fraction(0)
    physical_auxiliary: (
        Mapping[str, ExactInput] | tuple[tuple[str, ExactInput], ...]
    ) = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name in ("overlap", "local_constraint", "sector", "record", "holonomy"):
            object.__setattr__(self, name, _exact(getattr(self, name), field_name=name))
        raw_aux = (
            self.physical_auxiliary.items()
            if isinstance(self.physical_auxiliary, Mapping)
            else self.physical_auxiliary
        )
        aux: list[tuple[str, Fraction]] = []
        seen: set[str] = set()
        for raw_name, raw_value in raw_aux:
            name = str(raw_name)
            if not name or name in seen:
                raise ValueError(
                    "physical auxiliary names must be unique and non-empty"
                )
            seen.add(name)
            aux.append(
                (name, _exact(raw_value, field_name=f"physical_auxiliary.{name}"))
            )
        object.__setattr__(self, "physical_auxiliary", tuple(sorted(aux)))

    @property
    def auxiliary_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.physical_auxiliary)

    @property
    def lexicographic_vector(self) -> tuple[Fraction, ...]:
        return (
            self.record,
            self.sector,
            self.holonomy,
            self.overlap,
            self.local_constraint,
            *(value for _, value in self.physical_auxiliary),
        )

    def strictly_descends_from(self, before: MismatchLedger) -> bool:
        if self.auxiliary_names != before.auxiliary_names:
            return False
        return self.lexicographic_vector < before.lexicographic_vector

    def strict_repair_descends_from(self, before: MismatchLedger) -> bool:
        """Apply the fail-closed strict-repair order from the repair contract.

        Record and sector violations must remain exactly zero, holonomy may not
        change, and the quotient-visible overlap/local/auxiliary vector must
        strictly descend.  Thus a reduction in a forbidden violation cannot be
        used to mask a worsening overlap score.
        """

        if self.auxiliary_names != before.auxiliary_names:
            return False
        if before.record != 0 or self.record != 0:
            return False
        if before.sector != 0 or self.sector != 0:
            return False
        if self.holonomy != before.holonomy:
            return False
        after_visible = (
            self.overlap,
            self.local_constraint,
            *(value for _, value in self.physical_auxiliary),
        )
        before_visible = (
            before.overlap,
            before.local_constraint,
            *(value for _, value in before.physical_auxiliary),
        )
        return after_visible < before_visible

    def as_dict(self) -> dict[str, Any]:
        return {
            "record": _fraction_json(self.record),
            "sector": _fraction_json(self.sector),
            "holonomy": _fraction_json(self.holonomy),
            "overlap": _fraction_json(self.overlap),
            "local_constraint": _fraction_json(self.local_constraint),
            "physical_auxiliary": {
                name: _fraction_json(value) for name, value in self.physical_auxiliary
            },
            "lexicographic_order": [
                "record",
                "sector",
                "holonomy",
                "overlap",
                "local_constraint",
                *self.auxiliary_names,
            ],
        }


@dataclass(frozen=True)
class _FrozenMap:
    items: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class _FrozenList:
    items: tuple[Any, ...]


@dataclass(frozen=True)
class _FrozenTuple:
    items: tuple[Any, ...]


@dataclass(frozen=True)
class _FrozenSet:
    items: tuple[Any, ...]


def _freeze(value: Any) -> Any:
    """Deep-freeze the JSON-like register values accepted by this verifier."""

    if value is None or isinstance(value, (str, bytes, bool, int, Fraction, Decimal)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("register values may not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        invalid = [key for key in value if not isinstance(key, str) or not key]
        if invalid:
            raise TypeError(
                "mapping keys must be non-empty strings; refusing lossy key coercion for "
                f"{invalid!r}"
            )
        return _FrozenMap(
            tuple(sorted((key, _freeze(item)) for key, item in value.items()))
        )
    if isinstance(value, list):
        return _FrozenList(tuple(_freeze(item) for item in value))
    if isinstance(value, tuple):
        return _FrozenTuple(tuple(_freeze(item) for item in value))
    if isinstance(value, (set, frozenset)):
        items = [_freeze(item) for item in value]
        return _FrozenSet(tuple(sorted(items, key=lambda item: _stable_hash(item))))
    raise TypeError(
        "transaction registers must contain immutable/JSON-like values; "
        f"unsupported {type(value).__name__}"
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, _FrozenMap):
        return {key: _thaw(item) for key, item in value.items}
    if isinstance(value, _FrozenList):
        return [_thaw(item) for item in value.items]
    if isinstance(value, _FrozenTuple):
        return tuple(_thaw(item) for item in value.items)
    if isinstance(value, _FrozenSet):
        return {_thaw(item) for item in value.items}
    return copy.deepcopy(value)


def _canonical(value: Any) -> Any:
    if isinstance(value, Fraction):
        return {"__fraction__": [value.numerator, value.denominator]}
    if isinstance(value, Decimal):
        return {"__decimal__": str(value)}
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if isinstance(value, _FrozenMap):
        return {"__map__": [[key, _canonical(item)] for key, item in value.items]}
    if isinstance(value, _FrozenList):
        return {"__list__": [_canonical(item) for item in value.items]}
    if isinstance(value, _FrozenTuple):
        return {"__tuple__": [_canonical(item) for item in value.items]}
    if isinstance(value, _FrozenSet):
        return {"__set__": [_canonical(item) for item in value.items]}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        invalid = [key for key in value if not isinstance(key, str) or not key]
        if invalid:
            raise TypeError(
                "canonical mapping keys must be non-empty strings; refusing lossy key coercion"
            )
        return {key: _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=lambda item: repr(item))
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"cannot canonicalize {type(value).__name__}")


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fraction_json(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def canonical_state_hash(state: State) -> str:
    _validate_register_mapping(state, name="state")
    return _stable_hash(
        _FrozenMap(tuple(sorted((key, _freeze(value)) for key, value in state.items())))
    )


@dataclass(frozen=True)
class RegisterSnapshot:
    """One immutable register value and its optimistic-concurrency version."""

    ref: RegisterRef
    frozen_value: Any
    version: RegisterVersion

    def __post_init__(self) -> None:
        if not isinstance(self.ref, str) or not self.ref:
            raise ValueError("register references must be non-empty strings")
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version < 0
        ):
            raise ValueError("register versions must be non-negative integers")
        object.__setattr__(self, "frozen_value", _freeze(_thaw(self.frozen_value)))

    @property
    def value(self) -> Any:
        return _thaw(self.frozen_value)

    @property
    def value_hash(self) -> str:
        return _stable_hash(self.frozen_value)


@dataclass(frozen=True)
class Snapshot:
    """A content-addressed immutable snapshot of a declared read set."""

    registers: tuple[RegisterSnapshot, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.registers, key=lambda item: item.ref))
        refs = tuple(item.ref for item in ordered)
        if len(refs) != len(set(refs)):
            raise ValueError("snapshot contains duplicate register references")
        object.__setattr__(self, "registers", ordered)

    @classmethod
    def capture(
        cls,
        state: State,
        versions: Mapping[RegisterRef, RegisterVersion],
        read_set: Iterable[RegisterRef],
    ) -> Snapshot:
        refs = _refs(read_set)
        missing_state = sorted(refs - set(state))
        missing_versions = sorted(refs - set(versions))
        if missing_state:
            raise KeyError(
                f"snapshot read set contains missing registers: {missing_state}"
            )
        if missing_versions:
            raise KeyError(
                f"snapshot version map is missing registers: {missing_versions}"
            )
        return cls(
            tuple(
                RegisterSnapshot(
                    ref=ref, frozen_value=_freeze(state[ref]), version=versions[ref]
                )
                for ref in sorted(refs)
            )
        )

    @property
    def read_set(self) -> frozenset[RegisterRef]:
        return frozenset(item.ref for item in self.registers)

    @property
    def snapshot_hash(self) -> str:
        return _stable_hash(
            [
                {
                    "ref": item.ref,
                    "value_hash": item.value_hash,
                    "version": item.version,
                }
                for item in self.registers
            ]
        )

    def value(self, ref: RegisterRef) -> Any:
        for item in self.registers:
            if item.ref == ref:
                return item.value
        raise KeyError(ref)

    def version(self, ref: RegisterRef) -> RegisterVersion:
        for item in self.registers:
            if item.ref == ref:
                return item.version
        raise KeyError(ref)


def _refs(values: Iterable[RegisterRef]) -> frozenset[RegisterRef]:
    raw = tuple(values)
    invalid = [value for value in raw if not isinstance(value, str) or not value]
    if invalid:
        raise TypeError(
            "register references must be non-empty strings; refusing lossy coercion for "
            f"{invalid!r}"
        )
    result = frozenset(raw)
    return result


def _validate_register_mapping(values: Mapping[Any, Any], *, name: str) -> None:
    invalid = [ref for ref in values if not isinstance(ref, str) or not ref]
    if invalid:
        raise TypeError(
            f"{name} register keys must be non-empty strings; refusing lossy coercion for "
            f"{invalid!r}"
        )


@dataclass(frozen=True)
class RepairCollar:
    """A repair domain whose semantic footprint is declared before proposal generation."""

    collar_id: str
    visible_read_set: frozenset[RegisterRef]
    writable_registers: frozenset[RegisterRef]
    protected_boundary: frozenset[RegisterRef] = field(default_factory=frozenset)
    sector_registers: frozenset[RegisterRef] = field(default_factory=frozenset)
    record_registers: frozenset[RegisterRef] = field(default_factory=frozenset)
    checkpoint_registers: frozenset[RegisterRef] = field(default_factory=frozenset)
    interior_registers: frozenset[RegisterRef] = field(default_factory=frozenset)
    carrier_ids: frozenset[str] = field(default_factory=frozenset)
    seam_ids: frozenset[str] = field(default_factory=frozenset)
    forbidden_target_fields: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not isinstance(self.collar_id, str) or not self.collar_id:
            raise TypeError("collar_id must be a non-empty string")
        for name in (
            "visible_read_set",
            "writable_registers",
            "protected_boundary",
            "sector_registers",
            "record_registers",
            "checkpoint_registers",
            "interior_registers",
        ):
            object.__setattr__(self, name, _refs(getattr(self, name)))
        object.__setattr__(self, "carrier_ids", _refs(self.carrier_ids))
        object.__setattr__(self, "seam_ids", _refs(self.seam_ids))
        forbidden = tuple(self.forbidden_target_fields)
        if any(not isinstance(name, str) or not name for name in forbidden):
            raise TypeError("forbidden target field names must be non-empty strings")
        object.__setattr__(
            self,
            "forbidden_target_fields",
            frozenset(_normalize_field(name) for name in forbidden),
        )

    @property
    def read_set(self) -> frozenset[RegisterRef]:
        return self.visible_read_set

    @property
    def write_set(self) -> frozenset[RegisterRef]:
        return self.writable_registers

    @property
    def protected_set(self) -> frozenset[RegisterRef]:
        return self.protected_boundary


@dataclass(frozen=True)
class _FrozenUpdates:
    items: tuple[tuple[RegisterRef, Any], ...]

    @classmethod
    def from_mapping(cls, values: Mapping[RegisterRef, Any] | None) -> _FrozenUpdates:
        if values is None:
            return cls(())
        _validate_register_mapping(values, name="update")
        return cls(
            tuple(sorted((ref, _freeze(value)) for ref, value in values.items()))
        )

    @property
    def refs(self) -> frozenset[RegisterRef]:
        return frozenset(ref for ref, _ in self.items)

    def as_dict(self) -> dict[RegisterRef, Any]:
        return {ref: _thaw(value) for ref, value in self.items}


@dataclass(frozen=True)
class RepairProposal:
    """Immutable proposal plus its observed dependency trace."""

    proposal_id: str
    transition_kind: TransitionKind
    proposal_class: ProposalClass
    collar: RepairCollar
    snapshot: Snapshot
    declared_read_set: frozenset[RegisterRef]
    observed_read_set: frozenset[RegisterRef]
    _updates: _FrozenUpdates
    _inverse_updates: _FrozenUpdates = field(default_factory=lambda: _FrozenUpdates(()))
    _source_parameters: Any = field(default_factory=lambda: _FrozenMap(()))
    parent_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.proposal_id, str) or not self.proposal_id:
            raise TypeError("proposal_id must be a non-empty string")
        object.__setattr__(
            self, "transition_kind", TransitionKind(self.transition_kind)
        )
        object.__setattr__(self, "proposal_class", ProposalClass(self.proposal_class))
        object.__setattr__(self, "declared_read_set", _refs(self.declared_read_set))
        object.__setattr__(self, "observed_read_set", _refs(self.observed_read_set))
        object.__setattr__(
            self, "parent_event_ids", tuple(sorted(_refs(self.parent_event_ids)))
        )
        object.__setattr__(
            self, "_source_parameters", _freeze(_thaw(self._source_parameters))
        )

    @property
    def updates(self) -> dict[RegisterRef, Any]:
        return self._updates.as_dict()

    @property
    def inverse_updates(self) -> dict[RegisterRef, Any]:
        return self._inverse_updates.as_dict()

    @property
    def source_parameters(self) -> Any:
        return _thaw(self._source_parameters)

    @property
    def write_set(self) -> frozenset[RegisterRef]:
        return self._updates.refs


class _ReadTrackingState(Mapping[RegisterRef, Any]):
    """A read-only state view that conservatively records accessed registers."""

    def __init__(self, state: State) -> None:
        _validate_register_mapping(state, name="tracked state")
        self._state = {ref: copy.deepcopy(value) for ref, value in state.items()}
        self.observed: set[RegisterRef] = set()

    def __getitem__(self, ref: RegisterRef) -> Any:
        if not isinstance(ref, str) or not ref:
            raise TypeError("register references must be non-empty strings")
        key = ref
        self.observed.add(key)
        return copy.deepcopy(self._state[key])

    def __iter__(self) -> Iterator[RegisterRef]:
        # Enumerating register names is itself a dependency on the full state.
        self.observed.update(self._state)
        return iter(self._state)

    def __len__(self) -> int:
        self.observed.update(self._state)
        return len(self._state)


ProposalBuilder: TypeAlias = Callable[
    [Mapping[RegisterRef, Any]], Mapping[RegisterRef, Any]
]
MismatchEvaluator: TypeAlias = Callable[[State, RepairCollar], MismatchLedger]


def prepare_proposal(
    state: State,
    versions: Mapping[RegisterRef, RegisterVersion],
    *,
    proposal_id: str,
    transition_kind: TransitionKind,
    proposal_class: ProposalClass,
    collar: RepairCollar,
    declared_read_set: Iterable[RegisterRef],
    recovery: ProposalBuilder,
    inverse_updates: Mapping[RegisterRef, Any] | None = None,
    source_parameters: Mapping[str, Any] | None = None,
    parent_event_ids: Iterable[str] = (),
) -> RepairProposal:
    """Run a recovery function against a tracked view and freeze its proposal.

    The builder may read an undeclared register, but that dependency is captured
    and makes validation fail.  It can therefore never be silently committed.
    """

    reads = _refs(declared_read_set)
    tracker = _ReadTrackingState(state)
    raw_updates = recovery(tracker)
    if not isinstance(raw_updates, Mapping):
        raise TypeError("recovery must return a register-to-value mapping")
    snapshot = Snapshot.capture(state, versions, reads)
    return RepairProposal(
        proposal_id=proposal_id,
        transition_kind=TransitionKind(transition_kind),
        proposal_class=ProposalClass(proposal_class),
        collar=collar,
        snapshot=snapshot,
        declared_read_set=reads,
        observed_read_set=frozenset(tracker.observed),
        _updates=_FrozenUpdates.from_mapping(raw_updates),
        _inverse_updates=_FrozenUpdates.from_mapping(inverse_updates),
        _source_parameters=_freeze(dict(source_parameters or {})),
        parent_event_ids=tuple(_refs(parent_event_ids)),
    )


def proposals_conflict(left: RepairProposal, right: RepairProposal) -> bool:
    """Return the exact read/write-footprint conflict relation."""

    return bool(
        left.write_set & (right.declared_read_set | right.write_set)
        or right.write_set & (left.declared_read_set | left.write_set)
    )


def conflict_components(
    proposals: Sequence[RepairProposal],
) -> list[tuple[RepairProposal, ...]]:
    """Build deterministic connected components of the proposal conflict graph."""

    ordered = sorted(proposals, key=lambda proposal: proposal.proposal_id)
    ids = [proposal.proposal_id for proposal in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("proposal IDs must be unique within a batch")
    remaining = set(range(len(ordered)))
    components: list[tuple[RepairProposal, ...]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        component: set[int] = set()
        while stack:
            index = stack.pop()
            component.add(index)
            neighbors = [
                candidate
                for candidate in sorted(remaining)
                if proposals_conflict(ordered[index], ordered[candidate])
            ]
            for candidate in neighbors:
                remaining.remove(candidate)
                stack.append(candidate)
        components.append(tuple(ordered[index] for index in sorted(component)))
    return sorted(components, key=lambda component: component[0].proposal_id)


@dataclass(frozen=True)
class VerificationCheck:
    name: str
    passed: bool
    detail: str


_TRANSACTION_CHECK_NAMES = frozenset(
    {
        "unique_proposal_ids",
        "complete_read_set",
        "snapshot_current",
        "write_locality",
        "target_free_source",
        "conflict_component_support",
        "single_transition_kind",
        "compatible_component_writes",
        "unlisted_registers_unchanged",
        "protected_boundary_preserved",
        "sector_preserved",
        "mismatch_read_trace_complete",
        "transition_contract",
        "atomic_union_revalidated",
    }
)
_ROLLBACK_CHECK_NAMES = frozenset({"rollback_root_exists", "rollback_snapshot_current"})


def _check_schema_complete(
    checks: Sequence[VerificationCheck], transition_kind: TransitionKind
) -> bool:
    expected = (
        _ROLLBACK_CHECK_NAMES
        if transition_kind is TransitionKind.ROLLBACK
        else _TRANSACTION_CHECK_NAMES
    )
    names = [check.name for check in checks]
    return len(names) == len(set(names)) and set(names) == expected


@dataclass(frozen=True)
class ProposalFootprint:
    """Primitive dependency evidence used to recompute receipt booleans."""

    proposal_id: str
    declared_reads: tuple[RegisterRef, ...]
    observed_reads: tuple[RegisterRef, ...]
    required_reads: tuple[RegisterRef, ...]
    snapshot_reads: tuple[RegisterRef, ...]
    writes: tuple[RegisterRef, ...]

    @property
    def read_complete(self) -> bool:
        declared = frozenset(self.declared_reads)
        return bool(
            frozenset(self.required_reads) <= declared
            and frozenset(self.observed_reads) <= declared
            and frozenset(self.snapshot_reads) == declared
        )


def _check_value(checks: Sequence[VerificationCheck], name: str) -> bool:
    matches = [check.passed for check in checks if check.name == name]
    return len(matches) == 1 and matches[0]


def _footprints_conflict(left: ProposalFootprint, right: ProposalFootprint) -> bool:
    left_writes = frozenset(left.writes)
    right_writes = frozenset(right.writes)
    left_reads = frozenset(left.declared_reads)
    right_reads = frozenset(right.declared_reads)
    return bool(
        left_writes & (right_reads | right_writes)
        or right_writes & (left_reads | left_writes)
    )


def _expected_conflict_edges(
    footprints: Sequence[ProposalFootprint],
) -> tuple[tuple[str, str], ...]:
    ordered = sorted(footprints, key=lambda item: item.proposal_id)
    return tuple(
        (left.proposal_id, right.proposal_id)
        for index, left in enumerate(ordered)
        for right in ordered[index + 1 :]
        if _footprints_conflict(left, right)
    )


def _footprint_conflict_receipt(
    footprints: Sequence[ProposalFootprint], edges: Sequence[tuple[str, str]]
) -> bool:
    ordered = tuple(sorted(footprints, key=lambda item: item.proposal_id))
    ids = tuple(item.proposal_id for item in ordered)
    if not ids or len(ids) != len(set(ids)):
        return False
    expected = _expected_conflict_edges(ordered)
    normalized_edges = tuple(sorted(tuple(sorted(edge)) for edge in edges))
    if normalized_edges != expected:
        return False
    if len(ids) == 1:
        return not normalized_edges
    adjacency = {proposal_id: set() for proposal_id in ids}
    for left, right in normalized_edges:
        if left not in adjacency or right not in adjacency:
            return False
        adjacency[left].add(right)
        adjacency[right].add(left)
    reached = {ids[0]}
    stack = [ids[0]]
    while stack:
        current = stack.pop()
        for neighbor in sorted(adjacency[current]):
            if neighbor not in reached:
                reached.add(neighbor)
                stack.append(neighbor)
    return reached == set(ids)


def _conflict_support_payload(
    footprints: Sequence[ProposalFootprint],
    score_observed_reads: Sequence[RegisterRef],
    score_snapshot_reads: Sequence[RegisterRef],
    edges: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    return {
        "proposal_footprints": [
            {
                "proposal_id": item.proposal_id,
                "declared_reads": item.declared_reads,
                "observed_reads": item.observed_reads,
                "required_reads": item.required_reads,
                "snapshot_reads": item.snapshot_reads,
                "writes": item.writes,
            }
            for item in footprints
        ],
        "score_observed_reads": tuple(score_observed_reads),
        "score_snapshot_reads": tuple(score_snapshot_reads),
        "conflict_edges": tuple(edges),
    }


def _commit_seed(
    *,
    proposal_ids: Sequence[str],
    transition_kind: TransitionKind,
    proposal_classes: Sequence[ProposalClass],
    checks: Sequence[VerificationCheck],
    committed: bool,
    pre_state_hash: str,
    post_state_hash: str,
    mismatch_before: MismatchLedger | None,
    mismatch_after: MismatchLedger | None,
    snapshot_hashes: Sequence[str],
    conflict_support_hash: str,
    read_set_hash: str,
    write_set_hash: str,
    protected_boundary_hash: str,
    versions_before_hash: str,
    versions_after_hash: str,
    parent_event_ids: Sequence[str],
    transition_event_id: str | None,
    semantic_record_event_id: str | None,
) -> dict[str, Any]:
    return {
        "contract": REPAIR_CONTRACT_VERSION,
        "verifier": REPAIR_VERIFIER_VERSION,
        "proposal_ids": tuple(proposal_ids),
        "transition_kind": transition_kind.value,
        "proposal_classes": [value.value for value in proposal_classes],
        "checks": [(item.name, item.passed, item.detail) for item in checks],
        "committed": committed,
        "pre": pre_state_hash,
        "post": post_state_hash,
        "mismatch_before": mismatch_before.as_dict() if mismatch_before else None,
        "mismatch_after": mismatch_after.as_dict() if mismatch_after else None,
        "snapshots": tuple(snapshot_hashes),
        "conflict_support_hash": conflict_support_hash,
        "read": read_set_hash,
        "write": write_set_hash,
        "protected": protected_boundary_hash,
        "versions_before": versions_before_hash,
        "versions_after": versions_after_hash,
        "parents": tuple(parent_event_ids),
        "transition_event": transition_event_id,
        "semantic_record_event": semantic_record_event_id,
    }


@dataclass(frozen=True)
class ProposalValidation:
    proposal_ids: tuple[str, ...]
    checks: tuple[VerificationCheck, ...]
    mismatch_before: MismatchLedger | None
    mismatch_after: MismatchLedger | None
    pre_state_hash: str
    candidate_state_hash: str

    @property
    def admissible(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        return tuple(check.detail for check in self.checks if not check.passed)


@dataclass(frozen=True)
class RepairCommitReceipt:
    """Proof-carrying receipt whose verdicts are derived from primitive checks."""

    proposal_ids: tuple[str, ...]
    transition_kind: TransitionKind
    proposal_classes: tuple[ProposalClass, ...]
    checks: tuple[VerificationCheck, ...]
    committed: bool
    pre_state_hash: str
    post_state_hash: str
    mismatch_before: MismatchLedger | None
    mismatch_after: MismatchLedger | None
    snapshot_hashes: tuple[str, ...]
    proposal_footprints: tuple[ProposalFootprint, ...]
    score_observed_reads: tuple[RegisterRef, ...]
    score_snapshot_reads: tuple[RegisterRef, ...]
    conflict_edges: tuple[tuple[str, str], ...]
    conflict_support_hash: str
    read_set_hash: str
    write_set_hash: str
    protected_boundary_hash: str
    versions_before_hash: str
    versions_after_hash: str
    rollback_root: str
    parent_event_ids: tuple[str, ...]
    transition_event_id: str | None
    semantic_record_event_id: str | None
    commit_id: str
    contract_version: str = REPAIR_CONTRACT_VERSION
    verifier_version: str = REPAIR_VERIFIER_VERSION

    @property
    def all_checks_passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def diagnostic_only(self) -> bool:
        return ProposalClass.DIAGNOSTIC_HEURISTIC in self.proposal_classes

    @property
    def COMPLETE_READ_SET_RECEIPT(self) -> bool:  # noqa: N802 - stable artifact key
        declared = frozenset(
            ref
            for footprint in self.proposal_footprints
            for ref in footprint.declared_reads
        )
        return bool(
            self.proposal_footprints
            and all(footprint.read_complete for footprint in self.proposal_footprints)
            and frozenset(self.score_observed_reads) <= declared
            and frozenset(self.score_observed_reads)
            <= frozenset(self.score_snapshot_reads)
        )

    @property
    def CONFLICT_COMPONENT_SUPPORT_RECEIPT(self) -> bool:  # noqa: N802
        expected_hash = _stable_hash(
            _conflict_support_payload(
                self.proposal_footprints,
                self.score_observed_reads,
                self.score_snapshot_reads,
                self.conflict_edges,
            )
        )
        return bool(
            _footprint_conflict_receipt(self.proposal_footprints, self.conflict_edges)
            and expected_hash == self.conflict_support_hash
        )

    @property
    def ATOMIC_UNION_REVALIDATION_RECEIPT(self) -> bool:  # noqa: N802
        atomic_check = _check_value(self.checks, "atomic_union_revalidated")
        transition_check = _check_value(self.checks, "transition_contract")
        exact_descent = True
        if self.transition_kind is TransitionKind.STRICT_REPAIR:
            exact_descent = bool(
                self.mismatch_before is not None
                and self.mismatch_after is not None
                and self.mismatch_after.strict_repair_descends_from(
                    self.mismatch_before
                )
            )
        return bool(
            atomic_check
            and transition_check
            and exact_descent
            and _check_schema_complete(self.checks, self.transition_kind)
            and self.COMPLETE_READ_SET_RECEIPT
            and self.CONFLICT_COMPONENT_SUPPORT_RECEIPT
        )

    @property
    def semantic_record_written(self) -> bool:
        return bool(
            self.transition_kind is TransitionKind.RECORD_COMMIT
            and self.semantic_record_event_id is not None
        )

    @property
    def transition_event_emitted(self) -> bool:
        return self.transition_event_id is not None

    @property
    def physical_repair_receipt(self) -> bool:
        """Whether this is an admissible, non-heuristic strict repair.

        This proves only the finite repair transaction.  It does not by itself
        prove an H3, KMS, Standard Model, continuum, or ensemble claim.
        """

        return bool(
            self.committed
            and self.all_checks_passed
            and self.ATOMIC_UNION_REVALIDATION_RECEIPT
            and self.transition_kind is TransitionKind.STRICT_REPAIR
            and not self.diagnostic_only
        )

    @property
    def TRANSACTIONAL_REPAIR_RECEIPT(self) -> bool:  # noqa: N802
        return self.physical_repair_receipt

    @property
    def verdict(self) -> str:
        return (
            "VALID_PASS" if self.committed and self.all_checks_passed else "VALID_FAIL"
        )

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        return tuple(check.detail for check in self.checks if not check.passed)

    @property
    def claim_tier(self) -> str:
        if self.physical_repair_receipt:
            return "FINITE_TRANSACTIONAL_REPAIR"
        if self.committed and self.diagnostic_only:
            return "REGULATOR_DIAGNOSTIC"
        return "FINITE_THEOREM" if self.committed else "NO_CLAIM"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "oph_receipt_v2",
            "receipt_type": "TRANSACTIONAL_REPAIR",
            "contract_version": self.contract_version,
            "verifier_version": self.verifier_version,
            "proposal_ids": list(self.proposal_ids),
            "transition_kind": self.transition_kind.value,
            "proposal_classes": [value.value for value in self.proposal_classes],
            "checks": [
                {"name": check.name, "passed": check.passed, "detail": check.detail}
                for check in self.checks
            ],
            "committed": self.committed,
            "verdict": self.verdict,
            "physical_repair_receipt": self.physical_repair_receipt,
            "COMPLETE_READ_SET_RECEIPT": self.COMPLETE_READ_SET_RECEIPT,
            "CONFLICT_COMPONENT_SUPPORT_RECEIPT": self.CONFLICT_COMPONENT_SUPPORT_RECEIPT,
            "ATOMIC_UNION_REVALIDATION_RECEIPT": self.ATOMIC_UNION_REVALIDATION_RECEIPT,
            "TRANSACTIONAL_REPAIR_RECEIPT": self.TRANSACTIONAL_REPAIR_RECEIPT,
            "diagnostic_only": self.diagnostic_only,
            "claim_tier": self.claim_tier,
            "pre_state_hash": self.pre_state_hash,
            "post_state_hash": self.post_state_hash,
            "mismatch_before": (
                self.mismatch_before.as_dict()
                if self.mismatch_before is not None
                else None
            ),
            "mismatch_after": (
                self.mismatch_after.as_dict()
                if self.mismatch_after is not None
                else None
            ),
            "snapshot_hashes": list(self.snapshot_hashes),
            "proposal_footprints": [
                {
                    "proposal_id": item.proposal_id,
                    "declared_reads": list(item.declared_reads),
                    "observed_reads": list(item.observed_reads),
                    "required_reads": list(item.required_reads),
                    "snapshot_reads": list(item.snapshot_reads),
                    "writes": list(item.writes),
                }
                for item in self.proposal_footprints
            ],
            "score_observed_reads": list(self.score_observed_reads),
            "score_snapshot_reads": list(self.score_snapshot_reads),
            "conflict_edges": [list(edge) for edge in self.conflict_edges],
            "conflict_support_hash": self.conflict_support_hash,
            "read_set_hash": self.read_set_hash,
            "write_set_hash": self.write_set_hash,
            "protected_boundary_hash": self.protected_boundary_hash,
            "versions_before_hash": self.versions_before_hash,
            "versions_after_hash": self.versions_after_hash,
            "rollback_root": self.rollback_root,
            "parent_event_ids": list(self.parent_event_ids),
            "transition_event_id": self.transition_event_id,
            "transition_event_emitted": self.transition_event_emitted,
            "semantic_record_event_id": self.semantic_record_event_id,
            "semantic_record_written": self.semantic_record_written,
            "commit_id": self.commit_id,
            "scope": "finite quotient-visible transactional repair",
            "nonclaims": [
                "physical source provenance",
                "probability or vacuum selection",
                "H3 or event geometry",
                "KMS clock normalization",
                "continuum or Standard Model emergence",
            ],
        }


def _fraction_from_artifact(value: Any, *, field_name: str) -> Fraction:
    if not isinstance(value, Mapping) or set(value) != {"numerator", "denominator"}:
        raise TypeError(f"{field_name} must be an exact numerator/denominator object")
    numerator = value["numerator"]
    denominator = value["denominator"]
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        raise TypeError(f"{field_name} has invalid exact integer components")
    return Fraction(numerator, denominator)


def _ledger_from_artifact(value: Any, *, field_name: str) -> MismatchLedger | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object or null")
    required = {
        "record",
        "sector",
        "holonomy",
        "overlap",
        "local_constraint",
        "physical_auxiliary",
        "lexicographic_order",
    }
    if set(value) != required:
        raise ValueError(f"{field_name} has noncanonical fields")
    auxiliaries = value["physical_auxiliary"]
    if not isinstance(auxiliaries, Mapping):
        raise TypeError(f"{field_name}.physical_auxiliary must be an object")
    _validate_register_mapping(auxiliaries, name=f"{field_name}.physical_auxiliary")
    ledger = MismatchLedger(
        record=_fraction_from_artifact(
            value["record"], field_name=f"{field_name}.record"
        ),
        sector=_fraction_from_artifact(
            value["sector"], field_name=f"{field_name}.sector"
        ),
        holonomy=_fraction_from_artifact(
            value["holonomy"], field_name=f"{field_name}.holonomy"
        ),
        overlap=_fraction_from_artifact(
            value["overlap"], field_name=f"{field_name}.overlap"
        ),
        local_constraint=_fraction_from_artifact(
            value["local_constraint"], field_name=f"{field_name}.local_constraint"
        ),
        physical_auxiliary={
            name: _fraction_from_artifact(item, field_name=f"{field_name}.{name}")
            for name, item in auxiliaries.items()
        },
    )
    if ledger.as_dict() != dict(value):
        raise ValueError(f"{field_name} is not in canonical ledger form")
    return ledger


def _string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a JSON array")
    result = tuple(value)
    _refs(result)
    return result


_SERIALIZED_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "receipt_type",
        "contract_version",
        "verifier_version",
        "proposal_ids",
        "transition_kind",
        "proposal_classes",
        "checks",
        "committed",
        "verdict",
        "physical_repair_receipt",
        "COMPLETE_READ_SET_RECEIPT",
        "CONFLICT_COMPONENT_SUPPORT_RECEIPT",
        "ATOMIC_UNION_REVALIDATION_RECEIPT",
        "TRANSACTIONAL_REPAIR_RECEIPT",
        "diagnostic_only",
        "claim_tier",
        "pre_state_hash",
        "post_state_hash",
        "mismatch_before",
        "mismatch_after",
        "snapshot_hashes",
        "proposal_footprints",
        "score_observed_reads",
        "score_snapshot_reads",
        "conflict_edges",
        "conflict_support_hash",
        "read_set_hash",
        "write_set_hash",
        "protected_boundary_hash",
        "versions_before_hash",
        "versions_after_hash",
        "rollback_root",
        "parent_event_ids",
        "transition_event_id",
        "transition_event_emitted",
        "semantic_record_event_id",
        "semantic_record_written",
        "commit_id",
        "scope",
        "nonclaims",
    }
)


def _canonical_ref_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    result = _string_tuple(value, field_name=field_name)
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{field_name} must be sorted and duplicate-free")
    return result


def _verify_repair_receipt_self_consistency(
    artifact: RepairCommitReceipt | Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute invariants that are internal to a serialized receipt.

    The verifier deliberately ignores caller-authored receipt booleans until it
    has rebuilt dependency completeness, conflict edges/support, exact descent,
    event taxonomy, aggregate hashes, and the commit ID from primitive fields.

    Crucially, this function has no pre-state, version map, mismatch evaluator,
    collar, or executable proposal.  It can therefore detect isolated receipt
    corruption but cannot establish that the described transition happened.
    It is private so no promotion registry can accidentally admit it.
    """

    raw = artifact.as_dict() if isinstance(artifact, RepairCommitReceipt) else artifact
    failures: list[str] = []
    derived: RepairCommitReceipt | None = None
    try:
        if not isinstance(raw, Mapping):
            raise TypeError("repair receipt artifact must be an object")
        if set(raw) != _SERIALIZED_RECEIPT_FIELDS:
            raise ValueError(
                "serialized receipt has missing or unexpected top-level fields"
            )
        if raw.get("schema") != "oph_receipt_v2":
            raise ValueError("unexpected receipt schema")
        if raw.get("receipt_type") != "TRANSACTIONAL_REPAIR":
            raise ValueError("unexpected receipt type")
        if raw.get("contract_version") != REPAIR_CONTRACT_VERSION:
            raise ValueError("unexpected repair contract version")
        if raw.get("verifier_version") != REPAIR_VERIFIER_VERSION:
            raise ValueError("unexpected repair verifier version")

        proposal_ids = _string_tuple(raw.get("proposal_ids"), field_name="proposal_ids")
        if tuple(sorted(set(proposal_ids))) != proposal_ids:
            raise ValueError("proposal_ids must be sorted and unique")
        transition_kind = TransitionKind(raw.get("transition_kind"))
        raw_classes = raw.get("proposal_classes")
        if not isinstance(raw_classes, list):
            raise TypeError("proposal_classes must be an array")
        proposal_classes = tuple(ProposalClass(value) for value in raw_classes)
        if (
            tuple(sorted(set(proposal_classes), key=lambda value: value.value))
            != proposal_classes
        ):
            raise ValueError("proposal_classes must be sorted and unique")

        raw_checks = raw.get("checks")
        if not isinstance(raw_checks, list) or not raw_checks:
            raise TypeError("checks must be a non-empty array")
        checks: list[VerificationCheck] = []
        for row in raw_checks:
            if not isinstance(row, Mapping) or set(row) != {"name", "passed", "detail"}:
                raise ValueError("check rows must have exactly name/passed/detail")
            if not isinstance(row["name"], str) or not row["name"]:
                raise TypeError("check names must be non-empty strings")
            if not isinstance(row["passed"], bool) or not isinstance(
                row["detail"], str
            ):
                raise TypeError("check passed/detail fields have invalid types")
            checks.append(VerificationCheck(row["name"], row["passed"], row["detail"]))
        if len({item.name for item in checks}) != len(checks):
            raise ValueError("check names must be unique")
        if not _check_schema_complete(checks, transition_kind):
            raise ValueError("receipt check set does not match the transition contract")

        raw_footprints = raw.get("proposal_footprints")
        if not isinstance(raw_footprints, list):
            raise TypeError("proposal_footprints must be an array")
        footprints: list[ProposalFootprint] = []
        footprint_fields = {
            "proposal_id",
            "declared_reads",
            "observed_reads",
            "required_reads",
            "snapshot_reads",
            "writes",
        }
        for row in raw_footprints:
            if not isinstance(row, Mapping) or set(row) != footprint_fields:
                raise ValueError("proposal footprint fields are noncanonical")
            proposal_id = row["proposal_id"]
            if not isinstance(proposal_id, str) or not proposal_id:
                raise TypeError("footprint proposal IDs must be non-empty strings")
            footprints.append(
                ProposalFootprint(
                    proposal_id=proposal_id,
                    declared_reads=_canonical_ref_tuple(
                        row["declared_reads"], field_name="declared_reads"
                    ),
                    observed_reads=_canonical_ref_tuple(
                        row["observed_reads"], field_name="observed_reads"
                    ),
                    required_reads=_canonical_ref_tuple(
                        row["required_reads"], field_name="required_reads"
                    ),
                    snapshot_reads=_canonical_ref_tuple(
                        row["snapshot_reads"], field_name="snapshot_reads"
                    ),
                    writes=_canonical_ref_tuple(row["writes"], field_name="writes"),
                )
            )
        proposal_footprints = tuple(footprints)
        if tuple(item.proposal_id for item in proposal_footprints) != proposal_ids:
            raise ValueError("proposal footprint IDs do not match proposal_ids")

        score_observed_reads = _canonical_ref_tuple(
            raw.get("score_observed_reads"), field_name="score_observed_reads"
        )
        score_snapshot_reads = _canonical_ref_tuple(
            raw.get("score_snapshot_reads"), field_name="score_snapshot_reads"
        )
        raw_edges = raw.get("conflict_edges")
        if not isinstance(raw_edges, list):
            raise TypeError("conflict_edges must be an array")
        conflict_edges: tuple[tuple[str, str], ...] = tuple(
            _string_tuple(edge, field_name="conflict edge") for edge in raw_edges
        )
        if any(len(edge) != 2 for edge in conflict_edges):
            raise ValueError("each conflict edge must contain exactly two proposal IDs")

        committed = raw.get("committed")
        if not isinstance(committed, bool):
            raise TypeError("committed must be boolean")
        mismatch_before = _ledger_from_artifact(
            raw.get("mismatch_before"), field_name="mismatch_before"
        )
        mismatch_after = _ledger_from_artifact(
            raw.get("mismatch_after"), field_name="mismatch_after"
        )
        snapshot_hashes = _string_tuple(
            raw.get("snapshot_hashes"), field_name="snapshot_hashes"
        )
        if transition_kind is not TransitionKind.ROLLBACK and len(
            snapshot_hashes
        ) != len(proposal_footprints):
            raise ValueError(
                "snapshot hash count does not match proposal footprint count"
            )
        parent_event_ids = _canonical_ref_tuple(
            raw.get("parent_event_ids"), field_name="parent_event_ids"
        )
        for name in (
            "pre_state_hash",
            "post_state_hash",
            "conflict_support_hash",
            "read_set_hash",
            "write_set_hash",
            "protected_boundary_hash",
            "versions_before_hash",
            "versions_after_hash",
            "rollback_root",
            "commit_id",
        ):
            value = raw.get(name)
            if not isinstance(value, str) or len(value) != 64 or value != value.lower():
                raise TypeError(f"{name} must be a SHA-256 hex string")
            int(value, 16)
        for name in ("transition_event_id", "semantic_record_event_id"):
            value = raw.get(name)
            if value is not None:
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or value != value.lower()
                ):
                    raise TypeError(f"{name} must be null or a SHA-256 hex string")
                int(value, 16)

        derived = RepairCommitReceipt(
            proposal_ids=proposal_ids,
            transition_kind=transition_kind,
            proposal_classes=proposal_classes,
            checks=tuple(checks),
            committed=committed,
            pre_state_hash=raw["pre_state_hash"],
            post_state_hash=raw["post_state_hash"],
            mismatch_before=mismatch_before,
            mismatch_after=mismatch_after,
            snapshot_hashes=snapshot_hashes,
            proposal_footprints=proposal_footprints,
            score_observed_reads=score_observed_reads,
            score_snapshot_reads=score_snapshot_reads,
            conflict_edges=conflict_edges,
            conflict_support_hash=raw["conflict_support_hash"],
            read_set_hash=raw["read_set_hash"],
            write_set_hash=raw["write_set_hash"],
            protected_boundary_hash=raw["protected_boundary_hash"],
            versions_before_hash=raw["versions_before_hash"],
            versions_after_hash=raw["versions_after_hash"],
            rollback_root=raw["rollback_root"],
            parent_event_ids=parent_event_ids,
            transition_event_id=raw.get("transition_event_id"),
            semantic_record_event_id=raw.get("semantic_record_event_id"),
            commit_id=raw["commit_id"],
        )

        expected_read_hash = _stable_hash(
            sorted({ref for item in proposal_footprints for ref in item.declared_reads})
        )
        expected_write_hash = _stable_hash(
            sorted({ref for item in proposal_footprints for ref in item.writes})
        )
        if proposal_footprints and expected_read_hash != derived.read_set_hash:
            failures.append(
                "aggregate read_set_hash does not match proposal footprints"
            )
        if proposal_footprints and expected_write_hash != derived.write_set_hash:
            failures.append(
                "aggregate write_set_hash does not match proposal footprints"
            )
        if derived.rollback_root != derived.pre_state_hash:
            failures.append("rollback_root does not equal pre_state_hash")

        expected_transition_event_id = None
        if (
            committed
            and not derived.diagnostic_only
            and transition_kind
            in {TransitionKind.STRICT_REPAIR, TransitionKind.RECORD_COMMIT}
        ):
            expected_transition_event_id = _stable_hash(
                {
                    "event_domain": "finite_transition",
                    "canonical_payload": {
                        "proposal_ids": proposal_ids,
                        "transition_kind": transition_kind.value,
                        "pre": derived.pre_state_hash,
                        "post": derived.post_state_hash,
                    },
                    "visible_footprint": sorted(
                        {
                            ref
                            for item in proposal_footprints
                            for ref in item.declared_reads
                        }
                        | {ref for item in proposal_footprints for ref in item.writes}
                    ),
                    "semantic_parents": parent_event_ids,
                }
            )
        if derived.transition_event_id != expected_transition_event_id:
            failures.append(
                "transition_event_id violates the transition-event contract"
            )
        expected_record_event_id = None
        if (
            transition_kind is TransitionKind.RECORD_COMMIT
            and expected_transition_event_id
        ):
            expected_record_event_id = _stable_hash(
                {
                    "event_domain": "semantic_record_commit",
                    "transition_event_id": expected_transition_event_id,
                    "semantic_parents": parent_event_ids,
                }
            )
        if derived.semantic_record_event_id != expected_record_event_id:
            failures.append(
                "semantic_record_event_id violates the record-commit contract"
            )

        expected_commit_id = _stable_hash(
            _commit_seed(
                proposal_ids=proposal_ids,
                transition_kind=transition_kind,
                proposal_classes=proposal_classes,
                checks=checks,
                committed=committed,
                pre_state_hash=derived.pre_state_hash,
                post_state_hash=derived.post_state_hash,
                mismatch_before=mismatch_before,
                mismatch_after=mismatch_after,
                snapshot_hashes=snapshot_hashes,
                conflict_support_hash=derived.conflict_support_hash,
                read_set_hash=derived.read_set_hash,
                write_set_hash=derived.write_set_hash,
                protected_boundary_hash=derived.protected_boundary_hash,
                versions_before_hash=derived.versions_before_hash,
                versions_after_hash=derived.versions_after_hash,
                parent_event_ids=parent_event_ids,
                transition_event_id=derived.transition_event_id,
                semantic_record_event_id=derived.semantic_record_event_id,
            )
        )
        if derived.commit_id != expected_commit_id:
            failures.append(
                "commit_id does not match canonical primitive receipt fields"
            )

        claimed_fields = {
            "COMPLETE_READ_SET_RECEIPT": derived.COMPLETE_READ_SET_RECEIPT,
            "CONFLICT_COMPONENT_SUPPORT_RECEIPT": derived.CONFLICT_COMPONENT_SUPPORT_RECEIPT,
            "ATOMIC_UNION_REVALIDATION_RECEIPT": derived.ATOMIC_UNION_REVALIDATION_RECEIPT,
            "TRANSACTIONAL_REPAIR_RECEIPT": derived.TRANSACTIONAL_REPAIR_RECEIPT,
            "physical_repair_receipt": derived.physical_repair_receipt,
            "semantic_record_written": derived.semantic_record_written,
            "transition_event_emitted": derived.transition_event_emitted,
            "diagnostic_only": derived.diagnostic_only,
        }
        for name, expected in claimed_fields.items():
            if raw.get(name) is not expected:
                failures.append(
                    f"caller-authored {name} disagrees with recomputed value"
                )
        if raw.get("verdict") != derived.verdict:
            failures.append("verdict disagrees with recomputed checks")
        if raw.get("claim_tier") != derived.claim_tier:
            failures.append("claim_tier disagrees with recomputed transition status")
        if raw.get("scope") != "finite quotient-visible transactional repair":
            failures.append("scope is noncanonical")
        if raw.get("nonclaims") != [
            "physical source provenance",
            "probability or vacuum selection",
            "H3 or event geometry",
            "KMS clock normalization",
            "continuum or Standard Model emergence",
        ]:
            failures.append("nonclaims are missing or noncanonical")
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        failures.append(str(exc))

    artifact_hash = None
    if isinstance(raw, Mapping):
        try:
            artifact_hash = _stable_hash(raw)
        except (TypeError, ValueError) as exc:
            failures.append(f"artifact is not canonically hashable: {exc}")
    integrity = not failures and derived is not None
    return {
        "schema": "oph_repair_receipt_self_consistency_diagnostic_v1",
        "receipt_type": "TRANSACTIONAL_REPAIR_RECEIPT_SELF_CONSISTENCY_DIAGNOSTIC",
        "REPAIR_ARTIFACT_INTEGRITY_RECEIPT": integrity,
        "COMPLETE_READ_SET_RECEIPT": bool(
            integrity and derived and derived.COMPLETE_READ_SET_RECEIPT
        ),
        "CONFLICT_COMPONENT_SUPPORT_RECEIPT": bool(
            integrity and derived and derived.CONFLICT_COMPONENT_SUPPORT_RECEIPT
        ),
        "ATOMIC_UNION_REVALIDATION_RECEIPT": bool(
            integrity and derived and derived.ATOMIC_UNION_REVALIDATION_RECEIPT
        ),
        "TRANSACTIONAL_REPAIR_RECEIPT": bool(
            integrity and derived and derived.TRANSACTIONAL_REPAIR_RECEIPT
        ),
        "commit_id": derived.commit_id if derived is not None else None,
        "artifact_hash": artifact_hash,
        "failure_reasons": failures,
        "verifier_version": REPAIR_VERIFIER_VERSION,
    }


def verify_repair_receipt_artifact(
    artifact: RepairCommitReceipt | Mapping[str, Any],
) -> dict[str, Any]:
    """Run the non-promotional serialized-receipt consistency diagnostic.

    A receipt is an output of the transaction engine, not a replay input.  Its
    hashes, checks, and booleans can be coauthored into a mutually consistent
    but fictitious transition.  Consequently this compatibility entry point
    never emits a promotion-capable repair receipt.  Use
    :func:`verify_repair_replay_envelope` with primitive inputs instead.
    """

    diagnostic = _verify_repair_receipt_self_consistency(artifact)
    self_consistent = bool(
        diagnostic.get("REPAIR_ARTIFACT_INTEGRITY_RECEIPT") is True
    )
    return {
        "schema": "oph_repair_receipt_diagnostic_verification_v2",
        "receipt_type": "TRANSACTIONAL_REPAIR_RECEIPT_DIAGNOSTIC_VERIFICATION",
        "REPAIR_RECEIPT_SERIALIZATION_SELF_CONSISTENCY_DIAGNOSTIC_RECEIPT": (
            self_consistent
        ),
        "REPAIR_REPLAY_ENVELOPE_INTEGRITY_RECEIPT": False,
        "REPAIR_ARTIFACT_INTEGRITY_RECEIPT": False,
        "COMPLETE_READ_SET_RECEIPT": False,
        "CONFLICT_COMPONENT_SUPPORT_RECEIPT": False,
        "ATOMIC_UNION_REVALIDATION_RECEIPT": False,
        "TRANSACTIONAL_REPAIR_RECEIPT": False,
        "RECORD_COMMIT_REPLAY_RECEIPT": False,
        "commit_id": diagnostic.get("commit_id"),
        "artifact_hash": diagnostic.get("artifact_hash"),
        "failure_reasons": list(diagnostic.get("failure_reasons", [])),
        "promotion_blockers": ["primitive_replay_envelope_required"],
        "verifier_version": REPAIR_REPLAY_VERIFIER_VERSION,
    }


_REPLAY_ENVELOPE_FIELDS = frozenset(
    {
        "schema",
        "artifact_type",
        "replay_verifier_version",
        "initial_state",
        "initial_versions",
        "mismatch_evaluator",
        "proposals",
        "expected_receipt",
        "expected_post_state",
        "expected_versions_after",
        "scope",
        "nonclaims",
    }
)
_REPLAY_PROPOSAL_FIELDS = frozenset(
    {
        "proposal_id",
        "transition_kind",
        "proposal_class",
        "collar",
        "declared_read_set",
        "recovery",
        "inverse_updates",
        "source_parameters",
        "parent_event_ids",
    }
)
_REPLAY_COLLAR_FIELDS = frozenset(
    {
        "collar_id",
        "visible_read_set",
        "writable_registers",
        "protected_boundary",
        "sector_registers",
        "record_registers",
        "checkpoint_registers",
        "interior_registers",
        "carrier_ids",
        "seam_ids",
        "forbidden_target_fields",
    }
)
_REPLAY_LEDGER_COMPONENTS = frozenset(
    {"record", "sector", "holonomy", "overlap", "local_constraint"}
)


def _require_exact_mapping_fields(
    value: Any, expected: frozenset[str], *, field_name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    if set(value) != expected:
        raise ValueError(f"{field_name} has missing or unexpected fields")
    _validate_register_mapping(value, name=field_name)
    return value


def _validate_plain_json(value: Any, *, field_name: str) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite float")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_plain_json(item, field_name=f"{field_name}[{index}]")
        return
    if type(value) is dict:
        _validate_register_mapping(value, name=field_name)
        for key, item in value.items():
            _validate_plain_json(item, field_name=f"{field_name}.{key}")
        return
    raise TypeError(
        f"{field_name} must be a plain JSON value, received {type(value).__name__}"
    )


def _signed_fraction_from_replay(value: Any, *, field_name: str) -> Fraction:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an exact number, not bool")
    if isinstance(value, int):
        return Fraction(value)
    if not isinstance(value, Mapping) or set(value) != {"numerator", "denominator"}:
        raise TypeError(
            f"{field_name} must be an integer or exact numerator/denominator object"
        )
    numerator = value["numerator"]
    denominator = value["denominator"]
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        raise TypeError(f"{field_name} has invalid exact integer components")
    result = Fraction(numerator, denominator)
    if dict(value) != _fraction_json(result):
        raise ValueError(f"{field_name} is not a reduced canonical fraction")
    return result


def _parse_replay_expression(
    value: Any, *, field_name: str
) -> tuple[tuple[tuple[str, Fraction], ...], Fraction, str]:
    expression = _require_exact_mapping_fields(
        value,
        frozenset({"terms", "constant", "transform"}),
        field_name=field_name,
    )
    raw_terms = expression["terms"]
    if not isinstance(raw_terms, list):
        raise TypeError(f"{field_name}.terms must be an array")
    terms: list[tuple[str, Fraction]] = []
    for index, raw_term in enumerate(raw_terms):
        term = _require_exact_mapping_fields(
            raw_term,
            frozenset({"register", "coefficient"}),
            field_name=f"{field_name}.terms[{index}]",
        )
        register = term["register"]
        if not isinstance(register, str) or not register:
            raise TypeError(f"{field_name}.terms[{index}].register is invalid")
        terms.append(
            (
                register,
                _signed_fraction_from_replay(
                    term["coefficient"],
                    field_name=f"{field_name}.terms[{index}].coefficient",
                ),
            )
        )
    if terms != sorted(terms, key=lambda item: item[0]):
        raise ValueError(f"{field_name}.terms must be sorted by register")
    if len({register for register, _ in terms}) != len(terms):
        raise ValueError(f"{field_name}.terms contains duplicate registers")
    transform = expression["transform"]
    if transform not in {"identity", "absolute"}:
        raise ValueError(f"{field_name}.transform is not allowlisted")
    constant = _signed_fraction_from_replay(
        expression["constant"], field_name=f"{field_name}.constant"
    )
    return tuple(terms), constant, str(transform)


def _build_replay_mismatch_evaluator(spec: Any) -> MismatchEvaluator:
    raw = _require_exact_mapping_fields(
        spec,
        frozenset({"kind", "components", "physical_auxiliary"}),
        field_name="mismatch_evaluator",
    )
    if raw["kind"] != "exact_affine_ledger_v1":
        raise ValueError("mismatch_evaluator.kind is not allowlisted")
    components = _require_exact_mapping_fields(
        raw["components"],
        _REPLAY_LEDGER_COMPONENTS,
        field_name="mismatch_evaluator.components",
    )
    parsed_components = {
        name: _parse_replay_expression(
            components[name], field_name=f"mismatch_evaluator.components.{name}"
        )
        for name in sorted(_REPLAY_LEDGER_COMPONENTS)
    }
    auxiliaries = raw["physical_auxiliary"]
    if not isinstance(auxiliaries, Mapping):
        raise TypeError("mismatch_evaluator.physical_auxiliary must be an object")
    _validate_register_mapping(
        auxiliaries, name="mismatch_evaluator.physical_auxiliary"
    )
    parsed_auxiliaries = {
        name: _parse_replay_expression(
            expression,
            field_name=f"mismatch_evaluator.physical_auxiliary.{name}",
        )
        for name, expression in sorted(auxiliaries.items())
    }

    def evaluate_expression(
        state: State,
        expression: tuple[tuple[tuple[str, Fraction], ...], Fraction, str],
    ) -> Fraction:
        terms, constant, transform = expression
        result = constant
        for register, coefficient in terms:
            result += coefficient * _signed_fraction_from_replay(
                state[register], field_name=f"state.{register}"
            )
        return abs(result) if transform == "absolute" else result

    def evaluator(state: State, _collar: RepairCollar) -> MismatchLedger:
        values = {
            name: evaluate_expression(state, expression)
            for name, expression in parsed_components.items()
        }
        return MismatchLedger(
            record=values["record"],
            sector=values["sector"],
            holonomy=values["holonomy"],
            overlap=values["overlap"],
            local_constraint=values["local_constraint"],
            physical_auxiliary={
                name: evaluate_expression(state, expression)
                for name, expression in parsed_auxiliaries.items()
            },
        )

    return evaluator


def _parse_replay_collar(value: Any, *, field_name: str) -> RepairCollar:
    raw = _require_exact_mapping_fields(
        value, _REPLAY_COLLAR_FIELDS, field_name=field_name
    )
    collar_id = raw["collar_id"]
    if not isinstance(collar_id, str) or not collar_id:
        raise TypeError(f"{field_name}.collar_id must be a non-empty string")

    def refs(name: str) -> frozenset[str]:
        return frozenset(
            _canonical_ref_tuple(raw[name], field_name=f"{field_name}.{name}")
        )

    return RepairCollar(
        collar_id=collar_id,
        visible_read_set=refs("visible_read_set"),
        writable_registers=refs("writable_registers"),
        protected_boundary=refs("protected_boundary"),
        sector_registers=refs("sector_registers"),
        record_registers=refs("record_registers"),
        checkpoint_registers=refs("checkpoint_registers"),
        interior_registers=refs("interior_registers"),
        carrier_ids=refs("carrier_ids"),
        seam_ids=refs("seam_ids"),
        forbidden_target_fields=refs("forbidden_target_fields"),
    )


def _build_replay_recovery(value: Any, *, field_name: str) -> ProposalBuilder:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    kind = value.get("kind")
    if kind == "literal_updates_v1":
        raw = _require_exact_mapping_fields(
            value,
            frozenset({"kind", "updates"}),
            field_name=field_name,
        )
        updates = raw["updates"]
        if not isinstance(updates, Mapping):
            raise TypeError(f"{field_name}.updates must be an object")
        _validate_register_mapping(updates, name=f"{field_name}.updates")
        _validate_plain_json(updates, field_name=f"{field_name}.updates")

        def literal_recovery(_state: State) -> Mapping[str, Any]:
            return copy.deepcopy(dict(updates))

        return literal_recovery
    if kind == "append_literal_v1":
        raw = _require_exact_mapping_fields(
            value,
            frozenset({"kind", "register", "value"}),
            field_name=field_name,
        )
        register = raw["register"]
        if not isinstance(register, str) or not register:
            raise TypeError(f"{field_name}.register must be a non-empty string")
        appended_value = raw["value"]
        _validate_plain_json(appended_value, field_name=f"{field_name}.value")

        def append_recovery(state: State) -> Mapping[str, Any]:
            current = state[register]
            if type(current) is not list:
                raise TypeError("append_literal_v1 requires a list-valued register")
            return {register: [*current, copy.deepcopy(appended_value)]}

        return append_recovery
    raise ValueError(f"{field_name}.kind is not allowlisted")


def _instantiate_repair_replay(
    *,
    initial_state: Mapping[str, Any],
    initial_versions: Mapping[str, Any],
    mismatch_evaluator_spec: Any,
    proposal_specs: Any,
) -> tuple[RepairCommitReceipt, dict[str, Any], dict[str, int]]:
    if not isinstance(initial_state, Mapping) or not initial_state:
        raise TypeError("initial_state must be a non-empty object")
    _validate_register_mapping(initial_state, name="initial_state")
    _validate_plain_json(initial_state, field_name="initial_state")
    if not isinstance(initial_versions, Mapping):
        raise TypeError("initial_versions must be an object")
    _validate_register_mapping(initial_versions, name="initial_versions")
    if set(initial_versions) != set(initial_state):
        raise ValueError("initial_versions must cover exactly initial_state")
    versions: dict[str, int] = {}
    for register, version in initial_versions.items():
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise TypeError("initial_versions values must be non-negative integers")
        versions[register] = version
    evaluator = _build_replay_mismatch_evaluator(mismatch_evaluator_spec)
    engine = TransactionalRepairEngine(
        dict(initial_state), mismatch_evaluator=evaluator, versions=versions
    )
    if not isinstance(proposal_specs, list) or not proposal_specs:
        raise TypeError("proposals must be a non-empty array")
    parsed_rows: list[tuple[str, RepairProposal]] = []
    for index, raw_proposal in enumerate(proposal_specs):
        field_name = f"proposals[{index}]"
        proposal = _require_exact_mapping_fields(
            raw_proposal, _REPLAY_PROPOSAL_FIELDS, field_name=field_name
        )
        proposal_id = proposal["proposal_id"]
        if not isinstance(proposal_id, str) or not proposal_id:
            raise TypeError(f"{field_name}.proposal_id must be a non-empty string")
        declared_reads = _canonical_ref_tuple(
            proposal["declared_read_set"],
            field_name=f"{field_name}.declared_read_set",
        )
        inverse_updates = proposal["inverse_updates"]
        source_parameters = proposal["source_parameters"]
        if not isinstance(inverse_updates, Mapping):
            raise TypeError(f"{field_name}.inverse_updates must be an object")
        if not isinstance(source_parameters, Mapping):
            raise TypeError(f"{field_name}.source_parameters must be an object")
        _validate_register_mapping(
            inverse_updates, name=f"{field_name}.inverse_updates"
        )
        _validate_register_mapping(
            source_parameters, name=f"{field_name}.source_parameters"
        )
        _validate_plain_json(
            inverse_updates, field_name=f"{field_name}.inverse_updates"
        )
        _validate_plain_json(
            source_parameters, field_name=f"{field_name}.source_parameters"
        )
        parent_event_ids = _canonical_ref_tuple(
            proposal["parent_event_ids"],
            field_name=f"{field_name}.parent_event_ids",
        )
        parsed_rows.append(
            (
                proposal_id,
                engine.prepare(
                    proposal_id=proposal_id,
                    transition_kind=TransitionKind(proposal["transition_kind"]),
                    proposal_class=ProposalClass(proposal["proposal_class"]),
                    collar=_parse_replay_collar(
                        proposal["collar"], field_name=f"{field_name}.collar"
                    ),
                    declared_read_set=declared_reads,
                    recovery=_build_replay_recovery(
                        proposal["recovery"], field_name=f"{field_name}.recovery"
                    ),
                    inverse_updates=dict(inverse_updates),
                    source_parameters=dict(source_parameters),
                    parent_event_ids=parent_event_ids,
                ),
            )
        )
    if [proposal_id for proposal_id, _ in parsed_rows] != sorted(
        proposal_id for proposal_id, _ in parsed_rows
    ):
        raise ValueError("proposals must be sorted by proposal_id")
    proposals = [proposal for _, proposal in parsed_rows]
    if len(conflict_components(proposals)) != 1:
        raise ValueError("replay envelope must contain exactly one conflict component")
    receipts = engine.commit_batch(proposals)
    if len(receipts) != 1:
        raise RuntimeError("primitive replay did not produce exactly one receipt")
    return receipts[0], engine.state, engine.versions


def build_repair_replay_envelope(
    *,
    initial_state: Mapping[str, Any],
    initial_versions: Mapping[str, int],
    mismatch_evaluator: Mapping[str, Any],
    proposals: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create the canonical replay envelope from primitive transaction inputs."""

    receipt, post_state, versions_after = _instantiate_repair_replay(
        initial_state=initial_state,
        initial_versions=initial_versions,
        mismatch_evaluator_spec=mismatch_evaluator,
        proposal_specs=proposals,
    )
    envelope = {
        "schema": REPAIR_REPLAY_ENVELOPE_SCHEMA,
        "artifact_type": REPAIR_REPLAY_ENVELOPE_ARTIFACT_TYPE,
        "replay_verifier_version": REPAIR_REPLAY_VERIFIER_VERSION,
        "initial_state": copy.deepcopy(dict(initial_state)),
        "initial_versions": dict(initial_versions),
        "mismatch_evaluator": copy.deepcopy(dict(mismatch_evaluator)),
        "proposals": copy.deepcopy(proposals),
        "expected_receipt": receipt.as_dict(),
        "expected_post_state": post_state,
        "expected_versions_after": versions_after,
        "scope": _REPAIR_REPLAY_SCOPE,
        "nonclaims": list(_REPAIR_REPLAY_NONCLAIMS),
    }
    _validate_plain_json(envelope, field_name="replay_envelope")
    return envelope


def verify_repair_replay_envelope(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct a transaction from primitive inputs and compare every output."""

    failures: list[str] = []
    receipt: RepairCommitReceipt | None = None
    post_state: dict[str, Any] | None = None
    versions_after: dict[str, int] | None = None
    raw_expected: Mapping[str, Any] | None = None
    artifact_hash: str | None = None
    try:
        raw = _require_exact_mapping_fields(
            artifact, _REPLAY_ENVELOPE_FIELDS, field_name="replay_envelope"
        )
        _validate_plain_json(raw, field_name="replay_envelope")
        artifact_hash = _stable_hash(raw)
        if raw["schema"] != REPAIR_REPLAY_ENVELOPE_SCHEMA:
            raise ValueError("unexpected replay envelope schema")
        if raw["artifact_type"] != REPAIR_REPLAY_ENVELOPE_ARTIFACT_TYPE:
            raise ValueError("unexpected replay envelope artifact type")
        if raw["replay_verifier_version"] != REPAIR_REPLAY_VERIFIER_VERSION:
            raise ValueError("unexpected replay verifier version")
        if raw["scope"] != _REPAIR_REPLAY_SCOPE:
            raise ValueError("replay envelope scope is noncanonical")
        if raw["nonclaims"] != _REPAIR_REPLAY_NONCLAIMS:
            raise ValueError("replay envelope nonclaims are noncanonical")
        receipt, post_state, versions_after = _instantiate_repair_replay(
            initial_state=raw["initial_state"],
            initial_versions=raw["initial_versions"],
            mismatch_evaluator_spec=raw["mismatch_evaluator"],
            proposal_specs=raw["proposals"],
        )
        raw_expected = raw["expected_receipt"]
        if not isinstance(raw_expected, Mapping):
            raise TypeError("expected_receipt must be an object")
        diagnostic = _verify_repair_receipt_self_consistency(raw_expected)
        if diagnostic.get("REPAIR_ARTIFACT_INTEGRITY_RECEIPT") is not True:
            failures.extend(
                f"expected_receipt:{reason}"
                for reason in diagnostic.get("failure_reasons", [])
            )
        if _canonical(raw_expected) != _canonical(receipt.as_dict()):
            failures.append("expected_receipt_is_not_exact_replay_output")
        expected_post_state = raw["expected_post_state"]
        if not isinstance(expected_post_state, Mapping):
            raise TypeError("expected_post_state must be an object")
        if _canonical(expected_post_state) != _canonical(post_state):
            failures.append("expected_post_state_is_not_exact_replay_output")
        expected_versions_after = raw["expected_versions_after"]
        if not isinstance(expected_versions_after, Mapping):
            raise TypeError("expected_versions_after must be an object")
        if _canonical(expected_versions_after) != _canonical(versions_after):
            failures.append("expected_versions_after_is_not_exact_replay_output")
    except (KeyError, RuntimeError, TypeError, ValueError, ZeroDivisionError) as exc:
        failures.append(str(exc))
        if artifact_hash is None and isinstance(artifact, Mapping):
            try:
                artifact_hash = _stable_hash(artifact)
            except (TypeError, ValueError):
                pass

    integrity = not failures and receipt is not None
    replayed_transition = bool(
        integrity and receipt and receipt.committed and receipt.all_checks_passed
    )
    record_commit = bool(
        replayed_transition
        and receipt
        and receipt.transition_kind is TransitionKind.RECORD_COMMIT
        and receipt.semantic_record_written
    )
    pre_state = (
        copy.deepcopy(dict(artifact.get("initial_state", {})))
        if isinstance(artifact, Mapping)
        and isinstance(artifact.get("initial_state"), Mapping)
        else None
    )
    state_changes: dict[str, dict[str, Any]] = {}
    if integrity and pre_state is not None and post_state is not None:
        for register in sorted(set(pre_state) | set(post_state)):
            if (
                register not in pre_state
                or register not in post_state
                or not _same_value(pre_state[register], post_state[register])
            ):
                state_changes[register] = {
                    "before": copy.deepcopy(pre_state.get(register)),
                    "after": copy.deepcopy(post_state.get(register)),
                }
    return {
        "schema": "oph_repair_primitive_replay_verification_v1",
        "receipt_type": "TRANSACTIONAL_REPAIR_PRIMITIVE_REPLAY_VERIFICATION",
        "REPAIR_REPLAY_ENVELOPE_INTEGRITY_RECEIPT": integrity,
        "REPAIR_ARTIFACT_INTEGRITY_RECEIPT": integrity,
        "COMPLETE_READ_SET_RECEIPT": bool(
            integrity and receipt and receipt.COMPLETE_READ_SET_RECEIPT
        ),
        "CONFLICT_COMPONENT_SUPPORT_RECEIPT": bool(
            integrity and receipt and receipt.CONFLICT_COMPONENT_SUPPORT_RECEIPT
        ),
        "ATOMIC_UNION_REVALIDATION_RECEIPT": bool(
            integrity and receipt and receipt.ATOMIC_UNION_REVALIDATION_RECEIPT
        ),
        "REPLAYED_TRANSITION_RECEIPT": replayed_transition,
        "TRANSACTIONAL_REPAIR_RECEIPT": bool(
            integrity and receipt and receipt.TRANSACTIONAL_REPAIR_RECEIPT
        ),
        "RECORD_COMMIT_REPLAY_RECEIPT": record_commit,
        "transition_kind": (
            receipt.transition_kind.value if receipt is not None else None
        ),
        "semantic_record_event_id": (
            receipt.semantic_record_event_id if receipt is not None else None
        ),
        "commit_id": receipt.commit_id if receipt is not None else None,
        "receipt": receipt.as_dict() if receipt is not None else None,
        "pre_state": pre_state,
        "post_state": post_state,
        "versions_after": versions_after,
        "state_changes": state_changes,
        "artifact_hash": artifact_hash,
        "failure_reasons": failures,
        "verifier_version": REPAIR_REPLAY_VERIFIER_VERSION,
    }


@dataclass(frozen=True)
class _Evaluation:
    validation: ProposalValidation
    post_state: dict[RegisterRef, Any]
    read_set: frozenset[RegisterRef]
    write_set: frozenset[RegisterRef]
    protected_set: frozenset[RegisterRef]
    parent_event_ids: tuple[str, ...]
    collar: RepairCollar
    proposal_footprints: tuple[ProposalFootprint, ...]
    score_observed_reads: tuple[RegisterRef, ...]
    score_snapshot_reads: tuple[RegisterRef, ...]
    conflict_edges: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _CommitArchive:
    pre_state: _FrozenMap
    post_state_hash: str
    receipt: RepairCommitReceipt
    collar: RepairCollar


_DEFAULT_FORBIDDEN_FIELDS = frozenset(
    {
        "target",
        "target_value",
        "target_beta",
        "beta_target",
        "target_geometry",
        "preferred_geometry",
        "preferred_kms_normalization",
        "candidate_model",
        "candidate_geometry",
        "observed_cosmology",
        "pass_threshold",
        "desired_branch",
    }
)


def _normalize_field(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _target_leaks(
    value: Any, forbidden: frozenset[str], path: str = "source"
) -> list[str]:
    leaks: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = _normalize_field(str(raw_key))
            key_tokens = set(key.split("_"))
            forbidden_match = key in forbidden or any(
                candidate in key_tokens
                or key.startswith(f"{candidate}_")
                or key.endswith(f"_{candidate}")
                for candidate in forbidden
            )
            if forbidden_match:
                leaks.append(f"{path}.{key}")
            leaks.extend(_target_leaks(item, forbidden, f"{path}.{key}"))
    elif isinstance(value, (list, tuple, set, frozenset)):
        for index, item in enumerate(value):
            leaks.extend(_target_leaks(item, forbidden, f"{path}[{index}]"))
    elif isinstance(value, str):
        normalized = _normalize_field(value)
        if normalized in forbidden:
            leaks.append(path)
    return leaks


def _same_value(left: Any, right: Any) -> bool:
    return _stable_hash(_freeze(left)) == _stable_hash(_freeze(right))


def _changed_registers(before: State, after: State) -> frozenset[RegisterRef]:
    all_refs = set(before) | set(after)
    return frozenset(
        ref
        for ref in all_refs
        if ref not in before
        or ref not in after
        or not _same_value(before[ref], after[ref])
    )


def _union_collar(proposals: Sequence[RepairProposal]) -> RepairCollar:
    def union(attribute: str) -> frozenset[str]:
        values: set[str] = set()
        for proposal in proposals:
            values.update(getattr(proposal.collar, attribute))
        return frozenset(values)

    component_key = _stable_hash(
        sorted(proposal.proposal_id for proposal in proposals)
    )[:20]
    return RepairCollar(
        collar_id=f"union:{component_key}",
        visible_read_set=union("visible_read_set"),
        writable_registers=union("writable_registers"),
        protected_boundary=union("protected_boundary"),
        sector_registers=union("sector_registers"),
        record_registers=union("record_registers"),
        checkpoint_registers=union("checkpoint_registers"),
        interior_registers=union("interior_registers"),
        carrier_ids=union("carrier_ids"),
        seam_ids=union("seam_ids"),
        forbidden_target_fields=union("forbidden_target_fields"),
    )


def _append_only(before: Any, after: Any) -> bool:
    if isinstance(before, tuple) and isinstance(after, tuple):
        return len(after) > len(before) and after[: len(before)] == before
    if isinstance(before, list) and isinstance(after, list):
        return len(after) > len(before) and after[: len(before)] == before
    return False


class TransactionalRepairEngine:
    """In-memory reference verifier and atomic commit engine.

    It is an exact small-model/reference implementation, not a distributed
    scheduler.  Distributed engines should reproduce these component receipts
    with an independent verifier.
    """

    def __init__(
        self,
        initial_state: State,
        *,
        mismatch_evaluator: MismatchEvaluator,
        versions: Mapping[RegisterRef, RegisterVersion] | None = None,
    ) -> None:
        _validate_register_mapping(initial_state, name="initial state")
        self._state = {
            ref: copy.deepcopy(value) for ref, value in initial_state.items()
        }
        if not self._state:
            raise ValueError("transactional repair state must not be empty")
        # Validate/freeze all values at the boundary.
        for value in self._state.values():
            _freeze(value)
        if versions is None:
            self._versions = {ref: 0 for ref in self._state}
        else:
            _validate_register_mapping(versions, name="version")
            if set(versions) != set(self._state):
                raise ValueError("version map must cover exactly the state registers")
            self._versions = {}
            for ref, version in versions.items():
                if (
                    isinstance(version, bool)
                    or not isinstance(version, int)
                    or version < 0
                ):
                    raise ValueError("register versions must be non-negative integers")
                self._versions[ref] = version
        self._mismatch_evaluator = mismatch_evaluator
        self._archives: dict[str, _CommitArchive] = {}

    @property
    def state(self) -> dict[RegisterRef, Any]:
        return copy.deepcopy(self._state)

    @property
    def versions(self) -> dict[RegisterRef, RegisterVersion]:
        return dict(self._versions)

    def prepare(
        self,
        *,
        proposal_id: str,
        transition_kind: TransitionKind,
        proposal_class: ProposalClass,
        collar: RepairCollar,
        declared_read_set: Iterable[RegisterRef],
        recovery: ProposalBuilder,
        inverse_updates: Mapping[RegisterRef, Any] | None = None,
        source_parameters: Mapping[str, Any] | None = None,
        parent_event_ids: Iterable[str] = (),
    ) -> RepairProposal:
        return prepare_proposal(
            self._state,
            self._versions,
            proposal_id=proposal_id,
            transition_kind=transition_kind,
            proposal_class=proposal_class,
            collar=collar,
            declared_read_set=declared_read_set,
            recovery=recovery,
            inverse_updates=inverse_updates,
            source_parameters=source_parameters,
            parent_event_ids=parent_event_ids,
        )

    def assess(self, proposal: RepairProposal) -> ProposalValidation:
        return self._evaluate_component((proposal,)).validation

    def commit(self, proposal: RepairProposal) -> RepairCommitReceipt:
        return self._commit_component((proposal,))

    def commit_batch(
        self, proposals: Sequence[RepairProposal]
    ) -> tuple[RepairCommitReceipt, ...]:
        """Commit each conflict component atomically in canonical component order."""

        return tuple(
            self._commit_component(component)
            for component in conflict_components(proposals)
        )

    def rollback(self, commit_id: str) -> RepairCommitReceipt:
        """Restore an archived pre-state only when the archived post-state is current."""

        archive = self._archives.get(str(commit_id))
        pre_hash = canonical_state_hash(self._state)
        versions_before = dict(self._versions)
        checks: list[VerificationCheck] = []
        exists = archive is not None
        checks.append(
            VerificationCheck(
                "rollback_root_exists",
                exists,
                "archived commit exists"
                if exists
                else f"unknown rollback root {commit_id!r}",
            )
        )
        current_matches = bool(
            exists and archive and pre_hash == archive.post_state_hash
        )
        checks.append(
            VerificationCheck(
                "rollback_snapshot_current",
                current_matches,
                (
                    "current state equals archived post-state"
                    if current_matches
                    else "rollback rejected because later or divergent state is current"
                ),
            )
        )
        committed = all(check.passed for check in checks)
        mismatch_before: MismatchLedger | None = None
        mismatch_after: MismatchLedger | None = None
        protected: frozenset[str] = frozenset()
        rollback_write_set: frozenset[str] = frozenset()
        if committed and archive is not None:
            restored = _thaw(archive.pre_state)
            if not isinstance(
                restored, dict
            ):  # pragma: no cover - construction invariant
                raise RuntimeError("corrupt rollback archive")
            mismatch_before = self._measure(self._state, archive.collar)
            mismatch_after = self._measure(restored, archive.collar)
            changed = _changed_registers(self._state, restored)
            rollback_write_set = changed
            self._state = restored
            for ref in changed:
                self._versions[ref] += 1
            protected = archive.collar.protected_boundary
        post_hash = canonical_state_hash(self._state)
        proposal_ids = (f"rollback:{commit_id}",)
        receipt = self._make_receipt(
            proposal_ids=proposal_ids,
            transition_kind=TransitionKind.ROLLBACK,
            proposal_classes=(),
            checks=tuple(checks),
            committed=committed,
            pre_state_hash=pre_hash,
            post_state_hash=post_hash,
            mismatch_before=mismatch_before,
            mismatch_after=mismatch_after,
            snapshot_hashes=(),
            proposal_footprints=(),
            score_observed_reads=(),
            score_snapshot_reads=(),
            conflict_edges=(),
            read_set=frozenset(self._state),
            write_set=rollback_write_set,
            protected_set=protected,
            versions_before=versions_before,
            versions_after=self._versions,
            parent_event_ids=(),
            transition_event_id=None,
            semantic_record_event_id=None,
        )
        return receipt

    def _measure(self, state: State, collar: RepairCollar) -> MismatchLedger:
        result, _ = self._measure_with_trace(state, collar)
        return result

    def _measure_with_trace(
        self, state: State, collar: RepairCollar
    ) -> tuple[MismatchLedger, frozenset[RegisterRef]]:
        tracker = _ReadTrackingState(state)
        result = self._mismatch_evaluator(tracker, collar)
        if not isinstance(result, MismatchLedger):
            raise TypeError("mismatch evaluator must return MismatchLedger")
        return result, frozenset(tracker.observed)

    def _evaluate_component(self, component: Sequence[RepairProposal]) -> _Evaluation:
        proposals = tuple(sorted(component, key=lambda proposal: proposal.proposal_id))
        if not proposals:
            raise ValueError("cannot evaluate an empty conflict component")
        checks: list[VerificationCheck] = []
        proposal_ids = tuple(proposal.proposal_id for proposal in proposals)
        unique_ids = len(proposal_ids) == len(set(proposal_ids))
        checks.append(
            VerificationCheck(
                "unique_proposal_ids",
                unique_ids,
                "proposal IDs are unique" if unique_ids else "duplicate proposal IDs",
            )
        )
        collar = _union_collar(proposals)
        read_set = frozenset().union(
            *(proposal.declared_read_set for proposal in proposals)
        )
        write_set = frozenset().union(*(proposal.write_set for proposal in proposals))
        parent_ids = tuple(
            sorted(
                {
                    parent
                    for proposal in proposals
                    for parent in proposal.parent_event_ids
                }
            )
        )

        complete = True
        complete_details: list[str] = []
        fresh = True
        fresh_details: list[str] = []
        local = True
        local_details: list[str] = []
        target_free = True
        target_details: list[str] = []
        footprints: list[ProposalFootprint] = []
        for proposal in proposals:
            required_reads = (
                proposal.collar.visible_read_set
                | proposal.collar.protected_boundary
                | proposal.collar.sector_registers
                | proposal.collar.record_registers
                | proposal.collar.checkpoint_registers
                | proposal.write_set
            )
            footprints.append(
                ProposalFootprint(
                    proposal_id=proposal.proposal_id,
                    declared_reads=tuple(sorted(proposal.declared_read_set)),
                    observed_reads=tuple(sorted(proposal.observed_read_set)),
                    required_reads=tuple(sorted(required_reads)),
                    snapshot_reads=tuple(sorted(proposal.snapshot.read_set)),
                    writes=tuple(sorted(proposal.write_set)),
                )
            )
            missing = required_reads - proposal.declared_read_set
            untraced = proposal.observed_read_set - proposal.declared_read_set
            snapshot_mismatch = proposal.snapshot.read_set != proposal.declared_read_set
            if missing or untraced or snapshot_mismatch:
                complete = False
                complete_details.append(
                    f"{proposal.proposal_id}: missing={sorted(missing)}, "
                    f"observed_undeclared={sorted(untraced)}, "
                    f"snapshot_exact={not snapshot_mismatch}"
                )
            for item in proposal.snapshot.registers:
                if item.ref not in self._state or item.ref not in self._versions:
                    fresh = False
                    fresh_details.append(
                        f"{proposal.proposal_id}:{item.ref} no longer exists"
                    )
                    continue
                if self._versions[item.ref] != item.version or not _same_value(
                    self._state[item.ref], item.value
                ):
                    fresh = False
                    fresh_details.append(f"{proposal.proposal_id}:{item.ref} is stale")
            illegal_writes = proposal.write_set - proposal.collar.writable_registers
            if illegal_writes:
                local = False
                local_details.append(
                    f"{proposal.proposal_id}: writes outside collar {sorted(illegal_writes)}"
                )
            forbidden = (
                _DEFAULT_FORBIDDEN_FIELDS | proposal.collar.forbidden_target_fields
            )
            leaks = _target_leaks(proposal.source_parameters, forbidden)
            register_leaks = sorted(
                ref
                for ref in proposal.declared_read_set | proposal.write_set
                if _normalize_field(ref) in forbidden
            )
            if leaks or register_leaks:
                target_free = False
                target_details.append(
                    f"{proposal.proposal_id}: forbidden source paths={leaks}, "
                    f"registers={register_leaks}"
                )

        checks.extend(
            (
                VerificationCheck(
                    "complete_read_set",
                    complete,
                    "all proposal, score, protected and sector dependencies were snapshotted"
                    if complete
                    else "; ".join(complete_details),
                ),
                VerificationCheck(
                    "snapshot_current",
                    fresh,
                    "all snapshot values and register versions are current"
                    if fresh
                    else "; ".join(fresh_details),
                ),
                VerificationCheck(
                    "write_locality",
                    local,
                    "all writes lie in their declared collars"
                    if local
                    else "; ".join(local_details),
                ),
                VerificationCheck(
                    "target_free_source",
                    target_free,
                    "proposal source contains no forbidden downstream target fields"
                    if target_free
                    else "; ".join(target_details),
                ),
            )
        )
        proposal_footprints = tuple(
            sorted(footprints, key=lambda item: item.proposal_id)
        )
        conflict_edges = _expected_conflict_edges(proposal_footprints)
        conflict_supported = _footprint_conflict_receipt(
            proposal_footprints, conflict_edges
        )
        checks.append(
            VerificationCheck(
                "conflict_component_support",
                conflict_supported,
                "read/write footprints form one complete connected conflict component"
                if conflict_supported
                else "the supplied proposals do not form one connected conflict component",
            )
        )

        kinds = {proposal.transition_kind for proposal in proposals}
        one_kind = len(kinds) == 1
        checks.append(
            VerificationCheck(
                "single_transition_kind",
                one_kind,
                "component has one transition law"
                if one_kind
                else f"cannot aggregate transition kinds {sorted(value.value for value in kinds)}",
            )
        )
        transition_kind = proposals[0].transition_kind

        combined_updates: dict[str, Any] = {}
        compatible = True
        incompatible: list[str] = []
        for proposal in proposals:
            for ref, value in proposal.updates.items():
                if ref in combined_updates and not _same_value(
                    combined_updates[ref], value
                ):
                    compatible = False
                    incompatible.append(ref)
                else:
                    combined_updates[ref] = copy.deepcopy(value)
        checks.append(
            VerificationCheck(
                "compatible_component_writes",
                compatible,
                "component writes have a unique aggregate value"
                if compatible
                else f"conflicting values for registers {sorted(set(incompatible))}",
            )
        )

        post_state = copy.deepcopy(self._state)
        if compatible:
            post_state.update(combined_updates)
        changed = _changed_registers(self._state, post_state)
        unlisted_unchanged = changed <= write_set
        checks.append(
            VerificationCheck(
                "unlisted_registers_unchanged",
                unlisted_unchanged,
                "the candidate differs only on listed writes"
                if unlisted_unchanged
                else f"unlisted changes {sorted(changed - write_set)}",
            )
        )

        mutable_protected = (
            collar.record_registers
            if transition_kind is TransitionKind.RECORD_COMMIT
            else frozenset()
        )
        protected_to_compare = collar.protected_boundary - mutable_protected
        protected_preserved = all(
            ref in self._state
            and ref in post_state
            and _same_value(self._state[ref], post_state[ref])
            for ref in protected_to_compare
        )
        sector_preserved = all(
            ref in self._state
            and ref in post_state
            and _same_value(self._state[ref], post_state[ref])
            for ref in collar.sector_registers
        )
        checks.extend(
            (
                VerificationCheck(
                    "protected_boundary_preserved",
                    protected_preserved,
                    "protected boundary values are unchanged"
                    if protected_preserved
                    else "candidate mutates or omits a protected boundary register",
                ),
                VerificationCheck(
                    "sector_preserved",
                    sector_preserved,
                    "sector registers are unchanged"
                    if sector_preserved
                    else "candidate changes a sector register",
                ),
            )
        )

        before_ledger: MismatchLedger | None = None
        after_ledger: MismatchLedger | None = None
        score_reads: frozenset[RegisterRef] = frozenset()
        score_snapshot_reads = frozenset(
            ref for proposal in proposals for ref in proposal.snapshot.read_set
        )
        transition_ok = False
        transition_detail = "transition contract was not evaluated"
        if one_kind and compatible:
            before_ledger, before_score_reads = self._measure_with_trace(
                self._state, collar
            )
            after_ledger, after_score_reads = self._measure_with_trace(
                post_state, collar
            )
            score_reads = before_score_reads | after_score_reads
            same_objectives = (
                before_ledger.auxiliary_names == after_ledger.auxiliary_names
            )
            if transition_kind is TransitionKind.STRICT_REPAIR:
                semantic_state_unchanged = not bool(
                    changed & (collar.record_registers | collar.checkpoint_registers)
                )
                transition_ok = bool(
                    changed
                    and semantic_state_unchanged
                    and same_objectives
                    and after_ledger.strict_repair_descends_from(before_ledger)
                )
                transition_detail = (
                    "aggregate strictly descends the exact touched-collar ledger"
                    if transition_ok
                    else "strict repair must change state, preserve record and checkpoint "
                    "registers, retain the ledger schema, and strictly descend after atomic "
                    "union rescoring"
                )
            elif transition_kind in {
                TransitionKind.REVERSIBLE_PROPAGATION,
                TransitionKind.GAUGE_STUTTER,
            }:
                inverse: dict[str, Any] = {}
                inverse_compatible = True
                for proposal in reversed(proposals):
                    for ref, value in proposal.inverse_updates.items():
                        if ref in inverse and not _same_value(inverse[ref], value):
                            inverse_compatible = False
                        inverse[ref] = value
                reversed_state = copy.deepcopy(post_state)
                reversed_state.update(inverse)
                exact_inverse = inverse_compatible and canonical_state_hash(
                    reversed_state
                ) == canonical_state_hash(self._state)
                no_record = not bool(
                    changed & (collar.record_registers | collar.checkpoint_registers)
                )
                ledger_neutral = same_objectives and after_ledger == before_ledger
                transition_ok = bool(
                    changed and exact_inverse and no_record and ledger_neutral
                )
                transition_detail = (
                    "transition is mismatch-neutral, exactly invertible, and writes no record"
                    if transition_ok
                    else "reversible propagation requires a state change, an exact inverse, "
                    "ledger equality, and no record mutation"
                )
            elif transition_kind is TransitionKind.RECORD_COMMIT:
                legal_record_writes = bool(changed) and changed <= (
                    collar.record_registers | collar.checkpoint_registers
                )
                append_only = all(
                    ref in self._state
                    and ref in post_state
                    and _append_only(self._state[ref], post_state[ref])
                    for ref in changed & collar.record_registers
                )
                has_record = bool(changed & collar.record_registers)
                ledger_neutral = same_objectives and after_ledger == before_ledger
                transition_ok = bool(
                    legal_record_writes
                    and has_record
                    and append_only
                    and ledger_neutral
                )
                transition_detail = (
                    "record transition is append-only and mismatch-neutral"
                    if transition_ok
                    else "record commit must append to a declared record, write only record/"
                    "checkpoint registers, and preserve the mismatch ledger"
                )
            elif transition_kind is TransitionKind.CONTROLLED_EXPLORATION:
                transition_detail = "controlled exploration is fail-closed until a typed budget/drift verifier exists"
            elif transition_kind is TransitionKind.ROLLBACK:
                transition_detail = (
                    "rollback proposals are forbidden; use rollback(commit_id)"
                )
        score_trace_complete = bool(
            one_kind
            and compatible
            and score_reads <= read_set
            and score_reads <= score_snapshot_reads
        )
        checks.append(
            VerificationCheck(
                "mismatch_read_trace_complete",
                score_trace_complete,
                "all before/after mismatch-evaluator reads are in the declared component snapshot"
                if score_trace_complete
                else "mismatch evaluator read undeclared or unsnapshotted registers: "
                f"{sorted(score_reads - (read_set & score_snapshot_reads))}",
            )
        )
        checks.append(
            VerificationCheck("transition_contract", transition_ok, transition_detail)
        )

        all_prior = all(check.passed for check in checks)
        checks.append(
            VerificationCheck(
                "atomic_union_revalidated",
                all_prior,
                "the complete conflict component passed one atomic union-collar evaluation"
                if all_prior
                else "atomic component was rejected; no primitive member may commit independently",
            )
        )
        validation = ProposalValidation(
            proposal_ids=proposal_ids,
            checks=tuple(checks),
            mismatch_before=before_ledger,
            mismatch_after=after_ledger,
            pre_state_hash=canonical_state_hash(self._state),
            candidate_state_hash=canonical_state_hash(post_state),
        )
        return _Evaluation(
            validation=validation,
            post_state=post_state,
            read_set=read_set,
            write_set=write_set,
            protected_set=collar.protected_boundary,
            parent_event_ids=parent_ids,
            collar=collar,
            proposal_footprints=proposal_footprints,
            score_observed_reads=tuple(sorted(score_reads)),
            score_snapshot_reads=tuple(sorted(score_snapshot_reads)),
            conflict_edges=conflict_edges,
        )

    def _commit_component(
        self, component: Sequence[RepairProposal]
    ) -> RepairCommitReceipt:
        proposals = tuple(sorted(component, key=lambda proposal: proposal.proposal_id))
        evaluation = self._evaluate_component(proposals)
        validation = evaluation.validation
        versions_before = dict(self._versions)
        pre_state = copy.deepcopy(self._state)
        committed = validation.admissible
        if committed:
            changed = _changed_registers(self._state, evaluation.post_state)
            self._state = copy.deepcopy(evaluation.post_state)
            for ref in changed:
                self._versions[ref] += 1
        classes = tuple(
            sorted(
                {proposal.proposal_class for proposal in proposals},
                key=lambda x: x.value,
            )
        )
        transition_kind = proposals[0].transition_kind
        diagnostic = ProposalClass.DIAGNOSTIC_HEURISTIC in classes
        creates_transition_event = bool(
            committed
            and not diagnostic
            and transition_kind
            in {TransitionKind.STRICT_REPAIR, TransitionKind.RECORD_COMMIT}
        )
        transition_event_id = None
        if creates_transition_event:
            transition_event_id = _stable_hash(
                {
                    "event_domain": "finite_transition",
                    "canonical_payload": {
                        "proposal_ids": validation.proposal_ids,
                        "transition_kind": transition_kind.value,
                        "pre": validation.pre_state_hash,
                        "post": canonical_state_hash(self._state),
                    },
                    "visible_footprint": sorted(
                        evaluation.read_set | evaluation.write_set
                    ),
                    "semantic_parents": evaluation.parent_event_ids,
                }
            )
        semantic_record_event_id = None
        if (
            transition_kind is TransitionKind.RECORD_COMMIT
            and transition_event_id is not None
        ):
            semantic_record_event_id = _stable_hash(
                {
                    "event_domain": "semantic_record_commit",
                    "transition_event_id": transition_event_id,
                    "semantic_parents": evaluation.parent_event_ids,
                }
            )
        receipt = self._make_receipt(
            proposal_ids=validation.proposal_ids,
            transition_kind=transition_kind,
            proposal_classes=classes,
            checks=validation.checks,
            committed=committed,
            pre_state_hash=validation.pre_state_hash,
            post_state_hash=canonical_state_hash(self._state),
            mismatch_before=validation.mismatch_before,
            mismatch_after=validation.mismatch_after,
            snapshot_hashes=tuple(
                proposal.snapshot.snapshot_hash for proposal in proposals
            ),
            proposal_footprints=evaluation.proposal_footprints,
            score_observed_reads=evaluation.score_observed_reads,
            score_snapshot_reads=evaluation.score_snapshot_reads,
            conflict_edges=evaluation.conflict_edges,
            read_set=evaluation.read_set,
            write_set=evaluation.write_set,
            protected_set=evaluation.protected_set,
            versions_before=versions_before,
            versions_after=self._versions,
            parent_event_ids=evaluation.parent_event_ids,
            transition_event_id=transition_event_id,
            semantic_record_event_id=semantic_record_event_id,
        )
        if committed:
            self._archives[receipt.commit_id] = _CommitArchive(
                pre_state=_freeze(pre_state),
                post_state_hash=receipt.post_state_hash,
                receipt=receipt,
                collar=evaluation.collar,
            )
        return receipt

    def _make_receipt(
        self,
        *,
        proposal_ids: tuple[str, ...],
        transition_kind: TransitionKind,
        proposal_classes: tuple[ProposalClass, ...],
        checks: tuple[VerificationCheck, ...],
        committed: bool,
        pre_state_hash: str,
        post_state_hash: str,
        mismatch_before: MismatchLedger | None,
        mismatch_after: MismatchLedger | None,
        snapshot_hashes: tuple[str, ...],
        proposal_footprints: tuple[ProposalFootprint, ...],
        score_observed_reads: tuple[RegisterRef, ...],
        score_snapshot_reads: tuple[RegisterRef, ...],
        conflict_edges: tuple[tuple[str, str], ...],
        read_set: frozenset[str],
        write_set: frozenset[str],
        protected_set: frozenset[str],
        versions_before: Mapping[str, int],
        versions_after: Mapping[str, int],
        parent_event_ids: tuple[str, ...],
        transition_event_id: str | None,
        semantic_record_event_id: str | None,
    ) -> RepairCommitReceipt:
        read_hash = _stable_hash(sorted(read_set))
        write_hash = _stable_hash(sorted(write_set))
        protected_hash = _stable_hash(sorted(protected_set))
        versions_before_hash = _stable_hash(dict(versions_before))
        versions_after_hash = _stable_hash(dict(versions_after))
        conflict_support = _conflict_support_payload(
            proposal_footprints,
            score_observed_reads,
            score_snapshot_reads,
            conflict_edges,
        )
        conflict_support_hash = _stable_hash(conflict_support)
        seed = _commit_seed(
            proposal_ids=proposal_ids,
            transition_kind=transition_kind,
            proposal_classes=proposal_classes,
            checks=checks,
            committed=committed,
            pre_state_hash=pre_state_hash,
            post_state_hash=post_state_hash,
            mismatch_before=mismatch_before,
            mismatch_after=mismatch_after,
            snapshot_hashes=snapshot_hashes,
            conflict_support_hash=conflict_support_hash,
            read_set_hash=read_hash,
            write_set_hash=write_hash,
            protected_boundary_hash=protected_hash,
            versions_before_hash=versions_before_hash,
            versions_after_hash=versions_after_hash,
            parent_event_ids=parent_event_ids,
            transition_event_id=transition_event_id,
            semantic_record_event_id=semantic_record_event_id,
        )
        return RepairCommitReceipt(
            proposal_ids=proposal_ids,
            transition_kind=transition_kind,
            proposal_classes=proposal_classes,
            checks=checks,
            committed=committed,
            pre_state_hash=pre_state_hash,
            post_state_hash=post_state_hash,
            mismatch_before=mismatch_before,
            mismatch_after=mismatch_after,
            snapshot_hashes=snapshot_hashes,
            proposal_footprints=proposal_footprints,
            score_observed_reads=score_observed_reads,
            score_snapshot_reads=score_snapshot_reads,
            conflict_edges=conflict_edges,
            conflict_support_hash=conflict_support_hash,
            read_set_hash=read_hash,
            write_set_hash=write_hash,
            protected_boundary_hash=protected_hash,
            versions_before_hash=versions_before_hash,
            versions_after_hash=versions_after_hash,
            rollback_root=pre_state_hash,
            parent_event_ids=parent_event_ids,
            transition_event_id=transition_event_id,
            semantic_record_event_id=semantic_record_event_id,
            commit_id=_stable_hash(seed),
        )
