"""Finite modular-gearing diagnostics for oriented repair channels.

This module implements the finite, auditable MG0--MG3 part of the proposed
modular-gearing ladder.  It deliberately keeps three objects distinct:

* deterministic settlement/repair scheduling;
* an aggregate continuous-time process on physical quotient states; and
* the modular potential reconstructed from forward/reverse *rate ratios*.

The 24 labels are ``P0..P11 x {+,-}`` transition-channel labels.  They are not
24 Markov states, a successor cycle, modular eigenstates, or a clock.  Physical
promotion is fail-closed unless independently verified source provenance,
strong quotient lumpability, channel realization, and symmetry evidence are
all supplied.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from scipy import sparse

from oph_fpe.common_source_tower import verify_common_source_tower_report_file
from oph_fpe.core.icosahedral import icosahedral_a5_port_permutations


Orientation = Literal["+", "-"]
RateSemantics = Literal[
    "aggregate_quotient_continuous_time_rate",
    "source_model_continuous_time_rate",
    "synthetic_fixture_rate",
    "worker_attempt_count",
    "scheduler_order",
    "queue_position",
    "rejected_proposal_count",
]

_FORBIDDEN_RATE_SEMANTICS = {
    "worker_attempt_count",
    "scheduler_order",
    "queue_position",
    "rejected_proposal_count",
}
_ALLOWED_RATE_SEMANTICS = {
    "aggregate_quotient_continuous_time_rate",
    "source_model_continuous_time_rate",
    "synthetic_fixture_rate",
} | _FORBIDDEN_RATE_SEMANTICS
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def oriented_slot_names() -> tuple[str, ...]:
    """Return the canonical 24-slot order: plus layer, then minus layer."""

    return tuple(
        [f"P{port}+" for port in range(12)] + [f"P{port}-" for port in range(12)]
    )


def slot_index(base_port_id: int, orientation: Orientation) -> int:
    """Map an oriented port label to the canonical channel column."""

    if not 0 <= int(base_port_id) < 12:
        raise ValueError("base_port_id must be in 0..11")
    if orientation not in {"+", "-"}:
        raise ValueError("orientation must be '+' or '-'")
    return int(base_port_id) + (0 if orientation == "+" else 12)


@dataclass(frozen=True)
class OrientedRepairEdge:
    """One source-to-target transition in the quotient channel ledger."""

    edge_id: str
    source_quotient_state: str
    target_quotient_state: str
    reverse_edge_id: str
    base_port_id: int
    orientation: Orientation
    rate: int | float | Fraction
    rate_semantics: RateSemantics
    source_dag_hash: str | None = None
    scheduler_class: str | None = None
    attempt_clock_id: str | None = None

    @property
    def slot_name(self) -> str:
        return f"P{int(self.base_port_id)}{self.orientation}"


@dataclass(frozen=True)
class OrientedTransitionLedger:
    """Finite quotient state space and fixed-point-free oriented edge ledger."""

    quotient_states: tuple[str, ...]
    edges: tuple[OrientedRepairEdge, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "quotient_states",
            tuple(str(state) for state in self.quotient_states),
        )
        object.__setattr__(self, "edges", tuple(self.edges))

    @property
    def ledger_sha256(self) -> str:
        payload = {
            "schema": "oph.modular_gearing.oriented_ledger.v1",
            "quotient_states": list(self.quotient_states),
            "edges": [
                {
                    "edge_id": edge.edge_id,
                    "source": edge.source_quotient_state,
                    "target": edge.target_quotient_state,
                    "reverse_edge_id": edge.reverse_edge_id,
                    "base_port_id": int(edge.base_port_id),
                    "orientation": edge.orientation,
                    "rate": _stable_rate(edge.rate),
                    "rate_semantics": edge.rate_semantics,
                    "source_dag_hash": edge.source_dag_hash,
                    "scheduler_class": edge.scheduler_class,
                    "attempt_clock_id": edge.attempt_clock_id,
                }
                for edge in self.edges
            ],
        }
        return _sha256_json(payload)


@dataclass(frozen=True)
class PresentationTransitionRate:
    """Representative-level rate used only to compute quotient lumpability."""

    source_presentation: str
    target_presentation: str
    base_port_id: int
    orientation: Orientation
    rate: int | float | Fraction
    rate_semantics: RateSemantics


@dataclass(frozen=True)
class LumpabilityEvidence:
    """Computed strong channelwise quotient-lumpability certificate."""

    ledger_sha256: str
    presentation_count: int
    quotient_state_count: int
    exact_arithmetic_available: bool
    exact_strong_lumpability: bool | None
    maximum_absolute_defect: float
    maximum_relative_defect: float
    quotient_ledger_maximum_absolute_defect: float
    forbidden_rate_semantics_present: bool
    passed: bool
    evidence_sha256: str
    rows: tuple[dict[str, Any], ...]

    def report(self) -> dict[str, Any]:
        return {
            "schema": "oph.modular_gearing.strong_lumpability.v1",
            "ledger_sha256": self.ledger_sha256,
            "presentation_count": self.presentation_count,
            "quotient_state_count": self.quotient_state_count,
            "channel_key": "target_quotient_state_x_P0..P11_x_orientation",
            "exact_arithmetic_available": self.exact_arithmetic_available,
            "exact_strong_lumpability": self.exact_strong_lumpability,
            "maximum_absolute_defect": self.maximum_absolute_defect,
            "maximum_relative_defect": self.maximum_relative_defect,
            "quotient_ledger_maximum_absolute_defect": (
                self.quotient_ledger_maximum_absolute_defect
            ),
            "forbidden_rate_semantics_present": (self.forbidden_rate_semantics_present),
            "rows": list(self.rows),
            "evidence_sha256": self.evidence_sha256,
            "STRONG_QUOTIENT_LUMPABILITY_RECEIPT": self.passed,
            "FORBIDDEN_EXECUTION_METADATA_USED_AS_RATES": (
                self.forbidden_rate_semantics_present
            ),
            "claim_boundary": (
                "This is a computed channelwise strong-lumpability and ledger-match "
                "audit. It does not establish that the supplied representative rates "
                "were physically source-derived."
            ),
        }


@dataclass(frozen=True)
class PhysicalSourceEvidence:
    """Independent provenance packet required for physical MG promotion.

    The digests bind external evidence; this module does not manufacture those
    artifacts.  A caller-authored boolean is intentionally insufficient.
    """

    source_dag_sha256: str
    quotient_map_sha256: str
    rate_observation_bundle_sha256: str
    independent_verifier_receipt_sha256: str
    lumpability_evidence_sha256: str
    channel_realization_sha256: str
    symmetry_action_sha256: str
    artifact_paths: tuple[str, ...]
    common_source_tower_report_path: str | None = None
    rate_provenance_kind: str = "source_derived_aggregate_quotient_ctmc"
    forbidden_target_dependency_paths: tuple[str, ...] = ()
    worker_attempts_excluded: bool = True
    execution_counters_excluded_from_clock: bool = True


@dataclass(frozen=True)
class ChannelCompressionResult:
    """Whitened 24-channel compression and its modular closure diagnostics."""

    edge_order: tuple[str, ...]
    realization_sha256: str
    input_was_sparse: bool
    gram: np.ndarray
    whitened_realization: np.ndarray | None
    omega_24: np.ndarray | None
    full_column_rank: bool
    numerical_rank: int
    gram_condition_number: float | None
    whitening_residual: float | None
    closure_residual_hs: float | None
    relative_closure_residual: float | None
    affinities: np.ndarray

    def report(self) -> dict[str, Any]:
        return {
            "schema": "oph.modular_gearing.channel_compression.v1",
            "edge_count": len(self.edge_order),
            "channel_count": 24,
            "slot_order": list(oriented_slot_names()),
            "input_was_sparse": self.input_was_sparse,
            "realization_sha256": self.realization_sha256,
            "numerical_rank": self.numerical_rank,
            "full_column_rank": self.full_column_rank,
            "gram_condition_number": self.gram_condition_number,
            "whitening_residual": self.whitening_residual,
            "closure_residual_hs": self.closure_residual_hs,
            "relative_closure_residual": self.relative_closure_residual,
            "gram": _matrix_json(self.gram),
            "Omega_24": None if self.omega_24 is None else _matrix_json(self.omega_24),
            "CHANNEL_REALIZATION_FULL_RANK_RECEIPT": self.full_column_rank,
            "claim_boundary": (
                "Omega_24 is the compressed commutator grading on a realized "
                "channel subspace. It is not the state-space modular Hamiltonian K."
            ),
        }


@dataclass(frozen=True)
class SymmetrySpectrumResult:
    """A5/orientation covariance and paired-spectrum classifier result."""

    action_sha256: str
    report_payload: dict[str, Any]

    def report(self) -> dict[str, Any]:
        return dict(self.report_payload)


def validate_transition_ledger(
    ledger: OrientedTransitionLedger,
) -> dict[str, Any]:
    """Validate the quotient ledger and every reverse-oriented pair."""

    blockers: list[str] = []
    states = ledger.quotient_states
    state_set = set(states)
    if not states:
        blockers.append("quotient_state_space_is_empty")
    if len(state_set) != len(states):
        blockers.append("duplicate_quotient_state_ids")
    if not ledger.edges:
        blockers.append("oriented_transition_ledger_is_empty")

    by_id: dict[str, OrientedRepairEdge] = {}
    for edge in ledger.edges:
        if not edge.edge_id:
            blockers.append("empty_edge_id")
        elif edge.edge_id in by_id:
            blockers.append(f"duplicate_edge_id:{edge.edge_id}")
        by_id[edge.edge_id] = edge
        if edge.source_quotient_state not in state_set:
            blockers.append(f"unknown_source_state:{edge.edge_id}")
        if edge.target_quotient_state not in state_set:
            blockers.append(f"unknown_target_state:{edge.edge_id}")
        if not 0 <= int(edge.base_port_id) < 12:
            blockers.append(f"invalid_base_port_id:{edge.edge_id}")
        if edge.orientation not in {"+", "-"}:
            blockers.append(f"invalid_orientation:{edge.edge_id}")
        if not _positive_finite_rate(edge.rate):
            blockers.append(f"rate_not_strictly_positive_finite:{edge.edge_id}")
        if edge.rate_semantics in _FORBIDDEN_RATE_SEMANTICS:
            blockers.append(
                f"forbidden_rate_semantics:{edge.edge_id}:{edge.rate_semantics}"
            )
        elif edge.rate_semantics not in _ALLOWED_RATE_SEMANTICS:
            blockers.append(f"unknown_rate_semantics:{edge.edge_id}")

    pair_keys: set[tuple[str, str]] = set()
    reverse_rows: list[dict[str, Any]] = []
    for edge in ledger.edges:
        reverse = by_id.get(edge.reverse_edge_id)
        pair_passed = bool(
            reverse is not None
            and reverse.edge_id != edge.edge_id
            and reverse.reverse_edge_id == edge.edge_id
            and reverse.source_quotient_state == edge.target_quotient_state
            and reverse.target_quotient_state == edge.source_quotient_state
            and int(reverse.base_port_id) == int(edge.base_port_id)
            and reverse.orientation != edge.orientation
        )
        if not pair_passed:
            blockers.append(f"invalid_reverse_pair:{edge.edge_id}")
        else:
            pair_keys.add(tuple(sorted((edge.edge_id, reverse.edge_id))))
        reverse_rows.append(
            {
                "edge_id": edge.edge_id,
                "reverse_edge_id": edge.reverse_edge_id,
                "slot": edge.slot_name,
                "reverse_pair_valid": pair_passed,
            }
        )

    unique_blockers = sorted(set(blockers))
    passed = not unique_blockers
    return {
        "schema": "oph.modular_gearing.ledger_validation.v1",
        "ledger_sha256": ledger.ledger_sha256,
        "quotient_state_count": len(states),
        "oriented_edge_count": len(ledger.edges),
        "reverse_pair_count": len(pair_keys),
        "slot_alphabet": list(oriented_slot_names()),
        "used_slots": sorted({edge.slot_name for edge in ledger.edges}),
        "reverse_rows": reverse_rows,
        "blockers": unique_blockers,
        "ORIENTED_QUOTIENT_TRANSITION_LEDGER_RECEIPT": passed,
        "REVERSE_PAIR_INVOLUTION_RECEIPT": bool(
            passed and len(ledger.edges) == 2 * len(pair_keys)
        ),
        "FORBIDDEN_EXECUTION_METADATA_USED_AS_RATES": any(
            edge.rate_semantics in _FORBIDDEN_RATE_SEMANTICS
            or edge.rate_semantics not in _ALLOWED_RATE_SEMANTICS
            for edge in ledger.edges
        ),
        "WORKER_ATTEMPTS_USED_AS_RATES": any(
            edge.rate_semantics == "worker_attempt_count" for edge in ledger.edges
        ),
    }


def strong_quotient_lumpability_evidence(
    ledger: OrientedTransitionLedger,
    presentation_to_quotient: Mapping[str, str],
    representative_rates: Sequence[PresentationTransitionRate],
    *,
    tolerance: float = 1.0e-12,
) -> LumpabilityEvidence:
    """Compute strong lumpability per target quotient state *and* channel.

    The sum of representative-level rates from a presentation ``sigma`` into
    every ``(target quotient, oriented slot)`` bin must depend only on
    ``pi(sigma)``.  The descended rates are then compared with the supplied
    quotient ledger.  Forbidden counters and scheduler metadata never descend
    as rates.
    """

    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and nonnegative")
    ledger_valid = validate_transition_ledger(ledger)[
        "ORIENTED_QUOTIENT_TRANSITION_LEDGER_RECEIPT"
    ]
    mapping = {str(key): str(value) for key, value in presentation_to_quotient.items()}
    quotient_states = set(ledger.quotient_states)
    if not mapping:
        raise ValueError("presentation_to_quotient must be nonempty")
    if not set(mapping.values()) <= quotient_states:
        raise ValueError("presentation quotient keys must belong to the ledger")

    forbidden = any(
        item.rate_semantics in _FORBIDDEN_RATE_SEMANTICS
        or item.rate_semantics not in _ALLOWED_RATE_SEMANTICS
        for item in representative_rates
    )
    exact_available = all(
        _fraction_or_none(item.rate) is not None for item in representative_rates
    ) and all(_fraction_or_none(edge.rate) is not None for edge in ledger.edges)
    totals: dict[tuple[str, str, int, str], int | float | Fraction] = {}
    for sigma in mapping:
        for target in ledger.quotient_states:
            for channel in range(24):
                totals[(sigma, target, channel, "total")] = (
                    Fraction(0) if exact_available else 0.0
                )

    for item in representative_rates:
        if item.source_presentation not in mapping:
            raise ValueError(
                f"unknown source presentation {item.source_presentation!r}"
            )
        if item.target_presentation not in mapping:
            raise ValueError(
                f"unknown target presentation {item.target_presentation!r}"
            )
        if not _positive_finite_rate(item.rate):
            raise ValueError("representative transition rates must be positive")
        channel = slot_index(item.base_port_id, item.orientation)
        key = (
            item.source_presentation,
            mapping[item.target_presentation],
            channel,
            "total",
        )
        value: int | float | Fraction
        if exact_available:
            value = _fraction_or_none(item.rate) or Fraction(0)
        else:
            value = float(item.rate)
        totals[key] = totals[key] + value

    representatives: dict[str, list[str]] = defaultdict(list)
    for presentation, quotient in mapping.items():
        representatives[quotient].append(presentation)
    for values in representatives.values():
        values.sort()

    rows: list[dict[str, Any]] = []
    max_abs = 0.0
    max_rel = 0.0
    exact_equal = True
    descended: dict[tuple[str, str, int], float | Fraction] = {}
    for source_quotient in sorted(quotient_states):
        source_representatives = representatives.get(source_quotient, [])
        if not source_representatives:
            exact_equal = False
            rows.append(
                {
                    "source_quotient_state": source_quotient,
                    "missing_representatives": True,
                }
            )
            continue
        for target_quotient in sorted(quotient_states):
            for channel in range(24):
                values = [
                    totals[(sigma, target_quotient, channel, "total")]
                    for sigma in source_representatives
                ]
                if exact_available:
                    exact_values = [Fraction(value) for value in values]
                    exact_defect = max(exact_values) - min(exact_values)
                    defect = float(exact_defect)
                    exact_scale = max(
                        max(abs(value) for value in exact_values), Fraction(1)
                    )
                    relative = float(exact_defect / exact_scale)
                    representative_mismatch = exact_defect != 0
                else:
                    floats = [float(value) for value in values]
                    defect = max(floats) - min(floats)
                    scale = max(max(abs(value) for value in floats), 1.0)
                    relative = defect / scale
                    representative_mismatch = defect > tolerance
                max_abs = max(max_abs, defect)
                max_rel = max(max_rel, relative)
                if exact_available and representative_mismatch:
                    exact_equal = False
                descended[(source_quotient, target_quotient, channel)] = values[0]
                if representative_mismatch:
                    rows.append(
                        {
                            "source_quotient_state": source_quotient,
                            "target_quotient_state": target_quotient,
                            "slot": oriented_slot_names()[channel],
                            "representative_totals": [
                                _stable_rate(value) for value in values
                            ],
                            "absolute_defect": defect,
                            "relative_defect": relative,
                        }
                    )

    zero: float | Fraction = Fraction(0) if exact_available else 0.0
    ledger_totals: dict[tuple[str, str, int], float | Fraction] = defaultdict(
        lambda: zero
    )
    for edge in ledger.edges:
        ledger_totals[
            (
                edge.source_quotient_state,
                edge.target_quotient_state,
                slot_index(edge.base_port_id, edge.orientation),
            )
        ] += (
            _fraction_or_none(edge.rate) or Fraction(0)
            if exact_available
            else float(edge.rate)
        )
    all_rate_keys = set(descended) | set(ledger_totals)
    ledger_differences = [
        abs(descended.get(key, zero) - ledger_totals.get(key, zero))
        for key in all_rate_keys
    ]
    ledger_exact_equal = bool(
        exact_available and all(value == 0 for value in ledger_differences)
    )
    ledger_defect = max(
        (float(value) for value in ledger_differences),
        default=0.0,
    )
    exact_result = exact_equal if exact_available else None
    passed = bool(
        ledger_valid
        and not forbidden
        and (
            exact_equal and ledger_exact_equal
            if exact_available
            else max_abs <= tolerance and ledger_defect <= tolerance
        )
        and all(representatives.get(state) for state in quotient_states)
    )
    payload = {
        "ledger_sha256": ledger.ledger_sha256,
        "mapping": sorted(mapping.items()),
        "representative_rates": [
            {
                "source": item.source_presentation,
                "target": item.target_presentation,
                "slot": f"P{item.base_port_id}{item.orientation}",
                "rate": _stable_rate(item.rate),
                "semantics": item.rate_semantics,
            }
            for item in representative_rates
        ],
        "tolerance": tolerance,
    }
    return LumpabilityEvidence(
        ledger_sha256=ledger.ledger_sha256,
        presentation_count=len(mapping),
        quotient_state_count=len(quotient_states),
        exact_arithmetic_available=exact_available,
        exact_strong_lumpability=exact_result,
        maximum_absolute_defect=max_abs,
        maximum_relative_defect=max_rel,
        quotient_ledger_maximum_absolute_defect=ledger_defect,
        forbidden_rate_semantics_present=forbidden,
        passed=passed,
        evidence_sha256=_sha256_json(payload),
        rows=tuple(rows),
    )


def fundamental_cycle_holonomy(
    ledger: OrientedTransitionLedger,
    *,
    tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    """Audit a deterministic spanning forest and all fundamental cycles.

    Rational/integer rates receive an exact multiplicative check using
    :class:`fractions.Fraction`.  Every input also receives a floating
    logarithmic-affinity check.
    """

    validation = validate_transition_ledger(ledger)
    if not validation["ORIENTED_QUOTIENT_TRANSITION_LEDGER_RECEIPT"]:
        return {
            "schema": "oph.modular_gearing.fundamental_cycle_holonomy.v1",
            "ledger_sha256": ledger.ledger_sha256,
            "validation_blockers": validation["blockers"],
            "FUNDAMENTAL_CYCLE_HOLONOMY_RECEIPT": False,
        }
    pairs = _canonical_reverse_pairs(ledger)
    forest = _spanning_forest(ledger.quotient_states, pairs)
    edge_by_id = {edge.edge_id: edge for edge in ledger.edges}
    all_exact = all(_fraction_or_none(edge.rate) is not None for edge in ledger.edges)

    tree_adjacency: dict[str, list[tuple[str, OrientedRepairEdge, int]]] = defaultdict(
        list
    )
    for pair in forest["tree_pairs"]:
        plus = pair
        tree_adjacency[plus.source_quotient_state].append(
            (plus.target_quotient_state, plus, 1)
        )
        tree_adjacency[plus.target_quotient_state].append(
            (plus.source_quotient_state, plus, -1)
        )

    component_rows: list[dict[str, Any]] = []
    float_k: dict[str, float] = {}
    exact_mu_weight: dict[str, Fraction] = {}
    components = _undirected_components(ledger.quotient_states, pairs)
    for component_id, component in enumerate(components):
        root = min(component)
        float_k[root] = 0.0
        if all_exact:
            exact_mu_weight[root] = Fraction(1)
        queue: deque[str] = deque([root])
        visited = {root}
        while queue:
            source = queue.popleft()
            for target, plus, direction in sorted(
                tree_adjacency[source], key=lambda item: (item[0], item[1].edge_id)
            ):
                if target in visited:
                    continue
                reverse = edge_by_id[plus.reverse_edge_id]
                affinity = math.log(float(reverse.rate) / float(plus.rate))
                float_k[target] = float_k[source] + direction * affinity
                if all_exact:
                    q_plus = _fraction_or_none(plus.rate)
                    q_minus = _fraction_or_none(reverse.rate)
                    assert q_plus is not None and q_minus is not None
                    ratio = q_plus / q_minus
                    exact_mu_weight[target] = exact_mu_weight[source] * (
                        ratio if direction == 1 else 1 / ratio
                    )
                visited.add(target)
                queue.append(target)
        values = np.asarray([float_k[state] for state in component], dtype=float)
        shift = float(np.mean(values)) if values.size else 0.0
        for state in component:
            float_k[state] -= shift
        unnormalized = np.exp(-np.asarray([float_k[state] for state in component]))
        normalized = unnormalized / float(np.sum(unnormalized))
        component_rows.append(
            {
                "component_id": component_id,
                "states": list(component),
                "root": root,
                "state_count": len(component),
                "potential_gauge": "componentwise_arithmetic_mean_zero",
                "potential": {state: float_k[state] for state in component},
                "within_component_probability": {
                    state: float(probability)
                    for state, probability in zip(component, normalized, strict=True)
                },
                "sector_weight": "undetermined_by_intra_component_rate_ratios",
            }
        )

    cycle_rows: list[dict[str, Any]] = []
    maximum_affinity = 0.0
    exact_passed = True
    for plus in forest["chord_pairs"]:
        reverse = edge_by_id[plus.reverse_edge_id]
        expected_delta = (
            float_k[plus.target_quotient_state] - float_k[plus.source_quotient_state]
        )
        edge_affinity = math.log(float(reverse.rate) / float(plus.rate))
        cycle_affinity = edge_affinity - expected_delta
        maximum_affinity = max(maximum_affinity, abs(cycle_affinity))
        exact_holonomy: Fraction | None = None
        if all_exact:
            q_plus = _fraction_or_none(plus.rate)
            q_minus = _fraction_or_none(reverse.rate)
            assert q_plus is not None and q_minus is not None
            exact_holonomy = (
                exact_mu_weight[plus.source_quotient_state]
                * q_plus
                / (exact_mu_weight[plus.target_quotient_state] * q_minus)
            )
            exact_passed = exact_passed and exact_holonomy == 1
        cycle_rows.append(
            {
                "chord_edge_id": plus.edge_id,
                "reverse_edge_id": plus.reverse_edge_id,
                "source": plus.source_quotient_state,
                "target": plus.target_quotient_state,
                "slot": plus.slot_name,
                "cycle_log_affinity": cycle_affinity,
                "multiplicative_holonomy": math.exp(-cycle_affinity),
                "exact_detailed_balance_ratio": None
                if exact_holonomy is None
                else _stable_rate(exact_holonomy),
                "floating_cycle_passed": abs(cycle_affinity) <= tolerance,
                "exact_cycle_passed": None
                if exact_holonomy is None
                else exact_holonomy == 1,
            }
        )

    floating_passed = maximum_affinity <= tolerance
    passed = exact_passed if all_exact else floating_passed
    return {
        "schema": "oph.modular_gearing.fundamental_cycle_holonomy.v1",
        "ledger_sha256": ledger.ledger_sha256,
        "component_count": len(components),
        "sector_weight_ambiguity_dimension": max(0, len(components) - 1),
        "tree_pair_count": len(forest["tree_pairs"]),
        "fundamental_cycle_count": len(forest["chord_pairs"]),
        "exact_arithmetic_available": all_exact,
        "exact_fundamental_cycles_passed": exact_passed if all_exact else None,
        "maximum_absolute_cycle_log_affinity": maximum_affinity,
        "floating_fundamental_cycles_passed": floating_passed,
        "components": component_rows,
        "fundamental_cycles": cycle_rows,
        "FUNDAMENTAL_CYCLE_HOLONOMY_RECEIPT": bool(passed),
        "MODULAR_POTENTIAL_RECONSTRUCTION_RECEIPT": bool(passed),
        "GLOBAL_SECTOR_WEIGHT_SELECTION_RECEIPT": len(components) == 1,
        "claim_boundary": (
            "Rate ratios select a unique normalized law only inside each connected "
            "component. Cross-component sector weights remain undetermined."
        ),
    }


def weighted_hodge_reconstruction(
    ledger: OrientedTransitionLedger,
    *,
    edge_weights: Mapping[str, float] | None = None,
    affinity_noise_weighted_bound: float | None = None,
    tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    """Project edge affinities onto gradients and expose cycle residuals."""

    validation = validate_transition_ledger(ledger)
    if not validation["ORIENTED_QUOTIENT_TRANSITION_LEDGER_RECEIPT"]:
        return {
            "schema": "oph.modular_gearing.weighted_hodge.v1",
            "ledger_sha256": ledger.ledger_sha256,
            "validation_blockers": validation["blockers"],
            "WEIGHTED_HODGE_RECONSTRUCTION_RECEIPT": False,
        }
    pairs = _canonical_reverse_pairs(ledger)
    states = ledger.quotient_states
    state_index = {state: index for index, state in enumerate(states)}
    edge_by_id = {edge.edge_id: edge for edge in ledger.edges}
    incidence = np.zeros((len(pairs), len(states)), dtype=float)
    affinities = np.zeros(len(pairs), dtype=float)
    weights = np.ones(len(pairs), dtype=float)
    for row, plus in enumerate(pairs):
        reverse = edge_by_id[plus.reverse_edge_id]
        incidence[row, state_index[plus.source_quotient_state]] -= 1.0
        incidence[row, state_index[plus.target_quotient_state]] += 1.0
        affinities[row] = math.log(float(reverse.rate) / float(plus.rate))
        if edge_weights is not None:
            if plus.edge_id not in edge_weights:
                raise ValueError(f"missing Hodge weight for {plus.edge_id}")
            weights[row] = float(edge_weights[plus.edge_id])
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("all Hodge weights must be finite and positive")

    weighted_laplacian = incidence.T @ (weights[:, None] * incidence)
    rhs = incidence.T @ (weights * affinities)
    potential = np.linalg.pinv(weighted_laplacian, rcond=1.0e-13) @ rhs
    residual = affinities - incidence @ potential
    weighted_norm = float(math.sqrt(float(np.dot(weights * residual, residual))))
    infinity_norm = float(np.max(np.abs(residual))) if residual.size else 0.0
    normal_residual = float(np.linalg.norm(incidence.T @ (weights * residual), ord=2))
    eigenvalues = np.linalg.eigvalsh(weighted_laplacian)
    positive = eigenvalues[eigenvalues > max(1.0e-13, tolerance * 1.0e-3)]
    spectral_gap = float(np.min(positive)) if positive.size else None
    error_bound = None
    if affinity_noise_weighted_bound is not None:
        if affinity_noise_weighted_bound < 0.0:
            raise ValueError("affinity_noise_weighted_bound must be nonnegative")
        if spectral_gap is not None and spectral_gap > 0.0:
            error_bound = float(affinity_noise_weighted_bound / math.sqrt(spectral_gap))

    components = _undirected_components(states, pairs)
    probabilities: list[dict[str, Any]] = []
    for component_id, component in enumerate(components):
        indices = np.asarray([state_index[state] for state in component], dtype=int)
        values = np.exp(-potential[indices])
        values /= float(np.sum(values))
        probabilities.append(
            {
                "component_id": component_id,
                "within_component_probability": {
                    state: float(probability)
                    for state, probability in zip(component, values, strict=True)
                },
                "sector_weight": "undetermined_by_intra_component_rate_ratios",
            }
        )
    edge_rows = [
        {
            "edge_id": plus.edge_id,
            "slot": plus.slot_name,
            "affinity": float(affinities[row]),
            "integrable_part": float((incidence @ potential)[row]),
            "cycle_residual_h": float(residual[row]),
            "detailed_balance_log_residual": float(-residual[row]),
            "weight": float(weights[row]),
        }
        for row, plus in enumerate(pairs)
    ]
    passed = bool(weighted_norm <= tolerance and infinity_norm <= tolerance)
    return {
        "schema": "oph.modular_gearing.weighted_hodge.v1",
        "ledger_sha256": ledger.ledger_sha256,
        "potential_gauge": "minimum_norm_equals_componentwise_mean_zero",
        "potential": {
            state: float(potential[index]) for state, index in state_index.items()
        },
        "component_probabilities": probabilities,
        "component_count": len(components),
        "sector_weight_ambiguity_dimension": max(0, len(components) - 1),
        "weighted_cycle_residual_l2": weighted_norm,
        "cycle_residual_linf": infinity_norm,
        "weighted_normal_equation_residual": normal_residual,
        "weighted_laplacian_positive_spectral_gap": spectral_gap,
        "affinity_noise_weighted_bound": affinity_noise_weighted_bound,
        "potential_l2_error_bound": error_bound,
        "edge_rows": edge_rows,
        "WEIGHTED_HODGE_RECONSTRUCTION_RECEIPT": True,
        "EXACT_INTEGRABLE_MODULAR_PART_RECEIPT": passed,
        "claim_boundary": (
            "A small residual is an equilibrium approximation diagnostic. It is not "
            "a source-law or clock receipt, and its error amplification is controlled "
            "by the reported weighted Laplacian gap."
        ),
    }


def entropy_production_diagnostics(
    ledger: OrientedTransitionLedger,
    *,
    tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    """Compute aggregate and channel-resolved stationary entropy production."""

    validation = validate_transition_ledger(ledger)
    if not validation["ORIENTED_QUOTIENT_TRANSITION_LEDGER_RECEIPT"]:
        return {
            "schema": "oph.modular_gearing.entropy_production.v1",
            "ledger_sha256": ledger.ledger_sha256,
            "validation_blockers": validation["blockers"],
            "ENTROPY_PRODUCTION_DIAGNOSTIC_RECEIPT": False,
        }
    pairs = _canonical_reverse_pairs(ledger)
    states = ledger.quotient_states
    index = {state: offset for offset, state in enumerate(states)}
    q_matrix = np.zeros((len(states), len(states)), dtype=float)
    for edge in ledger.edges:
        source = index[edge.source_quotient_state]
        target = index[edge.target_quotient_state]
        if source != target:
            q_matrix[source, target] += float(edge.rate)
    np.fill_diagonal(q_matrix, -np.sum(q_matrix, axis=1))
    components = _undirected_components(states, pairs)
    edge_by_id = {edge.edge_id: edge for edge in ledger.edges}
    component_rows: list[dict[str, Any]] = []
    all_aggregate_db = True
    all_channel_db = True
    for component_id, component in enumerate(components):
        ids = np.asarray([index[state] for state in component], dtype=int)
        local_q = q_matrix[np.ix_(ids, ids)]
        if len(component) == 1:
            stationary = np.asarray([1.0])
        else:
            system = local_q.T.copy()
            target = np.zeros(len(component), dtype=float)
            system[-1, :] = 1.0
            target[-1] = 1.0
            stationary = np.linalg.solve(system, target)
            stationary = np.maximum(stationary, 0.0)
            stationary /= float(np.sum(stationary))
        local_index = {state: offset for offset, state in enumerate(component)}
        aggregate_epr = 0.0
        aggregate_db_residual = 0.0
        for left_offset in range(len(component)):
            for right_offset in range(left_offset + 1, len(component)):
                forward = stationary[left_offset] * local_q[left_offset, right_offset]
                reverse = stationary[right_offset] * local_q[right_offset, left_offset]
                if forward > 0.0 and reverse > 0.0:
                    log_ratio = math.log(forward / reverse)
                    aggregate_epr += (forward - reverse) * log_ratio
                    aggregate_db_residual = max(aggregate_db_residual, abs(log_ratio))
        channel_epr = 0.0
        channel_db_residual = 0.0
        for plus in pairs:
            if plus.source_quotient_state not in local_index:
                continue
            reverse_edge = edge_by_id[plus.reverse_edge_id]
            source_probability = stationary[local_index[plus.source_quotient_state]]
            target_probability = stationary[local_index[plus.target_quotient_state]]
            forward = source_probability * float(plus.rate)
            reverse = target_probability * float(reverse_edge.rate)
            if forward > 0.0 and reverse > 0.0:
                log_ratio = math.log(forward / reverse)
                channel_epr += (forward - reverse) * log_ratio
                channel_db_residual = max(channel_db_residual, abs(log_ratio))
        aggregate_pass = bool(
            aggregate_epr <= tolerance and aggregate_db_residual <= tolerance
        )
        channel_pass = bool(
            channel_epr <= tolerance and channel_db_residual <= tolerance
        )
        all_aggregate_db = all_aggregate_db and aggregate_pass
        all_channel_db = all_channel_db and channel_pass
        component_rows.append(
            {
                "component_id": component_id,
                "states": list(component),
                "stationary_probability": {
                    state: float(probability)
                    for state, probability in zip(component, stationary, strict=True)
                },
                "aggregate_entropy_production": float(aggregate_epr),
                "channelwise_entropy_production": float(channel_epr),
                "aggregate_detailed_balance_max_log_residual": float(
                    aggregate_db_residual
                ),
                "channelwise_detailed_balance_max_log_residual": float(
                    channel_db_residual
                ),
                "aggregate_detailed_balance_receipt": aggregate_pass,
                "channelwise_detailed_balance_receipt": channel_pass,
            }
        )
    connected = len(components) == 1
    return {
        "schema": "oph.modular_gearing.entropy_production.v1",
        "ledger_sha256": ledger.ledger_sha256,
        "component_count": len(components),
        "components": component_rows,
        "global_stationary_law_unique": connected,
        "global_entropy_production": component_rows[0]["aggregate_entropy_production"]
        if connected
        else None,
        "AGGREGATE_DETAILED_BALANCE_RECEIPT": all_aggregate_db,
        "CHANNELWISE_DETAILED_BALANCE_RECEIPT": all_channel_db,
        "ENTROPY_PRODUCTION_DIAGNOSTIC_RECEIPT": True,
        "claim_boundary": (
            "Aggregate detailed balance can hide opposed channel affinities. "
            "Modular gearing therefore requires the channelwise receipt."
        ),
    }


def raw_24_channel_realization(
    ledger: OrientedTransitionLedger,
    *,
    sparse_output: bool = True,
) -> np.ndarray | sparse.csr_matrix:
    """Map each raw oriented label to its normalized edge-support vector."""

    validation = validate_transition_ledger(ledger)
    if not validation["ORIENTED_QUOTIENT_TRANSITION_LEDGER_RECEIPT"]:
        raise ValueError("cannot realize channels from an invalid ledger")
    rows = np.arange(len(ledger.edges), dtype=int)
    columns = np.asarray(
        [slot_index(edge.base_port_id, edge.orientation) for edge in ledger.edges],
        dtype=int,
    )
    counts = np.bincount(columns, minlength=24)
    data = np.asarray(
        [1.0 / math.sqrt(float(counts[column])) for column in columns],
        dtype=float,
    )
    result = sparse.csr_matrix((data, (rows, columns)), shape=(len(ledger.edges), 24))
    return result if sparse_output else result.toarray()


def compress_modular_channels(
    ledger: OrientedTransitionLedger,
    realization: np.ndarray | sparse.spmatrix,
    *,
    rank_tolerance: float = 1.0e-12,
) -> ChannelCompressionResult:
    """Whiten ``C`` and compute ``Omega_24`` and its invariant-space defect."""

    validation = validate_transition_ledger(ledger)
    if not validation["ORIENTED_QUOTIENT_TRANSITION_LEDGER_RECEIPT"]:
        raise ValueError("cannot compress channels from an invalid ledger")
    input_was_sparse = sparse.issparse(realization)
    dense = (
        np.asarray(realization.toarray())
        if input_was_sparse
        else np.asarray(realization)
    )
    if dense.shape != (len(ledger.edges), 24):
        raise ValueError(
            "channel realization must have shape (oriented_edge_count, 24)"
        )
    if not np.issubdtype(dense.dtype, np.number):
        raise TypeError("channel realization must be numeric")
    dense = np.asarray(dense, dtype=np.complex128)
    if not np.all(np.isfinite(dense.real)) or not np.all(np.isfinite(dense.imag)):
        raise ValueError("channel realization must be finite")
    if rank_tolerance <= 0.0:
        raise ValueError("rank_tolerance must be positive")

    gram = dense.conj().T @ dense
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    largest = float(np.max(eigenvalues)) if eigenvalues.size else 0.0
    cutoff = max(rank_tolerance * max(largest, 1.0), np.finfo(float).eps)
    numerical_rank = int(np.sum(eigenvalues > cutoff))
    full_rank = numerical_rank == 24
    edge_by_id = {edge.edge_id: edge for edge in ledger.edges}
    affinities = np.asarray(
        [
            math.log(float(edge_by_id[edge.reverse_edge_id].rate) / float(edge.rate))
            for edge in ledger.edges
        ],
        dtype=float,
    )
    realization_hash = _hash_array(dense)
    if not full_rank:
        return ChannelCompressionResult(
            edge_order=tuple(edge.edge_id for edge in ledger.edges),
            realization_sha256=realization_hash,
            input_was_sparse=input_was_sparse,
            gram=gram,
            whitened_realization=None,
            omega_24=None,
            full_column_rank=False,
            numerical_rank=numerical_rank,
            gram_condition_number=None,
            whitening_residual=None,
            closure_residual_hs=None,
            relative_closure_residual=None,
            affinities=affinities,
        )
    inverse_sqrt = eigenvectors @ np.diag(eigenvalues**-0.5) @ eigenvectors.conj().T
    whitened = dense @ inverse_sqrt
    whitening_residual = float(
        np.linalg.norm(whitened.conj().T @ whitened - np.eye(24), ord="fro")
    )
    graded = affinities[:, None] * whitened
    omega = whitened.conj().T @ graded
    leakage = graded - whitened @ omega
    closure = float(np.linalg.norm(leakage, ord="fro"))
    graded_norm = float(np.linalg.norm(graded, ord="fro"))
    relative = closure / max(graded_norm, np.finfo(float).tiny)
    condition = float(np.max(eigenvalues) / np.min(eigenvalues))
    return ChannelCompressionResult(
        edge_order=tuple(edge.edge_id for edge in ledger.edges),
        realization_sha256=realization_hash,
        input_was_sparse=input_was_sparse,
        gram=gram,
        whitened_realization=whitened,
        omega_24=omega,
        full_column_rank=True,
        numerical_rank=numerical_rank,
        gram_condition_number=condition,
        whitening_residual=whitening_residual,
        closure_residual_hs=closure,
        relative_closure_residual=relative,
        affinities=affinities,
    )


def canonical_a5_slot_permutations() -> tuple[tuple[int, ...], ...]:
    """Lift the exact twelve-port A5 action to two orientation layers."""

    return tuple(
        tuple(list(permutation) + [12 + value for value in permutation])
        for permutation in icosahedral_a5_port_permutations()
    )


def orientation_slot_permutation() -> tuple[int, ...]:
    """Return the canonical fixed-point-free plus/minus swap."""

    return tuple(list(range(12, 24)) + list(range(12)))


def a5_oriented_spectrum_diagnostics(
    ledger: OrientedTransitionLedger,
    compression: ChannelCompressionResult,
    *,
    state_order: Sequence[str],
    a5_state_permutations: Sequence[Sequence[int]],
    a5_edge_permutations: Sequence[Sequence[int]],
    tolerance: float = 1.0e-9,
) -> SymmetrySpectrumResult:
    """Audit A5/orientation covariance and classify the paired spectrum.

    The 60 state and edge permutation rows must be aligned with
    :func:`icosahedral_a5_port_permutations`.  This alignment, the full group
    law, endpoint covariance, edge-frequency covariance, channel covariance,
    and orientation reversal are recomputed rather than accepted as flags.
    """

    if tolerance <= 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    slot_actions = canonical_a5_slot_permutations()
    state_order_tuple = tuple(str(state) for state in state_order)
    if state_order_tuple != ledger.quotient_states:
        raise ValueError("state_order must exactly match ledger.quotient_states")
    edge_actions = tuple(
        tuple(int(value) for value in row) for row in a5_edge_permutations
    )
    state_actions = tuple(
        tuple(int(value) for value in row) for row in a5_state_permutations
    )
    edge_valid = _permutation_family_shape_valid(edge_actions, 60, len(ledger.edges))
    state_valid = _permutation_family_shape_valid(
        state_actions, 60, len(ledger.quotient_states)
    )
    edge_group_law = bool(edge_valid and _aligned_a5_action_group_law(edge_actions))
    state_group_law = bool(state_valid and _aligned_a5_action_group_law(state_actions))
    edge_index = {edge.edge_id: index for index, edge in enumerate(ledger.edges)}
    state_index = {state: index for index, state in enumerate(ledger.quotient_states)}
    reversal = tuple(edge_index[edge.reverse_edge_id] for edge in ledger.edges)
    orientation_valid = bool(
        sorted(reversal) == list(range(len(ledger.edges)))
        and all(reversal[reversal[index]] == index for index in range(len(reversal)))
        and all(reversal[index] != index for index in range(len(reversal)))
    )

    endpoint_covariant = bool(edge_valid and state_valid)
    reversal_commutes = bool(edge_valid and orientation_valid)
    for edge_action, state_action in zip(edge_actions, state_actions, strict=False):
        if not endpoint_covariant and not reversal_commutes:
            break
        for old_index, edge in enumerate(ledger.edges):
            mapped_edge = ledger.edges[edge_action[old_index]] if edge_valid else edge
            if endpoint_covariant:
                source = ledger.quotient_states[
                    state_action[state_index[edge.source_quotient_state]]
                ]
                target = ledger.quotient_states[
                    state_action[state_index[edge.target_quotient_state]]
                ]
                if (
                    mapped_edge.source_quotient_state != source
                    or mapped_edge.target_quotient_state != target
                ):
                    endpoint_covariant = False
            if reversal_commutes and (
                edge_action[reversal[old_index]] != reversal[edge_action[old_index]]
            ):
                reversal_commutes = False
    # Port/orientation labels must transform by the canonical slot action too.
    label_covariant = bool(edge_valid)
    if edge_valid:
        for group_index, edge_action in enumerate(edge_actions):
            slot_action = slot_actions[group_index]
            for old_index, edge in enumerate(ledger.edges):
                mapped = ledger.edges[edge_action[old_index]]
                expected_slot = slot_action[
                    slot_index(edge.base_port_id, edge.orientation)
                ]
                if slot_index(mapped.base_port_id, mapped.orientation) != expected_slot:
                    label_covariant = False
                    break
            if not label_covariant:
                break

    whitened = compression.whitened_realization
    omega = compression.omega_24
    frequency_covariance_residual = math.inf
    channel_covariance_residual = math.inf
    orientation_channel_residual = math.inf
    omega_a5_residual = math.inf
    omega_orientation_residual = math.inf
    omega_hermitian_residual = math.inf
    if edge_valid and whitened is not None and omega is not None:
        frequency_covariance_residual = max(
            float(
                np.max(
                    np.abs(
                        compression.affinities
                        - compression.affinities[np.asarray(edge_action, dtype=int)]
                    )
                )
            )
            for edge_action in edge_actions
        )
        covariance_rows = []
        for edge_action, slot_action in zip(edge_actions, slot_actions, strict=True):
            inverse_edge = np.argsort(np.asarray(edge_action, dtype=int))
            covariance_rows.append(
                np.linalg.norm(
                    whitened[inverse_edge, :]
                    - whitened[:, np.asarray(slot_action, dtype=int)],
                    ord="fro",
                )
            )
        channel_covariance_residual = float(max(covariance_rows, default=0.0))
        inverse_reversal = np.argsort(np.asarray(reversal, dtype=int))
        orientation_channel_residual = float(
            np.linalg.norm(
                whitened[inverse_reversal, :]
                - whitened[:, np.asarray(orientation_slot_permutation(), dtype=int)],
                ord="fro",
            )
        )
        slot_matrices = [_permutation_matrix(row) for row in slot_actions]
        omega_a5_residual = float(
            max(
                np.linalg.norm(matrix @ omega - omega @ matrix, ord="fro")
                for matrix in slot_matrices
            )
        )
        orientation_matrix = _permutation_matrix(orientation_slot_permutation())
        omega_orientation_residual = float(
            np.linalg.norm(
                orientation_matrix @ omega @ orientation_matrix + omega,
                ord="fro",
            )
        )
        omega_hermitian_residual = float(
            np.linalg.norm(omega - omega.conj().T, ord="fro")
        )

    hypotheses_passed = bool(
        compression.full_column_rank
        and compression.closure_residual_hs is not None
        and compression.closure_residual_hs <= tolerance
        and edge_group_law
        and state_group_law
        and endpoint_covariant
        and label_covariant
        and orientation_valid
        and reversal_commutes
        and frequency_covariance_residual <= tolerance
        and channel_covariance_residual <= tolerance
        and orientation_channel_residual <= tolerance
        and omega_a5_residual <= tolerance
        and omega_orientation_residual <= tolerance
        and omega_hermitian_residual <= tolerance
    )
    irrep_rows: list[dict[str, Any]] = []
    projector_receipt = False
    paired_spectrum_receipt = False
    if hypotheses_passed and omega is not None:
        projectors = _a5_slot_character_projectors()
        expected_dimensions = {"1": 2, "3": 6, "3_prime": 6, "5": 10}
        projector_receipt = True
        paired_spectrum_receipt = True
        for name, dimension in (("1", 1), ("3", 3), ("3_prime", 3), ("5", 5)):
            projector = projectors[name]
            eigenvalues, eigenvectors = np.linalg.eigh(projector)
            basis = eigenvectors[:, eigenvalues > 0.5]
            restricted = basis.conj().T @ omega @ basis
            spectrum = np.linalg.eigvalsh(restricted)
            omega_value = float(
                (np.mean(spectrum[dimension:]) - np.mean(spectrum[:dimension])) / 2.0
            )
            expected = np.asarray(
                [-omega_value] * dimension + [omega_value] * dimension,
                dtype=float,
            )
            spectral_residual = float(np.max(np.abs(spectrum - expected)))
            rank = int(basis.shape[1])
            row_passed = bool(
                rank == expected_dimensions[name]
                and omega_value >= -tolerance
                and spectral_residual <= tolerance
            )
            projector_receipt = projector_receipt and rank == expected_dimensions[name]
            paired_spectrum_receipt = paired_spectrum_receipt and row_passed
            irrep_rows.append(
                {
                    "irrep": name,
                    "irrep_dimension": dimension,
                    "orientation_multiplicity": 2,
                    "projector_rank": rank,
                    "omega": max(0.0, omega_value),
                    "expected_eigenvalue_pair": [
                        -max(0.0, omega_value),
                        max(0.0, omega_value),
                    ],
                    "each_sign_multiplicity": dimension,
                    "restricted_eigenvalues": [float(value) for value in spectrum],
                    "paired_degeneracy_residual": spectral_residual,
                    "passed": row_passed,
                }
            )

    action_payload = {
        "state_order": list(state_order_tuple),
        "edge_order": list(compression.edge_order),
        "state_actions": [list(row) for row in state_actions],
        "edge_actions": [list(row) for row in edge_actions],
        "slot_actions": [list(row) for row in slot_actions],
        "orientation_edge": list(reversal),
        "orientation_slot": list(orientation_slot_permutation()),
    }
    action_hash = _sha256_json(action_payload)
    report = {
        "schema": "oph.modular_gearing.a5_oriented_spectrum.v1",
        "ledger_sha256": ledger.ledger_sha256,
        "realization_sha256": compression.realization_sha256,
        "action_sha256": action_hash,
        "a5_action_count": len(slot_actions),
        "edge_action_permutations_valid": edge_valid,
        "state_action_permutations_valid": state_valid,
        "edge_action_group_law_aligned_with_ports": edge_group_law,
        "state_action_group_law_aligned_with_ports": state_group_law,
        "edge_endpoint_covariance": endpoint_covariant,
        "oriented_slot_label_covariance": label_covariant,
        "orientation_reversal_fixed_point_free_involution": orientation_valid,
        "a5_action_commutes_with_orientation_reversal": reversal_commutes,
        "maximum_edge_frequency_covariance_residual": (frequency_covariance_residual),
        "maximum_channel_intertwiner_residual": channel_covariance_residual,
        "orientation_intertwiner_residual": orientation_channel_residual,
        "Omega_24_A5_commutator_residual": omega_a5_residual,
        "Omega_24_orientation_anticommutator_residual": (omega_orientation_residual),
        "Omega_24_hermitian_residual": omega_hermitian_residual,
        "irrep_rows": irrep_rows,
        "A5_ORIENTATION_COVARIANCE_RECEIPT": hypotheses_passed,
        "A5_CHARACTER_PROJECTOR_DIMENSIONS_RECEIPT": projector_receipt,
        "A5_ORIENTED_1_3_3_5_PAIRED_SPECTRUM_RECEIPT": bool(
            hypotheses_passed and projector_receipt and paired_spectrum_receipt
        ),
        "claim_boundary": (
            "A5 fixes the paired degeneracies 1,3,3,5 under the audited "
            "hypotheses. It does not fix the four nonnegative omega values."
        ),
    }
    return SymmetrySpectrumResult(action_sha256=action_hash, report_payload=report)


def modular_gearing_firewall() -> dict[str, Any]:
    """Return the finite no-go and non-identification boundaries."""

    return {
        "schema": "oph.modular_gearing.firewall.v1",
        "CENTRAL_REGISTER_MODULAR_FLOW_TRIVIAL_NO_GO": True,
        "SYMMETRIC_IRREDUCIBLE_24_STATE_CHAIN_MODULAR_FLOW_TRIVIAL_NO_GO": True,
        "NO_A5_CANONICAL_24_CYCLE_NO_GO": True,
        "REPAIR_GENERATOR_EQUALS_MODULAR_FLOW": False,
        "RATE_SCALE_DETERMINES_MODULAR_CLOCK": False,
        "CHANNEL_COUNT_24_DETERMINES_CLOCK": False,
        "DETAILED_BALANCE_SELECTS_UNIQUE_SOURCE_LAW": False,
        "DETERMINISTIC_NORMAL_FORM_SELECTS_PROBABILITY_LAW": False,
        "firewall_statements": [
            "The commutative slot-record algebra has trivial modular automorphism.",
            "The 24 labels are channels on X, not the state space X itself.",
            "A5 equivariance supplies no canonical order-24 successor map.",
            "Dissipative repair may respect modular grading but is not modular flow.",
            "Common rate rescaling changes kinetics but leaves affinities and K fixed.",
            "Every faithful law admits reversible generators, so detailed balance alone does not derive a source law.",
        ],
    }


def build_modular_gearing_receipt(
    ledger: OrientedTransitionLedger,
    *,
    lumpability: LumpabilityEvidence | None = None,
    channel_realization: np.ndarray | sparse.spmatrix | None = None,
    state_order: Sequence[str] | None = None,
    a5_state_permutations: Sequence[Sequence[int]] | None = None,
    a5_edge_permutations: Sequence[Sequence[int]] | None = None,
    physical_source_evidence: PhysicalSourceEvidence | None = None,
    tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    """Build a fail-closed MG0--MG3 theorem/claim-tier receipt."""

    ledger_report = validate_transition_ledger(ledger)
    holonomy = fundamental_cycle_holonomy(ledger, tolerance=tolerance)
    hodge = weighted_hodge_reconstruction(ledger, tolerance=tolerance)
    entropy = entropy_production_diagnostics(ledger, tolerance=tolerance)
    lumpability_report = None if lumpability is None else lumpability.report()
    lumpability_passed = bool(
        lumpability is not None
        and lumpability.ledger_sha256 == ledger.ledger_sha256
        and lumpability.passed
    )

    compression: ChannelCompressionResult | None = None
    compression_report: dict[str, Any] | None = None
    if (
        channel_realization is not None
        and ledger_report["ORIENTED_QUOTIENT_TRANSITION_LEDGER_RECEIPT"]
    ):
        compression = compress_modular_channels(ledger, channel_realization)
        compression_report = compression.report()
    symmetry: SymmetrySpectrumResult | None = None
    symmetry_report: dict[str, Any] | None = None
    symmetry_inputs_complete = all(
        value is not None
        for value in (
            state_order,
            a5_state_permutations,
            a5_edge_permutations,
        )
    )
    if compression is not None and symmetry_inputs_complete:
        assert state_order is not None
        assert a5_state_permutations is not None
        assert a5_edge_permutations is not None
        symmetry = a5_oriented_spectrum_diagnostics(
            ledger,
            compression,
            state_order=state_order,
            a5_state_permutations=a5_state_permutations,
            a5_edge_permutations=a5_edge_permutations,
            tolerance=tolerance,
        )
        symmetry_report = symmetry.report()

    mg0 = bool(
        ledger_report["ORIENTED_QUOTIENT_TRANSITION_LEDGER_RECEIPT"]
        and ledger_report["REVERSE_PAIR_INVOLUTION_RECEIPT"]
    )
    mg1 = bool(
        mg0
        and lumpability_passed
        and holonomy.get("FUNDAMENTAL_CYCLE_HOLONOMY_RECEIPT") is True
        and hodge.get("EXACT_INTEGRABLE_MODULAR_PART_RECEIPT") is True
        and entropy.get("CHANNELWISE_DETAILED_BALANCE_RECEIPT") is True
    )
    mg2 = bool(
        mg1
        and compression is not None
        and compression.full_column_rank
        and compression.closure_residual_hs is not None
        and compression.closure_residual_hs <= tolerance
    )
    mg3 = bool(
        mg2
        and symmetry_report is not None
        and symmetry_report["A5_ORIENTED_1_3_3_5_PAIRED_SPECTRUM_RECEIPT"] is True
    )

    source_report = _physical_source_evidence_report(
        ledger,
        lumpability=lumpability,
        compression=compression,
        symmetry=symmetry,
        evidence=physical_source_evidence,
    )
    physical_mg1 = bool(mg1 and source_report["PHYSICAL_RATE_SOURCE_RECEIPT"])
    physical_mg2 = bool(
        mg2 and source_report["PHYSICAL_CHANNEL_REALIZATION_SOURCE_RECEIPT"]
    )
    physical_mg3 = bool(mg3 and source_report["PHYSICAL_A5_ACTION_SOURCE_RECEIPT"])
    blockers: list[str] = []
    if not mg0:
        blockers.append("MG0_oriented_ledger_invalid")
    if not lumpability_passed:
        blockers.append("computed_strong_quotient_lumpability_missing_or_failed")
    if not mg1:
        blockers.append("MG1_finite_reversible_gearing_not_established")
    if not mg2:
        blockers.append("MG2_full_rank_invariant_24_channel_carrier_not_established")
    if not mg3:
        blockers.append("MG3_A5_oriented_paired_spectrum_not_established")
    blockers.extend(source_report["physical_promotion_blockers"])

    return {
        "schema": "oph.modular_gearing.claim_tiers.v1",
        "instrument": "FINITE_MODULAR_GEARING_MG0_MG3",
        "ledger_sha256": ledger.ledger_sha256,
        "firewall": modular_gearing_firewall(),
        "ledger": ledger_report,
        "lumpability": lumpability_report,
        "holonomy": holonomy,
        "weighted_hodge": hodge,
        "entropy_production": entropy,
        "channel_compression": compression_report,
        "a5_oriented_spectrum": symmetry_report,
        "physical_source_evidence": source_report,
        "claim_tiers": {
            "MG0_channel_diagnostic": {
                "receipt": mg0,
                "allowed_claim": "oriented quotient rates and P0..P11 x +/- labels were recorded and reverse-pair validated",
            },
            "MG1_finite_reversible_gearing": {
                "mathematical_receipt": mg1,
                "physical_receipt": physical_mg1,
                "allowed_claim": "a quotient modular potential was reconstructed within each connected sector",
            },
            "MG2_24_channel_modular_carrier": {
                "mathematical_receipt": mg2,
                "physical_receipt": physical_mg2,
                "allowed_claim": "a full-rank 24-channel subspace is invariant under the edge modular grading",
            },
            "MG3_A5_geared_spectrum": {
                "mathematical_receipt": mg3,
                "physical_receipt": physical_mg3,
                "allowed_claim": "A5/orientation covariance forces paired 1,3,3,5 degeneracies; frequencies remain dynamical",
            },
            "MG4_operationally_clocked": {
                "receipt": False,
                "allowed_claim": "not implemented or promoted by this finite module",
            },
            "MG5_BW_promoted": {
                "receipt": False,
                "allowed_claim": "not implemented or promoted by this finite module",
            },
        },
        "MG0_CHANNEL_DIAGNOSTIC_RECEIPT": mg0,
        "MG1_FINITE_REVERSIBLE_GEARING_DIAGNOSTIC_RECEIPT": mg1,
        "MG2_24_CHANNEL_CARRIER_DIAGNOSTIC_RECEIPT": mg2,
        "MG3_A5_GEARED_SPECTRUM_DIAGNOSTIC_RECEIPT": mg3,
        "PHYSICAL_MG1_FINITE_REVERSIBLE_GEARING_RECEIPT": physical_mg1,
        "PHYSICAL_MG2_24_CHANNEL_CARRIER_RECEIPT": physical_mg2,
        "PHYSICAL_MG3_A5_GEARED_SPECTRUM_RECEIPT": physical_mg3,
        "MG4_OPERATIONALLY_CLOCKED_RECEIPT": False,
        "MG5_BW_PROMOTED_RECEIPT": False,
        "ELECTROWEAK_HIERARCHY_FROM_MODULAR_GEARING_RECEIPT": False,
        "physical_promotion_blockers": sorted(set(blockers)),
        "claim_boundary": (
            "This module ends at the finite MG3 classifier. It neither identifies "
            "repair kinetics with modular flow nor derives a clock, BW promotion, "
            "or an electroweak hierarchy."
        ),
    }


def _physical_source_evidence_report(
    ledger: OrientedTransitionLedger,
    *,
    lumpability: LumpabilityEvidence | None,
    compression: ChannelCompressionResult | None,
    symmetry: SymmetrySpectrumResult | None,
    evidence: PhysicalSourceEvidence | None,
) -> dict[str, Any]:
    rate_blockers: list[str] = []
    channel_blockers: list[str] = []
    symmetry_blockers: list[str] = []
    if evidence is None:
        blockers = ["independent_physical_source_evidence_missing"]
        return {
            "provided": False,
            "PHYSICAL_RATE_SOURCE_RECEIPT": False,
            "PHYSICAL_CHANNEL_REALIZATION_SOURCE_RECEIPT": False,
            "PHYSICAL_A5_ACTION_SOURCE_RECEIPT": False,
            "physical_promotion_blockers": blockers,
        }
    digest_fields = {
        "source_dag_sha256": evidence.source_dag_sha256,
        "quotient_map_sha256": evidence.quotient_map_sha256,
        "rate_observation_bundle_sha256": evidence.rate_observation_bundle_sha256,
        "independent_verifier_receipt_sha256": (
            evidence.independent_verifier_receipt_sha256
        ),
        "lumpability_evidence_sha256": evidence.lumpability_evidence_sha256,
        "channel_realization_sha256": evidence.channel_realization_sha256,
        "symmetry_action_sha256": evidence.symmetry_action_sha256,
    }
    for name, value in digest_fields.items():
        if not _SHA256_RE.fullmatch(value):
            rate_blockers.append(f"invalid_{name}")
    if not evidence.artifact_paths or any(not path for path in evidence.artifact_paths):
        rate_blockers.append("source_artifact_paths_missing")
    if evidence.rate_provenance_kind != "source_derived_aggregate_quotient_ctmc":
        rate_blockers.append(
            "rate_provenance_is_not_source_derived_aggregate_quotient_ctmc"
        )
    if evidence.forbidden_target_dependency_paths:
        rate_blockers.append("forbidden_target_dependency_path_present")
    if not evidence.worker_attempts_excluded:
        rate_blockers.append("worker_attempts_not_excluded_from_rates")
    if not evidence.execution_counters_excluded_from_clock:
        rate_blockers.append("execution_counters_not_excluded_from_clock")
    edge_source_hashes = {edge.source_dag_hash for edge in ledger.edges}
    if edge_source_hashes != {evidence.source_dag_sha256}:
        rate_blockers.append("ledger_edge_source_dag_hash_mismatch")
    if lumpability is None or not lumpability.passed:
        rate_blockers.append("computed_lumpability_evidence_missing_or_failed")
    elif evidence.lumpability_evidence_sha256 != lumpability.evidence_sha256:
        rate_blockers.append("lumpability_evidence_digest_mismatch")
    if compression is None:
        channel_blockers.append("channel_realization_missing")
    elif evidence.channel_realization_sha256 != compression.realization_sha256:
        channel_blockers.append("channel_realization_digest_mismatch")
    if symmetry is None:
        symmetry_blockers.append("symmetry_action_evidence_missing")
    elif evidence.symmetry_action_sha256 != symmetry.action_sha256:
        symmetry_blockers.append("symmetry_action_digest_mismatch")
    if any(
        edge.rate_semantics != "aggregate_quotient_continuous_time_rate"
        for edge in ledger.edges
    ):
        rate_blockers.append("ledger_contains_nonaggregate_or_fixture_rate_semantics")

    source_tower_validation: dict[str, Any] = {
        "passed": False,
        "blockers": ["common_source_tower_report_path_missing"],
    }
    recomputed_source_tower: Mapping[str, Any] = {}
    if evidence.common_source_tower_report_path:
        source_tower_validation = verify_common_source_tower_report_file(
            Path(evidence.common_source_tower_report_path)
        )
        recomputed_source_tower = _mapping(
            source_tower_validation.get("recomputed_report")
        )
    if source_tower_validation.get("passed") is not True:
        rate_blockers.append("strict_common_source_tower_reverification_failed")
        rate_blockers.extend(
            f"common_source:{value}"
            for value in source_tower_validation.get("blockers", ())
        )

    if recomputed_source_tower:
        if (
            evidence.source_dag_sha256
            != recomputed_source_tower.get("computed_bundle_commitment")
        ):
            rate_blockers.append("source_dag_is_not_verified_bundle_commitment")
        if (
            evidence.independent_verifier_receipt_sha256
            != recomputed_source_tower.get("verification_report_sha256")
        ):
            rate_blockers.append("independent_verifier_digest_mismatch")

        provenance = _mapping(recomputed_source_tower.get("provenance"))
        role_artifact_ids = _mapping(provenance.get("role_artifact_ids"))
        descendant_ids = {
            str(value)
            for value in provenance.get("main_source_descendant_artifact_ids", ())
        }
        artifact_rows = {
            str(row.get("artifact_id")): row
            for row in _mapping(
                recomputed_source_tower.get("artifact_verification")
            ).get("rows", ())
            if isinstance(row, Mapping) and row.get("passed") is True
        }

        def row_hashes(artifact_id: Any) -> set[str]:
            row = _mapping(artifact_rows.get(str(artifact_id)))
            return {
                str(value)
                for value in (
                    row.get("actual_sha256"),
                    row.get("decoded_value_sha256"),
                )
                if _SHA256_RE.fullmatch(str(value or ""))
            }

        quotient_artifact_id = role_artifact_ids.get("physical_coarse_maps")
        if evidence.quotient_map_sha256 not in row_hashes(quotient_artifact_id):
            rate_blockers.append("quotient_map_not_bound_to_verified_physical_coarse_map")
        modular_artifact_id = role_artifact_ids.get("modular_data")
        if evidence.rate_observation_bundle_sha256 not in row_hashes(
            modular_artifact_id
        ):
            rate_blockers.append("rate_bundle_not_bound_to_verified_modular_data")

        descendant_rows = {
            artifact_id: row
            for artifact_id, row in artifact_rows.items()
            if artifact_id in descendant_ids
        }
        descendant_hashes = {
            digest
            for artifact_id in descendant_rows
            for digest in row_hashes(artifact_id)
        }
        if evidence.channel_realization_sha256 not in descendant_hashes:
            channel_blockers.append(
                "channel_realization_not_bound_to_verified_source_descendant"
            )
        if evidence.symmetry_action_sha256 not in descendant_hashes:
            symmetry_blockers.append(
                "symmetry_action_not_bound_to_verified_source_descendant"
            )
        verified_paths = {
            str(row.get("resolved_relative_path"))
            for row in descendant_rows.values()
            if row.get("resolved_relative_path")
        }
        declared_paths = set(evidence.artifact_paths)
        if not declared_paths <= verified_paths:
            rate_blockers.append("artifact_paths_include_unverified_source_files")

    rate_receipt = not rate_blockers
    channel_receipt = bool(rate_receipt and not channel_blockers)
    symmetry_receipt = bool(channel_receipt and not symmetry_blockers)
    blockers = sorted(set(rate_blockers + channel_blockers + symmetry_blockers))
    return {
        "provided": True,
        "rate_provenance_kind": evidence.rate_provenance_kind,
        "artifact_paths": list(evidence.artifact_paths),
        "common_source_tower_report_path": evidence.common_source_tower_report_path,
        "common_source_tower_validation": {
            "passed": source_tower_validation.get("passed") is True,
            "blockers": list(source_tower_validation.get("blockers") or []),
            "computed_bundle_commitment": recomputed_source_tower.get(
                "computed_bundle_commitment"
            ),
            "verification_report_sha256": recomputed_source_tower.get(
                "verification_report_sha256"
            ),
        },
        "digests": digest_fields,
        "PHYSICAL_RATE_SOURCE_RECEIPT": rate_receipt,
        "PHYSICAL_CHANNEL_REALIZATION_SOURCE_RECEIPT": channel_receipt,
        "PHYSICAL_A5_ACTION_SOURCE_RECEIPT": symmetry_receipt,
        "physical_promotion_blockers": blockers,
        "claim_boundary": (
            "Physical MG promotion requires strict on-disk replay of the common-source "
            "tower plus exact role/descendant artifact bindings. Caller-authored "
            "booleans, plausible hashes, and filenames cannot make fixtures physical."
        ),
    }


def _canonical_reverse_pairs(
    ledger: OrientedTransitionLedger,
) -> tuple[OrientedRepairEdge, ...]:
    return tuple(
        sorted(
            (edge for edge in ledger.edges if edge.orientation == "+"),
            key=lambda edge: edge.edge_id,
        )
    )


def _spanning_forest(
    states: Sequence[str],
    pairs: Sequence[OrientedRepairEdge],
) -> dict[str, tuple[OrientedRepairEdge, ...]]:
    parent = {state: state for state in states}
    rank = {state: 0 for state in states}

    def find(state: str) -> str:
        while parent[state] != state:
            parent[state] = parent[parent[state]]
            state = parent[state]
        return state

    def union(left: str, right: str) -> bool:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return False
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1
        return True

    tree: list[OrientedRepairEdge] = []
    chords: list[OrientedRepairEdge] = []
    for edge in sorted(pairs, key=lambda item: item.edge_id):
        if edge.source_quotient_state != edge.target_quotient_state and union(
            edge.source_quotient_state, edge.target_quotient_state
        ):
            tree.append(edge)
        else:
            chords.append(edge)
    return {"tree_pairs": tuple(tree), "chord_pairs": tuple(chords)}


def _undirected_components(
    states: Sequence[str],
    pairs: Sequence[OrientedRepairEdge],
) -> tuple[tuple[str, ...], ...]:
    adjacency: dict[str, set[str]] = {state: set() for state in states}
    for edge in pairs:
        adjacency[edge.source_quotient_state].add(edge.target_quotient_state)
        adjacency[edge.target_quotient_state].add(edge.source_quotient_state)
    components: list[tuple[str, ...]] = []
    remaining = set(states)
    while remaining:
        root = min(remaining)
        queue = [root]
        visited = {root}
        while queue:
            source = queue.pop()
            for target in sorted(adjacency[source]):
                if target not in visited:
                    visited.add(target)
                    queue.append(target)
        remaining -= visited
        components.append(tuple(sorted(visited)))
    return tuple(components)


def _positive_finite_rate(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, Fraction):
        return value > 0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(float(value) > 0.0 and math.isfinite(float(value)))
    return False


def _fraction_or_none(value: object) -> Fraction | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value), 1)
    return None


def _stable_rate(value: int | float | Fraction) -> dict[str, Any]:
    exact = _fraction_or_none(value)
    if exact is not None:
        return {
            "kind": "rational",
            "numerator": exact.numerator,
            "denominator": exact.denominator,
        }
    return {"kind": "float64", "hex": float(value).hex()}


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _hash_array(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(b"oph.modular_gearing.array.v1\0")
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(values.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(values.tobytes())
    return "sha256:" + digest.hexdigest()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _matrix_json(matrix: np.ndarray, *, tolerance: float = 1.0e-13) -> object:
    values = np.asarray(matrix)
    if np.max(np.abs(values.imag), initial=0.0) <= tolerance:
        return values.real.tolist()
    return {"real": values.real.tolist(), "imag": values.imag.tolist()}


def _compose_permutations(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(left[int(right[index])]) for index in range(len(left)))


def _inverse_permutation(permutation: Sequence[int]) -> tuple[int, ...]:
    inverse = [0] * len(permutation)
    for source, target in enumerate(permutation):
        inverse[int(target)] = source
    return tuple(inverse)


def _permutation_order(permutation: Sequence[int]) -> int:
    identity = tuple(range(len(permutation)))
    value = identity
    for order in range(1, 121):
        value = _compose_permutations(permutation, value)
        if value == identity:
            return order
    raise ValueError("permutation order exceeds A5 bound")


def _generated_subgroup(
    generators: Sequence[Sequence[int]],
) -> set[tuple[int, ...]]:
    identity = tuple(range(len(generators[0])))
    seen = {identity}
    frontier = [identity]
    while frontier:
        current = frontier.pop()
        for generator in generators:
            candidate = _compose_permutations(generator, current)
            if candidate not in seen:
                seen.add(candidate)
                frontier.append(candidate)
    return seen


def _a5_generator_indices() -> tuple[int, int]:
    ports = icosahedral_a5_port_permutations()
    for left_index, left in enumerate(ports):
        if _permutation_order(left) != 3:
            continue
        for right_index, right in enumerate(ports):
            if (
                _permutation_order(right) == 5
                and len(_generated_subgroup((left, right))) == 60
            ):
                return left_index, right_index
    raise AssertionError("canonical icosahedral action lacks A5 generators")


def _permutation_family_shape_valid(
    family: Sequence[Sequence[int]], expected_count: int, degree: int
) -> bool:
    return bool(
        len(family) == expected_count
        and all(
            len(row) == degree and sorted(row) == list(range(degree)) for row in family
        )
        and len({tuple(row) for row in family}) == expected_count
    )


def _aligned_a5_action_group_law(
    action_family: Sequence[Sequence[int]],
) -> bool:
    """Verify the supplied action against canonical A5 words in two generators."""

    if len(action_family) != 60:
        return False
    ports = icosahedral_a5_port_permutations()
    port_to_index = {row: index for index, row in enumerate(ports)}
    identity_port = tuple(range(12))
    identity_action = tuple(range(len(action_family[0])))
    identity_index = port_to_index.get(identity_port)
    if (
        identity_index is None
        or tuple(action_family[identity_index]) != identity_action
    ):
        return False
    generator_indices = _a5_generator_indices()
    generated: dict[tuple[int, ...], tuple[int, ...]] = {identity_port: identity_action}
    frontier = [identity_port]
    while frontier:
        current_port = frontier.pop()
        current_action = generated[current_port]
        for generator_index in generator_indices:
            next_port = _compose_permutations(ports[generator_index], current_port)
            next_action = _compose_permutations(
                action_family[generator_index], current_action
            )
            if next_port in generated:
                if generated[next_port] != next_action:
                    return False
            else:
                generated[next_port] = next_action
                frontier.append(next_port)
    return bool(
        len(generated) == 60
        and all(
            generated[port] == tuple(action_family[index])
            for port, index in port_to_index.items()
        )
    )


def _permutation_matrix(permutation: Sequence[int]) -> np.ndarray:
    matrix = np.zeros((len(permutation), len(permutation)), dtype=float)
    matrix[np.asarray(permutation, dtype=int), np.arange(len(permutation))] = 1.0
    return matrix


def _a5_slot_character_projectors() -> dict[str, np.ndarray]:
    ports = icosahedral_a5_port_permutations()
    slots = canonical_a5_slot_permutations()
    orders = [_permutation_order(row) for row in ports]
    order_five_indices = [index for index, order in enumerate(orders) if order == 5]
    seed_index = min(order_five_indices)
    seed = ports[seed_index]
    class_five_a = {
        _compose_permutations(
            _compose_permutations(group_element, seed),
            _inverse_permutation(group_element),
        )
        for group_element in ports
    }
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    phi_conjugate = (1.0 - math.sqrt(5.0)) / 2.0
    characters = {
        "1": (1, {1: 1.0, 2: 1.0, 3: 1.0, "5a": 1.0, "5b": 1.0}),
        "3": (3, {1: 3.0, 2: -1.0, 3: 0.0, "5a": phi, "5b": phi_conjugate}),
        "3_prime": (3, {1: 3.0, 2: -1.0, 3: 0.0, "5a": phi_conjugate, "5b": phi}),
        "5": (5, {1: 5.0, 2: 1.0, 3: -1.0, "5a": 0.0, "5b": 0.0}),
    }
    matrices = [_permutation_matrix(row) for row in slots]
    projectors: dict[str, np.ndarray] = {}
    for name, (dimension, character) in characters.items():
        projector = np.zeros((24, 24), dtype=float)
        for index, (port, matrix) in enumerate(zip(ports, matrices, strict=True)):
            order = orders[index]
            key: int | str = order
            if order == 5:
                key = "5a" if port in class_five_a else "5b"
            projector += float(character[key]) * matrix
        projector *= dimension / 60.0
        projectors[name] = (projector + projector.T) / 2.0
    return projectors
