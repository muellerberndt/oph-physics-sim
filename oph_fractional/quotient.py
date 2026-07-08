from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose
from types import MappingProxyType
from typing import Callable, Mapping

from .receipts import fail, pass_report


def _freeze_nested(mapping: Mapping[str, Mapping[str, float]]) -> Mapping[str, Mapping[str, float]]:
    return MappingProxyType({key: MappingProxyType(dict(value)) for key, value in mapping.items()})


@dataclass(frozen=True)
class QuotientSchema:
    canonical: Mapping[str, str]
    orbit_sizes: Mapping[str, int] = field(default_factory=dict)
    transition_kernel: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    detailed_balance_or_declared_nonequilibrium: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "canonical", MappingProxyType(dict(self.canonical)))
        object.__setattr__(self, "orbit_sizes", MappingProxyType(dict(self.orbit_sizes)))
        object.__setattr__(self, "transition_kernel", _freeze_nested(self.transition_kernel))

    def canonicalize(self, state: str) -> str:
        return self.canonical.get(state, state)

    @property
    def states(self) -> tuple[str, ...]:
        return tuple(sorted(self.canonical))

    @property
    def sectors(self) -> tuple[str, ...]:
        return tuple(sorted({self.canonicalize(state) for state in self.states}))


def canonicalizer_idempotence(schema: QuotientSchema) -> dict:
    bad = [
        state
        for state in schema.states
        if schema.canonicalize(schema.canonicalize(state)) != schema.canonicalize(state)
    ]
    if bad:
        return fail("CANONICALIZER_NOT_IDEMPOTENT", details={"states": bad})
    return pass_report(receipts={"CANONICALIZER_IDEMPOTENCE": True})


def representative_invariance(schema: QuotientSchema, observable: Callable[[str], object]) -> dict:
    values: dict[str, object] = {}
    bad: list[dict[str, object]] = []
    for state in schema.states:
        sector = schema.canonicalize(state)
        value = observable(state)
        if sector in values and values[sector] != value:
            bad.append({"sector": sector, "state": state, "value": value, "expected": values[sector]})
        values.setdefault(sector, value)
    if bad:
        return fail("NOT_QUOTIENT_INVARIANT", details={"violations": bad})
    return pass_report(receipts={"REPRESENTATIVE_INVARIANCE": True}, details={"sector_values": values})


def quotient_lumpability(schema: QuotientSchema, *, tolerance: float = 1e-12) -> dict:
    sector_rows: dict[str, dict[str, float]] = {}
    violations: list[dict[str, object]] = []
    for state in schema.states:
        row = schema.transition_kernel.get(state, {})
        sector_row: dict[str, float] = {sector: 0.0 for sector in schema.sectors}
        for target, probability in row.items():
            sector_row[schema.canonicalize(target)] = sector_row.get(schema.canonicalize(target), 0.0) + float(probability)
        sector = schema.canonicalize(state)
        if sector in sector_rows:
            expected = sector_rows[sector]
            if any(not isclose(sector_row.get(k, 0.0), expected.get(k, 0.0), abs_tol=tolerance) for k in schema.sectors):
                violations.append({"sector": sector, "state": state, "row": sector_row, "expected": expected})
        sector_rows.setdefault(sector, sector_row)
    if violations:
        return fail("KERNEL_NOT_LUMPABLE", details={"violations": violations})
    return pass_report(
        receipts={
            "QUOTIENT_LUMPABILITY": True,
            "DETAILED_BALANCE_OR_DECLARED_NONEQUILIBRIUM": schema.detailed_balance_or_declared_nonequilibrium,
        },
        details={"sector_kernel": sector_rows},
    )


def no_orbit_size_bias(schema: QuotientSchema, sector_weights: Mapping[str, float], *, tolerance: float = 1e-12) -> dict:
    biased = []
    per_representative: dict[str, float] = {}
    for sector in schema.sectors:
        orbit_size = int(schema.orbit_sizes.get(sector, 1))
        if orbit_size <= 0:
            biased.append({"sector": sector, "reason": "nonpositive orbit size"})
            continue
        per_representative[sector] = float(sector_weights.get(sector, 0.0)) / orbit_size
    positive = [value for value in per_representative.values() if value > tolerance]
    varying_orbit_sizes = len({int(schema.orbit_sizes.get(sector, 1)) for sector in schema.sectors}) > 1
    tracks_hidden_count = bool(
        varying_orbit_sizes
        and len(positive) > 1
        and all(isclose(value, positive[0], abs_tol=tolerance) for value in positive)
    )
    if tracks_hidden_count:
        biased.append(
            {
                "reason": "sector weights are proportional to hidden orbit sizes",
                "orbit_sizes": dict(schema.orbit_sizes),
                "sector_weights": dict(sector_weights),
            }
        )
    if biased:
        return fail("ORBIT_SIZE_BIAS_DETECTED", details={"biased": biased})
    return pass_report(
        receipts={"NO_ORBIT_SIZE_BIAS": True},
        details={"orbit_sizes": dict(schema.orbit_sizes), "sector_weights": dict(sector_weights)},
    )
