from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from types import MappingProxyType
from typing import Mapping

from .receipts import fail, pass_report


@dataclass(frozen=True)
class SourceLaw:
    action: Mapping[str, float] = field(default_factory=dict)
    base_weight: Mapping[str, float] = field(default_factory=dict)
    frozen: bool = False
    tag: str = "SOURCE_LAW_REQUIRED"

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", MappingProxyType(dict(self.action)))
        object.__setattr__(self, "base_weight", MappingProxyType(dict(self.base_weight)))

    def freeze(self) -> "SourceLaw":
        return SourceLaw(action=self.action, base_weight=self.base_weight, frozen=True, tag=self.tag)

    def probabilities(self) -> dict[str, float]:
        weights = {
            sector: float(self.base_weight.get(sector, 1.0)) * exp(-float(action))
            for sector, action in self.action.items()
        }
        total = sum(weights.values())
        if total <= 0:
            return {sector: 0.0 for sector in weights}
        return {sector: value / total for sector, value in weights.items()}


def require_frozen(source: SourceLaw) -> dict:
    if not source.frozen:
        return fail("SOURCE_NOT_FROZEN", details={"tag": source.tag})
    return pass_report(receipts={"SOURCE_HAMILTONIAN_FROZEN": True})


def normal_form_non_selection(candidate_sectors: list[str] | tuple[str, ...]) -> dict:
    return {
        "status": "conditional",
        "SOURCE_LAW_REQUIRED": True,
        "NORMAL_FORM_IS_NOT_SELECTOR": True,
        "candidate_sectors": list(candidate_sectors),
        "claim_boundary": (
            "Normal forms classify quotient sectors. They do not choose material sector weights "
            "without a frozen source law, Hamiltonian, transfer operator, or vacuum certificate."
        ),
    }
