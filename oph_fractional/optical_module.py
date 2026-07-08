from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class OpticalSector:
    label: str
    tau: str
    total_charge: float
    energy: float
    intensity: float
    polarization: str
    binding_derivative_bound: float | None = None
    eta: str = ""


@dataclass(frozen=True)
class OpticalModuleLedger:
    sectors: tuple[OpticalSector, ...]
    quotient_descended_operators: bool
    sector_to_topological_shadow: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sectors", tuple(self.sectors))
        if not self.sector_to_topological_shadow:
            mapping = {sector.label: sector.tau for sector in self.sectors}
        else:
            mapping = dict(self.sector_to_topological_shadow)
        object.__setattr__(self, "sector_to_topological_shadow", MappingProxyType(mapping))

    def to_report(self) -> dict:
        return {
            "schema": "oph_fractional_optical_module_v1",
            "OPTICAL_OPERATOR_CERTIFIED": self.quotient_descended_operators,
            "sectors": [
                {
                    "label": sector.label,
                    "tau": sector.tau,
                    "total_charge": sector.total_charge,
                    "energy": sector.energy,
                    "intensity": sector.intensity,
                    "polarization": sector.polarization,
                    "binding_derivative_bound": sector.binding_derivative_bound,
                    "eta": sector.eta,
                }
                for sector in self.sectors
            ],
        }
