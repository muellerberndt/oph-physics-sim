from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


def _frozen_mapping(mapping: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True)
class FractionalMaterialPresentation:
    """Finite fractional material presentation F_{x,r}."""

    material_id: str
    regulator: int
    representatives: tuple[str, ...]
    quotient_map: Mapping[str, str] = field(default_factory=dict)
    repair_descended: bool = True
    repair_terminating: bool = True
    repair_locally_confluent: bool = True
    repair_complete: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "representatives", tuple(self.representatives))
        object.__setattr__(self, "quotient_map", _frozen_mapping(self.quotient_map))

    def canonicalize(self, representative: str) -> str:
        return self.quotient_map.get(representative, representative)

    @property
    def quotient_sectors(self) -> tuple[str, ...]:
        return tuple(sorted({self.canonicalize(rep) for rep in self.representatives}))

    @property
    def material_quotient_normal_form_receipt(self) -> bool:
        return all(
            (
                self.repair_descended,
                self.repair_terminating,
                self.repair_locally_confluent,
                self.repair_complete,
            )
        )

    def to_report(self) -> dict:
        return {
            "schema": "oph_fractional_material_presentation_v1",
            "material_id": self.material_id,
            "regulator": self.regulator,
            "representative_count": len(self.representatives),
            "quotient_sectors": list(self.quotient_sectors),
            "material_quotient_normal_form_receipt": self.material_quotient_normal_form_receipt,
            "claim_boundary": (
                "Finite fractional material quotient presentation. It classifies quotient sectors "
                "only after the source law or Hamiltonian is supplied."
            ),
        }
