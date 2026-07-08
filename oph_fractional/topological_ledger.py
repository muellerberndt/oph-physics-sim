from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class TopologicalLedger:
    sectors: tuple[str, ...]
    charges: Mapping[str, Fraction | float]
    spins: Mapping[str, Fraction | float] = field(default_factory=dict)
    fusion: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    k_matrix: tuple[tuple[int, ...], ...] | None = None
    t_vector: tuple[int, ...] | None = None
    edge_spectrum_certified: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "sectors", tuple(self.sectors))
        object.__setattr__(self, "charges", MappingProxyType(dict(self.charges)))
        object.__setattr__(self, "spins", MappingProxyType(dict(self.spins)))
        object.__setattr__(self, "fusion", MappingProxyType({k: tuple(v) for k, v in self.fusion.items()}))

    def charge(self, sector: str) -> Fraction | float:
        return self.charges[sector]

    def to_report(self) -> dict:
        return {
            "schema": "oph_fractional_topological_ledger_v1",
            "sectors": list(self.sectors),
            "charges": {key: str(value) for key, value in self.charges.items()},
            "spins": {key: str(value) for key, value in self.spins.items()},
            "fusion": {key: list(value) for key, value in self.fusion.items()},
            "TOPOLOGICAL_SECTOR_LEDGER": True,
            "EDGE_SPECTRUM": self.edge_spectrum_certified,
        }


def _inverse_1x1_or_2x2(matrix: tuple[tuple[int, ...], ...]) -> tuple[tuple[Fraction, ...], ...]:
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return ((Fraction(1, matrix[0][0]),),)
    if len(matrix) == 2 and all(len(row) == 2 for row in matrix):
        a, b = matrix[0]
        c, d = matrix[1]
        det = a * d - b * c
        if det == 0:
            raise ValueError("singular K matrix")
        return ((Fraction(d, det), Fraction(-b, det)), (Fraction(-c, det), Fraction(a, det)))
    raise ValueError("only 1x1 and 2x2 K matrices are supported by this receipt helper")


def _quadratic(left: tuple[int, ...], matrix: tuple[tuple[Fraction, ...], ...], right: tuple[int, ...]) -> Fraction:
    return sum(Fraction(left[i]) * matrix[i][j] * Fraction(right[j]) for i in range(len(left)) for j in range(len(right)))


def abelian_k_matrix_readout(
    k_matrix: tuple[tuple[int, ...], ...],
    t_vector: tuple[int, ...],
    ell: tuple[int, ...],
) -> dict:
    inverse = _inverse_1x1_or_2x2(k_matrix)
    filling = _quadratic(t_vector, inverse, t_vector)
    charge = _quadratic(t_vector, inverse, ell)
    self_statistics_over_pi = _quadratic(ell, inverse, ell)
    return {
        "filling": str(filling),
        "quasiparticle_charge_over_e": str(charge),
        "self_statistics_over_pi": str(self_statistics_over_pi),
        "K_MATRIX_READOUT": True,
    }
