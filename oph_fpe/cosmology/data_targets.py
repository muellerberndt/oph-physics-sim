from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MeasurementTarget:
    name: str
    lane: str
    source_url: str
    status: str
    claim_boundary: str

    def as_jsonable(self) -> dict[str, Any]:
        return asdict(self)


TARGETS = {
    "sparc": MeasurementTarget(
        name="SPARC",
        lane="galaxy_proxy",
        source_url="https://astroweb.case.edu/SPARC/",
        status="external_table_required",
        claim_boundary="RAR/BTFR target metadata only; no SPARC rows are bundled with this package",
    ),
    "planck2018_low_l_tt": MeasurementTarget(
        name="Planck2018_low_l_TT_shape",
        lane="cosmo_proxy",
        source_url="https://arxiv.org/abs/1807.06209",
        status="external_table_required",
        claim_boundary="shape-only benchmark metadata; use a local binned TT table for comparisons",
    ),
    "act_dr6_lensing": MeasurementTarget(
        name="ACT_DR6_CMB_lensing_shape",
        lane="cosmo_proxy",
        source_url="https://arxiv.org/abs/2304.05202",
        status="external_table_required",
        claim_boundary="late-structure target metadata only; no lensing table is bundled",
    ),
    "desi_dr2_bao": MeasurementTarget(
        name="DESI_DR2_BAO",
        lane="background_adapter",
        source_url="https://arxiv.org/abs/2503.14738",
        status="adapter_not_enabled",
        claim_boundary="future background-adapter metadata only",
    ),
    "pantheon_plus": MeasurementTarget(
        name="PantheonPlus",
        lane="background_adapter",
        source_url="https://arxiv.org/abs/2202.04077",
        status="adapter_not_enabled",
        claim_boundary="future supernova-distance adapter metadata only",
    ),
}


def measurement_target(name: str) -> MeasurementTarget:
    key = str(name).lower().replace("-", "_")
    if key not in TARGETS:
        raise KeyError(f"unknown measurement target: {name}")
    return TARGETS[key]


def target_registry() -> dict[str, dict[str, Any]]:
    return {name: target.as_jsonable() for name, target in TARGETS.items()}
