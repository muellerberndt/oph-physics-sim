from __future__ import annotations

from dataclasses import dataclass

from .receipts import fail, pass_report


@dataclass(frozen=True)
class ManyBodyCertificate:
    gap: float
    ground_sector_degeneracy: int
    expected_degeneracy: int
    flux_pump_charge: float | None = None
    hall_conductance: float | None = None


def manybody_gap_certificate(cert: ManyBodyCertificate) -> dict:
    if cert.gap <= 0:
        return fail("NO_GAP_CERTIFICATE", details={"gap": cert.gap})
    return pass_report(
        receipts={
            "MANYBODY_GAP": True,
            "GROUND_SECTOR_DEGENERACY": cert.ground_sector_degeneracy == cert.expected_degeneracy,
            "FLUX_INSERTION_PUMP": cert.flux_pump_charge is not None,
            "HALL_CONDUCTANCE": cert.hall_conductance is not None,
        },
        details={
            "gap": cert.gap,
            "ground_sector_degeneracy": cert.ground_sector_degeneracy,
            "expected_degeneracy": cert.expected_degeneracy,
            "flux_pump_charge": cert.flux_pump_charge,
            "hall_conductance": cert.hall_conductance,
        },
    )
