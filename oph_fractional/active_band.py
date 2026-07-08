from __future__ import annotations

from dataclasses import dataclass

from .receipts import fail, pass_report


@dataclass(frozen=True)
class ChernBandCertificate:
    active_band_projector: bool
    chern_number: int | float
    chern_stable: bool
    band_geometry_bound: float | None = None


def chern_integer_stability(cert: ChernBandCertificate) -> dict:
    integer_chern = isinstance(cert.chern_number, int) or float(cert.chern_number).is_integer()
    passed = bool(cert.active_band_projector and cert.chern_stable and integer_chern)
    if not passed:
        return fail(
            "CHERN_NUMBER_UNSTABLE",
            details={
                "active_band_projector": cert.active_band_projector,
                "chern_number": cert.chern_number,
                "chern_stable": cert.chern_stable,
            },
        )
    return pass_report(
        receipts={
            "ACTIVE_BAND_PROJECTOR": True,
            "CHERN_NUMBER": True,
            "BAND_GEOMETRY": cert.band_geometry_bound is not None,
        },
        details={"chern_number": int(cert.chern_number), "band_geometry_bound": cert.band_geometry_bound},
    )
