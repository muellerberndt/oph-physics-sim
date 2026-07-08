from __future__ import annotations

from dataclasses import dataclass

from .optical_module import OpticalModuleLedger
from .receipts import fail, pass_report


@dataclass(frozen=True)
class LineFanPeak:
    label: str
    energy: float
    intensity: float
    gate_slope: float
    polarization: str
    tau: str
    total_charge: float
    eta: str = ""


def line_fan_from_module(module: OpticalModuleLedger, *, gate_coupling: float = 1.0) -> dict:
    if not module.quotient_descended_operators:
        return fail("OPTICAL_OPERATOR_UNCERTIFIED")
    peaks = [
        LineFanPeak(
            label=sector.label,
            energy=sector.energy,
            intensity=sector.intensity,
            gate_slope=gate_coupling * sector.total_charge,
            polarization=sector.polarization,
            tau=sector.tau,
            total_charge=sector.total_charge,
            eta=sector.eta,
        )
        for sector in module.sectors
    ]
    return {
        "status": "pass",
        "OPTICAL_OPERATOR_CERTIFIED": True,
        "LINE_FAN_DECOMPOSITION": True,
        "peaks": [peak.__dict__ for peak in peaks],
    }


def fractional_optical_slope_certificate(
    peak: LineFanPeak,
    *,
    binding_derivative_bound: float | None,
    gate_coupling: float = 1.0,
    tolerance: float = 1e-9,
) -> dict:
    if binding_derivative_bound is None:
        return fail("BINDING_DRIFT_UNBOUNDED", details={"peak": peak.__dict__})
    predicted = gate_coupling * peak.total_charge
    residual = abs(float(peak.gate_slope) - predicted)
    passed = residual <= float(binding_derivative_bound) + tolerance
    return pass_report(
        receipts={"BINDING_DRIFT_BOUNDED": passed},
        details={
            "peak": peak.__dict__,
            "predicted_charge_slope": predicted,
            "binding_derivative_bound": binding_derivative_bound,
            "residual": residual,
            "neutral_fractional_shadow": bool(abs(peak.total_charge) <= tolerance and peak.tau != "1"),
        },
    )
