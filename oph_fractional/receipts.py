"""Receipt names and fail-closed states for fractional OPH sandboxes."""

from __future__ import annotations

from typing import Any


PASS_RECEIPTS = (
    "SOURCE_HAMILTONIAN_FROZEN",
    "ACTIVE_BAND_PROJECTOR",
    "CHERN_NUMBER",
    "BAND_GEOMETRY",
    "MANYBODY_GAP",
    "GROUND_SECTOR_DEGENERACY",
    "FLUX_INSERTION_PUMP",
    "HALL_CONDUCTANCE",
    "EDGE_SPECTRUM",
    "TOPOLOGICAL_SECTOR_LEDGER",
    "REFINEMENT_STABILITY",
    "NO_TARGET_LEAK",
    "CANONICALIZER_IDEMPOTENCE",
    "REPRESENTATIVE_INVARIANCE",
    "QUOTIENT_LUMPABILITY",
    "DETAILED_BALANCE_OR_DECLARED_NONEQUILIBRIUM",
    "REFINEMENT_COMPATIBILITY",
    "NO_ORBIT_SIZE_BIAS",
)

FAIL_CLOSED_STATES = (
    "SOURCE_NOT_FROZEN",
    "NOT_QUOTIENT_INVARIANT",
    "CANONICALIZER_NOT_IDEMPOTENT",
    "KERNEL_NOT_LUMPABLE",
    "ORBIT_SIZE_BIAS_DETECTED",
    "NO_GAP_CERTIFICATE",
    "CHERN_NUMBER_UNSTABLE",
    "PHASE_CERTIFICATE_NONINJECTIVE",
    "SECTOR_AMBIGUOUS",
    "OPTICAL_OPERATOR_UNCERTIFIED",
    "BINDING_DRIFT_UNBOUNDED",
    "OPTICAL_SECTOR_AMBIGUOUS",
    "TARGET_LEAK_DETECTED",
    "REFINEMENT_DEFECT_TOO_LARGE",
    "DIAGNOSTIC_ONLY",
)


def receipt(name: str, passed: bool, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        name: bool(passed),
        "receipt": name,
        "status": "pass" if passed else "fail",
        "details": details or {},
    }


def fail(state: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    if state not in FAIL_CLOSED_STATES:
        raise ValueError(f"unknown fractional fail-closed state: {state}")
    return {
        "status": "fail",
        "fail_closed_state": state,
        "details": details or {},
    }


def pass_report(*, receipts: dict[str, bool], details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "status": "pass" if all(receipts.values()) else "fail",
        "receipts": {name: bool(value) for name, value in receipts.items()},
        "details": details or {},
    }
