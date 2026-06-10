from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


FINITE_CMB_SOURCES = {
    "finite_lattice",
    "finite_repair_transition_clock",
    "parent_collar_finite_difference",
    "neutral_bulk_freezeout",
    "scale_compressed_24_round_finite_ladder",
}

THEOREM_SIDE_SOURCES = {
    "OPH_pixel_branch_predeclared",
    "OPH_screen_capacity_branch_predeclared",
}


@dataclass
class PhysicalCMBInputContract:
    no_data_use_receipt: bool

    P_source: str
    N_source: str

    eta_R_source: str
    eta_R_value: float | None

    A_zeta_source: str
    A_zeta_value: float | None

    q_IR_source: str
    q_IR_value: float | None

    ell_IR_source: str
    ell_IR_value: float | None

    B_A_source: str
    B_A_k_a: np.ndarray | None

    Gamma_rec_source: str
    Gamma_rec_k_a: np.ndarray | None

    rho_A_source: str
    rho_A_a: np.ndarray | None

    freezeout_source: str
    freezeout_surface: dict[str, Any] | None

    official_likelihood_ready: bool
    cdm_limit_regression_passed: bool


def validate_physical_cmb_contract(contract: PhysicalCMBInputContract) -> dict[str, Any]:
    """Validate the hard gate for promoting CMB diagnostics to predictions.

    The simulator may emit measurement-comparable CMB curves before this
    contract passes. Those curves remain diagnostics. A physical CMB prediction
    requires finite-derived OPH inputs, an explicit no-data-use receipt, a
    CDM-limit regression, and an official likelihood-ready path.
    """

    blockers: list[str] = []

    if not bool(contract.no_data_use_receipt):
        blockers.append("no_data_use_receipt_false")

    if str(contract.eta_R_source) not in FINITE_CMB_SOURCES or not _finite_scalar(contract.eta_R_value):
        blockers.append("eta_R_not_finite_derived")

    if str(contract.A_zeta_source) not in FINITE_CMB_SOURCES or not _finite_positive_scalar(contract.A_zeta_value):
        blockers.append("A_zeta_not_finite_derived")

    if str(contract.q_IR_source) not in FINITE_CMB_SOURCES or not _finite_scalar(contract.q_IR_value):
        blockers.append("q_IR_not_finite_derived")

    if str(contract.ell_IR_source) not in FINITE_CMB_SOURCES or not _finite_positive_scalar(contract.ell_IR_value):
        blockers.append("ell_IR_not_finite_derived")

    if str(contract.B_A_source) not in FINITE_CMB_SOURCES or not _finite_array(contract.B_A_k_a):
        blockers.append("B_A_k_a_missing_or_not_finite")

    if str(contract.Gamma_rec_source) not in FINITE_CMB_SOURCES or not _finite_array(contract.Gamma_rec_k_a):
        blockers.append("Gamma_rec_k_a_missing_or_not_finite")

    if str(contract.rho_A_source) not in FINITE_CMB_SOURCES or not _finite_array(contract.rho_A_a):
        blockers.append("rho_A_missing_or_not_finite")

    if str(contract.freezeout_source) not in FINITE_CMB_SOURCES or not isinstance(contract.freezeout_surface, dict):
        blockers.append("freezeout_missing_or_not_finite")

    if not bool(contract.cdm_limit_regression_passed):
        blockers.append("cdm_limit_regression_not_passed")

    if not bool(contract.official_likelihood_ready):
        blockers.append("official_likelihood_not_ready")

    receipt = len(blockers) == 0
    return {
        "PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT": receipt,
        "physical_cmb_prediction_eligible": receipt,
        "blockers": blockers,
        "finite_sources": sorted(FINITE_CMB_SOURCES),
        "theorem_side_sources_allowed_as_constants": sorted(THEOREM_SIDE_SOURCES),
        "claim_boundary": (
            "Hard input contract for physical CMB prediction. Measurement-comparable TT curves remain "
            "diagnostics until every blocker is cleared by finite-derived inputs and likelihood plumbing."
        ),
    }


def contract_from_reports(
    *,
    no_data_use_receipt: bool,
    repair_clock_report: dict[str, Any],
    ba_kernel_report: dict[str, Any],
    scalar_release_report: dict[str, Any],
    freezeout_report: dict[str, Any],
    background_report: dict[str, Any],
    likelihood_report: dict[str, Any],
) -> PhysicalCMBInputContract:
    """Build a contract from current report dictionaries.

    This intentionally does not infer physical readiness from loose diagnostic
    rows. Reports must expose explicit source labels and finite arrays.
    """

    return PhysicalCMBInputContract(
        no_data_use_receipt=bool(no_data_use_receipt),
        P_source=str(background_report.get("P_source", "OPH_pixel_branch_predeclared")),
        N_source=str(background_report.get("N_source", "unknown")),
        eta_R_source=str(repair_clock_report.get("eta_R_source", repair_clock_report.get("source", "unknown"))),
        eta_R_value=_optional_float(repair_clock_report.get("eta_R_value", repair_clock_report.get("eta_R"))),
        A_zeta_source=str(scalar_release_report.get("A_zeta_source", scalar_release_report.get("source", "unknown"))),
        A_zeta_value=_optional_float(scalar_release_report.get("A_zeta_value", scalar_release_report.get("A_zeta"))),
        q_IR_source=str(freezeout_report.get("q_IR_source", freezeout_report.get("source", "unknown"))),
        q_IR_value=_optional_float(freezeout_report.get("q_IR_value", freezeout_report.get("q_IR"))),
        ell_IR_source=str(freezeout_report.get("ell_IR_source", freezeout_report.get("source", "unknown"))),
        ell_IR_value=_optional_float(freezeout_report.get("ell_IR_value", freezeout_report.get("ell_IR"))),
        B_A_source=str(ba_kernel_report.get("B_A_source", ba_kernel_report.get("source", "unknown"))),
        B_A_k_a=_optional_array(ba_kernel_report.get("B_A_k_a", ba_kernel_report.get("B_A_grid"))),
        Gamma_rec_source=str(
            repair_clock_report.get("Gamma_rec_source", repair_clock_report.get("source", "unknown"))
        ),
        Gamma_rec_k_a=_optional_array(repair_clock_report.get("Gamma_rec_k_a", repair_clock_report.get("Gamma_rec_grid"))),
        rho_A_source=str(background_report.get("rho_A_source", background_report.get("source", "unknown"))),
        rho_A_a=_optional_array(background_report.get("rho_A_a", background_report.get("rho_A_grid"))),
        freezeout_source=str(freezeout_report.get("freezeout_source", freezeout_report.get("source", "unknown"))),
        freezeout_surface=freezeout_report.get("freezeout_surface"),
        official_likelihood_ready=bool(likelihood_report.get("official_likelihood_ready", False)),
        cdm_limit_regression_passed=bool(likelihood_report.get("cdm_limit_regression_passed", False)),
    )


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _optional_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    return array if array.size and np.all(np.isfinite(array)) else None


def _finite_scalar(value: float | None) -> bool:
    return value is not None and bool(np.isfinite(float(value)))


def _finite_positive_scalar(value: float | None) -> bool:
    return _finite_scalar(value) and float(value) > 0.0


def _finite_array(value: np.ndarray | None) -> bool:
    if value is None:
        return False
    array = np.asarray(value, dtype=float)
    return bool(array.size and np.all(np.isfinite(array)))
