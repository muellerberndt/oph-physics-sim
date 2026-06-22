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
    "OPH_screen_capacity_observed_branch_readout",
    "OPH_independent_scale_bridge_supplied",
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
    screen_to_primordial_lift_receipt: bool = False
    finite_covariant_parent_receipt: bool = False
    stress_energy_closure_receipt: bool = False
    gauge_independence_receipt: bool = False
    causal_response_receipt: bool = False
    refinement_convergence_receipt: bool = False
    explicit_recipient_stress_receipt: bool = False
    source_provenance_receipt: bool = False
    pooled_source_reducer_receipt: bool = False
    contradiction_free_provenance_receipt: bool = False
    N_CRC_consensus_invariant_receipt: bool = False
    global_likelihood_reduction_receipt: bool = False
    frozen_likelihood_protocol_receipt: bool = False
    frozen_source_hash: str | None = None
    frozen_solver_hash: str | None = None
    frozen_likelihood_hash: str | None = None


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

    if not bool(contract.source_provenance_receipt):
        blockers.append("source_provenance_receipt_missing")

    if not bool(contract.pooled_source_reducer_receipt):
        blockers.append("pooled_source_reducer_receipt_missing")

    if not bool(contract.contradiction_free_provenance_receipt):
        blockers.append("source_provenance_contradiction_check_failed")

    if not bool(contract.N_CRC_consensus_invariant_receipt):
        blockers.append("N_CRC_consensus_invariant_receipt_missing")

    if not bool(contract.global_likelihood_reduction_receipt):
        blockers.append("global_likelihood_reduction_receipt_missing")

    if str(contract.eta_R_source) not in FINITE_CMB_SOURCES or not _finite_scalar(contract.eta_R_value):
        blockers.append("eta_R_not_finite_derived")

    if str(contract.A_zeta_source) not in FINITE_CMB_SOURCES or not _finite_positive_scalar(contract.A_zeta_value):
        blockers.append("A_zeta_not_finite_derived")

    if not bool(contract.screen_to_primordial_lift_receipt):
        blockers.append("screen_to_primordial_lift_receipt_missing")

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

    if not bool(contract.finite_covariant_parent_receipt):
        blockers.append("finite_covariant_parent_receipt_missing")

    if not bool(contract.stress_energy_closure_receipt):
        blockers.append("stress_energy_closure_not_certified")

    if _array_has_positive(contract.Gamma_rec_k_a) and not bool(contract.explicit_recipient_stress_receipt):
        blockers.append("recipient_stress_missing_for_nonzero_Gamma_rec")

    if not bool(contract.gauge_independence_receipt):
        blockers.append("gauge_independence_not_certified")

    if not bool(contract.causal_response_receipt):
        blockers.append("causal_response_not_certified")

    if not bool(contract.refinement_convergence_receipt):
        blockers.append("refinement_convergence_not_certified")

    if str(contract.freezeout_source) not in FINITE_CMB_SOURCES or not isinstance(contract.freezeout_surface, dict):
        blockers.append("freezeout_missing_or_not_finite")

    if not bool(contract.cdm_limit_regression_passed):
        blockers.append("cdm_limit_regression_not_passed")

    if not bool(contract.official_likelihood_ready):
        blockers.append("official_likelihood_not_ready")

    if not bool(contract.frozen_likelihood_protocol_receipt):
        blockers.append("frozen_likelihood_protocol_not_certified")

    if not _nonempty_string(contract.frozen_source_hash):
        blockers.append("frozen_source_hash_missing")

    if not _nonempty_string(contract.frozen_solver_hash):
        blockers.append("frozen_solver_hash_missing")

    if not _nonempty_string(contract.frozen_likelihood_hash):
        blockers.append("frozen_likelihood_hash_missing")

    receipt = len(blockers) == 0
    return {
        "PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT": receipt,
        "physical_cmb_prediction_eligible": receipt,
        "blockers": blockers,
        "finite_sources": sorted(FINITE_CMB_SOURCES),
        "theorem_side_sources_allowed_as_constants": sorted(THEOREM_SIDE_SOURCES),
        "claim_boundary": (
            "Hard input contract for physical CMB prediction. Measurement-comparable TT curves remain "
            "diagnostics until every blocker is cleared by finite-derived inputs, a finite covariant "
            "stress parent, source-only provenance, pooled reducers, frozen hashes, and likelihood plumbing."
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
        screen_to_primordial_lift_receipt=bool(
            scalar_release_report.get("SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT", False)
            or scalar_release_report.get("screen_to_primordial_lift_receipt", False)
        ),
        finite_covariant_parent_receipt=bool(
            background_report.get("FINITE_COVARIANT_COLLAR_PACKET_PARENT_RECEIPT", False)
        ),
        stress_energy_closure_receipt=bool(background_report.get("STRESS_ENERGY_CLOSURE_RECEIPT", False)),
        gauge_independence_receipt=bool(background_report.get("GAUGE_INDEPENDENCE_RECEIPT", False)),
        causal_response_receipt=bool(background_report.get("CAUSAL_RESPONSE_RECEIPT", False)),
        refinement_convergence_receipt=bool(background_report.get("REFINEMENT_CONVERGENCE_RECEIPT", False)),
        explicit_recipient_stress_receipt=bool(
            background_report.get("EXPLICIT_RECIPIENT_STRESS_RECEIPT", False)
        ),
        source_provenance_receipt=bool(likelihood_report.get("CMB_SOURCE_PROVENANCE_RECEIPT", False)),
        pooled_source_reducer_receipt=bool(likelihood_report.get("pooled_source_reducer_receipt", False)),
        contradiction_free_provenance_receipt=bool(
            likelihood_report.get("contradiction_free_provenance_receipt", False)
        ),
        N_CRC_consensus_invariant_receipt=bool(
            likelihood_report.get("N_CRC_consensus_invariant_receipt", False)
        ),
        global_likelihood_reduction_receipt=bool(
            likelihood_report.get("global_likelihood_reduction_receipt", False)
        ),
        frozen_likelihood_protocol_receipt=bool(
            likelihood_report.get("FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT", False)
            or background_report.get("FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT", False)
        ),
        frozen_source_hash=background_report.get("source_hash"),
        frozen_solver_hash=likelihood_report.get("solver_hash") or background_report.get("solver_hash"),
        frozen_likelihood_hash=likelihood_report.get("likelihood_hash") or background_report.get("likelihood_hash"),
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


def _array_has_positive(value: np.ndarray | None) -> bool:
    if value is None:
        return False
    array = np.asarray(value, dtype=float)
    if not array.size or not np.all(np.isfinite(array)):
        return False
    return bool(np.any(array > 0.0))


def _nonempty_string(value: str | None) -> bool:
    return bool(isinstance(value, str) and value.strip())
