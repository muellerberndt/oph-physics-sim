from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from oph_fpe.cosmology.anomaly_abundance_selector import (
    PHYSICAL_PARENT_WITH_CONDITIONAL_ABUNDANCE,
    SOURCE_ONLY_ANOMALY_ABUNDANCE,
)
from oph_fpe.cosmology.cosmological_scale_bridge import validate_physical_scale_bridge_receipts


FINITE_CMB_SOURCES = {
    "finite_lattice",
    "finite_repair_transition_clock",
    "parent_collar_finite_difference",
    "neutral_bulk_freezeout",
    "scale_compressed_24_round_finite_ladder",
}

THEOREM_SIDE_SOURCES = {
    "OPH_pixel_branch_predeclared",
    "OPH_direct_public_record_capacity",
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
    rho_A_transport_receipt: bool = False
    anomaly_abundance_source_receipt: bool = False
    rho_A_source_receipt: bool = False
    rho_A_claim_label: str = PHYSICAL_PARENT_WITH_CONDITIONAL_ABUNDANCE
    screen_to_primordial_lift_receipt: bool = False
    finite_covariant_parent_receipt: bool = False
    stress_energy_closure_receipt: bool = False
    gauge_independence_receipt: bool = False
    causal_response_receipt: bool = False
    refinement_convergence_receipt: bool = False
    explicit_recipient_stress_receipt: bool = False
    exchange_current_closure_receipt: bool = False
    physical_clock_receipt: bool = False
    active_fiber_receipt: bool = False
    conserved_sector_decomposition_receipt: bool = False
    common_parent_response_pole_receipt: bool = False
    source_provenance_receipt: bool = False
    pooled_source_reducer_receipt: bool = False
    contradiction_free_provenance_receipt: bool = False
    N_CRC_consensus_invariant_receipt: bool = False
    global_likelihood_reduction_receipt: bool = False
    frozen_likelihood_protocol_receipt: bool = False
    source_freeze_manifest_receipt: bool = False
    solver_assumption_pin_receipt: bool = False
    custom_parent_cdm_limit_regression_receipt: bool = False
    standard_model_off_regression_receipt: bool = False
    blinded_comparison_setup_receipt: bool = False
    full_observable_likelihood_receipt: bool = False
    frozen_transfer_likelihood_receipt: bool = False
    frozen_source_hash: str | None = None
    frozen_solver_hash: str | None = None
    frozen_likelihood_hash: str | None = None
    physical_scale_bridge_receipts: dict[str, Any] | None = None


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

    scale_bridge_validation = validate_physical_scale_bridge_receipts(contract.physical_scale_bridge_receipts)
    if not bool(scale_bridge_validation.get("PHYSICAL_SCALE_BRIDGE_RECEIPT", False)):
        blockers.append("physical_scale_bridge_receipt_missing")
    if not bool(scale_bridge_validation.get("PHYSICAL_SPATIAL_K_RECEIPT", False)):
        blockers.append("physical_spatial_k_receipt_missing")
    if not bool(scale_bridge_validation.get("SCREEN_TO_PHYSICAL_K_ASSOCIATION_RECEIPT", False)):
        blockers.append("screen_to_physical_k_association_receipt_missing")
    if not bool(scale_bridge_validation.get("SOURCE_ANGULAR_SECTOR_RECEIPT", False)):
        blockers.append("source_angular_sector_receipt_missing")
    if not bool(scale_bridge_validation.get("CALIBRATED_A_EVOLUTION_RECEIPT", False)):
        blockers.append("calibrated_a_evolution_receipt_missing")
    if not bool(scale_bridge_validation.get("PHYSICAL_MODE_FREEZEOUT_MAP_RECEIPT", False)):
        blockers.append("physical_mode_freezeout_map_receipt_missing")
    if not bool(scale_bridge_validation.get("PHYSICAL_FREEZEOUT_SURFACE_RECEIPT", False)):
        blockers.append("physical_freezeout_surface_receipt_missing")
    if not bool(scale_bridge_validation.get("COMMON_PRIMORDIAL_ANOMALY_MODE_BASIS_RECEIPT", False)):
        blockers.append("common_primordial_anomaly_mode_basis_receipt_missing")
    if not bool(scale_bridge_validation.get("CROSS_RECEIPT_CONSISTENCY_RECEIPT", False)):
        blockers.append("cross_receipt_consistency_receipt_missing")
    if not bool(scale_bridge_validation.get("NO_POSTHOC_CALIBRATION_RECEIPT", False)):
        blockers.append("no_posthoc_calibration_receipt_missing")

    if not bool(contract.source_provenance_receipt):
        blockers.append("source_provenance_receipt_missing")

    if not bool(contract.pooled_source_reducer_receipt):
        blockers.append("pooled_source_reducer_receipt_missing")

    if not bool(contract.contradiction_free_provenance_receipt):
        blockers.append("source_provenance_contradiction_check_failed")

    if not bool(contract.N_CRC_consensus_invariant_receipt):
        blockers.append("N_CRC_consensus_invariant_receipt_missing")

    if str(contract.N_source) != "OPH_direct_public_record_capacity":
        blockers.append("N_CRC_not_direct_public_record_capacity")

    if not bool(contract.global_likelihood_reduction_receipt):
        blockers.append("global_likelihood_reduction_receipt_missing")

    if str(contract.eta_R_source) not in FINITE_CMB_SOURCES or not _finite_scalar(contract.eta_R_value):
        blockers.append("eta_R_not_finite_derived")

    if str(contract.A_zeta_source) not in FINITE_CMB_SOURCES or not _finite_positive_scalar(contract.A_zeta_value):
        blockers.append("A_zeta_not_finite_derived")

    if not bool(contract.screen_to_primordial_lift_receipt):
        blockers.append("screen_to_primordial_lift_receipt_missing")

    if str(contract.q_IR_source) not in FINITE_CMB_SOURCES or not _finite_nonnegative_scalar(contract.q_IR_value):
        blockers.append("q_IR_not_finite_derived")

    if str(contract.ell_IR_source) not in FINITE_CMB_SOURCES or not _finite_positive_scalar(contract.ell_IR_value):
        blockers.append("ell_IR_not_finite_derived")

    if str(contract.B_A_source) not in FINITE_CMB_SOURCES or not _finite_table(contract.B_A_k_a, min_cols=3):
        blockers.append("B_A_k_a_missing_or_not_finite")

    gamma_rec_table_ready = str(contract.Gamma_rec_source) in FINITE_CMB_SOURCES and _finite_table(
        contract.Gamma_rec_k_a,
        min_cols=3,
    )
    if not gamma_rec_table_ready:
        blockers.append("Gamma_rec_k_a_missing_or_not_finite")
    else:
        if not bool(contract.physical_clock_receipt):
            blockers.append("physical_clock_missing_for_promoted_Gamma_rec")
        if not bool(contract.active_fiber_receipt):
            blockers.append("active_fiber_missing_for_promoted_Gamma_rec")
        if not bool(contract.conserved_sector_decomposition_receipt):
            blockers.append("conserved_sector_decomposition_missing_for_promoted_Gamma_rec")
        if not bool(contract.common_parent_response_pole_receipt):
            blockers.append("common_parent_response_pole_missing_for_promoted_Gamma_rec")

    rho_A_table_ready = bool(
        str(contract.rho_A_source) in FINITE_CMB_SOURCES
        and _finite_table(contract.rho_A_a, min_cols=2)
        and _table_column_positive(contract.rho_A_a, column=1)
    )
    if not rho_A_table_ready:
        blockers.append("rho_A_missing_or_not_finite")
    else:
        if not bool(contract.rho_A_transport_receipt):
            blockers.append("rho_A_transport_receipt_missing")
        if not bool(contract.anomaly_abundance_source_receipt):
            blockers.append("anomaly_abundance_source_receipt_missing")
        if not bool(contract.rho_A_source_receipt):
            blockers.append("rho_A_source_receipt_missing")
        if bool(contract.rho_A_source_receipt) and not (
            bool(contract.rho_A_transport_receipt)
            and bool(contract.anomaly_abundance_source_receipt)
        ):
            blockers.append("rho_A_source_receipt_not_decomposed")
        if (
            str(contract.rho_A_claim_label) == SOURCE_ONLY_ANOMALY_ABUNDANCE
            and not bool(contract.rho_A_source_receipt)
        ):
            blockers.append("rho_A_source_label_without_receipt")
        if (
            bool(contract.rho_A_transport_receipt)
            and not bool(contract.anomaly_abundance_source_receipt)
            and str(contract.rho_A_claim_label) != PHYSICAL_PARENT_WITH_CONDITIONAL_ABUNDANCE
        ):
            blockers.append("rho_A_transport_without_selector_label_invalid")

    if not bool(contract.finite_covariant_parent_receipt):
        blockers.append("finite_covariant_parent_receipt_missing")

    if not bool(contract.stress_energy_closure_receipt):
        blockers.append("stress_energy_closure_not_certified")

    if _array_has_positive(contract.Gamma_rec_k_a) and not bool(contract.explicit_recipient_stress_receipt):
        blockers.append("recipient_stress_missing_for_nonzero_Gamma_rec")

    if _array_has_positive(contract.Gamma_rec_k_a) and not bool(contract.exchange_current_closure_receipt):
        blockers.append("exchange_current_closure_missing_for_nonzero_Gamma_rec")

    if not bool(contract.gauge_independence_receipt):
        blockers.append("gauge_independence_not_certified")

    if not bool(contract.causal_response_receipt):
        blockers.append("causal_response_not_certified")

    if not bool(contract.refinement_convergence_receipt):
        blockers.append("refinement_convergence_not_certified")

    if (
        str(contract.freezeout_source) not in FINITE_CMB_SOURCES
        or not _valid_freezeout_surface(contract.freezeout_surface)
        or not bool(scale_bridge_validation.get("PHYSICAL_FREEZEOUT_SURFACE_RECEIPT", False))
    ):
        blockers.append("freezeout_missing_or_not_finite")

    if not bool(contract.cdm_limit_regression_passed):
        blockers.append("cdm_limit_regression_not_passed")

    if not bool(contract.official_likelihood_ready):
        blockers.append("official_likelihood_not_ready")

    if not bool(contract.frozen_likelihood_protocol_receipt):
        blockers.append("frozen_likelihood_protocol_not_certified")

    if not bool(contract.source_freeze_manifest_receipt):
        blockers.append("source_freeze_manifest_not_certified")

    if not bool(contract.solver_assumption_pin_receipt):
        blockers.append("solver_assumption_pin_not_certified")

    if not bool(contract.custom_parent_cdm_limit_regression_receipt):
        blockers.append("custom_parent_cdm_limit_regression_not_passed")

    if not bool(contract.standard_model_off_regression_receipt):
        blockers.append("standard_model_off_regression_not_passed")

    if not bool(contract.blinded_comparison_setup_receipt):
        blockers.append("blinded_comparison_setup_not_certified")

    if not bool(contract.full_observable_likelihood_receipt):
        blockers.append("full_observable_likelihood_not_executed")

    if not bool(contract.frozen_transfer_likelihood_receipt):
        blockers.append("frozen_transfer_likelihood_closure_not_certified")

    if not _valid_sha256_hash(contract.frozen_source_hash):
        blockers.append("frozen_source_hash_missing")

    if not _valid_sha256_hash(contract.frozen_solver_hash):
        blockers.append("frozen_solver_hash_missing")

    if not _valid_sha256_hash(contract.frozen_likelihood_hash):
        blockers.append("frozen_likelihood_hash_missing")

    receipt = len(blockers) == 0
    return {
        "PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT": receipt,
        "physical_cmb_prediction_eligible": receipt,
        "blockers": blockers,
        "finite_sources": sorted(FINITE_CMB_SOURCES),
        "theorem_side_sources_allowed_as_constants": sorted(THEOREM_SIDE_SOURCES),
        "physical_scale_bridge": scale_bridge_validation,
        "claim_boundary": (
            "Hard input contract for physical CMB prediction. Measurement-comparable TT curves remain "
            "diagnostics until every blocker is cleared by finite-derived inputs, a finite covariant "
            "stress parent, recipient stress and exchange-current closure for nonzero exchange, "
            "active-fiber/physical-clock/common-parent Gamma_rec receipts, source-only provenance, "
            "pooled reducers, source-only rho_A selector receipt, physical scale-bridge receipts, "
            "CMB1 custom-parent CDM regression, "
            "Standard-Model-off regression, blinded full-observable likelihood execution, solver pins, "
            "and frozen source/solver/likelihood hashes."
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
        rho_A_transport_receipt=bool(
            background_report.get("RHO_A_TRANSPORT_RECEIPT", False)
            or background_report.get("rho_A_transport_receipt", False)
        ),
        anomaly_abundance_source_receipt=bool(
            background_report.get("ANOMALY_ABUNDANCE_SOURCE_RECEIPT", False)
            or background_report.get("anomaly_abundance_source_receipt", False)
        ),
        rho_A_source_receipt=bool(
            background_report.get("RHO_A_SOURCE_RECEIPT", False)
            or background_report.get("rho_A_source_receipt", False)
        ),
        rho_A_claim_label=str(
            background_report.get("rho_A_claim_label", PHYSICAL_PARENT_WITH_CONDITIONAL_ABUNDANCE)
        ),
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
        exchange_current_closure_receipt=bool(
            background_report.get("EXCHANGE_CURRENT_CLOSURE_RECEIPT", False)
        ),
        physical_clock_receipt=bool(
            repair_clock_report.get("PHYSICAL_CLOCK_RECEIPT", False)
            or repair_clock_report.get("PHYSICAL_REPAIR_CLOCK_RECEIPT", False)
            or background_report.get("PHYSICAL_CLOCK_RECEIPT", False)
            or background_report.get("PHYSICAL_REPAIR_CLOCK_RECEIPT", False)
        ),
        active_fiber_receipt=bool(
            background_report.get("ACTIVE_FIBER_RECEIPT", False)
            or background_report.get("ACTIVE_FIBER_RESPONSE_RECEIPT", False)
            or repair_clock_report.get("ACTIVE_FIBER_RECEIPT", False)
            or repair_clock_report.get("ACTIVE_FIBER_RESPONSE_RECEIPT", False)
        ),
        conserved_sector_decomposition_receipt=bool(
            background_report.get("CONSERVED_SECTOR_DECOMPOSITION_RECEIPT", False)
            or repair_clock_report.get("CONSERVED_SECTOR_DECOMPOSITION_RECEIPT", False)
        ),
        common_parent_response_pole_receipt=bool(
            background_report.get("COMMON_PARENT_RESPONSE_POLE_RECEIPT", False)
            or background_report.get("COMMON_PARENT_RESPONSE_RECEIPT", False)
            or repair_clock_report.get("COMMON_PARENT_RESPONSE_POLE_RECEIPT", False)
            or repair_clock_report.get("COMMON_PARENT_RESPONSE_RECEIPT", False)
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
        source_freeze_manifest_receipt=bool(likelihood_report.get("FROZEN_SOURCE_MANIFEST_RECEIPT", False)),
        solver_assumption_pin_receipt=bool(likelihood_report.get("SOLVER_ASSUMPTION_PIN_RECEIPT", False)),
        custom_parent_cdm_limit_regression_receipt=bool(
            likelihood_report.get("CMB1_CUSTOM_PARENT_CDM_LIMIT_REGRESSION_RECEIPT", False)
            or likelihood_report.get("custom_parent_cdm_limit_regression_receipt", False)
        ),
        standard_model_off_regression_receipt=bool(
            likelihood_report.get("STANDARD_MODEL_OFF_REGRESSION_RECEIPT", False)
        ),
        blinded_comparison_setup_receipt=bool(
            likelihood_report.get("BLINDED_COMPARISON_SETUP_RECEIPT", False)
            or likelihood_report.get("blinded_comparison_setup_receipt", False)
        ),
        full_observable_likelihood_receipt=bool(
            likelihood_report.get("FULL_OBSERVABLE_LIKELIHOOD_RECEIPT", False)
        ),
        frozen_transfer_likelihood_receipt=bool(
            likelihood_report.get("FROZEN_TRANSFER_LIKELIHOOD_CLOSURE_RECEIPT", False)
        ),
        frozen_source_hash=background_report.get("source_hash"),
        frozen_solver_hash=likelihood_report.get("solver_hash") or background_report.get("solver_hash"),
        frozen_likelihood_hash=likelihood_report.get("likelihood_hash") or background_report.get("likelihood_hash"),
        physical_scale_bridge_receipts=(
            freezeout_report.get("physical_scale_bridge_receipts")
            or background_report.get("physical_scale_bridge_receipts")
            or likelihood_report.get("physical_scale_bridge_receipts")
        ),
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


def _finite_nonnegative_scalar(value: float | None) -> bool:
    return _finite_scalar(value) and float(value) >= 0.0


def _finite_array(value: np.ndarray | None) -> bool:
    if value is None:
        return False
    array = np.asarray(value, dtype=float)
    return bool(array.size and np.all(np.isfinite(array)))


def _finite_table(value: np.ndarray | None, *, min_cols: int) -> bool:
    if not _finite_array(value):
        return False
    array = np.asarray(value, dtype=float)
    return bool(array.ndim == 2 and array.shape[0] > 0 and array.shape[1] >= int(min_cols))


def _array_has_positive(value: np.ndarray | None) -> bool:
    if value is None:
        return False
    array = np.asarray(value, dtype=float)
    if not array.size or not np.all(np.isfinite(array)):
        return False
    return bool(np.any(array > 0.0))


def _table_column_positive(value: np.ndarray | None, *, column: int) -> bool:
    if not _finite_table(value, min_cols=int(column) + 1):
        return False
    array = np.asarray(value, dtype=float)
    return bool(np.all(array[:, int(column)] > 0.0))


def _nonempty_string(value: str | None) -> bool:
    return bool(isinstance(value, str) and value.strip())


def _valid_freezeout_surface(value: dict[str, Any] | None) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    required_hashes = ("surface_mesh_hash", "clock_hash", "state_vector_hash", "normal_derivative_hash")
    if not all(_valid_sha256_hash(value.get(key)) for key in required_hashes):
        return False
    if not bool(value.get("PHYSICAL_FREEZEOUT_SURFACE_RECEIPT")):
        return False
    return bool(value.get("common_surface_passed"))


def _valid_sha256_hash(value: str | None) -> bool:
    return bool(isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-fA-F]{64}", value.strip()))
