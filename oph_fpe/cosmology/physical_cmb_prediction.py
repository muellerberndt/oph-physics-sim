from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.cosmology.anomaly_abundance_selector import (
    PHYSICAL_PARENT_WITH_CONDITIONAL_ABUNDANCE,
    SOURCE_ONLY_ANOMALY_ABUNDANCE,
)
from oph_fpe.cosmology.physical_cmb_contract import (
    FINITE_CMB_SOURCES,
    PhysicalCMBInputContract,
    THEOREM_SIDE_SOURCES,
    validate_physical_cmb_contract,
)
from oph_fpe.cosmology.cosmological_scale_bridge import validate_physical_scale_bridge_receipts
from oph_fpe.cosmology.finite_covariant_parent import (
    CAUSAL_RESPONSE_RECEIPT,
    EXCHANGE_CURRENT_CLOSURE_RECEIPT,
    EXPLICIT_RECIPIENT_STRESS_RECEIPT,
    FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT,
    GAUGE_INDEPENDENCE_RECEIPT,
    PARENT_RECEIPT,
    REFINEMENT_CONVERGENCE_RECEIPT,
    STRESS_CLOSURE_RECEIPT,
)
from oph_fpe.cosmology.finite_repair_transition_clock import (
    validate_transition_clock_eligibility,
)
from oph_fpe.cosmology.source_provenance import (
    PROMOTED_CMB_SOURCE_QUANTITIES,
    certify_cmb_source_provenance,
)


def build_physical_cmb_input_contract(run_dirs: list[Path]) -> tuple[PhysicalCMBInputContract, dict[str, Any]]:
    roots = [Path(path) for path in run_dirs]
    no_data = _first_json(roots, "no_data_use_receipt.json")
    finite_transition = _first_json(roots, "finite_repair_transition_matrix_report.json")
    scalar = _first_json(roots, "scalar_repair_semigroup_report.json")
    finite_cert = _first_json(roots, "finite_certificate_report.json")
    finite_parent = _first_json(roots, "finite_covariant_collar_packet_parent_report.json")
    ba_kernel = _first_json(roots, "B_A_kernel_report.json")
    ba_kernel_refinement = _first_json(roots, "B_A_kernel_refinement_report.json")
    ba_parent = _first_json(roots, "b_a_parent_report.json")
    scale = _first_json(roots, "scale_compressed_repair_report.json")
    screen_to_radial_lift = _first_json(roots, "screen_to_radial_lift_report.json")
    screen_capacity = _first_json(roots, "screen_capacity_closure_report.json")
    strict_neutral = _first_json(roots, "strict_neutral_bulk_report.json")
    scalar_quotient = _first_json(roots, "scalar_quotient_report.json")
    compressed_likelihood = _first_json(roots, "oph_compressed_likelihood_report.json")
    official_likelihood = _first_json(roots, "official_planck_likelihood_readiness_report.json")
    frozen_transfer = _first_json(roots, "frozen_transfer_likelihood_report.json")
    camb_baseline = _first_json(roots, "camb_lcdm_baseline_report.json")
    physical_scale_bridge = (
        _first_json(roots, "physical_scale_bridge_report.json")
        or _first_json(roots, "cosmological_scale_bridge_report.json")
    )
    explicit_source_provenance = _first_json(roots, "cmb_source_provenance_report.json")
    transition_eligible = bool(
        validate_transition_clock_eligibility(finite_transition)["eligible"]
    )

    eta_source, eta_value = _eta_R_from_reports(finite_transition, scalar, scale, scalar_quotient)
    gamma_source, gamma_grid = _Gamma_rec_from_reports(finite_transition)
    a_source, a_value = _A_zeta_from_reports(finite_cert, scale)
    lift_receipt = _screen_to_primordial_lift_from_reports(finite_cert, scale, screen_to_radial_lift)
    q_source, q_value = _scalar_value_from_reports(scale, scalar_quotient, "q_IR")
    ell_source, ell_value = _scalar_value_from_reports(scale, scalar_quotient, "ell_IR")
    b_source, b_grid = _B_A_from_reports(ba_kernel, ba_parent)
    rho_source, rho_grid = _rho_A_from_reports(finite_cert, ba_parent)
    rho_transport_receipt = bool(
        finite_parent.get("RHO_A_TRANSPORT_RECEIPT", False)
        or finite_parent.get("rho_A_transport_receipt", False)
        or finite_cert.get("RHO_A_TRANSPORT_RECEIPT", False)
        or finite_cert.get("rho_A_transport_receipt", False)
    )
    anomaly_abundance_receipt = bool(
        finite_parent.get("ANOMALY_ABUNDANCE_SOURCE_RECEIPT", False)
        or finite_parent.get("anomaly_abundance_source_receipt", False)
        or finite_cert.get("ANOMALY_ABUNDANCE_SOURCE_RECEIPT", False)
        or finite_cert.get("anomaly_abundance_source_receipt", False)
    )
    rho_source_receipt = bool(
        finite_parent.get("RHO_A_SOURCE_RECEIPT", False)
        or finite_parent.get("rho_A_source_receipt", False)
        or finite_cert.get("RHO_A_SOURCE_RECEIPT", False)
        or finite_cert.get("rho_A_source_receipt", False)
        or (rho_transport_receipt and anomaly_abundance_receipt)
    )
    rho_claim_label = str(
        finite_parent.get("rho_A_claim_label")
        or finite_cert.get("rho_A_claim_label")
        or (
            SOURCE_ONLY_ANOMALY_ABUNDANCE
            if rho_source_receipt
            else PHYSICAL_PARENT_WITH_CONDITIONAL_ABUNDANCE
        )
    )
    freezeout_source, freezeout_surface = _freezeout_from_reports(strict_neutral, scale, physical_scale_bridge)
    official_likelihood_ready = bool(
        compressed_likelihood.get("official_likelihood_ready", False)
        or official_likelihood.get("official_likelihood_execution_ready", False)
        or frozen_transfer.get("official_likelihood_execution_ready", False)
    )
    cdm_limit_regression_passed = bool(
        compressed_likelihood.get("cdm_limit_regression_passed", False)
        or _truthy_any(_first_json(roots, "oph_boltzmann_input_report.json"), "cdm_limit_regression_passed")
        or camb_baseline.get("CDM_LIMIT_BOLTZMANN_RECEIPT", False)
        or frozen_transfer.get("CDM_LIMIT_REGRESSION_RECEIPT", False)
    )
    contract = PhysicalCMBInputContract(
        no_data_use_receipt=bool(no_data.get("no_data_use_receipt", False) or no_data.get("NO_DATA_USE_RECEIPT", False)),
        P_source="OPH_pixel_branch_predeclared",
        N_source=_N_source_from_screen_capacity(screen_capacity),
        eta_R_source=eta_source,
        eta_R_value=eta_value,
        A_zeta_source=a_source,
        A_zeta_value=a_value,
        q_IR_source=q_source,
        q_IR_value=q_value,
        ell_IR_source=ell_source,
        ell_IR_value=ell_value,
        B_A_source=b_source,
        B_A_k_a=b_grid,
        Gamma_rec_source=gamma_source,
        Gamma_rec_k_a=gamma_grid,
        rho_A_source=rho_source,
        rho_A_a=rho_grid,
        rho_A_transport_receipt=rho_transport_receipt,
        anomaly_abundance_source_receipt=anomaly_abundance_receipt,
        rho_A_source_receipt=rho_source_receipt,
        rho_A_claim_label=rho_claim_label,
        freezeout_source=freezeout_source,
        freezeout_surface=freezeout_surface,
        official_likelihood_ready=official_likelihood_ready,
        cdm_limit_regression_passed=cdm_limit_regression_passed,
        screen_to_primordial_lift_receipt=lift_receipt,
        finite_covariant_parent_receipt=bool(finite_parent.get(PARENT_RECEIPT, False)),
        stress_energy_closure_receipt=bool(finite_parent.get(STRESS_CLOSURE_RECEIPT, False)),
        gauge_independence_receipt=bool(finite_parent.get(GAUGE_INDEPENDENCE_RECEIPT, False)),
        causal_response_receipt=bool(finite_parent.get(CAUSAL_RESPONSE_RECEIPT, False)),
        refinement_convergence_receipt=bool(finite_parent.get(REFINEMENT_CONVERGENCE_RECEIPT, False)),
        explicit_recipient_stress_receipt=bool(finite_parent.get(EXPLICIT_RECIPIENT_STRESS_RECEIPT, False)),
        exchange_current_closure_receipt=bool(finite_parent.get(EXCHANGE_CURRENT_CLOSURE_RECEIPT, False)),
        physical_clock_receipt=bool(
            (
                transition_eligible
                and (
                    finite_transition.get("PHYSICAL_CLOCK_RECEIPT", False)
                    or finite_transition.get("PHYSICAL_REPAIR_CLOCK_RECEIPT", False)
                )
            )
            or finite_parent.get("PHYSICAL_CLOCK_RECEIPT", False)
            or finite_parent.get("PHYSICAL_REPAIR_CLOCK_RECEIPT", False)
        ),
        active_fiber_receipt=bool(
            finite_parent.get("ACTIVE_FIBER_RECEIPT", False)
            or finite_parent.get("ACTIVE_FIBER_RESPONSE_RECEIPT", False)
            or (
                transition_eligible
                and (
                    finite_transition.get("ACTIVE_FIBER_RECEIPT", False)
                    or finite_transition.get("ACTIVE_FIBER_RESPONSE_RECEIPT", False)
                )
            )
        ),
        conserved_sector_decomposition_receipt=bool(
            finite_parent.get("CONSERVED_SECTOR_DECOMPOSITION_RECEIPT", False)
            or (
                transition_eligible
                and finite_transition.get("CONSERVED_SECTOR_DECOMPOSITION_RECEIPT", False)
            )
        ),
        common_parent_response_pole_receipt=bool(
            finite_parent.get("COMMON_PARENT_RESPONSE_POLE_RECEIPT", False)
            or finite_parent.get("COMMON_PARENT_RESPONSE_RECEIPT", False)
            or (
                transition_eligible
                and (
                    finite_transition.get("COMMON_PARENT_RESPONSE_POLE_RECEIPT", False)
                    or finite_transition.get("COMMON_PARENT_RESPONSE_RECEIPT", False)
                )
            )
        ),
        frozen_likelihood_protocol_receipt=bool(
            finite_parent.get(FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT, False)
            or official_likelihood.get(FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT, False)
            or frozen_transfer.get(FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT, False)
        ),
        source_freeze_manifest_receipt=bool(frozen_transfer.get("FROZEN_SOURCE_MANIFEST_RECEIPT", False)),
        solver_assumption_pin_receipt=bool(frozen_transfer.get("SOLVER_ASSUMPTION_PIN_RECEIPT", False)),
        custom_parent_cdm_limit_regression_receipt=bool(
            frozen_transfer.get("CMB1_CUSTOM_PARENT_CDM_LIMIT_REGRESSION_RECEIPT", False)
        ),
        standard_model_off_regression_receipt=bool(
            frozen_transfer.get("STANDARD_MODEL_OFF_REGRESSION_RECEIPT", False)
        ),
        blinded_comparison_setup_receipt=bool(frozen_transfer.get("BLINDED_COMPARISON_SETUP_RECEIPT", False)),
        full_observable_likelihood_receipt=bool(
            frozen_transfer.get("FULL_OBSERVABLE_LIKELIHOOD_RECEIPT", False)
        ),
        frozen_transfer_likelihood_receipt=bool(
            frozen_transfer.get("FROZEN_TRANSFER_LIKELIHOOD_CLOSURE_RECEIPT", False)
        ),
        frozen_source_hash=(
            frozen_transfer.get("frozen_source_hash")
            or frozen_transfer.get("source_hash")
            or finite_parent.get("source_hash")
        ),
        frozen_solver_hash=(
            frozen_transfer.get("frozen_solver_hash")
            or frozen_transfer.get("solver_hash")
            or official_likelihood.get("solver_hash")
            or finite_parent.get("solver_hash")
        ),
        frozen_likelihood_hash=(
            frozen_transfer.get("frozen_likelihood_hash")
            or frozen_transfer.get("likelihood_hash")
            or official_likelihood.get("likelihood_hash")
            or finite_parent.get("likelihood_hash")
        ),
        physical_scale_bridge_receipts=physical_scale_bridge,
    )
    sources = {
        "no_data_use_receipt": no_data,
        "finite_transition_matrix_report": finite_transition,
        "scalar_repair_semigroup_report": scalar,
        "finite_certificate_report": finite_cert,
        "finite_covariant_collar_packet_parent_report": finite_parent,
        "B_A_kernel_report": ba_kernel,
        "B_A_kernel_refinement_report": ba_kernel_refinement,
        "b_a_parent_report": ba_parent,
        "scale_compressed_repair_report": scale,
        "screen_to_radial_lift_report": screen_to_radial_lift,
        "screen_capacity_closure_report": screen_capacity,
        "strict_neutral_bulk_report": strict_neutral,
        "scalar_quotient_report": scalar_quotient,
        "oph_compressed_likelihood_report": compressed_likelihood,
        "official_planck_likelihood_readiness_report": official_likelihood,
        "frozen_transfer_likelihood_report": frozen_transfer,
        "camb_lcdm_baseline_report": camb_baseline,
        "physical_scale_bridge_report": physical_scale_bridge,
    }
    source_provenance = _cmb_source_provenance_from_contract(
        contract,
        sources,
        explicit_source_provenance,
        roots,
    )
    contract.source_provenance_receipt = bool(source_provenance.get("CMB_SOURCE_PROVENANCE_RECEIPT", False))
    contract.pooled_source_reducer_receipt = bool(source_provenance.get("pooled_source_reducer_receipt", False))
    contract.contradiction_free_provenance_receipt = bool(
        source_provenance.get("contradiction_free_provenance_receipt", False)
    )
    contract.N_CRC_consensus_invariant_receipt = bool(
        source_provenance.get("N_CRC_consensus_invariant_receipt", False)
    )
    contract.global_likelihood_reduction_receipt = bool(
        source_provenance.get("global_likelihood_reduction_receipt", False)
    )
    sources["cmb_source_provenance_report"] = source_provenance
    return contract, sources


def write_physical_cmb_input_report(run_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    contract, sources = build_physical_cmb_input_contract(run_dirs)
    validation = validate_physical_cmb_contract(contract)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    contract_dict = _contract_to_jsonable(contract)
    (out / "physical_cmb_input_contract.json").write_text(json.dumps(contract_dict, indent=2, default=str), encoding="utf-8")
    (out / "physical_cmb_input_validation.json").write_text(json.dumps(validation, indent=2, default=str), encoding="utf-8")
    (out / "cmb_source_provenance_report.json").write_text(
        json.dumps(sources.get("cmb_source_provenance_report") or {}, indent=2, default=str),
        encoding="utf-8",
    )
    _write_array(out / "B_A_k_a.csv", contract.B_A_k_a, ["k_or_row", "a_or_col", "B_A"])
    _write_array(out / "Gamma_rec_k_a.csv", contract.Gamma_rec_k_a, ["k_or_row", "a_or_col", "Gamma_rec"])
    _write_array(out / "rho_A_a.csv", contract.rho_A_a, ["row", "col", "rho_A"])
    input_status = _input_status(contract)
    blockers = _physical_cmb_promotion_blockers(validation, input_status, sources, {}, {})
    report = {
        "mode": "physical_cmb_input_contract_report_v0",
        "run_dirs": [str(path) for path in run_dirs],
        "contract": contract_dict,
        "validation": validation,
        "PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT": validation["PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT"],
        "physical_cmb_prediction": False,
        "physical_cmb_prediction_eligible": validation["physical_cmb_prediction_eligible"],
        "blockers": blockers,
        "input_status": input_status,
        "source_summary": _source_summary(sources),
        "source_provenance": sources.get("cmb_source_provenance_report") or {},
        "claim_boundary": (
            "Physical CMB input contract assembly. This report may gather measurement-comparable diagnostics, "
            "but it does not run a physical CMB prediction unless every finite-input and likelihood gate passes."
        ),
    }
    (out / "physical_cmb_input_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "physical_cmb_input_report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def write_physical_cmb_input_no_data_use_receipt(run_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    """Write the no-data-use receipt for physical-CMB input construction.

    This receipt is deliberately narrow: it audits whether the OPH input
    functions assembled for a future physical CMB prediction are sourced from
    finite/theorem-side reports rather than fitted from measurement tables. It
    does not certify that those inputs are theorem-grade or likelihood-ready.
    """

    roots = [Path(path) for path in run_dirs]
    finite_transition = _first_json(roots, "finite_repair_transition_matrix_report.json")
    finite_cert = _first_json(roots, "finite_certificate_report.json")
    ba_kernel = _first_json(roots, "B_A_kernel_report.json")
    ba_kernel_refinement = _first_json(roots, "B_A_kernel_refinement_report.json")
    ba_parent = _first_json(roots, "b_a_parent_report.json")
    scale = _first_json(roots, "scale_compressed_repair_report.json")
    strict_neutral = _first_json(roots, "strict_neutral_bulk_report.json")
    scalar_quotient = _first_json(roots, "scalar_quotient_report.json")
    screen_capacity = _first_json(roots, "screen_capacity_closure_report.json")
    compressed_likelihood = _first_json(roots, "oph_compressed_likelihood_report.json")
    official_likelihood = _first_json(roots, "official_planck_likelihood_readiness_report.json")
    camb_baseline = _first_json(roots, "camb_lcdm_baseline_report.json")

    source_status = {
        "finite_repair_transition_matrix_report": {
            "present": bool(finite_transition),
            "used_for": ["eta_R", "Gamma_rec"],
            "measurement_data_used": _measurement_data_used(finite_transition),
        },
        "finite_certificate_report": {
            "present": bool(finite_cert),
            "used_for": ["A_zeta", "rho_A"],
            "measurement_data_used": _measurement_data_used(finite_cert),
        },
        "B_A_kernel_report": {
            "present": bool(ba_kernel),
            "used_for": ["B_A"],
            "measurement_data_used": _measurement_data_used(ba_kernel),
        },
        "B_A_kernel_refinement_report": {
            "present": bool(ba_kernel_refinement),
            "used_for": ["B_A_refinement_gate"],
            "measurement_data_used": _measurement_data_used(ba_kernel_refinement),
        },
        "b_a_parent_report": {
            "present": bool(ba_parent),
            "used_for": ["diagnostic_B_A", "diagnostic_rho_A"],
            "measurement_data_used": _measurement_data_used(ba_parent)
            or not bool(((ba_parent.get("readiness") or {}).get("checks") or {}).get("no_cmb_data_used", True)),
        },
        "scale_compressed_repair_report": {
            "present": bool(scale),
            "used_for": ["q_IR", "ell_IR", "freezeout_surface"],
            "measurement_data_used": _measurement_data_used(scale),
        },
        "strict_neutral_bulk_report": {
            "present": bool(strict_neutral),
            "used_for": ["neutral_freezeout_if_strict"],
            "measurement_data_used": _measurement_data_used(strict_neutral),
        },
        "scalar_quotient_report": {
            "present": bool(scalar_quotient),
            "used_for": ["diagnostic_eta_R", "diagnostic_q_IR", "diagnostic_ell_IR", "scalar_freezeout_gate"],
            "measurement_data_used": _measurement_data_used(scalar_quotient),
        },
        "screen_capacity_closure_report": {
            "present": bool(screen_capacity),
            "used_for": ["N_source"],
            "measurement_data_used": _measurement_data_used(screen_capacity),
        },
        "oph_compressed_likelihood_report": {
            "present": bool(compressed_likelihood),
            "used_for": ["validation_gate_only"],
            "measurement_data_used": False,
            "note": "Measurement reference reports may be present, but this receipt does not allow them to set OPH input functions.",
        },
        "official_planck_likelihood_readiness_report": {
            "present": bool(official_likelihood),
            "used_for": ["official_likelihood_gate_only"],
            "measurement_data_used": False,
            "note": "Environment readiness only; it does not set OPH input functions.",
        },
        "camb_lcdm_baseline_report": {
            "present": bool(camb_baseline),
            "used_for": ["cdm_limit_regression_gate_only"],
            "measurement_data_used": False,
            "note": "External LambdaCDM baseline checks Boltzmann plumbing only; it does not set OPH input functions.",
        },
        "finite_covariant_collar_packet_parent_report": {
            "present": bool(_first_json(roots, "finite_covariant_collar_packet_parent_report.json")),
            "used_for": ["stress_closure_gate", "recipient_stress_gate", "gauge_gate", "causal_response_gate"],
            "measurement_data_used": False,
            "note": "Source-contract receipt only; it does not fit OPH input functions to CMB data.",
        },
    }
    measurement_used = any(bool(row.get("measurement_data_used", False)) for row in source_status.values())
    report = {
        "mode": "physical_cmb_input_no_data_use_receipt_v0",
        "no_data_use_receipt": not measurement_used,
        "NO_DATA_USE_RECEIPT": not measurement_used,
        "measurement_data_used_for_input_functions": measurement_used,
        "measurement_reference_reports_present": bool(compressed_likelihood),
        "source_status": source_status,
        "run_dirs": [str(path) for path in roots],
        "claim_boundary": (
            "Certifies only the data firewall for assembling OPH CMB input functions. "
            "It does not certify finite-theorem grade B_A/rho_A/A_zeta inputs, CDM-limit regression, "
            "official likelihood readiness, or a physical CMB prediction."
        ),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "no_data_use_receipt.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def physical_cmb_promotion_audit_report(run_dirs: list[Path]) -> dict[str, Any]:
    """Summarize blockers between CMB diagnostics and prediction status."""

    roots = [Path(path) for path in run_dirs]
    contract, sources = build_physical_cmb_input_contract(roots)
    validation = validate_physical_cmb_contract(contract)
    input_status = _input_status(contract)
    finite_collar_boltzmann = _first_json(roots, "finite_collar_boltzmann_bundle_report.json")
    finite_collar_projection = _first_json(roots, "finite_collar_cmb_projection_report.json")
    blockers = _physical_cmb_promotion_blockers(
        validation,
        input_status,
        sources,
        finite_collar_boltzmann,
        finite_collar_projection,
    )
    report = {
        "mode": "physical_cmb_promotion_audit_v0",
        "run_dirs": [str(path) for path in roots],
        "physical_cmb_promotion_ready": bool(
            validation["PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT"] and not blockers
        ),
        "physical_cmb_prediction": False,
        "physical_cmb_input_contract_receipt": bool(validation["PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT"]),
        "no_data_use_receipt": bool(contract.no_data_use_receipt),
        "cdm_limit_regression_passed": bool(contract.cdm_limit_regression_passed),
        "official_likelihood_ready": bool(contract.official_likelihood_ready),
        "contract_blockers": validation["blockers"],
        "promotion_blockers": blockers,
        "input_status": input_status,
        "source_summary": _source_summary(sources),
        "source_readiness": _physical_cmb_source_readiness(
            sources,
            finite_collar_boltzmann,
            finite_collar_projection,
        ),
        "next_steps": _physical_cmb_next_steps(blockers),
        "claim_boundary": (
            "Promotion audit for the current OPH-FPE CMB pipeline. It identifies finite-source, "
            "calibration, and likelihood blockers. It does not change the hard physical-CMB gate "
            "and does not make a CMB prediction."
        ),
    }
    return report


def write_physical_cmb_promotion_audit_report(run_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    report = physical_cmb_promotion_audit_report(run_dirs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "physical_cmb_promotion_audit_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "physical_cmb_promotion_audit_report.md").write_text(
        _markdown_promotion_audit(report),
        encoding="utf-8",
    )
    return report


def physical_cmb_frontier_report(run_dirs: list[Path]) -> dict[str, Any]:
    """Write the hard-gate frontier between CMB diagnostics and prediction status."""

    roots = [Path(path) for path in run_dirs]
    contract, sources = build_physical_cmb_input_contract(roots)
    validation = validate_physical_cmb_contract(contract)
    input_status = _input_status(contract)
    finite_collar_boltzmann = _first_json(roots, "finite_collar_boltzmann_bundle_report.json")
    finite_collar_projection = _first_json(roots, "finite_collar_cmb_projection_report.json")
    physical_output = _first_json(roots, "physical_cmb_output_comparison_report.json")
    promotion = physical_cmb_promotion_audit_report(roots)
    source_readiness = _physical_cmb_source_readiness(
        sources,
        finite_collar_boltzmann,
        finite_collar_projection,
    )
    gate_rows = _physical_cmb_frontier_gate_rows(
        validation=validation,
        input_status=input_status,
        promotion=promotion,
        physical_output=physical_output,
        source_readiness=source_readiness,
        contract=contract,
    )
    blockers = _unique_strings(
        list(validation.get("blockers") or []) + list(promotion.get("promotion_blockers") or [])
    )
    gate_gap_rows = _physical_cmb_frontier_gap_rows(
        gate_rows=gate_rows,
        blockers=blockers,
        input_status=input_status,
        source_readiness=source_readiness,
        physical_output=physical_output,
    )
    return {
        "mode": "physical_cmb_frontier_v0",
        "run_dirs": [str(path) for path in roots],
        "PHYSICAL_CMB_FRONTIER_REPORT": True,
        "physical_cmb_prediction": False,
        "physical_cmb_prediction_ready": bool(
            validation.get("PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT", False)
            and promotion.get("physical_cmb_promotion_ready", False)
        ),
        "physical_cmb_output_comparison_receipt": bool(
            physical_output.get("PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT", False)
        ),
        "physical_cmb_prediction_receipt": bool(
            physical_output.get("PHYSICAL_CMB_PREDICTION_RECEIPT", False)
        ),
        "best_oph_diagnostic_model": physical_output.get("best_oph_diagnostic_model") or {},
        "measurement_comparable_model_count": int(physical_output.get("measurement_comparable_model_count") or 0),
        "oph_diagnostic_model_count": int(physical_output.get("oph_diagnostic_model_count") or 0),
        "official_likelihood_ready": bool(contract.official_likelihood_ready),
        "cdm_limit_regression_passed": bool(contract.cdm_limit_regression_passed),
        "physical_cmb_input_contract_receipt": bool(
            validation.get("PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT", False)
        ),
        "physical_cmb_promotion_ready": bool(promotion.get("physical_cmb_promotion_ready", False)),
        "gate_rows": gate_rows,
        "gate_gap_rows": gate_gap_rows,
        "blockers": blockers,
        "next_missing_receipts": _physical_cmb_next_steps(blockers),
        "input_status": input_status,
        "source_readiness": source_readiness,
        "source_summary": _source_summary(sources),
        "claim_boundary": (
            "Physical-CMB frontier report. Measurement-comparable TT curves are physical-unit outputs, "
            "but they remain diagnostic until the finite input contract, finite-source promotion gates, "
            "and official likelihood execution gates all pass."
        ),
    }


def write_physical_cmb_frontier_report(run_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    report = physical_cmb_frontier_report(run_dirs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "physical_cmb_frontier_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "physical_cmb_frontier_report.md").write_text(
        _markdown_physical_cmb_frontier(report),
        encoding="utf-8",
    )
    return report


def _eta_R_from_reports(
    finite_transition: dict[str, Any],
    scalar: dict[str, Any],
    scale: dict[str, Any],
    scalar_quotient: dict[str, Any],
) -> tuple[str, float | None]:
    transition_eligible = bool(
        validate_transition_clock_eligibility(finite_transition)["eligible"]
    )
    scalar_transition_eligible = bool(
        validate_transition_clock_eligibility(scalar)["eligible"]
    )
    if transition_eligible and finite_transition.get("eta_R_finite_lattice_derived", False):
        return "finite_repair_transition_clock", _float((finite_transition.get("primary") or {}).get("eta_R_estimate"))
    empirical = (finite_transition.get("clock_modes") or {}).get("empirical") or {}
    if transition_eligible and (
        finite_transition.get("eta_R_empirical_finite_lattice_derived", False)
        or empirical.get("eta_R_finite_lattice_derived", False)
    ):
        return "finite_repair_transition_clock", _float(empirical.get("eta_R_value"))
    if scalar_transition_eligible and scalar.get("eta_R_finite_lattice_derived", False):
        return "finite_repair_transition_clock", _float(
            (scalar.get("semigroup") or {}).get("eta_R_estimate", scalar.get("eta_R"))
        )
    params = scale.get("cmb_parameter_readouts") or {}
    if scale.get("scale_compressed_operator_receipt", False) and params.get("eta_R") is not None:
        return "scale_compressed_24_round_finite_ladder", _float(params.get("eta_R"))
    edge = scalar_quotient.get("edge_center_readout") or {}
    if scalar_quotient.get("finite_lattice_cmb_scalar_release_ready", False) and edge.get("theta_OPH_P_over_48") is not None:
        return "finite_lattice", _float(edge.get("theta_OPH_P_over_48"))
    if scalar_quotient and edge.get("theta_OPH_P_over_48") is not None:
        return "diagnostic_proxy", _float(edge.get("theta_OPH_P_over_48"))
    return "diagnostic_proxy", _float((finite_transition.get("primary") or {}).get("eta_R_estimate"))


def _Gamma_rec_from_reports(finite_transition: dict[str, Any]) -> tuple[str, np.ndarray | None]:
    primary = finite_transition.get("primary") or {}
    gamma = _float(primary.get("gamma_continuous"))
    if gamma is None:
        return "unknown", None
    transition_eligible = bool(
        validate_transition_clock_eligibility(finite_transition)["eligible"]
    )
    source = "finite_repair_transition_clock" if transition_eligible else "diagnostic_proxy"
    return source, np.asarray([[0.0, 1.0, gamma]], dtype=float)


def _A_zeta_from_reports(finite_cert: dict[str, Any], scale: dict[str, Any]) -> tuple[str, float | None]:
    derived = finite_cert.get("derived_outputs") or {}
    value = _float(derived.get("A_zeta", finite_cert.get("A_zeta")))
    if finite_cert.get("theorem_grade_finite_inputs", False) and value is not None:
        return "finite_lattice", value
    params = scale.get("cmb_parameter_readouts") or {}
    if params.get("A_zeta") is not None and scale.get("scale_compressed_operator_receipt", False):
        return "scale_compressed_24_round_finite_ladder", _float(params.get("A_zeta"))
    return "diagnostic_proxy", value


def _screen_to_primordial_lift_from_reports(
    finite_cert: dict[str, Any],
    scale: dict[str, Any],
    screen_to_radial_lift: dict[str, Any],
) -> bool:
    derived = finite_cert.get("derived_outputs") or {}
    params = scale.get("cmb_parameter_readouts") or {}
    return bool(
        derived.get("screen_to_primordial_lift_receipt", False)
        or finite_cert.get("SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT", False)
        or finite_cert.get("SCREEN_TO_RADIAL_LIFT_RECEIPT", False)
        or finite_cert.get("screen_to_primordial_lift_receipt", False)
        or screen_to_radial_lift.get("SCREEN_TO_RADIAL_LIFT_RECEIPT", False)
        or screen_to_radial_lift.get("SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT", False)
        or (
            params.get("A_zeta") is not None
            and scale.get("scale_compressed_operator_receipt", False)
            and scale.get("SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT", False)
        )
    )


def _scalar_value_from_reports(
    scale: dict[str, Any],
    scalar_quotient: dict[str, Any],
    key: str,
) -> tuple[str, float | None]:
    params = scale.get("cmb_parameter_readouts") or {}
    value = _float(params.get(key))
    if scale.get("scale_compressed_operator_receipt", False) and value is not None:
        return "scale_compressed_24_round_finite_ladder", value
    levels = scalar_quotient.get("active_angular_levels") or {}
    if key == "ell_IR":
        value = _float(levels.get("target_ell_IR"))
    elif key == "q_IR":
        value = 0.25 if scalar_quotient else None
    else:
        value = None
    if scalar_quotient.get("finite_lattice_cmb_scalar_release_ready", False) and value is not None:
        return "finite_lattice", value
    return "diagnostic_proxy", value


def _B_A_from_reports(ba_kernel: dict[str, Any], ba_parent: dict[str, Any]) -> tuple[str, np.ndarray | None]:
    if ba_kernel.get("B_A_KERNEL_RECEIPT", False):
        return "parent_collar_finite_difference", _array(ba_kernel.get("B_A_k_a"))
    if _ba_diagnostic_candidate(ba_kernel):
        candidate = _array(ba_kernel.get("B_A_k_a"))
        if candidate is not None:
            return "diagnostic_proxy", candidate[:, :3] if candidate.ndim == 2 and candidate.shape[1] >= 3 else candidate
    rows = ba_parent.get("rows") or ba_parent.get("observer_view_rows") or []
    values = []
    for row in rows:
        k = _float(row.get("k_h_mpc", row.get("k_proxy_inverse_theta")))
        a = _float(row.get("a"))
        b = _float(row.get("B_A_mean"))
        if k is not None and a is not None and b is not None:
            values.append([k, a, b])
    return ("diagnostic_proxy", np.asarray(values, dtype=float) if values else None)


def _rho_A_from_reports(finite_cert: dict[str, Any], ba_parent: dict[str, Any]) -> tuple[str, np.ndarray | None]:
    derived = finite_cert.get("derived_outputs") or {}
    finite_rho = _array(derived.get("rho_A_a", finite_cert.get("rho_A_a")))
    if finite_cert.get("theorem_grade_finite_inputs", False) and finite_rho is not None:
        return "finite_lattice", finite_rho
    rows = ba_parent.get("rows") or ba_parent.get("observer_view_rows") or []
    values = []
    for row in rows:
        a = _float(row.get("a"))
        base = _float(row.get("rho_A", row.get("rho_A_base")))
        if a is not None and base is not None:
            values.append([a, base])
    return ("diagnostic_proxy", np.asarray(values, dtype=float) if values else None)


def _freezeout_from_reports(
    strict_neutral: dict[str, Any],
    scale: dict[str, Any],
    physical_scale_bridge: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    bridge_validation = validate_physical_scale_bridge_receipts(physical_scale_bridge)
    if bridge_validation.get("PHYSICAL_FREEZEOUT_SURFACE_RECEIPT", False):
        return "neutral_bulk_freezeout", _freezeout_surface_from_scale_bridge(physical_scale_bridge)
    if strict_neutral.get("strict_neutral_bulk", False):
        cycle = strict_neutral.get("freezeout_cycle", strict_neutral.get("neutral_bulk_freezeout_cycle"))
        surface = {"source": "strict_neutral_bulk_report"}
        if cycle is not None:
            surface["freezeout_cycle"] = cycle
        return "neutral_bulk_freezeout", surface
    if scale.get("scale_compressed_operator_receipt", False):
        return "scale_compressed_24_round_finite_ladder", {
            "repair_rounds": scale.get("logical_repair_rounds"),
            "source": "scale_compressed_repair_report",
        }
    return "unknown", None


def _freezeout_surface_from_scale_bridge(physical_scale_bridge: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": physical_scale_bridge.get("source") or "physical_scale_bridge_report",
        "PHYSICAL_FREEZEOUT_SURFACE_RECEIPT": True,
        "surface_mesh_hash": physical_scale_bridge.get("surface_mesh_hash"),
        "clock_hash": physical_scale_bridge.get("clock_hash"),
        "state_vector_hash": physical_scale_bridge.get("state_vector_hash"),
        "normal_derivative_hash": physical_scale_bridge.get("normal_derivative_hash"),
        "common_surface_passed": bool(physical_scale_bridge.get("common_surface_passed", False)),
        "mode_dependent_freezeout_map": physical_scale_bridge.get("mode_dependent_freezeout_map"),
        "claim_tier": physical_scale_bridge.get("claim_tier"),
        "geometry_origin": physical_scale_bridge.get("geometry_origin"),
    }


def _cmb_source_provenance_from_contract(
    contract: PhysicalCMBInputContract,
    sources: dict[str, dict[str, Any]],
    explicit_report: dict[str, Any],
    roots: list[Path],
) -> dict[str, Any]:
    report_counts = _json_report_counts(
        roots,
        {
            _source_report_filename(report_name)
            for report_name in {
                "finite_transition_matrix_report",
                "finite_certificate_report",
                "scale_compressed_repair_report",
                "B_A_kernel_report",
                "b_a_parent_report",
                "screen_capacity_closure_report",
                "scalar_quotient_report",
            }
        },
    )
    generated_nodes = [
        _provenance_node("eta_R", contract.eta_R_source, "finite_transition_matrix_report", sources, contract),
        _provenance_node("Gamma_rec", contract.Gamma_rec_source, "finite_transition_matrix_report", sources, contract),
        _provenance_node("A_zeta", contract.A_zeta_source, "finite_certificate_report", sources, contract),
        _provenance_node("q_IR", contract.q_IR_source, "scale_compressed_repair_report", sources, contract),
        _provenance_node("ell_IR", contract.ell_IR_source, "scale_compressed_repair_report", sources, contract),
        _provenance_node("B_A_k_a", contract.B_A_source, "B_A_kernel_report", sources, contract),
        _provenance_node("rho_A_a", contract.rho_A_source, "finite_certificate_report", sources, contract),
        _provenance_node("N_CRC", contract.N_source, "screen_capacity_closure_report", sources, contract),
    ]
    explicit_nodes = explicit_report.get("nodes") if isinstance(explicit_report.get("nodes"), list) else []
    reducers = _source_reducers_from_contract(contract, sources, report_counts)
    explicit_reducers = explicit_report.get("reducers") or explicit_report.get("source_reducers") or {}
    if isinstance(explicit_reducers, dict):
        for quantity, reducer in explicit_reducers.items():
            if isinstance(reducer, dict):
                reducers[str(quantity)] = reducer
    global_checks = _global_likelihood_checks(sources)
    explicit_global = explicit_report.get("global_checks") if isinstance(explicit_report.get("global_checks"), dict) else {}
    global_checks.update(explicit_global)
    certificate = certify_cmb_source_provenance(
        list(explicit_nodes) if explicit_nodes else generated_nodes,
        reducers,
        global_checks=global_checks,
    )
    blockers = list(certificate.get("blockers") or [])
    if explicit_report and explicit_report.get("CMB_SOURCE_PROVENANCE_RECEIPT") is False:
        blockers.append("explicit_source_provenance_report_false")
        blockers.extend(str(blocker) for blocker in (explicit_report.get("blockers") or []))
    blockers = _unique_strings(blockers)
    receipt = len(blockers) == 0
    certificate.update(
        {
            "CMB_SOURCE_PROVENANCE_RECEIPT": receipt,
            "source_provenance_receipt": receipt,
            "blockers": blockers,
            "nodes": list(explicit_nodes) if explicit_nodes else generated_nodes,
            "reducers": reducers,
            "explicit_report_present": bool(explicit_report),
            "explicit_report_receipt": explicit_report.get("CMB_SOURCE_PROVENANCE_RECEIPT"),
        }
    )
    return certificate


def _provenance_node(
    quantity: str,
    source: str,
    default_report_name: str,
    sources: dict[str, dict[str, Any]],
    contract: PhysicalCMBInputContract,
) -> dict[str, Any]:
    report_name = _source_report_name_for_quantity(quantity, source, default_report_name)
    report = sources.get(report_name) or {}
    return {
        "node_id": quantity,
        "quantity": quantity,
        "source": source,
        "source_kind": source,
        "source_report": report_name,
        "source_hash": report.get("source_hash") or contract.frozen_source_hash,
        "parents": [],
        "source_only": source in FINITE_CMB_SOURCES or source in THEOREM_SIDE_SOURCES,
        "no_cmb_data_used": bool(contract.no_data_use_receipt or _explicit_no_cmb_data_used(report)),
        "fit_to_planck": bool(report.get("fit_to_planck", False)),
        "fit_to_measurement": bool(report.get("fit_to_measurement", False)),
        "measurement_data_used": _raw_measurement_data_used(report),
        "cmb_data_used": bool(report.get("cmb_data_used", False)),
        "cmb_data_used_for_input": bool(report.get("cmb_data_used_for_input", False)),
        "planck_data_used_for_input": bool(report.get("planck_data_used_for_input", False)),
        "uses_measurements_to_set_inputs": bool(report.get("uses_measurements_to_set_inputs", False)),
    }


def _source_reducers_from_contract(
    contract: PhysicalCMBInputContract,
    sources: dict[str, dict[str, Any]],
    report_counts: dict[str, int],
) -> dict[str, dict[str, Any]]:
    reducers = {
        "eta_R": _source_reducer_row(
            "eta_R", contract.eta_R_source, "finite_transition_matrix_report", sources, report_counts
        ),
        "Gamma_rec": _source_reducer_row(
            "Gamma_rec", contract.Gamma_rec_source, "finite_transition_matrix_report", sources, report_counts
        ),
        "A_zeta": _source_reducer_row(
            "A_zeta", contract.A_zeta_source, "finite_certificate_report", sources, report_counts
        ),
        "q_IR": _source_reducer_row("q_IR", contract.q_IR_source, "scale_compressed_repair_report", sources, report_counts),
        "ell_IR": _source_reducer_row(
            "ell_IR", contract.ell_IR_source, "scale_compressed_repair_report", sources, report_counts
        ),
        "B_A_k_a": _source_reducer_row("B_A_k_a", contract.B_A_source, "B_A_kernel_report", sources, report_counts),
        "rho_A_a": _source_reducer_row(
            "rho_A_a", contract.rho_A_source, "finite_certificate_report", sources, report_counts
        ),
        "N_CRC": {
            "mode": "consensus_invariant",
            "consensus_invariant": contract.N_source in THEOREM_SIDE_SOURCES,
            "additive_capacity_schema": False,
            "disjoint_coverage_receipt": False,
        },
    }
    return reducers


def _source_reducer_row(
    quantity: str,
    source: str,
    default_report_name: str,
    sources: dict[str, dict[str, Any]],
    report_counts: dict[str, int],
) -> dict[str, Any]:
    report_name = _source_report_name_for_quantity(quantity, source, default_report_name)
    report = sources.get(report_name) or {}
    nested = (report.get("source_reducers") or {}).get(quantity) if isinstance(report.get("source_reducers"), dict) else None
    if isinstance(nested, dict):
        return nested
    report_count = int(report_counts.get(_source_report_filename(report_name), 0))
    pooled_receipt = bool(
        report.get("pooled_sufficient_statistics_receipt", False)
        or report.get("POOLED_SOURCE_REDUCER_RECEIPT", False)
    )
    if pooled_receipt:
        return {
            "mode": "pooled_sufficient_statistics",
            "pooled_sufficient_statistics": True,
            "units_validated": bool(report.get("units_validated", False)),
            "coordinate_grid_validated": bool(report.get("coordinate_grid_validated", False)),
            "coverage_validated": bool(report.get("coverage_validated", False)),
            "duplicates_checked": bool(report.get("duplicates_checked", False)),
            "interpolation_policy_frozen": bool(report.get("interpolation_policy_frozen", False)),
            "covariance_validated": bool(report.get("covariance_validated", False)),
            "shard_local_nonlinear_average": False,
        }
    return {
        "mode": "requires_pooled_sufficient_statistics",
        "single_global_source": False,
        "pooled_sufficient_statistics": False,
        "shard_local_nonlinear_average": bool(report.get("shard_local_nonlinear_average", False)),
        "report_count": report_count,
    }


def _global_likelihood_checks(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    compressed = sources.get("oph_compressed_likelihood_report") or {}
    official = sources.get("official_planck_likelihood_readiness_report") or {}
    camb = sources.get("camb_lcdm_baseline_report") or {}
    return {
        "official_likelihood_rollup": _rollup_mode(compressed, official),
        "cdm_limit_rollup": _rollup_mode(compressed, camb),
    }


def _rollup_mode(*reports: dict[str, Any]) -> str:
    for report in reports:
        if not report:
            continue
        rollup = str(report.get("rollup") or report.get("reduction") or "")
        if rollup:
            return rollup
        if report.get("shard_local_any_rollup", False) or report.get("uses_shard_any_rollup", False):
            return "shard_any"
    return "missing"


def _source_report_name_for_quantity(quantity: str, source: str, default_report_name: str) -> str:
    if quantity in {"eta_R", "Gamma_rec"} and source == "scale_compressed_24_round_finite_ladder":
        return "scale_compressed_repair_report"
    if quantity in {"eta_R", "q_IR", "ell_IR"} and source == "finite_lattice":
        return "scalar_quotient_report"
    if quantity in {"A_zeta", "rho_A_a"} and source == "scale_compressed_24_round_finite_ladder":
        return "scale_compressed_repair_report"
    if quantity == "B_A_k_a" and source == "diagnostic_proxy":
        return "b_a_parent_report"
    return default_report_name


def _source_report_filename(report_name: str) -> str:
    filenames = {
        "finite_transition_matrix_report": "finite_repair_transition_matrix_report.json",
        "scalar_repair_semigroup_report": "scalar_repair_semigroup_report.json",
        "finite_certificate_report": "finite_certificate_report.json",
        "finite_covariant_collar_packet_parent_report": "finite_covariant_collar_packet_parent_report.json",
        "B_A_kernel_report": "B_A_kernel_report.json",
        "B_A_kernel_refinement_report": "B_A_kernel_refinement_report.json",
        "b_a_parent_report": "b_a_parent_report.json",
        "scale_compressed_repair_report": "scale_compressed_repair_report.json",
        "screen_capacity_closure_report": "screen_capacity_closure_report.json",
        "strict_neutral_bulk_report": "strict_neutral_bulk_report.json",
        "scalar_quotient_report": "scalar_quotient_report.json",
        "oph_compressed_likelihood_report": "oph_compressed_likelihood_report.json",
        "official_planck_likelihood_readiness_report": "official_planck_likelihood_readiness_report.json",
        "camb_lcdm_baseline_report": "camb_lcdm_baseline_report.json",
    }
    return filenames.get(report_name, f"{report_name}.json")


def _json_report_counts(roots: list[Path], names: set[str]) -> dict[str, int]:
    counts = {name: 0 for name in names}
    seen: set[Path] = set()
    for root in roots:
        root = Path(root)
        candidates = [root / name for name in names]
        if root.exists() and root.is_dir():
            for name in names:
                candidates.extend(sorted(root.glob(f"**/{name}")))
        for path in candidates:
            resolved = path.resolve() if path.exists() else path
            if resolved in seen or not path.exists() or not path.is_file():
                continue
            seen.add(resolved)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and data:
                counts[path.name] = counts.get(path.name, 0) + 1
    return counts


def _source_summary(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, report in sources.items():
        row = {
            "present": bool(report),
            "mode": report.get("mode"),
            "physical_cmb_prediction": report.get("physical_cmb_prediction"),
            "finite_certificate_compiler_ready": report.get("finite_certificate_compiler_ready"),
            "theorem_grade_finite_inputs": report.get("theorem_grade_finite_inputs"),
            "proxy_certificate": report.get("proxy_certificate"),
        }
        if name == "official_planck_likelihood_readiness_report":
            row.update(
                {
                    "official_likelihood_execution_ready": report.get("official_likelihood_execution_ready"),
                    "official_planck_likelihood_data_paths_configured": report.get(
                        "official_planck_likelihood_data_paths_configured"
                    ),
                    "official_clik_api_available": report.get("official_clik_api_available"),
                    "camb_available": report.get("camb_available"),
                    "cobaya_available": report.get("cobaya_available"),
                    "blockers": report.get("blockers") or [],
                }
            )
        if name == "frozen_transfer_likelihood_report":
            row.update(
                {
                    "FROZEN_TRANSFER_LIKELIHOOD_CLOSURE_RECEIPT": report.get(
                        "FROZEN_TRANSFER_LIKELIHOOD_CLOSURE_RECEIPT"
                    ),
                    "FROZEN_SOURCE_MANIFEST_RECEIPT": report.get("FROZEN_SOURCE_MANIFEST_RECEIPT"),
                    "SOLVER_ASSUMPTION_PIN_RECEIPT": report.get("SOLVER_ASSUMPTION_PIN_RECEIPT"),
                    "CMB1_CUSTOM_PARENT_CDM_LIMIT_REGRESSION_RECEIPT": report.get(
                        "CMB1_CUSTOM_PARENT_CDM_LIMIT_REGRESSION_RECEIPT"
                    ),
                    "STANDARD_MODEL_OFF_REGRESSION_RECEIPT": report.get(
                        "STANDARD_MODEL_OFF_REGRESSION_RECEIPT"
                    ),
                    "BLINDED_COMPARISON_SETUP_RECEIPT": report.get("BLINDED_COMPARISON_SETUP_RECEIPT"),
                    "FULL_OBSERVABLE_LIKELIHOOD_RECEIPT": report.get(
                        "FULL_OBSERVABLE_LIKELIHOOD_RECEIPT"
                    ),
                    "source_hash_present": _nonempty_string(
                        report.get("frozen_source_hash") or report.get("source_hash")
                    ),
                    "solver_hash_present": _nonempty_string(
                        report.get("frozen_solver_hash") or report.get("solver_hash")
                    ),
                    "likelihood_hash_present": _nonempty_string(
                        report.get("frozen_likelihood_hash") or report.get("likelihood_hash")
                    ),
                    "blockers": report.get("blockers") or [],
                }
            )
        if name == "finite_covariant_collar_packet_parent_report":
            row.update(
                {
                    "parent_receipt": report.get(PARENT_RECEIPT),
                    "stress_energy_closure_receipt": report.get(STRESS_CLOSURE_RECEIPT),
                    "explicit_recipient_stress_receipt": report.get(EXPLICIT_RECIPIENT_STRESS_RECEIPT),
                    "exchange_current_closure_receipt": report.get(EXCHANGE_CURRENT_CLOSURE_RECEIPT),
                    "physical_clock_receipt": (
                        report.get("PHYSICAL_CLOCK_RECEIPT")
                        or report.get("PHYSICAL_REPAIR_CLOCK_RECEIPT")
                    ),
                    "active_fiber_receipt": (
                        report.get("ACTIVE_FIBER_RECEIPT")
                        or report.get("ACTIVE_FIBER_RESPONSE_RECEIPT")
                    ),
                    "conserved_sector_decomposition_receipt": report.get(
                        "CONSERVED_SECTOR_DECOMPOSITION_RECEIPT"
                    ),
                    "common_parent_response_pole_receipt": (
                        report.get("COMMON_PARENT_RESPONSE_POLE_RECEIPT")
                        or report.get("COMMON_PARENT_RESPONSE_RECEIPT")
                    ),
                    "gauge_independence_receipt": report.get(GAUGE_INDEPENDENCE_RECEIPT),
                    "causal_response_receipt": report.get(CAUSAL_RESPONSE_RECEIPT),
                    "refinement_convergence_receipt": report.get(REFINEMENT_CONVERGENCE_RECEIPT),
                    "frozen_likelihood_protocol_receipt": report.get(FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT),
                    "source_hash_present": _nonempty_string(report.get("source_hash")),
                    "solver_hash_present": _nonempty_string(report.get("solver_hash")),
                    "likelihood_hash_present": _nonempty_string(report.get("likelihood_hash")),
                    "blockers": report.get("blockers") or [],
                }
            )
        if name == "cmb_source_provenance_report":
            row.update(
                {
                    "CMB_SOURCE_PROVENANCE_RECEIPT": report.get("CMB_SOURCE_PROVENANCE_RECEIPT"),
                    "pooled_source_reducer_receipt": report.get("pooled_source_reducer_receipt"),
                    "contradiction_free_provenance_receipt": report.get(
                        "contradiction_free_provenance_receipt"
                    ),
                    "N_CRC_consensus_invariant_receipt": report.get(
                        "N_CRC_consensus_invariant_receipt"
                    ),
                    "global_likelihood_reduction_receipt": report.get(
                        "global_likelihood_reduction_receipt"
                    ),
                    "blockers": report.get("blockers") or [],
                }
            )
        if name == "physical_scale_bridge_report":
            validation = validate_physical_scale_bridge_receipts(report)
            row.update(
                {
                    "PHYSICAL_SCALE_BRIDGE_RECEIPT": validation.get("PHYSICAL_SCALE_BRIDGE_RECEIPT"),
                    "PHYSICAL_SPATIAL_K_RECEIPT": validation.get("PHYSICAL_SPATIAL_K_RECEIPT"),
                    "PHYSICAL_K_RECEIPT": validation.get("PHYSICAL_K_RECEIPT"),
                    "SCREEN_TO_PHYSICAL_K_ASSOCIATION_RECEIPT": validation.get(
                        "SCREEN_TO_PHYSICAL_K_ASSOCIATION_RECEIPT"
                    ),
                    "SOURCE_ANGULAR_SECTOR_RECEIPT": validation.get("SOURCE_ANGULAR_SECTOR_RECEIPT"),
                    "SOURCE_ANGULAR_MODE_RECEIPT": validation.get("SOURCE_ANGULAR_MODE_RECEIPT"),
                    "CALIBRATED_A_EVOLUTION_RECEIPT": validation.get("CALIBRATED_A_EVOLUTION_RECEIPT"),
                    "PHYSICAL_MODE_FREEZEOUT_MAP_RECEIPT": validation.get(
                        "PHYSICAL_MODE_FREEZEOUT_MAP_RECEIPT"
                    ),
                    "PHYSICAL_FREEZEOUT_SURFACE_RECEIPT": validation.get("PHYSICAL_FREEZEOUT_SURFACE_RECEIPT"),
                    "COMMON_PRIMORDIAL_ANOMALY_MODE_BASIS_RECEIPT": validation.get(
                        "COMMON_PRIMORDIAL_ANOMALY_MODE_BASIS_RECEIPT"
                    ),
                    "SCALE_BRIDGE_REFINEMENT_RECEIPT": validation.get("SCALE_BRIDGE_REFINEMENT_RECEIPT"),
                    "CROSS_RECEIPT_CONSISTENCY_RECEIPT": validation.get("CROSS_RECEIPT_CONSISTENCY_RECEIPT"),
                    "NO_POSTHOC_CALIBRATION_RECEIPT": validation.get("NO_POSTHOC_CALIBRATION_RECEIPT"),
                    "claim_tier": validation.get("claim_tier"),
                    "geometry_origin": validation.get("geometry_origin"),
                    "blockers": validation.get("blockers") or [],
                }
            )
        summary[name] = row
    return summary


def _input_status(contract: PhysicalCMBInputContract) -> dict[str, Any]:
    scale_bridge_validation = validate_physical_scale_bridge_receipts(contract.physical_scale_bridge_receipts)
    rho_A_status = _array_status(contract.rho_A_source, contract.rho_A_a, positive_column=1)
    rho_A_status.update(
        {
            "rho_A_transport_receipt": bool(contract.rho_A_transport_receipt),
            "anomaly_abundance_source_receipt": bool(contract.anomaly_abundance_source_receipt),
            "rho_A_source_receipt": bool(contract.rho_A_source_receipt),
            "claim_label": contract.rho_A_claim_label,
        }
    )
    rho_A_status["physical_gate_passed"] = bool(
        rho_A_status.get("physical_gate_passed", False)
        and contract.rho_A_source_receipt
    )
    return {
        "P_source": _theorem_constant_status(contract.P_source),
        "N_source": _theorem_constant_status(contract.N_source),
        "eta_R": _scalar_status(contract.eta_R_source, contract.eta_R_value),
        "A_zeta": _scalar_status(contract.A_zeta_source, contract.A_zeta_value, positive=True),
        "q_IR": _scalar_status(contract.q_IR_source, contract.q_IR_value),
        "ell_IR": _scalar_status(contract.ell_IR_source, contract.ell_IR_value, positive=True),
        "B_A_k_a": _array_status(contract.B_A_source, contract.B_A_k_a),
        "Gamma_rec_k_a": _array_status(contract.Gamma_rec_source, contract.Gamma_rec_k_a),
        "rho_A_a": rho_A_status,
        "freezeout_surface": {
            "source": contract.freezeout_source,
            "source_is_finite_cmb_source": contract.freezeout_source in FINITE_CMB_SOURCES,
            "diagnostic_value_present": isinstance(contract.freezeout_surface, dict),
            "physical_gate_passed": (
                contract.freezeout_source in FINITE_CMB_SOURCES
                and isinstance(contract.freezeout_surface, dict)
                and bool(scale_bridge_validation.get("PHYSICAL_FREEZEOUT_SURFACE_RECEIPT", False))
                and bool((contract.freezeout_surface or {}).get("PHYSICAL_FREEZEOUT_SURFACE_RECEIPT", False))
            ),
        },
        "physical_scale_bridge": {
            "claim_tier": scale_bridge_validation.get("claim_tier"),
            "geometry_origin": scale_bridge_validation.get("geometry_origin"),
            "PHYSICAL_SCALE_BRIDGE_RECEIPT": bool(
                scale_bridge_validation.get("PHYSICAL_SCALE_BRIDGE_RECEIPT", False)
            ),
            "PHYSICAL_SPATIAL_K_RECEIPT": bool(
                scale_bridge_validation.get("PHYSICAL_SPATIAL_K_RECEIPT", False)
            ),
            "PHYSICAL_K_RECEIPT": bool(scale_bridge_validation.get("PHYSICAL_K_RECEIPT", False)),
            "SCREEN_TO_PHYSICAL_K_ASSOCIATION_RECEIPT": bool(
                scale_bridge_validation.get("SCREEN_TO_PHYSICAL_K_ASSOCIATION_RECEIPT", False)
            ),
            "SOURCE_ANGULAR_SECTOR_RECEIPT": bool(
                scale_bridge_validation.get("SOURCE_ANGULAR_SECTOR_RECEIPT", False)
            ),
            "SOURCE_ANGULAR_MODE_RECEIPT": bool(
                scale_bridge_validation.get("SOURCE_ANGULAR_MODE_RECEIPT", False)
            ),
            "CALIBRATED_A_EVOLUTION_RECEIPT": bool(
                scale_bridge_validation.get("CALIBRATED_A_EVOLUTION_RECEIPT", False)
            ),
            "PHYSICAL_MODE_FREEZEOUT_MAP_RECEIPT": bool(
                scale_bridge_validation.get("PHYSICAL_MODE_FREEZEOUT_MAP_RECEIPT", False)
            ),
            "PHYSICAL_FREEZEOUT_SURFACE_RECEIPT": bool(
                scale_bridge_validation.get("PHYSICAL_FREEZEOUT_SURFACE_RECEIPT", False)
            ),
            "COMMON_PRIMORDIAL_ANOMALY_MODE_BASIS_RECEIPT": bool(
                scale_bridge_validation.get("COMMON_PRIMORDIAL_ANOMALY_MODE_BASIS_RECEIPT", False)
            ),
            "SCALE_BRIDGE_REFINEMENT_RECEIPT": bool(
                scale_bridge_validation.get("SCALE_BRIDGE_REFINEMENT_RECEIPT", False)
            ),
            "CROSS_RECEIPT_CONSISTENCY_RECEIPT": bool(
                scale_bridge_validation.get("CROSS_RECEIPT_CONSISTENCY_RECEIPT", False)
            ),
            "NO_POSTHOC_CALIBRATION_RECEIPT": bool(
                scale_bridge_validation.get("NO_POSTHOC_CALIBRATION_RECEIPT", False)
            ),
            "physical_gate_passed": bool(
                scale_bridge_validation.get("PHYSICAL_SCALE_BRIDGE_RECEIPT", False)
            ),
            "blockers": scale_bridge_validation.get("blockers") or [],
        },
        "cdm_limit_regression": {
            "passed": bool(contract.cdm_limit_regression_passed),
        },
        "official_likelihood": {
            "ready": bool(contract.official_likelihood_ready),
        },
        "source_provenance": {
            "receipt": bool(contract.source_provenance_receipt),
            "pooled_source_reducer_receipt": bool(contract.pooled_source_reducer_receipt),
            "contradiction_free_provenance_receipt": bool(
                contract.contradiction_free_provenance_receipt
            ),
            "N_CRC_consensus_invariant_receipt": bool(contract.N_CRC_consensus_invariant_receipt),
            "global_likelihood_reduction_receipt": bool(contract.global_likelihood_reduction_receipt),
            "physical_gate_passed": bool(
                contract.source_provenance_receipt
                and contract.pooled_source_reducer_receipt
                and contract.contradiction_free_provenance_receipt
                and contract.N_CRC_consensus_invariant_receipt
                and contract.global_likelihood_reduction_receipt
            ),
        },
        "finite_covariant_parent": {
            "parent_receipt": bool(contract.finite_covariant_parent_receipt),
            "stress_energy_closure_receipt": bool(contract.stress_energy_closure_receipt),
            "explicit_recipient_stress_receipt": bool(contract.explicit_recipient_stress_receipt),
            "exchange_current_closure_receipt": bool(contract.exchange_current_closure_receipt),
            "physical_clock_receipt": bool(contract.physical_clock_receipt),
            "active_fiber_receipt": bool(contract.active_fiber_receipt),
            "conserved_sector_decomposition_receipt": bool(
                contract.conserved_sector_decomposition_receipt
            ),
            "common_parent_response_pole_receipt": bool(contract.common_parent_response_pole_receipt),
            "gauge_independence_receipt": bool(contract.gauge_independence_receipt),
            "causal_response_receipt": bool(contract.causal_response_receipt),
            "refinement_convergence_receipt": bool(contract.refinement_convergence_receipt),
            "Gamma_rec_nonzero": _array_has_positive(contract.Gamma_rec_k_a),
            "physical_gate_passed": bool(
                contract.finite_covariant_parent_receipt
                and contract.stress_energy_closure_receipt
                and contract.gauge_independence_receipt
                and contract.causal_response_receipt
                and contract.refinement_convergence_receipt
                and (
                    not _array_has_positive(contract.Gamma_rec_k_a)
                    or (
                        contract.explicit_recipient_stress_receipt
                        and contract.exchange_current_closure_receipt
                    )
                )
                and contract.physical_clock_receipt
                and contract.active_fiber_receipt
                and contract.conserved_sector_decomposition_receipt
                and contract.common_parent_response_pole_receipt
            ),
        },
        "frozen_likelihood_protocol": {
            "receipt": bool(contract.frozen_likelihood_protocol_receipt),
            "source_freeze_manifest_receipt": bool(contract.source_freeze_manifest_receipt),
            "solver_assumption_pin_receipt": bool(contract.solver_assumption_pin_receipt),
            "custom_parent_cdm_limit_regression_receipt": bool(
                contract.custom_parent_cdm_limit_regression_receipt
            ),
            "standard_model_off_regression_receipt": bool(contract.standard_model_off_regression_receipt),
            "blinded_comparison_setup_receipt": bool(contract.blinded_comparison_setup_receipt),
            "full_observable_likelihood_receipt": bool(contract.full_observable_likelihood_receipt),
            "frozen_transfer_likelihood_receipt": bool(contract.frozen_transfer_likelihood_receipt),
            "source_hash_present": _nonempty_string(contract.frozen_source_hash),
            "solver_hash_present": _nonempty_string(contract.frozen_solver_hash),
            "likelihood_hash_present": _nonempty_string(contract.frozen_likelihood_hash),
            "physical_gate_passed": bool(
                contract.frozen_likelihood_protocol_receipt
                and contract.source_freeze_manifest_receipt
                and contract.solver_assumption_pin_receipt
                and contract.custom_parent_cdm_limit_regression_receipt
                and contract.standard_model_off_regression_receipt
                and contract.blinded_comparison_setup_receipt
                and contract.full_observable_likelihood_receipt
                and contract.frozen_transfer_likelihood_receipt
                and _nonempty_string(contract.frozen_source_hash)
                and _nonempty_string(contract.frozen_solver_hash)
                and _nonempty_string(contract.frozen_likelihood_hash)
            ),
        },
        "claim_boundary": (
            "Presence means the simulator emitted a finite diagnostic value. Physical-gate passage also "
            "requires the source label to be one of the hard finite CMB sources accepted by the contract, "
            "plus source-only provenance, pooled reducers, a finite covariant stress parent, and frozen "
            "source/solver/likelihood hashes. P and N are reported separately as theorem-side constants."
        ),
    }


def _theorem_constant_status(source: str) -> dict[str, Any]:
    theorem_side = str(source) in THEOREM_SIDE_SOURCES
    return {
        "source": source,
        "source_is_theorem_side_constant": theorem_side,
        "diagnostic_value_present": str(source) != "unknown",
        "physical_gate_passed": theorem_side,
        "finite_cmb_input": False,
        "claim_boundary": "Theorem-side constant provenance only; not a finite simulator-derived CMB input.",
    }


def _scalar_status(source: str, value: float | None, *, positive: bool = False) -> dict[str, Any]:
    finite = _float(value) is not None
    positive_ok = finite and (not positive or float(value) > 0.0)
    source_ok = str(source) in FINITE_CMB_SOURCES
    return {
        "source": source,
        "source_is_finite_cmb_source": source_ok,
        "diagnostic_value_present": finite,
        "positive_required": bool(positive),
        "positive_value": bool(positive_ok) if positive else None,
        "physical_gate_passed": bool(source_ok and positive_ok),
    }


def _array_status(source: str, value: np.ndarray | None, *, positive_column: int | None = None) -> dict[str, Any]:
    array = None if value is None else np.asarray(value, dtype=float)
    finite = bool(array is not None and array.size and np.all(np.isfinite(array)))
    positive_ok = True
    if finite and positive_column is not None:
        positive_ok = bool(
            array.ndim == 2
            and array.shape[1] > positive_column
            and np.all(array[:, positive_column] > 0.0)
        )
    source_ok = str(source) in FINITE_CMB_SOURCES
    return {
        "source": source,
        "source_is_finite_cmb_source": source_ok,
        "diagnostic_value_present": finite,
        "row_count": int(array.shape[0]) if finite and array.ndim >= 1 else 0,
        "column_count": int(array.shape[1]) if finite and array.ndim >= 2 else (1 if finite else 0),
        "positive_column": positive_column,
        "positive_values": bool(positive_ok) if positive_column is not None else None,
        "physical_gate_passed": bool(source_ok and finite and positive_ok),
    }


def _physical_cmb_promotion_blockers(
    validation: dict[str, Any],
    input_status: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    finite_collar_boltzmann: dict[str, Any],
    finite_collar_projection: dict[str, Any],
) -> list[str]:
    blockers = list(validation.get("blockers") or [])
    finite_cert = sources.get("finite_certificate_report") or {}
    ba_kernel = sources.get("B_A_kernel_report") or {}
    ba_kernel_refinement = sources.get("B_A_kernel_refinement_report") or {}
    if finite_cert.get("proxy_certificate", False) and not finite_cert.get("theorem_grade_finite_inputs", False):
        blockers.append("finite_certificate_proxy_not_theorem_grade")
    elif finite_cert and not finite_cert.get("theorem_grade_finite_inputs", False):
        blockers.append("finite_certificate_theorem_grade_inputs_missing")
    if not bool(ba_kernel.get("B_A_KERNEL_RECEIPT", False)):
        blockers.append("B_A_kernel_receipt_missing")
    if _ba_diagnostic_candidate(ba_kernel) and not ba_kernel.get("B_A_KERNEL_RECEIPT", False):
        blockers.append("B_A_kernel_candidate_not_physical")
        for blocker in ba_kernel.get("promotion_blockers") or []:
            blockers.append(f"B_A_kernel_{blocker}")
    if ba_kernel_refinement and not bool(
        ba_kernel_refinement.get("B_A_KERNEL_REFINEMENT_CONVERGENCE_RECEIPT", False)
    ):
        blockers.append("B_A_kernel_refinement_convergence_not_passed")
        for blocker in ba_kernel_refinement.get("blockers") or []:
            blockers.append(f"B_A_kernel_refinement_{blocker}")
    if _diagnostic_present_not_physical(input_status, "A_zeta"):
        blockers.append("A_zeta_diagnostic_proxy_not_physical_source")
    if _diagnostic_present_not_physical(input_status, "B_A_k_a"):
        blockers.append("B_A_diagnostic_rows_not_physical_kernel")
    if _diagnostic_present_not_physical(input_status, "rho_A_a"):
        blockers.append("rho_A_diagnostic_rows_not_physical_source")
        blockers.append("rho_A_eq_diagnostic_rows_not_physical_source")
    if _diagnostic_present_not_physical(input_status, "Gamma_rec_k_a"):
        blockers.append("Gamma_rec_diagnostic_rows_not_physical_source")
    rho_status = input_status.get("rho_A_a") or {}
    if rho_status.get("positive_values") is False:
        blockers.append("rho_A_nonpositive_source_values")
    if finite_collar_boltzmann:
        if not bool(finite_collar_boltzmann.get("PHYSICAL_BOLTZMANN_EXPORT_CERTIFICATE", False)):
            blockers.append("finite_collar_boltzmann_physical_certificate_false")
        for gate in _false_readiness_checks(finite_collar_boltzmann.get("readiness") or {}, nested_key="checks"):
            blockers.append(f"finite_collar_boltzmann_missing_{gate}")
    if finite_collar_projection:
        if not bool(finite_collar_projection.get("PHYSICAL_K_CALIBRATION_RECEIPT", False)):
            blockers.append("finite_collar_projection_physical_k_calibration_missing")
        for gate in _false_readiness_checks(finite_collar_projection.get("readiness") or {}):
            blockers.append(f"finite_collar_projection_missing_{gate}")
    return _unique_strings(blockers)


def _physical_cmb_source_readiness(
    sources: dict[str, dict[str, Any]],
    finite_collar_boltzmann: dict[str, Any],
    finite_collar_projection: dict[str, Any],
) -> dict[str, Any]:
    finite_cert = sources.get("finite_certificate_report") or {}
    finite_parent = sources.get("finite_covariant_collar_packet_parent_report") or {}
    ba_kernel = sources.get("B_A_kernel_report") or {}
    ba_kernel_refinement = sources.get("B_A_kernel_refinement_report") or {}
    ba_parent = sources.get("b_a_parent_report") or {}
    derived = finite_cert.get("derived_outputs") if isinstance(finite_cert.get("derived_outputs"), dict) else {}
    ba_rows = ba_parent.get("rows") or ba_parent.get("observer_view_rows") or []
    boltzmann_readiness = finite_collar_boltzmann.get("readiness") if isinstance(
        finite_collar_boltzmann.get("readiness"), dict
    ) else {}
    projection_readiness = finite_collar_projection.get("readiness") if isinstance(
        finite_collar_projection.get("readiness"), dict
    ) else {}
    return {
        "finite_certificate": {
            "present": bool(finite_cert),
            "compiler_ready": bool(finite_cert.get("finite_certificate_compiler_ready", False)),
            "stack_ready": bool(finite_cert.get("finite_certificate_stack_ready", False)),
            "theorem_grade_finite_inputs": bool(finite_cert.get("theorem_grade_finite_inputs", False)),
            "proxy_certificate": bool(finite_cert.get("proxy_certificate", False)),
            "A_zeta_available": derived.get("A_zeta") is not None or finite_cert.get("A_zeta") is not None,
            "rho_A_a_available": derived.get("rho_A_a") is not None or finite_cert.get("rho_A_a") is not None,
        },
        "finite_covariant_parent": {
            "present": bool(finite_parent),
            "parent_receipt": bool(finite_parent.get(PARENT_RECEIPT, False)),
            "stress_energy_closure_receipt": bool(finite_parent.get(STRESS_CLOSURE_RECEIPT, False)),
            "explicit_recipient_stress_receipt": bool(
                finite_parent.get(EXPLICIT_RECIPIENT_STRESS_RECEIPT, False)
            ),
            "exchange_current_closure_receipt": bool(
                finite_parent.get(EXCHANGE_CURRENT_CLOSURE_RECEIPT, False)
            ),
            "physical_clock_receipt": bool(
                finite_parent.get("PHYSICAL_CLOCK_RECEIPT", False)
                or finite_parent.get("PHYSICAL_REPAIR_CLOCK_RECEIPT", False)
            ),
            "active_fiber_receipt": bool(
                finite_parent.get("ACTIVE_FIBER_RECEIPT", False)
                or finite_parent.get("ACTIVE_FIBER_RESPONSE_RECEIPT", False)
            ),
            "conserved_sector_decomposition_receipt": bool(
                finite_parent.get("CONSERVED_SECTOR_DECOMPOSITION_RECEIPT", False)
            ),
            "common_parent_response_pole_receipt": bool(
                finite_parent.get("COMMON_PARENT_RESPONSE_POLE_RECEIPT", False)
                or finite_parent.get("COMMON_PARENT_RESPONSE_RECEIPT", False)
            ),
            "gauge_independence_receipt": bool(finite_parent.get(GAUGE_INDEPENDENCE_RECEIPT, False)),
            "causal_response_receipt": bool(finite_parent.get(CAUSAL_RESPONSE_RECEIPT, False)),
            "refinement_convergence_receipt": bool(finite_parent.get(REFINEMENT_CONVERGENCE_RECEIPT, False)),
            "frozen_likelihood_protocol_receipt": bool(
                finite_parent.get(FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT, False)
            ),
            "source_hash_present": _nonempty_string(finite_parent.get("source_hash")),
            "solver_hash_present": _nonempty_string(finite_parent.get("solver_hash")),
            "likelihood_hash_present": _nonempty_string(finite_parent.get("likelihood_hash")),
            "Gamma_rec_nonzero": bool(finite_parent.get("Gamma_rec_nonzero", False)),
            "blockers": finite_parent.get("blockers") or [],
        },
        "B_A_kernel": {
            "present": bool(ba_kernel),
            "B_A_KERNEL_RECEIPT": bool(ba_kernel.get("B_A_KERNEL_RECEIPT", False)),
            "B_A_DIAGNOSTIC_CANDIDATE_RECEIPT": bool(
                ba_kernel.get("B_A_DIAGNOSTIC_CANDIDATE_RECEIPT", False)
            ),
            "B_A_KERNEL_CANDIDATE_RECEIPT": bool(ba_kernel.get("B_A_KERNEL_CANDIDATE_RECEIPT", False)),
            "row_count": int(ba_kernel.get("row_count") or 0),
            "promotion_blockers": ba_kernel.get("promotion_blockers") or [],
        },
        "B_A_kernel_refinement": {
            "present": bool(ba_kernel_refinement),
            "two_scale_diagnostic_receipt": bool(
                ba_kernel_refinement.get("two_scale_diagnostic_receipt", False)
            ),
            "B_A_KERNEL_REFINEMENT_CONVERGENCE_RECEIPT": bool(
                ba_kernel_refinement.get("B_A_KERNEL_REFINEMENT_CONVERGENCE_RECEIPT", False)
            ),
            "patch_counts": ba_kernel_refinement.get("patch_counts") or [],
            "blockers": ba_kernel_refinement.get("blockers") or [],
        },
        "B_A_parent": {
            "present": bool(ba_parent),
            "diagnostic_row_count": len(ba_rows) if isinstance(ba_rows, list) else 0,
            "B_A_PARENT_RECEIPT": bool(ba_parent.get("B_A_PARENT_RECEIPT", False)),
        },
        "finite_collar_boltzmann": {
            "present": bool(finite_collar_boltzmann),
            "source_bundle_receipt": bool(
                finite_collar_boltzmann.get("FINITE_COLLAR_BOLTZMANN_SOURCE_BUNDLE_RECEIPT", False)
            ),
            "physical_certificate": bool(
                finite_collar_boltzmann.get("PHYSICAL_BOLTZMANN_EXPORT_CERTIFICATE", False)
            ),
            "missing_readiness_checks": _false_readiness_checks(boltzmann_readiness, nested_key="checks"),
        },
        "finite_collar_projection": {
            "present": bool(finite_collar_projection),
            "projection_diagnostic_receipt": bool(
                finite_collar_projection.get("FINITE_COLLAR_CMB_PROJECTION_DIAGNOSTIC_RECEIPT", False)
            ),
            "physical_k_calibration": bool(
                finite_collar_projection.get("PHYSICAL_K_CALIBRATION_RECEIPT", False)
            ),
            "missing_readiness_checks": _false_readiness_checks(projection_readiness),
        },
    }


def _physical_cmb_next_steps(blockers: list[str]) -> list[dict[str, str]]:
    suggestions = {
        "finite_covariant_parent_receipt_missing": (
            "Supply a finite covariant collar-packet parent receipt before treating scalar source tables as physical."
        ),
        "stress_energy_closure_not_certified": (
            "Certify exact stress-energy closure of anomaly and recipient packet stresses."
        ),
        "recipient_stress_missing_for_nonzero_Gamma_rec": (
            "Add explicit recipient stress when Gamma_rec is nonzero; repair exchange cannot disappear into a scalar sink."
        ),
        "exchange_current_closure_missing_for_nonzero_Gamma_rec": (
            "Add equal-and-opposite exchange-current closure for the nonzero repair exchange branch."
        ),
        "physical_clock_missing_for_promoted_Gamma_rec": (
            "Certify the physical clock that converts repair-step decay into a Gamma_rec(k,a) kernel."
        ),
        "active_fiber_missing_for_promoted_Gamma_rec": (
            "Certify the active fiber used by the promoted Gamma_rec response calculation."
        ),
        "conserved_sector_decomposition_missing_for_promoted_Gamma_rec": (
            "Certify the conserved-sector decomposition for the promoted repair-response kernel."
        ),
        "common_parent_response_pole_missing_for_promoted_Gamma_rec": (
            "Certify the common-parent response pole before promoting gamma_repair_step to Gamma_rec."
        ),
        "gauge_independence_not_certified": (
            "Certify B_A(k,a) with the gauge-invariant baryon density measured in the anomaly frame."
        ),
        "causal_response_not_certified": (
            "Supply a causal auxiliary-response receipt with subluminal characteristics."
        ),
        "refinement_convergence_not_certified": (
            "Run the finite-parent regulator refinement ladder and freeze the converged source tables."
        ),
        "frozen_likelihood_protocol_not_certified": (
            "Freeze immutable source, solver, and likelihood hashes before running physical likelihood comparisons."
        ),
        "source_freeze_manifest_not_certified": (
            "Emit the frozen source allowlist manifest from source-side OPH reports only."
        ),
        "solver_assumption_pin_not_certified": (
            "Pin the Boltzmann solver version, recombination assumptions, neutrino assumptions, tolerances, and source-plugin hash."
        ),
        "custom_parent_cdm_limit_regression_not_passed": (
            "Run CMB1 custom-parent CDM-limit regression against the solver-native CDM baseline."
        ),
        "standard_model_off_regression_not_passed": (
            "Run the Standard-Model-off control transfer and verify the expected null/delta tolerance."
        ),
        "blinded_comparison_setup_not_certified": (
            "Freeze the blinded comparison setup before opening TT/TE/EE/lensing/BAO/growth/weak-lensing/RSD/S8 comparisons."
        ),
        "full_observable_likelihood_not_executed": (
            "Execute the full official likelihood/readout suite after source freeze and blinded setup."
        ),
        "frozen_transfer_likelihood_closure_not_certified": (
            "Pass the frozen transfer/likelihood closure report before promoting CMB outputs."
        ),
        "finite_certificate_proxy_not_theorem_grade": (
            "Replace proxy finite-certificate inputs with theorem-grade finite A_zeta/rho_A receipts."
        ),
        "B_A_kernel_receipt_missing": (
            "Run or supply the paired finite-difference B_A(k,a) kernel receipt instead of parent-row diagnostics."
        ),
        "finite_collar_projection_physical_k_calibration_missing": (
            "Close the finite screen-to-bulk scale calibration so projected ell/k axes are OPH-derived."
        ),
        "official_likelihood_not_ready": (
            "Connect the OPH source tables to the official likelihood path after finite-source gates pass."
        ),
        "source_provenance_receipt_missing": (
            "Emit a source-only dependency-DAG receipt for eta_R, Gamma_rec, A_zeta, q_IR, ell_IR, B_A, rho_A, and N_CRC."
        ),
        "pooled_source_reducer_receipt_missing": (
            "Pool additive sufficient statistics before nonlinear source estimates, or declare a single global source."
        ),
        "source_provenance_contradiction_check_failed": (
            "Remove contradictory provenance such as no_cmb_data_used together with fit_to_planck."
        ),
        "N_CRC_consensus_invariant_receipt_missing": (
            "Treat N_CRC as a consensus invariant unless an additive capacity schema proves disjoint coverage."
        ),
        "global_likelihood_reduction_receipt_missing": (
            "Use global official-likelihood and CDM-limit reductions, not shard-local any() rollups."
        ),
        "finite_collar_boltzmann_missing_refinement_convergence_passed": (
            "Run regulator/refinement convergence for the finite-collar source tables."
        ),
        "finite_collar_boltzmann_missing_energy_momentum_exchange_closed": (
            "Audit energy-momentum exchange closure for the finite-collar Boltzmann source bundle."
        ),
    }
    rows = []
    for blocker in blockers:
        rows.append(
            {
                "blocker": blocker,
                "next_step": suggestions.get(blocker, "Clear this hard-gate blocker with a finite-source receipt."),
            }
        )
    return rows


def _physical_cmb_frontier_gate_rows(
    *,
    validation: dict[str, Any],
    input_status: dict[str, Any],
    promotion: dict[str, Any],
    physical_output: dict[str, Any],
    source_readiness: dict[str, Any],
    contract: PhysicalCMBInputContract,
) -> list[dict[str, Any]]:
    finite_certificate = source_readiness.get("finite_certificate") or {}
    finite_parent = source_readiness.get("finite_covariant_parent") or {}
    b_a_kernel = source_readiness.get("B_A_kernel") or {}
    b_a_refinement = source_readiness.get("B_A_kernel_refinement") or {}
    finite_collar_boltzmann = source_readiness.get("finite_collar_boltzmann") or {}
    finite_collar_projection = source_readiness.get("finite_collar_projection") or {}
    return [
        {
            "gate": "measurement_comparable_cmb_outputs",
            "passed": bool(physical_output.get("PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT", False)),
            "detail": (
                f"{int(physical_output.get('oph_diagnostic_model_count') or 0)} OPH diagnostic models; "
                f"{int(physical_output.get('measurement_comparable_model_count') or 0)} total comparable models"
            ),
        },
        {
            "gate": "no_data_use_firewall",
            "passed": bool(contract.no_data_use_receipt),
            "detail": "OPH input functions were assembled without measurement tables",
        },
        {
            "gate": "source_provenance_firewall",
            "passed": bool(contract.source_provenance_receipt),
            "detail": "promoted CMB inputs have source-only dependency-DAG provenance",
        },
        {
            "gate": "pooled_source_reducers",
            "passed": bool(contract.pooled_source_reducer_receipt),
            "detail": "nonlinear CMB input estimates are global, not shard-local averages",
        },
        {
            "gate": "N_CRC_consensus_invariant",
            "passed": bool(contract.N_CRC_consensus_invariant_receipt),
            "detail": "N_CRC is not additively combined without a disjoint-capacity proof",
        },
        {
            "gate": "finite_theorem_A_zeta",
            "passed": bool((input_status.get("A_zeta") or {}).get("physical_gate_passed", False)),
            "detail": f"source={(input_status.get('A_zeta') or {}).get('source')}",
        },
        {
            "gate": "finite_B_A_kernel",
            "passed": bool((input_status.get("B_A_k_a") or {}).get("physical_gate_passed", False)),
            "detail": f"rows={(input_status.get('B_A_k_a') or {}).get('row_count', 0)}",
        },
        {
            "gate": "finite_rho_A",
            "passed": bool((input_status.get("rho_A_a") or {}).get("physical_gate_passed", False)),
            "detail": f"rows={(input_status.get('rho_A_a') or {}).get('row_count', 0)}",
        },
        {
            "gate": "finite_certificate_theorem_grade",
            "passed": bool(finite_certificate.get("theorem_grade_finite_inputs", False)),
            "detail": (
                f"compiler_ready={finite_certificate.get('compiler_ready')}; "
                f"proxy={finite_certificate.get('proxy_certificate')}"
            ),
        },
        {
            "gate": "finite_covariant_parent",
            "passed": bool(finite_parent.get("parent_receipt", False)),
            "detail": f"blockers={finite_parent.get('blockers', [])}",
        },
        {
            "gate": "stress_energy_closure",
            "passed": bool(finite_parent.get("stress_energy_closure_receipt", False)),
            "detail": "anomaly plus recipient stress tensors close exactly",
        },
        {
            "gate": "recipient_stress_for_nonzero_Gamma_rec",
            "passed": bool(
                (not _array_has_positive(contract.Gamma_rec_k_a))
                or finite_parent.get("explicit_recipient_stress_receipt", False)
            ),
            "detail": f"Gamma_rec_nonzero={_array_has_positive(contract.Gamma_rec_k_a)}",
        },
        {
            "gate": "exchange_current_closure_for_nonzero_Gamma_rec",
            "passed": bool(
                (not _array_has_positive(contract.Gamma_rec_k_a))
                or finite_parent.get("exchange_current_closure_receipt", False)
            ),
            "detail": f"Gamma_rec_nonzero={_array_has_positive(contract.Gamma_rec_k_a)}",
        },
        {
            "gate": "physical_clock_for_Gamma_rec",
            "passed": bool(contract.physical_clock_receipt),
            "detail": "repair-step transition rate has a certified physical clock",
        },
        {
            "gate": "active_fiber_for_Gamma_rec",
            "passed": bool(contract.active_fiber_receipt),
            "detail": "Gamma_rec response is calculated on the certified active fiber",
        },
        {
            "gate": "conserved_sector_decomposition_for_Gamma_rec",
            "passed": bool(contract.conserved_sector_decomposition_receipt),
            "detail": "repair exchange is decomposed against the conserved sectors before promotion",
        },
        {
            "gate": "common_parent_response_pole_for_Gamma_rec",
            "passed": bool(contract.common_parent_response_pole_receipt),
            "detail": "response pole is read in a common finite parent",
        },
        {
            "gate": "gauge_independent_B_A",
            "passed": bool(finite_parent.get("gauge_independence_receipt", False)),
            "detail": "B_A must be built from anomaly-frame baryon density, not a gauge-fixed scalar table",
        },
        {
            "gate": "causal_response",
            "passed": bool(finite_parent.get("causal_response_receipt", False)),
            "detail": "finite packet or auxiliary response characteristics are subluminal",
        },
        {
            "gate": "finite_parent_refinement_convergence",
            "passed": bool(finite_parent.get("refinement_convergence_receipt", False)),
            "detail": "stress, response, and source projections converge under regulator refinement",
        },
        {
            "gate": "B_A_kernel_physical_receipt",
            "passed": bool(b_a_kernel.get("B_A_KERNEL_RECEIPT", False)),
            "detail": (
                f"diagnostic_candidate={_ba_diagnostic_candidate(b_a_kernel)}; "
                f"rows={b_a_kernel.get('row_count', 0)}"
            ),
        },
        {
            "gate": "B_A_refinement_convergence",
            "passed": bool(b_a_refinement.get("B_A_KERNEL_REFINEMENT_CONVERGENCE_RECEIPT", False)),
            "detail": f"patch_counts={b_a_refinement.get('patch_counts', [])}",
        },
        {
            "gate": "finite_collar_boltzmann_physical_certificate",
            "passed": bool(finite_collar_boltzmann.get("physical_certificate", False)),
            "detail": f"missing={finite_collar_boltzmann.get('missing_readiness_checks', [])}",
        },
        {
            "gate": "finite_collar_projection_physical_k",
            "passed": bool(finite_collar_projection.get("physical_k_calibration", False)),
            "detail": f"missing={finite_collar_projection.get('missing_readiness_checks', [])}",
        },
        {
            "gate": "cdm_limit_regression",
            "passed": bool(contract.cdm_limit_regression_passed and contract.custom_parent_cdm_limit_regression_receipt),
            "detail": (
                "Boltzmann plumbing/CDM-limit regression gate; requires CMB1 custom-parent CDM "
                "match against solver-native CDM"
            ),
        },
        {
            "gate": "standard_model_off_regression",
            "passed": bool(contract.standard_model_off_regression_receipt),
            "detail": "control transfer with Standard Model/anomaly sources off",
        },
        {
            "gate": "official_planck_likelihood_ready",
            "passed": bool(contract.official_likelihood_ready),
            "detail": "requires local official clik/Cobaya path and Planck likelihood data",
        },
        {
            "gate": "blinded_full_observable_likelihood",
            "passed": bool(contract.blinded_comparison_setup_receipt and contract.full_observable_likelihood_receipt),
            "detail": "requires blinded TT/TE/EE/lensing/BAO/growth/weak-lensing/RSD/S8 execution",
        },
        {
            "gate": "frozen_likelihood_protocol",
            "passed": bool(
                contract.frozen_likelihood_protocol_receipt
                and contract.source_freeze_manifest_receipt
                and contract.solver_assumption_pin_receipt
                and contract.frozen_transfer_likelihood_receipt
                and _nonempty_string(contract.frozen_source_hash)
                and _nonempty_string(contract.frozen_solver_hash)
                and _nonempty_string(contract.frozen_likelihood_hash)
            ),
            "detail": (
                f"source_hash={_nonempty_string(contract.frozen_source_hash)}; "
                f"solver_hash={_nonempty_string(contract.frozen_solver_hash)}; "
                f"likelihood_hash={_nonempty_string(contract.frozen_likelihood_hash)}; "
                f"source_freeze={contract.source_freeze_manifest_receipt}; "
                f"solver_pins={contract.solver_assumption_pin_receipt}; "
                f"closure={contract.frozen_transfer_likelihood_receipt}"
            ),
        },
        {
            "gate": "physical_cmb_input_contract",
            "passed": bool(validation.get("PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT", False)),
            "detail": f"{len(validation.get('blockers') or [])} hard input blockers",
        },
        {
            "gate": "physical_cmb_promotion_ready",
            "passed": bool(promotion.get("physical_cmb_promotion_ready", False)),
            "detail": f"{len(promotion.get('promotion_blockers') or [])} promotion blockers",
        },
    ]


def _physical_cmb_frontier_gap_rows(
    *,
    gate_rows: list[dict[str, Any]],
    blockers: list[str],
    input_status: dict[str, Any],
    source_readiness: dict[str, Any],
    physical_output: dict[str, Any],
) -> list[dict[str, Any]]:
    gate_status = {str(row.get("gate")): bool(row.get("passed", False)) for row in gate_rows}
    finite_certificate = source_readiness.get("finite_certificate") or {}
    finite_parent = source_readiness.get("finite_covariant_parent") or {}
    b_a_kernel = source_readiness.get("B_A_kernel") or {}
    b_a_refinement = source_readiness.get("B_A_kernel_refinement") or {}
    finite_collar_boltzmann = source_readiness.get("finite_collar_boltzmann") or {}
    finite_collar_projection = source_readiness.get("finite_collar_projection") or {}

    def status(key: str) -> dict[str, Any]:
        return input_status.get(key) if isinstance(input_status.get(key), dict) else {}

    rows = [
        {
            "gate": "measurement_comparable_cmb_outputs",
            "passed": gate_status.get("measurement_comparable_cmb_outputs", False),
            "missing_receipt": None,
            "current_evidence": (
                f"{int(physical_output.get('oph_diagnostic_model_count') or 0)} OPH diagnostics / "
                f"{int(physical_output.get('measurement_comparable_model_count') or 0)} comparable models"
            ),
            "action_surface": "measurement_output",
            "blockers": [],
        },
        {
            "gate": "source_provenance_firewall",
            "passed": gate_status.get("source_provenance_firewall", False),
            "missing_receipt": "CMB source-provenance dependency-DAG receipt",
            "current_evidence": (
                f"receipt={status('source_provenance').get('receipt')}; "
                f"pooled={status('source_provenance').get('pooled_source_reducer_receipt')}; "
                f"N_CRC={status('source_provenance').get('N_CRC_consensus_invariant_receipt')}"
            ),
            "action_surface": "cmb_source_provenance_report",
            "blockers": [
                blocker for blocker in blockers
                if blocker
                in {
                    "source_provenance_receipt_missing",
                    "pooled_source_reducer_receipt_missing",
                    "source_provenance_contradiction_check_failed",
                    "N_CRC_consensus_invariant_receipt_missing",
                    "global_likelihood_reduction_receipt_missing",
                }
                or blocker.startswith("contradictory_no_data_use_provenance")
                or blocker.endswith("_source_reducer_missing")
                or blocker.endswith("_not_source_derived")
            ],
        },
        {
            "gate": "finite_theorem_A_zeta",
            "passed": gate_status.get("finite_theorem_A_zeta", False),
            "missing_receipt": "theorem-grade finite A_zeta source",
            "current_evidence": (
                f"source={status('A_zeta').get('source')}; "
                f"diagnostic_present={status('A_zeta').get('diagnostic_value_present')}; "
                f"finite_certificate_theorem_grade={finite_certificate.get('theorem_grade_finite_inputs')}"
            ),
            "action_surface": "finite_certificate_report",
            "blockers": [
                blocker for blocker in blockers
                if blocker in {"A_zeta_not_finite_derived", "finite_certificate_proxy_not_theorem_grade"}
            ],
        },
        {
            "gate": "finite_B_A_kernel",
            "passed": gate_status.get("finite_B_A_kernel", False),
            "missing_receipt": "physical finite B_A(k,a) kernel receipt",
            "current_evidence": (
                f"source={status('B_A_k_a').get('source')}; rows={status('B_A_k_a').get('row_count')}; "
                f"B_A_KERNEL_RECEIPT={b_a_kernel.get('B_A_KERNEL_RECEIPT')}; "
                f"refinement={b_a_refinement.get('B_A_KERNEL_REFINEMENT_CONVERGENCE_RECEIPT')}"
            ),
            "action_surface": "B_A_kernel_report/B_A_kernel_refinement_report",
            "blockers": [
                blocker for blocker in blockers
                if blocker.startswith("B_A") or blocker == "B_A_k_a_missing_or_not_finite"
            ],
        },
        {
            "gate": "finite_rho_A",
            "passed": gate_status.get("finite_rho_A", False),
            "missing_receipt": "theorem-grade finite rho_A(a) source",
            "current_evidence": (
                f"source={status('rho_A_a').get('source')}; rows={status('rho_A_a').get('row_count')}; "
                f"finite_certificate_theorem_grade={finite_certificate.get('theorem_grade_finite_inputs')}"
            ),
            "action_surface": "finite_certificate_report/parent_collar_ladder",
            "blockers": [
                blocker for blocker in blockers
                if blocker in {"rho_A_missing_or_not_finite", "rho_A_diagnostic_rows_not_physical_source"}
            ],
        },
        {
            "gate": "finite_covariant_parent",
            "passed": gate_status.get("finite_covariant_parent", False),
            "missing_receipt": "finite covariant collar-packet parent receipt",
            "current_evidence": (
                f"present={finite_parent.get('present')}; parent_receipt={finite_parent.get('parent_receipt')}; "
                f"blockers={finite_parent.get('blockers')}"
            ),
            "action_surface": "finite_covariant_collar_packet_parent_report",
            "blockers": [
                blocker for blocker in blockers
                if blocker
                in {
                    "finite_covariant_parent_receipt_missing",
                    "stress_energy_closure_not_certified",
                    "recipient_stress_missing_for_nonzero_Gamma_rec",
                    "exchange_current_closure_missing_for_nonzero_Gamma_rec",
                    "physical_clock_missing_for_promoted_Gamma_rec",
                    "active_fiber_missing_for_promoted_Gamma_rec",
                    "conserved_sector_decomposition_missing_for_promoted_Gamma_rec",
                    "common_parent_response_pole_missing_for_promoted_Gamma_rec",
                    "gauge_independence_not_certified",
                    "causal_response_not_certified",
                    "refinement_convergence_not_certified",
                }
            ],
        },
        {
            "gate": "frozen_likelihood_protocol",
            "passed": gate_status.get("frozen_likelihood_protocol", False),
            "missing_receipt": "frozen immutable source/solver/likelihood hash protocol",
            "current_evidence": (
                f"receipt={finite_parent.get('frozen_likelihood_protocol_receipt')}; "
                f"source_hash={finite_parent.get('source_hash_present')}; "
                f"solver_hash={finite_parent.get('solver_hash_present')}; "
                f"likelihood_hash={finite_parent.get('likelihood_hash_present')}"
            ),
            "action_surface": "finite_covariant_collar_packet_parent_report/official_likelihood_report",
            "blockers": [
                blocker for blocker in blockers
                if blocker
                in {
                    "frozen_likelihood_protocol_not_certified",
                    "source_freeze_manifest_not_certified",
                    "solver_assumption_pin_not_certified",
                    "custom_parent_cdm_limit_regression_not_passed",
                    "standard_model_off_regression_not_passed",
                    "blinded_comparison_setup_not_certified",
                    "full_observable_likelihood_not_executed",
                    "frozen_transfer_likelihood_closure_not_certified",
                    "frozen_source_hash_missing",
                    "frozen_solver_hash_missing",
                    "frozen_likelihood_hash_missing",
                }
            ],
        },
        {
            "gate": "finite_collar_boltzmann_physical_certificate",
            "passed": gate_status.get("finite_collar_boltzmann_physical_certificate", False),
            "missing_receipt": "finite-collar Boltzmann physical certificate",
            "current_evidence": (
                f"source_bundle={finite_collar_boltzmann.get('source_bundle_receipt')}; "
                f"missing={finite_collar_boltzmann.get('missing_readiness_checks')}"
            ),
            "action_surface": "finite_collar_boltzmann_bundle_report",
            "blockers": [
                blocker for blocker in blockers
                if blocker.startswith("finite_collar_boltzmann")
            ],
        },
        {
            "gate": "finite_collar_projection_physical_k",
            "passed": gate_status.get("finite_collar_projection_physical_k", False),
            "missing_receipt": "OPH-derived physical k/ell calibration",
            "current_evidence": (
                f"projection={finite_collar_projection.get('projection_diagnostic_receipt')}; "
                f"physical_k={finite_collar_projection.get('physical_k_calibration')}; "
                f"missing={finite_collar_projection.get('missing_readiness_checks')}"
            ),
            "action_surface": "finite_collar_cmb_projection_report/scale_bridge_report",
            "blockers": [
                blocker for blocker in blockers
                if blocker.startswith("finite_collar_projection")
            ],
        },
        {
            "gate": "official_planck_likelihood_ready",
            "passed": gate_status.get("official_planck_likelihood_ready", False),
            "missing_receipt": "official clik/Cobaya likelihood execution readiness",
            "current_evidence": "official readiness false or data paths not configured",
            "action_surface": "local Planck likelihood environment",
            "blockers": [
                blocker for blocker in blockers
                if blocker == "official_likelihood_not_ready"
            ],
        },
    ]
    return [
        {
            **row,
            "blocking": bool(not row["passed"] or row["blockers"]),
            "claim_boundary": (
                "Physical-CMB hard-gate gap row. It identifies the missing receipt required for "
                "promotion; it does not allow diagnostic rows to satisfy finite-source gates."
            ),
        }
        for row in rows
        if (not row["passed"]) or row["blockers"]
    ]


def _diagnostic_present_not_physical(input_status: dict[str, Any], key: str) -> bool:
    status = input_status.get(key) if isinstance(input_status.get(key), dict) else {}
    return bool(status.get("diagnostic_value_present", False) and not status.get("physical_gate_passed", False))


def _ba_diagnostic_candidate(report: dict[str, Any]) -> bool:
    return bool(
        report.get("B_A_DIAGNOSTIC_CANDIDATE_RECEIPT", False)
        or report.get("B_A_KERNEL_CANDIDATE_RECEIPT", False)
    )


def _false_readiness_checks(readiness: dict[str, Any], *, nested_key: str | None = None) -> list[str]:
    checks: Any = readiness
    if nested_key is not None:
        checks = readiness.get(nested_key) if isinstance(readiness, dict) else {}
    if not isinstance(checks, dict):
        return []
    return sorted(str(key) for key, value in checks.items() if value is False)


def _unique_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _array_has_positive(value: np.ndarray | None) -> bool:
    if value is None:
        return False
    array = np.asarray(value, dtype=float)
    if not array.size or not np.all(np.isfinite(array)):
        return False
    return bool(np.any(array > 0.0))


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _N_source_from_screen_capacity(screen_capacity: dict[str, Any]) -> str:
    if not screen_capacity:
        return "unknown"
    if screen_capacity.get("screen_capacity_closure_receipt", False) or screen_capacity.get(
        "SCREEN_CAPACITY_CLOSURE_RECEIPT", False
    ):
        return "OPH_screen_capacity_branch_predeclared"
    observed = screen_capacity.get("observed_branch_normalization") or {}
    gates = screen_capacity.get("readiness_gates") or {}
    if observed.get("N_CRC") is not None and gates.get("observed_branch_N_scr_readout_available", False):
        return "OPH_screen_capacity_observed_branch_readout"
    return "unknown"


def _contract_to_jsonable(contract: PhysicalCMBInputContract) -> dict[str, Any]:
    data = asdict(contract)
    for key in ("B_A_k_a", "Gamma_rec_k_a", "rho_A_a"):
        value = data.get(key)
        if value is not None:
            data[key] = np.asarray(value, dtype=float).tolist()
    return data


def _write_array(path: Path, array: np.ndarray | None, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        if array is None:
            return
        arr = np.asarray(array, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        for i in range(arr.shape[0]):
            if arr.shape[1] >= 3:
                writer.writerow([arr[i, 0], arr[i, 1], arr[i, 2]])
            elif arr.shape[1] == 2:
                writer.writerow([i, arr[i, 0], arr[i, 1]])
            else:
                writer.writerow([i, 0, arr[i, 0]])


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Physical CMB Input Contract",
        "",
        f"- contract receipt: `{str(report['PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT']).lower()}`",
        f"- physical CMB prediction eligible: `{str(report['physical_cmb_prediction_eligible']).lower()}`",
        f"- physical CMB prediction: `{str(report['physical_cmb_prediction']).lower()}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = report.get("blockers") or []
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")
    lines.extend(["", "## Claim Boundary", "", report.get("claim_boundary", ""), ""])
    return "\n".join(lines)


def _markdown_promotion_audit(report: dict[str, Any]) -> str:
    blockers = report.get("promotion_blockers") or []
    lines = [
        "# Physical CMB Promotion Audit",
        "",
        f"- promotion ready: `{str(report.get('physical_cmb_promotion_ready', False)).lower()}`",
        f"- input contract receipt: `{str(report.get('physical_cmb_input_contract_receipt', False)).lower()}`",
        f"- no-data receipt: `{str(report.get('no_data_use_receipt', False)).lower()}`",
        f"- CDM-limit regression: `{str(report.get('cdm_limit_regression_passed', False)).lower()}`",
        f"- official likelihood ready: `{str(report.get('official_likelihood_ready', False)).lower()}`",
        "",
        "## Promotion Blockers",
        "",
    ]
    lines.extend(f"- `{blocker}`" for blocker in blockers) if blockers else lines.append("- none")
    lines.extend(["", "## Claim Boundary", "", str(report.get("claim_boundary", "")), ""])
    return "\n".join(lines)


def _markdown_physical_cmb_frontier(report: dict[str, Any]) -> str:
    blockers = report.get("blockers") or []
    lines = [
        "# Physical CMB Frontier",
        "",
        f"- measurement-comparable output receipt: `{str(report.get('physical_cmb_output_comparison_receipt', False)).lower()}`",
        f"- physical CMB prediction receipt: `{str(report.get('physical_cmb_prediction_receipt', False)).lower()}`",
        f"- physical CMB prediction ready: `{str(report.get('physical_cmb_prediction_ready', False)).lower()}`",
        f"- official likelihood ready: `{str(report.get('official_likelihood_ready', False)).lower()}`",
        f"- CDM-limit regression: `{str(report.get('cdm_limit_regression_passed', False)).lower()}`",
        "",
        "## Gates",
        "",
    ]
    for row in report.get("gate_rows") or []:
        lines.append(
            f"- `{row.get('gate')}`: `{str(row.get('passed', False)).lower()}` - {row.get('detail', '')}"
        )
    lines.extend(["", "## Hard-Gate Gaps", ""])
    gap_rows = report.get("gate_gap_rows") or []
    if gap_rows:
        for row in gap_rows:
            blockers = ", ".join(str(blocker) for blocker in (row.get("blockers") or [])) or "none"
            lines.append(
                f"- `{row.get('gate')}`: missing `{row.get('missing_receipt')}`; "
                f"current `{row.get('current_evidence')}`; action `{row.get('action_surface')}`; "
                f"blockers `{blockers}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- `{blocker}`" for blocker in blockers) if blockers else lines.append("- none")
    best = report.get("best_oph_diagnostic_model") or {}
    lines.extend(
        [
            "",
            "## Best OPH Diagnostic Output",
            "",
            f"- model: `{best.get('model_id')}`",
            f"- chi2/bin: `{best.get('amplitude_fit_chi2_per_bin')}`",
            f"- source report: `{best.get('source_report')}`",
            "",
            "## Claim Boundary",
            "",
            str(report.get("claim_boundary", "")),
            "",
        ]
    )
    return "\n".join(lines)


def _first_json(roots: list[Path], name: str) -> dict[str, Any]:
    fallback_empty: dict[str, Any] | None = None
    for root in roots:
        root = Path(root)
        candidates = [root / name]
        if root.exists() and root.is_dir():
            candidates.extend(sorted(root.glob(f"**/{name}")))
        for path in candidates:
            if path.exists() and path.is_file():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    if data:
                        return data
                    fallback_empty = data
    return fallback_empty or {}


def _truthy_any(data: dict[str, Any], *keys: str) -> bool:
    return any(bool(data.get(key, False)) for key in keys)


def _explicit_no_cmb_data_used(report: dict[str, Any]) -> bool:
    return bool(
        report.get("no_cmb_data_used") is True
        or (((report.get("readiness") or {}).get("checks") or {}).get("no_cmb_data_used") is True)
    )


def _raw_measurement_data_used(report: dict[str, Any]) -> bool:
    if not report:
        return False
    data_use_keys = (
        "measurement_data_used",
        "cmb_data_used",
        "cmb_data_used_for_input",
        "planck_data_used_for_input",
        "fit_to_measurement",
        "fit_to_planck",
        "uses_measurements_to_set_inputs",
    )
    return any(bool(report.get(key, False)) for key in data_use_keys)


def _measurement_data_used(report: dict[str, Any]) -> bool:
    if not report:
        return False
    explicit_false = _explicit_no_cmb_data_used(report)
    raw_measurement_use = _raw_measurement_data_used(report)
    if explicit_false and raw_measurement_use:
        return True
    if explicit_false:
        return False
    return raw_measurement_use


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    return array if array.size and np.all(np.isfinite(array)) else None
