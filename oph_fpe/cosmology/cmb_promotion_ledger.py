from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from oph_fpe.cosmology.cosmological_scale_bridge import validate_physical_scale_bridge_receipts


CLAIM_LADDER = (
    "VISUAL_DIAGNOSTIC",
    "SPECTRUM_DIAGNOSTIC",
    "SOURCE_ONLY_FINITE_ARTIFACT",
    "CONDITIONAL_PHYSICAL_CMB_SOURCE",
    "OPH_NATIVE_PHYSICAL_CMB_SOURCE",
    "LIKELIHOOD_EVALUATED_PHYSICAL_CMB_PREDICTION",
)

PHYSICAL_PREDICTION_PARENT_GATES = (
    "FINITE_QUOTIENT_SOURCE_LAW_RECEIPT",
    "SOURCE_ONLY_FINITE_ARTIFACT_RECEIPT",
    "NO_DATA_USE_RECEIPT",
    "POOLED_SOURCE_REDUCER_RECEIPT",
    "GEOMETRIC_SCREEN_SCALAR_RECEIPT",
    "PHYSICAL_PRECISION_OPERATOR_RECEIPT",
    "SOURCE_RELEASE_AMPLITUDE_RECEIPT",
    "SCREEN_COVARIANCE_RECEIPT",
    "SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT",
    "FINITE_PRIMORDIAL_SOURCE_RECEIPT",
    "COMMON_FREEZEOUT_SURFACE_RECEIPT",
    "DARK_CONTINUATION_OFF_OR_PARENT_RECEIPT",
    "BOLTZMANN_TRANSFER_RECEIPT",
    "FROZEN_LIKELIHOOD_RECEIPT",
    "FROZEN_PARENT_HASHES_RECEIPT",
    "LIKELIHOOD_EVALUATED_OUTPUT_RECEIPT",
    "PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT",
    "PHYSICAL_CMB_PROMOTION_READY_RECEIPT",
)

FORBIDDEN_SOURCE_INPUTS = (
    "Planck or ACT maps, spectra, overlays, residuals, likelihoods, or posteriors before source freeze",
    "BAO, supernova, weak-lensing, RSD, SPARC, cluster, or H0/S8 target values before source freeze",
    "CMB peak alignment used to choose scale, basis, amplitude, cutoff, branch, or hyperparameters",
    "LambdaCDM best-fit tables silently reused as OPH source parameters",
    "trained emulators, cached posteriors, or human branch choices exposed to target spectra before freeze",
)

SEMANTIC_TYPE_BOUNDARIES = (
    "visualizer payload -> physical CMB prediction",
    "screen multipole ell_src -> observed CMB multipole L_CMB",
    "cap-opening, repair-gap, or cycle-count proxy -> Mpc^-1",
    "repair transition diagnostic gamma_repair_step -> recombination rate Gamma_rec",
    "screen scalar covariance -> primordial curvature source without radial-lift receipts",
    "imported FLRW geometry -> OPH-native geometry without CosmoGeomRead_r",
)


def cmb_promotion_ledger_report(run_dirs: list[Path]) -> dict[str, Any]:
    roots = [Path(path) for path in run_dirs]
    scalar_quotient = _first_json(roots, "scalar_quotient_report.json")
    quotient_ensemble = _first_json(roots, "quotient_ensemble.json")
    screen_power = _first_json(roots, "oph_screen_power_report.json")
    maxent_green = _first_json(roots, "maxent_green_spectrum_report.json")
    finite_certificate = _first_json(roots, "finite_certificate_report.json")
    source_provenance = _first_json(roots, "cmb_source_provenance_report.json")
    no_data = _first_json(roots, "no_data_use_receipt.json")
    physical_input = _first_json(roots, "physical_cmb_input_report.json")
    physical_validation = _first_json(roots, "physical_cmb_input_validation.json")
    promotion = _first_json(roots, "physical_cmb_promotion_audit_report.json")
    frontier = _first_json(roots, "physical_cmb_frontier_report.json")
    output = _first_json(roots, "physical_cmb_output_comparison_report.json")
    finite_collar_boltzmann = _first_json(roots, "finite_collar_boltzmann_bundle_report.json")
    frozen_transfer = _first_json(roots, "frozen_transfer_likelihood_report.json")
    cmb_derivation = _first_json(roots, "cmb_derivation_report.json")
    cmb_static_plots = _first_json(roots, "cmb_static_plots_summary.json")
    cmb_frontier_viewer = _first_json(roots, "cmb_neutral_frontier_viewer_summary.json")
    cmb_lite = _first_json(roots, "cmb_lite_comparison_report.json")
    exact_cmb = _first_json(roots, "oph_exact_cmb_camb_report.json")
    finite_clock_cmb = _first_json(roots, "finite_repair_clock_cmb_camb_report.json")
    scale_payload = (
        _first_json(roots, "physical_scale_bridge_report.json")
        or _first_json(roots, "cosmological_scale_bridge_report.json")
    )
    scale = validate_physical_scale_bridge_receipts(scale_payload)

    validation_payload = _validation_payload(physical_input, physical_validation)
    input_contract_receipt = _validated_input_contract_receipt(physical_input, validation_payload)
    promotion_ready = _validated_promotion_ready(promotion, input_contract_receipt=input_contract_receipt)
    source_gates = source_provenance or (physical_input.get("source_provenance") or {})
    finite_source_receipt = _source_only_finite_artifact_receipt(source_gates, no_data, finite_certificate)
    radial_lift_receipt = bool(
        validation_payload.get("screen_to_primordial_lift_receipt", False)
        or (physical_input.get("input_status") or {}).get("screen_to_primordial_lift", {}).get("passed", False)
        or finite_certificate.get("SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT", False)
        or finite_certificate.get("SCREEN_TO_RADIAL_LIFT_RECEIPT", False)
        or (finite_certificate.get("derived_outputs") or {}).get("screen_to_primordial_lift_receipt", False)
    )
    conditional_scale_ready = bool(
        scale.get("PHYSICAL_SCALE_BRIDGE_RECEIPT", False)
        and _norm(scale.get("claim_tier")) == "conditional_physical"
        and scale.get("geometry_origin") == "IMPORTED_FLRW"
    )
    native_scale_ready = bool(
        scale.get("PHYSICAL_SCALE_BRIDGE_RECEIPT", False)
        and _norm(scale.get("claim_tier")) == "oph_native_physical"
        and scale.get("geometry_origin") == "OPH_NATIVE"
        and scale.get("oph_native_k_derivation", False)
    )
    dark_mode = _dark_continuation_mode(
        finite_collar_boltzmann,
        frozen_transfer,
        physical_input,
        frontier,
    )
    dark_parent_or_off = bool(
        str(dark_mode).upper() == "OFF"
        or _truthy_any(finite_collar_boltzmann, "FINITE_COVARIANT_PARENT_RECEIPT", "finite_covariant_parent_receipt")
        or _truthy_any(physical_input, "finite_covariant_parent_receipt")
    )
    boltzmann_transfer_receipt = bool(
        _truthy_any(finite_collar_boltzmann, "BOLTZMANN_TRANSFER_RECEIPT", "boltzmann_transfer_receipt")
        or _truthy_any(frozen_transfer, "BOLTZMANN_TRANSFER_RECEIPT", "boltzmann_transfer_receipt")
        or _truthy_any(frozen_transfer, "FROZEN_TRANSFER_LIKELIHOOD_CLOSURE_RECEIPT")
        or _truthy_any(exact_cmb, "CMB_TRANSFER_RECEIPT", "transfer_receipt", "measurement_comparable_curve")
        or _truthy_any(finite_clock_cmb, "CMB_TRANSFER_RECEIPT", "transfer_receipt", "measurement_comparable_cmb_curve")
    )
    frozen_likelihood_receipt = _validated_frozen_likelihood_receipt(frozen_transfer)
    frozen_parent_hashes_receipt = _frozen_parent_hashes_receipt(frozen_transfer, physical_input)
    likelihood_evaluated_output_receipt = _likelihood_evaluated_output_receipt(frozen_transfer)
    visual_diagnostic = bool(cmb_static_plots or cmb_frontier_viewer or cmb_lite or output or frontier)
    spectrum_diagnostic = bool(
        output.get("PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT", False)
        or frontier.get("physical_cmb_output_comparison_receipt", False)
        or exact_cmb.get("measurement_comparable_curve", False)
        or exact_cmb.get("measurement_comparable_cmb_curve", False)
        or finite_clock_cmb.get("measurement_comparable_cmb_curve", False)
        or cmb_lite.get("comparison_receipt", False)
    )
    terminal_prediction_asserted = bool(
        output.get("PHYSICAL_CMB_PREDICTION_RECEIPT", False)
        or output.get("physical_cmb_prediction", False)
    )
    source_release_amplitude = bool(
        (finite_certificate.get("theorem_grade_finite_inputs", False) and (finite_certificate.get("A_q") is not None))
        or (finite_certificate.get("derived_outputs") or {}).get("A_zeta") is not None
    )
    screen_covariance = bool(
        _truthy_any(screen_power, "SCREEN_COVARIANCE_RECEIPT", "simulator_primordial_ready")
        or _truthy_any(maxent_green, "MAXENT_GREEN_SOURCE_RECEIPT", "maxent_green_source_receipt")
        or _truthy_any(scalar_quotient, "SCREEN_COVARIANCE_RECEIPT")
    )
    primordial_source = bool(finite_source_receipt and radial_lift_receipt and screen_covariance)
    conditional_physical_source = bool(
        finite_source_receipt
        and primordial_source
        and conditional_scale_ready
        and scale.get("PHYSICAL_MODE_FREEZEOUT_MAP_RECEIPT", False)
    )
    native_physical_source = bool(
        finite_source_receipt
        and primordial_source
        and native_scale_ready
        and scale.get("PHYSICAL_MODE_FREEZEOUT_MAP_RECEIPT", False)
    )

    readiness_gates: dict[str, bool] = {
        "VISUAL_DIAGNOSTIC_RECEIPT": visual_diagnostic,
        "SPECTRUM_DIAGNOSTIC_RECEIPT": spectrum_diagnostic,
        "FINITE_QUOTIENT_SOURCE_LAW_RECEIPT": _finite_quotient_source_law_receipt(quotient_ensemble),
        "SOURCE_ONLY_FINITE_ARTIFACT_RECEIPT": finite_source_receipt,
        "NO_DATA_USE_RECEIPT": _no_data_use_receipt(no_data, source_gates),
        "POOLED_SOURCE_REDUCER_RECEIPT": bool(source_gates.get("pooled_source_reducer_receipt", False)),
        "GEOMETRIC_SCREEN_SCALAR_RECEIPT": _geometric_screen_scalar_receipt(scalar_quotient),
        "PHYSICAL_PRECISION_OPERATOR_RECEIPT": bool(scale.get("PHYSICAL_SPATIAL_K_RECEIPT", False)),
        "SOURCE_RELEASE_AMPLITUDE_RECEIPT": source_release_amplitude,
        "SCREEN_COVARIANCE_RECEIPT": screen_covariance,
        "SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT": radial_lift_receipt,
        "FINITE_PRIMORDIAL_SOURCE_RECEIPT": primordial_source,
        "CONDITIONAL_IMPORTED_FLRW_GEOMETRY_RECEIPT": conditional_scale_ready,
        "OPH_NATIVE_COSMO_GEOM_READ_RECEIPT": native_scale_ready,
        "COMMON_FREEZEOUT_SURFACE_RECEIPT": bool(scale.get("PHYSICAL_FREEZEOUT_SURFACE_RECEIPT", False)),
        "DARK_CONTINUATION_OFF_OR_PARENT_RECEIPT": dark_parent_or_off,
        "BOLTZMANN_TRANSFER_RECEIPT": boltzmann_transfer_receipt,
        "FROZEN_LIKELIHOOD_RECEIPT": frozen_likelihood_receipt,
        "FROZEN_PARENT_HASHES_RECEIPT": frozen_parent_hashes_receipt,
        "LIKELIHOOD_EVALUATED_OUTPUT_RECEIPT": likelihood_evaluated_output_receipt,
        "PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT": input_contract_receipt,
        "PHYSICAL_CMB_PROMOTION_READY_RECEIPT": promotion_ready,
    }
    prediction_receipt = _physical_prediction_receipt(
        readiness_gates,
        conditional_physical_source=conditional_physical_source,
        native_physical_source=native_physical_source,
    )
    readiness_gates["PHYSICAL_CMB_PREDICTION_RECEIPT"] = prediction_receipt
    current_claim_tier = _current_claim_tier(
        prediction_receipt=prediction_receipt,
        native_physical_source=native_physical_source,
        conditional_physical_source=conditional_physical_source,
        finite_source_receipt=finite_source_receipt,
        spectrum_diagnostic=spectrum_diagnostic,
        visual_diagnostic=visual_diagnostic,
    )
    blockers = _blockers(readiness_gates, scale, validation_payload, promotion)
    if terminal_prediction_asserted and not prediction_receipt:
        blockers.append("untrusted_terminal_prediction_assertion_rejected")
        blockers = sorted(set(blockers))
    fail_closed_state = _fail_closed_state(current_claim_tier, prediction_receipt, blockers)
    report = {
        "mode": "cmb_promotion_ledger_v1",
        "schema": "oph_cmb_promotion_ledger_v1",
        "run_dirs": [str(path) for path in roots],
        "claim_ladder": list(CLAIM_LADDER),
        "current_claim_tier": current_claim_tier,
        "fail_closed_state": fail_closed_state,
        "conditional_physical_scale_bridge_ready": conditional_scale_ready,
        "conditional_physical_cmb_source_ready": conditional_physical_source,
        "oph_native_physical_cmb_source_ready": native_physical_source,
        "oph_native_geometry_ready": native_scale_ready,
        "likelihood_evaluated_physical_cmb_prediction": prediction_receipt,
        "hard_physical_cmb_input_contract_receipt": input_contract_receipt,
        "physical_cmb_promotion_ready": promotion_ready,
        "terminal_prediction_asserted_by_output": terminal_prediction_asserted,
        "prediction_parent_gates": list(PHYSICAL_PREDICTION_PARENT_GATES),
        "claim_tier": scale.get("claim_tier"),
        "geometry_origin": scale.get("geometry_origin"),
        "dark_continuation_mode": dark_mode,
        "readiness_gates": readiness_gates,
        "blockers": blockers,
        "first_blocked_gate": next((name for name, passed in readiness_gates.items() if not passed), None),
        "scale_bridge_validation": scale,
        "contract_blockers": sorted(set(validation_payload.get("blockers") or [])),
        "promotion_blockers": sorted(set(promotion.get("promotion_blockers") or [])),
        "source_summary": {
            "source_provenance_present": bool(source_gates),
            "quotient_ensemble_present": bool(quotient_ensemble),
            "scalar_quotient_present": bool(scalar_quotient),
            "screen_power_present": bool(screen_power),
            "maxent_green_present": bool(maxent_green),
            "cmb_derivation_present": bool(cmb_derivation),
        },
        "forbidden_source_inputs": list(FORBIDDEN_SOURCE_INPUTS),
        "semantic_type_boundaries": list(SEMANTIC_TYPE_BOUNDARIES),
        "claim_boundary": (
            "A visually CMB-like OPH simulation is not a physical CMB prediction. "
            "Imported frozen FLRW geometry can support the CONDITIONAL_PHYSICAL path; "
            "OPH_NATIVE_PHYSICAL additionally requires quotient-derived CosmoGeomRead_r. "
            "Only a frozen source, transfer, and likelihood bundle may emit PHYSICAL_CMB_PREDICTION_RECEIPT."
        ),
    }
    return report


def write_cmb_promotion_ledger_report(run_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    report = cmb_promotion_ledger_report(run_dirs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cmb_promotion_ledger_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "cmb_promotion_ledger_report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _first_json(roots: list[Path], name: str) -> dict[str, Any]:
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / name,
                root / "source" / name,
                root / "physical_cmb" / name,
                root / "cosmology" / name,
                root / "measurement_pack" / name,
            ]
        )
    for path in candidates:
        if path.exists() and path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict) and data:
                return data
    return {}


def _truthy_any(data: dict[str, Any], *keys: str) -> bool:
    return any(bool(data.get(key, False)) for key in keys)


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _validation_payload(physical_input: dict[str, Any], physical_validation: dict[str, Any]) -> dict[str, Any]:
    validation = physical_validation or physical_input.get("validation") or {}
    return validation if isinstance(validation, dict) else {}


def _validated_input_contract_receipt(
    physical_input: dict[str, Any],
    validation: dict[str, Any],
) -> bool:
    return bool(
        physical_input.get("PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT", False)
        and validation.get("PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT", False)
        and not (physical_input.get("blockers") or [])
        and not (validation.get("blockers") or [])
        and isinstance(physical_input.get("contract"), dict)
    )


def _validated_promotion_ready(
    promotion: dict[str, Any],
    *,
    input_contract_receipt: bool,
) -> bool:
    return bool(
        input_contract_receipt
        and promotion.get("physical_cmb_promotion_ready", False)
        and promotion.get("physical_cmb_input_contract_receipt", False)
        and promotion.get("no_data_use_receipt", False)
        and promotion.get("cdm_limit_regression_passed", False)
        and promotion.get("official_likelihood_ready", False)
        and not (promotion.get("contract_blockers") or [])
        and not (promotion.get("promotion_blockers") or [])
    )


def _validated_frozen_likelihood_receipt(frozen_transfer: dict[str, Any]) -> bool:
    return bool(
        frozen_transfer.get("FROZEN_TRANSFER_LIKELIHOOD_CLOSURE_RECEIPT", False)
        and frozen_transfer.get("FROZEN_SOURCE_MANIFEST_RECEIPT", False)
        and frozen_transfer.get("SOLVER_ASSUMPTION_PIN_RECEIPT", False)
        and frozen_transfer.get("FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT", False)
        and frozen_transfer.get("FROZEN_PHYSICAL_SPECTRUM_RECEIPT", False)
        and frozen_transfer.get("FULL_OBSERVABLE_LIKELIHOOD_RECEIPT", False)
        and frozen_transfer.get("official_likelihood_execution_ready", False)
        and not (frozen_transfer.get("blockers") or [])
    )


def _likelihood_evaluated_output_receipt(frozen_transfer: dict[str, Any]) -> bool:
    return bool(
        _validated_frozen_likelihood_receipt(frozen_transfer)
        and frozen_transfer.get("LIKELIHOOD_EVALUATED_PHYSICAL_PREDICTION_RECEIPT", False)
    )


def _frozen_parent_hashes_receipt(
    frozen_transfer: dict[str, Any],
    physical_input: dict[str, Any],
) -> bool:
    input_contract = physical_input.get("contract")
    if not isinstance(input_contract, dict):
        return False
    frozen_hashes = _frozen_hash_triplet(frozen_transfer)
    contract_hashes = _frozen_hash_triplet(input_contract)
    return bool(
        all(_valid_sha256_hash(value) for value in frozen_hashes)
        and all(_valid_sha256_hash(value) for value in contract_hashes)
        and frozen_hashes == contract_hashes
    )


def _frozen_hash_triplet(payload: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(payload.get("frozen_source_hash") or payload.get("source_hash") or ""),
        str(payload.get("frozen_solver_hash") or payload.get("solver_hash") or ""),
        str(payload.get("frozen_likelihood_hash") or payload.get("likelihood_hash") or ""),
    )


def _valid_sha256_hash(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(value or "")))


def _physical_prediction_receipt(
    readiness_gates: dict[str, bool],
    *,
    conditional_physical_source: bool,
    native_physical_source: bool,
) -> bool:
    physical_source_route = bool(conditional_physical_source or native_physical_source)
    return bool(
        physical_source_route
        and all(readiness_gates.get(gate, False) for gate in PHYSICAL_PREDICTION_PARENT_GATES)
    )


def _no_data_use_receipt(no_data: dict[str, Any], source_gates: dict[str, Any]) -> bool:
    return bool(
        no_data.get("NO_DATA_USE_RECEIPT", False)
        or no_data.get("no_data_use_receipt", False)
        or source_gates.get("no_data_use_receipt", False)
        or source_gates.get("NO_DATA_USE_RECEIPT", False)
    )


def _source_only_finite_artifact_receipt(
    source_gates: dict[str, Any],
    no_data: dict[str, Any],
    finite_certificate: dict[str, Any],
) -> bool:
    return bool(
        _no_data_use_receipt(no_data, source_gates)
        and source_gates.get("CMB_SOURCE_PROVENANCE_RECEIPT", False)
        and source_gates.get("pooled_source_reducer_receipt", False)
        and source_gates.get("contradiction_free_provenance_receipt", False)
        and (
            finite_certificate.get("theorem_grade_finite_inputs", False)
            or source_gates.get("finite_source_artifact_receipt", False)
        )
    )


def _finite_quotient_source_law_receipt(quotient_ensemble: dict[str, Any]) -> bool:
    if not quotient_ensemble:
        return False
    return bool(
        quotient_ensemble.get("claim_tier") == "E2_OPH_NATIVE_QUOTIENT_ENSEMBLE"
        and quotient_ensemble.get("base_measure")
        and quotient_ensemble.get("normal_form_hash")
        and quotient_ensemble.get("Q_schema_hash")
        and quotient_ensemble.get("Gamma_schema_hash")
        and (
            quotient_ensemble.get("pathwise_partition_invariance_receipt", False)
            or quotient_ensemble.get("PATHWISE_PARTITION_INVARIANCE_RECEIPT", False)
        )
    )


def _geometric_screen_scalar_receipt(scalar_quotient: dict[str, Any]) -> bool:
    return bool(
        scalar_quotient.get("SCREEN_SCALAR_QUOTIENT_RECEIPT", False)
        or scalar_quotient.get("SCALAR_QUOTIENT_RECEIPT", False)
        or (
            scalar_quotient.get("geometric_volume_jacobian_readout", False)
            and scalar_quotient.get("screen_mass_matrix_readout", False)
            and scalar_quotient.get("monopole_dipole_projector_receipt", False)
        )
    )


def _dark_continuation_mode(*reports: dict[str, Any]) -> str:
    for report in reports:
        value = report.get("dark_continuation") or report.get("dark_continuation_mode")
        if value is not None:
            return str(value)
    return "UNDECLARED"


def _current_claim_tier(
    *,
    prediction_receipt: bool,
    native_physical_source: bool,
    conditional_physical_source: bool,
    finite_source_receipt: bool,
    spectrum_diagnostic: bool,
    visual_diagnostic: bool,
) -> str:
    if prediction_receipt:
        return "LIKELIHOOD_EVALUATED_PHYSICAL_CMB_PREDICTION"
    if native_physical_source:
        return "OPH_NATIVE_PHYSICAL_CMB_SOURCE"
    if conditional_physical_source:
        return "CONDITIONAL_PHYSICAL_CMB_SOURCE"
    if finite_source_receipt:
        return "SOURCE_ONLY_FINITE_ARTIFACT"
    if spectrum_diagnostic:
        return "SPECTRUM_DIAGNOSTIC"
    if visual_diagnostic:
        return "VISUAL_DIAGNOSTIC"
    return "UNSTARTED_OR_INVALIDATED"


def _blockers(
    readiness_gates: dict[str, bool],
    scale: dict[str, Any],
    validation: dict[str, Any],
    promotion: dict[str, Any],
) -> list[str]:
    blockers = [name.lower() + "_missing" for name, passed in readiness_gates.items() if not passed]
    blockers.extend(str(item) for item in scale.get("blockers") or [])
    blockers.extend(str(item) for item in validation.get("blockers") or [])
    blockers.extend(str(item) for item in promotion.get("promotion_blockers") or [])
    return sorted(set(blockers))


def _fail_closed_state(current_claim_tier: str, prediction_receipt: bool, blockers: list[str]) -> str:
    if prediction_receipt:
        return "PASSED_DECLARED_TEST"
    if current_claim_tier == "UNSTARTED_OR_INVALIDATED":
        return "INVALIDATED"
    if blockers:
        return "NUMERICALLY_INCONCLUSIVE"
    return "FALSIFIED_ON_DECLARED_TEST"


def _markdown_report(report: dict[str, Any]) -> str:
    gates = "\n".join(
        f"- `{name}`: `{str(passed).lower()}`"
        for name, passed in sorted((report.get("readiness_gates") or {}).items())
    )
    blockers = "\n".join(f"- `{item}`" for item in report.get("blockers") or [])
    if not blockers:
        blockers = "- none"
    boundaries = "\n".join(f"- {item}" for item in report.get("semantic_type_boundaries") or [])
    return (
        "# CMB Promotion Ledger\n\n"
        f"{report.get('claim_boundary')}\n\n"
        f"- current claim tier: `{report.get('current_claim_tier')}`\n"
        f"- fail-closed state: `{report.get('fail_closed_state')}`\n"
        f"- scale claim tier: `{report.get('claim_tier')}`\n"
        f"- geometry origin: `{report.get('geometry_origin')}`\n"
        f"- conditional physical scale bridge: `{str(report.get('conditional_physical_scale_bridge_ready')).lower()}`\n"
        f"- OPH-native geometry: `{str(report.get('oph_native_geometry_ready')).lower()}`\n"
        f"- likelihood-evaluated prediction: `{str(report.get('likelihood_evaluated_physical_cmb_prediction')).lower()}`\n\n"
        "## Gates\n\n"
        f"{gates}\n\n"
        "## Blockers\n\n"
        f"{blockers}\n\n"
        "## Semantic Boundaries\n\n"
        f"{boundaries}\n"
    )
