from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oph_fpe.claims import CONTINUATION, with_claim_metadata


FINITE_HORIZON_RECORD_GATES = (
    "finite_horizon_collar_carrier",
    "same_boundary_transcript",
    "record_archive_readout",
    "checkpoint_reconstruction",
    "graph_local_scrambling_cone",
)

PHYSICAL_EVAPORATION_GATES = (
    "physical_microstate_embedding",
    "exterior_retarded_time",
    "radiation_algebra_readout",
    "fine_grained_radiation_entropy",
    "exterior_energy_flux_closure",
    "no_hidden_register_leakage",
    "no_future_data_leakage",
    "no_remnant_certificate",
    "refinement_stability",
)

PAGE_CURVE_GATES = (
    "physical_page_time_rule",
    "semiclassical_validity_interval",
    "generalized_entropy_or_area_readout",
)

QNM_RADIATIVE_GATES = (
    "physical_background_bridge",
    "radiative_quotient",
    "finite_operator_export",
    "continuum_resolvent_convergence",
    "future_horizon_boundary_rows",
    "future_null_infinity_boundary_rows",
    "bondi_or_asymptotic_readout",
    "frozen_qnm_window",
    "detector_space_comparison_protocol",
)

FORBIDDEN_PROMOTION_KEYS = (
    "uses_target_qnm_frequency",
    "uses_target_waveform",
    "uses_page_curve_target",
    "uses_measured_evaporation_endpoint",
    "uses_hidden_complement_registers",
    "uses_future_radiation_registers",
)


@dataclass(frozen=True)
class BlackHoleBridgeInputs:
    source_artifact: Path | None = None
    source: str = "black_hole_bridge_status_contract"


def black_hole_bridge_status_report(
    inputs: BlackHoleBridgeInputs | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    bridge_inputs = inputs if inputs is not None else BlackHoleBridgeInputs(**kwargs)
    artifact, artifact_meta, load_error = _load_artifact(bridge_inputs.source_artifact)

    gate_sources = artifact.get("readiness_gates") if isinstance(artifact.get("readiness_gates"), dict) else artifact
    # The legacy v0 artifact contains only caller-authored booleans.  Keep
    # those assertions visible for migration diagnostics, but do not mistake
    # them for independently verified receipts.  No producer/verifier for the
    # black-hole gates exists in this repository yet, so every physical gate
    # must fail closed.
    declared_finite_gates = _gate_status(gate_sources, FINITE_HORIZON_RECORD_GATES)
    declared_evaporation_gates = _gate_status(gate_sources, PHYSICAL_EVAPORATION_GATES)
    declared_page_gates = _gate_status(gate_sources, PAGE_CURVE_GATES)
    declared_qnm_gates = _gate_status(gate_sources, QNM_RADIATIVE_GATES)
    finite_gates = {name: False for name in FINITE_HORIZON_RECORD_GATES}
    evaporation_gates = {name: False for name in PHYSICAL_EVAPORATION_GATES}
    page_gates = {name: False for name in PAGE_CURVE_GATES}
    qnm_gates = {name: False for name in QNM_RADIATIVE_GATES}
    forbidden = [key for key in FORBIDDEN_PROMOTION_KEYS if _candidate_bool(artifact, key)]

    declared_positive_assertions = sorted(
        name
        for gates in (
            declared_finite_gates,
            declared_evaporation_gates,
            declared_page_gates,
            declared_qnm_gates,
        )
        for name, asserted in gates.items()
        if asserted
    )

    finite_receipt = all(finite_gates.values())
    evaporation_receipt = finite_receipt and all(evaporation_gates.values()) and not forbidden
    page_receipt = evaporation_receipt and all(page_gates.values())
    qnm_receipt = finite_receipt and all(qnm_gates.values()) and not forbidden
    physical_simulation_receipt = evaporation_receipt and qnm_receipt

    blockers = []
    if load_error is not None:
        blockers.append(load_error)
    if not finite_receipt:
        blockers.extend(f"finite_gate_missing:{name}" for name, passed in finite_gates.items() if not passed)
    if not evaporation_receipt:
        blockers.extend(
            f"evaporation_gate_missing:{name}"
            for name, passed in evaporation_gates.items()
            if not passed
        )
    if not page_receipt:
        blockers.extend(f"page_gate_missing:{name}" for name, passed in page_gates.items() if not passed)
    if not qnm_receipt:
        blockers.extend(f"qnm_gate_missing:{name}" for name, passed in qnm_gates.items() if not passed)
    blockers.extend(f"target_leakage_or_forbidden_dependency:{key}" for key in forbidden)
    if declared_positive_assertions:
        blockers.append("caller_gate_assertions_are_not_independent_receipts")
    blockers.append("independent_black_hole_gate_verifier_unavailable")

    report = {
        "mode": "black_hole_bridge_status_contract_v1_fail_closed",
        "source": bridge_inputs.source,
        "source_artifact": artifact_meta,
        "BLACK_HOLE_BRIDGE_STATUS_CONTRACT_RECEIPT": True,
        "FINITE_HORIZON_RECORD_REPAIR_DIAGNOSTIC_RECEIPT": finite_receipt,
        "BLACK_HOLE_PHYSICAL_EVAPORATION_BRIDGE_RECEIPT": evaporation_receipt,
        "BLACK_HOLE_PHYSICAL_PAGE_CURVE_RECEIPT": page_receipt,
        "BLACK_HOLE_QNM_RADIATIVE_BRIDGE_RECEIPT": qnm_receipt,
        "BLACK_HOLE_PHYSICAL_SIMULATION_RECEIPT": physical_simulation_receipt,
        "physical_claims": {
            "physical_page_time_claim": page_receipt,
            "semiclassical_evaporation_claim": evaporation_receipt,
            "full_black_hole_unitarity_claim": evaporation_receipt
            and _candidate_bool(artifact, "full_microstate_coverage"),
            "black_hole_information_problem_solved": evaporation_receipt
            and _candidate_bool(artifact, "full_microstate_coverage")
            and _candidate_bool(artifact, "standalone_information_problem_solution_gate"),
            "qnm_or_ringdown_prediction": qnm_receipt,
        },
        "readiness_gates": {
            "finite_horizon_record": finite_gates,
            "physical_evaporation": evaporation_gates,
            "page_curve": page_gates,
            "qnm_radiative": qnm_gates,
            "forbidden_dependencies_absent": not forbidden,
        },
        "declared_readiness_gate_assertions": {
            "finite_horizon_record": declared_finite_gates,
            "physical_evaporation": declared_evaporation_gates,
            "page_curve": declared_page_gates,
            "qnm_radiative": declared_qnm_gates,
        },
        "caller_positive_gate_assertions_ignored": declared_positive_assertions,
        "independent_gate_verifier_available": False,
        "blockers": sorted(set(blockers)),
        "claim_boundary": (
            "Legacy artifact booleans are declaration-only and cannot open a receipt. "
            "Finite horizon/collar, evaporation, Page-curve, and radiative claims remain "
            "closed until independently recomputable gate producers and verifiers exist."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=CONTINUATION,
        receipt="BLACK_HOLE_BRIDGE_STATUS_CONTRACT_RECEIPT",
        observable_id="black_hole_bridge_status",
        fit_objective="fail_closed_black_hole_physical_promotion",
    )


def write_black_hole_bridge_status_report(
    out: Path,
    inputs: BlackHoleBridgeInputs | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = black_hole_bridge_status_report(inputs, **kwargs)
    (out / "black_hole_bridge_status_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "black_hole_bridge_status_report.md").write_text(
        _markdown_report(report),
        encoding="utf-8",
    )
    return report


def _gate_status(source: dict[str, Any], names: tuple[str, ...]) -> dict[str, bool]:
    return {name: _candidate_bool(source, name) for name in names}


def _candidate_bool(source: Any, key: str) -> bool:
    if not isinstance(source, dict):
        return False
    value = source.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    for nested in source.values():
        if isinstance(nested, dict) and _candidate_bool(nested, key):
            return True
    return False


def _load_artifact(path: Path | None) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    if path is None:
        return {}, {"present": False}, "source_artifact_missing"
    resolved = Path(path)
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, {"present": False, "path": str(resolved)}, "source_artifact_missing"
    except json.JSONDecodeError:
        return {}, {"present": True, "path": str(resolved)}, "source_artifact_invalid_json"
    if not isinstance(data, dict):
        return {}, {"present": True, "path": str(resolved)}, "source_artifact_not_object"
    return data, {"present": True, "path": str(resolved)}, None


def _markdown_report(report: dict[str, Any]) -> str:
    claims = report["physical_claims"]
    return (
        "# Black-Hole Bridge Status\n\n"
        f"{report['claim_boundary']}\n\n"
        "## Receipts\n\n"
        f"- finite horizon record diagnostic: {report['FINITE_HORIZON_RECORD_REPAIR_DIAGNOSTIC_RECEIPT']}\n"
        f"- physical evaporation bridge: {report['BLACK_HOLE_PHYSICAL_EVAPORATION_BRIDGE_RECEIPT']}\n"
        f"- physical Page curve: {report['BLACK_HOLE_PHYSICAL_PAGE_CURVE_RECEIPT']}\n"
        f"- QNM radiative bridge: {report['BLACK_HOLE_QNM_RADIATIVE_BRIDGE_RECEIPT']}\n"
        f"- physical black-hole simulation: {report['BLACK_HOLE_PHYSICAL_SIMULATION_RECEIPT']}\n\n"
        "## Physical Claims\n\n"
        f"- physical Page time: {claims['physical_page_time_claim']}\n"
        f"- semiclassical evaporation: {claims['semiclassical_evaporation_claim']}\n"
        f"- full black-hole unitarity: {claims['full_black_hole_unitarity_claim']}\n"
        f"- information problem solved: {claims['black_hole_information_problem_solved']}\n"
        f"- QNM or ringdown prediction: {claims['qnm_or_ringdown_prediction']}\n\n"
        "## Blockers\n\n"
        + "\n".join(f"- {blocker}" for blocker in report["blockers"])
        + "\n"
    )
