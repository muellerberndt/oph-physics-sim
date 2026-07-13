"""Typed, non-promoting views of imported OPH particle artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oph_fpe.evidence.cross_repo_artifacts import verify_cross_repo_artifact_manifest


DEFAULT_PARTICLE_FRONTIER_ROOT = Path(__file__).resolve().parents[2] / "data" / "oph_cross_repo_current"


def particle_frontier_report(import_root: Path | None = None) -> dict[str, Any]:
    root = Path(import_root) if import_root is not None else DEFAULT_PARTICLE_FRONTIER_ROOT
    verification = verify_cross_repo_artifact_manifest(root)
    rows = {
        str(row.get("key")): row
        for row in verification.get("manifest", {}).get("artifacts", [])
        if isinstance(row, dict) and row.get("hash_verified") is True
    }
    artifacts = {key: _payload(root, row) for key, row in rows.items()}

    neutrino = _neutrino_status(artifacts)
    ew = _conditional_ew_status(artifacts)
    hadron = _empirical_hadron_status(artifacts)
    pixel = _pixel_profiles(artifacts)
    obstructions = _source_obstructions(artifacts)
    lattice = _lattice_status(artifacts)
    blockers = [] if verification["verified"] else list(verification["blockers"])
    if not neutrino["rejection_receipt"]:
        blockers.append("neutrino_rejection_artifacts_missing_or_invalid")
    if not hadron["artifact_present"]:
        blockers.append("empirical_hadron_artifact_missing")

    return {
        "schema": "oph_particle_frontier_report_v1",
        "artifact_manifest_verified": bool(verification["verified"]),
        "paper_release_id": verification.get("manifest", {}).get("paper_release_id"),
        "source_repository": verification.get("manifest", {}).get("source_repository", {}),
        "neutrino": neutrino,
        "conditional_electroweak": ew,
        "empirical_hadron_closure": hadron,
        "pixel_parameter_profiles": pixel,
        "source_obstructions": obstructions,
        "lattice_diagnostic": lattice,
        "physical_particle_prediction_receipt": False,
        "source_hadron_endpoint_receipt": False,
        "source_electroweak_mass_prediction_receipt": False,
        "simulation_receipts_promoted_by_import": False,
        "blockers": blockers,
        "claim_boundary": (
            "This report exposes rejected, conditional, empirical, and no-go particle results to the simulator. "
            "It is a comparison/frontier surface, not a microscopic particle law or a production-particle receipt."
        ),
    }


def write_particle_frontier_report(import_root: Path | None = None, out: Path | None = None) -> dict[str, Any]:
    report = particle_frontier_report(import_root)
    root = Path(import_root) if import_root is not None else DEFAULT_PARTICLE_FRONTIER_ROOT
    target = Path(out) if out is not None else root / "particle_frontier_report.json"
    if target.is_dir():
        target = target / "particle_frontier_report.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _neutrino_status(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    closure = artifacts.get("neutrino_lane_closure", {})
    score = artifacts.get("neutrino_nufit61_score", {})
    decision = _mapping(score.get("decision"))
    rejected = bool(
        closure.get("bridge_prediction_promotion_allowed") is False
        and closure.get("public_promotion_allowed") is False
        and decision.get("current_weighted_cycle_candidate_rejected_by_declared_gate") is True
    )
    threshold = decision.get("threshold_delta_chi2_2d_3sigma")
    score_rows = _mapping(score.get("scores"))
    lower_bounds = {
        key: _mapping(value).get("joint_fixed_candidate_delta_chi2_lower_bound")
        for key, value in score_rows.items()
        if isinstance(value, dict)
    }
    return {
        "oph_mass_prediction": None,
        "public_promotion_allowed": False,
        "rejection_receipt": rejected,
        "historical_benchmark_mode": "historical_rejected_weighted_cycle_benchmark",
        "historical_benchmark_opt_in_only": True,
        "declared_3sigma_threshold": threshold,
        "profile_delta_chi2_lower_bounds": lower_bounds,
        "reason": (
            "The target-informed weighted-cycle continuation is rejected; this does not falsify finite OPH."
            if rejected
            else "Pinned rejection evidence is incomplete."
        ),
    }


def _conditional_ew_status(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload = artifacts.get("conditional_ew_envelope", {})
    guards = _mapping(payload.get("guards"))
    selection = artifacts.get("d10_repair_selection", {})
    display = bool(
        payload.get("artifact") == "oph_conditional_ew_predictions"
        and guards.get("public_promotion_allowed") is False
        and guards.get("conditional_display_allowed") is True
    )
    return {
        "artifact_present": bool(payload),
        "row_class": payload.get("row_class"),
        "conditional_envelope_GeV": payload.get("conditional_envelope") if display else None,
        "display_allowed": display,
        "simulation_input_allowed": False,
        "public_prediction_allowed": False,
        "selection_status": selection.get("status"),
        "selection_promotion_allowed": selection.get("promotion_allowed") is True,
        "promotion_blockers": list(guards.get("promotion_blockers") or []),
    }


def _empirical_hadron_status(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload = artifacts.get("empirical_hadron_spectral_measure", {})
    guards = _mapping(payload.get("guards"))
    projection = _mapping(payload.get("projection"))
    measure = _mapping(payload.get("rho_had_or_measure"))
    consistency = _mapping(payload.get("consistency"))
    moments = _mapping(payload.get("transport_moments"))
    valid = bool(
        payload.get("artifact") == "oph_empirical_ward_projected_hadronic_spectral_measure"
        and projection.get("ward_projected") is True
        and measure.get("positivity_status") == "verified_nonnegative_on_exported_grids_and_atoms"
        and consistency.get("within_tolerance") is True
        and guards.get("empirical_hadron_closure") is True
        and guards.get("external_cross_section_data_used") is True
        and guards.get("promotable_as_oph_source_theorem") is False
        and isinstance(payload.get("systematics"), dict)
    )
    return {
        "artifact_present": bool(payload),
        "empirical_closure_validation_receipt": valid,
        "row_class": payload.get("row_class"),
        "profile_id": payload.get("profile_id"),
        "transport_moments": moments if valid else {},
        "source_only": False,
        "external_data_used": True if valid else None,
        "source_theorem_promotion_allowed": False,
        "production_constructive_receipt": False,
    }


def _pixel_profiles(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ew = artifacts.get("conditional_ew_envelope", {})
    endpoint = artifacts.get("empirical_thomson_endpoint", {})
    endpoint_values = _mapping(endpoint.get("endpoint"))
    return {
        "hierarchy_public": {
            "P": _nested(ew, "inputs", "P_calibration"),
            "epistemic_class": "endpoint_calibrated_comparison",
            "recovered_core_allowed": False,
        },
        "empirical_hadron_closure": {
            "P_central": _float_or_none(endpoint_values.get("P_central")),
            "P_interval": [_float_or_none(item) for item in endpoint_values.get("P_interval", [])],
            "epistemic_class": "external_data_closure",
            "recovered_core_allowed": False,
        },
        "measured_comparison": {
            "P": None,
            "epistemic_class": "comparison_only",
            "recovered_core_allowed": False,
        },
        "profile_selection_required": True,
        "implicit_global_profile_switch_allowed": False,
    }


def _source_obstructions(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    charged = artifacts.get("charged_trace_no_go", {})
    quark = artifacts.get("quark_sigma_no_go", {})
    scheme = artifacts.get("quark_scheme_obstruction", {})
    return {
        "charged_lepton_absolute_mass_origin_unresolved": charged.get("proof_status")
        == "no_go_confirmed_new_source_needed",
        "charged_lepton_public_promotion_allowed": _nested(charged, "promotion", "public_promotion_allowed")
        is True,
        "quark_two_moduli_source_nonidentifiability": quark.get("proof_status")
        == "closed_exact_current_corpus_obstruction",
        "quark_numeric_rows_allowed": quark.get("numeric_quark_rows_allowed") is True,
        "quark_scheme_obstruction": scheme.get("proof_status"),
        "numeric_mass_or_yukawa_packet_emitted": False,
    }


def _lattice_status(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload = artifacts.get("lattice_engine_status", {})
    export = artifacts.get("lattice_diagnostic_export", {})
    guards = _mapping(payload.get("guards"))
    executed = _mapping(payload.get("executed_output"))
    export_guards = _mapping(export.get("guards"))
    diagnostic = bool(
        payload.get("artifact") == "oph_lattice_engine_lane_status"
        and guards.get("diagnostic_output_promotable") is False
    )
    executed_receipt = bool(
        export.get("artifact") == "oph_lattice_diagnostic_backend_export"
        and export.get("execution_class") == "real_lattice_diagnostic_toy_scale"
        and export_guards.get("real_lattice_execution") is True
        and export_guards.get("target_anchored") is False
        and export_guards.get("surrogate_hadron_artifact") is False
        and export_guards.get("production_execution_class") is False
        and export_guards.get("public_promotion_allowed") is False
        and export_guards.get("satisfies_issue_425_closure") is False
    )
    return {
        "artifact_present": bool(payload),
        "diagnostic_adapter_receipt": diagnostic,
        "execution_class": "diagnostic" if diagnostic else "unavailable",
        "status": payload.get("status"),
        "export_present": bool(executed.get("export_present") is True or export),
        "executed_real_lattice_diagnostic_receipt": executed_receipt,
        "target_anchored": export_guards.get("target_anchored") if export else None,
        "execution_summary": {
            "ensemble": _mapping(export.get("ensemble")),
            "channels": _mapping(export.get("channels")),
            "dynamical_branch_validation": _mapping(export.get("dynamical_branch_validation")),
        }
        if executed_receipt
        else {},
        "production_qcd_receipt": False,
        "public_promotion_allowed": False,
        "remaining": _mapping(payload.get("issue_425_remaining")),
    }


def _payload(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    path = root / str(row.get("target_relpath"))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
