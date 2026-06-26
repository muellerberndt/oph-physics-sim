from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oph_fpe.constants.oph_pixel import (
    ALPHA_INV_ENDPOINT_CALIBRATED,
    ALPHA_INV_SOURCE_CANDIDATE,
)


DELTA_H_FP_CALIBRATED = ALPHA_INV_ENDPOINT_CALIBRATED - ALPHA_INV_SOURCE_CANDIDATE
DELTA_H_FP_CALIBRATED_SIGMA = 2.1e-8
C_Q_TARGET = 0.6580257599
C_Q_TOLERANCE = 1.0e-9
STANDARD_HADRONIC_RUNNING_DELTA_ALPHA_MZ = 0.0276

ACCEPTED_ENDPOINT_CONVENTIONS = {
    "oph_same_scheme_hadronic_endpoint",
    "oph_same_scheme_endpoint",
    "oph_a_z_same_scheme",
    "a_z_same_scheme",
}

LEAKAGE_FLAG_KEYS = (
    "uses_codata_alpha",
    "uses_CODATA_alpha",
    "uses_alpha0_measurement",
    "uses_alpha_0_measurement",
    "uses_measured_alpha",
    "uses_calibrated_gap",
    "uses_delta_h_calibrated",
    "uses_Delta_H_calibrated",
    "uses_target_gap",
    "uses_posthoc_fit",
    "uses_post_hoc_fit",
    "uses_fitted_polynomial_to_target",
    "root_chosen_after_seeing_target",
    "fit_to_target",
)

TARGET_FIELD_TOKENS = (
    "codata",
    "measured_alpha",
    "alpha0_measured",
    "alpha_0_measured",
    "calibrated_gap",
    "delta_h_cal",
    "delta_h_fp_cal",
    "target_delta",
    "target_gap",
    "posthoc",
    "post_hoc",
    "fit_to_target",
)


@dataclass(frozen=True)
class LeechEndpointBridgeInputs:
    """Inputs for the Leech/moonshine endpoint bridge quarantine audit.

    The audit checks whether a candidate artifact emits a source-only OPH
    same-scheme hadronic endpoint functional. It never promotes the
    fine-structure endpoint claim by itself.
    """

    source_artifact: Path | None = None
    delta_tolerance: float = DELTA_H_FP_CALIBRATED_SIGMA
    c_q_tolerance: float = C_Q_TOLERANCE
    source: str = "leech_moonshine_endpoint_bridge_quarantine"


def leech_endpoint_bridge_report(
    inputs: LeechEndpointBridgeInputs | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    bridge_inputs = inputs if inputs is not None else LeechEndpointBridgeInputs(**kwargs)
    delta_tolerance = _nonnegative_finite(bridge_inputs.delta_tolerance, "delta_tolerance")
    c_q_tolerance = _nonnegative_finite(bridge_inputs.c_q_tolerance, "c_q_tolerance")
    artifact, artifact_meta, load_error = _load_candidate_artifact(bridge_inputs.source_artifact)

    candidate_delta = _optional_finite(_candidate_value(artifact, ("delta_H_fp", "Delta_H_fp", "delta_h_fp")))
    candidate_c_q = _optional_finite(_candidate_value(artifact, ("c_Q", "C_Q", "cq")))
    candidate_alpha_endpoint = _optional_finite(
        _candidate_value(
            artifact,
            (
                "alpha_inverse_endpoint",
                "alpha_inv_endpoint",
                "alpha_inverse",
                "alpha_inv",
            ),
        )
    )
    delta_residual = (
        None if candidate_delta is None else float(candidate_delta - DELTA_H_FP_CALIBRATED)
    )
    c_q_residual = None if candidate_c_q is None else float(candidate_c_q - C_Q_TARGET)
    endpoint_alpha_residual = (
        None
        if candidate_alpha_endpoint is None
        else float(candidate_alpha_endpoint - ALPHA_INV_ENDPOINT_CALIBRATED)
    )

    endpoint_delta_emitted = candidate_delta is not None
    c_q_emitted = candidate_c_q is not None
    alpha_endpoint_only = (
        candidate_alpha_endpoint is not None and candidate_delta is None and candidate_c_q is None
    )
    delta_target_match = bool(
        candidate_delta is not None and abs(candidate_delta - DELTA_H_FP_CALIBRATED) <= delta_tolerance
    )
    c_q_target_match = bool(candidate_c_q is not None and abs(candidate_c_q - C_Q_TARGET) <= c_q_tolerance)
    target_value_match = bool(delta_target_match or c_q_target_match)

    leakage_flags = _leakage_flags(artifact)
    targetish_fields = _targetish_input_fields(artifact)
    target_leakage_free = not leakage_flags and not targetish_fields
    endpoint_convention = str(_candidate_value(artifact, ("endpoint_convention", "scheme")) or "").strip()
    normalized_endpoint_convention = endpoint_convention.lower()
    same_scheme_flag = _candidate_bool(
        artifact,
        (
            "same_scheme_endpoint_transport",
            "same_scheme_hadronic_endpoint_transport",
            "same_scheme_endpoint_functional",
        ),
    )
    same_scheme_endpoint = bool(
        same_scheme_flag or normalized_endpoint_convention in ACCEPTED_ENDPOINT_CONVENTIONS
    )
    hadronic_object = _candidate_bool(
        artifact,
        (
            "hadronic_em_spectral_transport",
            "hadronic_electromagnetic_spectral_transport",
            "hadronic_spectral_transport_object",
        ),
    )
    source_only_map = _candidate_bool(
        artifact,
        (
            "source_only_map_receipt",
            "source_only_functional_receipt",
            "no_data_use_receipt",
        ),
    )
    full_endpoint_map = _candidate_bool(
        artifact,
        (
            "full_endpoint_map_receipt",
            "R_Q_endpoint_map_receipt",
            "delta_had_delta_ew_endpoint_map_receipt",
        ),
    )
    string_branch_descends = _candidate_bool(
        artifact,
        (
            "oph_string_branch_descent_receipt",
            "edge_carrier_descent_receipt",
            "leech_descends_to_oph_edge_carrier",
        ),
    )
    alternative_branch_declared = _candidate_bool(
        artifact,
        (
            "alternative_branch_declared",
            "leech_alternative_branch",
            "c24_alternative_branch_declared",
        ),
    )
    fixed_point_loop = _candidate_bool(
        artifact,
        (
            "fixed_point_loop_reproduced",
            "fixed_point_loop_receipt",
            "full_fixed_point_loop_receipt",
        ),
    )
    interval_uniqueness = _candidate_bool(
        artifact,
        (
            "interval_uniqueness_receipt",
            "contraction_interval_certificate",
            "unique_interval_receipt",
        ),
    )
    fixed_point_interval = bool(fixed_point_loop and interval_uniqueness)

    gates = {
        "source_artifact_loaded": artifact is not None,
        "delta_H_fp_or_c_Q_emitted": bool(endpoint_delta_emitted or c_q_emitted),
        "alpha_endpoint_only_rejected": not alpha_endpoint_only,
        "delta_H_fp_target_match": delta_target_match,
        "c_Q_target_match": c_q_target_match,
        "target_value_match": target_value_match,
        "target_leakage_free": target_leakage_free,
        "same_scheme_endpoint_transport": same_scheme_endpoint,
        "hadronic_em_spectral_transport_object": bool(hadronic_object),
        "source_only_map_receipt": bool(source_only_map),
        "full_endpoint_map_receipt": bool(full_endpoint_map),
        "oph_string_branch_descent_receipt": bool(string_branch_descends),
        "alternative_branch_declared": bool(alternative_branch_declared),
        "fixed_point_loop_reproduced": bool(fixed_point_loop),
        "interval_uniqueness_receipt": bool(interval_uniqueness),
        "fixed_point_interval_receipt": bool(fixed_point_interval),
        "standard_hadronic_running_used_as_inverse_alpha_correction": False,
        "fine_structure_alpha_endpoint_promotion": False,
    }
    same_scheme_functional_receipt = bool(
        gates["source_artifact_loaded"]
        and gates["delta_H_fp_or_c_Q_emitted"]
        and gates["alpha_endpoint_only_rejected"]
        and gates["target_value_match"]
        and gates["target_leakage_free"]
        and gates["same_scheme_endpoint_transport"]
        and gates["hadronic_em_spectral_transport_object"]
        and gates["source_only_map_receipt"]
        and gates["full_endpoint_map_receipt"]
        and gates["oph_string_branch_descent_receipt"]
        and gates["fixed_point_interval_receipt"]
        and not gates["alternative_branch_declared"]
    )
    gates["same_scheme_hadronic_endpoint_functional_receipt"] = same_scheme_functional_receipt
    bridge_candidate_receipt = same_scheme_functional_receipt

    blockers = _blockers(
        load_error=load_error,
        gates=gates,
        leakage_flags=leakage_flags,
        targetish_fields=targetish_fields,
        alpha_endpoint_only=alpha_endpoint_only,
    )
    promotion_blockers = ["paper_review_promotion_gate_closed"]

    return {
        "mode": "leech_endpoint_bridge_quarantine_v0",
        "source": bridge_inputs.source,
        "question": "Can Leech/moonshine data derive the OPH same-scheme hadronic endpoint functional?",
        "LEECH_ENDPOINT_BRIDGE_QUARANTINE_RECEIPT": True,
        "LEECH_ENDPOINT_BRIDGE_CANDIDATE_RECEIPT": bool(bridge_candidate_receipt),
        "SAME_SCHEME_HADRONIC_ENDPOINT_FUNCTIONAL_RECEIPT": bool(same_scheme_functional_receipt),
        "FINE_STRUCTURE_ALPHA_ENDPOINT_PROMOTION_RECEIPT": False,
        "PHYSICAL_ALPHA_PREDICTION_RECEIPT": False,
        "bridge_status": _bridge_status(artifact is not None, bridge_candidate_receipt),
        "inputs": {
            "source_artifact": None if bridge_inputs.source_artifact is None else str(bridge_inputs.source_artifact),
            "delta_tolerance": delta_tolerance,
            "c_q_tolerance": c_q_tolerance,
        },
        "artifact": artifact_meta,
        "calibrated_endpoint_reference": {
            "alpha_inverse_source_candidate": ALPHA_INV_SOURCE_CANDIDATE,
            "alpha_inverse_endpoint_calibrated": ALPHA_INV_ENDPOINT_CALIBRATED,
            "Delta_H_cal_fp": DELTA_H_FP_CALIBRATED,
            "Delta_H_cal_fp_sigma": DELTA_H_FP_CALIBRATED_SIGMA,
            "c_Q_target_equivalent": C_Q_TARGET,
            "role": (
                "Audit target only. These values define the strict comparison row and do not supply "
                "a source-only hadronic endpoint derivation."
            ),
        },
        "candidate_outputs": {
            "delta_H_fp": candidate_delta,
            "delta_H_fp_residual_vs_calibrated": delta_residual,
            "delta_H_fp_abs_residual": None if delta_residual is None else abs(delta_residual),
            "c_Q": candidate_c_q,
            "c_Q_residual_vs_target": c_q_residual,
            "c_Q_abs_residual": None if c_q_residual is None else abs(c_q_residual),
            "alpha_inverse_endpoint": candidate_alpha_endpoint,
            "alpha_inverse_endpoint_residual_vs_calibrated": endpoint_alpha_residual,
            "alpha_endpoint_only": bool(alpha_endpoint_only),
        },
        "scheme_audit": {
            "endpoint_convention": endpoint_convention or None,
            "same_scheme_endpoint_transport": bool(same_scheme_endpoint),
            "hadronic_em_spectral_transport_object": bool(hadronic_object),
            "source_only_map_receipt": bool(source_only_map),
            "full_endpoint_map_receipt": bool(full_endpoint_map),
            "target_leakage_flags": leakage_flags,
            "targetish_input_fields": targetish_fields,
        },
        "string_branch_audit": {
            "policy": (
                "A Leech or c=24 object must descend to the OPH edge carrier. If the candidate is "
                "only an alternative branch, the bridge remains quarantined."
            ),
            "oph_string_branch_descent_receipt": bool(string_branch_descends),
            "alternative_branch_declared": bool(alternative_branch_declared),
        },
        "fixed_point_audit": {
            "fixed_point_loop_reproduced": bool(fixed_point_loop),
            "interval_uniqueness_receipt": bool(interval_uniqueness),
            "fixed_point_interval_receipt": bool(fixed_point_interval),
        },
        "convention_mixing_control": {
            "standard_hadronic_running_delta_alpha_had_5_M_Z_reference": STANDARD_HADRONIC_RUNNING_DELTA_ALPHA_MZ,
            "invalid_additive_inverse_alpha_insertion": (
                ALPHA_INV_SOURCE_CANDIDATE + STANDARD_HADRONIC_RUNNING_DELTA_ALPHA_MZ
            ),
            "invalid_insertion_residual_vs_endpoint": (
                ALPHA_INV_SOURCE_CANDIDATE
                + STANDARD_HADRONIC_RUNNING_DELTA_ALPHA_MZ
                - ALPHA_INV_ENDPOINT_CALIBRATED
            ),
            "rejected_as_same_scheme_inverse_alpha_correction": True,
            "claim_boundary": (
                "Standard hadronic running at M_Z is not the additive OPH inverse-alpha endpoint "
                "correction. This row is a convention-mixing guard, not a bridge input."
            ),
        },
        "readiness_gates": gates,
        "blockers": blockers,
        "promotion_blockers": promotion_blockers,
        "claim_boundary": (
            "Quarantined Leech/moonshine endpoint-bridge audit. A passing bridge candidate would "
            "mean that the supplied artifact emitted a source-only same-scheme OPH hadronic endpoint "
            "functional with string-branch descent and fixed-point interval receipts. The simulator "
            "does not promote the fine-structure endpoint prediction from this audit."
        ),
        "artifact_schema_hint": {
            "required_outputs": ["delta_H_fp or c_Q"],
            "required_receipts": [
                "same_scheme_endpoint_transport",
                "hadronic_em_spectral_transport",
                "source_only_map_receipt",
                "full_endpoint_map_receipt",
                "oph_string_branch_descent_receipt",
                "fixed_point_loop_reproduced",
                "interval_uniqueness_receipt",
            ],
            "forbidden_inputs_or_flags": list(LEAKAGE_FLAG_KEYS),
        },
    }


def write_leech_endpoint_bridge_report(
    out_dir: Path,
    inputs: LeechEndpointBridgeInputs | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    report = leech_endpoint_bridge_report(inputs, **kwargs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "leech_endpoint_bridge_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "leech_endpoint_bridge_report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _load_candidate_artifact(path: Path | None) -> tuple[dict[str, Any] | None, dict[str, Any], str | None]:
    if path is None:
        return None, {"path": None, "loaded": False}, "source_artifact_missing"
    source_path = Path(path)
    meta: dict[str, Any] = {"path": str(source_path), "loaded": False}
    if not source_path.exists():
        return None, meta, "source_artifact_missing"
    try:
        raw = source_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        meta["error"] = str(exc)
        return None, meta, "source_artifact_unreadable"
    if not isinstance(payload, dict):
        meta["error"] = "source artifact must be a JSON object"
        return None, meta, "source_artifact_not_json_object"
    meta.update(
        {
            "loaded": True,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
        }
    )
    return payload, meta, None


def _candidate_value(artifact: dict[str, Any] | None, keys: tuple[str, ...]) -> Any:
    if artifact is None:
        return None
    for key in keys:
        if key in artifact:
            return artifact[key]
    receipts = artifact.get("receipts")
    if isinstance(receipts, dict):
        for key in keys:
            if key in receipts:
                return receipts[key]
    outputs = artifact.get("outputs")
    if isinstance(outputs, dict):
        for key in keys:
            if key in outputs:
                return outputs[key]
    return None


def _candidate_bool(artifact: dict[str, Any] | None, keys: tuple[str, ...]) -> bool:
    value = _candidate_value(artifact, keys)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "pass", "passed", "receipt"}
    return False


def _optional_finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _nonnegative_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return result


def _leakage_flags(artifact: dict[str, Any] | None) -> list[str]:
    if artifact is None:
        return []
    flags: list[str] = []
    for key in LEAKAGE_FLAG_KEYS:
        if _candidate_bool(artifact, (key,)):
            flags.append(key)
    dependencies = _candidate_value(artifact, ("input_dependencies", "dependencies", "source_dependencies"))
    if isinstance(dependencies, list):
        lowered = " ".join(str(item).lower() for item in dependencies)
        for token in ("codata", "measured alpha", "alpha(0)", "calibrated gap", "target gap"):
            if token in lowered:
                flags.append(f"dependency:{token}")
    return sorted(set(flags))


def _targetish_input_fields(artifact: dict[str, Any] | None) -> list[str]:
    if artifact is None:
        return []
    fields: list[str] = []

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                normalized = str(key).lower()
                if key not in LEAKAGE_FLAG_KEYS and any(token in normalized for token in TARGET_FIELD_TOKENS):
                    fields.append(path)
                visit(path, nested)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                visit(f"{prefix}[{index}]", nested)

    visit("", artifact)
    return sorted(set(fields))


def _blockers(
    *,
    load_error: str | None,
    gates: dict[str, bool],
    leakage_flags: list[str],
    targetish_fields: list[str],
    alpha_endpoint_only: bool,
) -> list[str]:
    blockers: list[str] = []
    if load_error:
        blockers.append(load_error)
    if not gates["delta_H_fp_or_c_Q_emitted"]:
        blockers.append("delta_H_fp_or_c_Q_not_emitted")
    if alpha_endpoint_only:
        blockers.append("alpha_endpoint_decimal_without_endpoint_functional")
    if gates["delta_H_fp_or_c_Q_emitted"] and not gates["target_value_match"]:
        blockers.append("candidate_output_misses_calibrated_endpoint_interval")
    if leakage_flags or targetish_fields:
        blockers.append("target_leakage_or_posthoc_fit_not_excluded")
    if not gates["same_scheme_endpoint_transport"]:
        blockers.append("same_scheme_endpoint_transport_missing")
    if not gates["hadronic_em_spectral_transport_object"]:
        blockers.append("hadronic_em_spectral_transport_semantics_missing")
    if not gates["source_only_map_receipt"]:
        blockers.append("source_only_map_receipt_missing")
    if not gates["full_endpoint_map_receipt"]:
        blockers.append("full_endpoint_map_receipt_missing")
    if not gates["oph_string_branch_descent_receipt"]:
        blockers.append("leech_object_does_not_descend_to_oph_edge_carrier")
    if gates["alternative_branch_declared"]:
        blockers.append("leech_candidate_marked_alternative_branch")
    if not gates["fixed_point_loop_reproduced"] or not gates["interval_uniqueness_receipt"]:
        blockers.append("fixed_point_loop_or_interval_uniqueness_missing")
    blockers.append("paper_review_promotion_gate_closed")
    return blockers


def _bridge_status(artifact_loaded: bool, bridge_candidate_receipt: bool) -> str:
    if not artifact_loaded:
        return "no_candidate_artifact"
    if bridge_candidate_receipt:
        return "source_only_same_scheme_bridge_candidate_receipt"
    return "quarantined_bridge_candidate_failed"


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{float(value):.12g}"
    return str(value)


def _markdown_report(report: dict[str, Any]) -> str:
    gates = report["readiness_gates"]
    outputs = report["candidate_outputs"]
    convention = report["convention_mixing_control"]
    return "\n".join(
        [
            "# Leech Endpoint Bridge",
            "",
            str(report["claim_boundary"]),
            "",
            "## Receipt Summary",
            "",
            f"- bridge status: `{report['bridge_status']}`",
            f"- quarantine receipt: {_fmt(report['LEECH_ENDPOINT_BRIDGE_QUARANTINE_RECEIPT'])}",
            f"- bridge candidate receipt: {_fmt(report['LEECH_ENDPOINT_BRIDGE_CANDIDATE_RECEIPT'])}",
            f"- same-scheme hadronic endpoint functional receipt: {_fmt(report['SAME_SCHEME_HADRONIC_ENDPOINT_FUNCTIONAL_RECEIPT'])}",
            f"- fine-structure alpha endpoint promotion receipt: {_fmt(report['FINE_STRUCTURE_ALPHA_ENDPOINT_PROMOTION_RECEIPT'])}",
            "",
            "## Candidate Outputs",
            "",
            f"- Delta_H_fp: {_fmt(outputs['delta_H_fp'])}",
            f"- Delta_H_fp residual: {_fmt(outputs['delta_H_fp_residual_vs_calibrated'])}",
            f"- c_Q: {_fmt(outputs['c_Q'])}",
            f"- c_Q residual: {_fmt(outputs['c_Q_residual_vs_target'])}",
            f"- alpha endpoint only: {_fmt(outputs['alpha_endpoint_only'])}",
            "",
            "## Gates",
            "",
            *[f"- {key}: {_fmt(value)}" for key, value in gates.items()],
            "",
            "## Convention Mixing Control",
            "",
            (
                "- standard hadronic running Delta alpha_had^(5)(M_Z) reference: "
                f"{_fmt(convention['standard_hadronic_running_delta_alpha_had_5_M_Z_reference'])}"
            ),
            (
                "- invalid additive inverse-alpha insertion: "
                f"{_fmt(convention['invalid_additive_inverse_alpha_insertion'])}"
            ),
            "- rejected as same-scheme inverse-alpha correction: true",
            "",
            "## Blockers",
            "",
            *[f"- `{item}`" for item in report["blockers"]],
            "",
        ]
    )
