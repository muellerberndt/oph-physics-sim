from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


PARENT_RECEIPT = "FINITE_COVARIANT_COLLAR_PACKET_PARENT_RECEIPT"
STRESS_CLOSURE_RECEIPT = "STRESS_ENERGY_CLOSURE_RECEIPT"
GAUGE_INDEPENDENCE_RECEIPT = "GAUGE_INDEPENDENCE_RECEIPT"
EXPLICIT_RECIPIENT_STRESS_RECEIPT = "EXPLICIT_RECIPIENT_STRESS_RECEIPT"
CAUSAL_RESPONSE_RECEIPT = "CAUSAL_RESPONSE_RECEIPT"
REFINEMENT_CONVERGENCE_RECEIPT = "REFINEMENT_CONVERGENCE_RECEIPT"
FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT = "FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT"
RECIPIENT_PACKET_LABELS = {"recipient", "R", "repair_recipient"}
ANOMALY_PACKET_LABELS = {"anomaly", "A"}


def finite_covariant_collar_packet_parent_report(
    artifact: dict[str, Any],
    *,
    tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    """Validate the finite covariant collar-packet parent contract.

    This is a source-artifact validator, not a Boltzmann solver. It checks the
    pieces needed before scalar dark/anomaly tables can be promoted to physical
    CMB or matter-transfer inputs.
    """

    manifest = _section(artifact, "manifest")
    packets = _section(artifact, "packets")
    repair = _section(artifact, "repair")
    causal = _section(artifact, "causal_response")
    stress = _section(artifact, "stress")
    refinement = _section(artifact, "refinement")
    cdm_limit = _section(artifact, "cdm_limit")
    gauge = _section(artifact, "gauge")
    frozen = _section(artifact, "frozen_run")

    blockers: list[str] = []
    for key in ("source_hash", "regulator_id", "parent_theorem_version"):
        if not manifest.get(key):
            blockers.append(f"manifest_{key}_missing")

    packet_rows = packets.get("states") if isinstance(packets.get("states"), list) else []
    packet_check = _packet_state_check(packet_rows)
    blockers.extend(packet_check["blockers"])

    stress_residual = _float(
        stress.get(
            "stress_energy_closure_residual",
            repair.get("four_momentum_conservation_residual"),
        )
    )
    stress_closed = bool(
        artifact.get(STRESS_CLOSURE_RECEIPT, False)
        or stress.get(STRESS_CLOSURE_RECEIPT, False)
        or (stress_residual is not None and stress_residual <= float(tolerance))
    )
    if not stress_closed:
        blockers.append("stress_energy_closure_not_certified")

    gamma_rec = _float(repair.get("Gamma_rec", repair.get("gamma_rec", repair.get("spectral_gap"))))
    gamma_nonzero = bool(gamma_rec is not None and gamma_rec > float(tolerance))
    recipient_states = [row for row in packet_rows if str(row.get("label")) in RECIPIENT_PACKET_LABELS]
    recipient_receipt = bool(
        artifact.get(EXPLICIT_RECIPIENT_STRESS_RECEIPT, False)
        or repair.get(EXPLICIT_RECIPIENT_STRESS_RECEIPT, False)
        or (not gamma_nonzero)
        or (recipient_states and stress_closed)
    )
    if gamma_nonzero and not recipient_receipt:
        blockers.append("explicit_recipient_stress_missing_for_nonzero_Gamma_rec")
    if gamma_rec is not None and gamma_rec < -float(tolerance):
        blockers.append("Gamma_rec_negative")

    detailed_balance_residual = _float(repair.get("detailed_balance_residual"))
    detailed_balance = bool(
        repair.get("detailed_balance_receipt", False)
        or (detailed_balance_residual is not None and detailed_balance_residual <= float(tolerance))
    )
    if not detailed_balance:
        blockers.append("detailed_balance_not_certified")

    characteristic_speed = _float(causal.get("characteristic_speed_bound"))
    causal_receipt = bool(
        artifact.get(CAUSAL_RESPONSE_RECEIPT, False)
        or causal.get(CAUSAL_RESPONSE_RECEIPT, False)
        or (
            characteristic_speed is not None
            and characteristic_speed <= 1.0 + float(tolerance)
            and _matrices_present(causal, ("kinetic_matrix", "damping_matrix", "propagation_matrix", "source_matrix", "output_matrix"))
        )
    )
    if not causal_receipt:
        blockers.append("causal_response_not_certified")

    gauge_receipt = bool(
        artifact.get(GAUGE_INDEPENDENCE_RECEIPT, False)
        or gauge.get(GAUGE_INDEPENDENCE_RECEIPT, False)
        or gauge.get("newtonian_synchronous_agree_within_error", False)
    )
    if not gauge_receipt:
        blockers.append("gauge_independence_not_certified")

    refinement_receipt = bool(
        artifact.get(REFINEMENT_CONVERGENCE_RECEIPT, False)
        or refinement.get(REFINEMENT_CONVERGENCE_RECEIPT, False)
        or refinement.get("stress_and_response_converge", False)
    )
    if not refinement_receipt:
        blockers.append("refinement_convergence_not_certified")

    cdm_limit_receipt = bool(
        cdm_limit.get("CDM_LIMIT_RECOVERY_RECEIPT", False)
        or cdm_limit.get("cdm_limit_regression_passed", False)
    )
    if not cdm_limit_receipt:
        blockers.append("cdm_limit_recovery_not_certified")

    source_hash = str(frozen.get("source_hash") or manifest.get("source_hash") or "")
    solver_hash = str(frozen.get("solver_hash") or "")
    likelihood_hash = str(frozen.get("likelihood_hash") or "")
    frozen_receipt = bool(
        artifact.get(FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT, False)
        or frozen.get(FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT, False)
        or (source_hash and solver_hash and likelihood_hash and frozen.get("mutable_source_artifacts", False) is False)
    )
    if not frozen_receipt:
        blockers.append("frozen_likelihood_protocol_not_certified")

    parent_receipt = not blockers
    return {
        "mode": "finite_covariant_collar_packet_parent_v0",
        PARENT_RECEIPT: parent_receipt,
        STRESS_CLOSURE_RECEIPT: stress_closed,
        GAUGE_INDEPENDENCE_RECEIPT: gauge_receipt,
        EXPLICIT_RECIPIENT_STRESS_RECEIPT: recipient_receipt,
        CAUSAL_RESPONSE_RECEIPT: causal_receipt,
        REFINEMENT_CONVERGENCE_RECEIPT: refinement_receipt,
        "CDM_LIMIT_RECOVERY_RECEIPT": cdm_limit_receipt,
        FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT: frozen_receipt,
        "Gamma_rec_nonzero": gamma_nonzero,
        "source_hash": source_hash or None,
        "solver_hash": solver_hash or None,
        "likelihood_hash": likelihood_hash or None,
        "packet_state_count": len(packet_rows),
        "anomaly_state_count": packet_check["anomaly_state_count"],
        "recipient_state_count": len(recipient_states),
        "stress_energy_closure_residual": stress_residual,
        "detailed_balance_residual": detailed_balance_residual,
        "characteristic_speed_bound": characteristic_speed,
        "blockers": blockers,
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "Finite covariant collar-packet parent source contract. Passing this report licenses source "
            "functions for a Boltzmann handoff; it is not itself a CMB likelihood result."
        ),
    }


def write_finite_covariant_collar_packet_parent_report(source: Path, out: Path) -> dict[str, Any]:
    artifact = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError("finite parent artifact must be a JSON object")
    report = finite_covariant_collar_packet_parent_report(artifact)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _packet_state_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    blockers: list[str] = []
    if not rows:
        blockers.append("packet_states_missing")
    anomaly_count = 0
    for index, row in enumerate(rows):
        density = _float(row.get("rho", row.get("density", row.get("background_weight"))))
        if density is None or density < 0.0:
            blockers.append(f"packet_{index}_density_invalid")
        if str(row.get("label")) in ANOMALY_PACKET_LABELS:
            anomaly_count += 1
        velocity = row.get("u_mu", row.get("four_velocity"))
        if velocity is not None:
            velocity_array = np.asarray(velocity, dtype=float)
            if velocity_array.shape != (4,) or not np.all(np.isfinite(velocity_array)) or velocity_array[0] <= 0.0:
                blockers.append(f"packet_{index}_four_velocity_not_future_directed")
    if anomaly_count <= 0:
        blockers.append("anomaly_packet_states_missing")
    return {"blockers": blockers, "anomaly_state_count": anomaly_count}


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def _matrices_present(data: dict[str, Any], keys: tuple[str, ...]) -> bool:
    for key in keys:
        if key not in data:
            return False
        try:
            array = np.asarray(data[key], dtype=float)
        except (TypeError, ValueError):
            return False
        if array.size == 0 or not np.all(np.isfinite(array)):
            return False
    return True


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None
