from __future__ import annotations

import math
from typing import Any

import numpy as np

from oph_fpe.claims import (
    DECLARED_SHAPE_SUBSTRATE_WITNESS,
    SHAPE_CMB_CERTIFICATE_INPUT_RECEIPT,
    SHAPE_FOUR_SECTOR_IR_RECEIPT,
    SHAPE_VISIBLE_IR_TARGET_RECEIPT,
)
from oph_fpe.microphysics.shape_constants import DELTA_P


def fit_r2(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 1.0e-30:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def iqr(values: list[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return float(np.percentile(arr, 75.0) - np.percentile(arr, 25.0))


def estimate_kappa_rep(runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        delta = float(run.get("delta_P", DELTA_P))
        trace = np.asarray(run.get("phi_trace", []), dtype=float)
        time = np.arange(trace.shape[0], dtype=float)
        mask = np.isfinite(trace) & (trace > 1.0e-30)
        if delta <= 1.0e-12 or np.count_nonzero(mask) < 4:
            rows.append({"delta_P": delta, "usable": False, "reason": "insufficient_positive_trace_or_delta"})
            continue
        slope, intercept = np.polyfit(time[mask], np.log(trace[mask]), 1)
        r2 = fit_r2(time[mask], np.log(trace[mask]), float(slope), float(intercept))
        rows.append(
            {
                "delta_P": delta,
                "usable": bool(np.isfinite(slope) and np.isfinite(r2)),
                "slope": float(slope),
                "kappa_rep": float(-slope / max(delta, 1.0e-12)),
                "r2": float(r2),
            }
        )
    values = [float(row["kappa_rep"]) for row in rows if row.get("usable") and float(row.get("r2", 0.0)) > 0.8]
    return {
        "kappa_rep_estimate": float(np.median(values)) if values else None,
        "kappa_rep_iqr": iqr(values),
        "rows": rows,
        "passed": bool(len(values) >= 3),
        "not_inserted_by_hand": True,
    }


def eta_R_candidate(kappa_rep: float, delta_P: float) -> float:
    return float(kappa_rep * delta_P)


def q_ir_from_zero_mode(field_values: np.ndarray) -> float:
    values = np.asarray(field_values, dtype=float)
    if values.size == 0:
        return 0.0
    zero = float(np.mean(values))
    total = float(np.mean(values * values))
    return float(np.clip((zero * zero) / max(total, 1.0e-30), 0.0, 1.0))


def ell_ir_from_visible_covariance_rank(feature_matrix: np.ndarray, tol: float = 1.0e-9) -> dict[str, Any]:
    matrix = np.asarray(feature_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        return {"visible_covariance_rank": 0, "ell_IR_candidate": 0, "passed": False}
    cov = np.cov(matrix.T)
    cov = np.atleast_2d(cov)
    evals = np.linalg.eigvalsh(cov)
    scale = max(float(np.max(evals)), 1.0e-30)
    rank = int(np.sum(evals > float(tol) * scale))
    return {
        "visible_covariance_rank": rank,
        "ell_IR_candidate": max(rank - 1, 0),
        "passed": rank >= 8,
    }


def dodeca_visible_ir_target_certificate(
    *,
    face_count: int = 12,
    vertex_count: int = 20,
    identity_channel_count: int = 1,
) -> dict[str, Any]:
    visible_scalar_record_channels = int(face_count) + int(vertex_count)
    identity_augmented_channels = visible_scalar_record_channels + int(identity_channel_count)
    passed = bool(face_count == 12 and vertex_count == 20 and visible_scalar_record_channels == 32)
    return {
        "receipt": SHAPE_VISIBLE_IR_TARGET_RECEIPT,
        "receipt_name": SHAPE_VISIBLE_IR_TARGET_RECEIPT,
        "passed": passed,
        "face_channel_count": int(face_count),
        "vertex_channel_count": int(vertex_count),
        "visible_scalar_record_channels": int(visible_scalar_record_channels),
        "identity_channel_count": int(identity_channel_count),
        "identity_augmented_channels": int(identity_augmented_channels),
        "ell_IR_candidate": int(visible_scalar_record_channels),
        "theorem_side_target": True,
        "finite_lattice_derived": False,
        "physical_cmb_prediction": False,
        "claim_level": DECLARED_SHAPE_SUBSTRATE_WITNESS,
        "claim_boundary": (
            "Selector-elimination target from the declared dodecahedral visible scalar record channels "
            "F+V=12+20=32. This is not the same as the runtime covariance rank of a finite projected field."
        ),
    }


def four_sector_q_ir_certificate() -> dict[str, Any]:
    sector_count = 4
    return {
        "receipt": SHAPE_FOUR_SECTOR_IR_RECEIPT,
        "receipt_name": SHAPE_FOUR_SECTOR_IR_RECEIPT,
        "passed": True,
        "sector_count": sector_count,
        "q_IR_candidate": float(1.0 / sector_count),
        "sectors": [
            "spatial_collar_orientation_1",
            "spatial_collar_orientation_2",
            "spatial_collar_orientation_3",
            "record_clock_closure",
        ],
        "theorem_side_target": True,
        "finite_lattice_derived": False,
        "physical_cmb_prediction": False,
        "claim_level": DECLARED_SHAPE_SUBSTRATE_WITNESS,
        "claim_boundary": (
            "Selector-elimination target from the four-sector repair split. The finite simulator must still "
            "show that its repair/release ensemble realizes this split dynamically."
        ),
    }


def repair_transition_matrix(states_before: list[int], states_after: list[int], weights: list[float] | None = None):
    state_to_idx: dict[int, int] = {}
    counts: dict[tuple[int, int], float] = {}
    weights = weights or [1.0] * len(states_before)
    for before, after, weight in zip(states_before, states_after, weights):
        left = state_to_idx.setdefault(int(before), len(state_to_idx))
        right = state_to_idx.setdefault(int(after), len(state_to_idx))
        counts[(left, right)] = counts.get((left, right), 0.0) + float(weight)
    matrix = np.zeros((len(state_to_idx), len(state_to_idx)), dtype=float)
    for (left, right), value in counts.items():
        matrix[left, right] += value
    row_sum = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, np.maximum(row_sum, 1.0e-12))
    return matrix, state_to_idx


def gamma_rec_from_transition_matrix(matrix: np.ndarray, delta_eta: float = 1.0) -> dict[str, Any]:
    if matrix.size == 0:
        return {"lambda_2": None, "Gamma_rec": None, "finite": False}
    eig = np.linalg.eigvals(matrix)
    vals = sorted((abs(value) for value in eig), reverse=True)
    lam2 = vals[1] if len(vals) > 1 else 0.0
    gamma = -math.log(max(float(lam2), 1.0e-12)) / max(float(delta_eta), 1.0e-12)
    return {"lambda_2": float(lam2), "Gamma_rec": float(gamma), "finite": bool(np.isfinite(gamma))}


def shape_cmb_certificate_inputs_report(
    shape_runs: list[dict[str, Any]],
    projection_report: dict[str, Any],
    *,
    field_matrix: np.ndarray,
    primary_field: np.ndarray,
    transition_states: tuple[list[int], list[int]] | None = None,
) -> dict[str, Any]:
    kappa = estimate_kappa_rep(shape_runs)
    eta = None
    if kappa.get("passed") and kappa.get("kappa_rep_estimate") is not None:
        eta = eta_R_candidate(float(kappa["kappa_rep_estimate"]), float(shape_runs[0].get("delta_P", DELTA_P)))

    ell_ir = ell_ir_from_visible_covariance_rank(field_matrix)
    ell_ir_target = dodeca_visible_ir_target_certificate()
    q_ir_runtime = q_ir_from_zero_mode(primary_field)
    q_ir_target = four_sector_q_ir_certificate()
    if transition_states is None:
        transition_states = _transition_states_from_trace(shape_runs[0].get("phi_trace", []))
    transition, _ = repair_transition_matrix(list(transition_states[0]), list(transition_states[1]))
    gamma = gamma_rec_from_transition_matrix(transition)
    b_a_proxy = {
        "proxy_emitted": bool(primary_field.size > 0),
        "source": "shape_loop_repair_response_proxy",
        "mean_abs_response": float(np.mean(np.abs(primary_field))) if primary_field.size else None,
        "claim_boundary": "candidate Shape substrate response proxy; not a calibrated finite-collar B_A(k,a)",
    }
    passed = bool(kappa.get("passed") and eta is not None and gamma.get("finite") and b_a_proxy["proxy_emitted"])
    selector_elimination_target_input = bool(ell_ir_target.get("passed") and q_ir_target.get("passed"))
    return {
        "receipt": SHAPE_CMB_CERTIFICATE_INPUT_RECEIPT,
        "receipt_name": SHAPE_CMB_CERTIFICATE_INPUT_RECEIPT,
        "passed": passed,
        "selector_elimination_target_input_receipt": selector_elimination_target_input,
        "kappa_rep": kappa,
        "eta_R_candidate": eta,
        "q_IR_candidate": q_ir_target["q_IR_candidate"],
        "q_IR_runtime_zero_mode": q_ir_runtime,
        "q_IR_theorem_side_target": q_ir_target,
        "ell_IR_candidate": ell_ir_target["ell_IR_candidate"],
        "ell_IR_runtime_covariance": ell_ir,
        "ell_IR_theorem_side_target": ell_ir_target,
        "Gamma_rec_proxy": gamma,
        "B_A_proxy": b_a_proxy,
        "projection_point_count": projection_report.get("point_count"),
        "physical_cmb_prediction": False,
        "claim_level": DECLARED_SHAPE_SUBSTRATE_WITNESS,
        "neutral_oph_bulk_claim": False,
        "claim_boundary": (
            "Candidate inputs for a future Shape-to-CMB bridge. These values are emitted from the "
            "declared Shape substrate witness and are not a physical CMB prediction."
        ),
    }


def _transition_states_from_trace(trace: list[float] | np.ndarray) -> tuple[list[int], list[int]]:
    values = np.asarray(trace, dtype=float)
    if values.size < 2:
        return [0], [0]
    qs = np.quantile(values[np.isfinite(values)], [0.33, 0.66]) if np.any(np.isfinite(values)) else [0.0, 1.0]
    labels = np.digitize(values, qs).astype(int)
    return labels[:-1].tolist(), labels[1:].tolist()
