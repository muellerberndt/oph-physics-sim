from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.integrate import solve_ivp


def piecewise_affine_rg_receipt(
    initial_state: Sequence[float],
    *,
    initial_scale: float,
    segments: Sequence[dict[str, Any]],
    source_packet_verified: bool = False,
    same_branch: bool = False,
    scheme: str = "unspecified",
    loop_order: str = "unspecified",
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
) -> dict[str, Any]:
    """Transport a coefficient vector through frozen affine RG segments.

    Each segment implements dX/dlog(mu) = A X + b followed by an optional
    affine matching map. This is a numerical transport primitive, not a source
    theorem selecting A, b, thresholds, matching, scheme, or loop order.
    """

    state = np.asarray(initial_state, dtype=float)
    if state.ndim != 1 or state.size == 0:
        raise ValueError("initial RG state must be a nonempty vector")
    if initial_scale <= 0.0:
        raise ValueError("initial RG scale must be positive")
    if not segments:
        raise ValueError("at least one RG segment is required")

    current_scale = float(initial_scale)
    rows: list[dict[str, Any]] = []
    amplification = 1.0
    accumulated_error = 0.0
    all_success = True

    for index, segment in enumerate(segments):
        end_scale = float(segment["end_scale"])
        if end_scale <= 0.0 or end_scale == current_scale:
            raise ValueError("each RG segment must end at a distinct positive scale")
        matrix = np.asarray(segment["matrix"], dtype=float)
        offset = np.asarray(segment.get("offset", np.zeros(state.size)), dtype=float)
        if matrix.shape != (state.size, state.size) or offset.shape != state.shape:
            raise ValueError("RG matrix and offset dimensions must match the state")

        t0 = math.log(current_scale)
        t1 = math.log(end_scale)

        def beta(_time: float, value: np.ndarray) -> np.ndarray:
            return matrix @ value + offset

        solution = solve_ivp(beta, (t0, t1), state, method="DOP853", rtol=rtol, atol=atol)
        success = bool(solution.success and np.all(np.isfinite(solution.y[:, -1])))
        all_success = all_success and success
        transported = np.asarray(solution.y[:, -1], dtype=float)

        matching_matrix = np.asarray(segment.get("matching_matrix", np.eye(state.size)), dtype=float)
        matching_offset = np.asarray(segment.get("matching_offset", np.zeros(state.size)), dtype=float)
        if matching_matrix.shape != matrix.shape or matching_offset.shape != state.shape:
            raise ValueError("matching map dimensions must match the state")
        state = matching_matrix @ transported + matching_offset

        lipschitz = float(np.linalg.norm(matrix, ord=2))
        matching_bound = float(np.linalg.norm(matching_matrix, ord=2))
        local_factor = matching_bound * math.exp(lipschitz * abs(t1 - t0))
        local_error = float(segment.get("truncation_error_bound", 0.0))
        accumulated_error = local_factor * accumulated_error + local_error
        amplification *= local_factor
        rows.append(
            {
                "index": index,
                "start_scale": current_scale,
                "end_scale": end_scale,
                "solver_success": success,
                "solver_message": solution.message,
                "steps": int(len(solution.t)),
                "lipschitz_bound": lipschitz,
                "matching_operator_bound": matching_bound,
                "stability_factor": local_factor,
                "truncation_error_bound": local_error,
                "output_state": [float(value) for value in state],
            }
        )
        current_scale = end_scale

    frozen_convention = scheme != "unspecified" and loop_order != "unspecified"
    promoted = bool(source_packet_verified and same_branch and frozen_convention and all_success)
    blockers = []
    if not source_packet_verified:
        blockers.append("rg_source_packet_not_verified")
    if not same_branch:
        blockers.append("rg_same_branch_not_verified")
    if not frozen_convention:
        blockers.append("rg_scheme_or_loop_order_not_frozen")
    if not all_success:
        blockers.append("rg_numerical_transport_failed")

    return {
        "schema": "oph_wzh_rg_transport_receipt_v1",
        "scheme": scheme,
        "loop_order": loop_order,
        "initial_scale": float(initial_scale),
        "final_scale": current_scale,
        "initial_state": [float(value) for value in initial_state],
        "final_state": [float(value) for value in state],
        "segments": rows,
        "source_to_output_amplification_bound": amplification,
        "integrated_error_bound": accumulated_error,
        "numerical_transport_success": all_success,
        "rg_matching_receipt": promoted,
        "promotion_allowed": promoted,
        "blockers": blockers,
        "claim_boundary": (
            "The solver proves deterministic transport for the supplied affine packet; it does "
            "not derive the physical beta functions, thresholds, matching maps, or scheme."
        ),
    }
