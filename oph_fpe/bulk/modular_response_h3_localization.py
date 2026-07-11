from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.claims import MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT


ETA = np.diag([-1.0, 1.0, 1.0, 1.0])

CERTIFIED = "MODULAR_RESPONSE_H3_LOCALIZATION_CERTIFIED"
AMBIGUOUS = "H3_LOCALIZATION_AMBIGUOUS"
OUTSIDE_DECLARED_DOMAIN = "OUTSIDE_DECLARED_DOMAIN"
FRAME_UNCONDITIONED = "H3_CAP_FRAME_UNCONDITIONED"
POINT_SOURCE_UNCERTIFIED = "POINT_SOURCE_FACTOR_UNCERTIFIED"
ERROR_UNBOUNDED = "H3_LOCALIZATION_ERROR_UNBOUNDED"
INCOMPLETE = "H3_LOCALIZATION_CERTIFICATE_INCOMPLETE"


def modular_response_h3_localization_report(payload: dict[str, Any], *, tol: float = 1.0e-8) -> dict[str, Any]:
    """Audit issue #310 modular-response localization from primitive fields.

    The receipt certifies conditional observer-facing H3 record localization.
    It does not infer point locality from a fitted point, promote existing H3
    coordinates to an independent reconstruction, derive particle species,
    construct stress-energy, or establish neutral third-person bulk.
    """

    source = dict(payload or {})
    radius = float(source.get("curvature_radius", 1.0) or 1.0)
    normals = _points4(source.get("normals") or source.get("cap_normals") or [])
    weights = _weight_matrix(source.get("weights"), len(normals))
    frame = _frame_report(normals, weights, radius=radius)
    domain = _domain_report(source.get("domain") or {}, radius=radius)
    observations = list(source.get("tokens") or source.get("observations") or [])
    declared_alpha = _finite_positive(source.get("alpha"))
    alpha = float(declared_alpha if declared_alpha is not None else frame["sigma_min"])
    declared_lipschitz = _finite_positive(source.get("L") or source.get("lipschitz_L"))
    lipschitz = float(declared_lipschitz if declared_lipschitz is not None else max(frame["sigma_max"], alpha))
    epsilon = _finite_nonnegative(source.get("epsilon", domain.get("epsilon", 0.0)))
    tau = _finite_nonnegative(source.get("tau", 0.0))

    token_rows = [
        _token_report(
            row,
            normals=normals,
            weights=weights,
            radius=radius,
            alpha=alpha,
            lipschitz=lipschitz,
            epsilon=float(epsilon or 0.0),
            tau=float(tau or 0.0),
            tol=tol,
        )
        for row in observations
    ]
    token_gap_receipt = bool(token_rows and all(row["gap_receipt"] for row in token_rows))
    token_error_receipt = bool(token_rows and all(row["error_receipt"] for row in token_rows))
    token_handoff_receipt = bool(token_rows and all(row["worldline_handoff_receipt"] for row in token_rows))
    all_inside_domain = bool(token_rows and all(row["inside_declared_domain"] for row in token_rows))

    geometry_receipt = bool(normals.size and frame["normal_unit_residual"] <= tol)
    frame_receipt = bool(frame["rank"] >= 4 and frame["sigma_min"] > tol and alpha > 0.0)
    kernel_receipt = _kernel_receipt(source)
    point_receipt = _point_source_receipt(source)
    domain_receipt = bool(domain["compact"] and all_inside_domain)
    error_receipt = bool(token_error_receipt and np.isfinite(alpha) and alpha > 0.0 and lipschitz >= alpha)
    naturality_receipt = bool((source.get("chart_naturality") or {}).get("passed", False))
    refinement_receipt = bool((source.get("refinement") or {}).get("passed", False))
    negative_controls_receipt = _negative_controls_receipt(source)
    no_shortcut = not bool(source.get("preexisting_h3_coordinates_used_as_reconstruction", False))

    component_receipts = {
        "H3_CAP_GEOMETRY_RECEIPT": geometry_receipt,
        "H3_CAP_FRAME_RECEIPT": frame_receipt,
        "H3_MODULAR_RESPONSE_KERNEL_RECEIPT": kernel_receipt,
        "H3_POINT_SOURCE_FACTOR_RECEIPT": point_receipt,
        "H3_LOCALIZATION_DOMAIN_RECEIPT": domain_receipt,
        "H3_LOCALIZATION_ERROR_RECEIPT": error_receipt,
        "H3_LOCALIZATION_GAP_RECEIPT": token_gap_receipt,
        "H3_CHART_NATURALITY_RECEIPT": naturality_receipt,
        "H3_REFINEMENT_LOCALIZATION_RECEIPT": refinement_receipt,
        "H3_WORLDLINE_HANDOFF_RECEIPT": token_handoff_receipt,
        "H3_NEGATIVE_CONTROLS_RECEIPT": negative_controls_receipt,
        "H3_NO_PREEXISTING_COORDINATE_SHORTCUT_RECEIPT": no_shortcut,
    }
    blockers = _blockers(component_receipts, frame=frame, domain=domain, token_rows=token_rows)
    status = _terminal_status(blockers, token_rows)
    receipt = bool(status == CERTIFIED)

    return {
        "mode": "modular_response_h3_localization_audit",
        "claim_id": "OPH-GR-D3-H3-LOCALIZATION",
        "receipt_type": MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT,
        "terminal_status": status,
        MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT: receipt,
        "h3_modular_response_localization_receipt": receipt,
        "H3LOC": receipt,
        "curvature_radius": radius,
        "frame": frame,
        "domain": domain,
        "observability": {
            "alpha": alpha,
            "L": lipschitz,
            "epsilon": epsilon,
            "tau": tau,
            "alpha_source": "declared" if declared_alpha is not None else "sigma_min_frame",
            "L_source": "declared" if declared_lipschitz is not None else "sigma_max_frame_floor",
        },
        "tokens": token_rows,
        "component_receipts": component_receipts,
        **component_receipts,
        "blockers": blockers,
        "mandatory_nonclaims": {
            "POINT_LOCALITY_DERIVED_FROM_FIT": False,
            "UNLABELED_SOURCE_MIXTURE_SOLVED": False,
            "EXTENDED_SOURCE_COLLAPSED_TO_POINT": False,
            "PARTICLE_SPECIES_DERIVED": False,
            "PHYSICAL_STRESS_TENSOR": False,
            "NEUTRAL_CHART_BLIND_BULK": False,
            "EINSTEIN_BRANCH_ENTRY": False,
        },
        "claim_boundary": (
            "MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT certifies issue #310 only when "
            "record-conditioned cap responses, point-source factorization, compact source domain, "
            "positive cap-frame observability, bounded total error, residual-gap uniqueness, "
            "chart naturality, refinement, and negative controls all pass. Existing H3 coordinates "
            "or visualization point clouds cannot set H3LOC=true."
        ),
    }


def write_modular_response_h3_localization_report(source: Path, out: Path) -> dict[str, Any]:
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    report = modular_response_h3_localization_report(payload)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _frame_report(normals: np.ndarray, weights: np.ndarray, *, radius: float) -> dict[str, Any]:
    if not normals.size:
        return {
            "normal_count": 0,
            "rank": 0,
            "sigma_min": 0.0,
            "sigma_max": 0.0,
            "normal_unit_residual": float("inf"),
            "B": [],
        }
    b = np.column_stack([-normals[:, 0], normals[:, 1], normals[:, 2], normals[:, 3]])
    weighted = _sqrtm(weights) @ b
    singular = np.linalg.svd(weighted, compute_uv=False)
    normal_res = float(np.max(np.abs([_eta(row, row) - 1.0 for row in normals])))
    rank = int(np.linalg.matrix_rank(weighted, tol=1.0e-10))
    return {
        "normal_count": int(len(normals)),
        "rank": rank,
        "sigma_min": float(singular[-1]) if singular.size else 0.0,
        "sigma_max": float(singular[0]) if singular.size else 0.0,
        "normal_unit_residual": normal_res,
        "B": b.tolist(),
        "curvature_radius": float(radius),
    }


def _domain_report(domain: dict[str, Any], *, radius: float) -> dict[str, Any]:
    center = _vec4(domain.get("center", [radius, 0.0, 0.0, 0.0]))
    physical_radius = _finite_positive(domain.get("D") or domain.get("radius"))
    epsilon = _finite_nonnegative(domain.get("epsilon", 0.0))
    compact = bool(str(domain.get("type", "ball")) in {"ball", "compact", "bounded"} and physical_radius is not None)
    net_cardinality_bound = None
    if compact and epsilon and epsilon > 0.0:
        net_cardinality_bound = _ball_volume(float(physical_radius) + radius * epsilon / 2.0, radius) / _ball_volume(
            radius * epsilon / 2.0,
            radius,
        )
    return {
        "type": str(domain.get("type", "ball")),
        "compact": compact,
        "center": center.tolist() if center is not None else None,
        "D": physical_radius,
        "epsilon": epsilon,
        "net_cardinality_bound": float(net_cardinality_bound) if net_cardinality_bound else None,
    }


def _token_report(
    row: dict[str, Any],
    *,
    normals: np.ndarray,
    weights: np.ndarray,
    radius: float,
    alpha: float,
    lipschitz: float,
    epsilon: float,
    tau: float,
    tol: float,
) -> dict[str, Any]:
    token_id = str(row.get("token_id", row.get("id", len(str(row)))))
    responses = _vec(row.get("responses") or row.get("y") or [])
    candidates = _points4(row.get("candidate_points") or row.get("net_points") or [])
    sigma = _finite_nonnegative(row.get("sigma", row.get("total_error_sigma", 0.0)))
    tau_i = _finite_nonnegative(row.get("tau", tau))
    epsilon_i = _finite_nonnegative(row.get("epsilon", epsilon))
    if responses is None or candidates.size == 0 or normals.size == 0:
        return {
            "token_id": token_id,
            "terminal_status": INCOMPLETE,
            "error_receipt": False,
            "gap_receipt": False,
            "worldline_handoff_receipt": False,
            "inside_declared_domain": False,
            "blockers": ["missing_responses_or_candidates_or_normals"],
        }
    feature = _signed_features(candidates, normals, radius)
    if len(responses) != feature.shape[1]:
        return {
            "token_id": token_id,
            "terminal_status": INCOMPLETE,
            "error_receipt": False,
            "gap_receipt": False,
            "worldline_handoff_receipt": False,
            "inside_declared_domain": False,
            "blockers": ["response_dimension_mismatch"],
        }
    residuals = np.asarray([_weighted_norm(feature[index] - responses, weights) for index in range(len(candidates))])
    best_index = int(np.argmin(residuals))
    sorted_residuals = np.sort(residuals)
    lower = _vec(row.get("residual_lower_bounds") or [])
    upper = _vec(row.get("residual_upper_bounds") or [])
    if lower is not None and upper is not None and len(lower) == len(candidates) and len(upper) == len(candidates):
        best_upper = float(upper[best_index])
        competitor_lower = float(np.min(np.delete(lower, best_index))) if len(lower) > 1 else float("inf")
        gap = competitor_lower - best_upper
    elif len(sorted_residuals) > 1 and sigma is not None:
        gap = float(sorted_residuals[1] - sorted_residuals[0] - 2.0 * sigma)
    else:
        gap = float("-inf")
    localization_radius = None
    if alpha > 0.0 and sigma is not None and epsilon_i is not None and tau_i is not None:
        localization_radius = float(radius * ((lipschitz / alpha) * epsilon_i + (2.0 / alpha) * sigma + (1.0 / alpha) * tau_i))
    inside = bool(all(_is_h3_point(candidate, radius=radius, tol=tol) for candidate in candidates))
    error_receipt = bool(localization_radius is not None and np.isfinite(localization_radius))
    gap_receipt = bool(gap > 0.0)
    status = CERTIFIED if gap_receipt and error_receipt and inside else AMBIGUOUS if error_receipt and inside else INCOMPLETE
    return {
        "token_id": token_id,
        "terminal_status": status,
        "best_index": best_index,
        "estimated_point": candidates[best_index].tolist(),
        "residuals": residuals.tolist(),
        "residual_gap": gap,
        "Delta_loc": gap,
        "sigma": sigma,
        "epsilon": epsilon_i,
        "tau": tau_i,
        "localization_radius": localization_radius,
        "support_ball": {
            "center": candidates[best_index].tolist(),
            "radius": localization_radius,
        },
        "error_receipt": error_receipt,
        "gap_receipt": gap_receipt,
        "worldline_handoff_receipt": bool(error_receipt and inside),
        "inside_declared_domain": inside,
        "blockers": [] if gap_receipt and error_receipt and inside else ["nonpositive_Delta_loc_or_incomplete_error_bound"],
    }


def _blockers(
    receipts: dict[str, bool],
    *,
    frame: dict[str, Any],
    domain: dict[str, Any],
    token_rows: list[dict[str, Any]],
) -> list[str]:
    blockers = [name for name, passed in receipts.items() if not passed]
    if frame.get("rank", 0) < 4:
        blockers.append(FRAME_UNCONDITIONED)
    if not domain.get("compact", False):
        blockers.append(OUTSIDE_DECLARED_DOMAIN)
    for row in token_rows:
        blockers.extend(str(value) for value in row.get("blockers", []))
    return sorted(set(blockers))


def _terminal_status(blockers: list[str], token_rows: list[dict[str, Any]]) -> str:
    if not blockers:
        return CERTIFIED
    if token_rows and all(row.get("error_receipt", False) for row in token_rows):
        return AMBIGUOUS
    if any(blocker == OUTSIDE_DECLARED_DOMAIN for blocker in blockers):
        return OUTSIDE_DECLARED_DOMAIN
    if any(blocker == FRAME_UNCONDITIONED for blocker in blockers):
        return FRAME_UNCONDITIONED
    if any("POINT_SOURCE" in blocker or "POINT_SOURCE" in blocker.upper() for blocker in blockers):
        return POINT_SOURCE_UNCERTIFIED
    if any("ERROR" in blocker or "sigma" in blocker.lower() for blocker in blockers):
        return ERROR_UNBOUNDED
    return INCOMPLETE


def _kernel_receipt(source: dict[str, Any]) -> bool:
    kernel = source.get("kernel") or {}
    if isinstance(kernel, str):
        return kernel in {"signed_linear", "paired_hinge", "identity"}
    return bool(kernel.get("type") in {"signed_linear", "paired_hinge", "identity"} and kernel.get("declared", True))


def _point_source_receipt(source: dict[str, Any]) -> bool:
    point = source.get("point_source") or source.get("point_source_factorization") or {}
    if isinstance(point, bool):
        return bool(point)
    return bool(point.get("passed", False) and point.get("held_out_residual_passed", True) and not point.get("mixture_control_failed", False))


def _negative_controls_receipt(source: dict[str, Any]) -> bool:
    controls = source.get("negative_controls") or {}
    if not isinstance(controls, dict) or not controls:
        return False
    return all(bool(value) for value in controls.values())


def _signed_features(points: np.ndarray, normals: np.ndarray, radius: float) -> np.ndarray:
    return np.asarray([[float(_eta(point, normal) / radius) for normal in normals] for point in points], dtype=float)


def _weight_matrix(value: Any, size: int) -> np.ndarray:
    if size <= 0:
        return np.zeros((0, 0), dtype=float)
    if value is None:
        return np.eye(size)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        return np.diag(arr)
    if arr.shape == (size, size):
        return arr
    return np.eye(size)


def _sqrtm(matrix: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.maximum(vals, 0.0)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


def _weighted_norm(vec: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.maximum(vec.T @ weights @ vec, 0.0)))


def _is_h3_point(point: np.ndarray, *, radius: float, tol: float) -> bool:
    return bool(point.shape == (4,) and point[0] > 0.0 and abs(_eta(point, point) + radius**2) <= max(tol, 1.0e-7))


def _ball_volume(r: float, radius: float) -> float:
    return float(np.pi * radius**3 * (np.sinh(2.0 * r / radius) - 2.0 * r / radius))


def _eta(x: np.ndarray, y: np.ndarray) -> float:
    return float(x @ ETA @ y)


def _points4(value: Any) -> np.ndarray:
    if not value:
        return np.zeros((0, 4), dtype=float)
    rows = [_vec4(row) for row in value]
    rows = [row for row in rows if row is not None]
    return np.asarray(rows, dtype=float) if rows else np.zeros((0, 4), dtype=float)


def _vec4(value: Any) -> np.ndarray | None:
    vec = _vec(value)
    if vec is None or len(vec) != 4:
        return None
    return vec


def _vec(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 1:
        return None
    return arr


def _finite_positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0.0 else None


def _finite_nonnegative(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number >= 0.0 else None
