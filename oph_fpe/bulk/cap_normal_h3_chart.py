from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.claims import CAP_NORMAL_H3_CHART_RECEIPT


ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
U0 = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)

CERTIFIED = "CAP_NORMAL_H3_CHART_CERTIFIED"
APPROXIMATE = "CAP_NORMAL_H3_CHART_APPROXIMATE"
CAP_NOT_ROUND = "CAP_NOT_ROUND"
CAP_ORIENTATION_AMBIGUOUS = "CAP_ORIENTATION_AMBIGUOUS"
LORENTZ_TRANSITION_INVALID = "LORENTZ_TRANSITION_INVALID"
TIME_ORIENTATION_INVALID = "TIME_ORIENTATION_INVALID"
H3_RADIUS_UNSOURCED = "H3_RADIUS_UNSOURCED"
H3_CHART_INVALID = "H3_CHART_INVALID"
INCOMPLETE = "H3_CHART_CERTIFICATE_INCOMPLETE"


def cap_normal_h3_chart_report(payload: dict[str, Any], *, tol: float = 1.0e-8) -> dict[str, Any]:
    """Audit the issue #309 cap-normal H3 chart receipt from primitive fields.

    The receipt certifies the declared analytic chart semantics. It does not
    promote a fitted display, object population, neutral bulk, stress tensor,
    physical curvature radius, or Einstein branch entry.
    """

    source = dict(payload or {})
    sky = _points3(source.get("sky_points") or source.get("omegas") or [])
    caps = list(source.get("caps") or [])
    transitions = list(source.get("transitions") or [])
    h3_points = _points4(source.get("h3_points") or source.get("observer_frames") or [])
    radius_provenance = str(source.get("radius_provenance", "UNIT_CONVENTION"))
    curvature_radius = float(source.get("curvature_radius", 1.0) or 1.0)
    declared_exact = _declared_exact(source, caps)

    q = np.asarray([_q(row) for row in sky], dtype=float) if sky.size else np.zeros((0, 4), dtype=float)
    normals: list[np.ndarray] = []
    cap_rows: list[dict[str, Any]] = []
    boundary_values: list[float] = []
    interior_values: list[float] = []
    exterior_values: list[float] = []
    cap_errors: list[str] = []

    for index, cap in enumerate(caps):
        row = _cap_row(index, cap, tol=tol)
        cap_rows.append(row)
        if row["normal"] is not None:
            n = np.asarray(row["normal"], dtype=float)
            normals.append(n)
            boundary_values.extend(_incidences(n, _points3(cap.get("boundary_points") or [])))
            interior_values.extend(_incidences(n, _points3(cap.get("interior_points") or cap.get("inside_points") or [])))
            exterior_values.extend(_incidences(n, _points3(cap.get("exterior_points") or cap.get("outside_points") or [])))
        if row["status"] != "ok":
            cap_errors.append(row["status"])

    omega_residual = _max_abs(np.sum(sky * sky, axis=1) - 1.0) if sky.size else None
    null_residual = _max_abs(np.asarray([_eta(row, row) for row in q], dtype=float)) if q.size else None
    normal_residual = (
        _max_abs(np.asarray([_eta(row, row) - 1.0 for row in normals], dtype=float)) if normals else None
    )
    boundary_residual = _max_abs(np.asarray(boundary_values, dtype=float)) if boundary_values else None
    interior_margin = float(np.min(interior_values)) if interior_values else None
    exterior_margin = float(-np.max(exterior_values)) if exterior_values else None

    transition_rows = [_transition_row(row, sky=sky, q=q, caps=caps, normals=normals, tol=tol) for row in transitions]
    lorentz_residual = _max_optional(row.get("lorentz_residual") for row in transition_rows)
    ray_residual = _max_optional(row.get("ray_residual") for row in transition_rows)
    cap_equivariance_residual = _max_optional(row.get("cap_equivariance_residual") for row in transition_rows)

    h3 = _h3_row(h3_points, transitions, curvature_radius=curvature_radius)
    residuals = {
        "r_omega": omega_residual,
        "r_null": null_residual,
        "r_cap_normal": normal_residual,
        "r_boundary_C": boundary_residual,
        "m_in": interior_margin,
        "m_out": exterior_margin,
        "r_lorentz": lorentz_residual,
        "r_ray": ray_residual,
        "r_cap_equivariance": cap_equivariance_residual,
        "r_h3": h3["h3_sheet_residual"],
        "r_distance": h3["distance_invariance_residual"],
    }

    blockers = _blockers(
        residuals,
        caps=caps,
        sky=sky,
        h3_points=h3_points,
        transition_rows=transition_rows,
        cap_errors=cap_errors,
        declared_exact=declared_exact,
        radius_provenance=radius_provenance,
        source=source,
        tol=tol,
    )
    status = _status(blockers, declared_exact)
    receipt = bool(status == CERTIFIED)
    approximate = bool(status == APPROXIMATE)

    report = {
        "mode": "cap_normal_h3_chart_audit",
        "claim_id": "OPH-GR-D3-CAP-H3",
        "receipt_type": CAP_NORMAL_H3_CHART_RECEIPT,
        "terminal_status": status,
        CAP_NORMAL_H3_CHART_RECEIPT: receipt,
        "cap_normal_h3_chart_receipt": receipt,
        "h3_chart_approximate": approximate,
        "source_exactness": "exact_or_certified_round_cap" if declared_exact else "sampled_or_fitted_chart",
        "lorentz_convention": {
            "coordinate_order": ["time", "x", "y", "z"],
            "metric_signature": [-1, 1, 1, 1],
            "future_sheet": "q0>0 and u0>0",
            "orientation": source.get("orientation", "declared"),
            "radius_provenance": radius_provenance,
        },
        "celestial_section": {
            "q_formula": "q(Omega)=(1,Omega)",
            "sample_count": int(len(sky)),
        },
        "caps": cap_rows,
        "transitions": transition_rows,
        "h3_chart": h3,
        "residuals": residuals,
        "thresholds": {"tol": float(tol)},
        "blockers": blockers,
        "mandatory_nonclaims": {
            "SUPPORT_VISIBLE_H3_CHART": receipt,
            "POPULATED_BULK": False,
            "NEUTRAL_CHART_BLIND_BULK": False,
            "PHYSICAL_CURVATURE_RADIUS_DERIVED": bool(
                radius_provenance == "OPH_DERIVED_SCALE" and source.get("derived_scale_receipt")
            ),
            "PHYSICAL_STRESS_TENSOR": False,
            "EINSTEIN_METRIC": False,
            "EINSTEIN_BRANCH_ENTRY": False,
        },
        "claim_boundary": (
            "CAP_NORMAL_H3_CHART_RECEIPT certifies the issue #309 chart theorem from primitive "
            "null-section, cap-normal, Lorentz-transition, and H3-sheet fields. A finite fitted "
            "display without a global round-cap certificate is approximate. A passing chart does not "
            "populate H3 with records, establish chart-blind neutral bulk, derive physical R_H, "
            "produce stress-energy, or close Einstein branch entry."
        ),
    }
    return report


def write_cap_normal_h3_chart_report(source: Path, out: Path) -> dict[str, Any]:
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    report = cap_normal_h3_chart_report(payload)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _cap_row(index: int, cap: dict[str, Any], *, tol: float) -> dict[str, Any]:
    center = _vec3(cap.get("center") or cap.get("c"))
    alpha = _number(cap.get("alpha") if cap.get("alpha") is not None else cap.get("angular_radius"))
    if center is None or alpha is None:
        return {"index": index, "status": "missing_center_or_alpha", "normal": None}
    center_norm = float(np.linalg.norm(center))
    if center_norm <= 0.0:
        return {"index": index, "status": "zero_center", "normal": None}
    c = center / center_norm
    if not (0.0 < float(alpha) < np.pi):
        return {"index": index, "status": CAP_NOT_ROUND, "normal": None}
    n = np.concatenate([[1.0 / np.tan(alpha)], (1.0 / np.sin(alpha)) * c])
    normal_residual = abs(_eta(n, n) - 1.0)
    return {
        "index": index,
        "source_type": str(cap.get("source_type", "analytic_round_cap")),
        "status": "ok" if normal_residual <= max(tol, 1.0e-12) else CAP_NOT_ROUND,
        "center_norm_residual": abs(center_norm - 1.0),
        "alpha": float(alpha),
        "normal": [float(value) for value in n],
        "normal_norm_residual": float(normal_residual),
        "incidence_formula": "(c dot Omega - cos(alpha))/sin(alpha)",
    }


def _transition_row(
    row: dict[str, Any],
    *,
    sky: np.ndarray,
    q: np.ndarray,
    caps: list[dict[str, Any]],
    normals: list[np.ndarray],
    tol: float,
) -> dict[str, Any]:
    matrix = _matrix4(row.get("lambda") or row.get("Lambda") or row.get("lorentz_matrix"))
    if matrix is None:
        return {"status": "missing_lorentz_matrix"}
    lorentz_residual = float(np.max(np.abs(matrix.T @ ETA @ matrix - ETA)))
    det = float(np.linalg.det(matrix))
    future_u0 = float((matrix @ U0)[0])
    omega_min = None
    ray_residual = None
    if q.size:
        lq = (matrix @ q.T).T
        omega = lq[:, 0]
        omega_min = float(np.min(omega))
        projected = np.asarray([_q(row4[1:] / row4[0]) for row4 in lq], dtype=float)
        ray_residual = float(np.max(np.abs(lq - omega[:, None] * projected)))
    image_caps = list(row.get("cap_images") or row.get("transformed_caps") or [])
    cap_equivariance = []
    for image in image_caps:
        cap_index = int(image.get("cap_index", image.get("source_index", 0)))
        if 0 <= cap_index < len(normals):
            image_row = _cap_row(cap_index, image, tol=tol)
            if image_row.get("normal") is not None:
                image_normal = np.asarray(image_row["normal"], dtype=float)
                cap_equivariance.append(float(np.max(np.abs(image_normal - matrix @ normals[cap_index]))))
    cap_equivariance_residual = max(cap_equivariance) if cap_equivariance else None
    proper = bool(det > 0.0 and abs(det - 1.0) <= 1.0e-6)
    orthochronous = bool(future_u0 > 0.0 and (omega_min is None or omega_min > 0.0))
    return {
        "status": "ok" if lorentz_residual <= tol and proper and orthochronous else LORENTZ_TRANSITION_INVALID,
        "lorentz_residual": lorentz_residual,
        "det": det,
        "proper": proper,
        "future_u0": future_u0,
        "omega_min": omega_min,
        "orthochronous": orthochronous,
        "ray_residual": ray_residual,
        "cap_equivariance_residual": cap_equivariance_residual,
        "cap_equivariance_samples": len(cap_equivariance),
    }


def _h3_row(points: np.ndarray, transitions: list[dict[str, Any]], *, curvature_radius: float) -> dict[str, Any]:
    if not points.size:
        return {"point_count": 0, "h3_sheet_residual": None, "distance_invariance_residual": None}
    radius2 = float(curvature_radius) ** 2
    sheet = np.asarray([abs(_eta(row, row) + radius2) for row in points], dtype=float)
    future_min = float(np.min(points[:, 0]))
    distances = []
    for transition in transitions:
        matrix = _matrix4(transition.get("lambda") or transition.get("Lambda") or transition.get("lorentz_matrix"))
        if matrix is None:
            continue
        moved = (matrix @ points.T).T
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distances.append(abs(_h3_distance(moved[i], moved[j], curvature_radius) - _h3_distance(points[i], points[j], curvature_radius)))
    return {
        "model": "future_unit_timelike_hyperboloid",
        "point_count": int(len(points)),
        "curvature_radius": float(curvature_radius),
        "h3_sheet_residual": float(np.max(sheet)) if sheet.size else None,
        "future_time_min": future_min,
        "distance_invariance_residual": float(max(distances)) if distances else 0.0,
    }


def _blockers(
    residuals: dict[str, float | None],
    *,
    caps: list[dict[str, Any]],
    sky: np.ndarray,
    h3_points: np.ndarray,
    transition_rows: list[dict[str, Any]],
    cap_errors: list[str],
    declared_exact: bool,
    radius_provenance: str,
    source: dict[str, Any],
    tol: float,
) -> list[str]:
    blockers: list[str] = []
    if len(sky) == 0:
        blockers.append("missing_sky_points")
    if not caps:
        blockers.append("missing_caps")
    if not transition_rows:
        blockers.append("missing_lorentz_transitions")
    if len(h3_points) == 0:
        blockers.append("missing_h3_points")
    if cap_errors:
        blockers.append(CAP_NOT_ROUND)
    for key in ("r_omega", "r_null", "r_cap_normal", "r_boundary_C", "r_lorentz", "r_ray", "r_h3", "r_distance"):
        value = residuals.get(key)
        if value is not None and float(value) > tol:
            blockers.append(f"{key}_above_tol")
    if residuals.get("m_in") is None or residuals.get("m_out") is None:
        blockers.append(CAP_ORIENTATION_AMBIGUOUS)
    elif float(residuals["m_in"]) <= tol or float(residuals["m_out"]) <= tol:
        blockers.append(CAP_ORIENTATION_AMBIGUOUS)
    if residuals.get("r_cap_equivariance") is None:
        blockers.append("missing_cap_equivariance_samples")
    elif float(residuals["r_cap_equivariance"]) > tol:
        blockers.append("r_cap_equivariance_above_tol")
    if any(not row.get("orthochronous", False) for row in transition_rows):
        blockers.append(TIME_ORIENTATION_INVALID)
    if any(row.get("status") == LORENTZ_TRANSITION_INVALID for row in transition_rows):
        blockers.append(LORENTZ_TRANSITION_INVALID)
    if h3_points.size and np.min(h3_points[:, 0]) <= 0.0:
        blockers.append(H3_CHART_INVALID)
    if radius_provenance == "OPH_DERIVED_SCALE" and not source.get("derived_scale_receipt"):
        blockers.append(H3_RADIUS_UNSOURCED)
    if not declared_exact:
        blockers.append(APPROXIMATE)
    return list(dict.fromkeys(blockers))


def _status(blockers: list[str], declared_exact: bool) -> str:
    if any(item in blockers for item in ("missing_sky_points", "missing_caps", "missing_lorentz_transitions", "missing_h3_points", "missing_cap_equivariance_samples")):
        return INCOMPLETE
    if CAP_NOT_ROUND in blockers:
        return CAP_NOT_ROUND
    if LORENTZ_TRANSITION_INVALID in blockers:
        return LORENTZ_TRANSITION_INVALID
    if TIME_ORIENTATION_INVALID in blockers:
        return TIME_ORIENTATION_INVALID
    if H3_RADIUS_UNSOURCED in blockers:
        return H3_RADIUS_UNSOURCED
    if H3_CHART_INVALID in blockers or any(item.startswith("r_h3") for item in blockers):
        return H3_CHART_INVALID
    if CAP_ORIENTATION_AMBIGUOUS in blockers:
        return CAP_ORIENTATION_AMBIGUOUS
    if not declared_exact or APPROXIMATE in blockers:
        return APPROXIMATE
    return CERTIFIED if not blockers else INCOMPLETE


def _declared_exact(source: dict[str, Any], caps: list[dict[str, Any]]) -> bool:
    if source.get("global_roundness_certificate") or source.get("round_cap_theorem_certificate"):
        return True
    if not caps:
        return False
    return all(str(cap.get("source_type", "analytic_round_cap")) == "analytic_round_cap" for cap in caps)


def _q(omega: np.ndarray) -> np.ndarray:
    return np.concatenate([[1.0], np.asarray(omega, dtype=float)])


def _eta(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.asarray(a, dtype=float) @ ETA @ np.asarray(b, dtype=float))


def _h3_distance(a: np.ndarray, b: np.ndarray, radius: float) -> float:
    arg = max(1.0, -_eta(a, b) / (float(radius) ** 2))
    return float(radius) * float(np.arccosh(arg))


def _incidences(normal: np.ndarray, points: np.ndarray) -> list[float]:
    if not points.size:
        return []
    return [_eta(normal, _q(row)) for row in points]


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def _max_optional(values: Any) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return max(finite) if finite else None


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _vec3(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        return None
    return array


def _points3(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return np.zeros((0, 3), dtype=float)
    return array.reshape((-1, 3))


def _points4(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return np.zeros((0, 4), dtype=float)
    return array.reshape((-1, 4))


def _matrix4(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.shape != (4, 4):
        return None
    return array
