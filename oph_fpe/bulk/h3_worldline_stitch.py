from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


H3_STITCH_CERTIFIED = "H3_STITCH_CERTIFIED"
H3_WORLDLINE_STITCH_CERTIFIED = "H3_WORLDLINE_STITCH_CERTIFIED"
H3_STITCH_AMBIGUOUS = "H3_STITCH_AMBIGUOUS"
H3_STITCH_REJECTED = "H3_STITCH_REJECTED"
H3_INTERACTION_REQUIRED = "H3_INTERACTION_REQUIRED"
H3_ATLAS_INVALID = "H3_ATLAS_INVALID"
H3_CERTIFICATE_INCOMPLETE = "H3_CERTIFICATE_INCOMPLETE"

TERMINAL_STATUSES = (
    H3_STITCH_CERTIFIED,
    H3_STITCH_AMBIGUOUS,
    H3_STITCH_REJECTED,
    H3_INTERACTION_REQUIRED,
    H3_ATLAS_INVALID,
    H3_CERTIFICATE_INCOMPLETE,
)

_ETA = np.diag([-1.0, 1.0, 1.0, 1.0])


def minkowski_inner(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> float:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.shape != (4,) or right_arr.shape != (4,):
        raise ValueError("H3 hyperboloid points must have shape (4,)")
    return float(-left_arr[0] * right_arr[0] + np.dot(left_arr[1:], right_arr[1:]))


def h3_point_from_spatial_point(point: list[float] | np.ndarray, *, curvature_radius: float = 1.0) -> np.ndarray:
    """Return an H3_R hyperboloid point from either a 4-vector or tangent-chart row."""

    radius = _positive_radius(curvature_radius)
    arr = np.asarray(point, dtype=float)
    if arr.shape == (4,):
        return arr
    if arr.shape != (3,):
        raise ValueError("H3 spatial point must be a tangent 3-vector or a hyperboloid 4-vector")
    tangent_norm = float(np.linalg.norm(arr))
    if tangent_norm < 1e-15:
        return np.array([radius, 0.0, 0.0, 0.0], dtype=float)
    direction = arr / tangent_norm
    scaled = tangent_norm / radius
    out = np.empty(4, dtype=float)
    out[0] = radius * math.cosh(scaled)
    out[1:] = radius * math.sinh(scaled) * direction
    return out


def h3_distance(
    left: list[float] | np.ndarray,
    right: list[float] | np.ndarray,
    *,
    curvature_radius: float = 1.0,
) -> float:
    radius = _positive_radius(curvature_radius)
    left_point = h3_point_from_spatial_point(left, curvature_radius=radius)
    right_point = h3_point_from_spatial_point(right, curvature_radius=radius)
    gram = -minkowski_inner(left_point, right_point) / (radius * radius)
    return float(radius * math.acosh(max(1.0, gram)))


def lorentz_matrix_residual(matrix: list[list[float]] | np.ndarray) -> float:
    mat = np.asarray(matrix, dtype=float)
    if mat.shape != (4, 4):
        raise ValueError("Lorentz matrix must have shape (4, 4)")
    return float(np.max(np.abs(mat.T @ _ETA @ mat - _ETA)))


def h3_worldline_stitch_certificate_report(
    artifact: dict[str, Any],
    *,
    tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    atlas = _section(artifact, "atlas", "h3_atlas", "H3AtlasManifest")
    clock = _section(artifact, "clock", "clock_map", "ClockMap")
    extraction = _section(artifact, "extraction", "proto_record_extraction")
    descent = _section(artifact, "descent", "overlap_descent", "DescentCluster")
    crossing = _section(artifact, "crossing", "interface", "crossing_germ", "CrossingGerm")
    transport = _section(artifact, "transport", "sector_transport", "gauge_transport")
    assignment = _section(artifact, "assignment", "AssignmentCertificate")
    refinement = _section(artifact, "refinement", "RefinementCertificate")
    interaction = _section(artifact, "interaction", "InteractionEvent")

    blockers: list[str] = []
    missing: list[str] = []
    ambiguous: list[str] = []
    atlas_blockers: list[str] = []
    metric_rows: list[dict[str, Any]] = []

    radius = _float(atlas.get("R_H", atlas.get("curvature_radius")), default=0.0)
    if radius <= 0.0:
        atlas_blockers.append("h3_curvature_radius_missing_or_nonpositive")
        radius = 1.0
    model = str(atlas.get("model") or atlas.get("chart_model") or atlas.get("h3_model") or "")
    if not any(token in model.lower() for token in ("hyperboloid", "so+(3,1)", "so^+(1,3)", "so(3,1)")):
        atlas_blockers.append("h3_hyperboloid_model_not_declared")
    _validate_atlas_points(atlas, radius=radius, tolerance=tolerance, atlas_blockers=atlas_blockers)
    _validate_lorentz_transitions(atlas, tolerance=tolerance, atlas_blockers=atlas_blockers)

    _require_bool(clock, ("common_time_line", "comparison_time_line", "comparison_time_metric"), missing, blockers)
    _require_bool(clock, ("orientation_preserving", "clock_orientation_preserving"), missing, blockers)
    max_clock_error = _float(clock.get("max_error", clock.get("clock_uncertainty")), default=None)
    time_margin = _float(clock.get("time_order_margin", clock.get("adjacency_margin")), default=None)
    if max_clock_error is None or time_margin is None:
        missing.append("clock_uncertainty_or_time_order_margin_missing")
    elif time_margin <= max_clock_error:
        blockers.append("observer_time_adjacency_margin_not_positive_after_clock_error")

    _require_bool(extraction, ("thresholds_frozen", "detector_thresholds_frozen"), missing, blockers)
    if not extraction.get("detector_hash") and not extraction.get("detector_scalar_declared"):
        missing.append("chart_natural_detector_scalar_missing")
    _require_residual(extraction, "chart_naturality_residual", tolerance, missing, blockers)

    _require_bool(descent, ("overlap_graph_present", "component_cover_present"), missing, blockers)
    support_error = _float(descent.get("support_error", descent.get("support_radius_error")), default=0.0)
    join_margin = _float(descent.get("same_component_join_margin"), default=None)
    separation_margin = _float(descent.get("distinct_component_separation_margin"), default=None)
    if join_margin is None or separation_margin is None:
        missing.append("overlap_descent_margins_missing")
    elif min(join_margin, separation_margin) <= support_error:
        blockers.append("overlap_descent_margins_do_not_dominate_support_error")
    _require_residual(descent, "triple_overlap_cocycle_residual", tolerance, missing, blockers, optional=True)

    if _bool(interaction.get("interaction_required"), default=False):
        return _finish_report(
            artifact,
            status=H3_INTERACTION_REQUIRED,
            radius=radius,
            blockers=_unique(["interaction_solver_required"] + atlas_blockers + blockers + missing),
            missing=missing,
            ambiguous=ambiguous,
            metric_rows=metric_rows,
            gap=None,
        )
    _require_bool(interaction, ("free_propagation_slab", "free_slab_receipt"), missing, blockers)

    _require_bool(crossing, ("real_interface_contact", "real_boundary_contact"), missing, blockers)
    _require_bool(crossing, ("transverse", "transverse_crossing"), missing, blockers)
    if _bool(crossing.get("grazing", crossing.get("tangential")), default=False):
        blockers.append("tangential_or_grazing_interface_crossing")
    if _bool(crossing.get("file_boundary_only", crossing.get("synthetic_boundary_only")), default=False):
        blockers.append("file_boundary_is_not_a_physical_interface")
    normal_speed = _float(crossing.get("normal_velocity_lower_bound"), default=None)
    signed_margin = _float(crossing.get("signed_distance_margin"), default=None)
    if normal_speed is None or signed_margin is None:
        missing.append("oriented_interface_crossing_margins_missing")
    elif normal_speed <= 0.0 or signed_margin <= 0.0:
        blockers.append("oriented_interface_crossing_margin_not_positive")

    _require_bool(transport, ("common_chart", "common_chart_prediction"), missing, blockers)
    _require_bool(transport, ("sector_continuity", "sector_transport_continuity"), missing, blockers)
    _require_bool(transport, ("gauge_transport_continuity", "holonomy_transport_continuity"), missing, blockers)
    _require_bool(transport, ("holonomy_compared_covariantly", "atlas_and_gauge_holonomy_separated"), missing, blockers)
    _require_bool(transport, ("connector_declared", "connection_declared"), missing, blockers)
    _require_residual(transport, "transport_residual", tolerance, missing, blockers, optional=True)

    _require_bool(assignment, ("complete_graph", "complete_candidate_graph"), missing, blockers)
    _require_bool(assignment, ("one_to_one", "one_to_one_assignment"), missing, blockers)
    _require_bool(assignment, ("appearance_disappearance_penalties", "birth_death_penalties"), missing, blockers)
    if _bool(assignment.get("uses_record_ids", assignment.get("record_ids_used")), default=True):
        blockers.append("record_ids_used_in_admissibility_or_cost")
    shard_ids = assignment.get("distinct_shard_ids") or assignment.get("shard_ids") or []
    if not isinstance(shard_ids, list) or len({str(value) for value in shard_ids}) < 2:
        missing.append("cross_boundary_distinct_shard_ids_missing")
    proposed = _float(assignment.get("proposed_cost_upper", assignment.get("winner_cost_upper")), default=None)
    runner_up = _float(assignment.get("runner_up_cost_lower", assignment.get("runner_cost_lower")), default=None)
    required_gap = _float(assignment.get("required_gap", assignment.get("ambiguity_margin")), default=0.0)
    gap = None if proposed is None or runner_up is None else float(runner_up - proposed)
    if gap is None:
        missing.append("assignment_winner_and_runner_up_cost_bounds_missing")
    elif gap <= 0.0:
        ambiguous.append("assignment_cost_gap_nonpositive")
    elif gap <= required_gap:
        ambiguous.append("assignment_gap_below_required_ambiguity_margin")
    if _bool(assignment.get("multiple_optima"), default=False) or int(assignment.get("equal_cost_assignment_count") or 0) > 1:
        ambiguous.append("assignment_has_multiple_equal_cost_optima")

    _validate_candidate_metrics(artifact, radius=radius, metric_rows=metric_rows, missing=missing, blockers=blockers)

    _require_bool(refinement, ("coarse_fine_pair", "refinement_pair_declared"), missing, blockers)
    _require_bool(refinement, ("Q_sr_present", "refinement_projection_present"), missing, blockers)
    _require_bool(refinement, ("contracted_graph_isomorphic", "coarse_fine_contraction_agrees"), missing, blockers)
    eta_sr = _float(refinement.get("eta_sr", refinement.get("refinement_error")), default=None)
    coarse_gap = _float(refinement.get("coarse_gap", refinement.get("coarse_assignment_gap")), default=gap)
    if eta_sr is None or coarse_gap is None:
        missing.append("refinement_gap_or_eta_missing")
    elif coarse_gap <= 2.0 * eta_sr:
        ambiguous.append("refinement_gap_not_larger_than_two_eta")

    if atlas_blockers:
        status = H3_ATLAS_INVALID
    elif ambiguous:
        status = H3_STITCH_AMBIGUOUS
    elif missing:
        status = H3_CERTIFICATE_INCOMPLETE
    elif blockers:
        status = H3_STITCH_REJECTED
    else:
        status = H3_STITCH_CERTIFIED
    return _finish_report(
        artifact,
        status=status,
        radius=radius,
        blockers=_unique(atlas_blockers + blockers + missing + ambiguous),
        missing=missing,
        ambiguous=ambiguous,
        metric_rows=metric_rows,
        gap=gap,
    )


def write_h3_worldline_stitch_certificate_report(
    source: Path,
    out: Path,
    *,
    tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    source = Path(source)
    out = Path(out)
    with source.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    report = h3_worldline_stitch_certificate_report(artifact, tolerance=tolerance)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _positive_radius(value: float) -> float:
    radius = float(value)
    if radius <= 0.0 or not math.isfinite(radius):
        raise ValueError("H3 curvature radius must be positive")
    return radius


def _section(artifact: dict[str, Any], *names: str) -> dict[str, Any]:
    for name in names:
        value = artifact.get(name)
        if isinstance(value, dict):
            return value
    return {}


def _bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "pass", "passed"}
    return bool(value)


def _float(value: Any, *, default: float | None) -> float | None:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _require_bool(
    section: dict[str, Any],
    names: tuple[str, ...],
    missing: list[str],
    blockers: list[str],
) -> None:
    present = [name for name in names if name in section]
    label = names[0]
    if not present:
        missing.append(f"{label}_missing")
        return
    if not any(_bool(section.get(name), default=False) for name in names):
        blockers.append(f"{label}_failed")


def _require_residual(
    section: dict[str, Any],
    name: str,
    tolerance: float,
    missing: list[str],
    blockers: list[str],
    *,
    optional: bool = False,
) -> None:
    if name not in section:
        if not optional:
            missing.append(f"{name}_missing")
        return
    residual = _float(section.get(name), default=None)
    if residual is None:
        missing.append(f"{name}_not_numeric")
    elif abs(residual) > tolerance:
        blockers.append(f"{name}_above_tolerance")


def _validate_atlas_points(
    atlas: dict[str, Any],
    *,
    radius: float,
    tolerance: float,
    atlas_blockers: list[str],
) -> None:
    rows = atlas.get("points") or atlas.get("h3_points") or []
    if not isinstance(rows, list):
        atlas_blockers.append("h3_atlas_points_not_a_list")
        return
    for index, row in enumerate(rows):
        point = row.get("X", row.get("point")) if isinstance(row, dict) else row
        try:
            arr = h3_point_from_spatial_point(point, curvature_radius=radius)
        except (TypeError, ValueError):
            atlas_blockers.append(f"h3_atlas_point_{index}_invalid")
            continue
        norm = minkowski_inner(arr, arr)
        if arr[0] <= 0.0 or abs(norm + radius * radius) > max(tolerance, 1.0e-8):
            atlas_blockers.append(f"h3_atlas_point_{index}_off_hyperboloid")


def _validate_lorentz_transitions(
    atlas: dict[str, Any],
    *,
    tolerance: float,
    atlas_blockers: list[str],
) -> None:
    transitions = atlas.get("chart_transitions") or atlas.get("transitions") or []
    if not isinstance(transitions, list):
        atlas_blockers.append("h3_chart_transitions_not_a_list")
        return
    if not transitions:
        atlas_blockers.append("h3_chart_transitions_missing")
        return
    for index, row in enumerate(transitions):
        matrix = row.get("matrix") if isinstance(row, dict) else row
        try:
            residual = lorentz_matrix_residual(matrix)
            mat = np.asarray(matrix, dtype=float)
        except (TypeError, ValueError):
            atlas_blockers.append(f"h3_chart_transition_{index}_invalid")
            continue
        if residual > max(tolerance, 1.0e-8):
            atlas_blockers.append(f"h3_chart_transition_{index}_not_lorentz")
        if float(np.linalg.det(mat)) <= 0.0:
            atlas_blockers.append(f"h3_chart_transition_{index}_orientation_reversing")
        if mat[0, 0] <= 0.0:
            atlas_blockers.append(f"h3_chart_transition_{index}_not_future_oriented")


def _validate_candidate_metrics(
    artifact: dict[str, Any],
    *,
    radius: float,
    metric_rows: list[dict[str, Any]],
    missing: list[str],
    blockers: list[str],
) -> None:
    edges = _candidate_edges(artifact)
    if not edges:
        if not _bool(artifact.get("h3_metric_receipt"), default=False):
            missing.append("candidate_edge_h3_metric_rows_missing")
        return
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            blockers.append(f"candidate_edge_{index}_not_a_record")
            continue
        metric_name = str(edge.get("metric") or edge.get("distance_metric") or "").lower()
        if "euclidean" in metric_name or _bool(edge.get("euclidean_distance_used"), default=False):
            blockers.append(f"candidate_edge_{index}_uses_euclidean_h3_distance")
        predicted, observed = _edge_points(edge)
        if predicted is None or observed is None:
            missing.append(f"candidate_edge_{index}_hyperboloid_points_missing")
            continue
        try:
            distance = h3_distance(predicted, observed, curvature_radius=radius)
        except (TypeError, ValueError):
            blockers.append(f"candidate_edge_{index}_h3_distance_invalid")
            continue
        metric_rows.append(
            {
                "edge_index": index,
                "left": edge.get("left"),
                "right": edge.get("right"),
                "h3_geodesic_distance": distance,
            }
        )


def _candidate_edges(artifact: dict[str, Any]) -> list[Any]:
    edges = artifact.get("candidate_edges") or artifact.get("CandidateEdge") or artifact.get("edges") or []
    if isinstance(edges, dict):
        return [edges]
    if isinstance(edges, list):
        return edges
    return []


def _edge_points(edge: dict[str, Any]) -> tuple[Any, Any]:
    nested = edge.get("h3_points") or edge.get("h3Points") or {}
    predicted = (
        edge.get("predicted_h3_point")
        or edge.get("predictedH3Point")
        or edge.get("left_h3_point")
        or (nested.get("predicted") if isinstance(nested, dict) else None)
    )
    observed = (
        edge.get("observed_h3_point")
        or edge.get("observedH3Point")
        or edge.get("right_h3_point")
        or (nested.get("observed") if isinstance(nested, dict) else None)
    )
    return predicted, observed


def _finish_report(
    artifact: dict[str, Any],
    *,
    status: str,
    radius: float,
    blockers: list[str],
    missing: list[str],
    ambiguous: list[str],
    metric_rows: list[dict[str, Any]],
    gap: float | None,
) -> dict[str, Any]:
    return {
        "mode": "h3_record_worldline_stitch_certificate_v1",
        "terminal_status": status,
        "round1_terminal_status": H3_WORLDLINE_STITCH_CERTIFIED if status == H3_STITCH_CERTIFIED else status,
        "terminalStatuses": list(TERMINAL_STATUSES),
        "h3_worldline_stitch_certificate_receipt": status == H3_STITCH_CERTIFIED,
        "H3_WORLDLINE_STITCH_CERTIFICATE_RECEIPT": status == H3_STITCH_CERTIFIED,
        "H3_WORLDLINE_STITCH_CERTIFIED": status == H3_STITCH_CERTIFIED,
        "particle_matter_receipt": False,
        "physical_particle_claim": False,
        "h3_chart_model": "H3 hyperboloid in R^{1,3}",
        "minkowski_signature": "(-,+,+,+)",
        "curvature_radius": float(radius),
        "certified_assignment_gap": gap,
        "metric_rows": metric_rows,
        "missing_obligations": _unique(missing),
        "ambiguity_obligations": _unique(ambiguous),
        "blockers": _unique(blockers),
        "source_hash": (artifact.get("hashes") or {}).get("source_hash"),
        "configuration_hash": (artifact.get("hashes") or {}).get("configuration_hash"),
        "claim_boundary": (
            "Finite certificate for H3 record-token worldline stitching across real patch interfaces. "
            "It certifies continuation only when atlas, clock, descent, interface, transport, "
            "one-to-one assignment, ambiguity-gap, and refinement receipts are present. It does not "
            "promote a matter-particle claim or use record IDs as physical evidence."
        ),
    }


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out
