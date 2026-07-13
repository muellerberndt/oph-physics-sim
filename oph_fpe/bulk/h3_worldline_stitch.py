from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.evidence.hashes import stable_json_hash


H3_STITCH_CERTIFIED = "H3_STITCH_CERTIFIED"
H3_WORLDLINE_STITCH_CERTIFIED = "H3_WORLDLINE_STITCH_CERTIFIED"
H3_STITCH_AMBIGUOUS = "H3_STITCH_AMBIGUOUS"
H3_STITCH_REJECTED = "H3_STITCH_REJECTED"
H3_INTERACTION_REQUIRED = "H3_INTERACTION_REQUIRED"
H3_ATLAS_INVALID = "H3_ATLAS_INVALID"
H3_CERTIFICATE_INCOMPLETE = "H3_CERTIFICATE_INCOMPLETE"

H3_PRIMITIVE_SCHEMA = "oph_h3_worldline_stitch_primitives_v1"
H3_PRIMITIVE_SOURCE_KIND = "measured_cross_shard_interface"

_EXACT_ASSIGNMENT_REPLAY_LIMIT = 7

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
    if not np.all(np.isfinite(arr)):
        raise ValueError("H3 point coordinates must be finite")
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
    if not math.isfinite(gram):
        raise ValueError("H3 inner product must be finite")
    return float(radius * math.acosh(max(1.0, gram)))


def lorentz_matrix_residual(matrix: list[list[float]] | np.ndarray) -> float:
    mat = np.asarray(matrix, dtype=float)
    if mat.shape != (4, 4):
        raise ValueError("Lorentz matrix must have shape (4, 4)")
    if not np.all(np.isfinite(mat)):
        raise ValueError("Lorentz matrix entries must be finite")
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
    provenance = _validate_primitive_provenance(artifact, missing=missing, blockers=blockers)

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

    interaction_required = _optional_bool(
        interaction,
        ("interaction_required",),
        blockers,
        label="interaction_required",
        default=False,
    )
    if interaction_required:
        return _finish_report(
            artifact,
            status=H3_INTERACTION_REQUIRED,
            radius=radius,
            blockers=_unique(["interaction_solver_required"] + atlas_blockers + blockers + missing),
            missing=missing,
            ambiguous=ambiguous,
            metric_rows=metric_rows,
            gap=None,
            provenance=provenance,
            assignment_replay={},
        )
    _require_bool(interaction, ("free_propagation_slab", "free_slab_receipt"), missing, blockers)

    _require_bool(crossing, ("real_interface_contact", "real_boundary_contact"), missing, blockers)
    _require_bool(crossing, ("transverse", "transverse_crossing"), missing, blockers)
    if _optional_bool(
        crossing,
        ("grazing", "tangential"),
        blockers,
        label="grazing",
        default=False,
    ):
        blockers.append("tangential_or_grazing_interface_crossing")
    if _optional_bool(
        crossing,
        ("file_boundary_only", "synthetic_boundary_only"),
        blockers,
        label="file_boundary_only",
        default=False,
    ):
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
    if _optional_bool(
        assignment,
        ("uses_record_ids", "record_ids_used"),
        blockers,
        label="uses_record_ids",
        default=True,
    ):
        blockers.append("record_ids_used_in_admissibility_or_cost")
    proposed = _float(assignment.get("proposed_cost_upper", assignment.get("winner_cost_upper")), default=None)
    runner_up = _float(assignment.get("runner_up_cost_lower", assignment.get("runner_cost_lower")), default=None)
    required_gap = _float(assignment.get("required_gap", assignment.get("ambiguity_margin")), default=0.0)
    if proposed is None or runner_up is None:
        missing.append("assignment_winner_and_runner_up_cost_bounds_missing")
    if required_gap is None or required_gap < 0.0:
        blockers.append("assignment_required_gap_invalid")
        required_gap = 0.0

    assignment_replay = _validate_candidate_primitives(
        artifact,
        assignment=assignment,
        provenance=provenance,
        radius=radius,
        tolerance=tolerance,
        metric_rows=metric_rows,
        missing=missing,
        blockers=blockers,
    )
    winner_cost = _float(assignment_replay.get("winner_cost"), default=None)
    runner_cost = _float(assignment_replay.get("runner_up_cost"), default=None)
    gap = None if winner_cost is None or runner_cost is None else float(runner_cost - winner_cost)
    if winner_cost is not None and proposed is not None and proposed + tolerance < winner_cost:
        blockers.append("assignment_winner_cost_upper_below_recomputed_winner")
    if runner_cost is not None and runner_up is not None and runner_up > runner_cost + tolerance:
        blockers.append("assignment_runner_cost_lower_above_recomputed_runner_up")
    if gap is None:
        missing.append("recomputed_assignment_winner_or_runner_up_missing")
    elif gap <= tolerance:
        ambiguous.append("assignment_cost_gap_nonpositive")
    elif gap <= float(required_gap) + tolerance:
        ambiguous.append("assignment_gap_below_required_ambiguity_margin")
    declared_multiple = _optional_bool(
        assignment,
        ("multiple_optima",),
        blockers,
        label="multiple_optima",
        default=False,
    )
    recomputed_multiple = int(assignment_replay.get("equal_cost_assignment_count") or 0) > 1
    if declared_multiple != recomputed_multiple and "multiple_optima" in assignment:
        blockers.append("multiple_optima_declaration_disagrees_with_replay")
    if recomputed_multiple:
        ambiguous.append("assignment_has_multiple_equal_cost_optima")

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
    elif blockers:
        status = H3_STITCH_REJECTED
    elif ambiguous:
        status = H3_STITCH_AMBIGUOUS
    elif missing:
        status = H3_CERTIFICATE_INCOMPLETE
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
        provenance=provenance,
        assignment_replay=assignment_replay,
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


def h3_worldline_stitch_primitive_hash(artifact: dict[str, Any]) -> str:
    """Hash exactly the claim-bearing primitive fields consumed by this verifier.

    This is deliberately only a binding helper, not an H3 evidence producer.  A
    caller must still supply measured cross-shard primitives and producer
    provenance; the simulator currently has no runtime producer for them.
    """

    payload = {
        "atlas": _section(artifact, "atlas", "h3_atlas", "H3AtlasManifest"),
        "clock": _section(artifact, "clock", "clock_map", "ClockMap"),
        "extraction": _section(artifact, "extraction", "proto_record_extraction"),
        "descent": _section(artifact, "descent", "overlap_descent", "DescentCluster"),
        "crossing": _section(artifact, "crossing", "interface", "crossing_germ", "CrossingGerm"),
        "transport": _section(artifact, "transport", "sector_transport", "gauge_transport"),
        "assignment": _section(artifact, "assignment", "AssignmentCertificate"),
        "refinement": _section(artifact, "refinement", "RefinementCertificate"),
        "interaction": _section(artifact, "interaction", "InteractionEvent"),
        "candidate_edges": _candidate_edges(artifact),
    }
    edges = payload.get("candidate_edges")
    if isinstance(edges, list):
        payload["candidate_edges"] = [
            {
                key: value
                for key, value in edge.items()
                if key not in {"record_id", "right_record_id"}
            }
            if isinstance(edge, dict)
            else edge
            for edge in edges
        ]
    return stable_json_hash(payload)


def _optional_bool(
    section: dict[str, Any],
    names: tuple[str, ...],
    blockers: list[str],
    *,
    label: str,
    default: bool,
) -> bool:
    present = [name for name in names if name in section]
    if not present:
        return default
    values: list[bool] = []
    for name in present:
        value = section.get(name)
        if type(value) is not bool:
            blockers.append(f"{label}_not_exact_boolean")
            continue
        values.append(value)
    if not values:
        return default
    if len(set(values)) > 1:
        blockers.append(f"{label}_alias_values_disagree")
    return values[0]


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
    values: list[bool] = []
    for name in present:
        value = section.get(name)
        if type(value) is not bool:
            blockers.append(f"{label}_not_exact_boolean")
            continue
        values.append(value)
    if not values:
        return
    if len(set(values)) > 1:
        blockers.append(f"{label}_alias_values_disagree")
    if not all(values):
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
    if not rows:
        atlas_blockers.append("h3_atlas_points_missing")
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


def _validate_primitive_provenance(
    artifact: dict[str, Any],
    *,
    missing: list[str],
    blockers: list[str],
) -> dict[str, Any]:
    raw = artifact.get("primitive_provenance")
    if not isinstance(raw, dict):
        missing.append("h3_primitive_producer_provenance_missing")
        return {
            "verified": False,
            "source_hash": None,
            "configuration_hash": None,
            "primitive_hash": None,
        }

    if raw.get("schema") != H3_PRIMITIVE_SCHEMA:
        blockers.append("h3_primitive_provenance_schema_invalid")
    if raw.get("source_kind") != H3_PRIMITIVE_SOURCE_KIND:
        blockers.append("h3_primitive_source_kind_not_measured_cross_shard_interface")
    producer = raw.get("producer")
    producer_version = raw.get("producer_version")
    if not isinstance(producer, str) or not producer.strip():
        missing.append("h3_primitive_producer_name_missing")
    if not isinstance(producer_version, str) or not producer_version.strip():
        missing.append("h3_primitive_producer_version_missing")

    source_manifest = raw.get("source_manifest")
    configuration = raw.get("configuration")
    if not isinstance(source_manifest, dict) or not source_manifest:
        missing.append("h3_primitive_source_manifest_missing")
        source_manifest = None
    if not isinstance(configuration, dict) or not configuration:
        missing.append("h3_primitive_configuration_missing")
        configuration = None

    source_hash = stable_json_hash(source_manifest) if source_manifest is not None else None
    configuration_hash = stable_json_hash(configuration) if configuration is not None else None
    primitive_hash = h3_worldline_stitch_primitive_hash(artifact)
    _compare_bound_hash(
        raw.get("source_hash"),
        source_hash,
        "h3_primitive_source_hash",
        missing=missing,
        blockers=blockers,
    )
    _compare_bound_hash(
        raw.get("configuration_hash"),
        configuration_hash,
        "h3_primitive_configuration_hash",
        missing=missing,
        blockers=blockers,
    )
    _compare_bound_hash(
        raw.get("primitive_hash"),
        primitive_hash,
        "h3_primitive_payload_hash",
        missing=missing,
        blockers=blockers,
    )

    legacy_hashes = artifact.get("hashes")
    if legacy_hashes is not None:
        if not isinstance(legacy_hashes, dict):
            blockers.append("legacy_h3_hashes_not_a_record")
        else:
            if legacy_hashes.get("source_hash") != source_hash:
                blockers.append("legacy_h3_source_hash_not_bound_to_source_manifest")
            if legacy_hashes.get("configuration_hash") != configuration_hash:
                blockers.append("legacy_h3_configuration_hash_not_bound_to_configuration")

    verified = not any(
        item.startswith("h3_primitive_") or item.startswith("legacy_h3_")
        for item in blockers + missing
    )
    return {
        "verified": verified,
        "schema": raw.get("schema"),
        "source_kind": raw.get("source_kind"),
        "producer": producer,
        "producer_version": producer_version,
        "source_manifest": source_manifest,
        "source_hash": source_hash,
        "configuration_hash": configuration_hash,
        "primitive_hash": primitive_hash,
    }


def _compare_bound_hash(
    claimed: Any,
    computed: str | None,
    label: str,
    *,
    missing: list[str],
    blockers: list[str],
) -> None:
    if claimed is None:
        missing.append(f"{label}_missing")
    elif not isinstance(claimed, str) or not _is_sha256_receipt(claimed):
        blockers.append(f"{label}_malformed")
    elif computed is None or claimed != computed:
        blockers.append(f"{label}_mismatch")


def _is_sha256_receipt(value: str) -> bool:
    if not value.startswith("sha256:") or len(value) != 71:
        return False
    try:
        int(value[7:], 16)
    except ValueError:
        return False
    return True


def _validate_candidate_primitives(
    artifact: dict[str, Any],
    *,
    assignment: dict[str, Any],
    provenance: dict[str, Any],
    radius: float,
    tolerance: float,
    metric_rows: list[dict[str, Any]],
    missing: list[str],
    blockers: list[str],
) -> dict[str, Any]:
    edges = _candidate_edges(artifact)
    if not edges:
        missing.append("candidate_edge_h3_metric_primitives_missing")
        return {}

    edge_map: dict[tuple[str, str], float] = {}
    left_points: dict[str, np.ndarray] = {}
    right_points: dict[str, np.ndarray] = {}
    left_shards: set[str] = set()
    right_shards: set[str] = set()
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            blockers.append(f"candidate_edge_{index}_not_a_record")
            continue
        left = edge.get("left")
        right = edge.get("right")
        if not isinstance(left, (str, int)) or isinstance(left, bool) or str(left) == "":
            blockers.append(f"candidate_edge_{index}_left_token_invalid")
            continue
        if not isinstance(right, (str, int)) or isinstance(right, bool) or str(right) == "":
            blockers.append(f"candidate_edge_{index}_right_token_invalid")
            continue
        left_key = str(left)
        right_key = str(right)
        pair = (left_key, right_key)
        if pair in edge_map:
            blockers.append(f"candidate_edge_{index}_duplicate_pair")
            continue
        metric_name = str(edge.get("metric") or edge.get("distance_metric") or "").lower()
        euclidean_declared = _optional_bool(
            edge,
            ("euclidean_distance_used",),
            blockers,
            label=f"candidate_edge_{index}_euclidean_distance_used",
            default=False,
        )
        if "euclidean" in metric_name or euclidean_declared:
            blockers.append(f"candidate_edge_{index}_uses_euclidean_h3_distance")
        if "h3" not in metric_name or not any(token in metric_name for token in ("geodesic", "hyperboloid")):
            blockers.append(f"candidate_edge_{index}_metric_not_h3_geodesic")
        predicted, observed = _edge_points(edge)
        if predicted is None or observed is None:
            missing.append(f"candidate_edge_{index}_hyperboloid_points_missing")
            continue
        try:
            predicted_point = h3_point_from_spatial_point(predicted, curvature_radius=radius)
            observed_point = h3_point_from_spatial_point(observed, curvature_radius=radius)
            distance = h3_distance(predicted, observed, curvature_radius=radius)
        except (TypeError, ValueError):
            blockers.append(f"candidate_edge_{index}_h3_distance_invalid")
            continue
        if left_key in left_points and not np.allclose(
            left_points[left_key], predicted_point, rtol=0.0, atol=tolerance
        ):
            blockers.append(f"candidate_edge_{index}_left_point_inconsistent_across_graph")
        else:
            left_points[left_key] = predicted_point
        if right_key in right_points and not np.allclose(
            right_points[right_key], observed_point, rtol=0.0, atol=tolerance
        ):
            blockers.append(f"candidate_edge_{index}_right_point_inconsistent_across_graph")
        else:
            right_points[right_key] = observed_point

        claimed_cost = _float(edge.get("h3_geodesic_cost", edge.get("cost")), default=None)
        if claimed_cost is not None and abs(claimed_cost - distance) > tolerance:
            blockers.append(f"candidate_edge_{index}_claimed_cost_disagrees_with_h3_replay")
        edge_map[pair] = distance
        left_shard = edge.get("left_shard_id")
        right_shard = edge.get("right_shard_id")
        if not isinstance(left_shard, str) or not left_shard:
            missing.append(f"candidate_edge_{index}_left_shard_id_missing")
        else:
            left_shards.add(left_shard)
        if not isinstance(right_shard, str) or not right_shard:
            missing.append(f"candidate_edge_{index}_right_shard_id_missing")
        else:
            right_shards.add(right_shard)
        metric_rows.append(
            {
                "edge_index": index,
                "left": left,
                "right": right,
                "h3_geodesic_distance": distance,
            }
        )

    left_ids = sorted(left_points)
    right_ids = sorted(right_points)
    expected_pairs = {(left, right) for left in left_ids for right in right_ids}
    complete_graph = bool(left_ids and right_ids and set(edge_map) == expected_pairs)
    if not complete_graph:
        blockers.append("candidate_graph_not_complete_bipartite_from_primitives")
    if len(left_shards) != 1 or len(right_shards) != 1 or left_shards == right_shards:
        blockers.append("candidate_edges_do_not_cross_two_distinct_shards")
    _validate_source_shards(provenance, left_shards, right_shards, blockers=blockers, missing=missing)

    if max(len(left_ids), len(right_ids)) > _EXACT_ASSIGNMENT_REPLAY_LIMIT:
        missing.append("candidate_assignment_exceeds_exact_replay_limit")
        return {"complete_graph": complete_graph}
    appearance = _penalty_map(
        assignment.get("appearance_penalties"),
        expected_ids=right_ids,
        token_key="right",
        label="appearance_penalties",
        missing=missing,
        blockers=blockers,
    )
    disappearance = _penalty_map(
        assignment.get("disappearance_penalties"),
        expected_ids=left_ids,
        token_key="left",
        label="disappearance_penalties",
        missing=missing,
        blockers=blockers,
    )
    selected_pairs = _selected_pair_set(assignment, missing=missing, blockers=blockers)
    if not complete_graph or appearance is None or disappearance is None or selected_pairs is None:
        return {"complete_graph": complete_graph}

    assignments = _enumerate_partial_assignments(
        left_ids,
        right_ids,
        edge_costs=edge_map,
        appearance=appearance,
        disappearance=disappearance,
    )
    if len(assignments) < 2:
        missing.append("recomputed_assignment_runner_up_missing")
        return {"complete_graph": complete_graph}
    assignments.sort(key=lambda row: (row[0], row[1]))
    winner_cost, winner_pairs = assignments[0]
    runner_cost = assignments[1][0]
    equal_count = sum(1 for cost, _ in assignments if abs(cost - winner_cost) <= tolerance)
    selected_one_to_one = _pairs_are_one_to_one(selected_pairs)
    if not selected_one_to_one:
        blockers.append("selected_assignment_not_one_to_one_from_primitives")
    if selected_pairs != set(winner_pairs):
        blockers.append("selected_assignment_disagrees_with_recomputed_winner")
    return {
        "complete_graph": complete_graph,
        "one_to_one": selected_one_to_one,
        "record_id_independence_recomputed": True,
        "winner_cost": float(winner_cost),
        "runner_up_cost": float(runner_cost),
        "winner_pairs": [list(pair) for pair in winner_pairs],
        "equal_cost_assignment_count": int(equal_count),
        "enumerated_assignment_count": len(assignments),
    }


def _validate_source_shards(
    provenance: dict[str, Any],
    left_shards: set[str],
    right_shards: set[str],
    *,
    blockers: list[str],
    missing: list[str],
) -> None:
    source = provenance.get("source_manifest")
    if not isinstance(source, dict):
        return
    source_left = source.get("left_shard_id")
    source_right = source.get("right_shard_id")
    if not isinstance(source_left, str) or not isinstance(source_right, str):
        missing.append("h3_source_manifest_shard_ids_missing")
        return
    if source_left == source_right:
        blockers.append("h3_source_manifest_shards_not_distinct")
    if left_shards and left_shards != {source_left}:
        blockers.append("candidate_left_shard_disagrees_with_source_manifest")
    if right_shards and right_shards != {source_right}:
        blockers.append("candidate_right_shard_disagrees_with_source_manifest")


def _penalty_map(
    rows: Any,
    *,
    expected_ids: list[str],
    token_key: str,
    label: str,
    missing: list[str],
    blockers: list[str],
) -> dict[str, float] | None:
    if not isinstance(rows, list) or not rows:
        missing.append(f"{label}_primitive_rows_missing")
        return None
    out: dict[str, float] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            blockers.append(f"{label}_{index}_not_a_record")
            continue
        token = row.get(token_key)
        cost = _float(row.get("cost"), default=None)
        key = str(token) if isinstance(token, (str, int)) and not isinstance(token, bool) else ""
        if not key or cost is None or cost < 0.0:
            blockers.append(f"{label}_{index}_invalid")
            continue
        if key in out:
            blockers.append(f"{label}_{index}_duplicate_token")
            continue
        out[key] = float(cost)
    if set(out) != set(expected_ids):
        blockers.append(f"{label}_do_not_cover_candidate_tokens")
        return None
    return out


def _selected_pair_set(
    assignment: dict[str, Any],
    *,
    missing: list[str],
    blockers: list[str],
) -> set[tuple[str, str]] | None:
    rows = assignment.get("selected_pairs")
    if not isinstance(rows, list) or not rows:
        missing.append("selected_assignment_pair_primitives_missing")
        return None
    out: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            blockers.append(f"selected_assignment_pair_{index}_not_a_record")
            continue
        left = row.get("left")
        right = row.get("right")
        if not isinstance(left, (str, int)) or isinstance(left, bool):
            blockers.append(f"selected_assignment_pair_{index}_left_invalid")
            continue
        if not isinstance(right, (str, int)) or isinstance(right, bool):
            blockers.append(f"selected_assignment_pair_{index}_right_invalid")
            continue
        pair = (str(left), str(right))
        if pair in out:
            blockers.append(f"selected_assignment_pair_{index}_duplicate")
        out.add(pair)
    return out


def _pairs_are_one_to_one(pairs: set[tuple[str, str]]) -> bool:
    return len({left for left, _ in pairs}) == len(pairs) and len({right for _, right in pairs}) == len(pairs)


def _enumerate_partial_assignments(
    left_ids: list[str],
    right_ids: list[str],
    *,
    edge_costs: dict[tuple[str, str], float],
    appearance: dict[str, float],
    disappearance: dict[str, float],
) -> list[tuple[float, tuple[tuple[str, str], ...]]]:
    out: list[tuple[float, tuple[tuple[str, str], ...]]] = []

    def visit(index: int, used_right: set[str], pairs: list[tuple[str, str]], cost: float) -> None:
        if index == len(left_ids):
            total = cost + sum(appearance[right] for right in right_ids if right not in used_right)
            out.append((float(total), tuple(sorted(pairs))))
            return
        left = left_ids[index]
        visit(index + 1, used_right, pairs, cost + disappearance[left])
        for right in right_ids:
            if right in used_right:
                continue
            used_right.add(right)
            pairs.append((left, right))
            visit(index + 1, used_right, pairs, cost + edge_costs[(left, right)])
            pairs.pop()
            used_right.remove(right)

    visit(0, set(), [], 0.0)
    return out


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
    provenance: dict[str, Any],
    assignment_replay: dict[str, Any],
) -> dict[str, Any]:
    return {
        "mode": "h3_record_worldline_stitch_certificate_v2",
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
        "assignment_replay": assignment_replay,
        "primitive_provenance_verified": bool(provenance.get("verified", False)),
        "primitive_schema": provenance.get("schema"),
        "primitive_source_kind": provenance.get("source_kind"),
        "primitive_producer": provenance.get("producer"),
        "primitive_producer_version": provenance.get("producer_version"),
        "primitive_payload_hash": provenance.get("primitive_hash"),
        "missing_obligations": _unique(missing),
        "ambiguity_obligations": _unique(ambiguous),
        "blockers": _unique(blockers),
        "source_hash": provenance.get("source_hash"),
        "configuration_hash": provenance.get("configuration_hash"),
        "claim_boundary": (
            "Finite certificate for H3 record-token worldline stitching across real patch interfaces. "
            "It certifies continuation only from hash-bound measured cross-shard primitives, with "
            "atlas, clock, descent, interface, transport, candidate-graph, exact finite assignment, "
            "ambiguity-gap, and refinement checks replayed where possible. No runtime producer is "
            "currently wired. It does not promote a matter-particle claim or use record IDs as "
            "physical evidence."
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
