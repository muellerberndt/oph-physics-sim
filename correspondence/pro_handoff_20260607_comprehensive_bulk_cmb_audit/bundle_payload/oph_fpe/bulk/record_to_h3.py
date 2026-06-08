from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from oph_fpe.bulk.cap_geometry import RoundCap, cap_weights, lambda_cap
from oph_fpe.bulk.cap_normals import cap_normals
from oph_fpe.bulk.h3_chart import (
    h3_distance_matrix,
    h3_halfspace_profile,
    h3_point_from_tangent,
    h3_tangent_from_point,
    random_h3_points,
)
from oph_fpe.bulk.observer_reconstruction import neutral_dimension_report_from_distance
from oph_fpe.claims import H3_RESPONSE_CONTROL_SEPARATION_RECEIPT


DEFAULT_RECORD_FIELDS = (
    "record_signature",
    "stable_count",
    "repair_load",
    "cumulative_repair_load",
    "s3_class_density",
    "s3_sector_class",
)


def _h3_response_strict_receipt(h3_report: dict[str, Any]) -> bool:
    return bool(h3_report.get("MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT", False))


def _h3_response_control_separation_receipt(h3_report: dict[str, Any]) -> bool:
    if _h3_response_strict_receipt(h3_report):
        return True
    if h3_report.get(H3_RESPONSE_CONTROL_SEPARATION_RECEIPT, False):
        return True
    if h3_report.get("h3_control_separation_receipt", False):
        return True
    stage_gates = h3_report.get("h3_response_stage_gates", {})
    if not isinstance(stage_gates, dict):
        return False
    return bool(
        stage_gates.get(H3_RESPONSE_CONTROL_SEPARATION_RECEIPT, False)
        or stage_gates.get("intermediate_control_separation_receipt", False)
        or (
            stage_gates.get("signal_gate", False)
            and stage_gates.get("geometry_gate", False)
            and stage_gates.get("aggregate_wrong_scale_gate", False)
        )
    )


def record_cap_response_matrix(
    points: np.ndarray,
    caps: list[RoundCap],
    raw_fields: dict[str, np.ndarray],
    observer_views: list[dict[str, Any]],
    *,
    cell_entropy: np.ndarray | None = None,
    field_names: list[str] | tuple[str, ...] = DEFAULT_RECORD_FIELDS,
    geometry_blend: float = 0.0,
    response_mode: str = "field_summary_similarity",
    transport_time: float = 0.1,
    transport_scale: float = 2.0 * np.pi,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Build observer-visible cap-response profiles from record fields.

    The response is not a host-coordinate embedding. It compares each local
    observer support's visible record-field summary with each cap observer's
    visible record-field summary. A configurable geometry blend is exposed for
    diagnostics but defaults to zero so the fit is not just S2 membership.
    """

    patch_views = [view for view in observer_views if view.get("view_type") == "patch_observer"]
    if not patch_views or not caps:
        return np.zeros((0, 0), dtype=float), patch_views, {"field_names": [], "geometry_blend": float(geometry_blend)}
    weights = np.asarray(cell_entropy, dtype=float) if cell_entropy is not None else np.ones(points.shape[0], dtype=float)
    fields = _standardized_fields(raw_fields, field_names)
    if not fields:
        return np.zeros((len(patch_views), len(caps)), dtype=float), patch_views, {
            "field_names": [],
            "geometry_blend": float(geometry_blend),
            "response_mode": str(response_mode),
        }
    mode = str(response_mode)
    if mode == "cap_transport_similarity":
        response = _cap_transport_response(
            points,
            caps,
            fields,
            patch_views,
            weights,
            transport_time=float(transport_time),
            transport_scale=float(transport_scale),
        )
        return np.clip(response, 0.0, 1.0), patch_views, {
            "field_names": list(fields.keys()),
            "geometry_blend": 0.0,
            "response_mode": mode,
            "transport_time": float(transport_time),
            "transport_scale": float(transport_scale),
            "response_source": "observer_support_record_similarity_under_lambda_C_transport",
            "claim_boundary": (
                "observer-visible cap response: compare local support records before and after "
                "finite lambda_C(2*pi*t) transport; this is downstream of the BW/KMS receipt and "
                "does not by itself establish a populated 3D bulk"
            ),
        }
    cap_features = np.vstack([_cap_feature(points, cap, fields, weights) for cap in caps])
    observer_features = np.vstack([_observer_feature(view, fields, weights) for view in patch_views])
    distances = np.linalg.norm(observer_features[:, None, :] - cap_features[None, :, :], axis=2)
    scale = max(float(np.sqrt(len(fields))), 1e-12)
    record_response = np.exp(-distances / scale)
    blend = float(np.clip(geometry_blend, 0.0, 1.0))
    if blend > 0.0:
        boundary = s2_boundary_profiles(np.vstack([np.asarray(view.get("axis"), dtype=float) for view in patch_views]), caps)
        record_response = (1.0 - blend) * record_response + blend * boundary
    return np.clip(record_response, 0.0, 1.0), patch_views, {
        "field_names": list(fields.keys()),
        "geometry_blend": blend,
        "response_mode": mode,
        "response_source": "observer_support_record_field_similarity_to_cap_record_field_summary",
    }


def fit_response_profiles_to_h3(
    response: np.ndarray,
    caps: list[RoundCap],
    *,
    candidate_count: int = 2048,
    candidate_radius: float = 2.0,
    softness: float = 0.25,
    seed: int = 1,
    refine: bool = True,
    max_refine_rows: int = 128,
    refine_max_nfev: int = 48,
) -> dict[str, Any]:
    response = np.asarray(response, dtype=float)
    if response.size == 0 or not caps:
        return _empty_fit("empty_response")
    normals = cap_normals(caps)
    candidates = random_h3_points(int(candidate_count), seed=seed, radius=float(candidate_radius))
    candidate_profiles = h3_halfspace_profile(candidates, normals, softness=float(softness))
    residuals = _profile_residual_matrix(response, candidate_profiles)
    best_indices = np.argmin(residuals, axis=1)
    best_residuals = residuals[np.arange(response.shape[0]), best_indices]
    fitted = candidates[best_indices]
    refinement = {"enabled": False, "refined_rows": 0, "median_improvement": 0.0}
    if refine:
        fitted, best_residuals, refinement = _refine_h3_fit(
            response,
            normals,
            fitted,
            best_residuals,
            softness=float(softness),
            max_rows=int(max_refine_rows),
            max_nfev=int(refine_max_nfev),
        )
    return {
        "mode": "record_response_to_h3_fit",
        "candidate_count": int(candidate_count),
        "candidate_radius": float(candidate_radius),
        "softness": float(softness),
        "local_refinement": refinement,
        "median_residual": float(np.median(best_residuals)) if best_residuals.size else None,
        "mean_residual": float(np.mean(best_residuals)) if best_residuals.size else None,
        "p90_residual": float(np.percentile(best_residuals, 90)) if best_residuals.size else None,
        "best_candidate_indices": [int(value) for value in best_indices[:256]],
        "sample_fitted_h3_points": [[float(x) for x in row] for row in fitted[:32]],
        "fitted_h3_points": [[float(x) for x in row] for row in fitted],
    }


def observer_chart_object_population_report(
    observer_views: list[dict[str, Any]],
    record_families: list[dict[str, Any]],
    h3_report: dict[str, Any],
    *,
    seed: int = 1,
    min_objects: int = 8,
    min_observers_per_object: int = 2,
    pass_ratio: float = 0.85,
    max_objects: int = 2048,
    incidence_mode: str = "transition_history",
    min_packet_mass: float = 0.05,
    min_transition_affinity: float = 0.25,
    transition_affinity_score: str = "geometric_mean",
    observer_cluster_fields: tuple[str, ...] | list[str] = ("record_family", "s3_sector_class", "repair_load_bucket"),
    observer_cluster_top_k: int = 2,
    min_observer_cluster_weight: float = 0.05,
    history_window: int = 4,
    min_persistence: int = 3,
    max_observer_fraction_per_object: float = 0.65,
    max_h3_compactness: float = 0.35,
    min_localized_objects: int = 2,
    shuffle_control_count: int = 1,
    split_h3_components: bool = False,
    component_link_fraction: float = 0.35,
    component_min_observers: int | None = None,
    require_support_visibility: bool = False,
    min_support_visibility: float = 0.0,
    visibility_mode: str = "support_overlap",
    packet_visibility_weight: float = 0.5,
    boundary_gate_mode: str = "nonboundary",
) -> dict[str, Any]:
    """Populate the modular-response H3 observer chart with observer objects.

    This path deliberately avoids raw screen support cap profiles as evidence.
    Objects are placed from the sampled observers that can see the object's
    support and visible packet. The receipt is only a populated-chart diagnostic
    and requires compactness against shuffled observer-object incidence; the
    modular-response chart receipt itself remains a separate gate.
    """

    patch_views = [view for view in observer_views if view.get("view_type") == "patch_observer"]
    chart = _observer_h3_chart(h3_report)
    cluster_mode = str(incidence_mode) in {
        "observer_transition_cluster",
        "observer_visible_transition_cluster",
        "observer_readout_cluster",
        "observer_transition_mixture_cluster",
        "transition_history_mixture_cluster",
        "transition_history",
        "record_family_modular_response_mixture",
        "transition_affinity_modular_response_mixture",
    }
    if not patch_views or (not record_families and not cluster_mode) or chart["points"].size == 0:
        return {
            "mode": "observer_chart_object_h3_population",
            "observer_chart_object_h3_receipt": False,
            "observer_chart_bulk_population_receipt": False,
            "reason": "empty_observers_objects_or_h3_chart",
            "claim_boundary": "empty observer-chart object population diagnostic; no bulk claim",
        }
    observer_points: list[np.ndarray] = []
    observer_axes: list[np.ndarray] = []
    chart_observer_ids: list[int] = []
    chart_views: list[dict[str, Any]] = []
    for view in patch_views:
        observer_id = int(view.get("observer_id", len(chart_views)))
        point = chart["by_id"].get(observer_id)
        if point is None:
            continue
        axis = np.asarray(view.get("axis", [0.0, 0.0, 1.0]), dtype=float)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-12:
            axis = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            axis = axis / axis_norm
        observer_points.append(point)
        observer_axes.append(axis)
        chart_observer_ids.append(observer_id)
        chart_views.append(view)
    if not observer_points:
        return {
            "mode": "observer_chart_object_h3_population",
            "observer_chart_object_h3_receipt": False,
            "observer_chart_bulk_population_receipt": False,
            "reason": "no_patch_observers_matched_h3_chart_ids",
            "claim_boundary": "H3 chart has no matching observer ids; no bulk claim",
        }
    h3_points = np.vstack(observer_points)
    axes = np.vstack(observer_axes)
    global_h3_scale = _median_pairwise_h3_distance(h3_points)
    global_s2_scale = _median_pairwise_s2_distance(axes)
    max_observer_fraction = float(np.clip(max_observer_fraction_per_object, 0.0, 1.0))
    if str(incidence_mode) in {
        "record_family_modular_response_mixture",
        "transition_affinity_modular_response_mixture",
    }:
        object_rows = _record_family_modular_response_mixture_rows(
            chart_views,
            chart_observer_ids,
            h3_points,
            axes,
            record_families[: int(max_objects)],
            min_observers_per_object=int(min_observers_per_object),
            max_objects=int(max_objects),
            max_observer_fraction=max_observer_fraction,
            min_transition_affinity=float(min_transition_affinity),
            transition_affinity_score=str(transition_affinity_score),
            observer_cluster_fields=tuple(str(field) for field in observer_cluster_fields),
            top_k=int(observer_cluster_top_k),
            min_weight=float(min_observer_cluster_weight),
            split_h3_components=bool(split_h3_components),
            component_link_fraction=float(component_link_fraction),
            component_min_observers=component_min_observers,
            require_support_visibility=bool(require_support_visibility),
            min_support_visibility=float(min_support_visibility),
            visibility_mode=str(visibility_mode),
            packet_visibility_weight=float(packet_visibility_weight),
        )
    elif str(incidence_mode) in {"observer_transition_mixture_cluster", "transition_history_mixture_cluster"}:
        object_rows = _observer_transition_mixture_cluster_rows(
            chart_views,
            chart_observer_ids,
            h3_points,
            axes,
            min_observers_per_object=int(min_observers_per_object),
            max_objects=int(max_objects),
            observer_cluster_fields=tuple(str(field) for field in observer_cluster_fields),
            max_observer_fraction=max_observer_fraction,
            transition_affinity_score=str(transition_affinity_score),
            top_k=int(observer_cluster_top_k),
            min_weight=float(min_observer_cluster_weight),
            split_h3_components=bool(split_h3_components),
            component_link_fraction=float(component_link_fraction),
            component_min_observers=component_min_observers,
        )
    elif str(incidence_mode) == "transition_history":
        object_rows = _observer_transition_history_cluster_rows(
            chart_views,
            chart_observer_ids,
            h3_points,
            axes,
            min_observers_per_object=int(min_observers_per_object),
            max_objects=int(max_objects),
            max_observer_fraction=max_observer_fraction,
            history_window=int(history_window),
            min_persistence=int(min_persistence),
            split_h3_components=bool(split_h3_components),
            component_link_fraction=float(component_link_fraction),
            component_min_observers=component_min_observers,
        )
    elif cluster_mode:
        object_rows = _observer_transition_cluster_rows(
            chart_views,
            chart_observer_ids,
            h3_points,
            axes,
            min_observers_per_object=int(min_observers_per_object),
            max_objects=int(max_objects),
            observer_cluster_fields=tuple(str(field) for field in observer_cluster_fields),
            max_observer_fraction=max_observer_fraction,
            transition_affinity_score=str(transition_affinity_score),
        )
    else:
        object_rows = _record_family_object_chart_rows(
            chart_views,
            chart_observer_ids,
            h3_points,
            axes,
            record_families[: int(max_objects)],
            min_observers_per_object=int(min_observers_per_object),
            incidence_mode=str(incidence_mode),
            min_packet_mass=float(min_packet_mass),
            min_transition_affinity=float(min_transition_affinity),
            transition_affinity_score=str(transition_affinity_score),
        )
    if not object_rows:
        return {
            "mode": "observer_chart_object_h3_population",
            "observer_count": int(h3_points.shape[0]),
            "object_count": 0,
            "observer_chart_object_h3_receipt": False,
            "observer_chart_bulk_population_receipt": False,
            "reason": "no_objects_seen_by_enough_chart_observers",
            "min_observers_per_object": int(min_observers_per_object),
            "claim_boundary": "no observer-visible objects populated the modular-response H3 chart",
        }
    rng = np.random.default_rng(seed)
    shuffle_count = max(1, int(shuffle_control_count))
    shuffled_rows: list[dict[str, Any]] = []
    shuffled_medians: list[float] = []
    shuffled_localized_counts: list[int] = []
    shuffled_localized_not_boundary_counts: list[int] = []
    for _shuffle_index in range(shuffle_count):
        shuffled_points = h3_points[rng.permutation(h3_points.shape[0])]
        if str(incidence_mode) in {
            "record_family_modular_response_mixture",
            "transition_affinity_modular_response_mixture",
        }:
            current_shuffled_rows = _record_family_modular_response_mixture_rows(
                chart_views,
                chart_observer_ids,
                shuffled_points,
                axes,
                record_families[: int(max_objects)],
                min_observers_per_object=int(min_observers_per_object),
                max_objects=int(max_objects),
                max_observer_fraction=max_observer_fraction,
                min_transition_affinity=float(min_transition_affinity),
                transition_affinity_score=str(transition_affinity_score),
                observer_cluster_fields=tuple(str(field) for field in observer_cluster_fields),
                top_k=int(observer_cluster_top_k),
                min_weight=float(min_observer_cluster_weight),
                split_h3_components=bool(split_h3_components),
                component_link_fraction=float(component_link_fraction),
                component_min_observers=component_min_observers,
                require_support_visibility=bool(require_support_visibility),
                min_support_visibility=float(min_support_visibility),
                visibility_mode=str(visibility_mode),
                packet_visibility_weight=float(packet_visibility_weight),
            )
        elif str(incidence_mode) in {"observer_transition_mixture_cluster", "transition_history_mixture_cluster"}:
            current_shuffled_rows = _observer_transition_mixture_cluster_rows(
                chart_views,
                chart_observer_ids,
                shuffled_points,
                axes,
                min_observers_per_object=int(min_observers_per_object),
                max_objects=int(max_objects),
                observer_cluster_fields=tuple(str(field) for field in observer_cluster_fields),
                max_observer_fraction=max_observer_fraction,
                transition_affinity_score=str(transition_affinity_score),
                top_k=int(observer_cluster_top_k),
                min_weight=float(min_observer_cluster_weight),
                split_h3_components=bool(split_h3_components),
                component_link_fraction=float(component_link_fraction),
                component_min_observers=component_min_observers,
            )
        elif str(incidence_mode) == "transition_history":
            current_shuffled_rows = _observer_transition_history_cluster_rows(
                chart_views,
                chart_observer_ids,
                shuffled_points,
                axes,
                min_observers_per_object=int(min_observers_per_object),
                max_objects=int(max_objects),
                max_observer_fraction=max_observer_fraction,
                history_window=int(history_window),
                min_persistence=int(min_persistence),
                split_h3_components=bool(split_h3_components),
                component_link_fraction=float(component_link_fraction),
                component_min_observers=component_min_observers,
            )
        elif cluster_mode:
            current_shuffled_rows = _observer_transition_cluster_rows(
                chart_views,
                chart_observer_ids,
                shuffled_points,
                axes,
                min_observers_per_object=int(min_observers_per_object),
                max_objects=int(max_objects),
                observer_cluster_fields=tuple(str(field) for field in observer_cluster_fields),
                max_observer_fraction=max_observer_fraction,
                transition_affinity_score=str(transition_affinity_score),
            )
        else:
            current_shuffled_rows = _object_chart_compactness_rows(
                chart_views,
                chart_observer_ids,
                shuffled_points,
                axes,
                record_families[: int(max_objects)],
                min_observers_per_object=int(min_observers_per_object),
                incidence_mode=str(incidence_mode),
                min_packet_mass=float(min_packet_mass),
                min_transition_affinity=float(min_transition_affinity),
                transition_affinity_score=str(transition_affinity_score),
            )
        shuffled_rows.extend(current_shuffled_rows)
        current_norms = np.asarray([row["h3_compactness_normalized"] for row in current_shuffled_rows], dtype=float)
        if current_norms.size:
            shuffled_medians.append(float(np.median(current_norms)))
            shuffled_localized_counts.append(int(np.sum(current_norms <= float(max_h3_compactness))))
            current_s2_norms = np.asarray(
                [row["s2_boundary_compactness_normalized"] for row in current_shuffled_rows],
                dtype=float,
            )
            if current_s2_norms.size == current_norms.size:
                current_not_boundary = current_norms <= current_s2_norms / max(float(pass_ratio), 1e-12)
                shuffled_localized_not_boundary_counts.append(
                    int(np.sum((current_norms <= float(max_h3_compactness)) & current_not_boundary))
                )
    h3_norms = np.asarray([row["h3_compactness_normalized"] for row in object_rows], dtype=float)
    s2_norms = np.asarray([row["s2_boundary_compactness_normalized"] for row in object_rows], dtype=float)
    shuffled_norms = np.asarray(
        [row["h3_compactness_normalized"] for row in shuffled_rows],
        dtype=float,
    )
    median_h3 = float(np.median(h3_norms)) if h3_norms.size else float("nan")
    median_s2 = float(np.median(s2_norms)) if s2_norms.size else float("nan")
    median_shuffled = float(np.median(shuffled_medians)) if shuffled_medians else (
        float(np.median(shuffled_norms)) if shuffled_norms.size else float("nan")
    )
    p10_shuffled = float(np.percentile(shuffled_medians, 10)) if shuffled_medians else None
    p90_shuffled = float(np.percentile(shuffled_medians, 90)) if shuffled_medians else None
    h3_beats_shuffle = bool(np.isfinite(median_h3) and np.isfinite(median_shuffled) and median_h3 < float(pass_ratio) * median_shuffled)
    h3_beats_shuffle_robust = bool(
        np.isfinite(median_h3)
        and (
            (
                p10_shuffled is not None
                and np.isfinite(float(p10_shuffled))
                and median_h3 < float(pass_ratio) * float(p10_shuffled)
            )
            or (
                p10_shuffled is None
                and np.isfinite(median_shuffled)
                and median_h3 < float(pass_ratio) * median_shuffled
            )
        )
    )
    h3_not_boundary_dominated = bool(np.isfinite(median_h3) and np.isfinite(median_s2) and median_h3 <= median_s2 / max(float(pass_ratio), 1e-12))
    h3_localized = bool(np.isfinite(median_h3) and median_h3 <= float(max_h3_compactness))
    localized_mask = h3_norms <= float(max_h3_compactness)
    not_boundary_mask = h3_norms <= s2_norms / max(float(pass_ratio), 1e-12)
    localized_not_boundary_count = int(np.sum(localized_mask & not_boundary_mask)) if h3_norms.size and s2_norms.size else 0
    localized_count = int(np.sum(localized_mask)) if h3_norms.size else 0
    shuffled_localized_count = int(round(float(np.median(shuffled_localized_counts)))) if shuffled_localized_counts else (
        int(np.sum(shuffled_norms <= float(max_h3_compactness))) if shuffled_norms.size else 0
    )
    shuffled_localized_p90 = float(np.percentile(shuffled_localized_counts, 90)) if shuffled_localized_counts else None
    shuffled_localized_not_boundary_count = (
        int(round(float(np.median(shuffled_localized_not_boundary_counts))))
        if shuffled_localized_not_boundary_counts
        else 0
    )
    shuffled_localized_not_boundary_p90 = (
        float(np.percentile(shuffled_localized_not_boundary_counts, 90))
        if shuffled_localized_not_boundary_counts
        else None
    )
    h3_strict_receipt = _h3_response_strict_receipt(h3_report)
    h3_control_separation_receipt = _h3_response_control_separation_receipt(h3_report)
    localized_nonboundary_subpopulation_median_receipt = bool(
        localized_not_boundary_count >= int(min_localized_objects)
        and localized_not_boundary_count > shuffled_localized_not_boundary_count
        and h3_control_separation_receipt
    )
    localized_nonboundary_subpopulation_receipt = bool(
        localized_not_boundary_count >= int(min_localized_objects)
        and (
            (
                shuffled_localized_not_boundary_p90 is not None
                and localized_not_boundary_count > float(shuffled_localized_not_boundary_p90)
            )
            or (
                shuffled_localized_not_boundary_p90 is None
                and localized_not_boundary_count > shuffled_localized_not_boundary_count
            )
        )
        and h3_control_separation_receipt
    )
    localized_h3_subpopulation_median_receipt = bool(
        localized_count >= int(min_localized_objects)
        and localized_count > shuffled_localized_count
        and h3_control_separation_receipt
    )
    localized_h3_subpopulation_receipt = bool(
        localized_count >= int(min_localized_objects)
        and (
            (
                shuffled_localized_p90 is not None
                and localized_count > float(shuffled_localized_p90)
            )
            or (
                shuffled_localized_p90 is None
                and localized_count > shuffled_localized_count
            )
        )
        and h3_control_separation_receipt
    )
    eligible = bool(len(object_rows) >= int(min_objects))
    chart_median_receipt = bool(eligible and h3_beats_shuffle)
    chart_receipt = bool(eligible and h3_beats_shuffle_robust)
    localized_nonboundary_bulk_population_receipt = bool(
        eligible
        and localized_nonboundary_subpopulation_receipt
        and chart_receipt
        and h3_strict_receipt
    )
    localized_h3_bulk_population_receipt = bool(
        eligible
        and localized_h3_subpopulation_receipt
        and chart_receipt
        and h3_strict_receipt
    )
    normalized_boundary_gate_mode = str(boundary_gate_mode).strip().lower().replace("-", "_")
    if normalized_boundary_gate_mode in {
        "leakage_audit",
        "boundary_leakage_audit",
        "localized_h3_vs_shuffled",
        "h3_localized_vs_shuffled",
    }:
        selected_localized_median_receipt = localized_h3_subpopulation_median_receipt
        selected_localized_receipt = localized_h3_subpopulation_receipt
        bulk_receipt = bool(localized_h3_bulk_population_receipt)
        selected_gate_mode = "localized_h3_subpopulation_vs_shuffled_with_boundary_leakage_audit"
    else:
        selected_localized_median_receipt = localized_nonboundary_subpopulation_median_receipt
        selected_localized_receipt = localized_nonboundary_subpopulation_receipt
        bulk_receipt = bool(localized_nonboundary_bulk_population_receipt)
        selected_gate_mode = "localized_nonboundary_subpopulation_vs_same_filter_shuffled_controls"
    boundary_leakage_audit_pass = bool(h3_not_boundary_dominated)
    selected_claim_boundary = (
        "places persistent observer-facing record objects into the observer-derived modular-response "
        "H3 chart using only sampled observers that see the object. The active populated-chart gate "
        "requires localized H3 object incidence to beat shuffled incidence and the strict modular-response "
        "H3 chart gate. Screen-boundary compactness is reported as a leakage audit, not as an automatic "
        "veto, because the OPH papers treat S2 as the observer-facing angular chart for the H3/Lorentz "
        "branch. This is not a neutral third-person reconstruction, CMB, or particle claim."
        if selected_gate_mode.startswith("localized_h3")
        else (
            "places persistent observer-facing record objects into the observer-derived modular-response "
            "H3 chart using only sampled observers that see the object. A chart receipt reports whole-object-set "
            "compactness against shuffled observer-object incidence. A populated-bulk receipt is narrower: it "
            "requires a localized non-boundary object subpopulation that beats the same localized non-boundary "
            "criterion under shuffled H3 observer incidence and the strict modular-response H3 chart gate. The "
            "localized-object precursor can pass on the intermediate control-separation H3 receipt, but the "
            "populated-bulk receipt cannot. This is not a CMB or particle claim. When split_h3_components is "
            "true, broad observer-visible packet classes are split into local H3 components and the same split "
            "is applied to shuffled controls."
        )
    )
    return {
        "mode": "observer_chart_object_h3_population",
        "observer_count": int(h3_points.shape[0]),
        "object_count": int(len(object_rows)),
        "min_objects": int(min_objects),
        "min_observers_per_object": int(min_observers_per_object),
        "incidence_mode": str(incidence_mode),
        "min_packet_mass": float(min_packet_mass),
        "min_transition_affinity": float(min_transition_affinity),
        "transition_affinity_score": str(transition_affinity_score),
        "observer_cluster_fields": [str(field) for field in observer_cluster_fields],
        "observer_cluster_top_k": int(observer_cluster_top_k),
        "min_observer_cluster_weight": float(min_observer_cluster_weight),
        "history_window": int(history_window),
        "min_persistence": int(min_persistence),
        "max_observer_fraction_per_object": max_observer_fraction,
        "max_h3_compactness": float(max_h3_compactness),
        "min_localized_objects": int(min_localized_objects),
        "pass_ratio": float(pass_ratio),
        "shuffle_control_count": int(shuffle_count),
        "split_h3_components": bool(split_h3_components),
        "component_link_fraction": float(component_link_fraction),
        "component_min_observers": (
            int(component_min_observers) if component_min_observers is not None else int(min_observers_per_object)
        ),
        "require_support_visibility": bool(require_support_visibility),
        "min_support_visibility": float(min_support_visibility),
        "visibility_mode": str(visibility_mode),
        "packet_visibility_weight": float(packet_visibility_weight),
        "requested_boundary_gate_mode": str(boundary_gate_mode),
        "bulk_population_gate_mode": selected_gate_mode,
        "global_h3_pairwise_median": float(global_h3_scale),
        "global_s2_pairwise_median": float(global_s2_scale),
        "median_h3_compactness_normalized": median_h3 if np.isfinite(median_h3) else None,
        "median_s2_boundary_compactness_normalized": median_s2 if np.isfinite(median_s2) else None,
        "median_shuffled_h3_compactness_normalized": median_shuffled if np.isfinite(median_shuffled) else None,
        "p10_shuffled_h3_compactness_normalized": p10_shuffled,
        "p90_shuffled_h3_compactness_normalized": p90_shuffled,
        "localized_object_count": localized_count,
        "localized_not_boundary_object_count": localized_not_boundary_count,
        "shuffled_localized_object_count": shuffled_localized_count,
        "shuffled_localized_object_p90": shuffled_localized_p90,
        "shuffled_localized_not_boundary_object_count": shuffled_localized_not_boundary_count,
        "shuffled_localized_not_boundary_object_p90": shuffled_localized_not_boundary_p90,
        "localized_nonboundary_object_median_precursor_receipt": localized_nonboundary_subpopulation_median_receipt,
        "localized_nonboundary_object_precursor_receipt": localized_nonboundary_subpopulation_receipt,
        "localized_h3_object_median_precursor_receipt": localized_h3_subpopulation_median_receipt,
        "localized_h3_object_precursor_receipt": localized_h3_subpopulation_receipt,
        "localized_object_median_precursor_receipt": selected_localized_median_receipt,
        "localized_object_precursor_receipt": selected_localized_receipt,
        "localized_nonboundary_bulk_population_receipt": localized_nonboundary_bulk_population_receipt,
        "localized_h3_bulk_population_receipt": localized_h3_bulk_population_receipt,
        "h3_beats_shuffled_incidence": h3_beats_shuffle,
        "h3_beats_shuffled_incidence_robust": h3_beats_shuffle_robust,
        "h3_not_boundary_dominated": h3_not_boundary_dominated,
        "boundary_leakage_audit_pass": boundary_leakage_audit_pass,
        "h3_localized": h3_localized,
        "modular_response_h3_receipt": h3_strict_receipt,
        "modular_response_h3_strict_receipt": h3_strict_receipt,
        "modular_response_h3_control_separation_receipt": h3_control_separation_receipt,
        "observer_chart_object_h3_median_receipt": chart_median_receipt,
        "observer_chart_object_h3_receipt": chart_receipt,
        "observer_chart_bulk_population_receipt": bulk_receipt,
        "sample_objects": object_rows[:256],
        "claim_boundary": selected_claim_boundary,
    }


def s2_boundary_profiles(axes: np.ndarray, caps: list[RoundCap]) -> np.ndarray:
    axes = np.asarray(axes, dtype=float)
    if axes.size == 0 or not caps:
        return np.zeros((axes.shape[0], 0), dtype=float)
    profiles = []
    for cap in caps:
        profiles.append(cap_weights(axes, cap, soft=True))
    return np.vstack(profiles).T


def record_populated_h3_report(
    points: np.ndarray,
    caps: list[RoundCap],
    raw_fields: dict[str, np.ndarray],
    observer_views: list[dict[str, Any]],
    *,
    cell_entropy: np.ndarray | None = None,
    seed: int = 1,
    field_names: list[str] | tuple[str, ...] = DEFAULT_RECORD_FIELDS,
    candidate_count: int = 2048,
    candidate_radius: float = 2.0,
    softness: float = 0.25,
    pass_ratio: float = 0.85,
    geometry_blend: float = 0.0,
    response_mode: str = "field_summary_similarity",
    transport_time: float = 0.1,
    transport_scale: float = 2.0 * np.pi,
) -> dict[str, Any]:
    response, patch_views, response_report = record_cap_response_matrix(
        points,
        caps,
        raw_fields,
        observer_views,
        cell_entropy=cell_entropy,
        field_names=field_names,
        geometry_blend=geometry_blend,
        response_mode=response_mode,
        transport_time=transport_time,
        transport_scale=transport_scale,
    )
    h3_fit = fit_response_profiles_to_h3(
        response,
        caps,
        candidate_count=candidate_count,
        candidate_radius=candidate_radius,
        softness=softness,
        seed=seed,
    )
    axes = np.vstack([np.asarray(view.get("axis"), dtype=float) for view in patch_views]) if patch_views else np.zeros((0, 3))
    s2_profiles = s2_boundary_profiles(axes, caps)
    s2_residuals = _profile_residual_pairwise(response, s2_profiles)
    rng = np.random.default_rng(seed + 101)
    shuffled = response.copy()
    if shuffled.size:
        for row in shuffled:
            rng.shuffle(row)
    shuffled_fit = fit_response_profiles_to_h3(
        shuffled,
        caps,
        candidate_count=max(128, min(int(candidate_count), 1024)),
        candidate_radius=candidate_radius,
        softness=softness,
        seed=seed + 202,
    )
    h3_median = _as_float(h3_fit.get("median_residual"))
    s2_median = float(np.median(s2_residuals)) if s2_residuals.size else float("nan")
    shuffled_median = _as_float(shuffled_fit.get("median_residual"))
    h3_beats_s2 = bool(np.isfinite(h3_median) and np.isfinite(s2_median) and h3_median < float(pass_ratio) * s2_median)
    h3_beats_shuffle = bool(
        np.isfinite(h3_median) and np.isfinite(shuffled_median) and h3_median < float(pass_ratio) * shuffled_median
    )
    receipt = bool(h3_beats_s2 and h3_beats_shuffle)
    return {
        "mode": "record_populated_h3_fit",
        "observer_count": len(patch_views),
        "cap_count": len(caps),
        "response_report": response_report,
        "h3_fit": h3_fit,
        "s2_boundary_control": {
            "median_residual": s2_median if np.isfinite(s2_median) else None,
            "mean_residual": float(np.mean(s2_residuals)) if s2_residuals.size else None,
            "p90_residual": float(np.percentile(s2_residuals, 90)) if s2_residuals.size else None,
            "h3_beats_s2_boundary": h3_beats_s2,
        },
        "shuffled_cap_response_control": {
            "median_residual": shuffled_fit.get("median_residual"),
            "h3_beats_shuffled": h3_beats_shuffle,
        },
        "pass_ratio": float(pass_ratio),
        "record_populated_h3_receipt": receipt,
        "claim_boundary": (
            "fits observer-visible record/cap response profiles into the canonical H3 chart. "
            "A true receipt requires H3 residuals to beat S2-boundary and shuffled controls; "
            "this is still not a physical CMB or particle claim."
        ),
    }


def support_profiles_to_h3_report(
    points: np.ndarray,
    caps: list[RoundCap],
    supports: list[dict[str, Any]],
    *,
    cell_entropy: np.ndarray | None = None,
    seed: int = 1,
    support_key: str = "support_nodes",
    id_key: str = "object_id",
    label: str = "record_family",
    candidate_count: int = 2048,
    candidate_radius: float = 2.0,
    softness: float = 0.25,
    pass_ratio: float = 0.85,
    min_support_count: int = 8,
    min_cap_count: int = 6,
) -> dict[str, Any]:
    """Fit observer-visible support profiles to the canonical H3 chart.

    This is intended for record families and holonomy defect clusters: the
    input is only a support set on the screen plus cap weights, so it is still
    a support-visible reconstruction diagnostic rather than a host-coordinate
    embedding.
    """

    rows = [row for row in supports if row.get(support_key)]
    if not rows or not caps:
        return {
            "mode": "support_profile_h3_fit",
            "label": label,
            "support_count": 0,
            "record_populated_h3_receipt": False,
            "reason": "empty_supports_or_caps",
        }
    weights = np.asarray(cell_entropy, dtype=float) if cell_entropy is not None else np.ones(points.shape[0], dtype=float)
    response = np.vstack([_support_cap_profile(points, caps, row.get(support_key, []), weights) for row in rows])
    h3_fit = fit_response_profiles_to_h3(
        response,
        caps,
        candidate_count=candidate_count,
        candidate_radius=candidate_radius,
        softness=softness,
        seed=seed,
    )
    s2_axes = np.vstack([_support_axis(points, row.get(support_key, []), weights) for row in rows])
    s2_profiles = s2_boundary_profiles(s2_axes, caps)
    s2_residuals = _profile_residual_pairwise(response, s2_profiles)
    rng = np.random.default_rng(seed + 307)
    shuffled = response.copy()
    for row in shuffled:
        rng.shuffle(row)
    shuffled_fit = fit_response_profiles_to_h3(
        shuffled,
        caps,
        candidate_count=max(128, min(int(candidate_count), 1024)),
        candidate_radius=candidate_radius,
        softness=softness,
        seed=seed + 401,
    )
    h3_median = _as_float(h3_fit.get("median_residual"))
    s2_median = float(np.median(s2_residuals)) if s2_residuals.size else float("nan")
    shuffled_median = _as_float(shuffled_fit.get("median_residual"))
    h3_beats_s2 = bool(np.isfinite(h3_median) and np.isfinite(s2_median) and h3_median < float(pass_ratio) * s2_median)
    h3_beats_shuffle = bool(
        np.isfinite(h3_median) and np.isfinite(shuffled_median) and h3_median < float(pass_ratio) * shuffled_median
    )
    eligible = bool(len(rows) >= int(min_support_count) and len(caps) >= int(min_cap_count))
    dimension_debug = _h3_population_dimension_debug(h3_fit)
    return {
        "mode": "support_profile_h3_fit",
        "label": label,
        "support_count": len(rows),
        "cap_count": len(caps),
        "support_key": support_key,
        "id_key": id_key,
        "sample_ids": [str(row.get(id_key, index)) for index, row in enumerate(rows[:32])],
        "support_size_summary": _support_size_summary(rows, support_key),
        "h3_fit": h3_fit,
        "h3_population_dimension_debug": dimension_debug,
        "s2_boundary_control": {
            "median_residual": s2_median if np.isfinite(s2_median) else None,
            "mean_residual": float(np.mean(s2_residuals)) if s2_residuals.size else None,
            "p90_residual": float(np.percentile(s2_residuals, 90)) if s2_residuals.size else None,
            "h3_beats_s2_boundary": h3_beats_s2,
        },
        "shuffled_cap_response_control": {
            "median_residual": shuffled_fit.get("median_residual"),
            "h3_beats_shuffled": h3_beats_shuffle,
        },
        "pass_ratio": float(pass_ratio),
        "min_support_count": int(min_support_count),
        "min_cap_count": int(min_cap_count),
        "eligibility_gate_passed": eligible,
        "record_populated_h3_receipt": bool(eligible and h3_beats_s2 and h3_beats_shuffle),
        "record_family_h3_bulk_population_candidate": bool(
            eligible
            and h3_beats_s2
            and h3_beats_shuffle
            and dimension_debug.get("point_count", 0) >= int(min_support_count)
        ),
        "claim_boundary": (
            "support-visible cap-profile fit into H3 for record families or screen holonomy defects. "
            "A pass is a populated-chart diagnostic only; particle claims still require persistence, "
            "transport, fusion/scattering, and worldline controls."
        ),
    }


def _h3_population_dimension_debug(h3_fit: dict[str, Any]) -> dict[str, Any]:
    points = np.asarray(h3_fit.get("fitted_h3_points", []), dtype=float)
    if points.ndim != 2 or points.shape[0] < 8 or points.shape[1] != 4:
        return {
            "mode": "h3_population_dimension_debug",
            "point_count": int(points.shape[0]) if points.ndim == 2 else 0,
            "candidate_3d_dimension_window": False,
            "reason": "not_enough_h3_points",
            "claim_boundary": "debug-only dimension estimate on fitted H3 population points; no gate by itself",
        }
    report = neutral_dimension_report_from_distance(h3_distance_matrix(points))
    corr = report.get("correlation_dimension", {}).get("estimate")
    mle = report.get("local_mle_dimension", {}).get("estimate")
    candidate = bool(
        isinstance(corr, (int, float))
        and isinstance(mle, (int, float))
        and np.isfinite(float(corr))
        and np.isfinite(float(mle))
        and 2.7 <= float(corr) <= 3.3
        and 2.7 <= float(mle) <= 3.3
    )
    return {
        "mode": "h3_population_dimension_debug",
        "point_count": int(points.shape[0]),
        "candidate_3d_dimension_window": candidate,
        "dimension_estimators_agree": bool(report.get("dimension_estimators_agree", False)),
        "correlation_dimension": report.get("correlation_dimension"),
        "local_mle_dimension": report.get("local_mle_dimension"),
        "claim_boundary": (
            "debug-only estimate on fitted H3 population points. It cannot establish bulk by itself; "
            "population controls and receipt gates remain primary."
        ),
    }


def defect_timeline_to_h3_report(
    points: np.ndarray,
    caps: list[RoundCap],
    timeline_report: dict[str, Any],
    *,
    raw_fields: dict[str, np.ndarray] | None = None,
    field_names: list[str] | tuple[str, ...] = DEFAULT_RECORD_FIELDS,
    cell_entropy: np.ndarray | None = None,
    seed: int = 1,
    candidate_count: int = 2048,
    candidate_radius: float = 2.0,
    softness: float = 0.25,
    pass_ratio: float = 0.85,
    max_events: int = 1024,
    response_mode: str = "support_cap_profile",
    transport_time: float = 0.1,
    transport_scale: float = 2.0 * np.pi,
) -> dict[str, Any]:
    """Fit time-resolved defect supports into the canonical H3 chart.

    This is a visualization/precursor layer for particle-like defects. It uses
    support-visible screen nodes from the defect timeline. In
    ``support_cap_profile`` mode it fits the raw cap footprint of the support.
    In ``support_transport_similarity`` mode it fits the observer-visible
    cap-modular response of each defect support, which is closer to the
    BW/KMS mechanism and avoids treating boundary position alone as bulk
    evidence. The particle gate remains false until independent transport,
    fusion, scattering, and bulk controls pass.
    """

    event_rows = _timeline_event_rows(timeline_report, max_events=max_events)
    if not event_rows or not caps:
        return {
            "mode": "defect_timeline_h3_worldline_fit",
            "event_count": 0,
            "worldline_count": 0,
            "bulk_worldline_precursor_receipt": False,
            "particle_matter_receipt": False,
            "reason": "empty_timeline_or_caps",
        }
    weights = np.asarray(cell_entropy, dtype=float) if cell_entropy is not None else np.ones(points.shape[0], dtype=float)
    mode = str(response_mode)
    response_meta: dict[str, Any] = {
        "response_mode": mode,
        "cap_count": len(caps),
        "field_names": [],
        "transport_time": float(transport_time),
        "transport_scale": float(transport_scale),
    }
    if mode == "support_transport_similarity" and raw_fields:
        fields = _standardized_fields(raw_fields, field_names)
        if fields:
            tree = cKDTree(points)
            response = np.vstack(
                [
                    _support_transport_cap_profile(
                        points,
                        tree,
                        caps,
                        row.get("support_nodes", []),
                        fields,
                        weights,
                        transport_time=float(transport_time),
                        transport_scale=float(transport_scale),
                    )
                    for row in event_rows
                ]
            )
            response_meta.update(
                {
                    "field_names": list(fields.keys()),
                    "response_source": "observer_visible_defect_support_field_similarity_under_lambda_C_transport",
                    "claim_boundary": (
                        "defect event response uses only observer-visible fields on the defect support "
                        "before/after finite cap modular transport; this is a BW/KMS-aligned precursor, "
                        "not a particle or physical bulk claim"
                    ),
                }
            )
        else:
            response = np.vstack([_support_cap_profile(points, caps, row.get("support_nodes", []), weights) for row in event_rows])
            response_meta.update(
                {
                    "response_mode": "support_cap_profile",
                    "fallback_reason": "requested support_transport_similarity but no requested raw fields were available",
                }
            )
    else:
        response = np.vstack([_support_cap_profile(points, caps, row.get("support_nodes", []), weights) for row in event_rows])
        response_meta.update(
            {
                "response_source": "raw_defect_support_cap_membership",
                "claim_boundary": (
                    "defect event response is a support footprint on the screen; useful as a boundary "
                    "diagnostic but too screen-local to establish bulk transport"
                ),
            }
        )
    response_meta["response_summary"] = _response_summary(response)
    normals = cap_normals(caps)
    candidates = random_h3_points(int(candidate_count), seed=seed, radius=float(candidate_radius))
    candidate_profiles = h3_halfspace_profile(candidates, normals, softness=float(softness))
    residuals = _profile_residual_matrix(response, candidate_profiles)
    best_indices = np.argmin(residuals, axis=1)
    best_residuals = residuals[np.arange(response.shape[0]), best_indices]
    fitted = candidates[best_indices]
    fitted, best_residuals, refinement = _refine_h3_fit(
        response,
        normals,
        fitted,
        best_residuals,
        softness=float(softness),
        max_rows=128,
        max_nfev=48,
    )

    s2_axes = np.vstack([_support_axis(points, row.get("support_nodes", []), weights) for row in event_rows])
    s2_profiles = s2_boundary_profiles(s2_axes, caps)
    s2_residuals = _profile_residual_pairwise(response, s2_profiles)
    rng = np.random.default_rng(seed + 811)
    shuffled = response.copy()
    for row in shuffled:
        rng.shuffle(row)
    shuffled_candidates = random_h3_points(max(128, min(int(candidate_count), 1024)), seed=seed + 812, radius=float(candidate_radius))
    shuffled_profiles = h3_halfspace_profile(shuffled_candidates, normals, softness=float(softness))
    shuffled_residuals = _profile_residual_matrix(shuffled, shuffled_profiles)
    shuffled_best = np.min(shuffled_residuals, axis=1) if shuffled_residuals.size else np.zeros(0)
    if shuffled_best.size:
        shuffled_fitted = shuffled_candidates[np.argmin(shuffled_residuals, axis=1)]
        _shuffled_fitted, shuffled_best, _shuffled_refinement = _refine_h3_fit(
            shuffled,
            normals,
            shuffled_fitted,
            shuffled_best,
            softness=float(softness),
            max_rows=128,
            max_nfev=48,
        )

    h3_median = float(np.median(best_residuals)) if best_residuals.size else float("nan")
    s2_median = float(np.median(s2_residuals)) if s2_residuals.size else float("nan")
    shuffled_median = float(np.median(shuffled_best)) if shuffled_best.size else float("nan")
    response_degenerate = bool(float(response_meta["response_summary"]["mean_row_std"]) < 1e-3)
    h3_beats_s2 = bool(np.isfinite(h3_median) and np.isfinite(s2_median) and h3_median < float(pass_ratio) * s2_median)
    h3_beats_shuffle = bool(
        np.isfinite(h3_median) and np.isfinite(shuffled_median) and h3_median < float(pass_ratio) * shuffled_median
    )
    event_reports = []
    for index, row in enumerate(event_rows):
        event_reports.append(
            {
                "worldline_id": row["worldline_id"],
                "cycle": int(row["cycle"]),
                "class": row.get("class"),
                "support_node_count": int(row.get("support_node_count", 0)),
                "h3_point": [float(value) for value in fitted[index]],
                "h3_spatial_point": [float(value) for value in fitted[index][1:]],
                "fit_residual": float(best_residuals[index]),
            }
        )
    worldlines = _h3_worldline_summaries(event_reports)
    persistent_h3 = [row for row in worldlines if int(row.get("observation_count", 0)) >= 3]
    return {
        "mode": "defect_timeline_h3_worldline_fit",
        "event_count": len(event_reports),
        "worldline_count": len(worldlines),
        "persistent_h3_worldline_count": len(persistent_h3),
        "cap_count": len(caps),
        "candidate_count": int(candidate_count),
        "candidate_radius": float(candidate_radius),
        "softness": float(softness),
        "local_refinement": refinement,
        "response_report": response_meta,
        "median_h3_residual": h3_median if np.isfinite(h3_median) else None,
        "median_s2_boundary_residual": s2_median if np.isfinite(s2_median) else None,
        "median_shuffled_residual": shuffled_median if np.isfinite(shuffled_median) else None,
        "h3_beats_s2_boundary": h3_beats_s2,
        "h3_beats_shuffled": h3_beats_shuffle,
        "response_degenerate": response_degenerate,
        "bulk_worldline_precursor_receipt": bool(
            persistent_h3 and not response_degenerate and h3_beats_s2 and h3_beats_shuffle
        ),
        "particle_matter_receipt": False,
        "sample_events": event_reports[:256],
        "worldlines": worldlines[:256],
        "claim_boundary": (
            "support-visible defect timeline fitted into the canonical H3 chart for visualization and "
            "particle-precondition diagnostics. This is not a matter-particle claim; particles require "
            "neutral-bulk gates plus localization, transport, fusion/scattering, and repeated-seed controls."
        ),
    }


def _standardized_fields(raw_fields: dict[str, np.ndarray], field_names: list[str] | tuple[str, ...]) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    for name in field_names:
        if name not in raw_fields:
            continue
        values = np.asarray(raw_fields[name], dtype=float)
        std = float(np.std(values))
        fields[str(name)] = (values - float(np.mean(values))) / std if std > 1e-12 else values - float(np.mean(values))
    return fields


def _cap_feature(points: np.ndarray, cap: RoundCap, fields: dict[str, np.ndarray], weights: np.ndarray) -> np.ndarray:
    cap_w = cap_weights(points, cap, soft=True) * weights
    total = max(float(np.sum(cap_w)), 1e-12)
    return np.array([float(np.sum(values * cap_w) / total) for values in fields.values()], dtype=float)


def _observer_feature(view: dict[str, Any], fields: dict[str, np.ndarray], weights: np.ndarray) -> np.ndarray:
    support = np.asarray(view.get("support_nodes", []), dtype=np.int64)
    if support.size == 0:
        return np.zeros(len(fields), dtype=float)
    support = support[(support >= 0) & (support < weights.size)]
    support_w = weights[support]
    total = max(float(np.sum(support_w)), 1e-12)
    return np.array([float(np.sum(values[support] * support_w) / total) for values in fields.values()], dtype=float)


def _cap_transport_response(
    points: np.ndarray,
    caps: list[RoundCap],
    fields: dict[str, np.ndarray],
    patch_views: list[dict[str, Any]],
    weights: np.ndarray,
    *,
    transport_time: float,
    transport_scale: float,
) -> np.ndarray:
    if not patch_views or not caps:
        return np.zeros((len(patch_views), len(caps)), dtype=float)
    tree = cKDTree(points)
    source_features = np.vstack([_observer_feature(view, fields, weights) for view in patch_views])
    rows = np.zeros((len(patch_views), len(caps)), dtype=float)
    scale = max(float(np.sqrt(len(fields))), 1e-12)
    for cap_index, cap in enumerate(caps):
        transported_features = []
        for view in patch_views:
            support = np.asarray(view.get("support_nodes", []), dtype=np.int64)
            support = support[(support >= 0) & (support < points.shape[0])]
            if support.size == 0:
                transported_features.append(np.zeros(len(fields), dtype=float))
                continue
            transported_points = lambda_cap(points[support], cap, float(transport_scale) * float(transport_time))
            _, transported_indices = tree.query(transported_points, k=1)
            transported_view = {"support_nodes": np.asarray(transported_indices, dtype=np.int64)}
            transported_features.append(_observer_feature(transported_view, fields, weights))
        transported = np.vstack(transported_features)
        distances = np.linalg.norm(source_features - transported, axis=1)
        rows[:, cap_index] = np.exp(-distances / scale)
    return rows


def _support_cap_profile(
    points: np.ndarray,
    caps: list[RoundCap],
    support_nodes: list[int] | np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    support = np.asarray(support_nodes, dtype=np.int64)
    support = support[(support >= 0) & (support < points.shape[0])]
    if support.size == 0:
        return np.zeros(len(caps), dtype=float)
    support_w = weights[support]
    total = max(float(np.sum(support_w)), 1e-12)
    return np.array(
        [float(np.sum(cap_weights(points[support], cap, soft=True) * support_w) / total) for cap in caps],
        dtype=float,
    )


def _support_transport_cap_profile(
    points: np.ndarray,
    tree: cKDTree,
    caps: list[RoundCap],
    support_nodes: list[int] | np.ndarray,
    fields: dict[str, np.ndarray],
    weights: np.ndarray,
    *,
    transport_time: float,
    transport_scale: float,
) -> np.ndarray:
    support = np.asarray(support_nodes, dtype=np.int64)
    support = support[(support >= 0) & (support < points.shape[0])]
    if support.size == 0 or not fields:
        return np.zeros(len(caps), dtype=float)
    source = _support_feature(support, fields, weights)
    scale = max(float(np.sqrt(len(fields))), 1e-12)
    values: list[float] = []
    for cap in caps:
        transported_points = lambda_cap(points[support], cap, float(transport_scale) * float(transport_time))
        _, transported_indices = tree.query(transported_points, k=1)
        transported = np.asarray(transported_indices, dtype=np.int64)
        target = _support_feature(transported, fields, weights)
        distance = float(np.linalg.norm(source - target))
        values.append(float(np.exp(-distance / scale)))
    return np.asarray(values, dtype=float)


def _support_feature(support: np.ndarray, fields: dict[str, np.ndarray], weights: np.ndarray) -> np.ndarray:
    support = np.asarray(support, dtype=np.int64)
    support = support[(support >= 0) & (support < weights.size)]
    if support.size == 0:
        return np.zeros(len(fields), dtype=float)
    support_w = weights[support]
    total = max(float(np.sum(support_w)), 1e-12)
    return np.asarray([float(np.sum(values[support] * support_w) / total) for values in fields.values()], dtype=float)


def _support_axis(points: np.ndarray, support_nodes: list[int] | np.ndarray, weights: np.ndarray) -> np.ndarray:
    support = np.asarray(support_nodes, dtype=np.int64)
    support = support[(support >= 0) & (support < points.shape[0])]
    if support.size == 0:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    support_w = weights[support]
    centroid = np.sum(points[support] * support_w[:, None], axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return centroid / norm


def _support_size_summary(rows: list[dict[str, Any]], support_key: str) -> dict[str, float]:
    sizes = np.array([len(row.get(support_key, [])) for row in rows], dtype=float)
    return {
        "min": float(np.min(sizes)) if sizes.size else 0.0,
        "median": float(np.median(sizes)) if sizes.size else 0.0,
        "mean": float(np.mean(sizes)) if sizes.size else 0.0,
        "max": float(np.max(sizes)) if sizes.size else 0.0,
    }


def _timeline_event_rows(timeline_report: dict[str, Any], *, max_events: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for worldline in timeline_report.get("worldlines", []) if timeline_report else []:
        worldline_id = str(worldline.get("worldline_id"))
        if not worldline.get("persistent", False):
            continue
        for event in worldline.get("events", []):
            support = event.get("support_nodes", [])
            if not support:
                continue
            rows.append(
                {
                    "worldline_id": worldline_id,
                    "cycle": int(event.get("cycle", 0)),
                    "class": event.get("class"),
                    "support_node_count": int(event.get("support_node_count", len(support))),
                    "support_nodes": support,
                }
            )
            if len(rows) >= int(max_events):
                return rows
    return rows


def _h3_worldline_summaries(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault(str(event["worldline_id"]), []).append(event)
    rows = []
    for worldline_id, group in grouped.items():
        group.sort(key=lambda row: int(row["cycle"]))
        points = np.asarray([row["h3_point"] for row in group], dtype=float)
        segment_lengths = [
            _h3_pair_distance(points[index - 1], points[index])
            for index in range(1, points.shape[0])
        ]
        cycles = [int(row["cycle"]) for row in group]
        rows.append(
            {
                "worldline_id": worldline_id,
                "observation_count": len(group),
                "birth_cycle": min(cycles) if cycles else 0,
                "death_cycle": max(cycles) if cycles else 0,
                "h3_path_length": float(np.sum(segment_lengths)) if segment_lengths else 0.0,
                "mean_h3_step": float(np.mean(segment_lengths)) if segment_lengths else 0.0,
                "class_mode": _mode([row.get("class") for row in group]),
                "events": [
                    {
                        "cycle": int(row["cycle"]),
                        "h3_spatial_point": row["h3_spatial_point"],
                        "fit_residual": row["fit_residual"],
                        "support_node_count": row["support_node_count"],
                    }
                    for row in group
                ],
            }
        )
    rows.sort(key=lambda row: (-int(row["observation_count"]), str(row["worldline_id"])))
    return rows


def _h3_pair_distance(a: np.ndarray, b: np.ndarray) -> float:
    value = -(-float(a[0]) * float(b[0]) + float(np.dot(a[1:], b[1:])))
    return float(np.arccosh(max(1.0, value)))


def _mode(values: list[Any]) -> Any:
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get) if counts else None


def _refine_h3_fit(
    response: np.ndarray,
    normals: np.ndarray,
    fitted: np.ndarray,
    residuals: np.ndarray,
    *,
    softness: float,
    max_rows: int,
    max_nfev: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    response = np.asarray(response, dtype=float)
    fitted = np.asarray(fitted, dtype=float).copy()
    residuals = np.asarray(residuals, dtype=float).copy()
    if response.size == 0 or fitted.size == 0 or residuals.size == 0 or max_rows <= 0:
        return fitted, residuals, {"enabled": False, "refined_rows": 0, "median_improvement": 0.0}
    row_count = min(int(max_rows), int(response.shape[0]))
    order = np.argsort(residuals)[::-1][:row_count]
    improvements: list[float] = []
    width = max(float(softness), 1e-9)
    for row_index in order:
        initial = h3_tangent_from_point(fitted[int(row_index)])

        def objective(vector: np.ndarray) -> np.ndarray:
            point = h3_point_from_tangent(vector)
            signed = -point[0] * normals[:, 0] + normals[:, 1:] @ point[1:]
            profile = 1.0 / (1.0 + np.exp(-np.clip(signed / width, -60.0, 60.0)))
            return profile - response[int(row_index)]

        result = least_squares(objective, initial, max_nfev=max(1, int(max_nfev)), method="trf")
        if not result.success and result.cost >= 0.5 * residuals[int(row_index)] ** 2 * response.shape[1]:
            continue
        candidate = h3_point_from_tangent(result.x)
        candidate_residual = float(np.sqrt(np.mean(objective(result.x) ** 2)))
        old = float(residuals[int(row_index)])
        if np.isfinite(candidate_residual) and candidate_residual < old:
            fitted[int(row_index)] = candidate
            residuals[int(row_index)] = candidate_residual
            improvements.append(old - candidate_residual)
    return fitted, residuals, {
        "enabled": True,
        "refined_rows": int(len(improvements)),
        "attempted_rows": int(row_count),
        "max_nfev": int(max_nfev),
        "median_improvement": float(np.median(improvements)) if improvements else 0.0,
    }


def _profile_residual_matrix(observed: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    if observed.size == 0 or candidates.size == 0:
        return np.zeros((observed.shape[0], candidates.shape[0]), dtype=float)
    diff = observed[:, None, :] - candidates[None, :, :]
    return np.sqrt(np.mean(diff * diff, axis=2))


def _profile_residual_pairwise(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape or left.size == 0:
        return np.zeros(0, dtype=float)
    diff = left - right
    return np.sqrt(np.mean(diff * diff, axis=1))


def _response_summary(response: np.ndarray) -> dict[str, float]:
    response = np.asarray(response, dtype=float)
    if response.size == 0:
        return {
            "min": 0.0,
            "mean": 0.0,
            "max": 0.0,
            "std": 0.0,
            "mean_row_std": 0.0,
            "mean_col_std": 0.0,
        }
    row_std = np.std(response, axis=1) if response.ndim == 2 else np.asarray([float(np.std(response))])
    col_std = np.std(response, axis=0) if response.ndim == 2 else np.asarray([float(np.std(response))])
    return {
        "min": float(np.min(response)),
        "mean": float(np.mean(response)),
        "max": float(np.max(response)),
        "std": float(np.std(response)),
        "mean_row_std": float(np.mean(row_std)) if row_std.size else 0.0,
        "mean_col_std": float(np.mean(col_std)) if col_std.size else 0.0,
    }


def _empty_fit(reason: str) -> dict[str, Any]:
    return {
        "mode": "record_response_to_h3_fit",
        "status": "empty",
        "reason": reason,
        "median_residual": None,
        "mean_residual": None,
        "p90_residual": None,
    }


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _observer_h3_chart(h3_report: dict[str, Any]) -> dict[str, Any]:
    ids = [int(value) for value in h3_report.get("observer_ids", [])]
    fit = h3_report.get("h3_fit", {}) if h3_report else {}
    points = np.asarray(fit.get("fitted_h3_points", []), dtype=float)
    if points.ndim != 2 or points.shape[1] != 4:
        points = np.zeros((0, 4), dtype=float)
    if not ids:
        ids = list(range(points.shape[0]))
    count = min(len(ids), points.shape[0])
    ids = ids[:count]
    points = points[:count]
    return {
        "observer_ids": ids,
        "points": points,
        "by_id": {int(observer_id): points[index] for index, observer_id in enumerate(ids)},
    }


def _histogram_value(histogram: Any, key: int) -> float:
    if not isinstance(histogram, dict):
        return 0.0
    if key in histogram:
        return float(histogram[key])
    return float(histogram.get(str(int(key)), 0.0))


def _combine_affinity_scores(scores: list[float], *, mode: str) -> float:
    if not scores:
        return 0.0
    values = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    if values.size == 0:
        return 0.0
    score_mode = str(mode)
    if score_mode == "mean":
        return float(np.mean(values))
    if score_mode == "min":
        return float(np.min(values))
    if score_mode == "product":
        return float(np.prod(values))
    # Geometric mean penalizes weak transition descriptors without requiring
    # brittle all-or-nothing exact agreement.
    return float(np.exp(np.mean(np.log(np.maximum(values, 1e-12)))))


def _object_incidence_indices(
    chart_views: list[dict[str, Any]],
    family: dict[str, Any],
    *,
    mode: str,
    min_packet_mass: float,
    min_transition_affinity: float,
    transition_affinity_score: str,
) -> tuple[list[int], list[float]]:
    support = set(int(node) for node in family.get("support_nodes", []))
    signature = int(family.get("record_signature", -1))
    affinity = family.get("transition_affinity") if isinstance(family.get("transition_affinity"), dict) else {}
    indices: list[int] = []
    weights: list[float] = []
    for observer_index, view in enumerate(chart_views):
        histogram = view.get("object_packet_histogram") or view.get("record_signature_histogram", {})
        packet_mass = _histogram_value(histogram, signature)
        if mode in {"transition_affinity", "visible_transition_affinity", "packet_transition_affinity"}:
            score_terms: list[float] = []
            if packet_mass > 0.0:
                score_terms.append(float(packet_mass))
            transition_histograms = view.get("transition_affinity_histograms", {})
            if isinstance(transition_histograms, dict) and isinstance(affinity, dict):
                for name, target in affinity.items():
                    if str(name) == "object_packet":
                        continue
                    score_terms.append(_histogram_value(transition_histograms.get(str(name), {}), int(target)))
            score = _combine_affinity_scores(score_terms, mode=str(transition_affinity_score))
            if score < float(min_transition_affinity):
                continue
            indices.append(observer_index)
            weights.append(score)
            continue
        if mode == "packet_mass":
            if packet_mass < float(min_packet_mass):
                continue
            indices.append(observer_index)
            weights.append(float(packet_mass))
            continue
        view_support = set(int(node) for node in view.get("support_nodes", []))
        overlap = len(support & view_support)
        if overlap <= 0:
            continue
        weight = float(overlap / max(1, len(support)))
        if packet_mass > 0.0:
            weight *= 0.5 + 0.5 * float(packet_mass)
        indices.append(observer_index)
        weights.append(weight)
    return indices, weights


def _record_family_object_chart_rows(
    chart_views: list[dict[str, Any]],
    chart_observer_ids: list[int],
    h3_points: np.ndarray,
    axes: np.ndarray,
    record_families: list[dict[str, Any]],
    *,
    min_observers_per_object: int,
    incidence_mode: str,
    min_packet_mass: float,
    min_transition_affinity: float,
    transition_affinity_score: str,
) -> list[dict[str, Any]]:
    global_h3_scale = _median_pairwise_h3_distance(h3_points)
    global_s2_scale = _median_pairwise_s2_distance(axes)
    object_rows: list[dict[str, Any]] = []
    for family in record_families:
        support = set(int(node) for node in family.get("support_nodes", []))
        if not support:
            continue
        indices, weights = _object_incidence_indices(
            chart_views,
            family,
            mode=str(incidence_mode),
            min_packet_mass=float(min_packet_mass),
            min_transition_affinity=float(min_transition_affinity),
            transition_affinity_score=str(transition_affinity_score),
        )
        if len(indices) < int(min_observers_per_object):
            continue
        object_rows.append(
            _object_chart_row(
                object_id=str(family.get("object_id", len(object_rows))),
                indices=indices,
                weights=np.asarray(weights, dtype=float),
                h3_points=h3_points,
                axes=axes,
                chart_observer_ids=chart_observer_ids,
                global_h3_scale=global_h3_scale,
                global_s2_scale=global_s2_scale,
                support_size=len(support),
            )
        )
    return object_rows


def _record_family_modular_response_mixture_rows(
    chart_views: list[dict[str, Any]],
    chart_observer_ids: list[int],
    h3_points: np.ndarray,
    axes: np.ndarray,
    record_families: list[dict[str, Any]],
    *,
    min_observers_per_object: int,
    max_objects: int,
    max_observer_fraction: float,
    min_transition_affinity: float,
    transition_affinity_score: str,
    observer_cluster_fields: tuple[str, ...],
    top_k: int,
    min_weight: float,
    split_h3_components: bool = False,
    component_link_fraction: float = 0.35,
    component_min_observers: int | None = None,
    require_support_visibility: bool = False,
    min_support_visibility: float = 0.0,
    visibility_mode: str = "support_overlap",
    packet_visibility_weight: float = 0.5,
) -> list[dict[str, Any]]:
    """Split persistent record-family incidence by modular-response chart tokens.

    The pure modular-response cluster lane can become a chart-label clustering
    test rather than an observer-object test. This lane keeps the persistent
    record-family transition descriptor as the first gate, then asks whether the
    observers that see that family form localized modular-response chart
    components. Shuffled controls rebuild the same rows with shuffled H3
    observer coordinates.
    """

    cluster_fields = tuple(field for field in observer_cluster_fields if field)
    if not cluster_fields:
        cluster_fields = (
            "modular_response_cluster",
            "modular_response_component_0",
            "modular_response_component_1",
        )
    global_h3_scale = _median_pairwise_h3_distance(h3_points)
    global_s2_scale = _median_pairwise_s2_distance(axes)
    total_observers = max(1, len(chart_views))
    max_count = max(1, int(np.floor(float(max_observer_fraction) * total_observers))) if max_observer_fraction > 0.0 else total_observers
    view_support_sets = (
        [set(int(node) for node in view.get("support_nodes", [])) for view in chart_views]
        if bool(require_support_visibility)
        else []
    )
    normalized_visibility_mode = str(visibility_mode).strip().lower().replace("-", "_")
    packet_or_support_visibility = normalized_visibility_mode in {
        "packet_or_support",
        "transition_or_support",
        "observer_packet_or_support",
        "record_packet_or_support",
    }
    rows: list[dict[str, Any]] = []
    for family in record_families:
        support = set(int(node) for node in family.get("support_nodes", []))
        if not support:
            continue
        base_indices, base_weights = _object_incidence_indices(
            chart_views,
            family,
            mode="transition_affinity",
            min_packet_mass=0.0,
            min_transition_affinity=float(min_transition_affinity),
            transition_affinity_score=str(transition_affinity_score),
        )
        if len(base_indices) < int(min_observers_per_object):
            continue
        if bool(require_support_visibility):
            visible_indices: list[int] = []
            visible_weights: list[float] = []
            for observer_index, base_weight in zip(base_indices, base_weights, strict=True):
                view_support = view_support_sets[int(observer_index)]
                visibility = 0.0
                if view_support:
                    overlap = len(support & view_support)
                    if overlap > 0:
                        visibility = float(overlap / max(1, min(len(support), len(view_support))))
                support_visible = bool(visibility >= float(min_support_visibility) and visibility > 0.0)
                packet_visible = bool(packet_or_support_visibility and float(base_weight) >= float(min_transition_affinity))
                if not support_visible and not packet_visible:
                    continue
                visibility_weight = (
                    max(float(visibility), 1e-12)
                    if support_visible
                    else max(float(packet_visibility_weight), 1e-12)
                )
                visible_indices.append(int(observer_index))
                visible_weights.append(float(base_weight) * visibility_weight)
            base_indices = visible_indices
            base_weights = visible_weights
            if len(base_indices) < int(min_observers_per_object):
                continue
        clusters: dict[tuple[tuple[str, int], ...], dict[int, float]] = {}
        for observer_index, base_weight in zip(base_indices, base_weights, strict=True):
            view = chart_views[int(observer_index)]
            response_keys = _observer_transition_mixture_keys(
                view,
                cluster_fields=cluster_fields,
                score_mode=str(transition_affinity_score),
                top_k=int(top_k),
                min_weight=float(min_weight),
            )
            if not response_keys:
                continue
            for key, response_weight in response_keys:
                combined = float(base_weight) * float(response_weight)
                if combined < float(min_weight) * float(min_transition_affinity):
                    continue
                previous = clusters.setdefault(key, {}).get(int(observer_index), 0.0)
                clusters[key][int(observer_index)] = max(float(previous), combined)
        ordered = sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0]))
        for key, member_weights in ordered:
            if len(member_weights) < int(min_observers_per_object):
                continue
            if len(member_weights) > max_count:
                continue
            indices = [int(index) for index in member_weights.keys()]
            weights = np.asarray([float(value) for value in member_weights.values()], dtype=float)
            if weights.size and float(np.mean(weights)) < float(min_weight) * float(min_transition_affinity):
                continue
            current_rows = _component_object_chart_rows(
                object_id=f"{family.get('object_id', 'record_family')}_modular_{len(rows):06d}",
                indices=indices,
                weights=weights,
                h3_points=h3_points,
                axes=axes,
                chart_observer_ids=chart_observer_ids,
                global_h3_scale=global_h3_scale,
                global_s2_scale=global_s2_scale,
                min_observers_per_object=int(min_observers_per_object),
                split_h3_components=bool(split_h3_components),
                component_link_fraction=float(component_link_fraction),
                component_min_observers=component_min_observers,
            )
            for row in current_rows:
                row["cluster_key"] = {name: int(value) for name, value in key}
                row["family_mode"] = "record_family_modular_response_mixture"
                row["record_family_id"] = str(family.get("object_id", ""))
                row["record_signature"] = int(family.get("record_signature", -1))
                row["support_size"] = len(support)
                row["mean_observer_key_weight"] = float(np.mean(weights)) if weights.size else 0.0
                rows.append(row)
                if len(rows) >= int(max_objects):
                    return rows
        if len(rows) >= int(max_objects):
            break
    return rows


def _observer_transition_cluster_rows(
    chart_views: list[dict[str, Any]],
    chart_observer_ids: list[int],
    h3_points: np.ndarray,
    axes: np.ndarray,
    *,
    min_observers_per_object: int,
    max_objects: int,
    observer_cluster_fields: tuple[str, ...],
    max_observer_fraction: float,
    transition_affinity_score: str,
) -> list[dict[str, Any]]:
    """Build object rows from observer-visible transition/readout classes.

    This deliberately uses observer readouts rather than screen support overlap.
    Ubiquitous classes are filtered because an object that every observer sees
    is a background record, not a localized object candidate.
    """

    cluster_fields = tuple(field for field in observer_cluster_fields if field)
    if not cluster_fields:
        cluster_fields = ("record_family", "s3_sector_class", "repair_load_bucket")
    clusters: dict[tuple[tuple[str, int], ...], list[tuple[int, float]]] = {}
    for observer_index, view in enumerate(chart_views):
        key, weight = _observer_transition_cluster_key(
            view,
            cluster_fields=cluster_fields,
            score_mode=str(transition_affinity_score),
        )
        if not key:
            continue
        clusters.setdefault(key, []).append((observer_index, weight))
    total_observers = max(1, len(chart_views))
    max_count = max(1, int(np.floor(float(max_observer_fraction) * total_observers))) if max_observer_fraction > 0.0 else total_observers
    ordered = sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0]))
    global_h3_scale = _median_pairwise_h3_distance(h3_points)
    global_s2_scale = _median_pairwise_s2_distance(axes)
    rows: list[dict[str, Any]] = []
    for key, members in ordered:
        if len(members) < int(min_observers_per_object):
            continue
        if len(members) > max_count:
            continue
        indices = [int(index) for index, _weight in members]
        weights = np.asarray([float(weight) for _index, weight in members], dtype=float)
        row = _object_chart_row(
            object_id=f"observer_cluster_{len(rows):06d}",
            indices=indices,
            weights=weights,
            h3_points=h3_points,
            axes=axes,
            chart_observer_ids=chart_observer_ids,
            global_h3_scale=global_h3_scale,
            global_s2_scale=global_s2_scale,
            support_size=0,
        )
        row["cluster_key"] = {name: int(value) for name, value in key}
        row["family_mode"] = "observer_transition_cluster"
        rows.append(row)
        if len(rows) >= int(max_objects):
            break
    return rows


def _observer_transition_history_cluster_rows(
    chart_views: list[dict[str, Any]],
    chart_observer_ids: list[int],
    h3_points: np.ndarray,
    axes: np.ndarray,
    *,
    min_observers_per_object: int,
    max_objects: int,
    max_observer_fraction: float,
    history_window: int,
    min_persistence: int,
    split_h3_components: bool = False,
    component_link_fraction: float = 0.35,
    component_min_observers: int | None = None,
) -> list[dict[str, Any]]:
    """Build object rows from observer-visible transition-history keys.

    This is the production object-incidence path for the H3 population lane:
    it groups observers by persistent readout/transition histories, not by
    screen support overlap. Ubiquitous histories are filtered as background.
    """

    clusters: dict[int, list[tuple[int, float, int]]] = {}
    for observer_index, view in enumerate(chart_views):
        key = view.get("transition_history_key")
        if key is None:
            continue
        persistence = int(view.get("transition_history_persistence", 0))
        if persistence < int(min_persistence):
            continue
        mean_mass = float(view.get("transition_history_mean_modal_mass", 0.0))
        persistence_weight = float(persistence) / max(float(history_window), 1.0)
        weight = max(mean_mass, min(1.0, persistence_weight))
        clusters.setdefault(int(key), []).append((observer_index, weight, persistence))
    total_observers = max(1, len(chart_views))
    max_count = max(1, int(np.floor(float(max_observer_fraction) * total_observers))) if max_observer_fraction > 0.0 else total_observers
    ordered = sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0]))
    global_h3_scale = _median_pairwise_h3_distance(h3_points)
    global_s2_scale = _median_pairwise_s2_distance(axes)
    rows: list[dict[str, Any]] = []
    for key, members in ordered:
        if len(members) < int(min_observers_per_object):
            continue
        if len(members) > max_count:
            continue
        indices = [int(index) for index, _weight, _persistence in members]
        weights = np.asarray([float(weight) for _index, weight, _persistence in members], dtype=float)
        current_rows = _component_object_chart_rows(
            object_id=f"transition_history_{len(rows):06d}",
            indices=indices,
            weights=weights,
            h3_points=h3_points,
            axes=axes,
            chart_observer_ids=chart_observer_ids,
            global_h3_scale=global_h3_scale,
            global_s2_scale=global_s2_scale,
            min_observers_per_object=int(min_observers_per_object),
            split_h3_components=bool(split_h3_components),
            component_link_fraction=float(component_link_fraction),
            component_min_observers=component_min_observers,
        )
        for row in current_rows:
            row["cluster_key"] = {"transition_history_key": int(key)}
            row["family_mode"] = "transition_history"
            row["mean_history_persistence"] = float(np.mean([persistence for _index, _weight, persistence in members]))
            row["mean_observer_key_weight"] = float(np.mean(weights)) if weights.size else 0.0
            rows.append(row)
            if len(rows) >= int(max_objects):
                break
        if len(rows) >= int(max_objects):
            break
    return rows


def _observer_transition_mixture_cluster_rows(
    chart_views: list[dict[str, Any]],
    chart_observer_ids: list[int],
    h3_points: np.ndarray,
    axes: np.ndarray,
    *,
    min_observers_per_object: int,
    max_objects: int,
    observer_cluster_fields: tuple[str, ...],
    max_observer_fraction: float,
    transition_affinity_score: str,
    top_k: int,
    min_weight: float,
    split_h3_components: bool = False,
    component_link_fraction: float = 0.35,
    component_min_observers: int | None = None,
) -> list[dict[str, Any]]:
    """Build object rows from weighted observer-visible packet mixtures.

    The dominant-key lane can erase small local objects because each observer
    support contributes only one modal descriptor. This lane lets a support
    contribute several visible packet/transition classes with their histogram
    weights. The same incidence is then evaluated under shuffled H3 observer
    labels, so a pass still requires geometric coherence in the derived chart.
    """

    cluster_fields = tuple(field for field in observer_cluster_fields if field)
    if not cluster_fields:
        cluster_fields = ("object_packet", "record_family", "cumulative_repair_load_bucket")
    clusters: dict[tuple[tuple[str, int], ...], dict[int, float]] = {}
    for observer_index, view in enumerate(chart_views):
        for key, weight in _observer_transition_mixture_keys(
            view,
            cluster_fields=cluster_fields,
            score_mode=str(transition_affinity_score),
            top_k=int(top_k),
            min_weight=float(min_weight),
        ):
            previous = clusters.setdefault(key, {}).get(observer_index, 0.0)
            clusters[key][observer_index] = max(float(previous), float(weight))
    total_observers = max(1, len(chart_views))
    max_count = max(1, int(np.floor(float(max_observer_fraction) * total_observers))) if max_observer_fraction > 0.0 else total_observers
    ordered = sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0]))
    global_h3_scale = _median_pairwise_h3_distance(h3_points)
    global_s2_scale = _median_pairwise_s2_distance(axes)
    rows: list[dict[str, Any]] = []
    for key, member_weights in ordered:
        if len(member_weights) < int(min_observers_per_object):
            continue
        if len(member_weights) > max_count:
            continue
        indices = [int(index) for index in member_weights.keys()]
        weights = np.asarray([float(value) for value in member_weights.values()], dtype=float)
        if float(np.mean(weights)) < float(min_weight):
            continue
        current_rows = _component_object_chart_rows(
            object_id=f"observer_mixture_cluster_{len(rows):06d}",
            indices=indices,
            weights=weights,
            h3_points=h3_points,
            axes=axes,
            chart_observer_ids=chart_observer_ids,
            global_h3_scale=global_h3_scale,
            global_s2_scale=global_s2_scale,
            min_observers_per_object=int(min_observers_per_object),
            split_h3_components=bool(split_h3_components),
            component_link_fraction=float(component_link_fraction),
            component_min_observers=component_min_observers,
        )
        for row in current_rows:
            row["cluster_key"] = {name: int(value) for name, value in key}
            row["family_mode"] = "observer_transition_mixture_cluster"
            row["mean_observer_key_weight"] = float(np.mean(weights)) if weights.size else 0.0
            rows.append(row)
            if len(rows) >= int(max_objects):
                break
        if len(rows) >= int(max_objects):
            break
    return rows


def _observer_transition_mixture_keys(
    view: dict[str, Any],
    *,
    cluster_fields: tuple[str, ...],
    score_mode: str,
    top_k: int,
    min_weight: float,
) -> list[tuple[tuple[tuple[str, int], ...], float]]:
    field_choices: list[list[tuple[str, int, float]]] = []
    for field in cluster_fields:
        histogram = _observer_transition_histogram(view, str(field))
        if not histogram:
            continue
        choices = _top_histogram_choices(histogram, field=str(field), top_k=int(top_k), min_weight=float(min_weight))
        if choices:
            field_choices.append(choices)
    if not field_choices:
        return []
    rows: list[tuple[tuple[tuple[str, int], ...], float]] = []
    max_products = 64
    for combo_index, combo in enumerate(product(*field_choices)):
        if combo_index >= max_products:
            break
        parts = tuple((str(name), int(value)) for name, value, _mass in combo)
        weight = _combine_affinity_scores([float(mass) for _name, _value, mass in combo], mode=str(score_mode))
        if weight >= float(min_weight):
            rows.append((parts, weight))
    return rows


def _observer_transition_histogram(view: dict[str, Any], field: str) -> dict[str, float]:
    if field == "object_packet":
        histogram = view.get("object_packet_histogram", {})
        return histogram if isinstance(histogram, dict) else {}
    histograms = view.get("transition_affinity_histograms", {})
    if isinstance(histograms, dict):
        histogram = histograms.get(str(field), {})
        if isinstance(histogram, dict) and histogram:
            return histogram
    response_histograms = view.get("modular_response_histograms", {})
    if isinstance(response_histograms, dict):
        histogram = response_histograms.get(str(field), {})
        if isinstance(histogram, dict) and histogram:
            return histogram
    history_histograms = view.get("transition_history_histograms", {})
    if not isinstance(history_histograms, dict):
        return {}
    histogram = history_histograms.get(str(field), {})
    return histogram if isinstance(histogram, dict) else {}


def _top_histogram_choices(
    histogram: dict[str, float],
    *,
    field: str,
    top_k: int,
    min_weight: float,
) -> list[tuple[str, int, float]]:
    rows: list[tuple[str, int, float]] = []
    for key, value in sorted(histogram.items(), key=lambda item: (-float(item[1]), str(item[0]))):
        mass = float(value)
        if mass < float(min_weight):
            continue
        rows.append((str(field), int(key), mass))
        if len(rows) >= max(1, int(top_k)):
            break
    return rows


def _observer_transition_cluster_key(
    view: dict[str, Any],
    *,
    cluster_fields: tuple[str, ...],
    score_mode: str,
) -> tuple[tuple[tuple[str, int], ...], float]:
    dominants = view.get("transition_affinity_dominants", {})
    histograms = view.get("transition_affinity_histograms", {})
    if not isinstance(dominants, dict) or not isinstance(histograms, dict):
        return (), 0.0
    parts: list[tuple[str, int]] = []
    score_terms: list[float] = []
    for field in cluster_fields:
        value = dominants.get(str(field))
        if value is None:
            continue
        int_value = int(value)
        parts.append((str(field), int_value))
        score_terms.append(_histogram_value(histograms.get(str(field), {}), int_value))
    if not parts:
        packet = view.get("dominant_object_packet")
        if packet is None:
            return (), 0.0
        int_packet = int(packet)
        parts.append(("object_packet", int_packet))
        score_terms.append(_histogram_value(view.get("object_packet_histogram", {}), int_packet))
    return tuple(parts), _combine_affinity_scores(score_terms, mode=str(score_mode))


def _object_chart_row(
    *,
    object_id: str,
    indices: list[int],
    weights: np.ndarray,
    h3_points: np.ndarray,
    axes: np.ndarray,
    chart_observer_ids: list[int],
    global_h3_scale: float,
    global_s2_scale: float,
    support_size: int,
) -> dict[str, Any]:
    object_h3 = _weighted_h3_mean(h3_points[indices], weights)
    object_axis = _weighted_s2_mean(axes[indices], weights)
    h3_compactness = _weighted_h3_radius(h3_points[indices], object_h3, weights)
    s2_compactness = _weighted_s2_radius(axes[indices], object_axis, weights)
    return {
        "object_id": str(object_id),
        "observer_count": int(len(indices)),
        "support_size": int(support_size),
        "h3_point": [float(value) for value in object_h3],
        "h3_spatial_point": [float(value) for value in object_h3[1:]],
        "h3_compactness": float(h3_compactness),
        "h3_compactness_normalized": float(h3_compactness / max(global_h3_scale, 1e-12)),
        "s2_boundary_compactness": float(s2_compactness),
        "s2_boundary_compactness_normalized": float(s2_compactness / max(global_s2_scale, 1e-12)),
        "observer_ids_sample": [int(chart_observer_ids[index]) for index in indices[:32]],
    }


def _component_object_chart_rows(
    *,
    object_id: str,
    indices: list[int],
    weights: np.ndarray,
    h3_points: np.ndarray,
    axes: np.ndarray,
    chart_observer_ids: list[int],
    global_h3_scale: float,
    global_s2_scale: float,
    min_observers_per_object: int,
    split_h3_components: bool,
    component_link_fraction: float,
    component_min_observers: int | None,
) -> list[dict[str, Any]]:
    indices = [int(index) for index in indices]
    weights = np.asarray(weights, dtype=float)
    if not split_h3_components or len(indices) < 2 * int(min_observers_per_object):
        return [
            _object_chart_row(
                object_id=str(object_id),
                indices=indices,
                weights=weights,
                h3_points=h3_points,
                axes=axes,
                chart_observer_ids=chart_observer_ids,
                global_h3_scale=global_h3_scale,
                global_s2_scale=global_s2_scale,
                support_size=0,
            )
        ]
    components = _h3_local_components(
        h3_points[np.asarray(indices, dtype=np.int64)],
        link_radius=float(component_link_fraction) * max(float(global_h3_scale), 1e-12),
    )
    min_count = int(component_min_observers) if component_min_observers is not None else int(min_observers_per_object)
    rows: list[dict[str, Any]] = []
    for component_index, component in enumerate(components):
        if len(component) < min_count:
            continue
        component_indices = [indices[int(local_index)] for local_index in component]
        component_weights = weights[np.asarray(component, dtype=np.int64)]
        row = _object_chart_row(
            object_id=f"{object_id}_component_{component_index:03d}",
            indices=component_indices,
            weights=component_weights,
            h3_points=h3_points,
            axes=axes,
            chart_observer_ids=chart_observer_ids,
            global_h3_scale=global_h3_scale,
            global_s2_scale=global_s2_scale,
            support_size=0,
        )
        row["component_split"] = True
        row["parent_observer_count"] = int(len(indices))
        row["component_index"] = int(component_index)
        rows.append(row)
    if rows:
        return rows
    return [
        _object_chart_row(
            object_id=str(object_id),
            indices=indices,
            weights=weights,
            h3_points=h3_points,
            axes=axes,
            chart_observer_ids=chart_observer_ids,
            global_h3_scale=global_h3_scale,
            global_s2_scale=global_s2_scale,
            support_size=0,
        )
    ]


def _h3_local_components(points: np.ndarray, *, link_radius: float) -> list[list[int]]:
    points = np.asarray(points, dtype=float)
    if points.shape[0] == 0:
        return []
    if points.shape[0] == 1:
        return [[0]]
    distance = h3_distance_matrix(points)
    adjacency = distance <= max(float(link_radius), 1e-12)
    np.fill_diagonal(adjacency, True)
    seen = np.zeros(points.shape[0], dtype=bool)
    components: list[list[int]] = []
    for start in range(points.shape[0]):
        if seen[start]:
            continue
        stack = [int(start)]
        seen[start] = True
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(int(node))
            neighbors = np.flatnonzero(adjacency[node] & ~seen)
            for neighbor in neighbors.tolist():
                seen[int(neighbor)] = True
                stack.append(int(neighbor))
        components.append(sorted(component))
    components.sort(key=lambda row: (-len(row), row[0] if row else -1))
    return components


def _weighted_h3_mean(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if points.size == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    weights = np.maximum(weights, 0.0)
    total = float(np.sum(weights))
    if total <= 1e-12:
        weights = np.ones(points.shape[0], dtype=float)
        total = float(points.shape[0])
    mean = np.sum(points * weights[:, None], axis=0) / total
    spatial = mean[1:]
    projected = np.empty(4, dtype=float)
    projected[1:] = spatial
    projected[0] = float(np.sqrt(1.0 + float(np.dot(spatial, spatial))))
    return projected


def _weighted_s2_mean(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if points.size == 0:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    weights = np.maximum(weights, 0.0)
    total = float(np.sum(weights))
    if total <= 1e-12:
        weights = np.ones(points.shape[0], dtype=float)
        total = float(points.shape[0])
    mean = np.sum(points * weights[:, None], axis=0) / total
    norm = float(np.linalg.norm(mean))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return mean / norm


def _weighted_h3_radius(points: np.ndarray, center: np.ndarray, weights: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    distances = np.asarray([_h3_pair_distance(center, row) for row in points], dtype=float)
    return _weighted_mean_1d(distances, weights)


def _weighted_s2_radius(points: np.ndarray, center: np.ndarray, weights: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    dots = np.clip(np.asarray(points, dtype=float) @ np.asarray(center, dtype=float), -1.0, 1.0)
    distances = np.arccos(dots)
    return _weighted_mean_1d(distances, weights)


def _weighted_mean_1d(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    total = float(np.sum(weights))
    if values.size == 0:
        return 0.0
    if total <= 1e-12:
        return float(np.mean(values))
    return float(np.sum(values * weights) / total)


def _median_pairwise_h3_distance(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if points.shape[0] < 2:
        return 1.0
    distance = h3_distance_matrix(points)
    upper = distance[np.triu_indices(points.shape[0], k=1)]
    return float(np.median(upper)) if upper.size else 1.0


def _median_pairwise_s2_distance(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if points.shape[0] < 2:
        return 1.0
    dots = np.clip(points @ points.T, -1.0, 1.0)
    distance = np.arccos(dots)
    upper = distance[np.triu_indices(points.shape[0], k=1)]
    return float(np.median(upper)) if upper.size else 1.0


def _object_chart_compactness_rows(
    chart_views: list[dict[str, Any]],
    chart_observer_ids: list[int],
    h3_points: np.ndarray,
    axes: np.ndarray,
    record_families: list[dict[str, Any]],
    *,
    min_observers_per_object: int,
    incidence_mode: str = "transition_history",
    min_packet_mass: float = 0.05,
    min_transition_affinity: float = 0.25,
    transition_affinity_score: str = "geometric_mean",
) -> list[dict[str, Any]]:
    global_h3_scale = _median_pairwise_h3_distance(h3_points)
    global_s2_scale = _median_pairwise_s2_distance(axes)
    rows: list[dict[str, Any]] = []
    for family in record_families:
        indices, weights = _object_incidence_indices(
            chart_views,
            family,
            mode=str(incidence_mode),
            min_packet_mass=float(min_packet_mass),
            min_transition_affinity=float(min_transition_affinity),
            transition_affinity_score=str(transition_affinity_score),
        )
        if len(indices) < int(min_observers_per_object):
            continue
        weights_array = np.asarray(weights, dtype=float)
        object_h3 = _weighted_h3_mean(h3_points[indices], weights_array)
        object_axis = _weighted_s2_mean(axes[indices], weights_array)
        h3_compactness = _weighted_h3_radius(h3_points[indices], object_h3, weights_array)
        s2_compactness = _weighted_s2_radius(axes[indices], object_axis, weights_array)
        rows.append(
            {
                "object_id": str(family.get("object_id", len(rows))),
                "observer_count": int(len(indices)),
                "h3_compactness_normalized": float(h3_compactness / max(global_h3_scale, 1e-12)),
                "s2_boundary_compactness_normalized": float(s2_compactness / max(global_s2_scale, 1e-12)),
                "observer_ids_sample": [int(chart_observer_ids[index]) for index in indices[:32]],
            }
        )
    return rows
