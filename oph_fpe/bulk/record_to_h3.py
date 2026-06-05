from __future__ import annotations

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


DEFAULT_RECORD_FIELDS = (
    "record_signature",
    "stable_count",
    "repair_load",
    "cumulative_repair_load",
    "s3_class_density",
    "s3_sector_class",
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
    incidence_mode: str = "support_overlap",
    min_packet_mass: float = 0.05,
    max_h3_compactness: float = 0.35,
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
    if not patch_views or not record_families or chart["points"].size == 0:
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
    object_rows: list[dict[str, Any]] = []
    for family in record_families[: int(max_objects)]:
        support = set(int(node) for node in family.get("support_nodes", []))
        if not support:
            continue
        indices, weights = _object_incidence_indices(
            chart_views,
            family,
            mode=str(incidence_mode),
            min_packet_mass=float(min_packet_mass),
        )
        if len(indices) < int(min_observers_per_object):
            continue
        object_h3 = _weighted_h3_mean(h3_points[indices], np.asarray(weights, dtype=float))
        object_axis = _weighted_s2_mean(axes[indices], np.asarray(weights, dtype=float))
        h3_compactness = _weighted_h3_radius(h3_points[indices], object_h3, np.asarray(weights, dtype=float))
        s2_compactness = _weighted_s2_radius(axes[indices], object_axis, np.asarray(weights, dtype=float))
        object_rows.append(
            {
                "object_id": str(family.get("object_id", len(object_rows))),
                "observer_count": int(len(indices)),
                "support_size": int(len(support)),
                "h3_point": [float(value) for value in object_h3],
                "h3_spatial_point": [float(value) for value in object_h3[1:]],
                "h3_compactness": float(h3_compactness),
                "h3_compactness_normalized": float(h3_compactness / max(global_h3_scale, 1e-12)),
                "s2_boundary_compactness": float(s2_compactness),
                "s2_boundary_compactness_normalized": float(s2_compactness / max(global_s2_scale, 1e-12)),
                "observer_ids_sample": [int(chart_observer_ids[index]) for index in indices[:32]],
            }
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
    shuffled_points = h3_points[rng.permutation(h3_points.shape[0])]
    shuffled_rows = _object_chart_compactness_rows(
        chart_views,
        chart_observer_ids,
        shuffled_points,
        axes,
        record_families[: int(max_objects)],
        min_observers_per_object=int(min_observers_per_object),
    )
    h3_norms = np.asarray([row["h3_compactness_normalized"] for row in object_rows], dtype=float)
    s2_norms = np.asarray([row["s2_boundary_compactness_normalized"] for row in object_rows], dtype=float)
    shuffled_norms = np.asarray(
        [row["h3_compactness_normalized"] for row in shuffled_rows],
        dtype=float,
    )
    median_h3 = float(np.median(h3_norms)) if h3_norms.size else float("nan")
    median_s2 = float(np.median(s2_norms)) if s2_norms.size else float("nan")
    median_shuffled = float(np.median(shuffled_norms)) if shuffled_norms.size else float("nan")
    h3_beats_shuffle = bool(np.isfinite(median_h3) and np.isfinite(median_shuffled) and median_h3 < float(pass_ratio) * median_shuffled)
    h3_not_boundary_dominated = bool(np.isfinite(median_h3) and np.isfinite(median_s2) and median_h3 <= median_s2 / max(float(pass_ratio), 1e-12))
    h3_localized = bool(np.isfinite(median_h3) and median_h3 <= float(max_h3_compactness))
    eligible = bool(len(object_rows) >= int(min_objects))
    chart_receipt = bool(eligible and h3_beats_shuffle)
    bulk_receipt = bool(
        chart_receipt
        and h3_localized
        and h3_not_boundary_dominated
        and bool(h3_report.get("MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT", False))
    )
    return {
        "mode": "observer_chart_object_h3_population",
        "observer_count": int(h3_points.shape[0]),
        "object_count": int(len(object_rows)),
        "min_objects": int(min_objects),
        "min_observers_per_object": int(min_observers_per_object),
        "incidence_mode": str(incidence_mode),
        "min_packet_mass": float(min_packet_mass),
        "max_h3_compactness": float(max_h3_compactness),
        "global_h3_pairwise_median": float(global_h3_scale),
        "global_s2_pairwise_median": float(global_s2_scale),
        "median_h3_compactness_normalized": median_h3 if np.isfinite(median_h3) else None,
        "median_s2_boundary_compactness_normalized": median_s2 if np.isfinite(median_s2) else None,
        "median_shuffled_h3_compactness_normalized": median_shuffled if np.isfinite(median_shuffled) else None,
        "h3_beats_shuffled_incidence": h3_beats_shuffle,
        "h3_not_boundary_dominated": h3_not_boundary_dominated,
        "h3_localized": h3_localized,
        "modular_response_h3_receipt": bool(h3_report.get("MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT", False)),
        "observer_chart_object_h3_receipt": chart_receipt,
        "observer_chart_bulk_population_receipt": bulk_receipt,
        "sample_objects": object_rows[:256],
        "claim_boundary": (
            "places persistent observer-facing record objects into the observer-derived modular-response "
            "H3 chart using only sampled observers that see the object. A chart receipt requires compactness "
            "against shuffled observer-object incidence. A populated-bulk receipt additionally requires the "
            "modular-response H3 chart gate and non-boundary-dominated compactness. This is not a CMB or "
            "particle claim."
        ),
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
        "claim_boundary": (
            "support-visible cap-profile fit into H3 for record families or screen holonomy defects. "
            "A pass is a populated-chart diagnostic only; particle claims still require persistence, "
            "transport, fusion/scattering, and worldline controls."
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
) -> list[dict[str, Any]]:
    global_h3_scale = _median_pairwise_h3_distance(h3_points)
    global_s2_scale = _median_pairwise_s2_distance(axes)
    rows: list[dict[str, Any]] = []
    for family in record_families:
        support = set(int(node) for node in family.get("support_nodes", []))
        if not support:
            continue
        weights = []
        indices = []
        for observer_index, view in enumerate(chart_views):
            view_support = set(int(node) for node in view.get("support_nodes", []))
            overlap = len(support & view_support)
            if overlap <= 0:
                continue
            packet_mass = _histogram_value(
                view.get("object_packet_histogram") or view.get("record_signature_histogram", {}),
                int(family.get("record_signature", -1)),
            )
            weight = float(overlap / max(1, len(support)))
            if packet_mass > 0.0:
                weight *= 0.5 + 0.5 * float(packet_mass)
            weights.append(weight)
            indices.append(observer_index)
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
