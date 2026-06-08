from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist, squareform

from oph_fpe.bulk.h3_chart import h3_distance_matrix, random_h3_points
from oph_fpe.claims import DEBUG, with_claim_metadata

FORBIDDEN_BULK_FEATURES = {
    "axis",
    "support_nodes",
    "cap_membership",
    "cap_axis",
    "s2_centroid",
    "s2_boundary_compactness",
    "raw_screen_distance",
    "screen_distance",
    "radial_depth",
    "modular_depth",
}

BLIND_OBSERVER_FEATURE_KEYS = (
    "record_transition_histogram",
    "checkpoint_class_transition",
    "perturb_resettle_signature",
    "counterfactual_stability",
    "sector_change_signature",
    "repair_response_spectrum",
)


def observer_similarity_matrix(
    observer_views: list[dict[str, Any]],
    record_families: list[dict[str, Any]] | None = None,
    cap_responses: dict[str, Any] | None = None,
    *,
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Build a neutral observer-view similarity matrix without radial-depth coordinates."""

    components, ids = observer_similarity_components(observer_views, record_families, cap_responses, weights=weights)
    return components.get("composite", np.zeros((0, 0), dtype=float)), ids


def observer_similarity_components(
    observer_views: list[dict[str, Any]],
    record_families: list[dict[str, Any]] | None = None,
    cap_responses: dict[str, Any] | None = None,
    *,
    weights: dict[str, float] | None = None,
) -> tuple[dict[str, np.ndarray], list[int]]:
    """Return component similarities for the observer-similarity debug diagnostic.

    This is deliberately not a theorem-side bulk-construction object. The
    component matrices are emitted so runs can show whether a 3D-looking value
    comes from a single heuristic blend rather than conformal cap geometry.
    """

    weights = weights or {
        "overlap_projection_agreement": 0.35,
        "record_family_mutual_information": 0.35,
        "cap_modular_response_similarity": 0.20,
        "counterfactual_response_similarity": 0.10,
    }
    patch_views = [view for view in observer_views if view.get("view_type") == "patch_observer"]
    ids = [int(view.get("observer_id", index)) for index, view in enumerate(patch_views)]
    if not patch_views:
        return {"composite": np.zeros((0, 0), dtype=float)}, []
    features = np.array([_view_feature(view) for view in patch_views], dtype=float)
    features = _standardize_columns(features)
    distances = squareform(pdist(features, metric="euclidean")) if len(patch_views) > 1 else np.zeros((1, 1))
    overlap_similarity = np.exp(-distances)
    hash_similarity = np.equal.outer(
        [view.get("visible_readout_hash") for view in patch_views],
        [view.get("visible_readout_hash") for view in patch_views],
    ).astype(float)
    record_similarity = _record_family_similarity(patch_views, record_families or {})
    if record_similarity is None:
        record_similarity = hash_similarity
    cap_similarity = _cap_response_similarity(patch_views, observer_views, cap_responses or {})
    if cap_similarity is None:
        cap_similarity = overlap_similarity
    counterfactual_similarity = _counterfactual_similarity(patch_views, record_families or {})
    if counterfactual_similarity is None:
        counterfactual_similarity = record_similarity
    overlap_weight = float(weights.get("overlap_projection_agreement", 0.35))
    record_weight = float(weights.get("record_family_mutual_information", 0.35))
    cap_weight = float(weights.get("cap_modular_response_similarity", 0.20))
    cf_weight = float(weights.get("counterfactual_response_similarity", 0.10))
    total = max(overlap_weight + record_weight + cap_weight + cf_weight, 1e-12)
    similarity = (
        overlap_weight * overlap_similarity
        + record_weight * record_similarity
        + cap_weight * cap_similarity
        + cf_weight * counterfactual_similarity
    ) / total
    np.fill_diagonal(similarity, 1.0)
    return {
        "composite": similarity,
        "overlap_projection": overlap_similarity,
        "record_family": record_similarity,
        "cap_response": cap_similarity,
        "counterfactual": counterfactual_similarity,
    }, ids


def observer_distance_matrix(similarity: np.ndarray, *, xi: float = 1.0, eps: float = 1e-9) -> np.ndarray:
    if similarity.size == 0:
        return np.zeros_like(similarity, dtype=float)
    c0 = max(float(np.max(similarity)), eps)
    distance = -float(xi) * np.log((np.asarray(similarity, dtype=float) + eps) / c0)
    np.fill_diagonal(distance, 0.0)
    return distance


def neutral_dimension_report_from_distance(distance: np.ndarray) -> dict[str, Any]:
    logfit = _correlation_logfit(distance, 0.001, 0.05)
    estimate = float(logfit["estimate"])
    points_used = int(logfit["points_used"])
    mle = _local_mle_dimension(distance)
    mle_sweep = {
        str(k): _nullable_float(_local_mle_dimension(distance, k=k))
        for k in (6, 8, 12, 16, 24, 32)
        if distance.shape[0] > k + 2
    }
    quantile_sweep = {
        f"{int(low * 100)}-{int(high * 100)}": _correlation_logfit(distance, low, high)
        for low, high in (
            (0.001, 0.05),
            (0.002, 0.08),
            (0.005, 0.10),
            (0.01, 0.15),
            (0.05, 0.30),
            (0.15, 0.75),
        )
    }
    provisional = {
        "correlation_dimension": {"estimate": estimate, "points_used": points_used},
        "correlation_dimension_logfit": {"estimate": estimate, "points_used": points_used},
        "local_mle_dimension": {"estimate": mle},
    }
    estimators_agree = _dimension_estimators_agree(provisional, tolerance=0.5)
    primary_estimate = estimate if estimators_agree and np.isfinite(estimate) else None
    report = {
        "mode": "neutral_summary_distance_diagnostic",
        "receipt": "NEUTRAL_SUMMARY_DISTANCE_DIAGNOSTIC",
        "distance_source": "neutral_observer_record_similarity",
        "radial_depth_used": False,
        "physics_claim": False,
        "physical_claim": False,
        "primary_dimension": {
            "estimate": primary_estimate,
            "reason": "set only when logfit and local-MLE diagnostics agree within tolerance",
        },
        "dimension_estimators_agree": bool(estimators_agree),
        "correlation_dimension": {"estimate": estimate, "points_used": points_used},
        "correlation_dimension_logfit": {"estimate": estimate, "points_used": points_used},
        "local_mle_dimension": {"estimate": mle},
        "local_mle_k_sweep": mle_sweep,
        "correlation_logfit_quantile_sweep": quantile_sweep,
        "claim_boundary": (
            "debug estimator on the current observer-similarity matrix; no physics interpretation "
            "and not a bulk-dimension receipt"
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=DEBUG,
        receipt="NEUTRAL_SUMMARY_DISTANCE_DIAGNOSTIC",
        physical_claim=False,
        observable_id="neutral_observer_record_similarity",
        fit_objective="debug_dimension_estimator",
    )


def bulk_reconstruction_report(
    observer_views: list[dict[str, Any]],
    record_families: list[dict[str, Any]] | None = None,
    cap_responses: dict[str, Any] | None = None,
    *,
    seed: int = 1,
) -> dict[str, Any]:
    components, ids = observer_similarity_components(observer_views, record_families, cap_responses)
    similarity = components.get("composite", np.zeros((0, 0), dtype=float))
    distance = observer_distance_matrix(similarity)
    dimension = neutral_dimension_report_from_distance(distance)
    component_reports = {
        name: neutral_dimension_report_from_distance(observer_distance_matrix(matrix))
        for name, matrix in components.items()
    }
    blind_report = blind_observer_bulk_report(observer_views)
    controls = neutral_reconstruction_controls(observer_views, seed=seed)
    report = {
        "mode": "neutral_summary_distance_diagnostic",
        "receipt": "NEUTRAL_SUMMARY_DISTANCE_DIAGNOSTIC",
        "observer_count": len(ids),
        "distance_matrix_shape": list(distance.shape),
        "observer_similarity_debug_report": dimension,
        "neutral_dimension_report": dimension,
        "component_dimension_debug_reports": component_reports,
        "blind_observer_bulk_report": blind_report,
        "controls": controls,
        "control_gate_passed": bool(controls.get("all_expected_failures_observed", False)),
        "dimension_estimators_agree": _dimension_estimators_agree(dimension, tolerance=0.5),
        "candidate_3d_dimension_window": _dimension_in_window(dimension, 2.7, 3.3),
        "bulk_3d_established": False,
        "claim_boundary": (
            "observer-similarity debug scaffold; no 3D-bulk claim. The theorem-aligned path is "
            "cap/BW conformal geometry -> H3 chart -> record/object fit into that chart. Strict "
            "blind observer features exclude S2 coordinates, support nodes, cap membership, and "
            "radial-depth priors."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=DEBUG,
        receipt="NEUTRAL_SUMMARY_DISTANCE_DIAGNOSTIC",
        physical_claim=False,
        observable_id="neutral_observer_record_similarity",
        fit_objective="debug_dimension_estimator",
    )


def planted_dimension_report(points: np.ndarray) -> dict[str, Any]:
    distance = squareform(pdist(np.asarray(points, dtype=float), metric="euclidean"))
    return neutral_dimension_report_from_distance(distance)


def neutral_reconstruction_controls(
    observer_views: list[dict[str, Any]],
    *,
    seed: int = 1,
    planted_count: int = 900,
) -> dict[str, Any]:
    planted = planted_dimension_controls(seed=seed, sample_count=planted_count)
    shuffled = shuffled_observer_record_control(observer_views, seed=seed + 17)
    control_rows = [*planted.values(), shuffled]
    return {
        "mode": "neutral_reconstruction_controls",
        "planted_dimensions": planted,
        "shuffled_observer_records": shuffled,
        "all_expected_failures_observed": bool(control_rows) and all(
            bool(row.get("expected_failure_observed")) for row in control_rows
        ),
        "radial_depth_used": False,
        "claim_boundary": (
            "controls for neutral observer-record reconstruction; planted dimensions must recover "
            "their source dimension and shuffled observer records must degrade the reconstruction"
        ),
    }


def build_blind_observer_features(observer_views: list[dict[str, Any]]) -> tuple[np.ndarray, list[int], dict[str, Any]]:
    """Build strict observer features that exclude S2/support/cap coordinates."""

    patch_views = [view for view in observer_views if view.get("view_type") == "patch_observer"]
    ids = [int(view.get("observer_id", index)) for index, view in enumerate(patch_views)]
    vectors: list[list[float]] = []
    used_key_counts = {key: 0 for key in BLIND_OBSERVER_FEATURE_KEYS}
    forbidden_seen = sorted(
        {
            str(key)
            for view in patch_views
            for key in view
            if str(key) in FORBIDDEN_BULK_FEATURES
        }
    )
    for view in patch_views:
        feature: list[float] = []
        clean_view = {key: view.get(key) for key in BLIND_OBSERVER_FEATURE_KEYS if key in view}
        assert_no_forbidden_keys(clean_view)
        for key in BLIND_OBSERVER_FEATURE_KEYS:
            if key not in clean_view:
                continue
            values = _flatten_numeric(clean_view[key])
            if values:
                used_key_counts[key] += 1
                feature.extend(values[:32])
        vectors.append(feature)
    width = max((len(vector) for vector in vectors), default=0)
    matrix = np.zeros((len(vectors), width), dtype=float)
    for row_index, vector in enumerate(vectors):
        if vector:
            matrix[row_index, : len(vector)] = np.asarray(vector, dtype=float)
    metadata = {
        "observer_count": len(patch_views),
        "feature_width": int(width),
        "used_key_counts": used_key_counts,
        "forbidden_input_keys_seen_but_not_used": forbidden_seen,
        "forbidden_feature_keys_used": [],
        "s2_leakage_audit_pass": False,
        "claim_boundary": (
            "strict blind observer features; support_nodes, axes, cap membership, S2 compactness, "
            "raw screen distance, radial depth, and modular depth are not used"
        ),
    }
    return matrix, ids, metadata


def assert_no_forbidden_keys(mapping: dict[str, Any]) -> None:
    forbidden = sorted(str(key) for key in mapping if str(key) in FORBIDDEN_BULK_FEATURES)
    if forbidden:
        raise ValueError(f"forbidden bulk evidence keys used: {', '.join(forbidden)}")


def blind_observer_bulk_report(observer_views: list[dict[str, Any]], *, leakage_threshold: float = 0.35) -> dict[str, Any]:
    features, ids, metadata = build_blind_observer_features(observer_views)
    if features.shape[0] < 8 or features.shape[1] < 2:
        return {
            "mode": "strict_blind_observer_bulk_audit",
            "observer_count": len(ids),
            "feature_width": int(features.shape[1]) if features.ndim == 2 else 0,
            "usable": False,
            "reason": "insufficient_blind_transition_features",
            "forbidden_feature_keys_used": [],
            "forbidden_input_keys_seen_but_not_used": metadata["forbidden_input_keys_seen_but_not_used"],
            "s2_leakage_audit_pass": False,
            "bulk_3d_established": False,
            "claim_boundary": metadata["claim_boundary"],
        }
    features = _standardize_columns(features)
    distance = squareform(pdist(features, metric="euclidean")) if features.shape[0] > 1 else np.zeros((1, 1))
    dimension = neutral_dimension_report_from_distance(distance)
    axes = _observer_axes(observer_views, ids)
    s2_corr = None
    s2_leakage_pass = False
    if axes is not None:
        s2_distance = squareform(pdist(axes, metric="euclidean"))
        s2_corr_raw = _shape_correlation(_upper_triangle(distance), _upper_triangle(s2_distance))
        s2_corr = s2_corr_raw if np.isfinite(s2_corr_raw) else None
        s2_leakage_pass = bool(s2_corr is not None and abs(float(s2_corr)) <= float(leakage_threshold))
    candidate = _dimension_in_window(dimension, 2.7, 3.3)
    return {
        "mode": "strict_blind_observer_bulk_audit",
        "observer_count": len(ids),
        "feature_width": int(features.shape[1]),
        "usable": True,
        "distance_source": "blind_transition_continuation_features",
        "neutral_dimension_report": dimension,
        "candidate_3d_dimension_window": bool(candidate),
        "s2_distance_correlation": s2_corr,
        "s2_leakage_threshold": float(leakage_threshold),
        "s2_leakage_audit_pass": s2_leakage_pass,
        "forbidden_feature_keys_used": metadata["forbidden_feature_keys_used"],
        "forbidden_input_keys_seen_but_not_used": metadata["forbidden_input_keys_seen_but_not_used"],
        "bulk_3d_established": False,
        "claim_boundary": (
            metadata["claim_boundary"]
            + "; this audit is necessary but not sufficient for a 3D bulk proof"
        ),
    }


def planted_dimension_controls(*, seed: int = 1, sample_count: int = 900) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    tolerances = {2: 0.35, 3: 0.45, 4: 0.65}
    controls: dict[str, Any] = {}
    for dimension, tolerance in tolerances.items():
        report = planted_dimension_report(rng.random((int(sample_count), dimension)))
        estimate = float(report["correlation_dimension"]["estimate"])
        controls[f"planted_{dimension}d"] = {
            "target_dimension": dimension,
            "estimate": estimate,
            "tolerance": tolerance,
            "expected_failure_observed": bool(np.isfinite(estimate) and abs(estimate - dimension) <= tolerance),
            "distance_source": report["distance_source"],
            "radial_depth_used": bool(report["radial_depth_used"]),
            "failure_mode": f"neutral estimator should recover planted {dimension}D point cloud",
        }
    s2_points = rng.normal(size=(int(sample_count), 3))
    s2_points = s2_points / np.maximum(np.linalg.norm(s2_points, axis=1, keepdims=True), 1e-12)
    s2_report = planted_dimension_report(s2_points)
    s2_estimate = float(s2_report["correlation_dimension"]["estimate"])
    controls["planted_s2_boundary"] = {
        "target_dimension": 2,
        "estimate": s2_estimate,
        "tolerance": 0.35,
        "expected_failure_observed": bool(np.isfinite(s2_estimate) and abs(s2_estimate - 2.0) <= 0.35),
        "distance_source": "s2_boundary_chord_distance",
        "radial_depth_used": False,
        "failure_mode": "neutral estimator should recognize planted S2 boundary data as boundary-like",
    }
    h3_points = random_h3_points(int(sample_count), seed=seed + 301, radius=1.2)
    h3_report = neutral_dimension_report_from_distance(h3_distance_matrix(h3_points))
    h3_estimate = float(h3_report["correlation_dimension"]["estimate"])
    controls["planted_h3_bulk"] = {
        "target_dimension": 3,
        "estimate": h3_estimate,
        "tolerance": 0.55,
        "expected_failure_observed": bool(np.isfinite(h3_estimate) and abs(h3_estimate - 3.0) <= 0.55),
        "distance_source": "h3_hyperbolic_distance",
        "radial_depth_used": False,
        "failure_mode": "neutral estimator should recover planted H3 bulk data as 3D",
    }
    return controls


def shuffled_observer_record_control(observer_views: list[dict[str, Any]], *, seed: int = 1) -> dict[str, Any]:
    patch_views = [dict(view) for view in observer_views if view.get("view_type") == "patch_observer"]
    if len(patch_views) < 4:
        return {
            "observer_count": len(patch_views),
            "expected_failure_observed": False,
            "failure_mode": "not enough observer views to shuffle",
        }
    rng = np.random.default_rng(seed)
    original_similarity, _ = observer_similarity_matrix(patch_views)
    original_distance = observer_distance_matrix(original_similarity)
    shuffled = [dict(view) for view in patch_views]
    for key in (
        "visible_readout_hash",
        "record_stability_mean",
        "repair_load_mean",
        "mismatch_density_mean",
        "visible_signature_entropy",
    ):
        values = [view.get(key) for view in shuffled]
        order = rng.permutation(len(values))
        for index, source in enumerate(order):
            shuffled[index][key] = values[int(source)]
    shuffled_similarity, _ = observer_similarity_matrix(shuffled)
    shuffled_distance = observer_distance_matrix(shuffled_similarity)
    original_vec = _upper_triangle(original_distance)
    shuffled_vec = _upper_triangle(shuffled_distance)
    corr = _shape_correlation(original_vec, shuffled_vec)
    mean_abs_delta = float(np.mean(np.abs(original_vec - shuffled_vec))) if original_vec.size else 0.0
    degraded = bool((not np.isfinite(corr) or corr < 0.995) and mean_abs_delta > 1e-9)
    return {
        "observer_count": len(patch_views),
        "distance_shape_correlation": corr if np.isfinite(corr) else None,
        "mean_abs_distance_delta": mean_abs_delta,
        "expected_failure_observed": degraded,
        "failure_mode": "shuffled observer records should degrade neutral observer-distance reconstruction",
    }


def _view_feature(view: dict[str, Any]) -> list[float]:
    return [
        float(view.get("committed_fraction", 0.0)),
        float(view.get("record_stability_mean", 0.0)),
        float(view.get("repair_load_mean", 0.0)),
        float(view.get("mismatch_density_mean", 0.0)),
        float(view.get("visible_signature_entropy", 0.0)),
    ]


def _flatten_numeric(value: Any) -> list[float]:
    if isinstance(value, dict):
        keys = sorted(value)
        return [
            float(value[key])
            for key in keys
            if isinstance(value.get(key), (int, float)) and np.isfinite(float(value[key]))
        ]
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value, dtype=object).reshape(-1)
        out: list[float] = []
        for item in array:
            if isinstance(item, (int, float)) and np.isfinite(float(item)):
                out.append(float(item))
        return out
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return [float(value)]
    return []


def _observer_axes(observer_views: list[dict[str, Any]], ids: list[int]) -> np.ndarray | None:
    by_id = {
        int(view.get("observer_id", index)): view
        for index, view in enumerate(observer_views)
        if view.get("view_type") == "patch_observer"
    }
    rows: list[np.ndarray] = []
    for observer_id in ids:
        axis = np.asarray(by_id.get(int(observer_id), {}).get("axis", []), dtype=float)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            return None
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            return None
        rows.append(axis / norm)
    return np.vstack(rows) if rows else None


def _record_family_similarity(
    patch_views: list[dict[str, Any]],
    record_families: list[dict[str, Any]] | dict[str, Any],
) -> np.ndarray | None:
    families = list(record_families) if isinstance(record_families, list) else []
    if not families:
        return None
    supports = [set(int(node) for node in view.get("support_nodes", [])) for view in patch_views]
    if not any(supports):
        return None
    matrix = np.zeros((len(patch_views), len(families)), dtype=float)
    for family_index, family in enumerate(families):
        family_support = set(int(node) for node in family.get("support_nodes", []))
        if not family_support:
            continue
        signature = float(int(family.get("record_signature", 0)) % 997) / 997.0
        for view_index, support in enumerate(supports):
            overlap = len(support & family_support)
            if overlap:
                matrix[view_index, family_index] = (overlap / len(family_support)) * (1.0 + 0.05 * signature)
    if not np.any(matrix):
        return None
    return _cosine_similarity_matrix(matrix)


def _cap_response_similarity(
    patch_views: list[dict[str, Any]],
    all_views: list[dict[str, Any]],
    cap_responses: dict[str, Any],
) -> np.ndarray | None:
    cap_views = [view for view in all_views if view.get("view_type") == "cap_observer"]
    if not cap_views:
        return None
    cap_quality = _cap_quality_vector(cap_views, cap_responses)
    matrix = np.zeros((len(patch_views), len(cap_views)), dtype=float)
    for patch_index, patch in enumerate(patch_views):
        axis = np.asarray(patch.get("axis", []), dtype=float)
        if axis.shape != (3,):
            continue
        for cap_index, cap in enumerate(cap_views):
            cap_axis = np.asarray(cap.get("axis", []), dtype=float)
            if cap_axis.shape != (3,):
                continue
            theta0 = float(cap.get("theta0", 0.0))
            collar_width = max(float(cap.get("collar_width", 1e-3)), 1e-3)
            membership = 1.0 / (1.0 + np.exp(-((float(np.dot(axis, cap_axis)) - math.cos(theta0)) / collar_width)))
            matrix[patch_index, cap_index] = membership * cap_quality[cap_index]
    if not np.any(matrix):
        return None
    return _cosine_similarity_matrix(matrix)


def _counterfactual_similarity(
    patch_views: list[dict[str, Any]],
    record_families: list[dict[str, Any]] | dict[str, Any],
) -> np.ndarray | None:
    families = list(record_families) if isinstance(record_families, list) else []
    if not families:
        return None
    supports = [set(int(node) for node in view.get("support_nodes", [])) for view in patch_views]
    matrix = np.zeros((len(patch_views), len(families)), dtype=float)
    for family_index, family in enumerate(families):
        stability = float(family.get("counterfactual_stability", 0.0))
        if stability <= 0.0:
            continue
        family_support = set(int(node) for node in family.get("support_nodes", []))
        if not family_support:
            continue
        for view_index, support in enumerate(supports):
            overlap = len(support & family_support)
            if overlap:
                matrix[view_index, family_index] = stability * overlap / len(family_support)
    if not np.any(matrix):
        return None
    return _cosine_similarity_matrix(matrix)


def _cap_quality_vector(cap_views: list[dict[str, Any]], cap_responses: dict[str, Any]) -> np.ndarray:
    quality = np.ones(len(cap_views), dtype=float)
    rows = cap_responses.get("rows", []) if isinstance(cap_responses, dict) else []
    if not rows:
        return quality
    by_cap: dict[int, list[float]] = {}
    for row in rows:
        cap_id = row.get("cap_id")
        residual = row.get("support_visible_residual", row.get("raw_residual"))
        if cap_id is None or residual is None:
            continue
        by_cap.setdefault(int(cap_id), []).append(float(residual))
    for index, cap in enumerate(cap_views):
        values = by_cap.get(int(cap.get("cap_index", index)), [])
        if values:
            quality[index] = 1.0 / (1.0 + float(np.median(values)))
    return quality


def _cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1)
    normalized = matrix / np.maximum(norms[:, None], 1e-12)
    similarity = normalized @ normalized.T
    similarity = np.clip(similarity, 0.0, 1.0)
    zero = norms < 1e-12
    if np.any(zero):
        similarity[zero, :] = 0.0
        similarity[:, zero] = 0.0
        similarity[zero, zero] = 1.0
    np.fill_diagonal(similarity, 1.0)
    return similarity


def _standardize_columns(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    std = np.std(values, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (values - np.mean(values, axis=0)) / std


def _correlation_logfit(distance: np.ndarray, low_quantile: float, high_quantile: float) -> dict[str, Any]:
    values = distance[np.triu_indices_from(distance, k=1)]
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size < 8:
        return {
            "estimate": float("nan"),
            "points_used": 0,
            "quantile_window": [float(low_quantile), float(high_quantile)],
        }
    radii = np.quantile(values, np.linspace(float(low_quantile), float(high_quantile), 8))
    counts = np.array([np.mean(values < radius) for radius in radii], dtype=float)
    mask = (radii > 0.0) & (counts > 0.0)
    if int(np.sum(mask)) >= 2:
        estimate = float(np.polyfit(np.log(radii[mask]), np.log(counts[mask]), 1)[0])
        points_used = int(np.sum(mask))
    else:
        estimate = float("nan")
        points_used = int(np.sum(mask))
    return {
        "estimate": estimate,
        "points_used": points_used,
        "quantile_window": [float(low_quantile), float(high_quantile)],
    }


def _nullable_float(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _dimension_in_window(report: dict[str, Any], low: float, high: float) -> bool:
    estimates = _dimension_estimates(report)
    if not estimates:
        return False
    return bool(all(low <= estimate <= high for estimate in estimates) and _dimension_estimators_agree(report, tolerance=0.5))


def _dimension_estimators_agree(report: dict[str, Any], tolerance: float) -> bool:
    estimates = _dimension_estimates(report)
    if len(estimates) < 2:
        return False
    return bool(max(estimates) - min(estimates) <= float(tolerance))


def _dimension_estimates(report: dict[str, Any]) -> list[float]:
    keys = ("correlation_dimension", "correlation_dimension_logfit", "local_mle_dimension")
    estimates: list[float] = []
    for key in keys:
        estimate = float(report.get(key, {}).get("estimate", float("nan")))
        if np.isfinite(estimate):
            estimates.append(estimate)
    return estimates


def _upper_triangle(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.zeros(0, dtype=float)
    return np.asarray(matrix[np.triu_indices_from(matrix, k=1)], dtype=float)


def _shape_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return float("nan")
    count = min(left.size, right.size)
    left = left[:count] - float(np.mean(left[:count]))
    right = right[:count] - float(np.mean(right[:count]))
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(left, right) / denom)


def _local_mle_dimension(distance: np.ndarray, k: int = 12) -> float:
    distance = np.asarray(distance, dtype=float)
    if distance.shape[0] < k + 2:
        return float("nan")
    sorted_distances = np.sort(np.where(distance > 0.0, distance, np.inf), axis=1)
    nearest = sorted_distances[:, :k]
    finite = np.all(np.isfinite(nearest), axis=1) & (nearest[:, -1] > 0.0)
    nearest = nearest[finite]
    if nearest.shape[0] == 0:
        return float("nan")
    ratios = np.log(nearest[:, [-1]] / np.maximum(nearest[:, :-1], 1e-15))
    estimates = (k - 1) / np.maximum(np.sum(ratios, axis=1), 1e-15)
    estimates = estimates[np.isfinite(estimates)]
    return float(np.median(estimates)) if estimates.size else float("nan")
