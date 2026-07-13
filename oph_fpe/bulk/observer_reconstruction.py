from __future__ import annotations

import copy
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

BLIND_FEATURE_GROUPS = {
    "record_transition_histogram": ("record_transition_histogram",),
    "repair_response_spectrum": ("repair_response_spectrum",),
    "transition_continuation_core": (
        "record_transition_histogram",
        "checkpoint_class_transition",
        "sector_change_signature",
    ),
}


DEFAULT_OBSERVER_SIMILARITY_WEIGHTS = {
    "local_packet": 0.50,
    "measured_overlap": 0.35,
    "paired_perturb_resettle": 0.15,
    # These legacy diagnostics remain callable with explicit nonzero weights,
    # but they no longer enter the default geometry instrument.
    "record_family": 0.0,
    "cap_response": 0.0,
    "counterfactual": 0.0,
}


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

    weights = dict(DEFAULT_OBSERVER_SIMILARITY_WEIGHTS if weights is None else weights)
    patch_views = [view for view in observer_views if view.get("view_type") == "patch_observer"]
    ids = [int(view.get("observer_id", index)) for index, view in enumerate(patch_views)]
    if not patch_views:
        return {"composite": np.zeros((0, 0), dtype=float)}, []

    count = len(patch_views)
    unavailable = np.eye(count, dtype=float)
    local_features = _verified_local_packet_feature_matrix(patch_views)
    local_similarity = (
        _feature_block_similarity(local_features)
        if local_features is not None
        else unavailable.copy()
    )
    measured_overlap_features = _verified_measured_overlap_feature_matrix(patch_views)
    measured_overlap_similarity = (
        _feature_block_similarity(measured_overlap_features)
        if measured_overlap_features is not None
        else unavailable.copy()
    )
    paired_features = _verified_paired_response_feature_matrix(patch_views)
    paired_similarity = (
        _feature_block_similarity(paired_features)
        if paired_features is not None
        else unavailable.copy()
    )

    # The old record/cap/counterfactual matrices are retained only as explicit
    # debug lanes. In particular there is no visible-hash fallback: token/hash
    # equality is not a locality metric.
    record_similarity = _record_family_similarity(patch_views, record_families or {})
    record_available = record_similarity is not None
    if record_similarity is None:
        record_similarity = unavailable.copy()
    cap_similarity = _cap_response_similarity(patch_views, observer_views, cap_responses or {})
    cap_available = cap_similarity is not None
    if cap_similarity is None:
        cap_similarity = unavailable.copy()
    counterfactual_similarity = _counterfactual_similarity(patch_views, record_families or {})
    counterfactual_available = counterfactual_similarity is not None
    if counterfactual_similarity is None:
        counterfactual_similarity = unavailable.copy()

    components = {
        "local_packet": local_similarity,
        "measured_overlap": measured_overlap_similarity,
        "paired_perturb_resettle": paired_similarity,
        "record_family": record_similarity,
        "cap_response": cap_similarity,
        "counterfactual": counterfactual_similarity,
    }
    available = {
        "local_packet": local_features is not None,
        "measured_overlap": measured_overlap_features is not None,
        "paired_perturb_resettle": paired_features is not None,
        "record_family": record_available,
        "cap_response": cap_available,
        "counterfactual": counterfactual_available,
    }
    active = [
        (name, max(0.0, float(weights.get(name, 0.0))), matrix)
        for name, matrix in components.items()
        if available[name] and float(weights.get(name, 0.0)) > 0.0
    ]
    total = float(sum(weight for _, weight, _ in active))
    similarity = (
        sum(weight * matrix for _, weight, matrix in active) / total
        if total > 1e-12
        else unavailable.copy()
    )
    np.fill_diagonal(similarity, 1.0)
    return {
        "composite": similarity,
        **components,
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
    patch_views = [view for view in observer_views if view.get("view_type") == "patch_observer"]
    local_available = _verified_local_packet_feature_matrix(patch_views) is not None
    overlap_available = _verified_measured_overlap_feature_matrix(patch_views) is not None
    paired_available = _verified_paired_response_feature_matrix(patch_views) is not None
    instrument_blockers = []
    if not local_available:
        instrument_blockers.append("verified_local_packet_producer_unavailable_or_partial")
    if not overlap_available:
        instrument_blockers.append("measured_reciprocal_overlap_producer_unavailable_or_partial")
    if not paired_available:
        instrument_blockers.append("actual_paired_perturb_resettle_producer_unavailable_or_partial")
    if local_available or overlap_available:
        instrument_blockers.append("support_selection_carrier_has_s2_screen_chart_ancestry")
    if paired_available:
        instrument_blockers.append("paired_intervention_axes_have_s2_screen_chart_ancestry")
    report = {
        "mode": "neutral_summary_distance_diagnostic",
        "receipt": "NEUTRAL_SUMMARY_DISTANCE_DIAGNOSTIC",
        "observer_count": len(ids),
        "distance_matrix_shape": list(distance.shape),
        "observer_similarity_debug_report": dimension,
        "neutral_dimension_report": dimension,
        "component_dimension_debug_reports": component_reports,
        "geometry_instrument": {
            "default_channels": ["local_packet", "measured_overlap", "paired_perturb_resettle"],
            "verified_local_packet_available": local_available,
            "measured_reciprocal_overlap_available": overlap_available,
            "actual_paired_perturb_resettle_available": paired_available,
            "hash_or_token_geometry_used": False,
            "legacy_self_histogram_overlap_used": False,
            "strict_emergent_bulk_eligible": False,
            "blockers": instrument_blockers,
            "claim_boundary": (
                "The debug instrument now reads locality-preserving packets, literal reciprocal overlap "
                "correspondences, and actual paired perturb/resettle responses. Current support selection "
                "and cap interventions still descend from the S2 screen chart, so this instrument can "
                "diagnose geometry but cannot certify that the geometry emerged chart-blind."
            ),
        },
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

    return build_blind_observer_feature_matrix(observer_views, BLIND_OBSERVER_FEATURE_KEYS)


def build_blind_observer_feature_matrix(
    observer_views: list[dict[str, Any]],
    feature_keys: tuple[str, ...],
) -> tuple[np.ndarray, list[int], dict[str, Any]]:
    """Build a strict observer-feature matrix from a declared visible feature family."""

    patch_views = [view for view in observer_views if view.get("view_type") == "patch_observer"]
    ids = [int(view.get("observer_id", index)) for index, view in enumerate(patch_views)]
    vectors: list[list[float]] = []
    used_key_counts = {key: 0 for key in feature_keys}
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
        clean_view = {key: view.get(key) for key in feature_keys if key in view}
        assert_no_forbidden_keys(clean_view)
        for key in feature_keys:
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
        "feature_keys": list(feature_keys),
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
    feature_group_sweep = blind_observer_feature_group_sweep(
        observer_views,
        max_rank=6,
        leakage_threshold=float(leakage_threshold),
    )
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
            "blind_feature_group_sweep": feature_group_sweep,
            "strict_blind_record_transition_3d_candidate_receipt": bool(
                feature_group_sweep.get("record_transition_rank3_receipt", False)
            ),
            "bulk_3d_established": False,
            "claim_boundary": metadata["claim_boundary"],
        }
    features = _standardize_columns(features)
    distance = squareform(pdist(features, metric="euclidean")) if features.shape[0] > 1 else np.zeros((1, 1))
    dimension = neutral_dimension_report_from_distance(distance)
    low_rank = blind_observer_low_rank_sweep_from_features(
        features,
        observer_views,
        ids,
        leakage_threshold=float(leakage_threshold),
    )
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
        "low_rank_transition_chart_sweep": low_rank,
        "strict_blind_low_rank_3d_candidate_receipt": bool(
            low_rank.get("selected_rank_3d_candidate_receipt", False)
        ),
        "blind_feature_group_sweep": feature_group_sweep,
        "strict_blind_record_transition_3d_candidate_receipt": bool(
            feature_group_sweep.get("record_transition_rank3_receipt", False)
        ),
        "candidate_3d_dimension_window": bool(candidate),
        "s2_distance_correlation": s2_corr,
        "s2_leakage_threshold": float(leakage_threshold),
        "s2_leakage_audit_pass": s2_leakage_pass,
        "forbidden_feature_keys_used": metadata["forbidden_feature_keys_used"],
        "forbidden_input_keys_seen_but_not_used": metadata["forbidden_input_keys_seen_but_not_used"],
        "bulk_3d_established": False,
        "claim_boundary": (
            metadata["claim_boundary"]
            + "; this audit is necessary but not sufficient for a 3D bulk proof. The feature-group "
            "sweep reports predeclared visible subfamilies separately so a 3D-looking low-rank "
            "continuation cannot be hidden inside or manufactured by the full overcomplete vector."
        ),
    }


def blind_observer_feature_group_sweep(
    observer_views: list[dict[str, Any]],
    *,
    max_rank: int = 6,
    leakage_threshold: float = 0.35,
    predeclared_rank: int = 3,
) -> dict[str, Any]:
    """Audit predeclared strict-blind feature groups.

    The papers make records and continuation data primary. The full blind
    feature vector is intentionally overcomplete; this audit asks whether named
    observer-visible subfamilies carry a low-rank continuation without using
    support nodes, cap membership, S2 axes, radial depth, or modular depth.
    """

    group_reports: dict[str, Any] = {}
    for name, keys in BLIND_FEATURE_GROUPS.items():
        features, ids, metadata = build_blind_observer_feature_matrix(observer_views, tuple(keys))
        if features.shape[0] < 8 or features.shape[1] < 2:
            group_reports[name] = {
                "usable": False,
                "reason": "insufficient_blind_group_features",
                "feature_keys": list(keys),
                "feature_width": int(features.shape[1]) if features.ndim == 2 else 0,
                "rank3_candidate_receipt": False,
                "claim_boundary": "strict blind feature-group diagnostic only; no bulk claim",
            }
            continue
        standardized = _standardize_columns(features)
        low_rank = blind_observer_low_rank_sweep_from_features(
            standardized,
            observer_views,
            ids,
            max_rank=int(max_rank),
            leakage_threshold=float(leakage_threshold),
        )
        rank_report = next(
            (row for row in low_rank.get("rank_reports", []) if int(row.get("rank", -1)) == int(predeclared_rank)),
            {},
        )
        rank3_candidate = bool(
            rank_report.get("candidate_3d_dimension_window", False)
            and rank_report.get("dimension_estimators_agree", False)
            and rank_report.get("s2_leakage_audit_pass", False)
        )
        group_reports[name] = {
            "usable": bool(low_rank.get("usable", False)),
            "feature_keys": list(keys),
            "feature_width": int(features.shape[1]),
            "used_key_counts": metadata.get("used_key_counts", {}),
            "predeclared_rank": int(predeclared_rank),
            "rank3_candidate_receipt": rank3_candidate,
            "rank3_report": rank_report,
            "participation_rank": low_rank.get("participation_rank"),
            "entropy_rank": low_rank.get("entropy_rank"),
            "selected_rank": low_rank.get("selected_rank"),
            "selected_rank_3d_candidate_receipt": low_rank.get("selected_rank_3d_candidate_receipt"),
            "forbidden_feature_keys_used": metadata.get("forbidden_feature_keys_used", []),
            "forbidden_input_keys_seen_but_not_used": metadata.get("forbidden_input_keys_seen_but_not_used", []),
            "claim_boundary": (
                "strict blind feature-group low-rank diagnostic. A rank-3 candidate here is a "
                "predeclared observer-visible continuation signal, not a final third-person bulk proof."
            ),
        }
    record_transition = group_reports.get("record_transition_histogram", {})
    return {
        "mode": "strict_blind_observer_feature_group_sweep",
        "predeclared_rank": int(predeclared_rank),
        "group_reports": group_reports,
        "record_transition_rank3_receipt": bool(record_transition.get("rank3_candidate_receipt", False)),
        "record_transition_rank3_report": record_transition.get("rank3_report", {}),
        "physical_claim": False,
        "bulk_3d_established": False,
        "claim_boundary": (
            "Predeclared strict-blind observer-visible feature groups. This can establish a diagnostic "
            "rank-3 continuation candidate, but strict neutral bulk still requires refinement stability, "
            "control separation, object persistence, and theorem-aligned BW/collar gates."
        ),
    }


def blind_observer_low_rank_sweep(
    observer_views: list[dict[str, Any]],
    *,
    max_rank: int = 8,
    leakage_threshold: float = 0.35,
) -> dict[str, Any]:
    """Diagnostic low-rank sweep for strict blind observer features.

    The raw blind feature vector often carries several redundant visible
    transition/readout coordinates. This diagnostic asks whether an intrinsic
    low-rank continuation is visible without using support nodes, axes, cap
    membership, radial depth, or modular-depth coordinates. It is not used as a
    final bulk proof because selecting a rank can itself bias a dimension result.
    """

    features, ids, _metadata = build_blind_observer_features(observer_views)
    if features.shape[0] < 8 or features.shape[1] < 2:
        return {
            "mode": "strict_blind_observer_low_rank_transition_chart_sweep",
            "usable": False,
            "reason": "insufficient_blind_transition_features",
            "selected_rank_3d_candidate_receipt": False,
            "claim_boundary": "diagnostic only; no bulk claim",
        }
    features = _standardize_columns(features)
    return blind_observer_low_rank_sweep_from_features(
        features,
        observer_views,
        ids,
        max_rank=int(max_rank),
        leakage_threshold=float(leakage_threshold),
    )


def blind_observer_low_rank_sweep_from_features(
    features: np.ndarray,
    observer_views: list[dict[str, Any]],
    ids: list[int],
    *,
    max_rank: int = 8,
    leakage_threshold: float = 0.35,
) -> dict[str, Any]:
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[0] < 8 or features.shape[1] < 2:
        return {
            "mode": "strict_blind_observer_low_rank_transition_chart_sweep",
            "usable": False,
            "reason": "insufficient_feature_matrix",
            "selected_rank_3d_candidate_receipt": False,
            "claim_boundary": "diagnostic only; no bulk claim",
        }
    centered = features - np.mean(features, axis=0, keepdims=True)
    try:
        u, singular_values, _vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return {
            "mode": "strict_blind_observer_low_rank_transition_chart_sweep",
            "usable": False,
            "reason": "svd_failed",
            "selected_rank_3d_candidate_receipt": False,
            "claim_boundary": "diagnostic only; no bulk claim",
        }
    eigenvalues = singular_values**2
    total = float(np.sum(eigenvalues))
    if total <= 1e-15:
        return {
            "mode": "strict_blind_observer_low_rank_transition_chart_sweep",
            "usable": False,
            "reason": "degenerate_feature_matrix",
            "selected_rank_3d_candidate_receipt": False,
            "claim_boundary": "diagnostic only; no bulk claim",
        }
    explained = eigenvalues / total
    cumulative = np.cumsum(explained)
    participation_rank = float(total**2 / max(float(np.sum(eigenvalues**2)), 1e-15))
    entropy_rank = float(np.exp(-np.sum(explained[explained > 0.0] * np.log(explained[explained > 0.0]))))
    limit = min(max(2, int(max_rank)), u.shape[1], features.shape[0] - 2)
    axes = _observer_axes(observer_views, ids)
    rank_rows: list[dict[str, Any]] = []
    for rank in range(2, limit + 1):
        coords = u[:, :rank] * singular_values[:rank]
        distance = squareform(pdist(coords, metric="euclidean")) if coords.shape[0] > 1 else np.zeros((1, 1))
        dimension = neutral_dimension_report_from_distance(distance)
        s2_corr = None
        s2_pass = False
        if axes is not None:
            s2_distance = squareform(pdist(axes, metric="euclidean"))
            s2_corr_raw = _shape_correlation(_upper_triangle(distance), _upper_triangle(s2_distance))
            s2_corr = s2_corr_raw if np.isfinite(s2_corr_raw) else None
            s2_pass = bool(s2_corr is not None and abs(float(s2_corr)) <= float(leakage_threshold))
        rank_rows.append(
            {
                "rank": int(rank),
                "explained_variance": float(cumulative[rank - 1]) if rank - 1 < cumulative.size else 1.0,
                "candidate_3d_dimension_window": _dimension_in_window(dimension, 2.7, 3.3),
                "dimension_estimators_agree": _dimension_estimators_agree(dimension, tolerance=0.5),
                "correlation_dimension": dimension.get("correlation_dimension"),
                "local_mle_dimension": dimension.get("local_mle_dimension"),
                "primary_dimension": dimension.get("primary_dimension"),
                "s2_distance_correlation": s2_corr,
                "s2_leakage_audit_pass": s2_pass,
            }
        )
    selected_rank = int(np.clip(round(participation_rank), 2, limit))
    selected = next((row for row in rank_rows if int(row["rank"]) == selected_rank), rank_rows[-1])
    selected_rank_3d = bool(
        selected.get("candidate_3d_dimension_window", False)
        and selected.get("s2_leakage_audit_pass", False)
        and 2.5 <= participation_rank <= 3.5
    )
    return {
        "mode": "strict_blind_observer_low_rank_transition_chart_sweep",
        "usable": True,
        "observer_count": int(features.shape[0]),
        "feature_width": int(features.shape[1]),
        "singular_values": [float(value) for value in singular_values[: min(16, singular_values.size)]],
        "explained_variance_ratio": [float(value) for value in explained[: min(16, explained.size)]],
        "participation_rank": participation_rank,
        "entropy_rank": entropy_rank,
        "selected_rank": selected_rank,
        "selected_rank_report": selected,
        "rank_reports": rank_rows,
        "selected_rank_3d_candidate_receipt": selected_rank_3d,
        "physical_claim": False,
        "claim_boundary": (
            "strict blind low-rank transition-chart diagnostic. It removes redundant observer-visible "
            "transition features without using screen coordinates, but it is not a bulk proof because "
            "rank selection can bias dimension estimates. Use it to diagnose overcomplete observer "
            "features before any neutral 3D-bulk claim."
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
    patch_views = [copy.deepcopy(view) for view in observer_views if view.get("view_type") == "patch_observer"]
    if len(patch_views) < 4:
        return {
            "observer_count": len(patch_views),
            "expected_failure_observed": False,
            "failure_mode": "not enough observer views to shuffle",
        }
    rng = np.random.default_rng(seed)
    original_similarity, _ = observer_similarity_matrix(patch_views)
    original_distance = observer_distance_matrix(original_similarity)
    shuffled = [copy.deepcopy(view) for view in patch_views]
    channels_permuted: list[str] = []
    channel_fields = (
        ("local_packet", ("locality_preserving_packet_feature_vector",)),
        (
            "paired_perturb_resettle",
            ("paired_perturbation_response_tensor", "paired_perturbation_control_tensors"),
        ),
    )
    for channel, fields in channel_fields:
        if not any(any(field in view for field in fields) for view in shuffled):
            continue
        order = np.asarray(rng.permutation(len(shuffled)), dtype=np.int64)
        if len(shuffled) > 1 and np.array_equal(order, np.arange(len(shuffled))):
            order = np.roll(order, 1)
        for key in fields:
            values = [copy.deepcopy(view.get(key)) for view in shuffled]
            for index, source in enumerate(order):
                value = values[int(source)]
                if value is None:
                    shuffled[index].pop(key, None)
                else:
                    shuffled[index][key] = copy.deepcopy(value)
        channels_permuted.append(channel)
    if not channels_permuted:
        return {
            "observer_count": len(patch_views),
            "channels_permuted": [],
            "cross_observer_overlap_fixed": True,
            "expected_failure_observed": False,
            "failure_mode": "no active row-local measured geometry channels were available to shuffle",
        }
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
        "channels_permuted": channels_permuted,
        "cross_observer_overlap_fixed": True,
        "expected_failure_observed": degraded,
        "failure_mode": (
            "row-local measured channels were permuted relative to fixed observer identity and fixed "
            "cross-observer overlap; the reconstruction should degrade"
        ),
    }


def _view_feature(view: dict[str, Any]) -> list[float]:
    return [
        float(view.get("committed_fraction", 0.0)),
        float(view.get("record_stability_mean", 0.0)),
        float(view.get("repair_load_mean", 0.0)),
        float(view.get("mismatch_density_mean", 0.0)),
        float(view.get("visible_signature_entropy", 0.0)),
    ]


def _verified_local_packet_feature_matrix(patch_views: list[dict[str, Any]]) -> np.ndarray | None:
    vectors: list[list[float]] = []
    for view in patch_views:
        schema = view.get("locality_preserving_packet_feature_schema")
        vector = _flatten_numeric(view.get("locality_preserving_packet_feature_vector"))
        valid = bool(
            isinstance(schema, dict)
            and schema.get("support_selection_carrier") == "finite_patch_adjacency_bfs"
            and schema.get("excluded_hash_fields")
            and not schema.get("feature_value_coordinate_fields_used")
            and set(schema.get("fields", [])) <= {"repair_load", "cumulative_repair_load"}
            and vector
        )
        if not valid:
            return None
        vectors.append(vector)
    return _padded_feature_rows(vectors)


def _verified_measured_overlap_feature_matrix(patch_views: list[dict[str, Any]]) -> np.ndarray | None:
    ids: list[int] = []
    for index, view in enumerate(patch_views):
        try:
            ids.append(int(view.get("observer_id", index)))
        except (TypeError, ValueError):
            return None
    if len(set(ids)) != len(ids):
        return None
    by_id = {observer_id: index for index, observer_id in enumerate(ids)}
    directed = np.zeros((len(patch_views), len(patch_views)), dtype=float)
    for index, view in enumerate(patch_views):
        provenance = view.get("overlap_correspondence_evidence_provenance")
        correspondences = view.get("measured_overlap_correspondences")
        valid = bool(
            isinstance(provenance, dict)
            and provenance.get("cross_observer_measurement") is True
            and provenance.get("self_histogram_synthesis") is False
            and provenance.get("support_selection_carrier") == "finite_patch_adjacency_bfs"
            and isinstance(correspondences, list)
        )
        if not valid:
            return None
        for row in correspondences:
            if not isinstance(row, dict):
                continue
            try:
                peer = by_id[int(row.get("peer_observer_id"))]
                affinity = float(row.get("measured_affinity"))
            except (KeyError, TypeError, ValueError):
                continue
            if peer == index or not np.isfinite(affinity):
                continue
            directed[index, peer] = max(directed[index, peer], float(np.clip(affinity, 0.0, 1.0)))
    reciprocal = (directed > 0.0) & (directed.T > 0.0)
    if int(np.count_nonzero(np.triu(reciprocal, k=1))) <= 0:
        return None
    affinity = np.sqrt(np.maximum(directed, 0.0) * np.maximum(directed.T, 0.0))
    np.fill_diagonal(affinity, 1.0)
    return affinity


def _verified_paired_response_feature_matrix(patch_views: list[dict[str, Any]]) -> np.ndarray | None:
    vectors: list[list[float]] = []
    for view in patch_views:
        provenance = view.get("paired_perturbation_response_provenance")
        vector = _flatten_numeric(view.get("paired_perturbation_response_tensor"))
        valid = bool(
            view.get("paired_perturbation_response_producer_receipt") is True
            and isinstance(provenance, dict)
            and provenance.get("actual_paired_perturb_resettle") is True
            and provenance.get("observer_local_support_readout") is True
            and vector
        )
        if not valid:
            return None
        vectors.append(vector)
    return _padded_feature_rows(vectors)


def _padded_feature_rows(vectors: list[list[float]]) -> np.ndarray:
    width = max((len(vector) for vector in vectors), default=0)
    matrix = np.zeros((len(vectors), width), dtype=float)
    for index, vector in enumerate(vectors):
        matrix[index, : len(vector)] = np.asarray(vector, dtype=float)
    return np.where(np.isfinite(matrix), matrix, 0.0)


def _feature_block_similarity(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    count = int(features.shape[0]) if features.ndim == 2 else 0
    if count <= 0:
        return np.zeros((0, 0), dtype=float)
    if count == 1 or features.shape[1] <= 0:
        return np.eye(count, dtype=float)
    finite = np.where(np.isfinite(features), features, 0.0)
    varying = np.std(finite, axis=0) > 1e-12
    if not np.any(varying):
        return np.eye(count, dtype=float)
    standardized = _standardize_columns(finite[:, varying])
    distances = squareform(pdist(standardized, metric="euclidean"))
    positive = distances[np.triu_indices_from(distances, k=1)]
    positive = positive[np.isfinite(positive) & (positive > 1e-12)]
    if positive.size == 0:
        return np.eye(count, dtype=float)
    scale = max(float(np.median(positive)), 1e-12)
    similarity = np.exp(-np.square(distances / scale))
    np.fill_diagonal(similarity, 1.0)
    return similarity


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
    matrix = np.zeros((len(patch_views), len(families)), dtype=float)
    transition_columns = 0
    for family_index, family in enumerate(families):
        transition_column = False
        for view_index, view in enumerate(patch_views):
            score = _transition_affinity_view_score(family, view)
            if score > 0.0:
                matrix[view_index, family_index] = float(score)
                transition_column = True
        if transition_column:
            transition_columns += 1
            continue
        supports = [set(int(node) for node in view.get("support_nodes", [])) for view in patch_views]
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
    matrix = np.zeros((len(patch_views), len(families)), dtype=float)
    mass_cache: dict[tuple[str, int], np.ndarray] = {}
    descriptor_score_cache: dict[tuple[tuple[str, int], ...], np.ndarray] = {}
    support_sets: list[set[int]] | None = None
    for family_index, family in enumerate(families):
        stability = float(family.get("counterfactual_stability", 0.0))
        if stability <= 0.0:
            continue
        descriptor = family.get("transition_affinity") if isinstance(family.get("transition_affinity"), dict) else {}
        if descriptor:
            descriptor_key = tuple(sorted((str(name), int(value)) for name, value in descriptor.items()))
            view_scores = descriptor_score_cache.get(descriptor_key)
            if view_scores is None:
                view_scores = _transition_descriptor_view_scores(descriptor_key, patch_views, mass_cache)
                descriptor_score_cache[descriptor_key] = view_scores
            if np.any(view_scores > 0.0):
                matrix[:, family_index] = float(stability) * view_scores
                continue
        if support_sets is None:
            support_sets = [set(int(node) for node in view.get("support_nodes", [])) for view in patch_views]
        family_support = set(int(node) for node in family.get("support_nodes", []))
        if not family_support:
            continue
        for view_index, support in enumerate(support_sets):
            overlap = len(support & family_support)
            if overlap:
                matrix[view_index, family_index] = stability * overlap / len(family_support)
    if not np.any(matrix):
        return None
    return _cosine_similarity_matrix(matrix)


def _transition_descriptor_view_scores(
    descriptor_key: tuple[tuple[str, int], ...],
    patch_views: list[dict[str, Any]],
    mass_cache: dict[tuple[str, int], np.ndarray],
) -> np.ndarray:
    if not descriptor_key or not patch_views:
        return np.zeros(len(patch_views), dtype=float)
    term_arrays = [_view_histogram_mass_array(patch_views, field, value, mass_cache) for field, value in descriptor_key]
    if not term_arrays:
        return np.zeros(len(patch_views), dtype=float)
    masses = np.vstack(term_arrays)
    positive = masses > 0.0
    positive_counts = np.sum(positive, axis=0)
    scores = np.zeros(len(patch_views), dtype=float)
    valid = positive_counts > 0
    if np.any(valid):
        log_sum = np.sum(np.where(positive, np.log(np.maximum(masses, 1e-12)), 0.0), axis=0)
        scores[valid] = np.exp(log_sum[valid] / positive_counts[valid])
    return np.clip(scores, 0.0, 1.0)


def _view_histogram_mass_array(
    patch_views: list[dict[str, Any]],
    field: str,
    value: int,
    mass_cache: dict[tuple[str, int], np.ndarray],
) -> np.ndarray:
    key = (str(field), int(value))
    cached = mass_cache.get(key)
    if cached is not None:
        return cached
    masses = np.zeros(len(patch_views), dtype=float)
    for index, view in enumerate(patch_views):
        if field == "object_packet":
            masses[index] = _histogram_value(view.get("object_packet_histogram", {}), value)
            continue
        histograms = view.get("transition_affinity_histograms", {})
        mass = _histogram_value(histograms.get(field, {}) if isinstance(histograms, dict) else {}, value)
        if mass <= 0.0:
            history = view.get("transition_history_histograms", {})
            mass = _histogram_value(history.get(field, {}) if isinstance(history, dict) else {}, value)
        masses[index] = mass
    mass_cache[key] = masses
    return masses


def _transition_affinity_view_score(family: dict[str, Any], view: dict[str, Any]) -> float:
    descriptor = family.get("transition_affinity") if isinstance(family.get("transition_affinity"), dict) else {}
    if not descriptor:
        return 0.0
    score_terms: list[float] = []
    for name, target in descriptor.items():
        field = str(name)
        value = int(target)
        if field == "object_packet":
            mass = _histogram_value(view.get("object_packet_histogram", {}), value)
        else:
            histograms = view.get("transition_affinity_histograms", {})
            mass = _histogram_value(histograms.get(field, {}) if isinstance(histograms, dict) else {}, value)
            if mass <= 0.0:
                history = view.get("transition_history_histograms", {})
                mass = _histogram_value(history.get(field, {}) if isinstance(history, dict) else {}, value)
        if mass > 0.0:
            score_terms.append(float(mass))
    if not score_terms:
        return 0.0
    values = np.clip(np.asarray(score_terms, dtype=float), 0.0, 1.0)
    return float(np.exp(np.mean(np.log(np.maximum(values, 1e-12)))))


def _histogram_value(histogram: Any, key: int) -> float:
    if not isinstance(histogram, dict):
        return 0.0
    if key in histogram:
        return float(histogram[key])
    return float(histogram.get(str(int(key)), 0.0))


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
