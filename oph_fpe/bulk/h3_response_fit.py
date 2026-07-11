from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares

from oph_fpe.bulk.cap_normals import cap_normals
from oph_fpe.bulk.cap_geometry import RoundCap
from oph_fpe.bulk.h3_chart import (
    h3_distance_matrix,
    h3_halfspace_profile,
    h3_point_from_tangent,
    h3_tangent_from_point,
    random_h3_points,
)
from oph_fpe.bulk.observer_reconstruction import neutral_dimension_report_from_distance
from oph_fpe.bulk.record_to_h3 import fit_response_profiles_to_h3
from oph_fpe.claims import (
    DEMO,
    H3_RESPONSE_CANDIDATE_RECEIPT,
    H3_RESPONSE_CONTROL_SEPARATION_RECEIPT,
    with_claim_metadata,
)
from oph_fpe.core.graph import fibonacci_sphere_points


def modular_response_h3_report(
    kernel: dict[str, Any],
    caps: list[RoundCap],
    *,
    candidate_count: int = 2048,
    candidate_radius: float = 2.0,
    softness: float = 0.25,
    seed: int = 1,
    pass_ratio: float = 0.85,
    min_observers: int = 8,
    min_features: int = 12,
    fit_mode: str = "row_independent",
    heldout_fraction: float = 0.25,
    anchor_weight: float = 0.05,
    max_iterations: int = 4,
    feature_selection: str = "none",
    max_fit_features: int | None = None,
    min_feature_std: float = 0.0,
    min_wrong_scale_feature_delta: float = 0.0,
    exclude_observables: list[str] | tuple[str, ...] = (),
    exclude_feature_types: list[str] | tuple[str, ...] = (),
    max_features_per_cap_time_observable: int | None = None,
    refine_steps: int = 0,
    refine_max_rows: int | None = None,
    refine_max_nfev: int = 48,
    candidate_mode: str = "random",
    channel_mode: str = "time_observable_class",
    profile_mode: str = "static_halfspace",
    profile_time_scale: float = 2.0 * np.pi,
    control_fit_mode: str = "same_h3_model_not_affine_target_fit",
    blind_feature_selection: bool = False,
) -> dict[str, Any]:
    matrix = np.asarray(kernel.get("matrix", np.zeros((0, 0))), dtype=float)
    feature_rows = list(kernel.get("feature_rows", []))
    if matrix.size == 0 or not feature_rows or not caps:
        return with_claim_metadata(
            {
                "mode": "modular_response_kernel_to_h3_fit",
                "MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT": False,
                H3_RESPONSE_CONTROL_SEPARATION_RECEIPT: False,
                H3_RESPONSE_CANDIDATE_RECEIPT: False,
                "h3_bulk_candidate_receipt": False,
                "reason": "empty_kernel_or_caps",
                "claim_boundary": "empty modular response H3 fit; no bulk claim",
            },
            claim_level=DEMO,
            receipt=H3_RESPONSE_CANDIDATE_RECEIPT,
            physical_claim=False,
            observable_id="support_visible_modular_response_kernel",
            fit_objective="heldout_h3_response_beats_controls",
        )
    kernel, matrix, feature_rows, feature_selection_report = _select_fit_features(
        kernel,
        matrix,
        feature_rows,
        mode=str(feature_selection),
        max_features=max_fit_features,
        min_std=float(min_feature_std),
        min_wrong_scale_delta=float(min_wrong_scale_feature_delta),
        exclude_observables=tuple(str(value) for value in exclude_observables),
        exclude_feature_types=tuple(str(value) for value in exclude_feature_types),
        min_features=int(min_features),
        max_features_per_cap_time_observable=(
            int(max_features_per_cap_time_observable)
            if max_features_per_cap_time_observable is not None
            else None
        ),
        blind=bool(blind_feature_selection),
    )
    expanded_caps = [caps[int(row["cap_index"])] for row in feature_rows]
    if str(fit_mode) == "joint_global":
        report = _modular_response_joint_h3_report(
            kernel,
            caps,
            expanded_caps,
            feature_rows,
            matrix,
            candidate_count=int(candidate_count),
            candidate_radius=float(candidate_radius),
            softness=float(softness),
            seed=int(seed),
            pass_ratio=float(pass_ratio),
            min_observers=int(min_observers),
            min_features=int(min_features),
            heldout_fraction=float(heldout_fraction),
            anchor_weight=float(anchor_weight),
            max_iterations=int(max_iterations),
            refine_steps=int(refine_steps),
            refine_max_rows=int(refine_max_rows) if refine_max_rows is not None else None,
            refine_max_nfev=int(refine_max_nfev),
            candidate_mode=str(candidate_mode),
            channel_mode=str(channel_mode),
            profile_mode=str(profile_mode),
            profile_time_scale=float(profile_time_scale),
            control_fit_mode=str(control_fit_mode),
        )
        report["feature_selection"] = feature_selection_report
        report["inference_protocol"] = {
            "blind_feature_selection": bool(blind_feature_selection),
            "equal_target_control_candidate_capacity": True,
            "claim_boundary": (
                "When blind_feature_selection is true, response values and wrong-scale outcomes do not "
                "choose the fitted feature columns. Assumption-driven visualization may still use a "
                "predeclared H3 branch without promoting this diagnostic to a theorem proof."
            ),
        }
        return report
    h3_fit = fit_response_profiles_to_h3(
        matrix,
        expanded_caps,
        candidate_count=int(candidate_count),
        candidate_radius=float(candidate_radius),
        softness=float(softness),
        seed=int(seed),
    )
    s2_residuals = _pairwise_residuals(matrix, np.asarray(kernel.get("s2_boundary_control", np.zeros_like(matrix)), dtype=float))
    shuffled_fit = fit_response_profiles_to_h3(
        np.asarray(kernel.get("shuffled_control", np.zeros_like(matrix)), dtype=float),
        expanded_caps,
        candidate_count=int(candidate_count),
        candidate_radius=float(candidate_radius),
        softness=float(softness),
        seed=int(seed) + 101,
    )
    no_flow_residuals = _pairwise_residuals(
        matrix,
        np.asarray(kernel.get("no_modular_flow_control", np.zeros_like(matrix)), dtype=float),
    )
    h3_median = _as_float(h3_fit.get("median_residual"))
    s2_median = float(np.median(s2_residuals)) if s2_residuals.size else float("nan")
    shuffled_median = _as_float(shuffled_fit.get("median_residual"))
    no_flow_median = float(np.median(no_flow_residuals)) if no_flow_residuals.size else float("nan")
    response_summary = dict(kernel.get("response_summary", {}))
    response_degenerate = bool(float(response_summary.get("mean_row_std", 0.0)) < 1e-3)
    h3_beats_s2 = _beats(h3_median, s2_median, pass_ratio)
    h3_beats_shuffled = _beats(h3_median, shuffled_median, pass_ratio)
    h3_beats_no_flow = _beats(h3_median, no_flow_median, pass_ratio)
    eligible = bool(matrix.shape[0] >= int(min_observers) and matrix.shape[1] >= int(min_features))
    receipt = bool(eligible and not response_degenerate and h3_beats_s2 and h3_beats_shuffled and h3_beats_no_flow)
    report = {
        "mode": "modular_response_kernel_to_h3_fit",
        "fit_mode": "row_independent",
        "MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT": receipt,
        H3_RESPONSE_CONTROL_SEPARATION_RECEIPT: receipt,
        "h3_control_separation_receipt": receipt,
        H3_RESPONSE_CANDIDATE_RECEIPT: receipt,
        "h3_bulk_candidate_receipt": receipt,
        "observer_ids": [int(value) for value in kernel.get("observer_ids", range(matrix.shape[0]))],
        "observer_count": int(matrix.shape[0]),
        "feature_count": int(matrix.shape[1]),
        "cap_count": int(kernel.get("cap_count", len(caps))),
        "time_count": int(kernel.get("time_count", 0)),
        "field_names": list(kernel.get("field_names", [])),
        "feature_rows_sample": feature_rows[:128],
        "feature_selection": feature_selection_report,
        "h3_fit": h3_fit,
        "h3_chart_dimension_debug": _h3_chart_dimension_debug(h3_fit),
        "s2_boundary_control": {
            "median_residual": s2_median if np.isfinite(s2_median) else None,
            "mean_residual": float(np.mean(s2_residuals)) if s2_residuals.size else None,
            "h3_beats_s2_boundary": h3_beats_s2,
        },
        "shuffled_modular_response_control": {
            "median_residual": shuffled_fit.get("median_residual"),
            "h3_beats_shuffled": h3_beats_shuffled,
        },
        "no_modular_flow_control": {
            "median_residual": no_flow_median if np.isfinite(no_flow_median) else None,
            "h3_beats_no_flow": h3_beats_no_flow,
        },
        "response_summary": response_summary,
        "response_degenerate": response_degenerate,
        "eligibility_gate_passed": eligible,
        "pass_ratio": float(pass_ratio),
        "claim_boundary": (
            "fits the support-visible modular response kernel R[i,C,t,O] into the canonical H3 chart. "
            "This is a bulk-candidate receipt only when H3 beats S2-boundary, shuffled-response, and "
            "no-modular-flow controls. It is not a final 3D bulk emergence claim without refinement "
            "scaling and planted controls."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=DEMO,
        receipt=H3_RESPONSE_CANDIDATE_RECEIPT,
        physical_claim=False,
        observable_id="support_visible_modular_response_kernel",
        fit_objective="heldout_h3_response_beats_controls",
    )


def _select_fit_features(
    kernel: dict[str, Any],
    matrix: np.ndarray,
    feature_rows: list[dict[str, Any]],
    *,
    mode: str,
    max_features: int | None,
    min_std: float,
    min_wrong_scale_delta: float,
    exclude_observables: tuple[str, ...],
    exclude_feature_types: tuple[str, ...],
    min_features: int,
    max_features_per_cap_time_observable: int | None,
    blind: bool = False,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    matrix = np.asarray(matrix, dtype=float)
    original_count = int(matrix.shape[1]) if matrix.ndim == 2 else 0
    excluded_observable_set = {str(value) for value in exclude_observables if str(value)}
    excluded_feature_type_set = {str(value) for value in exclude_feature_types if str(value)}
    pre_exclusion_count = int(original_count)
    excluded_count = 0
    if original_count and (excluded_observable_set or excluded_feature_type_set):
        exclusion_mask = np.asarray(
            [
                str(row.get("observable", row.get("field", ""))) not in excluded_observable_set
                and str(row.get("feature_type", "")) not in excluded_feature_type_set
                for row in feature_rows
            ],
            dtype=bool,
        )
        if exclusion_mask.size == original_count and np.any(exclusion_mask):
            pre_indices = np.flatnonzero(exclusion_mask)
            excluded_count = int(original_count - pre_indices.size)
            matrix = matrix[:, pre_indices]
            feature_rows = [
                {**feature_rows[int(index)], "source_feature_index_before_exclusion": int(index)}
                for index in pre_indices
            ]
            kernel = dict(kernel)
            kernel["matrix"] = matrix
            kernel["feature_rows"] = feature_rows
            kernel["s2_boundary_control"] = _select_columns(kernel.get("s2_boundary_control"), pre_indices, matrix)
            kernel["shuffled_control"] = _select_columns(kernel.get("shuffled_control"), pre_indices, matrix)
            kernel["shuffled_response_control"] = _select_columns(
                kernel.get("shuffled_response_control", kernel.get("shuffled_control")),
                pre_indices,
                matrix,
            )
            kernel["shuffled_observer_labels_control"] = _select_columns(
                kernel.get("shuffled_observer_labels_control"),
                pre_indices,
                matrix,
            )
            kernel["no_modular_flow_control"] = _select_columns(
                kernel.get("no_modular_flow_control"),
                pre_indices,
                matrix,
            )
            wrong_controls = kernel.get("wrong_scale_controls", {}) or {}
            kernel["wrong_scale_controls"] = {
                str(label): _select_columns(value, pre_indices, matrix) for label, value in wrong_controls.items()
            }
            original_count = int(matrix.shape[1]) if matrix.ndim == 2 else 0
    normalized_mode = str(mode or "none").lower().replace("-", "_")
    aggregation_mode = "none"
    filter_mode = normalized_mode
    if filter_mode.startswith("grouped_"):
        aggregation_mode = "cap_time_observable_feature_type_mean"
        filter_mode = filter_mode[len("grouped_") :]
    rank_strategy = "feature_std"
    if filter_mode.endswith("_scale_rank"):
        filter_mode = filter_mode[: -len("_scale_rank")]
        rank_strategy = "wrong_scale_delta"
    elif filter_mode.endswith("_scale_weighted_rank"):
        filter_mode = filter_mode[: -len("_scale_weighted_rank")]
        rank_strategy = "feature_std_times_wrong_scale_delta"
    if (filter_mode in {"none", "off", "false"} and aggregation_mode == "none") or original_count == 0:
        scale_delta = _wrong_scale_feature_delta(matrix, kernel.get("wrong_scale_controls", {}) or {})
        return kernel, matrix, feature_rows, {
            "mode": "none",
            "original_feature_count": original_count,
            "selected_feature_count": original_count,
            "min_feature_std": float(min_std),
            "min_wrong_scale_feature_delta": float(min_wrong_scale_delta),
            "wrong_scale_delta_filter_applied": False,
            "scale_delta_median": float(np.median(scale_delta)) if scale_delta.size else None,
            "rank_strategy": rank_strategy,
            "aggregation_mode": "none",
            "max_fit_features": int(max_features) if max_features is not None else None,
            "exclude_observables": sorted(excluded_observable_set),
            "exclude_feature_types": sorted(excluded_feature_type_set),
            "pre_exclusion_feature_count": int(pre_exclusion_count),
            "excluded_feature_count": int(excluded_count),
        }
    std = np.std(matrix, axis=0)
    finite = np.isfinite(std)
    metadata_mask = np.ones(original_count, dtype=bool)
    metadata_filter = None
    if filter_mode in {"change_probability", "change_probability_only", "change_only"}:
        metadata_mask = np.asarray(
            [
                str(row.get("feature_type", "")) == "change_probability_delta"
                for row in feature_rows
            ],
            dtype=bool,
        )
        metadata_filter = "feature_type=change_probability_delta"
    elif filter_mode in {"signed_transition_distribution", "signed_transition", "transition_distribution"}:
        signed_types = {
            "class_distribution_delta",
            "target_distribution_delta",
            "class_log_odds_delta",
            "transition_matrix_delta",
            "entropy_delta",
            "sector_preservation_delta",
            "change_probability_delta",
        }
        metadata_mask = np.asarray(
            [str(row.get("feature_type", "")) in signed_types for row in feature_rows],
            dtype=bool,
        )
        metadata_filter = "signed_class_resolved_transition_features"
    elif filter_mode in {
        "signed_transition_low_order",
        "signed_transition_distribution_low_order",
        "signed_transition_no_matrix",
        "signed_class_resolved_transition_features_without_transition_matrix",
        "class_resolved_transition_features_without_transition_matrix",
        "signed_class_resolved_no_matrix",
    }:
        signed_types = {
            "class_distribution_delta",
            "target_distribution_delta",
            "class_log_odds_delta",
            "entropy_delta",
            "sector_preservation_delta",
            "change_probability_delta",
        }
        metadata_mask = np.asarray(
            [str(row.get("feature_type", "")) in signed_types for row in feature_rows],
            dtype=bool,
        )
        metadata_filter = "signed_class_resolved_transition_features_without_transition_matrix"
    elif filter_mode in {
        "class_distribution_only",
        "signed_class_distribution_only",
        "class_delta_only",
    }:
        metadata_mask = np.asarray(
            [
                str(row.get("feature_type", ""))
                in {"class_distribution_delta", "target_distribution_delta"}
                for row in feature_rows
            ],
            dtype=bool,
        )
        metadata_filter = "feature_type=class_distribution_delta"
    elif filter_mode in {
        "class_distribution_and_change",
        "signed_class_distribution_and_change",
        "class_delta_and_change",
    }:
        metadata_mask = np.asarray(
            [
                str(row.get("feature_type", ""))
                in {
                    "class_distribution_delta",
                    "target_distribution_delta",
                    "change_probability_delta",
                    "entropy_delta",
                    "sector_preservation_delta",
                }
                for row in feature_rows
            ],
            dtype=bool,
        )
        metadata_filter = "class_distribution_plus_scalar_transition_features"
    if excluded_observable_set or excluded_feature_type_set:
        exclusion_mask = np.asarray(
            [
                str(row.get("observable", row.get("field", ""))) not in excluded_observable_set
                and str(row.get("feature_type", "")) not in excluded_feature_type_set
                for row in feature_rows
            ],
            dtype=bool,
        )
        if exclusion_mask.size == original_count:
            metadata_mask = metadata_mask & exclusion_mask
    scale_delta = _wrong_scale_feature_delta(matrix, kernel.get("wrong_scale_controls", {}) or {})
    scale_filter_applied = bool(
        not blind and float(min_wrong_scale_delta) > 0.0 and scale_delta.size == original_count
    )
    scale_mask = np.ones(original_count, dtype=bool)
    if scale_filter_applied:
        scale_mask = scale_delta >= float(min_wrong_scale_delta)
    rank_score = std.copy()
    if blind:
        # Predeclared metadata and stable source order are allowed to select
        # columns. Target/control residuals and feature amplitudes are not.
        rank_strategy = "stable_source_order_blind"
        rank_score = -np.arange(original_count, dtype=float)
    elif rank_strategy == "wrong_scale_delta" and scale_delta.size == original_count:
        rank_score = scale_delta.copy()
    elif rank_strategy == "feature_std_times_wrong_scale_delta" and scale_delta.size == original_count:
        rank_score = std * scale_delta
    rank_score = np.where(np.isfinite(rank_score), rank_score, -np.inf)
    eligible = (
        np.all(np.isfinite(matrix), axis=0) & metadata_mask
        if blind
        else finite & metadata_mask & (std >= float(min_std)) & scale_mask
    )
    if not np.any(eligible):
        eligible = finite & metadata_mask & scale_mask
    if not np.any(eligible) and scale_filter_applied:
        # Keep the run diagnostic instead of silently creating an empty fit.
        # Prefer scale-distinctive features over metadata filters when the
        # metadata filter would otherwise remove every usable column.
        eligible = finite & scale_mask
    if not np.any(eligible) and scale_filter_applied:
        # The report records that the requested scale-distinctive filter could
        # not retain enough features, but still leaves a diagnostic fit.
        eligible = finite & metadata_mask
    if not np.any(eligible):
        eligible = finite
    indices = np.flatnonzero(eligible)
    if indices.size == 0:
        indices = np.arange(original_count, dtype=np.int64)
    if max_features_per_cap_time_observable is not None and int(max_features_per_cap_time_observable) > 0:
        indices = _limit_features_per_cap_time_observable(
            indices,
            rank_score,
            feature_rows,
            max_per_group=int(max_features_per_cap_time_observable),
        )
    if max_features is not None and int(max_features) > 0 and indices.size > int(max_features):
        order = np.argsort(rank_score[indices])[::-1]
        indices = indices[order[: int(max_features)]]
    indices = indices[np.argsort(indices)]
    if indices.size < int(min_features) and original_count >= int(min_features):
        fallback_pool = np.flatnonzero(finite & scale_mask) if scale_filter_applied else np.arange(original_count)
        if fallback_pool.size < int(min_features):
            fallback_pool = np.arange(original_count)
        order = fallback_pool[np.argsort(rank_score[fallback_pool])[::-1][: int(min_features)]]
        indices = np.unique(np.concatenate([indices, order])).astype(np.int64)
        if max_features is not None and int(max_features) > 0 and indices.size > int(max_features):
            order = np.argsort(rank_score[indices])[::-1]
            indices = indices[order[: int(max_features)]]
        indices = indices[np.argsort(indices)]
    selected_feature_rows = [
        {**feature_rows[int(index)], "selected_feature_index": int(out_index), "source_feature_index": int(index)}
        for out_index, index in enumerate(indices)
    ]
    selected_matrix = matrix[:, indices]
    selected_kernel = dict(kernel)
    selected_kernel["matrix"] = selected_matrix
    selected_kernel["feature_rows"] = selected_feature_rows
    selected_kernel["s2_boundary_control"] = _select_columns(kernel.get("s2_boundary_control"), indices, selected_matrix)
    selected_kernel["shuffled_control"] = _select_columns(kernel.get("shuffled_control"), indices, selected_matrix)
    selected_kernel["shuffled_response_control"] = _select_columns(
        kernel.get("shuffled_response_control", kernel.get("shuffled_control")),
        indices,
        selected_matrix,
    )
    selected_kernel["shuffled_observer_labels_control"] = _select_columns(
        kernel.get("shuffled_observer_labels_control"),
        indices,
        selected_matrix,
    )
    selected_kernel["no_modular_flow_control"] = _select_columns(kernel.get("no_modular_flow_control"), indices, selected_matrix)
    wrong_controls = kernel.get("wrong_scale_controls", {}) or {}
    selected_kernel["wrong_scale_controls"] = {
        str(label): _select_columns(value, indices, selected_matrix) for label, value in wrong_controls.items()
    }
    pre_aggregation_feature_count = int(selected_matrix.shape[1]) if selected_matrix.ndim == 2 else 0
    aggregation_report = {
        "mode": "none",
        "input_feature_count": pre_aggregation_feature_count,
        "output_feature_count": pre_aggregation_feature_count,
    }
    if aggregation_mode != "none":
        selected_kernel, selected_matrix, selected_feature_rows, aggregation_report = _aggregate_selected_feature_groups(
            selected_kernel,
            selected_matrix,
            selected_feature_rows,
            mode=aggregation_mode,
        )
    selected_kernel["response_summary"] = _matrix_summary(selected_matrix)
    return selected_kernel, selected_matrix, selected_feature_rows, {
        "mode": normalized_mode,
        "original_feature_count": original_count,
        "selected_feature_count": int(selected_matrix.shape[1]) if selected_matrix.ndim == 2 else 0,
        "pre_aggregation_selected_feature_count": pre_aggregation_feature_count,
        "metadata_filter": metadata_filter,
        "filter_mode": filter_mode,
        "rank_strategy": rank_strategy,
        "blind_feature_selection": bool(blind),
        "response_values_used_for_selection": bool(not blind),
        "aggregation_mode": aggregation_mode,
        "aggregation": aggregation_report,
        "min_feature_std": float(min_std),
        "min_wrong_scale_feature_delta": float(min_wrong_scale_delta),
        "exclude_observables": sorted(excluded_observable_set),
        "exclude_feature_types": sorted(excluded_feature_type_set),
        "pre_exclusion_feature_count": int(pre_exclusion_count),
        "excluded_feature_count": int(excluded_count),
        "wrong_scale_delta_filter_applied": scale_filter_applied,
        "wrong_scale_delta_filter_retained_count": (
            int(np.sum(finite & metadata_mask & (std >= float(min_std)) & scale_mask))
            if scale_filter_applied
            else None
        ),
        "scale_delta_min": float(np.min(scale_delta[indices])) if scale_delta.size and indices.size else None,
        "scale_delta_median": float(np.median(scale_delta[indices])) if scale_delta.size and indices.size else None,
        "scale_delta_max": float(np.max(scale_delta[indices])) if scale_delta.size and indices.size else None,
        "max_fit_features": int(max_features) if max_features is not None else None,
        "max_features_per_cap_time_observable": (
            int(max_features_per_cap_time_observable)
            if max_features_per_cap_time_observable is not None
            else None
        ),
        "selected_feature_indices_sample": [int(value) for value in indices[:128]],
        "selected_std_min": float(np.min(std[indices])) if indices.size else None,
        "selected_std_median": float(np.median(std[indices])) if indices.size else None,
        "selected_std_max": float(np.max(std[indices])) if indices.size else None,
        "selected_rank_score_min": float(np.min(rank_score[indices])) if indices.size else None,
        "selected_rank_score_median": float(np.median(rank_score[indices])) if indices.size else None,
        "selected_rank_score_max": float(np.max(rank_score[indices])) if indices.size else None,
        "claim_boundary": (
            "Fit-layer feature selection only; raw modular-response kernel is unchanged. Controls are "
            "filtered and, when requested, grouped with the same selected column groups. In blind mode "
            "only predeclared metadata, finite-value integrity, and stable source order select columns."
        ),
    }


def _wrong_scale_feature_delta(matrix: np.ndarray, wrong_controls: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not wrong_controls:
        return np.zeros(matrix.shape[1] if matrix.ndim == 2 else 0, dtype=float)
    deltas: list[np.ndarray] = []
    for value in wrong_controls.values():
        control = np.asarray(value, dtype=float)
        if control.shape != matrix.shape:
            continue
        deltas.append(np.mean(np.abs(matrix - control), axis=0))
    if not deltas:
        return np.zeros(matrix.shape[1], dtype=float)
    return np.min(np.vstack(deltas), axis=0)


def _select_columns(value: Any, indices: np.ndarray, selected_matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if indices.size == 0:
        return np.zeros_like(selected_matrix)
    if array.ndim != 2 or array.shape[0] != selected_matrix.shape[0] or array.shape[1] <= int(np.max(indices)):
        return np.zeros_like(selected_matrix)
    return array[:, indices]


def _aggregate_selected_feature_groups(
    kernel: dict[str, Any],
    matrix: np.ndarray,
    feature_rows: list[dict[str, Any]],
    *,
    mode: str,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Average class-level feature columns into paper-visible response channels.

    The modular-response kernel can contain many packet-class or transition-class
    columns for the same cap/time/observable. At small N those class columns are
    noisy enough that wrong-normalization controls can win isolated feature
    residuals even when the aggregate response separates correctly. This helper
    applies one deterministic grouping to the selected data and to every control
    matrix, so it is a denoising diagnostic rather than a result-forcing path.
    """

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not feature_rows:
        return kernel, matrix, feature_rows, {
            "mode": str(mode),
            "input_feature_count": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
            "output_feature_count": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
            "reason": "empty_matrix_or_features",
        }
    groups: dict[tuple[int, int, str, str], list[int]] = {}
    for index, row in enumerate(feature_rows):
        key = (
            int(row.get("cap_index", -1)),
            int(row.get("time_index", -1)),
            str(row.get("observable", row.get("field", "unknown"))),
            str(row.get("feature_type", "scalar")),
        )
        groups.setdefault(key, []).append(int(index))
    group_indices = list(groups.values())
    if len(group_indices) == matrix.shape[1]:
        return kernel, matrix, feature_rows, {
            "mode": str(mode),
            "input_feature_count": int(matrix.shape[1]),
            "output_feature_count": int(matrix.shape[1]),
            "group_count": int(len(group_indices)),
            "reason": "already_one_feature_per_group",
        }
    aggregated_matrix = _aggregate_columns_by_groups(matrix, group_indices)
    aggregated_rows: list[dict[str, Any]] = []
    for out_index, (key, indices) in enumerate(zip(groups.keys(), group_indices, strict=False)):
        first = dict(feature_rows[int(indices[0])])
        source_targets = sorted(
            {
                str(feature_rows[int(index)].get("target_class"))
                for index in indices
                if feature_rows[int(index)].get("target_class") is not None
            }
        )
        feature_type = str(key[3])
        first.update(
            {
                "feature_index": int(out_index),
                "selected_feature_index": int(out_index),
                "cap_index": int(key[0]),
                "time_index": int(key[1]),
                "observable": str(key[2]),
                "feature_type": f"grouped_{feature_type}",
                "target_class": "aggregate",
                "source_feature_type": feature_type,
                "source_feature_count": int(len(indices)),
                "source_selected_feature_indices_sample": [int(value) for value in indices[:32]],
                "source_target_classes_sample": source_targets[:32],
            }
        )
        aggregated_rows.append(first)
    aggregated_kernel = dict(kernel)
    aggregated_kernel["matrix"] = aggregated_matrix
    aggregated_kernel["feature_rows"] = aggregated_rows
    for key in (
        "s2_boundary_control",
        "shuffled_control",
        "shuffled_response_control",
        "shuffled_observer_labels_control",
        "no_modular_flow_control",
    ):
        aggregated_kernel[key] = _aggregate_control_by_groups(kernel.get(key), group_indices, aggregated_matrix)
    aggregated_kernel["wrong_scale_controls"] = {
        str(label): _aggregate_control_by_groups(value, group_indices, aggregated_matrix)
        for label, value in (kernel.get("wrong_scale_controls", {}) or {}).items()
    }
    aggregated_kernel["response_summary"] = _matrix_summary(aggregated_matrix)
    return aggregated_kernel, aggregated_matrix, aggregated_rows, {
        "mode": str(mode),
        "input_feature_count": int(matrix.shape[1]),
        "output_feature_count": int(aggregated_matrix.shape[1]),
        "group_count": int(len(group_indices)),
        "median_source_features_per_group": float(np.median([len(value) for value in group_indices])),
        "max_source_features_per_group": int(max(len(value) for value in group_indices)),
        "claim_boundary": (
            "deterministic fit-layer aggregation by cap/time/observable/feature_type. The same groups are "
            "applied to all controls; this is a denoising diagnostic, not a change to the raw simulator."
        ),
    }


def _aggregate_control_by_groups(value: Any, group_indices: list[list[int]], aggregated_matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[0] != aggregated_matrix.shape[0]:
        return np.zeros_like(aggregated_matrix)
    max_index = max((max(indices) for indices in group_indices if indices), default=-1)
    if array.shape[1] <= int(max_index):
        return np.zeros_like(aggregated_matrix)
    return _aggregate_columns_by_groups(array, group_indices)


def _aggregate_columns_by_groups(array: np.ndarray, group_indices: list[list[int]]) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    columns = [np.mean(array[:, np.asarray(indices, dtype=np.int64)], axis=1) for indices in group_indices]
    return np.vstack(columns).T if columns else np.zeros((array.shape[0], 0), dtype=float)


def _limit_features_per_cap_time_observable(
    indices: np.ndarray,
    std: np.ndarray,
    feature_rows: list[dict[str, Any]],
    *,
    max_per_group: int,
) -> np.ndarray:
    grouped: dict[tuple[int, int, str], list[int]] = {}
    for index in np.asarray(indices, dtype=np.int64):
        row = feature_rows[int(index)] if int(index) < len(feature_rows) else {}
        key = (
            int(row.get("cap_index", -1)),
            int(row.get("time_index", -1)),
            str(row.get("observable", row.get("field", "unknown"))),
        )
        grouped.setdefault(key, []).append(int(index))
    selected: list[int] = []
    for group_indices in grouped.values():
        ordered = sorted(group_indices, key=lambda idx: float(std[idx]), reverse=True)
        selected.extend(ordered[: int(max_per_group)])
    if not selected:
        return np.asarray(indices, dtype=np.int64)
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def _matrix_summary(matrix: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "std": 0.0, "mean_row_std": 0.0, "mean_col_std": 0.0}
    return {
        "min": float(np.min(matrix)),
        "mean": float(np.mean(matrix)),
        "max": float(np.max(matrix)),
        "std": float(np.std(matrix)),
        "mean_row_std": float(np.mean(np.std(matrix, axis=1))) if matrix.ndim == 2 else 0.0,
        "mean_col_std": float(np.mean(np.std(matrix, axis=0))) if matrix.ndim == 2 else 0.0,
    }


def _h3_chart_dimension_debug(h3_fit: dict[str, Any]) -> dict[str, Any]:
    points = np.asarray(h3_fit.get("fitted_h3_points", []), dtype=float)
    if points.ndim != 2 or points.shape[0] < 8 or points.shape[1] != 4:
        return {
            "mode": "h3_chart_dimension_debug",
            "point_count": int(points.shape[0]) if points.ndim == 2 else 0,
            "candidate_3d_dimension_window": False,
            "reason": "not_enough_h3_points",
            "claim_boundary": "debug-only dimension estimate on fitted H3 chart points; no bulk claim",
        }
    dimension = neutral_dimension_report_from_distance(h3_distance_matrix(points))
    corr = dimension.get("correlation_dimension", {}).get("estimate")
    mle = dimension.get("local_mle_dimension", {}).get("estimate")
    candidate = bool(
        isinstance(corr, (int, float))
        and isinstance(mle, (int, float))
        and np.isfinite(float(corr))
        and np.isfinite(float(mle))
        and 2.7 <= float(corr) <= 3.3
        and 2.7 <= float(mle) <= 3.3
    )
    return {
        "mode": "h3_chart_dimension_debug",
        "point_count": int(points.shape[0]),
        "candidate_3d_dimension_window": candidate,
        "dimension_estimators_agree": bool(dimension.get("dimension_estimators_agree", False)),
        "correlation_dimension": dimension.get("correlation_dimension"),
        "local_mle_dimension": dimension.get("local_mle_dimension"),
        "claim_boundary": (
            "debug-only dimension estimate on fitted modular-response H3 chart points. Since the chart "
            "is an H3 target space, this checks whether observer assignments populate a 3D-like region; "
            "it is not a neutral bulk reconstruction or physical 3D claim."
        ),
    }


def synthetic_h3_modular_kernel(
    caps: list[RoundCap],
    *,
    observer_count: int = 16,
    times: list[float] | tuple[float, ...] = (0.1,),
    field_names: list[str] | tuple[str, ...] = ("record_signature",),
    seed: int = 1,
    radius: float = 1.2,
    softness: float = 0.2,
) -> dict[str, Any]:
    points = random_h3_points(int(observer_count), seed=int(seed), radius=float(radius))
    feature_rows: list[dict[str, Any]] = []
    expanded_caps: list[RoundCap] = []
    for cap_index, cap in enumerate(caps):
        for time_index, time_value in enumerate(times):
            for field_name in field_names:
                feature_rows.append(
                    {
                        "feature_index": len(feature_rows),
                        "cap_index": int(cap_index),
                        "time_index": int(time_index),
                        "time": float(time_value),
                        "field": str(field_name),
                    }
                )
                expanded_caps.append(cap)
    matrix = h3_halfspace_profile(points, cap_normals(expanded_caps), softness=float(softness))
    rng = np.random.default_rng(seed + 33)
    shuffled = matrix.copy()
    for row in shuffled:
        rng.shuffle(row)
    return {
        "mode": "observer_modular_response_kernel",
        "matrix": matrix,
        "h3_source_points": [[float(x) for x in row] for row in points],
        "s2_boundary_control": 0.5 * np.ones_like(matrix),
        "shuffled_control": shuffled,
        "no_modular_flow_control": 0.5 * np.ones_like(matrix),
        "feature_rows": feature_rows,
        "observer_ids": list(range(int(observer_count))),
        "cap_count": len(caps),
        "time_count": len(times),
        "field_names": list(field_names),
        "response_summary": {
            "min": float(np.min(matrix)),
            "mean": float(np.mean(matrix)),
            "max": float(np.max(matrix)),
            "std": float(np.std(matrix)),
            "mean_row_std": float(np.mean(np.std(matrix, axis=1))),
            "mean_col_std": float(np.mean(np.std(matrix, axis=0))),
        },
    }


def _modular_response_joint_h3_report(
    kernel: dict[str, Any],
    caps: list[RoundCap],
    expanded_caps: list[RoundCap],
    feature_rows: list[dict[str, Any]],
    matrix: np.ndarray,
    *,
    candidate_count: int,
    candidate_radius: float,
    softness: float,
    seed: int,
    pass_ratio: float,
    min_observers: int,
    min_features: int,
    heldout_fraction: float,
    anchor_weight: float,
    max_iterations: int,
    refine_steps: int,
    refine_max_rows: int | None,
    refine_max_nfev: int,
    candidate_mode: str,
    channel_mode: str,
    profile_mode: str,
    profile_time_scale: float,
    control_fit_mode: str,
) -> dict[str, Any]:
    channel_keys = _channel_keys(feature_rows, mode=str(channel_mode))
    train_mask = _train_mask(matrix.shape[1], seed=seed + 77, heldout_fraction=heldout_fraction)
    observer_axes = _observer_axes(kernel, matrix.shape[0])
    h3_fit = _joint_global_h3_fit(
        matrix,
        expanded_caps,
        feature_rows,
        train_mask=train_mask,
        observer_axes=observer_axes,
        candidate_count=candidate_count,
        candidate_radius=candidate_radius,
        softness=softness,
        seed=seed,
        anchor_weight=anchor_weight,
        max_iterations=max_iterations,
        refine_steps=refine_steps,
        refine_max_rows=refine_max_rows,
        refine_max_nfev=refine_max_nfev,
        candidate_mode=candidate_mode,
        channel_mode=str(channel_mode),
        profile_mode=str(profile_mode),
        profile_time_scale=float(profile_time_scale),
    )
    s2_report = _affine_control_fit(
        matrix,
        np.asarray(kernel.get("s2_boundary_control", np.zeros_like(matrix)), dtype=float),
        channel_keys,
        train_mask,
        label="s2_boundary_affine",
    )
    control_reports: dict[str, Any] = {}
    control_reports["shuffled_response"] = _joint_global_h3_fit(
        np.asarray(kernel.get("shuffled_response_control", kernel.get("shuffled_control", np.zeros_like(matrix))), dtype=float),
        expanded_caps,
        feature_rows,
        train_mask=train_mask,
        observer_axes=observer_axes,
        candidate_count=int(candidate_count),
        candidate_radius=candidate_radius,
        softness=softness,
        seed=seed + 101,
        anchor_weight=anchor_weight,
        max_iterations=max_iterations,
        refine_steps=refine_steps,
        refine_max_rows=refine_max_rows,
        refine_max_nfev=refine_max_nfev,
        candidate_mode=candidate_mode,
        channel_mode=str(channel_mode),
        profile_mode=str(profile_mode),
        profile_time_scale=float(profile_time_scale),
    )
    if np.any(np.linalg.norm(observer_axes, axis=1) > 1e-9):
        control_reports["shuffled_observer_labels"] = _joint_global_h3_fit(
            np.asarray(kernel.get("shuffled_observer_labels_control", np.zeros_like(matrix)), dtype=float),
            expanded_caps,
            feature_rows,
            train_mask=train_mask,
            observer_axes=observer_axes,
            candidate_count=int(candidate_count),
            candidate_radius=candidate_radius,
            softness=softness,
            seed=seed + 102,
            anchor_weight=anchor_weight,
            max_iterations=max_iterations,
            refine_steps=refine_steps,
            refine_max_rows=refine_max_rows,
            refine_max_nfev=refine_max_nfev,
            candidate_mode=candidate_mode,
            channel_mode=str(channel_mode),
            profile_mode=str(profile_mode),
            profile_time_scale=float(profile_time_scale),
        )
    control_reports["no_perturbation"] = _affine_control_fit(
        matrix,
        np.asarray(kernel.get("no_modular_flow_control", np.zeros_like(matrix)), dtype=float),
        channel_keys,
        train_mask,
        label="no_perturbation_affine",
    )
    wrong_scale_reports: dict[str, Any] = {}
    wrong_scale_controls = kernel.get("wrong_scale_controls", {}) or {}
    for index, (label, value) in enumerate(sorted(wrong_scale_controls.items())):
        if str(control_fit_mode) in {"same_h3_model", "same_h3_model_not_affine_target_fit", "h3_scale_family"}:
            wrong_self_fit = _joint_global_h3_fit(
                np.asarray(value, dtype=float),
                expanded_caps,
                feature_rows,
                train_mask=train_mask,
                observer_axes=observer_axes,
                candidate_count=int(candidate_count),
                candidate_radius=candidate_radius,
                softness=softness,
                seed=seed + 1009 + index,
                anchor_weight=anchor_weight,
                max_iterations=max_iterations,
                refine_steps=refine_steps,
                refine_max_rows=refine_max_rows,
                refine_max_nfev=refine_max_nfev,
                candidate_mode=candidate_mode,
                channel_mode=str(channel_mode),
                profile_mode=str(profile_mode),
                profile_time_scale=float(profile_time_scale),
            )
            wrong_target_eval = _evaluate_h3_fit_against_response(
                wrong_self_fit,
                matrix,
                expanded_caps,
                train_mask,
                softness=softness,
                feature_rows=feature_rows,
                profile_mode=str(profile_mode),
                profile_time_scale=float(profile_time_scale),
                label=f"wrong_scale_{label}_same_h3_model_target_eval",
            )
            wrong_scale_reports[str(label)] = {
                **wrong_target_eval,
                "wrong_scale_self_fit": _compact_fit_summary(wrong_self_fit),
                "claim_boundary": (
                    "wrong-scale control fitted with the same H3 model class on the wrong-scale response, "
                    "then evaluated against the actual response without refitting channel nuisance parameters. "
                    "This avoids comparing a target self-fit to a wrong-scale self-fit."
                ),
            }
        else:
            wrong_scale_reports[str(label)] = _affine_control_fit(
                matrix,
                np.asarray(value, dtype=float),
                channel_keys,
                train_mask,
                label=f"wrong_scale_{label}_affine",
            )
    wrong_scale_feature_audit = _strict_wrong_scale_feature_audit(
        h3_fit,
        wrong_scale_reports,
        feature_rows,
        train_mask,
    )
    h3_nrmse = _metric_value(h3_fit, "heldout_normalized_rmse")
    s2_nrmse = _metric_value(s2_report, "heldout_normalized_rmse")
    h3_beats_s2 = _beats(h3_nrmse, s2_nrmse, pass_ratio)
    h3_beats_controls = {
        label: _beats(h3_nrmse, _metric_value(report, "heldout_normalized_rmse"), pass_ratio)
        for label, report in control_reports.items()
    }
    h3_beats_wrong = {
        label: _beats(h3_nrmse, _metric_value(report, "heldout_normalized_rmse"), pass_ratio)
        for label, report in wrong_scale_reports.items()
    }
    stage_gates = _h3_response_stage_gates(
        h3_fit,
        s2_report,
        control_reports,
        wrong_scale_reports,
        wrong_scale_feature_audit,
    )
    response_summary = dict(kernel.get("response_summary", {}))
    response_degenerate = bool(float(response_summary.get("mean_row_std", 0.0)) < 1e-3)
    eligible = bool(matrix.shape[0] >= min_observers and matrix.shape[1] >= min_features)
    control_separation_receipt = bool(
        eligible
        and not response_degenerate
        and stage_gates.get(H3_RESPONSE_CONTROL_SEPARATION_RECEIPT, False)
    )
    receipt = bool(
        eligible
        and not response_degenerate
        and stage_gates.get("H3_RESPONSE_CANDIDATE_RECEIPT", False)
    )
    report = {
        "mode": "modular_response_kernel_to_h3_fit",
        "fit_mode": "joint_global",
        "model": "h3_profile_shared_channel_affine",
        "channel_mode": str(channel_mode),
        "profile_mode": str(profile_mode),
        "profile_time_scale": float(profile_time_scale),
        "objective": "heldout_normalized_rmse_and_gaussian_negloglik",
        "MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT": receipt,
        H3_RESPONSE_CONTROL_SEPARATION_RECEIPT: control_separation_receipt,
        "h3_control_separation_receipt": control_separation_receipt,
        H3_RESPONSE_CANDIDATE_RECEIPT: receipt,
        "h3_bulk_candidate_receipt": receipt,
        "observer_ids": [int(value) for value in kernel.get("observer_ids", range(matrix.shape[0]))],
        "observer_count": int(matrix.shape[0]),
        "feature_count": int(matrix.shape[1]),
        "cap_count": int(kernel.get("cap_count", len(caps))),
        "time_count": int(kernel.get("time_count", 0)),
        "field_names": list(kernel.get("field_names", [])),
        "feature_rows_sample": feature_rows[:128],
        "train_feature_count": int(np.sum(train_mask)),
        "heldout_feature_count": int(np.sum(~train_mask)),
        "h3_fit": h3_fit,
        "h3_chart_dimension_debug": _h3_chart_dimension_debug(h3_fit),
        "s2_boundary_control": {
            **s2_report,
            "h3_beats_s2_boundary": h3_beats_s2,
        },
        "control_fits": control_reports,
        "wrong_scale_control_fits": wrong_scale_reports,
        "wrong_scale_scoring_mode": str(control_fit_mode),
        "wrong_scale_feature_audit": wrong_scale_feature_audit,
        "h3_response_stage_gates": stage_gates,
        "h3_beats_controls": h3_beats_controls,
        "h3_beats_wrong_scale_controls": h3_beats_wrong,
        "response_summary": response_summary,
        "response_degenerate": response_degenerate,
        "eligibility_gate_passed": eligible,
        "pass_ratio": float(pass_ratio),
        "claim_boundary": (
            "joint finite H3 fit of the support-visible modular response kernel. Observer rows share "
            "one H3 candidate assignment each, while each time/observable channel gets shared affine "
            "nuisance parameters. The control-separation receipt requires held-out H3 score to beat S2, "
            "shuffled, no-perturbation, and aggregate wrong-scale controls. The stricter candidate receipt "
            "also requires the per-feature material wrong-scale audit to clear. Neither is a physical bulk "
            "claim without refinement and planted controls."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=DEMO,
        receipt=H3_RESPONSE_CANDIDATE_RECEIPT,
        physical_claim=False,
        observable_id="support_visible_modular_response_kernel",
        fit_objective="heldout_h3_response_beats_controls",
    )


def _joint_global_h3_fit(
    response: np.ndarray,
    caps: list[RoundCap],
    feature_rows: list[dict[str, Any]],
    *,
    train_mask: np.ndarray,
    observer_axes: np.ndarray,
    candidate_count: int,
    candidate_radius: float,
    softness: float,
    seed: int,
    anchor_weight: float,
    max_iterations: int,
    refine_steps: int,
    refine_max_rows: int | None,
    refine_max_nfev: int,
    candidate_mode: str,
    channel_mode: str,
    profile_mode: str,
    profile_time_scale: float,
) -> dict[str, Any]:
    response = np.asarray(response, dtype=float)
    normals = cap_normals(caps)
    candidates = _candidate_h3_points(
        int(candidate_count),
        seed=int(seed),
        radius=float(candidate_radius),
        mode=str(candidate_mode),
    )
    candidate_profiles = _h3_profile_matrix(
        candidates,
        caps,
        feature_rows,
        softness=float(softness),
        profile_mode=str(profile_mode),
        profile_time_scale=float(profile_time_scale),
    )
    channel_keys = _channel_keys(feature_rows, mode=str(channel_mode))
    offsets = np.zeros(response.shape[1], dtype=float)
    amplitudes = np.ones(response.shape[1], dtype=float)
    assignments = _assign_candidates(
        response,
        candidate_profiles,
        train_mask,
        offsets,
        amplitudes,
        observer_axes,
        candidates,
        anchor_weight=float(anchor_weight),
    )
    for _ in range(max(1, int(max_iterations))):
        selected_base = candidate_profiles[assignments]
        offsets, amplitudes, channel_summary = _fit_channel_affine(response, selected_base, channel_keys, train_mask)
        new_assignments = _assign_candidates(
            response,
            candidate_profiles,
            train_mask,
            offsets,
            amplitudes,
            observer_axes,
            candidates,
            anchor_weight=float(anchor_weight),
        )
        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments
    selected_base = candidate_profiles[assignments]
    offsets, amplitudes, channel_summary = _fit_channel_affine(response, selected_base, channel_keys, train_mask)
    fitted = candidates[assignments].copy()
    refinement_report = {
        "enabled": False,
        "steps": 0,
        "attempted_rows": 0,
        "improved_rows": 0,
        "median_improvement": 0.0,
        "claim_boundary": "disabled; H3 fit used discrete random candidate assignment only",
    }
    if int(refine_steps) > 0:
        total_improvements: list[float] = []
        total_attempted = 0
        for _ in range(max(1, int(refine_steps))):
            fitted, step_report = _refine_joint_h3_points(
                response,
                normals,
                caps,
                feature_rows,
                fitted,
                train_mask,
                offsets,
                amplitudes,
                observer_axes,
                candidate_radius=float(candidate_radius),
                softness=float(softness),
                profile_mode=str(profile_mode),
                profile_time_scale=float(profile_time_scale),
                anchor_weight=float(anchor_weight),
                max_rows=refine_max_rows,
                max_nfev=int(refine_max_nfev),
            )
            total_attempted += int(step_report.get("attempted_rows", 0))
            total_improvements.extend(float(value) for value in step_report.get("improvements", []))
            selected_base = _h3_profile_matrix(
                fitted,
                caps,
                feature_rows,
                softness=float(softness),
                profile_mode=str(profile_mode),
                profile_time_scale=float(profile_time_scale),
            )
            offsets, amplitudes, channel_summary = _fit_channel_affine(response, selected_base, channel_keys, train_mask)
        refinement_report = {
            "enabled": True,
            "steps": int(refine_steps),
            "attempted_rows": int(total_attempted),
            "improved_rows": int(len(total_improvements)),
            "max_nfev": int(refine_max_nfev),
            "median_improvement": float(np.median(total_improvements)) if total_improvements else 0.0,
            "claim_boundary": (
                "local tangent-space H3 refinement using training cap features only. Heldout features and "
                "all control fits receive the same refinement path."
            ),
        }
    else:
        selected_base = candidate_profiles[assignments]
    prediction = offsets[None, :] + selected_base * amplitudes[None, :]
    metrics = _heldout_metrics(response, prediction, train_mask)
    feature_rmse = _feature_rmse(response, prediction)
    return {
        "mode": "joint_global_h3_tanh_halfspace_fit",
        "candidate_count": int(candidate_count),
        "candidate_mode": str(candidate_mode),
        "candidate_radius": float(candidate_radius),
        "softness": float(softness),
        "anchor_weight": float(anchor_weight),
        "channel_mode": str(channel_mode),
        "profile_mode": str(profile_mode),
        "profile_time_scale": float(profile_time_scale),
        "channel_count": int(len(set(channel_keys))),
        "channel_summary_sample": channel_summary[:64],
        "assignment_unique_count": int(np.unique(assignments).size),
        "refinement": refinement_report,
        "best_candidate_indices": [int(value) for value in assignments[:256]],
        "sample_fitted_h3_points": [[float(x) for x in row] for row in fitted[:32]],
        "fitted_h3_points": [[float(x) for x in row] for row in fitted],
        "feature_rmse": [float(value) for value in feature_rmse],
        "feature_rmse_sample": [float(value) for value in feature_rmse[:128]],
        "offsets": [float(value) for value in offsets],
        "amplitudes": [float(value) for value in amplitudes],
        **metrics,
    }


def _evaluate_h3_fit_against_response(
    fit: dict[str, Any],
    response: np.ndarray,
    caps: list[RoundCap],
    train_mask: np.ndarray,
    *,
    softness: float,
    feature_rows: list[dict[str, Any]],
    profile_mode: str,
    profile_time_scale: float,
    label: str,
) -> dict[str, Any]:
    response = np.asarray(response, dtype=float)
    fitted = np.asarray(fit.get("fitted_h3_points", []), dtype=float)
    offsets = np.asarray(fit.get("offsets", []), dtype=float)
    amplitudes = np.asarray(fit.get("amplitudes", []), dtype=float)
    if (
        response.ndim != 2
        or fitted.ndim != 2
        or fitted.shape[0] != response.shape[0]
        or fitted.shape[1] != 4
        or offsets.size != response.shape[1]
        or amplitudes.size != response.shape[1]
    ):
        return {
            "mode": label,
            "reason": "shape_mismatch",
            "heldout_normalized_rmse": float("inf"),
            "feature_rmse": [],
            "feature_rmse_sample": [],
        }
    base = _h3_profile_matrix(
        fitted,
        caps,
        feature_rows,
        softness=float(softness),
        profile_mode=str(profile_mode),
        profile_time_scale=float(profile_time_scale),
    )
    prediction = offsets[None, :] + base * amplitudes[None, :]
    feature_rmse = _feature_rmse(response, prediction)
    return {
        "mode": label,
        "source_fit_mode": str(fit.get("mode", "")),
        "feature_rmse": [float(value) for value in feature_rmse],
        "feature_rmse_sample": [float(value) for value in feature_rmse[:128]],
        **_heldout_metrics(response, prediction, train_mask),
    }


def _compact_fit_summary(fit: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "mode",
        "candidate_count",
        "candidate_mode",
        "candidate_radius",
        "softness",
        "profile_mode",
        "profile_time_scale",
        "channel_mode",
        "assignment_unique_count",
        "train_rmse",
        "heldout_rmse",
        "heldout_std",
        "heldout_normalized_rmse",
        "heldout_explained_variance",
    ]
    return {key: fit.get(key) for key in keys if key in fit}


def _affine_control_fit(
    response: np.ndarray,
    control_basis: np.ndarray,
    channel_keys: list[str],
    train_mask: np.ndarray,
    *,
    label: str,
) -> dict[str, Any]:
    response = np.asarray(response, dtype=float)
    control_basis = np.asarray(control_basis, dtype=float)
    if response.shape != control_basis.shape:
        return {"mode": label, "reason": "shape_mismatch", "heldout_normalized_rmse": float("inf")}
    offsets, amplitudes, channel_summary = _fit_channel_affine(response, control_basis, channel_keys, train_mask)
    prediction = offsets[None, :] + control_basis * amplitudes[None, :]
    feature_rmse = _feature_rmse(response, prediction)
    return {
        "mode": label,
        "channel_count": int(len(set(channel_keys))),
        "channel_summary_sample": channel_summary[:64],
        "feature_rmse": [float(value) for value in feature_rmse],
        "feature_rmse_sample": [float(value) for value in feature_rmse[:128]],
        **_heldout_metrics(response, prediction, train_mask),
    }


def _feature_rmse(response: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    response = np.asarray(response, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if response.shape != prediction.shape or response.ndim != 2 or response.size == 0:
        return np.zeros(0, dtype=float)
    diff = response - prediction
    return np.sqrt(np.mean(diff * diff, axis=0))


def _strict_wrong_scale_feature_audit(
    h3_fit: dict[str, Any],
    wrong_scale_reports: dict[str, Any],
    feature_rows: list[dict[str, Any]],
    train_mask: np.ndarray,
    *,
    max_group_rows: int = 96,
    material_margin: float = 0.02,
    material_absolute_margin: float = 0.0,
) -> dict[str, Any]:
    h3_rmse = np.asarray(h3_fit.get("feature_rmse", []), dtype=float)
    if h3_rmse.size == 0 or not wrong_scale_reports:
        return {
            "mode": "strict_wrong_scale_feature_audit",
            "eligible": False,
            "reason": "missing_feature_rmse_or_wrong_scale_controls",
            "claim_boundary": (
                "diagnostic only: requires joint H3 feature residuals and wrong-scale feature residuals"
            ),
        }
    control_rmse: dict[str, np.ndarray] = {}
    feature_count = int(h3_rmse.size)
    for label, report in sorted(wrong_scale_reports.items()):
        values = np.asarray(report.get("feature_rmse", []), dtype=float)
        if values.size:
            feature_count = min(feature_count, int(values.size))
            control_rmse[str(label)] = values
    if feature_count == 0 or not control_rmse:
        return {
            "mode": "strict_wrong_scale_feature_audit",
            "eligible": False,
            "reason": "wrong_scale_feature_residuals_missing",
            "claim_boundary": (
                "diagnostic only: requires joint H3 feature residuals and wrong-scale feature residuals"
            ),
        }
    h3_rmse = h3_rmse[:feature_count]
    control_rmse = {label: values[:feature_count] for label, values in control_rmse.items()}
    train = np.asarray(train_mask, dtype=bool)
    if train.size < feature_count:
        audited = np.ones(feature_count, dtype=bool)
    else:
        audited = ~train[:feature_count]
        if not np.any(audited):
            audited = np.ones(feature_count, dtype=bool)
    labels = ["2pi_h3_fit", *sorted(control_rmse)]
    winner_counts = {label: 0 for label in labels}
    material_winner_counts = {label: 0 for label in labels}
    group_rows: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    audited_indices = np.flatnonzero(audited)
    for feature_index in audited_indices:
        values = {"2pi_h3_fit": float(h3_rmse[feature_index])}
        values.update({label: float(residuals[feature_index]) for label, residuals in control_rmse.items()})
        winner = min(values, key=values.get)
        winner_counts[winner] = winner_counts.get(winner, 0) + 1
        h3_value = float(values["2pi_h3_fit"])
        best_wrong_label = min(control_rmse, key=lambda label: float(control_rmse[label][feature_index]))
        best_wrong = float(control_rmse[best_wrong_label][feature_index])
        material_winner = "2pi_h3_fit"
        if best_wrong + float(material_absolute_margin) < (1.0 - float(material_margin)) * h3_value:
            material_winner = str(best_wrong_label)
        elif h3_value > 0.0 and h3_value + float(material_absolute_margin) < (1.0 - float(material_margin)) * best_wrong:
            material_winner = "2pi_h3_fit"
        material_winner_counts[material_winner] = material_winner_counts.get(material_winner, 0) + 1
        row = feature_rows[feature_index] if feature_index < len(feature_rows) else {}
        group_key = (
            int(row.get("cap_index", -1)),
            int(row.get("time_index", -1)),
            str(row.get("observable", row.get("field", "unknown"))),
            str(row.get("feature_type", "scalar")),
        )
        group = group_rows.setdefault(
            group_key,
            {
                "cap_index": group_key[0],
                "time_index": group_key[1],
                "time": float(row.get("time", 0.0)) if row else 0.0,
                "observable": group_key[2],
                "feature_type": group_key[3],
                "feature_count": 0,
                "winner_counts": {label: 0 for label in labels},
                "material_winner_counts": {label: 0 for label in labels},
                "_h3": [],
                "_best_wrong": [],
            },
        )
        group["feature_count"] += 1
        group["winner_counts"][winner] = group["winner_counts"].get(winner, 0) + 1
        group["material_winner_counts"][material_winner] = group["material_winner_counts"].get(material_winner, 0) + 1
        group["_h3"].append(float(h3_rmse[feature_index]))
        group["_best_wrong"].append(float(best_wrong))
    wrong_win_count = sum(count for label, count in winner_counts.items() if label != "2pi_h3_fit")
    material_wrong_win_count = sum(
        count for label, count in material_winner_counts.items() if label != "2pi_h3_fit"
    )
    if audited_indices.size:
        audited_h3 = np.maximum(0.0, h3_rmse[audited_indices])
        controls = np.vstack(
            [
                np.asarray(values[:feature_count], dtype=float)[audited_indices]
                for values in control_rmse.values()
            ]
        )
        audited_best_wrong = np.min(controls, axis=0)
        advantage = np.maximum(
            0.0,
            (1.0 - float(material_margin)) * audited_h3
            - audited_best_wrong
            - float(material_absolute_margin),
        )
        h3_mass = float(np.sum(audited_h3))
        h3_energy = float(np.sum(audited_h3 * audited_h3))
        material_wrong_mask = advantage > 0.0
        advantage_mass_fraction = float(np.sum(advantage) / h3_mass) if h3_mass > 0.0 else 0.0
        advantage_energy_fraction = (
            float(np.sum(advantage * advantage) / h3_energy) if h3_energy > 0.0 else 0.0
        )
        residual_mass_fraction = (
            float(np.sum(audited_h3[material_wrong_mask]) / h3_mass) if h3_mass > 0.0 else 0.0
        )
        residual_energy_fraction = (
            float(np.sum((audited_h3[material_wrong_mask]) ** 2) / h3_energy)
            if h3_energy > 0.0
            else 0.0
        )
    else:
        advantage_mass_fraction = 0.0
        advantage_energy_fraction = 0.0
        residual_mass_fraction = 0.0
        residual_energy_fraction = 0.0
    group_output: list[dict[str, Any]] = []
    for group in group_rows.values():
        count = max(1, int(group["feature_count"]))
        wrong_count = sum(value for label, value in group["winner_counts"].items() if label != "2pi_h3_fit")
        material_wrong_count = sum(
            value for label, value in group["material_winner_counts"].items() if label != "2pi_h3_fit"
        )
        group_output.append(
            {
                "cap_index": int(group["cap_index"]),
                "time_index": int(group["time_index"]),
                "time": float(group["time"]),
                "observable": str(group["observable"]),
                "feature_type": str(group["feature_type"]),
                "feature_count": int(group["feature_count"]),
                "winner_counts": {str(label): int(value) for label, value in group["winner_counts"].items()},
                "material_winner_counts": {
                    str(label): int(value) for label, value in group["material_winner_counts"].items()
                },
                "wrong_scale_win_fraction": float(wrong_count / count),
                "material_wrong_scale_win_fraction": float(material_wrong_count / count),
                "median_h3_feature_rmse": float(np.median(group["_h3"])) if group["_h3"] else None,
                "median_best_wrong_scale_rmse": float(np.median(group["_best_wrong"])) if group["_best_wrong"] else None,
            }
        )
    group_output.sort(
        key=lambda row: (
            -float(row["wrong_scale_win_fraction"]),
            -int(row["feature_count"]),
            int(row["cap_index"]),
            int(row["time_index"]),
            str(row["observable"]),
        )
    )
    audited_count = int(audited_indices.size)
    margin_sweep = _wrong_scale_material_margin_sweep(
        h3_rmse,
        control_rmse,
        audited_indices,
        relative_margin=float(material_margin),
    )
    return {
        "mode": "strict_wrong_scale_feature_audit",
        "eligible": True,
        "audited_feature_set": "heldout_features" if train.size >= feature_count and np.any(~train[:feature_count]) else "all_features",
        "feature_count": int(feature_count),
        "audited_feature_count": audited_count,
        "material_margin": float(material_margin),
        "material_absolute_margin": float(material_absolute_margin),
        "material_absolute_margin_sweep": margin_sweep,
        "winner_counts": {str(label): int(value) for label, value in winner_counts.items()},
        "material_winner_counts": {str(label): int(value) for label, value in material_winner_counts.items()},
        "wrong_scale_win_count": int(wrong_win_count),
        "wrong_scale_win_fraction": float(wrong_win_count / max(audited_count, 1)),
        "material_wrong_scale_win_count": int(material_wrong_win_count),
        "material_wrong_scale_win_fraction": float(material_wrong_win_count / max(audited_count, 1)),
        "material_wrong_scale_advantage_mass_fraction": advantage_mass_fraction,
        "material_wrong_scale_advantage_energy_fraction": advantage_energy_fraction,
        "material_wrong_scale_residual_mass_fraction": residual_mass_fraction,
        "material_wrong_scale_residual_energy_fraction": residual_energy_fraction,
        "two_pi_h3_fit_win_fraction": float(winner_counts.get("2pi_h3_fit", 0) / max(audited_count, 1)),
        "material_two_pi_h3_fit_win_fraction": float(
            material_winner_counts.get("2pi_h3_fit", 0) / max(audited_count, 1)
        ),
        "red_flag_wrong_scale_wins": bool(wrong_win_count > 0),
        "material_red_flag_wrong_scale_wins": bool(material_wrong_win_count > 0),
        "worst_groups": group_output[: int(max_group_rows)],
        "claim_boundary": (
            "diagnostic only: per-feature heldout residual audit for H3 fit versus wrong-scale controls. "
            "It identifies cap/time/observable locations where wrong normalization is competitive; it does "
            "not establish or reject a physical bulk by itself. The material_* fields count only wrong-scale "
            "wins that improve over the 2pi/H3 feature residual by the declared relative margin; strict wins "
            "remain visible as red flags; the strict material gate uses residual-energy advantage so "
            "unsupported singleton bins cannot veto an otherwise controlled finite response kernel."
        ),
    }


def _wrong_scale_material_margin_sweep(
    h3_rmse: np.ndarray,
    control_rmse: dict[str, np.ndarray],
    audited_indices: np.ndarray,
    *,
    relative_margin: float,
) -> list[dict[str, Any]]:
    if h3_rmse.size == 0 or not control_rmse or audited_indices.size == 0:
        return []
    controls = np.vstack([np.asarray(values[: h3_rmse.size], dtype=float) for values in control_rmse.values()])
    best_wrong = np.min(controls, axis=0)
    audited = np.asarray(audited_indices, dtype=np.int64)
    sweep: list[dict[str, Any]] = []
    for absolute_margin in (0.0, 1e-6, 1e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 7.5e-2, 1e-1, 1.25e-1, 1.5e-1, 2e-1):
        material_wrong = best_wrong[audited] + float(absolute_margin) < (
            1.0 - float(relative_margin)
        ) * h3_rmse[audited]
        sweep.append(
            {
                "absolute_margin": float(absolute_margin),
                "material_wrong_scale_win_count": int(np.sum(material_wrong)),
                "material_wrong_scale_win_fraction": float(np.mean(material_wrong)),
            }
        )
    return sweep


def _h3_response_stage_gates(
    h3_fit: dict[str, Any],
    s2_report: dict[str, Any],
    control_reports: dict[str, Any],
    wrong_scale_reports: dict[str, Any],
    wrong_scale_feature_audit: dict[str, Any],
    *,
    min_explained_variance: float = 0.03,
    control_margin: float = 0.01,
    geometry_margin: float = 0.005,
    wrong_scale_margin: float = 0.005,
    max_material_wrong_fraction: float = 0.05,
) -> dict[str, Any]:
    h3_nrmse = _metric_value(h3_fit, "heldout_normalized_rmse")
    h3_ev = _metric_value(h3_fit, "heldout_explained_variance")
    no_perturbation = _metric_value(control_reports.get("no_perturbation", {}), "heldout_normalized_rmse")
    shuffled = _metric_value(control_reports.get("shuffled_response", {}), "heldout_normalized_rmse")
    s2 = _metric_value(s2_report, "heldout_normalized_rmse")
    signal_gate = bool(
        np.isfinite(h3_ev)
        and h3_ev > float(min_explained_variance)
        and np.isfinite(h3_nrmse)
        and (
            not np.isfinite(no_perturbation)
            or h3_nrmse < no_perturbation - float(control_margin)
        )
        and (
            not np.isfinite(shuffled)
            or h3_nrmse < shuffled - float(control_margin)
        )
    )
    geometry_gate = bool(
        np.isfinite(h3_nrmse)
        and (
            not np.isfinite(s2)
            or h3_nrmse < s2 - float(geometry_margin)
        )
    )
    wrong_scores = {
        str(label): _metric_value(report, "heldout_normalized_rmse")
        for label, report in wrong_scale_reports.items()
    }
    aggregate_wrong_gate = bool(
        not wrong_scores
        or all(
            np.isfinite(h3_nrmse)
            and (not np.isfinite(score) or h3_nrmse < score - float(wrong_scale_margin))
            for score in wrong_scores.values()
        )
    )
    raw_material_fraction = wrong_scale_feature_audit.get("material_wrong_scale_win_fraction")
    if raw_material_fraction is None:
        raw_material_fraction = wrong_scale_feature_audit.get("wrong_scale_win_fraction")
    raw_material_fraction_value = (
        float(raw_material_fraction) if raw_material_fraction is not None else 0.0
    )
    material_gate_metric = "material_wrong_scale_advantage_energy_fraction"
    material_gate_value = wrong_scale_feature_audit.get(material_gate_metric)
    if material_gate_value is None:
        material_gate_metric = "material_wrong_scale_win_fraction"
        material_gate_value = raw_material_fraction_value
    material_gate_value = float(material_gate_value) if material_gate_value is not None else 0.0
    feature_gate = bool(material_gate_value < float(max_material_wrong_fraction))
    control_separation_receipt = bool(signal_gate and geometry_gate and aggregate_wrong_gate)
    receipt = bool(control_separation_receipt and feature_gate)
    return {
        "mode": "staged_h3_response_candidate_gate",
        "signal_gate": signal_gate,
        "geometry_gate": geometry_gate,
        "aggregate_wrong_scale_gate": aggregate_wrong_gate,
        "material_feature_gate": feature_gate,
        "intermediate_control_separation_receipt": control_separation_receipt,
        H3_RESPONSE_CONTROL_SEPARATION_RECEIPT: control_separation_receipt,
        H3_RESPONSE_CANDIDATE_RECEIPT: receipt,
        "h3_heldout_normalized_rmse": h3_nrmse if np.isfinite(h3_nrmse) else None,
        "h3_heldout_explained_variance": h3_ev if np.isfinite(h3_ev) else None,
        "s2_boundary_normalized_rmse": s2 if np.isfinite(s2) else None,
        "no_perturbation_normalized_rmse": no_perturbation if np.isfinite(no_perturbation) else None,
        "shuffled_response_normalized_rmse": shuffled if np.isfinite(shuffled) else None,
        "wrong_scale_normalized_rmse": {
            label: score if np.isfinite(score) else None for label, score in wrong_scores.items()
        },
        "material_wrong_scale_win_fraction": raw_material_fraction_value,
        "material_wrong_scale_gate_metric": material_gate_metric,
        "material_wrong_scale_gate_value": material_gate_value,
        "material_wrong_scale_advantage_energy_fraction": wrong_scale_feature_audit.get(
            "material_wrong_scale_advantage_energy_fraction"
        ),
        "material_wrong_scale_advantage_mass_fraction": wrong_scale_feature_audit.get(
            "material_wrong_scale_advantage_mass_fraction"
        ),
        "thresholds": {
            "min_explained_variance": float(min_explained_variance),
            "control_margin": float(control_margin),
            "geometry_margin": float(geometry_margin),
            "wrong_scale_margin": float(wrong_scale_margin),
            "max_material_wrong_fraction": float(max_material_wrong_fraction),
        },
        "claim_boundary": (
            "staged internal gate for H3 response candidates. It first checks that the response model "
            "has positive heldout signal, then checks H3-vs-S2 geometry, aggregate wrong-normalization "
            "separation, and material wrong-scale residual advantage. Raw per-feature wrong-scale wins "
            "remain reported as red flags, but the strict material gate is keyed to residual-energy "
            "advantage so singleton feature bins do not outweigh the finite response kernel. The "
            "intermediate control-separation receipt is a chart-response precursor only; the strict H3 "
            "candidate receipt remains blocked until the material audit clears. This is still not a "
            "physical bulk claim."
        ),
    }


def _pairwise_residuals(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape or left.size == 0:
        return np.zeros(0, dtype=float)
    diff = left - right
    return np.sqrt(np.mean(diff * diff, axis=1))


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _beats(left: float, right: float, pass_ratio: float) -> bool:
    return bool(np.isfinite(left) and np.isfinite(right) and left < float(pass_ratio) * right)


def _tanh_halfspace_profile(points: np.ndarray, normals: np.ndarray, *, softness: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    signed = -np.outer(points[:, 0], normals[:, 0]) + points[:, 1:] @ normals[:, 1:].T
    return np.tanh(np.clip(signed / max(float(softness), 1e-9), -60.0, 60.0))


def _h3_profile_matrix(
    points: np.ndarray,
    caps: list[RoundCap],
    feature_rows: list[dict[str, Any]],
    *,
    softness: float,
    profile_mode: str,
    profile_time_scale: float,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 4 or not feature_rows:
        return np.zeros((points.shape[0] if points.ndim == 2 else 0, 0), dtype=float)
    feature_caps = _feature_caps(caps, feature_rows)
    normalized_mode = str(profile_mode or "static_halfspace").lower().replace("-", "_")
    if normalized_mode in {"static", "static_halfspace", "halfspace"}:
        return _tanh_halfspace_profile(points, cap_normals(feature_caps), softness=float(softness))
    columns: list[np.ndarray] = []
    width = max(float(softness) * 4.0, 0.25)
    for cap, row in zip(feature_caps, feature_rows, strict=False):
        rapidity = _cap_fixed_pair_rapidity(points, cap)
        modular_s = float(profile_time_scale) * float(row.get("time", 0.0))
        shifted = np.tanh(np.clip((rapidity + modular_s) / width, -60.0, 60.0))
        unshifted = np.tanh(np.clip(rapidity / width, -60.0, 60.0))
        if normalized_mode in {"modular_time_rapidity", "rapidity_shift", "bw_rapidity"}:
            profile = shifted
        elif normalized_mode in {"modular_time_derivative", "bw_time_derivative"}:
            profile = (shifted - unshifted) / max(abs(modular_s), 1e-9)
        elif normalized_mode in {"modular_time_symmetric_delta", "bw_symmetric_delta"}:
            reverse = np.tanh(np.clip((rapidity - modular_s) / width, -60.0, 60.0))
            profile = shifted - reverse
        else:
            profile = shifted - unshifted
        columns.append(np.asarray(profile, dtype=float))
    return np.vstack(columns).T if columns else np.zeros((points.shape[0], 0), dtype=float)


def _feature_caps(caps: list[RoundCap], feature_rows: list[dict[str, Any]]) -> list[RoundCap]:
    if len(caps) == len(feature_rows):
        return list(caps)
    if not caps:
        return []
    out: list[RoundCap] = []
    for row in feature_rows:
        index = int(row.get("cap_index", 0))
        index = max(0, min(index, len(caps) - 1))
        out.append(caps[index])
    return out


def _cap_fixed_pair_rapidity(points: np.ndarray, cap: RoundCap) -> np.ndarray:
    cap = cap.normalized()
    tangent = np.asarray(cap.tangent, dtype=float)
    tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-12)
    plus = np.cos(cap.theta0) * cap.axis + np.sin(cap.theta0) * tangent
    minus = np.cos(cap.theta0) * cap.axis - np.sin(cap.theta0) * tangent
    plus = plus / max(float(np.linalg.norm(plus)), 1e-12)
    minus = minus / max(float(np.linalg.norm(minus)), 1e-12)
    plus_distance = np.maximum(points[:, 0] - points[:, 1:] @ plus, 1e-12)
    minus_distance = np.maximum(points[:, 0] - points[:, 1:] @ minus, 1e-12)
    return np.log(plus_distance / minus_distance)


def _channel_keys(feature_rows: list[dict[str, Any]], *, mode: str = "time_observable_class") -> list[str]:
    normalized_mode = str(mode or "time_observable_class").lower().replace("-", "_")
    keys = []
    for row in feature_rows:
        observable = row.get("observable", row.get("field", "field"))
        feature_type = row.get("feature_type", "")
        target = row.get("target_class", "")
        cap = int(row.get("cap_index", 0))
        time = int(row.get("time_index", 0))
        if normalized_mode in {"time_observable_class", "time_observable_feature_class"}:
            key = f"t{time}:{observable}:{feature_type}:{target}"
        elif normalized_mode in {"cap_time_observable_class", "cap_time_observable_feature_class"}:
            key = f"c{cap}:t{time}:{observable}:{feature_type}:{target}"
        elif normalized_mode in {"cap_observable_class", "cap_observable_feature_class", "time_tied_cap"}:
            key = f"c{cap}:{observable}:{feature_type}:{target}"
        elif normalized_mode in {"observable_class", "observable_feature_class", "time_tied_global"}:
            key = f"{observable}:{feature_type}:{target}"
        elif normalized_mode in {"observable_type", "time_tied_observable_type"}:
            key = f"{observable}:{feature_type}"
        else:
            key = f"t{time}:{observable}:{feature_type}:{target}"
        keys.append(key)
    return keys


def _train_mask(feature_count: int, *, seed: int, heldout_fraction: float) -> np.ndarray:
    if feature_count <= 1:
        return np.ones(int(feature_count), dtype=bool)
    rng = np.random.default_rng(seed)
    mask = np.ones(int(feature_count), dtype=bool)
    heldout_count = max(1, int(round(float(heldout_fraction) * int(feature_count))))
    heldout_count = min(heldout_count, int(feature_count) - 1)
    heldout = rng.choice(int(feature_count), size=heldout_count, replace=False)
    mask[heldout] = False
    return mask


def _observer_axes(kernel: dict[str, Any], observer_count: int) -> np.ndarray:
    axes = np.asarray(kernel.get("observer_axes", np.zeros((0, 3))), dtype=float)
    if axes.shape != (int(observer_count), 3):
        return np.zeros((int(observer_count), 3), dtype=float)
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    return axes / np.maximum(norms, 1e-12)


def _assign_candidates(
    response: np.ndarray,
    candidate_profiles: np.ndarray,
    train_mask: np.ndarray,
    offsets: np.ndarray,
    amplitudes: np.ndarray,
    observer_axes: np.ndarray,
    candidates: np.ndarray,
    *,
    anchor_weight: float,
) -> np.ndarray:
    train_indices = np.flatnonzero(train_mask)
    if train_indices.size == 0:
        return np.zeros(response.shape[0], dtype=np.int64)
    predicted = offsets[None, :] + candidate_profiles * amplitudes[None, :]
    predicted_train = predicted[:, train_indices]
    assignments = np.zeros(response.shape[0], dtype=np.int64)
    anchor_cost = _anchor_cost(candidates, observer_axes)
    candidate_norm = np.mean(predicted_train * predicted_train, axis=1)
    scale = 2.0 / float(train_indices.size)
    chunk_size = max(1, min(256, int(response.shape[0])))
    predicted_train_t = predicted_train.T
    for start in range(0, response.shape[0], chunk_size):
        stop = min(start + chunk_size, response.shape[0])
        response_train = response[start:stop, :][:, train_indices]
        response_norm = np.mean(response_train * response_train, axis=1)
        costs = (
            response_norm[:, None]
            + candidate_norm[None, :]
            - scale * (response_train @ predicted_train_t)
        )
        if anchor_cost.size:
            costs = costs + float(anchor_weight) * anchor_cost[start:stop]
        assignments[start:stop] = np.argmin(costs, axis=1).astype(np.int64, copy=False)
    return assignments


def _anchor_cost(candidates: np.ndarray, observer_axes: np.ndarray) -> np.ndarray:
    if observer_axes.size == 0:
        return np.zeros((0, candidates.shape[0]), dtype=float)
    spatial = candidates[:, 1:]
    spatial_norm = np.linalg.norm(spatial, axis=1, keepdims=True)
    directions = spatial / np.maximum(spatial_norm, 1e-12)
    similarity = np.clip(observer_axes @ directions.T, -1.0, 1.0)
    return 1.0 - similarity


def _refine_joint_h3_points(
    response: np.ndarray,
    normals: np.ndarray,
    caps: list[RoundCap],
    feature_rows: list[dict[str, Any]],
    fitted: np.ndarray,
    train_mask: np.ndarray,
    offsets: np.ndarray,
    amplitudes: np.ndarray,
    observer_axes: np.ndarray,
    *,
    candidate_radius: float,
    softness: float,
    profile_mode: str,
    profile_time_scale: float,
    anchor_weight: float,
    max_rows: int | None,
    max_nfev: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    fitted = np.asarray(fitted, dtype=float).copy()
    if fitted.ndim != 2 or fitted.shape[1] != 4 or response.size == 0:
        return fitted, {"attempted_rows": 0, "improved_rows": 0, "improvements": []}
    train_indices = np.flatnonzero(train_mask)
    if train_indices.size == 0:
        return fitted, {"attempted_rows": 0, "improved_rows": 0, "improvements": []}
    predicted = offsets[None, :] + _h3_profile_matrix(
        fitted,
        caps,
        feature_rows,
        softness=float(softness),
        profile_mode=str(profile_mode),
        profile_time_scale=float(profile_time_scale),
    ) * amplitudes[None, :]
    base_errors = np.mean((response[:, train_indices] - predicted[:, train_indices]) ** 2, axis=1)
    row_count = int(fitted.shape[0]) if max_rows is None else min(int(max_rows), int(fitted.shape[0]))
    order = np.argsort(base_errors)[::-1][:row_count]
    refined = fitted.copy()
    improvements: list[float] = []
    radius = max(float(candidate_radius), 1e-6)
    width = max(float(softness), 1e-9)
    anchor = np.asarray(observer_axes, dtype=float)
    for row_index in order:
        initial = np.clip(h3_tangent_from_point(refined[int(row_index)]), -radius, radius)
        target = response[int(row_index), train_indices]
        target_offsets = offsets[train_indices]
        target_amplitudes = amplitudes[train_indices]
        target_normals = normals[train_indices]
        target_feature_rows = [feature_rows[int(index)] for index in train_indices]
        if len(caps) == len(feature_rows):
            target_caps = [caps[int(index)] for index in train_indices]
        else:
            target_caps = [caps[int(row.get("cap_index", 0))] for row in target_feature_rows]
        target_rows = [
            {**row, "cap_index": index}
            for index, row in enumerate(target_feature_rows)
        ]

        def objective(vector: np.ndarray) -> np.ndarray:
            point = h3_point_from_tangent(vector)
            if str(profile_mode) == "static_halfspace":
                signed = -point[0] * target_normals[:, 0] + target_normals[:, 1:] @ point[1:]
                profile = np.tanh(np.clip(signed / width, -60.0, 60.0))
            else:
                profile = _h3_profile_matrix(
                    point[None, :],
                    target_caps,
                    target_rows,
                    softness=float(softness),
                    profile_mode=str(profile_mode),
                    profile_time_scale=float(profile_time_scale),
                )[0]
            residual = target_offsets + profile * target_amplitudes - target
            if anchor.size and float(anchor_weight) > 0.0:
                norm = max(float(np.linalg.norm(vector)), 1e-12)
                direction = vector / norm
                anchor_residual = np.sqrt(float(anchor_weight)) * (direction - anchor[int(row_index)])
                residual = np.concatenate([residual, anchor_residual])
            return residual

        before = float(np.mean(objective(initial) ** 2))
        result = least_squares(
            objective,
            initial,
            bounds=(-radius, radius),
            max_nfev=max(1, int(max_nfev)),
            method="trf",
        )
        if not result.success and not np.isfinite(result.cost):
            continue
        after = float(np.mean(objective(result.x) ** 2))
        if np.isfinite(after) and after + 1e-12 < before:
            refined[int(row_index)] = h3_point_from_tangent(np.asarray(result.x, dtype=float))
            improvements.append(before - after)
    return refined, {
        "attempted_rows": int(row_count),
        "improved_rows": int(len(improvements)),
        "improvements": improvements,
    }


def _candidate_h3_points(count: int, *, seed: int, radius: float, mode: str) -> np.ndarray:
    normalized = str(mode or "random").lower().replace("-", "_")
    if normalized in {"random", "random_ball"}:
        return random_h3_points(int(count), seed=int(seed), radius=float(radius))
    if normalized not in {"fibonacci_ball", "deterministic_ball", "low_discrepancy_ball"}:
        return random_h3_points(int(count), seed=int(seed), radius=float(radius))
    count = max(1, int(count))
    if count == 1:
        return np.asarray([h3_point_from_tangent(np.zeros(3, dtype=float))], dtype=float)
    directions = fibonacci_sphere_points(count)
    # A deterministic volume-like radial law. The small angular/radial phase
    # shift avoids repeatedly assigning the largest radii to the same longitude
    # bands when count changes between refits.
    indices = np.arange(count, dtype=float)
    phase = (np.sqrt(5.0) - 1.0) / 2.0
    radial_rank = (indices * phase) % 1.0
    radii = float(radius) * np.power((radial_rank * (count - 1) + 0.5) / count, 1.0 / 3.0)
    tangents = directions * radii[:, None]
    tangents[0] = 0.0
    return np.vstack([h3_point_from_tangent(row) for row in tangents])


def _fit_channel_affine(
    response: np.ndarray,
    base: np.ndarray,
    channel_keys: list[str],
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    ridge = 1e-3
    offset_clip = 3.0
    amplitude_clip = 3.0
    offsets = np.zeros(response.shape[1], dtype=float)
    amplitudes = np.ones(response.shape[1], dtype=float)
    summary: list[dict[str, Any]] = []
    keys = np.asarray(channel_keys, dtype=object)
    for key in sorted(set(channel_keys)):
        cols = np.flatnonzero((keys == key) & train_mask)
        all_cols = np.flatnonzero(keys == key)
        if cols.size == 0:
            offsets[all_cols] = 0.0
            amplitudes[all_cols] = 1.0
            continue
        x = base[:, cols].reshape(-1)
        y = response[:, cols].reshape(-1)
        design = np.column_stack([np.ones_like(x), x])
        lhs = design.T @ design + ridge * np.diag([0.0, 1.0])
        rhs = design.T @ y
        try:
            coef = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        offset = float(np.clip(coef[0], -offset_clip, offset_clip))
        amplitude = float(np.clip(coef[1], -amplitude_clip, amplitude_clip))
        offsets[all_cols] = offset
        amplitudes[all_cols] = amplitude
        summary.append(
            {
                "channel": str(key),
                "feature_count": int(all_cols.size),
                "train_feature_count": int(cols.size),
                "offset": offset,
                "amplitude": amplitude,
                "raw_offset": float(coef[0]),
                "raw_amplitude": float(coef[1]),
                "ridge": float(ridge),
                "offset_clip": float(offset_clip),
                "amplitude_clip": float(amplitude_clip),
            }
        )
    return offsets, amplitudes, summary


def _heldout_metrics(response: np.ndarray, prediction: np.ndarray, train_mask: np.ndarray) -> dict[str, float]:
    response = np.asarray(response, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    test_mask = ~train_mask
    if not np.any(test_mask):
        test_mask = train_mask
    train_err = response[:, train_mask] - prediction[:, train_mask]
    test_values = response[:, test_mask]
    test_err = test_values - prediction[:, test_mask]
    train_rmse = float(np.sqrt(np.mean(train_err * train_err))) if train_err.size else 0.0
    heldout_rmse = float(np.sqrt(np.mean(test_err * test_err))) if test_err.size else 0.0
    heldout_std = float(np.std(test_values)) if test_values.size else 0.0
    normalized_rmse = heldout_rmse / max(heldout_std, 1e-9)
    sigma2 = max(float(np.mean(train_err * train_err)) if train_err.size else heldout_rmse * heldout_rmse, 1e-9)
    negloglik = float(0.5 * np.mean(np.log(2.0 * np.pi * sigma2) + (test_err * test_err) / sigma2)) if test_err.size else 0.0
    sst = float(np.sum((test_values - float(np.mean(test_values))) ** 2)) if test_values.size else 0.0
    sse = float(np.sum(test_err * test_err)) if test_err.size else 0.0
    explained = 1.0 - sse / max(sst, 1e-12)
    return {
        "train_rmse": train_rmse,
        "heldout_rmse": heldout_rmse,
        "heldout_std": heldout_std,
        "heldout_normalized_rmse": float(normalized_rmse),
        "heldout_negloglik": negloglik,
        "heldout_explained_variance": float(explained),
    }


def _metric_value(report: dict[str, Any], key: str) -> float:
    value = report.get(key)
    if value is None:
        return float("nan")
    return float(value)
