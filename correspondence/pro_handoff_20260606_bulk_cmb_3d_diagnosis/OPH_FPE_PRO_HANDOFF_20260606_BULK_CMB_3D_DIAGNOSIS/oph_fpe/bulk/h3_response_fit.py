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
from oph_fpe.claims import DEMO, H3_RESPONSE_CANDIDATE_RECEIPT, with_claim_metadata
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
    refine_steps: int = 0,
    refine_max_rows: int | None = None,
    refine_max_nfev: int = 48,
    candidate_mode: str = "random",
) -> dict[str, Any]:
    matrix = np.asarray(kernel.get("matrix", np.zeros((0, 0))), dtype=float)
    feature_rows = list(kernel.get("feature_rows", []))
    if matrix.size == 0 or not feature_rows or not caps:
        return with_claim_metadata(
            {
                "mode": "modular_response_kernel_to_h3_fit",
                "MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT": False,
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
        min_features=int(min_features),
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
        )
        report["feature_selection"] = feature_selection_report
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
        candidate_count=max(128, min(int(candidate_count), 1024)),
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
    min_features: int,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    matrix = np.asarray(matrix, dtype=float)
    original_count = int(matrix.shape[1]) if matrix.ndim == 2 else 0
    normalized_mode = str(mode or "none").lower().replace("-", "_")
    if normalized_mode in {"none", "off", "false"} or original_count == 0:
        return kernel, matrix, feature_rows, {
            "mode": "none",
            "original_feature_count": original_count,
            "selected_feature_count": original_count,
            "min_feature_std": float(min_std),
            "max_fit_features": int(max_features) if max_features is not None else None,
        }
    std = np.std(matrix, axis=0)
    finite = np.isfinite(std)
    metadata_mask = np.ones(original_count, dtype=bool)
    if normalized_mode in {"change_probability", "change_probability_only", "change_only"}:
        metadata_mask = np.asarray(
            [
                str(row.get("feature_type", "")) == "change_probability_delta"
                for row in feature_rows
            ],
            dtype=bool,
        )
    eligible = finite & metadata_mask & (std >= float(min_std))
    if not np.any(eligible):
        eligible = finite & metadata_mask
    if not np.any(eligible):
        eligible = finite
    indices = np.flatnonzero(eligible)
    if indices.size == 0:
        indices = np.arange(original_count, dtype=np.int64)
    if max_features is not None and int(max_features) > 0 and indices.size > int(max_features):
        order = np.argsort(std[indices])[::-1]
        indices = indices[order[: int(max_features)]]
    indices = indices[np.argsort(indices)]
    if indices.size < int(min_features) and original_count >= int(min_features):
        order = np.argsort(std)[::-1][: int(min_features)]
        indices = np.unique(np.concatenate([indices, order])).astype(np.int64)
        if max_features is not None and int(max_features) > 0 and indices.size > int(max_features):
            order = np.argsort(std[indices])[::-1]
            indices = indices[order[: int(max_features)]]
        indices = indices[np.argsort(indices)]
    selected_feature_rows = [{**feature_rows[int(index)], "selected_feature_index": int(out_index)} for out_index, index in enumerate(indices)]
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
    selected_kernel["response_summary"] = _matrix_summary(selected_matrix)
    return selected_kernel, selected_matrix, selected_feature_rows, {
        "mode": normalized_mode,
        "original_feature_count": original_count,
        "selected_feature_count": int(indices.size),
        "metadata_filter": "feature_type=change_probability_delta"
        if normalized_mode in {"change_probability", "change_probability_only", "change_only"}
        else None,
        "min_feature_std": float(min_std),
        "max_fit_features": int(max_features) if max_features is not None else None,
        "selected_feature_indices_sample": [int(value) for value in indices[:128]],
        "selected_std_min": float(np.min(std[indices])) if indices.size else None,
        "selected_std_median": float(np.median(std[indices])) if indices.size else None,
        "selected_std_max": float(np.max(std[indices])) if indices.size else None,
        "claim_boundary": (
            "fit-layer feature selection only; raw modular-response kernel is unchanged. Controls are "
            "filtered with the same selected columns."
        ),
    }


def _select_columns(value: Any, indices: np.ndarray, selected_matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if indices.size == 0:
        return np.zeros_like(selected_matrix)
    if array.ndim != 2 or array.shape[0] != selected_matrix.shape[0] or array.shape[1] <= int(np.max(indices)):
        return np.zeros_like(selected_matrix)
    return array[:, indices]


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
) -> dict[str, Any]:
    channel_keys = _channel_keys(feature_rows)
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
        candidate_count=max(128, min(int(candidate_count), 1024)),
        candidate_radius=candidate_radius,
        softness=softness,
        seed=seed + 101,
        anchor_weight=anchor_weight,
        max_iterations=max_iterations,
        refine_steps=refine_steps,
        refine_max_rows=refine_max_rows,
        refine_max_nfev=refine_max_nfev,
        candidate_mode=candidate_mode,
    )
    if np.any(np.linalg.norm(observer_axes, axis=1) > 1e-9):
        control_reports["shuffled_observer_labels"] = _joint_global_h3_fit(
            np.asarray(kernel.get("shuffled_observer_labels_control", np.zeros_like(matrix)), dtype=float),
            expanded_caps,
            feature_rows,
            train_mask=train_mask,
            observer_axes=observer_axes,
            candidate_count=max(128, min(int(candidate_count), 1024)),
            candidate_radius=candidate_radius,
            softness=softness,
            seed=seed + 102,
            anchor_weight=anchor_weight,
            max_iterations=max_iterations,
            refine_steps=refine_steps,
            refine_max_rows=refine_max_rows,
            refine_max_nfev=refine_max_nfev,
            candidate_mode=candidate_mode,
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
    response_summary = dict(kernel.get("response_summary", {}))
    response_degenerate = bool(float(response_summary.get("mean_row_std", 0.0)) < 1e-3)
    eligible = bool(matrix.shape[0] >= min_observers and matrix.shape[1] >= min_features)
    receipt = bool(
        eligible
        and not response_degenerate
        and h3_beats_s2
        and all(h3_beats_controls.values())
        and (not h3_beats_wrong or all(h3_beats_wrong.values()))
    )
    report = {
        "mode": "modular_response_kernel_to_h3_fit",
        "fit_mode": "joint_global",
        "model": "tanh_halfspace_shared_channel_affine",
        "objective": "heldout_normalized_rmse_and_gaussian_negloglik",
        "MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT": receipt,
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
        "wrong_scale_feature_audit": wrong_scale_feature_audit,
        "h3_beats_controls": h3_beats_controls,
        "h3_beats_wrong_scale_controls": h3_beats_wrong,
        "response_summary": response_summary,
        "response_degenerate": response_degenerate,
        "eligibility_gate_passed": eligible,
        "pass_ratio": float(pass_ratio),
        "claim_boundary": (
            "joint finite H3 fit of the support-visible modular response kernel. Observer rows share "
            "one H3 candidate assignment each, while each time/observable channel gets shared affine "
            "nuisance parameters. Receipt requires held-out H3 score to beat S2, shuffled, no-perturbation, "
            "and wrong-scale controls. This is not a physical bulk claim without refinement and planted controls."
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
) -> dict[str, Any]:
    response = np.asarray(response, dtype=float)
    normals = cap_normals(caps)
    candidates = _candidate_h3_points(
        int(candidate_count),
        seed=int(seed),
        radius=float(candidate_radius),
        mode=str(candidate_mode),
    )
    candidate_profiles = _tanh_halfspace_profile(candidates, normals, softness=float(softness))
    channel_keys = _channel_keys(feature_rows)
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
                fitted,
                train_mask,
                offsets,
                amplitudes,
                observer_axes,
                candidate_radius=float(candidate_radius),
                softness=float(softness),
                anchor_weight=float(anchor_weight),
                max_rows=refine_max_rows,
                max_nfev=int(refine_max_nfev),
            )
            total_attempted += int(step_report.get("attempted_rows", 0))
            total_improvements.extend(float(value) for value in step_report.get("improvements", []))
            selected_base = _tanh_halfspace_profile(fitted, normals, softness=float(softness))
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
        "channel_count": int(len(set(channel_keys))),
        "channel_summary_sample": channel_summary[:64],
        "assignment_unique_count": int(np.unique(assignments).size),
        "refinement": refinement_report,
        "best_candidate_indices": [int(value) for value in assignments[:256]],
        "sample_fitted_h3_points": [[float(x) for x in row] for row in fitted[:32]],
        "fitted_h3_points": [[float(x) for x in row] for row in fitted],
        "feature_rmse": [float(value) for value in feature_rmse],
        "feature_rmse_sample": [float(value) for value in feature_rmse[:128]],
        **metrics,
    }


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
    group_rows: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    audited_indices = np.flatnonzero(audited)
    for feature_index in audited_indices:
        values = {"2pi_h3_fit": float(h3_rmse[feature_index])}
        values.update({label: float(residuals[feature_index]) for label, residuals in control_rmse.items()})
        winner = min(values, key=values.get)
        winner_counts[winner] = winner_counts.get(winner, 0) + 1
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
                "_h3": [],
                "_best_wrong": [],
            },
        )
        group["feature_count"] += 1
        group["winner_counts"][winner] = group["winner_counts"].get(winner, 0) + 1
        best_wrong = min(float(residuals[feature_index]) for residuals in control_rmse.values())
        group["_h3"].append(float(h3_rmse[feature_index]))
        group["_best_wrong"].append(float(best_wrong))
    wrong_win_count = sum(count for label, count in winner_counts.items() if label != "2pi_h3_fit")
    group_output: list[dict[str, Any]] = []
    for group in group_rows.values():
        count = max(1, int(group["feature_count"]))
        wrong_count = sum(value for label, value in group["winner_counts"].items() if label != "2pi_h3_fit")
        group_output.append(
            {
                "cap_index": int(group["cap_index"]),
                "time_index": int(group["time_index"]),
                "time": float(group["time"]),
                "observable": str(group["observable"]),
                "feature_type": str(group["feature_type"]),
                "feature_count": int(group["feature_count"]),
                "winner_counts": {str(label): int(value) for label, value in group["winner_counts"].items()},
                "wrong_scale_win_fraction": float(wrong_count / count),
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
    return {
        "mode": "strict_wrong_scale_feature_audit",
        "eligible": True,
        "audited_feature_set": "heldout_features" if train.size >= feature_count and np.any(~train[:feature_count]) else "all_features",
        "feature_count": int(feature_count),
        "audited_feature_count": audited_count,
        "winner_counts": {str(label): int(value) for label, value in winner_counts.items()},
        "wrong_scale_win_count": int(wrong_win_count),
        "wrong_scale_win_fraction": float(wrong_win_count / max(audited_count, 1)),
        "two_pi_h3_fit_win_fraction": float(winner_counts.get("2pi_h3_fit", 0) / max(audited_count, 1)),
        "red_flag_wrong_scale_wins": bool(wrong_win_count > 0),
        "worst_groups": group_output[: int(max_group_rows)],
        "claim_boundary": (
            "diagnostic only: per-feature heldout residual audit for H3 fit versus wrong-scale controls. "
            "It identifies cap/time/observable locations where wrong normalization is competitive; it does "
            "not establish or reject a physical bulk by itself."
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


def _channel_keys(feature_rows: list[dict[str, Any]]) -> list[str]:
    keys = []
    for row in feature_rows:
        observable = row.get("observable", row.get("field", "field"))
        feature_type = row.get("feature_type", "")
        target = row.get("target_class", "")
        keys.append(f"t{int(row.get('time_index', 0))}:{observable}:{feature_type}:{target}")
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
    predicted = offsets[None, :] + candidate_profiles * amplitudes[None, :]
    predicted_train = predicted[:, train_indices]
    assignments = np.zeros(response.shape[0], dtype=np.int64)
    anchor_cost = _anchor_cost(candidates, observer_axes)
    for row_index in range(response.shape[0]):
        diff = predicted_train - response[row_index, train_indices][None, :]
        costs = np.mean(diff * diff, axis=1)
        if anchor_cost.size:
            costs = costs + float(anchor_weight) * anchor_cost[row_index]
        assignments[row_index] = int(np.argmin(costs))
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
    fitted: np.ndarray,
    train_mask: np.ndarray,
    offsets: np.ndarray,
    amplitudes: np.ndarray,
    observer_axes: np.ndarray,
    *,
    candidate_radius: float,
    softness: float,
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
    predicted = offsets[None, :] + _tanh_halfspace_profile(fitted, normals, softness=softness) * amplitudes[None, :]
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

        def objective(vector: np.ndarray) -> np.ndarray:
            point = h3_point_from_tangent(vector)
            signed = -point[0] * target_normals[:, 0] + target_normals[:, 1:] @ point[1:]
            profile = np.tanh(np.clip(signed / width, -60.0, 60.0))
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
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        offsets[all_cols] = float(coef[0])
        amplitudes[all_cols] = float(coef[1])
        summary.append(
            {
                "channel": str(key),
                "feature_count": int(all_cols.size),
                "train_feature_count": int(cols.size),
                "offset": float(coef[0]),
                "amplitude": float(coef[1]),
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
