from __future__ import annotations

from typing import Any

import numpy as np

from oph_fpe.bulk.cap_normals import cap_normals
from oph_fpe.bulk.cap_geometry import RoundCap
from oph_fpe.bulk.h3_chart import h3_halfspace_profile, random_h3_points
from oph_fpe.bulk.record_to_h3 import fit_response_profiles_to_h3
from oph_fpe.claims import DEMO, with_claim_metadata


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
) -> dict[str, Any]:
    matrix = np.asarray(kernel.get("matrix", np.zeros((0, 0))), dtype=float)
    feature_rows = list(kernel.get("feature_rows", []))
    if matrix.size == 0 or not feature_rows or not caps:
        return with_claim_metadata(
            {
                "mode": "modular_response_kernel_to_h3_fit",
                "MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT": False,
                "h3_bulk_candidate_receipt": False,
                "reason": "empty_kernel_or_caps",
                "claim_boundary": "empty modular response H3 fit; no bulk claim",
            },
            claim_level=DEMO,
            receipt="MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT",
            physical_claim=False,
        )
    expanded_caps = [caps[int(row["cap_index"])] for row in feature_rows]
    if str(fit_mode) == "joint_global":
        return _modular_response_joint_h3_report(
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
        )
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
        "h3_bulk_candidate_receipt": receipt,
        "observer_ids": [int(value) for value in kernel.get("observer_ids", range(matrix.shape[0]))],
        "observer_count": int(matrix.shape[0]),
        "feature_count": int(matrix.shape[1]),
        "cap_count": int(kernel.get("cap_count", len(caps))),
        "time_count": int(kernel.get("time_count", 0)),
        "field_names": list(kernel.get("field_names", [])),
        "feature_rows_sample": feature_rows[:128],
        "h3_fit": h3_fit,
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
    return with_claim_metadata(report, claim_level=DEMO, receipt="MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT", physical_claim=False)


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
        "s2_boundary_control": {
            **s2_report,
            "h3_beats_s2_boundary": h3_beats_s2,
        },
        "control_fits": control_reports,
        "wrong_scale_control_fits": wrong_scale_reports,
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
    return with_claim_metadata(report, claim_level=DEMO, receipt="MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT", physical_claim=False)


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
) -> dict[str, Any]:
    response = np.asarray(response, dtype=float)
    normals = cap_normals(caps)
    candidates = random_h3_points(int(candidate_count), seed=int(seed), radius=float(candidate_radius))
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
    prediction = offsets[None, :] + selected_base * amplitudes[None, :]
    metrics = _heldout_metrics(response, prediction, train_mask)
    fitted = candidates[assignments]
    return {
        "mode": "joint_global_h3_tanh_halfspace_fit",
        "candidate_count": int(candidate_count),
        "candidate_radius": float(candidate_radius),
        "softness": float(softness),
        "anchor_weight": float(anchor_weight),
        "channel_count": int(len(set(channel_keys))),
        "channel_summary_sample": channel_summary[:64],
        "assignment_unique_count": int(np.unique(assignments).size),
        "best_candidate_indices": [int(value) for value in assignments[:256]],
        "sample_fitted_h3_points": [[float(x) for x in row] for row in fitted[:32]],
        "fitted_h3_points": [[float(x) for x in row] for row in fitted],
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
    return {
        "mode": label,
        "channel_count": int(len(set(channel_keys))),
        "channel_summary_sample": channel_summary[:64],
        **_heldout_metrics(response, prediction, train_mask),
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
