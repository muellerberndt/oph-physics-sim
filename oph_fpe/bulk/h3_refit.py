from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.bulk.cap_geometry import RoundCap
from oph_fpe.bulk.h3_response_fit import modular_response_h3_report


def write_modular_response_kernel_cache(
    run_dir: Path,
    kernel: dict[str, Any],
    caps: list[RoundCap],
    *,
    prefix: str = "modular_response_kernel",
) -> dict[str, Any]:
    """Write the full modular-response tensor needed for cheap H3 refits."""

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(kernel.get("matrix", np.zeros((0, 0))), dtype=np.float64)
    arrays: dict[str, np.ndarray] = {
        "matrix": matrix,
        "s2_boundary_control": _array_like(kernel.get("s2_boundary_control"), matrix),
        "shuffled_control": _array_like(kernel.get("shuffled_control"), matrix),
        "shuffled_response_control": _array_like(
            kernel.get("shuffled_response_control", kernel.get("shuffled_control")),
            matrix,
        ),
        "shuffled_observer_labels_control": _array_like(kernel.get("shuffled_observer_labels_control"), matrix),
        "no_modular_flow_control": _array_like(kernel.get("no_modular_flow_control"), matrix),
    }
    wrong_scale_keys: list[dict[str, str]] = []
    for index, (label, value) in enumerate(sorted((kernel.get("wrong_scale_controls", {}) or {}).items())):
        key = f"wrong_scale_control_{index}"
        arrays[key] = _array_like(value, matrix)
        wrong_scale_keys.append({"label": str(label), "array_key": key})

    payload_path = run_dir / f"{prefix}_payload.npz"
    np.savez_compressed(payload_path, **arrays)
    metadata = {
        "mode": "modular_response_kernel_cache",
        "payload": payload_path.name,
        "observable_mode": kernel.get("observable_mode"),
        "observer_ids": [int(value) for value in kernel.get("observer_ids", range(matrix.shape[0]))],
        "feature_rows": list(kernel.get("feature_rows", [])),
        "caps": [_cap_row(cap) for cap in caps],
        "cap_count": int(kernel.get("cap_count", len(caps))),
        "time_count": int(kernel.get("time_count", 0)),
        "field_names": list(kernel.get("field_names", [])),
        "wrong_scale_controls": wrong_scale_keys,
        "response_summary": _summary(matrix),
        "raw_response_summary": kernel.get("raw_response_summary", {}),
        "transform_report": kernel.get("transform_report", {}),
        "claim_boundary": (
            "Cached support-visible modular-response matrix for post-run H3 refits. "
            "This cache does not add evidence; refit reports must still pass the same controls."
        ),
    }
    metadata_path = run_dir / f"{prefix}_cache.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return {
        "mode": "modular_response_kernel_cache_written",
        "payload_path": str(payload_path),
        "metadata_path": str(metadata_path),
        "observer_count": int(matrix.shape[0]),
        "feature_count": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "wrong_scale_control_count": len(wrong_scale_keys),
        "claim_boundary": metadata["claim_boundary"],
    }


def load_modular_response_kernel_cache(
    run_dir: Path,
    *,
    prefix: str = "modular_response_kernel",
) -> tuple[dict[str, Any], list[RoundCap], dict[str, Any]]:
    run_dir = Path(run_dir)
    metadata_path = run_dir / f"{prefix}_cache.json"
    payload_path = run_dir / f"{prefix}_payload.npz"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload_name = str(metadata.get("payload", payload_path.name))
    payload_path = run_dir / payload_name
    with np.load(payload_path, allow_pickle=False) as payload:
        matrix = np.asarray(payload["matrix"], dtype=np.float64)
        kernel = {
            "mode": "observer_modular_response_kernel",
            "observable_mode": metadata.get("observable_mode"),
            "matrix": matrix,
            "s2_boundary_control": np.asarray(payload["s2_boundary_control"], dtype=np.float64),
            "shuffled_control": np.asarray(payload["shuffled_control"], dtype=np.float64),
            "shuffled_response_control": np.asarray(payload["shuffled_response_control"], dtype=np.float64),
            "shuffled_observer_labels_control": np.asarray(
                payload["shuffled_observer_labels_control"],
                dtype=np.float64,
            ),
            "no_modular_flow_control": np.asarray(payload["no_modular_flow_control"], dtype=np.float64),
            "wrong_scale_controls": {
                str(row["label"]): np.asarray(payload[str(row["array_key"])], dtype=np.float64)
                for row in metadata.get("wrong_scale_controls", [])
            },
            "feature_rows": list(metadata.get("feature_rows", [])),
            "observer_ids": [int(value) for value in metadata.get("observer_ids", range(matrix.shape[0]))],
            "cap_count": int(metadata.get("cap_count", 0)),
            "time_count": int(metadata.get("time_count", 0)),
            "field_names": list(metadata.get("field_names", [])),
            "response_summary": _summary(matrix),
            "raw_response_summary": metadata.get("raw_response_summary", {}),
            "transform_report": metadata.get("transform_report", {}),
            "claim_boundary": metadata.get("claim_boundary"),
        }
    caps = [_cap_from_row(row) for row in metadata.get("caps", [])]
    return kernel, caps, metadata


def write_h3_refit_report(
    run_dir: Path,
    out: Path | None = None,
    *,
    candidate_count: int = 4096,
    candidate_radius: float = 2.0,
    softness: float = 0.25,
    seed: int = 1,
    pass_ratio: float = 1.0,
    min_observers: int = 8,
    min_features: int = 12,
    fit_mode: str = "joint_global",
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
) -> dict[str, Any]:
    kernel, caps, metadata = load_modular_response_kernel_cache(run_dir)
    report = modular_response_h3_report(
        kernel,
        caps,
        candidate_count=int(candidate_count),
        candidate_radius=float(candidate_radius),
        softness=float(softness),
        seed=int(seed),
        pass_ratio=float(pass_ratio),
        min_observers=int(min_observers),
        min_features=int(min_features),
        fit_mode=str(fit_mode),
        heldout_fraction=float(heldout_fraction),
        anchor_weight=float(anchor_weight),
        max_iterations=int(max_iterations),
        feature_selection=str(feature_selection),
        max_fit_features=int(max_fit_features) if max_fit_features is not None else None,
        min_feature_std=float(min_feature_std),
        min_wrong_scale_feature_delta=float(min_wrong_scale_feature_delta),
        exclude_observables=tuple(str(value) for value in exclude_observables),
        exclude_feature_types=tuple(str(value) for value in exclude_feature_types),
        max_features_per_cap_time_observable=(
            int(max_features_per_cap_time_observable)
            if max_features_per_cap_time_observable is not None
            else None
        ),
        refine_steps=int(refine_steps),
        refine_max_rows=int(refine_max_rows) if refine_max_rows is not None else None,
        refine_max_nfev=int(refine_max_nfev),
        candidate_mode=str(candidate_mode),
        channel_mode=str(channel_mode),
        profile_mode=str(profile_mode),
        profile_time_scale=float(profile_time_scale),
        control_fit_mode=str(control_fit_mode),
    )
    report["kernel_cache"] = {
        "mode": metadata.get("mode"),
        "payload": metadata.get("payload"),
        "observer_count": int(np.asarray(kernel.get("matrix")).shape[0]),
        "feature_count": int(np.asarray(kernel.get("matrix")).shape[1]),
        "claim_boundary": metadata.get("claim_boundary"),
    }
    out_path = Path(out) if out is not None else Path(run_dir) / "modular_response_h3_refit_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def write_h3_refit_ensemble_report(
    run_dir: Path,
    out: Path | None = None,
    *,
    seeds: list[int] | tuple[int, ...],
    required_receipt_fraction: float = 0.75,
    required_dim3_fraction: float = 0.5,
    required_median_ev: float = 0.08,
    required_p75_material_wrong_fraction: float = 0.075,
    **kwargs: Any,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    seed_values = [int(value) for value in seeds]
    if not seed_values:
        seed_values = [1]
    report_dir = run_dir / "h3_refit_seed_ensemble"
    report_dir.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    for seed in seed_values:
        report = write_h3_refit_report(run_dir, report_dir / f"seed_{seed}.json", seed=seed, **kwargs)
        h3 = report.get("h3_fit", {})
        gates = report.get("h3_response_stage_gates", {})
        dimension = report.get("h3_chart_dimension_debug", {})
        rows.append(
            {
                "seed": int(seed),
                "receipt": bool(report.get("MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT", False)),
                "control_separation_receipt": bool(report.get("H3_RESPONSE_CONTROL_SEPARATION_RECEIPT", False)),
                "candidate_receipt": bool(report.get("H3_RESPONSE_CANDIDATE_RECEIPT", False)),
                "heldout_normalized_rmse": _maybe_float(h3.get("heldout_normalized_rmse")),
                "heldout_explained_variance": _maybe_float(h3.get("heldout_explained_variance")),
                "material_wrong_scale_win_fraction": _maybe_float(gates.get("material_wrong_scale_win_fraction")),
                "signal_gate": bool(gates.get("signal_gate", False)),
                "geometry_gate": bool(gates.get("geometry_gate", False)),
                "aggregate_wrong_scale_gate": bool(gates.get("aggregate_wrong_scale_gate", False)),
                "material_feature_gate": bool(gates.get("material_feature_gate", False)),
                "assignment_unique_count": int(h3.get("assignment_unique_count", 0)),
                "candidate_3d_dimension_window": bool(dimension.get("candidate_3d_dimension_window", False)),
                "dimension_estimators_agree": bool(dimension.get("dimension_estimators_agree", False)),
                "correlation_dimension_estimate": _maybe_float(
                    (dimension.get("correlation_dimension") or {}).get("estimate")
                ),
                "local_mle_dimension_estimate": _maybe_float((dimension.get("local_mle_dimension") or {}).get("estimate")),
                "h3_beats_controls": report.get("h3_beats_controls", {}),
                "h3_beats_wrong_scale_controls": report.get("h3_beats_wrong_scale_controls", {}),
                "report_path": str(report_dir / f"seed_{seed}.json"),
            }
        )
    report = _h3_ensemble_summary_from_rows(
        rows,
        run_dir=str(run_dir),
        seed_values=seed_values,
        required_receipt_fraction=float(required_receipt_fraction),
        required_dim3_fraction=float(required_dim3_fraction),
        required_median_ev=float(required_median_ev),
        required_p75_material_wrong_fraction=float(required_p75_material_wrong_fraction),
    )
    report.update(
        {
            "physical_claim": False,
            "claim_boundary": (
                "Cached H3 candidate-seed ensemble. This checks robustness of the H3 response fit against "
                "random candidate sampling and applies the material wrong-scale audit across seeds. It is still "
                "an internal diagnostic, not a neutral 3D bulk claim. The ensemble receipt is gated by aggregate "
                "robustness metrics; single-seed candidate receipts are reported as diagnostics because they use "
                "a stricter per-seed material wrong-scale threshold."
            ),
        }
    )
    out_path = Path(out) if out is not None else run_dir / "h3_refit_seed_ensemble_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _h3_ensemble_summary_from_rows(
    rows: list[dict[str, Any]],
    *,
    run_dir: str | None = None,
    seed_values: list[int] | tuple[int, ...] | None = None,
    required_receipt_fraction: float = 0.75,
    required_dim3_fraction: float = 0.5,
    required_median_ev: float = 0.08,
    required_p75_material_wrong_fraction: float = 0.075,
) -> dict[str, Any]:
    """Summarize cached H3 candidate-seed refits.

    The ensemble receipt intentionally uses aggregate ensemble thresholds,
    not the stricter per-seed candidate receipt. Per-seed candidate receipts
    are retained as diagnostic context.
    """

    rows = [dict(row) for row in rows]
    seed_values = [int(value) for value in seed_values] if seed_values is not None else [
        int(row.get("seed", index)) for index, row in enumerate(rows)
    ]
    receipt_fraction = float(sum(bool(row.get("receipt")) for row in rows) / max(1, len(rows)))
    candidate_receipt_fraction = float(sum(bool(row.get("candidate_receipt")) for row in rows) / max(1, len(rows)))
    control_separation_fraction = float(
        sum(bool(row.get("control_separation_receipt")) for row in rows) / max(1, len(rows))
    )
    dim3_fraction = float(sum(bool(row.get("candidate_3d_dimension_window")) for row in rows) / max(1, len(rows)))
    nrmse_values = [row.get("heldout_normalized_rmse") for row in rows if row.get("heldout_normalized_rmse") is not None]
    ev_values = [row.get("heldout_explained_variance") for row in rows if row.get("heldout_explained_variance") is not None]
    material_wrong_values = [
        row.get("material_wrong_scale_win_fraction")
        for row in rows
        if row.get("material_wrong_scale_win_fraction") is not None
    ]
    unique_values = [int(row.get("assignment_unique_count", 0)) for row in rows]
    median_ev = float(np.median(ev_values)) if ev_values else None
    p75_material_wrong = float(np.percentile(material_wrong_values, 75)) if material_wrong_values else None
    response_ensemble_receipt = bool(
        control_separation_fraction >= float(required_receipt_fraction)
        and median_ev is not None
        and median_ev > float(required_median_ev)
        and p75_material_wrong is not None
        and p75_material_wrong < float(required_p75_material_wrong_fraction)
    )
    return {
        "mode": "h3_refit_seed_ensemble",
        "run_dir": str(run_dir) if run_dir is not None else None,
        "seed_count": int(len(rows)),
        "seeds": list(seed_values),
        "receipt_count": int(sum(bool(row.get("receipt")) for row in rows)),
        "receipt_fraction": receipt_fraction,
        "control_separation_receipt_count": int(sum(bool(row.get("control_separation_receipt")) for row in rows)),
        "control_separation_receipt_fraction": control_separation_fraction,
        "candidate_receipt_count": int(sum(bool(row.get("candidate_receipt")) for row in rows)),
        "candidate_receipt_fraction": candidate_receipt_fraction,
        "candidate_receipt_fraction_diagnostic_only": candidate_receipt_fraction,
        "candidate_3d_window_count": int(sum(bool(row.get("candidate_3d_dimension_window")) for row in rows)),
        "candidate_3d_window_fraction": dim3_fraction,
        "mean_heldout_normalized_rmse": float(np.mean(nrmse_values)) if nrmse_values else None,
        "mean_heldout_explained_variance": float(np.mean(ev_values)) if ev_values else None,
        "median_heldout_explained_variance": median_ev,
        "p75_material_wrong_scale_win_fraction": p75_material_wrong,
        "mean_assignment_unique_count": float(np.mean(unique_values)) if unique_values else None,
        "required_receipt_fraction": float(required_receipt_fraction),
        "required_dim3_fraction": float(required_dim3_fraction),
        "required_median_ev": float(required_median_ev),
        "required_p75_material_wrong_fraction": float(required_p75_material_wrong_fraction),
        "ensemble_gate_uses": [
            "control_separation_receipt_fraction",
            "median_heldout_explained_variance",
            "p75_material_wrong_scale_win_fraction",
        ],
        "H3_RESPONSE_ENSEMBLE_RECEIPT": response_ensemble_receipt,
        "h3_response_ensemble_receipt": response_ensemble_receipt,
        "h3_response_seed_robust_receipt": response_ensemble_receipt,
        "h3_chart_3d_seed_robust_receipt": bool(dim3_fraction >= float(required_dim3_fraction)),
        "rows": rows,
    }


def _array_like(value: Any, matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != matrix.shape:
        return np.zeros_like(matrix, dtype=np.float64)
    return array


def _cap_row(cap: RoundCap) -> dict[str, Any]:
    cap = cap.normalized()
    return {
        "axis": [float(value) for value in cap.axis],
        "theta0": float(cap.theta0),
        "tangent": [float(value) for value in cap.tangent],
        "collar_width": float(cap.collar_width),
    }


def _cap_from_row(row: dict[str, Any]) -> RoundCap:
    return RoundCap(
        axis=np.asarray(row["axis"], dtype=float),
        theta0=float(row["theta0"]),
        tangent=np.asarray(row["tangent"], dtype=float),
        collar_width=float(row.get("collar_width", 0.03)),
    ).normalized()


def _summary(matrix: np.ndarray) -> dict[str, float]:
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


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None
