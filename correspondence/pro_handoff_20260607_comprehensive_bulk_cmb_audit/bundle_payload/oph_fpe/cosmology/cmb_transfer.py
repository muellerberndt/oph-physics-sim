from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.cosmology.cmb_compare import load_planck_tt_binned


DEFAULT_TRANSFER_FIELDS = [
    "record_signature",
    "stable_count",
    "cumulative_repair_load",
    "repair_load",
    "s3_class_density",
    "record_signature_smooth_k16",
    "record_signature_smooth_k32",
    "record_signature_smooth_k64",
    "record_signature_band_k16_k64",
    "record_repair_mix",
    "record_sector_mix",
]


def collect_transfer_runs(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in run_dirs:
        for cl_path in sorted(Path(root).glob("**/cl_comparison_report.json")):
            run_path = cl_path.parent.resolve()
            if run_path in seen:
                continue
            seen.add(run_path)
            cl_report = json.loads(cl_path.read_text(encoding="utf-8"))
            manifest = _read_json(run_path / "manifest.json")
            gate = _read_json(run_path / "cosmology_gate_report.json")
            rows.append(
                {
                    "run_id": manifest.get("run_id", run_path.name),
                    "path": str(run_path),
                    "patch_count": int(manifest.get("patch_count", 0)),
                    "gate_allowed": bool(gate.get("allowed", False)),
                    "fields": cl_report.get("fields", {}),
                }
            )
    return rows


def cmb_transfer_report(
    rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, float]],
    *,
    train_patch_count: int | None = None,
    test_patch_count: int | None = None,
    field_names: list[str] | None = None,
    ridge: float = 1.0e-3,
    sample_count: int = 128,
    control_seed: int = 9137,
    bootstrap_count: int = 0,
    bootstrap_seed: int = 271828,
) -> dict[str, Any]:
    allowed = [row for row in rows if row.get("gate_allowed")]
    fields = _common_fields(allowed, field_names or DEFAULT_TRANSFER_FIELDS)
    if not allowed or not fields:
        return _empty_report(rows, fields, reason="no_gate_allowed_runs_or_common_fields")
    sizes = sorted({int(row.get("patch_count", 0)) for row in allowed})
    train_n = int(train_patch_count if train_patch_count is not None else sizes[0])
    test_n = int(test_patch_count if test_patch_count is not None else (sizes[-1] if len(sizes) > 1 else sizes[0]))
    train_rows = [row for row in allowed if int(row.get("patch_count", 0)) == train_n]
    test_rows = [row for row in allowed if int(row.get("patch_count", 0)) == test_n]
    if not train_rows or not test_rows:
        return _empty_report(rows, fields, reason="missing_train_or_test_patch_count")

    target = _benchmark_curve(benchmark_rows, sample_count=sample_count)
    train_matrices = [_run_field_matrix(row, fields, sample_count=sample_count) for row in train_rows]
    train_matrices = [matrix for matrix in train_matrices if matrix.size]
    if not train_matrices:
        return _empty_report(rows, fields, reason="empty_train_matrices")
    design = np.vstack(train_matrices)
    target_stack = np.tile(target, len(train_matrices))
    weights = _ridge_fit(design, target_stack, ridge=float(ridge))

    train_eval = _evaluate_rows(train_rows, fields, weights, target, sample_count=sample_count)
    test_eval = _evaluate_rows(test_rows, fields, weights, target, sample_count=sample_count)
    all_eval = _evaluate_rows(allowed, fields, weights, target, sample_count=sample_count)
    controls = _transfer_controls(
        train_rows,
        test_rows,
        fields,
        true_target=target,
        ridge=float(ridge),
        sample_count=int(sample_count),
        seed=int(control_seed),
    )
    test_corr = _finite_float(test_eval.get("mean_shape_correlation"))
    control_corrs = [
        _finite_float(control.get("test", {}).get("mean_shape_correlation"))
        for control in controls.values()
    ]
    control_corrs = [value for value in control_corrs if value is not None]
    max_control_corr = max(control_corrs) if control_corrs else None
    diagnostic_receipt = bool(
        test_corr is not None
        and test_corr > 0.5
        and (max_control_corr is None or test_corr > max_control_corr + 0.05)
    )
    control_gap = float(test_corr - max_control_corr) if test_corr is not None and max_control_corr is not None else None
    bootstrap = _bootstrap_transfer_uncertainty(
        test_rows,
        fields,
        weights,
        target,
        sample_count=int(sample_count),
        bootstrap_count=int(bootstrap_count),
        seed=int(bootstrap_seed),
    )
    return {
        "mode": "cmb_screen_basis_transfer_diagnostic",
        "run_count": len(rows),
        "gate_allowed_count": len(allowed),
        "available_patch_counts": sizes,
        "train_patch_count": train_n,
        "test_patch_count": test_n,
        "field_names": fields,
        "ridge": float(ridge),
        "sample_count": int(sample_count),
        "weights": {field: float(value) for field, value in zip(fields, weights, strict=True)},
        "dominant_weights": _dominant_weights(fields, weights),
        "train": train_eval,
        "test": test_eval,
        "all_gate_allowed": all_eval,
        "controls": controls,
        "max_control_test_shape_correlation": max_control_corr,
        "test_vs_max_control_shape_correlation_gap": control_gap,
        "bootstrap": bootstrap,
        "diagnostic_transfer_receipt": diagnostic_receipt,
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "Cross-scale linear screen-basis transfer fitted to Planck TT shape for diagnostics only. "
            "This uses observed data to choose weights, normalizes multipole axes, and rescales amplitude "
            "when scoring, so it is not a prediction, not a likelihood, not CAMB/CLASS input, and not an "
            "early-universe claim."
        ),
    }


def write_cmb_transfer_report(
    run_dirs: list[Path],
    benchmark_path: Path,
    out_dir: Path,
    *,
    train_patch_count: int | None = None,
    test_patch_count: int | None = None,
    field_names: list[str] | None = None,
    ridge: float = 1.0e-3,
    sample_count: int = 128,
    control_seed: int = 9137,
    bootstrap_count: int = 0,
    bootstrap_seed: int = 271828,
) -> dict[str, Any]:
    rows = collect_transfer_runs(run_dirs)
    benchmark_rows = load_planck_tt_binned(Path(benchmark_path))
    report = cmb_transfer_report(
        rows,
        benchmark_rows,
        train_patch_count=train_patch_count,
        test_patch_count=test_patch_count,
        field_names=field_names,
        ridge=float(ridge),
        sample_count=int(sample_count),
        control_seed=int(control_seed),
        bootstrap_count=int(bootstrap_count),
        bootstrap_seed=int(bootstrap_seed),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cmb_transfer_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / "cmb_transfer_rows.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    (out_dir / "cmb_transfer_report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _common_fields(rows: list[dict[str, Any]], candidates: list[str]) -> list[str]:
    if not rows:
        return []
    names: list[str] = []
    for name in candidates:
        if all(name in row.get("fields", {}) and row["fields"][name].get("spectrum") for row in rows):
            names.append(str(name))
    return names


def _benchmark_curve(benchmark_rows: list[dict[str, float]], *, sample_count: int) -> np.ndarray:
    rows = [(float(row["ell"]), float(row.get("D_ell", 0.0))) for row in benchmark_rows if float(row.get("D_ell", 0.0)) > 0]
    return _standardized_curve(rows, sample_count=sample_count)


def _run_field_matrix(
    row: dict[str, Any],
    fields: list[str],
    *,
    sample_count: int,
    field_permutation: np.ndarray | None = None,
) -> np.ndarray:
    curves: list[np.ndarray] = []
    for field in fields:
        spectrum = row.get("fields", {}).get(field, {}).get("spectrum", [])
        pairs = [(float(item["ell"]), float(item.get("D_ell", 0.0))) for item in spectrum if float(item["ell"]) >= 2]
        if len(pairs) < 2:
            return np.zeros((0, 0), dtype=float)
        curve = _standardized_curve(pairs, sample_count=sample_count)
        if field_permutation is not None:
            curve = curve[np.asarray(field_permutation, dtype=np.int64)]
        curves.append(curve)
    return np.vstack(curves).T


def _standardized_curve(rows: list[tuple[float, float]], *, sample_count: int) -> np.ndarray:
    x = np.asarray([row[0] for row in rows], dtype=float)
    y = np.asarray([row[1] for row in rows], dtype=float)
    span = max(float(np.max(x) - np.min(x)), 1e-12)
    x = (x - float(np.min(x))) / span
    grid = np.linspace(0.0, 1.0, int(sample_count))
    curve = np.interp(grid, x, y)
    curve = curve - float(np.mean(curve))
    scale = float(np.std(curve))
    if scale < 1e-12:
        return np.zeros_like(curve)
    return curve / scale


def _ridge_fit(design: np.ndarray, target: np.ndarray, *, ridge: float) -> np.ndarray:
    x = np.asarray(design, dtype=float)
    y = np.asarray(target, dtype=float)
    penalty = float(ridge) * np.eye(x.shape[1], dtype=float)
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def _evaluate_rows(
    rows: list[dict[str, Any]],
    fields: list[str],
    weights: np.ndarray,
    target: np.ndarray,
    *,
    sample_count: int,
    field_permutation: np.ndarray | None = None,
) -> dict[str, Any]:
    per_run: list[dict[str, Any]] = []
    for row in rows:
        matrix = _run_field_matrix(row, fields, sample_count=sample_count, field_permutation=field_permutation)
        if not matrix.size:
            continue
        prediction = matrix @ weights
        per_run.append(
            {
                "run_id": row.get("run_id"),
                "patch_count": int(row.get("patch_count", 0)),
                **_shape_metrics(prediction, target),
            }
        )
    return {
        "run_count": len(per_run),
        "patch_count_counts": dict(Counter(str(item["patch_count"]) for item in per_run)),
        "mean_shape_correlation": _mean(item.get("shape_correlation") for item in per_run),
        "mean_normalized_rmse": _mean(item.get("normalized_rmse") for item in per_run),
        "mean_best_fit_amplitude": _mean(item.get("best_fit_amplitude") for item in per_run),
        "per_run": per_run,
    }


def _transfer_controls(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    fields: list[str],
    *,
    true_target: np.ndarray,
    ridge: float,
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    reversed_target = true_target[::-1].copy()
    shuffled_target = true_target[rng.permutation(true_target.size)].copy()
    controls = {
        "reversed_target": _one_transfer_control(
            train_rows,
            test_rows,
            fields,
            fit_target=reversed_target,
            score_target=true_target,
            ridge=float(ridge),
            sample_count=int(sample_count),
        ),
        "shuffled_target": _one_transfer_control(
            train_rows,
            test_rows,
            fields,
            fit_target=shuffled_target,
            score_target=true_target,
            ridge=float(ridge),
            sample_count=int(sample_count),
        ),
        "shuffled_field_curves": _one_transfer_control(
            train_rows,
            test_rows,
            fields,
            fit_target=true_target,
            score_target=true_target,
            ridge=float(ridge),
            sample_count=int(sample_count),
            field_permutation=np.random.default_rng(seed + 1).permutation(true_target.size),
        ),
    }
    return controls


def _bootstrap_transfer_uncertainty(
    rows: list[dict[str, Any]],
    fields: list[str],
    weights: np.ndarray,
    target: np.ndarray,
    *,
    sample_count: int,
    bootstrap_count: int,
    seed: int,
) -> dict[str, Any]:
    count = max(0, int(bootstrap_count))
    if count == 0:
        return {"enabled": False, "bootstrap_count": 0}
    predictions: list[np.ndarray] = []
    for row in rows:
        matrix = _run_field_matrix(row, fields, sample_count=sample_count)
        if matrix.size:
            predictions.append(matrix @ weights)
    if not predictions:
        return {"enabled": False, "bootstrap_count": 0, "reason": "empty_test_predictions"}
    target = np.asarray(target, dtype=float)
    rng = np.random.default_rng(seed)
    correlations: list[float] = []
    rmses: list[float] = []
    amplitudes: list[float] = []
    for _ in range(count):
        indices = rng.integers(0, target.size, size=target.size)
        per_corr: list[float] = []
        per_rmse: list[float] = []
        per_amp: list[float] = []
        for prediction in predictions:
            metrics = _shape_metrics(np.asarray(prediction, dtype=float)[indices], target[indices])
            if metrics["shape_correlation"] is not None:
                per_corr.append(float(metrics["shape_correlation"]))
            if metrics["normalized_rmse"] is not None:
                per_rmse.append(float(metrics["normalized_rmse"]))
            if metrics["best_fit_amplitude"] is not None:
                per_amp.append(float(metrics["best_fit_amplitude"]))
        if per_corr:
            correlations.append(float(np.mean(per_corr)))
        if per_rmse:
            rmses.append(float(np.mean(per_rmse)))
        if per_amp:
            amplitudes.append(float(np.mean(per_amp)))
    return {
        "enabled": True,
        "bootstrap_count": count,
        "seed": int(seed),
        "test_shape_correlation": _interval(correlations),
        "test_normalized_rmse": _interval(rmses),
        "test_best_fit_amplitude": _interval(amplitudes),
        "claim_boundary": (
            "bootstrap over the normalized multipole grid for the fixed fitted screen-basis diagnostic; "
            "not a cosmological likelihood or parameter posterior"
        ),
    }


def _one_transfer_control(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    fields: list[str],
    *,
    fit_target: np.ndarray,
    score_target: np.ndarray,
    ridge: float,
    sample_count: int,
    field_permutation: np.ndarray | None = None,
) -> dict[str, Any]:
    train_matrices = [
        _run_field_matrix(row, fields, sample_count=sample_count, field_permutation=field_permutation)
        for row in train_rows
    ]
    train_matrices = [matrix for matrix in train_matrices if matrix.size]
    if not train_matrices:
        return {"reason": "empty_train_matrices"}
    design = np.vstack(train_matrices)
    target_stack = np.tile(fit_target, len(train_matrices))
    weights = _ridge_fit(design, target_stack, ridge=float(ridge))
    return {
        "train": _evaluate_rows(
            train_rows,
            fields,
            weights,
            score_target,
            sample_count=sample_count,
            field_permutation=field_permutation,
        ),
        "test": _evaluate_rows(
            test_rows,
            fields,
            weights,
            score_target,
            sample_count=sample_count,
            field_permutation=field_permutation,
        ),
        "dominant_weights": _dominant_weights(fields, weights),
    }


def _shape_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    if prediction.size != target.size or prediction.size < 2:
        return {"shape_correlation": None, "normalized_rmse": None, "best_fit_amplitude": None}
    amplitude = float(np.dot(prediction, target) / max(float(np.dot(prediction, prediction)), 1e-12))
    residual = amplitude * prediction - target
    corr = float(np.corrcoef(prediction, target)[0, 1]) if np.std(prediction) > 1e-12 else 0.0
    return {
        "shape_correlation": corr,
        "normalized_rmse": float(np.sqrt(np.mean(residual * residual))),
        "best_fit_amplitude": amplitude,
    }


def _dominant_weights(fields: list[str], weights: np.ndarray) -> list[dict[str, float | str]]:
    rows = [
        {"field": field, "weight": float(weight), "abs_weight": float(abs(weight))}
        for field, weight in zip(fields, weights, strict=True)
    ]
    return sorted(rows, key=lambda row: -float(row["abs_weight"]))[:8]


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _interval(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": None, "p05": None, "p50": None, "p95": None}
    return {
        "mean": float(np.mean(array)),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def _empty_report(rows: list[dict[str, Any]], fields: list[str], *, reason: str) -> dict[str, Any]:
    return {
        "mode": "cmb_screen_basis_transfer_diagnostic",
        "run_count": len(rows),
        "gate_allowed_count": sum(1 for row in rows if row.get("gate_allowed")),
        "field_names": fields,
        "reason": reason,
        "physical_cmb_prediction": False,
        "claim_boundary": "empty cross-scale screen-basis transfer diagnostic; no CMB prediction",
    }


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# CMB Screen-Basis Transfer Diagnostic",
        "",
        str(report.get("claim_boundary", "")),
        "",
        f"- runs: {report.get('run_count')}",
        f"- gate-allowed runs: {report.get('gate_allowed_count')}",
        f"- train patch count: {report.get('train_patch_count')}",
        f"- test patch count: {report.get('test_patch_count')}",
        f"- fields: {', '.join(report.get('field_names', []))}",
        "",
        "## Train",
        f"- mean shape correlation: {_fmt(report.get('train', {}).get('mean_shape_correlation'))}",
        f"- mean normalized RMSE: {_fmt(report.get('train', {}).get('mean_normalized_rmse'))}",
        "",
        "## Test",
        f"- mean shape correlation: {_fmt(report.get('test', {}).get('mean_shape_correlation'))}",
        f"- mean normalized RMSE: {_fmt(report.get('test', {}).get('mean_normalized_rmse'))}",
        f"- max control test correlation: {_fmt(report.get('max_control_test_shape_correlation'))}",
        f"- test-control correlation gap: {_fmt(report.get('test_vs_max_control_shape_correlation_gap'))}",
        f"- diagnostic transfer receipt: {bool(report.get('diagnostic_transfer_receipt', False))}",
        "",
        "## Dominant Weights",
    ]
    for row in report.get("dominant_weights", []):
        lines.append(f"- {row['field']}: {row['weight']:.8g}")
    controls = report.get("controls", {})
    if controls:
        lines.extend(["", "## Controls"])
        for name, control in controls.items():
            test = control.get("test", {}) if isinstance(control, dict) else {}
            lines.append(
                f"- {name}: test correlation {_fmt(test.get('mean_shape_correlation'))}, "
                f"test RMSE {_fmt(test.get('mean_normalized_rmse'))}"
            )
    bootstrap = report.get("bootstrap", {})
    if bootstrap.get("enabled"):
        corr = bootstrap.get("test_shape_correlation", {})
        rmse = bootstrap.get("test_normalized_rmse", {})
        lines.extend(
            [
                "",
                "## Bootstrap",
                f"- count: {bootstrap.get('bootstrap_count')}",
                f"- test correlation mean [p05, p95]: {_fmt(corr.get('mean'))} [{_fmt(corr.get('p05'))}, {_fmt(corr.get('p95'))}]",
                f"- test RMSE mean [p05, p95]: {_fmt(rmse.get('mean'))} [{_fmt(rmse.get('p05'))}, {_fmt(rmse.get('p95'))}]",
            ]
        )
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mean(values) -> float | None:
    nums = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.mean(nums)) if nums else None


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.10g}"
    except Exception:
        return str(value)
