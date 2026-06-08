from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_planck_tt_binned(path: Path) -> list[dict[str, float]]:
    """Load Planck/PLA-style TT binned spectrum text files.

    Expected columns are ell, D_ell, lower error, upper error, and optional best-fit D_ell.
    The loader is intentionally generic enough for any whitespace-delimited TT benchmark
    with the same leading columns.
    """

    rows: list[dict[str, float]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        if len(parts) < 2:
            continue
        row = {
            "ell": float(parts[0]),
            "D_ell": float(parts[1]),
        }
        if len(parts) >= 4:
            row["minus_dD_ell"] = float(parts[2])
            row["plus_dD_ell"] = float(parts[3])
        if len(parts) >= 5:
            row["best_fit_D_ell"] = float(parts[4])
        rows.append(row)
    return rows


def cmb_lite_comparison_report(
    cl_report: dict[str, Any],
    benchmark_rows: list[dict[str, float]],
    *,
    benchmark_label: str = "Planck2018_TT_binned",
    source_url: str | None = None,
    field_names: list[str] | None = None,
) -> dict[str, Any]:
    fields = cl_report.get("fields", {})
    selected = field_names or list(fields.keys())
    comparisons: dict[str, Any] = {}
    for name in selected:
        if name not in fields or "spectrum" not in fields[name]:
            continue
        comparisons[name] = _compare_spectrum_shape(fields[name]["spectrum"], benchmark_rows)
    best_field = None
    best_positive_field = None
    best_normalized_axis_diagnostic_field = None
    if comparisons:
        usable = {key: value for key, value in comparisons.items() if value.get("usable")}
        if usable:
            best_normalized_axis_diagnostic_field = min(
                usable,
                key=lambda key: usable[key].get("unconstrained_normalized_rmse", float("inf")),
            )
        positive = {
            key: value
            for key, value in comparisons.items()
            if value.get("usable_positive_shape")
        }
        if positive:
            best_positive_field = min(
                positive,
                key=lambda key: positive[key].get("positive_amp_normalized_rmse", float("inf")),
            )
            best_field = best_positive_field
        else:
            best_field = best_normalized_axis_diagnostic_field
    return {
        "mode": "cmb_lite_shape_comparison",
        "benchmark": {
            "label": benchmark_label,
            "source_url": source_url,
            "row_count": len(benchmark_rows),
            "ell_min": float(min(row["ell"] for row in benchmark_rows)) if benchmark_rows else None,
            "ell_max": float(max(row["ell"] for row in benchmark_rows)) if benchmark_rows else None,
        },
        "simulator": {
            "ell_max": cl_report.get("ell_max"),
            "estimator": cl_report.get("estimator"),
            "point_count": cl_report.get("point_count"),
            "gate_allowed": bool(cl_report.get("gate_report", {}).get("allowed", False)),
        },
        "best_shape_field": best_field,
        "best_positive_shape_field": best_positive_field,
        "best_normalized_axis_diagnostic_field": best_normalized_axis_diagnostic_field,
        "field_comparisons": comparisons,
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "Shape-only diagnostics between a gated OPH freezeout-screen C_l proxy and an observed TT "
            "benchmark. The normalized-axis diagnostic rescales multipole axes to [0,1]; the positive "
            "amplitude diagnostic forbids anti-correlated spectra from winning; the real-ell physical "
            "comparison is marked unusable unless the simulator covers the benchmark ell range. This is "
            "not a Planck likelihood, not CAMB/CLASS input, and not a physical CMB prediction."
        ),
    }


def write_cmb_lite_comparison(
    run_path: Path,
    benchmark_path: Path,
    out_path: Path | None = None,
    *,
    benchmark_label: str = "Planck2018_TT_binned",
    source_url: str | None = None,
    field_names: list[str] | None = None,
) -> dict[str, Any]:
    cl_path = Path(run_path) / "cl_comparison_report.json"
    if not cl_path.exists():
        raise FileNotFoundError(f"missing C_l report: {cl_path}")
    cl_report = json.loads(cl_path.read_text(encoding="utf-8"))
    benchmark_rows = load_planck_tt_binned(Path(benchmark_path))
    report = cmb_lite_comparison_report(
        cl_report,
        benchmark_rows,
        benchmark_label=benchmark_label,
        source_url=source_url,
        field_names=field_names,
    )
    destination = out_path or (Path(run_path) / "cmb_lite_comparison_report.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _compare_spectrum_shape(
    sim_spectrum: list[dict[str, float]],
    benchmark_rows: list[dict[str, float]],
    *,
    sample_count: int = 128,
) -> dict[str, Any]:
    sim = [(float(row["ell"]), float(row.get("D_ell", 0.0))) for row in sim_spectrum if float(row["ell"]) >= 2]
    bench = [(float(row["ell"]), float(row.get("D_ell", 0.0))) for row in benchmark_rows if float(row.get("D_ell", 0.0)) > 0]
    if len(sim) < 2 or len(bench) < 2:
        return {"usable": False, "reason": "not_enough_points"}
    sim_x, sim_y = _normalize_curve(sim)
    bench_x, bench_y = _normalize_curve(bench)
    grid = np.linspace(0.0, 1.0, int(sample_count))
    sim_curve = _standardize(np.interp(grid, sim_x, sim_y))
    bench_curve = _standardize(np.interp(grid, bench_x, bench_y))
    amplitude = float(np.dot(sim_curve, bench_curve) / max(float(np.dot(sim_curve, sim_curve)), 1e-12))
    positive_amplitude = max(0.0, amplitude)
    residual = amplitude * sim_curve - bench_curve
    positive_residual = positive_amplitude * sim_curve - bench_curve
    corr = float(np.corrcoef(sim_curve, bench_curve)[0, 1]) if np.std(sim_curve) > 1e-12 else 0.0
    sim_peak = _peak_fraction(sim)
    bench_peak = _peak_fraction(bench)
    normalized_axis = {
        "usable": True,
        "shape_correlation": corr,
        "unconstrained_amplitude": amplitude,
        "unconstrained_normalized_rmse": float(np.sqrt(np.mean(residual * residual))),
        "positive_amplitude": positive_amplitude,
        "positive_amp_normalized_rmse": float(np.sqrt(np.mean(positive_residual * positive_residual))),
        "usable_positive_shape": bool(corr > 0.0 and positive_amplitude > 0.0),
        "sim_peak_fraction": sim_peak,
        "benchmark_peak_fraction": bench_peak,
        "peak_fraction_delta": float(abs(sim_peak - bench_peak)),
        "sample_count": int(sample_count),
        "claim_boundary": "normalized-axis shape diagnostic; not physical ell-space",
    }
    real_ell = _real_ell_physical_comparison(sim, bench)
    return {
        "usable": True,
        "normalized_axis_shape_diagnostic": normalized_axis,
        "real_ell_physical_comparison": real_ell,
        "shape_correlation": corr,
        "normalized_rmse": normalized_axis["positive_amp_normalized_rmse"],
        "unconstrained_normalized_rmse": normalized_axis["unconstrained_normalized_rmse"],
        "best_fit_amplitude": amplitude,
        "positive_best_fit_amplitude": positive_amplitude,
        "positive_amp_normalized_rmse": normalized_axis["positive_amp_normalized_rmse"],
        "usable_positive_shape": normalized_axis["usable_positive_shape"],
        "sim_peak_fraction": sim_peak,
        "benchmark_peak_fraction": bench_peak,
        "peak_fraction_delta": float(abs(sim_peak - bench_peak)),
        "sample_count": int(sample_count),
    }


def _real_ell_physical_comparison(
    sim: list[tuple[float, float]],
    bench: list[tuple[float, float]],
) -> dict[str, Any]:
    sim_ell = np.asarray([row[0] for row in sim], dtype=float)
    sim_values = np.asarray([row[1] for row in sim], dtype=float)
    bench_ell = np.asarray([row[0] for row in bench], dtype=float)
    bench_values = np.asarray([row[1] for row in bench], dtype=float)
    if sim_ell.size < 2 or bench_ell.size < 2:
        return {"usable": False, "reason": "not_enough_points"}
    sim_min = float(np.min(sim_ell))
    sim_max = float(np.max(sim_ell))
    bench_min = float(np.min(bench_ell))
    bench_max = float(np.max(bench_ell))
    if sim_min > bench_min or sim_max < bench_max:
        return {
            "usable": False,
            "reason": "sim_ell_range_does_not_cover_benchmark_ell_range",
            "sim_ell_min": sim_min,
            "sim_ell_max": sim_max,
            "benchmark_ell_min": bench_min,
            "benchmark_ell_max": bench_max,
            "claim_boundary": "screen proxy does not cover physical CMB multipole range",
        }
    sim_interp = np.interp(bench_ell, sim_ell, sim_values)
    sim_z = _standardize(sim_interp)
    bench_z = _standardize(bench_values)
    amplitude = float(np.dot(sim_z, bench_z) / max(float(np.dot(sim_z, sim_z)), 1e-12))
    positive_amplitude = max(0.0, amplitude)
    residual = positive_amplitude * sim_z - bench_z
    corr = float(np.corrcoef(sim_z, bench_z)[0, 1]) if np.std(sim_z) > 1e-12 else 0.0
    return {
        "usable": True,
        "shape_correlation": corr,
        "positive_amplitude": positive_amplitude,
        "positive_amp_normalized_rmse": float(np.sqrt(np.mean(residual * residual))),
        "usable_positive_shape": bool(corr > 0.0 and positive_amplitude > 0.0),
        "sim_ell_min": sim_min,
        "sim_ell_max": sim_max,
        "benchmark_ell_min": bench_min,
        "benchmark_ell_max": bench_max,
        "claim_boundary": "real-ell diagnostic; still not a likelihood",
    }


def _normalize_curve(rows: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([row[0] for row in rows], dtype=float)
    y = np.asarray([row[1] for row in rows], dtype=float)
    span = max(float(np.max(x) - np.min(x)), 1e-12)
    return (x - float(np.min(x))) / span, y


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values - float(np.mean(values))
    scale = float(np.std(values))
    if scale < 1e-12:
        return np.zeros_like(values)
    return values / scale


def _peak_fraction(rows: list[tuple[float, float]]) -> float:
    x, y = _normalize_curve(rows)
    index = int(np.argmax(y))
    return float(x[index])
