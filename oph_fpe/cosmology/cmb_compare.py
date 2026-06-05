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
    if comparisons:
        best_field = min(comparisons, key=lambda key: comparisons[key]["normalized_rmse"])
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
        "field_comparisons": comparisons,
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "shape-only comparison between a gated OPH freezeout-screen C_l proxy and an observed TT "
            "benchmark. Multipole axes are normalized to [0,1] and amplitudes are least-squares "
            "rescaled, so this is not a Planck likelihood, not CAMB/CLASS input, and not a physical "
            "CMB prediction."
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
    residual = amplitude * sim_curve - bench_curve
    corr = float(np.corrcoef(sim_curve, bench_curve)[0, 1]) if np.std(sim_curve) > 1e-12 else 0.0
    sim_peak = _peak_fraction(sim)
    bench_peak = _peak_fraction(bench)
    return {
        "usable": True,
        "shape_correlation": corr,
        "normalized_rmse": float(np.sqrt(np.mean(residual * residual))),
        "best_fit_amplitude": amplitude,
        "sim_peak_fraction": sim_peak,
        "benchmark_peak_fraction": bench_peak,
        "peak_fraction_delta": float(abs(sim_peak - bench_peak)),
        "sample_count": int(sample_count),
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
