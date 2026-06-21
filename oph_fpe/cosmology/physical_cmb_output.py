from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def physical_cmb_output_comparison_report(run_dirs: list[Path]) -> dict[str, Any]:
    """Aggregate measurement-facing CMB TT outputs without promoting them.

    This report answers a different question from the hard physical-CMB input
    contract. It collects real-multipole, Planck-binned TT comparison metrics
    already emitted by CAMB transfer reports so the pack has one physical-unit
    comparison surface. The prediction gate remains owned by
    physical_cmb_input_report/physical_cmb_promotion_audit_report.
    """

    roots = [Path(path) for path in run_dirs]
    input_report = _first_json(roots, "physical_cmb_input_report.json")
    promotion = _first_json(roots, "physical_cmb_promotion_audit_report.json")
    rows: list[dict[str, Any]] = []
    rows.extend(_baseline_rows(_first_json(roots, "camb_lcdm_baseline_report.json")))
    rows.extend(
        _multi_model_rows(
            _first_json(roots, "scale_compressed_cmb_camb_report.json"),
            source_report="scale_compressed_cmb_camb_report.json",
            oph_prefixes=("scale_compressed_",),
        )
    )
    rows.extend(
        _multi_model_rows(
            _first_json(roots, "finite_repair_clock_cmb_camb_report.json"),
            source_report="finite_repair_clock_cmb_camb_report.json",
            oph_prefixes=("finite_repair_clock_",),
        )
    )
    rows = _dedupe_model_rows(rows)
    comparable = [
        row for row in rows
        if row.get("measurement_comparable") and _finite(row.get("amplitude_fit_chi2_per_bin"))
    ]
    oph_rows = [row for row in comparable if row.get("model_role") == "oph_diagnostic"]
    best_all = _best_by_chi2(comparable)
    best_oph = _best_by_chi2(oph_rows)
    best_oph_residual_rows = _best_model_residual_rows(roots, best_oph)
    best_oph_residual_summary = _residual_summary(best_oph_residual_rows)
    peak_feature_rows = _peak_feature_rows(roots, comparable)
    best_oph_peak_feature_summary = _peak_feature_summary(peak_feature_rows, best_oph)
    usable_physical_cmb_data = bool(
        oph_rows
        and best_oph is not None
        and best_oph_residual_summary.get("available", False)
        and best_oph_residual_rows
    )
    contract_receipt = bool(input_report.get("PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT", False))
    promotion_ready = bool(promotion.get("physical_cmb_promotion_ready", False))
    physical_prediction_receipt = bool(
        contract_receipt
        and promotion_ready
        and any(row.get("physical_cmb_prediction") for row in rows)
    )
    return {
        "mode": "physical_cmb_output_comparison_v0",
        "run_dirs": [str(path) for path in roots],
        "PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT": bool(comparable),
        "USABLE_PHYSICAL_CMB_DATA_RECEIPT": usable_physical_cmb_data,
        "usable_physical_cmb_data_receipt": usable_physical_cmb_data,
        "usable_physical_cmb_data_products": [
            "physical_cmb_output_comparison_rows.csv",
            "physical_cmb_best_oph_residuals.csv",
            "physical_cmb_peak_features.csv",
        ] if usable_physical_cmb_data else [],
        "PHYSICAL_CMB_PREDICTION_RECEIPT": physical_prediction_receipt,
        "physical_cmb_prediction": physical_prediction_receipt,
        "physical_cmb_input_contract_receipt": contract_receipt,
        "physical_cmb_promotion_ready": promotion_ready,
        "official_likelihood_ready": bool(promotion.get("official_likelihood_ready", False)),
        "measurement_comparable_model_count": len(comparable),
        "oph_diagnostic_model_count": len(oph_rows),
        "best_measurement_comparable_model": best_all,
        "best_oph_diagnostic_model": best_oph,
        "best_oph_residual_summary": best_oph_residual_summary,
        "best_oph_residual_rows": best_oph_residual_rows,
        "best_oph_peak_feature_summary": best_oph_peak_feature_summary,
        "peak_feature_rows": peak_feature_rows,
        "contract_blockers": input_report.get("blockers") or promotion.get("contract_blockers") or [],
        "promotion_blockers": promotion.get("promotion_blockers") or [],
        "rows": rows,
        "claim_boundary": (
            "Physical-unit TT comparison aggregator for the local Planck 2018 binned table. "
            "USABLE_PHYSICAL_CMB_DATA_RECEIPT means physical-unit OPH TT rows and per-bin residuals "
            "are available for inspection. The curves remain diagnostics unless "
            "PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT, physical_cmb_promotion_ready, and the source reports' "
            "physical_cmb_prediction gates all pass."
        ),
    }


def write_physical_cmb_output_comparison_report(run_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    report = physical_cmb_output_comparison_report(run_dirs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "physical_cmb_output_comparison_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "physical_cmb_output_comparison_report.md").write_text(
        _markdown_report(report),
        encoding="utf-8",
    )
    _write_rows_csv(out / "physical_cmb_output_comparison_rows.csv", report["rows"])
    _write_residual_rows_csv(out / "physical_cmb_best_oph_residuals.csv", report["best_oph_residual_rows"])
    _write_peak_feature_rows_csv(out / "physical_cmb_peak_features.csv", report["peak_feature_rows"])
    return report


def _baseline_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    comparison = report.get("comparison") if isinstance(report.get("comparison"), dict) else {}
    if not comparison:
        return []
    return [
        _comparison_row(
            comparison,
            source_report="camb_lcdm_baseline_report.json",
            model_id="lcdm_baseline",
            model_role="external_baseline",
            physical_cmb_prediction=bool(report.get("physical_cmb_prediction", False)),
        )
    ]


def _multi_model_rows(
    report: dict[str, Any],
    *,
    source_report: str,
    oph_prefixes: tuple[str, ...],
) -> list[dict[str, Any]]:
    comparisons = report.get("comparison") if isinstance(report.get("comparison"), dict) else {}
    rows: list[dict[str, Any]] = []
    for model_id, comparison in comparisons.items():
        if not isinstance(comparison, dict):
            continue
        role = "oph_diagnostic" if str(model_id).startswith(oph_prefixes) else "external_baseline"
        rows.append(
            _comparison_row(
                comparison,
                source_report=source_report,
                model_id=str(model_id),
                model_role=role,
                physical_cmb_prediction=bool(report.get("physical_cmb_prediction", False)),
            )
        )
    return rows


def _comparison_row(
    comparison: dict[str, Any],
    *,
    source_report: str,
    model_id: str,
    model_role: str,
    physical_cmb_prediction: bool,
) -> dict[str, Any]:
    first_peak = _float_or_none(comparison.get("first_peak_ell"))
    benchmark_peak = _float_or_none(comparison.get("benchmark_first_peak_ell"))
    return {
        "source_report": source_report,
        "model_id": model_id,
        "model_role": model_role,
        "measurement_comparable": bool(comparison.get("usable", False)),
        "physical_cmb_prediction": bool(physical_cmb_prediction),
        "bin_count": _int_or_none(comparison.get("bin_count")),
        "shape_correlation": _float_or_none(comparison.get("shape_correlation")),
        "normalized_rmse": _float_or_none(comparison.get("normalized_rmse")),
        "amplitude_fit_chi2_per_bin": _float_or_none(comparison.get("amplitude_fit_chi2_per_bin")),
        "best_fit_column_chi2_per_bin": _float_or_none(comparison.get("best_fit_column_chi2_per_bin")),
        "mean_absolute_fractional_error": _float_or_none(comparison.get("mean_absolute_fractional_error")),
        "first_peak_ell": first_peak,
        "benchmark_first_peak_ell": benchmark_peak,
        "first_peak_abs_delta": (
            abs(first_peak - benchmark_peak)
            if first_peak is not None and benchmark_peak is not None
            else None
        ),
    }


def _dedupe_model_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("source_report")), str(row.get("model_id")))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _best_by_chi2(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [row for row in rows if _finite(row.get("amplitude_fit_chi2_per_bin"))]
    if not usable:
        return None
    best = min(usable, key=lambda row: float(row["amplitude_fit_chi2_per_bin"]))
    return {
        "source_report": best.get("source_report"),
        "model_id": best.get("model_id"),
        "model_role": best.get("model_role"),
        "amplitude_fit_chi2_per_bin": best.get("amplitude_fit_chi2_per_bin"),
        "shape_correlation": best.get("shape_correlation"),
        "normalized_rmse": best.get("normalized_rmse"),
        "first_peak_abs_delta": best.get("first_peak_abs_delta"),
    }


def _best_model_residual_rows(roots: list[Path], best: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not best:
        return []
    source_report = str(best.get("source_report") or "")
    model_id = str(best.get("model_id") or "")
    source_csv = _source_csv_name(source_report)
    model_column = _model_column(source_report, model_id)
    if source_csv is None or model_column is None:
        return []
    path = _first_path(roots, source_csv)
    if path is None:
        return []
    rows: list[dict[str, Any]] = []
    for row in _read_csv_rows(path):
        ell = _float_or_none(row.get("ell"))
        observed = _float_or_none(row.get("observed_D_ell"))
        model = _float_or_none(row.get(model_column))
        sigma = _sigma_from_row(row)
        if ell is None or observed is None or model is None:
            continue
        residual = float(model - observed)
        residual_sigma = residual / sigma if sigma is not None and sigma > 0 else None
        rows.append(
            {
                "source_report": source_report,
                "source_csv": source_csv,
                "model_id": model_id,
                "model_column": model_column,
                "ell": ell,
                "observed_D_ell": observed,
                "sigma_D_ell": sigma,
                "model_D_ell": model,
                "residual_D_ell": residual,
                "residual_sigma": residual_sigma,
                "abs_residual_sigma": abs(residual_sigma) if residual_sigma is not None else None,
                "fractional_residual": residual / observed if abs(observed) > 1.0e-12 else None,
            }
        )
    return rows


def _residual_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    residuals = [float(row["residual_D_ell"]) for row in rows if _finite(row.get("residual_D_ell"))]
    sigma_residuals = [float(row["residual_sigma"]) for row in rows if _finite(row.get("residual_sigma"))]
    fractional = [float(row["fractional_residual"]) for row in rows if _finite(row.get("fractional_residual"))]
    if not rows:
        return {
            "available": False,
            "bin_count": 0,
            "claim_boundary": "No per-bin residual rows were available for the selected best OPH diagnostic model.",
        }
    max_abs_sigma_row = max(
        (row for row in rows if _finite(row.get("abs_residual_sigma"))),
        key=lambda row: float(row["abs_residual_sigma"]),
        default=None,
    )
    return {
        "available": True,
        "model_id": rows[0].get("model_id"),
        "source_report": rows[0].get("source_report"),
        "source_csv": rows[0].get("source_csv"),
        "bin_count": len(rows),
        "rms_residual_D_ell": _rms(residuals),
        "mean_abs_fractional_residual": _mean_abs(fractional),
        "rms_sigma_residual": _rms(sigma_residuals),
        "max_abs_sigma_residual": (
            float(max_abs_sigma_row["abs_residual_sigma"]) if max_abs_sigma_row is not None else None
        ),
        "max_abs_sigma_ell": max_abs_sigma_row.get("ell") if max_abs_sigma_row is not None else None,
        "claim_boundary": (
            "Per-bin residuals for the best measurement-comparable OPH diagnostic curve. "
            "This is a physical-unit comparison to the local Planck binned TT table, not a "
            "physical CMB prediction receipt."
        ),
    }


def _peak_feature_rows(roots: list[Path], rows: list[dict[str, Any]], *, max_peaks: int = 5) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for comparison in rows:
        if not comparison.get("measurement_comparable"):
            continue
        source_report = str(comparison.get("source_report") or "")
        model_id = str(comparison.get("model_id") or "")
        source_csv = _source_csv_name(source_report)
        model_column = _model_column(source_report, model_id)
        if source_csv is None or model_column is None:
            continue
        path = _first_path(roots, source_csv)
        if path is None:
            continue
        points = _curve_points(_read_csv_rows(path), model_column)
        observed_peaks = _curve_peaks(points["observed"], max_peaks=max_peaks)
        model_peaks = _curve_peaks(points["model"], max_peaks=max_peaks)
        for index, (observed, model) in enumerate(zip(observed_peaks, model_peaks), start=1):
            observed_ell, observed_value = observed
            model_ell, model_value = model
            height_delta = model_value - observed_value
            fractional_height_delta = (
                height_delta / observed_value if abs(observed_value) > 1.0e-12 else None
            )
            out.append(
                {
                    "source_report": source_report,
                    "source_csv": source_csv,
                    "model_id": model_id,
                    "model_role": comparison.get("model_role"),
                    "model_column": model_column,
                    "measurement_comparable": bool(comparison.get("measurement_comparable", False)),
                    "physical_cmb_prediction": bool(comparison.get("physical_cmb_prediction", False)),
                    "peak_index": index,
                    "observed_peak_ell": observed_ell,
                    "observed_peak_D_ell": observed_value,
                    "model_peak_ell": model_ell,
                    "model_peak_D_ell": model_value,
                    "ell_delta": model_ell - observed_ell,
                    "abs_ell_delta": abs(model_ell - observed_ell),
                    "D_ell_delta": height_delta,
                    "fractional_D_ell_delta": fractional_height_delta,
                    "abs_fractional_D_ell_delta": (
                        abs(fractional_height_delta) if fractional_height_delta is not None else None
                    ),
                    "claim_boundary": (
                        "Diagnostic peak feature from binned TT curves. This compares physical-unit "
                        "output curves to observed binned TT peaks and is not a physical CMB "
                        "prediction receipt."
                    ),
                }
            )
    return out


def _curve_points(rows: list[dict[str, str]], model_column: str) -> dict[str, list[tuple[float, float]]]:
    observed: list[tuple[float, float]] = []
    model: list[tuple[float, float]] = []
    for row in rows:
        ell = _float_or_none(row.get("ell"))
        observed_value = _float_or_none(row.get("observed_D_ell"))
        model_value = _float_or_none(row.get(model_column))
        if ell is None:
            continue
        if observed_value is not None:
            observed.append((ell, observed_value))
        if model_value is not None:
            model.append((ell, model_value))
    return {
        "observed": sorted(observed, key=lambda point: point[0]),
        "model": sorted(model, key=lambda point: point[0]),
    }


def _curve_peaks(points: list[tuple[float, float]], *, max_peaks: int) -> list[tuple[float, float]]:
    acoustic_points = [(ell, value) for ell, value in points if ell >= 30.0]
    if not acoustic_points:
        return []
    peaks: list[tuple[float, float]] = []
    for index, point in enumerate(acoustic_points):
        if index == 0 or index == len(acoustic_points) - 1:
            continue
        previous_value = acoustic_points[index - 1][1]
        value = point[1]
        next_value = acoustic_points[index + 1][1]
        if (value >= previous_value and value > next_value) or (value > previous_value and value >= next_value):
            peaks.append(point)
    if not peaks:
        peaks = [max(acoustic_points, key=lambda point: point[1])]
    return peaks[: max(0, int(max_peaks))]


def _peak_feature_summary(rows: list[dict[str, Any]], best: dict[str, Any] | None) -> dict[str, Any]:
    if not best:
        return {
            "available": False,
            "peak_count": 0,
            "claim_boundary": "No best OPH diagnostic model was available for peak-feature summarization.",
        }
    source_report = best.get("source_report")
    model_id = best.get("model_id")
    selected = [
        row
        for row in rows
        if row.get("source_report") == source_report and row.get("model_id") == model_id
    ]
    if not selected:
        return {
            "available": False,
            "model_id": model_id,
            "source_report": source_report,
            "peak_count": 0,
            "claim_boundary": (
                "No direct peak-feature rows were available for the selected best OPH diagnostic model."
            ),
        }
    ell_deltas = [float(row["abs_ell_delta"]) for row in selected if _finite(row.get("abs_ell_delta"))]
    height_deltas = [
        float(row["abs_fractional_D_ell_delta"])
        for row in selected
        if _finite(row.get("abs_fractional_D_ell_delta"))
    ]
    return {
        "available": True,
        "model_id": model_id,
        "source_report": source_report,
        "peak_count": len(selected),
        "mean_abs_peak_ell_delta": _mean(ell_deltas),
        "max_abs_peak_ell_delta": max(ell_deltas) if ell_deltas else None,
        "mean_abs_peak_height_fractional_delta": _mean(height_deltas),
        "max_abs_peak_height_fractional_delta": max(height_deltas) if height_deltas else None,
        "first_peak_ell_delta": selected[0].get("ell_delta"),
        "first_peak_height_fractional_delta": selected[0].get("fractional_D_ell_delta"),
        "claim_boundary": (
            "Direct binned-TT peak-position and peak-height comparison for the best OPH diagnostic "
            "curve. This is a measurement-facing diagnostic, not a physical CMB prediction receipt."
        ),
    }


def _source_csv_name(source_report: str) -> str | None:
    if source_report == "scale_compressed_cmb_camb_report.json":
        return "scale_compressed_cmb_tt_bins.csv"
    if source_report == "finite_repair_clock_cmb_camb_report.json":
        return "finite_repair_clock_cmb_tt_bins.csv"
    if source_report == "camb_lcdm_baseline_report.json":
        return "camb_lcdm_tt_bins.csv"
    return None


def _model_column(source_report: str, model_id: str) -> str | None:
    if model_id == "lcdm_baseline":
        return "amplitude_fit_camb_D_ell"
    if model_id == "camb_lcdm_powerlaw":
        return "camb_lcdm_powerlaw_D_ell"
    if source_report in {
        "scale_compressed_cmb_camb_report.json",
        "finite_repair_clock_cmb_camb_report.json",
    }:
        return f"{model_id}_D_ell"
    return None


def _sigma_from_row(row: dict[str, str]) -> float | None:
    sigma = _float_or_none(row.get("sigma_D_ell"))
    if sigma is not None:
        return sigma
    minus = _float_or_none(row.get("minus_dD_ell"))
    plus = _float_or_none(row.get("plus_dD_ell"))
    values = [value for value in (minus, plus) if value is not None and value > 0]
    if not values:
        return None
    return sum(values) / len(values)


def _rms(values: list[float]) -> float | None:
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values) / len(values))


def _mean_abs(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(abs(value) for value in values) / len(values)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "source_report",
        "model_id",
        "model_role",
        "measurement_comparable",
        "physical_cmb_prediction",
        "bin_count",
        "shape_correlation",
        "normalized_rmse",
        "amplitude_fit_chi2_per_bin",
        "best_fit_column_chi2_per_bin",
        "mean_absolute_fractional_error",
        "first_peak_ell",
        "benchmark_first_peak_ell",
        "first_peak_abs_delta",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _write_peak_feature_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "source_report",
        "source_csv",
        "model_id",
        "model_role",
        "model_column",
        "measurement_comparable",
        "physical_cmb_prediction",
        "peak_index",
        "observed_peak_ell",
        "observed_peak_D_ell",
        "model_peak_ell",
        "model_peak_D_ell",
        "ell_delta",
        "abs_ell_delta",
        "D_ell_delta",
        "fractional_D_ell_delta",
        "abs_fractional_D_ell_delta",
        "claim_boundary",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _write_residual_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "source_report",
        "source_csv",
        "model_id",
        "model_column",
        "ell",
        "observed_D_ell",
        "sigma_D_ell",
        "model_D_ell",
        "residual_D_ell",
        "residual_sigma",
        "abs_residual_sigma",
        "fractional_residual",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _markdown_report(report: dict[str, Any]) -> str:
    best = report.get("best_oph_diagnostic_model") or {}
    residuals = report.get("best_oph_residual_summary") or {}
    peaks = report.get("best_oph_peak_feature_summary") or {}
    lines = [
        "# Physical CMB Output Comparison",
        "",
        report.get("claim_boundary", ""),
        "",
        "## Receipts",
        "",
        f"- output comparison receipt: `{str(report.get('PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT', False)).lower()}`",
        f"- usable physical CMB data receipt: `{str(report.get('usable_physical_cmb_data_receipt', False)).lower()}`",
        f"- physical prediction receipt: `{str(report.get('PHYSICAL_CMB_PREDICTION_RECEIPT', False)).lower()}`",
        f"- input contract receipt: `{str(report.get('physical_cmb_input_contract_receipt', False)).lower()}`",
        f"- promotion ready: `{str(report.get('physical_cmb_promotion_ready', False)).lower()}`",
        f"- measurement-comparable model count: `{report.get('measurement_comparable_model_count', 0)}`",
        "",
        "## Best OPH Diagnostic",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- model: `{best.get('model_id')}`",
                f"- source report: `{best.get('source_report')}`",
                f"- amplitude-fit chi2/bin: `{best.get('amplitude_fit_chi2_per_bin')}`",
                f"- shape correlation: `{best.get('shape_correlation')}`",
                f"- residual bins: `{residuals.get('bin_count', 0)}`",
                f"- RMS sigma residual: `{residuals.get('rms_sigma_residual')}`",
                f"- max abs sigma residual: `{residuals.get('max_abs_sigma_residual')}` at ell `{residuals.get('max_abs_sigma_ell')}`",
                f"- peak feature count: `{peaks.get('peak_count', 0)}`",
                f"- mean abs peak ell delta: `{peaks.get('mean_abs_peak_ell_delta')}`",
                f"- mean abs peak-height fractional delta: `{peaks.get('mean_abs_peak_height_fractional_delta')}`",
            ]
        )
    else:
        lines.append("- none")
    blockers = report.get("promotion_blockers") or report.get("contract_blockers") or []
    lines.extend(["", "## Remaining Promotion Blockers", ""])
    lines.extend(f"- `{blocker}`" for blocker in blockers[:30]) if blockers else lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _first_json(roots: list[Path], name: str) -> dict[str, Any]:
    for root in roots:
        root = Path(root)
        candidates: list[Path] = []
        if root.is_file() and root.name == name:
            candidates.append(root)
        direct = root / name
        if direct.exists():
            candidates.append(direct)
        if root.exists() and root.is_dir():
            candidates.extend(sorted(root.glob(f"**/{name}")))
        for path in candidates:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict) and value:
                return value
    return {}


def _first_path(roots: list[Path], name: str) -> Path | None:
    for root in roots:
        root = Path(root)
        if root.is_file() and root.name == name:
            return root
        direct = root / name
        if direct.exists():
            return direct
        if root.exists() and root.is_dir():
            matches = sorted(root.glob(f"**/{name}"))
            if matches:
                return matches[0]
    return None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite(value: Any) -> bool:
    parsed = _float_or_none(value)
    return parsed is not None
