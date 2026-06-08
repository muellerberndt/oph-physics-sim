from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import eval_legendre

from oph_fpe.constants.oph_pixel import (
    OPHPixelConstants,
    screen_radius_planck,
    total_area_planck,
    total_entropy_capacity,
)
from oph_fpe.cosmology.oph_screen_power import screen_power_fit_from_spectrum


DEFAULT_LOW_LMAX = 29


def cmb_anomaly_report(
    cl_report: dict[str, Any],
    *,
    source_dir: Path | None = None,
    fields: list[str] | None = None,
    low_lmax: int = DEFAULT_LOW_LMAX,
    parity_lmax: int = DEFAULT_LOW_LMAX,
    s12_lmax: int = DEFAULT_LOW_LMAX,
) -> dict[str, Any]:
    """Compute screen-level OPH CMB anomaly diagnostics from a C_l receipt.

    This report is intentionally not a Boltzmann solver or Planck likelihood.
    It answers the finite-screen part of the OPH CMB question: whether a
    particular observer-screen run emits low-multipole, odd/even, large-angle,
    and finite-capacity signatures before any acoustic transfer is applied.
    """

    field_reports = cl_report.get("fields", {}) or {}
    control_reports = cl_report.get("controls", {}) or {}
    names = [name for name in (fields or list(field_reports)) if name in field_reports]
    point_count = _int_or_none(cl_report.get("point_count"))
    ell_max = _int_or_none(cl_report.get("ell_max"))
    pixel = OPHPixelConstants()
    field_anomalies = {}
    row_records: list[dict[str, Any]] = []
    for name in names:
        spectrum = field_reports.get(name, {}).get("spectrum", [])
        stats = spectrum_anomaly_stats(
            spectrum,
            low_lmax=low_lmax,
            parity_lmax=parity_lmax,
            s12_lmax=s12_lmax,
            field_name=name,
            point_count=point_count,
        )
        controls = {}
        for control_name, control in (control_reports.get(name, {}) or {}).items():
            control_stats = spectrum_anomaly_stats(
                control.get("spectrum", []),
                low_lmax=low_lmax,
                parity_lmax=parity_lmax,
                s12_lmax=s12_lmax,
                field_name=f"{name}:{control_name}",
                point_count=point_count,
            )
            controls[control_name] = control_stats
        separation = anomaly_control_separation(stats, controls)
        field_anomalies[name] = {
            "stats": stats,
            "controls": controls,
            "control_separation": separation,
            "question_readouts": _field_question_readouts(stats, separation),
        }
        row_records.extend(_field_rows(name, stats, controls, separation))

    aggregate = _aggregate_field_reports(field_anomalies)
    report = {
        "mode": "finite_screen_cmb_anomaly_diagnostics_v0",
        "input_mode": cl_report.get("estimator", "unknown"),
        "point_count": point_count,
        "ell_max": ell_max,
        "diagnostic_lmax": {
            "low_power": int(low_lmax),
            "parity": int(parity_lmax),
            "large_angle_s12_proxy": int(s12_lmax),
        },
        "screen_capacity": _screen_capacity(point_count, pixel),
        "public_reference": _load_public_reference(source_dir),
        "fields": field_anomalies,
        "aggregate": aggregate,
        "question_answer_status": _question_answer_status(aggregate),
        "physical_cmb_prediction": False,
        "bulk_3d_established": False,
        "claim_boundary": (
            "Finite observer-screen CMB-anomaly diagnostic from run C_l receipts. It can test whether "
            "screen synchronization leaves low-ell, parity, large-angle, and finite-capacity imprints. "
            "It is not a Planck likelihood, not a Boltzmann acoustic transfer, not a physical CMB "
            "prediction, and not proof of 3D bulk emergence."
        ),
    }
    report["_csv_rows"] = row_records
    return report


def spectrum_anomaly_stats(
    spectrum: list[dict[str, Any]],
    *,
    low_lmax: int = DEFAULT_LOW_LMAX,
    parity_lmax: int = DEFAULT_LOW_LMAX,
    s12_lmax: int = DEFAULT_LOW_LMAX,
    field_name: str = "field",
    point_count: int | None = None,
) -> dict[str, Any]:
    ell, c_ell, d_ell = _spectrum_arrays(spectrum)
    positive = d_ell > 0.0
    stats: dict[str, Any] = {
        "field": field_name,
        "usable": bool(ell.size >= 3 and np.any(positive)),
        "multipole_count": int(ell.size),
        "ell_min": float(np.min(ell)) if ell.size else None,
        "ell_max": float(np.max(ell)) if ell.size else None,
    }
    if not stats["usable"]:
        return stats | {"reason": "not_enough_positive_multipoles"}

    low_mask = (ell >= 2.0) & (ell <= float(low_lmax))
    high_mask = ell > float(low_lmax)
    stats |= {
        "low_lmax": int(low_lmax),
        "low_power_sum_Dell": _safe_sum(d_ell[low_mask]),
        "low_power_abs_sum_Dell": _safe_sum(np.abs(d_ell[low_mask])),
        "total_abs_power_Dell": _safe_sum(np.abs(d_ell[ell >= 2.0])),
        "high_power_abs_sum_Dell": _safe_sum(np.abs(d_ell[high_mask])),
        "low_multipole_count": int(np.sum(low_mask)),
        "positive_multipole_count": int(np.sum(positive)),
        "peak_ell": float(ell[np.argmax(d_ell)]) if ell.size else None,
    }
    total_abs = float(stats["total_abs_power_Dell"] or 0.0)
    low_abs = float(stats["low_power_abs_sum_Dell"] or 0.0)
    high_abs = float(stats["high_power_abs_sum_Dell"] or 0.0)
    stats["low_power_abs_fraction"] = low_abs / total_abs if total_abs > 0.0 else None
    stats["high_to_low_abs_power_ratio"] = high_abs / low_abs if low_abs > 0.0 else None
    stats["odd_even_ratio_Dell"] = _odd_even_ratio(ell, d_ell, parity_lmax, use_abs=False)
    stats["odd_even_abs_ratio_Dell"] = _odd_even_ratio(ell, d_ell, parity_lmax, use_abs=True)
    stats["parity_lmax"] = int(parity_lmax)
    stats["parity_log_abs_deviation"] = _log_abs_deviation_from_one(stats["odd_even_abs_ratio_Dell"])
    stats["S_1_2_scalar_proxy"] = _s12_scalar_proxy(ell, c_ell, s12_lmax)
    stats["S_1_2_lmax"] = int(s12_lmax)
    stats["screen_power_fit"] = screen_power_fit_from_spectrum(
        spectrum,
        field_name=field_name,
        point_count=point_count,
        ell_min=max(4.0, min(20.0, float(low_lmax))),
    )
    fit = stats["screen_power_fit"]
    stats["eta_R_estimate"] = fit.get("eta_R_estimate") if fit.get("fit_available") else None
    stats["n_s_proxy"] = fit.get("n_s_proxy") if fit.get("fit_available") else None
    stats["near_scale_invariant_proxy"] = bool(
        fit.get("fit_available")
        and stats["eta_R_estimate"] is not None
        and abs(float(stats["eta_R_estimate"])) < 0.25
    )
    stats["planck_tilt_compatible_proxy"] = bool(fit.get("within_planck_eta_R_1sigma", False))
    return stats


def anomaly_control_separation(
    target: dict[str, Any],
    controls: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metrics = [
        "low_power_abs_fraction",
        "odd_even_abs_ratio_Dell",
        "parity_log_abs_deviation",
        "S_1_2_scalar_proxy",
        "eta_R_estimate",
    ]
    result: dict[str, Any] = {}
    for metric in metrics:
        control_values = [
            float(value)
            for value in (control.get(metric) for control in controls.values())
            if isinstance(value, (int, float)) and np.isfinite(float(value))
        ]
        target_value = target.get(metric)
        if not isinstance(target_value, (int, float)) or not np.isfinite(float(target_value)):
            result[metric] = {"usable": False, "reason": "target_metric_missing"}
            continue
        if not control_values:
            result[metric] = {"usable": False, "reason": "control_metric_missing", "target": float(target_value)}
            continue
        mean = float(np.mean(control_values))
        std = float(np.std(control_values, ddof=1)) if len(control_values) > 1 else 0.0
        delta = float(target_value) - mean
        result[metric] = {
            "usable": True,
            "target": float(target_value),
            "control_mean": mean,
            "control_min": float(np.min(control_values)),
            "control_max": float(np.max(control_values)),
            "control_std": std,
            "target_minus_control_mean": delta,
            "target_ratio_to_control_mean": float(target_value) / mean if abs(mean) > 1.0e-30 else None,
            "target_z_vs_controls": delta / std if std > 0.0 else None,
        }
    result["low_power_suppressed_vs_controls"] = _metric_less_than_controls(result, "low_power_abs_fraction")
    result["large_angle_suppressed_vs_controls"] = _metric_less_than_controls(result, "S_1_2_scalar_proxy")
    result["parity_more_asymmetric_than_controls"] = _metric_greater_than_controls(
        result,
        "parity_log_abs_deviation",
    )
    return result


def write_cmb_anomaly_report(
    run_dir: Path,
    out: Path | None = None,
    *,
    source_dir: Path | None = None,
    fields: list[str] | None = None,
    low_lmax: int = DEFAULT_LOW_LMAX,
    parity_lmax: int = DEFAULT_LOW_LMAX,
    s12_lmax: int = DEFAULT_LOW_LMAX,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    cl_path = run_path / "cl_comparison_report.json"
    cl_report = json.loads(cl_path.read_text(encoding="utf-8"))
    report = cmb_anomaly_report(
        cl_report,
        source_dir=source_dir,
        fields=fields,
        low_lmax=low_lmax,
        parity_lmax=parity_lmax,
        s12_lmax=s12_lmax,
    )
    csv_rows = report.pop("_csv_rows", [])
    destination = Path(out) if out is not None else run_path
    if destination.suffix.lower() == ".json":
        out_dir = destination.parent
        json_path = destination
    else:
        out_dir = destination
        json_path = out_dir / "cmb_anomaly_report.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / "cmb_anomaly_report.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_csv(out_dir / "cmb_anomaly_rows.csv", csv_rows)
    return report


def _spectrum_arrays(spectrum: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    for item in spectrum:
        ell = _float_or_none(item.get("ell"))
        if ell is None or ell < 2.0:
            continue
        c_value = _float_or_none(item.get("C_ell"))
        d_value = _float_or_none(item.get("D_ell"))
        if c_value is None and d_value is None:
            continue
        if c_value is None:
            c_value = (2.0 * math.pi * float(d_value)) / (ell * (ell + 1.0))
        if d_value is None:
            d_value = ell * (ell + 1.0) * float(c_value) / (2.0 * math.pi)
        rows.append((ell, float(c_value), float(d_value)))
    if not rows:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    rows.sort(key=lambda row: row[0])
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def _odd_even_ratio(ell: np.ndarray, d_ell: np.ndarray, lmax: int, *, use_abs: bool) -> float | None:
    mask = (ell >= 2.0) & (ell <= float(lmax))
    values = np.abs(d_ell) if use_abs else d_ell
    odd = _safe_sum(values[mask & ((ell.astype(int) % 2) == 1)])
    even = _safe_sum(values[mask & ((ell.astype(int) % 2) == 0)])
    if even <= 0.0:
        return None
    return odd / even


def _s12_scalar_proxy(ell: np.ndarray, c_ell: np.ndarray, lmax: int) -> float | None:
    mask = (ell >= 2.0) & (ell <= float(lmax)) & np.isfinite(c_ell)
    if int(np.sum(mask)) < 2:
        return None
    use_ell = ell[mask].astype(int)
    use_c = c_ell[mask]
    nodes, weights = np.polynomial.legendre.leggauss(256)
    x = -0.25 + 0.75 * nodes
    dx_weights = 0.75 * weights
    corr = np.zeros_like(x, dtype=float)
    for l_value, c_value in zip(use_ell, use_c):
        corr += ((2 * int(l_value) + 1) / (4.0 * math.pi)) * float(c_value) * eval_legendre(
            int(l_value),
            x,
        )
    return float(np.sum(dx_weights * corr * corr))


def _screen_capacity(point_count: int | None, pixel: OPHPixelConstants) -> dict[str, Any]:
    if point_count is None or int(point_count) <= 0:
        return {"available": False}
    n = int(point_count)
    return {
        "available": True,
        "patch_count": n,
        "P": pixel.P,
        "cell_area_planck": pixel.cell_area_planck,
        "cell_entropy_capacity": pixel.cell_entropy_capacity,
        "total_area_planck": total_area_planck(n, pixel.P),
        "total_entropy_capacity": total_entropy_capacity(n, pixel.P),
        "physical_cell_toy_radius_planck": screen_radius_planck(n, pixel.P),
        "ell_sqrt_patch_capacity_proxy": math.sqrt(float(n)) - 1.0,
        "angular_mode_count_proxy_at_low_lmax_29": float((DEFAULT_LOW_LMAX + 1) ** 2),
        "low_l_capacity_fraction_proxy": float((DEFAULT_LOW_LMAX + 1) ** 2) / float(n),
        "claim_boundary": (
            "capacity metadata for finite screen diagnostics; patch_count is a regulator size unless "
            "the run explicitly declares physical-cell toy mode"
        ),
    }


def _field_question_readouts(stats: dict[str, Any], separation: dict[str, Any]) -> dict[str, Any]:
    return {
        "nearly_scale_invariant": {
            "answered_by_this_report": True,
            "proxy": "screen C_l high-multipole power-law eta_R",
            "near_scale_invariant_proxy": bool(stats.get("near_scale_invariant_proxy", False)),
            "planck_tilt_compatible_proxy": bool(stats.get("planck_tilt_compatible_proxy", False)),
            "eta_R_estimate": stats.get("eta_R_estimate"),
            "n_s_proxy": stats.get("n_s_proxy"),
        },
        "low_multipole_anomaly": {
            "answered_by_this_report": True,
            "low_power_abs_fraction": stats.get("low_power_abs_fraction"),
            "suppressed_vs_controls": separation.get("low_power_suppressed_vs_controls"),
        },
        "parity_asymmetry": {
            "answered_by_this_report": True,
            "odd_even_abs_ratio_Dell": stats.get("odd_even_abs_ratio_Dell"),
            "parity_log_abs_deviation": stats.get("parity_log_abs_deviation"),
            "more_asymmetric_than_controls": separation.get("parity_more_asymmetric_than_controls"),
        },
        "preferred_large_angle_correlations": {
            "answered_by_this_report": False,
            "screen_scalar_large_angle_proxy_available": stats.get("S_1_2_scalar_proxy") is not None,
            "S_1_2_scalar_proxy": stats.get("S_1_2_scalar_proxy"),
            "large_angle_suppressed_vs_controls": separation.get("large_angle_suppressed_vs_controls"),
            "missing": "map-space phases, masks, quadrupole/octopole axes, hemispherical asymmetry, and BipoSH",
        },
        "acoustic_peaks": {
            "answered_by_this_report": False,
            "missing": "photon-baryon transfer or CAMB/CLASS adapter from OPH-derived primordial/covariance inputs",
        },
        "finite_screen_capacity": {
            "answered_by_this_report": True,
            "proxy": "finite N, P-weighted entropy capacity, and low-l angular mode fraction",
        },
    }


def _aggregate_field_reports(fields: dict[str, Any]) -> dict[str, Any]:
    if not fields:
        return {"field_count": 0, "primary_field": None}
    primary = "record_signature" if "record_signature" in fields else next(iter(fields))
    low = _best_field(fields, "low_power_abs_fraction", lower=True)
    large = _best_field(fields, "S_1_2_scalar_proxy", lower=True)
    parity = _best_field(fields, "parity_log_abs_deviation", lower=False)
    tilt = _best_tilt_field(fields)
    return {
        "field_count": len(fields),
        "primary_field": primary,
        "best_low_power_suppression_field": low[0],
        "best_low_power_abs_fraction": low[1],
        "best_large_angle_suppression_field": large[0],
        "best_S_1_2_scalar_proxy": large[1],
        "best_parity_asymmetry_field": parity[0],
        "best_parity_log_abs_deviation": parity[1],
        "best_tilt_field": tilt[0],
        "best_eta_R_estimate": tilt[1],
        "best_n_s_proxy": tilt[2],
        "low_power_suppressed_vs_controls_count": sum(
            1 for field in fields.values() if field["control_separation"].get("low_power_suppressed_vs_controls")
        ),
        "large_angle_suppressed_vs_controls_count": sum(
            1 for field in fields.values() if field["control_separation"].get("large_angle_suppressed_vs_controls")
        ),
        "parity_more_asymmetric_than_controls_count": sum(
            1 for field in fields.values() if field["control_separation"].get("parity_more_asymmetric_than_controls")
        ),
        "planck_tilt_compatible_proxy_count": sum(
            1 for field in fields.values() if field["stats"].get("planck_tilt_compatible_proxy")
        ),
    }


def _question_answer_status(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "why_nearly_scale_invariant": {
            "status": "screen_power_law_proxy_computed",
            "best_eta_R_estimate": aggregate.get("best_eta_R_estimate"),
            "best_n_s_proxy": aggregate.get("best_n_s_proxy"),
            "planck_tilt_compatible_field_count": aggregate.get("planck_tilt_compatible_proxy_count", 0),
            "claim_boundary": "proxy only; physical n_s requires a validated primordial adapter",
        },
        "why_acoustic_peaks": {
            "status": "not_answered_by_screen_anomaly_report",
            "next_required_step": "feed OPH-derived primordial/covariance inputs into photon-baryon transfer",
        },
        "why_low_multipoles_anomalous": {
            "status": "finite_screen_low_ell_proxy_computed",
            "suppressed_vs_controls_field_count": aggregate.get("low_power_suppressed_vs_controls_count", 0),
        },
        "should_there_be_parity_asymmetries": {
            "status": "odd_even_proxy_computed",
            "more_asymmetric_than_controls_field_count": aggregate.get(
                "parity_more_asymmetric_than_controls_count",
                0,
            ),
        },
        "should_there_be_preferred_large_angle_correlations": {
            "status": "scalar_proxy_only",
            "large_angle_suppressed_vs_controls_field_count": aggregate.get(
                "large_angle_suppressed_vs_controls_count",
                0,
            ),
            "missing": "map-level axis/covariance diagnostics",
        },
        "finite_screen_capacity_signatures": {
            "status": "capacity_metadata_and_multipole_fraction_computed",
            "claim_boundary": "not yet a detected cutoff unless spectra show stable finite-N scaling",
        },
    }


def _load_public_reference(source_dir: Path | None) -> dict[str, Any]:
    base = source_dir or _default_source_dir()
    reference: dict[str, Any] = {"source_dir": str(base), "available": False}
    parity_path = base / "3" / "04_parity_lowpower_statistics_TT_TE_EE_v0_5.csv"
    s12_path = base / "3" / "05_scalar_S12_correlation_proxy_TT_EE_v0_5.csv"
    targets_path = base / "4" / "02_current_OPH_CMB_targets_v0_8.csv"
    if parity_path.exists():
        rows = _read_csv(parity_path)
        reference["v0_5_TT_lmax29_parity_lowpower"] = _source_rows(rows, spectrum="TT", lmax="29")
        reference["available"] = True
    if s12_path.exists():
        rows = _read_csv(s12_path)
        reference["v0_5_TT_lmax29_S12_proxy"] = _source_rows(rows, spectrum="TT", lmax="29")
        reference["available"] = True
    if targets_path.exists():
        reference["v0_8_current_targets"] = {
            row["quantity"]: _parse_value_row(row) for row in _read_csv(targets_path) if row.get("quantity")
        }
        reference["available"] = True
    reference["claim_boundary"] = (
        "Public-source OPH CMB-note values imported for context. They are not official Planck likelihoods "
        "and are not finite-screen simulator derivations."
    )
    return reference


def _default_source_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "cosmology" / "correspondence" / "cmb"


def _source_rows(rows: list[dict[str, str]], **selectors: str) -> dict[str, dict[str, Any]]:
    selected = {}
    for row in rows:
        if any(str(row.get(key)) != str(value) for key, value in selectors.items()):
            continue
        label = row.get("source") or row.get("source_label") or f"row_{len(selected)}"
        selected[str(label)] = {key: _maybe_float(value) for key, value in row.items()}
    return selected


def _parse_value_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "value": _maybe_float(row.get("value")),
        "unit": row.get("unit"),
        "method": row.get("method"),
        "status": row.get("status"),
        "public_source_url": row.get("public_source_url"),
    }


def _field_rows(
    name: str,
    stats: dict[str, Any],
    controls: dict[str, dict[str, Any]],
    separation: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        {
            "field": name,
            "kind": "target",
            "metric_source": "finite_screen",
            **_row_metric_subset(stats),
        }
    ]
    for control_name, control in controls.items():
        rows.append(
            {
                "field": name,
                "kind": f"control:{control_name}",
                "metric_source": "finite_screen_control",
                **_row_metric_subset(control),
            }
        )
    for metric, item in separation.items():
        if not isinstance(item, dict) or not item.get("usable"):
            continue
        rows.append(
            {
                "field": name,
                "kind": "target_vs_controls",
                "metric_source": metric,
                "target": item.get("target"),
                "control_mean": item.get("control_mean"),
                "target_minus_control_mean": item.get("target_minus_control_mean"),
                "target_ratio_to_control_mean": item.get("target_ratio_to_control_mean"),
                "target_z_vs_controls": item.get("target_z_vs_controls"),
            }
        )
    return rows


def _row_metric_subset(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "low_power_abs_fraction": stats.get("low_power_abs_fraction"),
        "odd_even_abs_ratio_Dell": stats.get("odd_even_abs_ratio_Dell"),
        "parity_log_abs_deviation": stats.get("parity_log_abs_deviation"),
        "S_1_2_scalar_proxy": stats.get("S_1_2_scalar_proxy"),
        "eta_R_estimate": stats.get("eta_R_estimate"),
        "n_s_proxy": stats.get("n_s_proxy"),
        "peak_ell": stats.get("peak_ell"),
    }


def _markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    questions = report["question_answer_status"]
    capacity = report["screen_capacity"]
    lines = [
        "# Finite-Screen CMB Anomaly Diagnostics",
        "",
        report["claim_boundary"],
        "",
        "## Summary",
        "",
        f"- field count: {aggregate.get('field_count')}",
        f"- primary field: {aggregate.get('primary_field')}",
        f"- best low-power field: {aggregate.get('best_low_power_suppression_field')}",
        f"- best low-power fraction: {_fmt(aggregate.get('best_low_power_abs_fraction'))}",
        f"- best large-angle field: {aggregate.get('best_large_angle_suppression_field')}",
        f"- best S1/2 proxy: {_fmt(aggregate.get('best_S_1_2_scalar_proxy'))}",
        f"- best parity field: {aggregate.get('best_parity_asymmetry_field')}",
        f"- best parity log deviation: {_fmt(aggregate.get('best_parity_log_abs_deviation'))}",
        f"- best eta_R: {_fmt(aggregate.get('best_eta_R_estimate'))}",
        f"- best n_s proxy: {_fmt(aggregate.get('best_n_s_proxy'))}",
        "",
        "## Capacity",
        "",
        f"- patch count: {capacity.get('patch_count', 'n/a')}",
        f"- total entropy capacity: {_fmt(capacity.get('total_entropy_capacity'))}",
        f"- sqrt-N ell capacity proxy: {_fmt(capacity.get('ell_sqrt_patch_capacity_proxy'))}",
        f"- low-l capacity fraction proxy: {_fmt(capacity.get('low_l_capacity_fraction_proxy'))}",
        "",
        "## Question Status",
        "",
    ]
    for key, value in questions.items():
        lines.append(f"- `{key}`: {value.get('status')}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `cmb_anomaly_report.json`",
            "- `cmb_anomaly_report.md`",
            "- `cmb_anomaly_rows.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _best_field(fields: dict[str, Any], metric: str, *, lower: bool) -> tuple[str | None, float | None]:
    best_name = None
    best_value = None
    for name, field in fields.items():
        value = field.get("stats", {}).get(metric)
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            continue
        if best_value is None or (float(value) < best_value if lower else float(value) > best_value):
            best_name = name
            best_value = float(value)
    return best_name, best_value


def _best_tilt_field(fields: dict[str, Any]) -> tuple[str | None, float | None, float | None]:
    target = 0.035
    best_name = None
    best_eta = None
    best_ns = None
    best_delta = None
    for name, field in fields.items():
        eta = field.get("stats", {}).get("eta_R_estimate")
        if not isinstance(eta, (int, float)) or not np.isfinite(float(eta)):
            continue
        delta = abs(float(eta) - target)
        if best_delta is None or delta < best_delta:
            best_name = name
            best_eta = float(eta)
            best_ns = field.get("stats", {}).get("n_s_proxy")
            best_ns = float(best_ns) if isinstance(best_ns, (int, float)) and np.isfinite(float(best_ns)) else None
            best_delta = delta
    return best_name, best_eta, best_ns


def _metric_less_than_controls(result: dict[str, Any], metric: str) -> bool | None:
    item = result.get(metric)
    if not isinstance(item, dict) or not item.get("usable"):
        return None
    return bool(float(item["target"]) < float(item["control_mean"]))


def _metric_greater_than_controls(result: dict[str, Any], metric: str) -> bool | None:
    item = result.get(metric)
    if not isinstance(item, dict) or not item.get("usable"):
        return None
    return bool(float(item["target"]) > float(item["control_mean"]))


def _safe_sum(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sum(values[np.isfinite(values)]))


def _log_abs_deviation_from_one(value: Any) -> float | None:
    parsed = _float_or_none(value)
    if parsed is None or parsed <= 0.0:
        return None
    return abs(float(math.log(parsed)))


def _int_or_none(value: Any) -> int | None:
    parsed = _float_or_none(value)
    return int(parsed) if parsed is not None else None


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _maybe_float(value: Any) -> Any:
    parsed = _float_or_none(value)
    return parsed if parsed is not None else value


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return f"{float(value):.10g}"
    return "n/a"
