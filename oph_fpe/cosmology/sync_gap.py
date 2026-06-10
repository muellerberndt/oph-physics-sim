from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean, median
from typing import Any

import numpy as np

from oph_fpe.claims import CONTINUATION, COSMOLOGY_PERTURBATION_RECEIPT, with_claim_metadata


def synchronization_gap_report(
    run_dirs: list[Path],
    *,
    ell_max_cmb: int = 32,
    min_gamma_per_cycle: float = 1.0e-3,
    min_control_z: float = 1.0,
) -> dict[str, Any]:
    """Audit the finite-lattice low-k synchronization gap gate.

    The paper-side inflation alternative needs either a same-boundary selector
    or a nonzero low-k repair gap across the observed CMB band. Current OPH-FPE
    runs usually store final freezeout spectra and a global mismatch trace, not
    time-resolved harmonic coefficients. This report therefore separates a real
    time-resolved gap receipt from the weaker cached-run proxy.
    """

    rows = [_run_row(path, ell_max_cmb=ell_max_cmb, min_gamma_per_cycle=min_gamma_per_cycle, min_control_z=min_control_z)
            for path in _find_run_dirs(run_dirs)]
    rows = [row for row in rows if row.get("has_gap_inputs")]
    aggregate = _aggregate(rows)
    report = {
        "mode": "oph_low_k_synchronization_gap_audit_v0",
        "run_count": len(rows),
        "ell_max_cmb": int(ell_max_cmb),
        "thresholds": {
            "min_gamma_per_cycle": float(min_gamma_per_cycle),
            "min_control_z": float(min_control_z),
        },
        "aggregate": aggregate,
        "rows": rows,
        "low_k_gap_established": bool(aggregate.get("time_resolved_gap_pass_count", 0) == len(rows) and rows),
        "same_boundary_selector_established": False,
        "inflation_replacement_ready": False,
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "Finite-lattice audit for the OPH horizon-coherence condition. A theorem-grade low-k gap "
            "requires time-resolved harmonic repair rates Gamma_sigma(ell) over the CMB band and controls. "
            "If a run only contains final C_l fields plus a global Phi trace, this report emits a useful "
            "proxy but keeps low_k_gap_established false."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=CONTINUATION,
        receipt=COSMOLOGY_PERTURBATION_RECEIPT,
        physical_claim=False,
        observable_id="oph_low_k_synchronization_gap",
        fit_objective="mode_wise_repair_gap_control_audit",
    )


def write_synchronization_gap_report(
    run_dirs: list[Path],
    out_dir: Path,
    *,
    ell_max_cmb: int = 32,
    min_gamma_per_cycle: float = 1.0e-3,
    min_control_z: float = 1.0,
) -> dict[str, Any]:
    report = synchronization_gap_report(
        run_dirs,
        ell_max_cmb=int(ell_max_cmb),
        min_gamma_per_cycle=float(min_gamma_per_cycle),
        min_control_z=float(min_control_z),
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sync_gap_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "sync_gap_report.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_csv(out / "sync_gap_rows.csv", report["rows"])
    return report


def _run_row(
    run_path: Path,
    *,
    ell_max_cmb: int,
    min_gamma_per_cycle: float,
    min_control_z: float,
) -> dict[str, Any]:
    run_path = Path(run_path)
    trace = _read_trace(run_path / "mismatch_trace.csv")
    cl = _read_json(run_path / "cl_comparison_report.json")
    time_trace_path = run_path / "harmonic_time_trace.npz"
    phi_fit = _fit_decay(trace, "phi")
    time_resolved = time_trace_path.exists()
    time_gap = _time_resolved_gap(time_trace_path, ell_max_cmb) if time_resolved else {}
    field_rows = _field_gap_rows(cl, ell_max_cmb=ell_max_cmb)
    best_field = _best_field(field_rows)
    global_gamma = _float_or_none(phi_fit.get("gamma_per_cycle"))
    cached_proxy_pass = bool(
        global_gamma is not None
        and global_gamma >= float(min_gamma_per_cycle)
        and best_field
        and _float_or_none(best_field.get("low_ell_power_z_vs_controls")) is not None
        and abs(float(best_field["low_ell_power_z_vs_controls"])) >= float(min_control_z)
    )
    time_gap_pass = bool(time_gap.get("low_k_gap_established", False))
    return {
        "run_path": str(run_path),
        "has_gap_inputs": bool(trace or cl or time_resolved),
        "time_resolved_harmonic_trace_available": bool(time_resolved),
        "global_phi_decay": phi_fit,
        "global_phi_gamma_per_cycle": global_gamma,
        "field_count": len(field_rows),
        "best_cached_gap_field": best_field.get("field") if best_field else None,
        "best_cached_low_ell_power_z_vs_controls": best_field.get("low_ell_power_z_vs_controls") if best_field else None,
        "best_cached_low_ell_power_ratio_to_controls": best_field.get("low_ell_power_ratio_to_controls") if best_field else None,
        "cached_final_spectrum_gap_proxy_pass": cached_proxy_pass,
        "time_resolved_gap": time_gap,
        "time_resolved_gap_pass": time_gap_pass,
        "low_k_gap_established": bool(time_gap_pass),
        "field_rows": field_rows,
        "missing_gates": [
            name
            for name, passed in {
                "time_resolved_harmonic_trace": time_resolved,
                "mode_wise_gamma_positive": bool(time_gap.get("mode_wise_gamma_positive", False)),
                "controls_fail": bool(time_gap.get("controls_fail", False)),
                "same_boundary_selector": False,
            }.items()
            if not passed
        ],
        "claim_boundary": (
            "Cached-run final-spectrum proxy is diagnostic only. It cannot establish the OPH low-k "
            "synchronization gap without time-resolved harmonic repair rates."
        ),
    }


def _field_gap_rows(cl: dict[str, Any], *, ell_max_cmb: int) -> list[dict[str, Any]]:
    if not isinstance(cl, dict):
        return []
    rows: list[dict[str, Any]] = []
    fields = cl.get("fields", {}) or {}
    controls = cl.get("controls", {}) or {}
    for name, payload in fields.items():
        spectrum = payload.get("spectrum", []) if isinstance(payload, dict) else []
        low_power = _low_ell_power(spectrum, ell_max_cmb=ell_max_cmb)
        control_values = [
            _low_ell_power(control.get("spectrum", []), ell_max_cmb=ell_max_cmb)
            for control in (controls.get(name, {}) or {}).values()
            if isinstance(control, dict)
        ]
        control_values = [value for value in control_values if value is not None and math.isfinite(float(value))]
        mean = float(np.mean(control_values)) if control_values else None
        std = float(np.std(control_values, ddof=1)) if len(control_values) > 1 else 0.0
        z = (float(low_power) - mean) / std if low_power is not None and mean is not None and std > 0.0 else None
        rows.append(
            {
                "field": str(name),
                "low_ell_power_abs_sum": low_power,
                "control_low_ell_power_mean": mean,
                "control_low_ell_power_std": std if control_values else None,
                "low_ell_power_z_vs_controls": z,
                "low_ell_power_ratio_to_controls": float(low_power) / mean if low_power is not None and mean and abs(mean) > 1.0e-30 else None,
            }
        )
    return rows


def _time_resolved_gap(path: Path, ell_max_cmb: int) -> dict[str, Any]:
    try:
        payload = np.load(path, allow_pickle=False)
    except Exception as exc:  # pragma: no cover - defensive corrupted artifact path
        return {"available": False, "reason": f"failed_to_load: {exc}"}
    if "cycles" not in payload or "ell" not in payload:
        return {"available": False, "reason": "missing_cycles_or_ell"}
    cycles = np.asarray(payload["cycles"], dtype=float)
    ell = np.asarray(payload["ell"], dtype=float)
    all_field_names = [name for name in payload.files if name not in {"cycles", "ell"}]
    field_names = [name for name in all_field_names if not str(name).startswith("control__")]
    control_map: dict[str, list[tuple[str, str]]] = {}
    for name in all_field_names:
        text = str(name)
        if not text.startswith("control__"):
            continue
        parts = text.split("__", 2)
        if len(parts) != 3:
            continue
        _prefix, field_name, control_name = parts
        control_map.setdefault(field_name, []).append((control_name, text))
    rows = []
    for name in field_names:
        values = np.asarray(payload[name], dtype=float)
        if values.ndim != 2 or values.shape[0] != cycles.size or values.shape[1] != ell.size:
            continue
        mask = (ell >= 2.0) & (ell <= float(ell_max_cmb))
        for index in np.where(mask)[0]:
            fit = _fit_series_decay(cycles, np.abs(values[:, index]))
            control_fits = []
            for control_name, control_key in control_map.get(str(name), []):
                control_values = np.asarray(payload[control_key], dtype=float)
                if control_values.ndim != 2 or control_values.shape[0] != cycles.size or control_values.shape[1] != ell.size:
                    continue
                control_fit = _fit_series_decay(cycles, np.abs(control_values[:, index]))
                control_fits.append({"control": control_name, **control_fit})
            control_gammas = [
                float(row["gamma_per_cycle"])
                for row in control_fits
                if row.get("available") and row.get("gamma_per_cycle") is not None
            ]
            fit = {
                **fit,
                "control_count": len(control_fits),
                "max_control_gamma_per_cycle": max(control_gammas) if control_gammas else None,
                "control_fits": control_fits,
            }
            rows.append({"field": name, "ell": float(ell[index]), **fit})
    gamma_values = [float(row["gamma_per_cycle"]) for row in rows if row.get("available")]
    positive = [value for value in gamma_values if value > 0.0]
    rows_with_controls = [
        row
        for row in rows
        if row.get("available")
        and row.get("gamma_per_cycle") is not None
        and row.get("max_control_gamma_per_cycle") is not None
    ]
    control_separations = [
        float(row["gamma_per_cycle"]) - float(row["max_control_gamma_per_cycle"])
        for row in rows_with_controls
    ]
    controls_fail = bool(control_separations and min(control_separations) > 0.0)
    by_field = _time_resolved_by_field_summary(rows)
    return {
        "available": bool(rows),
        "mode_count": len(rows),
        "mode_count_with_controls": len(rows_with_controls),
        "median_gamma_per_cycle": median(gamma_values) if gamma_values else None,
        "min_gamma_per_cycle": min(gamma_values) if gamma_values else None,
        "positive_gamma_fraction": len(positive) / len(gamma_values) if gamma_values else None,
        "mode_wise_gamma_positive": bool(gamma_values and min(gamma_values) > 0.0),
        "median_target_minus_max_control_gamma": median(control_separations) if control_separations else None,
        "min_target_minus_max_control_gamma": min(control_separations) if control_separations else None,
        "by_field": by_field,
        "best_field_by_control_separation": _best_time_resolved_field(by_field, "median_target_minus_max_control_gamma"),
        "best_field_by_positive_gamma_fraction": _best_time_resolved_field(by_field, "positive_gamma_fraction"),
        "residual_field_gap_candidate": _residual_field_gap_candidate(by_field),
        "controls_fail": controls_fail,
        "low_k_gap_established": bool(gamma_values and min(gamma_values) > 0.0 and controls_fail),
        "rows": rows[:512],
        "claim_boundary": (
            "Time-resolved harmonic trace loaded. A low-k gap requires positive target decay across the "
            "band and target decay stronger than time-resolved controls."
        ),
    }


def _time_resolved_by_field_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not row.get("available"):
            continue
        grouped.setdefault(str(row.get("field")), []).append(row)
    summary: dict[str, dict[str, Any]] = {}
    for field, field_rows in sorted(grouped.items()):
        gammas = [
            float(row["gamma_per_cycle"])
            for row in field_rows
            if row.get("gamma_per_cycle") is not None
        ]
        separations = [
            float(row["gamma_per_cycle"]) - float(row["max_control_gamma_per_cycle"])
            for row in field_rows
            if row.get("gamma_per_cycle") is not None
            and row.get("max_control_gamma_per_cycle") is not None
        ]
        summary[field] = {
            "mode_count": len(field_rows),
            "median_gamma_per_cycle": median(gammas) if gammas else None,
            "min_gamma_per_cycle": min(gammas) if gammas else None,
            "positive_gamma_fraction": len([value for value in gammas if value > 0.0]) / len(gammas)
            if gammas
            else None,
            "median_target_minus_max_control_gamma": median(separations) if separations else None,
            "min_target_minus_max_control_gamma": min(separations) if separations else None,
            "control_separation_positive_fraction": len([value for value in separations if value > 0.0])
            / len(separations)
            if separations
            else None,
            "all_modes_positive": bool(gammas and min(gammas) > 0.0),
            "all_modes_beat_controls": bool(separations and min(separations) > 0.0),
        }
    return summary


def _best_time_resolved_field(summary: dict[str, dict[str, Any]], key: str) -> str | None:
    usable = [
        (field, values)
        for field, values in summary.items()
        if isinstance(values.get(key), (int, float))
    ]
    if not usable:
        return None
    return max(usable, key=lambda item: float(item[1][key]))[0]


def _residual_field_gap_candidate(summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    residual_names = ("repair_load", "local_mismatch_density")
    rows = {name: summary[name] for name in residual_names if name in summary}
    if not rows:
        return {"available": False}
    best_name = max(
        rows,
        key=lambda name: (
            float(rows[name].get("control_separation_positive_fraction") or 0.0),
            float(rows[name].get("positive_gamma_fraction") or 0.0),
            float(rows[name].get("median_target_minus_max_control_gamma") or -1.0e9),
        ),
    )
    best = rows[best_name]
    return {
        "available": True,
        "field": best_name,
        "positive_gamma_fraction": best.get("positive_gamma_fraction"),
        "control_separation_positive_fraction": best.get("control_separation_positive_fraction"),
        "median_gamma_per_cycle": best.get("median_gamma_per_cycle"),
        "median_target_minus_max_control_gamma": best.get("median_target_minus_max_control_gamma"),
        "candidate_receipt": bool(
            (best.get("positive_gamma_fraction") or 0.0) >= 0.5
            and (best.get("control_separation_positive_fraction") or 0.0) >= 0.5
            and (best.get("median_target_minus_max_control_gamma") or 0.0) > 0.0
        ),
        "claim_boundary": (
            "Residual-field candidate only. It reports whether repair/mismatch low-k modes show a majority "
            "positive decay and majority control separation. It is weaker than the all-mode low-k gap gate."
        ),
    }


def _best_field(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if _float_or_none(row.get("low_ell_power_z_vs_controls")) is not None]
    if not usable:
        return rows[0] if rows else {}
    return max(usable, key=lambda row: abs(float(row["low_ell_power_z_vs_controls"])))


def _low_ell_power(spectrum: list[dict[str, Any]], *, ell_max_cmb: int) -> float | None:
    values = []
    for row in spectrum:
        ell = _float_or_none(row.get("ell")) if isinstance(row, dict) else None
        dell = _float_or_none(row.get("D_ell")) if isinstance(row, dict) else None
        if ell is not None and dell is not None and 2.0 <= ell <= float(ell_max_cmb):
            values.append(abs(float(dell)))
    return float(np.sum(values)) if values else None


def _fit_decay(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    cycles = np.asarray([float(row.get("cycle", index)) for index, row in enumerate(rows)], dtype=float)
    values = np.asarray([float(row.get(key, 0.0)) for row in rows], dtype=float)
    return _fit_series_decay(cycles, values)


def _fit_series_decay(cycles: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(cycles) & np.isfinite(values) & (values > 0.0)
    if int(np.sum(mask)) < 2:
        return {"available": False, "reason": "not_enough_positive_values"}
    x = cycles[mask]
    y = np.log(np.maximum(values[mask], 1.0e-300))
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return {
        "available": True,
        "gamma_per_cycle": float(max(0.0, -slope)),
        "log_slope": float(slope),
        "fit_point_count": int(x.size),
        "fit_cycle_min": float(np.min(x)),
        "fit_cycle_max": float(np.max(x)),
        "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 1.0e-300 else 1.0,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gammas = [float(row["global_phi_gamma_per_cycle"]) for row in rows if row.get("global_phi_gamma_per_cycle") is not None]
    return {
        "run_count": len(rows),
        "time_resolved_trace_count": sum(1 for row in rows if row.get("time_resolved_harmonic_trace_available")),
        "cached_proxy_pass_count": sum(1 for row in rows if row.get("cached_final_spectrum_gap_proxy_pass")),
        "time_resolved_gap_pass_count": sum(1 for row in rows if row.get("time_resolved_gap_pass")),
        "mean_global_phi_gamma_per_cycle": fmean(gammas) if gammas else None,
        "median_global_phi_gamma_per_cycle": median(gammas) if gammas else None,
        "readiness_summary": _readiness_summary(rows),
    }


def _readiness_summary(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["no runs with synchronization-gap inputs found"]
    if not any(row.get("time_resolved_harmonic_trace_available") for row in rows):
        return [
            "cached runs expose global Phi decay and final C_l fields, but not time-resolved harmonic repair rates",
            "low-k synchronization gap is therefore not established yet",
        ]
    if all(row.get("time_resolved_gap_pass") for row in rows):
        return ["all selected runs pass the time-resolved low-k gap gate"]
    return ["time-resolved traces exist, but one or more low-k gap/control gates fail"]


def _find_run_dirs(roots: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        root = Path(root)
        if root.is_file():
            paths.add(root.parent)
        if (
            (root / "manifest.json").exists()
            or (root / "cl_comparison_report.json").exists()
            or (root / "harmonic_time_trace.npz").exists()
        ):
            paths.add(root)
        if root.exists():
            for name in ("manifest.json", "cl_comparison_report.json", "mismatch_trace.csv", "harmonic_time_trace.npz"):
                for path in root.glob(f"**/{name}"):
                    paths.add(path.parent)
    return sorted(paths, key=lambda path: str(path))


def _read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flat_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key not in {"field_rows", "time_resolved_gap", "global_phi_decay", "missing_gates"}}
        flat["missing_gates"] = ",".join(row.get("missing_gates", []))
        flat_rows.append(flat)
    keys = list(dict.fromkeys(key for row in flat_rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat_rows)


def _markdown_report(report: dict[str, Any]) -> str:
    agg = report.get("aggregate", {})
    lines = [
        "# OPH Low-k Synchronization Gap Audit",
        "",
        f"- Runs: {report.get('run_count')}",
        f"- Time-resolved traces: {agg.get('time_resolved_trace_count')}",
        f"- Cached proxy passes: {agg.get('cached_proxy_pass_count')}",
        f"- Time-resolved low-k gap passes: {agg.get('time_resolved_gap_pass_count')}",
        f"- Low-k gap established: `{str(report.get('low_k_gap_established')).lower()}`",
        "",
        "## Readiness",
    ]
    lines.extend(f"- {item}" for item in agg.get("readiness_summary", []))
    lines.extend(["", "## Claim Boundary", "", str(report.get("claim_boundary", "")), ""])
    return "\n".join(lines)
