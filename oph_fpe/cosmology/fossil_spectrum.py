from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.cosmology.oph_screen_power import PLANCK_ETA_R_SIGMA, PLANCK_ETA_R_TARGET, screen_power_fit_from_spectrum


def write_fossil_spectrum_report(
    run_dir: Path,
    out_dir: Path,
    *,
    ell_min: float = 8.0,
    ell_max: float | None = 32.0,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """Audit time-resolved screen spectra without promoting a target-selected cycle.

    This report is intended to answer whether scale-invariant-looking spectra occur
    during repair/record formation, while keeping the physical freezeout markers and
    the target-closeness scan separate.
    """

    run = Path(run_dir)
    trace_path = run / "harmonic_time_trace.npz"
    if not trace_path.exists():
        raise FileNotFoundError(f"missing harmonic trace: {trace_path}")

    with np.load(trace_path) as trace:
        cycles = np.asarray(trace["cycles"], dtype=int)
        ell = np.asarray(trace["ell"], dtype=float)
        field_names = fields or [
            name for name in trace.files if name not in {"cycles", "ell"} and not name.startswith("control__")
        ]
        rows = []
        for field in field_names:
            if field not in trace.files:
                continue
            rows.extend(
                _fit_trace_field(
                    cycles,
                    ell,
                    np.asarray(trace[field], dtype=float),
                    field_name=field,
                    point_count=_point_count(run),
                    ell_min=ell_min,
                    ell_max=ell_max,
                    kind="field",
                )
            )
            for control_name in sorted(name for name in trace.files if name.startswith(f"control__{field}__")):
                rows.extend(
                    _fit_trace_field(
                        cycles,
                        ell,
                        np.asarray(trace[control_name], dtype=float),
                        field_name=field,
                        point_count=_point_count(run),
                        ell_min=ell_min,
                        ell_max=ell_max,
                        kind="control",
                        control_name=control_name,
                    )
                )

    marker_rows = _marker_rows(rows, _cycle_markers(run))
    scan_rows = [row for row in rows if row["kind"] == "field" and row.get("fit_available")]
    best = min(scan_rows, key=lambda row: abs(float(row["eta_R"]) - PLANCK_ETA_R_TARGET), default=None)
    best_controls = _controls_for(rows, best) if best else []
    best_control_delta = min(
        (abs(float(row["eta_R"]) - PLANCK_ETA_R_TARGET) for row in best_controls if row.get("fit_available")),
        default=None,
    )
    best_delta = abs(float(best["eta_R"]) - PLANCK_ETA_R_TARGET) if best else None
    report = {
        "mode": "oph_fossil_spectrum_time_resolved_diagnostic_v0",
        "run_dir": str(run),
        "trace_path": str(trace_path),
        "ell_min": float(ell_min),
        "ell_max": None if ell_max is None else float(ell_max),
        "field_count": len({row["field"] for row in rows if row["kind"] == "field"}),
        "cycle_count": len({int(row["cycle"]) for row in rows}),
        "fit_row_count": len([row for row in rows if row.get("fit_available")]),
        "cycle_markers": _cycle_markers(run),
        "marker_rows": marker_rows,
        "best_target_closeness_diagnostic": best,
        "best_same_field_control_delta_to_planck": best_control_delta,
        "best_beats_same_field_controls": bool(
            best is not None and best_control_delta is not None and best_delta is not None and best_delta < best_control_delta
        ),
        "near_scale_invariant_transient": bool(best is not None and best_delta is not None and best_delta <= PLANCK_ETA_R_SIGMA),
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "Time-resolved finite-screen fossil diagnostic. Cycle markers are objective run events; the "
            "best target-closeness row is explicitly a diagnostic selector and must not be used as a "
            "physical CMB prediction unless a paper-derived freezeout rule selects that cycle before "
            "measurement comparison and controls/refinement pass."
        ),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fossil_spectrum_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _write_csv(out / "fossil_spectrum_rows.csv", rows)
    _write_csv(out / "fossil_spectrum_marker_rows.csv", marker_rows)
    return report


def _fit_trace_field(
    cycles: np.ndarray,
    ell: np.ndarray,
    spectra: np.ndarray,
    *,
    field_name: str,
    point_count: int | None,
    ell_min: float,
    ell_max: float | None,
    kind: str,
    control_name: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, cycle in enumerate(cycles):
        spectrum = [
            {"ell": float(ell_value), "D_ell": float(d_ell), "C_ell": 0.0}
            for ell_value, d_ell in zip(ell, spectra[index], strict=True)
        ]
        fit = screen_power_fit_from_spectrum(
            spectrum,
            field_name=field_name,
            point_count=point_count,
            ell_min=ell_min,
            ell_max=ell_max,
        )
        row = {
            "kind": kind,
            "field": field_name,
            "control_name": control_name,
            "cycle": int(cycle),
            "fit_available": bool(fit.get("fit_available", False)),
            "eta_R": fit.get("eta_R_estimate"),
            "n_s": fit.get("n_s_proxy"),
            "eta_R_delta_to_planck": (
                float(fit["eta_R_estimate"]) - PLANCK_ETA_R_TARGET if fit.get("fit_available") else None
            ),
            "abs_eta_R_delta_to_planck": (
                abs(float(fit["eta_R_estimate"]) - PLANCK_ETA_R_TARGET) if fit.get("fit_available") else None
            ),
            "within_planck_eta_R_1sigma": bool(fit.get("within_planck_eta_R_1sigma", False)),
            "q_IR_proxy": (fit.get("low_ell_suppression") or {}).get("q_IR_proxy"),
            "ell_IR_proxy": (fit.get("low_ell_suppression") or {}).get("ell_IR_proxy"),
        }
        rows.append(row)
    return rows


def _cycle_markers(run: Path) -> dict[str, int | None]:
    mismatch = _read_mismatch_trace(run / "mismatch_trace.csv")
    freezeout = _read_json(run / "freezeout_map_summary.json")
    markers: dict[str, int | None] = {
        "freezeout_cycle": _int_or_none(freezeout.get("freezeout_cycle")),
        "phi_zero_cycle": None,
        "phi_half_cycle": None,
        "committed_fraction_50_cycle": None,
        "committed_fraction_90_cycle": None,
        "max_record_growth_cycle": None,
    }
    if not mismatch:
        return markers
    initial_phi = float(mismatch[0].get("phi_before") or mismatch[0].get("phi") or 0.0)
    previous_committed = None
    best_growth = (-math.inf, None)
    for row in mismatch:
        cycle = int(float(row.get("cycle", 0)))
        phi = float(row.get("phi", 0.0))
        committed_fraction = float(row.get("committed_fraction", 0.0))
        committed = float(row.get("committed_records", 0.0))
        if markers["phi_zero_cycle"] is None and phi <= 0.0:
            markers["phi_zero_cycle"] = cycle
        if markers["phi_half_cycle"] is None and initial_phi > 0.0 and phi <= initial_phi / 2.0:
            markers["phi_half_cycle"] = cycle
        if markers["committed_fraction_50_cycle"] is None and committed_fraction >= 0.5:
            markers["committed_fraction_50_cycle"] = cycle
        if markers["committed_fraction_90_cycle"] is None and committed_fraction >= 0.9:
            markers["committed_fraction_90_cycle"] = cycle
        if previous_committed is not None:
            growth = committed - previous_committed
            if growth > best_growth[0]:
                best_growth = (growth, cycle)
        previous_committed = committed
    markers["max_record_growth_cycle"] = best_growth[1]
    return markers


def _marker_rows(rows: list[dict[str, Any]], markers: dict[str, int | None]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        if row["kind"] != "field" or not row.get("fit_available"):
            continue
        by_key[(str(row["field"]), str(row["kind"]), int(row["cycle"]))] = row
    out = []
    fields = sorted({row["field"] for row in rows if row["kind"] == "field"})
    cycles = sorted({int(row["cycle"]) for row in rows})
    for marker, cycle in markers.items():
        if cycle is None or not cycles:
            continue
        nearest = min(cycles, key=lambda item: abs(item - int(cycle)))
        for field in fields:
            source = by_key.get((field, "field", nearest))
            if source:
                out.append({"marker": marker, "declared_cycle": int(cycle), "nearest_trace_cycle": nearest, **source})
    return out


def _controls_for(rows: list[dict[str, Any]], best: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not best:
        return []
    field = best["field"]
    cycle = int(best["cycle"])
    return [row for row in rows if row["kind"] == "control" and row["field"] == field and int(row["cycle"]) == cycle]


def _point_count(run: Path) -> int | None:
    manifest = _read_json(run / "manifest.json")
    for key in ("patch_count", "point_count"):
        value = manifest.get(key)
        if value:
            return int(value)
    cl_report = _read_json(run / "cl_comparison_report.json")
    return _int_or_none(cl_report.get("point_count"))


def _read_mismatch_trace(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row if not isinstance(row.get(key), (dict, list))})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
