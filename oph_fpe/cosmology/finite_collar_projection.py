from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from oph_fpe.cosmology.claim_tiers import ClaimTier, GeometryOrigin


DEFAULT_CHI_STAR_MPC = 13_850.0
DEFAULT_H = 0.6736


def finite_collar_cmb_projection_report(
    bundle_report: dict[str, Any],
    *,
    chi_star_mpc: float = DEFAULT_CHI_STAR_MPC,
    h: float = DEFAULT_H,
    ell_mapping: str = "pi_over_theta",
) -> dict[str, Any]:
    """Project finite-collar source diagnostics onto conventional CMB axes.

    This is a comparison adapter. It gives the current finite-collar B_A rows
    conventional labels such as ell_eff and k_h/Mpc so we can inspect their
    large-scale shape against cosmology workflows. It deliberately does not
    mark the physical k/a calibration gate as passed: the mapping uses an
    externally declared fiducial last-scattering distance, not a finite OPH
    screen-to-bulk scale theorem.
    """

    chi = float(chi_star_mpc)
    hubble = float(h)
    if chi <= 0.0 or hubble <= 0.0:
        raise ValueError("chi_star_mpc and h must be positive")
    b_rows = [
        _project_b_row(row, chi_star_mpc=chi, h=hubble, ell_mapping=ell_mapping)
        for row in (bundle_report.get("B_A_k_a_diagnostic", {}) or {}).get("rows", [])
        if isinstance(row, dict)
    ]
    b_rows = [row for row in b_rows if row is not None]
    rho_rows = _background_rows(
        (bundle_report.get("rho_A_a_diagnostic", {}) or {}).get("rows", []),
        (bundle_report.get("Gamma_rec_k_a_diagnostic", {}) or {}).get("rows", []),
    )
    shape = _shape_summary(b_rows)
    readiness = {
        "finite_collar_source_bundle_receipt": bool(
            bundle_report.get("FINITE_COLLAR_BOLTZMANN_SOURCE_BUNDLE_RECEIPT", False)
        ),
        "external_fiducial_geometry_declared": True,
        "ell_k_axes_emitted": bool(b_rows),
        "rho_A_background_rows_emitted": bool(rho_rows),
        "physical_k_units_calibrated": False,
        "finite_screen_to_bulk_scale_theorem_used": False,
        "official_likelihood_ready": False,
    }
    return {
        "mode": "finite_collar_cmb_projection_diagnostic_v0",
        "claim_tier": ClaimTier.DIAGNOSTIC_PROXY.value,
        "geometry_origin": GeometryOrigin.EXTERNAL_FIDUCIAL.value,
        "projection": {
            "chi_star_mpc": chi,
            "h": hubble,
            "ell_mapping": str(ell_mapping),
            "k_mpc_formula": "(ell_eff + 0.5) / chi_star_mpc",
            "k_h_mpc_formula": "k_mpc / h",
            "calibration_source": "external_fiducial_last_scattering_distance_for_comparison_only",
            "measurement_data_used_for_finite_source_functions": False,
            "physical_k_calibration": False,
            "FIDUCIAL_CMB_AXIS_PROJECTION_DIAGNOSTIC": True,
        },
        "projected_B_A_rows": b_rows,
        "background_rows": rho_rows,
        "shape_summary": shape,
        "readiness": readiness,
        "FINITE_COLLAR_CMB_PROJECTION_DIAGNOSTIC_RECEIPT": bool(
            readiness["finite_collar_source_bundle_receipt"] and readiness["ell_k_axes_emitted"]
        ),
        "PHYSICAL_K_CALIBRATION_RECEIPT": False,
        "PHYSICAL_K_RECEIPT": False,
        "FIDUCIAL_CMB_AXIS_PROJECTION_DIAGNOSTIC": True,
        "physical_cmb_prediction": False,
        "physical_matter_power_prediction": False,
        "claim_boundary": (
            "External-fiducial projection of finite-collar source diagnostics onto CMB ell/k labels. "
            "This makes the current simulator data easier to compare and pass into plumbing tests, "
            "but it is not a physical OPH k-calibration, not a Boltzmann likelihood, and not a CMB prediction."
        ),
    }


def write_finite_collar_cmb_projection_report(
    run_dirs: list[Path],
    out_dir: Path,
    *,
    chi_star_mpc: float = DEFAULT_CHI_STAR_MPC,
    h: float = DEFAULT_H,
    ell_mapping: str = "pi_over_theta",
) -> dict[str, Any]:
    bundle = _first_json(run_dirs, "finite_collar_boltzmann_bundle_report.json")
    if not bundle:
        raise FileNotFoundError("missing finite_collar_boltzmann_bundle_report.json in run dirs")
    report = finite_collar_cmb_projection_report(
        bundle,
        chi_star_mpc=chi_star_mpc,
        h=h,
        ell_mapping=ell_mapping,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "finite_collar_cmb_projection_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "finite_collar_cmb_projection_report.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_csv(out / "finite_collar_projected_B_A_rows.csv", report["projected_B_A_rows"])
    _write_csv(out / "finite_collar_projected_background_rows.csv", report["background_rows"])
    return report


def _project_b_row(
    row: dict[str, Any],
    *,
    chi_star_mpc: float,
    h: float,
    ell_mapping: str,
) -> dict[str, Any] | None:
    b_a = _float(row.get("B_A"))
    a = _float(row.get("a"))
    if b_a is None or a is None:
        return None
    theta = _float(row.get("theta0"))
    k_proxy = _float(row.get("k"))
    if theta is None and k_proxy is not None and str(row.get("k_units")) == "inverse_cap_opening_angle_proxy":
        theta = 1.0 / max(abs(k_proxy), 1.0e-300)
    ell_eff = _ell_eff(theta, k_proxy, ell_mapping)
    if ell_eff is None:
        return None
    k_mpc = (ell_eff + 0.5) / float(chi_star_mpc)
    return {
        "a": a,
        "z": (1.0 / a - 1.0) if a > 0.0 else None,
        "ell_eff": ell_eff,
        "k_Mpc^-1": k_mpc,
        "k_h_Mpc^-1": k_mpc / float(h),
        "B_A": b_a,
        "B_A_sem": _float(row.get("B_A_sem")),
        "theta0": theta,
        "original_k_proxy": k_proxy,
        "original_k_units": row.get("k_units"),
        "cap_index": row.get("cap_index"),
        "time": row.get("time"),
        "source_report_index": row.get("source_report_index"),
        "source": row.get("source"),
        "physical_kernel": False,
    }


def _ell_eff(theta: float | None, k_proxy: float | None, ell_mapping: str) -> float | None:
    if theta is not None and theta > 0.0:
        if ell_mapping == "pi_over_theta":
            return float(math.pi / theta)
        if ell_mapping == "one_over_theta":
            return float(1.0 / theta)
    if k_proxy is not None:
        return float(math.pi * k_proxy if ell_mapping == "pi_over_theta" else k_proxy)
    return None


def _background_rows(rho_rows: Any, gamma_rows: Any) -> list[dict[str, Any]]:
    grouped: dict[float, dict[str, list[float]]] = {}
    for row in rho_rows or []:
        if not isinstance(row, dict):
            continue
        a = _float(row.get("a"))
        if a is None:
            continue
        item = grouped.setdefault(round(a, 12), {"rho_A": [], "rho_A_eq": [], "Gamma_rec_over_H": []})
        for key in ("rho_A", "rho_A_eq"):
            value = _float(row.get(key))
            if value is not None:
                item[key].append(value)
    for row in gamma_rows or []:
        if not isinstance(row, dict):
            continue
        a = _float(row.get("a"))
        gamma = _float(row.get("Gamma_rec_over_H"))
        if a is None or gamma is None:
            continue
        item = grouped.setdefault(round(a, 12), {"rho_A": [], "rho_A_eq": [], "Gamma_rec_over_H": []})
        item["Gamma_rec_over_H"].append(gamma)
    out = []
    for a_key, values in sorted(grouped.items()):
        a = float(a_key)
        rho = _mean(values["rho_A"])
        rho_eq = _mean(values["rho_A_eq"])
        out.append(
            {
                "a": a,
                "z": (1.0 / a - 1.0) if a > 0.0 else None,
                "rho_A_mean": rho,
                "rho_A_eq_mean": rho_eq,
                "q_A_eq_over_A": (rho_eq / rho) if rho and rho_eq is not None else None,
                "Gamma_rec_over_H_mean": _mean(values["Gamma_rec_over_H"]),
                "physical_background": False,
            }
        )
    return out


def _shape_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([float(row["B_A"]) for row in rows if _float(row.get("B_A")) is not None], dtype=float)
    ells = np.asarray([float(row["ell_eff"]) for row in rows if _float(row.get("B_A")) is not None], dtype=float)
    abs_values = np.abs(values)
    slope = None
    if values.size >= 3 and np.all(ells > 0.0) and np.any(abs_values > 0.0):
        mask = abs_values > 0.0
        if int(np.sum(mask)) >= 3:
            slope = float(np.polyfit(np.log(ells[mask]), np.log(abs_values[mask]), 1)[0])
    return {
        "row_count": int(values.size),
        "ell_min": float(np.min(ells)) if ells.size else None,
        "ell_max": float(np.max(ells)) if ells.size else None,
        "mean_B_A": float(np.mean(values)) if values.size else None,
        "mean_abs_B_A": float(np.mean(abs_values)) if values.size else None,
        "positive_fraction": float(np.mean(values > 0.0)) if values.size else None,
        "log_abs_B_A_vs_log_ell_slope": slope,
        "largest_scale_mean_B_A": _bin_mean(rows, high_ell=False),
        "smallest_scale_mean_B_A": _bin_mean(rows, high_ell=True),
    }


def _bin_mean(rows: list[dict[str, Any]], *, high_ell: bool) -> float | None:
    if not rows:
        return None
    ordered = sorted(rows, key=lambda row: float(row["ell_eff"]))
    count = max(1, len(ordered) // 4)
    selected = ordered[-count:] if high_ell else ordered[:count]
    return _mean([_float(row.get("B_A")) for row in selected])


def _first_json(roots: list[Path], name: str) -> dict[str, Any]:
    for root in roots:
        root = Path(root)
        candidates = [root / name]
        if root.exists() and root.is_dir():
            candidates.extend(sorted(root.glob(f"**/{name}")))
        for path in candidates:
            if not path.exists() or not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
    return {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(report: dict[str, Any]) -> str:
    shape = report["shape_summary"]
    readiness = report["readiness"]
    lines = [
        "# Finite-Collar CMB Projection Diagnostic",
        "",
        report["claim_boundary"],
        "",
        "## Projection",
        "",
        f"- chi_star_mpc: {report['projection']['chi_star_mpc']}",
        f"- h: {report['projection']['h']}",
        f"- ell mapping: `{report['projection']['ell_mapping']}`",
        f"- physical k calibration: `{str(report['projection']['physical_k_calibration']).lower()}`",
        "",
        "## Shape",
        "",
        f"- rows: {shape['row_count']}",
        f"- ell range: {shape['ell_min']} .. {shape['ell_max']}",
        f"- mean B_A: {shape['mean_B_A']}",
        f"- mean |B_A|: {shape['mean_abs_B_A']}",
        f"- positive fraction: {shape['positive_fraction']}",
        f"- log |B_A| / log ell slope: {shape['log_abs_B_A_vs_log_ell_slope']}",
        f"- largest-scale mean B_A: {shape['largest_scale_mean_B_A']}",
        f"- smallest-scale mean B_A: {shape['smallest_scale_mean_B_A']}",
        "",
        "## Readiness",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in readiness.items())
    lines.append("")
    return "\n".join(lines)


def _mean(values: Any) -> float | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(fmean(clean)) if clean else None


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None
