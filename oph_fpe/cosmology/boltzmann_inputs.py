from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any


def oph_boltzmann_input_report(
    oph_cmb_reports: list[dict[str, Any]],
    *,
    camb_baseline_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build explicit Boltzmann-input readouts from current OPH-CMB diagnostics.

    This intentionally emits two different layers:

    * a CDM-limit table that is ready as a LambdaCDM regression target if the
      CAMB baseline receipt passed;
    * a repair-exchange diagnostic table that exposes current finite-collar
      proxy quantities but keeps the physical-prediction gate closed.
    """

    reports = [report for report in oph_cmb_reports if report]
    a_grid = _a_grid(reports)
    k_grid = _k_grid(reports)
    cdm_table = _cdm_limit_rows(a_grid, camb_baseline_report or {})
    diagnostic_table = _diagnostic_repair_rows(reports, a_grid)
    readiness = _readiness(reports, camb_baseline_report or {}, diagnostic_table)
    return {
        "mode": "oph_boltzmann_input_table_v0",
        "source_report_count": len(reports),
        "grids": {
            "a_grid": a_grid,
            "k_grid_h_mpc_required": k_grid,
            "k_proxy_source": "inverse_cap_opening_angle_from_finite_collar_samples",
        },
        "cdm_limit": {
            "status": "external_lambda_cdm_regression_target",
            "row_count": len(cdm_table),
            "rows": cdm_table,
            "claim_boundary": (
                "Pressureless conserved anomaly-stress limit. This is CAMB/LambdaCDM plumbing unless OPH "
                "independently supplies rho_A0."
            ),
        },
        "diagnostic_repair_exchange": {
            "status": "finite_collar_shape_proxy_not_physical_boltzmann_input",
            "row_count": len(diagnostic_table),
            "rows": diagnostic_table,
            "claim_boundary": (
                "Rows expose cap/collar repair-exchange proxy shapes only. B_A, Gamma_rec, and rho_A_eq "
                "are not theorem-grade physical inputs and must not be passed to a likelihood as predictions."
            ),
        },
        "readiness": readiness,
        "physical_cmb_prediction": False,
        "physical_matter_power_prediction": False,
        "claim_boundary": (
            "Machine-readable bridge from current OPH-CMB diagnostics toward a future CAMB/CLASS anomaly "
            "module. Only the CDM-limit rows are solver-ready as a standard regression target. The OPH "
            "repair-exchange rows are diagnostic finite-collar readouts, not physical CMB predictions."
        ),
    }


def write_oph_boltzmann_input_report(run_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    oph_reports: list[dict[str, Any]] = []
    camb_report: dict[str, Any] | None = None
    for root in _find_roots(run_dirs):
        for path in root.glob("**/oph_cmb_stress_report.json"):
            oph_reports.append(_read_json(path))
        for path in root.glob("**/camb_lcdm_baseline_report.json"):
            candidate = _read_json(path)
            if candidate:
                camb_report = candidate
    report = oph_boltzmann_input_report(oph_reports, camb_baseline_report=camb_report)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "oph_boltzmann_input_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "oph_boltzmann_input_report.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_csv(out_dir / "oph_boltzmann_cdm_limit_rows.csv", report["cdm_limit"]["rows"])
    _write_csv(out_dir / "oph_boltzmann_diagnostic_repair_rows.csv", report["diagnostic_repair_exchange"]["rows"])
    return report


def _cdm_limit_rows(a_grid: list[float], camb_report: dict[str, Any]) -> list[dict[str, Any]]:
    params = (camb_report.get("camb", {}) or {}).get("lambda_cdm_parameters", {}) or {}
    h = float(params.get("H0", 67.36)) / 100.0
    omega_c0 = float(params.get("omch2", 0.1200)) / max(h * h, 1.0e-12)
    omega_b0 = float(params.get("ombh2", 0.02237)) / max(h * h, 1.0e-12)
    rows = []
    for a in a_grid:
        a = float(a)
        rows.append(
            {
                "a": a,
                "z": float(1.0 / a - 1.0) if a > 0.0 else None,
                "Omega_A0_external_cdm_slot": omega_c0,
                "Omega_b0_external_baseline": omega_b0,
                "rho_A_over_rho_crit0": float(omega_c0 * a ** -3) if a > 0.0 else None,
                "w_A": 0.0,
                "c_s_A_squared": 0.0,
                "sigma_A": 0.0,
                "Gamma_rec_over_H": 0.0,
                "B_A": None,
                "source": "external_lambda_cdm_cdm_limit",
            }
        )
    return rows


def _diagnostic_repair_rows(reports: list[dict[str, Any]], a_grid: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report_index, report in enumerate(reports):
        parent = report.get("finite_collar_parent", {}) or {}
        kernel = report.get("diagnostic_kernel_proxy", {}) or {}
        parent_R = _float_or_none(parent.get("weighted_collar_repair_defect_R"))
        gamma_shape = _bounded_proxy(parent_R)
        rho_eq = _float_or_none(parent.get("rho_A_eq_proxy"))
        kernel_rows = list(kernel.get("kernel_proxy_rows", []) or [])
        for a in a_grid:
            for kernel_row in kernel_rows:
                rows.append(
                    {
                        "source_report_index": report_index,
                        "a": float(a),
                        "z": float(1.0 / float(a) - 1.0) if float(a) > 0.0 else None,
                        "k_proxy_inverse_theta": kernel_row.get("k_proxy_inverse_theta"),
                        "theta0": kernel_row.get("theta0"),
                        "rho_A_eq_proxy": rho_eq,
                        "rho_A_eq_proxy_units": parent.get("rho_A_eq_proxy_units"),
                        "weighted_collar_repair_defect_R": parent_R,
                        "Gamma_rec_over_H_shape_proxy": gamma_shape,
                        "B_A_shape_proxy": kernel_row.get("B_A_shape_proxy"),
                        "source": "finite_collar_diagnostic_shape_proxy",
                    }
                )
    return rows


def _readiness(
    reports: list[dict[str, Any]],
    camb_report: dict[str, Any],
    diagnostic_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    camb_receipt = bool(camb_report.get("CDM_LIMIT_BOLTZMANN_RECEIPT", False))
    checks = {
        "cdm_limit_regression_receipt": camb_receipt,
        "finite_collar_reports_available": bool(reports),
        "diagnostic_repair_rows_emitted": bool(diagnostic_rows),
        "finite_collar_parent_theorem_grade": False,
        "rho_A_a_physical_emitted": False,
        "rho_A_eq_a_physical_emitted": False,
        "Gamma_rec_k_a_physical_emitted": False,
        "B_A_k_a_physical_emitted": False,
        "gauge_consistency_audited": False,
        "full_likelihood_ready": False,
    }
    return {
        "cdm_limit_solver_ready": camb_receipt,
        "diagnostic_repair_exchange_table_ready": bool(reports and diagnostic_rows),
        "physical_prediction_ready": False,
        "checks": checks,
        "missing_gates": [name for name, passed in checks.items() if not passed],
    }


def _a_grid(reports: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for report in reports:
        values.extend(
            float(value)
            for value in (report.get("diagnostic_kernel_proxy", {}) or {}).get("a_grid", [])
            if isinstance(value, (int, float))
        )
    if not values:
        values = [1.0 / 1100.0, 0.01, 0.1, 1.0]
    return sorted(set(round(float(value), 12) for value in values))


def _k_grid(reports: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for report in reports:
        values.extend(
            float(value)
            for value in (report.get("diagnostic_kernel_proxy", {}) or {}).get(
                "k_grid_h_mpc_required_for_boltzmann", []
            )
            if isinstance(value, (int, float))
        )
    return sorted(set(round(float(value), 12) for value in values))


def _bounded_proxy(value: float | None) -> float | None:
    if value is None:
        return None
    value = max(float(value), 0.0)
    return float(value / (1.0 + value))


def _find_roots(run_dirs: list[Path]) -> list[Path]:
    return [Path(path) for path in run_dirs if Path(path).exists()]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(report: dict[str, Any]) -> str:
    ready = report["readiness"]
    cdm = report["cdm_limit"]
    diagnostic = report["diagnostic_repair_exchange"]
    diagnostic_rows = diagnostic["rows"]
    gamma_values = [
        row.get("Gamma_rec_over_H_shape_proxy")
        for row in diagnostic_rows
        if isinstance(row.get("Gamma_rec_over_H_shape_proxy"), (int, float))
    ]
    b_values = [
        row.get("B_A_shape_proxy")
        for row in diagnostic_rows
        if isinstance(row.get("B_A_shape_proxy"), (int, float))
    ]
    lines = [
        "# OPH Boltzmann Input Readout",
        "",
        report["claim_boundary"],
        "",
        "## Readiness",
        "",
        f"- CDM-limit solver ready: {ready['cdm_limit_solver_ready']}",
        f"- diagnostic repair-exchange table ready: {ready['diagnostic_repair_exchange_table_ready']}",
        f"- physical prediction ready: {ready['physical_prediction_ready']}",
        f"- missing gates: {', '.join(ready['missing_gates'])}",
        "",
        "## CDM Limit",
        "",
        f"- row count: {cdm['row_count']}",
        f"- status: {cdm['status']}",
        "",
        "## Diagnostic Repair Exchange",
        "",
        f"- row count: {diagnostic['row_count']}",
        f"- status: {diagnostic['status']}",
        f"- mean Gamma_rec/H shape proxy: {_fmt(fmean(gamma_values) if gamma_values else None)}",
        f"- mean B_A shape proxy: {_fmt(fmean(b_values) if b_values else None)}",
        "",
        "## Output Files",
        "",
        "- `oph_boltzmann_input_report.json`",
        "- `oph_boltzmann_cdm_limit_rows.csv`",
        "- `oph_boltzmann_diagnostic_repair_rows.csv`",
        "",
    ]
    return "\n".join(lines)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.10g}"
    return "n/a"
