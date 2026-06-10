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
    b_a_parent_reports: list[dict[str, Any]] | None = None,
    finite_transition_reports: list[dict[str, Any]] | None = None,
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
    b_a_parent_table = _b_a_parent_rows(b_a_parent_reports or [])
    finite_repair_clock_table = _finite_repair_clock_rows(finite_transition_reports or [], a_grid)
    readiness = _readiness(
        reports,
        camb_baseline_report or {},
        diagnostic_table,
        b_a_parent_table,
        finite_repair_clock_table,
    )
    return {
        "mode": "oph_boltzmann_input_table_v0",
        "source_report_count": len(reports),
        "b_a_parent_report_count": len([report for report in (b_a_parent_reports or []) if report]),
        "finite_transition_report_count": len([report for report in (finite_transition_reports or []) if report]),
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
        "b_a_parent_diagnostic": {
            "status": "finite_collar_report_backed_parent_diagnostic_not_physical_kernel",
            "row_count": len(b_a_parent_table),
            "rows": b_a_parent_table,
            "claim_boundary": (
                "Rows are copied from b_a_parent_report diagnostics. They exercise the B_A finite-difference "
                "contract but remain report-backed surrogate values until real baryon perturbation reruns, "
                "calibrated k/a units, energy-exchange closure, and refinement convergence pass."
            ),
        },
        "finite_repair_clock_diagnostic": {
            "status": "finite_transition_matrix_gamma_rec_diagnostic_not_certified_clock",
            "row_count": len(finite_repair_clock_table),
            "rows": finite_repair_clock_table,
            "claim_boundary": (
                "Rows expose Gamma_rec/H from finite observer-visible transition matrices. They are "
                "finite-lattice diagnostics, not physical recombination kernels, unless repair-step clock "
                "normalization is certified and energy-exchange closure is supplied."
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
    b_a_parent_reports: list[dict[str, Any]] = []
    finite_transition_reports: list[dict[str, Any]] = []
    camb_report: dict[str, Any] | None = None
    for root in _find_roots(run_dirs):
        for path in root.glob("**/oph_cmb_stress_report.json"):
            oph_reports.append(_read_json(path))
        for path in root.glob("**/b_a_parent_report.json"):
            b_a_parent_reports.append(_read_json(path))
        for path in root.glob("**/finite_repair_transition_matrix_report.json"):
            finite_transition_reports.append(_read_json(path))
        for path in root.glob("**/camb_lcdm_baseline_report.json"):
            candidate = _read_json(path)
            if candidate:
                camb_report = candidate
    report = oph_boltzmann_input_report(
        oph_reports,
        camb_baseline_report=camb_report,
        b_a_parent_reports=b_a_parent_reports,
        finite_transition_reports=finite_transition_reports,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "oph_boltzmann_input_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "oph_boltzmann_input_report.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_csv(out_dir / "oph_boltzmann_cdm_limit_rows.csv", report["cdm_limit"]["rows"])
    _write_csv(out_dir / "oph_boltzmann_diagnostic_repair_rows.csv", report["diagnostic_repair_exchange"]["rows"])
    _write_csv(out_dir / "oph_boltzmann_b_a_parent_rows.csv", report["b_a_parent_diagnostic"]["rows"])
    _write_csv(
        out_dir / "oph_boltzmann_finite_repair_clock_rows.csv",
        report["finite_repair_clock_diagnostic"]["rows"],
    )
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
    b_a_parent_rows: list[dict[str, Any]],
    finite_repair_clock_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    camb_receipt = bool(camb_report.get("CDM_LIMIT_BOLTZMANN_RECEIPT", False))
    clock_certified = any(bool(row.get("clock_normalization_certified", False)) for row in finite_repair_clock_rows)
    checks = {
        "cdm_limit_regression_receipt": camb_receipt,
        "finite_collar_reports_available": bool(reports),
        "diagnostic_repair_rows_emitted": bool(diagnostic_rows),
        "B_A_parent_diagnostic_rows_emitted": bool(b_a_parent_rows),
        "finite_repair_clock_diagnostic_rows_emitted": bool(finite_repair_clock_rows),
        "finite_transition_clock_certified": clock_certified,
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
        "B_A_parent_diagnostic_table_ready": bool(b_a_parent_rows),
        "finite_repair_clock_diagnostic_table_ready": bool(finite_repair_clock_rows),
        "physical_prediction_ready": False,
        "checks": checks,
        "missing_gates": [name for name, passed in checks.items() if not passed],
    }


def _b_a_parent_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report_index, report in enumerate(report for report in reports if report):
        for row in report.get("rows") or []:
            if not isinstance(row, dict):
                continue
            rows.append(
                {
                    "source_b_a_parent_report_index": report_index,
                    "a": row.get("a"),
                    "z": float(1.0 / float(row["a"]) - 1.0)
                    if isinstance(row.get("a"), (int, float)) and float(row["a"]) > 0.0
                    else None,
                    "k_proxy_inverse_theta": row.get("k_proxy_inverse_theta", row.get("k_h_mpc")),
                    "k_units": row.get("k_units", "inverse_cap_opening_angle_proxy"),
                    "B_A_parent_diagnostic": row.get("B_A_mean"),
                    "B_A_parent_sem": row.get("B_A_sem"),
                    "source": "b_a_parent_report_diagnostic",
                }
            )
    return rows


def _finite_repair_clock_rows(reports: list[dict[str, Any]], a_grid: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report_index, report in enumerate(report for report in reports if report):
        primary = report.get("primary", {}) or {}
        gamma = _float_or_none(primary.get("gamma_continuous"))
        lambda_2 = _float_or_none(primary.get("lambda_2"))
        kappa = _float_or_none(primary.get("kappa_rep_estimate"))
        eta = _float_or_none(primary.get("eta_R_estimate"))
        ns = _float_or_none(primary.get("n_s_estimate"))
        if gamma is None and lambda_2 is None and kappa is None and eta is None:
            continue
        for a in a_grid:
            a = float(a)
            rows.append(
                {
                    "source_transition_report_index": report_index,
                    "a": a,
                    "z": float(1.0 / a - 1.0) if a > 0.0 else None,
                    "k_proxy_inverse_theta": None,
                    "k_units": "scale_independent_transition_matrix_diagnostic",
                    "Gamma_rec_over_H_diagnostic": gamma,
                    "lambda_2": lambda_2,
                    "gamma_discrete_one_minus_lambda2": _float_or_none(
                        primary.get("gamma_discrete_one_minus_lambda2")
                    ),
                    "kappa_rep_estimate": kappa,
                    "eta_R_estimate": eta,
                    "n_s_estimate": ns,
                    "repair_step_time": _float_or_none(report.get("repair_step_time")),
                    "primary_matrix": report.get("primary_matrix"),
                    "state_count": report.get("state_count"),
                    "transition_count": report.get("transition_count"),
                    "finite_transition_matrix_ready": bool(report.get("finite_transition_matrix_ready", False)),
                    "finite_lattice_derived": bool(report.get("finite_lattice_derived", False)),
                    "clock_normalization_certified": bool(report.get("clock_normalization_certified", False)),
                    "repair_clock_certificate": bool(report.get("repair_clock_certificate", False)),
                    "physical_cmb_prediction": bool(report.get("physical_cmb_prediction", False)),
                    "source": "finite_repair_transition_matrix_diagnostic",
                }
            )
    return rows


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
    b_a_parent = report["b_a_parent_diagnostic"]
    finite_clock = report["finite_repair_clock_diagnostic"]
    diagnostic_rows = diagnostic["rows"]
    b_a_parent_rows = b_a_parent["rows"]
    finite_clock_rows = finite_clock["rows"]
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
    b_a_parent_values = [
        row.get("B_A_parent_diagnostic")
        for row in b_a_parent_rows
        if isinstance(row.get("B_A_parent_diagnostic"), (int, float))
    ]
    finite_clock_values = [
        row.get("Gamma_rec_over_H_diagnostic")
        for row in finite_clock_rows
        if isinstance(row.get("Gamma_rec_over_H_diagnostic"), (int, float))
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
        f"- B_A parent diagnostic table ready: {ready['B_A_parent_diagnostic_table_ready']}",
        f"- finite repair-clock diagnostic table ready: {ready['finite_repair_clock_diagnostic_table_ready']}",
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
        "## B_A Parent Diagnostic",
        "",
        f"- row count: {b_a_parent['row_count']}",
        f"- status: {b_a_parent['status']}",
        f"- mean B_A parent diagnostic: {_fmt(fmean(b_a_parent_values) if b_a_parent_values else None)}",
        "",
        "## Finite Repair-Clock Diagnostic",
        "",
        f"- row count: {finite_clock['row_count']}",
        f"- status: {finite_clock['status']}",
        f"- mean Gamma_rec/H diagnostic: {_fmt(fmean(finite_clock_values) if finite_clock_values else None)}",
        "",
        "## Output Files",
        "",
        "- `oph_boltzmann_input_report.json`",
        "- `oph_boltzmann_cdm_limit_rows.csv`",
        "- `oph_boltzmann_diagnostic_repair_rows.csv`",
        "- `oph_boltzmann_b_a_parent_rows.csv`",
        "- `oph_boltzmann_finite_repair_clock_rows.csv`",
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
