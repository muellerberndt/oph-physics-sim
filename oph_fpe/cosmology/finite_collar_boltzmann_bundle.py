from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from oph_fpe.cosmology.physical_cmb_contract import validate_physical_cmb_contract
from oph_fpe.cosmology.physical_cmb_prediction import build_physical_cmb_input_contract


def finite_collar_boltzmann_bundle_report(run_dirs: list[Path]) -> dict[str, Any]:
    """Assemble the finite-collar source functions required by the CMB bridge.

    This report is intentionally a source-side bundle, not a likelihood result.
    The current paper/cosmology stack asks the simulator to emit the parent
    finite-collar quantities before any CMB/BAO/weak-lensing data are allowed to
    influence them:

    * rho_A(a)
    * rho_A,eq(a)
    * B_A(k,a)
    * Gamma_rec(k,a)

    Existing OPH-FPE reports often expose these as scattered diagnostics. This
    module consolidates them into auditable tables and mirrors the hard physical
    CMB input contract so downstream tooling can distinguish "finite diagnostic
    exists" from "physical Boltzmann prediction is licensed".
    """

    roots = _find_roots(run_dirs)
    contract, contract_sources = build_physical_cmb_input_contract(roots)
    validation = validate_physical_cmb_contract(contract)
    b_a_reports = _collect_json(roots, ("b_a_parent_report.json", "paired_b_a_perturbation_report.json"))
    transition_reports = _collect_json(roots, ("finite_repair_transition_matrix_report.json",))
    no_data_receipts = _collect_json(roots, ("no_data_use_receipt.json",))

    b_a_rows = _b_a_rows(b_a_reports)
    rho_rows = _rho_rows(b_a_reports)
    gamma_rows = _gamma_rows(transition_reports, _a_grid_from_rows(b_a_rows, rho_rows))
    source_audit = _source_audit(b_a_reports, transition_reports, no_data_receipts)
    readiness = _readiness(
        b_a_reports=b_a_reports,
        transition_reports=transition_reports,
        no_data_receipts=no_data_receipts,
        b_a_rows=b_a_rows,
        rho_rows=rho_rows,
        gamma_rows=gamma_rows,
        contract_validation=validation,
    )
    return {
        "mode": "finite_collar_boltzmann_source_bundle_v0",
        "run_dirs": [str(path) for path in roots],
        "source_report_counts": {
            "b_a_parent": len(b_a_reports),
            "finite_transition": len(transition_reports),
            "no_data_use_receipt": len(no_data_receipts),
        },
        "source_audit": source_audit,
        "B_A_k_a_diagnostic": {
            "row_count": len(b_a_rows),
            "rows": b_a_rows,
            "source": "finite_collar_parent_diagnostic",
            "physical_kernel": False,
        },
        "rho_A_a_diagnostic": {
            "row_count": len(rho_rows),
            "rows": rho_rows,
            "source": "finite_collar_parent_diagnostic",
            "physical_background": False,
        },
        "Gamma_rec_k_a_diagnostic": {
            "row_count": len(gamma_rows),
            "rows": gamma_rows,
            "source": "finite_repair_transition_matrix_diagnostic",
            "physical_clock": False,
        },
        "physical_cmb_input_contract": _contract_to_jsonable(contract),
        "physical_cmb_input_validation": validation,
        "readiness": readiness,
        "FINITE_COLLAR_BOLTZMANN_SOURCE_BUNDLE_RECEIPT": readiness[
            "FINITE_COLLAR_BOLTZMANN_SOURCE_BUNDLE_RECEIPT"
        ],
        "PHYSICAL_BOLTZMANN_EXPORT_CERTIFICATE": readiness["PHYSICAL_BOLTZMANN_EXPORT_CERTIFICATE"],
        "physical_cmb_prediction": False,
        "physical_matter_power_prediction": False,
        "contract_source_summary": {
            name: {"present": bool(report), "mode": report.get("mode")}
            for name, report in contract_sources.items()
        },
        "claim_boundary": (
            "Finite-collar Boltzmann source bundle. The tables are source-side OPH-FPE diagnostics "
            "for rho_A(a), rho_A,eq(a), B_A(k,a), and Gamma_rec(k,a). They are not physical CMB "
            "or matter-power predictions until physical k/a calibration, energy-exchange closure, "
            "refinement convergence, strict freezeout/bulk gates, CDM-limit regression, and an official "
            "likelihood-ready path all pass."
        ),
    }


def write_finite_collar_boltzmann_bundle_report(run_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    report = finite_collar_boltzmann_bundle_report(run_dirs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "finite_collar_boltzmann_bundle_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "finite_collar_boltzmann_bundle_report.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_csv(out / "finite_collar_B_A_k_a_diagnostic.csv", report["B_A_k_a_diagnostic"]["rows"])
    _write_csv(out / "finite_collar_rho_A_a_diagnostic.csv", report["rho_A_a_diagnostic"]["rows"])
    _write_csv(out / "finite_collar_Gamma_rec_k_a_diagnostic.csv", report["Gamma_rec_k_a_diagnostic"]["rows"])
    return report


def _b_a_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report_index, report in enumerate(reports):
        for row in report.get("rows") or report.get("observer_view_rows") or []:
            if not isinstance(row, dict):
                continue
            b_a = _float(row.get("B_A_mean", row.get("B_A_parent_diagnostic")))
            k_value = _float(row.get("k_h_mpc", row.get("k_proxy_inverse_theta")))
            a_value = _float(row.get("a"))
            if b_a is None or k_value is None or a_value is None:
                continue
            rows.append(
                {
                    "source_report_index": report_index,
                    "a": a_value,
                    "z": (1.0 / a_value - 1.0) if a_value > 0.0 else None,
                    "k": k_value,
                    "k_units": row.get("k_units", "inverse_cap_opening_angle_proxy"),
                    "k_proxy_inverse_theta": k_value
                    if str(row.get("k_units", "inverse_cap_opening_angle_proxy")) != "Mpc^-1"
                    else None,
                    "k_comoving_Mpc_inverse": k_value
                    if str(row.get("k_units", "inverse_cap_opening_angle_proxy")) == "Mpc^-1"
                    else None,
                    "physical_calibration": str(row.get("k_units", "")) == "Mpc^-1"
                    and bool(row.get("physical_calibration", False)),
                    "B_A": b_a,
                    "B_A_sem": _float(row.get("B_A_sem")),
                    "theta0": _float(row.get("theta0")),
                    "cap_index": row.get("cap_index"),
                    "time": _float(row.get("time")),
                    "control": row.get("control"),
                    "source": row.get("source", report.get("primary_parent_source", report.get("mode"))),
                    "physical_kernel": False,
                }
            )
    return rows


def _rho_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report_index, report in enumerate(reports):
        for row in report.get("rows") or report.get("observer_view_rows") or []:
            if not isinstance(row, dict):
                continue
            a_value = _float(row.get("a"))
            if a_value is None:
                continue
            rho_a = _float(row.get("rho_A", row.get("rho_A_base")))
            cmi_diagnostic = _float(row.get("base_epsilon_cmi", row.get("cmi_diagnostic_nats")))
            rho_eq_plus = _float(row.get("rho_A_eq_plus_mean"))
            rho_eq_minus = _float(row.get("rho_A_eq_minus_mean"))
            rho_eq = _mean_optional([rho_eq_plus, rho_eq_minus])
            if rho_a is None and rho_eq is None:
                continue
            rows.append(
                {
                    "source_report_index": report_index,
                    "a": a_value,
                    "z": (1.0 / a_value - 1.0) if a_value > 0.0 else None,
                    "rho_A": rho_a if rho_a is not None else None,
                    "rho_A_eq": rho_eq if rho_eq is not None else None,
                    "rho_A_eq_plus": rho_eq_plus,
                    "rho_A_eq_minus": rho_eq_minus,
                    "cmi_diagnostic_nats": cmi_diagnostic,
                    "rho_units": "finite_screen_response_units",
                    "control": row.get("control"),
                    "source": row.get("source", report.get("primary_parent_source", report.get("mode"))),
                    "physical_background": False,
                }
            )
    return rows


def _gamma_rows(reports: list[dict[str, Any]], a_grid: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not a_grid:
        a_grid = [1.0 / 1100.0, 0.01, 0.1, 1.0]
    for report_index, report in enumerate(reports):
        primary = report.get("primary") or {}
        gamma = _float(primary.get("gamma_continuous"))
        if gamma is None:
            continue
        for a_value in a_grid:
            rows.append(
                {
                    "source_report_index": report_index,
                    "a": float(a_value),
                    "z": (1.0 / float(a_value) - 1.0) if float(a_value) > 0.0 else None,
                    "k": None,
                    "k_units": "scale_independent_transition_matrix_diagnostic",
                    "Gamma_rec_over_H": gamma,
                    "lambda_2": _float(primary.get("lambda_2")),
                    "eta_R_estimate": _float(primary.get("eta_R_estimate")),
                    "n_s_estimate": _float(primary.get("n_s_estimate")),
                    "clock_normalization_certified": bool(report.get("clock_normalization_certified", False)),
                    "finite_transition_matrix_ready": bool(report.get("finite_transition_matrix_ready", False)),
                    "physical_clock": False,
                }
            )
    return rows


def _readiness(
    *,
    b_a_reports: list[dict[str, Any]],
    transition_reports: list[dict[str, Any]],
    no_data_receipts: list[dict[str, Any]],
    b_a_rows: list[dict[str, Any]],
    rho_rows: list[dict[str, Any]],
    gamma_rows: list[dict[str, Any]],
    contract_validation: dict[str, Any],
) -> dict[str, Any]:
    no_data = _no_data_use_ok(no_data_receipts, b_a_reports, transition_reports)
    contract_passed = bool(contract_validation.get("PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT", False))
    bridge = contract_validation.get("physical_scale_bridge") or {}
    blockers = set(str(blocker) for blocker in (contract_validation.get("blockers") or []))
    b_a_diagnostic = any(
        bool(
            report.get("B_A_PAIRED_DIAGNOSTIC_RECEIPT")
            or (report.get("readiness", {}) or {}).get("B_A_PAIRED_DIAGNOSTIC_RECEIPT")
            or ((report.get("readiness", {}) or {}).get("checks", {}) or {}).get("paired_B_A_diagnostic_receipt")
        )
        for report in b_a_reports
    )
    controls_fail = any(
        bool(((report.get("readiness", {}) or {}).get("checks", {}) or {}).get("controls_fail"))
        for report in b_a_reports
    )
    checks = {
        "no_data_use_receipt": no_data,
        "B_A_rows_emitted": bool(b_a_rows),
        "rho_A_rows_emitted": any(row.get("rho_A") is not None for row in rho_rows),
        "rho_A_eq_rows_emitted": any(row.get("rho_A_eq") is not None for row in rho_rows),
        "Gamma_rec_rows_emitted": bool(gamma_rows),
        "paired_B_A_diagnostic_receipt": bool(b_a_diagnostic),
        "B_A_controls_fail": bool(controls_fail),
        "finite_transition_matrix_ready": any(
            bool(report.get("finite_transition_matrix_ready", False)) for report in transition_reports
        ),
        "physical_k_units_calibrated": all(
            bool(row.get("physical_calibration"))
            and str(row.get("k_units")) == "Mpc^-1"
            and row.get("k_comoving_Mpc_inverse") is not None
            for row in b_a_rows
        )
        if b_a_rows and bridge.get("PHYSICAL_K_RECEIPT", False)
        else False,
        "calibrated_a_evolution": bool(bridge.get("CALIBRATED_A_EVOLUTION_RECEIPT", False)),
        "energy_momentum_exchange_closed": "stress_energy_closure_not_certified" not in blockers
        and "recipient_stress_missing_for_nonzero_Gamma_rec" not in blockers,
        "gauge_consistency_audited": "gauge_independence_not_certified" not in blockers,
        "refinement_convergence_passed": bool(bridge.get("SCALE_BRIDGE_REFINEMENT_RECEIPT", False))
        and "refinement_convergence_not_certified" not in blockers,
        "physical_freezeout_surface": bool(bridge.get("PHYSICAL_FREEZEOUT_SURFACE_RECEIPT", False)),
        "no_posthoc_calibration_receipt": bool(bridge.get("NO_POSTHOC_CALIBRATION_RECEIPT", False)),
        "physical_cmb_input_contract_passed": contract_passed,
    }
    diagnostic_required = (
        "no_data_use_receipt",
        "B_A_rows_emitted",
        "rho_A_rows_emitted",
        "rho_A_eq_rows_emitted",
        "Gamma_rec_rows_emitted",
    )
    physical_required = (
        "no_data_use_receipt",
        "physical_k_units_calibrated",
        "calibrated_a_evolution",
        "energy_momentum_exchange_closed",
        "gauge_consistency_audited",
        "refinement_convergence_passed",
        "physical_freezeout_surface",
        "no_posthoc_calibration_receipt",
        "physical_cmb_input_contract_passed",
    )
    physical_missing_gates = [name for name in physical_required if not checks.get(name, False)]
    physical_export_certificate = all(checks[name] for name in physical_required)
    return {
        "checks": checks,
        "FINITE_COLLAR_BOLTZMANN_SOURCE_BUNDLE_RECEIPT": all(checks[name] for name in diagnostic_required),
        "PHYSICAL_BOLTZMANN_EXPORT_CERTIFICATE": physical_export_certificate,
        "diagnostic_missing_gates": [name for name in diagnostic_required if not checks.get(name, False)],
        "physical_missing_gates": physical_missing_gates,
        "mean_abs_B_A": _mean_abs(row.get("B_A") for row in b_a_rows),
        "mean_Gamma_rec_over_H": _mean_abs(row.get("Gamma_rec_over_H") for row in gamma_rows),
        "claim_boundary": (
            "The diagnostic receipt means source-side finite-collar tables exist behind a no-data firewall. "
            "The physical certificate remains closed until every physical gate passes."
        ),
    }


def _source_audit(
    b_a_reports: list[dict[str, Any]],
    transition_reports: list[dict[str, Any]],
    no_data_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "no_data_receipt_count": len(no_data_receipts),
        "no_data_use_ok": _no_data_use_ok(no_data_receipts, b_a_reports, transition_reports),
        "b_a_modes": _counts(report.get("mode") for report in b_a_reports),
        "b_a_primary_sources": _counts(report.get("primary_parent_source") for report in b_a_reports),
        "transition_modes": _counts(report.get("mode") for report in transition_reports),
        "measurement_data_used_flags": {
            "b_a_parent": any(_measurement_data_used(report) for report in b_a_reports),
            "finite_transition": any(_measurement_data_used(report) for report in transition_reports),
        },
    }


def _no_data_use_ok(
    no_data_receipts: list[dict[str, Any]],
    b_a_reports: list[dict[str, Any]],
    transition_reports: list[dict[str, Any]],
) -> bool:
    explicit = any(
        bool(report.get("NO_DATA_USE_RECEIPT", False) or report.get("no_data_use_receipt", False))
        for report in no_data_receipts
    )
    source_clean = not any(_measurement_data_used(report) for report in [*b_a_reports, *transition_reports])
    b_a_clean = all(
        bool(((report.get("readiness", {}) or {}).get("checks", {}) or {}).get("no_cmb_data_used", True))
        for report in b_a_reports
    )
    return bool(source_clean and b_a_clean and (explicit or b_a_reports or transition_reports))


def _find_roots(run_dirs: list[Path]) -> list[Path]:
    return [Path(path) for path in run_dirs if Path(path).exists()]


def _collect_json(roots: list[Path], names: tuple[str, ...]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        root = Path(root)
        candidates: list[Path] = []
        if root.is_file() and root.name in names:
            candidates.append(root)
        elif root.is_dir():
            for name in names:
                direct = root / name
                if direct.exists():
                    candidates.append(direct)
                candidates.extend(sorted(root.glob(f"**/{name}")))
        for path in candidates:
            path = path.resolve()
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                reports.append(data)
    return reports


def _a_grid_from_rows(*row_sets: list[dict[str, Any]]) -> list[float]:
    values = []
    for rows in row_sets:
        for row in rows:
            value = _float(row.get("a"))
            if value is not None:
                values.append(round(value, 12))
    return sorted(set(values))


def _contract_to_jsonable(contract: Any) -> dict[str, Any]:
    data = dict(contract.__dict__)
    for key in ("B_A_k_a", "Gamma_rec_k_a", "rho_A_a"):
        value = data.get(key)
        if value is not None:
            data[key] = np.asarray(value, dtype=float).tolist()
    return data


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
    readiness = report["readiness"]
    checks = readiness["checks"]
    lines = [
        "# Finite-Collar Boltzmann Source Bundle",
        "",
        report["claim_boundary"],
        "",
        "## Readiness",
        "",
        f"- diagnostic source bundle receipt: `{str(report['FINITE_COLLAR_BOLTZMANN_SOURCE_BUNDLE_RECEIPT']).lower()}`",
        f"- physical Boltzmann export certificate: `{str(report['PHYSICAL_BOLTZMANN_EXPORT_CERTIFICATE']).lower()}`",
        f"- physical CMB prediction: `{str(report['physical_cmb_prediction']).lower()}`",
        f"- diagnostic missing gates: {', '.join(readiness['diagnostic_missing_gates']) or 'none'}",
        f"- physical missing gates: {', '.join(readiness['physical_missing_gates']) or 'none'}",
        "",
        "## Source Tables",
        "",
        f"- B_A rows: {report['B_A_k_a_diagnostic']['row_count']}",
        f"- rho_A rows: {report['rho_A_a_diagnostic']['row_count']}",
        f"- Gamma_rec rows: {report['Gamma_rec_k_a_diagnostic']['row_count']}",
        f"- mean |B_A|: {readiness.get('mean_abs_B_A')}",
        f"- mean |Gamma_rec/H|: {readiness.get('mean_Gamma_rec_over_H')}",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in checks.items())
    lines.append("")
    return "\n".join(lines)


def _measurement_data_used(report: dict[str, Any]) -> bool:
    if not report:
        return False
    explicit_clean = (
        report.get("no_cmb_data_used") is True
        or (((report.get("readiness") or {}).get("checks") or {}).get("no_cmb_data_used") is True)
    )
    if explicit_clean:
        return False
    return any(
        bool(report.get(key, False))
        for key in (
            "measurement_data_used",
            "cmb_data_used",
            "planck_data_used_for_input",
            "fit_to_measurement",
            "fit_to_planck",
            "uses_measurements_to_set_inputs",
        )
    )


def _mean_optional(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(fmean(clean)) if clean else None


def _mean_abs(values: Any) -> float | None:
    clean = [abs(float(value)) for value in values if value is not None and np.isfinite(float(value))]
    return float(fmean(clean)) if clean else None


def _counts(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        label = str(value)
        out[label] = out.get(label, 0) + 1
    return out


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None
