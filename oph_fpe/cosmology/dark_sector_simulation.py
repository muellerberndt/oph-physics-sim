from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPORT_FILENAMES = {
    "static_galaxy": "static_galaxy_measurement_report.json",
    "finite_covariant_parent": "finite_covariant_collar_packet_parent_report.json",
    "finite_collar_boltzmann": "finite_collar_boltzmann_bundle_report.json",
    "boltzmann_inputs": "oph_boltzmann_input_report.json",
    "cmb_anomaly": "cmb_anomaly_report.json",
    "frozen_likelihood": "frozen_transfer_likelihood_report.json",
}


STAGE_ORDER = (
    "static_galaxy",
    "finite_covariant_parent",
    "finite_collar_boltzmann",
    "boltzmann_inputs",
    "frozen_likelihood",
)


STAGE_LABELS = {
    "static_galaxy": "Static galaxy RAR/BTFR diagnostic",
    "finite_covariant_parent": "Finite covariant collar-packet parent",
    "finite_collar_boltzmann": "Finite-collar Boltzmann source bundle",
    "boltzmann_inputs": "Boltzmann input/export bridge",
    "frozen_likelihood": "Frozen transfer and likelihood closure",
    "cmb_anomaly": "CMB anomaly diagnostic",
}


SUGGESTED_COMMANDS = {
    "static_galaxy": (
        "python3 -m oph_fpe.cli run-galaxy-static "
        "--sparc-dir data/measurements/sparc --out-dir runs/dark_sector/static_galaxy"
    ),
    "finite_covariant_parent": (
        "python3 -m oph_fpe.cli finite-covariant-collar-parent "
        "--source runs/<run_id>/finite_covariant_parent.json "
        "--out runs/<run_id>/finite_covariant_collar_packet_parent_report.json"
    ),
    "finite_collar_boltzmann": (
        "python3 -m oph_fpe.cli finite-collar-boltzmann-bundle "
        "--run-dir runs/<run_id> --out runs/<run_id>/finite_collar_boltzmann"
    ),
    "boltzmann_inputs": (
        "python3 -m oph_fpe.cli oph-boltzmann-inputs "
        "--run-dir runs/<run_id> --out runs/<run_id>/oph_boltzmann_inputs"
    ),
    "frozen_likelihood": (
        "python3 -m oph_fpe.cli frozen-transfer-likelihood "
        "--run-dir runs/<run_id> --out runs/<run_id>/frozen_transfer_likelihood"
    ),
    "cmb_anomaly": (
        "python3 -m oph_fpe.cli cmb-anomaly-report "
        "--run-dir runs/<run_id> --source-dir runs/<run_id> --out runs/<run_id>/cmb_anomaly"
    ),
}


CLAIM_BOUNDARIES = {
    "static_galaxy": (
        "Checks the settled-galaxy RAR/BTFR bridge only. It is not a cluster, CMB, "
        "or physical Boltzmann prediction."
    ),
    "finite_covariant_parent": (
        "Validates the covariant parent artifact required before physical stress, "
        "exchange-current, and recipient-channel claims can be promoted. Source-only "
        "dark abundance also requires RHO_A_SOURCE_RECEIPT from the anomaly abundance selector."
    ),
    "finite_collar_boltzmann": (
        "Collects rho_A(a), rho_A,eq(a), B_A(k,a), and Gamma_rec(k,a) diagnostics. "
        "Physical promotion requires the hard Boltzmann export certificate. Exact exp(-P/24) "
        "coefficient promotion additionally requires the local reserve, scalar-weighted z6 mean, "
        "and uniform product-thickening exact gates."
    ),
    "boltzmann_inputs": (
        "Exports solver-facing tables. Current repair-exchange rows are diagnostics "
        "unless upstream physical source, RHO_A_SOURCE_RECEIPT, and input-contract receipts pass."
    ),
    "frozen_likelihood": (
        "Freezes sources, solver pins, CDM-limit regressions, controls, and official "
        "likelihood receipts. It is a closure audit, not publication by itself."
    ),
    "cmb_anomaly": (
        "Records finite-screen anomaly diagnostics and should not be used as a dark "
        "matter likelihood or physical CMB prediction."
    ),
}


def dark_sector_simulation_plan(run_dirs: list[Path]) -> dict[str, Any]:
    """Summarize the dark-sector promotion ladder from existing run receipts.

    The function intentionally does not run physics. It reads emitted
    simulator receipts, marks the first missing gate, and names the next command
    that would improve the receipt stack.
    """

    roots = _find_roots(run_dirs)
    reports = _collect_reports(roots)
    stage_summary = {
        "static_galaxy": _static_galaxy_stage(reports["static_galaxy"]),
        "finite_covariant_parent": _finite_parent_stage(reports["finite_covariant_parent"]),
        "finite_collar_boltzmann": _finite_collar_stage(reports["finite_collar_boltzmann"]),
        "boltzmann_inputs": _boltzmann_stage(reports["boltzmann_inputs"], reports["finite_collar_boltzmann"]),
        "frozen_likelihood": _frozen_likelihood_stage(reports["frozen_likelihood"]),
        "cmb_anomaly": _cmb_anomaly_stage(reports["cmb_anomaly"]),
    }
    first_blocking_stage = next(
        (stage for stage in STAGE_ORDER if not stage_summary[stage]["gate_passed"]),
        None,
    )
    promotion_ready = bool(first_blocking_stage is None)
    suggestions = [
        _suggestion(stage, stage_summary[stage], next_stage=(stage == first_blocking_stage))
        for stage in (*STAGE_ORDER, "cmb_anomaly")
    ]
    return {
        "mode": "oph_dark_sector_simulation_plan_v0",
        "DARK_SECTOR_SIMULATION_PLAN_RECEIPT": True,
        "run_dirs": [str(path) for path in roots],
        "report_counts": {stage: len(items) for stage, items in reports.items()},
        "stage_summary": stage_summary,
        "first_blocking_stage": first_blocking_stage,
        "DARK_SECTOR_PHYSICAL_PROMOTION_READY": promotion_ready,
        "physical_dark_sector_prediction": False,
        "simulation_suggestions": suggestions,
        "claim_boundary": (
            "Integration receipt for the dark-sector simulator ladder. It reads galaxy, finite-parent, "
            "finite-collar Boltzmann, Boltzmann-input, anomaly, and frozen-likelihood reports and names "
            "the next blocked gate. It is not a physical dark-sector prediction, not a fitted likelihood, "
            "and not a substitute for the upstream source-generation receipts."
        ),
    }


def write_dark_sector_simulation_plan(run_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    report = dark_sector_simulation_plan(run_dirs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dark_sector_simulation_plan.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "dark_sector_simulation_plan.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _collect_reports(roots: list[Path]) -> dict[str, list[dict[str, Any]]]:
    return {stage: _read_reports(roots, filename) for stage, filename in REPORT_FILENAMES.items()}


def _static_galaxy_stage(reports: list[dict[str, Any]]) -> dict[str, Any]:
    best = _best_report(reports, ("STATIC_GALAXY_RAR_BTFR_RECEIPT", "OPH_STATIC_GALAXY_BRIDGE_RECEIPT"))
    receipt = bool(
        best.get("STATIC_GALAXY_RAR_BTFR_RECEIPT", False)
        or best.get("OPH_STATIC_GALAXY_BRIDGE_RECEIPT", False)
    )
    blockers = _blockers(best, fallback=[] if receipt else ["static_galaxy_rar_btfr_receipt_missing"])
    return _stage("static_galaxy", reports, receipt, best, blockers)


def _finite_parent_stage(reports: list[dict[str, Any]]) -> dict[str, Any]:
    best = _best_report(reports, ("FINITE_COVARIANT_COLLAR_PACKET_PARENT_RECEIPT",))
    receipt = bool(best.get("FINITE_COVARIANT_COLLAR_PACKET_PARENT_RECEIPT", False))
    blockers = _blockers(best, fallback=[] if receipt else ["finite_covariant_parent_receipt_missing"])
    return _stage("finite_covariant_parent", reports, receipt, best, blockers)


def _finite_collar_stage(reports: list[dict[str, Any]]) -> dict[str, Any]:
    best = _best_report(
        reports,
        ("PHYSICAL_BOLTZMANN_EXPORT_CERTIFICATE", "FINITE_COLLAR_BOLTZMANN_SOURCE_BUNDLE_RECEIPT"),
    )
    diagnostic = bool(best.get("FINITE_COLLAR_BOLTZMANN_SOURCE_BUNDLE_RECEIPT", False))
    physical = bool(best.get("PHYSICAL_BOLTZMANN_EXPORT_CERTIFICATE", False))
    blockers = _blockers(best, fallback=[] if physical else ["physical_boltzmann_export_certificate_missing"])
    stage = _stage("finite_collar_boltzmann", reports, physical, best, blockers)
    theorem_gates = best.get("theorem_gates", {}) or {}
    readiness = best.get("readiness", {}) or {}
    local_reserve = bool(
        theorem_gates.get("LOCAL_POISSON_RESERVE_SURVIVAL")
        or readiness.get("LOCAL_POISSON_RESERVE_SURVIVAL")
        or best.get("LOCAL_POISSON_RESERVE_SURVIVAL")
    )
    scalar_mean = bool(
        theorem_gates.get("SCALAR_WEIGHTED_Z6_MEAN")
        or readiness.get("SCALAR_WEIGHTED_Z6_MEAN")
        or best.get("SCALAR_WEIGHTED_Z6_MEAN")
    )
    exact_uniform = bool(
        theorem_gates.get("UNIFORM_PRODUCT_THICKENING_EXACT")
        or readiness.get("UNIFORM_PRODUCT_THICKENING_EXACT")
        or best.get("UNIFORM_PRODUCT_THICKENING_EXACT")
    )
    stage["diagnostic_receipt"] = diagnostic
    stage["physical_certificate"] = physical
    stage["scalar_coefficient_gates"] = {
        "LOCAL_POISSON_RESERVE_SURVIVAL": local_reserve,
        "SCALAR_WEIGHTED_Z6_MEAN": scalar_mean,
        "UNIFORM_PRODUCT_THICKENING_EXACT": exact_uniform,
    }
    stage["exact_uniform_lambda_claim_ready"] = bool(local_reserve and scalar_mean and exact_uniform)
    stage["finite_thickness_profile_default"] = "lambda_collar = integral dy w(y) exp[-epsilon_Z6(y)]"
    stage["exact_uniform_target_blockers"] = [
        name for name, passed in stage["scalar_coefficient_gates"].items() if not passed
    ]
    return stage


def _boltzmann_stage(
    boltzmann_reports: list[dict[str, Any]],
    finite_collar_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    best = _best_report(boltzmann_reports, ("physical_cmb_prediction", "physical_matter_power_prediction"))
    finite_physical = any(
        bool(report.get("PHYSICAL_BOLTZMANN_EXPORT_CERTIFICATE", False))
        for report in finite_collar_reports
    )
    cdm_ready = bool((best.get("readiness", {}) or {}).get("cdm_limit_solver_ready", False))
    diagnostic = bool(
        (best.get("readiness", {}) or {}).get("diagnostic_repair_exchange_table_ready", False)
        or (best.get("readiness", {}) or {}).get("B_A_parent_diagnostic_table_ready", False)
    )
    gate = bool(finite_physical and cdm_ready)
    blockers = _blockers(
        best,
        fallback=[] if gate else ["physical_boltzmann_input_export_missing"],
    )
    if not finite_physical and "finite_collar_physical_certificate_missing" not in blockers:
        blockers.append("finite_collar_physical_certificate_missing")
    if not cdm_ready and "cdm_limit_solver_ready_missing" not in blockers:
        blockers.append("cdm_limit_solver_ready_missing")
    stage = _stage("boltzmann_inputs", boltzmann_reports, gate, best, blockers)
    stage["diagnostic_table_ready"] = diagnostic
    stage["cdm_limit_solver_ready"] = cdm_ready
    return stage


def _frozen_likelihood_stage(reports: list[dict[str, Any]]) -> dict[str, Any]:
    best = _best_report(
        reports,
        (
            "LIKELIHOOD_EVALUATED_PHYSICAL_PREDICTION_RECEIPT",
            "FROZEN_TRANSFER_LIKELIHOOD_CLOSURE_RECEIPT",
            "FROZEN_PHYSICAL_SPECTRUM_RECEIPT",
        ),
    )
    receipt = bool(best.get("LIKELIHOOD_EVALUATED_PHYSICAL_PREDICTION_RECEIPT", False))
    blockers = _blockers(best, fallback=[] if receipt else ["likelihood_evaluated_prediction_receipt_missing"])
    stage = _stage("frozen_likelihood", reports, receipt, best, blockers)
    stage["closure_receipt"] = bool(best.get("FROZEN_TRANSFER_LIKELIHOOD_CLOSURE_RECEIPT", False))
    stage["physical_spectrum_receipt"] = bool(best.get("FROZEN_PHYSICAL_SPECTRUM_RECEIPT", False))
    return stage


def _cmb_anomaly_stage(reports: list[dict[str, Any]]) -> dict[str, Any]:
    best = _best_report(reports, ("CMB_ANOMALY_REPORT_RECEIPT", "CMB_ANOMALY_DIAGNOSTIC_RECEIPT"))
    receipt = bool(
        best.get("CMB_ANOMALY_REPORT_RECEIPT", False)
        or best.get("CMB_ANOMALY_DIAGNOSTIC_RECEIPT", False)
        or reports
    )
    stage = _stage("cmb_anomaly", reports, receipt, best, _blockers(best, fallback=[] if receipt else []))
    stage["optional"] = True
    return stage


def _stage(
    stage: str,
    reports: list[dict[str, Any]],
    gate_passed: bool,
    best: dict[str, Any],
    blockers: list[str],
) -> dict[str, Any]:
    return {
        "label": STAGE_LABELS[stage],
        "present": bool(reports),
        "report_count": len(reports),
        "gate_passed": bool(gate_passed),
        "blockers": blockers,
        "receipt_keys": _truthy_receipt_keys(best),
        "next_command": SUGGESTED_COMMANDS[stage],
        "claim_boundary": CLAIM_BOUNDARIES[stage],
    }


def _suggestion(stage: str, summary: dict[str, Any], *, next_stage: bool) -> dict[str, Any]:
    if summary["gate_passed"]:
        status = "passed"
    elif summary["present"]:
        status = "inspect_or_rerun"
    else:
        status = "missing"
    return {
        "stage": stage,
        "label": STAGE_LABELS[stage],
        "status": status,
        "next_blocker": bool(next_stage),
        "command": SUGGESTED_COMMANDS[stage],
        "claim_boundary": CLAIM_BOUNDARIES[stage],
    }


def _find_roots(paths: list[Path]) -> list[Path]:
    roots = [Path(path) for path in paths]
    return roots or [Path.cwd()]


def _read_reports(roots: list[Path], filename: str) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        root = Path(root)
        candidates = [root] if root.is_file() and root.name == filename else []
        if root.is_dir():
            candidates.extend(sorted(root.glob(f"**/{filename}")))
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            payload = _read_json(path)
            if payload:
                payload = dict(payload)
                payload.setdefault("_source_path", str(path))
                reports.append(payload)
    return reports


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _best_report(reports: list[dict[str, Any]], receipt_keys: tuple[str, ...]) -> dict[str, Any]:
    for key in receipt_keys:
        for report in reports:
            if bool(report.get(key, False)):
                return report
    return reports[0] if reports else {}


def _blockers(report: dict[str, Any], *, fallback: list[str]) -> list[str]:
    raw: list[Any] = []
    for key in ("blockers", "physical_blockers", "parent_blockers", "missing_gates"):
        value = report.get(key)
        if isinstance(value, list):
            raw.extend(value)
    readiness = report.get("readiness", {}) or {}
    if isinstance(readiness, dict):
        for key in ("physical_missing_gates", "missing_gates", "blockers"):
            value = readiness.get(key)
            if isinstance(value, list):
                raw.extend(value)
    validation = report.get("physical_cmb_input_validation", {}) or {}
    if isinstance(validation, dict):
        value = validation.get("blockers")
        if isinstance(value, list):
            raw.extend(value)
    strings = _unique_strings(raw)
    return strings or list(fallback)


def _truthy_receipt_keys(report: dict[str, Any]) -> list[str]:
    return sorted(
        key
        for key, value in report.items()
        if key.endswith("_RECEIPT") or key.endswith("_CERTIFICATE")
        if bool(value)
    )


def _unique_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# OPH Dark-Sector Simulation Plan",
        "",
        f"Receipt: `{report['DARK_SECTOR_SIMULATION_PLAN_RECEIPT']}`",
        f"Physical promotion ready: `{report['DARK_SECTOR_PHYSICAL_PROMOTION_READY']}`",
        f"Physical dark-sector prediction: `{report['physical_dark_sector_prediction']}`",
        f"First blocking stage: `{report['first_blocking_stage']}`",
        "",
        "## Gate Summary",
        "",
        "| Stage | Present | Gate | Blockers |",
        "| --- | ---: | ---: | --- |",
    ]
    for stage in (*STAGE_ORDER, "cmb_anomaly"):
        summary = report["stage_summary"][stage]
        blockers = ", ".join(summary["blockers"]) if summary["blockers"] else "none"
        lines.append(
            f"| {summary['label']} | {summary['present']} | {summary['gate_passed']} | {blockers} |"
        )
    lines.extend(["", "## Next Simulation Suggestions", ""])
    for item in report["simulation_suggestions"]:
        marker = "next blocker" if item["next_blocker"] else item["status"]
        lines.extend(
            [
                f"### {item['label']}",
                "",
                f"Status: `{marker}`",
                "",
                "```bash",
                item["command"],
                "```",
                "",
                item["claim_boundary"],
                "",
            ]
        )
    lines.extend(["## Claim Boundary", "", report["claim_boundary"], ""])
    return "\n".join(lines)
