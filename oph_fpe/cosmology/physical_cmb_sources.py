from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.cosmology.boltzmann_inputs import write_oph_boltzmann_input_report
from oph_fpe.cosmology.finite_collar_boltzmann_bundle import (
    write_finite_collar_boltzmann_bundle_report,
)
from oph_fpe.cosmology.finite_covariant_parent import (
    CAUSAL_RESPONSE_RECEIPT,
    EXPLICIT_RECIPIENT_STRESS_RECEIPT,
    FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT,
    GAUGE_INDEPENDENCE_RECEIPT,
    PARENT_RECEIPT,
    REFINEMENT_CONVERGENCE_RECEIPT,
    STRESS_CLOSURE_RECEIPT,
    finite_covariant_collar_packet_parent_report,
)


def write_physical_cmb_source_readiness_report(run_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    """Build and audit the source-side artifacts needed by physical-CMB gates.

    This function is intentionally fail-closed. It writes a readiness summary
    when no primitive parent artifact exists; it does not synthesize a
    parent-shaped model. A parent passes only when an explicit finite covariant
    collar-packet parent artifact is supplied and independently verified.
    """

    roots = _unique_roots([Path(path) for path in run_dirs])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    existing_parent = _first_json([*roots, out], "finite_covariant_collar_packet_parent_report.json")
    existing_artifact = _first_json([*roots, out], "finite_covariant_collar_packet_parent_artifact.json")
    existing_parent_generated = bool(existing_parent.get("source_readiness_builder_generated", False))
    existing_artifact_generated = bool(
        (existing_artifact.get("manifest") or {}).get("builder") == "physical_cmb_source_readiness_builder_v0"
    )
    parent_report_written = False
    parent_artifact_status: dict[str, Any]
    if existing_artifact and not existing_artifact_generated:
        parent_report = finite_covariant_collar_packet_parent_report(existing_artifact)
        parent_report_written = True
        (out / "finite_covariant_collar_packet_parent_report.json").write_text(
            json.dumps(parent_report, indent=2, default=str),
            encoding="utf-8",
        )
        parent_artifact_status = {
            "existing_parent_artifact_used": True,
            "existing_parent_report_used": False,
            "candidate_artifact_written": False,
            "readiness_summary_written": False,
            "source_blockers": [],
        }
    elif existing_parent and not existing_parent_generated:
        parent_report = existing_parent
        parent_artifact_status = {
            "existing_parent_artifact_used": False,
            "existing_parent_report_used": True,
            "candidate_artifact_written": False,
            "readiness_summary_written": False,
            "source_blockers": ["finite_parent_model_artifact_missing_for_report"],
        }
        parent_report = dict(parent_report)
        parent_report["blockers"] = _unique_strings(
            [*list(parent_report.get("blockers") or []), "finite_parent_model_artifact_missing_for_report"]
        )
        parent_report[PARENT_RECEIPT] = False
    else:
        summary, parent_artifact_status = build_finite_parent_readiness_summary_from_reports(roots)
        (out / "finite_parent_readiness_summary.json").write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )
        parent_report = _missing_parent_report(summary)

    source_roots = _unique_roots([*roots, out])
    boltzmann = write_oph_boltzmann_input_report(source_roots, out)
    finite_collar = write_finite_collar_boltzmann_bundle_report(source_roots, out)
    finite_collar_physical = bool(finite_collar.get("PHYSICAL_BOLTZMANN_EXPORT_CERTIFICATE", False))
    boltzmann_diagnostic_missing = [] if finite_collar_physical else list(
        ((boltzmann.get("readiness") or {}).get("missing_gates") or [])
    )
    finite_collar_physical_missing = list(
        ((finite_collar.get("readiness") or {}).get("physical_missing_gates") or [])
    )
    blockers = _unique_strings(
        [
            *list(parent_artifact_status.get("source_blockers") or []),
            *list(parent_report.get("blockers") or []),
            *[
                f"finite_collar_boltzmann_missing_{gate}"
                for gate in finite_collar_physical_missing
            ],
        ]
    )
    report = {
        "mode": "physical_cmb_source_readiness_builder_v0",
        "run_dirs": [str(path) for path in roots],
        "report_dir": str(out),
            "finite_covariant_parent": {
            "existing_parent_artifact_used": bool(
                parent_artifact_status.get("existing_parent_artifact_used", False)
            ),
            "existing_parent_report_used": bool(
                parent_artifact_status.get("existing_parent_report_used", False)
            ),
            "parent_report_written": parent_report_written,
            "readiness_summary_written": bool(parent_artifact_status.get("readiness_summary_written", False)),
            "candidate_artifact_written": bool(parent_artifact_status.get("candidate_artifact_written", False)),
            "parent_receipt": bool(parent_report.get(PARENT_RECEIPT, False)),
            "stress_energy_closure_receipt": bool(parent_report.get(STRESS_CLOSURE_RECEIPT, False)),
            "explicit_recipient_stress_receipt": bool(
                parent_report.get(EXPLICIT_RECIPIENT_STRESS_RECEIPT, False)
            ),
            "gauge_independence_receipt": bool(parent_report.get(GAUGE_INDEPENDENCE_RECEIPT, False)),
            "causal_response_receipt": bool(parent_report.get(CAUSAL_RESPONSE_RECEIPT, False)),
            "refinement_convergence_receipt": bool(
                parent_report.get(REFINEMENT_CONVERGENCE_RECEIPT, False)
            ),
            "frozen_likelihood_protocol_receipt": bool(
                parent_report.get(FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT, False)
            ),
            "source_hash_present": _nonempty(parent_report.get("source_hash")),
            "solver_hash_present": _nonempty(parent_report.get("solver_hash")),
            "likelihood_hash_present": _nonempty(parent_report.get("likelihood_hash")),
            "blockers": list(parent_report.get("blockers") or []),
            "artifact_status": parent_artifact_status,
        },
        "oph_boltzmann_input": {
            "written": True,
            "cdm_limit_solver_ready": bool((boltzmann.get("readiness") or {}).get("cdm_limit_solver_ready")),
            "diagnostic_repair_exchange_table_ready": bool(
                (boltzmann.get("readiness") or {}).get("diagnostic_repair_exchange_table_ready")
            ),
            "B_A_parent_diagnostic_table_ready": bool(
                (boltzmann.get("readiness") or {}).get("B_A_parent_diagnostic_table_ready")
            ),
            "finite_repair_clock_diagnostic_table_ready": bool(
                (boltzmann.get("readiness") or {}).get("finite_repair_clock_diagnostic_table_ready")
            ),
            "physical_prediction_ready": bool(
                (boltzmann.get("readiness") or {}).get("physical_prediction_ready")
            ),
            "missing_gates": list((boltzmann.get("readiness") or {}).get("missing_gates") or []),
            "diagnostic_missing_gates": boltzmann_diagnostic_missing,
            "hard_blocking_missing_gates": [],
        },
        "finite_collar_boltzmann_bundle": {
            "written": True,
            "source_bundle_receipt": bool(
                finite_collar.get("FINITE_COLLAR_BOLTZMANN_SOURCE_BUNDLE_RECEIPT", False)
            ),
            "physical_certificate": bool(
                finite_collar.get("PHYSICAL_BOLTZMANN_EXPORT_CERTIFICATE", False)
            ),
            "diagnostic_missing_gates": list(
                (finite_collar.get("readiness") or {}).get("diagnostic_missing_gates") or []
            ),
            "physical_missing_gates": list(
                (finite_collar.get("readiness") or {}).get("physical_missing_gates") or []
            ),
            "hard_blocking_missing_gates": [
                f"finite_collar_boltzmann_missing_{gate}"
                for gate in finite_collar_physical_missing
            ],
        },
        "gate_summary": {
            "finite_covariant_source_parent": bool(parent_report.get(PARENT_RECEIPT, False)),
            "stress_closure": bool(parent_report.get(STRESS_CLOSURE_RECEIPT, False)),
            "boltzmann_source_tables": bool(
                finite_collar.get("FINITE_COLLAR_BOLTZMANN_SOURCE_BUNDLE_RECEIPT", False)
                or (boltzmann.get("readiness") or {}).get("diagnostic_repair_exchange_table_ready", False)
                or (boltzmann.get("readiness") or {}).get("B_A_parent_diagnostic_table_ready", False)
                or (boltzmann.get("readiness") or {}).get("finite_repair_clock_diagnostic_table_ready", False)
            ),
            "physical_boltzmann_export_certificate": bool(
                finite_collar.get("PHYSICAL_BOLTZMANN_EXPORT_CERTIFICATE", False)
            ),
            "frozen_likelihood_protocol": bool(
                parent_report.get(FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT, False)
            ),
        },
        "blockers": blockers,
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "Source-side physical-CMB readiness builder. It writes/audits finite source, stress-parent, "
            "Boltzmann-handoff, and frozen-likelihood artifacts. Passing these gates can make the hard "
            "input contract eligible; it is not itself a CMB prediction or likelihood result."
        ),
    }
    (out / "physical_cmb_source_readiness_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "physical_cmb_source_readiness_report.md").write_text(
        _markdown_report(report),
        encoding="utf-8",
    )
    return report


def build_finite_parent_readiness_summary_from_reports(run_dirs: list[Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    roots = _unique_roots([Path(path) for path in run_dirs])
    reports = {
        "finite_transition": _first_json(roots, "finite_repair_transition_matrix_report.json"),
        "finite_certificate": _first_json(roots, "finite_certificate_report.json"),
        "B_A_kernel": _first_json(roots, "B_A_kernel_report.json"),
        "B_A_kernel_refinement": _first_json(roots, "B_A_kernel_refinement_report.json"),
        "b_a_parent": _first_json(roots, "b_a_parent_report.json"),
        "scale_compressed": _first_json(roots, "scale_compressed_repair_report.json"),
        "strict_neutral": _first_json(roots, "strict_neutral_bulk_report.json"),
        "camb_baseline": _first_json(roots, "camb_lcdm_baseline_report.json"),
        "compressed_likelihood": _first_json(roots, "oph_compressed_likelihood_report.json"),
        "official_likelihood": _first_json(roots, "official_planck_likelihood_readiness_report.json"),
    }
    source_hash = _source_hash(reports)
    gamma = _gamma_rec(reports["finite_transition"])
    anomaly_rho = _anomaly_rho(reports["finite_certificate"], reports["b_a_parent"])
    b_a_rows = _array(reports["B_A_kernel"].get("B_A_k_a"))
    transition_ready = bool(
        reports["finite_transition"].get("finite_transition_matrix_ready", False)
        and _finite(gamma)
        and (
            reports["finite_transition"].get("eta_R_finite_lattice_derived", False)
            or reports["finite_transition"].get("eta_R_empirical_finite_lattice_derived", False)
            or ((reports["finite_transition"].get("clock_modes") or {}).get("empirical") or {}).get(
                "eta_R_finite_lattice_derived",
                False,
            )
        )
    )
    finite_certificate_ready = bool(
        reports["finite_certificate"].get("theorem_grade_finite_inputs", False)
        and _finite(_a_zeta(reports["finite_certificate"]))
        and anomaly_rho is not None
        and (
            ((reports["finite_certificate"].get("derived_outputs") or {}).get(
                "screen_to_primordial_lift_receipt",
                False,
            ))
            or reports["finite_certificate"].get("SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT", False)
            or reports["finite_certificate"].get("screen_to_primordial_lift_receipt", False)
        )
    )
    ba_kernel_ready = bool(reports["B_A_kernel"].get("B_A_KERNEL_RECEIPT", False) and b_a_rows is not None)
    physical_checks = _physical_checks(reports["B_A_kernel"], reports["b_a_parent"])
    energy_ready = bool(physical_checks.get("energy_momentum_exchange_closed", False))
    gauge_ready = bool(physical_checks.get("gauge_consistency_audited", False))
    refinement_ready = bool(
        reports["B_A_kernel_refinement"].get("B_A_KERNEL_REFINEMENT_CONVERGENCE_RECEIPT", False)
        or physical_checks.get("refinement_convergence_passed", False)
    )
    stress_ready = bool(finite_certificate_ready and ba_kernel_ready and energy_ready)
    causal_ready = bool(transition_ready and ba_kernel_ready and stress_ready)
    cdm_ready = bool(
        reports["camb_baseline"].get("CDM_LIMIT_BOLTZMANN_RECEIPT", False)
        or reports["compressed_likelihood"].get("cdm_limit_regression_passed", False)
    )
    official_ready = bool(
        reports["official_likelihood"].get("official_likelihood_execution_ready", False)
        or reports["compressed_likelihood"].get("official_likelihood_ready", False)
    )
    solver_hash = _hash_value("solver", reports["official_likelihood"], reports["compressed_likelihood"])
    likelihood_hash = _hash_value("likelihood", reports["official_likelihood"], reports["compressed_likelihood"])
    frozen_ready = bool(
        official_ready
        and _nonempty(solver_hash)
        and _nonempty(likelihood_hash)
        and (
            reports["official_likelihood"].get(FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT, False)
            or reports["compressed_likelihood"].get(FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT, False)
            or reports["official_likelihood"].get("frozen_likelihood_protocol_receipt", False)
        )
    )
    source_blockers = _source_blockers(
        transition_ready=transition_ready,
        finite_certificate_ready=finite_certificate_ready,
        ba_kernel_ready=ba_kernel_ready,
        energy_ready=energy_ready,
        gauge_ready=gauge_ready,
        refinement_ready=refinement_ready,
        cdm_ready=cdm_ready,
        official_ready=official_ready,
        frozen_ready=frozen_ready,
    )
    source_blockers = _unique_strings(
        [
            "explicit_finite_covariant_parent_artifact_missing",
            *source_blockers,
        ]
    )
    summary = {
        "mode": "finite_parent_readiness_summary_from_reports_v0",
        "not_a_model_artifact": True,
        "not_a_verification_report": True,
        "physical_parent_builder": False,
        "manifest": {
            "source_hash": source_hash,
            "regulator_id": _regulator_id(reports),
            "parent_theorem_version": "finite_covariant_collar_packet_parent_from_sources_v1",
            "builder": "physical_cmb_source_readiness_builder_v0",
            "promotion_allowed": False,
        },
        "gamma_repair_step": float(gamma) if _finite(gamma) else None,
        "Gamma_rec": None,
        "Gamma_rec_status": "UNPROMOTED_REPAIR_STEP_DIAGNOSTIC",
        "finite_covariant_parent_artifact": None,
        "finite_covariant_parent_verification_report": None,
        "source_blockers": source_blockers,
        "source_gate_status": {
            "finite_transition_ready": transition_ready,
            "finite_certificate_ready": finite_certificate_ready,
            "B_A_kernel_ready": ba_kernel_ready,
            "energy_momentum_exchange_closed": energy_ready,
            "gauge_consistency_audited": gauge_ready,
            "refinement_convergence_passed": refinement_ready,
            "cdm_limit_regression_passed": cdm_ready,
            "official_likelihood_ready": official_ready,
            "frozen_likelihood_protocol_ready": frozen_ready,
        },
    }
    return summary, {
        "existing_parent_report_used": False,
        "candidate_artifact_written": False,
        "readiness_summary_written": True,
        "source_hash": source_hash,
        "source_blockers": source_blockers,
        "source_report_presence": {name: bool(report) for name, report in reports.items()},
        "source_gate_status": summary["source_gate_status"],
    }


def build_finite_covariant_parent_artifact_from_reports(run_dirs: list[Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compatibility wrapper returning a non-promotable readiness summary.

    This function no longer builds a parent-shaped JSON artifact from downstream
    reports. Call ``build_finite_parent_readiness_summary_from_reports`` for the
    explicit API name.
    """

    summary, status = build_finite_parent_readiness_summary_from_reports(run_dirs)
    status = dict(status)
    status["candidate_artifact_written"] = False
    status["compatibility_wrapper"] = True
    return summary, status


def _missing_parent_report(summary: dict[str, Any]) -> dict[str, Any]:
    blockers = _unique_strings(
        [
            "explicit_finite_covariant_parent_artifact_missing",
            *list(summary.get("source_blockers") or []),
        ]
    )
    source_hash = ((summary.get("manifest") or {}).get("source_hash")) if isinstance(summary.get("manifest"), dict) else None
    return {
        "mode": "finite_covariant_collar_packet_parent_missing_v0",
        PARENT_RECEIPT: False,
        STRESS_CLOSURE_RECEIPT: False,
        EXPLICIT_RECIPIENT_STRESS_RECEIPT: False,
        GAUGE_INDEPENDENCE_RECEIPT: False,
        CAUSAL_RESPONSE_RECEIPT: False,
        REFINEMENT_CONVERGENCE_RECEIPT: False,
        FROZEN_LIKELIHOOD_PROTOCOL_RECEIPT: False,
        "source_hash": source_hash,
        "solver_hash": None,
        "likelihood_hash": None,
        "blockers": blockers,
    }


def _physical_checks(ba_kernel: dict[str, Any], ba_parent: dict[str, Any]) -> dict[str, bool]:
    checks = {}
    for source in (
        ba_kernel.get("physical_checks"),
        (ba_parent.get("readiness") or {}).get("checks") if isinstance(ba_parent.get("readiness"), dict) else None,
    ):
        if isinstance(source, dict):
            checks.update({str(key): bool(value) for key, value in source.items()})
    return {
        "energy_momentum_exchange_closed": bool(checks.get("energy_momentum_exchange_closed", False)),
        "gauge_consistency_audited": bool(checks.get("gauge_consistency_audited", False)),
        "refinement_convergence_passed": bool(checks.get("refinement_convergence_passed", False)),
    }


def _source_blockers(**checks: bool) -> list[str]:
    names = {
        "transition_ready": "finite_repair_transition_clock_not_ready",
        "finite_certificate_ready": "finite_certificate_not_theorem_grade",
        "ba_kernel_ready": "B_A_kernel_receipt_missing",
        "energy_ready": "energy_momentum_exchange_not_closed",
        "gauge_ready": "gauge_consistency_not_audited",
        "refinement_ready": "regulator_refinement_convergence_not_passed",
        "cdm_ready": "cdm_limit_regression_not_passed",
        "official_ready": "official_likelihood_not_ready",
        "frozen_ready": "frozen_likelihood_protocol_not_ready",
    }
    return [names[key] for key, passed in checks.items() if not passed]


def _causal_response(ready: bool) -> dict[str, Any]:
    return {
        "characteristic_speed_bound": None,
        "synthetic_placeholder": True,
        "source_ready_diagnostic": bool(ready),
    }


def _source_hash(reports: dict[str, dict[str, Any]]) -> str:
    source_side_names = (
        "finite_transition",
        "finite_certificate",
        "B_A_kernel",
        "B_A_kernel_refinement",
        "b_a_parent",
        "scale_compressed",
        "strict_neutral",
    )
    payload = json.dumps(
        {name: reports.get(name, {}) for name in source_side_names},
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return f"sha256:{sha256(payload).hexdigest()}"


def _hash_value(prefix: str, *reports: dict[str, Any]) -> str | None:
    for report in reports:
        for key in (f"{prefix}_hash", f"frozen_{prefix}_hash"):
            value = report.get(key)
            if _nonempty(value):
                return str(value)
    return None


def _regulator_id(reports: dict[str, dict[str, Any]]) -> str:
    for report in reports.values():
        for key in ("regulator_id", "run_id", "config_id"):
            value = report.get(key)
            if _nonempty(value):
                return str(value)
    return "finite_source_reports"


def _gamma_rec(report: dict[str, Any]) -> float | None:
    return _float((report.get("primary") or {}).get("gamma_continuous"))


def _a_zeta(report: dict[str, Any]) -> float | None:
    return _float((report.get("derived_outputs") or {}).get("A_zeta", report.get("A_zeta")))


def _anomaly_rho(finite_certificate: dict[str, Any], ba_parent: dict[str, Any]) -> float | None:
    rho = _array((finite_certificate.get("derived_outputs") or {}).get("rho_A_a", finite_certificate.get("rho_A_a")))
    if rho is not None:
        arr = np.asarray(rho, dtype=float)
        if arr.ndim == 1 and arr.size:
            return float(arr[-1])
        if arr.ndim >= 2 and arr.shape[0] and arr.shape[1] >= 2:
            return float(arr[0, 1])
    for row in ba_parent.get("rows") or ba_parent.get("observer_view_rows") or []:
        if not isinstance(row, dict):
            continue
        value = _float(row.get("rho_A", row.get("rho_A_base", row.get("base_epsilon_cmi"))))
        if value is not None:
            return float(value)
    return None


def _first_json(roots: list[Path], name: str) -> dict[str, Any]:
    fallback_empty: dict[str, Any] | None = None
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
                if data:
                    return data
                fallback_empty = data
    return fallback_empty or {}


def _unique_roots(paths: list[Path]) -> list[Path]:
    roots = []
    seen: set[Path] = set()
    for path in paths:
        path = Path(path)
        key = path.resolve() if path.exists() else path
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            roots.append(path)
    return roots


def _array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return array if array.size and np.all(np.isfinite(array)) else None


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _finite(value: Any) -> bool:
    return _float(value) is not None


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _unique_strings(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _markdown_report(report: dict[str, Any]) -> str:
    parent = report["finite_covariant_parent"]
    boltzmann = report["oph_boltzmann_input"]
    collar = report["finite_collar_boltzmann_bundle"]
    lines = [
        "# Physical CMB Source Readiness",
        "",
        report["claim_boundary"],
        "",
        "## Parent",
        "",
        f"- parent receipt: `{str(parent['parent_receipt']).lower()}`",
        f"- stress closure: `{str(parent['stress_energy_closure_receipt']).lower()}`",
        f"- gauge independence: `{str(parent['gauge_independence_receipt']).lower()}`",
        f"- causal response: `{str(parent['causal_response_receipt']).lower()}`",
        f"- refinement convergence: `{str(parent['refinement_convergence_receipt']).lower()}`",
        f"- frozen likelihood protocol: `{str(parent['frozen_likelihood_protocol_receipt']).lower()}`",
        "",
        "## Boltzmann Handoff",
        "",
        f"- diagnostic input table written: `{str(boltzmann['written']).lower()}`",
        f"- CDM-limit solver ready: `{str(boltzmann['cdm_limit_solver_ready']).lower()}`",
        f"- finite-collar source bundle: `{str(collar['source_bundle_receipt']).lower()}`",
        f"- physical export certificate: `{str(collar['physical_certificate']).lower()}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = report.get("blockers") or []
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)
