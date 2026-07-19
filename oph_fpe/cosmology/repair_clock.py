from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping

import numpy as np

from oph_fpe.constants.oph_pixel import P_STAR
from oph_fpe.cosmology.edge_center_clock import (
    edge_center_clock_target,
    validate_edge_center_clock_evidence,
)
from oph_fpe.cosmology.finite_repair_transition_clock import (
    validate_transition_clock_eligibility,
)


def repair_clock_report(
    run_dirs: list[Path],
    *,
    out_dir: Path | None = None,
    P: float = P_STAR,
    cycle_time_normalization: float | None = None,
    r2_threshold: float = 0.85,
    relative_tolerance: float = 0.05,
    clock_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit whether finite runs realize the selected edge-center clock.

    The selected theorem target is the orientation half of the full-collar
    derivative: eta_R=theta=P/48 and kappa_rep=(P/48)/(P-phi). This audit does
    not promote trace fits or finite-step survival exponents to the source
    derivative. The explicit edge-center evidence bundle must also pass.
    """

    target = edge_center_clock_target(float(P))
    evidence_report = validate_edge_center_clock_evidence(clock_evidence, P=float(P))
    delta_p = target.delta_P
    target_kappa = target.kappa_rep
    target_eta = target.theta
    candidates = _find_candidate_dirs(run_dirs)
    rows: list[dict[str, Any]] = []
    for run_dir in candidates:
        rows.extend(
            _mismatch_trace_rows(
                run_dir,
                delta_p=delta_p,
                cycle_time_normalization=cycle_time_normalization,
                target_kappa=target_kappa,
                target_eta=target_eta,
                r2_threshold=r2_threshold,
                relative_tolerance=relative_tolerance,
            )
        )
        rows.extend(
            _shape_certificate_rows(
                run_dir,
                delta_p=delta_p,
                target_kappa=target_kappa,
                target_eta=target_eta,
                relative_tolerance=relative_tolerance,
            )
        )
        rows.extend(
            _fossil_spectrum_rows(
                run_dir,
                delta_p=delta_p,
                target_kappa=target_kappa,
                target_eta=target_eta,
                relative_tolerance=relative_tolerance,
            )
        )
        rows.extend(
            _scalar_repair_semigroup_rows(
                run_dir,
                delta_p=delta_p,
                target_kappa=target_kappa,
                target_eta=target_eta,
                relative_tolerance=relative_tolerance,
            )
        )
    eligible = [row for row in rows if row.get("eligible_for_certificate")]
    passed = [row for row in eligible if row.get("passed")]
    kappa_values = [float(row["kappa_rep_estimate"]) for row in rows if _finite(row.get("kappa_rep_estimate"))]
    eta_values = [float(row["eta_R_estimate"]) for row in rows if _finite(row.get("eta_R_estimate"))]
    clock_certificate = bool(
        len(passed) >= 3 and evidence_report["edge_center_clock_evidence_complete"]
    )
    blockers = _repair_clock_blockers(
        rows,
        eligible,
        passed,
        cycle_time_normalization,
        evidence_report,
    )
    report = {
        "mode": "oph_repair_clock_kappa_audit_v1",
        "target": {
            "formula": "rho_full=P/24; eta_R=theta=rho_full/2=P/48=kappa_rep*(P-phi)",
            "selected_branch": "edge_center_orientation_half",
            "required_kappa_rep": target_kappa,
            "required_eta_R": target_eta,
            "required_n_s": target.n_s,
            "full_collar_derivative_target": target.full_collar_derivative,
            "orientation_halves": target.orientation_halves,
            "P": target.P,
            "phi": target.phi,
            "delta_P": delta_p,
            "e_diagnostic_control": target.as_jsonable()["diagnostic_controls"]["e"],
        },
        "inputs": {
            "run_roots": [str(Path(path)) for path in run_dirs],
            "candidate_run_count": len(candidates),
            "cycle_time_normalization": cycle_time_normalization,
            "cycle_time_normalization_declared": cycle_time_normalization is not None,
            "r2_threshold": float(r2_threshold),
            "relative_tolerance": float(relative_tolerance),
        },
        "summary": {
            "estimator_count": len(rows),
            "eligible_estimator_count": len(eligible),
            "passed_estimator_count": len(passed),
            "median_kappa_rep_estimate": _median_or_none(kappa_values),
            "median_eta_R_estimate": _median_or_none(eta_values),
            "median_n_s_estimate": (1.0 - median(eta_values)) if eta_values else None,
            "target_kappa_rep": target_kappa,
            "target_eta_R": target_eta,
            "target_n_s": target.n_s,
        },
        "rows": rows,
        "edge_center_clock_evidence": evidence_report,
        **evidence_report["receipts"],
        "EDGE_CENTER_CLOCK_RECEIPT": evidence_report["EDGE_CENTER_CLOCK_RECEIPT"],
        "finite_repair_clock_certificate": clock_certificate,
        "repair_clock_certificate": clock_certificate,
        "eta_R_finite_lattice_derived": clock_certificate,
        "physical_cmb_prediction": False,
        "blockers": blockers,
        "claim_boundary": (
            "Finite kappa_rep repair-clock audit for the exact OPH CMB scalar-tilt branch. "
            "Rows marked ineligible are diagnostic only: they may fit finite traces, Shape witness "
            "proxies, or target-selected fossil spectra, but they do not by themselves prove that the "
            "finite OPH lattice realizes the P/48 edge-center clock. Euler's number remains a named "
            "nonpromoting diagnostic control."
        ),
    }
    if out_dir is not None:
        write_repair_clock_report(
            run_dirs,
            out_dir,
            P=P,
            cycle_time_normalization=cycle_time_normalization,
            r2_threshold=r2_threshold,
            relative_tolerance=relative_tolerance,
            clock_evidence=clock_evidence,
        )
    return report


def write_repair_clock_report(
    run_dirs: list[Path],
    out_dir: Path,
    *,
    P: float = P_STAR,
    cycle_time_normalization: float | None = None,
    r2_threshold: float = 0.85,
    relative_tolerance: float = 0.05,
    clock_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = repair_clock_report(
        run_dirs,
        P=P,
        cycle_time_normalization=cycle_time_normalization,
        r2_threshold=r2_threshold,
        relative_tolerance=relative_tolerance,
        clock_evidence=clock_evidence,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "repair_clock_certificate_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    _write_rows_csv(out_dir / "repair_clock_estimators.csv", report["rows"])
    (out_dir / "repair_clock_certificate_report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _find_candidate_dirs(roots: list[Path]) -> list[Path]:
    filenames = (
        "mismatch_trace.csv",
        "shape_cmb_certificate_inputs.json",
        "fossil_spectrum_report.json",
        "scalar_repair_semigroup_report.json",
    )
    candidates: set[Path] = set()
    for root in roots:
        root = Path(root)
        if root.is_file():
            if root.name in filenames:
                candidates.add(root.parent)
            continue
        if not root.exists():
            continue
        if any((root / name).exists() for name in filenames):
            candidates.add(root)
        for name in filenames:
            for path in root.glob(f"**/{name}"):
                candidates.add(path.parent)
    return sorted(candidates, key=lambda path: str(path))


def _mismatch_trace_rows(
    run_dir: Path,
    *,
    delta_p: float,
    cycle_time_normalization: float | None,
    target_kappa: float,
    target_eta: float,
    r2_threshold: float,
    relative_tolerance: float,
) -> list[dict[str, Any]]:
    path = run_dir / "mismatch_trace.csv"
    if not path.exists():
        return []
    data = _read_mismatch_trace(path)
    phi = data.get("phi")
    cycle = data.get("cycle")
    if phi is None or cycle is None:
        return [_row_failure(run_dir, "global_phi_exponential_decay", path, "missing cycle/phi columns")]
    mask = np.isfinite(phi) & np.isfinite(cycle) & (phi > 1.0e-30)
    if np.count_nonzero(mask) < 4:
        return [_row_failure(run_dir, "global_phi_exponential_decay", path, "insufficient positive phi samples")]
    x = cycle[mask].astype(float)
    if cycle_time_normalization is not None:
        x = x * float(cycle_time_normalization)
    y = np.log(phi[mask].astype(float))
    slope, intercept = np.polyfit(x, y, 1)
    r2 = _fit_r2(x, y, float(slope), float(intercept))
    raw_kappa = float(-slope / max(delta_p, 1.0e-30))
    eta = float(raw_kappa * delta_p)
    rel_error = abs(raw_kappa - target_kappa) / max(abs(target_kappa), 1.0e-30)
    required_dt = _required_cycle_time_for_target(slope, delta_p, target_kappa)
    eligible = bool(cycle_time_normalization is not None and r2 >= r2_threshold)
    passed = bool(eligible and rel_error <= relative_tolerance)
    return [
        {
            "run_dir": str(run_dir),
            "estimator": "global_phi_exponential_decay",
            "source_file": str(path),
            "eligible_for_certificate": eligible,
            "passed": passed,
            "quantity_semantics": "finite_trace_decay_exponent_diagnostic",
            "is_full_collar_derivative": False,
            "reason": "ok" if eligible else "diagnostic_only_missing_or_low_quality_repair_time_normalization",
            "sample_count": int(np.count_nonzero(mask)),
            "cycle_time_normalization": cycle_time_normalization,
            "required_cycle_time_for_selected_edge_center": required_dt,
            "slope": float(slope),
            "intercept": float(intercept),
            "fit_r2": float(r2),
            "kappa_rep_estimate": raw_kappa,
            "eta_R_estimate": eta,
            "n_s_estimate": 1.0 - eta,
            "target_kappa_rep": target_kappa,
            "target_eta_R": target_eta,
            "relative_error_to_selected_edge_center_kappa": float(rel_error),
        }
    ]


def _shape_certificate_rows(
    run_dir: Path,
    *,
    delta_p: float,
    target_kappa: float,
    target_eta: float,
    relative_tolerance: float,
) -> list[dict[str, Any]]:
    path = run_dir / "shape_cmb_certificate_inputs.json"
    if not path.exists():
        return []
    report = _read_json(path)
    rows = []
    for idx, source in enumerate(report.get("kappa_rep", {}).get("rows", [])):
        kappa = _float_or_none(source.get("kappa_rep"))
        eta = kappa * delta_p if kappa is not None else None
        rel_error = (
            abs(kappa - target_kappa) / max(abs(target_kappa), 1.0e-30) if kappa is not None else None
        )
        rows.append(
            {
                "run_dir": str(run_dir),
                "estimator": "shape_phi_trace_proxy",
                "source_file": str(path),
                "source_row": idx,
                "eligible_for_certificate": False,
                "passed": False,
                "quantity_semantics": "shape_witness_proxy_diagnostic",
                "is_full_collar_derivative": False,
                "reason": "Shape substrate witness proxy; not a finite OPH scalar repair-clock certificate",
                "sample_count": None,
                "delta_P_source": source.get("delta_P"),
                "slope": source.get("slope"),
                "fit_r2": source.get("r2"),
                "kappa_rep_estimate": kappa,
                "eta_R_estimate": eta,
                "n_s_estimate": (1.0 - eta) if eta is not None else None,
                "target_kappa_rep": target_kappa,
                "target_eta_R": target_eta,
                "relative_error_to_selected_edge_center_kappa": rel_error,
                "relative_tolerance": relative_tolerance,
            }
        )
    if not rows:
        rows.append(_row_failure(run_dir, "shape_phi_trace_proxy", path, "missing kappa rows"))
    return rows


def _fossil_spectrum_rows(
    run_dir: Path,
    *,
    delta_p: float,
    target_kappa: float,
    target_eta: float,
    relative_tolerance: float,
) -> list[dict[str, Any]]:
    path = run_dir / "fossil_spectrum_report.json"
    if not path.exists():
        return []
    report = _read_json(path)
    best = report.get("best_target_closeness_diagnostic", {})
    eta = _float_or_none(best.get("eta_R"))
    kappa = eta / delta_p if eta is not None and abs(delta_p) > 1.0e-30 else None
    rel_error = abs(kappa - target_kappa) / max(abs(target_kappa), 1.0e-30) if kappa is not None else None
    return [
        {
            "run_dir": str(run_dir),
            "estimator": "fossil_spectrum_best_target_closeness",
            "source_file": str(path),
            "eligible_for_certificate": False,
            "passed": False,
            "quantity_semantics": "target_closeness_diagnostic",
            "is_full_collar_derivative": False,
            "reason": "target-closeness diagnostic; not an objective repair-clock semigroup fit",
            "field": best.get("field"),
            "cycle": best.get("cycle"),
            "kappa_rep_estimate": kappa,
            "eta_R_estimate": eta,
            "n_s_estimate": (1.0 - eta) if eta is not None else None,
            "target_kappa_rep": target_kappa,
            "target_eta_R": target_eta,
            "relative_error_to_selected_edge_center_kappa": rel_error,
            "relative_tolerance": relative_tolerance,
            "best_beats_same_field_controls": report.get("best_beats_same_field_controls"),
        }
    ]


def _scalar_repair_semigroup_rows(
    run_dir: Path,
    *,
    delta_p: float,
    target_kappa: float,
    target_eta: float,
    relative_tolerance: float,
) -> list[dict[str, Any]]:
    path = run_dir / "scalar_repair_semigroup_report.json"
    if not path.exists():
        return []
    report = _read_json(path)
    semigroup = report.get("semigroup", {}) if isinstance(report, dict) else {}
    transition_certificate = report.get("transition_matrix_certificate", {}) if isinstance(report, dict) else {}
    kappa = _float_or_none(semigroup.get("kappa_rep_estimate"))
    eta = _float_or_none(semigroup.get("eta_R_estimate"))
    if eta is None and kappa is not None:
        eta = float(kappa * delta_p)
    rel_error = abs(kappa - target_kappa) / max(abs(target_kappa), 1.0e-30) if kappa is not None else None
    source = report.get("source")
    transition_eligibility = validate_transition_clock_eligibility(report)
    finite_lattice_derived = bool(transition_eligibility["eligible"])
    eligible = bool(
        report.get("eligible_for_repair_clock_certificate", False)
        and finite_lattice_derived
        and source == "finite_state_transition_matrix"
        and bool(report.get("semigroup_controls_passed", False))
    )
    passed = bool(eligible and rel_error is not None and rel_error <= relative_tolerance)
    if eligible:
        reason = "finite_state_transition_matrix_semigroup_certificate"
    elif source == "declared_edge_center_p_over_48_target":
        reason = "declared P/48 edge-center target; algebraic check only, not finite-lattice derived"
    elif source == "declared_euler_repair_time_target":
        reason = "legacy Euler diagnostic control; nonpromoting and not finite-lattice derived"
    elif source == "finite_state_transition_matrix" and transition_eligibility["eligible"]:
        reason = "finite transition matrix present, but repair-clock normalization is not certified"
    else:
        reason = "scalar semigroup report is not certificate-eligible"
    return [
        {
            "run_dir": str(run_dir),
            "estimator": "scalar_repair_semigroup_gap",
            "source_file": str(path),
            "source": source,
            "matrix_source": report.get("matrix_source"),
            "finite_lattice_derived": finite_lattice_derived,
            "eligible_for_certificate": eligible,
            "passed": passed,
            "quantity_semantics": "finite_step_survival_semigroup_diagnostic",
            "is_full_collar_derivative": False,
            "reason": reason,
            "dimension": report.get("dimension"),
            "centered_subspace_dimension": report.get("centered_subspace_dimension"),
            "constant_mode_zero": semigroup.get("constant_mode_zero"),
            "centered_scalar_relaxation": semigroup.get("centered_scalar_relaxation"),
            "contractive_at_t1": semigroup.get("contractive_at_t1"),
            "semigroup_controls_passed": report.get("semigroup_controls_passed"),
            "edge_center_clock_evidence_complete": False,
            "transition_matrix_ready": bool(transition_eligibility["eligible"]),
            "transition_clock_eligibility": transition_eligibility,
            "transition_clock_normalization_certified": bool(
                transition_eligibility["eligible"]
                and transition_certificate.get("clock_normalization_certified")
            ),
            "transition_required_repair_step_time_for_selected_edge_center": transition_certificate.get(
                "required_repair_step_time_for_selected_edge_center"
            ),
            "transition_repair_step_time_for_e_diagnostic_control": transition_certificate.get(
                "repair_step_time_for_e_diagnostic_control"
            ),
            "transition_primary_lambda_2": transition_certificate.get("primary_lambda_2"),
            "transition_primary_gamma": transition_certificate.get("primary_gamma"),
            "centered_gap": semigroup.get("centered_gap"),
            "kappa_rep_estimate": kappa,
            "eta_R_estimate": eta,
            "n_s_estimate": (1.0 - eta) if eta is not None else None,
            "target_kappa_rep": target_kappa,
            "target_eta_R": target_eta,
            "relative_error_to_selected_edge_center_kappa": rel_error,
            "relative_tolerance": relative_tolerance,
        }
    ]


def _read_mismatch_trace(path: Path) -> dict[str, np.ndarray]:
    columns: dict[str, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key, value in row.items():
                try:
                    columns.setdefault(key, []).append(float(value))
                except (TypeError, ValueError):
                    columns.setdefault(key, []).append(float("nan"))
    return {key: np.asarray(value, dtype=float) for key, value in columns.items()}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _row_failure(run_dir: Path, estimator: str, path: Path, reason: str) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "estimator": estimator,
        "source_file": str(path),
        "eligible_for_certificate": False,
        "passed": False,
        "reason": reason,
    }


def _fit_r2(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 1.0e-30:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _required_cycle_time_for_target(slope: float, delta_p: float, target_kappa: float) -> float | None:
    denominator = -target_kappa * delta_p
    if abs(denominator) <= 1.0e-30 or not np.isfinite(slope):
        return None
    value = float(slope / denominator)
    return value if value > 0.0 and math.isfinite(value) else None


def _repair_clock_blockers(
    rows: list[dict[str, Any]],
    eligible: list[dict[str, Any]],
    passed: list[dict[str, Any]],
    cycle_time_normalization: float | None,
    clock_evidence: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not rows:
        blockers.append("no finite repair-clock estimator inputs found")
    has_mismatch_rows = any(row.get("estimator") == "global_phi_exponential_decay" for row in rows)
    if has_mismatch_rows and cycle_time_normalization is None:
        blockers.append("no predeclared finite repair-time normalization for mismatch-trace slopes")
    if not eligible:
        blockers.append("no estimator rows are theorem-grade eligible")
    if eligible and len(passed) < 3:
        blockers.append(
            "fewer than three eligible estimators match the selected edge-center kappa tolerance"
        )
    if any(row.get("estimator") == "fossil_spectrum_best_target_closeness" for row in rows):
        blockers.append("fossil-spectrum eta fits are target-closeness diagnostics, not repair-clock derivations")
    if any(row.get("estimator") == "shape_phi_trace_proxy" for row in rows):
        blockers.append("Shape substrate kappa rows are declared-witness proxies, not OPH finite-lattice certificates")
    if any(
        row.get("estimator") == "scalar_repair_semigroup_gap"
        and row.get("source") == "finite_state_transition_matrix"
        and row.get("transition_matrix_ready")
        and not row.get("transition_clock_normalization_certified")
        for row in rows
    ):
        blockers.append(
            "finite transition-matrix rows do not match kappa_rep=(P/48)/(P-phi) under their "
            "declared repair-step time"
        )
    elif any(
        row.get("estimator") == "scalar_repair_semigroup_gap" and not row.get("eligible_for_certificate")
        for row in rows
    ):
        blockers.append(
            "scalar repair-semigroup rows are target/algebra diagnostics unless sourced from a finite transition matrix"
        )
    for receipt in clock_evidence.get("missing_receipts", []):
        blockers.append(f"missing edge-center clock evidence: {receipt}")
    return blockers


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["run_dir", "estimator", "eligible_for_certificate", "passed", "reason"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _markdown_report(report: dict[str, Any]) -> str:
    target = report["target"]
    summary = report["summary"]
    lines = [
        "# OPH Repair-Clock Kappa Audit",
        "",
        report["claim_boundary"],
        "",
        "## Target",
        "",
        f"- required kappa_rep: `{target['required_kappa_rep']:.12g}`",
        f"- required eta_R: `{target['required_eta_R']:.12g}`",
        f"- required n_s: `{target['required_n_s']:.12g}`",
        "",
        "## Result",
        "",
        f"- finite repair-clock certificate: `{report['finite_repair_clock_certificate']}`",
        f"- estimator rows: `{summary['estimator_count']}`",
        f"- eligible rows: `{summary['eligible_estimator_count']}`",
        f"- passed rows: `{summary['passed_estimator_count']}`",
        f"- median kappa_rep estimate: `{summary['median_kappa_rep_estimate']}`",
        f"- median eta_R estimate: `{summary['median_eta_R_estimate']}`",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- {item}" for item in report["blockers"])
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `repair_clock_certificate_report.json`",
            "- `repair_clock_estimators.csv`",
            "- `repair_clock_certificate_report.md`",
            "",
        ]
    )
    return "\n".join(lines)


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _median_or_none(values: list[float]) -> float | None:
    return float(median(values)) if values else None
