from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean, median
from typing import Any

from oph_fpe.cosmology.unique_predictions import unique_prediction_gate_report


RUN_REPORTS = (
    "cmb_anomaly_report.json",
    "collar_markov_report.json",
    "parent_collar_ladder_report.json",
    "oph_cmb_stress_report.json",
    "bulk_reconstruction_report.json",
    "cl_comparison_report.json",
)


def cmb_parameter_derivation_report(
    run_dirs: list[Path],
    *,
    source_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit whether finite lattice receipts currently derive CMB target numbers.

    The OPH unique-prediction gate gives measurement-comparable target values.
    This report asks the stricter simulator question: which of those targets
    are actually emitted by current finite cap/collar/screen runs, with controls?
    """

    targets = unique_prediction_gate_report(source_dir)
    rows = [_run_derivation_row(path, targets) for path in _find_run_dirs(run_dirs)]
    rows = [row for row in rows if row.get("has_cmb_derivation_inputs")]
    aggregate = _aggregate(rows, targets)
    return {
        "mode": "finite_lattice_cmb_parameter_derivation_audit_v0",
        "target_source": {
            "mode": targets.get("mode"),
            "source_files": targets.get("source_files", {}),
            "scalar_n_s": targets["scalar_tilt"]["n_s"],
            "eta_R": targets["scalar_tilt"]["eta_R"],
            "q_IR": targets["cmb_ir_kernel"]["q_IR"],
            "ell_IR": targets["cmb_ir_kernel"]["ell_IR"],
            "N_frz_proxy": targets["cmb_ir_kernel"]["N_frz_proxy"],
            "parity_R_OE_TT_2_29": targets["parity_envelope"]["predicted_R_OE_TT_2_29"],
        },
        "run_count": len(rows),
        "aggregate": aggregate,
        "rows": rows,
        "finite_lattice_cmb_parameters_ready": bool(aggregate.get("all_required_gates_pass")),
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "Derivation audit for current finite OPH-FPE runs. The v0.9 OPH CMB numbers are comparable "
            "to public measurements, but this report keeps them separate from what the finite lattice has "
            "actually emitted from observer-visible cap/collar/screen records. A physical CMB prediction "
            "requires simulator-derived eta_R, q_IR, ell_IR, parity/BipoSH covariance, anomaly stress kernels, "
            "and official likelihood/map-space controls."
        ),
    }


def write_cmb_parameter_derivation_report(run_dirs: list[Path], out_dir: Path, *, source_dir: Path | None = None) -> dict[str, Any]:
    report = cmb_parameter_derivation_report(run_dirs, source_dir=source_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cmb_parameter_derivation_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "cmb_parameter_derivation_report.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_csv(out / "cmb_parameter_derivation_rows.csv", report["rows"])
    return report


def _find_run_dirs(roots: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        root = Path(root)
        if root.is_file():
            paths.add(root.parent)
            continue
        if any((root / name).exists() for name in RUN_REPORTS):
            paths.add(root)
        if root.exists():
            for name in RUN_REPORTS:
                for report_path in root.glob(f"**/{name}"):
                    paths.add(report_path.parent)
    return sorted(paths, key=lambda path: str(path))


def _run_derivation_row(run_path: Path, targets: dict[str, Any]) -> dict[str, Any]:
    run_path = Path(run_path)
    anomaly = _read_json(run_path / "cmb_anomaly_report.json")
    collar = _read_json(run_path / "collar_markov_report.json")
    parent_collar_ladder = _read_json(run_path / "parent_collar_ladder_report.json")
    cmb_stress = _read_json(run_path / "oph_cmb_stress_report.json")
    bulk = _read_json(run_path / "bulk_reconstruction_report.json")
    bulk_proof = _read_json(run_path / "bulk_proof_certificate_report.json")
    emergence = _read_json(run_path / "emergence_status_report.json")
    cl = _read_json(run_path / "cl_comparison_report.json")
    aggregate = anomaly.get("aggregate", {}) if isinstance(anomaly, dict) else {}
    screen_capacity = anomaly.get("screen_capacity", {}) if isinstance(anomaly, dict) else {}
    q_row = _best_q_ir_row(anomaly, targets)
    field_count = int(aggregate.get("field_count", 0) or 0)
    low_count = int(aggregate.get("low_power_suppressed_vs_controls_count", 0) or 0)
    large_count = int(aggregate.get("large_angle_suppressed_vs_controls_count", 0) or 0)
    parity_count = int(aggregate.get("parity_more_asymmetric_than_controls_count", 0) or 0)
    tilt_count = int(aggregate.get("planck_tilt_compatible_proxy_count", 0) or 0)
    collar_median = _float_or_none(collar.get("median_epsilon_cmi")) if isinstance(collar, dict) else None
    collar_p90 = _float_or_none(collar.get("p90_epsilon_cmi")) if isinstance(collar, dict) else None
    parent_collar_ready = bool(parent_collar_ladder.get("parent_collar_recovery_ladder_receipt", False))
    parent_collar_density_ready = bool(parent_collar_ladder.get("local_recovery_density_receipt", False))
    parent_collar_theorem_grade = bool(parent_collar_ladder.get("theorem_grade_parent_collar_ladder", False))
    parent_collar_final_p90 = _final_parent_collar_value(parent_collar_ladder, "p90_epsilon_cmi")
    parent_collar_final_p90_density = _final_parent_collar_value(
        parent_collar_ladder,
        "p90_epsilon_cmi_per_collar_patch",
    )
    parent_collar_slope = _nested(parent_collar_ladder, "scaling", "slope")
    parent_collar_density_slope = _nested(parent_collar_ladder, "local_density_scaling", "slope")
    target_eta = float(targets["scalar_tilt"]["eta_R"])
    target_q = float(targets["cmb_ir_kernel"]["q_IR"])
    target_ell = float(targets["cmb_ir_kernel"]["ell_IR"])
    best_eta = _float_or_none(aggregate.get("best_eta_R_estimate"))
    best_ns = _float_or_none(aggregate.get("best_n_s_proxy"))
    strict_neutral_bulk = bool(bulk.get("bulk_3d_established", False)) if isinstance(bulk, dict) else False
    if isinstance(bulk_proof, dict) and bulk_proof:
        theorem_assisted_h3_bulk = bool(
            bulk_proof.get("theorem_assisted_h3_populated_chart_established", False)
            or bulk_proof.get("theorem_assisted_h3_nonboundary_population_established", False)
            or bulk_proof.get("bulk_3d_established_theorem_assisted", False)
        )
    else:
        theorem_assisted_h3_bulk = bool(
            isinstance(emergence, dict)
            and (
                emergence.get("OBJECT_H3_NONBOUNDARY_POPULATION_RECEIPT", False)
                or emergence.get("OBJECT_BULK_POPULATION_RECEIPT", False)
                or emergence.get("observer_chart_bulk_population_receipt", False)
                or emergence.get("paper_theorem_neutral_populated_bulk_receipt", False)
                or emergence.get("strict_blind_observer_bulk_receipt", False)
                or emergence.get("neutral_bulk_3d_established", False)
            )
        )
    run_gates = {
        "tilt_eta_R_simulator_compatible": bool(tilt_count > 0),
        "low_power_control_separation": bool(low_count >= max(1, field_count // 2)),
        "large_angle_control_separation": bool(large_count >= max(1, field_count // 2)),
        "parity_control_separation": bool(parity_count >= max(1, field_count // 2)),
        "collar_markov_error_small": bool(collar_median is not None and collar_median <= 0.05),
        "parent_collar_local_density_ladder_pass": parent_collar_density_ready,
        "parent_collar_recovery_ladder_pass": parent_collar_ready,
        "parent_collar_theorem_grade": parent_collar_theorem_grade,
        "theorem_assisted_h3_bulk_established": theorem_assisted_h3_bulk,
        "strict_neutral_bulk_3d_established": strict_neutral_bulk,
        "bulk_3d_established": strict_neutral_bulk,
        "anomaly_kernel_emitted": bool(
            ((cmb_stress.get("diagnostic_kernel_proxy", {}) or {}).get("B_A_k_a_emitted", False))
            if isinstance(cmb_stress, dict)
            else False
        ),
        "physical_cl_claim_allowed": bool(cl.get("cosmo_proxy_receipt", False)) if isinstance(cl, dict) else False,
    }
    all_required = all(
        run_gates[key]
        for key in (
            "tilt_eta_R_simulator_compatible",
            "low_power_control_separation",
            "large_angle_control_separation",
            "parity_control_separation",
            "collar_markov_error_small",
            "parent_collar_recovery_ladder_pass",
            "parent_collar_theorem_grade",
            "strict_neutral_bulk_3d_established",
            "anomaly_kernel_emitted",
        )
    )
    patch_count = _float_or_none(screen_capacity.get("patch_count"))
    n_frz = float(targets["cmb_ir_kernel"]["N_frz_proxy"])
    return {
        "run_path": str(run_path),
        "has_cmb_derivation_inputs": bool(anomaly or collar or parent_collar_ladder or cmb_stress or cl),
        "point_count": _int_or_none(anomaly.get("point_count")) if isinstance(anomaly, dict) else None,
        "ell_max": _int_or_none(anomaly.get("ell_max")) if isinstance(anomaly, dict) else None,
        "screen_capacity_total_entropy": _float_or_none(screen_capacity.get("total_entropy_capacity")),
        "screen_capacity_ell_sqrt_patch_proxy": _float_or_none(screen_capacity.get("ell_sqrt_patch_capacity_proxy")),
        "target_N_frz_proxy": int(n_frz),
        "target_N_frz_over_patch_count": float(n_frz / patch_count) if patch_count else None,
        "best_eta_R_estimate": best_eta,
        "target_eta_R": target_eta,
        "eta_R_abs_error": abs(best_eta - target_eta) if best_eta is not None else None,
        "best_n_s_proxy": best_ns,
        "target_n_s": float(targets["scalar_tilt"]["n_s"]),
        "best_q_IR_proxy": q_row.get("q_IR_proxy"),
        "best_q_IR_proxy_field": q_row.get("field"),
        "target_q_IR": target_q,
        "q_IR_abs_error": abs(q_row["q_IR_proxy"] - target_q) if q_row.get("q_IR_proxy") is not None else None,
        "best_ell_IR_proxy": q_row.get("ell_IR_proxy"),
        "target_ell_IR": target_ell,
        "ell_IR_abs_error": abs(q_row["ell_IR_proxy"] - target_ell) if q_row.get("ell_IR_proxy") is not None else None,
        "field_count": field_count,
        "low_power_suppressed_vs_controls_count": low_count,
        "large_angle_suppressed_vs_controls_count": large_count,
        "parity_more_asymmetric_than_controls_count": parity_count,
        "planck_tilt_compatible_proxy_count": tilt_count,
        "median_epsilon_cmi": collar_median,
        "p90_epsilon_cmi": collar_p90,
        "parent_collar_ladder_written": bool(parent_collar_ladder),
        "parent_collar_local_density_ladder_pass": parent_collar_density_ready,
        "parent_collar_recovery_ladder_pass": parent_collar_ready,
        "parent_collar_theorem_grade": parent_collar_theorem_grade,
        "parent_collar_final_p90_epsilon_cmi": parent_collar_final_p90,
        "parent_collar_final_p90_epsilon_cmi_per_collar_patch": parent_collar_final_p90_density,
        "parent_collar_log_cmi_slope": parent_collar_slope,
        "parent_collar_log_cmi_density_slope": parent_collar_density_slope,
        "theorem_assisted_h3_bulk_established": theorem_assisted_h3_bulk,
        "strict_neutral_bulk_3d_established": strict_neutral_bulk,
        "bulk_3d_established": run_gates["bulk_3d_established"],
        "gates": run_gates,
        "all_required_gates_pass": bool(all_required),
        "finite_lattice_physical_cmb_ready": False,
    }


def _best_q_ir_row(anomaly: dict[str, Any], targets: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(anomaly, dict):
        return {}
    target_q = float(targets["cmb_ir_kernel"]["q_IR"])
    target_ell = float(targets["cmb_ir_kernel"]["ell_IR"])
    rows = []
    for field, payload in (anomaly.get("fields", {}) or {}).items():
        fit = ((payload.get("stats", {}) or {}).get("screen_power_fit", {}) or {}) if isinstance(payload, dict) else {}
        low = fit.get("low_ell_suppression", {}) if isinstance(fit, dict) else {}
        q = _float_or_none(low.get("q_IR_proxy"))
        ell = _float_or_none(low.get("ell_IR_proxy"))
        if q is None or ell is None:
            continue
        rows.append(
            {
                "field": field,
                "q_IR_proxy": q,
                "ell_IR_proxy": ell,
                "target_distance": abs(q - target_q) + 0.01 * abs(ell - target_ell),
            }
        )
    if not rows:
        return {}
    return min(rows, key=lambda row: float(row["target_distance"]))


def _aggregate(rows: list[dict[str, Any]], targets: dict[str, Any]) -> dict[str, Any]:
    gate_names = sorted({gate for row in rows for gate in (row.get("gates", {}) or {})})
    gate_counts = {
        gate: sum(1 for row in rows if bool((row.get("gates", {}) or {}).get(gate)))
        for gate in gate_names
    }
    target_eta = float(targets["scalar_tilt"]["eta_R"])
    eta_values = [float(row["best_eta_R_estimate"]) for row in rows if row.get("best_eta_R_estimate") is not None]
    q_values = [float(row["best_q_IR_proxy"]) for row in rows if row.get("best_q_IR_proxy") is not None]
    ell_values = [float(row["best_ell_IR_proxy"]) for row in rows if row.get("best_ell_IR_proxy") is not None]
    cmi_values = [float(row["median_epsilon_cmi"]) for row in rows if row.get("median_epsilon_cmi") is not None]
    parent_cmi_values = [
        float(row["parent_collar_final_p90_epsilon_cmi"])
        for row in rows
        if row.get("parent_collar_final_p90_epsilon_cmi") is not None
    ]
    parent_slope_values = [
        float(row["parent_collar_log_cmi_slope"])
        for row in rows
        if row.get("parent_collar_log_cmi_slope") is not None
    ]
    parent_cmi_density_values = [
        float(row["parent_collar_final_p90_epsilon_cmi_per_collar_patch"])
        for row in rows
        if row.get("parent_collar_final_p90_epsilon_cmi_per_collar_patch") is not None
    ]
    parent_density_slope_values = [
        float(row["parent_collar_log_cmi_density_slope"])
        for row in rows
        if row.get("parent_collar_log_cmi_density_slope") is not None
    ]
    return {
        "all_required_gates_pass": bool(rows and all(bool(row.get("all_required_gates_pass")) for row in rows)),
        "gate_pass_counts": gate_counts,
        "mean_best_eta_R_estimate": fmean(eta_values) if eta_values else None,
        "median_best_eta_R_estimate": median(eta_values) if eta_values else None,
        "target_eta_R": target_eta,
        "mean_eta_R_abs_error": fmean(abs(value - target_eta) for value in eta_values) if eta_values else None,
        "mean_best_q_IR_proxy": fmean(q_values) if q_values else None,
        "target_q_IR": float(targets["cmb_ir_kernel"]["q_IR"]),
        "mean_best_ell_IR_proxy": fmean(ell_values) if ell_values else None,
        "target_ell_IR": float(targets["cmb_ir_kernel"]["ell_IR"]),
        "mean_median_epsilon_cmi": fmean(cmi_values) if cmi_values else None,
        "mean_parent_collar_final_p90_epsilon_cmi": fmean(parent_cmi_values) if parent_cmi_values else None,
        "mean_parent_collar_log_cmi_slope": fmean(parent_slope_values) if parent_slope_values else None,
        "mean_parent_collar_final_p90_epsilon_cmi_per_collar_patch": (
            fmean(parent_cmi_density_values) if parent_cmi_density_values else None
        ),
        "mean_parent_collar_log_cmi_density_slope": (
            fmean(parent_density_slope_values) if parent_density_slope_values else None
        ),
        "run_count": len(rows),
        "readiness_summary": _readiness_summary(gate_counts, len(rows)),
    }


def _readiness_summary(gate_counts: dict[str, int], run_count: int) -> list[str]:
    if run_count == 0:
        return ["no finite run reports with CMB derivation inputs were found"]
    missing = [name for name, count in gate_counts.items() if count < run_count]
    if not missing:
        return ["all current derivation gates pass in the selected run set"]
    return [
        "finite simulator has measurement-comparable diagnostics but not a physical CMB derivation yet",
        "missing or partial gates: " + ", ".join(missing),
    ]


def _read_json(path: Path) -> dict[str, Any]:
    if not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row if key != "gates"))
    gate_keys = sorted({gate for row in rows for gate in (row.get("gates", {}) or {})})
    columns = keys + [f"gate_{gate}" for gate in gate_keys]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {key: value for key, value in row.items() if key != "gates"}
            for gate in gate_keys:
                flat[f"gate_{gate}"] = (row.get("gates", {}) or {}).get(gate)
            writer.writerow(flat)


def _markdown_report(report: dict[str, Any]) -> str:
    target = report["target_source"]
    aggregate = report["aggregate"]
    lines = [
        "# Finite Lattice CMB Parameter Derivation Audit",
        "",
        f"- run count: `{report['run_count']}`",
        f"- target n_s: `{target['scalar_n_s']:.12f}`",
        f"- target eta_R: `{target['eta_R']:.12f}`",
        f"- target q_IR / ell_IR: `{target['q_IR']}` / `{target['ell_IR']}`",
        f"- finite lattice CMB parameters ready: `{report['finite_lattice_cmb_parameters_ready']}`",
        f"- physical CMB prediction: `{report['physical_cmb_prediction']}`",
        "",
        "## Aggregate",
        "",
    ]
    for key in (
        "mean_best_eta_R_estimate",
        "mean_eta_R_abs_error",
        "mean_best_q_IR_proxy",
        "mean_best_ell_IR_proxy",
        "mean_median_epsilon_cmi",
        "mean_parent_collar_final_p90_epsilon_cmi",
        "mean_parent_collar_log_cmi_slope",
        "mean_parent_collar_final_p90_epsilon_cmi_per_collar_patch",
        "mean_parent_collar_log_cmi_density_slope",
    ):
        lines.append(f"- {key}: `{aggregate.get(key)}`")
    lines.extend(["", "## Gate Counts", ""])
    for key, value in (aggregate.get("gate_pass_counts", {}) or {}).items():
        lines.append(f"- {key}: `{value}/{report['run_count']}`")
    lines.extend(["", "## Readiness", ""])
    for item in aggregate.get("readiness_summary", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Claim Boundary", "", str(report["claim_boundary"]), ""])
    return "\n".join(lines)


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _nested(data: dict[str, Any], *keys: str) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _final_parent_collar_value(report: dict[str, Any], key: str) -> Any:
    if not isinstance(report, dict) or not report:
        return None
    patch_counts = report.get("patch_counts") or []
    if not patch_counts:
        return None
    try:
        final_key = str(max(int(value) for value in patch_counts))
    except (TypeError, ValueError):
        return None
    return _nested(report, "by_patch_count", final_key, key)


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
