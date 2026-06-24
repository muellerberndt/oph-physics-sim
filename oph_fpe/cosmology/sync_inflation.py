from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from oph_fpe.claims import CONTINUATION, COSMOLOGY_PERTURBATION_RECEIPT, with_claim_metadata
from oph_fpe.cosmology.inflation_cmb_ladder import flat_sector_selection_report, screen_spectrum_prediction
from oph_fpe.cosmology.spatial_curvature import s3_holonomy_spatial_curvature_gate


def synchronization_inflation_report(run_dirs: list[Path], *, w_eff: float = 1.0 / 3.0) -> dict[str, Any]:
    rows = collect_synchronization_runs(run_dirs, w_eff=w_eff)
    aggregate = _aggregate(rows)
    screen_prediction = screen_spectrum_prediction()
    flat_selector = flat_sector_selection_report()
    report = {
        "mode": "oph_synchronization_inflation_diagnostic_v0",
        "run_count": len(rows),
        "rows": rows,
        "aggregate": aggregate,
        "flat_sector_selection": flat_selector,
        "screen_spectrum_prediction": screen_prediction,
        "theorem_targets": {
            "flat_sector_selection": (
                "Zero clock-slice spatial Levi-Civita holonomy identifies the flat FLRW branch. "
                "Exact selection is separate direct theorem, conditional CMH theorem, or explicit "
                "branch assumption; MAR is not used as a cosmological flatness selector."
            ),
            "screen_s3_defect_decay_diagnostic": (
                "Optional finite-run repair diagnostic on the S3 screen/collar permutation defect. "
                "It is not a spatial Levi-Civita holonomy and does not prove K=0 or Omega_K=0."
            ),
            "horizon_synchronization": (
                "C_sigma(k) = integral Gamma_sigma(k,eta)deta >> 1, requiring a low-k repair gap "
                "or same-boundary quotient-normal-form selector"
            ),
            "scale_invariant_screen_spectrum": (
                "Conditional screen Green spectrum: C_l^q uses A_q after scalar-release-energy "
                f"certification, n_s={screen_prediction['n_s']:.12g}; A_zeta is pending the "
                "screen-to-primordial lift receipt."
            ),
            "acoustic_transfer_boundary": (
                "OPH supplies coherent adiabatic zeta_k; standard photon-baryon transfer supplies peaks"
            ),
            "hot_release": (
                "Hot start is a synchronized-screen MaxEnt release into the realized SM state, not inflaton reheating."
            ),
            "adiabaticity": (
                "Same-boundary scalar normal form gives S_ij=0 up to synchronization residue and suppresses "
                "the decaying acoustic mode."
            ),
        },
        "inflation_replacement_ready": False,
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "Inflation-alternative theorem-target diagnostic derived from cached finite-screen receipts. "
            "The Pro v2 route keeps inflation replacement conditional on zero-holonomy flat-sector "
            "selection, a certified screen Green spectrum, scalar-release energy, screen-to-primordial "
            "lift, hot MaxEnt release, adiabatic same-boundary records, and Boltzmann transfer. The finite "
            "lattice still needs spatial Levi-Civita curvature-holonomy receipts or a CMH/direct theorem, "
            "low-k synchronization evidence, a lift "
            "receipt, and theorem-grade rho_A(a), B_A(k,a), Gamma_rec(k,a)."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=CONTINUATION,
        receipt=COSMOLOGY_PERTURBATION_RECEIPT,
        physical_claim=False,
        observable_id="oph_synchronization_inflation_theorem_targets",
        fit_objective="finite_repair_trace_to_inflation_replacement_gate_audit",
    )


def write_synchronization_inflation_report(run_dirs: list[Path], out_dir: Path, *, w_eff: float = 1.0 / 3.0) -> dict[str, Any]:
    report = synchronization_inflation_report(run_dirs, w_eff=float(w_eff))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sync_inflation_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "sync_inflation_report.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_csv(out / "sync_inflation_rows.csv", report["rows"])
    return report


def collect_synchronization_runs(run_dirs: list[Path], *, w_eff: float = 1.0 / 3.0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in run_dirs:
        for manifest_path in sorted(Path(root).glob("**/manifest.json")):
            run_path = manifest_path.parent.resolve()
            if run_path in seen:
                continue
            seen.add(run_path)
            trace = _read_trace(run_path / "mismatch_trace.csv")
            if not trace:
                continue
            manifest = _read_json(manifest_path)
            collar = _read_json(run_path / "collar_markov_report.json")
            holonomy = _read_json(run_path / "array_holonomy_report.json")
            stress = _read_json(run_path / "oph_cmb_stress_report.json")
            row = _run_sync_row(
                run_path=run_path,
                manifest=manifest,
                trace=trace,
                collar=collar,
                holonomy=holonomy,
                stress=stress,
                w_eff=float(w_eff),
            )
            rows.append(row)
    return rows


def _run_sync_row(
    *,
    run_path: Path,
    manifest: dict[str, Any],
    trace: list[dict[str, float]],
    collar: dict[str, Any],
    holonomy: dict[str, Any],
    stress: dict[str, Any],
    w_eff: float,
) -> dict[str, Any]:
    phi_fit = _fit_decay(trace, "phi")
    cycles = int(manifest.get("cycles") or (max(row.get("cycle", 0.0) for row in trace) + 1))
    gamma_sync = phi_fit.get("gamma_per_cycle")
    gamma_sync_over_H = float(gamma_sync * cycles) if isinstance(gamma_sync, (int, float)) else None
    sync_depth = float(gamma_sync * max(cycles - 1, 1)) if isinstance(gamma_sync, (int, float)) else None
    holonomy_fit = _screen_s3_defect_decay_proxy(holonomy)
    curvature_gate = s3_holonomy_spatial_curvature_gate(holonomy)
    finite_collar_parent_grade = bool(
        ((stress.get("physical_prediction_readiness", {}) or {}).get("checks", {}) or {}).get(
            "finite_collar_parent_theorem_grade", False
        )
    )
    cmi = _float_or_none(collar.get("median_epsilon_cmi"))
    same_boundary_selector_established = False
    low_k_gap_established = False
    horizon_ready = bool(same_boundary_selector_established or low_k_gap_established)
    exact_curvature_ready = False
    scale_spectrum_ready = False
    return {
        "run_id": manifest.get("run_id", run_path.name),
        "run_path": str(run_path),
        "patch_count": int(manifest.get("patch_count", 0)),
        "cycles": cycles,
        "repair_phi_decay": phi_fit,
        "Gamma_sync_over_H_proxy": gamma_sync_over_H,
        "C_sigma_depth_proxy": sync_depth,
        "collar_median_epsilon_cmi": cmi,
        "fawzi_renner_recovery_bound_proxy": 2.0 * math.sqrt(max(cmi, 0.0)) if cmi is not None else None,
        "holonomy_defect_fraction_final": _float_or_none(holonomy.get("defect_fraction")),
        "screen_s3_defect_decay_proxy": holonomy_fit,
        "s3_holonomy_spatial_curvature_gate": curvature_gate,
        "w_eff": float(w_eff),
        "flatness_repair_margin_proxy": None,
        "flatness_holonomy_damping_ready": False,
        "spatial_curvature_exact_selection_ready": exact_curvature_ready,
        "same_boundary_selector_established": same_boundary_selector_established,
        "low_k_repair_gap_established": low_k_gap_established,
        "horizon_synchronization_ready": horizon_ready,
        "finite_collar_parent_theorem_grade": finite_collar_parent_grade,
        "screen_spectrum_theorem_target": screen_spectrum_prediction(),
        "flat_sector_selection_target": flat_sector_selection_report(),
        "scale_invariant_screen_spectrum_ready": scale_spectrum_ready,
        "inflation_replacement_ready": False,
        "missing_gates": [
            name
            for name, passed in {
                "same_boundary_selector_or_low_k_gap": horizon_ready,
                "spatial_levi_civita_curvature_receipt_or_cmh": exact_curvature_ready,
                "finite_collar_parent_theorem_grade": finite_collar_parent_grade,
                "scale_spectrum_theta_sigma_derived": scale_spectrum_ready,
                "Boltzmann_transfer_likelihood": False,
            }.items()
            if not passed
        ],
    }


def _fit_decay(trace: list[dict[str, float]], key: str) -> dict[str, Any]:
    cycles = np.asarray([float(row.get("cycle", index)) for index, row in enumerate(trace)], dtype=float)
    values = np.asarray([float(row.get(key, 0.0)) for row in trace], dtype=float)
    mask = values > 0.0
    if int(np.sum(mask)) < 2:
        return {"available": False, "reason": "not_enough_positive_values"}
    x = cycles[mask]
    y = np.log(np.maximum(values[mask], 1.0e-300))
    slope, intercept = np.polyfit(x, y, 1)
    gamma = float(max(0.0, -slope))
    y_hat = slope * x + intercept
    residual = y - y_hat
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return {
        "available": True,
        "gamma_per_cycle": gamma,
        "log_slope": float(slope),
        "fit_cycle_min": float(np.min(x)),
        "fit_cycle_max": float(np.max(x)),
        "fit_point_count": int(x.size),
        "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 1.0e-300 else 1.0,
        "initial_value": float(values[0]) if values.size else None,
        "final_value": float(values[-1]) if values.size else None,
    }


def _screen_s3_defect_decay_proxy(holonomy: dict[str, Any]) -> dict[str, Any]:
    timeline = holonomy.get("timeline") or holonomy.get("timeline_trace") or []
    if isinstance(timeline, list) and len(timeline) >= 2:
        rows = []
        for item in timeline:
            cycle = _float_or_none(item.get("cycle"))
            fraction = _float_or_none(item.get("defect_fraction"))
            if cycle is not None and fraction is not None:
                rows.append({"cycle": cycle, "defect_fraction": fraction})
        fit = _fit_decay(rows, "defect_fraction") if rows else {"available": False}
        if fit.get("available"):
            cycles = max(row["cycle"] for row in rows) + 1.0
            return fit | {
                "screen_s3_defect_decay_per_H_proxy": float(fit["gamma_per_cycle"] * cycles),
                "structure_group": "S3",
                "geometric_connection": False,
                "spatial_levi_civita_interpretation": False,
            }
    final_fraction = _float_or_none(holonomy.get("defect_fraction"))
    return {
        "available": False,
        "reason": "holonomy_timeline_not_available",
        "final_defect_fraction": final_fraction,
        "screen_s3_defect_decay_per_H_proxy": None,
        "structure_group": "S3",
        "geometric_connection": False,
        "spatial_levi_civita_interpretation": False,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gamma = [row.get("Gamma_sync_over_H_proxy") for row in rows if isinstance(row.get("Gamma_sync_over_H_proxy"), (int, float))]
    depth = [row.get("C_sigma_depth_proxy") for row in rows if isinstance(row.get("C_sigma_depth_proxy"), (int, float))]
    cmi = [row.get("collar_median_epsilon_cmi") for row in rows if isinstance(row.get("collar_median_epsilon_cmi"), (int, float))]
    return {
        "mean_Gamma_sync_over_H_proxy": float(fmean(gamma)) if gamma else None,
        "mean_C_sigma_depth_proxy": float(fmean(depth)) if depth else None,
        "mean_collar_median_epsilon_cmi": float(fmean(cmi)) if cmi else None,
        "flatness_holonomy_damping_ready_count": 0,
        "spatial_curvature_exact_selection_ready_count": sum(
            bool(row.get("spatial_curvature_exact_selection_ready")) for row in rows
        ),
        "horizon_synchronization_ready_count": sum(bool(row.get("horizon_synchronization_ready")) for row in rows),
        "finite_collar_parent_theorem_grade_count": sum(bool(row.get("finite_collar_parent_theorem_grade")) for row in rows),
        "inflation_replacement_ready_count": sum(bool(row.get("inflation_replacement_ready")) for row in rows),
        "all_runs_ready": False,
    }


def _read_trace(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: float(value) for key, value in row.items() if value not in {None, ""}})
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    screen = report.get("screen_spectrum_prediction", {})
    flat = report.get("flat_sector_selection", {})
    lines = [
        "# OPH Synchronization / Inflation Diagnostic",
        "",
        report["claim_boundary"],
        "",
        "## Summary",
        "",
        f"- run count: {report['run_count']}",
        f"- curvature status: {flat.get('status', 'n/a')}",
        f"- flat selector Omega_K: {_fmt(flat.get('selected_Omega_K'))}",
        f"- screen-spectrum n_s = 1 - P/48: {_fmt(screen.get('n_s'))}",
        f"- screen-spectrum A_zeta: {screen.get('A_zeta') if screen.get('A_zeta') is not None else 'pending lift receipt'}",
        f"- mean Gamma_sync/H proxy: {_fmt(aggregate.get('mean_Gamma_sync_over_H_proxy'))}",
        f"- mean C_sigma depth proxy: {_fmt(aggregate.get('mean_C_sigma_depth_proxy'))}",
        f"- exact curvature selection ready count: {aggregate['spatial_curvature_exact_selection_ready_count']}/{report['run_count']}",
        f"- horizon synchronization ready count: {aggregate['horizon_synchronization_ready_count']}/{report['run_count']}",
        f"- finite collar parent theorem-grade count: {aggregate['finite_collar_parent_theorem_grade_count']}/{report['run_count']}",
        f"- inflation replacement ready: {report['inflation_replacement_ready']}",
        "",
        "## Missing Gates",
        "",
    ]
    missing = sorted({gate for row in report["rows"] for gate in row.get("missing_gates", [])})
    lines.extend(f"- `{gate}`" for gate in missing)
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `sync_inflation_report.json`",
            "- `sync_inflation_rows.csv`",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.10g}"
    return "n/a"
