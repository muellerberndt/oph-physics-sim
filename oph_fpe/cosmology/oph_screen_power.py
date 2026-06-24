from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from oph_fpe.claims import PROXY, SCREEN_PROXY_CMB_RECEIPT, with_claim_metadata
from oph_fpe.cosmology.screen_spectrum import screen_cl
from oph_fpe.cosmology.screen_to_primordial import screen_to_radial_lift_report


PLANCK_ETA_R_TARGET = 0.035
PLANCK_ETA_R_SIGMA = 0.004
DEFAULT_D_STAR_MPC = 13_800.0
DEFAULT_A_S = 2.1e-9
DEFAULT_K0_MPC = 0.05


@dataclass(frozen=True)
class OPHScreenPowerParams:
    """Scalar OPH screen covariance parameters.

    `eta_R = 1 - n_s` is the red-tilt convention from the CMB notes. `N_cap_eff`
    is deliberately not called `N_eff`, which is reserved for relativistic
    species in cosmology.
    """

    A_chi: float = 1.0
    eta_R: float = PLANCK_ETA_R_TARGET
    mu: float = 0.0
    ell_cap: float | None = None
    N_cap_eff: float | None = None
    q_IR: float = 0.0
    ell_IR: float = 6.0
    epsilon_parity: float = 0.0
    ell_parity: float = 8.0

    @property
    def n_s_proxy(self) -> float:
        return 1.0 - float(self.eta_R)

    def as_jsonable(self) -> dict[str, Any]:
        return asdict(self) | {"n_s_proxy": self.n_s_proxy}


def C_ell_oph(ell: np.ndarray | list[float], params: OPHScreenPowerParams) -> np.ndarray:
    ell_arr = np.asarray(ell, dtype=float)
    safe_ell = np.maximum(ell_arr, 1.0)
    base = np.asarray(
        screen_cl(
            safe_ell,
            A_q=float(params.A_chi),
            theta=float(params.eta_R),
            mu=float(params.mu),
            model="FRACTIONAL_LAPLACIAN_ASYMPTOTIC",
        ),
        dtype=float,
    )
    window = np.ones_like(safe_ell)
    if params.ell_cap is not None and float(params.ell_cap) > 0.0 and math.isfinite(float(params.ell_cap)):
        window = np.exp(-(safe_ell * (safe_ell + 1.0)) / (float(params.ell_cap) ** 2))
    fir = 1.0 - float(params.q_IR) * np.exp(
        -(safe_ell * (safe_ell + 1.0)) / max(float(params.ell_IR) * (float(params.ell_IR) + 1.0), 1.0e-12)
    )
    parity = 1.0 + float(params.epsilon_parity) * ((-1.0) ** np.rint(safe_ell)) * np.exp(
        -safe_ell / max(float(params.ell_parity), 1.0e-12)
    )
    cap_noise = 0.0
    if params.N_cap_eff is not None and float(params.N_cap_eff) > 0.0:
        cap_noise = float(params.A_chi) / float(params.N_cap_eff)
    return base * window * np.maximum(fir, 0.0) * np.maximum(parity, 0.0) + cap_noise


def D_ell_from_C_ell(ell: np.ndarray | list[float], c_ell: np.ndarray | list[float]) -> np.ndarray:
    ell_arr = np.asarray(ell, dtype=float)
    c_arr = np.asarray(c_ell, dtype=float)
    return ell_arr * (ell_arr + 1.0) * c_arr / (2.0 * math.pi)


def F_oph_k(
    k_mpc: np.ndarray | list[float],
    params: OPHScreenPowerParams,
    *,
    D_star_mpc: float = DEFAULT_D_STAR_MPC,
) -> dict[str, np.ndarray]:
    """Isotropic OPH correction for a scalar primordial spectrum.

    Parity and BipoSH corrections are intentionally excluded; those are angular
    covariance effects and belong in a_lm-space diagnostics.
    """

    k_arr = np.asarray(k_mpc, dtype=float)
    ell = np.maximum(k_arr * float(D_star_mpc), 1.0)
    f_ir = 1.0 - float(params.q_IR) * np.exp(
        -(ell * (ell + 1.0)) / max(float(params.ell_IR) * (float(params.ell_IR) + 1.0), 1.0e-12)
    )
    f_cap = np.ones_like(ell)
    if params.ell_cap is not None and float(params.ell_cap) > 0.0 and math.isfinite(float(params.ell_cap)):
        f_cap = np.exp(-(ell * (ell + 1.0)) / (float(params.ell_cap) ** 2))
    total = np.maximum(f_ir, 0.0) * f_cap
    return {"ell_proxy": ell, "F_IR": f_ir, "F_cap": f_cap, "F_OPH": total}


def primordial_power_oph(
    k_mpc: np.ndarray | list[float],
    params: OPHScreenPowerParams,
    *,
    A_s: float = DEFAULT_A_S,
    k0_mpc: float = DEFAULT_K0_MPC,
    D_star_mpc: float = DEFAULT_D_STAR_MPC,
) -> dict[str, np.ndarray]:
    """Return the legacy ell=kD primordial-table scaffold.

    The default amplitude is a diagnostic reference, not a derived OPH
    primordial amplitude. A passed screen-to-primordial lift receipt is required
    before this can be promoted to a primordial spectrum.
    """

    k_arr = np.asarray(k_mpc, dtype=float)
    correction = F_oph_k(k_arr, params, D_star_mpc=D_star_mpc)
    base = float(A_s) * (np.maximum(k_arr, 1.0e-30) / float(k0_mpc)) ** (-float(params.eta_R))
    return correction | {
        "k_mpc": k_arr,
        "P_R": base * correction["F_OPH"],
        "P_R_base": base,
        "A_s_source": "diagnostic_reference_not_derived",
        "ell_equals_kD_scaffold_only": True,
    }


def screen_power_fit_from_spectrum(
    spectrum: list[dict[str, Any]],
    *,
    field_name: str,
    point_count: int | None = None,
    ell_min: float = 20.0,
    ell_max: float | None = None,
) -> dict[str, Any]:
    pairs = _spectrum_pairs(spectrum)
    fit_pairs = [(ell, dell) for ell, _cell, dell in pairs if ell >= ell_min and dell > 0.0]
    if ell_max is not None:
        fit_pairs = [(ell, dell) for ell, dell in fit_pairs if ell <= float(ell_max)]
    if len(fit_pairs) < 4:
        fit_pairs = [(ell, dell) for ell, _cell, dell in pairs if ell >= 2.0 and dell > 0.0]
    if len(fit_pairs) < 4:
        return {
            "field": field_name,
            "fit_available": False,
            "reason": "not_enough_positive_multipoles",
            "point_count": point_count,
        }

    ell = np.asarray([item[0] for item in fit_pairs], dtype=float)
    dell = np.asarray([item[1] for item in fit_pairs], dtype=float)
    x = 0.5 * np.log(np.maximum(ell * (ell + 1.0), 1.0e-30))
    y = np.log(np.maximum(dell, 1.0e-300))
    slope, intercept = np.polyfit(x, y, 1)
    eta_R = float(-slope)
    A_chi = float(2.0 * math.pi * math.exp(intercept))
    low_ir = _low_ell_suppression_proxy(pairs, eta_R=eta_R, A_chi=A_chi)
    parity = _parity_proxy(pairs)
    params = OPHScreenPowerParams(
        A_chi=A_chi,
        eta_R=eta_R,
        mu=0.0,
        ell_cap=_default_ell_cap(point_count),
        N_cap_eff=float(point_count) if point_count else None,
        q_IR=low_ir["q_IR_proxy"],
        ell_IR=low_ir.get("ell_IR_proxy", 6.0),
        epsilon_parity=parity["epsilon_parity_proxy"],
        ell_parity=8.0,
    )
    return {
        "field": field_name,
        "fit_available": True,
        "point_count": point_count,
        "fit_ell_min": float(min(ell)),
        "fit_ell_max": float(max(ell)),
        "fit_multipole_count": int(ell.size),
        "log_D_ell_slope": float(slope),
        "eta_R_estimate": eta_R,
        "n_s_proxy": float(1.0 - eta_R),
        "A_chi_estimate": A_chi,
        "planck_eta_R_target": PLANCK_ETA_R_TARGET,
        "planck_eta_R_sigma": PLANCK_ETA_R_SIGMA,
        "eta_R_planck_target_delta": float(eta_R - PLANCK_ETA_R_TARGET),
        "within_planck_eta_R_1sigma": bool(abs(eta_R - PLANCK_ETA_R_TARGET) <= PLANCK_ETA_R_SIGMA),
        "parameters": params.as_jsonable(),
        "low_ell_suppression": low_ir,
        "parity": parity,
        "claim_boundary": (
            "Finite screen C_l power-law diagnostic. eta_R is estimated from the simulator field and "
            "compared with the Planck scalar-tilt target, but this is not a Planck likelihood and not "
            "a physical CMB prediction."
        ),
    }


def write_oph_screen_power_report(
    run_dirs: list[Path],
    out_dir: Path,
    *,
    field_names: list[str] | None = None,
    ell_min: float = 20.0,
    ell_max: float | None = None,
    reference_mode: str = "auto",
    primordial_k_count: int = 256,
    primordial_k_min: float = 1.0e-4,
    primordial_k_max: float = 1.0,
) -> dict[str, Any]:
    rows = collect_screen_power_runs(run_dirs, field_names=field_names, ell_min=ell_min, ell_max=ell_max)
    aggregate = _aggregate_rows(rows)
    reference_params, reference_source, simulator_screen_reference_ready = _reference_params(
        rows,
        reference_mode=reference_mode,
    )
    primordial_rows = _primordial_rows(
        reference_params,
        k_count=int(primordial_k_count),
        k_min=float(primordial_k_min),
        k_max=float(primordial_k_max),
    )
    radial_lift = _diagnostic_radial_lift_artifact(
        reference_params,
        primordial_rows,
        simulator_screen_reference_ready=simulator_screen_reference_ready,
    )
    report = {
        "mode": "oph_screen_power_effective_theory_v0",
        "source_run_count": len({row["run_path"] for row in rows}),
        "fit_row_count": len(rows),
        "fit_rows": rows,
        "aggregate": aggregate,
        "reference_mode": str(reference_mode),
        "simulator_screen_reference_ready": simulator_screen_reference_ready,
        "simulator_primordial_reference_ready": False,
        "screen_reference_source": reference_source,
        "primordial_reference_source": f"ell_kD_scaffold_from_{reference_source}_not_lift_receipt",
        "reference_screen_parameters": reference_params.as_jsonable(),
        "primordial_bridge": {
            "status": "ell_kD_diagnostic_scaffold_emitted",
            "A_s_reference": DEFAULT_A_S,
            "A_s_source": "diagnostic_reference_not_derived",
            "k0_mpc": DEFAULT_K0_MPC,
            "D_star_mpc": DEFAULT_D_STAR_MPC,
            "row_count": len(primordial_rows),
            "simulator_eta_R_ready": simulator_screen_reference_ready,
            "reference_source": reference_source,
            "ell_equals_kD_scaffold_only": True,
            "SCREEN_TO_RADIAL_LIFT_RECEIPT": radial_lift["SCREEN_TO_RADIAL_LIFT_RECEIPT"],
            "SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT": False,
            "excludes": ["parity_envelope", "BipoSH_off_diagonal_covariance"],
            "claim_boundary": (
                "Exports the old ell=kD isotropic scalar scaffold for diagnostics only. A primordial "
                "curvature spectrum requires the exact Bessel/gamma lift receipt; parity and BipoSH "
                "effects remain angular covariance diagnostics."
            ),
        },
        "screen_to_radial_lift": radial_lift,
        "screen_spectrum_derived": False,
        "primordial_spectrum_derived": False,
        "SCREEN_TO_RADIAL_LIFT_RECEIPT": radial_lift["SCREEN_TO_RADIAL_LIFT_RECEIPT"],
        "SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT": False,
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "OPH screen/freezeout covariance effective-theory diagnostic. It estimates screen parameters "
            "from finite C_l receipts and exports only an ell=kD primordial-table scaffold. It is not a "
            "derived primordial spectrum until a screen-to-radial lift receipt passes, and it is not a "
            "physical TT/TE/EE prediction until transfer and likelihood gates pass."
        ),
    }
    report = with_claim_metadata(
        report,
        claim_level=PROXY,
        receipt=SCREEN_PROXY_CMB_RECEIPT,
        physical_claim=False,
        observable_id="oph_screen_freezeout_covariance",
        fit_objective="finite_screen_eta_R_and_primordial_bridge_scaffold",
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "oph_screen_power_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "screen_to_radial_lift_report.json").write_text(
        json.dumps(radial_lift, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "oph_screen_power_report.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_csv(out / "oph_screen_power_fit_rows.csv", rows)
    _write_csv(out / "oph_primordial_power_table.csv", primordial_rows)
    _write_primordial_txt(out / "oph_primordial_power_CLASS_CAMB.txt", primordial_rows)
    return report


def _diagnostic_radial_lift_artifact(
    params: OPHScreenPowerParams,
    primordial_rows: list[dict[str, Any]],
    *,
    simulator_screen_reference_ready: bool,
) -> dict[str, Any]:
    ell_count = max(8, min(64, len(primordial_rows)))
    ell = np.arange(2, 2 + ell_count, dtype=float)
    k = np.asarray([float(row["k_mpc"]) for row in primordial_rows], dtype=float)
    prior = np.asarray([float(row["P_R"]) for row in primordial_rows], dtype=float)
    screen_cl = C_ell_oph(ell, params)
    artifact = screen_to_radial_lift_report(
        ell=ell,
        screen_cl=screen_cl,
        k=k,
        radial_prior_delta_zeta2=prior,
        background_geometry_branch="FLAT_ASSUMED",
        radius=DEFAULT_D_STAR_MPC,
        radial_prior_declared=False,
        source_only_screen_scalar=False,
        theorem_gate=False,
        total_stress_closure_receipt=False,
        single_clock_normal_form_receipt=False,
        entropy_repair_gap_receipt=False,
        curvature_freezeout_receipt=False,
        adiabatic_mode_receipt=False,
        isocurvature_bound_receipt=False,
        primordial_phase_coherence_receipt=False,
        no_observation_ancestry_receipt=False,
    )
    artifact["diagnostic_source"] = {
        "screen_reference_ready": bool(simulator_screen_reference_ready),
        "screen_parameters": params.as_jsonable(),
        "radial_prior_source": "legacy_ell_kD_scaffold_not_declared_source_prior",
    }
    return artifact


def collect_screen_power_runs(
    run_dirs: list[Path],
    *,
    field_names: list[str] | None = None,
    ell_min: float = 20.0,
    ell_max: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates: dict[Path, tuple[Path, dict[str, Any]]] = {}
    for root in run_dirs:
        for path in sorted(Path(root).glob("**/cl_comparison_report.json")):
            cl_report = _read_json(path)
            run_path = _source_run_root_for_cl(path).resolve()
            previous = candidates.get(run_path)
            if previous is None or int(cl_report.get("ell_max", 0)) > int(previous[1].get("ell_max", 0)):
                candidates[run_path] = (path, cl_report)
    for run_path, (cl_path, cl_report) in sorted(candidates.items(), key=lambda item: str(item[0])):
        manifest = _read_json(run_path / "manifest.json")
        point_count = int(cl_report.get("point_count") or manifest.get("patch_count") or 0) or None
        fields = cl_report.get("fields", {}) or {}
        selected = field_names or list(fields.keys())
        for field in selected:
            spectrum = (fields.get(field, {}) or {}).get("spectrum", [])
            fit = screen_power_fit_from_spectrum(
                list(spectrum),
                field_name=str(field),
                point_count=point_count,
                ell_min=float(ell_min),
                ell_max=ell_max,
            )
            fit.update(
                {
                    "run_path": str(run_path),
                    "cl_report_path": str(cl_path),
                    "run_id": manifest.get("run_id", run_path.name),
                    "patch_count": point_count,
                    "ell_max": cl_report.get("ell_max"),
                }
            )
            rows.append(fit)
    return rows


def _spectrum_pairs(spectrum: list[dict[str, Any]]) -> list[tuple[float, float, float]]:
    pairs = []
    for row in spectrum:
        ell = float(row.get("ell", 0.0))
        c_ell = float(row.get("C_ell", 0.0))
        d_ell = float(row.get("D_ell", 0.0))
        if d_ell == 0.0 and c_ell > 0.0:
            d_ell = float(D_ell_from_C_ell([ell], [c_ell])[0])
        if c_ell == 0.0 and d_ell > 0.0 and ell > 0.0:
            c_ell = float(d_ell * 2.0 * math.pi / (ell * (ell + 1.0)))
        pairs.append((ell, c_ell, d_ell))
    return pairs


def _low_ell_suppression_proxy(pairs: list[tuple[float, float, float]], *, eta_R: float, A_chi: float) -> dict[str, Any]:
    low = [(ell, dell) for ell, _cell, dell in pairs if 2.0 <= ell <= 30.0 and dell > 0.0]
    if len(low) < 3:
        return {"available": False, "q_IR_proxy": 0.0, "ell_IR_proxy": 6.0}
    values = np.asarray([dell for _ell, dell in low], dtype=float)
    ell = np.asarray([item[0] for item in low], dtype=float)
    baseline = (A_chi / (2.0 * math.pi)) * np.maximum(ell * (ell + 1.0), 1.0e-30) ** (-eta_R / 2.0)
    ratios = values / np.maximum(baseline, 1.0e-300)
    finite = np.isfinite(ratios) & (ratios > 0.0)
    if int(np.sum(finite)) < 3:
        return {"available": False, "q_IR_proxy": 0.0, "ell_IR_proxy": 6.0}
    ell = ell[finite]
    ratios = np.clip(ratios[finite], 0.0, 3.0)
    ratio = float(np.median(ratios))
    legacy_q = float(max(0.0, min(1.0, 1.0 - ratio)))
    fit = _fit_ir_envelope(ell, ratios)
    return {
        "available": True,
        "median_low_ell_to_powerlaw_ratio": ratio,
        "q_IR_proxy": fit.get("q_IR_proxy", legacy_q),
        "ell_IR_proxy": fit.get("ell_IR_proxy", 6.0),
        "legacy_q_IR_proxy_ell2_30_median": legacy_q,
        "fit_rmse": fit.get("fit_rmse"),
        "fit_point_count": int(ell.size),
        "fit_available": bool(fit.get("fit_available", False)),
        "ell_range": [2, 30],
    }


def _fit_ir_envelope(ell: np.ndarray, ratios: np.ndarray) -> dict[str, Any]:
    ell = np.asarray(ell, dtype=float)
    ratios = np.asarray(ratios, dtype=float)
    if ell.size < 3 or ratios.size != ell.size:
        return {"fit_available": False}

    def model(q_ir: float, ell_ir: float) -> np.ndarray:
        denom = max(float(ell_ir) * (float(ell_ir) + 1.0), 1.0e-12)
        return 1.0 - float(q_ir) * np.exp(-(ell * (ell + 1.0)) / denom)

    target = np.clip(ratios, 0.0, 1.5)
    best = None
    best_rmse = float("inf")
    starts = [
        np.array([max(0.0, min(0.9, 1.0 - float(np.median(target)))), 6.0]),
        np.array([0.15, 6.0]),
        np.array([0.25, 33.0]),
        np.array([0.4, 60.0]),
    ]
    for start in starts:
        result = least_squares(
            lambda x: model(float(x[0]), float(x[1])) - target,
            start,
            bounds=([0.0, 2.0], [0.95, 120.0]),
            max_nfev=400,
        )
        residual = model(float(result.x[0]), float(result.x[1])) - target
        rmse = float(np.sqrt(np.mean(residual * residual)))
        if rmse < best_rmse:
            best_rmse = rmse
            best = result
    if best is None:
        return {"fit_available": False}
    return {
        "fit_available": True,
        "q_IR_proxy": float(best.x[0]),
        "ell_IR_proxy": float(best.x[1]),
        "fit_rmse": best_rmse,
    }


def _parity_proxy(pairs: list[tuple[float, float, float]]) -> dict[str, Any]:
    low = [(int(round(ell)), dell) for ell, _cell, dell in pairs if 2.0 <= ell <= 30.0 and dell > 0.0]
    even = [dell for ell, dell in low if ell % 2 == 0]
    odd = [dell for ell, dell in low if ell % 2 == 1]
    if not even or not odd:
        return {"available": False, "epsilon_parity_proxy": 0.0}
    even_med = float(np.median(even))
    odd_med = float(np.median(odd))
    eps = (even_med - odd_med) / max(even_med + odd_med, 1.0e-300)
    return {
        "available": True,
        "even_median_D_ell": even_med,
        "odd_median_D_ell": odd_med,
        "epsilon_parity_proxy": float(max(-0.99, min(0.99, eps))),
        "ell_range": [2, 30],
    }


def _default_ell_cap(point_count: int | None) -> float | None:
    if not point_count or point_count <= 0:
        return None
    return float(max(8.0, math.sqrt(float(point_count))))


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row.get("fit_available")]
    by_field: dict[str, list[dict[str, Any]]] = {}
    for row in available:
        by_field.setdefault(str(row.get("field")), []).append(row)
    field_summary = {}
    for field, field_rows in by_field.items():
        eta = [float(row["eta_R_estimate"]) for row in field_rows]
        ns = [float(row["n_s_proxy"]) for row in field_rows]
        field_summary[field] = {
            "count": len(field_rows),
            "median_eta_R": float(median(eta)),
            "median_n_s_proxy": float(median(ns)),
            "median_eta_R_delta_to_planck": float(median(eta) - PLANCK_ETA_R_TARGET),
            "within_planck_eta_R_1sigma_count": sum(bool(row.get("within_planck_eta_R_1sigma")) for row in field_rows),
        }
    best_field = None
    if field_summary:
        best_field = min(
            field_summary,
            key=lambda name: abs(float(field_summary[name]["median_eta_R_delta_to_planck"])),
        )
    return {
        "available_fit_count": len(available),
        "field_summary": field_summary,
        "best_planck_eta_diagnostic_field": best_field,
        "planck_eta_R_target": PLANCK_ETA_R_TARGET,
        "planck_eta_R_sigma": PLANCK_ETA_R_SIGMA,
    }


def _reference_params(rows: list[dict[str, Any]], *, reference_mode: str = "auto") -> tuple[OPHScreenPowerParams, str, bool]:
    available = [row for row in rows if row.get("fit_available")]
    if not available:
        return OPHScreenPowerParams(), "phenomenological_planck_eta_target_no_simulator_fit", False
    mode = str(reference_mode or "auto")
    if mode not in {"auto", "planck-fallback", "simulator-best"}:
        raise ValueError(f"unknown screen-power reference_mode: {mode}")
    best = min(available, key=lambda row: abs(float(row.get("eta_R_estimate", 0.0)) - PLANCK_ETA_R_TARGET))
    params = dict(best.get("parameters", {}) or {})
    allowed = {field.name for field in OPHScreenPowerParams.__dataclass_fields__.values()}
    simulator_params = OPHScreenPowerParams(**{key: value for key, value in params.items() if key in allowed})
    simulator_ready = bool(
        math.isfinite(float(simulator_params.eta_R))
        and 0.0 <= float(simulator_params.eta_R) <= 0.2
        and bool(best.get("within_planck_eta_R_1sigma", False))
    )
    if mode == "simulator-best":
        source = "simulator_eta_R_estimate" if simulator_ready else "simulator_eta_R_diagnostic_outside_planck_target"
        return simulator_params, source, simulator_ready
    if simulator_ready:
        return simulator_params, "simulator_eta_R_estimate", True
    fallback = OPHScreenPowerParams(
        A_chi=float(simulator_params.A_chi) if math.isfinite(float(simulator_params.A_chi)) else 1.0,
        eta_R=PLANCK_ETA_R_TARGET,
        mu=float(simulator_params.mu),
        ell_cap=simulator_params.ell_cap,
        N_cap_eff=simulator_params.N_cap_eff,
        q_IR=max(0.0, min(1.0, float(simulator_params.q_IR))),
        ell_IR=float(simulator_params.ell_IR),
        epsilon_parity=0.0,
        ell_parity=float(simulator_params.ell_parity),
    )
    return fallback, "phenomenological_planck_eta_target_due_to_invalid_simulator_tilt", False


def _source_run_root_for_cl(cl_path: Path) -> Path:
    path = Path(cl_path).resolve()
    for parent in [path.parent, *path.parents]:
        if (parent / "manifest.json").exists():
            return parent
    return path.parent


def _primordial_rows(
    params: OPHScreenPowerParams,
    *,
    k_count: int,
    k_min: float,
    k_max: float,
) -> list[dict[str, Any]]:
    k = np.geomspace(float(k_min), float(k_max), int(k_count))
    power = primordial_power_oph(k, params)
    return [
        {
            "k_mpc": float(power["k_mpc"][index]),
            "ell_proxy": float(power["ell_proxy"][index]),
            "P_R": float(power["P_R"][index]),
            "P_R_base": float(power["P_R_base"][index]),
            "F_IR": float(power["F_IR"][index]),
            "F_cap": float(power["F_cap"][index]),
            "F_OPH": float(power["F_OPH"][index]),
            "eta_R": float(params.eta_R),
            "n_s_proxy": float(params.n_s_proxy),
            "A_s_source": str(power["A_s_source"]),
            "ell_equals_kD_scaffold_only": bool(power["ell_equals_kD_scaffold_only"]),
        }
        for index in range(k.size)
    ]


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


def _write_primordial_txt(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# OPH isotropic primordial power scaffold",
        "# columns: k_mpc P_R F_OPH F_IR F_cap ell_proxy",
        "# claim_boundary: not a physical CMB prediction until CAMB/CLASS likelihood is run",
    ]
    for row in rows:
        lines.append(
            f"{row['k_mpc']:.12e} {row['P_R']:.12e} {row['F_OPH']:.12e} "
            f"{row['F_IR']:.12e} {row['F_cap']:.12e} {row['ell_proxy']:.12e}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    best = aggregate.get("best_planck_eta_diagnostic_field")
    lines = [
        "# OPH Screen Power Report",
        "",
        report["claim_boundary"],
        "",
        "## Summary",
        "",
        f"- source run count: {report['source_run_count']}",
        f"- fit row count: {report['fit_row_count']}",
        f"- best Planck-eta diagnostic field: {best or 'n/a'}",
        f"- reference eta_R: {report['reference_screen_parameters']['eta_R']:.10g}",
        f"- reference n_s proxy: {report['reference_screen_parameters']['n_s_proxy']:.10g}",
        f"- screen reference source: {report['screen_reference_source']}",
        f"- simulator screen reference ready: {report['simulator_screen_reference_ready']}",
        f"- simulator primordial reference ready: {report['simulator_primordial_reference_ready']}",
        f"- physical CMB prediction: {report['physical_cmb_prediction']}",
        "",
        "## Field Summaries",
        "",
    ]
    for field, row in sorted((aggregate.get("field_summary") or {}).items()):
        lines.append(
            f"- `{field}`: median eta_R={row['median_eta_R']:.10g}, "
            f"median n_s={row['median_n_s_proxy']:.10g}, "
            f"1sigma count={row['within_planck_eta_R_1sigma_count']}/{row['count']}"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `oph_screen_power_report.json`",
            "- `oph_screen_power_fit_rows.csv`",
            "- `oph_primordial_power_table.csv`",
            "- `oph_primordial_power_CLASS_CAMB.txt`",
            "",
        ]
    )
    return "\n".join(lines)
