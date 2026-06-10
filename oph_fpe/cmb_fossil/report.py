from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.cmb_fossil.screen_covariance import (
    apply_low_l_repair_suppression,
    apply_parity_term,
    cl_oph_screen,
)


def cmb_fossil_bridge_report(params: dict[str, Any], benchmark_score: dict[str, Any]) -> dict[str, Any]:
    """Create a claim-bounded report for the OPH-CET CMB fossil bridge."""

    return {
        "mode": "oph_cmb_fossil_bridge_diagnostic",
        "receipt": "OPH_CMB_FOSSIL_BRIDGE_DIAGNOSTIC",
        "claim_level": "continuation_effective_theory",
        "physical_cmb_prediction": False,
        "physical_matter_power_prediction": False,
        "bulk_required": False,
        "parameters": dict(params),
        "benchmark_score": dict(benchmark_score),
        "claim_boundary": (
            "OPH-CET CMB fossil bridge: maps an analytic observer-consensus screen covariance "
            "to a primordial modulation. This is not a recovered-chain CMB prediction until "
            "OPH anomaly kernels and Boltzmann source terms are derived."
        ),
    }


def write_cmb_fossil_bridge_report(
    out_dir: Path,
    *,
    planck_tt: Path | None = None,
    ell_max: int = 2600,
    eta: float = 0.0351588569692228,
    q_ir: float = 0.25,
    ell_ir: float = 32.0,
    eps_p: float = 0.0,
    ell_p: float = 30.0,
    ell_cap: float = 3000.0,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ell = np.arange(2, int(ell_max) + 1, dtype=float)
    base_cl = cl_oph_screen(ell, A=1.0, eta=0.0, ell_cap=float(ell_cap))
    oph_cl = cl_oph_screen(ell, A=1.0, eta=float(eta), ell_cap=float(ell_cap))
    oph_cl = apply_low_l_repair_suppression(oph_cl, ell, q_ir=float(q_ir), ell_ir=float(ell_ir))
    oph_cl = apply_parity_term(oph_cl, ell, eps_p=float(eps_p), ell_p=float(ell_p))
    base_dl = _dl_from_cl(ell, base_cl)
    oph_dl = _dl_from_cl(ell, oph_cl)
    benchmark = _read_planck_tt(planck_tt) if planck_tt is not None else None
    score = _score_against_benchmark(ell, oph_dl, benchmark) if benchmark is not None else {}
    params = {
        "eta": float(eta),
        "q_ir": float(q_ir),
        "ell_ir": float(ell_ir),
        "eps_p": float(eps_p),
        "ell_p": float(ell_p),
        "ell_cap": float(ell_cap),
        "ell_max": int(ell_max),
        "planck_tt": str(planck_tt) if planck_tt is not None else None,
    }
    report = cmb_fossil_bridge_report(params, score)
    report["bridge_tables"] = {
        "tt_curve": "cmb_fossil_bridge_tt.csv",
        "params": "cmb_fossil_bridge_params.json",
    }
    (out / "cmb_fossil_bridge_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "cmb_fossil_bridge_params.json").write_text(json.dumps(params, indent=2, default=str), encoding="utf-8")
    with (out / "cmb_fossil_bridge_tt.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ell", "base_D_ell", "oph_fossil_D_ell"])
        writer.writeheader()
        for left, base, value in zip(ell, base_dl, oph_dl, strict=False):
            writer.writerow(
                {
                    "ell": f"{float(left):.8f}",
                    "base_D_ell": f"{float(base):.16e}",
                    "oph_fossil_D_ell": f"{float(value):.16e}",
                }
            )
    return report


def _dl_from_cl(ell: np.ndarray, cl: np.ndarray) -> np.ndarray:
    ell_values = np.asarray(ell, dtype=float)
    return ell_values * (ell_values + 1.0) * np.asarray(cl, dtype=float) / (2.0 * math.pi)


def _read_planck_tt(path: Path | None) -> dict[str, np.ndarray] | None:
    if path is None or not Path(path).exists():
        return None
    values = np.loadtxt(path, comments="#")
    if values.ndim != 2 or values.shape[1] < 2:
        return None
    return {"ell": values[:, 0].astype(float), "D_ell": values[:, 1].astype(float)}


def _score_against_benchmark(
    ell: np.ndarray,
    model_dl: np.ndarray,
    benchmark: dict[str, np.ndarray] | None,
) -> dict[str, Any]:
    if not benchmark:
        return {}
    bench_ell = np.asarray(benchmark["ell"], dtype=float)
    bench_dl = np.asarray(benchmark["D_ell"], dtype=float)
    interp = np.interp(bench_ell, np.asarray(ell, dtype=float), np.asarray(model_dl, dtype=float))
    scale = float(np.dot(bench_dl, interp) / max(float(np.dot(interp, interp)), 1e-300))
    fitted = scale * interp
    centered_bench = bench_dl - float(np.mean(bench_dl))
    centered_fit = fitted - float(np.mean(fitted))
    corr = float(
        np.dot(centered_bench, centered_fit)
        / max(float(np.linalg.norm(centered_bench) * np.linalg.norm(centered_fit)), 1e-300)
    )
    rmse = float(np.sqrt(np.mean((bench_dl - fitted) ** 2)))
    norm_rmse = rmse / max(float(np.sqrt(np.mean(bench_dl**2))), 1e-300)
    return {
        "benchmark": "Planck_TT_binned" if bench_ell.size else None,
        "benchmark_count": int(bench_ell.size),
        "shape_correlation": corr,
        "normalized_rmse": norm_rmse,
        "amplitude_scale": scale,
        "claim_boundary": "Shape/amplitude-rescaled OPH-CET fossil diagnostic, not official Planck likelihood.",
    }
