from __future__ import annotations

import json
import hashlib
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.cosmology.cmb_compare import load_planck_tt_binned


DEFAULT_K0_MPC = 0.05
DEFAULT_D_STAR_MPC = 13_800.0


@dataclass(frozen=True)
class LambdaCDMParameters:
    H0: float = 67.36
    ombh2: float = 0.02237
    omch2: float = 0.1200
    mnu: float = 0.06
    omk: float = 0.0
    tau: float = 0.0544
    As: float = 2.1e-9
    ns: float = 0.9649

    def as_jsonable(self) -> dict[str, float]:
        return asdict(self)


def camb_lcdm_baseline_report(
    benchmark_rows: list[dict[str, float]],
    *,
    params: LambdaCDMParameters | None = None,
    lmax: int = 2600,
    benchmark_label: str = "Planck2018_TT_binned",
    benchmark_sha256: str | None = None,
) -> dict[str, Any]:
    """Run a standard ΛCDM CAMB baseline and compare TT against a binned table.

    This is the CDM-limit/Boltzmann plumbing regression. It is deliberately not an
    OPH prediction: the parameters are external Planck-like defaults and the OPH
    anomaly sector is not injected into CAMB here.
    """

    params = params or LambdaCDMParameters()
    ell, d_ell_tt = _run_camb_tt(params, lmax=int(lmax))
    comparison = compare_camb_tt_to_benchmark(ell, d_ell_tt, benchmark_rows)
    first_peak = comparison.get("first_peak_ell")
    benchmark_first_peak = comparison.get("benchmark_first_peak_ell")
    peak_ok = (
        isinstance(first_peak, (int, float))
        and isinstance(benchmark_first_peak, (int, float))
        and abs(float(first_peak) - float(benchmark_first_peak)) <= 5.0
    )
    receipt = bool(
        comparison.get("usable")
        and float(comparison.get("shape_correlation", 0.0)) >= 0.995
        and float(comparison.get("amplitude_fit_chi2_per_bin", float("inf"))) <= 2.0
        and peak_ok
    )
    return {
        "mode": "camb_lcdm_baseline_regression",
        "benchmark": {
            "label": benchmark_label,
            "row_count": len(benchmark_rows),
            "ell_min": float(min(row["ell"] for row in benchmark_rows)) if benchmark_rows else None,
            "ell_max": float(max(row["ell"] for row in benchmark_rows)) if benchmark_rows else None,
        },
        "camb": {
            "lmax": int(lmax),
            "cmb_unit": "muK",
            "spectrum": "lensed_total_TT_D_ell",
            "lambda_cdm_parameters": params.as_jsonable(),
        },
        "software": _software_versions(),
        "input_hashes": {
            "benchmark_sha256": benchmark_sha256,
            "params_sha256": _sha256_json(params.as_jsonable()),
        },
        "receipt_thresholds": {
            "shape_correlation_min": 0.995,
            "amplitude_fit_chi2_per_bin_max": 2.0,
            "first_peak_ell_abs_delta_max": 5.0,
        },
        "comparison": comparison,
        "CDM_LIMIT_BOLTZMANN_RECEIPT": receipt,
        "oph_anomaly_module_ready": False,
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "Standard LambdaCDM CAMB baseline compared to the local TT benchmark table. This verifies "
            "Boltzmann plumbing and benchmark alignment only. Parameters are external baseline values; "
            "no OPH rho_A(a), Gamma_rec, B_A(k,a), repair-load source term, or covariance likelihood is "
            "included, so this is not an OPH physical CMB prediction."
        ),
    }


def compare_camb_tt_to_benchmark(
    ell: np.ndarray,
    d_ell_tt: np.ndarray,
    benchmark_rows: list[dict[str, float]],
) -> dict[str, Any]:
    benchmark = [
        row
        for row in benchmark_rows
        if float(row.get("ell", 0.0)) >= 2.0 and float(row.get("D_ell", 0.0)) > 0.0
    ]
    if len(benchmark) < 3 or len(ell) < 3:
        return {"usable": False, "reason": "not_enough_points"}
    bench_ell = np.asarray([float(row["ell"]) for row in benchmark], dtype=float)
    observed = np.asarray([float(row["D_ell"]) for row in benchmark], dtype=float)
    best_fit = np.asarray([float(row.get("best_fit_D_ell", row["D_ell"])) for row in benchmark], dtype=float)
    sigma = np.asarray(
        [
            0.5
            * (
                abs(float(row.get("minus_dD_ell", 0.0)))
                + abs(float(row.get("plus_dD_ell", row.get("minus_dD_ell", 0.0))))
            )
            for row in benchmark
        ],
        dtype=float,
    )
    sigma = np.where(sigma > 1.0e-12, sigma, np.maximum(0.05 * observed, 1.0))
    camb_at_bins = np.interp(bench_ell, np.asarray(ell, dtype=float), np.asarray(d_ell_tt, dtype=float))
    amplitude = _least_squares_amplitude(camb_at_bins, observed, sigma)
    raw_residual = camb_at_bins - observed
    fit_residual = amplitude * camb_at_bins - observed
    best_fit_residual = best_fit - observed
    binned_rows = [
        {
            "ell": float(bench_ell[index]),
            "observed_D_ell": float(observed[index]),
            "sigma_D_ell": float(sigma[index]),
            "camb_D_ell": float(camb_at_bins[index]),
            "amplitude_fit_camb_D_ell": float(amplitude * camb_at_bins[index]),
            "best_fit_column_D_ell": float(best_fit[index]),
        }
        for index in range(len(benchmark))
    ]
    return {
        "usable": True,
        "bin_count": int(len(benchmark)),
        "shape_correlation": _correlation(camb_at_bins, observed),
        "best_fit_column_shape_correlation": _correlation(best_fit, observed),
        "normalized_rmse": _standardized_rmse(camb_at_bins, observed, amplitude=amplitude),
        "best_fit_column_normalized_rmse": _standardized_rmse(best_fit, observed, amplitude=1.0),
        "best_fit_amplitude": amplitude,
        "raw_chi2_per_bin": float(np.mean((raw_residual / sigma) ** 2)),
        "amplitude_fit_chi2_per_bin": float(np.mean((fit_residual / sigma) ** 2)),
        "best_fit_column_chi2_per_bin": float(np.mean((best_fit_residual / sigma) ** 2)),
        "first_peak_ell": _first_peak_ell(bench_ell, camb_at_bins),
        "benchmark_first_peak_ell": _first_peak_ell(bench_ell, observed),
        "mean_absolute_fractional_error": float(np.mean(np.abs(fit_residual) / np.maximum(observed, 1.0e-12))),
        "ell_min": float(np.min(bench_ell)),
        "ell_max": float(np.max(bench_ell)),
        "binned_tt_comparison": binned_rows,
    }


def write_camb_lcdm_baseline_report(
    benchmark_path: Path,
    out_dir: Path,
    *,
    lmax: int = 2600,
    benchmark_label: str = "Planck2018_TT_binned",
    params: LambdaCDMParameters | None = None,
) -> dict[str, Any]:
    benchmark_rows = load_planck_tt_binned(Path(benchmark_path))
    report = camb_lcdm_baseline_report(
        benchmark_rows,
        params=params,
        lmax=int(lmax),
        benchmark_label=benchmark_label,
        benchmark_sha256=_sha256_file(Path(benchmark_path)),
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "camb_lcdm_baseline_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "camb_lcdm_baseline_report.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_spectrum_csv(out_dir / "camb_lcdm_tt_bins.csv", report)
    return report


def oph_screen_camb_report(
    screen_power_report: dict[str, Any],
    benchmark_rows: list[dict[str, float]],
    *,
    baseline_params: LambdaCDMParameters | None = None,
    lmax: int = 2600,
    benchmark_label: str = "Planck2018_TT_binned",
    benchmark_sha256: str | None = None,
) -> dict[str, Any]:
    """Run CAMB from the OPH screen scalar-tilt scaffold.

    This is the first actual CMB-transfer output lane. It is only a physical OPH
    prediction when the input screen report says the scalar tilt was simulator
    derived and all OPH anomaly-sector gates have passed. Current runs usually
    use the Planck-target fallback, so the output remains a transfer scaffold.
    """

    baseline = baseline_params or LambdaCDMParameters()
    screen_params = dict(screen_power_report.get("reference_screen_parameters", {}) or {})
    n_s = float(screen_params.get("n_s_proxy", baseline.ns))
    q_ir = _float_or(screen_params.get("q_IR"), 0.0)
    ell_ir = _float_or(screen_params.get("ell_IR"), 6.0)
    bridge = dict(screen_power_report.get("primordial_bridge", {}) or {})
    A_s = float(bridge.get("A_s", baseline.As))
    params = LambdaCDMParameters(
        H0=baseline.H0,
        ombh2=baseline.ombh2,
        omch2=baseline.omch2,
        mnu=baseline.mnu,
        omk=baseline.omk,
        tau=baseline.tau,
        As=A_s,
        ns=n_s,
    )
    use_ir_kernel = bool(q_ir > 1.0e-12)
    if use_ir_kernel:
        ell, d_ell_tt = _run_camb_tt_custom_power(
            params,
            lmax=int(lmax),
            power_fn=lambda k: _oph_p48_ir_power(
                k,
                A_s=A_s,
                ns=n_s,
                q_ir=q_ir,
                ell_ir=ell_ir,
                k0_mpc=DEFAULT_K0_MPC,
                d_star_mpc=DEFAULT_D_STAR_MPC,
            ),
            effective_ns=n_s,
        )
    else:
        ell, d_ell_tt = _run_camb_tt(params, lmax=int(lmax))
    comparison = compare_camb_tt_to_benchmark(ell, d_ell_tt, benchmark_rows)
    simulator_eta_ready = bool(screen_power_report.get("simulator_primordial_reference_ready", False))
    reference_source = str(screen_power_report.get("primordial_reference_source", "unknown"))
    physical_prediction = False
    return {
        "mode": "oph_screen_camb_transfer_scaffold",
        "benchmark": {
            "label": benchmark_label,
            "row_count": len(benchmark_rows),
            "ell_min": float(min(row["ell"] for row in benchmark_rows)) if benchmark_rows else None,
            "ell_max": float(max(row["ell"] for row in benchmark_rows)) if benchmark_rows else None,
        },
        "screen_input": {
            "reference_source": reference_source,
            "simulator_eta_R_ready": simulator_eta_ready,
            "eta_R": screen_params.get("eta_R"),
            "n_s_proxy": n_s,
            "N_cap_eff": screen_params.get("N_cap_eff"),
            "q_IR": screen_params.get("q_IR"),
            "ell_IR": screen_params.get("ell_IR"),
            "isotropic_ir_kernel_applied": use_ir_kernel,
            "excluded_from_scalar_transfer": bridge.get("excludes", []),
        },
        "camb": {
            "lmax": int(lmax),
            "cmb_unit": "muK",
            "spectrum": "lensed_total_TT_D_ell",
            "lambda_cdm_parameters_with_screen_ns": params.as_jsonable(),
            "custom_primordial_power": (
                "A_s*(k/k0)^(n_s-1)*(1-q_IR*exp[-ell(k)(ell(k)+1)/(ell_IR(ell_IR+1))])"
                if use_ir_kernel
                else None
            ),
        },
        "software": _software_versions(),
        "input_hashes": {
            "benchmark_sha256": benchmark_sha256,
            "screen_report_sha256": _sha256_json(screen_power_report),
            "params_sha256": _sha256_json(params.as_jsonable()),
        },
        "comparison": comparison,
        "physical_cmb_prediction": physical_prediction,
        "screen_camb_transfer_receipt": bool(comparison.get("usable", False)),
        "claim_boundary": (
            "CAMB TT transfer using the OPH screen scalar-tilt scaffold and any fitted isotropic q_IR/ell_IR "
            "kernel from the screen-power report. If the screen report uses the phenomenological Planck-target "
            "fallback, this is a benchmarked transfer scaffold, not a simulator-derived OPH CMB prediction. "
            "Parity/BipoSH angular covariance and OPH anomaly stress are not included in this scalar transfer."
        ),
    }


def write_oph_screen_camb_report(
    screen_report_path: Path,
    benchmark_path: Path,
    out_dir: Path,
    *,
    lmax: int = 2600,
    benchmark_label: str = "Planck2018_TT_binned",
    baseline_params: LambdaCDMParameters | None = None,
) -> dict[str, Any]:
    screen_report = json.loads(Path(screen_report_path).read_text(encoding="utf-8"))
    benchmark_rows = load_planck_tt_binned(Path(benchmark_path))
    report = oph_screen_camb_report(
        screen_report,
        benchmark_rows,
        baseline_params=baseline_params,
        lmax=int(lmax),
        benchmark_label=benchmark_label,
        benchmark_sha256=_sha256_file(Path(benchmark_path)),
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "oph_screen_camb_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / "oph_screen_camb_report.md").write_text(_oph_screen_markdown_report(report), encoding="utf-8")
    _write_spectrum_csv(out_dir / "oph_screen_camb_tt_bins.csv", report)
    return report


def oph_inflation_cmb_camb_report(
    bridge_report: dict[str, Any],
    benchmark_rows: list[dict[str, float]],
    *,
    baseline_params: LambdaCDMParameters | None = None,
    lmax: int = 2600,
    benchmark_label: str = "Planck2018_TT_binned",
    benchmark_sha256: str | None = None,
    k0_mpc: float = DEFAULT_K0_MPC,
    d_star_mpc: float = DEFAULT_D_STAR_MPC,
) -> dict[str, Any]:
    """Run CAMB TT curves for the OPH inflation/CMB theorem-target bridge.

    This is stronger than the scalar-tilt scaffold because it propagates the
    imported v0.4 OPH IR kernel through CAMB as a custom primordial spectrum.
    It is still not a finite-lattice prediction unless the bridge report says
    the screen spectrum and IR kernel were derived from finite cap/collar
    microphysics rather than imported from theorem-target/diagnostic notes.
    """

    baseline = baseline_params or LambdaCDMParameters()
    screen = dict(bridge_report.get("screen_spectrum_prediction", {}) or {})
    unique_gate = dict(bridge_report.get("unique_prediction_gate_v0_9", {}) or {})
    cmb_ladder = dict(bridge_report.get("cmb_success_ladder", {}) or {})
    core = dict(cmb_ladder.get("core_numbers", {}) or {})

    ns_oph = float(screen.get("n_s", baseline.ns))
    a_zeta = float(screen.get("A_zeta", baseline.As))
    q_ir = _float_or(core.get("v0_2_IR_bestfit_q_IR"), 0.0)
    ell_ir = _float_or(core.get("v0_2_IR_bestfit_ell_IR"), 6.0)

    lcdm_params = baseline
    p48_params = LambdaCDMParameters(
        H0=baseline.H0,
        ombh2=baseline.ombh2,
        omch2=baseline.omch2,
        mnu=baseline.mnu,
        omk=baseline.omk,
        tau=baseline.tau,
        As=a_zeta,
        ns=ns_oph,
    )
    ell_lcdm, tt_lcdm = _run_camb_tt(lcdm_params, lmax=int(lmax))
    ell_p48, tt_p48 = _run_camb_tt(p48_params, lmax=int(lmax))
    ell_ir_curve, tt_ir = _run_camb_tt_custom_power(
        p48_params,
        lmax=int(lmax),
        power_fn=lambda k: _oph_p48_ir_power(
            k,
            A_s=a_zeta,
            ns=ns_oph,
            q_ir=q_ir,
            ell_ir=ell_ir,
            k0_mpc=float(k0_mpc),
            d_star_mpc=float(d_star_mpc),
        ),
        effective_ns=ns_oph,
    )
    unique_curve: tuple[np.ndarray, np.ndarray] | None = None
    unique_params: LambdaCDMParameters | None = None
    unique_input: dict[str, Any] | None = None
    if unique_gate:
        unique_scalar = dict(unique_gate.get("scalar_tilt", {}) or {})
        unique_ir = dict(unique_gate.get("cmb_ir_kernel", {}) or {})
        ns_unique = _float_or(unique_scalar.get("n_s"), ns_oph)
        q_unique = _float_or(unique_ir.get("q_IR"), 0.25)
        ell_unique = _float_or(unique_ir.get("ell_IR"), 32.0)
        unique_params = LambdaCDMParameters(
            H0=baseline.H0,
            ombh2=baseline.ombh2,
            omch2=baseline.omch2,
            mnu=baseline.mnu,
            omk=baseline.omk,
            tau=baseline.tau,
            As=a_zeta,
            ns=ns_unique,
        )
        unique_curve = _run_camb_tt_custom_power(
            unique_params,
            lmax=int(lmax),
            power_fn=lambda k: _oph_p48_ir_power(
                k,
                A_s=a_zeta,
                ns=ns_unique,
                q_ir=q_unique,
                ell_ir=ell_unique,
                k0_mpc=float(k0_mpc),
                d_star_mpc=float(d_star_mpc),
            ),
            effective_ns=ns_unique,
        )
        unique_input = {
            "n_s": ns_unique,
            "eta_R": _float_or_none(unique_scalar.get("eta_R")),
            "A_zeta": a_zeta,
            "A_zeta_source": "legacy_collar_amplitude_selector_pending_unique_amplitude_derivation",
            "q_IR": q_unique,
            "ell_IR": ell_unique,
            "theta_IR_deg": _float_or_none(unique_ir.get("theta_IR_deg")),
            "k_IR_Mpc_inverse": _float_or_none(unique_ir.get("k_IR_Mpc_inverse")),
            "N_frz_proxy": _float_or_none(unique_ir.get("N_frz_proxy")),
            "finite_lattice_derived": False,
            "screen_spectrum_source": "unique_prediction_gate_v0_9_alpha_linked_tilt",
            "ir_kernel_source": "unique_prediction_gate_v0_9_exact_IR_kernel",
        }

    comparisons = {
        "camb_lcdm_powerlaw": compare_camb_tt_to_benchmark(ell_lcdm, tt_lcdm, benchmark_rows),
        "oph_p48_powerlaw": compare_camb_tt_to_benchmark(ell_p48, tt_p48, benchmark_rows),
        "oph_p48_ir_v04": compare_camb_tt_to_benchmark(ell_ir_curve, tt_ir, benchmark_rows),
    }
    if unique_curve is not None:
        comparisons["oph_unique_ir_v09"] = compare_camb_tt_to_benchmark(unique_curve[0], unique_curve[1], benchmark_rows)
    acoustic_ratios = _acoustic_ratio_summary(ell_lcdm, tt_lcdm, tt_ir)
    unique_acoustic_ratios = (
        _acoustic_ratio_summary(ell_lcdm, tt_lcdm, unique_curve[1]) if unique_curve is not None else None
    )
    model_curves = {
        "camb_lcdm_powerlaw": (ell_lcdm, tt_lcdm),
        "oph_p48_powerlaw": (ell_p48, tt_p48),
        "oph_p48_ir_v04": (ell_ir_curve, tt_ir),
    }
    if unique_curve is not None:
        model_curves["oph_unique_ir_v09"] = unique_curve
    binned_rows = _multi_model_binned_rows(benchmark_rows, model_curves)
    curve_rows = _curve_rows(ell_lcdm, {name: values for name, (_ell, values) in model_curves.items()})
    low_ell = {
        "source": "imported_OPH_CMB_success_ladder_v0_4",
        "available": bool(core),
        "CAMB_LCDM_chi2_ell2_29": _float_or_none(core.get("v0_3_camb_lowell_LCDM_chi2_ell2_29")),
        "CAMB_OPH_IR_chi2_ell2_29": _float_or_none(core.get("v0_3_camb_lowell_IR_bestfit_chi2_ell2_29")),
        "LCDM_R_OE_upper_PTE": _float_or_none(core.get("v0_4_LCDM_PTE_R_OE_upper")),
        "OPH_parity_R_OE_upper_PTE": _float_or_none(core.get("v0_4_parity_PTE_R_OE_upper")),
        "claim_boundary": (
            "Low-ell values are imported from the OPH-CMB v0.4 diagnostic ladder. "
            "This CAMB report does not rerun the unbinned low-ell Planck likelihood."
        ),
    }
    return {
        "mode": "oph_inflation_cmb_camb_transfer_v0",
        "benchmark": {
            "label": benchmark_label,
            "row_count": len(benchmark_rows),
            "ell_min": float(min(row["ell"] for row in benchmark_rows)) if benchmark_rows else None,
            "ell_max": float(max(row["ell"] for row in benchmark_rows)) if benchmark_rows else None,
        },
        "oph_input": {
            "bridge_mode": bridge_report.get("mode"),
            "P": screen.get("P"),
            "theta_OPH": screen.get("theta_OPH"),
            "n_s": ns_oph,
            "A_zeta": a_zeta,
            "q_IR": q_ir,
            "ell_IR": ell_ir,
            "k0_mpc": float(k0_mpc),
            "D_star_mpc": float(d_star_mpc),
            "finite_lattice_derived": False,
            "screen_spectrum_source": "theorem_target_P_over_48",
            "ir_kernel_source": "imported_v0_4_low_ell_fit",
        },
        "oph_unique_input": unique_input,
        "camb": {
            "lmax": int(lmax),
            "cmb_unit": "muK",
            "spectrum": "lensed_total_TT_D_ell",
            "baseline_lambda_cdm_parameters": lcdm_params.as_jsonable(),
            "oph_p48_lambda_cdm_parameters": p48_params.as_jsonable(),
            "oph_unique_lambda_cdm_parameters": unique_params.as_jsonable() if unique_params is not None else None,
            "custom_primordial_power": "A_zeta*(k/k0)^(n_s-1)*(1-q_IR*exp[-ell(k)(ell(k)+1)/(ell_IR(ell_IR+1))])",
        },
        "software": _software_versions(),
        "input_hashes": {
            "benchmark_sha256": benchmark_sha256,
            "bridge_report_sha256": _sha256_json(bridge_report),
            "oph_input_sha256": _sha256_json(
                {
                    "n_s": ns_oph,
                    "A_zeta": a_zeta,
                    "q_IR": q_ir,
                    "ell_IR": ell_ir,
                    "k0_mpc": float(k0_mpc),
                    "D_star_mpc": float(d_star_mpc),
                }
            ),
            "oph_unique_input_sha256": _sha256_json(unique_input) if unique_input is not None else None,
        },
        "comparison": comparisons,
        "acoustic_preservation": acoustic_ratios,
        "unique_acoustic_preservation": unique_acoustic_ratios,
        "low_ell_v04_diagnostic": low_ell,
        "binned_tt_comparison": binned_rows,
        "tt_curve_rows": curve_rows,
        "measurement_comparable_cmb_curve": True,
        "physical_cmb_prediction": False,
        "screen_camb_transfer_receipt": all(bool(item.get("usable", False)) for item in comparisons.values()),
        "claim_boundary": (
            "CAMB TT transfer for OPH theorem-target screen spectrum n_s=1-P/48 plus the imported v0.4 "
            "IR kernel. This is a real Boltzmann curve compared with the local Planck TT benchmark table, "
            "but not a finite-lattice derivation and not an official Planck likelihood. The current lattice "
            "must still derive A_zeta, q_IR, ell_IR, and any parity/BipoSH covariance from cap/collar state "
            "microphysics before this becomes a physical OPH CMB prediction."
        ),
    }


def write_oph_inflation_cmb_camb_report(
    bridge_report_path: Path,
    benchmark_path: Path,
    out_dir: Path,
    *,
    lmax: int = 2600,
    benchmark_label: str = "Planck2018_TT_binned",
    baseline_params: LambdaCDMParameters | None = None,
) -> dict[str, Any]:
    bridge_report = json.loads(Path(bridge_report_path).read_text(encoding="utf-8"))
    benchmark_rows = load_planck_tt_binned(Path(benchmark_path))
    report = oph_inflation_cmb_camb_report(
        bridge_report,
        benchmark_rows,
        baseline_params=baseline_params,
        lmax=int(lmax),
        benchmark_label=benchmark_label,
        benchmark_sha256=_sha256_file(Path(benchmark_path)),
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "oph_inflation_cmb_camb_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "oph_inflation_cmb_camb_report.md").write_text(
        _oph_inflation_cmb_markdown_report(report),
        encoding="utf-8",
    )
    _write_multi_model_csv(out_dir / "oph_inflation_cmb_tt_bins.csv", report["binned_tt_comparison"])
    _write_multi_model_csv(out_dir / "oph_inflation_cmb_tt_curves.csv", report["tt_curve_rows"])
    return report


def _run_camb_tt(params: LambdaCDMParameters, lmax: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        import camb
    except ModuleNotFoundError as exc:
        raise RuntimeError("CAMB is required for camb_lcdm_baseline_report") from exc

    camb_params = camb.CAMBparams()
    camb_params.set_cosmology(
        H0=float(params.H0),
        ombh2=float(params.ombh2),
        omch2=float(params.omch2),
        mnu=float(params.mnu),
        omk=float(params.omk),
        tau=float(params.tau),
    )
    camb_params.InitPower.set_params(As=float(params.As), ns=float(params.ns))
    camb_params.set_for_lmax(int(lmax), lens_potential_accuracy=1)
    results = camb.get_results(camb_params)
    powers = results.get_cmb_power_spectra(camb_params, CMB_unit="muK")
    d_ell_tt = np.asarray(powers["total"][:, 0], dtype=float)
    ell = np.arange(d_ell_tt.shape[0], dtype=float)
    return ell[2 : int(lmax) + 1], d_ell_tt[2 : int(lmax) + 1]


def _run_camb_tt_custom_power(
    params: LambdaCDMParameters,
    *,
    lmax: int,
    power_fn: Any,
    effective_ns: float,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import camb
    except ModuleNotFoundError as exc:
        raise RuntimeError("CAMB is required for oph_inflation_cmb_camb_report") from exc

    camb_params = camb.CAMBparams()
    camb_params.set_cosmology(
        H0=float(params.H0),
        ombh2=float(params.ombh2),
        omch2=float(params.omch2),
        mnu=float(params.mnu),
        omk=float(params.omk),
        tau=float(params.tau),
    )
    camb_params.set_initial_power_function(
        power_fn,
        kmin=1.0e-6,
        kmax=20.0,
        N_min=256,
        rtol=2.0e-5,
        effective_ns_for_nonlinear=float(effective_ns),
    )
    camb_params.set_for_lmax(int(lmax), lens_potential_accuracy=1)
    results = camb.get_results(camb_params)
    powers = results.get_cmb_power_spectra(camb_params, CMB_unit="muK")
    d_ell_tt = np.asarray(powers["total"][:, 0], dtype=float)
    ell = np.arange(d_ell_tt.shape[0], dtype=float)
    return ell[2 : int(lmax) + 1], d_ell_tt[2 : int(lmax) + 1]


def _oph_p48_ir_power(
    k_mpc: np.ndarray | list[float] | float,
    *,
    A_s: float,
    ns: float,
    q_ir: float,
    ell_ir: float,
    k0_mpc: float,
    d_star_mpc: float,
) -> np.ndarray:
    k = np.asarray(k_mpc, dtype=float)
    ell_proxy = np.maximum(k * float(d_star_mpc), 2.0)
    denom = max(float(ell_ir) * (float(ell_ir) + 1.0), 1.0e-12)
    f_ir = 1.0 - float(q_ir) * np.exp(-(ell_proxy * (ell_proxy + 1.0)) / denom)
    f_ir = np.maximum(f_ir, 0.0)
    base = float(A_s) * (np.maximum(k, 1.0e-30) / float(k0_mpc)) ** (float(ns) - 1.0)
    return np.asarray(base * f_ir, dtype=float)


def _software_versions() -> dict[str, str]:
    try:
        import camb

        camb_version = getattr(camb, "__version__", "unknown")
    except ModuleNotFoundError:
        camb_version = "not_installed"
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "camb_version": str(camb_version),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(data: dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _least_squares_amplitude(model: np.ndarray, observed: np.ndarray, sigma: np.ndarray) -> float:
    weight = 1.0 / np.maximum(sigma * sigma, 1.0e-24)
    denom = float(np.sum(weight * model * model))
    if denom < 1.0e-24:
        return 0.0
    return float(np.sum(weight * model * observed) / denom)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or float(np.std(a)) < 1.0e-12 or float(np.std(b)) < 1.0e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _standardized_rmse(model: np.ndarray, observed: np.ndarray, *, amplitude: float) -> float:
    model_z = _standardize(float(amplitude) * np.asarray(model, dtype=float))
    observed_z = _standardize(np.asarray(observed, dtype=float))
    residual = model_z - observed_z
    return float(np.sqrt(np.mean(residual * residual)))


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values - float(np.mean(values))
    scale = float(np.std(values))
    if scale < 1.0e-12:
        return np.zeros_like(values)
    return values / scale


def _first_peak_ell(ell: np.ndarray, values: np.ndarray) -> float | None:
    ell = np.asarray(ell, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = (ell >= 100.0) & (ell <= 350.0)
    if not np.any(mask):
        return None
    local_ell = ell[mask]
    local_values = values[mask]
    return float(local_ell[int(np.argmax(local_values))])


def _write_spectrum_csv(path: Path, report: dict[str, Any]) -> None:
    rows = report.get("comparison", {}).get("binned_tt_comparison", [])
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = [
        "ell",
        "observed_D_ell",
        "sigma_D_ell",
        "camb_D_ell",
        "amplitude_fit_camb_D_ell",
        "best_fit_column_D_ell",
    ]
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row.get(column, "")) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _multi_model_binned_rows(
    benchmark_rows: list[dict[str, float]],
    model_curves: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in benchmark_rows:
        ell_value = float(row.get("ell", 0.0))
        if ell_value < 2.0 or float(row.get("D_ell", 0.0)) <= 0.0:
            continue
        out = {
            "ell": ell_value,
            "observed_D_ell": float(row.get("D_ell", 0.0)),
            "minus_dD_ell": float(row.get("minus_dD_ell", 0.0)),
            "plus_dD_ell": float(row.get("plus_dD_ell", 0.0)),
            "best_fit_D_ell": float(row.get("best_fit_D_ell", row.get("D_ell", 0.0))),
        }
        for name, (ell, values) in model_curves.items():
            out[f"{name}_D_ell"] = float(np.interp(ell_value, ell, values))
        rows.append(out)
    return rows


def _curve_rows(ell: np.ndarray, model_values: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, ell_value in enumerate(np.asarray(ell, dtype=float)):
        row = {"ell": float(ell_value)}
        for name, values in model_values.items():
            row[f"{name}_D_ell"] = float(np.asarray(values, dtype=float)[index])
        rows.append(row)
    return rows


def _write_multi_model_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row.get(column, "")) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _acoustic_ratio_summary(ell: np.ndarray, baseline: np.ndarray, oph_ir: np.ndarray) -> dict[str, Any]:
    sample_ell = [2, 3, 10, 30, 50, 100, 220, 500, 1000]
    ell = np.asarray(ell, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    oph_ir = np.asarray(oph_ir, dtype=float)
    high_mask = ell >= 50.0
    if np.any(high_mask):
        model_high = oph_ir[high_mask]
        base_high = baseline[high_mask]
        denom = float(np.sum(model_high * model_high))
        amplitude_to_lcdm = float(np.sum(model_high * base_high) / denom) if denom > 1.0e-30 else 1.0
    else:
        amplitude_to_lcdm = 1.0
    fitted_oph = amplitude_to_lcdm * oph_ir
    ratios = []
    for value in sample_ell:
        if value < float(np.min(ell)) or value > float(np.max(ell)):
            continue
        base = float(np.interp(value, ell, baseline))
        raw_model = float(np.interp(value, ell, oph_ir))
        fitted_model = float(np.interp(value, ell, fitted_oph))
        ratios.append(
            {
                "ell": int(value),
                "raw_OPH_IR_over_LCDM_TT": float(raw_model / base) if base else None,
                "amplitude_matched_OPH_IR_over_LCDM_TT": float(fitted_model / base) if base else None,
            }
        )
    high_ratio = fitted_oph[high_mask] / np.maximum(baseline[high_mask], 1.0e-30)
    return {
        "ratios": ratios,
        "amplitude_to_lcdm_over_ell_ge_50": amplitude_to_lcdm,
        "mean_abs_fractional_delta_ell_ge_50": float(np.mean(np.abs(high_ratio - 1.0))) if high_ratio.size else None,
        "max_abs_fractional_delta_ell_ge_50": float(np.max(np.abs(high_ratio - 1.0))) if high_ratio.size else None,
        "claim_boundary": (
            "Amplitude-matched acoustic-region preservation diagnostic for the scalar OPH IR kernel. "
            "Raw amplitude is fitted out so this tests shape preservation rather than A_zeta normalization."
        ),
    }


def _markdown_report(report: dict[str, Any]) -> str:
    comparison = report["comparison"]
    software = report.get("software", {})
    hashes = report.get("input_hashes", {})
    return "\n".join(
        [
            "# CAMB LambdaCDM Baseline Regression",
            "",
            report["claim_boundary"],
            "",
            "## Receipt",
            "",
            f"- CDM-limit Boltzmann receipt: {report['CDM_LIMIT_BOLTZMANN_RECEIPT']}",
            f"- OPH anomaly module ready: {report['oph_anomaly_module_ready']}",
            f"- physical CMB prediction: {report['physical_cmb_prediction']}",
            "",
            "## TT Comparison",
            "",
            f"- bins: {comparison.get('bin_count')}",
            f"- shape correlation: {_fmt(comparison.get('shape_correlation'))}",
            f"- normalized RMSE: {_fmt(comparison.get('normalized_rmse'))}",
            f"- amplitude-fit chi2/bin: {_fmt(comparison.get('amplitude_fit_chi2_per_bin'))}",
            f"- raw chi2/bin: {_fmt(comparison.get('raw_chi2_per_bin'))}",
            f"- best-fit-column chi2/bin: {_fmt(comparison.get('best_fit_column_chi2_per_bin'))}",
            f"- first peak ell: {_fmt(comparison.get('first_peak_ell'))}",
            f"- benchmark first peak ell: {_fmt(comparison.get('benchmark_first_peak_ell'))}",
            "",
            "## Reproducibility",
            "",
            f"- Python: {software.get('python_version', 'n/a')}",
            f"- NumPy: {software.get('numpy_version', 'n/a')}",
            f"- CAMB: {software.get('camb_version', 'n/a')}",
            f"- benchmark SHA256: {hashes.get('benchmark_sha256', 'n/a')}",
            f"- params SHA256: {hashes.get('params_sha256', 'n/a')}",
            "",
        ]
    )


def _oph_inflation_cmb_markdown_report(report: dict[str, Any]) -> str:
    comparisons = report["comparison"]
    oph = report["oph_input"]
    unique = report.get("oph_unique_input") or {}
    software = report.get("software", {})
    low = report.get("low_ell_v04_diagnostic", {})
    acoustic = report.get("acoustic_preservation", {})
    unique_acoustic = report.get("unique_acoustic_preservation") or {}
    lines = [
        "# OPH Inflation/CMB CAMB Transfer",
        "",
        report["claim_boundary"],
        "",
        "## OPH Input",
        "",
        f"- n_s = 1 - P/48: {_fmt(oph.get('n_s'))}",
        f"- A_zeta: {_fmt(oph.get('A_zeta'))}",
        f"- q_IR: {_fmt(oph.get('q_IR'))}",
        f"- ell_IR: {_fmt(oph.get('ell_IR'))}",
        f"- finite-lattice derived: {oph.get('finite_lattice_derived')}",
        "",
        "## CAMB TT Comparison",
        "",
    ]
    if unique:
        lines[12:12] = [
            "## Unique v0.9 Input",
            "",
            f"- n_s = 1 - e alpha sqrt(pi): {_fmt(unique.get('n_s'))}",
            f"- eta_R: {_fmt(unique.get('eta_R'))}",
            f"- q_IR: {_fmt(unique.get('q_IR'))}",
            f"- ell_IR: {_fmt(unique.get('ell_IR'))}",
            f"- finite-lattice derived: {unique.get('finite_lattice_derived')}",
            "",
        ]
    for name, comparison in comparisons.items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- usable: {comparison.get('usable')}",
                f"- bins: {comparison.get('bin_count')}",
                f"- shape correlation: {_fmt(comparison.get('shape_correlation'))}",
                f"- normalized RMSE: {_fmt(comparison.get('normalized_rmse'))}",
                f"- amplitude-fit chi2/bin: {_fmt(comparison.get('amplitude_fit_chi2_per_bin'))}",
                f"- first peak ell: {_fmt(comparison.get('first_peak_ell'))}",
                "",
            ]
        )
    lines.extend(
        [
            "## Acoustic Preservation",
            "",
            f"- amplitude fit to LCDM over ell>=50: {_fmt(acoustic.get('amplitude_to_lcdm_over_ell_ge_50'))}",
            f"- mean |delta| for ell>=50: {_fmt(acoustic.get('mean_abs_fractional_delta_ell_ge_50'))}",
            f"- max |delta| for ell>=50: {_fmt(acoustic.get('max_abs_fractional_delta_ell_ge_50'))}",
        ]
    )
    for row in acoustic.get("ratios", []) or []:
        lines.append(
            f"- ell {row['ell']}: amplitude-matched OPH_IR/LCDM TT = "
            f"{_fmt(row.get('amplitude_matched_OPH_IR_over_LCDM_TT'))}"
        )
    if unique_acoustic:
        lines.extend(
            [
                "",
                "## Unique v0.9 Acoustic Preservation",
                "",
                f"- amplitude fit to LCDM over ell>=50: {_fmt(unique_acoustic.get('amplitude_to_lcdm_over_ell_ge_50'))}",
                f"- mean |delta| for ell>=50: {_fmt(unique_acoustic.get('mean_abs_fractional_delta_ell_ge_50'))}",
                f"- max |delta| for ell>=50: {_fmt(unique_acoustic.get('max_abs_fractional_delta_ell_ge_50'))}",
            ]
        )
    lines.extend(
        [
            "",
            "## Imported Low-Ell v0.4 Diagnostic",
            "",
            f"- available: {low.get('available')}",
            f"- LCDM chi2 ell=2..29: {_fmt(low.get('CAMB_LCDM_chi2_ell2_29'))}",
            f"- OPH IR chi2 ell=2..29: {_fmt(low.get('CAMB_OPH_IR_chi2_ell2_29'))}",
            f"- LCDM R_OE upper PTE: {_fmt(low.get('LCDM_R_OE_upper_PTE'))}",
            f"- OPH parity R_OE upper PTE: {_fmt(low.get('OPH_parity_R_OE_upper_PTE'))}",
            "",
            "## Reproducibility",
            "",
            f"- Python: {software.get('python_version', 'n/a')}",
            f"- NumPy: {software.get('numpy_version', 'n/a')}",
            f"- CAMB: {software.get('camb_version', 'n/a')}",
            "",
            "## Output Files",
            "",
            "- `oph_inflation_cmb_camb_report.json`",
            "- `oph_inflation_cmb_tt_bins.csv`",
            "- `oph_inflation_cmb_tt_curves.csv`",
            "",
        ]
    )
    return "\n".join(lines)


def _oph_screen_markdown_report(report: dict[str, Any]) -> str:
    comparison = report["comparison"]
    screen = report.get("screen_input", {})
    software = report.get("software", {})
    hashes = report.get("input_hashes", {})
    params = (report.get("camb", {}) or {}).get("lambda_cdm_parameters_with_screen_ns", {}) or {}
    return "\n".join(
        [
            "# OPH Screen CAMB Transfer Scaffold",
            "",
            report["claim_boundary"],
            "",
            "## Screen Input",
            "",
            f"- reference source: {screen.get('reference_source', 'n/a')}",
            f"- simulator eta_R ready: {screen.get('simulator_eta_R_ready')}",
            f"- eta_R: {_fmt(screen.get('eta_R'))}",
            f"- n_s proxy used in CAMB: {_fmt(screen.get('n_s_proxy'))}",
            f"- N_cap_eff: {_fmt(screen.get('N_cap_eff'))}",
            "",
            "## CAMB Transfer",
            "",
            f"- physical CMB prediction: {report['physical_cmb_prediction']}",
            f"- screen CAMB transfer receipt: {report['screen_camb_transfer_receipt']}",
            f"- H0: {_fmt(params.get('H0'))}",
            f"- ombh2: {_fmt(params.get('ombh2'))}",
            f"- omch2: {_fmt(params.get('omch2'))}",
            f"- As: {_fmt(params.get('As'))}",
            f"- ns: {_fmt(params.get('ns'))}",
            "",
            "## TT Comparison",
            "",
            f"- bins: {comparison.get('bin_count')}",
            f"- shape correlation: {_fmt(comparison.get('shape_correlation'))}",
            f"- normalized RMSE: {_fmt(comparison.get('normalized_rmse'))}",
            f"- amplitude-fit chi2/bin: {_fmt(comparison.get('amplitude_fit_chi2_per_bin'))}",
            f"- first peak ell: {_fmt(comparison.get('first_peak_ell'))}",
            f"- benchmark first peak ell: {_fmt(comparison.get('benchmark_first_peak_ell'))}",
            "",
            "## Reproducibility",
            "",
            f"- Python: {software.get('python_version', 'n/a')}",
            f"- NumPy: {software.get('numpy_version', 'n/a')}",
            f"- CAMB: {software.get('camb_version', 'n/a')}",
            f"- benchmark SHA256: {hashes.get('benchmark_sha256', 'n/a')}",
            f"- screen report SHA256: {hashes.get('screen_report_sha256', 'n/a')}",
            f"- params SHA256: {hashes.get('params_sha256', 'n/a')}",
            "",
        ]
    )


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.10g}"
    return "n/a"


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _float_or(value: Any, default: float) -> float:
    parsed = _float_or_none(value)
    return float(default if parsed is None else parsed)
