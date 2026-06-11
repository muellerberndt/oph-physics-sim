from __future__ import annotations

import json
import hashlib
import importlib
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.cosmology.cmb_compare import load_planck_tt_binned
from oph_fpe.cosmology.selector_elimination import selector_elimination_report
from oph_fpe.cosmology.unique_predictions import unique_prediction_gate_report


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
            "fallback, this is a benchmarked transfer scaffold; if it uses a failed finite diagnostic, this is "
            "a simulator-derived mismatch report. It is not a physical OPH CMB prediction unless the finite "
            "screen report passes its simulator_eta_ready/readiness gates and downstream finite-certificate gates. "
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


def scale_compressed_cmb_camb_report(
    scale_report: dict[str, Any],
    benchmark_rows: list[dict[str, float]],
    *,
    baseline_params: LambdaCDMParameters | None = None,
    lmax: int = 2600,
    benchmark_label: str = "Planck2018_TT_binned",
    benchmark_sha256: str | None = None,
    k0_mpc: float = DEFAULT_K0_MPC,
    d_star_mpc: float = DEFAULT_D_STAR_MPC,
) -> dict[str, Any]:
    """Run CAMB transfer for the logical scale-compressed repair branch.

    This is the closest current measurement-comparable CMB artifact tied to the
    24-round repair-depth simulator lane. It deliberately preserves the branch
    boundary: the numbers are emitted by the scale-compressed continuation, not
    by a literal 10^122-cell finite lattice or an official Planck likelihood.
    """

    baseline = baseline_params or LambdaCDMParameters()
    readouts = dict(scale_report.get("cmb_parameter_readouts", {}) or {})
    n_s = _float_or(readouts.get("n_s"), baseline.ns)
    eta_r = _float_or_none(readouts.get("eta_R"))
    q_ir = _float_or(readouts.get("q_IR"), 0.25)
    ell_ir = _float_or(readouts.get("ell_IR"), 32.0)
    branch_params = LambdaCDMParameters(
        H0=baseline.H0,
        ombh2=baseline.ombh2,
        omch2=baseline.omch2,
        mnu=baseline.mnu,
        omk=baseline.omk,
        tau=baseline.tau,
        As=baseline.As,
        ns=n_s,
    )

    ell_lcdm, tt_lcdm = _run_camb_tt(baseline, lmax=int(lmax))
    ell_scalar, tt_scalar = _run_camb_tt(branch_params, lmax=int(lmax))
    ell_ir_curve, tt_ir = _run_camb_tt_custom_power(
        branch_params,
        lmax=int(lmax),
        power_fn=lambda k: _oph_p48_ir_power(
            k,
            A_s=branch_params.As,
            ns=n_s,
            q_ir=q_ir,
            ell_ir=ell_ir,
            k0_mpc=float(k0_mpc),
            d_star_mpc=float(d_star_mpc),
        ),
        effective_ns=n_s,
    )
    comparisons = {
        "camb_lcdm_powerlaw": compare_camb_tt_to_benchmark(ell_lcdm, tt_lcdm, benchmark_rows),
        "scale_compressed_scalar_tilt": compare_camb_tt_to_benchmark(ell_scalar, tt_scalar, benchmark_rows),
        "scale_compressed_ir_kernel": compare_camb_tt_to_benchmark(ell_ir_curve, tt_ir, benchmark_rows),
    }
    model_curves = {
        "camb_lcdm_powerlaw": (ell_lcdm, tt_lcdm),
        "scale_compressed_scalar_tilt": (ell_scalar, tt_scalar),
        "scale_compressed_ir_kernel": (ell_ir_curve, tt_ir),
    }
    return {
        "mode": "oph_scale_compressed_cmb_camb_transfer_v0",
        "benchmark": {
            "label": benchmark_label,
            "row_count": len(benchmark_rows),
            "ell_min": float(min(row["ell"] for row in benchmark_rows)) if benchmark_rows else None,
            "ell_max": float(max(row["ell"] for row in benchmark_rows)) if benchmark_rows else None,
        },
        "scale_compressed_input": {
            "source_mode": scale_report.get("mode"),
            "logical_repair_rounds": scale_report.get("logical_repair_rounds"),
            "scale_compressed_operator_receipt": bool(scale_report.get("scale_compressed_operator_receipt", False)),
            "repair_round_trace_receipt": bool(scale_report.get("repair_round_trace_receipt", False)),
            "populated_h3_preview_receipt": bool(
                ((scale_report.get("h3_preview") or {}).get("populated_h3_preview_receipt", False))
            ),
            "strict_neutral_bulk": bool(scale_report.get("strict_neutral_bulk", False)),
            "eta_R": eta_r,
            "n_s": n_s,
            "A_s": branch_params.As,
            "q_IR": q_ir,
            "ell_IR": ell_ir,
            "N_CRC_implied_by_declared_repair_depth_ansatz": readouts.get(
                "N_CRC_implied_by_declared_repair_depth_ansatz",
                readouts.get("N_CRC_predicted_from_P"),
            ),
            "N_CRC_predicted_from_P": readouts.get("N_CRC_predicted_from_P"),
            "N_CRC_declared": readouts.get("N_CRC_declared"),
            "relative_error_ansatz_capacity_vs_declared_N_CRC": readouts.get(
                "relative_error_ansatz_capacity_vs_declared_N_CRC",
                readouts.get("relative_error_gprime_vs_N_CRC"),
            ),
            "relative_error_gprime_vs_N_CRC": readouts.get("relative_error_gprime_vs_N_CRC"),
            "k0_mpc": float(k0_mpc),
            "D_star_mpc": float(d_star_mpc),
            "finite_lattice_derived": False,
            "source": "scale_compressed_24_round_repair_branch",
        },
        "camb": {
            "lmax": int(lmax),
            "cmb_unit": "muK",
            "spectrum": "lensed_total_TT_D_ell",
            "baseline_lambda_cdm_parameters": baseline.as_jsonable(),
            "scale_compressed_lambda_cdm_parameters": branch_params.as_jsonable(),
            "custom_primordial_power": (
                "A_s*(k/k0)^(n_s-1)*(1-q_IR*exp[-ell(k)(ell(k)+1)/(ell_IR(ell_IR+1))])"
            ),
        },
        "software": _software_versions(),
        "input_hashes": {
            "benchmark_sha256": benchmark_sha256,
            "scale_compressed_report_sha256": _sha256_json(scale_report),
            "scale_compressed_input_sha256": _sha256_json(
                {
                    "n_s": n_s,
                    "eta_R": eta_r,
                    "A_s": branch_params.As,
                    "q_IR": q_ir,
                    "ell_IR": ell_ir,
                    "k0_mpc": float(k0_mpc),
                    "D_star_mpc": float(d_star_mpc),
                }
            ),
        },
        "comparison": comparisons,
        "acoustic_preservation": _acoustic_ratio_summary(ell_lcdm, tt_lcdm, tt_ir),
        "binned_tt_comparison": _multi_model_binned_rows(benchmark_rows, model_curves),
        "tt_curve_rows": _curve_rows(ell_lcdm, {name: values for name, (_ell, values) in model_curves.items()}),
        "measurement_comparable_cmb_curve": True,
        "physical_cmb_prediction": False,
        "screen_camb_transfer_receipt": all(bool(item.get("usable", False)) for item in comparisons.values()),
        "claim_boundary": (
            "CAMB TT transfer for the logical 24-round scale-compressed repair branch. The transfer uses "
            "the branch readouts n_s, q_IR, and ell_IR and compares the resulting TT curves with the local "
            "Planck TT binned table. It is measurement-comparable output, but it is not a physical OPH CMB "
            "prediction until the 24 repair rounds, scalar amplitude, OPH anomaly sector, and official "
            "likelihood/map-space gates are derived from finite support-visible dynamics."
        ),
    }


def write_scale_compressed_cmb_camb_report(
    scale_report_path: Path,
    benchmark_path: Path,
    out_dir: Path,
    *,
    lmax: int = 2600,
    benchmark_label: str = "Planck2018_TT_binned",
    baseline_params: LambdaCDMParameters | None = None,
) -> dict[str, Any]:
    scale_report = json.loads(Path(scale_report_path).read_text(encoding="utf-8"))
    benchmark_rows = load_planck_tt_binned(Path(benchmark_path))
    report = scale_compressed_cmb_camb_report(
        scale_report,
        benchmark_rows,
        baseline_params=baseline_params,
        lmax=int(lmax),
        benchmark_label=benchmark_label,
        benchmark_sha256=_sha256_file(Path(benchmark_path)),
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scale_compressed_cmb_camb_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "scale_compressed_cmb_camb_report.md").write_text(
        _scale_compressed_cmb_markdown_report(report),
        encoding="utf-8",
    )
    _write_multi_model_csv(out_dir / "scale_compressed_cmb_tt_bins.csv", report["binned_tt_comparison"])
    _write_multi_model_csv(out_dir / "scale_compressed_cmb_tt_curves.csv", report["tt_curve_rows"])
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


def oph_exact_cmb_camb_report(
    benchmark_rows: list[dict[str, float]],
    *,
    source_dir: Path | None = None,
    baseline_params: LambdaCDMParameters | None = None,
    lmax: int = 2600,
    benchmark_label: str = "Planck2018_TT_binned",
    benchmark_sha256: str | None = None,
    k0_mpc: float = DEFAULT_K0_MPC,
    d_star_mpc: float = DEFAULT_D_STAR_MPC,
) -> dict[str, Any]:
    """Run CAMB for the exact OPH-CMB scalar target branch.

    This is the native version of the v1.0 CMB handoff scripts. It keeps the
    exact alpha-linked scalar tilt and exact low-ell IR kernel inside the
    OPH-FPE reporting pipeline, while still marking official Planck likelihood
    execution as a separate external-data gate.
    """

    baseline = baseline_params or LambdaCDMParameters()
    target = unique_prediction_gate_report(source_dir)
    selector_report = selector_elimination_report(source_dir)
    scalar = dict(target.get("scalar_tilt", {}) or {})
    ir = dict(target.get("cmb_ir_kernel", {}) or {})
    ns_exact = _float_or(scalar.get("n_s"), baseline.ns)
    eta_r = _float_or_none(scalar.get("eta_R"))
    q_ir = _float_or(ir.get("q_IR"), 0.25)
    ell_ir = _float_or(ir.get("ell_IR"), 32.0)
    exact_params = LambdaCDMParameters(
        H0=baseline.H0,
        ombh2=baseline.ombh2,
        omch2=baseline.omch2,
        mnu=baseline.mnu,
        omk=baseline.omk,
        tau=baseline.tau,
        As=baseline.As,
        ns=ns_exact,
    )

    ell_lcdm, tt_lcdm = _run_camb_tt(baseline, lmax=int(lmax))
    ell_exact, tt_exact = _run_camb_tt(exact_params, lmax=int(lmax))
    ell_ir_curve, tt_ir = _run_camb_tt_custom_power(
        exact_params,
        lmax=int(lmax),
        power_fn=lambda k: _oph_p48_ir_power(
            k,
            A_s=exact_params.As,
            ns=ns_exact,
            q_ir=q_ir,
            ell_ir=ell_ir,
            k0_mpc=float(k0_mpc),
            d_star_mpc=float(d_star_mpc),
        ),
        effective_ns=ns_exact,
    )
    comparisons = {
        "camb_lcdm_powerlaw": compare_camb_tt_to_benchmark(ell_lcdm, tt_lcdm, benchmark_rows),
        "oph_exact_scalar_tilt": compare_camb_tt_to_benchmark(ell_exact, tt_exact, benchmark_rows),
        "oph_exact_ir_v10": compare_camb_tt_to_benchmark(ell_ir_curve, tt_ir, benchmark_rows),
    }
    model_curves = {
        "camb_lcdm_powerlaw": (ell_lcdm, tt_lcdm),
        "oph_exact_scalar_tilt": (ell_exact, tt_exact),
        "oph_exact_ir_v10": (ell_ir_curve, tt_ir),
    }
    binned_rows = _multi_model_binned_rows(benchmark_rows, model_curves)
    curve_rows = _curve_rows(ell_lcdm, {name: values for name, (_ell, values) in model_curves.items()})
    acoustic = _acoustic_ratio_summary(ell_lcdm, tt_lcdm, tt_ir)
    source_files = _exact_cmb_source_status(Path(source_dir) if source_dir is not None else None)
    official_readiness = official_planck_readiness_report()
    return {
        "mode": "oph_exact_cmb_camb_transfer_v1",
        "benchmark": {
            "label": benchmark_label,
            "row_count": len(benchmark_rows),
            "ell_min": float(min(row["ell"] for row in benchmark_rows)) if benchmark_rows else None,
            "ell_max": float(max(row["ell"] for row in benchmark_rows)) if benchmark_rows else None,
        },
        "oph_exact_input": {
            "n_s": ns_exact,
            "eta_R": eta_r,
            "q_IR": q_ir,
            "ell_IR": ell_ir,
            "theta_IR_deg": _float_or_none(ir.get("theta_IR_deg")),
            "k_IR_Mpc_inverse": _float_or_none(ir.get("k_IR_Mpc_inverse")),
            "N_frz_proxy": _float_or_none(ir.get("N_frz_proxy")),
            "A_s": exact_params.As,
            "k0_mpc": float(k0_mpc),
            "D_star_mpc": float(d_star_mpc),
            "finite_lattice_derived": bool(target.get("finite_lattice_derived", False)),
            "screen_spectrum_source": "unique_prediction_gate_alpha_linked_tilt",
            "ir_kernel_source": "selector_elimination_v1_5_affine_zero_mode_and_visible_covariance_rank",
            "selector_elimination_theorem_receipt": bool(
                selector_report.get("THEOREM_SIDE_SELECTOR_ELIMINATION_RECEIPT", False)
            ),
            "selector_elimination_source_audit_receipt": bool(
                selector_report.get("SOURCE_PACKET_AUDIT_RECEIPT", False)
            ),
            "kappa_rep_status": selector_report.get("scalar_tilt", {}).get("canonical_kappa_rep_status"),
        },
        "camb": {
            "lmax": int(lmax),
            "cmb_unit": "muK",
            "spectrum": "lensed_total_TT_D_ell",
            "baseline_lambda_cdm_parameters": baseline.as_jsonable(),
            "oph_exact_lambda_cdm_parameters": exact_params.as_jsonable(),
            "custom_primordial_power": "A_s*(k/k0)^(n_s-1)*(1-q_IR*exp[-ell(k)(ell(k)+1)/(ell_IR(ell_IR+1))])",
        },
        "software": _software_versions(),
        "source_files": source_files,
        "selector_elimination_v1_5": {
            "selector_elimination": selector_report.get("selector_elimination", {}),
            "scalar_tilt": selector_report.get("scalar_tilt", {}),
            "cmb_ir_kernel": {
                key: value
                for key, value in selector_report.get("cmb_ir_kernel", {}).items()
                if key != "exact_values"
            },
            "source_status_audit": selector_report.get("source_status_audit", {}),
            "exact_ir_kernel_csv_audit": {
                key: value
                for key, value in selector_report.get("exact_ir_kernel_csv_audit", {}).items()
                if key != "rows"
            },
            "THEOREM_SIDE_SELECTOR_ELIMINATION_RECEIPT": selector_report.get(
                "THEOREM_SIDE_SELECTOR_ELIMINATION_RECEIPT", False
            ),
            "SOURCE_PACKET_AUDIT_RECEIPT": selector_report.get("SOURCE_PACKET_AUDIT_RECEIPT", False),
            "finite_lattice_derived": selector_report.get("finite_lattice_derived", False),
        },
        "official_planck_likelihood_readiness": official_readiness,
        "input_hashes": {
            "benchmark_sha256": benchmark_sha256,
            "unique_prediction_target_sha256": _sha256_json(target),
            "selector_elimination_target_sha256": _sha256_json(selector_report),
            "oph_exact_input_sha256": _sha256_json(
                {
                    "n_s": ns_exact,
                    "eta_R": eta_r,
                    "q_IR": q_ir,
                    "ell_IR": ell_ir,
                    "A_s": exact_params.As,
                    "k0_mpc": float(k0_mpc),
                    "D_star_mpc": float(d_star_mpc),
                }
            ),
        },
        "comparison": comparisons,
        "acoustic_preservation": acoustic,
        "binned_tt_comparison": binned_rows,
        "tt_curve_rows": curve_rows,
        "measurement_comparable_cmb_curve": True,
        "official_planck_likelihood_run": False,
        "physical_cmb_prediction": False,
        "screen_camb_transfer_receipt": all(bool(item.get("usable", False)) for item in comparisons.values()),
        "claim_boundary": (
            "CAMB TT transfer for the exact OPH-CMB scalar target branch updated to the v1.5 selector-elimination "
            "surface: q_IR=1/4 follows the affine zero-mode reserve and ell_IR=32 follows the dodecahedral "
            "visible-covariance rank. The red tilt uses the canonical repair-clock branch "
            "n_s=1-e*alpha(0)*sqrt(pi), but kappa_rep=e is still a finite-patch repair-clock certificate. "
            "This is an organic Boltzmann-transfer diagnostic inside OPH-FPE, but it is not yet a finite-lattice "
            "derivation, not a masked map-space parity/BipoSH likelihood, and not an official Planck clik "
            "likelihood run."
        ),
    }


def write_oph_exact_cmb_camb_report(
    benchmark_path: Path,
    out_dir: Path,
    *,
    source_dir: Path | None = None,
    lmax: int = 2600,
    benchmark_label: str = "Planck2018_TT_binned",
    baseline_params: LambdaCDMParameters | None = None,
) -> dict[str, Any]:
    benchmark_rows = load_planck_tt_binned(Path(benchmark_path))
    report = oph_exact_cmb_camb_report(
        benchmark_rows,
        source_dir=source_dir,
        baseline_params=baseline_params,
        lmax=int(lmax),
        benchmark_label=benchmark_label,
        benchmark_sha256=_sha256_file(Path(benchmark_path)),
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "oph_exact_cmb_camb_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "oph_exact_cmb_camb_report.md").write_text(
        _oph_exact_cmb_markdown_report(report),
        encoding="utf-8",
    )
    _write_multi_model_csv(out_dir / "oph_exact_cmb_tt_bins.csv", report["binned_tt_comparison"])
    _write_multi_model_csv(out_dir / "oph_exact_cmb_tt_curves.csv", report["tt_curve_rows"])
    return report


def finite_repair_clock_cmb_camb_report(
    finite_clock_report: dict[str, Any],
    benchmark_rows: list[dict[str, float]],
    *,
    source_dir: Path | None = None,
    baseline_params: LambdaCDMParameters | None = None,
    lmax: int = 2600,
    benchmark_label: str = "Planck2018_TT_binned",
    benchmark_sha256: str | None = None,
    k0_mpc: float = DEFAULT_K0_MPC,
    d_star_mpc: float = DEFAULT_D_STAR_MPC,
) -> dict[str, Any]:
    """Run CAMB from the simulator-derived finite repair-clock scalar tilt.

    This is intentionally separate from the exact OPH target branch. It uses the
    empirical finite transition matrix output as the scalar tilt source, then
    optionally overlays the selector-elimination IR kernel as a theory-side
    diagnostic. That gives a measurement-facing curve from actual simulator data
    without silently promoting it to the certified kappa_rep=e branch.
    """

    baseline = baseline_params or LambdaCDMParameters()
    selector_report = selector_elimination_report(source_dir=source_dir)
    primary = finite_clock_report.get("primary", {}) if isinstance(finite_clock_report, dict) else {}
    if not primary:
        primary = finite_clock_report.get("semigroup", {}) if isinstance(finite_clock_report, dict) else {}
    ns_finite = _float_or_none(primary.get("n_s_estimate"))
    eta_finite = _float_or_none(primary.get("eta_R_estimate"))
    kappa_finite = _float_or_none(primary.get("kappa_rep_estimate"))
    if ns_finite is None and eta_finite is not None:
        ns_finite = 1.0 - float(eta_finite)
    if eta_finite is None and ns_finite is not None:
        eta_finite = 1.0 - float(ns_finite)
    if ns_finite is None:
        raise ValueError("finite repair-clock report does not expose n_s_estimate or eta_R_estimate")
    ir = selector_report.get("cmb_ir_kernel", {})
    q_ir = _float_or(ir.get("q_IR"), 0.0)
    ell_ir = _float_or(ir.get("ell_IR"), 32.0)
    finite_params = LambdaCDMParameters(
        H0=baseline.H0,
        ombh2=baseline.ombh2,
        omch2=baseline.omch2,
        mnu=baseline.mnu,
        omk=baseline.omk,
        tau=baseline.tau,
        As=baseline.As,
        ns=float(ns_finite),
    )
    ell_lcdm, tt_lcdm = _run_camb_tt(baseline, lmax=int(lmax))
    ell_finite, tt_finite = _run_camb_tt(finite_params, lmax=int(lmax))
    ell_ir_curve, tt_ir = _run_camb_tt_custom_power(
        finite_params,
        lmax=int(lmax),
        power_fn=lambda k: _oph_p48_ir_power(
            k,
            A_s=finite_params.As,
            ns=float(ns_finite),
            q_ir=q_ir,
            ell_ir=ell_ir,
            k0_mpc=float(k0_mpc),
            d_star_mpc=float(d_star_mpc),
        ),
        effective_ns=float(ns_finite),
    )
    comparisons = {
        "camb_lcdm_powerlaw": compare_camb_tt_to_benchmark(ell_lcdm, tt_lcdm, benchmark_rows),
        "finite_repair_clock_scalar_tilt": compare_camb_tt_to_benchmark(
            ell_finite, tt_finite, benchmark_rows
        ),
        "finite_repair_clock_plus_selector_ir": compare_camb_tt_to_benchmark(
            ell_ir_curve, tt_ir, benchmark_rows
        ),
    }
    model_curves = {
        "camb_lcdm_powerlaw": (ell_lcdm, tt_lcdm),
        "finite_repair_clock_scalar_tilt": (ell_finite, tt_finite),
        "finite_repair_clock_plus_selector_ir": (ell_ir_curve, tt_ir),
    }
    binned_rows = _multi_model_binned_rows(benchmark_rows, model_curves)
    curve_rows = _curve_rows(ell_lcdm, {name: values for name, (_ell, values) in model_curves.items()})
    finite_matrix_ready = bool(finite_clock_report.get("finite_transition_matrix_ready", False))
    finite_lattice_derived = bool(finite_clock_report.get("finite_lattice_derived", False))
    repair_clock_certificate = bool(finite_clock_report.get("repair_clock_certificate", False))
    selector_receipt = bool(selector_report.get("THEOREM_SIDE_SELECTOR_ELIMINATION_RECEIPT", False))
    return {
        "mode": "finite_repair_clock_cmb_camb_transfer_v0",
        "benchmark": {
            "label": benchmark_label,
            "row_count": len(benchmark_rows),
            "ell_min": float(min(row["ell"] for row in benchmark_rows)) if benchmark_rows else None,
            "ell_max": float(max(row["ell"] for row in benchmark_rows)) if benchmark_rows else None,
        },
        "finite_repair_clock_input": {
            "n_s": float(ns_finite),
            "eta_R": eta_finite,
            "kappa_rep": kappa_finite,
            "matrix_ready": finite_matrix_ready,
            "finite_lattice_derived": finite_lattice_derived,
            "repair_clock_certificate": repair_clock_certificate,
            "clock_normalization_certified": bool(
                finite_clock_report.get("clock_normalization_certified", False)
            ),
            "clock_normalization_numeric_match": bool(
                finite_clock_report.get("clock_normalization_numeric_match", False)
            ),
            "repair_scale_hypothesis_clock_match": bool(
                finite_clock_report.get("repair_scale_hypothesis_clock_match", False)
            ),
            "clock_normalization_source": finite_clock_report.get("clock_normalization_source"),
            "clock_normalization_source_status": finite_clock_report.get(
                "clock_normalization_source_status"
            ),
            "source_mode": finite_clock_report.get("mode"),
            "primary_matrix": finite_clock_report.get("primary_matrix"),
            "repair_step_time": finite_clock_report.get("repair_step_time"),
            "state_count": finite_clock_report.get("state_count"),
            "transition_count": finite_clock_report.get("transition_count"),
            "blockers": finite_clock_report.get("blockers", []),
        },
        "selector_ir_input": {
            "q_IR": q_ir,
            "ell_IR": ell_ir,
            "selector_elimination_theorem_receipt": selector_receipt,
            "selector_elimination_source_audit_receipt": bool(
                selector_report.get("SOURCE_PACKET_AUDIT_RECEIPT", False)
            ),
            "finite_lattice_derived": bool(selector_report.get("finite_lattice_derived", False)),
        },
        "camb": {
            "lmax": int(lmax),
            "cmb_unit": "muK",
            "spectrum": "lensed_total_TT_D_ell",
            "baseline_lambda_cdm_parameters": baseline.as_jsonable(),
            "finite_repair_clock_lambda_cdm_parameters": finite_params.as_jsonable(),
            "custom_primordial_power": (
                "A_s*(k/k0)^(n_s_finite-1)*(1-q_IR*exp[-ell(k)(ell(k)+1)/(ell_IR(ell_IR+1))])"
            ),
        },
        "software": _software_versions(),
        "input_hashes": {
            "benchmark_sha256": benchmark_sha256,
            "finite_repair_clock_report_sha256": _sha256_json(finite_clock_report),
            "selector_elimination_target_sha256": _sha256_json(selector_report),
        },
        "comparison": comparisons,
        "acoustic_preservation": _acoustic_ratio_summary(ell_lcdm, tt_lcdm, tt_ir),
        "binned_tt_comparison": binned_rows,
        "tt_curve_rows": curve_rows,
        "measurement_comparable_cmb_curve": True,
        "finite_lattice_clock_derived": finite_lattice_derived,
        "repair_clock_certificate": repair_clock_certificate,
        "selector_ir_theory_side": selector_receipt,
        "official_planck_likelihood_run": False,
        "screen_camb_transfer_receipt": all(bool(item.get("usable", False)) for item in comparisons.values()),
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "CAMB TT transfer using the simulator-derived finite transition-matrix repair clock for "
            "the scalar tilt. The finite clock is actual simulator data when finite_lattice_derived is true. "
            "The selector IR overlay remains theory-side unless its own finite-register certificate closes. "
            "This is measurement-comparable simulator output, but not a physical CMB prediction until "
            "kappa_rep=e/repair-clock, physical k/source, official likelihood, and no-data-use gates pass."
        ),
    }


def write_finite_repair_clock_cmb_camb_report(
    finite_clock_report_path: Path,
    benchmark_path: Path,
    out_dir: Path,
    *,
    source_dir: Path | None = None,
    lmax: int = 2600,
    benchmark_label: str = "Planck2018_TT_binned",
    baseline_params: LambdaCDMParameters | None = None,
) -> dict[str, Any]:
    finite_clock_report = json.loads(Path(finite_clock_report_path).read_text(encoding="utf-8"))
    benchmark_rows = load_planck_tt_binned(Path(benchmark_path))
    report = finite_repair_clock_cmb_camb_report(
        finite_clock_report,
        benchmark_rows,
        source_dir=source_dir,
        baseline_params=baseline_params,
        lmax=int(lmax),
        benchmark_label=benchmark_label,
        benchmark_sha256=_sha256_file(Path(benchmark_path)),
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "finite_repair_clock_cmb_camb_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "finite_repair_clock_cmb_camb_report.md").write_text(
        _finite_repair_clock_cmb_markdown_report(report),
        encoding="utf-8",
    )
    _write_multi_model_csv(out_dir / "finite_repair_clock_cmb_tt_bins.csv", report["binned_tt_comparison"])
    _write_multi_model_csv(out_dir / "finite_repair_clock_cmb_tt_curves.csv", report["tt_curve_rows"])
    return report


def official_planck_readiness_report() -> dict[str, Any]:
    modules: dict[str, Any] = {}
    for name in ("camb", "cobaya", "clik", "clipy", "healpy"):
        spec = importlib.util.find_spec(name)
        module_info: dict[str, Any] = {"importable": spec is not None}
        if spec is not None:
            module_info["origin"] = spec.origin
            try:
                module = importlib.import_module(name)
                module_info["version"] = str(getattr(module, "__version__", "unknown"))
                if name == "clik":
                    module_info["has_clik_api"] = bool(hasattr(module, "clik"))
                    module_info["has_lensing_api"] = bool(hasattr(module, "try_lensing"))
            except Exception as exc:  # pragma: no cover - import failures are environment-specific.
                module_info["import_error"] = repr(exc)
        modules[name] = module_info
    official_clik_ready = bool(
        modules.get("clik", {}).get("importable")
        and modules.get("clik", {}).get("has_clik_api")
    )
    return {
        "camb_available": bool(modules.get("camb", {}).get("importable")),
        "cobaya_available": bool(modules.get("cobaya", {}).get("importable")),
        "healpy_available": bool(modules.get("healpy", {}).get("importable")),
        "official_clik_api_available": official_clik_ready,
        "official_planck_likelihood_data_paths_configured": False,
        "official_likelihood_execution_ready": False,
        "modules": modules,
        "claim_boundary": (
            "Runtime readiness check only. Official Planck likelihood execution also needs the ESA PR3 "
            "likelihood data files and validated nuisance/path configuration."
        ),
    }


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


def _scale_compressed_cmb_markdown_report(report: dict[str, Any]) -> str:
    comparisons = report["comparison"]
    branch = report["scale_compressed_input"]
    acoustic = report.get("acoustic_preservation", {})
    software = report.get("software", {})
    lines = [
        "# OPH Scale-Compressed CMB CAMB Transfer",
        "",
        report["claim_boundary"],
        "",
        "## Scale-Compressed Input",
        "",
        f"- logical repair rounds: {_fmt(branch.get('logical_repair_rounds'))}",
        f"- operator receipt: {branch.get('scale_compressed_operator_receipt')}",
        f"- round-trace receipt: {branch.get('repair_round_trace_receipt')}",
        f"- populated H3 preview receipt: {branch.get('populated_h3_preview_receipt')}",
        f"- strict neutral bulk: {branch.get('strict_neutral_bulk')}",
        f"- eta_R: {_fmt(branch.get('eta_R'))}",
        f"- n_s: {_fmt(branch.get('n_s'))}",
        f"- q_IR: {_fmt(branch.get('q_IR'))}",
        f"- ell_IR: {_fmt(branch.get('ell_IR'))}",
        f"- finite-lattice derived: {branch.get('finite_lattice_derived')}",
        "",
        "## CAMB TT Comparison",
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
                f"- benchmark first peak ell: {_fmt(comparison.get('benchmark_first_peak_ell'))}",
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
            "",
            "## Reproducibility",
            "",
            f"- CAMB: {software.get('camb_version', 'n/a')}",
            f"- benchmark SHA256: {report.get('input_hashes', {}).get('benchmark_sha256', 'n/a')}",
            f"- scale report SHA256: {report.get('input_hashes', {}).get('scale_compressed_report_sha256', 'n/a')}",
            "",
        ]
    )
    return "\n".join(lines)


def _oph_exact_cmb_markdown_report(report: dict[str, Any]) -> str:
    comparisons = report["comparison"]
    oph = report["oph_exact_input"]
    selector = report.get("selector_elimination_v1_5", {})
    selector_core = selector.get("selector_elimination", {})
    readiness = report.get("official_planck_likelihood_readiness", {})
    acoustic = report.get("acoustic_preservation", {})
    software = report.get("software", {})
    lines = [
        "# OPH Exact CMB CAMB Transfer",
        "",
        report["claim_boundary"],
        "",
        "## Exact OPH Input",
        "",
        f"- n_s = 1 - e alpha sqrt(pi): {_fmt(oph.get('n_s'))}",
        f"- eta_R: {_fmt(oph.get('eta_R'))}",
        f"- q_IR: {_fmt(oph.get('q_IR'))}",
        f"- ell_IR: {_fmt(oph.get('ell_IR'))}",
        f"- N_frz proxy: {_fmt(oph.get('N_frz_proxy'))}",
        f"- selector-elimination theorem receipt: {oph.get('selector_elimination_theorem_receipt')}",
        f"- selector-elimination source audit: {oph.get('selector_elimination_source_audit_receipt')}",
        f"- kappa_rep status: {oph.get('kappa_rep_status')}",
        f"- finite-lattice derived: {oph.get('finite_lattice_derived')}",
        "",
        "## Selector Elimination v1.5",
        "",
        f"- q_IR selector removed: {selector_core.get('q_IR_selector_removed')}",
        f"- ell_IR selector removed: {selector_core.get('ell_IR_selector_removed')}",
        f"- eta_R reduced to repair-clock certificate: {selector_core.get('eta_R_reduced_to_repair_clock_certificate')}",
        f"- remaining eta_R certificate: {selector_core.get('remaining_eta_R_certificate')}",
        "",
        "## CAMB TT Comparison",
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
            f"- ell {row['ell']}: amplitude-matched OPH_exact_IR/LCDM TT = "
            f"{_fmt(row.get('amplitude_matched_OPH_IR_over_LCDM_TT'))}"
        )
    lines.extend(
        [
            "",
            "## Official Planck Likelihood Readiness",
            "",
            f"- CAMB available: {readiness.get('camb_available')}",
            f"- Cobaya available: {readiness.get('cobaya_available')}",
            f"- official clik API available: {readiness.get('official_clik_api_available')}",
            f"- official execution ready: {readiness.get('official_likelihood_execution_ready')}",
            "",
            "## Reproducibility",
            "",
            f"- Python: {software.get('python_version', 'n/a')}",
            f"- NumPy: {software.get('numpy_version', 'n/a')}",
            f"- CAMB: {software.get('camb_version', 'n/a')}",
            "",
            "## Output Files",
            "",
            "- `oph_exact_cmb_camb_report.json`",
            "- `oph_exact_cmb_tt_bins.csv`",
            "- `oph_exact_cmb_tt_curves.csv`",
            "",
        ]
    )
    return "\n".join(lines)


def _finite_repair_clock_cmb_markdown_report(report: dict[str, Any]) -> str:
    comparisons = report["comparison"]
    finite = report["finite_repair_clock_input"]
    selector = report["selector_ir_input"]
    acoustic = report.get("acoustic_preservation", {})
    software = report.get("software", {})
    lines = [
        "# Finite Repair-Clock CMB CAMB Transfer",
        "",
        report["claim_boundary"],
        "",
        "## Simulator-Derived Clock Input",
        "",
        f"- n_s: {_fmt(finite.get('n_s'))}",
        f"- eta_R: {_fmt(finite.get('eta_R'))}",
        f"- kappa_rep: {_fmt(finite.get('kappa_rep'))}",
        f"- matrix ready: {finite.get('matrix_ready')}",
        f"- finite-lattice derived: {finite.get('finite_lattice_derived')}",
        f"- repair-clock certificate: {finite.get('repair_clock_certificate')}",
        f"- clock normalization certified: {finite.get('clock_normalization_certified')}",
        f"- clock normalization numeric match: {finite.get('clock_normalization_numeric_match')}",
        f"- repair-scale hypothesis clock match: {finite.get('repair_scale_hypothesis_clock_match')}",
        f"- clock normalization source: {finite.get('clock_normalization_source')}",
        f"- source mode: {finite.get('source_mode')}",
        f"- primary matrix: {finite.get('primary_matrix')}",
        f"- repair step time: {_fmt(finite.get('repair_step_time'))}",
        "",
        "## Selector IR Overlay",
        "",
        f"- q_IR: {_fmt(selector.get('q_IR'))}",
        f"- ell_IR: {_fmt(selector.get('ell_IR'))}",
        f"- theorem receipt: {selector.get('selector_elimination_theorem_receipt')}",
        f"- finite-lattice derived: {selector.get('finite_lattice_derived')}",
        "",
        "## CAMB TT Comparison",
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
                f"- benchmark first peak ell: {_fmt(comparison.get('benchmark_first_peak_ell'))}",
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
            "",
            "## Blockers",
            "",
        ]
    )
    blockers = finite.get("blockers") or []
    lines.extend(f"- {item}" for item in blockers)
    if not blockers:
        lines.append("- none reported by finite clock source")
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Python: {software.get('python_version', 'n/a')}",
            f"- NumPy: {software.get('numpy_version', 'n/a')}",
            f"- CAMB: {software.get('camb_version', 'n/a')}",
            "",
            "## Output Files",
            "",
            "- `finite_repair_clock_cmb_camb_report.json`",
            "- `finite_repair_clock_cmb_tt_bins.csv`",
            "- `finite_repair_clock_cmb_tt_curves.csv`",
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


def _exact_cmb_source_status(source_dir: Path | None) -> dict[str, Any]:
    if source_dir is None:
        return {"source_dir": None, "files": {}, "claim_boundary": "no source directory was provided"}
    candidates = [
        "OPH-CMB-Official-Likelihood-and-Finite-Patch-v1.0.md",
        "finite_patch_cmb_derivations_v1_0.md",
        "official_likelihood_and_math_status_v1_0.csv",
        "missing_math_and_likelihood_gates_v1_0.csv",
        "OPH-Unique-Prediction-Gate-v0.9.md",
        "01_unique_prediction_ranking_v0_9.csv",
        "02_public_assessment_table_v0_9.csv",
        "oph_camb_generate_cls.py",
        "oph_official_planck_clik_eval.py",
        "oph_cobaya_likelihood_selfcontained.py",
        "OPH-CMB-Selector-Elimination-v1.5.md",
        "comms3-remove-all-selectors.md",
        "selector_elimination_status_v1_5.csv",
        "exact_ir_kernel_values_v1_5.csv",
        "OPH-CMB-selector-elimination-v1.5.zip",
        "math/OPH-CMB-Selector-Elimination-v1.5.md",
        "math/no_remaining_selectors_theorems_v1_5.md",
        "data/selector_elimination_status_v1_5.csv",
        "data/exact_ir_kernel_values_v1_5.csv",
        "data/selector_elimination_summary_v1_5.json",
        "data/numerical_targets_v1_5.json",
    ]
    files = {}
    for relative in candidates:
        path = source_dir / relative
        files[relative] = {
            "present": path.exists(),
            "sha256": _sha256_file(path) if path.exists() else None,
        }
    legacy_core_present = all(
        files[name]["present"]
        for name in (
            "OPH-CMB-Official-Likelihood-and-Finite-Patch-v1.0.md",
            "finite_patch_cmb_derivations_v1_0.md",
            "OPH-Unique-Prediction-Gate-v0.9.md",
        )
    )
    selector_v15_top_level_present = all(
        files[name]["present"]
        for name in (
            "OPH-CMB-Selector-Elimination-v1.5.md",
            "selector_elimination_status_v1_5.csv",
            "exact_ir_kernel_values_v1_5.csv",
        )
    )
    selector_v15_extracted_present = all(
        files[name]["present"]
        for name in (
            "math/OPH-CMB-Selector-Elimination-v1.5.md",
            "data/selector_elimination_status_v1_5.csv",
            "data/exact_ir_kernel_values_v1_5.csv",
        )
    )
    return {
        "source_dir": str(source_dir),
        "files": files,
        "legacy_v1_core_files_present": legacy_core_present,
        "selector_v1_5_core_files_present": bool(selector_v15_top_level_present or selector_v15_extracted_present),
        "all_core_files_present": bool(legacy_core_present or selector_v15_top_level_present or selector_v15_extracted_present),
    }
