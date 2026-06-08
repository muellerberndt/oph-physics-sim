from __future__ import annotations

import json
import hashlib
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.cosmology.cmb_compare import load_planck_tt_binned


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


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.10g}"
    return "n/a"
