from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from oph_fpe.constants.oph_pixel import OPHPixelConstants, P_STAR
from oph_fpe.cosmology.oph_constants import OPHConstants
from oph_fpe.cosmology.neutrino_status import neutrino_mass_status
from oph_fpe.cosmology.oph_screen_power import DEFAULT_D_STAR_MPC, DEFAULT_K0_MPC
from oph_fpe.cosmology.selector_elimination import selector_elimination_report


PLANCK_NS_TARGET = 0.965
PLANCK_NS_SIGMA = 0.004
PLANCK_RATIO_OMEGA_C_OVER_B = 5.357142857
PLANCK_RATIO_OMEGA_C_OVER_B_SIGMA = 0.050645345
ACT_PLANCK_S8 = 0.831
ACT_PLANCK_S8_SIGMA = 0.023
V09_PARITY_R_OE_TT_2_29 = 1.2160638411338078

H_REFERENCE = 0.674
OMEGA_M_REFERENCE = 0.3155
T_NU0_K = 1.945
NEUTRINO_NUMBER_DENSITY_CM3 = 339.5


def unique_prediction_gate_report(
    source_dir: Path | None = None,
    *,
    P: float = P_STAR,
    include_rejected_weighted_cycle_benchmark: bool = False,
) -> dict[str, Any]:
    """Return the current OPH-only public-comparison prediction gate.

    This imports the v0.9 local cosmology note when present and computes the
    same compact numerical targets from OPH constants. The report is a target
    and public-data comparison lane, not evidence that a finite lattice run has
    derived the targets from cap/collar state statistics.
    """

    selector_report = selector_elimination_report(source_dir, P=float(P))
    pixel = OPHPixelConstants(P=float(P))
    oph_constants = OPHConstants(P=float(P))
    eta_r = float(selector_report["scalar_tilt"]["eta_R"])
    n_s = float(selector_report["scalar_tilt"]["n_s"])
    q_ir = float(selector_report["cmb_ir_kernel"]["q_IR"])
    ell_ir = float(selector_report["cmb_ir_kernel"]["ell_IR"])
    theta_ir_deg = 180.0 / ell_ir
    k_ir_mpc = ell_ir / DEFAULT_D_STAR_MPC
    n_frz_proxy = int((ell_ir + 1.0) ** 2)
    chi_nu = oph_constants.lambda_collar_exact_uniform_product_thickening
    ranking_rows: list[dict[str, Any]] = []
    assessment_rows: list[dict[str, Any]] = []
    files: dict[str, Any] = {}
    if source_dir is not None:
        source = Path(source_dir)
        ranking_path = _first_existing(
            source,
            "01_unique_prediction_ranking_v0_9.csv",
            "5/01_unique_prediction_ranking_v0_9.csv",
        )
        assessment_path = _first_existing(
            source,
            "02_public_assessment_table_v0_9.csv",
            "5/02_public_assessment_table_v0_9.csv",
        )
        note_path = _first_existing(source, "OPH-Unique-Prediction-Gate-v0.9.md", "5/OPH-Unique-Prediction-Gate-v0.9.md")
        ranking_rows = _read_csv(ranking_path)
        assessment_rows = _read_csv(assessment_path)
        files = {
            "source_dir": str(source),
            "ranking_csv": str(ranking_path),
            "assessment_csv": str(assessment_path),
            "note_md": str(note_path),
            "ranking_csv_present": ranking_path.exists(),
            "assessment_csv_present": assessment_path.exists(),
            "note_md_present": note_path.exists(),
            "ranking_csv_sha256": _sha256_file_or_none(ranking_path),
            "assessment_csv_sha256": _sha256_file_or_none(assessment_path),
            "note_md_sha256": _sha256_file_or_none(note_path),
        }
    neutrino = neutrino_cosmology_report(
        include_rejected_weighted_cycle_benchmark=include_rejected_weighted_cycle_benchmark
    )
    report = {
        "mode": "oph_unique_prediction_gate_v0_9",
        "oph_constants": pixel.as_jsonable(),
        "source_files": files,
        "scalar_tilt": {
            "formula": "n_s = 1 - kappa_rep * alpha(0) * sqrt(pi)",
            "canonical_kappa_rep": float(selector_report["scalar_tilt"]["canonical_kappa_rep"]),
            "canonical_kappa_rep_status": selector_report["scalar_tilt"]["canonical_kappa_rep_status"],
            "eta_R": float(eta_r),
            "n_s": float(n_s),
            "planck_public_target": PLANCK_NS_TARGET,
            "planck_public_sigma": PLANCK_NS_SIGMA,
            "pull_vs_planck_sigma": float((n_s - PLANCK_NS_TARGET) / PLANCK_NS_SIGMA),
            "comparison_status": "public_parameter_comparable_now",
        },
        "cmb_ir_kernel": {
            "formula": "F_IR(ell)=1-q_IR*exp[-ell(ell+1)/(ell_IR(ell_IR+1))]",
            "q_IR": q_ir,
            "ell_IR": ell_ir,
            "theta_IR_deg": theta_ir_deg,
            "k_IR_Mpc_inverse": k_ir_mpc,
            "D_star_Mpc": DEFAULT_D_STAR_MPC,
            "N_frz_proxy": n_frz_proxy,
            "diagnostic_delta_chi2_TT_TE_EE_lowell_v0_8": 9.242,
            "diagnostic_delta_AIC_v0_8": -7.242,
            "diagnostic_delta_BIC_v0_8": -4.811,
            "official_planck_likelihood_run": False,
        },
        "selector_elimination_v1_5": {
            "q_IR_selector_removed": selector_report["selector_elimination"]["q_IR_selector_removed"],
            "ell_IR_selector_removed": selector_report["selector_elimination"]["ell_IR_selector_removed"],
            "eta_R_reduced_to_repair_clock_certificate": selector_report["selector_elimination"][
                "eta_R_reduced_to_repair_clock_certificate"
            ],
            "remaining_eta_R_certificate": selector_report["selector_elimination"]["remaining_eta_R_certificate"],
            "theorem_side_receipt": selector_report["THEOREM_SIDE_SELECTOR_ELIMINATION_RECEIPT"],
            "source_packet_audit_receipt": selector_report["SOURCE_PACKET_AUDIT_RECEIPT"],
            "source_status_audit": selector_report["source_status_audit"],
            "exact_ir_kernel_csv_audit": {
                key: value
                for key, value in selector_report["exact_ir_kernel_csv_audit"].items()
                if key != "rows"
            },
        },
        "parity_envelope": {
            "formula": "F_P(ell)=1-(-1)^ell*exp(-ell/4)",
            "predicted_R_OE_TT_2_29": V09_PARITY_R_OE_TT_2_29,
            "unweighted_envelope_R_OE_2_29_debug": parity_odd_even_ratio(range(2, 30)),
            "fitted_parity_diagnostic_R_OE_TT_2_29": 1.2787125962633745,
            "planck_PR3_R_OE_TT_2_29": 1.310820,
            "lcdm_exact_tilt_R_OE_TT_2_29": 1.002724,
            "parity_is_scalar_transfer": False,
            "required_next_test": "map-space parity/BipoSH with masks and a-posteriori penalty",
        },
        "neutrino_cosmology": neutrino,
        "compressed_dark_sector": {
            "a0_eff_m_s2": 1.179018696e-10,
            "rho_A_over_rho_b": 5.363470441,
            "Omega_A_source": "RESIDUAL_DEFINITION_FLAT_BENCHMARK",
            "Omega_K_source": "EXPLICIT_ASSUMPTION_FLAT_BENCHMARK",
            "rho_A_over_rho_b_planck_ratio": PLANCK_RATIO_OMEGA_C_OVER_B,
            "rho_A_over_rho_b_pull_sigma": (5.363470441 - PLANCK_RATIO_OMEGA_C_OVER_B)
            / PLANCK_RATIO_OMEGA_C_OVER_B_SIGMA,
            "S8": 0.828924037,
            "S8_ACT_Planck_lensing": ACT_PLANCK_S8,
            "S8_pull_sigma": (0.828924037 - ACT_PLANCK_S8) / ACT_PLANCK_S8_SIGMA,
            "claim_boundary": "compressed comparison target; full Boltzmann/growth likelihood not run here",
        },
        "coherent_matter_susceptibility": {
            "chi_nu_can": chi_nu,
            "formula": "lambda_profile; exact-uniform target exp(-P/24) only after gate closure",
            "lambda_collar_exact_gate": oph_constants.lambda_collar_exact_gate,
            "lambda_collar_exact_gate_pass": False,
            "lambda_collar_profile_default": oph_constants.lambda_collar_profile_default,
            "finite_thickness_jensen_band": oph_constants.finite_thickness_jensen_band,
            "z6_normalized_trace_mean": oph_constants.z6_normalized_trace_mean,
            "z6_reciprocal_trace": oph_constants.z6_reciprocal_trace,
            "public_cosmology_data_value": False,
            "claim_boundary": (
                "near-future laboratory target, not a CMB/public-data pass-fail value; the exact "
                "exp(-P/24) number is an exact-uniform/product-thickening target, not an unconditional "
                "finite-thickness scalar coefficient"
            ),
        },
        "ranking_rows": ranking_rows,
        "assessment_rows": assessment_rows,
        "measurement_comparable_now": True,
        "finite_lattice_derived": False,
        "physical_cmb_prediction": False,
        "claim_boundary": (
            "Current OPH-only prediction gate imported from local cosmology notes and recomputed from OPH "
            "constants. In the v1.5 selector-elimination surface, q_IR=1/4 and ell_IR=32 are theorem-side "
            "target counts rather than fit selectors; eta_R is reduced to the single repair-clock certificate "
            "kappa_rep=e. The scalar tilt, IR/parity anomaly templates, and compressed dark-sector rows are "
            "comparable to public measurements. OPH currently has no source-derived neutrino mass prediction; "
            "the 0.06 eV neutrino input is a conventional CAMB reference, while the old weighted-cycle row is a "
            "rejected retrospective benchmark available only by explicit opt-in. The remaining rows stay "
            "target/readout lanes until "
            "the finite OPH lattice derives kappa_rep, validates the finite-register IR/covariance certificates, "
            "derives parity covariance and anomaly kernels from state-derived cap/collar microphysics, and "
            "passes official likelihood/map-space tests."
        ),
    }
    return report


def neutrino_cosmology_report(
    *,
    h: float = H_REFERENCE,
    omega_m: float = OMEGA_M_REFERENCE,
    include_rejected_weighted_cycle_benchmark: bool = False,
) -> dict[str, Any]:
    status = neutrino_mass_status(
        include_rejected_benchmark=include_rejected_weighted_cycle_benchmark
    )
    conventional = status["conventional_camb_baseline"]
    conventional["cosmology"] = _propagate_neutrino_masses(
        tuple(float(value) for value in conventional["solver_mass_components_eV"]),
        h=float(h),
        omega_m=float(omega_m),
        mass_ordering="solver_reference_one_massive_two_massless",
    )
    rejected = status["historical_rejected_weighted_cycle_benchmark"]
    if rejected["included"]:
        rejected["cosmology"] = _propagate_neutrino_masses(
            tuple(float(value) for value in rejected["masses_eV"]),
            h=float(h),
            omega_m=float(omega_m),
            mass_ordering="normal",
            m_beta_proxy_eV=0.01956,
        )
    else:
        rejected["cosmology"] = None
    return status


def _propagate_neutrino_masses(
    masses_ev: tuple[float, ...],
    *,
    h: float,
    omega_m: float,
    mass_ordering: str,
    m_beta_proxy_eV: float | None = None,
) -> dict[str, Any]:
    mass_sum = float(sum(masses_ev))
    omega_nu_h2 = mass_sum / 93.12
    omega_nu = omega_nu_h2 / (float(h) ** 2)
    f_nu = omega_nu / float(omega_m)
    z_nr = [float(m / (3.15 * _k_boltzmann_ev_per_k() * T_NU0_K) - 1.0) for m in masses_ev]
    k_fs0 = [float(0.8 * math.sqrt(1.0) * m) for m in masses_ev]
    k_nr = [float(0.018 * math.sqrt(float(omega_m)) * math.sqrt(m)) for m in masses_ev]
    return {
        "N_eff": 3.044,
        "Delta_N_eff_coh": 0.0,
        "T_nu0_K": T_NU0_K,
        "number_density_total_cm3": NEUTRINO_NUMBER_DENSITY_CM3,
        "mass_ordering": str(mass_ordering),
        "masses_eV": [float(item) for item in masses_ev],
        "sum_mnu_eV": mass_sum,
        "m_beta_proxy_eV": None if m_beta_proxy_eV is None else float(m_beta_proxy_eV),
        "Omega_nu_h2": float(omega_nu_h2),
        "h_reference": float(h),
        "Omega_nu": float(omega_nu),
        "Omega_m_reference": float(omega_m),
        "f_nu": float(f_nu),
        "small_scale_power_suppression_fraction": float(-8.0 * f_nu),
        "growth_exponent_deformation": float(1.0 - 3.0 * f_nu / 5.0),
        "z_nonrelativistic": z_nr,
        "k_FS0_h_Mpc_inverse": k_fs0,
        "k_nr_h_Mpc_inverse": k_nr,
        "planck_bao_mass_bound_eV": 0.12,
        "desi_lcdm_mass_bound_eV": 0.064,
        "desi_w0wa_mass_bound_eV": 0.16,
        "claim_boundary": "Standard relic-neutrino propagation for a declared input; it is not a mass derivation.",
    }


def parity_envelope(ell: np.ndarray | list[float] | range) -> np.ndarray:
    ell_arr = np.asarray(list(ell), dtype=float)
    return 1.0 - ((-1.0) ** np.rint(ell_arr)) * np.exp(-ell_arr / 4.0)


def parity_odd_even_ratio(ell: np.ndarray | list[float] | range) -> float:
    ell_arr = np.asarray(list(ell), dtype=float)
    weights = parity_envelope(ell_arr)
    odd = (np.rint(ell_arr).astype(int) % 2) == 1
    even = ~odd
    return float(np.sum(weights[odd]) / max(float(np.sum(weights[even])), 1.0e-30))


def unique_ir_power(
    k_mpc: np.ndarray | list[float] | float,
    *,
    A_s: float,
    n_s: float,
    q_IR: float = 0.25,
    ell_IR: float = 32.0,
    k0_mpc: float = DEFAULT_K0_MPC,
    d_star_mpc: float = DEFAULT_D_STAR_MPC,
) -> np.ndarray:
    k = np.asarray(k_mpc, dtype=float)
    ell_proxy = np.maximum(k * float(d_star_mpc), 2.0)
    denom = max(float(ell_IR) * (float(ell_IR) + 1.0), 1.0e-12)
    f_ir = 1.0 - float(q_IR) * np.exp(-(ell_proxy * (ell_proxy + 1.0)) / denom)
    base = float(A_s) * (np.maximum(k, 1.0e-30) / float(k0_mpc)) ** (float(n_s) - 1.0)
    return np.asarray(base * np.maximum(f_ir, 0.0), dtype=float)


def write_unique_prediction_gate_report(
    source_dir: Path | None,
    out_dir: Path,
    *,
    include_rejected_weighted_cycle_benchmark: bool = False,
) -> dict[str, Any]:
    report = unique_prediction_gate_report(
        source_dir,
        include_rejected_weighted_cycle_benchmark=include_rejected_weighted_cycle_benchmark,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "oph_unique_prediction_gate_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "oph_unique_prediction_gate_report.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_csv(out / "oph_unique_prediction_ranking_rows.csv", report.get("ranking_rows", []))
    _write_csv(out / "oph_unique_prediction_assessment_rows.csv", report.get("assessment_rows", []))
    return report


def _first_existing(source: Path, *relative_paths: str) -> Path:
    candidates = [source / relative for relative in relative_paths]
    return next((path for path in candidates if path.exists()), candidates[0])


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file_or_none(path: Path) -> str | None:
    if not Path(path).exists():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _k_boltzmann_ev_per_k() -> float:
    return 8.617333262145e-5


def _markdown_report(report: dict[str, Any]) -> str:
    scalar = report["scalar_tilt"]
    ir = report["cmb_ir_kernel"]
    parity = report["parity_envelope"]
    nu = report["neutrino_cosmology"]
    dark = report["compressed_dark_sector"]
    return "\n".join(
        [
            "# OPH Unique Prediction Gate",
            "",
            f"- mode: `{report['mode']}`",
            f"- n_s: `{scalar['n_s']:.12f}`",
            f"- eta_R: `{scalar['eta_R']:.12f}`",
            f"- Planck pull: `{scalar['pull_vs_planck_sigma']:.3f} sigma`",
            f"- q_IR / ell_IR: `{ir['q_IR']}` / `{ir['ell_IR']}`",
            f"- theta_IR: `{ir['theta_IR_deg']:.6f} deg`",
            f"- k_IR: `{ir['k_IR_Mpc_inverse']:.8f} Mpc^-1`",
            f"- parity R_OE TT(2..29): `{parity['predicted_R_OE_TT_2_29']:.6f}`",
            f"- OPH-derived neutrino mass prediction available: `{nu['oph_derived_prediction']['available']}`",
            "- OPH-derived sum m_nu: `none`",
            f"- conventional CAMB baseline sum m_nu: `{nu['conventional_camb_baseline']['sum_mnu_eV']:.12f} eV`",
            f"- rejected weighted-cycle benchmark included: `{nu['historical_rejected_weighted_cycle_benchmark']['included']}`",
            f"- rho_A/rho_b: `{dark['rho_A_over_rho_b']:.9f}`",
            f"- S8: `{dark['S8']:.9f}`",
            "",
            "## Status",
            "",
            f"- measurement comparable now: `{report['measurement_comparable_now']}`",
            f"- finite lattice derived: `{report['finite_lattice_derived']}`",
            f"- physical CMB prediction: `{report['physical_cmb_prediction']}`",
            "",
            "## Claim Boundary",
            "",
            str(report["claim_boundary"]),
            "",
        ]
    )
