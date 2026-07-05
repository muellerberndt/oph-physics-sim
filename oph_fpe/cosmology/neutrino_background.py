from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from oph_fpe.claims import COSMOLOGY_PERTURBATION_RECEIPT, QUANTITATIVE_BRANCH, with_claim_metadata
from oph_fpe.constants.oph_pixel import P_STAR
from oph_fpe.cosmology.oph_constants import OPHConstants
from oph_fpe.cosmology.oph_kernels import (
    apply_projected_wl_selector,
    compressed_projection_fraction,
    projected_amplitude,
)
from oph_fpe.cosmology.unique_predictions import neutrino_cosmology_report


DEFAULT_SOURCE_DIR = Path("/Users/muellerberndt/Projects/oph-meta/cosmology/correspondence/neutrinos")

PLANCK_NEFF_VALUE = 2.99
PLANCK_NEFF_SIGMA = 0.17
PLANCK_BAO_SUM_MNU_BOUND_EV = 0.12
ACT_DR6_EXTENDED_SUM_MNU_BOUND_EV = 0.082
DESI_DR2_LCDM_SUM_MNU_BOUND_EV = 0.0642
DESI_DR2_OSC_LIGHTEST_BOUND_EV = 0.023

S8_CDM_BRANCH = 0.828924043
WEAK_LENSING_S8_TARGET = 0.790


def oph_cnb_background_report(
    source_dir: Path | None = DEFAULT_SOURCE_DIR,
    *,
    p_value: float = P_STAR,
    h: float = 0.674,
    omega_m: float = 0.3155,
    delta_neff_coh: float = 0.0,
) -> dict[str, Any]:
    """Return the OPH relic-neutrino measurement lane.

    This report treats the OPH weighted-cycle neutrino masses as an imported
    target from the particle/quantitative branch and propagates them through
    standard relic-neutrino cosmology. It deliberately does not claim the
    current finite screen run has derived those masses or the late repair
    kernel from collar state data.
    """

    background = dict(neutrino_cosmology_report(h=h, omega_m=omega_m))
    masses = [float(item) for item in background["masses_eV"]]
    n_eff = 3.044 + float(delta_neff_coh)
    sum_mnu = float(background["sum_mnu_eV"])
    omega_nu_h2 = float(background["Omega_nu_h2"])
    omega_nu = float(background["Omega_nu"])
    f_nu = float(background["f_nu"])
    oph_constants = OPHConstants(
        P=float(p_value),
        S8_oph_compressed=float(S8_CDM_BRANCH),
        S8_wl_target_reference=float(WEAK_LENSING_S8_TARGET),
    )
    eta_a = oph_constants.reserve
    epsilon_required = 1.0 - float(WEAK_LENSING_S8_TARGET) / float(S8_CDM_BRANCH)
    pi_wl_required = epsilon_required / eta_a if eta_a > 0.0 else math.nan
    s8_five_of_seven = apply_projected_wl_selector(S8_CDM_BRANCH, constants=oph_constants)
    s8_five_of_seven_pull = (s8_five_of_seven - WEAK_LENSING_S8_TARGET) / oph_constants.S8_wl_sigma_reference
    pi_wl_projected = compressed_projection_fraction(
        WEAK_LENSING_S8_TARGET,
        S8_CDM_BRANCH,
        constants=oph_constants,
    )
    report = {
        "mode": "oph_cnb_neutrino_background_v0",
        "source_files": _source_status(source_dir),
        "inputs": {
            "P": float(p_value),
            "h_reference": float(h),
            "Omega_m_reference": float(omega_m),
            "Delta_N_eff_coh": float(delta_neff_coh),
            "S8_cdm_branch": float(S8_CDM_BRANCH),
            "weak_lensing_S8_target": float(WEAK_LENSING_S8_TARGET),
        },
        "oph_neutrino_branch": {
            "mass_ordering": "normal",
            "masses_eV": masses,
            "sum_mnu_eV": sum_mnu,
            "delta_m21_squared_eV2": float(masses[1] ** 2 - masses[0] ** 2),
            "delta_m31_squared_eV2": float(masses[2] ** 2 - masses[0] ** 2),
            "m_lightest_eV": min(masses),
            "m_beta_proxy_eV": background.get("m_beta_proxy_eV"),
            "finite_lattice_derived": False,
            "source": "OPH weighted-cycle absolute neutrino target from local cosmology notes",
        },
        "relic_background": {
            "N_eff": float(n_eff),
            "baseline_N_eff": 3.044,
            "Delta_N_eff_coh": float(delta_neff_coh),
            "T_nu0_K": background["T_nu0_K"],
            "number_density_total_cm3": background["number_density_total_cm3"],
            "Omega_nu_h2": omega_nu_h2,
            "Omega_nu": omega_nu,
            "f_nu": f_nu,
            "small_scale_power_suppression_fraction": float(background["small_scale_power_suppression_fraction"]),
            "growth_exponent_deformation": float(background["growth_exponent_deformation"]),
        },
        "free_streaming": _free_streaming_rows(background),
        "measurement_comparisons": _measurement_comparisons(
            n_eff=n_eff,
            sum_mnu=sum_mnu,
            lightest_mass=min(masses),
            f_nu=f_nu,
        ),
        "late_repair_projection_target": {
            "theorem_form": "L_OPH/L_0 = 1 - eta_A * Pi_L",
            "eta_A": eta_a,
            "eta_A_formula": "1 - lambda_collar",
            "lambda_collar_exact_uniform_product_thickening": (
                oph_constants.lambda_collar_exact_uniform_product_thickening
            ),
            "lambda_collar_exact_gate": oph_constants.lambda_collar_exact_gate,
            "lambda_collar_exact_gate_pass": False,
            "lambda_collar_profile_default": oph_constants.lambda_collar_profile_default,
            "finite_thickness_jensen_band": oph_constants.finite_thickness_jensen_band,
            "z6_normalized_trace_mean": oph_constants.z6_normalized_trace_mean,
            "z6_reciprocal_trace": oph_constants.z6_reciprocal_trace,
            "epsilon_required_to_map_cdm_S8_to_WL_target": epsilon_required,
            "Pi_WL_compressed_required": pi_wl_projected,
            "Pi_WL_is_universal_microphysical_constant": False,
            "projected_amplitude_theorem": {
                "formula": "L_OPH/L_0 = 1 - eta_A Pi_L",
                "Pi_L_definition": "Pi_L = integral K_L(k,a) W(k,a) dlnk dlna",
                "minimal_kernel": "B_A(k,a)=1-(1-lambda_collar) W_k(k) W_a(a)",
                "W_k": "k^2/(k^2+k_A^2)",
                "W_a": "Xi_A/(Xi_A+H_conformal)",
                "compressed_target_not_microphysical_constant": True,
                "exact_uniform_target_requires": oph_constants.lambda_collar_exact_gate,
                "required_Pi_WL_from_compressed_rows": pi_wl_projected,
                "reconstructed_S8_from_required_Pi_WL": projected_amplitude(
                    S8_CDM_BRANCH,
                    pi_wl_projected,
                    constants=oph_constants,
                ),
                "claim_boundary": (
                    "Pi_WL is a projection of the derived OPH window through an observable/survey kernel. "
                    "The finite simulator must derive W(k,a), k_A, and Xi_A; it must not fit Pi_WL directly."
                ),
            },
            "z6_poisson_five_of_seven": {
                "branch": "z6_poisson_five_of_seven",
                "kernel_formula": "B_A(k,a)=1-(5/7)(1-lambda_collar) W_k(k) W_a(a)",
                "lambda_collar": oph_constants.lambda_collar,
                "lambda_collar_status": oph_constants.lambda_collar_claim_status,
                "lambda_collar_exact_gate": oph_constants.lambda_collar_exact_gate,
                "lambda_collar_exact_gate_pass": False,
                "reserve": oph_constants.reserve,
                "pi_wl": oph_constants.pi_wl,
                "epsilon_A_wl": oph_constants.epsilon_A_wl,
                "R_wl": oph_constants.R_wl,
                "S8_projected_from_cdm_branch": s8_five_of_seven,
                "S8_target_reference": float(WEAK_LENSING_S8_TARGET),
                "S8_sigma_reference": oph_constants.S8_wl_sigma_reference,
                "pull_sigma_reference": s8_five_of_seven_pull,
                "compressed_scorecard_mode_supported": True,
                "boltzmann_lite_kernel_callable": True,
                "finite_collar_emitted_kA_tau_rec": False,
                "claim_boundary": (
                    "Five-of-seven branch is a theorem-motivated compressed/kernel target. It is not a "
                    "full weak-lensing likelihood and does not close the B_A-from-finite-collar gate."
                ),
            },
            "S8_cdm_branch": float(S8_CDM_BRANCH),
            "S8_weak_lensing_target": float(WEAK_LENSING_S8_TARGET),
            "missing_inputs": [
                "finite_collar_parent_W_k_a",
                "observable_kernel_K_L_for_each_survey",
                "full_CAMB_CLASS_anomaly_module",
                "survey_covariance_likelihood",
            ],
            "claim_boundary": (
                "The compressed Pi_WL value is a target for pairing a derived OPH W(k,a) with a declared "
                "weak-lensing response kernel. It is not a universal OPH constant and is not derived by this report."
            ),
        },
        "theorem_package_coverage": {
            "static_to_linear_separation_declared": True,
            "parent_finite_collar_response_functional_target": True,
            "linear_repair_transfer_equation_available": True,
            "collar_reserve_diagnostic_target_available": True,
            "collar_reserve_amplitude_normalization_available": False,
            "uniform_product_thickening_exact_gate": False,
            "scalar_weighted_z6_mean_gate": False,
            "minimal_one_pole_kernel_callable": True,
            "projected_amplitude_theorem_available": True,
            "compressed_weak_lensing_target_available": True,
            "no_universal_weak_lensing_projection_constant_declared": True,
            "finite_collar_window_derived_by_simulator": False,
            "claim_boundary": (
                "The theorem package is represented as diagnostic equations and gates. The simulator still "
                "needs finite-collar packet data to derive the actual W(k,a) window and close B_A(k,a)."
            ),
        },
        "camb_class_inputs": {
            "nnu": float(n_eff),
            "num_massive_neutrinos": len(masses),
            "mnu_sum_eV": sum_mnu,
            "neutrino_hierarchy": "normal",
            "omnuh2": omega_nu_h2,
            "individual_masses_eV": masses,
            "note": (
                "CAMB commonly accepts a total mnu and neutrino_hierarchy; exact per-eigenstate transfer "
                "requires solver-specific setup and official likelihood validation."
            ),
        },
        "readiness_gates": {
            "measurement_comparable_relic_background": True,
            "finite_lattice_mass_derivation": False,
            "Delta_N_eff_coh_channel_declared": bool(float(delta_neff_coh) != 0.0),
            "z6_poisson_five_of_seven_kernel_callable": True,
            "z6_poisson_five_of_seven_compressed_projection": True,
            "uniform_product_thickening_exact_gate": False,
            "scalar_weighted_z6_mean_gate": False,
            "B_A_k_a_from_finite_collar_parent": False,
            "survey_projection_kernel_declared": False,
            "full_boltzmann_likelihood_run": False,
        },
        "measurement_comparable_now": True,
        "finite_lattice_derived": False,
        "physical_cmb_prediction": False,
        "physical_matter_power_prediction": False,
        "claim_boundary": (
            "OPH-CnuB target/readout lane. The neutrino masses are imported from the OPH weighted-cycle "
            "quantitative branch and propagated through standard relic-neutrino cosmology. Do not interpret "
            "this as a finite-lattice derivation, an official Planck/DESI likelihood, or a completed OPH "
            "late-repair Boltzmann kernel."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=QUANTITATIVE_BRANCH,
        receipt=COSMOLOGY_PERTURBATION_RECEIPT,
        physical_claim=False,
        observable_id="oph_cnb_neutrino_background",
        fit_objective="neutrino_relic_background_public_comparison",
    )


def write_oph_cnb_background_report(source_dir: Path | None, out_dir: Path, **kwargs: Any) -> dict[str, Any]:
    report = oph_cnb_background_report(source_dir, **kwargs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "oph_cnb_neutrino_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "oph_cnb_neutrino_report.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_csv(out / "oph_cnb_neutrino_mass_rows.csv", _mass_rows(report))
    _write_csv(out / "oph_cnb_neutrino_comparison_rows.csv", _comparison_rows(report))
    _write_csv(out / "oph_cnb_free_streaming_rows.csv", report["free_streaming"])
    return report


def _source_status(source_dir: Path | None) -> dict[str, Any]:
    if source_dir is None:
        return {"source_dir": None, "files": {}, "all_expected_files_present": False}
    source = Path(source_dir)
    expected = ["neutrinos-1.md", "neutrinos2.md", "neutrino3.md"]
    files = {}
    for name in expected:
        path = source / name
        files[name] = {
            "path": str(path),
            "present": path.exists(),
            "byte_size": path.stat().st_size if path.exists() else None,
            "sha256": _sha256_file_or_none(path),
        }
    return {
        "source_dir": str(source),
        "files": files,
        "all_expected_files_present": all(item["present"] for item in files.values()),
        "empty_files": [name for name, item in files.items() if item["present"] and item["byte_size"] == 0],
    }


def _free_streaming_rows(background: dict[str, Any]) -> list[dict[str, Any]]:
    masses = background["masses_eV"]
    z_nr = background["z_nonrelativistic"]
    k_fs0 = background["k_FS0_h_Mpc_inverse"]
    k_nr = background["k_nr_h_Mpc_inverse"]
    rows = []
    for index, mass in enumerate(masses):
        k_fs = float(k_fs0[index])
        rows.append(
            {
                "state": f"nu_{index + 1}",
                "mass_eV": float(mass),
                "z_nonrelativistic": float(z_nr[index]),
                "k_FS0_h_Mpc_inverse": k_fs,
                "lambda_FS0_h_inverse_Mpc": float(2.0 * math.pi / k_fs) if k_fs > 0 else None,
                "k_nr_h_Mpc_inverse": float(k_nr[index]),
            }
        )
    return rows


def _measurement_comparisons(*, n_eff: float, sum_mnu: float, lightest_mass: float, f_nu: float) -> dict[str, Any]:
    return {
        "Planck2018_N_eff": {
            "value": PLANCK_NEFF_VALUE,
            "sigma": PLANCK_NEFF_SIGMA,
            "pull_sigma": float((n_eff - PLANCK_NEFF_VALUE) / PLANCK_NEFF_SIGMA),
            "source": "local neutrino notes; public Planck compressed comparison",
        },
        "Planck2018_BAO_sum_mnu_bound": {
            "upper_95_eV": PLANCK_BAO_SUM_MNU_BOUND_EV,
            "sum_mnu_eV": float(sum_mnu),
            "passes_bound": bool(sum_mnu < PLANCK_BAO_SUM_MNU_BOUND_EV),
            "margin_eV": float(PLANCK_BAO_SUM_MNU_BOUND_EV - sum_mnu),
        },
        "ACT_DR6_extended_sum_mnu_bound": {
            "upper_95_eV": ACT_DR6_EXTENDED_SUM_MNU_BOUND_EV,
            "sum_mnu_eV": float(sum_mnu),
            "passes_bound": bool(sum_mnu < ACT_DR6_EXTENDED_SUM_MNU_BOUND_EV),
            "margin_eV": float(ACT_DR6_EXTENDED_SUM_MNU_BOUND_EV - sum_mnu),
            "model_dependent": True,
        },
        "DESI_DR2_LCDM_sum_mnu_bound": {
            "upper_95_eV": DESI_DR2_LCDM_SUM_MNU_BOUND_EV,
            "sum_mnu_eV": float(sum_mnu),
            "passes_bound": bool(sum_mnu < DESI_DR2_LCDM_SUM_MNU_BOUND_EV),
            "margin_eV": float(DESI_DR2_LCDM_SUM_MNU_BOUND_EV - sum_mnu),
            "model_dependent": True,
        },
        "DESI_DR2_oscillation_lightest_mass": {
            "upper_95_eV": DESI_DR2_OSC_LIGHTEST_BOUND_EV,
            "lightest_mass_eV": float(lightest_mass),
            "passes_bound": bool(lightest_mass < DESI_DR2_OSC_LIGHTEST_BOUND_EV),
            "margin_eV": float(DESI_DR2_OSC_LIGHTEST_BOUND_EV - lightest_mass),
            "model_dependent": True,
        },
        "linear_small_scale_power_suppression": {
            "f_nu": float(f_nu),
            "delta_P_over_P_approx": float(-8.0 * f_nu),
            "interpretation": "broad low-amplitude normal-ordering free-streaming imprint",
        },
    }


def _mass_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    free = {row["state"]: row for row in report["free_streaming"]}
    for index, mass in enumerate(report["oph_neutrino_branch"]["masses_eV"]):
        state = f"nu_{index + 1}"
        rows.append({"state": state, "mass_eV": mass, **free.get(state, {})})
    return rows


def _comparison_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, values in report["measurement_comparisons"].items():
        rows.append({"comparison": name, **values})
    rows.append({"comparison": "late_repair_projection_target", **report["late_repair_projection_target"]})
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
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


def _markdown_report(report: dict[str, Any]) -> str:
    branch = report["oph_neutrino_branch"]
    relic = report["relic_background"]
    wl = report["late_repair_projection_target"]
    comparisons = report["measurement_comparisons"]
    return "\n".join(
        [
            "# OPH-CnuB Neutrino Background",
            "",
            f"- mode: `{report['mode']}`",
            f"- mass ordering: `{branch['mass_ordering']}`",
            f"- masses eV: `{', '.join(f'{item:.12f}' for item in branch['masses_eV'])}`",
            f"- sum m_nu: `{branch['sum_mnu_eV']:.12f}` eV",
            f"- N_eff: `{relic['N_eff']:.6f}`",
            f"- Omega_nu h^2: `{relic['Omega_nu_h2']:.9g}`",
            f"- f_nu: `{relic['f_nu']:.9g}`",
            f"- small-scale Delta P/P approx: `{relic['small_scale_power_suppression_fraction']:.6f}`",
            f"- Planck N_eff pull: `{comparisons['Planck2018_N_eff']['pull_sigma']:.3f} sigma`",
            f"- Planck+BAO sum-mnu bound passes: `{comparisons['Planck2018_BAO_sum_mnu_bound']['passes_bound']}`",
            f"- DESI DR2 strict LCDM sum-mnu bound passes: `{comparisons['DESI_DR2_LCDM_sum_mnu_bound']['passes_bound']}`",
            f"- required compressed Pi_WL: `{wl['Pi_WL_compressed_required']:.10f}`",
            "",
            "## Gates",
            "",
            *[f"- {name}: `{str(value).lower()}`" for name, value in report["readiness_gates"].items()],
            "",
            "## Claim Boundary",
            "",
            str(report["claim_boundary"]),
            "",
        ]
    )
