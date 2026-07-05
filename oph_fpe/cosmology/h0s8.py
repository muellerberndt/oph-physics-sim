from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from oph_fpe.claims import CONTINUATION, COSMOLOGY_PERTURBATION_RECEIPT, with_claim_metadata
from oph_fpe.constants.oph_pixel import P_STAR
from oph_fpe.cosmology.oph_constants import OPHConstants
from oph_fpe.cosmology.h0s8_certificates import h0s8_lane8_certificate_report


H_DS_KM_S_MPC = 55.759940256
OMEGA_B_H2 = 0.02237
OMEGA_R_H2 = 4.18343e-5
SUM_MNU_OPH_EV = 0.09001192964464505
OMEGA_NU_H2_OPH = SUM_MNU_OPH_EV / 93.12
Q_A_TARGET = 5.363470441
H0_PLANCK_BRANCH = 67.4
SIGMA8_CDM_BRANCH = 0.807787208
S8_CDM_BRANCH = 0.828924043
WEAK_LENSING_S8_TARGET = 0.790
GAMMA_REC_MATRIX_GAP = 0.050717471


def h0s8_branch_report(
    *,
    q_a: float = Q_A_TARGET,
    h_ds: float = H_DS_KM_S_MPC,
    omega_b_h2: float = OMEGA_B_H2,
    omega_nu_h2: float = OMEGA_NU_H2_OPH,
    omega_r_h2: float = OMEGA_R_H2,
    s8_cdm: float = S8_CDM_BRANCH,
    sigma8_cdm: float = SIGMA8_CDM_BRANCH,
    p_value: float = P_STAR,
) -> dict[str, Any]:
    """Compute the H0/S8 diagnostic branches from the H0/S8 notes.

    This is intentionally a branch calculator, not a proof that the finite
    collar state has emitted Q_A, B_A, or the Jacobi clock.
    """

    h0_blind = h0_from_flat_q_a(
        q_a=float(q_a),
        h_ds=float(h_ds),
        omega_b_h2=float(omega_b_h2),
        omega_nu_h2=float(omega_nu_h2),
        omega_r_h2=float(omega_r_h2),
    )
    h = h0_blind / 100.0
    omega_b = float(omega_b_h2) / (h * h)
    omega_nu = float(omega_nu_h2) / (h * h)
    omega_r = float(omega_r_h2) / (h * h)
    omega_a = float(q_a) * omega_b
    omega_m = omega_b + omega_nu + omega_a
    omega_lambda = (float(h_ds) / h0_blind) ** 2
    constants = OPHConstants(P=float(p_value))
    lambda_collar = constants.lambda_collar_exact_uniform_product_thickening
    f_a = omega_a / omega_m if omega_m > 0.0 else 0.0
    mu_eff = (1.0 - f_a) + f_a * lambda_collar
    required_growth_factor = float(WEAK_LENSING_S8_TARGET) / float(s8_cdm)
    jacobi_s8 = float(s8_cdm) * required_growth_factor
    matrix_gap_suppression = 1.0 - (1.0 - mu_eff) * float(GAMMA_REC_MATRIX_GAP)
    matrix_gap_s8 = float(s8_cdm) * matrix_gap_suppression
    lane8_certificate = h0s8_lane8_certificate_report()
    report = {
        "mode": "oph_h0_s8_branch_diagnostic_v0",
        "inputs": {
            "P": float(p_value),
            "H_dS_km_s_Mpc": float(h_ds),
            "q_A": float(q_a),
            "omega_b_h2": float(omega_b_h2),
            "omega_nu_h2": float(omega_nu_h2),
            "omega_r_h2": float(omega_r_h2),
            "sum_mnu_OPH_eV": SUM_MNU_OPH_EV,
            "sigma8_cdm_branch": float(sigma8_cdm),
            "S8_cdm_branch": float(s8_cdm),
        },
        "flat_q_a_closure": {
            "H0_km_s_Mpc": h0_blind,
            "h": h,
            "Omega_Lambda_OPH": omega_lambda,
            "Omega_b": omega_b,
            "Omega_nu": omega_nu,
            "Omega_r": omega_r,
            "Omega_A": omega_a,
            "Omega_m": omega_m,
            "flat_sum": omega_lambda + omega_b + omega_nu + omega_r + omega_a,
        },
        "collar_tracking": {
            "lambda_collar": lambda_collar,
            "lambda_collar_exact_uniform_product_thickening": lambda_collar,
            "lambda_collar_status": constants.lambda_collar_claim_status,
            "lambda_collar_exact_gate": constants.lambda_collar_exact_gate,
            "lambda_collar_exact_gate_pass": False,
            "finite_thickness_profile_default": constants.lambda_collar_profile_default,
            "finite_thickness_jensen_band": constants.finite_thickness_jensen_band,
            "z6_normalized_trace_mean": constants.z6_normalized_trace_mean,
            "z6_reciprocal_trace": constants.z6_reciprocal_trace,
            "B_A_track": lambda_collar,
            "f_A": f_a,
            "mu_eff_source_suppression": mu_eff,
            "source_suppression_fraction": 1.0 - mu_eff,
        },
        "branches": {
            "A_conserved_cdm_like": {
                "H0_km_s_Mpc": H0_PLANCK_BRANCH,
                "S8": float(s8_cdm),
                "sigma8": float(sigma8_cdm),
                "status": "current_executable_planck_like_branch",
            },
            "B_direct_jacobi_repair": {
                "H0_km_s_Mpc": h0_blind,
                "S8": jacobi_s8,
                "growth_suppression_factor": required_growth_factor,
                "status": "conditional_if_Gamma_rec_equals_Gamma_J_and_Q_A_B_A_are_collar_outputs",
            },
            "C_matrix_gapped_jacobi": {
                "H0_km_s_Mpc": h0_blind,
                "S8": matrix_gap_s8,
                "growth_suppression_factor": matrix_gap_suppression,
                "gamma_rec_matrix_gap": GAMMA_REC_MATRIX_GAP,
                "status": "diagnostic_slow_repair_clock_alternative",
            },
        },
        "measurement_comparisons": {
            "Planck2018_H0": {"value": 67.36, "sigma": 0.54, "branch_pull_sigma": (h0_blind - 67.36) / 0.54},
            "SH0ES_H0": {"value": 73.04, "sigma": 1.04, "branch_pull_sigma": (h0_blind - 73.04) / 1.04},
            "Planck2018_S8": {"value": 0.832, "sigma": 0.013, "cdm_pull_sigma": (float(s8_cdm) - 0.832) / 0.013},
            "weak_lensing_S8_target": {
                "value": WEAK_LENSING_S8_TARGET,
                "sigma_proxy": 0.016,
                "direct_jacobi_pull_sigma": (jacobi_s8 - WEAK_LENSING_S8_TARGET) / 0.016,
                "cdm_pull_sigma": (float(s8_cdm) - WEAK_LENSING_S8_TARGET) / 0.016,
            },
        },
        "theorem_gates": {
            "Q_A_from_finite_collar_selector": False,
            "B_A_from_parent_collar_kernel": False,
            "LOCAL_POISSON_RESERVE_SURVIVAL": True,
            "SCALAR_WEIGHTED_Z6_MEAN": False,
            "UNIFORM_PRODUCT_THICKENING_EXACT": False,
            "lambda_collar_from_P_survival": False,
            "Gamma_rec_equals_Jacobi_clock": False,
            "full_CAMB_CLASS_anomaly_module": False,
            "full_likelihood_contract": False,
            "lane8_low_entropy_certificate_ready": bool(
                lane8_certificate["theorem_gates"]["low_entropy_ancestry_certificate_ready"]
            ),
        },
        "lane8_certificate_stack": lane8_certificate,
        "physical_prediction_ready": False,
        "physical_cmb_prediction": False,
        "physical_matter_power_prediction": False,
        "claim_boundary": (
            "H0/S8 branch diagnostic from OPH cosmology notes. It computes consequences of declared "
            "branch assumptions but does not prove that finite lattice runs derive Q_A, B_A(k,a), or "
            "Gamma_rec=Gamma_J. The exp(-P/24) collar value is reported as an exact-uniform "
            "product-thickening diagnostic target only; finite-thickness/local-coefficient promotion "
            "requires the UNIFORM_PRODUCT_THICKENING_EXACT gate. Treat as measurement-facing continuation "
            "data until those gates close."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=CONTINUATION,
        receipt=COSMOLOGY_PERTURBATION_RECEIPT,
        physical_claim=False,
        observable_id="oph_h0_s8_branch_diagnostic",
        fit_objective="h0_s8_branch_consequence_audit",
    )


def write_h0s8_branch_report(out_dir: Path, **kwargs: Any) -> dict[str, Any]:
    report = h0s8_branch_report(**kwargs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "h0s8_branch_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "h0s8_branch_report.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_csv(out / "h0s8_branch_rows.csv", _branch_rows(report))
    return report


def h0_from_flat_q_a(
    *,
    q_a: float,
    h_ds: float = H_DS_KM_S_MPC,
    omega_b_h2: float = OMEGA_B_H2,
    omega_nu_h2: float = OMEGA_NU_H2_OPH,
    omega_r_h2: float = OMEGA_R_H2,
) -> float:
    h2 = (float(h_ds) / 100.0) ** 2 + (1.0 + float(q_a)) * float(omega_b_h2) + float(omega_nu_h2) + float(omega_r_h2)
    return 100.0 * math.sqrt(h2)


def _branch_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, branch in (report.get("branches", {}) or {}).items():
        rows.append({"branch": name, **branch})
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


def _markdown_report(report: dict[str, Any]) -> str:
    flat = report["flat_q_a_closure"]
    branches = report["branches"]
    return "\n".join(
        [
            "# OPH H0/S8 Branch Diagnostic",
            "",
            f"- Flat q_A closure H0: `{flat['H0_km_s_Mpc']:.6f}` km/s/Mpc",
            f"- Omega_m: `{flat['Omega_m']:.9f}`",
            f"- Collar lambda exact-uniform target / B_A_track: `{report['collar_tracking']['lambda_collar']:.12f}`",
            f"- Effective source suppression: `{report['collar_tracking']['source_suppression_fraction']:.6f}`",
            "",
            "## Branches",
            "",
            f"- Conserved CDM-like: `H0={branches['A_conserved_cdm_like']['H0_km_s_Mpc']}`, `S8={branches['A_conserved_cdm_like']['S8']}`",
            f"- Direct Jacobi repair: `H0={branches['B_direct_jacobi_repair']['H0_km_s_Mpc']:.6f}`, `S8={branches['B_direct_jacobi_repair']['S8']:.6f}`",
            f"- Matrix-gapped Jacobi: `H0={branches['C_matrix_gapped_jacobi']['H0_km_s_Mpc']:.6f}`, `S8={branches['C_matrix_gapped_jacobi']['S8']:.6f}`",
            "",
            "## Missing Gates",
            "",
            *[f"- {name}: `{str(value).lower()}`" for name, value in report["theorem_gates"].items()],
            "",
            "## Claim Boundary",
            "",
            str(report.get("claim_boundary", "")),
            "",
        ]
    )
