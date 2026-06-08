from __future__ import annotations

import math
from typing import Any

import numpy as np

from oph_fpe.claims import COSMOLOGY_PERTURBATION_RECEIPT, PROXY, with_claim_metadata


DEFAULT_A_GRID = [1.0 / 1100.0, 0.01, 0.1, 1.0]
DEFAULT_K_GRID_H_MPC = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
DEFAULT_OUTPUT_OBSERVABLES = [
    "C_ell_TT",
    "C_ell_TE",
    "C_ell_EE",
    "C_ell_BB",
    "C_ell_phiphi",
    "P_k",
    "sigma_8",
    "S_8",
    "f_sigma_8",
    "r_s",
    "D_A",
    "H_z",
]


def oph_cmb_stress_adapter_report(
    *,
    collar_report: dict[str, Any],
    cosmology_gate_report: dict[str, Any] | None = None,
    freezeout_report: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit the OPH-CMB anomaly-stress bridge described in the CMB writeup.

    This is deliberately not a Boltzmann solver. It separates the standard
    photon-baryon/recombination baseline from the OPH anomaly stress sector,
    then records which parent quantities are available from finite-collar
    diagnostics and which are still missing for a physical CMB prediction.
    """

    cfg = dict(config or {})
    finite_parent = _finite_collar_parent(collar_report, cfg)
    readiness = _readiness(cosmology_gate_report or {}, finite_parent)
    report = {
        "mode": "oph_cmb_anomaly_stress_adapter_v0",
        "enabled": bool(cfg.get("enabled", True)),
        "standard_photon_baryon_baseline": _standard_baseline(cfg),
        "anomaly_stress_model": _anomaly_stress_model(cfg),
        "finite_collar_parent": finite_parent,
        "diagnostic_kernel_proxy": _diagnostic_kernel_proxy(finite_parent, cfg),
        "boltzmann_adapter_requirements": _boltzmann_requirements(),
        "observer_facing_outputs_required": list(DEFAULT_OUTPUT_OBSERVABLES),
        "screen_proxy_context": {
            "freezeout_screen_cl_available": bool(freezeout_report),
            "screen_proxy_gate_allowed": bool((cosmology_gate_report or {}).get("allowed", False)),
            "physical_cmb_prediction": False,
        },
        "physical_prediction_readiness": readiness,
        "COSMOLOGY_PERTURBATION_RECEIPT": False,
        "physical_cmb_prediction": False,
        "physical_matter_power_prediction": False,
        "claim_boundary": (
            "OPH-CMB anomaly-stress adapter scaffold derived from the CMB writeup. "
            "Standard photons/recombination are delegated to CAMB/CLASS-style physics. "
            "Finite-collar I(A:D|B) values are diagnostic parent samples only; they do not yet "
            "supply theorem-grade rho_A(a), Gamma_rec, or B_A(k,a). This is not a physical "
            "CMB prediction, not a likelihood, and not a Boltzmann run."
        ),
    }
    return with_claim_metadata(
        report,
        claim_level=PROXY,
        receipt="OPH_CMB_ANOMALY_STRESS_ADAPTER",
        physical_claim=False,
        observable_id="oph_cmb_anomaly_stress_bridge",
        fit_objective="finite_collar_parent_to_boltzmann_input_scaffold",
    )


def _standard_baseline(cfg: dict[str, Any]) -> dict[str, Any]:
    baseline_cfg = dict(cfg.get("standard_baseline", {}) or {})
    return {
        "status": "external_boltzmann_baseline_required",
        "photon_generation": "standard thermal photon-baryon plasma; photons redshift into microwaves",
        "keep_standard_physics": [
            "photons",
            "baryons",
            "electrons",
            "neutrinos",
            "Thomson_scattering",
            "recombination_visibility_function",
            "line_of_sight_integration",
        ],
        "visibility_function": "g(eta) = -tau'(eta) exp[-tau(eta)]",
        "cmb_temperature_today_K": baseline_cfg.get("T0_K", 2.725),
        "recombination_redshift_slot": baseline_cfg.get("z_star", 1100),
        "regression": (
            "Before OPH anomaly extensions are trusted, CAMB/CLASS must recover standard "
            "LambdaCDM TT/TE/EE/lensing spectra in the CDM limit."
        ),
    }


def _anomaly_stress_model(cfg: dict[str, Any]) -> dict[str, Any]:
    branch = str(cfg.get("branch", "repair_exchange_diagnostic"))
    return {
        "selected_branch": branch,
        "conserved_cdm_limit": {
            "rho_A": "rho_A0 * a^-3",
            "w_A": 0.0,
            "c_s_A_squared": 0.0,
            "sigma_A": 0.0,
            "Gamma_rec": 0.0,
            "status": "formula_emitted_regression_not_run",
        },
        "repair_exchange_branch": {
            "background": "rho_A' + 3 H rho_A = -a Gamma_rec (rho_A - rho_A_eq)",
            "density_perturbation": (
                "delta_A' = -theta_A + 3 Phi' - a Gamma_rec q_A "
                "(delta_A - B_A(k,a) delta_b)"
            ),
            "velocity_perturbation": "theta_A' = -H theta_A + k^2 Psi",
            "required_parent_quantities": ["rho_A(a)", "rho_A_eq(a)", "Gamma_rec(k,a)", "B_A(k,a)"],
            "status": "schema_emitted_parent_quantities_not_theorem_grade",
        },
        "static_galaxy_law_boundary": (
            "The settled static RAR law is not used as a homogeneous FLRW source and is not "
            "Taylor-expanded at g_b=0."
        ),
    }


def _finite_collar_parent(collar_report: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    rows = list(collar_report.get("rows", []) or [])
    weights = np.asarray([max(float(row.get("triplet_count", 1.0)), 1.0) for row in rows], dtype=float)
    eps = np.asarray([max(float(row.get("epsilon_cmi", 0.0)), 0.0) for row in rows], dtype=float)
    total_weight = float(np.sum(weights))
    r_collar = float(np.sum(weights * eps) / total_weight) if rows and total_weight > 0.0 else None
    scale = cfg.get("parent_length_scale_planck")
    density_prefactor = 15.0 / (8.0 * math.pi * math.pi)
    if scale is not None and float(scale) > 0.0 and r_collar is not None:
        rho_eq_proxy = float(density_prefactor * r_collar / (float(scale) ** 4))
        rho_units = "planck_energy_density_proxy"
    else:
        rho_eq_proxy = None
        rho_units = "not_available_without_parent_length_scale"

    sample_rows = []
    for row, weight, epsilon in zip(rows, weights, eps, strict=False):
        theta0 = float(row.get("theta0", 0.0))
        sample_rows.append(
            {
                "cap_id": int(row.get("cap_id", len(sample_rows))),
                "theta0": theta0,
                "k_proxy_inverse_theta": float(1.0 / theta0) if theta0 > 0.0 else None,
                "epsilon_cmi": float(epsilon),
                "weight": float(weight),
                "r_fr_bound": row.get("r_fr_bound"),
                "packet_alphabet_size": row.get("packet_alphabet_size"),
            }
        )
    return {
        "source": "collar_markov_report",
        "parent_formula": "rho_A_eq[X] c^2 = 15/(8*pi^2*l(X)^4) * int_C dmu_C I_omega_C(A:D|B)",
        "status": "diagnostic_finite_collar_samples_not_theorem_grade",
        "sample_count": len(rows),
        "weighted_collar_repair_defect_R": r_collar,
        "median_epsilon_cmi": collar_report.get("median_epsilon_cmi"),
        "mean_epsilon_cmi": collar_report.get("mean_epsilon_cmi"),
        "p90_epsilon_cmi": collar_report.get("p90_epsilon_cmi"),
        "rho_A_eq_proxy": rho_eq_proxy,
        "rho_A_eq_proxy_units": rho_units,
        "density_prefactor_15_over_8pi2": float(density_prefactor),
        "sample_rows": sample_rows,
        "claim_boundary": (
            "Uses diagonal empirical collar CMI samples as a parent diagnostic. It is not a "
            "theorem-grade finite-collar evaluator and must not be fitted to CMB likelihoods."
        ),
    }


def _diagnostic_kernel_proxy(parent: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    sample_rows = parent.get("sample_rows", []) or []
    eps_values = np.asarray([float(row.get("epsilon_cmi", 0.0)) for row in sample_rows], dtype=float)
    centered_scale = float(np.std(eps_values)) if eps_values.size else 0.0
    rows = []
    for row in sample_rows:
        epsilon = float(row.get("epsilon_cmi", 0.0))
        if centered_scale > 1e-12:
            b_proxy = float((epsilon - float(np.mean(eps_values))) / centered_scale)
        else:
            b_proxy = 0.0
        rows.append(
            {
                "k_proxy_inverse_theta": row.get("k_proxy_inverse_theta"),
                "theta0": row.get("theta0"),
                "B_A_shape_proxy": b_proxy,
            }
        )
    return {
        "status": "screen_cap_size_shape_proxy_only",
        "a_grid": [float(value) for value in cfg.get("a_grid", DEFAULT_A_GRID)],
        "k_grid_h_mpc_required_for_boltzmann": [float(value) for value in cfg.get("k_grid_h_mpc", DEFAULT_K_GRID_H_MPC)],
        "kernel_proxy_rows": rows,
        "B_A_k_a_emitted": False,
        "Gamma_rec_emitted": False,
        "claim_boundary": (
            "Cap-size variation of collar CMI is a diagnostic shape proxy, not a gauge-consistent "
            "B_A(k,a) kernel."
        ),
    }


def _boltzmann_requirements() -> dict[str, Any]:
    return {
        "baseline_solver": "CAMB_or_CLASS",
        "must_pass_first": [
            "LambdaCDM_CDM_limit_regression",
            "gauge_consistency_of_delta_A_theta_A_equations",
            "energy_momentum_exchange_sign_audit",
            "script_and_input_hashing",
            "full_likelihood_covariances",
        ],
        "required_anomaly_inputs": [
            "rho_A(a)",
            "rho_A_eq(a)",
            "Gamma_rec(k,a)",
            "B_A(k,a)",
            "initial_conditions_for_delta_A_theta_A",
            "compensating_sector_for_Q_A_mu_if_exchange_enabled",
        ],
        "first_physical_outputs": list(DEFAULT_OUTPUT_OBSERVABLES),
    }


def _readiness(cosmology_gate_report: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "screen_proxy_gate_allowed": bool(cosmology_gate_report.get("allowed", False)),
        "finite_collar_samples_available": int(parent.get("sample_count") or 0) > 0,
        "finite_collar_parent_theorem_grade": False,
        "rho_A_eq_scalar_proxy_available": parent.get("rho_A_eq_proxy") is not None,
        "rho_A_of_a_emitted": False,
        "rho_A_eq_of_a_emitted": False,
        "Gamma_rec_of_k_a_emitted": False,
        "B_A_of_k_a_emitted": False,
        "energy_momentum_exchange_closed": False,
        # Legacy aliases retained as false so old readers do not confuse a scalar proxy
        # with a physical function of scale factor or wavenumber.
        "rho_A_a_emitted": False,
        "rho_A_eq_a_emitted": False,
        "Gamma_rec_emitted": False,
        "B_A_k_a_emitted": False,
        "cdm_limit_regression_run": False,
        "gauge_consistency_audited": False,
        "camb_or_class_run": False,
        "full_likelihood_run": False,
    }
    ready = all(checks.values())
    missing = [name for name, passed in checks.items() if not passed]
    return {
        "boltzmann_ready": False,
        "physical_cmb_prediction_ready": bool(ready),
        "checks": checks,
        "missing_gates": missing,
    }
