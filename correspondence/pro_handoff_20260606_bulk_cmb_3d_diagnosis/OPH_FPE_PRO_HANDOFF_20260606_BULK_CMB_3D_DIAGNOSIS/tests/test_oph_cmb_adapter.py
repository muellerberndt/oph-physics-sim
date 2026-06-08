from __future__ import annotations

import pytest

from oph_fpe.cosmology.oph_cmb_adapter import oph_cmb_stress_adapter_report


def test_oph_cmb_adapter_emits_writeup_layers_and_keeps_gate_closed():
    collar = {
        "median_epsilon_cmi": 0.2,
        "mean_epsilon_cmi": 0.3,
        "p90_epsilon_cmi": 0.5,
        "rows": [
            {"cap_id": 0, "theta0": 0.5, "epsilon_cmi": 0.2, "triplet_count": 10, "r_fr_bound": 0.9},
            {"cap_id": 1, "theta0": 1.0, "epsilon_cmi": 0.4, "triplet_count": 30, "r_fr_bound": 1.2},
        ],
    }

    report = oph_cmb_stress_adapter_report(
        collar_report=collar,
        cosmology_gate_report={"allowed": True},
        freezeout_report={"fields": {}},
        config={"parent_length_scale_planck": 2.0},
    )

    assert report["mode"] == "oph_cmb_anomaly_stress_adapter_v0"
    assert report["standard_photon_baryon_baseline"]["status"] == "external_boltzmann_baseline_required"
    assert report["anomaly_stress_model"]["conserved_cdm_limit"]["rho_A"] == "rho_A0 * a^-3"
    assert report["finite_collar_parent"]["weighted_collar_repair_defect_R"] == pytest.approx(0.35)
    assert report["finite_collar_parent"]["rho_A_eq_proxy"] is not None
    assert report["diagnostic_kernel_proxy"]["B_A_k_a_emitted"] is False
    assert report["physical_prediction_readiness"]["boltzmann_ready"] is False
    checks = report["physical_prediction_readiness"]["checks"]
    assert checks["rho_A_eq_scalar_proxy_available"] is True
    assert checks["rho_A_eq_of_a_emitted"] is False
    assert checks["rho_A_eq_a_emitted"] is False
    assert checks["B_A_of_k_a_emitted"] is False
    assert checks["Gamma_rec_of_k_a_emitted"] is False
    assert checks["energy_momentum_exchange_closed"] is False
    assert report["COSMOLOGY_PERTURBATION_RECEIPT"] is False
    assert report["physical_cmb_prediction"] is False


def test_oph_cmb_adapter_handles_missing_parent_scale_as_diagnostic_only():
    report = oph_cmb_stress_adapter_report(
        collar_report={"rows": [{"theta0": 0.5, "epsilon_cmi": 1.0}]},
        cosmology_gate_report={"allowed": True},
        config={},
    )

    assert report["finite_collar_parent"]["rho_A_eq_proxy"] is None
    assert "rho_A_of_a_emitted" in report["physical_prediction_readiness"]["missing_gates"]
    assert "rho_A_eq_of_a_emitted" in report["physical_prediction_readiness"]["missing_gates"]
