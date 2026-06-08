"""Cosmology helpers with lazy public exports.

The package intentionally avoids eager submodule imports. Several OPH-FPE
handoff bundles include only the modules needed for one diagnostic lane; eager
imports made unrelated missing modules break otherwise-valid reproduction
commands such as importing the CAMB baseline adapter.
"""

from __future__ import annotations

from typing import Any

_EXPORTS = {
    "anomaly_background_rhs": ("oph_fpe.cosmology.anomaly_fluid", "anomaly_background_rhs"),
    "anomaly_perturbation_rhs": ("oph_fpe.cosmology.anomaly_fluid", "anomaly_perturbation_rhs"),
    "exchange_closure_report": ("oph_fpe.cosmology.anomaly_fluid", "exchange_closure_report"),
    "angular_power_report": ("oph_fpe.cosmology.angular_power", "angular_power_report"),
    "background_adapter_status": ("oph_fpe.cosmology.background_adapter", "background_adapter_status"),
    "estimate_b_a_grid": ("oph_fpe.cosmology.ba_parent", "estimate_b_a_grid"),
    "oph_boltzmann_input_report": ("oph_fpe.cosmology.boltzmann_inputs", "oph_boltzmann_input_report"),
    "write_oph_boltzmann_input_report": ("oph_fpe.cosmology.boltzmann_inputs", "write_oph_boltzmann_input_report"),
    "LambdaCDMParameters": ("oph_fpe.cosmology.camb_adapter", "LambdaCDMParameters"),
    "camb_lcdm_baseline_report": ("oph_fpe.cosmology.camb_adapter", "camb_lcdm_baseline_report"),
    "compare_camb_tt_to_benchmark": ("oph_fpe.cosmology.camb_adapter", "compare_camb_tt_to_benchmark"),
    "write_camb_lcdm_baseline_report": ("oph_fpe.cosmology.camb_adapter", "write_camb_lcdm_baseline_report"),
    "oph_screen_camb_report": ("oph_fpe.cosmology.camb_adapter", "oph_screen_camb_report"),
    "write_oph_screen_camb_report": ("oph_fpe.cosmology.camb_adapter", "write_oph_screen_camb_report"),
    "oph_inflation_cmb_camb_report": (
        "oph_fpe.cosmology.camb_adapter",
        "oph_inflation_cmb_camb_report",
    ),
    "write_oph_inflation_cmb_camb_report": (
        "oph_fpe.cosmology.camb_adapter",
        "write_oph_inflation_cmb_camb_report",
    ),
    "collect_cl_runs": ("oph_fpe.cosmology.cl_ensemble", "collect_cl_runs"),
    "cl_ensemble_report": ("oph_fpe.cosmology.cl_ensemble", "cl_ensemble_report"),
    "write_cl_ensemble_report": ("oph_fpe.cosmology.cl_ensemble", "write_cl_ensemble_report"),
    "cmb_lite_comparison_report": ("oph_fpe.cosmology.cmb_compare", "cmb_lite_comparison_report"),
    "load_planck_tt_binned": ("oph_fpe.cosmology.cmb_compare", "load_planck_tt_binned"),
    "write_cmb_lite_comparison": ("oph_fpe.cosmology.cmb_compare", "write_cmb_lite_comparison"),
    "cmb_anomaly_report": ("oph_fpe.cosmology.cmb_anomaly", "cmb_anomaly_report"),
    "write_cmb_anomaly_report": ("oph_fpe.cosmology.cmb_anomaly", "write_cmb_anomaly_report"),
    "cmb_parameter_derivation_report": (
        "oph_fpe.cosmology.cmb_derivation",
        "cmb_parameter_derivation_report",
    ),
    "write_cmb_parameter_derivation_report": (
        "oph_fpe.cosmology.cmb_derivation",
        "write_cmb_parameter_derivation_report",
    ),
    "compressed_likelihood_reference_report": (
        "oph_fpe.cosmology.compressed_likelihood",
        "compressed_likelihood_reference_report",
    ),
    "write_compressed_likelihood_reference_report": (
        "oph_fpe.cosmology.compressed_likelihood",
        "write_compressed_likelihood_reference_report",
    ),
    "screen_spectrum_prediction": (
        "oph_fpe.cosmology.inflation_cmb_ladder",
        "screen_spectrum_prediction",
    ),
    "flat_sector_selection_report": (
        "oph_fpe.cosmology.inflation_cmb_ladder",
        "flat_sector_selection_report",
    ),
    "cmb_success_ladder_report": (
        "oph_fpe.cosmology.inflation_cmb_ladder",
        "cmb_success_ladder_report",
    ),
    "inflation_cmb_bridge_report": (
        "oph_fpe.cosmology.inflation_cmb_ladder",
        "inflation_cmb_bridge_report",
    ),
    "write_inflation_cmb_bridge_report": (
        "oph_fpe.cosmology.inflation_cmb_ladder",
        "write_inflation_cmb_bridge_report",
    ),
    "collect_comparable_runs": ("oph_fpe.cosmology.comparable_data", "collect_comparable_runs"),
    "comparable_data_report": ("oph_fpe.cosmology.comparable_data", "comparable_data_report"),
    "write_comparable_data_package": ("oph_fpe.cosmology.comparable_data", "write_comparable_data_package"),
    "measurement_target": ("oph_fpe.cosmology.data_targets", "measurement_target"),
    "target_registry": ("oph_fpe.cosmology.data_targets", "target_registry"),
    "write_freezeout_products": ("oph_fpe.cosmology.freezeout", "write_freezeout_products"),
    "galaxy_proxy_receipt": ("oph_fpe.cosmology.galaxy_proxy", "galaxy_proxy_receipt"),
    "adiabaticity_report": ("oph_fpe.cosmology.adiabaticity", "adiabaticity_report"),
    "write_adiabaticity_report": ("oph_fpe.cosmology.adiabaticity", "write_adiabaticity_report"),
    "load_static_galaxy_dataset": ("oph_fpe.cosmology.galaxy_static", "load_static_galaxy_dataset"),
    "btfr_prediction_from_rar_fit": (
        "oph_fpe.cosmology.galaxy_static",
        "btfr_prediction_from_rar_fit",
    ),
    "nu_oph": ("oph_fpe.cosmology.galaxy_proxy", "nu_oph"),
    "rar_curve": ("oph_fpe.cosmology.galaxy_proxy", "rar_curve"),
    "static_galaxy_measurement_report": ("oph_fpe.cosmology.galaxy_static", "static_galaxy_measurement_report"),
    "static_galaxy_holdout_report": ("oph_fpe.cosmology.galaxy_static", "static_galaxy_holdout_report"),
    "write_static_galaxy_measurement_report": (
        "oph_fpe.cosmology.galaxy_static",
        "write_static_galaxy_measurement_report",
    ),
    "h0s8_branch_report": ("oph_fpe.cosmology.h0s8", "h0s8_branch_report"),
    "write_h0s8_branch_report": ("oph_fpe.cosmology.h0s8", "write_h0s8_branch_report"),
    "hot_release_report": ("oph_fpe.cosmology.hot_release", "hot_release_report"),
    "write_hot_release_report": ("oph_fpe.cosmology.hot_release", "write_hot_release_report"),
    "oph_cmb_stress_adapter_report": ("oph_fpe.cosmology.oph_cmb_adapter", "oph_cmb_stress_adapter_report"),
    "C_ell_oph": ("oph_fpe.cosmology.oph_screen_power", "C_ell_oph"),
    "D_ell_from_C_ell": ("oph_fpe.cosmology.oph_screen_power", "D_ell_from_C_ell"),
    "F_oph_k": ("oph_fpe.cosmology.oph_screen_power", "F_oph_k"),
    "OPHScreenPowerParams": ("oph_fpe.cosmology.oph_screen_power", "OPHScreenPowerParams"),
    "primordial_power_oph": ("oph_fpe.cosmology.oph_screen_power", "primordial_power_oph"),
    "write_oph_screen_power_report": ("oph_fpe.cosmology.oph_screen_power", "write_oph_screen_power_report"),
    "cosmo_proxy_receipt": ("oph_fpe.cosmology.proxy_pipeline", "cosmo_proxy_receipt"),
    "synchronization_inflation_report": (
        "oph_fpe.cosmology.sync_inflation",
        "synchronization_inflation_report",
    ),
    "write_synchronization_inflation_report": (
        "oph_fpe.cosmology.sync_inflation",
        "write_synchronization_inflation_report",
    ),
    "synchronization_gap_report": (
        "oph_fpe.cosmology.sync_gap",
        "synchronization_gap_report",
    ),
    "write_synchronization_gap_report": (
        "oph_fpe.cosmology.sync_gap",
        "write_synchronization_gap_report",
    ),
    "unique_prediction_gate_report": (
        "oph_fpe.cosmology.unique_predictions",
        "unique_prediction_gate_report",
    ),
    "write_unique_prediction_gate_report": (
        "oph_fpe.cosmology.unique_predictions",
        "write_unique_prediction_gate_report",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    from importlib import import_module

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
