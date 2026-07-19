"""Cosmology helpers with lazy public exports.

The package intentionally avoids eager submodule imports. Several OPH-FPE
handoff bundles include only the modules needed for one diagnostic lane; eager
imports made unrelated missing modules break otherwise-valid reproduction
commands such as importing the CAMB baseline adapter.
"""

from __future__ import annotations

from typing import Any

_EXPORTS = {
    "particle_frontier_report": ("oph_fpe.cosmology.particle_frontier", "particle_frontier_report"),
    "write_particle_frontier_report": (
        "oph_fpe.cosmology.particle_frontier",
        "write_particle_frontier_report",
    ),
    "anomaly_background_rhs": ("oph_fpe.cosmology.anomaly_fluid", "anomaly_background_rhs"),
    "anomaly_perturbation_rhs": ("oph_fpe.cosmology.anomaly_fluid", "anomaly_perturbation_rhs"),
    "covariant_exchange_current": ("oph_fpe.cosmology.anomaly_fluid", "covariant_exchange_current"),
    "covariant_exchange_closure_report": (
        "oph_fpe.cosmology.anomaly_fluid",
        "covariant_exchange_closure_report",
    ),
    "exchange_closure_report": ("oph_fpe.cosmology.anomaly_fluid", "exchange_closure_report"),
    "AnomalyReleaseStateArtifact": (
        "oph_fpe.cosmology.anomaly_abundance_selector",
        "AnomalyReleaseStateArtifact",
    ),
    "AnomalyLoadObservableArtifact": (
        "oph_fpe.cosmology.anomaly_abundance_selector",
        "AnomalyLoadObservableArtifact",
    ),
    "SourceMaxEntReleaseLawArtifact": (
        "oph_fpe.cosmology.anomaly_abundance_selector",
        "SourceMaxEntReleaseLawArtifact",
    ),
    "AnomalyAbundanceSelectorArtifact": (
        "oph_fpe.cosmology.anomaly_abundance_selector",
        "AnomalyAbundanceSelectorArtifact",
    ),
    "LoadRefinementCompatibilityArtifact": (
        "oph_fpe.cosmology.anomaly_abundance_selector",
        "LoadRefinementCompatibilityArtifact",
    ),
    "verify_anomaly_abundance_source_receipt": (
        "oph_fpe.cosmology.anomaly_abundance_selector",
        "verify_anomaly_abundance_source_receipt",
    ),
    "verify_anomaly_release_state": (
        "oph_fpe.cosmology.anomaly_abundance_selector",
        "verify_anomaly_release_state",
    ),
    "verify_anomaly_load_observable": (
        "oph_fpe.cosmology.anomaly_abundance_selector",
        "verify_anomaly_load_observable",
    ),
    "verify_source_maxent_release_law": (
        "oph_fpe.cosmology.anomaly_abundance_selector",
        "verify_source_maxent_release_law",
    ),
    "verify_load_refinement_compatibility": (
        "oph_fpe.cosmology.anomaly_abundance_selector",
        "verify_load_refinement_compatibility",
    ),
    "compute_anomaly_load": (
        "oph_fpe.cosmology.anomaly_abundance_selector",
        "compute_anomaly_load",
    ),
    "compute_selector": ("oph_fpe.cosmology.anomaly_abundance_selector", "compute_selector"),
    "angular_power_report": ("oph_fpe.cosmology.angular_power", "angular_power_report"),
    "background_adapter_status": ("oph_fpe.cosmology.background_adapter", "background_adapter_status"),
    "estimate_b_a_grid": ("oph_fpe.cosmology.ba_parent", "estimate_b_a_grid"),
    "BAKernelRow": ("oph_fpe.cosmology.ba_kernel", "BAKernelRow"),
    "B_A_kernel_receipt": ("oph_fpe.cosmology.ba_kernel", "B_A_kernel_receipt"),
    "ba_kernel_report_from_paired_csv": (
        "oph_fpe.cosmology.ba_kernel",
        "ba_kernel_report_from_paired_csv",
    ),
    "estimate_B_A_from_paired_runs": (
        "oph_fpe.cosmology.ba_kernel",
        "estimate_B_A_from_paired_runs",
    ),
    "finite_certificate_bundle": ("oph_fpe.cosmology.finite_certificates", "finite_certificate_bundle"),
    "write_finite_certificate_bundle": (
        "oph_fpe.cosmology.finite_certificates",
        "write_finite_certificate_bundle",
    ),
    "run_proxy_certificate_input": (
        "oph_fpe.cosmology.finite_certificates",
        "run_proxy_certificate_input",
    ),
    "write_run_proxy_finite_certificate_bundle": (
        "oph_fpe.cosmology.finite_certificates",
        "write_run_proxy_finite_certificate_bundle",
    ),
    "parent_collar_ladder_report": (
        "oph_fpe.cosmology.parent_collar_ladder",
        "parent_collar_ladder_report",
    ),
    "write_parent_collar_ladder_report": (
        "oph_fpe.cosmology.parent_collar_ladder",
        "write_parent_collar_ladder_report",
    ),
    "oph_boltzmann_input_report": ("oph_fpe.cosmology.boltzmann_inputs", "oph_boltzmann_input_report"),
    "write_oph_boltzmann_input_report": ("oph_fpe.cosmology.boltzmann_inputs", "write_oph_boltzmann_input_report"),
    "PhysicalCMBInputContract": (
        "oph_fpe.cosmology.physical_cmb_contract",
        "PhysicalCMBInputContract",
    ),
    "ClaimTier": ("oph_fpe.cosmology.claim_tiers", "ClaimTier"),
    "GeometryOrigin": ("oph_fpe.cosmology.claim_tiers", "GeometryOrigin"),
    "validate_physical_scale_bridge_receipts": (
        "oph_fpe.cosmology.cosmological_scale_bridge",
        "validate_physical_scale_bridge_receipts",
    ),
    "imported_flrw_reference_receipts": (
        "oph_fpe.cosmology.cosmological_scale_bridge",
        "imported_flrw_reference_receipts",
    ),
    "contract_from_reports": (
        "oph_fpe.cosmology.physical_cmb_contract",
        "contract_from_reports",
    ),
    "validate_physical_cmb_contract": (
        "oph_fpe.cosmology.physical_cmb_contract",
        "validate_physical_cmb_contract",
    ),
    "build_physical_cmb_input_contract": (
        "oph_fpe.cosmology.physical_cmb_prediction",
        "build_physical_cmb_input_contract",
    ),
    "write_physical_cmb_input_report": (
        "oph_fpe.cosmology.physical_cmb_prediction",
        "write_physical_cmb_input_report",
    ),
    "write_physical_cmb_input_no_data_use_receipt": (
        "oph_fpe.cosmology.physical_cmb_prediction",
        "write_physical_cmb_input_no_data_use_receipt",
    ),
    "physical_cmb_frontier_report": (
        "oph_fpe.cosmology.physical_cmb_prediction",
        "physical_cmb_frontier_report",
    ),
    "write_physical_cmb_frontier_report": (
        "oph_fpe.cosmology.physical_cmb_prediction",
        "write_physical_cmb_frontier_report",
    ),
    "physical_cmb_output_comparison_report": (
        "oph_fpe.cosmology.physical_cmb_output",
        "physical_cmb_output_comparison_report",
    ),
    "write_physical_cmb_output_comparison_report": (
        "oph_fpe.cosmology.physical_cmb_output",
        "write_physical_cmb_output_comparison_report",
    ),
    "build_finite_covariant_parent_artifact_from_reports": (
        "oph_fpe.cosmology.physical_cmb_sources",
        "build_finite_covariant_parent_artifact_from_reports",
    ),
    "build_finite_parent_readiness_summary_from_reports": (
        "oph_fpe.cosmology.physical_cmb_sources",
        "build_finite_parent_readiness_summary_from_reports",
    ),
    "write_physical_cmb_source_readiness_report": (
        "oph_fpe.cosmology.physical_cmb_sources",
        "write_physical_cmb_source_readiness_report",
    ),
    "FrozenTransferConfig": (
        "oph_fpe.cosmology.frozen_transfer_likelihood",
        "FrozenTransferConfig",
    ),
    "frozen_transfer_likelihood_report": (
        "oph_fpe.cosmology.frozen_transfer_likelihood",
        "frozen_transfer_likelihood_report",
    ),
    "write_frozen_transfer_likelihood_report": (
        "oph_fpe.cosmology.frozen_transfer_likelihood",
        "write_frozen_transfer_likelihood_report",
    ),
    "finite_collar_boltzmann_bundle_report": (
        "oph_fpe.cosmology.finite_collar_boltzmann_bundle",
        "finite_collar_boltzmann_bundle_report",
    ),
    "write_finite_collar_boltzmann_bundle_report": (
        "oph_fpe.cosmology.finite_collar_boltzmann_bundle",
        "write_finite_collar_boltzmann_bundle_report",
    ),
    "finite_collar_cmb_projection_report": (
        "oph_fpe.cosmology.finite_collar_projection",
        "finite_collar_cmb_projection_report",
    ),
    "write_finite_collar_cmb_projection_report": (
        "oph_fpe.cosmology.finite_collar_projection",
        "write_finite_collar_cmb_projection_report",
    ),
    "scalar_quotient_report": ("oph_fpe.cosmology.scalar_quotient", "scalar_quotient_report"),
    "write_scalar_quotient_report": (
        "oph_fpe.cosmology.scalar_quotient",
        "write_scalar_quotient_report",
    ),
    "LambdaCDMParameters": ("oph_fpe.cosmology.camb_adapter", "LambdaCDMParameters"),
    "camb_lcdm_baseline_report": ("oph_fpe.cosmology.camb_adapter", "camb_lcdm_baseline_report"),
    "compare_camb_tt_to_benchmark": ("oph_fpe.cosmology.camb_adapter", "compare_camb_tt_to_benchmark"),
    "write_camb_lcdm_baseline_report": ("oph_fpe.cosmology.camb_adapter", "write_camb_lcdm_baseline_report"),
    "oph_screen_camb_report": ("oph_fpe.cosmology.camb_adapter", "oph_screen_camb_report"),
    "write_oph_screen_camb_report": ("oph_fpe.cosmology.camb_adapter", "write_oph_screen_camb_report"),
    "scale_compressed_cmb_camb_report": (
        "oph_fpe.cosmology.camb_adapter",
        "scale_compressed_cmb_camb_report",
    ),
    "write_scale_compressed_cmb_camb_report": (
        "oph_fpe.cosmology.camb_adapter",
        "write_scale_compressed_cmb_camb_report",
    ),
    "oph_inflation_cmb_camb_report": (
        "oph_fpe.cosmology.camb_adapter",
        "oph_inflation_cmb_camb_report",
    ),
    "write_oph_inflation_cmb_camb_report": (
        "oph_fpe.cosmology.camb_adapter",
        "write_oph_inflation_cmb_camb_report",
    ),
    "oph_exact_cmb_camb_report": (
        "oph_fpe.cosmology.camb_adapter",
        "oph_exact_cmb_camb_report",
    ),
    "write_oph_exact_cmb_camb_report": (
        "oph_fpe.cosmology.camb_adapter",
        "write_oph_exact_cmb_camb_report",
    ),
    "official_planck_readiness_report": (
        "oph_fpe.cosmology.camb_adapter",
        "official_planck_readiness_report",
    ),
    "write_official_planck_readiness_report": (
        "oph_fpe.cosmology.camb_adapter",
        "write_official_planck_readiness_report",
    ),
    "OPHConstants": ("oph_fpe.cosmology.oph_constants", "OPHConstants"),
    "B_A_z6_poisson_five_of_seven": (
        "oph_fpe.cosmology.oph_kernels",
        "B_A_z6_poisson_five_of_seven",
    ),
    "collar_poisson_counting_certificate": (
        "oph_fpe.cosmology.collar_poisson",
        "collar_poisson_counting_certificate",
    ),
    "write_collar_poisson_certificate": (
        "oph_fpe.cosmology.collar_poisson",
        "write_collar_poisson_certificate",
    ),
    "B_A_minimal_one_pole": ("oph_fpe.cosmology.oph_kernels", "B_A_minimal_one_pole"),
    "apply_projected_wl_selector": ("oph_fpe.cosmology.oph_kernels", "apply_projected_wl_selector"),
    "compressed_projection_fraction": (
        "oph_fpe.cosmology.oph_kernels",
        "compressed_projection_fraction",
    ),
    "normalized_projection_average": (
        "oph_fpe.cosmology.oph_kernels",
        "normalized_projection_average",
    ),
    "projected_amplitude": ("oph_fpe.cosmology.oph_kernels", "projected_amplitude"),
    "projected_amplitude_ratio": (
        "oph_fpe.cosmology.oph_kernels",
        "projected_amplitude_ratio",
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
    "selector_elimination_report": (
        "oph_fpe.cosmology.selector_elimination",
        "selector_elimination_report",
    ),
    "write_selector_elimination_report": (
        "oph_fpe.cosmology.selector_elimination",
        "write_selector_elimination_report",
    ),
    "DEFAULT_N_CRC": (
        "oph_fpe.cosmology.screen_capacity",
        "DEFAULT_N_CRC",
    ),
    "OPHScreenCapacityConstants": (
        "oph_fpe.cosmology.screen_capacity",
        "OPHScreenCapacityConstants",
    ),
    "screen_capacity_closure_report": (
        "oph_fpe.cosmology.screen_capacity",
        "screen_capacity_closure_report",
    ),
    "write_screen_capacity_closure_report": (
        "oph_fpe.cosmology.screen_capacity",
        "write_screen_capacity_closure_report",
    ),
    "capacity_readback_proxy_report": (
        "oph_fpe.cosmology.screen_capacity",
        "capacity_readback_proxy_report",
    ),
    "write_capacity_readback_proxy_report": (
        "oph_fpe.cosmology.screen_capacity",
        "write_capacity_readback_proxy_report",
    ),
    "build_public_record_reference_packet": (
        "oph_fpe.cosmology.public_record_capacity",
        "build_reference_packet",
    ),
    "evaluate_public_record_terminal": (
        "oph_fpe.cosmology.public_record_capacity",
        "evaluate_terminal",
    ),
    "evaluate_public_record_terminal_fiber": (
        "oph_fpe.cosmology.public_record_capacity",
        "evaluate_terminal_fiber",
    ),
    "write_public_record_capacity_report": (
        "oph_fpe.cosmology.public_record_capacity",
        "write_public_record_capacity_report",
    ),
    "ScaleBridgeInputs": (
        "oph_fpe.cosmology.scale_bridge",
        "ScaleBridgeInputs",
    ),
    "NoGClockBridgeInputs": (
        "oph_fpe.cosmology.scale_bridge",
        "NoGClockBridgeInputs",
    ),
    "dimensionless_pn_invariants": (
        "oph_fpe.cosmology.scale_bridge",
        "dimensionless_pn_invariants",
    ),
    "scale_bridge_report": (
        "oph_fpe.cosmology.scale_bridge",
        "scale_bridge_report",
    ),
    "write_scale_bridge_report": (
        "oph_fpe.cosmology.scale_bridge",
        "write_scale_bridge_report",
    ),
    "no_g_clock_bridge_report": (
        "oph_fpe.cosmology.scale_bridge",
        "no_g_clock_bridge_report",
    ),
    "write_no_g_clock_bridge_report": (
        "oph_fpe.cosmology.scale_bridge",
        "write_no_g_clock_bridge_report",
    ),
    "PNResonanceInputs": (
        "oph_fpe.cosmology.pn_resonance",
        "PNResonanceInputs",
    ),
    "ew_bridge_capacity_from_p_alpha": (
        "oph_fpe.cosmology.pn_resonance",
        "ew_bridge_capacity_from_p_alpha",
    ),
    "pn_resonance_report": (
        "oph_fpe.cosmology.pn_resonance",
        "pn_resonance_report",
    ),
    "write_pn_resonance_report": (
        "oph_fpe.cosmology.pn_resonance",
        "write_pn_resonance_report",
    ),
    "LeechEndpointBridgeInputs": (
        "oph_fpe.cosmology.leech_endpoint_bridge",
        "LeechEndpointBridgeInputs",
    ),
    "leech_endpoint_bridge_report": (
        "oph_fpe.cosmology.leech_endpoint_bridge",
        "leech_endpoint_bridge_report",
    ),
    "write_leech_endpoint_bridge_report": (
        "oph_fpe.cosmology.leech_endpoint_bridge",
        "write_leech_endpoint_bridge_report",
    ),
    "HadronSourceBackendInputs": (
        "oph_fpe.cosmology.hadron_source_backend",
        "HadronSourceBackendInputs",
    ),
    "hadron_source_backend_report": (
        "oph_fpe.cosmology.hadron_source_backend",
        "hadron_source_backend_report",
    ),
    "write_hadron_source_backend_bundle": (
        "oph_fpe.cosmology.hadron_source_backend",
        "write_hadron_source_backend_bundle",
    ),
    "GammaMorphologyInputs": (
        "oph_fpe.cosmology.gamma_morphology",
        "GammaMorphologyInputs",
    ),
    "gamma_morphology_report": (
        "oph_fpe.cosmology.gamma_morphology",
        "gamma_morphology_report",
    ),
    "write_gamma_morphology_bundle": (
        "oph_fpe.cosmology.gamma_morphology",
        "write_gamma_morphology_bundle",
    ),
    "signed_template_amplitude_interval": (
        "oph_fpe.cosmology.gamma_morphology",
        "signed_template_amplitude_interval",
    ),
    "SilenceToObservationInputs": (
        "oph_fpe.cosmology.silence_to_observation",
        "SilenceToObservationInputs",
    ),
    "silence_to_observation_report": (
        "oph_fpe.cosmology.silence_to_observation",
        "silence_to_observation_report",
    ),
    "write_silence_to_observation_report": (
        "oph_fpe.cosmology.silence_to_observation",
        "write_silence_to_observation_report",
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
    "GeometryBranch": ("oph_fpe.cosmology.spatial_curvature", "GeometryBranch"),
    "CurvatureClaimStatus": ("oph_fpe.cosmology.spatial_curvature", "CurvatureClaimStatus"),
    "spatial_curvature_status_report": (
        "oph_fpe.cosmology.spatial_curvature",
        "spatial_curvature_status_report",
    ),
    "friedmann_curvature_readout": (
        "oph_fpe.cosmology.spatial_curvature",
        "friedmann_curvature_readout",
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
    "inflation_certificate_bundle_report": (
        "oph_fpe.cosmology.inflation_certificates",
        "inflation_certificate_bundle_report",
    ),
    "write_inflation_certificate_bundle_report": (
        "oph_fpe.cosmology.inflation_certificates",
        "write_inflation_certificate_bundle_report",
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
    "H0S8CertificateInputs": ("oph_fpe.cosmology.h0s8_certificates", "H0S8CertificateInputs"),
    "h0s8_lane8_certificate_report": (
        "oph_fpe.cosmology.h0s8_certificates",
        "h0s8_lane8_certificate_report",
    ),
    "write_h0s8_lane8_certificate_report": (
        "oph_fpe.cosmology.h0s8_certificates",
        "write_h0s8_lane8_certificate_report",
    ),
    "oph_cnb_background_report": ("oph_fpe.cosmology.neutrino_background", "oph_cnb_background_report"),
    "write_oph_cnb_background_report": (
        "oph_fpe.cosmology.neutrino_background",
        "write_oph_cnb_background_report",
    ),
    "neutrino_mass_status": ("oph_fpe.cosmology.neutrino_status", "neutrino_mass_status"),
    "CONVENTIONAL_CAMB_NEUTRINO_ASSUMPTION": (
        "oph_fpe.cosmology.neutrino_status",
        "CONVENTIONAL_CAMB_NEUTRINO_ASSUMPTION",
    ),
    "CONVENTIONAL_CAMB_SUM_MNU_EV": (
        "oph_fpe.cosmology.neutrino_status",
        "CONVENTIONAL_CAMB_SUM_MNU_EV",
    ),
    "hot_release_report": ("oph_fpe.cosmology.hot_release", "hot_release_report"),
    "write_hot_release_report": ("oph_fpe.cosmology.hot_release", "write_hot_release_report"),
    "oph_cmb_stress_adapter_report": ("oph_fpe.cosmology.oph_cmb_adapter", "oph_cmb_stress_adapter_report"),
    "C_ell_oph": ("oph_fpe.cosmology.oph_screen_power", "C_ell_oph"),
    "D_ell_from_C_ell": ("oph_fpe.cosmology.oph_screen_power", "D_ell_from_C_ell"),
    "F_oph_k": ("oph_fpe.cosmology.oph_screen_power", "F_oph_k"),
    "OPHScreenPowerParams": ("oph_fpe.cosmology.oph_screen_power", "OPHScreenPowerParams"),
    "primordial_power_oph": ("oph_fpe.cosmology.oph_screen_power", "primordial_power_oph"),
    "write_oph_screen_power_report": ("oph_fpe.cosmology.oph_screen_power", "write_oph_screen_power_report"),
    "ScreenSpectrumParams": ("oph_fpe.cosmology.screen_spectrum", "ScreenSpectrumParams"),
    "screen_precision_eigenvalue": ("oph_fpe.cosmology.screen_spectrum", "screen_precision_eigenvalue"),
    "screen_cl": ("oph_fpe.cosmology.screen_spectrum", "screen_cl"),
    "screen_d_ell": ("oph_fpe.cosmology.screen_spectrum", "screen_d_ell"),
    "red_tilt_slope_check": ("oph_fpe.cosmology.screen_spectrum", "red_tilt_slope_check"),
    "source_edge_center_tilt": (
        "oph_fpe.cosmology.source_screen_spectrum",
        "edge_center_tilt",
    ),
    "source_amplitude_from_samples": (
        "oph_fpe.cosmology.source_screen_spectrum",
        "source_amplitude_from_samples",
    ),
    "primordial_amplitude_from_screen": (
        "oph_fpe.cosmology.source_screen_spectrum",
        "primordial_amplitude_from_screen",
    ),
    "source_screen_cl": (
        "oph_fpe.cosmology.source_screen_spectrum",
        "screen_cl",
    ),
    "source_thin_shell_cl": (
        "oph_fpe.cosmology.source_screen_spectrum",
        "thin_shell_cl",
    ),
    "source_radial_kernel_matrix": (
        "oph_fpe.cosmology.source_screen_spectrum",
        "radial_kernel_matrix",
    ),
    "source_radial_null_space": (
        "oph_fpe.cosmology.source_screen_spectrum",
        "radial_null_space",
    ),
    "source_family_forward_residual": (
        "oph_fpe.cosmology.source_screen_spectrum",
        "source_family_forward_residual",
    ),
    "source_spectrum_receipt": (
        "oph_fpe.cosmology.source_screen_spectrum",
        "build_receipt",
    ),
    "mellin_spherical_bessel_square": (
        "oph_fpe.cosmology.source_screen_spectrum",
        "mellin_spherical_bessel_square",
    ),
    "derivative_mellin_norm": (
        "oph_fpe.cosmology.source_screen_spectrum",
        "derivative_mellin_norm",
    ),
    "finite_window_stability_bound": (
        "oph_fpe.cosmology.source_screen_spectrum",
        "finite_window_stability_bound",
    ),
    "radial_projection_matrix": (
        "oph_fpe.cosmology.source_screen_spectrum",
        "radial_projection_matrix",
    ),
    "minimum_prior_continuation": (
        "oph_fpe.cosmology.source_screen_spectrum",
        "minimum_prior_continuation",
    ),
    "dilation_intertwiner_receipt": (
        "oph_fpe.cosmology.source_screen_spectrum",
        "dilation_intertwiner_receipt",
    ),
    "approximate_dilation_shape_bound": (
        "oph_fpe.cosmology.source_screen_spectrum",
        "approximate_dilation_shape_bound",
    ),
    "build_radial_receipt": (
        "oph_fpe.cosmology.source_screen_spectrum",
        "build_radial_receipt",
    ),
    "write_radial_receipt": (
        "oph_fpe.cosmology.source_screen_spectrum",
        "write_radial_receipt",
    ),
    "edge_center_clock_target": (
        "oph_fpe.cosmology.edge_center_clock",
        "edge_center_clock_target",
    ),
    "validate_edge_center_clock_evidence": (
        "oph_fpe.cosmology.edge_center_clock",
        "validate_edge_center_clock_evidence",
    ),
    "canonical_edge_clock_hash": (
        "oph_fpe.cosmology.edge_center_clock",
        "canonical_edge_clock_hash",
    ),
    "write_edge_center_clock_certificate": (
        "oph_fpe.cosmology.edge_center_clock",
        "write_edge_center_clock_certificate",
    ),
    "verify_collar_clause_packet": (
        "oph_fpe.cosmology.collar_clause",
        "verify_collar_clause_packet",
    ),
    "write_collar_clause_certificate": (
        "oph_fpe.cosmology.collar_clause",
        "write_collar_clause_certificate",
    ),
    "collar_clause_negative_controls": (
        "oph_fpe.cosmology.collar_clause",
        "collar_clause_negative_controls",
    ),
    "canonical_collar_evidence_hash": (
        "oph_fpe.cosmology.collar_clause",
        "canonical_evidence_hash",
    ),
    "maxent_green_spectrum_report": (
        "oph_fpe.cosmology.maxent_green_spectrum",
        "maxent_green_spectrum_report",
    ),
    "write_maxent_green_spectrum_report": (
        "oph_fpe.cosmology.maxent_green_spectrum",
        "write_maxent_green_spectrum_report",
    ),
    "repair_clock_report": (
        "oph_fpe.cosmology.repair_clock",
        "repair_clock_report",
    ),
    "write_repair_clock_report": (
        "oph_fpe.cosmology.repair_clock",
        "write_repair_clock_report",
    ),
    "repair_scale_closure_report": (
        "oph_fpe.cosmology.repair_scale_closure",
        "repair_scale_closure_report",
    ),
    "write_repair_scale_closure_report": (
        "oph_fpe.cosmology.repair_scale_closure",
        "write_repair_scale_closure_report",
    ),
    "ScalarRepairSemigroupSpec": (
        "oph_fpe.cosmology.scalar_repair_semigroup",
        "ScalarRepairSemigroupSpec",
    ),
    "scalar_repair_semigroup_report": (
        "oph_fpe.cosmology.scalar_repair_semigroup",
        "scalar_repair_semigroup_report",
    ),
    "write_scalar_repair_semigroup_report": (
        "oph_fpe.cosmology.scalar_repair_semigroup",
        "write_scalar_repair_semigroup_report",
    ),
    "finite_repair_transition_clock_report": (
        "oph_fpe.cosmology.finite_repair_transition_clock",
        "finite_repair_transition_clock_report",
    ),
    "write_finite_repair_transition_clock_report": (
        "oph_fpe.cosmology.finite_repair_transition_clock",
        "write_finite_repair_transition_clock_report",
    ),
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
