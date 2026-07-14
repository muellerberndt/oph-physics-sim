# Physical CMB Source Readiness

Source-side physical-CMB readiness builder. It writes/audits finite source, stress-parent, Boltzmann-handoff, and frozen-likelihood artifacts. Passing these gates can make the hard input contract eligible; it is not itself a CMB prediction or likelihood result.

## Parent

- parent receipt: `false`
- stress closure: `false`
- gauge independence: `false`
- causal response: `false`
- refinement convergence: `false`
- frozen likelihood protocol: `false`

## Boltzmann Handoff

- diagnostic input table written: `true`
- CDM-limit solver ready: `false`
- finite-collar source bundle: `true`
- physical export certificate: `false`

## Blockers

- `explicit_finite_covariant_parent_artifact_missing`
- `finite_certificate_not_theorem_grade`
- `B_A_kernel_receipt_missing`
- `energy_momentum_exchange_not_closed`
- `gauge_consistency_not_audited`
- `regulator_refinement_convergence_not_passed`
- `cdm_limit_regression_not_passed`
- `official_likelihood_not_ready`
- `frozen_likelihood_protocol_not_ready`
- `finite_collar_boltzmann_missing_physical_k_units_calibrated`
- `finite_collar_boltzmann_missing_screen_to_physical_k_association_calibrated`
- `finite_collar_boltzmann_missing_calibrated_a_evolution`
- `finite_collar_boltzmann_missing_energy_momentum_exchange_closed`
- `finite_collar_boltzmann_missing_gauge_consistency_audited`
- `finite_collar_boltzmann_missing_refinement_convergence_passed`
- `finite_collar_boltzmann_missing_physical_freezeout_surface`
- `finite_collar_boltzmann_missing_common_primordial_anomaly_mode_basis`
- `finite_collar_boltzmann_missing_cross_receipt_consistency`
- `finite_collar_boltzmann_missing_no_posthoc_calibration_receipt`
- `finite_collar_boltzmann_missing_physical_cmb_input_contract_passed`
- `finite_collar_boltzmann_missing_B_A_physical_kernel_receipt`
- `finite_collar_boltzmann_missing_rho_A_physical_source_receipt`
- `finite_collar_boltzmann_missing_rho_A_eq_physical_source_receipt`
- `finite_collar_boltzmann_missing_Gamma_rec_physical_source_receipt`
- `finite_collar_boltzmann_missing_common_source_functional_receipt`
- `finite_collar_boltzmann_missing_admissible_source_tangent_receipt`
- `finite_collar_boltzmann_missing_source_vector_sufficiency_receipt`
- `finite_collar_boltzmann_missing_native_repair_generator_receipt`
- `finite_collar_boltzmann_missing_static_dynamic_response_consistency_receipt`
