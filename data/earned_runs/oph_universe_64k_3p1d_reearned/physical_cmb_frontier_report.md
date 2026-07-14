# Physical CMB Frontier

- measurement-comparable output receipt: `false`
- physical CMB prediction receipt: `false`
- physical CMB prediction ready: `false`
- official likelihood ready: `false`
- CDM-limit regression: `false`

## Gates

- `measurement_comparable_cmb_outputs`: `false` - 0 OPH diagnostic models; 0 total comparable models
- `no_data_use_firewall`: `true` - OPH input functions were assembled without measurement tables
- `source_provenance_firewall`: `false` - promoted CMB inputs have source-only dependency-DAG provenance
- `pooled_source_reducers`: `false` - nonlinear CMB input estimates are global, not shard-local averages
- `N_CRC_consensus_invariant`: `false` - N_CRC is not additively combined without a disjoint-capacity proof
- `finite_theorem_A_zeta`: `false` - source=diagnostic_proxy
- `finite_B_A_kernel`: `false` - rows=32
- `finite_rho_A`: `false` - rows=32
- `finite_certificate_theorem_grade`: `false` - compiler_ready=False; proxy=False
- `finite_covariant_parent`: `false` - blockers=[]
- `stress_energy_closure`: `false` - anomaly plus recipient stress tensors close exactly
- `recipient_stress_for_nonzero_Gamma_rec`: `false` - Gamma_rec_nonzero=True
- `exchange_current_closure_for_nonzero_Gamma_rec`: `false` - Gamma_rec_nonzero=True
- `physical_clock_for_Gamma_rec`: `false` - repair-step transition rate has a certified physical clock
- `active_fiber_for_Gamma_rec`: `false` - Gamma_rec response is calculated on the certified active fiber
- `conserved_sector_decomposition_for_Gamma_rec`: `false` - repair exchange is decomposed against the conserved sectors before promotion
- `common_parent_response_pole_for_Gamma_rec`: `false` - response pole is read in a common finite parent
- `gauge_independent_B_A`: `false` - B_A must be built from anomaly-frame baryon density, not a gauge-fixed scalar table
- `causal_response`: `false` - finite packet or auxiliary response characteristics are subluminal
- `finite_parent_refinement_convergence`: `false` - stress, response, and source projections converge under regulator refinement
- `B_A_kernel_physical_receipt`: `false` - diagnostic_candidate=False; rows=20
- `B_A_refinement_convergence`: `false` - patch_counts=[]
- `finite_collar_boltzmann_physical_certificate`: `false` - missing=['B_A_controls_fail', 'B_A_physical_kernel_receipt', 'Gamma_rec_physical_source_receipt', 'admissible_source_tangent_receipt', 'calibrated_a_evolution', 'common_primordial_anomaly_mode_basis', 'common_source_functional_receipt', 'cross_receipt_consistency', 'energy_momentum_exchange_closed', 'gauge_consistency_audited', 'native_repair_generator_receipt', 'no_posthoc_calibration_receipt', 'paired_B_A_diagnostic_receipt', 'physical_cmb_input_contract_passed', 'physical_freezeout_surface', 'physical_k_units_calibrated', 'refinement_convergence_passed', 'rho_A_eq_physical_source_receipt', 'rho_A_physical_source_receipt', 'screen_to_physical_k_association_calibrated', 'source_vector_sufficiency_receipt', 'static_dynamic_response_consistency_receipt']
- `finite_collar_projection_physical_k`: `false` - missing=[]
- `cdm_limit_regression`: `false` - Boltzmann plumbing/CDM-limit regression gate; requires CMB1 custom-parent CDM match against solver-native CDM
- `standard_model_off_regression`: `false` - control transfer with Standard Model/anomaly sources off
- `official_planck_likelihood_ready`: `false` - requires local official clik/Cobaya path and Planck likelihood data
- `blinded_full_observable_likelihood`: `false` - requires blinded TT/TE/EE/lensing/BAO/growth/weak-lensing/RSD/S8 execution
- `frozen_likelihood_protocol`: `false` - source_hash=False; solver_hash=False; likelihood_hash=False; source_freeze=False; solver_pins=False; closure=False
- `physical_cmb_input_contract`: `false` - 46 hard input blockers
- `physical_cmb_promotion_ready`: `false` - 73 promotion blockers

## Hard-Gate Gaps

- `measurement_comparable_cmb_outputs`: missing `None`; current `0 OPH diagnostics / 0 comparable models`; action `measurement_output`; blockers `none`
- `source_provenance_firewall`: missing `CMB source-provenance dependency-DAG receipt`; current `receipt=False; pooled=False; N_CRC=False`; action `cmb_source_provenance_report`; blockers `source_provenance_receipt_missing, pooled_source_reducer_receipt_missing, source_provenance_contradiction_check_failed, N_CRC_consensus_invariant_receipt_missing, global_likelihood_reduction_receipt_missing`
- `finite_theorem_A_zeta`: missing `theorem-grade finite A_zeta source`; current `source=diagnostic_proxy; diagnostic_present=False; finite_certificate_theorem_grade=False`; action `finite_certificate_report`; blockers `A_zeta_not_finite_derived`
- `finite_B_A_kernel`: missing `physical finite B_A(k,a) kernel receipt`; current `source=diagnostic_proxy; rows=32; B_A_KERNEL_RECEIPT=False; refinement=False`; action `B_A_kernel_report/B_A_kernel_refinement_report`; blockers `B_A_k_a_missing_or_not_finite, B_A_kernel_receipt_missing, B_A_diagnostic_rows_not_physical_kernel`
- `finite_rho_A`: missing `theorem-grade finite rho_A(a) source`; current `source=diagnostic_proxy; rows=32; finite_certificate_theorem_grade=False`; action `finite_certificate_report/parent_collar_ladder`; blockers `rho_A_missing_or_not_finite, rho_A_diagnostic_rows_not_physical_source`
- `finite_covariant_parent`: missing `finite covariant collar-packet parent receipt`; current `present=False; parent_receipt=False; blockers=[]`; action `finite_covariant_collar_packet_parent_report`; blockers `physical_clock_missing_for_promoted_Gamma_rec, active_fiber_missing_for_promoted_Gamma_rec, conserved_sector_decomposition_missing_for_promoted_Gamma_rec, common_parent_response_pole_missing_for_promoted_Gamma_rec, finite_covariant_parent_receipt_missing, stress_energy_closure_not_certified, recipient_stress_missing_for_nonzero_Gamma_rec, exchange_current_closure_missing_for_nonzero_Gamma_rec, gauge_independence_not_certified, causal_response_not_certified, refinement_convergence_not_certified`
- `frozen_likelihood_protocol`: missing `frozen immutable source/solver/likelihood hash protocol`; current `receipt=False; source_hash=False; solver_hash=False; likelihood_hash=False`; action `finite_covariant_collar_packet_parent_report/official_likelihood_report`; blockers `frozen_likelihood_protocol_not_certified, source_freeze_manifest_not_certified, solver_assumption_pin_not_certified, custom_parent_cdm_limit_regression_not_passed, standard_model_off_regression_not_passed, blinded_comparison_setup_not_certified, full_observable_likelihood_not_executed, frozen_transfer_likelihood_closure_not_certified, frozen_source_hash_missing, frozen_solver_hash_missing, frozen_likelihood_hash_missing`
- `finite_collar_boltzmann_physical_certificate`: missing `finite-collar Boltzmann physical certificate`; current `source_bundle=True; missing=['B_A_controls_fail', 'B_A_physical_kernel_receipt', 'Gamma_rec_physical_source_receipt', 'admissible_source_tangent_receipt', 'calibrated_a_evolution', 'common_primordial_anomaly_mode_basis', 'common_source_functional_receipt', 'cross_receipt_consistency', 'energy_momentum_exchange_closed', 'gauge_consistency_audited', 'native_repair_generator_receipt', 'no_posthoc_calibration_receipt', 'paired_B_A_diagnostic_receipt', 'physical_cmb_input_contract_passed', 'physical_freezeout_surface', 'physical_k_units_calibrated', 'refinement_convergence_passed', 'rho_A_eq_physical_source_receipt', 'rho_A_physical_source_receipt', 'screen_to_physical_k_association_calibrated', 'source_vector_sufficiency_receipt', 'static_dynamic_response_consistency_receipt']`; action `finite_collar_boltzmann_bundle_report`; blockers `finite_collar_boltzmann_physical_certificate_false, finite_collar_boltzmann_missing_B_A_controls_fail, finite_collar_boltzmann_missing_B_A_physical_kernel_receipt, finite_collar_boltzmann_missing_Gamma_rec_physical_source_receipt, finite_collar_boltzmann_missing_admissible_source_tangent_receipt, finite_collar_boltzmann_missing_calibrated_a_evolution, finite_collar_boltzmann_missing_common_primordial_anomaly_mode_basis, finite_collar_boltzmann_missing_common_source_functional_receipt, finite_collar_boltzmann_missing_cross_receipt_consistency, finite_collar_boltzmann_missing_energy_momentum_exchange_closed, finite_collar_boltzmann_missing_gauge_consistency_audited, finite_collar_boltzmann_missing_native_repair_generator_receipt, finite_collar_boltzmann_missing_no_posthoc_calibration_receipt, finite_collar_boltzmann_missing_paired_B_A_diagnostic_receipt, finite_collar_boltzmann_missing_physical_cmb_input_contract_passed, finite_collar_boltzmann_missing_physical_freezeout_surface, finite_collar_boltzmann_missing_physical_k_units_calibrated, finite_collar_boltzmann_missing_refinement_convergence_passed, finite_collar_boltzmann_missing_rho_A_eq_physical_source_receipt, finite_collar_boltzmann_missing_rho_A_physical_source_receipt, finite_collar_boltzmann_missing_screen_to_physical_k_association_calibrated, finite_collar_boltzmann_missing_source_vector_sufficiency_receipt, finite_collar_boltzmann_missing_static_dynamic_response_consistency_receipt`
- `finite_collar_projection_physical_k`: missing `OPH-derived physical k/ell calibration`; current `projection=False; physical_k=False; missing=[]`; action `finite_collar_cmb_projection_report/scale_bridge_report`; blockers `none`
- `official_planck_likelihood_ready`: missing `official clik/Cobaya likelihood execution readiness`; current `official readiness false or data paths not configured`; action `local Planck likelihood environment`; blockers `official_likelihood_not_ready`

## Blockers

- `o`
- `f`
- `f`
- `i`
- `c`
- `i`
- `a`
- `l`
- `_`
- `l`
- `i`
- `k`
- `e`
- `l`
- `i`
- `h`
- `o`
- `o`
- `d`
- `_`
- `n`
- `o`
- `t`
- `_`
- `r`
- `e`
- `a`
- `d`
- `y`

## Best OPH Diagnostic Output

- model: `None`
- chi2/bin: `None`
- source report: `None`

## Claim Boundary

Physical-CMB frontier report. Measurement-comparable TT curves are physical-unit outputs, but they remain diagnostic until the finite input contract, finite-source promotion gates, and official likelihood execution gates all pass.
