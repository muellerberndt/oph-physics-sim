# OPH Boltzmann Input Readout

Machine-readable bridge from current OPH-CMB diagnostics toward a future CAMB/CLASS anomaly module. Only the CDM-limit rows are solver-ready as a standard regression target. The OPH repair-exchange rows are diagnostic finite-collar readouts, not physical CMB predictions.

## Readiness

- CDM-limit solver ready: False
- diagnostic repair-exchange table ready: True
- B_A parent diagnostic table ready: True
- finite repair-clock diagnostic table ready: True
- physical prediction ready: False
- missing gates: cdm_limit_regression_receipt, finite_transition_clock_certified, finite_collar_parent_theorem_grade, rho_A_transport_receipt, anomaly_abundance_source_receipt, rho_A_source_receipt, rho_A_a_physical_emitted, rho_A_eq_a_physical_emitted, Gamma_rec_k_a_physical_emitted, B_A_k_a_physical_emitted, gauge_consistency_audited, full_likelihood_ready

## CDM Limit

- row count: 4
- status: external_lambda_cdm_regression_target

## Diagnostic Repair Exchange

- row count: 16
- status: finite_collar_shape_proxy_not_physical_boltzmann_input
- mean Gamma_rec/H shape proxy: 0
- mean B_A shape proxy: 0

## B_A Parent Diagnostic

- row count: 8
- status: finite_collar_report_backed_parent_diagnostic_not_physical_kernel
- mean B_A parent diagnostic: 0

## Finite Repair-Clock Diagnostic

- row count: 4
- status: finite_transition_matrix_gamma_rec_diagnostic_not_certified_clock
- mean Gamma_rec/H diagnostic: 0

## Output Files

- `oph_boltzmann_input_report.json`
- `oph_boltzmann_cdm_limit_rows.csv`
- `oph_boltzmann_diagnostic_repair_rows.csv`
- `oph_boltzmann_b_a_parent_rows.csv`
- `oph_boltzmann_finite_repair_clock_rows.csv`
