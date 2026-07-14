# Paired B_A Perturb/Resettle Diagnostic

- mode: `paired_cap_collar_perturb_resettle_B_A_parent_v0`
- rows: 8
- control rows: 24
- mean |B_A|: 0.0
- controls_fail: False
- control failures: {'no_perturbation': True, 'phase_shuffled_baryon_mode': False, 'wrong_k_label': False}
- B_A_PAIRED_DIAGNOSTIC_RECEIPT: False
- B_A_PARENT_RECEIPT: False
- physical_cmb_prediction: False

## Control Metrics

- `no_perturbation`: mode=null_suppression, ratio=0.0, separation=0.0, corr=None, pass=True
- `phase_shuffled_baryon_mode`: mode=paired_response_separation, ratio=0.0, separation=0.0, corr=None, pass=False
- `wrong_k_label`: mode=paired_response_separation, ratio=0.0, separation=0.0, corr=None, pass=False

Actual paired finite cap/collar perturb-resettle B_A parent diagnostic. No CMB data are used. Rows exercise finite screen repair dynamics, but they are not physical Boltzmann kernels until a common source functional, admissible tangent, lift-independent source vector, calibrated k/a units, exchange and gauge closure, and derivative-level refinement pass.
