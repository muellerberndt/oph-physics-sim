# OPH-FPE Pro Handoff: Neutral 3D Bulk And CMB Prediction Audit

Date: 2026-06-10

This package is self-contained. Do not assume access to the local filesystem. The ZIP beside this file includes the relevant simulator code, tests, configs, selected paper sources, cosmology notes, current result artifacts, and a recent diff patch.

## Executive Status

We need exact instructions for two goals:

1. Prove that the OPH paper theorem chain is correctly instantiated by the simulator and that observers see a 3+1D Lorentz/H3 experience.
2. Turn the current CMB/cosmology diagnostics into genuine finite-lattice-derived physical predictions, or specify the exact missing objects/gates.

Current top-level result flags from `data/current_pack/claims.json`:

```json
{
  "chart_level_3p1": true,
  "theorem_assisted_h3_bulk": true,
  "strict_blind_record_transition_3d_candidate": true,
  "strict_neutral_bulk": false,
  "physical_cmb_prediction": false,
  "scale_compressed_cmb_curve_comparable": true,
  "scale_compressed_cmb_physical_prediction": false,
  "boltzmann_finite_repair_clock_rows_emitted": true
}
```

Interpretation:

- We now have a theorem-assisted observer-facing H3 populated chart result.
- We have a strict blind record-transition rank-3 candidate.
- We still do not have a strict neutral third-person 3D bulk proof.
- We have measurement-comparable CMB curves, but not a certified physical CMB prediction.

## Current Best 3D/Bulk Result

Current corrected run:

```text
data/stage266/
```

Key chart report:

```json
{
  "incidence_mode": "transition_affinity_modular_response_mixture",
  "observer_chart_object_h3_receipt": true,
  "observer_chart_bulk_population_receipt": true,
  "compactness_distribution_population_receipt": true,
  "h3_beats_shuffled_incidence_robust": true,
  "h3_not_boundary_dominated": true,
  "median_h3_compactness_normalized": 0.12123398066476862,
  "median_s2_boundary_compactness_normalized": 0.4964235350533584,
  "median_shuffled_h3_compactness_normalized": 0.14905445034468198,
  "object_count": 4096,
  "localized_not_boundary_object_count": 3989
}
```

Strict blind observer-record candidate:

```json
{
  "record_transition_rank3_receipt": true,
  "correlation_dimension": 2.701366590578913,
  "local_mle_dimension": 2.945863254948163,
  "s2_distance_correlation": 0.0025379516378554352,
  "strict_neutral_bulk": false
}
```

Claim boundary:

- This is good evidence that observer-visible transition objects populate the theorem-side H3 chart and are not boundary dominated.
- It is not yet a strict neutral third-person bulk proof, because the chart-level H3 route is theorem-assisted rather than fully reconstructed from neutral observer records alone.

## Current Best CMB Result

From `data/current_pack/scale_compressed_cmb_camb_report.json`:

```json
{
  "measurement_comparable_cmb_curve": true,
  "screen_camb_transfer_receipt": true,
  "physical_cmb_prediction": false,
  "scale_compressed_scalar_tilt": {
    "shape_correlation": 0.9998574616665208,
    "normalized_rmse": 0.0168842135427968,
    "amplitude_fit_chi2_per_bin": 0.9544981569825821,
    "first_peak_ell": 225.164945,
    "benchmark_first_peak_ell": 225.164945,
    "mean_absolute_fractional_error": 0.01943526118368078
  },
  "scale_compressed_ir_kernel": {
    "shape_correlation": 0.9998534542805271,
    "normalized_rmse": 0.01711991352040137,
    "amplitude_fit_chi2_per_bin": 0.9615050793060024,
    "first_peak_ell": 225.164945,
    "benchmark_first_peak_ell": 225.164945,
    "mean_absolute_fractional_error": 0.019487176662533225
  }
}
```

Claim boundary:

- This is a measurement-comparable TT curve pipeline.
- It is not yet a physical OPH CMB prediction because not all inputs are finite-lattice-derived under no-data-use gates.

## What Changed Recently

The code was corrected away from support-overlap objects:

- `code/oph_fpe/observers/objects.py`
  - Transition-affinity record families are first-class.
  - Object consensus scoring now uses observer-visible transition/packet histograms before support fallback.

- `code/oph_fpe/bulk/observer_reconstruction.py`
  - Neutral record-family and counterfactual similarities now prefer transition-affinity evidence over support overlap.
  - Added strict blind feature group sweep.
  - Added predeclared `record_transition_histogram` rank-3 diagnostic.

- `code/oph_fpe/bulk/record_to_h3.py`
  - H3 object population can use `transition_affinity_modular_response_mixture`.
  - Added `compactness_distribution_population_receipt`, because the old localized-count gate saturated for both real and shuffled objects even when real median compactness robustly beat shuffled and S2.

- Active configs now use:

```yaml
observer_objects.family_mode: transition_affinity
observer_chart_population.incidence_mode: transition_affinity_modular_response_mixture
```

Validation:

```text
356 passed, 1 warning
```

## Included Package Contents

High-level:

```text
code/                 selected simulator implementation files
tests/                focused tests covering object, reconstruction, H3, CMB pack
configs/              active 64k/256k/1M configs
papers/               relevant current OPH paper TeX sources
cosmology_notes/      selected CMB/H0S8/neutrino notes
data/current_pack/    current measurement pack reports and CSV outputs
data/stage266/        corrected 256k run reports and sampled observer views
data/summary_metrics.json
recent_code_diff.patch
included_files.txt
```

Important files to read first:

```text
data/summary_metrics.json
data/current_pack/claims.json
data/current_pack/comparable_data_snapshot.json
data/stage266/bulk_proof_certificate_report.json
data/stage266/observer_chart_object_h3_report.json
data/stage266/bulk_reconstruction_report.json
code/oph_fpe/observers/objects.py
code/oph_fpe/bulk/observer_reconstruction.py
code/oph_fpe/bulk/record_to_h3.py
code/oph_fpe/cosmology/boltzmann_inputs.py
code/oph_fpe/cosmology/finite_repair_transition_clock.py
```

## Questions For Pro

Please answer as an implementation/theorem audit, not as motivational prose.

### A. Neutral 3D Bulk Proof

1. What exact finite objects must exist for a strict neutral 3D bulk proof in OPH-FPE?
2. Is the current distinction correct?

```text
chart-level/theorem-assisted H3 populated bulk = true
strict neutral third-person bulk = false
```

3. Is `transition_affinity_modular_response_mixture` a legitimate observer-facing object construction under the papers, or does it still smuggle in H3/chart data?
4. Does the new `compactness_distribution_population_receipt` gate correctly replace the old localized-count gate when localized counts saturate for both real and shuffled incidence?
5. What controls are still missing before calling `observer_chart_bulk_population_receipt` a proof of “observers see 3+1D”?
6. What controls are still missing before calling `strict_neutral_bulk = true`?
7. Should the neutral reconstruction use:
   - record-transition histograms only,
   - transition-affinity object incidence,
   - counterfactual continuation classes,
   - cap/BW response kernels,
   - or some mathematically specified combination?
8. Is the strict blind rank-3 result meaningful?

```text
d_corr = 2.701366590578913
d_mle = 2.945863254948163
s2_distance_correlation = 0.0025379516378554352
```

9. What exact pass/fail thresholds should be used for neutral dimension gates?
10. Should fractional finite dimension estimates like 2.70 be accepted as finite-regulator convergence, or must the estimator return much closer to 3?
11. Do we need refinement scaling across 64k/256k/1M with identical seeds/config families before any proof claim?
12. What is the minimal next run matrix?

### B. Paper Theorem Instantiation

Please audit the implementation against the included papers:

```text
papers/recovering_relativity_and_standard_model_structure_from_observer_overlap_consistency_compact.tex
papers/reality_as_consensus_protocol.tex
papers/screen_microphysics_and_observer_synchronization.tex
papers/observers_are_all_you_need.tex
```

Questions:

1. Which theorem hypotheses are currently instantiated?
2. Which theorem hypotheses are merely approximated or surrogate-coded?
3. Is the simulator correctly separating:

```text
S2 screen/cap chart
support-visible BW/KMS cap flow
Conf+(S2) ~= SO+(3,1)
H3 = SO+(3,1)/SO(3)
observer-record/object population of H3
strict neutral third-person bulk reconstruction
```

4. Is there a theorem-side route that should make `strict_neutral_bulk` unnecessary for proving observer experience?
5. How should the simulator prove “observers see 3+1D” without overclaiming a God’s-eye third-person bulk?

### C. CMB Physical Prediction

Current status:

```text
measurement-comparable TT curves exist
physical_cmb_prediction = false
```

We need exact instructions to make `physical_cmb_prediction = true`.

Please specify:

1. What finite-lattice-derived quantities must be emitted?
2. Which current quantities are acceptable as theorem-side constants and which must be rederived from runs?
3. What is the exact role of:

```text
P
N / screen capacity
q_IR
ell_IR
eta_R
n_s
A_zeta
B_A(k,a)
Gamma_rec(k,a)
repair clock kappa_rep
freezeout cycle/surface
```

4. Is the scale-compressed 24-round branch a valid way to obtain CMB-scale observables, or only a visual/diagnostic compression?
5. How do we derive acoustic peaks from OPH repair/synchronization rather than importing them through CAMB/LambdaCDM parameters?
6. What minimal Boltzmann adapter is legitimate?
7. What no-data-use firewall is required before comparing to Planck?
8. What exact outputs should be written:

```text
P_prim(k)
transfer_functions(k,a)
C_l^TT, TE, EE
lensing phi-phi
H(a), Omega_i(a)
```

9. Can we get any physical prediction before full CAMB/CLASS, such as:
   - scalar tilt `n_s`,
   - low-l parity/asymmetry,
   - IR cutoff,
   - finite screen capacity signatures,
   - neutrino/CNB values,
   - H0/S8 effective ranges,
   - galaxy RAR/BTFR?

10. Which of those can honestly be called predictions now?

### D. Specific Code Review Requests

Please inspect:

```text
code/oph_fpe/observers/objects.py
code/oph_fpe/bulk/observer_reconstruction.py
code/oph_fpe/bulk/record_to_h3.py
code/oph_fpe/cosmology/comparable_data.py
code/oph_fpe/cosmology/boltzmann_inputs.py
code/oph_fpe/cosmology/finite_repair_transition_clock.py
code/oph_fpe/scale/bw_array.py
```

Please return:

1. Concrete code changes, with function names.
2. New test cases to add.
3. Config changes for the next 64k/256k/1M sweeps.
4. Which current claims should be downgraded or upgraded.
5. Whether the new `compactness_distribution_population_receipt` is valid or too permissive.
6. Whether `observer_chart_bulk_population_receipt = true` is now justified for stage266.
7. Exact instructions for how to make `strict_neutral_bulk = true`.
8. Exact instructions for how to make `physical_cmb_prediction = true`.

## Proposed Next Run Matrix Unless Pro Changes It

Do not spend cloud money until Pro confirms the gates. The likely next local/GCP matrix:

```bash
python3 -m oph_fpe.cli run-bw-sweep \
  --configs configs/e4_shared_observer_bulk_64k_local_tokens_feature_select.yml \
            configs/e4_shared_observer_bulk_256k_particles.yml \
  --seeds 20261049,20261050,20261051,20261052 \
  --workers 4 \
  --inner-jobs 1 \
  --out-dir runs/transition_affinity_h3_refinement_20260610
```

Then:

```bash
python3 -m oph_fpe.cli comparable-data \
  --run-dir runs/transition_affinity_h3_refinement_20260610 \
  --include runs/current_finite_collar_cosmology_sources_20260610/comparable_data_snapshot \
  --out runs/transition_affinity_h3_refinement_20260610/comparable_data_snapshot
```

Acceptance target:

```text
chart-level 3+1D: true
theorem-assisted H3 object population: true across seeds
non-boundary H3 population: true across seeds
strict blind record-transition rank-3: stable across seeds/refinement
strict neutral bulk: only true if Pro-defined neutral gates pass
physical CMB prediction: only true if finite-derived CMB input gates pass
```

## Current Known Blockers

1. Strict neutral bulk is false.
2. Physical CMB prediction is false.
3. Repair clock is not certified to `kappa_rep = e`.
4. Some CMB constants are theorem-side/selector-side, not finite-lattice-derived.
5. Acoustic peaks are currently mostly CAMB transfer plumbing, not independently produced by finite OPH microphysics.
6. Need Pro to decide whether theorem-assisted observer experience is sufficient for the desired “observers see 3+1D” proof.

## Requested Pro Output Format

Please provide:

```text
1. Verdict on current theorem alignment.
2. Exact missing theorem objects.
3. Exact neutral bulk proof recipe.
4. Exact CMB prediction recipe.
5. Concrete code patch plan.
6. Concrete test plan.
7. Concrete run plan.
8. Claim-language corrections.
```

Please avoid general encouragement. We need implementable instructions.
