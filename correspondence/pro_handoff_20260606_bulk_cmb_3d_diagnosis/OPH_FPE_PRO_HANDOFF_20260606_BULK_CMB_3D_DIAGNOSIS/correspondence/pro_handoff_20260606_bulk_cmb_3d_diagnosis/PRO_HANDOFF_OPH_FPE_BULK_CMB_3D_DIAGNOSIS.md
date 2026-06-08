# OPH-FPE Pro Handoff: Bulk/CMB/3D Diagnosis

Date: 2026-06-06

Repository: `oph-physics-sim`

## Executive Status

The simulator now has stable finite-screen receipts and measurement-facing diagnostic outputs, but it still does **not** establish populated 3D bulk emergence, matter particles, or a physical CMB prediction.

Current strongest positive receipts:

- Finite repair settles: `final_phi = 0` on 4k, 64k, and 256k diagnostics.
- Direct finite BW/KMS cap-flow sanity selects `2pi` and beats wrong-normalization/no-flow controls on the direct transition-response lane.
- Chart-level conformal Lorentz/H3 construction is present:
  - `Conf+(S2) ~ SO+(3,1)`
  - spatial chart `H3 = SO+(3,1)/SO(3)`
  - emitted as `conformal_h3_spatial_chart_receipt = true`
- Observer-facing readouts, object families, S3 screen/collar holonomy defects, and C_l screen proxies are emitted.
- CMB-lite diagnostics compare freezeout-screen angular spectra to local Planck 2018 TT binned data.

Current blocking failures:

- `bulk_3d_established = false`
- `physical_cmb_prediction = false`
- `particle_matter_receipt = false`
- Endogenous modular generator lane still fails controls.
- H3 modular-response fit beats shuffle/no-perturbation controls, but wrong-scale controls still win.
- Strict blind observer reconstruction is usable and not S2-leaky, but dimension estimates are not 3.

The current result is best described as:

> A finite OPH screen diagnostic with repair, records, direct BW/KMS cap-flow sanity, chart-level Lorentz/H3 construction, screen C_l comparison, and controlled negative results for populated neutral 3D bulk and physical CMB prediction.

It must **not** be described as:

- proof of 3D bulk emergence,
- early-universe simulation,
- physical CMB prediction,
- particle emergence in a 3D bulk.

## Key Recent Code Changes

1. Fixed an overclaim bug in the 3D gate.

Previously `emergence_status_report.json` could report `bulk_3d_established = true` from object/H3 support-profile candidates even when `bulk_reconstruction_report.json` had `bulk_3d_established = false`. This is now blocked.

Current gate requires:

- object/bulk-population receipt, and
- neutral reconstruction pass, and
- strict blind observer bulk audit usable, and
- no forbidden S2/support/radial evidence leakage.

2. Added strict blind observer-feature audit.

File: `oph_fpe/bulk/observer_reconstruction.py`

Forbidden evidence keys:

- `axis`
- `support_nodes`
- `cap_membership`
- `cap_axis`
- `s2_centroid`
- `s2_boundary_compactness`
- `raw_screen_distance`
- `screen_distance`
- `radial_depth`
- `modular_depth`

Allowed blind feature keys:

- `record_transition_histogram`
- `checkpoint_class_transition`
- `perturb_resettle_signature`
- `counterfactual_stability`
- `sector_change_signature`
- `repair_response_spectrum`

3. Wired blind transition features into emitted observer rows.

File: `oph_fpe/scale/bw_array.py`

The observer rows now expose:

- distributional transition summaries,
- sector/record-family change counters,
- perturb-resettle response spectra,
- modular-response cluster bins.

Important: I initially used exact categorical transition-token hash identities in the blind feature vector. That inflated dimension artificially. I changed it to distributional moments/top-mass summaries. The exact histograms are still available for object extraction, but the blind dimension estimator now consumes only summary statistics.

4. Improved CMB readouts.

Files:

- `oph_fpe/cosmology/cmb_compare.py`
- `oph_fpe/cosmology/comparable_data.py`
- `oph_fpe/cosmology/camb_adapter.py`
- `oph_fpe/cosmology/oph_cmb_adapter.py`
- `oph_fpe/cosmology/ba_parent.py`
- `oph_fpe/cosmology/anomaly_fluid.py`

Main changes:

- Positive-amplitude CMB shape scoring so anti-correlated spectra cannot win by negative amplitude.
- Clear separation between normalized-axis screen diagnostics and real-ell physical comparison.
- CAMB LambdaCDM baseline report for plumbing only, not OPH prediction.
- OPH-CMB adapter remains gated: no physical prediction until real `rho_A(a)`, `Gamma_rec(k,a)`, and `B_A(k,a)` are emitted.

5. Added reproducibility improvements.

- Lazy imports in `oph_fpe/cosmology/__init__.py`
- CLI branch-local imports in `oph_fpe/cli.py`
- Optional dependency extras in `pyproject.toml`
- `REPRODUCTION.md`

## Current Best Runs

### 256k local-token bulk/CMB diagnostic

Run:

```text
runs/bulk_blind_moment_features_256k_anchor0_20260606/
  e4_shared_observer_bulk_256k_local_tokens_multishuffle_seed20260751_seed20260752_373e0a48/
```

Summary:

```text
patch_count: 262144
final_phi: 0
bw_primary_mode: state_derived_modular_probe
bw_primary_median: 1.841190449272222e-15
transition_scale_selection.primary_source: kms_collar_transport_response
transition_scale_selection.selected_label: 2pi
state_derived_correct_beats_controls: true
bulk_3d_established: false
SCREEN_PROXY_CMB_RECEIPT: true
particle_matter_receipt: false
```

Neutral/blind reconstruction:

```text
neutral correlation dimension: 6.329321067910755
neutral local MLE dimension: 7.165568774068259
blind usable: true
blind feature width: 55
blind S2 distance correlation: 0.00781518865357141
blind S2 leakage audit pass: true
blind candidate 3D window: false
blind correlation dimension: 4.652946438929299
blind local MLE dimension: 5.292639713930718
```

Object/H3 population:

```text
observer_chart_object_count: 349
localized_object_count: 172
localized_not_boundary_object_count: 172
shuffled_localized_object_p90: 171
observer_chart_bulk_population_receipt: false
record_family_h3_bulk_population_candidate: false
OBJECT_BULK_POPULATION_RECEIPT: false
```

Interpretation:

The observer features are not simply leaking S2 boundary geometry, but they still do not form a 3D neutral bulk. The object population is close to shuffled controls, so it cannot be accepted.

### 64k direct transition-response control variant

Run:

```text
runs/bw_state_mode_variants_64k_20260606/
  e4_shared_observer_bulk_64k_anchor0_direct_transition/
```

Summary:

```text
patch_count: 65536
final_phi: 0
state_mode: transition_response_unitary
bw_primary_median: 1.556220012389479e-15
state_selected_scale_label: 2pi
state_derived_correct_beats_controls: true
direct_transition_automorphism: true
endogenous_modular_generator: false
bulk_3d_established: false
```

Blind reconstruction:

```text
neutral primary dimension: 6.2808660060403705
blind correlation dimension: 4.291494035542733
blind local MLE dimension: 5.463960878142723
blind S2 distance correlation: 0.007269207982178252
```

H3 modular response:

```text
MODULAR_RESPONSE_KERNEL_TO_H3_RECEIPT: false
heldout_normalized_rmse: about 0.9666
h3 beats shuffled/no-perturbation: true
wrong-scale controls still win: true
```

H3 candidate-seed robustness:

```text
seed_count: 4
receipt_count: 0
receipt_fraction: 0.0
candidate_3d_window_count: 0
mean_heldout_normalized_rmse: 0.9537760927530359
mean_heldout_explained_variance: 0.09029022371992776
```

Interpretation:

The direct finite automorphism sanity path is clean, but the H3 modular-response fit is not robust and wrong-scale controls win. This blocks populated bulk evidence.

### 64k endogenous history-kernel contrast

Run:

```text
runs/bw_state_mode_variants_64k_20260606/
  e4_shared_observer_bulk_64k_anchor0_reg1e12/
```

Summary:

```text
state_mode: history_transition_kernel
endogenous_modular_generator: true
state_selected_scale_label: 1x
correct_beats_controls: false
```

Interpretation:

The endogenous finite history-kernel surrogate does not reach the BW `2pi` normalization surface. This is a key implementation/theorem-surrogate problem.

### Small-time negative variant

Run:

```text
runs/h3_smalltime_variants_64k_20260606/
  e4_shared_observer_bulk_64k_anchor0_smalltime_direct/
```

Summary:

```text
times: [0.025, 0.05, 0.1]
transition_scale_selection.selected_label: 1x
support_visible_lorentz_3p1_kinematics_receipt: false
SCREEN_PROXY_CMB_RECEIPT: false
```

Interpretation:

Small modular times did not fix wrong-scale control issues. The previous `0.1` KMS lane remains the better finite-regulator setting.

## CMB/Measurement Status

Current CMB lanes:

1. `CAMB LambdaCDM baseline`

This matches local Planck 2018 TT binned data well, but it is only external LambdaCDM plumbing.

```text
shape_correlation: 0.9998604571617498
amplitude_fit_chi2_per_bin: 0.9444957497070865
first_peak_ell: 225.164945
physical_cmb_prediction: false
```

2. `OPH screen C_l proxy`

Best current 256k run:

```text
field: record_signature_smooth_k32
normalized-axis Planck-lite TT shape correlation: 0.4547349483094521
normalized RMSE: 0.8906268167902872
sim ell max: 48
Planck binned ell max: 2499.024
real-ell comparison usable: false
physical_cmb_prediction: false
```

3. `CMB screen-basis transfer`

This is the strongest current Planck-facing diagnostic, but it fits weights against Planck TT and tests transfer across patch counts. It is not a prediction.

```text
train shape correlation: 0.870257367328186
test shape correlation: 0.7480240172497512
max control test correlation: 0.1576983228734992
physical_cmb_prediction: false
```

4. `OPH Boltzmann/CMB adapter`

The adapter is scaffolded but gated closed.

Missing physical quantities:

- theorem-grade `rho_A(a)`
- `rho_A_eq(a)`
- `Gamma_rec(k,a)`
- non-fit `B_A(k,a)` from controlled perturbation response
- CAMB/CLASS anomaly module

Current adapter output is finite-collar diagnostic only.

## Main Questions for Pro

### 1. Is the finite endogenous modular-generator surrogate wrong?

The paper says the theorem-side object is the modular automorphism group of the support-visible prime cap pair. The finite sim currently has:

- direct transition-response automorphism lane: passes `2pi`;
- endogenous history-transition kernel lane: selects `1x`, fails controls.

Question:

What finite surrogate should replace `history_transition_kernel` so it better approximates the paper’s support-visible modular automorphism?

Specific concern:

The current finite `rho_C` / transition-kernel construction may be too diagonal/classical, too Markov-summary-like, or using the wrong support-visible basis. It may not encode the correct noncommutative/collar operator structure.

### 2. Are the wrong-scale H3 controls correctly defined?

The H3 modular-response fit beats shuffled/no-perturbation controls, but wrong-scale controls win strongly:

```text
wrong_scale_win_fraction ~ 0.90-0.95
two_pi_h3_fit_win_fraction ~ 0.05-0.10
```

Question:

Does that mean the H3 modular-response observable is wrong, or are the wrong-scale controls unfairly easier because they fit local response features with affine nuisance parameters rather than testing a geometric action in the correct basis?

### 3. What observer-visible object should populate H3?

Current object extraction still fails controls:

- object families exist,
- overlap agreement is about `0.50`,
- counterfactual stability is high,
- but object incidence is close to shuffled controls.

Question:

In the paper mechanism, should H3 population come from:

- persistent record families,
- cap modular response kernels,
- edge-sector/holonomy transport,
- Markov collar recovery channels,
- observer checkpoint continuation classes,
- some other support-visible object?

The current object extraction may still be too screen-local or too histogram-like.

### 4. Should neutral reconstruction estimate 3D at all?

The paper gives chart-level 3D/H3 from:

```text
Conf+(S2) ~= SO+(3,1)
H3 = SO+(3,1)/SO(3)
```

The simulator already emits that chart receipt. But the neutral blind observer-distance reconstruction estimates roughly 4.6-5.3 at 256k.

Question:

Should the finite sim require a neutral distance estimator to return 3, or should the correct claim be:

1. chart-level H3 is theorem-derived,
2. observer records must be shown to populate this chart under controls,
3. dimension estimation is only a secondary sanity check?

If so, what is the right populated-H3 receipt?

### 5. What is the first honest CMB prediction path?

Current screen C_l proxy has weak normalized-axis correlation and cannot compare in real ell-space. The adapter needs OPH anomaly source terms.

Question:

Is the next CMB path:

- increase screen ell resolution and keep screen C_l shape diagnostics,
- derive non-fit `B_A(k,a)` from finite collar perturbations,
- implement a standalone anomaly-fluid ODE first,
- or directly implement a CAMB/CLASS custom source module?

My current judgment is:

```text
CDM-limit CAMB baseline is ready as plumbing.
OPH physical CMB prediction is not ready.
Next physical gate should be non-fit B_A(k,a) from controlled finite-difference collar perturbation, then standalone anomaly ODE, then CAMB/CLASS.
```

Please confirm or correct.

### 6. Are current 3D failure modes a contradiction of the papers?

My current interpretation:

No. The papers derive chart-level Lorentz/H3 on the support-visible BW scaling branch. The finite sim has chart-level H3, but it has not yet shown that records/objects/defects populate H3 under strict observer-visible controls.

Question:

Is that interpretation faithful to:

- `recovering_relativity_and_standard_model_structure_from_observer_overlap_consistency_compact.tex`
- `reality_as_consensus_protocol.tex`
- `screen_microphysics_and_observer_synchronization.tex`

Or should the simulator mechanically produce populated 3D bulk once the cap/BW receipt passes?

## Files Included in ZIP

The ZIP beside this markdown contains:

### Source modules

- `oph_fpe/scale/bw_array.py`
- `oph_fpe/bulk/observer_reconstruction.py`
- `oph_fpe/bulk/modular_probe.py`
- `oph_fpe/bulk/modular_response_kernel.py`
- `oph_fpe/bulk/record_to_h3.py`
- `oph_fpe/bulk/h3_refit.py`
- `oph_fpe/bulk/h3_response_fit.py`
- `oph_fpe/bulk/h3_chart.py`
- `oph_fpe/cosmology/cmb_compare.py`
- `oph_fpe/cosmology/comparable_data.py`
- `oph_fpe/cosmology/cmb_transfer.py`
- `oph_fpe/cosmology/camb_adapter.py`
- `oph_fpe/cosmology/oph_cmb_adapter.py`
- `oph_fpe/cosmology/boltzmann_inputs.py`
- `oph_fpe/cosmology/ba_parent.py`
- `oph_fpe/cosmology/anomaly_fluid.py`
- `oph_fpe/viz/run_viewer.py`
- `oph_fpe/cli.py`

### Tests

- `tests/test_bw_array.py`
- `tests/test_observer_reconstruction.py`
- `tests/test_cmb_compare.py`
- `tests/test_comparable_data.py`
- `tests/test_cmb_transfer.py`
- `tests/test_camb_adapter.py`
- `tests/test_boltzmann_inputs.py`
- `tests/test_oph_cmb_adapter.py`
- `tests/test_ba_parent.py`
- `tests/test_anomaly_fluid.py`

### Configs

- `configs/e4_shared_observer_bulk_4k.yml`
- `configs/e4_shared_observer_bulk_64k_local_tokens_anchor0.yml`
- `configs/e4_shared_observer_bulk_256k_local_tokens_anchor0.yml`
- `configs/e1_s3_state_modular_screen_1m.yml`
- `configs/e2_kms_freezeout_cl_screen_64k.yml`
- `configs/e2_kms_freezeout_cl_screen_256k.yml`
- `configs/e3_cosmo_proxy_screen_64k.yml`
- `configs/e3_cosmo_proxy_screen_256k.yml`

### Current reports

- `runs/comparable_data_current_bulk_cmb_20260606/comparable_data_snapshot.md`
- `runs/comparable_data_current_bulk_cmb_20260606/comparable_data_snapshot.json`
- `runs/comparable_data_current_bulk_cmb_20260606/comparable_data_rows.csv`

Key run report files from:

- best 256k diagnostic,
- 64k direct transition-response variant,
- 64k H3 seed-ensemble robustness.

### Reproduction metadata

- `pyproject.toml`
- `REPRODUCTION.md`

## Commands Already Run

Full tests:

```bash
python3 -m pytest -q
```

Result:

```text
150 passed in 25.17s
```

Best 256k run:

```bash
python3 -m oph_fpe.cli run-bw-array \
  --config configs/e4_shared_observer_bulk_256k_local_tokens_anchor0.yml \
  --out-dir runs/bulk_blind_moment_features_256k_anchor0_20260606
```

CMB comparison:

```bash
python3 -m oph_fpe.cli cmb-lite-compare \
  --run-dir runs/bulk_blind_moment_features_256k_anchor0_20260606/e4_shared_observer_bulk_256k_local_tokens_multishuffle_seed20260751_seed20260752_373e0a48 \
  --benchmark data/measurements/planck2018/COM_PowerSpect_CMB-TT-binned_R3.01.txt
```

Comparable package:

```bash
python3 -m oph_fpe.cli comparable-data \
  --run-dir \
    runs/camb_lcdm_baseline_20260606 \
    runs/cmb_adapter_smoke_64k_20260606 \
    runs/cmb_adapter_smoke_64k_256k_20260606 \
    runs/oph_boltzmann_inputs_cmb_adapter_64k_256k_20260606 \
    runs/bulk_blind_moment_features_4k_20260606 \
    runs/bulk_blind_moment_features_64k_anchor0_20260606 \
    runs/bw_state_mode_variants_64k_20260606 \
    runs/bulk_blind_moment_features_256k_anchor0_20260606 \
  --include runs/cmb_transfer_adapter_smoke_64k_to_256k_20260606 \
  --out runs/comparable_data_current_bulk_cmb_20260606
```

Viewer:

```bash
python3 -m oph_fpe.cli run-viewer \
  --run-dir runs/bulk_blind_moment_features_256k_anchor0_20260606/e4_shared_observer_bulk_256k_local_tokens_multishuffle_seed20260751_seed20260752_373e0a48 \
  --out runs/bulk_blind_moment_features_256k_anchor0_20260606/e4_shared_observer_bulk_256k_local_tokens_multishuffle_seed20260751_seed20260752_373e0a48/oph_viewer.html \
  --max-screen-points 6000
```

## Current Hypothesis

The simulator’s current issue is not merely scale. At 256k:

- BW direct sanity improves and passes,
- screen C_l normalized-axis shape improves,
- but blind neutral reconstruction remains above 3D,
- object population remains near shuffled controls,
- H3 modular-response wrong-scale controls still win.

Therefore the main missing piece is likely one of:

1. wrong finite surrogate for the state-derived modular operator,
2. wrong observable family for modular-response-to-H3 fitting,
3. wrong observer-object extraction for populated H3,
4. wrong interpretation of what the finite simulator should prove beyond chart-level H3.

Please diagnose which of these is most likely, and recommend the next concrete code changes.

