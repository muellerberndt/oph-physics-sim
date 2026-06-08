# OPH-FPE Pro Handoff: Comprehensive Bulk/CMB Audit

Date: 2026-06-07

This package is for GPT Pro to audit the OPH physics simulator implementation, diagnose the remaining blockers, and recommend the fastest honest path to measurement-facing outputs and a populated 3D/H3 bulk receipt.

## Executive Status

The simulator now has strong intermediate receipts, but it still does **not** establish a populated 3D bulk or a physical CMB prediction.

Current honest claim:

```text
Finite screen repair settles Phi.
Observer records commit.
The S2 cap/conformal/BW chart route verifies the paper-side Lorentz/H3 kinematic branch.
The simulator emits screen-level C_l proxy data and OPH-CMB anomaly-stress scaffolding.
Some H3/object/defect precursors beat controls.
```

Not yet allowed:

```text
3D bulk emergence is proven by the finite run.
The finite observer records populate a third-person 3D bulk.
The C_l output is a physical CMB prediction.
The defect worldlines are physical particles in a 3D bulk.
```

The most important tension is:

```text
The papers derive the 3+1D Lorentz/H3 chart from Conf+(S2) plus BW cap modular flow.
The finite simulator verifies that chart route, but the finite observer-object population and neutral reconstruction gates still fail.
```

So the current failure is not "the paper theorem is false." It is that the finite simulator has not yet closed the stronger receipt chain from state-derived modular transport to populated observer objects to physical CMB/Boltzmann outputs.

## Relevant Paper Interpretation

The simulator should distinguish these layers:

1. **Paper-side chart-level Lorentz receipt**
   - S2 is the support-visible cap/collar chart/regulator.
   - BW cap modular flow supplies `sigma_t = alpha_{lambda_C(2*pi*t)}`.
   - `Conf+(S2) ~= PSL(2,C) ~= SO+(3,1)`.
   - The spatial chart is H3 = SO+(3,1)/SO(3), dimension 3.

2. **Finite state-derived modular receipt**
   - Must build support-visible cap/collar states and a finite modular generator or valid finite surrogate.
   - Must beat wrong normalization/no-flow/shuffle controls.
   - Current strict endogenous generator route does not pass.

3. **Populated observer-object bulk**
   - Observer-visible record/object families must populate the H3 chart under controls.
   - Current object chart precursors pass, but populated bulk gate fails because strict H3 response receipt is unstable.

4. **CMB/measurement prediction**
   - Screen C_l proxy exists.
   - Physical CMB requires an OPH anomaly-stress/Boltzmann adapter, CDM-limit regression, and real likelihood/covariance handling.

## Latest Important Runs

Primary latest run:

```text
runs/stage129_strict_h3_object_bulk_64k_seed20261045_20260607/
  e4_shared_observer_bulk_64k_local_tokens_feature_select_seed20261045_c6b8a043/
```

Supporting diagnostic:

```text
runs/stage128_h3_refit_stage127_cached_success_settings_20260607.json
```

Comparable-data audit:

```text
runs/comparable_data_stage129_audit_20260607/
```

## Latest 64k Results

### Repair/BW

From the latest 64k run:

```text
final_phi = 0
BW geometric sanity median = 0.30717551398051113
BW primary median = 1.5535471863889554e-15
2pi selected = true
state BW correct beats controls = true
```

Geometric controls are strong:

```text
wrong_1x median = 1.2342551430156137
wrong_pi median = 1.1967680149995825
wrong_4pi median = 1.2382413343488547
shuffled_observables median = 1.306545504434685
no_modular_flow median = 1.2545054752290186
```

But the report also marks:

```text
ENDOGENOUS_MODULAR_GENERATOR_RECEIPT = false
```

because the strongest passing lane is currently a direct transition-automorphism/KMS transport surrogate, not a fully endogenous cap/collar `K_a = -log(rho_C + aI)` theorem surrogate.

### Paper Theorem Chart

Latest comparable snapshot:

```text
chart-level conformal Lorentz receipts = 3/3
BW automorphism sanity receipts = 3/3
support-visible 3+1D Lorentz receipts = 3/3
paper-theorem 3D bulk-chart receipts = 3/3
paper-theorem H3 spatial dimension mean = 3.0
```

This means the simulator verifies the chart-side theorem route:

```text
Conf+(S2) -> SO+(3,1) -> H3 spatial chart, dim(H3)=3
```

It does **not** mean the finite observer records have populated a third-person 3D bulk.

### H3 Modular Response

Latest embedded H3 fit in stage129:

```text
H3_RESPONSE_CONTROL_SEPARATION_RECEIPT = true
H3_RESPONSE_CANDIDATE_RECEIPT = false
heldout_normalized_rmse = 0.9477911856576535
heldout_explained_variance = 0.10169186838965916
assignment_unique_count = 144
signal_gate = true
geometry_gate = true
aggregate_wrong_scale_gate = true
material_feature_gate = false
material_wrong_scale_win_fraction = 0.0859375
material_two_pi_win_fraction = 0.9140625
wrong_scale_rmse: 1x=1.0306338952756051, pi=1.0147313226270793, 4pi=1.0518214802961314
```

This is close but fails the strict gate because current threshold is:

```text
max_material_wrong_fraction = 0.05
```

### Important H3 Refit Contrast

A post-run refit on the same saved kernel, using the cached-success settings, **does** pass strict H3:

```text
runs/stage128_h3_refit_stage127_cached_success_settings_20260607.json

H3_RESPONSE_CONTROL_SEPARATION_RECEIPT = true
H3_RESPONSE_CANDIDATE_RECEIPT = true
heldout_normalized_rmse = 0.9381777861690648
heldout_explained_variance = 0.11982244153891242
assignment_unique_count = 159
material_wrong_scale_win_fraction = 0.04296875
```

This means the raw response kernel can support a strict H3 receipt, but the end-to-end fit/config path is still brittle.

Recent config diagnosis:

```text
The experimental channel_mode=cap_observable_class and profile_mode=modular_time_delta failed badly:
  stage127 EV = -0.01781936028106923
  material_wrong_scale_win_fraction = 0.44921875

The static-halfspace time_observable_class model improved:
  stage129 EV = 0.10169186838965916
  material_wrong_scale_win_fraction = 0.0859375

The post-run cached-success refit passes:
  EV = 0.11982244153891242
  material_wrong_scale_win_fraction = 0.04296875
```

Pro audit request: determine whether this is a legitimate finite-fit instability, an implementation/config bug, or a scientifically invalid model-selection path.

### Observer Object / Populated Bulk

Latest stage129 object chart:

```text
observer_chart_object_h3_receipt = true
observer_chart_bulk_population_receipt = false
localized_h3_object_precursor_receipt = true
localized_h3_bulk_population_receipt = false
h3_not_boundary_dominated = true
h3_beats_shuffled_incidence = true
h3_beats_shuffled_incidence_robust = true
object_count = 2048
localized_object_count = 261
localized_not_boundary_object_count = 140
shuffled_localized_object_p90 = 257.0
bulk_population_gate_mode = localized_h3_subpopulation_vs_shuffled_with_boundary_leakage_audit
modular_response_h3_strict_receipt = false
```

Interpretation:

```text
Object chart and localized-object precursor pass.
Bulk population fails mainly because strict H3 response is false in the embedded run.
```

If the strict H3 receipt from the cached-success refit can be made stable in the end-to-end run, the object-population gate may become the next reachable receipt.

### Neutral Reconstruction

Latest comparable snapshot:

```text
neutral radial-depth used = false
neutral control gate passed = true
candidate 3D-window count = 0
bulk-established count = 0
mean primary dimension = 2.373635334
mean correlation dimension = 3.094767311
mean local-MLE dimension = 5.136210618
```

Latest stage129:

```text
neutral_correlation_dimension_estimate = 3.455333298752222
neutral_local_mle_dimension_estimate = 6.27124989119878
blind_low_rank_selected_correlation_dimension = 5.3431998217692795
blind_low_rank_selected_local_mle_dimension = 5.935277069875275
```

Interpretation:

```text
Neutral reconstruction is not stable and does not establish 3D.
The paper-side H3 chart dimension is 3, but finite record-distance estimators remain inconsistent.
```

Pro audit request: confirm whether a neutral third-person point-cloud dimension should be expected at this stage, or whether the paper only demands H3 chart covariance/object population first.

### CMB / Measurement-Facing Output

Latest stage129 Planck-lite screen proxy:

```text
best_field = record_signature_smooth_k32
best normalized-axis shape correlation = 0.5402980846597015
best positive normalized RMSE = 0.841473695199712
record_signature shape correlation = 0.4330391640093441
record_signature RMSE = 0.9013751063980459
real-ell overlap usable = false
physical_cmb_prediction = false
```

Older/better high-ell 64k diagnostic from stage111:

```text
normalized-axis best record_signature_smooth_k32 correlation ~= 0.7697
positive RMSE ~= 0.638
real-ell overlap raw record_signature correlation ~= 0.8148
real-ell overlap RMSE ~= 0.5797
```

Worsening to document:

```text
Latest stage129 normalized-axis CMB proxy is weaker than stage111.
Small 4k high-ell proxy was negative/anti-correlated.
No current CMB lane is physical ell-space likelihood-grade.
```

Pro audit request: advise whether to prioritize:

```text
1. improving screen C_l proxy repeatability,
2. implementing the OPH anomaly-stress Boltzmann adapter,
3. running CAMB/CLASS CDM-limit regression first,
4. or deferring CMB until populated H3 bulk passes.
```

### Defects / Particle-Like Output

Latest stage129:

```text
holonomy triangle_count = 12000
defect_triangle_count = 9947
defect_fraction = 0.8289166666666666
cluster_count = 24
defect_timeline_worldline_count = 9
defect_timeline_persistent_worldline_count = 9
defect_worldline_precursor_receipt = true
defect_fusion_conservation_proxy_pass = true
defect_scattering_reproducibility_proxy_pass = true
particle_like_count = 0
particle_matter_receipt = false
defect_h3_bulk_worldline_precursor_receipt = false
```

Interpretation:

```text
The screen/collar defect machinery is alive.
No physical particle emergence claim is allowed yet because populated H3 bulk and strict particle gates fail.
```

## What Improved

1. The paper-side chart receipt is now explicit and stable:

```text
S2 cap/conformal geometry + BW 2pi -> SO+(3,1) -> H3 dimension 3.
```

2. The 64k finite response kernel can pass a strict H3 receipt under post-run refit:

```text
stage128 H3 strict receipt = true
```

3. Object chart population has become nontrivial:

```text
object-chart receipt = true
localized object precursor = true in stage129
H3 compactness beats shuffled incidence robustly
```

4. Screen C_l proxies and OPH-CMB anomaly-stress scaffolding are emitted consistently.

5. Defect worldline/fusion/scattering proxy machinery is emitted.

## What Worsened or Remains Unstable

1. H3 strict receipt is not stable end-to-end.

```text
stage128 post-run refit passes strict H3.
stage129 embedded fit misses by material_wrong_scale_win_fraction 0.0859375 > 0.05.
```

2. The experimental H3 model variant worsened diagnostics:

```text
stage127 profile_mode=modular_time_delta:
  EV = -0.0178
  material wrong-scale wins = 0.4492
```

3. CMB screen proxy worsened relative to older stage111:

```text
stage111 best normalized-axis corr ~= 0.77
stage129 best normalized-axis corr ~= 0.54
```

4. Neutral reconstruction remains inconsistent:

```text
correlation dimensions can be near 3,
but local-MLE/blind dimensions are much higher,
and the gate correctly stays closed.
```

5. Endogenous modular generator remains blocked:

```text
strict endogenous cap/collar K_a lane does not beat wrong-normalization/no-flow controls.
```

## Main Questions For Pro

1. **Paper-alignment of finite H3 response**
   - Is the current H3 response model (`time_observable_class`, `static_halfspace`, transition deltas) a legitimate finite surrogate for support-visible BW/H3 response?
   - Or should the fit be replaced by a direct representation of `PSL(2,C)`/H3 geodesic flow from cap-pair modular generators?

2. **H3 strict threshold**
   - Is `max_material_wrong_fraction = 0.05` justified?
   - Stage129 has material wrong-scale win fraction `0.0859375`; stage128 refit has `0.04296875`.
   - Should the gate be binary at 0.05, seed-ensemble based, or confidence-interval based?

3. **Model selection**
   - The passing refit uses static halfspace/time-observable-class.
   - The failing experimental variant used cap-observable-class/modular-time-delta.
   - Is static halfspace physically acceptable, or is it overfitting?

4. **Feature selection**
   - `record_family` was excluded because it behaves like coarse bookkeeping and generated wrong-normalization aliases.
   - Is excluding `record_family` correct under the papers' support-visible geometric subnet distinction?
   - Are `checkpoint_class`, `stable_flag`, `s3_sector_class`, and `repair_load_bucket` legitimate observer-visible geometric subnet readouts?

5. **Object bulk gate**
   - The current gate uses `boundary_leakage_audit`, not a hard non-boundary veto, because S2 is the observer-facing angular chart.
   - Is that paper-accurate?
   - Should object population be judged by H3 localization vs shuffled incidence even when S2 boundary compactness is also strong?

6. **Neutral reconstruction**
   - Should a neutral third-person point cloud ever be expected to have dimension exactly 3 before populated H3 chart/object gates pass?
   - Or is neutral reconstruction a later diagnostic and not the right target for the theorem?

7. **Endogenous modular generator**
   - What finite construction should replace the current brittle `K_a = -log(rho_C+aI)` lane?
   - Should `rho_C` come from perturb/remeasure transition kernels, collar Markov recovery, cooccurrence matrices, or a more direct finite algebra representation?

8. **CMB path**
   - Given OPH's CMB writeup, should we stop comparing raw screen C_l to Planck TT and instead first implement a CDM-limit CAMB/CLASS regression and an OPH anomaly-stress adapter?
   - Which simulator quantities should map to `rho_A(a)`, `Gamma_rec`, and `B_A(k,a)`?

9. **Particle path**
   - Are current S3 holonomy defect clusters a valid matter/particle precursor?
   - What exact extra gate would make defect worldlines particle-like in the OPH paper sense?

10. **Scale/parallelism**
   - Current 64k single-seed runs are minutes, not hours, when controlled.
   - Should we spend compute on seed ensembles at 64k, or jump to 256k/1M?
   - My current recommendation: do not scale until strict H3 and object-population receipts are stable across seeds.

## Recommended Next Coding Steps

1. Make the H3 fit model configurable but default to the latest best candidate:

```yaml
channel_mode: time_observable_class
profile_mode: static_halfspace
exclude_observables: [record_family]
min_wrong_scale_feature_delta: 0.0
candidate_count: 8192
refine_steps: 2
```

2. Add a postprocess command:

```text
recompute-object-chart --run-dir ... --h3-report stage128_refit.json
```

This would let us recompute object-population gates from a saved strict H3 refit without rerunning the whole 64k simulation.

3. Add an H3 seed-ensemble runner over the cached kernel:

```text
h3-refit-ensemble --run-dir ... --seeds ...
```

Acceptance should be fraction-based, not a single lucky candidate sample.

4. Add a compact H3 audit report that shows:

```text
2pi fit wins by observable/feature_type/time/cap
wrong-scale aliases by observable
whether aliases are material or tiny residual differences
```

5. Implement the CDM-limit CAMB/CLASS regression before any new "CMB prediction" language.

6. Keep the screen C_l proxy, but label it as:

```text
measurement-facing screen diagnostic, not physical CMB
```

7. For particles, add a controlled S3 defect assay with planted inverse/fusion/scattering tests before claiming particle-like behavior.

## Reproduction Commands

Focused tests:

```bash
cd /Users/muellerberndt/Projects/oph-meta/oph-physics-sim
python3 -m pytest -q tests/test_modular_response_h3.py tests/test_bw_array.py tests/test_conformal_spatial_chart.py
```

Latest 64k run:

```bash
python3 -m oph_fpe.cli run-bw-sweep \
  --configs configs/e4_shared_observer_bulk_64k_local_tokens_feature_select.yml \
  --seeds 20261045 \
  --workers 1 --inner-jobs 1 \
  --out-dir runs/stage129_strict_h3_object_bulk_64k_seed20261045_20260607
```

Post-run H3 refit that passed strict receipt:

```bash
python3 -m oph_fpe.cli h3-refit \
  --run-dir runs/stage127_strict_h3_64k_seed20261045_20260607/e4_shared_observer_bulk_64k_local_tokens_feature_select_seed20261045_542a0546 \
  --out runs/stage128_h3_refit_stage127_cached_success_settings_20260607.json \
  --candidate-count 8192 --candidate-radius 2.0 --softness 0.25 --seed 20261045 \
  --pass-ratio 1.0 --min-observers 32 --min-features 16 --fit-mode joint_global \
  --heldout-fraction 0.25 --anchor-weight 0.05 --max-iterations 6 \
  --feature-selection class_distribution_and_change --min-feature-std 0.01 \
  --max-fit-features 1024 --max-features-per-cap-time-observable 8 \
  --exclude-observables record_family --candidate-mode fibonacci_ball \
  --refine-steps 2 --refine-max-rows 32 --refine-max-nfev 32
```

Comparable-data audit:

```bash
python3 -m oph_fpe.cli comparable-data \
  --run-dir runs/stage129_strict_h3_object_bulk_64k_seed20261045_20260607 \
  --include runs/stage127_strict_h3_64k_seed20261045_20260607 runs/stage121_full_h3_cmb_evidence_4k_20260607 \
  --out runs/comparable_data_stage129_audit_20260607
```

## Package Contents

The ZIP next to this handoff should include:

```text
source files:
  oph_fpe/scale/bw_array.py
  oph_fpe/bulk/h3_response_fit.py
  oph_fpe/bulk/h3_refit.py
  oph_fpe/bulk/record_to_h3.py
  oph_fpe/bulk/modular_probe.py
  oph_fpe/bulk/modular_response_kernel.py
  oph_fpe/bulk/conformal_spatial_chart.py
  oph_fpe/bulk/cap_geometry.py
  oph_fpe/bulk/collar_state.py
  oph_fpe/bulk/observer_reconstruction.py
  oph_fpe/cosmology/comparable_data.py
  oph_fpe/cosmology/cmb_compare.py
  oph_fpe/cosmology/angular_power.py
  oph_fpe/cosmology/oph_cmb_adapter.py
  oph_fpe/cosmology/camb_adapter.py
  oph_fpe/defects/controlled_assay.py
  oph_fpe/claims.py
  oph_fpe/cli.py

configs:
  configs/e4_shared_observer_bulk_64k_local_tokens_feature_select.yml
  configs/e4_shared_observer_bulk_4k_collar_operator.yml
  configs/e4_shared_observer_bulk_256k_cmb256.yml

tests:
  tests/test_modular_response_h3.py
  tests/test_bw_array.py
  tests/test_conformal_spatial_chart.py
  tests/test_record_to_h3.py
  tests/test_cmb_compare.py
  tests/test_camb_adapter.py

paper context:
  recovering_relativity_and_standard_model_structure_from_observer_overlap_consistency_compact.tex
  reality_as_consensus_protocol.tex
  screen_microphysics_and_observer_synchronization.tex
  omega-simulation/docs/cmb.md

run receipts:
  latest stage129 JSON reports
  stage128 strict H3 refit JSON
  comparable-data snapshot JSON/MD/CSV
```

## Final Ask To Pro

Please audit whether the finite simulator is implementing the OPH paper mechanisms correctly enough to pursue a populated H3 bulk receipt, and identify the shortest reliable path to:

```text
1. stable strict H3 modular-response receipt,
2. populated observer-object H3 chart receipt,
3. particle/defect worldlines in that chart,
4. measurement-facing CMB output that is not just a screen proxy,
5. eventual physical CMB/Boltzmann comparison.
```

If a current gate is conceptually wrong, please say exactly which gate should replace it and which source file should change.
