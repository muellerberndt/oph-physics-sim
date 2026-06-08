# OPH-FPE Pro Handoff: Cache, H3 Bulk, CMB, and Code-Audit Request

Date: 2026-06-08

This package is for a Pro model that does not have filesystem access. The ZIP alongside this file bundles the relevant source files, configs, tests, paper sources, CMB writeup, current run reports, and recent patches.

## Executive Summary

We are likely close to a clean diagnostic pipeline, but not close enough to claim physical 3D bulk emergence or CMB prediction.

Current honest status:

- Chart-level OPH Lorentz/H3 route passes in the simulator: `Conf+(S2) -> SO+(3,1) -> H3 chart`, with chart dimension exactly `3`.
- Minimal cap-profile-to-H3 reconstruction now passes strongly.
- Some observer-object and defect support-profile H3 precursors pass controls.
- Full populated 3D bulk is still not established.
- Endogenous state-derived modular generator still fails controls.
- CMB comparison is currently a Planck-lite screen-shape diagnostic, not a physical CMB prediction.
- Particle/matter emergence is not established; current S3 worldlines are screen/collar defect precursors only.

The user expects 3D to emerge mechanically from the spherical screen. Our code now separates:

1. Paper-side chart/Lorentz theorem receipt.
2. Finite state-derived modular/BW receipt.
3. Observer-object population of the H3 chart.
4. Neutral reconstruction/dimension estimates.
5. CMB/galaxy measurement-facing outputs.

This separation may be too conservative, too weak, or still misaligned with the papers. Please audit.

## Latest Numerical Status

From `runs/comparable_data_stage129_audit_20260607/comparable_data_snapshot.*`:

### Lorentz / H3 chart lane

- `chart_level_conformal_lorentz_count`: `3/3`
- `bw_automorphism_sanity_count`: `3/3`
- `support_visible_lorentz_3p1_count`: `3/3`
- `paper_theorem_3d_bulk_chart_count`: `3/3`
- `mean_paper_theorem_h3_spatial_dimension`: `3.0`
- `paper_theorem_object_populated_chart_precursor_count`: `2/3`
- `paper_theorem_neutral_populated_bulk_count`: `0/3`
- `object_bulk_population_count`: `0/3`
- `bulk_3d_established_count`: `0/3`

Interpretation: the simulator verifies the chart-level conformal/Lorentz branch, but does not prove that record families populate an observer-facing 3D bulk.

### State-derived BW / modular matrix lane

- `receipt_count`: `2/3`
- `selected_2pi_count`: `2/3`
- `endogenous_modular_generator_count`: `0/3`
- `endogenous_correct_beats_controls_count`: `0/3`
- `best_control_counts`: `wrong_1x_normalization: 3`
- one endogenous audit says `no_flow_best`.

Interpretation: strict state-derived cap/collar `rho_C -> K_a=-log(rho_C+aI)` still does not beat wrong/no-flow controls. This is a primary theorem-alignment blocker.

### H3 modular-response controls

From the comparable snapshot:

- `receipt_count`: `0/3`
- `control_separation_receipt_count`: `2/3`
- `signal_gate_count`: `2/3`
- `geometry_gate_count`: `3/3`
- `aggregate_wrong_scale_gate_count`: `3/3`
- `material_feature_gate_count`: `0/3`
- `mean_h3_rmse`: `0.9784`
- `mean_h3_explained_variance`: `0.0421`
- `mean_s2_boundary_rmse`: `1.0415`
- `mean_shuffled_response_rmse`: `1.0123`
- wrong-scale red flags remain.

From latest cached 64k refit ensemble `runs/stage140_h3_refit_ensemble_64k_20260608/h3_refit_ensemble_report.json`:

- `seed_count`: `8`
- `control_separation_receipt_count`: `8/8`
- `candidate_receipt_count`: `4/8`
- `receipt_fraction`: `0.5`
- `H3_RESPONSE_ENSEMBLE_RECEIPT`: `false`
- `mean_heldout_normalized_rmse`: `0.9330`
- `mean_heldout_explained_variance`: `0.1295`
- `median_heldout_explained_variance`: `0.1307`
- `p75_material_wrong_scale_win_fraction`: `0.0586`

Interpretation: refit is better than earlier runs, especially control separation, but seed-robust H3 response is still not established under the current receipt threshold.

### Minimal cap-profile H3 sanity

From `runs/stage139_caps_to_h3_minimal_20260608/caps_to_h3_minimal_report.json`:

- `S2_CAP_PROFILE_TO_H3_RECEIPT`: `true`
- `median_reconstruction_mse`: `8.81e-34`
- `median_shuffled_profile_mse`: `0.1393`
- `median_s2_boundary_profile_mse`: `0.3386`
- H3 beats shuffled and S2-boundary controls.

Interpretation: the pure cap-profile geometry mechanism works. The open question is whether actual observer record/object profiles are being extracted in the right observable space.

### Observer-object H3 population

- `object_chart_receipt_count`: `3/3`
- `object_chart_median_receipt_count`: `3/3`
- `modular_response_h3_control_separation_receipt_count`: `2/3`
- `localized_object_precursor_receipt_count`: `2/3`
- `bulk_population_receipt_count`: `0/3`
- `mean_object_count`: `1421.33`
- `mean_localized_object_count`: `457.67`
- `mean_localized_not_boundary_object_count`: `385`
- `mean_h3_compactness`: `0.4428`
- `mean_s2_boundary_compactness`: `0.4576`
- H3 beats robust shuffled control in `3/3`.

Interpretation: object extraction is no longer obviously dead, but still not good enough for a populated-bulk receipt. H3 compactness is only slightly better than S2-boundary compactness in the aggregate, which is suspicious.

### Neutral reconstruction

- `candidate_3d_window_count`: `0/3`
- `bulk_3d_established_count`: `0/3`
- `radial_depth_used_count`: `0`
- `mean_primary_dimension`: `2.3736`
- `mean_correlation_dimension`: `3.0948`
- `mean_local_mle_dimension`: `5.1362`
- blind low-rank selected dimensions are far above 3.

Interpretation: neutral observer-record reconstruction is not yet a proof of 3D. We should not use fractional/high dimensions as a contradiction of the papers yet, because the finite observable may still be wrong. But the current reconstruction does not give the desired bulk.

### CMB / measurement lanes

Planck-lite TT screen-shape diagnostic:

- best field: `record_signature_smooth_k32`
- mean best shape correlation: `0.5403`
- mean best normalized RMSE: `0.8415`
- physical CMB prediction: `false`
- real ell-overlap usable count: `0`

OPH-CMB anomaly-stress adapter:

- run count: `3`
- Boltzmann-ready count: `0`
- physical-prediction-ready count: `0`
- mean weighted collar repair-defect `R`: `0.8562`
- missing gate count: `15`

Static galaxy measurement fit:

- no external RAR/SPARC/BTFR rows fit yet.

Interpretation: we have measurement-facing diagnostics, not physical predictions. A useful near-term path may be static galaxy RAR/BTFR using external SPARC-like data, while CMB remains gated on a better OPH-to-Boltzmann kernel.

## Recent Cache Work

Because runs are slow, we added deterministic geometry caching:

- `oph_fpe/cache/geometry_cache.py`
  - persistent content-addressed cache for cap transport maps.
  - key includes screen point digest plus full cap geometry plus `s` plus `k`.
  - cache stores only `indices` and `weights`, not stochastic physics state.
  - report includes memory hits, disk hits/misses/writes.

- `oph_fpe/scale/bw_array.py`
  - reads `cache.geometry.enabled` and `cache.geometry.cache_dir`.
  - passes `GeometryCache` into H3 modular-response kernel.

- `oph_fpe/bulk/modular_response_kernel.py`
  - active `collar_operator_transition` mode now uses `GeometryCache.transport_support`.

- `.gitignore`
  - ignores `.oph_fpe_cache/`.

Smoke result:

```text
first  pass: disk_misses=6, disk_writes=6, disk_hits=0
second pass: disk_misses=0, disk_writes=0, disk_hits=6
```

This should reduce repeat cost when rerunning the same screen/cap/time grid. It does not create evidence.

Also changed active 64k/256k configs to preserve `observer_views.jsonl` and `observer_objects.jsonl`; older compact runs disabled JSONL and therefore could not be post-processed for object-chart recomputes.

## Key Suspected Mistakes / Audit Targets

Please audit these in order.

### 1. Are we still confusing boundary observers with bulk points?

Pro previously warned that the correct path is:

```text
S2 caps -> cap normals -> H3 chart -> cap-response profiles/objects -> H3 object localization -> defect worldlines
```

and not:

```text
boundary observer similarity -> dimension estimator
```

We added a minimal cap-profile H3 receipt and it passes. But actual record/object extraction may still bake in S2 locality through support overlap, packet incidence, or observer support definitions.

Please check:

- `oph_fpe/bulk/cap_profile_geometry.py`
- `oph_fpe/bulk/record_to_h3.py`
- `oph_fpe/observers/objects.py`
- `oph_fpe/scale/bw_array.py`

Question: are `record_family_modular_response_mixture`, object support profiles, and compactness metrics actually testing cap-response/H3 localization, or are they still boundary-local objects with H3 post-fit labels?

### 2. Is `collar_operator_transition` a valid finite surrogate?

The active 64k config uses:

```yaml
h3_modular_response:
  observable_mode: collar_operator_transition
```

This mode transports visible packet support by `lambda_C(s)` and reads class/entropy/sector transition features. It is theorem-aligned in the weak sense that it uses cap flow on support-visible packets, but it is not a true finite state-derived modular operator from `rho_C`.

Question: should this mode be considered an acceptable intermediate diagnostic, or is it too kinematic and therefore likely to give false H3 positives?

### 3. Why does strict state-derived `rho_C/K_a` still fail?

State-derived BW lane says:

- endogenous-generator receipts: `0`
- no-flow/wrong-scale controls beat or match.

Please audit:

- `oph_fpe/bulk/modular_probe.py`
- `oph_fpe/bulk/collar_state.py`
- `oph_fpe/bulk/markov_collar.py`
- `oph_fpe/scale/bw_array.py` state BW call sites

Questions:

- Is `rho_C` built from the correct support-visible cap/collar algebra?
- Are observables too diagonal, making modular transport trivial?
- Is `K_a=-log(rho+aI)` being regularized or scaled in a way that destroys the `2*pi` signature?
- Should the finite surrogate use transition/co-occurrence blocks differently?

### 4. Is the H3 response fit overfitting controls or using the wrong feature space?

Latest refit improved:

- control separation `8/8`
- median EV `0.1307`
- p75 material wrong-scale fraction `0.0586`

But candidate receipt only `4/8`.

Please audit:

- `oph_fpe/bulk/h3_response_fit.py`
- `oph_fpe/bulk/h3_refit.py`
- feature selection in configs.

Questions:

- Are current thresholds too strict or too lenient?
- Is `record_family` exclusion correct, or did it remove useful observer-object information?
- Are wrong-scale controls fair?
- Is `profile_mode: static_halfspace` the right H3 profile, or should modular time enter the profile for the theorem branch?
- Is the H3 assignment/fitting objective geometrically correct?

### 5. Are object compactness and boundary dominance metrics correct?

The object-chart lane beats shuffled controls but not enough for populated bulk. H3 compactness and S2 compactness are close.

Please audit:

- `observer_chart_object_population_report`
- object compactness definitions
- shuffled object controls
- boundary dominance gate

Questions:

- Should H3 compactness be compared to S2-boundary compactness this way?
- Are localized non-boundary objects counted correctly?
- Is “non-boundary” meaningful inside an H3 chart built from cap normals?

### 6. Are neutral dimension estimators being used too early?

Neutral reconstruction currently reports fractional/high dimensions. We are not using radial depth, which is good. But it may still be the wrong object to measure before the H3 object lane passes.

Please audit:

- `oph_fpe/bulk/observer_reconstruction.py`
- `oph_fpe/bulk/dimensions.py`

Questions:

- Should neutral reconstruction wait for a passing H3 cap-response object lane?
- Are planted 2D/3D/4D controls sufficient?
- Are blind low-rank selections invalid because rank selection biases dimension?

### 7. What is the shortest honest measurement-facing path?

Current CMB data:

- Planck-lite proxy corr `0.5403`
- not physical ell-space
- not CAMB/CLASS

Current galaxy data:

- static proxy scaffolding exists, no external rows fit yet.

Please audit:

- `oph_fpe/cosmology/cmb_compare.py`
- `oph_fpe/cosmology/cmb_transfer.py`
- `oph_fpe/cosmology/oph_cmb_adapter.py`
- `oph_fpe/cosmology/boltzmann_inputs.py`
- `oph_fpe/cosmology/camb_adapter.py`
- `oph_fpe/cosmology/galaxy_static.py`

Questions:

- Is the fastest honest comparable output static galaxy RAR/BTFR rather than CMB?
- What exact OPH quantities must be emitted before a CAMB/CLASS anomaly module is meaningful?
- Can the screen-level `C_l` proxy be made into a transfer-stable diagnostic without claiming physical CMB?

### 8. Cache safety

Please audit the persistent cache:

- `oph_fpe/cache/geometry_cache.py`
- `oph_fpe/bulk/modular_response_kernel.py`
- `oph_fpe/scale/bw_array.py`

Questions:

- Is the cache key sufficient to prevent stale/incorrect geometry reuse?
- Are cap IDs safe now that cap parameters are included in the key?
- Should we also persist harmonic bases for `angular_power.py`, or is memory too risky?
- Should cache reports be written into every run manifest?

## Current Recommended Next Runs

Do not scale to 1M until the 64k/256k object/H3 lane is stable.

First rerun 64k with cache and JSONL payloads:

```bash
python3 -m oph_fpe.cli run-bw-sweep \
  --configs configs/e4_shared_observer_bulk_64k_local_tokens_feature_select.yml \
  --seeds 20261045,20261046,20261047,20261048 \
  --workers 4 \
  --inner-jobs 1 \
  --out-dir runs/stage141_cached_feature_select_64k_20260608
```

Then refit cached H3 kernels across more seeds:

```bash
python3 -m oph_fpe.cli h3-refit-ensemble \
  --run-dir <one-run-dir-with-modular_response_kernel_cache.json> \
  --seeds 20261045,20261046,20261047,20261048,20261049,20261050,20261051,20261052 \
  --candidate-count 8192 \
  --candidate-radius 2.0 \
  --softness 0.25 \
  --pass-ratio 1.0 \
  --min-observers 32 \
  --min-features 16 \
  --fit-mode joint_global \
  --heldout-fraction 0.25 \
  --anchor-weight 0.05 \
  --max-iterations 6 \
  --feature-selection class_distribution_and_change \
  --exclude-observables record_family \
  --min-feature-std 0.01 \
  --max-fit-features 1024 \
  --max-features-per-cap-time-observable 8 \
  --candidate-mode fibonacci_ball \
  --refine-steps 2 \
  --refine-max-rows 32 \
  --refine-max-nfev 32 \
  --out <out>/h3_refit_ensemble_report.json
```

After that, recompute object-chart from cached H3 fit and JSONL payloads. If this command does not exist yet, implement it before another large run.

## What Is Included In The ZIP

The ZIP should contain:

- this handoff markdown
- recent patch diff
- key run reports
- relevant OPH-FPE source files
- relevant configs
- relevant tests
- core paper TeX sources:
  - `recovering_relativity_and_standard_model_structure_from_observer_overlap_consistency_compact.tex`
  - `reality_as_consensus_protocol.tex`
  - `screen_microphysics_and_observer_synchronization.tex`
- OPH CMB writeup if present:
  - `omega-simulation/docs/cmb.md`

## Bottom Line For Pro

Please diagnose whether our current implementation is failing because:

1. The code still measures the wrong object.
2. The object extraction is leaking S2 boundary locality.
3. The finite state-derived modular operator is incorrectly constructed.
4. The H3 fitting/control logic has a bug.
5. The papers imply only chart-level Lorentz kinematics, while populated observer-object bulk requires an additional finite mechanism we have not implemented.
6. The CMB path should be delayed in favor of static galaxy/RAR/BTFR measurement fits.

The user’s expectation is that 3D should emerge clearly. If that expectation is mathematically correct under the paper hypotheses, please identify the exact missing finite mechanism or code mistake preventing the simulator from showing it.
