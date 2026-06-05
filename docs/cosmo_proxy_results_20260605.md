# OPH-FPE Cosmo Proxy Results - 2026-06-05

## Scope

This note records the first June 5 measurement-facing OPH-FPE proxy sweep.

Claim boundary:

```text
These are gated screen-level freezeout C_l proxies and shape-only comparison diagnostics.
They are not physical CMB predictions, not CAMB/CLASS inputs, not P(k), and not evidence
that a populated 3D bulk has emerged.
```

Benchmark used:

```text
runs/benchmarks/COM_PowerSpect_CMB-TT-binned_R3.01.txt
source: https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/COM_PowerSpect_CMB-TT-binned_R3.01.txt
```

## Runs

64k four-seed compact proxy sweep:

```text
runs/e3_cosmo_proxy_64k_sweep_20260605/
seeds: 20260651, 20260652, 20260653, 20260654
workers: 2
inner_jobs: 4
elapsed_seconds: 54.17
```

256k one-seed compact follow-up:

```text
runs/e3_cosmo_proxy_256k_followup_20260605/
seed: 20260655
workers: 1
inner_jobs: 8
elapsed_seconds: 84.89
```

All five runs:

```text
final_phi = 0
2pi selected by KMS collar transport response
theorem-core receipts passed:
  Lyapunov descent
  exact repair equals projection
  finite SM quotient gate
freezeout-screen C_l gate allowed
```

## 64k Ensemble Findings

The strongest stable screen-spectrum field is `record_signature`:

```text
peak_ell_mode: 9
peak_ell_values: [9, 9, 9, 9]
mean_pairwise_shape_correlation: 0.999896
median_min_relative_l2_delta_to_controls: 0.997209
median_max_shape_correlation_to_controls: -0.259425
mean_total_abs_D_ell_2_plus: 7.498459
```

This is interesting because it is highly seed-stable and strongly separated from shuffled/random controls.

Other fields:

```text
cumulative_repair_load:
  peak_ell_mode: 46
  mean_pairwise_shape_correlation: 0.979020
  median_min_relative_l2_delta_to_controls: 0.701281

s3_class_density:
  peak_ell_mode: 48
  mean_pairwise_shape_correlation: 0.937719
  median_min_relative_l2_delta_to_controls: 0.479500

stable_count:
  peak_ell_mode: 48
  mean_pairwise_shape_correlation: 0.931597
  median_min_relative_l2_delta_to_controls: 0.212106
```

`repair_load` and `local_mismatch_density` are zero after settlement in this compact proxy run and are not useful spectra.

## Planck-Lite Shape Comparison

The comparison normalizes multipole axes to `[0,1]` and least-squares rescales amplitudes. It is a shape-only diagnostic, not a likelihood.

64k mean by field:

```text
record_signature:
  mean_shape_correlation:  0.372170
  mean_normalized_rmse:    0.928163
  mean_peak_fraction_delta: 0.079783

s3_class_density:
  mean_shape_correlation: -0.776323
  mean_normalized_rmse:    0.629924
  mean_peak_fraction_delta: 0.911304

stable_count:
  mean_shape_correlation: -0.761626
  mean_normalized_rmse:    0.647839
  mean_peak_fraction_delta: 0.873261

cumulative_repair_load:
  mean_shape_correlation: -0.671350
  mean_normalized_rmse:    0.741109
  mean_peak_fraction_delta: 0.895000
```

Interpretation:

```text
record_signature is the only field with positive Planck-lite shape correlation, but the fit is weak.
s3_class_density and stable_count get lower normalized RMSE only by anticorrelated/rescaled shape behavior.
None of these should be described as a Planck match.
```

## 256k Follow-Up

The 256k run keeps the same `record_signature` peak:

```text
record_signature:
  peak_ell: 9
  low_ell_power_2_10: 0.585824
  total_abs_D_ell_2_plus: 7.186766
  Planck-lite shape_correlation: 0.377013
  Planck-lite normalized_rmse: 0.926208
  Planck-lite peak_fraction_delta: 0.079783
```

This is a useful refinement sanity check: the stable `ell=9` record-signature structure survives from 64k to 256k, but it still does not become a physical CMB-like spectrum.

Other 256k fields:

```text
cumulative_repair_load:
  peak_ell: 48
  Planck-lite shape_correlation: -0.660639
  normalized_rmse: 0.750704

s3_class_density:
  peak_ell: 43
  Planck-lite shape_correlation: -0.763224
  normalized_rmse: 0.646134

stable_count:
  peak_ell: 48
  Planck-lite shape_correlation: -0.768021
  normalized_rmse: 0.640424
```

## Result

The useful data so far is:

```text
observer-facing record signatures produce a stable, control-separated screen angular spectrum;
the dominant peak is ell=9 at both 64k and 256k;
the shape-only Planck comparison is weak and not physically meaningful as a CMB fit.
```

Recommended next runs:

```text
1. Run 256k x 4 seeds to confirm the ell=9 record-signature stability at scale.
2. Add geometry/harmonic caching before running broad 256k or 1M sweeps.
3. Add a targeted parameter sweep over boundary cap_count, endpoint_noise_probability,
   tangent_weight, and repair/cap coupling to see whether the screen-spectrum peak can move
   or broaden in controlled ways.
4. Do not run 1M until the parameter sweep shows a measurement-facing trend worth scaling.
```

## Parameter Variants

After the baseline seed and scale checks, four 64k parameter variants were run with seed `20260660`:

```text
runs/e3_cosmo_proxy_param_variants_20260605/

low_noise:   endpoint_noise_probability = 0.15
high_noise:  endpoint_noise_probability = 0.55
low_tangent: tangent_weight = 0.05
more_caps:   boundary_program.cap_count = 64
```

All variants settled and passed the same finite BW/theorem-core gates.

`record_signature` comparison:

```text
variant       peak_ell   total_abs_D_ell   Planck-lite corr   normalized_rmse
low_noise     9          8.3342            0.3787             0.9255
high_noise    9          6.0203            0.3424             0.9396
low_tangent   9          8.4938            0.1344             0.9909
more_caps     13         8.3641            0.4263             0.9046
```

The useful variant result is:

```text
Increasing the boundary cap count from 32 to 64 moves the stable record-signature peak
from ell=9 to ell=13 and improves the weak Planck-lite shape correlation.
```

This is not a Planck match, but it is a real tunable screen-spectrum response.

## 256k More-Caps Follow-Up

The `more_caps` variant was repeated at 256k:

```text
runs/e3_cosmo_proxy_256k_more_caps_20260605/
seed: 20260661
boundary_program.cap_count: 64
elapsed_seconds: 87.08
```

The shifted record peak survived refinement:

```text
record_signature:
  peak_ell: 13
  total_abs_D_ell_2_plus: 8.0342
  Planck-lite shape_correlation: 0.435666
  Planck-lite normalized_rmse: 0.900109
```

Current interpretation:

```text
The simulator can produce stable, control-gated, observer-record screen spectra whose
dominant peak is tunable by boundary-program structure. This is the first measurement-facing
data product worth optimizing further. It is still a screen proxy and not a physical CMB result.
```

## Perturb-Resettle H3 Run CMB-Lite Check

After adding the cap/collar perturb-resettle H3 response, the dense 64k scale run also emitted a
screen-level freezeout spectrum:

```text
runs/e3_perturb_resettle_h3_64k_dense_20260605/e3_perturb_resettle_h3_screen_64k_dense_seed20260696
```

Screen proxy:

```text
record_signature peak_ell: 9
record_signature low_ell_power_2_10: 0.5890010803
record_signature control separation: 0.9978360653
```

Planck-lite shape comparison:

```text
benchmark: Planck2018_TT_binned
benchmark rows: 83
benchmark ell range: 47.711224 - 2499.024

record_signature:
  shape_correlation: 0.2916787921
  normalized_rmse: 0.9565163262
  peak_fraction_delta: 0.1609420323

stable_count:
  shape_correlation: -0.7823889391
  normalized_rmse: 0.6227901316
  peak_fraction_delta: 0.9276086990
```

Interpretation:

```text
The screen spectrum remains measurement-facing but not physically CMB-like.
The best RMSE field is anticorrelated and peaks at the wrong normalized location.
record_signature is positive-correlated but weak.
This should not be presented as a Planck match.
```

Updated next runs:

```text
1. Cap-count sweep at 64k: cap_count = 16, 24, 32, 48, 64, 96, 128.
2. For the best two cap-count values, run 256k x 4 seeds.
3. Add geometry and spherical-harmonic caching before any 1M run.
4. Only then test 1M for the best cap-count value.
```

## Dense H3 Perturb-Resettle CMB-Lite Ensemble

The dense 64k perturb-resettle H3 ensemble also emitted freezeout-screen `C_l` proxies:

```text
runs/e3_perturb_resettle_h3_64k_dense_ensemble_20260605
  seeds: 20260701, 20260702, 20260703, 20260704
  patch_count: 65536
  H3 candidate receipts: 4 / 4
```

Planck-lite shape comparison against the local binned TT benchmark:

```text
best-field counts:
  stable_count: 2
  s3_class_density: 2

mean best-field shape_correlation: -0.7739024335
mean best-field normalized_rmse: 0.6327142576
mean best-field peak_fraction_delta: 0.8942753657

record_signature mean shape_correlation: 0.2922865741
record_signature mean normalized_rmse: 0.9563262464
```

The 256k scale probe:

```text
runs/e3_perturb_resettle_h3_256k_probe_20260605/e3_perturb_resettle_h3_screen_256k_probe_1780630733
  patch_count: 262144
  H3 candidate receipt: true

best field: s3_class_density
  shape_correlation: -0.7652205419
  normalized_rmse: 0.6437682209
  peak_fraction_delta: 0.8609420323

record_signature:
  shape_correlation: 0.2953212453
  normalized_rmse: 0.9553980124
  peak_fraction_delta: 0.1609420323
```

Interpretation:

```text
The larger perturb-resettle runs strengthen the H3-support receipt but do not improve the
measurement-facing screen spectrum. The lowest-RMSE fields remain anticorrelated with the
Planck TT shape, and the positive record-signature correlation is weak. These numbers are
usable as diagnostic data, not as CMB predictions.
```

## Comparable Data Export

Added:

```text
oph_fpe/cosmology/comparable_data.py
python3 -m oph_fpe.cli comparable-data
```

The exporter writes:

```text
comparable_data_snapshot.json
comparable_data_rows.csv
comparable_data_snapshot.md
```

Current package:

```text
runs/comparable_data_snapshot_with_e4_20260605
```

Aggregated lanes:

```text
H3 modular-response controls:
  run_count: 6
  receipt_count: 5
  mean H3 RMSE: 0.9734522723
  mean H3 explained variance: 0.0523714364
  mean S2-boundary RMSE: 1.0121224876
  mean shuffled-response RMSE: 1.0008597212

Planck TT shape-lite:
  run_count: 6
  best-field counts:
    s3_class_density: 4
    stable_count: 2
  mean best-field shape correlation: -0.7811254155
  mean best-field normalized RMSE: 0.6230682222
  mean record-signature shape correlation: 0.2855160484
  mean record-signature normalized RMSE: 0.9582169758

Screen holonomy defect proxy:
  run_count: 6
  mean defect fraction: 0.834725
  mean cluster count: 58.3333333333
```

Interpretation:

```text
This is the answer to "what comparable data do we have yet?": H3 residuals versus controls,
screen C_l shape-lite versus Planck TT, and S3 screen-holonomy defect statistics. These are
diagnostic, measurement-facing values. They are not physical CMB, P(k), BAO, SPARC, or particle
predictions.
```

The combined observer-facing route run:

```text
runs/e4_shared_observer_bulk_4k_20260605/e4_shared_observer_bulk_4k_1780638364
```

Planck-lite values:

```text
best field: s3_class_density
  shape_correlation: -0.8259222171
  normalized_rmse: 0.5637840822
  peak_fraction_delta: 0.8609420323

record_signature:
  shape_correlation: 0.2486287483
  normalized_rmse: 0.9685988569
  peak_fraction_delta: 0.1609420323
```

Again, this remains a negative/diagnostic Planck-lite comparison, not a CMB prediction.
