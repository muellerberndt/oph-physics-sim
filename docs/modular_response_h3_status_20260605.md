# Modular-Response H3 Status - 2026-06-05

## Scope

This note records the first OPH-FPE run where observer-facing modular response is assembled as a
finite tensor and fitted into the canonical H3 chart.

Claim boundary:

```text
This is a support-visible modular-response bulk-candidate diagnostic.
It is not a 3D bulk emergence receipt, not a particle receipt, and not a CMB prediction.
```

## Implementation

New code paths:

```text
oph_fpe/cache/geometry_cache.py
oph_fpe/bulk/modular_response_kernel.py
oph_fpe/bulk/h3_response_fit.py
configs/e3_modular_response_h3_screen_4k.yml
```

The new path builds:

```text
screen points + observer views + cap family + times + observer-visible fields
-> modular-response kernel R[observer, cap, time, field]
-> canonical H3 fit
-> S2-boundary, shuffled-response, and no-modular-flow controls
```

The geometry cache reuses the KD tree and cap transport maps. This is a speed/consistency helper
only; it carries no physics claim.

## Smoke Run

Run command:

```bash
OPH_FPE_CPUS=9 python3 -m oph_fpe.cli run-bw-sweep \
  --configs configs/e3_modular_response_h3_screen_4k.yml \
  --seeds 20260670 \
  --workers 1 \
  --inner-jobs 8 \
  --out-dir runs/e3_modular_response_h3_smoke_20260605
```

Run directory:

```text
runs/e3_modular_response_h3_smoke_20260605/e3_modular_response_h3_screen_4k_seed20260670_0fd0348b
```

Core run status:

```text
patch_count: 4096
final Phi: 0
BW median: 0.2117404523
state-derived primary BW median: 2.2774474986e-15
support-visible Lorentz/BW receipt: true
modular_response_h3_candidate_receipt: false
bulk_3d_established: false
```

Kernel summary:

```text
observer_count: 48
feature_count: 256
cap_count: 32
time_count: 2
fields: record_signature, stable_count, cumulative_repair_load, s3_class_density
response mean: 0.4856991962
response std: 0.2437299732
mean row std: 0.2325758573
mean column std: 0.2376735359
geometry cache transport maps: 64
```

H3 fit and controls:

```text
H3 median residual: 0.4603321916
S2-boundary control median residual: 0.5011442287
shuffled-response control median residual: 0.4604151147
no-modular-flow control median residual: 0.2395836908
H3 beats S2-boundary: false
H3 beats shuffled-response: false
H3 beats no-modular-flow: false
```

## Interpretation

The useful positive result is that the response tensor is nondegenerate and cheap enough to compute
at 4k. The important negative result is that it does not yet carry a controlled H3 population
signal. In particular, no-modular-flow beating H3 means the current response definition is not yet
the right observer-populated bulk observable.

This failure does not contradict the OPH paper-side Lorentz branch. The simulator is still at the
finite-regulator diagnostic layer: it can represent the BW/H3 chart, but it has not shown that the
observer records produced by the repair dynamics populate that chart in a way that beats controls.

## Four-Seed Check

Sweep:

```text
runs/e3_modular_response_h3_4k_sweep_20260605
seeds: 20260670, 20260671, 20260672, 20260673
workers: 2
inner_jobs: 4
elapsed_seconds: 4.57
```

Aggregate:

```text
mean H3 median residual: 0.4744574760
mean S2-boundary median residual: 0.5117947134
mean shuffled-response median residual: 0.4727385418
mean no-modular-flow median residual: 0.2389668382
mean response std: 0.2430165255
H3 candidate receipts: 0 / 4
H3 beats S2-boundary: 0 / 4
H3 beats shuffled-response: 0 / 4
H3 beats no-modular-flow: 0 / 4
```

The failed receipt is stable across seeds. This should be treated as a semantic/readout issue, not a
sampling-size issue.

## Next Engineering Targets

1. Improve the response source before scaling:
   use record-family/object response, collar transition observables, and defect transport response
   rather than only scalar settled fields.

2. Add repeated 4k ablations:
   field subsets, cap sizes, times, observer counts, and no-flow/shuffled controls.

3. Keep 64k as the next scale only after at least one 4k response variant beats shuffled and
   no-modular-flow controls.

4. Do not run 1M for this path yet. The current failure is semantic, not a sampling-noise problem.

5. Keep CMB-like output restricted to screen-level `C_l` proxy data until the modular-response H3
   and neutral reconstruction gates pass.

## Object-Transition / Joint-H3 Update

The scalar/sigmoid response has now been replaced in the 4k config with:

```text
observable_mode: object_transition
transform: signed_zscore
fit_mode: joint_global
model: tanh_halfspace_shared_channel_affine
```

The object-transition tensor uses observer-visible finite packets:

```text
checkpoint_class
stable_flag
record_family
s3_sector_class
repair_load_bucket
```

and emits signed transition deltas rather than sigmoided scalar means. The H3 fit now uses shared
per-channel affine nuisance parameters and held-out feature scoring. This is closer to the Pro audit
recommendation, but it still fails controls.

Single corrected smoke:

```text
runs/e3_object_transition_h3_smoke_20260605/e3_modular_response_h3_screen_4k_seed20260680_b560a3ee
  patch_count: 4096
  observer_count: 64
  feature_count: 2232
  final Phi: 0
  response std: 0.8107145395
  support-visible Lorentz/BW receipt: true
  H3 heldout normalized RMSE: 1.0001035234
  S2-boundary heldout normalized RMSE: 0.9994575371
  shuffled-response heldout normalized RMSE: 1.0009219822
  shuffled-observer-label heldout normalized RMSE: 1.0007529711
  no-perturbation heldout normalized RMSE: 1.0000000000
  best wrong-scale heldout normalized RMSE: 0.9749422281
  modular_response_h3_candidate_receipt: false
```

Four-seed corrected sweep:

```text
runs/e3_object_transition_h3_4k_sweep_20260605
  seeds: 20260680, 20260681, 20260682, 20260683
  elapsed_seconds: 13.04 with workers=2, inner_jobs=4
  mean response std: 0.8153592304
  mean H3 heldout normalized RMSE: 1.0006216056
  mean S2-boundary heldout normalized RMSE: 0.9998865428
  mean shuffled-response heldout normalized RMSE: 1.0012412889
  mean shuffled-observer-label heldout normalized RMSE: 1.0005429270
  mean no-perturbation heldout normalized RMSE: 1.0000000000
  mean best wrong-scale heldout normalized RMSE: 0.9704604464
  H3 candidate receipts: 0 / 4
```

Interpretation: the corrected response is no longer a constant-control collapse. It is a real signed
transition tensor. However, the joint H3 model explains essentially none of the held-out variance,
and wrong-scale controls are better. This is a sharper negative result: the current finite
record-packet transition readout still does not populate the H3 chart. The next change should use
actual perturb/resettle or collar Markov transition probabilities, not cap-transported packet pairs.

## Cap/Collar Perturb-Resettle Update

The implementation now includes:

```text
observable_mode: perturb_resettle_transition
```

This mode perturbs cap/collar edge packets, runs local overlap repair for the declared short horizon,
then reads observer-visible packet transition deltas. Wrong-normalization controls now keep the same
perturbation amount and change the modular collar phase/selection, so they are not just weaker or
stronger perturbations.

4k four-seed result:

```text
runs/e3_perturb_resettle_h3_phase_4k_sweep_20260605
  seeds: 20260691, 20260692, 20260693, 20260694
  elapsed_seconds: 12.30 with workers=2, inner_jobs=4
  mean response std: 0.8328525536
  mean raw response std: 0.1486260667
  mean H3 heldout normalized RMSE: 0.9762158545
  mean H3 heldout explained variance: 0.0469851675
  mean S2-boundary heldout normalized RMSE: 1.0461278685
  mean shuffled-response heldout normalized RMSE: 1.0015168559
  mean shuffled-observer-label heldout normalized RMSE: 0.9830016206
  mean no-perturbation heldout normalized RMSE: 1.0000000000
  H3 candidate receipts: 4 / 4
```

64k checks:

```text
runs/e3_perturb_resettle_h3_64k_scale_20260605/e3_perturb_resettle_h3_screen_64k_seed20260695
  observer_count: 96
  H3 heldout normalized RMSE: 0.9798087529
  receipt: false
  failed control: wrong_pi

runs/e3_perturb_resettle_h3_64k_more_seeds_20260605
  seeds: 20260696, 20260697, 20260698
  receipts: 2 / 3
  seed 20260696 failed shuffled-observer-label by 0.000313

runs/e3_perturb_resettle_h3_64k_dense_20260605/e3_perturb_resettle_h3_screen_64k_dense_seed20260696
  observer_count: 160
  candidate_count: 4096
  H3 heldout normalized RMSE: 0.9720827219
  S2-boundary heldout normalized RMSE: 1.0007384439
  shuffled-response heldout normalized RMSE: 1.0005809849
  shuffled-observer-label heldout normalized RMSE: 0.9753893856
  no-perturbation heldout normalized RMSE: 1.0000000000
  wrong 1x/pi/4pi heldout normalized RMSE: 1.0003602125 / 1.0265973843 / 1.0049326046
  receipt: true
```

Interpretation: this is the first controlled modular-response-to-H3 candidate that passes at 4k and
has a passing 64k dense-observer check. The effect is still weak: held-out explained variance is only
about 3-5 percent, and standard 64k observer sampling is mixed. This should be treated as an
interesting candidate receipt, not as established 3D bulk emergence. The next scale gate should be
64k x 4 seeds with the denser observer policy, then 256k only if at least 3/4 pass with a stronger
margin.

## Dense 64k Ensemble And 256k Probe

Added reproducible configs:

```text
configs/e3_perturb_resettle_h3_screen_64k_dense.yml
configs/e3_perturb_resettle_h3_screen_256k_probe.yml
```

Dense 64k ensemble:

```text
runs/e3_perturb_resettle_h3_64k_dense_ensemble_20260605
  seeds: 20260701, 20260702, 20260703, 20260704
  patch_count: 65536
  observer_count: 160
  candidate_count: 4096
  elapsed_seconds: 68.57 with workers=2, inner_jobs=4
  final Phi: 0 in all runs
  state-derived BW controls: pass in all runs
  transition scale: 2pi selected in all runs
  H3 candidate receipts: 4 / 4
```

H3 fit ensemble means:

```text
H3 heldout normalized RMSE: 0.9733434030
H3 heldout explained variance: 0.0525812089
S2-boundary heldout normalized RMSE: 1.0199137577
shuffled-response heldout normalized RMSE: 1.0009355708
shuffled-observer-label heldout normalized RMSE: 0.9773435787
no-perturbation heldout normalized RMSE: 1.0000000000
wrong 1x/pi/4pi heldout normalized RMSE:
  1.0149530709 / 1.1096976578 / 1.0093099850
```

256k scale probe:

```text
runs/e3_perturb_resettle_h3_256k_probe_20260605/e3_perturb_resettle_h3_screen_256k_probe_1780630733
  patch_count: 262144
  observer_count: 192
  candidate_count: 4096
  final Phi: 0
  state-derived BW controls: pass
  transition scale: 2pi selected
  H3 candidate receipt: true
```

256k H3 fit:

```text
H3 heldout normalized RMSE: 0.9698205272
H3 heldout explained variance: 0.0594481450
S2-boundary heldout normalized RMSE: 0.9999759709
shuffled-response heldout normalized RMSE: 1.0006104426
shuffled-observer-label heldout normalized RMSE: 0.9850312835
no-perturbation heldout normalized RMSE: 1.0000000000
wrong 1x/pi/4pi heldout normalized RMSE:
  1.0720820096 / 1.2367849076 / 1.0362036374
```

Interpretation:

```text
The perturb-resettle modular response now gives a repeatable weak H3-support candidate:
4/4 at dense 64k and one passing 256k probe. The fit improves slightly at 256k, but the
explained variance is still only about 6 percent. This is a real receipt worth scaling and
debugging, not yet a completed 3D-bulk emergence result.
```
