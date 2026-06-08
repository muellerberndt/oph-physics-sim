# OPH-FPE Pro Handoff: Stage 30 Bulk/CMB Blocker

Date: 2026-06-06

This package is for GPT Pro diagnosis. Pro does not have filesystem access, so the companion zip contains the relevant code, configs, tests, and run reports.

## Bottom Line

We have not yet produced a strict 3D-bulk proof or a physical CMB prediction.

Current simulator can emit:

- finite screen repair receipts (`Phi -> 0`, records commit),
- chart-level conformal Lorentz / BW automorphism sanity receipts,
- observer-facing readout and object diagnostics,
- screen-level Planck-lite `C_l` diagnostics,
- a static SPARC/RAR measurement-facing galaxy bridge,
- one 64k H3 chart debug fit that landed near dimension 3 in the original run report.

But the strict gates still fail:

- endogenous state-derived modular-generator receipt: **0**
- object bulk-population receipt: **0**
- neutral observer-record 3D reconstruction: **0**
- physical CMB prediction: **0**
- physical matter-power prediction: **0**
- particle/matter worldline receipt: **0**

The likely blocker is not scale alone. The 64k run still fails because the support-visible response/kernel/object extraction is weak or incorrectly finite-surrogated relative to the paper mechanism.

## Claim Boundary To Preserve

Do not claim:

- 3D bulk emergence,
- physical CMB prediction,
- physical `P(k)`,
- matter particles in a 3D bulk,
- dimension estimator confirms OPH.

Allowed current wording:

> The simulator currently produces finite BW/cap-flow and observer-consensus diagnostics. It has not yet established 3D bulk emergence. The next theorem-aligned target is a state-derived, support-visible modular transport receipt with collar/Markov/recovery errors carried through refinement, followed by neutral observer-record reconstruction and controls.

## Latest Important Runs

### Stage 29: 4k, modular-only observer object clustering, 4 seeds

Run dir:

`runs/stage29_modular_only_object_chart_4k_4seed_20260606`

Comparable package:

`runs/comparable_data_stage29_modular_only_object_current_20260606`

Summary:

- `Phi -> 0` all seeds.
- KMS/BW 2pi selected in 3/4 seeds.
- strict state-derived BW still fails all seeds.
- H3 response receipt 3/4.
- object-chart receipt 3/4.
- `h3_not_boundary_dominated` improved to 3/4 after switching object clustering to modular-only fields.
- neutral reconstruction still fails: mean corr dimension ~2.69, local MLE ~3.71, estimators agree 0/4, candidate 0/4.
- no strict bulk proof.

### Stage 30: 64k compact probe

Run dir:

`runs/stage30_modular_only_transport_64k_probe_20260606/e4_shared_observer_bulk_64k_local_tokens_feature_select_seed20261112_da9acef4`

Comparable package:

`runs/comparable_data_stage30_64k_probe_current_20260606`

Important values:

- final `Phi`: 0.
- state BW mode: `transition_response_unitary`.
- state BW median residual: ~`1.61e-15`.
- state BW controls pass, but this is direct/declared transition automorphism, not endogenous modular generator.
- H3 response receipt: **false**.
- H3 fit heldout normalized RMSE: ~`1.0056`.
- H3 explained variance: ~`-0.011`.
- S2 boundary RMSE: ~`1.0003`.
- shuffled-response RMSE: ~`1.0051`.
- strict wrong-scale win fraction: `0.625`.
- original H3 chart debug dimension: corr ~`3.0977`, local MLE ~`2.8896`, estimator agreement true.
- object chart receipt: **false**.
- object count: 16.
- localized non-boundary objects: 8.
- H3 compactness: ~`0.436`; S2 compactness ~`0.661`; but shuffled-H3 compactness ~`0.269`, so object chart fails shuffled controls.
- neutral reconstruction: corr ~`5.122`, MLE ~`6.097`, candidate false.
- blind neutral: corr ~`7.456`, MLE ~`11.068`, candidate false.
- Planck-lite best positive screen field: `record_signature_smooth_k32`.
- Planck-lite shape correlation: ~`0.2828`; RMSE ~`0.9592`.
- CMB transfer diagnostic: train corr ~`0.8767`, test corr ~`0.7772`, control gap ~`0.5209`; this is a diagnostic fitted to Planck shape, not a physical prediction.

### Cached 64k H3 Refit With Material Wrong-Scale Audit

Report:

`runs/stage30_modular_only_transport_64k_probe_20260606/e4_shared_observer_bulk_64k_local_tokens_feature_select_seed20261112_da9acef4/h3_refit_material_audit.json`

Command used:

```bash
python3 -m oph_fpe.cli h3-refit \
  --run-dir runs/stage30_modular_only_transport_64k_probe_20260606/e4_shared_observer_bulk_64k_local_tokens_feature_select_seed20261112_da9acef4 \
  --out runs/stage30_modular_only_transport_64k_probe_20260606/e4_shared_observer_bulk_64k_local_tokens_feature_select_seed20261112_da9acef4/h3_refit_material_audit.json \
  --candidate-count 4096 \
  --candidate-radius 2.0 \
  --softness 0.25 \
  --seed 20261202 \
  --pass-ratio 1.0 \
  --min-observers 32 \
  --min-features 16 \
  --fit-mode joint_global \
  --heldout-fraction 0.25 \
  --anchor-weight 0.05 \
  --max-iterations 4 \
  --feature-selection change_probability_only \
  --min-feature-std 0.05 \
  --candidate-mode fibonacci_ball \
  --control-fit-mode same_h3_model_not_affine_target_fit
```

Result:

- H3 receipt: false.
- H3 nRMSE: `1.0053859644`.
- H3 explained variance: `-0.0108009375`.
- S2 nRMSE: `1.0001394832`.
- no-perturbation nRMSE: `1.0`.
- wrong-scale nRMSE: `1x=1.0121`, `pi=1.0027`, `4pi=1.6055`.
- H3 chart dimension now not 3 for this seed: corr ~`2.0458`, MLE ~`2.0036`.
- strict wrong-scale win fraction: `0.625`.
- material wrong-scale win fraction at 1% margin: `0.2083`.

Interpretation:

Some strict wrong-scale wins are near-ties, but there are still material wrong-scale wins. More importantly, the response signal is too weak: H3 has negative explained variance and does not beat no-perturbation/S2 clearly.

## Recent Code Changes

### Transported observer-support readout

File:

`oph_fpe/bulk/modular_response_kernel.py`

Added `transition_readout_mode="transported_support"` for perturb/resettle modular-response kernels:

- perturb/resettle still happens cap-locally,
- observer support can be read after applying `lambda_C(scale*t)` to the support and nearest-neighboring back to the screen,
- this improved wrong-scale behavior at 4k from the earlier ~0.759 wrong-scale win fraction to ~0.439-ish in several runs,
- it did not solve strict H3 response controls.

### Modular-only object clustering

Configs:

- `configs/e4_shared_observer_bulk_4k_response_mix.yml`
- `configs/e4_shared_observer_bulk_64k_local_tokens_feature_select.yml`

Changed `observer_chart_population.observer_cluster_fields` to:

```yaml
observer_cluster_fields:
  - modular_response_cluster
  - modular_response_component_0
  - modular_response_component_1
```

This improved boundary dominance:

- 4k: `h3_not_boundary_dominated` improved from 0/4 to 3/4 in one sweep.
- 64k: H3 compactness beat S2 compactness.

But shuffled-H3 compactness often still beats actual H3 compactness, so object chart controls fail.

### Material wrong-scale audit

File:

`oph_fpe/bulk/h3_response_fit.py`

Added material wrong-scale counts:

- strict wrong-scale wins still block receipt,
- `material_*` fields only count wrong-scale wins that beat the 2pi/H3 feature residual by a relative margin, default `0.01`.

File:

`oph_fpe/cosmology/comparable_data.py`

The comparable snapshot now reports both strict and material wrong-scale win fractions.

Tests:

```bash
python3 -m pytest -q
```

Result after these changes:

`163 passed in 36.43s`

## Best Measurement-Facing Data So Far

### CMB / early-universe side

Not a physical prediction yet.

Best current measurement-facing diagnostics:

- Planck-lite screen `C_l` shape from Stage 30:
  - best field: `record_signature_smooth_k32`
  - corr ~`0.2828`
  - RMSE ~`0.9592`
- CMB screen-basis transfer:
  - train corr ~`0.8767`
  - test corr ~`0.7772`
  - max control test corr ~`0.2563`
  - test-control gap ~`0.5209`
  - caveat: weights are fitted to Planck TT shape, so this is not an OPH prediction.
- OPH-CMB anomaly-stress adapter:
  - finite-collar parent diagnostics only.
  - missing theorem/physical gates remain closed.

### Static galaxy side

This is the strongest measurement-facing lane, but it is static phenomenology, not CMB/bulk proof.

SPARC/RAR bridge values from current comparable snapshots:

- 175 galaxies.
- 6207 rows overall in the local data path.
- RAR point count: 2693.
- RAR scatter: ~`0.1328 dex`.
- holdout test velocity RMSE: ~`22.69 km/s`.
- baryon-only holdout test velocity RMSE: ~`59.18 km/s`.
- improvement fraction: ~`0.6166`.
- claim tier: `Tier1_phenomenological_continuation`.

## Suspected Scientific/Implementation Blockers

### 1. The finite modular-response observable may still be wrong

The paper mechanism is cap-local modular flow / BW scaling. Current finite kernel is a surrogate:

```text
cap/collar perturb -> local repair -> observer packet transition -> H3 halfspace fit
```

Even with transported-support readout, the signal remains weak:

- H3 explained variance near or below zero,
- no-perturbation can beat H3,
- wrong normalizations still materially win on some features.

Question for Pro:

> What finite support-visible observable should be used so the simulator tests the paper theorem surface rather than a noisy repair surrogate?

### 2. Endogenous modular generator is not solved

The strict endogenous `K_a = -log(rho_C + aI)` route still fails in 4k runs. The 64k Stage 30 state BW pass uses `transition_response_unitary`, which is a direct/declared automorphism sanity lane, not an endogenous state-derived modular generator.

Question for Pro:

> Is an endogenous finite modular generator required for the simulation proof, or is chart-level conformal Lorentz + BW automorphism sanity enough for the intended paper-accurate finite computation?

### 3. Neutral observer reconstruction is too high-dimensional

The neutral observer reconstruction excludes radial depth and S2 support leakage, but distances over blind features are high-dimensional:

- Stage 30 neutral corr ~5.12, MLE ~6.10.
- Blind neutral corr ~7.46, MLE ~11.07.

Question for Pro:

> Should neutral bulk reconstruction use raw observer-transition feature vectors, or should it use a paper-derived quotient/chart distance from the cap algebra/modular response?

### 4. Object extraction may be underpowered or still wrong

Modular-only object clustering fixed boundary dominance but did not pass shuffled controls. At 64k:

- actual H3 compactness beats S2,
- shuffled-H3 compactness beats actual H3.

Question for Pro:

> What is the correct finite definition of an observer-facing object/record family for the OPH papers: transition-affinity clusters, record-family persistence, cap algebra sectors, checkpoint continuation classes, or something else?

### 5. CMB prediction path is not yet physical

Current screen `C_l` and transfer diagnostics are useful, but not physical predictions. The OPH-CMB adapter exposes placeholders/diagnostic parent quantities, not a validated `rho_A(a)`, `Gamma_rec`, `B_A(k,a)`, or CAMB/CLASS source module.

Question for Pro:

> What is the minimal OPH-derived quantity we can honestly compare to CMB measurements first: freezeout-screen `C_l` shape, anomaly-stress kernel shapes, repair-load spectra, or a constrained Boltzmann source term?

## Concrete Questions For Pro

1. In the papers, is the simulator expected to prove a third-person 3D bulk, or only observer-facing Lorentz/H3 chart compatibility plus object population?
2. Is chart-level `Conf^+(S^2) ≅ PSL(2,C) ≅ SO^+(3,1)` enough as the finite Lorentz receipt, or must a finite state-derived modular generator reproduce it?
3. What exact finite cap/collar state should replace the current perturb/resettle surrogate?
4. How should we construct `rho_C` so `K_a=-log(rho_C+aI)` is nontrivial, support-visible, and does not leak the geometric `lambda_C(2pi*t)` target?
5. Should the simulator fit observer response into canonical H3 charts, or derive the distance matrix neutrally from observer record similarities?
6. Are fractional dimensions in neutral diagnostics expected finite-regulator artifacts, or do they indicate that our observer-object extraction is wrong?
7. Should wrong-scale controls use strict per-feature wins, material-margin wins, aggregate heldout nRMSE, or all three?
8. What is the first CMB-comparable output that is paper-accurate enough to publish as a diagnostic?
9. Is the static SPARC/RAR lane scientifically connected to the OPH simulator, or should it remain separate from the early-universe/bulk proof lane?
10. What is the next smallest code change likely to turn the H3 response from negative explained variance into a real controlled signal?

## Suggested Next Coding Steps If Pro Agrees

1. Replace raw blind Euclidean reconstruction with a cap-response quotient/diffusion distance that is explicitly paper-derived and excludes S2 leakage.
2. Add a noncommutative/sparse block `rho_C` built from perturb/remeasure transition co-occurrences, not just diagonal packet distributions.
3. Use object families based on checkpoint continuation and sector-preserving transition classes, not just support overlap or modular response clusters.
4. Rerun 4k four-seed, then 64k one-seed, only after the 4k material wrong-scale win fraction is near zero and H3 explained variance is positive.
5. Keep CMB output at diagnostic `C_l`/adapter level until the bulk/object gates pass.

## Files Included In Zip

The zip includes:

- relevant source files under `oph_fpe/bulk`, `oph_fpe/scale`, and `oph_fpe/cosmology`,
- relevant tests,
- relevant configs,
- Stage 29 and Stage 30 comparable snapshots,
- Stage 30 detailed reports,
- cached H3 refit material-audit report.

No `.env`, `.env.example`, credentials, cloud config secrets, or local credential files are included.
