# oph-physics-sim

Finite OPH screen-consensus and BW/cap-flow diagnostic.

Working package: `OPH-FPE` — Observer-Patch Fundamental Physics Emergence.

This is currently a CPU-first MVP. It tests finite regulator mechanics before any 3D-bulk,
early-universe, or physical cosmology claim:

- finite patch graphs with per-edge interface packets,
- finite groups (`Z2`, `S3`, and clock groups),
- local overlap mismatch `Phi`,
- annealed local repair with verifier receipts,
- record commits after a stability window,
- cycle holonomy defects and simple worldline tracking,
- explicit OPH pixel-ratio `P` screen-cell architecture receipts,
- federated echosahedral 12-port screen microphysics metadata,
- explicit named `P0..P11` port assignment receipts,
- BW cap-flow residual verifier for `lambda_C(2*pi*t)`,
- state-derived support-visible modular-probe receipts,
- collar Markov / recovery-error receipts,
- support-visible cap-net hot boundary program receipts,
- record/object/defect support-profile fits into the canonical H3 chart,
- time-resolved S3 defect timelines and H3 fitted worldline-path diagnostics,
- mandatory negative-control receipts,
- gated freezeout-screen `C_l` proxy receipts from observer-facing fields,
- planted and smoke-test dimension estimators,
- evidence bundles under `runs/<run_id>/`.

The original E1 dimension path is deliberately labeled `graph_shortest_path_mvp`; it is a scaffold
for later observer-record reconstruction, not a claim that a 3D bulk emerged. The older large-screen
path `array_modular_lift_record_history` is a visualization/regulator diagnostic only. The current
theorem-aligned path is `state_derived_modular_probe`: build cap/collar observer-visible states,
compute collar Markov receipts, form regularized `K_a = -log(rho_C + aI)`, and compare
state-derived modular matrix elements against `lambda_C(2*pi*t)`.

## Quickstart

```bash
python3 -m pytest -q
python3 -m oph_fpe.cli run --config configs/e0_z2_patchnet.yml --out-dir runs
python3 -m oph_fpe.cli run --config configs/e1_s3_bulk_emergence.yml --out-dir runs
python3 -m oph_fpe.cli run-array --config configs/e1_s3_modular_screen_64k.yml --out-dir runs
python3 -m oph_fpe.cli run-bw-array --config configs/e1_s3_bw_screen_64k.yml --out-dir runs
python3 -m oph_fpe.cli run-bw-array --config configs/e1_s3_state_modular_screen_4k.yml --out-dir runs
python3 -m oph_fpe.cli run-bw-array --config configs/e2_kms_freezeout_cl_screen_64k.yml --out-dir runs
python3 -m oph_fpe.cli run-bw-sweep --configs configs/e1_s3_bw_screen_4k.yml --seeds 20260601,20260602 --out-dir runs
python3 -m oph_fpe.cli cl-ensemble-report --run-dir runs/kms_freezeout_cl_4k_sweep_20260602 --out runs/cl_ensemble_20260602
python3 -m oph_fpe.cli h3-ensemble-report --run-dir runs/coarse_object_h3_4k_sweep_20260602 --include runs/coarse_object_h3_64k_20260602 --out runs/h3_ensemble_20260602
python3 -m oph_fpe.cli cmb-lite-compare --run-dir runs/<run_id> --benchmark runs/benchmarks/COM_PowerSpect_CMB-TT-binned_R3.01.txt
python3 -m oph_fpe.cli run-viewer --run-dir runs/<run_id>
```

`run-bw-sweep` now auto-fills available CPUs when `--workers` and `--inner-jobs` are omitted.
Set `OPH_FPE_CPUS=<N>` to override the detected CPU budget, or pass explicit worker caps when the
machine is shared. The sweep planner applies the inner-job budget to both BW cap probes and
freezeout angular spectra. For a single `run-bw-array` run, set `bw.n_jobs: auto` and
`cosmology.angular_power.n_jobs: auto` in the config to use the available CPU budget inside that run.

Object-engine runs write state snapshots, repair events, record events, defect worldlines, and
controls. Array-engine runs write compact summaries instead. Both write a manifest, config copy,
mismatch trace, `pixel_scale.json`, `screen_microphysics.json`, dimension report, cosmology proxy
placeholder, and `verifier_receipts.jsonl`.
Main BW-array runs also write paper-stack fixed-cutoff receipts:
`edge_sector_heat_kernel_report.json`, `central_record_born_report.json`, and
`observer_checkpoint_restoration_report.json`.

## Milestones

- `E0`: finite patch-net calibration with `Z2` repair, record stability, and exposed defects.
- `E1`: `S3` smoke run with holonomy defects and dimension-estimator scaffolding.
- `E1-large`: vectorized spherical screen plus legacy modular-flow record-history lift.
- `E1-state`: state-derived cap/collar modular probe with CMI/recovery receipts.
- `E1-microphysics`: fixed-cutoff screen receipts for the edge-sector heat-kernel/Casimir
  surrogate, central record/Born-Luders event algebra, and observer checkpoint/restoration.
- `E2-screen`: first gated freezeout-screen `C_l` proxy from observer-facing fields.
- `E2-defects`: S3 screen holonomy clusters can be tested as H3 support profiles, sampled as
  repair-time defect-worldline precursors, fitted into H3 worldline paths for visualization, and
  scored by a conservative particle-likeness diagnostic. The current particle receipt remains false.
- Next: stronger record-object construction, repeated-seed refinement scaling, neutral
  observer-record reconstruction, defect worldlines/fusion, then physical cosmology adapters.

## Current Large-Scale Smoke

On this workstation:

```text
4k S3 mutual 12-port screen:  final Phi=0, d ~= 2.77, 69,632 modular-lift points
64k S3 mutual 12-port screen: final Phi=0, d ~= 2.79, 200,000 sampled modular-lift points
1M S3 mutual 12-port screen:  final Phi=0, d ~= 2.72, 500,000 sampled modular-lift points
```

Those numbers are legacy diagnostics. They do not establish 3D bulk emergence. GPUs are not needed
until we add large dense modular blocks, learned repair policies, persistent homology at large point
count, or later Boltzmann/CAMB sweeps.

## BW Cap-Flow Receipts

The primary theorem-aligned diagnostic is now the state-derived BW residual, not the radialized
modular-depth dimension estimate. The legacy kinematic verifier compares finite cap-local geometric
transport against `lambda_C(2*pi*t)` and remains useful as a geometry/interpolation sanity check.
The state-derived verifier instead constructs `rho_C` from observer-visible cap/collar packets,
uses `K_a = -log(rho_C + aI)`, and compares finite modular matrix elements to the same
`lambda_C(2*pi*t)` target.

Legacy P-weighted, support-visible-regularized kinematic receipts:

```text
4k BW screen:  R_BW median ~= 0.385, p90 ~= 0.471, controls median ~= 0.627-1.257
64k BW screen: R_BW median ~= 0.383, p90 ~= 0.470, controls median ~= 1.089-1.272
1M BW screen:  R_BW median ~= 0.384, p90 ~= 0.465, controls median ~= 1.260-1.275
```

The correct `2*pi` transport separates from controls and p90 improves with refinement. The median
improves from 4k to 64k but rebounds slightly at 1M, so this is not a theorem-grade monotone BW
refinement result. These runs are now labeled `kinematic_geometric_bw_sanity`; new
`state_derived_modular_probe` receipts are required before neutral bulk reconstruction.

Current state-derived status has two layers:

```text
cooccurrence-density state surrogate:
  fails the state-derived control gate

transition-response automorphism probe:
  runs at 4k, 64k, and 256k
  final Phi = 0
  mandatory negative controls pass
  state-derived BW median ~= numerical floor
  correct 2*pi beats wrong-normalization / shuffled / no-flow controls
  refinement slope is not meaningful because residuals are at numerical floor
```

The transition-response mode is a branch-instantiation test: it constructs the finite cap
automorphism generator from a declared KMS/BW-normalized perturb/remeasure transition operator. It
is useful for testing the downstream observer-record machinery, but by itself it does not prove
that consensus dynamics generated the BW branch. So 3D bulk emergence is still not observed.

Current status distinguishes three layers:

```text
support-visible BW/KMS Lorentz / 3+1D kinematics receipt: true on the KMS branch
conformal H3 spatial chart receipt: true when cap-normal/H3 checks pass
record-populated 3D bulk reconstruction receipt: false
```

The paper-side mechanism is that cap modular flow on the support-visible \(S^2\) chart gives the
Lorentz group, and the canonical spatial chart is the 3D homogeneous space
`SO+(3,1)/SO(3)` represented by `H3`. The simulator now records this conformal/H3 chart
diagnostic separately from record-populated spatial bulk reconstruction. The old observer-similarity
dimension estimator is retained only as a debug diagnostic and has no physics claim.

Current observer-similarity debug diagnostics:

```text
4k neutral sweep:
  4/4 seeds pass BW/KMS and planted/shuffled controls
  old local-MLE debug estimates: 3.016-3.215
  old median debug estimate ~= 3.165

current stricter 4k report:
  local MLE ~= 3.162
  correlation log-fit ~= 2.105
  dimension_estimators_agree = false
  spatial_bulk_3d_reconstruction_receipt = false

64k scale check:
  local MLE ~= 6.278
  correlation log-fit ~= 2.025
  dimension_estimators_agree = false
  spatial_bulk_3d_reconstruction_receipt = false
```

Those fractional/debug values are not bulk dimensions. The simulator now writes
`record_populated_h3_report.json`, which fits observer record/cap-response profiles into the
conformal H3 chart and compares residuals against S2-boundary and shuffled controls. The populated
bulk receipt remains false unless H3 wins those controls. The screen holonomy report now writes
pre-bulk S3 defect clusters, screen-local interaction proxies, and a particle-likeness diagnostic.
Current runs find localized/persistent/sector-stable screen defects plus screen-local
transport/fusion/scattering proxies, but they do not yet pass H3 worldline, contractible-path
transport, full interaction controls, or neutral-bulk gates, so they are not matter particles.

## First Measurement-Facing Screen Proxy

The first reproducible comparison-facing output is now a gated freezeout-screen angular spectrum,
not a physical CMB prediction. The run:

```text
runs/kms_freezeout_cl_20260602/e2_kms_freezeout_cl_screen_64k_1780360849
  final Phi = 0
  primary transition source = kms_collar_transport_response
  selected scale = 2pi
  state-derived BW median ~= 2.49e-15
  state-derived controls pass
  cosmology gate allowed = true
  neutral reconstruction written = false
  bulk_3d_established = false
```

It writes:

```text
freezeout_fields.npz
freezeout_map_summary.json
cl_comparison_report.json
cl_proxy.csv
cl_controls.csv
cosmology_observables.json
cosmology_gate_report.json
```

`cl_comparison_report.json` uses the direct `spherical_harmonic` auto-power estimator with OPH
cell-entropy quadrature weights, so reported `C_ell` values are nonnegative. In this 64k receipt,
the strongest fields peak at high regulator multipoles:

```text
record_signature:       peak ell 32, peak D_ell ~= 0.0821
s3_class_density:       peak ell 32, peak D_ell ~= 0.0642
cumulative_repair_load: peak ell 31, peak D_ell ~= 0.0560
```

Control separation is present but not yet strong enough for measurement claims: relative L2
separation from shuffled/random controls is roughly `0.21-0.52`, while shape correlations can remain
high. This is therefore a useful reproducible screen statistic to compare across seeds and controls,
not a Planck likelihood, not CAMB/CLASS input, and not evidence that a 3D bulk has emerged.

Repeated-seed screen ensemble:

```text
runs/cl_ensemble_20260602
  input runs: 5
  gate-allowed runs: 5
  4k seeds: 4/4 gate allowed
  64k seeds: 1/1 gate allowed
```

For the 4k repeated-seed set, the screen-spectrum shapes are stable but not yet measurement-grade:

```text
record_signature:       peak ell mode 22, seed-shape corr ~= 0.912, control L2 delta ~= 0.369
s3_class_density:       peak ell mode 24, seed-shape corr ~= 0.914, control L2 delta ~= 0.398
cumulative_repair_load: peak ell mode 23, seed-shape corr ~= 0.887, control L2 delta ~= 0.359
stable_count:           peak ell mode 24, seed-shape corr ~= 0.908, control L2 delta ~= 0.253
modular_depth:          peak ell mode 24, seed-shape corr ~= 0.883, control L2 delta ~= 0.289
```

Interpretation: the finite KMS/BW screen pipeline now emits reproducible angular statistics under
strict gates. The next accuracy step is stronger control separation and scale/refinement behavior,
not a direct comparison to Planck or DESI.

See `docs/paper_stack_alignment.md` for the current claim-by-claim alignment against the core paper
stack.

## OPH Screen Microphysics

Run configs now include:

```yaml
screen:
  chart: support_visible_s2_cellulation
  carrier: federated_echosahedral_patch
  ports_per_patch: 12
  cap_family: round_caps_on_s2
  edge_sector_law: fixed_cutoff_heat_kernel_casimir_surrogate

oph_constants:
  P: 1.6309682094039593
  P_source: endpoint_public
  use_P_for:
    - cell_area
    - cell_entropy
    - cap_capacity
    - residual_weighting

screen_units:
  mode: numerical_regulator
```

The simulator writes `pixel_report.json`, `pixel_scale.json`, and `screen_microphysics.json`, then
mirrors the same structures into the manifest and cosmology proxy. `P` is now part of the finite
screen normalization layer: `cell_area_planck = P`, `cell_entropy_capacity = P/4`, cap reports
include `cap_area_planck` and `cap_entropy_capacity`, and BW residual norms use the local
cell-entropy measure. The conformal map and BW normalization remain unchanged:
`s = 2*pi*t`.

The default `screen_units.mode` is `numerical_regulator`, so `N` is a sampling/refinement count, not
a literal cosmological horizon cell count. In later `physical_cell_toy_universe` runs, the toy
radius is `R/lP = sqrt(NP / 4pi)`. In both modes `P` does not force the BW/Lorentz dimension
estimate; the BW cap-flow verifier is the required mechanism for that.

BW configs use a cell-scaled regulator collar:

```text
collar_width = collar_k * sqrt(4*pi/N_patch)
```

This keeps the finite collar tied to the screen resolution, matching the paper stack's shrinking
collar / carried-error framing better than a fixed angular collar.

## Cloud

Use a local Google Cloud project configured outside the repository. A convenient pattern is a named
`gcloud` configuration:

```bash
gcloud config configurations activate <your-config-name>
```

Cloud template defaults live in `configs/` and secret-free local placeholders are documented in
`.env.example`. Put real project IDs, bucket names, account IDs, tokens, and keys in `.env.local`,
shell exports, or cloud-native identity, not in committed files.
See `docs/cloud.md` for quota notes and provider boundaries.

For parallel execution and hardware sizing, see `docs/parallel_cloud_plan.md`.
For DigitalOcean fixed-pool worker setup, see `docs/digitalocean_pool_setup.md`.
For the current 3D-bulk claim boundary and observer-facing readouts, see `docs/bulk_emergence_status.md`.

Current implementation boundary: the finite OPH KMS/BW collar-transport branch now selects `2pi`
at 4k, 64k, and 256k, with the selector score improving over refinement. The raw
perturb/remeasure and repair-affinity selectors still select `1x`, so this is a
branch-instantiation receipt rather than proof that raw repair dynamics endogenously selected the
Lorentz/BW branch.
