# OPH-FPE CMB, 3D Bulk, And Particle Execution Plan

Updated: 2026-06-02

## Claim Boundary

The simulator may currently claim only finite OPH screen repair, BW/KMS cap-flow receipts, observer
consensus receipts, and gated screen-statistic outputs.

It must not claim:

```text
physical CMB prediction
3D bulk emergence
particle spectrum emergence
early-universe simulation
CAMB/CLASS likelihood
```

until the gates below pass.

## Core Dependency Chain

```text
BW/KMS Lorentz gate
-> conformal cap-normal / H3 spatial chart receipt
-> observer-object consensus
-> record/object response fit into H3
-> planted and shuffled controls
-> defect support-profile fit into H3
-> defect/worldline tracking in reconstructed bulk
-> calibrated freezeout / CMB-lite outputs
-> physical CMB or particle interpretation
```

A screen-level `C_l` proxy can be produced earlier as a diagnostic because the CMB is observed as an
angular field. A physical CMB prediction should wait for neutral 3D bulk reconstruction, or for a
paper-side theorem that justifies direct screen-to-sky prediction without reconstructed bulk.

## Track A: 3D Bulk And Particle Output

### A1. Strengthen BW/KMS Receipts

Run the same cap/time observable family at:

```text
4k, 64k, 256k, 1M
```

Required outputs:

```text
R_BW(N)
KMS selected scale
wrong-normalization controls
shuffled/no-flow controls
collar CMI and recovery errors
observer consensus metrics
```

Pass condition:

```text
KMS selects 2pi across sizes and seeds.
State-derived BW controls fail as expected.
CMI/recovery errors remain controlled.
Non-KMS raw repair sources remain separately reported, not hidden.
```

### A2. Observer-Object Consensus

Extract persistent record families and score:

```text
overlap agreement
record persistence
counterfactual perturb/remeasure stability
bad rewrite rejection
```

Pass condition:

```text
persistent object families exist
overlap agreement is stable across observer views
fake record rewrite fails
shuffled interfaces fail
```

### A3. Conformal H3 Chart And Record-Populated Bulk

Build the spatial chart mechanically from the cap/BW branch:

```text
round caps on S2
-> de Sitter cap normals in R^{3,1}
-> Minkowski Gram / cross-ratio proxy
-> canonical H3 chart, H3 ~= SO+(3,1)/SO(3)
```

Then fit observer-facing data into that chart:

```text
observer record / object / cap response profile
-> H3 point or H3 cell fit
-> record-populated spatial chart residual
```

Implemented output:

```text
record_populated_h3_report.json
record_family_h3_report.json
defect_cluster_h3_report.json
defect_h3_worldlines_report.json
h3_ensemble_report.json
```

These receipts are allowed to be false. A false record/object result means the current record/cap
response fields are still boundary-like or insufficient for populated-bulk claims. A true
defect-cluster support receipt is a particle precursor, not by itself a full bulk receipt.

Current status:

```text
fixed-cutoff screen microphysics: edge-sector/Casimir, central-record/Born, checkpoint/restoration receipts pass
object-transition modular-response kernel: implemented, signed and nondegenerate at 4k
joint global H3 fit: implemented with held-out scoring, but currently fails controls
record/object populated bulk: false
defect-cluster H3 support: false under the stricter dedicated reconstruction cap net
time-resolved defect worldline precursor: screen persistence exists, H3-worldline gate false
particle-likeness diagnostic: localized/persistent/sector-stable screen defects exist, screen-local interaction proxies measured, particle receipt false
screen C_l proxy: stable for record_signature and useful as a measurement-facing diagnostic
C_l execution: independent target/control spectra parallelize through cosmology.angular_power.n_jobs
latest scaled check: 256k still keeps populated-bulk, H3-worldline, particle, and physical-CMB gates false
```

Latest corrected modular-response check:

```text
runs/e3_object_transition_h3_4k_sweep_20260605
  H3 candidate receipts: 0 / 4
  wrong-scale controls beat H3 on held-out score
```

Next bulk-readout target:

```text
replace cap-transported packet-pair surrogates with actual cap/collar perturb-resettle
or empirical collar Markov transition probabilities before scaling beyond 4k
```

Forbidden:

```text
radial modular-depth lift as evidence
hard-coded 3D coordinates
host-side smoothing presented as observer evidence
fractional observer-similarity dimension as a physics receipt
```

Keep the old observer-similarity estimator only as debug:

```text
observer_similarity_debug_report
component_dimension_debug_reports
local MLE k sweep
correlation logfit quantile sweep
```

### A4. Neutral Reconstruction Controls

Mandatory controls:

```text
planted 2D -> returns near 2
planted 3D -> returns near 3
planted 4D -> returns near 4
planted S2 boundary -> remains boundary-like
planted H3 bulk -> returns near 3
shuffled observer records -> degrades reconstruction
random same-degree graph -> fails geometry claim
radial-depth evidence path excluded
```

Pass condition:

```text
conformal H3 chart receipt passes
record/object H3 residual beats S2-boundary and random controls
debug estimators do not contradict the populated-H3 fit over a scaling window
planted controls pass and shuffled/random controls fail
BW/KMS and observer-object gates already pass
```

Allowed wording after pass:

```text
The finite regulator emits a controlled observer-record reconstruction with a 3D-compatible
dimension window after BW/collar/object controls.
```

### A5. Defect And Particle-Like Output

Track S3 holonomy defects on the screen first:

```text
oriented triangle holonomy
defect class
defect cluster
cluster persistence
fusion/annihilation events
```

Map defects into reconstructed bulk only after neutral bulk gate passes.

Particle-like criteria:

```text
localized support
persistent lifetime
transportable sector
conserved fusion rules
reproducible scattering
```

Deliverables:

```text
array_holonomy_trace
defect_cluster_trace
defect_worldlines
defect_fusion_report
particle_likeness_report
bulk_3d_with_worldlines visualization
```

Current particle status:

```text
screen holonomy defects can be localized, persistent, and sector-stable;
screen-local transport, inverse-holonomy fusion, and class-transition scattering proxies are implemented;
contractible-path transport in a neutral/H3 bulk is not yet implemented/passing;
full fusion conservation and scattering reproducibility controls are not yet implemented/passing;
particle_matter_receipt remains false.
```

Latest interaction smoke:

```text
runs/e2_paper_microphysics_scale_256k_20260603/e2_kms_freezeout_cl_screen_256k_seed20260641_a4fe9c31
  patch_count: 262144
  elapsed_seconds: 254.4 on local CPU path with inner_jobs=8
  edge-sector heat-kernel/Casimir receipt: true
  central-record/Born receipt: true
  observer checkpoint/restoration receipt: true
  persistent screen defect worldlines: 6
  screen transport proxy count: 6
  inverse-holonomy fusion candidates: 39
  scattering reproducibility proxy: true
  defect_h3_worldline_precursor_receipt: false
  particle_like_count: 0
  particle_matter_receipt: false
```

Latest scaled run:

```text
runs/e2_h3_reconcap_worldline_256k_20260602/e2_kms_freezeout_cl_screen_256k_seed20260636_3f5efd6c
  patch_count: 262144
  elapsed_seconds: 309.9 on local 10-core CPU path with inner_jobs=8
  final Phi: 0
  BW/KMS Lorentz receipt: true
  conformal H3 chart receipt: true
  record_populated_h3_spatial_receipt: false
  defect_h3_worldline_precursor_receipt: false
  particle_matter_receipt: false
  CMB-lite best normalized RMSE: 0.6306
  physical_cmb_prediction: false
```

## Track B: CMB Output

### B1. Current Diagnostic

Current implemented stage:

```text
freezeout screen fields
-> gated screen C_l proxy
-> repeated-seed C_l ensemble report
-> shape-only CMB-lite comparison against external TT benchmark files
-> standalone screen/observer/H3/defect/C_l viewer
```

This is useful for seed/control stability. It is not a physical CMB prediction.

First actual-measurement diagnostic:

```text
benchmark:
  Planck 2018 TT binned spectrum
  COM_PowerSpect_CMB-TT-binned_R3.01.txt
  source: https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/COM_PowerSpect_CMB-TT-binned_R3.01.txt

run:
  runs/cap_net_freezeout_cl_64k_seed20260622_20260602/e2_kms_freezeout_cl_screen_64k_seed20260622_b614e4e7

output:
  cmb_lite_comparison_report.json
  best shape field: s3_class_density
  best normalized RMSE: 0.6417
  physical_cmb_prediction: false
```

Interpretation: the current OPH screen proxy is not close to a physical Planck TT prediction. This
is a useful negative/diagnostic receipt because it gives us a real measured curve to improve against
after the bulk/freezeout gates are stronger.

### B2. Freezeout Surface

Define a freezeout rule:

```text
records committed
repair opacity drops
observer-visible packet field remains stable
later cycles do not rewrite beyond tolerance
```

The run bundle must write:

```text
freezeout_cycle
committed_fraction
field stability before/after freezeout
rewrite/reheating controls
```

### B3. Candidate Temperature Source

Implement `cmb_lite` candidate fields:

```text
dT/T = a1 * record_entropy_fluctuation
     + a2 * repair_load_fluctuation
     + a3 * defect_density_fluctuation
     + a4 * modular_response_fluctuation
```

Each candidate must carry:

```text
formula
normalization
fields used
claim boundary
control results
seed ensemble stability
```

### B4. CMB-Lite Metrics

For each candidate:

```text
remove monopole and dipole
compute C_l^TT proxy
peak locations
peak ratios
low/mid/high ell power
seed stability
control separation
refinement behavior
```

Compare only shape-level diagnostics until physical calibration exists.

### B5. CMB Controls

Mandatory controls:

```text
shuffled freezeout field
random freezeout surface
no repair
no modular flow
shuffled observer records
shuffled defect field
seed-scrambled phase history
```

### B6. Physical CMB Gate

Only after the neutral 3D gate:

```text
reconstructed bulk source fields
freezeout surface in reconstructed bulk
line-of-sight / projection model
calibrated angular scale
physical dT/T normalization
```

Final deliverables:

```text
cmb_lite_report
cmb_candidate_spectra
cmb_control_report
cmb_seed_ensemble_report
cmb_refinement_report
later: CAMB/CLASS adapter inputs
```

Viewer deliverable now available:

```bash
python3 -m oph_fpe.cli run-viewer --run-dir runs/<run_id>
```

The viewer writes `plots/oph_realtime_viewer.html` with the S2 lattice, observer readout sample,
screen defect clusters, optional repair-time defect timeline scrubber, H3 support-fit samples,
repair/record trace, and C_l proxy. It keeps gate badges visible so a failed bulk or particle gate
is not visually upgraded.

Timeline-enabled smoke artifact:

```text
runs/e2_h3_worldline_freezeout_4k_20260602/e2_kms_freezeout_cl_screen_4k_seed20260627_30b9fea6
  defect_timeline_report.json
  timeline snapshots: 8
  persistent defect-worldline precursors: 76
  max lifetime cycles: 47
  defect_h3_worldlines_report.json
  H3-fitted defect events: 607
  persistent H3 worldlines: 76
  H3 beats S2-boundary control: false
  H3 beats shuffled control: false
  bulk_worldline_precursor_receipt: false
  particle_matter_receipt: false
  viewer: plots/oph_realtime_viewer.html
```

Interpretation: the viewer can now show fitted H3 defect paths, but the receipt correctly keeps the
bulk/particle gate closed because the H3 worldline fit does not beat the S2-boundary control.

Historical 64k combined artifact before the stricter dedicated H3 reconstruction cap net:

```text
runs/e2_h3_worldline_freezeout_64k_20260602/e2_kms_freezeout_cl_screen_64k_seed20260628_5de1901d
  final Phi: 0
  run time on local CPU path: 52.9 s with workers=1, inner_jobs=4
  BW/KMS Lorentz receipt: true
  conformal H3 chart receipt: true
  static defect-cluster H3 support receipt: true
  defect timeline snapshots: 8
  persistent screen defect-worldlines: 9
  H3-fitted defect events: 72
  persistent H3 worldlines: 9
  H3 worldline median residual: 0.00268
  S2-boundary median residual: 0.00261
  H3 beats S2-boundary control: false
  bulk_worldline_precursor_receipt: false
  bulk_3d_established: false
  particle_matter_receipt: false
  viewer: plots/oph_realtime_viewer.html
```

Ensemble comparison:

```text
runs/e2_h3_worldline_ensemble_4k_64k_20260602
  4k static defect H3 receipt fraction: 0.0
  64k static defect H3 receipt fraction: 1.0
  4k H3 worldline receipt fraction: 0.0
  64k H3 worldline receipt fraction: 0.0
```

Latest stricter 64k/256k ensemble:

```text
runs/e2_h3_reconcap_worldline_64k_256k_ensemble_20260602/h3_ensemble_report.json
  run_count: 4
  patch_counts: 65536 and 262144
  support_visible_lorentz_all: true
  conformal_h3_chart_all: true
  record_populated_bulk_any_scale: false
  defect_h3_support_any_scale: false
  defect_h3_worldline_any_scale: false
  bulk_3d_established: false
  physical_particle_prediction: false
```

Interpretation: scaling and denser reconstruction caps are useful diagnostics, but they do not yet
populate H3 with observer records or defect worldlines. The next repair-law target is coherent
observer-visible transport/fusion/scattering response, not just larger screen count.

## Execution Order

1. Add neutral reconstruction planted/shuffled controls to every neutral bulk report.
2. Run a 4k neutral reconstruction smoke and inspect whether observer distances carry any dimension signal.
3. Add defect cluster persistence/worldline summaries to the array screen receipts. Done for
   timeline-enabled E2 runs; still needs repeated-seed transport/fusion controls.
4. Add `cmb_lite` candidate temperature fields as diagnostic-only outputs. First shape-only
   comparison exists; candidate-source calibration still open.
5. Run 4k multi-seed diagnostics for neutral bulk and CMB-lite.
6. Run 64k diagnostics after the 4k gates are stable.
7. Use 256k as a single-seed scale check after each meaningful dynamics/readout change.
8. Attempt 1M only after the lower sizes show a useful H3 population or particle scaling signal.
9. Add visualization once neutral reconstruction has a nontrivial controlled dimension signal.
10. Start CAMB/CLASS adapter work only after physical calibration is defined.

## First Serious Claim Gate

All of the following must pass:

```text
BW/KMS gate passes across N
observer-object consensus passes
neutral reconstruction gives a 3D-compatible window
planted 2D/3D/4D controls pass
random/shuffled controls fail
defects localize in reconstructed bulk
C_l candidate is seed-stable
C_l controls do not reproduce it
```

Allowed claim:

```text
The finite OPH regulator emits a controlled observer-record reconstruction with a 3D-compatible
bulk and stable defect/worldline structures, plus a gated freezeout angular-spectrum proxy.
```

Still not allowed:

```text
We predicted the physical CMB.
We simulated the real early universe.
We derived the Standard Model particle spectrum from the simulator.
```
