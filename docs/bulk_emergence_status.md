# Bulk Emergence Status

Updated: 2026-06-08

## Current 2026-06-08 Receipt Split

The current best 256k reference run now supports a tiered, paper-aligned result:

```text
reference run:
  runs/stage97_particles_defect_threshold_256k_seed20261047_20260607/e4_shared_observer_bulk_256k_particles_seed20261047_6a562ca8

chart-level 3+1D Lorentz/H3: true
theorem-assisted populated H3 chart: true
strict neutral third-person 3D bulk: false
screen C_l proxy available: true
physical CMB prediction: false
production particle matter receipt: false
```

The clean certificate is:

```text
runs/stage97_particles_defect_threshold_256k_seed20261047_20260607/e4_shared_observer_bulk_256k_particles_seed20261047_6a562ca8/bulk_proof_certificate_report.json
```

Interpretation:

```text
T2-T3 pass: finite support-visible BW/KMS 2*pi transport plus Conf+(S2) -> SO+(3,1) -> H3 chart.
T5 passes: observer-facing record objects populate the theorem-side H3 chart under current controls.
T6 fails: neutral observer-record reconstruction does not yet independently recover a strict 3D bulk.
T7 fails: production defects are not yet matter particles.
T9 fails: CMB outputs remain measurement-facing diagnostics/targets, not physical CMB predictions.
```

This resolves the old ambiguity: the simulator can now show a paper-route, theorem-assisted
populated H3 chart. It still has not closed the stricter neutral third-person bulk, particle, or
physical CMB gates.

Current consolidated comparable-data snapshot:

```text
runs/current_measurement_and_3d_snapshot_20260608/comparable_data_snapshot.json
runs/current_measurement_and_3d_snapshot_20260608/comparable_data_snapshot.md
```

The main comparable values in that snapshot are:

```text
paper-theorem H3 spatial dimension mean: 3.0
record-family H3 median residual: 0.047467988266974376
defect-cluster H3 median residual: 0.0021301833513037672
Planck-lite best field: record_signature_smooth_k32
Planck-lite best shape correlation: 0.45478318175733456
Planck-lite best normalized RMSE: 0.8906021881798153
exact CMB target n_s: 0.9648411430307772
exact CMB target q_IR: 0.25
exact CMB target ell_IR: 32.0
neutrino sum m_nu: 0.09001192964464505 eV
H0/S8 branch H0: 67.40002854274209 km/s/Mpc
H0/S8 direct-Jacobi S8 target: 0.79
```

These are comparable diagnostics and target-lane numbers. They are not yet a finite-lattice physical
CMB prediction because the CMB derivation audit remains false:

```text
runs/current_cmb_derivation_audit_20260608/cmb_parameter_derivation_report.json
```

Missing CMB gates include simulator-derived \(\eta_R\), \(q_{\rm IR}\), \(\ell_{\rm IR}\),
large-angle/parity/low-power control separation, small collar CMI, and anomaly kernels.

Updated: 2026-06-05

## Current Answer

We do not yet have a receipt that establishes 3D bulk emergence from observer consensus.

What the current BW path can support is narrower:

```text
finite screen repair settles Phi;
records commit;
correct 2pi cap transport beats implemented controls in the kinematic sanity path;
state-derived cap/collar modular-probe receipts can now be emitted;
observer-facing local/cap readouts are now recorded;
observer-accessible records can be checked for intersubjective overlap agreement.
transition-scale selection receipts now distinguish declared BW sanity from repair-derived response.
KMS/BW collar-transport branch receipts now select 2pi at 4k, 64k, and 256k.
S2 cap normals and the canonical H3 chart can now be emitted as a conformal/Lorentz chart receipt.
support-visible cap-net hot boundary programs are now recorded.
S3 defect clusters and time-resolved defect worldlines now emit H3 support/transport-profile diagnostics.
screen holonomy defects now emit conservative particle-likeness and screen-local interaction diagnostics.
fixed-cutoff screen microphysics receipts now include edge-sector/Casimir, central-record/Born, and checkpoint/restoration reports.
first gated freezeout-screen C_l proxy can be emitted from observer-facing fields.
observer modular-response kernels can now be emitted and fitted into the canonical H3 chart.
```

That is a BW/cap-flow, conformal chart, and observer-consensus receipt, not yet a
record-populated third-person 3D bulk receipt.

Current-code status now separates the layers:

```text
support_visible_lorentz_3p1_kinematics_receipt: true
conformal_h3_spatial_chart_receipt: true
edge_sector_heat_kernel_receipt: true
central_record_born_receipt: true
observer_checkpoint_restoration_receipt: true
record_populated_h3_spatial_receipt: false
record_family_h3_support_receipt: false
defect_cluster_h3_support_receipt: false under the stricter dedicated H3 reconstruction cap net
defect_h3_worldline_precursor_receipt: false
particle_matter_receipt: false
spatial_bulk_3d_reconstruction_receipt: false
modular_response_h3_candidate_receipt: false
```

The first is the finite simulator receipt corresponding to the paper-side mechanism
`lambda_C(2*pi*t)` on the support-visible cap chart. The H3 chart receipt is the deterministic
`SO+(3,1)/SO(3)` spatial chart implied by the conformal/Lorentz branch. The populated-bulk receipt
remains false because observer records/object families and defect trajectories have not yet beaten
S2-boundary and shuffled controls as populated-bulk data. An earlier eight-cap defect support pass
was too weak; the stricter dedicated 32-cap H3 reconstruction net correctly reclassifies the 64k
defect supports as still boundary-like. The simulator now writes these fits as receipts rather than
treating a fractional dimension estimate as the bulk mechanism.

Latest modular-response H3 smoke:

```text
runs/e3_modular_response_h3_smoke_20260605/e3_modular_response_h3_screen_4k_seed20260670_0fd0348b
  patch_count: 4096
  final Phi: 0
  support-visible Lorentz/BW receipt: true
  observer modular-response kernel written: true
  response matrix: 48 observers x 256 cap/time/field features
  response std: 0.2437299732
  geometry cache: KD tree built, 64 cap-transport maps reused
  H3 median residual: 0.4603321916
  S2-boundary control median residual: 0.5011442287
  shuffled-response control median residual: 0.4604151147
  no-modular-flow control median residual: 0.2395836908
  modular_response_h3_candidate_receipt: false
  bulk_3d_established: false
```

Interpretation: this is the first nondegenerate observer modular-response tensor fitted into the
paper-side H3 chart, but it fails the honest bulk gate. H3 slightly beats the S2 boundary fit on this
single smoke, but it does not beat shuffled response, and no-modular-flow is better. That means the
current record/repair response kernel is not yet a populated 3D bulk signal. The next implementation
target is a more theorem-aligned response source and repeated 4k/64k controls, not a blind 1M scale
run.

Four-seed 4k repeat:

```text
runs/e3_modular_response_h3_4k_sweep_20260605
  seeds: 20260670, 20260671, 20260672, 20260673
  elapsed_seconds: 4.57 with workers=2, inner_jobs=4
  mean H3 residual: 0.4744574760
  mean S2-boundary residual: 0.5117947134
  mean shuffled-response residual: 0.4727385418
  mean no-modular-flow residual: 0.2389668382
  H3 candidate receipts: 0 / 4
```

Corrected object-transition / joint-H3 repeat:

```text
runs/e3_object_transition_h3_4k_sweep_20260605
  seeds: 20260680, 20260681, 20260682, 20260683
  observable_mode: object_transition
  fit_mode: joint_global
  mean response std: 0.8153592304
  mean H3 heldout normalized RMSE: 1.0006216056
  mean S2-boundary heldout normalized RMSE: 0.9998865428
  mean shuffled-response heldout normalized RMSE: 1.0012412889
  mean no-perturbation heldout normalized RMSE: 1.0000000000
  mean best wrong-scale heldout normalized RMSE: 0.9704604464
  H3 candidate receipts: 0 / 4
```

Interpretation: replacing scalar cap-pullback means with signed record-packet transition deltas and
using a joint H3 fit improves the diagnostic, but not the physics receipt. The failure is now
sharper: the finite transition readout has structure, but the H3 chart does not explain held-out
response better than controls. The next target is actual cap/collar perturb-resettle or collar
Markov transition probabilities, not more patches.

Cap/collar perturb-resettle response:

```text
runs/e3_perturb_resettle_h3_phase_4k_sweep_20260605
  observable_mode: perturb_resettle_transition
  seeds: 20260691, 20260692, 20260693, 20260694
  mean H3 heldout normalized RMSE: 0.9762158545
  mean S2-boundary heldout normalized RMSE: 1.0461278685
  mean shuffled-response heldout normalized RMSE: 1.0015168559
  mean no-perturbation heldout normalized RMSE: 1.0000000000
  H3 candidate receipts: 4 / 4

runs/e3_perturb_resettle_h3_64k_dense_20260605/e3_perturb_resettle_h3_screen_64k_dense_seed20260696
  patch_count: 65536
  observer_count: 160
  H3 heldout normalized RMSE: 0.9720827219
  S2-boundary heldout normalized RMSE: 1.0007384439
  shuffled-response heldout normalized RMSE: 1.0005809849
  shuffled-observer-label heldout normalized RMSE: 0.9753893856
  no-perturbation heldout normalized RMSE: 1.0000000000
  wrong 1x/pi/4pi heldout normalized RMSE: 1.0003602125 / 1.0265973843 / 1.0049326046
  modular_response_h3_candidate_receipt: true
```

Interpretation: the simulator now has a first weak H3 candidate receipt from actual cap/collar
perturbation followed by local repair. This is not enough to set `bulk_3d_established=true`: standard
64k observer sampling was mixed, the explained variance is small, and planted 2D/3D/4D plus neutral
bulk controls still have to pass. It is, however, a real improvement over the scalar and
cap-transported packet surrogates.

Latest stricter H3 reconstruction ensemble:

```text
runs/e2_h3_reconcap_worldline_64k_256k_ensemble_20260602
  runs aggregated: 4
  64k runs: 3
  256k runs: 1
  support_visible_lorentz_fraction: 1.0 at both sizes
  conformal_h3_chart_fraction: 1.0 at both sizes
  record_populated_bulk_fraction: 0.0 at both sizes
  record_family_support_fraction: 0.0 at both sizes
  defect_cluster_support_fraction: 0.0 at both sizes
  defect_h3_worldline_precursor_fraction: 0.0 at both sizes
  bulk_3d_established_fraction: 0.0 at both sizes

64k observer cap response:
  median H3/S2 residual ratio ~= 0.92
  median H3/shuffled residual ratio ~= 0.99

256k observer cap response:
  H3/S2 residual ratio ~= 0.93
  H3/shuffled residual ratio ~= 0.99

64k defect H3 worldlines:
  median H3/S2 residual ratio ~= 0.96
  median H3/shuffled residual ratio ~= 0.99

256k defect H3 worldlines:
  H3/S2 residual ratio ~= 0.92
  H3/shuffled residual ratio ~= 0.99

screen particle-likeness diagnostic:
  localized and persistent holonomy worldlines exist
  screen-local transport/fusion/scattering proxies can now be measured
  contractible-path transport and neutral/H3 bulk particle gates are not passing
  particle_matter_receipt: false
```

Latest state-derived smoke:

```text
runs/state_modular_smoke_20260601_gated/e1_s3_state_modular_screen_4k_seed20260621_59124198
  final Phi: 0
  kinematic BW median: 0.3766970215
  state-derived BW median: 1.2101983225
  mandatory negative controls: passed
  state-derived correct-beats-controls gate: failed
  best state-derived control: wrong_1x_normalization
```

This is an important failure, not a cosmetic one. The finite `rho_C/K_a` surrogate currently does
not generate the BW `2*pi` cap speed; its modular transport is closer to the wrong `1x` geometric
normalization. Therefore the simulator still has no 3D-bulk emergence receipt.

Latest transition-response automorphism probe:

```text
runs/transition_response_automorphism_20260602
  4k state-derived median:   2.4468319778341804e-15
  64k state-derived median:  2.493021660556307e-15
  256k state-derived median: 2.3350247185091426e-15
  correct 2pi beats state-derived controls: true
  mandatory negative controls: passed
  numerical_floor_detected: true
```

This is a useful finite BW/cap automorphism receipt, but its claim boundary is narrower: the
transition operator is declared with KMS/BW `2*pi` normalization and the generator is inferred from
that finite perturb/remeasure automorphism. It tests the downstream finite machinery on the intended
BW branch; it does not show that generic observer consensus dynamically discovered the branch.

Latest KMS/BW collar-transport repair:

```text
runs/kms_collar_transport_smoke_20260602/e1_s3_transition_response_screen_4k_1780359037
  primary source: kms_collar_transport_response
  selected scale: 2pi
  KMS source identity fraction: 0.390625
  perturb_remeasure_response: still selects 1x and is identity-degenerate
  repair_affinity_response: still selects 1x
  bulk_3d_established: false

runs/kms_collar_transport_64k_20260602/e1_s3_transition_response_screen_64k_1780359046
  primary source: kms_collar_transport_response
  selected scale: 2pi
  KMS source identity fraction: 0.3125
  perturb_remeasure_response: still selects 1x and is identity-degenerate
  repair_affinity_response: still selects 1x
  bulk_3d_established: false

runs/kms_collar_transport_256k_20260602/e1_s3_transition_response_screen_256k_1780359157
  primary source: kms_collar_transport_response
  selected scale: 2pi
  KMS source identity fraction: 0.421875
  KMS 2pi score: 0.1404455589798214
  perturb_remeasure_response: still selects 1x and is identity-degenerate
  repair_affinity_response: still selects 1x
  bulk_3d_established: false
```

Interpretation: the implementation now supports a branch-accurate OPH KMS/BW collar-transport
receipt. The response is no longer identity-degenerate on that branch, and it selects `2pi` at both
4k, 64k, and 256k. The KMS selector score improves over that ladder:

```text
4k:   0.9126399884019214
64k:  0.42260423715680323
256k: 0.1404455589798214
```

This fixes the finite branch-instantiation problem. It does not prove that the raw repair dynamics
selected the BW branch endogenously; the non-KMS perturb and affinity sources remain reported as
negative controls.

Latest non-circular transition-scale selector smoke:

```text
runs/transition_selection_perturb_smoke_20260602/e1_s3_transition_response_screen_4k_1780358563
  final Phi: 0
  declared geometric sanity source: selects 2pi with score 0
  primary perturb/remeasure response source: selects 1x
  repair response identity fraction: 1.0
  2pi minus best score: 1.1409480960095264
  emergence status: diagnostic_only_transition_response_degenerate

runs/transition_selection_perturb_64k_20260602/e1_s3_transition_response_screen_64k_1780358623
  final Phi: 0
  declared geometric sanity source: selects 2pi
  primary perturb/remeasure response source: selects 1x
  repair response identity fraction: 1.0
  2pi minus best score: 1.2740723255303221
  emergence status: diagnostic_only_transition_response_degenerate
```

Interpretation: the selector is now non-circular. It shows that the present settled repair trace
does not emit a nontrivial endogenous cap response. A bounded cap-local perturb/remeasure probe
also repairs back locally rather than transporting packets around the cap. The declared BW branch
still verifies the automorphism plumbing, but the repair/collar dynamics has not yet selected the
BW `2*pi` transition-response branch by itself.

The remaining 3D-bulk blocker is therefore not "can we represent OPH KMS/BW collar transport?" The
blocker is:

```text
derive or select the BW transition-response branch from raw OPH repair/collar dynamics,
then use observer-record reconstruction to obtain a controlled 3D-compatible dimension window.
```

Latest measurement-facing screen proxy:

```text
runs/cap_net_freezeout_cl_64k_20260602/e2_kms_freezeout_cl_screen_64k_1780386032
  boundary program: support_visible_cap_net_hot
  final Phi: 0
  primary transition source: kms_collar_transport_response
  selected transition scale: 2pi
  state-derived BW median: 2.5597153352365042e-15
  state-derived controls pass: true
  freezeout cycle: 11
  committed fraction: 1.0
  angular estimator: spherical_harmonic
  ell_max: 32
  cosmology gate allowed: true
  neutral reconstruction written: false
  bulk_3d_established: false
  matter_defect_h3_support_receipt: true
```

This run writes a nonnegative harmonic auto-power `C_l` proxy for these observer-facing screen
fields:

```text
record_signature:       peak ell 9,  peak D_ell ~= 2.510891, min control L2 delta ~= 0.998
stable_count:           peak ell 29, peak D_ell ~= 0.034686, min control L2 delta ~= 0.191
cumulative_repair_load: peak ell 32, peak D_ell ~= 0.137558, min control L2 delta ~= 0.749
s3_class_density:       peak ell 29, peak D_ell ~= 0.053203, min control L2 delta ~= 0.473
modular_depth:          peak ell 32, peak D_ell ~= 0.038625, min control L2 delta ~= 0.190
```

The cap-net boundary program gives stronger screen-statistic separation for `record_signature` and
`cumulative_repair_load` than the older iid-hot screen proxy. Several fields still have high
control shape correlations, and there is still no bulk reconstruction gate. This is useful as a
reproducible screen statistic and seed/control target. It is not a CMB prediction, not a CAMB/CLASS
input, and not a 3D bulk receipt.

Latest repeated-seed screen ensemble:

```text
runs/cap_net_cl_ensemble_with_64k_seed_20260602
  runs aggregated: 6
  gate-allowed runs: 6
  4k repeated seeds: 4
  64k repeated seeds: 2
  physical_cmb_prediction: false
  bulk_3d_established: false
```

The cap-net repeated-seed spectra are internally stable for the strongest fields:

```text
4k record_signature:
  peak ell mode 9
  mean pairwise shape corr ~= 0.9977
  median min L2 delta to controls ~= 0.9744

4k cumulative_repair_load:
  peak ell mode 16
  mean pairwise shape corr ~= 0.9459
  median min L2 delta to controls ~= 0.6437

64k record_signature:
  peak ell mode 9
  mean pairwise shape corr ~= 0.9999
  median min L2 delta to controls ~= 0.9980
```

Several other fields still have high correlation to shuffled/random controls. This is therefore a
measurement-facing screen statistic and seed-stability object, not a physical CMB prediction.

First Planck TT shape diagnostic:

```text
runs/cap_net_freezeout_cl_64k_seed20260622_20260602/e2_kms_freezeout_cl_screen_64k_seed20260622_b614e4e7/cmb_lite_comparison_report.json
  benchmark: Planck 2018 TT binned spectrum, release 3
  benchmark rows: 83
  benchmark ell range: 47.7-2499.0
  simulator ell max: 32
  best shape field: s3_class_density
  best normalized RMSE: 0.6417
  physical_cmb_prediction: false
```

This is not a success claim. It is a real-measurement comparison scaffold and currently reads as a
shape mismatch diagnostic.

Standalone receipt viewer:

```text
runs/cap_net_freezeout_cl_64k_seed20260622_20260602/e2_kms_freezeout_cl_screen_64k_seed20260622_b614e4e7/plots/oph_realtime_viewer.html
```

The viewer shows the S2 lattice/screen, observer perspectives, screen defect clusters, H3 support-fit
samples, repair trace, and C_l proxy in one page. Its gate badges still show `bulk_3d_established:
false` and `physical_cmb_prediction: false`.

Timeline-enabled H3-worldline viewer smoke:

```text
runs/e2_h3_worldline_freezeout_4k_20260602/e2_kms_freezeout_cl_screen_4k_seed20260627_30b9fea6
  final Phi: 0
  BW/KMS Lorentz receipt: true
  conformal H3 chart receipt: true
  defect timeline snapshots: 8
  persistent defect-worldline precursors: 76
  max defect-worldline lifetime: 47 cycles
  H3-fitted defect events: 607
  persistent H3 worldlines: 76
  H3 worldline median residual: 0.0176
  S2-boundary median residual: 0.00933
  H3 beats S2-boundary control: false
  H3 beats shuffled control: false
  bulk_worldline_precursor_receipt: false
  defect_cluster_h3_support_receipt: false
  bulk_3d_established: false
  particle_matter_receipt: false
  viewer: plots/oph_realtime_viewer.html
```

This is the first run bundle that can be scrubbed in repair time for the lattice/observer/defect
view and can draw fitted H3 defect-worldline paths. It still does not certify particles because the
H3 fit loses to the S2-boundary control, and transport/fusion, neutral-bulk mapping, and repeated-seed
controls are not complete.

64k scaled H3-worldline run:

```text
runs/e2_h3_worldline_freezeout_64k_20260602/e2_kms_freezeout_cl_screen_64k_seed20260628_5de1901d
  final Phi: 0
  local runtime: 52.9 s with workers=1, inner_jobs=4
  BW/KMS Lorentz receipt: true
  conformal H3 chart receipt: true
  static defect_cluster_h3_support_receipt: true
  defect timeline snapshots: 8
  persistent screen defect-worldlines: 9
  H3-fitted defect events: 72
  persistent H3 worldlines: 9
  H3 worldline median residual: 0.00268
  S2-boundary median residual: 0.00261
  H3 worldline receipt: false
  bulk_3d_established: false
  particle_matter_receipt: false
  viewer: plots/oph_realtime_viewer.html
```

The 64k static defect support now beats S2-boundary and shuffled controls, while the time-resolved
H3 worldline fit still loses narrowly to S2-boundary. That separates two tasks: static defect support
is becoming bulk-chart-like at 64k, but particle-like transport is not yet established.

4k/64k H3-worldline ensemble:

```text
runs/e2_h3_worldline_ensemble_4k_64k_20260602
  support-visible Lorentz all: true
  conformal H3 chart all: true
  static defect H3 support any scale: true
  H3 worldline support any scale: false
  record-populated bulk any scale: false
  bulk_3d_established: false
```

Latest neutral-bulk diagnostics:

```text
runs/kms_neutral_bulk_4k_sweep_20260602
  4k seeds: 4
  BW/KMS gates: 4/4 pass
  planted neutral controls: pass
  shuffled observer-record control: pass
  local-MLE/correlation primary estimates: 3.016 to 3.215
  median primary estimate: 3.165
  screen holonomy clusters: median 240
```

However, after the Pro revision the observer-similarity dimension output is explicitly demoted to a
debug diagnostic. It is not the mechanism that makes space 3D. The current-code single 4k diagnostic
therefore remains useful only for debugging the record-similarity matrix:

```text
runs/kms_neutral_bulk_4k_current_20260602/e2_kms_neutral_bulk_screen_4k_1780382091
  support_visible_lorentz_3p1_kinematics_receipt: true
  spatial_bulk_3d_reconstruction_receipt: false
  local MLE estimate: 3.162
  correlation log-fit estimate: 2.105
  dimension_estimators_agree: false
  candidate_3d_dimension_window: false
```

And the first 64k scale check also fails the spatial-bulk gate:

```text
runs/kms_neutral_bulk_64k_20260602/e2_kms_neutral_bulk_screen_64k_1780374810
  local MLE estimate: 6.278
  correlation log-fit estimate: 2.025
  dimension_estimators_agree: false
  candidate_3d_dimension_window: false
```

Interpretation: the BW/KMS Lorentz branch is now represented as a finite support-visible receipt,
and the conformal cap-normal/H3 chart now gives the deterministic 3D spatial chart. The remaining
bulk task is not to increase spherical pixels blindly or tune a fractional dimension estimator; it
is to fit observer records, repair load, and defects into that H3 chart and show that this populated
chart beats S2-boundary and random controls.

Latest screen defect/particle diagnostic:

```text
runs/e2_paper_microphysics_scale_256k_20260603/e2_kms_freezeout_cl_screen_256k_seed20260641_a4fe9c31
  final Phi: 0
  local runtime: 254.4 s with workers=1, inner_jobs=8
  edge-sector heat-kernel/Casimir receipt: true
  central-record/Born receipt: true
  observer checkpoint/restoration receipt: true
  BW/KMS Lorentz receipt: true
  conformal H3 chart receipt: true
  persistent screen defect worldlines: 6
  screen transport proxy count: 6
  inverse-holonomy fusion candidates: 39
  scattering reproducibility proxy: true
  defect_h3_worldline_precursor_receipt: false
  H3/S2 worldline residual ratio: 0.99
  H3/shuffled worldline residual ratio: 0.99
  bulk_3d_established: false
  particle_like_count: 0
  particle_matter_receipt: false
  viewer: plots/oph_realtime_viewer.html
```

These are not matter particles yet. The new interaction receipt measures screen-local S3 transport,
inverse-holonomy fusion candidates, and class-transition stability, but this is still a
screen/collar diagnostic. Defects become particle-like candidates only after neutral/H3 bulk
localization, contractible-path transport, fusion/scattering controls, and repeated-seed controls
pass. Older eight-cap/coarse-support defect-H3 passes are superseded by the stricter dedicated H3
reconstruction cap-net results above.

Coarse observer-object extraction now exists:

```text
4k current-code object median support size: 4
64k current-code object median support size: 12
record-family H3 support receipt: false
```

This improves the exact-signature fragmentation problem, but record families are still boundary-like:
S2-boundary controls beat H3. Full populated bulk therefore remains false.

## Paper-Guided Target

The compact paper's relativity branch is not a fixed-cutoff dimension-estimator claim. The target is
the support-visible extracted prime geometric cap pair. On that branch, cap modular flow satisfies:

```text
sigma_t^omega = alpha_{lambda_C(2*pi*t)}
```

and the conformal group of the support-visible S2 cap chart gives the Lorentz group:

```text
Conf+(S2) ~= PSL(2,C) ~= SO+(3,1)
```

Therefore the simulator should not declare success because a point-cloud dimension drifts toward 3.
It should declare intermediate success only when:

```text
state-derived BW residual improves under refinement;
correct 2pi beats wrong normalizations and shuffled controls;
collar Markov / recovery errors are carried through refinement;
observer-facing record views settle and agree on overlaps;
neutral observer-record reconstruction gives a robust bulk diagnostic;
controls fail in the expected ways.
```

## Receipt Ladder

### Receipt 0 - finite repair sanity

```text
Phi settles to 0 or a declared residual tolerance.
Records commit.
No fake record rewrite.
Basic S3 orientation tests pass.
```

### Receipt 1 - kinematic BW sanity

```text
lambda_C identity/inverse tests pass.
correct 2pi beats wrong-normalization controls.
shuffled caps/observables fail.
```

### Receipt 2 - collar Markov receipt

```text
A/B/D collar partition is written.
epsilon_cmi is written.
sector-conditioned CMI is written.
r_FR bound is written.
regularizer_a is declared for downstream modular probes.
```

### Receipt 3 - state-derived modular transport

```text
rho_C is built from observer-visible cap/collar state.
K_a = -log(rho_C + aI) is built.
state-derived modular matrix elements are computed.
R_BW compares state-derived transport to lambda_C(2pi*t).
error terms are emitted.
```

### Receipt 4 - refinement scaling

```text
R_BW improves with N.
correct 2pi beats controls at every N.
CMI/recovery errors remain controlled.
bootstrap slope is negative.
```

### Receipt 5 - observer-object consensus

```text
persistent overlap-stable record families are extracted.
observer object consensus report passes.
counterfactual stability report passes.
```

### Receipt 6 - neutral bulk reconstruction

```text
observer-record distance matrix is built without radial-depth lift.
dimension estimators run.
planted 2D/3D/4D controls pass.
shuffled/random controls fail.
```

### Receipt 7 - cautious 3D-bulk diagnostic

Allowed wording:

```text
The finite regulator emits a controlled observer-record reconstruction with a 3D-compatible
dimension window after BW/collar/observer controls.
```

Still not allowed:

```text
We simulated the early universe.
```

### Receipt 8 - first cosmology-like output

Allowed before Receipt 7 only as a diagnostic screen statistic with an explicit gate:

```text
freezeout-screen C_l proxy from observer-facing record fields
```

Allowed as a physical cosmology-like output only after Receipt 7:

```text
calibrated freezeout C_l comparison with physical-unit normalization and full controls
reconstructed-bulk P(k) proxy
Boltzmann-adapter inputs
```

Still not allowed:

```text
full physical CMB prediction
full P(k)
CAMB/CLASS likelihood
Boltzmann adapter claim
```

## New Observer-Facing Outputs

BW runs now write:

```text
observer_views.jsonl
observer_consensus_report.json
observer_objects.jsonl
object_consensus_report.json
boundary_program_report.json
record_populated_h3_report.json
record_family_h3_report.json
defect_cluster_h3_report.json
emergence_status_report.json
```

State-derived BW runs also write:

```text
collar_markov_report.json
bw_state_derived_report.json
transition_scale_selection_report.json
mandatory_controls_report.json
screen_ports.json
```

`observer_views.jsonl` contains two readout types:

```text
patch_observer:
  local finite-neighborhood readout visible to one patch observer

cap_observer:
  cap-local readout with observer-relative modular time grid
```

`observer_consensus_report.json` measures overlap agreement among sampled observer views.

`observer_objects.jsonl` and `object_consensus_report.json` are enabled in the state-derived configs.
They construct persistent overlap-stable record families as observer-facing object surrogates. They
are still a pre-bulk objectivity receipt, not a neutral reconstruction result.

`record_family_h3_report.json` and `defect_cluster_h3_report.json` fit support-visible cap profiles
into the canonical H3 chart and compare against S2-boundary and shuffled controls. A passing defect
cluster report is a matter/particle precursor receipt, not a full bulk receipt.

`emergence_status_report.json` explicitly marks:

```text
bulk_3d_established: false unless record/object populated-bulk or neutral reconstruction gates pass
requires_refinement_scaling: true
requires_neutral_bulk_reconstruction: true
```

This prevents a BW/control receipt from being confused with a completed 3D-bulk emergence claim.

`screen_ports.json` now records explicit named `P0..P11` echosahedral port assignment. The current
4k state-derived smoke has no port overflow.

`mandatory_controls_report.json` now covers `no_repair`, `shuffled_interfaces`,
`random_same_degree_graph`, `wrong_s3_orientation`, and `fake_record_rewrite`. These are necessary
negative controls; passing them only removes obvious false-positive paths.

`transition_scale_selection_report.json` now has three separate meanings:

```text
declared_geometric_sanity:
  checks that the finite automorphism machinery can select 2pi when the BW branch is declared.

kms_collar_transport_response:
  branch-accurate KMS/BW-normalized cap/collar transport after perturb/remeasure.
  This is an OPH branch instantiation, not endogenous selection from raw repair.

perturb_remeasure_response:
  non-circular selector built from bounded cap-local port perturbation,
  local repair, and observer-visible remeasurement.

repair_affinity_response:
  secondary non-circular selector built from observer-visible repair/collar packet affinities.
```

The 2026-06-02 smoke shows `kms_collar_transport_response` selects `2pi`, while
`perturb_remeasure_response` and `repair_affinity_response` still select `1x`. This is a good
branch-instantiation receipt and a continuing endogenous-selection failure receipt.

## 2026-06-05 Perturb-Resettle H3 Status

The simulator now has a stronger but still bounded H3-support receipt:

```text
configs/e3_perturb_resettle_h3_screen_64k_dense.yml
configs/e3_perturb_resettle_h3_screen_256k_probe.yml
```

Dense 64k ensemble:

```text
runs/e3_perturb_resettle_h3_64k_dense_ensemble_20260605
  seeds: 20260701, 20260702, 20260703, 20260704
  final Phi: 0 in all runs
  state-derived BW controls: pass in all runs
  transition scale: 2pi selected in all runs
  H3 candidate receipts: 4 / 4
  mean H3 heldout normalized RMSE: 0.9733434030
  mean H3 heldout explained variance: 0.0525812089
```

256k scale probe:

```text
runs/e3_perturb_resettle_h3_256k_probe_20260605/e3_perturb_resettle_h3_screen_256k_probe_1780630733
  final Phi: 0
  state-derived BW controls: pass
  transition scale: 2pi selected
  H3 candidate receipt: true
  H3 heldout normalized RMSE: 0.9698205272
  H3 heldout explained variance: 0.0594481450
```

The generated viewer:

```text
runs/e3_perturb_resettle_h3_256k_probe_20260605/e3_perturb_resettle_h3_screen_256k_probe_1780630733/oph_receipt_viewer.html
```

correctly reports:

```text
bulk_3d_established: false
physical_cmb_prediction: false
persistent_defect_worldlines: 0
particle_like_count: 0
```

Interpretation:

```text
This is a repeatable modular-response-to-H3 support candidate. It is not yet Receipt 6 or
Receipt 7, because the neutral observer-record reconstruction and planted 2D/3D/4D controls
are still not passing as a single end-to-end bulk-emergence gate.
```

## Observer-Facing Shared Bulk Route

OPH should not treat a hidden God-view 3D lattice as the target. The simulator target is an
observer-facing shared normal form:

```text
observer screen records
-> overlap/quotient consensus
-> cap modular response
-> shared reconstructed geometry
-> public 3D bulk view if controls pass
```

not:

```text
hidden 3D host coordinates
-> observers rendered inside them
```

The run engine now records this distinction in the receipt gates:

```text
bulk_3d_established: false
physical_cmb_prediction: false
particle_matter_receipt: false
```

unless record/object families populate the conformal H3 chart under controls and defect worldlines
also pass the neutral-bulk localization gate.

## 2026-06-05 Shared-Observer-Bulk Smoke

Added:

```text
configs/e4_shared_observer_bulk_4k.yml
```

This config combines:

```text
state-derived BW/KMS cap flow
perturb-resettle modular-response-to-H3 fitting
observer-object extraction
neutral observer-record reconstruction
record-family and defect support H3 controls
S3 defect timeline and particle-likeness diagnostics
freezeout-screen C_l proxy
```

Run:

```text
runs/e4_shared_observer_bulk_4k_20260605/e4_shared_observer_bulk_4k_1780638364
```

Useful receipts:

```text
final Phi: 0
support_visible_lorentz_3p1_kinematics_receipt: true
object_count: 401
persistent_object_count: 401
median_overlap_agreement: 0.5335286458
p10_overlap_agreement: 0.5164930556
median_counterfactual_stability: 1.0
neutral_reconstruction_written: true
mandatory_controls_pass: true
```

Failed gates:

```text
modular_response_h3_candidate_receipt: false
record_populated_h3_spatial_receipt: false
record_family_h3_support_receipt: false
defect_cluster_h3_support_receipt: false
defect_h3_worldline_precursor_receipt: false
particle_matter_receipt: false
bulk_3d_established: false
```

Why the bulk gate remains false:

```text
neutral observer primary dimension: not set
neutral local-MLE dimension: 3.2395869214
record-family H3 residual: 0.0368349879
record-family S2 residual: 0.0072883911
defect-cluster H3 residual: 0.0369028957
defect-cluster S2 residual: 0.0137416135
defect-worldline H3 residual: 0.4977114970
defect-worldline shuffled residual: 0.4980388824
```

Interpretation:

```text
The observer-object layer is now meaningful, but the support-visible objects and defect clusters
still look more like boundary support profiles than populated H3 bulk objects. This is the right
failure mode for the OPH route: stable observer objects exist, but they do not yet populate a
controlled shared 3D bulk.
```
