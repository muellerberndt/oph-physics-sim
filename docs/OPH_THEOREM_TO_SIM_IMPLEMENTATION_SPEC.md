# OPH-FPE Theorem-To-Simulator Contract

This document is the stable contract between OPH paper claims and the finite OPH-FPE simulator. It
describes what the simulator exports, which receipts gate each claim, and which visual or diagnostic
surfaces must remain non-claims. Concrete run outcomes, dated run notes, and intermediate
experiment logs belong in run artifacts, reports, handoffs, or correspondence directories, not here.

## Scope

OPH-FPE instantiates observer-like self-reading systems: bounded finite carriers with local state,
ports or boundaries, readback, records, feedback or repair moves, and public evidence bundles. A
simulator output is a claim only when its explicit receipt gate passes in the concrete bundle. A
false receipt is part of the evidence, not an error to hide.

The simulator is not a substitute for the paper proof stack. Its role is to expose finite-regulator
counterparts of the paper structures, test whether the required gates can be made concrete, and keep
diagnostic visuals separate from physical promotions.

## Receipt Lanes

Use these lanes consistently in reports, payloads, and documentation:

```text
C0a  finite_settle_diagnostic
C0b  finite_consensus_theorem_receipt
R0   finite_record_readback_surface
R1   observer_checkpoint_restoration_receipt
L0   bw_kms_branch_replay_receipt
L1   endogenous_cap_state_receipt
L2   endogenous_modular_generator_receipt
L3   inferred_kms_clock_receipt
L4   support_visible_bw_covariance_receipt
L5   ordered_cut_pair_rigidity_receipt
L6   lorentz_algebra_closure_receipt
L7   refinement_naturality_receipt
H0   conformal_h3_chart_receipt
H1   observer_facing_3p1d_h3_experience_receipt
H1b  observer_facing_populated_h3_experience_receipt
H2   strict_neutral_third_person_bulk_receipt
QV0  fluctuating_quantum_vacuum_diagnostic_view
S0   effective_edge_string_diagnostic_view
S1   critical_edge_cft_receipt
P0   bulk_worldline_precursor_receipt
P1   production_particle_matter_receipt
E0   einstein_branch_entry_receipt
E1   null_generator_stress_charge_receipt
E2   fixed_cap_entropy_stationarity_receipt
E3   small_ball_area_bridge_receipt
E4   all_timelike_tensor_upgrade_receipt
E5   lambda_constancy_conservation_receipt
E6   newton_coupling_forbidden_input_audit_receipt
G2   production_gravity_receipt
CMB0 screen_cmb_diagnostic_receipt
CMB1 finite_primordial_source_kernel_receipt
CMB2 physical_cmb_prediction_receipt
G0   finite_gauge_candidate_sieve
G1   four_dimensional_os_gauge_certificate
BH0  finite_horizon_record_repair_diagnostic_receipt
BH1  black_hole_physical_evaporation_bridge_receipt
BH2  black_hole_qnm_radiative_bridge_receipt
D0   distributed_artifact_packaging_smoke_receipt
D1   distributed_kernel_scaling_readiness_receipt
```

Legacy aliases may remain in JSON for compatibility, but new docs and UIs should present the
canonical lane names above.

## Implemented Surfaces

The current simulator has these stable surfaces:

- Finite patch/screen runs with `Z2`, `S3`, and clock-group fixtures.
- A small exact finite-consensus harness under `tools/verify_small_universe.py` and
  `configs/sou_v1_icosa12.yml`.
- Screen microphysics reports for local pixel scale, port budget, support-visible S2 cellulation,
  edge-sector heat-kernel diagnostics, central-record/Born diagnostics, and observer checkpoint
  restoration.
- BW/KMS and cap-flow reports that distinguish branch replay from endogenous finite Lorentz
  theorem contracts.
- Observer-local modular-time reports, subjective observer camera payloads, H3 chart readouts, and
  observer-object population diagnostics.
- Strict neutral-bulk frontier reports, kept separate from theorem-assisted H3 chart readouts.
- Screen holonomy clusters, defect timelines, H3 worldline fits, proto-particle reports, and
  particle-likeness diagnostics.
- Screen-level CMB and comparable-observation diagnostics, with physical-CMB promotion gates
  closed unless finite source, transfer, and likelihood receipts pass.
- Leech/moonshine endpoint bridge audits for the fine-structure hadronic endpoint gap, with the
  OPH-QCD hadronic backend gates separated from endpoint promotion. The two-current spectral
  export is only the running-alpha/HVP marginal; HLbL and rare-decay rows require higher-point and
  transition spectral exports from the same backend.
- JWST compact-object source-release workbench reports for high-redshift compact objects, early
  massive-galaxy candidates, little red dots, and apparently mature black-hole candidates. The
  workbench separates quotient/source/release, compact record surfaces, finite object parents,
  forward mocks, degeneracy audits, abundance selectors, and frozen catalog likelihoods.
- Reference vacuum baselines and compact lattice-gauge baselines, with OPH-native promotion
  receipts false unless paper-side transfer/lift conditions are supplied.
- Shape-substrate and positive-geometry diagnostics, treated as declared substrate or accelerator
  witnesses rather than neutral OPH bulk proofs.
- Distributed universe pack/reducer surfaces with explicit global carrier, partition, cut-interface,
  observer-registry, and run-pack contract receipts.
- Universe-timeline visualization payloads with first-class `visualizationViews` contracts for
  fluctuating quantum-vacuum diagnostics, observer-camera views, effective edge-string views, and
  silence-to-observation views.

## Non-Promotion Rules

These are hard boundaries:

- `final_phi == 0` is not `C0b`; finite consensus needs strict descent, confluence, local-diamond,
  repair-completeness, and uniqueness receipts.
- A declared or replayed `2pi` BW branch is `L0`, not the full `L1-L7` finite Lorentz theorem
  contract.
- The theorem-assisted H3 chart is not strict neutral third-person bulk. It is `H0/H1`; chart-blind
  neutral reconstruction is `H2`.
- `subjectiveObserverCameras` are visible-readout cameras, not hidden global cameras.
- `visualizationViews.fluctuatingQuantumVacuum` is a finite screen/readback fluctuation view, not a
  literal QFT vacuum or physical CMB prediction.
- `visualizationViews.effectiveStringTheory` is a schematic edge-cycle/worldsheet diagnostic view,
  not a critical string CFT, heterotic worldsheet derivation, or production-particle proof.
- H3 proto-worldlines and screen holonomy defects are not matter particles until `P1` passes.
- `visualizationViews.emergentCurvedSpacetime` is a quotient-visible H3 source/compaction
  diagnostic, not production gravity. `G2` requires `E0`, and `E0` requires
  `einstein_bridge_manifest.json` plus all theorem-tagged bridge sidecar receipts.
- Screen `C_l` and CMB-lite curves are `CMB0` diagnostics, not `CMB2`.
- A Leech, c=24, or moonshine artifact is not a fine-structure prediction unless it emits the
  OPH-QCD hadronic backend from source data: QCD quotient ensemble, source QCD parameter map,
  Euclidean slab/vacuum transfer, hadronic Hilbert quotient, Ward-normalized current ledger,
  two-current spectral export, same-scheme remainder, systematics ledger, and no-target-leak DAG.
  Full hadronic precision also requires higher-point and transition spectral exports. A decimal
  near `alpha^{-1}=137.036` is not a promotion receipt.
- The Q4 hadronic backend scaffold is emitted with
  `oph-fpe hadron-source-backend --out <run>/hadron_source_backend`. Its default claim is
  `SOURCE_PROTOTYPE_NOT_PROMOTED` at tier `H2`; this is the correct state until a source QCD law,
  Ward current ledger, spectral exports, and no-target-leak certificates are actually populated.
- The Q3 JWST compact-object workbench is emitted with commands such as
  `oph-fpe jwst-object-source-artifact --out <run>/jwst/source` and
  `oph-fpe jwst-compact-object-simulation-plan --run-dir <run> --out <run>/jwst/plan`.
  Its default claim is `J0_DIAGNOSTIC_PROXY`. Compactness, red color, luminosity, broad lines,
  source release, and catalog counts do not become mass, age, assembly, black-hole mass,
  physical abundance, or OPH confirmation unless the required degeneracy, parent, source,
  forward-operator, and frozen-likelihood receipts pass.
- A finite black-hole archive, finite reconstruction threshold, island-gain score, or repair
  spectrum is not a physical Page curve, geometric island, evaporation channel, or QNM/ringdown
  prediction. `BH0` records horizon/collar record-repair diagnostics only. `BH1` requires exterior
  time, radiation entropy, physical microstate coverage, flux closure, no-leakage and no-remnant
  controls, and refinement stability. `BH2` requires an exterior background bridge, radiative
  quotient, continuum operator convergence, future-horizon and future-null-infinity boundary rows,
  Bondi/asymptotic readout, and frozen detector comparison gates.
- Reference Gaussian/free-field or compact-`U(1)` baselines do not become OPH-native vacuum or
  primordial-field claims without explicit promotion receipts.
- Shape-substrate and positive-geometry reports do not alter trusted OPH repair/readout outcomes
  unless fail-closed equivalence and provenance receipts pass.

## Finite Consensus Contract

`C0b` requires a theorem-eligible finite carrier and an independent replayable verifier. The small
universe harness is the reference fixture. A larger run can claim `C0b` only if it emits:

- strict descent or no-op accounting for every theorem-phase repair;
- zero accepted positive-score moves;
- disjoint commutation checks;
- local-diamond checks for overlapping repairs;
- repair-completeness checks at terminal states;
- randomized fair-schedule confluence;
- unique terminal normal form or a declared finite set of terminal quotient fibers;
- source hashes, config hash, replay tables, and bundle checksums.

Exploration runs may emit `C0a` and still keep `C0b` false.

## Lorentz And H3 Contract

The paper-route Lorentz/H3 claim is layered:

- `BWRec_r`: the issue #308 finite cap-normal BW receipt object
  `(CapNormal_r, Frame_r, Order_r, Support_r, CrossRatio_r, Matrix_r, KMS_r, Error_r)`.
  It is audited by `issue-308-bw-certificate` and emits exactly one of `BW0`, `BW1`, `BW2`, or
  `BW3`. The BW3 tier is recomputed from primitive fields and a passing refinement envelope; it
  must not trust caller-provided `bw_passed` or `tier` booleans.
- `CAP_NORMAL_H3_CHART_RECEIPT`: the issue #309 cap-normal H3 chart receipt. It recomputes
  `q(Omega)=(1,Omega)`, analytic or globally certified round-cap normals
  `n_C=(cot(alpha),csc(alpha)c)`, the signed boundary incidence rule,
  Lorentz residuals, `n_{gC}=Lambda_g n_C`, H3 future-sheet residuals, and distance invariance
  from primitive fields. Fitted caps without a global round-cap certificate are approximate and
  cannot emit theorem-certified chart status.
- `MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT`: the issue #310 record-populated H3 localization
  receipt. It recomputes the record-conditioned response inverse from primitive response fields:
  cap normals, `B`, `W`, frame rank, `sigma_min(W^1/2 B)`, compact source domain, net radius,
  calibrated responses, total error, residual minimizer, localization radius
  `R_H[(L/alpha)epsilon+(2/alpha)sigma+(1/alpha)tau]`, and `Delta_loc`. It may emit a unique
  finite point only when `Delta_loc > 0`; otherwise the output is `H3_LOCALIZATION_AMBIGUOUS`.
  Pre-existing H3 point clouds, object packets, or viewer coordinates are controls or diagnostics,
  not independent localization receipts.
- `L0`: the simulator replayed a declared BW/KMS branch and controls did not trivially reproduce it.
- `L1-L2`: the cap state and modular generator were built from observer-visible finite record data
  without using hidden geometry or the target flow.
- `L3-L4`: the `2pi` coefficient was inferred from held-out support-visible action, not inserted as
  the tested answer.
- `L5-L6`: ordered cut-pair maps satisfy rigidity, composition, inverse, null-cone, and Lorentz Lie
  algebra checks.
- `L7`: refinement and naturality persist across fixed regulator families without threshold tuning.
- `H0`: the conformal H3 chart is emitted from the support-visible Lorentz branch.
- `H1`: observer-local readouts experience the H3 chart through visible records and modular time.
- `H1b`: persistent observer objects populate the H3 chart under controls.

Only the conjunction of the relevant `L` receipts plus a BW3 issue #308 certificate should be
presented as a theorem-aligned finite cap-net BW receipt. Only a passing
`CAP_NORMAL_H3_CHART_RECEIPT` should be presented as the issue #309 chart theorem receipt. Only a
passing `MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT` should be presented as the issue #310
record-populated observer-facing H3 localization receipt. `H0/H1` alone are useful paper-route
diagnostics and visualizations. A renderer cap, fitted boost, finite cap-ID permutation, dimension
estimate, coefficient near `2pi`, or existing H3 coordinate file is not a theorem receipt without
the primitive `BWRec_r`, cap-normal H3 chart fields, and modular-response localization fields.

## Strict Neutral Bulk Contract

Strict neutral bulk is the chart-blind audit. It must not receive screen coordinates, cap axes, H3
coordinates, theorem depth, known icosahedral coordinates, or geometry-derived proximity labels.

An `H2` candidate must use visible local observer data:

- local port or boundary packet hashes;
- overlap correspondences;
- record and checkpoint order;
- transition counts by port pair and lag;
- perturbation-response and repair-current tensors;
- first-passage or response-time observables;
- local-port permutation controls.

Producer-side observer rows emit the local evidence fields consumed by the strict neutral extractor:

- `local_boundary_packet_hash_histogram` / `boundary_packet_hash_histogram`
- `local_overlap_correspondence_histogram` / `overlap_correspondence_histogram`
- `port_pair_lag_histogram` / `transition_port_pair_lag_histogram`
- `local_repair_current_tensor` / `repair_current_tensor`
- `local_perturbation_response_tensor` / `perturbation_response_tensor`
- `local_first_passage_histogram` / `first_passage_time_histogram` / `response_time_histogram`

These are chart-blind local summaries. They do not make `H2` pass by themselves because the
quotient-geometry, control, ancestry, and refinement gates remain separate closed receipts.

The receipt requires quotient-safe geometry, held-out predictive reconstruction, planted controls,
shuffled/random controls, metric-safe missingness, partition/presentation distortion bounds,
refinement tail modulus, and train/validation/test ancestry separation.

## Particle, Gauge, And String Contracts

Proto-particle diagnostics can use screen holonomy clusters, collar-like defect tracks, and H3
worldline fits. `P1` requires localized shared-bulk support, stable topological/sector charge,
contractible-path transport, fusion conservation, scattering reproducibility, speed/causality
controls, observer-resampling stability, and refinement stability.

Gauge and Standard Model claims are not supplied by toy `S3` sectors. `G0` may sieve finite
candidate packets. `G1` requires a four-dimensional OS/gauge certificate and must remain separate
from visualization labels.

Effective string visualization starts at `S0`: finite edge cycles, repair-history ribbons,
collar/defect tracks, and H3 proto-worldlines may be rendered as an edge-string/worldsheet
diagnostic. `S1` requires a critical-edge receipt suite, including finite current algebra,
Virasoro/Sugawara, supercurrent, spin-structure, anomaly, and modular-invariance checks.

## Einstein Branch-Entry And Gravity Contract

The simulator must not treat curved-spacetime visuals, stress-pair contraction, organic defect
motion, or apparent attraction as a proof of gravity. Those lanes may emit diagnostic source,
compaction, and curvature fields for rendering, but production-gravity wording is closed unless the
E0 bridge manifest passes.

`E0` names the OPH5 recovered-core Einstein bridge manifest. Generic
finite-consensus promotion is outside this receipt. The paper-side theorem
discharge is recorded in `einstein_bridge_manifest.json` via
`EINSTEIN_BRIDGE_DEPENDENCY_DISCHARGE_RECEIPT=true` and provenance tags such as
`S2_screen=AXIOM_1`, `BW_2pi=THEOREM_4_2`, `BoundedInterval=LEMMA_E0_5`,
`RemainderControl=LEMMA_E0_6`, `AllTimelikeCoverage=LEMMA_E0_7`,
`StressClosure=PROPOSITION_E2`, and `Lambda=D6_NCRC_CLOSURE`.

The run-specific branch-entry receipt remains fail-closed. `EINSTEIN_BRANCH_ENTRY_RECEIPT` is true
only when the manifest has all required theorem-tagged sidecars:

- `sphere_fold_receipt.json`
- `bw_receipt.json`
- `null_stress_receipt.json`
- `bounded_interval_receipt.json`
- `fixed_cap_entropy_receipt.json`
- `small_ball_area_receipt.json`
- `remainder_receipt.json`
- `timelike_coverage_receipt.json`
- `stress_closure_receipt.json`
- `lambda_closure_receipt.json`
- `newton_forbidden_input_receipt.json`
- `einstein_residual_receipt.json`

`G2` production gravity requires `E0` plus the relevant matter/source and metric-solution receipts.
Until then:

- `raw_production_gravity_requested` may be true only as a diagnostic lane signal;
- `production_gravity_receipt`, `physical_gravity_prediction`, and
  `einstein_equation_solution_receipt` must remain false;
- visualizer gates should show closed/blocked promotion, not red simulation errors.

## CMB And Vacuum Contract

`CMB0` covers screen-level angular spectra, CMB-lite shape comparisons, CMB anomaly diagnostics, and
comparable-observation tables. It is measurement-facing but not physical prediction.

`CMB2` requires finite OPH source arrays, no-data-use dependency graph, finite covariant
collar-packet parent, stress closure, gauge-independent source fields, screen-to-radial or
bulk-to-sky handoff, Boltzmann/CLASS/CAMB transfer, CMB1 custom-parent CDM-limit regression,
Standard-Model-off control regression, solver/recombination/neutrino/tolerance pins,
source-plugin hashes, blinded full-observable likelihood execution, and frozen
source/solver/likelihood hashes. The simulator emits these checks in
`frozen_transfer_likelihood_report.json`; physical prediction receipts remain false unless
that closure report and the physical CMB input contract both pass.

The quantum-vacuum visualization view uses finite screen/readback fluctuations, repair traces,
holonomy residues, and optional reference-vacuum baselines. It must keep
`OPH_NATIVE_VACUUM_PROMOTION_RECEIPT` and physical-CMB receipts visible.

## Distributed Contract

Distributed packs are one declared finite carrier exposed through workers. The pack must emit the
global graph, global initial state, partition map, cut interfaces, observer registry, per-shard
configs, and reducer manifest. `D0` is packaging/export health. `D1` requires online distributed
kernel receipts: linearized committed-event log, restart/rollback roots, seam reciprocity, visible
restriction, atomic commit, local diamond, repair completeness, global quotient hash, holonomy,
schedule independence, and partition naturality.

Cut-link replay metadata is valid for audit and visualization, but it is not live cross-shard
repair unless the online kernel receipts pass.

## Config And Documentation Rules

Tracked configs are curated fixtures. Local/generated configs belong under ignored paths described
in `docs/configuration.md`. Every committed config needs a `claim_boundary` naming the finite
observer-like system, the diagnostic or claim lane, and the receipts required for promotion.

Stable docs should describe contracts, schemas, commands, and claim boundaries. They should not
embed run-specific results, dated success/failure ledgers, local timing baselines, or one-off
experiment paths.
