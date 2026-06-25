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
CMB0 screen_cmb_diagnostic_receipt
CMB1 finite_primordial_source_kernel_receipt
CMB2 physical_cmb_prediction_receipt
G0   finite_gauge_candidate_sieve
G1   four_dimensional_os_gauge_certificate
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
- Screen `C_l` and CMB-lite curves are `CMB0` diagnostics, not `CMB2`.
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

Only the conjunction of the relevant `L` receipts should be presented as a finite Lorentz theorem
contract. `H0/H1` alone are useful paper-route diagnostics and visualizations.

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
