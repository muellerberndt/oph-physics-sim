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
N0   finite_public_record_capacity_receipt
N1   complete_terminal_fibre_scalarization_receipt
N2   unique_regulator_stable_zero_slack_receipt
N3   physical_horizon_capacity_receipt
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
SCR0 source_screen_radial_diagnostic_receipt
SCR1 source_dilation_or_tomography_receipt
SCR2 physical_sky_transfer_receipt
G0   finite_gauge_target_conformance_diagnostic
G1   four_dimensional_os_gauge_certificate
A50  exact_a5_icosahedral_structural_receipt
A51  conditional_exterior_sm_witness_receipt
A52  physical_standard_model_from_screen_receipt
BH0  finite_horizon_record_repair_diagnostic_receipt
BH1  black_hole_physical_evaporation_bridge_receipt
BH2  black_hole_qnm_radiative_bridge_receipt
D0   distributed_artifact_packaging_smoke_receipt
D1   distributed_kernel_scaling_readiness_receipt
WZH0 boson_numerical_diagnostic_receipt
WZH1 source_clock_scale_receipt
WZH2 frozen_rg_matching_receipt
WZH3 brst_complex_pole_receipt
WZH4 full_source_only_wzh_prediction_receipt
```

Legacy aliases may remain in JSON for compatibility, but new docs and UIs should present the
canonical lane names above.

## Implemented Surfaces

The current simulator has these stable surfaces:

- Finite patch/screen runs with `Z2`, `S3`, and clock-group fixtures.
- A small exact finite-consensus harness under `tools/verify_small_universe.py` and
  `configs/sou_v1_icosa12.yml`.
- An exact finite public-record-capacity evaluator: global sections and source-supplied joint
  kernels, compound-graph maximum independent sets, complete terminal fibres, carrier projection,
  scalarization, robust fixed-point closure, and the unique zero-slack gate.
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
- An exact A5/icosahedral structural lane and conditional exterior-generation witness with an
  exhaustive 30-row one-Higgs candidate ledger, with
  physical Standard-Model promotion kept separate and fail-closed.
- An SCR330 source-screen radial lane with Mellin-window, tomography, null-space, quadrature,
  stability, ancestry, and forward-residual receipts behind an E4/E5 firewall.
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
- Pair re-gauging and Cech closure over one committed shared record certify
  shared-record gauge-chart self-consistency. They do not certify agreement
  between independently produced per-observer commit histories, finite
  consensus, or a strict neutral third-person bulk.
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
  `P1` is recomputed from `particle_promotion_evidence.json`; legacy producer fields such as
  `particle_matter_receipt` and `physical_particle_emergence` are never promotion inputs.
- `visualizationViews.emergentCurvedSpacetime` is a quotient-visible H3 source/compaction
  diagnostic, not production gravity. `G2` requires `E0`, and `E0` requires
  `einstein_bridge_manifest.json` plus all theorem-tagged bridge sidecar receipts.
- Screen `C_l` and CMB-lite curves are `CMB0` diagnostics, not `CMB2`.
- A measured horizon radius, measured cosmological constant, electroweak target, supplied decimal,
  or regulator patch count cannot produce `N`. `N0-N2` require the exact public-record evaluator;
  the 12-port/30-interface packet is a reversible control, not a physical horizon capacity proof.
- A radial SCR330 source receipt is `SCR0/SCR1`, never `SCR2` or `CMB2`. Prior continuation can
  stabilize an inverse problem but cannot promote a source theorem, transfer function, or sky
  spectrum.
- Matching the SM target tuple is a conformance diagnostic, not `A50`, `A51`, or `A52`.
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
  Requesting `SOURCE_INTERVAL_PROMOTED` does not populate those receipts. Promotion additionally
  requires `--evidence-dir <independent-bundle>` with a clean source commit, complete SHA-256
  manifest, typed positivity and systematics evidence, deterministic replay, and a frozen
  no-target-leak audit. The manifest schema is
  `docs/hadron_source_promotion_evidence_v1.schema.json`.
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
- Synthetic W/Z/H determinant zeros are `WZH0` controls only. `WZH1` requires
  the source clock packet; `WZH2` requires the frozen physical RG packet;
  `WZH3` requires source BRST blocks, identities, sheets, residues, and error
  bounds; and `WZH4` additionally requires hash-pinned D10/D11 certificates,
  one strict source branch, no target ancestry, and a prospective claim freeze.
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

## Collar CMI Contract

The issue #307 finite audit is `ISSUE_307_COLLAR_CMI_DECAY_FINITE_RECEIPT`. It is the conjunction
of:

- `FINITE_RANGE_GIBBS_EVIDENCE_RECEIPT`, with hash-pinned local Hamiltonian terms, uniform range,
  norm and degree bounds, and Gibbs reconstruction residuals;
- `STRONG_CONDITIONAL_MATRIX_MIXING_RECEIPT`, with predeclared constants uniform over the stage,
  cap and boundary-condition family;
- `REGIONAL_COLLAR_CMI_EVIDENCE_RECEIPT`, computed from the four regional entropy terms in nats;
- `BOUNDARY_PREFACTORED_CMI_BOUND_RECEIPT`, which recomputes
  `I_upper <= c_mix * boundary_size_uv * exp[-delta/(xi_uv_cells*ell_uv)]`;
- `SHARP_DOUBLE_SCALING_RATE_RECEIPT`, which checks a common family and the full log margin
  `delta/(xi_uv_cells*ell_uv) - log(c_mix*boundary_size_uv)`.

Use `issue-307-collar-cmi-decay --source <primitive.json> --out <report.json>`. Ordinary
two-point clustering, a fitted correlation length from the same rows, the local packet-triplet CMI,
or a caller pass flag cannot satisfy this contract. The output claim level is
`branch_instantiation_sanity`; it never promotes a CMI value to modular source charge, stress,
Einstein branch entry, or a physical claim.

The retained collar clause is a separate positive-source obligation. Gibbs/state assumptions,
relative entropy or CMI bounds, a flux conditional expectation, and modular-centralizer or
diagonal membership do not force it. The simulator recomputes a bounded, hash-pinned retained-family
packet and explicit factorization of every supplied cross-cut coordinate through the declared flux
basis. That earns only `COLLAR_CLAUSE_PACKET_CONSISTENCY_RECEIPT`; caller-created simulation
derivation nodes are not independently resolved, so `COLLAR_CLAUSE_SOURCE_RECEIPT` remains false.
The T0/T1/T2 no-go cases are mandatory negative controls and never promotion inputs.

## Lorentz And H3 Contract

The paper-route Lorentz/H3 claim is layered:

- `FiniteCapBWRec_r`: the issue #308 six-clause geometric/support-flow/
  normalization object. It emits `FC0` through `FC3` after primitive-field and
  refinement-envelope recomputation.
- `MGNS1Rec_r`: the independent modular algebra-state object. It carries the
  common comparison maps, compatible state/vector data, compact-time modular
  residuals, inverse/group-law control, modular support covariance,
  cap-family uniformity, and cofinal Cauchy modulus. Repeated copies of one
  state across levels do not satisfy this object.
- `SUPPORT_VISIBLE_BW_THEOREM_APPLICABLE_RECEIPT`: true exactly when `FC3` and
  complete `MGNS-1` pass with distinct source artifacts and matching tower IDs
  and hashes. Caller-provided pass or tier booleans are ignored.
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

Only the conjunction of the relevant `L` receipts plus the issue #308
same-tower finite-cap/`MGNS-1` pair should be presented as a theorem-aligned
finite cap-net BW receipt. Only a passing
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
controls, observer-resampling stability, and refinement stability. Those conditions form `P0`,
not `P1` by themselves.

The independent particle contract has three lanes. `P0` is the controlled proto-worldline lane.
The classical carrier-mode lane requires a stated background and phase, explicit quadratic action,
positive physical kinetic coefficient, physical projector/constraint reduction, positive reduced
Hamiltonian, and the structural-speed wave kernel. The quantum-particle lane separately requires
positive-energy vacuum quantization, a physical Hilbert space from constraint reduction or BRST,
a nonnegative Kallen--Lehmann measure with positive-residue mass shell, and stable asymptotic/LSZ
hypotheses. A colored candidate must additionally provide a deconfined asymptotic sector. Thus
`P1 = P0 AND classical-carrier AND quantum-particle AND colored-deconfinement-when-applicable`.

Each lane's primitive JSON must byte-match a hash-pinned sidecar below the run directory. The
contract rejects missing primitives, truthy integers or strings in place of booleans, path escapes,
hash mismatches, dirty source provenance, and caller-supplied top-level pass flags. The schema is
`docs/particle_promotion_evidence_v1.schema.json`; audit a run with
`oph-fpe particle-promotion-contract --run-dir <run>`.

Gauge and Standard Model claims are not supplied by toy `S3` sectors or by caller-injected target
tuples. `A50` exactly recomputes the 60-element A5 group, its faithful transitive action on 12
cosets, permutation character `(12,0,0,2,2)=1+3+3'+5`, the invariant icosahedral graph and its
spectrum, both opposite-triplet adjoint restrictions, and the exhaustive no-go for an invariant
point partition of sizes `8+3+1`. That no-go does not contradict the linear-module decomposition.

`A51` is the conditional one-generation witness `Lambda^2(C+W)+Lambda^4(C+W)`: dimensions
`10+5=15`, the five `Q,u_c,e_c,d_c,L` representation rows and hypercharges, and an exhaustive
enumeration of all 15 unordered fermion pairs including diagonals against `H` and `Hdag`.
Among those 30 representation-level candidates, six are nonabelian singlets and exactly three are
also hypercharge-neutral invariant lines: `Q-H-u_c`, `Q-Hdag-d_c`, and `L-Hdag-e_c`. The witness
also recomputes five perturbative anomaly cancellations and even SU(2) Witten parity.
Neither receipt is `A52`. Physical promotion additionally requires native port-current and weak/load
maps, refinement and global-form descent, exterior/Higgs selection, family attachment, exclusion of
extra light sectors, continuum/spin/QFT realization, and the four-dimensional OS/gauge certificate.
Legacy `G0` target matching is explicitly nonpromoting.

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

Each sidecar must be present, use the receipt schema, carry the expected theorem tag, and set its
own accepted key to the literal boolean `true`. Aggregate or legacy manifest booleans cannot stand
in for a missing, malformed, or wrongly tagged sidecar.

`G2` production gravity requires `E0` plus the relevant matter/source and metric-solution receipts.
Until then:

- `raw_production_gravity_requested` may be true only as a diagnostic lane signal;
- `production_gravity_receipt`, `physical_gravity_prediction`, and
  `einstein_equation_solution_receipt` must remain false;
- visualizer gates should show closed/blocked promotion, not red simulation errors.

## CMB And Vacuum Contract

`CMB0` covers screen-level angular spectra, CMB-lite shape comparisons, CMB anomaly diagnostics, and
comparable-observation tables. It is measurement-facing but not physical prediction.

The canonical edge-center clock uses the full-collar derivative `P/24`, the orientation half
`theta=P/48`, `n_s=1-P/48`, and `kappa=P/[48(P-phi)]`. The standalone validator recomputes the
derivative, orientation-half, semigroup-defect, refinement-defect, clock-binding packet consistency,
clean-source-DAG, and generative-P packet gates. Euler's number and numerical target matches are
diagnostics only. The finite-step survival exponent `-log(lambda_2)/Delta t` is a distinct quantity
and must not be relabelled as the infinitesimal source clock. Standalone packets use
`schemas/cosmology/edge_center_clock_receipt.schema.json`; copied receipt booleans, mismatched binding
digests, cyclic ancestry, and measurement/fit/likelihood ancestors fail closed because both the
binding payload and source DAG are canonically rehashed. Root theorem digests are not externally
resolved by this validator, so canonical rehashing proves only internal packet consistency. Until a
resolver independently replays raw finite-run artifacts, `PHYSICAL_CLOCK_BINDING_RECEIPT`,
`INDEPENDENT_FINITE_RUN_REPLAY_RECEIPT`, `EDGE_CENTER_CLOCK_RECEIPT`, and every downstream physical
clock promotion remain false.

The SCR330 source-screen contract (`scr330-radial-v2`) computes Mellin windows and derivatives,
radial projection, unrestricted null-space and finite-window errors, quadrature and stability
checks, ancestry, and a forward residual. It can emit `SOURCE_DILATION` or
`RADIAL_TOMOGRAPHY`; `PRIOR_CONTINUATION` never promotes. Its E4 source outputs have no edge back
from transfer, likelihood, residual, posterior, fit, or observed TT/TE/EE data, and cannot claim an
E5 observable spectrum. The current E5 firewall contract also cannot promote one: the upstream E4
receipt, transfer source, solver assumptions, and frozen likelihood must first be independently
resolved and replayed rather than supplied as digest declarations.

The neutrino lane follows the separate contract in `docs/neutrino_status.md` and
`schemas/cosmology/neutrino_status.schema.json`. OPH currently has no source-derived neutrino mass
prediction. The default `sum_mnu=0.06 eV` input is a conventional CAMB/CLASS reference and must be
tagged `counts_as_oph_prediction=false`. The former weighted-cycle triple is a target-informed branch
rejected by its declared NuFIT 6.1 gate; it may appear only behind an explicit historical-benchmark
opt-in with `public_promotion_allowed=false`.

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
