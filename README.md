# oph-physics-sim

OPH-FPE is the finite simulator for Observer-Patch Holography physics experiments. It models bounded software patches with local state, ports, readback, records, feedback and repair moves, then writes public evidence bundles that say which claims passed and which gates are closed.

The package name is `OPH-FPE`: Observer-Patch Fundamental Physics Emergence.

The working surface includes finite consensus receipts, observer record algebra, support-visible BW/KMS and H3 chart diagnostics, theorem-assisted H3 object population, strict neutral-bulk frontiers, screen-level CMB diagnostics, finite cosmology certificate gates, defect and proto-particle assays, scale/capacity audits, viewer exports, and handoff bundles.

## Receipt Boundaries

The README describes the simulator surface. Pass/fail receipt labels belong in run artifacts under `runs/`, measurement packs, frontier reports, and handoff bundles. Mutable project progress belongs in GitHub issues.

The theorem-assisted H3 route, strict neutral-bulk route, physical CMB route, production particle route, and production-gravity route are separate claim paths. A diagnostic chart, curved-spacetime compaction field, stress-pair motion, or screen spectrum must not be promoted into a paper-faithful physical claim unless the corresponding receipt gate is present in the concrete run output. Gravity promotion is additionally gated by the post-Lean-audit Einstein branch-entry contract (`EINSTEIN_BRANCH_ENTRY_RECEIPT` / issue #503). `tools/import_oph_artifacts.py` hash-pins the current paper/particle/geometry status into a run or staging directory; imported status is always informational or diagnostic and never flips a run receipt. The current pinned snapshot is under `data/oph_cross_repo_current/`. See `docs/PAPER_PARTICLE_ARTIFACT_INTEGRATION.md`.

For explanatory universe rendering, the simulator also supports an explicit
`simulation_assumptions` lane. It can supply paper bridges such as the BW `2*pi`
branch, H3 record population, an open-slicing dS4 background, and the visual
interpretation of stable defect candidates as matter. These inputs are written
to `simulation_assumption_manifest.json` and use separately named
`SIMULATION_ASSUMED_*` statuses; they never turn computed theorem, neutral-bulk,
particle, Einstein, gravity, or physical-CMB receipts true. See
`docs/SIMULATION_ASSUMPTION_POLICY.md`.

Every ensemble-facing output must name its claim tier:

- `E0`: seed noise, proposal noise, or repair jitter.
- `E1`: conventional reference ensemble.
- `E2`: OPH-native quotient ensemble.
- `E3`: OPH vacuum.
- `E4`: OPH primordial field.
- `E5`: observable cosmological prediction.

Reference baselines are useful for distributed correctness, visualization calibration, and regression tests, but they do not become OPH-native vacuum or primordial-field claims. The explicit receipts `OPH_NATIVE_VACUUM_PROMOTION_RECEIPT` and `OPH_PRIMORDIAL_FIELD_PROMOTION_RECEIPT` must stay false unless the corresponding paper-side transfer or lift theorem is supplied.

The issue #361 continuum bridge is represented by `oph_fpe.scale.reference_tower` and
`docs/oph_issue361_certificate_schema.json`. A passing finite reference-tower identity check is
only the finite-regulator gate. Continuum correlations, BW modular convergence, Lorentzian
unitarity, and Yang-Mills identification remain closed or conditional until the emitted
certificate includes Cauchy envelopes, transported-state/cutoff bounds, positive-transfer plus
transfer-tower convergence, and the four-dimensional OS/gauge certificate.

Issue #307 has a separate fail-closed collar-CMI audit. Run
`python3 -m oph_fpe.cli issue-307-collar-cmi-decay --source <primitive.json> --out <report.json>`.
The audit recomputes five clauses: finite-range Gibbs evidence, uniform strong conditional matrix
mixing, regional CMI in nats, the boundary-prefactored exponential bound, and the sharp scaling
margin

```text
delta / xi - log(|partial C|_UV).
```

The ratio `delta / ell_UV -> infinity` does not pass the scaling gate by itself. Caller-provided
pass flags are ignored. A passing `ISSUE_307_COLLAR_CMI_DECAY_FINITE_RECEIPT` is a finite
branch-instantiation sanity check. It does not certify the continuum limit, turn local packet CMI
into regional quantum CMI, produce a stress tensor, or open the Einstein gate.

Issue #308 is represented by the finite cap-normal BW certificate audit:

```text
BWRec_r = (
  CapNormal_r,
  Frame_r,
  Order_r,
  Support_r,
  CrossRatio_r,
  Matrix_r,
  KMS_r,
  Error_r
)
```

Use `python3 -m oph_fpe.cli issue-308-bw-certificate --source <BWRec_r.json> --out <report.json>`
to recompute BW0-BW3 from primitive fields. A renderer cap, fitted boost, finite cap-ID
permutation, or numerical coefficient near `2*pi` is not a BW theorem receipt. BW3 requires the
primitive fields and a passing refinement error envelope; the audit ignores producer-supplied
`bw_passed` or `tier` booleans.

## What The Simulator Emits

Main run families:

- finite patch graphs and vectorized screens with `Z2`, `S3`, and clock groups;
- explicit screen ports, local pixel closure, screen-capacity readouts, and scale-bridge reports;
- finite overlap mismatch, annealed repair, stability-window records, readback hashes, and replayable verifier receipts;
- Borel-Weil one-Higgs carrier receipts for `H_OPH = H^0(CP1,O(1)) ~= C^2`, with Higgs mass, quartic, weak scale, and Coleman-Weinberg claims explicitly out of scope;
- support-visible cap/collar states, diagnostic collar Markov reports, the fail-closed issue #307 regional CMI audit, BW/KMS branch replay, transition-scale selection, and H3 chart receipts;
- fine-to-coarse MaxEnt I-projection reports with relative-entropy closure defect, moment residual, Hessian floor, trace residual, and Pinsker bound, plus a separate noncommutative central-interface MSA receipt with Bell/noncentral negative controls;
- observer objects, observer-local modular time, H3 object-population reports, and theorem-assisted consensus readouts;
- strict neutral frontiers with overlap-native controls, graph sweeps, residualized graph sweeps, independent rank-selector audits, and closed promotion gates;
- freezeout-screen angular spectra, CMB frontier reports, no-data-use receipts, finite certificate schemas, and Boltzmann/likelihood gate reports;
- screen holonomy clusters, defect timelines, H3 worldline fits, interaction proxies, particle-likeness reports, and controlled planted-defect assays;
- fractional Hall and fractional Chern sandbox receipts through `oph_fractional`: material quotient presentations, source-freeze gates, Hamiltonian promotion certificates, topological ledgers, optical modules, line-fan identifiability, binding-drift bounds, no-target-leak audits, quotient-correctness checks, and generated `fractional_quotient_report` bundles;
- high-energy messenger coefficient-emission receipts: finite MaxEnt source coefficients for shared hidden compact-engine source laws, no-UHE-data-use dependency scans, moment-polytope and feature-minimality gates, and common-source locks across neutrino, cosmic-ray, and gamma forward channels;
- compact-transient receipt ladders for FRBs, old-host compact sources, and black-hole recycling, with detector thinning, censoring, marked-catalog likelihood, repeater-history, control-model, refinement, simulator-accuracy, promotion, and no-generation-leakage gates separated from physical claims;
- H3 record-worldline stitch certificates that require a declared hyperboloid atlas, real cut interface, observer-time adjacency, sector/gauge transport, ID-independent assignment gap, and coarse/fine contraction before emitting a cross-boundary continuation receipt;
- static galaxy, neutrino, H0/S8, dark-response, CMB anomaly, and compressed-likelihood diagnostics with physical-prediction gates separated from diagnostics; there is currently no OPH-derived neutrino-mass default, the conventional CAMB baseline is 0.06 eV, and the rejected weighted-cycle triple is opt-in and non-promoting;
- viewer bundles for screen/repair playback, object-H3 displays, universe timelines, CMB/neutral frontiers, and scale-compressed outputs.

Every substantive output is receipt-gated. A false receipt is part of the result, especially for strict neutral bulk, production particles, and physical CMB.

## Quickstart

```bash
python3 -m pytest -q

python3 -m oph_fpe.cli run --config configs/e0_z2_patchnet.yml --out-dir runs
python3 -m oph_fpe.cli run-array --config configs/e1_s3_modular_screen_4k.yml --out-dir runs
python3 -m oph_fpe.cli run-bw-array --config configs/e1_s3_state_modular_screen_4k.yml --out-dir runs
python3 -m oph_fpe.cli run-bw-array --config configs/e2_kms_freezeout_cl_screen_64k.yml --out-dir runs

python3 -m oph_fpe.cli run-oph-universe \
  --config configs/e4_shared_observer_bulk_256k_observers4096_theorem.yml \
  --out-dir runs \
  --run-id oph_universe_256k_observers4096_theorem_local

python3 -m oph_fpe.cli physics-problem-outputs \
  --out-dir runs/physics_problem_outputs_<date_or_run_id>
```

Reference vacuum baselines:

```bash
python3 -m oph_fpe.cli reference-vacuum-baseline \
  --out runs/reference_vacuum_baseline \
  --ell-max 16 \
  --sample-count 256 \
  --smoothing-sigma 0.05
```

This writes a direct-sampled free-scalar harmonic Gaussian baseline, a compact-`U(1)` lattice-gauge reference sampler, deterministic replay receipts, smoothing provenance, finite-mode refinement diagnostics, and false OPH-native promotion receipts. Semantic-stream replay and canonical serial-chain replay are reported separately from pathwise partition invariance, which remains false unless a concrete commuting-event or transaction-serialization receipt passes.

Lean-mirrored consensus fixture:

```bash
python3 -m oph_fpe.cli rule90-consensus-fixture \
  --out runs/rule90_consensus_fixture_report.json
```

This writes an exact finite Rule-90 carrier receipt mirroring the Lean consensus
fixtures: good boundary-fiber uniqueness, bad-boundary failure, nontrivial
gauge equivalence, and the local H1-H2-H3 repair no-go witness. It is a
regression contract for simulator receipt logic, not a physical prediction and
not a runtime dependency on Lean.

Matscheko proof-chain import gate:

```bash
python3 -m oph_fpe.cli matscheko-proof-chain-import \
  --out runs/matscheko_proof_chain_import_report.json
```

This writes a compact receipt for the additional finite audit imports: finite
modular flow/KMS, the scalar channel bridge, twelve-port surface bookkeeping,
the QBFT quorum caveat, the two-P provenance, the chi-nu G9/G10 gates, and the
v10 Rule-90 theorems T38--T41 (parity splitting, power-of-two adjacent-pair
universality, and sharp lightlike-diagonal screens). The Rule-90 entries remain
finite binary audit-fixture status imports, not microscopic physics or
spacetime-causality claims. G9 is false when no record-DeltaS to gravity-DeltaS
calibration is supplied.

Parallel BW sweeps:

```bash
python3 -m oph_fpe.cli run-bw-sweep \
  --configs configs/e1_s3_bw_screen_4k.yml \
  --seeds 20260601,20260602 \
  --out-dir runs
```

CPU planning:

- `run-bw-sweep` fills available CPUs when `--workers` and `--inner-jobs` are omitted.
- `OPH_FPE_CPUS=<N>` overrides detected CPU count.
- For a single `run-bw-array`, set `bw.n_jobs: auto` and `cosmology.angular_power.n_jobs: auto` in the config.
- The hot path is CPU/RAM bound: NumPy, SciPy, KD-tree geometry, harmonic estimators, CAMB where installed, and receipt aggregation.

## Distributed Universe Runs

The distributed runner prepares one finite OPH carrier before it writes shard configs. Each shard remains an observer-like self-reading system: bounded local state, ports or boundaries, readback, records, feedback or repair moves, and public receipts. The pack emits `global_graph.npz`, `global_initial_state.npz`, `partition_map.json`, `cut_interfaces.json`, and `global_observer_registry.json` under `global_carrier/`. Shard configs then receive owned nodes, ghost nodes, cut-edge IDs, and the manifest-declared carrier artifact paths. Atlas centers are visualization coordinates only.

Generated distributed packs belong under `distributed/` and are ignored by Git. Commit the kernel code, base configs, docs, and reducer logic, not generated shard YAMLs.

Prepare a distributed pack:

```bash
python3 -m oph_fpe.cli prepare-distributed-oph-universe \
  --config configs/e4_shared_observer_bulk_256k_observers4096_theorem.yml \
  --out-dir distributed/<run_id> \
  --run-id <run_id> \
  --shard-count 64 \
  --patch-count-per-shard 65536 \
  --observers-per-shard 256 \
  --worker-count 8 \
  --seam-halo-width 2048
```

Run the generated worker scripts locally or on cloud workers, then reduce the shard receipts:

```bash
python3 -m oph_fpe.cli reduce-distributed-oph-universe \
  --manifest distributed/<run_id>/distributed_universe_manifest.json \
  --shard-root runs/<run_id>/shards \
  --out-dir runs/<run_id>/reduced
```

The reducer emits a distributed summary, carrier-contract report, cut-link metadata replay,
schema-valid visualization payload, `simulation_assumption_manifest.json`, a hash-verified
`oph_visualizer_pack_v2.tar.zst` under the same strict sub-256M gate, and
`DISTRIBUTED_RUN_PACK_CONTRACT.json`, plus sidecar report directories for
`global_carrier_contract/`, `halo_exchange_global/`, `strict_neutral_global/`,
`observer_modular_time_global/`, `proto_particles_global/`, `pn_resonance_global/`, and
`physical_cmb_global/`. The distributed payload carries the same H3 coordinate contract and
explicit assumed-dS4/observer-frame/defect-matter renderer lane as a monolithic export. The cut-link
replay is endpoint-interpolated metadata for audit and visualization; it is not live cross-shard
repair and is marked `physics_receipt_eligible: false`.

Use the run-pack contract as the small-scale-first gate before launching cloud jobs. `distributed_artifact_packaging_smoke_receipt` checks packaging/export health plus the exact global-carrier manifest, path, run/config/code-hash, partition, cut-interface, stable-initial-state, and observer-registry receipts. `distributed_kernel_scaling_readiness_receipt` and the legacy `large_scale_cloud_run_ready_receipt` remain false until the online distributed kernel emits a linearized committed-event log, restart/rollback roots, final monolithic normal-form/readout certificates, seam packet reciprocity, visible restriction, repair descent, atomic commit, local diamond, repair completeness, selected-fiber nontrivial elimination, same-boundary multistart confluence, quotient normal-form canonical hash, holonomy, fair-block contraction, schedule independence, and partition naturality receipts. The hash is an equality receipt after quotienting, not a selector among distinct physical endpoints.

The CMB reducer preserves shard-local source statistics as diagnostics, but it does not average nonlinear shard estimates into physical inputs. Physical CMB promotion requires a pooled sufficient-statistic reducer, screen-to-radial lift receipt, Boltzmann transfer, CDM-limit regression, and frozen likelihood comparison. The P/N reducer similarly distinguishes `all_shards_local_scale_compressed_pn_witness_receipt` from `global_pn_resonance_receipt`; the latter requires a finite global readback map and fixed-point solve.

The intended large-run claim is conservative: one declared global finite carrier presented through distributed workers, with cut interfaces visible in the evidence bundle. A strict single neutral third-person 3D bulk requires the normal neutral-bulk gates plus genuine online cut repair, common quotient transport, and global neutral reduction receipts. The strict neutral report embeds a quotient-geometry contract receipt: raw rows must descend to terminal quotient IDs, atlas transport and feature descent must be certified, missingness must be metric-safe, presentation and partition distortions must be bounded, refinement must have a tail modulus, and train/validation/test ancestry must be split by generative batch. Without that receipt, rank/model/leakage diagnostics remain diagnostics.

## Claim-Gate Commands

```bash
python3 -m oph_fpe.cli bulk-proof-certificate \
  --run-dir runs/<run_id> \
  --out runs/<run_id>/bulk_proof_certificate_report.json

python3 -m oph_fpe.cli strict-neutral-bulk-frontier \
  --report runs/<run_id>/neutral_3d_bulk_audit_report.json \
  --out runs/<run_id>/strict_neutral_bulk_frontier

python3 -m oph_fpe.cli physical-cmb-frontier \
  --run-dir runs/<run_id> \
  --out runs/<run_id>/physical_cmb_frontier

python3 -m oph_fpe.cli cmb-promotion-ledger \
  --run-dir runs/<run_id> \
  --out runs/<run_id>/cmb_promotion_ledger

python3 -m oph_fpe.cli compact-transient-audit \
  --out runs/<run_id>/compact_transient_audit

python3 -m oph_fpe.cli uhe-emit-coefficients \
  --out runs/<run_id>/uhe_coefficient_emission

python3 -m oph_fpe.cli borel-weil-higgs-carrier \
  --out runs/<run_id>/borel_weil_higgs_carrier

python3 -m oph_fpe.cli h3-worldline-stitch-certificate \
  --source runs/<run_id>/h3_worldline_stitch_source.json \
  --out runs/<run_id>/h3_worldline_stitch_certificate_report.json

python3 -m oph_fpe.cli export-measurement-pack \
  --run-dir runs/<run_id> \
  --include runs/physics_problem_outputs_<date_or_run_id> \
  --out runs/<run_id>/measurement_pack

jq '{neutral_3d_bulk_data_bundle_written,physical_cmb_data_bundle_written,physical_cmb_output_tables_written,physical_cmb_source_arrays_written,physics_problem_outputs_all_notes_registered}' \
  runs/<run_id>/measurement_pack/claims.json
```

Targeted POFT direct-carrier assay:

```bash
PYTHONPATH=. python3 tools/audit_poft_transport_emission.py \
  --state fresh_4k runs/<fresh-4k>/s3_gauge_state.npz \
  --state replication_64k runs/<replication-64k>/s3_gauge_state.npz \
  --out runs/poft_transport_emission/report.json
```

The observable is frozen as the edge-average of the natural three-label
permutation representation. The assay compares scale- and unitary-invariant
singular ratios with POFT (T_0,T_1), checks for an independently exported
complex oriented amplitude and coarse/fine intertwiner, and fails closed when
either is absent. It must not construct a complex lift by fitting POFT itself.

Selected cosmology and scale gates:

```bash
python3 -m oph_fpe.cli screen-capacity-report --out runs/screen_capacity_closure
python3 -m oph_fpe.cli pn-resonance-report --out runs/pn_resonance
python3 -m oph_fpe.cli leech-endpoint-bridge --out runs/leech_endpoint_bridge
python3 -m oph_fpe.cli fractional-quotient-report --out runs/fractional/quotient_sector_sandbox
python3 -m oph_fpe.cli jwst-object-source-artifact --out runs/jwst_compact_object/source
python3 -m oph_fpe.cli jwst-compact-object-simulation-plan \
  --run-dir runs/jwst_compact_object \
  --out runs/jwst_compact_object/plan
python3 -m oph_fpe.cli scale-bridge-report --out runs/pn_scale_bridge_no_bridge
python3 -m oph_fpe.cli repair-scale-closure --out runs/repair_scale_closure
python3 -m oph_fpe.cli finite-certificates --out runs/finite_certificates
python3 -m oph_fpe.cli finite-covariant-collar-parent \
  --source runs/<run_id>/finite_covariant_parent_source.json \
  --out runs/<run_id>/finite_covariant_collar_packet_parent_report.json
python3 -m oph_fpe.cli frozen-transfer-likelihood \
  --run-dir runs/<run_id> \
  --out runs/<run_id>/frozen_transfer_likelihood \
  --solver CAMB \
  --solver-version-pin <camb-version> \
  --source-plugin-hash sha256:<source-plugin-hash>
python3 -m oph_fpe.cli physical-cmb-promotion-audit --run-dir runs/<run_id> --out runs/cmb_promotion_audit
python3 -m oph_fpe.cli official-planck-readiness --out runs/official_planck_readiness
```

`leech-endpoint-bridge` audits a candidate Leech/moonshine same-scheme hadronic endpoint artifact.
It keeps the fine-structure endpoint prediction receipt false unless a separate paper-side
promotion gate is supplied.

The JWST compact-object commands are mirrored on the paper-stack side by
`reverse-engineering-reality/code/particles/jwst/build_compact_object_source_release_receipts.py`;
both surfaces keep the default claim at `J0_DIAGNOSTIC_PROXY`.

The compact-transient audit is mirrored on the paper-stack side by
`reverse-engineering-reality/code/particles/compact_transients/build_compact_transient_receipts.py`;
both surfaces keep the default claim at `CR2_CONDITIONAL_PHENOMENOLOGY`.

The physical CMB gate remains closed unless finite source arrays are backed by a
finite covariant collar-packet parent with stress closure, recipient stress for
nonzero repair exchange, gauge-independent \(B_A\), convergence/CDM-limit
receipts, CMB1 custom-parent CDM-limit regression, Standard-Model-off control
regression, blinded full-observable likelihood execution, and frozen
source/solver/likelihood hashes. CAMB/CLASS-compatible curves without those
reports are diagnostic plumbing.

Viewer exports:

```bash
python3 -m oph_fpe.cli run-viewer --run-dir runs/<run_id> --out runs/<run_id>/oph_realtime_viewer.html
python3 -m oph_fpe.cli run-object-h3-viewer --run-dir runs/<run_id> --out runs/<run_id>/object_h3_bulk_viewer.html
python3 -m oph_fpe.cli run-cmb-neutral-frontier-viewer --run-dir runs/<run_id> --out runs/<run_id>/cmb_neutral_frontier_viewer.html
```

## Screen, Scale, And Capacity

`P` is carried as the local pixel/cell closure value. In finite screen runs it normalizes cell area, cell entropy `P/4`, cap capacity, and residual weights.

`N_CRC` is carried as the global screen-capacity closure value. Regulator patch counts such as `4096`, `65536`, `262144`, and `1048576` are sampling counts unless a dedicated capacity readback map and terminal normal-form enumerator close the finite capacity gate.

The scale reports write dimensionless-invariant and independent-bridge receipts. `finite_simulator_derived_G_SI` remains false without an accepted dimensionful scale bridge and finite capacity proof.

## BW, H3, And Bulk

The simulator separates these layers:

```text
finite settling and finite consensus: receipt-gated
BW/KMS 2*pi branch replay: receipt-gated
cap-normal H3 chart theorem: primitive-field receipt-gated
conformal Lorentz/H3 chart: receipt-gated
observer-facing H3 object population: receipt-gated
finite Lorentz theorem contract: receipt-gated proof contract
strict neutral third-person bulk: receipt-gated frontier
```

Issue #309 is represented by `CAP_NORMAL_H3_CHART_RECEIPT`. It recomputes
`q(Omega)=(1,Omega)`, `n_C=(cot(alpha), csc(alpha) c)`, boundary incidence,
Lorentz equivariance `n_{gC}=Lambda_g n_C`, and the future-sheet `H3` checks
from primitive chart fields. A sampled/fitted display without a global
round-cap certificate is `CAP_NORMAL_H3_CHART_APPROXIMATE`, not theorem
evidence.

```bash
python3 -m oph_fpe.cli cap-normal-h3-chart \
  --source runs/<run_id>/cap_normal_h3_chart_source.json \
  --out runs/<run_id>/cap_normal_h3_chart_report.json
```

Issue #310 is represented by `MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT`. The
allowed path is:

```text
record projector
-> R_i(C,t,O)
-> calibrated response vector
-> conditioned residual inverse
-> H3 localization ball
```

The audit recomputes cap-frame rank and singular values, compact-domain/net
metadata, total error, the residual minimizer, the localization radius, and
`Delta_loc`. A unique finite point requires `Delta_loc > 0`; otherwise the
state is `H3_LOCALIZATION_AMBIGUOUS`. Existing H3 point fields, viewer
coordinates, or object packets may be injected controls or ground truth for a
self-consistency test, but they cannot set `H3LOC=true` by themselves.

```bash
python3 -m oph_fpe.cli modular-response-h3-localization \
  --source runs/<run_id>/modular_response_h3_localization_source.json \
  --out runs/<run_id>/modular_response_h3_localization_report.json
```

The older chart receipt records the theorem-level dimension count
`H3 = SO+(3,1)/SO(3)`, hence `6-3=3`, before any finite neutral point-cloud
estimator is consulted. The old graph-distance and modular-lift dimension
estimates are debug diagnostics. Bulk claims flow through BW/KMS,
support-visible cap/collar data, the cap-normal/conformal H3 chart receipts,
observer records, object population, and neutral-bulk gates.

## CMB And Cosmology

Screen-level angular spectra are measurement-facing diagnostics. They are useful for seed/control studies and viewer payloads.

Physical CMB prediction has a stricter contract: finite OPH sources for amplitude, scalar quotient, IR selectors, finite-collar kernels, recovery rates, Boltzmann handoff, CMB1 custom-parent CDM-limit regression, Standard-Model-off control regression, frozen solver assumptions, blinded comparison setup, and official full-observable likelihood execution. Those gates are reported by the frozen transfer/likelihood closure, physical CMB frontier, and promotion-audit outputs for each concrete run.

Related commands include:

```bash
python3 -m oph_fpe.cli cmb-lite-compare --run-dir runs/<run_id> --benchmark runs/benchmarks/COM_PowerSpect_CMB-TT-binned_R3.01.txt
python3 -m oph_fpe.cli cl-from-freezeout-npz --run-dir runs/<run_id> --out runs/<run_id>/cl_recomputed
python3 -m oph_fpe.cli oph-screen-power --run-dir runs/<run_id> --out runs/screen_power
python3 -m oph_fpe.cli cmb-anomaly-report --run-dir runs/<run_id> --source-dir runs/<run_id> --out runs/cmb_anomaly
python3 -m oph_fpe.cli dark-sector-simulation-plan --run-dir runs/<run_id> --out runs/<run_id>/dark_sector_simulation_plan
python3 -m oph_fpe.cli physical-cmb-output-comparison --run-dir runs/<run_id> --out runs/physical_cmb_output_comparison
```

The dark-sector simulation plan is an integration receipt. It reads the static
galaxy, finite covariant parent, finite-collar Boltzmann bundle, Boltzmann-input,
CMB anomaly, and frozen likelihood reports, then names the first blocked
promotion gate and the next simulator command to run. It is not a dark-matter
prediction or likelihood by itself. Exact `exp(-P/24)` collar-coefficient
promotion is tracked separately through local-reserve, scalar-weighted z6-mean,
and uniform product-thickening gates; finite-thickness profile coefficients
remain the default unless those receipts close.

## Defects And Particles

The screen-holonomy layer writes defect clusters, timelines, interaction proxies, H3 worldline fits, and particle-likeness reports. These are screen/collar diagnostics. Production P1 is independently recomputed as P0 proto-worldline evidence AND a classical carrier-mode receipt AND a quantum Hilbert/spectral/asymptotic receipt, with deconfinement required for colored candidates. Legacy producer booleans cannot promote it.

Useful commands:

```bash
python3 -m oph_fpe.cli controlled-defect-assay --out runs/controlled_defect_assay
python3 -m oph_fpe.cli shape-dodeca-smoke --config configs/shape_dodeca_vertex_smoke.yml --out-dir runs
python3 -m oph_fpe.cli shape-ensemble --config configs/shape_dodeca_ensemble.yml --seeds 1,2,3,4 --out-dir runs
```

## Positive Geometry Kernel

The amplituhedron and positive-geometry checker is a fail-closed optimization layer. The trusted path remains finite patches, records, mismatch, accepted repair, quotient normal forms, observer readout, and evidence receipts.

```bash
python3 -m oph_fpe.cli positive-geometry-kernel-report --out runs/positive_geometry_kernel
```

Normal runs can request the checker through:

```yaml
kernels:
  positive_geometry:
    enabled: true
```

The expected safe verdict is `GEOMETRY_CERTIFIED_BACKEND_NOT_ENABLED` unless OPH sector recognition, native geometry certification, observer-readout equivalence, resource accounting, provenance hashes, and fallback receipts all pass for the concrete sector.

## W/Z/H Numerical Backend

The fail-closed boson backend consumes hash-pinned D10/D11 theorem artifacts
and computes source-clock gaps, frozen affine RG controls, BRST-block
determinants, and Rouché pole enclosures:

```bash
python3 -m oph_fpe.bosons \
  --config configs/bosons/wzh_source_closure_diagnostic_v1.yml \
  --out runs/bosons/wzh_source_closure_diagnostic_v1
```

The tracked config is synthetic and diagnostic. Actual W/Z/H predictions
require the independently frozen carriers, source clock, physical RG packet,
BRST-complete kernels, identities, uncertainties, and prospective provenance
listed in `docs/WZH_NUMERICAL_BACKEND.md`.

## Cloud And Reproducibility

Cloud credentials, project IDs, bucket names, account IDs, tokens, and keys belong in `.env.local`, shell exports, or cloud-native identity. They do not belong in committed files.

Curated example configs live in `configs/`. Local or generated configs belong under ignored paths
such as `configs/local/`, `configs/generated/`, or `*.local.yml`; see `docs/configuration.md` for
the format and claim-boundary rules. See `docs/cloud.md`, `docs/parallel_cloud_plan.md`, and
`docs/digitalocean_pool_setup.md` for provider boundaries and sizing notes.

For CPU sweeps, cap BLAS fan-out per worker:

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python3 -m oph_fpe.cli run-bw-sweep --configs configs/e1_s3_state_modular_screen_4k.yml --seeds 1,2,3,4 --out-dir runs
```

## Key Documentation

- `docs/README.md`: simulator documentation policy and stable-doc index.
- `docs/configuration.md`: config format, tracked-fixture policy, and local-config ignore rules.
- `configs/README.md`: current curated config inventory.
- `docs/OPH_THEOREM_TO_SIM_IMPLEMENTATION_SPEC.md`: single simulator state, theorem-to-code,
  paper-alignment, and claim-promotion contract.
- `docs/oph_universe_timeline_visualization_payload_v1.schema.json`: visualizer payload schema, including fluctuating-vacuum, observer-camera, and effective-string view contracts.
- `docs/VISUALIZATION_APP_AGENT_MANUAL.md`: app-agent manual for producing the quantum-vacuum,
  observer-camera, effective-string, repair, H3, and CMB diagnostic visualizations from the payload.
- `docs/small_oph_universe_v1.md`: exact finite-consensus calibration harness.
- `REPRODUCTION.md`: reproducibility notes.
