# Observer-Patch Fundamental Physics Emergence

OPH-FPE is the finite simulator for Observer-Patch Holography physics experiments. It models bounded software patches with local state, ports, readback, records, feedback and repair moves, then writes public evidence bundles that say which claims passed and which gates are closed.

**Scope.** The lattice tests consensus and geometry-emergence claims only. The
relaxation dynamics never consumes the pixel constant P, and no quantitative
physical constant (alpha, n_s, particle masses, Lambda) is produced by the
lattice. Any such number appearing in sim documents is a paper-side comparison
and carries that label; the paper-side surfaces live in
`reverse-engineering-reality/`.

**Consistency-stack mapping.** The structural receipts the simulator earns —
observer-local modular time, the KMS replay contract, the H3 chart, and the
overlap mutual-agreement certificate — are finite instantiations of rows C1–C3
of the consistency stack (`reverse-engineering-reality/docs/CONSISTENCY_STACK.md`,
a companion-repo file). The simulator's open items — strict neutral bulk,
Einstein branch entry, physical particles, physical CMB — sit on C5 and
downstream rows. The simulator emits no ledger-row closures; closure verdicts
live only in `reverse-engineering-reality/docs/CLOSURE_LEDGER.md`. The simulator
syncs those theorem surfaces through hash-pinned, nonpromoting imports and
native recomputation receipts. The current drift audit and exact claim boundary
matrix are in `docs/THEORY_SYNC_AUDIT_2026-07-19.md`.

The package name is `OPH-FPE`: Observer-Patch Fundamental Physics Emergence.

Live visualizations of simulator runs: <https://simulation.floatingpragma.io>
(built by the web coding agent from the visualizer bundles this repo emits;
see `docs/RUN_OUTPUTS_AND_VISUALIZATION.md` and
`scripts/README_FOR_WEB_CODING_AGENT.md`).

The working surface includes finite consensus receipts, observer record algebra, support-visible BW/KMS and H3 chart diagnostics, theorem-assisted H3 object population, strict neutral-bulk frontiers, screen-level CMB diagnostics, finite cosmology certificate gates, defect and proto-particle assays, scale/capacity audits, viewer exports, and handoff bundles.

## Status (2026-07-20)

Local execution reports cover 4k, 16k, 64k, 128k (32,000 observers), 256k,
and 1,048,576 patches with 64,000 materialized observers. The populated
repository-visible earned surface stops at 128k. The directory named for the
1M earned run records that its source was empty when the snapshot was created
(`data/earned_runs/oph_universe_1m_earned/SOURCE_EMPTY.md`). Million-patch
results are local, unpinned observations until their artifacts are committed
or hash-pinned with a fetch path; scale claims cite the 64k and 128k earned
runs.

- The current audited physical H3/KMS instrument is fail-closed.  Its fresh 4k
  v7 acceptance cell passes all 31 replay booleans (21 top-level instrument
  receipts plus ten artifact-byte receipts), exact fresh-process replay, and
  independent family reduction.  P0--P8 are 0 pass, 9 not evaluated, and 0
  valid scientific failures; the reducer sees 1/12 cells and reports
  `ready_for_64k=false`.  Promotion, scale authorization, and retirement are
  false.  The previously reported 3+1D and `2*pi` results are legacy diagnostic
  receipts: `kms_collar_transport_response` embeds a `2*pi` normalization and
  is not the independent modular/geometric clock pair required by the physical
  campaign.  The endogenous perturb-remeasure probe selected `pi` without
  discrimination (scores near 0.93--1.01).  See
  `docs/PHYSICAL_H3_KMS_THEORY_CODE_AUDIT_2026-07-20.md` and the workspace
  `survival-proof-4/status.md`.
- The observer mutual-agreement certificate records self-consistency of the
  shared record under hash-frame conjugation: every evaluated pair re-gauges
  at defect 0.0 from 4k through the 128k earned run (1M pending artifacts, per
  the qualification above), every evaluated observer triple closes the Cech
  cocycle, shuffled controls sit near 0.8. The re-gauging map exists by
  construction for views of one stored record; the discriminating experiment,
  independent per-observer commit histories, is open work.
- Proto-particles are receipted (organic defect population, worldlines,
  two-defect dynamics assays); particle promotion stays gated on
  gauge-covariant fusion transport.
- Strict neutral bulk and the Einstein branch-entry gates E1-E6 stay
  false with executable blockers; the 4k/16k/64k/256k refinement ladder
  and the bw_2pi blocker resolution feed the next audit pass.
- The exact A5 lane recomputes the faithful 12-point action, the icosahedral
  adjacency spectrum, the `1+3+3'+5` module decomposition, the invariant-point-
  partition no-go, and the conditional 15-state exterior-generation witness.
  Its v2 certificate exhausts all 15 unordered fermion pairs (including
  diagonals) against `H` and `Hdag`: exactly three of the 30 representation-
  level candidates are gauge-invariant lines.
  These are structural certificates; the port-current, refinement, descent,
  family, continuum, and QFT gates keep physical Standard-Model promotion false.
- The canonical capacity lane now computes finite public record capacity from
  global sections, joint kernels, exact maximum independent sets, complete
  terminal fibres, carrier projections, and robust zero-slack closure. Observed
  horizon values and regulator sizes are comparison inputs, never producers of
  `N = log M0(U_N)`; legacy screen-capacity booleans are also rejected by the
  physical-CMB source path.
- The SCR330 radial lane emits bounded packet-contract checks for dilation,
  tomography, null-space, and forward-residual evidence. E4 source promotion
  and physical TT/TE/EE remain false until their source and transfer artifacts
  can be independently resolved and replayed.

The living scoreboard, including public-measurement comparisons, is
`docs/OPH_SIGNATURE_EXPERIMENT_TRACKER.md` (at-a-glance tables in
section 0a).

## Receipt Boundaries

The README describes the simulator surface. Pass/fail receipt labels belong in run artifacts under `runs/`, measurement packs, frontier reports, and handoff bundles. Curated immutable snapshots selected for repository publication live under `data/earned_runs/`; their working sources remain under the ignored `runs/` tree. Mutable project progress belongs in GitHub issues.

The theorem-assisted H3 route, strict neutral-bulk route, physical CMB route, production particle route, and production-gravity route are separate claim paths. A diagnostic chart, curved-spacetime compaction field, stress-pair motion, or screen spectrum must not be promoted into a paper-faithful physical claim unless the corresponding receipt gate is present in the concrete run output. Gravity promotion is additionally gated by the post-Lean-audit Einstein branch-entry contract (`EINSTEIN_BRANCH_ENTRY_RECEIPT` / [issue #503](https://github.com/FloatingPragma/observer-patch-holography/issues/503)). `tools/import_oph_artifacts.py` hash-pins the current paper/particle/geometry status into a run or staging directory; imported status is always informational or diagnostic and never flips a run receipt. The current pinned snapshot is under `data/oph_cross_repo_current/`. See `docs/PAPER_PARTICLE_ARTIFACT_INTEGRATION.md`.

For explanatory universe rendering, the simulator also supports an explicit
`simulation_assumptions` lane. It can supply paper bridges such as the BW `2*pi`
branch, H3 record population, an open-slicing dS4 background, and the visual
interpretation of stable defect candidates as matter. These inputs are written
to `simulation_assumption_manifest.json` and use separately named
`SIMULATION_ASSUMED_*` statuses; they never turn computed theorem, neutral-bulk,
particle, Einstein, gravity, or physical-CMB receipts true. See
`docs/SIMULATION_ASSUMPTION_POLICY.md`.

Since 2026-07-14 the perturb-remeasure response probes (BW/KMS branch replay
and transition-scale selection) replay the production sector-repair law
against a per-source local gauge (`repair_production_sector_links`), so the
`2*pi` clock probe is gauge-covariant and production-faithful; the
probes fail closed only when the replay config is absent. Receipts earned
before that date by the pre-covariant probe are superseded. The `2*pi`
selection itself still rests on the `kms_collar_transport_response` source,
whose scale normalization defaults to 2 pi; the endogenous probe does not yet
discriminate among scales, and gate repair is open work.

Every ensemble-facing output must name its claim tier:

- `E0`: seed noise, proposal noise, or repair jitter.
- `E1`: conventional reference ensemble.
- `E2`: OPH-native quotient ensemble.
- `E3`: OPH vacuum.
- `E4`: OPH primordial field.
- `E5`: observable cosmological prediction.

Reference baselines are useful for distributed correctness, visualization calibration, and regression tests, but they do not become OPH-native vacuum or primordial-field claims. The explicit receipts `OPH_NATIVE_VACUUM_PROMOTION_RECEIPT` and `OPH_PRIMORDIAL_FIELD_PROMOTION_RECEIPT` must stay false unless the corresponding paper-side transfer or lift theorem is supplied.

The fail-closed proof-packet audits (issues #307, #308, #309, #310, #361)
are documented in `docs/PROOF_PACKET_AUDITS.md`.

## What The Simulator Tests

Eight families of receipt-gated tests. Artifact-by-artifact detail:
`docs/RUN_OUTPUTS_AND_VISUALIZATION.md`. Lane contracts:
`docs/CLAIM_LANES.md`. Live verdicts against public measurement:
`docs/OPH_SIGNATURE_EXPERIMENT_TRACKER.md` (section 0a).

- **Consensus and observer experience.** Finite overlap repair,
  replayable verifier receipts, observer-local modular time, the
  gauge-covariant 2 pi KMS clock, and the observer mutual-agreement
  certificate (pair re-gauging, Cech cocycle triples, shuffled controls,
  integer-only chart verdicts; `bulk_dimension_claim` null by schema).
- **Spacetime reconstruction.** Cap-normal and conformal H3 charts, H3
  modular-response localization, H3 worldline stitch certificates, strict
  neutral third-person bulk frontiers with the 4k/16k/64k/256k refinement
  ladder, and the Einstein branch-entry gate set E1-E6
  ([issue #503](https://github.com/FloatingPragma/observer-patch-holography/issues/503)).
- **Defects and proto-particles.** S3 holonomy clusters, defect
  timelines and worldlines, organic population receipts, two-defect
  dynamics assays, interaction/fusion proxies, and the fail-closed P1
  particle-promotion contract.
- **Cosmology and CMB.** Freezeout-screen spectra, no-data-use receipts,
  finite-collar Boltzmann bundles, physical-CMB frontier and promotion
  audits, CMB anomaly statistics, and static-galaxy/H0-S8/dark-response
  diagnostics, all behind physical-prediction gates.
- **Parity and chirality diagnostics.** Cross-gradient pseudo-scalar
  with mirror/shuffle controls (Stokes-scoped) and the defect-worldline
  signed-turning statistic with sign-flip nulls.
- **Gauge structure and proof packets.** Exact A5/icosahedral action and
  spectrum certificates, the exhaustive conditional exterior-generation SM
  witness,
  Borel-Weil one-Higgs carrier, Yang-Mills gap certificates, MaxEnt
  I-projection and central-interface MSA receipts, and the fail-closed
  proof-packet audits (`docs/PROOF_PACKET_AUDITS.md`).
- **Messengers and transients.** UHE coefficient emission with
  no-UHE-data-use scans and common-source locks; compact-transient
  receipt ladders for FRBs and old-host sources.
- **Sandboxes and backends.** Fractional Hall/Chern quotient sandbox,
  positive-geometry kernel, the fail-closed W/Z/H boson backend, JWST
  compact-object lane, and reference vacuum baselines with false
  OPH-native promotion receipts by construction.

Every substantive output is receipt-gated. A false receipt is part of the
result, especially for strict neutral bulk, production particles, and
physical CMB.

## Quickstart

```bash
python3 -m pytest -q

python3 -m oph_fpe.cli run --config configs/e0_z2_patchnet.yml --out-dir runs
python3 -m oph_fpe.cli run-array --config configs/e1_s3_modular_screen_4k.yml --out-dir runs
python3 -m oph_fpe.cli run-bw-array --config configs/e1_s3_state_modular_screen_4k.yml --out-dir runs
python3 -m oph_fpe.cli run-bw-array --config configs/e2_kms_freezeout_cl_screen_64k.yml --out-dir runs

python3 -m oph_fpe.cli public-record-capacity --out runs/public_record_capacity
python3 -m oph_fpe.cli a5-sm-structural-certificate --out runs/a5_sm_certificate.json
python3 -m oph_fpe.cli scr330-radial-receipt \
  --source-dag path/to/source_dag.json \
  --receipt SCR330_RADIAL_NULL_REPORT --claim-tier E3 \
  --out runs/scr330_radial_receipt.json
python3 -m oph_fpe.cli edge-center-clock-certificate \
  --evidence path/to/clock_evidence.json --out runs/edge_center_clock.json
python3 -m oph_fpe.cli collar-clause-certificate \
  --packet path/to/collar_packet.json --out runs/collar_clause.json
python3 -m oph_fpe.cli collar-poisson-certificate \
  --packet path/to/collar_poisson_packet.json --out runs/collar_poisson.json
python3 -m oph_fpe.cli fair-block-certificate \
  --packet path/to/fair_block_packet.json --out runs/fair_block.json
python3 -m oph_fpe.cli boundary-fiber-certificate \
  --packet path/to/boundary_fiber_packet.json --out runs/boundary_fiber.json

python3 -m oph_fpe.cli run-oph-universe \
  --config configs/e4_shared_observer_bulk_256k_observers4096_theorem.yml \
  --out-dir runs \
  --run-id oph_universe_256k_observers4096_theorem_local

python3 -m oph_fpe.cli physics-problem-outputs \
  --out-dir runs/physics_problem_outputs_<date_or_run_id>
```

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

python3 -m oph_fpe.cli observer-agreement-report \
  --run-dir runs/<run_id>

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

## Lane Notes

Compact pointers; contracts and prose in `docs/CLAIM_LANES.md`.

- Screen/scale/capacity: `P` normalizes local closure; `N_CRC` is the
  global capacity closure; `finite_simulator_derived_G_SI` stays false
  without a dimensionful bridge.
- BW/H3/bulk: seven receipt-gated layers from finite settling to strict
  neutral bulk; proof-packet audits for
  [issue #308](https://github.com/FloatingPragma/observer-patch-holography/issues/308),
  [issue #309](https://github.com/FloatingPragma/observer-patch-holography/issues/309), and
  [issue #310](https://github.com/FloatingPragma/observer-patch-holography/issues/310)
  recompute the chart theorems from primitive fields.
- CMB/cosmology: screen spectra are diagnostics; physical prediction
  needs the full frozen source/transfer/likelihood contract.
- Defects/particles: screen diagnostics feed the independently
  recomputed P0/P1 promotion contract; legacy producer booleans cannot
  promote.
- Positive geometry: fail-closed optimization layer; safe verdict is
  `GEOMETRY_CERTIFIED_BACKEND_NOT_ENABLED`.
- W/Z/H backend: hash-pinned theorem artifacts, synthetic diagnostic
  config; see `docs/WZH_NUMERICAL_BACKEND.md`.

## Cloud And Reproducibility

Cloud credentials, project IDs, bucket names, account IDs, tokens, and keys belong in `.env.local`, shell exports, or cloud-native identity. They do not belong in committed files.

Curated example configs live in `configs/`. Local or generated configs belong under ignored paths
such as `configs/local/`, `configs/generated/`, or `*.local.yml`; see `docs/configuration.md` for
the format and claim-boundary rules, and `docs/GCP_SCALING_PLAN.md` for provider boundaries and
sizing notes.

For CPU sweeps, cap BLAS fan-out per worker:

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python3 -m oph_fpe.cli run-bw-sweep --configs configs/e1_s3_state_modular_screen_4k.yml --seeds 1,2,3,4 --out-dir runs
```

Physical H3/KMS campaign cells additionally freeze the numerical stack and
reject a mismatch before source evolution.  Use the project virtual
environment and all five caps:

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
.venv/bin/python -m oph_fpe.bulk.physical_h3_kms_campaign \
  ../survival-proof-4/outputs/run_4k_acceptance_20260720_v7 \
  --rung 4096 --execute-physical-cell
```

Reduce one or more exact cell bundles independently with:

```bash
.venv/bin/python -m oph_fpe.bulk.physical_h3_kms_aggregate \
  ../survival-proof-4/outputs/physical_h3_kms_family_aggregate_v7.json \
  ../survival-proof-4/outputs/run_4k_acceptance_20260720_v7
```

These commands prove instrument/replay/reducer acceptance only.  The physical
launcher refuses every rung above 4k unless repeated
`--prerequisite-run-dir` arguments supply all lower-rung cells and fresh family
reduction marks the requested rung ready.  The current aggregate does not
authorize a 16k or 64k physical launch.

## Key Documentation

Status and experiments:

- `docs/WHAT_OPH_FPE_DOES.md`: one-page account of what the simulator
  computes, publishes, and leaves gated.
- `docs/OPH_SIGNATURE_EXPERIMENT_TRACKER.md`: the living experiment tracker
  (at-a-glance verdict tables, anomaly docket, results log).
- `docs/RUN_OUTPUTS_AND_VISUALIZATION.md`: full run-output inventory with the
  visualizer-bundle selection marked.
- `docs/BEST_OF_PUBLIC_DATA_COMPARISONS.md`: the provenance-bound public-data
  comparison suite.
- `docs/SCALING_MILESTONE_ESTIMATES_2026-07-13.md`: scale milestones and
  empirical lower bounds.

Contracts and lanes:

- `docs/OPH_THEOREM_TO_SIM_IMPLEMENTATION_SPEC.md`: theorem-to-code and
  claim-promotion contract.
- `docs/CLAIM_LANES.md`: lane-by-lane contracts (screen/scale, BW/H3/bulk,
  CMB, defects, positive geometry, W/Z/H).
- `docs/STRING_VACUUM_SELECTION_RECEIPT_CONTRACT.md`: candidate, augmented
  rank/isolation, branch-coverage, and catalogue receipt requirements for the
  fail-closed string-vacuum lane.
- `docs/STRING_VACUUM_SIMULATOR_TARGETS.md`: canonical observable values,
  machine-readable receipt targets, dependency scopes, promotion rules, and
  simulator replay commands for the string-vacuum lane.
- `docs/PROOF_PACKET_AUDITS.md`: fail-closed proof-packet audits
  (issues #307, #308, #309, #310, #361) and Lean-mirrored fixtures.
- `docs/SIMULATION_ASSUMPTION_POLICY.md`: the assumed-bridge visualization
  lane and its `SIMULATION_ASSUMED_*` statuses.
- `docs/PAPER_PARTICLE_ARTIFACT_INTEGRATION.md`: cross-repo artifact
  import and hash-pinning.

Configuration, visualization, operations:

- `docs/README.md`: documentation policy and the grouped doc index.
- `docs/configuration.md` and `configs/README.md`: config format and the
  curated config inventory.
- `docs/oph_universe_timeline_visualization_payload_v1.schema.json`,
  `docs/oph_visualizer_pack_v2.schema.json`,
  `docs/oph_distributed_universe_visualization_payload_v1.schema.json`:
  visualizer payload schemas.
- `docs/VISUALIZATION_APP_AGENT_MANUAL.md`: app-agent manual for the
  payload-driven visualizations; `scripts/README_FOR_WEB_CODING_AGENT.md`:
  run-agnostic bundle contract and claim-boundary checklist.
- `docs/GCP_SCALING_PLAN.md`: scaling and cloud operations (fleet ensembles,
  monolithic sizing, parallelism, credentials policy).
- `docs/small_oph_universe_v1.md`: exact finite-consensus calibration
  harness.
- `REPRODUCTION.md`: reproducibility notes.
