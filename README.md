# oph-physics-sim

OPH-FPE is the finite simulator for Observer-Patch Holography physics experiments. It models bounded software patches with local state, ports, readback, records, feedback and repair moves, then writes public evidence bundles that say which claims passed and which gates are closed.

The package name is `OPH-FPE`: Observer-Patch Fundamental Physics Emergence.

The working surface includes finite consensus receipts, observer record algebra, support-visible BW/KMS and H3 chart diagnostics, theorem-assisted H3 object population, strict neutral-bulk frontiers, screen-level CMB diagnostics, finite cosmology certificate gates, defect and proto-particle assays, scale/capacity audits, viewer exports, and handoff bundles.

## Receipt Boundaries

The README describes the simulator surface, not the outcome of any particular run. Current pass/fail status belongs in run artifacts under `runs/`, measurement packs, frontier reports, and handoff bundles.

The theorem-assisted H3 route, strict neutral-bulk route, physical CMB route, and production particle route are separate claim paths. A diagnostic chart or screen spectrum must not be promoted into a paper-faithful physical claim unless the corresponding receipt gate is present in the concrete run output.

## What The Simulator Emits

Main run families:

- finite patch graphs and vectorized screens with `Z2`, `S3`, and clock groups;
- explicit screen ports, local pixel closure, screen-capacity readouts, and scale-bridge reports;
- finite overlap mismatch, annealed repair, stability-window records, readback hashes, and replayable verifier receipts;
- support-visible cap/collar states, collar Markov reports, BW/KMS branch replay, transition-scale selection, and H3 chart receipts;
- observer objects, observer-local modular time, H3 object-population reports, and theorem-assisted consensus readouts;
- strict neutral frontiers with overlap-native controls, graph sweeps, residualized graph sweeps, independent rank-selector audits, and closed promotion gates;
- freezeout-screen angular spectra, CMB frontier reports, no-data-use receipts, finite certificate schemas, and Boltzmann/likelihood gate reports;
- screen holonomy clusters, defect timelines, H3 worldline fits, interaction proxies, particle-likeness reports, and controlled planted-defect assays;
- static galaxy, neutrino, H0/S8, dark-response, CMB anomaly, and compressed-likelihood diagnostics with physical-prediction gates separated from diagnostics;
- viewer bundles for screen/repair playback, object-H3 displays, universe timelines, CMB/neutral frontiers, and scale-compressed outputs.

Every substantive output is receipt-gated. A false receipt is part of the result, especially for strict neutral bulk, production particles, and physical CMB.

## Quickstart

```bash
python3 -m pytest -q

python3 -m oph_fpe.cli run --config configs/e0_z2_patchnet.yml --out-dir runs
python3 -m oph_fpe.cli run-array --config configs/e1_s3_modular_screen_64k.yml --out-dir runs
python3 -m oph_fpe.cli run-bw-array --config configs/e1_s3_state_modular_screen_4k.yml --out-dir runs
python3 -m oph_fpe.cli run-bw-array --config configs/e2_kms_freezeout_cl_screen_256k.yml --out-dir runs

python3 -m oph_fpe.cli run-oph-universe \
  --config configs/e4_shared_observer_bulk_256k_observers4096_theorem.yml \
  --out-dir runs \
  --run-id oph_universe_256k_observers4096_theorem_local
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

The distributed runner prepares one unified OPH atlas and splits it into shard-local observer-patch jobs. Each shard remains an observer-like self-reading system: bounded local state, ports or boundaries, readback, records, feedback or repair moves, and public receipts. Shards are not independent toy universes; their configs carry global patch ranges, global observer ranges, atlas centers, and core/halo seam metadata so cross-shard overlaps can be reduced back into one federated universe witness.

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

The reducer emits a distributed summary, cross-shard seam readout, and visualization payload. The intended large-run claim is conservative: a single global atlas/federated large-universe witness with explicit overlap seams. A strict single neutral third-person 3D bulk still requires the normal neutral-bulk gates plus stronger per-cycle halo exchange and global neutral reduction receipts.

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

python3 -m oph_fpe.cli export-measurement-pack \
  --run-dir runs/<run_id> \
  --out runs/<run_id>/measurement_pack
```

Selected cosmology and scale gates:

```bash
python3 -m oph_fpe.cli screen-capacity-report --out runs/screen_capacity_closure
python3 -m oph_fpe.cli pn-resonance-report --out runs/pn_resonance
python3 -m oph_fpe.cli scale-bridge-report --out runs/pn_scale_bridge_no_bridge
python3 -m oph_fpe.cli repair-scale-closure --out runs/repair_scale_closure
python3 -m oph_fpe.cli finite-certificates --out runs/finite_certificates
python3 -m oph_fpe.cli physical-cmb-promotion-audit --run-dir runs/<run_id> --out runs/cmb_promotion_audit
python3 -m oph_fpe.cli official-planck-readiness --out runs/official_planck_readiness
```

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
conformal Lorentz/H3 chart: receipt-gated
observer-facing H3 object population: receipt-gated
finite Lorentz theorem contract: receipt-gated proof contract
strict neutral third-person bulk: receipt-gated frontier
```

The old graph-distance and modular-lift dimension estimates are debug diagnostics. Bulk claims flow through BW/KMS, support-visible cap/collar data, the conformal H3 chart, observer records, object population, and neutral-bulk gates.

## CMB And Cosmology

Screen-level angular spectra are measurement-facing diagnostics. They are useful for seed/control studies and viewer payloads.

Physical CMB prediction has a stricter contract: finite OPH sources for amplitude, scalar quotient, IR selectors, finite-collar kernels, recovery rates, Boltzmann handoff, CDM-limit regression, and official likelihood execution. Those gates are reported by the physical CMB frontier and promotion-audit outputs for each concrete run.

Related commands include:

```bash
python3 -m oph_fpe.cli cmb-lite-compare --run-dir runs/<run_id> --benchmark runs/benchmarks/COM_PowerSpect_CMB-TT-binned_R3.01.txt
python3 -m oph_fpe.cli cl-from-freezeout-npz --run-dir runs/<run_id> --out runs/<run_id>/cl_recomputed
python3 -m oph_fpe.cli oph-screen-power --run-dir runs/<run_id> --out runs/screen_power
python3 -m oph_fpe.cli cmb-anomaly-report --run-dir runs/<run_id> --source-dir runs/<run_id> --out runs/cmb_anomaly
python3 -m oph_fpe.cli physical-cmb-output-comparison --run-dir runs/<run_id> --out runs/physical_cmb_output_comparison
```

## Defects And Particles

The screen-holonomy layer writes defect clusters, timelines, interaction proxies, H3 worldline fits, and particle-likeness reports. These are screen/collar diagnostics. Production particle matter requires a separate receipt, and the particle derivation is work in progress.

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

## Cloud And Reproducibility

Cloud credentials, project IDs, bucket names, account IDs, tokens, and keys belong in `.env.local`, shell exports, or cloud-native identity. They do not belong in committed files.

Cloud template defaults live in `configs/`. See `docs/cloud.md`, `docs/parallel_cloud_plan.md`, and `docs/digitalocean_pool_setup.md` for provider boundaries and sizing notes.

For CPU sweeps, cap BLAS fan-out per worker:

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python3 -m oph_fpe.cli run-bw-sweep --configs configs/e1_s3_state_modular_screen_64k.yml --seeds 1,2,3,4 --out-dir runs
```

## Key Documentation

- `docs/OPH_THEOREM_TO_SIM_IMPLEMENTATION_SPEC.md`: theorem-to-simulator contract.
- `docs/theory_conformance_audit_20260609.md`: paper-stack conformance and open gates.
- `docs/bulk_emergence_status.md`: dated bulk, H3, CMB, and particle receipt splits.
- `docs/cmb_bulk_particle_execution_plan.md`: CMB, neutral-bulk, and particle execution boundaries.
- `docs/cosmo_proxy_results_20260605.md`: screen-spectrum diagnostics and comparison-facing caveats.
- `REPRODUCTION.md`: reproducibility notes.
