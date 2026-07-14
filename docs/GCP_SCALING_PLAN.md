# GCP scaling plan: 1M patches, 128k observers

Date: 2026-07-14. Target stated by the program owner: one simulation with
1,048,576 patches and 131,072 materialized observers, spread over Google
Cloud CPU infrastructure, each run shipping a visualization bundle.

## What exists

- Prepared monolithic configs (produced by `prepare` tooling, staged under
  `runs/accuracy_audit_{128k,1m}_*/prepared/config.yml`):
  - `e5_shared_observer_bulk_1m_bounded_visualizer`: 1,048,576 patches,
    64,000 materialized observers, deterministic 8,192-observer analysis
    cohort, 128 cycles, bounded viz lanes.
  - `e5_shared_observer_bulk_128k_observers32000_pilot`: 131,072 patches,
    32,000 observers (the pilot for the 1M configuration).
- `prepare-distributed-oph-universe` CLI: emits portable shard configs and
  worker scripts.
- Local reference wall-clock (10-core M-series, 24 GB): 64k full run with
  visualization ~35 min; 4k dense ~4 min. Base loop is O(edges); the
  report/visualization tail dominates above 64k.

## Honest constraint (from the 2026-07-13 audit, unchanged)

The distributed runner has no inter-shard IPC: ghost/halo config fields
are read by no runtime code, seam receipts are written by nothing, and
the transactional repair substrate (`consensus/transactional_repair.py`)
is never driven across shard boundaries. Consequently:

- **Mode A (available today): fleet ensembles.** N independent universes
  (distinct seeds) across N workers. Embarrassingly parallel, zero seam
  work, scientifically valuable immediately: seed-ensemble nulls and
  reproducibility bands for every screen statistic (the VN-04 parity
  verdict was decided by exactly two seeds; a 64-seed band is the proper
  object), plus refinement ladders (4k/16k/64k/256k per seed) for the
  agreement-vs-scale curve.
- **Mode B (blocked on the seam kernel): one sharded universe.** A single
  1M+ patch record sharded across workers with transactional overlap
  repair at seam edges. Blockers, in order: (1) drive
  `transactional_repair` across seam edges with ownership leases;
  (2) range-encoded ownership + npz metadata (the JSON/dict prepare and
  reduce paths break around 1e7 patches); (3) seam receipts written by
  the runtime, verified by the existing checkers.
- The 1M monolithic config needs no sharding: it fits one large-memory
  node. GCP buys parallel throughput and RAM headroom, never correctness,
  until the seam kernel lands.

## Recommended shapes

| Job | Machine | Est. wall-clock | Est. cost |
|---|---|---|---|
| 64k full (viz) | c4-standard-16 (16 vCPU, 60 GB) | ~20-30 min | <$1 |
| 128k/32k pilot | c4-standard-16 | ~1-2 h | ~$1-2 |
| 1M/64k monolithic | c4-highmem-32 (32 vCPU, 248 GB) | ~6-12 h | ~$15-30 |
| 64-seed 64k fleet (Mode A) | 64 x c4-standard-8, preemptible | ~30 min each | ~$10-15 total |
| 1M/128k-observer variant | c4-highmem-64 | ~12-24 h | ~$50-100 |

Preemptible/spot instances suit the fleet; the monolithic 1M run wants an
on-demand highmem node plus checkpoint-resume (the per-cycle snapshot
bound from the audit keeps memory in range; confirm with the 128k pilot's
peak RSS before sizing down).

## Launch mechanics (Mode A, ready to script)

1. Build once: `docker build` on a python:3.12-slim base with the repo and
   `pip install -e .`; push to Artifact Registry.
2. Per worker: `run-oph-universe --config <cfg> --out-dir /out --run-id
   <name>_seed<S>` with the seed overridden per worker; sync `/out` to
   `gs://<bucket>/runs/`.
3. Fan out with Cloud Batch (one task group, 64 tasks, seed = task index
   base + offset); Batch handles retries and machine provisioning.
4. Pull results locally; the analysis cohort tooling and the
   `observer-agreement-report` / parity statistics run on the artifacts
   directly.
5. Bundle: `scripts/build_visualizer_zip.py` per flagship run (256 MB
   hard cap enforced by the builder).

Credentials and project selection sit with the program owner; nothing in
the repo assumes a project id. The plan is executable the moment a
`gcloud` context exists.

## The 128k-observer question

The staged 1M config materializes 64,000 observers and analyzes a
deterministic 8,192 cohort. Doubling materialized observers to 131,072
doubles observer-lane memory and roughly doubles the observer-report
tail; the physics lanes read bounded cohorts either way. Recommendation:
keep 64,000 materialized for the monolithic 1M flagship, and reach
131,072 as 2 x 64k-observer runs in the fleet, or bump
`observer_readback_drive` counts after the pilot's timing report. The
agreement certificate needs overlap density, never raw observer count:
32,000 observers on 131,072 patches gives richer pair statistics than
131,072 observers on 1M patches at fixed support size.


## Credentials and wiring policy (absorbed from cloud.md, 2026-07-14)

Production cloud identifiers stay out of the repo: configure projects,
buckets, tokens, and account ids through `.env.local`, shell exports,
named CLI profiles, or cloud-native identity. Baseline APIs for a GCP
worker: Compute Engine and Cloud Storage; add Cloud Build, IAM, Service
Usage, and Artifact Registry for automated flows.

## Code-level parallelism (absorbed from parallel_cloud_plan.md, 2026-07-14)

The hot path is CPU-bound NumPy/SciPy/KD-tree and graph-array work; no
useful GPU path exists today. Two levels of parallelism: inner cap/time
parallelism inside one run (`bw.n_jobs: auto`,
`cosmology.angular_power.n_jobs: auto`) and outer sweep parallelism
(`run-bw-sweep` fills CPUs when `--workers`/`--inner-jobs` are omitted;
`OPH_FPE_CPUS=<N>` overrides detection). Cap BLAS fan-out per worker
(`OMP_NUM_THREADS=1` etc.). The direct spherical-harmonic `C_l`
estimator is the cost driver for screen-spectrum receipts: use it for
final receipts, the sampled pair proxy or lower `ell_max` for sweeps.
The DigitalOcean fixed-pool runbook was removed 2026-07-14 (git history
`docs/digitalocean_pool_setup.md`); GCP is the provider of record.
