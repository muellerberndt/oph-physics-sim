# Parallel Execution And Cloud Plan

## Current Bottleneck

The current OPH-FPE BW path is CPU-bound:

- spherical point generation;
- kNN / `cKDTree` screen construction;
- vectorized edge repair with NumPy;
- graph-neighbor support-visible smoothing;
- cap/time BW residual tasks using `cKDTree` queries and NumPy interpolation;
- direct spherical-harmonic freezeout `C_l` receipts, when enabled.

The simulator has no useful GPU or TPU acceleration path because the hot path is
SciPy/NumPy/KD-tree and graph-array work. GPU use belongs to CUDA/JAX matrix
kernels, dense cap-state matrix evolution, persistent homology at large scale,
or neural repair policies.

The direct `spherical_harmonic` `C_l` estimator is used for reported single
receipts because it gives nonnegative auto-power. It is CPU-bound and
parallelizes independent target/control spectra with `angular_power.n_jobs`.
Use it for final screen-spectrum receipts; use the sampled pair proxy or lower
`ell_max` for broad seed/config sweeps if harmonic receipts dominate runtime.

## Parallelism

Two levels are available:

1. Inner cap/time parallelism inside one BW run:

```yaml
bw:
  n_jobs: auto
```

2. Coarse process-level sweep parallelism across configs and seeds:

```bash
python3 -m oph_fpe.cli run-bw-sweep \
  --configs configs/e1_s3_bw_screen_4k.yml configs/e1_s3_bw_screen_64k.yml \
  --seeds 11,12,13,14 \
  --workers 4 \
  --inner-jobs 1 \
  --out-dir runs
```

Use either high `--workers` or high `--inner-jobs`, not both, unless the machine has enough cores and
RAM. For seed sweeps, prefer `--workers N --inner-jobs 1`. For one very large run, prefer
`--workers 1 --inner-jobs N`.

`run-bw-sweep` auto-plans CPU use when `--workers` and `--inner-jobs` are omitted:

```text
workers = min(job_count, available_cpus)
inner_jobs = floor(available_cpus / workers)
```

`available_cpus` is `os.cpu_count() - 1` by default, reserving one local CPU for the OS/UI. Set
`OPH_FPE_CPUS=<N>` to override this on cloud VMs or batch schedulers. The sweep planner passes the
same `inner_jobs` budget into both `bw.n_jobs` and `cosmology.angular_power.n_jobs` unless a config
sets an explicit angular-power value. Single `run-bw-array` configs can use `bw.n_jobs: auto` and
`cosmology.angular_power.n_jobs: auto` to use the same available-CPU count.

For local iteration, use 4k and 64k runs. Use 256k as a single-seed scale check after a meaningful
dynamics/readout change. Use cloud CPU batches for repeated 256k runs or 1M runs. GPUs and TPUs are
still not useful for the current engine until the hot path is ported to GPU-native kernels or dense
modular-state evolution dominates runtime.

## Immediate Hardware Recommendation

Start with CPU instances.

Recommended first cloud shape:

```text
1 x CPU VM
32-64 vCPUs
128-256 GB RAM
fast local NVMe if available
0 GPUs
0 TPUs
```

Run:

```bash
python3 -m oph_fpe.cli run-bw-sweep \
  --configs configs/e1_s3_bw_screen_4k.yml configs/e1_s3_bw_screen_64k.yml \
  --seeds 11,12,13,14 \
  --workers 4 \
  --inner-jobs 1 \
  --out-dir runs
```

For a larger batch:

```text
2-4 x CPU VMs
64-96 vCPUs each
256-512 GB RAM each
workers per VM: 4-8
inner_jobs: 1
```

For a single bigger-than-1M run:

```text
1 x large-memory CPU VM
96-192 vCPUs
512 GB-1 TB RAM
workers: 1
inner_jobs: 16-32
```

## Provider Pick

Best fit for the simulator: a CPU cloud VM with dedicated CPU resources. If
Hetzner Cloud is unavailable, use one of these fallbacks:

1. DigitalOcean: simplest single-account fallback for ordinary CPU Droplets and
   GPU Droplets. Use CPU Droplets for BW sweeps. GPU Droplets belong to
   GPU-native engine paths.
2. Vultr: also a reasonable single-account fallback because it offers Cloud Compute, Optimized Cloud
   Compute, Cloud GPU, and Bare Metal. Use CPU/optimized compute for the
   CPU-bound engine.
3. CoreWeave: suitable for larger GPU-backed or mixed CPU/GPU workloads. It is
   less necessary for CPU-bound BW sweeps.
4. RunPod or Vast.ai: best for fast GPU experiments or opportunistic marketplace
   capacity. Pure CPU sweeps fit better on CPU VMs.

AWS/GCP/Azure are fine technically, but quota friction is more likely for large CPU and especially
GPU counts.

## Google Cloud CPU Notes

Google Cloud is a good provider for the CPU phase if the project has enough Compute Engine vCPU
quota. Check both:

```text
CPUS_ALL_REGIONS
regional CPU quota, for example us-central1/CPUS, us-east1/CPUS
machine-family CPU quota, for example C2D_CPUS, N2_CPUS, C3_CPUS, CPUS_PER_VM_FAMILY
```

The smallest applicable quota wins. A project can show hundreds of regional CPUs but still be capped
by a smaller `CPUS_ALL_REGIONS` quota.

Useful checks:

```bash
gcloud compute project-info describe --format=json \
  | jq '.quotas[] | select(.metric|test("CPU|CPUS"))'

gcloud compute regions describe us-central1 --format=json \
  | jq '.quotas[] | select(.metric|test("CPU|CPUS|C2D|N2|C3|C4"))'
```

For this simulator, prefer CPU families such as `c3d-standard`, `c3d-highmem`, `c2d-standard`, or
`n2-standard`. Use multiple VMs for seed/config sweeps instead of one huge VM unless running a single
large 1M+ screen job.

Quota request template:

```text
CPUS-ALL-REGIONS-per-project: request enough for your planned worker count
CPUS-per-project-region <region>: request enough for your planned worker count
N2-CPUS-per-project-region <region>: optional if using N2
C2D-CPUS-per-project-region <region>: optional if using C2D
```

Example 32-vCPU fallback worker shape:

```text
instance: oph-fpe-n2-32-<date>
zone: ${GCP_ZONE}
machine: n2-standard-32
remote_path: ~/oph-fpe/oph-physics-sim
tmux_session: oph-fpe-gcp-<date>
run_dir: runs/gcp_<date>
```

Example queued follow-up run:

```text
tmux_session: oph-fpe-gcp-256k-next
behavior: waits for active session to finish, then starts
config: configs/e4_shared_observer_bulk_256k_observers4096_theorem.yml
seeds: 11,12,13,14
workers: 4
inner_jobs: 1
run_dir: runs/gcp_256k_<date>
```

Monitor:

```bash
gcloud compute ssh "${GCP_WORKER_NAME}" --zone="${GCP_ZONE}" \
  --project="${GCP_PROJECT_ID}" \
  --command='tmux ls; pgrep -af "oph_fpe.cli run-bw-sweep"; find ~/oph-fpe/oph-physics-sim/runs/"${GCP_RUN_DIR}" -maxdepth 2 -name bw_report.json | wc -l'
```

Collect:

```bash
mkdir -p runs/"${GCP_RUN_DIR}"
gcloud compute scp --recurse \
  "${GCP_WORKER_NAME}:~/oph-fpe/oph-physics-sim/runs/${GCP_RUN_DIR}" \
  runs/ \
  --zone="${GCP_ZONE}" \
  --project="${GCP_PROJECT_ID}"
```

Stop spending:

```bash
gcloud compute instances stop "${GCP_WORKER_NAME}" \
  --zone="${GCP_ZONE}" \
  --project="${GCP_PROJECT_ID}"
```

## AWS CPU Quota Notes

Configure AWS locally with a profile and region. Do not commit account IDs, IAM user names, access
keys, or quota snapshots.

Quota checks:

```bash
aws service-quotas list-service-quotas \
  --service-code ec2 \
  --profile "${AWS_PROFILE}" \
  --region "${AWS_REGION}"
```

Quota request template:

```text
Running On-Demand Standard vCPUs: requested 512, status PENDING
All Standard Spot Instance Requests vCPUs: requested 512, status PENDING
```

Use the smallest approved vCPU quota across the relevant EC2 quota classes as the actual cap.

For future GPU work:

- Use RunPod for quick fixed-price GPU pods when we have a CUDA/JAX/CuPy path.
- Use Vast.ai for marketplace availability and cheap opportunistic GPU/CPU hosts; expect variability.
- Use Lambda/CoreWeave only when we need serious multi-GPU clusters after the code is GPU-native.

## Resource Count Guidance

```text
CPU-bound engine:
  CPUs: 32-64 vCPUs to start, 256+ vCPUs for real seed batches
  GPUs: 0
  TPUs: 0

After cap-state matrices or JAX/CuPy port:
  GPUs: 1 x L40S/A100/H100 for prototype
  then 4-8 GPUs only if profiling proves GPU saturation
  TPUs: still 0 unless the whole engine becomes JAX/XLA and dense-linear-algebra-heavy
```
