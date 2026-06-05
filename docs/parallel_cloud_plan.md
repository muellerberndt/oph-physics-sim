# Parallel Execution And Cloud Plan

Updated: 2026-06-02

## Current Bottleneck

The current OPH-FPE BW path is CPU-bound:

- spherical point generation;
- kNN / `cKDTree` screen construction;
- vectorized edge repair with NumPy;
- graph-neighbor support-visible smoothing;
- cap/time BW residual tasks using `cKDTree` queries and NumPy interpolation;
- direct spherical-harmonic freezeout `C_l` receipts, when enabled.

There is no useful GPU or TPU acceleration yet because the hot path is SciPy/NumPy/KD-tree and
graph-array work, not CUDA/JAX matrix kernels. GPUs become useful only after we port the array engine
to CuPy/JAX or add dense cap-state matrix evolution, persistent homology at large scale, or neural
repair policies.

The direct `spherical_harmonic` `C_l` estimator is intentionally used for reported single receipts
because it gives nonnegative auto-power. It is CPU-bound, and now parallelizes independent
target/control spectra with `angular_power.n_jobs`. Use it for final screen-spectrum receipts; use
the sampled pair proxy or lower `ell_max` for broad seed/config sweeps if harmonic receipts dominate
runtime.

## Parallelism Now

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
  --seeds 20260601,20260602,20260603,20260604 \
  --workers 4 \
  --inner-jobs 1 \
  --out-dir runs
```

Use either high `--workers` or high `--inner-jobs`, not both, unless the machine has enough cores and
RAM. For seed sweeps, prefer `--workers N --inner-jobs 1`. For one very large run, prefer
`--workers 1 --inner-jobs N`.

As of 2026-06-02, `run-bw-sweep` auto-plans CPU use when `--workers` and `--inner-jobs` are omitted:

```text
workers = min(job_count, available_cpus)
inner_jobs = floor(available_cpus / workers)
```

`available_cpus` is `os.cpu_count() - 1` by default, reserving one local CPU for the OS/UI. Set
`OPH_FPE_CPUS=<N>` to override this on cloud VMs or batch schedulers. The sweep planner passes the
same `inner_jobs` budget into both `bw.n_jobs` and `cosmology.angular_power.n_jobs` unless a config
sets an explicit angular-power value. Single `run-bw-array` configs can use `bw.n_jobs: auto` and
`cosmology.angular_power.n_jobs: auto` to use the same available-CPU count.

On this local machine:

```text
os.cpu_count(): 10
auto available CPUs: 9
single BW run with n_jobs:auto: 9 inner BW threads
3-job sweep with no explicit caps: workers=3, inner_jobs=3
```

Current E2 timing baseline on this machine:

```text
64k full E2 run, workers=1, inner_jobs=8: about 32-38 seconds
two 64k seeds, workers=2, inner_jobs=4: about 52 seconds total
256k full E2 run, workers=1, inner_jobs=8, ell_max=48: about 255-310 seconds
```

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
  --configs configs/e1_s3_bw_screen_4k.yml configs/e1_s3_bw_screen_64k.yml configs/e1_s3_bw_screen_1m.yml \
  --seeds 20260601,20260602,20260603,20260604 \
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

Best immediate fit for the current simulator: a CPU cloud VM with dedicated CPU resources. If
Hetzner Cloud is unavailable, use one of these fallbacks:

1. DigitalOcean: simplest single-account fallback for ordinary CPU Droplets now and GPU Droplets
   later. Use CPU Droplets for BW sweeps; do not rent GPU Droplets until a GPU-native engine path
   exists.
2. Vultr: also a reasonable single-account fallback because it offers Cloud Compute, Optimized Cloud
   Compute, Cloud GPU, and Bare Metal. Use CPU/optimized compute for the current engine.
3. CoreWeave: good when the project is ready for larger GPU-backed or mixed CPU/GPU workloads, but
   it is less necessary for the current CPU-bound BW sweeps.
4. RunPod or Vast.ai: best for fast GPU experiments or opportunistic marketplace capacity. They are
   not the first choice for pure CPU sweeps, but they are useful once the code has a CUDA/JAX/CuPy
   path.

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

Quota preferences submitted for `observer-patch-holography` on 2026-06-01:

```text
CPUS-ALL-REGIONS-per-project: current 32, requested 1200
CPUS-per-project-region us-central1: current 200, requested 512
N2-CPUS-per-project-region us-central1: current 200, requested 512
C2D-CPUS-per-project-region us-central1: current 100, requested 512
```

Active 32-vCPU fallback worker started on 2026-06-01:

```text
instance: oph-fpe-n2-32-20260601
zone: us-central1-a
machine: n2-standard-32
external_ip: 34.42.212.46
remote_path: ~/oph-fpe/oph-physics-sim
tmux_session: oph-fpe-gcp-20260601
run_dir: runs/gcp_20260601
```

Queued follow-up run:

```text
tmux_session: oph-fpe-gcp-256k-next
behavior: waits for oph-fpe-gcp-20260601 to finish, then starts
config: configs/e1_s3_bw_screen_256k.yml
seeds: 20260620,20260621,20260622,20260623
workers: 4
inner_jobs: 1
run_dir: runs/gcp_256k_20260601
```

Monitor:

```bash
gcloud compute ssh oph-fpe-n2-32-20260601 --zone=us-central1-a \
  --project=observer-patch-holography \
  --command='tmux ls; pgrep -af "oph_fpe.cli run-bw-sweep"; find ~/oph-fpe/oph-physics-sim/runs/gcp_20260601 -maxdepth 2 -name bw_report.json | wc -l; tail -40 ~/oph-fpe/oph-physics-sim/runs/gcp_20260601.log'
```

Collect:

```bash
mkdir -p runs/gcp_20260601
gcloud compute scp --recurse \
  oph-fpe-n2-32-20260601:~/oph-fpe/oph-physics-sim/runs/gcp_20260601 \
  runs/ \
  --zone=us-central1-a \
  --project=observer-patch-holography
```

Stop spending:

```bash
gcloud compute instances stop oph-fpe-n2-32-20260601 \
  --zone=us-central1-a \
  --project=observer-patch-holography
```

## AWS CPU Quota Notes

AWS CLI is configured for account `484819296236` as IAM user `ophminer` in `us-east-1`.

Current EC2 quota checked on 2026-06-01:

```text
Running On-Demand Standard vCPUs: 384
All Standard Spot Instance Requests vCPUs: 384
```

Quota increases submitted on 2026-06-01:

```text
Running On-Demand Standard vCPUs: requested 512, status PENDING
All Standard Spot Instance Requests vCPUs: requested 512, status PENDING
```

The existing 384 vCPU quota is already enough for a 256k BW campaign if we use AWS.

For future GPU work:

- Use RunPod for quick fixed-price GPU pods when we have a CUDA/JAX/CuPy path.
- Use Vast.ai for marketplace availability and cheap opportunistic GPU/CPU hosts; expect variability.
- Use Lambda/CoreWeave only when we need serious multi-GPU clusters after the code is GPU-native.

## Current Count Guidance

```text
Right now:
  CPUs: 32-64 vCPUs to start, 256+ vCPUs for real seed batches
  GPUs: 0
  TPUs: 0

After cap-state matrices or JAX/CuPy port:
  GPUs: 1 x L40S/A100/H100 for prototype
  then 4-8 GPUs only if profiling proves GPU saturation
  TPUs: still 0 unless the whole engine becomes JAX/XLA and dense-linear-algebra-heavy
```
