# DigitalOcean Pool Setup

## Pool Type

Use a fixed-size Droplet autoscale pool for OPH-FPE batch simulation.

Fixed size is better than dynamic autoscale for the current simulator because each worker is expected
to saturate CPU while running assigned seeds. CPU-utilization autoscaling would mostly measure that
the workload is doing what it should, not that more workers are needed. Dynamic autoscale becomes
useful later only after we add a real job queue and worker shutdown handling.

## Recommended Pool

```text
Configuration: Fixed Size
Number of Droplets: 2 to start, 4 to 8 after the setup is verified
Region: nearest available region with the chosen CPU plan
Image: Ubuntu 24.04 LTS
Plan: CPU-Optimized / Premium CPU if available
Per-Droplet size: 16-32 vCPUs to start
SSH keys: include the local public key used from this workstation
Tags: oph-fpe-worker, oph-fpe
IPv6: optional
Startup script: paste infra/digitalocean/cloud-init-worker.yml after replacing the SSH key placeholder
```

Do not use GPU Droplets for the current BW engine. The hot path is still NumPy/SciPy/KD-tree and
graph-array work, so GPUs will not be useful until a JAX/CuPy/CUDA path exists.

## What I Need To Use It

After the pool exists, provide either:

```text
1. A DigitalOcean API token exported locally as DIGITALOCEAN_ACCESS_TOKEN, or
2. The public IP addresses of the Droplets plus SSH access through the local public key.
```

For API-based pool discovery and droplet management, install and authenticate `doctl`:

```bash
brew install doctl
doctl auth init
doctl compute droplet-autoscale list
doctl compute droplet-autoscale get <pool-id>
doctl compute droplet-autoscale list-members <pool-id>
```

If you prefer not to install `doctl`, paste the Droplet IPs from the pool's Resources tab.

## Deploy From This Machine

For each worker:

```bash
rsync -az --delete \
  /Users/muellerberndt/Projects/oph-meta/oph-physics-sim/ \
  oph@DROPLET_IP:/opt/oph-fpe/oph-physics-sim/

ssh oph@DROPLET_IP '
  cd /opt/oph-fpe/oph-physics-sim &&
  python3 -m venv .venv &&
  source .venv/bin/activate &&
  pip install -e ".[dev]" &&
  python3 -m pytest -q
'
```

## Run Split Seed Batches

Assign different seeds to each Droplet. Example:

```bash
ssh oph@DROPLET_IP '
  cd /opt/oph-fpe/oph-physics-sim &&
  source .venv/bin/activate &&
  tmux new -d -s oph-fpe \
    "python3 -m oph_fpe.cli run-bw-sweep \
      --configs configs/e1_s3_bw_screen_4k.yml configs/e1_s3_bw_screen_64k.yml \
      --seeds 11,12,13,14 \
      --workers 4 \
      --inner-jobs 1 \
      --out-dir runs"
'
```

Use non-overlapping seed lists per Droplet.

## Collect Results

```bash
mkdir -p runs/digitalocean
rsync -az oph@DROPLET_IP:/opt/oph-fpe/oph-physics-sim/runs/ runs/digitalocean/DROPLET_NAME/
```

## Why Tags Matter

Tag the pool Droplets with `oph-fpe-worker` and `oph-fpe`. DigitalOcean recommends tagging autoscale
pool Droplets so load balancers and cloud firewalls can target the whole pool automatically. We do
not need a load balancer for batch simulation, but tags make firewalling and discovery cleaner.
