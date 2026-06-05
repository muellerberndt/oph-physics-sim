# Cloud Wiring

The default Google Cloud project for this repo is:

```text
observer-patch-holography
```

Local `gcloud` is configured with a separate named configuration:

```bash
gcloud config configurations activate oph-physics-sim
gcloud config list
```

Baseline APIs are enabled for Compute Engine, Cloud Storage, Cloud Build, IAM, Service Usage, and
Artifact Registry. Billing is attached.

The default artifact bucket exists:

```text
gs://oph-physics-sim-artifacts-640205981110-us-central1
```

## GCP Starting Quota

Checked on 2026-06-01:

```text
us-central1: 200 CPUs, 1 NVIDIA_L4_GPU, 0 A100/H100/H200/B200 GPUs
us-east1:    200 CPUs, 1 NVIDIA_L4_GPU, 0 A100/H100/H200/B200 GPUs
us-west1:    100 CPUs, 1 NVIDIA_L4_GPU, 0 A100/H100/H200/B200 GPUs
```

The practical immediate GCP GPU smoke target is one `g2-standard-8` L4 VM. B200/H200/A100 work
needs a quota increase first.

## Processing Recommendation

The current vectorized large-screen engine is CPU/RAM bound, not GPU bound. A 64k-patch S3 modular
screen smoke run completes locally in a few seconds and uses sampled point-cloud estimators. For the
next scale step, prefer high-core CPU workers:

```text
Local / workstation: 64k patches, repeated seeds, control sweeps
GCP CPU: c3-standard-44 / c3-standard-88 or n2-standard-64 for 1M-patch sweeps
AWS CPU: c7i/c7a/c8g high-core instances for parallel seed batches
RunPod/Lambda/Modal GPU: defer until dense eigensolvers, persistent homology, learned repair, or CAMB sweeps dominate
```

Use the existing `observer-patch-holography` GCP project for artifacts and coordination. Renting
GPU pods now would mostly accelerate the wrong layer unless we first add GPU-native kernels.

## Local Secrets

Use `.env.local`, shell exports, local Terraform var files, or cloud-native identity. Do not commit
service-account keys, AWS keys, OAuth tokens, SSH keys, or provider credentials.

The `ophminer` repo remains a reference for cloud topology, but this repo should keep neutral names
and simulation-specific buckets, service accounts, tags, and Terraform state.
