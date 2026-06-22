# Cloud Wiring

This repo should not contain production cloud identifiers. Configure real provider details locally
with `.env.local`, shell exports, named CLI profiles, or cloud-native identity.

Example Google Cloud project placeholder:

```text
your-gcp-project-id
```

Recommended local `gcloud` pattern:

```bash
gcloud config configurations activate <your-config-name>
gcloud config list
```

Baseline APIs needed for a GCP worker are Compute Engine and Cloud Storage. Cloud Build, IAM,
Service Usage, and Artifact Registry are useful for more automated workflows.

Artifact bucket placeholder:

```text
gs://your-gcs-artifact-bucket
```

## GCP Quota

Check quotas in the project/region you intend to use:

```bash
gcloud compute project-info describe --format=json \
  | jq '.quotas[] | select(.metric|test("CPU|CPUS|GPU|NVIDIA"))'

gcloud compute regions describe "${GCP_REGION}" --format=json \
  | jq '.quotas[] | select(.metric|test("CPU|CPUS|GPU|NVIDIA|C2D|N2|C3|C4"))'
```

The practical immediate GPU smoke target, when quota exists, is one `g2-standard-8` L4 VM.
B200/H200/A100 work usually needs quota approval first.

## Processing Recommendation

The current vectorized large-screen engine is CPU/RAM bound, not GPU bound. For scale checks and
seed batches, prefer high-core CPU workers:

```text
Local / workstation: 64k patches, repeated seeds, control sweeps
GCP CPU: c3-standard-44 / c3-standard-88 or n2-standard-64 for 1M-patch sweeps
AWS CPU: c7i/c7a/c8g high-core instances for parallel seed batches
RunPod/Lambda/Modal GPU: defer until dense eigensolvers, persistent homology, learned repair, or CAMB sweeps dominate
```

Use a dedicated simulation project for artifacts and coordination. Renting GPU pods now would
mostly accelerate the wrong layer unless we first add GPU-native kernels.

## Local Secrets

Use `.env.local`, shell exports, local Terraform var files, or cloud-native identity. Do not commit
service-account keys, AWS keys, OAuth tokens, SSH keys, or provider credentials.

Keep this repo neutral: use simulation-specific buckets, service accounts, tags, and Terraform
state, and do not mirror production details from other projects.
