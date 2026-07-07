# OPH-FPE Configuration

This document defines the stable configuration surface for simulator runs. Config files describe
finite observer-like self-reading systems: bounded patch carriers with local state, ports or
boundaries, readback, records, feedback or repair moves, and public receipts.

## Repository Policy

Tracked files under `configs/` are curated examples and regression fixtures. They are intentionally
small in number:

- smoke/calibration configs used by tests and quickstarts;
- the exact small-universe finite-consensus harness;
- the current OPH-universe object-chart and theorem-scale examples;
- shape/defect assay examples.

Do not commit one-off sweeps, cloud-local variants, dated experiment attempts, or provider
credentials. Put those under one of the ignored paths:

```text
configs/local/
configs/generated/
configs/*.local.yml
configs/*.local.toml
configs/*.private.yml
configs/*.tmp.yml
correspondence/
runs/
distributed/
```

If a local config becomes a reusable public fixture, rename it deliberately, remove machine-specific
values, add a claim boundary, and update `configs/README.md`.

## YAML Run Format

The standard simulator config is YAML. The loader accepts a mapping and passes nested sections to the
selected CLI command. Common top-level sections are:

```yaml
name: e4_shared_observer_bulk_64k_object_chart
run_mode: observer_shared_bulk_candidate_compact
seed: 20260751
claim_boundary: >
  State what the run may and may not claim. A diagnostic chart, visualizer export,
  or proxy spectrum is not a physical claim unless the corresponding receipts pass.

graph:
  family: fibonacci_sphere
  patch_count: 65536
  neighbors: 12

group:
  name: S3

boundary_program:
  mode: support_visible_cap_net_hot

screen:
  chart: support_visible_s2_cellulation
  carrier: federated_echosahedral_patch
  ports_per_patch: 12

oph_constants:
  pixel_mode: source_candidate

screen_units:
  mode: numerical_regulator

dynamics:
  cycles: 64
  # Optional. When present, this scales repair throughput with patch_count and
  # overrides repairs_per_cycle. Use this for comparable 4k/64k/256k runs.
  repair_fraction_per_cycle: 0.0625
  repairs_per_cycle: 4096
  beta_schedule:
    kind: geometric
    beta_start: 0.05
    beta_end: 15.0

modular_flow:
  enabled: true

bw:
  mode: state_derived_modular_probe
  n_jobs: auto

observers:
  sample_count: 1024
  neighborhood_size: 96

observer_objects:
  enabled: true

observer_chart_population:
  enabled: true

neutral_reconstruction:
  enabled: false

h3_modular_response:
  enabled: true

theorem_core:
  consensus_replay:
    enabled: true

cosmology:
  freezeout:
    enabled: true

defects:
  enabled: true

controls:
  - shuffled_incidence

outputs:
  write_viewer_payload: true
```

Commands read only the sections they understand. Unknown sections are allowed when they are consumed
by a specific pipeline, but they should not be used as hidden switches for physical claims.

## Claim Boundaries

Every committed config needs a `claim_boundary`. It should say:

- what finite carrier and observer-readout structure is being instantiated;
- whether the run is a smoke test, diagnostic, strict finite theorem harness, or claim-scale
  candidate;
- which physical promotions remain closed unless emitted receipts pass.

This is especially important for CMB, neutral bulk, proto-particle, and effective-string visualizer
exports. The visualization payload can render those views diagnostically, but the config cannot turn
them into physical claims without the corresponding public evidence bundle.

## Local Cloud Config

Provider and quota details are local. Use environment variables, cloud-native identity, or an
ignored TOML file such as `configs/local/cloud.local.toml`:

```toml
[cloud]
provider = "gcp"
project = "${GCP_PROJECT_ID}"
region = "${GCP_REGION}"
zone = "${GCP_ZONE}"
artifact_bucket = "${GCP_ARTIFACT_BUCKET}"

[worker]
purpose = "cpu-sweep"
machine_type = "n2-standard-64"
nodes = 1

[runtime]
workers = 8
inner_jobs = 8
blas_threads = 1
```

Never commit account IDs, service-account keys, access tokens, SSH keys, Terraform state, live quota
snapshots, or production bucket names.
