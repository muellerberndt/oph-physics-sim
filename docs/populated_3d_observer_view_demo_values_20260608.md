# Populated 3D Observer-View Demo Values

Status: demo calibration, not final theorem closure.

The current reliable path to a working "screen + observer perspective + H3 bulk chart + defect overlay" view is the paper-chart route:

```text
S2 cap screen -> BW-normalized cap flow s = 2*pi*t -> Conf+(S2) / SO(3) H3 chart -> observer-object support profiles -> defect support overlay
```

Do not use the neutral summary-distance dimension estimator as the primary gate for this demo. It remains a debug/audit lane and is currently not passing.

## Fixed Values For The Working Demo

Use the 256k particle/object-chart profile:

```yaml
graph.patch_count: 262144
graph.neighbors: 12
group.name: S3

dynamics.cycles: 64
dynamics.repairs_per_cycle: 262144
dynamics.record_commit_cycles: 8

bw.transition_response_scale: 6.283185307179586
bw.normalization: 2pi
bw.times: [0.1]
bw.theta0: [0.35, 0.55, 0.75, 1.0]
bw.collar_k: 1.0
bw.max_basis: 48

observers.sample_count: 1024
observers.neighborhood_size: 256

observer_objects.family_mode: connected_packets
observer_objects.max_support_size: 768
observer_objects.max_families: 2048
observer_objects.persistence_horizon: 8

observer_chart_population.incidence_mode: record_family_modular_response_mixture
observer_chart_population.min_objects: 12
observer_chart_population.min_observers_per_object: 4
observer_chart_population.max_h3_compactness: 0.5
observer_chart_population.min_localized_objects: 2
observer_chart_population.pass_ratio: 1.0
observer_chart_population.shuffle_control_count: 16
observer_chart_population.boundary_gate_mode: boundary_leakage_audit

h3_support_profiles.cap_count: 24
h3_support_profiles.theta0: [0.35, 0.55, 0.75, 1.0, 1.25]
h3_support_profiles.candidate_count: 1024
h3_support_profiles.timeline_candidate_count: 1024
h3_support_profiles.defect_min_support_count: 4

h3_modular_response.observable_mode: collar_operator_transition
h3_modular_response.times: [0.125, 0.25, 0.5]
h3_modular_response.transport_scale: 6.283185307179586
h3_modular_response.candidate_count: 4096
h3_modular_response.candidate_radius: 2.0
h3_modular_response.softness: 0.25
h3_modular_response.feature_selection: class_distribution_and_change

defects.timeline.enabled: true
defects.timeline.sample_count: 8
defects.timeline.max_triangles: 6000
defects.timeline.persistence_cycles: 3
defects.timeline.particle_min_observations: 3
defects.timeline.particle_max_support_fraction: 0.05

cosmology.freezeout.require_neutral_reconstruction: false
cosmology.angular_power.ell_max: 48
```

Reference config:

```text
configs/e4_shared_observer_bulk_256k_particles.yml
```

Reference successful run:

```text
runs/stage97_particles_defect_threshold_256k_seed20261047_20260607/e4_shared_observer_bulk_256k_particles_seed20261047_6a562ca8
```

Generated viewer:

```text
runs/stage214_populated_3d_observer_view_demo_20260608/oph_realtime_viewer.html
```

## Current Demo Receipts

From the reference run:

```text
bulk_3d_established: true
PAPER_THEOREM_3D_BULK_CHART_RECEIPT: true
CHART_LEVEL_CONFORMAL_LORENTZ_RECEIPT: true
CHART_LORENTZ_H3_RECEIPT: true
BW_KMS_DIRECT_2PI_RECEIPT: true
H3_RESPONSE_CANDIDATE_RECEIPT: true
H3_RESPONSE_CONTROL_SEPARATION_RECEIPT: true
OBJECT_BULK_POPULATION_RECEIPT: true
PAPER_THEOREM_ASSISTED_H3_POPULATED_CHART_RECEIPT: true
support_visible_defect_h3_population_receipt: true
particle_matter_receipt: false
physical_cmb_prediction: false
ENDOGENOUS_MODULAR_GENERATOR_RECEIPT: false
```

Viewer summary:

```text
screen_point_count: 12000
defect_cluster_count: 7
defect_timeline_snapshots: 8
persistent_defect_worldlines: 1
H3 support points: 71
H3 worldlines: 1
particle_like_count: 0
```

## Why These Values

The papers derive the 3+1D Lorentz/H3 chart from the support-visible BW/cap-flow branch, not from a raw finite S2 point-cloud dimension. Therefore the demo fixes the chart-side constants first:

```text
s = 2*pi*t
Conf+(S2) -> SO+(3,1)
H3 = SO+(3,1) / SO(3)
spatial chart dimension = 3
```

Then it asks whether observer-facing objects and defect support profiles populate that H3 chart under shuffled-incidence/control comparisons.

## Open Gates

These are still not fixed by the demo:

```text
neutral third-person observer-distance reconstruction
endogenous finite rho_C modular generator beating controls
particle matter receipt
physical CMB prediction
official Planck likelihood / map-space anomaly tests
finite-lattice repair-clock certificate kappa_rep=e
time-resolved harmonic Gamma_sigma(ell) low-k synchronization gap
```

The demo is therefore useful for the combined visual target and for debugging object/population mechanics. It is not the final OPH proof package.
