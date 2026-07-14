# Simulator documentation

Policy: stable documentation lives here; pass/fail receipt labels live in
run artifacts under `runs/`; mutable progress lives in GitHub issues
(linked as full URLs). Generated packs and local configs stay ignored.

## Status and experiments

- `OPH_SIGNATURE_EXPERIMENT_TRACKER.md`: living experiment tracker with
  at-a-glance verdict tables and the results log.
- `RUN_OUTPUTS_AND_VISUALIZATION.md`: run-output inventory; visualizer
  bundle selection marked.
- `BEST_OF_PUBLIC_DATA_COMPARISONS.md`: provenance-bound public-data
  comparison suite.
- `SCALING_MILESTONE_ESTIMATES_2026-07-13.md`: scale milestones,
  empirical lower bounds, and measured actuals.

## Contracts and lanes

- `OPH_THEOREM_TO_SIM_IMPLEMENTATION_SPEC.md`: theorem-to-code and
  claim-promotion contract.
- `CLAIM_LANES.md`: lane-by-lane contracts.
- `PROOF_PACKET_AUDITS.md`: fail-closed proof-packet audits and
  Lean-mirrored fixtures.
- `SIMULATION_ASSUMPTION_POLICY.md`: assumed-bridge visualization lane.
- `PAPER_PARTICLE_ARTIFACT_INTEGRATION.md`: cross-repo artifact import.
- `WZH_NUMERICAL_BACKEND.md`: boson backend contract.
- `neutrino_status.md`: neutrino lane status.
- `small_oph_universe_v1.md`: exact finite-consensus calibration harness.

## Schemas

- `oph_universe_timeline_visualization_payload_v1.schema.json`
- `oph_visualizer_pack_v2.schema.json`
- `oph_distributed_universe_visualization_payload_v1.schema.json`
- `particle_promotion_evidence_v1.schema.json`
- `hadron_source_promotion_evidence_v1.schema.json`
- `oph_issue361_certificate_schema.json`

## Configuration and operations

- `configuration.md`: config format and claim-boundary rules.
- `VISUALIZATION_APP_AGENT_MANUAL.md`: payload-driven visualization
  manual (`../scripts/README_FOR_WEB_CODING_AGENT.md` for bundles).
- `GCP_SCALING_PLAN.md`: scaling and cloud operations (fleet ensembles,
  monolithic sizing, code-level parallelism, credentials policy).
