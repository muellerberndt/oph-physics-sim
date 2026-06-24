# OPH-FPE Documentation

This directory is for stable documentation about how the simulator works:

- one current theorem-to-code and claim-boundary contract;
- payload and receipt schemas;
- curated config format and claim-boundary rules;
- cloud/runtime setup notes;
- stable harness-specific run instructions.

Run-specific outputs, dated run reports, intermediate experiment notes, benchmark receipts,
and one-off closeout bundles do not belong here. Keep those under `runs/`, `reports/`,
`measurement_packs/`, `correspondence/`, or a task-specific handoff directory.

## Stable Docs

- `OPH_THEOREM_TO_SIM_IMPLEMENTATION_SPEC.md`: the single source of truth for simulator state,
  paper-stack alignment, receipt lanes, CMB/bulk/particle/string/vacuum boundaries, and promotion
  rules.
- `VISUALIZATION_APP_AGENT_MANUAL.md`: implementation manual for visualization app agents building
  quantum-vacuum, observer-camera, effective-string, repair, H3, and CMB diagnostic views from the
  universe-timeline payload.
- `cloud.md`
- `configuration.md`
- `digitalocean_pool_setup.md`
- `oph_issue361_certificate_schema.json`
- `oph_universe_timeline_visualization_payload_v1.schema.json`
- `parallel_cloud_plan.md`
- `small_oph_universe_v1.md`
