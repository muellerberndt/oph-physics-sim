# Config Fixtures

This directory contains curated OPH-FPE configuration fixtures. It is not a scratch directory for
one-off runs.

Tracked configs are limited to:

- `e0_z2_patchnet.yml`: E0 finite patch-net smoke run.
- `e1_s3_modular_screen_4k.yml`: S3 modular-screen smoke run.
- `e1_s3_bw_screen_4k.yml` and `e1_s3_bw_screen_64k.yml`: BW sweep fixtures.
- `e1_s3_state_modular_screen_4k.yml`: state-derived BW fixture.
- `e1_s3_transition_response_screen_4k.yml`: transition-response fixture.
- `e2_kms_freezeout_cl_screen_64k.yml`: KMS/freezeout diagnostic fixture.
- `e3_cosmo_proxy_screen_64k.yml`: compact cosmology-proxy fixture.
- `e4_shared_observer_bulk_64k_object_chart.yml`: current local OPH-universe object-chart fixture.
- `e4_shared_observer_bulk_256k_observers4096_theorem.yml`: current large OPH-universe theorem-scale fixture; filename is legacy, while the YAML now materializes 32,768 observer-local readout neighborhoods and exports 4,096 observer perspectives.
- `sou_v1_icosa12.yml`: exact small-universe finite-consensus harness.
- `shape_dodeca_vertex_smoke.yml` and `shape_dodeca_ensemble.yml`: shape/defect assay fixtures.

Put local variants under `configs/local/` or use suffixes such as `.local.yml`, `.private.yml`, or
`.tmp.yml`; those paths are ignored by Git. See `docs/configuration.md` for the accepted format and
claim-boundary rules.
