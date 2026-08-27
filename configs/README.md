# Config Fixtures

This directory contains curated OPH-FPE configuration fixtures. It is not a scratch directory for
one-off runs.

Tracked configs are limited to:

- `icosa_smoke.yml`, `icosa_20k.yml`, `icosa_82k.yml`, `icosa_328k.yml`, and
  `icosa_1310k.yml`: the production `icosahedral_tower` family, from the small
  engineering rung through the named campaign sizes. These are non-evidential
  unless a separately preregistered instrument says otherwise.
- `icosa_82k_protected_consensus.yml`: an evidence-profile tower run for the
  protected-authority finite consensus protocol. It tests confluence within a
  frozen authority-conditioned boundary fiber and does not derive the
  authority, observer feedback, spacetime, or laboratory physics.
- `e0_z2_patchnet.yml`: E0 finite patch-net smoke run.
- `e1_s3_modular_screen_4k.yml`: S3 modular-screen smoke run.
- `e1_s3_bw_screen_4k.yml` and `e1_s3_bw_screen_64k.yml`: BW sweep fixtures.
- `e1_s3_state_modular_screen_4k.yml`: state-derived BW fixture.
- `e1_s3_transition_response_screen_4k.yml`: transition-response fixture.
- `e2_kms_freezeout_cl_screen_64k.yml`: KMS/freezeout diagnostic fixture.
- `e3_cosmo_proxy_screen_64k.yml`: compact cosmology-proxy fixture.
- `e4_shared_observer_bulk_64k_object_chart.yml`: current local OPH-universe object-chart fixture.
- `e4_shared_observer_bulk_256k_observers4096_theorem.yml`: current large OPH-universe theorem-scale fixture; filename is legacy, while the YAML now materializes 32,768 observer-local readout neighborhoods and exports 4,096 observer perspectives.

All tracked fixtures whose `graph.family` is `fibonacci_sphere` are legacy
support-chart/KNN controls. Their geometry receipt deliberately keeps
`TRUE_ICOSAHEDRAL_REFINEMENT_TOWER_RECEIPT` false; a declared twelve-port
screen stanza does not turn that graph into the production A1 carrier.

The E4 visual-universe configurations declare `simulation_assumptions` for
paper bridges that the renderer needs but the Python run is not intended to
prove. These assumptions are visualization-only and remain separate from all
computed receipts; see `../docs/SIMULATION_ASSUMPTION_POLICY.md`.
- `e5_128k_observers32k_earned.yml` and `e5_128k_observers32k_night1.yml`: bounded 128k
  screen profiles with 32,768 observer neighborhoods.
- `e5_1m_bounded_visualizer_earned.yml` and `e5_1m_bounded_visualizer_night1.yml`: bounded
  million-patch visualizer profiles with reduced field exports.
- `e6_axiom_manifest_16k_dense_observers.yml` and
  `e6_axiom_manifest_64k_dense_observers.yml`: dense-observer diagnostics on the
  axiom-manifest lane; each run writes the pinned verbatim A1-A3 manifest and the
  exact conditional-resampling realization receipt.
- `k1_population_transfer_4k_dense.yml` with its `_kmsfix`, `_obs4`, and
  `_parentwins` variants, `k1_population_transfer_16k_dense_ladder.yml`, and the
  `k1_4k_dense_OLDCFG.yml` / `k1_4k_dense_OLDCFG_noba.yml` pair: population-transfer
  and transition-clock diagnostics at fixed dense-observer settings.
- `sou_v1_icosa12.yml`: exact small-universe finite-consensus harness.
- `shape_dodeca_vertex_smoke.yml` and `shape_dodeca_ensemble.yml`: shape/defect assay fixtures.
- `bosons/wzh_source_closure_diagnostic_v1.yml`: fail-closed synthetic control
  for the W/Z/H source-clock, RG, BRST-block, and complex-pole backend. It is
  not a physical mass prediction fixture.

Put local variants under `configs/local/` or use suffixes such as `.local.yml`, `.private.yml`, or
`.tmp.yml`; those paths are ignored by Git. See `docs/configuration.md` for the accepted format and
claim-boundary rules.
