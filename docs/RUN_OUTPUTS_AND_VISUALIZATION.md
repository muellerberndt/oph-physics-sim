# Run outputs: postprocessing and visualization inventory

Generated against `runs/oph_universe_64k_3p1d_reearned` (2026-07-14, full
e4 64k run: ~180 root artifacts, ~750 MB with the timeline). Every file
below appears in a standard `run-oph-universe` output directory unless
marked config- or gate-dependent. The visualizer bundle builder
(`scripts/build_visualizer_zip.py`) selects the subset marked [ZIP].

## 1. Visualization payloads and viewers

- `universe_timeline/` (the visualization export, usually the largest
  item): `visualization_payload.json` [ZIP] and the chunked
  `oph_visualizer_pack_v2.tar.zst` [ZIP] (schema:
  `docs/oph_universe_timeline_visualization_payload_v1.schema.json`),
  `VISUALIZATION_INSTRUCTIONS.md` + `WEB_CODING_AGENT_VISUALIZATION_BRIEF.md`
  [ZIP], binary/CSV sidecars (cameras, observers, cinema, anatomy,
  `emergent_curved_spacetime*.{json,csv}`, `effective_string_theory.json`,
  `string_vacuum_selector_*.csv`, `finite_edge_string_vibration_samples.csv`)
  [ZIP].
- Standalone single-file viewers (open directly in a browser) [ZIP as
  reference viewers]: `oph_realtime_viewer.html`,
  `object_h3_bulk_viewer.html`, `cmb_neutral_frontier_viewer.html`
  (+ `*_summary.json` for each).
- `screen_evolution_frames.npz` [ZIP]: per-cycle raw screen frames for
  repair/freezeout animation. Config-gated
  (`harmonic_time_trace.save_raw_frames`); present in e4 configs, absent
  in the bounded e5 profiles.
- `plots/`: optional static plots (often empty).

## 2. Postprocessing arrays (npz)

- `freezeout_fields.npz` [ZIP]: per-patch committed fields at freezeout
  (`points`, `record_signature`, `s3_class_density`, `cell_entropy`,
  `repair_load`, smoothed variants). Gate-dependent: written only when
  the cosmology freezeout gate allows products. Bounded e5 profiles emit
  a reduced field set.
- `s3_gauge_state.npz` [ZIP]: final edge gauge state (`left`, `right`,
  `gauge` in S3 indices, `points`); input to the agreement certificate
  and gauge-web rendering.
- `harmonic_time_trace.npz`: harmonic-coefficient time series of screen
  fields.
- `modular_response_kernel_payload.npz` (+ `_cache.json`): the cached
  modular response kernel behind the H3 fit; heavyweight, postprocessing
  only.
- `finite_consensus_source_state.npz` + `finite_consensus_replay_report.json`:
  replayable consensus source state.
- `finite_repair_transition_matrix.npz` (+ report, rows CSV): repair
  transition spectrum (`lambda_2`, `gamma_continuous`), the internal
  decoherence-rate object.

## 3. Observer and agreement lane

- `observer_views.jsonl` [ZIP] (largest root file): one row per
  materialized observer (support nodes, modular depth, spectra,
  histograms, response tensors). Input to the agreement certificate and
  the strict-neutral cohort.
- `observer_agreement_report.json` [ZIP]: pair re-gauging defects,
  cocycle triples, controls, integer chart verdicts
  (`observer-agreement-report` CLI; `bulk_dimension_claim` null by
  schema).
- `observer_modular_experience_report.json` [ZIP]: the 3+1D experience
  receipt with its four component gates.
- `observer_consensus_report.json`, `observer_consensus_bulk/`,
  `observer_objects.jsonl`, `observer_population_report.json`,
  `observer_perspective_rows.csv` [ZIP], `observer_chart_object_h3_report.json`.

## 4. Defect / proto-particle lane

- `defect_timeline_report.json` [ZIP]: snapshot clusters, worldlines with
  per-cycle centroids and class labels (input to the turning statistic
  and the block-universe scene).
- `organic_defect_population_report.json` + `_trajectory.csv` +
  `_worldline_events.csv` + `_worldlines.csv`: the organic
  (un-planted) defect population with receipts.
- `defect_interaction_report.json` [ZIP]: encounter/fusion candidates,
  identity fractions, conservation proxies, scattering transitions.
- `defect_h3_worldlines_report.json` [ZIP], `defect_cluster_h3_report.json`:
  H3-lifted worldlines and clusters.
- `free_two_defect_dynamics_report.json` (+ trajectory CSV) and
  `two_defect_stress_contraction_assay_report.json` (+ CSVs): the
  two-defect dynamics assays.
- `array_holonomy_report.json` [ZIP], `s3_class_counts.json` [ZIP],
  `defect_worldline_turning_report.json` [ZIP],
  `screen_parity_report.json` [ZIP].

## 5. Neutral bulk and Einstein lane (effective-spacetime status)

- `strict_neutral_bulk_report.json` [ZIP],
  `strict_neutral_object_bulk_report.json`, source manifests, and
  `neutral_objects.jsonl`.
- `neutral_3d_bulk_audit_report.{json,md}` [ZIP]: the blocker list
  (refinement ladder, rank-3 stability, overlap gates).
- `neutral_prime_rank_sweep/`, `neutral_prime_rank_refinement/`,
  `neutral_rank_selector_audit/`, `neutral_overlap_graph_sweep/`,
  `neutral_overlap_residual_graph_sweep/`, `neutral_overlap_control/`:
  the sweep evidence the ladder audit consumes.
- `einstein_bridge_manifest.json` [ZIP]: branch-entry gates E1-E6 and
  blockers.
- `strict_neutral_bulk_frontier_report.{json,md}`.
- `dimension_report.json`: continuous estimators,
  `claim_level: internal_diagnostic_only` (stub on BW-primary runs).

## 6. Cosmology lane

- `freezeout_map_summary.json` [ZIP], `cl_comparison_report.json` [ZIP],
  `cmb_lite_comparison_report.json` [ZIP], `cosmology_observables.json`,
  `cosmology_gate_report.json` (records the gate checks verbatim,
  including any declared product-gating override).
- Boltzmann/collar chain: `finite_collar_boltzmann_bundle_report.{json,md}`,
  `oph_boltzmann_*.{json,csv}`, `B_A_*.csv`, `rho_A_a.csv`,
  `Gamma_rec_k_a.csv`, `paired_b_a_perturbation_*`, `b_a_parent_*`.
- `physical_cmb_*` reports (frontier, promotion audit, source readiness,
  output comparison): the physical-CMB gate ladder, fail-closed.
- `galaxy_proxy_report.json`, `oph_compressed_likelihood_*`,
  `cmb_source_provenance_report.json`, `no_data_use_receipt.json`.

## 7. Certificates, receipts, and provenance

- `manifest.json` [ZIP]: run manifest (patch count, flags, hashes).
- `config.yml` [ZIP], `git_commit.txt`: exact configuration and code
  revision.
- `bulk_proof_certificate_report.{json,md}` [ZIP],
  `finite_oph_theorem_contract_report.{json,md}`,
  `emergence_status_report.json`, `receipt_ladder_report.json`,
  `AUTO_THEOREM_UNIVERSE_SUMMARY.json` [ZIP],
  `transition_scale_selection_report.json` [ZIP] (2 pi clock selection
  with replay metadata), `bw_report.json`, `bw_state_derived_report.json`,
  `mandatory_controls_report.json`, `simulation_assumption_manifest.json`
  [ZIP], `large_run_readiness_report.json`.
- `mismatch_trace.csv` [ZIP]: global mismatch/repair time series.

## 8. Other lanes

- Yang-Mills gap certificate: `yang_mills_gap_certificate_report.md` +
  six CSV traces (plaquette, Wilson/Polyakov loops, refinement,
  promotion gates).
- `auto_theorem_refinement/`, `reference_vacuum_baseline/`,
  `boundary_program_report.json`, `edge_sector_heat_kernel_report.json`,
  `central_record_born_report.json`, `silence_to_observation_report.md`.

## Consumers

- `scripts/build_visualizer_zip.py <run_dir> <out.zip> .` assembles the
  web-agent bundle (256 MB hard cap) from the [ZIP] subset plus
  `scripts/README_FOR_WEB_CODING_AGENT.md` (run-agnostic implementation
  contract and claim-boundary checklist). The run-generated web brief and
  visualization manual carry the detailed scene contracts.
- `run-universe-timeline-viewer` regenerates `universe_timeline/` from a
  completed run.
- `observer-agreement-report`, `strict-neutral-bulk-report`,
  `neutral-3d-bulk-audit`, `bulk-proof-certificate`, and the other
  claim-gate commands (README section) recompute their artifacts
  post-hoc.
