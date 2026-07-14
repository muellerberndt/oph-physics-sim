# OPH Universe Visualizer Bundle — Instructions for the Web Coding Agent

This bundle contains everything needed to build an interactive web visualization of one
complete OPH (Observer-Patch Holography) universe simulation run:
`oph_universe_64k_visual_release_20260711` — 65,536 patches on a finite S2 screen,
128 repair cycles, 1,024 observers (256 exported with full detail), 52 consensus H3
objects, and 128 proto-particle candidate worldlines.

## What OPH is (one paragraph you must get right)

OPH instantiates **observer-like self-reading systems**: bounded software patches with
local state, ports/boundaries, readback, records, feedback/repair moves, and public
receipts. The universe is not "particles in a box" — space (an H3 chart), time
(observer-local modular time), and matter candidates (persistent record objects and
defect worldlines) all *emerge* from observers repairing mismatches on their shared
boundaries. Your first visible text should say this.

## The product

An interactive single-page 3D web app with these views (full per-view contracts in
`docs/WEB_CODING_AGENT_VISUALIZATION_BRIEF.md` and `docs/VISUALIZATION_APP_AGENT_MANUAL.md`):

1. **Fluctuating vacuum / finite screen** — the 65,536-point S2 screen, colored by
   readback fields, with a field selector (`screen.fields`) and a **measured time
   animation**: `screen.evolution` (48 frames over 128 cycles) shows mismatch decaying
   from 205k edges to zero as the universe "freezes in" its records. Full-resolution
   animation frames are the `sidecars/screen_frames_<field>_65536x48.bin` files.
2. **Overlap repair between observers** — `observerModularTime.overlapLinks` (655 links; the 512 strongest carry measured trajectories).
   Each link carries a **measured per-cycle repair trajectory** on the pair's shared
   support (`repairTrajectory[*].overlapMismatchDensity`); pulse each link as the
   timeline plays. This is the core OPH dynamic: neighboring observers negotiating
   their shared boundary until records commit.
3. **3D bulk (H3 chart)** — `consensusBulk.objects` (52 objects, positions are
   hyperboloid spatial components; use the intrinsic H3 math in the docs, never
   Euclidean/Poincaré) plus `consensusBulk.protoParticleCandidates.worldlines`
   (128 tracks; prefer the organic defect population source) animated over cycles.
4. **3+1D observer cameras with interactive timeline** — 256 `subjectiveObserverCameras`,
   each with `eye/lookAt/up/right/forward/fovDegrees/h3TangentFrame`, 96 time frames,
   per-frame `visibleProtoWorldlines` (projected into the observer's own tangent-frame
   screen), and `visibleConsensusObjects` (the static object field of the first-person
   view). Provide an observer selector plus a global timeline slider that drives all
   views simultaneously.
5. **Emergent curved spacetime** — `emergentCurvedSpacetime` compaction/potential fields
   for fog / warped-grid rendering, with the two-defect assays as animated trajectories.
6. **CMB diagnostics** — screen angular spectra and the Planck-reference comparison
   (diagnostic only; see gates below).
7. **Mini-universe repair playback** — `smallUniverse` exact repair frames if content is
   available; otherwise show the receipt-only card as the brief specifies.

## Files

- `payload/visualization_payload.json` — the canonical payload (~614 MB raw JSON; all
  sections). Load it server-side or stream it; do not ship it uncompressed to browsers.
- `payload/oph_visualizer_pack_v2.tar.zst` — the same payload chunked and
  content-addressed (~28 MB) for on-demand browser loading; `payload_index.json` inside
  describes sections and `$chunkList` reconstruction. Prefer this for the deployed app.
- `payload/visualization_export_manifest.json` — sidecar inventory with formats.
- `sidecars/*.csv` — flat renderer tables (cameras, camera frames, ~393k proto-worldline
  sightings, H3 objects, worldline events, curvature fields, cluster tracks,
  `repair_trace_full.csv`).
- `sidecars/screen_full_65536.bin` — float32-le `[x,y,z,value]` rows for the full screen.
- `sidecars/screen_frames_<field>_65536x48.bin` — float32-le frame-major animation
  frames (same row order as `screen_full_65536.bin`; frame cycles in
  `screen.evolution.cycles`).
- `sidecars/observers_full_*.json`, `sidecars/cameras_full_*.json` — full observer and
  camera transform tables.
- `data/*.npz` — raw NumPy arrays (freezeout fields, evolution frames, harmonic
  time-trace spectra) for any preprocessing you want to do at build time.
- `data/observer_views.jsonl` — the complete observer table (all 1,028 observers, one
  JSON object per line: axis, support nodes, record-signature spectra, packet
  histograms, per-bin repair-current tensors). Use this to extend the observer layer
  beyond the 256 fully exported cameras.
- `reference_viewers/*.html` — three self-contained viewers the simulator itself
  emitted from this run. They are *reference implementations*, not the product; read
  them to see working field/H3/CMB rendering code against the same data.
- `run_reports/` — run manifest, config, receipts, consensus/CMB/defect reports for
  provenance panels.
- `docs/` — the payload JSON schema, the two per-view contract documents, and the
  simulation-assumption policy.

## Claim boundaries (non-negotiable)

Every substantive layer is receipt-gated. Render gates from `receipts` objects in the
payload — never hard-code success:

- `simulation_assumed_visual_universe_receipt: true` — the dS4 background, H3 camera
  embedding, and defect-as-matter rendering are **explicit simulation assumptions**
  (gold "assumed" badges), not derived physics.
- `strict_neutral_third_person_bulk_receipt: false` and
  `physical_cmb_prediction_receipt: false` — show these as closed promotion gates, not
  errors. Do not label anything "physical prediction", "solved metric", or "literal QFT
  vacuum".
- The observer-facing receipts that ARE true in this run: observer-like self-reading
  system, observer modular time, H3 object population, observer-facing 3+1D H3
  experience, theorem-assisted consensus 3D bulk readout.
- Keep the payload's `claimBoundary` strings visible where the docs require them.

## Practical notes

- Timeline: cycles 0–127. The repair story happens in cycles 0–64 (mismatch 205,745 → 0);
  committed fraction reaches 1.0 by ~cycle 96. Default the timeline scrubber to the
  active phase.
- The two freezeout fields `repair_load`/`local_mismatch_density` are all-zero in the
  *static* freezeout snapshot (captured post-convergence) — the *animation* frames carry
  the live mismatch story instead. `screen.fields` already excludes constant fields.
- All H3 vec3s are hyperboloid spatial components (`coordinateSystems` in the payload):
  lift `x` to `X=(sqrt(1+|x|²), x)`, signature (-,+,+,+), `d=acosh(-η(X,Y))`, geodesic
  interpolation. The docs contain the exact formulas; the cameras' `h3TangentFrame` is
  the authoritative projection frame.
- Performance: prefer the .tar.zst pack chunks or the binary sidecars for anything big;
  the monolithic payload JSON is provided for convenience and server-side tooling.

## New in this bundle: observer agreement graph (consensus formation)

`run_reports/observer_agreement_report.json` carries the observer
mutual-agreement certificate: OPH's operational statement that a shared
spacetime is what overlapping observers AGREE on, never a container.

Data for a dedicated "consensus" scene:

- Nodes: patch observers (`pair_records[].observer_a/b` ids; positions via
  each observer's support centroid from `data/observer_views.jsonl`
  `support_nodes` + screen `points` in `data/s3_gauge_state.npz`).
- Edges: `pair_records[]` (up to 64 sampled pairs), each with
  `overlap_edges`, `defect` (0 = the two observers' charts glue exactly;
  render green), `section_unique` (solid vs dashed), `components`.
- Triangles: `triple_records[]` with `cocycle_defect` (0 = the three
  re-gaugings compose exactly around the loop; shade the triangle).
- Headline badges: `MUTUAL_GAUGE_CHART_AGREEMENT_RECEIPT`,
  `OBSERVER_SPACETIME_CONSENSUS_RECEIPT`, and `blockers` (render blockers
  verbatim; they are the claim boundary).
- Controls panel: `control.median_defect_shuffled` (what disagreement
  would look like) vs `pair_agreement.median_defect` (what the record
  produces). The contrast IS the physics story.
- Policy string: display `policy` verbatim somewhere visible. Never
  render a fractional "bulk dimension" anywhere: `bulk_dimension_claim`
  is null by construction.

`data/s3_gauge_state.npz` additionally enables a gauge-edge layer:
`left`, `right` (patch indices into `points`), `gauge` (0..5, S3 element
per edge; class map: 0 identity, {1,2,5} transpositions, {3,4}
three-cycles). Rendering transposition-class edges as a faint web over
the screen shows the defect-carrying skeleton that the S3 class density
field summarizes.

`run_reports/screen_parity_report.json` (if present) reports the
parity-odd pseudo-scalar with mirror and shuffle controls; a small
diagnostic card (chi, z-score, mirror flip check) is enough.

## New in this bundle: effective-spacetime status lane (2026-07-14)

This bundle adds the third-person "effective 4D spacetime" direction-check
artifacts. The story for a status panel or dedicated scene:

- `run_reports/observer_modular_experience_report.json`: the
  first-person claim. `observer_facing_3p1d_h3_experience_receipt` with
  its four component gates (modular time, 2 pi KMS clock, conformal H3
  chart, H3 modular response). In bundles from 2026-07-14 onward this
  receipt is earned under the gauge-covariant contract with production
  sector-repair replay (`run_reports/transition_scale_selection_report.json`
  carries `two_pi_selected`, `production_sector_repair_replayed`,
  `probe_side_selection: production_coin`).
- `run_reports/einstein_bridge_manifest.json`: the Einstein branch-entry
  gates E1-E6 with named blockers. Render the gate ladder verbatim,
  green/red per gate; the `blockers` list is the claim boundary.
- `run_reports/neutral_3d_bulk_audit_report.md` (+ .json): the strict
  third-person bulk audit: refinement-ladder status
  (4k/16k/64k/256k), integer rank-3 candidate stability, leakage and
  control gates. Dimensionality renders ONLY as integer rank candidates
  plus stability verdicts; fractional dimension estimates in sweep rows
  are estimator diagnostics and must never headline.
- Sidecars `emergent_curved_spacetime.json` (+ `_curvature_proxy.csv`,
  `_time_slices.csv`, `_continuous_field.csv`) and
  `effective_string_theory.json`, `string_vacuum_selector_*.csv`: the
  assumed-bridge visualization layers for the curved-spacetime and
  string-side effective descriptions (their manifests carry
  `simulation_assumptions` scope; keep that label visible).

Suggested scene: a three-column "effective description status" board
(SR/GR bridge, Standard-Model/string selector, first-person 3+1
experience), each column listing its receipts as shipped in the JSON,
with the claim-boundary strings rendered verbatim under each column.

## New scene ideas for the 1M bundle (2026-07-14, in value order)

1. **Proto-particle block universe.** Render defect worldlines as world-tubes
   in (screen x cycle) 3D: tube color = S3 class (transposition vs
   threecycle), radius = support size, glow at near-contact events.
   Data: `run_reports/defect_timeline_report.json` (worldlines + events +
   centroids per cycle), `defect_h3_worldlines_report.json`,
   `defect_interaction_report.json` (900 verified encounter candidates at
   64k; `fusion_identity_fraction` labels each encounter's composition
   channel). This is the first spacetime diagram of OPH proto-matter.
2. **Consensus crystallization graph.** Observers as nodes at support
   centroids; edges appear as pair re-gauging defects hit 0; triangles
   shade in as cocycles close. End state: one connected agreement web =
   the operational meaning of a shared spacetime.
   Data: `run_reports/observer_agreement_report.json` (pair_records,
   triple_records, controls), `data/observer_views.jsonl`,
   `data/s3_gauge_state.npz` points.
3. **The emergence gate board.** An animated receipt ladder across the
   scale ladder (4k, 16k, 64k, 128k, 256k, 1M): the four 3+1 experience
   gates, the Einstein branch-entry gates E1-E6, the neutral-bulk
   blockers, each lighting green as runs earn them (with the 2026-07-14
   2 pi re-earn as a highlighted event). Honest storytelling: gates that
   stay red render red, with their blocker strings.
   Data: `run_reports/observer_modular_experience_report.json`,
   `einstein_bridge_manifest.json`, `neutral_3d_bulk_audit_report.json`,
   `transition_scale_selection_report.json` per bundle.
4. **Gauge-relativity toggle.** A control that re-dresses the visible
   sector labels into a chosen observer's chart frame and back,
   with the recovered re-gauging map animating between charts: same
   record, different descriptions, exact translation. Data:
   `s3_gauge_state.npz` + `observer_agreement_report.json`
   (`pair_records` carry the per-pair recovered maps for sampled pairs).
5. **The 2 pi clock dial.** Radial dial of transition-scale selection
   scores across candidate scales with the winner at 2 pi; a needle
   animation landing on 2 pi = the universe selecting its clock.
   Data: `run_reports/transition_scale_selection_report.json`
   (per-scale scores, `two_pi_selected`, replay metadata).
6. **Curvature weather.** Animated heatmap of the curvature proxy over
   the H3 chart across time slices (assumed-bridge layer; keep the
   simulation-assumption label visible).
   Data: `sidecars/emergent_curved_spacetime_curvature_proxy.csv`,
   `_time_slices.csv`, `emergent_curved_spacetime.json`.
7. **Scale-ladder fly-through.** Nested spheres 4k -> 1M; zooming
   crossfades the freezeout fields of each scale; a side meter shows
   which statistics stay fixed across the zoom (the refinement story).
   Data: per-bundle `data/freezeout_fields.npz` from the ladder runs
   (this bundle carries its own scale; the ladder bundles ship
   separately).
8. **Repair lightning / healing front.** Transient arcs where mismatch
   repairs fire per cycle, an advancing freezeout aurora behind them.
   Data: `data/screen_evolution_frames.npz` WHERE PRESENT (raw frames
   are config-gated; check availability before building the scene) plus
   `run_reports/mismatch_trace.csv` for the global curve.

Availability note: scenes 1-5 are fully served by every 2026-07-14+
bundle; 6 ships in all e4/e5 bundles; 7 needs the sibling ladder
bundles; 8 depends on `save_raw_frames` in the source run config.

## Flagship new scene: the emergent bulk (consensus solidification)

OPH doctrine, rendered literally: a bulk does not exist as a container;
it is what observers agree on. This scene builds the bulk on screen from
nothing except agreement data.

Data: `data/agreement_bulk_field.npz` with per-patch arrays over
`points`:

- `coverage`: how many cohort observers hold the patch in support
  (subjective reach; render as a faint ghost layer);
- `pair_certified`: how many evaluated zero-defect re-gauged observer
  pairs contain the patch in their overlap (agreement solidity; render
  as opacity/solidity);
- `triple_certified`: cocycle-closed triple coverage (the strongest
  agreement; render as a crystalline highlight).

Plus `run_reports/agreement_bulk_field_summary.json` (fractions,
multiplicity maxima, and the claim boundary: zero certified multiplicity
means UNTESTED, never disagreement; disagreement would show as nonzero
pair defects in `observer_agreement_report.json`).

Interactions that carry the physics:

- **Consensus slider**: threshold k sweeps 0 -> max multiplicity; the
  rendered bulk is exactly the patch set with `pair_certified >= k`.
  At k = 0 everything is ghost; raising k dissolves the subjective fringe
  and leaves the agreed core. That sweep IS the OPH statement.
- **Solidification play**: animate k downward from max to 0 so the bulk
  crystallizes outward from the most-agreed regions.
- **Subjectivity heatmap**: `(coverage - pair_certified) / max(coverage, 1)`
  colors the untested-overlap fringe.
- **Observer parliament** (companion panel): force-directed graph of the
  cohort, edges = certified pairs, triangles = closed cocycles
  (`observer_agreement_report.json` pair/triple records); clicking an
  observer highlights its support patches in the main view.
- H3 lift: reuse the existing camera/H3 contract to render the same
  multiplicity field in the hyperboloid chart; the ghost-vs-solid story
  is identical there.

## Run highlights panel and the run switcher

Every bundle carries `run_reports/RUN_HIGHLIGHTS.md` and
`run_reports/run_highlights.json`: this run's milestone receipts (from
its own gate values), run-derived diagnostics, the curated near-fit
scoreboard against public data, and the banked negatives. Render it as a
"highlights" panel: milestone ladder with PASS/open states, the near-fit
table with pulls, the claim-boundary string verbatim.

Bundles are per run (64k, 128k, 1M, ...). `run_highlights.json` carries
`run_id` and a human `label` with patch and observer counts: when the app
loads more than one bundle, use those for a run switcher so the viewer
can flip between scales and watch which milestones and diagnostics move.
The near-fit table is paper-side and identical across bundles by design;
the milestone ladder and diagnostics are where scale shows.

Source excerpts ride in `docs/SCALING_MILESTONE_ESTIMATES_2026-07-13.md`
and `docs/BEST_OF_PUBLIC_DATA_COMPARISONS.md` inside the bundle.

Note on large runs: when the full `observer_views.jsonl` exceeds the
bundle budget, the bundle ships `data/observer_supports.jsonl` instead
(one row per observer: `observer_id`, `support_nodes`,
`modular_depth_mean`). Every consensus/parliament scene works from that
slim file; the heavyweight per-observer spectra stay in the run
directory.
