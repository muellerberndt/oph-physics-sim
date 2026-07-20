# OPH universe visualizer bundle: web implementation contract

This bundle contains the data and evidence needed to visualize one finite
OPH-FPE run. Read the run identity, scale, available layers, and receipt values
from the bundled files. This document carries no hard-coded run outcome.

## Public Lovable Deko override versus local analyzer

This document historically described the local technical analyzer. Keep its
receipt tables, PASS/OPEN statuses, blocker panels, and `DEMO_ASSUMPTION`
controls in that local product only. The public Lovable Deko app instead uses
`contracts/public_cinematic_story.json` as its primary contract and must have no
gate/pass-fail UI in the main experience. Its only persistent epistemic label is
the subtle exact phrase `illustrative reconstruction`; detailed receipts may be
placed in an optional collapsed provenance drawer.

For the public app, bind
`contracts/public_visualization_anchor.json` generated from the public-safe
16k anchor. Begin with deterministic seeded disorder, then provenance-tagged
light/readback and repair iterations across exact carrier edges and exported
seams, then normal-form settling. The 16k physical runner refused before RNG,
so this settling trajectory is `synthetic`, never measured or run-anchored.
Zoom into the exact 12-port/30-edge/20-face carrier; browse all 60 A5 actions;
continue through A5-to-SM, events, H3, gravity, cosmology, and an observer-frame
finale with modular clock, visible particles/proton/composites/atoms, and
gravity. Introduce synthetic particle imagery only after the emergence
transition.

Every public scene record carries `visualizationProvenance.status` from exactly
`measured`, `computed`, `interpolated`, `synthetic`, or `frozen`. Deterministic
interpolation retains parent IDs/weights/method; model-based synthesis retains
a stable generated ID, generator/version, seed, deterministic index, and source
refs. Instructions below about missing-data non-synthesis are scientific-data
rules for the local analyzer; the public renderer may synthesize missing visual
samples only under this provenance contract and may never fabricate receipts.

## Read these files first

1. `docs/WHAT_OPH_FPE_DOES.md` gives the one-page account of the simulator.
2. `run_reports/manifest.json` and `run_reports/config.yml` identify the run.
3. `run_reports/run_highlights.json` summarizes run-derived receipt values.
4. `docs/WEB_CODING_AGENT_VISUALIZATION_BRIEF.md` defines the story and scene
   contracts generated for this run.
5. `docs/SCREEN_A5_SM_VISUALIZER_AGENT_BRIEF.md` is the authoritative focused
   contract for the primary screen-geometry and A5-to-SM views, demo-display
   controls, exact-small/scale panels, and revised story order.
6. `docs/VISUALIZATION_APP_AGENT_MANUAL.md` defines the remaining payload
   fields and stable rendering rules.
7. `docs/SIMULATION_ASSUMPTION_POLICY.md` separates computed results from
   explanatory assumptions.

When two descriptions disagree, concrete run artifacts control physical
receipt values. The focused screen/A5 brief controls the current application
structure and display semantics; stable documentation explains the remaining
semantics and does not override a run.

## The object being visualized

OPH-FPE instantiates observer-like self-reading systems: bounded software
patches with local state, ports or boundaries, readback, records, feedback or
repair moves, and public evidence bundles. A run starts from a declared finite
carrier. Neighboring patches compare shared-boundary data, attempt local
repairs, and commit stable records. Post-processing tests what those records
support.

The visualizer should make this sequence legible:

```text
bounded patches -> local readback -> overlap repair -> committed records -> gated reconstruction
```

The declared screen is a finite regulator input. Three-dimensional geometry,
physical matter, gravity, and a physical CMB receive their own downstream
receipts.

## Bundle contents

- `payload/visualization_payload.json`: canonical monolithic payload. Prefer
  server-side use when it is large.
- `payload/oph_visualizer_pack_v2.tar.zst`: chunked, content-addressed payload
  for browser delivery when present.
- `payload/visualization_export_manifest.json`: sidecar inventory and formats
  when present.
- `sidecars/`: renderer tables and binary arrays selected by the run exporter.
- `data/`: raw NumPy arrays and observer rows selected by the bundle builder.
- `run_reports/`: run-local receipts, controls, blockers, and provenance.
- `reference_viewers/`: simulator-generated examples when present. They are
  reference implementations rather than claim authorities.
- `docs/`: the generated scene brief, payload schemas, assumption policy,
  implementation manual, and scientific-scope account.

Every layer is optional unless its schema marks it required. Resolve
availability before rendering. Missing data receives an unavailable state; do
not synthesize a replacement.

## Status grammar

The application has two global modes. **Receipt mode** displays only canonical
computed data and receipts. **DEMO_ASSUMPTION mode** may force missing display
stages or freeze display values to verify an end-to-end renderer. It must show
`PHYSICAL RECEIPTS UNCHANGED` continuously. Never call a forced display a
receipt, write it to a run or production envelope, or use it to authorize
simulation work.

Every screen/A5 stage has separate **Physical** and **Display** cells. Display
states are `computed`, `forced_demo_assumption`, `frozen_target_demo`,
`blocked`, or `not_applicable`. The required master and local controls are
`Force all missing display stages`, one toggle per catalogued stage, reset,
gaps-only, target-freeze, and receipt-vs-demo comparison. Forced stages use a
gold striped treatment and the text `DEMO ASSUMPTION — NOT A RECEIPT`.

Even with force-all enabled, the interface must preserve:

```text
promotion_allowed = false
scientific_receipts_unchanged = true
SCALE_CAMPAIGN_ALLOWED = false
target_ancestry_eligible = false
```

Known W/Z/H values, `2pi`, and other exposed empirical targets are frozen
`post_exposure_validation` display references only. Do not provide a
prospective switch.

Use the same five statuses in captions, legends, drawers, screenshots, and
accessible text:

- **COMPUTED / PASSED**: the relevant concrete-run receipt is true.
- **DIAGNOSTIC DATA**: computed output without the physical promotion receipt.
- **ASSUMED VISUAL LAYER**: an explicit row from
  `simulation_assumption_manifest.json` supplies the scene element.
- **CLOSED PROMOTION GATE**: the relevant receipt is false and its blockers are
  shown.
- **UNAVAILABLE IN THIS EXPORT**: the required data is absent.

A closed gate is a scientific result, not an application error. Reserve error
styling for malformed files, failed hashes, schema violations, or renderer
failures.

## Claim boundaries

Render receipt values from the payload or run reports. Never infer a receipt
from visual resemblance or from the presence of a file.

- Low or zero mismatch does not by itself certify finite consensus. Show the
  stronger descent, completeness, confluence, replay, and terminal-form gates.
- The observer-agreement report tests gauge-frame self-consistency of views of
  one committed shared record. It does not test independently produced
  per-observer commit histories unless the run explicitly supplies that
  experiment.
- An observer-facing H3 chart or 3+1D readout is separate from a chart-blind
  neutral third-person bulk.
- Holonomy clusters, persistent defects, and worldlines are proto-particle
  diagnostics until the particle-promotion receipt passes.
- Compaction fields, stress-pair assays, and curved-spacetime renderings are
  diagnostics or assumed visual layers until the Einstein branch-entry and
  production-gravity receipts pass.
- Screen angular spectra and reference-shape comparisons are diagnostics until
  the physical source, transfer, no-data-use, and frozen-likelihood receipts
  pass.
- A replayed or source-normalized `2*pi` branch should display its source and
  discrimination status. It does not establish the full finite Lorentz
  contract.
- Paper-side constants and public-data comparisons are contextual material.
  The lattice relaxation law does not derive them.

Keep every `claimBoundary`, `policy`, `blockers`, provenance field, and
receipt-source field available in the interface. Do not replace exact blocker
text with a generic “work in progress” label.

## Core views

### Screen geometry (primary)

Use `screenA5Ladder.localCarrier`, `.federation`,
`.observerRepairBridge`, `.clockSeparation`, `.demoControls`, `.receipts`, and
`visualizationViews.screenGeometry`. Synchronize three scales: one exact
carrier, its typed seam/collar neighborhood, and the carrier federation with
the separate translucent S2 support regulator. Never identify a carrier with
an S2 point.

Render exactly 12 ports, 30 edges, 20 outward faces, six antipodal pairs, the
exact exported vertex order, and all 60 orientation-preserving A5 actions.
Provide port/edge/face permutation and composition inspection plus distinct
`1 + 3 + 3-prime + 5` structural sectors. Keep physical current-tomography
receipts separate from the finite structural decomposition. Show the complete
readback -> mismatch -> typed proposal -> proof-carrying replay -> atomic
commit/rollback -> semantic record -> observer checkpoint handoff.

### Physical H3/KMS bridge (local audit and public cinematic data)

Use `screenA5Ladder.physicalH3KmsDemoOverlay.stageNodes` for the P0--P8 bridge
from screen repair through BW/clock selection, H3 controls, semantic events,
and the frozen rung ladder. The embedded `physicalSnapshot` and
`physicalGateStatus` / `physicalScientificStatus` fields are read-only. A row
with `demoNudgeApplied=true` supplies renderer-only `displayData`; every such
field has a matching `fieldProvenance` row and must remain
`DEMO_ASSUMPTION`.

The local analyzer may show physical and display status side by side. The
public cinematic app must not present a gate dashboard: turn the overlay into
beautiful transitions, motion, and a brief explanatory caption for each rung.
Keep the watermark/assumption disclosure available, and never describe
`displayComplete` as physical success. Before rendering a nudged overlay, verify
`physicalSnapshotDigestBefore == physicalSnapshotDigestAfter` and all guards
for promotion, branch retirement, target ancestry, and scale remain false.

### A5-to-Standard-Model ladder (primary)

Render `screenA5Ladder.a5ToSm` through
`visualizationViews.a5ToStandardModel` as a dependency DAG. It is not a flat
completion checklist. The Q2 topology is:

```text
Q2_H
   or
Q2_E -> POSITIVITY_OR_POSITIVE_TRANSFER
```

The selected valid branch rejoins only at refinement/physical identification.
Show separate finite-structural, full-interacting-SM, and continuum claim
cards. Include issue #565/#566/#567/#569/#590 status, the complete
`EXACT_SMALL_ORACLE` checklist, and scale readiness. #569 includes physical
identification only at the issue aggregate; #590 is delimitation, not a
physical pass. Demo toggles can complete the drawing but can never change the
exact-small oracle or `SCALE_CAMPAIGN_ALLOWED`.

### A5 currents and W/Z continuation

Keep finite A5 incidence/decomposition separate from a measured reversible
12-channel current law. Visualize generator/full-`M6` ranks, the rank-11
negative control, `u(3) + so(3)` target, covariance, Fisher scales, odd signed
response, and refinement as individually gated objects.

The W/Z continuation is a separate source-to-pole branch. Provide selectors
for the five claim lanes and `W_ONLY / Z_ONLY / WZ / WZH`; keep W and Z pole
gates independent and require H only for WZH. Label WZH0 synthetic and
nonpromoting. Never let an external/imported pass color an OPH-native lane as
passed.

Expose a three-clock strip in these views: BW/geometric normalization (the
`1x`, `pi`, `2pi`, `4pi` comparison), A5 oriented primitive-volume clock, and
W/Z operational transition clock. These clocks are never interchangeable. A
demo-frozen `2pi` is `FROZEN DISPLAY TARGET` and leaves its physical selector
receipt false.

### Finite screen and repair

Render the declared screen points and exported fields. Animate only measured
frames or repair traces. Show mismatch and commit histories beside the screen.
An outside camera is an explanatory overview and must carry that label.

### Observer readback and shared-record agreement

Use observer-local supports, records, modular-time rows, and camera payloads.
First-person views may expose only the selected observer’s exported readback.
Agreement edges and cocycle triangles should display the shared-record scope
and shuffled controls. A null `bulk_dimension_claim` stays null.

Retain a carrier-support mini-map, port readable/writable/protected overlays,
primitive-record ancestry, and side-by-side objective/subjective panels. The
objective panel must say `EXPLANATORY OVERVIEW — NOT OBSERVER VISIBLE`. Start
camera transitions on semantic commits, prevent global H3 coordinates from
entering the subjective panel, and let users trace an H3 object back to every
contributing record and seam.

### H3 charts, records, and defect worldlines

Use the coordinate-system contract in the payload. H3 vectors are hyperboloid
spatial components unless the run declares another representation. Interpolate
with intrinsic H3 geometry. Label record objects and defect tracks according to
their actual receipt tier.

Enter H3 from a selected observer record rather than an omniscient bulk. Keep
computed H3 charts, assumed dS4, and diagnostic curvature proxies as separate
layers across synchronized spatial-slice, worldline, and source/curvature
panels. Assumed dS4 always says `ASSUMED VISUAL LAYER — NOT DERIVED`.

### Effective spacetime and gravity diagnostics

Render neutral-bulk and Einstein branch-entry gates separately. Assumed de
Sitter geometry, observer tetrads, or defect-as-matter styling require a
persistent assumption badge. A failed production-gravity receipt remains
visible beside any curved visual layer.

### Cosmology and CMB diagnostics

Distinguish run-derived screen data, pinned reference measurements, and assumed
best-fit reference shapes. Plot each with its provenance and evidence tier.
Resemblance to a measured sky cannot promote a physical-CMB receipt.

Animate the provenance handoff from finite readback to diagnostic sky pixels.
Use distinct encodings for simulation rows, pinned references, and residuals;
separate `diagnostic resemblance`, `usable comparison`, and `physical
prediction` badges. Render exported uncertainties only and link plotted points
to source field, transform, and hash.

## Multi-run interfaces

Read `run_id`, patch count, observer count, config hash, code revision, and
receipt values from each bundle. Do not infer scale from a filename. A run
switcher must preserve each bundle’s independent status and provenance.

Repository placeholders, planning configurations, and prose-only run reports
are not earned-run artifacts. Present a scale as repository-visible evidence
only when its concrete files and hashes are available.

## Accessibility and performance

- Provide keyboard access, visible focus, reduced-motion support, and text or
  table alternatives for every plot and animated state.
- Use color, shape, line style, icon, and text together for status.
- Lazy-load the active and next scene. Parse large chunks off the main thread.
- Use typed arrays, instancing, level of detail, and frustum culling for large
  carriers.
- Release GPU and decoded payload resources when a scene is left.
- Keep the generated bundle below the builder’s exclusive 256,000,000-byte
  ceiling.

## Acceptance checklist

- Run identity and provenance come from the bundle.
- `screenGeometry` and `a5ToStandardModel` are primary views and implement the
  focused brief rather than a decorative approximation.
- A carrier has exactly 12 ports, 30 edges, 20 faces, six antipodal pairs, all
  60 incidence-preserving A5 actions, and distinct `3`/`3-prime` sectors.
- Carriers, S2 support points, typed seams, and external boundaries are not
  conflated.
- The A5/SM Q2 fork is correct; issue, exact-small, and scale panels are
  present; `SCALE_CAMPAIGN_ALLOWED` remains false.
- Force-all and per-stage toggles alter only display state, retain demo
  badges/watermarks in exports, and never mutate a physical receipt.
- Frozen targets are post-exposure/display-only and the three clocks remain
  distinct.
- Every visible scientific interpretation has a receipt or assumption source.
- Computed, diagnostic, assumed, closed, and unavailable states remain
  distinguishable without color.
- First-person views expose only observer-local exported data.
- H3 geometry uses the declared coordinate contract.
- Objective and subjective cameras remain distinct and H3 objects trace back
  to observer-visible records.
- Shared-record agreement is not described as independent observer history.
- Neutral bulk, particles, gravity, and physical CMB stay closed when their
  receipts are false.
- Cosmology/CMB panels distinguish diagnostic resemblance, usable comparison,
  and physical prediction, with provenance and only exported uncertainties.
- Missing files produce explicit unavailable states.
- Captions and status labels are included in screenshots and exports.
