# OPH universe visualizer bundle: web implementation contract

This bundle contains the data and evidence needed to visualize one finite
OPH-FPE run. Read the run identity, scale, available layers, and receipt values
from the bundled files. This document carries no hard-coded run outcome.

## Read these files first

1. `docs/WHAT_OPH_FPE_DOES.md` gives the one-page account of the simulator.
2. `run_reports/manifest.json` and `run_reports/config.yml` identify the run.
3. `run_reports/run_highlights.json` summarizes run-derived receipt values.
4. `docs/WEB_CODING_AGENT_VISUALIZATION_BRIEF.md` defines the story and scene
   contracts generated for this run.
5. `docs/VISUALIZATION_APP_AGENT_MANUAL.md` defines the payload fields and
   rendering rules.
6. `docs/SIMULATION_ASSUMPTION_POLICY.md` separates computed results from
   explanatory assumptions.

When two descriptions disagree, the concrete run artifacts and their receipt
values control the display. Stable documentation explains semantics; it does
not override a run.

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

### Finite screen and repair

Render the declared screen points and exported fields. Animate only measured
frames or repair traces. Show mismatch and commit histories beside the screen.
An outside camera is an explanatory overview and must carry that label.

### Observer readback and shared-record agreement

Use observer-local supports, records, modular-time rows, and camera payloads.
First-person views may expose only the selected observer’s exported readback.
Agreement edges and cocycle triangles should display the shared-record scope
and shuffled controls. A null `bulk_dimension_claim` stays null.

### H3 charts, records, and defect worldlines

Use the coordinate-system contract in the payload. H3 vectors are hyperboloid
spatial components unless the run declares another representation. Interpolate
with intrinsic H3 geometry. Label record objects and defect tracks according to
their actual receipt tier.

### Effective spacetime and gravity diagnostics

Render neutral-bulk and Einstein branch-entry gates separately. Assumed de
Sitter geometry, observer tetrads, or defect-as-matter styling require a
persistent assumption badge. A failed production-gravity receipt remains
visible beside any curved visual layer.

### Cosmology and CMB diagnostics

Distinguish run-derived screen data, pinned reference measurements, and assumed
best-fit reference shapes. Plot each with its provenance and evidence tier.
Resemblance to a measured sky cannot promote a physical-CMB receipt.

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
- Every visible scientific interpretation has a receipt or assumption source.
- Computed, diagnostic, assumed, closed, and unavailable states remain
  distinguishable without color.
- First-person views expose only observer-local exported data.
- H3 geometry uses the declared coordinate contract.
- Shared-record agreement is not described as independent observer history.
- Neutral bulk, particles, gravity, and physical CMB stay closed when their
  receipts are false.
- Missing files produce explicit unavailable states.
- Captions and status labels are included in screenshots and exports.
