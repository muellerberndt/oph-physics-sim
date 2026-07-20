# Screen geometry and A5-to-SM visualizer agent brief

Date: 2026-07-20

This brief is the implementation contract for the next OPH visualizer revision.
It supplements `VISUALIZATION_APP_AGENT_MANUAL.md`. The visualizer should now
make the finite screen architecture and the A5-to-Standard-Model ladder the
main explanatory spine while retaining the observer camera, repair, H3,
curved-spacetime, CMB, and comparable-observation views.

## Authoritative two-product split

There are two separate products. Do not merge their presentation rules.

### 1. Public Lovable Deko app

The public app is a beautiful, gate-free cinematic explanation. Its primary UX
must not contain receipt tables, PASS/OPEN columns, gate matrices, force-stage
controls, blocker dashboards, or a scientific-control-room layout. Keep the
exact lowercase disclosure `illustrative reconstruction` quietly but
persistently visible. A collapsed provenance drawer may expose record IDs,
source refs, hashes, and provenance status to interested readers.

The public renderer consumes
`contracts/public_cinematic_story.json`. The raw screen/A5 contract, receipt
snapshot, and export manifest are secondary provenance sources only. If any
later section of this brief asks for persistent receipt controls, two-column
physical/display status, demo watermarks, or gate colors, that instruction
applies **only to the local analyzer** and is overridden for the public app.
Bind the public story to the validated copy of
`survival-proof-4/outputs/run_16k/visualization_anchor.json` by exporting with
`--public-anchor`. That anchor supplies the exact 16,384-carrier and 16,383-seam
structural census, its hashes, seed 20260751, and the explicit pre-RNG refusal
boundary; it supplies no dynamics or settling trajectory.

The public story is one continuous camera journey:

1. Begin with deterministic seeded disorder in the port/readback state of a
   huge visible federation of exact icosahedral compute elements. The screen
   must not look pre-solved, prepopulated, or already at its target geometry.
2. Propagate provenance-tagged light/readback iterations inside visible
   carriers and across the exported seam graph. Show mismatch, repair, and
   convergence toward the fixed-point observable normal form. The current 16k
   physical attempt refused before RNG and therefore supplies no measured or
   run-anchored settling samples; the explanatory settling sequence is
   deterministic and `synthetic`.
3. Fly into one carrier and show its exact 12 ports, 30 edges, 20 triangular
   faces, six antipodal pairs, local processing/readback, and relationship to
   the global screen without substituting a decorative hexagon or sphere.
4. Browse and animate all 60 exported A5 actions on the twelve ports, then
   resolve the exact `1 + 3 + 3-prime + 5` sector decomposition.
5. Unfold the A5-to-SM dependency story, preserving the actual Q2 fork. Use a
   narrative dependency animation rather than a gate checklist.
6. Only after the repair-to-emergence transition, introduce the particle
   family, worldlines, interaction events, and atom/composite records. Join
   them by exported IDs. Model-based imagery should approximate available
   frozen 16k records as closely as possible and remain provenance-tagged.
7. Continue through H3 event geometry, gravity-response imagery, cosmology,
   and CMB/observer-facing sky layers.
8. Finish inside one observer-frame spacetime with the modular clock visible,
   particle families in motion, a clearly visible proton/composite and atom
   family, and gravity responding around them.

Use cinematic depth, restrained glow, legible transitions, instancing, level
of detail, bounded visible windows, hidden-tab pausing, and a composed static
frame for reduced-motion users. Propagate pulses through the actual exported
seam endpoints; do not draw generic neighbor links when seam records exist.

Every scene record must carry `visualizationProvenance.status`, with exactly one
of these values:

```text
measured
computed
interpolated
synthetic
frozen
```

Preserve the envelope through every transform and ID join. Interpolation is
allowed only deterministically between adjacent exported samples of the same
stable actor and must name both parents, weights, and method. If valid parents
do not exist, deterministic model-based synthesis is allowed for visualization:
give the generated sample a stable ID, generator ID/version, seed, deterministic
index, source refs, and status `synthetic`. Never infer `measured`; never call
the refused pre-RNG 16k settling path run-anchored; never relabel a frozen target
as computed.

### 2. Local technical analyzer

The local app is the audit instrument. It supports:

1. **Receipt mode**, which shows only recomputed scientific and structural
   receipts; and
2. **Demo-assumption mode**, which lets a user freeze missing objects or force
   display gates so the complete proposed architecture can be exercised.

Demo-assumption mode is a renderer test and explanatory model. It must never
change a scientific receipt, write into a source bundle, authorize a scale
campaign, relabel an exposed target as prospective, or appear in a production
evidence envelope. All receipt/gate/status instructions below are local-analyzer
instructions unless a section explicitly says public.

## Authoritative data

Read the canonical `visualization_payload.json` and validate it before
rendering.  The new screen/A5 content lives at:

```text
screenA5Ladder.localCarrier
screenA5Ladder.federation
screenA5Ladder.observerRepairBridge
screenA5Ladder.a5ToSm
screenA5Ladder.clockSeparation
screenA5Ladder.demoControls
screenA5Ladder.receipts
screenA5Ladder.claimBoundary
```

The corresponding panel contracts are:

```text
visualizationViews.screenGeometry
visualizationViews.a5ToStandardModel
```

Continue to treat these existing paths as authoritative:

```text
screen
smallUniverse
observerModularTime
subjectiveObserverCameras
consensusBulk
emergentCurvedSpacetime
assumedDs4Spacetime
cmbComparison
comparableObservations
paperAccuracy
visualizationRenderData
```

The checked-in force-all example is
`configs/screen_a5_demo_force_all.json`.  It is an explicit UI/demo input,
not a simulator campaign configuration and not a production evidence file.

Never infer a physical pass from a node color, a completed animation, a demo
toggle, or the existence of renderer data.

## Global application structure

The first screen is the visualizer, not a marketing landing page.  Use a
compact chapter rail with these top-level destinations:

1. Screen architecture
2. Federation and repair
3. Observer camera
4. A5 currents
5. A5-to-SM ladder
6. W/Z continuation
7. H3 and cosmology
8. CMB and observations
9. Receipts, assumptions, and provenance

Keep the selected time, carrier, observer, stage, and display mode in one
shared application state.  A selection in one view should remain highlighted
when the user changes views.

The persistent top bar must contain:

- `Receipt mode` / `Demo-assumption mode` segmented control;
- a visible `PHYSICAL RECEIPTS UNCHANGED` indicator in demo mode;
- global time/playback controls;
- selected carrier and observer IDs;
- open-blocker count;
- provenance drawer button; and
- reduced-motion and contrast controls.

## Demo-assumption and forced-display controls

The purpose of forcing is to verify that all UI paths, animations, layouts,
and downstream visual handoffs work before the mathematical producers close.
It is not a shortcut into the scientific DAG.

Render each stage with two separate status cells:

| Column | Meaning |
|---|---|
| Physical | Canonical recomputed receipt/status from the payload |
| Display | Computed data, frozen demo assumption, or unavailable |

Never replace the physical cell when a display toggle changes.  The permitted
display states are:

- `computed`: renderer is using exported data;
- `forced_demo_assumption`: renderer is using an explicit forced stage;
- `frozen_target_demo`: renderer is using an explicit display value;
- `blocked`: neither computed nor forced; and
- `not_applicable`: outside the chosen story scope.

Required controls:

- `Force all missing display stages` master toggle;
- one toggle per stage in `demoControls.toggleCatalog`;
- `Reset demo assumptions`;
- `Show only gaps`;
- `Show target freezes`;
- `Compare receipt vs demo` split view; and
- a JSON/export drawer containing the active demo state.

For every forced stage show:

- a gold striped outline;
- `DEMO ASSUMPTION — NOT A RECEIPT` badge;
- the reason it was forced;
- upstream physical blockers;
- whether it is a frozen structure, frozen numeric value, or forced handoff;
  and
- a pointer to the canonical physical status.

The master toggle may complete the visual path, but the payload and UI must
continue to show:

```text
promotion_allowed = false
scientific_receipts_unchanged = true
SCALE_CAMPAIGN_ALLOWED = false
target_ancestry_eligible = false
```

Do not send demo settings to a simulator run endpoint.  Do not persist them in
a scientific run directory.  Browser persistence, if desired, must use a
separate `oph_visualizer_demo_state` key and must never be presented as
provenance.  Screenshots and videos made in demo mode must include the
watermark and a compact frozen-value legend.

Already exposed W/Z/H values or other empirical targets are
`post_exposure_validation` display references.  The UI must not offer a
`prospective` switch.

## View: screen architecture

This is now the primary view.  It should communicate that the simulator is a
large federation of local regular icosahedral carriers, not one global
icosahedron and not a collection of S2 points masquerading as carriers.

### Three synchronized scales

Use three linked canvases:

1. **One carrier.** A clean 3D regular icosahedron with 12 port vertices, 30
   edges, 20 outward faces, six antipodal port pairs, and an orientation
   marker.
2. **A local neighborhood.** Several carriers joined by typed seam/collar
   bundles.  Draw only the ports participating in each seam and leave external
   ports visibly terminated or measured.
3. **The federation/support view.** Pull back to many carriers and show the S2
   support regulator as a translucent separate layer.  A carrier must never be
   rendered as identical to one S2 mesh point.

Selecting any object synchronizes all three views.  For example, selecting a
port highlights its antipode, incident five edges, incident five faces, seam
membership, readback channel, and current A5 sector.

### Carrier rendering

Use exact fields from `localCarrier`; do not regenerate a decorative
icosahedron with a different vertex order.  Recommended encodings:

- ports: luminous small spheres with stable integer/hidden labels;
- antipodes: paired arcs through a faint center point;
- edges: low-opacity silver lines, brighter on activity;
- faces: translucent outward-oriented triangular facets;
- seam ports: teal rings;
- external boundary ports: amber caps;
- protected ports/registers: red-violet hatch; and
- selected local state amplitude: radial halo or glyph, never vertex motion
  that changes the geometry.

Provide toggles for ports, edges, faces, antipodes, face normals, seams,
external boundaries, local state, and A5 orbit traces.  Include a 2D unfolded
incidence/net view for accessibility and exact inspection.

### A5 action explorer

The user must be able to:

- scrub through all 60 orientation-preserving actions;
- animate a selected group action as a port permutation;
- see cycle notation and the induced edge/face permutation;
- pin two actions and inspect their composition;
- reset to identity; and
- turn on an orbit trail for a port, edge, or face.

Do not suggest that order-60 incidence alone proves physical current
tomography or the Standard Model.  Show structural conformance and physical
current receipts separately.

### `1 + 3 + 3-prime + 5` sector view

Add a sector inspector next to the carrier:

- one scalar channel;
- two visually distinct triplet channels, labelled `3` and `3-prime`; and
- one five-dimensional channel.

Use a fixed accessible palette and never collapse the two triplets into one
color.  A port heatmap, spectrum/multiplicity table, and selectable basis
projection should update together.  When only the structural Laplacian
decomposition exists, label it `finite structural A5 response`.  The physical
rank-twelve current badge remains separate.

### Repair and observer handoff

Show the complete local sequence:

```text
port readback
  -> overlap mismatch
  -> typed collar proposal
  -> proof-carrying validation
  -> atomic commit or rollback
  -> semantic record commit
  -> observer-visible checkpoint
```

Use a stepper with before/after state, visible read set, writable set,
protected registers, mismatch ledger, decision, and record ancestry.  A raw
serialized receipt is a grey diagnostic; a primitive replay envelope is a
teal software-replay badge.  Neither should be styled as a physical-emergence
receipt.

## View: A5 currents

Transition from the carrier's structural sector decomposition to the proposed
physical current experiment.  Keep both on screen so their distinction is
obvious.

The view should show:

- the twelve two-sided response channels;
- the reversible/lossless response requirement;
- rank tests for the generator map and the full `M6` inner-derivation image;
- the rank-11 block-only negative control;
- the `u(3) + so(3)` closure target;
- A5 covariance residuals;
- four Fisher-scale equations;
- signed odd-response/orientation evidence; and
- refinement intertwiners.

In receipt mode, absent measured response histories leave these channels
open.  In demo mode, the display can animate a frozen full-rank response, but
the entire layer must retain the demo watermark.

## View: A5-to-Standard-Model ladder

Render `a5ToSm.nodes` and `a5ToSm.edges` as a dependency DAG, not as a flat
checklist.  Use horizontally grouped rungs:

1. Root and geometry
2. Reversible current
3. Global form and Spin/exchange
4. Source registry, scalar, and family
5. Q1 local action
6. Alternative Q2 construction
7. Refinement and physical identification
8. Full interacting additions
9. Optional continuum

The Q2 region must visibly fork:

```text
Q2_H
   or
Q2_E -> POSITIVITY_OR_POSITIVE_TRANSFER
```

and rejoin only at refinement/physical-identification.  A bare Q2-E node must
not visually close the physical branch.

Use these node styles:

- passed physical: solid teal;
- open/missing producer: hollow amber;
- unresolved/non-identifiable: violet stipple;
- failed submitted evidence: red outline;
- not applicable: muted grey; and
- forced demo: gold diagonal stripes over the original physical style.

Clicking a node opens:

- purpose and claim boundary;
- hard and alternative dependencies;
- required receipt keys;
- route alternatives;
- producer/checker identity when present;
- blockers;
- physical status;
- demo display status; and
- downstream claims affected.

### Required semantic callouts

Keep these statements visible in the relevant detail drawers:

- A5 incidence does not by itself produce a physical current law.
- The compact-current theorem is a conditional implication whose physical
  hypotheses require independent receipts.
- The oriented-volume clock is not the BW `2pi` clock or the W/Z unit clock.
- The Z6/global-form result needs physical loop/category/deck evidence.
- A rank-15 module projector is not yet a matter spectrum.
- Scalar quantum numbers do not prove a Higgs phase.
- Family attachment does not by itself provide Yukawa matrices.
- Q1 is not Q2, and finite Q2 is not a continuum Wightman theory.
- Physical identification is an external evidence gate, not simulator
  self-certification.

### Claim-tier summary

Above the DAG show three independent summary cards:

- finite structural physical model;
- full interacting Standard Model; and
- nonperturbative continuum/Wightman theory.

Each card lists its exact conjunction.  Demo completion may make the story
path visually continuous, but the physical card remains open until all
canonical stages pass.

### Issue and exact-small drawer

Add a drawer for issue closure and campaign readiness:

- #565, #566, #567, and #569 list required stages;
- #569 includes physical identification at the issue aggregate, not as a
  circular dependency of family attachment;
- #590 is a delimitation closure, not a physical pass;
- `EXACT_SMALL_ORACLE` lists every required exact-small check; and
- `SCALE_CAMPAIGN_ALLOWED` stays false until the exact-small oracle and frozen
  same-verifier/root/threshold/control/seed/grid conditions pass.

A demo force must never change the last two gates.

## View: W/Z continuation

Place the W/Z continuation to the right of the finite A5/SM ladder.  It is a
separate quantitative source-to-pole branch, not an automatic output of A5.

Offer lane and scope selectors:

```text
OPH_CHART_ONLY
EXTERNAL_SM_EFT_VALIDATION
OPH_NATIVE_DIMENSIONLESS
OPH_NATIVE_PHYSICAL
TARGET_COMPARISON_ONLY

W_ONLY / Z_ONLY / WZ / WZH
```

The selected lane should reveal its exact conjunction.  W and Z current-pole
gates are separate.  H is required only for `WZH`.  Passing an imported lane
must never color an OPH-native lane green.

Show the WZH engineering ladder (`WZH0` through `WZH5`) with `WZH0` explicitly
labelled synthetic/nonpromoting.  Display direct/converted FJ engines,
BRST/ST/Ward/Nielsen checks, analytic continuation, simple-pole enclosure,
physical-current overlap, source law/covariance, and operational clock as
separate objects rather than one `pole passed` badge.

## Three-clock comparison strip

Every screen/A5/WZ view should have access to a compact clock strip showing:

1. BW/geometric normalization, whose campaign compares `1x`, `pi`, `2pi`, and
   `4pi`;
2. A5 oriented primitive-volume clock used in global-form descent; and
3. W/Z operational transition clock used to attach physical units.

In demo mode, `2pi` may be frozen for rendering.  Label the value
`FROZEN DISPLAY TARGET`, show that its physical selector receipt is false, and
do not reuse it for the other two clocks.

## Observer camera and repair improvements

Retain the existing observer/camera view, with these improvements:

- a persistent mini-map showing the selected observer's connected carrier
  support;
- port-level visible/readable/writable overlays;
- a record-ancestry ribbon from primitive commit to visible object;
- side-by-side objective explanatory overview and subjective local camera;
- a strict mask that prevents hidden global H3 coordinates entering the
  subjective panel;
- camera/time transitions synchronized to semantic commits rather than a
  generic loop; and
- one-click return from an H3 object to all contributing records and seams.

The objective view must say `EXPLANATORY OVERVIEW — NOT OBSERVER VISIBLE`.

## H3, cosmology, and CMB improvements

The H3/cosmology views remain important but should now read as downstream of
the screen/observer ladder.

### H3 and curved spacetime

- Start from the selected observer record, then reveal its H3 chart location;
  never start with an unexplained omniscient bulk.
- Use the declared hyperboloid metric and geodesic interpolation.
- Provide synchronized spatial slice, spacetime/worldline, and
  source/curvature-proxy panels.
- Separate the computed H3 chart, assumed dS4 background, and diagnostic
  curvature proxy in the layer legend.
- Put `ASSUMED VISUAL LAYER — NOT DERIVED` directly on dS4 geometry.
- Add a before/after split for physical Einstein-branch entry.  The assumed
  scene may remain beautiful while the physical side is visibly blocked.

### CMB and observations

- Animate the provenance handoff from finite screen readback to diagnostic sky
  pixels.
- Show simulation rows, pinned observational/reference rows, and residuals in
  distinct colors.
- Keep `diagnostic resemblance`, `usable comparison`, and `physical
  prediction` as three separate badges.
- Include uncertainty/error layers when exported; never invent them from line
  thickness.
- Link every plotted point back to its source field, transform, and hash.

## Story mode

The guided story should use this revised order:

1. One twelve-port carrier reads itself.
2. The exact icosahedral incidence and A5 actions appear.
3. Many carriers form a typed federation.
4. Overlap mismatch produces a proof-carrying repair and record.
5. A bounded observer reads the committed record.
6. Reversible A5 current tomography is distinguished from repair.
7. The A5-to-SM dependency ladder unfolds.
8. The quantitative W/Z continuation branches from a complete source packet.
9. One observer's H3/cosmology view appears.
10. CMB/observation comparisons close the explanatory tour.

If demo-assumption mode is active, introduce it before the first forced node
and retain the watermark for all later chapters.

## Local streaming frontend and export profile

The primary development visualizer should run locally against an unpacked run
directory.  It must not require the full dataset to be embedded into one HTML
file or squeezed into a 256 MB ZIP.  Serve a small manifest first, then request
hash-addressed or paged sidecars for the selected view, spatial cell, time
window, carrier range, atom range, or receipt drawer.

The local server must:

- bind to loopback by default;
- serve only files contained under the explicitly selected visualization/run
  roots;
- reject traversal, symlinks escaping those roots, dotfiles, credentials,
  keys, and unrelated workspace content;
- expose a finite census, chunk catalogue, hashes, byte counts, media types,
  and pagination metadata;
- support HTTP range or explicit page/chunk requests for large data;
- leave the physical run directory read-only; and
- store any demo/UI state outside the scientific evidence tree.

Keep a separate **Lovable export** action.  It should produce a bounded web
handoff containing the schema, view contracts, summaries, receipts, exact
carrier prototype, a representative multiresolution sample, and a manifest of
omitted/externally hosted chunks.  Exporting a sample must never change the
reported full census.  Include hashes and enough instructions for a remote
visualizer agent to attach additional chunks later.

## Full observer-to-universe demo composition

Provide a guided demo chapter in which one virtual observer follows the whole
proposed path:

```text
local port state
  -> light-like readback pulses over icosahedral edges and seams
  -> mismatch settling and fixed-point record
  -> A5 sector/current handoff
  -> forced Standard-Model particle catalogue and interactions
  -> stress/energy handoff and forced gravity response
  -> atoms and larger structures
  -> observer-local camera
  -> cosmological overview and CMB comparison
```

Every rung after the last physically replayed receipt retains the
`DEMO_ASSUMPTION` watermark.  Call the moving carrier signals `readback
pulses` unless a physical photon receipt exists.  Standard-Model names,
charges, colors, interaction vertices, atom species, and gravity labels may be
canonical reference content, but their simulated occurrence is a synthetic
demo until the corresponding source, Q2, physical-identification, pole, clock,
and gravity receipts pass.

The demo should include:

- a deterministic pulse schedule on every finite carrier in the selected
  run, with each pulse trace addressable by carrier, port, seam, and tick;
- a convergence/fixed-point panel showing mismatch before and after every
  accepted repair rather than merely fading the animation to calm;
- canonical quark, lepton, gauge-boson, and scalar actor classes, explicit
  interaction-vertex labels, conserved quantities, and source provenance;
- a virtual detector/observer view containing only events inside that
  observer's visible support and causal/readback ancestry;
- separate particle trajectories, stress/energy proxy, assumed metric/gravity
  response, atomic bindings, larger structures, and cosmological layers; and
- synchronized back-links from an observed particle or atom to the screen
  carrier pulses and forced ladder nodes used to display it.

Each view and each story chapter must begin with a two-to-four sentence
summary answering: what is shown, which data or assumptions supply it, what is
physically established, and what remains a demo gap.

### Complete finite census, progressive rendering

`Every carrier` and `every atom` means every member of the explicitly finite
simulated census is stable-ID-addressable, queryable, selectable, and
renderable.  It does not mean drawing all entities in one frame, duplicating
the carrier mesh millions of times in memory, or claiming to enumerate every
atom in the physical universe.

Use GPU instancing where available, spatial/time chunking, frustum and
occlusion culling, aggregate far-field glyphs, and progressive level of
detail.  The UI must always show both the full census and the currently loaded
and visible counts.  Zooming or selecting a census range must resolve down to
the exact carrier/atom records without silently replacing them with random
decorations.  Procedural demo entities require a frozen generator version,
seed, index-to-record function, and chunk hashes so any entity can be
reconstructed deterministically.

## Visual language

Use a restrained dark scientific-instrument aesthetic:

- background: near-black navy `#07111F`;
- computed structural geometry: cool silver `#B9C6D8`;
- passed/replayed evidence: teal `#42D6C7`;
- open dependency: amber `#F5C66B`;
- demo assumption: gold `#F2B84B` with diagonal stripe texture;
- failed evidence: coral `#FF6B6B`;
- unresolved: violet `#B39DFF`;
- selected observer: lavender `#C4B5FD`;
- H3/dS4 geometry: blue `#60A5FA`;
- defect/worldline: magenta `#F472B6`; and
- hidden/protected data: desaturated red-violet `#9F5F80`.

Never rely on color alone.  Pair every status with shape, texture, icon, and
text.  Keep labels legible over animation and support reduced motion.

## Performance requirements

- Stream only the selected carrier neighborhood and current ladder detail.
- Instance repeated carrier geometry; do not duplicate the 12/30/20 mesh per
  carrier in memory.
- Render distant federation carriers as low-detail glyphs.
- Keep all 60 A5 permutations as compact integer arrays.
- Virtualize receipt/stage tables.
- Load screen and evolution sidecars on demand.
- Do not reconstruct the entire visualizer pack for one view.
- Pause rendering when the tab or panel is hidden.
- Keep every finite carrier and atom addressable through the census even when
  most are culled or represented by aggregate far-field glyphs.
- Display full-census, loaded-chunk, and currently visible entity counts
  separately.

## Acceptance tests

The visualization is acceptable only if all of the following hold:

- the local carrier shows exactly 12 ports, 30 edges, 20 faces, and six
  antipodal pairs;
- all 60 A5 actions preserve the rendered incidence;
- `3` and `3-prime` are visually distinct;
- a carrier is never identified with one S2 support point;
- seams and external boundaries are separately visible;
- screen, repair, record, observer, A5 current, and SM stages are navigable end
  to end;
- the Q2 alternative branch is topologically correct;
- physical, demo, failed, unresolved, open, and not-applicable statuses are
  distinguishable without color;
- force-all completes only the display path;
- toggling a demo stage does not mutate any physical receipt or campaign gate;
- target freezes are labelled post-exposure/display-only;
- the three clocks remain distinct;
- screenshots/videos in demo mode always retain the watermark;
- objective and subjective cameras are not conflated;
- H3 uses the declared hyperboloid contract;
- CMB resemblance is not presented as a prediction; and
- every visible interpretation has a receipt, a structural-data source, or an
  explicit demo-assumption source;
- every view opens with an explanatory summary;
- the local frontend pages data without a monolithic ZIP or embedded payload;
- the Lovable export reports its sampling/omission manifest honestly;
- every carrier and atom in the finite demo census has a stable reconstructible
  ID; and
- selecting an observed particle or atom can trace back to the exact display
  assumptions and carrier/readback records that supplied it.

## Non-negotiable boundary

A beautiful complete animation is a product and integration success.  It is
not a scientific emergence receipt.  The app should make the proposed ladder
easy to understand and debug while making it impossible to mistake a frozen
gap for a derived theorem or physical measurement.
