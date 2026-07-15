# OPH Universe Visualization Instructions

Open the standalone viewer:

```bash
open runs/oph_universe_64k_3p1d_reearned/universe_timeline/oph_universe_timeline_viewer.html
```


Data payload for custom viewers:

```bash
runs/oph_universe_64k_3p1d_reearned/universe_timeline/visualization_payload.json
```

## Canonical pedagogical cinematic storyboard

Build a guided **Story mode** and an immediately available **Explore mode**. Story mode is a
progressive reveal of one idea, not a montage of unrelated panels. Keep a compact chapter rail so
the viewer can pause, scrub, replay, or jump without losing the active observer or time selection.

1. **Bounded self-reading patches.** Begin close on one finite patch against a quiet dark field.
   Reveal its boundary/ports, local state, readback pulse, record write, and feedback arrow; then pull
   back to show many such patches on the finite S2 screen. Caption: "Observers begin as bounded
   systems that read and update local records." No H3 bulk exists in this shot.
2. **Overlap repair.** Illuminate shared boundaries in amber. Animate a mismatch, the selected local
   repair move, and decreasing `phi`; repaired relations cool to teal. Use the exact mini-universe
   path only when `smallUniverse.contentAvailable=true`; otherwise use the separately labelled
   large-run `screen.repairTrace` and show the theorem receipt card, never a fabricated exact path.
3. **Shared-record gauge views.** Keep the patch network visible while sampled observers read one
   committed shared record through different local gauge frames. Render recovered re-gauging maps,
   Cech-closed triples, and shuffled controls only from their exported rows. Caption: "Different gauge
   views of one shared record agree." Independent commit histories require a separate experiment.
4. **Enter one observer's 3+1D view.** Select a patch, retain a small labelled overview inset, and
   move the main camera through its boundary into `subjectiveObserverCameras[*]`. Reveal only its
   visible records, object packets, nominal-FOV worldline sightings, and modular time. If
   `assumedDs4Spacetime.enabled`, `assumed_ds4_visualization_layer_receipt`, and
   `SIMULATION_ASSUMED_VISUAL_UNIVERSE_RECEIPT` pass, grow the declared open-H3 dS4 grid around that
   camera with a persistent **ASSUMED VISUAL LAYER — NOT DERIVED** badge.
5. **Show the observer-facing H3 layers.** Reveal `consensusBulk.objects` along intrinsic H3 geodesics
   only with their exact data and status. Keep a faint provenance tether to contributing patch
   records. Label a computed H3 chart diagnostic, a receipted record population, and any assumed
   scene completion separately. Do not call the scene a strict neutral third-person bulk unless that
   receipt passes.
6. **Defect worldlines styled as matter.** Draw exported holonomy/defect worldlines as luminous
   magenta candidate tracks with distinct birth/contact markers. When
   `assumedDs4Spacetime.defectMatterRendering.enabled` and
   `assumed_topological_defects_render_as_matter_receipt` pass, matter-like glow, trails, and
   silhouettes are permitted only under a persistent
   **ASSUMED DEFECT-AS-MATTER STYLING** badge. Keep "proto-particle/defect candidate" wording while
   `particle_matter_receipt` is false; visual matter styling never promotes that receipt.
7. **CMB-shaped sky and comparison.** Widen the selected observer camera into its finite readback
   sky, using exported screen/CMB data only, and label it "diagnostic CMB-shaped sky." Prefer
   `cmbComparison.residualRows` when a genuine run comparison exists. Otherwise, when
   `cmbComparison.assumedVisualization.dataAvailable=true`, render its SHA-256-pinned
   `referenceRows` and `assumedModelRows` with a persistent **ASSUMED VISUAL LAYER — NOT DERIVED**
   badge. The latter is the source table's published best-fit reference, never an OPH model. An
   optional deterministic sky may use only `skyRealizationContract.seed`; it remains explanatory.
   Show usable-data and physical-prediction gates separately; resemblance to the observed CMB is not
   a prediction receipt.

## Camera and transition grammar

- An outside camera is an **explanatory overview (not observer-visible)**. It may show patch topology,
  overlap links, and provenance, but it must never masquerade as what an observer sees.
- A first-person camera may use only that observer's exported local readout and tangent-frame
  projection. Never expose hidden global H3 positions in the subjective view. Peripheral diagnostic
  IDs belong to a separately styled orientation overlay and are not nominal-FOV sightings.
- Use a matched transition from a selected overview patch to its first-person camera: highlight the
  patch, shrink the overview to a labelled inset, then reveal local records before H3 objects. Reverse
  the same mapping when returning to overview so the relation remains teachable.
- Interpolate H3 motion with the declared hyperboloid geometry, never Euclidean lerp or guessed
  Poincare coordinates. Camera easing is presentation only; do not interpolate absent measurements
  or invent intermediate repair states.
- Prefer calm 0.6-1.2 second focus transitions and short holds for captions. Avoid flashing, rapid
  parallax, and constant camera drift; honor `prefers-reduced-motion` with cuts/fades and a static
  chapter sequence.

## Cohesive visual language and status grammar

Aim for a precise scientific observatory: deep navy `#07111F` background, warm-white `#E8F1FA`
text, cyan `#38BDF8` patch boundaries/readback, amber `#F59E0B` mismatch/repair, teal `#2DD4BF`
shared consensus, lavender `#C4B5FD` selected-observer framing, blue `#60A5FA` H3/dS4 geometry,
magenta `#F472B6` defect candidates, and gold `#FDE68A` assumed layers. Use a colorblind-safe
blue-to-orange diverging scale for signed CMB residuals. Do not use color alone: repeat every class
with a shape, line pattern, icon, and text label, and maintain WCAG AA text contrast.

Use one stable badge grammar everywhere, including captions, legends, drawers, and exported images:

- **COMPUTED / PASSED** — green check-circle, only from the corresponding true receipt.
- **DIAGNOSTIC DATA** — cyan diamond, exported/computed visual data without the physical promotion.
- **ASSUMED VISUAL LAYER** — gold dashed hexagon, sourced only from the assumption manifest; it can
  never satisfy a computed or physical receipt.
- **CLOSED PROMOTION GATE** — amber outlined lock, false receipt but not a runtime error.
- **UNAVAILABLE IN THIS EXPORT** — gray slashed circle, missing/empty data; never silently replace it.

Reserve red for actual load/validation/falsification errors. Every chapter gets a one-sentence
"what you see" caption, a one-sentence epistemic-status caption, a compact legend, and the relevant
gate badges. Keep these visible long enough to read and include them in screenshots.

## Interaction, accessibility, performance, and graceful availability

- Provide Play/Pause, previous/next chapter, scrubber, speed, observer selector, overview/first-person
  toggle, layer toggles, captions, legend, and receipt/provenance drawers. All controls require visible
  focus, keyboard operation, ARIA labels, and a text/table alternative for plots and animated states.
- Preserve the active observer, time, and layer selection when changing chapters. Never autoplay
  audio. Pause animation when offscreen, and offer reduced-motion and high-contrast modes.
- Treat `oph_visualizer_pack_v2.tar.zst` and its manifest as a streaming scene source. Stay below the
  256,000,000-byte hard package ceiling; lazy-load only the current/next chapter, parse large chunks
  in a worker, use typed arrays/instancing/LOD/frustum culling, cap device pixel ratio, and release
  prior chapter GPU resources. Do not reconstruct the full payload in the browser merely to draw
  one observer. Target smooth 60 fps desktop and a stable 30 fps reduced/low-power mode.
- Resolve layer availability before Story mode starts. An unavailable chapter remains in the rail as
  an honest status card, then Story mode advances without a blank stage. Never synthesize geometry,
  worldlines, exact repair frames, CMB samples, or receipts.
- If subjective cameras are absent, remain in the labelled overview and show the modular-time table.
  If H3 objects are absent, stop at consensus records. If worldlines are absent, keep the populated
  H3 layer without tracks. If run CMB rows are absent but the pinned assumed reference is available,
  show that reference under its gold assumption badge; if both are absent, end on the finite readback
  sky plus its unavailable comparison gate. Independent available layers must remain usable when a
  sibling layer is empty.


What to inspect:

- Panel 1 shows the fluctuating-vacuum diagnostic view: the finite S2 observer screen/boundary readback from the larger observer-flow run. Colors are screen readback fields; rings are screen-local defect/holonomy residues. This is a diagnostic OPH readback field, not a literal QFT vacuum unless a future receipt says so.
- The same view may include `visualizationViews.fluctuatingQuantumVacuum.yangMillsGapCertificate`: finite SU(2) Wilson-lattice plaquette/Wilson/Polyakov traces and transfer-gap proxies. Show its finite diagnostic receipt separately from `YANG_MILLS_GAP_REPRODUCED_RECEIPT`, which should remain closed unless a future continuum certificate promotes it.
- Panel 2 is receipt-only for this export: the finite-consensus theorem receipt and its source remain visible, but no exact mini-universe graph/path may be drawn. Use the separately labelled large-run `screen.repairTrace` for repair animation.
- Panel 3 shows the observer-camera view and observer-local modular time. Each dot is an observer-like self-reading row with local support, records, readback hash, and modular-depth readout. Use the observer selector to inspect one observer's objective readout across its modular-time frames: record packet, object packet, transition step, local packet histograms, and the global trace cycle used only for synchronization.
- The payload also exports `subjectiveObserverCameras`: first-person rendering cameras derived from visible observer-local readouts. These are the right inputs for a subjective observer camera map.
- Every H3 vec3 uses `h3_hyperboloid_spatial_components_v1`: lift it to `(sqrt(1+|x|^2),x)`, use intrinsic distance/geodesics, and project through the observer logarithm map. Never guess Poincare coordinates from the numeric range.
- `visibleProtoWorldlines` contains only nominal-FOV hits. Compact `peripheralDiagnosticProtoWorldlineIds` drive a separate on-demand full-sky diagnostic lane and must not be counted as directly visible.
- `assumedDs4Spacetime` completes the narrative renderer with an explicitly assumed open-H3 dS4 background, observer frames/tetrads, and defect-as-matter styling sourced from `simulation_assumption_manifest.json`. Its physical receipts remain false.
- `simulationAssumptions` is the authoritative thirteen-row assumption ledger. Keep its assumption-completeness receipt separate from `visualizationRenderData.visualUniverseCompleteness` data completeness and render readiness; none is a theorem or physical receipt.
- Panel 4 shows the emergent curved-spacetime proxy view. It uses `emergentCurvedSpacetime.sourceMath`, `curvatureProxyPoints`, `continuousBulkField`, and `timeSlices` to render quotient-visible source density, H3 Green potential, curvature, compactification, continuous volume samples, and warped slices over the observer-facing H3 chart. `dataRefs` names compatibility aliases without duplicating arrays. It is a diagnostic warped-grid/field layer, not production gravity or a physical metric unless `einstein_branch_entry_receipt`, `production_gravity_receipt`, and related promotion receipts are true.
- Panel 5 shows the effective string-theory diagnostic view. Record-object packets group shared readback events from overlapping observers. Magenta/red tracks are holonomy/defect worldlines fitted into the same H3 chart: proto-particle candidates and edge-worldline/collar diagnostics, not matter particles or a critical worldsheet unless the corresponding receipts pass.
- The finite-edge vibration sublayer is unavailable and must be hidden. Keep the other populated H3/worldline string-view layers; do not synthesize replacement string oscillations.
- Panel 6 places the measured record-silence/readout transition beside a postprocessed P/N branch overlay. The relaxation dynamics did not consume P, and the view must not imply P caused the transition while dynamic detuning controls are false.
- Panel 7 shows usable run CMB diagnostics when present. If those rows are absent, it may instead show the SHA-256-pinned `cmbComparison.assumedVisualization` reference and published-best-fit shape under a persistent assumption badge; `assumedModelRows` is not an OPH model. `comparableObservations` also carries compact measurement-lane summaries. None of this is a physical prediction unless the relevant prediction receipt passes.
- `visualizationViews.fluctuatingQuantumVacuum`, `visualizationViews.observerCamera`, `visualizationViews.emergentCurvedSpacetime`, and `visualizationViews.effectiveStringTheory` are the canonical view contracts for a custom visualizer.
- `paperAccuracy` is the fail-closed paper-accuracy guard. Use it to render which claims are allowed, which promotions remain blocked, and why visual similarity alone is not a receipt.

Claim boundary:

Visualization/readout bundle for OPH observer-like self-reading systems. It shows a finite observer screen, overlap repair, observer-local modular-time readouts, controlled H3 chart diagnostics, record-object layers, and measurement-comparable CMB diagnostics when present. Computed H3 receipts, assumed scene completion, strict neutral bulk, and physical CMB promotion remain separately labeled.
