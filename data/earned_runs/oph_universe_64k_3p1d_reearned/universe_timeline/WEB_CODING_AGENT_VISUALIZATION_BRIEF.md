# Web Coding Agent Visualization Brief

Build from `visualization_payload.json` or the chunked `oph_visualizer_pack_v2.tar.zst`:

```text
runs/oph_universe_64k_3p1d_reearned/universe_timeline/visualization_payload.json
```

Core product goal:

Create an interactive OPH visualization of observer-like self-reading systems. The differentiator must remain explicit: OPH is not generic particles in a box. It is bounded patches with local state, ports/boundaries, readback, records, feedback/repair moves, and public receipts.

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
3. **Shared records and consensus.** Keep the patch network visible while equal readout hashes and
   record/object packets synchronize across overlaps. Let repeated packets coalesce into stable
   teal consensus glyphs. Caption: "Objectivity is the agreement carried by shared records." This
   visual transition must not imply that a neutral bulk was present before consensus.
4. **Enter one observer's 3+1D view.** Select a patch, retain a small labelled overview inset, and
   move the main camera through its boundary into `subjectiveObserverCameras[*]`. Reveal only its
   visible records, object packets, nominal-FOV worldline sightings, and modular time. If
   `assumedDs4Spacetime.enabled`, `assumed_ds4_visualization_layer_receipt`, and
   `SIMULATION_ASSUMED_VISUAL_UNIVERSE_RECEIPT` pass, grow the declared open-H3 dS4 grid around that
   camera with a persistent **ASSUMED VISUAL LAYER — NOT DERIVED** badge.
5. **Populate the observer-facing H3 bulk.** Match consensus glyphs to
   `consensusBulk.objects` and reveal them along intrinsic H3 geodesics. Keep a faint provenance
   tether back to the contributing patch records during the transition. Caption the space
   "observer-facing H3 consensus chart," not a strict neutral third-person bulk unless its receipt
   passes.
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


Paper-accuracy requirement:

- Read `paperAccuracy.checks` and render blocked promotions as closed gates, not errors.
- Do not promote visual similarity, apparent attraction, CMB-shape resemblance, or finite lattice gaps beyond the exact receipts in `paperAccuracy.receipts`.
- Treat `EINSTEIN_BRANCH_ENTRY_RECEIPT` / `OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1` as the mandatory gate for any production-gravity wording. If false, show the E0 manifest's missing sidecar receipts as a closed promotion gate, not a simulation error.

Required views:

1. **Fluctuating quantum vacuum / finite screen view**
   - Render `screen.points` as an S2/equirectangular or sphere view.
   - Color by `screen.values` and label the field with `screen.fieldName`.
   - Add a field selector over `screen.fields` (every other non-constant freezeout field at the same points).
   - When `screen.evolution.available=true`, animate the vacuum: `screen.evolution.fields[<name>].frames[i][j]`
     colors `screen.points[j]` at `screen.evolution.cycles[i]`. Bind the animation to the global timeline
     slider. Full-resolution frames are in the `screen_frames_<field>_<rows>x<frames>.bin` sidecars
     (float32-le, frame-major, same row order as `screen_full_<rows>.bin`).
   - Overlay `screen.clusters.snapshots[*].clusters` as repair/holonomy residues.
   - Use `visualizationViews.fluctuatingQuantumVacuum` for the canonical layer list and claim boundary.
   - If present, render `visualizationViews.fluctuatingQuantumVacuum.yangMillsGapCertificate` as finite SU(2) gauge diagnostics: plaquette trace, Wilson/Polyakov traces, refinement gap rows, and promotion blockers.
   - Explain that this is the observer boundary/readback surface, not a literal QFT vacuum or a pre-existing 3D bulk.
   - Never label finite SU(2) diagnostics as a reproduced Yang-Mills mass gap unless `YANG_MILLS_GAP_REPRODUCED_RECEIPT` is true.

2. **Overlap repair view**
   - Check `smallUniverse.contentAvailable` before rendering. If false, show a receipt-only state with `receiptSource` and `contentBlockers`; do not draw or imply an exact mini-universe graph/path.
   - When content is available, render `smallUniverse.nodes`, `smallUniverse.edges`, and `smallUniverse.repairFrames`.
   - Animate repair frames with a slider.
   - Highlight `frame.parent -> frame.node`, show `phi`, `deltaPhi`, and strict descent.
   - Show exact zero-holonomy branch beside the frustrated control nonzero-holonomy count.

3. **P/N silence-to-observation view**
   - Render `pnSilenceToObservation.closureCoordinates`.
   - Show the sequence: `silenceInitialState` -> P detuning -> `finiteRegulatorDepth` -> `observationEmergence`.
   - Gate `scale_compressed_pn_silence_to_observation_receipt`, `literal_global_N_capacity_simulated_receipt`, and `dynamic_p_detuning_control_receipt` separately.
   - Explain that this is a finite scale-compressed witness of the OPH thesis, not literal global `N_CRC` cells.

4. **Observer camera / modular-time view**
   - Render `observerModularTime.observers` on the screen by `axis`.
   - Render `observerModularTime.overlapLinks` as the observer-overlap substrate. These are not decorative graph edges; they are the local shared-support relations from which objectivity is read.
   - Animate overlap repair: the strongest links carry `repairTrajectory`. When
     `overlapSummary.overlapRepairDynamics.available=true` those trajectories are measured per-cycle
     mismatch intensity on the pair's shared support (`overlapMismatchDensity` over `cycle`), so pulse
     or color each link by its own trajectory as the timeline plays; links without a trajectory derive
     one from `timeFrames` scaled by `overlapCommittedFraction`/`overlapRepairLoadMean` per `trajectorySource`.
   - Animate `observerModularTime.timeFrames`.
   - Add an observer selector backed by `observerModularTime.objectiveObserverViews`.
   - For the selected observer, animate `objectiveObserverViews[*].timeFrames`: show `relativeTime`, `localTransitionStep`, `dominantRecordSignature`, `dominantObjectPacket`, `visibleReadoutHash`, support size, and packet histograms.
   - Use `subjectiveObserverCameras` for subjective/first-person camera rendering. Each camera is derived from one observer-local visible readout and includes `eye`, `lookAt`, `up`, `right`, `forward`, `fovDegrees`, support samples, and time frames.
   - Treat every H3 vec3 as hyperboloid spatial components. Use `coordinateSystems.h3_hyperboloid_spatial_components_v1`, intrinsic geodesics, and each camera's `h3TangentFrame`; never reinterpret points as Poincare-ball or Euclidean coordinates.
   - Render `visibleProtoWorldlines` inside the nominal FOV. Resolve `peripheralDiagnosticProtoWorldlineIds` on demand for a separate optional orientation overlay and never duplicate them into visible counts.
   - Render each camera's `visibleConsensusObjects` as the static object field of the first-person view:
     place at `(observerLocalReadout.u, observerLocalReadout.v)`, size by `observerCount`, depth-cue by
     `observerLocalReadout.range`, and dim entries with `outsideNominalFov=true`.
   - Use `visualizationViews.observerCamera` for the canonical layer list and claim boundary.
   - Do not present this as external global time. It is observer-local modular readout.

5. **Effective string-theory edge/worldsheet view**
   - Use `visualizationViews.effectiveStringTheory` for the canonical layer list and claim boundary.
   - Honor `effectiveStringTheory.layerAvailability` and `hiddenLayers`. When `finite_edge_string_vibration_pulses=false`, hide that sublayer while retaining populated H3/worldline layers.
   - Render `smallUniverse.cycles` and `smallUniverse.repairFrames` only when their corresponding availability entries pass, as cyclic edge normal forms and swept repair histories.
   - When `finite_edge_string_vibration_pulses=true`, render `effectiveStringTheory.finiteEdgeStringVibrationSamples` as the exact finite edge-pulse/vibration layer. Animate `frameStep`, `loopPhase`, and `normalizedAmplitude`; never replace it with generic sine-wave string oscillations.
   - Render `screen.clusters.snapshots[*].clusters` as collar/defect fluctuation markers.
   - Render `consensusBulk.objects` as a 3D scatter/cloud.
   - Size by `observerCount`; color by `h3CompactnessNormalized`.
   - Render `consensusBulk.protoParticleCandidates.worldlines[*].events` as H3 tracks.
   - Label the selected track source with `consensusBulk.protoParticleCandidates.worldlineSource`.
   - Treat `proto_particle_worldlines.csv` as a legacy fallback only; do not let stale sidecars
     override organic or free dynamics JSON reports.
   - Use neutral wording: "edge-worldline diagnostic", "consensus object packet", and "holonomy/proto-particle candidate" unless the stronger receipts pass.
   - Do not label it a critical string CFT unless a future critical-edge receipt is true.
   - Gate labels must show observer-facing H3 consensus bulk and chart-blind neutral quotient bulk separately.

6. **Emergent curved-spacetime proxy view**
   - Use `visualizationViews.emergentCurvedSpacetime` for the canonical layer list and claim boundary.
   - Use `emergentCurvedSpacetime.continuousBulkField.volumeSamples` for the main continuous
     bulk-field rendering when it is available. Render it as fog, density points, an isosurface, or
     a warped volume, not just as isolated source balls.
   - Use `emergentCurvedSpacetime.continuousBulkField.sliceSamples` and `temporalSliceSamples` for
     warped grid slices and animated field slices.
   - Render `emergentCurvedSpacetime.curvatureProxyPoints` as stress-source glyphs in the observer-facing H3 chart.
   - Size glyphs by `sourceDensity`; drive grid bend, contour strength, or surface displacement by `curvaturePotential`.
   - Drive local spatial contraction by `compactificationFactor`; use `emergentSpatialScaleFactor` as the local grid/cell-size multiplier.
   - Show `sourceMath.sourceDefinition` and `gravitySourceInterpretation` so users see that the source is quotient-visible OPH stress/readout, not raw rest mass.
   - Animate `emergentCurvedSpacetime.timeSlices` and proto-worldline event cycles when available.
   - Display `einstein_branch_entry_receipt`, `production_gravity_receipt`, `physical_gravity_prediction`, and `einstein_equation_solution_receipt` separately.
   - Use `emergentCurvedSpacetime.einsteinBranchEntry` and `visualizationViews.emergentCurvedSpacetime.einsteinBranchEntry` for the E0 manifest, provenance tags, receipt rows, blockers, and claim boundary.
   - Never label this view as physical gravity, a solved metric, or a matter stress tensor unless the Einstein branch-entry and gravity receipts are true.
   - Use `assumedDs4Spacetime` for the complete 3+1D narrative scene when its simulation-assumption receipt is true. Render the declared open-slicing metric, scale factor, comoving observer frames, and assumed defect-matter links, while displaying that the derived dS4, Einstein-equation, and physical-particle receipts remain false.

7. **CMB diagnostics view**
   - Plot `cmbComparison.residualRows`.
   - If residual rows are absent and `cmbComparison.assumedVisualization.dataAvailable=true`, plot
     `referenceRows` and `assumedModelRows` only after checking
     `provenance.sha256Matches=true`. Label the second series "published best-fit reference (not OPH)"
     and keep the gold assumption badge visible.
   - An optional S2 realization may use `skyRealizationContract.seed` and `sourceSpectrum`; it must be
     labelled deterministic explanatory rendering, not simulated OPH CMB output.
   - Use `comparableObservations.measurementLanes` and `comparableObservations.datasets` for additional public-data-facing diagnostics such as galaxy, CNB, H0/S8, anomaly, repair-clock, object-population, and neutral-frontier lanes when present.
   - Display `USABLE_PHYSICAL_CMB_DATA_RECEIPT` and `PHYSICAL_CMB_PREDICTION_RECEIPT` separately.
   - Never label diagnostic TT comparison as a physical prediction unless the receipt is true.

Visual language:

- Favor direct geometry: finite S2 screen, repair graph, H3 scatter, receipt gates.
- Avoid decorative sci-fi metaphors. Use the OPH explanation text in the payload.
- The first visible text should state that OPH tech instantiates observer-like self-reading systems.
- Every gate badge must be data-driven from `receipts`; no hard-coded success labels.

Receipt boundary to preserve verbatim:

Visualization/readout bundle for OPH observer-like self-reading systems. It shows a finite observer screen, overlap repair, observer-local modular-time readouts, theorem-assisted H3 consensus object charts, and measurement-comparable CMB diagnostics when present. It does not promote chart-blind strict neutral quotient bulk or physical CMB prediction unless those receipts are true in the payload.
