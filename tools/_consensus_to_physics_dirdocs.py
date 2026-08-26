# -*- coding: utf-8 -*-
"""DISPLAY_INSTRUCTIONS.md written into each data/ subdirectory of the handoff."""

DIR_DOCS = {

"screen": """# `screen/` — the observer's sky

**Panels:** V12b (the observer's sky), V31 (the frozen sky and its spectrum), V05, V15.

## What is here

`screen_frames_<field>_65536x48.bin` — flat little-endian **float32**, 48 frames of 65,536
patches, frame-major. Read frame *k* as bytes `[k*65536*4, (k+1)*65536*4)`. Four fields ship:
`record_port_entropy`, `local_mismatch_density`, `modular_depth`, `cumulative_repair_load`.

`screen_points.csv` — the sky direction of every patch. **Join by row order**: row *i* is index
*i* in every binary. `screen_full_65536.bin` is the final committed state.

Cycle numbers for the 48 frames live in `../run/harmonic_time_trace.npz`, key `cycles`.

Every frame is **standardised per frame** (mean 0, standard deviation 1). They carry *pattern*,
not amplitude. Amplitude is in `../run/mismatch_trace.csv`.

## How to make physics out of it

This is a **celestial sphere**: the set of directions an observer can look. Render it as a real
sphere the reader is inside — planetarium, not scatter plot. Dim the far hemisphere; never cull
it, because an observer's sky has no back.

**Lead with `local_mismatch_density` and animate cycles 0 → 127.** It begins as dense fog over the
whole sky and ends at exactly zero spatial variance. That is consensus happening, seen from
inside, and it is the most legible emergence moment in the run. Stage it: hold on the final still
frame for a beat.

Then offer `record_port_entropy`. Its mottled end state is the **CMB analogue** — the same map
whose angular spectrum V31 plots. Use a diverging colour scale centred at zero, since the fields
are standardised.

`modular_depth` is the observer's own clock reading across the sky; `cumulative_repair_load` is
how much work each direction has absorbed. Both reward a slow orbit.

Pair the sphere with a linked amplitude sparkline from `mismatch_trace.csv` so the reader can see
that the pattern flattening and the disagreement counter hitting zero are the same event.
""",

"timeline": """# `timeline/` — the visualization export

**Panels:** V12b, V24, V24b, V26, V28, V29, V29b, V31, V12, V27.

## What is here

Purpose-built sidecars emitted by the run for exactly this job. The important ones:

- `subjective_observer_camera_frames.csv` — what each camera *sees*, per frame:
  `polar_field_readout_json`, `visible_object_packets_json`, `visible_record_packets_json`,
  `dominant_record_signature`, `local_transition_step`. This is the observer's first-person view.
- `observer_proto_worldline_sightings_sample.csv` — sightings of defect worldlines in
  **observer-local coordinates**: `observer_local_u`, `observer_local_v`, `observer_local_range`,
  `observer_local_angular_separation_degrees`, `visibility_score`, `outside_nominal_fov`.
  Evenly-spaced sample of 196,608 rows.
- `proto_particle_worldlines.csv` — 128 worldlines with birth/death cycle, `h3_path_length`,
  `class_mode`, and five gate columns including `particle_like`.
- `screen_cluster_tracks.csv` — the same objects in god-view screen coordinates.
- `emergent_curved_spacetime.json` — a complete, self-describing view: `renderLayers` (7),
  `animationChannels` (3), `curvatureProxyPoints` (1,631), `timeSlices` (12), `continuousBulkField`,
  plus its own `receipts`, `nonClaims` and `claimBoundary`.
- `emergent_curved_spacetime_curvature_proxy.csv` — per-source `mass_proxy`,
  `stress_energy_proxy`, `h3_green_potential`, `curvature_potential`,
  `local_metric_conformal_factor`, `emergent_spatial_scale_factor`.
- `cmb_screen_spectrum_rows.csv`, `cmb_residual_rows.csv` — C_ℓ and residuals.
- `yang_mills_su2_*.csv` — plaquette, Wilson and Polyakov traces.
- `cameras_full_128.json`, `observers_full_128.json`, `observer_anatomy.json`,
  `consensus_h3_objects.csv`.

## How to make physics out of it

**Gravity (V29b) is the headline.** `emergent_curved_spacetime.json` ships its own render layers —
use them rather than inventing an encoding. The physics to draw: matter sources produce an H3
Green potential, and the resulting `local_metric_conformal_factor` makes the bulk **contract**
around them. Render a regular lattice of the H3 ball whose cells visibly shrink and crowd toward
mass. **Do not draw a rubber sheet with a dent** — that picture adds a fake extra dimension and
teaches the wrong intuition. Rulers get shorter near mass; that is curvature. Animate the 12 time
slices and plot `totalCurvaturePotential` responding as sources move. Caption it with the Einstein
equation: matter tells the bulk how to contract.

**First-person matter (V24b).** Plot sightings at (`observer_local_u`, `observer_local_v`) inside
a real field-of-view frame, sized by `visibility_score`, shaded by range bucket. Put the god view
beside it from `screen_cluster_tracks.csv` and link them on hover. A telescope view next to a star
chart: the same object, from inside the universe and from outside it. This is how a topological
defect becomes a proto-particle to an observer.

**Relativity (V28).** Put the measured objects in a Poincaré ball and let a boost **aberrate** the
surrounding celestial sphere — stars crowding toward the direction of motion. The modular flow
acts on the direction sphere by exactly the Möbius maps that generate aberration, so this image is
earned by the structure even where the receipt is open.

**Cosmology (V31).** `cmb_screen_spectrum_rows.csv` against the frozen sky gives the Planck
all-sky-plus-C_ℓ pairing. Draw the shuffled control faintly beneath the spectrum.
""",

"run": """# `run/` — reports and traces from the source run

**Panels:** most of them. See `../DATA_INDEX.md` for the exact per-panel map.

## What is here

- `mismatch_trace.csv` — **the single most useful file in the bundle.** One row per cycle:
  `phi` (total disagreement), `record_packet_entropy`, `beta`, `committed_fraction`,
  `modular_depth_mean`, `total_repair_actions`. Thermodynamics lives here.
- `central_record_born_report.json` — a legacy-named classical categorical-partition diagnostic
  on all 65,536 records: `probability_sum` 1.0, zero indicator-idempotence error,
  `record_partition_filters_commute` true, `partition_filter_idempotent` true, and
  `repeat_read_stability_fraction` 1.0, plus `sample_events` to animate. No ambient algebra is
  supplied, so centrality and a Lüders instrument are not tested. The displayed probabilities
  are empirical frequencies normalized from these same counts; this is not a Born-law test,
  quantum instrument, or physical measurement receipt.
- `observer_modular_experience_report.json`, `observer_perspective_rows.csv`,
  `observer_consensus_report.json` — the observer population and its clocks.
- `defect_*` reports, `array_holonomy_report.json`, `s3_class_counts.json` — matter.
- `cl_comparison_report.json`, `freezeout_map_summary.json`, `harmonic_time_trace.npz` — cosmology.
- `yang_mills_gap_certificate_report.json`, `einstein_bridge_manifest.json`,
  `neutral_3d_bulk_audit_report.json` — the gate ladders.
- `finite_repair_transition_matrix.npz` + report — the decoherence spectrum.

## How to make physics out of it

**Thermodynamics (V13, V14).** Plot `phi` falling and `record_packet_entropy` rising on one time
axis: an H-theorem and the second law, same process seen twice. Draw `ln N` = 11.090355 as a lid
the entropy presses against and settles under at 11.084308. Then difference the entropy series
and switch to the 16k control in `../control/` — three cycles run backwards there and none do at
64k. That contrast *is* the fluctuation theorem: the law bounds an average, and fluctuations
shrink like 1/√N.

**Classical record conditioning (V18).** Take one committed event class from `sample_events` and
show the finite partition, then filter the records to that class. Reapplying the same filter changes
nothing. Show the exact partition and idempotence identities as literal numbers beside it, and label
the frequencies as same-sample empirical counts. Do not depict superposition, wave-function
collapse, a quantum instrument, or a Born-law comparison: this file supplies none of them.

**Gate ladders (V26, V30).** Render as staircases climbing toward a familiar equation, rungs lit
where discharged and dark where not. Put the destination — `G_ab + Λg_ab = 8πG⟨T_ab⟩` — at the top,
drawn beautifully and clearly not yet reached.

Read chart-object counts from `observer_chart_object_h3_report.json`, **never** from
`manifest.json`; see the bundle README's data caveat.
""",

"derived": """# `derived/` — exact structure, computed rather than run

**Panels:** V01, V06, V07, V08, V11, V17, V19, V20, V21, V22, V23, V25.

## What is here

- `physics_payload.json` (schema `oph.physics_first_visualizer.v1`) — the carrier, the exact A5
  band decomposition and damping curves, the rank-three limit, the thermodynamic traces for both
  run sizes, a downsampled sky, the Standard Model row table, the defect census, and the Born
  scenarios. Each block is tagged `measured`, `exact`, or `declared`.
- `a5_symmetry.json` — the 60 rotations, the icosahedron, sector dimensions in damping order
  `[1, 3, 5, 3′]`.
- `QM_OBSERVER_VIZ.v1.json` / `QM_OBSERVER_RECEIPT.v1.json` — the eight measurement contexts, the
  branch trees, collapse chains, and the interference scenario. Every rational is a
  `[numerator, denominator]` pair.
- `electromagnetic_response.json` — 12-port Green potential and 30 seam fluxes for a unit dipole,
  plus a temporal bundle.
- `dimension_probe_receipt.json` — the calibrated spectral-dimension instrument.

## How to make physics out of it

**Three dimensions (V07) is the centrepiece and needs no run data at all** — recompute it in the
browser from the carrier in a few lines, so the reader can check it. The icosahedron Laplacian has
spectrum `0(×1), 5−√5(×3), 6(×5), 5+√5(×3)`; repair `T = I − L/60` damps those blocks at
`1, 0.953934, 0.9, 0.879399`. Animate the race: two mode-clouds dim and die, the third brightens,
and twelve scattered points **snap into an exact icosahedron**. Crystal nucleation. The surviving
block is three-dimensional, and its three axes are the coordinate frame every later panel uses.

**One generation of matter (V23), also exact.** Lay the ten exterior rows out as a weight diagram
with hypercharge on an axis, then fade in the names: Q_L, u_R, d_R, L, e_R and conjugates. The
hypercharges `1/6, 2/3, −1/3, −1/2, −1` are forced by `2t + 3d + q ≡ 0 (mod 6)`, the condition for
a representation to descend to `(SU(3)×SU(2)×U(1))/Z6`. Keep them as fractions.

**Finite Born-form scenario (V17).** A beam of 179 tokens hitting a schematic analyser and
splitting can visualize the exact finite algebraic fixture. Show the refinement into
sub-configurations before the split so `716` is not mysterious, and label the panel explicitly as
a generated theorem fixture rather than laboratory data, a source-produced preparation, or an
instrument receipt.

**Spectral dimension (V11).** Lead with the calibration: the estimator recovers 1.0030, 2.0223 and
3.0460 on a ring, a 2-torus and a 3-torus. Then show the tower climbing from 1.95 at κ=0 to about
2.48 — between 2 and 3, not at 3. Draw the gap; it is more interesting than hiding it.
""",

"receipts": """# `receipts/` — committed algebraic receipts

**Panels:** V07, V08, V09, V10.

## What is here

Repository receipts (not run output) recording the exact finite results behind the spatial chain:

- `port_gram_completion_bridge_receipt.json` — the centred response kernel
  `C_n = Q T^(2n) Q`, its exact spectral decomposition, the normalised limit
  `12·C_n/tr(C_n) → 4·P_low`, and `attainment.signed_module_gram_rank_three: true`.
- `port_load_metric_quotient_receipt.json` — `Z^12 → Z^6` by antipodal difference, seam-current
  image `D6 = {z : Σz even}`, Smith invariants `(1,1,1,1,1,2)`, index 2.
- `seam_current_same_metric_scale_receipt.json`, `port_repair_propagation_bridge_receipt.json`.

## How to make physics out of it

These are the algebra behind the pictures, and they are worth surfacing because they let the
reader verify rather than trust.

**V07/V08.** The receipt records something a careless animation gets wrong: at any *finite* step
the kernel has rank **eleven**. Three appears only in the normalised limit. Put a live rank readout
in the panel that reads 11 all the way along and flips to 3 at the end — that single number is the
difference between a derivation and a coincidence.

**V09.** Draw the charge lattice as a lit grid where even-sum sites glow and odd-sum sites stay
dark, then fire a seam current through it as a bright vector that always lands lit. Reciprocal
lattice, selection rule, conserved quantum number — three familiar pictures for one exact fact.

**V10.** Nested shells, four children per parent, with a value pushed up and pulled back down to
show `marg ∘ sect = id`. This is the holographic radial direction and the RG flow: coarse is far,
fine is near.
""",

"control": """# `control/` — the size control

**Panels:** V14 (and the run-size toggle wherever it appears).

## What is here

`mismatch_trace.csv` from the 16,384-patch run at the same simulator revision. Same columns as
`../run/mismatch_trace.csv`.

## How to make physics out of it

This one file carries a real physics result, and it only appears when you compare.

Difference the `record_packet_entropy` column in both runs. At **64k**: 57 steps up, 70 flat,
**zero down**. At **16k**: 41 up, 83 flat, **three down** — at cycles 66, 67 and 69, all after
consensus is reached, together worth 0.04% of the total rise.

Entropy runs *backwards* on those three cycles. That is not a bug and not noise to be smoothed:
it is a fluctuation theorem doing what it says. `⟨e^−σ⟩ = 1` exactly, Jensen gives `⟨σ⟩ ≥ 0`, and
individual negative-production events keep a non-zero probability. They vanish at 64k because
relative fluctuations fall like 1/√N — which is precisely why the second law looks absolute at
everyday scale and stops looking absolute when you make the system small.

Build this as a single toggle that animates between the two sizes, so the reader *watches* the
negative spikes shrink away. Do not put the two runs in separate tabs; the transition is the point.
""",
}
