# Simulation Assumption Policy

`oph-physics-sim` has two independent evidence lanes.

1. **Computed receipts** are produced only from checks implemented by the
   simulator. Missing inputs, failed invariants, or unavailable refinement data
   keep these receipts false.
2. **Simulation assumptions** complete a coherent visual scene when a paper
   bridge is proved elsewhere, intentionally outside the simulator, or simply
   accepted for explanatory rendering. They are recorded in
   `simulation_assumption_manifest.json` with
   `provenance=explicit_simulation_assumption`.

The canonical ledger contains thirteen explicit assumptions: finite S2 screen,
BW `2*pi` geometric branch, modular time interpretation, H3 observer chart,
S2-observer-to-H3-camera embedding, record population on H3, refinement-natural
visual handoff, open-slicing dS4 background, positive cosmological constant,
observer tetrads, defect-as-matter styling, screen-to-temperature rendering, and
a pinned TT reference shape. Every row is independently visible and must be a
literal JSON boolean. It must not mutate a computed theorem, neutral-bulk,
particle, Einstein, gravity, or physical-CMB receipt.

The canonical visual-universe configurations use the profile
`known_observer_universe_v1`. Its dS4 renderer convention is

```text
ds^2 = -d_tau^2 + H^-2 sinh(H tau)^2 dH3^2
```

with de Sitter radius `R_dS = 1/H`; the manifest validates that relation and
normalizes the renderer values to it. Time sampling and units are also declared
there. These values are renderer inputs unless a separate computed receipt
supplies physical calibration.

The CMB completion lane uses a declared local reference file, source URL,
SHA-256 digest, transfer-model identifier, and integer sky seed. Export fails
closed for that layer when the file is missing, its bytes do not match the
declared digest, or its rows cannot be parsed. `referenceRows` are pinned
measurement/reference values and `assumedModelRows` are the source table's
published best-fit reference; neither is an OPH prediction. A frontend may use
the seed to make a deterministic explanatory S2 realization, but it must keep
the gold assumption badge and the physical-CMB receipt false.

Every frontend should display three statuses separately:

- `computed`: verified by the simulator run;
- `assumed_for_visualization`: supplied by the explicit assumption ledger;
- `blocked`: neither computed nor assumed.

Frontends must also keep assumption completeness and data completeness
separate. `SIMULATION_ASSUMPTIONS_COMPLETE_RECEIPT` says the explicit ledger is
internally valid. `VISUAL_UNIVERSE_RENDER_DATA_COMPLETE_RECEIPT` says the
required exported layers are present. Only their conjunction is the
visualization-only `VISUAL_UNIVERSE_RENDER_READY_RECEIPT`; none is a proof or
physical-measurement receipt.

This policy makes it possible to visualize the intended OPH observer universe
without pretending that the Python simulator replaces the formal Lean surface
or the paper's branch hypotheses.
