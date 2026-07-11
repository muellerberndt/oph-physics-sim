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

An assumption may populate an H3 chart, supply the BW `2*pi` branch, place that
chart into an open-slicing dS4 background, or render stable defect candidates as
matter. It must not mutate a computed theorem, neutral-bulk, particle,
Einstein, gravity, or physical-CMB receipt.

The canonical visual-universe configurations use the profile
`known_observer_universe_v1`. Its dS4 renderer convention is

```text
ds^2 = -d_tau^2 + H^-2 sinh(H tau)^2 dH3^2
```

with de Sitter radius `R_dS = 1/H`; the manifest validates that relation and
normalizes the renderer values to it. Time sampling and units are also declared
there. These values are renderer inputs unless a separate computed receipt
supplies physical calibration.

Every frontend should display three statuses separately:

- `computed`: verified by the simulator run;
- `assumed_for_visualization`: supplied by the explicit assumption ledger;
- `blocked`: neither computed nor assumed.

This policy makes it possible to visualize the intended OPH observer universe
without pretending that the Python simulator replaces the formal Lean surface
or the paper's branch hypotheses.
