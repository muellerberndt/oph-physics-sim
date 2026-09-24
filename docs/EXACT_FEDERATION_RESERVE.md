# Reserve-generator readout of the settled tower (issue 985)

`oph_exact/federation_reserve.py` measures, on the exact glued federation under the canonical
integer nearest-agreement law, the natural candidates for the objects that the edge-center
reserve-generator receipt of the research repository asks a finite source to emit
(`paper/tex_fragments/SCREEN_SPECTRUM_THEOREMS.tex`, `def:oph-screen-reserve-generator`):
a covariance-survival cocycle across the full oriented collar under logarithmic refinement,
its infinitesimal density `P*/24`, the half-collar identity `P*/48`, and a clock binding.
Receipts: `data/exact/reserve/reserve_L{5..9}.json`. Verifier:
`oph_exact/verify_federation_reserve_independent.py`. Tests:
`tests/test_exact_federation_reserve.py`.

## Identifications (declared before any number was read)

- Depth `m` is the tower level; one refinement step is `b = 2`.
- The collar at depth `m` is the set of inter-face seams of the production gluing, the seams
  that cross the thirty reference edges (cells are face-major). Their number is exactly
  `30 * 2^m` at every level, the certificate's oriented slot count. The two transfer
  directions across a collar seam are the oriented slots; forward runs from the lower face
  index to the higher. Intra-face inter-carrier seams and intra-carrier seams are the other
  two classes.
- The sub-step is one sweep (`|S|` attempts). The tick at depth `m` is `2^m` sweeps, the
  certificate's on-grid family `u_m(1) = (1 - eps 2^-m)^(2^m)`; `2^(m-1)` and `2^(m+1)` sweeps
  are run as controls.
- Reserves: (a) the load of a face, whose presence hazard per sweep is the fraction of all
  units that cross a collar seam in that sweep (descent units plus swaps); (b) the descent
  excess `V - V_min`; (c) the settled-field covariance under one refinement step, read from the
  texture receipts of consecutive levels in the angular-spectrum basis (`C_l` at fixed `l`)
  and in the cap-mean basis.

## Method

A C kernel with the engine's exact move semantics accumulates, per sweep, descents, swaps,
waits and descent units by seam class and direction. The schedules replay the engine's draws
(same loads, seeds, kernel semantics), and every schedule reproduces the engine receipt's
terminal digest, sweep count, ledger and move totals. The readout computes the per-sweep
hazard `h_t`, the depth-scaled hazard `2^m h_t`, the tick survival `u = prod (1 - h_t)` over
the first tick, its generator `-log u`, the semigroup defect `u(2 ticks) / u(1 tick)^2` where
two ticks fit, the forward share of collar crossings, and the per-sweep contraction of the
excess. Negative controls: shuffled seam classes (same trajectory, different collar hazard),
the alternative tick bindings, and the i.i.d. reference without repair (zero hazard).

## Readings (levels 5 to 8; four schedules per level, two at level 5)

| L | sweeps | depth-scaled collar hazard, sweep 1 | after settling, per sweep | generator per tick of `2^m` sweeps | per tick of `2^(m-1)` sweeps | forward share | excess contraction per sweep |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 53 to 59 | 0.0247 to 0.0257 | 0.0117 to 0.0121 | 0.0120 to 0.0123 (32 sweeps) | 0.0064 to 0.0065 | 0.5001 to 0.5042 | 0.828 to 0.831 |
| 6 | 57 to 62 | 0.0253 to 0.0261 | 0.0115 to 0.0122 | tick of 64 sweeps outlives the settlement | 0.0061 to 0.0062 | 0.4989 to 0.5009 | 0.811 to 0.825 |
| 7 | 67 to 86 | 0.0252 to 0.0264 | 0.0117 to 0.0124 | tick of 128 sweeps outlives the settlement | 0.0060 to 0.0061 | 0.4982 to 0.4997 | 0.820 to 0.821 |
| 8 | 69 to 80 | 0.0250 to 0.0263 | 0.0118 to 0.0123 | tick of 256 sweeps outlives the settlement | tick of 128 sweeps outlives the settlement | 0.4980 to 0.4992 | 0.820 to 0.824 |

Targets of the certificate: full-collar density `P*/24 = 0.0680`, half `P*/48 = 0.0340`, with
`P* = 1.6309682094` entering only as the comparison value.

Covariance survival under one refinement step, angular basis, `theta = -log2 lambda`
(white noise: `lambda = 1/4`, `theta = 2`; the receipt needs `theta = 0.034`,
`lambda = 0.977`):

| step | `l` 2 to 8, initial / settled | `l` 9 to 20 | `l` 21 to 40 |
| --- | --- | --- | --- |
| 5 to 6 | 1.90 / 1.86 | 1.86 / 1.60 | 2.22 / 1.31 |
| 6 to 7 | 2.28 / 2.27 | 2.04 / 2.00 | 1.91 / 1.68 |
| 8 to 9 | 1.76 / 1.76 | 2.13 / 2.15 | 2.05 / 2.04 |
| 9 to 10 | 2.33 / 2.32 | 1.96 / 1.94 | 1.83 / 1.80 |

The initial values scatter around 2 because a single realisation's low multipoles are few;
the settled values equal the initial ones within 0.03 below the settling horizon (levels 8 to
10 at every band) and fall by up to 0.9 above it (levels 5 to 7 at `l` 21 to 40).

## What the dynamics supplies, and what it does not

- The depth-scaled collar hazard is one function of the sweep index at every level: the
  certificate's sub-step structure (hazard proportional to `2^-m`) is realised by the repair
  law, and the sweep is a level-independent operational clock (contraction 0.82 per sweep).
- The half identity holds by symmetry: the loads carry no orientation, so the forward share
  is one half with the dynamics' own weights. It is not a source-facing asymmetry.
- Under the dyadic tick binding the tick outlives the settlement from depth 6 upward, because
  the settlement lasts `4.3 log2 N - 9` sweeps while the tick doubles per level. Where the
  tick fits, the generator reads 0.012 (level 5) or 0.006 (half-length binding, levels 5 to 7,
  depth-independent as a cocycle requires), 5.5 to 11 times below the targets; the excess
  reserve reads 0.196 per sweep, 2.9 times above. No reserve lands within a factor of 2.9.
- The settled covariance survives refinement at one quarter per step, as white noise does,
  against the required 0.977. This is the decisive gap: the reserve generator is a property of
  the record ensemble, and the declared i.i.d. loads are white. A receipt with
  `theta = 0.034` needs a record ensemble whose scalar covariance is nearly invariant under
  tower refinement, that is, a source law for the records.

## Reproduction

    python3 -m oph_exact.federation_reserve --level 6 --cache CACHE --out reserve_L6.json --schedules 4 \
        --run-dir RUN_L6 --textures RUN_L5/texture.json RUN_L6/texture.json RUN_L7/texture.json
    python3 -m oph_exact.verify_federation_reserve_independent reserve_L6.json --engine RUN_L6/receipt.json \
        --textures RUN_L5/texture.json RUN_L6/texture.json RUN_L7/texture.json
    python3 -m pytest -q tests/test_exact_federation_reserve.py

`--refresh-refinement` rewrites only the refinement rows of an existing receipt from the
given texture receipts.
