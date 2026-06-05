# BW Refinement Test: 2026-06-01

This records the post-`P` BW/cap-flow test after revisiting the OPH paper stack and the `extra/`
screen notes.

## Paper Corrections Applied

The first P-weighted sweep still separated controls but did not give a clean refinement curve. A
paper review identified three implementation mismatches:

1. `P` must remain a local area/capacity normalization, not a BW normalization or dimension knob.
2. The BW refinement sweep must use the same cap/time observable family across patch counts.
3. The finite collar should be tied to the regulator resolution; a fixed angular collar is not a
   good proxy for the paper stack's shrinking collar with carried errors.
4. Raw record/sector fields are too regulator-scale. The BW theorem concerns regularized
   support-visible matrix elements on the extracted prime geometric cap pair, so the simulator needs
   a finite support-visible extraction proxy before scoring cap covariance.

Implemented changes:

- `P = 1.6309682094039593` is used for `cell_area_planck`, `cell_entropy_capacity`, cap capacity,
  and BW residual weights.
- `lambda_C(2*pi*t)` remains unchanged.
- BW configs now use the same cap count, cap sizes, time grid, dynamics, and modular-flow parameters.
- BW configs use `collar_width = collar_k * sqrt(4*pi/N_patch)`.
- BW fields pass through graph-neighbor smoothing before scoring as a finite support-visible
  regularization proxy.

## Final Comparable Sweep

```text
runs/e1_s3_bw_screen_4k_1780293892
  patches: 4,096
  final Phi: 0
  R_BW median: 0.385094900337284
  R_BW p90: 0.470952585890742
  control medians: 0.6265728456377062 to 1.2573969652631858

runs/e1_s3_bw_screen_64k_1780293893
  patches: 65,536
  final Phi: 0
  R_BW median: 0.38272334680897835
  R_BW p90: 0.4696630435668558
  control medians: 1.0886953936652999 to 1.2721357778565607

runs/e1_s3_bw_screen_1m_1780293924
  patches: 1,048,576
  final Phi: 0
  R_BW median: 0.38353693032266634
  R_BW p90: 0.46518841336585426
  control medians: 1.2598824098616817 to 1.2754244217662838
```

## Interpretation

Passes:

- All three runs settle to `Phi=0`.
- Correct `2*pi` transport is well below wrong-normalization, shuffled-cap, shuffled-observable, and
  no-flow controls.
- The p90 residual improves monotonically from 4k to 64k to 1M.
- The corrected branch is much cleaner than the raw P-weighted scalar-field sweep.

Still not passed:

- The median refinement curve is not strictly monotone: it improves from 4k to 64k, then rebounds
  slightly at 1M.

Claim boundary:

> The simulator now has a paper-aligned finite BW/cap-flow diagnostic with P-weighted capacity
> normalization, cell-scaled collars, and support-visible field regularization. It passes control
> separation and high-quantile improvement, but it is not yet a theorem-grade finite approximation to
> the support-visible BW scaling theorem because the median residual does not yet show clean monotone
> refinement.

## Remaining Paper-Side Gap

The current implementation still transports regularized scalar fields. The compact paper's BW
surface is stronger: it uses a finite cap algebra/state, collar/Markov recovery control, regularized
modular transport, weak-* / GNS support extraction, and automorphism-level convergence on the prime
geometric cap pair.

The next implementation should therefore add:

1. finite cap-local algebras or low-rank cap-state matrices;
2. reduced cap states `rho_C^(delta)` and regularized modular generators `K_a = -log(rho_C + a I)`;
3. collar/Markov diagnostics, preferably conditional-mutual-information or splice residuals;
4. matrix-element/moment residuals for cap-local observables, not pointwise scalar-field residuals;
5. a refinement sweep over the same cap-local test family with carried collar-error accounting.

Until that exists, do not claim that OPH-FPE has demonstrated the full BW/Lorentz theorem surface.
The current result is a stronger finite-regulator receipt that the simulator is now testing the
right direction.
