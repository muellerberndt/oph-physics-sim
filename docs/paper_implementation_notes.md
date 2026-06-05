# Paper Implementation Notes

These notes track what the simulator imports from the current OPH papers and what remains only a
regulator-side diagnostic.

## Imported Design Constraints

- The screen is a support-visible regulator and symmetry chart, not a literal spherical substrate.
- Finite patches and collars are type-I regulator objects; the intended continuum observer net may
  leave that class.
- Local repair should stay quasi-local and record-preserving, matching the local MaxEnt /
  finite-constraint branch.
- The Lorentz-facing branch is carried by modular flow on spherical caps, with the `2*pi`
  normalization and the symmetry bridge `Conf+(S2) ~= SO+(3,1)`.
- Bulk spacetime should be reconstructed as a compression of screen/modular data, not initialized
  as a microscopic bulk lattice.
- The pixel ratio `P` belongs to the D10/Phase-II quantitative closure surface. It supplies the
  local pixel scale and familiar-unit readout. In the simulator it must also define the finite
  screen cell area, since the microphysics screen is a cellulated support-visible chart.
- The fixed-cutoff microphysics surface is a federated patch carrier with exposed ports, records,
  repair interfaces, checkpoints, caps/collars, and edge-sector bookkeeping. The echosahedral
  reference patch has twelve labeled overlap ports.

## Simulator Translation

The current large-screen configs therefore use:

```text
finite spherical screen graph
-> P-scaled cellulated S2 screen architecture
-> mutual-routed federated echosahedral 12-port patch budget
-> local S3 overlap repair
-> record-stability commits
-> explicit pixel/cell-scale receipt
-> finite-cutoff modular-flow surrogate
-> BW cap-flow residual against lambda_C(2*pi*t)
-> record-history modular radial lift
-> sampled point-cloud dimension estimate
```

The modular lift is intentionally labeled `modular_lift_record_history`. It uses the 2D screen chart
plus record-visible modular-depth history as the third reconstructed spatial coordinate. Simulation
cycle number remains repair/cosmological time, not a hidden initialized bulk coordinate.

The current configs declare:

```text
P = a_cell / ell_P^2 ~= 1.6309682094039593
cell_area_planck = P
cell_entropy_capacity = P / 4
cap_area_planck = sum_i P * w_C(i)
cap_entropy_capacity = sum_i (P / 4) * w_C(i)
ports_per_patch = 12
```

Those quantities are emitted as `pixel_report.json`, `pixel_scale.json`, and
`screen_microphysics.json` in every main run bundle. They now define the local screen-cell
normalization layer: cell area, cell entropy capacity, cap area/capacity, residual weighting, and
the echosahedral port budget. The array engine uses mutual-kNN routing so the undirected overlap
graph stays within that port budget, with a small number of unused ports allowed near finite-cell
irregularities.

In the default `numerical_regulator` mode, patch count is a refinement/sampling count and `P` is
used for area/capacity metadata and normalized residual weights. It does not change
`lambda_C(2*pi*t)` and should not be read as causing a finite point-cloud dimension estimator to
equal three; that still requires the BW cap-flow diagnostic.

BW screen configs use a cell-scaled regulator collar:

```text
collar_width = collar_k * sqrt(4*pi/N_patch)
```

This is the finite simulator's current proxy for the paper stack's fixed-cutoff collar with carried
errors that shrink under refinement. The cap/time grid is kept identical across the 4k, 64k, and 1M
BW configs so refinement receipts measure the same observable family.

Before BW scoring, the array engine can apply a finite support-visible extraction proxy:

```text
raw record/sector fields
-> graph-neighbor diffusion on the screen overlap graph
-> restandardized cap-local observable fields
-> BW residual
```

This is not the full weak-* / GNS cap-pair extraction. It is the current finite-regulator
regularization step that removes raw patch-scale noise before measuring cap-flow covariance, matching
the paper stack's emphasis on regularized support-visible modular matrix elements.

## Claim Boundary

This is not yet the support-visible BW theorem. It is an executable diagnostic for whether the
finite patch federation can produce stable overlap records, nontrivial holonomy receipts, and a
3D-like modular-lift reconstruction under controls.

The next theorem-aligned diagnostic is a BW cap-flow residual comparing finite simulated cap
transport against `lambda_C(2*pi*t)` on the support-visible cap chart.

## BW Residual Policy

The radial modular-depth lift is now secondary. The primary OPH/BW diagnostic is:

```text
R_BW(N, C, t, O) =
  distance(simulated finite cap transport of O,
           conformal pullback by lambda_C(2*pi*t) of O)
```

The current implementation is a first finite-regulator transport verifier. It uses explicit round
caps, stereographic cap automorphisms, kNN field pullback, wrong-normalization controls, shuffled
cap/observable controls, and no-flow controls. It is parallelized over cap/time tasks and reuses
one screen KD-tree per run. Residual norms use the OPH cell-entropy measure `P/4`; the conformal
target remains `lambda_C(2*pi*t)`. A fresh 4k P-weighted smoke separates correct `2*pi` transport
from controls. The next receipt should be a repeated 4k/64k/1M P-weighted refinement curve.
