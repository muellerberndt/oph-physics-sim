# Emergent spatial dimensionality probe (lane D1)

Status: exploratory, non-evidential. Every artifact this package produces is
labeled `exploratory_non_evidential`. The deliverable is a table of measured
numbers per configuration. No verdict, no threshold for success, and no
physical claim is attached to any number. Nothing here arms an instrument or
freezes a decision rule.

## 1. Question

The carrier is a 2-sphere by construction: every tower level is a geodesic
icosahedral cellulation of the sphere with `20 * 4**L` triangular cells
(`oph_fpe/core/array_geometry.py`, `oph_fpe/core/icosahedral.py`). The theory
positions the third spatial dimension as the refinement-depth direction, made
physical by join/gluing transport between tower levels. This probe asks one
operational question: does the linearized repair dynamics on a multi-level
union treat depth as a metric dimension, in the sense that the effective
spectral dimension of the coupled operator moves away from the single-level
value near 2 when levels couple through the committed join maps? Either
direction of the answer is a result and is reported as numbers.

## 2. Committed structure mirrored, with citations

The inter-level coupling mirrors the committed join/refinement transport.
The committed sources, read before this design was written:

* `reverse-engineering-reality/Lean/QFT/CarrierJoinTransport.lean`.
  The committed join of two carriers sharing a layer is the pushout along a
  marginalization/section pair: `marg` sums over the fiber, `sect` spreads a
  shared-layer value uniformly over the fiber (`sect h = h / fiber_size`),
  and `marg ∘ sect = id` (`marg88_sect88`). The two embeddings agree exactly
  on the shared layer (`embed_agree_on_hinge`). One-step transport
  intertwines both maps (`evolveFn88_sect88`, `marg88_evolveFn88`) and
  descends to the join (`joinStep_embed88`, `joinStep_embed247`,
  `hingeRestrict_joinStep`). The module's own boundary text records the
  fiber-uniform sections as a declared normalization.
* `reverse-engineering-reality/Lean/QFT/JoinNetMorphism.lean`.
  The record-layer expectation onto the glued layer is section composed with
  marginalization (`recordExpect88` on the left region equals
  `sect88 ∘ marg88`; `joinHingeProject` descends to the join, is idempotent,
  fixes the glued layer, and commutes with transport;
  `joinStep_maps_joinRegionLayer`). The join carries no algebra structure
  (`join_quotient_mul_ill_defined`); only the record (function) layer
  transports. This probe works entirely at the record layer: real-valued
  cell fields.
* `oph_fpe/core/icosahedral.py` (`CellRefinementMap`,
  `GeodesicIcosahedralTower`). The committed refinement transport between
  adjacent tower levels: lineage `child_to_parent` with exactly four children
  per parent, embedding = childwise-constant pullback, conditional
  expectation = spherical-area-weighted child average with weights
  `conditional_expectation_weights` that sum to one over each parent's
  children. The pair is unital, positive, a left inverse, and preserves the
  normalized spherical-area state. This is the sim-side instance of the Lean
  section/marginalization pair, with area weights in place of the exact
  fiber-uniform `1/4`.
* `oph_fpe/em/base_carrier.py` and `oph_fpe/em/green.py`. The committed
  level-0 operator is the unit-seam-weight graph Laplacian
  `L = boundary ∘ coboundary` with kernel equal to the constants; the Green
  solve and the repair-response observables are built on it. This commits
  the unit-weight seam Laplacian as the intra-level disagreement operator at
  level 0.
* `reverse-engineering-reality/paper/recovering_observer_spacetime_and_einstein_dynamics_from_overlap_consistency.tex`.
  The abstract and derivation map: the celestial 2-sphere, the refinement
  tower carrying every continuum condition, and the statement that overlap
  and refinement gluing remain open premises. The paper does not fix an
  inter-level coupling strength or a union-graph presentation; those are
  declared conventions below.

## 3. State and dynamics

### 3.1 State

Real-valued cell fields `x_L` on each tower level `L` in a window
`[0, L_max]`, `L_max <= 5` (cell counts 20, 80, 320, 1280, 5120, 20480;
level-5 union node count 27300). The union state space is the direct sum of
the level fields; union node indexing is by ascending level with offsets
`offset[L] = sum of cell counts below L` (declared convention C1).

### 3.2 Intra-level coupling

The graph Laplacian of the level's cell adjacency: one unit-weight edge per
shared mesh edge (dual seam), from
`geodesic_icosahedral_patch_arrays(level, patch_basis="cells")`. Every level
is 3-regular. The linearization reading: repair relaxes disagreement between
overlapping neighbors; the quadratic disagreement energy per seam is
`(x_i - x_j)^2` and its gradient flow is the graph heat flow. The unit seam
weight is the committed level-0 normalization of
`oph_fpe/em/base_carrier.laplacian_matrix` (port basis) carried to the cell
basis and to every level uniformly (declared convention C2).

### 3.3 Inter-level coupling

For each adjacent level pair `(L, L+1)` in the window, one edge per
committed lineage pair `(parent p, child c)` from
`CellRefinementMap.child_to_parent`, with weight

```
w(p, c) = kappa * conditional_expectation_weights[c]
```

where `kappa` is scanned over the pinned grid and the second factor is the
committed spherical-area expectation weight of the child (sums to one over
each parent's four children). The mirrored structure is exact at
stationarity: minimizing the inter-level quadratic energy
`kappa * sum_c w_c (x_{L+1}(c) - x_L(p(c)))^2`

* in the fine field alone sets `x_{L+1} = embed(x_L)` (the committed
  childwise-constant pullback, the sim instance of the Lean `sect`), and
* in the coarse field alone sets `x_L = conditional_expectation(x_{L+1})`
  (the committed area-weighted child average, the sim instance of the Lean
  `marg` followed by normalization; `E = sect ∘ marg` is the committed
  record-layer expectation of `JoinNetMorphism.lean`).

Single-site relaxation of the coupling term therefore moves each field
toward the two committed presentations of one shared datum, which is the
join/gluing transport read as overlap repair. The coupling strength `kappa`
and the quadratic form itself are not fixed by the corpus (declared
conventions C3, C4).

### 3.4 Probe operator

The symmetric weighted graph Laplacian `L_union = D - W` on the union node
set, with `W` the union of intra-level unit weights and inter-level
`kappa`-scaled committed weights. `L_union` is symmetric positive
semidefinite by construction; symmetry is verified and recorded per
configuration (`symmetry_max_abs_asymmetry`, required exactly 0.0).

### 3.5 Controls

* DECOUPLED control: `kappa = 0` (block-diagonal union; kernel dimension
  equals the level count) and single-level operators at `L` in `{3, 4, 5}`.
* STATIC control: the union graph with every weight set to 1.0, intra-level
  and inter-level alike. This control is partially by construction: its
  spectral content reflects the fixed union graph shape, not the
  dynamics-derived weights. It is labeled `construction_control` in the
  receipt and the report.

## 4. Estimators (all pinned before running)

### 4.1 Heat-kernel spectral dimension

Return probability `P(sigma) = (1/N) tr exp(-sigma * L_union)` on the pinned
sigma grid `numpy.logspace(-3, 4, 71)` (ten points per decade). Spectral
dimension curve by centered differences in log-log:

```
d_s(sigma_i) = -2 * (ln P_{i+1} - ln P_{i-1}) / (ln sigma_{i+1} - ln sigma_{i-1})
```

Trace paths:

* Dense path (primary for `N <= 2000`, the pinned dense cap):
  `numpy.linalg.eigvalsh` of the dense operator; `P` from the full spectrum.
* Stochastic path (primary for `N > 2000`): Hutchinson trace with
  Rademacher probes, 128 probes split into eight independent sixteen-probe
  seed batches. The reported curve is their mean; the pointwise standard
  error and the standard error, minimum, and maximum of the guarded
  per-seed plateau medians are recorded. Each probe's scalar
  curve `z^T exp(-sigma L) z` is evaluated for the whole sigma grid from one
  Lanczos factorization of 120 steps with full reorthogonalization
  (Gauss quadrature on the probe's spectral measure). Exact kernel
  deflation: probes are projected orthogonal to the exact kernel basis
  (per-connected-component indicator vectors, components known by
  construction), every Lanczos vector is re-orthogonalized against that
  basis, and the exact kernel contribution `kernel_dim / N` is added in
  closed form. Ritz values are clipped at zero from below.

Cross-check (asserted in tests on dense-cap-sized graphs): the two paths
agree with `max |d_s_dense - d_s_stochastic| <= 0.15` pointwise over the
comparison window and `|median of d_s difference| <= 0.10` between the two
window medians; return-probability relative error `<= 0.05` over the same
window. The comparison window is the set of grid points with
`sigma <= 1 / lambda_2`, finite `d_s` on both paths, and dense
`P(sigma) >= 4 * kernel_dim / N` (saturation guard); the `sigma_lo` floor
of 4.2 applies to the plateau statistic only. Recorded fixes 2026-08-20:
(a) the first draft compared over the 4.2 fit window; that window is empty
for `torus3d_12` (`lambda_2 = 0.268`, `sigma_hi = 3.73 < 4.0`), so the
agreement check is pinned to the comparison window above. (b) Within a few
relaxation times of `1 / lambda_2` the non-kernel trace remainder falls
under the 64-probe Hutchinson noise floor and no fixed probe count holds a
relative tolerance there (measured on `cycle_512`: `max |d_s|` difference
0.2349 unguarded at `sigma` in {5012, 6310} where dense `P` is under
`4 / N`, against 0.0385 with the guard), so the saturation guard above is
part of the pinned window. The tolerances did not move.

Probe-count reductions, if any configuration is slow, are recorded in the
receipt under `probe_count_reductions`; the field is present and empty when
no reduction occurred. No silent reduction.

### 4.2 Fit window (pinned rule)

* `sigma_lo = 4.0`, an absolute constant: four unit-weight hop times, past
  the ballistic-to-diffusive knee of unit-scale lattice Laplacians. The
  calibration suite checks this placement.
* `sigma_hi = 1.0 / lambda_2`, the relaxation time of the slowest nonzero
  mode; `lambda_2` is the smallest nonzero eigenvalue from the Weyl
  eigensolve (4.3). Beyond a few multiples of this time the return
  probability saturates at `kernel_dim / N` and the curve leaves the
  scaling regime.
* Window `W` = grid points with `sigma_lo <= sigma <= sigma_hi` and
  `P(sigma) >= 4 * kernel_dim / N`. This is the same kernel-saturation guard
  used by the dense/stochastic cross-check; it applies to every reported
  plateau median, not only to that cross-check. If `W` has
  fewer than 5 points the configuration is reported with
  `window_degenerate = true` and the point count; the statistic is still
  reported over the available points, or `null` at zero points.
* Plateau statistic: the median of `d_s` over `W` (the grid is log-spaced,
  so this is the log-median). The window minimum, maximum, endpoints, and
  point count are recorded next to it.

### 4.3 Weyl-law dimension (independent second view)

The `k = 200` smallest eigenvalues (or `N - 2` if smaller) by
`scipy.sparse.linalg.eigsh` in shift-invert mode at `sigma_shift = -1e-6`
with a pinned start vector (dense path: from the full spectrum). Zero modes
(eigenvalues `<= 1e-9`) are the kernel and are cross-checked against the
construction component count. Nonzero eigenvalues ranked `i = 1..M` define
the counting function `N(lambda_i) = i`; least squares of `ln i` against
`ln lambda_i` over ranks `i in [8, M]` gives `d_weyl = 2 * slope`. The rank
floor 8 skips low-lying degenerate multiplets. Recorded fix 2026-08-20:
the first draft pinned `k = 64`; pre-run calibration on exact torus
spectra places the `torus3d_24` Weyl fit at 2.801 with `k = 64`, outside
the pinned band, and at 3.027 with `k = 200`. The eigencount moved; the
band did not.

## 5. Calibration suite (mandatory)

The same estimator code, the same pins, applied to graphs of known
dimension. Tolerances are pinned here; if calibration fails, the estimator
gets fixed and the fix is recorded; the tolerance does not move.

| case | graph | nodes | expectation | assertion |
|---|---|---|---|---|
| `cycle_4096` | cycle `C_4096` | 4096 | `d = 1` | `|d_s_median - 1| <= 0.15` and `|d_weyl - 1| <= 0.15` |
| `torus2d_64` | 64 x 64 periodic square lattice | 4096 | `d = 2` | `|d_s_median - 2| <= 0.15` and `|d_weyl - 2| <= 0.15` |
| `torus3d_24` | 24^3 periodic cubic lattice | 13824 | `d = 3` | `|d_s_median - 3| <= 0.15` and `|d_weyl - 3| <= 0.15` |
| `tree4_depth8` | rooted 4-regular tree (root degree 4, interior degree 4), depth 8 | 13121 | anomalous, non-integer control | `d_weyl` outside every band `[d - 0.15, d + 0.15]`, `d in {1, 2, 3}`; `d_s_median` band status recorded without an assertion |

The tree control's rationale: a 4-regular tree has exponential volume
growth; the infinite-tree return probability carries the factor
`exp(-(4 - 2 sqrt(3)) sigma)`, so `d_s(sigma)` has no finite plateau.
Recorded fix 2026-08-20: on the finite truncated tree the window cap at
`1 / lambda_2` cuts the growth of `d_s(sigma)` and the window median lands
near 1.9 (measured 1.8457 at depth 6, 1.9011 at depth 8, the latter inside
the `d = 2` band), so the asserted band exclusion moved to `d_weyl`
(measured 1.6262 at depth 8, outside every band); `d_s_median` and the
full curve stay recorded. The bands did not move; the asserted statistic
did.

Dense/stochastic cross-check variants (both paths run, agreement asserted
per 4.1): `cycle_512` (512), `torus2d_40` (1600), `torus3d_12` (1728),
`tree4_depth6` (1457).

## 6. Exploratory sweep (decision-free reporting)

Configurations, every one reported, none thresholded:

* Calibration: the four cases of section 5 plus the four cross-check
  variants.
* Single-level controls: level `L` in `{3, 4, 5}`, intra-level operator
  only.
* Coupled unions: levels `[0..L]` for `L` in `{3, 4, 5}`, `kappa` in
  `{0.25, 0.5, 1.0, 2.0}` (dynamics-weighted operator) and `kappa = 0`
  (decoupled union control).
* Static-control unions: levels `[0..L]` for `L` in `{3, 4, 5}`, all
  weights 1.0, labeled `construction_control`.

The deliverable is the table: per configuration the `d_s` window median
(with window data), the Weyl dimension, `lambda_2`, `lambda_max`, kernel
dimension, node/edge counts, estimator path, and runtime. The static union
is partially by construction; the dynamics-weighted operator family is the
informative object; the single-level control is expected near 2 and the
measured value is reported as measured.

## 7. Seeds, sizes, receipt schema

* `HUTCHINSON_SEED = 20260820` (Rademacher probes,
  `numpy.random.Generator(numpy.random.PCG64(seed))`, one base seed per
  configuration derived by `seed + block + index` with block 1000 for
  calibration rows, 2000 for cross-check rows, 3000 for tower rows,
  and eight child seeds `base + j * 100003`, `j = 0..7`, recorded per row
  where the stochastic path runs).
* `EIGSH_SEED = 20260821` (pinned `v0` start vectors).
* Probes: 128 total, split into eight independent batches of sixteen. Lanczos
  steps: 120, full reorthogonalization. Dense cap:
  2000 nodes. Sigma grid: `logspace(-3, 4, 71)`. Weyl: `k = 200` (recorded
  fix of 4.3), rank floor 8, shift `-1e-6`, zero-mode cut `1e-9`. Kappa
  grid: `{0.0, 0.25, 0.5, 1.0, 2.0}`. Level ceiling: 5.
* Receipt: one JSON document, canonical serialization
  (`json.dumps(..., sort_keys=True, separators=(",", ":"))` plus a trailing
  newline), every float rounded to 10 significant digits before
  serialization (`float(f"{x:.10g}")`), SHA-256 of the exact bytes printed
  by the `__main__` CLI and written next to the receipt. Determinism scope:
  identical bytes on repeated runs on one interpreter/BLAS build.
  Wall-clock timings are excluded from the hashed receipt and live in the
  unhashed sidecar `dimension_probe_timings.json` next to it (recorded fix
  2026-08-20: the first draft placed `runtime_seconds_total` inside the
  receipt, which contradicts the byte-reproducibility scope).
* Receipt top-level keys: `schema` (`oph.dimension_probe.v1`),
  `evidential_status` (`exploratory_non_evidential`), `claim_boundary`,
  `environment` (python/numpy/scipy versions), `pins` (everything in this
  section plus the window rule constants), `conventions` (section 8
  strings), `citations` (section 2 module names), `calibration` (rows),
  `cross_checks` (rows per 4.1), `configurations` (rows, each with the
  fields of section 6 plus the per-config seed,
  `symmetry_max_abs_asymmetry`, the `P(sigma)` and `d_s` curves on the
  pinned grid), and `probe_count_reductions`. Wall-clock seconds live in
  the unhashed timing sidecar.

## 8. Declared conventions beyond the committed corpus

* C1 (union presentation). The probe state keeps every level's cells as
  separate nodes and couples them, rather than quotienting the coarse level
  onto the childwise-constant subspace of the fine level as the Lean pushout
  would when the coarse carrier is itself the shared layer. The
  union-with-coupling presentation makes a coupling-strength family
  scannable; the static control exposes the bare union graph.
* C2 (intra-level normalization). Unit seam weight at every level, the
  level-0 convention of `oph_fpe/em/base_carrier.laplacian_matrix` extended
  level-uniformly and carried from the port basis to the committed cell
  patch basis. The corpus does not fix relative intra-level time scales
  across levels.
* C3 (inter-level quadratic form). The coupling enters as a symmetric
  graph Laplacian built from one parent-child edge per committed lineage
  pair, weighted by the committed conditional-expectation weight. The
  corpus commits the transport maps, not a quadratic energy; the pairwise
  form above is the declared linearization whose single-field stationarity
  reproduces the committed `embed` and `conditional_expectation` exactly.
* C4 (coupling strength). `kappa` grid `{0.0, 0.25, 0.5, 1.0, 2.0}`;
  the corpus is silent on the strength.
* C5 (estimator pins). Sigma grid, window constants (`sigma_lo = 4.0`,
  `sigma_hi = 1 / lambda_2`, `P >= 4 * kernel_dim / N`, minimum 5 points),
  probe ensemble and Lanczos counts,
  Weyl `k`/rank floor/shift/zero cut, dense cap, and seeds, as in section 7.
* C6 (receipt float convention). 10 significant digits before canonical
  serialization.
* C7 (area weights as the fiber weights). The Lean sections are
  fiber-uniform (`1/4` per child); the sim commits spherical-area weights.
  The committed area weights are used; the maximum deviation from `1/4` is
  recorded per level pair in the receipt.
* C8 (saturation guard). Every reported plateau median, and the estimator
  agreement comparison, is restricted to
  grid points with `sigma <= 1 / lambda_2`, finite curves on both paths,
  and dense `P(sigma) >= 4 * kernel_dim / N` (saturation guard); the
  plateau window additionally applies `sigma >= 4.0`, while the agreement
  check does not.
* C9 (timing sidecar). Wall-clock values live outside the hashed receipt,
  in `dimension_probe_timings.json` next to it.

## 9. What the numbers do and do not show

The numbers are spectral statistics of declared finite operators. The
static union statistic is partially a property of the constructed graph.
The dynamics-weighted statistics are properties of the declared
linearization of committed transport, at declared coupling strengths, on
finite windows of a finite tower; they carry the window and size effects
recorded next to them. None of the numbers is a measurement of physical
spatial dimension, and no number here discharges any open premise row of
the corpus (overlap gluing, physical scale, spacetime attachment stay open
premises of the committed papers).
