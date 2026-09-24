# Exact source-net causal limit (lane L2)

Producer `oph_exact/source_net.py`, verifier
`oph_exact/verify_source_net_independent.py`, tests
`tests/test_exact_source_net.py`, receipt
`data/exact/source_net_causal_limit_receipt.json` (schema
`oph.exact.source-net-causal-limit.v1`).

```
.venv/bin/python -m oph_exact.source_net --write      # rebuild the receipt (about 2.5 min on 10 cores)
.venv/bin/python -m oph_exact.source_net --check      # rebuild and compare byte for byte
.venv/bin/python -m oph_exact.source_net --repin      # refresh the file pins of the committed receipt
.venv/bin/python -m oph_exact.verify_source_net_independent
.venv/bin/python -m pytest tests/test_exact_source_net.py -q
```

## 1. The construction

The lane replays the r2039 construction of the theory repository
(`code/causal_refinement/source_net_causet.py`, propositions in
`paper/tex_fragments/SOURCE_NET_CAUSAL_LIMIT.tex`, clock in
`paper/tex_fragments/SOURCE_COUNT_CLOCK.tex`, cone statement in
`Lean/Geometry/SourceNetCausalCone.lean`) and extends it from
`q = 3, 5, 8, 13` to `q = 5, 8, 13, 21, 34, 55`.

* Golden orbit: `xi_b = b*phi - floor(b*phi)` for `0 <= b < q`, `q = F_n`,
  stored as the integer pair `(m, b)` meaning `m + b*phi` with
  `m = -floor(b*phi)`.  `phi = (1 + sqrt 5)/2`.
* Population: every triple `b in [0, q)^3`, position `s(b) = L (xi_b1, xi_b2, xi_b3)`
  in the window `[0, L]^3`, `L^2 = 12/5 - (4/5) phi`, `L = 2/sqrt(phi + 2)`.
  The source record `z(b)` and its integer current section are RER's
  `source_control` / `source_currents`; their digest is part of the cross-check.
* Read law: an event at `(j+1, t)` reads every `(j, s)` with `|t - s| <= a_q`,
  `a_q = L/sqrt q`, including `s = t` (waiting).  The decision
  `q*(A + B*phi) <= 1` on the squared Gram distance `(A + B*phi) L^2` uses the
  exact sign rule of RER's `sign()`: for `x = a + b*phi`, `c = 2a + b` and the
  sign of `c^2 - 5 b^2` decide.  The rule is vectorized in int64.
* Layers: `K_q = ceil(sqrt q)`, model time `T_q = K_q a_q`, layer duration
  `Delta_q = a_q/c` with `c = 1`.
* Order: `(j, s) <= (j', t)` iff `d(s, t) <= j' - j`, `d` the graph distance on
  the site graph.  Distances come from breadth-first search; the per-start
  searches for pair counting are pruned to the balls
  `B(x, (K - alpha_s - m) a_q) U B(y, (K - alpha_s - m) a_q)` at layer `m`,
  which contain every shortest path to a target that can form an ordered
  pair inside the interval, so the pruned distances are exact on those pairs
  (`tests/test_exact_source_net.py::test_cone_pruned_bfs_matches_full_distances_on_relevant_pairs`
  and the explicit-enumeration tests check this).
* Density: `rho_q = q^3/(L^3 Delta_q) = q^(7/2)/L^4`.  Normalized counts
  `N/rho_q` are reported in units of `L^4` (the controls use `q^(d+1/2)/L^(d+1)`).

Levels of the theory receipt are cross-checked field by field: neighbour
digest (sha256 of the canonical JSON of the neighbour lists, waiting
included), undirected edge count, minimum and maximum neighbour counts,
reachability probes (starts 0, centre, last; reachable ids and finite-cone
missing ids digests), centre intervals (`counts_by_layer`), orbit, grid
permutation, covering radius `r_q`, `h_q^2`, `h/a` upper bound, certified
inner speed, source-record digest and word lengths.  All 28 compared fields
agree at `q = 5, 8, 13` against the pinned theory receipt
(`sha256 c0f790ad38...5681`).  The theory's `q = 3` level is below the lane's
range.

## 2. Statistics per level

For each `q` and each family (three-dimensional source net, two- and
one-dimensional controls) the receipt carries:

* the vertical centre diamonds with `k = 1..K` layers (tips `(0, x)`, `(k, x)`,
  `x` the RER centre site): inclusive event count `N`, `counts_by_layer`,
  `N/rho_q` against `pi c^3 T^4/24` and against the diamond clipped to the
  cube (the `K`-layer diamond has `T/2 >= L/2`, so it touches or leaves the
  cube at every level), the paper's bound (eq. `source-net-volume-error`)
  with `H_q = 2 sqrt3 L/q`, `h_q = sqrt3 L r_q`, `a_q`, `Delta_q`, the flag of
  its hypothesis `B(x, T/2 + H_q)` inside the cube, the strict ordered pair
  count `C`, the fraction `2C/(N(N-1))`, its distance to `1/10`, the inverted
  Myrheim-Meyer dimension (`oph_fpe.bulk.causet_likeness`; `f(4) = 1/10`
  exactly);
* one moving-tip diamond with `K` layers;
* the count clock `(N_K/N_K')^(1/4)` with `K' = floor(K/2)` against `K/K'`,
  and the finite enclosure of `SOURCE_COUNT_CLOCK.tex` evaluated with the
  bound values.

Pair counting is exact all-pairs at every level of the committed receipt
(`--exact-support-limit 5000000`; the vertical support has 98,598 sites at
`q = 55` and 412,526 at `q = 89`, the moving support 26,496 and 369,757).
Below the declared limit the producer falls back to stratified estimates from
seeded samples (proportional allocation over `d(x, s)` strata, at least two
starts per stratum, without replacement, seed base `20260909`), with exact
per-start counts, per-stratum sums and sums of squares, and the standard
error; the verifier recomputes such an estimate and its error from the strata
rows.  The sampled `q = 55` values of the 2026-09-09 receipt, 0.0926 (SE
0.0003) and 0.1057 (SE 0.0015), enclose the exact values 0.0925 and 0.1077.

### Three spatial dimensions (target `1/10`), vertical `K`-layer diamond

| q | K | N | N/rho_q | pi T^4/24 | clipped | bound | 2C/(N(N-1)) | MM dim | (N_K/N_K')^(1/4) | K/K' | counting |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| 5 | 3 | 80 | 0.2862 | 0.4241 | 0.3850 | 62.35 | 0.3063 | 2.634 | 2.515 | 3.0 | exact |
| 8 | 3 | 188 | 0.1298 | 0.1657 | 0.1650 | 20.31 | 0.2414 | 2.932 | 3.114 | 3.0 | exact |
| 13 | 4 | 1529 | 0.1930 | 0.1983 | 0.1964 | 8.26 | 0.0882 | 4.149 | 1.705 | 2.0 | exact |
| 21 | 5 | 6482 | 0.1527 | 0.1855 | 0.1849 | 5.82 | 0.1383 | 3.612 | 2.010 | 2.5 | exact |
| 34 | 6 | 32265 | 0.1408 | 0.1468 | 0.1467 | 2.15 | 0.0920 | 4.099 | 2.088 | 2.0 | exact |
| 55 | 8 | 212252 | 0.1720 | 0.1772 | 0.1768 | 2.25 | 0.0925 | 4.093 | 1.907 | 2.0 | exact |
| 89 | 10 | 1081730 | 0.1626 | 0.1653 | 0.1651 | 1.05 | 0.0942 | 4.071 | 2.051 | 2.0 | exact |

Volumes and bounds in units of `L^4`.  The bound is the value of the formula;
its hypothesis fails for the `K`-layer diamond at every level (the diamond
leaves the cube), and the bound exceeds the volume by a factor 13 to 150 at
these `q`.  The actual deviation of `N/rho_q` from `pi T^4/24` is 2.9 percent
at `q = 55` (2.7 percent against the clipped volume).

Largest `k` with the bound hypothesis `B(x, T/2 + H_q)` inside the cube:

| q | k_in | N | N/rho_q | pi T^4/24 | rel. dev. | bound/volume | 2C/(N(N-1)) | MM dim |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| 21 | 3 | 792 | 0.0187 | 0.0240 | -0.224 | 54 | 0.2363 | 2.959 |
| 34 | 4 | 7457 | 0.0325 | 0.0290 | +0.122 | 21 | 0.0769 | 4.310 |
| 55 | 6 | 71212 | 0.0577 | 0.0561 | +0.029 | 15 | 0.0838 | 4.209 |
| 89 | 8 | 455669 | 0.0685 | 0.0677 | +0.012 | 7 | 0.0900 | 4.125 |

At `q = 5, 8, 13` no diamond beyond one layer satisfies that hypothesis.

Moving-tip diamond (`K` layers, tips symmetric about the centre on the rank
diagonal, smallest rank shift with the declared buffer rule):

| q | shift | l/T | tau/L | buffer/L | rule | N | N/rho_q | V(D) | rel. dev. | 2C/(N(N-1)) | MM dim | counting |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| 5 | none | | | | | | | | | | | |
| 8 | 2 | 0.771 | 0.675 | 0.060 | positive | 32 | 0.0221 | 0.0273 | -0.189 | 0.2500 | 2.889 | exact |
| 13 | 3 | 0.737 | 0.750 | 0.029 | positive | 238 | 0.0300 | 0.0413 | -0.273 | 0.1349 | 3.642 | exact |
| 21 | 4 | 0.606 | 0.868 | 0.009 | positive | 2512 | 0.0592 | 0.0742 | -0.202 | 0.1210 | 3.772 | exact |
| 34 | 8 | 0.795 | 0.625 | 0.102 | at least H_q | 3177 | 0.0139 | 0.0199 | -0.304 | 0.1151 | 3.833 | exact |
| 55 | 13 | 0.758 | 0.703 | 0.070 | at least H_q | 31935 | 0.0259 | 0.0321 | -0.193 | 0.1077 | 3.911 | exact |
| 89 | 17 | 0.624 | 0.828 | 0.040 | at least H_q | 369757 | 0.0556 | 0.0616 | -0.097 | 0.1024 | 3.971 | exact |

`V(D) = pi tau^4/24` with `tau^2 = T^2 - l^2`.  A tip at the centre admits no
`K`-layer moving diamond with a positive buffer: the face-normal extent of the
diamond's spatial projection is at least `T/2 > L/2` for every separation, so
the lane places the two tips symmetric about the centre.  The rule
"buffer at least `H_q`" fires where a shift satisfies it (`q = 34, 55`); the
positive-buffer rule fires otherwise; `q = 5` has no timelike symmetric pair
with a positive buffer.  The general bound (eq.
`source-net-general-volume-error`) is in the receipt and exceeds `V(D)` by
factors of 800 or more at these `q`.

### Controls, vertical `K`-layer diamond

Two spatial dimensions, target `8/35 = 0.2286`:

| q | K | N | N/rho_q | pi T^3/12 | clipped | 2C/(N(N-1)) | MM dim | moving-tip fraction | moving MM dim |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| 5 | 3 | 28 | 0.5009 | 0.6322 | 0.5741 | 0.4444 | 2.156 | none | |
| 8 | 3 | 52 | 0.2873 | 0.3124 | 0.3106 | 0.3333 | 2.527 | none | |
| 13 | 4 | 205 | 0.3364 | 0.3575 | 0.3530 | 0.2205 | 3.045 | 0.3476 | 2.474 |
| 21 | 5 | 606 | 0.2999 | 0.3401 | 0.3381 | 0.2729 | 2.779 | 0.2369 | 2.956 |
| 34 | 6 | 1743 | 0.2586 | 0.2852 | 0.2851 | 0.2276 | 3.005 | 0.2425 | 2.927 |
| 55 | 8 | 7040 | 0.3138 | 0.3286 | 0.3272 | 0.2255 | 3.017 | 0.2404 | 2.938 |
| 89 | 10 | 22561 | 0.3019 | 0.3118 | 0.3111 | 0.2245 | 3.022 | 0.2350 | 2.966 |

One spatial dimension, target `1/2`:

| q | K | N | N/rho_q | T^2/2 | clipped | 2C/(N(N-1)) | MM dim |
|:--|:--|:--|:--|:--|:--|:--|:--|
| 5 | 3 | 10 | 0.8944 | 0.9000 | 0.8401 | 0.6889 | 1.561 |
| 8 | 3 | 12 | 0.5303 | 0.5625 | 0.5591 | 0.6061 | 1.739 |
| 13 | 4 | 29 | 0.6187 | 0.6154 | 0.6078 | 0.4975 | 2.007 |
| 21 | 5 | 54 | 0.5611 | 0.5952 | 0.5910 | 0.5220 | 1.942 |
| 34 | 6 | 97 | 0.4893 | 0.5294 | 0.5289 | 0.5006 | 1.998 |
| 55 | 8 | 231 | 0.5663 | 0.5818 | 0.5786 | 0.4980 | 2.005 |
| 89 | 10 | 459 | 0.5467 | 0.5618 | 0.5600 | 0.4970 | 2.008 |

No symmetric timelike pair with a positive buffer exists in one dimension
at these levels (the projection of a moving diamond is the segment of length
`T > L` about the midpoint).

Inverted Myrheim-Meyer dimension at `q = 55`, side by side: three
spatial dimensions 4.09 (vertical) and 3.93 (moving tips); two 3.02 and 2.94;
one 2.005.  All pair counts of the controls are exact.

## 3. Reading

* The three-dimensional ordering fraction at even `K` (`q = 13, 34, 55`) is
  0.0882, 0.0920, 0.0926, below `1/10` and moving toward it; at odd `K`
  (`q = 5, 8, 21`) it is 0.306, 0.241, 0.138, above `1/10` and moving toward
  it.  The alternation is a layer-parity effect of the discrete diamond: at
  fixed `q` the receipt shows `k` even low and `k` odd high across
  `k = 1..K` (at `q = 55`: 0.0024, 0.224, 0.064, 0.130, 0.084, 0.115, 0.093
  for `k = 2..8`), with the amplitude shrinking in `k`.  The moving-tip
  diamond at `q = 55` gives 0.1057 with standard error 0.0015.
* The controls respond to dimension: the same read law and layer rule on the
  one- and two-dimensional golden populations give fractions within 0.002 of
  `1/2` and within 0.003 of `8/35` at `q = 55`.
* Normalized counts sit below the continuum volume by 3 to 33 percent.  The
  discrete cone is narrower than the continuum cone by up to the factor
  `1 - 2h_q/a_q` (`h_q/a_q` = 0.35, 0.44, 0.21, 0.27 at `q = 13, 21, 34, 55`;
  the `q` dependence of `r_q` is the three-gap structure of the golden
  orbit), and the finite-cone missing counts of the centre probes record the
  sites inside the metric cone that the graph distance does not reach
  (`q = 55`: 861, 3663, 7642, 4382, 340 sites at `k = 2..6`).  Narrower
  moving-tip diamonds show the larger deficits.
* The paper's finite bounds are not informative at `q <= 55` (bound over
  volume 13 to 150 for the vertical diamonds), and the count-clock enclosure
  of the finite clock proposition is `[0, undefined)` at every level because
  the count error of the reference diamond exceeds its count.  The raw clock
  `(N_K/N_K')^(1/4)` is within 5 percent of `K/K'` at `q = 34` (+4.4) and
  `q = 55` (-4.6); at `q = 5, 8, 13, 21` the deviations are -16, +4, -15 and
  -20 percent.
* Byte-level determinism: `--check` rebuilds the receipt identically on this
  machine (pool ordering is aggregated by start site; sample selection is
  seeded; derived floats are rounded to twelve significant digits).

## 4. Relation to the earlier "not similar" verdict

`data/causal_order/causet_likeness_receipt.json` reports
`NOT_SIMILAR_AT_CURRENT_CUTOFF__INTERVAL_ORDERING_FRACTIONS_OUTSIDE_EXPLORATORY_4D_BAND`.
Its field `source_binding.event_carrier_scope` names the population of that
verdict: `observer_instrumentation_history_over_source_state_snapshots`, and
its block `event_carrier_selection_controls` classifies that population as
`EVALUATED_FINITE_ORDER_NOT_COMPLETE_PHYSICAL_EVENT_CARRIER` with the
repair-only carrier as an antichain
(`REPAIR_ONLY_EVENT_CARRIER_IS_ANTICHAIN`, 24 events, zero versioned
provenance edges).  The theory does not name the observer-instrumentation log
as the event carrier.  The event carrier the theory declares
(`prop:golden-source-count-limit`, `cor:source-record-ordering-fraction`) is
the conservative source-record family run here.  The two receipts measure
different populations with the same instrument; this lane runs the declared
one.  That receipt and its producer are unchanged.

## 5. Claim boundary

Supplied, as in the paper: the population (product of golden orbits on the
source Gram metric), the complete-neighbour read law with waiting, the tick
`Delta_q = a_q/c`, and one counted event per site and layer.  Not selected by
native repair.  No physical clock or spacetime is identified.  Finite runs at
`q <= 89` do not demonstrate the asymptotic limit of the propositions; the
statistics are finite diagnostics of the supplied law, and the dimension
statistic is not an acceptance criterion.  Poisson sprinkling is not used.
The read/write hash-chained traces of the theory receipt are not executed
here (RER executes them at `q <= 13`).

## 6. Verification and pins

The verifier imports nothing from the producer.  It rebuilds `q = 5, 8, 13`
for all three families in the `sqrt 5` basis with dense all-pairs metric
decisions, scipy shortest paths and explicit event enumeration for every
interval (vertical `k = 1..K` and moving), recomputes the RER structural
digests, and compares them with the values embedded in the cross-check
blocks and, when the theory checkout is present, with the theory receipt
itself.  For `q = 21, 34, 55` it checks shell counts against probes, event
counts against shells, exact fractions against pair counts, the histogram
against the pair counts, stratified estimates and errors against their
strata rows, seeds, bounds, clocks, continuum references and an independent
clipped-volume quadrature.  Pins: producer, verifier, test file, and the four
theory files (`source_net_causet.py`, its receipt,
`SOURCE_NET_CAUSAL_LIMIT.tex`, `SourceNetCausalCone.lean`).

Timing of `--write` on the 10-core machine: `q <= 13` under 0.3 s per
level; `q = 21` 2.2 s; `q = 34` 27 s (19,047 exact vertical starts in 19 s,
3,143 moving starts in 4 s); `q = 55` 105 s (graph and digest 21 s, vertical
sample 31 s, moving sample 45 s) plus 5 s for its two-dimensional control;
2 min 20 s wall in total on an idle machine (4 min 50 s with three other
lanes running).  Verifier 2.3 s; tests 26 s idle, 56 s under load.

## 7. Work in progress

* Exact all-pairs counting through `q = 89` is in the committed receipt.  On a
  64-vCPU box with 62 workers the `q = 89` family took 10.7 h (graph and
  digest 254 s, vertical 22,837 s over 412,526 starts, moving 15,470 s over
  369,757 starts); the next level `q = 144` (2,985,984 sites, about 1.1e10
  neighbour entries) is untested.
* The neighbour table has 2.18e9 entries at `q = 89`; gather positions are
  int64 and every frontier expansion is sliced (`FRONTIER_CHUNK`), after a
  first `q = 89` attempt with int32 positions failed the lane's own cone
  check.
* A finite bound sharp enough to make the count-clock enclosure informative
  at reachable `q` is work in progress on the theory side; the receipt
  reports the formula values and the actual deviations.
