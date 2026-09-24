# Exploratory lane: record density acting back on the read law

Status: a proposed law scanned over its one free exponent. It is not the declared
architecture, nothing in it is derived, and no paper states it. It exists to answer one
question with numbers: if the records a site holds set its own scale factor, in the way the
FLRW fragment's count-measure identity `n = rho sigma^4 Delta v` would read locally, does the
record field on the golden population inflate, clump, or settle?

Module: `oph_exact/sourced_read_law.py`. Receipts: `data/exploratory/sourced_read_law_q21.json`,
`sourced_read_law_q34.json`. Tests: `tests/test_exploratory_sourced_read_law.py`.

## The law

On the golden population `s(b) = L (xi_b1, xi_b2, xi_b3)`, `xi_b = frac(b phi)`, with the read
radius `a = L / sqrt q` of the papers (float positions, periodic torus of side `L`):

- scale of a site: `sigma(s) = (n(s) / N0)^(g/4)`, where `n(s)` is the site's record count of the
  previous layer and `N0` the mean neighbour count of the fixed law (`g = 1` is the fragment's
  identity read backwards, `g = 0` is the declared law);
- physical read radius `sigma(s) a`; physical distance between two sites the mean of their
  scales times the comoving distance; so `(j+1, s)` reads `(j, t)` iff
  `|t - s| <= a * 2 sigma(s) / (sigma(s) + sigma(t))`, self read included;
- record production `n_{j+1}(s) = min(cap, sigma_j(s)^beta * reads_j(s))`, `beta = 1` when the
  rate also scales with the local tick, `beta = 0` when it does not; `cap` is an optional
  capacity per site and layer.

At `g = 0` every site reads its fixed neighbourhood and produces the fixed law's count; the test
suite checks this and checks that the windowed neighbour counts equal the exact receipt's
`minimum/maximum_neighbor_count_including_wait` and `undirected_spatial_edges` at q = 5 and 8.

## Linear theory

Around the uniform point `n = N0`, with `delta = n / N0 - 1` and `delta_bar` its ball average,
one round maps `delta -> (g/4) (beta delta + (3/2)(delta - delta_bar))`. Contrasts shorter than
the read radius grow by `lambda_short = (g/4)(beta + 3/2)` per round, contrasts much longer by
`lambda_long = beta g / 4`, and the uniform mode obeys `n -> N0 (n/N0)^(beta g / 4)` exactly.
Instability of sub-radius contrasts therefore begins at `g_c = 4 / (beta + 3/2)`: 1.6 for
`beta = 1`, 2.67 for `beta = 0`. The uniform mode alone never runs away below `beta g = 4`; the
runaway seen above the threshold is contrast growth feeding the mean through the convexity of
the production law.

## Measured (q = 21 and q = 34, `beta = 1`)

Tangent response: the amplitude growth per round of a 5 percent perturbation of the relaxed
native field, from the difference of two trajectories, binned at scales from 2.3 to 0.29 read
radii (q = 21) and 2.9 to 0.36 (q = 34). Fixed-point amplification: the relaxed contrast of the
native quasi-crystal density variation over its fixed-law value, `1 / (1 - lambda_eff)` for a
stable point.

| g | predicted `lambda_short` | measured growth per round, q = 21 | q = 34 | outcome |
| --- | --- | --- | --- | --- |
| 0.5 | 0.31 | 0.44 to 0.52 | 0.59 to 0.68 | decays |
| 1 (fragment identity) | 0.625 | 0.67 to 0.79 | 0.83 to 0.87 | decays; native contrast amplified 1.7 (q = 21) and 2.7 (q = 34) then frozen |
| 1.5 | 0.94 | 0.93 to 1.12 | 0.98 to 1.02 | marginal |
| 2 | 1.25 | 1.21 to 1.32 | 1.12 to 1.19 | contrasts grow to order one |
| 3 | 1.88 | runaway | runaway | mean scale 55 (q = 21), 36 (q = 34) after 16 to 40 rounds |

The measured decay rates below threshold are upper bounds: the read counts are integers, so a
perturbation injects threshold-crossing noise each round and the measured ratio is pulled
toward one where the signal decays. The fixed-point amplification at q = 34 and `g = 1`,
2.7, corresponds to `lambda_eff = 0.63`, the linear value.

With `beta = 0` the same scan reads 0.06 to 0.09 (g = 0.5), 0.61 to 0.80 (g = 1), 0.78 to 0.90
(g = 2) and 1.09 to 1.20 (g = 3) at q = 21, against predictions 0.19, 0.375, 0.75 and 1.125.

## Runaway and capacity (q = 21, `g = 3`, `beta = 1`, native start)

| capacity | mean scale by round 0, 4, 8, 12, 20, 40 | sites at capacity, round 40 | frozen contrast sd |
| --- | --- | --- | --- | 
| none | 1, 1.02, 3.1, 20, 49, 55 | | 1.7 |
| 4 N0 | 1, 1.02, 1.66, 2.22, 2.44, 2.50 | 0.84 | 0.41 |
| 16 N0 | 1, 1.02, 2.5, 5.2, 6.2, 6.4 | 0.72 | 0.57 |

The uncapped history is exponential between rounds 4 and 12 and then saturates because the
candidate pair list ends at `2a`; radii above that are truncated by construction and the
receipt reports the fraction of pairs where this ceiling binds. With a capacity the same law
gives an exponential phase, an exit when most sites reach capacity, and a frozen structured
record field; the number of e-folds is set by the capacity.

## What this shows and what it does not

- With the fragment's identity (`g = 1`) the record field is stable: no inflation, no
  clumping, a bounded amplification of the population's own density variation. The size of the
  population does not change this; the law is a contraction there.
- Records acting on geometry produce exponential record production and structure only if the
  scale responds to the record count more strongly than the fourth root, by a factor of at
  least 1.6 in the exponent, or through a different coupling. That number is a target for a
  derivation, not a result about the architecture.
- The population is only `sqrt q` read radii wide, so super-radius modes (`lambda_long`) are not
  resolved below q = 144.
- Float geometry on a torus; the exact windowed lane is `oph_exact/source_net.py`.

## Reproduction

    python3 -m oph_exact.sourced_read_law --q 21 --rounds 16 --couplings 0.5 1 1.5 2 3 --betas 1 0 --out OUT.json
    python3 -m pytest -q tests/test_exploratory_sourced_read_law.py
