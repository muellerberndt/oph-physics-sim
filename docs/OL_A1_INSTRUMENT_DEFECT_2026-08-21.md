# OL-A1 instrument defect: event-chart conditioning

Status: instrument defect found and repaired. The 2026-08-12 Tier A campaign
is **retained unchanged** in `data/ol_a1_replication/` and is superseded as an
instrument reading, not withdrawn as a record.

## The defect

`_event_chart` builds four coordinates per semantic event: one integer
ancestry-depth coordinate and three spectral-embedding means. Their numerical
spreads differ by about three orders of magnitude. Measured on a 2,048-carrier
capture (seed 20260812):

| coordinate | standard deviation | range |
|---|---|---|
| ancestry depth | 29.11 | 119 |
| spectral 1 | 0.046 | 0.614 |
| spectral 2 | 0.043 | 0.632 |
| spectral 3 | 0.049 | 0.628 |

`_fit_quadratic_form` regressed on quadratic monomials of the raw coordinate
differences, so the depth direction dominated the design matrix and the
spectral directions collapsed. The fitted eigenvalues on that capture were

```
[-2.019e+01, -9.031e+00, -2.698e-04, 2.312e-01]
```

with relative magnitudes `[1.0, 0.447, 1.34e-05, 0.011]`. The reported
signature therefore turned on whether a direction at about `1e-5` relative
magnitude fell above or below the inertia cut. That is the seed fragility and
the threshold sensitivity recorded in the archived campaign caution.

## The repair

`standardize_chart` centres each coordinate and scales it to unit spread, and
`_fit_quadratic_form` takes a `standardize` flag. On the same capture the
repaired fit gives relative eigenvalues `[1.0, 0.673, 0.330, 0.075]`: every
direction is resolved, the inertia is `[1, 3, 0]` rather than the fragile
`[1, 2, 1]`, and the signature is invariant under per-coordinate rescaling.

The flag defaults to off, so every previously committed receipt replays byte
for byte. `tests/test_event_chart_standardization.py` pins the repair, the
rescaling invariance, and the unchanged default path.

## What the repair does not fix

The repair removes the conditioning artifact. It does **not** give the
measurement discriminating power.

Against the instrument's own ancestry-permutation null (permute reachability,
recompute the causal/spacelike classes, refit on the same chart) the null
still reproduces the data inertia in 4 of 5 permutations, both before and
after standardization, and the held-out cone margin is unchanged at -4.549
and remains negative.

The reason is structural: the inertia is a property of the chart's coordinate
covariance, which permuting ancestry labels does not change. Eigenvalue sign
counts are the wrong statistic for testing whether a fitted signature is
attributable to ancestry structure; a margin or classification-accuracy
contrast between data and null would be the right one. Designing that
statistic is open work, not a configuration change.

Consequently a repaired rerun of the Tier A campaign would return a better
conditioned signature and still fail the attribution criterion. No rerun was
performed on that basis.

## Provenance

- Archived campaign: `data/ol_a1_replication/`, campaign id
  `ol_a1_tier_a_2026-08-12`, verdict FAILED, retained.
- Preregistration: `docs/OL_A1_PREREGISTERED_SIGNATURE_REPLICATION_2026-08-12.md`,
  unchanged.
- The OL-A1 code path was unchanged between the campaign run and this
  investigation: the capture module last changed 2026-07-27, the icosahedral
  core 2026-07-20, and the estimator ladder 2026-07-29. The 2026-08-20
  icosahedral carrier work touched `core/array_geometry.py` and the EM and
  quantum-phase modules, none of which this path imports.
