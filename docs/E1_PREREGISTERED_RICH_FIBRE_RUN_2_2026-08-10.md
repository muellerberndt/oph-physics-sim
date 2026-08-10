# E1 second preregistered rich-fibre bounded run (declared 2026-08-10, run id `e1_prereg2_64k_20260810`)

This document is the second preregistration for the E1 rich-fibre program
of issue #692 in `reverse-engineering-reality`. The first preregistration
(`docs/E1_PREREGISTERED_RICH_FIBRE_RUN_2026-08-10.md`, run
`e1_prereg_64k_20260810`) closed negatively under its fail-closed clause:
G1 passed (all four file-order observers realize a split fibre at
truncation 12), G4's byte replay passed, but G2 failed (one split fibre
per window) and G3 failed (the four first-12 windows overlap; union 37 of
48). That closure stands; the pilot run is not reused here.

This second preregistration declares a fresh-seed run with two design
changes calibrated on the closed pilot's diagnostics and declared before
execution: the support window widens to twenty nodes, and the observer
rule selects windows that are pairwise disjoint by construction. On the
pilot, greedy-disjoint selection at truncation 20 realizes split-fibre
counts [2, 4, 4, 4]; the fresh-seed gates below test whether that
structure is reproducible, not whether it can be found post hoc.

## 1. Preregistration integrity rules

1. Exactly one run is performed under this preregistration. There are no
   seed re-draws, no reruns on gate failure, and no post-hoc edits to the
   gates or to this document.
2. A failed gate is a valid negative result and will be reported as such
   on issue #692. Failure closes this preregistration negatively; it does
   not justify alternative truncations, fields, or selection rules.
3. The extraction script recomputes the class structure from the raw run
   artifacts with the pinned producer and fails hard on any disagreement
   with the run's own conditional-resampling receipt.
4. The observer selection uses support node identities only. No record or
   companion datum enters the selection.

## 2. Bounded run configuration (declared before execution)

- Base configuration (pinned, unmodified):
  `configs/e6_axiom_manifest_64k_dense_observers.yml`,
  sha256 `8dcdf70d55676b1576673445d41b188851d027fa21ce6f11df1a0abd539a42a8`.
- Derived run configuration (byte-identical to the base configuration
  except for the declared line `seed: 20260811` replacing `seed: 20260805`
  and the added line `run_id: e1_prereg2_64k_20260810`):
  `configs/local/e1_prereg2_64k_20260810.yml`,
  sha256 `46027e028c16ce74637cf9a960c754cefb8dae97a3f34e1a18c937177b5b671b`.
- Seed: **20260811**. Fresh for this preregistration: no run,
  configuration, or script in the repository references it at declaration
  time. It differs from the pilot seed 20260810 and from every campaign
  seed.
- Caps, inherited verbatim from the pinned base configuration: patch count
  **65536**, observer sample count **2048** with 96-node neighborhoods,
  128 dynamics cycles, 2048 repairs per cycle, 12-neighbor
  Fibonacci-sphere graph, group S3. Laptop-scale; declared wall-clock cap
  two hours (the pilot completed in about twenty-five minutes).
- Run id: `e1_prereg2_64k_20260810`; output directory `runs/`; raw bundle
  retained locally under the standing custody policy, artifact hashes
  pinned in the committed payload.
- Exact run command:

```
.venv/bin/python -m oph_fpe.cli run-oph-universe \
  --config configs/local/e1_prereg2_64k_20260810.yml \
  --out-dir runs \
  --run-id e1_prereg2_64k_20260810 \
  --seed 20260811 \
  --skip-visualizations
```

- Producer execution and fallback: identical to the first preregistration
  (pipeline writes the conditional-resampling receipt; single declared
  fallback invokes
  `write_conditional_resampling_realization` with `seed=20260811` once).
- Pinned producer code (read-only, untouched):
  - `oph_fpe/dynamics/conditional_resampling.py`,
    sha256 `3828608b860a3c5c3df223e71044bc73a5ae02c22c4849baee71a505557c8ed4`;
  - `oph_fpe/quotient/observable_normal_form.py`,
    sha256 `a5575469646880ac908955f6f970f7d95c86041ffe05dd4961585382c5c89959`.

## 3. Frozen extraction rule

- Extraction script (written and hashed before the run):
  `scripts/extract_e1_rich_fibre_payload2.py`,
  sha256 `001095bb4f40e6fcc124bb4ab4911e9331740b3e032e9297c32d933325f65bbb`.
- Exactly two declared changes relative to the closed first
  preregistration's rule: `support_truncation = 20`, and the observer rule
  is greedy-disjoint selection (scan `patch_observer` rows of
  `observer_views.jsonl` in file order; select a row exactly when its
  first-20 window is disjoint from every previously selected window; stop
  at four). Unchanged: `record_field = record_signature`,
  `companion_field = cumulative_repair_load`, the split-fibre rule, and
  the per-observer designated-block rule (first split fibre in truncated
  support position order).
- Exact extraction command:

```
.venv/bin/python scripts/extract_e1_rich_fibre_payload2.py
```

  writing `docs/E1_RICH_FIBRE_PAYLOAD_2.json` (schema
  `oph.sim.e1_rich_fibre_payload.v2`, cap 200000 bytes).

## 4. Preregistered gates

Evaluated exactly on the truncated windows of the four selected
observers; verdicts are recorded inside the payload.

- **G1**: at least three of the four observers realize at least one split
  fibre on their first-20 window.
- **G2**: at least two observers realize at least two split fibres each on
  their first-20 windows.
- **G3**: the four first-20 windows are pairwise disjoint as node sets
  (union size exactly 80). The selection rule guarantees this by
  construction; the gate independently checks the realized windows.
- **G4**: the payload bytes are hash-pinned in the campaign record, and an
  independent rerun of the extraction script reproduces them byte for
  byte before any Lean literal is written.
- **G5**: fail-closed. If any of G1 through G4 fails, the negative result
  is reported on issue #692 and E1 stays open; no synthetic fibre, no
  post-hoc rule change, no rerun.

## 5. What a passing run licenses

Identical to the first preregistration: enriched noncommutative blocks at
every observer with realized split fibres, region-separating receipts on
the pairwise-disjoint truncated supports, and the genuine-coverage net
over them, mirrored as exact literals into a Lean module of
`reverse-engineering-reality` in the style of
`QFT/SourceRegionalNet.lean`. Source-produced CP/CPTP instrument
provenance stays with E2; continuum causal and time-slice structure stays
with E3; no physical causality, clock, or prediction is claimed.
