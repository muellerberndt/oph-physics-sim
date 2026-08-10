# E1 preregistered rich-fibre bounded run (declared 2026-08-10, run id `e1_prereg_64k_20260810`)

This document is the preregistration for the bounded simulator source run
declared on issue #692 (E1) of `reverse-engineering-reality` (comment of
2026-08-10). The pinned truncation-8 payload realizes split fibres at only
one of four observers, and the Lean module `QFT/SourceRegionalNet.lean`
proves the collapse at the other three instead of repairing it. This run
tests whether a wider truncation window on a fresh 64k source run realizes
split fibres at more regions. Every declaration below is written and hashed
before the run command is issued.

## 1. Preregistration integrity rules

1. Exactly one run is performed under this preregistration. There are no
   seed re-draws, no reruns on gate failure, and no post-hoc edits to the
   gates or to this document.
2. A failed gate is a valid negative result and will be reported as such on
   issue #692. Failure closes this preregistration negatively; it does not
   justify alternative truncations, alternative record or companion fields,
   alternative observer selections, or synthetic fibre data. The Lean net
   over the truncation-8 payload stands unchanged in either outcome.
3. The extraction script recomputes the class structure from the raw run
   artifacts with the pinned producer and fails hard on any disagreement
   with the run's own conditional-resampling receipt.

## 2. Bounded run configuration (declared before execution)

- Base configuration (pinned, unmodified):
  `configs/e6_axiom_manifest_64k_dense_observers.yml`,
  sha256 `8dcdf70d55676b1576673445d41b188851d027fa21ce6f11df1a0abd539a42a8`.
- Derived run configuration (byte-identical to the base configuration
  except for the declared line `seed: 20260810` replacing `seed: 20260805`
  and the added line `run_id: e1_prereg_64k_20260810`):
  `configs/local/e1_prereg_64k_20260810.yml`,
  sha256 `8e6c65e706ba3f6b35c5fb1ae7269aac0920c148c643101072df31feb84307ed`.
- Seed: **20260810**. This seed is fresh for this preregistration: no run,
  configuration, or script in the repository references it at declaration
  time.
- Caps, inherited verbatim from the pinned base configuration: patch count
  **65536** (screen cap), observer sample count **2048** with 96-node
  neighborhoods, 128 dynamics cycles, 2048 repairs per cycle, 12-neighbor
  Fibonacci-sphere graph, group S3. Laptop-scale; declared wall-clock cap
  two hours (the pinned 20260805 base run of this configuration completed
  in about seventeen minutes).
- Run id: `e1_prereg_64k_20260810`; output directory `runs/`. The raw run
  bundle is retained locally under the standing custody policy (`runs/` is
  not committed); its artifact hashes enter the committed payload.
- Exact run command (visualization export is skipped; it feeds no gate):

```
.venv/bin/python -m oph_fpe.cli run-oph-universe \
  --config configs/local/e1_prereg_64k_20260810.yml \
  --out-dir runs \
  --run-id e1_prereg_64k_20260810 \
  --seed 20260810 \
  --skip-visualizations
```

- Producer execution: `run_oph_universe_pipeline` invokes
  `write_conditional_resampling_realization` on the run directory as part
  of the pipeline. If the pipeline terminates after the base run without
  writing the receipt, the single declared fallback is to invoke
  `oph_fpe.dynamics.conditional_resampling.write_conditional_resampling_realization`
  on the run directory with `seed=20260810` and default sweep count, once.
- Pinned producer code (read-only, untouched by this campaign):
  - `oph_fpe/dynamics/conditional_resampling.py`,
    sha256 `3828608b860a3c5c3df223e71044bc73a5ae02c22c4849baee71a505557c8ed4`;
  - recognizer `oph_fpe/quotient/observable_normal_form.py`,
    sha256 `a5575469646880ac908955f6f970f7d95c86041ffe05dd4961585382c5c89959`.

## 3. Frozen extraction rule

- Extraction script (written and hashed before the run):
  `scripts/extract_e1_rich_fibre_payload.py`,
  sha256 `dbf0a2248a1941a40522149866b5b0cd9bf97a590e301bd84d1e4d355bc2a969`.
- The rule is the pinned truncation-8 rule of
  `scripts/extract_e1_regional_payload.py` (sha256
  `46be5e9d8deea73001383391bcc1ccfb18aa2e1ed04e91eedcf00ac1c656d6f9`)
  with exactly one declared change: `support_truncation = 12`.
  Unchanged: `record_field = record_signature`,
  `companion_field = cumulative_repair_load`; the observer rule (the first
  four `patch_observer` rows of `observer_views.jsonl` in file order); the
  split-fibre rule (a record class of a truncated support is a split fibre
  when it carries at least two distinct companion classes there); and the
  designated-block rule, generalized per observer (for every observer with
  a split fibre, the first split fibre in truncated support position
  order).
- Exact extraction command:

```
.venv/bin/python scripts/extract_e1_rich_fibre_payload.py
```

  writing `docs/E1_RICH_FIBRE_PAYLOAD.json` (schema
  `oph.sim.e1_rich_fibre_payload.v1`, cap 200000 bytes).

## 4. Preregistered gates

Evaluated exactly on the truncated windows of the four selected observers;
verdicts are recorded inside the payload.

- **G1**: at least three of the four observers realize at least one split
  fibre on their first-12 support window.
- **G2**: at least two observers realize at least two split fibres each on
  their first-12 windows.
- **G3**: the four first-12 windows are pairwise disjoint as node sets
  (union size exactly 48).
- **G4**: the payload bytes are hash-pinned in the campaign record, and an
  independent rerun of the extraction script reproduces them byte for
  byte before any Lean literal is written.
- **G5**: fail-closed. If any of G1 through G4 fails, the negative result
  is reported on issue #692 and E1 stays open; no synthetic fibre, no
  post-hoc rule change, no rerun.

## 5. What a passing run licenses

Enriched noncommutative blocks at every observer with realized split
fibres, region-separating receipts on the disjoint truncated supports, and
the genuine-coverage net over them, mirrored as exact literals into a Lean
module of `reverse-engineering-reality` in the style of
`QFT/SourceRegionalNet.lean`. This is the remaining bounded E1 closure
surface. Source-produced CP/CPTP instrument provenance stays with E2;
continuum causal and time-slice structure stays with E3; no physical
causality, clock, or prediction is claimed.
