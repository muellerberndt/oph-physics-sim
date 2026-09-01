# OL-A1 INS-02 factorial preregistration (design-frozen 2026-09-01, campaign id `ol_a1_ins02_factorial_2026-09-01`)

This document is the stage-S1 design freeze of instrument INS-02 (upstream
register `docs/INSTRUMENT_REGISTER_V3.md` in
`FloatingPragma/observer-patch-holography`), the OL-A1 factorial
carrier-count and absolute-support-size follow-up, written to
`docs/INSTRUMENT_PREREGISTRATION_TEMPLATE_V1.md` in this repository. Freeze
manifest: `docs/OL_A1_INS02_FREEZE_MANIFEST_2026-09-01.json`.

**Phase boundary, controlling this document.** V3 issue #737 states: "The
current factorial follow-up is design-only: no seed draw or execution is
authorized in this phase." Accordingly this freeze is stage S1: no seed
integer exists, no driver run, no pilot run, and no capture or result data
of any kind was generated in producing it. `execution_authorized` is false
in the manifest. Stages S2 (frozen driver) and S3 (seed commitment) and any
run, including a runtime-only cost pilot, require a later explicit
authorization on the owning lane.

**Eigenvalue caution, carried by every sentence about this instrument.** The
archived (1,3) threshold verdict rides on one eigenvalue at relative
magnitude about 3e-5 against a 1e-12 threshold; at a relative threshold of
1e-3 the robust inertia of the retained rungs is (1,2) with one degenerate
direction, and a fresh seed can flip (1,3) to (2,2) with no change in
physical content. The 2026-08-21 instrument-defect investigation
(`docs/OL_A1_INSTRUMENT_DEFECT_2026-08-21.md`) traced this fragility to
chart conditioning and repaired it with `standardize_chart`; the repair does
not by itself restore discriminating power against ancestry nulls. This
instrument exists to test, with the repaired estimator and quantitative
attribution gates, whether the signature reproduces and is attributable.

No claim in this document is tagged PROVEN: OL-A1 carries no machine-checked
theorem, and a preregistration is design, not evidence.

## 1. Identity and binding

- Instrument: INS-02. Ledger row: **OL-A1** ("Three space dimensions and one
  time direction"), the only observation-ledger row currently marked for the
  emergent rung with an open, owed status. Composition lane: upstream issue
  #728 (Spacetime) owns the target; lane #737 owns this instrument.
- Lineage predecessor: INS-01, verdict FAILED, campaign
  `ol_a1_tier_a_2026-08-12`, retained unchanged under
  `data/ol_a1_replication/`. **INS-01 remains the controlling completed
  verdict and OL-A1 remains owed.** This document changes no scientific
  classification.
- Declared ledger consequence: a REPLICATED verdict makes this instrument
  *eligible* for explicit selection by the upstream OL-A1 ledger-control
  lineage (consequence `attain_row`); absent that explicit upstream
  selection it moves nothing. A FAILED verdict leaves OL-A1 owed and is
  reported with equal prominence. An INCONCLUSIVE verdict promotes and
  demotes nothing.

## 2. Preregistration integrity rules

1. Exactly one campaign is executed under this preregistration: sixty
   source runs (six cells by ten replicates), plus the declared sham and
   synthetic control computations. No seed re-draws, no reruns on any
   outcome, no interim analysis, no optional stopping, and no post-hoc
   edits to this document, the frozen driver, or the decision rule. All
   sixty runs execute regardless of interim outcomes; the decision rule is
   evaluated only after every receipt is written.
2. FAILED and INCONCLUSIVE are reported with the same structure and
   prominence as REPLICATED. A VOIDED_EXECUTION_ERROR is reported with the
   same prominence as any verdict and authorizes no re-draw.
3. N = 10 replicates per cell resolve seed fragility at the 1-in-10 level
   per cell only, and every report of this campaign states so.
4. Any analysis not derivable from this document's text is outside the
   campaign, is labelled exploratory, and is never labelled validation.

## 3. Design: retained factorial cells (declared)

Two carrier counts crossed with three absolute support sizes, observer
count fixed, all at the archived ladder configuration otherwise:

| cell id | carrier_count | observer_support_size | observer_count |
| --- | --- | --- | --- |
| c16k_s48 | 16384 | 48 | 256 |
| c16k_s96 | 16384 | 96 | 256 |
| c16k_s192 | 16384 | 192 | 256 |
| c64k_s48 | 65536 | 48 | 256 |
| c64k_s96 | 65536 | 96 | 256 |
| c64k_s192 | 65536 | 192 | 256 |

Fixed for every cell (ASSUMED, equal to the archived ladder configuration
used by INS-01): cycles 16, observer_samples 6, observer_cross_reads true,
snapshot_coverage spanning, geometry_transport held_out_flow. Observer
count is fixed at 256 (the register's provisional value) to remove the
carrier-count/observer-count confound INS-01 carried (A1 ran 128 observers,
A2 ran 256). Replicate ids `ins02.r1` through `ins02.r10`.

The register's design envelope also names carrier counts 131,072 and
262,144. They are **not retained**: the owning issue bounds this program to
laptop scale, INS-01's receipts record no wall-clock cost (UNKNOWN runtime
at those scales), and a campaign that needs them requires the separate
larger-campaign authorization. Dropping them here is a design restriction
(ASSUMED), not a finding about them.

Design risk, stated openly (UNKNOWN): no committed run has exercised
observer_support_size 192 or support 48 at observer_count 256. Whether the
pinned producer accepts these cells without constraint violations must be
established at stage S2 by code inspection and existing unit-test machinery
only; if a retained cell is infeasible in the pinned producer, this
instrument is VOID at S2 and returns to S0 under a new campaign id. No
scientific run may be used to test feasibility.

## 4. Pinned code and environment

Pinned at repository commit `3bc1dd1b47d8367b661c64f1737a1d348d37e62a`
(read-only for this campaign):

- `oph_fpe/bulk/event_manifold_producer.py`,
  sha256 `02da8d0a419d092a4ece509bf1d8caeeb13f2d337a292847b859001e5782c6fa`;
- `oph_fpe/bulk/physical_h3_kms_source_capture.py`,
  sha256 `7e68714ab3e78ffd89b2aec2bb89e9f3c24766c105209bb50497f205eef4b8b5`;
- `scripts/einstein_convergence_ladder.py`,
  sha256 `56b99043daddb5f1456d5631b27e728064fb503f2c155968fac0784f6571f7ed`;
- `tests/test_event_chart_standardization.py`,
  sha256 `b4799e4fdab030a126e1124e8af64c15f73c03c74002a20eea669abac80bf34b`;
- context pin `docs/OL_A1_INSTRUMENT_DEFECT_2026-08-21.md`,
  sha256 `2bd136fa9ffce912021d76e4971436da270d932e41973d63f775ca63aa664262`.

Analysis path, fixed: the ladder pipeline exactly as INS-01 pinned it —
`_event_table`, `_pair_classes_capped` with pair cap 300000, the restricted
spectral embedding over touched carriers, `_event_chart`, and
`_fit_quadratic_form` with training parity 0 — with exactly one declared
difference: **`standardize=True`** on every fit in this campaign (the
repaired, conditioning-safe path; TESTED by the pinned
`tests/test_event_chart_standardization.py`). The flag value is pinned here
and may not be revisited after data.

The frozen driver does not exist yet. Stage S2 commits it with an addendum
manifest pinning its sha256; the driver must implement this document with
no semantic discretion, and any deviation voids the campaign to S0.
Environment at run time: single process, one recorded runtime environment,
thread pinning `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1`.

## 5. Declared observables

Per (cell, replicate), all recorded in the per-run receipt:

- **O1 threshold inertia** (recorded, non-gating): the (positive, negative)
  eigenvalue counts of the parity-0 standardized fit at the 1e-12 absolute
  threshold. Non-gating because the defect investigation showed threshold
  inertia rides on conditioning; it is reported in every verdict sentence
  together with the eigenvalue caution.
- **O2 robust inertia** (gating, gate **G-SIG**): with eigenvalues lambda_i
  of the parity-0 standardized fit and scale = max|lambda_i|, the triple
  (positive, negative, degenerate) at tau = 1e-3. G-SIG passes iff the
  triple equals **(1, 3, 0)**. This target is definitional for the row (one
  time direction, three space dimensions, every direction resolved), not a
  calibrated band. The only committed standardized-fit precedent is the
  2,048-carrier defect-investigation capture, which gave (1,3,0); as a
  quantitative reference at the retained scales that precedent is ASSUMED.
- **O3 resolution margin** (gating, gate **G-RES**): min_i |lambda_i| /
  max_i |lambda_i| of the same fit. G-RES passes iff the ratio is at least
  **0.01** (ASSUMED design threshold; the repaired 2,048-carrier fit gave
  0.075, the unrepaired archived fits gave about 1.3e-5 to 4e-5).
- **O4 held-out cone margin** (recorded, non-gating on its sign): the
  pinned estimator's `cone_margin` (the smallest correctly-signed held-out
  quadratic-form value; positive means the fitted cone separates causal
  from spacelike pairs on held-out pairs, and one wrongly-signed held-out
  pair makes it negative). The margin value and the positive-margin
  indicator are reported per run with the same prominence as gating
  observables. Sign is not gated because the committed archived margins are
  negative and gating on sign was never part of the register's design
  envelope; attribution is gated instead (O5).
- **O5 attribution contrast** (gating, gate **G-ATTR**): for each null
  family F in {C-REWIRE, C-DEPTH} (Section 6), M = 64 null draws are
  computed for the run. G-ATTR passes for family F iff the data
  `cone_margin` strictly exceeds at least 61 of the 64 null-family margins.
  Under an exchangeable null the data margin's rank among the 65 values is
  uniform, so the per-family false-pass probability is exactly 4/65
  (about 0.0615) with no distributional assumption beyond exchangeability
  (ASSUMED: exchangeability of the null construction). G-ATTR for the run
  passes iff it passes for **both** families. This is the
  margin-contrast statistic the defect report identified as the correct
  attribution test; the full 64-value margin vector per family is retained
  in the receipt.
- **O6 structural diagnostics** (recorded): event count, cross-observer
  ancestry edges, causal and spacelike pair totals, subsampling strides,
  train/held parity composition, held-out pair count, and the full fitted
  eigenvalue list of every fit in the run.
- **O7 factorial endpoints** (recorded, reported in the campaign summary):
  per cell and per gate, the pass count of 10 with Clopper-Pearson 95%
  intervals; the carrier-count main effect (difference in mean
  per-replicate composite pass rate, 65536 minus 16384 cells), the
  support-size trend across 48/96/192, and their interaction (difference of
  differences), each with Newcombe score intervals. These are reported
  endpoints; they gate nothing, and no endpoint estimate may erase a
  separately reported component outcome.

**Composite pass** for a (cell, replicate): G-SIG and G-RES and G-ATTR
(both families) all pass. An unfitted cell (estimator returns fitted false)
counts as a non-pass for every gate it touches.

## 6. Declared controls

Null-draw and control seeds derive from the campaign master seed (Section
8): stream j of family `<fam>` for run (cell, replicate) uses
`numpy.random.Generator(PCG64(s))` where s is the first eight bytes, big
endian, of sha256 of the string
`ol_a1_ins02:<fam>:<master_seed>:<cell_id>:<replicate_id>:<j>`.

- **C-REWIRE** (degree-preserving ancestry rewiring; null family for
  G-ATTR): double-edge swaps on the ancestry DAG — swap (a→b, c→d) to
  (a→d, c→b) — accepted only when the swap creates no cycle and preserves
  both target nodes' ancestry depth; attempted swaps per draw equal ten
  times the edge count. The rewiring preserves carrier count, in- and
  out-degree sequences, per-node support counts, depth values, and
  one-point record marginals, and breaks lineage correlations. After each
  draw: recompute reachability, recompute the capped pair classes with the
  pinned subsampling, refit parity-0 standardized on the unchanged chart,
  record `cone_margin`.
- **C-DEPTH** (depth-stratified lineage shuffle; null family for G-ATTR):
  permute node identities uniformly within each ancestry-depth stratum,
  preserving the population at every depth while breaking cross-depth
  family identity; then recompute pair classes and refit as above.
- **C-SHAM** (invariance sham; conformance authority, not a verdict gate):
  for each cell, on replicate `ins02.r1`'s capture, apply a derived
  permutation to event identifiers and rerun the entire analysis. Every
  declared observable (O1-O6, including the canonical-ordered eigenvalue
  lists and both null-margin vectors) must be exactly identical to the
  unpermuted run. Any mismatch is an implementation defect and a P0
  nonconformance: the campaign is VOIDED_EXECUTION_ERROR.
- **C-SYNTH** (sensitivity synthetic; validity authority, no promotion
  authority): for each cell, construct a synthetic event set with the same
  event count as replicate `ins02.r1`'s capture: a 4-column chart drawn
  from the derived PCG64 stream, pair labels assigned causal or spacelike
  by the sign of Q*(dx) for Q* = diag(+1, -1, -1, -1), keeping only pairs
  with |Q*(dx)| >= 0.1 (declared margin band, ASSUMED), and a synthetic
  ancestry DAG built from the causal pairs. Run the identical estimator and
  the identical G-SIG/G-RES/G-ATTR machinery (both null families, M = 64)
  on the synthetic. **Recovery gate G-SYNTH**: the synthetic passes G-SIG,
  G-RES, and G-ATTR in every cell. By construction the planted effect has
  known sign; failure to recover it means the instrument as implemented
  cannot detect ancestry-borne structure, and the verdict is capped at
  INCONCLUSIVE. A synthetic recovery promotes nothing.

Every declared control outcome appears in the campaign summary; an absent
control result is a P0 nonconformance.

## 7. Frozen decision rule

Evaluation order: P0 conformance (Section 10) first; then C-SHAM; then
G-SYNTH; then FAILED; then REPLICATED; else INCONCLUSIVE.

1. Any P0 check false, or any C-SHAM mismatch: **VOIDED_EXECUTION_ERROR**.
   No verdict, no re-draw, reported with full prominence.
2. G-SYNTH fails in any cell: verdict capped at **INCONCLUSIVE**
   (instrument insensitive), regardless of the data gates, with every
   component outcome still reported.
3. **FAILED** if either:
   - F-a: in any cell, at most 4 of 10 replicates achieve the composite
     pass; or
   - F-b: pooled across all sixty replicates, at most 30 achieve
     both-family G-ATTR.
4. **REPLICATED** if: in every cell, at least 8 of 10 replicates achieve
   the composite pass (and rules 1-3 did not apply).
5. **INCONCLUSIVE** otherwise.

No component pass overwrites a failed control or endpoint; no component
failure erases a separately reported component pass; the campaign summary
reports every gate count per cell.

**Replicate count from a prospective calculation** (all figures ASSUMED:
independence across replicates and cells, and the stated effect sizes; no
empirical calibration exists and none may be run in this phase). Declared
design point: a robust phenomenon gives per-replicate composite pass
probability at least 0.95; then P(a cell reaches 8 of 10) = 0.9885 and
P(REPLICATED across six cells) = 0.9885^6 ≈ 0.93. Under an exchangeable
(unattributed) null, per-replicate both-family G-ATTR pass probability is
at most 4/65 per family, so P(a cell reaches 8 of 10 composite) is below
1e-7 and false REPLICATED is negligible; F-b additionally fails the
campaign when attribution pools at chance. At the design point the
false-FAILED probability from F-a is about 6 x 2.7e-6. N = 10 is the
smallest replicate count in blocks of five whose all-cell power meets 0.93
at the 0.8N per-cell threshold (N = 5 gives 0.977^6 ≈ 0.87 at threshold 4;
retained N = 10 also supports the 1-in-10 fragility statement). No fixed
count is presumed sufficient beyond this stated calculation.

## 8. Seed protocol (declared; no seed exists at S1)

- One master seed is drawn only at stage S3, after execution is separately
  authorized. Freshness rule: the integer appears nowhere in
  `oph-physics-sim` or `observer-patch-holography` at draw time, verified
  by grep and recorded in the seed table.
- Per-run seed for (cell, replicate): the first eight bytes, big endian, of
  sha256 of `ol_a1_ins02:<master_seed>:<cell_id>:<replicate_id>`, as an
  unsigned integer. One committed master seed therefore determines every
  run, null draw, sham, and synthetic stream in the campaign.
- Execution order: ten replicate-major blocks `ins02.r1` .. `ins02.r10`;
  within each block the six cells run in an order drawn from PCG64 seeded
  by sha256 of `ol_a1_ins02:order:<master_seed>:<replicate_id>`.
- The seed table (`docs/OL_A1_INS02_SEED_TABLE_<date>.json`) is committed
  before any run and records: master seed, freshness evidence, this
  document's sha256, the S2 driver's sha256, and the manifest's sha256.
  After that commit the seed is immutable no matter what. Every receipt
  records the seed-table commit hash; a receipt whose seed does not
  reproduce from the committed table under the pinned derivation is a P0
  nonconformance.

## 9. Execution plan and cost boundary (declared; not authorized)

- Command (S3, verbatim except the date in the seed-table name):

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python scripts/ol_a1_ins02_factorial.py \
  --spec docs/OL_A1_INS02_SEED_TABLE_<date>.json \
  --out data/ol_a1_ins02_factorial
```

- Laptop cost cap: 24 hours wall-clock in the single recorded environment
  (ASSUMED cap; INS-01's receipts record no runtime, so the true cost is
  UNKNOWN). Before S3, one runtime-only cost pilot per carrier count is
  required: pilot mode must hard-suppress every scientific field — no
  signature, inertia, margin, or control outcome may be computed into any
  retained or displayed output — and pilot receipts (wall-clock, memory,
  environment only) are committed. Pilot seeds derive from the fixed
  string `ol_a1_ins02:pilot:<cell_id>` and never touch the campaign
  derivation. If the projected sixty-run cost exceeds the cap, S3 is
  blocked pending the separate larger-campaign authorization. **No pilot
  is authorized in the current phase either**: the phase forbids all
  execution.
- The working tree is clean at run start and HEAD is recorded in the
  campaign summary.

## 10. Receipts, manifest, and conformance block (declared)

Outputs under `data/ol_a1_ins02_factorial/`: sixty per-run receipts
`run_<cell_id>_<replicate_id>.json` (schema
`oph.ol-a1-ins02-factorial.receipt.v1`), six sham receipts
`sham_<cell_id>.json`, six synthetic receipts `synth_<cell_id>.json`, one
`campaign_summary.json` (schema `oph.ol-a1-ins02-factorial.summary.v1`)
with the decision-rule evaluation, every per-cell gate count, the O7
endpoints, and the P0 block, and one `manifest.json` with the sha256 of
every file. All are committed regardless of the verdict.

Reproducibility boundary, improving on INS-01's recorded gap: each receipt
retains the full fitted eigenvalue lists, both complete 64-value null
margin vectors, the held-out pair count and parity composition, and the
capture sha256. Raw chart matrices and pair-index arrays are written as
`.npz` files whose sha256 enters `manifest.json`; whether those `.npz`
files are committed to git or retained in the custody archive is decided at
S2 by size and recorded in the addendum manifest. Until raw matrices are
retained, validation of any verdict remains producer-based, and every
report of this campaign says so.

P0 conformance checks (all must be true for CONFORMANT; any false gives
VOIDED_EXECUTION_ERROR, no re-draw): clean tree at run start with HEAD
recorded; this document's sha256 matches the seed table's declared value;
the pinned-code sha256 values of Section 4 and the S2 driver hash match the
seed table; thread environment as pinned; runtime versions uniform across
all runs; executed receipt count exactly sixty plus six sham plus six
synthetic, no more, no fewer; every receipt's seed reproduces from the
committed table; C-SHAM exact invariance in all six cells.

## 11. Claim boundary

This campaign is a simulation instrument bound to the emergent rung of
observation-ledger row OL-A1 and to nothing else. A REPLICATED verdict
supports only the statement: *under the repaired standardized estimator,
the (1,3,0) robust signature reproduces under fresh seeds across the
retained factorial envelope (carrier 16384-65536, absolute support 48-192,
observer count 256) and is attributable to ancestry structure against the
two predeclared null families at rank-test size 4/65 per family, at N = 10
per cell,* which resolves seed fragility at the 1-in-10 level per cell
only — and that verdict moves the ledger only if the upstream
ledger-control lineage explicitly selects it. It claims no open-chart
topology, no continuum limit, no physical metric, no laboratory
prediction, and no empirical validation of any physical statement; the
frozen-prediction ladder is a separate surface and this instrument never
enters it. A FAILED verdict is bounded to this configuration envelope and
estimator: it is never promoted to a universal obstruction, exactly as the
passive rank-29 reading with its thirteen-dimensional kernel remains a
passive-history property and not a universal one. Every public sentence
about this instrument carries the eigenvalue caution stated at the top of
this document.

## 12. Fail-closed mapping (instance)

| Clause | Mechanical check in this instance |
| --- | --- |
| Post-hoc analysis labelled as validation | Analysis fixed in Sections 4-7 at S1; receipts carry this document's sha256; anything else is labelled exploratory (Section 2.4). |
| Instrument parameters revised after data | S1 text immutable; `standardize=True` and every gate constant pinned here; P0 hashes document, code, and driver at run start. |
| Silent reruns | One campaign per this document; sixty runs exactly, counted by P0; receipts committed regardless of outcome; seeds immutable after the seed-table commit; no interim analysis. |
| Controls omitted | C-REWIRE/C-DEPTH gate the verdict via G-ATTR; C-SHAM voids on mismatch; C-SYNTH caps at INCONCLUSIVE; an absent control result is a P0 nonconformance. |
| Passive rank defect promoted to universal obstruction | Claim boundary (Section 11) bounds every negative reading to this envelope and estimator. |
| Run not reproducible from committed artifacts | Hash-derived seeds from one committed master seed; pinned code, environment, and driver; P0 seed-reproduction check; null-margin vectors retained in receipts. |
