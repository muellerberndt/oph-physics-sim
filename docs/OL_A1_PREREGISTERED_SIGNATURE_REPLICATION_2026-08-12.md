# OL-A1 preregistered Tier A signature replication (declared 2026-08-12, campaign id `ol_a1_tier_a_2026-08-12`)

This document is the preregistration for the OL-A1 Tier A signature
replication campaign required by V3 issue #737 in
`reverse-engineering-reality` (instrument register row INS-01, observation
ledger row OL-A1). It freezes the arms, the observables, the controls, and
the decision rule specified in
`plan/OL_A1_SIGNATURE_REPLICATION_SPEC.md` (oph-meta planning workspace)
before any seed is drawn and before any run is executed. The archived
baseline is the frozen four-row Einstein convergence ladder under
`data/einstein_convergence/` at seed 20260751.

Eigenvalue caution, carried by every sentence about this instrument: the
archived (1,3) threshold verdict rides on one eigenvalue at relative
magnitude about 3e-5 against a 1e-12 threshold; at a relative threshold of
1e-3 the robust inertia of the retained rungs is (1,2) with one degenerate
direction, matching the independently measured (1,2) receipt, and a fresh
seed can flip (1,3) to (2,2) with no change in physical content.

## 1. Preregistration integrity rules

1. Exactly one campaign is executed under this preregistration: fifteen
   source runs, one per (arm, replicate) cell. There are no seed re-draws,
   no reruns on any outcome, and no post-hoc edits to this document, to
   the frozen driver, or to the decision rule.
2. One fresh seed is drawn after this document and the driver are
   committed. Freshness means the integer appears nowhere in
   `oph-physics-sim` or `reverse-engineering-reality` at draw time,
   verified by grep. The seed and the five declared replicate ids are
   committed in the seed table
   `docs/OL_A1_SEED_TABLE_2026-08-12.json` together with this document's
   sha256. After that commit the seed is immutable no matter what.
3. FAILED and INCONCLUSIVE are reported with the same structure and
   prominence as REPLICATED. A FAILED verdict demotes the OL-A1 emergent
   rung with equal prominence.
4. A P0 nonconformance (Section 8) voids the campaign as an execution
   error, is reported as such, and authorizes no seed re-draw.
5. N = 5 replicates resolve seed fragility at the 1-in-5 level only, and
   every report of this campaign states so.

## 2. Arms and replicates (declared)

Three arms, each run at the one fresh seed with five declared replicate
ids `ola1.r1` through `ola1.r5` forwarded to the pinned source producer as
`replicate_id`, with this document's sha256 forwarded as
`preregistered_plan_sha256`:

- **A1** (scale row): carrier_count 16384, observer_count 128,
  observer_support_size 96.
- **A2** (scale row): carrier_count 65536, observer_count 256,
  observer_support_size 96.
- **C1** (matched support-density control): carrier_count 16384,
  observer_count 128, observer_support_size 6.

All other configuration fields equal the archived ladder configuration:
cycles 16, observer_samples 6, observer_cross_reads true,
snapshot_coverage spanning, geometry_transport held_out_flow. Fifteen
cells total, run sequentially in one process, arm-major in the order A1,
A2, C1, replicates in declared order.

## 3. Pinned code (read-only, untouched by this campaign)

- `oph_fpe/bulk/event_manifold_producer.py`,
  sha256 `a075829cac265284fc3faa205216023bf428f97ab7c04b4ded8b6594809e9482`;
- `oph_fpe/bulk/physical_h3_kms_source_capture.py`,
  sha256 `223cab93ad720579fde6bcc1e67dbe130875113fb2b022335996744200d1a874`;
- `scripts/einstein_convergence_ladder.py`,
  sha256 `56b99043daddb5f1456d5631b27e728064fb503f2c155968fac0784f6571f7ed`;
- frozen driver `scripts/ol_a1_signature_replication.py`,
  sha256 `6cd6b179e9f62e7d7f5f1177ec4cfa329a07c0279ef9e7aedc7aef2383e8acc1`.

The driver imports the estimator path read-only and reproduces the
archived ladder pipeline byte for byte: `_event_table`, the ladder's
`_pair_classes_capped` with pair cap 300000, the restricted spectral
embedding over touched carriers, `_event_chart`, and
`_fit_quadratic_form` with training parity 0 and inertia threshold 1e-12.

## 4. Declared observables

Per (arm, replicate) cell, all recorded in the per-run receipt:

- **O1 threshold inertia**: the (positive, negative) eigenvalue counts of
  the parity-0 quadratic-form fit at the pinned 1e-12 threshold, computed
  by the pinned estimator byte-identically to the archived rows.
- **O2 robust inertia**: with eigenvalues lambda_i of the same fit and
  scale = max|lambda_i|, the triple (positive, negative, degenerate) with
  positive = #{lambda_i > tau * scale}, negative =
  #{lambda_i < -tau * scale}, tau = 1e-3. Declared references: A1 (1,2)
  with one degenerate direction, A2 (1,2) with one degenerate direction,
  C1 (2,1) with one degenerate direction.
- **O3 degeneracy ratio**: min|lambda_i| / max|lambda_i|. Declared
  reference band [2e-5, 4e-5]; the archived values are 4.007e-5 (A1 rung)
  and 3.626e-5 (A2 rung). Recorded diagnostic; the archived A1 value sits
  at the band edge, so O3 does not gate the verdict.
- **O4 cone margin and rung ratio**: the parity-0 held-out cone margin per
  cell, and per replicate the margin-magnitude ratio
  |margin(A2)| / |margin(A1)|. Archived reference ratio 0.5725
  (|-3.219901300819808| / |-5.624222634635595|). Declared band
  **[0.35, 0.80]**.
- **O5 structural diagnostics**: event count, cross-observer ancestry
  edges, causal and spacelike pair totals, subsampling strides, the
  train/held parity composition of the kept pairs per class, and the
  held-out pair count. Recorded diagnostics.
- **O6 split-half concordance**: a second fit with training parity 1 on
  the same chart and pairs; concordance flags for threshold inertia and
  robust inertia between the parity-0 and parity-1 fits. Recorded
  diagnostic readout; it does not gate the verdict. The archived rows
  carry no parity-1 baseline, so gating on O6 would be uncalibrated.

O1, O3, O5, and O6 are recorded and reported with the same prominence as
the gating readouts; the realized O1 (1,3) rate per scale arm appears in
every verdict sentence together with the eigenvalue caution.

## 5. Declared controls

- **C-ANCESTRY** (ancestry-permutation null, zero extra source runs): per
  cell, permute the event reachability relation by a permutation drawn
  from `numpy.random.Generator(PCG64(s))` where s is the first eight
  bytes, big endian, of sha256 of the string
  `ol_a1_ancestry_null:<arm_id>:<replicate_id>`; recompute the capped pair
  classes on the permuted relation; refit the parity-0 quadratic form on
  the unchanged chart. The null **destroys** the cell's signature when the
  refit threshold inertia differs from the cell's measured O1 threshold
  inertia. Declared requirement: destroyed on at least 4 of 5 replicates
  on each scale arm.
- **C-SUPPORT** (matched support-density control via arm C1): with
  rate(X) = the fraction of replicates on arm X whose O2 robust inertia
  equals the scale reference (1,2) with one degenerate direction, and
  mean_cross(X) = the mean cross-observer edge count on arm X, the
  control **fires** when rate(C1) < rate(A1) and
  mean_cross(C1) <= 0.5 * mean_cross(A1). The control is **inverted**
  when rate(C1) >= rate(A1) and rate(A1) >= 0.8. The C1 robust reference
  (2,1) is recorded per replicate as a diagnostic.

## 6. Frozen decision rule

Evaluated by the frozen driver over the fifteen receipts. FAILED is
evaluated first, then REPLICATED, else INCONCLUSIVE.

**FAILED** if any of:

- F-a: on either scale arm, O2 robust inertia equals (1,2) with one
  degenerate direction on at most 2 of 5 replicates;
- F-b: the O4 margin-magnitude ratio lies in [0.35, 0.80] on at most 2 of
  5 replicates;
- F-c: on either scale arm, the C-ANCESTRY null destroys the measured
  threshold inertia on at most 3 of 5 replicates;
- F-d: C-SUPPORT is inverted.

**REPLICATED** if all of:

- R-a: on each scale arm, O2 robust inertia equals (1,2) with one
  degenerate direction on at least 4 of 5 replicates;
- R-b: the O4 margin-magnitude ratio lies in [0.35, 0.80] on at least 4
  of 5 replicates;
- R-c: on each scale arm, the C-ANCESTRY null destroys the measured
  threshold inertia on at least 4 of 5 replicates;
- R-d: C-SUPPORT fires.

**INCONCLUSIVE** otherwise.

A FAILED verdict demotes the OL-A1 emergent rung and is reported with the
same structure and prominence as REPLICATED. An INCONCLUSIVE verdict
leaves the OL-A1 emergent rung unpromoted and blocks any replication
claim. No verdict authorizes a seed re-draw, a rerun, or an alternative
analysis. An unfitted cell (estimator returns fitted false) counts as a
non-match for every gating readout it touches.

## 7. Execution plan (declared)

- Environment: the repository venv `.venv/bin/python`, single process for
  all fifteen cells, runtime versions recorded per cell and required
  identical. Thread pinning per `docs/EINSTEIN_BRANCH.md`:

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python scripts/ol_a1_signature_replication.py \
  --spec docs/OL_A1_SEED_TABLE_2026-08-12.json \
  --out data/ol_a1_replication
```

- The working tree is clean at run start and HEAD is recorded in the
  campaign summary.
- Outputs under `data/ol_a1_replication/`: fifteen per-run receipts
  `run_<arm>_<replicate>.json`, one `campaign_summary.json` with the
  decision-rule evaluation and the P0 conformance block, and one
  `manifest.json` with the sha256 of every file. Receipts carry the
  capture sha256 of each source run; raw capture arrays are recomputable
  from the seed determinism of the pinned producer and are retained only
  as hashes.
- Receipts and summary are committed regardless of the verdict.

## 8. P0 conformance block (declared)

The campaign summary records: clean tree at run start and the HEAD
commit; the sha256 match of this document against the seed table's
declared value; the sha256 match of the four pinned files of Section 3
against the seed table's declared values; the thread-pinning environment;
runtime-version uniformity across the fifteen cells; and the executed
cell count. All checks true gives run_status CONFORMANT. Any check false
gives run_status VOIDED_EXECUTION_ERROR: the campaign is void as an
execution error, the void is reported with the same prominence as any
verdict, and no seed re-draw is authorized.

## 9. Claim boundary

This campaign is a simulation instrument bound to the emergent rung of
observation-ledger row OL-A1 and to nothing else. A REPLICATED verdict
promotes only the statement that the archived signature pattern (robust
inertia (1,2) with one degenerate direction on the scale arms, margin
ratio inside the declared band, controls discriminating) reproduces under
one fresh seed at N = 5, which resolves seed fragility at the 1-in-5
level only. It claims no open-chart topology, no continuum limit, no
physical metric, and no laboratory prediction; the frozen-prediction
ladder is a separate surface and this instrument never enters it. The O1
threshold reading (1,3) stays under the eigenvalue caution stated at the
top of this document in every public sentence.
