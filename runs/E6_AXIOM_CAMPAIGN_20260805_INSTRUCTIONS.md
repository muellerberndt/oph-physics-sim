# E6 axiom-manifest campaign: frontend packaging instructions

Two dense-observer runs on the axiom-manifest lane, packaged for the
visualization frontend. Each ZIP is self-contained, follows the layout
described by its embedded `README_FOR_WEB_CODING_AGENT.md`, and stays
under the 256 MB upload limit.

| Bundle | Screen | Observers | Size |
| --- | --- | --- | --- |
| `e6_16k_dense_axiom_20260805_visualizer.zip` | 16,384 patches | 1,024 (4x density) | 73.4 MB |
| `e6_64k_dense_axiom_20260805_visualizer.zip` | 65,536 patches | 2,048 (2x density) | 117.0 MB |

Config sources: `configs/e6_axiom_manifest_16k_dense_observers.yml` and
`configs/e6_axiom_manifest_64k_dense_observers.yml` (both derived from
the e4 observer-chart lane; seed 20260805).

## What is new in these bundles

Four artifacts appear here for the first time and deserve first-class
visualization treatment. All live under `run_reports/` inside the ZIP.

### 1. `axiom_manifest.json` - the three axioms, verbatim

Carries the canonical A1/A2/A3 statements copied verbatim from the
theory repository's machine registry (pinned by commit and content
hash in `canonical_source`), plus a `simulator_realizations` block
mapping each axiom to the concrete simulator structures realizing a
finite fragment of it in this run.

Suggested panel: a three-column "Axioms" view. For each axiom show the
`title`, the `informal` statement, and the realization rows (patch
count, ports per patch, observer counts, receipt paths). Keep the
`realization_status` string visible; it states honestly which fragment
of the axiom the run instantiates. Do not paraphrase the axiom text.

### 2. `conditional_resampling_realization_receipt.json` - the A3 event

The exact conditional-resampling kernel on observation fibers: the
committed record class of each patch is the protected datum, the
companion field (named in `companion.label`) is resampled from the
pinned reference law restricted to the record fiber. The kernel is
verified exactly over the rationals (recognizer R1-R3, idempotence,
stationarity, chi-squared contraction), then an empirical trajectory is
driven from a deliberately displaced start.

The 16k run measures a one-sweep collapse:

```
displaced start  chi^2 = 42.97
after sweep 1    chi^2 = 0.0135   (sampling-noise floor)
sweeps 2..7      chi^2 ~ 0.013-0.014 (flat)
```

The kernel is idempotent, so one sweep reaches the fiber reference law
and later sweeps only re-sample the noise floor. The protected record
never changes (`protected_record.unchanged_by_resampling`).

Suggested visualization: an animated two-layer screen view.
Layer 1 (static): record classes painted on the sphere from
`data/freezeout_fields.npz` field `record_signature` (this is the
protected structure). Layer 2 (animated): the companion field churning
under resampling, with the chi-squared trace as a side plot collapsing
onto the floor. The receipt only records per-sweep chi-squared values;
if you animate frames between sweeps, generate them by interpolation
and label them "interpolated between measured sweeps". The measured
rows are in `empirical_realization.sweeps`.

### 3. `theorem_core_receipts.json` (`lyapunov` block) - drive vs cooling

The chained Lyapunov receipt distinguishes the externally driven phase
from autonomous cooling. In the 16k run the observer-readback drive is
active for cycles 0-72 and the receipt reports it faithfully:

- `driven_trajectory: true`, 69 cross-cycle injections, largest
  injection 172 broken edges (`cross_cycle_injections` rows carry
  `index`, `previous_phi`, `phi_before`, `injection_delta`).
- Zero within-cycle violations: every repair step descends.
- `final_phi: 0.0` - after the drive stops, the screen cools to full
  repair.

Suggested visualization: the phi trajectory over cycles (from
`run_reports/mismatch_trace.csv`) with injection markers at the driven
cycles, the drive window shaded, and the receipt verdict displayed as
stated: the trajectory is driven, within-cycle descent is clean, and
the system reaches phi = 0 once autonomous. Do not present the driven
phase as spontaneous fluctuation.

### 4. `observer_chart_object_h3_report.json` - objects vs control

16k dense-observer result: 67 of 67 chart objects localize in the H3
chart (97% off-boundary) while shuffled-incidence controls localize
only ~6% (p90 = 6.5 objects). Suggested visualization: the existing
object-chart scene plus a small "control separation" inset showing the
real-vs-shuffled localization counts side by side.

## Cross-run findings

| Measurement | 16k dense | 64k dense |
| --- | --- | --- |
| A3 chi-squared, displaced start | 42.97 | 60.26 |
| A3 chi-squared after one sweep | 0.0135 | 0.0034 |
| Lyapunov cross-cycle injections | 69 (max 172) | 69 (max 434) |
| Final phi after drive stops | 0.0 | 0.0 |
| H3 objects localized / total | 67 / 67 | 88 / 88 |
| Shuffled-control localized | 4 | 4 |
| H3 response candidate gate | pass | fail |
| Median observer overlap Jaccard | 0.1098 | 0.1163 |
| Committed fraction at freezeout | 1.0 | 1.0 |

Reading of the table: the one-sweep A3 collapse reproduces at both
sizes and its residual floor drops fourfold at fourfold patch count,
which is the expected sampling-noise scaling for an integer-count
empirical law; the drive injection count is schedule-determined
(identical at both sizes) while its magnitude scales with the per-cycle
edge budget; object localization against shuffled controls reproduces
at both sizes. The 64k H3 response-candidate gate fails its fixed
control margin (median fit improvement 0.0015 against control 0.0006,
gap below the 0.01 margin) and the summary reports it as recorded; the
object-chart panels for the 64k bundle must show the gate verdict
alongside the localization counts.

## Mapping onto the deployed frontend

The deployed frontend (simulation.floatingpragma.io) is a ten-chapter
cinematic build anchored on a 16k run, with the provenance vocabulary
"measured / computed / interpolated / synthetic / frozen". The e6 16k
bundle is a direct refresh of that anchor with a real settling
trajectory, and the new artifacts slot into the existing chapters:

- Axiom panel (`axiom_manifest.json`): fits the data-format /
  self-reading-loop chapter as the opening statement of what the run
  instantiates. Provenance tag: frozen (verbatim pinned text).
- A3 resampling event (`conditional_resampling_realization_receipt.json`):
  fits the quotient-overlap / confluence-repair chapters. Sweep rows are
  measured; inter-sweep animation frames are interpolated.
- Drive-versus-cooling Lyapunov story (`theorem_core_receipts.json` +
  `mismatch_trace.csv`): fits the confluence repair-transaction chapter.
  Trajectory and injections are measured; the receipt verdict is
  computed.
- Object-versus-control separation
  (`observer_chart_object_h3_report.json`): fits the H3 camera chapter
  as a measured inset.

## Run-accuracy rules

- `run_reports/simulation_assumption_manifest.json` separates computed
  receipts from assumed visualization bridges. Anything listed under
  the assumption profile (S2 embedding, H3 camera, dS4 slicing, CMB
  transfer, defect-matter rendering) is scene dressing and must stay
  labeled as assumed in the UI, exactly as the existing frontend does.
- Interpolated or estimated frames are welcome (smooth camera paths,
  inter-sweep animation, field smoothing) as long as every interpolated
  layer carries an "interpolated" tag and the measured checkpoints
  remain accessible.
- Receipt verdicts must be displayed as recorded, including negative
  ones (`final_receipts` in `AUTO_THEOREM_UNIVERSE_SUMMARY.json`: the
  3D-bulk and Lorentz-contract gates are false at these compact sizes;
  that is the honest state of the run).
- The per-run `docs/VISUALIZATION_INSTRUCTIONS.md` and
  `docs/WEB_CODING_AGENT_VISUALIZATION_BRIEF.md` inside each ZIP remain
  the authoritative scene-by-scene guide; this file only adds the new
  axiom-lane artifacts and campaign findings.

## Rebuild commands

```
.venv/bin/oph-fpe run-oph-universe \
  --config configs/e6_axiom_manifest_16k_dense_observers.yml \
  --out-dir runs --run-id e6_16k_dense_axiom_20260805 \
  --max-screen-points 6000 --max-observers 512 --max-h3-objects 512

.venv/bin/python scripts/build_visualizer_zip.py \
  runs/e6_16k_dense_axiom_20260805 \
  runs/e6_16k_dense_axiom_20260805_visualizer.zip .
```

(The 64k lane uses the 64k config, run id `e6_64k_dense_axiom_20260805`,
and `--max-screen-points 5000`.)
