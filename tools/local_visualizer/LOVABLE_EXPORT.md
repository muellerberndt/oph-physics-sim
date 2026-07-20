# Lovable Deko visualizer directory export

`export.py` turns either a full universe-timeline payload or a standalone
`screenA5Ladder` payload into a portable, auditable directory for the public
Lovable Deko visualizer agent. It does not create a ZIP and does not modify the
source payload.

```bash
cd oph-physics-sim
python3 -m tools.local_visualizer.export \
  --payload ../survival-proof-4/outputs/architecture/screen_a5_ladder_16k_force_all_demo.json \
  --public-anchor ../survival-proof-4/outputs/run_16k/visualization_anchor.json \
  --out-dir ../survival-proof-4/outputs/lovable_deko_16k_handoff \
  --max-sidecar-bytes 240000000
```

Do not describe the frozen 16k attempt as a settling run: it refused before
RNG, so it has no measured or run-anchored settling trajectory. The exporter
supports deterministic synthetic/interpolated visual samples, but it keeps that
provenance explicit.

An adjacent `visualization_export_manifest.json` is discovered automatically.
Use `--sidecar-manifest` and `--sidecar-root` to select it explicitly. Only the
recognized `oph_universe_visualization_sidecars_v1` schema is accepted. The
exporter confines every source to that root, rejects symlinks and traversal,
omits sensitive or unsupported files, scans structured sidecars for recognized
credential keys/private paths, verifies declared byte counts and SHA-256 hashes,
and stops including files at the configured aggregate byte cap. Every declared
file appears in `export_manifest.json` as included or omitted with a controlled
reason. The exporter does not claim general secret detection for arbitrary
values under benign field names.

The output contains:

- `contracts/public_cinematic_story.json`: the public app's gate-free primary
  scene/story contract;
- `payload/visualization_payload.json`: canonical redacted payload;
- `payload/summary.json`: census plus concise view summaries;
- `contracts/screen_a5_ladder.json`: the detailed technical ladder;
- `contracts/local_carrier_prototype.json`: exact 12/30/20 carrier geometry;
- `contracts/visualization_views.json`: technical view contracts;
- `sidecars/`: safe, hash-verified, byte-capped sidecar selection;
- `documentation/SCREEN_A5_SM_VISUALIZER_AGENT_BRIEF.md`;
- `VISUALIZER_AGENT_INSTRUCTIONS.md`; and
- `export_manifest.json`: hashes, byte counts, media types, full/included/omitted
  census, redaction ledger, presentation profile, and immutable claim boundary.

Before describing contracts as exact, export validation fails closed unless it
finds the full 12-port/30-edge/20-face carrier, six fixed-point-free antipodal
pairs, 60 unique edge-preserving A5 actions, exact `1 + 3 + 3-prime + 5`
sectors, and the canonical 18-stage IDs and edge sequence.

## Public Lovable Deko presentation

The main public UI consumes only `public_cinematic_story.json`. Raw receipts,
gate state, and detailed manifests may appear in an optional collapsed
provenance drawer, but the primary UX must have no PASS/OPEN table, gate matrix,
force toggle, or blocker dashboard. Keep the exact lowercase disclosure
`illustrative reconstruction` subtly visible at all times.

The required sequence is:

1. deterministic seeded disorder/random initial port states across a huge
   visible federation of exact icosahedral compute elements;
2. provenance-tagged light/readback iterations inside carriers and across the
   actual exported seam graph, then mismatch repair and normal-form settling;
3. exact zoom into one 12-port/30-edge/20-face carrier and its local processing;
4. browse/animate all 60 A5 actions and resolve `1 + 3 + 3-prime + 5` sectors;
5. narrative A5-to-SM ladder, preserving the Q2 alternative fork;
6. after the emergence transition only, ID-joined particle families,
   worldlines, interactions, composites/proton imagery, and atoms;
7. H3 event geometry, gravity-response imagery, cosmology, and sky; and
8. an observer-frame finale with modular clock, visible particle/atom family,
   proton/composite matter, and gravity.

Never open on a pre-solved/prepopulated lattice. Prefer actual frozen 16k
records wherever they exist. When an in-between sample is missing, interpolate
only between adjacent samples of the same actor. When there are no valid
parents, deterministic model-based synthesis is allowed for visual continuity,
but it must not begin before emergence and must remain explicitly synthetic.

Every record carries `visualizationProvenance.status` from exactly:

```text
measured
computed
interpolated
synthetic
frozen
```

Interpolated records retain stable IDs, parent IDs, weights, and method.
Synthetic records retain stable IDs, generator ID/version, seed, deterministic
index, and source refs. Frozen records retain their source target and
post-exposure boundary. Never infer `measured`.

## Local analyzer remains separate

For receipt/gate inspection, run the local analyzer and open:

```text
http://127.0.0.1:8765/?mode=receipt
http://127.0.0.1:8765/?mode=demo
```

`DEMO_ASSUMPTION` toggles are local renderer tests. They never change the
physical snapshot, authorize scale, or create target ancestry. Do not reproduce
that local technical control surface in the primary public Lovable Deko app.

If present, consume `screenA5Ladder.physicalH3KmsDemoOverlay` as the display-only
P0--P8 bridge. Use `stageNodes[*].displayData` only when
`demoNudgeApplied=true`, retain each `fieldProvenance` row, and verify
`physicalSnapshotDigestBefore == physicalSnapshotDigestAfter`. For the public
cinematic UI, translate those rows into motion and short explanatory captions;
do not show them as gates or receipts. Never copy overlay values into the
embedded `physicalSnapshot`, a run report, promotion state, retirement decision,
or scale-control field.
