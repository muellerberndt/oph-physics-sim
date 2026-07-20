# OPH local visualizer

This is a dependency-free, read-only browser instrument for OPH visualization
payloads. It uses Python's standard-library HTTP server and bundled HTML/CSS/
Canvas JavaScript. It makes no network or CDN requests.

It accepts either:

- a full `oph_universe_timeline_visualization_payload_v1` payload; or
- a standalone `oph.screen-a5-visualization-ladder/1.0.0` artifact, including
  `survival-proof-4/outputs/architecture/screen_a5_ladder_force_all_demo.json`.

## Launch

From the `oph-physics-sim` repository root:

```bash
python3 -m tools.local_visualizer \
  --payload /absolute/path/to/visualization_payload.json \
  --open-browser
```

Standalone screen/A5 example from the workspace root:

```bash
cd oph-physics-sim
python3 -m tools.local_visualizer \
  --payload ../survival-proof-4/outputs/architecture/screen_a5_ladder_16k_force_all_demo.json \
  --data-root ../survival-proof-4/outputs \
  --manifest ../survival-proof-4/outputs/run_16k/visualization_export_manifest.json
```

The default address is `http://127.0.0.1:8765/`. Choose another port with
`--port 9000`. The server is deliberately loopback-only. The deprecated
`--allow-remote` option is non-operative; remote serving requires a separate,
authenticated deployment design rather than exposing this analyzer.

When `visualization_export_manifest.json` is beside the payload, the server
admits its recognized, contained visualization sidecars automatically. Use
`--manifest` and `--data-root` to select explicit locations; the payload and
every admitted sidecar must remain inside the resolved data root.

Stop the server with Ctrl-C.

Open the two local analyzer modes explicitly after launch:

```text
http://127.0.0.1:8765/?mode=receipt
http://127.0.0.1:8765/?mode=demo
```

The query parser accepts only `receipt` and `demo`; an absent or unknown value
falls back to receipt mode. `receipt` is always the default. The segmented
control rewrites the URL to one of those two allowlisted values, so a specific
local view can be bookmarked without putting arbitrary state in the URL.

## What the interface shows

The browser has these views, each with a short explanatory boundary:

1. Exact screen geometry and a windowed carrier federation
2. A5-to-Standard-Model dependency ladder
3. Observer repair and subjective camera handoff
4. Emergent universe, gravity, particles, and atoms
5. Cosmology and CMB comparison
6. Addressable census and provenance pages

The census distinguishes the exact finite **total**, the literal rows currently
**loaded/exported**, and the **visible** Canvas window. Any valid finite index
can be selected. Only the selected chunk is drawn; the app does not claim that
millions or billions of cells/atoms are literally rendered simultaneously.

The exact screen/A5 payload supplies the 12 ports, 30 edges, 20 faces, six
antipodal pairs, 60 A5 actions, `1 + 3 + 3-prime + 5` sectors, typed federation,
repair bridge, three distinct clocks, physical stage DAG, finite demo census,
and forced-display catalog when available.

## Receipt and demo modes (local analyzer only)

`Receipt mode` shows canonical payload status. `DEMO_ASSUMPTION` mode can force
missing display nodes and animate the complete explanatory path. Those toggles
exist only in browser memory. There is no write endpoint and no simulator or
campaign endpoint.

When a physical H3/KMS preflight snapshot is supplied to the timeline exporter,
`screenA5Ladder.physicalH3KmsDemoOverlay` keeps its exact copied P0--P8 rows and
digest beside a separate display overlay. In force-all demo exports, only stages
that are not physical `VALID_PASS` receive deterministic illustrative residuals,
spectra, clock/model comparisons, events, or rung rows. Every inserted field has
`DEMO_ASSUMPTION` provenance. `physicalGateStatus`,
`physicalScientificStatus`, the snapshot digest, promotion, retirement, and
scale authorization are never changed. A visually complete overlay means only
`displayComplete=true`; it is not a nine-stage physical pass.

When `screenA5Ladder.demoUniverse` records exist, demo animation reads their
light/readback, repair-residual, camera, gravity-proxy, particle, atom, and
cosmology rows. Otherwise the UI explicitly labels its motion a synthetic
renderer fallback. Gold pulses show light zapping/settling along exact carrier
edges and the visible carrier window; the observer view advances mismatch to a
display fixed point; the universe view carries a demo actor through the
worldline/gravity/observer layers. Motion pauses when the page is hidden.
`prefers-reduced-motion` produces a static settled/fixed-point frame.

The carrier census resolves finite addresses through the payload's exported
`addressSpaces` generator contract, or requests the matching manifested row API
when one exists. The readout names that source and its provenance. Carrier
miniatures reuse the exact 12-port/30-edge prototype. The visible federation
draws exported seam endpoints and propagates bounded pulses across that seam
graph; it does not merely connect adjacent screen pixels. The exact-carrier view
also exposes all 60 A5 actions as a clickable navigator and animates their
exported rotation actions in demo mode.

The following guards are supplied independently by the server and remain
immutable even if a payload or every display toggle requests completion:

```text
promotion_allowed = false
scientific_receipts_unchanged = true
SCALE_CAMPAIGN_ALLOWED = false
target_ancestry_eligible = false
```

Frozen `2pi`, W/Z/H values, particle assignments, atom rows, gravity responses,
or cosmological parameters are post-exposure display assumptions. They are not
prospective predictions or physical receipts.

## Progressive data API

The frontend never embeds the payload and does not consume a ZIP:

- `GET /api/health` — read-only server health
- `GET /api/summary` — bounded renderer summary and exact screen/A5 contract
- `GET /api/manifest` — opaque file IDs, sizes, paging endpoints, and guards
- `GET /api/files/<id>` — manifested file with standard single HTTP Range support
- `GET /api/pages/<id>?page=0&pageSize=262144` — bounded raw byte page
- `GET /api/rows/<id>?offset=0&limit=250` — bounded CSV/JSONL/top-level-array JSON rows

`pageSize` is capped at 4 MiB and `limit` at 2,000 rows. The Canvas uses
progressive row pages and a virtualized census window; it does not decode all
sidecars to render one scene. A data file larger than 4 MiB requires a Range or
page request; an accidental unbounded `GET` is rejected.

## Safety boundary

The server never maps a URL to a filesystem path and never provides directory
listing. It serves only the chosen payload and safe files listed by the
recognized `oph_universe_visualization_sidecars_v1` manifest. Opaque IDs are
used after admission.

Admission rejects:

- paths outside the selected root, including intermediary symlink escapes;
- symlink files and hidden/secret-like filenames;
- unsupported/archive/executable file types;
- payloads containing credential-bearing key names;
- bounded JSON sidecars containing credential-bearing keys;
- CSV or JSONL sidecars with credential-bearing header/object keys; and
- large JSON sidecars that cannot be safely scanned before raw serving.

The Content Security Policy permits only bundled same-origin scripts, styles,
images, and API calls. All mutation methods return `405`.

## Bounded handoff export

The companion bounded handoff command is:

```bash
python3 -m tools.local_visualizer.export --help
```

For the current public-safe 16k structural anchor, pass:

```bash
python3 -m tools.local_visualizer.export \
  --payload ../survival-proof-4/outputs/architecture/screen_a5_ladder_16k_force_all_demo.json \
  --public-anchor ../survival-proof-4/outputs/run_16k/visualization_anchor.json \
  --out-dir ../survival-proof-4/outputs/lovable_visualizer_handoff_v2 \
  --max-sidecar-bytes 240000000
```

The destination must not already exist. The anchor is accepted only when it is
marked public-safe, names the exact 16,384-carrier structure, records that
physical dynamics never started, and forbids run-anchored settling provenance.

It creates a plain-directory, hash-manifested handoff with a redacted payload,
selected small sidecars, full-vs-included census, omissions, and visualizer-agent
instructions. It also emits
`contracts/public_cinematic_story.json`, the gate-free primary contract for the
public Lovable Deko app. It does not ZIP by default. Use that export for a
Lovable/web coding handoff; use this local server for complete local inspection.

The two products deliberately differ:

- local analyzer: receipt tables, blockers, Receipt vs `DEMO_ASSUMPTION`, and
  immutable physical/display separation;
- public Lovable Deko app: cinematic story, subtle persistent
  `illustrative reconstruction`, and an optional provenance drawer, with no
  gate/pass-fail UI in the primary experience.

## Tests

```bash
pytest -q tests/test_local_visualizer_server.py
```

The tests cover health/summary/manifest endpoints, standalone ladder detection,
byte pages, HTTP Range, row pagination, path traversal, root escape and symlink
rejection, secret-bearing payload/sidecar rejection, read-only methods, and
frontend asset availability.
