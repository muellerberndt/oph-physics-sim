"""Build the upload-ready consensus-to-physics visualizer bundle.

Deterministic: fixed member order and fixed timestamps, so repeated builds of an
unchanged suite produce identical bytes.

    python tools/build_consensus_to_physics_bundle.py [suite_dir] [out.zip]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUITE = ROOT / "visualizer_handoffs/consensus-to-physics-2026-08-20"
DEFAULT_OUTPUT = ROOT / "visualizer_bundles/oph-consensus-to-physics-2026-08-20.zip"
BUNDLE_SCHEMA = "oph.visualizer-builder-upload-bundle.v2"
MAX_BUNDLE_BYTES = 200_000_000
ZIP_TIMESTAMP = (2026, 8, 20, 0, 0, 0)

BUILDER_PROMPT = """# Builder prompt

Build a visualizer that shows where physics comes from, and make it beautiful.

`handoff/CONSENSUS_TO_PHYSICS_BRIEF.md` is the contract: 35 panels, each with a **Render** block
saying what it should look like and which physics image to land on, a **Chain** from observer
consensus to observed physics, its data, and a one-line footnote boundary.

`handoff/reference_visualizer.html` is a working seven-stage example. Open it first - it sets the
tone, the colour discipline and the epistemic marks. It is not the target scope.

`handoff/DATA_INDEX.md` maps every panel to its files, and **every `data/` subdirectory carries its
own `DISPLAY_INSTRUCTIONS.md`** explaining how to turn that data into physics on screen. Read those
before writing render code for that directory.

## Scope

This suite shows OPH's **best results**, connected to physics. Panels organised around a number
that misses its target, or a ladder of gates that have not been discharged, are out of scope. Two
such panels were written and pulled; they live in `handoff/DEFERRED_PANELS.py`. Do not build them,
and do not reintroduce their framing - no panel should be built around a failing receipt or a
blocker list. Each panel still ends with one honest sentence about what it does not establish.

## Priorities, in order

1. **Visual splendor.** Real 3D, real animation, real depth. Full-dome skies, glowing lattices,
   fields with contours, worldlines threading a block universe. Canvas and WebGL, not static SVG.
   This should look like a planetarium exhibit about a new physics.
2. **Legible physics.** Wherever an OPH object matches a standard construction, say so and draw the
   correspondence - finite algebraic analyser fixtures, spherical harmonics, the H-theorem,
   classical record conditioning,
   Minkowski diagrams, Gauss's law, parallel transport, the Einstein equation, the CMB.
3. **Complete the picture.** Where the run's data is thinner than the scene, draw the scene anyway
   and mark it declared with a visually distinct treatment. A vivid scene with an honest label
   teaches more than a blank panel.
4. **Honest footnotes, not warning banners.** One quiet line at the base of each panel.

## Build these first

V07 (three dimensions crystallising), V12b (the observer's sky), V18 (classical record conditioning),
V23 (one generation of matter), V24b (a proto-particle seen from inside), V29b (the bulk
contracting around mass), then V14 and V17. V07 and V23 are exact and need no run data at all -
they can be recomputed in the browser and verified immediately.

## Two things not to get wrong

* **Gravity is contraction, not a dent.** For V29b, render the H3 bulk lattice shrinking and
  crowding toward mass, driven by `local_metric_conformal_factor`. Do not draw a rubber sheet with
  a ball on it - that picture smuggles in an extra dimension and teaches the wrong intuition.
* **Read chart-object values from `observer_chart_object_h3_report.json`, never from
  `manifest.json`** - see the handoff README's data caveat.

The output is a static, responsive site loading files by relative path. No backend, no credentials,
no simulator execution.
"""


def _add(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
    info = zipfile.ZipInfo(arcname, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zf.writestr(info, data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("suite", nargs="?", type=Path, default=DEFAULT_SUITE)
    ap.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    suite, out = args.suite, args.output
    if not suite.is_dir():
        raise SystemExit(f"suite directory not found: {suite}")
    out.parent.mkdir(parents=True, exist_ok=True)

    members = sorted(
        (p for p in suite.rglob("*") if p.is_file()),
        key=lambda p: str(p.relative_to(suite)),
    )
    rows = []
    for p in members:
        data = p.read_bytes()
        rows.append({
            "path": str(PurePosixPath("handoff") / p.relative_to(suite)),
            "bytes": len(data),
            "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        })

    prompt = BUILDER_PROMPT.encode()
    rows.append({"path": "BUILDER_PROMPT.md", "bytes": len(prompt),
                 "sha256": "sha256:" + hashlib.sha256(prompt).hexdigest()})

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "bundle_id": out.stem,
        "built": "2026-08-20",
        "source_run": "runs/e6_64k_dense_20260820",
        "simulator_revision": "b52196b296435d704b14d005d1f69caaaa662f97",
        "visualization_count": 33,
        "entry_points": ["BUILDER_PROMPT.md", "handoff/CONSENSUS_TO_PHYSICS_BRIEF.md",
                         "handoff/reference_visualizer.html", "handoff/DATA_INDEX.md"],
        "per_directory_instructions": [
            "handoff/data/screen/DISPLAY_INSTRUCTIONS.md",
            "handoff/data/timeline/DISPLAY_INSTRUCTIONS.md",
            "handoff/data/run/DISPLAY_INSTRUCTIONS.md",
            "handoff/data/derived/DISPLAY_INSTRUCTIONS.md",
            "handoff/data/receipts/DISPLAY_INSTRUCTIONS.md",
            "handoff/data/control/DISPLAY_INSTRUCTIONS.md"],
        "display_data_only": True,
        "whole_run_included": False,
        "files": sorted(rows, key=lambda r: r["path"]),
        "total_bytes": sum(r["bytes"] for r in rows),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        _add(zf, "bundle_manifest.json", manifest_bytes)
        _add(zf, "BUILDER_PROMPT.md", prompt)
        for p in members:
            _add(zf, str(PurePosixPath("handoff") / p.relative_to(suite)), p.read_bytes())

    size = out.stat().st_size
    if size > MAX_BUNDLE_BYTES:
        out.unlink()
        raise SystemExit(f"bundle exceeds {MAX_BUNDLE_BYTES} bytes: {size}")

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    Path(str(out) + ".sha256").write_text(f"{digest}  {out.name}\n")
    print(f"wrote {out}  {size/1e6:.2f} MB  ({len(members)+2} members)")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
