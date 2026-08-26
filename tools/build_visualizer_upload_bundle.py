"""Build and validate the upload-ready OPH headline visualizer bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any
import zipfile


DEFAULT_SUITE = Path("visualizer_handoffs/oph-headlines-2026-08-20")
DEFAULT_OUTPUT = Path(
    "visualizer_bundles/oph-headlines-visualizer-builder-2026-08-20.zip"
)
BUNDLE_ID = "oph-headlines-visualizer-builder-2026-08-20"
BUNDLE_SCHEMA = "oph.visualizer-builder-upload-bundle.v1"
MAX_BUNDLE_BYTES = 200_000_000
ZIP_TIMESTAMP = (2026, 8, 20, 0, 0, 0)


BUILDER_INSTRUCTIONS = """# OPH visualizer builder instructions

This upload contains fourteen display-only OPH handoff packages. It contains
selected JSON/NPZ display data, manifests, hashes, and display instructions. It
does not contain a complete simulator run or simulator source code.

## Start here

1. Preserve the archive's directory structure when extracting it.
2. Read `bundle_manifest.json`, then `handoffs/suite_manifest.json`.
3. Treat every package's `manifest.json` and `DISPLAY_INSTRUCTIONS.md` as its
   rendering and interpretation contract.
4. Use `BUILDER_PROMPT.md` as the initial visualizer-builder prompt.
5. Build a static, responsive visualizer that loads files by relative path.
   No database, backend, credentials, or simulator execution is required.

## Recommended information architecture

- Begin with a suite overview that lists all fourteen views and their explicit
  claim boundaries.
- Give first-class navigation to the finite quantum-form conditioning fixture, refinement
  depth, defect emergence, and defect grouping/interactions.
- Keep the remaining carrier, repair, camera, modular-time, spacetime, A5,
  electromagnetic, cosmology, and evidence-atlas views available as peers.
- On every view, keep provenance status and the package claim boundary visible.

## Data rules

- Load only files listed in the package manifest's `files` array.
- Resolve those paths relative to that package directory.
- JSON numbers that also carry numerator/denominator fields must display the
  exact fraction as the authoritative value; decimals are presentation aids.
- NPZ files contain display arrays. Decode them in the browser with an
  NPZ-capable library or convert them to typed arrays at build time without
  changing values, labels, ordering, or integer types.
- Paths in a manifest's `sources` array are provenance references. They are not
  bundled inputs and must not be fetched by the visualizer.
- Do not infer scientific closure, physical identification, or evidential
  status from filenames, graph edges, colors, or visual proximity. Render the
  explicit status and claim-boundary fields.

## Required checks before display

- Verify every bundled handoff file against its byte count and SHA-256 in the
  package manifest.
- Verify every package manifest against `handoffs/suite_manifest.json`.
- Confirm all fourteen package IDs are represented.
- Confirm the interface works at desktop and mobile widths and provides
  keyboard-accessible controls and text alternatives for animated scenes.
- Confirm animations have pause/step controls and respect reduced-motion
  preferences.

## Suggested presentation order

1. Finite quantum-form weights and conditioning fixture
2. Refinement-depth emergence
3. Defect emergence
4. Defect grouping and interactions
5. S2 carrier-network interactions
6. Individual-carrier repair
7. Repair confluence and public records
8. Observer cameras
9. Observer modular time
10. Observer spacetime emergence
11. A5 symmetry and sector decomposition
12. Finite electromagnetic response
13. Screen-to-cosmology diagnostics
14. Theorem, paper, and simulation evidence atlas

## Local preview

After extraction, serve the archive root rather than opening files through a
`file://` URL:

```bash
python3 -m http.server 8000
```

Then open the builder output through `http://localhost:8000/` and test every
view against its package instructions.
"""


BUILDER_PROMPT = """Build a polished, responsive interactive OPH visualizer from this uploaded bundle.

First read `VISUALIZER_BUILDER_INSTRUCTIONS.md`, `bundle_manifest.json`, and
`handoffs/suite_manifest.json`. Then read every package's `manifest.json` and
`DISPLAY_INSTRUCTIONS.md` before implementing that view.

Create one navigable static web experience for all fourteen packages. Prioritize
the finite quantum-form conditioning fixture, refinement-depth, defect-emergence,
and defect-grouping views, while retaining every other package. Load only the
bundled display data by relative path. Keep exact fractions authoritative,
preserve integer/categorical meanings, and never manufacture missing values.
Show each package's provenance and claim boundary in its view. Do not turn a
computed, exploratory, conditional, axiomatized, legacy-control, or blocked
result into a physical derivation or promoted empirical claim.

Use accessible controls, responsive layouts, pause/step controls for animation,
reduced-motion support, and concise text alternatives. Prefer direct labels and
coherent cross-view navigation. Validate hashes and byte counts before treating
a package as ready. Do not add a backend, call external scientific APIs, fetch
the provenance source paths, or attempt to run the simulator.
"""


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe bundle path: {value}")
    return path


def validate_suite(suite: Path) -> tuple[dict[str, Any], list[Path]]:
    """Validate suite manifests and return every file approved for bundling."""

    suite = suite.resolve()
    suite_manifest_path = suite / "suite_manifest.json"
    if not suite_manifest_path.is_file():
        raise FileNotFoundError(suite_manifest_path)
    suite_manifest = _load_json(suite_manifest_path)
    if suite_manifest.get("schema") != "oph.visualizer-handoff-suite.v1":
        raise ValueError("unsupported handoff suite schema")
    if suite_manifest.get("display_data_only") is not True:
        raise ValueError("suite is not marked display-data-only")
    if suite_manifest.get("whole_run_included") is not False:
        raise ValueError("suite claims to include a whole run")

    packages = suite_manifest.get("packages", [])
    if suite_manifest.get("package_count") != len(packages):
        raise ValueError("suite package count does not match manifest rows")

    approved = [suite / "README.md", suite_manifest_path]
    for package_row in packages:
        package_id = str(_safe_relative_path(package_row["package_id"]))
        package = suite / package_id
        manifest_path = package / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        if manifest_path.is_symlink():
            raise ValueError(f"symlink is not allowed: {manifest_path}")
        if _sha256_path(manifest_path) != package_row["manifest_sha256"]:
            raise ValueError(f"package manifest hash mismatch: {package_id}")

        manifest = _load_json(manifest_path)
        if manifest.get("package_id") != package_id:
            raise ValueError(f"package id mismatch: {package_id}")
        if manifest.get("display_data_only") is not True:
            raise ValueError(f"package is not display-data-only: {package_id}")
        if manifest.get("whole_run_included") is not False:
            raise ValueError(f"package includes a whole run: {package_id}")
        if manifest.get("total_bytes", MAX_BUNDLE_BYTES) >= MAX_BUNDLE_BYTES:
            raise ValueError(f"package exceeds size limit: {package_id}")

        package_files = [manifest_path]
        for file_row in manifest.get("files", []):
            relative = _safe_relative_path(file_row["path"])
            path = package.joinpath(*relative.parts)
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.is_symlink():
                raise ValueError(f"symlink is not allowed: {path}")
            if path.stat().st_size != file_row["bytes"]:
                raise ValueError(f"file size mismatch: {path}")
            if _sha256_path(path) != file_row["sha256"]:
                raise ValueError(f"file hash mismatch: {path}")
            package_files.append(path)

        actual_files = {path.resolve() for path in package.rglob("*") if path.is_file()}
        expected_files = {path.resolve() for path in package_files}
        if actual_files != expected_files:
            unexpected = sorted(str(path) for path in actual_files - expected_files)
            missing = sorted(str(path) for path in expected_files - actual_files)
            raise ValueError(
                f"package contents differ from manifest for {package_id}; "
                f"unexpected={unexpected}, missing={missing}"
            )
        approved.extend(package_files)

    for path in approved:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"invalid suite file: {path}")
    return suite_manifest, sorted(set(approved))


def _zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def build_bundle(suite: Path = DEFAULT_SUITE, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    suite = (root / suite).resolve() if not suite.is_absolute() else suite.resolve()
    output = (root / output).resolve() if not output.is_absolute() else output.resolve()
    suite_manifest, source_files = validate_suite(suite)

    generated_entries = {
        f"{BUNDLE_ID}/VISUALIZER_BUILDER_INSTRUCTIONS.md": BUILDER_INSTRUCTIONS.encode(),
        f"{BUNDLE_ID}/BUILDER_PROMPT.md": BUILDER_PROMPT.encode(),
    }
    source_entries: dict[str, Path] = {}
    for path in source_files:
        relative = path.relative_to(suite).as_posix()
        source_entries[f"{BUNDLE_ID}/handoffs/{relative}"] = path

    entry_rows = []
    for archive_path, value in sorted(generated_entries.items()):
        entry_rows.append(
            {"path": archive_path, "bytes": len(value), "sha256": _sha256_bytes(value)}
        )
    for archive_path, path in sorted(source_entries.items()):
        entry_rows.append(
            {
                "path": archive_path,
                "bytes": path.stat().st_size,
                "sha256": _sha256_path(path),
            }
        )

    bundle_manifest = {
        "schema": BUNDLE_SCHEMA,
        "bundle_id": BUNDLE_ID,
        "display_data_only": True,
        "whole_run_included": False,
        "source_suite": suite.name,
        "source_suite_schema": suite_manifest["schema"],
        "package_count": suite_manifest["package_count"],
        "maximum_archive_bytes": MAX_BUNDLE_BYTES,
        "entries_excluding_bundle_manifest": entry_rows,
        "uncompressed_bytes_excluding_bundle_manifest": sum(
            row["bytes"] for row in entry_rows
        ),
    }
    manifest_bytes = (
        json.dumps(bundle_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    manifest_archive_path = f"{BUNDLE_ID}/bundle_manifest.json"

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for archive_path, value in sorted(generated_entries.items()):
                archive.writestr(_zip_info(archive_path), value)
            archive.writestr(_zip_info(manifest_archive_path), manifest_bytes)
            for archive_path, path in sorted(source_entries.items()):
                archive.writestr(_zip_info(archive_path), path.read_bytes())
        if temporary.stat().st_size >= MAX_BUNDLE_BYTES:
            raise ValueError(f"bundle exceeds {MAX_BUNDLE_BYTES} bytes")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    archive_hash = _sha256_path(output)
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(f"{archive_hash.removeprefix('sha256:')}  {output.name}\n")
    return {
        "archive": str(output),
        "archive_bytes": output.stat().st_size,
        "archive_sha256": archive_hash,
        "checksum": str(checksum_path),
        "package_count": suite_manifest["package_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.suite, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
