from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

from tools.build_visualizer_upload_bundle import (
    BUNDLE_ID,
    BUNDLE_SCHEMA,
    MAX_BUNDLE_BYTES,
    build_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "visualizer_handoffs/oph-headlines-2026-08-20"


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def test_generated_visualizer_outputs_are_ignored() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "visualizer_handoffs/" in ignore
    assert "visualizer_bundles/" in ignore


def test_upload_bundle_is_complete_safe_and_under_limit(tmp_path: Path) -> None:
    output = tmp_path / "oph-headlines.zip"
    receipt = build_bundle(SUITE, output)

    assert receipt["package_count"] == 14
    assert receipt["archive_bytes"] < MAX_BUNDLE_BYTES
    assert receipt["archive_sha256"] == _sha256(output.read_bytes())
    assert output.with_suffix(".zip.sha256").is_file()

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert all(not PurePosixPath(name).is_absolute() for name in names)
        assert all(".." not in PurePosixPath(name).parts for name in names)
        assert f"{BUNDLE_ID}/VISUALIZER_BUILDER_INSTRUCTIONS.md" in names
        assert f"{BUNDLE_ID}/BUILDER_PROMPT.md" in names
        manifest = json.loads(archive.read(f"{BUNDLE_ID}/bundle_manifest.json"))
        assert manifest["schema"] == BUNDLE_SCHEMA
        assert manifest["display_data_only"] is True
        assert manifest["whole_run_included"] is False
        assert manifest["package_count"] == 14
        for row in manifest["entries_excluding_bundle_manifest"]:
            value = archive.read(row["path"])
            assert len(value) == row["bytes"]
            assert _sha256(value) == row["sha256"]

        suite_manifest = json.loads(
            archive.read(f"{BUNDLE_ID}/handoffs/suite_manifest.json")
        )
        bundled_ids = {
            PurePosixPath(name).parts[2]
            for name in names
            if len(PurePosixPath(name).parts) > 3
            and PurePosixPath(name).parts[1] == "handoffs"
            and PurePosixPath(name).parts[2]
            not in {"README.md", "suite_manifest.json"}
        }
        assert bundled_ids == {
            package["package_id"] for package in suite_manifest["packages"]
        }
