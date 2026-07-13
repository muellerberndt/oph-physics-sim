#!/usr/bin/env python3
"""Import the paper-side realized-branch receipt report as a pinned sidecar.

The canonical Einstein branch-entry receipt evaluation (issue #503) lives in
the reverse-engineering-reality repository at
code/geometry/runs/realized_branch_receipt_report.json. This tool validates
that report and copies it into a simulator run directory, where the Einstein
bridge manifest surfaces it as the informational paperSideRealizedBranch
block. The block never flips the run-receipt branch-entry gate: gravity
promotion keeps its own sidecar requirements.

Usage:
    python3 tools/import_realized_branch_receipts.py RUN_DIR [SOURCE_REPORT]
"""

from __future__ import annotations

from dataclasses import replace
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oph_fpe.evidence.cross_repo_artifacts import (  # noqa: E402
    DEFAULT_ARTIFACT_SPECS,
    import_cross_repo_artifacts,
    verify_cross_repo_artifact_manifest,
)

DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "reverse-engineering-reality"
    / "code"
    / "geometry"
    / "runs"
    / "realized_branch_receipt_report.json"
)

def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    run_dir = Path(sys.argv[1])
    source = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SOURCE
    if not source.is_file():
        print(f"source report not found: {source}")
        return 1
    try:
        source_repo = Path(
            subprocess.check_output(
                ["git", "-C", str(source.parent), "rev-parse", "--show-toplevel"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        ).resolve()
        source_relpath = source.resolve().relative_to(source_repo).as_posix()
    except (OSError, subprocess.CalledProcessError, ValueError):
        print(f"source report is not inside a Git research checkout: {source}")
        return 1
    base_spec = next(
        spec for spec in DEFAULT_ARTIFACT_SPECS if spec.key == "einstein_realized_branch"
    )
    spec = replace(base_spec, source_relpath=source_relpath)
    try:
        manifest = import_cross_repo_artifacts(source_repo, run_dir, specs=(spec,))
    except (ValueError, RuntimeError) as exc:
        print(f"source report failed validation: {exc}")
        return 1
    verified = verify_cross_repo_artifact_manifest(run_dir)
    row = manifest["artifacts"][0]
    print(f"imported {source} -> {run_dir / row['target_relpath']}")
    print(f"sha256: {row['sha256']}")
    print(f"source commit: {manifest['source_repository']['commit']}")
    print(f"source dirty: {manifest['source_repository']['dirty']}")
    print("realized_geometric_branch_certified_nonempty remains informational")
    return 0 if verified["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
