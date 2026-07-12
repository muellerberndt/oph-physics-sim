#!/usr/bin/env python3
"""Import the paper-side realized-branch receipt report as a run sidecar.

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

import json
import shutil
import sys
from pathlib import Path

DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "reverse-engineering-reality"
    / "code"
    / "geometry"
    / "runs"
    / "realized_branch_receipt_report.json"
)

REQUIRED_KEYS = (
    "artifact",
    "issue",
    "status",
    "realized_geometric_branch_certified_nonempty",
)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    run_dir = Path(sys.argv[1])
    source = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SOURCE
    if not source.exists():
        print(f"source report not found: {source}")
        return 1
    with open(source) as f:
        report = json.load(f)
    missing = [k for k in REQUIRED_KEYS if k not in report]
    if missing or report.get("issue") != 503:
        print(f"source report failed validation (missing {missing})")
        return 1
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / "realized_branch_receipt_report.json"
    shutil.copyfile(source, target)
    print(f"imported {source} -> {target}")
    print(f"status: {report['status'][:100]}")
    nonempty = report["realized_geometric_branch_certified_nonempty"]
    print(f"realized_geometric_branch_certified_nonempty: {nonempty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
