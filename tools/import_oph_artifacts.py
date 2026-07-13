#!/usr/bin/env python3
"""Import hash-pinned paper/particle artifacts and emit diagnostic summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oph_fpe.cosmology.particle_frontier import write_particle_frontier_report  # noqa: E402
from oph_fpe.bulk.paper_geometry_regressions import (  # noqa: E402
    write_paper_geometry_regression_report,
)
from oph_fpe.evidence.cross_repo_artifacts import (  # noqa: E402
    import_cross_repo_artifacts,
    verify_cross_repo_artifact_manifest,
)


DEFAULT_SOURCE = REPO_ROOT.parent / "reverse-engineering-reality"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="run or staging directory")
    parser.add_argument("--source-repo", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()

    manifest = import_cross_repo_artifacts(args.source_repo, args.destination)
    verification = verify_cross_repo_artifact_manifest(args.destination)
    frontier = write_particle_frontier_report(args.destination)
    geometry = write_paper_geometry_regression_report(args.destination)
    print(f"imported {sum(row.get('present') is True for row in manifest['artifacts'])} artifacts")
    print(f"paper release: {manifest.get('paper_release_id') or 'unknown'}")
    print(f"source commit: {manifest['source_repository']['commit']}")
    print(f"source dirty: {manifest['source_repository']['dirty']}")
    print(f"manifest verified: {verification['verified']}")
    print(f"particle receipts promoted: {frontier['simulation_receipts_promoted_by_import']}")
    print(f"paper geometry regressions pass: {geometry['all_golden_regressions_pass']}")
    return 0 if verification["verified"] and not manifest["missing_required_artifacts"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
