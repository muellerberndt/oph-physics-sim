#!/usr/bin/env python3
"""Verify the exact public measurement bytes consumed by OPH-FPE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oph_fpe.cosmology.public_measurement_source_binding import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    validate_public_measurement_sources,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail closed unless the selected Planck/SPARC/Cassini-summary "
            "bytes match the committed public measurement manifest."
        )
    )
    parser.add_argument(
        "--planck-tt",
        type=Path,
        default=Path(
            "data/measurements/planck2018/COM_PowerSpect_CMB-TT-binned_R3.01.txt"
        ),
    )
    parser.add_argument(
        "--sparc-dir", type=Path, default=Path("data/measurements/sparc")
    )
    parser.add_argument(
        "--cassini-summary",
        type=Path,
        default=Path("data/measurements/cassini/cassini_q2_2026.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=(
            "diagnostic override only; any path other than the code-pinned "
            "canonical manifest fails the canonical source-binding receipt"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the complete deterministic receipt instead of one status line",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    receipt = validate_public_measurement_sources(
        planck_tt_path=args.planck_tt,
        sparc_dir=args.sparc_dir,
        cassini_summary_path=args.cassini_summary,
        manifest_path=args.manifest,
    )
    if args.json:
        print(json.dumps(receipt, indent=2, allow_nan=False))
    else:
        print(
            f"{receipt['status']}: "
            f"{len(receipt['files'])} files, "
            f"{len(receipt['integrity_errors'])} errors"
        )
    return 0 if receipt["integrity_receipt"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
