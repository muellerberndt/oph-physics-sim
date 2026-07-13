#!/usr/bin/env python3
"""Measure whether saved OPH S3 carrier states directly emit POFT T0/T1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oph_fpe.flavor import build_poft_transport_emission_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        action="append",
        nargs=2,
        metavar=("LABEL", "S3_GAUGE_STATE_NPZ"),
        required=True,
    )
    parser.add_argument("--refinement-map", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = build_poft_transport_emission_report(
        [(label, Path(path)) for label, path in args.state],
        refinement_map=args.refinement_map,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    print(report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
