"""CLI for the exploratory dimension probe.

Runs the pinned sweep of DESIGN.md section 6, writes the canonical receipt,
and prints its SHA-256.  Non-evidential; the output is a table of numbers.

Usage:
    python -m oph_fpe.dimension [--out PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from oph_fpe.dimension import EVIDENTIAL_STATUS, probe, receipts

DEFAULT_OUT = Path("runs/dimension_probe/dimension_probe_receipt.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="oph_fpe.dimension",
        description=(
            "Exploratory, non-evidential spatial-dimensionality probe over "
            "the committed icosahedral tower."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="receipt path (canonical JSON; SHA-256 sidecar written next to it)",
    )
    arguments = parser.parse_args(argv)
    document, timings = probe.build_receipt()
    path, digest = receipts.write_receipt(document, arguments.out)
    timing_path = receipts.write_timings(timings, path)
    for line in probe.summary_lines(document):
        print(line)
    print(f"receipt: {path}")
    print(f"sha256: {digest}")
    print(f"timings_sidecar: {timing_path}")
    print(f"runtime_seconds_total: {timings['total_seconds']:.1f}")
    print(f"evidential_status: {EVIDENTIAL_STATUS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
