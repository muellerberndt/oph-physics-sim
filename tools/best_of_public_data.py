#!/usr/bin/env python3
"""Emit a provenance-bound best-available public-data comparison suite."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oph_fpe.cosmology.public_data_comparisons import write_public_data_comparison_suite


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one explicitly selected OPH run with local public Planck/SPARC data, "
            "with optional declared history and baseline runs."
        )
    )
    parser.add_argument("--primary-run", required=True, type=Path)
    parser.add_argument("--history-run", action="append", default=[], type=Path)
    parser.add_argument("--baseline-run", type=Path, default=None)
    parser.add_argument(
        "--planck-tt",
        type=Path,
        default=Path("data/measurements/planck2018/COM_PowerSpect_CMB-TT-binned_R3.01.txt"),
    )
    parser.add_argument(
        "--sparc-dir",
        type=Path,
        default=Path("data/measurements/sparc"),
    )
    parser.add_argument(
        "--cassini-summary",
        type=Path,
        default=Path("data/measurements/cassini/cassini_q2_2026.json"),
    )
    parser.add_argument("--planned-config", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero when the suite integrity receipt is false",
    )
    parser.add_argument(
        "--require-frozen-prediction",
        action="store_true",
        help="exit nonzero unless at least one frozen physical-prediction receipt passes",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = write_public_data_comparison_suite(
        args.primary_run,
        args.out,
        planck_tt_path=args.planck_tt,
        sparc_dir=args.sparc_dir,
        history_run_dirs=args.history_run,
        baseline_run_dir=args.baseline_run,
        planned_config_path=args.planned_config,
        cassini_summary_path=args.cassini_summary,
    )
    summary = report["summary"]
    print(f"wrote: {args.out / 'best_of_public_data_comparisons.json'}")
    print(f"public comparisons: {summary['public_comparison_count']}")
    print(f"frozen predictions: {summary['frozen_prediction_count']}")
    print(f"integrity receipt: {str(summary['integrity_receipt']).lower()}")
    if args.strict and not summary["integrity_receipt"]:
        return 2
    if args.require_frozen_prediction and not summary["physical_prediction_available"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
