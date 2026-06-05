from __future__ import annotations

import argparse
import json
from pathlib import Path

from oph_fpe.cosmology.cl_ensemble import write_cl_ensemble_report
from oph_fpe.cosmology.cmb_compare import write_cmb_lite_comparison
from oph_fpe.cosmology.comparable_data import write_comparable_data_package
from oph_fpe.bulk.h3_ensemble import write_h3_ensemble_report
from oph_fpe.experiments import load_config, run_config
from oph_fpe.scale import run_array_screen_config, run_bw_array_config
from oph_fpe.scale.bw_sweep import run_bw_sweep
from oph_fpe.scale.refinement_report import write_refinement_report
from oph_fpe.viz import write_run_viewer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oph-fpe")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run an OPH-FPE config")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--out-dir", default=Path("runs"), type=Path)

    array_parser = subparsers.add_parser("run-array", help="run vectorized large-screen OPH-FPE config")
    array_parser.add_argument("--config", required=True, type=Path)
    array_parser.add_argument("--out-dir", default=Path("runs"), type=Path)

    bw_array_parser = subparsers.add_parser("run-bw-array", help="run vectorized BW cap-flow OPH-FPE config")
    bw_array_parser.add_argument("--config", required=True, type=Path)
    bw_array_parser.add_argument("--out-dir", default=Path("runs"), type=Path)

    bw_sweep_parser = subparsers.add_parser("run-bw-sweep", help="run BW cap-flow configs/seeds in parallel")
    bw_sweep_parser.add_argument("--configs", required=True, nargs="+", type=Path)
    bw_sweep_parser.add_argument("--out-dir", default=Path("runs"), type=Path)
    bw_sweep_parser.add_argument("--workers", type=int, default=None)
    bw_sweep_parser.add_argument("--inner-jobs", type=int, default=None)
    bw_sweep_parser.add_argument("--seeds", default=None, help="comma-separated seed list; defaults to each config seed")

    refinement_parser = subparsers.add_parser("refinement-report", help="aggregate state-derived BW refinement runs")
    refinement_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    refinement_parser.add_argument("--include", nargs="*", default=[], type=Path)
    refinement_parser.add_argument("--out", required=True, type=Path)

    cl_parser = subparsers.add_parser("cl-ensemble-report", help="aggregate gated freezeout-screen C_l runs")
    cl_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    cl_parser.add_argument("--include", nargs="*", default=[], type=Path)
    cl_parser.add_argument("--out", required=True, type=Path)

    h3_parser = subparsers.add_parser("h3-ensemble-report", help="aggregate H3 record/object/defect support receipts")
    h3_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    h3_parser.add_argument("--include", nargs="*", default=[], type=Path)
    h3_parser.add_argument("--out", required=True, type=Path)

    cmb_parser = subparsers.add_parser("cmb-lite-compare", help="compare a gated C_l proxy to a TT benchmark shape")
    cmb_parser.add_argument("--run-dir", required=True, type=Path)
    cmb_parser.add_argument("--benchmark", required=True, type=Path)
    cmb_parser.add_argument("--out", default=None, type=Path)
    cmb_parser.add_argument("--label", default="Planck2018_TT_binned")
    cmb_parser.add_argument("--source-url", default=None)
    cmb_parser.add_argument("--fields", default=None, help="comma-separated C_l field names; defaults to all")

    comparable_parser = subparsers.add_parser(
        "comparable-data",
        help="export current comparable diagnostic data from run receipts",
    )
    comparable_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    comparable_parser.add_argument("--include", nargs="*", default=[], type=Path)
    comparable_parser.add_argument("--out", required=True, type=Path)

    viewer_parser = subparsers.add_parser("run-viewer", help="write a standalone OPH receipt HTML viewer")
    viewer_parser.add_argument("--run-dir", required=True, type=Path)
    viewer_parser.add_argument("--out", default=None, type=Path)
    viewer_parser.add_argument("--max-screen-points", default=6000, type=int)

    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_config(load_config(args.config), args.out_dir)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "run-array":
        result = run_array_screen_config(load_config(args.config), args.out_dir)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "run-bw-array":
        result = run_bw_array_config(load_config(args.config), args.out_dir)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "run-bw-sweep":
        seeds = [int(value) for value in args.seeds.split(",")] if args.seeds else None
        result = run_bw_sweep(
            args.configs,
            args.out_dir,
            seeds=seeds,
            workers=args.workers,
            inner_jobs=args.inner_jobs,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "refinement-report":
        result = write_refinement_report([*args.run_dir, *args.include], args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "cl-ensemble-report":
        result = write_cl_ensemble_report([*args.run_dir, *args.include], args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "h3-ensemble-report":
        result = write_h3_ensemble_report([*args.run_dir, *args.include], args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "cmb-lite-compare":
        fields = [item.strip() for item in args.fields.split(",")] if args.fields else None
        result = write_cmb_lite_comparison(
            args.run_dir,
            args.benchmark,
            args.out,
            benchmark_label=args.label,
            source_url=args.source_url,
            field_names=fields,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "comparable-data":
        result = write_comparable_data_package([*args.run_dir, *args.include], args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "run-viewer":
        result = write_run_viewer(args.run_dir, args.out, max_screen_points=args.max_screen_points)
        print(json.dumps(result, indent=2, default=str))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
