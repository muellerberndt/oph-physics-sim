from __future__ import annotations

import argparse
import json
from pathlib import Path


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

    h3_refit_parser = subparsers.add_parser("h3-refit", help="refit cached modular-response kernel into H3")
    h3_refit_parser.add_argument("--run-dir", required=True, type=Path)
    h3_refit_parser.add_argument("--out", default=None, type=Path)
    h3_refit_parser.add_argument("--candidate-count", default=4096, type=int)
    h3_refit_parser.add_argument("--candidate-radius", default=2.0, type=float)
    h3_refit_parser.add_argument("--softness", default=0.25, type=float)
    h3_refit_parser.add_argument("--seed", default=1, type=int)
    h3_refit_parser.add_argument("--pass-ratio", default=1.0, type=float)
    h3_refit_parser.add_argument("--min-observers", default=8, type=int)
    h3_refit_parser.add_argument("--min-features", default=12, type=int)
    h3_refit_parser.add_argument("--fit-mode", default="joint_global")
    h3_refit_parser.add_argument("--heldout-fraction", default=0.25, type=float)
    h3_refit_parser.add_argument("--anchor-weight", default=0.05, type=float)
    h3_refit_parser.add_argument("--max-iterations", default=4, type=int)
    h3_refit_parser.add_argument("--feature-selection", default="none")
    h3_refit_parser.add_argument("--max-fit-features", default=None, type=int)
    h3_refit_parser.add_argument("--min-feature-std", default=0.0, type=float)
    h3_refit_parser.add_argument("--min-wrong-scale-feature-delta", default=0.0, type=float)
    h3_refit_parser.add_argument("--exclude-observables", default="")
    h3_refit_parser.add_argument("--exclude-feature-types", default="")
    h3_refit_parser.add_argument("--max-features-per-cap-time-observable", default=None, type=int)
    h3_refit_parser.add_argument("--refine-steps", default=0, type=int)
    h3_refit_parser.add_argument("--refine-max-rows", default=None, type=int)
    h3_refit_parser.add_argument("--refine-max-nfev", default=48, type=int)
    h3_refit_parser.add_argument("--candidate-mode", default="random")
    h3_refit_parser.add_argument("--channel-mode", default="time_observable_class")
    h3_refit_parser.add_argument("--profile-mode", default="static_halfspace")
    h3_refit_parser.add_argument("--profile-time-scale", default=6.283185307179586, type=float)
    h3_refit_parser.add_argument("--control-fit-mode", default="same_h3_model_not_affine_target_fit")

    h3_refit_ensemble_parser = subparsers.add_parser(
        "h3-refit-ensemble",
        help="run multiple cached H3 refits to test candidate-seed robustness",
    )
    h3_refit_ensemble_parser.add_argument("--run-dir", required=True, type=Path)
    h3_refit_ensemble_parser.add_argument("--out", default=None, type=Path)
    h3_refit_ensemble_parser.add_argument("--seeds", required=True, help="comma-separated H3 candidate seeds")
    h3_refit_ensemble_parser.add_argument("--required-receipt-fraction", default=0.75, type=float)
    h3_refit_ensemble_parser.add_argument("--required-dim3-fraction", default=0.5, type=float)
    h3_refit_ensemble_parser.add_argument("--required-median-ev", default=0.08, type=float)
    h3_refit_ensemble_parser.add_argument("--required-p75-material-wrong-fraction", default=0.075, type=float)
    h3_refit_ensemble_parser.add_argument("--candidate-count", default=4096, type=int)
    h3_refit_ensemble_parser.add_argument("--candidate-radius", default=2.0, type=float)
    h3_refit_ensemble_parser.add_argument("--softness", default=0.25, type=float)
    h3_refit_ensemble_parser.add_argument("--pass-ratio", default=1.0, type=float)
    h3_refit_ensemble_parser.add_argument("--min-observers", default=8, type=int)
    h3_refit_ensemble_parser.add_argument("--min-features", default=12, type=int)
    h3_refit_ensemble_parser.add_argument("--fit-mode", default="joint_global")
    h3_refit_ensemble_parser.add_argument("--heldout-fraction", default=0.25, type=float)
    h3_refit_ensemble_parser.add_argument("--anchor-weight", default=0.05, type=float)
    h3_refit_ensemble_parser.add_argument("--max-iterations", default=4, type=int)
    h3_refit_ensemble_parser.add_argument("--feature-selection", default="none")
    h3_refit_ensemble_parser.add_argument("--max-fit-features", default=None, type=int)
    h3_refit_ensemble_parser.add_argument("--min-feature-std", default=0.0, type=float)
    h3_refit_ensemble_parser.add_argument("--min-wrong-scale-feature-delta", default=0.0, type=float)
    h3_refit_ensemble_parser.add_argument("--exclude-observables", default="")
    h3_refit_ensemble_parser.add_argument("--exclude-feature-types", default="")
    h3_refit_ensemble_parser.add_argument("--max-features-per-cap-time-observable", default=None, type=int)
    h3_refit_ensemble_parser.add_argument("--refine-steps", default=0, type=int)
    h3_refit_ensemble_parser.add_argument("--refine-max-rows", default=None, type=int)
    h3_refit_ensemble_parser.add_argument("--refine-max-nfev", default=48, type=int)
    h3_refit_ensemble_parser.add_argument("--candidate-mode", default="random")
    h3_refit_ensemble_parser.add_argument("--channel-mode", default="time_observable_class")
    h3_refit_ensemble_parser.add_argument("--profile-mode", default="static_halfspace")
    h3_refit_ensemble_parser.add_argument("--profile-time-scale", default=6.283185307179586, type=float)
    h3_refit_ensemble_parser.add_argument("--control-fit-mode", default="same_h3_model_not_affine_target_fit")

    cmb_parser = subparsers.add_parser("cmb-lite-compare", help="compare a gated C_l proxy to a TT benchmark shape")
    cmb_parser.add_argument("--run-dir", required=True, type=Path)
    cmb_parser.add_argument("--benchmark", required=True, type=Path)
    cmb_parser.add_argument("--out", default=None, type=Path)
    cmb_parser.add_argument("--label", default="Planck2018_TT_binned")
    cmb_parser.add_argument("--source-url", default=None)
    cmb_parser.add_argument("--fields", default=None, help="comma-separated C_l field names; defaults to all")

    cl_from_npz_parser = subparsers.add_parser(
        "cl-from-freezeout-npz",
        help="recompute freezeout-screen C_l from a saved freezeout_fields.npz bundle",
    )
    cl_from_npz_parser.add_argument("--run-dir", required=True, type=Path)
    cl_from_npz_parser.add_argument("--out", default=None, type=Path)
    cl_from_npz_parser.add_argument("--ell-max", default=256, type=int)
    cl_from_npz_parser.add_argument("--harmonic-batch-size", default=512, type=int)
    cl_from_npz_parser.add_argument("--n-jobs", default=1, type=int)
    cl_from_npz_parser.add_argument("--fields", default=None, help="comma-separated field names; defaults to all saved fields")
    cl_from_npz_parser.add_argument("--benchmark", default=None, type=Path)
    cl_from_npz_parser.add_argument("--label", default="Planck2018_TT_binned")
    cl_from_npz_parser.add_argument("--source-url", default=None)
    cl_from_npz_parser.add_argument("--seed", default=1, type=int)

    transfer_parser = subparsers.add_parser(
        "cmb-transfer-report",
        help="fit/test a cross-scale screen-field basis against a TT benchmark shape",
    )
    transfer_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    transfer_parser.add_argument("--include", nargs="*", default=[], type=Path)
    transfer_parser.add_argument("--benchmark", required=True, type=Path)
    transfer_parser.add_argument("--out", required=True, type=Path)
    transfer_parser.add_argument("--train-patch-count", default=None, type=int)
    transfer_parser.add_argument("--test-patch-count", default=None, type=int)
    transfer_parser.add_argument("--fields", default=None, help="comma-separated C_l field names; defaults to declared basis")
    transfer_parser.add_argument("--ridge", default=1.0e-3, type=float)
    transfer_parser.add_argument("--sample-count", default=128, type=int)
    transfer_parser.add_argument("--control-seed", default=9137, type=int)
    transfer_parser.add_argument("--bootstrap-count", default=0, type=int)
    transfer_parser.add_argument("--bootstrap-seed", default=271828, type=int)

    camb_parser = subparsers.add_parser(
        "camb-baseline-report",
        help="run a standard LambdaCDM CAMB TT baseline against a local benchmark table",
    )
    camb_parser.add_argument("--benchmark", required=True, type=Path)
    camb_parser.add_argument("--out", required=True, type=Path)
    camb_parser.add_argument("--lmax", default=2600, type=int)
    camb_parser.add_argument("--label", default="Planck2018_TT_binned")

    screen_camb_parser = subparsers.add_parser(
        "oph-screen-camb",
        help="run CAMB TT transfer from an OPH screen-power report scaffold",
    )
    screen_camb_parser.add_argument("--screen-report", required=True, type=Path)
    screen_camb_parser.add_argument("--benchmark", required=True, type=Path)
    screen_camb_parser.add_argument("--out", required=True, type=Path)
    screen_camb_parser.add_argument("--lmax", default=2600, type=int)
    screen_camb_parser.add_argument("--label", default="Planck2018_TT_binned")

    inflation_camb_parser = subparsers.add_parser(
        "oph-inflation-cmb-camb",
        help="run CAMB TT transfer for the OPH P/48 spectrum plus imported v0.4 IR kernel",
    )
    inflation_camb_parser.add_argument("--bridge-report", required=True, type=Path)
    inflation_camb_parser.add_argument("--benchmark", required=True, type=Path)
    inflation_camb_parser.add_argument("--out", required=True, type=Path)
    inflation_camb_parser.add_argument("--lmax", default=2600, type=int)
    inflation_camb_parser.add_argument("--label", default="Planck2018_TT_binned")

    boltzmann_input_parser = subparsers.add_parser(
        "oph-boltzmann-inputs",
        help="export CDM-limit and OPH diagnostic anomaly-stress inputs for future CAMB/CLASS modules",
    )
    boltzmann_input_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    boltzmann_input_parser.add_argument("--include", nargs="*", default=[], type=Path)
    boltzmann_input_parser.add_argument("--out", required=True, type=Path)

    screen_power_parser = subparsers.add_parser(
        "oph-screen-power",
        help="estimate OPH screen-covariance parameters and export a primordial table scaffold",
    )
    screen_power_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    screen_power_parser.add_argument("--include", nargs="*", default=[], type=Path)
    screen_power_parser.add_argument("--out", required=True, type=Path)
    screen_power_parser.add_argument("--fields", default=None, help="comma-separated C_l fields; defaults to all")
    screen_power_parser.add_argument("--ell-min", default=20.0, type=float)
    screen_power_parser.add_argument("--ell-max", default=None, type=float)
    screen_power_parser.add_argument("--primordial-k-count", default=256, type=int)
    screen_power_parser.add_argument("--primordial-k-min", default=1.0e-4, type=float)
    screen_power_parser.add_argument("--primordial-k-max", default=1.0, type=float)

    sync_inflation_parser = subparsers.add_parser(
        "sync-inflation-report",
        help="audit OPH synchronization/flatness/horizon theorem-target diagnostics from cached runs",
    )
    sync_inflation_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    sync_inflation_parser.add_argument("--include", nargs="*", default=[], type=Path)
    sync_inflation_parser.add_argument("--out", required=True, type=Path)
    sync_inflation_parser.add_argument("--w-eff", default=1.0 / 3.0, type=float)

    sync_gap_parser = subparsers.add_parser(
        "sync-gap-report",
        help="audit mode-wise low-k synchronization-gap readiness from cached runs",
    )
    sync_gap_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    sync_gap_parser.add_argument("--include", nargs="*", default=[], type=Path)
    sync_gap_parser.add_argument("--out", required=True, type=Path)
    sync_gap_parser.add_argument("--ell-max-cmb", default=32, type=int)
    sync_gap_parser.add_argument("--min-gamma-per-cycle", default=1.0e-3, type=float)
    sync_gap_parser.add_argument("--min-control-z", default=1.0, type=float)

    hot_release_parser = subparsers.add_parser(
        "hot-release-report",
        help="audit OPH hot MaxEnt release surfaces from cached runs",
    )
    hot_release_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    hot_release_parser.add_argument("--include", nargs="*", default=[], type=Path)
    hot_release_parser.add_argument("--out", required=True, type=Path)
    hot_release_parser.add_argument("--phi-tolerance", default=0.0, type=float)
    hot_release_parser.add_argument("--min-committed-fraction", default=0.99, type=float)
    hot_release_parser.add_argument("--max-collar-cmi", default=0.05, type=float)

    adiabaticity_parser = subparsers.add_parser(
        "adiabaticity-report",
        help="audit same-boundary adiabaticity/isocurvature proxy from freezeout fields",
    )
    adiabaticity_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    adiabaticity_parser.add_argument("--include", nargs="*", default=[], type=Path)
    adiabaticity_parser.add_argument("--out", required=True, type=Path)
    adiabaticity_parser.add_argument("--max-entropy-residual-std", default=0.25, type=float)
    adiabaticity_parser.add_argument("--min-common-clock-corr", default=0.85, type=float)

    h0s8_parser = subparsers.add_parser(
        "h0s8-branch-report",
        help="write OPH H0/S8 branch diagnostics from finite-collar/Jacobi assumptions",
    )
    h0s8_parser.add_argument("--out", required=True, type=Path)
    h0s8_parser.add_argument("--q-a", default=5.363470441, type=float)

    compressed_parser = subparsers.add_parser(
        "oph-compressed-likelihood",
        help="write the OPH compressed CMB/BAO/growth/S8 reference diagnostic",
    )
    compressed_parser.add_argument("--out", required=True, type=Path)

    inflation_cmb_parser = subparsers.add_parser(
        "oph-inflation-cmb-bridge",
        help="write the OPH P/48 screen-spectrum and CMB v0.4/v0.5 diagnostic bridge report",
    )
    inflation_cmb_parser.add_argument("--source-dir", default=None, type=Path)
    inflation_cmb_parser.add_argument("--out", required=True, type=Path)

    unique_cmb_parser = subparsers.add_parser(
        "oph-unique-predictions",
        help="write the OPH v0.9 unique prediction gate with public-comparison targets",
    )
    unique_cmb_parser.add_argument("--source-dir", default=None, type=Path)
    unique_cmb_parser.add_argument("--out", required=True, type=Path)

    cmb_derivation_parser = subparsers.add_parser(
        "cmb-derivation-report",
        help="audit whether finite lattice receipts currently derive the OPH CMB target parameters",
    )
    cmb_derivation_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    cmb_derivation_parser.add_argument("--include", nargs="*", default=[], type=Path)
    cmb_derivation_parser.add_argument("--source-dir", default=None, type=Path)
    cmb_derivation_parser.add_argument("--out", required=True, type=Path)

    cmb_anomaly_parser = subparsers.add_parser(
        "cmb-anomaly-report",
        help="write finite-screen low-ell/parity/large-angle/capacity CMB anomaly diagnostics",
    )
    cmb_anomaly_parser.add_argument("--run-dir", required=True, type=Path)
    cmb_anomaly_parser.add_argument("--out", default=None, type=Path)
    cmb_anomaly_parser.add_argument("--source-dir", default=None, type=Path)
    cmb_anomaly_parser.add_argument("--fields", default=None, help="comma-separated C_l field names; defaults to all")
    cmb_anomaly_parser.add_argument("--low-lmax", default=29, type=int)
    cmb_anomaly_parser.add_argument("--parity-lmax", default=29, type=int)
    cmb_anomaly_parser.add_argument("--s12-lmax", default=29, type=int)

    comparable_parser = subparsers.add_parser(
        "comparable-data",
        help="export current comparable diagnostic data from run receipts",
    )
    comparable_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    comparable_parser.add_argument("--include", nargs="*", default=[], type=Path)
    comparable_parser.add_argument("--out", required=True, type=Path)

    recompute_object_parser = subparsers.add_parser(
        "recompute-object-chart",
        help="recompute observer-object H3 population from saved observer JSONL and a cached H3 report",
    )
    recompute_object_parser.add_argument("--run-dir", required=True, type=Path)
    recompute_object_parser.add_argument("--h3-report", required=True, type=Path)
    recompute_object_parser.add_argument("--out", required=True, type=Path)
    recompute_object_parser.add_argument("--shuffle-count", default=128, type=int)
    recompute_object_parser.add_argument(
        "--incidence-mode",
        default="record_family_modular_response_mixture",
        choices=[
            "record_family_modular_response_mixture",
            "record_sector_checkpoint_lineage",
            "transition_history",
            "observer_transition_mixture_cluster",
        ],
    )

    galaxy_static_parser = subparsers.add_parser(
        "run-galaxy-static",
        help="fit the OPH static RAR/BTFR continuation law to external galaxy rows",
    )
    galaxy_static_parser.add_argument("--dataset", required=True, type=Path)
    galaxy_static_parser.add_argument("--out-dir", required=True, type=Path)
    galaxy_static_parser.add_argument("--a0-initial", default=1.2e-10, type=float)
    galaxy_static_parser.add_argument("--lambda-initial", default=1.0, type=float)
    galaxy_static_parser.add_argument("--min-points", default=12, type=int)
    galaxy_static_parser.add_argument("--min-galaxies", default=1, type=int)
    galaxy_static_parser.add_argument("--diagnostic-only", action="store_true")

    viewer_parser = subparsers.add_parser("run-viewer", help="write a standalone OPH receipt HTML viewer")
    viewer_parser.add_argument("--run-dir", required=True, type=Path)
    viewer_parser.add_argument("--out", default=None, type=Path)
    viewer_parser.add_argument("--max-screen-points", default=6000, type=int)

    defect_assay_parser = subparsers.add_parser(
        "controlled-defect-assay",
        help="write a controlled planted S3 inverse-defect particle-gate assay report",
    )
    defect_assay_parser.add_argument("--out", required=True, type=Path)
    defect_assay_parser.add_argument("--patch-count", default=65_536, type=int)
    defect_assay_parser.add_argument("--observation-count", default=5, type=int)
    defect_assay_parser.add_argument("--support-node-count", default=8, type=int)
    defect_assay_parser.add_argument("--holonomy", default=1, type=int)
    defect_assay_parser.add_argument("--cycle-stride", default=8, type=int)
    defect_assay_parser.add_argument("--max-support-fraction", default=0.05, type=float)

    caps_to_h3_parser = subparsers.add_parser(
        "caps-to-h3-minimal",
        help="run the minimal S2 cap-response profile -> H3 reconstruction receipt",
    )
    caps_to_h3_parser.add_argument("--out", default=None, type=Path)
    caps_to_h3_parser.add_argument("--axis-count", default=64, type=int)
    caps_to_h3_parser.add_argument("--theta-values", default="0.35,0.55,0.75,1.0,1.25")
    caps_to_h3_parser.add_argument("--object-count", default=16, type=int)
    caps_to_h3_parser.add_argument("--object-radius", default=1.0, type=float)
    caps_to_h3_parser.add_argument("--fit-radius", default=2.0, type=float)
    caps_to_h3_parser.add_argument("--softness", default=0.25, type=float)
    caps_to_h3_parser.add_argument("--restarts", default=24, type=int)
    caps_to_h3_parser.add_argument("--seed", default=1, type=int)
    caps_to_h3_parser.add_argument("--max-median-error", default=0.02, type=float)

    args = parser.parse_args(argv)
    if args.command == "run":
        from oph_fpe.experiments import load_config, run_config

        result = run_config(load_config(args.config), args.out_dir)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "run-array":
        from oph_fpe.experiments import load_config
        from oph_fpe.scale import run_array_screen_config

        result = run_array_screen_config(load_config(args.config), args.out_dir)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "run-bw-array":
        from oph_fpe.experiments import load_config
        from oph_fpe.scale import run_bw_array_config

        result = run_bw_array_config(load_config(args.config), args.out_dir)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "run-bw-sweep":
        from oph_fpe.scale.bw_sweep import run_bw_sweep

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
        from oph_fpe.scale.refinement_report import write_refinement_report

        result = write_refinement_report([*args.run_dir, *args.include], args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "cl-ensemble-report":
        from oph_fpe.cosmology.cl_ensemble import write_cl_ensemble_report

        result = write_cl_ensemble_report([*args.run_dir, *args.include], args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "h3-ensemble-report":
        from oph_fpe.bulk.h3_ensemble import write_h3_ensemble_report

        result = write_h3_ensemble_report([*args.run_dir, *args.include], args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "h3-refit":
        from oph_fpe.bulk.h3_refit import write_h3_refit_report

        result = write_h3_refit_report(
            args.run_dir,
            args.out,
            candidate_count=args.candidate_count,
            candidate_radius=args.candidate_radius,
            softness=args.softness,
            seed=args.seed,
            pass_ratio=args.pass_ratio,
            min_observers=args.min_observers,
            min_features=args.min_features,
            fit_mode=args.fit_mode,
            heldout_fraction=args.heldout_fraction,
            anchor_weight=args.anchor_weight,
            max_iterations=args.max_iterations,
            feature_selection=args.feature_selection,
            max_fit_features=args.max_fit_features,
            min_feature_std=args.min_feature_std,
            min_wrong_scale_feature_delta=args.min_wrong_scale_feature_delta,
            exclude_observables=_csv_values(args.exclude_observables),
            exclude_feature_types=_csv_values(args.exclude_feature_types),
            max_features_per_cap_time_observable=args.max_features_per_cap_time_observable,
            refine_steps=args.refine_steps,
            refine_max_rows=args.refine_max_rows,
            refine_max_nfev=args.refine_max_nfev,
            candidate_mode=args.candidate_mode,
            channel_mode=args.channel_mode,
            profile_mode=args.profile_mode,
            profile_time_scale=args.profile_time_scale,
            control_fit_mode=args.control_fit_mode,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "h3-refit-ensemble":
        from oph_fpe.bulk.h3_refit import write_h3_refit_ensemble_report

        seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
        result = write_h3_refit_ensemble_report(
            args.run_dir,
            args.out,
            seeds=seeds,
            required_receipt_fraction=args.required_receipt_fraction,
            required_dim3_fraction=args.required_dim3_fraction,
            required_median_ev=args.required_median_ev,
            required_p75_material_wrong_fraction=args.required_p75_material_wrong_fraction,
            candidate_count=args.candidate_count,
            candidate_radius=args.candidate_radius,
            softness=args.softness,
            pass_ratio=args.pass_ratio,
            min_observers=args.min_observers,
            min_features=args.min_features,
            fit_mode=args.fit_mode,
            heldout_fraction=args.heldout_fraction,
            anchor_weight=args.anchor_weight,
            max_iterations=args.max_iterations,
            feature_selection=args.feature_selection,
            max_fit_features=args.max_fit_features,
            min_feature_std=args.min_feature_std,
            min_wrong_scale_feature_delta=args.min_wrong_scale_feature_delta,
            exclude_observables=_csv_values(args.exclude_observables),
            exclude_feature_types=_csv_values(args.exclude_feature_types),
            max_features_per_cap_time_observable=args.max_features_per_cap_time_observable,
            refine_steps=args.refine_steps,
            refine_max_rows=args.refine_max_rows,
            refine_max_nfev=args.refine_max_nfev,
            candidate_mode=args.candidate_mode,
            channel_mode=args.channel_mode,
            profile_mode=args.profile_mode,
            profile_time_scale=args.profile_time_scale,
            control_fit_mode=args.control_fit_mode,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "cmb-lite-compare":
        from oph_fpe.cosmology.cmb_compare import write_cmb_lite_comparison

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
    if args.command == "cl-from-freezeout-npz":
        from oph_fpe.cosmology.cl_postprocess import write_cl_from_freezeout_npz

        fields = [item.strip() for item in args.fields.split(",")] if args.fields else None
        result = write_cl_from_freezeout_npz(
            args.run_dir,
            args.out,
            ell_max=args.ell_max,
            fields=fields,
            harmonic_batch_size=args.harmonic_batch_size,
            n_jobs=args.n_jobs,
            benchmark=args.benchmark,
            benchmark_label=args.label,
            source_url=args.source_url,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "cmb-transfer-report":
        from oph_fpe.cosmology.cmb_transfer import write_cmb_transfer_report

        fields = [item.strip() for item in args.fields.split(",")] if args.fields else None
        result = write_cmb_transfer_report(
            [*args.run_dir, *args.include],
            args.benchmark,
            args.out,
            train_patch_count=args.train_patch_count,
            test_patch_count=args.test_patch_count,
            field_names=fields,
            ridge=args.ridge,
            sample_count=args.sample_count,
            control_seed=args.control_seed,
            bootstrap_count=args.bootstrap_count,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "camb-baseline-report":
        from oph_fpe.cosmology.camb_adapter import write_camb_lcdm_baseline_report

        result = write_camb_lcdm_baseline_report(
            args.benchmark,
            args.out,
            lmax=args.lmax,
            benchmark_label=args.label,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "oph-screen-camb":
        from oph_fpe.cosmology.camb_adapter import write_oph_screen_camb_report

        result = write_oph_screen_camb_report(
            args.screen_report,
            args.benchmark,
            args.out,
            lmax=args.lmax,
            benchmark_label=args.label,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "oph-inflation-cmb-camb":
        from oph_fpe.cosmology.camb_adapter import write_oph_inflation_cmb_camb_report

        result = write_oph_inflation_cmb_camb_report(
            args.bridge_report,
            args.benchmark,
            args.out,
            lmax=args.lmax,
            benchmark_label=args.label,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "oph-boltzmann-inputs":
        from oph_fpe.cosmology.boltzmann_inputs import write_oph_boltzmann_input_report

        result = write_oph_boltzmann_input_report([*args.run_dir, *args.include], args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "oph-screen-power":
        from oph_fpe.cosmology.oph_screen_power import write_oph_screen_power_report

        fields = [item.strip() for item in args.fields.split(",")] if args.fields else None
        result = write_oph_screen_power_report(
            [*args.run_dir, *args.include],
            args.out,
            field_names=fields,
            ell_min=args.ell_min,
            ell_max=args.ell_max,
            primordial_k_count=args.primordial_k_count,
            primordial_k_min=args.primordial_k_min,
            primordial_k_max=args.primordial_k_max,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "sync-inflation-report":
        from oph_fpe.cosmology.sync_inflation import write_synchronization_inflation_report

        result = write_synchronization_inflation_report([*args.run_dir, *args.include], args.out, w_eff=args.w_eff)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "sync-gap-report":
        from oph_fpe.cosmology.sync_gap import write_synchronization_gap_report

        result = write_synchronization_gap_report(
            [*args.run_dir, *args.include],
            args.out,
            ell_max_cmb=args.ell_max_cmb,
            min_gamma_per_cycle=args.min_gamma_per_cycle,
            min_control_z=args.min_control_z,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "hot-release-report":
        from oph_fpe.cosmology.hot_release import write_hot_release_report

        result = write_hot_release_report(
            [*args.run_dir, *args.include],
            args.out,
            phi_tolerance=args.phi_tolerance,
            min_committed_fraction=args.min_committed_fraction,
            max_collar_cmi=args.max_collar_cmi,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "adiabaticity-report":
        from oph_fpe.cosmology.adiabaticity import write_adiabaticity_report

        result = write_adiabaticity_report(
            [*args.run_dir, *args.include],
            args.out,
            max_entropy_residual_std=args.max_entropy_residual_std,
            min_common_clock_corr=args.min_common_clock_corr,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "h0s8-branch-report":
        from oph_fpe.cosmology.h0s8 import write_h0s8_branch_report

        result = write_h0s8_branch_report(args.out, q_a=args.q_a)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "oph-compressed-likelihood":
        from oph_fpe.cosmology.compressed_likelihood import write_compressed_likelihood_reference_report

        result = write_compressed_likelihood_reference_report(args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "oph-inflation-cmb-bridge":
        from oph_fpe.cosmology.inflation_cmb_ladder import write_inflation_cmb_bridge_report

        result = write_inflation_cmb_bridge_report(args.source_dir, args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "oph-unique-predictions":
        from oph_fpe.cosmology.unique_predictions import write_unique_prediction_gate_report

        result = write_unique_prediction_gate_report(args.source_dir, args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "cmb-derivation-report":
        from oph_fpe.cosmology.cmb_derivation import write_cmb_parameter_derivation_report

        result = write_cmb_parameter_derivation_report([*args.run_dir, *args.include], args.out, source_dir=args.source_dir)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "cmb-anomaly-report":
        from oph_fpe.cosmology.cmb_anomaly import write_cmb_anomaly_report

        fields = [item.strip() for item in args.fields.split(",")] if args.fields else None
        result = write_cmb_anomaly_report(
            args.run_dir,
            args.out,
            source_dir=args.source_dir,
            fields=fields,
            low_lmax=args.low_lmax,
            parity_lmax=args.parity_lmax,
            s12_lmax=args.s12_lmax,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "comparable-data":
        from oph_fpe.cosmology.comparable_data import write_comparable_data_package

        result = write_comparable_data_package([*args.run_dir, *args.include], args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "recompute-object-chart":
        from oph_fpe.bulk.record_to_h3 import recompute_object_chart_from_saved_run

        result = recompute_object_chart_from_saved_run(
            run_dir=args.run_dir,
            h3_report_path=args.h3_report,
            out_path=args.out,
            shuffle_control_count=args.shuffle_count,
            incidence_mode=args.incidence_mode,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "run-galaxy-static":
        from oph_fpe.cosmology.galaxy_static import write_static_galaxy_measurement_report

        result = write_static_galaxy_measurement_report(
            args.dataset,
            args.out_dir,
            a0_initial=args.a0_initial,
            lambda_initial=args.lambda_initial,
            min_points=args.min_points,
            min_galaxies=args.min_galaxies,
            physical_claim=not bool(args.diagnostic_only),
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "run-viewer":
        from oph_fpe.viz import write_run_viewer

        result = write_run_viewer(args.run_dir, args.out, max_screen_points=args.max_screen_points)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "controlled-defect-assay":
        from oph_fpe.defects.controlled_assay import write_controlled_s3_particle_assay_report

        result = write_controlled_s3_particle_assay_report(
            args.out,
            patch_count=args.patch_count,
            observation_count=args.observation_count,
            support_node_count=args.support_node_count,
            holonomy=args.holonomy,
            cycle_stride=args.cycle_stride,
            max_support_fraction=args.max_support_fraction,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "caps-to-h3-minimal":
        from oph_fpe.bulk.cap_profile_geometry import caps_to_h3_minimal_receipt

        theta_values = [float(value.strip()) for value in args.theta_values.split(",") if value.strip()]
        result = caps_to_h3_minimal_receipt(
            axis_count=args.axis_count,
            theta_values=theta_values,
            object_count=args.object_count,
            object_radius=args.object_radius,
            fit_radius=args.fit_radius,
            softness=args.softness,
            restarts=args.restarts,
            seed=args.seed,
            max_median_error=args.max_median_error,
        )
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(json.dumps(result, indent=2, default=str))
        return 0
    raise AssertionError(args.command)


def _csv_values(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


if __name__ == "__main__":
    raise SystemExit(main())
