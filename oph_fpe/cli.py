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

    universe_parser = subparsers.add_parser(
        "run-oph-universe",
        help="run the canonical theorem-following OPH universe pipeline",
    )
    universe_parser.add_argument("--config", required=True, type=Path)
    universe_parser.add_argument("--out-dir", default=Path("runs"), type=Path)
    universe_parser.add_argument("--run-id", default=None)
    universe_parser.add_argument("--source-run-dir", default=None, type=Path)
    universe_parser.add_argument("--skip-base-run", action="store_true")
    universe_parser.add_argument("--max-screen-points", default=5000, type=int)
    universe_parser.add_argument("--max-observers", default=128, type=int)
    universe_parser.add_argument("--max-h3-objects", default=512, type=int)
    universe_parser.add_argument("--skip-visualizations", action="store_true")

    shape_parser = subparsers.add_parser("shape-dodeca-smoke", help="run declared Shape dodecahedral substrate witness")
    shape_parser.add_argument("--config", default=None, type=Path)
    shape_parser.add_argument("--out-dir", default=Path("runs"), type=Path)
    shape_parser.add_argument("--cycles", default=None, type=int)
    shape_parser.add_argument("--repair-rate", default=None, type=float)
    shape_parser.add_argument("--seed", default=None, type=int)
    shape_parser.add_argument("--ell-max", default=None, type=int)
    shape_parser.add_argument("--particle-energy-threshold", default=None, type=float)

    shape_alias_parser = subparsers.add_parser(
        "run-shape-substrate",
        help="alias for shape-dodeca-smoke",
    )
    shape_alias_parser.add_argument("--config", default=None, type=Path)
    shape_alias_parser.add_argument("--out-dir", default=Path("runs"), type=Path)
    shape_alias_parser.add_argument("--cycles", default=None, type=int)
    shape_alias_parser.add_argument("--repair-rate", default=None, type=float)
    shape_alias_parser.add_argument("--seed", default=None, type=int)
    shape_alias_parser.add_argument("--ell-max", default=None, type=int)
    shape_alias_parser.add_argument("--particle-energy-threshold", default=None, type=float)

    shape_ensemble_parser = subparsers.add_parser("shape-ensemble", help="run declared Shape substrate seeds")
    shape_ensemble_parser.add_argument("--config", required=True, type=Path)
    shape_ensemble_parser.add_argument("--out-dir", default=Path("runs"), type=Path)
    shape_ensemble_parser.add_argument("--seeds", required=True, help="comma-separated seed list")
    shape_ensemble_parser.add_argument("--workers", type=int, default=1)

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

    exact_cmb_camb_parser = subparsers.add_parser(
        "oph-exact-cmb-camb",
        help="run CAMB TT transfer for the exact OPH CMB scalar target branch",
    )
    exact_cmb_camb_parser.add_argument("--benchmark", required=True, type=Path)
    exact_cmb_camb_parser.add_argument("--out", required=True, type=Path)
    exact_cmb_camb_parser.add_argument("--source-dir", default=None, type=Path)
    exact_cmb_camb_parser.add_argument("--lmax", default=2600, type=int)
    exact_cmb_camb_parser.add_argument("--label", default="Planck2018_TT_binned")

    finite_clock_camb_parser = subparsers.add_parser(
        "finite-repair-clock-cmb-camb",
        help="run CAMB TT transfer from a simulator-derived finite repair-clock report",
    )
    finite_clock_camb_parser.add_argument("--finite-clock-report", required=True, type=Path)
    finite_clock_camb_parser.add_argument("--benchmark", required=True, type=Path)
    finite_clock_camb_parser.add_argument("--out", required=True, type=Path)
    finite_clock_camb_parser.add_argument("--source-dir", default=None, type=Path)
    finite_clock_camb_parser.add_argument("--lmax", default=2600, type=int)
    finite_clock_camb_parser.add_argument("--label", default="Planck2018_TT_binned")

    selector_elimination_parser = subparsers.add_parser(
        "oph-cmb-selector-elimination",
        help="write the OPH CMB v1.5 selector-elimination target/certificate audit",
    )
    selector_elimination_parser.add_argument(
        "--source-dir",
        default=Path("/Users/muellerberndt/Projects/oph-meta/cosmology/correspondence/cmb/7"),
        type=Path,
    )
    selector_elimination_parser.add_argument("--out", required=True, type=Path)
    selector_elimination_parser.add_argument("--p-value", default=None, type=float)

    boltzmann_input_parser = subparsers.add_parser(
        "oph-boltzmann-inputs",
        help="export CDM-limit and OPH diagnostic anomaly-stress inputs for future CAMB/CLASS modules",
    )
    boltzmann_input_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    boltzmann_input_parser.add_argument("--include", nargs="*", default=[], type=Path)
    boltzmann_input_parser.add_argument("--out", required=True, type=Path)

    ba_parent_parser = subparsers.add_parser(
        "b-a-parent-report",
        help="emit the finite-collar B_A(k,a) parent diagnostic from cached stress reports",
    )
    ba_parent_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    ba_parent_parser.add_argument("--include", nargs="*", default=[], type=Path)
    ba_parent_parser.add_argument("--out", required=True, type=Path)
    ba_parent_parser.add_argument(
        "--a-grid",
        default=None,
        help="optional comma-separated scale-factor grid; defaults to source stress report grid",
    )
    ba_parent_parser.add_argument(
        "--k-grid",
        default=None,
        help=(
            "optional comma-separated k grid. Current report-backed mode treats values as "
            "inverse-cap-opening-angle proxies unless a calibrated k map exists"
        ),
    )
    ba_parent_parser.add_argument("--eps", default=1.0e-3, type=float)
    ba_parent_parser.add_argument("--modes-per-k", default=8, type=int)
    ba_parent_parser.add_argument("--seeds", default="0,1,2,3")

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
    screen_power_parser.add_argument(
        "--reference-mode",
        choices=["auto", "planck-fallback", "simulator-best"],
        default="auto",
        help=(
            "auto uses simulator eta only when it passes the Planck-target diagnostic; "
            "simulator-best exports the best finite fit even when it fails, with claim gates closed"
        ),
    )
    screen_power_parser.add_argument("--primordial-k-count", default=256, type=int)
    screen_power_parser.add_argument("--primordial-k-min", default=1.0e-4, type=float)
    screen_power_parser.add_argument("--primordial-k-max", default=1.0, type=float)

    maxent_green_parser = subparsers.add_parser(
        "maxent-green-spectrum",
        help="write the OPH paper-side MaxEnt Green-spectrum CMB source certificate",
    )
    maxent_green_parser.add_argument("--out", required=True, type=Path)
    maxent_green_parser.add_argument(
        "--source-dir",
        default=Path("/Users/muellerberndt/Projects/oph-meta/cosmology/correspondence/cmb/7"),
        type=Path,
    )
    maxent_green_parser.add_argument("--patch-count", default=262_144, type=int)
    maxent_green_parser.add_argument("--ell-max", default=256, type=int)
    maxent_green_parser.add_argument("--kappa-rep", default=None, type=float)
    maxent_green_parser.add_argument("--amplitude", default=1.0, type=float)
    maxent_green_parser.add_argument("--mu", default=0.0, type=float)
    maxent_green_parser.add_argument("--primordial-k-count", default=256, type=int)
    maxent_green_parser.add_argument("--primordial-k-min", default=1.0e-4, type=float)
    maxent_green_parser.add_argument("--primordial-k-max", default=1.0, type=float)

    repair_clock_parser = subparsers.add_parser(
        "repair-clock-report",
        help="audit whether finite repair traces derive the OPH kappa_rep=e scalar clock",
    )
    repair_clock_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    repair_clock_parser.add_argument("--include", nargs="*", default=[], type=Path)
    repair_clock_parser.add_argument("--out", required=True, type=Path)
    repair_clock_parser.add_argument(
        "--cycle-time-normalization",
        default=None,
        type=float,
        help="optional predeclared repair-time per simulator cycle; omitted rows are diagnostic only",
    )
    repair_clock_parser.add_argument("--r2-threshold", default=0.85, type=float)
    repair_clock_parser.add_argument("--relative-tolerance", default=0.05, type=float)

    scalar_semigroup_parser = subparsers.add_parser(
        "scalar-repair-semigroup",
        help="write the scalar repair-semigroup gap audit used by the exact OPH CMB kappa clock",
    )
    scalar_semigroup_parser.add_argument("--out", required=True, type=Path)
    scalar_semigroup_parser.add_argument("--dimension", default=33, type=int)
    scalar_semigroup_parser.add_argument("--kappa-rep", default=2.718281828459045, type=float)
    scalar_semigroup_parser.add_argument(
        "--source",
        default="declared_euler_repair_time_target",
        choices=["declared_euler_repair_time_target", "finite_state_transition_matrix", "external_theorem_certificate"],
    )
    scalar_semigroup_parser.add_argument("--finite-lattice-derived", action="store_true")
    scalar_semigroup_parser.add_argument("--matrix-source", default=None)

    finite_transition_clock_parser = subparsers.add_parser(
        "finite-repair-transition-clock",
        help="derive a scalar repair-clock transition matrix from observer-visible transition histories",
    )
    finite_transition_clock_parser.add_argument("--run-dir", required=True, type=Path)
    finite_transition_clock_parser.add_argument("--out", required=True, type=Path)
    finite_transition_clock_parser.add_argument(
        "--packet-fields",
        default="checkpoint_class,stable_flag,s3_sector_class,repair_load_bucket",
        help="comma-separated transition-history packet fields for the quotient alphabet",
    )
    finite_transition_clock_parser.add_argument(
        "--primary-matrix",
        default="raw_empirical",
        choices=["raw_empirical", "reversible_empirical"],
    )
    finite_transition_clock_parser.add_argument("--repair-step-time", default=1.0, type=float)
    finite_transition_clock_parser.add_argument(
        "--clock-normalization-source",
        default="declared_cli_value",
        help=(
            "label for the repair-step time source; theorem-grade sources are "
            "paper_theorem_predeclared or finite_theorem_predeclared"
        ),
    )
    finite_transition_clock_parser.add_argument("--weight-field", default="transition_history_mean_modal_mass")

    finite_transition_sweep_parser = subparsers.add_parser(
        "finite-repair-transition-sweep",
        help="sweep observer-visible transition-history quotients for the scalar repair-clock matrix",
    )
    finite_transition_sweep_parser.add_argument("--run-dir", required=True, type=Path)
    finite_transition_sweep_parser.add_argument("--out", required=True, type=Path)
    finite_transition_sweep_parser.add_argument(
        "--repair-step-times",
        default="1.0",
        help="comma-separated declared repair-step times to audit",
    )
    finite_transition_sweep_parser.add_argument(
        "--primary-matrices",
        default="raw_empirical,reversible_empirical",
        help="comma-separated primary matrices to audit",
    )
    finite_transition_sweep_parser.add_argument(
        "--clock-normalization-source",
        default="sweep_declared_values",
        help="label for the repair-step time source used by this sweep",
    )
    finite_transition_sweep_parser.add_argument("--weight-field", default="transition_history_mean_modal_mass")

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

    fossil_parser = subparsers.add_parser(
        "fossil-spectrum-report",
        help="audit time-resolved finite-screen spectra from harmonic_time_trace.npz",
    )
    fossil_parser.add_argument("--run-dir", required=True, type=Path)
    fossil_parser.add_argument("--out", required=True, type=Path)
    fossil_parser.add_argument("--fields", default=None, help="comma-separated trace fields; defaults to all")
    fossil_parser.add_argument("--ell-min", default=8.0, type=float)
    fossil_parser.add_argument("--ell-max", default=32.0, type=float)

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

    h0s8_cert_parser = subparsers.add_parser(
        "h0s8-lane8-certificate",
        help="write the Lane-8 record/provenance certificate stack for H0/S8 diagnostics",
    )
    h0s8_cert_parser.add_argument("--out", required=True, type=Path)

    cnb_parser = subparsers.add_parser(
        "oph-cnb-neutrinos",
        help="write the OPH-CnuB relic-neutrino background and weak-lensing projection target report",
    )
    cnb_parser.add_argument(
        "--source-dir",
        default=Path("/Users/muellerberndt/Projects/oph-meta/cosmology/correspondence/neutrinos"),
        type=Path,
    )
    cnb_parser.add_argument("--out", required=True, type=Path)
    cnb_parser.add_argument("--delta-neff-coh", default=0.0, type=float)

    compressed_parser = subparsers.add_parser(
        "oph-compressed-likelihood",
        help="write the OPH compressed CMB/BAO/growth/S8 reference diagnostic",
    )
    compressed_parser.add_argument("--out", required=True, type=Path)

    screen_capacity_parser = subparsers.add_parser(
        "screen-capacity-report",
        help="write OPH cosmic record/screen-capacity closure readout and regulator-scale comparison",
    )
    screen_capacity_parser.add_argument("--out", required=True, type=Path)
    screen_capacity_parser.add_argument(
        "--n-crc",
        default=None,
        type=float,
        help="direct OPH global screen-capacity closure value; if omitted, use R_dS/l_P readout",
    )
    screen_capacity_parser.add_argument("--r-ds-m", default=1.66e26, type=float)
    screen_capacity_parser.add_argument("--planck-length-m", default=1.616e-35, type=float)
    screen_capacity_parser.add_argument(
        "--regulator-patch-counts",
        default="4096,65536,262144,1048576",
        help="comma-separated finite regulator patch counts to compare against N_scr",
    )

    capacity_proxy_parser = subparsers.add_parser(
        "capacity-readback-proxy-report",
        help="write finite-regulator proxy rows for the OPH F(N) capacity readback audit",
    )
    capacity_proxy_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    capacity_proxy_parser.add_argument("--out", required=True, type=Path)
    capacity_proxy_parser.add_argument("--p-value", default=None, type=float)
    capacity_proxy_parser.add_argument(
        "--n-crc",
        default=None,
        type=float,
        help="global OPH capacity value used only for finite-regulator fraction reporting",
    )
    capacity_proxy_parser.add_argument("--max-observer-views", default=4096, type=int)

    scale_bridge_parser = subparsers.add_parser(
        "scale-bridge-report",
        help="write OPH P/N dimensionless-invariant and independent-scale-bridge gate report",
    )
    scale_bridge_parser.add_argument("--out", required=True, type=Path)
    scale_bridge_parser.add_argument("--p-star", default=None, type=float)
    scale_bridge_parser.add_argument(
        "--n-star",
        default=None,
        type=float,
        help="OPH capacity N for dimensionless P/N invariants; defaults to screen_capacity.DEFAULT_N_CRC",
    )
    scale_bridge_parser.add_argument(
        "--lambda-star-m2",
        default=None,
        type=float,
        help="independent Lambda_star in m^-2; requires --n-star or the default N_CRC",
    )
    scale_bridge_parser.add_argument(
        "--b-ell-m2",
        default=None,
        type=float,
        help="independent B_ell in m^-2; if supplied, gives ell_star^2 and G_SI",
    )
    scale_bridge_parser.add_argument("--source", default="cli_scale_bridge_report")

    pn_resonance_parser = subparsers.add_parser(
        "pn-resonance-report",
        help="write the paper-faithful P/N resonance replay and promotion-gate report",
    )
    pn_resonance_parser.add_argument("--out", required=True, type=Path)
    pn_resonance_parser.add_argument(
        "--n-source",
        default="ew-bridge",
        choices=("ew-bridge", "screen-capacity-default", "direct"),
        help="capacity source for the replay; default uses N=pi*exp(6*pi/(P*alpha_U))",
    )
    pn_resonance_parser.add_argument("--n-star", default=None, type=float)
    pn_resonance_parser.add_argument("--p-star", default=None, type=float)
    pn_resonance_parser.add_argument("--alpha-u", default=None, type=float)
    pn_resonance_parser.add_argument("--b-ell-m2", default=None, type=float)
    pn_resonance_parser.add_argument("--lambda-star-m2", default=None, type=float)
    pn_resonance_parser.add_argument("--repair-rounds", default=24, type=int)
    pn_resonance_parser.add_argument(
        "--regulator-patch-counts",
        default="4096,65536,262144,1048576",
        help="comma-separated finite regulator patch counts for scale/capacity sidecars",
    )
    pn_resonance_parser.add_argument("--run-dir", nargs="*", default=None, type=Path)
    pn_resonance_parser.add_argument("--source", default="cli_pn_resonance_report")

    silence_parser = subparsers.add_parser(
        "silence-to-observation-report",
        help="write the finite scale-compressed P/N silence-to-observation witness for a run",
    )
    silence_parser.add_argument("--run-dir", required=True, type=Path)
    silence_parser.add_argument("--out", default=None, type=Path)
    silence_parser.add_argument(
        "--n-source",
        default="ew-bridge",
        choices=("ew-bridge", "screen-capacity-default", "direct"),
        help="capacity source for the P/N replay; default uses N=pi*exp(6*pi/(P*alpha_U))",
    )
    silence_parser.add_argument("--n-star", default=None, type=float)
    silence_parser.add_argument("--p-star", default=None, type=float)
    silence_parser.add_argument("--alpha-u", default=None, type=float)
    silence_parser.add_argument("--repair-rounds", default=24, type=int)
    silence_parser.add_argument("--source", default="cli_silence_to_observation_report")

    pgk_parser = subparsers.add_parser(
        "positive-geometry-kernel-report",
        help="run the OPH positive-geometry kernel checker and emit fail-closed simulator gates",
    )
    pgk_parser.add_argument("--out", required=True, type=Path)
    pgk_parser.add_argument(
        "--manifest",
        default=None,
        type=Path,
        help="PGK manifest; defaults to the bundled A614 geometry pilot",
    )
    pgk_parser.add_argument(
        "--pgk-root",
        default=None,
        type=Path,
        help="directory containing pgk_reference.py; defaults to the amplituhedron bundle",
    )
    pgk_parser.add_argument("--source", default="cli_positive_geometry_kernel_report")

    no_g_clock_parser = subparsers.add_parser(
        "no-g-clock-bridge-report",
        help="write the OPH R_gamma no-G clock-bridge checksum and source-predictive gate audit",
    )
    no_g_clock_parser.add_argument("--out", required=True, type=Path)
    no_g_clock_parser.add_argument("--epsilon-cs", default=None, type=float)
    no_g_clock_parser.add_argument("--nu-cs-hz", default=None, type=float)
    no_g_clock_parser.add_argument(
        "--dependency-graph",
        default=None,
        type=Path,
        help="JSON object mapping R_gamma/gamma_star nodes to their source dependencies",
    )
    no_g_clock_parser.add_argument("--source", default="compact_proof_R_gamma_display")
    no_g_clock_parser.add_argument("--public-dependency-graph", action="store_true")
    no_g_clock_parser.add_argument("--source-readback-map-emitted", action="store_true")
    no_g_clock_parser.add_argument("--contraction-certificate", action="store_true")
    no_g_clock_parser.add_argument("--residual-certificate", action="store_true")

    repair_scale_parser = subparsers.add_parser(
        "repair-scale-closure",
        help="write Maarten/OPH 24-round repair-depth scale-closure diagnostic",
    )
    repair_scale_parser.add_argument("--out", required=True, type=Path)
    repair_scale_parser.add_argument(
        "--n-crc",
        default=None,
        type=float,
        help="direct OPH global screen-capacity value; defaults to screen_capacity.DEFAULT_N_CRC",
    )
    repair_scale_parser.add_argument("--repair-rounds", default=24, type=int)
    repair_scale_parser.add_argument(
        "--regulator-patch-counts",
        default="4096,65536,262144,1048576,1000000000",
        help="comma-separated finite regulator patch counts for effective round-depth estimates",
    )

    scale_compressed_parser = subparsers.add_parser(
        "scale-compressed-repair",
        help="run the logical 24-round scale-compressed OPH repair branch preview",
    )
    scale_compressed_parser.add_argument("--out", required=True, type=Path)
    scale_compressed_parser.add_argument("--repair-rounds", default=24, type=int)
    scale_compressed_parser.add_argument("--object-count", default=48, type=int)
    scale_compressed_parser.add_argument("--particle-count", default=6, type=int)
    scale_compressed_parser.add_argument("--cap-axis-count", default=24, type=int)
    scale_compressed_parser.add_argument("--ell-max", default=256, type=int)
    scale_compressed_parser.add_argument("--seed", default=20260610, type=int)
    scale_compressed_parser.add_argument("--planck-tt", default=None, type=Path)

    scale_compressed_camb_parser = subparsers.add_parser(
        "scale-compressed-cmb-camb",
        help="run CAMB TT transfer from a scale-compressed repair branch report",
    )
    scale_compressed_camb_parser.add_argument("--scale-report", required=True, type=Path)
    scale_compressed_camb_parser.add_argument("--benchmark", required=True, type=Path)
    scale_compressed_camb_parser.add_argument("--out", required=True, type=Path)
    scale_compressed_camb_parser.add_argument("--lmax", default=2600, type=int)
    scale_compressed_camb_parser.add_argument("--label", default="Planck2018_TT_binned")

    inflation_cmb_parser = subparsers.add_parser(
        "oph-inflation-cmb-bridge",
        help="write the OPH P/48 screen-spectrum and CMB v0.4/v0.5 diagnostic bridge report",
    )
    inflation_cmb_parser.add_argument("--source-dir", default=None, type=Path)
    inflation_cmb_parser.add_argument("--out", required=True, type=Path)

    inflation_cert_parser = subparsers.add_parser(
        "inflation-certificates",
        help="validate OPH inflation/CMB finite certificate artifacts and write schemas/templates",
    )
    inflation_cert_parser.add_argument("--cert-dir", default=None, type=Path)
    inflation_cert_parser.add_argument(
        "--source-path",
        default=Path("/Users/muellerberndt/Projects/oph-meta/cosmology/correspondence/inflation/3/comms.md"),
        type=Path,
    )
    inflation_cert_parser.add_argument("--out", required=True, type=Path)

    finite_cert_parser = subparsers.add_parser(
        "finite-certificates",
        help="compute OPH finite cosmology certificates from release/collar/repair finite input JSON",
    )
    finite_cert_parser.add_argument("--input", default=None, type=Path)
    finite_cert_parser.add_argument("--out", required=True, type=Path)
    finite_cert_parser.add_argument(
        "--toy",
        action="store_true",
        help="write the built-in toy certificate bundle; validates format only and is not a physical output",
    )

    finite_cert_from_run_parser = subparsers.add_parser(
        "finite-certificates-from-run",
        help="build proxy finite cosmology certificates from cached OPH-FPE run receipts",
    )
    finite_cert_from_run_parser.add_argument("--run-dir", required=True, type=Path)
    finite_cert_from_run_parser.add_argument("--out", required=True, type=Path)

    parent_collar_parser = subparsers.add_parser(
        "parent-collar-ladder",
        help="aggregate cached collar Markov reports into a finite parent-collar recovery ladder",
    )
    parent_collar_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    parent_collar_parser.add_argument("--include", nargs="*", default=[], type=Path)
    parent_collar_parser.add_argument("--out", required=True, type=Path)

    ba_kernel_parser = subparsers.add_parser(
        "b-a-kernel-paired",
        help="estimate a paired finite-difference B_A(k,a) kernel from base/perturbed CSV rows",
    )
    ba_kernel_parser.add_argument("--base", required=True, type=Path)
    ba_kernel_parser.add_argument("--perturbed", required=True, type=Path)
    ba_kernel_parser.add_argument("--control", default=None, type=Path)
    ba_kernel_parser.add_argument("--out", required=True, type=Path)
    ba_kernel_parser.add_argument("--min-good-rows", default=3, type=int)
    ba_kernel_parser.add_argument("--min-sample-count", default=16, type=int)

    ba_kernel_from_parent_parser = subparsers.add_parser(
        "b-a-kernel-from-parent",
        help="promote a paired parent-collar B_A report into a kernel candidate audit",
    )
    ba_kernel_from_parent_parser.add_argument("--parent-report", required=True, type=Path)
    ba_kernel_from_parent_parser.add_argument("--out", required=True, type=Path)

    ba_kernel_refinement_parser = subparsers.add_parser(
        "b-a-kernel-refinement",
        help="audit B_A kernel stability across finite regulator patch counts",
    )
    ba_kernel_refinement_parser.add_argument("--report", nargs="+", required=True, type=Path)
    ba_kernel_refinement_parser.add_argument("--out", required=True, type=Path)

    physical_cmb_inputs_parser = subparsers.add_parser(
        "derive-physical-cmb-inputs",
        help="assemble the hard physical CMB input contract from finite OPH-FPE run receipts",
    )
    physical_cmb_inputs_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    physical_cmb_inputs_parser.add_argument("--include", nargs="*", default=[], type=Path)
    physical_cmb_inputs_parser.add_argument("--out", required=True, type=Path)

    physical_cmb_no_data_parser = subparsers.add_parser(
        "physical-cmb-no-data-use-receipt",
        help="write the no-measurement-data firewall receipt for physical CMB input assembly",
    )
    physical_cmb_no_data_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    physical_cmb_no_data_parser.add_argument("--include", nargs="*", default=[], type=Path)
    physical_cmb_no_data_parser.add_argument("--out", required=True, type=Path)

    physical_cmb_promotion_parser = subparsers.add_parser(
        "physical-cmb-promotion-audit",
        help="write a compact audit of blockers to physical CMB prediction promotion",
    )
    physical_cmb_promotion_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    physical_cmb_promotion_parser.add_argument("--include", nargs="*", default=[], type=Path)
    physical_cmb_promotion_parser.add_argument("--out", required=True, type=Path)

    physical_cmb_frontier_parser = subparsers.add_parser(
        "physical-cmb-frontier",
        help="write a gate-by-gate frontier between CMB diagnostics and physical prediction status",
    )
    physical_cmb_frontier_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    physical_cmb_frontier_parser.add_argument("--include", nargs="*", default=[], type=Path)
    physical_cmb_frontier_parser.add_argument("--out", required=True, type=Path)

    physical_cmb_output_parser = subparsers.add_parser(
        "physical-cmb-output-comparison",
        help="aggregate measurement-comparable CMB TT output metrics without promoting the hard CMB gate",
    )
    physical_cmb_output_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    physical_cmb_output_parser.add_argument("--include", nargs="*", default=[], type=Path)
    physical_cmb_output_parser.add_argument("--out", required=True, type=Path)

    observer_consensus_bulk_parser = subparsers.add_parser(
        "observer-consensus-bulk-readout",
        help="write observer-local self-reading and theorem-assisted consensus 3D bulk readouts",
    )
    observer_consensus_bulk_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    observer_consensus_bulk_parser.add_argument("--include", nargs="*", default=[], type=Path)
    observer_consensus_bulk_parser.add_argument("--out", required=True, type=Path)
    observer_consensus_bulk_parser.add_argument("--observer-sample-count", default=12, type=int)
    observer_consensus_bulk_parser.add_argument("--object-sample-count", default=24, type=int)

    official_planck_readiness_parser = subparsers.add_parser(
        "official-planck-readiness",
        help="write an environment readiness receipt for official Planck likelihood execution",
    )
    official_planck_readiness_parser.add_argument("--out", required=True, type=Path)

    finite_collar_boltzmann_parser = subparsers.add_parser(
        "finite-collar-boltzmann-bundle",
        help="assemble finite-collar rho_A/B_A/Gamma_rec diagnostics for the Boltzmann/CMB bridge",
    )
    finite_collar_boltzmann_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    finite_collar_boltzmann_parser.add_argument("--include", nargs="*", default=[], type=Path)
    finite_collar_boltzmann_parser.add_argument("--out", required=True, type=Path)

    finite_collar_projection_parser = subparsers.add_parser(
        "finite-collar-cmb-projection",
        help="project finite-collar B_A/rho/Gamma diagnostics onto external-fiducial CMB ell/k axes",
    )
    finite_collar_projection_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    finite_collar_projection_parser.add_argument("--include", nargs="*", default=[], type=Path)
    finite_collar_projection_parser.add_argument("--out", required=True, type=Path)
    finite_collar_projection_parser.add_argument("--chi-star-mpc", default=13850.0, type=float)
    finite_collar_projection_parser.add_argument("--h", default=0.6736, type=float)
    finite_collar_projection_parser.add_argument(
        "--ell-mapping",
        default="pi_over_theta",
        choices=("pi_over_theta", "one_over_theta"),
    )

    scalar_quotient_parser = subparsers.add_parser(
        "scalar-quotient-report",
        help="write the finite observer-visible scalar/geometric quotient report for CMB input gating",
    )
    scalar_quotient_parser.add_argument("--run-dir", required=True, type=Path)
    scalar_quotient_parser.add_argument("--out", default=None, type=Path)
    scalar_quotient_parser.add_argument("--target-ell-ir", default=32, type=int)
    scalar_quotient_parser.add_argument("--bins", default=8, type=int)

    scalar_cert_parser = subparsers.add_parser(
        "emit-scalar-release-certificate",
        help="emit a proxy scalar-release certificate from a cached collar Markov run",
    )
    scalar_cert_parser.add_argument("--run-dir", required=True, type=Path)
    scalar_cert_parser.add_argument("--out", required=True, type=Path)
    scalar_cert_parser.add_argument("--kappa-rel", default=1.0, type=float)
    scalar_cert_parser.add_argument(
        "--source-path",
        default=Path("/Users/muellerberndt/Projects/oph-meta/cosmology/correspondence/inflation/3/comms.md"),
        type=Path,
    )

    edge_cert_parser = subparsers.add_parser(
        "emit-edge-center-certificate",
        help="emit the P/48 edge-center finite certificate into a certificate directory",
    )
    edge_cert_parser.add_argument("--out", required=True, type=Path)
    edge_cert_parser.add_argument("--p-value", default=None, type=float)
    edge_cert_parser.add_argument(
        "--source-path",
        default=Path("/Users/muellerberndt/Projects/oph-meta/cosmology/correspondence/inflation/3/comms.md"),
        type=Path,
    )

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

    cmb_fossil_parser = subparsers.add_parser(
        "cmb-fossil-bridge",
        help="write the OPH-CET analytic screen covariance to CMB fossil bridge diagnostic",
    )
    cmb_fossil_parser.add_argument("--planck-tt", default=None, type=Path)
    cmb_fossil_parser.add_argument("--out-dir", required=True, type=Path)
    cmb_fossil_parser.add_argument("--ell-max", default=2600, type=int)
    cmb_fossil_parser.add_argument("--eta", default=0.0351588569692228, type=float)
    cmb_fossil_parser.add_argument("--q-ir", default=0.25, type=float)
    cmb_fossil_parser.add_argument("--ell-ir", default=32.0, type=float)
    cmb_fossil_parser.add_argument("--eps-p", default=0.0, type=float)
    cmb_fossil_parser.add_argument("--ell-p", default=30.0, type=float)
    cmb_fossil_parser.add_argument("--ell-cap", default=3000.0, type=float)

    comparable_parser = subparsers.add_parser(
        "comparable-data",
        help="export current comparable diagnostic data from run receipts",
    )
    comparable_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    comparable_parser.add_argument("--include", nargs="*", default=[], type=Path)
    comparable_parser.add_argument("--out", required=True, type=Path)

    measurement_pack_parser = subparsers.add_parser(
        "export-measurement-pack",
        help="export standard measurement-facing CSV/JSON tables from one or more run directories",
    )
    measurement_pack_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    measurement_pack_parser.add_argument("--include", nargs="*", default=[], type=Path)
    measurement_pack_parser.add_argument("--out", required=True, type=Path)

    bulk_proof_parser = subparsers.add_parser(
        "bulk-proof-certificate",
        help="write a tiered OPH 3D-bulk/chart/CMB/particle proof certificate from run receipts",
    )
    bulk_proof_parser.add_argument("--run-dir", required=True, type=Path)
    bulk_proof_parser.add_argument("--out", default=None, type=Path)

    strict_neutral_parser = subparsers.add_parser(
        "strict-neutral-bulk-report",
        help="write strict neutral observer-record bulk diagnostics from observer_views.jsonl",
    )
    strict_neutral_parser.add_argument("--run-dir", required=True, type=Path)
    strict_neutral_parser.add_argument("--out", default=None, type=Path)
    strict_neutral_parser.add_argument("--seed", default=1, type=int)
    strict_neutral_parser.add_argument("--max-model-points", default=512, type=int)
    strict_neutral_parser.add_argument("--planted-control-points", default=160, type=int)

    strict_neutral_object_parser = subparsers.add_parser(
        "strict-neutral-object-bulk-report",
        help="write strict neutral object-bulk diagnostics from observer_views.jsonl",
    )
    strict_neutral_object_parser.add_argument("--run-dir", required=True, type=Path)
    strict_neutral_object_parser.add_argument("--out", default=None, type=Path)
    strict_neutral_object_parser.add_argument("--seed", default=1, type=int)
    strict_neutral_object_parser.add_argument("--min-objects", default=16, type=int)
    strict_neutral_object_parser.add_argument("--min-observers-per-object", default=3, type=int)
    strict_neutral_object_parser.add_argument("--max-observer-fraction-per-object", default=0.35, type=float)
    strict_neutral_object_parser.add_argument("--max-model-points", default=192, type=int)
    strict_neutral_object_parser.add_argument("--heldout-fraction", default=0.25, type=float)

    neutral_profile_parser = subparsers.add_parser(
        "neutral-profile-audit",
        help="write bounded neutral-distance feature-profile diagnostics from observer_views.jsonl",
    )
    neutral_profile_parser.add_argument("--run-dir", required=True, type=Path)
    neutral_profile_parser.add_argument("--out", default=None, type=Path)
    neutral_profile_parser.add_argument("--seed", default=1, type=int)
    neutral_profile_parser.add_argument("--sample-count", default=256, type=int)
    neutral_profile_parser.add_argument("--max-model-points", default=128, type=int)
    neutral_profile_parser.add_argument(
        "--profiles",
        default=None,
        help="comma-separated neutral profile names to audit; defaults to all profiles",
    )

    prime_response_parser = subparsers.add_parser(
        "attach-prime-geometric-response",
        help="attach cached support-visible modular-response spectra to observer_views.jsonl",
    )
    prime_response_parser.add_argument("--run-dir", required=True, type=Path)
    prime_response_parser.add_argument("--out", default=None, type=Path)
    prime_response_parser.add_argument("--spectrum-width", default=64, type=int)
    prime_response_parser.add_argument("--component-bins", default=8, type=int)
    prime_response_parser.add_argument("--no-backup", action="store_true")

    prime_rank_parser = subparsers.add_parser(
        "neutral-prime-rank-sweep",
        help="write a diagnostic rank sweep over the prime-geometric modular response spectrum",
    )
    prime_rank_parser.add_argument("--run-dir", required=True, type=Path)
    prime_rank_parser.add_argument("--out", default=None, type=Path)
    prime_rank_parser.add_argument("--ranks", default="2,3,4,5,6,7,8,9,10,11,12,13,14,15,16")
    prime_rank_parser.add_argument("--seed", default=1, type=int)
    prime_rank_parser.add_argument("--sample-count", default=256, type=int)
    prime_rank_parser.add_argument("--max-model-points", default=128, type=int)

    prime_rank_refinement_parser = subparsers.add_parser(
        "neutral-prime-rank-refinement",
        help="aggregate prime-geometric rank-sweep reports across regulator sizes",
    )
    prime_rank_refinement_parser.add_argument("--report", nargs="+", required=True, type=Path)
    prime_rank_refinement_parser.add_argument("--out", required=True, type=Path)

    neutral_bulk_audit_parser = subparsers.add_parser(
        "neutral-3d-bulk-audit",
        help="write a compact audit of blockers to strict neutral 3D bulk promotion",
    )
    neutral_bulk_audit_parser.add_argument("--report", nargs="+", required=True, type=Path)
    neutral_bulk_audit_parser.add_argument("--out", required=True, type=Path)

    neutral_rank_selector_parser = subparsers.add_parser(
        "neutral-independent-rank-selector-audit",
        help="audit independent rank-selector evidence for neutral 3D bulk promotion",
    )
    neutral_rank_selector_parser.add_argument("--report", nargs="+", required=True, type=Path)
    neutral_rank_selector_parser.add_argument("--out", required=True, type=Path)

    neutral_frontier_parser = subparsers.add_parser(
        "strict-neutral-bulk-frontier",
        help="write a strict-neutral-bulk proof-frontier report from neutral audit receipts",
    )
    neutral_frontier_parser.add_argument("--report", nargs="+", required=True, type=Path)
    neutral_frontier_parser.add_argument("--out", required=True, type=Path)

    overlap_neutral_parser = subparsers.add_parser(
        "neutral-overlap-control-report",
        help="write observer-overlap native neutral negative-control diagnostics from observer_views.jsonl",
    )
    overlap_neutral_parser.add_argument("--run-dir", required=True, type=Path)
    overlap_neutral_parser.add_argument("--out", default=None, type=Path)
    overlap_neutral_parser.add_argument("--seed", default=1, type=int)
    overlap_neutral_parser.add_argument("--max-model-points", default=256, type=int)

    overlap_graph_parser = subparsers.add_parser(
        "neutral-overlap-graph-geometry",
        help="write observer-overlap graph geometry diagnostics from observer_views.jsonl",
    )
    overlap_graph_parser.add_argument("--run-dir", required=True, type=Path)
    overlap_graph_parser.add_argument("--out", default=None, type=Path)
    overlap_graph_parser.add_argument("--seed", default=1, type=int)
    overlap_graph_parser.add_argument("--max-model-points", default=256, type=int)
    overlap_graph_parser.add_argument("--k-neighbors", default=12, type=int)

    overlap_graph_sweep_parser = subparsers.add_parser(
        "neutral-overlap-graph-sweep",
        help="sweep observer-overlap graph geometry parameters over one or more observer-view runs",
    )
    overlap_graph_sweep_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    overlap_graph_sweep_parser.add_argument("--out", required=True, type=Path)
    overlap_graph_sweep_parser.add_argument("--seeds", default="1")
    overlap_graph_sweep_parser.add_argument("--max-model-points-values", default="256")
    overlap_graph_sweep_parser.add_argument("--k-neighbor-values", default="12")

    overlap_residual_graph_parser = subparsers.add_parser(
        "neutral-overlap-residual-graph",
        help="write residualized observer-overlap graph geometry diagnostics from observer_views.jsonl",
    )
    overlap_residual_graph_parser.add_argument("--run-dir", required=True, type=Path)
    overlap_residual_graph_parser.add_argument("--out", default=None, type=Path)
    overlap_residual_graph_parser.add_argument("--seed", default=1, type=int)
    overlap_residual_graph_parser.add_argument("--max-model-points", default=256, type=int)
    overlap_residual_graph_parser.add_argument("--k-neighbors", default=12, type=int)
    overlap_residual_graph_parser.add_argument("--remove-modes", default=1, type=int)

    overlap_residual_graph_sweep_parser = subparsers.add_parser(
        "neutral-overlap-residual-graph-sweep",
        help="sweep residualized observer-overlap graph geometry parameters over one or more runs",
    )
    overlap_residual_graph_sweep_parser.add_argument("--run-dir", required=True, nargs="+", type=Path)
    overlap_residual_graph_sweep_parser.add_argument("--out", required=True, type=Path)
    overlap_residual_graph_sweep_parser.add_argument("--seeds", default="1")
    overlap_residual_graph_sweep_parser.add_argument("--max-model-points-values", default="256")
    overlap_residual_graph_sweep_parser.add_argument("--k-neighbor-values", default="12")
    overlap_residual_graph_sweep_parser.add_argument("--remove-mode-values", default="1")

    paper_chart_parser = subparsers.add_parser(
        "paper-chart-receipts",
        help="write paper-side S2 conformal/Lorentz/H3 chart receipts into a run folder",
    )
    paper_chart_parser.add_argument("--run-dir", required=True, type=Path)
    paper_chart_parser.add_argument("--point-count", default=4096, type=int)
    paper_chart_parser.add_argument("--cap-count", default=32, type=int)
    paper_chart_parser.add_argument("--theta-values", default="0.35,0.55,0.75,1.0,1.25")
    paper_chart_parser.add_argument("--seed", default=20260610, type=int)

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
    galaxy_static_parser.add_argument("--dataset", type=Path)
    galaxy_static_parser.add_argument("--sparc-dir", type=Path)
    galaxy_static_parser.add_argument("--rar", type=Path)
    galaxy_static_parser.add_argument("--btfr", type=Path)
    galaxy_static_parser.add_argument("--rotation", type=Path)
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

    object_h3_viewer_parser = subparsers.add_parser(
        "run-object-h3-viewer",
        help="write a standalone 3D object-H3 bulk viewer from exported object coordinates",
    )
    object_h3_viewer_parser.add_argument("--run-dir", required=True, type=Path)
    object_h3_viewer_parser.add_argument("--out", default=None, type=Path)
    object_h3_viewer_parser.add_argument("--max-objects", default=512, type=int)

    universe_timeline_viewer_parser = subparsers.add_parser(
        "run-universe-timeline-viewer",
        help="write a combined OPH screen/repair/observer-time/H3/CMB visualization bundle",
    )
    universe_timeline_viewer_parser.add_argument("--small-universe-dir", required=True, type=Path)
    universe_timeline_viewer_parser.add_argument("--observer-run-dir", required=True, type=Path)
    universe_timeline_viewer_parser.add_argument("--consensus-pack-dir", default=None, type=Path)
    universe_timeline_viewer_parser.add_argument("--consensus-readout-dir", default=None, type=Path)
    universe_timeline_viewer_parser.add_argument("--out-dir", required=True, type=Path)
    universe_timeline_viewer_parser.add_argument("--max-screen-points", default=3500, type=int)
    universe_timeline_viewer_parser.add_argument("--max-observers", default=96, type=int)
    universe_timeline_viewer_parser.add_argument("--max-h3-objects", default=512, type=int)

    scale_viewer_parser = subparsers.add_parser(
        "run-scale-compressed-viewer",
        help="write a standalone viewer for scale-compressed repair/CMB outputs",
    )
    scale_viewer_parser.add_argument("--run-dir", required=True, type=Path)
    scale_viewer_parser.add_argument("--out", default=None, type=Path)

    cmb_neutral_viewer_parser = subparsers.add_parser(
        "run-cmb-neutral-frontier-viewer",
        help="write a pack-level CMB/neutral-bulk frontier viewer",
    )
    cmb_neutral_viewer_parser.add_argument("--run-dir", required=True, type=Path)
    cmb_neutral_viewer_parser.add_argument("--out", default=None, type=Path)

    cmb_static_plots_parser = subparsers.add_parser(
        "run-cmb-static-plots",
        help="write static CMB comparison and neutral-gate PNG plots",
    )
    cmb_static_plots_parser.add_argument("--run-dir", required=True, type=Path)
    cmb_static_plots_parser.add_argument("--out-dir", default=None, type=Path)

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
    if args.command == "run-oph-universe":
        from oph_fpe.pipelines import run_oph_universe_pipeline

        result = run_oph_universe_pipeline(
            config_path=args.config,
            out_dir=args.out_dir,
            run_id=args.run_id,
            source_run_dir=args.source_run_dir,
            skip_base_run=args.skip_base_run,
            max_screen_points=args.max_screen_points,
            max_observers=args.max_observers,
            max_h3_objects=args.max_h3_objects,
            emit_visualizations=not args.skip_visualizations,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command in {"shape-dodeca-smoke", "run-shape-substrate"}:
        from oph_fpe.experiments import load_config
        from oph_fpe.scale import run_shape_dodeca_smoke

        if args.config is not None:
            config = load_config(args.config)
        else:
            config = {
                "name": "shape_dodeca_smoke",
                "seed": 20260609,
                "cycles": 512,
                "repair_rate": 0.02,
                "ell_max": 8,
                "particle_energy_threshold": 0.001,
                "certificate_detuning_multipliers": [0.5, 1.0, 1.5, 2.0],
            }
        for key, value in {
            "cycles": args.cycles,
            "repair_rate": args.repair_rate,
            "seed": args.seed,
            "ell_max": args.ell_max,
            "particle_energy_threshold": args.particle_energy_threshold,
        }.items():
            if value is not None:
                config[key] = value
        result = run_shape_dodeca_smoke(config, args.out_dir)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "shape-ensemble":
        from oph_fpe.experiments import load_config
        from oph_fpe.scale import run_shape_ensemble

        seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
        result = run_shape_ensemble(load_config(args.config), args.out_dir, seeds=seeds, workers=args.workers)
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
    if args.command == "oph-exact-cmb-camb":
        from oph_fpe.cosmology.camb_adapter import write_oph_exact_cmb_camb_report

        result = write_oph_exact_cmb_camb_report(
            args.benchmark,
            args.out,
            source_dir=args.source_dir,
            lmax=args.lmax,
            benchmark_label=args.label,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "finite-repair-clock-cmb-camb":
        from oph_fpe.cosmology.camb_adapter import write_finite_repair_clock_cmb_camb_report

        result = write_finite_repair_clock_cmb_camb_report(
            args.finite_clock_report,
            args.benchmark,
            args.out,
            source_dir=args.source_dir,
            lmax=args.lmax,
            benchmark_label=args.label,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "oph-cmb-selector-elimination":
        from oph_fpe.constants.oph_pixel import P_STAR
        from oph_fpe.cosmology.selector_elimination import write_selector_elimination_report

        result = write_selector_elimination_report(
            args.source_dir,
            args.out,
            P=P_STAR if args.p_value is None else args.p_value,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "oph-boltzmann-inputs":
        from oph_fpe.cosmology.boltzmann_inputs import write_oph_boltzmann_input_report

        result = write_oph_boltzmann_input_report([*args.run_dir, *args.include], args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "b-a-parent-report":
        from oph_fpe.cosmology.ba_parent import write_b_a_parent_report

        a_grid = [float(value) for value in _csv_values(args.a_grid)] if args.a_grid else None
        k_grid = [float(value) for value in _csv_values(args.k_grid)] if args.k_grid else None
        seeds = [int(value) for value in _csv_values(args.seeds)] if args.seeds else []
        result = write_b_a_parent_report(
            [*args.run_dir, *args.include],
            args.out,
            a_grid=a_grid,
            k_grid_h_mpc=k_grid,
            eps=args.eps,
            modes_per_k=args.modes_per_k,
            seeds=seeds,
        )
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
            reference_mode=args.reference_mode,
            primordial_k_count=args.primordial_k_count,
            primordial_k_min=args.primordial_k_min,
            primordial_k_max=args.primordial_k_max,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "maxent-green-spectrum":
        from oph_fpe.constants.oph_pixel import P_STAR
        from oph_fpe.cosmology.maxent_green_spectrum import write_maxent_green_spectrum_report

        result = write_maxent_green_spectrum_report(
            args.out,
            source_dir=args.source_dir,
            patch_count=args.patch_count,
            ell_max=args.ell_max,
            P=P_STAR,
            kappa_rep=args.kappa_rep,
            amplitude=args.amplitude,
            mu=args.mu,
            primordial_k_count=args.primordial_k_count,
            primordial_k_min=args.primordial_k_min,
            primordial_k_max=args.primordial_k_max,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "repair-clock-report":
        from oph_fpe.constants.oph_pixel import P_STAR
        from oph_fpe.cosmology.repair_clock import write_repair_clock_report

        result = write_repair_clock_report(
            [*args.run_dir, *args.include],
            args.out,
            P=P_STAR,
            cycle_time_normalization=args.cycle_time_normalization,
            r2_threshold=args.r2_threshold,
            relative_tolerance=args.relative_tolerance,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "scalar-repair-semigroup":
        from oph_fpe.constants.oph_pixel import P_STAR
        from oph_fpe.cosmology.scalar_repair_semigroup import write_scalar_repair_semigroup_report

        result = write_scalar_repair_semigroup_report(
            args.out,
            dimension=args.dimension,
            kappa_rep=args.kappa_rep,
            source=args.source,
            finite_lattice_derived=args.finite_lattice_derived,
            matrix_source=args.matrix_source,
            p_value=P_STAR,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "finite-repair-transition-clock":
        from oph_fpe.constants.oph_pixel import P_STAR
        from oph_fpe.cosmology.finite_repair_transition_clock import (
            write_finite_repair_transition_clock_report,
        )

        result = write_finite_repair_transition_clock_report(
            args.run_dir,
            args.out,
            packet_fields=tuple(_csv_values(args.packet_fields)),
            primary_matrix=args.primary_matrix,
            repair_step_time=args.repair_step_time,
            clock_normalization_source=args.clock_normalization_source,
            weight_field=args.weight_field,
            p_value=P_STAR,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "finite-repair-transition-sweep":
        from oph_fpe.constants.oph_pixel import P_STAR
        from oph_fpe.cosmology.finite_repair_transition_clock import (
            write_finite_repair_transition_clock_sweep_report,
        )

        result = write_finite_repair_transition_clock_sweep_report(
            args.run_dir,
            args.out,
            primary_matrices=tuple(_csv_values(args.primary_matrices)),
            repair_step_times=tuple(float(value) for value in _csv_values(args.repair_step_times)),
            clock_normalization_source=args.clock_normalization_source,
            weight_field=args.weight_field,
            p_value=P_STAR,
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
    if args.command == "fossil-spectrum-report":
        from oph_fpe.cosmology.fossil_spectrum import write_fossil_spectrum_report

        fields = [item.strip() for item in args.fields.split(",")] if args.fields else None
        result = write_fossil_spectrum_report(
            args.run_dir,
            args.out,
            ell_min=args.ell_min,
            ell_max=args.ell_max,
            fields=fields,
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
    if args.command == "h0s8-lane8-certificate":
        from oph_fpe.cosmology.h0s8_certificates import write_h0s8_lane8_certificate_report

        result = write_h0s8_lane8_certificate_report(args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "oph-cnb-neutrinos":
        from oph_fpe.cosmology.neutrino_background import write_oph_cnb_background_report

        result = write_oph_cnb_background_report(
            args.source_dir,
            args.out,
            delta_neff_coh=args.delta_neff_coh,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "oph-compressed-likelihood":
        from oph_fpe.cosmology.compressed_likelihood import write_compressed_likelihood_reference_report

        result = write_compressed_likelihood_reference_report(args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "screen-capacity-report":
        from oph_fpe.cosmology.screen_capacity import write_screen_capacity_closure_report

        patch_counts = tuple(
            int(value.strip()) for value in args.regulator_patch_counts.split(",") if value.strip()
        )
        result = write_screen_capacity_closure_report(
            args.out,
            n_crc=args.n_crc,
            radius_m=args.r_ds_m,
            planck_length_m=args.planck_length_m,
            regulator_patch_counts=patch_counts,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "capacity-readback-proxy-report":
        from oph_fpe.constants.oph_pixel import P_STAR
        from oph_fpe.cosmology.screen_capacity import DEFAULT_N_CRC, write_capacity_readback_proxy_report

        result = write_capacity_readback_proxy_report(
            args.run_dir,
            args.out,
            p_value=P_STAR if args.p_value is None else args.p_value,
            n_crc=DEFAULT_N_CRC if args.n_crc is None else args.n_crc,
            max_observer_views=args.max_observer_views,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "scale-bridge-report":
        from oph_fpe.constants.oph_pixel import P_STAR
        from oph_fpe.cosmology.scale_bridge import ScaleBridgeInputs, write_scale_bridge_report
        from oph_fpe.cosmology.screen_capacity import DEFAULT_N_CRC

        result = write_scale_bridge_report(
            args.out,
            ScaleBridgeInputs(
                P_star=P_STAR if args.p_star is None else args.p_star,
                N_star=DEFAULT_N_CRC if args.n_star is None else args.n_star,
                Lambda_star_m2_inverse=args.lambda_star_m2,
                B_ell_m2_inverse=args.b_ell_m2,
                source=args.source,
            ),
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "pn-resonance-report":
        from oph_fpe.constants.oph_pixel import P_STAR
        from oph_fpe.cosmology.pn_resonance import (
            ALPHA_U_P_STAR,
            PNResonanceInputs,
            write_pn_resonance_report,
        )

        patch_counts = tuple(
            int(value.strip()) for value in args.regulator_patch_counts.split(",") if value.strip()
        )
        result = write_pn_resonance_report(
            args.out,
            PNResonanceInputs(
                P_star=P_STAR if args.p_star is None else args.p_star,
                alpha_U=ALPHA_U_P_STAR if args.alpha_u is None else args.alpha_u,
                N_star=args.n_star,
                N_source=args.n_source,
                B_ell_m2_inverse=args.b_ell_m2,
                Lambda_star_m2_inverse=args.lambda_star_m2,
                repair_rounds=args.repair_rounds,
                regulator_patch_counts=patch_counts,
                source=args.source,
            ),
            run_dirs=args.run_dir,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "silence-to-observation-report":
        from oph_fpe.cosmology.pn_resonance import ALPHA_U_P_STAR
        from oph_fpe.cosmology.silence_to_observation import (
            SilenceToObservationInputs,
            write_silence_to_observation_report,
        )

        result = write_silence_to_observation_report(
            args.run_dir,
            args.out,
            SilenceToObservationInputs(
                P_star=args.p_star,
                alpha_U=ALPHA_U_P_STAR if args.alpha_u is None else args.alpha_u,
                N_star=args.n_star,
                N_source=args.n_source,
                repair_rounds=args.repair_rounds,
                source=args.source,
            ),
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "positive-geometry-kernel-report":
        from oph_fpe.dynamics.positive_geometry import write_positive_geometry_kernel_report

        result = write_positive_geometry_kernel_report(
            args.out,
            manifest_path=args.manifest,
            pgk_root=args.pgk_root,
            source=args.source,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "no-g-clock-bridge-report":
        from oph_fpe.cosmology.scale_bridge import (
            EPSILON_CS_SELECTED,
            NU_CS_HZ,
            NoGClockBridgeInputs,
            write_no_g_clock_bridge_report,
        )

        dependency_graph = None
        if args.dependency_graph is not None:
            dependency_graph = json.loads(args.dependency_graph.read_text(encoding="utf-8"))
        result = write_no_g_clock_bridge_report(
            args.out,
            NoGClockBridgeInputs(
                epsilon_cs=EPSILON_CS_SELECTED if args.epsilon_cs is None else args.epsilon_cs,
                nu_cs_hz=NU_CS_HZ if args.nu_cs_hz is None else args.nu_cs_hz,
                dependency_graph=dependency_graph,
                source=args.source,
                source_readback_map_emitted=args.source_readback_map_emitted,
                contraction_certificate=args.contraction_certificate,
                residual_certificate=args.residual_certificate,
                public_dependency_graph=args.public_dependency_graph,
            ),
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "repair-scale-closure":
        from oph_fpe.cosmology.repair_scale_closure import write_repair_scale_closure_report
        from oph_fpe.cosmology.screen_capacity import DEFAULT_N_CRC

        patch_counts = tuple(
            int(value.strip()) for value in args.regulator_patch_counts.split(",") if value.strip()
        )
        result = write_repair_scale_closure_report(
            args.out,
            n_crc=DEFAULT_N_CRC if args.n_crc is None else args.n_crc,
            repair_rounds=args.repair_rounds,
            regulator_patch_counts=patch_counts,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "scale-compressed-repair":
        from oph_fpe.scale.scale_compressed_repair import scale_compressed_repair_run

        result = scale_compressed_repair_run(
            args.out,
            repair_rounds=args.repair_rounds,
            object_count=args.object_count,
            particle_count=args.particle_count,
            cap_axis_count=args.cap_axis_count,
            ell_max=args.ell_max,
            seed=args.seed,
            planck_tt=args.planck_tt,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "scale-compressed-cmb-camb":
        from oph_fpe.cosmology.camb_adapter import write_scale_compressed_cmb_camb_report

        result = write_scale_compressed_cmb_camb_report(
            args.scale_report,
            args.benchmark,
            args.out,
            lmax=args.lmax,
            benchmark_label=args.label,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "oph-inflation-cmb-bridge":
        from oph_fpe.cosmology.inflation_cmb_ladder import write_inflation_cmb_bridge_report

        result = write_inflation_cmb_bridge_report(args.source_dir, args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "inflation-certificates":
        from oph_fpe.cosmology.inflation_certificates import write_inflation_certificate_bundle_report

        result = write_inflation_certificate_bundle_report(args.cert_dir, args.out, source_path=args.source_path)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "finite-certificates":
        from oph_fpe.cosmology.finite_certificates import write_finite_certificate_bundle

        result = write_finite_certificate_bundle(args.input, args.out, toy=args.toy)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "finite-certificates-from-run":
        from oph_fpe.cosmology.finite_certificates import write_run_proxy_finite_certificate_bundle

        result = write_run_proxy_finite_certificate_bundle(args.run_dir, args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "parent-collar-ladder":
        from oph_fpe.cosmology.parent_collar_ladder import write_parent_collar_ladder_report

        result = write_parent_collar_ladder_report([*args.run_dir, *args.include], args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "b-a-kernel-paired":
        from oph_fpe.cosmology.ba_kernel import ba_kernel_report_from_paired_csv

        result = ba_kernel_report_from_paired_csv(
            args.base,
            args.perturbed,
            args.out,
            control_csv=args.control,
            min_good_rows=args.min_good_rows,
            min_sample_count=args.min_sample_count,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "b-a-kernel-from-parent":
        from oph_fpe.cosmology.ba_kernel import ba_kernel_report_from_parent_report

        result = ba_kernel_report_from_parent_report(args.parent_report, args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "b-a-kernel-refinement":
        from oph_fpe.cosmology.ba_kernel import write_ba_kernel_refinement_report

        result = write_ba_kernel_refinement_report(args.report, args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "derive-physical-cmb-inputs":
        from oph_fpe.cosmology.physical_cmb_prediction import write_physical_cmb_input_report

        result = write_physical_cmb_input_report([*args.run_dir, *args.include], args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "physical-cmb-no-data-use-receipt":
        from oph_fpe.cosmology.physical_cmb_prediction import write_physical_cmb_input_no_data_use_receipt

        result = write_physical_cmb_input_no_data_use_receipt([*args.run_dir, *args.include], args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "physical-cmb-promotion-audit":
        from oph_fpe.cosmology.physical_cmb_prediction import write_physical_cmb_promotion_audit_report

        result = write_physical_cmb_promotion_audit_report([*args.run_dir, *args.include], args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "physical-cmb-frontier":
        from oph_fpe.cosmology.physical_cmb_prediction import write_physical_cmb_frontier_report

        result = write_physical_cmb_frontier_report([*args.run_dir, *args.include], args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "physical-cmb-output-comparison":
        from oph_fpe.cosmology.physical_cmb_output import write_physical_cmb_output_comparison_report

        result = write_physical_cmb_output_comparison_report([*args.run_dir, *args.include], args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "observer-consensus-bulk-readout":
        from oph_fpe.bulk.observer_consensus_bulk import write_observer_consensus_bulk_readout_report

        result = write_observer_consensus_bulk_readout_report(
            [*args.run_dir, *args.include],
            args.out,
            observer_sample_count=args.observer_sample_count,
            object_sample_count=args.object_sample_count,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "official-planck-readiness":
        from oph_fpe.cosmology.camb_adapter import write_official_planck_readiness_report

        result = write_official_planck_readiness_report(args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "finite-collar-boltzmann-bundle":
        from oph_fpe.cosmology.finite_collar_boltzmann_bundle import (
            write_finite_collar_boltzmann_bundle_report,
        )

        result = write_finite_collar_boltzmann_bundle_report([*args.run_dir, *args.include], args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "finite-collar-cmb-projection":
        from oph_fpe.cosmology.finite_collar_projection import write_finite_collar_cmb_projection_report

        result = write_finite_collar_cmb_projection_report(
            [*args.run_dir, *args.include],
            args.out,
            chi_star_mpc=args.chi_star_mpc,
            h=args.h,
            ell_mapping=args.ell_mapping,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "scalar-quotient-report":
        from oph_fpe.cosmology.scalar_quotient import write_scalar_quotient_report

        result = write_scalar_quotient_report(
            args.run_dir,
            args.out,
            target_ell_ir=args.target_ell_ir,
            bins=args.bins,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "emit-scalar-release-certificate":
        from oph_fpe.cosmology.inflation_certificates import emit_scalar_release_certificate_from_collar_run

        result = emit_scalar_release_certificate_from_collar_run(
            args.run_dir,
            args.out,
            kappa_rel=args.kappa_rel,
            source_path=args.source_path,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "emit-edge-center-certificate":
        from oph_fpe.constants.oph_pixel import P_STAR
        from oph_fpe.cosmology.inflation_certificates import emit_edge_center_certificate

        result = emit_edge_center_certificate(
            args.out,
            p_value=P_STAR if args.p_value is None else args.p_value,
            source_path=args.source_path,
        )
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
    if args.command == "cmb-fossil-bridge":
        from oph_fpe.cmb_fossil import write_cmb_fossil_bridge_report

        result = write_cmb_fossil_bridge_report(
            args.out_dir,
            planck_tt=args.planck_tt,
            ell_max=args.ell_max,
            eta=args.eta,
            q_ir=args.q_ir,
            ell_ir=args.ell_ir,
            eps_p=args.eps_p,
            ell_p=args.ell_p,
            ell_cap=args.ell_cap,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "comparable-data":
        from oph_fpe.cosmology.comparable_data import write_comparable_data_package

        result = write_comparable_data_package([*args.run_dir, *args.include], args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "export-measurement-pack":
        from oph_fpe.measurement_pack import export_measurement_pack

        result = export_measurement_pack([*args.run_dir, *args.include], args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "bulk-proof-certificate":
        from oph_fpe.bulk.proof_certificate import write_bulk_proof_certificate

        result = write_bulk_proof_certificate(args.run_dir, args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "strict-neutral-bulk-report":
        from oph_fpe.bulk.neutral_bulk import write_strict_neutral_bulk_report

        result = write_strict_neutral_bulk_report(
            args.run_dir,
            args.out,
            seed=args.seed,
            max_model_points=args.max_model_points,
            planted_control_points=args.planted_control_points,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "strict-neutral-object-bulk-report":
        from oph_fpe.bulk.neutral_object_bulk import write_strict_neutral_object_bulk_report

        result = write_strict_neutral_object_bulk_report(
            args.run_dir,
            args.out,
            seed=args.seed,
            min_objects=args.min_objects,
            min_observers_per_object=args.min_observers_per_object,
            max_observer_fraction_per_object=args.max_observer_fraction_per_object,
            max_model_points=args.max_model_points,
            heldout_fraction=args.heldout_fraction,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "neutral-profile-audit":
        from oph_fpe.bulk.neutral_bulk import NEUTRAL_PROFILE_WEIGHTS, write_neutral_profile_audit_report

        profiles = None
        if args.profiles:
            names = [name.strip() for name in str(args.profiles).split(",") if name.strip()]
            unknown = [name for name in names if name not in NEUTRAL_PROFILE_WEIGHTS]
            if unknown:
                raise SystemExit(f"unknown neutral profiles: {', '.join(unknown)}")
            profiles = {name: NEUTRAL_PROFILE_WEIGHTS[name] for name in names}

        result = write_neutral_profile_audit_report(
            args.run_dir,
            args.out,
            seed=args.seed,
            sample_count=args.sample_count,
            max_model_points=args.max_model_points,
            profiles=profiles,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "attach-prime-geometric-response":
        from oph_fpe.bulk.prime_geometric_response import write_prime_geometric_response_attachment

        result = write_prime_geometric_response_attachment(
            args.run_dir,
            args.out,
            spectrum_width=args.spectrum_width,
            component_bins=args.component_bins,
            backup=not args.no_backup,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "neutral-prime-rank-sweep":
        from oph_fpe.bulk.neutral_bulk import write_prime_geometric_rank_sweep_report

        ranks = [int(value.strip()) for value in str(args.ranks).split(",") if value.strip()]
        result = write_prime_geometric_rank_sweep_report(
            args.run_dir,
            args.out,
            ranks=ranks,
            seed=args.seed,
            sample_count=args.sample_count,
            max_model_points=args.max_model_points,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "neutral-prime-rank-refinement":
        from oph_fpe.bulk.neutral_bulk import write_prime_geometric_rank_refinement_report

        result = write_prime_geometric_rank_refinement_report(args.report, args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "neutral-3d-bulk-audit":
        from oph_fpe.bulk.neutral_bulk import write_neutral_3d_bulk_audit_report

        result = write_neutral_3d_bulk_audit_report(args.report, args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "neutral-independent-rank-selector-audit":
        from oph_fpe.bulk.neutral_bulk import write_neutral_independent_rank_selector_audit_report

        result = write_neutral_independent_rank_selector_audit_report(args.report, args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "strict-neutral-bulk-frontier":
        from oph_fpe.bulk.neutral_bulk import write_strict_neutral_bulk_frontier_report

        result = write_strict_neutral_bulk_frontier_report(args.report, args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "neutral-overlap-control-report":
        from oph_fpe.bulk.neutral_bulk import write_overlap_native_neutral_control_report

        result = write_overlap_native_neutral_control_report(
            args.run_dir,
            args.out,
            seed=args.seed,
            max_model_points=args.max_model_points,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "neutral-overlap-graph-geometry":
        from oph_fpe.bulk.neutral_bulk import write_overlap_native_graph_geometry_report

        result = write_overlap_native_graph_geometry_report(
            args.run_dir,
            args.out,
            seed=args.seed,
            max_model_points=args.max_model_points,
            k_neighbors=args.k_neighbors,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "neutral-overlap-graph-sweep":
        from oph_fpe.bulk.neutral_bulk import write_overlap_native_graph_geometry_sweep_report

        result = write_overlap_native_graph_geometry_sweep_report(
            args.run_dir,
            args.out,
            seeds=tuple(int(value) for value in _csv_values(args.seeds)),
            max_model_points_values=tuple(int(value) for value in _csv_values(args.max_model_points_values)),
            k_neighbor_values=tuple(int(value) for value in _csv_values(args.k_neighbor_values)),
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "neutral-overlap-residual-graph":
        from oph_fpe.bulk.neutral_bulk import write_overlap_residualized_graph_geometry_report

        result = write_overlap_residualized_graph_geometry_report(
            args.run_dir,
            args.out,
            seed=args.seed,
            max_model_points=args.max_model_points,
            k_neighbors=args.k_neighbors,
            remove_modes=args.remove_modes,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "neutral-overlap-residual-graph-sweep":
        from oph_fpe.bulk.neutral_bulk import write_overlap_residualized_graph_geometry_sweep_report

        result = write_overlap_residualized_graph_geometry_sweep_report(
            args.run_dir,
            args.out,
            seeds=tuple(int(value) for value in _csv_values(args.seeds)),
            max_model_points_values=tuple(int(value) for value in _csv_values(args.max_model_points_values)),
            k_neighbor_values=tuple(int(value) for value in _csv_values(args.k_neighbor_values)),
            remove_mode_values=tuple(int(value) for value in _csv_values(args.remove_mode_values)),
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "paper-chart-receipts":
        from oph_fpe.bulk.conformal_spatial_chart import write_paper_chart_receipts

        theta_values = tuple(float(value) for value in _csv_values(args.theta_values))
        result = write_paper_chart_receipts(
            args.run_dir,
            point_count=args.point_count,
            cap_count=args.cap_count,
            theta_values=theta_values,
            seed=args.seed,
        )
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
        from oph_fpe.cosmology.galaxy_static import (
            StaticGalaxyDataset,
            load_static_galaxy_dataset,
            static_galaxy_measurement_report,
            write_static_galaxy_measurement_outputs,
        )

        dataset_paths = [
            path
            for path in (args.dataset, args.sparc_dir, args.rar, args.btfr, args.rotation)
            if path is not None
        ]
        if not dataset_paths:
            parser.error("run-galaxy-static requires --dataset, --sparc-dir, --rar, --btfr, or --rotation")
        loaded = [load_static_galaxy_dataset(path) for path in dataset_paths]
        dataset = StaticGalaxyDataset(
            rows=[row for item in loaded for row in item.rows],
            source_paths=[path for item in loaded for path in item.source_paths],
        )
        result = static_galaxy_measurement_report(
            dataset,
            a0_initial=args.a0_initial,
            lambda_initial=args.lambda_initial,
            min_points=args.min_points,
            min_galaxies=args.min_galaxies,
            physical_claim=not bool(args.diagnostic_only),
        )
        write_static_galaxy_measurement_outputs(dataset, result, args.out_dir)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "run-viewer":
        from oph_fpe.viz import write_run_viewer

        result = write_run_viewer(args.run_dir, args.out, max_screen_points=args.max_screen_points)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "run-object-h3-viewer":
        from oph_fpe.viz import write_object_h3_bulk_viewer

        result = write_object_h3_bulk_viewer(args.run_dir, args.out, max_objects=args.max_objects)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "run-universe-timeline-viewer":
        from oph_fpe.viz import write_universe_timeline_bundle

        result = write_universe_timeline_bundle(
            small_universe_dir=args.small_universe_dir,
            observer_run_dir=args.observer_run_dir,
            consensus_pack_dir=args.consensus_pack_dir,
            consensus_readout_dir=args.consensus_readout_dir,
            out_dir=args.out_dir,
            max_screen_points=args.max_screen_points,
            max_observers=args.max_observers,
            max_h3_objects=args.max_h3_objects,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "run-scale-compressed-viewer":
        from oph_fpe.viz import write_scale_compressed_viewer

        result = write_scale_compressed_viewer(args.run_dir, args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "run-cmb-neutral-frontier-viewer":
        from oph_fpe.viz import write_cmb_neutral_frontier_viewer

        result = write_cmb_neutral_frontier_viewer(args.run_dir, args.out)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "run-cmb-static-plots":
        from oph_fpe.viz import write_cmb_static_plots

        result = write_cmb_static_plots(args.run_dir, args.out_dir)
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
