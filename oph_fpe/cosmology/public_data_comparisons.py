from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import yaml

from oph_fpe.cosmology.cassini_external_field import cassini_external_field_report
from oph_fpe.cosmology.btfr_likelihood import btfr_error_aware_report
from oph_fpe.cosmology.cmb_promotion_ledger import cmb_promotion_ledger_report
from oph_fpe.cosmology.galaxy_static import (
    load_static_galaxy_dataset,
    static_galaxy_measurement_report,
)
from oph_fpe.cosmology.rar_fixed_comparison import fixed_oph_rar_report


SCHEMA_VERSION = "oph_best_of_public_data_comparisons_v1"
PLANCK_TT_SOURCE_URL = (
    "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/"
    "cosmoparams/COM_PowerSpect_CMB-TT-binned_R3.01.txt"
)
SPARC_SOURCE_URL = "https://astroweb.case.edu/SPARC/"
CASSINI_SOURCE_URL = "https://arxiv.org/abs/2602.17884"
DEFAULT_CASSINI_SUMMARY_PATH = (
    Path(__file__).resolve().parents[2]
    / "data/measurements/cassini/cassini_q2_2026.json"
)


def public_data_comparison_report(
    primary_run_dir: Path,
    *,
    planck_tt_path: Path,
    sparc_dir: Path,
    history_run_dirs: Iterable[Path] = (),
    baseline_run_dir: Path | None = None,
    planned_config_path: Path | None = None,
    cassini_summary_path: Path | None = None,
) -> dict[str, Any]:
    """Build a fail-closed, provenance-bound summary of public-data comparisons.

    The primary run is selected before public metrics are inspected. Historical
    runs provide declared context only. Every CMB report and sidecar is resolved
    from one run directory; this deliberately avoids the existing multi-root
    first-match aggregation behavior.
    """

    primary = Path(primary_run_dir)
    history = _unique_paths(history_run_dirs, exclude=primary)
    baseline = Path(baseline_run_dir) if baseline_run_dir is not None else primary
    all_runs = _unique_paths([primary, *history, baseline])

    planck_rows, planck_errors = _read_planck_rows(Path(planck_tt_path))
    planck_dataset = _dataset_record(
        dataset_id="planck_pr3_tt_binned_r3_01",
        release="Planck 2018 / PR3 R3.01",
        path=Path(planck_tt_path),
        public_source=PLANCK_TT_SOURCE_URL,
        record_count=len(planck_rows),
        errors=planck_errors,
    )
    sparc_dataset, sparc_report, sparc_errors = _sparc_dataset_and_report(Path(sparc_dir))
    cassini_dataset, cassini_row, cassini_errors = _cassini_dataset_and_comparison(
        Path(cassini_summary_path or DEFAULT_CASSINI_SUMMARY_PATH)
    )

    run_bundles = [_run_bundle(path, primary=primary, baseline=baseline) for path in all_runs]
    run_by_key = {_path_key(Path(row["run_dir"])): row for row in run_bundles}

    cmb_rows: list[dict[str, Any]] = []
    for run_dir in all_runs:
        bundle = run_by_key[_path_key(run_dir)]
        cmb_rows.append(
            _cmb_comparison(
                run_dir,
                bundle=bundle,
                dataset=planck_dataset,
                planck_rows=planck_rows,
            )
        )

    sparc_rows = _sparc_comparisons(sparc_dataset, sparc_report, sparc_errors)
    compressed_row = _compressed_reference_comparison(primary)
    comparisons = [*cmb_rows, *sparc_rows, cassini_row]
    if compressed_row is not None:
        comparisons.append(compressed_row)

    readiness = [_repair_readiness(path, run_by_key[_path_key(path)]) for path in all_runs]
    scale_contract = (
        _config_scale_contract(Path(planned_config_path), planck_rows=planck_rows, role="planned")
        if planned_config_path is not None
        else None
    )
    baseline_delta = _primary_baseline_delta(cmb_rows, primary=primary, baseline=baseline)
    diagnostic_order = _diagnostic_order(cmb_rows)

    featured = _featured_by_evidence_class(comparisons, diagnostic_order)
    integrity_errors = [*planck_errors, *sparc_errors, *cassini_errors]
    integrity_errors.extend(
        error
        for bundle in run_bundles
        for error in bundle.get("integrity_errors", [])
        if bundle.get("selection_role") != "history"
    )
    integrity_errors.extend(
        error
        for comparison in comparisons
        for error in comparison.get("integrity_errors", [])
        if comparison.get("selection_role") != "history"
    )
    integrity_errors.extend(
        error
        for row in readiness
        for error in row.get("integrity_errors", [])
        if row.get("selection_role") != "history"
    )
    if scale_contract is not None:
        integrity_errors.extend(scale_contract.get("integrity_errors", []))
    integrity_errors = sorted(set(str(value) for value in integrity_errors))
    public_rows = [
        row
        for row in comparisons
        if row.get("comparison_receipt") and row.get("dataset_id") != "compressed_cosmology_reference"
    ]
    frozen_predictions = [row for row in comparisons if row.get("physical_prediction_receipt")]

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "oph_best_of_public_data_comparisons",
        "selection_policy": {
            "primary_run": str(primary),
            "baseline_run": str(baseline),
            "history_runs": [str(path) for path in history],
            "policy": "explicit_primary_run_selected_before_public_metric_evaluation",
            "patch_count_never_promotes_evidence": True,
            "no_cross_domain_score_or_rank": True,
            "history_is_context_not_primary_selection": True,
        },
        "datasets": {
            "planck_tt": planck_dataset,
            "sparc": sparc_dataset,
            "cassini_q2": cassini_dataset,
        },
        "run_bundles": run_bundles,
        "planned_run_scale_contract": scale_contract,
        "comparisons": comparisons,
        "repair_and_boltzmann_readiness": readiness,
        "planck_tt_diagnostic_order": diagnostic_order,
        "primary_vs_baseline": baseline_delta,
        "featured_by_evidence_class": featured,
        "summary": {
            "integrity_receipt": not integrity_errors,
            "integrity_errors": integrity_errors,
            "any_public_comparison": bool(public_rows),
            "public_comparison_count": len(public_rows),
            "frozen_prediction_count": len(frozen_predictions),
            "physical_prediction_available": bool(frozen_predictions),
            "available_evidence_classes": sorted(
                {
                    str(row.get("evaluation_class"))
                    for row in public_rows
                    if row.get("evaluation_class")
                }
            ),
            "overall_score": None,
            "overall_winner": None,
        },
        "claim_boundary": (
            "Best available evidence is summarized by evidence class, never by a cross-domain scalar score. "
            "Planck TT values are diagonal-error diagnostics with one amplitude fitted on the same bins, not "
            "the official Planck likelihood. SPARC RAR is a calibration, BTFR is a separate descriptive check, "
            "and the mass-model result is a galaxy-level holdout of a phenomenological continuation. Cassini "
            "strongly excludes the natural universal QUMOND extension of the OPH static galaxy equation at "
            "fixed inputs, but the current paper scopes that equation to settled galaxies and supplies no "
            "Solar-System applicability rule; this is a conditional external-domain falsifier and hard missing "
            "gate, not a falsification of recovered OPH core. Only a "
            "same-run promotion ledger with frozen source/solver/likelihood hashes and an evaluated official "
            "likelihood can count as a physical prediction. More carrier patches do not change that status."
        ),
    }


def write_public_data_comparison_suite(
    primary_run_dir: Path,
    out_dir: Path,
    *,
    planck_tt_path: Path,
    sparc_dir: Path,
    history_run_dirs: Iterable[Path] = (),
    baseline_run_dir: Path | None = None,
    planned_config_path: Path | None = None,
    cassini_summary_path: Path | None = None,
) -> dict[str, Any]:
    report = public_data_comparison_report(
        primary_run_dir,
        planck_tt_path=planck_tt_path,
        sparc_dir=sparc_dir,
        history_run_dirs=history_run_dirs,
        baseline_run_dir=baseline_run_dir,
        planned_config_path=planned_config_path,
        cassini_summary_path=cassini_summary_path,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "best_of_public_data_comparisons.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (out / "best_of_public_data_comparisons.md").write_text(
        _markdown_report(report),
        encoding="utf-8",
    )
    _write_metrics_csv(out / "best_of_public_data_metrics.csv", report)
    return report


def _cmb_comparison(
    run_dir: Path,
    *,
    bundle: dict[str, Any],
    dataset: dict[str, Any],
    planck_rows: list[dict[str, float]],
) -> dict[str, Any]:
    errors: list[str] = []
    output_path = Path(run_dir) / "physical_cmb_output_comparison_report.json"
    output, load_error = _load_json(output_path)
    if load_error:
        errors.append(load_error)

    run_id = str(bundle.get("run_id") or Path(run_dir).name)
    base: dict[str, Any] = {
        "comparison_id": f"planck_tt:{run_id}",
        "domain": "cmb_tt",
        "dataset_id": "planck_pr3_tt_binned_r3_01",
        "selection_role": str(bundle.get("selection_role") or "history"),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "carrier_patch_count": bundle.get("carrier_patch_count"),
        "materialized_observer_count": bundle.get("materialized_observer_count"),
        "run_dependent": True,
        "model_role": "oph_diagnostic",
        "claim_level": "diagnostic",
        "evaluation_class": "calibrated_same_data",
        "data_use": {
            "fit_to_evaluation_data": True,
            "fitted_nuisance_parameters": ["overall_TT_amplitude"],
            "target_used_for_model_selection": True,
            "holdout": False,
        },
        "comparison_receipt": False,
        "physical_prediction_receipt": False,
        "rank_eligible_within_planck_diagnostic_lane": False,
        "integrity_receipt": False,
        "integrity_errors": errors,
        "metrics": {},
        "baseline": {},
        "prediction_blockers": [],
    }
    if load_error:
        base["rank_exclusion_reasons"] = ["missing_or_invalid_output_report"]
        return base

    declared_roots = output.get("run_dirs")
    if not isinstance(declared_roots, list) or len(declared_roots) != 1:
        errors.append("output_report_must_declare_exactly_one_run_dir")
    elif Path(str(declared_roots[0])).name != Path(run_dir).name:
        errors.append("output_report_run_dir_does_not_match_bundle")

    best = output.get("best_oph_diagnostic_model")
    if not isinstance(best, dict):
        errors.append("best_oph_diagnostic_model_missing")
        base["integrity_errors"] = errors
        base["rank_exclusion_reasons"] = ["no_oph_diagnostic_model"]
        return base
    if best.get("model_role") != "oph_diagnostic":
        errors.append("best_model_role_not_explicitly_oph_diagnostic")

    model_id = str(best.get("model_id") or "")
    source_name = str(best.get("source_report") or "")
    if not model_id:
        errors.append("best_model_id_missing")
    if not source_name or Path(source_name).name != source_name:
        errors.append("invalid_source_report_name")
    source_path = Path(run_dir) / source_name
    source, source_error = _load_json(source_path)
    if source_error:
        errors.append(source_error)

    comparison: dict[str, Any] = {}
    if source:
        comparisons = source.get("comparison")
        if isinstance(comparisons, dict) and isinstance(comparisons.get(model_id), dict):
            comparison = comparisons[model_id]
        elif model_id == "lcdm_baseline" and isinstance(comparisons, dict):
            comparison = comparisons
        else:
            errors.append("selected_model_missing_from_bound_source_report")

    output_model_rows = [
        row
        for row in output.get("rows", [])
        if isinstance(row, dict) and row.get("model_id") == model_id
    ]
    if len(output_model_rows) != 1 or output_model_rows[0].get("model_role") != "oph_diagnostic":
        errors.append("selected_model_has_no_unique_explicit_oph_role_row")

    embedded = comparison.get("binned_tt_comparison") if isinstance(comparison, dict) else None
    if not isinstance(embedded, list) or not embedded:
        errors.append("profiled_binned_tt_rows_missing_from_source_report")
        embedded = []
    alignment_ok = _planck_alignment_ok(embedded, planck_rows)
    if not alignment_ok:
        errors.append("source_report_planck_rows_do_not_match_local_dataset")

    reported_benchmark_hash = (source.get("input_hashes") or {}).get("benchmark_sha256")
    if reported_benchmark_hash != dataset.get("sha256"):
        errors.append("source_report_benchmark_hash_mismatch")

    profiled_chi2 = _profiled_chi2_per_bin(embedded)
    reported_chi2 = _finite_or_none(comparison.get("amplitude_fit_chi2_per_bin"))
    if profiled_chi2 is None:
        errors.append("profiled_chi2_not_recomputable")
    elif reported_chi2 is None or not math.isclose(profiled_chi2, reported_chi2, rel_tol=1e-10, abs_tol=1e-10):
        errors.append("reported_profiled_chi2_does_not_recompute")

    residuals = _residual_metrics(embedded)
    external_rows = [
        row
        for row in output.get("rows", [])
        if isinstance(row, dict)
        and row.get("model_role") == "external_baseline"
        and _finite_or_none(row.get("amplitude_fit_chi2_per_bin")) is not None
    ]
    external = min(
        external_rows,
        key=lambda row: float(row["amplitude_fit_chi2_per_bin"]),
        default=None,
    )
    if external is None:
        errors.append("external_lcdm_baseline_missing")

    residual_source_name = str((output.get("best_oph_residual_summary") or {}).get("source_csv") or "")
    residual_source_path = Path(run_dir) / residual_source_name if residual_source_name else None
    if residual_source_path is None or not residual_source_path.is_file():
        errors.append("bound_tt_sidecar_missing")
    elif residual_source_path.parent.resolve() != Path(run_dir).resolve():
        errors.append("bound_tt_sidecar_cross_root_mismatch")

    physical_residual_path = Path(run_dir) / "physical_cmb_best_oph_residuals.csv"
    ledger: dict[str, Any]
    try:
        ledger = cmb_promotion_ledger_report([Path(run_dir)])
    except Exception as exc:  # pragma: no cover - defensive against malformed third-party bundles
        ledger = {"blockers": [f"promotion_ledger_error:{type(exc).__name__}"]}
    prediction_receipt = bool(ledger.get("likelihood_evaluated_physical_cmb_prediction", False))
    prediction_blockers = list(ledger.get("blockers") or [])
    if not prediction_receipt:
        prediction_blockers.extend(
            [
                "same_data_amplitude_profiled_diagnostic",
                "not_an_official_planck_likelihood_evaluation",
            ]
        )

    comparison_receipt = bool(
        not errors
        and output.get("USABLE_PHYSICAL_CMB_DATA_RECEIPT") is True
        and profiled_chi2 is not None
    )
    bin_count = len(embedded)
    baseline_value = (
        float(external["amplitude_fit_chi2_per_bin"])
        if external is not None
        else None
    )
    delta_per_bin = (
        float(profiled_chi2 - baseline_value)
        if profiled_chi2 is not None and baseline_value is not None
        else None
    )

    source_files = [
        _file_hash_record(output_path),
        _file_hash_record(source_path),
    ]
    if residual_source_path is not None:
        source_files.append(_file_hash_record(residual_source_path))
    if physical_residual_path.is_file():
        source_files.append(_file_hash_record(physical_residual_path))

    base.update(
        {
            "comparison_id": f"planck_tt:{run_id}:{model_id or 'missing_model'}",
            "model_id": model_id or None,
            "run_derived_inputs": {
                "finite_repair_clock_input": source.get("finite_repair_clock_input"),
                "selector_ir_input": source.get("selector_ir_input"),
                "fixed_external_background_parameters": (source.get("camb") or {}).get(
                    "baseline_lambda_cdm_parameters"
                ),
            },
            "fit_protocol": {
                "statistic": "diagonal_chi2_per_bin_after_one_amplitude_fit",
                "fitted_parameter_count": 1,
                "fitted_parameters": ["overall_TT_amplitude"],
                "covariance": "diagonal_binned_errors_only",
                "official_likelihood": False,
                "profiled_residuals_recomputed_from_source_json": True,
                "legacy_residual_sidecar_uses_raw_curve": True,
            },
            "metrics": {
                "diagonal_chi2_per_bin_after_one_amplitude_fit": _metric(
                    profiled_chi2,
                    unit="dimensionless",
                    direction="lower_is_better_within_this_diagnostic_only",
                    baseline_value=baseline_value,
                    delta=delta_per_bin,
                ),
                "total_diagonal_chi2_after_one_amplitude_fit": _metric(
                    profiled_chi2 * bin_count if profiled_chi2 is not None else None,
                    unit="dimensionless",
                    direction="lower_is_better_within_this_diagnostic_only",
                    baseline_value=baseline_value * bin_count if baseline_value is not None else None,
                    delta=delta_per_bin * bin_count if delta_per_bin is not None else None,
                ),
                "shape_correlation": _metric(
                    comparison.get("shape_correlation"),
                    unit="dimensionless",
                    direction="higher_is_better",
                ),
                "normalized_rmse": _metric(
                    comparison.get("normalized_rmse"),
                    unit="dimensionless",
                    direction="lower_is_better",
                ),
                "profiled_rms_residual_sigma": _metric(
                    residuals.get("profiled_rms_sigma"),
                    unit="sigma",
                    direction="lower_is_better",
                ),
                "raw_rms_residual_sigma": _metric(
                    residuals.get("raw_rms_sigma"),
                    unit="sigma",
                    direction="lower_is_better",
                ),
                "best_fit_amplitude": _metric(
                    comparison.get("best_fit_amplitude"),
                    unit="multiplicative",
                    direction="descriptive",
                ),
                "bin_count": _metric(bin_count, unit="bins", direction="descriptive"),
            },
            "baseline": {
                "model_id": external.get("model_id") if external else None,
                "model_role": "external_baseline",
                "diagonal_chi2_per_bin_after_one_amplitude_fit": baseline_value,
                "oph_minus_baseline_per_bin": delta_per_bin,
                "oph_minus_baseline_total_over_bins": (
                    delta_per_bin * bin_count if delta_per_bin is not None else None
                ),
            },
            "source_bundle": {
                "same_parent_directory": all(
                    Path(row["path"]).parent.resolve() == Path(run_dir).resolve()
                    for row in source_files
                    if row.get("exists")
                ),
                "files": source_files,
                "planck_data_sha256": dataset.get("sha256"),
                "source_report_benchmark_sha256": reported_benchmark_hash,
            },
            "comparison_receipt": comparison_receipt,
            "physical_prediction_receipt": prediction_receipt,
            "prediction_class": (
                "frozen_prediction" if prediction_receipt else "measurement_facing_amplitude_fitted_diagnostic"
            ),
            "rank_eligible_within_planck_diagnostic_lane": comparison_receipt,
            "rank_exclusion_reasons": [] if comparison_receipt else sorted(set(errors)),
            "integrity_receipt": not errors,
            "integrity_errors": sorted(set(errors)),
            "prediction_blockers": sorted(set(str(value) for value in prediction_blockers)),
            "promotion_ledger": {
                "current_claim_tier": ledger.get("current_claim_tier"),
                "first_blocked_gate": ledger.get("first_blocked_gate"),
                "terminal_prediction_assertion_rejected": (
                    "untrusted_terminal_prediction_assertion_rejected" in prediction_blockers
                ),
            },
            "claim_boundary": (
                "Run-derived CAMB TT diagnostic against 83 Planck PR3 binned TT points. The displayed "
                "chi-square uses diagonal bin errors and profiles one amplitude on those same bins; it is "
                "neither a reduced chi-square nor the official Planck likelihood. The selected OPH curve is "
                "compared explicitly with the external LambdaCDM baseline."
            ),
        }
    )
    return base


def _sparc_dataset_and_report(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    errors: list[str] = []
    table_names = (
        "RAR.mrt",
        "RARbins.mrt",
        "BTFR_Lelli2019.mrt",
        "MassModels_Lelli2016c.mrt",
    )
    files = [_file_hash_record(path / name) for name in table_names]
    missing = [row["path"] for row in files if not row["exists"]]
    if missing:
        errors.extend(f"missing_sparc_table:{value}" for value in missing)
        report: dict[str, Any] = {}
    else:
        try:
            dataset = load_static_galaxy_dataset(path)
            report = static_galaxy_measurement_report(dataset, physical_claim=False)
            report["btfr_error_aware_likelihood"] = btfr_error_aware_report(
                path / "BTFR_Lelli2019.mrt"
            )
            report["fixed_oph_rar_comparison"] = fixed_oph_rar_report(
                path / "RAR.mrt", path / "RARbins.mrt"
            )
        except Exception as exc:  # pragma: no cover - defensive for malformed external tables
            errors.append(f"sparc_parse_or_fit_error:{type(exc).__name__}:{exc}")
            report = {}
    record = {
        "dataset_id": "sparc_rar_btfr_massmodels",
        "release": "SPARC public RAR, BTFR, and 2016 mass-model tables",
        "public_source": SPARC_SOURCE_URL,
        "path": str(path),
        "files": files,
        "integrity_receipt": not errors,
        "parsed_dataset_row_count": report.get("dataset_row_count"),
        "named_galaxy_count": report.get("dataset_galaxy_count"),
        "rar_point_count": report.get("rar_point_count"),
        "rar_galaxy_label_count": report.get("rar_galaxy_count"),
        "public_rar_source_galaxy_count": (
            (report.get("fixed_oph_rar_comparison") or {}).get("public_rar_galaxy_count")
        ),
        "btfr_galaxy_count": report.get("btfr_galaxy_count"),
        "integrity_errors": errors,
    }
    return record, report, errors


def _cassini_dataset_and_comparison(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    source, load_error = _load_json(path)
    errors = [load_error] if load_error else []
    observable = source.get("observable") if isinstance(source.get("observable"), dict) else {}
    external = (
        source.get("galactic_external_field")
        if isinstance(source.get("galactic_external_field"), dict)
        else {}
    )
    benchmark = source.get("rar_benchmark") if isinstance(source.get("rar_benchmark"), dict) else {}

    central = _finite_or_none(observable.get("central_value_s2"))
    sigma = _finite_or_none(observable.get("standard_uncertainty_s2"))
    external_central = _finite_or_none(external.get("measured_acceleration_m_s2"))
    external_sigma = _finite_or_none(external.get("standard_uncertainty_m_s2"))
    benchmark_a0 = _finite_or_none(benchmark.get("a0_m_s2"))
    benchmark_q2 = _finite_or_none(benchmark.get("spherical_Q2_s2"))
    if source and source.get("dataset_id") != "cassini_de440_q2_2026":
        errors.append("cassini_summary_dataset_id_mismatch")
    for name, value in (
        ("central_Q2", central),
        ("standard_uncertainty_Q2", sigma),
        ("external_acceleration", external_central),
        ("external_acceleration_uncertainty", external_sigma),
        ("benchmark_a0", benchmark_a0),
        ("benchmark_Q2", benchmark_q2),
    ):
        if value is None:
            errors.append(f"cassini_summary_missing_or_nonfinite:{name}")
    if sigma is not None and sigma <= 0.0:
        errors.append("cassini_summary_nonpositive_Q2_uncertainty")
    if external_central is not None and external_central <= 0.0:
        errors.append("cassini_summary_nonpositive_external_acceleration")
    if external_sigma is not None and external_sigma < 0.0:
        errors.append("cassini_summary_negative_external_acceleration_uncertainty")
    if benchmark_a0 is not None and benchmark_a0 <= 0.0:
        errors.append("cassini_summary_nonpositive_benchmark_a0")

    calculation: dict[str, Any] = {}
    if not errors:
        try:
            calculation = cassini_external_field_report(
                cassini_central_s2=float(central),
                cassini_sigma_s2=float(sigma),
                external_acceleration_m_s2=float(external_central),
                external_acceleration_sigma_m_s2=float(external_sigma),
                park_benchmark_a0_m_s2=float(benchmark_a0),
                park_benchmark_q2_s2=float(benchmark_q2),
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            errors.append(f"cassini_qumond_calculation_error:{type(exc).__name__}:{exc}")
    validation = calculation.get("validation") or {}
    if calculation and not validation.get("receipt"):
        errors.append("cassini_spherical_benchmark_reproduction_failed")
    if calculation and not validation.get("disk_receipt"):
        errors.append("cassini_disk_benchmark_reproduction_failed")

    dataset = {
        "dataset_id": "cassini_de440_q2_2026",
        "release": str(
            source.get("release") or "Park et al. 2026 Cassini DE440 quadrupole constraint"
        ),
        "public_source": CASSINI_SOURCE_URL,
        "path": str(path),
        "sha256": _sha256(path) if path.is_file() else None,
        "record_count": 1 if source else 0,
        "data_kind": source.get("data_kind"),
        "raw_tracking_data_bundled": False,
        "citation": source.get("citation") or {},
        "integrity_receipt": not errors,
        "integrity_errors": sorted(set(str(value) for value in errors)),
    }
    fixed = calculation.get("fixed_input_diagnostic") or {}
    z6 = ((calculation.get("oph_branches") or {}).get("z6_exact_uniform_target") or {})
    unit = ((calculation.get("oph_branches") or {}).get("unit_lambda_endpoint") or {})
    band = ((calculation.get("oph_branches") or {}).get("jensen_lambda_band") or {})
    applicability = calculation.get("applicability") or {
        "assumption_tested": "universal_full_source_static_QUMOND_extension",
        "assumption_derived_by_current_oph": False,
        "current_paper_scope_match": False,
        "missing_gate": "source_derived_solar_system_applicability_or_screening_reduction",
    }
    comparison_receipt = bool(calculation.get("comparison_receipt") and not errors)
    row = {
        "comparison_id": "cassini:conditional_static_external_field",
        "domain": "solar_system_external_field",
        "dataset_id": "cassini_de440_q2_2026",
        "selection_role": "independent_external_constraint",
        "run_id": None,
        "run_dir": None,
        "carrier_patch_count": None,
        "materialized_observer_count": None,
        "run_dependent": False,
        "model_id": "oph_static_rar_qumond_universal_extension",
        "model_role": "oph_universal_static_qumond_extension",
        "claim_level": "conditional_exclusion",
        "evaluation_class": (
            "calibrated_independent_dataset" if calculation else "blocked"
        ),
        "data_use": {
            "fit_to_evaluation_data": False,
            "parameters_fit_to_cassini": False,
            "evaluated_on": "published_Cassini_Q2_summary_statistic",
            "raw_tracking_likelihood_evaluated": False,
            "oph_a0_is_external_benchmark": True,
            "oph_lambda_is_conditional_exact_uniform_target": True,
        },
        "applicability": applicability,
        "applicability_receipt": False,
        "current_scope_match": False,
        "raw_pull_not_nuisance_marginalized": True,
        "metrics": {
            "z6_Q2": _metric(
                z6.get("Q2_s2"),
                unit="s^-2",
                direction="closer_to_cassini_is_better",
                baseline_value=central,
                delta=(z6.get("Q2_s2") - central) if z6.get("Q2_s2") is not None and central is not None else None,
            ),
            "z6_raw_fixed_input_pull": _metric(
                fixed.get("z6_raw_pull_sigma"),
                unit="sigma",
                direction="closer_to_zero_is_better",
            ),
            "z6_gaia_only_combined_pull": _metric(
                fixed.get("z6_gaia_only_combined_pull_sigma"),
                unit="sigma",
                direction="closer_to_zero_is_better",
            ),
            "unit_lambda_Q2": _metric(
                unit.get("Q2_s2"),
                unit="s^-2",
                direction="closer_to_cassini_is_better",
                baseline_value=central,
                delta=(unit.get("Q2_s2") - central) if unit.get("Q2_s2") is not None and central is not None else None,
            ),
            "unit_lambda_raw_fixed_input_pull": _metric(
                fixed.get("unit_lambda_raw_pull_sigma"),
                unit="sigma",
                direction="closer_to_zero_is_better",
            ),
            "jensen_band_minimum_raw_fixed_input_pull": _metric(
                band.get("raw_pull_min_sigma"),
                unit="sigma",
                direction="closer_to_zero_is_better",
            ),
            "maximum_z6_fraction_for_cassini_two_sigma_upper": _metric(
                fixed.get("maximum_multiplicative_fraction_for_two_sigma_upper_z6"),
                unit="fraction",
                direction="descriptive",
            ),
            "published_benchmark_relative_error": _metric(
                validation.get("relative_error"),
                unit="fraction",
                direction="lower_is_better",
            ),
        },
        "comparison_receipt": comparison_receipt,
        "physical_prediction_receipt": False,
        "conditional_external_domain_falsifier": comparison_receipt,
        "rank_eligible_within_planck_diagnostic_lane": False,
        "rank_exclusion_reasons": [
            "different_domain",
            "universal_applicability_not_derived",
            "published_summary_statistic_not_raw_likelihood",
            "raw_pull_not_nuisance_marginalized",
        ],
        "source_bundle": {
            "files": [
                _file_hash_record(path),
                _file_hash_record(Path(__file__).with_name("cassini_external_field.py")),
            ]
        },
        "calculation": calculation,
        "integrity_receipt": not errors,
        "integrity_errors": sorted(set(str(value) for value in errors)),
        "assessment": str(
            calculation.get("assessment")
            or "Cassini comparison unavailable because the summary or benchmark receipt failed."
        ),
        "claim_boundary": str(
            calculation.get("claim_boundary")
            or "No Cassini claim can be made because the source summary or numerical receipt failed."
        ),
    }
    return dataset, row, sorted(set(str(value) for value in errors))


def _sparc_comparisons(
    dataset: dict[str, Any],
    report: dict[str, Any],
    errors: list[str],
) -> list[dict[str, Any]]:
    source_files = list(dataset.get("files") or [])
    common = {
        "domain": "static_galaxy",
        "dataset_id": "sparc_rar_btfr_massmodels",
        "selection_role": "public_data_bridge",
        "run_id": None,
        "run_dir": None,
        "carrier_patch_count": None,
        "materialized_observer_count": None,
        "run_dependent": False,
        "model_role": "oph_phenomenological_continuation",
        "claim_level": "continuation",
        "physical_prediction_receipt": False,
        "source_bundle": {"files": source_files},
        "integrity_receipt": not errors,
        "integrity_errors": sorted(set(errors)),
    }
    if not report:
        return [
            {
                **common,
                "comparison_id": "sparc:unavailable",
                "evaluation_class": "blocked",
                "comparison_receipt": False,
                "rank_eligible_within_planck_diagnostic_lane": False,
                "metrics": {},
                "claim_boundary": "SPARC tables were missing or invalid; no galaxy comparison was made.",
            }
        ]

    a0 = _finite_or_none(report.get("shared_a0"))
    collar = _finite_or_none(report.get("shared_lambda_collar"))
    effective_a0 = a0 / (collar * collar) if a0 is not None and collar not in (None, 0.0) else None
    rar = {
        **common,
        "comparison_id": "sparc:rar_calibration",
        "evaluation_class": "calibrated_same_data",
        "data_use": {
            "fit_to_evaluation_data": True,
            "calibrated_on": "aggregate_RAR_acceleration_rows",
            "holdout": False,
        },
        "parameter_identifiability": {
            "reported_parameters": ["a0", "lambda_collar"],
            "identifiable_combinations": ["effective_a0=a0/lambda_collar^2"],
            "identifiable_parameter_count": 1,
            "separate_a0_and_lambda_identifiable": False,
            "effective_a0": effective_a0,
        },
        "metrics": {
            "rar_scatter": _metric(
                report.get("rar_scatter_dex"), unit="dex", direction="lower_is_better"
            ),
            "rar_point_count": _metric(
                report.get("rar_point_count"), unit="points", direction="descriptive"
            ),
            "rar_galaxy_label_count": _metric(
                report.get("rar_galaxy_count"), unit="galaxy_labels", direction="descriptive"
            ),
            "effective_a0": _metric(
                effective_a0, unit="m/s^2", direction="descriptive"
            ),
        },
        "comparison_receipt": bool(report.get("receipt")),
        "rank_eligible_within_planck_diagnostic_lane": False,
        "rank_exclusion_reasons": ["same_data_calibration", "different_domain"],
        "assessment": "positive calibration fit, not an out-of-sample prediction",
        "claim_boundary": (
            "The aggregate RAR table calibrates the continuation law. The public RAR rows do not carry "
            "175 distinct galaxy labels; companion SPARC tables must not be used to relabel the aggregate "
            "RAR curve. Only a0/lambda_collar^2 is identifiable in this law."
        ),
    }

    fixed_rar = report.get("fixed_oph_rar_comparison") or {}
    fixed_rar_branches = fixed_rar.get("branches") or {}
    z6_rar = fixed_rar_branches.get("z6_exact_uniform_target") or {}
    unit_rar = fixed_rar_branches.get("unit_lambda_endpoint") or {}
    best_rar = fixed_rar_branches.get("same_data_best_fit") or {}
    fixed_rar_row = {
        **common,
        "comparison_id": "sparc:rar_fixed_oph_branches",
        "model_role": "oph_fixed_static_continuation",
        "evaluation_class": "calibrated_independent_dataset",
        "data_use": {
            "fit_to_evaluation_data": False,
            "parameters_fit_to_rar": False,
            "oph_a0_source": "external_cosmology_benchmark",
            "oph_lambda_source": "conditional_exact_uniform_product_thickening_target",
            "evaluated_on": "SPARC_RAR_Lelli2017",
            "aggregate_rows_have_galaxy_ids": False,
        },
        "metrics": {
            "z6_effective_a0": _metric(
                z6_rar.get("effective_a0_m_s2"), unit="m/s^2", direction="descriptive"
            ),
            "z6_rms_residual": _metric(
                z6_rar.get("rms_residual_dex"), unit="dex", direction="lower_is_better"
            ),
            "z6_mean_residual_data_minus_model": _metric(
                z6_rar.get("mean_residual_data_minus_model_dex"),
                unit="dex",
                direction="closer_to_zero_is_better",
            ),
            "z6_binned_weighted_rms_residual": _metric(
                (z6_rar.get("binned") or {}).get("point_count_weighted_rms_residual_dex"),
                unit="dex",
                direction="lower_is_better",
            ),
            "unit_lambda_rms_residual": _metric(
                unit_rar.get("rms_residual_dex"), unit="dex", direction="lower_is_better"
            ),
            "same_data_best_rms_residual": _metric(
                best_rar.get("rms_residual_dex"), unit="dex", direction="lower_is_better"
            ),
            "z6_minus_same_data_best_rms": _metric(
                (fixed_rar.get("z6_vs_same_data_best") or {}).get("rms_residual_delta_dex"),
                unit="dex",
                direction="closer_to_zero_is_better",
            ),
            "rar_point_count": _metric(
                fixed_rar.get("point_count"), unit="points", direction="descriptive"
            ),
            "rar_source_galaxy_count": _metric(
                fixed_rar.get("public_rar_galaxy_count"),
                unit="galaxies",
                direction="descriptive",
            ),
        },
        "comparison_receipt": bool(fixed_rar.get("comparison_receipt")),
        "fixed_formula_report": fixed_rar,
        "rank_eligible_within_planck_diagnostic_lane": False,
        "rank_exclusion_reasons": [
            "different_domain",
            "retrospective_calibrated_coefficient",
            "aggregate_rows_share_galaxy_systematics",
        ],
        "assessment": str(fixed_rar.get("assessment") or "Fixed RAR comparison unavailable."),
        "claim_boundary": str(fixed_rar.get("claim_boundary") or "Fixed RAR comparison unavailable."),
    }

    btfr = report.get("btfr_prediction_from_rar_fit") or {}
    btfr_eiv = report.get("btfr_error_aware_likelihood") or {}
    free_btfr = btfr_eiv.get("free_orthogonal_ml_fit") or {}
    slope_four_btfr = btfr_eiv.get("slope_four_fit") or {}
    slope_test = btfr_eiv.get("oph_slope_test") or {}
    fixed_branches = btfr_eiv.get("oph_fixed_normalization_branches") or {}
    z6_btfr = fixed_branches.get("z6_exact_uniform_target") or {}
    unit_btfr = fixed_branches.get("unit_lambda_endpoint") or {}
    btfr_row = {
        **common,
        "comparison_id": "sparc:btfr_independent_table_check",
        "evaluation_class": "calibrated_independent_dataset",
        "data_use": {
            "fit_to_evaluation_data": False,
            "parameters_fit_to_btfr": False,
            "oph_a0_source": "external_cosmology_benchmark",
            "oph_lambda_source": "conditional_exact_uniform_product_thickening_target",
            "evaluated_on": "SPARC_BTFR_Lelli2019",
            "holdout": True,
            "holdout_unit": "dataset",
        },
        "metrics": {
            "predicted_log_mass_vs_log_velocity_slope": _metric(
                slope_test.get("predicted_slope"), unit="dimensionless", direction="descriptive"
            ),
            "observed_error_aware_log_mass_vs_log_velocity_slope": _metric(
                slope_test.get("observed_slope"), unit="dimensionless", direction="descriptive"
            ),
            "observed_minus_predicted_slope": _metric(
                (
                    float(slope_test["observed_slope"]) - float(slope_test["predicted_slope"])
                    if slope_test.get("observed_slope") is not None
                    and slope_test.get("predicted_slope") is not None
                    else None
                ),
                unit="dimensionless",
                direction="closer_to_zero_is_better",
            ),
            "predicted_minus_observed_slope_pull": _metric(
                slope_test.get("predicted_minus_observed_pull_sigma"),
                unit="sigma",
                direction="closer_to_zero_is_better",
            ),
            "observed_slope_standard_error": _metric(
                slope_test.get("observed_slope_standard_error_hessian"),
                unit="dimensionless",
                direction="descriptive",
            ),
            "slope_four_observed_pivot_mass_at_100_km_s": _metric(
                slope_four_btfr.get("pivot_log10_mass_msun"),
                unit="log10(Msun)",
                direction="descriptive",
            ),
            "z6_predicted_pivot_mass_at_100_km_s": _metric(
                z6_btfr.get("predicted_pivot_log10_mass_msun_at_100_km_s"),
                unit="log10(Msun)",
                direction="descriptive",
            ),
            "z6_observed_minus_predicted_pivot": _metric(
                z6_btfr.get("observed_minus_predicted_pivot_dex"),
                unit="dex",
                direction="closer_to_zero_is_better",
            ),
            "z6_normalization_stat_only_pull": _metric(
                z6_btfr.get("stat_only_normalization_pull_sigma"),
                unit="sigma",
                direction="closer_to_zero_is_better",
            ),
            "unit_lambda_normalization_stat_only_pull": _metric(
                unit_btfr.get("stat_only_normalization_pull_sigma"),
                unit="sigma",
                direction="closer_to_zero_is_better",
            ),
            "orthogonal_intrinsic_scatter": _metric(
                free_btfr.get("intrinsic_scatter_perpendicular_dex"),
                unit="dex",
                direction="lower_is_better",
            ),
            "galaxy_count": _metric(
                btfr_eiv.get("row_count"), unit="galaxies", direction="descriptive"
            ),
        },
        "comparison_receipt": bool(btfr_eiv.get("comparison_receipt")),
        "rank_eligible_within_planck_diagnostic_lane": False,
        "rank_exclusion_reasons": ["different_domain", "no_prefrozen_btfr_acceptance_threshold"],
        "superseded_diagnostics": {
            "naive_unweighted_ols_slope": btfr.get("observed_slope_logM_vs_logV"),
            "invalid_unpivoted_mixed_slope_intercept_delta_dex": btfr.get(
                "intercept_delta_observed_minus_predicted"
            ),
            "reason": (
                "Unweighted OLS discards both-axis errors. Intercepts from unequal slopes were compared "
                "at V=1 km/s, far outside the sample, and are not a normalization test."
            ),
        },
        "error_aware_likelihood": btfr_eiv,
        "assessment": (
            "corrected mixed result: the asymptotic slope 4 is compatible with the error-aware fit, "
            "while the fixed OPH normalization is high under the table's fixed mass-to-light convention; "
            "the normalization pull is statistical only and global galaxy systematics are not marginalized"
        ),
        "claim_boundary": str(btfr_eiv.get("claim_boundary") or "BTFR comparison unavailable."),
    }

    holdout = report.get("holdout_validation") or {}
    test = holdout.get("test") or {}
    train_a0 = _finite_or_none(holdout.get("shared_a0"))
    train_collar = _finite_or_none(holdout.get("shared_lambda_collar"))
    holdout_effective_a0 = (
        train_a0 / (train_collar * train_collar)
        if train_a0 is not None and train_collar not in (None, 0.0)
        else None
    )
    holdout_row = {
        **common,
        "comparison_id": "sparc:galaxy_level_massmodel_holdout",
        "evaluation_class": "heldout_test",
        "data_use": {
            "fit_to_evaluation_data": False,
            "holdout": True,
            "holdout_unit": "galaxy",
            "split_seed": holdout.get("split_seed"),
            "train_galaxy_count": holdout.get("train_galaxy_count"),
            "test_galaxy_count": holdout.get("test_galaxy_count"),
            "fixed_upsilon_disk": holdout.get("fixed_upsilon_disk"),
            "fixed_upsilon_bulge": holdout.get("fixed_upsilon_bulge"),
        },
        "parameter_identifiability": {
            "identifiable_combinations": ["effective_a0=a0/lambda_collar^2"],
            "identifiable_parameter_count": 1,
            "effective_a0_fitted_on_train": holdout_effective_a0,
        },
        "metrics": {
            "test_log_acceleration_rmse": _metric(
                test.get("log_acceleration_rmse_dex"), unit="dex", direction="lower_is_better"
            ),
            "test_baryon_only_log_acceleration_rmse": _metric(
                test.get("baryon_only_log_acceleration_rmse_dex"),
                unit="dex",
                direction="lower_is_better",
            ),
            "test_velocity_rmse": _metric(
                test.get("velocity_rmse_km_s"), unit="km/s", direction="lower_is_better"
            ),
            "test_baryon_only_velocity_rmse": _metric(
                test.get("baryon_only_velocity_rmse_km_s"),
                unit="km/s",
                direction="lower_is_better",
            ),
            "test_velocity_rmse_improvement_fraction": _metric(
                test.get("velocity_rmse_improvement_fraction"),
                unit="fraction",
                direction="higher_is_better",
            ),
            "test_velocity_diagonal_chi2_proxy_per_point": _metric(
                test.get("velocity_chi2_proxy_per_point"),
                unit="dimensionless",
                direction="lower_is_better",
            ),
            "test_galaxy_count": _metric(
                holdout.get("test_galaxy_count"), unit="galaxies", direction="descriptive"
            ),
            "test_point_count": _metric(
                holdout.get("test_point_count"), unit="points", direction="descriptive"
            ),
        },
        "comparison_receipt": bool(holdout.get("usable") and holdout.get("receipt")),
        "rank_eligible_within_planck_diagnostic_lane": False,
        "rank_exclusion_reasons": ["different_domain"],
        "assessment": (
            "positive held-out RMSE improvement over baryons alone, alongside a large diagonal velocity "
            "chi-square proxy; phenomenological continuation, not a full galaxy likelihood"
        ),
        "claim_boundary": str(holdout.get("claim_boundary") or "Galaxy holdout unavailable."),
    }
    return [rar, fixed_rar_row, btfr_row, holdout_row]


def _compressed_reference_comparison(run_dir: Path) -> dict[str, Any] | None:
    path = Path(run_dir) / "oph_compressed_likelihood_report.json"
    report, error = _load_json(path)
    if error:
        return None
    point = report.get("oph_compressed_point") or {}
    scans = report.get("scan_points") or []
    fixed = next((row for row in scans if row.get("case") == "Best grid point, H0 fixed"), {})
    free = next((row for row in scans if row.get("case") == "Free compressed best"), {})
    return {
        "comparison_id": "compressed_cosmology:static_reference",
        "domain": "compressed_cosmology",
        "dataset_id": "compressed_cosmology_reference",
        "selection_role": "historical_regression_reference",
        "run_id": None,
        "run_dir": str(run_dir),
        "carrier_patch_count": None,
        "materialized_observer_count": None,
        "run_dependent": False,
        "model_role": "oph_historical_compressed_point",
        "claim_level": "proxy",
        "evaluation_class": "blocked",
        "data_use": {
            "public_data_files_attached": False,
            "static_hard_coded_reference": True,
            "invalidated_archival_point": True,
            "rejected_neutrino_input": True,
        },
        "metrics": {
            "oph_reference_diagonal_chi2": _metric(
                point.get("chi2_diag"), unit="dimensionless", direction="lower_is_better"
            ),
            "fixed_h0_grid_best_diagonal_chi2": _metric(
                fixed.get("chi2"), unit="dimensionless", direction="lower_is_better"
            ),
            "free_compressed_best_diagonal_chi2": _metric(
                free.get("chi2"), unit="dimensionless", direction="lower_is_better"
            ),
        },
        "comparison_receipt": False,
        "physical_prediction_receipt": False,
        "rank_eligible_within_planck_diagnostic_lane": False,
        "rank_exclusion_reasons": [
            "static_reference_not_run_derived",
            "public_data_files_and_covariances_not_attached",
            "different_domain",
        ],
        "integrity_receipt": True,
        "integrity_errors": [],
        "source_bundle": {"files": [_file_hash_record(path)]},
        "assessment": (
            "invalidated archival diagnostic: hard-coded values, no attached public covariance, and a "
            "rejected neutrino input; excluded from the evidence scoreboard"
        ),
        "claim_boundary": (
            "The archived constants are retained only for regression provenance. Reproducing a hard-coded "
            "chi-square against itself is not a comparison receipt, and this row must not be interpreted as "
            "evidence for or against current OPH."
        ),
    }


def _repair_readiness(run_dir: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    paired_path = Path(run_dir) / "paired_b_a_perturbation_report.json"
    kernel_path = Path(run_dir) / "B_A_kernel_report.json"
    paired, paired_error = _load_json(paired_path)
    kernel, kernel_error = _load_json(kernel_path)
    errors = [value for value in (paired_error, kernel_error) if value]
    readiness = paired.get("readiness") or {}
    checks = readiness.get("checks") or {}
    return {
        "run_id": bundle.get("run_id"),
        "run_dir": str(run_dir),
        "selection_role": bundle.get("selection_role"),
        "carrier_patch_count": bundle.get("carrier_patch_count"),
        "materialized_observer_count": bundle.get("materialized_observer_count"),
        "paired_reruns_present": bool(checks.get("real_baryon_perturbation_runs_present")),
        "paired_full_rerun": bool(checks.get("full_perturbation_rerun")),
        "controls_fail_as_required": bool(checks.get("controls_fail")),
        "sign_stable": bool(checks.get("sign_stable")),
        "paired_b_a_diagnostic_receipt": bool(readiness.get("B_A_PAIRED_DIAGNOSTIC_RECEIPT")),
        "b_a_parent_receipt": bool(readiness.get("B_A_PARENT_RECEIPT")),
        "b_a_kernel_candidate_receipt": bool(kernel.get("B_A_KERNEL_CANDIDATE_RECEIPT")),
        "b_a_physical_kernel_receipt": bool(kernel.get("B_A_KERNEL_RECEIPT")),
        "physical_prediction_ready": bool(readiness.get("physical_prediction_ready")),
        "kernel_row_count": kernel.get("row_count"),
        "missing_gates": list(readiness.get("missing_gates") or []),
        "promotion_blockers": list(kernel.get("promotion_blockers") or []),
        "integrity_receipt": not errors,
        "integrity_errors": errors,
        "source_bundle": {
            "files": [_file_hash_record(paired_path), _file_hash_record(kernel_path)]
        },
        "claim_boundary": (
            "Scale-dependent repair/Boltzmann readiness only. This lane uses no public target data and "
            "cannot be ranked against Planck or SPARC; it records whether a larger run closes physical-source gates."
        ),
    }


def _run_bundle(run_dir: Path, *, primary: Path, baseline: Path) -> dict[str, Any]:
    path = Path(run_dir)
    manifest_path = path / "manifest.json"
    config_path = path / "config.yml"
    manifest, manifest_error = _load_json(manifest_path)
    config, config_error = _load_yaml(config_path)
    errors = [value for value in (manifest_error, config_error) if value]
    patch_manifest = _int_or_none(manifest.get("patch_count"))
    patch_config = _int_or_none((config.get("graph") or {}).get("patch_count"))
    if patch_manifest is not None and patch_config is not None and patch_manifest != patch_config:
        errors.append("manifest_config_patch_count_mismatch")
    observer_count = _int_or_none((config.get("observers") or {}).get("sample_count"))
    prep = config.get("million_patch_preparation") or {}
    prep_observers = _int_or_none(prep.get("materialized_observer_count"))
    if observer_count is not None and prep_observers is not None and observer_count != prep_observers:
        errors.append("config_preparation_observer_count_mismatch")
    patch_count = patch_manifest if patch_manifest is not None else patch_config
    role = "primary" if _same_path(path, primary) else "history"
    if _same_path(path, baseline):
        role = "primary_baseline" if role == "primary" else "baseline"
    git_commit_path = path / "git_commit.txt"
    git_commit = git_commit_path.read_text(encoding="utf-8").strip() if git_commit_path.is_file() else None
    return {
        "run_id": str(manifest.get("run_id") or path.name),
        "run_dir": str(path),
        "selection_role": role,
        "carrier_patch_count": patch_count,
        "materialized_observer_count": observer_count,
        "scale_label": _scale_label(patch_count, observer_count),
        "is_at_least_one_million_carrier_patches": bool(patch_count is not None and patch_count >= 1_000_000),
        "is_at_least_one_million_materialized_observers": bool(
            observer_count is not None and observer_count >= 1_000_000
        ),
        "git_commit": manifest.get("git_commit") or git_commit,
        "manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
        "config_sha256": _sha256(config_path) if config_path.is_file() else None,
        "integrity_receipt": not errors,
        "integrity_errors": errors,
    }


def _config_scale_contract(path: Path, *, planck_rows: list[dict[str, float]], role: str) -> dict[str, Any]:
    config, error = _load_yaml(path)
    if error:
        return {
            "config_path": str(path),
            "role": role,
            "integrity_receipt": False,
            "integrity_errors": [error],
        }
    prep = config.get("million_patch_preparation") or {}
    patches = _int_or_none(prep.get("carrier_patch_count"))
    if patches is None:
        patches = _int_or_none((config.get("graph") or {}).get("patch_count"))
    observers = _int_or_none(prep.get("materialized_observer_count"))
    if observers is None:
        observers = _int_or_none((config.get("observers") or {}).get("sample_count"))
    ell_max = _int_or_none(
        ((config.get("cosmology") or {}).get("angular_power") or {}).get("ell_max")
    )
    planck_min = min((row["ell"] for row in planck_rows), default=None)
    real_ell_overlap = bool(ell_max is not None and planck_min is not None and ell_max >= planck_min)
    errors: list[str] = []
    if patches is None:
        errors.append("carrier_patch_count_missing")
    if observers is None:
        errors.append("materialized_observer_count_missing")
    return {
        "config_path": str(path),
        "config_sha256": _sha256(path),
        "role": role,
        "carrier_patch_count": patches,
        "materialized_observer_count": observers,
        "scale_label": _scale_label(patches, observers),
        "is_at_least_one_million_carrier_patches": bool(patches is not None and patches >= 1_000_000),
        "is_at_least_one_million_materialized_observers": bool(
            observers is not None and observers >= 1_000_000
        ),
        "screen_angular_ell_max": ell_max,
        "planck_binned_ell_min": planck_min,
        "raw_screen_real_ell_overlap_with_planck_bins": real_ell_overlap,
        "screen_comparison_warning": (
            None
            if real_ell_overlap
            else "configured raw screen ell range does not overlap the local Planck binned TT range"
        ),
        "integrity_receipt": not errors,
        "integrity_errors": errors,
        "claim_boundary": str(config.get("claim_boundary") or ""),
    }


def _primary_baseline_delta(
    cmb_rows: list[dict[str, Any]], *, primary: Path, baseline: Path
) -> dict[str, Any]:
    primary_row = next((row for row in cmb_rows if _same_path(Path(row["run_dir"]), primary)), None)
    baseline_row = next((row for row in cmb_rows if _same_path(Path(row["run_dir"]), baseline)), None)
    if primary_row is None or baseline_row is None:
        return {"status": "not_comparable", "reason": "primary_or_baseline_row_missing"}
    p_value = _metric_value(primary_row, "diagonal_chi2_per_bin_after_one_amplitude_fit")
    b_value = _metric_value(baseline_row, "diagonal_chi2_per_bin_after_one_amplitude_fit")
    if p_value is None or b_value is None:
        return {"status": "not_comparable", "reason": "primary_or_baseline_metric_missing"}
    p_bins = _metric_value(primary_row, "bin_count")
    delta = p_value - b_value
    same_artifact = _comparison_source_hash(primary_row) == _comparison_source_hash(baseline_row)
    if same_artifact or math.isclose(delta, 0.0, abs_tol=1e-12):
        verdict = "tied_or_identical_artifact"
    elif delta < 0:
        verdict = "diagnostic_improved"
    else:
        verdict = "diagnostic_regressed"
    p_patches = _int_or_none(primary_row.get("carrier_patch_count"))
    b_patches = _int_or_none(baseline_row.get("carrier_patch_count"))
    return {
        "status": "comparable",
        "primary_comparison_id": primary_row.get("comparison_id"),
        "baseline_comparison_id": baseline_row.get("comparison_id"),
        "primary_value": p_value,
        "baseline_value": b_value,
        "delta_primary_minus_baseline_per_bin": delta,
        "delta_primary_minus_baseline_total": delta * p_bins if p_bins is not None else None,
        "carrier_patch_ratio": (
            p_patches / b_patches if p_patches is not None and b_patches not in (None, 0) else None
        ),
        "same_source_artifact_hash": same_artifact,
        "diagnostic_verdict": verdict,
        "physical_prediction_implication": False,
    }


def _diagnostic_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = []
    for row in rows:
        value = _metric_value(row, "diagonal_chi2_per_bin_after_one_amplitude_fit")
        if row.get("rank_eligible_within_planck_diagnostic_lane") and value is not None:
            eligible.append((value, str(row.get("comparison_id")), row))
    eligible.sort(key=lambda item: (item[0], item[1]))
    return [
        {
            "diagnostic_rank": index,
            "comparison_id": row.get("comparison_id"),
            "run_id": row.get("run_id"),
            "selection_role": row.get("selection_role"),
            "diagonal_chi2_per_bin_after_one_amplitude_fit": value,
            "physical_prediction_receipt": bool(row.get("physical_prediction_receipt")),
        }
        for index, (value, _, row) in enumerate(eligible, start=1)
    ]


def _featured_by_evidence_class(
    comparisons: list[dict[str, Any]], diagnostic_order: list[dict[str, Any]]
) -> dict[str, Any]:
    by_id = {str(row.get("comparison_id")): row for row in comparisons}
    heldout = [
        row.get("comparison_id")
        for row in comparisons
        if row.get("evaluation_class") == "heldout_test" and row.get("comparison_receipt")
    ]
    independent = [
        row.get("comparison_id")
        for row in comparisons
        if row.get("evaluation_class") == "calibrated_independent_dataset"
        and row.get("comparison_receipt")
    ]
    conditional_external = [
        row.get("comparison_id")
        for row in comparisons
        if row.get("conditional_external_domain_falsifier") and row.get("comparison_receipt")
    ]
    primary_cmb = next(
        (
            row.get("comparison_id")
            for row in comparisons
            if row.get("domain") == "cmb_tt"
            and row.get("selection_role") in {"primary", "primary_baseline"}
            and row.get("comparison_receipt")
        ),
        None,
    )
    best_diag = diagnostic_order[0]["comparison_id"] if diagnostic_order else None
    return {
        "primary_run_cmb_diagnostic": primary_cmb,
        "lowest_chi2_declared_run_cmb_diagnostic": best_diag,
        "heldout_tests": heldout,
        "independent_dataset_checks": independent,
        "conditional_external_domain_falsifiers": conditional_external,
        "frozen_predictions": [
            row_id
            for row_id, row in by_id.items()
            if row.get("physical_prediction_receipt")
        ],
        "note": "These are separate evidence classes; none is an overall OPH winner.",
    }


def _read_planck_rows(path: Path) -> tuple[list[dict[str, float]], list[str]]:
    if not path.is_file():
        return [], [f"missing_planck_tt_table:{path}"]
    rows: list[dict[str, float]] = []
    errors: list[str] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            if len(parts) < 5:
                errors.append(f"malformed_planck_row:{line_number}")
                continue
            values = [float(value) for value in parts[:5]]
            rows.append(
                {
                    "ell": values[0],
                    "observed_D_ell": values[1],
                    "minus_dD_ell": values[2],
                    "plus_dD_ell": values[3],
                    "best_fit_D_ell": values[4],
                }
            )
    except (OSError, ValueError) as exc:
        errors.append(f"invalid_planck_tt_table:{type(exc).__name__}:{exc}")
    if not rows:
        errors.append("planck_tt_table_has_no_data_rows")
    return rows, errors


def _planck_alignment_ok(embedded: list[dict[str, Any]], planck_rows: list[dict[str, float]]) -> bool:
    if len(embedded) != len(planck_rows) or not embedded:
        return False
    for source_row, data_row in zip(embedded, planck_rows, strict=True):
        values = (
            (_finite_or_none(source_row.get("ell")), data_row["ell"]),
            (_finite_or_none(source_row.get("observed_D_ell")), data_row["observed_D_ell"]),
            (_finite_or_none(source_row.get("sigma_D_ell")), data_row["plus_dD_ell"]),
        )
        if any(left is None or not math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-9) for left, right in values):
            return False
    return True


def _profiled_chi2_per_bin(rows: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for row in rows:
        observed = _finite_or_none(row.get("observed_D_ell"))
        sigma = _finite_or_none(row.get("sigma_D_ell"))
        profiled = _finite_or_none(row.get("amplitude_fit_camb_D_ell"))
        if observed is None or sigma in (None, 0.0) or profiled is None:
            return None
        values.append(((profiled - observed) / sigma) ** 2)
    return sum(values) / len(values) if values else None


def _residual_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    profiled: list[float] = []
    raw: list[float] = []
    for row in rows:
        observed = _finite_or_none(row.get("observed_D_ell"))
        sigma = _finite_or_none(row.get("sigma_D_ell"))
        fit = _finite_or_none(row.get("amplitude_fit_camb_D_ell"))
        raw_value = _finite_or_none(row.get("camb_D_ell"))
        if observed is None or sigma in (None, 0.0):
            continue
        if fit is not None:
            profiled.append((fit - observed) / sigma)
        if raw_value is not None:
            raw.append((raw_value - observed) / sigma)
    return {
        "profiled_rms_sigma": _rms(profiled),
        "raw_rms_sigma": _rms(raw),
    }


def _dataset_record(
    *,
    dataset_id: str,
    release: str,
    path: Path,
    public_source: str,
    record_count: int,
    errors: list[str],
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "release": release,
        "path": str(path),
        "public_source": public_source,
        "sha256": _sha256(path) if path.is_file() else None,
        "record_count": int(record_count),
        "integrity_receipt": not errors,
        "integrity_errors": list(errors),
    }


def _metric(
    value: Any,
    *,
    unit: str,
    direction: str,
    baseline_value: Any = None,
    delta: Any = None,
) -> dict[str, Any]:
    return {
        "value": _finite_or_none(value),
        "unit": unit,
        "direction": direction,
        "baseline_value": _finite_or_none(baseline_value),
        "delta": _finite_or_none(delta),
    }


def _metric_value(row: dict[str, Any], name: str) -> float | None:
    metric = (row.get("metrics") or {}).get(name) or {}
    return _finite_or_none(metric.get("value"))


def _comparison_source_hash(row: dict[str, Any]) -> str | None:
    for file_row in (row.get("source_bundle") or {}).get("files", []):
        if str(file_row.get("path", "")).endswith("finite_repair_clock_cmb_camb_report.json"):
            return file_row.get("sha256")
    return None


def _file_hash_record(path: Path) -> dict[str, Any]:
    file_path = Path(path)
    return {
        "path": str(file_path),
        "exists": file_path.is_file(),
        "size_bytes": file_path.stat().st_size if file_path.is_file() else None,
        "sha256": _sha256(file_path) if file_path.is_file() else None,
    }


def _load_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, f"missing_json:{path}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"invalid_json:{path}:{type(exc).__name__}"
    if not isinstance(value, dict):
        return {}, f"json_root_not_object:{path}"
    return value, None


def _load_yaml(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, f"missing_yaml:{path}"
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return {}, f"invalid_yaml:{path}:{type(exc).__name__}"
    if not isinstance(value, dict):
        return {}, f"yaml_root_not_object:{path}"
    return value, None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_paths(paths: Iterable[Path], *, exclude: Path | None = None) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    excluded = _path_key(exclude) if exclude is not None else None
    for value in paths:
        path = Path(value)
        key = _path_key(path)
        if key == excluded or key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _path_key(path: Path | None) -> str:
    return str(Path(path).resolve()) if path is not None else ""


def _same_path(left: Path, right: Path) -> bool:
    return _path_key(left) == _path_key(right)


def _scale_label(patches: int | None, observers: int | None) -> str:
    if observers is not None and observers >= 1_000_000:
        return "million_materialized_observers"
    if patches is not None and patches >= 1_000_000:
        return "million_patch_bounded_observer_sample"
    return "finite_patch_observer_sample"


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _rms(values: list[float]) -> float | None:
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else None


def _write_metrics_csv(path: Path, report: dict[str, Any]) -> None:
    fieldnames = [
        "comparison_id",
        "domain",
        "dataset_id",
        "evaluation_class",
        "selection_role",
        "run_id",
        "run_dependent",
        "model_id",
        "model_role",
        "metric",
        "value",
        "unit",
        "direction",
        "baseline_value",
        "delta",
        "comparison_receipt",
        "physical_prediction_receipt",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for comparison in report.get("comparisons", []):
            for name, metric in (comparison.get("metrics") or {}).items():
                writer.writerow(
                    {
                        "comparison_id": comparison.get("comparison_id"),
                        "domain": comparison.get("domain"),
                        "dataset_id": comparison.get("dataset_id"),
                        "evaluation_class": comparison.get("evaluation_class"),
                        "selection_role": comparison.get("selection_role"),
                        "run_id": comparison.get("run_id"),
                        "run_dependent": comparison.get("run_dependent"),
                        "model_id": comparison.get("model_id"),
                        "model_role": comparison.get("model_role"),
                        "metric": name,
                        "value": metric.get("value"),
                        "unit": metric.get("unit"),
                        "direction": metric.get("direction"),
                        "baseline_value": metric.get("baseline_value"),
                        "delta": metric.get("delta"),
                        "comparison_receipt": comparison.get("comparison_receipt"),
                        "physical_prediction_receipt": comparison.get("physical_prediction_receipt"),
                    }
                )


def _markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Best available public-data comparisons",
        "",
        f"Primary run: `{report['selection_policy']['primary_run']}`",
        f"Baseline run: `{report['selection_policy']['baseline_run']}`",
        "",
        "There is no cross-domain OPH score or overall winner. Results are separated by evidence class.",
        "",
        "## Current scoreboard",
        "",
        "| Evidence | Evaluation class | Result | Status |",
        "|---|---|---:|---|",
    ]
    comparisons = {row.get("comparison_id"): row for row in report.get("comparisons", [])}
    primary_cmb_id = report["featured_by_evidence_class"].get("primary_run_cmb_diagnostic")
    primary_cmb = comparisons.get(primary_cmb_id) if primary_cmb_id else None
    if primary_cmb:
        chi2 = _metric_value(primary_cmb, "diagonal_chi2_per_bin_after_one_amplitude_fit")
        baseline = (primary_cmb.get("baseline") or {}).get(
            "diagonal_chi2_per_bin_after_one_amplitude_fit"
        )
        delta = (primary_cmb.get("baseline") or {}).get("oph_minus_baseline_total_over_bins")
        lines.append(
            "| Planck PR3 TT (primary OPH curve) | calibrated same-data diagnostic | "
            f"chi2/bin {_fmt(chi2)} vs LambdaCDM {_fmt(baseline)}; total delta {_fmt(delta)} | "
            "comparison only |"
        )
    cassini = comparisons.get("cassini:conditional_static_external_field")
    if cassini:
        q2 = _metric_value(cassini, "z6_Q2")
        raw_pull = _metric_value(cassini, "z6_raw_fixed_input_pull")
        gaia_pull = _metric_value(cassini, "z6_gaia_only_combined_pull")
        lines.append(
            "| Cassini Solar-System external field | conditional independent-dataset test | "
            f"Q2 {_fmt(q2)} s^-2; raw fixed-input pull {_fmt(raw_pull)} sigma; "
            f"Gaia-only combined pull {_fmt(gaia_pull)} sigma | universal extension strongly excluded; "
            "current OPH applicability gate missing |"
        )
    for comparison_id, label, metric_name in (
        ("sparc:rar_calibration", "SPARC RAR", "rar_scatter"),
        ("sparc:galaxy_level_massmodel_holdout", "SPARC mass-model holdout", "test_log_acceleration_rmse"),
        ("sparc:btfr_independent_table_check", "SPARC BTFR", "observed_minus_predicted_slope"),
        ("compressed_cosmology:static_reference", "Compressed cosmology reference", "oph_reference_diagonal_chi2"),
    ):
        row = comparisons.get(comparison_id)
        if row:
            metric = (row.get("metrics") or {}).get(metric_name) or {}
            lines.append(
                f"| {label} | {row.get('evaluation_class')} | {_fmt(metric.get('value'))} {metric.get('unit') or ''} | "
                f"{row.get('assessment') or 'descriptive'} |"
            )
    lines.extend(
        [
            "",
            "## Prediction status",
            "",
            f"Frozen physical predictions: **{summary['frozen_prediction_count']}**.",
            "",
            "A comparison receipt is not a prediction receipt. The Planck lane currently profiles an amplitude "
            "on the same 83 bins and uses diagonal errors, while the SPARC lane is a phenomenological continuation.",
        ]
    )
    planned = report.get("planned_run_scale_contract")
    if planned:
        lines.extend(
            [
                "",
                "## Planned scale contract",
                "",
                f"- carrier patches: `{planned.get('carrier_patch_count')}`",
                f"- materialized observers: `{planned.get('materialized_observer_count')}`",
                f"- label: `{planned.get('scale_label')}`",
                f"- raw screen/Planck real-ell overlap: `{planned.get('raw_screen_real_ell_overlap_with_planck_bins')}`",
            ]
        )
        if planned.get("screen_comparison_warning"):
            lines.append(f"- warning: {planned['screen_comparison_warning']}")
    delta = report.get("primary_vs_baseline") or {}
    lines.extend(
        [
            "",
            "## Primary versus declared baseline",
            "",
            f"- status: `{delta.get('status')}`",
            f"- diagnostic verdict: `{delta.get('diagnostic_verdict')}`",
            f"- delta chi2/bin: `{_fmt(delta.get('delta_primary_minus_baseline_per_bin'))}`",
            f"- delta total diagonal chi2: `{_fmt(delta.get('delta_primary_minus_baseline_total'))}`",
            "",
            "## Integrity",
            "",
            f"Integrity receipt: `{str(summary['integrity_receipt']).lower()}`",
        ]
    )
    for error in summary.get("integrity_errors", []):
        lines.append(f"- `{error}`")
    lines.extend(["", "## Claim boundary", "", str(report.get("claim_boundary") or ""), ""])
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    finite = _finite_or_none(value)
    if finite is None:
        return "n/a"
    return f"{finite:.6g}"
