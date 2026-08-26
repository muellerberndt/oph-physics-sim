from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from oph_fpe.cosmology.cassini_external_field import cassini_external_field_report
from oph_fpe.cosmology.public_data_comparisons import (
    _cassini_dataset_and_comparison,
    public_data_comparison_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = (
    REPO_ROOT / "schemas/cosmology/best_of_public_data_comparisons.schema.json"
)
SPARC_DIR = REPO_ROOT / "data/measurements/sparc"
CANONICAL_PLANCK_PATH = (
    REPO_ROOT
    / "data/measurements/planck2018/COM_PowerSpect_CMB-TT-binned_R3.01.txt"
)
CANONICAL_CASSINI_PATH = (
    REPO_ROOT / "data/measurements/cassini/cassini_q2_2026.json"
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _planck_fixture(path: Path) -> list[dict[str, float]]:
    path.write_bytes(CANONICAL_PLANCK_PATH.read_bytes())
    return _planck_values(path)


def _fake_run(
    root: Path,
    *,
    planck_path: Path,
    chi2_per_bin: float,
    patch_count: int = 65_536,
    observer_count: int = 1_024,
    assert_prediction: bool = False,
) -> Path:
    root.mkdir(parents=True)
    observed_rows = _planck_values(planck_path)
    embedded = _common_amplitude_fixture_rows(
        observed_rows, chi2_per_bin=chi2_per_bin, amplitude=0.9
    )
    model_id = "finite_repair_clock_scalar_tilt"
    source_name = "finite_repair_clock_cmb_camb_report.json"
    benchmark_hash = hashlib.sha256(planck_path.read_bytes()).hexdigest()
    source = {
        "mode": "fixture",
        "input_hashes": {"benchmark_sha256": benchmark_hash},
        "finite_repair_clock_input": {"n_s": 0.98, "finite_lattice_derived": True},
        "selector_ir_input": {"finite_lattice_derived": False},
        "camb": {"baseline_lambda_cdm_parameters": {"H0": 67.36}},
        "comparison": {
            model_id: {
                "usable": True,
                "bin_count": len(embedded),
                "amplitude_fit_chi2_per_bin": chi2_per_bin,
                "best_fit_amplitude": 0.9,
                "shape_correlation": 0.99,
                "normalized_rmse": 0.03,
                "binned_tt_comparison": embedded,
            }
        },
    }
    _write_json(root / source_name, source)
    baseline_chi2_per_bin = 0.1
    baseline_rows = _common_amplitude_fixture_rows(
        observed_rows,
        chi2_per_bin=baseline_chi2_per_bin,
        amplitude=1.0,
    )
    _write_json(
        root / "camb_lcdm_baseline_report.json",
        {
            "mode": "camb_lcdm_baseline_regression",
            "benchmark": {
                "label": "Planck2018_TT_binned",
                "row_count": len(baseline_rows),
            },
            "camb": {
                "lmax": 2600,
                "spectrum": "lensed_total_TT_D_ell",
                "lambda_cdm_parameters": {
                    "H0": 67.36,
                    "ombh2": 0.02237,
                    "omch2": 0.12,
                    "mnu": 0.06,
                    "omk": 0.0,
                    "tau": 0.0544,
                    "As": 2.1e-9,
                    "ns": 0.9649,
                },
            },
            "input_hashes": {
                "benchmark_sha256": benchmark_hash,
                "params_sha256": (
                    "4c8c754b73d19c9f0f3bf0defc098f4435031ce23037367d70e60f1ddd112821"
                ),
            },
            "comparison": {
                "usable": True,
                "bin_count": len(baseline_rows),
                "best_fit_amplitude": 1.0,
                "amplitude_fit_chi2_per_bin": baseline_chi2_per_bin,
                "binned_tt_comparison": baseline_rows,
            },
            "CDM_LIMIT_BOLTZMANN_RECEIPT": True,
            "physical_cmb_prediction": False,
        },
    )
    (root / "finite_repair_clock_cmb_tt_bins.csv").write_text(
        "ell,observed_D_ell,finite_repair_clock_scalar_tilt_D_ell\n",
        encoding="utf-8",
    )
    output = {
        "mode": "physical_cmb_output_comparison_v0",
        "run_dirs": [str(root)],
        "PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT": True,
        "USABLE_PHYSICAL_CMB_DATA_RECEIPT": True,
        "PHYSICAL_CMB_PREDICTION_RECEIPT": assert_prediction,
        "physical_cmb_prediction": assert_prediction,
        "best_oph_diagnostic_model": {
            "source_report": source_name,
            "model_id": model_id,
            "model_role": "oph_diagnostic",
            "amplitude_fit_chi2_per_bin": chi2_per_bin,
        },
        "best_oph_residual_summary": {
            "source_csv": "finite_repair_clock_cmb_tt_bins.csv"
        },
        "rows": [
            {
                "source_report": "camb_lcdm_baseline_report.json",
                "model_id": "lcdm_baseline",
                "model_role": "external_baseline",
                "measurement_comparable": True,
                "physical_cmb_prediction": False,
                "dataset_id": "planck_pr3_tt_binned_r3_01",
                "amplitude_fit_chi2_per_bin": baseline_chi2_per_bin,
                "bin_count": len(embedded),
            },
            {
                "source_report": source_name,
                "model_id": model_id,
                "model_role": "oph_diagnostic",
                "amplitude_fit_chi2_per_bin": chi2_per_bin,
                "bin_count": len(embedded),
            },
        ],
    }
    _write_json(root / "physical_cmb_output_comparison_report.json", output)
    _write_json(
        root / "manifest.json",
        {"run_id": root.name, "patch_count": patch_count, "git_commit": "0" * 40},
    )
    (root / "config.yml").write_text(
        yaml.safe_dump(
            {
                "name": root.name,
                "graph": {"patch_count": patch_count},
                "observers": {"sample_count": observer_count},
            }
        ),
        encoding="utf-8",
    )
    _write_json(
        root / "paired_b_a_perturbation_report.json",
        {
            "readiness": {
                "checks": {},
                "B_A_PAIRED_DIAGNOSTIC_RECEIPT": False,
                "B_A_PARENT_RECEIPT": False,
                "physical_prediction_ready": False,
            }
        },
    )
    _write_json(
        root / "B_A_kernel_report.json",
        {
            "B_A_KERNEL_CANDIDATE_RECEIPT": False,
            "B_A_KERNEL_RECEIPT": False,
            "row_count": 0,
        },
    )
    return root


def _planck_values(path: Path) -> list[dict[str, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        ell, observed, minus, plus, _ = (float(value) for value in line.split()[:5])
        assert minus == plus
        rows.append({"ell": ell, "observed_D_ell": observed, "sigma_D_ell": plus})
    return rows


def _common_amplitude_fixture_rows(
    observed_rows: list[dict[str, float]],
    *,
    chi2_per_bin: float,
    amplitude: float,
) -> list[dict[str, float]]:
    """Construct a curve whose weighted scalar optimum is exactly ``amplitude``."""

    whitened_observed = [
        row["observed_D_ell"] / row["sigma_D_ell"] for row in observed_rows
    ]
    observed_norm_sq = sum(value * value for value in whitened_observed)
    residual_norm_sq = len(observed_rows) * chi2_per_bin
    radial_fraction = 1.0 - residual_norm_sq / observed_norm_sq
    assert 0.0 < radial_fraction <= 1.0
    tangent = [0.0] * len(observed_rows)
    pair_norm = math.hypot(whitened_observed[0], whitened_observed[1])
    tangent[0] = whitened_observed[1] / pair_norm
    tangent[1] = -whitened_observed[0] / pair_norm
    tangent_scale = math.sqrt(
        radial_fraction * (1.0 - radial_fraction) * observed_norm_sq
    )
    fitted_whitened = [
        radial_fraction * observed + tangent_scale * tangent_value
        for observed, tangent_value in zip(
            whitened_observed, tangent, strict=True
        )
    ]
    rows: list[dict[str, float]] = []
    for source, fitted_sigma_units in zip(
        observed_rows, fitted_whitened, strict=True
    ):
        fitted = fitted_sigma_units * source["sigma_D_ell"]
        rows.append(
            {
                "ell": source["ell"],
                "observed_D_ell": source["observed_D_ell"],
                "sigma_D_ell": source["sigma_D_ell"],
                "camb_D_ell": fitted / amplitude,
                "amplitude_fit_camb_D_ell": fitted,
                "best_fit_column_D_ell": source["observed_D_ell"],
            }
        )
    return rows


def _report(
    primary: Path,
    planck: Path,
    *,
    history: list[Path] | None = None,
    baseline: Path | None = None,
    planned_config: Path | None = None,
) -> dict:
    return public_data_comparison_report(
        primary,
        planck_tt_path=planck,
        sparc_dir=SPARC_DIR,
        history_run_dirs=history or [],
        baseline_run_dir=baseline,
        planned_config_path=planned_config,
    )


def _cmb(report: dict, run_id: str) -> dict:
    return next(
        row
        for row in report["comparisons"]
        if row.get("domain") == "cmb_tt" and row.get("run_id") == run_id
    )


def test_suite_recomputes_profiled_residuals_and_validates_schema(
    tmp_path: Path,
) -> None:
    planck = tmp_path / "planck.txt"
    _planck_fixture(planck)
    run = _fake_run(tmp_path / "primary", planck_path=planck, chi2_per_bin=0.25)

    report = _report(run, planck)
    row = _cmb(report, "primary")

    assert row["comparison_receipt"] is True
    assert row["physical_prediction_receipt"] is False
    assert row["metrics"]["diagonal_chi2_per_bin_after_one_amplitude_fit"][
        "value"
    ] == pytest.approx(0.25)
    assert row["metrics"]["profiled_rms_residual_sigma"]["value"] == pytest.approx(0.5)
    assert row["metrics"]["best_fit_amplitude"]["value"] == pytest.approx(0.9)
    assert row["metrics"]["raw_rms_residual_sigma"]["value"] > 0.5
    assert row["baseline"]["binding_receipt"] is True
    assert row["baseline"]["fit_recomputed_from_bound_source_rows"] is True
    assert row["fit_protocol"]["inferential_degrees_of_freedom"] is None
    assert row["fit_protocol"]["p_value"] is None
    assert row["fit_protocol"][
        "n_minus_fitted_parameter_count_used_for_inference"
    ] is False
    assert row["fit_protocol"]["target_used_for_model_selection"] is True
    assert row["source_bundle"]["same_parent_directory"] is True

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(report)

    forged_inference = copy.deepcopy(report)
    forged_cmb = _cmb(forged_inference, "primary")
    forged_cmb["fit_protocol"]["inferential_degrees_of_freedom"] = 82
    forged_cmb["fit_protocol"]["p_value"] = 0.5
    assert list(Draft202012Validator(schema).iter_errors(forged_inference))


def test_cmb_rejects_nonproportional_profiled_curve(tmp_path: Path) -> None:
    planck = tmp_path / "planck.txt"
    _planck_fixture(planck)
    run = _fake_run(tmp_path / "primary", planck_path=planck, chi2_per_bin=0.25)
    source_path = run / "finite_repair_clock_cmb_camb_report.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    comparison = source["comparison"]["finite_repair_clock_scalar_tilt"]
    comparison["binned_tt_comparison"][0]["amplitude_fit_camb_D_ell"] += 1.0
    _write_json(source_path, source)

    row = _cmb(_report(run, planck), "primary")

    assert row["comparison_receipt"] is False
    assert "reported_profiled_curve_not_common_amplitude_scaled" in row[
        "integrity_errors"
    ]


def test_cmb_rejects_wrong_declared_common_amplitude(tmp_path: Path) -> None:
    planck = tmp_path / "planck.txt"
    _planck_fixture(planck)
    run = _fake_run(tmp_path / "primary", planck_path=planck, chi2_per_bin=0.25)
    source_path = run / "finite_repair_clock_cmb_camb_report.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["comparison"]["finite_repair_clock_scalar_tilt"][
        "best_fit_amplitude"
    ] = 0.8
    _write_json(source_path, source)

    row = _cmb(_report(run, planck), "primary")

    assert row["comparison_receipt"] is False
    assert "reported_best_fit_amplitude_does_not_recompute" in row[
        "integrity_errors"
    ]


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("negative_chi2", "bound_external_lcdm_baseline_output_chi2_mismatch"),
        ("nan_chi2", "bound_external_lcdm_baseline_output_chi2_mismatch"),
        ("wrong_bin_count", "bound_external_lcdm_baseline_bin_count_mismatch"),
        ("fractional_bin_count", "bound_external_lcdm_baseline_bin_count_mismatch"),
        ("wrong_dataset", "bound_external_lcdm_baseline_dataset_mismatch"),
        ("wrong_model", "bound_external_lcdm_baseline_missing"),
    ],
)
def test_cmb_bound_baseline_rejects_mutated_output_rows(
    tmp_path: Path, mutation: str, expected_error: str
) -> None:
    planck = tmp_path / "planck.txt"
    _planck_fixture(planck)
    run = _fake_run(tmp_path / "primary", planck_path=planck, chi2_per_bin=0.25)
    output_path = run / "physical_cmb_output_comparison_report.json"
    output = json.loads(output_path.read_text(encoding="utf-8"))
    baseline = output["rows"][0]
    if mutation == "negative_chi2":
        baseline["amplitude_fit_chi2_per_bin"] = -100.0
    elif mutation == "nan_chi2":
        baseline["amplitude_fit_chi2_per_bin"] = float("nan")
    elif mutation == "wrong_bin_count":
        baseline["bin_count"] -= 1
    elif mutation == "fractional_bin_count":
        baseline["bin_count"] = float(baseline["bin_count"]) + 0.5
    elif mutation == "wrong_dataset":
        baseline["dataset_id"] = "forged_planck_dataset"
    elif mutation == "wrong_model":
        baseline["model_id"] = "forged_lcdm_model"
    _write_json(output_path, output)

    row = _cmb(_report(run, planck), "primary")

    assert row["comparison_receipt"] is False
    assert expected_error in row["integrity_errors"]


def test_cmb_ignores_unbound_fake_baseline_instead_of_taking_minimum(
    tmp_path: Path,
) -> None:
    planck = tmp_path / "planck.txt"
    _planck_fixture(planck)
    run = _fake_run(tmp_path / "primary", planck_path=planck, chi2_per_bin=0.25)
    output_path = run / "physical_cmb_output_comparison_report.json"
    output = json.loads(output_path.read_text(encoding="utf-8"))
    output["rows"].append(
        {
            "source_report": "unbound_forged_report.json",
            "model_id": "forged_baseline",
            "model_role": "external_baseline",
            "amplitude_fit_chi2_per_bin": -100.0,
            "bin_count": len(_planck_values(planck)),
        }
    )
    _write_json(output_path, output)

    row = _cmb(_report(run, planck), "primary")

    assert row["comparison_receipt"] is True
    assert row["baseline"][
        "diagonal_chi2_per_bin_after_one_amplitude_fit"
    ] == pytest.approx(0.1)


def test_cmb_recomputes_and_rejects_forged_bound_baseline_curve_and_model(
    tmp_path: Path,
) -> None:
    planck = tmp_path / "planck.txt"
    _planck_fixture(planck)
    run = _fake_run(tmp_path / "primary", planck_path=planck, chi2_per_bin=0.25)
    source_path = run / "camb_lcdm_baseline_report.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["comparison"]["binned_tt_comparison"][0][
        "amplitude_fit_camb_D_ell"
    ] += 1.0
    source["camb"]["lambda_cdm_parameters"]["H0"] = 70.0
    _write_json(source_path, source)

    row = _cmb(_report(run, planck), "primary")

    assert row["comparison_receipt"] is False
    assert "bound_external_lcdm_baseline_scaled_curve_mismatch" in row[
        "integrity_errors"
    ]
    assert "bound_external_lcdm_baseline_model_parameters_mismatch" in row[
        "integrity_errors"
    ]


def test_schema_rejects_forged_or_hashless_positive_source_binding(
    tmp_path: Path,
) -> None:
    planck = tmp_path / "planck.txt"
    _planck_fixture(planck)
    run = _fake_run(tmp_path / "primary", planck_path=planck, chi2_per_bin=0.25)
    report = _report(run, planck)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    renamed = copy.deepcopy(report)
    files = renamed["public_measurement_source_binding"]["files"]
    renamed["public_measurement_source_binding"]["files"] = {
        f"fake_role_{index}": value
        for index, value in enumerate(files.values())
    }
    assert list(validator.iter_errors(renamed))

    incoherent = copy.deepcopy(report)
    binding = incoherent["public_measurement_source_binding"]
    binding["status"] = "PUBLIC_MEASUREMENT_BYTES_BOUND"
    binding["canonical_source_binding_receipt"] = False
    binding["integrity_receipt"] = False
    assert list(validator.iter_errors(incoherent))

    hashless = copy.deepcopy(report)
    binding = hashless["public_measurement_source_binding"]
    binding["manifest_sha256"] = None
    for row in binding["files"].values():
        row["expected_sha256"] = None
        row["actual_sha256"] = None
        row["expected_bytes"] = None
        row["actual_bytes"] = None
        row["data_kind"] = None
        row["representation"] = None
        row["public_source"] = None
        row["integrity_receipt"] = True
        row["integrity_errors"] = []
    assert list(validator.iter_errors(hashless))


def test_well_formed_planck_mutation_blocks_comparison_receipt(tmp_path: Path) -> None:
    planck = tmp_path / "planck.txt"
    _planck_fixture(planck)
    text = planck.read_text(encoding="utf-8")
    planck.write_text(text.replace("1.47933552e+03", "1.47933553e+03", 1), encoding="utf-8")
    run = _fake_run(tmp_path / "mutated", planck_path=planck, chi2_per_bin=0.25)

    report = _report(run, planck)
    row = _cmb(report, "mutated")

    assert report["public_measurement_source_binding"][
        "canonical_source_binding_receipt"
    ] is False
    assert report["datasets"]["planck_tt"]["integrity_receipt"] is False
    assert row["comparison_receipt"] is False
    assert row["integrity_receipt"] is False
    assert any(
        "planck_tt_binned:selected_file_sha256_mismatch" in error
        for error in row["integrity_errors"]
    )


def test_planck_binding_failure_blocks_even_a_positive_prediction_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planck = tmp_path / "planck.txt"
    _planck_fixture(planck)
    text = planck.read_text(encoding="utf-8")
    planck.write_text(
        text.replace("1.47933552e+03", "1.47933553e+03", 1),
        encoding="utf-8",
    )
    run = _fake_run(tmp_path / "mutated", planck_path=planck, chi2_per_bin=0.25)
    monkeypatch.setattr(
        "oph_fpe.cosmology.public_data_comparisons.cmb_promotion_ledger_report",
        lambda _run_dirs: {
            "likelihood_evaluated_physical_cmb_prediction": True,
            "blockers": [],
            "current_claim_tier": "mock_positive_ledger",
            "first_blocked_gate": None,
        },
    )

    report = _report(run, planck)
    row = _cmb(report, "mutated")

    assert row["integrity_receipt"] is False
    assert row["comparison_receipt"] is False
    assert row["physical_prediction_receipt"] is False
    assert (
        "public_measurement_integrity_or_comparison_receipt_failed"
        in row["prediction_blockers"]
    )
    assert report["summary"]["frozen_prediction_count"] == 0
    assert report["summary"]["physical_prediction_available"] is False

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    row["physical_prediction_receipt"] = True
    assert list(Draft202012Validator(schema).iter_errors(report))


def test_valid_same_data_cmb_cannot_be_promoted_by_unrelated_positive_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planck = tmp_path / "planck.txt"
    _planck_fixture(planck)
    run = _fake_run(tmp_path / "valid", planck_path=planck, chi2_per_bin=0.25)
    monkeypatch.setattr(
        "oph_fpe.cosmology.public_data_comparisons.cmb_promotion_ledger_report",
        lambda _run_dirs: {
            "likelihood_evaluated_physical_cmb_prediction": True,
            "blockers": [],
            "current_claim_tier": "mock_unrelated_positive_ledger",
            "first_blocked_gate": None,
        },
    )

    report = _report(run, planck)
    row = _cmb(report, "valid")

    assert row["comparison_receipt"] is True
    assert row["physical_prediction_receipt"] is False
    assert row["promotion_ledger"]["ledger_asserted_prediction_receipt"] is True
    assert row["promotion_ledger"]["same_data_row_promotion_allowed"] is False
    assert "promotion_ledger_cannot_promote_same_data_diagnostic" in row[
        "prediction_blockers"
    ]
    assert report["summary"]["frozen_prediction_count"] == 0

    forged = copy.deepcopy(report)
    forged_row = _cmb(forged, "valid")
    forged_row["physical_prediction_receipt"] = True
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert list(Draft202012Validator(schema).iter_errors(forged))

    forged_falsifier = copy.deepcopy(report)
    _cmb(forged_falsifier, "valid")[
        "conditional_external_domain_falsifier"
    ] = True
    assert list(Draft202012Validator(schema).iter_errors(forged_falsifier))


def test_run_bundle_integrity_failure_blocks_cmb_and_positive_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planck = tmp_path / "planck.txt"
    _planck_fixture(planck)
    run = _fake_run(tmp_path / "missing_config", planck_path=planck, chi2_per_bin=0.25)
    (run / "config.yml").unlink()
    monkeypatch.setattr(
        "oph_fpe.cosmology.public_data_comparisons.cmb_promotion_ledger_report",
        lambda _run_dirs: {
            "likelihood_evaluated_physical_cmb_prediction": True,
            "blockers": [],
            "current_claim_tier": "mock_positive_ledger",
            "first_blocked_gate": None,
        },
    )

    report = _report(run, planck)
    row = _cmb(report, "missing_config")

    assert report["datasets"]["planck_tt"]["integrity_receipt"] is True
    assert report["run_bundles"][0]["integrity_receipt"] is False
    assert row["integrity_receipt"] is False
    assert row["comparison_receipt"] is False
    assert row["physical_prediction_receipt"] is False
    assert any("missing_yaml:" in error for error in row["integrity_errors"])
    assert (
        "public_measurement_integrity_or_comparison_receipt_failed"
        in row["prediction_blockers"]
    )
    assert report["summary"]["frozen_prediction_count"] == 0


def test_primary_is_not_replaced_by_better_history_run(tmp_path: Path) -> None:
    planck = tmp_path / "planck.txt"
    _planck_fixture(planck)
    primary = _fake_run(tmp_path / "primary", planck_path=planck, chi2_per_bin=0.4)
    history = _fake_run(tmp_path / "history", planck_path=planck, chi2_per_bin=0.05)

    report = _report(primary, planck, history=[history], baseline=history)

    assert report["featured_by_evidence_class"][
        "primary_run_cmb_diagnostic"
    ].startswith("planck_tt:primary:")
    assert report["planck_tt_diagnostic_order"][0]["run_id"] == "history"
    assert report["primary_vs_baseline"]["diagnostic_verdict"] == "diagnostic_regressed"
    assert (
        report["selection_policy"]["history_is_context_not_primary_selection"] is True
    )


def test_self_asserted_prediction_is_rejected_by_same_run_ledger(
    tmp_path: Path,
) -> None:
    planck = tmp_path / "planck.txt"
    _planck_fixture(planck)
    run = _fake_run(
        tmp_path / "asserted",
        planck_path=planck,
        chi2_per_bin=0.2,
        assert_prediction=True,
    )

    row = _cmb(_report(run, planck), "asserted")

    assert row["physical_prediction_receipt"] is False
    assert row["promotion_ledger"]["terminal_prediction_assertion_rejected"] is True
    assert (
        "untrusted_terminal_prediction_assertion_rejected" in row["prediction_blockers"]
    )


def test_planned_scale_keeps_carriers_and_materialized_observers_distinct(
    tmp_path: Path,
) -> None:
    planck = tmp_path / "planck.txt"
    _planck_fixture(planck)
    run = _fake_run(tmp_path / "primary", planck_path=planck, chi2_per_bin=0.2)
    config = tmp_path / "million.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "graph": {"patch_count": 1_048_576},
                "observers": {"sample_count": 64_000},
                "cosmology": {"angular_power": {"ell_max": 8}},
                "million_patch_preparation": {
                    "carrier_patch_count": 1_048_576,
                    "materialized_observer_count": 64_000,
                },
            }
        ),
        encoding="utf-8",
    )

    scale = _report(run, planck, planned_config=config)["planned_run_scale_contract"]

    assert scale["scale_label"] == "million_patch_bounded_observer_sample"
    assert scale["is_at_least_one_million_carrier_patches"] is True
    assert scale["is_at_least_one_million_materialized_observers"] is False
    assert scale["raw_screen_real_ell_overlap_with_planck_bins"] is False


def test_sparc_evidence_classes_and_identifiability_are_separate(
    tmp_path: Path,
) -> None:
    planck = tmp_path / "planck.txt"
    _planck_fixture(planck)
    run = _fake_run(tmp_path / "primary", planck_path=planck, chi2_per_bin=0.2)

    report = _report(run, planck)
    by_id = {row["comparison_id"]: row for row in report["comparisons"]}

    rar = by_id["sparc:rar_calibration"]
    btfr = by_id["sparc:btfr_separate_table_diagnostic"]
    holdout = by_id["sparc:galaxy_level_massmodel_holdout"]
    assert rar["evaluation_class"] == "calibrated_same_data"
    assert rar["parameter_identifiability"]["identifiable_parameter_count"] == 1
    assert rar["metrics"]["rar_galaxy_label_count"]["value"] == 1
    assert btfr["evaluation_class"] == "diagnostic_proxy"
    assert btfr["metrics"]["observed_minus_predicted_slope"]["value"] == pytest.approx(
        -0.1543456734
    )
    assert btfr["metrics"][
        "predicted_minus_observed_slope_wald_hessian_stat_only_pull"
    ][
        "value"
    ] == pytest.approx(1.79854098, rel=1.0e-6)
    assert btfr["metrics"][
        "slope_four_signed_sqrt_profile_deviance_vs_free_fit"
    ]["value"] == pytest.approx(1.72693919, rel=1.0e-6)
    assert btfr["metrics"]["z6_observed_minus_predicted_pivot"][
        "value"
    ] == pytest.approx(-0.13476264, rel=1.0e-6)
    assert btfr["data_use"]["fit_to_evaluation_data"] is True
    assert btfr["data_use"]["oph_prediction_parameters_fit_to_btfr"] is False
    assert btfr["data_use"]["fitted_nuisance_parameters"] == [
        "orthogonal_intrinsic_scatter_perpendicular_dex"
    ]
    assert btfr["data_use"]["holdout"] is False
    assert "intrinsic-scatter nuisance is fitted" in btfr["claim_boundary"]
    assert btfr["metrics"][
        "z6_normalization_wald_hessian_stat_only_pull"
    ]["value"] == pytest.approx(-6.47073535, rel=1.0e-6)
    assert btfr["metrics"][
        "z6_normalization_signed_sqrt_profile_deviance"
    ]["value"] == pytest.approx(-math.sqrt(34.742808147775065))
    assert btfr["comparison_id"] not in report["featured_by_evidence_class"][
        "independent_dataset_checks"
    ]
    assert holdout["evaluation_class"] == "heldout_test"
    assert holdout["metrics"]["test_velocity_rmse"]["value"] == pytest.approx(
        22.6876200616
    )
    assert (
        holdout["metrics"]["test_velocity_diagonal_chi2_proxy_per_point"]["value"] > 30
    )


def test_sparc_byte_binding_failure_suppresses_all_comparison_receipts(
    tmp_path: Path,
) -> None:
    planck = tmp_path / "planck.txt"
    _planck_fixture(planck)
    run = _fake_run(tmp_path / "primary", planck_path=planck, chi2_per_bin=0.2)
    sparc = tmp_path / "sparc"
    shutil.copytree(SPARC_DIR, sparc)
    rar = sparc / "RAR.mrt"
    rar.write_bytes(rar.read_bytes().replace(b"-11.23", b"-11.24", 1))

    report = public_data_comparison_report(
        run,
        planck_tt_path=planck,
        sparc_dir=sparc,
    )
    rows = [
        row
        for row in report["comparisons"]
        if row.get("dataset_id") == "sparc_rar_btfr_massmodels"
    ]
    featured = json.dumps(report["featured_by_evidence_class"], sort_keys=True)

    assert report["datasets"]["sparc"]["integrity_receipt"] is False
    assert rows
    assert all(row["integrity_receipt"] is False for row in rows)
    assert all(row["comparison_receipt"] is False for row in rows)
    assert all(row["comparison_id"] not in featured for row in rows)


def test_cassini_integral_reproduces_benchmarks_and_oph_endpoints() -> None:
    report = cassini_external_field_report()
    validation = report["validation"]
    branches = report["oph_branches"]
    fixed = report["fixed_input_diagnostic"]

    assert validation["receipt"] is True
    assert validation["disk_receipt"] is True
    assert validation["park_rar_spherical"]["Q2_s2"] == pytest.approx(
        3.3872263348329e-26, rel=1.0e-9
    )
    assert validation["park_rar_disk"]["Q2_s2"] == pytest.approx(
        3.4116772499560e-26, rel=1.0e-9
    )
    assert branches["z6_exact_uniform_target"]["Q2_s2"] == pytest.approx(
        3.62017781533e-26, rel=1.0e-9
    )
    assert branches["unit_lambda_endpoint"]["Q2_s2"] == pytest.approx(
        3.40218755877e-26, rel=1.0e-9
    )
    assert fixed["z6_raw_pull_sigma"] == pytest.approx(19.2232100852)
    assert fixed["z6_gaia_only_combined_pull_sigma"] == pytest.approx(10.7072254620)
    assert branches["jensen_lambda_band"]["raw_pull_min_sigma"] > 18.0
    assert report["physical_prediction_receipt"] is False


def test_cassini_is_run_independent_stress_diagnostic_without_frozen_threshold(
    tmp_path: Path,
) -> None:
    planck = tmp_path / "planck.txt"
    _planck_fixture(planck)
    small = _fake_run(
        tmp_path / "small", planck_path=planck, chi2_per_bin=0.2, patch_count=65_536
    )
    large = _fake_run(
        tmp_path / "large", planck_path=planck, chi2_per_bin=0.3, patch_count=1_048_576
    )

    reports = [_report(small, planck), _report(large, planck)]
    rows = [
        next(
            row
            for row in report["comparisons"]
            if row["comparison_id"] == "cassini:conditional_static_external_field"
        )
        for report in reports
    ]

    assert rows[0]["calculation"] == rows[1]["calculation"]
    assert rows[0]["run_dependent"] is False
    assert rows[0]["comparison_receipt"] is True
    assert rows[0]["conditional_external_domain_falsifier"] is False
    assert rows[0]["falsifier_decision"]["decision_rule_receipt"] is False
    assert rows[0]["falsifier_decision"][
        "candidate_minimum_abs_pull_sigma"
    ] is None
    assert "no_candidate_tension_threshold" in rows[0]["falsifier_decision"][
        "blockers"
    ]
    assert rows[0]["falsifier_decision"][
        "independent_preregistration_custody_receipt"
    ] is False
    assert rows[0]["falsifier_decision"][
        "global_band_minimization_receipt"
    ] is False
    assert rows[0]["diagnostic_reference"][
        "exceeds_conventional_reference"
    ] is True
    assert rows[0]["diagnostic_reference"][
        "is_acceptance_or_falsifier_threshold"
    ] is False
    assert (
        rows[0]["metrics"][
            "jensen_endpoints_minimum_absolute_raw_fixed_input_pull"
        ]["value"]
        > 18.0
    )
    assert "jensen_band_minimum_absolute_raw_fixed_input_pull" not in rows[0][
        "metrics"
    ]
    assert rows[0]["applicability_receipt"] is False
    assert rows[0]["current_scope_match"] is False
    assert rows[0]["raw_pull_not_nuisance_marginalized"] is True
    assert rows[0]["physical_prediction_receipt"] is False
    assert reports[0]["featured_by_evidence_class"][
        "conditional_external_domain_falsifiers"
    ] == []


def test_cassini_self_attested_candidate_threshold_can_never_promote(
    tmp_path: Path,
) -> None:
    source = json.loads(CANONICAL_CASSINI_PATH.read_text(encoding="utf-8"))
    rule = {
        "statistic": (
            "jensen_lambda_endpoints_minimum_absolute_raw_fixed_input_pull_sigma"
        ),
        "minimum_abs_pull_sigma": 20.0,
        "declared_before_comparison": True,
        "provenance": "synthetic_prefrozen_protocol_for_mutation_test",
    }
    source["conditional_external_domain_falsifier_rule"] = rule
    path = tmp_path / "cassini.json"
    _write_json(path, source)

    _, above_threshold_row, errors = _cassini_dataset_and_comparison(path)

    assert errors == []
    assert above_threshold_row["comparison_receipt"] is True
    assert above_threshold_row["falsifier_decision"][
        "candidate_rule_syntactically_valid"
    ] is True
    assert above_threshold_row["falsifier_decision"][
        "candidate_threshold_exceeded"
    ] is False
    assert above_threshold_row["falsifier_decision"]["decision_rule_receipt"] is False
    assert above_threshold_row["falsifier_decision"]["threshold_passed"] is False
    assert above_threshold_row["conditional_external_domain_falsifier"] is False

    source["conditional_external_domain_falsifier_rule"][
        "minimum_abs_pull_sigma"
    ] = 10.0
    _write_json(path, source)
    _, below_threshold_row, errors = _cassini_dataset_and_comparison(path)

    assert errors == []
    assert below_threshold_row["falsifier_decision"][
        "candidate_threshold_exceeded"
    ] is True
    assert below_threshold_row["falsifier_decision"][
        "independent_preregistration_custody_receipt"
    ] is False
    assert below_threshold_row["falsifier_decision"]["threshold_passed"] is False
    assert below_threshold_row["falsifier_decision"]["decision_receipt"] is False
    assert below_threshold_row["conditional_external_domain_falsifier"] is False


def test_schema_enforces_cassini_scope_boundary(tmp_path: Path) -> None:
    planck = tmp_path / "planck.txt"
    _planck_fixture(planck)
    run = _fake_run(tmp_path / "primary", planck_path=planck, chi2_per_bin=0.2)
    report = _report(run, planck)
    row = next(
        row
        for row in report["comparisons"]
        if row["comparison_id"] == "cassini:conditional_static_external_field"
    )
    row["current_scope_match"] = True
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    errors = list(Draft202012Validator(schema).iter_errors(report))

    assert any(list(error.path)[-1:] == ["current_scope_match"] for error in errors)

    row["current_scope_match"] = False
    row["conditional_external_domain_falsifier"] = True
    errors = list(Draft202012Validator(schema).iter_errors(report))
    assert any(
        list(error.path)[-1:] == ["conditional_external_domain_falsifier"]
        for error in errors
    )


def test_schema_rejects_string_boolean(tmp_path: Path) -> None:
    planck = tmp_path / "planck.txt"
    _planck_fixture(planck)
    run = _fake_run(tmp_path / "primary", planck_path=planck, chi2_per_bin=0.2)
    report = _report(run, planck)
    report["comparisons"][0]["comparison_receipt"] = "true"
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    errors = list(Draft202012Validator(schema).iter_errors(report))

    assert any(list(error.path)[-1:] == ["comparison_receipt"] for error in errors)
