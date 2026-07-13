from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from oph_fpe.cosmology.cassini_external_field import cassini_external_field_report
from oph_fpe.cosmology.public_data_comparisons import public_data_comparison_report


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schemas/cosmology/best_of_public_data_comparisons.schema.json"
SPARC_DIR = REPO_ROOT / "data/measurements/sparc"


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _planck_fixture(path: Path) -> list[dict[str, float]]:
    path.write_text(
        "# l Dl -dDl +dDl BestFit\n"
        "10 100 10 10 99\n"
        "20 80 8 8 81\n",
        encoding="utf-8",
    )
    return [
        {"ell": 10.0, "observed_D_ell": 100.0, "sigma_D_ell": 10.0},
        {"ell": 20.0, "observed_D_ell": 80.0, "sigma_D_ell": 8.0},
    ]


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
    offset = math.sqrt(chi2_per_bin)
    embedded = []
    for row in observed_rows:
        observed = row["observed_D_ell"]
        sigma = row["sigma_D_ell"]
        embedded.append(
            {
                "ell": row["ell"],
                "observed_D_ell": observed,
                "sigma_D_ell": sigma,
                "camb_D_ell": observed + 2.0 * offset * sigma,
                "amplitude_fit_camb_D_ell": observed + offset * sigma,
                "best_fit_column_D_ell": observed,
            }
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
        "best_oph_residual_summary": {"source_csv": "finite_repair_clock_cmb_tt_bins.csv"},
        "rows": [
            {
                "source_report": "camb_lcdm_baseline_report.json",
                "model_id": "lcdm_baseline",
                "model_role": "external_baseline",
                "amplitude_fit_chi2_per_bin": 0.1,
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
        {"B_A_KERNEL_CANDIDATE_RECEIPT": False, "B_A_KERNEL_RECEIPT": False, "row_count": 0},
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


def test_suite_recomputes_profiled_residuals_and_validates_schema(tmp_path: Path) -> None:
    planck = tmp_path / "planck.txt"
    _planck_fixture(planck)
    run = _fake_run(tmp_path / "primary", planck_path=planck, chi2_per_bin=0.25)

    report = _report(run, planck)
    row = _cmb(report, "primary")

    assert row["comparison_receipt"] is True
    assert row["physical_prediction_receipt"] is False
    assert row["metrics"]["diagonal_chi2_per_bin_after_one_amplitude_fit"]["value"] == pytest.approx(0.25)
    assert row["metrics"]["profiled_rms_residual_sigma"]["value"] == pytest.approx(0.5)
    assert row["metrics"]["raw_rms_residual_sigma"]["value"] == pytest.approx(1.0)
    assert row["source_bundle"]["same_parent_directory"] is True

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(report)


def test_primary_is_not_replaced_by_better_history_run(tmp_path: Path) -> None:
    planck = tmp_path / "planck.txt"
    _planck_fixture(planck)
    primary = _fake_run(tmp_path / "primary", planck_path=planck, chi2_per_bin=0.4)
    history = _fake_run(tmp_path / "history", planck_path=planck, chi2_per_bin=0.05)

    report = _report(primary, planck, history=[history], baseline=history)

    assert report["featured_by_evidence_class"]["primary_run_cmb_diagnostic"].startswith(
        "planck_tt:primary:"
    )
    assert report["planck_tt_diagnostic_order"][0]["run_id"] == "history"
    assert report["primary_vs_baseline"]["diagnostic_verdict"] == "diagnostic_regressed"
    assert report["selection_policy"]["history_is_context_not_primary_selection"] is True


def test_self_asserted_prediction_is_rejected_by_same_run_ledger(tmp_path: Path) -> None:
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
    assert "untrusted_terminal_prediction_assertion_rejected" in row["prediction_blockers"]


def test_planned_scale_keeps_carriers_and_materialized_observers_distinct(tmp_path: Path) -> None:
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


def test_sparc_evidence_classes_and_identifiability_are_separate(tmp_path: Path) -> None:
    planck = tmp_path / "planck.txt"
    _planck_fixture(planck)
    run = _fake_run(tmp_path / "primary", planck_path=planck, chi2_per_bin=0.2)

    report = _report(run, planck)
    by_id = {row["comparison_id"]: row for row in report["comparisons"]}

    rar = by_id["sparc:rar_calibration"]
    btfr = by_id["sparc:btfr_independent_table_check"]
    holdout = by_id["sparc:galaxy_level_massmodel_holdout"]
    assert rar["evaluation_class"] == "calibrated_same_data"
    assert rar["parameter_identifiability"]["identifiable_parameter_count"] == 1
    assert rar["metrics"]["rar_galaxy_label_count"]["value"] == 1
    assert btfr["evaluation_class"] == "calibrated_independent_dataset"
    assert btfr["metrics"]["observed_minus_predicted_slope"]["value"] == pytest.approx(
        -0.1543456734
    )
    assert btfr["metrics"]["predicted_minus_observed_slope_pull"]["value"] == pytest.approx(
        1.79854098, rel=1.0e-6
    )
    assert btfr["metrics"]["z6_observed_minus_predicted_pivot"]["value"] == pytest.approx(
        -0.13476264, rel=1.0e-6
    )
    assert holdout["evaluation_class"] == "heldout_test"
    assert holdout["metrics"]["test_velocity_rmse"]["value"] == pytest.approx(22.6876200616)
    assert holdout["metrics"]["test_velocity_diagonal_chi2_proxy_per_point"]["value"] > 30


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


def test_cassini_is_run_independent_conditional_falsifier(tmp_path: Path) -> None:
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
    assert rows[0]["conditional_external_domain_falsifier"] is True
    assert rows[0]["applicability_receipt"] is False
    assert rows[0]["current_scope_match"] is False
    assert rows[0]["raw_pull_not_nuisance_marginalized"] is True
    assert rows[0]["physical_prediction_receipt"] is False
    assert (
        "cassini:conditional_static_external_field"
        in reports[0]["featured_by_evidence_class"]["conditional_external_domain_falsifiers"]
    )


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


def test_schema_rejects_string_boolean(tmp_path: Path) -> None:
    planck = tmp_path / "planck.txt"
    _planck_fixture(planck)
    run = _fake_run(tmp_path / "primary", planck_path=planck, chi2_per_bin=0.2)
    report = _report(run, planck)
    report["comparisons"][0]["comparison_receipt"] = "true"
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    errors = list(Draft202012Validator(schema).iter_errors(report))

    assert any(list(error.path)[-1:] == ["comparison_receipt"] for error in errors)
