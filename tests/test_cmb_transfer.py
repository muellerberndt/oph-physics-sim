from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.cosmology.cmb_transfer import cmb_transfer_report, write_cmb_transfer_report


def test_cmb_transfer_fits_train_and_tests_other_scale():
    rows = [
        _row("train_a", 65536, [1.0, 3.0, 2.0], [2.0, 1.0, 0.5]),
        _row("train_b", 65536, [1.1, 2.9, 2.1], [1.8, 1.1, 0.7]),
        _row("test_a", 262144, [1.0, 3.1, 2.0], [2.1, 1.0, 0.6]),
    ]
    benchmark = [
        {"ell": 50.0, "D_ell": 1000.0},
        {"ell": 100.0, "D_ell": 3000.0},
        {"ell": 150.0, "D_ell": 2000.0},
    ]

    report = cmb_transfer_report(
        rows,
        benchmark,
        train_patch_count=65536,
        test_patch_count=262144,
        field_names=["record_signature", "stable_count"],
        ridge=1.0e-6,
        sample_count=16,
        bootstrap_count=32,
        bootstrap_seed=42,
    )

    assert report["physical_cmb_prediction"] is False
    assert report["train_patch_count"] == 65536
    assert report["test_patch_count"] == 262144
    assert report["train"]["run_count"] == 2
    assert report["test"]["run_count"] == 1
    assert report["test"]["mean_shape_correlation"] > 0.9
    assert "record_signature" in report["weights"]
    assert report["diagnostic_transfer_receipt"] is True
    assert set(report["controls"]) == {"reversed_target", "shuffled_target", "shuffled_field_curves"}
    assert report["max_control_test_shape_correlation"] < report["test"]["mean_shape_correlation"]
    assert report["test_vs_max_control_shape_correlation_gap"] > 0.0
    assert report["bootstrap"]["enabled"] is True
    assert report["bootstrap"]["bootstrap_count"] == 32
    assert report["bootstrap"]["test_shape_correlation"]["p05"] is not None


def test_write_cmb_transfer_report_collects_run_bundles(tmp_path: Path):
    run = tmp_path / "runs" / "run_a"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"run_id": "run_a", "patch_count": 65536}), encoding="utf-8")
    (run / "cosmology_gate_report.json").write_text(json.dumps({"allowed": True}), encoding="utf-8")
    (run / "cl_comparison_report.json").write_text(
        json.dumps(
            {
                "fields": {
                    "record_signature": {
                        "spectrum": [
                            {"ell": 2, "D_ell": 1.0},
                            {"ell": 3, "D_ell": 3.0},
                            {"ell": 4, "D_ell": 2.0},
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    benchmark = tmp_path / "planck.txt"
    benchmark.write_text("50 1000\n100 3000\n150 2000\n", encoding="utf-8")

    report = write_cmb_transfer_report(
        [tmp_path / "runs"],
        benchmark,
        tmp_path / "out",
        field_names=["record_signature"],
        train_patch_count=65536,
        test_patch_count=65536,
    )

    assert (tmp_path / "out" / "cmb_transfer_report.json").exists()
    assert (tmp_path / "out" / "cmb_transfer_report.md").exists()
    assert report["test"]["run_count"] == 1
    assert "Controls" in (tmp_path / "out" / "cmb_transfer_report.md").read_text(encoding="utf-8")


def _row(run_id: str, patch_count: int, record: list[float], stable: list[float]):
    return {
        "run_id": run_id,
        "patch_count": patch_count,
        "gate_allowed": True,
        "fields": {
            "record_signature": {
                "spectrum": [
                    {"ell": ell, "D_ell": value}
                    for ell, value in enumerate(record, start=2)
                ]
            },
            "stable_count": {
                "spectrum": [
                    {"ell": ell, "D_ell": value}
                    for ell, value in enumerate(stable, start=2)
                ]
            },
        },
    }
