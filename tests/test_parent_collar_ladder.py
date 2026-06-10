from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.cosmology.comparable_data import comparable_data_report
from oph_fpe.cosmology.parent_collar_ladder import (
    parent_collar_ladder_report,
    write_parent_collar_ladder_report,
)


def test_parent_collar_ladder_passes_on_decreasing_cmi(tmp_path: Path) -> None:
    roots = [
        _write_collar_run(tmp_path, 4096, 0.20),
        _write_collar_run(tmp_path, 65536, 0.03),
        _write_collar_run(tmp_path, 262144, 0.004),
    ]

    report = parent_collar_ladder_report(roots)

    assert report["compiler_ready"] is True
    assert report["regulator_ladder_ready"] is True
    assert report["cmi_scaling_improves"] is True
    assert report["parent_collar_recovery_ladder_receipt"] is True
    assert report["theorem_grade_parent_collar_ladder"] is False
    assert report["physical_cmb_prediction"] is False


def test_parent_collar_ladder_fails_on_worsening_cmi(tmp_path: Path) -> None:
    roots = [
        _write_collar_run(tmp_path, 4096, 0.20),
        _write_collar_run(tmp_path, 65536, 0.80),
        _write_collar_run(tmp_path, 262144, 1.60),
    ]

    report = parent_collar_ladder_report(roots)

    assert report["regulator_ladder_ready"] is True
    assert report["cmi_scaling_improves"] is False
    assert report["parent_collar_recovery_ladder_receipt"] is False
    assert "does not improve" in report["diagnosis"]


def test_parent_collar_ladder_reports_local_density_improvement(tmp_path: Path) -> None:
    roots = [
        _write_collar_run(tmp_path, 4096, 0.40),
        _write_collar_run(tmp_path, 65536, 1.20),
        _write_collar_run(tmp_path, 262144, 1.60),
    ]

    report = parent_collar_ladder_report(roots)

    assert report["cmi_scaling_improves"] is False
    assert report["cmi_density_scaling_improves"] is True
    assert report["local_recovery_density_receipt"] is True
    assert report["parent_collar_recovery_ladder_receipt"] is False
    assert "CMI per collar patch does improve" in report["diagnosis"]


def test_parent_collar_ladder_marks_mixed_cap_family_exploratory(tmp_path: Path) -> None:
    roots = [
        _write_collar_run(tmp_path, 4096, 0.40),
        _write_collar_run(tmp_path, 65536, 1.20),
        _write_collar_run(tmp_path, 262144, 1.60),
        _write_collar_run(tmp_path, 1048576, 1.80, repeat_theta=True),
    ]

    report = parent_collar_ladder_report(roots)

    assert report["cap_family"]["strict_cap_family_matched"] is False
    assert report["cap_family"]["unique_theta_family_matched"] is True
    assert report["local_recovery_density_receipt"] is True
    assert report["strict_local_recovery_density_receipt"] is False


def test_parent_collar_ladder_surfaces_in_comparable_snapshot(tmp_path: Path) -> None:
    roots = [
        _write_collar_run(tmp_path, 4096, 0.20),
        _write_collar_run(tmp_path, 65536, 0.03),
        _write_collar_run(tmp_path, 262144, 0.004),
    ]
    out = tmp_path / "ladder"
    write_parent_collar_ladder_report(roots, out)

    report = comparable_data_report([out])
    lane = report["measurement_lanes"]["oph_parent_collar_recovery_ladder"]

    assert lane["run_count"] == 1
    assert lane["compiler_ready_count"] == 1
    assert lane["regulator_ready_count"] == 1
    assert lane["scaling_pass_count"] == 1
    assert lane["theorem_grade_count"] == 0
    assert lane["physical_cmb_prediction_count"] == 0


def _write_collar_run(
    tmp_path: Path,
    patch_count: int,
    epsilon_cmi: float,
    *,
    repeat_theta: bool = False,
) -> Path:
    run_dir = tmp_path / f"run_{patch_count}"
    run_dir.mkdir()
    theta_values = [0.35, 0.45, 0.55, 0.65]
    if repeat_theta:
        theta_values = [0.35, 0.45, 0.55, 0.65, 0.35, 0.45, 0.55, 0.65]
    rows = [
        {
            "cap_id": cap_id,
            "theta0": theta0,
            "collar_width": 1.0 / patch_count**0.5,
            "inside_count": patch_count // 4,
            "collar_count": patch_count // 16,
            "outside_count": patch_count - patch_count // 4 - patch_count // 16,
            "epsilon_cmi": epsilon_cmi,
            "r_fr_bound": 2.0 * epsilon_cmi**0.5,
            "sample_count": patch_count,
            "packet_alphabet_size": 16,
            "triplet_count": patch_count // 8,
        }
        for cap_id, theta0 in enumerate(theta_values)
    ]
    report = {
        "mode": "diagonal_empirical_collar_state",
        "cap_count": len(rows),
        "median_epsilon_cmi": epsilon_cmi,
        "p90_epsilon_cmi": epsilon_cmi,
        "rows": rows,
    }
    (run_dir / "collar_markov_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return run_dir
