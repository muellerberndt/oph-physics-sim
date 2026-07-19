from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.cosmology.edge_center_clock import edge_center_clock_target
from oph_fpe.cosmology.repair_clock import repair_clock_report
from oph_fpe.cosmology.scalar_repair_semigroup import (
    ScalarRepairSemigroupSpec,
    scalar_repair_semigroup_report,
    write_scalar_repair_semigroup_report,
)


def test_declared_scalar_semigroup_hits_selected_edge_center_target_but_is_not_certificate() -> None:
    report = scalar_repair_semigroup_report()
    target = edge_center_clock_target()

    assert report["SEMIGROUP_TARGET_RECEIPT"] is False
    assert report["finite_lattice_derived"] is False
    assert report["repair_clock_certificate"] is False
    assert report["matrix_source_audit"]["matrix_loaded"] is False
    assert report["semigroup"]["constant_mode_zero"] is True
    assert report["semigroup"]["centered_scalar_relaxation"] is True
    assert report["semigroup"]["kappa_rep_estimate"] == target.kappa_rep
    assert report["target"]["required_eta_R"] == target.theta
    assert report["target"]["required_n_s"] == target.n_s
    assert report["controls"]["e_is_nonpromoting_diagnostic"] is True
    assert report["EDGE_CENTER_CLOCK_RECEIPT"] is False
    assert all(value is False for value in report["edge_center_clock_evidence"]["receipts"].values())


def test_repair_clock_keeps_declared_semigroup_target_diagnostic_only(tmp_path: Path) -> None:
    run = tmp_path / "run"
    write_scalar_repair_semigroup_report(run)

    report = repair_clock_report([run])

    assert report["summary"]["estimator_count"] == 1
    assert report["summary"]["eligible_estimator_count"] == 0
    assert report["finite_repair_clock_certificate"] is False
    assert report["rows"][0]["estimator"] == "scalar_repair_semigroup_gap"
    assert report["rows"][0]["eligible_for_certificate"] is False
    assert "declared P/48 edge-center target" in report["rows"][0]["reason"]


def test_repair_clock_rejects_finite_semigroup_rows_without_matrix_artifacts(tmp_path: Path) -> None:
    roots = []
    for idx in range(3):
        run = tmp_path / f"run{idx}"
        run.mkdir()
        report = scalar_repair_semigroup_report(
            ScalarRepairSemigroupSpec(
                source="finite_state_transition_matrix",
                finite_lattice_derived=True,
                matrix_source=f"transition_matrix_{idx}.npz",
            )
        )
        (run / "scalar_repair_semigroup_report.json").write_text(json.dumps(report), encoding="utf-8")
        roots.append(run)

    aggregate = repair_clock_report(roots, relative_tolerance=1.0e-12)
    target = edge_center_clock_target()

    assert aggregate["summary"]["eligible_estimator_count"] == 0
    assert aggregate["summary"]["passed_estimator_count"] == 0
    assert aggregate["finite_repair_clock_certificate"] is False
    assert abs(aggregate["summary"]["median_kappa_rep_estimate"] - target.kappa_rep) < 1.0e-12


def test_write_scalar_repair_semigroup_report_writes_artifacts(tmp_path: Path) -> None:
    report = write_scalar_repair_semigroup_report(tmp_path, dimension=12)

    assert report["dimension"] == 12
    assert (tmp_path / "scalar_repair_semigroup_report.json").exists()
    assert (tmp_path / "scalar_repair_semigroup_report.md").exists()
