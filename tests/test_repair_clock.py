from __future__ import annotations

import csv
import math
from pathlib import Path

from oph_fpe.constants.oph_pixel import OPHPixelConstants
from oph_fpe.cosmology.edge_center_clock import edge_center_clock_target
from oph_fpe.cosmology.repair_clock import repair_clock_report, write_repair_clock_report


def test_repair_clock_target_match_stays_fail_closed_without_edge_center_evidence(tmp_path: Path) -> None:
    pixel = OPHPixelConstants()
    target = edge_center_clock_target(pixel.P)
    delta = pixel.P - pixel.phi
    dt = 0.25
    roots = []
    for idx in range(3):
        run = tmp_path / f"run{idx}"
        run.mkdir()
        _write_exponential_mismatch_trace(run / "mismatch_trace.csv", delta=delta, dt=dt, amplitude=10.0 + idx)
        roots.append(run)

    report = repair_clock_report(roots, cycle_time_normalization=dt, relative_tolerance=1.0e-8)

    assert report["summary"]["passed_estimator_count"] == 3
    assert report["finite_repair_clock_certificate"] is False
    assert abs(report["summary"]["median_kappa_rep_estimate"] - target.kappa_rep) < 1.0e-10
    assert abs(report["summary"]["median_eta_R_estimate"] - target.theta) < 1.0e-12
    assert report["EDGE_CENTER_CLOCK_RECEIPT"] is False
    assert all(value is False for value in report["edge_center_clock_evidence"]["receipts"].values())


def test_repair_clock_keeps_un_normalized_trace_diagnostic_only(tmp_path: Path) -> None:
    pixel = OPHPixelConstants()
    delta = pixel.P - pixel.phi
    run = tmp_path / "run"
    run.mkdir()
    _write_exponential_mismatch_trace(run / "mismatch_trace.csv", delta=delta, dt=0.25, amplitude=10.0)

    report = repair_clock_report([run])

    assert report["summary"]["estimator_count"] == 1
    assert report["summary"]["eligible_estimator_count"] == 0
    assert report["finite_repair_clock_certificate"] is False
    assert any("no predeclared finite repair-time normalization" in item for item in report["blockers"])


def test_write_repair_clock_report_writes_cached_artifacts(tmp_path: Path) -> None:
    pixel = OPHPixelConstants()
    delta = pixel.P - pixel.phi
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_exponential_mismatch_trace(run / "mismatch_trace.csv", delta=delta, dt=1.0, amplitude=10.0)

    report = write_repair_clock_report([run], out, cycle_time_normalization=1.0)

    assert report["summary"]["estimator_count"] == 1
    assert (out / "repair_clock_certificate_report.json").exists()
    assert (out / "repair_clock_estimators.csv").exists()
    assert (out / "repair_clock_certificate_report.md").exists()


def _write_exponential_mismatch_trace(path: Path, *, delta: float, dt: float, amplitude: float) -> None:
    target = edge_center_clock_target()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cycle", "phi"])
        writer.writeheader()
        for cycle in range(24):
            phi = amplitude * math.exp(-target.kappa_rep * delta * float(cycle) * float(dt))
            writer.writerow({"cycle": cycle, "phi": phi})
