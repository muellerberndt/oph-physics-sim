from __future__ import annotations

import csv
import json
from pathlib import Path

from oph_fpe.cosmology.sync_inflation import synchronization_inflation_report, write_synchronization_inflation_report


def test_sync_inflation_report_keeps_horizon_gate_closed_without_selector(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps({"run_id": "sync_synthetic", "patch_count": 4096, "cycles": 8}),
        encoding="utf-8",
    )
    with (run / "mismatch_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cycle", "phi"])
        writer.writeheader()
        for cycle, phi in enumerate([100.0, 50.0, 25.0, 12.5, 6.25, 3.125, 1.56, 0.78]):
            writer.writerow({"cycle": cycle, "phi": phi})
    (run / "collar_markov_report.json").write_text(json.dumps({"median_epsilon_cmi": 0.01}), encoding="utf-8")
    (run / "array_holonomy_report.json").write_text(json.dumps({"defect_fraction": 0.2}), encoding="utf-8")

    report = synchronization_inflation_report([tmp_path])

    assert report["run_count"] == 1
    assert report["rows"][0]["Gamma_sync_over_H_proxy"] > 1.0
    assert report["rows"][0]["horizon_synchronization_ready"] is False
    assert "same_boundary_selector_or_low_k_gap" in report["rows"][0]["missing_gates"]
    assert report["inflation_replacement_ready"] is False


def test_write_sync_inflation_report(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"run_id": "sync_synthetic", "cycles": 3}), encoding="utf-8")
    (run / "mismatch_trace.csv").write_text("cycle,phi\n0,9\n1,3\n2,1\n", encoding="utf-8")

    report = write_synchronization_inflation_report([tmp_path], tmp_path / "out")

    assert report["physical_cmb_prediction"] is False
    assert (tmp_path / "out" / "sync_inflation_report.json").exists()
    assert (tmp_path / "out" / "sync_inflation_rows.csv").exists()
