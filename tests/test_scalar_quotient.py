from __future__ import annotations

import json
import math
from pathlib import Path

from oph_fpe.constants.oph_pixel import P_STAR
from oph_fpe.cosmology.comparable_data import comparable_data_report
from oph_fpe.cosmology.physical_cmb_prediction import write_physical_cmb_input_report
from oph_fpe.cosmology.scalar_quotient import scalar_quotient_report, write_scalar_quotient_report


def test_scalar_quotient_report_emits_center_free_scalar_without_physical_claim(tmp_path: Path):
    run = _write_run(tmp_path, observer_count=36, patch_count=4096)

    report = scalar_quotient_report(run)

    assert report["mode"] == "oph_scalar_geometric_quotient_report_v0"
    assert report["SCALAR_QUOTIENT_RECEIPT"] is False
    assert report["SCREEN_SCALAR_QUOTIENT_RECEIPT"] is False
    assert report["DIAGNOSTIC_SCALAR_FEATURE_PROXY_RECEIPT"] is True
    assert report["physical_cmb_prediction"] is False
    assert report["scalar_packet_alphabet_size"] > 1
    assert report["scalar_field_statistics"]["monopole_dipole_removed"] is True
    assert report["readiness_gates"]["primary_no_forbidden_geometry"] is True
    assert report["readiness_gates"]["geometric_volume_jacobian_readout"] is False
    assert report["readiness_gates"]["theorem_grade_scalar_release_code"] is False
    assert "screen_scalar_quotient_receipt_missing" in report["blockers"]
    assert math.isclose(report["edge_center_readout"]["n_s_P_over_48"], 1.0 - P_STAR / 48.0)


def test_scalar_quotient_report_blocks_ell32_when_active_support_is_too_small(tmp_path: Path):
    run = _write_run(tmp_path, observer_count=36, patch_count=4096)

    report = scalar_quotient_report(run, target_ell_ir=32)

    assert report["active_angular_levels"]["observer_level_proxy_floor_sqrt_observers"] == 6
    assert report["readiness_gates"]["active_33_level_freezeout_clause"] is False
    assert report["finite_lattice_cmb_scalar_release_ready"] is False
    assert "active_33_level_freezeout_clause_not_established" in report["blockers"]


def test_scalar_quotient_writer_and_comparable_lane(tmp_path: Path):
    run = _write_run(tmp_path, observer_count=36, patch_count=4096)
    write_scalar_quotient_report(run)

    report = comparable_data_report([run])
    lane = report["measurement_lanes"]["oph_scalar_geometric_quotient"]

    assert (run / "scalar_quotient_report.json").exists()
    assert (run / "scalar_quotient_report.md").exists()
    assert (run / "scalar_quotient_packets.csv").exists()
    assert lane["run_count"] == 1
    assert lane["scalar_quotient_receipt_count"] == 0
    assert lane["finite_ready_count"] == 0
    assert lane["theorem_grade_release_count"] == 0


def test_scalar_quotient_can_feed_contract_only_as_diagnostic_when_not_finite_ready(tmp_path: Path):
    run = _write_run(tmp_path, observer_count=36, patch_count=4096)
    write_scalar_quotient_report(run)
    (run / "no_data_use_receipt.json").write_text(json.dumps({"no_data_use_receipt": True}), encoding="utf-8")

    report = write_physical_cmb_input_report([run], tmp_path / "out")

    assert report["physical_cmb_prediction_eligible"] is False
    assert report["contract"]["eta_R_source"] == "diagnostic_proxy"
    assert report["contract"]["q_IR_source"] == "diagnostic_proxy"
    assert "eta_R_not_finite_derived" in report["blockers"]
    assert "q_IR_not_finite_derived" in report["blockers"]


def _write_run(tmp_path: Path, *, observer_count: int, patch_count: int) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps({"run_id": "scalar_test", "patch_count": patch_count}),
        encoding="utf-8",
    )
    rows = []
    for i in range(observer_count):
        theta = 2.0 * math.pi * i / observer_count
        z = -1.0 + 2.0 * (i + 0.5) / observer_count
        r = math.sqrt(max(0.0, 1.0 - z * z))
        rows.append(
            {
                "view_type": "patch_observer",
                "observer_id": i,
                "axis": [r * math.cos(theta), r * math.sin(theta), z],
                "committed_fraction": (i % 5) / 4.0,
                "record_stability_mean": ((i * 2) % 7) / 6.0,
                "repair_load_mean": ((i * 3) % 11) / 10.0,
                "mismatch_density_mean": ((i * 5) % 13) / 12.0,
                "visible_signature_entropy": ((i * 7) % 17) / 16.0,
                "counterfactual_stability": ((i * 11) % 19) / 18.0,
                "support_nodes": [i, i + 1],
                "h3_point": [1.0, 0.0, 0.0, 0.0],
                "modular_depth": 999.0,
            }
        )
    with (run / "observer_views.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return run
