from __future__ import annotations

import json
import math
from pathlib import Path

from oph_fpe.constants.oph_pixel import P_STAR
from oph_fpe.cosmology.screen_capacity import (
    DEFAULT_L_PLANCK_M,
    OPHScreenCapacityConstants,
    bare_horizon_area_ratio,
    capacity_readback_proxy_report,
    entropy_capacity_from_radius,
    lambda_planck2_from_capacity,
    physical_cells_for_entropy_capacity,
    screen_capacity_closure_report,
    write_capacity_readback_proxy_report,
    write_screen_capacity_closure_report,
)


def test_screen_capacity_observed_branch_identities():
    n_patch = bare_horizon_area_ratio()
    n_scr = entropy_capacity_from_radius()

    assert n_scr == math.pi * n_patch
    assert lambda_planck2_from_capacity(n_scr) == 3.0 * math.pi / n_scr
    assert physical_cells_for_entropy_capacity(n_scr) == 4.0 * n_scr / P_STAR


def test_screen_capacity_report_keeps_regulator_separate():
    report = screen_capacity_closure_report(regulator_patch_counts=(256, 1024))
    observed = report["observed_branch_normalization"]
    rows = report["regulator_scale_comparison"]

    assert report["mode"] == "oph_screen_capacity_closure_v0"
    assert observed["N_scr_entropy_capacity"] > 1.0e122
    assert observed["Lambda_lP2"] < 1.0e-121
    assert rows[0]["patch_count"] == 256
    assert rows[0]["fraction_of_observed_N_scr"] < 1.0e-118
    assert report["readiness_gates"]["observed_branch_N_scr_readout_available"] is True
    assert report["readiness_gates"]["active_edge_center_predictive_quotient_implemented"] is False
    assert report["readiness_gates"]["capacity_readback_map_from_terminal_records_implemented"] is False
    assert report["readiness_gates"]["N_CRC_fixed_point_solved_from_finite_simulator"] is False
    assert report["physical_cmb_prediction"] is False
    assert report["active_capacity_requirements"]["capacity_variable"] == "entropy_capacity_N_not_raw_Hilbert_dimension"


def test_direct_n_crc_input_is_declared_capacity_not_patch_count():
    n_crc = 3.25e122
    report = screen_capacity_closure_report(n_crc=n_crc, regulator_patch_counts=(1024,))
    observed = report["observed_branch_normalization"]
    constants = OPHScreenCapacityConstants(n_crc=n_crc)

    assert observed["input_mode"] == "direct_N_CRC_closure_input"
    assert observed["N_CRC"] == n_crc
    assert observed["N_scr_entropy_capacity"] == n_crc
    assert observed["N_patch_bare_radius_squared_ratio"] == n_crc / math.pi
    assert observed["R_dS_m"] == math.sqrt(n_crc / math.pi) * DEFAULT_L_PLANCK_M
    assert observed["constants"]["N_CRC"] == constants.n_crc
    assert observed["constants"]["N_cells_if_tiled_by_local_P_cells"] == constants.physical_cell_count
    assert report["regulator_scale_comparison"][0]["patch_count"] == 1024
    assert report["regulator_scale_comparison"][0]["fraction_of_observed_N_scr"] < 1.0e-118
    assert report["readiness_gates"]["banach_contraction_certificate_implemented"] is False


def test_write_screen_capacity_report(tmp_path: Path):
    report = write_screen_capacity_closure_report(tmp_path / "capacity")

    assert (tmp_path / "capacity" / "screen_capacity_closure_report.json").exists()
    assert (tmp_path / "capacity" / "screen_capacity_closure_report.md").exists()
    assert report["readiness_gates"]["F_N_readback_map_implemented"] is False


def test_capacity_readback_proxy_reports_finite_rows_without_fixed_point(tmp_path: Path):
    run = tmp_path / "run_4096"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"run_id": "proxy", "patch_count": 4096}), encoding="utf-8")
    (run / "observer_views.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"record_signature": "a", "object_id": "o1"}),
                json.dumps({"record_signature": "b", "object_id": "o2"}),
                json.dumps({"record_signature": "a", "object_id": "o3"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "observer_chart_object_h3_report.json").write_text(
        json.dumps({"observer_chart_object_h3_receipt": True, "object_count": 5}),
        encoding="utf-8",
    )

    report = capacity_readback_proxy_report([tmp_path])
    row = report["rows"][0]

    assert report["row_count"] == 1
    assert row["patch_count"] == 4096
    assert row["observer_count"] == 3
    assert row["active_record_signature_count"] == 2
    assert row["terminal_normal_form_count_proxy"] == 5
    assert report["readiness_gates"]["finite_regulator_rows_present"] is True
    assert report["readiness_gates"]["F_N_readback_map_implemented"] is False
    assert report["readiness_gates"]["N_CRC_fixed_point_solved_from_finite_simulator"] is False


def test_write_capacity_readback_proxy_report(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "observer_chart_object_h3_lineage_report.json").write_text(
        json.dumps({"observer_chart_object_h3_receipt": True, "object_count": 7}),
        encoding="utf-8",
    )

    report = write_capacity_readback_proxy_report([run], tmp_path / "proxy")

    assert report["row_count"] == 1
    assert (tmp_path / "proxy" / "capacity_readback_proxy_report.json").exists()
    assert (tmp_path / "proxy" / "capacity_readback_proxy_report.md").exists()
    assert (tmp_path / "proxy" / "capacity_readback_proxy_rows.csv").exists()
