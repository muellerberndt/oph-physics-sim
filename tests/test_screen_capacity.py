from __future__ import annotations

import math
from pathlib import Path

from oph_fpe.constants.oph_pixel import P_STAR
from oph_fpe.cosmology.screen_capacity import (
    bare_horizon_area_ratio,
    entropy_capacity_from_radius,
    lambda_planck2_from_capacity,
    physical_cells_for_entropy_capacity,
    screen_capacity_closure_report,
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
    assert report["readiness_gates"]["N_CRC_fixed_point_solved_from_finite_simulator"] is False
    assert report["physical_cmb_prediction"] is False


def test_write_screen_capacity_report(tmp_path: Path):
    report = write_screen_capacity_closure_report(tmp_path / "capacity")

    assert (tmp_path / "capacity" / "screen_capacity_closure_report.json").exists()
    assert (tmp_path / "capacity" / "screen_capacity_closure_report.md").exists()
    assert report["readiness_gates"]["F_N_readback_map_implemented"] is False
