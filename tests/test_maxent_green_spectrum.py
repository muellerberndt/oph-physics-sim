from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.constants.oph_pixel import OPHPixelConstants
from oph_fpe.cosmology.edge_center_clock import edge_center_clock_target
from oph_fpe.cosmology.maxent_green_spectrum import (
    maxent_green_spectrum_report,
    write_maxent_green_spectrum_report,
)


def test_maxent_green_spectrum_encodes_paper_source_without_physical_claim():
    report = maxent_green_spectrum_report(patch_count=4096, ell_max=32, primordial_k_count=8)
    pixel = OPHPixelConstants()
    target = edge_center_clock_target(pixel.P)

    assert report["MAXENT_GREEN_SOURCE_RECEIPT"] is False
    assert report["maxent_inverse_laplacian"]["eta0_flat_D_ell_receipt"] is True
    assert report["finite_regulator"]["bandlimit_for_ir_receipt"] is True
    assert report["finite_regulator"]["bandlimit_for_requested_ell_receipt"] is True
    assert report["selector_elimination_v1_5"]["q_IR"] == 0.25
    assert report["selector_elimination_v1_5"]["ell_IR"] == 32.0
    assert report["selector_elimination_v1_5"]["N_frz_proxy"] == 1089
    assert report["selector_elimination_v1_5"]["theorem_side_receipt"] is False
    assert report["selector_elimination_v1_5"]["q_IR_selector_removed"] is False
    assert report["selector_elimination_v1_5"]["ell_IR_selector_removed"] is False
    assert report["fractional_repair_tilt"]["eta_R"] == target.theta
    assert report["fractional_repair_tilt"]["n_s"] == target.n_s
    assert report["fractional_repair_tilt"]["kappa_rep"] == target.kappa_rep
    assert report["fractional_repair_tilt"]["kappa_rep_source"] == (
        "selected_edge_center_p_over_48_target"
    )
    assert report["fractional_repair_tilt"]["e_diagnostic_control"]["promoting"] is False
    assert report["fractional_repair_tilt"]["repair_clock_certificate"] is False
    assert report["EDGE_CENTER_CLOCK_RECEIPT"] is False
    assert report["finite_lattice_derived"] is False
    assert report["physical_cmb_prediction"] is False


def test_maxent_kappa_override_is_explicitly_nonselected():
    report = maxent_green_spectrum_report(
        patch_count=4096,
        ell_max=32,
        primordial_k_count=8,
        kappa_rep=9.0,
    )
    tilt = report["fractional_repair_tilt"]

    assert tilt["selected_branch"] == "diagnostic_kappa_override"
    assert tilt["selected_theorem_target"] is False
    assert tilt["kappa_rep_source"] == "diagnostic_override"
    assert "not the selected P/48 target" in tilt["formula"]
    assert report["MAXENT_GREEN_SOURCE_RECEIPT"] is False
    assert report["physical_cmb_prediction"] is False


def test_write_maxent_green_spectrum_exports_screen_power_scaffold(tmp_path: Path):
    report = write_maxent_green_spectrum_report(tmp_path / "out", patch_count=65_536, ell_max=64, primordial_k_count=8)

    assert (tmp_path / "out" / "maxent_green_spectrum_report.json").exists()
    assert (tmp_path / "out" / "maxent_green_spectrum_rows.csv").exists()
    assert (tmp_path / "out" / "oph_screen_power_report.json").exists()
    assert (tmp_path / "out" / "oph_primordial_power_CLASS_CAMB.txt").exists()
    scaffold = json.loads((tmp_path / "out" / "oph_screen_power_report.json").read_text(encoding="utf-8"))
    assert scaffold["simulator_primordial_reference_ready"] is False
    assert scaffold["physical_cmb_prediction"] is False
    assert scaffold["primordial_reference_source"] == (
        "paper_maxent_green_spectrum_plus_selector_elimination_not_finite_lattice"
    )
    assert scaffold["reference_screen_parameters"]["q_IR"] == 0.25
    assert scaffold["reference_screen_parameters"]["ell_IR"] == 32.0
    assert scaffold["reference_screen_parameters"]["eta_R"] == report["fractional_repair_tilt"]["eta_R"]
