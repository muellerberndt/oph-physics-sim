from __future__ import annotations

import json
import math
from pathlib import Path

from oph_fpe.constants.oph_pixel import OPHPixelConstants
from oph_fpe.cosmology.maxent_green_spectrum import (
    maxent_green_spectrum_report,
    write_maxent_green_spectrum_report,
)


def test_maxent_green_spectrum_encodes_paper_source_without_physical_claim():
    report = maxent_green_spectrum_report(patch_count=4096, ell_max=32, primordial_k_count=8)
    pixel = OPHPixelConstants()
    expected_eta = math.e * (pixel.P - pixel.phi)

    assert report["MAXENT_GREEN_SOURCE_RECEIPT"] is True
    assert report["maxent_inverse_laplacian"]["eta0_flat_D_ell_receipt"] is True
    assert report["finite_regulator"]["bandlimit_for_ir_receipt"] is True
    assert report["finite_regulator"]["bandlimit_for_requested_ell_receipt"] is True
    assert report["selector_elimination_v1_5"]["q_IR"] == 0.25
    assert report["selector_elimination_v1_5"]["ell_IR"] == 32.0
    assert report["selector_elimination_v1_5"]["N_frz_proxy"] == 1089
    assert math.isclose(report["fractional_repair_tilt"]["eta_R"], expected_eta)
    assert abs(report["fractional_repair_tilt"]["n_s"] - 0.964841143031) < 2.0e-12
    assert report["fractional_repair_tilt"]["repair_clock_certificate"] is False
    assert report["finite_lattice_derived"] is False
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
