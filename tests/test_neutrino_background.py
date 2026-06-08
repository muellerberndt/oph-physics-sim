from __future__ import annotations

import math
from pathlib import Path

from oph_fpe.constants.oph_pixel import P_STAR
from oph_fpe.cosmology.neutrino_background import oph_cnb_background_report, write_oph_cnb_background_report


def test_oph_cnb_background_report_keeps_kernel_gates_closed():
    report = oph_cnb_background_report(source_dir=None)
    expected_eta = 1.0 - math.exp(-P_STAR / 24.0)
    expected_epsilon = 1.0 - 0.790 / 0.828924043

    assert report["mode"] == "oph_cnb_neutrino_background_v0"
    assert abs(report["oph_neutrino_branch"]["sum_mnu_eV"] - 0.09001192964464505) < 1.0e-14
    assert report["relic_background"]["N_eff"] == 3.044
    assert abs(report["late_repair_projection_target"]["eta_A"] - expected_eta) < 1.0e-15
    assert abs(report["late_repair_projection_target"]["Pi_WL_compressed_required"] - expected_epsilon / expected_eta) < 1.0e-12
    projected = report["late_repair_projection_target"]["projected_amplitude_theorem"]
    assert projected["compressed_target_not_microphysical_constant"] is True
    assert abs(projected["reconstructed_S8_from_required_Pi_WL"] - 0.790) < 1.0e-14
    five = report["late_repair_projection_target"]["z6_poisson_five_of_seven"]
    assert five["branch"] == "z6_poisson_five_of_seven"
    assert five["pi_wl"] == 5.0 / 7.0
    assert abs(five["S8_projected_from_cdm_branch"] - 0.7900242005) < 1.0e-9
    assert five["boltzmann_lite_kernel_callable"] is True
    assert five["finite_collar_emitted_kA_tau_rec"] is False
    assert report["readiness_gates"]["measurement_comparable_relic_background"] is True
    assert report["readiness_gates"]["finite_lattice_mass_derivation"] is False
    assert report["readiness_gates"]["z6_poisson_five_of_seven_kernel_callable"] is True
    assert report["readiness_gates"]["B_A_k_a_from_finite_collar_parent"] is False
    assert report["theorem_package_coverage"]["minimal_one_pole_kernel_callable"] is True
    assert report["theorem_package_coverage"]["no_universal_weak_lensing_projection_constant_declared"] is True
    assert report["theorem_package_coverage"]["finite_collar_window_derived_by_simulator"] is False
    assert report["physical_cmb_prediction"] is False


def test_write_oph_cnb_background_report_exports_rows(tmp_path: Path):
    source = tmp_path / "neutrinos"
    source.mkdir()
    (source / "neutrinos-1.md").write_text("one", encoding="utf-8")
    (source / "neutrinos2.md").write_text("two", encoding="utf-8")
    (source / "neutrino3.md").write_text("", encoding="utf-8")

    report = write_oph_cnb_background_report(source, tmp_path / "out")

    assert (tmp_path / "out" / "oph_cnb_neutrino_report.json").exists()
    assert (tmp_path / "out" / "oph_cnb_neutrino_report.md").exists()
    assert (tmp_path / "out" / "oph_cnb_neutrino_mass_rows.csv").exists()
    assert (tmp_path / "out" / "oph_cnb_neutrino_comparison_rows.csv").exists()
    assert (tmp_path / "out" / "oph_cnb_free_streaming_rows.csv").exists()
    assert report["source_files"]["all_expected_files_present"] is True
    assert report["source_files"]["empty_files"] == ["neutrino3.md"]
