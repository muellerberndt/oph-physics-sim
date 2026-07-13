from __future__ import annotations

import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator

from oph_fpe.constants.oph_pixel import P_STAR
from oph_fpe.cosmology.neutrino_background import oph_cnb_background_report, write_oph_cnb_background_report


def test_oph_cnb_background_report_keeps_kernel_gates_closed():
    report = oph_cnb_background_report(source_dir=None)
    expected_eta = 1.0 - math.exp(-P_STAR / 24.0)
    expected_epsilon = 1.0 - 0.790 / 0.828924043

    assert report["mode"] == "oph_cnb_neutrino_background_v1"
    assert report["oph_neutrino_mass_status"]["available"] is False
    assert report["oph_neutrino_mass_status"]["masses_eV"] is None
    assert report["oph_neutrino_mass_status"]["sum_mnu_eV"] is None
    conventional = report["conventional_camb_baseline"]
    assert conventional["sum_mnu_eV"] == 0.06
    assert conventional["counts_as_oph_prediction"] is False
    assert conventional["relic_background"]["N_eff"] == 3.044
    rejected = report["historical_rejected_weighted_cycle_benchmark"]
    assert rejected["included"] is False
    assert rejected["masses_eV"] is None
    assert rejected["public_promotion_allowed"] is False
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
    assert report["readiness_gates"]["oph_neutrino_mass_prediction_available"] is False
    assert report["readiness_gates"]["conventional_baseline_relic_background_callable"] is True
    assert report["readiness_gates"]["finite_lattice_mass_derivation"] is False
    assert report["readiness_gates"]["z6_poisson_five_of_seven_kernel_callable"] is True
    assert report["readiness_gates"]["B_A_k_a_from_finite_collar_parent"] is False
    assert report["theorem_package_coverage"]["minimal_one_pole_kernel_callable"] is True
    assert report["theorem_package_coverage"]["no_universal_weak_lensing_projection_constant_declared"] is True
    assert report["theorem_package_coverage"]["finite_collar_window_derived_by_simulator"] is False
    assert report["physical_cmb_prediction"] is False

    schema_path = Path(__file__).parents[1] / "schemas" / "cosmology" / "neutrino_status.schema.json"
    Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(report)


def test_oph_cnb_rejected_weighted_cycle_benchmark_is_explicit_and_non_promotable():
    report = oph_cnb_background_report(
        source_dir=None,
        include_rejected_weighted_cycle_benchmark=True,
    )
    rejected = report["historical_rejected_weighted_cycle_benchmark"]

    assert rejected["included"] is True
    assert abs(rejected["sum_mnu_eV"] - 0.09001192964464505) < 1.0e-14
    assert rejected["public_promotion_allowed"] is False
    assert rejected["declared_gate"]["current_weighted_cycle_candidate_rejected"] is True


def test_write_oph_cnb_background_report_exports_rows(tmp_path: Path):
    source = tmp_path / "neutrinos"
    source.mkdir()
    (source / "neutrinos-1.md").write_text("one", encoding="utf-8")
    (source / "neutrinos2.md").write_text("two", encoding="utf-8")
    (source / "neutrino3.md").write_text("", encoding="utf-8")

    report = write_oph_cnb_background_report(source, tmp_path / "out")

    assert (tmp_path / "out" / "oph_cnb_neutrino_report.json").exists()
    assert (tmp_path / "out" / "oph_cnb_neutrino_report.md").exists()
    assert (tmp_path / "out" / "oph_cnb_conventional_baseline_mass_rows.csv").exists()
    assert (tmp_path / "out" / "oph_cnb_conventional_baseline_comparison_rows.csv").exists()
    assert (tmp_path / "out" / "oph_cnb_conventional_baseline_free_streaming_rows.csv").exists()
    assert (tmp_path / "out" / "oph_cnb_historical_rejected_weighted_cycle_benchmark_rows.csv").exists()
    assert report["source_files"]["all_expected_files_present"] is True
    assert report["source_files"]["empty_files"] == ["neutrino3.md"]
