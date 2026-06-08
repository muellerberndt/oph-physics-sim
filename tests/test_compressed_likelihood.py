from __future__ import annotations

from pathlib import Path

from oph_fpe.cosmology.compressed_likelihood import (
    compressed_likelihood_reference_report,
    write_compressed_likelihood_reference_report,
)


def test_compressed_likelihood_keeps_reference_point_and_s8_tension():
    report = compressed_likelihood_reference_report()

    assert report["oph_compressed_point"]["chi2_diag"] == 11.46326129
    assert report["acceptance"]["reproduces_reference_chi2"] is True
    assert report["acceptance"]["weak_lensing_S8_tension_visible"] is True
    assert report["row_contributions"][0]["row"] == "Weak-lensing S8"
    assert report["physical_cmb_prediction"] is False


def test_write_compressed_likelihood_reference_report(tmp_path: Path):
    report = write_compressed_likelihood_reference_report(tmp_path)

    assert report["physical_matter_power_prediction"] is False
    assert (tmp_path / "oph_compressed_likelihood_report.json").exists()
    assert (tmp_path / "oph_compressed_likelihood_rows.csv").exists()
    assert (tmp_path / "oph_compressed_likelihood_scan_points.csv").exists()
