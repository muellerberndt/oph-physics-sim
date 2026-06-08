from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.bulk.proof_certificate import bulk_proof_certificate, write_bulk_proof_certificate
from oph_fpe.cosmology.comparable_data import comparable_data_report


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_bulk_proof_certificate_splits_theorem_assisted_from_strict_neutral(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "emergence_status_report.json",
        {
            "BW_KMS_DIRECT_2PI_RECEIPT": True,
            "PAPER_THEOREM_3D_BULK_CHART_RECEIPT": True,
            "CHART_LEVEL_CONFORMAL_LORENTZ_RECEIPT": True,
            "CHART_LORENTZ_H3_RECEIPT": True,
            "H3_RESPONSE_CANDIDATE_RECEIPT": True,
            "H3_RESPONSE_CONTROL_SEPARATION_RECEIPT": True,
            "OBJECT_BULK_POPULATION_RECEIPT": True,
            "observer_chart_bulk_population_receipt": True,
            "PAPER_THEOREM_ASSISTED_H3_POPULATED_CHART_RECEIPT": True,
            "SCREEN_PROXY_CMB_RECEIPT": True,
            "particle_matter_receipt": False,
            "physical_cmb_prediction": False,
        },
    )
    _write_json(run / "bulk_reconstruction_report.json", {"bulk_3d_established": False})
    _write_json(run / "cmb_lite_comparison_report.json", {"physical_cmb_prediction": False})

    report = bulk_proof_certificate(run)

    assert report["chart_level_3p1_lorentz_kinematics_established"] is True
    assert report["theorem_assisted_h3_populated_chart_established"] is True
    assert report["strict_neutral_third_person_bulk_established"] is False
    assert report["bulk_3d_established_theorem_assisted"] is True
    assert report["bulk_3d_established_strict"] is False
    assert report["screen_cmb_proxy_available"] is True
    assert report["physical_cmb_prediction"] is False


def test_bulk_proof_certificate_writes_and_comparable_data_collects_tiers(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "emergence_status_report.json",
        {
            "BW_KMS_DIRECT_2PI_RECEIPT": True,
            "PAPER_THEOREM_3D_BULK_CHART_RECEIPT": True,
            "H3_RESPONSE_CANDIDATE_RECEIPT": True,
            "OBJECT_BULK_POPULATION_RECEIPT": True,
            "SCREEN_PROXY_CMB_RECEIPT": True,
        },
    )
    write_bulk_proof_certificate(run)

    snapshot = comparable_data_report([tmp_path])
    lane = snapshot["measurement_lanes"]["support_visible_lorentz_branch"]

    assert lane["bulk_proof_certificate_count"] == 1
    assert lane["bulk_proof_chart_level_3p1_count"] == 1
    assert lane["bulk_proof_theorem_assisted_h3_populated_chart_count"] == 1
    assert lane["bulk_proof_strict_neutral_3d_bulk_count"] == 0
    assert lane["bulk_proof_screen_cmb_proxy_count"] == 1
    assert lane["bulk_proof_physical_cmb_prediction_count"] == 0
