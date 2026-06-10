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
    assert report["theorem_assisted_h3_object_preview_established"] is True
    assert report["theorem_assisted_h3_nonboundary_population_established"] is True
    assert report["theorem_assisted_h3_populated_chart_established"] is True
    assert report["theorem_assisted_observer_facing_h3_population"] is True
    assert report["observer_facing_h3_object_population_receipt"] is True
    assert report["OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT"] is True
    assert report["strict_neutral_third_person_bulk_established"] is False
    assert report["bulk_3d_established_theorem_assisted"] is True
    assert report["bulk_3d_established_strict"] is False
    assert report["screen_cmb_proxy_available"] is True
    assert report["physical_cmb_prediction"] is False


def test_bulk_proof_preview_does_not_promote_to_nonboundary_population(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "emergence_status_report.json",
        {
            "BW_KMS_DIRECT_2PI_RECEIPT": True,
            "PAPER_THEOREM_3D_BULK_CHART_RECEIPT": True,
            "H3_RESPONSE_CANDIDATE_RECEIPT": True,
            "THEOREM_ASSISTED_H3_OBJECT_PREVIEW_RECEIPT": True,
            "PAPER_THEOREM_ASSISTED_H3_POPULATED_CHART_RECEIPT": True,
        },
    )

    report = bulk_proof_certificate(run)

    assert report["theorem_assisted_h3_object_preview_established"] is True
    assert report["theorem_assisted_h3_nonboundary_population_established"] is False
    assert report["theorem_assisted_h3_populated_chart_established"] is False
    assert report["bulk_3d_established_theorem_assisted"] is False


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
    assert lane["bulk_proof_theorem_assisted_h3_object_preview_count"] == 1
    assert lane["bulk_proof_theorem_assisted_h3_nonboundary_population_count"] == 1
    assert lane["bulk_proof_theorem_assisted_h3_populated_chart_count"] == 1
    assert lane["bulk_proof_strict_neutral_3d_bulk_count"] == 0
    assert lane["bulk_proof_screen_cmb_proxy_count"] == 1
    assert lane["bulk_proof_physical_cmb_prediction_count"] == 0


def test_bulk_proof_certificate_accepts_output_directory(tmp_path: Path):
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_json(
        run / "emergence_status_report.json",
        {
            "BW_KMS_DIRECT_2PI_RECEIPT": True,
            "PAPER_THEOREM_3D_BULK_CHART_RECEIPT": True,
        },
    )

    write_bulk_proof_certificate(run, out)

    assert (out / "bulk_proof_certificate_report.json").exists()
    assert (out / "bulk_proof_certificate_report.md").exists()


def test_bulk_proof_certificate_reads_scale_compressed_branch_without_overclaiming(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "scale_compressed_repair_report.json",
        {
            "logical_repair_rounds": 24,
            "scale_compressed_operator_receipt": True,
            "repair_round_trace_receipt": True,
            "cmb_parameter_readouts": {
                "eta_R": 0.0339785,
                "n_s": 0.9660215,
                "q_IR": 0.25,
                "ell_IR": 32.0,
            },
            "h3_preview": {
                "object_count": 48,
                "cap_count": 120,
                "cap_profile_receipt": True,
                "populated_h3_preview_receipt": True,
                "strict_neutral_third_person_bulk_established": False,
            },
            "particle_preview": {
                "particle_worldline_count": 6,
                "particle_preview_receipt": True,
                "production_particle_matter_receipt": False,
            },
            "physical_cmb_prediction": False,
            "strict_neutral_bulk": False,
        },
    )
    _write_json(
        run / "scale_compressed_cmb_camb_report.json",
        {
            "measurement_comparable_cmb_curve": True,
            "screen_camb_transfer_receipt": True,
            "physical_cmb_prediction": False,
            "comparison": {
                "scale_compressed_ir_kernel": {
                    "shape_correlation": 0.999,
                    "normalized_rmse": 0.017,
                    "best_fit_column_chi2_per_bin": 0.96,
                }
            },
        },
    )
    _write_json(
        run / "scale_compressed_particle_report.json",
        {
            "worldline_count": 6,
            "particle_preview_receipt": True,
            "production_particle_matter_receipt": False,
        },
    )

    report = bulk_proof_certificate(run)

    assert report["scale_compressed_operator_receipt"] is True
    assert report["scale_compressed_repair_round_trace_receipt"] is True
    assert report["scale_compressed_h3_preview_established"] is True
    assert report["scale_compressed_measurement_comparable_cmb_curve"] is True
    assert report["scale_compressed_particle_preview_established"] is True
    assert report["screen_cmb_proxy_available"] is True
    assert report["physical_cmb_prediction"] is False
    assert report["production_particle_matter_receipt"] is False
    assert report["strict_neutral_third_person_bulk_established"] is False
    assert report["bulk_3d_established_theorem_assisted"] is False
    assert report["proof_tiers"]["T5c_scale_compressed_h3_preview"]["passed"] is True
    assert report["proof_tiers"]["T8b_scale_compressed_camb_transfer"]["passed"] is True
    assert report["scale_compressed_summary"]["h3_object_count"] == 48
    assert report["scale_compressed_summary"]["camb_ir_chi2_per_bin"] == 0.96


def test_bulk_proof_certificate_reads_paper_chart_files_without_strict_bulk(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "paper_3d_bulk_chart_report.json",
        {
            "PAPER_THEOREM_3D_BULK_CHART_RECEIPT": True,
            "paper_theorem_3d_bulk_chart_receipt": True,
            "chart_level_conformal_lorentz_receipt": True,
            "bw_2pi_cap_flow_receipt": True,
            "lorentz_group": "SO+(3,1)",
            "spatial_homogeneous_space": "H3 = SO+(3,1)/SO(3)",
            "h3_spatial_dimension_from_boost_orbit": 3,
            "h3_chart_spatial_dimension": 3,
            "finite_point_cloud_dimension_estimator_used": False,
        },
    )
    _write_json(run / "bulk_reconstruction_report.json", {"bulk_3d_established": False})

    report = bulk_proof_certificate(run)

    assert report["chart_level_3p1_lorentz_kinematics_established"] is True
    assert report["strict_neutral_third_person_bulk_established"] is False
    assert report["bulk_3d_established_theorem_assisted"] is False
    assert report["paper_chart_summary"]["h3_spatial_dimension_from_boost_orbit"] == 3
    assert report["paper_chart_summary"]["finite_point_cloud_dimension_estimator_used"] is False
