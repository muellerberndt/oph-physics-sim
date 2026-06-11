from __future__ import annotations

import json
import math
from pathlib import Path

from oph_fpe.constants.oph_pixel import P_STAR
from oph_fpe.cosmology.comparable_data import comparable_data_report
from oph_fpe.measurement_pack import export_measurement_pack
from oph_fpe.scale.scale_compressed_repair import scale_compressed_repair_run
from oph_fpe.viz.scale_compressed_viewer import write_scale_compressed_viewer


def test_scale_compressed_repair_run_writes_branch_outputs(tmp_path: Path):
    report = scale_compressed_repair_run(
        tmp_path / "compressed",
        repair_rounds=24,
        object_count=8,
        particle_count=2,
        cap_axis_count=6,
        ell_max=32,
        seed=123,
    )

    out = tmp_path / "compressed"
    assert (out / "scale_compressed_repair_report.json").exists()
    assert (out / "scale_compressed_repair_rounds.csv").exists()
    assert (out / "scale_compressed_h3_objects.csv").exists()
    assert (out / "scale_compressed_particles.csv").exists()
    assert (out / "scale_compressed_screen_cl.csv").exists()
    assert (out / "cl_comparison_report.json").exists()

    assert report["scale_compressed_operator_receipt"] is True
    assert report["repair_round_trace_receipt"] is True
    assert report["h3_preview"]["populated_h3_preview_receipt"] is True
    assert report["h3_preview"]["strict_neutral_third_person_bulk_established"] is False
    assert report["particle_preview"]["particle_preview_receipt"] is True
    assert report["particle_preview"]["production_particle_matter_receipt"] is False
    assert report["physical_cmb_prediction"] is False
    assert math.isclose(report["cmb_parameter_readouts"]["n_s"], 1.0 - P_STAR / 48.0)


def test_scale_compressed_repair_aggregates_without_opening_physical_gates(tmp_path: Path):
    run_dir = tmp_path / "compressed"
    scale_compressed_repair_run(
        run_dir,
        repair_rounds=24,
        object_count=8,
        particle_count=2,
        cap_axis_count=6,
        ell_max=32,
        seed=456,
    )
    (run_dir / "scale_compressed_cmb_camb_report.json").write_text(
        json.dumps(
            {
                "mode": "oph_scale_compressed_cmb_camb_transfer_v0",
                "measurement_comparable_cmb_curve": True,
                "screen_camb_transfer_receipt": True,
                "physical_cmb_prediction": False,
                "scale_compressed_input": {
                    "logical_repair_rounds": 24,
                    "scale_compressed_operator_receipt": True,
                    "populated_h3_preview_receipt": True,
                    "eta_R": 1.0 - (1.0 - P_STAR / 48.0),
                    "n_s": 1.0 - P_STAR / 48.0,
                    "q_IR": 0.25,
                    "ell_IR": 32.0,
                },
                "comparison": {
                    "camb_lcdm_powerlaw": {
                        "shape_correlation": 0.99,
                        "amplitude_fit_chi2_per_bin": 1.2,
                    },
                    "scale_compressed_scalar_tilt": {
                        "shape_correlation": 0.98,
                        "amplitude_fit_chi2_per_bin": 1.3,
                    },
                    "scale_compressed_ir_kernel": {
                        "shape_correlation": 0.97,
                        "amplitude_fit_chi2_per_bin": 1.4,
                    },
                },
                "acoustic_preservation": {"mean_abs_fractional_delta_ell_ge_50": 0.02},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "scale_compressed_cmb_tt_bins.csv").write_text("ell,observed_D_ell\n2,100\n", encoding="utf-8")
    (run_dir / "scale_compressed_cmb_tt_curves.csv").write_text(
        "ell,scale_compressed_ir_kernel_D_ell\n2,100\n",
        encoding="utf-8",
    )

    report = comparable_data_report([run_dir])
    lane = report["measurement_lanes"]["oph_scale_compressed_repair_branch"]
    camb_lane = report["measurement_lanes"]["oph_scale_compressed_cmb_camb_transfer"]

    assert lane["run_count"] == 1
    assert lane["operator_receipt_count"] == 1
    assert lane["populated_h3_preview_count"] == 1
    assert lane["strict_neutral_bulk_count"] == 0
    assert lane["physical_cmb_prediction_count"] == 0
    assert math.isclose(lane["mean_n_s"], 1.0 - P_STAR / 48.0)
    assert camb_lane["run_count"] == 1
    assert camb_lane["measurement_comparable_curve_count"] == 1
    assert camb_lane["physical_cmb_prediction_count"] == 0
    assert math.isclose(camb_lane["mean_n_s"], 1.0 - P_STAR / 48.0)

    pack = export_measurement_pack([run_dir], tmp_path / "pack")
    claims = json.loads((tmp_path / "pack" / "claims.json").read_text())
    assert "scale_compressed_repair_report.json" in pack["files"]
    assert "scale_compressed_h3_objects.csv" in pack["files"]
    assert "scale_compressed_cmb_camb_report.json" in pack["files"]
    assert "scale_compressed_cmb_tt_bins.csv" in pack["files"]
    assert claims["scale_compressed_operator_receipt"] is True
    assert claims["scale_compressed_populated_h3_preview"] is True
    assert claims["scale_compressed_physical_cmb_prediction"] is False
    assert claims["scale_compressed_cmb_curve_comparable"] is True
    assert claims["scale_compressed_cmb_physical_prediction"] is False


def test_scale_compressed_viewer_writes_html_without_opening_gates(tmp_path: Path):
    run_dir = tmp_path / "compressed"
    scale_compressed_repair_run(
        run_dir,
        repair_rounds=24,
        object_count=8,
        particle_count=2,
        cap_axis_count=6,
        ell_max=32,
        seed=789,
    )

    summary = write_scale_compressed_viewer(run_dir)

    viewer = Path(summary["viewer_path"])
    assert viewer.exists()
    assert "OPH Scale-Compressed Repair Viewer" in viewer.read_text(encoding="utf-8")
    assert summary["scale_compressed_operator_receipt"] is True
    assert summary["populated_h3_preview_receipt"] is True
    assert summary["physical_cmb_prediction"] is False
    assert summary["strict_neutral_bulk"] is False


def test_scale_compressed_viewer_reads_nested_camb_transfer(tmp_path: Path):
    run_dir = tmp_path / "compressed"
    scale_compressed_repair_run(
        run_dir,
        repair_rounds=24,
        object_count=8,
        particle_count=2,
        cap_axis_count=6,
        ell_max=32,
        seed=790,
    )
    nested = run_dir / "camb_transfer"
    nested.mkdir()
    (nested / "scale_compressed_cmb_camb_report.json").write_text(
        json.dumps(
            {
                "measurement_comparable_cmb_curve": True,
                "comparison": {
                    "scale_compressed_ir_kernel": {
                        "shape_correlation": 0.97,
                        "amplitude_fit_chi2_per_bin": 1.4,
                    },
                    "camb_lcdm_powerlaw": {"shape_correlation": 0.99},
                },
            }
        ),
        encoding="utf-8",
    )
    (nested / "scale_compressed_cmb_tt_bins.csv").write_text(
        "ell,observed_D_ell,scale_compressed_ir_kernel_D_ell,camb_lcdm_powerlaw_D_ell\n"
        "2,100,80,100\n",
        encoding="utf-8",
    )

    summary = write_scale_compressed_viewer(run_dir)

    assert summary["measurement_comparable_cmb_curve"] is True
    assert summary["camb_bin_count"] == 1
    assert summary["camb_report_path"].endswith("camb_transfer/scale_compressed_cmb_camb_report.json")
