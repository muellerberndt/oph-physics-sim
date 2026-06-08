from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.cosmology.cmb_anomaly import (
    anomaly_control_separation,
    cmb_anomaly_report,
    spectrum_anomaly_stats,
    write_cmb_anomaly_report,
)


def _spectrum(values: list[float]) -> list[dict[str, float]]:
    rows = []
    for ell, dell in enumerate(values):
        rows.append({"ell": float(ell), "D_ell": float(dell)})
    return rows


def test_spectrum_anomaly_stats_computes_low_power_parity_and_s12():
    spectrum = _spectrum([0.0, 0.0, 2.0, 6.0, 2.0, 6.0, 2.0, 6.0, 2.0, 6.0])

    stats = spectrum_anomaly_stats(spectrum, low_lmax=9, parity_lmax=9, s12_lmax=9, point_count=256)

    assert stats["usable"] is True
    assert stats["low_power_abs_fraction"] == 1.0
    assert stats["odd_even_abs_ratio_Dell"] > 1.0
    assert stats["parity_log_abs_deviation"] > 0.0
    assert stats["S_1_2_scalar_proxy"] >= 0.0
    assert stats["screen_power_fit"]["fit_available"] is True


def test_anomaly_control_separation_flags_suppressed_low_power_and_large_angle():
    target = {
        "low_power_abs_fraction": 0.1,
        "S_1_2_scalar_proxy": 0.2,
        "parity_log_abs_deviation": 0.8,
        "odd_even_abs_ratio_Dell": 2.2,
        "eta_R_estimate": 0.03,
    }
    controls = {
        "a": {
            "low_power_abs_fraction": 0.4,
            "S_1_2_scalar_proxy": 0.9,
            "parity_log_abs_deviation": 0.1,
            "odd_even_abs_ratio_Dell": 1.0,
            "eta_R_estimate": 0.2,
        },
        "b": {
            "low_power_abs_fraction": 0.5,
            "S_1_2_scalar_proxy": 1.1,
            "parity_log_abs_deviation": 0.2,
            "odd_even_abs_ratio_Dell": 1.1,
            "eta_R_estimate": 0.1,
        },
    }

    separation = anomaly_control_separation(target, controls)

    assert separation["low_power_suppressed_vs_controls"] is True
    assert separation["large_angle_suppressed_vs_controls"] is True
    assert separation["parity_more_asymmetric_than_controls"] is True


def test_write_cmb_anomaly_report_from_cl_bundle(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    cl_report = {
        "estimator": "spherical_harmonic",
        "ell_max": 12,
        "point_count": 256,
        "fields": {
            "record_signature": {
                "spectrum": _spectrum([0.0, 0.0, 1.0, 2.0, 1.5, 2.5, 1.7, 2.4, 1.8, 2.2, 2.0, 2.1, 2.2])
            }
        },
        "controls": {
            "record_signature": {
                "shuffled_field": {
                    "spectrum": _spectrum(
                        [0.0, 0.0, 4.0, 4.0, 3.5, 3.5, 3.0, 3.0, 2.8, 2.8, 2.6, 2.6, 2.4]
                    )
                }
            }
        },
    }
    (run / "cl_comparison_report.json").write_text(json.dumps(cl_report), encoding="utf-8")

    report = write_cmb_anomaly_report(run, low_lmax=6, parity_lmax=9, s12_lmax=9)

    assert (run / "cmb_anomaly_report.json").exists()
    assert (run / "cmb_anomaly_report.md").exists()
    assert (run / "cmb_anomaly_rows.csv").exists()
    assert report["physical_cmb_prediction"] is False
    assert report["aggregate"]["primary_field"] == "record_signature"
    assert report["question_answer_status"]["why_acoustic_peaks"]["status"] == "not_answered_by_screen_anomaly_report"


def test_cmb_anomaly_report_imports_public_reference_when_present(tmp_path: Path):
    source = tmp_path / "cmb"
    (source / "3").mkdir(parents=True)
    (source / "4").mkdir()
    (source / "3" / "04_parity_lowpower_statistics_TT_TE_EE_v0_5.csv").write_text(
        "spectrum,lmax,source,R_odd_even,low_power_sum_Dell\nTT,29,Planck_obs,1.3,10\n",
        encoding="utf-8",
    )
    (source / "3" / "05_scalar_S12_correlation_proxy_TT_EE_v0_5.csv").write_text(
        "spectrum,lmax,source,S_1_2_scalar_Dell_proxy\nTT,29,Planck_obs,42\n",
        encoding="utf-8",
    )
    (source / "4" / "02_current_OPH_CMB_targets_v0_8.csv").write_text(
        "quantity,value,unit,method,status\nguardrail_IR_q_IR,0.1,dimensionless,fit,diagnostic\n",
        encoding="utf-8",
    )

    report = cmb_anomaly_report(
        {"ell_max": 4, "point_count": 64, "fields": {"record_signature": {"spectrum": _spectrum([0, 0, 1, 1, 1])}}},
        source_dir=source,
    )

    assert report["public_reference"]["available"] is True
    assert report["public_reference"]["v0_8_current_targets"]["guardrail_IR_q_IR"]["value"] == 0.1
