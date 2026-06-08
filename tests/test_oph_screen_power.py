from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oph_fpe.cosmology.oph_screen_power import (
    C_ell_oph,
    D_ell_from_C_ell,
    OPHScreenPowerParams,
    F_oph_k,
    primordial_power_oph,
    screen_power_fit_from_spectrum,
    write_oph_screen_power_report,
)


def test_red_eta_r_makes_D_ell_red_not_blue():
    ell = np.arange(30, 1000, dtype=float)
    flat = D_ell_from_C_ell(ell, C_ell_oph(ell, OPHScreenPowerParams(eta_R=0.0)))
    red = D_ell_from_C_ell(ell, C_ell_oph(ell, OPHScreenPowerParams(eta_R=0.035)))

    flat_slope = np.polyfit(np.log(ell), np.log(flat), 1)[0]
    red_slope = np.polyfit(np.log(ell), np.log(red), 1)[0]

    assert abs(flat_slope) < 0.01
    assert red_slope < -0.03


def test_screen_power_fit_recovers_eta_convention():
    ell = np.arange(2, 200, dtype=float)
    params = OPHScreenPowerParams(A_chi=2.5, eta_R=0.04, ell_cap=None, N_cap_eff=None)
    c_ell = C_ell_oph(ell, params)
    d_ell = D_ell_from_C_ell(ell, c_ell)
    spectrum = [
        {"ell": float(e), "C_ell": float(c), "D_ell": float(d)}
        for e, c, d in zip(ell, c_ell, d_ell, strict=True)
    ]

    fit = screen_power_fit_from_spectrum(spectrum, field_name="synthetic", point_count=4096, ell_min=20)

    assert fit["fit_available"] is True
    assert abs(fit["eta_R_estimate"] - 0.04) < 1.0e-3
    assert abs(fit["n_s_proxy"] - 0.96) < 1.0e-3
    assert "N_eff" not in json.dumps(fit)
    assert "N_cap_eff" in json.dumps(fit)


def test_primordial_bridge_excludes_parity_from_isotropic_correction():
    k = np.array([0.001, 0.01, 0.1])
    base = OPHScreenPowerParams(eta_R=0.035, epsilon_parity=0.0)
    parity = OPHScreenPowerParams(eta_R=0.035, epsilon_parity=0.9)

    assert np.allclose(F_oph_k(k, base)["F_OPH"], F_oph_k(k, parity)["F_OPH"])
    assert np.all(primordial_power_oph(k, base)["P_R"] > 0.0)


def test_write_oph_screen_power_report_exports_primordial_table(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    ell = np.arange(2, 64, dtype=float)
    c_ell = C_ell_oph(ell, OPHScreenPowerParams(A_chi=1.2, eta_R=0.035, ell_cap=None, N_cap_eff=None))
    d_ell = D_ell_from_C_ell(ell, c_ell)
    cl_report = {
        "point_count": 4096,
        "fields": {
            "record_signature": {
                "spectrum": [
                    {"ell": float(e), "C_ell": float(c), "D_ell": float(d)}
                    for e, c, d in zip(ell, c_ell, d_ell, strict=True)
                ]
            }
        },
    }
    (run / "cl_comparison_report.json").write_text(json.dumps(cl_report), encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps({"run_id": "synthetic", "patch_count": 4096}), encoding="utf-8")

    report = write_oph_screen_power_report([tmp_path], tmp_path / "out", primordial_k_count=8)

    assert report["physical_cmb_prediction"] is False
    assert report["aggregate"]["available_fit_count"] == 1
    assert report["simulator_primordial_reference_ready"] is True
    assert report["primordial_reference_source"] == "simulator_eta_R_estimate"
    assert (tmp_path / "out" / "oph_screen_power_report.json").exists()
    assert (tmp_path / "out" / "oph_primordial_power_CLASS_CAMB.txt").exists()


def test_blue_screen_fit_falls_back_to_labeled_planck_target_scaffold(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    ell = np.arange(2, 64, dtype=float)
    d_ell = ell**2
    c_ell = d_ell * 2.0 * np.pi / (ell * (ell + 1.0))
    cl_report = {
        "point_count": 4096,
        "fields": {
            "record_signature": {
                "spectrum": [
                    {"ell": float(e), "C_ell": float(c), "D_ell": float(d)}
                    for e, c, d in zip(ell, c_ell, d_ell, strict=True)
                ]
            }
        },
    }
    (run / "cl_comparison_report.json").write_text(json.dumps(cl_report), encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps({"run_id": "blue", "patch_count": 4096}), encoding="utf-8")

    report = write_oph_screen_power_report([tmp_path], tmp_path / "out", primordial_k_count=8)

    assert report["simulator_primordial_reference_ready"] is False
    assert report["primordial_reference_source"] == "phenomenological_planck_eta_target_due_to_invalid_simulator_tilt"
    assert abs(report["reference_screen_parameters"]["eta_R"] - 0.035) < 1.0e-12
