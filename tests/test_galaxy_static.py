from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oph_fpe.cosmology.galaxy_proxy import rar_curve
from oph_fpe.cosmology.galaxy_static import (
    btfr_prediction_from_rar_fit,
    load_static_galaxy_dataset,
    static_galaxy_holdout_report,
    static_galaxy_measurement_report,
    write_static_galaxy_measurement_report,
)

KPC_IN_M = 3.0856775814913673e19
KM2_IN_M2 = 1.0e6


def test_static_galaxy_report_fits_direct_rar_rows(tmp_path: Path):
    csv_path = tmp_path / "rar.csv"
    a0 = 1.2e-10
    lam = 1.35
    gb = np.logspace(-13, -9, 24)
    go = rar_curve(gb, a0_oph=a0, lambda_collar=lam)
    csv_path.write_text(
        "galaxy,g_baryon,g_observed,baryonic_mass,flat_velocity\n"
        + "\n".join(
            f"G{idx % 4},{left:.12e},{right:.12e},{1.0e9 * (idx + 1):.6e},{80.0 + idx:.6f}"
            for idx, (left, right) in enumerate(zip(gb, go, strict=True))
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_static_galaxy_dataset(csv_path)
    report = static_galaxy_measurement_report(dataset, a0_initial=a0, lambda_initial=1.0, min_points=12)

    assert report["STATIC_GALAXY_RAR_BTFR_RECEIPT"] is True
    assert report["OPH_STATIC_GALAXY_BRIDGE_RECEIPT"] is True
    assert report["bridge"] == "static_galaxy"
    assert report["claim_tier"] == "Tier1_phenomenological_continuation"
    assert report["claim_level"] == "continuation"
    assert report["bulk_required"] is False
    assert report["physical_cmb_claim"] is False
    assert report["receipt_name"] == "STATIC_GALAXY_RAR_BTFR_RECEIPT"
    assert report["rar_point_count"] == 24
    assert report["galaxy_count"] == 4
    assert report["measurement_galaxy_count"] == 4
    assert report["rar_galaxy_count"] == 4
    assert report["rar_scatter_dex"] < 1.0e-4
    assert report["shared_a0"] > 0.0
    assert report["shared_lambda_collar"] > 0.0
    assert report["btfr"]["usable"] is True
    assert report["btfr_prediction_from_rar_fit"]["predicted_slope_logM_vs_logV"] == 4.0
    assert "abs_slope_delta" in report["btfr_prediction_from_rar_fit"]
    assert report["holdout_validation"]["usable"] is False


def test_static_galaxy_receipt_uses_companion_measurement_support_for_aggregate_rar(tmp_path: Path):
    csv_path = tmp_path / "aggregate_rar_with_btfr.csv"
    a0 = 1.2e-10
    lam = 1.0
    gb = np.logspace(-12, -10, 16)
    go = rar_curve(gb, a0_oph=a0, lambda_collar=lam)
    lines = ["galaxy,g_baryon,g_observed,baryonic_mass,flat_velocity"]
    for left, right in zip(gb, go, strict=True):
        lines.append(f",{left:.12e},{right:.12e},,")
    for galaxy_idx in range(25):
        lines.append(f"G{galaxy_idx},,,{1.0e8 * (galaxy_idx + 1):.6e},{55.0 + galaxy_idx:.6f}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dataset = load_static_galaxy_dataset(csv_path)
    report = static_galaxy_measurement_report(
        dataset,
        a0_initial=a0,
        lambda_initial=lam,
        min_points=12,
        min_galaxies=20,
        physical_claim=False,
    )

    assert report["rar_galaxy_count"] == 1
    assert report["rar_galaxy_support_count"] == 25
    assert report["measurement_galaxy_count"] == 25
    assert report["STATIC_GALAXY_RAR_BTFR_RECEIPT"] is True
    assert report["physical_claim"] is False


def test_static_galaxy_cli_writer_marks_missing_external_rows(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("galaxy,foo\nG1,1\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    report = write_static_galaxy_measurement_report(csv_path, out_dir)

    assert report["STATIC_GALAXY_RAR_BTFR_RECEIPT"] is False
    assert report["OPH_STATIC_GALAXY_BRIDGE_RECEIPT"] is False
    assert report["bulk_required"] is False
    assert report["physical_cmb_claim"] is False
    assert report["physical_claim"] is False
    written = json.loads((out_dir / "static_galaxy_measurement_report.json").read_text(encoding="utf-8"))
    assert written["reason"] == "no usable RAR acceleration rows or SPARC-style velocity rows"


def test_static_galaxy_loader_accepts_official_sparc_mrt_tables(tmp_path: Path):
    rar_path = tmp_path / "RAR.mrt"
    rar_path.write_text(
        "Title: RAR\n"
        "--------------------------------------------------------------------------------\n"
        " -11.00 0.04 -10.50 0.08\n"
        " -10.00 0.04  -9.75 0.08\n",
        encoding="utf-8",
    )
    btfr_path = tmp_path / "BTFR_Lelli2019.mrt"
    btfr_path.write_text(
        "Title: BTFR\n"
        "--------------------------------------------------------------------------------\n"
        "     DDO154  8.59  0.06 64.00  3.00  47.0   1.7  18.9   0.9  26.2   1.0  48.2   1.4 114.6   5.0  97.4   8.4  95.1   8.4\n",
        encoding="utf-8",
    )

    dataset = load_static_galaxy_dataset(tmp_path)

    assert dataset.row_count == 3
    direct = [row for row in dataset.rows if "g_baryon" in row]
    btfr = [row for row in dataset.rows if "baryonic_mass" in row]
    assert len(direct) == 2
    assert len(btfr) == 1
    assert direct[0]["g_baryon"] == 1.0e-11
    assert direct[0]["g_observed"] == 10.0**-10.5
    assert btfr[0]["galaxy"] == "DDO154"
    assert btfr[0]["flat_velocity"] == 47.0


def test_btfr_prediction_from_rar_fit_uses_low_acceleration_oph_limit():
    report = btfr_prediction_from_rar_fit(
        a0=1.2e-10,
        lambda_collar=1.0,
        observed_btfr={
            "usable": True,
            "slope_logM_vs_logV": 3.8,
            "intercept_logM_vs_logV": 2.1,
            "rms_dex": 0.2,
            "galaxy_count": 50,
        },
    )

    assert report["usable"] is True
    assert report["predicted_slope_logM_vs_logV"] == 4.0
    assert report["predicted_intercept_logM_vs_logV"] > 0.0
    assert report["slope_delta_observed_minus_predicted"] == pytest.approx(-0.2)
    assert report["observed_galaxy_count"] == 50


def test_static_galaxy_holdout_splits_by_galaxy_and_scores_heldout_rows(tmp_path: Path):
    csv_path = tmp_path / "mass_models.csv"
    a0 = 1.1e-10
    lam = 1.25
    lines = ["galaxy,radius_kpc,v_obs,e_v_obs,v_gas,v_disk,v_bulge"]
    for galaxy_idx in range(16):
        for row_idx in range(5):
            radius_kpc = 0.4 + 0.2 * row_idx + 0.03 * galaxy_idx
            radius_m = radius_kpc * KPC_IN_M
            gb = 10.0 ** (-12.6 + 0.08 * row_idx + 0.015 * galaxy_idx)
            v_gas = np.sqrt(gb * radius_m / KM2_IN_M2)
            go = rar_curve(np.asarray([gb]), a0_oph=a0, lambda_collar=lam)[0]
            v_obs = np.sqrt(go * radius_m / KM2_IN_M2)
            lines.append(f"G{galaxy_idx},{radius_kpc:.12f},{v_obs:.12f},2.0,{v_gas:.12f},0.0,0.0")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dataset = load_static_galaxy_dataset(csv_path)
    report = static_galaxy_holdout_report(
        dataset,
        a0_initial=1.0e-10,
        lambda_initial=1.0,
        train_fraction=0.625,
        seed=123,
    )

    assert report["usable"] is True
    assert report["receipt"] is True
    assert report["train_galaxy_count"] == 10
    assert report["test_galaxy_count"] == 6
    assert report["train_point_count"] == 50
    assert report["test_point_count"] == 30
    assert report["shared_a0"] / (report["shared_lambda_collar"] ** 2) == pytest.approx(
        a0 / (lam * lam),
        rel=0.01,
    )
    assert report["test"]["log_acceleration_rmse_dex"] < 1.0e-3
    assert report["test"]["velocity_rmse_improvement_fraction"] > 0.95
