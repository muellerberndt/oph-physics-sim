from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from oph_fpe.cosmology.comparable_data import comparable_data_report
from oph_fpe.cosmology.finite_certificates import (
    cmi_bits,
    finite_certificate_bundle,
    no_data_use_receipt,
    toy_certificate_input,
    write_finite_certificate_bundle,
)


def test_cmi_bits_zero_for_conditionally_independent_table():
    p = np.array(
        [
            [[0.25, 0.25], [0.0, 0.0]],
            [[0.0, 0.0], [0.25, 0.25]],
        ],
        dtype=float,
    )

    assert math.isclose(cmi_bits(p), 0.0)


def test_finite_certificate_bundle_recomputes_toy_values():
    report = finite_certificate_bundle(toy_certificate_input())
    release = report["release_code"]
    parent = report["parent_collar"]
    repair = report["repair_matrix"]
    boltzmann = report["boltzmann_export"]

    assert release["minimizer_packets"] == ["q2_defect"]
    assert release["N_rel"] == 2
    assert math.isclose(
        release["A_zeta"],
        100.0 * math.log(2.0) * release["kappa_rel"] * release["epsilon_star_bits"],
    )
    assert parent["small_field_support"]["passes"] is True
    assert parent["Q_A"] > 0.0
    assert parent["kernels"][0]["B_A"] > 0.0
    assert repair["row_sum_max_error"] < 1.0e-12
    assert repair["detailed_balance_max_error"] < 1.0e-12
    assert repair["Gamma_rec"] > 0.0
    assert math.isclose(boltzmann["n_s"], 1.0 - toy_certificate_input()["metadata"]["P"] / 48.0)
    assert report["report"]["finite_certificate_stack_ready"] is True
    assert report["report"]["physical_cmb_prediction"] is False
    assert report["report"]["real_physics_certificate"] is False


def test_finite_certificate_writer_outputs_manifest_and_report(tmp_path: Path):
    out_dir = tmp_path / "certs"
    report = write_finite_certificate_bundle(None, out_dir, toy=True)

    assert (out_dir / "release_code_certificate.json").exists()
    assert (out_dir / "parent_collar_certificate.json").exists()
    assert (out_dir / "repair_matrix_certificate.json").exists()
    assert (out_dir / "boltzmann_export_certificate.json").exists()
    assert (out_dir / "no_data_use_receipt.json").exists()
    assert (out_dir / "finite_certificate_manifest.json").exists()
    assert (out_dir / "finite_certificate_report.json").exists()
    assert (out_dir / "finite_certificate_report.md").exists()
    manifest = json.loads((out_dir / "finite_certificate_manifest.json").read_text())
    assert manifest["manifest_type"] == "oph_finite_certificate_manifest"
    assert manifest["finite_certificate_stack_ready"] is True
    assert report["physical_cmb_prediction"] is False


def test_no_data_use_receipt_rejects_forbidden_measurement_keys():
    data = toy_certificate_input()
    data["metadata"]["planck_likelihood"] = {"chi2": 0.0}

    receipt = no_data_use_receipt(data)

    assert receipt["no_data_use_receipt"] is False
    assert "planck_likelihood" in receipt["forbidden_inputs_present"]


def test_comparable_data_includes_finite_certificate_lane(tmp_path: Path):
    out_dir = tmp_path / "run"
    write_finite_certificate_bundle(None, out_dir, toy=True)

    report = comparable_data_report([out_dir])
    lane = report["measurement_lanes"]["oph_finite_certificate_authority"]

    assert report["run_count"] == 1
    assert lane["run_count"] == 1
    assert lane["stack_ready_count"] == 1
    assert lane["no_data_use_count"] == 1
    assert lane["real_physics_certificate_count"] == 0
    assert lane["physical_cmb_prediction_count"] == 0
    assert lane["mean_A_zeta"] is not None
