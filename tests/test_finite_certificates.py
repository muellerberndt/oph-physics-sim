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
    run_proxy_certificate_input,
    toy_certificate_input,
    write_finite_certificate_bundle,
    write_run_proxy_finite_certificate_bundle,
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
        release["A_q_cmi_upper_bound"],
        4.0 * math.log(2.0) * release["kappa_rel"] * release["epsilon_star_bits"],
    )
    assert release["A_q_energy"] is None
    assert release["A_zeta"] is None
    assert release["SCREEN_TO_PRIMORDIAL_LIFT_RECEIPT"] is False
    assert parent["small_field_support"]["passes"] is True
    assert parent["Q_A"] > 0.0
    assert parent["kernels"][0]["B_A"] > 0.0
    assert repair["row_sum_max_error"] < 1.0e-12
    assert repair["detailed_balance_max_error"] < 1.0e-12
    assert repair["Gamma_rec"] > 0.0
    assert math.isclose(boltzmann["n_s"], 1.0 - toy_certificate_input()["metadata"]["P"] / 48.0)
    assert boltzmann["A_zeta"] is None
    assert boltzmann["primordial_lift_ready"] is False
    assert report["report"]["finite_certificate_compiler_ready"] is True
    assert report["report"]["finite_certificate_stack_ready"] is False
    assert report["report"]["theorem_grade_finite_inputs"] is False
    assert report["report"]["proxy_certificate"] is True
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
    assert manifest["finite_certificate_compiler_ready"] is True
    assert manifest["finite_certificate_stack_ready"] is False
    assert manifest["theorem_grade_finite_inputs"] is False
    assert manifest["proxy_certificate"] is True
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
    assert lane["compiler_ready_count"] == 1
    assert lane["stack_ready_count"] == 0
    assert lane["theorem_grade_finite_inputs_count"] == 0
    assert lane["proxy_certificate_count"] == 1
    assert lane["no_data_use_count"] == 1
    assert lane["real_physics_certificate_count"] == 0
    assert lane["physical_cmb_prediction_count"] == 0
    assert lane["mean_A_zeta"] is None


def test_run_proxy_certificate_bundle_uses_cached_run_receipts(tmp_path: Path):
    run = tmp_path / "cached_run"
    run.mkdir()
    (run / "collar_markov_report.json").write_text(
        json.dumps(
            {
                "mode": "diagonal_empirical_collar_state",
                "rows": [
                    {
                        "cap_id": 0,
                        "theta0": 0.5,
                        "collar_count": 10,
                        "sample_count": 100,
                        "epsilon_cmi": 0.2,
                    },
                    {
                        "cap_id": 1,
                        "theta0": 0.8,
                        "collar_count": 20,
                        "sample_count": 100,
                        "epsilon_cmi": 0.3,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (run / "manifest.json").write_text(
        json.dumps({"run_id": "r1", "oph_constants": {"P": 1.6309682094039593}}),
        encoding="utf-8",
    )
    (run / "s3_class_counts.json").write_text(
        json.dumps({"identity": 4, "transposition": 3, "threecycle": 2}),
        encoding="utf-8",
    )
    (run / "h0s8_branch_report.json").write_text(
        json.dumps({"background_values": {"Omega_A": 0.26, "Omega_b": 0.05}}),
        encoding="utf-8",
    )
    (run / "oph_boltzmann_input_report.json").write_text(
        json.dumps({"grids": {"a_grid": [0.1, 1.0]}, "readiness": {"cdm_limit_solver_ready": True}}),
        encoding="utf-8",
    )

    data = run_proxy_certificate_input(run)
    assert data["metadata"]["real_physics_certificate"] is False
    assert data["metadata"]["proxy_certificate"] is True
    assert data["metadata"]["theorem_grade_release_code"] is False
    assert data["parent_collar"]["theorem_grade_parent_collar_ladder"] is False
    assert data["parent_collar"]["small_field_support"]["passes"] is False
    assert data["parent_collar"]["refinement_convergence"]["passes"] is False
    assert data["repair_matrix"]["theorem_grade_repair_matrix"] is False
    assert data["repair_matrix"]["actual_repair_event_trace"] is False
    assert len(data["release_code"]["packets"]) == 2
    assert data["parent_collar"]["a_values"] == [0.1, 1.0]
    assert data["repair_matrix"]["states"] == ["identity", "transposition", "threecycle"]

    out = tmp_path / "bundle"
    report = write_run_proxy_finite_certificate_bundle(run, out)
    assert report["finite_certificate_compiler_ready"] is True
    assert report["finite_certificate_stack_ready"] is False
    assert report["theorem_grade_finite_inputs"] is False
    assert report["real_physics_certificate"] is False
    assert report["physical_cmb_prediction"] is False
    assert report["proxy_certificate"] is True
    assert (out / "finite_certificate_input_from_run.json").exists()
