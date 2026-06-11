from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from oph_fpe.constants.oph_pixel import OPHPixelConstants
from oph_fpe.cosmology.cmb_derivation import write_cmb_parameter_derivation_report
from oph_fpe.cosmology.unique_predictions import (
    parity_odd_even_ratio,
    unique_prediction_gate_report,
    write_unique_prediction_gate_report,
)


def test_unique_prediction_gate_computes_alpha_linked_targets(tmp_path: Path):
    report = unique_prediction_gate_report()
    pixel = OPHPixelConstants()
    expected_eta = math.e * pixel.alpha_from_P * pixel.sqrt_pi

    assert math.isclose(report["scalar_tilt"]["eta_R"], expected_eta)
    assert abs(report["scalar_tilt"]["n_s"] - 0.964841143031) < 2.0e-12
    assert report["cmb_ir_kernel"]["q_IR"] == 0.25
    assert report["cmb_ir_kernel"]["ell_IR"] == 32.0
    assert report["cmb_ir_kernel"]["N_frz_proxy"] == 1089
    assert report["selector_elimination_v1_5"]["q_IR_selector_removed"] is True
    assert report["selector_elimination_v1_5"]["ell_IR_selector_removed"] is True
    assert report["selector_elimination_v1_5"]["eta_R_reduced_to_repair_clock_certificate"] is True
    assert report["selector_elimination_v1_5"]["theorem_side_receipt"] is True
    assert report["scalar_tilt"]["canonical_kappa_rep_status"] == "certificate_pending"
    assert abs(report["parity_envelope"]["predicted_R_OE_TT_2_29"] - 1.2160638411338078) < 1e-12
    assert abs(report["parity_envelope"]["unweighted_envelope_R_OE_2_29_debug"] - parity_odd_even_ratio(range(2, 30))) < 1e-12
    assert abs(report["neutrino_cosmology"]["sum_mnu_eV"] - 0.09001192964464505) < 1e-14
    assert report["measurement_comparable_now"] is True
    assert report["finite_lattice_derived"] is False
    assert report["physical_cmb_prediction"] is False


def test_unique_prediction_gate_imports_v09_csvs(tmp_path: Path):
    source = tmp_path / "cmb"
    source.mkdir()
    _write_csv(source / "01_unique_prediction_ranking_v0_9.csv", [{"rank": 1, "prediction_id": "U1"}])
    _write_csv(source / "02_public_assessment_table_v0_9.csv", [{"assessment_id": "A1", "quantity": "n_s"}])

    report = write_unique_prediction_gate_report(source, tmp_path / "out")

    assert (tmp_path / "out" / "oph_unique_prediction_gate_report.json").exists()
    assert (tmp_path / "out" / "oph_unique_prediction_gate_report.md").exists()
    assert report["source_files"]["ranking_csv_present"] is True
    assert report["source_files"]["assessment_csv_present"] is True
    assert report["ranking_rows"][0]["prediction_id"] == "U1"
    assert report["assessment_rows"][0]["assessment_id"] == "A1"


def test_cmb_derivation_audit_separates_targets_from_finite_readiness(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "cmb_anomaly_report.json").write_text(
        json.dumps(
            {
                "point_count": 4096,
                "ell_max": 128,
                "screen_capacity": {
                    "patch_count": 4096,
                    "total_entropy_capacity": 1670.1,
                    "ell_sqrt_patch_capacity_proxy": 63,
                },
                "aggregate": {
                    "field_count": 2,
                    "best_eta_R_estimate": -2.0,
                    "best_n_s_proxy": 3.0,
                    "low_power_suppressed_vs_controls_count": 0,
                    "large_angle_suppressed_vs_controls_count": 0,
                    "parity_more_asymmetric_than_controls_count": 1,
                    "planck_tilt_compatible_proxy_count": 0,
                },
                "fields": {
                    "record_signature": {
                        "stats": {
                            "screen_power_fit": {
                                "low_ell_suppression": {
                                    "q_IR_proxy": 0.01,
                                    "ell_IR_proxy": 2,
                                }
                            }
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "collar_markov_report.json").write_text(
        json.dumps({"median_epsilon_cmi": 0.4, "p90_epsilon_cmi": 0.5}),
        encoding="utf-8",
    )
    (run / "bulk_reconstruction_report.json").write_text(
        json.dumps({"bulk_3d_established": False}),
        encoding="utf-8",
    )
    (run / "bulk_proof_certificate_report.json").write_text(
        json.dumps(
            {
                "theorem_assisted_h3_populated_chart_established": True,
                "strict_neutral_third_person_bulk_established": False,
            }
        ),
        encoding="utf-8",
    )
    (run / "oph_cmb_stress_report.json").write_text(
        json.dumps({"diagnostic_kernel_proxy": {"B_A_k_a_emitted": False}}),
        encoding="utf-8",
    )

    report = write_cmb_parameter_derivation_report([run], tmp_path / "out")

    assert (tmp_path / "out" / "cmb_parameter_derivation_report.json").exists()
    assert report["run_count"] == 1
    assert report["finite_lattice_cmb_parameters_ready"] is False
    assert report["physical_cmb_prediction"] is False
    assert report["rows"][0]["gates"]["tilt_eta_R_simulator_compatible"] is False
    assert report["rows"][0]["gates"]["theorem_assisted_h3_bulk_established"] is True
    assert report["rows"][0]["gates"]["strict_neutral_bulk_3d_established"] is False
    assert report["rows"][0]["bulk_3d_established"] is False
    assert report["rows"][0]["target_q_IR"] == 0.25


def test_cmb_derivation_prefers_bulk_proof_over_legacy_emergence_flag(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "cmb_anomaly_report.json").write_text(
        json.dumps(
            {
                "aggregate": {"field_count": 1, "parity_more_asymmetric_than_controls_count": 1},
                "fields": {},
            }
        ),
        encoding="utf-8",
    )
    (run / "emergence_status_report.json").write_text(
        json.dumps(
            {
                "bulk_3d_established": True,
                "PAPER_THEOREM_ASSISTED_H3_POPULATED_CHART_RECEIPT": True,
            }
        ),
        encoding="utf-8",
    )
    (run / "bulk_proof_certificate_report.json").write_text(
        json.dumps(
            {
                "theorem_assisted_h3_populated_chart_established": False,
                "theorem_assisted_h3_nonboundary_population_established": False,
                "bulk_3d_established_theorem_assisted": False,
                "strict_neutral_third_person_bulk_established": False,
            }
        ),
        encoding="utf-8",
    )

    report = write_cmb_parameter_derivation_report([run], tmp_path / "out")

    assert report["rows"][0]["gates"]["theorem_assisted_h3_bulk_established"] is False
    assert report["rows"][0]["gates"]["strict_neutral_bulk_3d_established"] is False


def test_cmb_derivation_audit_includes_scalar_quotient_lane(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "scalar_quotient_report.json").write_text(
        json.dumps(
            {
                "SCALAR_QUOTIENT_RECEIPT": True,
                "observer_count": 1089,
                "patch_count": 262144,
                "scalar_packet_alphabet_size": 199,
                "scalar_packet_entropy_bits": 7.0,
                "edge_center_readout": {
                    "theta_OPH_P_over_48": 0.033978504362582485,
                    "n_s_P_over_48": 0.9660214956374176,
                },
                "active_angular_levels": {
                    "target_ell_IR": 32,
                    "observer_level_proxy_floor_sqrt_observers": 33,
                    "patch_capacity_level_proxy_floor_sqrt_patches": 512,
                },
                "readiness_gates": {
                    "active_33_level_freezeout_clause": True,
                    "theorem_grade_scalar_release_code": False,
                },
                "finite_lattice_cmb_scalar_release_ready": False,
                "blockers": ["theorem_grade_scalar_release_code_missing"],
            }
        ),
        encoding="utf-8",
    )

    report = write_cmb_parameter_derivation_report([run], tmp_path / "out")
    row = report["rows"][0]

    assert report["run_count"] == 1
    assert row["scalar_quotient_receipt"] is True
    assert row["scalar_quotient_observer_count"] == 1089
    assert row["scalar_quotient_active_33_level_freezeout_clause"] is True
    assert row["scalar_quotient_theorem_grade_release_code"] is False
    assert row["gates"]["scalar_quotient_receipt"] is True
    assert row["gates"]["scalar_quotient_active_33_level_freezeout_clause"] is True
    assert row["gates"]["scalar_quotient_finite_ready"] is False
    assert report["aggregate"]["mean_scalar_quotient_n_s"] == 0.9660214956374176
    assert report["finite_lattice_cmb_parameters_ready"] is False


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
