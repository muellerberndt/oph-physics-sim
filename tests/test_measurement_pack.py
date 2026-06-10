from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.measurement_pack import export_measurement_pack


def test_export_measurement_pack_copies_static_galaxy_tables(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "static_galaxy_measurement_report.json").write_text(
        json.dumps(
            {
                "STATIC_GALAXY_RAR_BTFR_RECEIPT": True,
                "OPH_STATIC_GALAXY_BRIDGE_RECEIPT": True,
                "physical_claim": False,
            }
        ),
        encoding="utf-8",
    )
    (run / "galaxy_rar_fit.csv").write_text("row,g_baryon\n0,1e-11\n", encoding="utf-8")
    (run / "galaxy_btfr_fit.csv").write_text("galaxy,flat_velocity\nG1,100\n", encoding="utf-8")
    (run / "galaxy_rotation_residuals.csv").write_text("galaxy,residual_km_s\nG1,1\n", encoding="utf-8")

    out = tmp_path / "pack"
    report = export_measurement_pack([run], out)

    assert report["claims"]["static_galaxy_measurement_fit"] is True
    assert report["claims"]["physical_cmb_prediction"] is False
    assert "README.md" in report["files"]
    assert "measurement_pack_report.json" in report["files"]
    assert (out / "claims.json").exists()
    assert (out / "README.md").exists()
    assert (out / "galaxy_rar_fit.csv").read_text(encoding="utf-8").startswith("row,g_baryon")
    assert (out / "galaxy_btfr_fit.csv").exists()
    assert (out / "galaxy_rotation_residuals.csv").exists()


def test_export_measurement_pack_copies_bulk_and_comparable_receipts(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "bulk_proof_certificate_report.json").write_text(
        json.dumps({"chart_level_3p1_lorentz_kinematics_established": True}),
        encoding="utf-8",
    )
    (run / "bulk_proof_certificate_report.md").write_text("# proof\n", encoding="utf-8")
    (run / "comparable_data_snapshot.json").write_text(
        json.dumps({"chart_level_3p1_any": True}),
        encoding="utf-8",
    )
    (run / "comparable_data_snapshot.md").write_text("# snapshot\n", encoding="utf-8")
    (run / "paper_3d_bulk_chart_report.json").write_text(json.dumps({"receipt": True}), encoding="utf-8")
    (run / "conformal_h3_spatial_chart_report.json").write_text(json.dumps({"receipt": True}), encoding="utf-8")
    (run / "transition_selection_report.json").write_text(json.dumps({"two_pi_selected": True}), encoding="utf-8")

    out = tmp_path / "pack"
    report = export_measurement_pack([run], out)

    assert report["claims"]["chart_level_3p1"] is True
    assert "bulk_proof_certificate_report.json" in report["files"]
    assert "bulk_proof_certificate_report.md" in report["files"]
    assert "comparable_data_snapshot.json" in report["files"]
    assert "comparable_data_snapshot.md" in report["files"]
    assert "paper_3d_bulk_chart_report.json" in report["files"]
    assert "conformal_h3_spatial_chart_report.json" in report["files"]
    assert "transition_selection_report.json" in report["files"]


def test_export_measurement_pack_aggregates_sweep_cl_and_transition_report(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    (sweep / "comparable_data_snapshot.json").write_text(
        json.dumps(
            {
                "chart_level_3p1_count": 2,
                "theorem_assisted_h3_bulk_count": 0,
                "strict_neutral_3d_bulk_count": 0,
                "physical_cmb_prediction": False,
                "measurement_lanes": {
                    "support_visible_lorentz_branch": {
                        "support_visible_lorentz_3p1_count": 2,
                        "paper_theorem_3d_bulk_chart_count": 2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    for index in range(2):
        run = sweep / f"seed_{index}"
        run.mkdir()
        (run / "cl_proxy.csv").write_text(
            "field,ell,C_ell,D_ell\nrecord_signature,2,0.1,0.2\n",
            encoding="utf-8",
        )
        (run / "transition_scale_selection_report.json").write_text(
            json.dumps({"two_pi_selected": True, "selected_label": "2pi"}),
            encoding="utf-8",
        )

    out = tmp_path / "pack"
    report = export_measurement_pack([sweep], out)

    assert report["claims"]["chart_level_3p1"] is True
    assert (out / "cmb_screen_cl.csv").read_text(encoding="utf-8").count("record_signature") == 2
    assert json.loads((out / "transition_selection_report.json").read_text(encoding="utf-8"))[
        "two_pi_selected"
    ] is True


def test_export_measurement_pack_backfills_cl_rows_from_report(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "cl_comparison_report.json").write_text(
        json.dumps(
            {
                "fields": {
                    "record_signature": {
                        "spectrum": [
                            {"ell": 2, "C_ell": 0.1, "D_ell": 0.2},
                            {"ell": 3, "C_ell": 0.3, "D_ell": 0.4},
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "pack"
    export_measurement_pack([run], out)

    text = (out / "cmb_screen_cl.csv").read_text(encoding="utf-8")
    assert "source_run,field,ell,C_ell,D_ell" in text
    assert text.count("record_signature") == 2


def test_export_measurement_pack_missing_json_placeholders_are_parseable(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()

    out = tmp_path / "pack"
    export_measurement_pack([run], out)

    assert json.loads((out / "boltzmann_export_certificate.json").read_text(encoding="utf-8")) == {}
    assert json.loads((out / "finite_certificate_report.json").read_text(encoding="utf-8")) == {}


def test_export_measurement_pack_copies_receipt_viewer(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "oph_receipt_viewer.html").write_text("<html>viewer</html>\n", encoding="utf-8")
    (run / "oph_realtime_viewer_summary.json").write_text(json.dumps({"viewer": True}), encoding="utf-8")

    out = tmp_path / "pack"
    report = export_measurement_pack([run], out)

    assert "oph_receipt_viewer.html" in report["files"]
    assert "oph_realtime_viewer_summary.json" in report["files"]
    assert (out / "oph_receipt_viewer.html").read_text(encoding="utf-8").startswith("<html>")


def test_export_measurement_pack_copies_source_side_cosmology_reports(tmp_path: Path) -> None:
    run = tmp_path / "sources"
    (run / "screen_power").mkdir(parents=True)
    (run / "maxent_green").mkdir(parents=True)
    (run / "parent").mkdir(parents=True)
    (run / "neutrinos").mkdir(parents=True)
    (run / "h0s8").mkdir(parents=True)
    (run / "cmb").mkdir(parents=True)

    (run / "screen_capacity_closure_report.json").write_text(
        json.dumps(
            {
                "readiness_gates": {
                    "observed_branch_N_scr_readout_available": True,
                    "N_CRC_fixed_point_solved_from_finite_simulator": False,
                }
            }
        ),
        encoding="utf-8",
    )
    (run / "parent" / "parent_collar_ladder_report.json").write_text(
        json.dumps({"local_recovery_density_receipt": True, "theorem_grade_parent_collar_ladder": False}),
        encoding="utf-8",
    )
    (run / "repair_clock_certificate_report.json").write_text(
        json.dumps({"repair_clock_certificate": False, "eta_R_finite_lattice_derived": False}),
        encoding="utf-8",
    )
    (run / "finite_repair_transition_matrix_report.json").write_text(
        json.dumps(
            {
                "finite_transition_matrix_ready": True,
                "clock_normalization_certified": False,
                "eta_R_finite_lattice_derived": False,
            }
        ),
        encoding="utf-8",
    )
    (run / "finite_repair_transition_rows.csv").write_text(
        "matrix,kappa_rep_estimate\nreversible_empirical,2.47\n",
        encoding="utf-8",
    )
    (run / "scalar_repair_semigroup_report.json").write_text(
        json.dumps({"repair_clock_certificate": False, "eta_R_finite_lattice_derived": False}),
        encoding="utf-8",
    )
    (run / "oph_boltzmann_input_report.json").write_text(
        json.dumps(
            {
                "physical_cmb_prediction": False,
                "readiness": {
                    "checks": {
                        "finite_repair_clock_diagnostic_rows_emitted": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "oph_boltzmann_finite_repair_clock_rows.csv").write_text(
        "a,Gamma_rec_over_H_diagnostic\n1,0.032\n",
        encoding="utf-8",
    )
    (run / "b_a_parent_report.json").write_text(
        json.dumps(
            {
                "B_A_PARENT_RECEIPT": False,
                "physical_prediction_ready": False,
                "physical_cmb_prediction": False,
                "readiness": {
                    "checks": {
                        "finite_difference_rows_emitted": True,
                        "finite_observer_view_parent_variation": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "b_a_parent_rows.csv").write_text("a,k_proxy_inverse_theta,B_A_mean\n1,1,0.1\n", encoding="utf-8")
    (run / "b_a_parent_observer_view_rows.csv").write_text(
        "a,k_proxy_inverse_theta,B_A_mean\n1,1,0.1\n", encoding="utf-8"
    )
    (run / "screen_power" / "oph_screen_power_report.json").write_text(
        json.dumps({"simulator_primordial_reference_ready": False}),
        encoding="utf-8",
    )
    (run / "screen_power" / "oph_primordial_power_table.csv").write_text("k,P_R\n1,2\n", encoding="utf-8")
    (run / "maxent_green" / "maxent_green_spectrum_report.json").write_text(
        json.dumps({"MAXENT_GREEN_SOURCE_RECEIPT": True}),
        encoding="utf-8",
    )
    (run / "oph_cmb_selector_elimination_report.json").write_text(
        json.dumps({"THEOREM_SIDE_SELECTOR_ELIMINATION_RECEIPT": True}),
        encoding="utf-8",
    )
    (run / "cmb" / "cmb_anomaly_report.json").write_text(
        json.dumps(
            {
                "aggregate": {
                    "parity_more_asymmetric_than_controls_count": 2,
                    "low_power_suppressed_vs_controls_count": 0,
                    "planck_tilt_compatible_proxy_count": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    (run / "neutrinos" / "oph_cnb_neutrino_report.json").write_text(
        json.dumps({"measurement_comparable_now": True, "finite_lattice_derived": False}),
        encoding="utf-8",
    )
    (run / "h0s8" / "h0s8_branch_report.json").write_text(
        json.dumps({"measurement_comparisons": {"Planck2018_H0": {"branch_pull_sigma": 0.1}}}),
        encoding="utf-8",
    )
    (run / "oph_compressed_likelihood_report.json").write_text(
        json.dumps({"physical_cmb_prediction": False}),
        encoding="utf-8",
    )

    out = tmp_path / "pack"
    report = export_measurement_pack([run], out)

    claims = report["claims"]
    assert claims["screen_capacity_observed_branch_available"] is True
    assert claims["screen_capacity_finite_fixed_point_solved"] is False
    assert claims["parent_collar_local_density_receipt"] is True
    assert claims["parent_collar_theorem_grade"] is False
    assert claims["finite_transition_matrix_ready"] is True
    assert claims["finite_transition_clock_certified"] is False
    assert claims["finite_transition_eta_R_finite_lattice_derived"] is False
    assert claims["boltzmann_input_table_written"] is True
    assert claims["boltzmann_finite_repair_clock_rows_emitted"] is True
    assert claims["b_a_parent_rows_emitted"] is True
    assert claims["b_a_parent_observer_view_variation"] is True
    assert claims["b_a_parent_receipt"] is False
    assert claims["b_a_parent_physical_prediction"] is False
    assert claims["screen_power_simulator_primordial_ready"] is False
    assert claims["maxent_green_source_receipt"] is True
    assert claims["selector_elimination_theorem_side_receipt"] is True
    assert claims["cmb_anomaly_parity_asymmetry_proxy"] is True
    assert claims["neutrino_measurement_comparable"] is True
    assert claims["neutrino_finite_lattice_derived"] is False
    assert claims["h0s8_measurement_comparable"] is True
    assert claims["compressed_likelihood_reference"] is True
    assert "oph_screen_power_primordial_table.csv" in report["files"]
    assert "finite_repair_transition_matrix_report.json" in report["files"]
    assert "finite_repair_transition_rows.csv" in report["files"]
    assert "scalar_repair_semigroup_report.json" in report["files"]
    assert "oph_boltzmann_finite_repair_clock_rows.csv" in report["files"]
    assert "b_a_parent_report.json" in report["files"]
    assert "b_a_parent_rows.csv" in report["files"]
    assert "b_a_parent_observer_view_rows.csv" in report["files"]
    assert "maxent_green_spectrum_report.json" in report["files"]
    assert "cmb_anomaly_report.json" in report["files"]
