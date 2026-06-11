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


def test_export_measurement_pack_regenerates_combined_bulk_certificate(tmp_path: Path) -> None:
    h3_run = tmp_path / "h3_run"
    h3_run.mkdir()
    (h3_run / "emergence_status_report.json").write_text(
        json.dumps(
            {
                "final_phi_zero": True,
                "records_committed": True,
                "H3_RESPONSE_CANDIDATE_RECEIPT": True,
            }
        ),
        encoding="utf-8",
    )
    (h3_run / "paper_3d_bulk_chart_report.json").write_text(
        json.dumps(
            {
                "paper_theorem_3d_bulk_chart_receipt": True,
                "bw_2pi_cap_flow_receipt": True,
                "lorentz_group": "SO+(3,1)",
                "spatial_homogeneous_space": "H3 = SO+(3,1)/SO(3)",
                "h3_spatial_dimension_from_boost_orbit": 3,
                "h3_chart_spatial_dimension": 3,
            }
        ),
        encoding="utf-8",
    )
    (h3_run / "observer_chart_object_h3_lineage_report.json").write_text(
        json.dumps(
            {
                "observer_chart_object_h3_receipt": True,
                "observer_chart_bulk_population_receipt": True,
                "object_count": 12,
                "localized_object_count": 11,
                "localized_not_boundary_object_count": 10,
            }
        ),
        encoding="utf-8",
    )

    scale_run = tmp_path / "scale_run"
    scale_run.mkdir()
    (scale_run / "scale_compressed_repair_report.json").write_text(
        json.dumps(
            {
                "logical_repair_rounds": 24,
                "scale_compressed_operator_receipt": True,
                "repair_round_trace_receipt": True,
                "h3_preview": {
                    "cap_profile_receipt": True,
                    "populated_h3_preview_receipt": True,
                    "object_count": 48,
                    "cap_count": 96,
                },
                "particle_preview": {
                    "particle_preview_receipt": True,
                    "particle_worldline_count": 8,
                },
                "cmb_parameter_readouts": {
                    "eta_R": 0.0339,
                    "n_s": 0.9661,
                    "q_IR": 0.25,
                    "ell_IR": 32.0,
                },
                "physical_cmb_prediction": False,
            }
        ),
        encoding="utf-8",
    )
    (scale_run / "scale_compressed_cmb_camb_report.json").write_text(
        json.dumps(
            {
                "measurement_comparable_cmb_curve": True,
                "screen_camb_transfer_receipt": True,
                "physical_cmb_prediction": False,
                "comparison": {
                    "scale_compressed_ir_kernel": {
                        "shape_correlation": 0.999,
                        "normalized_rmse": 0.02,
                        "best_fit_column_chi2_per_bin": 0.95,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "pack"
    report = export_measurement_pack([h3_run, scale_run], out)
    proof = json.loads((out / "bulk_proof_certificate_report.json").read_text(encoding="utf-8"))

    assert report["claims"]["theorem_assisted_h3_bulk"] is True
    assert report["claims"]["scale_compressed_cmb_curve_comparable"] is True
    assert proof["theorem_assisted_h3_nonboundary_population_established"] is True
    assert proof["scale_compressed_h3_preview_established"] is True
    assert proof["scale_compressed_measurement_comparable_cmb_curve"] is True
    assert proof["scale_compressed_particle_preview_established"] is True
    assert proof["strict_neutral_third_person_bulk_established"] is False
    assert proof["physical_cmb_prediction"] is False


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
    (run / "finite_repair_clock_cmb_camb_report.json").write_text(
        json.dumps(
            {
                "measurement_comparable_cmb_curve": True,
                "finite_lattice_clock_derived": True,
                "physical_cmb_prediction": False,
            }
        ),
        encoding="utf-8",
    )
    (run / "finite_repair_clock_cmb_camb_report.md").write_text("# finite clock", encoding="utf-8")
    (run / "finite_repair_clock_cmb_tt_bins.csv").write_text("ell,D_ell\n50,100\n", encoding="utf-8")
    (run / "finite_repair_clock_cmb_tt_curves.csv").write_text("ell,D_ell\n2,10\n", encoding="utf-8")
    (run / "camb_lcdm_baseline_report.json").write_text(
        json.dumps({"CDM_LIMIT_BOLTZMANN_RECEIPT": True, "oph_anomaly_module_ready": False}),
        encoding="utf-8",
    )
    (run / "camb_lcdm_baseline_report.md").write_text("# camb baseline", encoding="utf-8")
    (run / "camb_lcdm_tt_bins.csv").write_text("ell,D_ell\n225,5700\n", encoding="utf-8")
    (run / "physical_cmb_input_report.json").write_text(
        json.dumps(
            {
                "PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT": False,
                "physical_cmb_prediction_eligible": False,
                "physical_cmb_prediction": False,
                "blockers": ["B_A_k_a_missing_or_not_finite"],
                "input_status": {
                    "A_zeta": {"diagnostic_value_present": True, "physical_gate_passed": False},
                    "B_A_k_a": {
                        "diagnostic_value_present": True,
                        "row_count": 4,
                        "physical_gate_passed": False,
                    },
                    "rho_A_a": {
                        "diagnostic_value_present": True,
                        "row_count": 3,
                        "physical_gate_passed": False,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "physical_cmb_input_report.md").write_text("# physical CMB input", encoding="utf-8")
    (run / "physical_cmb_input_contract.json").write_text(json.dumps({"B_A_source": "diagnostic_proxy"}), encoding="utf-8")
    (run / "physical_cmb_input_validation.json").write_text(
        json.dumps({"PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT": False}),
        encoding="utf-8",
    )
    (run / "physical_cmb_promotion_audit_report.json").write_text(
        json.dumps(
            {
                "physical_cmb_promotion_ready": False,
                "official_likelihood_ready": False,
                "promotion_blockers": [
                    "finite_certificate_proxy_not_theorem_grade",
                    "B_A_kernel_receipt_missing",
                ],
            }
        ),
        encoding="utf-8",
    )
    (run / "physical_cmb_promotion_audit_report.md").write_text("# promotion", encoding="utf-8")
    (run / "B_A_k_a.csv").write_text("k_or_row,a_or_col,B_A\n1,1,0.1\n", encoding="utf-8")
    (run / "Gamma_rec_k_a.csv").write_text("k_or_row,a_or_col,Gamma_rec\n1,1,0.03\n", encoding="utf-8")
    (run / "rho_A_a.csv").write_text("row,col,rho_A\n1,1,0.2\n", encoding="utf-8")
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
    (run / "B_A_kernel_report.json").write_text(
        json.dumps(
            {
                "B_A_KERNEL_CANDIDATE_RECEIPT": True,
                "B_A_KERNEL_RECEIPT": False,
                "row_count": 4,
                "promotion_blockers": ["physical_check_failed_scale_calibrated_k_h_mpc"],
            }
        ),
        encoding="utf-8",
    )
    (run / "B_A_kernel_report.md").write_text("# B_A kernel", encoding="utf-8")
    (run / "B_A_kernel_candidate.csv").write_text("k_bin,a_bin,B_A\n1,1,0.1\n", encoding="utf-8")
    (run / "B_A_kernel_refinement_report.json").write_text(
        json.dumps(
            {
                "two_scale_diagnostic_receipt": True,
                "B_A_KERNEL_REFINEMENT_CONVERGENCE_RECEIPT": False,
                "patch_count_count": 2,
                "key_pair_row_count": 2,
                "key_pair_stable_fraction": 0.5,
                "blockers": [
                    "requires_at_least_three_patch_counts_for_refinement_convergence",
                    "B_A_kernel_pairwise_drift_or_sign_instability",
                ],
            }
        ),
        encoding="utf-8",
    )
    (run / "B_A_kernel_refinement_report.md").write_text("# B_A refinement", encoding="utf-8")
    (run / "B_A_kernel_refinement_pairs.csv").write_text(
        "left_patch_count,right_patch_count,common_key_count\n4096,16384,16\n",
        encoding="utf-8",
    )
    (run / "B_A_kernel_refinement_key_pairs.csv").write_text(
        "left_patch_count,right_patch_count,k_bin,a_bin,left_B_A,right_B_A,key_refinement_pass\n"
        "4096,16384,0.1,0.5,1.0,1.1,true\n",
        encoding="utf-8",
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
    (run / "neutral_profile_audit_report.json").write_text(
        json.dumps(
            {
                "profile_rows": [
                    {"profile": "prime_geometric_rank3", "strict_3d_ready": False},
                    {"profile": "control_probe", "strict_3d_ready": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    (run / "prime_geometric_rank_refinement_report.json").write_text(
        json.dumps(
            {
                "control_quotient_rank3_refinement_candidate_receipt": True,
                "independent_rank3_selector_all": False,
                "candidate_dimension_drift": 0.015,
                "strict_neutral_bulk_refinement_receipt": False,
                "physical_claim": False,
            }
        ),
        encoding="utf-8",
    )
    (run / "neutral_3d_bulk_audit_report.json").write_text(
        json.dumps(
            {
                "strict_neutral_bulk_ready": False,
                "directional_strict_ready_total": 0,
                "control_quotient_candidate_count": 1,
                "blockers": ["independent_svd_rank3_selector_not_stable_or_false"],
            }
        ),
        encoding="utf-8",
    )
    (run / "neutral_3d_bulk_audit_report.md").write_text("# neutral", encoding="utf-8")
    (run / "neutral_independent_rank_selector_audit_report.json").write_text(
        json.dumps(
            {
                "NEUTRAL_INDEPENDENT_RANK3_SELECTOR_RECEIPT": False,
                "run_count": 2,
                "control_quotient_rank3_selector_count": 0,
                "control_quotient_rank3_candidate_count": 2,
                "control_quotient_median_effective_rank": 126.0,
                "control_quotient_median_rank3_cumulative_explained_variance": 0.05,
            }
        ),
        encoding="utf-8",
    )
    (run / "neutral_independent_rank_selector_audit_report.md").write_text("# selector", encoding="utf-8")

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
    assert claims["finite_repair_clock_cmb_curve_comparable"] is True
    assert claims["finite_repair_clock_cmb_finite_lattice_clock"] is True
    assert claims["finite_repair_clock_cmb_physical_prediction"] is False
    assert claims["camb_lcdm_cdm_limit_boltzmann_receipt"] is True
    assert claims["camb_lcdm_oph_anomaly_module_ready"] is False
    assert claims["physical_cmb_input_contract_receipt"] is False
    assert claims["physical_cmb_input_prediction_eligible"] is False
    assert claims["physical_cmb_input_A_zeta_diagnostic_present"] is True
    assert claims["physical_cmb_input_A_zeta_physical_gate_passed"] is False
    assert claims["physical_cmb_input_B_A_diagnostic_rows"] == 4
    assert claims["physical_cmb_input_B_A_physical_gate_passed"] is False
    assert claims["physical_cmb_input_rho_A_diagnostic_rows"] == 3
    assert claims["physical_cmb_input_rho_A_physical_gate_passed"] is False
    assert claims["physical_cmb_promotion_audit_written"] is True
    assert claims["physical_cmb_promotion_ready"] is False
    assert claims["physical_cmb_promotion_official_likelihood_ready"] is False
    assert claims["physical_cmb_promotion_blocker_count"] == 2
    assert claims["b_a_parent_rows_emitted"] is True
    assert claims["b_a_parent_observer_view_variation"] is True
    assert claims["b_a_parent_receipt"] is False
    assert claims["b_a_parent_physical_prediction"] is False
    assert claims["B_A_kernel_candidate_receipt"] is True
    assert claims["B_A_kernel_physical_receipt"] is False
    assert claims["B_A_kernel_row_count"] == 4
    assert claims["B_A_kernel_promotion_blocker_count"] == 1
    assert claims["B_A_kernel_refinement_two_scale_diagnostic"] is True
    assert claims["B_A_kernel_refinement_convergence_receipt"] is False
    assert claims["B_A_kernel_refinement_patch_count_count"] == 2
    assert claims["B_A_kernel_refinement_key_pair_row_count"] == 2
    assert claims["B_A_kernel_refinement_key_pair_stable_fraction"] == 0.5
    assert claims["B_A_kernel_refinement_blocker_count"] == 2
    assert claims["screen_power_simulator_primordial_ready"] is False
    assert claims["maxent_green_source_receipt"] is True
    assert claims["selector_elimination_theorem_side_receipt"] is True
    assert claims["cmb_anomaly_parity_asymmetry_proxy"] is True
    assert claims["neutrino_measurement_comparable"] is True
    assert claims["neutrino_finite_lattice_derived"] is False
    assert claims["h0s8_measurement_comparable"] is True
    assert claims["compressed_likelihood_reference"] is True
    assert claims["neutral_profile_audit_written"] is True
    assert claims["neutral_profile_strict_3d_ready_count"] == 1
    assert claims["control_residualized_rank3_refinement_candidate"] is True
    assert claims["control_residualized_rank3_independent_selector_all"] is False
    assert claims["control_residualized_rank3_dimension_drift"] == 0.015
    assert claims["strict_neutral_bulk_refinement_receipt"] is False
    assert claims["control_residualized_rank3_physical_claim"] is False
    assert claims["neutral_3d_bulk_audit_written"] is True
    assert claims["neutral_3d_bulk_audit_ready"] is False
    assert claims["neutral_3d_bulk_audit_directional_strict_ready_total"] == 0
    assert claims["neutral_3d_bulk_audit_control_quotient_candidate_count"] == 1
    assert claims["neutral_independent_rank_selector_audit_written"] is True
    assert claims["neutral_independent_rank3_selector_receipt"] is False
    assert claims["neutral_independent_rank_selector_run_count"] == 2
    assert claims["neutral_independent_rank_selector_control_rank3_count"] == 0
    assert claims["neutral_independent_rank_selector_control_candidate_count"] == 2
    assert claims["neutral_independent_rank_selector_control_median_effective_rank"] == 126.0
    assert claims["neutral_independent_rank_selector_control_median_rank3_ev"] == 0.05
    assert "oph_screen_power_primordial_table.csv" in report["files"]
    assert "finite_repair_transition_matrix_report.json" in report["files"]
    assert "finite_repair_transition_rows.csv" in report["files"]
    assert "scalar_repair_semigroup_report.json" in report["files"]
    assert "oph_boltzmann_finite_repair_clock_rows.csv" in report["files"]
    assert "finite_repair_clock_cmb_camb_report.json" in report["files"]
    assert "finite_repair_clock_cmb_tt_bins.csv" in report["files"]
    assert "finite_repair_clock_cmb_tt_curves.csv" in report["files"]
    assert "camb_lcdm_baseline_report.json" in report["files"]
    assert "camb_lcdm_baseline_report.md" in report["files"]
    assert "camb_lcdm_tt_bins.csv" in report["files"]
    assert "physical_cmb_input_report.json" in report["files"]
    assert "physical_cmb_input_report.md" in report["files"]
    assert "physical_cmb_promotion_audit_report.json" in report["files"]
    assert "physical_cmb_promotion_audit_report.md" in report["files"]
    assert "physical_cmb_input_contract.json" in report["files"]
    assert "physical_cmb_input_validation.json" in report["files"]
    assert "physical_cmb_B_A_k_a.csv" in report["files"]
    assert "physical_cmb_Gamma_rec_k_a.csv" in report["files"]
    assert "physical_cmb_rho_A_a.csv" in report["files"]
    assert "b_a_parent_report.json" in report["files"]
    assert "b_a_parent_rows.csv" in report["files"]
    assert "b_a_parent_observer_view_rows.csv" in report["files"]
    assert "B_A_kernel_report.json" in report["files"]
    assert "B_A_kernel_report.md" in report["files"]
    assert "B_A_kernel_candidate.csv" in report["files"]
    assert "B_A_kernel_refinement_report.json" in report["files"]
    assert "B_A_kernel_refinement_report.md" in report["files"]
    assert "B_A_kernel_refinement_pairs.csv" in report["files"]
    assert "B_A_kernel_refinement_key_pairs.csv" in report["files"]
    assert "maxent_green_spectrum_report.json" in report["files"]
    assert "cmb_anomaly_report.json" in report["files"]
    assert "neutral_profile_audit_report.json" in report["files"]
    assert "prime_geometric_rank_refinement_report.json" in report["files"]
    assert "neutral_3d_bulk_audit_report.json" in report["files"]
    assert "neutral_3d_bulk_audit_report.md" in report["files"]
    assert "neutral_independent_rank_selector_audit_report.json" in report["files"]
    assert "neutral_independent_rank_selector_audit_report.md" in report["files"]


def test_export_measurement_pack_prefers_stronger_cmb_json_reports(tmp_path: Path) -> None:
    stale = tmp_path / "stale"
    fresh = tmp_path / "fresh"
    stale.mkdir()
    fresh.mkdir()
    _write_json(
        stale / "finite_collar_boltzmann_bundle_report.json",
        {
            "FINITE_COLLAR_BOLTZMANN_SOURCE_BUNDLE_RECEIPT": True,
            "readiness": {"checks": {"no_data_use_receipt": True}},
            "contract_source_summary": {"no_data_use_receipt": {"present": True}},
            "physical_cmb_input_validation": {
                "blockers": [
                    "eta_R_not_finite_derived",
                    "q_IR_not_finite_derived",
                    "B_A_k_a_missing_or_not_finite",
                ]
            },
        },
    )
    _write_json(
        fresh / "finite_collar_boltzmann_bundle_report.json",
        {
            "FINITE_COLLAR_BOLTZMANN_SOURCE_BUNDLE_RECEIPT": True,
            "readiness": {"checks": {"no_data_use_receipt": True}},
            "contract_source_summary": {
                "no_data_use_receipt": {"present": True},
                "scale_compressed_repair_report": {"present": True},
                "screen_capacity_closure_report": {"present": True},
            },
            "physical_cmb_input_validation": {"blockers": ["B_A_k_a_missing_or_not_finite"]},
        },
    )
    _write_json(stale / "no_data_use_receipt.json", {})
    _write_json(fresh / "no_data_use_receipt.json", {"NO_DATA_USE_RECEIPT": True})

    out = tmp_path / "pack"
    report = export_measurement_pack([stale, fresh], out)
    copied_bundle = json.loads((out / "finite_collar_boltzmann_bundle_report.json").read_text(encoding="utf-8"))
    copied_receipt = json.loads((out / "no_data_use_receipt.json").read_text(encoding="utf-8"))

    assert report["claims"]["finite_collar_boltzmann_source_bundle"] is True
    assert copied_bundle["physical_cmb_input_validation"]["blockers"] == ["B_A_k_a_missing_or_not_finite"]
    assert copied_receipt["NO_DATA_USE_RECEIPT"] is True


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")
