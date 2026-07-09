from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.measurement_pack import export_measurement_pack
from oph_fpe.physics_problem_outputs import write_physics_problem_outputs_report


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


def test_export_measurement_pack_copies_physics_problem_outputs(tmp_path: Path) -> None:
    run = tmp_path / "run"
    out = tmp_path / "pack"
    write_physics_problem_outputs_report(run)

    report = export_measurement_pack([run], out)
    claims = report["claims"]

    assert (out / "physics_problem_outputs_report.json").exists()
    assert (out / "physics_problem_outputs_report.md").exists()
    assert "physics_problem_outputs_report.json" in report["files"]
    assert "physics_problem_outputs_report.md" in report["files"]
    assert claims["physics_problem_outputs_written"] is True
    assert claims["physics_problem_outputs_source_document_count"] == 12
    assert claims["physics_problem_outputs_output_count"] == 12
    assert claims["physics_problem_outputs_all_notes_registered"] is True
    assert claims["physics_problem_outputs_jwst_claim"] == "J0_DIAGNOSTIC_PROXY"
    assert claims["physics_problem_outputs_gamma_claim"] == "DIAGNOSTIC_GAMMA_MAP"
    assert claims["physics_problem_outputs_cmb_claim_tier"] == "UNSTARTED_OR_INVALIDATED"
    assert claims["physics_problem_outputs_e8_receipt_status"] == "pending_raw_bundle"
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "physics problem outputs written: True" in readme


def test_export_measurement_pack_copies_neutral_3d_and_physical_cmb_data(tmp_path: Path) -> None:
    run = tmp_path / "run"
    out = tmp_path / "pack"
    run.mkdir()
    _write_json(run / "neutral_3d_bulk_audit_report.json", {"strict_neutral_bulk_ready": False})
    (run / "neutral_3d_bulk_audit_report.md").write_text("# neutral audit\n", encoding="utf-8")
    _write_json(run / "strict_neutral_bulk_report.json", {"strict_neutral_bulk": False})
    _write_json(
        run / "strict_neutral_object_bulk_report.json",
        {"strict_neutral_object_bulk": True, "object_count": 18},
    )
    _write_json(
        run / "strict_neutral_bulk_frontier_report.json",
        {
            "strict_neutral_bulk_ready": False,
            "gate_gap_rows": [],
            "overlap_native_negative_control_receipt_all": True,
        },
    )
    (run / "strict_neutral_bulk_frontier_report.md").write_text("# neutral frontier\n", encoding="utf-8")
    _write_json(
        run / "physical_cmb_input_report.json",
        {
            "PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT": True,
            "input_status": {
                "B_A_k_a": {"row_count": 2},
                "rho_A_a": {"row_count": 2},
            },
        },
    )
    (run / "physical_cmb_input_report.md").write_text("# physical CMB input\n", encoding="utf-8")
    _write_json(run / "physical_cmb_promotion_audit_report.json", {"physical_cmb_promotion_ready": False})
    (run / "physical_cmb_promotion_audit_report.md").write_text("# CMB promotion\n", encoding="utf-8")
    _write_json(
        run / "physical_cmb_output_comparison_report.json",
        {
            "PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT": True,
            "USABLE_PHYSICAL_CMB_DATA_RECEIPT": True,
            "measurement_comparable_model_count": 1,
            "oph_diagnostic_model_count": 1,
            "best_oph_diagnostic_model": {
                "model_id": "finite_repair_clock_plus_selector_ir",
                "amplitude_fit_chi2_per_bin": 1.5,
            },
            "best_oph_residual_summary": {
                "available": True,
                "bin_count": 2,
                "rms_sigma_residual": 0.5,
            },
            "best_oph_peak_feature_summary": {"peak_count": 1},
        },
    )
    (run / "physical_cmb_output_comparison_report.md").write_text("# CMB output\n", encoding="utf-8")
    _write_json(
        run / "physical_cmb_frontier_report.json",
        {"physical_cmb_prediction_ready": False, "gate_rows": [], "gate_gap_rows": []},
    )
    (run / "physical_cmb_frontier_report.md").write_text("# CMB frontier\n", encoding="utf-8")
    (run / "physical_cmb_output_comparison_rows.csv").write_text("ell,model\n2,x\n", encoding="utf-8")
    (run / "physical_cmb_best_oph_residuals.csv").write_text("ell,residual\n2,0.1\n", encoding="utf-8")
    (run / "physical_cmb_peak_features.csv").write_text("ell,height\n200,1.0\n", encoding="utf-8")
    (run / "B_A_k_a.csv").write_text("k,value\n1,0.1\n", encoding="utf-8")
    (run / "Gamma_rec_k_a.csv").write_text("k,value\n1,0.2\n", encoding="utf-8")
    (run / "rho_A_a.csv").write_text("k,value\n1,0.3\n", encoding="utf-8")

    report = export_measurement_pack([run], out)
    claims = report["claims"]

    assert claims["neutral_3d_bulk_data_bundle_written"] is True
    assert claims["strict_neutral_record_report_written"] is True
    assert claims["strict_neutral_object_report_written"] is True
    assert claims["strict_neutral_object_bulk"] is True
    assert claims["strict_neutral_object_count"] == 18
    assert claims["physical_cmb_data_bundle_written"] is True
    assert claims["physical_cmb_reports_written"] is True
    assert claims["physical_cmb_output_tables_written"] is True
    assert claims["physical_cmb_source_arrays_written"] is True
    assert claims["physical_cmb_output_rows_written"] is True
    assert claims["physical_cmb_best_residuals_written"] is True
    assert claims["physical_cmb_peak_features_written"] is True
    for name in (
        "neutral_3d_bulk_audit_report.json",
        "strict_neutral_bulk_report.json",
        "strict_neutral_object_bulk_report.json",
        "strict_neutral_bulk_frontier_report.json",
        "physical_cmb_output_comparison_rows.csv",
        "physical_cmb_best_oph_residuals.csv",
        "physical_cmb_peak_features.csv",
        "physical_cmb_B_A_k_a.csv",
        "physical_cmb_Gamma_rec_k_a.csv",
        "physical_cmb_rho_A_a.csv",
    ):
        assert name in report["files"]
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "neutral 3D bulk data bundle written: True" in readme
    assert "physical CMB data bundle written: True" in readme


def test_export_measurement_pack_copies_borel_weil_higgs_receipt(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "borel_weil_higgs_carrier_report.json").write_text(
        json.dumps(
            {
                "BOREL_WEIL_HIGGS_CARRIER_RECEIPT": True,
                "physical_claim": False,
                "checks": {"forbidden_quantitative_promotions_absent": True},
                "promoted_forbidden_claims": [],
            }
        ),
        encoding="utf-8",
    )
    (run / "borel_weil_higgs_carrier_report.md").write_text("# Higgs carrier\n", encoding="utf-8")

    out = tmp_path / "pack"
    report = export_measurement_pack([run], out)

    assert report["claims"]["borel_weil_higgs_carrier_written"] is True
    assert report["claims"]["borel_weil_higgs_carrier_receipt"] is True
    assert report["claims"]["borel_weil_higgs_physical_claim"] is False
    assert report["claims"]["borel_weil_higgs_forbidden_promotions_absent"] is True
    assert "borel_weil_higgs_carrier_report.json" in report["files"]
    assert "borel_weil_higgs_carrier_report.md" in report["files"]


def test_export_measurement_pack_copies_compact_transient_audit(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "compact_transient_audit_report.json").write_text(
        json.dumps(
            {
                "claim": "CR2_CONDITIONAL_PHENOMENOLOGY",
                "first_blocked_gate": "CONTROLS",
                "promotion_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    (run / "compact_transient_audit_report.md").write_text("# Compact transient\n", encoding="utf-8")

    out = tmp_path / "pack"
    report = export_measurement_pack([run], out)

    assert report["claims"]["compact_transient_audit_written"] is True
    assert report["claims"]["compact_transient_claim"] == "CR2_CONDITIONAL_PHENOMENOLOGY"
    assert report["claims"]["compact_transient_first_blocked_gate"] == "CONTROLS"
    assert report["claims"]["compact_transient_promotion_allowed"] is False
    assert "compact_transient_audit_report.json" in report["files"]
    assert "compact_transient_audit_report.md" in report["files"]


def test_export_measurement_pack_copies_uhe_coefficient_emission(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "uhe_coefficient_emission_report.json").write_text(
        json.dumps(
            {
                "claim_tier": "SOURCE_ONLY",
                "strongest_allowed_claim": "SOURCE_ONLY_COEFFICIENT_EMITTED",
                "readiness_gates": {
                    "NO_UHE_DATA_USE": True,
                    "COMMON_SOURCE_LOCK": True,
                    "COEFFICIENT_SOLVE_CONVERGED": True,
                },
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )
    (run / "uhe_coefficient_emission_report.md").write_text("# UHE coefficient\n", encoding="utf-8")

    out = tmp_path / "pack"
    report = export_measurement_pack([run], out)

    assert report["claims"]["uhe_coefficient_emission_written"] is True
    assert report["claims"]["uhe_coefficient_claim_tier"] == "SOURCE_ONLY"
    assert report["claims"]["uhe_coefficient_strongest_allowed_claim"] == "SOURCE_ONLY_COEFFICIENT_EMITTED"
    assert report["claims"]["uhe_coefficient_no_data_use_receipt"] is True
    assert report["claims"]["uhe_coefficient_common_source_lock"] is True
    assert report["claims"]["uhe_coefficient_solve_converged"] is True
    assert "uhe_coefficient_emission_report.json" in report["files"]
    assert "uhe_coefficient_emission_report.md" in report["files"]


def test_export_measurement_pack_copies_bulk_and_comparable_receipts(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "emergence_status_report.json").write_text(
        json.dumps(
            {
                "BW_KMS_BRANCH_REPLAY_RECEIPT": True,
                "PAPER_THEOREM_3D_BULK_CHART_RECEIPT": True,
                "H3_RESPONSE_CANDIDATE_RECEIPT": True,
            }
        ),
        encoding="utf-8",
    )
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
    (run / "paper_3d_bulk_chart_report.json").write_text(
        json.dumps(
            {
                "paper_theorem_3d_bulk_chart_receipt": True,
                "bw_2pi_cap_flow_receipt": True,
            }
        ),
        encoding="utf-8",
    )
    (run / "conformal_h3_spatial_chart_report.json").write_text(
        json.dumps({"conformal_h3_spatial_chart_receipt": True}),
        encoding="utf-8",
    )
    (run / "transition_selection_report.json").write_text(
        json.dumps(
            {
                "primary_source": "kms_collar_transport_response",
                "two_pi_selected": True,
                "response_degenerate": False,
            }
        ),
        encoding="utf-8",
    )
    (run / "observer_modular_experience_report.json").write_text(
        json.dumps(
            {
                "observer_modular_time_receipt": True,
                "OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT": True,
                "observer_facing_3p1d_h3_experience_receipt": True,
                "observer_facing_populated_h3_experience_receipt": False,
                "observer_h3_object_population_receipt": False,
                "observer_count": 16,
                "observer_relative_time_count": 2,
                "blockers": [],
                "populated_h3_experience_blockers": ["observer_h3_object_population_receipt"],
            }
        ),
        encoding="utf-8",
    )
    (run / "observer_views.jsonl").write_text(
        json.dumps({"observer_id": 1, "modular_depth_mean": 0.5, "observer_relative_times": [0.1, 0.2]})
        + "\n",
        encoding="utf-8",
    )
    timeline = run / "universe_timeline"
    timeline.mkdir()
    (timeline / "visualization_payload.json").write_text(
        json.dumps(
            {
                "schema": "oph_universe_timeline_visualization_payload_v1",
                "subjectiveObserverCameras": [],
                "comparableObservations": {"datasets": []},
            }
        ),
        encoding="utf-8",
    )
    (timeline / "oph_universe_timeline_viewer.html").write_text("<html>viewer</html>", encoding="utf-8")
    (timeline / "universe_timeline_summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (timeline / "VISUALIZATION_INSTRUCTIONS.md").write_text("# instructions\n", encoding="utf-8")
    (timeline / "WEB_CODING_AGENT_VISUALIZATION_BRIEF.md").write_text("# brief\n", encoding="utf-8")
    (timeline / "screen_full_16.bin").write_bytes(b"screen")
    (timeline / "observers_full_4.json").write_text("[]\n", encoding="utf-8")
    (timeline / "visualization_export_manifest.json").write_text(
        json.dumps(
            {
                "schema": "oph_universe_visualization_sidecars_v1",
                "files": {
                    "screen_full_bin": {
                        "path": str(timeline / "screen_full_16.bin"),
                        "row_count": 16,
                        "written": True,
                    },
                    "observers_full_json": {
                        "path": str(timeline / "observers_full_4.json"),
                        "row_count": 4,
                        "written": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "pack"
    report = export_measurement_pack([run], out)

    assert report["claims"]["chart_level_3p1"] is True
    assert report["claims"]["observer_modular_time_receipt"] is True
    assert report["claims"]["observer_facing_3p1d_h3_experience_receipt"] is True
    assert report["claims"]["observer_facing_populated_h3_experience_receipt"] is False
    assert report["claims"]["observer_modular_time_observer_count"] == 16
    assert report["claims"]["observer_modular_experience_blockers"] == []
    assert "observer_h3_object_population_receipt" in report["claims"]["observer_populated_h3_experience_blockers"]
    assert report["claims"]["observer_modular_experience_source_blockers"] == []
    assert report["claims"]["observer_populated_h3_experience_blockers"] == [
        "observer_h3_object_population_receipt"
    ]
    assert "bulk_proof_certificate_report.json" in report["files"]
    assert "bulk_proof_certificate_report.md" in report["files"]
    assert "comparable_data_snapshot.json" in report["files"]
    assert "comparable_data_snapshot.md" in report["files"]
    assert "paper_3d_bulk_chart_report.json" in report["files"]
    assert "conformal_h3_spatial_chart_report.json" in report["files"]
    assert "transition_selection_report.json" in report["files"]
    assert "observer_modular_experience_report.json" in report["files"]
    assert "observer_views.jsonl" in report["files"]
    assert "visualization_payload.json" in report["files"]
    assert "oph_universe_timeline_viewer.html" in report["files"]
    assert "WEB_CODING_AGENT_VISUALIZATION_BRIEF.md" in report["files"]
    assert "visualization_export_manifest.json" in report["files"]
    assert "screen_full_16.bin" in report["files"]
    assert "observers_full_4.json" in report["files"]
    manifest = json.loads((out / "visualization_export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["screen_full_bin"]["path"] == "screen_full_16.bin"
    assert manifest["files"]["observers_full_json"]["path"] == "observers_full_4.json"


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


def test_export_measurement_pack_prefers_overlap_aware_neutral_audits(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh"
    stale = tmp_path / "stale"
    fresh.mkdir()
    stale.mkdir()

    (fresh / "neutral_3d_bulk_audit_report.json").write_text(
        json.dumps(
            {
                "mode": "neutral_3d_bulk_audit_v0",
                "strict_neutral_bulk_ready": False,
                "control_residualized_rank3_refinement_candidate": True,
                "sweep_report_count": 4,
                "control_quotient_candidate_count": 4,
                "overlap_native_negative_control_report_count": 4,
                "overlap_native_negative_control_receipt_count": 4,
                "blockers": ["strict_neutral_bulk_refinement_receipt_false"],
            }
        ),
        encoding="utf-8",
    )
    (fresh / "neutral_3d_bulk_audit_report.md").write_text("fresh neutral\n", encoding="utf-8")
    (fresh / "neutral_independent_rank_selector_audit_report.json").write_text(
        json.dumps(
            {
                "mode": "neutral_independent_rank_selector_audit_v0",
                "NEUTRAL_INDEPENDENT_RANK3_SELECTOR_RECEIPT": False,
                "run_count": 4,
                "control_quotient_rank3_candidate_count": 4,
                "control_quotient_rank3_selector_count": 0,
                "control_quotient_median_effective_rank": 127.0,
                "blockers": ["control_quotient_lane_is_not_a_negative_control"],
            }
        ),
        encoding="utf-8",
    )
    (fresh / "neutral_independent_rank_selector_audit_report.md").write_text(
        "fresh selector\n",
        encoding="utf-8",
    )

    (stale / "neutral_3d_bulk_audit_report.json").write_text(
        json.dumps(
            {
                "mode": "neutral_3d_bulk_audit_v0",
                "strict_neutral_bulk_ready": False,
                "control_residualized_rank3_refinement_candidate": True,
                "sweep_report_count": 99,
                "control_quotient_candidate_count": 99,
                "blockers": ["strict_neutral_bulk_refinement_receipt_false"],
            }
        ),
        encoding="utf-8",
    )
    (stale / "neutral_3d_bulk_audit_report.md").write_text("stale neutral\n", encoding="utf-8")
    (stale / "neutral_independent_rank_selector_audit_report.json").write_text(
        json.dumps(
            {
                "mode": "neutral_independent_rank_selector_audit_v0",
                "NEUTRAL_INDEPENDENT_RANK3_SELECTOR_RECEIPT": False,
                "run_count": 99,
                "control_quotient_rank3_candidate_count": 99,
                "control_quotient_rank3_selector_count": 0,
                "control_quotient_median_effective_rank": 999.0,
                "blockers": ["control_quotient_lane_is_not_a_negative_control"],
            }
        ),
        encoding="utf-8",
    )
    (stale / "neutral_independent_rank_selector_audit_report.md").write_text(
        "stale selector\n",
        encoding="utf-8",
    )

    out = tmp_path / "pack"
    report = export_measurement_pack([fresh, stale], out)

    neutral = json.loads((out / "neutral_3d_bulk_audit_report.json").read_text(encoding="utf-8"))
    selector = json.loads(
        (out / "neutral_independent_rank_selector_audit_report.json").read_text(encoding="utf-8")
    )
    assert neutral["control_quotient_candidate_count"] == 4
    assert neutral["overlap_native_negative_control_receipt_count"] == 4
    assert (out / "neutral_3d_bulk_audit_report.md").read_text(encoding="utf-8") == "fresh neutral\n"
    assert selector["run_count"] == 4
    assert (out / "neutral_independent_rank_selector_audit_report.md").read_text(
        encoding="utf-8"
    ) == "fresh selector\n"
    assert report["claims"]["neutral_3d_bulk_audit_control_quotient_candidate_count"] == 4
    assert report["claims"]["neutral_independent_rank_selector_run_count"] == 4


def test_export_measurement_pack_copies_receipt_viewer(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "oph_receipt_viewer.html").write_text("<html>viewer</html>\n", encoding="utf-8")
    (run / "oph_realtime_viewer_summary.json").write_text(json.dumps({"viewer": True}), encoding="utf-8")
    (run / "object_h3_bulk_viewer.html").write_text("<html>object h3</html>\n", encoding="utf-8")
    (run / "object_h3_bulk_viewer_summary.json").write_text(
        json.dumps(
            {
                "object_count": 2,
                "observer_overlap_link_count": 7,
                "theorem_assisted_h3_bulk": True,
                "strict_neutral_bulk": False,
            }
        ),
        encoding="utf-8",
    )
    (run / "cmb_neutral_frontier_viewer.html").write_text("<html>cmb frontier</html>\n", encoding="utf-8")
    (run / "cmb_neutral_frontier_viewer_summary.json").write_text(
        json.dumps({"tt_bin_count": 11, "physical_cmb_prediction": False}),
        encoding="utf-8",
    )

    out = tmp_path / "pack"
    report = export_measurement_pack([run], out)

    assert "oph_receipt_viewer.html" in report["files"]
    assert "oph_realtime_viewer_summary.json" in report["files"]
    assert "object_h3_bulk_viewer.html" in report["files"]
    assert "object_h3_bulk_viewer_summary.json" in report["files"]
    assert report["claims"]["object_h3_bulk_viewer_written"] is True
    assert report["claims"]["object_h3_bulk_viewer_object_count"] == 2
    assert report["claims"]["object_h3_bulk_viewer_observer_overlap_link_count"] == 7
    assert "cmb_neutral_frontier_viewer.html" in report["files"]
    assert "cmb_neutral_frontier_viewer_summary.json" in report["files"]
    assert report["claims"]["cmb_neutral_frontier_viewer_written"] is True
    assert report["claims"]["cmb_neutral_frontier_viewer_tt_bin_count"] == 11
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
    (run / "screen_capacity_closure_report.md").write_text("# screen capacity", encoding="utf-8")
    (run / "capacity_readback_proxy_report.json").write_text(
        json.dumps(
            {
                "row_count": 1,
                "max_observer_count": 16,
                "max_terminal_normal_form_count_proxy": 8,
                "readiness_gates": {
                    "finite_regulator_rows_present": True,
                    "F_N_readback_map_implemented": False,
                    "N_CRC_fixed_point_solved_from_finite_simulator": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "capacity_readback_proxy_report.md").write_text("# capacity proxy", encoding="utf-8")
    (run / "capacity_readback_proxy_rows.csv").write_text(
        "path,patch_count,observer_count\nrun,4096,16\n",
        encoding="utf-8",
    )
    (run / "no_g_clock_bridge_report.json").write_text(
        json.dumps(
            {
                "NO_G_CLOCK_BRIDGE_RECEIPT": False,
                "source_predictive_G_SI": False,
                "dimensionful_G_SI_eligible": False,
                "clock_bridge": {"G_SI": 6.6743e-11},
                "readiness_gates": {"forbidden_dependency_path_count": 0},
            }
        ),
        encoding="utf-8",
    )
    (run / "no_g_clock_bridge_report.md").write_text("# no G", encoding="utf-8")
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
                    "P_source": {
                        "source": "OPH_pixel_branch_predeclared",
                        "source_is_theorem_side_constant": True,
                    },
                    "N_source": {
                        "source": "OPH_screen_capacity_observed_branch_readout",
                        "source_is_theorem_side_constant": True,
                    },
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
    (run / "physical_cmb_frontier_report.json").write_text(
        json.dumps(
            {
                "PHYSICAL_CMB_FRONTIER_REPORT": True,
                "physical_cmb_prediction_ready": False,
                "physical_cmb_output_comparison_receipt": True,
                "physical_cmb_prediction_receipt": False,
                "official_likelihood_ready": False,
                "gate_rows": [
                    {
                        "gate": "measurement_comparable_cmb_outputs",
                        "passed": True,
                        "detail": "2 OPH diagnostic models; 3 total comparable models",
                    },
                    {
                        "gate": "finite_theorem_A_zeta",
                        "passed": False,
                        "detail": "source=finite_certificate_report",
                    },
                    {
                        "gate": "finite_B_A_kernel",
                        "passed": False,
                        "detail": "rows=4",
                    },
                    {
                        "gate": "finite_rho_A",
                        "passed": False,
                        "detail": "rows=3",
                    },
                    {
                        "gate": "official_planck_likelihood_ready",
                        "passed": False,
                        "detail": "requires local official clik/Cobaya path",
                    },
                ],
                "gate_gap_rows": [
                    {
                        "gate": "finite_theorem_A_zeta",
                        "missing_receipt": "finite A_zeta source",
                        "current_evidence": "diagnostic proxy only",
                        "action_surface": "finite certificate stack",
                    },
                    {
                        "gate": "official_planck_likelihood_ready",
                        "missing_receipt": "official likelihood readiness",
                        "current_evidence": "local path missing",
                        "action_surface": "Planck clik/Cobaya config",
                    },
                ],
                "blockers": [
                    "A_zeta_not_finite_derived",
                    "B_A_k_a_missing_or_not_finite",
                    "official_likelihood_not_ready",
                ],
            }
        ),
        encoding="utf-8",
    )
    (run / "physical_cmb_frontier_report.md").write_text("# frontier", encoding="utf-8")
    (run / "physical_cmb_output_comparison_report.json").write_text(
        json.dumps(
            {
                "PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT": True,
                "PHYSICAL_CMB_PREDICTION_RECEIPT": False,
                "measurement_comparable_model_count": 3,
                "oph_diagnostic_model_count": 2,
                "best_oph_diagnostic_model": {
                    "model_id": "scale_compressed_ir_kernel",
                    "amplitude_fit_chi2_per_bin": 0.94,
                },
                "best_oph_residual_summary": {
                    "available": True,
                    "bin_count": 2,
                    "rms_sigma_residual": 1.5,
                    "max_abs_sigma_residual": 2.0,
                    "max_abs_sigma_ell": 80.0,
                },
                "best_oph_peak_feature_summary": {
                    "available": True,
                    "peak_count": 1,
                    "mean_abs_peak_ell_delta": 0.0,
                    "max_abs_peak_ell_delta": 0.0,
                    "mean_abs_peak_height_fractional_delta": 0.033,
                    "max_abs_peak_height_fractional_delta": 0.033,
                },
                "promotion_blockers": ["official_likelihood_not_ready"],
            }
        ),
        encoding="utf-8",
    )
    (run / "physical_cmb_output_comparison_report.md").write_text("# CMB output", encoding="utf-8")
    (run / "physical_cmb_output_comparison_rows.csv").write_text(
        "model_id,amplitude_fit_chi2_per_bin\nscale_compressed_ir_kernel,0.94\n",
        encoding="utf-8",
    )
    (run / "physical_cmb_best_oph_residuals.csv").write_text(
        "ell,residual_sigma\n50,-1\n80,2\n",
        encoding="utf-8",
    )
    (run / "physical_cmb_peak_features.csv").write_text(
        "model_id,model_role,peak_index,observed_peak_ell,model_peak_ell,ell_delta,fractional_D_ell_delta\n"
        "scale_compressed_ir_kernel,oph_diagnostic,1,80,80,0,0.033\n",
        encoding="utf-8",
    )
    (run / "official_planck_likelihood_readiness_report.json").write_text(
        json.dumps(
            {
                "mode": "official_planck_likelihood_readiness_v0",
                "official_likelihood_execution_ready": False,
                "official_planck_likelihood_data_paths_configured": False,
                "official_clik_api_available": False,
                "camb_available": True,
                "cobaya_available": False,
                "blockers": [
                    "official_clik_api_not_available",
                    "official_planck_likelihood_data_path_not_configured",
                    "cobaya_not_importable",
                ],
            }
        ),
        encoding="utf-8",
    )
    (run / "official_planck_likelihood_readiness_report.md").write_text("# official readiness", encoding="utf-8")
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
    (run / "overlap_native_neutral_control_report.json").write_text(
        json.dumps(
            {
                "OVERLAP_NATIVE_NEGATIVE_CONTROL_RECEIPT": True,
                "overlap_native_spatial_3d_candidate": False,
                "overlap_native_strict_h3_candidate": False,
            }
        ),
        encoding="utf-8",
    )
    (run / "overlap_native_neutral_control_report.md").write_text("# overlap", encoding="utf-8")
    (run / "overlap_native_graph_geometry_report.json").write_text(
        json.dumps(
            {
                "OVERLAP_NATIVE_GRAPH_GEOMETRY_RECEIPT": True,
                "overlap_graph_spatial_3d_candidate": False,
                "overlap_graph_strict_h3_candidate": False,
                "rank_selection": {"rank3_selector_receipt": False},
                "graph_summary": {"edge_count": 12, "component_count": 1},
            }
        ),
        encoding="utf-8",
    )
    (run / "overlap_native_graph_geometry_report.md").write_text("# overlap graph", encoding="utf-8")
    (run / "overlap_native_graph_geometry_sweep_report.json").write_text(
        json.dumps(
            {
                "OVERLAP_NATIVE_GRAPH_GEOMETRY_SWEEP_RECEIPT": True,
                "case_count": 3,
                "graph_geometry_receipt_count": 3,
                "spatial_3d_candidate_count": 1,
                "strict_h3_candidate_count": 0,
                "rank3_selector_count": 0,
                "rank_obstruction_summary": {
                    "dominant_largest_gap_rank": "2",
                    "nontrivial_rank3_selector_count": 2,
                    "dominant_nontrivial_largest_gap_rank": "3",
                    "max_nontrivial_rank3_cumulative_explained_variance": 0.52,
                    "median_nontrivial_effective_rank": 9.5,
                    "spatial_max_nontrivial_rank3_cumulative_explained_variance": 0.49,
                    "spatial_median_nontrivial_rank3_cumulative_explained_variance": 0.44,
                    "nontrivial_largest_gap_rank_counts": {"3": 2, "4": 1},
                },
                "gate_coincidence_summary": {
                    "available": True,
                    "case_count": 3,
                    "spatial_h3_geometry_count": 1,
                    "independent_rank3_selector_count": 0,
                    "nontrivial_rank3_selector_count": 2,
                    "spatial_h3_independent_rank3_selector_count": 0,
                    "spatial_h3_nontrivial_rank3_selector_count": 1,
                    "strict_h3_candidate_count": 0,
                },
                "best_case": {
                    "source_run_dir": "run-a",
                    "seed": 3,
                    "max_model_points": 32,
                    "k_neighbors": 8,
                    "median_dimension": 3.2,
                    "selected_model": "H3",
                },
                "closest_strict_rows": [
                    {
                        "source_run_dir": "run-a",
                        "seed": 3,
                        "max_model_points": 32,
                        "k_neighbors": 8,
                        "gate_score": 6,
                        "missing_strict_gates": ["independent_rank3_selector", "strict_h3_candidate"],
                        "median_dimension": 3.2,
                        "selected_model": "H3",
                    }
                ],
                "blockers": ["overlap_graph_sweep_no_strict_h3_candidate"],
            }
        ),
        encoding="utf-8",
    )
    (run / "overlap_native_graph_geometry_sweep_report.md").write_text("# overlap graph sweep", encoding="utf-8")
    (run / "overlap_native_graph_geometry_sweep_rows.csv").write_text(
        "source_run_dir,seed,max_model_points,k_neighbors,spatial_3d_candidate\nrun-a,3,32,8,true\n",
        encoding="utf-8",
    )
    (run / "overlap_residualized_graph_geometry_report.json").write_text(
        json.dumps(
            {
                "OVERLAP_RESIDUALIZED_GRAPH_GEOMETRY_RECEIPT": True,
                "overlap_residual_graph_spatial_3d_candidate": True,
                "overlap_residual_graph_strict_h3_candidate": False,
                "rank_selection": {"rank3_selector_receipt": False},
                "graph_summary": {"edge_count": 9, "component_count": 1},
            }
        ),
        encoding="utf-8",
    )
    (run / "overlap_residualized_graph_geometry_report.md").write_text(
        "# residual graph",
        encoding="utf-8",
    )
    (run / "overlap_residualized_graph_geometry_sweep_report.json").write_text(
        json.dumps(
            {
                "OVERLAP_RESIDUALIZED_GRAPH_GEOMETRY_SWEEP_RECEIPT": True,
                "case_count": 4,
                "residual_graph_receipt_count": 4,
                "spatial_3d_candidate_count": 2,
                "strict_h3_candidate_count": 0,
                "rank3_selector_count": 1,
                "rank_obstruction_summary": {
                        "dominant_largest_gap_rank": "2",
                        "dominant_nontrivial_largest_gap_rank": "3",
                        "raw_largest_gap_rank1_count": 3,
                        "nontrivial_rank3_selector_count": 2,
                        "max_rank3_cumulative_explained_variance": 0.57,
                        "max_nontrivial_rank3_cumulative_explained_variance": 0.54,
                        "median_effective_rank": 9.0,
                        "median_nontrivial_effective_rank": 8.8,
                        "nontrivial_largest_gap_rank_counts": {"3": 2, "4": 2},
                },
                "gate_coincidence_summary": {
                    "available": True,
                    "case_count": 4,
                    "spatial_h3_geometry_count": 2,
                    "independent_rank3_selector_count": 1,
                    "nontrivial_rank3_selector_count": 2,
                    "spatial_h3_independent_rank3_selector_count": 1,
                    "spatial_h3_nontrivial_rank3_selector_count": 0,
                    "strict_h3_candidate_count": 0,
                },
                "closest_strict_rows": [
                    {
                        "source_run_dir": "run-a",
                        "seed": 3,
                        "max_model_points": 32,
                        "k_neighbors": 8,
                        "remove_modes": 1,
                        "gate_score": 7,
                        "missing_strict_gates": ["strict_h3_candidate"],
                        "median_dimension": 3.1,
                        "selected_model": "H3",
                    }
                ],
                "blockers": ["overlap_residual_graph_sweep_no_strict_h3_candidate"],
            }
        ),
        encoding="utf-8",
    )
    (run / "overlap_residualized_graph_geometry_sweep_report.md").write_text(
        "# residual graph sweep",
        encoding="utf-8",
    )
    (run / "overlap_residualized_graph_geometry_sweep_rows.csv").write_text(
        "source_run_dir,seed,max_model_points,k_neighbors,remove_modes,spatial_3d_candidate\nrun-a,3,32,8,1,true\n",
        encoding="utf-8",
    )
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
    (run / "strict_neutral_bulk_frontier_report.json").write_text(
        json.dumps(
            {
                "STRICT_NEUTRAL_BULK_FRONTIER_REPORT": True,
                "strict_neutral_bulk": False,
                "strict_neutral_bulk_ready": False,
                "control_residualized_rank3_refinement_candidate": True,
                "overlap_native_negative_control_receipt_all": True,
                "overlap_native_graph_geometry_receipt_count": 2,
                "overlap_native_graph_spatial_3d_candidate_count": 1,
                "overlap_native_graph_strict_h3_candidate_count": 0,
                "overlap_native_graph_model_order_rank3_selector_count": 0,
                "overlap_native_graph_nontrivial_model_order_rank3_selector_count": 0,
                "overlap_residualized_graph_geometry_receipt_count": 4,
                "overlap_residualized_graph_spatial_3d_candidate_count": 2,
                "overlap_residualized_graph_strict_h3_candidate_count": 0,
                "overlap_residualized_graph_rank3_selector_count": 1,
                "overlap_residualized_graph_model_order_rank3_selector_count": 0,
                "overlap_residualized_graph_nontrivial_model_order_rank3_selector_count": 0,
                "neutral_independent_rank3_selector_receipt": False,
                "directional_strict_ready_total": 0,
                "gate_gap_rows": [
                    {
                        "gate": "independent_rank3_selector",
                        "missing_receipt": "independent rank-3 selector",
                        "current_evidence": "control quotient candidate only",
                        "action_surface": "neutral rank selector audit",
                    }
                ],
                "blockers": ["independent_svd_rank3_selector_not_stable_or_false"],
            }
        ),
        encoding="utf-8",
    )
    (run / "strict_neutral_bulk_frontier_report.md").write_text("# frontier", encoding="utf-8")

    out = tmp_path / "pack"
    report = export_measurement_pack([run], out)

    claims = report["claims"]
    assert claims["screen_capacity_observed_branch_available"] is True
    assert claims["screen_capacity_finite_fixed_point_solved"] is False
    assert claims["capacity_readback_proxy_written"] is True
    assert claims["capacity_readback_proxy_row_count"] == 1
    assert claims["capacity_readback_proxy_max_observer_count"] == 16
    assert claims["capacity_readback_proxy_fixed_point_solved"] is False
    assert claims["capacity_readback_proxy_F_N_implemented"] is False
    assert claims["no_g_clock_bridge_written"] is True
    assert claims["no_g_clock_bridge_receipt"] is False
    assert claims["no_g_clock_bridge_source_predictive_G_SI"] is False
    assert claims["no_g_clock_bridge_G_SI_checksum"] == 6.6743e-11
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
    assert claims["physical_cmb_input_P_source"] == "OPH_pixel_branch_predeclared"
    assert claims["physical_cmb_input_P_source_theorem_side"] is True
    assert claims["physical_cmb_input_N_source"] == "OPH_screen_capacity_observed_branch_readout"
    assert claims["physical_cmb_input_N_source_theorem_side"] is True
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
    assert claims["physical_cmb_frontier_written"] is True
    assert claims["physical_cmb_frontier_ready"] is False
    assert claims["physical_cmb_frontier_gate_count"] == 5
    assert claims["physical_cmb_frontier_gap_count"] == 2
    assert claims["physical_cmb_frontier_blocker_count"] == 3
    assert claims["physical_cmb_frontier_measurement_outputs"] is True
    assert claims["physical_cmb_frontier_finite_A_zeta"] is False
    assert claims["physical_cmb_frontier_finite_B_A"] is False
    assert claims["physical_cmb_frontier_finite_rho_A"] is False
    assert claims["physical_cmb_frontier_official_likelihood"] is False
    assert claims["official_planck_likelihood_readiness_written"] is True
    assert claims["official_planck_likelihood_execution_ready"] is False
    assert claims["official_planck_likelihood_data_paths_configured"] is False
    assert claims["official_planck_clik_api_available"] is False
    assert claims["official_planck_likelihood_blocker_count"] == 3
    assert claims["physical_cmb_output_comparison_written"] is True
    assert claims["physical_cmb_output_comparison_receipt"] is True
    assert claims["physical_cmb_output_usable_data_receipt"] is True
    assert claims["physical_cmb_output_prediction_receipt"] is False
    assert claims["physical_cmb_output_measurement_comparable_model_count"] == 3
    assert claims["physical_cmb_output_oph_diagnostic_model_count"] == 2
    assert claims["physical_cmb_output_best_oph_model"] == "scale_compressed_ir_kernel"
    assert claims["physical_cmb_output_best_oph_chi2_per_bin"] == 0.94
    assert claims["physical_cmb_output_best_oph_residual_bin_count"] == 2
    assert claims["physical_cmb_output_best_oph_rms_sigma_residual"] == 1.5
    assert claims["physical_cmb_output_best_oph_max_abs_sigma_residual"] == 2.0
    assert claims["physical_cmb_output_best_oph_max_abs_sigma_ell"] == 80.0
    assert claims["physical_cmb_output_best_oph_peak_count"] == 1
    assert claims["physical_cmb_output_best_oph_mean_abs_peak_ell_delta"] == 0.0
    assert claims["physical_cmb_output_best_oph_max_abs_peak_ell_delta"] == 0.0
    assert claims["physical_cmb_output_best_oph_mean_abs_peak_height_fractional_delta"] == 0.033
    assert claims["physical_cmb_output_best_oph_max_abs_peak_height_fractional_delta"] == 0.033
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
    assert claims["overlap_native_neutral_control_written"] is True
    assert claims["overlap_native_negative_control_receipt"] is True
    assert claims["overlap_native_spatial_3d_candidate"] is False
    assert claims["overlap_native_strict_h3_candidate"] is False
    assert claims["overlap_native_graph_geometry_written"] is True
    assert claims["overlap_native_graph_geometry_receipt"] is True
    assert claims["overlap_native_graph_spatial_3d_candidate"] is False
    assert claims["overlap_native_graph_strict_h3_candidate"] is False
    assert claims["overlap_native_graph_rank3_selector"] is False
    assert claims["overlap_native_graph_sweep_written"] is True
    assert claims["overlap_native_graph_sweep_receipt"] is True
    assert claims["overlap_native_graph_sweep_case_count"] == 3
    assert claims["overlap_native_graph_sweep_receipt_count"] == 3
    assert claims["overlap_native_graph_sweep_spatial_candidates"] == 1
    assert claims["overlap_native_graph_sweep_strict_h3_candidates"] == 0
    assert claims["overlap_native_graph_sweep_rank3_selectors"] == 0
    assert claims["overlap_native_graph_sweep_closest_strict_candidate_count"] == 1
    assert claims["overlap_native_graph_sweep_nontrivial_rank3_selectors"] == 2
    assert claims["overlap_native_graph_sweep_dominant_nontrivial_largest_gap_rank"] == "3"
    assert claims["overlap_native_graph_sweep_max_nontrivial_rank3_ev"] == 0.52
    assert claims["overlap_native_graph_sweep_median_nontrivial_effective_rank"] == 9.5
    assert claims["overlap_residualized_graph_geometry_written"] is True
    assert claims["overlap_residualized_graph_geometry_receipt"] is True
    assert claims["overlap_residualized_graph_spatial_3d_candidate"] is True
    assert claims["overlap_residualized_graph_strict_h3_candidate"] is False
    assert claims["overlap_residualized_graph_rank3_selector"] is False
    assert claims["overlap_residualized_graph_sweep_written"] is True
    assert claims["overlap_residualized_graph_sweep_receipt"] is True
    assert claims["overlap_residualized_graph_sweep_case_count"] == 4
    assert claims["overlap_residualized_graph_sweep_receipt_count"] == 4
    assert claims["overlap_residualized_graph_sweep_spatial_candidates"] == 2
    assert claims["overlap_residualized_graph_sweep_strict_h3_candidates"] == 0
    assert claims["overlap_residualized_graph_sweep_rank3_selectors"] == 1
    assert claims["overlap_residualized_graph_sweep_closest_strict_candidate_count"] == 1
    assert claims["overlap_residualized_graph_sweep_nontrivial_rank3_selectors"] == 2
    assert claims["overlap_residualized_graph_sweep_dominant_largest_gap_rank"] == "2"
    assert claims["overlap_residualized_graph_sweep_dominant_nontrivial_largest_gap_rank"] == "3"
    assert claims["overlap_residualized_graph_sweep_raw_rank1_cases"] == 3
    assert claims["overlap_residualized_graph_sweep_max_nontrivial_rank3_ev"] == 0.54
    assert claims["overlap_residualized_graph_sweep_median_nontrivial_effective_rank"] == 8.8
    assert claims["neutral_independent_rank_selector_audit_written"] is True
    assert claims["neutral_independent_rank3_selector_receipt"] is False
    assert claims["neutral_independent_rank_selector_run_count"] == 2
    assert claims["neutral_independent_rank_selector_control_rank3_count"] == 0
    assert claims["neutral_independent_rank_selector_control_candidate_count"] == 2
    assert claims["neutral_independent_rank_selector_control_median_effective_rank"] == 126.0
    assert claims["neutral_independent_rank_selector_control_median_rank3_ev"] == 0.05
    assert claims["strict_neutral_bulk_frontier_written"] is True
    assert claims["strict_neutral_bulk_frontier_ready"] is False
    assert claims["strict_neutral_bulk_frontier_gap_count"] == 1
    assert claims["strict_neutral_bulk_frontier_rank3_candidate"] is True
    assert claims["strict_neutral_bulk_frontier_overlap_controls"] is True
    assert claims["strict_neutral_bulk_frontier_overlap_graph_receipts"] == 2
    assert claims["strict_neutral_bulk_frontier_overlap_graph_spatial_candidates"] == 1
    assert claims["strict_neutral_bulk_frontier_overlap_graph_strict_h3_candidates"] == 0
    assert claims["strict_neutral_bulk_frontier_overlap_graph_model_order_rank3_selectors"] == 0
    assert claims["strict_neutral_bulk_frontier_overlap_graph_nontrivial_model_order_rank3_selectors"] == 0
    assert claims["strict_neutral_bulk_frontier_overlap_residual_graph_receipts"] == 4
    assert claims["strict_neutral_bulk_frontier_overlap_residual_graph_spatial_candidates"] == 2
    assert claims["strict_neutral_bulk_frontier_overlap_residual_graph_strict_h3_candidates"] == 0
    assert claims["strict_neutral_bulk_frontier_overlap_residual_graph_rank3_selectors"] == 1
    assert claims["strict_neutral_bulk_frontier_overlap_residual_graph_model_order_rank3_selectors"] == 0
    assert (
        claims[
            "strict_neutral_bulk_frontier_overlap_residual_graph_nontrivial_model_order_rank3_selectors"
        ]
        == 0
    )
    assert claims["strict_neutral_bulk_frontier_independent_selector"] is False
    assert claims["overlap_native_graph_sweep_spatial_h3_rank3_coincidences"] == 0
    assert claims["overlap_native_graph_sweep_spatial_h3_nontrivial_rank3_coincidences"] == 1
    assert claims["overlap_residualized_graph_sweep_spatial_h3_rank3_coincidences"] == 1
    assert claims["overlap_residualized_graph_sweep_spatial_h3_nontrivial_rank3_coincidences"] == 0
    assert "oph_screen_power_primordial_table.csv" in report["files"]
    assert "screen_capacity_closure_report.json" in report["files"]
    assert "screen_capacity_closure_report.md" in report["files"]
    assert "capacity_readback_proxy_report.json" in report["files"]
    assert "capacity_readback_proxy_report.md" in report["files"]
    assert "capacity_readback_proxy_rows.csv" in report["files"]
    assert "no_g_clock_bridge_report.json" in report["files"]
    assert "no_g_clock_bridge_report.md" in report["files"]
    assert "overlap_native_neutral_control_report.json" in report["files"]
    assert "overlap_native_neutral_control_report.md" in report["files"]
    assert "overlap_native_graph_geometry_report.json" in report["files"]
    assert "overlap_native_graph_geometry_report.md" in report["files"]
    assert "overlap_native_graph_geometry_sweep_report.json" in report["files"]
    assert "overlap_native_graph_geometry_sweep_report.md" in report["files"]
    assert "overlap_native_graph_geometry_sweep_rows.csv" in report["files"]
    assert "overlap_residualized_graph_geometry_report.json" in report["files"]
    assert "overlap_residualized_graph_geometry_report.md" in report["files"]
    assert "overlap_residualized_graph_geometry_sweep_report.json" in report["files"]
    assert "overlap_residualized_graph_geometry_sweep_report.md" in report["files"]
    assert "overlap_residualized_graph_geometry_sweep_rows.csv" in report["files"]
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
    assert "official_planck_likelihood_readiness_report.json" in report["files"]
    assert "official_planck_likelihood_readiness_report.md" in report["files"]
    assert "physical_cmb_promotion_audit_report.json" in report["files"]
    assert "physical_cmb_promotion_audit_report.md" in report["files"]
    assert "physical_cmb_frontier_report.json" in report["files"]
    assert "physical_cmb_frontier_report.md" in report["files"]
    assert "physical_cmb_output_comparison_report.json" in report["files"]
    assert "physical_cmb_output_comparison_report.md" in report["files"]
    assert "physical_cmb_output_comparison_rows.csv" in report["files"]
    assert "physical_cmb_best_oph_residuals.csv" in report["files"]
    assert "physical_cmb_peak_features.csv" in report["files"]
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
    assert "strict_neutral_bulk_frontier_report.json" in report["files"]
    assert "strict_neutral_bulk_frontier_report.md" in report["files"]


def test_export_measurement_pack_prefers_residualized_strict_frontier(tmp_path: Path) -> None:
    stale = tmp_path / "stale"
    fresh = tmp_path / "fresh"
    stale.mkdir()
    fresh.mkdir()

    _write_json(
        stale / "strict_neutral_bulk_frontier_report.json",
        {
            "STRICT_NEUTRAL_BULK_FRONTIER_REPORT": True,
            "strict_neutral_bulk_ready": False,
            "control_residualized_rank3_refinement_candidate": True,
            "overlap_native_negative_control_receipt_all": True,
            "neutral_independent_rank3_selector_receipt": False,
            "directional_strict_ready_total": 12,
            "blockers": ["overlap_graph_strict_h3_candidate_false"],
        },
    )
    (stale / "strict_neutral_bulk_frontier_report.md").write_text("stale frontier\n", encoding="utf-8")
    _write_json(
        fresh / "strict_neutral_bulk_frontier_report.json",
        {
            "STRICT_NEUTRAL_BULK_FRONTIER_REPORT": True,
            "strict_neutral_bulk_ready": False,
            "control_residualized_rank3_refinement_candidate": True,
            "overlap_native_negative_control_receipt_all": True,
            "overlap_residualized_graph_geometry_report_count": 3600,
            "overlap_residualized_graph_geometry_receipt_count": 3593,
            "overlap_residualized_graph_spatial_3d_candidate_count": 45,
            "overlap_residualized_graph_strict_h3_candidate_count": 0,
            "neutral_independent_rank3_selector_receipt": False,
            "directional_strict_ready_total": 0,
            "gate_rows": [
                {"gate": "overlap_residualized_graph_geometry", "passed": False},
                {"gate": "overlap_residualized_graph_strict_h3", "passed": False},
            ],
            "blockers": [
                "overlap_graph_strict_h3_candidate_false",
                "overlap_residual_graph_strict_h3_candidate_false",
            ],
        },
    )
    (fresh / "strict_neutral_bulk_frontier_report.md").write_text("fresh frontier\n", encoding="utf-8")

    out = tmp_path / "pack"
    report = export_measurement_pack([stale, fresh], out)
    copied = json.loads((out / "strict_neutral_bulk_frontier_report.json").read_text(encoding="utf-8"))

    assert copied["overlap_residualized_graph_geometry_report_count"] == 3600
    assert copied["overlap_residualized_graph_spatial_3d_candidate_count"] == 45
    assert (out / "strict_neutral_bulk_frontier_report.md").read_text(encoding="utf-8") == "fresh frontier\n"
    assert report["claims"]["strict_neutral_bulk_frontier_ready"] is False


def test_export_measurement_pack_prefers_model_order_strict_frontier(tmp_path: Path) -> None:
    count_only = tmp_path / "count_only"
    model_order = tmp_path / "model_order"
    count_only.mkdir()
    model_order.mkdir()

    _write_json(
        count_only / "strict_neutral_bulk_frontier_report.json",
        {
            "STRICT_NEUTRAL_BULK_FRONTIER_REPORT": True,
            "strict_neutral_bulk_ready": False,
            "overlap_residualized_graph_geometry_report_count": 5000,
            "overlap_residualized_graph_geometry_receipt_count": 5000,
            "overlap_residualized_graph_spatial_3d_candidate_count": 120,
            "gate_rows": [{"gate": "overlap_residualized_graph_geometry", "passed": True}],
            "gate_gap_rows": [{"gate": "overlap_residualized_graph_strict_h3", "passed": False}],
        },
    )
    (count_only / "strict_neutral_bulk_frontier_report.md").write_text(
        "count-only frontier\n",
        encoding="utf-8",
    )
    _write_json(
        model_order / "strict_neutral_bulk_frontier_report.json",
        {
            "STRICT_NEUTRAL_BULK_FRONTIER_REPORT": True,
            "strict_neutral_bulk_ready": False,
            "overlap_residualized_graph_geometry_report_count": 16,
            "overlap_residualized_graph_geometry_receipt_count": 16,
            "overlap_residualized_graph_spatial_3d_candidate_count": 1,
            "overlap_residualized_graph_model_order_rank3_selector_count": 0,
            "overlap_residualized_graph_nontrivial_model_order_rank3_selector_count": 0,
            "gate_rows": [{"gate": "overlap_residualized_graph_geometry", "passed": True}],
            "gate_gap_rows": [{"gate": "overlap_residualized_graph_strict_h3", "passed": False}],
        },
    )
    (model_order / "strict_neutral_bulk_frontier_report.md").write_text(
        "model-order frontier\n",
        encoding="utf-8",
    )

    out = tmp_path / "pack"
    report = export_measurement_pack([count_only, model_order], out)
    copied = json.loads((out / "strict_neutral_bulk_frontier_report.json").read_text(encoding="utf-8"))

    assert copied["overlap_residualized_graph_geometry_report_count"] == 16
    assert copied["overlap_residualized_graph_model_order_rank3_selector_count"] == 0
    assert (out / "strict_neutral_bulk_frontier_report.md").read_text(encoding="utf-8") == (
        "model-order frontier\n"
    )
    assert report["claims"]["strict_neutral_bulk_frontier_overlap_residual_graph_receipts"] == 16


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


def test_export_measurement_pack_copies_cmb_promotion_ledger(tmp_path: Path) -> None:
    run = tmp_path / "run"
    out = tmp_path / "pack"
    run.mkdir()
    _write_json(
        run / "cmb_promotion_ledger_report.json",
        {
            "current_claim_tier": "SPECTRUM_DIAGNOSTIC",
            "fail_closed_state": "NUMERICALLY_INCONCLUSIVE",
            "claim_tier": "conditional_physical",
            "geometry_origin": "IMPORTED_FLRW",
            "conditional_physical_scale_bridge_ready": True,
            "oph_native_geometry_ready": False,
            "readiness_gates": {
                "NO_DATA_USE_RECEIPT": True,
                "SOURCE_ONLY_FINITE_ARTIFACT_RECEIPT": False,
                "GEOMETRIC_SCREEN_SCALAR_RECEIPT": False,
                "FROZEN_LIKELIHOOD_RECEIPT": False,
                "PHYSICAL_CMB_PREDICTION_RECEIPT": False,
            },
            "blockers": ["source_only_finite_artifact_receipt_missing"],
        },
    )
    (run / "cmb_promotion_ledger_report.md").write_text("# CMB Promotion Ledger\n", encoding="utf-8")

    report = export_measurement_pack([run], out)

    assert (out / "cmb_promotion_ledger_report.json").exists()
    assert (out / "cmb_promotion_ledger_report.md").exists()
    assert report["claims"]["cmb_promotion_ledger_written"] is True
    assert report["claims"]["cmb_promotion_current_claim_tier"] == "SPECTRUM_DIAGNOSTIC"
    assert report["claims"]["cmb_promotion_conditional_scale_bridge_ready"] is True
    assert report["claims"]["cmb_promotion_physical_prediction_receipt"] is False


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")
