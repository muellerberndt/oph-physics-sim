import json
from pathlib import Path

import numpy as np

from oph_fpe.viz import write_universe_timeline_bundle
from oph_fpe.viz.universe_timeline_viewer import _small_universe_payload


def test_small_universe_payload_uses_theorem_core_receipt_fallback(tmp_path: Path):
    (tmp_path / "theorem_core_receipts.json").write_text(
        json.dumps(
            {
                "receipt": True,
                "FINITE_CONSENSUS_THEOREM_RECEIPT": True,
                "finite_consensus_theorem_receipt": True,
                "claim_boundary": "large run theorem core",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "finite_consensus_replay_report.json").write_text(
        json.dumps({"receipt": True}),
        encoding="utf-8",
    )

    payload = _small_universe_payload(tmp_path)

    assert payload["receipts"]["FINITE_CONSENSUS_THEOREM_RECEIPT"] is True
    assert payload["receipts"]["bundle_receipt"] is True
    assert payload["claimBoundary"] == "large run theorem core"


def test_universe_timeline_viewer_writes_payload_html_and_briefs(tmp_path: Path):
    small = tmp_path / "small"
    small.mkdir()
    (small / "all_states.jsonl").write_text(
        json.dumps(
            {
                "branch": "exact_consensus",
                "state_id": "000",
                "state": [0, 0, 0],
                "phi": 2,
                "enabled_repair_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (small / "repair_transition_table.jsonl").write_text(
        json.dumps(
            {
                "branch": "exact_consensus",
                "state_id": "000",
                "next_state_id": "010",
                "next_state": [0, 1, 0],
                "phi_before": 2,
                "phi_after": 0,
                "delta_phi": -2,
                "node": 1,
                "parent": 0,
                "strict_descent": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cycles = {
        "exact_consensus": [{"cycle": [0, 1, 2], "edges": [[0, 1], [1, 2], [0, 2]], "holonomy_z2": 0}],
        "frustrated_control": [{"cycle": [0, 1, 2], "edges": [[0, 1], [1, 2], [0, 2]], "holonomy_z2": 1}],
    }
    (small / "cycle_holonomy.json").write_text(json.dumps(cycles), encoding="utf-8")
    (small / "exact_consensus_receipt.json").write_text(
        json.dumps(
            {
                "FINITE_CONSENSUS_THEOREM_RECEIPT": True,
                "nonzero_holonomy_cycle_count": 0,
                "strict_descent_violation_count": 0,
                "schedule_confluence_violation_count": 0,
                "unique_terminal_normal_form_count": 1,
                "terminal_phi": 0,
                "terminal_normal_form": [0, 1, 0],
            }
        ),
        encoding="utf-8",
    )
    (small / "frustrated_control_receipt.json").write_text(
        json.dumps({"HOLONOMY_OBSTRUCTION_RECEIPT": True, "nonzero_holonomy_cycle_count": 1}),
        encoding="utf-8",
    )
    (small / "small_oph_universe_evidence.json").write_text(
        json.dumps({"bundle_receipt": True}),
        encoding="utf-8",
    )

    observer = tmp_path / "observer"
    observer.mkdir()
    np.savez_compressed(
        observer / "freezeout_fields.npz",
        points=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        record_signature=np.array([0.0, 1.0], dtype=np.float32),
    )
    (observer / "observer_modular_experience_report.json").write_text(
        json.dumps(
            {
                "observer_modular_time_receipt": True,
                "observer_facing_3p1d_h3_experience_receipt": False,
                "observer_relative_time_grid": [0.1, 0.2],
                "claim_boundary": "observer-local modular time",
            }
        ),
        encoding="utf-8",
    )
    (observer / "emergence_status_report.json").write_text(
        json.dumps(
            {
                "support_visible_lorentz_3p1_kinematics_receipt": True,
                "conformal_h3_spatial_chart_receipt": True,
                "bulk_3d_established": False,
            }
        ),
        encoding="utf-8",
    )
    (observer / "observer_views.jsonl").write_text(
        json.dumps(
            {
                "observer_id": 0,
                "axis": [1.0, 0.0, 0.0],
                "support_patch_count": 2,
                "visible_signature_entropy": 0.5,
                "modular_depth_mean": 0.4,
                "dominant_record_signature": 7,
                "dominant_object_packet": 3,
                "object_packet_histogram": {"3": 0.75, "4": 0.25},
                "record_signature_histogram": {"7": 1.0},
                "transition_history_descriptor": {
                    "steps": [
                        {"record_family": 1, "checkpoint_class": 0, "s3_sector_class": 2},
                        {"record_family": 1, "checkpoint_class": 1, "s3_sector_class": 2},
                    ]
                },
                "visible_readout_hash": "abc",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (observer / "mismatch_trace.csv").write_text(
        "cycle,phase,phi,committed_fraction,modular_depth_mean,mismatch_edges\n"
        "0,exploration,2,0.0,0.3,2\n"
        "1,exploration,0,1.0,0.4,0\n",
        encoding="utf-8",
    )
    (observer / "array_holonomy_report.json").write_text(
        json.dumps({"clusters": [{"cluster_id": "c0", "centroid": [0.0, 1.0, 0.0], "class": "threecycle"}]}),
        encoding="utf-8",
    )
    (observer / "defect_timeline_report.json").write_text(
        json.dumps(
            {
                "persistent_worldline_count": 1,
                "snapshots": [
                    {"cycle": 0, "clusters": [{"centroid": [0.0, 1.0, 0.0], "class": "threecycle"}]}
                ],
            }
        ),
        encoding="utf-8",
    )

    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "h3_objects.csv").write_text(
        "object_id,h3_spatial_point,observer_count,h3_compactness,h3_compactness_normalized,support_size\n"
        'obj0,"[0.1, 0.2, 0.3]",4,0.2,0.4,12\n',
        encoding="utf-8",
    )
    (pack / "object_h3_bulk_viewer_summary.json").write_text(
        json.dumps(
            {
                "object_count": 1,
                "theorem_assisted_h3_bulk": True,
                "observer_chart_bulk_population_receipt": True,
                "strict_neutral_bulk": False,
            }
        ),
        encoding="utf-8",
    )
    (pack / "observer_consensus_bulk_readout_report.json").write_text(
        json.dumps(
            {
                "observer_like_self_reading_system_receipt": True,
                "observer_modular_time_receipt": True,
                "observer_facing_3p1d_h3_experience_receipt": True,
                "observer_facing_populated_h3_experience_receipt": False,
                "observer_h3_object_population_receipt": False,
                "theorem_assisted_consensus_3d_bulk_readout_receipt": True,
                "strict_neutral_third_person_bulk_receipt": False,
                "physical_cmb_output_comparison_receipt": True,
                "physical_cmb_prediction_receipt": False,
            }
        ),
        encoding="utf-8",
    )
    (pack / "physical_cmb_output_comparison_report.json").write_text(
        json.dumps(
            {
                "PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT": True,
                "USABLE_PHYSICAL_CMB_DATA_RECEIPT": True,
                "PHYSICAL_CMB_PREDICTION_RECEIPT": False,
                "best_oph_residual_summary": {"rms_sigma_residual": 0.9},
                "best_oph_residual_rows": [{"ell": 2, "observed": 1000, "model": 990, "residual_sigma": -0.1}],
            }
        ),
        encoding="utf-8",
    )
    (pack / "cl_comparison_report.json").write_text(
        json.dumps(
            {
                "ell_max": 4,
                "point_count": 128,
                "receipt_name": "SCREEN_PROXY_CMB_RECEIPT",
                "cosmo_proxy_receipt": {"receipt": True},
                "fields": {
                    "record_signature": {
                        "spectrum": [
                            {"ell": 2, "C_ell": 0.1, "D_ell": 1.0},
                            {"ell": 3, "C_ell": 0.2, "D_ell": 3.0},
                            {"ell": 4, "C_ell": 0.1, "D_ell": 2.0},
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (pack / "cmb_lite_comparison_report.json").write_text(
        json.dumps(
            {
                "best_shape_field": "record_signature",
                "best_positive_shape_field": "record_signature",
                "physical_cmb_prediction": False,
            }
        ),
        encoding="utf-8",
    )
    (pack / "defect_h3_worldlines_report.json").write_text(
        json.dumps(
            {
                "persistent_h3_worldline_count": 1,
                "bulk_worldline_precursor_receipt": False,
                "particle_matter_receipt": False,
                "worldlines": [
                    {
                        "worldline_id": "w0",
                        "observation_count": 2,
                        "birth_cycle": 0,
                        "death_cycle": 1,
                        "class_mode": "transposition",
                        "events": [
                            {"cycle": 0, "h3_spatial_point": [0.1, 0.0, 0.0], "fit_residual": 0.2},
                            {"cycle": 1, "h3_spatial_point": [0.2, 0.1, 0.0], "fit_residual": 0.2},
                        ],
                    }
                ],
                "claim_boundary": "proto-particle candidate only",
            }
        ),
        encoding="utf-8",
    )
    (pack / "particle_likeness_report.json").write_text(
        json.dumps(
            {
                "particle_matter_receipt": False,
                "worldlines": [
                    {
                        "worldline_id": "w0",
                        "particle_like": False,
                        "localization_pass": True,
                        "persistence_pass": True,
                        "bulk_localization_pass": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    vacuum = pack / "reference_vacuum_baseline"
    vacuum.mkdir()
    (vacuum / "reference_vacuum_baseline_report.json").write_text(
        json.dumps(
            {
                "mode": "reference_vacuum_baseline_bundle",
                "claim_tier": "E1",
                "claim_tier_meaning": "conventional reference ensemble",
                "free_scalar_gaussian": {
                    "mode_count": 3,
                    "sample_count": 2,
                    "raw_spectrum": [{"ell": 2, "mean_coefficient_power": 1.0}],
                    "smoothed_spectrum": [{"ell": 2, "mean_coefficient_power": 0.9}],
                    "reference_theory_regression_receipt": True,
                },
                "compact_u1_lattice_gauge": {
                    "lattice_size": 4,
                    "sweeps": 8,
                    "acceptance_rate": 0.5,
                    "post_burn_in_mean_plaquette": 0.1,
                    "plaquette_trace": [0.1, 0.2],
                    "reference_theory_regression_receipt": True,
                },
                "receipt_contract": {"reference_theory_regression": True},
                "OPH_NATIVE_QUOTIENT_ENSEMBLE_RECEIPT": False,
                "OPH_NATIVE_VACUUM_PROMOTION_RECEIPT": False,
                "OPH_PRIMORDIAL_FIELD_PROMOTION_RECEIPT": False,
                "explicit_nonclaims": ["literal QFT vacuum"],
            }
        ),
        encoding="utf-8",
    )
    (pack / "two_defect_stress_contraction_assay_report.json").write_text(
        json.dumps(
            {
                "mode": "controlled_two_defect_stress_contraction_assay_v0",
                "controlled_planted_assay": True,
                "patch_count": 4096,
                "steps": 4,
                "support_node_count": 8,
                "declared_stress_contraction_law": {"stress_coupling": 0.04},
                "stress_contraction_summary": {
                    "initial_h3_separation": 1.2,
                    "final_h3_separation": 0.8,
                    "absolute_h3_approach": 0.4,
                    "approach_fraction": 0.333,
                },
                "no_contraction_control_summary": {
                    "initial_h3_separation": 1.2,
                    "final_h3_separation": 1.2,
                    "absolute_h3_approach": 0.0,
                    "approach_fraction": 0.0,
                },
                "shuffled_pair_control_summary": {
                    "initial_h3_separation": 1.2,
                    "final_h3_separation": 1.5,
                    "absolute_h3_approach": -0.3,
                    "approach_fraction": -0.25,
                },
                "approach_margin_vs_controls": 0.333,
                "two_defect_stress_contraction_assay_receipt": True,
                "gravity_like_attraction_diagnostic_receipt": True,
                "production_gravity_receipt": False,
                "physical_gravity_prediction": False,
                "particle_matter_receipt": False,
                "trajectory_rows": [
                    {
                        "mode": "stress_contraction",
                        "step": 0,
                        "cycle": 0,
                        "left_h3_spatial_point": [-0.6, 0.0, 0.0],
                        "right_h3_spatial_point": [0.6, 0.0, 0.0],
                        "h3_separation": 1.2,
                        "local_readout_contraction": 0.98,
                    }
                ],
                "control_trajectory_rows": {"no_contraction": [], "shuffled_pair": []},
                "worldlines": [
                    {
                        "worldline_id": "stress_pair_left",
                        "observation_count": 1,
                        "events": [
                            {
                                "cycle": 0,
                                "event": "birth",
                                "class": "transposition",
                                "h3_spatial_point": [-0.6, 0.0, 0.0],
                                "pair_h3_separation": 1.2,
                            }
                        ],
                    }
                ],
                "claim_boundary": "controlled diagnostic only",
            }
        ),
        encoding="utf-8",
    )

    summary = write_universe_timeline_bundle(
        small_universe_dir=small,
        observer_run_dir=observer,
        consensus_pack_dir=pack,
        out_dir=tmp_path / "bundle",
        max_screen_points=32,
        max_observers=8,
        max_h3_objects=8,
    )

    viewer = Path(summary["viewer_path"])
    payload = Path(summary["payload_path"])
    assert viewer.exists()
    assert payload.exists()
    assert "OPH Universe Timeline Visualization" in viewer.read_text(encoding="utf-8")
    parsed = json.loads(payload.read_text(encoding="utf-8"))
    assert parsed["schemaVersion"] == "oph_universe_timeline_visualization_payload_v1"
    assert parsed["smallUniverse"]["receipts"]["FINITE_CONSENSUS_THEOREM_RECEIPT"] is True
    assert "pnSilenceToObservation" in parsed
    assert len(parsed["screen"]["clusters"]["snapshots"]) >= 2
    assert parsed["subjectiveObserverCameras"][0]["kind"] == "observer_local_subjective_camera"
    assert parsed["subjectiveObserverCameras"][0]["timeFrames"][0]["visibleReadoutHash"] == "abc"
    assert parsed["observerModularTime"]["receipts"]["observer_modular_time_receipt"] is True
    assert parsed["observerModularTime"]["receipts"]["observer_facing_3p1d_h3_experience_receipt"] is True
    assert parsed["observerModularTime"]["receipts"]["observer_facing_populated_h3_experience_receipt"] is False
    assert parsed["observerModularTime"]["receipts"]["observer_h3_object_population_receipt"] is False
    assert parsed["observerModularTime"]["subjectiveObserverCameras"][0]["cameraId"].startswith(
        "subjective_observer_"
    )
    assert len(parsed["observerModularTime"]["objectiveObserverViews"][0]["timeFrames"]) >= 32
    assert parsed["observerModularTime"]["objectiveObserverViews"][0]["timeFrames"][1]["dominantObjectPacket"] == 3
    assert parsed["observerModularTime"]["objectiveObserverViews"][0]["timeFrames"][1]["visibleObjectPackets"]
    assert parsed["observerModularTime"]["objectiveObserverViews"][0]["timeFrames"][1]["polarFieldReadout"]
    assert parsed["consensusBulk"]["receipts"]["theorem_assisted_consensus_3d_bulk_readout_receipt"] is True
    assert parsed["consensusBulk"]["receipts"]["observer_facing_populated_h3_experience_receipt"] is False
    assert parsed["consensusBulk"]["protoParticleCandidates"]["worldlines"][0]["worldlineId"] == "w0"
    assert parsed["consensusBulk"]["protoParticleCandidates"]["receipts"]["particle_matter_receipt"] is False
    assert parsed["cmbComparison"]["receipts"]["PHYSICAL_CMB_PREDICTION_RECEIPT"] is False
    assert parsed["cmbComparison"]["screenDiagnosticModel"]["field"] == "record_signature"
    assert len(parsed["cmbComparison"]["screenDiagnosticSpectrumRows"]) == 3
    assert parsed["comparableObservations"]["datasets"][0]["id"] == "cmb_tt_residual_rows"
    assert parsed["comparableObservations"]["datasets"][0]["datasetId"] == "cmb_tt_residual_rows"
    assert parsed["comparableObservations"]["receipts"]["physical_cmb_prediction"] is False
    assert parsed["visualizationViews"]["fluctuatingQuantumVacuum"]["viewId"] == "fluctuatingQuantumVacuum"
    assert parsed["visualizationViews"]["fluctuatingQuantumVacuum"]["sectionKind"] == (
        "quantum_vacuum_fluctuation"
    )
    assert parsed["visualizationViews"]["fluctuatingQuantumVacuum"]["animationChannels"]
    assert "literal QFT vacuum" in parsed["visualizationViews"]["fluctuatingQuantumVacuum"]["nonClaims"]
    assert parsed["visualizationViews"]["fluctuatingQuantumVacuum"]["receipts"][
        "physical_cmb_prediction_receipt"
    ] is False
    assert parsed["visualizationViews"]["fluctuatingQuantumVacuum"]["referenceVacuumBaseline"][
        "written"
    ] is True
    assert parsed["visualizationViews"]["fluctuatingQuantumVacuum"]["referenceVacuumBaseline"][
        "claimTier"
    ] == "E1"
    assert parsed["visualizationViews"]["fluctuatingQuantumVacuum"]["receipts"][
        "reference_vacuum_regression_receipt"
    ] is True
    assert parsed["visualizationViews"]["fluctuatingQuantumVacuum"]["receipts"][
        "OPH_NATIVE_VACUUM_PROMOTION_RECEIPT"
    ] is False
    assert parsed["visualizationViews"]["observerCamera"]["exportSufficiency"] == (
        "sufficient_for_observer_local_camera_visualization"
    )
    assert "subjectiveObserverCameras" in parsed["visualizationViews"]["observerCamera"]["dataSources"]
    assert parsed["visualizationViews"]["effectiveStringTheory"]["receipts"]["critical_edge_cft_receipt"] is False
    assert parsed["visualizationViews"]["effectiveStringTheory"]["sectionKind"] == (
        "effective_string_theory_edge_worldsheet"
    )
    assert "critical_edge_cft_receipt" in parsed["visualizationViews"]["effectiveStringTheory"][
        "promotionReceiptsRequired"
    ]
    assert parsed["visualizationViews"]["effectiveStringTheory"]["twoDefectStressContractionAssay"][
        "written"
    ] is True
    assert parsed["visualizationViews"]["effectiveStringTheory"]["receipts"][
        "two_defect_stress_contraction_assay_receipt"
    ] is True
    assert parsed["visualizationViews"]["effectiveStringTheory"]["receipts"][
        "physical_gravity_prediction"
    ] is False
    assert "critical string CFT" in parsed["visualizationViews"]["effectiveStringTheory"]["nonClaims"]
    assert "smallUniverse.cycles" in parsed["visualizationViews"]["effectiveStringTheory"]["dataSources"]
    sidecars = summary["sidecar_exports"]
    assert Path(sidecars["manifest_path"]).exists()
    assert sidecars["files"]["screen_points_csv"]["row_count"] == 2
    assert sidecars["files"]["screen_full_bin"]["row_count"] == 2
    assert sidecars["files"]["screen_full_bin"]["dtype"] == "float32-le"
    assert sidecars["files"]["screen_full_bin"]["layout"] == "x,y,z,value"
    assert Path(sidecars["files"]["screen_full_bin"]["path"]).stat().st_size == 2 * 4 * 4
    assert sidecars["files"]["observers_full_json"]["row_count"] == 1
    assert sidecars["files"]["cameras_full_json"]["row_count"] == 1
    assert sidecars["files"]["subjective_observer_cameras_csv"]["row_count"] == 1
    assert sidecars["files"]["subjective_observer_camera_frames_csv"]["row_count"] >= 32
    assert sidecars["files"]["consensus_h3_objects_csv"]["row_count"] == 1
    assert sidecars["files"]["cmb_residual_rows_csv"]["row_count"] == 1
    assert sidecars["files"]["cmb_screen_spectrum_rows_csv"]["row_count"] == 3
    assert sidecars["files"]["reference_vacuum_scalar_spectrum_csv"]["row_count"] == 2
    assert sidecars["files"]["reference_vacuum_u1_plaquette_trace_csv"]["row_count"] == 2
    assert sidecars["files"]["finite_repair_frames_csv"]["row_count"] == 2
    assert sidecars["files"]["finite_cycle_rows_csv"]["row_count"] == 2
    assert sidecars["files"]["screen_cluster_tracks_csv"]["row_count"] >= 1
    assert sidecars["files"]["proto_particle_worldlines_csv"]["row_count"] == 1
    assert sidecars["files"]["proto_particle_worldline_events_csv"]["row_count"] == 2
    assert sidecars["files"]["two_defect_stress_trajectory_csv"]["row_count"] == 1
    assert sidecars["files"]["two_defect_stress_worldlines_csv"]["row_count"] == 1
    assert sidecars["files"]["two_defect_stress_worldline_events_csv"]["row_count"] == 1
    assert Path(sidecars["files"]["subjective_observer_cameras_csv"]["path"]).exists()
    manifest = json.loads(Path(sidecars["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["schema"] == "oph_universe_visualization_sidecars_v1"
    assert manifest["files"]["screen_full_bin"]["written"] is True
    assert manifest["receipts"]["observer_facing_consensus_3d_bulk_readout_receipt"] is True
    assert manifest["receipts"]["physical_cmb_prediction_receipt"] is False
    assert manifest["receipts"]["reference_vacuum_regression_receipt"] is True
    assert manifest["receipts"]["oph_native_vacuum_promotion_receipt"] is False
    assert manifest["receipts"]["particle_matter_receipt"] is False
    assert manifest["receipts"]["two_defect_stress_contraction_assay_receipt"] is True
    assert manifest["receipts"]["production_gravity_receipt"] is False
    assert Path(summary["web_coding_agent_brief_path"]).exists()
