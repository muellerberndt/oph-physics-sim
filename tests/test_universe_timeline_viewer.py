import json
from pathlib import Path

import numpy as np

from oph_fpe.viz import write_universe_timeline_bundle


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
    assert "critical string CFT" in parsed["visualizationViews"]["effectiveStringTheory"]["nonClaims"]
    assert "smallUniverse.cycles" in parsed["visualizationViews"]["effectiveStringTheory"]["dataSources"]
    assert Path(summary["web_coding_agent_brief_path"]).exists()
