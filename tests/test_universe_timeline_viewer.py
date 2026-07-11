import json
from pathlib import Path

import numpy as np

from oph_fpe.viz import write_universe_timeline_bundle
from oph_fpe.viz.universe_timeline_viewer import (
    _effective_string_theory_payload,
    _observer_modular_time_payload,
    _read_proto_particle_candidates,
    _small_universe_payload,
    _visualization_instructions,
    _write_full_screen_field_bin,
)


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
    assert payload["contentAvailable"] is False
    assert payload["dataMode"] == "theorem_receipt_summary_only"
    assert payload["receiptSource"] == "theorem_core_receipts"
    assert payload["bundleReceiptKind"] == "theorem_core_receipt_bundle"
    assert payload["renderableExactMiniUniverseReceipt"] is False
    assert payload["receipts"]["renderable_exact_mini_universe_receipt"] is False
    assert payload["receipts"]["exact_nonzero_holonomy_cycle_count"] is None
    assert payload["receipts"]["strict_descent_violation_count"] is None
    assert payload["receipts"]["unique_terminal_normal_form_count"] is None
    assert "exact_mini_universe_nodes_missing" in payload["contentBlockers"]
    assert "receipt summary" in payload["description"]
    assert payload["claimBoundary"] == "large run theorem core"

    effective = _effective_string_theory_payload(
        visualization_views={
            "effectiveStringTheory": {
                "renderLayers": [
                    {"layer": "finite_edge_string_vibration_pulses"},
                    {"layer": "h3_worldline_overlay"},
                ]
            }
        },
        small_payload=payload,
        bulk_payload={"protoParticleCandidates": {"worldlines": [{"worldlineId": "w0"}]}},
    )
    assert effective["contentAvailable"] is True
    assert effective["finiteEdgeStringVibrationReceipt"] is False
    assert effective["finiteEdgeStringVibrationSamples"] == []
    assert effective["layerAvailability"]["finite_edge_string_vibration_pulses"] is False
    assert effective["layerAvailability"]["h3_worldline_overlay"] is True
    assert "finite_edge_string_vibration_pulses" in effective["hiddenLayers"]
    assert "h3_worldline_overlay" not in effective["hiddenLayers"]
    assert next(
        row
        for row in effective["renderLayers"]
        if row["layer"] == "finite_edge_string_vibration_pulses"
    )["hideWhenEmpty"] is True

    instructions = _visualization_instructions(
        None,
        tmp_path / "visualization_payload.json",
        {
            "smallUniverse": payload,
            "effectiveStringTheory": effective,
            "claimBoundary": "test boundary",
        },
    )
    assert "Panel 2 is receipt-only" in instructions
    assert "no exact mini-universe graph/path may be drawn" in instructions
    assert "finite-edge vibration sublayer is unavailable and must be hidden" in instructions
    assert "Canonical pedagogical cinematic storyboard" in instructions
    assert "Bounded self-reading patches" in instructions
    assert "Shared records and consensus" in instructions
    assert "Enter one observer's 3+1D view" in instructions
    assert "ASSUMED VISUAL LAYER — NOT DERIVED" in instructions
    assert "Defect worldlines styled as matter" in instructions
    assert "diagnostic CMB-shaped sky" in instructions
    assert "explanatory overview (not observer-visible)" in instructions
    assert "prefers-reduced-motion" in instructions
    assert "256,000,000-byte hard package ceiling" in instructions


def test_full_screen_sidecar_falls_back_to_manifest_patch_count(tmp_path: Path):
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    out.mkdir()
    (run / "manifest.json").write_text(json.dumps({"patch_count": 6}), encoding="utf-8")

    report = _write_full_screen_field_bin(out, run, {"screen": {}})

    assert report["written"] is True
    assert report["row_count"] == 6
    assert report["dtype"] == "float32-le"
    assert report["layout"] == "x,y,z,value"
    assert report["field_name"] == "screen_position_support"
    assert report["exact_freezeout_values"] is False
    assert report["reason"] == "freezeout_fields_npz_missing_full_s2_support_only"
    path = Path(report["path"])
    assert path.stat().st_size == 6 * 4 * 4
    packed = np.fromfile(path, dtype="<f4").reshape((-1, 4))
    assert packed.shape == (6, 4)
    assert np.allclose(np.linalg.norm(packed[:, :3], axis=1), 1.0)


def test_observer_payload_synthesizes_compact_run_views(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"patch_count": 16}), encoding="utf-8")
    (run / "observer_modular_experience_report.json").write_text(
        json.dumps({"observer_count": 4, "observer_modular_time_receipt": True}),
        encoding="utf-8",
    )
    (run / "mismatch_trace.csv").write_text(
        "cycle,phi,mismatch_edges,committed_fraction\n0,1.0,4,0.0\n1,0.0,0,1.0\n",
        encoding="utf-8",
    )

    payload = _observer_modular_time_payload(
        run,
        max_observers=4,
        max_objective_observer_views=None,
    )

    assert len(payload["observers"]) == 4
    assert len(payload["objectiveObserverViews"]) == 4
    assert payload["objectiveObserverViews"][0]["timeFrames"]
    assert payload["objectiveObserverViews"][0]["visibleObjectPackets"]
    assert payload["overlapLinks"]
    assert payload["receipts"]["observer_modular_time_receipt"] is True


def test_proto_particle_candidates_fall_back_to_two_defect_assay(tmp_path: Path):
    (tmp_path / "two_defect_stress_contraction_assay_report.json").write_text(
        json.dumps(
            {
                "controlled_planted_assay": True,
                "two_defect_stress_contraction_assay_receipt": True,
                "worldlines": [
                    {
                        "worldline_id": "stress_pair_left",
                        "observation_count": 2,
                        "events": [
                            {"cycle": 0, "h3_spatial_point": [-0.6, 0.0, 0.0], "class": "transposition"},
                            {"cycle": 1, "h3_spatial_point": [-0.4, 0.0, 0.0], "class": "transposition"},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = _read_proto_particle_candidates(tmp_path, max_worldlines=8)

    assert payload["worldlineSource"] == "two_defect_stress_contraction_assay_report"
    assert payload["receipts"]["bulk_worldline_precursor_receipt"] is False
    assert payload["receipts"]["controlled_two_defect_worldline_count"] == 1
    assert payload["worldlines"][0]["diagnosticOnly"] is True
    assert payload["worldlines"][0]["particleLike"] is False
    assert len(payload["worldlines"][0]["events"]) == 2


def test_proto_particle_candidates_prefer_free_two_defect_dynamics_before_controlled_assay(tmp_path: Path):
    (tmp_path / "free_two_defect_dynamics_report.json").write_text(
        json.dumps(
            {
                "free_dynamics_diagnostic": True,
                "free_two_defect_dynamics_receipt": True,
                "free_dynamics_summary": {"contact_outcome": "scatter"},
                "worldlines": [
                    {
                        "worldline_id": "free_pair_left",
                        "observation_count": 2,
                        "persistent": True,
                        "contact_outcome": "scatter",
                        "events": [
                            {
                                "cycle": 0,
                                "h3_spatial_point": [-0.45, 0.11, -0.08],
                                "velocity": [0.03, 0.01, 0.0],
                                "class": "transposition",
                                "support_overlap_fraction": 0.0,
                                "contact_outcome": None,
                                "charge_conservation_pass": True,
                            },
                            {
                                "cycle": 1,
                                "h3_spatial_point": [-0.40, 0.13, -0.06],
                                "velocity": [0.02, 0.02, 0.01],
                                "class": "transposition",
                                "support_overlap_fraction": 0.5,
                                "contact_outcome": "scatter",
                                "charge_conservation_pass": True,
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "two_defect_stress_contraction_assay_report.json").write_text(
        json.dumps(
            {
                "controlled_planted_assay": True,
                "two_defect_stress_contraction_assay_receipt": True,
                "worldlines": [
                    {
                        "worldline_id": "stress_pair_left",
                        "observation_count": 1,
                        "events": [{"cycle": 0, "h3_spatial_point": [-0.6, 0.0, 0.0]}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = _read_proto_particle_candidates(tmp_path, max_worldlines=8)

    assert payload["worldlineSource"] == "free_two_defect_dynamics_report"
    assert payload["receipts"]["free_two_defect_worldline_count"] == 1
    assert payload["receipts"]["controlled_two_defect_worldline_count"] == 1
    assert payload["receipts"]["free_two_defect_dynamics_receipt"] is True
    assert payload["worldlines"][0]["worldlineId"] == "free_pair_left"
    assert payload["worldlines"][0]["freeDynamicsDiagnostic"] is True
    assert payload["worldlines"][0]["controlledPlantedAssay"] is False
    assert payload["worldlines"][0]["contactOutcome"] == "scatter"
    assert payload["worldlines"][0]["events"][1]["supportOverlapFraction"] == 0.5


def test_proto_particle_candidates_prefer_organic_defect_population_before_free_pair(tmp_path: Path):
    (tmp_path / "organic_defect_population_report.json").write_text(
        json.dumps(
            {
                "organic_defect_population_diagnostic": True,
                "organic_defect_population_receipt": True,
                "organic_population_summary": {"worldline_count": 2},
                "worldlines": [
                    {
                        "worldline_id": "organic_defect_00",
                        "observation_count": 2,
                        "persistent": True,
                        "class_mode": "transposition",
                        "events": [
                            {
                                "cycle": 0,
                                "h3_spatial_point": [0.1, 0.2, 0.3],
                                "velocity": [0.01, 0.02, 0.03],
                                "class": "transposition",
                                "holonomy_mode": 1,
                                "local_stress_density": 0.4,
                                "nearest_defect_id": "organic_defect_01",
                            },
                            {
                                "cycle": 1,
                                "h3_spatial_point": [0.2, 0.25, 0.32],
                                "velocity": [0.02, 0.01, 0.01],
                                "class": "transposition",
                                "holonomy_mode": 1,
                                "local_stress_density": 0.5,
                                "nearest_defect_id": "organic_defect_01",
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "free_two_defect_dynamics_report.json").write_text(
        json.dumps(
            {
                "free_dynamics_diagnostic": True,
                "free_two_defect_dynamics_receipt": True,
                "worldlines": [
                    {
                        "worldline_id": "free_pair_left",
                        "observation_count": 1,
                        "events": [{"cycle": 0, "h3_spatial_point": [-0.45, 0.11, -0.08]}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = _read_proto_particle_candidates(tmp_path, max_worldlines=8)

    assert payload["worldlineSource"] == "organic_defect_population_report"
    assert payload["receipts"]["organic_defect_worldline_count"] == 1
    assert payload["receipts"]["free_two_defect_worldline_count"] == 1
    assert payload["receipts"]["organic_defect_population_receipt"] is True
    assert payload["worldlines"][0]["worldlineId"] == "organic_defect_00"
    assert payload["worldlines"][0]["organicDefectPopulationDiagnostic"] is True
    assert payload["worldlines"][0]["events"][0]["localStressDensity"] == 0.4


def test_proto_particle_candidates_prefer_organic_json_over_legacy_csv_sidecars(tmp_path: Path):
    (tmp_path / "proto_particle_worldlines.csv").write_text(
        "worldline_id,observation_count,birth_cycle,death_cycle,particle_like\n"
        "stress_pair_left,1,0,0,false\n",
        encoding="utf-8",
    )
    (tmp_path / "proto_particle_worldline_events.csv").write_text(
        "worldline_id,event_index,cycle,x,y,z,fit_residual,support_node_count,particle_like\n"
        "stress_pair_left,0,0,-0.6,0.0,0.0,0.0,8,false\n",
        encoding="utf-8",
    )
    (tmp_path / "organic_defect_population_report.json").write_text(
        json.dumps(
            {
                "organic_defect_population_diagnostic": True,
                "organic_defect_population_receipt": True,
                "organic_population_summary": {"worldline_count": 1},
                "worldlines": [
                    {
                        "worldline_id": "organic_defect_00",
                        "observation_count": 1,
                        "persistent": True,
                        "events": [
                            {
                                "cycle": 0,
                                "h3_spatial_point": [0.1, 0.2, 0.3],
                                "class": "transposition",
                                "local_stress_density": 0.4,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = _read_proto_particle_candidates(tmp_path, max_worldlines=8)

    assert payload["worldlineSource"] == "organic_defect_population_report"
    assert payload["worldlines"][0]["worldlineId"] == "organic_defect_00"
    assert payload["receipts"]["csv_proto_worldline_count"] == 1
    assert payload["receipts"]["legacy_csv_sidecar_used"] is False


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
    (pack / "strict_neutral_object_bulk_report.json").write_text(
        json.dumps(
            {
                "mode": "strict_neutral_object_bulk_v0",
                "object_count": 1,
                "strict_neutral_object_bulk": False,
                "STRICT_NEUTRAL_OBJECT_BULK_RECEIPT": False,
                "dimension": {"median_dimension_estimate": 7.5},
                "latent_geometry_selection": {"selected_model": "E4", "h3_selected": False},
                "blockers": ["object_dimension_estimators_do_not_agree_3d"],
                "claim_boundary": "neutral candidates only",
            }
        ),
        encoding="utf-8",
    )
    (pack / "neutral_objects.jsonl").write_text(
        json.dumps(
            {
                "object_id": "neutral_1",
                "observer_ids": [0, 1, 2],
                "visible_signature_key": "3:7:2",
                "persistence": 4.0,
                "overlap_agreement": 0.75,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (pack / "physical_cmb_output_comparison_report.json").write_text(
        json.dumps(
            {
                "PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT": True,
                "USABLE_PHYSICAL_CMB_DATA_RECEIPT": True,
                "PHYSICAL_CMB_PREDICTION_RECEIPT": False,
                "best_oph_residual_summary": {"rms_sigma_residual": 0.9},
                "best_oph_residual_rows": [
                    {"ell": 2, "observed_D_ell": 1000, "model_D_ell": 990, "residual_sigma": -0.1}
                ],
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
    (pack / "yang_mills_gap_certificate_report.json").write_text(
        json.dumps(
            {
                "schema": "oph_yang_mills_gap_certificate_v0",
                "paper_source": "markdown/yang_mills_gap_clay_problem.md",
                "gauge_group": {"name": "SU(2)", "compact": True, "simple": True, "nonabelian": True},
                "regulator": {
                    "dimension": 4,
                    "lattice_size": 2,
                    "site_count": 16,
                    "link_count": 64,
                    "plaquette_count": 96,
                    "boundary": "periodic_hypercubic",
                    "beta": 2.2,
                    "sweeps": 2,
                    "proposal_width": 0.35,
                    "seed": 7,
                    "transition_bins": 4,
                },
                "lattice_gauge_stage": {
                    "compact_simple_nonabelian_reference": True,
                    "su2_reference": True,
                    "su3_reference": False,
                    "u1_reference": False,
                    "four_dimensional_wilson_lattice": True,
                    "continuum_yang_mills_theory_constructed": False,
                },
                "finite_lattice_diagnostics": {
                    "mean_plaquette": 0.12,
                    "plaquette_variance": 0.01,
                    "acceptance_rate": 0.9,
                    "finite_nontriviality_proxy_receipt": True,
                    "canonical_serial_chain_replay_receipt": True,
                },
                "finite_transfer_gap_diagnostic": {
                    "spectral_gap_estimate": 0.2,
                    "finite_transfer_gap_proxy_receipt": True,
                },
                "reflection_positivity_proxy": {
                    "reflection_gram_lower_bound": 0.0,
                    "finite_reflection_gram_proxy_receipt": True,
                },
                "continuum_certificate": {
                    "candidate_complete": False,
                    "continuum_certificate_receipt": False,
                    "missing": ["support_visible_extraction_receipt"],
                },
                "promotion_status": {
                    "finite_nonabelian_regulator": "pass",
                    "finite_positive_gap_floor": "pass",
                    "continuum_certificate": "pending",
                    "os_reconstruction": "pending",
                    "yang_mills_identification": "conditional",
                    "yang_mills_mass_gap": "not_promoted",
                    "reasons": ["missing continuum certificate field: support_visible_extraction_receipt"],
                },
                "plaquette_trace": [
                    {
                        "sweep": 0,
                        "mean_plaquette": 0.1,
                        "plaquette_variance": 0.01,
                        "action_density": 0.2,
                        "acceptance_rate": 0.9,
                    },
                    {
                        "sweep": 1,
                        "mean_plaquette": 0.2,
                        "plaquette_variance": 0.02,
                        "action_density": 0.4,
                        "acceptance_rate": 0.88,
                    },
                ],
                "wilson_loop_trace": [
                    {"sweep": 0, "loop": "plaquette_1x1", "mean_normalized_trace": 0.1, "variance": 0.01}
                ],
                "polyakov_loop_trace": [
                    {"sweep": 0, "loop": "time_polyakov_abs", "mean_abs_normalized_trace": 0.3}
                ],
                "orientation_plaquette_rows": [
                    {"sweep": 0, "orientation": "01", "mean_plaquette": 0.1}
                ],
                "refinement_gap_rows": [
                    {
                        "lattice_size": 2,
                        "site_count": 16,
                        "sweeps": 2,
                        "mean_plaquette": 0.12,
                        "plaquette_variance": 0.01,
                        "acceptance_rate": 0.9,
                        "finite_transfer_gap_estimate": 0.2,
                        "lambda2_abs": 0.8,
                        "screening_mass_proxy": 0.1,
                        "finite_transfer_gap_proxy_receipt": True,
                    }
                ],
                "finite_nonabelian_gauge_gap_diagnostic_receipt": True,
                "finite_repair_gap_proxy_receipt": True,
                "continuum_yang_mills_mass_gap_receipt": False,
                "YANG_MILLS_GAP_REPRODUCED_RECEIPT": False,
                "CLAY_YANG_MILLS_GAP_RECEIPT": False,
                "explicit_nonclaims": ["reproduced Yang-Mills mass gap"],
                "claim_boundary": "Finite SU(2) diagnostic only.",
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
    viewer_text = viewer.read_text(encoding="utf-8")
    assert "OPH Universe Timeline Visualization" in viewer_text
    assert "Exact mini-universe graph not exported" in viewer_text
    assert "repair.disabled=!DATA.smallUniverse.contentAvailable" in viewer_text
    parsed = json.loads(payload.read_text(encoding="utf-8"))
    assert parsed["schemaVersion"] == "oph_universe_timeline_visualization_payload_v1"
    assert parsed["smallUniverse"]["receipts"]["FINITE_CONSENSUS_THEOREM_RECEIPT"] is True
    assert parsed["smallUniverse"]["contentAvailable"] is True
    assert parsed["smallUniverse"]["dataMode"] == "exact_mini_universe"
    assert parsed["smallUniverse"]["receiptSource"] == "exact_consensus_receipt"
    assert parsed["smallUniverse"]["bundleReceiptKind"] == "exact_mini_universe_evidence_bundle"
    assert parsed["smallUniverse"]["renderableExactMiniUniverseReceipt"] is True
    assert parsed["smallUniverse"]["receipts"]["renderable_exact_mini_universe_receipt"] is True
    assert parsed["smallUniverse"]["contentBlockers"] == []
    assert parsed["smallUniverse"]["receipts"]["exact_nonzero_holonomy_cycle_count"] == 0
    assert "pnSilenceToObservation" in parsed
    assert len(parsed["screen"]["clusters"]["snapshots"]) >= 2
    assert parsed["subjectiveObserverCameras"][0]["kind"] == "observer_local_subjective_camera"
    assert parsed["subjectiveObserverCameras"][0]["timeFrames"][0]["visibleReadoutHash"] == "abc"
    assert parsed["subjectiveObserverCameras"][0]["visibleProtoWorldlineIds"] == ["w0"]
    assert parsed["subjectiveObserverCameras"][0]["visibleProtoWorldlineSightingCount"] > 0
    first_sighting = parsed["subjectiveObserverCameras"][0]["timeFrames"][0]["visibleProtoWorldlines"][0]
    assert first_sighting["worldlineId"] == "w0"
    assert first_sighting["particleLike"] is False
    assert first_sighting["diagnosticOnly"] is True
    assert "h3SpatialPoint" not in first_sighting
    assert first_sighting["observerLocalReadout"]["coordinateSystem"] == "observer_local_tangent_screen_readout"
    assert first_sighting["observerLocalReadout"]["hiddenGlobalH3Suppressed"] is True
    assert 0.0 <= first_sighting["observerLocalReadout"]["visibilityScore"] <= 1.0
    assert first_sighting["worldlineRef"].startswith("consensusBulk.protoParticleCandidates.worldlines[")
    assert parsed["observerModularTime"]["receipts"]["observer_modular_time_receipt"] is True
    assert parsed["observerModularTime"]["receipts"]["observer_facing_3p1d_h3_experience_receipt"] is True
    assert parsed["observerModularTime"]["receipts"]["observer_facing_populated_h3_experience_receipt"] is False
    assert parsed["observerModularTime"]["receipts"]["observer_h3_object_population_receipt"] is False
    assert parsed["observerModularTime"]["subjectiveObserverCameraRefs"][0]["cameraId"].startswith(
        "subjective_observer_"
    )
    assert len(parsed["observerModularTime"]["objectiveObserverViews"][0]["timeFrames"]) >= 32
    assert parsed["observerModularTime"]["objectiveObserverViews"][0]["timeFrames"][1]["dominantObjectPacket"] == 3
    assert parsed["observerModularTime"]["objectiveObserverViews"][0]["timeFrames"][1]["visibleObjectPackets"]
    assert parsed["observerModularTime"]["objectiveObserverViews"][0]["timeFrames"][1]["polarFieldReadout"]
    assert parsed["consensusBulk"]["receipts"]["theorem_assisted_consensus_3d_bulk_readout_receipt"] is True
    assert parsed["consensusBulk"]["receipts"]["observer_facing_populated_h3_experience_receipt"] is True
    assert parsed["consensusBulk"]["receipts"]["observer_h3_object_population_receipt"] is True
    assert parsed["consensusBulk"]["receipts"]["strict_neutral_object_bulk_receipt"] is False
    assert parsed["consensusBulk"]["neutralObjectCandidates"][0]["objectId"] == "neutral_1"
    assert parsed["consensusBulk"]["neutralObjectCandidates"][0]["spatialEmbeddingAvailable"] is False
    assert parsed["consensusBulk"]["neutralObjectSummary"]["objectCount"] == 1
    assert parsed["consensusBulk"]["neutralObjectSummary"]["receipt"] is False
    assert parsed["consensusBulk"]["h3ChartStatus"]["renderable"] is True
    assert parsed["consensusBulk"]["h3ChartStatus"]["displayStatus"] == "available"
    strict_display = parsed["consensusBulk"]["receiptDisplay"]["strict_neutral_third_person_bulk_receipt"]
    assert strict_display["displayStatus"] == "not_promoted"
    assert strict_display["renderAsError"] is False
    assert parsed["consensusBulk"]["protoParticleCandidates"]["worldlines"][0]["worldlineId"] == "w0"
    assert parsed["consensusBulk"]["protoParticleCandidates"]["receipts"]["particle_matter_receipt"] is False
    assert parsed["cmbComparison"]["receipts"]["PHYSICAL_CMB_PREDICTION_RECEIPT"] is False
    assert len(parsed["cmbComparison"]["observedRows"]) == 1
    assert len(parsed["cmbComparison"]["modelRows"]) == 1
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
    assert parsed["visualizationViews"]["fluctuatingQuantumVacuum"]["yangMillsGapCertificate"][
        "written"
    ] is True
    assert parsed["visualizationViews"]["fluctuatingQuantumVacuum"]["yangMillsGapCertificate"][
        "gaugeGroup"
    ]["name"] == "SU(2)"
    assert parsed["visualizationViews"]["fluctuatingQuantumVacuum"]["receipts"][
        "finite_nonabelian_gauge_gap_diagnostic_receipt"
    ] is True
    assert parsed["visualizationViews"]["fluctuatingQuantumVacuum"]["receipts"][
        "YANG_MILLS_GAP_REPRODUCED_RECEIPT"
    ] is False
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
    assert parsed["visualizationViews"]["emergentCurvedSpacetime"]["viewId"] == "emergentCurvedSpacetime"
    branch_gate = parsed["visualizationViews"]["emergentCurvedSpacetime"]["einsteinBranchEntry"]
    assert branch_gate["issue"] == 503
    assert branch_gate["receipts"]["einstein_branch_entry_receipt"] is False
    assert "E0_einstein_branch_entry_umbrella" in branch_gate["blockers"]
    assert parsed["visualizationViews"]["emergentCurvedSpacetime"]["receipts"][
        "einstein_branch_entry_receipt"
    ] is False
    assert parsed["visualizationViews"]["emergentCurvedSpacetime"]["receipts"][
        "raw_production_gravity_requested"
    ] is False
    assert parsed["visualizationViews"]["emergentCurvedSpacetime"]["receipts"][
        "production_gravity_receipt"
    ] is False
    assert "physical gravity prediction" in parsed["visualizationViews"]["emergentCurvedSpacetime"]["nonClaims"]
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
    selector = parsed["visualizationViews"]["effectiveStringTheory"]["stringVacuumSelector"]
    assert selector["schema"] == "oph_string_vacuum_selector_visualization_v1"
    assert selector["receipts"]["string_vacuum_selector_visualization_receipt"] is True
    assert selector["receipts"]["critical_edge_cft_receipt"] is False
    assert selector["receipts"]["global_singleton_string_vacuum_receipt"] is False
    selected_candidate = next(row for row in selector["encodedCandidateSieve"] if row["selected"])
    assert selected_candidate["candidate"] == "BD_{n=1,+}^{SU(5),Z2}"
    assert selected_candidate["scoreNumerator"] == 9
    assert len(selector["criticalEdgeCertificateGates"]) == 9
    assert len(selector["operatorSafetyTable"]) == 10
    assert parsed["visualizationViews"]["effectiveStringTheory"]["receipts"][
        "two_defect_stress_contraction_assay_receipt"
    ] is True
    assert parsed["visualizationViews"]["effectiveStringTheory"]["receipts"][
        "string_vacuum_selector_visualization_receipt"
    ] is True
    assert parsed["visualizationViews"]["effectiveStringTheory"]["receipts"][
        "global_singleton_string_vacuum_receipt"
    ] is False
    assert parsed["visualizationViews"]["effectiveStringTheory"]["receipts"][
        "physical_gravity_prediction"
    ] is False
    assert "critical string CFT" in parsed["visualizationViews"]["effectiveStringTheory"]["nonClaims"]
    assert "smallUniverse.cycles" in parsed["visualizationViews"]["effectiveStringTheory"]["dataSources"]
    assert parsed["effectiveStringTheory"]["schema"] == "oph_effective_string_theory_visualization_v1"
    assert parsed["effectiveStringTheory"]["contentAvailable"] is True
    assert parsed["effectiveStringTheory"]["h3ProtoParticleWorldlines"]
    assert parsed["effectiveStringTheory"]["finiteEdgeStringVibrationReceipt"] is True
    assert parsed["effectiveStringTheory"]["finiteEdgeStringVibrationSamples"]
    assert parsed["effectiveStringTheory"]["finiteEdgeStringVibrationSamples"][0]["sampleKind"] == (
        "exact_finite_repair_edge_pulse"
    )
    assert parsed["effectiveStringTheory"]["layerAvailability"][
        "finite_edge_string_vibration_pulses"
    ] is True
    assert "finite_edge_string_vibration_pulses" not in parsed["effectiveStringTheory"]["hiddenLayers"]
    assert parsed["effectiveStringTheory"]["stringVacuumSelector"]["candidateNamedWitness"] == "BD_{n=1,+}^{OPH}"
    assert parsed["emergentCurvedSpacetime"]["schema"] == "oph_emergent_curved_spacetime_visualization_v1"
    assert parsed["emergentCurvedSpacetime"]["contentAvailable"] is True
    assert parsed["emergentCurvedSpacetime"]["curvatureProxyPoints"]
    assert parsed["emergentCurvedSpacetime"]["dataRefs"]["spacetimeCompactionField"] == (
        "emergentCurvedSpacetime.curvatureProxyPoints"
    )
    assert parsed["emergentCurvedSpacetime"]["continuousBulkField"]["contentAvailable"] is True
    assert parsed["emergentCurvedSpacetime"]["continuousBulkField"]["volumeSamples"]
    assert parsed["emergentCurvedSpacetime"]["continuousBulkField"]["sliceSamples"]
    assert parsed["emergentCurvedSpacetime"]["sourceMath"]["model"] == (
        "oph_quotient_visible_source_to_h3_compaction_v1"
    )
    curved_point = parsed["emergentCurvedSpacetime"]["curvatureProxyPoints"][0]
    assert curved_point["sourceDensity"] > 0.0
    assert curved_point["h3GreenPotential"] > 0.0
    assert 0.0 <= curved_point["compactificationFactor"] <= 0.5
    assert 0.5 <= curved_point["emergentSpatialScaleFactor"] <= 1.0
    assert curved_point["gravitySourceInterpretation"] == "quotient_visible_stress_readout_not_rest_mass"
    assert parsed["emergentCurvedSpacetime"]["einsteinBranchEntry"]["issue"] == 503
    assert parsed["emergentCurvedSpacetime"]["receipts"]["einstein_branch_entry_receipt"] is False
    assert parsed["emergentCurvedSpacetime"]["receipts"]["EINSTEIN_BRANCH_ENTRY_RECEIPT"] is False
    assert parsed["emergentCurvedSpacetime"]["receipts"]["production_gravity_receipt"] is False
    assert parsed["emergentCurvedSpacetime"]["receipts"]["physical_gravity_prediction"] is False
    assert parsed["observerCinema"]["schema"] == "oph_observer_cinema_v1"
    assert parsed["observerCinema"]["observerViewRefs"]
    assert parsed["observerCinema"]["subjectiveCameraRefs"]
    assert parsed["observerCinema"]["availability"]["observerViewCount"] == 1
    assert parsed["observerCinema"]["availability"]["protoWorldlineSightingCount"] > 0
    assert parsed["coordinateSystems"]["h3_hyperboloid_spatial_components_v1"]["model"] == (
        "future_unit_hyperboloid_spatial_components"
    )
    assert parsed["assumedDs4Spacetime"]["provenance"]["kind"] == "simulation_assumption"
    assert parsed["assumedDs4Spacetime"]["receipts"]["derived_physical_ds4_receipt"] is False
    assert parsed["assumedDs4Spacetime"]["receipts"]["physical_particle_matter_receipt"] is False
    assert parsed["assumedDs4Spacetime"]["assumptionLedger"][
        "computedTheoremReceiptsUnchanged"
    ] is True
    assert parsed["hilbertSpaceObserverAlgebra"]["schema"] == "oph_hilbert_observer_algebra_summary_v1"
    assert parsed["hilbertSpaceObserverAlgebra"]["finiteSupportAlgebraPopulated"] is True
    assert parsed["hilbertSpaceObserverAlgebra"]["visibleObjectPacketCount"] > 0
    assert parsed["hilbertSpaceObserverAlgebra"]["representativeObservers"]
    assert parsed["observerAnatomy"]["schema"] == "oph_observer_anatomy_v1"
    assert parsed["observerAnatomy"]["observers"]
    assert parsed["observerAnatomy"]["populationSummary"]["finiteSupportAlgebraPopulated"] is True
    assert parsed["paperAccuracy"]["schema"] == "oph_paper_accuracy_guard_v1"
    assert parsed["paperAccuracy"]["receipts"]["paper_accuracy_guard_receipt"] is True
    assert parsed["paperAccuracy"]["receipts"]["EINSTEIN_BRANCH_ENTRY_RECEIPT"] is False
    assert parsed["paperAccuracy"]["receipts"]["OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1"] is False
    assert parsed["paperAccuracy"]["receipts"]["PHYSICAL_CMB_PREDICTION_RECEIPT"] is False
    assert parsed["paperAccuracy"]["receipts"]["production_gravity_receipt"] is False
    assert next(row for row in parsed["paperAccuracy"]["checks"] if row["id"] == "curved_spacetime_compaction")[
        "paperStatus"
    ] == "Einstein-branch visualization diagnostic"
    einstein_check = next(row for row in parsed["paperAccuracy"]["checks"] if row["id"] == "einstein_branch_entry")
    assert einstein_check["passed"] is False
    assert einstein_check["issue"] == 503
    render_data = parsed["visualizationRenderData"]
    assert render_data["schema"] == "oph_visualization_render_data_v1"
    assert render_data["availability"]["finiteRepairGraphContentAvailable"] is True
    assert render_data["availability"]["finiteRepairGraphNodeCount"] == 3
    assert render_data["availability"]["finiteRepairGraphEdgeCount"] == 3
    assert render_data["availability"]["finiteRepairGraphFrameCount"] == 2
    assert render_data["availability"]["finiteEdgeStringVibrationContentAvailable"] is True
    assert render_data["availability"]["subjectiveCameraCount"] == 1
    assert render_data["availability"]["h3ObjectCount"] == 1
    assert render_data["availability"]["neutralObjectCandidateCount"] == 1
    assert render_data["availability"]["protoWorldlineCount"] == 1
    assert render_data["availability"]["observerProtoWorldlineSightingCount"] > 0
    assert render_data["availability"]["curvatureProxyPointCount"] >= 1
    assert render_data["availability"]["continuousBulkFieldVolumeSampleCount"] > 0
    assert render_data["availability"]["cmbResidualPointCount"] == 1
    assert render_data["cameraPresets"]
    assert render_data["sceneGraph"]["screen"]["points"]
    assert render_data["sceneGraph"]["observerGraph"]["nodes"]
    assert render_data["sceneGraph"]["bulk"]["h3Objects"][0]["id"] == "obj0"
    assert render_data["sceneGraph"]["bulk"]["h3ChartStatus"]["renderable"] is True
    assert render_data["sceneGraph"]["curvedSpacetime"]["continuousBulkField"]["contentAvailable"] is True
    assert render_data["sceneGraph"]["bulk"]["receiptDisplay"]["strict_neutral_third_person_bulk_receipt"][
        "displayStatus"
    ] == "not_promoted"
    assert render_data["sceneGraph"]["bulk"]["protoWorldlines"][0]["polyline"]
    assert render_data["sceneGraph"]["finiteRepairGraph"]["contentAvailable"] is True
    assert render_data["sceneGraph"]["finiteRepairGraph"]["renderableExactMiniUniverseReceipt"] is True
    assert render_data["sceneGraph"]["curvedSpacetime"]["proxyPoints"]
    assert render_data["sceneGraph"]["curvedSpacetime"]["sourceMath"]["model"] == (
        "oph_quotient_visible_source_to_h3_compaction_v1"
    )
    assert render_data["sceneGraph"]["curvedSpacetime"]["proxyPoints"][0]["compactificationFactor"] is not None
    assert render_data["sceneGraph"]["curvedSpacetime"]["proxyPoints"][0]["emergentSpatialScaleFactor"] is not None
    assert render_data["plotSeries"]["curvatureProxyBySource"]
    assert render_data["animationTimeline"]
    assert render_data["rendererHints"]["recommendedRepairTimeField"] == "playbackRelativeTime"
    assert render_data["repairPlayback"]["rawTimeField"] == "rawRelativeTime"
    assert "rawRelativeTime" in render_data["animationTimeline"][0]
    assert "playbackRelativeTime" in render_data["animationTimeline"][0]
    assert render_data["plotSeries"]["cmbResidualSigma"][0]["x"] == 2
    assert render_data["legend"]
    assert any(row["id"] == "physical_cmb_prediction" for row in render_data["claimBadges"])
    assert next(row for row in render_data["claimBadges"] if row["id"] == "observer_h3_object_population")[
        "passed"
    ] is True
    strict_badge = next(row for row in render_data["claimBadges"] if row["id"] == "strict_neutral_bulk")
    assert strict_badge["displayStatus"] == "not_promoted"
    assert strict_badge["severity"] == "blocked"
    assert strict_badge["renderAsError"] is False
    assert next(row for row in render_data["claimBadges"] if row["id"] == "view:observerCamera")[
        "style"
    ] == "view_contract_info"
    instructions = Path(summary["instructions_path"]).read_text(encoding="utf-8")
    assert "Panel 2 shows one deterministic repair path through the exact finite mini-universe" in instructions
    assert "finiteEdgeStringVibrationSamples` is available" in instructions
    assert "Objectivity is the agreement carried by shared records" in instructions
    assert "ASSUMED DEFECT-AS-MATTER STYLING" in instructions
    sidecars = summary["sidecar_exports"]
    assert summary["visualizer_pack"]["under_hard_limit"] is True
    assert summary["visualizer_pack"]["byte_count"] < 256_000_000
    assert Path(summary["visualizer_pack"]["path"]).exists()
    assert Path(sidecars["manifest_path"]).exists()
    assert sidecars["files"]["screen_points_csv"]["row_count"] == 2
    assert sidecars["files"]["screen_full_bin"]["row_count"] == 2
    assert sidecars["files"]["screen_full_bin"]["dtype"] == "float32-le"
    assert sidecars["files"]["screen_full_bin"]["layout"] == "x,y,z,value"
    assert Path(sidecars["files"]["screen_full_bin"]["path"]).stat().st_size == 2 * 4 * 4
    assert sidecars["files"]["visualization_render_data_json"]["written"] is True
    assert sidecars["files"]["visualization_render_data_json"]["schema"] == "oph_visualization_render_data_v1"
    assert sidecars["files"]["effective_string_theory_json"]["schema"] == (
        "oph_effective_string_theory_visualization_v1"
    )
    assert sidecars["files"]["emergent_curved_spacetime_json"]["schema"] == (
        "oph_emergent_curved_spacetime_visualization_v1"
    )
    assert sidecars["files"]["observer_cinema_json"]["schema"] == "oph_observer_cinema_v1"
    assert sidecars["files"]["hilbert_space_observer_algebra_json"]["schema"] == (
        "oph_hilbert_observer_algebra_summary_v1"
    )
    assert sidecars["files"]["observer_anatomy_json"]["schema"] == "oph_observer_anatomy_v1"
    assert sidecars["files"]["paper_accuracy_json"]["schema"] == "oph_paper_accuracy_guard_v1"
    assert sidecars["files"]["observers_full_json"]["row_count"] == 1
    assert sidecars["files"]["cameras_full_json"]["row_count"] == 1
    assert sidecars["files"]["subjective_observer_cameras_csv"]["row_count"] == 1
    assert sidecars["files"]["subjective_observer_camera_frames_csv"]["row_count"] >= 32
    assert sidecars["files"]["observer_proto_worldline_sightings_csv"]["row_count"] > 0
    assert sidecars["files"]["consensus_h3_objects_csv"]["row_count"] == 1
    assert sidecars["files"]["neutral_object_candidates_csv"]["row_count"] == 1
    assert sidecars["files"]["cmb_residual_rows_csv"]["row_count"] == 1
    assert sidecars["files"]["cmb_screen_spectrum_rows_csv"]["row_count"] == 3
    assert sidecars["files"]["reference_vacuum_scalar_spectrum_csv"]["row_count"] == 2
    assert sidecars["files"]["reference_vacuum_u1_plaquette_trace_csv"]["row_count"] == 2
    assert sidecars["files"]["yang_mills_su2_plaquette_trace_csv"]["row_count"] == 2
    assert sidecars["files"]["yang_mills_su2_wilson_loop_trace_csv"]["row_count"] == 1
    assert sidecars["files"]["yang_mills_su2_polyakov_loop_trace_csv"]["row_count"] == 1
    assert sidecars["files"]["yang_mills_su2_orientation_plaquettes_csv"]["row_count"] == 1
    assert sidecars["files"]["yang_mills_su2_refinement_gap_csv"]["row_count"] == 1
    assert sidecars["files"]["yang_mills_gap_promotion_gates_csv"]["row_count"] >= 1
    assert sidecars["files"]["finite_repair_frames_csv"]["row_count"] == 2
    assert sidecars["files"]["finite_cycle_rows_csv"]["row_count"] == 2
    assert sidecars["files"]["finite_edge_string_vibration_samples_csv"]["row_count"] >= 1
    assert sidecars["files"]["screen_cluster_tracks_csv"]["row_count"] >= 1
    assert sidecars["files"]["proto_particle_worldlines_csv"]["row_count"] == 1
    assert sidecars["files"]["proto_particle_worldline_events_csv"]["row_count"] == 2
    assert sidecars["files"]["emergent_curved_spacetime_curvature_proxy_csv"]["row_count"] >= 1
    assert sidecars["files"]["emergent_curved_spacetime_time_slices_csv"]["row_count"] >= 1
    curvature_header = Path(sidecars["files"]["emergent_curved_spacetime_curvature_proxy_csv"]["path"]).read_text(
        encoding="utf-8"
    ).splitlines()[0]
    assert "compactification_factor" in curvature_header
    assert "emergent_spatial_scale_factor" in curvature_header
    time_slice_header = Path(sidecars["files"]["emergent_curved_spacetime_time_slices_csv"]["path"]).read_text(
        encoding="utf-8"
    ).splitlines()[0]
    assert "max_compactification_factor" in time_slice_header
    assert sidecars["files"]["two_defect_stress_trajectory_csv"]["row_count"] == 1
    assert sidecars["files"]["two_defect_stress_worldlines_csv"]["row_count"] == 1
    assert sidecars["files"]["two_defect_stress_worldline_events_csv"]["row_count"] == 1
    assert sidecars["files"]["free_two_defect_dynamics_trajectory_csv"]["row_count"] == 0
    assert sidecars["files"]["string_vacuum_selector_candidates_csv"]["row_count"] == 6
    assert sidecars["files"]["string_vacuum_selector_gates_csv"]["row_count"] == 9
    assert sidecars["files"]["string_vacuum_selector_critical_edge_gates_csv"]["row_count"] == 9
    assert sidecars["files"]["string_vacuum_selector_operator_safety_csv"]["row_count"] == 10
    assert sidecars["files"]["string_vacuum_selector_quantitative_targets_csv"]["row_count"] == 9
    assert Path(sidecars["files"]["subjective_observer_cameras_csv"]["path"]).exists()
    manifest = json.loads(Path(sidecars["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["schema"] == "oph_universe_visualization_sidecars_v1"
    assert manifest["files"]["screen_full_bin"]["written"] is True
    assert manifest["files"]["visualization_render_data_json"]["written"] is True
    render_sidecar = json.loads(
        Path(manifest["files"]["visualization_render_data_json"]["path"]).read_text(encoding="utf-8")
    )
    assert render_sidecar["availability"]["protoWorldlineEventCount"] == 2
    assert manifest["receipts"]["observer_facing_consensus_3d_bulk_readout_receipt"] is True
    assert manifest["receipts"]["physical_cmb_prediction_receipt"] is False
    assert manifest["receipts"]["reference_vacuum_regression_receipt"] is True
    assert manifest["receipts"]["finite_nonabelian_gauge_gap_diagnostic_receipt"] is True
    assert manifest["receipts"]["yang_mills_gap_reproduced_receipt"] is False
    assert manifest["receipts"]["clay_yang_mills_gap_receipt"] is False
    assert manifest["receipts"]["oph_native_vacuum_promotion_receipt"] is False
    assert manifest["receipts"]["particle_matter_receipt"] is False
    assert manifest["receipts"]["emergent_curved_spacetime_visualization_receipt"] is True
    assert manifest["receipts"]["two_defect_stress_contraction_assay_receipt"] is True
    assert manifest["receipts"]["string_vacuum_selector_visualization_receipt"] is True
    assert manifest["receipts"]["critical_edge_cft_receipt"] is False
    assert manifest["receipts"]["einstein_branch_entry_receipt"] is False
    assert manifest["receipts"]["EINSTEIN_BRANCH_ENTRY_RECEIPT"] is False
    assert manifest["receipts"]["OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1"] is False
    assert manifest["receipts"]["raw_production_gravity_requested"] is False
    assert manifest["receipts"]["production_gravity_receipt"] is False
    assert manifest["receipts"]["paper_accuracy_guard_receipt"] is True
    assert manifest["receipts"]["no_semantic_promotion_by_relabeling_receipt"] is True
    web_brief_path = Path(summary["web_coding_agent_brief_path"])
    assert web_brief_path.exists()
    web_brief = web_brief_path.read_text(encoding="utf-8")
    assert "Canonical pedagogical cinematic storyboard" in web_brief
    assert "observer-facing H3 consensus chart" in web_brief
    assert "COMPUTED / PASSED" in web_brief
    assert "ASSUMED VISUAL LAYER" in web_brief
    assert "Target smooth 60 fps" in web_brief
