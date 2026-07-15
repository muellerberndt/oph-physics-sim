import json
from pathlib import Path

import numpy as np

from oph_fpe.viz import (
    write_cmb_neutral_frontier_viewer,
    write_cmb_static_plots,
    write_object_h3_bulk_viewer,
    write_run_viewer,
)


def test_run_viewer_writes_html_with_gate_boundary(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "viewer_smoke", "patch_count": 16}),
        encoding="utf-8",
    )
    (run_dir / "emergence_status_report.json").write_text(
        json.dumps(
            {
                "support_visible_lorentz_3p1_kinematics_receipt": True,
                "conformal_h3_spatial_chart_receipt": True,
                "bulk_3d_established": False,
                "defect_cluster_h3_support_receipt": True,
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        run_dir / "freezeout_fields.npz",
        points=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        record_signature=np.array([0.0, 1.0, 2.0], dtype=np.float32),
        cell_area_planck=np.ones(3, dtype=np.float32),
        cell_entropy=np.ones(3, dtype=np.float32),
    )
    (run_dir / "observer_views.jsonl").write_text(
        json.dumps({"axis": [1.0, 0.0, 0.0], "visible_signature_entropy": 0.5}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "array_holonomy_report.json").write_text(
        json.dumps(
            {
                "clusters": [
                    {
                        "cluster_id": "d0",
                        "centroid": [0.0, 1.0, 0.0],
                        "class": "threecycle",
                        "support_node_count": 4,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "defect_timeline_report.json").write_text(
        json.dumps(
            {
                "persistent_worldline_count": 1,
                "snapshots": [
                    {
                        "cycle": 0,
                        "cluster_count": 1,
                        "clusters": [
                            {
                                "cluster_id": "d0",
                                "centroid": [0.0, 1.0, 0.0],
                                "class": "threecycle",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "defect_cluster_h3_report.json").write_text(
        json.dumps({"h3_fit": {"sample_fitted_h3_points": [[1.1, 0.2, 0.3, 0.4]]}}),
        encoding="utf-8",
    )
    (run_dir / "modular_response_h3_report.json").write_text(
        json.dumps({"h3_fit": {"sample_fitted_h3_points": [[0.1, 0.2, 0.3]]}}),
        encoding="utf-8",
    )
    (run_dir / "defect_h3_worldlines_report.json").write_text(
        json.dumps(
            {
                "worldlines": [
                    {
                        "worldline_id": "w0",
                        "observation_count": 2,
                        "events": [
                            {"cycle": 0, "h3_spatial_point": [0.1, 0.2, 0.3]},
                            {"cycle": 1, "h3_spatial_point": [0.2, 0.3, 0.4]},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = write_run_viewer(run_dir)

    html_path = Path(summary["viewer_path"])
    assert html_path.exists()
    text = html_path.read_text(encoding="utf-8")
    assert "OPH-FPE Receipt Viewer" in text
    assert "Diagnostic viewer only" in text
    assert summary["bulk_3d_established"] is False
    assert summary["defect_timeline_snapshots"] == 1
    assert summary["persistent_defect_worldlines"] == 1
    assert summary["h3_point_count"] == 2
    assert summary["h3_worldline_count"] == 1


def test_object_h3_bulk_viewer_writes_object_cloud(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "h3_objects.csv").write_text(
        "object_id,h3_spatial_point,observer_count,h3_compactness,s2_boundary_compactness\n"
        'obj0,"[0.1, 0.2, 0.3]",4,0.2,0.7\n'
        'obj1,"[-0.4, 0.3, 0.2]",9,0.4,0.6\n',
        encoding="utf-8",
    )
    (run_dir / "observer_chart_object_h3_lineage_report.json").write_text(
        json.dumps(
            {
                "object_count": 2,
                "localized_object_count": 2,
                "THEOREM_ASSISTED_H3_OBJECT_PREVIEW_RECEIPT": True,
                "observer_chart_bulk_population_receipt": True,
            }
        ),
        encoding="utf-8",
    )

    summary = write_object_h3_bulk_viewer(run_dir)

    viewer = Path(summary["viewer_path"])
    assert viewer.exists()
    assert "OPH Object-H3 Viewer" in viewer.read_text(encoding="utf-8")
    assert summary["object_count"] == 2
    assert summary["theorem_assisted_h3_bulk"] is True
    assert (viewer.parent / "object_h3_bulk_viewer_summary.json").exists()


def test_object_h3_bulk_viewer_reads_split_h3_object_columns(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "h3_objects.csv").write_text(
        "object_id,observer_count,support_size,h3_compactness,h3_compactness_normalized,h3_x,h3_y,h3_z\n"
        "obj0,4,12,0.2,0.5,0.1,0.2,0.3\n"
        "obj1,9,22,0.4,0.7,-0.4,0.3,0.2\n",
        encoding="utf-8",
    )

    summary = write_object_h3_bulk_viewer(run_dir)

    assert summary["source_path"].endswith("h3_objects.csv")
    assert summary["object_count"] == 2


def test_cmb_neutral_frontier_viewer_writes_gate_and_cmb_summary(tmp_path: Path):
    run_dir = tmp_path / "pack"
    run_dir.mkdir()
    (run_dir / "claims.json").write_text(
        json.dumps(
            {
                "WORKING_MINI_UNIVERSE_V0": True,
                "physical_cmb_prediction": False,
                "physical_cmb_promotion_ready": False,
                "physical_cmb_promotion_blocker_count": 2,
                "strict_neutral_bulk": False,
                "theorem_assisted_h3_bulk": True,
                "object_h3_bulk_viewer_object_count": 2,
                "object_h3_bulk_viewer_observer_overlap_link_count": 7,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "physical_cmb_input_report.json").write_text(
        json.dumps(
            {
                "PHYSICAL_CMB_INPUT_CONTRACT_RECEIPT": False,
                "blockers": ["A_zeta_not_finite_derived"],
                "input_status": {
                    "A_zeta": {
                        "source": "diagnostic_proxy",
                        "diagnostic_value_present": True,
                        "physical_gate_passed": False,
                    },
                    "eta_R": {
                        "source": "finite_repair_transition_clock",
                        "diagnostic_value_present": True,
                        "physical_gate_passed": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "physical_cmb_promotion_audit_report.json").write_text(
        json.dumps(
            {
                "physical_cmb_promotion_ready": False,
                "contract_blockers": ["A_zeta_not_finite_derived"],
                "promotion_blockers": ["official_likelihood_not_ready"],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "physical_cmb_frontier_report.json").write_text(
        json.dumps(
            {
                "PHYSICAL_CMB_FRONTIER_REPORT": True,
                "physical_cmb_prediction_ready": False,
                "physical_cmb_output_comparison_receipt": True,
                "physical_cmb_prediction_receipt": False,
                "gate_rows": [
                    {
                        "gate": "measurement_comparable_cmb_outputs",
                        "passed": True,
                        "detail": "2 OPH diagnostic models",
                    },
                    {
                        "gate": "finite_theorem_A_zeta",
                        "passed": False,
                        "detail": "source=diagnostic_proxy",
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
                        "current_evidence": "diagnostic proxy",
                        "action_surface": "finite certificate stack",
                    }
                ],
                "blockers": ["A_zeta_not_finite_derived", "official_likelihood_not_ready"],
                "next_missing_receipts": [
                    {
                        "blocker": "official_likelihood_not_ready",
                        "next_step": "Connect official likelihood execution.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "physical_cmb_output_comparison_report.json").write_text(
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
                    "mean_abs_peak_height_fractional_delta": 0.033,
                },
                "rows": [
                    {
                        "model_id": "scale_compressed_ir_kernel",
                        "model_role": "oph_diagnostic",
                        "measurement_comparable": True,
                        "bin_count": 2,
                        "shape_correlation": 0.999,
                        "amplitude_fit_chi2_per_bin": 0.94,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "physical_cmb_peak_features.csv").write_text(
        "model_id,model_role,peak_index,observed_peak_ell,model_peak_ell,ell_delta,fractional_D_ell_delta\n"
        "scale_compressed_ir_kernel,oph_diagnostic,1,80,80,0,0.033\n",
        encoding="utf-8",
    )
    (run_dir / "official_planck_likelihood_readiness_report.json").write_text(
        json.dumps(
            {
                "official_likelihood_execution_ready": False,
                "official_planck_likelihood_data_paths_configured": False,
                "official_clik_api_available": False,
                "camb_available": True,
                "cobaya_available": False,
                "blockers": ["official_planck_likelihood_data_path_not_configured"],
                "data_paths": [
                    {
                        "env_var": "OPH_PLANCK_LIKELIHOOD_DIR",
                        "configured": False,
                        "exists": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "neutral_3d_bulk_audit_report.json").write_text(
        json.dumps(
            {
                "strict_neutral_bulk_ready": False,
                "strict_neutral_bulk": False,
                "directional_strict_ready_total": 0,
                "control_quotient_candidate_count": 1,
                "overlap_native_negative_control_report_count": 1,
                "overlap_native_negative_control_receipt_count": 1,
                "blockers": ["directional_h3_strict_rank_gate_not_passed"],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "strict_neutral_bulk_frontier_report.json").write_text(
        json.dumps(
            {
                "STRICT_NEUTRAL_BULK_FRONTIER_REPORT": True,
                "strict_neutral_bulk_ready": False,
                "control_residualized_rank3_refinement_candidate": True,
                "overlap_native_negative_control_receipt_all": True,
                "neutral_independent_rank3_selector_receipt": False,
                "gate_rows": [
                    {
                        "gate": "control_residualized_rank3_refinement_candidate",
                        "passed": True,
                        "detail": "rank-3 diagnostic candidate",
                    },
                    {
                        "gate": "independent_rank3_selector",
                        "passed": False,
                        "detail": "selector did not pick rank 3",
                    },
                ],
                "gate_gap_rows": [
                    {
                        "gate": "independent_rank3_selector",
                        "missing_receipt": "independent rank-3 selector",
                        "current_evidence": "rank-3 candidate only",
                        "action_surface": "neutral rank selector audit",
                    }
                ],
                "blockers": ["independent_svd_rank3_selector_not_stable_or_false"],
                "next_missing_receipts": ["independent rank-3 selector receipt"],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "overlap_native_neutral_control_report.json").write_text(
        json.dumps(
            {
                "OVERLAP_NATIVE_NEGATIVE_CONTROL_RECEIPT": True,
                "overlap_native_spatial_3d_candidate": False,
                "overlap_native_strict_h3_candidate": False,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "overlap_native_graph_geometry_report.json").write_text(
        json.dumps(
            {
                "OVERLAP_NATIVE_GRAPH_GEOMETRY_RECEIPT": True,
                "overlap_graph_spatial_3d_candidate": False,
                "overlap_graph_strict_h3_candidate": False,
                "rank_selection": {"rank3_selector_receipt": False},
                "graph_summary": {"edge_count": 12, "component_count": 1},
                "blockers": ["overlap_graph_not_spatial_3d_candidate"],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "overlap_native_graph_geometry_sweep_report.json").write_text(
        json.dumps(
            {
                "OVERLAP_NATIVE_GRAPH_GEOMETRY_SWEEP_RECEIPT": True,
                "case_count": 4,
                "graph_geometry_receipt_count": 4,
                "spatial_3d_candidate_count": 1,
                "strict_h3_candidate_count": 0,
                "rank3_selector_count": 0,
                "rank_obstruction_summary": {
                    "available": True,
                    "case_count": 4,
                    "primary_obstruction": "no_independent_rank3_selector",
                    "dominant_largest_gap_rank": "2",
                    "rank3_selector_count": 0,
                    "nontrivial_rank3_selector_count": 2,
                    "largest_gap_rank_counts": {"2": 3, "3": 1},
                    "nontrivial_largest_gap_rank_counts": {"3": 2, "4": 2},
                    "max_nontrivial_rank3_cumulative_explained_variance": 0.55,
                    "median_nontrivial_effective_rank": 8.5,
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
    (run_dir / "overlap_residualized_graph_geometry_report.json").write_text(
        json.dumps(
            {
                "OVERLAP_RESIDUALIZED_GRAPH_GEOMETRY_RECEIPT": True,
                "overlap_residual_graph_spatial_3d_candidate": True,
                "overlap_residual_graph_strict_h3_candidate": False,
                "rank_selection": {"rank3_selector_receipt": False},
                "graph_summary": {"edge_count": 10, "component_count": 1},
                "blockers": ["overlap_residual_graph_not_strict_h3_candidate"],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "overlap_residualized_graph_geometry_sweep_report.json").write_text(
        json.dumps(
            {
                "OVERLAP_RESIDUALIZED_GRAPH_GEOMETRY_SWEEP_RECEIPT": True,
                "case_count": 5,
                "residual_graph_receipt_count": 5,
                "spatial_3d_candidate_count": 2,
                "strict_h3_candidate_count": 0,
                "rank3_selector_count": 1,
                "best_case": {
                    "source_run_dir": "run-a",
                    "seed": 3,
                    "max_model_points": 32,
                    "k_neighbors": 8,
                    "remove_modes": 1,
                    "median_dimension": 3.1,
                    "selected_model": "H3",
                },
                "rank_obstruction_summary": {
                    "available": True,
                    "case_count": 5,
                    "primary_obstruction": "residualized_no_independent_rank3_selector",
                    "dominant_largest_gap_rank": "2",
                    "rank3_selector_count": 1,
                    "nontrivial_rank3_selector_count": 2,
                    "largest_gap_rank_counts": {"2": 4, "3": 1},
                    "nontrivial_largest_gap_rank_counts": {"3": 2, "4": 3},
                    "raw_largest_gap_rank1_count": 4,
                    "max_rank3_cumulative_explained_variance": 0.6,
                    "max_nontrivial_rank3_cumulative_explained_variance": 0.58,
                    "median_effective_rank": 8.0,
                    "median_nontrivial_effective_rank": 7.8,
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
    (run_dir / "object_h3_bulk_viewer_summary.json").write_text(
        json.dumps(
            {
                "object_count": 2,
                "reported_object_count": 2,
                "observer_overlap_link_count": 7,
                "fundamental_operation": "Overlapping observations by observers.",
                "dot_semantics": "Object packets.",
                "color_encoding": "H3 compactness.",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "camb_lcdm_tt_bins.csv").write_text(
        "ell,observed_D_ell,sigma_D_ell,camb_D_ell\n50,1000,10,990\n80,1200,12,1190\n",
        encoding="utf-8",
    )
    (run_dir / "finite_repair_clock_cmb_tt_bins.csv").write_text(
        "ell,observed_D_ell,finite_repair_clock_plus_selector_ir_D_ell\n50,1000,970\n80,1200,1180\n",
        encoding="utf-8",
    )
    (run_dir / "scale_compressed_cmb_tt_bins.csv").write_text(
        "ell,observed_D_ell,scale_compressed_ir_kernel_D_ell\n50,1000,980\n80,1200,1185\n",
        encoding="utf-8",
    )

    summary = write_cmb_neutral_frontier_viewer(run_dir)

    viewer = Path(summary["viewer_path"])
    assert viewer.exists()
    text = viewer.read_text(encoding="utf-8")
    assert "OPH CMB and Neutral Bulk Frontier" in text
    assert "CMB Output Metrics" in text
    assert "Physical CMB Frontier" in text
    assert "Hard-gate gaps" in text
    assert "Official Planck Likelihood Readiness" in text
    assert "scale_compressed_ir_kernel" in text
    assert "best OPH RMS sigma residual" in text
    assert "OPH Peak Features" in text
    assert "measurement_comparable_cmb_outputs" in text
    assert "overlap graph geometry receipt" in text
    assert "overlap graph sweep strict H3 candidates" in text
    assert "Residualized overlap graph sweep" in text
    assert "Closest strict candidates" in text
    assert "Closest residualized strict candidates" in text
    assert "residualized graph sweep rank-3 selectors" in text
    assert "nontrivial rank-3 selector count" in text
    assert summary["tt_bin_count"] == 2
    assert summary["physical_cmb_prediction"] is False
    assert summary["physical_cmb_frontier_written"] is True
    assert summary["physical_cmb_frontier_ready"] is False
    assert summary["physical_cmb_frontier_gate_count"] == 3
    assert summary["physical_cmb_frontier_gap_count"] == 1
    assert summary["physical_cmb_frontier_blocker_count"] == 2
    assert summary["physical_cmb_output_comparison_receipt"] is True
    assert summary["physical_cmb_output_prediction_receipt"] is False
    assert summary["physical_cmb_output_best_oph_chi2_per_bin"] == 0.94
    assert summary["physical_cmb_output_best_oph_residual_bin_count"] == 2
    assert summary["physical_cmb_output_best_oph_rms_sigma_residual"] == 1.5
    assert summary["physical_cmb_output_best_oph_max_abs_sigma_residual"] == 2.0
    assert summary["physical_cmb_output_best_oph_peak_count"] == 1
    assert summary["physical_cmb_output_best_oph_mean_abs_peak_ell_delta"] == 0.0
    assert summary["physical_cmb_output_best_oph_mean_abs_peak_height_fractional_delta"] == 0.033
    assert summary["official_planck_likelihood_readiness_written"] is True
    assert summary["official_planck_likelihood_execution_ready"] is False
    assert summary["official_planck_likelihood_data_paths_configured"] is False
    assert summary["official_planck_clik_api_available"] is False
    assert summary["official_planck_likelihood_blocker_count"] == 1
    assert summary["strict_neutral_bulk_frontier_written"] is True
    assert summary["strict_neutral_bulk_frontier_ready"] is False
    assert summary["strict_neutral_bulk_frontier_gate_count"] == 2
    assert summary["strict_neutral_bulk_frontier_gap_count"] == 1
    assert summary["overlap_native_negative_control_receipt"] is True
    assert summary["overlap_native_graph_geometry_receipt"] is True
    assert summary["overlap_native_graph_spatial_3d_candidate"] is False
    assert summary["overlap_native_graph_sweep_case_count"] == 4
    assert summary["overlap_native_graph_sweep_spatial_3d_candidate_count"] == 1
    assert summary["overlap_native_graph_sweep_strict_h3_candidate_count"] == 0
    assert summary["overlap_native_graph_sweep_rank3_selector_count"] == 0
    assert summary["overlap_native_graph_sweep_closest_strict_candidate_count"] == 1
    assert summary["overlap_native_graph_sweep_nontrivial_rank3_selector_count"] == 2
    assert summary["overlap_residualized_graph_sweep_case_count"] == 5
    assert summary["overlap_residualized_graph_sweep_spatial_3d_candidate_count"] == 2
    assert summary["overlap_residualized_graph_sweep_strict_h3_candidate_count"] == 0
    assert summary["overlap_residualized_graph_sweep_rank3_selector_count"] == 1
    assert summary["overlap_residualized_graph_sweep_closest_strict_candidate_count"] == 1
    assert summary["overlap_residualized_graph_sweep_nontrivial_rank3_selector_count"] == 2


def test_cmb_static_plots_write_measurement_and_gate_pngs(tmp_path: Path):
    run_dir = tmp_path / "pack"
    run_dir.mkdir()
    (run_dir / "claims.json").write_text(
        json.dumps({"physical_cmb_prediction": False, "strict_neutral_bulk": False}),
        encoding="utf-8",
    )
    (run_dir / "physical_cmb_output_comparison_report.json").write_text(
        json.dumps(
            {
                "PHYSICAL_CMB_OUTPUT_COMPARISON_RECEIPT": True,
                "best_oph_diagnostic_model": {
                    "model_id": "scale_compressed_scalar_tilt",
                    "amplitude_fit_chi2_per_bin": 0.95,
                },
                "best_oph_residual_summary": {
                    "bin_count": 3,
                    "rms_sigma_residual": 1.1,
                },
                "best_oph_peak_feature_summary": {
                    "available": True,
                    "peak_count": 1,
                    "mean_abs_peak_ell_delta": 0.0,
                    "mean_abs_peak_height_fractional_delta": 0.02,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "overlap_residualized_graph_geometry_sweep_report.json").write_text(
        json.dumps(
            {
                "case_count": 6,
                "gate_coincidence_summary": {
                    "spatial_h3_geometry_count": 2,
                    "nontrivial_rank3_selector_count": 3,
                    "spatial_h3_nontrivial_rank3_selector_count": 0,
                    "strict_h3_candidate_count": 0,
                },
                "rows": [
                    {
                        "source_run_dir": "runs/a",
                        "seed": 11,
                        "max_model_points": 32,
                        "k_neighbors": 14,
                        "residual_graph_receipt": True,
                        "spatial_3d_candidate": True,
                        "strict_h3_candidate": False,
                        "selected_model": "H3",
                        "h3_beats_s2": True,
                        "h3_beats_h2_h4": True,
                        "s2_leakage_pass": True,
                        "rank3_selector": False,
                        "nontrivial_rank3_selector": True,
                        "median_dimension": 2.94,
                        "rank3_cumulative_explained_variance": 0.41,
                        "nontrivial_rank3_cumulative_explained_variance": 0.53,
                        "largest_gap_rank": 1,
                        "nontrivial_largest_gap_rank": 3,
                        "effective_rank": 9.1,
                        "nontrivial_effective_rank": 6.5,
                    },
                    {
                        "source_run_dir": "runs/b",
                        "seed": 12,
                        "max_model_points": 48,
                        "k_neighbors": 18,
                        "residual_graph_receipt": True,
                        "spatial_3d_candidate": False,
                        "strict_h3_candidate": False,
                        "selected_model": "H4",
                        "rank3_selector": False,
                        "nontrivial_rank3_selector": True,
                        "median_dimension": 4.1,
                        "nontrivial_rank3_cumulative_explained_variance": 0.58,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "camb_lcdm_tt_bins.csv").write_text(
        "ell,observed_D_ell,sigma_D_ell,amplitude_fit_camb_D_ell\n"
        "50,1000,10,990\n"
        "80,1200,12,1190\n"
        "110,1300,13,1280\n",
        encoding="utf-8",
    )
    (run_dir / "scale_compressed_cmb_tt_bins.csv").write_text(
        "ell,observed_D_ell,scale_compressed_scalar_tilt_D_ell,scale_compressed_ir_kernel_D_ell\n"
        "50,1000,980,975\n"
        "80,1200,1185,1180\n"
        "110,1300,1290,1288\n",
        encoding="utf-8",
    )
    (run_dir / "finite_repair_clock_cmb_tt_bins.csv").write_text(
        "ell,observed_D_ell,finite_repair_clock_scalar_tilt_D_ell\n"
        "50,1000,970\n"
        "80,1200,1175\n"
        "110,1300,1270\n",
        encoding="utf-8",
    )
    (run_dir / "physical_cmb_best_oph_residuals.csv").write_text(
        "ell,residual_sigma\n50,-1.0\n80,0.2\n110,1.5\n",
        encoding="utf-8",
    )
    (run_dir / "physical_cmb_output_comparison_rows.csv").write_text(
        "model_id,model_role,measurement_comparable\n"
        "scale_compressed_scalar_tilt,oph_diagnostic,True\n",
        encoding="utf-8",
    )
    (run_dir / "physical_cmb_peak_features.csv").write_text(
        "model_id,model_role,peak_index,observed_peak_ell,model_peak_ell,ell_delta,fractional_D_ell_delta\n"
        "scale_compressed_scalar_tilt,oph_diagnostic,1,80,80,0,0.02\n",
        encoding="utf-8",
    )

    summary = write_cmb_static_plots(run_dir)

    assert summary["physical_cmb_prediction"] is False
    assert summary["physical_cmb_output_comparison_receipt"] is True
    assert summary["best_oph_model"] == "scale_compressed_scalar_tilt"
    assert summary["best_oph_peak_count"] == 1
    assert summary["best_oph_mean_abs_peak_ell_delta"] == 0.0
    assert summary["best_oph_mean_abs_peak_height_fractional_delta"] == 0.02
    assert summary["residual_sweep_spatial_h3_count"] == 2
    assert summary["residual_sweep_spatial_h3_nontrivial_rank3_coincidence_count"] == 0
    assert summary["strict_neutral_near_miss_count"] == 2
    assert summary["strict_neutral_best_near_miss_gate_score"] == 6
    assert summary["strict_neutral_best_near_miss_nontrivial_rank3_ev"] == 0.53
    for name in summary["files"]:
        path = run_dir / name
        assert path.exists()
        assert path.stat().st_size > 0
