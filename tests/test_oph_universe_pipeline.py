import json
from pathlib import Path

from oph_fpe.experiments import load_config
from oph_fpe.pipelines.oph_universe import (
    H3_RESPONSE_EXCLUDED_AUX_OBSERVABLES,
    REFIT_RECIPES,
    _h3_score,
    _cmb_diagnostic_summary,
    _postprocess_observer_experience,
    _post_theorem_large_run_readiness,
    refresh_post_theorem_large_run_readiness_report,
    _write_physical_cmb_source_artifacts,
    _write_physical_cmb_transfer_artifacts,
    _write_visualizer_csv_aliases,
    _write_visualization_diagnostic_artifacts,
)


def test_oph_universe_postprocess_splits_3p1d_from_populated_h3():
    report = _postprocess_observer_experience(
        {
            "observer_modular_time_receipt": True,
            "component_gates": {
                "bw_kms_branch_replay_receipt": True,
                "conformal_h3_chart_receipt": True,
            },
        },
        {
            "H3_RESPONSE_CANDIDATE_RECEIPT": True,
        },
        {
            "observer_chart_bulk_population_receipt": False,
        },
    )

    assert report["observer_facing_3p1d_h3_experience_receipt"] is True
    assert report["OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT"] is True
    assert report["observer_facing_populated_h3_experience_receipt"] is False
    assert report["blockers"] == []
    assert report["populated_h3_experience_blockers"] == ["observer_h3_object_population_receipt"]


def test_post_theorem_large_run_readiness_routes_visualization_without_physical_promotion():
    report = _post_theorem_large_run_readiness(
        theorem_contract={
            "paper_geometric_branch_consensus_bulk_emergence_receipt": True,
            "paper_faithful_consensus_bulk_emergence_receipt": True,
            "strict_neutral_bulk_contract_receipt": False,
            "strict_neutral_blockers": ["L7_refinement_naturality"],
        },
        proof={"physical_cmb_prediction": False, "bulk_3d_established_chart_blind_strict_neutral": False},
        readout={"observer_facing_consensus_3d_bulk_readout_receipt": True},
        frontier_artifacts={
            "physical_cmb_prediction_receipt": False,
            "physical_cmb_blockers": ["finite_covariant_parent_receipt_missing"],
            "reference_vacuum_regression_receipt": True,
            "oph_native_vacuum_promotion_receipt": False,
            "two_defect_stress_contraction_assay_receipt": True,
            "production_gravity_receipt": False,
        },
        cmb_diagnostics={"screen_proxy_cmb_receipt": True},
    )

    assert report["recommended_large_run_lane"] == (
        "observer_facing_visualization_large_run_with_diagnostic_cmb_vacuum"
    )
    assert report["cloud_run_safe_for_visualization_data"] is True
    assert report["cloud_run_safe_for_physical_cmb_prediction"] is False
    assert report["cloud_run_safe_for_strict_neutral_bulk_claim"] is False
    assert report["lanes"]["screen_cmb_proxy"]["scale_candidate"] is True
    assert "finite_covariant_parent_receipt_missing" in report["blockers"]
    assert "production_gravity_receipt_false" in report["blockers"]


def test_refresh_large_run_readiness_uses_post_theorem_summary(tmp_path: Path):
    expected = {
        "mode": "post_theorem_large_run_readiness_v0",
        "recommended_large_run_lane": "observer_facing_visualization_large_run_with_diagnostic_cmb_vacuum",
        "cloud_run_safe_for_visualization_data": True,
    }
    (tmp_path / "AUTO_THEOREM_UNIVERSE_SUMMARY.json").write_text(
        json.dumps({"post_theorem_large_run_readiness": expected}),
        encoding="utf-8",
    )
    (tmp_path / "large_run_readiness_report.json").write_text(
        json.dumps({"mode": "legacy_base_preflight"}),
        encoding="utf-8",
    )

    report = refresh_post_theorem_large_run_readiness_report(tmp_path)

    assert report == expected
    written = json.loads((tmp_path / "large_run_readiness_report.json").read_text(encoding="utf-8"))
    assert written == expected


def test_oph_universe_h3_score_uses_material_gate_value_before_raw_count():
    raw_count_noisy = {
        "candidate_receipt": True,
        "control_separation_receipt": True,
        "signal_gate": True,
        "geometry_gate": True,
        "aggregate_wrong_scale_gate": True,
        "material_feature_gate": True,
        "material_wrong_scale_win_fraction": 0.40,
        "material_wrong_scale_gate_value": 0.01,
        "heldout_explained_variance": 0.12,
        "heldout_normalized_rmse": 0.94,
        "feature_count": 272,
    }
    superficially_cleaner_raw_count = {
        **raw_count_noisy,
        "candidate_receipt": False,
        "material_feature_gate": False,
        "material_wrong_scale_win_fraction": 0.04,
        "material_wrong_scale_gate_value": 0.08,
        "heldout_explained_variance": 0.16,
        "heldout_normalized_rmse": 0.90,
        "feature_count": 2232,
    }

    assert _h3_score(raw_count_noisy) > _h3_score(superficially_cleaner_raw_count)


def test_oph_universe_h3_score_prefers_theorem_clean_response_policy():
    stale_baseline = {
        "candidate_receipt": True,
        "control_separation_receipt": True,
        "signal_gate": True,
        "geometry_gate": True,
        "aggregate_wrong_scale_gate": True,
        "material_feature_gate": True,
        "theorem_clean_feature_policy": False,
        "material_wrong_scale_gate_value": 0.01,
        "heldout_explained_variance": 0.20,
        "heldout_normalized_rmse": 0.90,
        "feature_count": 512,
    }
    theorem_clean_refit = {
        **stale_baseline,
        "theorem_clean_feature_policy": True,
        "material_wrong_scale_gate_value": 0.03,
        "heldout_explained_variance": 0.12,
        "heldout_normalized_rmse": 0.95,
        "feature_count": 272,
    }

    assert _h3_score(theorem_clean_refit) > _h3_score(stale_baseline)


def test_oph_universe_refit_recipes_exclude_auxiliary_h3_response_labels():
    excluded = set(H3_RESPONSE_EXCLUDED_AUX_OBSERVABLES)

    for recipe in REFIT_RECIPES:
        assert excluded <= set(recipe.kwargs["exclude_observables"])


def test_oph_universe_object_chart_config_excludes_auxiliary_h3_response_labels():
    repo = Path(__file__).resolve().parents[1]
    config = load_config(repo / "configs" / "e4_shared_observer_bulk_64k_object_chart.yml")
    h3_config = config["h3_modular_response"]
    bw_config = config["bw"]
    theorem_core = config["theorem_core"]
    cosmology = config["cosmology"]

    assert set(H3_RESPONSE_EXCLUDED_AUX_OBSERVABLES) <= set(h3_config["exclude_observables"])
    assert h3_config["transform"] == "signed_robust_zscore"
    assert h3_config["control_fit_mode"] == "same_h3_model_not_affine_target_fit"
    assert bw_config["source_state"] == "theorem_observer"
    assert bw_config["state_mode"] == "history_koopman_generator_state"
    assert bw_config["history_window"] >= 16
    assert len(bw_config["times"]) >= 3
    assert config["observer_chart_population"]["incidence_mode"] == "transition_history_mixture_cluster"
    assert theorem_core["consensus_replay"]["enabled"] is True
    assert cosmology["b_a_paired_perturbation"]["enabled"] is True
    assert cosmology["b_a_paired_perturbation"]["source_state"] == "theorem_observer"


def test_oph_universe_cmb_diagnostic_summary_splits_screen_from_physical(tmp_path):
    (tmp_path / "cl_comparison_report.json").write_text(
        json.dumps(
            {
                "receipt_name": "SCREEN_PROXY_CMB_RECEIPT",
                "cosmo_proxy_receipt": {"receipt": True},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "cmb_lite_comparison_report.json").write_text(
        json.dumps(
            {
                "best_shape_field": "record_signature",
                "best_positive_shape_field": "record_signature",
                "field_comparisons": {
                    "record_signature": {
                        "usable_positive_shape": True,
                        "real_ell_physical_comparison": {"usable": False},
                        "overlap_ell_physical_comparison": {"usable": False},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = _cmb_diagnostic_summary(tmp_path)

    assert report["screen_proxy_cmb_receipt"] is True
    assert report["cmb_lite_shape_comparison_receipt"] is True
    assert report["cmb_lite_real_ell_physical_comparison_receipt"] is False
    assert report["cmb_lite_positive_shape_field_count"] == 1


def test_oph_universe_writes_stable_visualizer_csv_aliases(tmp_path):
    run_dir = tmp_path / "run"
    readout_dir = run_dir / "observer_consensus_bulk"
    readout_dir.mkdir(parents=True)
    (readout_dir / "consensus_h3_object_rows.csv").write_text(
        "object_id,h3_spatial_point,observer_count\n"
        'obj0,"[0.1, 0.2, 0.3]",4\n',
        encoding="utf-8",
    )
    (readout_dir / "observer_perspective_rows.csv").write_text(
        "observer_id,axis\n"
        '1,"[1.0, 0.0, 0.0]"\n',
        encoding="utf-8",
    )

    report = _write_visualizer_csv_aliases(run_dir, readout_dir)

    assert report["h3_objects.csv"]["written"] is True
    assert report["observer_perspective_rows.csv"]["written"] is True
    assert (run_dir / "h3_objects.csv").read_text(encoding="utf-8").startswith("object_id")
    assert (run_dir / "observer_perspective_rows.csv").exists()


def test_oph_universe_writes_visualization_diagnostic_artifacts(tmp_path):
    report = _write_visualization_diagnostic_artifacts(
        tmp_path,
        {
            "visualization_diagnostics": {
                "reference_vacuum": {
                    "ell_max": 8,
                    "sample_count": 64,
                    "coarse_ell_max": 4,
                    "u1_lattice_size": 3,
                    "u1_sweeps": 8,
                },
                "two_defect_gravity_assay": {
                    "patch_count": 512,
                    "steps": 12,
                    "support_node_count": 6,
                },
            }
        },
    )

    assert report["reference_vacuum_baseline_written"] is True
    assert (tmp_path / "reference_vacuum_baseline" / "reference_vacuum_baseline_report.json").exists()
    assert report["oph_native_vacuum_promotion_receipt"] is False
    assert report["oph_primordial_field_promotion_receipt"] is False
    assert report["two_defect_stress_contraction_assay_written"] is True
    assert (tmp_path / "two_defect_stress_contraction_assay_report.json").exists()
    assert report["two_defect_stress_contraction_assay_receipt"] is True
    assert report["gravity_like_attraction_diagnostic_receipt"] is True
    assert report["production_gravity_receipt"] is False
    assert report["physical_gravity_prediction"] is False


def test_oph_universe_writes_fail_closed_physical_cmb_source_artifacts(tmp_path, monkeypatch):
    calls = []

    def writer(name, filename):
        def _write(*args):
            calls.append(name)
            (tmp_path / filename).write_text(json.dumps({"writer": name}), encoding="utf-8")
            return {"writer": name}

        return _write

    monkeypatch.setattr(
        "oph_fpe.pipelines.oph_universe.write_finite_repair_transition_clock_report",
        writer("finite_transition", "finite_repair_transition_matrix_report.json"),
    )
    monkeypatch.setattr(
        "oph_fpe.pipelines.oph_universe.write_run_proxy_finite_certificate_bundle",
        writer("finite_certificate", "finite_certificate_report.json"),
    )
    monkeypatch.setattr(
        "oph_fpe.pipelines.oph_universe.ba_kernel_report_from_parent_report",
        writer("B_A_kernel", "B_A_kernel_report.json"),
    )
    monkeypatch.setattr(
        "oph_fpe.pipelines.oph_universe.write_compressed_likelihood_reference_report",
        writer("compressed_likelihood", "oph_compressed_likelihood_report.json"),
    )
    monkeypatch.setattr(
        "oph_fpe.pipelines.oph_universe.write_official_planck_readiness_report",
        writer("official_likelihood", "official_planck_likelihood_readiness_report.json"),
    )
    (tmp_path / "b_a_parent_report.json").write_text("{}", encoding="utf-8")

    report = _write_physical_cmb_source_artifacts(tmp_path)

    assert report["finite_repair_transition_clock_written"] is True
    assert report["finite_certificate_report_written"] is True
    assert report["B_A_kernel_report_written"] is True
    assert report["compressed_likelihood_reference_written"] is True
    assert report["official_planck_likelihood_readiness_written"] is True
    assert report["physical_cmb_source_artifact_errors"] == []
    assert set(calls) == {
        "finite_transition",
        "finite_certificate",
        "B_A_kernel",
        "compressed_likelihood",
        "official_likelihood",
    }


def test_oph_universe_writes_physical_cmb_transfer_artifacts_when_sources_exist(tmp_path, monkeypatch):
    calls = []
    benchmark = tmp_path / "planck.txt"
    benchmark.write_text("# l Dl -dDl +dDl BestFit\n50 1000 10 10 990\n", encoding="utf-8")
    (tmp_path / "finite_repair_transition_matrix_report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "scale_compressed_repair_report.json").write_text("{}", encoding="utf-8")

    def writer(name, filename):
        def _write(*args, **kwargs):
            calls.append((name, args, kwargs))
            (tmp_path / filename).write_text(json.dumps({"writer": name}), encoding="utf-8")
            return {"writer": name}

        return _write

    monkeypatch.setattr(
        "oph_fpe.pipelines.oph_universe.write_camb_lcdm_baseline_report",
        writer("baseline", "camb_lcdm_baseline_report.json"),
    )
    monkeypatch.setattr(
        "oph_fpe.pipelines.oph_universe.write_scale_compressed_cmb_camb_report",
        writer("scale", "scale_compressed_cmb_camb_report.json"),
    )
    monkeypatch.setattr(
        "oph_fpe.pipelines.oph_universe.write_finite_repair_clock_cmb_camb_report",
        writer("finite_clock", "finite_repair_clock_cmb_camb_report.json"),
    )

    report = _write_physical_cmb_transfer_artifacts(
        tmp_path,
        {
            "cosmology": {
                "physical_cmb_transfer": {
                    "benchmark_path": str(benchmark),
                    "benchmark_label": "test_planck",
                    "lmax": 64,
                }
            }
        },
    )

    assert report["physical_cmb_transfer_benchmark_found"] is True
    assert report["camb_lcdm_baseline_written"] is True
    assert report["scale_compressed_cmb_camb_written"] is True
    assert report["finite_repair_clock_cmb_camb_written"] is True
    assert report["physical_cmb_transfer_errors"] == []
    assert [call[0] for call in calls] == ["baseline", "scale", "finite_clock"]


def test_oph_universe_physical_cmb_transfer_reports_missing_benchmark(tmp_path):
    report = _write_physical_cmb_transfer_artifacts(
        tmp_path,
        {"cosmology": {"physical_cmb_transfer": {"benchmark_path": str(tmp_path / "missing.txt")}}},
    )

    assert report["physical_cmb_transfer_benchmark_found"] is False
    assert report["camb_lcdm_baseline_written"] is False
    assert report["physical_cmb_transfer_errors"][0].startswith("benchmark_missing:")
