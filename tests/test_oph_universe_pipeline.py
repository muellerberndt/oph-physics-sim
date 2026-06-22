import json
from pathlib import Path

from oph_fpe.experiments import load_config
from oph_fpe.pipelines.oph_universe import (
    H3_RESPONSE_EXCLUDED_AUX_OBSERVABLES,
    REFIT_RECIPES,
    _h3_score,
    _cmb_diagnostic_summary,
    _postprocess_observer_experience,
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

    assert set(H3_RESPONSE_EXCLUDED_AUX_OBSERVABLES) <= set(h3_config["exclude_observables"])
    assert h3_config["transform"] == "signed_robust_zscore"
    assert h3_config["control_fit_mode"] == "same_h3_model_not_affine_target_fit"
    assert bw_config["source_state"] == "theorem_observer"
    assert bw_config["state_mode"] == "maxent_record_operator_state"
    assert len(bw_config["times"]) >= 3
    assert theorem_core["consensus_replay"]["enabled"] is True


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
