from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.bulk.theorem_contract import finite_oph_theorem_contract_report


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_finite_theorem_contract_blocks_branch_replay_without_endogenous_contract(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "theorem_core_receipts.json",
        {
            "FINITE_CONSENSUS_THEOREM_RECEIPT": False,
            "finite_consensus_missing_evidence": ["local_diamond_violation_count"],
        },
    )
    _write_json(
        run / "emergence_status_report.json",
        {
            "BW_KMS_DIRECT_2PI_RECEIPT": True,
            "BW_KMS_BRANCH_REPLAY_RECEIPT": True,
            "ENDOGENOUS_MODULAR_GENERATOR_RECEIPT": False,
        },
    )
    _write_json(run / "conformal_h3_spatial_chart_report.json", {"lorentz_algebra_receipt": True})
    _write_json(
        run / "modular_response_h3_report.json",
        {
            "H3_RESPONSE_CANDIDATE_RECEIPT": False,
            "h3_response_stage_gates": {
                "signal_gate": True,
                "geometry_gate": True,
                "aggregate_wrong_scale_gate": True,
                "material_feature_gate": False,
                "material_wrong_scale_win_fraction": 0.08,
            },
        },
    )
    _write_json(run / "observer_chart_object_h3_report.json", {"OBJECT_BULK_POPULATION_RECEIPT": False})
    _write_json(run / "observer_modular_experience_report.json", {"observer_modular_time_receipt": True})
    (run / "observer_views.jsonl").write_text(
        json.dumps(
            {
                "view_type": "patch_observer",
                "observer_id": 1,
                "support_nodes": [1, 2],
                "visible_readout_hash": "abc",
                "observer_relative_times": [0.1],
                "transition_history_descriptor": {"persistence": 2},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = finite_oph_theorem_contract_report(run)

    assert report["observer_like_self_reading_system_receipt"] is True
    assert report["stages"]["L1_observer_record_algebra"]["passed"] is True
    assert report["stages"]["L3_kms_modular_clock_fit"]["passed"] is False
    assert report["stages"]["L6_lorentz_algebra_closure"]["passed"] is True
    assert report["stages"]["C0_finite_consensus_theorem"]["passed"] is False
    assert report["stages"]["L2_endogenous_modular_generator"]["passed"] is False
    assert report["stages"]["L4_support_visible_bw_covariance"]["passed"] is False
    assert report["finite_lorentz_theorem_contract_receipt"] is False
    assert report["paper_faithful_observer_spacetime_emergence_receipt"] is False
    assert "L2_endogenous_modular_generator" in report["blockers"]


def test_finite_theorem_contract_can_pass_when_all_hypothesis_receipts_exist(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_json(run / "theorem_core_receipts.json", {"FINITE_CONSENSUS_THEOREM_RECEIPT": True})
    _write_json(
        run / "emergence_status_report.json",
        {
            "BW_KMS_DIRECT_2PI_RECEIPT": True,
            "ENDOGENOUS_MODULAR_GENERATOR_RECEIPT": True,
            "KMS_GEOMETRIC_CLOCK_FIT_RECEIPT": True,
            "ordered_cut_pair_rigidity_receipt": True,
            "refinement_naturality_receipt": True,
            "strict_blind_observer_bulk_receipt": True,
        },
    )
    _write_json(
        run / "bw_state_derived_report.json",
        {
            "KMS_GEOMETRIC_CLOCK_FIT_RECEIPT": True,
            "inferred_modular_clock_fit": {
                "kappa_hat": 6.283185307179586,
                "kappa_95ci": [6.1, 6.4],
                "nearest_known_scale": "2pi",
                "blockers": [],
            },
        },
    )
    _write_json(run / "conformal_h3_spatial_chart_report.json", {"lorentz_algebra_receipt": True})
    _write_json(
        run / "modular_response_h3_report.json",
        {
            "H3_RESPONSE_CANDIDATE_RECEIPT": True,
            "h3_response_stage_gates": {
                "signal_gate": True,
                "geometry_gate": True,
                "aggregate_wrong_scale_gate": True,
                "material_feature_gate": True,
            },
        },
    )
    _write_json(run / "observer_chart_object_h3_report.json", {"OBJECT_BULK_POPULATION_RECEIPT": True})
    _write_json(
        run / "observer_modular_experience_report.json",
        {"OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT": True},
    )
    _write_json(run / "bulk_reconstruction_report.json", {"bulk_3d_established": True})
    (run / "observer_views.jsonl").write_text(
        json.dumps(
            {
                "view_type": "patch_observer",
                "observer_id": 1,
                "support_nodes": [1, 2],
                "visible_readout_hash": "abc",
                "observer_relative_times": [0.1],
                "transition_history_histograms": {"transition_history_key": {"1": 1.0}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = finite_oph_theorem_contract_report(run)

    assert report["finite_lorentz_theorem_contract_receipt"] is True
    assert report["paper_faithful_observer_spacetime_emergence_receipt"] is True
    assert report["paper_faithful_consensus_bulk_emergence_receipt"] is True
    assert report["blockers"] == []


def test_finite_theorem_contract_splits_observer_h3_from_populated_and_neutral(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_json(run / "theorem_core_receipts.json", {"FINITE_CONSENSUS_THEOREM_RECEIPT": True})
    _write_json(
        run / "emergence_status_report.json",
        {
            "ENDOGENOUS_MODULAR_GENERATOR_RECEIPT": True,
            "KMS_GEOMETRIC_CLOCK_FIT_RECEIPT": True,
            "ordered_cut_pair_rigidity_receipt": True,
        },
    )
    _write_json(
        run / "bw_state_derived_report.json",
        {
            "KMS_GEOMETRIC_CLOCK_FIT_RECEIPT": True,
            "inferred_modular_clock_fit": {"kappa_hat": 6.283185307179586, "blockers": []},
        },
    )
    _write_json(run / "conformal_h3_spatial_chart_report.json", {"lorentz_algebra_receipt": True})
    _write_json(
        run / "modular_response_h3_report.json",
        {
            "H3_RESPONSE_CANDIDATE_RECEIPT": True,
            "h3_response_stage_gates": {
                "signal_gate": True,
                "geometry_gate": True,
                "aggregate_wrong_scale_gate": True,
                "material_feature_gate": True,
                "material_wrong_scale_win_fraction": 0.38,
                "material_wrong_scale_gate_metric": "material_wrong_scale_advantage_energy_fraction",
                "material_wrong_scale_gate_value": 0.015,
                "material_wrong_scale_advantage_energy_fraction": 0.015,
            },
        },
    )
    _write_json(run / "observer_chart_object_h3_report.json", {"OBJECT_BULK_POPULATION_RECEIPT": False})
    _write_json(
        run / "observer_modular_experience_report.json",
        {"OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT": True},
    )
    (run / "observer_views.jsonl").write_text(
        json.dumps(
            {
                "view_type": "patch_observer",
                "observer_id": 1,
                "support_nodes": [1, 2],
                "visible_readout_hash": "abc",
                "observer_relative_times": [0.1],
                "transition_history_histograms": {"transition_history_key": {"1": 1.0}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = finite_oph_theorem_contract_report(run)

    assert report["finite_lorentz_theorem_contract_receipt"] is True
    assert report["paper_faithful_observer_spacetime_emergence_receipt"] is True
    assert report["paper_faithful_populated_h3_observer_experience_receipt"] is False
    assert report["paper_faithful_consensus_bulk_emergence_receipt"] is False
    assert report["stages"]["L7_refinement_naturality"]["passed"] is False
    assert report["stages"]["B2_observer_object_population"]["passed"] is False
    assert report["stages"]["B4_strict_neutral_bulk_audit"]["passed"] is False


def test_observer_facing_consensus_bulk_does_not_require_chart_blind_neutral(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_json(run / "theorem_core_receipts.json", {"FINITE_CONSENSUS_THEOREM_RECEIPT": True})
    _write_json(
        run / "emergence_status_report.json",
        {
            "ENDOGENOUS_MODULAR_GENERATOR_RECEIPT": True,
            "KMS_GEOMETRIC_CLOCK_FIT_RECEIPT": True,
            "ordered_cut_pair_rigidity_receipt": True,
        },
    )
    _write_json(
        run / "bw_state_derived_report.json",
        {
            "KMS_GEOMETRIC_CLOCK_FIT_RECEIPT": True,
            "inferred_modular_clock_fit": {"kappa_hat": 6.283185307179586, "blockers": []},
        },
    )
    _write_json(run / "conformal_h3_spatial_chart_report.json", {"lorentz_algebra_receipt": True})
    _write_json(
        run / "modular_response_h3_report.json",
        {
            "H3_RESPONSE_CANDIDATE_RECEIPT": True,
            "h3_response_stage_gates": {
                "signal_gate": True,
                "geometry_gate": True,
                "aggregate_wrong_scale_gate": True,
                "material_feature_gate": True,
            },
        },
    )
    _write_json(run / "observer_chart_object_h3_report.json", {"OBJECT_BULK_POPULATION_RECEIPT": True})
    _write_json(
        run / "observer_modular_experience_report.json",
        {"OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT": True},
    )
    _write_json(run / "strict_neutral_bulk_frontier_report.json", {"strict_neutral_bulk": False})
    (run / "observer_views.jsonl").write_text(
        json.dumps(
            {
                "view_type": "patch_observer",
                "observer_id": 1,
                "support_nodes": [1, 2],
                "visible_readout_hash": "abc",
                "observer_relative_times": [0.1],
                "transition_history_histograms": {"transition_history_key": {"1": 1.0}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = finite_oph_theorem_contract_report(run)

    assert report["finite_lorentz_theorem_contract_receipt"] is True
    assert report["paper_faithful_populated_h3_observer_experience_receipt"] is True
    assert report["observer_facing_consensus_3d_bulk_emergence_receipt"] is True
    assert report["paper_faithful_consensus_bulk_emergence_receipt"] is True
    assert report["chart_blind_strict_neutral_quotient_bulk_receipt"] is False
    assert report["strict_neutral_bulk_contract_receipt"] is False
    assert report["blockers"] == []
    assert report["strict_neutral_blockers"] == [
        "L7_refinement_naturality",
        "B4_strict_neutral_bulk_audit",
    ]


def test_finite_theorem_contract_accepts_chart_verifier_as_cut_pair_rigidity(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_json(run / "theorem_core_receipts.json", {"FINITE_CONSENSUS_THEOREM_RECEIPT": False})
    _write_json(
        run / "paper_3d_bulk_chart_report.json",
        {
            "PAPER_THEOREM_3D_BULK_CHART_RECEIPT": True,
            "bw_2pi_cap_flow_receipt": True,
            "lorentz_algebra_receipt": True,
        },
    )

    report = finite_oph_theorem_contract_report(run)

    assert report["stages"]["L5_ordered_cut_pair_rigidity"]["passed"] is True
    assert report["stages"]["L5_ordered_cut_pair_rigidity"]["details"][
        "paper_chart_receipt_inferred_from_cap_lorentz_verifier"
    ] is True
