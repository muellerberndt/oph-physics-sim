from __future__ import annotations

import json
from pathlib import Path

from oph_fpe.bulk.einstein_bridge import (
    EINSTEIN_SIDECAR_SCHEMA,
    RECEIPT_SPECS,
    write_einstein_bridge_manifest,
)
from oph_fpe.bulk.proof_certificate import bulk_proof_certificate, write_bulk_proof_certificate
from oph_fpe.cosmology.comparable_data import comparable_data_report
from tests.test_theorem_contract import _write_computed_consensus_replay


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_all_einstein_bridge_sidecars(run: Path) -> None:
    for spec in RECEIPT_SPECS:
        _write_json(
            run / spec.file_name,
            {
                "schema_version": EINSTEIN_SIDECAR_SCHEMA,
                "theorem_tag": spec.theorem_tag,
                spec.keys[0]: True,
            },
        )


def _canonical_refinement_receipt() -> dict:
    required = [4_096, 16_384, 65_536, 262_144]
    return {
        "mode": "prime_geometric_rank_refinement_v0",
        "sizes": [{"patch_count": value} for value in required],
        "required_patch_count_ladder": required,
        "missing_required_patch_counts": [],
        "required_ladder_complete": True,
        "multi_scale": True,
        "all_control_quotient_spatial_3d_candidates": True,
        "all_candidate_s2_leakage_pass": True,
        "all_candidate_rank3_e3": True,
        "candidate_dimension_stable": True,
        "independent_rank3_selector_all": True,
        "proper_negative_control_all": True,
        "directional_h3_strict_all": True,
        "measured_overlap_geometry_all": True,
        "strict_neutral_bulk_refinement_receipt": True,
        "proof_blockers": [],
    }


def _strict_neutral_proof_report() -> dict:
    refinement = _canonical_refinement_receipt()
    return {
        "mode": "strict_neutral_bulk_record_transition_audit",
        "dimension": {"estimators_agree_3d": True},
        "model_selection": {
            "best_model": "H3",
            "h3_beats_s2": True,
            "h3_beats_h2_h4": True,
        },
        "leakage": {"s2_leakage_pass": True},
        "controls": {
            "shuffled_records_fail": True,
            "shuffled_transition_labels_fail": True,
            "planted_2d_returns_2d": True,
            "planted_3d_returns_3d": True,
            "planted_h3_returns_h3": True,
        },
        "refinement": refinement,
        "channel_audit": {
            "duplicate_channel_gate_pass": True,
            "feature_ancestry_gate_pass": True,
        },
        "strict_neutral_theory_alignment": {"theory_required_channels_present": True},
        "quotient_geometry_contract": {
            "QUOTIENT_GEOMETRY_CONTRACT_RECEIPT": True,
            "bulk_promotion_allowed": True,
            "refinement": refinement,
            "metric": {
                "valid_pseudometric": True,
                "valid_metric": True,
                "triangle_checked_exact": True,
                "blockers": [],
                "metric_blockers": [],
            },
            "blockers": [],
        },
        "receipt": {
            "receipt": "STRICT_NEUTRAL_BULK_RECEIPT",
            "strict_neutral_bulk": True,
            "physical_claim": True,
        },
        "strict_neutral_bulk": True,
        "blockers": [],
    }


def test_bulk_proof_certificate_splits_theorem_assisted_from_strict_neutral(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "emergence_status_report.json",
        {
            "BW_KMS_DIRECT_2PI_RECEIPT": True,
            "PAPER_THEOREM_3D_BULK_CHART_RECEIPT": True,
            "CHART_LEVEL_CONFORMAL_LORENTZ_RECEIPT": True,
            "CHART_LORENTZ_H3_RECEIPT": True,
            "H3_RESPONSE_CANDIDATE_RECEIPT": True,
            "H3_RESPONSE_CONTROL_SEPARATION_RECEIPT": True,
            "OBJECT_BULK_POPULATION_RECEIPT": True,
            "observer_chart_bulk_population_receipt": True,
            "PAPER_THEOREM_ASSISTED_H3_POPULATED_CHART_RECEIPT": True,
            "SCREEN_PROXY_CMB_RECEIPT": True,
            "particle_matter_receipt": False,
            "physical_cmb_prediction": False,
        },
    )
    _write_json(run / "bulk_reconstruction_report.json", {"bulk_3d_established": False})
    _write_json(run / "cmb_lite_comparison_report.json", {"physical_cmb_prediction": False})

    report = bulk_proof_certificate(run)

    assert report["chart_level_3p1_lorentz_kinematics_established"] is False
    assert report["proof_tiers"]["L0_bw_kms_branch_replay"]["passed"] is False
    assert report["proof_tiers"]["L0_bw_kms_branch_replay"]["receipt_name"] == "BW_KMS_BRANCH_REPLAY_RECEIPT"
    assert report["proof_tiers"]["T2_bw_kms_2pi_branch"]["legacy_receipt_name"] == (
        "BW_KMS_BRANCH_INSTANTIATION_RECEIPT"
    )
    assert report["proof_tiers"]["L_full_oph_lorentz_finite_contract"]["passed"] is False
    assert report["OPH_LORENTZ_THEOREM_FINITE_CONTRACT_V1"] is False
    assert report["finite_lorentz_theorem_contract_receipt"] is False
    # Raw emergence booleans remain diagnostics.  They cannot manufacture a
    # theorem-assisted population without validated C0 and record-commit gates.
    assert report["theorem_assisted_h3_object_preview_established"] is False
    assert report["theorem_assisted_h3_nonboundary_population_established"] is False
    assert report["theorem_assisted_h3_populated_chart_established"] is False
    assert report["theorem_assisted_observer_facing_h3_population"] is False
    assert report["observer_facing_h3_object_population_receipt"] is False
    assert report["OBSERVER_FACING_H3_CHART_RECEIPT"] is False
    assert report["theorem_assisted_source_validation"][
        "ignored_declaration_diagnostics"
    ]["chart"] is True
    assert report["OBSERVER_EXPERIENCED_3P1D_HISTORY_RECEIPT"] is False
    assert report["OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT"] is False
    assert report["THEOREM_ASSISTED_H3_OBJECT_POPULATION_RECEIPT"] is False
    assert report["STRICT_NEUTRAL_BULK_RECEIPT"] is False
    assert report["strict_neutral_third_person_bulk_established"] is False
    assert report["bulk_3d_established_theorem_assisted"] is False
    assert report["bulk_3d_established_strict"] is False
    assert report["screen_cmb_proxy_available"] is True
    assert report["physical_cmb_prediction"] is False
    assert report["einstein_branch_entry_contract_receipt"] is False
    assert report["EINSTEIN_BRANCH_ENTRY_RECEIPT"] is False
    assert report["production_gravity_receipt"] is False
    assert report["physical_gravity_prediction"] is False
    assert report["proof_tiers"]["E0_einstein_branch_entry_contract"]["passed"] is False
    assert report["proof_tiers"]["G2_production_gravity"]["passed"] is False


def test_bulk_proof_certificate_rejects_string_truthiness_and_debug_aliases(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "bulk_reconstruction_report.json",
        {
            "bulk_3d_established": "false",
            "strict_neutral_bulk": "false",
            "STRICT_NEUTRAL_BULK_RECEIPT": "false",
        },
    )
    _write_json(
        run / "emergence_status_report.json",
        {
            "FINITE_SETTLE_DIAGNOSTIC_RECEIPT": "false",
            "BW_KMS_BRANCH_REPLAY_RECEIPT": "false",
            "PAPER_THEOREM_3D_BULK_CHART_RECEIPT": "false",
            "SCREEN_PROXY_CMB_RECEIPT": "false",
        },
    )

    report = bulk_proof_certificate(run)

    assert report["FINITE_SETTLE_DIAGNOSTIC_RECEIPT"] is False
    assert report["proof_tiers"]["L0_bw_kms_branch_replay"]["passed"] is False
    assert report["STRICT_NEUTRAL_BULK_RECEIPT"] is False
    assert report["proof_tiers"]["T6_chart_blind_strict_neutral_quotient_bulk"]["passed"] is False
    assert report["screen_cmb_proxy_available"] is False


def test_bulk_proof_ignores_precomputed_theorem_assisted_receipts_without_primitives(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "finite_oph_theorem_contract_report.json",
        {
            "mode": "finite_oph_theorem_contract_audit_v1",
            "OPH_LORENTZ_THEOREM_FINITE_CONTRACT_V1": True,
            "paper_faithful_consensus_bulk_emergence_receipt": True,
            "stages": {
                "T308_finite_cap_bw_certificate": {"passed": True},
                "T309_cap_normal_h3_chart": {"passed": True},
                "T310_modular_response_h3_localization": {"passed": True},
            },
        },
    )
    _write_json(run / "issue_308_bw_certificate_report.json", {"receipt": True})
    _write_json(
        run / "cap_normal_h3_chart_report.json",
        {"CAP_NORMAL_H3_CHART_RECEIPT": True, "terminal_status": "CERTIFIED"},
    )
    _write_json(
        run / "modular_response_h3_localization_report.json",
        {"MODULAR_RESPONSE_H3_LOCALIZATION_RECEIPT": True, "terminal_status": "CERTIFIED"},
    )
    _write_json(
        run / "observer_chart_object_h3_report.json",
        {"OBJECT_BULK_POPULATION_RECEIPT": True},
    )

    report = bulk_proof_certificate(run)

    assert report["proof_tiers"]["L0_bw_kms_branch_replay"]["passed"] is False
    assert report["proof_tiers"]["T3_chart_lorentz_h3"]["passed"] is False
    assert report["proof_tiers"]["T4_h3_response_controls"]["passed"] is False
    assert report["proof_tiers"]["T5b_nonboundary_h3_object_population"]["passed"] is False
    assert report["finite_lorentz_theorem_contract_receipt"] is False
    assert report["persisted_finite_theorem_contract_diagnostic"][
        "finite_lorentz_contract_claim"
    ] is True


def test_bulk_proof_certificate_rejects_handcrafted_strict_neutral_fixture(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    strict = _strict_neutral_proof_report()
    _write_json(run / "strict_neutral_bulk_report.json", strict)
    _write_json(run / "prime_geometric_rank_refinement_report.json", strict["refinement"])

    report = bulk_proof_certificate(run)

    assert report["strict_neutral_derived_report_diagnostic"][
        "persisted_typed_bulk_candidate"
    ] is True
    assert report["STRICT_NEUTRAL_QUOTIENT_METRIC_RECEIPT"] is False
    assert report["STRICT_NEUTRAL_BULK_RECEIPT"] is False
    assert report["proof_tiers"]["T6_chart_blind_strict_neutral_quotient_bulk"]["passed"] is False
    assert "strict_neutral_source_manifest_missing" in report[
        "strict_neutral_source_validation"
    ]["blockers"]

    strict["channel_audit"]["feature_ancestry_gate_pass"] = "true"
    _write_json(run / "strict_neutral_bulk_report.json", strict)

    malformed = bulk_proof_certificate(run)
    assert malformed["STRICT_NEUTRAL_BULK_RECEIPT"] is False
    assert malformed["proof_tiers"]["T6_chart_blind_strict_neutral_quotient_bulk"]["passed"] is False


def test_bulk_proof_preview_does_not_promote_to_nonboundary_population(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "emergence_status_report.json",
        {
            "BW_KMS_DIRECT_2PI_RECEIPT": True,
            "PAPER_THEOREM_3D_BULK_CHART_RECEIPT": True,
            "H3_RESPONSE_CANDIDATE_RECEIPT": True,
            "THEOREM_ASSISTED_H3_OBJECT_PREVIEW_RECEIPT": True,
            "PAPER_THEOREM_ASSISTED_H3_POPULATED_CHART_RECEIPT": True,
        },
    )

    report = bulk_proof_certificate(run)

    assert report["theorem_assisted_h3_object_preview_established"] is False
    assert report["theorem_assisted_h3_nonboundary_population_established"] is False
    assert report["theorem_assisted_h3_populated_chart_established"] is False
    assert report["bulk_3d_established_theorem_assisted"] is False
    assert report["STRICT_NEUTRAL_BULK_RECEIPT"] is False


def test_missing_observer_history_blocks_experienced_3p1d_history(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "emergence_status_report.json",
        {
            "BW_KMS_BRANCH_REPLAY_RECEIPT": True,
            "PAPER_THEOREM_3D_BULK_CHART_RECEIPT": True,
            "CHART_LORENTZ_H3_RECEIPT": True,
            "H3_RESPONSE_CANDIDATE_RECEIPT": True,
        },
    )

    report = bulk_proof_certificate(run)

    assert report["OBSERVER_FACING_H3_CHART_RECEIPT"] is False
    assert report["OBSERVER_EXPERIENCED_3P1D_HISTORY_RECEIPT"] is False
    assert report["OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT"] is False
    assert report["observer_modular_experience_summary"]["written"] is False
    assert "observer_modular_experience_written" in report["observer_modular_experience_summary"]["blockers"]


def test_strict_neutral_object_candidate_does_not_promote_full_neutral_bulk(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "strict_neutral_object_bulk_report.json",
        {
            "STRICT_NEUTRAL_OBJECT_BULK_RECEIPT": True,
            "strict_neutral_object_bulk": True,
            "object_count": 24,
        },
    )

    report = bulk_proof_certificate(run)

    assert report["STRICT_NEUTRAL_OBJECT_BULK_CANDIDATE_RECEIPT"] is False
    assert report["STRICT_NEUTRAL_OBJECT_BULK_RECEIPT"] is False
    assert report["strict_neutral_object_bulk_summary"][
        "persisted_declaration_diagnostic"
    ] is True
    assert "strict_neutral_object_source_manifest_missing" in report[
        "strict_neutral_object_source_validation"
    ]["blockers"]
    assert report["STRICT_NEUTRAL_BULK_RECEIPT"] is False
    assert report["STRICT_NEUTRAL_THIRD_PERSON_BULK_RECEIPT"] is False
    assert report["strict_neutral_third_person_bulk_established"] is False


def test_physical_cmb_prediction_requires_staged_contracts_not_output_booleans(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(run / "emergence_status_report.json", {"physical_cmb_prediction": True})
    _write_json(run / "cmb_lite_comparison_report.json", {"physical_cmb_prediction": True})
    _write_json(run / "cl_comparison_report.json", {"physical_cmb_prediction": True})
    _write_json(run / "physical_cmb_frontier_report.json", {"physical_cmb_prediction_receipt": True})
    _write_json(run / "physical_cmb_output_comparison_report.json", {"PHYSICAL_CMB_PREDICTION_RECEIPT": True})

    report = bulk_proof_certificate(run)

    assert report["physical_cmb_staged_contract"]["CMB2_OUTPUT_ARTIFACT_PRESENT"] is True
    assert report["CMB1_SOURCE_INPUT_CONTRACT"] is False
    assert report["CMB1_FROZEN_TRANSFER_LIKELIHOOD_CLOSURE"] is False
    assert report["CMB2_PHYSICAL_CMB_PREDICTION_RECEIPT"] is False
    assert report["physical_cmb_prediction"] is False


def test_bulk_proof_certificate_rejects_handwritten_finite_contract_promotions(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "finite_oph_theorem_contract_report.json",
        {
            "finite_lorentz_theorem_contract_receipt": True,
            "paper_faithful_observer_spacetime_emergence_receipt": True,
            "paper_faithful_populated_h3_observer_experience_receipt": True,
            "paper_faithful_consensus_bulk_emergence_receipt": True,
            "chart_blind_strict_neutral_quotient_bulk_receipt": False,
            "strict_neutral_blockers": ["B4_strict_neutral_bulk_audit"],
        },
    )

    report = bulk_proof_certificate(run)

    assert report["observer_facing_consensus_3d_bulk_emergence_receipt"] is False
    assert report["paper_faithful_consensus_bulk_emergence_receipt"] is False
    assert report["bulk_3d_established_observer_facing_consensus"] is False
    assert report["chart_blind_strict_neutral_quotient_bulk_receipt"] is False
    assert report["bulk_3d_established_chart_blind_strict_neutral"] is False
    assert report["persisted_finite_theorem_contract_diagnostic"][
        "consensus_bulk_claim"
    ] is True
    assert report["finite_theorem_contract_summary"]["recomputed_in_memory"] is True
    assert report["finite_theorem_contract_summary"]["einstein_branch_entry_contract_receipt"] is False


def test_bulk_proof_certificate_keeps_gravity_closed_after_branch_entry_without_source_bridge(
    tmp_path: Path,
):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "finite_oph_theorem_contract_report.json",
        {
            "OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1": True,
            "EINSTEIN_BRANCH_ENTRY_RECEIPT": True,
            "einstein_branch_entry_contract_receipt": True,
            "einstein_branch_entry_blockers": [],
        },
    )

    report = bulk_proof_certificate(run)

    assert report["einstein_branch_entry_contract_receipt"] is False
    assert report["proof_tiers"]["E0_einstein_branch_entry_contract"]["passed"] is False
    assert report["production_gravity_receipt"] is False
    assert report["physical_gravity_prediction"] is False
    assert report["proof_tiers"]["G2_production_gravity"]["passed"] is False
    assert "null_stress" in report["proof_tiers"]["G2_production_gravity"]["blockers"]


def test_bulk_proof_certificate_uses_einstein_bridge_manifest(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_all_einstein_bridge_sidecars(run)
    write_einstein_bridge_manifest(run)

    report = bulk_proof_certificate(run)

    assert report["einstein_branch_entry_contract_receipt"] is True
    assert report["proof_tiers"]["E0_einstein_branch_entry_contract"]["passed"] is True
    assert report["einstein_branch_entry_summary"]["manifest_written"] is True
    assert report["einstein_branch_entry_summary"]["theorem_e0_dependency_discharge_receipt"] is True
    assert report["einstein_branch_entry_summary"]["einstein_bridge_run_receipts_receipt"] is True
    assert report["einstein_branch_entry_summary"]["provenance_tags"]["AllTimelikeCoverage"] == (
        "LEMMA_E0_7"
    )
    assert report["production_gravity_receipt"] is False
    assert "production_source_stress_bridge_missing" in report["proof_tiers"]["G2_production_gravity"][
        "blockers"
    ]


def test_bulk_proof_certificate_ignores_forged_einstein_manifest(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "einstein_bridge_manifest.json",
        {
            "OPH_EINSTEIN_BRANCH_ENTRY_CONTRACT_V1": True,
            "EINSTEIN_BRANCH_ENTRY_RECEIPT": True,
            "einstein_branch_entry_receipt": True,
        },
    )

    report = bulk_proof_certificate(run)

    assert report["einstein_branch_entry_contract_receipt"] is False
    assert report["einstein_branch_entry_summary"][
        "persisted_manifest_branch_entry_ignored"
    ] is True


def test_bulk_proof_certificate_splits_c0a_settle_from_c0b_consensus(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "theorem_core_receipts.json",
        {
            "FINITE_SETTLE_DIAGNOSTIC_RECEIPT": True,
            "finite_settle_diagnostic_receipt": True,
            "FINITE_CONSENSUS_THEOREM_RECEIPT": False,
            "finite_consensus_theorem_receipt": False,
        },
    )

    report = bulk_proof_certificate(run)

    assert report["finite_settle_diagnostic_receipt"] is True
    assert report["finite_consensus_theorem_receipt"] is False
    assert report["proof_tiers"]["C0a_finite_settle_diagnostic"]["passed"] is True
    assert report["proof_tiers"]["C0b_finite_consensus_theorem"]["passed"] is False
    assert report["proof_tiers"]["T0_finite_repair_core"]["canonical_tier"] == "C0a"


def test_bulk_proof_c0b_rejects_handwritten_theorem_stage(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "emergence_status_report.json",
        {"FINITE_CONSENSUS_THEOREM_RECEIPT": True},
    )
    _write_json(
        run / "theorem_core_receipts.json",
        {
            "FINITE_CONSENSUS_THEOREM_RECEIPT": True,
            "finite_consensus_theorem_receipt": True,
        },
    )

    declarations_only = bulk_proof_certificate(run)
    assert declarations_only["finite_consensus_declaration_diagnostic"] is True
    assert declarations_only["FINITE_CONSENSUS_THEOREM_RECEIPT"] is False
    assert declarations_only["proof_tiers"]["C0b_finite_consensus_theorem"]["passed"] is False

    contract = {
        "mode": "finite_oph_theorem_contract_audit_v1",
        "stages": {"C0_finite_consensus_theorem": {"passed": "true"}},
    }
    _write_json(run / "finite_oph_theorem_contract_report.json", contract)
    string_stage = bulk_proof_certificate(run)
    assert string_stage["FINITE_CONSENSUS_THEOREM_RECEIPT"] is False

    contract["stages"]["C0_finite_consensus_theorem"]["passed"] = True
    _write_json(run / "finite_oph_theorem_contract_report.json", contract)
    handwritten = bulk_proof_certificate(run)
    assert handwritten["FINITE_CONSENSUS_THEOREM_RECEIPT"] is False
    assert handwritten["proof_tiers"]["C0b_finite_consensus_theorem"]["passed"] is False
    assert "computed_v3_gauge_quotient_consensus_certificate_missing" in handwritten[
        "finite_consensus_primitive_validation"
    ]["blockers"]


def test_bulk_proof_c0b_accepts_primitive_bound_independent_replay(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_computed_consensus_replay(run)

    report = bulk_proof_certificate(run)

    assert report["finite_consensus_primitive_validation"]["passed"] is True
    assert report["FINITE_CONSENSUS_THEOREM_RECEIPT"] is True
    assert report["proof_tiers"]["C0b_finite_consensus_theorem"]["passed"] is True


def test_bulk_proof_certificate_reads_observer_modular_experience(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "emergence_status_report.json",
        {
            "BW_KMS_BRANCH_REPLAY_RECEIPT": True,
            "PAPER_THEOREM_3D_BULK_CHART_RECEIPT": True,
            "H3_RESPONSE_CANDIDATE_RECEIPT": True,
        },
    )
    _write_json(
        run / "observer_modular_experience_report.json",
        {
            "observer_modular_time_receipt": True,
            "semantic_history_receipt": True,
            "observer_clock_naturality_receipt": True,
            "observer_registry_descent_receipt": True,
            "state_preserving_observer_algebra_receipt": True,
            "support_cap_chart_naturality_receipt": True,
            "OBSERVER_FACING_3P1D_H3_EXPERIENCE_RECEIPT": True,
            "observer_facing_3p1d_h3_experience_receipt": True,
            "observer_facing_populated_h3_experience_receipt": False,
            "observer_h3_object_population_receipt": False,
            "observer_count": 16,
            "observer_relative_time_count": 2,
            "blockers": [],
            "populated_h3_experience_blockers": ["observer_h3_object_population_receipt"],
            "component_gates": {
                "observer_modular_time_receipt": True,
                "bw_kms_branch_replay_receipt": True,
                "conformal_h3_chart_receipt": True,
                "h3_modular_response_receipt": True,
                "semantic_history_receipt": True,
                "observer_clock_naturality_receipt": True,
                "observer_registry_descent_receipt": True,
                "state_preserving_observer_algebra_receipt": True,
                "support_cap_chart_naturality_receipt": True,
            },
        },
    )

    report = bulk_proof_certificate(run)

    assert report["observer_modular_time_receipt"] is True
    assert report["observer_facing_h3_chart_receipt"] is False
    assert report["observer_experienced_3p1d_history_receipt"] is False
    assert report["observer_facing_3p1d_h3_experience_receipt"] is False
    assert report["observer_facing_populated_h3_experience_receipt"] is False
    assert report["observer_modular_experience_summary"]["observer_count"] == 16
    assert report["observer_modular_experience_summary"]["source_report_blockers"] == []
    assert report["observer_modular_experience_summary"]["source_report_populated_h3_blockers"] == [
        "observer_h3_object_population_receipt"
    ]
    assert "conformal_h3_chart_receipt" in report[
        "observer_modular_experience_summary"
    ]["blockers"]
    assert "observer_h3_object_population_receipt" in report["observer_modular_experience_summary"][
        "populated_h3_experience_blockers"
    ]


def test_bulk_proof_certificate_writes_and_comparable_data_collects_tiers(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "emergence_status_report.json",
        {
            "BW_KMS_DIRECT_2PI_RECEIPT": True,
            "PAPER_THEOREM_3D_BULK_CHART_RECEIPT": True,
            "H3_RESPONSE_CANDIDATE_RECEIPT": True,
            "OBJECT_BULK_POPULATION_RECEIPT": True,
            "SCREEN_PROXY_CMB_RECEIPT": True,
        },
    )
    write_bulk_proof_certificate(run)

    snapshot = comparable_data_report([tmp_path])
    lane = snapshot["measurement_lanes"]["support_visible_lorentz_branch"]

    assert lane["bulk_proof_certificate_count"] == 1
    assert lane["bulk_proof_chart_level_3p1_count"] == 0
    assert lane["bulk_proof_theorem_assisted_h3_object_preview_count"] == 0
    assert lane["bulk_proof_theorem_assisted_h3_nonboundary_population_count"] == 0
    assert lane["bulk_proof_theorem_assisted_h3_populated_chart_count"] == 0
    assert lane["bulk_proof_strict_neutral_3d_bulk_count"] == 0
    assert lane["bulk_proof_screen_cmb_proxy_count"] == 1
    assert lane["bulk_proof_physical_cmb_prediction_count"] == 0


def test_bulk_proof_certificate_accepts_output_directory(tmp_path: Path):
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    _write_json(
        run / "emergence_status_report.json",
        {
            "BW_KMS_DIRECT_2PI_RECEIPT": True,
            "PAPER_THEOREM_3D_BULK_CHART_RECEIPT": True,
        },
    )

    write_bulk_proof_certificate(run, out)

    assert (out / "bulk_proof_certificate_report.json").exists()
    assert (out / "bulk_proof_certificate_report.md").exists()


def test_bulk_proof_certificate_reads_scale_compressed_branch_without_overclaiming(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "scale_compressed_repair_report.json",
        {
            "logical_repair_rounds": 24,
            "scale_compressed_operator_receipt": True,
            "repair_round_trace_receipt": True,
            "cmb_parameter_readouts": {
                "eta_R": 0.0339785,
                "n_s": 0.9660215,
                "q_IR": 0.25,
                "ell_IR": 32.0,
            },
            "h3_preview": {
                "object_count": 48,
                "cap_count": 120,
                "cap_profile_receipt": True,
                "populated_h3_preview_receipt": True,
                "strict_neutral_third_person_bulk_established": False,
            },
            "particle_preview": {
                "particle_worldline_count": 6,
                "particle_preview_receipt": True,
                "production_particle_matter_receipt": False,
            },
            "physical_cmb_prediction": False,
            "strict_neutral_bulk": False,
        },
    )
    _write_json(
        run / "scale_compressed_cmb_camb_report.json",
        {
            "measurement_comparable_cmb_curve": True,
            "screen_camb_transfer_receipt": True,
            "physical_cmb_prediction": False,
            "comparison": {
                "scale_compressed_ir_kernel": {
                    "shape_correlation": 0.999,
                    "normalized_rmse": 0.017,
                    "best_fit_column_chi2_per_bin": 0.96,
                }
            },
        },
    )
    _write_json(
        run / "scale_compressed_particle_report.json",
        {
            "worldline_count": 6,
            "particle_preview_receipt": True,
            "production_particle_matter_receipt": False,
        },
    )

    report = bulk_proof_certificate(run)

    assert report["scale_compressed_operator_receipt"] is True
    assert report["scale_compressed_repair_round_trace_receipt"] is True
    assert report["scale_compressed_h3_preview_established"] is True
    assert report["scale_compressed_measurement_comparable_cmb_curve"] is True
    assert report["scale_compressed_particle_preview_established"] is True
    assert report["screen_cmb_proxy_available"] is True
    assert report["physical_cmb_prediction"] is False
    assert report["production_particle_matter_receipt"] is False
    assert report["strict_neutral_third_person_bulk_established"] is False
    assert report["bulk_3d_established_theorem_assisted"] is False
    assert report["proof_tiers"]["T5c_scale_compressed_h3_preview"]["passed"] is True
    assert report["proof_tiers"]["T8b_scale_compressed_camb_transfer"]["passed"] is True
    assert report["scale_compressed_summary"]["h3_object_count"] == 48
    assert report["scale_compressed_summary"]["camb_ir_chi2_per_bin"] == 0.96


def test_bulk_proof_certificate_reads_paper_chart_files_without_strict_bulk(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_json(
        run / "paper_3d_bulk_chart_report.json",
        {
            "PAPER_THEOREM_3D_BULK_CHART_RECEIPT": True,
            "paper_theorem_3d_bulk_chart_receipt": True,
            "chart_level_conformal_lorentz_receipt": True,
            "bw_2pi_cap_flow_receipt": True,
            "lorentz_group": "SO+(3,1)",
            "spatial_homogeneous_space": "H3 = SO+(3,1)/SO(3)",
            "h3_spatial_dimension_from_boost_orbit": 3,
            "h3_chart_spatial_dimension": 3,
            "finite_point_cloud_dimension_estimator_used": False,
        },
    )
    _write_json(run / "bulk_reconstruction_report.json", {"bulk_3d_established": False})

    report = bulk_proof_certificate(run)

    assert report["chart_level_3p1_lorentz_kinematics_established"] is False
    assert report["strict_neutral_third_person_bulk_established"] is False
    assert report["bulk_3d_established_theorem_assisted"] is False
    assert report["paper_chart_summary"]["h3_spatial_dimension_from_boost_orbit"] == 3
    assert report["paper_chart_summary"]["finite_point_cloud_dimension_estimator_used"] is False
